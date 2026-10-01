#!/usr/bin/env python3
"""Simulator-free OC3 diagnostics, including abrupt subprocess exits."""

import multiprocessing as mp
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import _oc3_runtime as runtime


def exit_without_reply(connection, log_path):
    with open(log_path, "w", encoding="utf-8") as log:
        log.write("[OC3] Foraging: launching Isaac AppLauncher\nNative startup failed\n")
    os._exit(23)


class WorkerDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="oc3_runtime_test_")
        self.addCleanup(self.temp.cleanup)
        self.worker = runtime.Worker.__new__(runtime.Worker)
        self.worker.name = "Foraging"
        self.worker.timeout = 30
        self.worker.pending_command = "initialization"
        self.worker.log_path = str(Path(self.temp.name) / "worker.log")
        self.worker.process = mock.Mock(pid=123, exitcode=23)
        self.worker.connection = mock.Mock()

    def test_abrupt_process_exit_includes_code_phase_and_log(self):
        context = mp.get_context("spawn")
        self.worker.connection, child = context.Pipe()
        self.worker.process = context.Process(
            target=exit_without_reply, args=(child, self.worker.log_path),
        )
        self.worker.process.start()
        child.close()
        self.addCleanup(self.worker.close)
        with self.assertRaises(RuntimeError) as raised:
            self.worker.receive()
        message = str(raised.exception)
        self.assertIn("Foraging", message)
        self.assertIn("during initialization", message)
        self.assertIn("exitcode=23", message)
        self.assertIn("Native startup failed", message)
        self.assertIn(self.worker.log_path, message)

    def test_timeout_and_missing_log_keep_original_context(self):
        self.worker.connection.poll.return_value = False
        self.worker.process.exitcode = None
        with self.assertRaises(TimeoutError) as raised:
            self.worker.receive()
        self.assertIn("timed out", str(raised.exception))
        self.assertIn("log unavailable", str(raised.exception))
        self.worker.process.join.assert_not_called()

    def test_send_failure_includes_command_and_close_still_cleans_up(self):
        self.worker.connection.send_bytes.side_effect = BrokenPipeError("closed")
        with self.assertRaisesRegex(RuntimeError, "during collect"):
            self.worker.send("collect", steps=3)
        self.worker.process.is_alive.return_value = False
        with mock.patch.object(self.worker, "send", side_effect=RuntimeError("closed")):
            self.worker.process.is_alive.side_effect = [True, False, False]
            self.worker.close()
        self.worker.connection.close.assert_called_once()

    def test_log_tail_limits_output_and_tolerates_invalid_utf8(self):
        path = Path(self.worker.log_path)
        path.write_bytes(b"old\n" * 10000 + b"last-1\nlast-2\xff\n")
        tail = runtime.log_tail(path, max_bytes=128, max_lines=2)
        self.assertEqual(tail.splitlines(), ["last-1", "last-2\ufffd"])
        path.write_bytes(b"")
        self.assertIn("empty", runtime.log_tail(path))

    def test_python_error_and_successful_reply(self):
        self.worker.connection.poll.return_value = True
        with mock.patch.object(runtime, "receive", return_value={"error": "ValueError: failed"}):
            with self.assertRaisesRegex(RuntimeError, "ValueError: failed"):
                self.worker.receive()
        with mock.patch.object(runtime, "receive", return_value={"result": {"step": 20}}):
            self.assertEqual(self.worker.receive(), {"step": 20})
        with mock.patch.object(runtime, "receive", return_value={"ready": True}):
            self.assertEqual(self.worker.receive(), {"ready": True})

    @unittest.skipUnless(os.name == "posix", "POSIX signal exit codes")
    def test_signal_exit_is_named_without_assuming_oom(self):
        self.worker.process.exitcode = -9
        text = self.worker.failure_details("connection closed")
        self.assertIn("SIGKILL", text)
        self.assertNotIn("out of memory", text.lower())


if __name__ == "__main__":
    unittest.main(verbosity=2)
