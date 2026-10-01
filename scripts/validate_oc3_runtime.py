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


def report_thread_environment(connection):
    connection.send({key: os.environ.get(key) for key in runtime.THREAD_ENV_VARS})
    connection.close()


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

    def test_thread_budget_reaches_spawned_process_and_restores_parent(self):
        context = mp.get_context("spawn")
        parent, child = context.Pipe()
        process = context.Process(target=report_thread_environment, args=(child,))
        previous = {key: os.environ.get(key) for key in runtime.THREAD_ENV_VARS}
        try:
            with runtime.worker_thread_environment(2):
                process.start()
            child.close()
            self.assertTrue(parent.poll(30))
            self.assertEqual(parent.recv(), {key: "2" for key in runtime.THREAD_ENV_VARS})
            self.assertEqual({key: os.environ.get(key) for key in runtime.THREAD_ENV_VARS}, previous)
        finally:
            process.join(10)
            if process.is_alive():
                process.kill()
                process.join()
            parent.close()
            child.close()

    def test_worker_start_receives_caps_and_restores_environment_on_failure(self):
        context = mock.Mock()
        parent, child = mock.Mock(), mock.Mock()
        context.Pipe.return_value = (parent, child)
        previous = dict(os.environ)

        def fail_start():
            self.assertTrue(all(os.environ[key] == "2" for key in runtime.THREAD_ENV_VARS))
            raise OSError("unable to spawn")

        context.Process.return_value.start.side_effect = fail_start
        spec = {"name": "Foraging", "trainer": {"log_dir": self.temp.name}}
        with self.assertRaisesRegex(OSError, "unable to spawn"):
            runtime.Worker(context, spec, {}, False, "cpu", 2, 30)
        self.assertEqual(dict(os.environ), previous)
        parent.close.assert_called_once()
        child.close.assert_called_once()

    def test_kit_thread_caps_and_budget_validation(self):
        self.assertEqual(runtime.worker_kit_args(2).split(),
                         [f"--{key}=2" for key in runtime.KIT_THREAD_SETTINGS])
        for invalid in (0, -1, True, 1.5):
            with self.subTest(value=invalid), self.assertRaises(ValueError):
                with runtime.worker_thread_environment(invalid):
                    pass

    def test_startup_panic_survives_crash_metadata_tail(self):
        text = "failed to spawn thread: Resource temporarily unavailable\n"
        text += "crash metadata\n" * 120
        Path(self.worker.log_path).write_text(text, encoding="utf-8")
        self.assertIn("failed to spawn thread", runtime.log_tail(self.worker.log_path))

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
