"""OC3 process isolation and simulator-free agent imports."""

from __future__ import annotations

import importlib
import io
import os
from pathlib import Path
import signal
import sys
import traceback
import types


ROOT = Path(__file__).resolve().parents[1]


def agents_module(name):
    # Importing the extension's root registers Isaac tasks. The coordinator
    # intentionally has no SimulationApp, so load only the pure torch package.
    package_name = "swarmacb_oc3_runtime"
    if package_name not in sys.modules:
        package = types.ModuleType(package_name)
        package.__path__ = [str(ROOT / "source/SwarmACB_isaac/SwarmACB_isaac/tasks/direct/agents")]
        sys.modules[package_name] = package
    return importlib.import_module(f"{package_name}.{name}")


def send(connection, payload):
    import torch
    stream = io.BytesIO()
    torch.save(payload, stream)
    connection.send_bytes(stream.getvalue())


def receive(connection):
    import torch
    return torch.load(io.BytesIO(connection.recv_bytes()), map_location="cpu", weights_only=False)


def log_tail(path, max_bytes=32768, max_lines=100):
    """Bound diagnostic output without reading an entire training log."""
    try:
        with open(path, "rb") as stream:
            stream.seek(0, os.SEEK_END)
            stream.seek(max(0, stream.tell() - max_bytes))
            text = stream.read(max_bytes).decode("utf-8", errors="replace")
        return "\n".join(text.splitlines()[-max_lines:]) or "(worker log is empty)"
    except OSError as error:
        return f"(worker log unavailable: {error})"


def worker_main(connection, spec, library_state, frozen, device, threads, log_path):
    app = env = trainer = None
    try:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        log = open(log_path, "a", encoding="utf-8", buffering=1)
        os.dup2(log.fileno(), 1)
        os.dup2(log.fileno(), 2)
        sys.stdout = sys.stderr = log
        import faulthandler
        faulthandler.enable(file=log, all_threads=True)
        log.write(f"[OC3] {spec['name']}: worker pid={os.getpid()} starting on {device}\n")
        import argparse
        import random
        import numpy as np
        import torch
        torch.set_num_threads(threads)
        from isaaclab.app import AppLauncher
        from _isaac_launch import apply_windows_kit_defaults
        args = argparse.Namespace(headless=True, device=device, kit_args="")
        apply_windows_kit_defaults(args, "OC3")
        log.write(f"[OC3] {spec['name']}: launching Isaac AppLauncher\n")
        launcher = AppLauncher(args)
        app = launcher.app
        log.write(f"[OC3] {spec['name']}: Isaac app ready; registering tasks\n")
        import gymnasium as gym
        sys.path.insert(0, str(ROOT / "source/SwarmACB_isaac"))
        import SwarmACB_isaac.tasks  # noqa: F401

        training = agents_module("learned_option_critic_trainer")
        multi = agents_module("multi_mission_trainer")
        cfg = training.LearnedOptionCriticConfig(**spec["trainer"])
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg.seed)
        registration = gym.spec(spec["task"])
        module, cls = registration.kwargs["env_cfg_entry_point"].rsplit(":", 1)
        env_cfg = getattr(importlib.import_module(module), cls)()
        env_cfg.update_variant("cyclamen")
        env_cfg.use_continuous_actions(full_observations=True)
        env_cfg.seed = cfg.seed
        env_cfg.sim.device = device
        for key, value in spec["environment"].items():
            if key == "num_envs":
                env_cfg.scene.num_envs = value
            elif hasattr(env_cfg, key):
                setattr(env_cfg, key, value)
            else:
                raise ValueError(f"Unknown environment setting {key}")
        log.write(f"[OC3] {spec['name']}: creating environment {spec['task']}\n")
        env = gym.make(spec["task"], cfg=env_cfg)
        log.write(f"[OC3] {spec['name']}: environment ready; creating trainer\n")
        trainer = training.LearnedOptionCriticTrainer(env, cfg)
        if trainer.num_agents != 20:
            raise ValueError("OC3 benchmark collection expects 20 robots per environment")
        steps = env.unwrapped.max_episode_length
        if steps % cfg.decision_period:
            raise ValueError("Episode length must be divisible by the action decision period")
        learner = multi.MissionLearner(trainer, library_state, frozen)
        log.write(f"[OC3] {spec['name']} ready on {device}\n")
        send(connection, {"ready": True})
        while True:
            request = receive(connection)
            command = request.pop("command")
            if command in ("collect", "finish", "restore", "export", "close"):
                log.write(f"[OC3] {spec['name']}: {command}\n")
            if command == "close":
                break
            if command == "collect":
                result = learner.collect(**request)
            elif command == "epoch":
                result = learner.start_epoch()
            elif command == "gradients":
                result = learner.gradients(**request)
            elif command == "commit":
                result = learner.commit(**request)
            elif command == "finish":
                result = learner.finish()
            elif command == "snapshot":
                result = learner.snapshot()
            elif command == "restore":
                result = learner.restore(request["state"])
            elif command == "export":
                result = learner.export_actor(request["path"], spec["name"])
            else:
                raise ValueError(f"Unknown worker command: {command}")
            send(connection, {"result": result})
    except BaseException:
        error = traceback.format_exc()
        print(error, flush=True)
        try:
            send(connection, {"error": error})
        except (OSError, EOFError):
            pass
    finally:
        if trainer is not None:
            trainer.writer.close()
        if env is not None:
            env.close()
        if app is not None:
            app.close()
        connection.close()


class Worker:
    def __init__(self, context, spec, library_state, frozen, device, threads, timeout):
        self.name = spec["name"]
        self.timeout = timeout
        self.pending_command = "initialization"
        self.log_path = str(Path(spec["trainer"]["log_dir"]) / "worker.log")
        self.connection, child = context.Pipe()
        self.process = context.Process(
            target=worker_main,
            args=(child, spec, library_state, frozen, device, threads, self.log_path),
            name=f"OC3-{self.name}",
        )
        self.process.start()
        child.close()

    def send(self, command, **kwargs):
        self.pending_command = command
        try:
            send(self.connection, {"command": command, **kwargs})
        except (EOFError, OSError) as error:
            raise RuntimeError(self.failure_details("connection failed", wait=True)) from error

    def failure_details(self, reason, wait=False):
        # EOF can arrive just before the OS makes the exit status available.
        if wait:
            self.process.join(timeout=1)
        code = self.process.exitcode
        status = "not available (process may still be exiting)" if code is None else str(code)
        if os.name == "posix" and code is not None and code < 0:
            try:
                status += f" ({signal.Signals(-code).name})"
            except ValueError:
                pass
        return (
            f"{self.name} worker {reason} during {self.pending_command}; "
            f"pid={self.process.pid}, exitcode={status}\n"
            f"Worker log: {self.log_path}\n"
            f"--- last worker log lines ---\n{log_tail(self.log_path)}\n"
            "--- end worker log ---"
        )

    def receive(self):
        try:
            if not self.connection.poll(self.timeout):
                raise TimeoutError(self.failure_details("timed out"))
            reply = receive(self.connection)
        except TimeoutError:
            raise
        except (EOFError, OSError) as error:
            raise RuntimeError(self.failure_details("connection closed", wait=True)) from error
        if "error" in reply:
            raise RuntimeError(f"{self.failure_details('failed')}\n{reply['error']}")
        return reply.get("result", reply)

    def close(self):
        try:
            if self.process.is_alive():
                self.send("close")
        except (OSError, EOFError, RuntimeError):
            pass
        self.process.join(15)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(10)
        if self.process.is_alive():
            self.process.kill()
            self.process.join()
        self.connection.close()
