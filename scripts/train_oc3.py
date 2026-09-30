#!/usr/bin/env python3
"""Train shared reactive options across missions, or adapt a frozen library.

The coordinator is CPU-only; spawned headless Isaac workers own task simulators
and compute gradients. Shared Adam state exists exactly once, here.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import multiprocessing as mp
from pathlib import Path
import time

from _oc3_config import resolve
from _oc3_runtime import Worker, agents_module


SCHEMA = 1


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--config", default="configs/OC3_cyclamen.yaml")
    result.add_argument("--missions", nargs="+", help="Subset of named mission configs")
    result.add_argument("--exclude", nargs="+", help="Held-out mission(s)")
    result.add_argument("--steps-per-mission", type=int)
    result.add_argument("--num_envs", type=int, help="Environments per mission")
    result.add_argument("--rollout-steps", type=int)
    result.add_argument("--seed", type=int, default=0)
    result.add_argument("--device", default="cuda:0")
    result.add_argument("--headless", action="store_true", help="Accepted for HPC compatibility; always headless")
    result.add_argument("--log_dir")
    result.add_argument("--checkpoint_dir")
    result.add_argument("--checkpoint", help="Resume a completed-round OC3 bundle")
    result.add_argument("--transfer", help="Freeze an exported library and learn one unseen mission")
    result.add_argument("--dry-run", action="store_true", help="Validate and print the campaign without Isaac")
    return result


def broadcast(workers, command, **kwargs):
    for worker in workers:
        worker.send(command, **kwargs)
    return [worker.receive() for worker in workers]


def atomic_save(value, path):
    import torch
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def validate_transfer(data, targets, contract):
    if data.get("schema") != SCHEMA or data.get("kind") != "oc3_library":
        raise ValueError("--transfer requires an exported OC3 option library")
    if data.get("observation_contract") != contract:
        raise ValueError("Frozen library observation/action contract mismatch")
    if set(targets) & set(data["training_missions"]):
        raise ValueError("Held-out transfer target was already used to train this library")


def main(argv=None):
    import torch
    from torch.utils.tensorboard import SummaryWriter
    args = parser().parse_args(argv)
    torch.set_num_threads(1)
    config, specs, manifest, fingerprint, log_dir, checkpoint_dir = resolve(args.config, args)
    networks = agents_module("multi_mission_networks")
    training = agents_module("learned_option_critic_trainer")
    cpu_tree = agents_module("multi_mission_trainer").cpu_tree
    cfg = training.LearnedOptionCriticConfig(**specs[0]["trainer"])
    torch.manual_seed(args.seed)
    library = networks.SharedOptionLibrary(networks.make_actor(cfg))
    frozen = bool(args.transfer)
    source_missions = [spec["name"] for spec in specs]
    source_cfg = asdict(cfg)
    resume = None
    if args.transfer:
        transfer = torch.load(args.transfer, map_location="cpu", weights_only=False)
        validate_transfer(transfer, source_missions, networks.OBSERVATION_CONTRACT)
        library.load_state_dict(transfer["library"], strict=True)
        source_missions = transfer["training_missions"]
        source_cfg = transfer["source_config"]
    if args.checkpoint:
        resume = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        if resume.get("schema") != SCHEMA or resume.get("kind") != "oc3_training":
            raise ValueError("Not an OC3 training bundle")
        if resume["fingerprint"] != fingerprint:
            raise ValueError("Resume configuration differs from the saved campaign")
        library.load_state_dict(resume["library"], strict=True)
        frozen = resume["frozen"]
        source_missions = resume["source_missions"]
        source_cfg = resume["source_config"]
    if frozen:
        library.freeze()
    print(json.dumps({
        "missions": [s["name"] for s in specs], "library_options": cfg.num_options,
        "steps_per_mission": cfg.total_timesteps,
        "aggregate_steps": cfg.total_timesteps * len(specs),
        "envs_per_mission": manifest["num_envs"], "frozen_library": frozen,
        "log_dir": str(log_dir), "checkpoint_dir": str(checkpoint_dir),
    }, indent=2), flush=True)
    if args.dry_run:
        return 0
    if not args.checkpoint and ((log_dir / "campaign.json").exists() or (checkpoint_dir / "oc3_latest.pt").exists()):
        raise ValueError("Campaign already exists; use --checkpoint or new output directories")
    log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "campaign.json").write_text(json.dumps({
        **manifest, "fingerprint": fingerprint, "frozen": frozen,
        "source_missions": source_missions,
        "resume_semantics": "weights/optimizers/RNG/counters restored; fresh simulator episodes",
    }, indent=2), encoding="utf-8")
    optimizer = None if frozen else torch.optim.Adam(library.parameters(), lr=cfg.actor_lr, eps=cfg.adam_eps)
    if resume and optimizer is not None:
        optimizer.load_state_dict(resume["library_optimizer"])
    version = resume["version"] if resume else 0
    round_index = resume["round"] if resume else 0
    progress = resume["progress"] if resume else 0
    writer = SummaryWriter(str(log_dir / "shared"), purge_step=progress * len(specs) + 1 if resume else None)
    workers = []
    started = time.monotonic()

    def checkpoint(final=False):
        states = broadcast(workers, "snapshot")
        bundle = {
            "schema": SCHEMA, "kind": "oc3_training", "fingerprint": fingerprint,
            "manifest": manifest, "frozen": frozen, "library": cpu_tree(library.state_dict()),
            "library_optimizer": cpu_tree(optimizer.state_dict()) if optimizer else None,
            "missions": {w.name: state for w, state in zip(workers, states)},
            "round": round_index, "version": version, "progress": progress,
            "source_missions": source_missions, "source_config": source_cfg,
        }
        atomic_save(bundle, checkpoint_dir / "oc3_latest.pt")
        atomic_save(bundle, checkpoint_dir / ("oc3_final.pt" if final else f"oc3_round_{round_index:08d}.pt"))
        atomic_save({
            "schema": SCHEMA, "kind": "oc3_library", "library": cpu_tree(library.state_dict()),
            "training_missions": source_missions, "source_config": source_cfg,
            "observation_contract": networks.OBSERVATION_CONTRACT,
            "library_version": version,
        }, checkpoint_dir / "option_library.pt")
        numbered = sorted(checkpoint_dir.glob("oc3_round_*.pt"))
        for old in numbered[:-config["keep_checkpoints"]]:
            old.unlink()
        print(f"[OC3] Checkpoint: {checkpoint_dir / 'oc3_latest.pt'}", flush=True)

    try:
        context = mp.get_context("spawn")
        for spec in specs:
            print(f"[OC3] Starting {spec['name']}; log: {spec['trainer']['log_dir']}/worker.log", flush=True)
            worker = Worker(context, spec, cpu_tree(library.state_dict()), frozen, args.device,
                            config["worker_threads"], config["worker_timeout_s"])
            workers.append(worker)
            if not worker.receive().get("ready"):
                raise RuntimeError("Worker did not complete initialization")
            if resume:
                worker.send("restore", state=resume["missions"][worker.name])
                worker.receive()
        while progress < cfg.total_timesteps:
            remaining = (cfg.total_timesteps - progress) // (manifest["num_envs"] * 20)
            steps = min(cfg.horizon, remaining)
            print(f"[OC3] Round {round_index + 1}: collecting {steps} decisions/env/mission", flush=True)
            collected = broadcast(workers, "collect", steps=steps, version=version)
            if len({item["step"] for item in collected}) != 1:
                raise RuntimeError("Mission sample counts diverged")
            batches = min(item["batches"] for item in collected)
            if optimizer:
                for group in optimizer.param_groups:
                    group["lr"] = collected[0]["actor_lr"]
            actor_stopped = False
            gradient_norms = []
            for _epoch in range(cfg.num_epochs):
                broadcast(workers, "epoch")
                for _batch in range(batches):
                    replies = broadcast(workers, "gradients", version=version)
                    if any(reply["version"] != version for reply in replies):
                        raise RuntimeError("Worker supplied a stale shared gradient")
                    actor_stopped |= any(reply["stop_actor"] for reply in replies)
                    if not actor_stopped:
                        if optimizer:
                            optimizer.zero_grad(set_to_none=True)
                            networks.average_gradients(library, [reply["gradients"] for reply in replies])
                            norm = torch.nn.utils.clip_grad_norm_(library.parameters(), cfg.actor_max_grad_norm, error_if_nonfinite=True)
                            gradient_norms.append(float(norm))
                            optimizer.step()
                        version += 1
                    broadcast(workers, "commit", apply_actor=not actor_stopped,
                              library_state=cpu_tree(library.state_dict()), version=version)
            metrics = broadcast(workers, "finish")
            progress = collected[0]["step"]
            round_index += 1
            total = progress * len(workers)
            writer.add_scalar("OC3/Per Mission Decisions", progress, total)
            writer.add_scalar("OC3/Global Actor Early Stop", int(actor_stopped), total)
            writer.add_scalar("OC3/Shared Gradient Norm", sum(gradient_norms) / max(1, len(gradient_norms)), total)
            writer.flush()
            scores = ", ".join(
                f"{w.name}={m['episode_return']:.2f}" if "episode_return" in m
                else f"{w.name}=no completed episode" for w, m in zip(workers, metrics)
            )
            print(f"[OC3] round={round_index} per_mission={progress:,} aggregate={total:,} "
                  f"returns({scores}) actor_stop={actor_stopped} elapsed={time.monotonic()-started:.0f}s", flush=True)
            if round_index % config["checkpoint_rounds"] == 0:
                checkpoint()
        checkpoint(final=True)
        for worker in workers:
            worker.send("export", path=str(checkpoint_dir / worker.name / "option_critic_2_final.pt"))
            worker.receive()
        print("[OC3] Complete. Mission exports are compatible with play.py; resume from the OC3 bundle.", flush=True)
    finally:
        for worker in workers:
            worker.close()
        writer.close()
    return 0


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())
