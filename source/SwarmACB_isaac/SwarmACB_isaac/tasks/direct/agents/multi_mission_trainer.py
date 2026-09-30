"""Synchronous OC3 mission learner, reusing the tested OC2 objectives.

Each process owns one simulator, private Q/beta networks and critics. A single
coordinator owns the motor-library optimizer. No stale-policy replay is used.
"""

from __future__ import annotations

import copy
import random
from pathlib import Path

import numpy as np
import torch

from .multi_mission_networks import MissionOptionController, SharedOptionLibrary


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(cpu_tree(item) for item in value)
    return copy.deepcopy(value)


def rng_state():
    return {
        "python": random.getstate(), "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [],
    }


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


class MissionLearner:
    def __init__(self, trainer, library_state, frozen=False):
        self.trainer = trainer
        self.frozen = frozen
        self.library = SharedOptionLibrary(trainer.actor).to(trainer.device)
        self.library.load_state_dict(library_state, strict=True)
        if frozen:
            self.library.freeze()
        self.controller = MissionOptionController(trainer.actor, self.library)
        # Reference must be independent, including its library parameters.
        trainer.reference_actor = copy.deepcopy(trainer.actor).eval().requires_grad_(False)
        self.private_parameters = self.controller.private_parameters()
        trainer.actor_parameters = [p for p in trainer.actor.parameters() if p.requires_grad]
        trainer.actor_optimizer = torch.optim.Adam(
            self.private_parameters, lr=trainer.cfg.actor_lr, eps=trainer.cfg.adam_eps,
        )
        self.obs = trainer.env.reset()[0]
        trainer._zero_memories()
        self.version = 0
        self.pending = False
        self.round_active = False

    def collect(self, steps, version):
        if self.round_active or self.pending or version != self.version:
            raise RuntimeError("Invalid rollout boundary or stale library version")
        t = self.trainer
        t._apply_schedules()
        before = len(t._completed_episode_returns)
        self.obs = t.collect_rollout(self.obs, rollout_steps=steps)
        t._apply_schedules()
        t.buffer.action_advantages[:t.buffer.ptr] = t._normalize(
            t.buffer.action_advantages[:t.buffer.ptr],
        )
        t.reference_actor.load_state_dict(t.actor.state_dict())
        t.reference_actor.flatten_recurrent_parameters()
        self.round_active = True
        self.totals = {}
        self.batch_updates = 0
        self.actor_updates = 0
        self.actor_norm_total = 0.0
        self.critic_norm_total = 0.0
        self.max_kl = 0.0
        self.round_metrics = self.rollout_metrics()
        scores = t._completed_group_rewards[before:]
        returns = t._completed_episode_returns[before:]
        self.round_metrics["episodes"] = len(scores)
        if scores:
            self.round_metrics["group_return"] = float(np.mean(scores))
            self.round_metrics["episode_return"] = float(np.mean(returns))
            self.round_metrics["episode_length"] = float(np.mean(t._completed_episode_lengths[before:]))
        for name in ("_completed_episode_returns", "_completed_group_rewards", "_completed_episode_lengths"):
            getattr(t, name).clear()
        return {
            "batches": self.batch_count(), "step": t.global_step,
            "actor_lr": t.current_actor_lr, "metrics": self.round_metrics,
        }

    def batch_count(self):
        b = self.trainer.buffer
        length = min(self.trainer.cfg.sequence_length, b.ptr)
        chunks = 0
        for env in range(b.num_envs):
            ends = (b.dones[:b.ptr, env] > 0.5).nonzero().flatten().cpu().tolist()
            ends = [index + 1 for index in ends]
            if not ends or ends[-1] != b.ptr:
                ends.append(b.ptr)
            start = 0
            for end in ends:
                chunks += ((end - start + length - 1) // length) * b.num_agents
                start = end
        per_batch = max(1, self.trainer.cfg.mini_batch_size // length)
        return max(1, chunks // per_batch)

    def start_epoch(self):
        if not self.round_active or self.pending:
            raise RuntimeError("Invalid epoch boundary")
        t = self.trainer
        self.batches = iter(t.buffer.get_sequence_batches(t.cfg.sequence_length, t.cfg.mini_batch_size))

    def gradients(self, version):
        if not self.round_active or self.pending or version != self.version:
            raise RuntimeError("Stale gradient request or uncommitted minibatch")
        t = self.trainer
        batch = next(self.batches)
        losses = t._compute_sequence_losses(batch, t.current_eps, t.reference_actor)
        terms, critic_loss = t.loss_objectives(losses)
        actor_loss = sum(terms.values())
        if not torch.isfinite(actor_loss) or not torch.isfinite(critic_loss):
            raise FloatingPointError("Non-finite OC3 objective")
        t.actor.zero_grad(set_to_none=True)
        actor_loss.backward()
        # Match OC2's per-actor clipping before averaging mission gradients.
        actor_norm = torch.nn.utils.clip_grad_norm_(
            t.actor_parameters, t.cfg.actor_max_grad_norm, error_if_nonfinite=True,
        )
        t.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        critic_norm = torch.nn.utils.clip_grad_norm_(
            t.critic_parameters, t.cfg.max_grad_norm, error_if_nonfinite=True,
        )
        kl = max(losses["action_approx_kl"].item(), losses["option_approx_kl"].item())
        if self.batch_updates == 0 and kl > 1e-6:
            raise RuntimeError(f"OC3 update-start reference mismatch: KL={kl}")
        self.max_kl = max(self.max_kl, kl)
        self.actor_norm_total += float(actor_norm)
        self.critic_norm_total += float(critic_norm)
        for name, value in {**losses, **terms, "actor_objective": actor_loss, "critic_objective": critic_loss}.items():
            self.totals[name] = self.totals.get(name, 0.0) + float(value.detach())
        self.pending = True
        return {
            "version": version, "gradients": self.library.gradient_payload(),
            "kl": kl, "gradient_norm": float(actor_norm),
            "stop_actor": t.cfg.target_kl > 0 and kl > 1.5 * t.cfg.target_kl,
        }

    def commit(self, apply_actor, library_state, version):
        if not self.pending or version != self.version + int(apply_actor):
            raise RuntimeError("Invalid OC3 commit/version")
        t = self.trainer
        if apply_actor:
            t.actor_optimizer.step()
            self.actor_updates += 1
        t.critic_optimizer.step()
        self.library.load_state_dict(library_state, strict=True)
        for module in (t.actor, t.team_critic, t.action_critic, t.option_critic):
            if any(not torch.isfinite(p).all() for p in module.parameters()):
                raise FloatingPointError("Non-finite OC3 parameter after update")
        self.version = version
        self.batch_updates += 1
        self.pending = False

    def finish(self):
        if not self.round_active or self.pending or not self.actor_updates:
            raise RuntimeError("Cannot finish an incomplete OC3 update")
        t = self.trainer
        t.update_count += 1
        metrics = {key: value / self.batch_updates for key, value in self.totals.items()}
        metrics.update(self.round_metrics)
        metrics.update({
            "actor_updates": self.actor_updates, "critic_updates": self.batch_updates,
            "actor_gradient_norm": self.actor_norm_total / self.batch_updates,
            "critic_gradient_norm": self.critic_norm_total / self.batch_updates,
            "max_policy_kl": self.max_kl, "actor_lr": t.current_actor_lr,
            "option_epsilon": t.current_option_epsilon,
        })
        for name, value in metrics.items():
            t.writer.add_scalar(f"OC3/{name}", value, t.global_step)
        if "episode_return" in metrics:
            t.writer.add_scalar("Environment/Cumulative Reward", metrics["episode_return"], t.global_step)
            t.writer.add_scalar("Environment/Episode Length", metrics["episode_length"], t.global_step)
            t.writer.add_scalar("Extra/Group Reward Mean", metrics["group_return"], t.global_step)
        t.writer.flush()
        self.round_active = False
        return metrics

    def rollout_metrics(self):
        b = self.trainer.buffer
        n = b.ptr
        options = b.options[:n]
        metrics = {
            "decision_samples": n * b.num_envs * b.num_agents,
            "scaled_reward_mean": float(b.rewards[:n].mean()),
            "action_clipping": float((b.actions[:n].abs() > 3).float().mean()),
        }
        valid = b.termination_valid[:n] > 0.5
        denom = valid.sum().clamp_min(1)
        metrics["termination_event_rate"] = float((b.option_masks[:n] * valid).sum() / denom)
        metrics["actual_option_change_rate"] = float(
            ((options != b.termination_options[:n]) & valid).sum() / denom,
        )
        metrics["mean_termination_probability"] = float((b.beta_probs[:n] * valid).sum() / denom)
        for option in range(b.num_options):
            metrics[f"option_usage/{option}"] = float((options == option).float().mean())
            metrics[f"option_std/{option}"] = float(self.library.log_std[option].detach().exp().mean())
        return metrics

    def snapshot(self):
        if self.round_active or self.pending:
            raise RuntimeError("Only completed OC3 rounds can be checkpointed")
        t = self.trainer
        return cpu_tree({
            "controller": self.controller.private_state_dict(),
            "critics": {name: getattr(t, name).state_dict() for name in ("team_critic", "action_critic", "option_critic")},
            "actor_optimizer": t.actor_optimizer.state_dict(),
            "critic_optimizer": t.critic_optimizer.state_dict(),
            "step": t.global_step, "updates": t.update_count,
            "version": self.version, "rng": rng_state(),
        })

    def restore(self, state):
        t = self.trainer
        self.controller.load_private_state_dict(state["controller"])
        for name, values in state["critics"].items():
            getattr(t, name).load_state_dict(values, strict=True)
        t.actor_optimizer.load_state_dict(state["actor_optimizer"])
        t.critic_optimizer.load_state_dict(state["critic_optimizer"])
        t.global_step, t.update_count, self.version = state["step"], state["updates"], state["version"]
        from torch.utils.tensorboard import SummaryWriter
        t.writer.close()
        t.writer = SummaryWriter(t.cfg.log_dir, purge_step=t.global_step + 1)
        # Simulator state is deliberately not serialized. Resume starts fresh
        # episodes, never combining saved recurrent memory with a new layout.
        restore_rng(state["rng"])
        self.obs = t.env.reset()[0]
        t._zero_memories()
        t._episode_reward_acc.zero_()
        t._episode_step_count.zero_()
        t._apply_schedules()

    def export_actor(self, path, mission):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        self.trainer.save_checkpoint(temporary)
        data = torch.load(temporary, map_location="cpu", weights_only=False)
        data["oc3_inference_export"] = True
        data["oc3_mission"] = mission
        data["option_critic_phase"] = 4 if self.frozen else 3
        data.pop("actor_optimizer")
        data.pop("critic_optimizer")
        torch.save(data, temporary)
        temporary.replace(path)
