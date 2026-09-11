#!/usr/bin/env python3
"""Simulator-free regressions for fixed-module swarm Option-Critic training."""

from __future__ import annotations

import copy
import importlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from validate_oc2_training import AutoResetEnv, BUFFER as LearnedBuffer


TRAINING = importlib.import_module("swarmacb_oc2_validation.option_critic_trainer")
BUFFER = importlib.import_module(
    "swarmacb_oc2_validation.option_critic_buffer"
).FixedOptionRolloutBuffer


class FixedAutoResetEnv(AutoResetEnv):
    """The same reset fixture with OC1's four sensors and six module IDs."""

    def __init__(self):
        super().__init__(agents=20)
        self.cfg.discrete_actions = True
        self.cfg.num_actions = 6

    def _get_observations(self):
        return {agent: obs[:, :4] for agent, obs in super()._get_observations().items()}

    def step(self, actions):
        for value in actions.values():
            assert value.dtype == torch.long and value.shape == (2, 1)
            assert bool(((value >= 0) & (value < 6)).all())
        self.episode_length_buf += 1
        terminated, truncated = self._get_dones()
        done = truncated[self.cfg.possible_agents[0]]
        self.completed_group_reward[done] = self.episode_length_buf[done].float()
        self.episode_length_buf[done] = 0
        rewards = {agent: torch.full((2,), 0.1) for agent in self.cfg.possible_agents}
        return self._get_observations(), rewards, terminated, truncated, {}


def make_buffer(horizon=16, envs=2, agents=1, lam=0.95):
    return BUFFER(horizon, envs, agents, 4, 5, 8, 8, 0.99, lam, "cpu")


class ReturnTests(unittest.TestCase):
    def test_persistent_options_do_not_blend_their_values(self):
        buffer = make_buffer(horizon=1000)
        buffer.ptr = 1000
        buffer.rewards[:, 0] = 1.0
        buffer.options[:, 1] = 1
        buffer.joint_option_values[:, 0] = 100.0
        buffer.team_values.fill_(50.0)
        buffer.compute_returns_and_advantages(torch.tensor([100.0, 0.0]))
        torch.testing.assert_close(buffer.returns[:, 0], torch.full((1000,), 100.0))
        torch.testing.assert_close(buffer.returns[:, 1], torch.zeros(1000))

    def test_terminal_and_timeout_ignore_reset_and_cutoff_values(self):
        buffer = make_buffer(horizon=3)
        buffer.ptr = 3
        buffer.rewards.fill_(1.0)
        buffer.joint_option_values.fill_(10000.0)
        buffer.dones[0] = 1.0
        buffer.timeouts[0, 1] = 1.0
        buffer.timeout_values[0, 1] = 7.0
        buffer.dones[2] = 1.0
        buffer.timeouts[2, 1] = 1.0
        buffer.timeout_values[2, 1] = 9.0
        buffer.compute_returns_and_advantages(torch.full((2,), 10000.0))
        torch.testing.assert_close(buffer.returns[0], torch.tensor([1.0, 7.93]))
        torch.testing.assert_close(buffer.returns[2], torch.tensor([1.0, 9.91]))

    def test_one_step_and_monte_carlo_limits(self):
        for lam in (0.0, 1.0):
            buffer = make_buffer(horizon=3, envs=1, lam=lam)
            buffer.ptr = 3
            buffer.rewards[:, 0] = torch.tensor([1.0, 2.0, 3.0])
            buffer.joint_option_values[:, 0] = torch.tensor([10.0, 20.0, 30.0])
            buffer.compute_returns_and_advantages(torch.tensor([40.0]))
            expected = 1 + 0.99 * 20 if lam == 0 else 1 + 0.99 * (2 + 0.99 * (3 + 0.99 * 40))
            self.assertAlmostEqual(buffer.returns[0, 0].item(), expected, places=4)

    def test_oc1_and_oc2_use_the_same_return_operator(self):
        torch.manual_seed(13)
        fixed = make_buffer(horizon=32, agents=20)
        learned = LearnedBuffer(32, 2, 20, 24, 5, 2, 8, 8, 0.99, 0.95, "cpu")
        fixed.ptr = learned.ptr = 32
        for name in ("rewards", "joint_option_values", "team_values", "timeout_values"):
            values = torch.randn_like(getattr(fixed, name))
            getattr(fixed, name).copy_(values)
            getattr(learned, name).copy_(values)
        for buffer in (fixed, learned):
            buffer.dones[6, 0] = 1.0
            buffer.dones[20, 1] = buffer.timeouts[20, 1] = 1.0
            buffer.compute_returns_and_advantages(torch.tensor([2.0, 3.0]))
        torch.testing.assert_close(fixed.returns, learned.returns)

    def test_missing_arrival_context_is_rejected(self):
        buffer = make_buffer()
        buffer.ptr = 1
        with self.assertRaisesRegex(RuntimeError, "missing next-option"):
            next(buffer.get_sequence_batches(1, 2))


class TrainerTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)
        self.temp = tempfile.TemporaryDirectory(prefix="oc1_training_test_")
        self.addCleanup(self.temp.cleanup)
        cfg = TRAINING.FixedOptionCriticConfig(
            hidden_dim=32, critic_hidden_dim=32, critic_num_layers=1,
            critic_num_heads=4, memory_size=16, decision_period=1,
            horizon=6, sequence_length=4, mini_batch_size=160, num_epochs=1,
            total_timesteps=1000, log_dir=str(Path(self.temp.name) / "logs"),
            checkpoint_dir=str(Path(self.temp.name) / "checkpoints"),
        )
        self.trainer = TRAINING.FixedOptionCriticTrainer(FixedAutoResetEnv(), cfg)
        self.addCleanup(self.trainer.writer.close)

    def test_actual_arrivals_replace_cutoff_sample_without_crossing_resets(self):
        trainer = self.trainer
        obs = trainer.collect_rollout(trainer.env.reset()[0], rollout_steps=2)
        trainer.buffer.next_options[1].fill_(-42)
        trainer.collect_rollout(obs, rollout_steps=4, reset_buffer=False)
        buffer = trainer.buffer
        for t in range(buffer.ptr - 1):
            active = ~buffer.dones[t].bool()
            torch.testing.assert_close(buffer.next_options[t, active], buffer.options[t + 1, active])
            memory = tuple(
                getattr(buffer, name)[t + 1].reshape(1, 40, -1)
                for name in ("memory_h", "memory_c")
            )
            with torch.no_grad():
                logits = trainer.manager.step(buffer.obs[t + 1].reshape(-1, 4), memory)[0]
                mu = logits.softmax(-1).view(2, 20, 6)
            torch.testing.assert_close(buffer.next_option_probs[t, active], mu[active])
            self.assertFalse(bool(buffer.next_options_valid[t, ~active].any()))
        self.assertEqual(buffer.memory_h[3, 0].abs().sum().item(), 0)
        self.assertEqual(buffer.joint_memory_h[3, 0].abs().sum().item(), 0)
        self.assertEqual(buffer.next_option_probs[2, 0].abs().sum().item(), 0)
        self.assertTrue(buffer.next_options_valid[5, 1].item())

    def test_bootstrap_keeps_live_memory_and_options_unchanged(self):
        trainer = self.trainer
        obs = trainer.collect_rollout(trainer.env.reset()[0], rollout_steps=2)
        names = ("manager_memory_h", "manager_memory_c", "joint_memory_h", "joint_memory_c", "current_options")
        before = {name: getattr(trainer, name).clone() for name in names}
        with patch.object(trainer.critic, "critic_pass", side_effect=AssertionError("option-blind bootstrap")):
            value, choices, mu = trainer._bootstrap_option_value(
                torch.stack(list(obs.values()), dim=1), trainer.env.get_critic_state(),
                trainer.current_options,
                (trainer.manager_memory_h, trainer.manager_memory_c),
                (trainer.joint_memory_h, trainer.joint_memory_c),
            )
        for name in names:
            torch.testing.assert_close(getattr(trainer, name), before[name])
        self.assertEqual(value.shape, (2,))
        self.assertEqual(choices.shape, (2, 20))
        torch.testing.assert_close(mu.sum(-1), torch.ones(2, 20))

    def test_bootstrap_respects_beta_zero_one_and_fresh_episodes(self):
        trainer = self.trainer
        logits = torch.full((40, 6), -float("inf"))
        logits[:, 5] = 0.0
        obs = torch.zeros(2, 20, 4)
        previous = torch.zeros(2, 20, dtype=torch.long)
        previous[1] = -1
        for logit, expected in ((-1000.0, 0), (1000.0, 5)):
            outputs = (logits, torch.full_like(logits, logit), None)
            with patch.object(trainer.manager, "step", return_value=outputs):
                _, choices, _ = trainer._bootstrap_option_value(
                    obs, trainer.env.get_critic_state(), previous,
                    (trainer.manager_memory_h, trainer.manager_memory_c),
                    (trainer.joint_memory_h, trainer.joint_memory_c),
                )
            self.assertTrue(bool((choices[0] == expected).all()))
            self.assertTrue(bool((choices[1] == 5).all()))

    def test_timeout_bootstrap_uses_terminal_sensors_and_unreset_memory(self):
        trainer = self.trainer
        original = trainer._bootstrap_option_value
        terminal_calls = []

        def bootstrap(obs, states, options, manager_memory, joint_memory):
            if obs.shape[0] == 1:
                terminal_calls.append(obs.clone())
                self.assertEqual(obs.shape, (1, 20, 4))
                self.assertTrue(bool((obs == 3).all()))
                self.assertTrue(bool((states == 3).all()))
                self.assertTrue(bool((options >= 0).all()))
                self.assertGreater(manager_memory[0].abs().sum().item(), 0)
                self.assertGreater(joint_memory[0].abs().sum().item(), 0)
                return torch.tensor([7.0]), options, torch.full((1, 20, 6), 1 / 6)
            return original(obs, states, options, manager_memory, joint_memory)

        with patch.object(trainer, "_bootstrap_option_value", side_effect=bootstrap):
            trainer.collect_rollout(trainer.env.reset()[0], rollout_steps=3)
        self.assertEqual(len(terminal_calls), 1)
        self.assertAlmostEqual(trainer.buffer.returns[2, 0].item(), 0.1 + 0.99 * 7, places=5)
        self.assertEqual(trainer.buffer.next_obs[2, 0].abs().sum().item(), 0)

    def test_termination_uses_next_peers_and_detached_behavior_probabilities(self):
        trainer = self.trainer
        trainer.collect_rollout(trainer.env.reset()[0], rollout_steps=2)
        batch = next(trainer.buffer.get_sequence_batches(2, 80))
        batch["options"].zero_()
        batch["critic_options"].zero_()
        batch["next_critic_options"].fill_(1)
        batch["next_option_probs"].zero_()
        batch["next_option_probs"][..., 0] = 0.25
        batch["next_option_probs"][..., 1] = 0.75

        def candidates(states, choices, focal_ids, num_options, memory):
            self.assertTrue(bool((choices == 1).all()))
            return torch.tensor([0.0, 10.0, 2.0, 2.0, 2.0, 2.0]).expand(states.shape[0], -1)

        with patch.object(trainer.critic, "focal_discrete_counterfactual_values", side_effect=candidates):
            losses = trainer._compute_sequence_losses(batch, trainer.current_eps)
            self.assertAlmostEqual(losses[-1].item(), -7.5, places=5)
            losses[4].backward()
            self.assertLess(trainer.manager.termination_head.bias.grad[0].item(), 0)
            self.assertIsNone(trainer.manager.option_head.weight.grad)
            self.assertTrue(all(p.grad is None for p in trainer.critic.parameters()))
            batch["dones"].fill_(1)
            terminal_losses = trainer._compute_sequence_losses(batch, trainer.current_eps)
            self.assertEqual(terminal_losses[4].item(), 0)

    def test_real_update_checkpoint_roundtrip_and_legacy_rejection(self):
        trainer = self.trainer
        trainer.collect_rollout(trainer.env.reset()[0], rollout_steps=6)
        before = {name: value.clone() for name, value in trainer.manager.named_parameters()}
        metrics = trainer.update()
        self.assertTrue(all(bool(torch.isfinite(torch.as_tensor(value)).all()) for value in metrics.values()))
        self.assertTrue(any(not torch.equal(before[name], value) for name, value in trainer.manager.named_parameters()))
        path = Path(self.temp.name) / "oc1.pt"
        trainer.save_checkpoint(path)
        checkpoint = torch.load(path, weights_only=False)
        self.assertEqual(checkpoint["option_critic_version"], 7)
        self.assertEqual(checkpoint["training_checkpoint_version"], 8)
        self.assertEqual(checkpoint["return_bootstrap"], "sampled_joint_option_q")
        trainer.load_checkpoint(path)
        reloaded = copy.deepcopy(trainer.manager)
        reloaded.load_state_dict(checkpoint["manager"])
        obs = torch.randn(3, 4)
        with torch.no_grad():
            for actual, expected in zip(reloaded.step(obs)[:2], trainer.manager.step(obs)[:2]):
                torch.testing.assert_close(actual, expected)
        del checkpoint["training_checkpoint_version"]
        torch.save(checkpoint, path)
        with self.assertRaisesRegex(RuntimeError, "fresh training"):
            trainer.load_checkpoint(path)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
