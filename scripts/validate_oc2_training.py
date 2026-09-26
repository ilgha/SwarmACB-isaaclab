#!/usr/bin/env python3
"""Simulator-free regression tests for OC2's swarm Option-Critic update."""

from __future__ import annotations

import ast
import importlib
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from validate_oc2_architecture import _load_networks


ROOT = Path(__file__).resolve().parents[1]
NETWORKS = _load_networks(ROOT)
TRAINING = importlib.import_module(
    "swarmacb_oc2_validation.learned_option_critic_trainer"
)
BUFFER = importlib.import_module(
    "swarmacb_oc2_validation.learned_option_critic_buffer"
).LearnedOptionRolloutBuffer


def _mission_done_method():
    """Exercise the real pre-reset capture method without importing Isaac Sim."""
    path = ROOT / (
        "source/SwarmACB_isaac/SwarmACB_isaac/tasks/direct/missions/"
        "directional_gate/directional_gate_env.py"
    )
    tree = ast.parse(path.read_text(encoding="utf-8"))
    env_class = next(node for node in tree.body if isinstance(node, ast.ClassDef))
    method = next(node for node in env_class.body if getattr(node, "name", None) == "_get_dones")
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["_get_dones"]


class AutoResetEnv:
    """Small vector environment with distinguishable terminal/reset observations."""

    _get_dones = _mission_done_method()

    def __init__(self, agents=20):
        self.unwrapped = self
        self.device = "cpu"
        self.num_envs = 2
        self.scene = SimpleNamespace(num_envs=2)
        self.cfg = SimpleNamespace(
            num_agents=agents, variant="cyclamen", discrete_actions=False,
            possible_agents=[f"robot_{i}" for i in range(agents)],
            action_spaces={f"robot_{i}": 2 for i in range(agents)},
        )
        self.max_episode_length = torch.tensor([3, 5])
        self.episode_length_buf = torch.zeros(2, dtype=torch.long)
        self.capture_terminal_policy_observations = False
        self.completed_terminal_policy_observations = None
        self.completed_terminal_critic_state = torch.zeros(2, agents, 5)
        self.completed_group_reward = torch.zeros(2)

    def _get_observations(self):
        return {
            agent: self.episode_length_buf[:, None].float().expand(-1, 24).clone()
            for agent in self.cfg.possible_agents
        }

    def get_critic_state(self):
        return self.episode_length_buf[:, None, None].float().expand(
            -1, self.cfg.num_agents, 5,
        ).clone()

    def reset(self):
        self.episode_length_buf.zero_()
        return self._get_observations(), {}

    def step(self, actions):
        assert all(bool((value.abs() <= 1).all()) for value in actions.values())
        self.episode_length_buf += 1
        terminated, truncated = self._get_dones()
        done = truncated[self.cfg.possible_agents[0]]
        self.completed_group_reward[done] = self.episode_length_buf[done].float()
        self.episode_length_buf[done] = 0
        rewards = {agent: torch.full((2,), 0.1) for agent in self.cfg.possible_agents}
        return self._get_observations(), rewards, terminated, truncated, {}


def make_buffer(horizon=16, envs=2, agents=1, options=2, lam=0.95):
    return BUFFER(horizon, envs, agents, 24, 5, 2, 8, 8, 0.99, lam, "cpu", options)


class ReturnTests(unittest.TestCase):
    def test_persistent_options_keep_different_values_at_same_state(self):
        buffer = make_buffer()
        buffer.ptr = 16
        buffer.rewards[:, 0] = 1.0
        buffer.joint_option_values[:, 0] = 100.0
        buffer.team_values.fill_(50.0)
        buffer.compute_returns_and_advantages(torch.tensor([100.0, 0.0]))
        torch.testing.assert_close(buffer.returns[:, 0], torch.full((16,), 100.0))
        torch.testing.assert_close(buffer.returns[:, 1], torch.zeros(16))

    def test_one_step_and_monte_carlo_limits(self):
        for lam in (0.0, 1.0):
            buffer = make_buffer(horizon=3, envs=1, lam=lam)
            buffer.ptr = 3
            buffer.rewards[:, 0] = torch.tensor([1.0, 2.0, 3.0])
            buffer.joint_option_values[:, 0] = torch.tensor([10.0, 20.0, 30.0])
            buffer.compute_returns_and_advantages(torch.tensor([40.0]))
            expected = 1 + 0.99 * 20 if lam == 0 else 1 + 0.99 * (2 + 0.99 * (3 + 0.99 * 40))
            self.assertAlmostEqual(buffer.returns[0, 0].item(), expected, places=4)

    def test_terminal_and_timeout_never_use_reset_values(self):
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

    def test_missing_context_fails_before_optimization(self):
        buffer = make_buffer()
        buffer.ptr = 1
        with self.assertRaisesRegex(RuntimeError, "missing next-option"):
            next(buffer.get_sequence_batches(1, 2))

    def test_arrival_distribution_and_single_agent_oc_identity(self):
        mu = torch.tensor([[0.25, 0.75]]).expand(4, -1)
        beta = torch.tensor([0.0, 1.0, 0.4, 0.0])
        previous = torch.tensor([0, 0, 0, -1])
        probs = NETWORKS.option_transition_probs(mu, beta, previous)
        torch.testing.assert_close(probs, torch.tensor([
            [1.0, 0.0], [0.25, 0.75], [0.7, 0.3], [0.25, 0.75],
        ]))
        q = torch.tensor([10.0, 2.0])
        arrival_value = (probs[2] * q).sum()
        expected = (1 - beta[2]) * q[0] + beta[2] * (mu[2] * q).sum()
        torch.testing.assert_close(arrival_value, expected)


class TrainerTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)
        self.temp = tempfile.TemporaryDirectory(prefix="oc2_training_test_")
        self.addCleanup(self.temp.cleanup)

    def trainer(self, options=2, agents=20, reactive=False, linear=False):
        cfg = TRAINING.LearnedOptionCriticConfig(
            num_options=options, hidden_dim=32, option_hidden_dim=32,
            num_layers=1, option_num_layers=1, memory_size=16, option_memory_size=16,
            critic_hidden_dim=32, critic_num_heads=4, decision_period=1,
            reactive_intra_options=reactive,
            linear_intra_options=linear,
            horizon=6, sequence_length=4, mini_batch_size=160, num_epochs=1,
            option_epsilon_schedule="constant", total_timesteps=1000,
            fused_optimizer=False, matmul_precision="highest",
            log_dir=str(Path(self.temp.name) / "logs"),
            checkpoint_dir=str(Path(self.temp.name) / "checkpoints"),
        )
        trainer = TRAINING.LearnedOptionCriticTrainer(AutoResetEnv(agents), cfg)
        self.addCleanup(trainer.writer.close)
        return trainer

    def test_actual_next_choices_replace_cutoff_sample_and_resets_are_masked(self):
        trainer = self.trainer()
        obs = trainer.env.reset()[0]
        obs = trainer.collect_rollout(obs, rollout_steps=2)
        trainer.collect_rollout(obs, rollout_steps=4, reset_buffer=False)
        buffer = trainer.buffer
        for t in range(buffer.ptr - 1):
            active = ~buffer.dones[t].bool()
            torch.testing.assert_close(buffer.next_options[t, active], buffer.options[t + 1, active])
            obs_next = buffer.obs[t + 1].reshape(-1, 24)
            memory = tuple(getattr(buffer, name)[t + 1].reshape(1, 40, -1) for name in ("memory_h", "memory_c"))
            with torch.no_grad():
                values = trainer.actor.step(obs_next, memory)[1]
                mu = trainer.actor.option_dist(values, epsilon=trainer.current_option_epsilon).probs.view(2, 20, -1)
            torch.testing.assert_close(buffer.next_option_probs[t, active], mu[active])
            self.assertFalse(bool(buffer.next_options_valid[t, ~active].any()))
        self.assertTrue(bool(buffer.next_options_valid[-1].logical_not().all()))  # unused capacity
        for name in ("memory_h", "option_joint_memory_h"):
            self.assertEqual(getattr(buffer, name)[3, 0].abs().sum().item(), 0.0)

    def test_bootstrap_does_not_advance_live_memory_or_options(self):
        trainer = self.trainer()
        obs = trainer.collect_rollout(trainer.env.reset()[0], rollout_steps=2)
        names = ("actor_memory_h", "actor_memory_c", "option_joint_memory_h", "option_joint_memory_c", "current_options")
        before = {name: getattr(trainer, name).clone() for name in names}
        with patch.object(trainer.team_critic, "critic_pass", side_effect=AssertionError("option-blind bootstrap")):
            value, choices, mu = trainer._bootstrap_option_value(
                torch.stack(list(obs.values()), dim=1), trainer.env.get_critic_state(),
                trainer.current_options,
                (trainer.actor_memory_h, trainer.actor_memory_c),
                (trainer.option_joint_memory_h, trainer.option_joint_memory_c),
            )
        for name in names:
            torch.testing.assert_close(getattr(trainer, name), before[name])
        self.assertEqual(value.shape, (2,))
        self.assertEqual(choices.shape, (2, 20))
        torch.testing.assert_close(mu.sum(-1), torch.ones(2, 20))

    def test_timeout_bootstrap_uses_pre_reset_sensor_state_and_memory(self):
        trainer = self.trainer()
        original = trainer._bootstrap_option_value
        terminal_calls = []

        def bootstrap(obs, states, options, actor_memory, option_memory):
            if obs.shape[0] == 1:
                terminal_calls.append(obs.clone())
                self.assertTrue(bool((obs == 3).all()))
                self.assertTrue(bool((states == 3).all()))
                self.assertTrue(bool((options >= 0).all()))
                self.assertGreater(actor_memory[0].abs().sum().item(), 0)
                self.assertGreater(option_memory[0].abs().sum().item(), 0)
                return torch.tensor([7.0]), options, torch.full((1, 20, 2), 0.5)
            return original(obs, states, options, actor_memory, option_memory)

        with patch.object(trainer, "_bootstrap_option_value", side_effect=bootstrap):
            trainer.collect_rollout(trainer.env.reset()[0], rollout_steps=3)
        self.assertEqual(len(terminal_calls), 1)
        self.assertAlmostEqual(trainer.buffer.returns[2, 0].item(), 0.1 + 0.99 * 7, places=5)
        self.assertEqual(trainer.buffer.next_obs[2, 0].abs().sum().item(), 0)

    def test_termination_uses_next_peers_and_stored_behavior_selector(self):
        trainer = self.trainer()
        trainer.collect_rollout(trainer.env.reset()[0], rollout_steps=2)
        batch = next(trainer.buffer.get_sequence_batches(2, 80))
        batch["options"].zero_()
        batch["critic_options"].zero_()
        batch["next_critic_options"].fill_(1)
        batch["next_option_probs"][..., 0] = 0.25
        batch["next_option_probs"][..., 1] = 0.75

        def candidates(states, peer_choices, focal_ids, num_options, memory):
            self.assertTrue(bool((peer_choices == 1).all()))
            return torch.tensor([0.0, 10.0]).expand(states.shape[0], -1)

        with patch.object(trainer.option_critic, "focal_discrete_counterfactual_values", side_effect=candidates):
            losses = trainer._compute_sequence_losses(batch, trainer.current_eps, trainer.reference_actor)
        self.assertAlmostEqual(losses["mean_termination_advantage"].item(), -7.5, places=5)
        self.assertLess(losses["termination_loss"].item(), 0.0)

    def test_peer_marginal_can_reverse_termination_gradient(self):
        # Q(own, peer) rewards agreement; the peer reselects option 1 with p=.8.
        q = torch.tensor([[10.0, 0.0], [0.0, 10.0]])
        mu = torch.tensor([0.5, 0.5])
        peer_probs = torch.tensor([0.2, 0.8])
        advantages = q[0] - (mu[:, None] * q).sum(0)
        self.assertAlmostEqual(advantages[0].item(), 5.0)
        self.assertAlmostEqual((advantages * peer_probs).sum().item(), -3.0)
        beta_logit = torch.tensor(0.0, requires_grad=True)
        loss = NETWORKS.termination_objective(
            beta_logit.sigmoid(), (advantages * peer_probs).sum(), 0.0, torch.tensor(1.0),
        )
        loss.backward()
        self.assertLess(beta_logit.grad.item(), 0.0)  # Descent increases termination.

    def test_real_rollout_update_and_checkpoint_for_two_and_six_options(self):
        for options in (2, 6):
            with self.subTest(options=options):
                trainer = self.trainer(options)
                trainer.collect_rollout(trainer.env.reset()[0], rollout_steps=6)
                before = {name: value.clone() for name, value in trainer.actor.named_parameters()}
                metrics = trainer.update()
                self.assertTrue(all(bool(torch.isfinite(torch.as_tensor(value)).all()) for value in metrics.values()))
                self.assertTrue(any(not torch.equal(before[name], value) for name, value in trainer.actor.named_parameters()))
                path = Path(self.temp.name) / f"oc2_{options}.pt"
                trainer.save_checkpoint(path)
                checkpoint = torch.load(path, weights_only=False)
                self.assertEqual(checkpoint["training_checkpoint_version"], 7)
                self.assertEqual(checkpoint["return_bootstrap"], "sampled_joint_option_q")
                trainer.load_checkpoint(path)
                NETWORKS.LearnedOptionActor.from_checkpoint(checkpoint, "cpu")
                checkpoint["training_checkpoint_version"] = 6
                torch.save(checkpoint, path)
                with self.assertRaisesRegex(RuntimeError, "fresh training"):
                    trainer.load_checkpoint(path)

    def test_terminal_sensor_capture_is_opt_in(self):
        env = AutoResetEnv()
        env.episode_length_buf[:] = 10
        env._get_dones()
        self.assertIsNone(env.completed_terminal_policy_observations)
        env.capture_terminal_policy_observations = True
        env._get_dones()
        self.assertTrue(bool((env.completed_terminal_policy_observations == 10).all()))


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
