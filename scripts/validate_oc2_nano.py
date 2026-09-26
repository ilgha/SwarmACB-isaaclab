#!/usr/bin/env python3
"""Simulator-free OC2-nano motor-capacity and workflow regressions."""

from pathlib import Path
import unittest
from unittest.mock import patch

import torch
import yaml

import validate_oc2_mini as mini_tests
import validate_oc2_training as training_tests


ROOT = mini_tests.ROOT
NETWORKS = mini_tests.NETWORKS
CONFIG = mini_tests.CONFIG


class NanoArchitectureTests(mini_tests.MiniArchitectureTests):
    """Retain mini's causality/attention tests with the affine motor path."""

    def actor(self, options=6):
        return NETWORKS.LearnedOptionActor(
            obs_dim=24, act_dim=2, num_options=options, hidden=32,
            num_layers=1, memory_size=32, option_hidden=32,
            option_num_layers=1, option_memory_size=32,
            reactive_intra_options=True, linear_intra_options=True,
        )

    def test_motor_gradients_cannot_reach_history_or_recurrent_weights(self):
        actor = self.actor()
        obs = torch.randn(2, 5, 24, requires_grad=True)
        state = tuple(s.requires_grad_() for s in actor.initial_state(2, "cpu"))
        actor.forward_sequence(obs, state)[3][:, -1].square().sum().backward()
        self.assertEqual(obs.grad[:, :-1].abs().sum().item(), 0)
        self.assertGreater(obs.grad[:, -1].abs().sum().item(), 0)
        self.assertTrue(all(s.grad is None for s in state))
        for layer in (actor.option_sensor_encoder, actor.option_lstm,
                      actor.option_output_encoder, actor.option_value_heads,
                      actor.termination_heads):
            self.assertTrue(all(p.grad is None for p in layer.parameters()))
        for layer in (actor.attention_encoder, actor.attention_head, actor.action_heads):
            self.assertGreater(sum(p.grad.abs().sum().item() for p in layer.parameters()), 0)

    def test_exact_affine_map_on_attended_sensors_and_parameter_count(self):
        actor = self.actor()
        obs = torch.randn(3, 5, 24)
        outputs = actor.forward_sequence(obs)
        attended = obs.unsqueeze(-2) * outputs[5]
        for option, head in enumerate(actor.action_heads):
            self.assertIsInstance(head, torch.nn.Linear)
            self.assertEqual((head.in_features, head.out_features), (24, 2))
            expected = attended[..., option, :] @ head.weight.T + head.bias
            torch.testing.assert_close(outputs[3][..., option, :], expected)
        # Six affine heads (48 weights + 2 biases) and 12 learned log sigmas.
        self.assertEqual(sum(p.numel() for p in actor.action_heads.parameters())
                         + actor.log_std.numel(), 312)

    def test_wheels_are_independent_of_high_level_hidden_encoder(self):
        actor = self.actor()
        obs = torch.randn(3, 24)
        first = actor.step(obs)
        with torch.no_grad():
            for p in actor.option_sensor_encoder.parameters():
                p.add_(torch.randn_like(p))
        second = actor.step(obs)
        torch.testing.assert_close(first[3], second[3], rtol=0, atol=0)
        self.assertGreater((first[1] - second[1]).abs().max().item(), 1e-6)

    def test_affine_identity_with_attention_held_fixed(self):
        actor = self.actor()
        x, y = torch.randn(2, 24), torch.randn(2, 24)
        with patch.object(actor.attention_head, "forward",
                          side_effect=lambda z: z.new_zeros((*z.shape[:-1], 6 * 24))):
            combined = actor.step(0.3 * x + 0.7 * y)[3]
            expected = 0.3 * actor.step(x)[3] + 0.7 * actor.step(y)[3]
        torch.testing.assert_close(combined, expected)

    def test_q_beta_and_attention_equal_mini_with_same_nonmotor_weights(self):
        nano, mini = self.actor(), mini_tests.MiniArchitectureTests.actor(self)
        shared = {k: v for k, v in mini.state_dict().items() if not k.startswith("action_heads.")}
        result = nano.load_state_dict(shared, strict=False)
        self.assertEqual(set(result.missing_keys),
                         {f"action_heads.{i}.{p}" for i in range(6) for p in ("weight", "bias")})
        self.assertFalse(result.unexpected_keys)
        obs = torch.randn(2, 4, 24)
        state = tuple(torch.randn_like(s) for s in nano.initial_state(2, "cpu"))
        first, second = nano.forward_sequence(obs, state), mini.forward_sequence(obs, state)
        for index in (0, 1, 2, 4, 5):
            torch.testing.assert_close(first[index], second[index], rtol=0, atol=0)
        for a, b in zip(first[6], second[6]):
            torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_five_configs_only_enable_mini_and_keep_benchmark_settings(self):
        for mission in ("DirGate", "XOR", "Homing", "Foraging", "Sheltering"):
            with self.subTest(mission=mission):
                mini_path = ROOT / "configs" / f"OC2-mini_{mission}_cyclamen.yaml"
                nano_path = ROOT / "configs" / f"OC2-nano_{mission}_cyclamen.yaml"
                mini = next(iter(yaml.safe_load(mini_path.read_text())["behaviors"].values()))
                nano = next(iter(yaml.safe_load(nano_path.read_text())["behaviors"].values()))
                self.assertIs(nano["network_settings"].pop("linear_intra_options"), True)
                self.assertEqual(nano, mini)
                name, variant, cfg, _ = CONFIG.load_config(str(nano_path))
                self.assertEqual(name, f"OC2-nano_{mission}_cyclamen")
                self.assertEqual(variant, "cyclamen")
                self.assertTrue(cfg.reactive_intra_options)
                self.assertTrue(cfg.linear_intra_options)
                self.assertEqual(cfg.num_options, 6)
                launcher = (ROOT / "scripts" / "hpc" / f"train_oc2_nano_{mission.lower()}.slurm").read_text()
                self.assertIn("#SBATCH --array=0-9", launcher)
                self.assertIn(f'"configs/{nano_path.name}"', launcher)
                self.assertIn(f'"OC2-nano_{mission}_cyclamen_v1"', launcher)
                self.assertIn('SCRIPT_DIR="$PROJECT_DIR/scripts/hpc"', launcher)

    def test_invalid_nano_settings_fail_early(self):
        cfg = training_tests.TRAINING.LearnedOptionCriticConfig()
        with self.assertRaisesRegex(ValueError, "YAML boolean"):
            CONFIG.apply_network_settings(cfg, {"linear_intra_options": "true"}, {}, "cyclamen", {})
        with self.assertRaisesRegex(ValueError, "requires reactive"):
            CONFIG.apply_network_settings(cfg, {"linear_intra_options": True}, {}, "cyclamen", {})
        with self.assertRaisesRegex(ValueError, "requires reactive"):
            NETWORKS.LearnedOptionActor(obs_dim=24, act_dim=2, num_options=6,
                                       linear_intra_options=True)


class NanoTrainerTests(training_tests.TrainerTests):
    """Exercise the existing counterfactual rollout/update suite for nano."""

    def trainer(self, options=2, agents=20):
        return super().trainer(options, agents, reactive=True, linear=True)

    def test_checkpoint_playback_resume_and_cross_variant_rejection(self):
        nano = self.trainer(options=6)
        nano.collect_rollout(nano.env.reset()[0], rollout_steps=6)
        before = nano.actor.action_heads[0].weight.detach().clone()
        nano.update()
        self.assertFalse(torch.equal(before, nano.actor.action_heads[0].weight))
        path = Path(self.temp.name) / "nano.pt"
        nano.save_checkpoint(path)
        checkpoint = torch.load(path, weights_only=False)
        self.assertEqual(checkpoint["learned_option_critic_version"], 6)
        self.assertEqual(checkpoint["training_checkpoint_version"], 7)
        self.assertTrue(checkpoint["linear_intra_options"])
        self.assertTrue(checkpoint["reactive_intra_options"])
        restored = NETWORKS.LearnedOptionActor.from_checkpoint(checkpoint, "cpu")
        obs = torch.randn(3, 24)
        state = tuple(torch.randn_like(s) for s in nano.actor.initial_state(3, "cpu"))
        for a, b in zip(restored.step(obs, state)[:6], nano.actor.step(obs, state)[:6]):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        resumed = self.trainer(options=6)
        resumed.load_checkpoint(path)
        resumed.collect_rollout(resumed.env.reset()[0], rollout_steps=6)
        resumed.update()
        self.assertTrue(all(torch.isfinite(p).all() for p in resumed.actor.parameters()))
        for reactive in (False, True):
            legacy = training_tests.TrainerTests.trainer(self, options=6, reactive=reactive)
            with self.assertRaisesRegex(RuntimeError, "different actor architectures"):
                legacy.load_checkpoint(path)
            legacy_path = Path(self.temp.name) / f"legacy_{reactive}.pt"
            legacy.save_checkpoint(legacy_path)
            with self.assertRaisesRegex(RuntimeError, "different actor architectures"):
                nano.load_checkpoint(legacy_path)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
