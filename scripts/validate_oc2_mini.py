#!/usr/bin/env python3
"""Simulator-free OC2-mini causality, config, and training regressions."""

from __future__ import annotations

import importlib
from pathlib import Path
import unittest
from unittest.mock import patch

import torch
import yaml

import validate_oc2_training as training_tests


ROOT = Path(__file__).resolve().parents[1]
NETWORKS = training_tests.NETWORKS
CONFIG = importlib.import_module("swarmacb_oc2_validation.config_loader")


class MiniArchitectureTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(37)

    def actor(self, options=6):
        return NETWORKS.LearnedOptionActor(
            obs_dim=24, act_dim=2, num_options=options, hidden=32,
            num_layers=1, memory_size=32, option_hidden=32,
            option_num_layers=1, option_memory_size=32,
            reactive_intra_options=True,
        )

    def test_motor_and_attention_are_invariant_to_all_recurrent_states(self):
        for options in (1, 2, 6):
            with self.subTest(options=options):
                actor = self.actor(options)
                obs = torch.randn(3, 24)
                zero_state = actor.initial_state(3, "cpu")
                other_state = tuple(torch.randn_like(s) for s in zero_state)
                first = actor.step(obs, zero_state)
                second = actor.step(obs, other_state)
                for index in (3, 4, 5):  # Means, scales, and attention masks.
                    torch.testing.assert_close(first[index], second[index], rtol=0, atol=0)
                for index in (1, 2):  # High-level Q and beta retain memory.
                    self.assertGreater((first[index] - second[index]).abs().max().item(), 1e-6)
                self.assertFalse(hasattr(actor, "manager_lstm"))
                self.assertEqual(actor.hidden_size, options * 16)

    def test_identical_current_sensors_override_different_histories_for_wheels(self):
        actor = self.actor()
        histories = torch.randn(2, 7, 24)
        histories[1, -1] = histories[0, -1]
        outputs = actor.forward_sequence(histories)
        for index in (3, 4, 5):
            torch.testing.assert_close(outputs[index][0, -1], outputs[index][1, -1])
        self.assertGreater((outputs[1][0, -1] - outputs[1][1, -1]).abs().max().item(), 1e-6)

    def test_motor_gradients_cannot_reach_history_or_recurrent_weights(self):
        actor = self.actor()
        obs = torch.randn(2, 5, 24, requires_grad=True)
        state = tuple(s.requires_grad_() for s in actor.initial_state(2, "cpu"))
        outputs = actor.forward_sequence(obs, state)
        outputs[3][:, -1].square().sum().backward()
        self.assertEqual(obs.grad[:, :-1].abs().sum().item(), 0)
        self.assertGreater(obs.grad[:, -1].abs().sum().item(), 0)
        self.assertTrue(all(s.grad is None for s in state))
        self.assertTrue(all(p.grad is None for p in actor.option_lstm.parameters()))
        self.assertTrue(all(p.grad is None for p in actor.option_output_encoder.parameters()))
        for layer in (actor.attention_encoder, actor.attention_head, actor.option_sensor_encoder):
            self.assertGreater(sum(p.grad.abs().sum().item() for p in layer.parameters()), 0)

    def test_high_level_outputs_still_learn_through_attention_and_memory(self):
        actor = self.actor()
        for index in (1, 2):
            actor.zero_grad(set_to_none=True)
            obs = torch.randn(2, 5, 24, requires_grad=True)
            state = tuple(s.requires_grad_() for s in actor.initial_state(2, "cpu"))
            actor.forward_sequence(obs, state)[index][:, -1].square().sum().backward()
            self.assertGreater(obs.grad[:, :-1].abs().sum().item(), 0)
            self.assertTrue(all(s.grad.abs().sum().item() > 0 for s in state))
            self.assertGreater(actor.attention_head.weight.grad.abs().sum().item(), 0)
            self.assertTrue(all(p.grad is not None for p in actor.option_lstm.parameters()))

    def test_closing_attention_blocks_sensor_shortcut_to_all_heads(self):
        actor = self.actor()
        obs = torch.randn(2, 4, 24)
        with patch.object(actor.attention_head, "forward", side_effect=lambda x: x.new_full((*x.shape[:-1], 6 * 24), -1000)):
            first = actor.forward_sequence(obs)
            second = actor.forward_sequence(obs * 10 + 7)
        for index in (1, 2, 3):
            torch.testing.assert_close(first[index], second[index], rtol=0, atol=0)

    def test_sequence_and_step_match_for_all_outputs_and_memory(self):
        actor = self.actor()
        obs = torch.randn(3, 6, 24)
        sequence = actor.forward_sequence(obs)
        state = None
        collected = [[] for _ in range(6)]
        for t in range(obs.shape[1]):
            step = actor.step(obs[:, t], state)
            state = step[6]
            for index in range(6):
                collected[index].append(step[index])
        for index in range(6):
            torch.testing.assert_close(sequence[index], torch.stack(collected[index], dim=1))
        for expected, actual in zip(sequence[6], state):
            torch.testing.assert_close(expected, actual)

    def test_five_configs_only_enable_mini_and_keep_benchmark_settings(self):
        for mission in ("DirGate", "XOR", "Homing", "Foraging", "Sheltering"):
            with self.subTest(mission=mission):
                baseline_path = ROOT / "configs" / f"OC2_{mission}_cyclamen.yaml"
                mini_path = ROOT / "configs" / f"OC2-mini_{mission}_cyclamen.yaml"
                baseline = next(iter(yaml.safe_load(baseline_path.read_text())["behaviors"].values()))
                mini = next(iter(yaml.safe_load(mini_path.read_text())["behaviors"].values()))
                self.assertIs(mini["network_settings"].pop("reactive_intra_options"), True)
                self.assertEqual(mini, baseline)
                name, variant, cfg, overrides = CONFIG.load_config(str(mini_path))
                self.assertEqual(name, f"OC2-mini_{mission}_cyclamen")
                self.assertEqual(variant, "cyclamen")
                self.assertTrue(cfg.reactive_intra_options)
                self.assertEqual(cfg.num_options, 6)
                self.assertEqual((cfg.option_hidden_dim, cfg.option_num_layers), (128, 1))
                self.assertEqual(overrides["task"], baseline["task"])
                full_cfg = CONFIG.load_config(str(baseline_path))[2]
                self.assertFalse(full_cfg.reactive_intra_options)

    def test_config_rejects_string_instead_of_boolean(self):
        cfg = training_tests.TRAINING.LearnedOptionCriticConfig()
        with self.assertRaisesRegex(ValueError, "YAML boolean"):
            CONFIG.apply_network_settings(cfg, {"reactive_intra_options": "false"}, {}, "cyclamen", {})


class MiniTrainerTests(training_tests.TrainerTests):
    """Run the complete OC2 return/termination trainer suite with mini as well."""

    def trainer(self, options=2, agents=20):
        return super().trainer(options, agents, reactive=True)

    def test_checkpoint_playback_roundtrip_and_cross_variant_resume_rejection(self):
        mini = self.trainer(options=6)
        mini.collect_rollout(mini.env.reset()[0], rollout_steps=6)
        mini.update()
        path = Path(self.temp.name) / "mini.pt"
        mini.save_checkpoint(path)
        checkpoint = torch.load(path, weights_only=False)
        self.assertEqual(checkpoint["learned_option_critic_version"], 5)
        self.assertTrue(checkpoint["reactive_intra_options"])
        restored = NETWORKS.LearnedOptionActor.from_checkpoint(checkpoint, "cpu")
        obs = torch.randn(3, 24)
        state = tuple(torch.randn_like(s) for s in mini.actor.initial_state(3, "cpu"))
        for actual, expected in zip(restored.step(obs, state)[:6], mini.actor.step(obs, state)[:6]):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        full = training_tests.TrainerTests.trainer(self, options=6)
        with self.assertRaisesRegex(RuntimeError, "different actor architectures"):
            full.load_checkpoint(path)
        full.save_checkpoint(path)
        with self.assertRaisesRegex(RuntimeError, "different actor architectures"):
            mini.load_checkpoint(path)
        # Full-v4 files saved before mini existed have no mode metadata.
        checkpoint = torch.load(path, weights_only=False)
        checkpoint.pop("reactive_intra_options")
        torch.save(checkpoint, path)
        full.load_checkpoint(path)
        restored = NETWORKS.LearnedOptionActor.from_checkpoint(checkpoint, "cpu")
        self.assertFalse(restored.reactive_intra_options)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
