#!/usr/bin/env python3
"""Simulator-free regressions for OC3 sharing, updates, resume and transfer."""

from __future__ import annotations

import copy
import importlib
from pathlib import Path
import tempfile
import unittest

import torch

import validate_oc2_training as base
from _oc3_config import resolve
from train_oc3 import parser, validate_transfer


NET = importlib.import_module("swarmacb_oc2_validation.multi_mission_networks")
MULTI = importlib.import_module("swarmacb_oc2_validation.multi_mission_trainer")
ROOT = Path(__file__).resolve().parents[1]


class OC3Tests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(23)
        self.temp = tempfile.TemporaryDirectory(prefix="oc3_test_")
        self.addCleanup(self.temp.cleanup)
        self.created = 0

    def trainer(self):
        self.created += 1
        cfg = base.TRAINING.LearnedOptionCriticConfig(
            num_options=6, hidden_dim=32, option_hidden_dim=32,
            num_layers=1, option_num_layers=1, memory_size=16, option_memory_size=16,
            critic_hidden_dim=32, critic_num_heads=4, decision_period=1,
            reactive_intra_options=True, horizon=6, sequence_length=4,
            mini_batch_size=32, num_epochs=1, option_epsilon_schedule="constant",
            total_timesteps=1000, fused_optimizer=False, matmul_precision="highest",
            target_kl=0, log_dir=str(Path(self.temp.name) / str(self.created)),
        )
        trainer = base.TRAINING.LearnedOptionCriticTrainer(base.AutoResetEnv(2), cfg)
        self.addCleanup(trainer.writer.close)
        return trainer

    def learner(self, library=None, frozen=False):
        trainer = self.trainer()
        library = library or NET.SharedOptionLibrary(trainer.actor)
        return MULTI.MissionLearner(trainer, copy.deepcopy(library.state_dict()), frozen)

    def round(self, learners, library, optimizer):
        version = learners[0].version
        replies = [learner.collect(6, version) for learner in learners]
        for learner in learners:
            learner.start_epoch()
        for _ in range(min(reply["batches"] for reply in replies)):
            packets = [learner.gradients(version) for learner in learners]
            if optimizer:
                optimizer.zero_grad(set_to_none=True)
                NET.average_gradients(library, [p["gradients"] for p in packets])
                optimizer.step()
            version += 1
            for learner in learners:
                learner.commit(True, library.state_dict(), version)
        return [learner.finish() for learner in learners]

    def test_split_preserves_all_mini_outputs(self):
        trainer = self.trainer()
        actor = trainer.actor
        obs = torch.randn(3, 5, 24)
        state = tuple(torch.randn_like(s) for s in actor.initial_state(3, "cpu"))
        before = actor.forward_sequence(obs, state)
        library = NET.SharedOptionLibrary(actor)
        controller = NET.MissionOptionController(actor, library)
        after = actor.forward_sequence(obs, state)
        for index in range(6):
            torch.testing.assert_close(before[index], after[index], rtol=0, atol=0)
        shared = {id(p) for p in library.parameters()}
        private = {id(p) for p in controller.private_parameters()}
        self.assertFalse(shared & private)
        self.assertEqual(shared | private, {id(p) for p in actor.parameters()})

    def test_motor_contract_and_shared_identity_across_tasks(self):
        first, second = self.trainer(), self.trainer()
        library = NET.SharedOptionLibrary(first.actor)
        NET.MissionOptionController(first.actor, library)
        NET.MissionOptionController(second.actor, library)
        obs = torch.randn(3, 24)
        a = first.actor.step(obs)
        b = second.actor.step(obs, tuple(torch.randn_like(s) for s in second.actor.initial_state(3, "cpu")))
        for index in (3, 4, 5):
            torch.testing.assert_close(a[index], b[index], rtol=0, atol=0)
        self.assertGreater((a[1] - b[1]).abs().max().item(), 0)
        self.assertIs(first.actor.action_heads, second.actor.action_heads)
        self.assertIsNot(first.actor.option_lstm, second.actor.option_lstm)

    def test_average_includes_absent_mission_gradients_as_zero(self):
        library = NET.SharedOptionLibrary(self.trainer().actor)
        a = {k: torch.ones_like(p) for k, p in library.named_parameters()}
        b = {k: None for k in a}
        NET.average_gradients(library, [a, b])
        for parameter in library.parameters():
            torch.testing.assert_close(parameter.grad, torch.full_like(parameter, 0.5))
        with self.assertRaises(ValueError):
            NET.average_gradients(library, [{"bad": torch.ones(1)}])

    def test_two_missions_contribute_and_stay_synchronized(self):
        library = NET.SharedOptionLibrary(self.trainer().actor)
        learners = [self.learner(library), self.learner(library)]
        original = copy.deepcopy(library.state_dict())
        optimizer = torch.optim.Adam(library.parameters(), lr=3e-4)
        metrics = self.round(learners, library, optimizer)
        self.assertTrue(any(not torch.equal(original[k], v) for k, v in library.state_dict().items()))
        for learner, values in zip(learners, metrics):
            self.assertGreater(values["actor_gradient_norm"], 0)
            for k, v in library.state_dict().items():
                torch.testing.assert_close(learner.library.state_dict()[k], v, rtol=0, atol=0)
        self.assertIsNot(learners[0].trainer.buffer, learners[1].trainer.buffer)
        self.assertIsNot(learners[0].trainer.actor_memory_h, learners[1].trainer.actor_memory_h)

    def test_single_task_adam_matches_existing_oc2_update(self):
        direct = self.trainer()
        distributed = self.trainer()
        for name in ("actor", "team_critic", "action_critic", "option_critic"):
            getattr(distributed, name).load_state_dict(getattr(direct, name).state_dict())
        library = NET.SharedOptionLibrary(direct.actor)
        learner = MULTI.MissionLearner(distributed, library.state_dict())
        torch.manual_seed(47)
        direct.collect_rollout(direct.env.reset()[0], rollout_steps=6)
        torch.manual_seed(47)
        reply = learner.collect(6, 0)
        for key in ("obs", "actions", "returns", "next_options"):
            torch.testing.assert_close(getattr(direct.buffer, key)[:6], getattr(distributed.buffer, key)[:6])
        torch.manual_seed(61)
        direct.update()
        torch.manual_seed(61)
        learner.start_epoch()
        optimizer = torch.optim.Adam(library.parameters(), lr=distributed.cfg.actor_lr, eps=distributed.cfg.adam_eps)
        for version in range(reply["batches"]):
            packet = learner.gradients(version)
            optimizer.zero_grad(set_to_none=True)
            NET.average_gradients(library, [packet["gradients"]])
            optimizer.step()
            learner.commit(True, library.state_dict(), version + 1)
        learner.finish()
        for name in ("actor", "team_critic", "action_critic", "option_critic"):
            for key, expected in getattr(direct, name).state_dict().items():
                torch.testing.assert_close(getattr(distributed, name).state_dict()[key], expected, rtol=1e-5, atol=2e-7)

    def test_frozen_library_unchanged_but_manager_learns(self):
        library = NET.SharedOptionLibrary(self.trainer().actor).freeze()
        learner = self.learner(library, frozen=True)
        before = copy.deepcopy(library.state_dict())
        private = copy.deepcopy(learner.controller.private_state_dict())
        self.round([learner], library, None)
        for key, value in before.items():
            torch.testing.assert_close(learner.library.state_dict()[key], value, rtol=0, atol=0)
        self.assertTrue(any(not torch.equal(private[k], v) for k, v in learner.controller.private_state_dict().items()))
        self.assertTrue(all(p.grad is None for p in learner.library.parameters()))

    def test_resume_export_and_boundary_guards(self):
        library = NET.SharedOptionLibrary(self.trainer().actor)
        first = self.learner(library)
        self.round([first], library, torch.optim.Adam(library.parameters(), lr=3e-4))
        state = first.snapshot()
        second = self.learner(library)
        second.restore(state)
        self.assertEqual(second.version, first.version)
        self.assertEqual(second.trainer.global_step, first.trainer.global_step)
        self.assertEqual(second.trainer.actor_memory_h.abs().sum().item(), 0)
        for key, value in first.trainer.actor.state_dict().items():
            torch.testing.assert_close(second.trainer.actor.state_dict()[key], value, rtol=0, atol=0)
        path = Path(self.temp.name) / "export.pt"
        first.export_actor(path, "DirGate")
        checkpoint = torch.load(path, weights_only=False)
        loaded = base.NETWORKS.LearnedOptionActor.from_checkpoint(checkpoint, "cpu")
        obs = torch.randn(3, 24)
        for index in range(6):
            torch.testing.assert_close(loaded.step(obs)[index], first.trainer.actor.step(obs)[index])
        with self.assertRaisesRegex(RuntimeError, "playback only"):
            self.trainer().load_checkpoint(path)
        with self.assertRaises(RuntimeError):
            first.collect(6, first.version + 1)
        first.collect(6, first.version)
        with self.assertRaises(RuntimeError):
            first.snapshot()

    def test_campaign_config_and_transfer_guards(self):
        args = parser().parse_args(["--exclude", "DirGate", "--steps-per-mission", "400"])
        _, specs, manifest, _, _, _ = resolve(ROOT / "configs/OC3_cyclamen.yaml", args)
        self.assertEqual(len(specs), 4)
        self.assertNotIn("DirGate", [s["name"] for s in specs])
        self.assertTrue(all(s["trainer"]["total_timesteps"] == 400 for s in specs))
        self.assertEqual(manifest["num_envs"], 1)
        artifact = {"schema": 1, "kind": "oc3_library", "observation_contract": NET.OBSERVATION_CONTRACT,
                    "training_missions": [s["name"] for s in specs]}
        validate_transfer(artifact, ["DirGate"], NET.OBSERVATION_CONTRACT)
        with self.assertRaisesRegex(ValueError, "already used"):
            validate_transfer(artifact, ["Foraging"], NET.OBSERVATION_CONTRACT)
        with self.assertRaises(ValueError):
            resolve(ROOT / "configs/OC3_cyclamen.yaml", parser().parse_args(["--missions", "bad"]))

    def test_actor_stop_preserves_both_shared_and_private_parameters(self):
        learner = self.learner()
        learner.collect(6, 0)
        learner.start_epoch()
        before = copy.deepcopy(learner.trainer.actor.state_dict())
        critic_before = copy.deepcopy(learner.trainer.option_critic.state_dict())
        learner.gradients(0)
        learner.commit(False, learner.library.state_dict(), 0)
        for key, value in before.items():
            torch.testing.assert_close(learner.trainer.actor.state_dict()[key], value, rtol=0, atol=0)
        self.assertTrue(any(not torch.equal(critic_before[k], v)
                            for k, v in learner.trainer.option_critic.state_dict().items()))
        with self.assertRaisesRegex(RuntimeError, "incomplete"):
            learner.finish()

    def test_each_mission_reaches_shared_weights_and_private_optimizers_exclude_them(self):
        library = NET.SharedOptionLibrary(self.trainer().actor)
        for _ in range(2):
            learner = self.learner(library)
            learner.collect(6, 0)
            learner.start_epoch()
            packet = learner.gradients(0)
            self.assertGreater(sum(g.abs().sum().item() for g in packet["gradients"].values() if g is not None), 0)
            shared = {id(p) for p in learner.library.parameters()}
            private = {id(p) for group in learner.trainer.actor_optimizer.param_groups for p in group["params"]}
            self.assertFalse(shared & private)
            with self.assertRaisesRegex(RuntimeError, "uncommitted"):
                learner.gradients(0)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
