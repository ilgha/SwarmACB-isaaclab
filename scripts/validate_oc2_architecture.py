#!/usr/bin/env python3
"""Dependency-light checks for the OC2 attention and termination contracts."""

from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
import sys
import types

import torch


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_networks(root: Path):
    agents = (
        root
        / "source/SwarmACB_isaac/SwarmACB_isaac/tasks/direct/agents"
    )
    package_name = "swarmacb_oc2_validation"
    package = types.ModuleType(package_name)
    package.__path__ = [str(agents)]
    sys.modules[package_name] = package
    _load_module(f"{package_name}.poca_networks", agents / "poca_networks.py")
    return _load_module(
        f"{package_name}.learned_option_critic_networks",
        agents / "learned_option_critic_networks.py",
    )


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _attention_gradient(actor, obs: torch.Tensor, output_id: int) -> float:
    actor.zero_grad(set_to_none=True)
    outputs = actor.forward_sequence(obs)
    outputs[output_id].square().mean().backward()
    gradient = actor.attention_head.weight.grad
    _require(gradient is not None, "attention head received no gradient")
    return float(gradient.abs().sum())


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    networks = _load_networks(root)
    torch.manual_seed(7)

    actor = networks.LearnedOptionActor(
        obs_dim=24,
        act_dim=2,
        num_options=6,
        hidden=128,
        num_layers=1,
        memory_size=128,
        option_hidden=64,
        option_num_layers=2,
        option_memory_size=64,
        initial_termination_probability=0.05,
        initial_log_std=-0.7,
        min_log_std=-2.5,
        max_log_std=0.0,
    )
    obs = torch.randn(3, 5, 24)
    outputs = actor.forward_sequence(obs)
    option_values, termination_logits, means, stds, attentions, state = outputs

    _require(option_values.shape == (3, 5, 6), "invalid option-value shape")
    _require(
        termination_logits.shape == (3, 5, 6),
        "invalid termination-logit shape",
    )
    _require(means.shape == (3, 5, 6, 2), "invalid action-mean shape")
    _require(stds.shape == means.shape, "invalid action-std shape")
    _require(attentions.shape == (3, 5, 6, 24), "invalid attention shape")
    _require(
        state[0].shape == (1, 3, actor.hidden_size),
        "invalid packed recurrent state",
    )

    checkpoint = {
        "learned_option_critic_version": (
            networks.LEARNED_OPTION_CRITIC_VERSION
        ),
        "obs_dim": 24,
        "act_dim": 2,
        "num_options": 6,
        "hidden_dim": 128,
        "num_layers": 1,
        "memory_size": 128,
        "option_hidden_dim": 64,
        "option_num_layers": 2,
        "option_memory_size": 64,
        "initial_termination_probability": 0.05,
        "initial_log_std": -0.7,
        "min_log_std": -2.5,
        "max_log_std": 0.0,
        "option_value_temperature": 1.0,
        "actor": actor.state_dict(),
    }
    reloaded = networks.LearnedOptionActor.from_checkpoint(checkpoint, "cpu")
    with torch.no_grad():
        reloaded_outputs = reloaded.forward_sequence(obs)
    for output_id in range(5):
        _require(
            torch.allclose(reloaded_outputs[output_id], outputs[output_id]),
            f"checkpoint roundtrip changed actor output {output_id}",
        )

    step_state = actor.initial_state(3, obs.device)
    step_outputs = [[] for _ in range(5)]
    with torch.no_grad():
        for time_id in range(obs.shape[1]):
            current = actor.step(obs[:, time_id], step_state)
            step_state = current[5]
            for output_id in range(5):
                step_outputs[output_id].append(current[output_id])
    for output_id in range(5):
        stacked = torch.stack(step_outputs[output_id], dim=1)
        _require(
            torch.allclose(stacked, outputs[output_id], atol=1e-5),
            f"step/sequence mismatch for actor output {output_id}",
        )
    _require(
        torch.allclose(step_state[0], state[0], atol=1e-5)
        and torch.allclose(step_state[1], state[1], atol=1e-5),
        "step/sequence mismatch for packed recurrent state",
    )

    for output_id, label in (
        (0, "option value/selector"),
        (1, "termination"),
        (2, "action"),
    ):
        gradient = _attention_gradient(actor, obs, output_id)
        _require(gradient > 0.0, f"{label} bypasses learned attention")

    with torch.no_grad():
        actor.attention_head.weight.zero_()
        actor.attention_head.bias.fill_(12.0)
        open_outputs = actor.forward_sequence(obs)[0]
        actor.attention_head.bias.fill_(-12.0)
        closed_outputs = actor.forward_sequence(obs)[0]
    _require(
        not torch.allclose(open_outputs, closed_outputs, atol=1e-6),
        "option value/selector is invariant to its attention mask",
    )

    selected = torch.arange(3).view(3, 1).expand(3, 5) % 6
    action_dist = actor.selected_action_dist(means, stds, selected)
    actions = action_dist.sample()
    log_probs = action_dist.log_prob(actions)
    _require(bool((actions.abs() < 1.0).all()), "squashed actions left [-1, 1]")
    _require(bool(torch.isfinite(log_probs).all()), "non-finite action log probability")

    initial_actor = networks.LearnedOptionActor(
        24, 2, 6,
        initial_termination_probability=0.05,
    )
    with torch.no_grad():
        initial_beta = torch.sigmoid(
            initial_actor.forward_sequence(torch.zeros(2, 1, 24))[1]
        ).mean()
    _require(
        abs(float(initial_beta) - 0.05) < 1e-5,
        f"termination initialization is {float(initial_beta):.6f}, not 0.05",
    )

    good_logit = torch.tensor(0.0, requires_grad=True)
    good_loss = networks.termination_objective(
        good_logit.sigmoid(),
        torch.tensor(1.0),
        0.01,
        torch.tensor(1.0),
    )
    good_loss.backward()
    _require(
        float(good_logit.grad) > 0.0,
        "a useful option is not encouraged to continue",
    )
    bad_logit = torch.tensor(0.0, requires_grad=True)
    bad_loss = networks.termination_objective(
        bad_logit.sigmoid(),
        torch.tensor(-1.0),
        0.01,
        torch.tensor(1.0),
    )
    bad_loss.backward()
    _require(
        float(bad_logit.grad) < 0.0,
        "an inferior option is not encouraged to terminate",
    )

    # PPO must compare every recurrent minibatch with one immutable policy
    # snapshot taken at the beginning of the update.
    reference_actor = copy.deepcopy(actor).eval()
    reference_actor.requires_grad_(False)
    replay_obs = torch.randn(4, 17, 24)
    replay_state = actor.initial_state(4, replay_obs.device)
    replay_state = tuple(torch.randn_like(item) for item in replay_state)
    replay_options = torch.randint(0, 6, (4, 17))
    with torch.no_grad():
        current_replay = actor.forward_sequence(replay_obs, replay_state)
        reference_replay = reference_actor.forward_sequence(
            replay_obs,
            tuple(item.clone() for item in replay_state),
        )
        replay_actions = actor.selected_action_dist(
            current_replay[2], current_replay[3], replay_options
        ).sample()
        current_action_log_probs = actor.selected_action_dist(
            current_replay[2], current_replay[3], replay_options
        ).log_prob(replay_actions)
        reference_action_log_probs = reference_actor.selected_action_dist(
            reference_replay[2], reference_replay[3], replay_options
        ).log_prob(replay_actions)
        current_option_log_probs = actor.option_dist(
            current_replay[0]
        ).log_prob(replay_options)
        reference_option_log_probs = reference_actor.option_dist(
            reference_replay[0]
        ).log_prob(replay_options)
    _require(
        torch.allclose(current_action_log_probs, reference_action_log_probs),
        "frozen reference changed recurrent action log probabilities",
    )
    _require(
        torch.allclose(current_option_log_probs, reference_option_log_probs),
        "frozen reference changed recurrent option log probabilities",
    )
    reference_before = {
        name: parameter.detach().clone()
        for name, parameter in reference_actor.named_parameters()
    }
    with torch.no_grad():
        next(actor.parameters()).add_(0.01)
    _require(
        all(
            torch.equal(parameter, reference_before[name])
            for name, parameter in reference_actor.named_parameters()
        ),
        "frozen PPO reference shares mutable actor parameters",
    )

    print("OC2 architecture validation passed:")
    print("  option value, action, and termination backpropagate through attention")
    print("  recurrent memory is separated by option and packed for rollout storage")
    print("  tanh-squashed actions and corrected log probabilities are finite")
    print("  termination gradient has the Option-Critic continuation/switch signs")
    print("  frozen recurrent PPO reference is exact and immutable")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
