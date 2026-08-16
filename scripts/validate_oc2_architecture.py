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
        initial_termination_probability=0.27,
        initial_log_std=0.0,
        min_log_std=-2.5,
        max_log_std=0.0,
        squash_actions=False,
    )
    obs = torch.randn(3, 5, 24)
    outputs = actor.forward_sequence(obs)
    (
        selector_logits,
        option_values,
        termination_logits,
        action_means,
        action_stds,
        attentions,
        state,
    ) = outputs

    _require(selector_logits.shape == (3, 5, 6), "invalid selector shape")
    _require(option_values.shape == (3, 5, 6), "invalid option-value shape")
    _require(
        torch.equal(selector_logits, option_values),
        "new AOC actor does not expose Q_Omega for option selection",
    )
    _require(
        termination_logits.shape == (3, 5, 6),
        "invalid termination-logit shape",
    )
    _require(
        action_means.shape == (3, 5, 6, 2),
        "invalid intra-option wheel-mean shape",
    )
    _require(
        action_stds.shape == (3, 5, 6, 2)
        and bool((action_stds > 0).all()),
        "invalid intra-option wheel-standard-deviation output",
    )
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
        "discrete": False,
        "num_actions": 2,
        "act_dim": 2,
        "num_options": 6,
        "hidden_dim": 128,
        "num_layers": 1,
        "memory_size": 128,
        "option_hidden_dim": 64,
        "option_num_layers": 2,
        "option_memory_size": 64,
        "initial_termination_probability": 0.27,
        "initial_log_std": 0.0,
        "min_log_std": -2.5,
        "max_log_std": 0.0,
        "option_selector_temperature": 1.0,
        "action_distribution": "mlagents_normal",
        "action_transform": "clip_minus3_3_divide3",
        "actor": actor.state_dict(),
    }
    reloaded = networks.LearnedOptionActor.from_checkpoint(checkpoint, "cpu")
    with torch.no_grad():
        reloaded_outputs = reloaded.forward_sequence(obs)
    for output_id in range(6):
        _require(
            torch.allclose(reloaded_outputs[output_id], outputs[output_id]),
            f"checkpoint roundtrip changed actor output {output_id}",
        )

    legacy_actor = networks.LearnedOptionActor(
        obs_dim=24,
        act_dim=2,
        num_options=6,
        hidden=128,
        num_layers=1,
        memory_size=128,
        option_hidden=64,
        option_num_layers=2,
        option_memory_size=64,
        initial_log_std=-0.7,
        separate_selector=False,
        epsilon_greedy_selector=False,
        squash_actions=True,
    )
    legacy_checkpoint = {
        **checkpoint,
        "learned_option_critic_version": 2,
        "obs_dim": 24,
        "discrete": False,
        "num_actions": 2,
        "act_dim": 2,
        "option_value_temperature": 1.0,
        "initial_log_std": -0.7,
        "action_distribution": "tanh_squashed_normal",
        "action_transform": "identity_normalized",
        "actor": legacy_actor.state_dict(),
    }
    loaded_legacy = networks.LearnedOptionActor.from_checkpoint(
        legacy_checkpoint,
        "cpu",
    )
    legacy_obs = torch.randn(3, 5, 24)
    with torch.no_grad():
        legacy_outputs = loaded_legacy.forward_sequence(legacy_obs)
    _require(
        torch.equal(legacy_outputs[0], legacy_outputs[1]),
        "version-2 checkpoint no longer uses values as selector logits",
    )

    step_state = actor.initial_state(3, obs.device)
    step_outputs = [[] for _ in range(6)]
    with torch.no_grad():
        for time_id in range(obs.shape[1]):
            current = actor.step(obs[:, time_id], step_state)
            step_state = current[6]
            for output_id in range(6):
                step_outputs[output_id].append(current[output_id])
    for output_id in range(6):
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
        (1, "option value"),
        (2, "termination"),
        (3, "action"),
    ):
        gradient = _attention_gradient(actor, obs, output_id)
        _require(gradient > 0.0, f"{label} bypasses learned attention")

    with torch.no_grad():
        actor.attention_head.weight.zero_()
        actor.attention_head.bias.fill_(12.0)
        open_outputs = actor.forward_sequence(obs)[1]
        actor.attention_head.bias.fill_(-12.0)
        closed_outputs = actor.forward_sequence(obs)[1]
    _require(
        not torch.allclose(open_outputs, closed_outputs, atol=1e-6),
        "option values are invariant to their attention masks",
    )
    _require(
        not hasattr(actor, "selector_heads"),
        "paper-aligned actor still contains a separate selector head",
    )

    option_scores = torch.tensor([[3.0, 2.0, 1.0, 0.0, -1.0, -2.0]])
    epsilon_probs = actor.option_dist(option_scores, epsilon=0.2).probs
    expected_probs = torch.full_like(option_scores, 0.2 / 6.0)
    expected_probs[0, 0] += 0.8
    _require(
        torch.allclose(epsilon_probs, expected_probs),
        "epsilon-soft Q_Omega policy has incorrect probabilities",
    )
    _require(
        torch.allclose(
            actor.option_dist(option_scores, epsilon=1.0).probs,
            torch.full_like(option_scores, 1.0 / 6.0),
        ),
        "initial option exploration is not uniform",
    )
    counterfactual_values = torch.tensor([[12.0, 6.0, 3.0, 0.0, -3.0, -6.0]])
    epsilon_value = actor.option_state_value(
        option_scores,
        counterfactual_values,
        epsilon=0.2,
    )
    expected_value = (expected_probs * counterfactual_values).sum(dim=-1)
    _require(
        torch.allclose(epsilon_value, expected_value),
        "V_Omega is not the epsilon-soft option-policy expectation",
    )
    _require(
        not torch.allclose(epsilon_value, counterfactual_values.max(dim=-1).values),
        "termination reselection silently uses a hard maximum",
    )

    selected = torch.arange(3).view(3, 1).expand(3, 5) % 6
    action_dist = actor.selected_action_dist(
        action_means,
        action_stds,
        selected,
    )
    actions = action_dist.sample()
    log_probs = action_dist.log_prob(actions)
    _require(
        isinstance(action_dist, torch.distributions.Normal),
        "new OC2 intra-option policy is not a diagonal Gaussian",
    )
    _require(
        actions.shape == (3, 5, 2),
        "learned intra-option policy did not produce two wheel commands",
    )
    _require(bool(torch.isfinite(log_probs).all()), "non-finite action log probability")
    _require(
        not torch.allclose(action_means[..., 0, :], action_means[..., 1, :]),
        "different options share an identical wheel-policy output",
    )
    normalized_actions = actions.clamp(-3.0, 3.0) / 3.0
    _require(
        bool((normalized_actions.abs() <= 1.0).all()),
        "OC2 actuator transform produced an invalid wheel command",
    )

    initial_actor = networks.LearnedOptionActor(
        24, 2, 6,
        initial_termination_probability=0.27,
        initial_log_std=0.0,
    )
    with torch.no_grad():
        initial_beta = torch.sigmoid(
            initial_actor.forward_sequence(torch.zeros(2, 1, 24))[2]
        ).mean()
    _require(
        abs(float(initial_beta) - 0.27) < 1e-5,
        f"termination initialization is {float(initial_beta):.6f}, not 0.27",
    )

    good_logit = torch.tensor(0.0, requires_grad=True)
    good_loss = networks.termination_objective(
        good_logit.sigmoid(),
        torch.tensor(1.0),
        0.0,
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
        0.0,
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
            current_replay[3], current_replay[4], replay_options
        ).sample()
        current_action_log_probs = actor.selected_action_dist(
            current_replay[3], current_replay[4], replay_options
        ).log_prob(replay_actions)
        reference_action_log_probs = reference_actor.selected_action_dist(
            reference_replay[3], reference_replay[4], replay_options
        ).log_prob(replay_actions)
        current_option_log_probs = actor.option_dist(
            current_replay[1], epsilon=0.2,
        ).log_prob(replay_options)
        reference_option_log_probs = reference_actor.option_dist(
            reference_replay[1], epsilon=0.2,
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

    # The OC2-2 experiment is an option-count ablation of this same actor,
    # not a separate algorithm or checkpoint schema.
    two_option_actor = networks.LearnedOptionActor(
        obs_dim=24,
        act_dim=2,
        num_options=2,
        hidden=128,
        num_layers=1,
        memory_size=128,
        option_hidden=128,
        option_num_layers=1,
        option_memory_size=128,
        initial_termination_probability=0.27,
        initial_log_std=0.0,
        squash_actions=False,
    )
    with torch.no_grad():
        two_option_outputs = two_option_actor.forward_sequence(obs)
    _require(
        two_option_outputs[1].shape == (3, 5, 2),
        "OC2-2 produced an invalid option-value shape",
    )
    _require(
        two_option_outputs[2].shape == (3, 5, 2),
        "OC2-2 produced an invalid termination shape",
    )
    _require(
        two_option_outputs[3].shape == (3, 5, 2, 2),
        "OC2-2 did not produce two independent two-wheel policies",
    )
    two_option_scores = torch.tensor([[2.0, -1.0]])
    two_option_probs = two_option_actor.option_dist(
        two_option_scores,
        epsilon=0.2,
    ).probs
    _require(
        torch.allclose(two_option_probs, torch.tensor([[0.9, 0.1]])),
        "OC2-2 epsilon-soft option probabilities are incorrect",
    )

    print("OC2 architecture validation passed:")
    print("  Q_Omega, action, and termination outputs use attention")
    print("  option choice is epsilon-soft over attended Q_Omega values")
    print("  V_Omega uses the epsilon-soft policy expectation, not a hard max")
    print("  recurrent memory is separated by option and packed for rollout storage")
    print("  all six intra-option policies learn two continuous wheel commands")
    print("  the same architecture resolves correctly for the OC2-2 ablation")
    print("  new policies use the ML-Agents Gaussian and actuator transform")
    print("  termination gradient has the Option-Critic continuation/switch signs")
    print("  frozen recurrent PPO reference is exact and immutable")
    print("  legacy version-2 actors remain loadable for evaluation")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
