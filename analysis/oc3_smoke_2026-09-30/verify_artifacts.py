"""Check the saved OC3 execution tests; these are not performance evaluations."""

import json
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parent


def load(path):
    return torch.load(ROOT / path, map_location="cpu", weights_only=False)


def library_key(name):
    return name.split(".", 1)[0] in (
        "attention_encoder", "attention_head", "option_sensor_encoder", "action_heads", "log_std",
    )


results = {}
for folder, steps, missions in (
    ("checkpoints", 1280, 2), ("all_checkpoints", 8000, 5),
    ("cpu_checkpoints", 8000, 5), ("resume_checkpoints", 8000, 5),
    ("transfer_checkpoints", 8000, 1),
):
    bundle = load(f"{folder}/oc3_final.pt")
    assert bundle["progress"] == steps
    assert len(bundle["missions"]) == missions
    assert all(torch.isfinite(v).all() for v in bundle["library"].values())
    for name, state in bundle["missions"].items():
        assert state["step"] == steps and state["version"] == bundle["version"]
        assert not any(library_key(k) for k in state["controller"])
        export = load(f"{folder}/{name}/option_critic_2_final.pt")
        assert export["oc3_inference_export"]
        for key, value in bundle["library"].items():
            assert torch.equal(export["actor"][key], value)
    results[folder] = {
        "per_mission_decisions": steps, "aggregate_decisions": steps * missions,
        "rounds": bundle["round"], "version": bundle["version"],
        "frozen": bundle["frozen"], "all_exports_match_shared_library": True,
    }

original = load("checkpoints/option_library.pt")
transferred = load("transfer_checkpoints/option_library.pt")
assert original["training_missions"] == transferred["training_missions"]
assert all(torch.equal(value, transferred["library"][key]) for key, value in original["library"].items())
results["frozen_transfer_bitwise_unchanged"] = True

before = load("cpu_checkpoints/oc3_round_00000001.pt")
after = load("resume_checkpoints/oc3_final.pt")
assert before["progress"] == 4000 and after["progress"] == 8000
assert after["version"] > before["version"]
for key, state in before["library_optimizer"]["state"].items():
    assert after["library_optimizer"]["state"][key]["step"] > state["step"]
for mission, state in before["missions"].items():
    later = after["missions"][mission]
    for optimizer in ("actor_optimizer", "critic_optimizer"):
        for key, entry in state[optimizer]["state"].items():
            assert later[optimizer]["state"][key]["step"] > entry["step"]
results["resume_optimizer_steps_advanced"] = True
(ROOT / "verification.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
print(json.dumps(results, indent=2))
