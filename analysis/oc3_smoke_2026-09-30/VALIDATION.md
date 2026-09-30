# OC3 Implementation Validation

Date: 30 September 2026. No long training campaign or cluster job was launched.
These checks establish execution and implementation contracts, not skill
diversity, useful transfer or benchmark performance.

## Regression Tests

59 tests passed across `validate_oc3.py`, `validate_oc2_training.py`,
`validate_oc2_mini.py` and `validate_oc2_nano.py`.

The ten OC3 tests cover:

- Exact preservation of mini forward outputs after the network partition.
- Shared motor identity and independence from mission-specific recurrent state.
- Averaging shared gradients, including unused-parameter zero contributions.
- Contributions from both missions and synchronized library weights.
- One-mission optimizer agreement with the existing OC2 update.
- Bitwise frozen motor/attention parameters while the manager learns.
- Checkpoint restoration, inference exports and stale-version guards.
- Configuration validation and rejection of seen-mission transfer.
- Stopping both shared and private actor updates while critics continue.
- Exclusion of shared parameters from task-specific optimizers.

Python syntax checks, Bash syntax checks for both affected HPC files and
`git diff --check` also passed. Slurm/Apptainer execution was not tested here.

## Real Isaac Tests

All tests used 20 robots per environment, one environment per mission, and the
existing 0.1-second simulation / five-substep decision period.

| Test | Per-mission decisions | Result |
|---|---:|---|
| DirGate + Foraging, GPU | 1,280 | Two shared updates; checkpoint and exports |
| All five missions, GPU | 8,000 | Two rounds, episode resets, checkpoint and exports |
| All five missions, CPU | 8,000 | Same workflow with all networks/env tensors on CPU |
| Frozen transfer to XOR, CPU | 8,000 | New task networks trained; library unchanged |
| Resume all-five CPU checkpoint | 4,000 to 8,000 | Counters and optimizer steps advanced |

The transfer test used the tiny DirGate/Foraging library strictly to exercise the
held-out loading path. It is not evidence that a trained repertoire transfers.
`verify_artifacts.py` checks all saved mission exports against the common library,
verifies every frozen library tensor is unchanged, and verifies shared/private
optimizer step counters advanced after resume. Results are in `verification.json`.

Resume deliberately starts fresh simulator episodes and recurrent states; it
does not reproduce the interrupted physical trajectory. The resume test wrote
to separate directories to preserve the original smoke artifacts.

## Local Resource Observations

On the local RTX 5080, five GPU workers approached the 16 GB VRAM limit
(15,542 MiB observed). The all-five GPU smoke took about 427 seconds including
startup, versus 120 seconds in CPU mode. These tiny runs are not a representative
long-run throughput benchmark. CPU mode is a practical lower-VRAM local check;
the cluster's 48 GB GPUs should still be profiled with a pilot.

Early smoke logs used `scores` for unscaled group returns. The final implementation
labels episode returns explicitly and preserves OC2's TensorBoard reward tags.
Time-accumulated occupancy rewards must not be interpreted as final robot counts.

## Artifacts

- `checkpoints/`, `runs/`: two-mission GPU smoke.
- `all_checkpoints/`, `all_runs/`: five-mission GPU smoke.
- `cpu_checkpoints/`, `cpu_runs/`: five-mission CPU smoke.
- `transfer_checkpoints/`, `transfer_runs/`: frozen XOR transfer smoke.
- `resume_checkpoints/`, `resume_runs/`: continued all-five CPU smoke.
- `all_missions.yaml`: reduced-budget config for full-mission checks.

The implementation does not add predefined behavior labels, forced balanced
option usage, mission IDs to the motor policies, or a new intrinsic reward.
Long-run learning quality and frozen-library generalization remain to be tested.
