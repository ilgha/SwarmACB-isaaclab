# SwarmACB Isaac Lab

Multi-agent swarm robotics reinforcement learning built on
[Isaac Lab](https://isaac-sim.github.io/IsaacLab/).

This repository reimplements the SwarmACB/CASA swarm missions in Isaac Lab. It
uses 20 simulated e-puck robots in a shared dodecagonal arena and trains the
collective behavior with MA-POCA (Multi-Agent POsthumous Credit Assignment).

## Implemented Missions

All five benchmark missions are available as Gymnasium tasks:

| Mission | Task ID | Duration | Light | Reward / performance |
|---|---|---:|---|---|
| XOR aggregation | `SwarmACB-XOR-v0` | 180 s | No | Number of robots in the most occupied black target |
| Homing | `SwarmACB-Homing-v0` | 120 s | No | Final number of robots inside the black goal |
| Foraging | `SwarmACB-Foraging-v0` | 180 s | Yes | Food-to-nest trips completed by the swarm |
| Sheltering / SCA | `SwarmACB-Sheltering-v0` | 180 s | Yes | Number of robots inside the central white shelter |
| Directional Gate | `SwarmACB-DirectionalGate-v0` | 120 s | Yes | Correct minus incorrect gate crossings |

Sheltering aliases are also registered:

```text
SwarmACB-SCA-v0
SwarmACB-SHL-v0
```

## Architecture

```text
SwarmACB_isaac/
  configs/
    DirGate_*.yaml              # Directional Gate configs for all CASA variants
    XOR_cyclamen.yaml           # Cyclamen config for XOR
    Homing_cyclamen.yaml        # Cyclamen config for Homing
    Foraging_cyclamen.yaml      # Cyclamen config for Foraging
    Sheltering_cyclamen.yaml    # Cyclamen config for Sheltering/SCA
  scripts/
    train.py                    # Training entry point
    play.py                     # Evaluation / replay from checkpoint
    manual_control.py           # Pygame manual control, no Isaac Sim
    manual_control_isaac.py     # Isaac Sim viewport manual control / fast viewer
    hpc/                        # Cluster helper scripts
  source/SwarmACB_isaac/SwarmACB_isaac/tasks/direct/
    agents/                     # POCA and both Option-Critic phases
    epuck/                      # E-puck sensors and behavior modules
    missions/
      directional_gate/
      xor_aggregation/
      homing/
      foraging/
      sheltering/
```

## CASA Variants

The trainer supports the five CASA variants:

| Variant | Observation | Action | Notes |
|---|---:|---|---|
| `dandelion` | 24 | Continuous wheel velocities | Full sensor vector, direct motor control |
| `daisy` | 24 | Discrete module ID | Full sensors, behavior module selection |
| `lily` | 4 | Discrete module ID | Ground sensors plus neighborhood density |
| `tulip` | 4 | Discrete module ID | Lily observation with a smaller network |
| `cyclamen` | 4 | Discrete module ID | Lily observation with LSTM memory |

The six discrete behavior modules are:

```text
0 Stop, 1 Exploration, 2 Attraction, 3 Repulsion, 4 Phototaxis, 5 Anti-phototaxis
```

## Fixed-Option Option-Critic

Phase 1 of the Option-Critic implementation starts from `cyclamen`, the final
SwarmACB controller form: 4 local observation values, a recurrent memory, and
the six predefined ACB behavior modules.

In this mode, the behavior modules are fixed options. The learner does not learn
new intra-option motor policies yet. It learns:

- a shared recurrent local policy over the six fixed options
- a shared local per-option termination model
- a centralized permutation-invariant team-value critic, `V(s)`
- a centralized permutation-invariant collective option critic,
  `Q_Omega(s, omega_vector)`
- a centralized POCA-style counterfactual baseline for each robot

Execution is still decentralized: each robot keeps its current module until the
learned termination model switches it to another module. This isolates the value
of temporal abstraction before adding learned intra-option policies and
attention/diversity in a later phase. The centralized critics are used only
during training and are not part of the deployed robot controller.

The implementation keeps the SwarmACB counterfactual philosophy: when evaluating
robot `i`, all peers keep their active modules fixed. The selector PPO advantage
uses the learned POCA-style approximation:

```text
A_i(s, omega_vector) = lambda_return(s) - b_i(s, omega_-i)
```

The termination update follows the Option-Critic arrival-state theorem. For the
option vector executed at time `t`, the critic evaluates each of robot `i`'s six
replacement options at `s_t+1` while the other robots' options stay fixed:

```text
V_i(s_t+1, omega_-i) =
  sum_omega_i pi_O(omega_i | h_i,t+1)
    Q_Omega(s_t+1, (omega_-i, omega_i))

A_i^term(s_t+1, omega_vector) =
  Q_Omega(s_t+1, omega_vector) - V_i(s_t+1, omega_-i)
```

The termination loss uses `A_i^term + xi`, where `xi` is configured as
`termination_penalty`. A small positive value encourages temporally extended
options. The six-way replacement calculation is linear in robots and options;
it does not enumerate the collective joint-option space. Option selection
remains a recurrent PPO policy trained at option boundaries, which is an
SMDP-level policy-over-options implementation.

## Learned Option-Critic Phase 2

Phase 2 is configured as `trainer_type: learned_option_critic` and currently
uses the working name `OC2_cyclamen`. The six options are no longer mapped to
the six predefined behavior modules. Each option learns its own stochastic
continuous intra-option policy over the two normalized wheel commands.
The number of learned options is controlled by
`network_settings.num_options`; the initial benchmark uses six for a direct
comparison with Phase 1.

The local actor contains:

- Cyclamen's recurrent policy over six options
- six learned termination functions
- six learned Gaussian intra-option wheel policies
- six state-dependent feature-attention masks

The recurrent selector receives exactly Cyclamen's four inputs
`[ground_0, ground_1, ground_2, ztilde]`. Learned motor options also need
proximity, light, and directional RAB information that was previously available
inside the fixed behavior modules. They therefore attend over the full local
24-channel sensor vector. This keeps the high-level Cyclamen comparison while
giving learned motor options enough local information to avoid obstacles,
follow light, and react directionally to neighbors.

Training remains centralized and execution decentralized. Three critic roles
separate the objectives:

```text
V(s)                                  team value for lambda returns
b_i^U((s, omega), a_-i)              primitive-action counterfactual baseline
Q_Omega(s, omega_vector),
b_i^Omega(s, omega_-i)               collective option value and baseline
```

The intra-option PPO advantage omits only robot `i`'s wheel action; its active
option and every peer option/action remain fixed. The selector advantage omits
only robot `i`'s option:

```text
A_i^U     = lambda_return - b_i^U((s, omega), a_-i)
A_i^Omega = lambda_return - b_i^Omega(s, omega_-i)
```

The arrival-state termination theorem is the same collective replacement test
as Phase 1. The intra-option policy is updated every decision, while the policy
over options is updated only at sampled option boundaries. Pairwise attention
cosine similarity is minimized to discourage option collapse, and a temporal
attention loss discourages rapidly changing option focus.

## Sensor Suite

Each e-puck has:

- Differential-drive kinematics matching `Epuck.cs`: wheelbase 0.055 m and max wheel speed 0.16 m/s.
- 8 IR proximity sensors, range 0.10 m
- 8 directional light sensors using the Unity mission light intensity (1000), disabled in XOR and Homing
- 3 ground sensors returning grey, white, or black
- Range-and-bearing neighborhood sensing, range 0.60 m with Unity-style line-of-sight and packet loss

For GUI inspection, add these keys under a config `environment:` block:

```yaml
debug_visual_sensors: true
sensor_visual_robot_index: -1   # -1 shows all env-0 robots
sensor_visual_rab_ring_segments: 48
```

Then run `scripts/play.py` with `--exact-env` and without `--headless`. The live overlay draws proximity rays, light rays, RAB range rings, visible RAB neighbor links, and three ground-channel dots colored black, grey, or white. For the lightweight viewer, pass `--show-sensors`; its overlay is sampled at 10 Hz by default while robot animation remains at 60 Hz.

For manual IsaacSim inspection, run:

```bash
python scripts/manual_control_isaac.py --task SwarmACB-DirectionalGate-v0 --show-sensors --sensor-robot 0
```

Use `--sensor-robot -1` to draw the overlay for every robot.

## Installation

1. Install Isaac Lab following the
   [official guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html).

2. Clone this repository:

   ```bash
   git clone https://github.com/ilgha/SwarmACB-isaaclab.git
   cd SwarmACB-isaaclab
   ```

3. Install the extension in editable mode:

   ```bash
   python -m pip install -e source/SwarmACB_isaac
   ```

On Windows/RTX 50-series machines, the scripts automatically add the Isaac Sim
Kit arguments that avoid the Vulkan startup crash:

```text
--/app/vulkan=false --/crashreporter/preserveDump=true
```

## Training

Use the YAML configs:

```bash
python scripts/train.py --config configs/DirGate_cyclamen.yaml --headless
python scripts/train.py --config configs/XOR_cyclamen.yaml --headless
python scripts/train.py --config configs/Homing_cyclamen.yaml --headless
python scripts/train.py --config configs/Foraging_cyclamen.yaml --headless
python scripts/train.py --config configs/Sheltering_cyclamen.yaml --headless
```

Train the fixed-option Option-Critic phase-1 controller:

```bash
python scripts/train.py --config configs/OC_DirGate_cyclamen.yaml --headless
python scripts/train.py --config configs/OC_XOR_cyclamen.yaml --headless
python scripts/train.py --config configs/OC_Homing_cyclamen.yaml --headless
python scripts/train.py --config configs/OC_Foraging_cyclamen.yaml --headless
python scripts/train.py --config configs/OC_Sheltering_cyclamen.yaml --headless
```

Train the learned-option Option-Critic phase-2 controller:

```bash
python scripts/train.py --config configs/OC2_DirGate_cyclamen.yaml --headless
python scripts/train.py --config configs/OC2_XOR_cyclamen.yaml --headless
python scripts/train.py --config configs/OC2_Homing_cyclamen.yaml --headless
python scripts/train.py --config configs/OC2_Foraging_cyclamen.yaml --headless
python scripts/train.py --config configs/OC2_Sheltering_cyclamen.yaml --headless
```

Submit ten independent Phase 2 designs per mission on the cluster with
`scripts/hpc/train_oc2_<mission>.slurm`, for example:

```bash
sbatch scripts/hpc/train_oc2_dirgate.slurm
```

Or override the task and variant from the command line:

```bash
python scripts/train.py --task SwarmACB-Foraging-v0 --variant tulip --headless
python scripts/train.py --task SwarmACB-SCA-v0 --variant cyclamen --num_envs 5 --headless
```

Useful smoke test:

```bash
python scripts/train.py --config configs/Sheltering_cyclamen.yaml --headless --total_timesteps 2000 --num_envs 1 --log_dir runs/Sheltering_smoke --checkpoint_dir checkpoints/Sheltering_smoke
python scripts/train.py --config configs/OC_Sheltering_cyclamen.yaml --headless --total_timesteps 2000 --num_envs 1 --log_dir runs/OC_Sheltering_smoke --checkpoint_dir checkpoints/OC_Sheltering_smoke
python scripts/train.py --config configs/OC2_Sheltering_cyclamen.yaml --headless --total_timesteps 2000 --num_envs 1 --log_dir runs/OC2_Sheltering_smoke --checkpoint_dir checkpoints/OC2_Sheltering_smoke
```

## Evaluation

Evaluate the full IsaacLab environment exactly:

```bash
python scripts/play.py --config configs/DirGate_cyclamen.yaml --checkpoint checkpoints/DirGate_cyclamen/poca_final.pt --exact-env --num_envs 1 --num_episodes 10 --deterministic
python scripts/play.py --config configs/OC_DirGate_cyclamen.yaml --checkpoint checkpoints/OC_DirGate_cyclamen/option_critic_final.pt --exact-env --num_envs 1 --num_episodes 10
python scripts/play.py --config configs/OC2_DirGate_cyclamen.yaml --checkpoint checkpoints/OC2_DirGate_cyclamen/option_critic_2_final.pt --exact-env --num_envs 1 --num_episodes 10
```

Use stochastic playback for the primary Option-Critic evaluation. Its learned
termination probability is Bernoulli-sampled during training. Passing
`--deterministic` thresholds termination at `0.5`, which is useful only as a
diagnostic.

GUI playback uses the lightweight Isaac viewport by default. It reproduces the
mission kinematics and sensors without advancing an unused PhysX scene, batches
policy inference across all robots, and updates the visible swarm through one
USD point instancer:

```bash
python scripts/play.py --config configs/Sheltering_cyclamen.yaml --checkpoint checkpoints/Sheltering_cyclamen/poca_final.pt --deterministic
python scripts/play.py --config configs/Sheltering_cyclamen.yaml --checkpoint checkpoints/Sheltering_cyclamen/poca_final.pt --show-sensors --sensor-robot 0 --deterministic
```

`--fast-viewer` remains as a compatibility alias. Use `--exact-env` whenever
the full IsaacLab physics/environment implementation itself is under test.

Compare behavior-module usage for the 10 classical Cyclamen controllers and
the 10 fixed-option OC-Cyclamen controllers in headless mode:

```bash
python scripts/evaluate_behavior_time.py --mission dirgate --checkpoint-root checkpoints
python scripts/evaluate_behavior_time.py --mission xor --checkpoint-root checkpoints
python scripts/evaluate_behavior_time.py --mission homing --checkpoint-root checkpoints
python scripts/evaluate_behavior_time.py --mission foraging --checkpoint-root checkpoints
python scripts/evaluate_behavior_time.py --mission sheltering --checkpoint-root checkpoints
```

This writes CSV summaries and plots under `analysis/<mission>_behavior_time/`.
If `poca_final.pt` or `option_critic_final.pt` is missing for a run, the script
falls back to the latest numbered checkpoint in that run directory.
For speed, all available controllers for a method are evaluated concurrently in
one vectorized IsaacLab environment by default. The script reuses that single
IsaacLab environment for Cyclamen and OC-Cyclamen to avoid slow or fragile
stage teardown/recreation between methods. Use `--batch-size N` to limit
concurrency, or `--sequential` for one checkpoint at a time while still reusing
the same IsaacLab environment.
Use `--deterministic` for argmax actions and thresholded OC terminations;
the default is stochastic playback, matching `scripts/play.py`.

GUI playback keeps normal viewport fidelity by default: native resolution,
scene materials, a 60 Hz swarm animation, and the terminal/editor status HUD.
The lightweight viewer evaluates policies and sensor-dependent modules at the
Unity-matching control frequency, normally 10 Hz, while retaining the last
command between decisions. Sensor debug geometry is also refreshed at 10 Hz
instead of rebuilding it every rendered frame.

The default `--gui-performance-preset same` disables VSync/rate-limit sleeps and
RTX eco mode without changing materials or resolution. The lightweight viewer
also enables Isaac's asynchronous render thread. Use
`--gui-async-rendering off` if a driver or Isaac Sim version shows stale USD
updates, `--gui-performance-preset off` for entirely stock Kit behavior, or
`--gui-performance-preset fast` for rendering-side quality/performance tweaks.

Playback is paced to real time by default. `--playback-speed 2` runs at twice
real time when the machine can sustain it, while `--playback-speed 0` removes
the pacer for maximum throughput. `--sim-hz 120` can provide a smoother
high-refresh animation; it does not increase the policy decision frequency.

Use these flags when you need a more aggressive speed profile:

```bash
# Faster GUI if the viewport is still lagging
python scripts/play.py --config configs/DirGate_dandelion.yaml --checkpoint checkpoints/DirGate_dandelion/poca_final.pt --gui-performance-preset fast --gui-resolution 640x360 --gui-texture-budget 0.25 --sim-hz 60 --deterministic

# Maximum viewport speed, at the cost of material/color fidelity
python scripts/play.py --config configs/DirGate_dandelion.yaml --checkpoint checkpoints/DirGate_dandelion/poca_final.pt --gui-performance-preset fast --gui-disable-materials --gui-resolution 640x360 --sim-hz 60 --deterministic
```

The same GUI flags work with `scripts/manual_control_isaac.py`. If Isaac Sim
over-subscribes a many-core CPU, try `--gui-cpu-threads 16`. The lightweight
viewer defaults its small CPU policy workload to one PyTorch thread; set
`--viewer-torch-threads 0` to leave PyTorch's global setting unchanged.

## Manual Mission Checks

Pygame viewer, without Isaac Sim:

```bash
python scripts/manual_control.py --task SwarmACB-XOR-v0
python scripts/manual_control.py --task SwarmACB-Homing-v0
python scripts/manual_control.py --task SwarmACB-Foraging-v0
python scripts/manual_control.py --task SwarmACB-SCA-v0
python scripts/manual_control.py --task SwarmACB-DirectionalGate-v0
```

Isaac Sim viewport viewer:

```bash
python scripts/manual_control_isaac.py --task SwarmACB-XOR-v0
python scripts/manual_control_isaac.py --task SwarmACB-Homing-v0
python scripts/manual_control_isaac.py --task SwarmACB-Foraging-v0
python scripts/manual_control_isaac.py --task SwarmACB-SCA-v0
python scripts/manual_control_isaac.py --task SwarmACB-DirectionalGate-v0
```

Keyboard controls:

```text
Z / Up       forward
S / Down     backward
Q / Left     turn left
D / Right    turn right
A            stop
R            reset
Esc          quit
Numpad 0-5   set behavior module for the other robots
```

## TensorBoard

Local runs:

```bash
tensorboard --logdir runs/
```

For HPC runs, either copy the run directory locally or forward a port from the
cluster login node:

```bash
ssh -L 6006:localhost:6006 user@cluster
tensorboard --logdir /path/to/SwarmACB-isaaclab/runs --host 127.0.0.1 --port 6006
```

Then open `http://localhost:6006`.

## Configuration

Training is configured through ML-Agents-style YAML files:

```yaml
behaviors:
  Sheltering_cyclamen:
    task: SwarmACB-Sheltering-v0
    variant: cyclamen
    trainer_type: poca
    hyperparameters:
      batch_size: 2048
      learning_rate: 0.0003
    network_settings:
      hidden_units: 128
      num_layers: 1
      memory:
        memory_size: 128
        sequence_length: 128
    max_steps: 180000000
    time_horizon: 1000
    environment:
      num_envs: 5
      decision_period: 1
      episode_length_s: 180.0
```

The paper-parity clock is 10 Hz: Isaac runs with `dt: 0.1`, `decimation: 1`,
and `decision_period: 1`. This gives 1200 decisions in a 120 s episode and
1800 in a 180 s episode. It also makes Cyclamen's 128-sample recurrent window
span 12.8 s, as reported in the paper. The `DecisionPeriod: 5` serialized in
the available Unity prefab conflicts with all three paper constraints and is
therefore treated as stale scene metadata.

`max_steps` counts individual robot decisions, as ML-Agents does. With 20
robots, five parallel environments, and 5000 total episodes, use 120,000,000
for 120 s missions (Directional Gate and Homing) and 180,000,000 for 180 s
missions (XOR, Foraging, and Sheltering).

Fixed-option Option-Critic configs use the same layout with
`trainer_type: option_critic`:

```yaml
behaviors:
  OC_Sheltering_cyclamen:
    task: SwarmACB-Sheltering-v0
    variant: cyclamen
    trainer_type: option_critic
    hyperparameters:
      learning_rate: 0.0003
      termination_penalty: 0.01
      termination_coef: 0.1
      termination_entropy_coef: 0.001
      option_value_coef: 0.5
      baseline_coef: 0.25
    network_settings:
      hidden_units: 128
      num_layers: 1
      num_options: 6
      memory:
        memory_size: 128
        sequence_length: 128
```

Learned-option Phase 2 configs use `trainer_type: learned_option_critic`.
`num_options` controls the number of policies learned from scratch:

```yaml
behaviors:
  OC2_Sheltering_cyclamen:
    task: SwarmACB-Sheltering-v0
    variant: cyclamen
    trainer_type: learned_option_critic
    hyperparameters:
      learning_rate: 0.0003
      intra_option_coef: 1.0
      selector_coef: 1.0
      option_entropy_coef: 0.005
      termination_penalty: 0.01
      termination_coef: 0.1
      action_baseline_coef: 0.25
      option_value_coef: 0.5
      option_baseline_coef: 0.25
      attention_diversity_coef: 0.01
      attention_temporal_coef: 0.01
    network_settings:
      hidden_units: 128
      num_layers: 1
      num_options: 6
      memory:
        memory_size: 128
        sequence_length: 128
```

Checkpoints produced before paper-parity version 4 use incompatible network or
training semantics. Version 4 makes the centralized critic inherit
`hidden_units` and `num_layers` from `network_settings`, exactly as Unity passes
the same `NetworkSettings` object to its actor and POCA critic. They remain
usable for historical evaluation, but parity experiments must start from fresh
weights. At startup, Cyclamen and fixed Option-Critic must report a
`hidden=128 layers=1` critic; Dandelion, Daisy, and Lily must report
`hidden=512 layers=2`.

Before submitting the full HPC matrix, run the dependency-light parity audit:

```bash
python scripts/validate_paper_parity.py
```

It validates all 35 YAML files, experiment budgets, raw and runtime-resolved
actor/critic network sizes, recurrent semantics, and the mission
geometry/constants that are most sensitive to Unity scene overrides.

The HPC array launcher passes `SLURM_ARRAY_TASK_ID` as `--seed`, so the ten
controllers use reproducible seeds 0 through 9. Training also follows the
ML-Agents update-buffer rule: complete `time_horizon` or terminal trajectories
are accumulated until `buffer_size` is exceeded before each update.

CLI arguments override YAML values, including `--task`, `--variant`,
`--num_envs`, `--total_timesteps`, `--seed`, `--log_dir`, and
`--checkpoint_dir`.

## License

BSD-3-Clause
