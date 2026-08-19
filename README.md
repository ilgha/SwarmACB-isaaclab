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

Phase 2 is configured as `trainer_type: learned_option_critic` and uses the
working name `OC2_cyclamen`. Phase 1 treats each behavior module as one fixed
option. Phase 2 removes those predefined modules from the action path and
learns every option end to end from sensors to the two wheel commands.
`network_settings.num_options` controls the number of learned options; the
original OC2 benchmark learns six. `OC2-2` is the controlled two-option
ablation: it changes only `num_options` while preserving the architecture,
optimizer, exploration schedule, mission budget, and counterfactual losses.
The `cyclamen` name denotes the recurrent, decentralized actor and collective
counterfactual lineage, not reuse of Cyclamen's behavior modules.

OC2 architecture version 4 follows Attention Option-Critic (AOC). For robot
`i` and option `omega`:

```text
h_i,omega = attention_omega(x_i, manager_memory_i)
x_i,omega = h_i,omega * x_i

local option value Q_i^Omega(x_i,omega),
intra-option wheel policy pi_i,omega(a_i | x_i,omega), and
termination beta_i,omega(x_i)
    all consume only the encoded x_i,omega and its option memory
```

Here `x_i` is the full 24-value local sensor vector used by the direct motor
controller. Every option has separate recurrent state and its own value,
two-wheel policy, termination, and attention output heads, while all robots
share the same parameters. There is no unmasked shortcut to an option value,
wheel policy, or termination head.

As in the original AOC implementation, there is no separately learned PPO
selector. New options are selected epsilon-soft from the attended option
scores. As in AOC, exploration is annealed early: epsilon decreases linearly
from `1.0` to `0.1` during the first 10% of training and stays at `0.1`
afterward. The local `Q_i^Omega` head is regressed to the team lambda return so it
is a genuine option-value estimate and can define the epsilon-soft manager.
Counterfactual baselines are used for policy credit, not as value targets.

Training remains centralized and execution decentralized. One local value and
three centralized critic roles separate the objectives:

```text
Q_i^Omega(x_i, omega_i)              local attended option value (execution)
V(s)                                 team value for lambda returns
b_i^U((s, omega), a_-i)              wheel-action counterfactual baseline
Q_Omega(s, omega_vector),
b_i^Omega(s, omega_-i)               collective option value and baseline
```

The intra-option PPO advantage omits only robot `i`'s wheel action; its active
option and every peer option/action remain fixed:

```text
A_i^U       = lambda_return - b_i^U((s, omega), a_-i)
Q_i target  = lambda_return
```

The arrival-state termination loss uses the AOC option advantage while keeping
all peer options fixed. The decentralized epsilon-soft option policy supplies
the probabilities and the centralized critic supplies the candidate values:

```text
V_i^Omega(s') = sum_omega' pi_i^Omega(omega' | x_i') *
  Q_Omega(s', (omega_-i, omega'))

L_beta = beta_i,omega(s') *
  [Q_Omega(s', omega_vector) - V_i^Omega(s')]
```

There is one termination function per learned option. Beta starts at `0.27`,
matching Phase 1's `sigmoid(-1)` initialization; this avoids the weak gradient
caused by the former `0.05` initialization while leaving persistence to be
learned. AOC does not add a deliberation cost to this gradient; its temporal
persistence comes from the L1 difference penalty between consecutive attention
masks. OC2 therefore uses `termination_penalty: 0.0`, no beta prior, no
option-usage balance loss, and no direct option-entropy objective. Pairwise
cosine attention similarity and the temporal L1 loss are the two AOC
regularizers. Their `2:1` weight ratio follows the paper; the absolute scale is
reduced for the normalized batched losses used here.

Each option learns a diagonal Gaussian over raw left/right wheel actions, with
its own learned state-independent standard deviations. The sampled raw action
is used for PPO and the counterfactual critic; the actuator receives
`clip(action, -3, 3) / 3`, matching the continuous ML-Agents policy used by the
Unity-parity controller. PPO ratios use the same per-wheel convention as that
baseline and are evaluated against one immutable recurrent actor snapshot per
update. The original AOC experiments use discrete primitive actions, so this
Gaussian is the necessary continuous-control adaptation; option selection,
attention conditioning, and termination retain the AOC construction.

The local option value is value-regressed, option choice is epsilon-soft, and
termination uses the arrival-state gradient above. Actor and critics both
start at the Cyclamen learning rate `3e-4`; automatic KL learning-rate
adaptation is off. The actor, option paths, and centralized critics use the
`128 x 1` Cyclamen capacity. Fused Adam and TF32 are used when supported,
without changing the learning objective.

For Phase 2 validation, monitor `Policy/Option Usage/*`,
`Policy/Intra-Option Std/*`, `Policy/Mean Absolute Wheel Action`,
`Policy/Wheel Action Clipping`, `Policy/Mean Termination Probability`,
`Policy/Termination Probability/*`, `Policy/Switch Rate`,
`Policy/Option Switch Rate/*`, `Policy/Mean Option Duration Decisions`,
`Policy/Local Option Value Spread`, wheel entropy, and both attention losses.
`Diagnostics/Initial Policy KL` must be approximately zero. Comparable task
reward alone is not sufficient evidence that distinct temporal options formed.

Version-4 actor and schema-6 training checkpoints identify this corrected AOC
manager and continuous action space. Version-2 and version-3 squashed-wheel OC2
checkpoints remain playable for diagnosis, but cannot be resumed into this
algorithm.

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

Run the otherwise identical two-option ablation with the `OC2-2` configs:

```bash
python scripts/train.py --config configs/OC2-2_DirGate_cyclamen.yaml --headless
python scripts/train.py --config configs/OC2-2_XOR_cyclamen.yaml --headless
python scripts/train.py --config configs/OC2-2_Homing_cyclamen.yaml --headless
python scripts/train.py --config configs/OC2-2_Foraging_cyclamen.yaml --headless
python scripts/train.py --config configs/OC2-2_Sheltering_cyclamen.yaml --headless
```

Submit ten independent Phase 2 designs per mission on the cluster with
`scripts/hpc/train_oc2_<mission>.slurm`, for example:

```bash
sbatch scripts/hpc/train_oc2_dirgate.slurm
```

The corrected launchers write to `OC2_<mission>_cyclamen_aoc_hpc_<seed>`
run and checkpoint directories so their fresh experiments cannot mix with
legacy OC2 TensorBoard events or checkpoints.

The OC2-2 launchers follow `scripts/hpc/train_oc2_2_<mission>.slurm` and write
to `OC2-2_<mission>_cyclamen_aoc_hpc_<seed>`. For example:

```bash
sbatch scripts/hpc/train_oc2_2_dirgate.slurm
```

HPC training uses the base Isaac Sim SIF plus the bound repository and
`.syslibs`; it does not mount the legacy ext3 overlay by default. This avoids
intermittent `fuse2fs failed to mount ... in 10s` failures when many array jobs
start together. The common launcher retries failures that happen before the
container command starts (five attempts by default), while Python or training
failures are reported immediately and are never restarted. It also uses a clean
container environment, preserves the allocated CUDA device, and accepts the
Omniverse Kit EULA through `OMNI_KIT_ACCEPT_EULA`. Override the launch policy
with `APPTAINER_LAUNCH_ATTEMPTS` and `APPTAINER_RETRY_DELAY`. A custom image
that genuinely needs the old overlay can opt in with
`HPC_USE_OVERLAY=1` and, optionally, `OVERLAY=/path/to/overlay.img`.

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
The lightweight viewer advances robot motion at 10 Hz. Policies make a new
decision at 2 Hz, matching Unity's five-update `DecisionRequester` period, and
the selected action is retained between decisions. Sensor debug geometry is
refreshed at 10 Hz instead of rebuilding it every rendered frame.

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
      decision_period: 5
      episode_length_s: 180.0
```

For paper-runtime parity, one Isaac environment update reproduces one Unity
`FixedUpdate` displacement and the policy action is retained for five updates.
This matches the serialized Epuck `DecisionRequester` (`DecisionPeriod: 5`) and
the archived training logs, which report 239/359 completed policy decisions for
1200/1800-update episodes. The `dt: 0.1` value keeps the benchmark's displayed
duration at 120/180 seconds; `decision_period` controls the learned MDP cadence,
not the number of environment updates.

`max_steps` counts individual robot decisions, as ML-Agents does. With 20
robots, five parallel environments, and 5000 episode cycles per environment,
use 120,000,000 for 120 s missions (Directional Gate and Homing) and
180,000,000 for 180 s missions (XOR, Foraging, and Sheltering).

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
`num_options` controls the number of policies learned from scratch. The
six-option OC2 configs set it to `6`; the controlled OC2-2 configs set it to
`2` and leave every other field unchanged:

```yaml
behaviors:
  OC2_Sheltering_cyclamen:
    task: SwarmACB-Sheltering-v0
    variant: cyclamen
    trainer_type: learned_option_critic
    hyperparameters:
      batch_size: 2048
      learning_rate: 0.0003
      actor_learning_rate: 0.0003
      beta: 0.005
      intra_option_coef: 1.0
      selector_coef: 0.0
      local_option_value_coef: 0.5
      option_entropy_coef: 0.0
      option_balance_coef: 0.0
      option_balance_final_coef: 0.0
      termination_penalty: 0.0
      termination_coef: 1.0
      termination_entropy_coef: 0.0
      initial_termination_probability: 0.27
      termination_prior_probability: 0.05
      termination_prior_coef: 0.0
      termination_prior_final_coef: 0.0
      action_baseline_coef: 0.25
      option_value_coef: 0.5
      option_baseline_coef: 0.25
      attention_diversity_coef: 0.002
      attention_temporal_coef: 0.001
      initial_log_std: 0.0
      max_grad_norm: 10.0
      actor_max_grad_norm: 1.0
      target_kl: 0.01
      adaptive_actor_lr: false
      fused_optimizer: true
      matmul_precision: high
      option_epsilon_start: 1.0
      option_epsilon_final: 0.1
      option_epsilon_schedule: linear
      option_epsilon_decay_fraction: 0.1
    network_settings:
      hidden_units: 128
      num_layers: 1
      option_hidden_units: 128
      option_num_layers: 1
      num_options: 6
      memory:
        memory_size: 128
        option_memory_size: 128
        sequence_length: 128
    critic_settings:
      hidden_units: 128
      num_layers: 1
      num_heads: 4
```

Paper-parity version 5 is the Unity-matched timing and architecture for
classical Cyclamen and Phase 1. OC2 architecture version 4 is its hierarchical extension:
all actor, option, and critic paths use `128 x 1`, and learned intra-option
policies independently produce the two primitive wheel commands. No predefined
Cyclamen behavior module is called by OC2. Training checkpoints use schema
version 6. Version-2 and version-3 squashed-wheel actors can still be viewed,
but cannot be resumed as version-4/schema-6 training. Start corrected OC2 runs
from fresh weights and keep them in the `_aoc_hpc_` run directories.

Before submitting the full HPC matrix, run the dependency-light parity audit:

```bash
python scripts/validate_paper_parity.py
python scripts/validate_oc2_architecture.py
```

The first command validates all 40 YAML files, experiment budgets, resolved
network sizes, recurrent semantics, and sensitive mission constants. The OC2
validator additionally checks tensor shapes, recurrent step/sequence parity,
attention gradients to all option outputs, two- and six-option continuous wheel
policies, the signs of the termination theorem, and exact immutable
frozen-policy replay.

The HPC array launcher passes `SLURM_ARRAY_TASK_ID` as `--seed`, so the ten
controllers use reproducible seeds 0 through 9. Training also follows the
ML-Agents update-buffer rule: complete `time_horizon` or terminal trajectories
are accumulated until `buffer_size` is exceeded before each update.

CLI arguments override YAML values, including `--task`, `--variant`,
`--num_envs`, `--total_timesteps`, `--seed`, `--log_dir`, and
`--checkpoint_dir`.

## License

BSD-3-Clause
