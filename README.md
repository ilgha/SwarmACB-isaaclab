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
    agents/                     # POCA and fixed-option Option-Critic trainers
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
Exploration, Stop, Phototaxis, Anti-phototaxis, Attraction, Repulsion
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

## Sensor Suite

Each e-puck has:

- 8 IR proximity sensors, range 0.10 m
- 8 directional light sensors, disabled in XOR and Homing
- 3 ground sensors returning grey, white, or black
- Range-and-bearing neighborhood sensing, range 0.20 m

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

Or override the task and variant from the command line:

```bash
python scripts/train.py --task SwarmACB-Foraging-v0 --variant tulip --headless
python scripts/train.py --task SwarmACB-SCA-v0 --variant cyclamen --num_envs 5 --headless
```

Useful smoke test:

```bash
python scripts/train.py --config configs/Sheltering_cyclamen.yaml --headless --total_timesteps 2000 --num_envs 1 --log_dir runs/Sheltering_smoke --checkpoint_dir checkpoints/Sheltering_smoke
python scripts/train.py --config configs/OC_Sheltering_cyclamen.yaml --headless --total_timesteps 2000 --num_envs 1 --log_dir runs/OC_Sheltering_smoke --checkpoint_dir checkpoints/OC_Sheltering_smoke
```

## Evaluation

Evaluate an IsaacLab environment exactly:

```bash
python scripts/play.py --config configs/DirGate_cyclamen.yaml --checkpoint checkpoints/DirGate_cyclamen/poca_final.pt --num_envs 1 --num_episodes 10 --deterministic
python scripts/play.py --config configs/OC_DirGate_cyclamen.yaml --checkpoint checkpoints/OC_DirGate_cyclamen/option_critic_final.pt --num_envs 1 --num_episodes 10 --deterministic
```

The play script can also use a fast Isaac viewport viewer for smoother POCA
inspection:

```bash
python scripts/play.py --config configs/Sheltering_cyclamen.yaml --checkpoint checkpoints/Sheltering_cyclamen/poca_final.pt --fast-viewer --deterministic
```

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
        sequence_length: 64
    max_steps: 120000000
    time_horizon: 1000
    environment:
      num_envs: 5
      decision_period: 1
      episode_length_s: 180.0
```

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
        sequence_length: 64
```

CLI arguments override YAML values, including `--task`, `--variant`,
`--num_envs`, `--total_timesteps`, `--log_dir`, and `--checkpoint_dir`.

## License

BSD-3-Clause
