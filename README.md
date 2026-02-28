# SwarmACB — Isaac Lab

Multi-agent swarm robotics reinforcement learning built on [Isaac Lab](https://isaac-sim.github.io/IsaacLab/).

This project implements the **Directional Gate** mission from the CASA (Collective Autonomous Swarm Adaptation) framework: 20 simulated e-puck robots must learn to cross a gate in a specific direction inside a dodecagonal arena, trained with **MA-POCA** (Multi-Agent POsthumous Credit Assignment).

---

## Architecture

```
SwarmACB_isaac/
├── configs/                        # ML-Agents-style YAML training configs
│   ├── DirGate_dandelion.yaml      #   continuous 2D wheels (obs=24)
│   ├── DirGate_daisy.yaml          #   discrete 6 modules  (obs=24)
│   ├── DirGate_lily.yaml           #   discrete 6 modules  (obs=4)
│   ├── DirGate_tulip.yaml          #   discrete 6 modules  (obs=4, small net)
│   └── DirGate_cyclamen.yaml       #   discrete 6 modules  (obs=4, LSTM)
├── scripts/
│   ├── train.py                    # Training entry point (Isaac Sim + POCA)
│   ├── play.py                     # Evaluation / replay from checkpoint
│   ├── manual_control.py           # Pygame manual control (no Isaac Sim)
│   └── manual_control_isaac.py     # Isaac Sim viewport manual control
└── source/SwarmACB_isaac/SwarmACB_isaac/tasks/direct/
    ├── agents/                     # POCA trainer, networks, buffer, config loader
    ├── epuck/                      # E-puck sensor suite & behaviour modules
    └── missions/directional_gate/  # DirectMARLEnv + env config
```

## The Mission

| Element | Detail |
|---|---|
| **Arena** | Regular dodecagon (12 sides), area = 4.91 m², circumradius ≈ 1.28 m |
| **Robots** | 20 e-puck cylinders (r = 0.035 m, differential drive) |
| **Gate** | White strip (0.45 m wide) with a black corridor (0.50 m wide) north of it |
| **Light** | Point light source at the south edge |
| **Reward** | r(t) = K⁺ − K⁻ (correct north→south crossings minus incorrect south→north) |
| **Episode** | 120 s at 10 Hz = 1200 steps |

## CASA Variants

All five variants from the paper are supported:

| Variant | Obs dim | Action | Description |
|---|---|---|---|
| **Dandelion** | 24 | Continuous (left/right wheel) | Full sensor vector, direct motor control |
| **Daisy** | 24 | Discrete (6 modules) | Full sensors, behaviour module selection |
| **Lily** | 4 | Discrete (6 modules) | Minimal sensors (ground + z̃) |
| **Tulip** | 4 | Discrete (6 modules) | Same as Lily, smaller network |
| **Cyclamen** | 4 | Discrete (6 modules) | Same as Lily, LSTM memory |

The 6 behaviour modules are: **Exploration**, **Stop**, **Phototaxis**, **Anti-phototaxis**, **Attraction**, **Repulsion**.

## Sensor Suite

Each e-puck has:
- **8 IR proximity sensors** — short-range obstacle detection (0.10 m)
- **8 light sensors** — directional light intensity
- **3-channel ground sensor** — grey (0.5), white (1.0), or black (0.0)
- **Range-and-bearing (RAB)** — neighbour presence (z̃) + 4D directional projection (0.20 m range)

## Physics

The simulation is **kinematic** (pure PyTorch tensors, no USD articulations):
- Differential-drive integration at 10 Hz
- Analytical wall collision (dodecagonal boundary + gate side walls)
- Elastic inter-robot push-out
- Massively parallelisable on GPU

---

## Installation

1. **Install Isaac Lab** following the [official guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html) (conda or uv recommended).

2. **Clone this repo** (outside the IsaacLab directory):
   ```bash
   git clone https://github.com/ilgha/SwarmACB-isaaclab.git
   cd SwarmACB-isaaclab
   ```

3. **Install the extension** in editable mode:
   ```bash
   python -m pip install -e source/SwarmACB_isaac
   ```

## Usage

### Training

```bash
# Dandelion, 64 parallel envs, headless
python scripts/train.py --config configs/DirGate_dandelion.yaml --num_envs 64 --headless

# Daisy variant
python scripts/train.py --config configs/DirGate_daisy.yaml --num_envs 64 --headless

# Resume from checkpoint
python scripts/train.py --config configs/DirGate_dandelion.yaml --checkpoint checkpoints/DirGate_dandelion/poca_1000000.pt
```

### Evaluation

```bash
python scripts/play.py --config configs/DirGate_dandelion.yaml --checkpoint checkpoints/DirGate_dandelion/poca_best.pt
```

### Manual Control (Pygame)

```bash
python scripts/manual_control.py
```
Drive robot #0 with **Z/↑** (forward), **S/↓** (backward), **Q/←** (left), **D/→** (right). Numpad 0–5 sets the behaviour module for the other 19 robots.

### Manual Control (Isaac Sim)

```bash
python scripts/manual_control_isaac.py
```
Same controls, rendered in the Isaac Sim viewport with full 3D arena visualisation.

### TensorBoard

```bash
tensorboard --logdir runs/
```

## Configuration

Training is configured via YAML files in `configs/`. The format mirrors ML-Agents:

```yaml
behaviors:
  DirGate_dandelion:
    variant: dandelion
    hyperparameters:
      batch_size: 2048
      learning_rate: 0.0003
      learning_rate_schedule: linear
      # ...
    network_settings:
      hidden_units: 512
      num_layers: 2
    max_steps: 120000000
    time_horizon: 1000
    environment:
      num_envs: 5
```

CLI arguments override YAML values: `--num_envs`, `--variant`, `--total_timesteps`, etc.

---

## License

BSD-3-Clause