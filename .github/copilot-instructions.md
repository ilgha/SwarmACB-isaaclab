# SwarmACB Isaac Lab - AI Agent Instructions

## Project Overview

**SwarmACB_isaac** is an **Isaac Lab extension** for multi-agent reinforcement learning (MARL). It currently implements two environments:
1. **Cart-Double-Pendulum** (baseline template): Two agents cooperate to balance a double pendulum
2. **Foraging Environment** (main project): Swarm robots collect food cubes in an arena using POCA/SwarmACB algorithm

### Architecture at a Glance

```
source/SwarmACB_isaac/SwarmACB_isaac/
├── tasks/direct/
│   ├── swarmacb_isaac_marl/          # Baseline: Cart-double-pendulum MARL
│   │   ├── swarmacb_isaac_marl_env.py
│   │   ├── swarmacb_isaac_marl_env_cfg.py
│   │   └── agents/skrl_mappo_cfg.yaml
│   └── swarmacb_foraging/            # Main: Foraging task
│       ├── swarmacb_foraging_env.py       # Foraging environment
│       ├── swarmacb_foraging_env_cfg.py   # Configuration
│       └── agents/__init__.py             # POCA config (TODO)
├── ui_extension_example.py              # Optional Omniverse UI extension
└── __init__.py                          # Package init (auto-imports tasks)

scripts/
├── zero_agent.py                        # Dummy zero-action agent
├── random_agent.py                      # Dummy random-action agent  
├── manual_control.py                    # Keyboard-controlled robot for testing
├── validate_foraging_env.py             # Foraging environment validator
└── skrl/
    ├── train.py                         # Training script (MAPPO/POCA)
    └── play.py                          # Inference/evaluation script
```

## Foraging Environment - Key Architecture

The **foraging task** implements a cooperative multi-robot system where agents:
- Navigate a dodecagonal arena to find food cubes
- Push food into a nest area (white circular zone)
- Use only **local observations** (actor policy): raycaster, light sensor, nest detection
- **Centralized critic** sees full state (all robot positions) during training only

### Observation Architecture (POCA-Inspired)

**Actor Observations (46D - Local Only, No Privileged Info):**
```python
obs = [
    raycaster_distances,  # 36D: proximity to walls/food/robots (normalized 0-1)
    light_sensor,         # 8D: beacon detection in 8 directions
    in_nest_flag,         # 1D: binary flag if robot in nest zone
    neighbor_count,       # 1D: count of nearby robots
]
# NO position/velocity - actor never sees absolute coordinates
```

**Critic State (Privileged Access - Training Only):**
- All robot positions (2D x N agents)
- Food positions and collection status
- Will be implemented when adding POCA trainer

### Sensor Implementation Details

**RayCaster (36 rays):**
```python
# Isaac Lab 5.x API (NOT ray_hits_distance!)
ray_hits = self.raycaster.data.ray_hits_w      # (num_envs, 36, 3)
sensor_pos = self.raycaster.data.pos_w         # (num_envs, 3)
distances = torch.linalg.norm(ray_hits - sensor_pos.unsqueeze(1), dim=-1)
```

**Pattern Configuration:**
```python
pattern_cfg=patterns.LidarPatternCfg(
    channels=1,
    vertical_fov_range=(0.0, 0.0),    # Flat horizontal plane
    horizontal_fov_range=(-180, 180),  # Full 360°
    horizontal_res=10.0,               # 36 rays = 360° / 10°
)

## Critical Knowledge for Code Changes

### 0. Isaac Lab 5.x Primitive Spawning (BREAKING CHANGE)

**Isaac Lab 5.x uses configuration-based spawning, NOT `spawn_primitive()`.**

✅ **Correct Isaac Lab 5.x API:**
```python
import isaaclab.sim as sim_utils

# Spawn a cube/box
cfg = sim_utils.CuboidCfg(
    size=(0.2, 0.2, 0.15),
    rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
    collision_props=sim_utils.CollisionPropertiesCfg(),
    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.5, 0.0)),
    mass_props=sim_utils.MassPropertiesCfg(mass=0.05),  # NOT mass= in rigid_props!
)
cfg.func(prim_path="/World/envs/env_.*/Food_0", cfg=cfg, translation=(x, y, z))

# Spawn a sphere
sphere_cfg = sim_utils.SphereCfg(radius=0.1, ...)
sphere_cfg.func(prim_path="/World/Beacon", cfg=sphere_cfg)

# Spawn a cylinder
cylinder_cfg = sim_utils.CylinderCfg(radius=0.6, height=0.1, ...)
cylinder_cfg.func(prim_path="/World/Nest", cfg=cylinder_cfg)
```

❌ **Wrong (old API, doesn't exist):**
```python
sim_utils.spawn_primitive(prim_path="...", primitive_type="Cube", ...)  # AttributeError!
```

**Key differences:**
- Use `CuboidCfg`, `SphereCfg`, `CylinderCfg` (NOT `primitive_type=` string)
- Call `.func(prim_path, cfg, translation=...)` pattern
- Color via `visual_material=PreviewSurfaceCfg(diffuse_color=...)` (NOT `color=`)
- Mass via `mass_props=MassPropertiesCfg(mass=...)` (NOT `rigid_props.mass`)

### 1. MARL Action/Observation Spaces (Key Gotcha!)

**In this MARL environment, `action_space` is a CALLABLE function, NOT a `gym.Space` object.**

✅ **Correct usage:**
```python
agent_names = env.possible_agents  # ["cart", "pendulum"]
actions = {
    agent: torch.zeros(env.action_space(agent).shape, device=env.device)
    for agent in env.possible_agents
}
env.step(actions)  # Pass dict[agent_name -> action_tensor]
```

❌ **Wrong (causes AttributeError):**
```python
actions = torch.zeros(env.action_space.shape)  # 'function' has no attribute 'shape'
```

**Why?** `DirectMARLEnv` uses per-agent spaces; `action_space(agent_name)` returns `gym.spaces.Box` for that agent.

### 2. Environment Registration Pattern

All environments are registered in `tasks/direct/swarmacb_isaac_marl/__init__.py`:

```python
gym.register(
    id="Template-Swarmacb-Isaac-Marl-Direct-v0",
    entry_point="...SwarmacbIsaacMarlEnv",
    kwargs={
        "env_cfg_entry_point": "...SwarmacbIsaacMarlEnvCfg",
        "skrl_mappo_cfg_entry_point": "...agents:skrl_mappo_cfg.yaml",
    },
)
```

When adding new environments:
- Derive from `DirectMARLEnvCfg` (configuration) and `DirectMARLEnv` (class)
- Register with **`skrl_mappo_cfg_entry_point`** (not `skrl_cfg_entry_point`) for MAPPO support
- Name tasks starting with "Template-" (used by `list_envs.py` filtering)

### 3. Configuration Split: Two-Layer Design

**Layer 1: Environment Config** (`*_env_cfg.py`)
- Inherits `DirectMARLEnvCfg`
- Defines: agents, action/observation spaces, sim physics, rewards
- Example: `possible_agents = ["cart", "pendulum"]`, `action_spaces = {"cart": 1, "pendulum": 1}`

**Layer 2: Agent Config** (YAML, e.g., `skrl_mappo_cfg.yaml`)
- Defines: neural networks, learning rates, memory, trainer settings
- Instantiated by skrl runner; modifiable at training time

Both are loaded by `@hydra_task_config` decorator in `train.py`/`play.py`.

### 4. Agent Methods to Override (DirectMARLEnv)

When creating a new MARL environment:

| Method | Purpose | Returns |
|--------|---------|---------|
| `_setup_scene()` | Initialize robots, spawners, lights | None |
| `_pre_physics_step(actions)` | Store actions before physics step | None |
| `_apply_action()` | Convert actions to joint commands | None |
| `_get_observations()` | Compute per-agent observations | `dict[agent_name -> tensor]` |
| `_get_rewards()` | Compute per-agent rewards | `dict[agent_name -> tensor]` |
| `_get_dones()` | Check termination conditions | `(terminated_dict, truncated_dict)` |
| `_reset_idx(env_ids)` | Reset selected environment instances | None |

**Key pattern:** All dict returns use agent names as keys (e.g., `{"cart": tensor, "pendulum": tensor}`).

## Development Workflows

### Testing Foraging Environment (Manual Control)
```bash
# Keyboard-controlled robot - test sensors and rewards interactively
python scripts/manual_control.py --task Template-Swarmacb-Foraging-Direct-v0 --num_envs 1

# Controls: W/S (forward/back), A/D (turn), Q (stop), ESC (exit)
# Displays: Live observations breakdown, rewards, raycaster distances
```

### Validating Environment Setup (No Training)
```bash
# Zero actions (cart-pendulum baseline)
python scripts/zero_agent.py --task Template-Swarmacb-Isaac-Marl-Direct-v0 --num_envs 16

# Random actions (foraging environment)
python scripts/random_agent.py --task Template-Swarmacb-Foraging-Direct-v0 --num_envs 4

# Custom foraging validator
python scripts/validate_foraging_env.py --num_envs 4 --num_steps 500
```

### Training a Model
```bash
# Train cart-pendulum with MAPPO (default algorithm)
python scripts/skrl/train.py --task Template-Swarmacb-Isaac-Marl-Direct-v0 --num_envs 4096

# Train foraging with POCA (TODO - algorithm not yet implemented)
python scripts/skrl/train.py --task Template-Swarmacb-Foraging-Direct-v0 --algorithm POCA
```

Logs go to `logs/skrl/{task_name}/{timestamp}_{algorithm}_torch/`.

### Playing/Evaluating Trained Policy
```bash
# Auto-find latest checkpoint
python scripts/skrl/play.py --task Template-Swarmacb-Isaac-Marl-Direct-v0 --num_envs 4

# Specify custom checkpoint + record video
python scripts/skrl/play.py --task Template-Swarmacb-Isaac-Marl-Direct-v0 \
  --checkpoint logs/skrl/.../checkpoints/agent.pt --video --num_envs 4
```

### Listing Available Tasks
```bash
python scripts/list_envs.py  # Filters for "Template-" prefix
```

## Code Patterns & Conventions

### Per-Agent Iteration
```python
# Correct: iterate over agent names
for agent in self.cfg.possible_agents:
    reward_dict[agent] = compute_agent_reward(...)

# Using agent-specific indices
cart_obs = obs[:, self.cfg.cart_dof_name]  # (num_envs, 1)
```

### Tensor Device Consistency
```python
# Always use environment device (GPU by default for Isaac)
device = env.unwrapped.device  # "cuda:0" or "cpu"
actions = torch.zeros(shape, device=device)
```

### @torch.jit.script for Performance
Physics-heavy reward/observation functions use JIT compilation:
```python
@torch.jit.script
def compute_rewards(...) -> dict[str, torch.Tensor]:
    rew_alive = ...
    return {"cart": rew_alive, "pendulum": rew_alive}
```

### Reward Scaling
Follow convention: combine raw reward signals with weights defined in cfg:
```python
rew_alive = self.cfg.rew_scale_alive * (1.0 - terminated.float())
rew_pos = self.cfg.rew_scale_cart_pos * position_error
total = rew_alive + rew_pos + ...
```

## Integration Points & Dependencies

- **IsaacLab**: `isaaclab.envs.DirectMARLEnv`, `isaaclab.assets.Articulation`
- **skrl**: MAPPO trainer, gym registration via `entry_point`
- **Gymnasium**: Standard gym API (`env.observation_space()`, `env.action_space()`, `env.step()`)
- **Hydra**: Config composition in `train.py`/`play.py` via `@hydra_task_config`

## Common Pitfalls & Solutions

| Issue | Cause | Fix |
|-------|-------|-----|
| `'function' has no attribute 'shape'` | Using `env.action_space.shape` instead of `env.action_space(agent).shape` | Wrap in callable: `env.action_space(agent_name).shape` |
| `Could not find configuration for environment` | Using `--algorithm PPO` with MARL (needs MAPPO) | Change to `--algorithm MAPPO` or update train.py default |
| `AttributeError: module 'isaaclab.sim' has no attribute 'spawn_primitive'` | Using old Isaac Lab API | Use `CuboidCfg(...).func()` pattern (see section 0 above) |
| `'RayCasterData' has no attribute 'ray_hits_distance'` | Wrong raycaster attribute name | Use `ray_hits_w` and compute distances: `torch.linalg.norm(ray_hits - pos, dim=-1)` |
| `RayCaster only supports one mesh prim` | Passing multiple `mesh_prim_paths` | Use single mesh (e.g., `["/World/ground"]` only) |
| `ValueError: negative dimensions are not allowed` | `observation_spaces` set to `-1` | Set explicit positive dimension (e.g., `46` for foraging) |
| Task not listed by `list_envs.py` | Task name doesn't start with "Template-" | Rename environment ID in `__init__.py` |
| Out-of-memory during training | Too many environments or network layers | Reduce `num_envs` or simplify networks in YAML config |

## When Modifying This Codebase

1. **Adding a new task?** Copy `swarmacb_foraging/` folder, update class/config names, register in `__init__.py`, test with `manual_control.py` or `zero_agent.py`.
2. **Changing reward logic?** Edit `_get_rewards()` in env class and update `reward_*` config parameters.
3. **Adding agent types?** Update `possible_agents`, `action_spaces`, `observation_spaces` in cfg; ensure all env methods return per-agent dicts.
4. **Spawning primitives?** Use `CuboidCfg`/`SphereCfg`/`CylinderCfg` with `.func()` pattern (Isaac Lab 5.x); avoid old `spawn_primitive()`.
5. **Adding sensors?** Import `RayCasterCfg` + `patterns`, use `pattern_cfg=patterns.LidarPatternCfg(...)`, query data via `.data.ray_hits_w`.
6. **Tuning training?** Modify `agents/*.yaml` (learning rates, networks, rollout sizes) before retraining.
7. **Testing sensors/observations?** Run `manual_control.py` with keyboard control to see live sensor output.

## Next Steps: POCA Algorithm Implementation

The foraging environment is complete and ready for POCA training. Planned implementation:

**Phase 1: POCA Trainer (skrl extension)**
- Centralized critic (attention-based, sees all agents)
- Decentralized actors (RNN-based, local observations only)
- Posthumous credit assignment for cooperative rewards

**Phase 2: Network Architectures**
- Actor: GRU/LSTM for temporal memory (handle partial observability)
- Critic: Multi-head attention over agent embeddings (scalable to variable swarm sizes)

**Phase 3: Reward Shaping**
- Sparse: +1.0 per food in nest, -0.001 step penalty
- Dense (optional): Distance-to-food, distance-to-nest shaping

See `.github/swarmacb_roadmap.md` for detailed 4-phase plan.

---

**Last Updated:** Feb 12, 2026 | **IsaacLab Version:** 5.x | **skrl Version:** 1.4.3+
