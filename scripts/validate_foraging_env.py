# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to validate the SwarmACB Foraging environment."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Validate SwarmACB Foraging environment.")
parser.add_argument("--num_envs", type=int, default=4, help="Number of environments to simulate.")
parser.add_argument("--num_steps", type=int, default=500, help="Number of simulation steps to run.")
parser.add_argument("--visualize", action="store_true", default=True, help="Visualize the environment.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch
import numpy as np

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import SwarmACB_isaac.tasks  # noqa: F401


def main():
    """Validate the foraging environment."""
    print("[INFO] Initializing environment validation...")
    
    # Create environment configuration
    env_cfg = parse_env_cfg(
        "Template-Swarmacb-Foraging-Direct-v0",
        device=args_cli.device,
        num_envs=args_cli.num_envs,
    )
    
    # Override to smaller number for validation
    env_cfg.scene.num_envs = args_cli.num_envs
    
    # Create environment
    env = gym.make("Template-Swarmacb-Foraging-Direct-v0", cfg=env_cfg)
    
    print(f"[INFO] Environment created successfully!")
    print(f"[INFO] Number of environments: {env_cfg.scene.num_envs}")
    print(f"[INFO] Number of robots per environment: {env_cfg.num_robots}")
    print(f"[INFO] Observation space per agent: {env.observation_space}")
    print(f"[INFO] Action space per agent: {env.action_space}")
    
    # Reset environment
    print("[INFO] Resetting environment...")
    obs, info = env.reset()
    
    print("[INFO] Environment reset successful!")
    print(f"[INFO] Observation keys: {list(obs.keys())}")
    for agent_name, agent_obs in obs.items():
        print(f"[INFO]   {agent_name}: observation shape = {agent_obs.shape}")
    
    # Run simulation
    print(f"[INFO] Running {args_cli.num_steps} simulation steps...")
    step_count = 0
    
    try:
        while simulation_app.is_running() and step_count < args_cli.num_steps:
            # Random actions for all agents
            actions = {
                agent: torch.tensor(
                    np.random.uniform(-1, 1, (env_cfg.scene.num_envs, 2)),
                    device=env.unwrapped.device,
                    dtype=torch.float32,
                )
                for agent in env_cfg.possible_agents
            }
            
            # Step environment
            obs, rewards, dones, truncs, info = env.step(actions)
            
            # Validation checks
            if step_count % 100 == 0:
                print(f"[INFO] Step {step_count}/{args_cli.num_steps}")
                print(f"[INFO]   Observation shapes: {[(k, v.shape) for k, v in obs.items()]}")
                print(f"[INFO]   Reward ranges: {[(k, float(v.min()), float(v.max())) for k, v in rewards.items()]}")
            
            step_count += 1
        
        print(f"\n[SUCCESS] Environment validation completed!")
        print(f"[SUCCESS] Simulated {step_count} steps without errors.")
        
    except Exception as e:
        print(f"\n[ERROR] Environment validation failed!")
        print(f"[ERROR] Error: {str(e)}")
        import traceback
        traceback.print_exc()
    
    # Close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
