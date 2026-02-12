# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Keyboard-controlled robot for testing sensors and rewards in foraging environment.

Controls:
    UP/DOWN: Forward/Backward
    LEFT/RIGHT: Turn Left/Right
    SPACE: Stop
    ESC: Exit
    
Viewport Camera:
    W/A/S/D: Pan camera
    Right-click drag: Rotate camera
    Scroll: Zoom
"""

import argparse
import torch
import numpy as np

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Manual robot control with keyboard for testing.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Template-Swarmacb-Foraging-Direct-v0", help="Name of the task.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import carb.input
import omni.appwindow
from isaacsim.util.debug_draw import _debug_draw

import SwarmACB_isaac.tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


# ==================
# Debug Ray Visualization
# ==================
def draw_raycaster_debug(env, draw_interface):
    """Draw analytical raycaster rays as colored lines from robot to hit points.
    
    Green = far (safe), Yellow = medium, Red = close (obstacle nearby).
    Also draws a translucent sensor range circle around the robot.
    """
    draw_interface.clear_lines()
    draw_interface.clear_points()
    
    unwrapped = env.unwrapped
    
    # Get analytical ray data for env 0
    ray_hits = unwrapped.debug_ray_hits_w[0].cpu().numpy()   # (num_rays, 3)
    sensor_pos = unwrapped.debug_ray_origins_w[0].cpu().numpy()  # (3,)
    
    num_rays = ray_hits.shape[0]
    max_dist = unwrapped.cfg.raycaster_max_distance
    
    starts = []
    ends = []
    colors = []
    sizes = []
    
    # Points at ray hit locations (show obstacle detection)
    hit_points = []
    hit_colors = []
    hit_sizes = []
    
    for i in range(num_rays):
        hit = ray_hits[i]
        dist = float(np.linalg.norm(hit - sensor_pos))
        norm_dist = min(dist / max_dist, 1.0)
        
        # Color: red (close) -> yellow (medium) -> green (far)
        if norm_dist < 0.5:
            r = 1.0
            g = norm_dist * 2.0
        else:
            r = 1.0 - (norm_dist - 0.5) * 2.0
            g = 1.0
        
        starts.append(sensor_pos.tolist())
        ends.append(hit.tolist())
        colors.append((r, g, 0.0, 0.6))
        sizes.append(1.5)
        
        # Draw a dot at the hit point if it detected something (not at max range)
        if norm_dist < 0.95:
            hit_points.append(hit.tolist())
            hit_colors.append((1.0, 0.2, 0.2, 1.0))
            hit_sizes.append(10.0)
    
    draw_interface.draw_lines(starts, ends, colors, sizes)
    
    if hit_points:
        draw_interface.draw_points(hit_points, hit_colors, hit_sizes)
    
    # ==================
    # Draw sensor range circle around robot
    # ==================
    range_points = []
    range_colors = []
    range_sizes = []
    num_circle_pts = 48
    robot_z = float(sensor_pos[2])
    robot_x = float(sensor_pos[0])
    robot_y = float(sensor_pos[1])
    
    for a in range(num_circle_pts):
        angle = 2 * np.pi * a / num_circle_pts
        px = robot_x + max_dist * np.cos(angle)
        py = robot_y + max_dist * np.sin(angle)
        range_points.append((px, py, robot_z))
        range_colors.append((0.3, 0.5, 1.0, 0.5))  # Blue, semi-transparent
        range_sizes.append(4.0)
    
    draw_interface.draw_points(range_points, range_colors, range_sizes)
    
    # ==================
    # Draw nest area as a circle of dots
    # ==================
    nest_local = unwrapped.nest_pos_local.cpu().numpy()
    env_origin = unwrapped.scene.env_origins[0].cpu().numpy()
    nest_world = env_origin[:2] + nest_local
    nest_z = env_origin[2] + 0.05
    radius = unwrapped.cfg.nest_radius
    
    nest_points = []
    nest_colors = []
    nest_sizes = []
    for a in range(36):
        angle = 2 * np.pi * a / 36
        px = nest_world[0] + radius * np.cos(angle)
        py = nest_world[1] + radius * np.sin(angle)
        nest_points.append((px, py, nest_z))
        nest_colors.append((0.2, 1.0, 0.2, 1.0))
        nest_sizes.append(8.0)
    
    draw_interface.draw_points(nest_points, nest_colors, nest_sizes)


class KeyboardController:
    """Keyboard controller for manual robot control using event subscription."""
    
    def __init__(self):
        self.v_left = 0.0
        self.v_right = 0.0
        
        # Track key states
        self.key_states = {}
        
        # Get input interface
        self.input = carb.input.acquire_input_interface()
        
        # Get the keyboard device from app window
        app_window = omni.appwindow.get_default_app_window()
        self.keyboard = app_window.get_keyboard()
        
        # Subscribe to keyboard events (needs keyboard device as first arg)
        self.keyboard_sub = self.input.subscribe_to_keyboard_events(
            self.keyboard,
            self._on_keyboard_event
        )
    
    def _on_keyboard_event(self, event, *args, **kwargs):
        """Handle keyboard events."""
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            self.key_states[event.input] = True
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            self.key_states[event.input] = False
        return True
    
    def _is_key_pressed(self, key_code):
        """Check if a key is currently pressed."""
        return self.key_states.get(key_code, False)
    
    def get_action(self):
        """Compute action from current key states (polled each frame)."""
        # Reset velocities
        accel = 0.0
        turn = 0.0
        
        # Forward/backward (arrow keys)
        if self._is_key_pressed(carb.input.KeyboardInput.UP):
            accel = 1.0
        if self._is_key_pressed(carb.input.KeyboardInput.DOWN):
            accel = -1.0
        
        # Turning (arrow keys)
        if self._is_key_pressed(carb.input.KeyboardInput.LEFT):
            turn = -0.5  # Turn left (reduce right wheel)
        if self._is_key_pressed(carb.input.KeyboardInput.RIGHT):
            turn = 0.5   # Turn right (reduce left wheel)
        
        # Stop command (spacebar)
        if self._is_key_pressed(carb.input.KeyboardInput.SPACE):
            self.v_left = 0.0
            self.v_right = 0.0
        else:
            # Differential drive: both wheels forward + differential for turning
            self.v_left = np.clip(accel - turn, -1.0, 1.0)
            self.v_right = np.clip(accel + turn, -1.0, 1.0)
        
        return torch.tensor([[self.v_left, self.v_right]], dtype=torch.float32)
    
    def should_exit(self):
        """Check if ESC key is pressed."""
        return self._is_key_pressed(carb.input.KeyboardInput.ESCAPE)
    
    def cleanup(self):
        """Cleanup keyboard subscription."""
        if self.keyboard_sub is not None:
            self.input.unsubscribe_to_keyboard_events(self.keyboard, self.keyboard_sub)
            self.keyboard_sub = None


def print_observation_summary(obs, agent_name, env):
    """Print formatted observation data."""
    obs_np = obs.cpu().numpy()[0]  # First environment
    obs_len = len(obs_np)
    unwrapped = env.unwrapped
    
    # Get actual raycaster ray count from environment
    num_rays = getattr(unwrapped, 'num_raycaster_rays', 36)
    num_light = unwrapped.cfg.light_sensor_num_rays
    
    print("\n" + "="*60)
    print(f"OBSERVATIONS for {agent_name} ({obs_len}D, raycaster={num_rays} rays)")
    print("="*60)
    
    # Parse observation components using actual sizes
    idx = 0
    
    # Raycaster (num_rays D)
    raycaster = obs_np[idx:idx+num_rays]
    print(f"Raycaster ({num_rays} rays):")
    print(f"  Min dist: {raycaster.min():.3f}, Max dist: {raycaster.max():.3f}, Mean: {raycaster.mean():.3f}")
    mid = num_rays // 2
    print(f"  Front (ray 0): {raycaster[0]:.3f}, Back (ray {mid}): {raycaster[mid]:.3f}")
    idx += num_rays
    
    # Light sensor (8D)
    light_sensor = obs_np[idx:idx+num_light]
    print(f"\nLight Sensor ({num_light} directions):")
    print(f"  Values: [{', '.join([f'{v:.2f}' for v in light_sensor])}]")
    print(f"  Strongest direction: {np.argmax(light_sensor)} (index)")
    idx += num_light
    
    # In-nest flag (1D)
    if idx < obs_len:
        in_nest = obs_np[idx]
        print(f"\nIn-Nest Flag: {'YES' if in_nest > 0.5 else 'NO'} ({in_nest:.3f})")
        idx += 1
    
    # Carrying food (1D)
    if idx < obs_len:
        carrying = obs_np[idx]
        print(f"Carrying Food: {'YES' if carrying > 0.5 else 'NO'} ({carrying:.3f})")
        idx += 1
    
    # Neighbor count (1D)
    if idx < obs_len:
        neighbor_count = obs_np[idx]
        print(f"Neighbor Count: {int(neighbor_count)}")
    
    print("="*60 + "\n")


def main():
    """Run keyboard-controlled robot test."""
    
    # Parse configuration
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    
    # Create environment
    env = gym.make(args_cli.task, cfg=env_cfg)
    
    # Initialize keyboard controller
    controller = KeyboardController()
    
    print("\n" + "="*60)
    print("MANUAL ROBOT CONTROL - Foraging Environment Test")
    print("="*60)
    print(f"Controlling robot_0 (other {env.unwrapped.cfg.num_robots - 1} robots are idle)")
    print("Robot Controls:")
    print("  UP/DOWN: Forward/Backward")
    print("  LEFT/RIGHT: Turn Left/Right")
    print("  SPACE: Stop")
    print("  ESC: Exit")
    print()
    print("Camera Controls (use these to navigate viewport):")
    print("  W/A/S/D: Pan camera")
    print("  Right-click drag: Rotate view")
    print("  Scroll wheel: Zoom in/out")
    print("="*60 + "\n")
    
    # Reset environment
    obs_dict, info = env.reset()
    
    # Initialize debug draw
    draw = _debug_draw.acquire_debug_draw_interface()
    
    step_count = 0
    episode_reward = {agent: 0.0 for agent in env.unwrapped.cfg.possible_agents}
    
    try:
        while simulation_app.is_running() and not controller.should_exit():
            # Get keyboard action (controls robot_0 only)
            action_tensor = controller.get_action().to(env.unwrapped.device)
            zero_action = torch.zeros_like(action_tensor)
            
            # robot_0 gets keyboard, all others get zero
            actions = {}
            for agent in env.unwrapped.cfg.possible_agents:
                if agent == "robot_0":
                    actions[agent] = action_tensor.expand(args_cli.num_envs, -1)
                else:
                    actions[agent] = zero_action.expand(args_cli.num_envs, -1)
            
            # Step environment
            obs_dict, rewards_dict, terminated_dict, truncated_dict, info = env.step(actions)
            
            step_count += 1
            
            # Draw raycaster debug lines every frame
            draw_raycaster_debug(env, draw)
            
            # Accumulate rewards
            for agent in env.unwrapped.cfg.possible_agents:
                episode_reward[agent] += rewards_dict[agent][0].item()
            
            # Print status every 30 steps (~0.5 seconds at 60Hz)
            if step_count % 30 == 0:
                agent = "robot_0"
                print_observation_summary(obs_dict[agent], agent, env)
                # Show robot_0 position and distance to nest
                unwrapped = env.unwrapped
                robot_pos = unwrapped.robots[0].data.root_pos_w[0, :3].cpu().numpy()
                env_origin = unwrapped.scene.env_origins[0, :3].cpu().numpy()
                local_pos = robot_pos - env_origin
                nest_pos = unwrapped.nest_pos_local.cpu().numpy()
                dist_to_nest = np.linalg.norm(local_pos[:2] - nest_pos)
                print(f"Robot_0 Local Pos: ({local_pos[0]:.2f}, {local_pos[1]:.2f})")
                print(f"Nest Pos: ({nest_pos[0]:.2f}, {nest_pos[1]:.2f}), Dist: {dist_to_nest:.2f}m (radius={unwrapped.cfg.nest_radius}m)")
                print(f"Neighbors in range: {int(obs_dict[agent][0, -1].item())}")
                print(f"Food collected: {unwrapped.food_collected[0].sum().item()}/{unwrapped.cfg.num_food_units}")
                print(f"Step: {step_count}")
                print(f"Current Reward: {rewards_dict[agent][0].item():.4f}")
                print(f"Episode Total Reward: {episode_reward[agent]:.4f}")
                print(f"Action (v_left, v_right): ({controller.v_left:.2f}, {controller.v_right:.2f})")
            
            # Check for episode termination
            if any(terminated_dict.values()) or any(truncated_dict.values()):
                print("\n" + "="*60)
                print("EPISODE FINISHED")
                print("="*60)
                for agent in env.unwrapped.cfg.possible_agents:
                    print(f"{agent} Total Reward: {episode_reward[agent]:.4f}")
                print("="*60 + "\n")
                
                # Reset
                obs_dict, info = env.reset()
                step_count = 0
                episode_reward = {agent: 0.0 for agent in env.unwrapped.cfg.possible_agents}
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        # Cleanup
        controller.cleanup()
        env.close()
        print("\nEnvironment closed")


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
