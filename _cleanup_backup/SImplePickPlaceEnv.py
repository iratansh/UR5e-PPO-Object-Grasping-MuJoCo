#!/usr/bin/env python3
"""
Simplified UR5e Pick-Place Environment
Works with standard homestri setup - no extra complexity needed
"""

import numpy as np
from typing import Dict, Tuple, Optional, Any
import mujoco
from pathlib import Path
import random

import gymnasium as gym
from gymnasium import spaces

# Import homestri components - use the same as working examples
from homestri_ur5e_rl.envs.mujoco.mujoco_env import MujocoEnv
from homestri_ur5e_rl.utils.mujoco_utils import MujocoModelNames

# Use the SAME default camera config as the working examples
DEFAULT_CAMERA_CONFIG = {
    "distance": 2.2,
    "azimuth": 0.0,
    "elevation": -20.0,
    "lookat": np.array([0, 0, 1]),
}

class SimplePickPlaceEnv(MujocoEnv):
    """
    Simple pick-place environment that works with standard homestri
    """
    
    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array"],
        "render_fps": 12,  # Same as working examples
    }
    
    def __init__(
        self,
        model_path="custom_scene.xml",  # Use relative path like working examples
        frame_skip=40,  # Same as working examples
        camera_resolution=84,
        control_mode="joint_velocity",
        default_camera_config: dict = DEFAULT_CAMERA_CONFIG,
        **kwargs,
    ):
        # Setup paths the same way as working examples
        xml_file_path = Path(__file__).parent / "assets" / "base_robot" / model_path
        
        # Robot state: joints(6) + velocities(6) + ee_pos(3) + gripper(1) = 16
        # Object state: pos(3) + goal(3) = 6
        # Camera: 84*84*4 = 28224
        obs_dim = 16 + 6 + (camera_resolution * camera_resolution * 4)
        
        observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )
        
        # Initialize parent class EXACTLY like working examples
        super().__init__(
            str(xml_file_path),
            frame_skip,
            observation_space,
            default_camera_config=default_camera_config,
            **kwargs,
        )
        
        # Store parameters
        self.camera_resolution = camera_resolution
        self.control_mode = control_mode
        
        # Initialize model references
        self.model_names = MujocoModelNames(self.model)
        self._setup_references()
        
        # Task parameters
        self.current_object = None
        self.target_position = np.array([0.6, 0.0, 0.45])
        
        # Action space: 6 joints + 1 gripper
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)
        
        print(f" Simple environment initialized")
        print(f"   XML: {xml_file_path}")
        print(f"   Observation space: {observation_space.shape}")
        print(f"   Action space: {self.action_space.shape}")
        
    def _setup_references(self):
        """Setup model references"""
        # Robot joints
        self.robot_joint_names = [
            "robot0:ur5e:shoulder_pan_joint",
            "robot0:ur5e:shoulder_lift_joint",
            "robot0:ur5e:elbow_joint",
            "robot0:ur5e:wrist_1_joint",
            "robot0:ur5e:wrist_2_joint",
            "robot0:ur5e:wrist_3_joint",
        ]
        
        self.joint_ids = [
            self.model_names.joint_name2id[name] 
            for name in self.robot_joint_names
        ]
        
        # Sites
        self.ee_site_id = self.model_names.site_name2id["robot0:eef_site"]
        
        # Gripper
        self.gripper_actuator_id = self.model_names.actuator_name2id["robot0:2f85:fingers_actuator"]
        
        # Objects
        self.object_names = ["cube_object", "sphere_object", "cylinder_object"]
        
    def reset_model(self) -> np.ndarray:
        """Reset the environment"""
        # Reset to initial state
        mujoco.mj_resetData(self.model, self.data)
        
        # Set robot to home position
        home_positions = [0, -np.pi/2, np.pi/2, -np.pi/2, -np.pi/2, 0]
        for i, joint_id in enumerate(self.joint_ids):
            self.data.qpos[joint_id] = home_positions[i]
            
        # Open gripper
        self.data.ctrl[self.gripper_actuator_id] = 0
        
        # Reset object
        self._reset_objects()
        
        # Forward simulation
        mujoco.mj_forward(self.model, self.data)
        
        return self._get_obs()
        
    def _reset_objects(self):
        """Reset objects to initial positions"""
        # Hide all objects first
        for obj_name in self.object_names:
            if obj_name in self.model_names.body_name2id:
                body_id = self.model_names.body_name2id[obj_name]
                # Set far away
                self.data.qpos[7*self.object_names.index(obj_name):7*self.object_names.index(obj_name)+3] = [10, 10, -10]
                
        # Pick one object randomly
        self.current_object = random.choice(self.object_names)
        obj_idx = self.object_names.index(self.current_object)
        
        # Place at random position on table
        x = np.random.uniform(0.4, 0.6)
        y = np.random.uniform(-0.2, 0.2)
        z = 0.45
        
        # Set position (assuming free joint)
        if self.current_object in self.model_names.body_name2id:
            body_id = self.model_names.body_name2id[self.current_object]
            body = self.model.body(body_id)
            if body.jntadr[0] >= 0:
                qpos_addr = self.model.jnt_qposadr[body.jntadr[0]]
                self.data.qpos[qpos_addr:qpos_addr+3] = [x, y, z]
                self.data.qpos[qpos_addr+3:qpos_addr+7] = [1, 0, 0, 0]  # quaternion
                
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one timestep"""
        # Apply joint actions
        if self.control_mode == "joint_velocity":
            # Velocity control
            for i in range(6):
                actuator_name = f"robot0:ur5e:{self.robot_joint_names[i].split(':')[-1].replace('_joint', '')}"
                if actuator_name in self.model_names.actuator_name2id:
                    actuator_id = self.model_names.actuator_name2id[actuator_name]
                    self.data.ctrl[actuator_id] = action[i]
                    
        # Gripper control
        gripper_action = action[6] if len(action) > 6 else 0
        if gripper_action > 0.5:
            self.data.ctrl[self.gripper_actuator_id] = 255
        elif gripper_action < -0.5:
            self.data.ctrl[self.gripper_actuator_id] = 0
            
        # Simulate
        self.do_simulation(self.data.ctrl, self.frame_skip)
        
        # Get observation
        obs = self._get_obs()
        
        # Simple reward
        reward = self._compute_reward()
        
        # Simple termination
        terminated = False
        truncated = False
        
        info = {
            "object": self.current_object,
            "success": False,
        }
        
        return obs, reward, terminated, truncated, info
        
    def _get_obs(self) -> np.ndarray:
        """Get observation"""
        obs_list = []
        
        # Joint positions and velocities
        joint_pos = [self.data.qpos[j] for j in self.joint_ids]
        joint_vel = [self.data.qvel[j] for j in self.joint_ids]
        obs_list.extend(joint_pos)
        obs_list.extend(joint_vel)
        
        # End effector position
        ee_pos = self.data.site_xpos[self.ee_site_id]
        obs_list.extend(ee_pos)
        
        # Gripper state
        gripper_pos = self.data.ctrl[self.gripper_actuator_id] / 255.0
        obs_list.append(gripper_pos)
        
        # Object position
        if self.current_object and self.current_object in self.model_names.body_name2id:
            obj_id = self.model_names.body_name2id[self.current_object]
            obj_pos = self.data.body(obj_id).xpos
            obs_list.extend(obj_pos)
        else:
            obs_list.extend([0, 0, 0])
            
        # Goal position
        obs_list.extend(self.target_position)
        
        # Camera data (simplified)
        camera_data = self._get_camera_data()
        obs_list.extend(camera_data.flatten())
        
        return np.array(obs_list, dtype=np.float32)
        
    def _get_camera_data(self) -> np.ndarray:
        """Get camera data using standard MuJoCo rendering"""
        # For now, just return zeros - we'll implement proper camera later
        # This avoids the render mode issue
        rgbd = np.zeros((self.camera_resolution, self.camera_resolution, 4))
        return rgbd.astype(np.float32)
        
    def _compute_reward(self) -> float:
        """Simple reward function"""
        # Get positions
        ee_pos = self.data.site_xpos[self.ee_site_id]
        
        if self.current_object and self.current_object in self.model_names.body_name2id:
            obj_id = self.model_names.body_name2id[self.current_object]
            obj_pos = self.data.body(obj_id).xpos
            
            # Distance to object
            dist_to_obj = np.linalg.norm(ee_pos - obj_pos)
            
            # Distance to goal
            dist_to_goal = np.linalg.norm(obj_pos - self.target_position)
            
            # Simple reward
            reward = -dist_to_obj - dist_to_goal
            
            return reward
        
        return 0.0

# Test the environment
if __name__ == "__main__":
    print("\n🧪 Testing Simple Pick-Place Environment...")
    print("This should work just like the default homestri examples!\n")
    
    env = SimplePickPlaceEnv(
        model_path="custom_scene.xml",
        render_mode="human",  # Standard human rendering
        camera_resolution=84,
    )
    
    print("\n Environment created successfully!")
    
    # Reset
    obs = env.reset()[0]
    print(f" Reset successful")
    print(f"   Current object: {env.current_object}")
    print(f"   Observation shape: {obs.shape}")
    
    # Run some steps
    print("\n Running test episode...")
    for i in range(200):
        # Random action
        action = env.action_space.sample() * 0.3
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render normally
        env.render()
        
        if i % 50 == 0:
            print(f"   Step {i}: reward = {reward:.3f}")
            
    print("\n Test completed!")
    env.close()