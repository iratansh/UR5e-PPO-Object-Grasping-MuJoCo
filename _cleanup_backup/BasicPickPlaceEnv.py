#!/usr/bin/env python3
"""
Most basic pick-place environment - just get the viewer working!
No camera data, minimal observation space
"""

import numpy as np
from pathlib import Path
import gymnasium as gym
from gymnasium import spaces

from homestri_ur5e_rl.envs.mujoco.mujoco_env import MujocoEnv
from homestri_ur5e_rl.utils.mujoco_utils import MujocoModelNames

class BasicPickPlaceEnv(MujocoEnv):
    """
    Minimal pick-place environment that just works
    """
    
    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array"],
        "render_fps": 12,
    }
    
    def __init__(
        self,
        model_path="assets/base_robot/custom_scene.xml",
        frame_skip=40,
        **kwargs,
    ):
        # Simple observation space - just joint positions (6)
        observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(6,),
            dtype=np.float32
        )
        
        # Use absolute path
        xml_file_path = Path(__file__).parent / model_path
        
        # Initialize parent with macOS compatibility
        super().__init__(
            str(xml_file_path),
            frame_skip,
            observation_space,
            **kwargs,
        )
        
        # Action space
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)
        
        # Get model names
        self.model_names = MujocoModelNames(self.model)
        
        print(f" Basic environment created")
        print(f"   XML path: {xml_file_path}")
        print(f"   File exists: {xml_file_path.exists()}")
        
    def reset_model(self):
        """Simple reset"""
        # Just reset to initial state
        self.set_state(self.init_qpos, self.init_qvel)
        return self._get_obs()
        
    def _get_obs(self):
        """Minimal observation - just joint positions"""
        # Get first 6 joint positions (robot arm)
        return self.data.qpos[:6].copy()
        
    def step(self, action):
        """Simple step"""
        # Apply action directly to control
        self.data.ctrl[:7] = action
        
        # Simulate
        self.do_simulation(self.data.ctrl, self.frame_skip)
        
        # Get observation
        obs = self._get_obs()
        
        # Dummy reward
        reward = 0.0
        
        # Never terminate
        terminated = False
        truncated = False
        
        info = {}
        
        return obs, reward, terminated, truncated, info

# Test it
if __name__ == "__main__":
    print("🧪 Testing Basic Pick-Place Environment")
    print("This is the simplest possible version\n")
    
    env = BasicPickPlaceEnv(render_mode="human")
    
    print("\nResetting environment...")
    obs = env.reset()[0]
    print(f" Reset successful, obs shape: {obs.shape}")
    
    print("\n Running visualization...")
    print("   The MuJoCo viewer should open")
    print("   🖱  Camera Controls:")
    print("      • Left mouse drag: Rotate camera (orbit)")
    print("      • Right mouse drag: Pan camera")
    print("      • Mouse scroll: Zoom in/out")
    print("      • R key: Reset camera view")
    print("      • F key: Focus on origin")
    print("      • ESC: Exit viewer")
    print("   Press ESC to exit\n")
    
    for i in range(10000):
        # Simple sine wave motion
        action = np.zeros(7)
        action[0] = 0.3 * np.sin(i * 0.02)
        action[1] = -0.3 * np.sin(i * 0.02)
        
        obs, _, _, _, _ = env.step(action)
        env.render()
        
    env.close()
    print("\n Test completed!")