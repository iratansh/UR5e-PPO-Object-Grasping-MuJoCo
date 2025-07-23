#!/usr/bin/env python3
"""
Direct test of custom_scene.xml using the standard homestri approach
This should work if homestri's default scenes work
"""

from homestri_ur5e_rl.envs.custom_robot_env import CustomRobot
import numpy as np

# Try using CustomRobot (which works with default scene) but with your custom scene
print("🧪 Testing custom_scene.xml with standard homestri approach...\n")

try:
    env = CustomRobot(
        model_path="../assets/base_robot/custom_scene.xml",  # Your custom scene
        render_mode="human",
        frame_skip=40,  # Same as default
    )
    
    print(" Environment created successfully!")
    
    # Reset
    obs = env.reset()[0]
    print(" Reset successful")
    print(f"   Observation shape: {obs.shape}")
    
    # Get some info about the scene
    if hasattr(env, 'model_names'):
        print(f"\n📋 Scene contains:")
        print(f"   Bodies: {len(env.model_names.body_names)}")
        print(f"   Joints: {len(env.model_names.joint_names)}")
        print(f"   Actuators: {len(env.model_names.actuator_names)}")
    
    print("\n Running visualization...")
    print("   The MuJoCo viewer should open now")
    print("   Press ESC or close window to quit\n")
    
    # Run simulation
    step = 0
    while True:
        # Simple control
        action = np.zeros(7)
        
        # Move joints slowly
        action[0] = 0.1 * np.sin(step * 0.01)  # Base rotation
        action[1] = -0.1 * np.sin(step * 0.01)  # Shoulder
        
        # Step
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render (this should open the standard MuJoCo viewer)
        env.render()
        
        step += 1
        
        if terminated or truncated:
            break
            
except Exception as e:
    print(f" Error: {e}")
    import traceback
    traceback.print_exc()
    
print("\n Test completed!")