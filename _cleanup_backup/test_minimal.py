#!/usr/bin/env python3
"""
Minimal test to get MuJoCo viewer working with custom scene
This removes ALL complexity to isolate the issue
"""

from homestri_ur5e_rl.envs.base_robot_env import BaseRobot
import numpy as np
from pathlib import Path

print("🧪 Minimal MuJoCo Viewer Test")
print("=" * 50)

# First, let's verify the default scene works
print("\n1⃣ Testing default scene (should work):")
try:
    env = BaseRobot(render_mode="human")
    print(" Default BaseRobot created")
    
    obs = env.reset()[0]
    print(" Reset successful")
    
    print(" Running for 100 steps (viewer should open)...")
    for i in range(100):
        action = np.zeros(7)
        env.step(action)
        env.render()
    
    env.close()
    print(" Default scene works!\n")
    
except Exception as e:
    print(f" Default scene failed: {e}\n")

# Now test with custom scene using the EXACT same approach
print("2⃣ Testing custom scene:")

# Check if file exists first
custom_scene_path = Path(__file__).parent / "assets" / "base_robot" / "custom_scene.xml"
print(f"   Looking for: {custom_scene_path}")
print(f"   Exists: {custom_scene_path.exists()}")

if not custom_scene_path.exists():
    # Try alternative paths
    alt_paths = [
        Path(__file__).parent / "custom_scene.xml",
        Path(__file__).parent.parent / "envs" / "assets" / "base_robot" / "custom_scene.xml",
    ]
    for alt_path in alt_paths:
        print(f"   Trying: {alt_path}")
        if alt_path.exists():
            print(f"    Found at: {alt_path}")
            break

try:
    env = BaseRobot(
        model_path="../assets/base_robot/custom_scene.xml",  # Relative path like default
        render_mode="human",
        frame_skip=40,
    )
    print(" Custom scene BaseRobot created")
    
    obs = env.reset()[0]
    print(" Reset successful")
    
    print(" Running visualization (viewer should open)...")
    print("   Controls: Ctrl+scroll to zoom, drag to rotate")
    print("   Press ESC or close window to exit\n")
    
    step = 0
    while True:
        # Simple motion
        action = np.zeros(7)
        action[0] = 0.1 * np.sin(step * 0.02)
        
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        
        step += 1
        
        if step > 1000:  # Safety limit
            break
            
except KeyboardInterrupt:
    print("\n⏹  Stopped by user")
    
except Exception as e:
    print(f"\n Custom scene error: {e}")
    import traceback
    traceback.print_exc()
    
    print("\n Debugging suggestions:")
    print("1. Check that custom_scene.xml is in the correct location")
    print("2. Check that all mesh files referenced in the XML exist")
    print("3. Look for error messages about missing files above")
    
print("\n Test completed!")