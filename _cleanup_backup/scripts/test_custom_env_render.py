import gymnasium as gym
import homestri_ur5e_rl
import matplotlib.pyplot as plt
import numpy as np

env = gym.make("CustomRobot-v0", render_mode="rgb_array")

print("Creating environment...")
observation, info = env.reset(seed=42)

print("Environment initialized successfully!")
print("Attempting to render the environment without stepping...")

# Try to render the initial state
frame = env.render()
if frame is not None:
    print(f"Successfully rendered frame with shape: {frame.shape}")
    
    plt.figure(figsize=(10, 8))
    plt.imshow(frame)
    plt.title("Custom UR5e Environment - Initial State")
    plt.axis('off')
    plt.savefig('/Users/ishaanratanshi/homestri-ur5e-rl/custom_env_initial_render.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("Environment rendering successful!")
    print("Your custom environment contains:")
    print("- UR5e robot with 6 DOF arm")
    print("- Robotiq 2F-85 gripper") 
    print("- Table with graspable objects")
    print("- RealSense camera")
    print("- All components should be visible in the rendered image")
else:
    print("Failed to render the environment")

env.close()
print("Test completed!")