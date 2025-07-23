import gymnasium as gym
import homestri_ur5e_rl
import matplotlib.pyplot as plt
import numpy as np

env = gym.make("CustomRobot-v0", render_mode="rgb_array")

plt.ion()
fig, ax = plt.subplots(figsize=(10, 8))
ax.set_title("Custom UR5e + Robotiq Gripper Environment")
ax.axis('off')

observation, info = env.reset(seed=42)
print("Custom environment initialized successfully!")
print(f"Observation type: {type(observation)}")
if hasattr(observation, 'shape'):
    print(f"Observation shape: {observation.shape}")
else:
    print(f"Observation: {observation}")
print(f"Action space: {env.action_space}")

# Print some useful information about the environment
print("\nEnvironment Information:")
print(f"- UR5e robot with 6 DOF arm")
print(f"- Robotiq 2F-85 gripper")
print(f"- Table with graspable objects (cube, sphere, cylinder)")
print(f"- RealSense camera for visual feedback")

# Get initial object positions
if hasattr(env.unwrapped, 'get_object_positions'):
    object_positions = env.unwrapped.get_object_positions()
    print(f"\nInitial object positions:")
    for obj_name, pos in object_positions.items():
        print(f"  {obj_name}: {pos}")

# Get end effector position
if hasattr(env.unwrapped, 'get_end_effector_position'):
    eef_pos = env.unwrapped.get_end_effector_position()
    print(f"\nEnd effector position: {eef_pos}")

# Run for a limited number of steps
print("\nRunning simulation...")
for i in range(300):
    # Take random actions (smaller actions for more controlled movement)
    action = env.action_space.sample() * 0.3  # Scale down for gentler movements
    
    # Step the environment
    observation, reward, terminated, truncated, info = env.step(action)
    
    # Render and display every few steps
    if i % 10 == 0:  # Update display every 10 steps
        frame = env.render()
        if frame is not None:
            ax.clear()
            ax.imshow(frame)
            ax.set_title(f"Custom UR5e Environment - Step {i}")
            ax.axis('off')
            plt.pause(0.05)  # Small pause to update display
    
    # Print some debug information occasionally
    if i % 50 == 0:
        if hasattr(env.unwrapped, 'get_end_effector_position'):
            eef_pos = env.unwrapped.get_end_effector_position()
            print(f"Step {i}: End effector at {eef_pos}")
    
    if terminated or truncated:
        observation, info = env.reset()
        print(f"Episode ended at step {i}, resetting...")

env.close()
plt.ioff()
plt.show()
print("Custom environment visualization completed!")