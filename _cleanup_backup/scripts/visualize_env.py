import gymnasium as gym
import homestri_ur5e_rl
import matplotlib.pyplot as plt
import numpy as np

env = gym.make("BaseRobot-v0", render_mode="rgb_array")

plt.ion()
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_title("UR5e + Robotiq Gripper in MuJoCo")
ax.axis('off')

observation, info = env.reset(seed=42)
print("Environment initialized successfully!")
print(f"Observation shape: {observation.shape}")

# Run for a limited number of steps
for i in range(200):
    # Take random actions
    action = env.action_space.sample()
    
    # Step the environment
    observation, reward, terminated, truncated, info = env.step(action)
    
    # Render and display every few steps
    if i % 5 == 0:  # Update display every 5 steps
        frame = env.render()
        if frame is not None:
            ax.clear()
            ax.imshow(frame)
            ax.set_title(f"UR5e + Robotiq Gripper - Step {i}")
            ax.axis('off')
            plt.pause(0.01)  # Small pause to update display
    
    if terminated or truncated:
        observation, info = env.reset()
        print(f"Episode ended at step {i}, resetting...")

env.close()
plt.ioff()
plt.show()
print("Visualization completed!") 