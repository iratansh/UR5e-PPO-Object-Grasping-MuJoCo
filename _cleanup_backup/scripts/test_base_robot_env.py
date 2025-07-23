import gymnasium as gym
import homestri_ur5e_rl
from homestri_ur5e_rl.input_devices.keyboard_input import (
    KeyboardInput,
)

keyboard_input = KeyboardInput()

# Use rgb_array rendering instead of human to avoid OpenGL context issues
env = gym.make("BaseRobot-v0", render_mode="rgb_array")

observation, info = env.reset(seed=42)
print("Environment initialized successfully!")
print(f"Observation shape: {observation.shape}")
print(f"Action space: {env.action_space}")

for i in range(100):  # Run for 100 steps
    action = env.action_space.sample()
    
    # You can uncomment this line if you want to use keyboard input
    # action = keyboard_input.get_action()

    observation, reward, terminated, truncated, info = env.step(action)
    
    # Render and optionally save the frame
    if i % 10 == 0:  # Render every 10 steps
        frame = env.render()
        if frame is not None:
            print(f"Step {i}: Rendered frame shape: {frame.shape}")

    if terminated or truncated:
        observation, info = env.reset()
        print(f"Episode ended at step {i}, resetting...")

env.close()
print("Test completed successfully!")
