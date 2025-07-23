from setuptools import setup, find_packages

setup(
    name="homestri-ur5e-rl",
    version="0.1.0",
    description="UR5e reinforcement learning environment",
    packages=find_packages(),
    install_requires=[
        "gymnasium-robotics",
        "mujoco",
        "gymnasium", 
        "pynput",
        "numpy",
        "torch",
        "stable-baselines3",
        "wandb",
        "opencv-python",
        "pyyaml"
    ],
    python_requires=">=3.8",
) 