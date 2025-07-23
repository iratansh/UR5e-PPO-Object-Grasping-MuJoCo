#!/usr/bin/env python3
"""
Automated setup script for integrating custom components into homestri-ur5e-rl
Run this from the root of your homestri-ur5e-rl clone
"""

import os
import shutil
from pathlib import Path
import subprocess
import sys

def create_directories():
    """Create necessary directories"""
    dirs = [
        "homestri_ur5e_rl/training",
        "homestri_ur5e_rl/configs", 
        "homestri_ur5e_rl/envs/assets",
        "models",
        "logs",
        "tensorboard",
        "scripts",
    ]
    
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f" Created directory: {dir_path}")

def create_init_files():
    """Create or update __init__.py files"""
    
    # Training __init__.py
    training_init = """from homestri_ur5e_rl.training.realsense import RealSenseD435iSimulator
from homestri_ur5e_rl.training.ur5e_stuck_detection_mujoco import StuckDetectionMixin

__all__ = ["RealSenseD435iSimulator", "StuckDetectionMixin"]
"""
    
    with open("homestri_ur5e_rl/training/__init__.py", "w") as f:
        f.write(training_init)
    print(" Created homestri_ur5e_rl/training/__init__.py")
    
    # Update envs __init__.py
    envs_init_path = Path("homestri_ur5e_rl/envs/__init__.py")
    if envs_init_path.exists():
        with open(envs_init_path, "r") as f:
            content = f.read()
        
        # Add our environment if not already there
        if "UR5ePickPlaceEnvEnhanced" not in content:
            lines = content.strip().split('\n')
            # Add import
            lines.insert(0, "from homestri_ur5e_rl.envs.UR5ePickPlaceEnvEnhanced import UR5ePickPlaceEnvEnhanced")
            # Update __all__ if it exists
            for i, line in enumerate(lines):
                if line.startswith("__all__"):
                    if "UR5ePickPlaceEnvEnhanced" not in line:
                        # Add to existing __all__
                        lines[i] = line.rstrip(']') + ', "UR5ePickPlaceEnvEnhanced"]'
                    break
            else:
                # No __all__ found, add one
                lines.append('__all__ = ["UR5ePickPlaceEnvEnhanced"]')
            
            with open(envs_init_path, "w") as f:
                f.write('\n'.join(lines) + '\n')
            print(" Updated homestri_ur5e_rl/envs/__init__.py")
    else:
        with open(envs_init_path, "w") as f:
            f.write('from homestri_ur5e_rl.envs.UR5ePickPlaceEnvEnhanced import UR5ePickPlaceEnvEnhanced\n\n')
            f.write('__all__ = ["UR5ePickPlaceEnvEnhanced"]\n')
        print(" Created homestri_ur5e_rl/envs/__init__.py")

def create_main_training_script():
    """Create main training entry point script"""
    
    script_content = '''#!/usr/bin/env python3
"""
Main entry point for sim-to-real training
"""

import sys
from pathlib import Path

# Add the package to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from homestri_ur5e_rl.training.training_script_integrated import main

if __name__ == "__main__":
    main()
'''
    
    script_path = Path("scripts/train_sim2real.py")
    with open(script_path, "w") as f:
        f.write(script_content)
    
    # Make executable
    script_path.chmod(0o755)
    print(" Created scripts/train_sim2real.py")

def check_dependencies():
    """Check and install missing dependencies"""
    print("\n📦 Checking dependencies...")
    
    required_packages = {
        "wandb": "wandb",
        "opencv-python": "cv2",
        "pyyaml": "yaml",
        "mujoco": "mujoco",
        "gymnasium": "gymnasium",
        "stable-baselines3": "stable_baselines3",
    }
    
    missing_packages = []
    
    for package, import_name in required_packages.items():
        try:
            __import__(import_name)
            print(f" {package} is installed")
        except ImportError:
            print(f" {package} is missing")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n📦 Installing missing packages: {', '.join(missing_packages)}")
        subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing_packages)
        print(" All dependencies installed")
    else:
        print(" All dependencies are already installed")

def create_example_files():
    """Create example configuration and test files"""
    
    # Minimal test script
    test_script = '''#!/usr/bin/env python3
"""Quick test of the enhanced environment"""

from homestri_ur5e_rl.envs import UR5ePickPlaceEnvEnhanced

print("Testing environment creation...")
env = UR5ePickPlaceEnvEnhanced(
    render_mode="human",
    camera_resolution=84,  # Lower resolution for testing
)

print(f" Environment created successfully!")
print(f"   Observation space: {env.observation_space.shape}")
print(f"   Action space: {env.action_space.shape}")

obs, info = env.reset()
print(f" Environment reset successfully!")

print("\\nRunning 100 test steps...")
for i in range(100):
    action = env.action_space.sample() * 0.1  # Small actions
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    
    if terminated or truncated:
        print(f"   Episode ended at step {i}")
        break

env.close()
print(" Test completed!")
'''
    
    with open("test_env.py", "w") as f:
        f.write(test_script)
    Path("test_env.py").chmod(0o755)
    print(" Created test_env.py")

def print_file_placement_guide():
    """Print guide for where to place user's files"""
    
    print("\n" + "="*60)
    print("📁 FILE PLACEMENT GUIDE")
    print("="*60)
    print("\nNow place your files in the following locations:")
    print("\n1. ENVIRONMENT FILES:")
    print("   - UR5ePickPlaceEnvEnhanced.py → homestri_ur5e_rl/envs/")
    print("   - custom_scene.xml → homestri_ur5e_rl/envs/assets/")
    
    print("\n2. TRAINING COMPONENTS:")
    print("   - realsense.py → homestri_ur5e_rl/training/")
    print("   - ur5e_stuck_detection_mujoco.py → homestri_ur5e_rl/training/")
    print("   - training_script_integrated.py → homestri_ur5e_rl/training/")
    
    print("\n3. UTILITIES:")
    print("   - pid_controller_utils.py → homestri_ur5e_rl/utils/")
    print("   - solver_utils.py → homestri_ur5e_rl/utils/")
    
    print("\n4. CONFIGURATION:")
    print("   - config_enhanced.yaml → homestri_ur5e_rl/configs/")
    
    print("\n" + "="*60)
    print(" NEXT STEPS:")
    print("="*60)
    print("\n1. Place all your files in the locations above")
    print("2. Test the environment: python test_env.py")
    print("3. Start training: python scripts/train_sim2real.py train")
    print("4. Monitor with: tensorboard --logdir tensorboard/")
    print("\n" + "="*60)

def main():
    """Main setup function"""
    print(" Setting up homestri-ur5e-rl integration...")
    print("="*60)
    
    # Check if we're in the right directory
    if not Path("homestri_ur5e_rl").exists():
        print(" Error: This script must be run from the root of homestri-ur5e-rl repository")
        print("   Current directory:", os.getcwd())
        print("   Please cd to your homestri-ur5e-rl clone and run again.")
        sys.exit(1)
    
    # Run setup steps
    create_directories()
    create_init_files()
    create_main_training_script()
    check_dependencies()
    create_example_files()
    
    # Print final instructions
    print_file_placement_guide()
    
    print("\n Setup complete! Follow the file placement guide above.")

if __name__ == "__main__":
    main()