"""
Test Trained Model with Visual Playback
Watch your trained robot perform after training!
"""

import os
import sys
from pathlib import Path
import argparse

# Add homestri to path
sys.path.insert(0, str(Path(__file__).parent))

from homestri_ur5e_rl.training.training_script_integrated import IntegratedTrainer

def find_latest_model():
    """Find the most recent trained model"""
    experiments_dir = Path("experiments")
    if not experiments_dir.exists():
        return None
    
    # Get all experiment directories
    exp_dirs = [d for d in experiments_dir.iterdir() if d.is_dir()]
    if not exp_dirs:
        return None
    
    # Find the most recent one with a best_model
    for exp_dir in sorted(exp_dirs, key=lambda d: d.stat().st_mtime, reverse=True):
        best_model = exp_dir / "best_model"
        if best_model.exists():
            return best_model
    
    return None

def main():
    """Test a trained model with visual playback"""
    parser = argparse.ArgumentParser(description="Test Trained UR5e Model")
    parser.add_argument("--model", type=str, help="Path to trained model")
    parser.add_argument("--episodes", type=int, default=5, help="Number of test episodes")
    
    args = parser.parse_args()
    
    print("UR5e Model Testing with Visual Playback")
    print("="*50)
    
    # Find model to test
    if args.model:
        model_path = Path(args.model)
    else:
        model_path = find_latest_model()
        
    if not model_path or not model_path.exists():
        print(" No trained model found!")
        print(" Train a model first with: python headless_training.py")
        return
    
    print(f"Testing model: {model_path}")
    print(f"Running {args.episodes} episodes with visual playback")
    print(f" You can navigate the 3D scene with mouse")
    print("="*50)
    
    trainer = IntegratedTrainer()
    
    try:
        trainer.test_model(str(model_path), args.episodes)
    except Exception as e:
        print(f" Testing failed: {e}")
        raise

if __name__ == "__main__":
    main()
