#!/usr/bin/env python3
"""
Diagnostic script to identify and fix training issues
Run this to diagnose why your model isn't learning
"""

import numpy as np
import torch
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Dict, List, Optional

# Import your environment
from homestri_ur5e_rl.envs.UR5ePickPlaceEnvEnhanced import UR5ePickPlaceEnvEnhanced as UR5ePickPlaceIntegrated


class TrainingDiagnostics:
    """Diagnose and fix common training issues"""
    
    def __init__(self, model_path: Optional[str] = None, n_envs: int = 2):
        self.n_envs = n_envs
        
        # Create environment
        print("🔍 Creating diagnostic environment...")
        self.env = self._create_env()
        
        # Load or create model
        if model_path and Path(model_path).exists():
            print(f"📂 Loading model from {model_path}")
            self.model = PPO.load(model_path, env=self.env)
        else:
            print("🆕 Creating new model for testing")
            self.model = self._create_test_model()
    
    def _create_env(self):
        """Create environment with proper settings"""
        def make_env():
            env = UR5ePickPlaceIntegrated(
                control_mode="operational_space",
                reward_type="dense",
                use_domain_randomization=False,
                frame_skip=5,  # Keep default frame_skip to match render_fps
                camera_resolution=64,
            )
            return env
        
        # Create vectorized environment
        vec_env = DummyVecEnv([make_env for _ in range(self.n_envs)])
        
        # Add normalization
        vec_env = VecNormalize(
            vec_env,
            norm_obs=True,
            norm_reward=True,
            clip_obs=10.0,
            clip_reward=10.0,
        )
        
        return vec_env
    
    def _create_test_model(self):
        """Create a test model with proper hyperparameters"""
        return PPO(
            "MlpPolicy",
            self.env,
            learning_rate=3e-4,  # Proper learning rate
            n_steps=1024,
            batch_size=256,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            device='cpu',  # Force CPU to avoid device conflicts
            verbose=1,
        )
    
    def diagnose_reward_hacking(self, n_episodes: int = 10):
        """Check if model is exploiting rewards"""
        print("\n🎯 Diagnosing Reward Hacking...")
        
        rewards_per_episode = []
        task_progress_counts = {
            'approached': 0,
            'contacted': 0,
            'grasped': 0,
            'lifted': 0,
            'placed': 0,
        }
        
        for episode in range(n_episodes):
            obs = self.env.reset()
            episode_reward = 0
            done = np.array([False] * self.n_envs)
            step_count = 0
            
            while not done.any() and step_count < 100:
                action, _ = self.model.predict(obs, deterministic=False)
                obs, reward, done, info = self.env.step(action)
                episode_reward += reward[0] if isinstance(reward, np.ndarray) else reward
                step_count += 1
                
                # Check task progress from first environment
                if len(info) > 0 and isinstance(info[0], dict):
                    if 'task_progress' in info[0]:
                        for key, value in info[0]['task_progress'].items():
                            if value:
                                task_progress_counts[key] = task_progress_counts.get(key, 0) + 1
                    elif 'object_grasped' in info[0]:
                        if info[0]['object_grasped']:
                            task_progress_counts['grasped'] += 1
                    elif 'task_completed' in info[0]:
                        if info[0]['task_completed']:
                            task_progress_counts['placed'] += 1
            
            rewards_per_episode.append(episode_reward)
            print(f"   Episode {episode+1}: Reward={episode_reward:.2f}, Steps={step_count}")
        
        # Analyze results
        avg_reward = np.mean(rewards_per_episode)
        std_reward = np.std(rewards_per_episode)
        
        print(f"\n📊 Reward Statistics:")
        print(f"   Average: {avg_reward:.2f} ± {std_reward:.2f}")
        print(f"   Range: [{np.min(rewards_per_episode):.2f}, {np.max(rewards_per_episode):.2f}]")
        
        print(f"\n📈 Task Progress (across {n_episodes} episodes):")
        for task, count in task_progress_counts.items():
            percentage = (count / n_episodes) * 100
            print(f"   {task}: {count}/{n_episodes} ({percentage:.1f}%)")
        
        # Diagnosis
        if avg_reward > 100:
            print("\n⚠️ WARNING: Likely reward hacking detected!")
            print("   - Average reward is too high for 0% success")
            print("   - Model is exploiting distance rewards without completing tasks")
            print("   ✅ SOLUTION: Use the anti-hack reward structure")
        elif avg_reward < -50:
            print("\n⚠️ WARNING: Excessive penalties!")
            print("   - Model is being over-penalized")
            print("   ✅ SOLUTION: Reduce penalty magnitudes")
        else:
            print("\n✅ Reward structure appears balanced")
    
    def diagnose_action_diversity(self, n_steps: int = 100):
        """Check if actions are diverse enough"""
        print("\n🎲 Diagnosing Action Diversity...")
        
        obs = self.env.reset()
        actions = []
        
        for _ in range(n_steps):
            action, _ = self.model.predict(obs, deterministic=False)
            actions.append(action[0] if len(action.shape) > 1 else action)
            obs, _, done, _ = self.env.step(action)
            if done.any() if isinstance(done, np.ndarray) else done:
                obs = self.env.reset()
        
        actions = np.array(actions)
        
        # Analyze action statistics
        action_means = np.mean(actions, axis=0)
        action_stds = np.std(actions, axis=0)
        
        print(f"\n📊 Action Statistics (7 dimensions):")
        for i in range(min(7, actions.shape[1])):
            label = f"Joint {i}" if i < 6 else "Gripper"
            print(f"   {label}: mean={action_means[i]:.3f}, std={action_stds[i]:.3f}")
        
        # Check for stuck actions
        stuck_dims = np.where(action_stds < 0.01)[0]
        if len(stuck_dims) > 0:
            print(f"\n⚠️ WARNING: Actions stuck in dimensions {stuck_dims}")
            print("   - Model is not exploring action space")
            print("   ✅ SOLUTION: Increase entropy coefficient or learning rate")
        else:
            print("\n✅ Action diversity is healthy")
        
        # Visualize
        self._plot_action_distribution(actions)
    
    def diagnose_learning_progress(self, n_iterations: int = 5):
        """Check if model is actually learning"""
        print("\n📚 Diagnosing Learning Progress...")
        
        initial_obs = self.env.reset()
        
        # Get initial predictions
        initial_actions = []
        initial_values = []
        
        for _ in range(10):
            obs = self.env.reset()
            action, _ = self.model.predict(obs, deterministic=True)
            # Ensure obs tensor is on the same device as the model
            obs_tensor = torch.FloatTensor(obs).to(self.model.device)
            value = self.model.policy.predict_values(obs_tensor)[0]
            initial_actions.append(action)
            initial_values.append(value.item())
        
        print(f"   Initial avg value: {np.mean(initial_values):.3f}")
        
        # Train briefly
        print(f"   Training for {n_iterations * 1024} steps...")
        self.model.learn(total_timesteps=n_iterations * 1024, progress_bar=False)
        
        # Get post-training predictions
        final_actions = []
        final_values = []
        
        for _ in range(10):
            obs = self.env.reset()
            action, _ = self.model.predict(obs, deterministic=True)
            # Ensure obs tensor is on the same device as the model
            obs_tensor = torch.FloatTensor(obs).to(self.model.device)
            value = self.model.policy.predict_values(obs_tensor)[0]
            final_actions.append(action)
            final_values.append(value.item())
        
        print(f"   Final avg value: {np.mean(final_values):.3f}")
        
        # Check if anything changed
        value_change = np.mean(final_values) - np.mean(initial_values)
        action_change = np.mean(np.abs(np.array(final_actions) - np.array(initial_actions)))
        
        print(f"\n📊 Learning Metrics:")
        print(f"   Value change: {value_change:.3f}")
        print(f"   Action change: {action_change:.3f}")
        
        if abs(value_change) < 0.01 and action_change < 0.01:
            print("\n⚠️ WARNING: Model is not learning!")
            print("   Possible causes:")
            print("   - Learning rate too low")
            print("   - Gradient clipping too aggressive")
            print("   - Reward signal too sparse")
            print("   ✅ SOLUTION: Increase learning rate to 3e-4")
        else:
            print("\n✅ Model is learning")
    
    def diagnose_controller_response(self):
        """Test if homestri controllers are working"""
        print("\n🎮 Testing Controller Response...")
        
        obs = self.env.reset()
        
        # Test different action magnitudes - create actions for all environments
        test_actions = [
            np.array([0.1, 0, 0, 0, 0, 0, 0]),  # Small X movement
            np.array([0, 0.1, 0, 0, 0, 0, 0]),  # Small Y movement
            np.array([0, 0, 0.1, 0, 0, 0, 0]),  # Small Z movement
            np.array([0, 0, 0, 0, 0, 0, 1]),    # Close gripper
            np.array([0, 0, 0, 0, 0, 0, -1]),   # Open gripper
        ]
        
        for i, single_action in enumerate(test_actions):
            obs = self.env.reset()
            
            # Get initial end-effector position from first environment
            initial_ee_pos = self.env.envs[0].data.site_xpos[
                self.env.envs[0].ee_site_id
            ].copy()
            
            # Create action array for all environments (repeat the same action)
            action = np.tile(single_action, (self.n_envs, 1))
            obs, reward, done, info = self.env.step(action)
            
            # Get final position
            final_ee_pos = self.env.envs[0].data.site_xpos[
                self.env.envs[0].ee_site_id
            ].copy()
            
            movement = final_ee_pos - initial_ee_pos
            print(f"   Action {i}: Movement = {movement}")
        
        print("\n✅ Controller test complete")
    
    def _plot_action_distribution(self, actions: np.ndarray):
        """Plot action distribution"""
        fig, axes = plt.subplots(2, 4, figsize=(12, 6))
        axes = axes.flatten()
        
        for i in range(min(7, actions.shape[1])):
            ax = axes[i]
            ax.hist(actions[:, i], bins=30, alpha=0.7)
            label = f"Joint {i}" if i < 6 else "Gripper"
            ax.set_title(label)
            ax.set_xlabel("Action Value")
            ax.set_ylabel("Frequency")
        
        # Hide unused subplot
        if actions.shape[1] < 8:
            axes[7].axis('off')
        
        plt.tight_layout()
        plt.savefig("action_distribution.png")
        print("   📊 Action distribution saved to action_distribution.png")
    
    def run_full_diagnostic(self):
        """Run all diagnostics"""
        print("\n" + "="*60)
        print("🔬 FULL TRAINING DIAGNOSTIC")
        print("="*60)
        
        self.diagnose_reward_hacking()
        self.diagnose_action_diversity()
        self.diagnose_learning_progress()
        self.diagnose_controller_response()
        
        print("\n" + "="*60)
        print("📋 DIAGNOSTIC SUMMARY")
        print("="*60)
        
        print("\n🔧 Key Fixes to Apply:")
        print("1. Replace reward function with anti-hack version")
        print("2. Use homestri operational space controller")
        print("3. Set learning_rate=3e-4 (not 1e-5)")
        print("4. Set clip_range=0.2 (not 0.1)")
        print("5. Set n_epochs=10 (not 4)")
        print("6. Reduce frame_skip to 10 for more frequent updates")
        print("7. Use curriculum that advances on partial progress")
        
        print("\n💡 Expected Timeline After Fixes:")
        print("   - First movements: 100k steps")
        print("   - First contacts: 500k steps")
        print("   - First grasps: 2M steps")
        print("   - 70% success: 8-10M steps")


def main():
    """Main diagnostic entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Diagnose UR5e training issues")
    parser.add_argument("--model", type=str, help="Path to saved model")
    parser.add_argument("--envs", type=int, default=2, help="Number of parallel envs")
    
    args = parser.parse_args()
    
    # Run diagnostics
    diagnostics = TrainingDiagnostics(args.model, args.envs)
    diagnostics.run_full_diagnostic()


if __name__ == "__main__":
    main()