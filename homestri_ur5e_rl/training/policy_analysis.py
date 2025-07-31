#!/usr/bin/env python3
"""
Policy Analysis and Visualization Tool
Helps debug and understand learned behaviors
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import pandas as pd
from typing import Dict, List, Optional
import cv2
from tqdm import tqdm

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

from homestri_ur5e_rl.envs import UR5ePickPlaceEnvEnhanced

class PolicyAnalyzer:
    """Analyze and visualize trained policies"""
    
    def __init__(self, model_path: str, env_config: Optional[Dict] = None):
        self.model_path = Path(model_path)
        
        # Load model
        self.model = PPO.load(str(self.model_path / "best_model.zip"))
        
        # Load normalization
        vec_norm_path = self.model_path / "vec_normalize.pkl"
        if vec_norm_path.exists():
            self.vec_normalize = VecNormalize.load(str(vec_norm_path))
            self.vec_normalize.training = False
        else:
            self.vec_normalize = None
            
        if env_config is None:
            env_config = {
                "camera_resolution": 128,
                "render_mode": "rgb_array",
                "use_domain_randomization": False,  # Deterministic for analysis
            }
            
        self.env = UR5ePickPlaceEnvEnhanced(**env_config)
        
        # Analysis results
        self.trajectories = []
        self.attention_maps = []
        self.action_distributions = []
        
    def collect_trajectories(self, n_episodes: int = 10, render: bool = False) -> List[Dict]:
        """Collect trajectories for analysis"""
        
        print(f" Collecting {n_episodes} trajectories...")
        
        trajectories = []
        
        for episode in tqdm(range(n_episodes)):
            trajectory = {
                'observations': [],
                'actions': [],
                'rewards': [],
                'values': [],
                'action_probs': [],
                'features': [],
                'infos': [],
                'images': [],
            }
            
            obs, info = self.env.reset()
            done = False
            
            while not done:
                # Get action and additional info
                if self.vec_normalize:
                    norm_obs = self.vec_normalize.normalize_obs(obs)
                else:
                    norm_obs = obs
                    
                # Get policy outputs
                with torch.no_grad():
                    obs_tensor = torch.FloatTensor(norm_obs).unsqueeze(0)
                    
                    # Get features from CNN
                    features = self.model.policy.features_extractor(obs_tensor)
                    trajectory['features'].append(features.numpy())
                    
                    # Get action distribution
                    action_dist = self.model.policy.get_distribution(obs_tensor)
                    action, log_prob = action_dist.sample(), action_dist.log_prob(action_dist.sample())
                    
                    # Get value estimate
                    value = self.model.policy.value_net(features)
                    
                # Store data
                trajectory['observations'].append(obs)
                trajectory['actions'].append(action.numpy()[0])
                trajectory['values'].append(value.item())
                trajectory['action_probs'].append(torch.exp(log_prob).numpy())
                
                if render:
                    img = self.env.render()
                    trajectory['images'].append(img)
                    
                # Step environment
                obs, reward, terminated, truncated, info = self.env.step(action.numpy()[0])
                done = terminated or truncated
                
                trajectory['rewards'].append(reward)
                trajectory['infos'].append(info)
                
            # Convert lists to arrays
            for key in ['observations', 'actions', 'rewards', 'values']:
                if key in trajectory and trajectory[key]:
                    trajectory[key] = np.array(trajectory[key])
                    
            trajectory['success'] = info.get('task_completed', False)
            trajectory['episode_length'] = len(trajectory['rewards'])
            trajectory['total_reward'] = np.sum(trajectory['rewards'])
            
            trajectories.append(trajectory)
            
        self.trajectories = trajectories
        return trajectories
        
    def analyze_action_distribution(self):
        """Analyze action discurriculum_managertribution across episodes"""
        
        all_actions = []
        for traj in self.trajectories:
            all_actions.append(traj['actions'])
            
        all_actions = np.vstack(all_actions)
        
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()
        
        action_names = [
            'Shoulder Pan', 'Shoulder Lift', 'Elbow',
            'Wrist 1', 'Wrist 2', 'Wrist 3', 'Gripper'
        ]
        
        for i in range(7):
            ax = axes[i]
            ax.hist(all_actions[:, i], bins=50, alpha=0.7, edgecolor='black')
            ax.set_title(f'{action_names[i]} Actions')
            ax.set_xlabel('Action Value')
            ax.set_ylabel('Frequency')
            ax.axvline(0, color='red', linestyle='--', alpha=0.5)
            
        # Overall statistics
        axes[7].axis('off')
        stats_text = f"Action Statistics:\n"
        stats_text += f"Total actions: {len(all_actions)}\n"
        stats_text += f"Episodes: {len(self.trajectories)}\n"
        stats_text += f"Success rate: {np.mean([t['success'] for t in self.trajectories]):.2%}\n"
        stats_text += f"Avg episode length: {np.mean([t['episode_length'] for t in self.trajectories]):.1f}\n"
        stats_text += f"Avg reward: {np.mean([t['total_reward'] for t in self.trajectories]):.2f}"
        axes[7].text(0.1, 0.5, stats_text, fontsize=12, verticalalignment='center')
        
        plt.tight_layout()
        plt.savefig(self.model_path / 'action_distribution.png', dpi=300)
        plt.show()
        
        return all_actions
        
    def visualize_value_function(self):
        """Visualize value function across episodes"""
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Value over time for successful vs failed episodes
        ax = axes[0, 0]
        for traj in self.trajectories:
            color = 'green' if traj['success'] else 'red'
            alpha = 0.7 if traj['success'] else 0.3
            ax.plot(traj['values'], color=color, alpha=alpha)
        ax.set_title('Value Function Over Episode')
        ax.set_xlabel('Step')
        ax.set_ylabel('Value Estimate')
        ax.legend(['Success', 'Failure'])
        
        # Average value by phase
        ax = axes[0, 1]
        phase_values = {'reaching': [], 'grasping': [], 'lifting': [], 'placing': []}
        
        for traj in self.trajectories:
            # Simple phase detection based on gripper and height
            for i, info in enumerate(traj['infos']):
                if not info.get('object_grasped', False):
                    phase = 'reaching'
                elif i < len(traj['infos']) - 20:  # Not near end
                    phase = 'lifting'
                else:
                    phase = 'placing'
                    
                if i < len(traj['values']):
                    phase_values[phase].append(traj['values'][i])
                    
        # Box plot of values by phase
        phase_data = [phase_values[p] for p in ['reaching', 'grasping', 'lifting', 'placing']]
        ax.boxplot(phase_data, labels=['Reaching', 'Grasping', 'Lifting', 'Placing'])
        ax.set_title('Value Estimates by Task Phase')
        ax.set_ylabel('Value')
        
        # Reward vs Value correlation
        ax = axes[1, 0]
        all_rewards = []
        all_values = []
        for traj in self.trajectories:
            all_rewards.extend(traj['rewards'])
            all_values.extend(traj['values'][:-1])  # Exclude last value
            
        ax.scatter(all_rewards, all_values, alpha=0.5)
        ax.set_title('Reward vs Value Correlation')
        ax.set_xlabel('Actual Reward')
        ax.set_ylabel('Value Estimate')
        
        # Learning curves if available
        ax = axes[1, 1]
        eval_results_path = self.model_path.parent / 'eval' / 'evaluations.npz'
        if eval_results_path.exists():
            data = np.load(eval_results_path)
            timesteps = data['timesteps']
            results = data['results']
            
            mean_rewards = np.mean(results, axis=1)
            std_rewards = np.std(results, axis=1)
            
            ax.plot(timesteps, mean_rewards)
            ax.fill_between(timesteps, 
                          mean_rewards - std_rewards,
                          mean_rewards + std_rewards,
                          alpha=0.3)
            ax.set_title('Training Progress')
            ax.set_xlabel('Timesteps')
            ax.set_ylabel('Mean Reward')
        else:
            ax.text(0.5, 0.5, 'No evaluation data found', 
                   ha='center', va='center', transform=ax.transAxes)
            
        plt.tight_layout()
        plt.savefig(self.model_path / 'value_function_analysis.png', dpi=300)
        plt.show()
        
    def analyze_failure_modes(self):
        """Analyze common failure modes"""
        
        failures = [t for t in self.trajectories if not t['success']]
        
        if not failures:
            print("No failures to analyze!")
            return
            
        failure_reasons = {
            'timeout': 0,
            'dropped': 0,
            'missed_grasp': 0,
            'stuck': 0,
            'out_of_bounds': 0,
            'other': 0,
        }
        
        failure_details = []
        
        for traj in failures:
            # Analyze final state
            final_info = traj['infos'][-1]
            
            detail = {
                'episode_length': traj['episode_length'],
                'total_reward': traj['total_reward'],
                'final_info': final_info,
            }
            
            # Classify failure
            if traj['episode_length'] >= 999:
                failure_reasons['timeout'] += 1
                detail['reason'] = 'timeout'
            elif final_info.get('stuck', False):
                failure_reasons['stuck'] += 1
                detail['reason'] = 'stuck'
            elif not any(info.get('object_grasped', False) for info in traj['infos']):
                failure_reasons['missed_grasp'] += 1
                detail['reason'] = 'missed_grasp'
            else:
                failure_reasons['other'] += 1
                detail['reason'] = 'other'
                
            failure_details.append(detail)
            
        # Visualize failure modes
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Pie chart of failure reasons
        ax = axes[0]
        reasons = list(failure_reasons.keys())
        counts = list(failure_reasons.values())
        ax.pie(counts, labels=reasons, autopct='%1.1f%%')
        ax.set_title('Failure Mode Distribution')
        
        # Failure timeline
        ax = axes[1]
        for detail in failure_details:
            color = {
                'timeout': 'red',
                'stuck': 'orange',
                'missed_grasp': 'yellow',
                'other': 'gray'
            }.get(detail['reason'], 'gray')
            
            ax.scatter(detail['episode_length'], detail['total_reward'],
                      color=color, alpha=0.6, s=100)
            
        ax.set_xlabel('Episode Length')
        ax.set_ylabel('Total Reward')
        ax.set_title('Failure Characteristics')
        ax.legend(list(set(d['reason'] for d in failure_details)))
        
        plt.tight_layout()
        plt.savefig(self.model_path / 'failure_analysis.png', dpi=300)
        plt.show()
        
        return failure_details
        
    def visualize_attention(self, episode_idx: int = 0):
        """Visualize what the CNN is attending to"""
        
        if episode_idx >= len(self.trajectories):
            print(f"Episode {episode_idx} not found")
            return
            
        traj = self.trajectories[episode_idx]
        
        # Get a few key frames
        key_frames = [0, len(traj['observations'])//4, len(traj['observations'])//2, -1]
        
        fig, axes = plt.subplots(len(key_frames), 4, figsize=(16, 4*len(key_frames)))
        
        for i, frame_idx in enumerate(key_frames):
            obs = traj['observations'][frame_idx]
            
            # Extract camera data
            camera_start = 26 + 13 + 3  # After robot state, object state, goal
            camera_data = obs[camera_start:].reshape(128, 128, 4)
            
            # RGB image
            rgb = camera_data[:, :, :3]
            axes[i, 0].imshow(rgb)
            axes[i, 0].set_title(f'RGB (Step {frame_idx})')
            axes[i, 0].axis('off')
            
            # Depth image
            depth = camera_data[:, :, 3]
            axes[i, 1].imshow(depth, cmap='viridis')
            axes[i, 1].set_title('Depth')
            axes[i, 1].axis('off')
            
            # Action taken
            if frame_idx < len(traj['actions']):
                action = traj['actions'][frame_idx]
                action_str = f"J: [{action[0]:.2f}, {action[1]:.2f}, {action[2]:.2f}]\n"
                action_str += f"   [{action[3]:.2f}, {action[4]:.2f}, {action[5]:.2f}]\n"
                action_str += f"G: {action[6]:.2f}"
                axes[i, 2].text(0.1, 0.5, action_str, fontsize=10,
                              transform=axes[i, 2].transAxes,
                              verticalalignment='center')
                axes[i, 2].set_title('Action')
                axes[i, 2].axis('off')
                
            # Info
            if frame_idx < len(traj['infos']):
                info = traj['infos'][frame_idx]
                info_str = f"Grasped: {info.get('object_grasped', False)}\n"
                info_str += f"Reward: {traj['rewards'][frame_idx]:.3f}\n"
                info_str += f"Value: {traj['values'][frame_idx]:.3f}"
                axes[i, 3].text(0.1, 0.5, info_str, fontsize=10,
                              transform=axes[i, 3].transAxes,
                              verticalalignment='center')
                axes[i, 3].set_title('State Info')
                axes[i, 3].axis('off')
                
        plt.suptitle(f"Episode {episode_idx} - Success: {traj['success']}")
        plt.tight_layout()
        plt.savefig(self.model_path / f'episode_{episode_idx}_frames.png', dpi=300)
        plt.show()
        
    def create_summary_video(self, episode_idx: int = 0, output_path: Optional[str] = None):
        """Create a summary video of an episode"""
        
        if episode_idx >= len(self.trajectories):
            print(f"Episode {episode_idx} not found")
            return
            
        traj = self.trajectories[episode_idx]
        
        if not traj['images']:
            print("No images found. Re-run collect_trajectories with render=True")
            return
            
        if output_path is None:
            output_path = self.model_path / f'episode_{episode_idx}_summary.mp4'
            
        height, width = traj['images'][0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, 10.0, (width, height))
        
        for i, img in enumerate(traj['images']):
            # Add overlay information
            img_copy = img.copy()
            
            # Add text overlay
            info_text = f"Step: {i}"
            if i < len(traj['rewards']):
                info_text += f" | Reward: {traj['rewards'][i]:.2f}"
            if i < len(traj['values']):
                info_text += f" | Value: {traj['values'][i]:.2f}"
                
            cv2.putText(img_copy, info_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Add action visualization
            if i < len(traj['actions']):
                action = traj['actions'][i]
                action_text = f"Action: [{action[0]:.2f}, {action[1]:.2f}, {action[2]:.2f}]"
                cv2.putText(img_copy, action_text, (10, height - 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                gripper_text = f"Gripper: {'CLOSE' if action[6] > 0 else 'OPEN'}"
                cv2.putText(img_copy, gripper_text, (10, height - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
            # Convert RGB to BGR for OpenCV
            img_bgr = cv2.cvtColor(img_copy, cv2.COLOR_RGB2BGR)
            out.write(img_bgr)
            
        out.release()
        print(f" Video saved to: {output_path}")
        
    def generate_report(self):
        """Generate comprehensive analysis report"""
        
        report = {
            'model_path': str(self.model_path),
            'n_episodes_analyzed': len(self.trajectories),
            'success_rate': np.mean([t['success'] for t in self.trajectories]),
            'avg_episode_length': np.mean([t['episode_length'] for t in self.trajectories]),
            'avg_total_reward': np.mean([t['total_reward'] for t in self.trajectories]),
        }
        
        # Success vs failure statistics
        successes = [t for t in self.trajectories if t['success']]
        failures = [t for t in self.trajectories if not t['success']]
        
        if successes:
            report['success_stats'] = {
                'count': len(successes),
                'avg_length': np.mean([t['episode_length'] for t in successes]),
                'avg_reward': np.mean([t['total_reward'] for t in successes]),
                'min_length': min(t['episode_length'] for t in successes),
                'max_length': max(t['episode_length'] for t in successes),
            }
            
        if failures:
            report['failure_stats'] = {
                'count': len(failures),
                'avg_length': np.mean([t['episode_length'] for t in failures]),
                'avg_reward': np.mean([t['total_reward'] for t in failures]),
            }
            
        # Save report
        report_path = self.model_path / 'analysis_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
            
        print(f" Analysis Report:")
        print(f"   Success Rate: {report['success_rate']:.2%}")
        print(f"   Avg Episode Length: {report['avg_episode_length']:.1f}")
        print(f"   Avg Total Reward: {report['avg_total_reward']:.2f}")
        print(f"\n📁 Full report saved to: {report_path}")
        
        return report

def main():
    """Main entry point for policy analysis"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze trained UR5e policies")
    parser.add_argument("--model", type=str, required=True,
                       help="Path to model directory")
    parser.add_argument("--episodes", type=int, default=20,
                       help="Number of episodes to analyze")
    parser.add_argument("--render", action="store_true",
                       help="Render episodes (slower)")
    parser.add_argument("--video", action="store_true",
                       help="Create summary videos")
    
    args = parser.parse_args()
    
    print("🔍 Policy Analysis Tool")
    print("="*60)
    
    analyzer = PolicyAnalyzer(args.model)
    
    # Collect trajectories
    trajectories = analyzer.collect_trajectories(
        n_episodes=args.episodes,
        render=args.render or args.video
    )
    
    # Run analyses
    print("\n Running analyses...")
    
    # Action distribution
    print("  - Analyzing action distribution...")
    analyzer.analyze_action_distribution()
    
    # Value function
    print("  - Analyzing value function...")
    analyzer.visualize_value_function()
    
    # Failure modes
    print("  - Analyzing failure modes...")
    analyzer.analyze_failure_modes()
    
    # Attention visualization
    print("  - Visualizing attention...")
    analyzer.visualize_attention(0)  # First episode
    
    if args.video:
        print("  - Creating summary videos...")
        rewards = [t['total_reward'] for t in trajectories]
        best_idx = np.argmax(rewards)
        worst_idx = np.argmin(rewards)
        
        analyzer.create_summary_video(best_idx)
        analyzer.create_summary_video(worst_idx)
        
    # Generate report
    print("  - Generating report...")
    report = analyzer.generate_report()
    
    print("\n Analysis complete!")

if __name__ == "__main__":
    main()