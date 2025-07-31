#!/usr/bin/env python3
"""
Progressive Training Callback for Curriculum Learning - FIXED
Gradually increases task difficulty and domain randomization
"""

import numpy as np
from typing import Dict, Optional
import gymnasium as gym
from stable_baselines3.common.callbacks import BaseCallback
import wandb

class ProgressiveTrainingCallback(BaseCallback):
    """
    Implements curriculum learning and progressive domain randomization - FIXED
    """
    
    def __init__(
        self,
        eval_env: Optional[gym.Env] = None,
        eval_freq: int = 25000,
        n_eval_episodes: int = 20,
        curriculum_threshold: float = 0.7,
        randomization_schedule: Optional[Dict] = None,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.curriculum_threshold = curriculum_threshold
        self.randomization_schedule = randomization_schedule or {
            0: 0.1,
            500000: 0.3,
            1500000: 0.6,
            3000000: 1.0,
        }
        
        # Tracking
        self.best_success_rate = 0.0
        self.current_curriculum_level = 0.1
        self.current_randomization_level = 0.1
        self.evaluation_history = []
        
        # Training phases
        self.training_phases = {
            'reaching': {'threshold': 0.5, 'complete': False},
            'grasping': {'threshold': 0.6, 'complete': False},
            'placing': {'threshold': 0.7, 'complete': False},
        }
        
    def _on_step(self) -> bool:
        # Update domain randomization
        self._update_domain_randomization()
        
        # Periodic evaluation
        if self.n_calls % self.eval_freq == 0:
            self._run_evaluation()
            
        return True
        
    def _update_domain_randomization(self):
        """Update domain randomization level based on schedule - FIXED"""
        current_timestep = self.num_timesteps
        new_level = 0.1
        
        for timestep, level in sorted(self.randomization_schedule.items()):
            if current_timestep >= timestep:
                new_level = level
                
        if new_level != self.current_randomization_level:
            self.current_randomization_level = new_level
            
            # FIXED: Robust environment method calling
            if hasattr(self.training_env, "env_method"):
                try:
                    # Check if the environment actually has the method before calling
                    if hasattr(self.training_env, "get_attr"):
                        # Test if any environment has the method
                        try:
                            test_attrs = self.training_env.get_attr("set_randomization_level", indices=[0])
                            if test_attrs and callable(test_attrs[0]):
                                self.training_env.env_method("set_randomization_level", new_level)
                        except (AttributeError, IndexError):
                            # Method doesn't exist, skip silently
                            pass
                    else:
                        # Fallback: try calling the method with error handling
                        self.training_env.env_method("set_randomization_level", new_level)
                except Exception as e:
                    if self.verbose > 0:
                        print(f"   Warning: Could not set randomization level: {e}")
                    pass
                    
            if self.verbose > 0:
                print(f"\n Domain randomization level: {new_level:.2f}")
                
            # Log to wandb
            if wandb.run is not None:
                wandb.log({
                    "training/randomization_level": new_level,
                }, step=self.num_timesteps)
                
    def _run_evaluation(self):
        """Run evaluation and update curriculum"""
        if self.eval_env is None:
            return
            
        print(f"\n Evaluation at {self.num_timesteps:,} steps...")
        
        # Evaluation metrics
        episode_rewards = []
        episode_successes = []
        episode_lengths = []
        grasp_successes = []
        camera_visibility_rates = []
        
        for episode in range(self.n_eval_episodes):
            obs = self.eval_env.reset()
            if isinstance(obs, tuple):
                obs = obs[0]
                
            episode_reward = 0
            episode_length = 0
            camera_visibilities = []
            
            done = False
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                step_result = self.eval_env.step(action)
                
                # Handle both old (4) and new (5) Gymnasium API
                if len(step_result) == 5:
                    obs, reward, terminated, truncated, info = step_result
                    done = terminated or truncated
                else:
                    obs, reward, done, info = step_result
                
                # Handle vectorized env
                if isinstance(done, np.ndarray):
                    done = done[0]
                if isinstance(reward, np.ndarray):
                    reward = reward[0]
                if isinstance(info, list):
                    info = info[0]
                    
                episode_reward += reward
                episode_length += 1
                
                # Track metrics
                camera_vis = info.get('camera_sees_object', False)
                camera_visibilities.append(camera_vis)
                
            # Record episode results
            success = info.get('task_completed', False)
            episode_rewards.append(episode_reward)
            episode_successes.append(float(success))
            episode_lengths.append(episode_length)
            camera_visibility_rates.append(np.mean(camera_visibilities))
            
            if info.get('object_grasped', False):
                grasp_successes.append(float(success))
                
        # Calculate metrics
        metrics = {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'success_rate': np.mean(episode_successes),
            'mean_episode_length': np.mean(episode_lengths),
            'camera_visibility_rate': np.mean(camera_visibility_rates),
            'grasp_success_rate': np.mean(grasp_successes) if grasp_successes else 0.0,
        }
        
        # Update curriculum
        self._update_curriculum(metrics['success_rate'])
        
        # Track best model
        if metrics['success_rate'] > self.best_success_rate:
            self.best_success_rate = metrics['success_rate']
            if self.verbose > 0:
                print(f" New best success rate: {self.best_success_rate:.2%}")
                
        # Log results
        if self.verbose > 0:
            print(f"   Success rate: {metrics['success_rate']:.2%}")
            print(f"   Mean reward: {metrics['mean_reward']:.2f} ± {metrics['std_reward']:.2f}")
            print(f"   Camera visibility: {metrics['camera_visibility_rate']:.2%}")
            print(f"   Curriculum level: {self.current_curriculum_level:.2f}")
            
        # Store history
        self.evaluation_history.append({
            'timestep': self.num_timesteps,
            'metrics': metrics,
            'curriculum_level': self.current_curriculum_level,
            'randomization_level': self.current_randomization_level,
        })
        
        # Log to wandb
        if wandb.run is not None:
            wandb.log({
                "eval/success_rate": metrics['success_rate'],
                "eval/mean_reward": metrics['mean_reward'],
                "eval/camera_visibility_rate": metrics['camera_visibility_rate'],
                "eval/best_success_rate": self.best_success_rate,
                "curriculum/level": self.current_curriculum_level,
            }, step=self.num_timesteps)
            
    def _update_curriculum(self, success_rate: float):
        """Update curriculum level based on performance"""
        # Check phase completion
        if not self.training_phases['reaching']['complete'] and success_rate > 0.5:
            self.training_phases['reaching']['complete'] = True
            print(" Phase 1 (Reaching) complete!")
            
        if not self.training_phases['grasping']['complete'] and success_rate > 0.6:
            self.training_phases['grasping']['complete'] = True
            print(" Phase 2 (Grasping) complete!")
            
        if not self.training_phases['placing']['complete'] and success_rate > 0.7:
            self.training_phases['placing']['complete'] = True
            print(" Phase 3 (Placing) complete!")
            
        # Update curriculum level
        if success_rate >= self.curriculum_threshold and self.current_curriculum_level < 1.0:
            old_level = self.current_curriculum_level
            self.current_curriculum_level = min(1.0, self.current_curriculum_level + 0.1)
            
            # FIXED: Robust curriculum level updating
            if hasattr(self.training_env, "env_method"):
                try:
                    # Check if the environment has the method
                    if hasattr(self.training_env, "get_attr"):
                        try:
                            test_attrs = self.training_env.get_attr("set_curriculum_level", indices=[0])
                            if test_attrs and callable(test_attrs[0]):
                                self.training_env.env_method("set_curriculum_level", self.current_curriculum_level)
                        except (AttributeError, IndexError):
                            pass
                    else:
                        self.training_env.env_method("set_curriculum_level", self.current_curriculum_level)
                except Exception as e:
                    if self.verbose > 0:
                        print(f"   Warning: Could not set curriculum level: {e}")
                    pass
                    
            if self.verbose > 0:
                print(f" Curriculum advanced: {old_level:.2f} → {self.current_curriculum_level:.2f}")
                
    def get_training_summary(self) -> Dict:
        """Get training summary"""
        if not self.evaluation_history:
            return {}
            
        latest_eval = self.evaluation_history[-1]['metrics']
        
        return {
            'final_success_rate': latest_eval['success_rate'],
            'best_success_rate': self.best_success_rate,
            'final_curriculum_level': self.current_curriculum_level,
            'final_randomization_level': self.current_randomization_level,
            'total_evaluations': len(self.evaluation_history),
            'camera_visibility_final': latest_eval['camera_visibility_rate'],
            'ready_for_real_robot': latest_eval['success_rate'] >= 0.7,
            'phases_completed': sum(1 for phase in self.training_phases.values() if phase['complete']),
        }