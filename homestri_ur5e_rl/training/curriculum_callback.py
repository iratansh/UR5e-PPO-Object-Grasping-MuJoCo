#!/usr/bin/env python3
"""
Curriculum Manager Integration Callback - FIXED
Properly integrates the CurriculumManager with training
"""

import numpy as np
from typing import Dict, Optional
import gymnasium as gym
from stable_baselines3.common.callbacks import BaseCallback

class CurriculumCallback(BaseCallback):
    """
    Callback that properly integrates curriculum manager with training
    """
    
    def __init__(
        self,
        curriculum_manager,
        check_freq: int = 512,  # Check every 512 steps (1 rollout)
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.curriculum_manager = curriculum_manager
        self.check_freq = check_freq
        
        # Tracking
        self.episode_successes = []
        self.episode_count = 0
        self.last_check_step = 0
        
    def _on_step(self) -> bool:
        return True
        
    def _on_rollout_end(self) -> None:
        """Called at the end of each rollout - perfect for curriculum updates"""
        if self.num_timesteps - self.last_check_step >= self.check_freq:
            self._update_curriculum()
            self.last_check_step = self.num_timesteps
    
    def _update_curriculum(self):
        """Update curriculum based on recent performance"""
        try:
            # Get recent success rate from training environment
            if hasattr(self.training_env, 'get_attr'):
                # Get success rates from all environments
                success_rates = self.training_env.get_attr('success_rate_tracker')
                
                # Calculate recent success rate
                if success_rates and success_rates[0] is not None:
                    recent_success_rate = np.mean([sr for sr in success_rates if sr is not None])
                else:
                    recent_success_rate = 0.0
            else:
                recent_success_rate = 0.0
            
            if self.verbose > 1:
                print(f"🎓 Curriculum check: {recent_success_rate:.1%} success rate")
            
            # Update curriculum manager
            curriculum_result = self.curriculum_manager.update(recent_success_rate)
            
            # Log phase changes
            if curriculum_result.get('phase_changed', False):
                old_phase = curriculum_result['old_phase']
                new_phase = curriculum_result['new_phase']
                duration = curriculum_result['phase_duration_hours']
                
                if self.verbose > 0:
                    print(f"\n🎓 CURRICULUM ADVANCEMENT!")
                    print(f"   {old_phase} → {new_phase}")
                    print(f"   Phase duration: {duration:.1f} hours")
                    print(f"   Success rate: {recent_success_rate:.1%}")
                    
                # Apply new phase settings to environments
                if hasattr(self.training_env, 'env_method'):
                    try:
                        # Update all training environments
                        self.training_env.env_method('_apply_curriculum_phase_settings')
                    except Exception as e:
                        if self.verbose > 0:
                            print(f"   Warning: Could not update environment settings: {e}")
                            
        except Exception as e:
            if self.verbose > 0:
                print(f"Warning: Curriculum update failed: {e}")

class SuccessTrackingCallback(BaseCallback):
    """
    Callback that tracks success rates for curriculum advancement
    """
    
    def __init__(
        self,
        window_size: int = 100,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.window_size = window_size
        self.success_history = []
        self.episode_count = 0
        
    def _on_step(self) -> bool:
        # Get info from the step
        infos = self.locals.get('infos', [])
        
        for info in infos:
            if isinstance(info, dict):
                # Check if episode ended
                if info.get('episode_ended', False) or info.get('_episode', False):
                    success = info.get('task_completed', False)
                    self.success_history.append(float(success))
                    self.episode_count += 1
                    
                    # Keep only recent history
                    if len(self.success_history) > self.window_size:
                        self.success_history = self.success_history[-self.window_size:]
                    
                    # Store in environment for curriculum callback to access
                    if hasattr(self.training_env, 'set_attr'):
                        recent_rate = np.mean(self.success_history) if self.success_history else 0.0
                        try:
                            self.training_env.set_attr('success_rate_tracker', recent_rate)
                        except:
                            pass
        
        return True