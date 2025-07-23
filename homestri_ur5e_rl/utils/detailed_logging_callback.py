"""
Detailed Logging Callback for Training Monitoring
"""

import numpy as np
import time
from stable_baselines3.common.callbacks import BaseCallback

class DetailedLoggingCallback(BaseCallback):
    """Callback for detailed training logs and debugging"""
    
    def __init__(self, log_freq: int = 2048, curriculum_manager=None, verbose: int = 1):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.last_log_time = time.time()
        self.curriculum_manager = curriculum_manager
        
        # Episode tracking
        self.episode_rewards = []
        self.episode_lengths = []
        self.success_count = 0
        self.total_episodes = 0
        
        # Performance tracking
        self.action_magnitudes = []
        self.collision_count = 0
        self.stuck_count = 0
        
    def _on_step(self) -> bool:
        # Collect episode data
        if len(self.locals.get('infos', [])) > 0:
            for info in self.locals['infos']:
                if 'episode' in info:
                    episode_reward = info['episode']['r']
                    episode_length = info['episode']['l']
                    
                    self.episode_rewards.append(episode_reward)
                    self.episode_lengths.append(episode_length)
                    self.total_episodes += 1
                    
                    # Track success rate
                    if 'task_completed' in info and info['task_completed']:
                        self.success_count += 1
                
                # Track issues
                if 'collision' in info and info['collision']:
                    self.collision_count += 1
                if 'stuck_detected' in info and info['stuck_detected']:
                    self.stuck_count += 1
        
        # Track action magnitudes
        if 'actions' in self.locals:
            actions = self.locals['actions']
            if hasattr(actions, 'numpy'):
                actions = actions.numpy()
            action_mag = np.linalg.norm(actions)
            self.action_magnitudes.append(action_mag)
        
        # Detailed logging
        if self.n_calls % self.log_freq == 0:
            self._log_detailed_metrics()
            
        return True
    
    def _log_detailed_metrics(self):
        """Log detailed training metrics"""
        current_time = time.time()
        time_elapsed = current_time - self.last_log_time
        steps_per_sec = self.log_freq / time_elapsed if time_elapsed > 0 else 0
        
        # Calculate metrics
        recent_rewards = self.episode_rewards[-50:] if self.episode_rewards else [0]
        recent_lengths = self.episode_lengths[-50:] if self.episode_lengths else [0]
        
        success_rate = self.success_count / max(self.total_episodes, 1) * 100
        reward_mean = np.mean(recent_rewards)
        reward_std = np.std(recent_rewards)
        
        print(f"\n{'='*70}")
        print(f" TRAINING PROGRESS - Step {self.n_calls:,}")
        print(f"{'='*70}")
        
        # Performance metrics
        print(f"⚡ Performance:")
        print(f"   Steps/sec: {steps_per_sec:.1f}")
        print(f"   Episodes: {self.total_episodes}")
        
        # Learning metrics
        print(f" Learning Progress:")
        print(f"   Success rate: {success_rate:.1f}% {'' if success_rate > 50 else '' if success_rate > 20 else ''}")
        print(f"   Avg reward: {reward_mean:.2f} ± {reward_std:.2f}")
        print(f"   Avg episode length: {np.mean(recent_lengths):.1f}")
        
        # Behavioral health
        print(f"🏥 Training Health:")
        if self.action_magnitudes:
            action_std = np.std(self.action_magnitudes[-100:])
            print(f"   Action diversity: {action_std:.3f} {'' if action_std > 0.1 else '  Low!'}")
        
        if self.total_episodes > 0:
            print(f"   Collision rate: {self.collision_count/self.total_episodes*100:.1f}%")
            print(f"   Stuck episodes: {self.stuck_count}")
        
        # Curriculum status
        if self.curriculum_manager:
            self.curriculum_manager.update(success_rate / 100)
            print(f"📚 Curriculum Status:")
            print(f"   Current phase: {self.curriculum_manager.current_phase}")
            print(f"   Phase focus: {self.curriculum_manager.phases[self.curriculum_manager.current_phase]['focus']}")
            print(f"   Success threshold: {self.curriculum_manager.phases[self.curriculum_manager.current_phase]['success_threshold']:.1%}")
            
            # Check if collision-aware curriculum is working
            terminate_on_collision = self.curriculum_manager.should_terminate_on_bad_collision()
            print(f"   Terminate on bad collision: {terminate_on_collision}")
        
        print(f"{'='*70}\n")
        
        self.last_log_time = current_time
        
        # Cleanup old data
        if len(self.episode_rewards) > 200:
            self.episode_rewards = self.episode_rewards[-100:]
            self.episode_lengths = self.episode_lengths[-100:]
        if len(self.action_magnitudes) > 2000:
            self.action_magnitudes = self.action_magnitudes[-1000:]