"""
Performance-based Early Stopping Callback to Prevent Catastrophic Forgetting
"""

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from typing import List, Optional
import time

class PerformanceEarlyStoppingCallback(BaseCallback):
    """
    Callback that stops training when performance degrades significantly
    to prevent catastrophic forgetting
    """
    
    def __init__(
        self,
        patience: int = 5,  # Number of evaluations to wait for improvement
        min_delta: float = 0.02,  # Minimum improvement threshold (2%)
        performance_window: int = 10,  # Window for performance averaging
        degradation_threshold: float = 0.05,  # Significant degradation threshold (5%)
        evaluation_freq: int = 16000,  # How often to check (should match eval_freq)
        verbose: int = 1
    ):
        super().__init__(verbose)
        self.patience = patience
        self.min_delta = min_delta
        self.performance_window = performance_window
        self.degradation_threshold = degradation_threshold
        self.evaluation_freq = evaluation_freq
        
        # Performance tracking
        self.success_rates: List[float] = []
        self.best_performance = 0.0
        self.best_performance_step = 0
        self.patience_counter = 0
        self.last_eval_step = 0
        
        # Distance tracking for approach phase
        self.median_distances: List[float] = []
        self.best_median_distance = float('inf')
        
        # Contact rate tracking
        self.contact_rates: List[float] = []
        self.best_contact_rate = 0.0
        
    def _on_step(self) -> bool:
        # Only check at evaluation intervals
        if self.num_timesteps - self.last_eval_step < self.evaluation_freq:
            return True
            
        self.last_eval_step = self.num_timesteps
        
        # Get current performance metrics from training logs
        current_success_rate = self._get_current_success_rate()
        current_median_distance = self._get_current_median_distance()
        current_contact_rate = self._get_current_contact_rate()
        
        # Store metrics
        self.success_rates.append(current_success_rate)
        self.median_distances.append(current_median_distance)
        self.contact_rates.append(current_contact_rate)
        
        # Calculate average performance over recent window
        if len(self.success_rates) >= self.performance_window:
            recent_success = np.mean(self.success_rates[-self.performance_window:])
            recent_distance = np.mean(self.median_distances[-self.performance_window:])
            recent_contact = np.mean(self.contact_rates[-self.performance_window:])
        else:
            recent_success = np.mean(self.success_rates) if self.success_rates else 0.0
            recent_distance = np.mean(self.median_distances) if self.median_distances else float('inf')
            recent_contact = np.mean(self.contact_rates) if self.contact_rates else 0.0
        
        # Check for improvement
        improved = False
        
        # Primary metric: success rate improvement
        if recent_success > self.best_performance + self.min_delta:
            self.best_performance = recent_success
            self.best_performance_step = self.num_timesteps
            improved = True
            
        # Secondary metric: distance improvement (for approach phase)
        if recent_distance < self.best_median_distance * (1 - self.min_delta):
            self.best_median_distance = recent_distance
            improved = True
            
        # Tertiary metric: contact rate improvement
        if recent_contact > self.best_contact_rate + self.min_delta:
            self.best_contact_rate = recent_contact
            improved = True
        
        if improved:
            self.patience_counter = 0
            if self.verbose > 0:
                print(f"🎯 Performance improvement detected at step {self.num_timesteps}")
                print(f"   Success: {recent_success:.1%}, Distance: {recent_distance:.3f}m, Contact: {recent_contact:.1%}")
        else:
            self.patience_counter += 1
            
        # Check for catastrophic forgetting - only after sufficient training
        # Require at least 3 evaluations and best_performance > 0.08 (8%) before checking
        if len(self.success_rates) >= 3 and self.best_performance > 0.08:
            performance_drop = self.best_performance - recent_success
            # Only check distance increase if we have a valid baseline (not inf)
            distance_increase = recent_distance - self.best_median_distance if self.best_median_distance != float('inf') else 0.0
            contact_drop = self.best_contact_rate - recent_contact
            
            # Significant degradation check - HARD STOP
            catastrophic_forgetting = (
                performance_drop > self.degradation_threshold and performance_drop > 0.03 or  # Significant drop from established performance
                (distance_increase > 0.1 and self.best_median_distance != float('inf')) or  # 10cm increase in distance from valid baseline
                (contact_drop > self.degradation_threshold and self.best_contact_rate > 0.05) or  # Contact rate drop from established baseline
                (self.best_performance > 0.15 and recent_success < 0.05)  # Major collapse: >15% down to <5%
            )
        else:
            catastrophic_forgetting = False
        
        if catastrophic_forgetting:
            if self.verbose > 0:
                performance_drop = self.best_performance - recent_success
                distance_increase = recent_distance - self.best_median_distance if self.best_median_distance != float('inf') else 0.0
                contact_drop = self.best_contact_rate - recent_contact
                print(f"🚨 CATASTROPHIC FORGETTING DETECTED at step {self.num_timesteps}")
                print(f"   Performance drop: {performance_drop:.1%}")
                print(f"   Distance increase: {distance_increase:.3f}m")
                print(f"   Contact rate drop: {contact_drop:.1%}")
                print(f"   Best was at step {self.best_performance_step}")
                print(f"   HARD STOP: Training terminated immediately")
            
            # Raise exception to ensure training actually stops
            raise KeyboardInterrupt("Early stopping triggered due to catastrophic forgetting")
            
        # Patience-based stopping
        if self.patience_counter >= self.patience:
            if self.verbose > 0:
                print(f"⏹️  Early stopping triggered at step {self.num_timesteps}")
                print(f"   No improvement for {self.patience} evaluations")
                print(f"   Best performance: {self.best_performance:.1%} at step {self.best_performance_step}")
                print(f"   Current performance: {recent_success:.1%}")
                print(f"   Performance preserved - avoiding catastrophic forgetting")
            
            return False  # Stop training
            
        return True
    
    def _get_current_success_rate(self) -> float:
        """Extract current success rate from training environment"""
        try:
            # Use same source as curriculum callback for consistency
            if hasattr(self.training_env, 'get_attr'):
                # Try curriculum callback's success tracker first
                success_rates = self.training_env.get_attr('success_rate_tracker')
                if success_rates and success_rates[0] is not None:
                    return np.mean([sr for sr in success_rates if sr is not None])
                
                # Fallback to episode success rate if available
                success_rates = self.training_env.get_attr('episode_success_rate')
                if success_rates and success_rates[0] is not None:
                    return np.mean([sr for sr in success_rates if sr is not None])
            return 0.0
        except:
            return 0.0
    
    def _get_current_median_distance(self) -> float:
        """Extract current median distance from training environment"""
        try:
            if hasattr(self.training_env, 'get_attr'):
                distances = self.training_env.get_attr('episode_min_object_distance')
                valid_distances = [d for d in distances if d is not None and d != float('inf')]
                return np.median(valid_distances) if valid_distances else float('inf')
            return float('inf')
        except:
            return float('inf')
    
    def _get_current_contact_rate(self) -> float:
        """Extract current contact rate from training environment"""
        try:
            if hasattr(self.training_env, 'get_attr'):
                contact_rates = self.training_env.get_attr('episode_contact_rate')
                return np.mean([cr for cr in contact_rates if cr is not None])
            return 0.0
        except:
            return 0.0