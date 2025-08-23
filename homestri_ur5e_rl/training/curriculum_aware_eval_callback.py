"""
Curriculum-Aware Evaluation Callback
Ensures evaluation environment uses same curriculum settings as training environment
"""

import numpy as np
import time
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy

class CurriculumAwareEvalCallback(EvalCallback):
    """
    Enhanced EvalCallback that syncs curriculum state before each evaluation
    """
    
    def __init__(self, 
                 eval_env,
                 trainer=None,  # Reference to training script for sync_vecnormalize_stats
                 **kwargs):
        super().__init__(eval_env, **kwargs)
        self.trainer = trainer
        self.last_curriculum_sync = 0
        # CRITICAL FIX: Make evaluation deterministic
        self.deterministic = True  # Force deterministic evaluation
        
    def _on_step(self) -> bool:
        continue_training = True
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # CRITICAL FIX: Sync curriculum state before evaluation
            self._sync_curriculum_before_evaluation()
            
            # Proceed with normal evaluation
            continue_training = super()._on_step()
            
        return continue_training
    
    def _sync_curriculum_before_evaluation(self):
        """Sync curriculum state from training to evaluation environment"""
        if self.trainer is None:
            return
            
        current_time = time.time()
        
        # Only sync if significant time has passed to avoid excessive syncing
        if current_time - self.last_curriculum_sync < 30.0:  # 30 second cooldown
            return
            
        # CRITICAL FIX: Set deterministic seed for evaluation consistency
        try:
            # Fix numpy seed for deterministic evaluation
            import numpy as np
            np.random.seed(42)
            
            # Try to seed the evaluation environment
            if hasattr(self.eval_env, 'seed'):
                self.eval_env.seed(42)
            elif hasattr(self.eval_env, 'env_method'):
                try:
                    self.eval_env.env_method('seed', 42)
                except:
                    pass
        except Exception as e:
            print(f"   ⚠️ Could not set evaluation seed: {e}")
            
        try:
            print(f"\n🔄 EVALUATION CURRICULUM SYNC at step {self.n_calls:,}")
            
            # Call the trainer's sync method
            self.trainer.sync_vecnormalize_stats()
            
            # Additional curriculum-specific sync
            if hasattr(self.trainer, 'curriculum_manager'):
                curriculum_manager = self.trainer.curriculum_manager
                
                # Force curriculum level sync to evaluation environment  
                phase_levels = {
                    "milestone_0_percent": 0.05, "milestone_5_percent": 0.16, "milestone_10_percent": 0.18,
                    "milestone_15_percent": 0.20, "milestone_20_percent": 0.22, "milestone_25_percent": 0.30,
                    "milestone_30_percent": 0.35, "grasping": 0.45, "manipulation": 0.60, "mastery": 0.80
                }
                
                target_level = phase_levels.get(curriculum_manager.current_phase, 0.05)
                
                # Sync to eval environment
                sync_success = False
                if hasattr(self.eval_env, 'env_method'):
                    try:
                        self.eval_env.env_method('set_curriculum_level', target_level)
                        actual_levels = self.eval_env.get_attr('curriculum_level')
                        if actual_levels and abs(actual_levels[0] - target_level) < 0.01:
                            sync_success = True
                            print(f"   ✅ Eval curriculum level synced via env_method: {actual_levels[0]:.3f}")
                    except Exception as e:
                        print(f"   ❌ env_method sync failed: {e}")
                        
                elif hasattr(self.eval_env, 'venv') and hasattr(self.eval_env.venv, 'env_method'):
                    try:
                        self.eval_env.venv.env_method('set_curriculum_level', target_level)
                        actual_levels = self.eval_env.venv.get_attr('curriculum_level')
                        if actual_levels and abs(actual_levels[0] - target_level) < 0.01:
                            sync_success = True
                            print(f"   ✅ Eval curriculum level synced via venv.env_method: {actual_levels[0]:.3f}")
                    except Exception as e:
                        print(f"   ❌ venv.env_method sync failed: {e}")
                
                elif hasattr(self.eval_env, 'set_curriculum_level'):
                    try:
                        self.eval_env.set_curriculum_level(target_level)
                        if hasattr(self.eval_env, 'curriculum_level') and abs(self.eval_env.curriculum_level - target_level) < 0.01:
                            sync_success = True
                            print(f"   ✅ Eval curriculum level synced directly: {self.eval_env.curriculum_level:.3f}")
                    except Exception as e:
                        print(f"   ❌ Direct sync failed: {e}")
                
                if not sync_success:
                    # Fallback: direct environment access
                    try:
                        if hasattr(self.eval_env, 'envs') and self.eval_env.envs:
                            env = self.eval_env.envs[0]
                            if hasattr(env, 'set_curriculum_level'):
                                env.set_curriculum_level(target_level)
                                sync_success = True
                                print(f"   ✅ Eval curriculum level set via direct access: {target_level:.3f}")
                        elif hasattr(self.eval_env, 'venv') and hasattr(self.eval_env.venv, 'envs') and self.eval_env.venv.envs:
                            env = self.eval_env.venv.envs[0]
                            if hasattr(env, 'set_curriculum_level'):
                                env.set_curriculum_level(target_level)
                                sync_success = True
                                print(f"   ✅ Eval curriculum level set via venv direct access: {target_level:.3f}")
                    except Exception as e:
                        print(f"   ❌ Fallback sync failed: {e}")
                
                if sync_success:
                    print(f"   📋 Evaluation now using phase: {curriculum_manager.current_phase}")
                    print(f"   🎯 Expected spawn difficulty: {'NEAR SPAWN (8-14cm)' if target_level < 0.15 else 'INTERMEDIATE SPAWN' if target_level < 0.40 else 'FULL AREA'}")
                else:
                    print(f"   ⚠️ WARNING: Evaluation curriculum sync failed - may use wrong difficulty!")
                    
            self.last_curriculum_sync = current_time
            print(f"   ⏱️ Next sync in 30+ seconds")
                    
        except Exception as e:
            print(f"⚠️ Curriculum sync error during evaluation: {e}")