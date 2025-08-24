"""
Curriculum Manager Integration Callback
Ensures curriculum manager is properly integrated with training
"""

import numpy as np
from typing import Dict, Any, Tuple
import time

class CurriculumManager:
    """curriculum manager with reasonable rewards that prevent physics exploitation"""
    
    def __init__(self, env, on_phase_change_callback=None):
        self.env = env
        self.current_phase = "milestone_0_percent"
        self.on_phase_change_callback = on_phase_change_callback  # Callback to reset training metrics
        self.phase_progress = 0.0
        self.success_history: list[float] = []
        self.collision_history: list[Dict[str, Any]] = []
        self.phase_start_time = time.time()
        self.total_timesteps = 0
        # Episode-level tracking to stabilize success estimates and gate advancement
        self.phase_episode_count = 0
        self.phase_success_count = 0
        self.episode_successes_window: list[float] = []  # sliding window of recent episode results (1/0)
        self.success_ema = 0.0
        self.success_ema_alpha = 0.1  # EMA smoothing factor
        self.last_advance_time = 0.0
        self.last_curriculum_level = None  # for smoothing level jumps

        # Progressive Domain Randomization Curriculum - Approach Phase Breakdown
        # Phase 1: Core Approach Learning (Fixed Environment)
        # Phase 2: Environmental Robustness (Gradual Randomization)
        # Phase 3: Full Robustness (Complete Randomization)
        self.phases = {
            # Phase 1: Core Approach Learning (25% success threshold)
            "milestone_0_percent": {
                "timesteps": 30_000,
                "success_threshold": 0.25,  # 25% success to advance
                "focus": "basic_approach_fixed",
                "description": "Learn basic approach - NEAR SPAWN 8-14cm, fixed properties",
                "collision_rewards": {"gripper_object_contact": 0.2, "arm_object_collision": -1.0},
                "termination_on_bad_collision": False,
                "spawn_radius": 0.11,  # NEAR SPAWN 8-14cm (avg 11cm)
                "objects": ["cube_only"],
                "mass_range": (50, 50),  # Fixed mass
                "color_randomization": False,  # Fixed
                "lighting_randomization": False,  # Fixed  
                "friction_randomization": False,  # Fixed
                "domain_randomization": False,
                # Gating parameters per phase 
                "min_phase_episodes": 80,  # Increased from 60
                "min_phase_time_sec": 600,  # Increased from 300 (10 minutes minimum)
                "advance_cooldown_sec": 180,  # Increased from 120
            },
            "milestone_5_percent": {
                "timesteps": 30_000,
                "success_threshold": 0.25,  #  25% for Phase 1
                "focus": "expanded_approach_fixed", 
                "description": "Expand approach area - spawn_radius 0.20m, fixed properties",
                "collision_rewards": {"gripper_object_contact": 0.2, "arm_object_collision": -1.0},
                "termination_on_bad_collision": False,
                "spawn_radius": 0.20,  #  0.20m
                "objects": ["cube_only"],
                "mass_range": (50, 50),  # Fixed mass
                "color_randomization": False,  # Fixed
                "lighting_randomization": False,  # Fixed
                "friction_randomization": False,  # Fixed
                "domain_randomization": False,
                "min_phase_episodes": 80,  # Increased from 60  
                "min_phase_time_sec": 600,  # Increased from 300 (10 minutes minimum)
                "advance_cooldown_sec": 180,  # Increased from 120
            },
            # Phase 2: Environmental Robustness (10-30% success threshold)
            "milestone_10_percent": {
                "timesteps": 30_000,
                "success_threshold": 0.25,  # Should be 25% success to advance, not 10%
                "focus": "mass_variation_introduction",
                "description": "Add mass variation - spawn_radius 0.25m, mass random 30-70g",
                "collision_rewards": {"gripper_object_contact": 0.2, "arm_object_collision": -1.0},
                "termination_on_bad_collision": False,
                "spawn_radius": 0.25,  # 0.25m
                "objects": ["cube_only"],
                "mass_range": (30, 70),  # random 30-70g
                "color_randomization": False,  # Fixed
                "lighting_randomization": False,  # Fixed
                "friction_randomization": False,  # Fixed  
                "domain_randomization": True,
                "min_phase_episodes": 100,  # Increased from 80
                "min_phase_time_sec": 720,  # Increased from 420 (12 minutes minimum) 
                "advance_cooldown_sec": 240,  # Increased from 180
            },
            "milestone_15_percent": {
                "timesteps": 30_000,
                "success_threshold": 0.15,  # 15% success to advance
                "focus": "visual_variation_introduction",
                "description": "Add color randomization - spawn_radius 0.30m, color random RGB",
                "collision_rewards": {"gripper_object_contact": 0.2, "arm_object_collision": -1.0},
                "termination_on_bad_collision": False,
                "spawn_radius": 0.30,  # 0.30m
                "objects": ["cube_only"],
                "mass_range": (30, 70),  # Continue mass variation
                "color_randomization": True,  # Add color randomization
                "lighting_randomization": False,  # Fixed
                "friction_randomization": False,  # Fixed
                "domain_randomization": True,
                "min_phase_episodes": 100,  # Increased from 80
                "min_phase_time_sec": 720,  # Increased from 420 (12 minutes minimum)
                "advance_cooldown_sec": 240,  # Increased from 180
            },
            "milestone_20_percent": {
                "timesteps": 35_000,
                "success_threshold": 0.20,  # 20% success threshold
                "focus": "physics_variation_introduction", 
                "description": "Add lighting + friction - spawn_radius 0.35m, lighting + friction random",
                "collision_rewards": {"gripper_object_contact": 0.2, "arm_object_collision": -1.0},
                "termination_on_bad_collision": False,
                "spawn_radius": 0.35,  # 0.35m
                "objects": ["cube_only"],
                "mass_range": (20, 100),  # Expanded mass range
                "color_randomization": True,
                "lighting_randomization": True,  # Add lighting variation
                "friction_randomization": True,  # Add friction randomization
                "domain_randomization": True,
                "min_phase_episodes": 100,
                "min_phase_time_sec": 600,
                "advance_cooldown_sec": 180,
            },
            "milestone_25_percent": {
                "timesteps": 35_000,
                "success_threshold": 0.25,  # 25% success to advance
                "focus": "multi_object_introduction",
                "description": "Add second object - spawn_radius 0.40m, [cube, sphere]",
                "collision_rewards": {"gripper_object_contact": 0.2, "arm_object_collision": -1.0},
                "termination_on_bad_collision": False,
                "spawn_radius": 0.40,  # 0.40m
                "objects": ["cube", "sphere"],  # Add sphere
                "mass_range": (20, 150),
                "color_randomization": True,
                "lighting_randomization": True,
                "friction_randomization": True,
                "domain_randomization": True,
                "min_phase_episodes": 120,
                "min_phase_time_sec": 900,
                "advance_cooldown_sec": 240,
            },
            # Phase 3: Full Robustness (30%+ success)
            "milestone_30_percent": {
                "timesteps": 40_000,
                "success_threshold": 0.30,  # 30% success to advance to grasping
                "focus": "full_approach_robustness",
                "description": "Full robustness - spawn_radius 0.45m, [cube, sphere, cylinder] + full randomization",
                "collision_rewards": {"gripper_object_contact": 0.2, "arm_object_collision": -1.0},
                "termination_on_bad_collision": False,
                "spawn_radius": 0.45,  # 0.45m
                "objects": ["cube", "sphere", "cylinder"],  # All object types
                "mass_range": (20, 200),
                "color_randomization": True,
                "lighting_randomization": True,
                "friction_randomization": True,
                "domain_randomization": True,
                "min_phase_episodes": 150,
                "min_phase_time_sec": 1200,
                "advance_cooldown_sec": 300,
            },
            # Then proceed to grasping stage
            "grasping": {
                "timesteps": 400_000,  # Increased training time
                "success_threshold": 0.35,  # More realistic with gentler spawn progression (was 0.50)
                "focus": "successful_grasp",
                "description": "Consistently grasp and lift objects",
                "collision_rewards": {"successful_grasp": 1.5, "arm_object_collision": -1.0},
                "termination_on_bad_collision": False,
                "domain_randomization": True,
                "min_phase_episodes": 200,
                "min_phase_time_sec": 1800,
                "advance_cooldown_sec": 300,
            },
            "manipulation": {
                "timesteps": 600_000,  # Increased training time
                "success_threshold": 0.70,  #70% for real-world transfer
                "focus": "lift_and_place",
                "description": "Reliably place objects at targets",
                "collision_rewards": {"successful_place": 3.0, "arm_object_collision": -0.5},
                "termination_on_bad_collision": False,
                "min_phase_episodes": 250,
                "min_phase_time_sec": 2400,
                "advance_cooldown_sec": 300,
            },
            "mastery": {
                "timesteps": 800_000,  # Increased training time
                "success_threshold": 0.85,  # 85% for robust deployment
                "focus": "efficient_place",
                "description": "Achieve deployment-ready performance",
                "collision_rewards": {"collision_free_success": 4.0, "any_unwanted_collision": -0.2},
                "termination_on_bad_collision": False,
                "min_phase_episodes": 300,
                "min_phase_time_sec": 3600,
                "advance_cooldown_sec": 300,
            },
        }
        
        # Set initial phase in environment
        print(f"🎯 CURRICULUM INITIALIZATION:")
        print(f"   Starting Phase: {self.current_phase}")
        print(f"   Expected Behavior: {self.phases[self.current_phase]['description']}")
        # Expose current phase on base env (for env-side checks/logs)
        try:
            # Prefer proper set_attr on vectorized envs
            if hasattr(self.env, 'set_attr'):
                self.env.set_attr('current_phase', self.current_phase)
            elif hasattr(self.env, 'venv') and hasattr(self.env.venv, 'set_attr'):
                self.env.venv.set_attr('current_phase', self.current_phase)
            elif hasattr(self.env, 'envs') and self.env.envs:
                for e in self.env.envs:
                    setattr(e, 'current_phase', self.current_phase)
            else:
                setattr(self.env, 'current_phase', self.current_phase)
        except Exception:
            pass
        self._apply_phase_settings()
        
        # Ensure curriculum level is applied immediately
        sync_success = self._force_curriculum_level_sync()
        
        # Additional verification and debugging
        print(f"🔍 POST-INITIALIZATION VERIFICATION:")
        if sync_success:
            print(f"   ✅ Curriculum synchronization successful")
        else:
            print(f"   ⚠️ Curriculum synchronization had issues - monitoring spawn behavior")
        
        # Try to get current curriculum level for final verification
        try:
            if hasattr(self.env, 'get_attr'):
                current_levels = self.env.get_attr('curriculum_level')
                if current_levels:
                    print(f"   📊 Current curriculum levels: {[f'{level:.3f}' for level in current_levels]}")
            elif hasattr(self.env, 'venv') and hasattr(self.env.venv, 'get_attr'):
                current_levels = self.env.venv.get_attr('curriculum_level')
                if current_levels:
                    print(f"   📊 Current curriculum levels (via venv): {[f'{level:.3f}' for level in current_levels]}")
            elif hasattr(self.env, 'curriculum_level'):
                print(f"   📊 Current curriculum level (direct): {self.env.curriculum_level:.3f}")
        except Exception as e:
            print(f"   ⚠️ Could not verify final curriculum level: {e}")
        
        print(f"   🎯 Expected behavior: NEAR SPAWN with 8-14cm radius from end-effector")
        print("="*60)
        
    def update(self, success_rate: float = None, collision_info: Dict[str, Any] = None) -> Dict[str, Any]:
        """Update curriculum status; prefer per-episode history when available.
        success_rate is optional; if provided, used as a hint; otherwise derive from episode window/EMA."""
        if success_rate is not None:
            self.success_history.append(success_rate)

        if collision_info:
            self.collision_history.append(collision_info)
        
        # Periodic curriculum level verification (every ~20 episodes)
        if len(self.success_history) % 20 == 0 and len(self.success_history) > 0:
            self.verify_curriculum_level()

        # Compute recent success based on episode-level data first (preferred)
        ep_window_size = 50
        if self.episode_successes_window:
            use_n = min(ep_window_size, len(self.episode_successes_window))
            recent_success = float(np.mean(self.episode_successes_window[-use_n:]))
        else:
            # Fallback to callback-provided success rate, smoothed
            window_size = 50
            if len(self.success_history) >= window_size:
                recent_success = float(np.mean(self.success_history[-window_size:]))
            else:
                recent_success = float(np.mean(self.success_history)) if self.success_history else 0.0

        current_phase_info = self.phases[self.current_phase]
        phase_complete = False

        # Time-based check
        time_in_phase = time.time() - self.phase_start_time
        max_phase_time = 3600 * 6  # 6 hours max per phase (reduced from 12)

        # Primary advancement with proper approach success metrics
        recent_collisions = self._analyze_recent_collisions()

        # Gate advancements: require minimum episodes/time in phase and respect a cooldown between advancements
        min_episodes = current_phase_info.get("min_phase_episodes", 40)
        min_time_sec = current_phase_info.get("min_phase_time_sec", 300)  # 5 min default
        cooldown_sec = current_phase_info.get("advance_cooldown_sec", 120)  # 2 min between advances
        has_min_episodes = self.phase_episode_count >= min_episodes
        has_min_time = time_in_phase >= min_time_sec
        passed_cooldown = (time.time() - self.last_advance_time) >= cooldown_sec

        # Improved stability gating with sustained success requirement
        sustained_success_required = 5  # Need sustained success over multiple checks
        
        # Track sustained success over recent evaluations
        if not hasattr(self, '_sustained_success_count'):
            self._sustained_success_count = 0
            
        if recent_success >= current_phase_info["success_threshold"]:
            self._sustained_success_count += 1
        else:
            self._sustained_success_count = 0  # Reset if success drops
            
        # Require ALL gating conditions PLUS sustained success
        meets_threshold = recent_success >= current_phase_info["success_threshold"]
        sustained_success = self._sustained_success_count >= sustained_success_required
        
        if meets_threshold and has_min_episodes and has_min_time and passed_cooldown and sustained_success:
            phase_complete = True
            print(f"✅ Phase complete via sustained success: {recent_success:.1%} >= {current_phase_info['success_threshold']:.1%}")
            print(f"   Sustained for {self._sustained_success_count} consecutive checks")
        elif meets_threshold and (not has_min_episodes or not has_min_time or not sustained_success):
            # Informative note: threshold reached but waiting for stability
            missing = []
            if not has_min_episodes:
                missing.append(f"episodes {self.phase_episode_count}/{min_episodes}")
            if not has_min_time:
                missing.append(f"time {time_in_phase:.0f}s/{min_time_sec}s")
            if not sustained_success:
                missing.append(f"sustained success {self._sustained_success_count}/{sustained_success_required}")
            if not passed_cooldown:
                missing.append(f"cooldown {(time.time() - self.last_advance_time):.0f}s/{cooldown_sec}s")
            print(f"⏳ Threshold met but waiting for stability: {', '.join(missing)}")
        elif self.current_phase == "approach_learning" and time_in_phase > max_phase_time:
            # time-based advancement for approach phase only if really stuck
            if recent_success >= 0.10:  # At least 10% success before time-based advancement
                phase_complete = True
                print(f"⏰ Phase complete via time + adequate progress: {recent_success:.1%} after {time_in_phase/3600:.1f}h")

        # For grasping phase, use proper grasp success metrics
        elif self.current_phase == "grasping":
            # Calculate grasp success rate
            grasp_attempts = recent_collisions.get("grasp_attempts", 0)
            successful_grasps = recent_collisions.get("successful_grasps", 0)
            grasp_success_rate = successful_grasps / max(grasp_attempts, 1) if grasp_attempts > 0 else 0

            # More lenient grasping advancement since model needs to learn this new behavior
            # 1. Any successful grasp attempts show learning progress
            if successful_grasps >= 3 and grasp_attempts >= 10:  # Just need some success
                phase_complete = True
                print(f"✅ Grasping phase complete: Learning progress - {successful_grasps} successes in {grasp_attempts} attempts ({grasp_success_rate:.1%})")

            # 2. Time-based forced progression - robot needs time to learn
            elif time_in_phase > max_phase_time * 0.5:  # 3 hours for grasping learning
                phase_complete = True
                print(f"⏰ Grasping phase complete via time: {time_in_phase/3600:.1f} hours")
                print(f"   Metrics: grasp_rate={grasp_success_rate:.1%} ({successful_grasps}/{grasp_attempts}), task={recent_success:.2%}")

        # Advance phase if ready
        if phase_complete:
            next_phase = self._get_next_phase()
            if next_phase:
                old_phase = self.current_phase
                self.current_phase = next_phase
                self.phase_progress = 0.0
                self.phase_start_time = time.time()
                self.last_advance_time = self.phase_start_time
                self._apply_phase_settings()

                # Keep recent history for continuity
                if len(self.success_history) > 100:
                    self.success_history = self.success_history[-50:]
                if len(self.collision_history) > 100:
                    self.collision_history = self.collision_history[-50:]

                print(f"\n🎓 CURRICULUM ADVANCEMENT:")
                print(f"   From: {old_phase} → To: {self.current_phase}")
                print(f"   Phase Duration: {time_in_phase/3600:.1f} hours")
                print(f"   Final Success Rate: {recent_success:.2%}")
                print(f"   New Focus: {self.phases[self.current_phase]['description']}")
                print(f"   New Threshold: {self.phases[self.current_phase]['success_threshold']:.1%}")
                print("="*60)

                # Reset ALL tracking for new phase - must be more aggressive
                self.success_history = []  # Clear success history to start fresh
                print(f"   ✅ Success tracking reset for new phase")

                # Also clear collision history to prevent immediate re-advancement
                self.collision_history = []  # Clear all collision history for fresh start
                print(f"   ✅ Collision history completely cleared for fresh start")
                
                # Reset episode-level counters
                self.phase_episode_count = 0
                self.phase_success_count = 0
                self.episode_successes_window = []
                self.success_ema = 0.0
                
                # Reset sustained success tracking for new phase
                self._sustained_success_count = 0
                print(f"   ✅ Sustained success counter reset for new phase")

                # Force curriculum level sync after phase change to ensure immediate application
                sync_success = self._force_curriculum_level_sync()
                if not sync_success:
                    print(f"   ⚠️ WARNING: Curriculum level sync failed after phase change!")

                # Apply new phase settings to eval environment if available
                self._sync_with_eval_environment()

                # Update env attribute for current phase so env logs reflect the new phase
                try:
                    if hasattr(self.env, 'set_attr'):
                        self.env.set_attr('current_phase', self.current_phase)
                    elif hasattr(self.env, 'venv') and hasattr(self.env.venv, 'set_attr'):
                        self.env.venv.set_attr('current_phase', self.current_phase)
                    elif hasattr(self.env, 'envs') and self.env.envs:
                        for e in self.env.envs:
                            setattr(e, 'current_phase', self.current_phase)
                    else:
                        setattr(self.env, 'current_phase', self.current_phase)
                except Exception:
                    pass
                
                #  Notify training script to reset metrics
                if self.on_phase_change_callback:
                    try:
                        self.on_phase_change_callback()
                        print(f"   ✅ Training script metrics reset via callback")
                    except Exception as e:
                        print(f"   ⚠️ Failed to reset training metrics: {e}")

                return {
                    "phase_changed": True,
                    "old_phase": old_phase,
                    "new_phase": self.current_phase,
                    "success_rate": recent_success,
                    "phase_duration_hours": time_in_phase/3600,
                    "curriculum_sync_success": sync_success
                }

        # Update phase progress
        if current_phase_info.get("timesteps"):
            # Estimate progress based on time and expected duration
            expected_hours = current_phase_info["timesteps"] / 100_000  # Rough estimate
            self.phase_progress = min(1.0, time_in_phase / (expected_hours * 3600))

        return self._get_current_status()
    
    def _analyze_recent_collisions(self) -> Dict[str, float]:
        """collision analysis with proper approach success tracking"""
        if len(self.collision_history) < 10:
            return {
                "stability_score": 0.0, 
                "gentle_contact_rate": 0.0,
                "approach_success_rate": 0.0,
                "successful_grasps": 0,
                "grasp_attempts": 0
            }
        
        recent_collisions = self.collision_history[-50:]  # Look at more history
        
        analysis = {
            "stability_score": 0.0,
            "gentle_contact_rate": 0.0,
            "approach_success_rate": 0.0,
            "successful_grasps": 0,
            "physics_stable_episodes": 0,
            "object_flung_episodes": 0,
            "grasp_attempts": 0,
            "contact_episodes": 0,
            "approach_success_episodes": 0
        }
        
        for collision_data in recent_collisions:
            if isinstance(collision_data, dict):
                # Check for stable physics
                if not collision_data.get("object_flung", False):
                    analysis["physics_stable_episodes"] += 1
                else:
                    analysis["object_flung_episodes"] += 1
                
                # Check for any object contact
                if collision_data.get("made_contact", False) or collision_data.get("gentle_approach", False):
                    analysis["contact_episodes"] += 1
                
                # CRITICAL: Track approach success (getting close to object)
                # This should be based on min_distance achieved during episode
                min_distance = collision_data.get("min_object_distance", float('inf'))
                approach_threshold = collision_data.get("approach_threshold", 0.10)  #  Match proximity bonus threshold
                
                if min_distance <= approach_threshold:
                    analysis["approach_success_episodes"] += 1
                
                # Count grasp attempts
                if collision_data.get("grasp_attempted", False):
                    analysis["grasp_attempts"] += 1
                
                # Count successful grasps
                if collision_data.get("successful_grasp", False) or collision_data.get("object_grasped", False):
                    analysis["successful_grasps"] += 1
        
        # Calculate rates
        episode_count = max(len(recent_collisions), 1)
        analysis["stability_score"] = analysis["physics_stable_episodes"] / episode_count
        analysis["gentle_contact_rate"] = analysis["contact_episodes"] / episode_count
        analysis["approach_success_rate"] = analysis["approach_success_episodes"] / episode_count
        
        return analysis
    
    def _get_next_phase(self) -> str:
        """Get the next phase in curriculum"""
        phase_order = [
            "milestone_0_percent", "milestone_5_percent", "milestone_10_percent", 
            "milestone_15_percent", "milestone_20_percent", "milestone_25_percent", 
            "milestone_30_percent", "grasping", "manipulation", "mastery"
        ]
        try:
            current_idx = phase_order.index(self.current_phase)
        except ValueError:
            print(f"⚠️ Warning: Current phase '{self.current_phase}' not found in phase_order")
            return None
        
        if current_idx < len(phase_order) - 1:
            return phase_order[current_idx + 1]
        return None
    
    def get_current_phase_index(self) -> int:
        """Get the index of current phase in the curriculum sequence"""
        phase_order = [
            "milestone_0_percent", "milestone_5_percent", "milestone_10_percent", 
            "milestone_15_percent", "milestone_20_percent", "milestone_25_percent", 
            "milestone_30_percent", "grasping", "manipulation", "mastery"
        ]
        try:
            return phase_order.index(self.current_phase)
        except ValueError:
            print(f"⚠️ Warning: Current phase '{self.current_phase}' not found in phase_order")
            return -1
    
    def _apply_phase_settings(self):
        """Apply current phase settings to environment"""
        current_phase_info = self.phases[self.current_phase]
        
        # Set curriculum level based on phase - progressive approach learning
        #  Match environment curriculum level ranges exactly
        # < 0.15: NEAR SPAWN (8-14cm), 0.15-0.25: INTERMEDIATE SPAWN 1 (15-25cm), 
        # 0.25-0.40: INTERMEDIATE SPAWN 2 (±8cm), > 0.40: FULL AREA SPAWN
        phase_levels = {
            "milestone_0_percent": 0.05,   # NEAR SPAWN: 8-14cm radius (< 0.15) - 0.15m spawn_radius
            "milestone_5_percent": 0.16,   # INTERMEDIATE SPAWN 1: 15-25cm radius - 0.20m spawn_radius
            "milestone_10_percent": 0.18,  # INTERMEDIATE SPAWN 1: 15-25cm radius (0.15-0.25) - 0.25m
            "milestone_15_percent": 0.20,  # INTERMEDIATE SPAWN 1: 15-25cm radius (0.15-0.25) - 0.30m
            "milestone_20_percent": 0.22,  # INTERMEDIATE SPAWN 1: 15-25cm radius (0.15-0.25) - 0.35m
            "milestone_25_percent": 0.30,  # INTERMEDIATE SPAWN 2: ±8cm square area (0.25-0.40) - 0.40m
            "milestone_30_percent": 0.35,  # INTERMEDIATE SPAWN 2: ±8cm square area (0.25-0.40) - 0.45m
            "grasping": 0.45,              # FULL AREA SPAWN: Standard spawning area (> 0.40)
            "manipulation": 0.60,          # FULL AREA SPAWN: Standard spawning area
            "mastery": 0.80               # FULL AREA SPAWN: Complete challenge
        }
        
        target_level = phase_levels.get(self.current_phase, 0.5)
        # Smooth large jumps in difficulty to avoid sudden spawn distance spikes
        if self.last_curriculum_level is not None and target_level > self.last_curriculum_level:
            max_step = 0.08  # Increased from 0.05 to 0.08 to allow reasonable progression between phases
            if target_level - self.last_curriculum_level > max_step:
                print(f"⚠️  Smoothing curriculum jump: {self.last_curriculum_level:.2f} → {target_level:.2f} -> {self.last_curriculum_level + max_step:.2f}")
                target_level = self.last_curriculum_level + max_step
        curriculum_level = target_level
        
        # Handle different environment wrapper types
        #  Use robust curriculum level setting with sync
        success = self._set_curriculum_level_robust(curriculum_level)
        
        # Remember last applied level
        self.last_curriculum_level = curriculum_level
        
        # Progressive domain randomization based on milestone curriculum
        current_phase_info = self.phases[self.current_phase]
        enable_randomization = current_phase_info.get("domain_randomization", False)
        
        # Handle different environment wrapper types for domain randomization
        if hasattr(self.env, 'env_method'):
            # Direct vectorized environment - use env_method
            try:
                self.env.env_method('set_domain_randomization', enable_randomization)
                print(f"🎲 Domain randomization via env_method: {'enabled' if enable_randomization else 'disabled'} for {self.current_phase}")
            except:
                print(f"⚠️ Warning: Could not set domain randomization via env_method")
        elif hasattr(self.env, 'venv') and hasattr(self.env.venv, 'env_method'):
            # VecNormalize wrapped - access underlying venv
            try:
                self.env.venv.env_method('set_domain_randomization', enable_randomization)
                print(f"🎲 Domain randomization via venv.env_method: {'enabled' if enable_randomization else 'disabled'} for {self.current_phase}")
            except:
                print(f"⚠️ Warning: Could not set domain randomization via venv.env_method")
        elif hasattr(self.env, 'set_domain_randomization'):
            # Single environment - direct call
            self.env.set_domain_randomization(enable_randomization)
            print(f"🎲 Domain randomization directly: {'enabled' if enable_randomization else 'disabled'} for {self.current_phase}")
        else:
            print(f"⚠️ Warning: Could not set domain randomization - no method available")
        
        # Apply milestone-specific settings (spawn radius, object types, mass range, etc.)
        self._apply_milestone_settings(current_phase_info)
        
        # collision settings with better error handling and verification
        collision_rewards = current_phase_info["collision_rewards"]
        collision_termination = current_phase_info.get("termination_on_bad_collision", False)
        
        print(f"🔧 Applying collision settings for {self.current_phase}:")
        print(f"   Rewards: {collision_rewards}")
        print(f"   Termination: {collision_termination}")
        
        success = False
        if hasattr(self.env, 'env_method'):
            # Direct vectorized environment
            try:
                self.env.env_method('set_collision_rewards', collision_rewards)
                self.env.env_method('set_collision_termination', collision_termination)
                # Verify it was applied
                try:
                    applied_rewards = self.env.get_attr('collision_rewards')
                    if applied_rewards and applied_rewards[0] == collision_rewards:
                        success = True
                        print(f"✅ Collision settings applied via env_method: {collision_rewards}")
                    else:
                        print(f"⚠️ Collision rewards mismatch: set {collision_rewards}, got {applied_rewards[0] if applied_rewards else 'None'}")
                except Exception:
                    # Verification failed, but setting might have worked
                    success = True
                    print(f"✅ Collision settings applied via env_method (verification failed): {collision_rewards}")
            except Exception as e:
                print(f"⚠️ Warning: Could not set collision settings via env_method: {e}")
                
        elif hasattr(self.env, 'venv') and hasattr(self.env.venv, 'env_method'):
            # VecNormalize wrapped - access underlying venv
            try:
                self.env.venv.env_method('set_collision_rewards', collision_rewards)
                self.env.venv.env_method('set_collision_termination', collision_termination)
                # Verify through venv
                try:
                    applied_rewards = self.env.venv.get_attr('collision_rewards')
                    if applied_rewards and applied_rewards[0] == collision_rewards:
                        success = True
                        print(f"✅ Collision settings applied via venv.env_method: {collision_rewards}")
                    else:
                        print(f"⚠️ Collision rewards mismatch via venv: set {collision_rewards}, got {applied_rewards[0] if applied_rewards else 'None'}")
                except Exception:
                    success = True
                    print(f"✅ Collision settings applied via venv.env_method (verification failed): {collision_rewards}")
            except Exception as e:
                print(f"⚠️ Warning: Could not set collision settings via venv.env_method: {e}")
                
        elif hasattr(self.env, 'set_collision_rewards'):
            # Single environment
            try:
                self.env.set_collision_rewards(collision_rewards)
                self.env.set_collision_termination(collision_termination)
                success = True
                print(f"✅ Collision settings applied directly: {collision_rewards}")
            except Exception as e:
                print(f"⚠️ Warning: Could not set collision settings directly: {e}")
        else:
            print(f"⚠️ Warning: No collision setting method available")
            print(f"   Environment type: {type(self.env)}")
            print(f"   Available methods: {[m for m in dir(self.env) if 'collision' in m.lower()]}")
            
        if not success:
            print(f"❌ COLLISION SETTINGS FAILED: Phase-specific collision penalties not active!")
            print(f"   Target rewards: {collision_rewards}")
            print(f"   Target termination: {collision_termination}")
        
        # Final verification of all settings
        print(f"🔍 FINAL CURRICULUM VERIFICATION for {self.current_phase}:")
        self.verify_curriculum_level()
    
    def _apply_milestone_settings(self, phase_info: Dict[str, Any]):
        """Apply milestone-specific domain randomization settings"""
        milestone_settings = {
            'spawn_radius': phase_info.get('spawn_radius', 0.15),
            'objects': phase_info.get('objects', ['cube_only']),
            'mass_range': phase_info.get('mass_range', (50, 50)),
            'color_randomization': phase_info.get('color_randomization', False),
            'lighting_randomization': phase_info.get('lighting_randomization', False),
            'friction_randomization': phase_info.get('friction_randomization', False)
        }
        
        # Apply settings through environment interface
        if hasattr(self.env, 'env_method'):
            try:
                self.env.env_method('set_milestone_settings', milestone_settings)
                print(f"🎯 Milestone settings applied via env_method: {milestone_settings}")
            except Exception as e:
                print(f"⚠️ Warning: Could not set milestone settings via env_method: {e}")
        elif hasattr(self.env, 'venv') and hasattr(self.env.venv, 'env_method'):
            try:
                self.env.venv.env_method('set_milestone_settings', milestone_settings)
                print(f"🎯 Milestone settings applied via venv.env_method: {milestone_settings}")
            except Exception as e:
                print(f"⚠️ Warning: Could not set milestone settings via venv.env_method: {e}")
        elif hasattr(self.env, 'set_milestone_settings'):
            try:
                self.env.set_milestone_settings(milestone_settings)
                print(f"🎯 Milestone settings applied directly: {milestone_settings}")
            except Exception as e:
                print(f"⚠️ Warning: Could not set milestone settings directly: {e}")
        else:
            print(f"⚠️ Warning: No method available to set milestone settings")
            print(f"   Settings would be: {milestone_settings}")
        
        print(f"📊 Progressive Curriculum Status:")
        print(f"   Phase: {self.current_phase}")
        print(f"   Domain Randomization: {'ON' if phase_info.get('domain_randomization', False) else 'OFF'}")
        print(f"   Spawn Radius: {milestone_settings['spawn_radius']}m")
        print(f"   Object Types: {milestone_settings['objects']}")
        print(f"   Mass Range: {milestone_settings['mass_range'][0]}-{milestone_settings['mass_range'][1]}g")
        print(f"   Color Randomization: {'ON' if milestone_settings['color_randomization'] else 'OFF'}")
        print(f"   Physics Randomization: {'ON' if milestone_settings['friction_randomization'] else 'OFF'}")
    
    def _get_current_status(self) -> Dict[str, Any]:
        """Get current curriculum status"""
        return {
            "phase_changed": False,
            "current_phase": self.current_phase,
            "phase_index": self.get_current_phase_index(),  #  Include proper phase index
            "phase_progress": self.phase_progress,
            "focus": self.phases[self.current_phase]["focus"],
            "success_threshold": self.phases[self.current_phase]["success_threshold"],
            "current_collision_rewards": self.phases[self.current_phase]["collision_rewards"]
        }
    
    def get_collision_rewards_for_phase(self) -> Dict[str, float]:
        """Get collision reward configuration for current phase"""
        return self.phases[self.current_phase]["collision_rewards"]
    
    def should_terminate_on_bad_collision(self) -> bool:
        """Check if episodes should terminate on bad collisions"""
        return self.phases[self.current_phase].get("termination_on_bad_collision", False)
    
    def log_collision_episode(self, collision_data: Dict[str, Any]):
        """Log collision data for an episode"""
        # Add physics stability info
        if hasattr(self.env, 'physics_stable'):
            collision_data['physics_stable'] = self.env.physics_stable
            
        self.collision_history.append(collision_data)
        
        # Keep only recent history
        if len(self.collision_history) > 100:
            self.collision_history = self.collision_history[-50:]

    def register_episode_result(self, success: bool):
        """Record per-episode result to stabilize curriculum decisions."""
        self.phase_episode_count += 1
        if success:
            self.phase_success_count += 1
        self.episode_successes_window.append(1.0 if success else 0.0)
        if len(self.episode_successes_window) > 200:
            self.episode_successes_window = self.episode_successes_window[-200:]
        # Update EMA for robustness when window is small
        x = 1.0 if success else 0.0
        self.success_ema = (1 - self.success_ema_alpha) * self.success_ema + self.success_ema_alpha * x
    
    def _force_curriculum_level_sync(self):
        """Force immediate curriculum level synchronization to fix startup timing issue"""
        current_phase_info = self.phases[self.current_phase]
        phase_levels = {
            "milestone_0_percent": 0.05,   # NEAR SPAWN: 8-14cm radius (< 0.15)
            "milestone_5_percent": 0.16,   # INTERMEDIATE SPAWN 1: 15-25cm radius - FIXED to match 0.20m spec
            "milestone_10_percent": 0.18,  # INTERMEDIATE SPAWN 1: 15-25cm radius
            "milestone_15_percent": 0.20,  # INTERMEDIATE SPAWN 1: 15-25cm radius
            "milestone_20_percent": 0.22,  # INTERMEDIATE SPAWN 1: 15-25cm radius
            "milestone_25_percent": 0.30,  # INTERMEDIATE SPAWN 2: ±8cm square area
            "milestone_30_percent": 0.35,  # INTERMEDIATE SPAWN 2: ±8cm square area
            "grasping": 0.45,              # FULL AREA SPAWN
            "manipulation": 0.60,
            "mastery": 0.80
        }
        
        target_level = phase_levels.get(self.current_phase, 0.05)
        
        print(f"🔧 FORCING CURRICULUM LEVEL SYNC:")
        print(f"   Target Level: {target_level:.2f} for phase {self.current_phase}")
        
        # synchronization with multiple attempts and fallbacks
        sync_success = False
        attempts = 0
        max_attempts = 5
        
        while not sync_success and attempts < max_attempts:
            attempts += 1
            print(f"   Attempt {attempts}/{max_attempts}...")
            
            # Try method 1: env_method
            if hasattr(self.env, 'env_method'):
                try:
                    self.env.env_method('set_curriculum_level', target_level)
                    # Add delay and verify
                    import time
                    time.sleep(0.02)
                    actual_levels = self.env.get_attr('curriculum_level')
                    if actual_levels and abs(actual_levels[0] - target_level) < 0.01:
                        print(f"   ✅ Level set via env_method: {actual_levels[0]:.3f}")
                        sync_success = True
                        continue
                    else:
                        print(f"   ❌ env_method verification failed: got {actual_levels[0]:.3f}, expected {target_level:.3f}")
                except Exception as e:
                    print(f"   ❌ env_method failed: {e}")
            
            # Try method 2: venv.env_method
            if not sync_success and hasattr(self.env, 'venv') and hasattr(self.env.venv, 'env_method'):
                try:
                    self.env.venv.env_method('set_curriculum_level', target_level)
                    # Add delay and verify
                    import time
                    time.sleep(0.02)
                    actual_levels = self.env.venv.get_attr('curriculum_level')
                    if actual_levels and abs(actual_levels[0] - target_level) < 0.01:
                        print(f"   ✅ Level set via venv.env_method: {actual_levels[0]:.3f}")
                        sync_success = True
                        continue
                    else:
                        print(f"   ❌ venv.env_method verification failed: got {actual_levels[0]:.3f}, expected {target_level:.3f}")
                except Exception as e:
                    print(f"   ❌ venv.env_method failed: {e}")
            
            # Try method 3: direct method
            if not sync_success and hasattr(self.env, 'set_curriculum_level'):
                try:
                    self.env.set_curriculum_level(target_level)
                    # Add delay and verify
                    import time
                    time.sleep(0.02)
                    if hasattr(self.env, 'curriculum_level') and abs(self.env.curriculum_level - target_level) < 0.01:
                        print(f"   ✅ Level set directly: {self.env.curriculum_level:.3f}")
                        sync_success = True
                        continue
                    else:
                        print(f"   ❌ Direct method verification failed: got {getattr(self.env, 'curriculum_level', 'N/A')}, expected {target_level:.3f}")
                except Exception as e:
                    print(f"   ❌ Direct method failed: {e}")
            
            # Try method 4: direct access to underlying environments
            if not sync_success:
                try:
                    env_count = 0
                    if hasattr(self.env, 'envs'):
                        for i, env in enumerate(self.env.envs):
                            if hasattr(env, 'set_curriculum_level'):
                                env.set_curriculum_level(target_level)
                                env_count += 1
                        if env_count > 0:
                            print(f"   ✅ Level set on {env_count} envs directly: {target_level:.3f}")
                            sync_success = True
                            continue
                    elif hasattr(self.env, 'venv') and hasattr(self.env.venv, 'envs'):
                        for i, env in enumerate(self.env.venv.envs):
                            if hasattr(env, 'set_curriculum_level'):
                                env.set_curriculum_level(target_level)
                                env_count += 1
                        if env_count > 0:
                            print(f"   ✅ Level set on {env_count} venv.envs directly: {target_level:.3f}")
                            sync_success = True
                            continue
                except Exception as e:
                    print(f"   ❌ Direct env access failed: {e}")
            
            # Wait before retry
            if not sync_success and attempts < max_attempts:
                import time
                time.sleep(0.1 * attempts)  # Increasing delay
        
        if not sync_success:
            print(f"   ⚠️ WARNING: Could not sync curriculum level after {attempts} attempts!")
            print(f"   Environment type: {type(self.env)}")
            if hasattr(self.env, 'venv'):
                print(f"   Underlying venv type: {type(self.env.venv)}")
            print(f"   Available methods: {[m for m in dir(self.env) if 'curriculum' in m.lower() or 'set_' in m.lower()]}")
        else:
            print(f"   🎯 Expected spawn type: {'NEAR SPAWN (8-14cm)' if target_level < 0.15 else 'INTERMEDIATE SPAWN'}")
            
        #  Update last_curriculum_level to prevent smoothing issues in future calls
        self.last_curriculum_level = target_level
        print(f"   📌 Updated last_curriculum_level to {target_level:.2f} for consistent smoothing")
        
        return sync_success
    
    def _sync_with_eval_environment(self):
        """Sync curriculum settings with evaluation environment if available"""
        # This method will be called by the training script to ensure eval environment 
        # stays in sync with training environment curriculum progression
        print(f"🔄 Syncing curriculum settings with evaluation environment...")
        
        # The actual sync logic is handled by the training script's sync_vecnormalize_stats method
        # This method serves as a hook for future enhancements if needed
        pass
    
    def verify_curriculum_level(self) -> bool:
        """Verify that the curriculum level is correctly set in the environment"""
        #  Add cooldown to prevent continuous fighting
        current_time = time.time()
        if not hasattr(self, '_last_sync_attempt'):
            self._last_sync_attempt = 0
        
        # Don't check more than once every 30 seconds to prevent fighting
        if current_time - self._last_sync_attempt < 30.0:
            return True  # Assume synced if we checked recently
        
        phase_levels = {
            "milestone_0_percent": 0.05, "milestone_5_percent": 0.16, "milestone_10_percent": 0.18,
            "milestone_15_percent": 0.20, "milestone_20_percent": 0.22, "milestone_25_percent": 0.30,
            "milestone_30_percent": 0.35, "grasping": 0.45, "manipulation": 0.60, "mastery": 0.80
        }
        
        expected_level = phase_levels.get(self.current_phase, 0.05)
        
        try:
            actual_levels = None
            if hasattr(self.env, 'get_attr'):
                actual_levels = self.env.get_attr('curriculum_level')
            elif hasattr(self.env, 'venv') and hasattr(self.env.venv, 'get_attr'):
                actual_levels = self.env.venv.get_attr('curriculum_level')
            elif hasattr(self.env, 'curriculum_level'):
                actual_levels = [self.env.curriculum_level]
                
            if actual_levels and len(actual_levels) > 0:
                actual_level = actual_levels[0]
                # RELAXED TOLERANCE: 0.05 instead of 0.01 to prevent constant fighting over small differences
                is_synchronized = abs(actual_level - expected_level) < 0.05
                
                if not is_synchronized:
                    print(f"⚠️ CURRICULUM DESYNC DETECTED: expected {expected_level:.3f}, got {actual_level:.3f}")
                    self._last_sync_attempt = current_time
                    # Try to resync only if significant difference
                    self._force_curriculum_level_sync()
                    return False
                else:
                    return True
            else:
                print(f"⚠️ Could not retrieve curriculum level for verification")
                return False
                
        except Exception as e:
            print(f"⚠️ Curriculum verification failed: {e}")
            return False
    
    def _set_curriculum_level_robust(self, curriculum_level: float) -> bool:
        """Robust curriculum level setting with multiple attempts and fallbacks"""
        success = False
        attempts = 0
        max_attempts = 3
        
        while not success and attempts < max_attempts:
            attempts += 1
            
            # Method 1: env_method (vectorized environments)
            if hasattr(self.env, 'env_method'):
                try:
                    self.env.env_method('set_curriculum_level', curriculum_level)
                    import time
                    time.sleep(0.01)
                    actual_levels = self.env.get_attr('curriculum_level')
                    if actual_levels and abs(actual_levels[0] - curriculum_level) < 0.001:
                        success = True
                        print(f"🎯 Curriculum level synchronized: {curriculum_level:.2f} for {self.current_phase} (attempt {attempts})")
                    else:
                        print(f"⚠️ Curriculum level mismatch: set {curriculum_level:.2f}, got {actual_levels[0]:.2f} (attempt {attempts})")
                except Exception as e:
                    print(f"⚠️ Failed to set curriculum via env_method (attempt {attempts}): {e}")
                    
            # Method 2: venv.env_method (VecNormalize wrapped)
            elif hasattr(self.env, 'venv') and hasattr(self.env.venv, 'env_method'):
                try:
                    self.env.venv.env_method('set_curriculum_level', curriculum_level)
                    import time
                    time.sleep(0.01)
                    actual_levels = self.env.venv.get_attr('curriculum_level')
                    if actual_levels and abs(actual_levels[0] - curriculum_level) < 0.001:
                        success = True
                        print(f"🎯 Curriculum level synchronized via venv: {curriculum_level:.2f} for {self.current_phase} (attempt {attempts})")
                    else:
                        print(f"⚠️ Curriculum level mismatch via venv: set {curriculum_level:.2f}, got {actual_levels[0]:.2f} (attempt {attempts})")
                except Exception as e:
                    print(f"⚠️ Failed to set curriculum via venv.env_method (attempt {attempts}): {e}")
                    
            # Method 3: Direct method (single environment)
            elif hasattr(self.env, 'set_curriculum_level'):
                try:
                    self.env.set_curriculum_level(curriculum_level)
                    import time
                    time.sleep(0.01)
                    if hasattr(self.env, 'curriculum_level') and abs(self.env.curriculum_level - curriculum_level) < 0.001:
                        success = True
                        print(f"🎯 Curriculum level synchronized directly: {curriculum_level:.2f} for {self.current_phase} (attempt {attempts})")
                    else:
                        print(f"⚠️ Curriculum level mismatch direct: set {curriculum_level:.2f}, got {getattr(self.env, 'curriculum_level', 'N/A')} (attempt {attempts})")
                except Exception as e:
                    print(f"⚠️ Failed to set curriculum directly (attempt {attempts}): {e}")
            
            # Fallback: Direct environment access
            if not success:
                try:
                    if hasattr(self.env, 'envs') and self.env.envs:
                        for env in self.env.envs:
                            if hasattr(env, 'set_curriculum_level'):
                                env.set_curriculum_level(curriculum_level)
                                success = True
                                print(f"✅ Fallback: Level set directly on envs: {curriculum_level:.2f}")
                                break
                    elif hasattr(self.env, 'venv') and hasattr(self.env.venv, 'envs') and self.env.venv.envs:
                        for env in self.env.venv.envs:
                            if hasattr(env, 'set_curriculum_level'):
                                env.set_curriculum_level(curriculum_level)
                                success = True
                                print(f"✅ Fallback: Level set directly on venv.envs: {curriculum_level:.2f}")
                                break
                except Exception as e:
                    print(f"❌ Fallback failed (attempt {attempts}): {e}")
            
            # Wait before retry
            if not success and attempts < max_attempts:
                import time
                time.sleep(0.05 * attempts)  # Increasing delay
                
        if not success:
            print(f"❌ CURRICULUM SYNC FAILED after {attempts} attempts!")
            print(f"   Environment type: {type(self.env)}")
            if hasattr(self.env, 'venv'):
                print(f"   Underlying venv type: {type(self.env.venv)}")
                
        return success