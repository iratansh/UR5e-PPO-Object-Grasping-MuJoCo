import numpy as np
from typing import Dict, Any, Tuple
import time

class CurriculumManager:
    """curriculum manager with reasonable rewards that prevent physics exploitation"""
    
    def __init__(self, env):
        self.env = env
        self.current_phase = "approach_learning"
        self.phase_progress = 0.0
        self.success_history = []
        self.collision_history = []
        self.phase_start_time = time.time()
        self.total_timesteps = 0
        
        # Dramatically reduced rewards to prevent exploitation
        self.phases = {
            "approach_learning": {
                "timesteps": 5_000_000,
                "success_threshold": 0.05,  # 5% success to advance
                "focus": "contact_only",
                "description": "Learn to approach object safely and make gentle gripper contact",
                "collision_rewards": {
                    "gripper_object_contact": 0.5,      
                    "arm_object_collision": -2.0,     
                    "object_flung": -5.0,              
                    "gentle_approach_bonus": 0.2,      
                    "stability_bonus": 0.1,             
                    "smooth_movement_bonus": 0.1,       
                    "exploration_bonus": 0.3            
                },
                "termination_on_bad_collision": False
            },
            "contact_refinement": {
                "timesteps": 4_000_000,
                "success_threshold": 0.3,
                "focus": "precise_gripper_positioning_and_contact",
                "description": "Refine gripper positioning and contact quality",
                "collision_rewards": {
                    "gripper_object_contact": 0.3,     
                    "arm_object_collision": -2.0,       
                    "object_flung": -5.0,               
                    "gentle_approach_bonus": 0.1,       
                    "contact_quality_bonus": 0.2,
                    "grasp_preparation_bonus": 0.3     
                },
                "termination_on_bad_collision": False
            },
            "grasping": {
                "timesteps": 8_000_000,
                "success_threshold": 0.5,
                "focus": "successful_object_grasping",
                "description": "Learn to successfully grasp and lift objects",
                "collision_rewards": {
                    "gripper_object_contact": 0.1,      
                    "arm_object_collision": -1.0,       
                    "object_flung": -3.0,               
                    "successful_grasp": 2.0,            
                    "grasp_stability": 0.5,             
                    "lift_progress": 0.3                
                },
                "termination_on_bad_collision": False
            },
            "manipulation": {
                "timesteps": 6_000_000,
                "success_threshold": 0.7,
                "focus": "full_pick_and_place_with_collision_awareness",
                "description": "Complete pick and place tasks with collision awareness",
                "collision_rewards": {
                    "gripper_object_contact": 0.05,     
                    "arm_object_collision": -0.5,       
                    "object_flung": -2.0,               
                    "successful_place": 3.0,          
                    "transport_stability": 0.2,         
                    "placement_accuracy": 0.5           
                },
                "termination_on_bad_collision": False
            },
            "mastery": {
                "timesteps": 4_000_000,
                "success_threshold": 0.85,
                "focus": "robust_manipulation_with_minimal_collisions",
                "description": "Master efficient, collision-free manipulation",
                "collision_rewards": {
                    "collision_free_success": 5.0,      
                    "any_unwanted_collision": -0.2,     
                    "efficiency_bonus": 1.0,           
                    "perfect_execution": 2.0           
                },
                "termination_on_bad_collision": False
            }
        }
        
        # Set initial phase in environment
        self._apply_phase_settings()
        
    def update(self, success_rate: float, collision_info: Dict[str, Any] = None) -> Dict[str, Any]:
        """FIXED update that allows progression even with 0% task success"""
        self.success_history.append(success_rate)
        
        if collision_info:
            self.collision_history.append(collision_info)
        
        # Use larger window for more stable metrics
        window_size = 50  # Reduced from 100 for faster adaptation
        if len(self.success_history) >= window_size:
            recent_success = np.mean(self.success_history[-window_size:])
        else:
            recent_success = np.mean(self.success_history) if self.success_history else 0.0
        
        current_phase_info = self.phases[self.current_phase]
        phase_complete = False
        
        # Time-based check
        time_in_phase = time.time() - self.phase_start_time
        max_phase_time = 3600 * 6  # 6 hours max per phase (reduced from 12)
        
        # Primary advancement: success threshold
        if recent_success >= current_phase_info["success_threshold"]:
            phase_complete = True
            print(f"✅ Phase complete via success: {recent_success:.1%} >= {current_phase_info['success_threshold']:.1%}")
        
        # CRITICAL FIX: Alternative advancement criteria for approach phase
        elif self.current_phase == "approach_learning":
            # Check for ANY progress indicators
            recent_collisions = self._analyze_recent_collisions()
            
            # Multiple ways to advance from approach phase:
            # 1. Any successful grasp attempts
            if recent_collisions.get("successful_grasps", 0) > 0:
                phase_complete = True
                print(f"✅ Approach phase complete: successful grasps detected!")
            
            # 2. Consistent object contact (even without grasps)
            elif recent_collisions.get("gentle_contact_rate", 0) > 0.2:  # 20% contact rate
                phase_complete = True
                print(f"✅ Approach phase complete: consistent contact achieved ({recent_collisions['gentle_contact_rate']:.1%})")
            
            # 3. Low but non-zero success
            elif recent_success > 0.001:  # Even 0.1% is progress!
                phase_complete = True
                print(f"✅ Approach phase complete: minimal success detected ({recent_success:.2%})")
            
            # 4. Time-based forced progression
            elif time_in_phase > max_phase_time:
                phase_complete = True
                print(f"⏰ Forcing approach phase completion after {time_in_phase/3600:.1f} hours")
                print(f"   Recent metrics: success={recent_success:.2%}, contacts={recent_collisions.get('gentle_contact_rate', 0):.1%}")
        
        # For other phases, also check alternative metrics
        elif self.current_phase == "contact_refinement":
            recent_collisions = self._analyze_recent_collisions()
            
            # Alternative: consistent grasping attempts
            if recent_collisions.get("successful_grasps", 0) > 2:  # A few grasps
                phase_complete = True
                print(f"✅ Contact phase complete: multiple grasps achieved")
            elif time_in_phase > max_phase_time * 1.5:  # Give more time for contact
                phase_complete = True
                print(f"⏰ Forcing contact phase completion after {time_in_phase/3600:.1f} hours")
        
        # Advance phase if ready
        if phase_complete:
            next_phase = self._get_next_phase()
            if next_phase:
                old_phase = self.current_phase
                self.current_phase = next_phase
                self.phase_progress = 0.0
                self.phase_start_time = time.time()
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
                
                return {
                    "phase_changed": True,
                    "old_phase": old_phase,
                    "new_phase": self.current_phase,
                    "success_rate": recent_success,
                    "phase_duration_hours": time_in_phase/3600
                }
        
        # Update phase progress
        if current_phase_info.get("timesteps"):
            # Estimate progress based on time and expected duration
            expected_hours = current_phase_info["timesteps"] / 100_000  # Rough estimate
            self.phase_progress = min(1.0, time_in_phase / (expected_hours * 3600))
        
        return self._get_current_status()
    
    def _analyze_recent_collisions(self) -> Dict[str, float]:
        """Enhanced collision analysis for better progress detection"""
        if len(self.collision_history) < 10:
            return {
                "stability_score": 0.0, 
                "gentle_contact_rate": 0.0,
                "successful_grasps": 0,
                "grasp_attempts": 0
            }
        
        recent_collisions = self.collision_history[-50:]  # Look at more history
        
        analysis = {
            "stability_score": 0.0,
            "gentle_contact_rate": 0.0,
            "successful_grasps": 0,
            "physics_stable_episodes": 0,
            "object_flung_episodes": 0,
            "grasp_attempts": 0,
            "contact_episodes": 0
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
                    analysis["gentle_contact_rate"] += 1
                
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
        
        return analysis
    
    def _get_next_phase(self) -> str:
        """Get the next phase in curriculum"""
        phase_order = ["approach_learning", "contact_refinement", "grasping", "manipulation", "mastery"]
        current_idx = phase_order.index(self.current_phase)
        
        if current_idx < len(phase_order) - 1:
            return phase_order[current_idx + 1]
        return None
    
    def _apply_phase_settings(self):
        """Apply current phase settings to environment"""
        current_phase_info = self.phases[self.current_phase]
        
        # Set curriculum level based on phase
        phase_levels = {
            "approach_learning": 0.1,
            "contact_refinement": 0.3,
            "grasping": 0.5,
            "manipulation": 0.8,
            "mastery": 1.0
        }
        
        curriculum_level = phase_levels.get(self.current_phase, 0.5)
        if hasattr(self.env, 'set_curriculum_level'):
            self.env.set_curriculum_level(curriculum_level)
        
        # Apply collision reward settings
        if hasattr(self.env, 'set_collision_rewards'):
            self.env.set_collision_rewards(current_phase_info["collision_rewards"])
            
        # Set termination behavior
        if hasattr(self.env, 'set_collision_termination'):
            self.env.set_collision_termination(current_phase_info.get("termination_on_bad_collision", False))
    
    def _get_current_status(self) -> Dict[str, Any]:
        """Get current curriculum status"""
        return {
            "phase_changed": False,
            "current_phase": self.current_phase,
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