#!/usr/bin/env python3
"""
UR5e Pick-Place Environment with Homestri Integration
This environment is designed for training UR5e robots in pick-and-place tasks
Includes enhanced physics stability, curriculum learning, and detailed logging
"""

import numpy as np
from typing import Dict, Tuple, Optional, Any, List
import mujoco
from pathlib import Path
import random
import time

import gymnasium as gym
from gymnasium import spaces
from homestri_ur5e_rl.envs.mujoco.mujoco_env import MujocoEnv, apply_apple_silicon_fixes
from homestri_ur5e_rl.utils.mujoco_utils import MujocoModelNames, get_site_jac, get_fullM
from homestri_ur5e_rl.utils.controller_utils import pose_error, task_space_inertia_matrix
from homestri_ur5e_rl.utils.transform_utils import quat_multiply, quat_conjugate, quat2mat, mat2quat
from homestri_ur5e_rl.utils.realsense import RealSenseD435iSimulator
from homestri_ur5e_rl.utils.ur5e_stuck_detection_mujoco import StuckDetectionMixin
from homestri_ur5e_rl.utils.domain_randomization import CurriculumDomainRandomizer
from homestri_ur5e_rl.training.curriculum_manager import CurriculumManager

class UR5ePickPlaceEnvEnhanced(MujocoEnv, StuckDetectionMixin):
    """
    UR5e Pick and Place environment with stable physics and reasonable rewards
    """
    
    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array"],
        "render_fps": 200,  
    }
    
    def __init__(
        self,
        xml_file: str = "custom_scene.xml",
        frame_skip: int = 5,
        camera_resolution: int = 64,
        render_mode: Optional[str] = None,
        reward_type: str = "dense",
        control_mode: str = "joint",
        use_stuck_detection: bool = True,
        use_domain_randomization: bool = True,
        curriculum_level: float = 0.1,
        default_camera_config: Optional[Dict] = None,
        **kwargs
    ):
        apply_apple_silicon_fixes()
        
        StuckDetectionMixin.__init__(self)
        
        self.original_render_mode = render_mode
        
        self.camera_resolution = camera_resolution
        self.reward_type = reward_type
        self.control_mode = control_mode
        self.use_stuck_detection = use_stuck_detection
        self.use_domain_randomization = use_domain_randomization
        self.curriculum_level = np.clip(curriculum_level, 0.1, 1.0)
        
        self.max_joint_velocity = 0.3  
        self.max_action_magnitude = 0.1  
        self.action_scale = 0.02 

        self.last_action = None
        self.action_smoothing_factor = 0.3  # Less aggressive smoothing
                
        # Physics stability tracking
        self.physics_stable = True
        self.consecutive_physics_errors = 0
        self.max_consecutive_errors = 3  # Terminate faster
        
        self.table_height = 0.42
        self.table_center = np.array([0.5, 0.0, self.table_height])
        
        self.home_joint_positions = np.array([
            0.0,
            -np.pi/3,
            -np.pi/3,
            -2*np.pi/3,
            np.pi/2,
            0.0
        ])
        
        if default_camera_config is None:
            default_camera_config = {
                'distance': 2.5,
                'azimuth': 120.0,
                'elevation': -25.0,
                'lookat': np.array([0.5, 0.0, 0.6]),
            }
        
        robot_state_dim = 35
        object_state_dim = 13
        goal_dim = 7
        camera_dim = camera_resolution * camera_resolution * 4
        visibility_dim = 1
        
        obs_dim = robot_state_dim + object_state_dim + goal_dim + camera_dim + visibility_dim
        observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        
        fullpath = Path(__file__).parent / "assets" / "base_robot" / xml_file
        
        initial_camera_name = "scene_camera" 
        
        super().__init__(
            model_path=str(fullpath),
            frame_skip=frame_skip,
            observation_space=observation_space,
            render_mode=render_mode,
            camera_name=initial_camera_name,
            camera_id=None,
            default_camera_config=default_camera_config,
            width=640,
            height=480,
            **kwargs
        )
        
        self._initialize_model_references()
        self._setup_enhanced_controllers()
        self._init_camera_simulator()
        
        if self.use_domain_randomization:
            self._setup_domain_randomization()
        
        self.reward_config = {
            'distance_reward_scale': 2.0,        # Enhanced guidance signal
            'approach_bonus': 0.5,               # Meaningful approach bonus  
            'contact_bonus': 1.0,                # Contact reward
            'grasp_bonus': 5.0,                  # Significant grasp reward
            'lift_bonus': 3.0,                   # Lifting object
            'place_bonus': 10.0,                 # Successful placement
            'success_bonus': 20.0,               # Task completion
            
            # Minimal penalties to prevent accumulation
            'time_penalty': -0.0001,             # Ultra-small time penalty
            'energy_penalty': -0.00001,          # Negligible energy penalty
            'velocity_penalty_threshold': 2.0,   # Higher threshold
            'velocity_penalty_scale': -0.01,     # Minimal penalty
            'physics_violation_penalty': -5.0,   # Clear but not extreme penalty
            'stuck_penalty': -0.1,               # Minimal stuck penalty
        }
        
        self._set_init_qpos_to_home()
        self._calculate_optimal_spawning_area()
        
        self.current_object = None
        self.object_grasped = False
        self.object_initial_pos = None
        self.target_position = None
        self.target_orientation = None
        self.step_count = 0
        self.max_episode_steps = 500
        
        self.episode_data = {
            'rewards': [],
            'actions': [],
            'ee_poses': [],
            'object_poses': [],
            'camera_sees_object': [],
            'grasp_attempts': 0,
            'success_time': None,
            'object_in_camera_view_rate': 0.0,
            'physics_instabilities': 0,
        }
        
        if self.use_stuck_detection:
            self.initialize_stuck_detection(0)
            
        print(f" UR5e Environment initialized")
        print(f"   Control mode: {self.control_mode}")
        print(f"   Max velocity: {self.max_joint_velocity} rad/s")
        print(f"   Action scale: {self.action_scale}")
        print(f"   Physics stability checks: ENABLED")
        
        # Initialize curriculum manager
        self.curriculum_manager = CurriculumManager(self)
        self.collision_rewards = {}
        self.episode_collision_data = {}
        self.terminate_on_bad_collision = False
        print(f"   Curriculum: Phase '{self.curriculum_manager.current_phase}'")

    def _initialize_model_references(self):
        """Initialize model references with homestri integration"""
        self.model_names = MujocoModelNames(self.model)
        
        # Robot configuration
        self.robot_prefix = "robot0:"
        
        # Joint names and IDs
        self.robot_joint_names = [
            f"{self.robot_prefix}ur5e:shoulder_pan_joint",
            f"{self.robot_prefix}ur5e:shoulder_lift_joint", 
            f"{self.robot_prefix}ur5e:elbow_joint",
            f"{self.robot_prefix}ur5e:wrist_1_joint",
            f"{self.robot_prefix}ur5e:wrist_2_joint",
            f"{self.robot_prefix}ur5e:wrist_3_joint",
        ]
        
        self.arm_joint_ids = [
            self.model_names.joint_name2id[name] for name in self.robot_joint_names
        ]
        
        # Actuators
        self.actuator_names = [
            f"{self.robot_prefix}ur5e:shoulder_pan",
            f"{self.robot_prefix}ur5e:shoulder_lift",
            f"{self.robot_prefix}ur5e:elbow",
            f"{self.robot_prefix}ur5e:wrist_1",
            f"{self.robot_prefix}ur5e:wrist_2",
            f"{self.robot_prefix}ur5e:wrist_3",
        ]
        
        self.actuator_ids = [
            self.model_names.actuator_name2id[name] for name in self.actuator_names
        ]
        
        # Sites
        self.ee_site_id = self.model_names.site_name2id[f"{self.robot_prefix}eef_site"]
        self.gripper_site_id = self.model_names.site_name2id[f"{self.robot_prefix}2f85:pinch"]
        
        # Gripper
        self.gripper_actuator_id = self.model_names.actuator_name2id[f"{self.robot_prefix}2f85:fingers_actuator"]
        
        # Objects
        self.object_names = ["cube_object", "sphere_object", "cylinder_object"]
        self.object_body_ids = {}
        self.object_geom_ids = {}
        
        for name in self.object_names:
            if name in self.model_names.body_name2id:
                self.object_body_ids[name] = self.model_names.body_name2id[name]
                geom_name = name.replace("_object", "")
                if geom_name in self.model_names.geom_name2id:
                    self.object_geom_ids[name] = self.model_names.geom_name2id[geom_name]
                    
        # Joint limits
        self.joint_limits = np.array([
            self.model.jnt_range[j] for j in self.arm_joint_ids
        ])
        
        # Reasonable workspace bounds
        self.workspace_bounds = {
            'x': [0.1, 0.9],
            'y': [-0.5, 0.5],
            'z': [self.table_height - 0.05, self.table_height + 0.8]
        }

    def _setup_enhanced_controllers(self):
        """Setup enhanced controller with reasonable gains"""
        # Much more conservative control gains
        self.position_gains = np.array([100, 100, 100])  
        self.velocity_gains = np.array([10, 10, 10])    
        
        # Joint space gains
        self.joint_position_gains = np.array([50, 50, 30, 20, 20, 20])  
        self.joint_velocity_gains = np.array([5, 5, 3, 2, 2, 2])      

    def _init_camera_simulator(self):
        """Initialize RealSense camera simulator"""
        realsense_camera_name = "realsense_rgb"
        
        self._camera_sim = None
        self.realsense_camera_name = None
        
        try:
            camera_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, realsense_camera_name)
            if camera_id >= 0:
                self._camera_sim = RealSenseD435iSimulator(
                    self.model,
                    self.data,
                    camera_name=realsense_camera_name,
                    render_resolution=self.camera_resolution
                )
                self.realsense_camera_name = realsense_camera_name
                print(f"RealSense camera simulator initialized with: {realsense_camera_name}")
            else:
                print(f"Camera '{realsense_camera_name}' not found in model")
                    
        except Exception as e:
            print(f"Failed to initialize camera simulator: {e}")

    def _setup_domain_randomization(self):
        """Setup domain randomization with curriculum learning"""
        self.domain_randomizer = CurriculumDomainRandomizer(
            self.model,
            randomize_joints=True,
            randomize_materials=True,
            randomize_lighting=True,
            randomize_camera=True,
        )
        self.domain_randomizer.set_curriculum_level(self.curriculum_level)
        print(f"Domain randomization initialized (level: {self.curriculum_level})")

    def _calculate_optimal_spawning_area(self):
        """Calculate spawning area based on spawn_area_marker in XML"""
        try:
            spawn_marker_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "spawn_area_marker")
            if spawn_marker_id >= 0:
                mujoco.mj_forward(self.model, self.data)
                
                marker_pos = self.data.geom_xpos[spawn_marker_id].copy()
                marker_size = self.model.geom_size[spawn_marker_id].copy()
                
                x_half_width = marker_size[0]
                y_half_width = marker_size[1]
                spawn_z = marker_pos[2] + 0.015
                
                self.object_spawning_area = {
                    'center': np.array([marker_pos[0], marker_pos[1], spawn_z]),
                    'x_range': [marker_pos[0] - x_half_width, marker_pos[0] + x_half_width],
                    'y_range': [marker_pos[1] - y_half_width, marker_pos[1] + y_half_width],
                    'z': spawn_z,
                }
                
                print(f" Spawning area from spawn_area_marker:")
                print(f"   Center: {self.object_spawning_area['center']}")
                print(f"   X range: {self.object_spawning_area['x_range']} (width: {2*x_half_width:.3f}m)")
                print(f"   Y range: {self.object_spawning_area['y_range']} (width: {2*y_half_width:.3f}m)")
                print(f"   Z spawn height: {spawn_z:.3f}m")
                
            else:
                print(" spawn_area_marker not found, using fallback")
                self._calculate_fallback_spawning_area()
                
        except Exception as e:
            print(f" Error reading spawn_area_marker: {e}, using fallback")
            self._calculate_fallback_spawning_area()

    def _calculate_fallback_spawning_area(self):
        """Fallback spawning area calculation"""
        temp_qpos = self.init_qpos.copy()
        temp_qvel = self.init_qvel.copy()
        
        for i, joint_id in enumerate(self.arm_joint_ids):
            qpos_id = self.model.jnt_qposadr[joint_id]
            temp_qpos[qpos_id] = self.home_joint_positions[i]
        
        self.set_state(temp_qpos, temp_qvel)
        mujoco.mj_forward(self.model, self.data)
        
        ee_pos = self.data.site_xpos[self.ee_site_id].copy()
        
        self.object_spawning_area = {
            'center': np.array([ee_pos[0], ee_pos[1], self.table_height + 0.025]),
            'x_range': [ee_pos[0] - 0.15, ee_pos[0] + 0.15],
            'y_range': [ee_pos[1] - 0.15, ee_pos[1] + 0.15],
            'z': self.table_height + 0.025,
        }

    def _set_init_qpos_to_home(self):
        """Set init_qpos to optimized home position"""
        for i, joint_id in enumerate(self.arm_joint_ids):
            qpos_id = self.model.jnt_qposadr[joint_id]
            self.init_qpos[qpos_id] = self.home_joint_positions[i]
                
        self.init_qvel[:] = 0.0
        print(f" Home position set for optimal camera view")

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Step with proper action processing and reasonable rewards"""
        self.step_count += 1
        
        # Simple action processing
        action = np.clip(action, -1.0, 1.0)
        
        # Simple action smoothing (less aggressive)
        if self.last_action is not None:
            action = self.action_smoothing_factor * action + (1 - self.action_smoothing_factor) * self.last_action
        
        # Apply small, consistent action scaling
        scaled_action = action * self.action_scale
        self.last_action = action.copy()
        
        # Apply control
        self._apply_joint_control(scaled_action)
        
        # Step simulation with error handling
        try:
            self.do_simulation(self.data.ctrl, self.frame_skip)
            physics_ok = self._check_physics_stability()
            
            if not physics_ok:
                self.consecutive_physics_errors += 1
                if self.consecutive_physics_errors > self.max_consecutive_errors:
                    print("Physics instability - terminating episode")
                    obs = self._get_obs()
                    return obs, -10.0, True, False, {"physics_error": True}
            else:
                self.consecutive_physics_errors = 0
                
        except Exception as e:
            print(f"Simulation error: {e}")
            obs = self._get_obs()
            return obs, -10.0, True, False, {"simulation_error": str(e)}
        
        # Update states
        self._update_states()
        
        # Get observation and reward
        obs = self._get_obs()
        reward, reward_info = self._compute_reasonable_reward(action)
        
        # Check termination
        terminated = self._check_termination()
        truncated = self.step_count >= self.max_episode_steps
        
        info = {
            "step": self.step_count,
            "object_grasped": self.object_grasped,
            "task_completed": self._check_success(),
            "reward_info": reward_info,
            "physics_stable": physics_ok,
            "camera_sees_object": self._check_camera_sees_object(),
        }
        
        return obs, reward, terminated, truncated, info

    def _apply_joint_control(self, action: np.ndarray):
        """Simple, stable joint control"""
        if len(action) < 6:
            return
            
        # Get current joint velocities
        dof_indices = [self.model.jnt_dofadr[j] for j in self.arm_joint_ids]
        current_velocities = self.data.qvel[dof_indices]
        
        # Apply velocity commands with limits
        for i in range(min(6, len(self.actuator_ids))):
            # Target velocity from action (already scaled)
            target_vel = action[i]
            
            # Apply velocity limits
            target_vel = np.clip(target_vel, -self.max_joint_velocity, self.max_joint_velocity)
            
            # Simple PD control with gravity compensation
            dof_id = self.model.jnt_dofadr[self.arm_joint_ids[i]]
            gravity_comp = self.data.qfrc_bias[dof_id] * 0.5  # Partial gravity comp
            
            # conservative gains
            kp = 10.0   
            kd = 1.0    
            
            current_vel = current_velocities[i]
            
            # Control signal
            control = kp * target_vel - kd * current_vel + gravity_comp
            control = np.clip(control, -50, 50)  # Limit control forces
            
            self.data.ctrl[self.actuator_ids[i]] = control
        
        # Gripper control
        gripper_action = action[6] if len(action) > 6 else 0
        if gripper_action > 0.5:
            self.data.ctrl[self.gripper_actuator_id] = 255
        elif gripper_action < -0.5:
            self.data.ctrl[self.gripper_actuator_id] = 0

    def _compute_reasonable_reward(self, action: np.ndarray) -> Tuple[float, Dict]:
        """Reasonable reward structure that doesn't cause extreme negatives"""
        reward_components = {}
        total_reward = 0.0
        
        # Get current state
        if self.current_object and self.current_object in self.object_body_ids:
            obj_id = self.object_body_ids[self.current_object]
            obj_pos = self.data.body(obj_id).xpos.copy()
            gripper_pos = self.data.site_xpos[self.gripper_site_id].copy()
            
            # Main guidance reward - distance-based, normalized
            dist_to_obj = np.linalg.norm(gripper_pos - obj_pos)
            max_dist = 1.0  # Maximum expected distance
            distance_reward = (max_dist - np.clip(dist_to_obj, 0, max_dist)) / max_dist
            distance_reward *= self.reward_config['distance_reward_scale']
            reward_components['distance'] = distance_reward
            total_reward += distance_reward
            
            # Approach bonus - when getting close
            if dist_to_obj < 0.2:
                approach_bonus = self.reward_config['approach_bonus'] * (0.2 - dist_to_obj) / 0.2
                reward_components['approach'] = approach_bonus
                total_reward += approach_bonus
            
            if dist_to_obj < 0.03 and not hasattr(self, '_contact_achieved'):
                self._contact_achieved = True
                # Force curriculum progression check
                if self.curriculum_manager:
                    self.curriculum_manager.update(1.0)  # Perfect success for contact
            
            # Contact bonus - when very close
            if dist_to_obj < 0.05:
                contact_bonus = self.reward_config['contact_bonus']
                reward_components['contact'] = contact_bonus
                total_reward += contact_bonus
            
            # Grasp detection and rewards
            gripper_closed = self.data.ctrl[self.gripper_actuator_id] > 200
            if not self.object_grasped and gripper_closed and dist_to_obj < 0.03:
                self.object_grasped = True
                grasp_bonus = self.reward_config['grasp_bonus']
                reward_components['grasp'] = grasp_bonus
                total_reward += grasp_bonus
                print(f"🤏 Object grasped! Bonus: {grasp_bonus}")
                
            # Lift bonus
            if self.object_grasped and obj_pos[2] > self.object_initial_pos[2] + 0.05:
                lift_bonus = self.reward_config['lift_bonus']
                reward_components['lift'] = lift_bonus
                total_reward += lift_bonus
                
            # Placement bonus
            if self.target_position is not None:
                dist_to_target = np.linalg.norm(obj_pos[:2] - self.target_position[:2])
                if dist_to_target < 0.05:
                    place_bonus = self.reward_config['place_bonus']
                    reward_components['place'] = place_bonus
                    total_reward += place_bonus
                    
                    if self._check_success():
                        success_bonus = self.reward_config['success_bonus']
                        reward_components['success'] = success_bonus
                        total_reward += success_bonus
        
        # Small, reasonable penalties
        time_penalty = self.reward_config['time_penalty']
        reward_components['time'] = time_penalty
        total_reward += time_penalty
        
        # Energy penalty - very small
        energy_penalty = self.reward_config['energy_penalty'] * np.sum(np.square(action[:6]))
        reward_components['energy'] = energy_penalty
        total_reward += energy_penalty
        
        # Velocity penalty - only for extreme velocities
        joint_velocities = self.data.qvel[:6] if len(self.data.qvel) >= 6 else np.zeros(6)
        high_velocity_count = np.sum(np.abs(joint_velocities) > self.reward_config['velocity_penalty_threshold'])
        if high_velocity_count > 0:
            velocity_penalty = self.reward_config['velocity_penalty_scale'] * high_velocity_count
            reward_components['velocity'] = velocity_penalty
            total_reward += velocity_penalty
        
        # Physics stability penalty
        if not self.physics_stable:
            physics_penalty = self.reward_config['physics_violation_penalty']
            reward_components['physics'] = physics_penalty
            total_reward += physics_penalty
        
        total_reward = np.clip(total_reward, -10.0, 50.0)
        
        return total_reward, reward_components

    def _check_physics_stability(self) -> bool:
        """Check if physics simulation is stable"""
        try:
            # Check for NaN/Inf
            if np.any(np.isnan(self.data.qpos)) or np.any(np.isinf(self.data.qpos)):
                return False
            if np.any(np.isnan(self.data.qvel)) or np.any(np.isinf(self.data.qvel)):
                return False
                
            # Check joint velocities are within reasonable limits
            joint_velocities = self.data.qvel[:6]
            if np.any(np.abs(joint_velocities) > 5.0):  # More lenient
                return False
                
            return True
            
        except Exception:
            return False

    def _update_states(self):
        """Update internal states"""
        self._update_grasp_state()
        
        # Store poses for analysis
        ee_pos = self.data.site_xpos[self.ee_site_id].copy()
        ee_mat = self.data.site_xmat[self.ee_site_id].reshape(3, 3)
        ee_quat = mat2quat(ee_mat)
        self.episode_data['ee_poses'].append(np.concatenate([ee_pos, ee_quat]))
        
        if self.current_object:
            obj_id = self.object_body_ids[self.current_object]
            obj_pos = self.data.body(obj_id).xpos.copy()
            obj_quat = self.data.body(obj_id).xquat.copy()
            self.episode_data['object_poses'].append(np.concatenate([obj_pos, obj_quat]))
            
        # Camera visibility tracking
        sees_object = self._check_camera_sees_object()
        self.episode_data['camera_sees_object'].append(sees_object)

    def _update_grasp_state(self):
        """Simple grasp detection"""
        if not self.current_object:
            return
            
        obj_id = self.object_body_ids[self.current_object]
        obj_pos = self.data.body(obj_id).xpos
        gripper_pos = self.data.site_xpos[self.gripper_site_id]
        
        dist = np.linalg.norm(gripper_pos - obj_pos)
        gripper_closed = self.data.ctrl[self.gripper_actuator_id] > 200
        
        # Release detection
        if self.object_grasped and (not gripper_closed or dist > 0.1):
            self.object_grasped = False
            print(f"📤 Object released")

    def _check_success(self) -> bool:
        """Check success criteria"""
        if not self.current_object or self.target_position is None:
            return False
            
        # Must be physics stable
        if not self.physics_stable:
            return False
            
        try:
            obj_id = self.object_body_ids[self.current_object]
            obj_pos = self.data.body(obj_id).xpos.copy()
            
            # Check object velocity - must be stable
            try:
                obj_vel = np.linalg.norm(self.data.body(obj_id).cvel[:3])
                if obj_vel > 0.5:
                    return False
            except:
                pass
            
            # Standard success: object placed at target
            # Increase tolerance to 5.2cm to account for physics settling precision
            distance_to_target = np.linalg.norm(obj_pos - self.target_position)
            
            # objects should be within reasonable placement zone
            height_diff_abs = abs(obj_pos[2] - self.target_position[2])
            height_diff_signed = obj_pos[2] - self.target_position[2]
            
            # Allow settling below target, but be more strict about objects above target
            # Fine-tuned thresholds based on actual physics settling analysis
            if height_diff_signed > 0:  # Object above target
                height_correct = height_diff_abs <= 0.024  # 2.4cm above target max (fine-tuned for physics)
            else:  # Object below target (more lenient for physics settling)
                height_correct = height_diff_abs <= 0.0225  # 2.25cm below target max (fine-tuned for physics)
            
            placement_success = distance_to_target < 0.052 and height_correct
            
            return placement_success
                
        except Exception:
            return False

    def _check_termination(self) -> bool:
        """Check termination conditions"""
        # Physics breakdown
        if not self.physics_stable and self.consecutive_physics_errors >= self.max_consecutive_errors:
            print("❌ Terminating: Physics instability")
            return True
            
        # Success
        if self._check_success():
            print("✅ Task completed successfully!")
            return True
            
        # Dropped object - attempt to respawn if physics are unstable
        if self.current_object in self.object_body_ids:
            obj_id = self.object_body_ids[self.current_object]
            obj_z = self.data.body(obj_id).xpos[2]
            if obj_z < self.table_height - 0.1:
                print("❌ Object dropped!")
                
                # If domain randomization caused instability, try to respawn once
                if self.use_domain_randomization and not hasattr(self, '_respawn_attempted'):
                    print("⚠️ Attempting to respawn dropped object with better physics...")
                    self._respawn_attempted = True
                    self._reset_object_in_spawning_area()
                    # Don't terminate, give it a chance to work
                    return False
                else:
                    return True
            
        return False

    def _check_camera_sees_object(self) -> bool:
        """Enhanced camera visibility check"""
        if not self._camera_sim or not self.current_object:
            return False

        try:
            obj_id = self.object_body_ids[self.current_object]
            obj_pos = self.data.body(obj_id).xpos
            
            camera_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "realsense_camera")
            if camera_body_id >= 0:
                camera_body_pos = self.data.xpos[camera_body_id]
                camera_body_mat = self.data.xmat[camera_body_id].reshape(3, 3)
                
                rgb_relative_pos = np.array([0, 0.015, 0])
                camera_pos = camera_body_pos + camera_body_mat @ rgb_relative_pos
                
                cos_x, sin_x = 0.0, 1.0
                rotation_x = np.array([
                    [1,     0,      0],
                    [0, cos_x, -sin_x],
                    [0, sin_x,  cos_x]
                ])
                camera_mat = camera_body_mat @ rotation_x
            else:
                camera_pos = self.data.site_xpos[self.ee_site_id]
                camera_mat = self.data.site_xmat[self.ee_site_id].reshape(3, 3)

            camera_forward = -camera_mat[:, 2]
            camera_right = camera_mat[:, 0]
            camera_up = camera_mat[:, 1]
            
            to_object = obj_pos - camera_pos
            distance = np.linalg.norm(to_object)
            
            if distance < self._camera_sim.min_depth or distance > self._camera_sim.max_depth:
                return False
                
            to_object_norm = to_object / distance
            
            forward_dot = np.dot(to_object_norm, camera_forward)
            if forward_dot < 0.1:
                return False
                
            right_component = np.dot(to_object, camera_right)
            up_component = np.dot(to_object, camera_up)
            forward_component = np.dot(to_object, camera_forward)
            
            horizontal_angle = np.abs(np.arctan2(right_component, forward_component))
            vertical_angle = np.abs(np.arctan2(up_component, forward_component))
            
            horizontal_fov_half = np.radians(self._camera_sim.rgb_fov_horizontal / 2)
            vertical_fov_half = np.radians(self._camera_sim.rgb_fov_vertical / 2)
            
            in_fov = (horizontal_angle < horizontal_fov_half and 
                    vertical_angle < vertical_fov_half)
            
            return in_fov

        except Exception as e:
            return False

    def reset_model(self) -> np.ndarray:
        """Reset with guaranteed robot positioning and physics stability"""
        # Reset to initial state
        self.set_state(self.init_qpos, self.init_qvel)

        # Apply domain randomization if enabled
        if self.use_domain_randomization and hasattr(self, 'domain_randomizer'):
            self.domain_randomizer.randomize()

        # Set robot to EXACT home position with multiple enforcement steps
        for i, joint_id in enumerate(self.arm_joint_ids):
            qpos_id = self.model.jnt_qposadr[joint_id]
            dof_id = self.model.jnt_dofadr[joint_id]
            # Set exact position and zero velocity
            self.data.qpos[qpos_id] = self.home_joint_positions[i]
            self.data.qvel[dof_id] = 0.0
        
        # Set gripper to open
        self.data.ctrl[self.gripper_actuator_id] = 0.0
        
        # Multiple physics updates to ensure stability
        for _ in range(10):
            # Keep enforcing exact positions
            for i, joint_id in enumerate(self.arm_joint_ids):
                qpos_id = self.model.jnt_qposadr[joint_id]
                dof_id = self.model.jnt_dofadr[joint_id]
                self.data.qpos[qpos_id] = self.home_joint_positions[i]
                self.data.qvel[dof_id] = 0.0
            
            mujoco.mj_forward(self.model, self.data)
        
        # Initialize control targets for operational space mode
        if self.control_mode == "operational_space":
            mujoco.mj_forward(self.model, self.data)
            self.target_ee_pos = self.data.site_xpos[self.ee_site_id].copy()
            ee_mat = self.data.site_xmat[self.ee_site_id].reshape(3, 3)
            self.target_ee_quat = mat2quat(ee_mat)

        # Reset object with guaranteed visibility
        max_spawn_attempts = 10
        for attempt in range(max_spawn_attempts):
            self._reset_object_in_spawning_area()
            
            # Let physics settle briefly
            for _ in range(5):
                mujoco.mj_step(self.model, self.data)
            
            # Check visibility
            if self._check_camera_sees_object():
                print(f" Object visible on attempt {attempt + 1}")
                break
            else:
                print(f"Object not visible, attempt {attempt + 1}/{max_spawn_attempts}")
        
        # Set target location
        self._set_target_location()

        # Reset tracking
        self._reset_episode_tracking()

        # Reset stuck detection
        if self.use_stuck_detection:
            self.initialize_stuck_detection(0)

        # Update curriculum manager
        if hasattr(self, 'episode_collision_data') and hasattr(self, 'curriculum_manager'):
            recent_success = float(self._check_success()) if hasattr(self, '_check_success') else 0.0
            curriculum_status = self.curriculum_manager.update(
                success_rate=recent_success,
                collision_info=self.episode_collision_data
            )
            self.curriculum_manager.log_collision_episode(self.episode_collision_data)
            
            if curriculum_status.get('phase_changed', False):
                print(f"🎓 Curriculum advanced to: {curriculum_status['new_phase']}")
        
        # Reset episode collision data
        self.episode_collision_data = {}
        
        # Reset respawn attempt flag
        if hasattr(self, '_respawn_attempted'):
            delattr(self, '_respawn_attempted')
        
        # Get observation for returning AND for object perception logging
        obs = self._get_obs()
        
        # IMPORTANT: Let physics settle and rendering update before logging perception
        # This ensures the CNN gets the updated visual feed
        for _ in range(3):  # Small number of physics steps
            mujoco.mj_step(self.model, self.data)
        
        # Get fresh observation after physics settling
        obs_after_settling = self._get_obs()
        
        # Log object perception for debugging CNN/visual learning
        self._log_object_perception_during_reset(obs_after_settling)
        
        return obs

    def _reset_object_in_spawning_area(self):
        """Reset object in spawning area with randomized properties"""
        # Hide all objects first
        for obj_name in self.object_names:
            if obj_name in self.object_body_ids:
                obj_id = self.object_body_ids[obj_name]
                self._hide_object(obj_id)
                
        # Select object based on curriculum
        if self.curriculum_level < 0.3:
            possible_objects = ["cube_object"]
        elif self.curriculum_level < 0.7:
            possible_objects = ["cube_object", "sphere_object"]
        else:
            possible_objects = self.object_names
            
        available_objects = [obj for obj in possible_objects if obj in self.object_body_ids]
        if not available_objects:
            available_objects = [obj for obj in self.object_names if obj in self.object_body_ids]
            
        self.current_object = random.choice(available_objects)
        obj_id = self.object_body_ids[self.current_object]
        
        # Apply randomized properties only if domain randomization is enabled
        if self.use_domain_randomization:
            self._randomize_object_properties(obj_id)
        else:
            # Log default object properties without randomization
            self._log_default_object_properties(obj_id)
        
        # Position within spawning area
        x = np.random.uniform(*self.object_spawning_area['x_range'])
        y = np.random.uniform(*self.object_spawning_area['y_range'])
        z = self.object_spawning_area['z']
        
        # Set position
        body = self.model.body(obj_id)
        if body.jntadr[0] >= 0:
            qpos_adr = self.model.jnt_qposadr[body.jntadr[0]]
            self.data.qpos[qpos_adr:qpos_adr+3] = [x, y, z]
            
            # Set stable orientation
            if "cylinder" in self.current_object:
                self.data.qpos[qpos_adr+3:qpos_adr+7] = [0.707, 0.707, 0, 0]
            else:
                self.data.qpos[qpos_adr+3:qpos_adr+7] = [1, 0, 0, 0]
                
            # Zero velocities
            if body.jntadr[0] >= 0 and body.jntadr[0] < self.model.njnt:
                dof_adr = self.model.jnt_dofadr[body.jntadr[0]]
                self.data.qvel[dof_adr:dof_adr+6] = 0
                
        mujoco.mj_forward(self.model, self.data)
        
        # Ensure stable physics after object spawning
        self._stabilize_object_physics(obj_id)
        
        # Forward simulate a few steps to ensure stable placement
        for _ in range(10):  # Small simulation to settle physics
            mujoco.mj_step(self.model, self.data)
        
        # Store initial position
        self.object_initial_pos = self.data.body(obj_id).xpos.copy()
        
        actual_pos = self.data.body(obj_id).xpos.copy()
        print(f" {self.current_object} spawned with randomized properties:")
        print(f"   Target position: [{x:.3f}, {y:.3f}, {z:.3f}]")
        print(f"   Actual position: [{actual_pos[0]:.3f}, {actual_pos[1]:.3f}, {actual_pos[2]:.3f}]")
        print(f"   Spawning area: X[{self.object_spawning_area['x_range'][0]:.2f}, {self.object_spawning_area['x_range'][1]:.2f}], Y[{self.object_spawning_area['y_range'][0]:.2f}, {self.object_spawning_area['y_range'][1]:.2f}]")
        print(f"   Properties: {getattr(self, '_current_object_properties', 'Applied')}")

    def _stabilize_object_physics(self, obj_id: int):
        """Improve physics stability for spawned objects"""
        try:
            # Improve contact parameters for more stable physics
            for geom_id in range(self.model.ngeom):
                if self.model.geom_bodyid[geom_id] == obj_id:
                    # More stable contact parameters
                    self.model.geom_solimp[geom_id][0] = 0.9   # Higher impedance for stability
                    self.model.geom_solimp[geom_id][1] = 0.95  # Better damping
                    self.model.geom_solimp[geom_id][2] = 0.001 # Small time constant
                    
                    # Better friction for stability  
                    self.model.geom_friction[geom_id][0] = max(0.8, self.model.geom_friction[geom_id][0])  # Sliding friction
                    self.model.geom_friction[geom_id][1] = 0.005  # Torsional friction
                    self.model.geom_friction[geom_id][2] = 0.0001  # Rolling friction
                    
                    # Ensure minimum contact margins
                    self.model.geom_margin[geom_id] = 0.001  # Small margin for better contact detection
                    
        except Exception as e:
            print(f"Failed to stabilize object physics: {e}")

    def _log_default_object_properties(self, obj_id: int):
        """Log default object properties without randomization"""
        try:
            body = self.model.body(obj_id)
            properties = {}
            
            # Default mass
            properties['mass'] = f"{body.mass[0]*1000:.0f}g"
            
            # Default size (read from original XML)
            for geom_id in range(self.model.ngeom):
                if self.model.geom_bodyid[geom_id] == obj_id:
                    original_size = self.model.geom_size[geom_id].copy()
                    
                    if self.model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_BOX:
                        properties['size'] = f"box({original_size[0]:.3f}x{original_size[1]:.3f}x{original_size[2]:.3f})"
                    elif self.model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_SPHERE:
                        properties['size'] = f"sphere(r={original_size[0]:.3f})"
                    elif self.model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_CYLINDER:
                        properties['size'] = f"cylinder(r={original_size[0]:.3f}, h={original_size[1]:.3f})"
            
            # Default properties
            properties['color'] = 'Default'
            properties['texture'] = 'default'
            properties['friction'] = 'default'
            properties['bounce'] = 'default'
            properties['density'] = '1.0x'
            
            self._current_object_properties = properties
            
        except Exception as e:
            print(f" Failed to log default object properties: {e}")
            self._current_object_properties = {"error": str(e)}

    def _randomize_object_properties(self, obj_id: int):
        """Randomize object properties"""
        try:
            body = self.model.body(obj_id)
            properties = {}
            
            # Mass randomization (50g - 500g)
            base_mass = 0.1
            mass_variation = np.random.uniform(0.5, 5.0)
            new_mass = base_mass * mass_variation
            
            if body.rootid >= 0:
                self.model.body_mass[obj_id] = new_mass
                properties['mass'] = f"{new_mass*1000:.0f}g"
            
            # Size randomization (±20%)
            size_scale = np.random.uniform(0.8, 1.2)
            
            for geom_id in range(self.model.ngeom):
                if self.model.geom_bodyid[geom_id] == obj_id:
                    original_size = self.model.geom_size[geom_id].copy()  # Store original before modification
                    
                    if self.model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_BOX:
                        # Apply scaling to the model
                        self.model.geom_size[geom_id] = original_size * size_scale
                        # Log using original size (before modification) 
                        properties['size'] = f"box({original_size[0]*size_scale:.3f}x{original_size[1]*size_scale:.3f}x{original_size[2]*size_scale:.3f})"
                        
                    elif self.model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_SPHERE:
                        # Apply scaling to the model
                        self.model.geom_size[geom_id][0] = original_size[0] * size_scale
                        # Log using original size
                        properties['size'] = f"sphere(r={original_size[0]*size_scale:.3f})"
                        
                    elif self.model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_CYLINDER:
                        # Apply scaling to the model
                        self.model.geom_size[geom_id][0] = original_size[0] * size_scale
                        self.model.geom_size[geom_id][1] = original_size[1] * size_scale
                        # Log using original size
                        properties['size'] = f"cylinder(r={original_size[0]*size_scale:.3f}, h={original_size[1]*size_scale:.3f})"
            
            # Color randomization
            realistic_colors = [
                [0.8, 0.2, 0.2, 1.0], [0.2, 0.8, 0.2, 1.0], [0.2, 0.2, 0.8, 1.0],
                [0.8, 0.6, 0.2, 1.0], [0.6, 0.2, 0.8, 1.0], [0.8, 0.8, 0.2, 1.0],
                [0.4, 0.2, 0.1, 1.0], [0.7, 0.7, 0.7, 1.0], [0.1, 0.1, 0.1, 1.0],
                [0.9, 0.9, 0.9, 1.0],
            ]
            
            selected_color = random.choice(realistic_colors)
            color_names = ['Red', 'Green', 'Blue', 'Orange', 'Purple', 'Yellow', 'Brown', 'Gray', 'Black', 'White']
            color_name = color_names[realistic_colors.index(selected_color)]
            
            for geom_id in range(self.model.ngeom):
                if self.model.geom_bodyid[geom_id] == obj_id:
                    self.model.geom_rgba[geom_id] = selected_color
            
            properties['color'] = color_name
            
            # Material properties
            shininess = np.random.uniform(0.1, 0.9)
            friction_coeff = np.random.uniform(0.5, 1.5)
            restitution = np.random.uniform(0.1, 0.6)
            
            for geom_id in range(self.model.ngeom):
                if self.model.geom_bodyid[geom_id] == obj_id:
                    self.model.geom_friction[geom_id][0] = friction_coeff
                    self.model.geom_friction[geom_id][1] = friction_coeff * 0.1
                    self.model.geom_friction[geom_id][2] = friction_coeff * 0.05
                    self.model.geom_solimp[geom_id][0] = restitution
            
            properties['texture'] = f"shine={shininess:.2f}"
            properties['friction'] = f"{friction_coeff:.2f}"
            properties['bounce'] = f"{restitution:.2f}"
            
            density_variation = np.random.uniform(0.7, 1.8)
            properties['density'] = f"{density_variation:.2f}x"
            
            self._current_object_properties = properties
            
        except Exception as e:
            print(f" Failed to randomize object properties: {e}")
            self._current_object_properties = {"error": str(e)}

    def _set_target_location(self):
        """Set target locations optimized for camera visibility"""
        if self.curriculum_level < 0.5:
            targets = [
                np.array([0.25, 0.0, self.table_height + 0.03]),
                np.array([0.35, 0.25, self.table_height + 0.03]),
                np.array([0.4, 0.1, self.table_height + 0.03]),
            ]
        else:
            targets = [
                np.array([0.2, -0.1, self.table_height + 0.03]),
                np.array([0.45, -0.15, self.table_height + 0.03]),
                np.array([0.3, 0.3, self.table_height + 0.03]),
                np.array([0.5, 0.2, self.table_height + 0.03]),
            ]
            
        self.target_position = random.choice(targets)
        self.target_orientation = np.array([1, 0, 0, 0])
        self.target_radius = 0.05 * (2.0 - self.curriculum_level)
        
        print(f" Target set at {self.target_position}")

    def _hide_object(self, obj_id: int):
        """Hide object by moving it away"""
        body = self.model.body(obj_id)
        if body.jntadr[0] != -1:
            joint_type = self.model.jnt_type[body.jntadr[0]]
            if joint_type == mujoco.mjtJoint.mjJNT_FREE:
                qpos_adr = self.model.jnt_qposadr[body.jntadr[0]]
                self.data.qpos[qpos_adr:qpos_adr+3] = [10, 10, -10]

    def _reset_episode_tracking(self):
        """Reset episode tracking"""
        self.step_count = 0
        self.object_grasped = False
        self.last_action = None
        self.consecutive_physics_errors = 0
        self.physics_stable = True
        self.episode_data = {
            'rewards': [],
            'actions': [],
            'ee_poses': [],
            'object_poses': [],
            'camera_sees_object': [],
            'grasp_attempts': 0,
            'success_time': None,
            'object_in_camera_view_rate': 0.0,
        }
        
        # Increment episode counter for object perception logging
        if not hasattr(self, 'reset_episode_count'):
            self.reset_episode_count = 0
    
    def _log_object_perception_during_reset(self, obs):
        """Log what the CNN should perceive vs what was actually spawned"""
        self.reset_episode_count += 1
        
        # Log EVERY episode now for immediate feedback
        try:
            # Get current object info - use the correct attribute names
            current_object = getattr(self, 'current_object', 'unknown')
            
            # Get object size from the spawned object
            if current_object != 'unknown' and hasattr(self, 'object_body_ids'):
                obj_id = self.object_body_ids.get(current_object)
                if obj_id is not None:
                    # Get the actual size from MuJoCo model
                    current_object_size = self._get_object_size_from_model(obj_id)
                else:
                    current_object_size = 'unknown_id'
            else:
                current_object_size = 'unknown'
            
            print(f"🔍 Object Perception Check - Episode {self.reset_episode_count}")
            print(f"   📦 Spawned Object: {current_object}")
            print(f"   📏 Object Size: {current_object_size}")
            
            # Extract camera observation from full observation
            if len(obs) >= 16384:  # Ensure we have camera data
                # Camera observation is RGB (12288) + Depth (4096) = 16384 pixels
                rgb_start = 56  # After kinematic features
                rgb_end = rgb_start + 12288  # 64x64x3 = 12288
                depth_start = rgb_end
                depth_end = depth_start + 4096  # 64x64x1 = 4096
                
                rgb_obs = obs[rgb_start:rgb_end]
                depth_obs = obs[depth_start:depth_end]
                
                # Analyze RGB content
                rgb_mean = np.mean(rgb_obs)
                rgb_std = np.std(rgb_obs)
                rgb_min, rgb_max = np.min(rgb_obs), np.max(rgb_obs)
                
                # Analyze depth content  
                depth_mean = np.mean(depth_obs)
                depth_std = np.std(depth_obs)
                depth_min, depth_max = np.min(depth_obs), np.max(depth_obs)
                
                print(f"   👁️ CNN Visual Input:")
                print(f"      RGB: mean={rgb_mean:.3f}, std={rgb_std:.3f}, range=[{rgb_min:.3f}, {rgb_max:.3f}]")
                print(f"      Depth: mean={depth_mean:.3f}, std={depth_std:.3f}, range=[{depth_min:.3f}, {depth_max:.3f}]")
                
                # Check if we have meaningful visual content
                rgb_has_content = rgb_std > 0.01  # Some visual variation
                depth_has_content = depth_std > 0.01  # Some depth variation
                
                print(f"      Visual content detected: RGB={rgb_has_content}, Depth={depth_has_content}")
                
                if not (rgb_has_content or depth_has_content):
                    print(f"      ⚠️ WARNING: Very low visual variation detected!")
                    
            else:
                print(f"   ⚠️ Observation too short for camera analysis: {len(obs)} < 16384")
                
        except Exception as e:
            print(f"   ⚠️ Object perception logging failed: {e}")
    
    def _get_object_size_from_model(self, obj_id):
        """Get object size from MuJoCo model for perception logging"""
        try:
            # Get geom associated with the body
            for geom_id in range(self.model.ngeom):
                if self.model.geom_bodyid[geom_id] == obj_id:
                    geom_size = self.model.geom_size[geom_id]
                    geom_type = self.model.geom_type[geom_id]
                    
                    if geom_type == 6:  # Box type
                        return f"box({geom_size[0]:.3f}x{geom_size[1]:.3f}x{geom_size[2]:.3f})"
                    elif geom_type == 2:  # Sphere type
                        return f"sphere(r={geom_size[0]:.3f})"
                    elif geom_type == 5:  # Cylinder type  
                        return f"cylinder(r={geom_size[0]:.3f}, h={geom_size[1]:.3f})"
                    else:
                        return f"geom_type_{geom_type}({geom_size[0]:.3f})"
                        
            return "size_unknown"
        except Exception as e:
            return f"size_error: {e}"

    def _get_obs(self) -> np.ndarray:
        """Get real-time observation with proper camera handling"""
        try:
            # Robot state (35 dim)
            robot_obs = self._get_robot_obs()
            
            # Object state (13 dim) 
            object_obs = self._get_object_obs()
            
            # Goal state (7 dim)
            goal_obs = self._get_goal_obs()
            
            # Get REAL-TIME camera data (not stored)
            camera_obs = self._get_camera_obs()
            
            # Current visibility flag (1 dim)
            visibility = np.array([float(self._check_camera_sees_object())])
            
            # Combine all observations
            full_obs = np.concatenate([
                robot_obs,      # 35
                object_obs,     # 13
                goal_obs,       # 7
                camera_obs,     # camera_resolution^2 * 4
                visibility      # 1
            ]).astype(np.float32)
            
            # Verify dimensions
            expected_dim = 35 + 13 + 7 + (self.camera_resolution ** 2 * 4) + 1
            assert full_obs.shape[0] == expected_dim, \
                f"Observation dimension mismatch: {full_obs.shape[0]} vs {expected_dim}"
            
            return full_obs
            
        except Exception as e:
            print(f" Observation error: {e}")
            expected_dim = 35 + 13 + 7 + (self.camera_resolution ** 2 * 4) + 1
            return np.zeros(expected_dim, dtype=np.float32)

    def _get_robot_obs(self) -> np.ndarray:
        """Get robot state observation (35 dimensions)"""
        try:
            # Joint positions (6)
            qpos_indices = [self.model.jnt_qposadr[j] for j in self.arm_joint_ids]
            joint_pos = self.data.qpos[qpos_indices]
            
            # Joint velocities (6)
            dof_indices = [self.model.jnt_dofadr[j] for j in self.arm_joint_ids]
            joint_vel = self.data.qvel[dof_indices]
            
            # End-effector pose (7: position + quaternion)
            ee_pos = self.data.site_xpos[self.ee_site_id].copy()
            ee_mat = self.data.site_xmat[self.ee_site_id].reshape(3, 3)
            ee_quat = mat2quat(ee_mat)
            
            # Gripper state (2)
            gripper_pos = self.data.ctrl[self.gripper_actuator_id] / 255.0
            gripper_vel = 0.0
            
            # Control forces (6)
            ctrl_forces = self.data.ctrl[self.actuator_ids]
            
            # Contact forces (3)
            ee_contact_force = np.zeros(3)
            
            # Stability metrics (3)
            max_joint_vel = np.max(np.abs(joint_vel))
            workspace_violation = 0.0 if self._check_workspace_bounds(ee_pos) else 1.0
            stability_score = 1.0 if self.physics_stable else 0.0
            stability_metrics = np.array([max_joint_vel, workspace_violation, stability_score])
            
            # Operational state (2)
            grasp_state = 1.0 if self.object_grasped else 0.0
            step_normalized = self.step_count / 1000.0
            operational_state = np.array([grasp_state, step_normalized])
            
            # Combine (6+6+7+2+6+3+3+2 = 35)
            robot_obs = np.concatenate([
                joint_pos,           # 6
                joint_vel,           # 6
                ee_pos,              # 3
                ee_quat,             # 4
                [gripper_pos, gripper_vel],  # 2
                ctrl_forces,         # 6
                ee_contact_force,    # 3
                stability_metrics,   # 3
                operational_state    # 2
            ])
            
            return robot_obs.astype(np.float32)
            
        except Exception as e:
            print(f" Robot obs error: {e}")
            return np.zeros(35, dtype=np.float32)

    def _get_object_obs(self) -> np.ndarray:
        """Get object state observation (13 dimensions)"""
        try:
            if self.current_object and self.current_object in self.object_body_ids:
                obj_id = self.object_body_ids[self.current_object]
                
                # Object position (3)
                obj_pos = self.data.body(obj_id).xpos.copy()
                
                # Object orientation (4: quaternion)
                obj_quat = self.data.body(obj_id).xquat.copy()
                
                # Object velocity (3)
                obj_vel = np.zeros(3)
                
                # Object-robot relative state (3)
                ee_pos = self.data.site_xpos[self.ee_site_id].copy()
                obj_to_ee = ee_pos - obj_pos
                distance_to_object = np.linalg.norm(obj_to_ee)
                relative_pos = obj_to_ee / max(distance_to_object, 0.01)
                
                # Combine (3+4+3+3 = 13)
                object_obs = np.concatenate([
                    obj_pos,        # 3
                    obj_quat,       # 4
                    obj_vel,        # 3
                    relative_pos    # 3
                ])
                
            else:
                object_obs = np.zeros(13)
                
            return object_obs.astype(np.float32)
            
        except Exception as e:
            print(f" Object obs error: {e}")
            return np.zeros(13, dtype=np.float32)

    def _get_goal_obs(self) -> np.ndarray:
        """Get goal state observation (7 dimensions)"""
        try:
            if self.target_position is not None:
                # Target position (3)
                target_pos = self.target_position.copy()
                
                # Target orientation (4: quaternion)
                target_quat = self.target_orientation.copy() if self.target_orientation is not None else np.array([1, 0, 0, 0])
                
                # Combine (3+4 = 7)
                goal_obs = np.concatenate([target_pos, target_quat])
                
            else:
                goal_obs = np.zeros(7)
                
            return goal_obs.astype(np.float32)
            
        except Exception as e:
            print(f" Goal obs error: {e}")
            return np.zeros(7, dtype=np.float32)

    def _get_camera_obs(self) -> np.ndarray:
        """Get real-time camera observation"""
        try:
            # Get real-time camera data
            if self._camera_sim and self._camera_sim.is_available():
                return self._camera_sim.render_rgbd()
            
            # Fallback
            camera_dim = self.camera_resolution * self.camera_resolution * 4
            return np.zeros(camera_dim, dtype=np.float32)
            
        except Exception as e:
            print(f" Camera obs error: {e}")
            camera_dim = self.camera_resolution * self.camera_resolution * 4
            return np.zeros(camera_dim, dtype=np.float32)

    def _check_workspace_bounds(self, position: np.ndarray) -> bool:
        """Check if position is within workspace bounds"""
        x, y, z = position
        return (self.workspace_bounds['x'][0] <= x <= self.workspace_bounds['x'][1] and
                self.workspace_bounds['y'][0] <= y <= self.workspace_bounds['y'][1] and
                self.workspace_bounds['z'][0] <= z <= self.workspace_bounds['z'][1])

    # Curriculum and collision methods
    def set_collision_rewards(self, collision_rewards: Dict[str, float]):
        """Set collision reward configuration"""
        self.collision_rewards = collision_rewards
        
    def set_collision_termination(self, terminate: bool):
        """Set collision termination behavior"""
        self.terminate_on_bad_collision = terminate

    def set_curriculum_level(self, level: float):
        """Set curriculum level"""
        self.curriculum_level = np.clip(level, 0.1, 1.0)
        if hasattr(self, 'domain_randomizer'):
            self.domain_randomizer.set_curriculum_level(self.curriculum_level)

    def get_episode_info(self) -> Dict:
        """Get episode information"""
        camera_visibility_rate = (np.mean(self.episode_data['camera_sees_object']) 
                                 if self.episode_data['camera_sees_object'] else 0)
        
        return {
            'total_reward': sum(self.episode_data['rewards']),
            'episode_length': self.step_count,
            'success': self._check_success(),
            'success_time': self.episode_data['success_time'],
            'grasp_attempts': self.episode_data['grasp_attempts'],
            'camera_visibility_rate': camera_visibility_rate,
            'object_in_camera_view_rate': self.episode_data['object_in_camera_view_rate'],
            'curriculum_level': self.curriculum_level,
            'object_type': self.current_object,
            'object_grasped': self.object_grasped,
            'physics_stable': self.physics_stable,
        }

    def close(self):
        """Enhanced cleanup"""
        try:
            if hasattr(self, '_camera_sim') and self._camera_sim is not None:
                if hasattr(self._camera_sim, 'close'):
                    self._camera_sim.close()
        except Exception:
            pass
        super().close()

    def switch_to_realsense_view(self):
        """Switch MuJoCo viewer to RealSense camera view"""
        try:
            camera_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "realsense_rgb")
            if camera_id < 0:
                print(f"RealSense camera 'realsense_rgb' not found in model")
                return False
            
            self.camera_name = "realsense_rgb"
            self.camera_id = camera_id
            
            if hasattr(self, 'mujoco_renderer') and self.mujoco_renderer is not None:
                self.mujoco_renderer.camera.type = mujoco.mjtCamera.mjCAMERA_FIXED
                self.mujoco_renderer.camera.fixedcamid = camera_id
                print(f"Switched to RealSense camera view (ID: {camera_id})")
                return True
            
            print(f"RealSense camera set (ID: {camera_id}) - will apply on next render")
            return True
            
        except Exception as e:
            print(f"Failed to switch to RealSense camera: {e}")
            return False

    def switch_to_scene_view(self):
        """Switch MuJoCo viewer to scene camera view"""
        try:
            camera_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "scene_camera")
            if camera_id < 0:
                print(f"Scene camera not found")
                return False
            
            self.camera_name = "scene_camera"
            self.camera_id = camera_id
            
            if hasattr(self, 'mujoco_renderer') and self.mujoco_renderer is not None:
                self.mujoco_renderer.camera.type = mujoco.mjtCamera.mjCAMERA_FIXED
                self.mujoco_renderer.camera.fixedcamid = camera_id
                print(f"Switched to scene camera view (ID: {camera_id})")
                return True
                
            return True
            
        except Exception as e:
            print(f"Failed to switch to scene camera: {e}")
            return False