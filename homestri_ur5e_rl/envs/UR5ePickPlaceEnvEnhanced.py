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
try:
    from homestri_ur5e_rl.training.curriculum_manager import CurriculumManager
except ImportError:
    CurriculumManager = None 
from homestri_ur5e_rl.controllers.joint_position_controller import JointPositionController

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
        curriculum_level: float = 0.05,  # Start with NEAR SPAWN for milestone_0_percent
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
        self.curriculum_level = np.clip(curriculum_level, 0.01, 1.0)  # CRITICAL FIX: Allow levels as low as 0.01 for milestone_0_percent (0.05)
        # Default target radius used in goal observation; updated when target is set
        self.target_radius = 0.08
        self.max_joint_velocity = 0.5
        self.max_action_magnitude = 0.25  # Increased for more diverse actions
        self.action_scale = 0.15  # Increased from 0.1 to encourage exploration
        self.action_smoothing_factor = 0.3
        self.max_joint_delta_normal = 0.05
        self.max_joint_delta_early = 0.07
        self.approach_threshold_enter = None
        self.approach_threshold_exit = None
        self._in_approach_zone = False
        self._planar_assist_active = False
        self.prev_vertical_signed = None
        self.physics_stable = True
        self.consecutive_physics_errors = 0
        self.max_consecutive_errors = 3
        self.table_height = 0.42
        self.reset_episode_count = 0
        self.table_center = np.array([0.5, 0.0, self.table_height])
        self.home_joint_positions = np.array([
            0.0,
            -np.pi/3,   # slightly lower shoulder lift for closer vertical start
            -np.pi/3,   # slightly more elbow flexion
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
        observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
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
        self.reward_config = {
            'approach_bonus': 0.8,
            'contact_bonus': 1.5,
            'grasp_bonus': 2.0,
            'lift_bonus': 2.0,
            'place_bonus': 3.0,
            'success_bonus': 5.0,
            'physics_violation_penalty': -5.0,
        }
        if not hasattr(self, 'max_episode_steps'):
            self.max_episode_steps = 200
        self._initialize_model_references()
        self._init_camera_simulator()
        self._calculate_optimal_spawning_area()
        self._set_init_qpos_to_home()
        self._setup_domain_randomization()
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)
        self.use_delta_position_control = True
        self.use_external_joint_position_controller = True
        self.joint_kp = 120.0
        self.joint_kd = 4.0
        self.target_joint_positions = None
        self.prev_dist_to_object = None
        self.total_timesteps_seen = 0
        if self.use_external_joint_position_controller:
            self._setup_external_joint_position_controller()
        self.last_action = None

        self._jac_tmp_jacp = None
        self._jac_tmp_jacr = None
        self._joint_z_signs = None
        self._contact_breakthrough = False  # tighten contact thresholds once achieved
        # New state for vertical stall / escalation
        self._vertical_stall_steps = 0
        self._escalation_active_until_step = 0
        self._last_vertical_signed = None

    def _setup_external_joint_position_controller(self):
        try:
            min_pos = self.joint_limits[:, 0].tolist()
            max_pos = self.joint_limits[:, 1].tolist()
            min_effort = []
            max_effort = []
            for act_id in self.actuator_ids:
                ctrl_low, ctrl_high = self.model.actuator_ctrlrange[act_id]
                gear = self.model.actuator_gear[act_id][0] if self.model.actuator_gear.shape[0] > act_id else 100.0
                min_effort.append(ctrl_low * gear)
                max_effort.append(ctrl_high * gear)
            kp = [120, 120, 120, 60, 40, 30]
            kd = [int(np.sqrt(k)+0.5) for k in kp]
            self.joint_position_controller = JointPositionController(
                self.model,
                self.data,
                self.model_names,
                eef_name=f"{self.robot_prefix}eef_site",
                joint_names=self.robot_joint_names,
                actuator_names=self.actuator_names,
                kp=kp,
                kd=kd,
                min_effort=min_effort,
                max_effort=max_effort,
                min_position=min_pos,
                max_position=max_pos,
            )
            print("✅ External JointPositionController initialized")
        except Exception as e:
            self.use_external_joint_position_controller = False
            print(f"⚠️ Failed to init external JointPositionController, fallback to internal PD: {e}")

    def _initialize_model_references(self):
        self.model_names = MujocoModelNames(self.model)
        
        self.robot_prefix = "robot0:"
        
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
        
        self.ee_site_id = self.model_names.site_name2id[f"{self.robot_prefix}eef_site"]
        self.gripper_site_id = self.model_names.site_name2id[f"{self.robot_prefix}2f85:pinch"]
        
        self.gripper_actuator_id = self.model_names.actuator_name2id[f"{self.robot_prefix}2f85:fingers_actuator"]
        
        self.object_names = ["cube_object", "sphere_object", "cylinder_object"]
        self.object_body_ids = {}
        self.object_geom_ids = {}
        
        for name in self.object_names:
            if name in self.model_names.body_name2id:
                self.object_body_ids[name] = self.model_names.body_name2id[name]
                geom_name = name.replace("_object", "")
                if geom_name in self.model_names.geom_name2id:
                    self.object_geom_ids[name] = self.model_names.geom_name2id[geom_name]
                    
        self.joint_limits = np.array([
            self.model.jnt_range[j] for j in self.arm_joint_ids
        ])
        
        self.workspace_bounds = {
            'x': [0.1, 0.9],
            'y': [-0.5, 0.5],
            'z': [self.table_height - 0.05, self.table_height + 0.8]
        }

    def _setup_enhanced_controllers(self):
        self.position_gains = np.array([100, 100, 100])  
        self.velocity_gains = np.array([10, 10, 10])    
        
        self.joint_position_gains = np.array([50, 50, 30, 20, 20, 20])  
        self.joint_velocity_gains = np.array([5, 5, 3, 2, 2, 2])      

    def _init_camera_simulator(self):
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
        # Robust domain randomizer initialization with error handling
        try:
            self.domain_randomizer = CurriculumDomainRandomizer(
                self.model,
                randomize_joints=True,
                randomize_materials=True,
                randomize_lighting=True,
                randomize_camera=True,
            )
            self.domain_randomizer.set_curriculum_level(self.curriculum_level)
            print(f"✅ Domain randomization initialized successfully (level: {self.curriculum_level})")
            # Apply any pending milestone settings captured before DR was available
            try:
                if hasattr(self, '_pending_milestone_settings') and self._pending_milestone_settings:
                    settings = self._pending_milestone_settings
                    if hasattr(self.domain_randomizer, 'set_milestone_parameters'):
                        self.domain_randomizer.set_milestone_parameters(
                            mass_range=settings.get('mass_range', (50, 50)),
                            color_randomization=settings.get('color_randomization', False),
                            lighting_randomization=settings.get('lighting_randomization', False),
                            friction_randomization=settings.get('friction_randomization', False),
                            objects=settings.get('objects', ['cube_only'])
                        )
                        print(f"📝 Pending milestone settings applied to domain randomizer")
                    # Clear pending after applying
                    self._pending_milestone_settings = None
            except Exception as _:
                # Non-fatal: pending application failed; continue
                pass
        except Exception as e:
            print(f"❌ Domain randomizer initialization failed: {e}")
            print(f"   Will create minimal fallback domain randomizer")
            # Create minimal fallback that won't crash
            try:
                self.domain_randomizer = CurriculumDomainRandomizer(
                    self.model,
                    randomize_joints=False,
                    randomize_materials=False,
                    randomize_lighting=False,
                    randomize_camera=False,
                )
                print(f"✅ Fallback domain randomizer created (disabled features)")
            except Exception as e2:
                print(f"❌ Even fallback domain randomizer failed: {e2}")
                self.domain_randomizer = None

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
        # Initialize target joint positions for delta control
        self.target_joint_positions = self.home_joint_positions.copy()
        print(f" Home position set for optimal camera view")

    def _apply_joint_velocity_control(self, action: np.ndarray):
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
            
            target_vel = np.clip(target_vel, -self.max_joint_velocity, self.max_joint_velocity)
            
            # Simple PD control with gravity compensation
            dof_id = self.model.jnt_dofadr[self.arm_joint_ids[i]]
            gravity_comp = self.data.qfrc_bias[dof_id] * 0.5  # Partial gravity comp
            
            kp = 10.0   
            kd = 1.0    
            
            current_vel = current_velocities[i]
            
            control = kp * target_vel - kd * current_vel + gravity_comp
            control = np.clip(control, -50, 50)  # Limit control forces
            
            self.data.ctrl[self.actuator_ids[i]] = control
        
        gripper_action = action[6] if len(action) > 6 else 0
        if gripper_action > 0.5:
            self.data.ctrl[self.gripper_actuator_id] = 255
        elif gripper_action < -0.5:
            self.data.ctrl[self.gripper_actuator_id] = 0

    def _apply_joint_control(self, action: np.ndarray):
        """Control with planar assist and dynamic smoothing"""
        if len(action) < 6:
            return
        
        if self.target_joint_positions is None:
            qpos_indices = [self.model.jnt_qposadr[j] for j in self.arm_joint_ids]
            self.target_joint_positions = self.data.qpos[qpos_indices].copy()
        
        total_steps = getattr(self, 'total_timesteps_seen', 0)
        
        # Simplified action scaling to reduce delta utilization
        max_delta_base = self.max_joint_delta_normal  # Use consistent scaling
        joint_scale = np.ones(6, dtype=np.float32) * 0.5  # Reduce overall action magnitude
        joint_deltas = action[:6] * max_delta_base * joint_scale
        episodes_so_far = getattr(self, 'reset_episode_count', 0)
        planar_dist = getattr(self, 'prev_obs_planar_dist', None)
        vertical_signed_prev = getattr(self, 'prev_obs_vertical_signed', None)
        # Determine adaptive smoothing alpha tiers (lower alpha = less smoothing)
        if planar_dist is not None:
            if planar_dist > 0.28:
                smoothing_alpha = 0.7
            elif planar_dist > 0.18:
                smoothing_alpha = 0.55
            else:
                smoothing_alpha = 0.40
        else:
            smoothing_alpha = 0.55
        # Disable complex control systems causing instability
        escalation_active = False  # Disable escalation system entirely
        
        # Planar Jacobian assist system removed - was causing high delta_util and instability
        # The robot should learn to approach through RL, not through complex engineered assists
        self._planar_assist_active = False
       
        max_abs = np.max(np.abs(joint_deltas[:6]))
        limit_cap = 0.95 * max_delta_base
        if max_abs > limit_cap:
            joint_deltas[:6] *= (limit_cap / (max_abs + 1e-9))
        # Dynamic low-pass filtering (disable if escalation active)
        if total_steps < 3000 and hasattr(self, 'target_joint_positions'):
            if not hasattr(self, '_prev_target_joints'):
                self._prev_target_joints = self.target_joint_positions.copy()
            if escalation_active:
                alpha = 0.96  # nearly bypass smoothing during escalation for maximum responsiveness
            else:
                # reuse previously computed smoothing_alpha if exists else default
                alpha = locals().get('smoothing_alpha', 0.55)
            proposed = self.target_joint_positions + joint_deltas
            self.target_joint_positions = alpha * proposed + (1 - alpha) * self._prev_target_joints
            joint_deltas = self.target_joint_positions - self._prev_target_joints
            self._prev_target_joints = self.target_joint_positions.copy()
        else:
            self.target_joint_positions += joint_deltas
        for i in range(len(self.arm_joint_ids)):
            low, high = self.joint_limits[i]
            self.target_joint_positions[i] = np.clip(self.target_joint_positions[i], low, high)
        if self.use_external_joint_position_controller and hasattr(self, 'joint_position_controller'):
            self.joint_position_controller.run(self.target_joint_positions, self.data.ctrl)
            for act_id in self.actuator_ids:
                low, high = self.model.actuator_ctrlrange[act_id]
                self.data.ctrl[act_id] = np.clip(self.data.ctrl[act_id], low, high)
        else:
            # Fallback PD control
            qpos_indices = [self.model.jnt_qposadr[j] for j in self.arm_joint_ids]
            dof_indices = [self.model.jnt_dofadr[j] for j in self.arm_joint_ids]
            current_pos = self.data.qpos[qpos_indices]
            current_vel = self.data.qvel[dof_indices]
            pos_err = self.target_joint_positions - current_pos
            torque = self.joint_kp * pos_err - self.joint_kd * current_vel
            torque = np.clip(torque, -80.0, 80.0)
            
            for i, act_id in enumerate(self.actuator_ids[:6]):
                gear = self.model.actuator_gear[act_id][0] if self.model.actuator_gear.shape[0] > act_id else 100.0
                ctrl_val = torque[i] / max(gear, 1e-6)
                low, high = self.model.actuator_ctrlrange[act_id]
                self.data.ctrl[act_id] = np.clip(ctrl_val, low, high)
        
        if len(action) > 6:
            g = action[6]
            if g > 0.2:
                self.data.ctrl[self.gripper_actuator_id] = 255
            elif g < -0.2:
                self.data.ctrl[self.gripper_actuator_id] = 0
        
        # Auto gripper close heuristic (early curriculum)
        if episodes_so_far < 120 and vertical_signed_prev is not None and planar_dist is not None:
            vct = self._estimate_vertical_contact_threshold(episodes_so_far)
            if planar_dist < 0.08 and vertical_signed_prev < vct + 0.05:
                if self.data.ctrl[self.gripper_actuator_id] < 200:
                    self.data.ctrl[self.gripper_actuator_id] = 255
        # Logging every 25 steps early
        if episodes_so_far < 15 and self.step_count % 25 == 0:
            utilization = np.mean(np.abs(joint_deltas[:6])) / max_delta_base
            print(f"[CONTROL] ep={episodes_so_far} step={self.step_count} delta_util={utilization:.2%} esc={escalation_active} planar={(planar_dist if planar_dist is not None else -1):.3f}")

    def _compute_reasonable_reward(self, action: np.ndarray) -> Tuple[float, Dict]:
            """Potential-based shaping to prevent reward hacking.
            Key idea: reward only progress (delta) toward objectives; avoid per-step
            negative baselines (distance/time) that incentivize early termination.
            """
            try:
                reward_components: Dict[str, Any] = {}
                total_reward = 0.0
                # Defaults
                planar_dist = None
                vertical_abs = None
                vertical_signed = None
                current_dist = None

                episodes_so_far_local = getattr(self, 'reset_episode_count', 0)
                # Progress-only shaping (no absolute distance penalty)
                planar_progress_scale = 5.0
                planar_progress_clip = 0.03
                close_zone_attraction_max = 0.15
                close_zone_attraction_scale = 0.04
                # Make time penalty negligible to avoid incentive for early truncation
                time_penalty = -0.0005
                proximity_bonus_threshold = 0.10
                proximity_bonus_rate = 0.012

                # 1) Acquire object / gripper relative metrics (with fallbacks)
                try:
                    if self.current_object and self.current_object in self.object_body_ids:
                        obj_id = self.object_body_ids[self.current_object]
                        obj_pos = self.data.body(obj_id).xpos.copy()
                        gripper_pos = self.data.site_xpos[self.gripper_site_id].copy()
                        diff_vec = gripper_pos - obj_pos
                        current_dist = float(np.linalg.norm(diff_vec))
                        planar_dist = float(np.linalg.norm(diff_vec[:2]))
                        vertical_signed = float(diff_vec[2])
                        vertical_abs = float(abs(vertical_signed))

                        if np.isnan(planar_dist) or np.isnan(vertical_signed):
                            print(f"[REWARD-DEBUG] NaN detected! obj_pos={obj_pos} gripper_pos={gripper_pos} diff={diff_vec}")
                            planar_dist = 1.0 if np.isnan(planar_dist) else planar_dist
                            vertical_signed = 0.5 if np.isnan(vertical_signed) else vertical_signed
                            vertical_abs = abs(vertical_signed)

                        # Expose for control module
                        self.prev_obs_planar_dist = planar_dist
                        self.prev_obs_vertical_signed = vertical_signed
                        self.prev_obs_vertical_diff = vertical_abs
                    else:
                        # No object available - fallback
                        if getattr(self, 'step_count', 0) <= 5:
                            print(f"[REWARD-DEBUG] No current object! current_object={getattr(self, 'current_object', None)}")
                        planar_dist = 1.0
                        vertical_signed = 0.5
                        vertical_abs = 0.5
                        current_dist = planar_dist
                        self.prev_obs_planar_dist = planar_dist
                        self.prev_obs_vertical_signed = vertical_signed
                        self.prev_obs_vertical_diff = vertical_abs
                except Exception as e:
                    print(f"[REWARD-DEBUG] Exception in distance computation: {e}")
                    planar_dist = 1.0
                    vertical_signed = 0.5
                    vertical_abs = 0.5
                    current_dist = planar_dist
                    self.prev_obs_planar_dist = planar_dist
                    self.prev_obs_vertical_signed = vertical_signed
                    self.prev_obs_vertical_diff = vertical_abs

                # 2) Shaping logic 
                if planar_dist is None:
                    # Absolute fallback (should not happen)
                    planar_dist = 1.0
                    vertical_signed = 0.5
                    vertical_abs = 0.5

                # Episode min tracking
                if not hasattr(self, 'episode_min_object_distance'):
                    self.episode_min_object_distance = np.inf
                if current_dist is not None and current_dist < self.episode_min_object_distance:
                    self.episode_min_object_distance = current_dist

                episodes_so_far = getattr(self, 'reset_episode_count', 0)
                # Tightened approach threshold schedule
                if episodes_so_far < 100:
                    base_sched = 0.25 - (episodes_so_far / 100.0) * 0.07
                    if episodes_so_far < 10:
                        approach_threshold = max(0.20, base_sched)  # enforce min 0.20 very early
                    else:
                        approach_threshold = base_sched
                else:
                    approach_threshold = 0.18
                if self.approach_threshold_enter is None:
                    self.approach_threshold_enter = approach_threshold
                    self.approach_threshold_exit = approach_threshold + 0.025
                enter_thresh = self.approach_threshold_enter
                exit_thresh = self.approach_threshold_exit

                # Vertical/contact thresholds 
                vertical_contact_thresh = self._estimate_vertical_contact_threshold(episodes_so_far)
                contact_planar_thresh = 0.18 if episodes_so_far < 60 else 0.14  # relax early contact gating

                # Init safe guards
                if not hasattr(self, '_in_approach_zone'):
                    self._in_approach_zone = False
                if not hasattr(self, '_in_contact_zone'):
                    self._in_contact_zone = False
                if not hasattr(self, 'episode_approach_events'):
                    self.episode_approach_events = 0
                if not hasattr(self, 'episode_contact_events'):
                    self.episode_contact_events = 0

                # NOTE: No absolute distance or far penalty to avoid pushing agent
                # toward early termination. We strictly reward progress deltas below.

                # --- Hysteresis approach zone ---
                if (not self._in_approach_zone) and planar_dist < enter_thresh:
                    self._in_approach_zone = True
                    self.episode_approach_events += 1
                    bonus = 0.2
                    reward_components['approach_entry'] = bonus
                    total_reward += bonus
                elif self._in_approach_zone and planar_dist > exit_thresh:
                    self._in_approach_zone = False

                # --- Contact zone detection ---
                # Calculate contact_zone (needed later for is_in_contact)
                if (not self._contact_breakthrough) and episodes_so_far < 30:
                    relaxed_planar = contact_planar_thresh + 0.02
                    relaxed_vertical = vertical_contact_thresh + 0.05
                    contact_zone = (planar_dist < relaxed_planar) and (vertical_abs < relaxed_vertical)
                else:
                    contact_zone = (planar_dist < contact_planar_thresh) and (vertical_abs < vertical_contact_thresh)
                
                if contact_zone and not self._in_contact_zone:
                    self.episode_contact_events += 1
                    if not self._contact_breakthrough:
                        self._contact_breakthrough = True
                        print(f"🎯 CONTACT BREAKTHROUGH at ep={episodes_so_far}!")
                    reward_components['contact_event'] = 0.5
                    total_reward += 0.5
                self._in_contact_zone = contact_zone

                # --- Planar progress shaping ---
                if not hasattr(self, 'prev_planar_dist') or self.prev_planar_dist is None:
                    self.prev_planar_dist = planar_dist
                planar_progress = self.prev_planar_dist - planar_dist
                if planar_progress > 0:
                    capped = min(planar_progress, planar_progress_clip)
                    # Stronger progress reward when far to counteract drift tendency (piecewise gain)
                    if planar_dist > 0.45:
                        far_gain = 2.0
                    elif planar_dist > 0.30:
                        far_gain = 1.5
                    elif planar_dist > 0.18:
                        far_gain = 1.1
                    else:
                        far_gain = 1.0
                    prog_reward = far_gain * planar_progress_scale * capped
                    reward_components['planar_progress'] = prog_reward
                    total_reward += prog_reward
                elif planar_progress < -0.002 and planar_dist > 0.20:
                    # Gentle penalty when moving away from the object beyond 20cm
                    regress_pen = -min(0.01, 0.4 * abs(planar_progress))
                    reward_components['planar_regress'] = regress_pen
                    total_reward += regress_pen
                self.prev_planar_dist = planar_dist

                # REWARD HACKING SIMPLE FIX: Remove continuous proximity rewards that teach hovering
                # Instead of continuous rewards for being close, only reward progress and contact
                
                # --- Sparse attraction only when making progress ---
                if planar_dist < close_zone_attraction_max and self.prev_planar_dist is not None:
                    if planar_dist < self.prev_planar_dist:  # Only reward when getting closer
                        progress_bonus = 0.01 * (self.prev_planar_dist - planar_dist)  # Scaled by progress made
                        reward_components['progress_attraction'] = progress_bonus
                        total_reward += progress_bonus

                # REMOVED: Sustained proximity bonus that encourages hovering
                # This was the main cause of reward hacking - continuous reward for staying close

                # --- One-off approach achievement bonus ---
                if planar_dist < 0.12 and not hasattr(self, '_approach_achieved'):
                    self._approach_achieved = True
                    bonus = self.reward_config['approach_bonus'] * 0.6
                    reward_components['approach_bonus'] = bonus
                    total_reward += bonus

                # --- Vertical descent shaping ---
                # Relax planar gate early so vertical shaping can start sooner
                descent_planar_gate = 0.34 if episodes_so_far < 50 else 0.22
                if planar_dist < descent_planar_gate and vertical_signed > vertical_contact_thresh:
                    if hasattr(self, 'prev_vertical_signed') and self.prev_vertical_signed is not None:
                        descent_progress = self.prev_vertical_signed - vertical_signed
                        if descent_progress > 0:
                            # Reward descent progress (potential-based)
                            if planar_dist < 0.08:
                                base_mult = 12.0
                            elif planar_dist < 0.12:
                                base_mult = 9.0
                            else:
                                base_mult = 6.0
                            descent_reward = base_mult * min(descent_progress, 0.01)
                            reward_components['vertical_descent'] = descent_reward
                            total_reward += descent_reward
                        else:
                            # Make stall penalty negligible to avoid punitive loops
                            stall_pen = -0.0005
                            reward_components['vertical_stall'] = stall_pen
                            total_reward += stall_pen
                    self.prev_vertical_signed = vertical_signed

                # --- contact probing (encourage final descent when close) ---
                if (not contact_zone) and planar_dist < 0.12 and vertical_signed > vertical_contact_thresh:
                    # Scale reward based on proximity - closer = higher reward
                    proximity_factor = max(0.1, (0.12 - planar_dist) / 0.12)
                    micro_probe = 0.005 * proximity_factor  # Increased from 0.0015
                    reward_components['contact_probe'] = micro_probe
                    total_reward += micro_probe

                # --- Vertical alignment ---
                if planar_dist < 0.12:
                    alignment_factor = max(0.0, (vertical_contact_thresh + 0.05 - vertical_abs) / (vertical_contact_thresh + 0.05))
                    align_reward = 0.25 * alignment_factor
                    if align_reward > 0:
                        reward_components['vertical_alignment'] = align_reward
                        total_reward += align_reward

                # --- Hover penalty ---
                if planar_dist < 0.12 and vertical_abs > vertical_contact_thresh + 0.05:
                    self.hover_counter = getattr(self, 'hover_counter', 0) + 1
                    if self.hover_counter > 8:
                        hover_penalty = -0.002 * (self.hover_counter - 8)
                        hover_penalty = max(hover_penalty, -0.02)
                        reward_components['hover_penalty'] = hover_penalty
                        total_reward += hover_penalty
                else:
                    self.hover_counter = 0

                # --- Contact bonus / grasp / lift / place ---
                # Add proper MuJoCo collision detection alongside distance-based detection
                actual_contact = self._detect_actual_collisions()
                
                # CONTACT LEARNING: Make contact much more rewarding than hovering
                # Use both distance-based contact_zone AND actual collision detection for robustness
                is_in_contact = contact_zone or actual_contact
                
                
                if is_in_contact and not hasattr(self, '_contact_achieved'):
                    self._contact_achieved = True
                    contact_bonus = self.reward_config['contact_bonus'] * 2.0  # Increased from 0.8 to 2.0
                    reward_components['contact_bonus'] = contact_bonus
                    total_reward += contact_bonus
                    detection_method = "physics" if actual_contact else "distance"
                    print(f"🎯 CONTACT ACHIEVED! Bonus: {contact_bonus:.2f} ({detection_method})")
                
                # Additional: Continuous but small contact reward to maintain contact
                elif is_in_contact and hasattr(self, '_contact_achieved'):
                    maintain_contact_bonus = 0.02  # Small continuous reward for maintaining contact
                    reward_components['maintain_contact'] = maintain_contact_bonus
                    total_reward += maintain_contact_bonus
                gripper_closed = self.data.ctrl[self.gripper_actuator_id] > 200
                
                # Add grasp cooldown to prevent multiple rapid grasps
                if not hasattr(self, '_last_grasp_step'):
                    self._last_grasp_step = -1
                    
                grasp_cooldown_steps = 50  # Minimum steps between grasp attempts
                can_grasp = (self.step_count - self._last_grasp_step) > grasp_cooldown_steps
                
                # More generous grasp detection thresholds + cooldown
                if (not self.object_grasped) and gripper_closed and planar_dist < 0.08 and vertical_abs < (vertical_contact_thresh + 0.02) and can_grasp:
                    self.object_grasped = True
                    self._last_grasp_step = self.step_count
                    print(f"🔧 GRASP ACHIEVED! gripper={gripper_closed}, planar={planar_dist:.3f}, vertical={vertical_abs:.3f}")
                    
                    # Increment grasp counter for metrics
                    if not hasattr(self, '_episode_grasp_count'):
                        self._episode_grasp_count = 0
                    self._episode_grasp_count += 1
                    print(f"🔍 ENVIRONMENT GRASP COUNT: Episode grasp count now {self._episode_grasp_count}")
                    
                    if hasattr(self, '_contact_achieved'):
                        grasp_bonus = self.reward_config['grasp_bonus']
                        reward_components['grasp'] = grasp_bonus
                        total_reward += grasp_bonus
                if self.object_grasped and hasattr(self, 'object_initial_pos') and getattr(self, 'object_initial_pos', None) is not None:
                    try:
                        obj_id_tmp = self.object_body_ids.get(self.current_object, None)
                        if obj_id_tmp is not None:
                            obj_pos_tmp = self.data.body(obj_id_tmp).xpos.copy()
                            if obj_pos_tmp[2] > self.object_initial_pos[2] + 0.05 and not hasattr(self, '_lift_achieved'):
                                self._lift_achieved = True
                                lift_bonus = self.reward_config['lift_bonus']
                                reward_components['lift'] = lift_bonus
                                total_reward += lift_bonus
                            if self.target_position is not None and hasattr(self, '_lift_achieved'):
                                dist_to_target_xy = np.linalg.norm(obj_pos_tmp[:2] - self.target_position[:2])
                                placement_progress = max(0.0, 0.25 - dist_to_target_xy) * 3.0
                                reward_components['place_progress'] = placement_progress
                                total_reward += placement_progress
                                if dist_to_target_xy < 0.07 and abs(obj_pos_tmp[2] - self.target_position[2]) < 0.04 and not hasattr(self, '_place_achieved'):
                                    self._place_achieved = True
                                    place_bonus = self.reward_config['place_bonus']
                                    reward_components['place'] = place_bonus
                                    total_reward += place_bonus
                    except Exception as e_lift:
                        print(f"[REWARD-DEBUG] lift/place section error: {e_lift}")

                # --- Stall / escalation scheduling ---
                vct_local = vertical_contact_thresh
                if planar_dist < 0.16 and vertical_signed > vct_local + 0.10:
                    if hasattr(self, '_last_vertical_signed') and self._last_vertical_signed is not None and (self._last_vertical_signed - vertical_signed) < 0.002:
                        self._vertical_stall_steps += 1
                    else:
                        self._vertical_stall_steps = 0
                    if self._vertical_stall_steps == 8 and self.step_count >= getattr(self, '_escalation_active_until_step', 0):
                        self._escalation_active_until_step = self.step_count + 6  # shorter window
                        self._vertical_stall_steps = 0
                        print(f"[VERT-ESC] activating stronger descent (signed={vertical_signed:.3f})")
                else:
                    self._vertical_stall_steps = 0
                
                # Persist _last_vertical_signed for next step's stall detection
                self._last_vertical_signed = vertical_signed

                # 3) Time & smoothness penalties (always applied)
                reward_components['time'] = time_penalty
                total_reward += time_penalty
                action_smooth_penalty = -0.0005 * np.sum(np.square(action[:6]))
                reward_components['smoothness'] = action_smooth_penalty
                total_reward += action_smooth_penalty

                if not self.physics_stable:
                    physics_penalty = self.reward_config['physics_violation_penalty']
                    reward_components['physics'] = physics_penalty
                    total_reward += physics_penalty

                # Fixed clipping range for better value function learning
                total_reward = float(np.clip(total_reward, -8.0, 25.0))
                # Metrics logging
                reward_components['planar_dist'] = planar_dist
                reward_components['vertical_diff'] = vertical_abs
                reward_components['vertical_signed'] = vertical_signed
                return total_reward, reward_components
            except Exception as e_outer:
                print(f"[REWARD-ERROR] Unhandled exception: {e_outer}")
                return 0.0, {'error': 'reward_exception', 'exception': str(e_outer)}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one time step within the environment (DEFENSIVE)."""
        if not hasattr(self, 'step_count'):
            self.step_count = 0
        if self.step_count == 0:
            print(f"[STEP-ENTRY] ep={getattr(self,'reset_episode_count',None)} current_object={getattr(self,'current_object',None)} objs={list(getattr(self,'object_body_ids',{}).keys())}")
        self.step_count += 1
        self.total_timesteps_seen += 1
        if not isinstance(action, np.ndarray):
            action = np.array([action])
        if action.shape == ():
            action = np.array([action])
        self.last_action = action.copy() if action is not None else None
        self._apply_joint_control(action)
        self.do_simulation(self.data.ctrl, self.frame_skip)
        self._update_states()
        obs = self._get_obs()
        reward_out = self._compute_reasonable_reward(action)
        if isinstance(reward_out, tuple) and len(reward_out) == 2:
            reward, reward_info = reward_out
        else:
            print(f"[WARN] reward function returned invalid value type={type(reward_out)}; defaulting to zero")
            reward, reward_info = 0.0, {}
        # Inject fallback planar distance if missing
        if ('planar_dist' not in reward_info or reward_info.get('planar_dist') is None) and hasattr(self,'current_object') and self.current_object in getattr(self,'object_body_ids',{}):
            try:
                obj_id = self.object_body_ids[self.current_object]
                obj_pos = self.data.body(obj_id).xpos.copy()
                grip_pos = self.data.site_xpos[self.gripper_site_id].copy()
                pd = float(np.linalg.norm((grip_pos - obj_pos)[:2]))
                reward_info['planar_dist'] = pd
            except Exception:
                pass
        # Early debug for first few steps
        if self.step_count <= 5:
            print(f"[STEP-DEBUG2] step={self.step_count} keys={list(reward_info.keys())} planar={reward_info.get('planar_dist',None)} vert={reward_info.get('vertical_signed',None)}")
        # Debug proximity bonus
        if self.step_count <= 10 and reward_info.get('planar_dist', 1.0) < 0.15:
            print(f"[PROXIMITY-DEBUG] step={self.step_count} planar={reward_info.get('planar_dist'):.3f} threshold=0.10 bonus={'proximity_bonus' in reward_info}")
        # Check termination conditions
        terminated = self._check_success()
        truncated = self._check_termination() or (self.step_count >= self.max_episode_steps)
        
        # Build info dictionary
        grasp_count = getattr(self, '_episode_grasp_count', 0)
        
        # DEBUG: Log grasp events being reported (only when status changes to avoid spam)
        if grasp_count > 0 and not hasattr(self, '_last_reported_grasp_count'):
            self._last_reported_grasp_count = grasp_count
            print(f"🔍 INFO DICT DEBUG: Started reporting {grasp_count} grasp_events at step {self.step_count}")
        elif grasp_count != getattr(self, '_last_reported_grasp_count', 0):
            old_count = getattr(self, '_last_reported_grasp_count', 0)
            self._last_reported_grasp_count = grasp_count
            if grasp_count > old_count:
                print(f"🔍 INFO DICT DEBUG: Grasp count increased {old_count} → {grasp_count} at step {self.step_count}")
            elif grasp_count == 0:
                print(f"🔍 INFO DICT DEBUG: Stopped reporting grasp_events at step {self.step_count}")
            
        info = {
            'is_success': terminated,
            'episode_length': self.step_count,
            'reward_components': reward_info,
            'planar_distance': reward_info.get('planar_dist', None),
            'approach_events': getattr(self,'episode_approach_events',0),
            'contact_events': getattr(self,'episode_contact_events',0),
            'grasp_events': grasp_count,  # CRITICAL FIX: Add grasp count
            'vertical_signed': reward_info.get('vertical_signed', None),
            'task_completed': bool(terminated),
            'object_grasped': bool(getattr(self, 'object_grasped', False)),
            'physics_stable': bool(getattr(self, 'physics_stable', True)),
        }
        # Store episode data
        if not hasattr(self, 'episode_data'):
            self.episode_data = {'rewards': [], 'success_time': None, 'ee_poses': [], 'object_poses': [], 'camera_sees_object': []}
        self.episode_data['rewards'].append(reward)
        if terminated and self.episode_data['success_time'] is None:
            self.episode_data['success_time'] = self.step_count
        return obs, reward, terminated, truncated, info

    def _estimate_vertical_contact_threshold(self, episodes_so_far: int) -> float:
        """Estimate vertical contact threshold based on training progress (BALANCED)"""
        if episodes_so_far < 30:
            # More reasonable threshold for milestone 0 - still requires some approach
            return 0.35  # 35cm - robot needs to approach somewhat vertically
        elif episodes_so_far < 100:
            # Gradually tighten for better approach learning  
            return 0.35 - ((episodes_so_far - 30) / 70.0) * 0.15  # 0.35m -> 0.20m
        elif episodes_so_far < 200:
            # Contact learning phase
            return 0.20 - ((episodes_so_far - 100) / 100.0) * 0.08  # 0.20m -> 0.12m
        else:
            # Standard threshold for advanced training
            return 0.12

    def _get_ee_z_jacobian(self) -> np.ndarray:
        """Get the Z-component of the end-effector Jacobian"""
        if not hasattr(self, 'gripper_site_id'):
            return None
            
        try:
            # Initialize Jacobian arrays if needed
            if self._jac_tmp_jacp is None:
                self._jac_tmp_jacp = np.zeros((3, self.model.nv))
                self._jac_tmp_jacr = np.zeros((3, self.model.nv))
            
            # Compute Jacobian for gripper site
            mujoco.mj_jacSite(self.model, self.data, self._jac_tmp_jacp, self._jac_tmp_jacr, self.gripper_site_id)
            
            # Return Z-component of position Jacobian for arm joints only
            return self._jac_tmp_jacp[2, :6]  # Z-component, first 6 joints (arm)
        except Exception:
            return None

    def _compute_planar_jacobian_assist(self, planar_dist: float) -> np.ndarray:
        """Compute planar Jacobian assistance for approaching object"""
        if not hasattr(self, 'current_object') or self.current_object is None:
            return None
        
        if self.current_object not in self.object_body_ids:
            return None
        
        try:
            # Get positions
            obj_id = self.object_body_ids[self.current_object]
            obj_pos = self.data.body(obj_id).xpos.copy()
            gripper_pos = self.data.site_xpos[self.gripper_site_id].copy()
            
            # Planar difference (XY only)
            planar_diff = (obj_pos - gripper_pos)[:2]
            planar_norm = np.linalg.norm(planar_diff)
            
            if planar_norm < 1e-6:
                return None
            
            # Desired planar direction (normalized)
            planar_dir = planar_diff / planar_norm
            
            # Initialize Jacobian arrays if needed
            if self._jac_tmp_jacp is None:
                self._jac_tmp_jacp = np.zeros((3, self.model.nv))
                self._jac_tmp_jacr = np.zeros((3, self.model.nv))
            
            # Compute Jacobian
            mujoco.mj_jacSite(self.model, self.data, self._jac_tmp_jacp, self._jac_tmp_jacr, self.gripper_site_id)
            
            # Get planar Jacobian (XY components, arm joints only)
            J_planar = self._jac_tmp_jacp[:2, :6]  # 2x6 matrix
            
            # Desired planar motion magnitude (scale with distance)
            motion_scale = min(0.02, planar_dist * 0.1)
            desired_planar = planar_dir * motion_scale
            
            # Pseudo-inverse solution for joint deltas
            try:
                joint_deltas = np.linalg.pinv(J_planar) @ desired_planar
                return joint_deltas
            except np.linalg.LinAlgError:
                return None
                
        except Exception:
            return None

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
        gripper_closed = self.data.ctrl[self.gripper_actuator_id] > 200;
        
        # release detection
        if self.object_grasped and (not gripper_closed or dist > 0.25):  # Increased from 0.1 to 0.25
            self.object_grasped = False
            print(f"📤 Object released")

    def _check_success(self) -> bool:
        """Curriculum-aware success criteria per phase.
        - milestone_* (approach-learning phase 0–30%): contact OR grasp counts as success
        - grasping: grasp (with contact) required
        - manipulation/mastery: placement criteria
        """
        if not self.current_object or self.target_position is None:
            return False
        
        # FIXED: Cache success result to prevent duplicate counting
        if hasattr(self, '_cached_success_step') and self._cached_success_step == self.step_count:
            return self._cached_success_result

        # Resolve phase for logging: prefer curriculum_manager.current_phase, then env.current_phase
        mgr_phase = getattr(self.curriculum_manager, 'current_phase', None) if hasattr(self, 'curriculum_manager') else None
        env_phase = getattr(self, 'current_phase', None)
        derived_phase = None
        if not (mgr_phase or env_phase):
            # Fallback: derive phase name from current curriculum_level (handles propagation delays)
            level = getattr(self, 'curriculum_level', None)
            if isinstance(level, (int, float)):
                phase_levels = {
                    'milestone_0_percent': 0.05,
                    'milestone_5_percent': 0.16,
                    'milestone_10_percent': 0.18,
                    'milestone_15_percent': 0.20,
                    'milestone_20_percent': 0.22,
                    'milestone_25_percent': 0.30,
                    'milestone_30_percent': 0.35,
                    'grasping': 0.45,
                    'manipulation': 0.60,
                    'mastery': 0.80,
                }
                # Choose the closest target level to current level
                derived_phase = min(phase_levels.keys(), key=lambda k: abs(phase_levels[k] - float(level)))
        current_phase = mgr_phase or env_phase or derived_phase or 'milestone_0_percent'
        
        # DEBUG: Log phase mismatches to identify propagation gaps
        if hasattr(self, '_last_logged_phase') and self._last_logged_phase != current_phase:
            if mgr_phase and env_phase and mgr_phase != env_phase:
                print(f"🔍 PHASE MISMATCH: manager={mgr_phase} env={env_phase} → using {current_phase} for success logic")
            else:
                print(f"🔄 SUCCESS CRITERIA UPDATED: {self._last_logged_phase} → {current_phase}")
        self._last_logged_phase = current_phase
        
        try:
            obj_id = self.object_body_ids[self.current_object] 
            obj_pos = self.data.body(obj_id).xpos.copy()
            
            # Phase-appropriate success criteria using ACTUAL phase names
            if current_phase.startswith('milestone_'):
                contact_achieved = hasattr(self, '_contact_achieved') and self._contact_achieved
                grasp_achieved = bool(self.object_grasped)  # Grasp implies contact
                # Approach-learning phase (milestones 0–30%): contact OR grasp qualifies
                success = contact_achieved or grasp_achieved
                
                # Success check completed (debug logging removed for production)
                
                if success:
                    min_dist_ever = getattr(self, 'episode_min_object_distance', float('inf'))
                    success_type = "GRASP" if grasp_achieved else "CONTACT"
                    print(f"✅ {success_type} SUCCESS: min_dist={min_dist_ever:.3f}, contact={contact_achieved}, grasp={grasp_achieved}, phase={current_phase}")
                # Cache result to prevent duplicate counting
                self._cached_success_result = success
                self._cached_success_step = self.step_count
                return success
                
            elif current_phase == 'grasping':
                # GRASPING PHASE: Success = ACTUAL GRASP + HOLD
                # Must have achieved contact first AND be grasping
                contact_achieved = hasattr(self, '_contact_achieved') and self._contact_achieved
                grasp_achieved = self.object_grasped and contact_achieved
                if grasp_achieved:
                    min_dist_ever = getattr(self, 'episode_min_object_distance', float('inf'))
                    print(f"✅ GRASP SUCCESS: min_dist={min_dist_ever:.3f}, grasp_achieved=True, phase={current_phase}")
                else:
                    # Debug why grasp isn't achieved
                    if self.step_count % 50 == 0:  # Debug every 50 steps
                        min_dist = getattr(self, 'episode_min_object_distance', float('inf'))
                        print(f"🎯 GRASPING PHASE: contact={contact_achieved}, grasp={self.object_grasped}, min_dist={min_dist:.3f}m")
                # Cache result
                self._cached_success_result = grasp_achieved
                self._cached_success_step = self.step_count
                return grasp_achieved
                
            elif current_phase == 'manipulation':
                # Successful grasp + rough placement
                if not self.object_grasped:
                    # Cache failed result
                    self._cached_success_result = False
                    self._cached_success_step = self.step_count
                    return False
                if not self.physics_stable:
                    # Cache failed result
                    self._cached_success_result = False
                    self._cached_success_step = self.step_count
                    return False
                
                # More lenient placement criteria
                distance_to_target = np.linalg.norm(obj_pos - self.target_position)
                height_diff_abs = abs(obj_pos[2] - self.target_position[2])
                
                # Relaxed placement success (vs original 8cm+4cm)
                placement_success = distance_to_target < 0.15 and height_diff_abs <= 0.08
                # FIXED: Cache result to prevent duplicate counting
                self._cached_success_result = placement_success
                self._cached_success_step = self.step_count
                return placement_success
                
            else:  # 'mastery' phase
                # SUCCESS: Original strict criteria for mastery
                if not self.object_grasped or not self.physics_stable:
                    return False
                    
                try:
                    obj_vel = np.linalg.norm(self.data.body(obj_id).cvel[:3])
                    if obj_vel > 0.5:
                        return False
                except:
                    pass
                    
                distance_to_target = np.linalg.norm(obj_pos - self.target_position)
                height_diff_abs = abs(obj_pos[2] - self.target_position[2])
                placement_success = distance_to_target < 0.08 and height_diff_abs <= 0.04
                # Cache result to prevent duplicate counting
                self._cached_success_result = placement_success
                self._cached_success_step = self.step_count
                return placement_success
                
        except Exception as e:
            # DEBUG: Log what exception is being caught
            print(f"❌ SUCCESS CHECK EXCEPTION: {e}")
            import traceback
            traceback.print_exc()
            # FIXED: Cache failed result
            self._cached_success_result = False
            self._cached_success_step = self.step_count
            return False

    def _check_termination(self) -> bool:
        """Check termination conditions (with drift handling)."""
        # Physics breakdown
        if not self.physics_stable and self.consecutive_physics_errors >= self.max_consecutive_errors:
            print("❌ Terminating: Physics instability")
            return True
            
        # Success
        if self._check_success():
            print("✅ Task completed successfully!")
            return True
            
        # Early termination for excessive drift (more conservative; disabled longer)
        if hasattr(self, 'prev_obs_planar_dist') and self.prev_obs_planar_dist is not None:
            ep_idx = getattr(self, 'reset_episode_count', 0)
            if ep_idx >= 600 and self.step_count > 40:
                # Keep drift termination off for much longer; when enabled, be lenient
                if ep_idx < 900:
                    drift_thresh = 0.70
                    steps_required = 140
                elif ep_idx < 1200:
                    drift_thresh = 0.65
                    steps_required = 120
                else:
                    drift_thresh = 0.60
                    steps_required = 100

                # Count only when both far AND trending away on average
                if not hasattr(self, '_drift_counter'):
                    self._drift_counter = 0
                if not hasattr(self, '_drift_regress_counter'):
                    self._drift_regress_counter = 0
                # Track short-window planar progress sign
                if not hasattr(self, '_drift_prev_planar'):
                    self._drift_prev_planar = self.prev_obs_planar_dist
                planar_prog = self._drift_prev_planar - self.prev_obs_planar_dist
                self._drift_prev_planar = self.prev_obs_planar_dist
                moving_away = planar_prog < -0.001  # small regress

                if self.prev_obs_planar_dist > drift_thresh and moving_away:
                    self._drift_counter += 1
                else:
                    # Decay counter slowly to avoid flicker
                    self._drift_counter = max(0, self._drift_counter - 1)

                if self._drift_counter >= steps_required:
                    print(f"❌ Terminating: Excessive drift (planar > {drift_thresh:.2f}m, moving away for {steps_required}+ steps)")
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
                
                # Standard 90° downward tilt for table-top view
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

        except Exception:
            return False

    def reset_model(self) -> np.ndarray:
        """Reset with proper near-spawn and episode counting"""
        # Increment episode counter
        self.reset_episode_count += 1
        
        # Clear achievement tracking
        for attr in ['_approach_achieved', '_contact_achieved', '_lift_achieved', '_place_achieved']:
            if hasattr(self, attr):
                delattr(self, attr)
        
        # Reset episode counters
        self._episode_grasp_count = 0
        self._last_grasp_step = -1
        
        # Reset robot to home
        self.set_state(self.init_qpos, self.init_qvel)
        
        # Apply domain randomization if enabled
        if self.use_domain_randomization and hasattr(self, 'domain_randomizer'):
            self.domain_randomizer.randomize()
        
        for i, joint_id in enumerate(self.arm_joint_ids):
            qpos_id = self.model.jnt_qposadr[joint_id]
            dof_id = self.model.jnt_dofadr[joint_id]
            self.data.qpos[qpos_id] = self.home_joint_positions[i]
            self.data.qvel[dof_id] = 0.0
        
        self.data.ctrl[self.gripper_actuator_id] = 0.0
        
        # Physics settling
        for _ in range(10):
            for i, joint_id in enumerate(self.arm_joint_ids):
                qpos_id = self.model.jnt_qposadr[joint_id]
                dof_id = self.model.jnt_dofadr[joint_id]
                self.data.qpos[qpos_id] = self.home_joint_positions[i]
                self.data.qvel[dof_id] = 0.0
            mujoco.mj_forward(self.model, self.data)
        
        self.target_joint_positions = self.home_joint_positions.copy()
        
        # Reset object with PROPER near-spawn for early curriculum
        self._reset_object_with_curriculum_spawn()
        
        # Set target location
        self._set_target_location()
        
        # Reset tracking with signed vertical
        self._reset_episode_tracking()
        
        # Reset stuck detection
        if self.use_stuck_detection:
            self.initialize_stuck_detection(0)
        
        # Let physics settle with more steps for camera rendering
        for _ in range(15):
            mujoco.mj_step(self.model, self.data)
        
        # Force camera rendering refresh before observation
        if self._camera_sim:
            try:
                self._camera_sim.render()  # Prime the rendering buffers
            except Exception:
                pass
        
        # Additional settling specifically for camera/visibility check
        for _ in range(20):  # Extra settling for robot positioning
            mujoco.mj_step(self.model, self.data)
        
        # Ensure robot is at exact home position before visibility check
        for i, joint_id in enumerate(self.arm_joint_ids):
            qpos_id = self.model.jnt_qposadr[joint_id]
            dof_id = self.model.jnt_dofadr[joint_id]
            self.data.qpos[qpos_id] = self.home_joint_positions[i]
            self.data.qvel[dof_id] = 0.0
        
        # Final forward step to ensure settled state
        mujoco.mj_forward(self.model, self.data)
        
        obs_after_settling = self._get_obs()
        self._log_object_perception_during_reset(obs_after_settling)
        # Restore visibility attempt log (attempt 1 = initial placement)
        try:
            visible_flag = bool(obs_after_settling[-1] > 0.5) if len(obs_after_settling) > 0 else False
            if visible_flag:
                print("✅ Object visible on attempt 1")
            else:
                print("❌ Object not visible on attempt 1")
        except Exception:
            pass
        
        # Debug object state after reset
        print(f"[RESET-DEBUG] current_object={getattr(self, 'current_object', None)}")
        print(f"[RESET-DEBUG] object_body_ids keys={list(self.object_body_ids.keys()) if hasattr(self, 'object_body_ids') else 'NO_ATTR'}")
        if hasattr(self, 'current_object') and self.current_object:
            in_ids = self.current_object in self.object_body_ids if hasattr(self, 'object_body_ids') else False
            print(f"[RESET-DEBUG] current_object in object_body_ids: {in_ids}")
        
        return obs_after_settling

    def _log_object_perception_during_reset(self, obs: np.ndarray):
        """Log object perception details for debugging CNN/visual learning."""
        try:
            episodes_so_far = getattr(self, 'reset_episode_count', 'unknown')
            print(f"🔍 Object Perception Check - Episode {episodes_so_far}")
            print(f"   📦 Spawned Object: {getattr(self, 'current_object', 'unknown')}")
            
            # Extract object properties if available
            if hasattr(self, '_current_object_properties') and self._current_object_properties:
                properties = self._current_object_properties
                if 'size' in properties:
                    print(f"   📏 Object Size: {properties['size']}")
                else:
                    print(f"   📏 Object Properties: {properties}")
            
            # Extract camera portion from observation (last 4*res*res components)
            camera_dim = self.camera_resolution * self.camera_resolution * 4
            if len(obs) >= camera_dim:
                camera_obs = obs[-camera_dim-1:-1]  # exclude visibility flag
                rgb_part = camera_obs[:camera_dim//4*3]  # first 3/4 is RGB
                depth_part = camera_obs[camera_dim//4*3:]  # last 1/4 is depth
                
                rgb_stats = f"mean={np.mean(rgb_part):.3f}, std={np.std(rgb_part):.3f}, range=[{np.min(rgb_part):.3f}, {np.max(rgb_part):.3f}]"
                depth_stats = f"mean={np.mean(depth_part):.3f}, std={np.std(depth_part):.3f}, range=[{np.min(depth_part):.3f}, {np.max(depth_part):.3f}]"
                
                # Basic content detection
                rgb_has_content = np.std(rgb_part) > 0.05  # some visual variation
                depth_has_content = np.std(depth_part) > 0.05  # some depth variation
                
                print(f"   👁️ CNN Visual Input:")
                print(f"      RGB: {rgb_stats}")
                print(f"      Depth: {depth_stats}")
                print(f"      Visual content detected: RGB={rgb_has_content}, Depth={depth_has_content}")
            else:
                print(f"   ⚠️ Camera observation extraction failed: obs_len={len(obs)}, expected_camera_dim={camera_dim}")
                
        except Exception as e:
            print(f"   ❌ Perception logging failed: {e}")
        
    def _reset_object_with_curriculum_spawn(self):
        """FIXED: Proper near-spawn for early curriculum"""
        for obj_name in self.object_names:
            if obj_name in self.object_body_ids:
                obj_id = self.object_body_ids[obj_name]
                self._hide_object(obj_id)
        
        # CRITICAL FIX: Select object based on milestone settings, not curriculum level
        if hasattr(self, 'allowed_objects') and self.allowed_objects:
            # Use milestone-specific object restrictions
            possible_objects = self.allowed_objects
            print(f"🎯 Using milestone object restriction: {possible_objects}")
        else:
            # Fallback to curriculum level logic if no milestone settings
            if self.curriculum_level < 0.3:
                possible_objects = ["cube_object"]
            elif self.curriculum_level < 0.7:
                possible_objects = ["cube_object", "sphere_object"]
            else:
                possible_objects = self.object_names
            print(f"🎯 Using curriculum-level object selection: {possible_objects} (level {self.curriculum_level:.2f})")
        
        available_objects = [obj for obj in possible_objects if obj in self.object_body_ids]
        if not available_objects:
            available_objects = [obj for obj in self.object_names if obj in self.object_body_ids]
        
        self.current_object = random.choice(available_objects)
        obj_id = self.object_body_ids[self.current_object]
        
        # Apply properties
        if self.use_domain_randomization:
            self._randomize_object_properties(obj_id)
        else:
            self._log_default_object_properties(obj_id)
        
        # FIXED: Gradual difficulty progression spawn logic with intermediate zones
        # Avoid dramatic 4-6x difficulty spike from 8-14cm to 35-65cm
        mujoco.mj_forward(self.model, self.data)
        ee_pos = self.data.site_xpos[self.ee_site_id].copy()
        
        if self.curriculum_level < 0.15:
            # NEAR SPAWN: 8-14cm from EE (approach learning)
            radius = np.random.uniform(0.08, 0.14)  # 8-14cm from EE
            angle = np.random.uniform(-np.pi, np.pi)

            x = ee_pos[0] + radius * np.cos(angle)
            y = ee_pos[1] + radius * np.sin(angle)
            z = self.table_height + 0.025

            print(f"🎯 NEAR SPAWN (ep {self.reset_episode_count}): radius={radius:.3f}m from EE")
            
        elif self.curriculum_level < 0.25:
            # INTERMEDIATE SPAWN 1: 15-25cm from EE (gentle transition)
            radius = np.random.uniform(0.15, 0.25)  # 15-25cm from EE
            angle = np.random.uniform(-np.pi, np.pi)

            x = ee_pos[0] + radius * np.cos(angle)
            y = ee_pos[1] + radius * np.sin(angle)
            z = self.table_height + 0.025

            print(f"🎯 INTERMEDIATE SPAWN 1 (ep {self.reset_episode_count}): radius={radius:.3f}m from EE")
            
        elif self.curriculum_level < 0.40:
            # INTERMEDIATE SPAWN 2: Square area ±8cm (moderate challenge)
            x_offset = np.random.uniform(-0.08, 0.08)  # ±8cm from EE
            y_offset = np.random.uniform(-0.08, 0.08)  # ±8cm from EE

            x = ee_pos[0] + x_offset
            y = ee_pos[1] + y_offset
            z = self.table_height + 0.025

            print(f"🎯 INTERMEDIATE SPAWN 2 (ep {self.reset_episode_count}): offset=({x_offset:.3f}, {y_offset:.3f})m from EE")
        else:
            # FULL AREA SPAWN: Standard spawning area (full challenge)
            x = np.random.uniform(*self.object_spawning_area['x_range'])
            y = np.random.uniform(*self.object_spawning_area['y_range'])
            z = self.object_spawning_area['z']
            print(f"📍 FULL AREA SPAWN (ep {self.reset_episode_count})")

        body = self.model.body(obj_id)
        if body.jntadr[0] >= 0:
            qpos_adr = self.model.jnt_qposadr[body.jntadr[0]]
            self.data.qpos[qpos_adr:qpos_adr+3] = [x, y, z]

            if "cylinder" in self.current_object:
                self.data.qpos[qpos_adr+3:qpos_adr+7] = [0.707, 0.707, 0, 0]
            else:
                self.data.qpos[qpos_adr+3:qpos_adr+7] = [1, 0, 0, 0]

            dof_adr = self.model.jnt_dofadr[body.jntadr[0]]
            self.data.qvel[dof_adr:dof_adr+6] = 0

        mujoco.mj_forward(self.model, self.data)
        self._stabilize_object_physics(obj_id)

        for _ in range(10):
            mujoco.mj_step(self.model, self.data)

        self.object_initial_pos = self.data.body(obj_id).xpos.copy()
        actual_pos = self.data.body(obj_id).xpos.copy()

        # Calculate actual planar distance for verification
        ee_pos = self.data.site_xpos[self.ee_site_id].copy()
        actual_planar_dist = np.linalg.norm(actual_pos[:2] - ee_pos[:2])

        print(f"   {self.current_object} at [{actual_pos[0]:.3f},{actual_pos[1]:.3f},{actual_pos[2]:.3f}]")
        print(f"   Planar distance from EE: {actual_planar_dist:.3f}m")

    def _reset_object_in_spawning_area(self):
        """Reset object in spawning area with randomized properties (early curriculum: spawn near EE)"""
        for obj_name in self.object_names:
            if obj_name in self.object_body_ids:
                obj_id = self.object_body_ids[obj_name]
                self._hide_object(obj_id)
        # CRITICAL FIX: Select object based on milestone settings, not curriculum level  
        if hasattr(self, 'allowed_objects') and self.allowed_objects:
            # Use milestone-specific object restrictions
            possible_objects = self.allowed_objects
            print(f"🎯 Using milestone object restriction: {possible_objects}")
        else:
            # Fallback to curriculum level logic if no milestone settings
            if self.curriculum_level < 0.3:
                possible_objects = ["cube_object"]
            elif self.curriculum_level < 0.7:
                possible_objects = ["cube_object", "sphere_object"]
            else:
                possible_objects = self.object_names
            print(f"🎯 Using curriculum-level object selection: {possible_objects} (level {self.curriculum_level:.2f})")
        available_objects = [obj for obj in possible_objects if obj in self.object_body_ids]
        if not available_objects:
            available_objects = [obj for obj in self.object_names if obj in self.object_body_ids]
        self.current_object = random.choice(available_objects)
        obj_id = self.object_body_ids[self.current_object]
        # Apply randomized properties only if domain randomization is enabled
        if self.use_domain_randomization:
            self._randomize_object_properties(obj_id)
        else:
            self._log_default_object_properties(obj_id)
        # Early episodes: spawn near end-effector to guarantee interactions
        near_spawn = hasattr(self, 'reset_episode_count') and self.reset_episode_count < 150
        print(f"DEBUG_NEAR_SPAWN_FLAG={near_spawn} (episode {getattr(self, 'reset_episode_count', 'unknown')})")
        if near_spawn:
            mujoco.mj_forward(self.model, self.data)
            ee_pos = self.data.site_xpos[self.ee_site_id].copy()
            # Much closer spawn for guaranteed interaction
            x = np.random.uniform(ee_pos[0]-0.03, ee_pos[0]+0.03)
            y = np.random.uniform(ee_pos[1]-0.03, ee_pos[1]+0.03)
            z = self.table_height + 0.025
        else:
            # Standard spawning area
            x = np.random.uniform(*self.object_spawning_area['x_range'])
            y = np.random.uniform(*self.object_spawning_area['y_range'])
            z = self.object_spawning_area['z']
            # Clamp to workspace
            if hasattr(self, 'workspace_bounds'):
                margin = 0.02
                x = np.clip(x, self.workspace_bounds['x'][0] + margin, self.workspace_bounds['x'][1] - margin)
                y = np.clip(y, self.workspace_bounds['y'][0] + margin, self.workspace_bounds['y'][1] - margin)
        body = self.model.body(obj_id)
        if body.jntadr[0] >= 0:
            qpos_adr = self.model.jnt_qposadr[body.jntadr[0]]
            self.data.qpos[qpos_adr:qpos_adr+3] = [x, y, z]
            if "cylinder" in self.current_object:
                self.data.qpos[qpos_adr+3:qpos_adr+7] = [0.707, 0.707, 0, 0]
            else:
                self.data.qpos[qpos_adr+3:qpos_adr+7] = [1, 0, 0, 0]
            dof_adr = self.model.jnt_dofadr[body.jntadr[0]]
            self.data.qvel[dof_adr:dof_adr+6] = 0
        mujoco.mj_forward(self.model, self.data)
        self._stabilize_object_physics(obj_id)
        for _ in range(10):
            mujoco.mj_step(self.model, self.data)
        self.object_initial_pos = self.data.body(obj_id).xpos.copy()
        actual_pos = self.data.body(obj_id).xpos.copy()
        tag = "NEAR" if near_spawn else "AREA"
        print(f" {self.current_object} spawned ({tag}) at [{actual_pos[0]:.3f},{actual_pos[1]:.3f},{actual_pos[2]:.3f}]")

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
            
            self._current_object_properties = properties;
            
        except Exception as e:
            print(f" Failed to log default object properties: {e}")
            self._current_object_properties = {"error": str(e)}

    def _randomize_object_properties(self, obj_id: int):
        """Randomize object mass, friction, color for domain randomization."""
        try:
            body = self.model.body(obj_id)
            properties = {}
            # Mass (50g - 500g)
            base_mass = 0.1
            scale = np.random.uniform(0.5, 5.0)
            new_mass = base_mass * scale
            body.mass[0] = new_mass
            properties['mass'] = f"{new_mass*1000:.0f}g"
            # Geom modifications
            for geom_id in range(self.model.ngeom):
                if self.model.geom_bodyid[geom_id] == obj_id:
                    # Friction: slide, spin, roll
                    self.model.geom_friction[geom_id, 0] = np.random.uniform(0.5, 1.5)
                    self.model.geom_friction[geom_id, 1] = np.random.uniform(0.005, 0.02)
                    self.model.geom_friction[geom_id, 2] = np.random.uniform(0.00005, 0.0002)
                    # Color
                    if hasattr(self.model, 'geom_rgba'):
                        rgba = self.model.geom_rgba[geom_id]
                        rgba[:3] = np.random.uniform(0.2, 1.0, size=3)
                        rgba[3] = 1.0
                        self.model.geom_rgba[geom_id] = rgba
            mujoco.mj_forward(self.model, self.data)
            properties['friction'] = 'rnd'
            properties['color'] = 'rnd'
            self._current_object_properties = properties
        except Exception as e:
            print(f" Randomization failed: {e}")

    def _set_target_location(self):
        """Set target location on table with curriculum-aware offsets."""
        try:
            # Simple strategy: place target a moderate lateral offset to require horizontal + vertical motion
            if not hasattr(self, 'object_initial_pos') or self.object_initial_pos is None:
                self.target_position = None
                return
            base = self.object_initial_pos.copy()
            # Offsets scale with curriculum level
            radius_min = 0.12 if self.curriculum_level < 0.5 else 0.18
            radius_max = 0.18 if self.curriculum_level < 0.5 else 0.28
            r = np.random.uniform(radius_min, radius_max)
            angle = np.random.uniform(-np.pi, np.pi)
            dx = r * np.cos(angle)
            dy = r * np.sin(angle)
            tx = np.clip(base[0] + dx, self.workspace_bounds['x'][0]+0.03, self.workspace_bounds['x'][1]-0.03)
            ty = np.clip(base[1] + dy, self.workspace_bounds['y'][0]+0.03, self.workspace_bounds['y'][1]-0.03)
            tz = base[2] + 0.02  # Slightly above table to encourage lift before place
            self.target_position = np.array([tx, ty, tz])
            # Set target tolerance based on phase difficulty
            # Looser in early phases, stricter later
            if self.curriculum_level < 0.25:
                self.target_radius = 0.12
            elif self.curriculum_level < 0.45:
                self.target_radius = 0.10
            elif self.curriculum_level < 0.60:
                self.target_radius = 0.08
            else:
                self.target_radius = 0.06
        except Exception as e:
            print(f"Failed to set target location: {e}")
            self.target_position = None

    def _hide_object(self, obj_id: int):
        """Temporarily hide object by moving it below table & making transparent."""
        try:
            body = self.model.body(obj_id)
            if body.jntadr[0] >= 0:
                qpos_adr = self.model.jnt_qposadr[body.jntadr[0]]
                # Move far below
                self.data.qpos[qpos_adr:qpos_adr+3] = [0, 0, -2.0]
            for geom_id in range(self.model.ngeom):
                if self.model.geom_bodyid[geom_id] == obj_id and hasattr(self.model, 'geom_rgba'):
                    rgba = self.model.geom_rgba[geom_id]
                    rgba[3] = 0.0
                    self.model.geom_rgba[geom_id] = rgba
        except Exception:
            pass

    def _reset_episode_tracking(self):
        """Reset tracking with signed vertical support"""
        self.step_count = 0
        self.episode_min_object_distance = np.inf
        self.episode_approach_events = 0
        self.episode_contact_events = 0
        
        # FIXED: Enable success criteria debugging  
        self._debug_success_log = True
        self.episode_vertical_penalty_events = 0
        self._in_approach_zone = False
        self._in_contact_zone = False
        self.object_grasped = False
        self.prev_vertical_diff = None
        self.prev_vertical_signed = None  # NEW
        self.prev_planar_dist = 1.5  # Fix: Initialize to a large value
        self.hover_counter = 0
        self._planar_assist_active = False  # NEW
        
        self.approach_threshold_enter = None
        self.approach_threshold_exit = None

        self.episode_data = {
            'ee_poses': [],
            'object_poses': [],
            'camera_sees_object': [],
            'rewards': [],
            'success_time': None,
            'grasp_attempts': 0
       
        }
        
        # Keep breakthrough state if achieved
        if not hasattr(self, '_contact_breakthrough'):
            self._contact_breakthrough = False

    def set_collision_rewards(self, collision_rewards: Dict):
        self.collision_rewards = collision_rewards

    def set_collision_termination(self, terminate: bool):
        self.collision_termination = terminate

    def set_curriculum_level(self, level: float):
        """Set curriculum level"""
        self.curriculum_level = np.clip(level, 0.01, 1.0)

        if hasattr(self, 'domain_randomizer'):
            self.domain_randomizer.set_curriculum_level(self.curriculum_level)

    def get_episode_info(self) -> Dict:
        """Get episode information"""
        camera_visibility_rate = (np.mean(self.episode_data['camera_sees_object'])
                                  if self.episode_data['camera_sees_object'] else 0)
        return {
            'total_reward': sum(self.episode_data['rewards']),
            'episode_length': self.step_count,
            'success': self._check_success(),  # FIXED: Curriculum-aware success
            'task_completed': self._check_success(),  # FIXED: For evaluation callback compatibility
            'episode_ended': True,  # FIXED: For curriculum tracking
            'success_time': self.episode_data['success_time'],
            'grasp_attempts': self.episode_data['grasp_attempts'],
            'camera_visibility_rate': camera_visibility_rate,
            'object_in_camera_view_rate': self.episode_data.get('object_in_camera_view_rate', 0.0),
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

    def _get_obs(self):
        """Compose full observation from subcomponents (robot 35 + object 13 + goal 7 + camera + visibility 1)."""
        try:
            robot = self._get_robot_obs()        # 35
            obj = self._get_object_obs()         # 13
            goal = self._get_goal_obs()          # 7
            cam = self._get_camera_obs()         # res*res*4
            visibility = np.array([1.0 if self._check_camera_sees_object() else 0.0], dtype=np.float32)
            obs = np.concatenate([robot, obj, goal, cam, visibility])
            return obs.astype(np.float32)
        except Exception:
            # Fallback zero observation
            total = 35 + 13 + 7 + (self.camera_resolution * self.camera_resolution * 4) + 1
            return np.zeros(total, dtype=np.float32)

    def _get_robot_obs(self) -> np.ndarray:
        """Get robot state observation (35 dimensions)"""
        try:
            qpos_indices = [self.model.jnt_qposadr[j] for j in self.arm_joint_ids]
            joint_pos = self.data.qpos[qpos_indices]
            dof_indices = [self.model.jnt_dofadr[j] for j in self.arm_joint_ids]
            joint_vel = self.data.qvel[dof_indices]
            ee_pos = self.data.site_xpos[self.ee_site_id].copy()
            ee_mat = self.data.site_xmat[self.ee_site_id].reshape(3, 3)
            ee_quat = mat2quat(ee_mat)
            gripper_pos = self.data.ctrl[self.gripper_actuator_id] / 255.0
            gripper_vel = 0.0
            ctrl_forces = self.data.ctrl[self.actuator_ids]
            ee_contact_force = np.zeros(3)
            max_joint_vel = np.max(np.abs(joint_vel))
            workspace_violation = 0.0 if self._check_workspace_bounds(ee_pos) else 1.0
            stability_score = 1.0 if self.physics_stable else 0.0
            stability_metrics = np.array([max_joint_vel, workspace_violation, stability_score])
            grasp_state = 1.0 if self.object_grasped else 0.0
            step_normalized = min(1.0, self.step_count / max(1, self.max_episode_steps))
            operational_state = np.array([grasp_state, step_normalized])
            robot_obs = np.concatenate([
                joint_pos,            # 6
                joint_vel,            # 6 -> 12
                ee_pos, ee_quat,      # 3 + 4 -> 19
                np.array([gripper_pos, gripper_vel]),  # 2 -> 21
                ctrl_forces,          # 6 -> 27
                ee_contact_force,     # 3 -> 30
                stability_metrics,    # 3 -> 33
                operational_state     # 2 -> 35
            ])
            assert robot_obs.shape[0] == 35
            return robot_obs.astype(np.float32)
        except Exception:
            return np.zeros(35, dtype=np.float32)

    def _get_object_obs(self) -> np.ndarray:
        """Get object state observation (13 dimensions)"""
        obs = np.zeros(13, dtype=np.float32)
        try:
            if not self.current_object or self.current_object not in self.object_body_ids:
                return obs
            obj_id = self.object_body_ids[self.current_object]
            obj_pos = self.data.body(obj_id).xpos.copy()
            obj_vel = self.data.cvel[obj_id][:3] if hasattr(self.data, 'cvel') else np.zeros(3)
            gripper_pos = self.data.site_xpos[self.gripper_site_id].copy()
            rel_pos = obj_pos - gripper_pos
            dist_to_target = 0.0
            if self.target_position is not None:
                dist_to_target = np.linalg.norm(obj_pos - self.target_position)
            height_above_table = obj_pos[2] - self.table_height
            grasped = 1.0 if self.object_grasped else 0.0
            object_type_id = float(self.object_names.index(self.current_object)) / max(1, len(self.object_names)-1)
            obs = np.concatenate([
                obj_pos,          # 3
                obj_vel,          # 3 -> 6
                rel_pos,          # 3 -> 9
                np.array([dist_to_target]),      # 1 -> 10
                np.array([height_above_table]),  # 1 -> 11
                np.array([grasped]),             # 1 -> 12
                np.array([object_type_id])       # 1 -> 13
            ])
            return obs.astype(np.float32)
        except Exception:
            return obs

    def _get_goal_obs(self) -> np.ndarray:
        """Goal observation (7 dims)"""
        try:
            if self.target_position is None:
                return np.zeros(7, dtype=np.float32)
            success_flag = 1.0 if self._check_success() else 0.0
            grasp_flag = 1.0 if self.object_grasped else 0.0
            goal_obs = np.concatenate([
                self.target_position,        # 3
                np.array([self.target_radius]),  # 1 -> 4
                np.array([self.curriculum_level]),  # 1 -> 5
                np.array([success_flag]),    # 1 -> 6
                np.array([grasp_flag])       # 1 -> 7
            ])
            return goal_obs.astype(np.float32)
        except Exception:
            return np.zeros(7, dtype=np.float32)

    def _get_camera_obs(self) -> np.ndarray:
        """Get flattened RGBD camera observation (res*res*4)"""
        size = self.camera_resolution * self.camera_resolution * 4
        if self._camera_sim is None:
            return np.zeros(size, dtype=np.float32)
        try:
            rgb, depth = self._camera_sim.render()
            # Normalize RGB to 0-1 and depth as-is (clip reasonable range)
            rgb = (rgb.astype(np.float32) / 255.0)
            depth = depth.astype(np.float32)
            # Clip depth to 0-2m then normalize to 0-1
            depth = np.clip(depth, 0.0, 2.0) / 2.0
            # Ensure correct resolution (camera may render at requested res)
            if rgb.shape[0] != self.camera_resolution:
                # Simple resize via slicing/padding (fallback) to avoid bringing in cv2
                rgb = rgb[:self.camera_resolution, :self.camera_resolution]
                depth = depth[:self.camera_resolution, :self.camera_resolution]
                
            # FIXED: Final validation to prevent black pixels
            rgb_min, rgb_max = np.min(rgb), np.max(rgb)
            if rgb_min == 0.0 and rgb_max < 0.1:  # Mostly black image detected
                print(f"⚠️ Black pixels detected (range: {rgb_min:.3f}-{rgb_max:.3f}), applying gray fallback")
                rgb = np.full_like(rgb, 0.5)  # Mid-gray fallback
                # Add slight noise for variety
                noise = np.random.normal(0, 0.02, rgb.shape)
                rgb = np.clip(rgb + noise, 0.0, 1.0)
                
            stacked = np.dstack([rgb, depth[..., None]])  # (H,W,4)
            return stacked.reshape(-1).astype(np.float32)
        except Exception:
            return np.zeros(size, dtype=np.float32)

    def _check_workspace_bounds(self, position: np.ndarray) -> bool:
        try:
            return (self.workspace_bounds['x'][0] <= position[0] <= self.workspace_bounds['x'][1] and
                    self.workspace_bounds['y'][0] <= position[1] <= self.workspace_bounds['y'][1] and
                    self.workspace_bounds['z'][0] <= position[2] <= self.workspace_bounds['z'][1])
        except Exception:
            return True

    def set_collision_rewards(self, collision_rewards: Dict):
        self.collision_rewards = collision_rewards

    def set_collision_termination(self, terminate: bool):
        self.collision_termination = terminate

    def _detect_actual_collisions(self) -> bool:
        """
        CRITICAL FIX: Detect actual MuJoCo physics collisions between gripper and objects
        This was missing and causing 0% collision detection!
        """
        if not hasattr(self, 'current_object') or not self.current_object:
            return False
            
        # Get object body ID
        if not hasattr(self, 'object_body_ids') or self.current_object not in self.object_body_ids:
            return False
        
        obj_body_id = self.object_body_ids[self.current_object]
        
        # Check all active contacts in the physics simulation
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            
            # Get the body IDs of the two contacting geoms
            geom1_id = contact.geom1
            geom2_id = contact.geom2
            
            # Get body IDs from geom IDs
            body1_id = self.model.geom_bodyid[geom1_id]
            body2_id = self.model.geom_bodyid[geom2_id]
            
            # Check if one body is the object and the other is part of the gripper/arm
            is_object_involved = (body1_id == obj_body_id or body2_id == obj_body_id)
            
            if is_object_involved:
                # Get geom names for debugging
                geom1_name = self.model.geom(geom1_id).name or f"geom_{geom1_id}"
                geom2_name = self.model.geom(geom2_id).name or f"geom_{geom2_id}"
                
                # Check if the other geom is part of the gripper or arm
                gripper_geoms = ['robotiq_85_base_link', 'robotiq_85_left_finger', 'robotiq_85_right_finger',
                               'robotiq_85_left_inner_knuckle', 'robotiq_85_right_inner_knuckle', 
                               'robotiq_85_left_finger_tip', 'robotiq_85_right_finger_tip']
                arm_geoms = ['base_link', 'shoulder_link', 'upper_arm_link', 'forearm_link', 'wrist_1_link', 'wrist_2_link', 'wrist_3_link']
                robot_geoms = gripper_geoms + arm_geoms
                
                # Check if contact is with robot parts
                is_robot_contact = any(robot_part in geom1_name or robot_part in geom2_name for robot_part in robot_geoms)
                
                if is_robot_contact:
                    # This is a valid robot-object contact!
                    return True
                    
        return False

    def close(self):
        """Enhanced cleanup"""
        try:
            if hasattr(self, '_camera_sim') and self._camera_sim is not None:
                if hasattr(self._camera_sim, 'close'):
                    self._camera_sim.close()
        except Exception:
            pass
        super().close()
    
    def set_domain_randomization(self, enable: bool):
        """Enable/disable domain randomization - FIXED: Progressive randomization control"""
        self.use_domain_randomization = enable
        
        if hasattr(self, "domain_randomizer") and self.domain_randomizer is not None:
            if enable:
                print(f"🎲 Domain randomization ENABLED (curriculum level: {self.curriculum_level:.2f})")
            else:
                print(f"🎲 Domain randomization DISABLED - learning basic skills")
        else:
            print(f"⚠️  Domain randomizer not available - using fixed environment parameters")
    
    def set_milestone_settings(self, settings: Dict[str, Any]):
        """Apply progressive curriculum milestone settings"""
        # CRITICAL FIX: Don't override curriculum level - it's already set correctly by curriculum manager
        # The curriculum level should remain as set by the curriculum manager
        
        # CRITICAL FIX: Store milestone settings for object selection logic
        self.milestone_settings = settings
        print(f"📝 Milestone settings stored: {settings}")
        
        # CRITICAL FIX: Apply object restrictions immediately
        objects = settings.get('objects', ['cube_only'])
        if 'cube_only' in objects:
            # Force only cube objects - override any randomization
            self.allowed_objects = ['cube_object']
            print(f"🎯 Object restriction: Only cube_object allowed")
        else:
            # Allow specified objects
            object_mapping = {
                'cube': 'cube_object',
                'sphere': 'sphere_object', 
                'cylinder': 'cylinder_object'
            }
            self.allowed_objects = [object_mapping.get(obj, 'cube_object') for obj in objects]
            print(f"🎯 Object restriction: {self.allowed_objects} allowed")
        
        # Store milestone settings for domain randomizer
        if hasattr(self, 'domain_randomizer'):
            # Check if domain randomizer has the milestone method
            if hasattr(self.domain_randomizer, 'set_milestone_parameters'):
                self.domain_randomizer.set_milestone_parameters(
                    mass_range=settings.get('mass_range', (50, 50)),
                    color_randomization=settings.get('color_randomization', False),
                    lighting_randomization=settings.get('lighting_randomization', False),
                    friction_randomization=settings.get('friction_randomization', False),
                    objects=settings.get('objects', ['cube_only'])
                )
                print(f"📝 Domain randomizer updated with milestone settings")
            else:
                print(f"⚠️ Domain randomizer does not support milestone parameters")
        else:
            print(f"⚠️ Domain randomizer not available, storing settings for later")
            # Store settings to apply when domain randomizer is created
            self._pending_milestone_settings = settings
