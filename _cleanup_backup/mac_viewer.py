#!/usr/bin/env python3
"""
Enhanced UR5e Pick-Place Environment with Proper Mac Support and Full Viewport
Integrates all components for successful sim-to-real transfer
"""

import numpy as np
from typing import Dict, Tuple, Optional, Any, List
import mujoco
import mujoco.viewer
from pathlib import Path
import time
import random
import cv2
import threading
import queue

import gymnasium as gym
from gymnasium import spaces

# Import homestri components
from homestri_ur5e_rl.envs.mujoco.mujoco_env import MujocoEnv
from homestri_ur5e_rl.utils.mujoco_utils import MujocoModelNames, get_site_jac, get_fullM
from homestri_ur5e_rl.utils.controller_utils import pose_error

# Import your custom components
from homestri_ur5e_rl.training.realsense import RealSenseD435iSimulator
from homestri_ur5e_rl.training.ur5e_stuck_detection_mujoco import StuckDetectionMixin

class MacCompatibleViewer:
    """OpenCV-based viewer with interactive camera controls for Mac compatibility"""
    
    def __init__(self, model, data, width=1280, height=960):
        self.model = model
        self.data = data
        self.width = width
        self.height = height
        
        # Camera parameters
        self.cam_distance = 3.0
        self.cam_azimuth = 135.0
        self.cam_elevation = -20.0
        self.cam_lookat = np.array([0.5, 0.0, 0.5])
        
        # Mouse control state
        self.mouse_last_x = None
        self.mouse_last_y = None
        self.mouse_button = None
        
        # Rendering setup
        self.scene = mujoco.MjvScene(model, maxgeom=10000)
        self.context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)
        self.camera = mujoco.MjvCamera()
        self.option = mujoco.MjvOption()
        
        # Set camera mode
        self.camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.camera.fixedcamid = -1
        
        # Update camera
        self._update_camera()
        
        # Viewport
        self.viewport = mujoco.MjrRect(0, 0, width, height)
        
        # Control instructions
        self.show_help = True
        self.help_text = [
            "Camera Controls:",
            "  Left Mouse + Drag: Rotate",
            "  Right Mouse + Drag: Pan", 
            "  Scroll/Middle: Zoom",
            "  R: Reset view",
            "  H: Toggle help",
            "  1-5: Preset views",
            "  Q/ESC: Quit"
        ]
        
        # Preset camera views
        self.preset_views = {
            '1': {'distance': 2.5, 'azimuth': 135, 'elevation': -20, 'lookat': [0.5, 0.0, 0.5]},
            '2': {'distance': 2.0, 'azimuth': 90, 'elevation': -45, 'lookat': [0.5, 0.0, 0.5]},
            '3': {'distance': 1.5, 'azimuth': 180, 'elevation': -10, 'lookat': [0.5, 0.0, 0.7]},
            '4': {'distance': 3.0, 'azimuth': 45, 'elevation': -30, 'lookat': [0.5, 0.0, 0.4]},
            '5': {'distance': 4.0, 'azimuth': 135, 'elevation': -60, 'lookat': [0.5, 0.0, 0.3]},
        }
        
    def _update_camera(self):
        """Update MuJoCo camera from parameters"""
        self.camera.lookat[:] = self.cam_lookat
        self.camera.distance = self.cam_distance
        self.camera.azimuth = self.cam_azimuth
        self.camera.elevation = self.cam_elevation
        
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for camera control"""
        
        if event == cv2.EVENT_LBUTTONDOWN:
            self.mouse_button = 'left'
            self.mouse_last_x = x
            self.mouse_last_y = y
            
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.mouse_button = 'right'
            self.mouse_last_x = x
            self.mouse_last_y = y
            
        elif event == cv2.EVENT_MBUTTONDOWN:
            self.mouse_button = 'middle'
            self.mouse_last_x = x
            self.mouse_last_y = y
            
        elif event == cv2.EVENT_LBUTTONUP or event == cv2.EVENT_RBUTTONUP or event == cv2.EVENT_MBUTTONUP:
            self.mouse_button = None
            
        elif event == cv2.EVENT_MOUSEMOVE and self.mouse_button:
            dx = x - self.mouse_last_x
            dy = y - self.mouse_last_y
            
            if self.mouse_button == 'left':
                # Rotate camera
                self.cam_azimuth -= dx * 0.5
                self.cam_elevation -= dy * 0.5
                self.cam_elevation = np.clip(self.cam_elevation, -89, 89)
                
            elif self.mouse_button == 'right':
                # Pan camera
                # Calculate pan in camera space
                az_rad = np.radians(self.cam_azimuth)
                el_rad = np.radians(self.cam_elevation)
                
                # Camera right vector
                cam_right = np.array([np.cos(az_rad), np.sin(az_rad), 0])
                # Camera up vector (approximation)
                cam_up = np.array([0, 0, 1])
                
                # Pan amount
                pan_speed = 0.001 * self.cam_distance
                pan_delta = -dx * pan_speed * cam_right + dy * pan_speed * cam_up
                self.cam_lookat += pan_delta
                
            elif self.mouse_button == 'middle':
                # Zoom
                zoom_speed = 0.01
                self.cam_distance *= (1.0 + dy * zoom_speed)
                self.cam_distance = np.clip(self.cam_distance, 0.5, 10.0)
                
            self.mouse_last_x = x
            self.mouse_last_y = y
            self._update_camera()
            
        elif event == cv2.EVENT_MOUSEWHEEL:
            # Zoom with scroll
            zoom_speed = 0.1
            if flags > 0:
                self.cam_distance *= (1.0 - zoom_speed)
            else:
                self.cam_distance *= (1.0 + zoom_speed)
            self.cam_distance = np.clip(self.cam_distance, 0.5, 10.0)
            self._update_camera()
            
    def render(self):
        """Render the scene and return image"""
        # Update scene
        mujoco.mjv_updateScene(
            self.model, self.data, self.option, None, self.camera,
            mujoco.mjtCatBit.mjCAT_ALL, self.scene
        )
        
        # Render
        mujoco.mjr_render(self.viewport, self.scene, self.context)
        
        # Read pixels
        rgb_array = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        mujoco.mjr_readPixels(rgb_array, None, self.viewport, self.context)
        
        # Flip vertically (OpenGL convention)
        rgb_array = np.flipud(rgb_array)
        
        # Add overlay text
        if self.show_help:
            y_offset = 30
            for text in self.help_text:
                cv2.putText(rgb_array, text, (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                y_offset += 25
                
        # Add status info
        status_text = f"Cam: Az={self.cam_azimuth:.1f} El={self.cam_elevation:.1f} Dist={self.cam_distance:.2f}"
        cv2.putText(rgb_array, status_text, (10, self.height - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return rgb_array
        
    def handle_keyboard(self, key):
        """Handle keyboard input"""
        if key == ord('r') or key == ord('R'):
            # Reset view
            self.cam_distance = 3.0
            self.cam_azimuth = 135.0
            self.cam_elevation = -20.0
            self.cam_lookat = np.array([0.5, 0.0, 0.5])
            self._update_camera()
            return True
            
        elif key == ord('h') or key == ord('H'):
            # Toggle help
            self.show_help = not self.show_help
            return True
            
        elif chr(key) in self.preset_views:
            # Apply preset view
            preset = self.preset_views[chr(key)]
            self.cam_distance = preset['distance']
            self.cam_azimuth = preset['azimuth'] 
            self.cam_elevation = preset['elevation']
            self.cam_lookat = np.array(preset['lookat'])
            self._update_camera()
            return True
            
        return False

class UR5ePickPlaceEnvEnhanced(MujocoEnv, StuckDetectionMixin):
    """
    Enhanced UR5e environment with Mac-compatible visualization and full viewport
    """
    
    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array"],
        "render_fps": 100,
    }
    
    def __init__(
        self,
        xml_file: str = "custom_scene.xml",
        frame_skip: int = 5,
        camera_resolution: int = 128,
        render_mode: Optional[str] = None,
        reward_type: str = "dense",
        control_mode: str = "joint_velocity",
        use_stuck_detection: bool = True,
        use_domain_randomization: bool = True,
        curriculum_level: float = 0.1,
        default_camera_config: Optional[Dict] = None,
        use_mac_compatible_viewer: bool = True,  # New parameter
        **kwargs
    ):
        # Initialize stuck detection
        StuckDetectionMixin.__init__(self)
        
        # Enhanced environment parameters
        self.camera_resolution = camera_resolution
        self.reward_type = reward_type
        self.control_mode = control_mode
        self.use_stuck_detection = use_stuck_detection
        self.use_domain_randomization = use_domain_randomization
        self.curriculum_level = np.clip(curriculum_level, 0.1, 1.0)
        self.use_mac_compatible_viewer = use_mac_compatible_viewer
        
        # Enhanced anti-reward-hacking configuration
        self.anti_hack_config = {
            'min_grasp_time': 30,
            'max_grasp_attempts': 5,
            'grasp_position_threshold': 0.4,
            'max_lift_height': 0.3,
            'min_lift_height': 0.05,
            'drop_penalty_height': 0.02,
            'max_velocity_reward': 2.0,
            'oscillation_window': 20,
            'oscillation_threshold': 0.005,
            'placement_hold_time': 10,
            'max_episode_reward': 15.0,
            'object_push_threshold': 0.1,
            'energy_penalty_weight': 0.001,
            'time_penalty': 0.002,
            'stuck_penalty': -5.0,
            'joint_limit_penalty': -0.1,
            'smoothness_weight': 0.01,
        }
        
        # Domain randomization parameters
        self.domain_rand_config = {
            'object_mass': [0.05, 0.2],
            'object_friction': [0.5, 1.5],
            'table_friction': [0.8, 1.2],
            'joint_damping': [0.9, 1.1],
            'actuator_gain': [0.95, 1.05],
            'camera_pos_noise': 0.02,
            'camera_rot_noise': 5.0,
            'light_pos_range': 0.3,
            'light_intensity': [0.7, 1.3],
        }
        
        # Task parameters
        self.table_height = 0.42
        self.workspace_bounds = {
            'x': [0.3, 0.7],
            'y': [-0.3, 0.3],
            'z': [self.table_height, self.table_height + 0.4]
        }
        
        # Target positions for curriculum learning
        self.target_positions_easy = [
            np.array([0.5, 0.0, self.table_height + 0.03]),
            np.array([0.55, 0.1, self.table_height + 0.03]),
        ]
        self.target_positions_hard = [
            np.array([0.6, -0.2, self.table_height + 0.03]),
            np.array([0.4, 0.2, self.table_height + 0.03]),
            np.array([0.65, 0.0, self.table_height + 0.03]),
        ]
        
        robot_state_dim = 26  # joints(6) + velocities(6) + ee_pose(7) + gripper(1) + forces(6)
        object_state_dim = 13  # pose(7) + velocities(6)
        goal_dim = 3
        camera_dim = camera_resolution * camera_resolution * 4
        
        obs_dim = robot_state_dim + object_state_dim + goal_dim + camera_dim
        
        observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )
        
        # Better default camera config for full scene view
        if default_camera_config is None:
            default_camera_config = {
                'distance': 3.0,  # Increased distance
                'azimuth': 135.0,
                'elevation': -20.0,
                'lookat': np.array([0.5, 0.0, 0.5]),  # Center on workspace
            }
        
        # Initialize MuJoCo environment
        fullpath = Path(__file__).parent / "assets" / "base_robot" / xml_file
        super().__init__(
            model_path=str(fullpath),
            frame_skip=frame_skip,
            observation_space=observation_space,
            render_mode=render_mode,
            camera_name="realsense_rgb",
            default_camera_config=default_camera_config,
            width=1280,
            height=960,
            **kwargs
        )
        
        # Initialize model references
        self._initialize_model_references()
        
        # Initialize camera simulator
        self._init_camera_simulator()
        
        # Initialize Mac-compatible viewer if needed
        self.mac_viewer = None
        self.viewer_window_name = "UR5e Sim Environment"
        if self.use_mac_compatible_viewer and render_mode == "human":
            self.mac_viewer = MacCompatibleViewer(
                self.model, self.data,
                width=self.width, height=self.height
            )
            cv2.namedWindow(self.viewer_window_name, cv2.WINDOW_NORMAL)
            cv2.setMouseCallback(self.viewer_window_name, self.mac_viewer.mouse_callback)
            
        # Enhanced tracking
        self.episode_data = {
            'rewards': [],
            'actions': [],
            'joint_positions': [],
            'ee_positions': [],
            'object_positions': [],
            'success_steps': 0,
            'grasp_quality': 0.0,
            'energy_used': 0.0,
            'stuck_count': 0,
        }
        
        # Initialize components
        self.current_object = None
        self.object_grasped = False
        self.grasp_start_time = None
        self.step_count = 0
        
        if self.use_stuck_detection:
            self.initialize_stuck_detection(0)
            
    def _initialize_model_references(self):
        """Initialize all model references with enhanced error handling"""
        self.model_names = MujocoModelNames(self.model)
        
        # Robot configuration
        self.robot_prefix = "robot0:"
        
        # Joint names
        self.robot_joint_names = [
            f"{self.robot_prefix}ur5e:shoulder_pan_joint",
            f"{self.robot_prefix}ur5e:shoulder_lift_joint",
            f"{self.robot_prefix}ur5e:elbow_joint",
            f"{self.robot_prefix}ur5e:wrist_1_joint",
            f"{self.robot_prefix}ur5e:wrist_2_joint",
            f"{self.robot_prefix}ur5e:wrist_3_joint",
        ]
        
        # Get joint IDs safely
        self.arm_joint_ids = []
        for name in self.robot_joint_names:
            if name in self.model_names.joint_name2id:
                self.arm_joint_ids.append(self.model_names.joint_name2id[name])
            else:
                raise ValueError(f"Joint {name} not found in model")
                
        # Actuator names
        self.actuator_names = [
            f"{self.robot_prefix}ur5e:shoulder_pan",
            f"{self.robot_prefix}ur5e:shoulder_lift",
            f"{self.robot_prefix}ur5e:elbow",
            f"{self.robot_prefix}ur5e:wrist_1",
            f"{self.robot_prefix}ur5e:wrist_2",
            f"{self.robot_prefix}ur5e:wrist_3",
        ]
        
        self.actuator_ids = []
        for name in self.actuator_names:
            if name in self.model_names.actuator_name2id:
                self.actuator_ids.append(self.model_names.actuator_name2id[name])
                
        # Sites
        self.ee_site_id = self.model_names.site_name2id[f"{self.robot_prefix}eef_site"]
        self.gripper_site_id = self.model_names.site_name2id[f"{self.robot_prefix}2f85:pinch"]
        
        # Force/torque sensors
        if f"{self.robot_prefix}eef_force" in self.model_names.sensor_name2id:
            self.force_sensor_id = self.model_names.sensor_name2id[f"{self.robot_prefix}eef_force"]
            self.torque_sensor_id = self.model_names.sensor_name2id[f"{self.robot_prefix}eef_torque"]
        else:
            self.force_sensor_id = None
            self.torque_sensor_id = None
            
        # Objects
        self.object_names = ["cube_object", "sphere_object", "cylinder_object"]
        self.object_body_ids = {}
        self.object_geom_ids = {}
        
        for name in self.object_names:
            if name in self.model_names.body_name2id:
                self.object_body_ids[name] = self.model_names.body_name2id[name]
                # Get geom for each object
                geom_name = name.replace("_object", "")
                if geom_name in self.model_names.geom_name2id:
                    self.object_geom_ids[name] = self.model_names.geom_name2id[geom_name]
                    
        # Gripper
        self.gripper_actuator_id = self.model_names.actuator_name2id[f"{self.robot_prefix}2f85:fingers_actuator"]
        
        # Joint limits
        self.joint_limits = np.array([
            self.model.jnt_range[j] for j in self.arm_joint_ids
        ])
        
        print(f" Model initialized: {len(self.arm_joint_ids)} joints, {len(self.object_names)} objects")
        
    def _init_camera_simulator(self):
        """Initialize RealSense camera simulator with error handling"""
        try:
            self._camera_sim = RealSenseD435iSimulator(
                self.model,
                self.data,
                camera_name="realsense_rgb",
                render_resolution=self.camera_resolution
            )
            print(f" RealSense camera simulator initialized at {self.camera_resolution}x{self.camera_resolution}")
        except Exception as e:
            print(f" Could not initialize camera simulator: {e}")
            self._camera_sim = None
            
    def reset_model(self) -> np.ndarray:
        """Reset with domain randomization and curriculum learning"""
        # Reset simulation
        mujoco.mj_resetData(self.model, self.data)
        
        # Apply domain randomization
        if self.use_domain_randomization:
            self._apply_domain_randomization()
            
        # Set robot to home position with slight randomization
        home_positions = np.array([0, -np.pi/2, np.pi/2, -np.pi/2, -np.pi/2, 0])
        if self.use_domain_randomization:
            home_positions += np.random.uniform(-0.1, 0.1, size=6) * self.curriculum_level
            
        for i, joint_id in enumerate(self.arm_joint_ids):
            self.data.qpos[joint_id] = home_positions[i]
            self.data.qvel[joint_id] = 0.0
            
        # Set initial control
        if self.control_mode == "joint_position":
            for i, actuator_id in enumerate(self.actuator_ids):
                self.data.ctrl[actuator_id] = home_positions[i]
        else:  # velocity control
            self.data.ctrl[:6] = 0.0
            
        # Open gripper
        self.data.ctrl[self.gripper_actuator_id] = 0
        
        # Reset tracking
        self.step_count = 0
        self.object_grasped = False
        self.grasp_start_time = None
        self.episode_data = {
            'rewards': [],
            'actions': [],
            'joint_positions': [],
            'ee_positions': [],
            'object_positions': [],
            'success_steps': 0,
            'grasp_quality': 0.0,
            'energy_used': 0.0,
            'stuck_count': 0,
        }
        
        # Reset object with curriculum
        self._reset_object_curriculum()
        
        # Reset stuck detection
        if self.use_stuck_detection:
            self.initialize_stuck_detection(0)
            
        # Forward simulation
        mujoco.mj_forward(self.model, self.data)
        
        return self._get_obs()
        
    def _apply_domain_randomization(self):
        """Apply domain randomization for better sim-to-real transfer"""
        # Randomize object properties
        for obj_name, obj_id in self.object_body_ids.items():
            if obj_name in self.object_geom_ids:
                geom_id = self.object_geom_ids[obj_name]
                
                # Mass
                body = self.model.body(obj_id)
                original_mass = body.mass[0]
                body.mass[0] = original_mass * np.random.uniform(*self.domain_rand_config['object_mass'])
                
                # Friction
                geom = self.model.geom(geom_id)
                geom.friction[:] = geom.friction * np.random.uniform(*self.domain_rand_config['object_friction'])
                
        # Randomize joint damping
        for joint_id in self.arm_joint_ids:
            joint = self.model.joint(joint_id)
            joint.damping[0] *= np.random.uniform(*self.domain_rand_config['joint_damping'])
            
        # Randomize actuator gains
        for actuator_id in self.actuator_ids:
            actuator = self.model.actuator(actuator_id)
            actuator.gainprm[0] *= np.random.uniform(*self.domain_rand_config['actuator_gain'])
            
        # Randomize lighting
        if self.model.nlight > 0:
            light = self.model.light(0)
            light.pos[:2] += np.random.uniform(-self.domain_rand_config['light_pos_range'],
                                              self.domain_rand_config['light_pos_range'], size=2)
            light.diffuse[:] *= np.random.uniform(*self.domain_rand_config['light_intensity'])
            
    def _reset_object_curriculum(self):
        """Reset object position based on curriculum level"""
        # Hide all objects
        for obj_name in self.object_names:
            if obj_name in self.object_body_ids:
                obj_id = self.object_body_ids[obj_name]
                body = self.model.body(obj_id)
                if body.jntadr[0] >= 0:
                    qpos_adr = self.model.jnt_qposadr[body.jntadr[0]]
                    self.data.qpos[qpos_adr:qpos_adr+3] = [10, 10, -10]
                    
        # Select object based on curriculum (harder objects at higher levels)
        if self.curriculum_level < 0.3:
            # Easy: only cube
            possible_objects = ["cube_object"]
        elif self.curriculum_level < 0.7:
            # Medium: cube or cylinder
            possible_objects = ["cube_object", "cylinder_object"]
        else:
            # Hard: all objects
            possible_objects = self.object_names
            
        self.current_object = random.choice(possible_objects)
        obj_id = self.object_body_ids[self.current_object]
        
        # Position based on curriculum
        if self.curriculum_level < 0.5:
            # Easy: close to robot
            x = np.random.uniform(0.45, 0.55)
            y = np.random.uniform(-0.1, 0.1)
        else:
            # Hard: farther positions
            x = np.random.uniform(0.4, 0.6)
            y = np.random.uniform(-0.2, 0.2)
            
        z = self.table_height + 0.05
        
        # Set position
        body = self.model.body(obj_id)
        if body.jntadr[0] >= 0:
            qpos_adr = self.model.jnt_qposadr[body.jntadr[0]]
            self.data.qpos[qpos_adr:qpos_adr+3] = [x, y, z]
            self.data.qpos[qpos_adr+3:qpos_adr+7] = [1, 0, 0, 0]
            
            # Zero velocities
            if body.dofadr[0] >= 0:
                self.data.qvel[body.dofadr[0]:body.dofadr[0]+6] = 0
                
        self.object_initial_pos = np.array([x, y, z])
        
        # Select target position based on curriculum
        if self.curriculum_level < 0.5:
            self.target_position = random.choice(self.target_positions_easy)
        else:
            all_targets = self.target_positions_easy + self.target_positions_hard
            self.target_position = random.choice(all_targets)
            
        self.target_radius = 0.05 * (2.0 - self.curriculum_level)  # Smaller target at higher levels
        
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one timestep with enhanced safety and tracking"""
        self.step_count += 1
        
        # Store action for analysis
        self.episode_data['actions'].append(action.copy())
        
        # Clip and scale actions
        action = np.clip(action, -1.0, 1.0)
        
        # Apply action based on control mode
        if self.control_mode == "joint_velocity":
            # Velocity control (better for sim-to-real)
            max_vel = 1.0  # rad/s
            for i in range(6):
                if i < len(self.actuator_ids):
                    self.data.ctrl[self.actuator_ids[i]] = action[i] * max_vel
                    
        elif self.control_mode == "joint_position":
            # Position control with safety limits
            for i in range(6):
                if i < len(self.arm_joint_ids):
                    joint_id = self.arm_joint_ids[i]
                    current_pos = self.data.qpos[joint_id]
                    
                    # Compute target with safety margin
                    joint_range = self.joint_limits[i]
                    mid = (joint_range[0] + joint_range[1]) / 2
                    half_range = (joint_range[1] - joint_range[0]) / 2
                    target_pos = mid + action[i] * half_range * 0.8  # 80% of range
                    
                    # Smooth transition
                    alpha = 0.1  # smoothing factor
                    target_pos = (1 - alpha) * current_pos + alpha * target_pos
                    
                    self.data.ctrl[self.actuator_ids[i]] = target_pos
                    
        # Gripper control with hysteresis
        gripper_action = action[6] if len(action) > 6 else 0
        current_gripper = self.data.ctrl[self.gripper_actuator_id]
        
        if gripper_action > 0.5 and current_gripper < 200:
            self.data.ctrl[self.gripper_actuator_id] = 255
        elif gripper_action < -0.5 and current_gripper > 50:
            self.data.ctrl[self.gripper_actuator_id] = 0
            
        # Step simulation with error recovery
        try:
            self.do_simulation(self.data.ctrl, self.frame_skip)
        except Exception as e:
            print(f"Simulation error: {e}")
            obs = self._get_obs()
            return obs, -10.0, True, False, {"error": str(e)}
            
        # Update states
        self._update_grasp_state()
        self._update_tracking()
        
        # Get observation
        obs = self._get_obs()
        
        # Compute reward
        reward, reward_info = self._compute_enhanced_reward()
        self.episode_data['rewards'].append(reward)
        
        # Check termination
        terminated = self._check_termination()
        truncated = self.step_count >= 1000
        
        # Compile info
        info = {
            "step": self.step_count,
            "object_grasped": self.object_grasped,
            "task_completed": self._check_success(),
            "reward_info": reward_info,
            "curriculum_level": self.curriculum_level,
            "episode_data": self.episode_data.copy() if terminated or truncated else {},
        }
        
        # Check stuck
        if self.use_stuck_detection:
            is_stuck, stuck_reason = self.check_robot_stuck(0)
            if is_stuck:
                info["stuck"] = True
                info["stuck_reason"] = stuck_reason
                terminated = True
                self.episode_data['stuck_count'] += 1
                reward += self.anti_hack_config['stuck_penalty']
                
        return obs, reward, terminated, truncated, info
        
    def render(self):
        """Enhanced render with Mac-compatible viewer"""
        if self.render_mode == "human" and self.use_mac_compatible_viewer and self.mac_viewer:
            # Use Mac-compatible viewer
            frame = self.mac_viewer.render()
            
            # Convert RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # Add environment status
            status_y = 30
            status_texts = [
                f"Step: {self.step_count}",
                f"Object: {self.current_object or 'None'}",
                f"Grasped: {self.object_grasped}",
                f"Curriculum: {self.curriculum_level:.2f}",
            ]
            
            for text in status_texts:
                cv2.putText(frame_bgr, text, (self.width - 300, status_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                status_y += 25
                
            # Display
            cv2.imshow(self.viewer_window_name, frame_bgr)
            
            # Handle keyboard
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # q or ESC
                cv2.destroyAllWindows()
                return None
            else:
                self.mac_viewer.handle_keyboard(key)
                
            return frame
            
        else:
            # Use default renderer
            return super().render()
            
    def close(self):
        """Clean up resources"""
        if self.mac_viewer and cv2.getWindowProperty(self.viewer_window_name, cv2.WND_PROP_VISIBLE) >= 0:
            cv2.destroyAllWindows()
        super().close()
        
    # Include all other methods from your original implementation
    # (_get_obs, _get_camera_data_safe, _update_grasp_state, _update_tracking,
    #  _compute_enhanced_reward, _check_termination, _check_success, _mat2quat)
    # [Methods remain the same as in your original implementation]
    
    def _get_obs(self) -> np.ndarray:
        """Get enhanced observation including force/torque feedback"""
        obs_dict = {}
        
        # Joint states
        joint_pos = np.array([self.data.qpos[j] for j in self.arm_joint_ids])
        joint_vel = np.array([self.data.qvel[j] for j in self.arm_joint_ids])
        obs_dict["joint_pos"] = joint_pos
        obs_dict["joint_vel"] = joint_vel
        
        # End effector pose
        ee_pos = self.data.site_xpos[self.ee_site_id].copy()
        ee_mat = self.data.site_xmat[self.ee_site_id].reshape(3, 3)
        ee_quat = self._mat2quat(ee_mat)
        obs_dict["ee_pose"] = np.concatenate([ee_pos, ee_quat])
        
        # Gripper state
        gripper_ctrl = self.data.ctrl[self.gripper_actuator_id] / 255.0
        obs_dict["gripper"] = np.array([gripper_ctrl])
        
        # Force/torque sensing
        if self.force_sensor_id is not None:
            force = self.data.sensordata[self.force_sensor_id:self.force_sensor_id+3]
            torque = self.data.sensordata[self.torque_sensor_id:self.torque_sensor_id+3]
            obs_dict["force_torque"] = np.concatenate([force, torque])
        else:
            obs_dict["force_torque"] = np.zeros(6)
            
        # Object state with velocities
        if self.current_object and self.current_object in self.object_body_ids:
            obj_id = self.object_body_ids[self.current_object]
            body = self.model.body(obj_id)
            
            obj_pos = self.data.body(obj_id).xpos.copy()
            obj_quat = self.data.body(obj_id).xquat.copy()
            
            # Get velocities
            if body.dofadr[0] >= 0:
                obj_vel = self.data.qvel[body.dofadr[0]:body.dofadr[0]+3].copy()
                obj_angvel = self.data.qvel[body.dofadr[0]+3:body.dofadr[0]+6].copy()
            else:
                obj_vel = np.zeros(3)
                obj_angvel = np.zeros(3)
                
            obs_dict["object_state"] = np.concatenate([obj_pos, obj_quat, obj_vel, obj_angvel])
        else:
            obs_dict["object_state"] = np.zeros(13)
            
        # Goal
        obs_dict["goal"] = self.target_position
        
        # Camera data
        camera_data = self._get_camera_data_safe()
        obs_dict["camera"] = camera_data
        
        # Concatenate all observations
        obs = np.concatenate([
            obs_dict["joint_pos"],
            obs_dict["joint_vel"],
            obs_dict["ee_pose"],
            obs_dict["gripper"],
            obs_dict["force_torque"],
            obs_dict["object_state"],
            obs_dict["goal"],
            obs_dict["camera"],
        ])
        
        return obs.astype(np.float32)
        
    def _get_camera_data_safe(self) -> np.ndarray:
        """Get camera data with fallback"""
        try:
            if self._camera_sim is not None:
                return self._camera_sim.render_rgbd()
            else:
                return np.zeros(self.camera_resolution * self.camera_resolution * 4)
        except Exception as e:
            print(f"Camera error: {e}")
            return np.zeros(self.camera_resolution * self.camera_resolution * 4)
            
    def _update_grasp_state(self):
        """Update grasp state with quality metrics"""
        if not self.current_object:
            return
            
        # Get positions
        gripper_pos = self.data.site_xpos[self.gripper_site_id]
        obj_id = self.object_body_ids[self.current_object]
        obj_pos = self.data.body(obj_id).xpos
        
        # Distance check
        dist = np.linalg.norm(gripper_pos - obj_pos)
        
        # Gripper state
        gripper_closed = self.data.ctrl[self.gripper_actuator_id] > 200
        
        # Grasp detection with quality
        if not self.object_grasped and gripper_closed and dist < 0.05:
            # Additional checks for stable grasp
            obj_lifted = obj_pos[2] > self.table_height + 0.03
            
            # Check force feedback for grasp quality
            if self.force_sensor_id is not None:
                force_magnitude = np.linalg.norm(
                    self.data.sensordata[self.force_sensor_id:self.force_sensor_id+3]
                )
                grasp_quality = np.clip(force_magnitude / 10.0, 0, 1)
            else:
                grasp_quality = 0.5 if obj_lifted else 0.2
                
            if obj_lifted or grasp_quality > 0.3:
                self.object_grasped = True
                self.grasp_start_time = self.step_count
                self.episode_data['grasp_quality'] = grasp_quality
                
        elif self.object_grasped and (not gripper_closed or dist > 0.1):
            self.object_grasped = False
            self.grasp_start_time = None
            
    def _update_tracking(self):
        """Update tracking data for analysis"""
        # Joint positions
        joint_pos = [self.data.qpos[j] for j in self.arm_joint_ids]
        self.episode_data['joint_positions'].append(joint_pos)
        
        # End effector position
        ee_pos = self.data.site_xpos[self.ee_site_id].copy()
        self.episode_data['ee_positions'].append(ee_pos)
        
        # Object position
        if self.current_object:
            obj_id = self.object_body_ids[self.current_object]
            obj_pos = self.data.body(obj_id).xpos.copy()
            self.episode_data['object_positions'].append(obj_pos)
            
        # Energy usage
        power = np.sum(np.abs(self.data.ctrl[:6] * self.data.qvel[self.arm_joint_ids]))
        self.episode_data['energy_used'] += power * self.model.opt.timestep * self.frame_skip
        
    def _compute_enhanced_reward(self) -> Tuple[float, Dict]:
        """Compute reward with all enhancements for sim-to-real"""
        reward_components = {}
        
        # Base time penalty
        reward_components['time'] = -self.anti_hack_config['time_penalty']
        
        # Get current states
        ee_pos = self.data.site_xpos[self.ee_site_id]
        gripper_pos = self.data.site_xpos[self.gripper_site_id]
        
        if not self.current_object:
            return sum(reward_components.values()), reward_components
            
        obj_id = self.object_body_ids[self.current_object]
        obj_pos = self.data.body(obj_id).xpos.copy()
        
        # Phase 1: Reaching
        if not self.object_grasped:
            # Distance to object
            dist_to_obj = np.linalg.norm(gripper_pos - obj_pos)
            reward_components['reaching'] = np.exp(-5 * dist_to_obj)
            
            # Alignment bonus
            obj_above = obj_pos.copy()
            obj_above[2] += 0.1
            alignment = 1.0 - np.linalg.norm(gripper_pos - obj_above) / 0.2
            reward_components['alignment'] = max(0, alignment) * 0.2
            
            # Penalty for knocking
            obj_displacement = np.linalg.norm(obj_pos[:2] - self.object_initial_pos[:2])
            if obj_displacement > self.anti_hack_config['object_push_threshold']:
                reward_components['knock_penalty'] = -obj_displacement
                
        # Phase 2: Grasping/Lifting
        else:
            # Lift reward
            lift_height = obj_pos[2] - self.table_height
            if lift_height > self.anti_hack_config['min_lift_height']:
                reward_components['lifting'] = 0.5 + 0.5 * np.tanh(lift_height * 10)
                
            # Transport reward
            dist_to_target = np.linalg.norm(obj_pos - self.target_position)
            reward_components['transport'] = np.exp(-3 * dist_to_target)
            
            # Placement reward
            if dist_to_target < self.target_radius:
                self.episode_data['success_steps'] += 1
                reward_components['placement'] = 1.0
                
                # Success bonus after holding
                if self.episode_data['success_steps'] >= self.anti_hack_config['placement_hold_time']:
                    reward_components['success'] = 10.0
                    
            # Grasp quality bonus
            if self.episode_data['grasp_quality'] > 0.5:
                reward_components['grasp_quality'] = 0.2
                
        # Safety penalties
        # Joint limits
        for i, joint_id in enumerate(self.arm_joint_ids):
            pos = self.data.qpos[joint_id]
            limits = self.joint_limits[i]
            margin = 0.1
            if pos < limits[0] + margin or pos > limits[1] - margin:
                reward_components['joint_limits'] = self.anti_hack_config['joint_limit_penalty']
                
        # Smoothness reward (penalize jerky motion)
        if len(self.episode_data['actions']) > 1:
            action_diff = np.linalg.norm(
                self.episode_data['actions'][-1] - self.episode_data['actions'][-2]
            )
            reward_components['smoothness'] = -self.anti_hack_config['smoothness_weight'] * action_diff
            
        # Energy penalty
        reward_components['energy'] = -self.anti_hack_config['energy_penalty_weight'] * \
                                     self.episode_data['energy_used'] / max(1, self.step_count)
                                     
        # Compute total
        total_reward = sum(reward_components.values())
        
        # Curriculum scaling
        total_reward *= (0.5 + 0.5 * self.curriculum_level)
        
        return total_reward, reward_components
        
    def _check_termination(self) -> bool:
        """Check termination conditions"""
        # Success
        if self._check_success():
            return True
            
        # Object fell
        if self.current_object:
            obj_id = self.object_body_ids[self.current_object]
            obj_pos = self.data.body(obj_id).xpos
            if obj_pos[2] < self.table_height - 0.1:
                return True
                
        # Workspace violation
        ee_pos = self.data.site_xpos[self.ee_site_id]
        if not (self.workspace_bounds['x'][0] <= ee_pos[0] <= self.workspace_bounds['x'][1] and
                self.workspace_bounds['y'][0] <= ee_pos[1] <= self.workspace_bounds['y'][1] and
                self.workspace_bounds['z'][0] <= ee_pos[2] <= self.workspace_bounds['z'][1]):
            return True
            
        return False
        
    def _check_success(self) -> bool:
        """Check if task is successfully completed"""
        return self.episode_data['success_steps'] >= self.anti_hack_config['placement_hold_time']
        
    def _mat2quat(self, mat):
        """Convert rotation matrix to quaternion"""
        # Proper conversion
        trace = mat[0, 0] + mat[1, 1] + mat[2, 2]
        
        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (mat[2, 1] - mat[1, 2]) * s
            y = (mat[0, 2] - mat[2, 0]) * s
            z = (mat[1, 0] - mat[0, 1]) * s
        elif mat[0, 0] > mat[1, 1] and mat[0, 0] > mat[2, 2]:
            s = 2.0 * np.sqrt(1.0 + mat[0, 0] - mat[1, 1] - mat[2, 2])
            w = (mat[2, 1] - mat[1, 2]) / s
            x = 0.25 * s
            y = (mat[0, 1] + mat[1, 0]) / s
            z = (mat[0, 2] + mat[2, 0]) / s
        elif mat[1, 1] > mat[2, 2]:
            s = 2.0 * np.sqrt(1.0 + mat[1, 1] - mat[0, 0] - mat[2, 2])
            w = (mat[0, 2] - mat[2, 0]) / s
            x = (mat[0, 1] + mat[1, 0]) / s
            y = 0.25 * s
            z = (mat[1, 2] + mat[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + mat[2, 2] - mat[0, 0] - mat[1, 1])
            w = (mat[1, 0] - mat[0, 1]) / s
            x = (mat[0, 2] + mat[2, 0]) / s
            y = (mat[1, 2] + mat[2, 1]) / s
            z = 0.25 * s
            
        return np.array([w, x, y, z])
    
    @property
    def dt(self):
        """Simulation timestep"""
        return self.model.opt.timestep * self.frame_skip
        
    def set_curriculum_level(self, level: float):
        """Update curriculum difficulty"""
        self.curriculum_level = np.clip(level, 0.1, 1.0)

# Test the enhanced environment
if __name__ == "__main__":
    print("🧪 Testing Enhanced UR5e Pick-Place Environment with Mac Support...")
    print("📌 Camera Controls:")
    print("   - Left Mouse: Rotate")
    print("   - Right Mouse: Pan")
    print("   - Scroll/Middle: Zoom")
    print("   - R: Reset view")
    print("   - H: Toggle help")
    print("   - 1-5: Preset views")
    print("   - Q/ESC: Exit\n")
    
    try:
        env = UR5ePickPlaceEnvEnhanced(
            xml_file="custom_scene.xml",
            render_mode="human",
            camera_resolution=128,
            control_mode="joint_velocity",
            use_stuck_detection=True,
            use_domain_randomization=True,
            curriculum_level=0.3,
            use_mac_compatible_viewer=True,  # Enable Mac support
        )
        
        print(f" Environment created")
        print(f"   Observation space: {env.observation_space.shape}")
        print(f"   Action space: {env.action_space.shape}")
        
        # Test reset
        obs, info = env.reset()
        print(f" Reset successful")
        print(f"   Current object: {env.current_object}")
        print(f"   Target position: {env.target_position}")
        
        # Run test episode with simple policy
        total_reward = 0
        for i in range(500):
            if i < 100:
                # Move towards object
                action = np.zeros(7)
                action[1] = -0.3  # shoulder down
                action[2] = 0.3   # elbow
            elif i < 150:
                # Close gripper
                action = np.zeros(7)
                action[6] = 1.0
            elif i < 300:
                # Lift and move
                action = np.zeros(7)
                action[1] = 0.2   # shoulder up
                action[0] = 0.3   # base rotation
            else:
                # Place
                action = np.zeros(7)
                action[1] = -0.2  # lower
                
            # Add some noise
            action[:6] += np.random.normal(0, 0.05, 6)
            
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            if i % 50 == 0:
                print(f"Step {i}: reward={reward:.3f}, total={total_reward:.3f}, "
                      f"grasped={info['object_grasped']}, level={info['curriculum_level']:.2f}")
                
            env.render()
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
            if terminated or truncated:
                print(f"\nEpisode ended: success={info.get('task_completed', False)}")
                print(f"Total reward: {total_reward:.3f}")
                
                # Show episode data
                if 'episode_data' in info and info['episode_data']:
                    data = info['episode_data']
                    print(f"Episode stats:")
                    print(f"  Energy used: {data['energy_used']:.3f}")
                    print(f"  Grasp quality: {data['grasp_quality']:.3f}")
                    print(f"  Success steps: {data['success_steps']}")
                    print(f"  Stuck count: {data['stuck_count']}")
                    
                # Reset for next episode
                obs, info = env.reset()
                total_reward = 0
                
                # Increase curriculum level
                new_level = min(1.0, env.curriculum_level + 0.1)
                env.set_curriculum_level(new_level)
                print(f"\nStarting new episode with curriculum level: {new_level:.2f}")
                
        env.close()
        print("\n Test completed successfully!")
        
    except Exception as e:
        print(f" Test failed: {e}")
        import traceback
        traceback.print_exc()