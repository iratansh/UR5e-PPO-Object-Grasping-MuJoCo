from os import path
import sys

import numpy as np
from homestri_ur5e_rl.controllers.operational_space_controller import ImpedanceController, ComplianceController, OperationalSpaceController, TargetType
from homestri_ur5e_rl.controllers.joint_effort_controller import JointEffortController
from homestri_ur5e_rl.controllers.joint_velocity_controller import JointVelocityController
from homestri_ur5e_rl.controllers.joint_position_controller import JointPositionController
from homestri_ur5e_rl.controllers.force_torque_sensor_controller import ForceTorqueSensorController
from gymnasium import spaces
from homestri_ur5e_rl.envs.mujoco.mujoco_env import MujocoEnv
from gymnasium_robotics.utils.mujoco_utils import MujocoModelNames, robot_get_obs

np.set_printoptions(formatter={'float': lambda x: "{0:0.5f}".format(x)})

DEFAULT_CAMERA_CONFIG = {
    "distance": 2.2,
    "azimuth": 0.0,
    "elevation": -20.0,
    "lookat": np.array([0, 0, 1]),
}

class CustomRobot(MujocoEnv):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 12,
    }

    def __init__(
        self,
        model_path="../assets/base_robot/custom_scene.xml",
        frame_skip=40,
        default_camera_config: dict = DEFAULT_CAMERA_CONFIG,
        **kwargs,
    ):
        # Use absolute path resolution to make it more robust
        xml_file_path = path.join(
            path.dirname(path.realpath(__file__)),
            model_path,
        )

        observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(28,), dtype=np.float64)

        super().__init__(
            xml_file_path,
            frame_skip,
            observation_space,
            default_camera_config=default_camera_config,
            **kwargs,
        )

        self.init_qvel = self.data.qvel.copy()
        self.init_ctrl = self.data.ctrl.copy()

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float64)
        self.model_names = MujocoModelNames(self.model) 

        self.controller = OperationalSpaceController(
            model=self.model, 
            data=self.data, 
            model_names=self.model_names,
            eef_name='robot0:eef_site', 
            joint_names=[
                'robot0:ur5e:shoulder_pan_joint',
                'robot0:ur5e:shoulder_lift_joint',
                'robot0:ur5e:elbow_joint',
                'robot0:ur5e:wrist_1_joint',
                'robot0:ur5e:wrist_2_joint',
                'robot0:ur5e:wrist_3_joint',
            ],
            actuator_names=[
                'robot0:ur5e:shoulder_pan',
                'robot0:ur5e:shoulder_lift',
                'robot0:ur5e:elbow',
                'robot0:ur5e:wrist_1',
                'robot0:ur5e:wrist_2',
                'robot0:ur5e:wrist_3',
            ],
            min_effort=[-150, -150, -150, -150, -150, -150],
            max_effort=[150, 150, 150, 150, 150, 150],
            target_type=TargetType.TWIST,
            kp=200.0,
            ko=200.0,
            kv=50.0,
            vmax_xyz=0.2,
            vmax_abg=1,
            null_damp_kv=10,
        )

        # Initialize additional controllers later if needed
        self.force_torque_sensor = None
        self.gripper_controller = None
        
        # Set initial robot configuration
        self.init_qpos_config = {
            "robot0:ur5e:shoulder_pan_joint": 0,
            "robot0:ur5e:shoulder_lift_joint": -np.pi / 2.0,
            "robot0:ur5e:elbow_joint": -np.pi / 2.0,
            "robot0:ur5e:wrist_1_joint": -np.pi / 2.0,
            "robot0:ur5e:wrist_2_joint": np.pi / 2.0,
            "robot0:ur5e:wrist_3_joint": 0,
        }
        for joint_name, joint_pos in self.init_qpos_config.items():
            joint_id = self.model_names.joint_name2id[joint_name]
            qpos_id = self.model.jnt_qposadr[joint_id]
            self.init_qpos[qpos_id] = joint_pos

    def step(self, action):
        # Extract robot arm action (first 6 elements) and gripper action (last element)
        robot_action = action[:6]
        gripper_action = action[6]
        
        for i in range(self.frame_skip):
            ctrl = self.data.ctrl.copy()
            
            # Apply operational space control for the robot arm
            self.controller.run(robot_action, ctrl)
            
            # Simple gripper control (direct torque)
            ctrl[6] = gripper_action * 5.0  # Scale gripper action
            
            # Apply the control
            self.do_simulation(ctrl, n_frames=1)

        # Get new observation
        obs = self._get_obs()

        # Calculate reward (simple distance-based reward for now)
        reward = 0.0
        terminated = False
        truncated = False
        info = {}

        return obs, reward, terminated, truncated, info

    def reset_model(self):
        """Reset the model state"""
        # Reset robot to initial state
        self.data.qvel[:] = self.init_qvel
        self.data.ctrl[:] = self.init_ctrl
        
        # Reset object positions to initial positions
        self.reset_objects()
        
        # Update controller
        self.controller.reset()
        
        return self._get_obs()

    def reset(self, *, seed=None, options=None):
        return super().reset(seed=seed, options=options)

    def reset_objects(self):
        """Reset graspable objects to their initial positions"""
        try:
            # Reset cube
            if 'cube_object' in self.model_names.body_names:
                cube_id = self.model_names.body_names.index('cube_object')
                if cube_id < len(self.model.body_jntadr) and self.model.body_jntadr[cube_id] >= 0:
                    start_idx = self.model.body_jntadr[cube_id]
                    self.data.qpos[start_idx:start_idx+7] = [0.5, 0.1, 0.45, 1, 0, 0, 0]
            
            # Reset sphere
            if 'sphere_object' in self.model_names.body_names:
                sphere_id = self.model_names.body_names.index('sphere_object')
                if sphere_id < len(self.model.body_jntadr) and self.model.body_jntadr[sphere_id] >= 0:
                    start_idx = self.model.body_jntadr[sphere_id]
                    self.data.qpos[start_idx:start_idx+7] = [0.5, -0.1, 0.45, 1, 0, 0, 0]
            
            # Reset cylinder
            if 'cylinder_object' in self.model_names.body_names:
                cylinder_id = self.model_names.body_names.index('cylinder_object')
                if cylinder_id < len(self.model.body_jntadr) and self.model.body_jntadr[cylinder_id] >= 0:
                    start_idx = self.model.body_jntadr[cylinder_id]
                    self.data.qpos[start_idx:start_idx+7] = [0.6, 0, 0.45, 1, 0, 0, 0]
        except Exception as e:
            print(f"Warning: Could not reset objects: {e}")
            pass

    def _get_obs(self):
        """Get current observation"""
        # Get robot joint observations
        robot_obs = robot_get_obs(self.model, self.data, self.model_names.joint_names)
        
        return robot_obs

    def get_camera_image(self, camera_name="realsense_rgb", width=640, height=480):
        """Get camera image from the RealSense camera"""
        return self.render(mode="rgb_array", camera_name=camera_name, width=width, height=height)

    def get_depth_image(self, camera_name="realsense_depth", width=640, height=480):
        """Get depth image from the RealSense camera"""
        return self.render(mode="depth_array", camera_name=camera_name, width=width, height=height)

    def get_object_positions(self):
        """Get positions of all graspable objects"""
        positions = {}
        
        try:
            # Get cube position
            if 'cube_object' in self.model_names.body_names:
                cube_id = self.model_names.body_names.index('cube_object')
                positions['cube'] = self.data.xpos[cube_id].copy()
            
            # Get sphere position
            if 'sphere_object' in self.model_names.body_names:
                sphere_id = self.model_names.body_names.index('sphere_object')
                positions['sphere'] = self.data.xpos[sphere_id].copy()
            
            # Get cylinder position
            if 'cylinder_object' in self.model_names.body_names:
                cylinder_id = self.model_names.body_names.index('cylinder_object')
                positions['cylinder'] = self.data.xpos[cylinder_id].copy()
        except Exception as e:
            print(f"Warning: Could not get object positions: {e}")
        
        return positions

    def get_end_effector_position(self):
        """Get end effector position"""
        eef_id = self.model_names.site_names.index('robot0:eef_site')
        return self.data.site_xpos[eef_id].copy()

    def get_gripper_position(self):
        """Get gripper opening position"""
        return self.data.qpos[self.model_names.joint_names.index('robot0:2f85:right_driver_joint')]