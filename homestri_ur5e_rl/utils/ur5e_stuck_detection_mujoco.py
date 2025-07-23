import numpy as np
import mujoco
from collections import deque
from typing import Dict, List, Tuple

class StuckDetectionMixin:
    """
    Mixin class for detecting robot stuck/tangled states in MuJoCo
    Works with single environments (not vectorized)
    """
    
    def __init__(self):
        # Stuck detection parameters
        self.stuck_velocity_threshold = 0.01  # rad/s or m/s
        self.stuck_time_threshold = 2.0  # seconds
        self.stuck_penalty = -2.0
        
        # Tracking for single environment (simplified)
        self.stuck_tracking = {
            'stuck_timer': 0.0,
            'is_stuck': False,
            'last_ee_position': None,
            'last_joint_positions': None,
            'position_history': deque(maxlen=100),  # ~2 seconds at 50Hz
            'velocity_history': deque(maxlen=100),
            'last_stuck_config': None,
        }
        
        # Critical collision pairs to monitor
        self.critical_link_pairs = [
            ("gripper_base", "shoulder_link"),
            ("gripper_base", "upper_arm_link"), 
            ("gripper_base", "forearm_link"),
            ("wrist_3_link", "shoulder_link"),
            ("wrist_3_link", "upper_arm_link"),
            ("forearm_link", "shoulder_link"),
        ]
        
        # Distance thresholds for different link pairs
        self.collision_thresholds = {
            ("gripper_base", "shoulder_link"): 0.12,
            ("gripper_base", "upper_arm_link"): 0.10,
            ("gripper_base", "forearm_link"): 0.08,
            ("wrist_3_link", "shoulder_link"): 0.15,
            ("wrist_3_link", "upper_arm_link"): 0.12,
            ("forearm_link", "shoulder_link"): 0.15,
        }
        
    def initialize_stuck_detection(self, env_id: int):
        """Initialize tracking for environment (simplified for single env)"""
        
        self.stuck_tracking = {
            'stuck_timer': 0.0,
            'is_stuck': False,
            'last_ee_position': None,
            'last_joint_positions': None,
            'position_history': deque(maxlen=100),  # ~2 seconds at 50Hz
            'velocity_history': deque(maxlen=100),
            'last_stuck_config': None,
        }
        
    def check_robot_stuck(self, env_id: int) -> Tuple[bool, str]:
        """
        Check if robot is stuck/tangled in MuJoCo (single environment)
        Returns: (is_stuck, reason)
        """
        
        if not hasattr(self, 'data') or not hasattr(self, 'model'):
            return False, "MuJoCo model/data not available"
            
        tracking = self.stuck_tracking  # Direct access (no env_id needed)
        
        # Get current state
        ee_pos = self.data.site_xpos[self.ee_site_id].copy()
        joint_positions = np.array([self.data.qpos[joint_id] for joint_id in self.arm_joint_ids])
        joint_velocities = np.array([self.data.qvel[joint_id] for joint_id in self.arm_joint_ids])
        
        # 1. Check for self-collision/tangling
        is_tangled, tangle_info = self._check_self_tangling_mujoco()
        if is_tangled:
            return True, f"SELF_TANGLE: {tangle_info}"
        
        # 2. Check end-effector movement
        ee_stuck = False
        if tracking['last_ee_position'] is not None:
            ee_displacement = np.linalg.norm(ee_pos - tracking['last_ee_position'])
            tracking['position_history'].append(ee_displacement)
            
            # Check if EE hasn't moved significantly over time window
            if len(tracking['position_history']) >= 100:  # Full window
                total_movement = sum(tracking['position_history'])
                if total_movement < 0.01:  # Less than 1cm total movement
                    ee_stuck = True
                    
        # 3. Check joint velocities
        joints_stuck = False
        total_joint_movement = np.sum(np.abs(joint_velocities))
        tracking['velocity_history'].append(total_joint_movement)
        
        if len(tracking['velocity_history']) >= 100:
            avg_velocity = np.mean(tracking['velocity_history'])
            if avg_velocity < self.stuck_velocity_threshold:
                joints_stuck = True
                
        # 4. Combined stuck detection
        if ee_stuck and joints_stuck:
            tracking['stuck_timer'] += getattr(self, 'dt', 0.01) * getattr(self, 'frame_skip', 1)
            
            # Store the stuck configuration
            if tracking['stuck_timer'] > 0.5 and tracking['last_stuck_config'] is None:
                tracking['last_stuck_config'] = {
                    'joint_positions': joint_positions.copy(),
                    'ee_position': ee_pos.copy(),
                }
                
        else:
            tracking['stuck_timer'] = 0.0
            tracking['is_stuck'] = False
            tracking['last_stuck_config'] = None
            
        # Update tracking
        tracking['last_ee_position'] = ee_pos.copy()
        tracking['last_joint_positions'] = joint_positions.copy()
        
        # Check if stuck for too long
        if tracking['stuck_timer'] > self.stuck_time_threshold:
            tracking['is_stuck'] = True
            stuck_reason = self._get_stuck_diagnostics_mujoco()
            return True, stuck_reason
            
        return False, ""
        
    def _check_self_tangling_mujoco(self) -> Tuple[bool, str]:
        """
        Check for gripper tangled with robot joints in MuJoCo
        """
        
        # Get link positions from MuJoCo
        link_positions = self._get_all_link_positions_mujoco()
        
        # Check critical collision pairs
        for link1_name, link2_name in self.critical_link_pairs:
            if link1_name in link_positions and link2_name in link_positions:
                pos1 = link_positions[link1_name]
                pos2 = link_positions[link2_name]
                
                distance = np.linalg.norm(pos1 - pos2)
                threshold = self.collision_thresholds.get(
                    (link1_name, link2_name), 
                    0.1  # Default threshold
                )
                
                if distance < threshold:
                    # Additional check: gripper orientation
                    if "gripper" in link1_name:
                        if self._is_gripper_orientation_tangled_mujoco():
                            return True, f"{link1_name} tangled with {link2_name} (dist: {distance:.3f}m)"
                            
        return False, ""
        
    def _get_all_link_positions_mujoco(self) -> Dict[str, np.ndarray]:
        """Get positions of all robot links from MuJoCo"""
        
        link_positions = {}
        
        # Map MuJoCo body names to our link names
        body_name_mapping = {
            "robot0:ur5e:base": "base_link",
            "robot0:ur5e:shoulder_link": "shoulder_link",
            "robot0:ur5e:upper_arm_link": "upper_arm_link",
            "robot0:ur5e:forearm_link": "forearm_link",
            "robot0:ur5e:wrist_1_link": "wrist_1_link",
            "robot0:ur5e:wrist_2_link": "wrist_2_link",
            "robot0:ur5e:wrist_3_link": "wrist_3_link",
            "robot0:2f85:base": "gripper_base",
        }
        
        for mujoco_name, our_name in body_name_mapping.items():
            try:
                body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, mujoco_name)
                if body_id >= 0:
                    link_positions[our_name] = self.data.xpos[body_id].copy()
            except:
                # Body not found, skip
                continue
                
        return link_positions
        
    def _is_gripper_orientation_tangled_mujoco(self) -> bool:
        """
        Check if gripper is in a tangled orientation in MuJoCo
        """
        
        try:
            # Get gripper body
            gripper_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "robot0:2f85:base")
            if gripper_body_id < 0:
                return False
                
            # Get gripper orientation (rotation matrix)
            gripper_rot_mat = self.data.xmat[gripper_body_id].reshape(3, 3)
            
            # Convert to euler angles (simplified check)
            # Check if gripper is pointing in a problematic direction
            gripper_z_axis = gripper_rot_mat[:, 2]  # Z-axis of gripper
            
            # If gripper Z-axis is pointing up (tangled backward)
            if gripper_z_axis[2] > 0.7:  # Pointing significantly upward
                return True
                
            # Check joint limits - if multiple joints near limits, likely tangled
            joint_positions = np.array([self.data.qpos[joint_id] for joint_id in self.arm_joint_ids])
            joints_near_limit = 0
            
            for i, joint_id in enumerate(self.arm_joint_ids):
                joint_range = self.model.jnt_range[joint_id]
                if joint_range[0] < joint_range[1]:  # Has limits
                    pos = joint_positions[i]
                    margin = np.deg2rad(10)  # 10 degree margin
                    if pos < joint_range[0] + margin or pos > joint_range[1] - margin:
                        joints_near_limit += 1
                        
            if joints_near_limit >= 3:  # Multiple joints at limits = likely tangled
                return True
                
        except Exception as e:
            # If we can't check orientation, assume not tangled
            pass
            
        return False
        
    def _get_stuck_diagnostics_mujoco(self) -> str:
        """Generate detailed diagnostics for stuck state in MuJoCo"""
        
        tracking = self.stuck_tracking
        
        # Build diagnostic string
        diag = f"STUCK after {tracking['stuck_timer']:.1f}s"
        
        if tracking['last_stuck_config']:
            config = tracking['last_stuck_config']
            joint_str = [f"{np.rad2deg(j):.1f}°" for j in config['joint_positions']]
            diag += f" | Joints: {joint_str}"
            diag += f" | EE: [{config['ee_position'][0]:.3f}, {config['ee_position'][1]:.3f}, {config['ee_position'][2]:.3f}]"
            
        # Check specific stuck patterns
        if self._is_wrist_wrapped_mujoco():
            diag += " | WRIST_WRAPPED"
        if self._is_elbow_locked_mujoco():
            diag += " | ELBOW_LOCKED"
            
        return diag
        
    def _is_wrist_wrapped_mujoco(self) -> bool:
        """Check if wrist joints are wrapped around in MuJoCo"""
        
        # Get wrist joint positions (last 3 joints)
        wrist_joints = self.arm_joint_ids[-3:]  # wrist_1, wrist_2, wrist_3
        wrist_positions = [self.data.qpos[joint_id] for joint_id in wrist_joints]
        
        # Check for extreme positions
        for pos in wrist_positions:
            if abs(np.rad2deg(pos)) > 350:
                return True
                
        # Check for opposing extremes
        if len(wrist_positions) >= 2:
            pos1, pos2 = np.rad2deg(wrist_positions[0]), np.rad2deg(wrist_positions[2])
            if (pos1 > 170 and pos2 < -170) or (pos1 < -170 and pos2 > 170):
                return True
                
        return False
        
    def _is_elbow_locked_mujoco(self) -> bool:
        """Check if elbow is in locked position in MuJoCo"""
        
        if len(self.arm_joint_ids) < 3:
            return False
            
        # Get elbow and shoulder positions
        elbow_pos = self.data.qpos[self.arm_joint_ids[2]]  # elbow_joint
        shoulder_pos = self.data.qpos[self.arm_joint_ids[1]]  # shoulder_lift_joint
        
        elbow_deg = np.rad2deg(elbow_pos)
        shoulder_deg = np.rad2deg(shoulder_pos)
        
        # Locked configuration: elbow fully bent and shoulder lifted
        if elbow_deg < -150 and shoulder_deg < -120:
            # Check if gripper is close to upper arm
            link_positions = self._get_all_link_positions_mujoco()
            if "gripper_base" in link_positions and "upper_arm_link" in link_positions:
                dist = np.linalg.norm(
                    link_positions["gripper_base"] - link_positions["upper_arm_link"]
                )
                if dist < 0.15:
                    return True
                    
        return False
        
    def get_stuck_info(self) -> Dict:
        """Get detailed stuck detection info for logging"""
        
        tracking = self.stuck_tracking
        
        info = {
            'stuck_timer': tracking['stuck_timer'],
            'is_stuck': tracking['is_stuck'],
            'position_history_len': len(tracking['position_history']),
            'velocity_history_len': len(tracking['velocity_history']),
        }
        
        if tracking['position_history']:
            info['recent_movement'] = sum(list(tracking['position_history'])[-10:])
            
        if tracking['velocity_history']:
            info['avg_velocity'] = np.mean(list(tracking['velocity_history'])[-10:])
            
        if tracking['last_stuck_config']:
            info['stuck_config'] = tracking['last_stuck_config'].copy()
            
        return info