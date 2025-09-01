"""
Real Robot Deployment Script for UR5e with Robotiq 2F-85
Includes safety features and real-time monitoring
NOTE: THIS SCRIPT IS NOT FULLY IMPLEMENTED - I LEAVE IMPLEMENTATION UPTO YOU
NOTE: The purpose of the study is mostly on implementing PPO via simulation WHICH CAN SUPPORT REAL ROBOT DEPLOYMENT VIA EXTENSIVE DOMAIN-RANDOMIZATION TECHNIQUES, as my lab doesn't have access to the Robotiq 2F-85 gripper.
"""

import numpy as np
import torch
import time
from pathlib import Path
from typing import Dict, Optional, Tuple, List
import threading
import queue
from dataclasses import dataclass
from datetime import datetime
import cv2
import pyrealsense2 as rs

# ROS imports (uncomment when using with ROS)
# import rospy
# from sensor_msgs.msg import JointState, Image
# from geometry_msgs.msg import WrenchStamped
# from cv_bridge import CvBridge

# Import trained model components
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from homestri_ur5e_rl.envs import UR5ePickPlaceEnvEnhanced

# UR RTDE for real-time control
import rtde_control
import rtde_receive

@dataclass
class SafetyConfig:
    """Safety parameters for real robot operation"""
    max_velocity: float = 1.0  # rad/s
    max_acceleration: float = 1.2  # rad/s^2
    max_force: float = 150.0  # N
    max_torque: float = 50.0  # Nm
    workspace_limits: Dict[str, Tuple[float, float]] = None
    emergency_stop_force: float = 200.0  # N
    
    def __post_init__(self):
        if self.workspace_limits is None:
            self.workspace_limits = {
                'x': (0.2, 0.8),
                'y': (-0.5, 0.5),
                'z': (0.0, 1.0)
            }

class RealSenseInterface:
    """Interface for RealSense D435i camera"""
    
    def __init__(self, resolution: int = 128):
        self.resolution = resolution
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        
        # Configure streams
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        
        # Start streaming
        self.profile = self.pipeline.start(self.config)
        
        # Get depth scale
        depth_sensor = self.profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()
        
        # Align depth to color
        self.align = rs.align(rs.stream.color)
        
        print(" RealSense D435i initialized")
        
    def get_rgbd_observation(self) -> np.ndarray:
        """Get RGBD observation matching training format"""
        
        # Wait for frames
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        
        # Get aligned frames
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        
        if not depth_frame or not color_frame:
            return np.zeros(self.resolution * self.resolution * 4)
            
        # Convert to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        
        # Convert depth to meters
        depth_meters = depth_image * self.depth_scale
        
        # Resize to match training resolution
        color_resized = cv2.resize(color_image, (self.resolution, self.resolution))
        depth_resized = cv2.resize(depth_meters, (self.resolution, self.resolution))
        
        # Normalize depth (matching training)
        depth_normalized = np.clip(depth_resized / 3.0, 0, 1)  # 3m max range
        
        # Convert to RGBD format
        rgb_normalized = color_resized.astype(np.float32) / 255.0
        rgbd = np.zeros((self.resolution, self.resolution, 4), dtype=np.float32)
        rgbd[:, :, :3] = rgb_normalized
        rgbd[:, :, 3] = depth_normalized
        
        return rgbd.flatten()
        
    def close(self):
        """Stop the pipeline"""
        self.pipeline.stop()

class UR5eRealRobotInterface:
    """Safe interface for real UR5e robot"""
    
    def __init__(
        self,
        robot_ip: str,
        safety_config: SafetyConfig,
        gripper_port: int = 63352,
    ):
        self.robot_ip = robot_ip
        self.safety_config = safety_config
        
        # Initialize RTDE interfaces
        self.rtde_c = rtde_control.RTDEControlInterface(robot_ip)
        self.rtde_r = rtde_receive.RTDEReceiveInterface(robot_ip)
        
        # Initialize gripper (Robotiq specific)
        # You'll need the Robotiq gripper library
        # self.gripper = RobotiqGripper(robot_ip, gripper_port)
        
        # Safety monitoring thread
        self.safety_thread = threading.Thread(target=self._safety_monitor)
        self.safety_stop = threading.Event()
        self.emergency_stop = threading.Event()
        
        # Start safety monitoring
        self.safety_thread.start()
        
        print(f" Connected to UR5e at {robot_ip}")
        
    def _safety_monitor(self):
        """Continuous safety monitoring in separate thread"""
        
        while not self.safety_stop.is_set():
            # Check force/torque
            tcp_force = self.rtde_r.getActualTCPForce()
            force_magnitude = np.linalg.norm(tcp_force[:3])
            torque_magnitude = np.linalg.norm(tcp_force[3:])
            
            if force_magnitude > self.safety_config.emergency_stop_force:
                print(f" EMERGENCY STOP: Force {force_magnitude:.1f}N exceeds limit!")
                self.emergency_stop.set()
                self.rtde_c.stopScript()
                
            # Check workspace limits
            tcp_pose = self.rtde_r.getActualTCPPose()
            for axis, (min_val, max_val) in self.safety_config.workspace_limits.items():
                idx = {'x': 0, 'y': 1, 'z': 2}[axis]
                if not min_val <= tcp_pose[idx] <= max_val:
                    print(f" Workspace violation: {axis}={tcp_pose[idx]:.3f}")
                    self.emergency_stop.set()
                    
            time.sleep(0.01)  # 100Hz monitoring
            
    def get_observation(self, camera: RealSenseInterface, object_tracker=None) -> np.ndarray:
        """Get full observation matching training format"""
        
        obs_dict = {}
        
        # Joint states
        joint_pos = np.array(self.rtde_r.getActualQ())
        joint_vel = np.array(self.rtde_r.getActualQd())
        obs_dict["joint_pos"] = joint_pos
        obs_dict["joint_vel"] = joint_vel
        
        # End effector pose
        tcp_pose = self.rtde_r.getActualTCPPose()
        ee_pos = np.array(tcp_pose[:3])
        ee_quat = self._rpy_to_quat(tcp_pose[3:])  # Convert RPY to quaternion
        obs_dict["ee_pose"] = np.concatenate([ee_pos, ee_quat])
        
        # Gripper state (0-1)
        # gripper_pos = self.gripper.get_position() / 255.0
        gripper_pos = 0.0  # Placeholder
        obs_dict["gripper"] = np.array([gripper_pos])
        
        # Force/torque
        tcp_force = self.rtde_r.getActualTCPForce()
        obs_dict["force_torque"] = np.array(tcp_force)
        
        # Object state (from vision/tracker)
        if object_tracker is not None:
            object_state = object_tracker.get_object_state()
        else:
            # Placeholder - would use AprilTags or vision
            object_state = np.zeros(13)  # pos(3) + quat(4) + vel(6)
        obs_dict["object_state"] = object_state
        
        # Goal position (set by task)
        obs_dict["goal"] = self.target_position
        
        # Camera data
        camera_data = camera.get_rgbd_observation()
        obs_dict["camera"] = camera_data
        
        # Concatenate in same order as training
        obs = np.concatenate([
            obs_dict["joint_pos"],      # 6
            obs_dict["joint_vel"],      # 6
            obs_dict["ee_pose"],        # 7
            obs_dict["gripper"],        # 1
            obs_dict["force_torque"],   # 6
            obs_dict["object_state"],   # 13
            obs_dict["goal"],          # 3
            obs_dict["camera"],        # resolution^2 * 4
        ])
        
        return obs.astype(np.float32)
        
    def execute_action(self, action: np.ndarray, safety_scale: float = 0.7):
        """Execute action with safety checks"""
        
        if self.emergency_stop.is_set():
            print(" Emergency stop active - cannot execute action")
            return False
            
        # Scale down actions for safety
        action = action * safety_scale
        
        # Get current joint positions
        current_joints = self.rtde_r.getActualQ()
        
        if self.control_mode == "joint_velocity":
            # Velocity control
            joint_velocities = action[:6] * self.safety_config.max_velocity
            
            # Apply velocity limits
            joint_velocities = np.clip(
                joint_velocities,
                -self.safety_config.max_velocity,
                self.safety_config.max_velocity
            )
            
            # Send velocity command
            self.rtde_c.speedJ(joint_velocities.tolist(), self.safety_config.max_acceleration)
            
        else:  # joint_position
            # Position control with interpolation
            target_positions = current_joints + action[:6] * 0.1  # Small increments
            
            # Check joint limits
            # UR5e limits: [-2π, 2π] for most joints
            target_positions = np.clip(target_positions, -2*np.pi, 2*np.pi)
            
            # Move to target
            self.rtde_c.moveJ(
                target_positions.tolist(),
                self.safety_config.max_velocity,
                self.safety_config.max_acceleration
            )
            
        # Gripper control
        gripper_action = action[6] if len(action) > 6 else 0
        if gripper_action > 0.5:
            # self.gripper.close()
            pass  # Placeholder
        elif gripper_action < -0.5:
            # self.gripper.open()
            pass  # Placeholder
            
        return True
        
    def _rpy_to_quat(self, rpy):
        """Convert roll-pitch-yaw to quaternion"""
        roll, pitch, yaw = rpy
        
        cr = np.cos(roll * 0.5)
        sr = np.sin(roll * 0.5)
        cp = np.cos(pitch * 0.5)
        sp = np.sin(pitch * 0.5)
        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)
        
        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy
        
        return np.array([w, x, y, z])
        
    def move_to_home(self):
        """Move to safe home position"""
        home_joints = [0, -np.pi/2, np.pi/2, -np.pi/2, -np.pi/2, 0]
        self.rtde_c.moveJ(home_joints, 0.5, 0.5)
        print(" Moved to home position")
        
    def close(self):
        """Clean shutdown"""
        self.safety_stop.set()
        self.safety_thread.join()
        self.rtde_c.stopScript()
        # self.gripper.close_connection()

class RealRobotDeployment:
    """Main deployment class for real robot"""
    
    def __init__(
        self,
        model_path: str,
        robot_ip: str,
        safety_config: Optional[SafetyConfig] = None,
        camera_resolution: int = 128,
        control_mode: str = "joint_velocity",
    ):
        self.model_path = Path(model_path)
        self.camera_resolution = camera_resolution
        self.control_mode = control_mode
        
        # Safety configuration
        self.safety_config = safety_config or SafetyConfig()
        
        # Load trained model
        print(f"📂 Loading model from {model_path}")
        self.model = PPO.load(str(self.model_path / "best_model.zip"))
        
        # Load normalization statistics
        vec_normalize_path = self.model_path / "vec_normalize.pkl"
        if vec_normalize_path.exists():
            self.vec_normalize = VecNormalize.load(str(vec_normalize_path))
            self.vec_normalize.training = False
        else:
            print(" No normalization statistics found")
            self.vec_normalize = None
            
        # Initialize robot interface
        self.robot = UR5eRealRobotInterface(robot_ip, self.safety_config)
        self.robot.control_mode = control_mode
        
        # Initialize camera
        self.camera = RealSenseInterface(camera_resolution)
        
        # Task parameters
        self.target_position = np.array([0.6, -0.2, 0.45])  # Default target
        self.robot.target_position = self.target_position
        
        # Performance tracking
        self.episode_count = 0
        self.success_count = 0
        self.performance_log = []
        
        print(" Real robot deployment initialized")
        
    def run_episode(self, render: bool = True) -> Dict:
        """Run one complete pick-and-place episode"""
        
        print(f"\n Starting episode {self.episode_count + 1}")
        start_time = time.time()
        
        # Move to home position
        self.robot.move_to_home()
        time.sleep(1.0)
        
        # Episode tracking
        episode_data = {
            'steps': 0,
            'success': False,
            'termination_reason': None,
            'total_reward': 0.0,
            'duration': 0.0,
        }
        
        # Get initial observation
        obs = self.robot.get_observation(self.camera)
        
        # Normalize observation if needed
        if self.vec_normalize is not None:
            obs = self.vec_normalize.normalize_obs(obs)
            
        # Run episode
        done = False
        max_steps = 1000
        
        while not done and episode_data['steps'] < max_steps:
            # Get action from policy
            action, _ = self.model.predict(obs, deterministic=True)
            
            # Safety checks
            if self.robot.emergency_stop.is_set():
                episode_data['termination_reason'] = 'emergency_stop'
                break
                
            # Execute action
            success = self.robot.execute_action(action)
            if not success:
                episode_data['termination_reason'] = 'action_failed'
                break
                
            # Wait for action to complete
            time.sleep(0.1)  # 10Hz control
            
            # Get new observation
            obs = self.robot.get_observation(self.camera)
            if self.vec_normalize is not None:
                obs = self.vec_normalize.normalize_obs(obs)
                
            # Check success (simplified - would use vision in practice)
            if self._check_task_success():
                episode_data['success'] = True
                episode_data['termination_reason'] = 'success'
                done = True
                
            # Render if requested
            if render:
                self._render_status(episode_data)
                
            episode_data['steps'] += 1
            
        # Episode complete
        episode_data['duration'] = time.time() - start_time
        self.episode_count += 1
        if episode_data['success']:
            self.success_count += 1
            
        # Log performance
        self.performance_log.append(episode_data)
        
        print(f"\n Episode {self.episode_count} complete:")
        print(f"   Success: {episode_data['success']}")
        print(f"   Steps: {episode_data['steps']}")
        print(f"   Duration: {episode_data['duration']:.1f}s")
        print(f"   Reason: {episode_data['termination_reason']}")
        print(f"   Overall success rate: {self.success_count}/{self.episode_count} "
              f"({100*self.success_count/self.episode_count:.1f}%)")
              
        return episode_data
        
    def _check_task_success(self) -> bool:
        """Check if pick-and-place task is successful"""
        # Simplified check - in practice would use vision
        tcp_pose = self.robot.rtde_r.getActualTCPPose()
        current_pos = np.array(tcp_pose[:3])
        
        # Check if near target with object
        distance = np.linalg.norm(current_pos - self.target_position)
        return distance < 0.05  # 5cm tolerance
        
    def _render_status(self, episode_data: Dict):
        """Display real-time status"""
        # Would implement visualization here
        pass
        
    def run_experiments(self, n_episodes: int = 10):
        """Run multiple episodes for evaluation"""
        
        print(f"\n🔬 Running {n_episodes} episodes for evaluation...")
        
        for i in range(n_episodes):
            try:
                episode_data = self.run_episode(render=(i == 0))  # Only render first
                
                # Safety pause between episodes
                time.sleep(2.0)
                
            except KeyboardInterrupt:
                print("\n Interrupted by user")
                break
            except Exception as e:
                print(f"\n Episode failed with error: {e}")
                self.robot.rtde_c.stopScript()
                time.sleep(1.0)
                
        # Final statistics
        self._print_statistics()
        
    def _print_statistics(self):
        """Print performance statistics"""
        
        if not self.performance_log:
            return
            
        successes = [ep['success'] for ep in self.performance_log]
        durations = [ep['duration'] for ep in self.performance_log if ep['success']]
        steps = [ep['steps'] for ep in self.performance_log if ep['success']]
        
        print("\n" + "="*60)
        print(" DEPLOYMENT STATISTICS")
        print("="*60)
        print(f"Total episodes: {len(self.performance_log)}")
        print(f"Successful episodes: {sum(successes)}")
        print(f"Success rate: {100*np.mean(successes):.1f}%")
        
        if durations:
            print(f"Average success time: {np.mean(durations):.1f}s ± {np.std(durations):.1f}s")
            print(f"Average success steps: {np.mean(steps):.0f} ± {np.std(steps):.0f}")
            
        # Save results
        results_path = self.model_path / f"real_robot_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        import json
        with open(results_path, 'w') as f:
            json.dump({
                'episodes': self.performance_log,
                'summary': {
                    'success_rate': float(np.mean(successes)),
                    'avg_duration': float(np.mean(durations)) if durations else 0,
                    'avg_steps': float(np.mean(steps)) if steps else 0,
                }
            }, f, indent=2)
            
        print(f"\n💾 Results saved to: {results_path}")
        
    def close(self):
        """Clean shutdown"""
        print("\n🔚 Shutting down...")
        self.robot.close()
        self.camera.close()
        print(" Shutdown complete")

def main():
    """Main entry point for real robot deployment"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description="Deploy trained model to real UR5e")
    parser.add_argument("--model", type=str, required=True,
                       help="Path to trained model directory")
    parser.add_argument("--robot-ip", type=str, required=True,
                       help="IP address of UR5e robot")
    parser.add_argument("--episodes", type=int, default=10,
                       help="Number of episodes to run")
    parser.add_argument("--camera-res", type=int, default=128,
                       help="Camera resolution")
    parser.add_argument("--control-mode", type=str, default="joint_velocity",
                       choices=["joint_velocity", "joint_position"],
                       help="Control mode")
    parser.add_argument("--safety-scale", type=float, default=0.7,
                       help="Safety scaling factor for actions (0-1)")
    
    args = parser.parse_args()
    
    print("🤖 UR5e Real Robot Deployment")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"Robot IP: {args.robot_ip}")
    print(f"Episodes: {args.episodes}")
    print(f"Camera: {args.camera_res}x{args.camera_res}")
    print(f"Control: {args.control_mode}")
    print(f"Safety scale: {args.safety_scale}")
    print("="*60)
    
    # Safety configuration
    safety_config = SafetyConfig(
        max_velocity=0.5,  # Conservative limits
        max_acceleration=0.5,
        max_force=100.0,
        max_torque=30.0,
    )
    
    deployment = RealRobotDeployment(
        model_path=args.model,
        robot_ip=args.robot_ip,
        safety_config=safety_config,
        camera_resolution=args.camera_res,
        control_mode=args.control_mode,
    )
    
    try:
        # Run experiments
        deployment.run_experiments(n_episodes=args.episodes)
        
    except Exception as e:
        print(f"\n Deployment error: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        deployment.close()

if __name__ == "__main__":
    main()