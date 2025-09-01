"""
Real Robot Deployment Utilities for UR5e Pick-and-Place
Bridges sim-to-real gap with homestri integration
NOTE: As mentioned earlier in documentation, this codebase is intended for simulation based training and hasn't been tested via sim-to-real. Although the randomization factors should support some level of generalization, real-world performance may vary.
"""

import numpy as np
import cv2
import time
import json
import threading
from typing import Dict, Tuple, Optional, List
from pathlib import Path
import torch

# RealSense camera
import pyrealsense2 as rs

# UR5e robot control (using homestri patterns)
try:
    import rtde_control
    import rtde_receive
    RTDE_AVAILABLE = True
except ImportError:
    print(" RTDE not available. Install python-rtde for real robot control.")
    RTDE_AVAILABLE = False

# ROS integration
try:
    import rospy
    from sensor_msgs.msg import Image, PointCloud2
    from geometry_msgs.msg import Pose, PoseStamped
    from std_msgs.msg import Bool
    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize

class RealSenseCamera:
    """
    Real RealSense D435i camera interface matching simulation
    """
    
    def __init__(self, resolution: int = 128, fps: int = 30):
        self.resolution = resolution
        self.fps = fps
        
        # Configure RealSense pipeline
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        
        # Configure streams
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, fps)
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, fps)
        
        # Start pipeline
        self.profile = self.pipeline.start(self.config)
        
        # Get camera intrinsics
        color_profile = self.profile.get_stream(rs.stream.color)
        self.intrinsics = color_profile.as_video_stream_profile().get_intrinsics()
        
        # Align depth to color
        self.align = rs.align(rs.stream.color)
        
        # Filters for depth processing
        self.decimation_filter = rs.decimation_filter()
        self.spatial_filter = rs.spatial_filter()
        self.temporal_filter = rs.temporal_filter()
        self.hole_filling_filter = rs.hole_filling_filter()
        
        print(f" RealSense camera initialized: {resolution}x{resolution} @ {fps}fps")
        
    def get_rgbd(self) -> np.ndarray:
        """Get RGBD data matching simulation format"""
        try:
            # Wait for frames
            frames = self.pipeline.wait_for_frames()
            aligned_frames = self.align.process(frames)
            
            # Get color and depth frames
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            
            if not color_frame or not depth_frame:
                return np.zeros((self.resolution, self.resolution, 4), dtype=np.float32)
                
            # Apply filters to depth
            depth_frame = self.decimation_filter.process(depth_frame)
            depth_frame = self.spatial_filter.process(depth_frame)
            depth_frame = self.temporal_filter.process(depth_frame)
            depth_frame = self.hole_filling_filter.process(depth_frame)
            
            # Convert to numpy arrays
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
            
            # Convert BGR to RGB
            color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
            
            # Resize to match simulation
            color_resized = cv2.resize(color_image, (self.resolution, self.resolution))
            depth_resized = cv2.resize(depth_image, (self.resolution, self.resolution))
            
            # Normalize
            color_normalized = color_resized.astype(np.float32) / 255.0
            
            # Convert depth to meters and normalize like simulation
            depth_meters = depth_resized.astype(np.float32) / 1000.0  # mm to m
            depth_normalized = np.clip(depth_meters / 3.0, 0, 1)  # Normalize to [0,1]
            
            # Combine RGBD
            rgbd = np.zeros((self.resolution, self.resolution, 4), dtype=np.float32)
            rgbd[:, :, :3] = color_normalized
            rgbd[:, :, 3] = depth_normalized
            
            return rgbd
            
        except Exception as e:
            print(f" Camera error: {e}")
            return np.zeros((self.resolution, self.resolution, 4), dtype=np.float32)
            
    def get_point_cloud(self) -> np.ndarray:
        """Get 3D point cloud for debugging"""
        try:
            frames = self.pipeline.wait_for_frames()
            aligned_frames = self.align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            
            if not depth_frame:
                return np.array([])
                
            pc = rs.pointcloud()
            points = pc.calculate(depth_frame)
            
            # Convert to numpy
            vertices = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, 3)
            
            # Filter valid points
            valid_mask = ~np.isnan(vertices).any(axis=1) & (vertices[:, 2] > 0)
            return vertices[valid_mask]
            
        except Exception as e:
            print(f" Point cloud error: {e}")
            return np.array([])
            
    def close(self):
        """Clean shutdown"""
        self.pipeline.stop()

class UR5eRealRobot:
    """
    Real UR5e robot interface using RTDE
    """
    
    def __init__(self, robot_ip: str = "192.168.1.102"):
        if not RTDE_AVAILABLE:
            raise RuntimeError("RTDE not available. Install python-rtde.")
            
        self.robot_ip = robot_ip
        
        # Connect to robot
        self.rtde_c = rtde_control.RTDEControlInterface(robot_ip)
        self.rtde_r = rtde_receive.RTDEReceiveInterface(robot_ip)
        
        # Robot parameters
        self.home_joints = [0.0, -np.pi/3, -np.pi/3, -2*np.pi/3, np.pi/2, 0.0]
        self.max_velocity = 0.5  # rad/s
        self.max_acceleration = 1.0  # rad/s^2
        
        # Workspace limits (safety)
        self.workspace_bounds = {
            'x': [0.2, 0.8],
            'y': [-0.4, 0.4],
            'z': [0.1, 0.8],
        }
        
        # Gripper control (assuming Robotiq 2F-85)
        self.gripper_open_width = 85  # mm
        self.gripper_closed_width = 0  # mm
        
        print(f" UR5e robot connected: {robot_ip}")
        
    def get_joint_positions(self) -> np.ndarray:
        """Get current joint positions"""
        return np.array(self.rtde_r.getActualQ())
        
    def get_joint_velocities(self) -> np.ndarray:
        """Get current joint velocities"""
        return np.array(self.rtde_r.getActualQd())
        
    def get_tcp_pose(self) -> np.ndarray:
        """Get current TCP pose [x, y, z, rx, ry, rz]"""
        return np.array(self.rtde_r.getActualTCPPose())
        
    def get_tcp_force(self) -> np.ndarray:
        """Get TCP force/torque"""
        return np.array(self.rtde_r.getActualTCPForce())
        
    def move_joints(self, target_joints: np.ndarray, velocity: float = 0.5, acceleration: float = 1.0) -> bool:
        """Move to target joint positions"""
        try:
            # Safety checks
            target_joints = np.clip(target_joints, -2*np.pi, 2*np.pi)
            velocity = min(velocity, self.max_velocity)
            acceleration = min(acceleration, self.max_acceleration)
            
            # Execute movement
            self.rtde_c.moveJ(target_joints.tolist(), velocity, acceleration)
            return True
            
        except Exception as e:
            print(f" Joint movement error: {e}")
            return False
            
    def move_tcp(self, target_pose: np.ndarray, velocity: float = 0.1, acceleration: float = 0.3) -> bool:
        """Move TCP to target pose"""
        try:
            # Safety check workspace bounds
            if not self._check_workspace_bounds(target_pose[:3]):
                print(f" Target pose outside workspace: {target_pose[:3]}")
                return False
                
            # Execute movement
            self.rtde_c.moveL(target_pose.tolist(), velocity, acceleration)
            return True
            
        except Exception as e:
            print(f" TCP movement error: {e}")
            return False
            
    def set_gripper(self, close: bool, force: float = 50.0, speed: float = 50.0) -> bool:
        """Control gripper (implementation depends on gripper type)"""
        try:
            # TODO: Implement gripper control
            
            if close:
                # Close gripper
                width = self.gripper_closed_width
            else:
                # Open gripper
                width = self.gripper_open_width
                
            # Send gripper command (implement based on your setup)
            # self.gripper_interface.move(width, force, speed)
            
            print(f"Gripper {'closed' if close else 'opened'}")
            return True
            
        except Exception as e:
            print(f" Gripper error: {e}")
            return False
            
    def move_to_home(self) -> bool:
        """Move robot to home position"""
        return self.move_joints(np.array(self.home_joints))
        
    def emergency_stop(self):
        """Emergency stop"""
        try:
            self.rtde_c.stopScript()
            print("Emergency stop activated")
        except Exception as e:
            print(f" Emergency stop error: {e}")
            
    def _check_workspace_bounds(self, position: np.ndarray) -> bool:
        """Check if position is within safe workspace"""
        x, y, z = position
        return (self.workspace_bounds['x'][0] <= x <= self.workspace_bounds['x'][1] and
                self.workspace_bounds['y'][0] <= y <= self.workspace_bounds['y'][1] and
                self.workspace_bounds['z'][0] <= z <= self.workspace_bounds['z'][1])
                
    def close(self):
        """Disconnect from robot"""
        try:
            self.rtde_c.disconnect()
            self.rtde_r.disconnect()
            print(" Robot disconnected")
        except:
            pass

class SimToRealInterface:
    """
    Main interface for deploying trained policies on real robot
    """
    
    def __init__(
        self,
        model_path: str,
        robot_ip: str = "192.168.1.102",
        camera_resolution: int = 128,
        use_ros: bool = False,
    ):
        self.model_path = Path(model_path)
        self.use_ros = use_ros and ROS_AVAILABLE
        
        # Load trained model
        self._load_model()
        
        # Initialize hardware
        self.camera = RealSenseCamera(resolution=camera_resolution)
        self.robot = UR5eRealRobot(robot_ip)
        
        # State tracking
        self.current_observation = None
        self.episode_data = {
            'actions': [],
            'observations': [],
            'rewards': [],
            'camera_frames': [],
        }
        
        # Safety parameters
        self.safety_checks_enabled = True
        self.max_action_change = 0.1  # Limit action changes for safety
        self.last_action = np.zeros(7)
        
        # Performance tracking
        self.execution_times = []
        
        print(f" Sim-to-Real interface initialized")
        print(f"   Model: {self.model_path}")
        print(f"   Robot IP: {robot_ip}")
        print(f"   Camera resolution: {camera_resolution}")
        
    def _load_model(self):
        """Load trained PPO model and normalization"""
        try:
            # Load model
            self.model = PPO.load(str(self.model_path / "best_model.zip"))
            
            # Load normalization if available
            vec_norm_path = self.model_path / "vec_normalize.pkl"
            if vec_norm_path.exists():
                self.vec_normalize = VecNormalize.load(str(vec_norm_path), None)
                self.vec_normalize.training = False
                print(" Observation normalization loaded")
            else:
                self.vec_normalize = None
                print(" No observation normalization found")
                
            self.model.set_env(None)  # Disable environment dependency
            print(f" Model loaded: {self.model_path}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")
            
    def get_observation(self) -> np.ndarray:
        """Get current observation in simulation format"""
        # Get robot state
        joint_pos = self.robot.get_joint_positions()
        joint_vel = self.robot.get_joint_velocities()
        tcp_pose = self.robot.get_tcp_pose()
        tcp_force = self.robot.get_tcp_force()
        
        # Get camera data
        rgbd = self.camera.get_rgbd()
        camera_flat = rgbd.flatten()
        
        # Construct observation (matching simulation format)
        robot_state = np.concatenate([
            joint_pos,              # 6 joints
            joint_vel,              # 6 joint velocities
            tcp_pose[:3],           # TCP position
            tcp_pose[3:],           # TCP orientation (RPY)
            [0.0],                  # Gripper state (placeholder)
            tcp_force,              # Force/torque (6 values)
            [0.0],                  # Object grasped (placeholder)
            [0.0],                  # Normalized time (placeholder)
            [1.0],                  # Curriculum level
        ])
        
        # Object state (placeholder - would need object detection)
        object_state = np.zeros(13)
        
        # Goal state (placeholder - set based on task)
        goal_state = np.zeros(7)
        
        # Combine observation
        observation = np.concatenate([robot_state, object_state, goal_state, camera_flat])
        
        # Apply normalization if available
        if self.vec_normalize is not None:
            observation = self.vec_normalize.normalize_obs(observation.reshape(1, -1)).flatten()
            
        return observation.astype(np.float32)
        
    def predict_action(self, observation: np.ndarray) -> np.ndarray:
        """Predict action using trained model"""
        try:
            start_time = time.time()
            
            # Get action from model
            action, _ = self.model.predict(observation, deterministic=True)
            
            # Safety checks
            if self.safety_checks_enabled:
                action = self._apply_safety_checks(action)
                
            execution_time = time.time() - start_time
            self.execution_times.append(execution_time)
            
            return action
            
        except Exception as e:
            print(f" Prediction error: {e}")
            return np.zeros(7)  # Safe fallback
            
    def _apply_safety_checks(self, action: np.ndarray) -> np.ndarray:
        """Apply safety checks to actions"""
        # Limit action magnitude
        action = np.clip(action, -1.0, 1.0)
        
        # Limit action changes for smooth motion
        action_change = action - self.last_action
        max_change = self.max_action_change
        
        limited_change = np.clip(action_change, -max_change, max_change)
        safe_action = self.last_action + limited_change
        
        self.last_action = safe_action
        return safe_action
        
    def execute_action(self, action: np.ndarray) -> bool:
        """Execute action on real robot"""
        try:
            # Parse action
            joint_action = action[:6]  # Joint control
            gripper_action = action[6] if len(action) > 6 else 0.0
            
            # Scale actions appropriately
            current_joints = self.robot.get_joint_positions()
            
            # Convert action to joint targets 
            joint_velocities = joint_action * 0.5  # Scale to reasonable velocities
            
            # For position control, you might do:
            # target_joints = current_joints + joint_action * 0.1
            
            # Execute robot motion
            # For now, this is a placeholder
            print(f"🤖 Executing action: joints={joint_action}, gripper={gripper_action}")
            
            # Execute gripper action
            if gripper_action > 0.3:
                self.robot.set_gripper(close=True)
            elif gripper_action < -0.3:
                self.robot.set_gripper(close=False)
                
            return True
            
        except Exception as e:
            print(f" Action execution error: {e}")
            return False
            
    def run_episode(self, max_steps: int = 200, visualize: bool = True) -> Dict:
        """Run a complete pick-and-place episode"""
        print(f"\n Starting pick-and-place episode...")
        
        # Reset episode data
        self.episode_data = {
            'actions': [],
            'observations': [],
            'camera_frames': [],
            'execution_times': [],
            'success': False,
        }
        
        # Move to home position
        print(" Moving to home position...")
        if not self.robot.move_to_home():
            return {'success': False, 'error': 'Failed to move to home'}
            
        time.sleep(2.0)  # Allow robot to settle
        
        # Main execution loop
        for step in range(max_steps):
            try:
                step_start = time.time()
                
                # Get observation
                observation = self.get_observation()
                self.episode_data['observations'].append(observation.copy())
                
                # Get camera frame for visualization
                if visualize:
                    rgbd = self.camera.get_rgbd()
                    self.episode_data['camera_frames'].append(rgbd.copy())
                    
                    # Display camera feed
                    rgb_display = (rgbd[:, :, :3] * 255).astype(np.uint8)
                    depth_display = (rgbd[:, :, 3] * 255).astype(np.uint8)
                    depth_colored = cv2.applyColorMap(depth_display, cv2.COLORMAP_JET)
                    
                    combined = np.hstack([rgb_display, depth_colored])
                    cv2.imshow("Real Robot Camera", combined)
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print(" Episode terminated by user")
                        break
                        
                # Predict action
                action = self.predict_action(observation)
                self.episode_data['actions'].append(action.copy())
                
                # Execute action
                if not self.execute_action(action):
                    print(f" Action execution failed at step {step}")
                    break
                    
                step_time = time.time() - step_start
                self.episode_data['execution_times'].append(step_time)
                
                # Check for success (implement your success criteria)
                success = self._check_success()
                if success:
                    print(f" Episode completed successfully at step {step}!")
                    self.episode_data['success'] = True
                    break
                    
                # Print progress
                if step % 10 == 0:
                    avg_time = np.mean(self.episode_data['execution_times'])
                    print(f"Step {step}: Avg execution time: {avg_time:.3f}s")
                    
                # Small delay for safety
                time.sleep(0.1)
                
            except KeyboardInterrupt:
                print("\n Episode interrupted by user")
                break
                
            except Exception as e:
                print(f" Error at step {step}: {e}")
                break
                
        # Episode summary
        episode_summary = {
            'success': self.episode_data['success'],
            'steps': len(self.episode_data['actions']),
            'avg_execution_time': np.mean(self.episode_data['execution_times']) if self.episode_data['execution_times'] else 0,
            'total_time': sum(self.episode_data['execution_times']) if self.episode_data['execution_times'] else 0,
        }
        
        print(f"\n Episode Summary:")
        print(f"   Success: {episode_summary['success']}")
        print(f"   Steps: {episode_summary['steps']}")
        print(f"   Avg execution time: {episode_summary['avg_execution_time']:.3f}s")
        print(f"   Total time: {episode_summary['total_time']:.1f}s")
        
        return episode_summary
        
    def _check_success(self) -> bool:
        """Check if task is completed successfully"""
        # Implement your success criteria here        
        # Placeholder: always return False for now
        return False
        
    def save_episode_data(self, filename: str):
        """Save episode data for analysis"""
        save_path = Path(filename)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_data = {}
        for key, value in self.episode_data.items():
            if key == 'camera_frames':
                # Save camera frames as separate files
                frame_dir = save_path.parent / f"{save_path.stem}_frames"
                frame_dir.mkdir(exist_ok=True)
                
                for i, frame in enumerate(value):
                    frame_path = frame_dir / f"frame_{i:04d}.npz"
                    np.savez_compressed(frame_path, rgbd=frame)
                    
                serializable_data[key] = str(frame_dir)
            elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], np.ndarray):
                serializable_data[key] = [arr.tolist() for arr in value]
            else:
                serializable_data[key] = value
                
        with open(save_path, 'w') as f:
            json.dump(serializable_data, f, indent=2)
            
        print(f"Episode data saved: {save_path}")
        
    def close(self):
        """Clean shutdown"""
        try:
            cv2.destroyAllWindows()
            self.camera.close()
            self.robot.close()
            print(" Sim-to-Real interface closed")
        except Exception as e:
            print(f" Shutdown error: {e}")

def main():
    """Main deployment script"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Deploy trained model on real UR5e robot")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model directory")
    parser.add_argument("--robot-ip", type=str, default="192.168.1.102", help="Robot IP address")
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes to run")
    parser.add_argument("--max-steps", type=int, default=200, help="Max steps per episode")
    parser.add_argument("--save-data", action="store_true", help="Save episode data")
    
    args = parser.parse_args()
    
    # Safety check
    response = input(f" About to connect to robot at {args.robot_ip}. Continue? (y/N): ")
    if response.lower() != 'y':
        print("Deployment cancelled.")
        return
        
    try:
        # Initialize interface
        interface = SimToRealInterface(
            model_path=args.model,
            robot_ip=args.robot_ip,
        )
        
        # Run episodes
        results = []
        for episode in range(args.episodes):
            print(f"\n{'='*50}")
            print(f"Episode {episode + 1}/{args.episodes}")
            print('='*50)
            
            result = interface.run_episode(max_steps=args.max_steps)
            results.append(result)
            
            # Save episode data
            if args.save_data:
                timestamp = int(time.time())
                filename = f"episode_data/real_robot_episode_{episode}_{timestamp}.json"
                interface.save_episode_data(filename)
                
            # Brief pause between episodes
            if episode < args.episodes - 1:
                time.sleep(5.0)
                
        # Final summary
        success_rate = sum(r['success'] for r in results) / len(results)
        avg_steps = np.mean([r['steps'] for r in results])
        avg_time = np.mean([r['total_time'] for r in results])
        
        print(f"\n Final Results ({args.episodes} episodes):")
        print(f"   Success rate: {success_rate:.2%}")
        print(f"   Average steps: {avg_steps:.1f}")
        print(f"   Average time: {avg_time:.1f}s")
        
        # Save summary
        summary = {
            'episodes': results,
            'summary': {
                'success_rate': success_rate,
                'avg_steps': avg_steps,
                'avg_time': avg_time,
            },
            'config': {
                'model_path': args.model,
                'robot_ip': args.robot_ip,
                'episodes': args.episodes,
                'max_steps': args.max_steps,
            }
        }
        
        with open(f"real_robot_results_{int(time.time())}.json", 'w') as f:
            json.dump(summary, f, indent=2)
            
    except Exception as e:
        print(f" Deployment failed: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        if 'interface' in locals():
            interface.close()

if __name__ == "__main__":
    main()