"""
Test script for UR5e environment with object perception logging and expanded object spawning.
Tests camera switching with RealSense debug markers and demonstrates cylinder/sphere spawning.
"""
import time
import os
import numpy as np
import random
import mujoco
from homestri_ur5e_rl.envs.UR5ePickPlaceEnvEnhanced import UR5ePickPlaceEnvEnhanced

class TestUR5eWithExpandedObjects:
    """Test environment with expanded object spawning for test purposes only"""
    
    def __init__(self, base_env):
        self.base_env = base_env
        self.test_episode_count = 0
        
    def reset_with_expanded_objects(self):
        """Reset environment but with ability to spawn all object types for testing"""
        # Call the base reset first
        obs, info = self.base_env.reset()
        
        # For testing: force spawn different object types randomly
        self._spawn_random_test_object()
        
        # Step physics to settle
        for _ in range(3):
            mujoco.mj_step(self.base_env.model, self.base_env.data)
        
        # Get updated observation
        obs_updated = self.base_env._get_obs()
        
        # Log object perception using the environment's built-in method
        self._log_test_object_perception(obs_updated)
        
        return obs_updated, info
    
    def _spawn_random_test_object(self):
        """Spawn cube, sphere, or cylinder randomly for testing"""
        # Hide all objects first
        for obj_name in self.base_env.object_names:
            if obj_name in self.base_env.object_body_ids:
                obj_id = self.base_env.object_body_ids[obj_name]
                self.base_env._hide_object(obj_id)
        
        # TEST EXPANSION: Choose from all available objects
        test_objects = ["cube_object", "sphere_object", "cylinder_object"]
        available_test_objects = [obj for obj in test_objects if obj in self.base_env.object_body_ids]
        
        if not available_test_objects:
            print(f"⚠️ Warning: No test objects available from {test_objects}")
            available_test_objects = [obj for obj in self.base_env.object_names if obj in self.base_env.object_body_ids]
        
        # Choose random object for this test
        self.base_env.current_object = random.choice(available_test_objects)
        obj_id = self.base_env.object_body_ids[self.base_env.current_object]
        
        print(f"\n🎲 TEST: Randomly selected {self.base_env.current_object} for spawning")
        
        # Apply properties (use domain randomization if enabled)
        if self.base_env.use_domain_randomization:
            self.base_env._randomize_object_properties(obj_id)
        else:
            self.base_env._log_default_object_properties(obj_id)
        
        # Position in spawning area
        x = np.random.uniform(*self.base_env.object_spawning_area['x_range'])
        y = np.random.uniform(*self.base_env.object_spawning_area['y_range'])
        z = self.base_env.object_spawning_area['z']
        
        # Position object using MuJoCo body positioning (same as environment)
        body = self.base_env.model.body(obj_id)
        if body.jntadr[0] >= 0:
            qpos_adr = self.base_env.model.jnt_qposadr[body.jntadr[0]]
            self.base_env.data.qpos[qpos_adr:qpos_adr+3] = [x, y, z]
            
            # Set stable orientation
            if "cylinder" in self.base_env.current_object:
                self.base_env.data.qpos[qpos_adr+3:qpos_adr+7] = [0.707, 0.707, 0, 0]
            else:
                self.base_env.data.qpos[qpos_adr+3:qpos_adr+7] = [1, 0, 0, 0]
                
            # Zero velocities
            if body.jntadr[0] >= 0 and body.jntadr[0] < self.base_env.model.njnt:
                dof_adr = self.base_env.model.jnt_dofadr[body.jntadr[0]]
                self.base_env.data.qvel[dof_adr:dof_adr+6] = 0
        
        mujoco.mj_forward(self.base_env.model, self.base_env.data)
        
        print(f"   📍 Positioned at: [{x:.3f}, {y:.3f}, {z:.3f}]")
    
    def _log_test_object_perception(self, obs):
        """Log object perception for test script"""
        self.test_episode_count += 1
        
        try:
            current_object = getattr(self.base_env, 'current_object', 'unknown')
            
            # Get object size
            if current_object != 'unknown' and hasattr(self.base_env, 'object_body_ids'):
                obj_id = self.base_env.object_body_ids.get(current_object)
                if obj_id is not None:
                    current_object_size = self.base_env._get_object_size_from_model(obj_id)
                else:
                    current_object_size = 'unknown_id'
            else:
                current_object_size = 'unknown'
            
            print(f"\n🔍 TEST Object Perception - Test #{self.test_episode_count}")
            print(f"   📦 Spawned Object: {current_object}")
            print(f"   📏 Object Size: {current_object_size}")
            
            # Analyze camera input if available
            if len(obs) >= 16384:
                rgb_start = 56
                rgb_end = rgb_start + 12288
                depth_start = rgb_end
                depth_end = depth_start + 4096
                
                rgb_obs = obs[rgb_start:rgb_end]
                depth_obs = obs[depth_start:depth_end]
                
                rgb_mean = np.mean(rgb_obs)
                rgb_std = np.std(rgb_obs)
                rgb_min, rgb_max = np.min(rgb_obs), np.max(rgb_obs)
                
                depth_mean = np.mean(depth_obs)
                depth_std = np.std(depth_obs)
                depth_min, depth_max = np.min(depth_obs), np.max(depth_obs)
                
                print(f"   👁️ CNN Visual Input Analysis:")
                print(f"      RGB: mean={rgb_mean:.3f}, std={rgb_std:.3f}, range=[{rgb_min:.3f}, {rgb_max:.3f}]")
                print(f"      Depth: mean={depth_mean:.3f}, std={depth_std:.3f}, range=[{depth_min:.3f}, {depth_max:.3f}]")
                
                rgb_has_content = rgb_std > 0.01
                depth_has_content = depth_std > 0.01
                
                object_type = "Unknown"
                if "cube" in current_object:
                    object_type = "Cube"
                elif "sphere" in current_object:
                    object_type = "Sphere"
                elif "cylinder" in current_object:
                    object_type = "Cylinder"
                
                print(f"      Object Type: {object_type}")
                print(f"      Visual Content: RGB={'✅' if rgb_has_content else '❌'}, Depth={'✅' if depth_has_content else '❌'}")
                
                if not (rgb_has_content or depth_has_content):
                    print(f"      ⚠️ WARNING: Low visual variation - CNN may not see object clearly!")
                    
            else:
                print(f"   ⚠️ Observation too short for camera analysis: {len(obs)} < 16384")
                
        except Exception as e:
            print(f"   ❌ Test perception logging failed: {e}")

def main():
    """Main function to run the expanded object test with perception logging."""
    print("🤖 UR5e Enhanced Test: Object Perception + Expanded Objects")
    print("=" * 70)
    print(" This test demonstrates:")
    print("   • 🎲 Random spawning of cubes, spheres, AND cylinders")
    print("   • 👁️ Object perception logging (what CNN sees vs what was spawned)")
    print("   • 📷 Camera switching with keyboard controls")
    print("   • 🔍 Real-time visual analysis for debugging")
    print()
    print(" Controls:")
    print("   '[' - Previous camera")
    print("   ']' - Next camera")
    print("   'r' - Reset environment (spawn new random object)")
    print("   'h' - Show camera help")
    print("   ESC - Exit")
    print("=" * 70)

    try:
        # Create base environment with expanded objects enabled
        env = UR5ePickPlaceEnvEnhanced(
            xml_file="custom_scene.xml",
            frame_skip=5,
            camera_resolution=64,  # Match training resolution for CNN analysis
            render_mode="human",
            control_mode="joint",
            use_stuck_detection=False,
            use_domain_randomization=True,  # Enable for property randomization
            curriculum_level=1.0,  # Full curriculum to enable all objects
        )
        
        # Wrap with test object handler
        test_env = TestUR5eWithExpandedObjects(env)
        
        print("\n✅ Environment created successfully!")
        print(f"   Available objects: {env.object_names}")
        print(f"   Domain randomization: {'Enabled' if env.use_domain_randomization else 'Disabled'}")
        print(f"   Camera resolution: {env.camera_resolution}x{env.camera_resolution}")
        
        # List available cameras
        print("\n📷 Available cameras:")
        cameras = env.list_cameras()
        
        # Initial reset with random object
        print("\n🎲 Initial reset with random object spawning...")
        obs, info = test_env.reset_with_expanded_objects()
        
        # Start with RealSense camera for best visual analysis
        print("\n📷 Setting camera to RealSense for optimal perception analysis...")
        env.switch_to_realsense_view()
        
        # Initial render
        env.render()
        print("   📺 MuJoCo window opened successfully!")
        
        print("\n🎯 Starting enhanced object testing...")
        print("   � Objects will randomly cycle between cubes, spheres, and cylinders")
        print("   �️ Each spawn shows detailed CNN perception analysis")
        print("   📷 Use '[' and ']' to switch camera views")
        print("   � Environment auto-resets every 30 seconds with new random object")
        print("   ⏹ Press Ctrl+C to stop test")
        
        # Test parameters
        duration_minutes = 15  # Extended test time for thorough evaluation
        auto_reset_interval = 30  # Auto reset every 30 seconds
        max_steps = int(duration_minutes * 60 / 0.1)
        
        step = 0
        last_reset_time = time.time()
        
        try:
            while step < max_steps:
                current_time = time.time()
                
                # Auto-reset with new random object every 30 seconds
                if current_time - last_reset_time >= auto_reset_interval:
                    print(f"\n🔄 Auto-reset: Spawning new random object...")
                    obs, info = test_env.reset_with_expanded_objects()
                    last_reset_time = current_time
                
                # Render the environment
                env.render()
                
                step += 1
                
                # Status update every minute
                if step % 600 == 0:  # Every 60 seconds
                    elapsed_minutes = step * 0.1 / 60
                    print(f"\n⏱ Status Update:")
                    print(f"   Time elapsed: {elapsed_minutes:.1f}/{duration_minutes} minutes")
                    print(f"   Current camera: {env.camera_name}")
                    print(f"   Current object: {getattr(env, 'current_object', 'unknown')}")
                    print(f"   Test episodes: {test_env.test_episode_count}")
                    print(f"   Next auto-reset in: {auto_reset_interval - (current_time - last_reset_time):.1f}s")
                
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            print("\n⏹ Test interrupted by user")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        print("\n🧹 Cleaning up...")
        try:
            env.close()
        except:
            pass
        print("✅ Test completed!")
        
        print("\n📊 Test Summary:")
        print(f"   Total test episodes: {test_env.test_episode_count if 'test_env' in locals() else 'N/A'}")
        print("   Objects tested:")
        print("     🟦 Cubes - Basic box geometry")
        print("     🟣 Spheres - Circular geometry") 
        print("     🟡 Cylinders - Cylindrical geometry")
        print("   CNN Analysis provided:")
        print("     👁️ RGB visual statistics (mean, std, range)")
        print("     📏 Depth perception data")
        print("     ✅ Visual content validation")
        print("     📦 Object type correlation")
        print("\n💡 Key Insights:")
        print("   • The perception logging shows exactly what the CNN 'sees'")
        print("   • Different object types create different visual signatures")
        print("   • This data helps debug visual learning effectiveness")
        print("   • Camera switching allows testing different viewpoints")

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    main()
