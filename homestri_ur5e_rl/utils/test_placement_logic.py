"""
Comprehensive placement logic testing script for UR5e environment.
Tests placement criteria, success detection, and visualizes target zones.

Updated with empirically calibrated height thresholds:
- Above target: 2.4cm max (physics-aware strict boundary)
- Below target: 2.25cm max (gravity settling compensated)
"""

import time
import os
import numpy as np
import mujoco
from homestri_ur5e_rl.envs.UR5ePickPlaceEnvEnhanced import UR5ePickPlaceEnvEnhanced

class PlacementTester:
    """Test placement logic and visualization"""
    
    def __init__(self):
        self.env = None
        self.test_results = []
        
    def create_environment(self):
        """Create environment for placement testing"""
        print("Creating placement test environment...")
        
        self.env = UR5ePickPlaceEnvEnhanced(
            xml_file="custom_scene.xml",
            render_mode="human",
            control_mode="joint",
            use_stuck_detection=False,
            use_domain_randomization=False,  # Disable for consistent testing
            curriculum_level=1.0,  # Full curriculum
            frame_skip=5
        )
        
        print("Environment created successfully!")
        return self.env
        
    def analyze_target_markers(self):
        """Analyze the target markers in the scene"""
        print("\nAnalyzing Target Markers in Scene...")
        
        # Reset to get initial state
        obs, info = self.env.reset()
        
        target_pos = self.env.target_position
        print(f"   Main target position: [{target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f}]")
        
        # Find target marker geometries using mujoco name lookup
        target_markers = []
        
        # Look for known target marker names
        target_names = ['target1', 'target2', 'target3', 'target4', 'target_center', 'success_boundary']
        
        for target_name in target_names:
            try:
                geom_id = mujoco.mj_name2id(self.env.model, mujoco.mjtObj.mjOBJ_GEOM, target_name)
                if geom_id >= 0:
                    geom_pos = self.env.model.geom_pos[geom_id]
                    marker_info = {
                        'name': target_name,
                        'position': geom_pos.copy(),
                        'id': geom_id
                    }
                    target_markers.append(marker_info)
            except:
                # Marker doesn't exist, skip
                continue
                
        print(f"   Found {len(target_markers)} target markers:")
        for marker in target_markers:
            pos = marker['position']
            print(f"     • {marker['name']}: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
            
        if len(target_markers) == 0:
            print("   ⚠️ No target markers found - visual markers may not be available")
            
        return target_markers
        
    def test_placement_scenarios(self):
        """Test various placement scenarios systematically"""
        print("\n🧪 Testing Placement Scenarios...")
        
        # Get environment state
        obs, info = self.env.reset()
        target_pos = self.env.target_position
        
        # Define test scenarios
        # NOTE: Height thresholds empirically calibrated through systematic testing:
        # - Above target: max 2.4cm (stricter due to physics lifting)
        # - Below target: max 2.25cm (compensated for gravity settling)
        # Test positions account for ~1.5-2.0cm physics settling toward target
        scenarios = [
            # Perfect placements
            {"name": "Perfect Center", "offset": np.array([0.0, 0.0, 0.0]), "expected": True},
            {"name": "Slightly Off-Center", "offset": np.array([0.01, 0.01, 0.0]), "expected": True},
            
            # Boundary tests (empirically calibrated asymmetric tolerance)
            {"name": "Boundary X+ (4.9cm)", "offset": np.array([0.049, 0.0, 0.0]), "expected": True},
            {"name": "Boundary X- (4.9cm)", "offset": np.array([-0.049, 0.0, 0.0]), "expected": True},
            {"name": "Boundary Y+ (4.9cm)", "offset": np.array([0.0, 0.049, 0.0]), "expected": True},
            {"name": "Boundary Y- (4.9cm)", "offset": np.array([0.0, -0.049, 0.0]), "expected": True},
            {"name": "Boundary Z+ (2.3cm)", "offset": np.array([0.0, 0.0, 0.023]), "expected": True},
            {"name": "Boundary Z+ (3.5cm)", "offset": np.array([0.0, 0.0, 0.035]), "expected": False},
            {"name": "Boundary Z- (2.2cm)", "offset": np.array([0.0, 0.0, -0.022]), "expected": True},
            {"name": "Boundary Z- (3.5cm)", "offset": np.array([0.0, 0.0, -0.035]), "expected": False},
            {"name": "Boundary Z- (4.0cm)", "offset": np.array([0.0, 0.0, -0.040]), "expected": False},
            {"name": "Boundary Z- (4.5cm)", "offset": np.array([0.0, 0.0, -0.045]), "expected": False},
            {"name": "Boundary Z- (4.6cm)", "offset": np.array([0.0, 0.0, -0.046]), "expected": False},
            
            # Edge of boundary  
            {"name": "Boundary Edge (5.2cm)", "offset": np.array([0.052, 0.0, 0.0]), "expected": False},
            {"name": "Just Over Edge (5.3cm)", "offset": np.array([0.053, 0.0, 0.0]), "expected": False},
            {"name": "Boundary Diagonal", "offset": np.array([0.035, 0.035, 0.0]), "expected": True},
            
            # Failure cases
            {"name": "Too Far X (6cm)", "offset": np.array([0.06, 0.0, 0.0]), "expected": False},
            {"name": "Too Far Y (6cm)", "offset": np.array([0.0, 0.06, 0.0]), "expected": False},
            {"name": "Too High (6cm)", "offset": np.array([0.0, 0.0, 0.06]), "expected": False},
            {"name": "Too Low (6cm)", "offset": np.array([0.0, 0.0, -0.06]), "expected": False},
            {"name": "Far Away", "offset": np.array([0.15, 0.15, 0.0]), "expected": False},
            
            # Corner cases
            {"name": "Very Small Offset", "offset": np.array([0.001, 0.001, 0.001]), "expected": True},
            {"name": "Diagonal Far", "offset": np.array([0.04, 0.04, 0.04]), "expected": False},  # sqrt(3*0.04²) > 0.05
        ]
        
        print(f"   Running {len(scenarios)} placement scenarios...")
        print(f"   Success criteria: Distance < 5.2cm AND Height: +2.4cm/-2.25cm from target (empirically calibrated)")
        print("   Target position for all tests: [{:.3f}, {:.3f}, {:.3f}]".format(target_pos[0], target_pos[1], target_pos[2]))
        print("   " + "="*70)
        
        for i, scenario in enumerate(scenarios, 1):
            success = self._test_single_scenario(scenario, target_pos, i)
            self.test_results.append({
                'scenario': scenario['name'],
                'expected': scenario['expected'],
                'actual': success,
                'passed': success == scenario['expected']
            })
            
        return self.test_results
        
    def _test_single_scenario(self, scenario, target_pos, test_num):
        """Test a single placement scenario"""
        test_pos = target_pos + scenario['offset']
        
        print(f"\n   Test {test_num:2d}: {scenario['name']}")
        print(f"          Offset: [{scenario['offset'][0]:+.3f}, {scenario['offset'][1]:+.3f}, {scenario['offset'][2]:+.3f}]")
        print(f"          Test pos: [{test_pos[0]:.3f}, {test_pos[1]:.3f}, {test_pos[2]:.3f}]")
        
        # Reset environment
        obs, info = self.env.reset()
        
        # IMPORTANT: Get the target position AFTER reset, as it changes
        actual_target = self.env.target_position
        actual_test_pos = actual_target + scenario['offset']
        
        print(f"          Target (after reset): [{actual_target[0]:.3f}, {actual_target[1]:.3f}, {actual_target[2]:.3f}]")
        print(f"          Adjusted test pos: [{actual_test_pos[0]:.3f}, {actual_test_pos[1]:.3f}, {actual_test_pos[2]:.3f}]")
        
        # Position object at test location
        obj_id = self.env.object_body_ids[self.env.current_object]
        self._position_object_at(obj_id, actual_test_pos)
        
        # Let physics settle and check stability
        print(f"          Settling physics...")
        for i in range(50):  # Increased settling time
            mujoco.mj_step(self.env.model, self.env.data)
            if i % 10 == 0:  # Check every 10 steps
                physics_ok = self.env._check_physics_stability()
                if not physics_ok:
                    print(f"          Physics unstable at step {i}")
                    
        # Final physics check
        final_physics = self.env._check_physics_stability()
        
        # Check success and debug information
        success = self.env._check_success()
        distance = np.linalg.norm(actual_test_pos - actual_target)
        height_diff = abs(actual_test_pos[2] - actual_target[2])
        
        # Get actual object position after physics settling
        actual_pos = self.env.data.body(obj_id).xpos.copy()
        actual_distance = np.linalg.norm(actual_pos - actual_target)
        actual_height_diff = abs(actual_pos[2] - actual_target[2])
        
        # Debug information
        physics_stable = self.env.physics_stable
        try:
            obj_vel = np.linalg.norm(self.env.data.body(obj_id).cvel[:3])
        except:
            obj_vel = -1
            
        print(f"          Distance: {distance:.4f}m (actual: {actual_distance:.4f}m)")
        print(f"          Height diff: {height_diff:.4f}m (actual: {actual_height_diff:.4f}m)")
        print(f"          Physics stable: {physics_stable}")
        print(f"          Object velocity: {obj_vel:.4f} m/s")
        print(f"          Success criteria: dist<0.052 AND height: above +2.4cm OR below -2.25cm AND stable AND vel<0.5")
        print(f"          Expected: {'✅ SUCCESS' if scenario['expected'] else '❌ FAIL'}")
        print(f"          Actual: {'✅ SUCCESS' if success else '❌ FAIL'}")
        print(f"          Result: {'✅ PASS' if success == scenario['expected'] else '❌ FAIL TEST'}")
        
        # Render for visualization
        self.env.render()
        
        # Show visual markers
        self._highlight_test_position(actual_test_pos, actual_target, success, scenario['expected'])
        
        # Wait for user input to continue
        input(f"          Press Enter to continue to next test...")
        
        return success
        
    def _position_object_at(self, obj_id, position):
        """Position object at specified location"""
        body = self.env.model.body(obj_id)
        if body.jntadr[0] >= 0:
            qpos_adr = self.env.model.jnt_qposadr[body.jntadr[0]]
            self.env.data.qpos[qpos_adr:qpos_adr+3] = position
            self.env.data.qpos[qpos_adr+3:qpos_adr+7] = [1, 0, 0, 0]  # Stable orientation
            
            # Zero velocities
            if body.jntadr[0] < self.env.model.njnt:
                dof_adr = self.env.model.jnt_dofadr[body.jntadr[0]]
                self.env.data.qvel[dof_adr:dof_adr+6] = 0
                
        mujoco.mj_forward(self.env.model, self.env.data)
        
    def _highlight_test_position(self, test_pos, target_pos, actual_success, expected_success):
        """Provide visual feedback about test position"""
        distance = np.linalg.norm(test_pos - target_pos)
        
        if actual_success == expected_success:
            status = "✅ TEST PASSED"
        else:
            status = "❌ TEST FAILED"
            
        print(f"          Visual: Object at test position, distance {distance:.3f}m")
        print(f"          Target markers (green cylinders) show placement zone")
        print(f"           {status}")
        
    def test_object_types(self):
        """Test placement logic with different object types"""
        print("\n🎲 Testing Placement with Different Object Types...")
        
        object_types = ["cube_object", "sphere_object", "cylinder_object"]
        target_pos = None
        
        for obj_type in object_types:
            if obj_type not in self.env.object_body_ids:
                print(f"    Skipping {obj_type} - not available in scene")
                continue
                
            print(f"\n   Testing with {obj_type}...")
            
            # Reset and set object type
            obs, info = self.env.reset()
            self.env.current_object = obj_type
            target_pos = self.env.target_position
            
            # Test perfect placement
            obj_id = self.env.object_body_ids[obj_type]
            self._position_object_at(obj_id, target_pos)
            
            # Let physics settle
            for _ in range(20):
                mujoco.mj_step(self.env.model, self.env.data)
                
            success = self.env._check_success()
            actual_pos = self.env.data.body(obj_id).xpos.copy()
            distance = np.linalg.norm(actual_pos - target_pos)
            
            print(f"      Position: [{actual_pos[0]:.3f}, {actual_pos[1]:.3f}, {actual_pos[2]:.3f}]")
            print(f"      Distance: {distance:.4f}m")
            print(f"      Result: {'✅ SUCCESS' if success else '❌ FAIL'}")
            
            self.env.render()
            input(f"      Press Enter to test next object type...")
            
    def analyze_results(self):
        """Analyze and display test results"""
        if not self.test_results:
            print("❌ No test results to analyze")
            return
            
        print("\n📊 Placement Test Results Analysis")
        print("="*70)
        
        passed = sum(1 for r in self.test_results if r['passed'])
        total = len(self.test_results)
        
        print(f"Overall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
        print()
        
        # Group by result type
        correct_success = [r for r in self.test_results if r['expected'] and r['actual'] and r['passed']]
        correct_failure = [r for r in self.test_results if not r['expected'] and not r['actual'] and r['passed']]
        false_positive = [r for r in self.test_results if not r['expected'] and r['actual'] and not r['passed']]
        false_negative = [r for r in self.test_results if r['expected'] and not r['actual'] and not r['passed']]
        
        print(f"✅ Correct Success: {len(correct_success)} tests")
        print(f"✅ Correct Failure: {len(correct_failure)} tests")
        print(f"❌ False Positive: {len(false_positive)} tests (should fail but passed)")
        print(f"❌ False Negative: {len(false_negative)} tests (should pass but failed)")
        
        if false_positive:
            print("\n⚠️ FALSE POSITIVES (placement logic too lenient):")
            for r in false_positive:
                print(f"   • {r['scenario']}")
                
        if false_negative:
            print("\n⚠️ FALSE NEGATIVES (placement logic too strict):")
            for r in false_negative:
                print(f"   • {r['scenario']}")
            
        return {
            'total': total,
            'passed': passed,
            'false_positives': len(false_positive),
            'false_negatives': len(false_negative)
        }

def main():
    """Main test function"""
    print("🤖 UR5e Placement Logic Comprehensive Testing")
    print("="*70)
    print("This script tests placement detection with empirically calibrated thresholds!")
    print("Height boundaries: +2.4cm above target, -2.25cm below target")
    print("Features:")
    print("  • Visual target markers (green cylinders)")
    print("  • Systematic boundary testing")
    print("  • Success criteria validation") 
    print("  • Multi-object testing")
    print("  • Interactive testing mode")
    print("="*70)
    
    # Change to correct directory
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    tester = PlacementTester()
    
    try:
        # Create environment
        tester.create_environment()
        
        # Analyze target markers
        markers = tester.analyze_target_markers()
        
        # Test systematic scenarios
        results = tester.test_placement_scenarios()
        
        # Test different object types
        tester.test_object_types()
        
        # Analyze results
        analysis = tester.analyze_results()
        
        # Interactive testing
        print("\n🎮 Would you like to run interactive testing? (y/n)")
        if input().lower() == 'y':
            tester.run_interactive_test()
        
        print("\nPlacement testing completed!")
        print(f" Final Score: {analysis['passed']}/{analysis['total']} tests passed")
        
        if analysis['passed'] == analysis['total']:
            print(" Placement logic is working correctly.")
        elif analysis['false_positives'] == 0 and analysis['false_negatives'] == 0:
            print(" All logic tests passed - any failures were expected.")
        else:
            print(" Some logic issues detected - check the analysis above.")
            
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        if tester.env:
            print("\nCleaning up...")
            tester.env.close()
            print(" Test completed!")

if __name__ == "__main__":
    main()
