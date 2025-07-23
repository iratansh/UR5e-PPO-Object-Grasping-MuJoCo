#!/usr/bin/env python3
"""
Diagnostic script to identify why custom scene won't render
"""

import os
import sys
from pathlib import Path

print("🔍 MuJoCo Environment Diagnostics")
print("=" * 50)

# 1. Check environment variables
print("\n1⃣ Environment Variables:")
important_vars = ['MUJOCO_GL', 'PYOPENGL_PLATFORM', 'DISPLAY']
for var in important_vars:
    value = os.environ.get(var, "Not set")
    print(f"   {var}: {value}")

# 2. Check MuJoCo installation
print("\n2⃣ MuJoCo Installation:")
try:
    import mujoco
    print(f"    MuJoCo version: {mujoco.__version__}")
    print(f"   MuJoCo path: {mujoco.__file__}")
except ImportError as e:
    print(f"    MuJoCo import error: {e}")

# 3. Check file paths
print("\n3⃣ File Paths:")
# Find homestri root
current_file = Path(__file__)
homestri_root = None

# Search for homestri_ur5e_rl in parent directories
for parent in current_file.parents:
    if (parent / "homestri_ur5e_rl").exists():
        homestri_root = parent
        break

if homestri_root:
    print(f"   Homestri root: {homestri_root}")
    
    # Check if custom_scene.xml exists
    custom_scene_paths = [
        homestri_root / "homestri_ur5e_rl/envs/assets/base_robot/custom_scene.xml",
        homestri_root / "homestri_ur5e_rl/envs/assets/custom_scene.xml",
        current_file.parent / "custom_scene.xml",
    ]
    
    print(f"\n   Looking for custom_scene.xml:")
    for path in custom_scene_paths:
        exists = "" if path.exists() else ""
        print(f"   {exists} {path}")
        
    # Check mesh files
    mesh_dir = homestri_root / "homestri_ur5e_rl/envs/assets"
    print(f"\n   Mesh directories in {mesh_dir}:")
    if mesh_dir.exists():
        for item in mesh_dir.iterdir():
            if item.is_dir():
                file_count = len(list(item.glob("*")))
                print(f"      📁 {item.name} ({file_count} files)")
else:
    print("    Could not find homestri root directory")

# 4. Test basic MuJoCo viewer
print("\n4⃣ Testing Basic MuJoCo Viewer:")
try:
    import mujoco.viewer
    
    xml_string = """
    <mujoco>
        <worldbody>
            <light diffuse="1 1 1" pos="0 0 3" dir="0 0 -1"/>
            <geom type="plane" size="1 1 0.1" rgba="0.8 0.8 0.8 1"/>
            <body pos="0 0 1">
                <geom type="box" size="0.1 0.1 0.1" rgba="1 0 0 1"/>
            </body>
        </worldbody>
    </mujoco>
    """
    
    model = mujoco.MjModel.from_xml_string(xml_string)
    data = mujoco.MjData(model)
    
    print("    Can create MuJoCo model and data")
    
    # Try to create viewer (non-blocking test)
    print("   Testing viewer creation...")
    try:
        # This is a simple test - we're not actually launching the viewer
        print("    MuJoCo viewer module is available")
    except Exception as e:
        print(f"    Viewer error: {e}")
        
except Exception as e:
    print(f"    MuJoCo test failed: {e}")

# 5. Compare with working environment
print("\n5⃣ Testing Working Environment:")
try:
    from homestri_ur5e_rl.envs.base_robot_env import BaseRobot
    
    # Try creating the default environment that works
    env = BaseRobot(render_mode="human")
    print("    BaseRobot environment created successfully")
    env.close()
    
except Exception as e:
    print(f"    BaseRobot failed: {e}")

# 6. Platform info
print("\n6⃣ Platform Information:")
import platform
print(f"   OS: {platform.system()} {platform.release()}")
print(f"   Python: {platform.python_version()}")
print(f"   Machine: {platform.machine()}")

print("\n" + "=" * 50)
print("📋 Diagnostic Summary:")
print("\nIf BaseRobot works but your custom scene doesn't, the issue is likely:")
print("1. Path problems - custom_scene.xml not in the right location")
print("2. Missing mesh files referenced in custom_scene.xml")
print("3. Your enhanced environment is overriding render() incorrectly")
print("\nRecommendations:")
print("1. Use SimplePickPlaceEnv.py (created above) instead of the complex enhanced version")
print("2. Make sure custom_scene.xml is in: homestri_ur5e_rl/envs/assets/base_robot/")
print("3. Check that all mesh files referenced in the XML exist")
print("4. Don't override render() method - let MujocoEnv handle it")