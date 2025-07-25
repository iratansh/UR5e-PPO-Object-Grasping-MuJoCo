#!/bin/bash

# Ubuntu MuJoCo Training Script
# Fixes rendering issues for Ubuntu by setting proper environment variables

echo "🚀 Setting up Ubuntu MuJoCo Environment..."

# Force EGL rendering for headless operation
export MUJOCO_GL="egl"

# Disable X11 dependencies 
export DISPLAY=""

# Force headless mode
export MUJOCO_HEADLESS="1"

# Additional stability fixes
export MESA_GL_VERSION_OVERRIDE="3.3"
export MESA_GLSL_VERSION_OVERRIDE="330"

# Prevent OpenGL errors
export MUJOCO_GL_DISABLE_EXTENSIONS="1"

echo "✅ Environment variables set for Ubuntu:"
echo "   MUJOCO_GL=egl"
echo "   MUJOCO_HEADLESS=1" 
echo "   DISPLAY=(disabled)"

# Activate conda environment
echo "🔧 Activating conda environment..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate homestri-ur5e-rl

echo "🎯 Starting training..."
cd homestri_ur5e_rl/training

# Run the training with proper environment
python3 training_script_integrated.py --config config_rtx4060_optimized.yaml "$@"
