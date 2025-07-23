#!/bin/bash

# HP Victus RTX 4060 Setup Script for Homestri UR5e RL
# This script sets up the complete environment for CUDA-accelerated training

set -e  # Exit on any error

echo "🚀 Setting up Homestri UR5e RL on HP Victus RTX 4060..."
echo "========================================================"

# Check if NVIDIA GPU is available
echo "🔍 Checking NVIDIA GPU..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
    echo "✅ NVIDIA GPU detected"
else
    echo "❌ NVIDIA GPU not detected. Please install NVIDIA drivers first."
    exit 1
fi

# Check CUDA version
echo ""
echo "🔍 Checking CUDA version..."
if command -v nvcc &> /dev/null; then
    nvcc --version
    echo "✅ CUDA toolkit detected"
else
    echo "⚠️ CUDA toolkit not found. Will install PyTorch with bundled CUDA."
fi

# Create conda environment
echo ""
echo "🐍 Creating conda environment..."
if conda env list | grep -q "homestri-ur5e-rl"; then
    echo "⚠️ Environment 'homestri-ur5e-rl' already exists."
    read -p "Do you want to remove it and recreate? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        conda env remove -n homestri-ur5e-rl -y
    else
        echo "❌ Setup cancelled."
        exit 1
    fi
fi

echo "Creating new environment from environment.yml..."
conda env create -f environment.yml

echo ""
echo "🔧 Activating environment and installing additional packages..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate homestri-ur5e-rl

# Verify PyTorch CUDA installation
echo ""
echo "🔍 Verifying PyTorch CUDA installation..."
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
    print(f'GPU name: {torch.cuda.get_device_name(0)}')
else:
    print('❌ CUDA not available in PyTorch')
    exit(1)
"

# Install the package in development mode
echo ""
echo "📦 Installing homestri-ur5e-rl in development mode..."
pip install -e .

# Test MuJoCo installation
echo ""
echo "🔍 Testing MuJoCo installation..."
python -c "
import mujoco
print(f'MuJoCo version: {mujoco.__version__}')
print('✅ MuJoCo installation successful')
"

# Test Gymnasium installation
echo ""
echo "🔍 Testing Gymnasium installation..."
python -c "
import gymnasium as gym
print(f'Gymnasium version: {gym.__version__}')
print('✅ Gymnasium installation successful')
"

# Test Stable-Baselines3 installation
echo ""
echo "🔍 Testing Stable-Baselines3 installation..."
python -c "
import stable_baselines3 as sb3
print(f'Stable-Baselines3 version: {sb3.__version__}')
print('✅ Stable-Baselines3 installation successful')
"

# Create directory structure
echo ""
echo "📁 Creating directory structure..."
mkdir -p experiments
mkdir -p logs
mkdir -p models
mkdir -p tensorboard
mkdir -p checkpoints

echo ""
echo "🎯 Setup Summary:"
echo "================"
echo "✅ Conda environment: homestri-ur5e-rl"
echo "✅ PyTorch with CUDA support"
echo "✅ MuJoCo physics simulation"
echo "✅ Stable-Baselines3 RL library"
echo "✅ All dependencies installed"
echo "✅ Directory structure created"
echo ""
echo "🚀 Ready for training!"
echo ""
echo "To activate the environment:"
echo "  conda activate homestri-ur5e-rl"
echo ""
echo "To start training:"
echo "  cd homestri_ur5e_rl/training"
echo "  python training_script_integrated.py"
echo ""
echo "To monitor training:"
echo "  tensorboard --logdir experiments/[experiment_name]/tensorboard"
echo ""
echo "🎊 Setup complete! Happy training on your RTX 4060!"
