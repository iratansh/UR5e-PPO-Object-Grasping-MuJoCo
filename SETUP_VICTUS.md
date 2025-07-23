# HP Victus RTX 4060 Setup Guide for Homestri UR5e RL

This guide will help you set up the complete training environment on your HP Victus laptop with RTX 4060 8GB GPU.

## 🚀 Quick Setup

### Prerequisites
1. **NVIDIA Drivers**: Ensure latest RTX 4060 drivers are installed
2. **Anaconda/Miniconda**: Download from [anaconda.com](https://www.anaconda.com/products/distribution)
3. **Git**: For cloning the repository

### One-Click Setup

**For Linux/MacOS:**
```bash
./setup_victus.sh
```

**For Windows:**
```cmd
setup_victus.bat
```

### Manual Setup (if scripts don't work)

1. **Clone the repository:**
```bash
git clone https://github.com/iratansh/UR5e-PPO-Object-Grasping.git
cd UR5e-PPO-Object-Grasping
```

2. **Create conda environment:**
```bash
conda env create -f environment.yml
conda activate homestri-ur5e-rl
```

3. **Install in development mode:**
```bash
pip install -e .
```

4. **Verify CUDA installation:**
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU name: {torch.cuda.get_device_name(0)}")
```

## 🎯 Environment Details

- **Python Version**: 3.9.23
- **PyTorch**: 2.0+ with CUDA 11.8 support
- **GPU**: Optimized for RTX 4060 8GB
- **Memory**: 16GB RAM recommended
- **Storage**: ~5GB for environment + datasets

## 📦 Key Packages

### Core ML/RL Stack
- `torch>=2.0.0` (CUDA-enabled)
- `stable-baselines3>=2.0.0`
- `gymnasium>=0.29.0`
- `tensorboard` for monitoring

### Robotics & Physics
- `mujoco>=2.3.0` for physics simulation
- `robotics-toolbox-python` for kinematics
- Custom `homestri-ur5e-rl` framework

### Computer Vision
- `opencv-python` for image processing
- `pillow` for image manipulation
- `imageio` for video recording

## 🏃‍♂️ Getting Started

1. **Activate environment:**
```bash
conda activate homestri-ur5e-rl
```

2. **Start training:**
```bash
cd homestri_ur5e_rl/training
python training_script_integrated.py
```

3. **Monitor progress:**
```bash
tensorboard --logdir experiments/[experiment_name]/tensorboard
```

## 🔧 Configuration

### CUDA Settings
The environment is pre-configured for RTX 4060:
- CUDA 11.8 compatibility
- Optimized batch sizes for 8GB VRAM
- Memory-efficient training settings

### Training Parameters
Located in `training_script_integrated.py`:
- **Batch Size**: 64 (optimized for RTX 4060)
- **Learning Rate**: 0.0003
- **Target FPS**: 100+ on RTX 4060

## 📊 Expected Performance

### Training Speed
- **RTX 4060 8GB**: ~116 FPS
- **Phase 1 (5M steps)**: ~12 hours
- **Full curriculum (27M steps)**: ~65 hours

### Memory Usage
- **GPU VRAM**: ~6GB during training
- **System RAM**: ~8GB during training
- **Storage**: ~2GB per experiment

## 🐛 Troubleshooting

### CUDA Issues
```bash
# Check NVIDIA driver
nvidia-smi

# Check CUDA toolkit
nvcc --version

# Verify PyTorch CUDA
python -c "import torch; print(torch.cuda.is_available())"
```

### Memory Issues
- Reduce batch size in config
- Close other GPU applications
- Monitor with `nvidia-smi`

### Package Conflicts
```bash
# Clean reinstall
conda env remove -n homestri-ur5e-rl
conda env create -f environment.yml
```

## 📈 Monitoring Training

### TensorBoard
```bash
tensorboard --logdir experiments/
```
- Navigate to `http://localhost:6006`
- Monitor rewards, loss, and curriculum progress

### Log Files
- Training logs: `experiments/[experiment_name]/logs/`
- Checkpoints: `experiments/[experiment_name]/checkpoints/`
- Best models: `experiments/[experiment_name]/best_model/`

## 🎯 Training Phases

1. **Phase 1 (0-5M)**: Approach Learning
2. **Phase 2 (5M-9M)**: Contact Refinement  
3. **Phase 3 (9M-17M)**: Grasping
4. **Phase 4 (17M-23M)**: Manipulation
5. **Phase 5 (23M-27M)**: Mastery

## 🔄 Resuming Training

Training automatically saves checkpoints every 51,200 steps:
```bash
# Resume from checkpoint
python training_script_integrated.py --resume experiments/[experiment_name]/checkpoints/ppo_ur5e_[steps]_steps.zip
```

## 📞 Support

For issues specific to HP Victus RTX 4060 setup:
1. Check GPU drivers are up to date
2. Ensure sufficient power supply
3. Monitor temperatures during training
4. Verify 16GB RAM is sufficient

Happy training! 🚀
