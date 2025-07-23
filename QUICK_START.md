# HP Victus RTX 4060 - Quick Setup Guide

## 🚀 Ultra-Fast Setup (5 minutes)

### Option 1: Conda Setup (Recommended)
```bash
# Clone and setup
git clone <your-repo-url> homestri-ur5e-rl
cd homestri-ur5e-rl

# Create environment (automatic)
conda env create -f environment.yml
conda activate homestri-rl

# Verify setup
python verify_setup.py

# Start training
python homestri_ur5e_rl/training/training_script_integrated.py
```

### Option 2: Automated Script
```bash
# Linux/WSL
chmod +x setup_victus.sh
./setup_victus.sh

# Windows
setup_victus.bat
```

## ⚡ Expected Performance
- **Training Speed**: ~116 FPS (RTX 4060 8GB)
- **Phase 1**: ~12 hours (5M steps)
- **Memory Usage**: ~6.5GB GPU, ~8GB RAM
- **Batch Size**: 64 (optimal)

## 🎯 Quick Verification
Run `python verify_setup.py` and look for:
- ✅ RTX 4060 detected
- ✅ CUDA operations working
- ✅ Homestri environment functional
- ✅ ~116 FPS benchmark

## 🔧 Common Issues & Fixes

### CUDA Not Detected
```bash
# Check drivers
nvidia-smi

# Reinstall PyTorch if needed
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### MuJoCo Issues
```bash
# Linux: Install missing dependencies
sudo apt-get install libosmesa6-dev libgl1-mesa-glx libglfw3

# Windows: Visual C++ Redistributable required
```

### Memory Errors
- Reduce `batch_size` from 64 to 32
- Close other GPU applications
- Monitor with `nvidia-smi`

## 📊 Training Progress Monitoring
```bash
# Real-time monitoring
tensorboard --logdir=experiments/

# Check logs
tail -f experiments/latest/training.log

# GPU usage
watch -n 1 nvidia-smi
```

## 🎮 Controls & Commands
```bash
# Pause training: Ctrl+C
# Resume: python training_script_integrated.py --resume
# Monitor: tensorboard --logdir=experiments/
# Verify: python verify_setup.py
```

## 📈 Training Timeline
- **Phase 1** (5M): Approach learning (~12h)
- **Phase 2** (5M): Grasp refinement (~12h)  
- **Phase 3** (5M): Lift stability (~12h)
- **Phase 4** (7M): Place precision (~17h)
- **Phase 5** (5M): Full integration (~12h)

**Total**: ~65 hours for complete curriculum

For detailed setup instructions, see `SETUP_VICTUS.md`
