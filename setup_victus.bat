@echo off
REM HP Victus RTX 4060 Setup Script for Homestri UR5e RL (Windows)
REM This script sets up the complete environment for CUDA-accelerated training

echo 🚀 Setting up Homestri UR5e RL on HP Victus RTX 4060...
echo ========================================================

REM Check if NVIDIA GPU is available
echo 🔍 Checking NVIDIA GPU...
nvidia-smi >nul 2>&1
if %errorlevel% == 0 (
    nvidia-smi
    echo ✅ NVIDIA GPU detected
) else (
    echo ❌ NVIDIA GPU not detected. Please install NVIDIA drivers first.
    pause
    exit /b 1
)

REM Check CUDA version
echo.
echo 🔍 Checking CUDA version...
nvcc --version >nul 2>&1
if %errorlevel% == 0 (
    nvcc --version
    echo ✅ CUDA toolkit detected
) else (
    echo ⚠️ CUDA toolkit not found. Will install PyTorch with bundled CUDA.
)

REM Create conda environment
echo.
echo 🐍 Creating conda environment...
conda env list | findstr "homestri-ur5e-rl" >nul
if %errorlevel% == 0 (
    echo ⚠️ Environment 'homestri-ur5e-rl' already exists.
    set /p choice="Do you want to remove it and recreate? (y/N): "
    if /i "%choice%"=="y" (
        conda env remove -n homestri-ur5e-rl -y
    ) else (
        echo ❌ Setup cancelled.
        pause
        exit /b 1
    )
)

echo Creating new environment from environment.yml...
conda env create -f environment.yml

echo.
echo 🔧 Activating environment and installing additional packages...
call conda activate homestri-ur5e-rl

REM Verify PyTorch CUDA installation
echo.
echo 🔍 Verifying PyTorch CUDA installation...
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}' if torch.cuda.is_available() else 'CUDA not available'); print(f'GPU count: {torch.cuda.device_count()}' if torch.cuda.is_available() else ''); print(f'GPU name: {torch.cuda.get_device_name(0)}' if torch.cuda.is_available() else '')"

REM Install the package in development mode
echo.
echo 📦 Installing homestri-ur5e-rl in development mode...
pip install -e .

REM Test installations
echo.
echo 🔍 Testing MuJoCo installation...
python -c "import mujoco; print(f'MuJoCo version: {mujoco.__version__}'); print('✅ MuJoCo installation successful')"

echo.
echo 🔍 Testing Gymnasium installation...
python -c "import gymnasium as gym; print(f'Gymnasium version: {gym.__version__}'); print('✅ Gymnasium installation successful')"

echo.
echo 🔍 Testing Stable-Baselines3 installation...
python -c "import stable_baselines3 as sb3; print(f'Stable-Baselines3 version: {sb3.__version__}'); print('✅ Stable-Baselines3 installation successful')"

REM Create directory structure
echo.
echo 📁 Creating directory structure...
if not exist experiments mkdir experiments
if not exist logs mkdir logs
if not exist models mkdir models
if not exist tensorboard mkdir tensorboard
if not exist checkpoints mkdir checkpoints

echo.
echo 🎯 Setup Summary:
echo ================
echo ✅ Conda environment: homestri-ur5e-rl
echo ✅ PyTorch with CUDA support
echo ✅ MuJoCo physics simulation
echo ✅ Stable-Baselines3 RL library
echo ✅ All dependencies installed
echo ✅ Directory structure created
echo.
echo 🚀 Ready for training!
echo.
echo To activate the environment:
echo   conda activate homestri-ur5e-rl
echo.
echo To start training:
echo   cd homestri_ur5e_rl\training
echo   python training_script_integrated.py
echo.
echo To monitor training:
echo   tensorboard --logdir experiments\[experiment_name]\tensorboard
echo.
echo 🎊 Setup complete! Happy training on your RTX 4060!
pause
