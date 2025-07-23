#!/usr/bin/env python3
"""
System verification script for HP Victus RTX 4060 setup
Run this after setup to ensure everything is working correctly
"""

import sys
import subprocess
import importlib
import platform

def check_system_info():
    """Check basic system information"""
    print("🖥️  System Information:")
    print(f"   OS: {platform.system()} {platform.release()}")
    print(f"   Python: {sys.version}")
    print(f"   Architecture: {platform.machine()}")
    print()

def check_nvidia():
    """Check NVIDIA GPU and drivers"""
    print("🔍 NVIDIA GPU Check:")
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            lines = result.stdout.split('\n')
            for line in lines:
                if 'RTX' in line:
                    print(f"   ✅ {line.strip()}")
                    break
            print("   ✅ NVIDIA drivers working")
        else:
            print("   ❌ nvidia-smi failed")
            return False
    except FileNotFoundError:
        print("   ❌ nvidia-smi not found")
        return False
    print()
    return True

def check_cuda():
    """Check CUDA toolkit"""
    print("🔍 CUDA Toolkit Check:")
    try:
        result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            version_line = [line for line in result.stdout.split('\n') if 'release' in line.lower()]
            if version_line:
                print(f"   ✅ {version_line[0].strip()}")
        else:
            print("   ⚠️  nvcc not found (PyTorch will use bundled CUDA)")
    except FileNotFoundError:
        print("   ⚠️  CUDA toolkit not installed (PyTorch will use bundled CUDA)")
    print()

def check_package(package_name, import_name=None, version_attr='__version__'):
    """Check if a package is installed and get version"""
    if import_name is None:
        import_name = package_name
    
    try:
        module = importlib.import_module(import_name)
        version = getattr(module, version_attr, 'Unknown')
        print(f"   ✅ {package_name}: {version}")
        return True
    except ImportError:
        print(f"   ❌ {package_name}: Not installed")
        return False

def check_pytorch_cuda():
    """Detailed PyTorch CUDA check"""
    print("🔥 PyTorch CUDA Check:")
    try:
        import torch
        print(f"   ✅ PyTorch: {torch.__version__}")
        print(f"   ✅ CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"   ✅ CUDA version: {torch.version.cuda}")
            print(f"   ✅ cuDNN version: {torch.backends.cudnn.version()}")
            print(f"   ✅ GPU count: {torch.cuda.device_count()}")
            
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                memory_gb = props.total_memory / (1024**3)
                print(f"   ✅ GPU {i}: {props.name} ({memory_gb:.1f}GB)")
            
            # Test tensor operation
            x = torch.randn(1000, 1000).cuda()
            y = torch.randn(1000, 1000).cuda()
            z = torch.mm(x, y)
            print(f"   ✅ CUDA tensor operations: Working")
            
            return True
        else:
            print("   ❌ CUDA not available in PyTorch")
            return False
    except Exception as e:
        print(f"   ❌ PyTorch error: {e}")
        return False
    print()

def check_core_packages():
    """Check all core packages"""
    print("📦 Core Packages Check:")
    
    packages = [
        ('numpy', 'numpy'),
        ('scipy', 'scipy'),
        ('matplotlib', 'matplotlib'),
        ('opencv-python', 'cv2'),
        ('pillow', 'PIL'),
        ('gymnasium', 'gymnasium'),
        ('stable-baselines3', 'stable_baselines3'),
        ('tensorboard', 'tensorboard'),
        ('mujoco', 'mujoco'),
        ('yaml', 'yaml'),
        ('tqdm', 'tqdm'),
    ]
    
    all_good = True
    for package_name, import_name in packages:
        if not check_package(package_name, import_name):
            all_good = False
    
    print()
    return all_good

def check_homestri_env():
    """Check if homestri environment works"""
    print("🤖 Homestri Environment Check:")
    try:
        from homestri_ur5e_rl.envs import UR5ePickPlaceEnvEnhanced
        print("   ✅ UR5ePickPlaceEnvEnhanced: Importable")
        
        # Try to create environment
        env = UR5ePickPlaceEnvEnhanced(
            xml_file="custom_scene.xml",
            camera_resolution=64,
            render_mode=None
        )
        print("   ✅ Environment creation: Working")
        
        # Test reset
        obs, info = env.reset()
        print(f"   ✅ Environment reset: Working (obs shape: {obs.shape})")
        
        env.close()
        return True
        
    except Exception as e:
        print(f"   ❌ Homestri environment error: {e}")
        return False
    print()

def check_training_config():
    """Check if training configuration is accessible"""
    print("⚙️  Training Configuration Check:")
    try:
        from homestri_ur5e_rl.training.training_script_integrated import IntegratedTrainer
        print("   ✅ IntegratedTrainer: Importable")
        
        trainer = IntegratedTrainer()
        print("   ✅ Trainer creation: Working")
        print(f"   ✅ Config loaded: {len(trainer.config)} parameters")
        
        return True
    except Exception as e:
        print(f"   ❌ Training config error: {e}")
        return False
    print()

def performance_benchmark():
    """Quick GPU performance test"""
    print("⚡ GPU Performance Benchmark:")
    try:
        import torch
        import time
        
        if not torch.cuda.is_available():
            print("   ⚠️  CUDA not available, skipping benchmark")
            return
        
        device = torch.device('cuda')
        
        # Small benchmark
        size = 2048
        x = torch.randn(size, size, device=device)
        y = torch.randn(size, size, device=device)
        
        # Warmup
        for _ in range(5):
            torch.mm(x, y)
        
        torch.cuda.synchronize()
        start_time = time.time()
        
        for _ in range(100):
            torch.mm(x, y)
        
        torch.cuda.synchronize()
        end_time = time.time()
        
        ops_per_sec = 100 / (end_time - start_time)
        print(f"   ✅ Matrix operations: {ops_per_sec:.1f} ops/sec")
        
        # Memory test
        memory_used = torch.cuda.memory_allocated() / (1024**3)
        memory_total = torch.cuda.memory_reserved() / (1024**3)
        print(f"   ✅ GPU memory: {memory_used:.1f}GB used, {memory_total:.1f}GB allocated")
        
    except Exception as e:
        print(f"   ❌ Benchmark error: {e}")
    print()

def main():
    """Main verification function"""
    print("🚀 HP Victus RTX 4060 - Homestri UR5e RL Verification")
    print("=" * 60)
    print()
    
    all_checks = []
    
    check_system_info()
    all_checks.append(check_nvidia())
    check_cuda()
    all_checks.append(check_pytorch_cuda())
    all_checks.append(check_core_packages())
    all_checks.append(check_homestri_env())
    all_checks.append(check_training_config())
    performance_benchmark()
    
    print("📊 Verification Summary:")
    print("=" * 30)
    
    if all(all_checks):
        print("🎉 All systems go! Your setup is ready for training.")
        print("🚀 Start training with: python homestri_ur5e_rl/training/training_script_integrated.py")
    else:
        print("⚠️  Some issues detected. Please review the errors above.")
        failed_checks = sum(1 for check in all_checks if not check)
        print(f"   {failed_checks}/{len(all_checks)} checks failed")
    
    print()
    print("💡 Tips for RTX 4060 8GB:")
    print("   - Use batch_size=64 for optimal performance")
    print("   - Monitor GPU memory with nvidia-smi")
    print("   - Expected training speed: ~116 FPS")
    print("   - Phase 1 (5M steps): ~12 hours")

if __name__ == "__main__":
    main()
