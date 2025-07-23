# UR5e PPO Object Grasping with Homestri Integration

A sophisticated reinforcement learning environment for training a UR5e robotic arm to perform pick-and-place tasks, built on top of the [Homestri UR5e RL framework](https://github.com/ian-chuang/homestri-ur5e-rl) with significant enhancements for sim-to-real transfer and Apple Silicon optimization. I adapted the the training for Apple Silicon as I use a Macbook Pro M2 as my main system, however the repo is intended for training a model using an Ubuntu system with an RTX GPU.

## Project Overview

This project extends the original Homestri UR5e RL environment with advanced features for robust object manipulation training using Proximal Policy Optimization (PPO). 

### Current Status

- **Training Performance**: Successfully reached 330k+ timesteps with stable training on M2 Pro
- **Physics Stability**: Perfect collision-free operation (0% collision rate)
- **Reward Convergence**: Stable rewards in 65-80 range during successful training runs
- **Object Spawning**: Fixed critical bugs in domain randomization and object physics
- **Camera Integration**: Comprehensive CNN perception logging and object visibility tracking
- **Apple Silicon**: Optimized for M1/M2 MacBook Pro

### Key Features

- **Enhanced UR5e Environment**: Advanced pick-place environment with RealSense D435i camera simulation
- **PPO Training Pipeline**: Optimized reinforcement learning with Stable-Baselines3
- **Camera-Aware Training**: Intel RealSense D435i camera integration with comprehensive CNN perception logging
- **Domain Randomization**: Physics and visual randomization with fixed object spawning logic
- **Curriculum Learning**: Progressive difficulty scaling with curriculum management
- **Apple Silicon Optimization**: Native M1/M2 MacBook Pro support
- **Advanced Safety**: Stuck detection, physics stability, and object drop recovery
- **Real-time Analytics**: Comprehensive logging, TensorBoard integration, and policy analysis
- **Physics Debugging**: Fixed object size randomization and physics stability issues
- **Object Perception**: Deep CNN analysis with object-camera correlation tracking

## Architecture & Technologies

### Core Technologies

- **Physics Simulation**: [MuJoCo](https://mujoco.org/) for high-fidelity robot simulation
- **RL Framework**: [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) with PPO algorithm
- **Environment Interface**: [Gymnasium](https://gymnasium.farama.org/) for standardized RL environments
- **Deep Learning**: [PyTorch](https://pytorch.org/) with Apple Silicon MPS support and Nvidia Cuda Su
- **Computer Vision**: Custom CNN feature extractors for visual perception
- **Robot Control**: Operational space controllers from Homestri framework

### Environment Components

```
homestri_ur5e_rl/
├── envs/
│   ├── UR5ePickPlaceEnvEnhanced.py    # Main enhanced environment 
│   ├── base_robot/                     # Base robot components from Homestri
│   │   ├── base_robot_env.py          # Core robot environment
│   │   └── custom_robot_env.py        # Custom robot implementations
│   └── assets/
│       └── base_robot/
│           └── custom_scene.xml       # Enhanced MuJoCo scene with optimized physics
├── training/
│   ├── training_script_integrated.py  # Main PPO training script with object logging
│   ├── sim_to_real_cnn.py             # CNN feature extractor for visual perception
│   ├── progressive_callback.py        # Curriculum learning callbacks
│   ├── curriculum_manager.py          # Advanced curriculum management
│   ├── config_m2_optimized.yaml       # Optimized config for Apple Silicon
│   └── config_rtx4060_optimized.yaml  # Optimized config for Nvidia RTX 4060 GPU
├── utils/
│   ├── realsense.py                   # RealSense D435i camera simulation
│   ├── domain_randomization.py       # Physics randomization 
│   ├── ur5e_stuck_detection_mujoco.py # Advanced safety mechanisms
│   ├── detailed_logging_callback.py  # Comprehensive training analytics
│   └── deployment_utils.py           # Real robot deployment utilities
├── deployment/
│   └── real_robot_deployment.py      # Real UR5e deployment pipeline
├── experiments/                       # Training experiment results 
│   └── ur5e_pickplace_*/             # Individual experiment directories
└── configs/                          # Training configurations
```

##  Getting Started

### Prerequisites

- **Python 3.8+**
- **macOS** (optimized for Apple Silicon M1/M2) or **Ubuntu 24.04** (Recommended)
- **MuJoCo 2.3+**
- **8GB+ RAM** (16GB+ recommended for training)
- **RTX 4060 8GB GPU**

## Training Features

- **Curriculum Learning**: Progressive difficulty with curriculum manager
- **Domain Randomization**: Physics, lighting, and object property variations 
- **Real-time Monitoring**: TensorBoard integration with custom training metrics
- **Checkpoint Management**: Automatic model saving and evaluation callbacks
- **Memory Optimization**: Efficient training on consumer hardware (M1/M2 optimized)
- **Object Perception Logging**: Comprehensive CNN analysis correlating spawned vs perceived objects
- **Physics Stability**: Enhanced contact parameters and object drop recovery

## Camera Integration

The environment features sophisticated camera integration with debugging capabilities:

- **RealSense D435i Simulation**: Accurate camera modeling for sim-to-real transfer
- **RGBD Processing**: Combined RGB and depth information processing
- **Dynamic Camera Switching**: Multiple viewpoints for comprehensive visual training
- **Visual Feature Extraction**: Custom CNN architectures with SimToRealCNNExtractor
- **Object-Camera Correlation**: Deep logging system tracking object visibility and CNN perception
- **Camera Visibility Metrics**: Real-time monitoring of object detection success rates

## Robot Control

### Controller Hierarchy

The project uses the enhanced Homestri controller framework:

1. **Operational Space Controller**: End-effector pose control with improved stability
2. **Joint Position Controller**: Direct joint angle control with safety limits
3. **Joint Velocity Controller**: Velocity-based control with smoothing
4. **Compliance Controller**: Force/torque control for contact tasks

### Enhanced Safety Features

- **Advanced Stuck Detection**: Multi-modal detection and recovery from stuck states
- **Workspace Monitoring**: Real-time boundary checking and collision avoidance
- **Grasp Detection**: Intelligent gripper control and object attachment logic
- **Smooth Trajectories**: Action smoothing for stable robot behavior
- **Physics Stability**: Enhanced contact parameters, friction settings, and object drop recovery
- **Emergency Stops**: Comprehensive safety mechanisms for training and deployment

## Performance & Results

### Current Training Metrics (330k+ Steps Training Run)

- **Training Steps**: Successfully completed 330k+ timesteps on M2 MacBook Pro
- **Physics Stability**: Perfect 0% collision rate throughout training
- **Reward Convergence**: Stable rewards in 65-80 range during successful episodes
- **Training Speed**: ~2000 FPS on MacBook Pro M2 (optimized)
- **Memory Usage**: Efficient operation within 8-16GB RAM constraints
- **Camera Integration**: Fixed object spawning enables proper visual learning

### Success Criteria 

- **Physics Stability**
- **Object Spawning**
- **Camera Visibility**
- **Reaching**
- **Grasping**
- **Transport**

## 📚 Acknowledgments

This project is built upon the excellent [Homestri UR5e RL framework](https://github.com/ian-chuang/homestri-ur5e-rl) by Ian Chuang. The original framework provided the foundation for:

- UR5e robot modeling and simulation with MuJoCo
- Basic environment setup and Gymnasium integration
- Controller implementations and robot kinematics
- Initial pick-place task structure

### Original Homestri Repository References

The base framework was inspired by and references:

- [ARISE-Initiative/robosuite](https://github.com/ARISE-Initiative/robosuite) - Robot simulation environments
- [ir-lab/irl_control](https://github.com/ir-lab/irl_control) - Robot control algorithms
- [abr/abr_control](https://github.com/abr/abr_control) - Adaptive robot control systems

## Key Enhancements Over Original Homestri

This project significantly extends the original framework with production-ready features:

### Advanced RL Training

- **Complete PPO Pipeline**: Full training implementation 
- **Curriculum Learning**: Progressive difficulty scaling with curriculum manager
- **Comprehensive Callbacks**: Detailed logging, evaluation, and checkpoint management
- **Apple Silicon Optimization**: Native M1/M2 support 

### Computer Vision Integration

- **RealSense D435i Simulation**: Accurate camera modeling for sim-to-real transfer
- **CNN Feature Extraction**: Custom SimToRealCNNExtractor for visual perception
- **Object-Camera Correlation**: Deep logging system tracking visual perception accuracy
- **Multi-Camera Support**: Dynamic camera switching and comprehensive viewpoints

### Robust Domain Randomization

- **Physics Randomization**: Comprehensive physics parameter variation (DEBUGGED)
- **Visual Randomization**: Lighting, material, and environmental variations
- **Object Property Randomization**: Size, mass, friction randomization (FIXED)
- **Conditional Randomization**: Proper flag-based control of randomization features

### Production Safety & Stability

- **Advanced Stuck Detection**: Multi-modal detection and recovery mechanisms
- **Physics Stability**: Enhanced contact parameters and object drop recovery
- **Workspace Safety**: Real-time boundary checking and collision avoidance
- **Emergency Systems**: Comprehensive safety mechanisms for training and deployment

### Research & Development Tools

- **Comprehensive Analytics**: TensorBoard integration with custom training metrics
- **Object Perception Logging**: CNN analysis correlating spawned vs perceived objects
- **Policy Analysis Tools**: Model evaluation and behavior analysis utilities
- **Deployment Pipeline**: Complete sim-to-real transfer system for real UR5e robots

## License

This project maintains compatibility with the original Homestri framework licensing. Please refer to the original repository for license details.

## Project Status

**Note**: This repository represents an **active research project**.

The enhanced features focus on achieving high sim-to-real transfer success rates for robotic manipulation tasks through rigorous debugging, comprehensive logging, and stable training performance. **This repository will not be maintained after project completion** but serves as a complete reference implementation for UR5e reinforcement learning research.