# UR5e PPO Object Grasping (Homestri‑based)

End-to-end RL training for a UR5e pick‑and‑place task with RGB‑D perception, curriculum learning, and domain randomization. Runs on Ubuntu (CUDA) and macOS (MPS) with Stable‑Baselines3 PPO and MuJoCo.

## What’s here

- Enhanced UR5e environment with RealSense‑style RGB‑D observation
- PPO training pipeline with custom CNN feature extractor (SimToRealCNN)
- Progressive curriculum with milestone phases and eval sync
- Domain randomization with per‑milestone gating (mass, color, lighting, friction)
- RGB rendering validation and detailed metrics via callbacks
- Works on macOS (MPS) and Ubuntu (CUDA) with auto MuJoCo GL setup

## Repo structure (current)

```
.
├── environment.yml
├── setup.py
├── verify_setup.py
├── experiments/
│   └── ur5e_pickplace_*/
│       ├── config.yaml
│       ├── logs/
│       ├── tensorboard/
│       ├── monitor.csv (optional)
│       ├── vec_normalize.pkl (when saved)
│       └── final_model.zip (when saved)
└── homestri_ur5e_rl/
		├── envs/
		│   ├── UR5ePickPlaceEnvEnhanced.py
		│   ├── assets/
		│   └── mujoco/
		├── training/
		│   ├── training_script_integrated.py
		│   ├── sim_to_real_cnn.py
		│   ├── curriculum_manager.py
		│   ├── progressive_callback.py
		│   ├── curriculum_aware_eval_callback.py
		│   ├── performance_early_stopping_callback.py
		│   ├── config_m2_optimized.yaml
		│   └── config_rtx4060_optimized.yaml
		└── utils/
				├── domain_randomization.py
				├── detailed_logging_callback.py
				├── realsense.py
				└── deployment_utils.py
```

## Key components

### Training script
- File: `training/training_script_integrated.py`
- Sets device: CUDA > MPS > CPU
- Configures MuJoCo GL automatically:
	- Linux with display: `MUJOCO_GL=glfw`
	- Linux headless: `MUJOCO_GL=egl`
	- macOS: `MUJOCO_GL=glfw`
- Builds train/eval envs with `VecNormalize`, validates RGB, and sets up callbacks:
	- `RGBMonitoringCallback` for per‑step/episode metrics and non‑disruptive RGB checks
	- `CheckpointCallback` for periodic saves
	- `CurriculumAwareEvalCallback` to evaluate/sync curriculum
	- `PerformanceEarlyStoppingCallback` to stop on long‑term regressions
- Enables domain randomization after initial grasp successes

### Curriculum manager (current)
- File: `training/curriculum_manager.py`
- Phases (in order):
	- milestone_0_percent → 5% → 10% → 15% → 20% → 25% → 30% → grasping → manipulation → mastery
- Each milestone defines:
	- spawn_radius, objects, mass_range (grams), randomization flags (color, lighting, friction)
	- success thresholds, min episodes/time, and cooldown before advancing
- Provides robust curriculum level sync with env wrappers (env_method/venv/direct), milestone setting, collision reward config, and verification

### Domain randomization (current)
- File: `utils/domain_randomization.py`
- `DomainRandomizer` with dynamics, material, lighting, geometry, and camera noise randomization
- `set_milestone_parameters(mass_range, color_randomization, lighting_randomization, friction_randomization, objects)`:
	- mass_range in grams is converted to kg; object masses sampled as absolute values
	- independently gates color, lighting, and friction randomization
- `CurriculumDomainRandomizer.set_curriculum_level(level)` scales ranges by curriculum level

### Visual perception
- Custom CNN features: `training/sim_to_real_cnn.py`
- RGB validation indices handled consistently in the trainer
- Optional object perception logging in tests for visibility/content checks

## Setup

- Create a Conda env using `environment.yml`
- MuJoCo GL is auto‑configured by the training script; no manual export needed in common cases

## Train

From `homestri_ur5e_rl/training`:

```bash
# macOS (MPS) or Ubuntu (CUDA); pick a config
python training_script_integrated.py --config config_m2_optimized.yaml
# or
python training_script_integrated.py --config config_rtx4060_optimized.yaml
```

Artifacts are written under `experiments/ur5e_pickplace_YYYYMMDD_HHMMSS/`.

## Evaluate

```bash
# After training, test best model (renders if supported)
python training_script_integrated.py --config config_rtx4060_optimized.yaml --test experiments/<run>/best_model --episodes 5 --visual
```

## Notes

- Training uses `VecNormalize` for observations; evaluation env stats are synced after each training chunk
- RGB validation runs non‑disruptively during training; issues print GL/DISPLAY hints
- Domain randomization is gated per‑milestone and typically enabled after initial grasp successes

## Acknowledgments

Built on the Homestri UR5e RL framework and the MuJoCo/Gymnasium/Stable‑Baselines3/PyTorch ecosystem.

## License

See upstream Homestri project for original license terms. This repo follows compatible licensing.
