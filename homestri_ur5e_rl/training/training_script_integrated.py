"""
Integrated Training Script for UR5e Pick-Place
Fixed for RGB rendering issues on Ubuntu
"""

import os
import sys
import platform
import numpy as np
import torch
from datetime import datetime
from pathlib import Path
import yaml
from typing import Dict, Optional
import time

# Setup Ubuntu MuJoCo compatibility with RGB fix
def setup_ubuntu_mujoco():
    """Setup MuJoCo for Ubuntu with proper RGB rendering"""
    if platform.system() == "Linux":
        print("🐧 Configuring MuJoCo for Ubuntu...")
        
        # Check if we have a display available
        has_display = "DISPLAY" in os.environ
        
        if has_display:
            # If display is available, use GLFW for better RGB rendering
            print("   Display detected - using GLFW for RGB support")
            os.environ["MUJOCO_GL"] = "glfw"
            # Don't force headless mode
            if "MUJOCO_HEADLESS" in os.environ:
                del os.environ["MUJOCO_HEADLESS"]
        else:
            # Only use EGL if truly headless
            print("   No display detected - using EGL (RGB may be limited)")
            os.environ["MUJOCO_GL"] = "egl"
            os.environ["MUJOCO_HEADLESS"] = "1"
        
        # OpenGL settings for stability
        os.environ["MESA_GL_VERSION_OVERRIDE"] = "3.3"
        os.environ["MESA_GLSL_VERSION_OVERRIDE"] = "330"
        
        # Don't disable extensions - they may be needed for proper rendering
        if "MUJOCO_GL_DISABLE_EXTENSIONS" in os.environ:
            del os.environ["MUJOCO_GL_DISABLE_EXTENSIONS"]
        
        print("✅ Ubuntu MuJoCo environment configured")
    elif platform.system() == "Darwin":
        print("🍎 macOS detected - using Apple Silicon optimizations")
        os.environ["MUJOCO_GL"] = "glfw"

# Apply Ubuntu fixes immediately
setup_ubuntu_mujoco()

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from homestri_ur5e_rl.training.curriculum_aware_eval_callback import CurriculumAwareEvalCallback
from stable_baselines3.common.monitor import Monitor

# Import fixed components
from homestri_ur5e_rl.envs import UR5ePickPlaceEnvEnhanced
from homestri_ur5e_rl.training.sim_to_real_cnn import SimToRealCNNExtractor
from homestri_ur5e_rl.training.progressive_callback import ProgressiveTrainingCallback
from homestri_ur5e_rl.utils.detailed_logging_callback import DetailedLoggingCallback
from homestri_ur5e_rl.training.curriculum_manager import CurriculumManager
from homestri_ur5e_rl.training.performance_early_stopping_callback import PerformanceEarlyStoppingCallback

class IntegratedTrainer:
    """Trainer with optimized hyperparameters for 200-step episodes and improved reward balance"""
    
    def __init__(self, config_path: Optional[str] = None, visual_training: bool = False):
        # Load configuration
        self.config = self.load_config(config_path)
        
        # Setup visual training mode
        self.visual_training = visual_training
        if visual_training:
            print("Visual training mode enabled - you can watch the training process")
            self.config["environment"]["render_mode"] = "human"
        
        # Setup device
        self.setup_device()
        
        self.setup_directories()
        
        # Initialize components
        self.curriculum_manager = None
        self.env = None
        self.model = None
        
    def load_config(self, config_path: Optional[str] = None) -> Dict:
        """Load training configuration"""
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                
            # Ensure headless mode is disabled for RGB rendering on Linux
            if platform.system() == "Linux" and "DISPLAY" in os.environ:
                if "headless" in config.get("environment", {}):
                    config["environment"]["headless"] = False
                    print("   Disabled headless mode for RGB rendering")
                    
            return config
        
        # Default configuration
        return {
            "environment": {
                "xml_file": "custom_scene.xml",
                "camera_resolution": 64,
                "control_mode": "joint",
                "use_stuck_detection": True,
                "use_domain_randomization": False,  # start disabled; enable after early grasps
                "frame_skip": 5,
                "initial_curriculum_level": 0.05,  # Match milestone_0_percent phase level
                "render_mode": None,
                "headless": False,
            },
            "training": {
                "total_timesteps": 800_000,
                "learning_rate": 0.0005,  # INCREASED: better learning with improved rewards
                "n_steps": 400,  # REDUCED: optimal for 200-step episodes (2x episode length)
                "batch_size": 200,  # REDUCED: matches shorter episodes better
                "n_epochs": 8,  # REDUCED: prevent overfitting on smaller batches
                "gamma": 0.995,
                "gae_lambda": 0.95,
                "clip_range": 0.2,
                "ent_coef": 0.005,  # SMALL INCREASE: encourage exploration
                "vf_coef": 0.5,
                "max_grad_norm": 0.7,
                "detailed_log_freq": 1600,  # REDUCED: matches new n_steps
                "target_kl": 0.02,
            },
            "evaluation": {
                "eval_freq": 16_000,  # REDUCED: matches new n_steps (40x400)
                "n_eval_episodes": 5,
            },
            "logging": {
                "save_freq": 40_000,  # REDUCED: matches new n_steps (100x400)
                "log_interval": 10,
            }
        }
    
    def setup_device(self):
        """Setup device for training"""
        if torch.cuda.is_available():
            self.device = "cuda"
            print(f"Using CUDA acceleration (GPU: {torch.cuda.get_device_name(0)})")
            # RTX 4060 optimizations
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
        elif torch.backends.mps.is_available():
            self.device = "mps"
            print("Using Apple Silicon MPS acceleration")
            # M2 optimizations
            os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
            torch.set_num_threads(8)
        else:
            self.device = "cpu"
            print("Using CPU")
    
    def setup_directories(self):
        """Create experiment directories"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.exp_name = f"ur5e_pickplace_{timestamp}"
        self.exp_dir = Path("experiments") / self.exp_name
        
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        (self.exp_dir / "checkpoints").mkdir(exist_ok=True)
        (self.exp_dir / "eval").mkdir(exist_ok=True)
        (self.exp_dir / "logs").mkdir(exist_ok=True)
        (self.exp_dir / "tensorboard").mkdir(exist_ok=True)
        
        # Save config
        with open(self.exp_dir / "config.yaml", 'w') as f:
            yaml.dump(self.config, f)
            
        print(f"📁 Experiment directory: {self.exp_dir}")
    
    def create_env(self):
        """Create environment with RGB rendering validation"""
        print("\n🚀 Starting Integrated Training")
        print("="*50)
        print(f"🎯 Target: 70%+ real-world success rate")
        print(f"📷 RealSense D435i camera simulation")
        print(f"🧠 SimToRealCNN with camera memory")
        print(f"📈 Progressive curriculum learning")
        print(f"💻 Platform: {platform.system()} - {self.device}")
        print("="*50)
        
        # Create environment with explicit headless=False for RGB
        env_config = self.config["environment"].copy()
        if platform.system() == "Linux":
            env_config["headless"] = False
            
        env = UR5ePickPlaceEnvEnhanced(**env_config)
        
        # Validate RGB rendering is working
        print("\n🔍 Validating RGB rendering...")
        obs, _ = env.reset()
        self._validate_rgb_rendering(obs, "Initial environment test")
        
        # Wrap with Monitor for logging
        env = Monitor(env, filename=str(self.exp_dir / "monitor.csv"))
        
        # Create vectorized environment
        if platform.system() == "Linux":
            # For Linux, ensure each subprocess also has proper GL settings
            def make_env():
                # Ensure GL settings are propagated to subprocess
                if "DISPLAY" in os.environ:
                    os.environ["MUJOCO_GL"] = "glfw"
                    if "MUJOCO_HEADLESS" in os.environ:
                        del os.environ["MUJOCO_HEADLESS"]
                
                env_config = self.config["environment"].copy()
                env_config["headless"] = False
                return Monitor(UR5ePickPlaceEnvEnhanced(**env_config))
            
            # Create multiple environments for parallel training
            n_envs = self.config.get("environment", {}).get("n_envs", 2)
            self.train_env = DummyVecEnv([make_env for _ in range(n_envs)])
        else:
            # For other platforms, use simple setup
            self.train_env = DummyVecEnv([lambda: env])
        
        self.train_env = VecNormalize(
            self.train_env, 
            norm_obs=True, 
            norm_reward=False,  # REVERTED: Disable reward normalization to match baseline
            clip_obs=10.0,
            clip_reward=25.0,  # FIXED: Match environment reward clipping bounds
            gamma=self.config["training"]["gamma"]
        )
        
        # Evaluation environment - should be deterministic for consistent evaluation
        eval_env_config = self.config["environment"].copy()
        eval_env_config["render_mode"] = None
        eval_env_config["headless"] = False  # Ensure RGB works in eval too
        eval_env_config["use_domain_randomization"] = False  # Disable randomization for consistent evaluation
        eval_env = UR5ePickPlaceEnvEnhanced(**eval_env_config)
        eval_env = Monitor(eval_env)
        self.eval_env = DummyVecEnv([lambda: eval_env])
        
        # Normalize eval env but don't update stats
        self.eval_env = VecNormalize(
            self.eval_env, 
            norm_obs=True, 
            norm_reward=False,  # REVERTED: Match training environment normalization
            clip_obs=10.0, 
            clip_reward=25.0,  # FIXED: Match training environment clipping
            training=False
        )
        
        # Initialize curriculum manager with vectorized environment
        # CRITICAL FIX: Create curriculum manager with reset callback
        def reset_training_metrics():
            """Reset training metrics when curriculum phase changes"""
            # CRITICAL FIX: Initialize attributes if they don't exist
            if not hasattr(self, 'total_grasps'):
                self.total_grasps = 0
            if not hasattr(self, 'total_successes'):
                self.total_successes = 0
                
            print(f"   🔄 Resetting training metrics: grasps={self.total_grasps} → 0, successes={self.total_successes} → 0")
            self.total_grasps = 0
            self.total_successes = 0
            # Note: Don't reset total_episodes as it should accumulate across phases
            
        self.curriculum_manager = CurriculumManager(self.train_env, on_phase_change_callback=reset_training_metrics)
        
        # FIXED: Set curriculum manager reference in environment for phase-aware success criteria
        env.curriculum_manager = self.curriculum_manager
        eval_env.curriculum_manager = self.curriculum_manager
        
        # CRITICAL FIX: Force evaluation environment to use SAME curriculum phase and success criteria as training
        try:
            # Force curriculum level sync immediately (not just during evaluation)
            if hasattr(env, 'curriculum_level') and hasattr(eval_env, 'set_curriculum_level'):
                eval_env.set_curriculum_level(env.curriculum_level)
                print(f"✅ Evaluation environment curriculum level synced to training: {env.curriculum_level}")
            else:
                initial_curriculum_level = self.config["environment"].get("initial_curriculum_level", 0.05)
                if hasattr(eval_env, 'set_curriculum_level'):
                    eval_env.set_curriculum_level(initial_curriculum_level)
                    print(f"✅ Evaluation environment curriculum level set to: {initial_curriculum_level}")
                    
            # CRUCIAL: Ensure evaluation uses same success criteria as training by sharing curriculum manager
            print(f"✅ Evaluation environment using curriculum phase: {self.curriculum_manager.current_phase}")
        except Exception as e:
            print(f"⚠️ Warning: Could not sync curriculum to evaluation environment: {e}")
        
        print(f"✅ Environment created with camera resolution: {self.config['environment']['camera_resolution']}")
        print(f"✅ Curriculum manager initialized")
        
        # Final RGB validation
        obs = self.train_env.reset()
        self._validate_rgb_rendering(obs[0], "Vectorized environment test")
    
    def _validate_rgb_rendering(self, obs: np.ndarray, context: str = ""):
        """Validate that RGB data is being rendered properly"""
        try:
            cam_res = self.config["environment"]["camera_resolution"]
            
            # Extract RGB data (same indices as in test script)
            rgb_start_idx = 55  # After kinematic data
            rgb_end_idx = rgb_start_idx + (cam_res * cam_res * 3)
            
            if len(obs) >= rgb_end_idx:
                rgb_obs = obs[rgb_start_idx:rgb_end_idx]
                rgb_mean = np.mean(rgb_obs)
                rgb_std = np.std(rgb_obs)
                rgb_max = np.max(rgb_obs)
                
                if rgb_max == 0.0:
                    print(f"⚠️  RGB RENDERING ISSUE - {context}")
                    print(f"   RGB data is all zeros! This will prevent visual learning.")
                    print(f"   Attempting to diagnose...")
                    
                    # Check environment variables
                    print(f"   MUJOCO_GL: {os.environ.get('MUJOCO_GL', 'not set')}")
                    print(f"   MUJOCO_HEADLESS: {os.environ.get('MUJOCO_HEADLESS', 'not set')}")
                    print(f"   DISPLAY: {os.environ.get('DISPLAY', 'not set')}")
                    
                    # Try to fix by switching renderer
                    if platform.system() == "Linux" and os.environ.get("MUJOCO_GL") == "egl":
                        print("   Attempting to switch from EGL to GLFW...")
                        os.environ["MUJOCO_GL"] = "glfw"
                        if "MUJOCO_HEADLESS" in os.environ:
                            del os.environ["MUJOCO_HEADLESS"]
                else:
                    print(f"✅ RGB rendering validated - {context}")
                    print(f"   RGB: mean={rgb_mean:.3f}, std={rgb_std:.3f}, max={rgb_max:.3f}")
            else:
                print(f"⚠️  Cannot validate RGB - observation too short")
                
        except Exception as e:
            print(f"⚠️  RGB validation error: {e}")
    
    def create_model(self):
        """Create PPO model with fixed hyperparameters"""
        print("\n🧠 Creating PPO model with SimToRealCNN...")
        
        # Ensure tensorboard directory exists
        tensorboard_log = str(self.exp_dir / "tensorboard")
        Path(tensorboard_log).mkdir(parents=True, exist_ok=True)
        
        # Policy kwargs with conservative settings
        policy_kwargs = dict(
            features_extractor_class=SimToRealCNNExtractor,
            features_extractor_kwargs=dict(
                features_dim=512,
                camera_resolution=self.config["environment"]["camera_resolution"]
            ),
            net_arch=dict(
                pi=[256, 256],  # Policy network
                vf=[256, 256],  # Value network
            ),
            activation_fn=torch.nn.Tanh,  # Tanh for stability
            ortho_init=True,  # Orthogonal initialization
            log_std_init=0.0,  # Initial policy variance
        )
        
        # Optimized PPO settings for 200-step episodes
        self.model = PPO(
            "MlpPolicy",
            self.train_env,
            learning_rate=self.config["training"]["learning_rate"],
            n_steps=self.config["training"]["n_steps"],
            batch_size=self.config["training"]["batch_size"],
            n_epochs=self.config["training"]["n_epochs"],
            gamma=self.config["training"]["gamma"],
            gae_lambda=self.config["training"]["gae_lambda"],
            clip_range=self.config["training"]["clip_range"],
            clip_range_vf=None,
            ent_coef=self.config["training"]["ent_coef"],
            vf_coef=self.config["training"]["vf_coef"],
            max_grad_norm=self.config["training"]["max_grad_norm"],
            target_kl=self.config["training"].get("target_kl", None),
            normalize_advantage=True,
            policy_kwargs=policy_kwargs,
            tensorboard_log=tensorboard_log,
            device=self.device,
            verbose=1,
        )
        
        print(f"✅ Model created with device: {self.device}")
        print(f"📊 Policy architecture: {policy_kwargs}")

    def create_callbacks(self):
        """Create training callbacks"""
        callbacks = []
        
        # Detailed logging callback with RGB monitoring
        class RGBMonitoringCallback(DetailedLoggingCallback):
            def __init__(self, trainer, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.trainer = trainer
                self.rgb_check_freq = 50000  # Check RGB every 50k steps (reduced frequency to avoid training disruption)
                # Custom counters
                self.total_successes = 0
                self.total_grasps = 0
                self.total_episodes = 0
                self.phase_name_to_idx = {name: i for i, name in enumerate([
                    "milestone_0_percent", "milestone_5_percent", "milestone_10_percent",
                    "milestone_15_percent", "milestone_20_percent", "milestone_25_percent",
                    "milestone_30_percent", "grasping", "manipulation", "mastery"
                ])}
                # New aggregation buffers
                self.recent_min_dists = []
                self.recent_approach_events = []
                self.recent_contact_events = []
                self.recent_vertical_signed_medians = []
                self.recent_episode_successes = []  # Track individual episode success/failure
                self.window = 100  # aggregation window
                self._current_episode_min_planar = np.inf
                self._current_episode_vertical_signed = []
            
            def _update_curriculum_only(self):
                """Update curriculum manager without full logging - called every rollout"""
                if not self.trainer.curriculum_manager:
                    return
                
                # CRITICAL FIX: Use exact same calculation as detailed logging callback
                recent_window = 50  # Always use 50 episodes for consistency across all metrics
                if hasattr(self, 'recent_episode_successes') and self.recent_episode_successes:
                    recent_success_rate = np.mean(self.recent_episode_successes[-recent_window:]) if len(self.recent_episode_successes) >= 1 else 0.0
                else:
                    recent_success_rate = 0.0  # No episodes recorded yet
                
                # Update curriculum manager
                curriculum_result = self.trainer.curriculum_manager.update(recent_success_rate)
                
                # Log phase changes immediately
                if curriculum_result.get('phase_changed', False):
                    print(f"\n🎓 CURRICULUM ADVANCED: {curriculum_result['old_phase']} → {curriculum_result['new_phase']}")
                    print(f"   Recent success rate: {recent_success_rate:.1%}, Duration: {curriculum_result['phase_duration_hours']:.1f}h")
                    print(f"   Step: {self.n_calls}, Episodes: {self.total_episodes}")

            def _log_custom_metrics(self):
                # URGENT FIX: Update curriculum manager with recent success rate
                overall_success_rate = self.total_successes / max(1, self.total_episodes)
                
                # CRITICAL FIX: Use exact same calculation as detailed logging callback
                recent_window = 50  # Always use 50 episodes for consistency across all metrics
                if hasattr(self, 'recent_episode_successes') and self.recent_episode_successes:
                    # Use exact same calculation method as detailed_logging_callback.py:125-126
                    recent_success_rate = np.mean(self.recent_episode_successes[-recent_window:]) if len(self.recent_episode_successes) >= 1 else 0.0
                else:
                    recent_success_rate = 0.0  # No episodes recorded yet
                
                if self.trainer.curriculum_manager:
                    curriculum_result = self.trainer.curriculum_manager.update(recent_success_rate)
                    if curriculum_result.get('phase_changed', False):
                        print(f"\n🎓 CURRICULUM ADVANCED: {curriculum_result['old_phase']} → {curriculum_result['new_phase']}")
                        print(f"   Recent success rate: {recent_success_rate:.1%}, Duration: {curriculum_result['phase_duration_hours']:.1f}h")
                
                phase = self.trainer.curriculum_manager.current_phase if self.trainer.curriculum_manager else "unknown"
                phase_idx = self.phase_name_to_idx.get(phase, -1)
                grasp_rate = self.total_grasps / max(1, self.total_episodes)
                # Distance/contact metrics
                if self.recent_min_dists:
                    median_min_dist = float(np.median(self.recent_min_dists))
                    p25_min_dist = float(np.percentile(self.recent_min_dists, 25))
                    p75_min_dist = float(np.percentile(self.recent_min_dists, 75))
                else:
                    median_min_dist = p25_min_dist = p75_min_dist = np.nan
                avg_approach_events = float(np.mean(self.recent_approach_events)) if self.recent_approach_events else 0.0
                avg_contact_events = float(np.mean(self.recent_contact_events)) if self.recent_contact_events else 0.0
                approach_episode_rate = float(np.mean([1.0 if x>0 else 0.0 for x in self.recent_approach_events])) if self.recent_approach_events else 0.0
                contact_episode_rate = float(np.mean([1.0 if x>0 else 0.0 for x in self.recent_contact_events])) if self.recent_contact_events else 0.0
                median_vertical_signed = float(np.median(self.recent_vertical_signed_medians)) if self.recent_vertical_signed_medians else np.nan
                self.logger.record("custom/total_grasps", float(self.total_grasps))
                self.logger.record("custom/total_successes", float(self.total_successes))
                self.logger.record("custom/episodes", float(self.total_episodes))
                self.logger.record("custom/grasp_rate_per_episode", float(grasp_rate))
                # CRITICAL FIX: Log the same success rate that all other metrics use (recent 50-episode window)
                self.logger.record("custom/success_rate_per_episode", float(recent_success_rate))
                self.logger.record("custom/lifetime_success_rate", float(overall_success_rate))  # Keep lifetime for reference
                self.logger.record("custom/curriculum_phase_idx", float(phase_idx))
                self.logger.record("custom/curriculum_phase_name", phase)
                # New logs
                self.logger.record("distance/median_min_object_distance", median_min_dist)
                self.logger.record("distance/p25_min_object_distance", p25_min_dist)
                self.logger.record("distance/p75_min_object_distance", p75_min_dist)
                self.logger.record("distance/approach_events_mean", avg_approach_events)
                self.logger.record("distance/contact_events_mean", avg_contact_events)
                self.logger.record("distance/approach_episode_rate", approach_episode_rate)
                self.logger.record("distance/contact_episode_rate", contact_episode_rate)
                self.logger.record("vertical/median_final_vertical_signed", median_vertical_signed)

            def _on_step(self) -> bool:
                result = super()._on_step()
                infos = self.locals.get("infos", [])
                dones = self.locals.get("dones", [])
                for i, info in enumerate(infos):
                    if not isinstance(info, dict):
                        continue
                    # CRITICAL FIX: Only count grasps on episode termination (done=True)
                    if dones[i]:  # Only count when episode ends
                        grasp_events = info.get("grasp_events", 0)
                        if grasp_events > 0:
                            # DEBUG: Log grasp counting to detect phantom increments
                            print(f"🔍 GRASP COUNT DEBUG: Episode {i} ENDED - adding {grasp_events} grasps (total was {self.total_grasps})")
                            self.total_grasps += grasp_events  # Add all grasps from this episode
                            print(f"🔍 GRASP COUNT DEBUG: Total grasps now {self.total_grasps}")
                    
                    # Extract reward components for other metrics
                    rc = info.get("reward_components", {})
                    
                    # Fallback extraction if planar_distance missing
                    pd = info.get("planar_distance", None)
                    if pd is None and isinstance(rc, dict):
                        pd = rc.get("planar_dist", None)
                    if pd is not None:
                        if pd < self._current_episode_min_planar:
                            self._current_episode_min_planar = pd
                    vs = info.get("vertical_signed", None)
                    if vs is None and isinstance(rc, dict):
                        vs = rc.get("vertical_signed", None)
                    if vs is not None:
                        self._current_episode_vertical_signed.append(vs)
                    # Debug: sample first few episodes if metrics absent
                    if self.total_episodes < 5 and (pd is None or vs is None) and self.n_calls < 2000 and isinstance(rc, dict):
                        if not hasattr(self, '_early_debug_printed'):
                            print("[DEBUG] Missing per-step metrics. Sample reward_components keys:", list(rc.keys())[:10])
                            self._early_debug_printed = True
                    done_flag = False
                    if isinstance(dones, (list, np.ndarray)) and i < len(dones):
                        done_flag = bool(dones[i])
                    if done_flag:
                        self.total_episodes += 1
                        episode_success = info.get("is_success", False)
                        if episode_success:
                            self.total_successes += 1
                        
                        # Track individual episode results for curriculum
                        self.recent_episode_successes.append(1.0 if episode_success else 0.0)
                        # Register with curriculum manager for gated advancement
                        if self.trainer.curriculum_manager:
                            try:
                                self.trainer.curriculum_manager.register_episode_result(bool(episode_success))
                            except Exception as e:
                                pass
                        # Keep only recent episodes (last 50)
                        if len(self.recent_episode_successes) > 50:
                            self.recent_episode_successes = self.recent_episode_successes[-50:]
                        if np.isfinite(self._current_episode_min_planar):
                            self.recent_min_dists.append(self._current_episode_min_planar)
                        # Episode-level fallback for approach/contact counts
                        ae = info.get("approach_events", rc.get("approach_events", 0) if isinstance(rc, dict) else 0)
                        ce = info.get("contact_events", rc.get("contact_events", 0) if isinstance(rc, dict) else 0)
                        self.recent_approach_events.append(ae)
                        self.recent_contact_events.append(ce)
                        if self._current_episode_vertical_signed:
                            self.recent_vertical_signed_medians.append(float(np.median(self._current_episode_vertical_signed)))
                        # Maintain windows
                        if len(self.recent_min_dists) > self.window:
                            self.recent_min_dists = self.recent_min_dists[-self.window:]
                        if len(self.recent_approach_events) > self.window:
                            self.recent_approach_events = self.recent_approach_events[-self.window:]
                        if len(self.recent_contact_events) > self.window:
                            self.recent_contact_events = self.recent_contact_events[-self.window:]
                        if len(self.recent_vertical_signed_medians) > self.window:
                            self.recent_vertical_signed_medians = self.recent_vertical_signed_medians[-self.window:]
                        # Reset episode trackers
                        self._current_episode_min_planar = np.inf
                        self._current_episode_vertical_signed = []
                # CRITICAL FIX: Don't reset environment during training for RGB validation
                # This was causing training disruption at step 10k
                if self.n_calls % self.rgb_check_freq == 0 and self.n_calls > 0:
                    # Just validate current observation without resetting
                    current_obs = self.locals.get('observations', None)
                    if current_obs is not None and len(current_obs) > 0:
                        self.trainer._validate_rgb_rendering(current_obs[0], f"Step {self.n_calls} (non-disruptive check)")
                # CURRICULUM FIX: Update curriculum every rollout (512 steps) not just every log_freq (1600 steps)
                if self.n_calls % 512 == 0:
                    self._update_curriculum_only()
                
                if self.n_calls % self.log_freq == 0:
                    self._log_custom_metrics()
                return result
        
        detailed_logger = RGBMonitoringCallback(
            self,
            log_freq=self.config["training"]["detailed_log_freq"],
            curriculum_manager=self.curriculum_manager
        )
        callbacks.append(detailed_logger)
        
        # Checkpoint callback
        checkpoint_callback = CheckpointCallback(
            save_freq=self.config["logging"]["save_freq"],
            save_path=str(self.exp_dir / "checkpoints"),
            name_prefix="ppo_ur5e",
            save_vecnormalize=True,
        )
        callbacks.append(checkpoint_callback)
        
        # CRITICAL FIX: Use curriculum-aware evaluation callback that syncs before evaluations
        eval_callback = CurriculumAwareEvalCallback(
            self.eval_env,
            trainer=self,  # Pass reference for curriculum sync
            best_model_save_path=str(self.exp_dir / "best_model"),
            log_path=str(self.exp_dir / "eval"),
            eval_freq=self.config["evaluation"]["eval_freq"],
            n_eval_episodes=self.config["evaluation"]["n_eval_episodes"],
            deterministic=True,
            render=False,
        )
        callbacks.append(eval_callback)
        
        # FIXED: More lenient early stopping to avoid premature termination
        early_stopping_callback = PerformanceEarlyStoppingCallback(
            patience=10,  # FIXED: Allow 10 evaluations (100k steps) without improvement - was too aggressive at 5
            min_delta=0.01,  # FIXED: 1% improvement threshold - was too strict at 2%
            performance_window=15,  # FIXED: Average over 15 evaluations for more stability
            degradation_threshold=0.10,  # FIXED: Stop only if 10% performance drop - was too sensitive at 5%
            evaluation_freq=self.config["evaluation"]["eval_freq"],
            verbose=1
        )
        # DISABLED: Comment out early stopping until evaluation metrics are fixed
        # callbacks.append(early_stopping_callback)
        
        # Remove ProgressiveTrainingCallback to avoid duplicate curriculum logic
        
        return CallbackList(callbacks)
    
    def sync_vecnormalize_stats(self):
        """Sync VecNormalize stats and curriculum state from training to evaluation environment"""
        if isinstance(self.train_env, VecNormalize) and isinstance(self.eval_env, VecNormalize):
            self.eval_env.obs_rms = self.train_env.obs_rms
            self.eval_env.ret_rms = self.train_env.ret_rms
            print("✅ VecNormalize stats synced from training to evaluation environment")
        
        # FIXED: Sync curriculum level to evaluation environment
        try:
            # Get current curriculum level from training environment
            if hasattr(self.train_env, 'venv'):
                # VecNormalize wrapped environment
                training_base_env = self.train_env.venv.envs[0]
            else:
                # Direct access
                training_base_env = self.train_env.envs[0]
            
            if hasattr(self.eval_env, 'venv'):
                # VecNormalize wrapped environment
                eval_base_env = self.eval_env.venv.envs[0]
            else:
                # Direct access
                eval_base_env = self.eval_env.envs[0]
            
            # FIXED: Store curriculum state BEFORE evaluation, will apply AFTER reset
            self.curriculum_state_to_sync = None
            if hasattr(training_base_env, 'curriculum_manager'):
                self.curriculum_state_to_sync = {
                    'current_phase': training_base_env.curriculum_manager.current_phase,
                    'curriculum_level': getattr(training_base_env, 'curriculum_level', 0.05),
                    'phase_progress': training_base_env.curriculum_manager.phase_progress,
                    'phase_episode_count': training_base_env.curriculum_manager.phase_episode_count
                }
                print(f"📋 Stored curriculum state for eval sync: {self.curriculum_state_to_sync['current_phase']} @ level {self.curriculum_state_to_sync['curriculum_level']:.3f}")
            
            # Will sync AFTER reset to prevent initialization override
                
        except Exception as e:
            print(f"⚠️ Warning: Could not sync curriculum state to evaluation environment: {e}")
            # Continue without failing - evaluation will use default settings
    
    def train(self):
        """Main training loop with RGB monitoring"""
        self.create_env()
        self.create_model()
        callbacks = self.create_callbacks()
        # Track early grasp successes to toggle domain randomization
        grasp_success_counter = 0
        domain_randomization_enabled = False
        try:
            start_time = time.time()
            while self.model.num_timesteps < self.config["training"]["total_timesteps"]:
                self.model.learn(total_timesteps=min(50_000, self.config["training"]["total_timesteps"] - self.model.num_timesteps),
                                  callback=callbacks,
                                  reset_num_timesteps=False,
                                  progress_bar=True,  # re-enable progress bar
                                  log_interval=self.config["logging"]["log_interval"])
                
                # Sync normalization and curriculum to align training/evaluation behavior
                self.sync_vecnormalize_stats()
                
                # After each chunk, check recent monitor file for grasp events
                if self.curriculum_manager and self.curriculum_manager.collision_history:
                    recent = self.curriculum_manager.collision_history[-20:]
                    for ep in recent:
                        if isinstance(ep, dict) and (ep.get("successful_grasp") or ep.get("object_grasped")):
                            grasp_success_counter += 1
                if (not domain_randomization_enabled) and grasp_success_counter >= 10:
                    print("🌈 Enabling domain randomization after 10 grasp successes")
                    # Enable in underlying env instance(s)
                    base_env = self.train_env.envs[0].env
                    if hasattr(base_env, 'use_domain_randomization'):
                        base_env.use_domain_randomization = True
                        if hasattr(base_env, 'domain_randomizer'):
                            base_env.domain_randomizer.set_curriculum_level(base_env.curriculum_level)
                    domain_randomization_enabled = True
        except KeyboardInterrupt:
            print("\n⚠️  Training interrupted by user")
        # Training completion summary
        end_time = time.time()
        training_duration = end_time - start_time
        
        print(f"\n🎯 Training Summary:")
        print(f"   Duration: {training_duration/3600:.1f} hours")
        print(f"   Average FPS: {self.config['training']['total_timesteps']/training_duration:.1f}")
        print(f"   Final curriculum phase: {self.curriculum_manager.current_phase if self.curriculum_manager else 'N/A'}")
        
        # Log final model stats
        self._log_final_training_stats()
        
        # Analyze final approach learning progress
        self.analyze_approach_learning_progress()
        
        # Get actual current timesteps from the model
        actual_timesteps = self.model.num_timesteps if hasattr(self.model, 'num_timesteps') else 0
        print(f"\n🔍 Debugging Step Count:")
        print(f"   Configured total timesteps: {self.config['training']['total_timesteps']:,}")
        print(f"   Actual completed timesteps: {actual_timesteps:,}")
        
        self.predict_breakthrough_timeline(actual_timesteps)
        
        # Save final model
        print("\n💾 Saving final model...")
        self.model.save(str(self.exp_dir / "final_model"))
        self.train_env.save(str(self.exp_dir / "vec_normalize.pkl"))
        
        print(f"\n✅ Training completed!")
        print(f"📁 Results saved to: {self.exp_dir}")
        
        # Print tensorboard command
        print(f"\n📊 View training progress with:")
        print(f"   tensorboard --logdir {self.exp_dir / 'tensorboard'}")
    
    def test_model(self, model_path: Optional[str] = None, n_episodes: int = 5):
        """Test trained model with detailed object perception logging"""
        if model_path is None:
            model_path = self.exp_dir / "best_model"
            
        print(f"\n🧪 Testing model: {model_path}")
        
        # Load model
        model = PPO.load(str(model_path / "best_model.zip"))
        
        test_env_config = self.config["environment"].copy()
        test_env_config["render_mode"] = "human"
        test_env_config["use_domain_randomization"] = False
        test_env_config["headless"] = False  # Ensure RGB works
        
        env = UR5ePickPlaceEnvEnhanced(**test_env_config)
        
        # Test episodes
        successes = 0
        total_rewards = []
        
        for episode in range(n_episodes):
            print(f"\n📺 Episode {episode + 1}/{n_episodes}")
            
            obs, info = env.reset()
            done = False
            episode_reward = 0
            step_count = 0
            
            # Log initial object perception and RGB status
            self._log_object_perception(env, obs, step_count, episode)
            self._validate_rgb_rendering(obs, f"Test episode {episode}")
            
            while not done and step_count < 500:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                episode_reward += reward
                step_count += 1
                
                # Log object perception every 50 steps
                if step_count % 50 == 0:
                    self._log_object_perception(env, obs, step_count, episode)
                
                # Render
                env.render()
                
                # Small delay for visualization
                if self.visual_training:
                    time.sleep(0.01)
                
            success = info.get('is_success', False)
            if success:
                successes += 1
                
            total_rewards.append(episode_reward)
                
            print(f"   Result: {'✅ Success' if success else '❌ Failed'}")
            print(f"   Steps: {step_count}")
            print(f"   Reward: {episode_reward:.2f}")
            
        print(f"\n📊 Test Results:")
        print(f"   Success rate: {successes}/{n_episodes} ({successes/n_episodes:.0%})")
        print(f"   Average reward: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
        print(f"   Reward range: [{np.min(total_rewards):.2f}, {np.max(total_rewards):.2f}]")
        
        env.close()
    
    def _log_object_perception(self, env, obs, step, episode):
        """Log what the CNN thinks it sees in the camera feed"""
        try:
            # Extract camera observations from the observation
            if hasattr(env, 'unwrapped'):
                unwrapped_env = env.unwrapped
            else:
                unwrapped_env = env
            
            # Get actual spawned object info
            if hasattr(unwrapped_env, 'current_object') and unwrapped_env.current_object:
                obj_name = unwrapped_env.current_object
                
                # Get object properties from environment
                if hasattr(unwrapped_env, 'object_body_ids'):
                    obj_id = unwrapped_env.object_body_ids.get(obj_name)
                    if obj_id is not None:
                        obj_pos = unwrapped_env.data.body(obj_id).xpos
                        obj_size = self._get_object_size(unwrapped_env, obj_id)
                        
                        # Check camera visibility
                        camera_sees = unwrapped_env._check_camera_sees_object() if hasattr(unwrapped_env, '_check_camera_sees_object') else "Unknown"
                        
                        # Analyze CNN input - extract camera portion of observation
                        cnn_analysis = self._analyze_visual_perception(obs)
                        
                        print(f"\n🔍 Object Perception Log - Episode {episode}, Step {step}")
                        print(f"   📦 Spawned Object:")
                        print(f"       Name: {obj_name}")
                        print(f"       Position: [{obj_pos[0]:.3f}, {obj_pos[1]:.3f}, {obj_pos[2]:.3f}]")
                        print(f"       Size: {obj_size}")
                        print(f"       Camera Sees: {camera_sees}")
                        print(f"   👁️ CNN Perception:")
                        print(f"{cnn_analysis}")
                        
        except Exception as e:
            print(f"   ⚠️ Object perception logging failed: {e}")
    
    def _get_object_size(self, env, obj_id):
        """Extract object size from MuJoCo model"""
        try:
            # Get geom associated with the body
            for geom_id in range(env.model.ngeom):
                if env.model.geom_bodyid[geom_id] == obj_id:
                    geom_size = env.model.geom_size[geom_id]
                    geom_type = env.model.geom_type[geom_id]
                    
                    if geom_type == 6:  # Box type
                        return f"box({geom_size[0]:.3f}x{geom_size[1]:.3f}x{geom_size[2]:.3f})"
                    else:
                        return f"geom_type_{geom_type}({geom_size[0]:.3f})"
                        
            return "size_unknown"
        except:
            return "size_error"
    
    def _analyze_visual_perception(self, obs: np.ndarray) -> str:
        """
        Analyzes the visual part of the observation vector (RGB and Depth).
        """
        try:
            cam_res = self.config["environment"]["camera_resolution"]
            
            if len(obs) < 56 + (cam_res * cam_res * 4):
                return "   ⚠️ Observation too short for camera analysis."
            
            # Define indices for RGB and Depth data
            rgb_start_idx = 55  # Kinematic data is 55 elements
            rgb_end_idx = rgb_start_idx + (cam_res * cam_res * 3)
            depth_start_idx = rgb_end_idx
            
            # Extract flattened RGB and Depth data
            rgb_obs = obs[rgb_start_idx:rgb_end_idx]
            depth_obs = obs[depth_start_idx:depth_start_idx + (cam_res * cam_res)]
            
            # Perform analysis on RGB data
            rgb_mean = np.mean(rgb_obs)
            rgb_std = np.std(rgb_obs)
            rgb_min, rgb_max = np.min(rgb_obs), np.max(rgb_obs)
            rgb_has_content = rgb_std > 0.01

            # Perform analysis on Depth data
            depth_mean = np.mean(depth_obs)
            depth_std = np.std(depth_obs)
            depth_min, depth_max = np.min(depth_obs), np.max(depth_obs)
            depth_has_content = depth_std > 0.01
            
            analysis_str = (
                f"      RGB Stats  : mean={rgb_mean:.3f}, std={rgb_std:.3f}, range=[{rgb_min:.3f}, {rgb_max:.3f}]\n"
                f"      Depth Stats: mean={depth_mean:.3f}, std={depth_std:.3f}, range=[{depth_min:.3f}, {depth_max:.3f}]\n"
                f"      Visual Content: RGB={'✅' if rgb_has_content else '❌'}, Depth={'✅' if depth_has_content else '❌'}"
            )
            
            if not rgb_has_content:
                analysis_str += "\n      ⚠️ WARNING: No RGB variation - visual learning impaired!"
                
            return analysis_str

        except Exception as e:
            return f"   ❌ Visual perception analysis failed: {e}"

    def _log_evaluation_perception(self):
        """Log object perception during evaluation phases"""
        try:
            print("\n📊 Evaluation Object Perception Summary:")
            
            # Get the environment from eval_env
            if hasattr(self.eval_env, 'envs') and len(self.eval_env.envs) > 0:
                env = self.eval_env.envs[0]
                
                # CRITICAL FIX: Don't reset eval environment during training - this interferes!
                # Only get current state without disrupting episodes
                # obs = self.eval_env.reset()  # REMOVED: This was causing episode reset interference
                obs = None  # Will skip logging if no current observation available
                
                # FIXED: Apply curriculum state AFTER reset to prevent initialization override
                if hasattr(self, 'curriculum_state_to_sync') and self.curriculum_state_to_sync:
                    try:
                        # Apply to the base environment after reset
                        if hasattr(env, 'curriculum_manager'):
                            env.curriculum_manager.current_phase = self.curriculum_state_to_sync['current_phase']
                            env.curriculum_manager.phase_progress = self.curriculum_state_to_sync['phase_progress']
                            env.curriculum_manager.phase_episode_count = self.curriculum_state_to_sync['phase_episode_count']
                            
                        # Apply curriculum level
                        if hasattr(env, 'set_curriculum_level'):
                            env.set_curriculum_level(self.curriculum_state_to_sync['curriculum_level'])
                            
                        print(f"✅ Curriculum state applied AFTER reset: {self.curriculum_state_to_sync['current_phase']} @ level {self.curriculum_state_to_sync['curriculum_level']:.3f}")
                    except Exception as e:
                        print(f"⚠️ Failed to apply curriculum state after reset: {e}")
                
                # CRITICAL FIX: Only log if we have observation without causing interference
                if obs is not None:
                    self._log_object_perception(env, obs[0], 0, "eval")
                    self._validate_rgb_rendering(obs[0], "Evaluation environment")
                else:
                    print("   Skipping eval perception logging to avoid training interference")
                
        except Exception as e:
            print(f"   ⚠️ Evaluation perception logging failed: {e}")
    
    def _log_final_training_stats(self):
        """Log comprehensive training completion statistics"""
        try:
            print(f"\n📊 Final Training Analysis:")
            
            # Policy network statistics
            if hasattr(self.model.policy, 'mlp_extractor'):
                # Get final action distribution statistics
                print(f"   🧠 Policy Network:")
                print(f"       Device: {self.model.device}")
                print(f"       Action space: {self.model.action_space}")
                
                # Get some sample predictions to analyze policy behavior
                obs = self.train_env.reset()
                action, _ = self.model.predict(obs, deterministic=False)
                action_det, _ = self.model.predict(obs, deterministic=True)
                
                print(f"       Sample stochastic action: {action}")
                print(f"       Sample deterministic action: {action_det}")
                print(f"       Action magnitude (stoch): {np.linalg.norm(action):.3f}")
                print(f"       Action magnitude (det): {np.linalg.norm(action_det):.3f}")
            
            # Environment statistics
            if hasattr(self.train_env, 'envs') and len(self.train_env.envs) > 0:
                env = self.train_env.envs[0]
                if hasattr(env, 'unwrapped'):
                    unwrapped = env.unwrapped
                    print(f"   🎮 Environment Stats:")
                    print(f"       Episode count: {getattr(unwrapped, 'episode_count', 'Unknown')}")
                    print(f"       Current curriculum level: {getattr(unwrapped, 'curriculum_level', 'Unknown')}")
                    
                    # Camera system validation
                    if hasattr(unwrapped, '_check_camera_sees_object'):
                        obs = self.train_env.reset()
                        camera_working = unwrapped._check_camera_sees_object()
                        print(f"       Camera system: {'✅ Working' if camera_working else '❌ Issues'}")
            
            print(f"   💾 Model saved to: {self.exp_dir}")
            print(f"   📈 View results: tensorboard --logdir {self.exp_dir / 'tensorboard'}")
            
        except Exception as e:
            print(f"   ⚠️ Final stats logging failed: {e}")
    
    def analyze_approach_learning_progress(self):
        """Analyze specific metrics for approach learning phase"""
        try:
            print(f"\n🎯 Approach Learning Analysis:")
            
            # Get current training state
            if hasattr(self.train_env, 'envs') and len(self.train_env.envs) > 0:
                env = self.train_env.envs[0]
                if hasattr(env, 'unwrapped'):
                    unwrapped = env.unwrapped
                    
                    # Sample some actions to see behavior patterns
                    obs = self.train_env.reset()
                    actions_sample = []
                    
                    for i in range(10):
                        action, _ = self.model.predict(obs, deterministic=True)
                        actions_sample.append(action[0])  # Unwrap from array
                        step_result = self.train_env.step(action)
                        if len(step_result) == 5:
                            obs, _, _, _, _ = step_result
                        else:
                            obs, _, _, _ = step_result
                    
                    # Analyze action patterns
                    actions_array = np.array(actions_sample)
                    joint_variability = np.std(actions_array, axis=0)
                    
                    print(f"   🤖 Action Analysis:")
                    print(f"       Action consistency: {np.mean(joint_variability):.3f}")
                    print(f"       Joint-wise variability: {joint_variability}")
                    print(f"       Dominant action magnitude: {np.mean(np.linalg.norm(actions_array, axis=1)):.3f}")
                    
                    # Check if model is showing approach-like behaviors
                    if np.mean(joint_variability) < 0.5:
                        print(f"       🎯 Status: Developing consistent strategies")
                    elif np.mean(joint_variability) > 1.0:
                        print(f"       🔄 Status: Still exploring broadly")
                    else:
                        print(f"       ⚖️ Status: Balanced exploration/exploitation")
                        
        except Exception as e:
            print(f"   ⚠️ Approach analysis failed: {e}")
    
    def predict_breakthrough_timeline(self, current_step: int):
        """Predict breakthrough milestones for the simplified 4-phase curriculum"""
        try:
            print(f"\n⏱️ Curriculum Timeline Prediction:")
            # Phase definitions (must match CurriculumManager)
            phases = [
                ("approach_learning", 80_000, 0.02, "Reach & consistent contact"),
                ("grasping", 160_000, 0.15, "Close, lift briefly"),
                ("manipulation", 240_000, 0.35, "Stable lift & move toward target"),
                ("mastery", 320_000, 0.60, "Reliable place & efficiency"),
            ]
            cumulative = 0
            current_phase_name = "complete"
            phase_progress = 1.0
            remaining = 0
            for name, steps, thresh, desc in phases:
                if current_step < cumulative + steps:
                    current_phase_name = name
                    phase_progress = (current_step - cumulative) / steps
                    remaining = (cumulative + steps) - current_step
                    break
                cumulative += steps
            total_curriculum = sum(p[1] for p in phases)
            print(f"   Step: {current_step:,} / {total_curriculum:,}")
            if current_phase_name != "complete":
                print(f"   Phase: {current_phase_name} ({phase_progress*100:.1f}% complete, {remaining:,} steps remaining)")
            else:
                print("   Phase: complete")
            # Milestone expectations (heuristic ranges)
            print("\n   🎯 Expected Milestones (typical ranges):")
            print("   - First consistent object approaches: 5k - 20k steps")
            print("   - First soft contacts (distance < 6cm): 10k - 30k steps")
            print("   - First accidental grasps (gripper closes near object): 90k - 140k steps")
            print("   - First stable grasp & lift (>5cm): 130k - 190k steps")
            print("   - First partial carry toward target: 220k - 300k steps")
            print("   - First successful placement (within 8cm radius): 260k - 360k steps")
            print("   - Consistent (>=30%) placements: 340k - 450k steps")
            print("   - Crossing 50% success: 500k - 650k steps (early mastery phase)")
            print("   - Approaching 60%+ stable success: 650k - 780k steps")
            # Phase advice
            if current_phase_name == "approach_learning":
                print("\n   🔍 Focus now: maximize progress reward (distance reduction). Look for rising 'progress' component and sporadic contact bonuses.")
            elif current_phase_name == "grasping":
                print("\n   🔍 Focus now: convert contacts into grasps. Expect grasp bonus frequency to start >0 after ~20% of this phase elapsed.")
            elif current_phase_name == "manipulation":
                print("\n   🔍 Focus now: stabilize post-grasp lift heights and begin lateral motion toward target.")
            elif current_phase_name == "mastery":
                print("\n   🔍 Focus now: reduce time & smoothness penalties; stabilize placement orientation.")
            # Early warning heuristics
            if current_step > 40_000 and current_phase_name == "approach_learning" and self.model.num_timesteps:  # safeguard
                print("\n   ⚠️ If no contact bonuses by 40k steps: consider increasing approach_bonus or reducing progress scale from 8.0 → 6.0.")
            if current_step > 140_000 and current_phase_name == "grasping":
                print("   ⚠️ If zero grasp bonuses by 140k steps: check gripper closing threshold & object spawn height.")
        except Exception as e:
            print(f"   ⚠️ Prediction failed: {e}")

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="UR5e Training with RGB Fix")
    parser.add_argument("--config", type=str, default="config_rtx4060_optimized.yaml", 
                       help="Path to config file")
    parser.add_argument("--visual", action="store_true", 
                       help="Enable visual training mode")
    parser.add_argument("--test", type=str, help="Path to model to test")
    parser.add_argument("--episodes", type=int, default=5, help="Test episodes")
    parser.add_argument("--verify", action="store_true", 
                       help="Run training verification test")
    
    args = parser.parse_args()

    print("🤖 UR5e Pick-Place Training System")
    print("🔧 Fixed for RGB rendering on Ubuntu")

    if args.verify:
        # Verification mode
        print("\n💡 For verification testing, please run:")
        print("   python simple_test.py")
        print("   (Basic environment test)")
        print("\n   python test_training_setup.py")
        print("   (Full training verification)")
        return
    elif args.test:
        # Test mode
        trainer = IntegratedTrainer(args.config, visual_training=args.visual)
        trainer.test_model(args.test, args.episodes)
    else:
        # Training mode
        trainer = IntegratedTrainer(args.config, visual_training=args.visual)
        trainer.train()

if __name__ == "__main__":
    main()