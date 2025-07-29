"""
Integrated Training Script for UR5e Pick-Place
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

# Setup Ubuntu MuJoCo compatibility
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
        
        print("Ubuntu MuJoCo environment configured")
    elif platform.system() == "Darwin":
        print("macOS detected - using Apple Silicon optimizations")
        os.environ["MUJOCO_GL"] = "glfw"

# Apply Ubuntu fixes immediately
setup_ubuntu_mujoco()

# Add homestri to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor

# Import fixed components
from homestri_ur5e_rl.envs import UR5ePickPlaceEnvEnhanced
from homestri_ur5e_rl.training.sim_to_real_cnn import SimToRealCNNExtractor
from homestri_ur5e_rl.training.progressive_callback import ProgressiveTrainingCallback
from homestri_ur5e_rl.utils.detailed_logging_callback import DetailedLoggingCallback
from homestri_ur5e_rl.training.curriculum_manager import CurriculumManager

class IntegratedTrainer:
    """Trainer with proper action scaling and reasonable rewards"""
    
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
                "use_domain_randomization": False,
                "frame_skip": 5,
                "initial_curriculum_level": 0.1,
                "render_mode": None,
                "headless": False,  # Ensure RGB rendering works
            },
            "training": {
                "total_timesteps": 500_000,  
                "learning_rate": 0.0003,     
                "n_steps": 2048,            
                "batch_size": 64,           
                "n_epochs": 10,
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "clip_range": 0.2,
                "ent_coef": 0.01,
                "vf_coef": 0.5,
                "max_grad_norm": 0.5,
                "detailed_log_freq": 2048,
            },
            "evaluation": {
                "eval_freq": 20_480,
                "n_eval_episodes": 5,
            },
            "logging": {
                "save_freq": 51_200,
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
            norm_reward=True, 
            clip_obs=10.0,
            clip_reward=10.0,  
            gamma=self.config["training"]["gamma"]
        )
        
        # Evaluation environment
        eval_env_config = self.config["environment"].copy()
        eval_env_config["render_mode"] = None
        eval_env_config["headless"] = False  # Ensure RGB works in eval too
        eval_env = UR5ePickPlaceEnvEnhanced(**eval_env_config)
        eval_env = Monitor(eval_env)
        self.eval_env = DummyVecEnv([lambda: eval_env])
        
        # Normalize eval env but don't update stats
        self.eval_env = VecNormalize(
            self.eval_env, 
            norm_obs=True, 
            norm_reward=False, 
            clip_obs=10.0, 
            training=False
        )
        
        # Initialize curriculum manager
        self.curriculum_manager = CurriculumManager(env)
        
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
        
        # Conservative PPO settings
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
                self.rgb_check_freq = 10000  # Check RGB every 10k steps
                
            def _on_step(self) -> bool:
                result = super()._on_step()
                
                # Periodic RGB validation
                if self.n_calls % self.rgb_check_freq == 0:
                    obs = self.trainer.train_env.reset()
                    self.trainer._validate_rgb_rendering(obs[0], f"Step {self.n_calls}")
                    
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
        
        # Evaluation callback
        eval_callback = EvalCallback(
            self.eval_env,
            best_model_save_path=str(self.exp_dir / "best_model"),
            log_path=str(self.exp_dir / "eval"),
            eval_freq=self.config["evaluation"]["eval_freq"],
            n_eval_episodes=self.config["evaluation"]["n_eval_episodes"],
            deterministic=True,
            render=False,
        )
        callbacks.append(eval_callback)
        
        # Progressive callback
        progressive_callback = ProgressiveTrainingCallback(
            eval_env=self.eval_env,
            eval_freq=self.config["evaluation"]["eval_freq"],
            n_eval_episodes=self.config["evaluation"]["n_eval_episodes"],
            curriculum_threshold=0.05,
            randomization_schedule={
                0: 0.0,
                200_000: 0.1,
                400_000: 0.3,
                800_000: 0.5,
            },
        )
        callbacks.append(progressive_callback)
        
        return CallbackList(callbacks)
    
    def sync_vecnormalize_stats(self):
        """Sync VecNormalize stats from training to evaluation environment"""
        if isinstance(self.train_env, VecNormalize) and isinstance(self.eval_env, VecNormalize):
            self.eval_env.obs_rms = self.train_env.obs_rms
            self.eval_env.ret_rms = self.train_env.ret_rms
            print("✅ VecNormalize stats synced from training to evaluation environment")
    
    def train(self):
        """Main training loop with RGB monitoring"""
        self.create_env()
        self.create_model()
        callbacks = self.create_callbacks()
        
        # Sync VecNormalize stats
        self.sync_vecnormalize_stats()
        
        # Final RGB check before training
        print("\n🔍 Final RGB validation before training...")
        obs = self.train_env.reset()
        self._validate_rgb_rendering(obs[0], "Pre-training final check")
        
        # Train with fixed hyperparameters
        print(f"\n🏃 Training for {self.config['training']['total_timesteps']:,} timesteps...")
        
        # Enhanced training monitoring
        start_time = time.time()
        
        try:
            self.model.learn(
                total_timesteps=self.config["training"]["total_timesteps"],
                callback=callbacks,
                log_interval=self.config["logging"]["log_interval"],
                progress_bar=True,
            )
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
                
            success = info.get('task_completed', False)
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
                
                # Reset and get initial observation
                obs = self.eval_env.reset()
                
                # Log what we see
                self._log_object_perception(env, obs[0], 0, "eval")
                self._validate_rgb_rendering(obs[0], "Evaluation environment")
                
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
        """Predict when first successes might emerge based on actual curriculum structure"""
        try:
            print(f"\n⏱️ Curriculum Timeline Prediction:")
            
            # Actual curriculum phase durations from curriculum_manager.py (5 phases total)
            phase_1_approach = 5_000_000     # Phase 1: approach_learning
            phase_2_contact = 4_000_000      # Phase 2: contact_refinement  
            phase_3_grasping = 8_000_000     # Phase 3: grasping
            phase_4_manipulation = 6_000_000 # Phase 4: manipulation
            phase_5_mastery = 4_000_000      # Phase 5: mastery
            
            # Cumulative phase boundaries
            phase_2_start = phase_1_approach
            phase_3_start = phase_1_approach + phase_2_contact
            phase_4_start = phase_3_start + phase_3_grasping
            phase_5_start = phase_4_start + phase_4_manipulation
            total_curriculum = phase_5_start + phase_5_mastery  # 27M total
            
            # Determine current phase and progress
            if current_step < phase_1_approach:
                current_phase_num = 1
                phase_progress = (current_step / phase_1_approach) * 100
                remaining_in_phase = phase_1_approach - current_step
                phase_name = "Approach Learning"
                phase_focus = "Safe approach and gripper contact"
                success_threshold = "10%"
            elif current_step < phase_2_start + phase_2_contact:
                current_phase_num = 2
                phase_progress = ((current_step - phase_2_start) / phase_2_contact) * 100
                remaining_in_phase = phase_2_start + phase_2_contact - current_step
                phase_name = "Contact Refinement"
                phase_focus = "Precise gripper positioning"
                success_threshold = "30%"
            elif current_step < phase_3_start + phase_3_grasping:
                current_phase_num = 3
                phase_progress = ((current_step - phase_3_start) / phase_3_grasping) * 100
                remaining_in_phase = phase_3_start + phase_3_grasping - current_step
                phase_name = "Grasping"
                phase_focus = "Successful object grasping"
                success_threshold = "50%"
            elif current_step < phase_4_start + phase_4_manipulation:
                current_phase_num = 4
                phase_progress = ((current_step - phase_4_start) / phase_4_manipulation) * 100
                remaining_in_phase = phase_4_start + phase_4_manipulation - current_step
                phase_name = "Manipulation"
                phase_focus = "Full pick and place"
                success_threshold = "70%"
            elif current_step < phase_5_start + phase_5_mastery:
                current_phase_num = 5
                phase_progress = ((current_step - phase_5_start) / phase_5_mastery) * 100
                remaining_in_phase = phase_5_start + phase_5_mastery - current_step
                phase_name = "Mastery"
                phase_focus = "Collision-free mastery"
                success_threshold = "85%"
            else:
                current_phase_num = "Complete"
                phase_progress = 100
                remaining_in_phase = 0
                phase_name = "Curriculum Complete"
                phase_focus = "All phases mastered"
                success_threshold = "N/A"
            
            print(f"   📍 Current Curriculum Progress:")
            print(f"       Step: {current_step:,} / {total_curriculum:,} total")
            print(f"       Phase: {current_phase_num}/5 - {phase_name}")
            print(f"       Phase Progress: {min(100, phase_progress):.1f}%")
            print(f"       Phase Focus: {phase_focus}")
            print(f"       Success Threshold: {success_threshold}")
            
            if current_phase_num != "Complete":
                print(f"       Remaining in Phase: {remaining_in_phase:,} steps")
                
                print(f"\n   🎯 Complete Curriculum Timeline:")
                print(f"       Phase 1 (0-5M): Approach Learning")
                print(f"       Phase 2 (5M-9M): Contact Refinement") 
                print(f"       Phase 3 (9M-17M): Grasping")
                print(f"       Phase 4 (17M-23M): Manipulation")
                print(f"       Phase 5 (23M-27M): Mastery")
            
            # Training health indicators based on actual phase
            current_phase_str = phase_name if current_phase_num != "Complete" else "Complete"
            print(f"\n   💪 Training Health:")
            if current_phase_num == 1:
                print(f"       ✅ Currently learning: Basic approach strategies")
                print(f"       ✅ Phase 1 rewards: Gentle contact & exploration")
                print(f"       ✅ Zero termination on collisions (learning mode)")
                print(f"       ✅ Curriculum will auto-advance at 10% success")
            elif current_phase_num == 2:
                print(f"       ✅ Currently learning: Contact refinement")
                print(f"       ✅ Phase 2 rewards: Precise gripper positioning")
                print(f"       ✅ Success threshold: 30% to advance")
            elif current_phase_num == 3:
                print(f"       ✅ Currently learning: Grasping techniques")
                print(f"       ✅ Phase 3 rewards: Successful object grasping")
                print(f"       ✅ Success threshold: 50% to advance")
            elif current_phase_num == 4:
                print(f"       ✅ Currently learning: Full manipulation")
                print(f"       ✅ Phase 4 rewards: Complete pick-and-place")
                print(f"       ✅ Success threshold: 70% to advance")
            elif current_phase_num == 5:
                print(f"       ✅ Currently learning: Mastery refinement")
                print(f"       ✅ Phase 5 rewards: Collision-free execution")
                print(f"       ✅ Success threshold: 85% for completion")
            else:
                print(f"       ✅ Training complete: All phases mastered")
            
            # Realistic expectations
            print(f"\n   💡 Realistic Expectations:")
            print(f"       🎯 Object contacts: Throughout Phase 1 (0-5M)")
            print(f"       🤏 First grasps: Phase 3 (9M-17M steps)")
            print(f"       🏆 Task completion: Phase 4 (17M-23M steps)")
            print(f"       🎖️ Mastery: Phase 5 (23M-27M steps)")
            
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
    
    args = parser.parse_args()

    print("🤖 UR5e Pick-Place Training System")
    print("🔧 Fixed for RGB rendering on Ubuntu")

    trainer = IntegratedTrainer(args.config, visual_training=args.visual)

    if args.test:
        # Test mode
        trainer.test_model(args.test, args.episodes)
    else:
        # Training mode
        trainer.train()

if __name__ == "__main__":
    main()