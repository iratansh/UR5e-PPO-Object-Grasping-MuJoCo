import os
import sys
import argparse
import numpy as np
import torch
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from homestri_ur5e_rl.envs import UR5ePickPlaceEnvEnhanced  # noqa: E402

print(f"[DEBUG] Environment class imported from: {UR5ePickPlaceEnvEnhanced.__module__}")
print(f"[DEBUG] Environment file path: {UR5ePickPlaceEnvEnhanced.__file__ if hasattr(UR5ePickPlaceEnvEnhanced, '__file__') else 'No __file__ attr'}")
import inspect
print(f"[DEBUG] Environment step method source file: {inspect.getfile(UR5ePickPlaceEnvEnhanced.step)}")

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
except ImportError:
    PPO = None
    BaseCallback = object


def set_global_seed(seed: int = 0):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class InfoProbeCallback(BaseCallback):
    """Lightweight callback to sample and print info key presence during PPO training."""
    def __init__(self, probe_freq: int = 1024):
        super().__init__()
        self.probe_freq = probe_freq
        self._seen_keys = set()
        self._missing_counter = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if isinstance(info, dict):
                self._seen_keys.update(info.keys())
                rc = info.get('reward_components')
                if isinstance(rc, dict):
                    if len(rc) == 0:
                        self._missing_counter += 1
                else:
                    self._missing_counter += 1
        if self.num_timesteps and self.num_timesteps % self.probe_freq == 0:
            print(f"[INFO-PROBE] t={self.num_timesteps} seen_info_keys={sorted(list(self._seen_keys))[:12]} rc_missing_events={self._missing_counter}")
        return True


def run_random(env, episodes: int):
    print("Running random policy episodes...")
    for ep in range(1, episodes + 1):
        obs, info = env.reset()
        done = False
        ep_reward = 0.0
        min_planar = np.inf
        vertical_series = []
        approach_events = 0
        contact_events = 0
        steps = 0
        while not done and steps < 600:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_reward += reward
            rc = info.get('reward_components', {})
            pd = info.get('planar_distance', rc.get('planar_dist')) if isinstance(rc, dict) else None
            vs = info.get('vertical_signed', rc.get('vertical_signed')) if isinstance(rc, dict) else None
            if pd is not None and np.isfinite(pd):
                min_planar = min(min_planar, pd) if np.isfinite(min_planar) else pd
            if vs is not None:
                vertical_series.append(vs)
            approach_events = info.get('approach_events', approach_events)
            contact_events = info.get('contact_events', contact_events)
            steps += 1
        vert_median = float(np.median(vertical_series)) if vertical_series else np.nan
        mp = f"{min_planar:.3f}" if np.isfinite(min_planar) else "none"
        print(f"[RANDOM] Ep {ep:02d} steps={steps} reward={ep_reward:.2f} min_planar={mp} approach={approach_events} contact={contact_events} vert_med={vert_median:.3f}")


def run_ppo(env, timesteps: int, episodes_preview: int, device: str):
    if PPO is None:
        print("stable-baselines3 not installed; cannot run PPO")
        return
    policy_kwargs = dict(net_arch=dict(pi=[128, 128], vf=[128, 128]))
    model = PPO(
        "MlpPolicy", env,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=256,
        n_epochs=5,
        gamma=0.995,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        device=device,
        policy_kwargs=policy_kwargs,
    )
    probe = InfoProbeCallback()
    print("Starting minimal PPO training...")
    model.learn(total_timesteps=timesteps, callback=probe, progress_bar=False)

    # Post-training evaluation episodes (deterministic)
    print("\nEvaluating learned policy (deterministic)...")
    for ep in range(1, episodes_preview + 1):
        obs, info = env.reset()
        done = False
        ep_reward = 0.0
        min_planar = np.inf
        vertical_series = []
        steps = 0
        approach_events = 0
        contact_events = 0
        while not done and steps < 600:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_reward += reward
            rc = info.get('reward_components', {})
            pd = info.get('planar_distance', rc.get('planar_dist')) if isinstance(rc, dict) else None
            vs = info.get('vertical_signed', rc.get('vertical_signed')) if isinstance(rc, dict) else None
            if pd is not None and pd < min_planar:
                min_planar = pd
            if vs is not None:
                vertical_series.append(vs)
            approach_events = info.get('approach_events', approach_events)
            contact_events = info.get('contact_events', contact_events)
            steps += 1
        vert_median = float(np.median(vertical_series)) if vertical_series else np.nan
        keys_preview = list(rc.keys())[:10] if isinstance(rc, dict) else []
        print(f"[PPO-EVAL] Ep {ep:02d} steps={steps} reward={ep_reward:.2f} min_planar={min_planar if np.isfinite(min_planar) else 'nan'} approach={approach_events} contact={contact_events} vert_med={vert_median:.3f} rc_keys={keys_preview}")


def main():
    parser = argparse.ArgumentParser(description="Minimal training debug for UR5e env")
    parser.add_argument('--episodes-random', type=int, default=5, help='Random policy test episodes')
    parser.add_argument('--ppo-timesteps', type=int, default=5000, help='Minimal PPO training timesteps')
    parser.add_argument('--ppo-eval-episodes', type=int, default=2, help='Evaluation episodes after PPO')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--no-ppo', action='store_true', help='Skip PPO training (just random)')
    args = parser.parse_args()

    set_global_seed(args.seed)

    # Minimal env config: disable domain randomization for determinism & speed
    env = UR5ePickPlaceEnvEnhanced(
        xml_file='custom_scene.xml',
        camera_resolution=64,
        control_mode='joint',
        use_stuck_detection=True,
        use_domain_randomization=False,
        frame_skip=5,
        curriculum_level=0.1,
        render_mode=None,
    )

    # Random rollouts first
    run_random(env, args.episodes_random)

    if not args.no_ppo:
        run_ppo(env, args.ppo_timesteps, args.ppo_eval_episodes, args.device)

    env.close()
    print("Done.")


if __name__ == '__main__':
    main()
