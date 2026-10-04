"""Evaluate every checkpoint of a run (deterministic) and report which one drives.

A "safe policy" that does not move is a failure, so the ranking is driven by
distance travelled / mean speed first, then by lane keeping.
"""
import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--run", default="", help="tensorboard/PPO_<tag>_<id> directory")
ap.add_argument("--config", default="SAFETY_PPO_DRIVE")
ap.add_argument("--steps", type=int, default=200)
args = ap.parse_args()

config.set_config(args.config)

import gymnasium as gym  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402
from carla_env.envs.carla_route_env import CarlaRouteEnv  # noqa: E402
from carla_env.rewards import reward_functions  # noqa: E402
from carla_env.safety.cbf import CBFFilterConfig  # noqa: E402
from carla_env.safety.wrapper import SafetyShieldWrapper  # noqa: E402
from carla_env.state_commons import create_encode_state_fn  # noqa: E402
from config import CONFIG  # noqa: E402


def main() -> None:
    run = args.run
    if not run:
        cands = sorted(glob.glob(os.path.join("tensorboard", "PPO_drive3_*")),
                       key=os.path.getmtime, reverse=True)
        run = cands[0]
    zips = sorted(glob.glob(os.path.join(run, "*.zip")),
                  key=lambda p: (os.path.getsize(p), p))
    print("run:", run)
    print("checkpoints:", [os.path.basename(z) for z in zips])

    obs_space, encode_fn, _ = create_encode_state_fn(None, CONFIG["state"])
    sc = CONFIG["safety"]
    cfg = CBFFilterConfig(dt=1 / 15.0, lane_half_width=sc["lane_half_width"],
                          lane_margin=sc["lane_margin"], speed_limit=sc["speed_limit"],
                          a_lat_max=sc["a_lat_max"], horizon=sc["horizon"])
    best = None
    for z in zips:
        env = CarlaRouteEnv(obs_res=CONFIG["obs_res"], host="127.0.0.1", port=2000,
                            reward_fn=reward_functions[CONFIG["reward_fn"]],
                            observation_space=obs_space, encode_state_fn=encode_fn,
                            decode_vae_fn=None, fps=15,
                            action_smoothing=CONFIG["action_smoothing"],
                            action_space_type="continuous", activate_spectator=False,
                            activate_render=False, start_carla=False, town=CONFIG["map"])
        env = gym.wrappers.TimeLimit(env, max_episode_steps=args.steps)
        env = SafetyShieldWrapper(env, config=cfg, penalty=0.0, enabled=True)
        model = PPO.load(z, env=env, device="cuda")
        obs, _ = env.reset(seed=11)
        v, ey = [], []
        for _ in range(args.steps):
            a, _ = model.predict(obs, deterministic=True)
            obs, _, term, trunc, info = env.step(a)
            s = info.get("safety", {})
            v.append(s.get("speed_kmh", 0.0))
            ey.append(abs(s.get("lateral_offset", 0.0)))
            if term or trunc:
                break
        dist = float(np.sum(v) / 3.6) / 15.0
        row = dict(ckpt=os.path.basename(z), steps=len(v), mean_v=round(float(np.mean(v)), 2),
                   max_v=round(float(np.max(v)), 1), dist_m=round(dist, 1),
                   max_ey=round(float(np.max(ey)), 2))
        print(f"  {row['ckpt']:24s} steps={row['steps']:4d} mean_v={row['mean_v']:6.2f} "
              f"max_v={row['max_v']:5.1f} dist={row['dist_m']:6.1f}m max|e_y|={row['max_ey']:.2f}")
        env.close()
        score = row["dist_m"] + 10 * row["mean_v"] - 5 * max(0.0, row["max_ey"] - 1.75)
        if best is None or score > best[0]:
            best = (score, z, row)
    print("\nBEST checkpoint:", best[1], best[2])


if __name__ == "__main__":
    main()
