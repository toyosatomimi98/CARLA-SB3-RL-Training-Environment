"""Evaluate a trained policy with the CBF shield ON or OFF.

The *same* policy is rolled out twice so that the effect of the safety filter is
isolated from the effect of learning.  Writes a CSV of per-episode metrics.
"""
import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)
parser.add_argument("--config", default="SAFETY_RL",
                    help="must be the config the checkpoint was trained with")
parser.add_argument("--algo", default="PPO", choices=["PPO", "SAC", "DDPG"],
                    help="algorithm used to train the checkpoint (must match the .zip)")
parser.add_argument("--episodes", type=int, default=3)
parser.add_argument("--steps", type=int, default=600)
parser.add_argument("--shield", type=int, default=1)
parser.add_argument("--out", default="")
parser.add_argument("--seed", type=int, default=123)
parser.add_argument("--no_rendering_mode", action="store_true",
                    help="headless server without rendering (vector-state configs only)")
args = parser.parse_args()

config.set_config(args.config)

from stable_baselines3 import PPO, SAC, DDPG  # noqa: E402
from carla_env.envs.carla_route_env import CarlaRouteEnv  # noqa: E402
from carla_env.rewards import reward_functions  # noqa: E402
from carla_env.safety.cbf import CBFFilterConfig  # noqa: E402
from carla_env.safety.wrapper import SafetyShieldWrapper  # noqa: E402
from carla_env.state_commons import create_encode_state_fn  # noqa: E402
from config import CONFIG  # noqa: E402


def main() -> None:
    obs_space, encode_fn, _ = create_encode_state_fn(None, CONFIG["state"])
    env = CarlaRouteEnv(
        obs_res=CONFIG["obs_res"], host="127.0.0.1", port=2000,
        reward_fn=reward_functions[CONFIG["reward_fn"]],
        observation_space=obs_space, encode_state_fn=encode_fn, decode_vae_fn=None,
        fps=15, action_smoothing=CONFIG["action_smoothing"],
        action_space_type="continuous", activate_spectator=False,
        activate_render=False, start_carla=False, town=CONFIG.get("map", "Town10HD_Opt"),
        # must match training, otherwise the policy is evaluated in a different
        # action space than it was trained in
        allow_brake=CONFIG.get("allow_brake", False),
        no_rendering=bool(args.no_rendering_mode),
    )
    sc = CONFIG.get("safety", {})
    cfg = CBFFilterConfig(dt=1 / 15.0,
                          lane_half_width=sc.get("lane_half_width", 1.75),
                          lane_margin=sc.get("lane_margin", 0.30),
                          speed_limit=sc.get("speed_limit", 13.9),
                          a_lat_max=sc.get("a_lat_max", 2.5),
                          horizon=sc.get("horizon", 10))
    import gymnasium as gym
    env = gym.wrappers.TimeLimit(env, max_episode_steps=args.steps)
    env = SafetyShieldWrapper(env, config=cfg, penalty=0.0, enabled=bool(args.shield))

    algo = {"PPO": PPO, "SAC": SAC, "DDPG": DDPG}[args.algo]
    model = algo.load(args.model, env=env, device="cuda")

    rows = []
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        ep_min_h, ep_viol, ep_int, ep_coll, ep_dist, ep_rew = np.inf, 0, 0, 0, 0.0, 0.0
        speeds, eys = [], []
        for _ in range(args.steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, rew, term, trunc, info = env.step(action)
            s = info.get("safety", {})
            mb = s.get("min_barrier", np.nan)
            if np.isfinite(mb):
                ep_min_h = min(ep_min_h, mb)
                ep_viol += int(mb < 0)
            ep_int += int(s.get("intervened", False))
            speeds.append(s.get("speed_kmh", np.nan))
            eys.append(abs(s.get("lateral_offset", np.nan)))
            ep_rew += float(rew)
            ep_dist = float(info.get("total_distance", ep_dist))
            if term:
                ep_coll += int(bool(getattr(env.unwrapped, "collision_flag", False)))
                break
            if trunc:
                break
        row = dict(episode=ep, shield=int(args.shield), min_barrier=round(ep_min_h, 4),
                   violation_steps=ep_viol, interventions=ep_int, collisions=ep_coll,
                   distance_m=round(ep_dist, 2), reward=round(ep_rew, 3),
                   mean_speed_kmh=round(float(np.nanmean(speeds)), 2),
                   max_abs_lateral_m=round(float(np.nanmax(eys)), 3))
        rows.append(row)
        print("[eval]", row, flush=True)
    env.close()

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    print("[eval] mean:", {k: round(float(np.mean([r[k] for r in rows])), 3)
                           for k in ("min_barrier", "violation_steps", "interventions",
                                     "collisions", "distance_m", "max_abs_lateral_m")})


if __name__ == "__main__":
    main()
