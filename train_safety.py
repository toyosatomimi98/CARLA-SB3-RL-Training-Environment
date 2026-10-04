"""Train a CARLA driving agent with (or without) the CBF safety shield.

Example
-------
python train_safety.py --config SAFETY --total_timesteps 20000 --tag shield
python train_safety.py --config SAFETY_UNSHIELDED --total_timesteps 20000 --tag baseline
"""
import os
import time
import warnings

warnings.filterwarnings("ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import argparse  # noqa: E402

import numpy as np  # noqa: E402
import gymnasium as gym  # noqa: E402

import config  # noqa: E402

parser = argparse.ArgumentParser(description="Trains a CARLA agent with a CBF safety shield")
parser.add_argument("--host", default="127.0.0.1", type=str)
parser.add_argument("--port", default=2000, type=int)
parser.add_argument("--total_timesteps", type=int, default=20_000)
parser.add_argument("--config", type=str, default="SAFETY")
parser.add_argument("--tag", type=str, default="shield")
parser.add_argument("--fps", type=int, default=15)
parser.add_argument("--num_checkpoints", type=int, default=4)
parser.add_argument("--max_episode_steps", type=int, default=400,
                    help="episode time limit (PORT: a 'stand still' policy would otherwise never end)")
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--no_render", action="store_false")
parser.add_argument("--no_rendering_mode", action="store_true",
                    help="tell CARLA not to render at all (vector-state configs only; "
                         "much faster, but cameras return no frames)")
args = vars(parser.parse_args())

config.set_config(args["config"])

from stable_baselines3 import PPO, DDPG, SAC  # noqa: E402
from stable_baselines3.common.callbacks import CheckpointCallback  # noqa: E402
from stable_baselines3.common.logger import configure  # noqa: E402

from carla_env.envs.carla_route_env import CarlaRouteEnv  # noqa: E402
from carla_env.rewards import reward_functions  # noqa: E402
from carla_env.safety.cbf import CBFFilterConfig  # noqa: E402
from carla_env.safety.wrapper import SafetyShieldWrapper  # noqa: E402
from carla_env.state_commons import create_encode_state_fn  # noqa: E402
from config import CONFIG  # noqa: E402
from utils import HParamCallback, write_json  # noqa: E402


def main() -> None:
    log_dir = os.environ.get("CARLA_SB3_LOG_DIR", "tensorboard")
    os.makedirs(log_dir, exist_ok=True)

    vae = None  # vector observations only (headless friendly)
    observation_space, encode_state_fn, decode_vae_fn = create_encode_state_fn(vae, CONFIG["state"])

    env = CarlaRouteEnv(
        obs_res=CONFIG["obs_res"], host=args["host"], port=args["port"],
        reward_fn=reward_functions[CONFIG["reward_fn"]],
        observation_space=observation_space,
        encode_state_fn=encode_state_fn, decode_vae_fn=decode_vae_fn,
        fps=args["fps"], action_smoothing=CONFIG["action_smoothing"],
        action_space_type="continuous", activate_spectator=False,
        activate_render=args["no_render"], start_carla=False,
        town=CONFIG.get("map", "Town10HD_Opt"),
        throttle_bias=CONFIG.get("throttle_bias", 0.0),
        weather=CONFIG.get("weather"),
        # PORT (2026-10-04): the trained policy must share the action space of
        # the evaluation / demo environments, where the throttle channel is
        # [-1, 1] so that the CBF filter has a brake to use.
        allow_brake=CONFIG.get("allow_brake", False),
        no_rendering=bool(args["no_rendering_mode"]),
    )

    safety_cfg = CONFIG.get("safety", {})
    results_dir = os.path.join("safety_results", args["tag"])
    os.makedirs(results_dir, exist_ok=True)
    cfg = CBFFilterConfig(
        dt=1.0 / args["fps"],
        lane_half_width=safety_cfg.get("lane_half_width", 1.75),
        lane_margin=safety_cfg.get("lane_margin", 0.30),
        speed_limit=safety_cfg.get("speed_limit", 13.9),
        a_lat_max=safety_cfg.get("a_lat_max", 2.5),
        horizon=safety_cfg.get("horizon", 10),
        alpha_lane=safety_cfg.get("alpha_lane", 0.30),
        alpha_speed=safety_cfg.get("alpha_speed", 0.30),
        alpha_curve=safety_cfg.get("alpha_curve", 0.30),
    )
    # PORT: episode time limit - without it a degenerate "stand still" policy
    # never terminates and the learner has no pressure to make progress.
    if args["max_episode_steps"] > 0:
        env = gym.wrappers.TimeLimit(env, max_episode_steps=args["max_episode_steps"])
    env = SafetyShieldWrapper(env, config=cfg,
                              penalty=safety_cfg.get("penalty", 0.0),
                              log_path=os.path.join(results_dir, "episodes.csv"),
                              enabled=safety_cfg.get("enabled", True))

    algo_cls = {"PPO": PPO, "DDPG": DDPG, "SAC": SAC}[CONFIG["algorithm"]]
    suffix = f"{args['tag']}_{int(time.time())}"
    model_dir = os.path.join(log_dir, f"{CONFIG['algorithm']}_{suffix}")
    model = algo_cls("MultiInputPolicy", env, verbose=1, seed=CONFIG["seed"],
                     tensorboard_log=log_dir, device=args["device"],
                     **CONFIG["algorithm_params"])
    model.set_logger(configure(model_dir, ["stdout", "csv", "tensorboard"]))
    write_json(CONFIG, os.path.join(model_dir, "config.json"))

    checkpoint_freq = max(args["total_timesteps"] // args["num_checkpoints"], 1000)
    print(f"[train] algorithm={CONFIG['algorithm']} shield={safety_cfg.get('enabled')} "
          f"steps={args['total_timesteps']} dir={model_dir}")
    t0 = time.time()
    model.learn(total_timesteps=args["total_timesteps"],
                callback=[HParamCallback(CONFIG),
                          CheckpointCallback(save_freq=checkpoint_freq,
                                             save_path=model_dir, name_prefix="model")],
                reset_num_timesteps=False)
    model.save(os.path.join(model_dir, "final_model"))
    dt = time.time() - t0
    stats = env.safety_stats()
    print(f"[train] done in {dt:.0f}s ({args['total_timesteps'] / max(dt, 1):.1f} steps/s)")
    print("[train] safety stats:", {k: (round(v, 4) if isinstance(v, float) else v)
                                    for k, v in stats.items()})
    write_json({**stats, "wall_seconds": dt, "total_timesteps": args["total_timesteps"]},
               os.path.join(model_dir, "safety_stats.json"))
    env.close()


if __name__ == "__main__":
    main()
