"""Print what the trained policy actually outputs (deterministic vs sampled)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402

config.set_config("SAFETY_PPO_DRIVE")

import gymnasium as gym  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402
from carla_env.envs.carla_route_env import CarlaRouteEnv  # noqa: E402
from carla_env.rewards import reward_functions  # noqa: E402
from carla_env.state_commons import create_encode_state_fn  # noqa: E402
from config import CONFIG  # noqa: E402

model_path = sys.argv[1]

obs_space, encode_fn, _ = create_encode_state_fn(None, CONFIG["state"])
env = CarlaRouteEnv(obs_res=CONFIG["obs_res"], host="127.0.0.1", port=2000,
                    reward_fn=reward_functions[CONFIG["reward_fn"]],
                    observation_space=obs_space, encode_state_fn=encode_fn,
                    decode_vae_fn=None, fps=15, action_smoothing=CONFIG["action_smoothing"],
                    action_space_type="continuous", activate_spectator=False,
                    activate_render=False, start_carla=False, town=CONFIG["map"])
env = gym.wrappers.TimeLimit(env, max_episode_steps=400)
model = PPO.load(model_path, env=env, device="cuda")

print("policy log_std:", model.policy.log_std.detach().cpu().numpy())
obs, _ = env.reset(seed=7)
for i in range(8):
    det, _ = model.predict(obs, deterministic=True)
    sto, _ = model.predict(obs, deterministic=False)
    obs, rew, term, trunc, info = env.step(det)
    print(f"step {i}: det={np.round(det, 3)} sampled={np.round(sto, 3)} "
          f"-> v={env.unwrapped.vehicle.get_speed():5.2f} km/h rew={rew:+.3f}")
print("-- now driving with the SAMPLED action --")
obs, _ = env.reset(seed=7)
for i in range(8):
    sto, _ = model.predict(obs, deterministic=False)
    obs, rew, term, trunc, info = env.step(sto)
    print(f"step {i}: sampled={np.round(sto, 3)} -> v={env.unwrapped.vehicle.get_speed():5.2f} km/h rew={rew:+.3f}")
env.close()
