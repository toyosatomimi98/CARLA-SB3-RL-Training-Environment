"""Compare the old (+k*ang) and the corrected (-k*ang) reference controller."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config  # noqa: E402

config.set_config("SAFETY_PPO_SPEED")

import gymnasium as gym  # noqa: E402
from carla_env.envs.carla_route_env import CarlaRouteEnv  # noqa: E402
from carla_env.rewards import reward_functions  # noqa: E402
from carla_env.state_commons import create_encode_state_fn  # noqa: E402
from carla_env.safety.cbf import CBFFilterConfig  # noqa: E402
from carla_env.safety.wrapper import SafetyShieldWrapper  # noqa: E402
from config import CONFIG  # noqa: E402
from carla_env.wrappers import vector  # noqa: E402


def run(sign: float, steps: int = 330, seed: int = 33):
    obs_space, encode_fn, _ = create_encode_state_fn(None, CONFIG["state"])
    env = CarlaRouteEnv(obs_res=CONFIG["obs_res"], host="127.0.0.1", port=2000,
                        reward_fn=reward_functions[CONFIG["reward_fn"]],
                        observation_space=obs_space, encode_state_fn=encode_fn,
                        decode_vae_fn=None, fps=15, action_smoothing=0.0,
                        action_space_type="continuous", activate_spectator=False,
                        activate_render=False, start_carla=False, town=CONFIG["map"],
                        allow_brake=True)
    env = gym.wrappers.TimeLimit(env, max_episode_steps=steps)
    env = SafetyShieldWrapper(env, config=CBFFilterConfig(dt=1 / 15.0),
                              penalty=0.0, enabled=False)  # metrics only, no filtering
    env.reset(seed=seed)
    u = env.unwrapped
    eys, steers, speeds = [], [], []
    for k in range(steps):
        ang = float(u.vehicle.get_angle(u.current_waypoint))
        steer = float(np.clip(sign * 1.4 * ang, -1, 1))
        throttle = float(np.clip((16.0 - u.vehicle.get_speed()) / 14.0, 0.0, 1.0))
        obs, rew, term, trunc, info = env.step(np.array([steer, throttle], dtype=np.float32))
        # true signed lateral offset w.r.t. the *current* lane (curve-aware)
        eys.append(float(env.ego_state().e_y))
        steers.append(steer)
        speeds.append(u.vehicle.get_speed())
        if term or trunc:
            break
    env.close()
    return dict(steps=len(eys), mean_ey=float(np.mean(eys)), max_abs_ey=float(np.max(np.abs(eys))),
                final_ey=float(eys[-1]), mean_steer=float(np.mean(steers)),
                mean_speed=float(np.mean(speeds)))


if __name__ == "__main__":
    for sign, tag in [(+1.0, "OLD  steer=+k*ang (positive feedback)"),
                      (-1.0, "FIXED steer=-k*ang (negative feedback)")]:
        r = run(sign)
        print(f"{tag}: steps={r['steps']} mean_e_y={r['mean_ey']:+.2f} m "
              f"max|e_y|={r['max_abs_ey']:.2f} m final_e_y={r['final_ey']:+.2f} m "
              f"mean_steer={r['mean_steer']:+.3f} mean_speed={r['mean_speed']:.1f} km/h", flush=True)
