"""Does the vehicle still move after the env's soft reset (new_route)?"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402

config.set_config("SAFETY")

from carla_env.envs.carla_route_env import CarlaRouteEnv  # noqa: E402
from carla_env.rewards import reward_functions  # noqa: E402
from carla_env.state_commons import create_encode_state_fn  # noqa: E402
from config import CONFIG  # noqa: E402


def drive(env, tag, n=40, throttle=0.9, steer=0.0):
    speeds = []
    for _ in range(n):
        _, _, term, trunc, _ = env.step(np.array([steer, throttle], dtype=np.float32))
        speeds.append(env.unwrapped.vehicle.get_speed())
        if term or trunc:
            print(f"  [{tag}] terminated early at step {len(speeds)}")
            break
    print(f"  [{tag}] speed km/h: start={speeds[0]:.1f} end={speeds[-1]:.1f} max={max(speeds):.1f}")
    return max(speeds)


def main() -> None:
    obs_space, encode_fn, _ = create_encode_state_fn(None, CONFIG["state"])
    env = CarlaRouteEnv(obs_res=(160, 80), host="127.0.0.1", port=2000,
                        reward_fn=reward_functions[CONFIG["reward_fn"]],
                        observation_space=obs_space, encode_state_fn=encode_fn,
                        decode_vae_fn=None, fps=15, action_smoothing=0.75,
                        action_space_type="continuous", activate_spectator=False,
                        activate_render=False, start_carla=False,
                        town=CONFIG.get("map", "Town10HD_Opt"))
    v = env.unwrapped.vehicle
    print("simulate_physics after __init__ reset:",
          v.actor.get_physics_control() is not None, "| speed", v.get_speed())
    drive(env, "1st episode (after __init__ reset)")

    print("-- calling env.reset() (soft reset / new_route) --")
    env.reset(seed=1)
    drive(env, "2nd episode (after reset)")

    print("-- forcing set_simulate_physics(True) and retrying --")
    v.actor.set_simulate_physics(True)
    drive(env, "3rd attempt (physics re-enabled)")

    env.close()


if __name__ == "__main__":
    main()
