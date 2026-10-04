"""Port smoke test: build CarlaRouteEnv against a running CARLA 0.10.0 server
and roll out a few random actions.  No VAE, no rendering (headless)."""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402

config.set_config("2")  # SAC, state 3, reward_fn5

from carla_env.rewards import reward_functions  # noqa: E402
from carla_env.state_commons import create_encode_state_fn  # noqa: E402
from carla_env.envs.carla_route_env import CarlaRouteEnv  # noqa: E402

STATE = ["steer", "throttle", "speed", "maneuver"]


def main() -> None:
    def log(*a):
        print(*a, flush=True)

    obs_space, encode_fn, _ = create_encode_state_fn(None, STATE)
    log("observation_space:", obs_space)
    log("[1] constructing env (loads map, spawns ego + camera, resets)...")
    t = time.time()
    env = CarlaRouteEnv(
        host="127.0.0.1", port=2000,
        obs_res=(160, 80),
        reward_fn=reward_functions["reward_fn5"],
        observation_space=obs_space,
        encode_state_fn=encode_fn,
        decode_vae_fn=None,
        fps=15, action_smoothing=0.75,
        action_space_type="continuous",
        activate_spectator=False,
        activate_render=False,
        start_carla=False,
        town="Town10HD_Opt",
    )
    log(f"[2] env ready in {time.time() - t:.1f}s")
    log("map:", env.world.map.name, "spawn points:", len(env.world.map.get_spawn_points()))
    log("route waypoints:", len(env.route_waypoints))
    rng = np.random.RandomState(0)
    t0 = time.time()
    total = 0.0
    for i in range(30):
        action = np.array([rng.uniform(-0.3, 0.3), rng.uniform(0.3, 0.6)], dtype=np.float32)
        obs, rew, terminated, truncated, info = env.step(action)
        total += rew
        if i % 5 == 0:
            log(f"  step {i:3d} rew={rew:+.3f} v={env.vehicle.get_speed():5.1f}km/h "
                f"dev={env.distance_from_center:4.2f}m wp={env.current_waypoint_index}/{len(env.route_waypoints)}")
        if terminated or truncated:
            log("  episode ended:", terminated, truncated)
            obs, _ = env.reset()
    dt = time.time() - t0
    print(f"30 steps in {dt:.1f}s ({30 / dt:.2f} steps/s), total reward {total:+.2f}")
    print("obs keys:", list(obs.keys()), {k: np.asarray(v).shape for k, v in obs.items()})
    env.close()


if __name__ == "__main__":
    main()
