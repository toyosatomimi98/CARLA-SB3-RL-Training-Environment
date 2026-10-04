"""Establish the sign conventions empirically (CARLA is left-handed and the
steering sign is easy to get wrong).

For steer = +0.6 and steer = -0.6 we log
  * the yaw change (Rotation.yaw grows = turning right in CARLA),
  * the signed lateral offset e_y (lane cross product),
  * what vehicle.get_angle() reports (the quantity the reference controller uses).
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402

config.set_config("SAFETY_PPO_SPEED")

import gymnasium as gym  # noqa: E402
from carla_env.envs.carla_route_env import CarlaRouteEnv  # noqa: E402
from carla_env.rewards import reward_functions  # noqa: E402
from carla_env.state_commons import create_encode_state_fn  # noqa: E402
from config import CONFIG  # noqa: E402
from carla_env.wrappers import vector  # noqa: E402


def run(steer_sign: float, seed: int = 33):
    obs_space, encode_fn, _ = create_encode_state_fn(None, CONFIG["state"])
    env = CarlaRouteEnv(obs_res=CONFIG["obs_res"], host="127.0.0.1", port=2000,
                        reward_fn=reward_functions[CONFIG["reward_fn"]],
                        observation_space=obs_space, encode_state_fn=encode_fn,
                        decode_vae_fn=None, fps=15, action_smoothing=0.0,
                        action_space_type="continuous", activate_spectator=False,
                        activate_render=False, start_carla=False, town=CONFIG["map"],
                        allow_brake=True)
    env = gym.wrappers.TimeLimit(env, max_episode_steps=400)
    env.reset(seed=seed)
    u = env.unwrapped
    yaw0 = u.vehicle.get_transform().rotation.yaw
    wp0 = u.current_waypoint
    loc0 = vector(u.vehicle.get_transform().location)
    fwd0 = vector(wp0.transform.rotation.get_forward_vector())
    right0 = np.array([-fwd0[1], fwd0[0], 0.0])
    print(f"\n--- steer = {steer_sign:+.2f} ---")
    for k in range(45):  # 3 s
        env.step(np.array([steer_sign * 0.6, 0.55], dtype=np.float32))
    yaw1 = u.vehicle.get_transform().rotation.yaw
    dyaw = ((yaw1 - yaw0 + 180) % 360) - 180  # wrapped
    d = vector(u.vehicle.get_transform().location) - loc0
    lat = float(np.dot(d, right0))
    ang_reported = float(u.vehicle.get_angle(u.current_waypoint))
    print(f"  yaw {yaw0:.1f} -> {yaw1:.1f} (Δ={dyaw:+.1f} deg)   "
          f"[+Δyaw = turning RIGHT]")
    print(f"  lateral displacement = {lat:+.2f} m  (cross-product convention, "
          f"+ = to the RIGHT of the lane)")
    print(f"  vehicle.get_angle() now reports {ang_reported:+.3f} rad")
    print(f"  controlled speed = {u.vehicle.get_speed():.1f} km/h, "
          f"env.distance_from_center = {u.distance_from_center:.2f} m")
    env.close()
    return dyaw, lat, ang_reported


if __name__ == "__main__":
    r_plus = run(+1.0)
    r_minus = run(-1.0)
    print("\n=== interpretation ===")
    dyp, latp, angp = r_plus
    dym, latm, angm = r_minus
    print(f"  steer=+0.6  -> Δyaw {dyp:+.1f}°, lateral {latp:+.2f} m, reported angle {angp:+.3f}")
    print(f"  steer=-0.6  -> Δyaw {dym:+.1f}°, lateral {latm:+.2f} m, reported angle {angm:+.3f}")
    print("  => positive steer turns the car " +
          ("RIGHT" if dyp > 0 else "LEFT") + " in this CARLA build")
    print("  => get_angle() is " +
          ("POSITIVE when pointing to the RIGHT" if (angp > 0 and latp > 0) or
           (angp < 0 and latp < 0) else "POSITIVE when pointing to the LEFT"))
