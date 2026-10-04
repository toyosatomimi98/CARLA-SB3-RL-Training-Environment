"""Full-state steering diagnosis.

Prints, every 0.5 s, everything that could explain a permanent rightward steer:
  * signed lane offset e_y (three independent definitions, so a sign mistake in
    any single one is visible),
  * heading error from vehicle.get_angle() and the raw yaw of route vs car,
  * the commanded steer, the steer actually applied, and the resulting yaw rate,
  * lane bookkeeping: the lane of the ego (from the map) vs the lane of the
    route waypoint the controller is chasing (a mismatch here would make the car
    "chase" a lane that is not the one it is in).
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import carla  # noqa: E402
import config  # noqa: E402

config.set_config("SAFETY_PPO_SPEED")

import gymnasium as gym  # noqa: E402
from carla_env.envs.carla_route_env import CarlaRouteEnv  # noqa: E402
from carla_env.rewards import reward_functions  # noqa: E402
from carla_env.safety.cbf import CBFFilterConfig  # noqa: E402
from carla_env.safety.wrapper import SafetyShieldWrapper  # noqa: E402
from carla_env.state_commons import create_encode_state_fn  # noqa: E402
from config import CONFIG  # noqa: E402
from driving_control import lane_follow_action, signed_lateral_offset  # noqa: E402


def main() -> None:
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 240
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
                              penalty=0.0, enabled=False)   # metrics only
    env.reset(seed=33)
    u = env.unwrapped
    print(f"{'t':>5} {'v':>6} {'e_y(mine)':>9} {'d2c':>5} {'ang(deg)':>8} {'steer':>6} "
          f"{'applied':>7} {'yawrate':>8} {'yaw':>7} {'ego_lane':>8} {'route_lane':>10} {'idx':>5}")
    prev_yaw = u.vehicle.get_transform().rotation.yaw
    for k in range(steps):
        a = lane_follow_action(u, ref_speed_kmh=18.0)
        obs, rew, term, trunc, info = env.step(a)
        if k % 8:
            continue
        tf = u.vehicle.get_transform()
        yaw = tf.rotation.yaw
        yawrate = ((yaw - prev_yaw + 180) % 360) - 180
        prev_yaw = yaw
        ego_wp = u.world.map.get_waypoint(tf.location, project_to_road=True)
        route_wp = u.current_waypoint
        ang = u.vehicle.get_angle(route_wp)
        print(f"{k / 15:5.1f} {u.vehicle.get_speed():6.1f} {signed_lateral_offset(u):+9.2f} "
              f"{u.distance_from_center:5.2f} {np.degrees(ang):+8.1f} {a[0]:+6.2f} "
              f"{u.vehicle.control.steer:+7.2f} {yawrate:+8.2f} {yaw:+7.1f} "
              f"{ego_wp.lane_id:>8} {route_wp.lane_id:>10} {u.current_waypoint_index:>5}", flush=True)
        if term or trunc:
            print("episode ended at t=%.1f s" % (k / 15))
            break
    env.close()


if __name__ == "__main__":
    main()
