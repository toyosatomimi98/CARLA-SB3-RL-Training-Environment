"""Per-step diagnostic of the CBF filter inside CARLA."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402

config.set_config("2")

from carla_env.rewards import reward_functions  # noqa: E402
from carla_env.state_commons import create_encode_state_fn  # noqa: E402
from carla_env.envs.carla_route_env import CarlaRouteEnv  # noqa: E402
from carla_env.safety.cbf import CBFFilterConfig  # noqa: E402
from carla_env.safety.wrapper import SafetyShieldWrapper  # noqa: E402

STATE = ["steer", "throttle", "speed", "maneuver"]
AGGRESSIVE = np.array([0.55, 0.95], dtype=np.float32)


def main() -> None:
    obs_space, encode_fn, _ = create_encode_state_fn(None, STATE)
    env = CarlaRouteEnv(host="127.0.0.1", port=2000, obs_res=(160, 80),
                        reward_fn=reward_functions["reward_fn5"],
                        observation_space=obs_space, encode_state_fn=encode_fn,
                        decode_vae_fn=None, fps=15, action_smoothing=0.75,
                        action_space_type="continuous", activate_spectator=False,
                        activate_render=False, start_carla=False, town="Town10HD_Opt")
    cfg = CBFFilterConfig(dt=1 / 15.0)
    sh = SafetyShieldWrapper(env, config=cfg, enabled=True)
    sh.reset(seed=42)
    print("cfg: lane_half_width=%.2f speed_limit=%.2f horizon=%d" %
          (cfg.lane_half_width, cfg.speed_limit, cfg.horizon))
    print(f"{'k':>3} {'v_kmh':>6} {'e_y':>6} {'e_psi':>6} {'rate':>6} "
          f"{'u_rl':>12} {'u_safe':>12} {'minh':>7} {'pred_e_y@H':>10} {'int':>3}")
    for k in range(20):
        st = sh.ego_state()
        step = sh.filter.solve(st, AGGRESSIVE)
        pred = sh.filter.rollout(st, step.action)
        sh.env.step(step.action)
        if k % 2 == 0:
            print(f"{k:>3} {st.v * 3.6:6.1f} {st.e_y:6.2f} {st.e_psi:6.2f} "
                  f"{(st.e_y_dot or 0):6.2f} "
                  f"[{AGGRESSIVE[0]:+.2f},{AGGRESSIVE[1]:+.2f}] "
                  f"[{step.action[0]:+.2f},{step.action[1]:+.2f}] "
                  f"{step.min_barrier:7.3f} {pred[-1].e_y:10.2f} {int(step.intervened):>3}")
    print("filter stats:", sh.filter.stats())
    sh.close()


if __name__ == "__main__":
    main()
