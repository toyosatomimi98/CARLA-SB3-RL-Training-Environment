"""Compare an 'aggressive' fixed policy with and without the CBF safety shield
inside CARLA.  Prints safety statistics and writes a CSV trace."""
import csv
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
# A deliberately reckless policy: full throttle, strong constant left steer.
AGGRESSIVE = np.array([0.55, 0.95], dtype=np.float32)
STEPS = 220


def build_env(shield: bool, log_csv: str | None = None):
    obs_space, encode_fn, _ = create_encode_state_fn(None, STATE)
    env = CarlaRouteEnv(
        host="127.0.0.1", port=2000, obs_res=(160, 80),
        reward_fn=reward_functions["reward_fn5"],
        observation_space=obs_space, encode_state_fn=encode_fn, decode_vae_fn=None,
        fps=15, action_smoothing=0.75, action_space_type="continuous",
        activate_spectator=False, activate_render=False, start_carla=False,
        town="Town10HD_Opt",
    )
    cfg = CBFFilterConfig(dt=1.0 / 15.0, lane_half_width=1.75,
                          speed_limit=13.9, a_lat_max=2.5)
    # both variants use the wrapper so that identical safety metrics are logged;
    # the baseline just never overrides the policy action.
    return SafetyShieldWrapper(env, config=cfg, penalty=0.0, log_path=log_csv,
                               enabled=shield)


def run(shield: bool, tag: str, csv_path: str | None = None) -> dict:
    env = build_env(shield, log_csv=None)
    obs, _ = env.reset(seed=42)
    rows = []
    stats = {"min_barrier": float("inf"), "viol_steps": 0, "interventions": 0,
             "collisions": 0, "terminations": 0, "max_speed": 0.0,
             "max_abs_ey": 0.0, "distance": 0.0}
    for i in range(STEPS):
        obs, rew, term, trunc, info = env.step(AGGRESSIVE)
        s = info.get("safety", {})
        mb = s.get("min_barrier", float("nan"))
        if np.isfinite(mb):
            stats["min_barrier"] = min(stats["min_barrier"], mb)
            stats["viol_steps"] += int(mb < 0)
        stats["interventions"] += int(s.get("intervened", False))
        stats["max_speed"] = max(stats["max_speed"], s.get("speed_kmh", 0.0))
        stats["max_abs_ey"] = max(stats["max_abs_ey"], abs(s.get("lateral_offset", 0.0)))
        stats["distance"] = float(info.get("total_distance", stats["distance"]))
        rows.append([i, s.get("speed_kmh", np.nan), s.get("lateral_offset", np.nan),
                     mb, int(s.get("intervened", False))])
        if term:
            stats["terminations"] += 1
            stats["collisions"] += int(bool(getattr(env.unwrapped, "collision_flag", False)))
            break
        if trunc:
            break
    env.close()
    if csv_path:
        os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["step", "speed_kmh", "lateral_offset_m", "min_barrier", "intervened"])
            w.writerows(rows)
    print(f"[{tag}] steps={len(rows)} min_barrier={stats['min_barrier']:+.3f} "
          f"violation_steps={stats['viol_steps']} interventions={stats['interventions']} "
          f"terminations={stats['terminations']} collisions={stats['collisions']} "
          f"max_speed={stats['max_speed']:.1f}km/h "
          f"max|e_y|={stats['max_abs_ey']:.2f}m distance={stats['distance']:.1f}m")
    return stats


def main() -> None:
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "safety_results")
    run(False, "no shield ", os.path.join(out, "trace_no_shield.csv"))
    run(True, "with shield", os.path.join(out, "trace_shield.csv"))


if __name__ == "__main__":
    main()
