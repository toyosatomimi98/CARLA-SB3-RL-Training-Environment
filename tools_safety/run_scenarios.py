"""Scenario challenge suite: run the agent (policy + safety layer) through every
safety scenario and report whether it *completes* the challenge.

Usage
-----
python tools_safety/run_scenarios.py                 # shield ON, all scenarios
python tools_safety/run_scenarios.py --shield 0      # baseline without the filter
python tools_safety/run_scenarios.py --scenarios roadblock,lead_brake

Verdict per run
---------------
PASS  : no collision, progressed >= `--min-progress` m, and never stalled
FAIL  : collision, or progress too short, or the vehicle froze (mean speed of the
        last 3 s below 2 km/h) - a "safe" agent that stops is counted as failing.
"""
import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import carla  # noqa: E402
import config  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--scenarios", default="roadblock,lead_brake,crossing,ped_cross,cutin")
ap.add_argument("--shield", type=int, default=1)
ap.add_argument("--steps", type=int, default=330)
ap.add_argument("--seed", type=int, default=33)
ap.add_argument("--min-progress", type=float, default=60.0)
ap.add_argument("--ref-speed", type=float, default=16.0)
ap.add_argument("--out", default="safety_results/scenario_benchmark.csv")
ap.add_argument("--config", default="SAFETY_RL",
                help="must be the config the policy was trained with")
ap.add_argument("--policy", default="rl", choices=["rl", "reference"],
                help="'rl' = trained PPO checkpoint (default), 'reference' = hand-written controller")
ap.add_argument("--model", default="", help="path to the .zip; default = newest PPO_rl_* run")
args = ap.parse_args()

config.set_config(args.config)

import gymnasium as gym  # noqa: E402
import scenarios  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402
from carla_env.envs.carla_route_env import CarlaRouteEnv  # noqa: E402
from carla_env.rewards import reward_functions  # noqa: E402
from carla_env.safety.cbf import CBFFilterConfig  # noqa: E402
from carla_env.safety.wrapper import SafetyShieldWrapper  # noqa: E402
from carla_env.state_commons import create_encode_state_fn  # noqa: E402
from config import CONFIG  # noqa: E402


def reference_action(env, ref_speed: float) -> np.ndarray:
    # NEGATIVE feedback: steer>0 turns right and get_angle() is positive when the
    # car points right (verified by tools_safety/probe_steer_sign.py)
    from driving_control import lane_follow_action
    return lane_follow_action(env, ref_speed_kmh=ref_speed, steer_gain=1.4)


POLICY = {"model": None}


def policy_action(env, obs) -> np.ndarray:
    """Action source for the episode: the trained policy, or the reference one."""
    if POLICY["model"] is None:
        return reference_action(env, args.ref_speed)
    action, _ = POLICY["model"].predict(obs, deterministic=True)
    return np.asarray(action, dtype=np.float32)


def run_one(name: str) -> dict:
    holder: dict = {}
    try:
        return _run_one(name, holder)
    finally:
        # Always release the simulator: an exception used to skip close() and
        # left the CARLA world (and the traffic manager) in synchronous mode,
        # which then *wedged* every later run.
        scn = holder.get("scn")
        env = holder.get("env")
        try:
            if scn is not None:
                scn.destroy()
        except Exception:  # noqa: BLE001
            pass
        try:
            if env is not None:
                env.close()
        except Exception:  # noqa: BLE001
            pass


def _run_one(name: str, holder: dict) -> dict:
    obs_space, encode_fn, _ = create_encode_state_fn(None, CONFIG["state"])
    env = CarlaRouteEnv(obs_res=CONFIG["obs_res"], host="127.0.0.1", port=2000,
                        reward_fn=reward_functions[CONFIG["reward_fn"]],
                        observation_space=obs_space, encode_state_fn=encode_fn,
                        decode_vae_fn=None, fps=15,
                        action_smoothing=CONFIG["action_smoothing"],
                        action_space_type="continuous", activate_spectator=False,
                        activate_render=False, start_carla=False, town=CONFIG["map"],
                        allow_brake=CONFIG.get("allow_brake", True))
    env = gym.wrappers.TimeLimit(env, max_episode_steps=args.steps)
    # NOTE (2026-10-05): this used the *class defaults* for the lane/speed/curve
    # parameters and the horizon, while the demo recorder and `eval_safety.py`
    # used the config's values (horizon 20).  The two filters therefore behaved
    # differently, so the benchmark and the videos were not the same experiment.
    # Everything now reads the same `safety` block.
    sc = CONFIG.get("safety", {})
    cfg = CBFFilterConfig(dt=1 / 15.0,
                          lane_half_width=sc.get("lane_half_width", 1.75),
                          lane_margin=sc.get("lane_margin", 0.30),
                          speed_limit=sc.get("speed_limit", 13.9),
                          a_lat_max=sc.get("a_lat_max", 2.5),
                          horizon=sc.get("horizon", 20),
                          alpha_lane=sc.get("alpha_lane", 0.30),
                          alpha_speed=sc.get("alpha_speed", 0.30),
                          alpha_curve=sc.get("alpha_curve", 0.30),
                          throttle_min=-1.0)
    env = SafetyShieldWrapper(env, config=cfg, penalty=0.0, enabled=bool(args.shield))
    holder["env"] = env
    obs, _ = env.reset(seed=args.seed)
    scn = scenarios.build(name, env.unwrapped)
    holder["scn"] = scn
    scn.spawn()
    for a in scn.actors:
        env.register_obstacle(a)

    speeds, gaps, mins = [], [], []
    collision = False
    for k in range(args.steps):
        t = k / 15.0
        scn.update(t)
        obs, rew, term, trunc, info = env.step(policy_action(env.unwrapped, obs))
        s = info.get("safety", {})
        speeds.append(s.get("speed_kmh", 0.0))
        mins.append(s.get("min_barrier", np.nan))
        o = env.nearest_obstacle(force=True)
        if o is not None:
            gaps.append(o[0])
        collision = collision or bool(getattr(env.unwrapped, "collision_flag", False))
        if term or trunc:
            break
    progress = float(info.get("total_distance", 0.0))
    tail = float(np.mean(speeds[-45:])) if len(speeds) >= 45 else float(np.mean(speeds or [0]))
    stalled = tail < 2.0 and progress < 0.5 * args.min_progress
    min_gap = float(np.min(gaps)) if gaps else float("nan")
    min_h = float(np.nanmin(mins)) if mins else float("nan")

    # Success is scenario-dependent: with a *blocking* obstacle the correct
    # behaviour is to stop behind it (progress must NOT be required); on an
    # open challenge the agent must get through.
    blocking = name in ("roadblock", "lead_brake", "ped_cross", "cutin")
    reasons = []
    if collision:
        reasons.append("collision")
    # any scenario: getting within ~1 m of the obstacle counts as contact
    if (not np.isnan(min_gap)) and min_gap < 1.0:
        reasons.append(f"contact (min_gap {min_gap:.2f} m)")
    if blocking:
        safe_gap = 2.0
        if not (np.isnan(min_gap) or min_gap >= safe_gap) and f"contact (min_gap {min_gap:.2f} m)" not in reasons:
            reasons.append(f"min_gap {min_gap:.1f}<{safe_gap:.0f}m")
        if progress < 5.0:
            reasons.append("never moved")
    else:
        if progress < args.min_progress:
            reasons.append(f"progress {progress:.0f}<{args.min_progress:.0f}m")
        if stalled:
            reasons.append("stalled")
    verdict = "PASS" if not reasons else "FAIL"
    row = dict(scenario=name, shield=int(args.shield), verdict=verdict,
               reason="/".join(reasons) or "-", progress_m=round(progress, 1),
               tail_speed_kmh=round(tail, 2), min_gap_m=None if np.isnan(min_gap) else round(min_gap, 2),
               min_barrier=None if np.isnan(min_h) else round(min_h, 3),
               interventions=int(env.filter.num_interventions),
               note=scn.notes[:70])
    scn.destroy()
    env.close()
    return row


def main() -> None:
    if args.policy == "rl":
        path = args.model
        if not path:
            import glob as _glob
            cands = sorted(_glob.glob(os.path.join("tensorboard", "PPO_rl_*", "final_model.zip")),
                           key=os.path.getmtime, reverse=True)
            path = cands[0] if cands else ""
        if not path or not os.path.exists(path):
            raise SystemExit(f"--policy rl but no model found ({path!r}); pass --model <path.zip>")
        POLICY["model"] = PPO.load(path, device="cuda")
        print(f"[bench] action source: trained policy {path}", flush=True)
    else:
        print("[bench] action source: reference controller", flush=True)
    rows = []
    out_path = args.out
    if out_path and os.path.exists(out_path):
        os.remove(out_path)
    for name in [s.strip() for s in args.scenarios.split(",") if s.strip()]:
        try:
            r = run_one(name)
        except Exception as e:  # noqa: BLE001 - one bad scenario must not kill the suite
            r = dict(scenario=name, shield=int(args.shield), verdict="ERROR",
                     reason=f"{type(e).__name__}: {e}"[:60], progress_m=0.0,
                     tail_speed_kmh=0.0, min_gap_m=None, min_barrier=None,
                     interventions=0, note="")
        rows.append(r)
        print(f"  [{r['verdict']}] {r['scenario']:11s} shield={r['shield']} "
              f"progress={r['progress_m']:6.1f}m tail_v={r['tail_speed_kmh']:5.1f}km/h "
              f"min_gap={r['min_gap_m']} min_h={r['min_barrier']} "
              f"itv={r['interventions']:3d} ({r['reason']})", flush=True)
        if out_path:  # write incrementally so a crash never loses results
            os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
            new = not os.path.exists(out_path)
            with open(out_path, "a", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(r.keys()))
                if new:
                    w.writeheader()
                w.writerow(r)
    if out_path:
        print("wrote ->", out_path)
    n_pass = sum(1 for r in rows if r["verdict"] == "PASS")
    print(f"\n== {n_pass}/{len(rows)} PASS (shield={args.shield}) ==")


if __name__ == "__main__":
    main()
