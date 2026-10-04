"""Verify that a scenario's obstacle is actually detected by the safety layer."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import carla  # noqa: E402
import config  # noqa: E402

config.set_config("SAFETY_PPO_SPEED")

import gymnasium as gym  # noqa: E402
import scenarios  # noqa: E402
from carla_env.envs.carla_route_env import CarlaRouteEnv  # noqa: E402
from carla_env.rewards import reward_functions  # noqa: E402
from carla_env.safety.cbf import CBFFilterConfig  # noqa: E402
from carla_env.safety.wrapper import SafetyShieldWrapper  # noqa: E402
from carla_env.state_commons import create_encode_state_fn  # noqa: E402
from config import CONFIG  # noqa: E402
from carla_env.wrappers import vector  # noqa: E402

name = sys.argv[1] if len(sys.argv) > 1 else "roadblock"

obs_space, encode_fn, _ = create_encode_state_fn(None, CONFIG["state"])
env = CarlaRouteEnv(obs_res=CONFIG["obs_res"], host="127.0.0.1", port=2000,
                    reward_fn=reward_functions[CONFIG["reward_fn"]],
                    observation_space=obs_space, encode_state_fn=encode_fn,
                    decode_vae_fn=None, fps=15, action_smoothing=0.0,
                    action_space_type="continuous", activate_spectator=False,
                    activate_render=False, start_carla=False, town=CONFIG["map"],
                    allow_brake=True)
env = gym.wrappers.TimeLimit(env, max_episode_steps=400)
cfg = CBFFilterConfig(dt=1 / 15.0, throttle_min=-1.0)
env = SafetyShieldWrapper(env, config=cfg, penalty=0.0, enabled=True)

obs, _ = env.reset(seed=33)
scn = scenarios.build(name, env.unwrapped)
scn.spawn()
for _a in scn.actors:
    env.register_obstacle(_a)
print("scenario:", name, "|", scn.notes, flush=True)
if getattr(scn, "actors", None):
    for a in scn.actors:
        print("   spawned:", a.type_id, "at", a.get_transform().location)

ego = env.unwrapped.vehicle.actor
for k in range(240):
    t = k / 15.0
    scn.update(t)
    # corrected lane-following controller (negative feedback + lateral term)
    from driving_control import lane_follow_action
    a = lane_follow_action(env.unwrapped, ref_speed_kmh=16.0)
    obs, rew, term, trunc, info = env.step(a)
    obs_obs = env.nearest_obstacle(force=True)
    if k % 15 == 0:
        s = info.get("safety", {})
        gap = None if obs_obs is None else obs_obs[0]
        print(f"  t={t:5.1f}s v={env.unwrapped.vehicle.get_speed():5.1f}km/h "
              f"gap={('none' if gap is None else f'{gap:5.1f} m')} "
              f"v_lead={0 if obs_obs is None else obs_obs[1]:.1f} "
              f"h_gap={float(np.asarray(s.get('barriers', [0]*5))[4]):+8.2f} "
              f"minh={s.get('min_barrier', 0):+7.2f} int={int(bool(s.get('intervened')))} "
              f"qp={int(bool(s.get('qp_success', 1)))} fb={int(bool(s.get('fallback', 0)))}",
              flush=True)
    if term or trunc:
        print("  episode ended at t=%.1f s" % t)
        break
scn.destroy()
env.close()
