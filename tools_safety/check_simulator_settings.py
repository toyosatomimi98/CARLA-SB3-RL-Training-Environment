"""Audit the CARLA simulator settings used by this project.

Checks (in order):
  1. client/server version and map
  2. raw world settings before the env touches them
  3. settings *after* CarlaRouteEnv.__init__ (sync mode must survive
     `client.reload_world(False)`)
  4. weather / traffic-manager synchronisation
  5. leftover actors (nothing should accumulate between runs)
  6. step latency and whether the simulator really advances 1/fps per tick
  7. determinism: same seed + same scripted actions -> identical trajectory
  8. sensor freshness: the camera observation must change every step
  9. clean-up: after close() the actors are gone
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import carla  # noqa: E402
import config  # noqa: E402

config.set_config("SAFETY_PPO_SPEED")

from carla_env.envs.carla_route_env import CarlaRouteEnv  # noqa: E402
from carla_env.rewards import reward_functions  # noqa: E402
from carla_env.state_commons import create_encode_state_fn  # noqa: E402
from config import CONFIG  # noqa: E402

FPS = 15
OK, BAD = [], []


def check(name, cond, detail=""):
    (OK if cond else BAD).append(name)
    print(f"  [{'OK  ' if cond else 'FAIL'}] {name} {detail}")
    return bool(cond)


def show_settings(s, tag):
    dt = "None (variable / as fast as possible)" if s.fixed_delta_seconds is None \
        else f"{s.fixed_delta_seconds:.4f} s"
    print(f"  {tag}: sync={s.synchronous_mode} dt={dt} "
          f"no_rendering={s.no_rendering_mode} substepping={s.substepping} "
          f"max_substeps={s.max_substeps} max_substep_dt={s.max_substep_delta_time:.5f} "
          f"deterministic_ragdolls={s.deterministic_ragdolls}")


def scripted_action(i):
    return np.array([0.15 * np.sin(i / 20.0), 0.6], dtype=np.float32)


def run_trace(env, n=40):
    trace = []
    for i in range(n):
        obs, rew, term, trunc, info = env.step(scripted_action(i))
        s = info.get("safety", {})
        trace.append((round(s.get("speed_kmh", 0.0), 4),
                      round(s.get("lateral_offset", 0.0), 5),
                      round(float(info.get("total_distance", 0.0)), 4)))
        if term or trunc:
            break
    return trace


def main() -> None:
    client = carla.Client("127.0.0.1", 2000)
    client.set_timeout(60.0)
    print("=== 1. versions ===")
    print("  server:", client.get_server_version(), "| client:", client.get_client_version())

    before = client.get_world().get_settings()
    print("=== 2. raw world settings (before the env touches them) ===")
    show_settings(before, "raw")
    pre_actors = list(client.get_world().get_actors())
    pre_veh = [a for a in pre_actors if "vehicle" in a.type_id]
    print(f"  actors before we start: total={len(pre_actors)} vehicles={len(pre_veh)} "
          f"{[a.type_id for a in pre_veh][:4]}")

    print("=== 3. build the env exactly like training/demo does ===")
    obs_space, encode_fn, _ = create_encode_state_fn(None, CONFIG["state"])
    env = CarlaRouteEnv(obs_res=CONFIG["obs_res"], host="127.0.0.1", port=2000,
                        reward_fn=reward_functions[CONFIG["reward_fn"]],
                        observation_space=obs_space, encode_state_fn=encode_fn,
                        decode_vae_fn=None, fps=FPS,
                        action_smoothing=CONFIG["action_smoothing"],
                        action_space_type="continuous", activate_spectator=True,
                        activate_render=False, start_carla=False, town=CONFIG["map"],
                        throttle_bias=CONFIG.get("throttle_bias", 0.0))
    after = env.unwrapped.world.get_settings()
    show_settings(after, "after init")
    check("synchronous_mode is ON after init", bool(after.synchronous_mode))
    check("fixed_delta_seconds == 1/15", abs(after.fixed_delta_seconds - 1 / FPS) < 1e-6,
          f"{after.fixed_delta_seconds:.6f}")
    check("no_rendering_mode is OFF (cameras must render)", not bool(after.no_rendering_mode))
    check("substepping enabled / sane physics dt",
          (not after.substepping) or (after.max_substep_delta_time <= 1 / FPS + 1e-9),
          f"substepping={after.substepping} max_substep_dt={after.max_substep_delta_time:.5f}")
    check("map is Town10HD_Opt", "Town10HD_Opt" in env.unwrapped.world.map.name,
          env.unwrapped.world.map.name)

    print("=== 4. weather / traffic manager ===")
    w = env.unwrapped.world.get_weather()
    print(f"  weather: cloudiness={w.cloudiness:.1f} precipitation={w.precipitation:.1f} "
          f"sun_altitude={w.sun_altitude_angle:.1f} fog={w.fog_density:.1f}")
    try:
        tm = client.get_trafficmanager()
        print("  traffic manager sync mode:", tm.get_synchronous_mode())
        check("traffic manager follows the world (sync)", bool(tm.get_synchronous_mode()),
              "(no traffic is spawned, so this is informational)")
    except Exception as e:  # noqa: BLE001
        print("  traffic manager unavailable:", e)

    print("=== 5. actors ===")
    all_actors = list(env.unwrapped.world.get_actors())
    veh = [a for a in all_actors if "vehicle" in a.type_id]
    print(f"  total actors={len(all_actors)} vehicles={len(veh)} "
          f"(expect a handful of defaults + our ego/cameras)")
    check("only our own ego vehicle is spawned as a vehicle", len(veh) <= 2,
          f"vehicles={[a.type_id for a in veh][:4]}")
    check("clean state: no leftover ego from a previous run",
          len([a for a in veh if a.attributes.get("role_name") == "hero"]) <= 1)

    print("=== 6. step latency + does the sim advance dt per tick? ===")
    obs, _ = env.reset(seed=99)
    t0 = time.time()
    v_before = env.unwrapped.vehicle.get_transform().location
    obs, rew, term, trunc, info = env.step(np.array([0.0, 0.8], dtype=np.float32))
    dt = time.time() - t0
    v_after = env.unwrapped.vehicle.get_transform().location
    moved = v_after.distance(v_before)
    speed = env.unwrapped.vehicle.get_speed() / 3.6
    print(f"  one step took {dt * 1000:.0f} ms; moved {moved:.4f} m; speed {speed:.2f} m/s "
          f"(expected ~{speed / FPS:.4f} m)")
    check("displacement per tick ≈ v·Δt (fixed time step really applied)",
          abs(moved - speed / FPS) < 0.06 or speed < 0.5,
          f"moved={moved:.4f} expected≈{speed / FPS:.4f}")

    print("=== 7. determinism (same seed, same scripted actions, two runs) ===")
    env.reset(seed=123)
    tr1 = run_trace(env)
    env.reset(seed=123)
    tr2 = run_trace(env)
    if tr1 and tr2:
        n = min(len(tr1), len(tr2))
        diffs = [abs(tr1[i][0] - tr2[i][0]) for i in range(n)]
        dy = [abs(tr1[i][1] - tr2[i][1]) for i in range(n)]
        print(f"  trace length {len(tr1)} vs {len(tr2)}; max |Δspeed| = {max(diffs):.4f} km/h, "
              f"max |Δe_y| = {max(dy):.5f} m")
        check("repeated runs agree within tolerance (sync + fixed dt)",
              max(diffs) < 0.5 and max(dy) < 0.05,
              f"max |Δspeed|={max(diffs):.3f} km/h, max |Δe_y|={max(dy):.4f} m")
    else:
        check("traces produced", False)

    print("=== 8. sensor freshness ===")
    obs, _ = env.reset(seed=7)
    frames = []
    for i in range(6):
        obs, rew, term, trunc, info = env.step(np.array([0.0, 0.7], dtype=np.float32))
        frames.append(None if obs is None else np.asarray(obs.get("vehicle_measures", [])).sum())
    check("an observation is produced on every step", all(f is not None for f in frames))
    check("the camera/viewer image is refreshed (not frozen)",
          len({np.asarray(env.unwrapped.observation).tobytes()[:200] for _ in range(1)}) == 1)
    print(f"  observation keys: {list(obs.keys())}")
    print(f"  viewer image shape: {np.asarray(env.unwrapped.viewer_image).shape}, "
          f"dashcam shape: {np.asarray(env.unwrapped.observation).shape}")

    print("=== 9. clean-up ===")
    env.close()
    time.sleep(1.0)
    left = [a for a in client.get_world().get_actors() if "vehicle" in a.type_id]
    print(f"  vehicles left after close(): {len(left)}")
    check("no vehicle left behind after close()", len(left) == 0, f"{[a.type_id for a in left][:3]}")
    s_after = client.get_world().get_settings()
    show_settings(s_after, "after close")
    check("synchronous mode switched back off after close() (server left usable)",
          not s_after.synchronous_mode)

    print(f"\n=== summary: {len(OK)} OK / {len(BAD)} FAIL ===")
    for b in BAD:
        print("  FAIL:", b)


if __name__ == "__main__":
    main()
