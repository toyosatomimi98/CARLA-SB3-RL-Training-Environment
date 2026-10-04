"""Record a driving demonstration video with rich auxiliary instrumentation.

Layout (1280x720 @ 15 fps):
  +---------------------------+---------------------------+
  | chase camera (spectator)  |  map BEV (top-down camera)|
  +---------------------------+---------------------------+
  | driver camera + state     | control / navigation /    |
  |                           | safety bars + time series |
  +---------------------------+---------------------------+

The scenario deliberately mixes "normal" driving with a disturbance window so
that the CBF safety filter can be *seen* intervening (red flash + bars).
"""
import argparse
import csv
import os
import sys
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import carla  # noqa: E402
import config  # noqa: E402

# NOTE (2026-10-04): the config is now chosen on the command line, and it has to
# be set BEFORE `carla_env.rewards` is imported - that module reads the reward
# parameters out of CONFIG at import time.  Training, the benchmark and these
# videos now all run the same config, so the action source that is recorded is
# the one that was trained.
_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--config", default="SAFETY_RL")
_pre_args, _ = _pre.parse_known_args()
config.set_config(_pre_args.config)

import cv2  # noqa: E402
from carla_env.envs.carla_route_env import CarlaRouteEnv  # noqa: E402
from carla_env.navigation.planner import RoadOption  # noqa: E402
from carla_env.rewards import reward_functions  # noqa: E402
from carla_env.safety.cbf import CBFFilterConfig  # noqa: E402
from carla_env.safety.wrapper import SafetyShieldWrapper  # noqa: E402
from carla_env.state_commons import create_encode_state_fn  # noqa: E402
from config import CONFIG  # noqa: E402

W, H = 1280, 720
TOP_H = 360
BOT = TOP_H + 6

# RoadOption defines __eq__ without __hash__, so it cannot be used as a dict
# key - look the label up by enum *value* instead.
MANEUVER = {0: "LANE FOLLOW", 1: "TURN LEFT", 2: "TURN RIGHT", 3: "GO STRAIGHT"}

# reference lane-following speed, kept identical to run_scenarios.py so that the
# demo videos show the same behaviour as the benchmark table
REF_SPEED = [16.0]


def maneuver_name(opt) -> str:
    try:
        return MANEUVER.get(int(opt.value), "-")
    except Exception:  # noqa: BLE001
        return "-"


def _font(size):
    for name in ("consola.ttf", "arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:  # noqa: BLE001
            continue
    return ImageFont.load_default()


F_SM, F_MD, F_LG = _font(14), _font(16), _font(19)
F_XL, F_HUGE = _font(30), _font(46)


def banner(draw, x, y, w, h, text, color, sub=""):
    """Big, unmissable status banner."""
    draw.rectangle([x, y, x + w, y + h], fill=color)
    draw.text((x + 10, y + 4), text, font=F_XL, fill=(255, 255, 255))
    if sub:
        draw.text((x + 12, y + h - 20), sub, font=F_SM, fill=(240, 240, 240))


def big_numbers(draw, x, y, speed_kmh, min_h, interventions, steps):
    """Large, readable decisive numbers."""
    draw.text((x, y), f"{speed_kmh:5.1f}", font=F_HUGE,
              fill=(140, 235, 140) if speed_kmh > 3 else (255, 200, 90))
    draw.text((x + 130, y + 16), "km/h", font=F_MD, fill=(225, 225, 225))
    col = (160, 235, 160) if min_h > 0 else (255, 90, 90)
    draw.text((x, y + 52), f"min h = {min_h:+.2f}", font=F_XL, fill=col)
    draw.text((x, y + 90), f"CBF interventions: {interventions} / {steps}",
              font=F_MD, fill=(255, 120, 120) if interventions else (200, 200, 200))


def event_strip(draw, x0, y0, width, height, events, total, color_on=(230, 60, 60),
                color_off=(40, 90, 50)):
    """One pixel-column per control step: red = the filter overrode the policy."""
    draw.rectangle([x0 - 1, y0 - 1, x0 + width + 1, y0 + height + 1], outline=(120, 120, 120))
    n = max(total, 1)
    for i, on in enumerate(events):
        xa = x0 + int(i * width / n)
        xb = x0 + max(int((i + 1) * width / n), 1)
        draw.rectangle([xa, y0, xb, y0 + height], fill=color_on if on else color_off)


def controller(env, disturbance: float) -> np.ndarray:
    """Reference lane-following policy (corrected signs - see driving_control)."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from driving_control import lane_follow_action
    return lane_follow_action(env, ref_speed_kmh=REF_SPEED[0], steer_gain=1.4,
                              disturbance=disturbance)


def bar(draw, x, y, w, h, frac, color, label, value):
    draw.rectangle([x, y, x + w, y + h], outline=(90, 90, 90))
    frac = max(0.0, min(1.0, frac))
    draw.rectangle([x, y, x + int(w * frac), y + h], fill=color)
    draw.text((x + w + 6, y - 1), f"{label} {value}", font=F_SM, fill=(230, 230, 230))


def draw_series(draw, box, series, labels, vmin, vmax, colors):
    x0, y0, x1, y1 = box
    draw.rectangle(box, outline=(90, 90, 90))
    draw.line([x0, (y0 + y1) / 2, x1, (y0 + y1) / 2], fill=(70, 70, 70))
    n = max(len(s) for s in series)
    if n < 2:
        return
    for s, color, lab in zip(series, colors, labels):
        if not s:
            continue
        arr = np.asarray(s, dtype=float)
        arr = np.clip((arr - vmin) / (vmax - vmin), 0, 1)
        xs = np.linspace(x0 + 2, x1 - 2, len(arr))
        ys = y1 - 3 - arr * (y1 - y0 - 6)
        draw.line(list(zip(xs.tolist(), ys.tolist())), fill=color, width=2)
        draw.text((x1 - 74, y0 + 2 + 13 * colors.index(color)), lab, font=F_SM, fill=color)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=24.0)
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--out", default="safety_results/demo/driving_demo.mp4")
    ap.add_argument("--disturb-start", type=float, default=8.0)
    ap.add_argument("--disturb-end", type=float, default=17.0)
    ap.add_argument("--policy", default="rl", choices=["rl", "reference"],
                    help="'rl' = trained PPO policy (default), 'reference' = hand-written P-controller")
    ap.add_argument("--model", default="", help="path to the PPO .zip (default: newest drive_shielded run)")
    ap.add_argument("--algo", default="PPO", choices=["PPO", "SAC", "DDPG"])
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--disturb-mix", type=float, default=1.0,
                    help="how strongly the injected action overrides the policy (0..1)")
    ap.add_argument("--max-episode-steps", type=int, default=400)
    ap.add_argument("--shield", type=int, default=1,
                    help="1 = CBF filter active, 0 = raw policy (for the A/B comparison)")
    ap.add_argument("--ref-speed", type=float, default=16.0,
                    help="reference lane-following speed [km/h] (must match run_scenarios.py)")
    ap.add_argument("--traffic", type=int, default=0,
                    help="number of autopilot NPC vehicles to spawn (live traffic)")
    ap.add_argument("--walkers", type=int, default=0,
                    help="number of pedestrians to spawn")
    ap.add_argument("--scenario", default="none",
                    choices=["none", "roadblock", "lead_brake", "crossing", "ped_cross", "cutin"],
                    help="interactive traffic scenario that challenges the barriers")
    ap.add_argument("--allow-brake", type=int, default=1,
                    help="1 = the action's throttle channel is [-1,1] with negative = brake, "
                         "so the safety filter can actually brake (needed for headway)")
    ap.add_argument("--config", default="SAFETY_RL",
                    help="config used for the episode (reward params, smoothing, action space); "
                         "must be the same one the policy was trained with")
    args = ap.parse_args()
    REF_SPEED[0] = float(args.ref_speed)

    obs_space, encode_fn, _ = create_encode_state_fn(None, CONFIG["state"])
    env = CarlaRouteEnv(obs_res=(160, 80), host="127.0.0.1", port=2000,
                        reward_fn=reward_functions[CONFIG["reward_fn"]],
                        observation_space=obs_space, encode_state_fn=encode_fn,
                        decode_vae_fn=None, fps=args.fps,
                        action_smoothing=CONFIG["action_smoothing"],
                        action_space_type="continuous", activate_spectator=True,
                        activate_render=False, start_carla=False,
                        town=CONFIG.get("map", "Town10HD_Opt"),
                        allow_brake=bool(args.allow_brake),
                        # the HUD's driver-view panel needs the dashboard frame
                        # even though the policy only consumes the vector state
                        use_camera_obs=True)
    # chase camera at 640x360 (smaller than the default 1120x560 viewer)
    import gymnasium as gym
    env = gym.wrappers.TimeLimit(env, max_episode_steps=args.max_episode_steps)
    cfg = CBFFilterConfig(dt=1.0 / args.fps, lane_half_width=1.75, lane_margin=0.30,
                          speed_limit=13.9, a_lat_max=2.5, horizon=CONFIG["safety"]["horizon"],
                          throttle_min=-1.0 if args.allow_brake else 0.0)
    env = SafetyShieldWrapper(env, config=cfg, penalty=0.0, enabled=bool(args.shield))

    # top-down BEV camera attached to the ego
    bev = {"img": None}
    bp = env.unwrapped.world.get_blueprint_library().find("sensor.camera.rgb")
    bp.set_attribute("image_size_x", "640")
    bp.set_attribute("image_size_y", "360")
    bp.set_attribute("fov", "80")
    cam = env.unwrapped.world.spawn_actor(
        bp, carla.Transform(carla.Location(x=0.0, z=45.0), carla.Rotation(pitch=-90)),
        attach_to=env.unwrapped.vehicle.actor)
    cam.listen(lambda im: bev.update(img=np.frombuffer(im.raw_data, dtype=np.uint8)
                                     .reshape(im.height, im.width, 4)[:, :, :3][:, :, ::-1].copy()))

    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (W, H))
    rows = []
    model = None
    if args.policy == "rl":
        from stable_baselines3 import PPO as _PPO, SAC as _SAC, DDPG as _DDPG
        path = args.model
        if not path:
            import glob as _glob
            cands = sorted(_glob.glob(os.path.join("tensorboard", "PPO_drive_shielded_*", "final_model.zip")),
                           key=os.path.getmtime, reverse=True)
            cands += sorted(_glob.glob(os.path.join("tensorboard", "PPO_shielded_*", "final_model.zip")),
                            key=os.path.getmtime, reverse=True)
            path = cands[0] if cands else ""
        if not path or not os.path.exists(path):
            raise SystemExit("no PPO model found; pass --model <path.zip>")
        model = {"PPO": _PPO, "SAC": _SAC, "DDPG": _DDPG}[args.algo].load(path, env=env, device="cuda")
        print("using RL policy:", path, flush=True)

    obs, _ = env.reset(seed=args.seed)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import scenarios as _scenarios
    scn = _scenarios.build(args.scenario, env.unwrapped)
    scn.spawn()
    for _a in scn.actors:
        env.register_obstacle(_a)   # measure these along the route (curve-exact)
    print(f"scenario: {args.scenario} - {scn.notes}", flush=True)
    traffic = None
    if args.traffic > 0 or args.walkers > 0:
        import traffic as _traffic
        traffic = _traffic.Traffic(env.unwrapped, n_vehicles=args.traffic,
                                   n_walkers=args.walkers, seed=args.seed)
        traffic.spawn()
        print("traffic:", traffic.notes, flush=True)
    hist = {"steer": [], "throttle": [], "minh": [], "speed": [], "itv": []}
    t0 = time.time()
    n = int(args.seconds * args.fps)
    total_steps = n
    min_gap_seen = float('inf')
    for k in range(n):
        t = k / args.fps
        scn.update(t)
        disturb = 1.0 if args.disturb_start <= t < args.disturb_end else 0.0
        if model is not None:
            action, _ = model.predict(obs, deterministic=True)
            action = np.asarray(action, dtype=np.float32)
            if disturb > 0:
                # inject a bad command: hard left + full throttle
                bad = np.array([1.0, 1.0], dtype=np.float32)
                action = (1 - args.disturb_mix * disturb) * action + (args.disturb_mix * disturb) * bad
        else:
            action = controller(env.unwrapped, disturb)
        if not args.allow_brake:
            action = np.array([action[0], max(float(action[1]), 0.0)], dtype=np.float32)
        obs, rew, term, trunc, info = env.step(action)
        s = info.get("safety", {})
        st = env.ego_state()
        u = s.get("barriers", [0, 0, 0, 0])

        chase = env.unwrapped.viewer_image
        dash = env.unwrapped.observation
        bev_img = bev["img"]
        if bev_img is None:
            continue

        canvas = Image.new("RGB", (W, H), (18, 18, 20))
        canvas.paste(Image.fromarray(chase).resize((640, TOP_H)), (0, 0))
        canvas.paste(Image.fromarray(bev_img).resize((640, TOP_H)), (640, 0))
        canvas.paste(Image.fromarray(dash).resize((320, 160)), (0, BOT + 20))
        d = ImageDraw.Draw(canvas)
        d.rectangle([0, 0, 640, TOP_H], outline=(60, 60, 60))
        d.rectangle([640, 0, W, TOP_H], outline=(60, 60, 60))
        d.text((648, 48), "map BEV (top-down)", font=F_SM, fill=(255, 255, 120))
        d.text((8, BOT), "driver view", font=F_SM, fill=(255, 255, 120))
        # ---- unmissable status banner: the whole point of the comparison
        if args.shield:
            if s.get("intervened"):
                banner(d, 0, 0, W, 44, "CBF SAFETY FILTER: ON  ->  OVERRIDING THE POLICY",
                       (200, 40, 40),
                       f"step {k}: the policy's action was replaced by the closest SAFE action")
            else:
                banner(d, 0, 0, W, 44, "CBF SAFETY FILTER: ON  (monitoring - action allowed)",
                       (30, 110, 60), "no override needed at this step")
        else:
            banner(d, 0, 0, W, 44, "NO SAFETY FILTER  (raw policy)", (90, 90, 95),
                   "whatever the policy asks for goes straight to the car")
        # ---- big decisive numbers (bottom-left, under the driver view)
        big_numbers(d, 12, BOT + 190, st.v * 3.6, float(np.min(u)), env.filter.num_interventions,
                    env.filter.num_calls)
        # ---- whole-run intervention strip: red = the filter overrode the policy
        event_strip(d, 10, H - 16, W - 20, 10, hist["itv"], total_steps)
        d.text((10, H - 32), "safety-filter overrides over the whole run "
                             "(green = raw policy allowed, red = CBF overrode it)",
               font=F_SM, fill=(200, 200, 200))

        # ---- state block
        wp_i, wp_n = env.unwrapped.current_waypoint_index, len(env.unwrapped.route_waypoints)
        man = maneuver_name(env.unwrapped.next_road_maneuver)
        d.text((336, BOT + 4), "DRIVING STATE", font=F_MD, fill=(255, 255, 120))
        state = [
            f"speed      {st.v * 3.6:6.1f} km/h",
            f"lateral e_y{st.e_y:+6.2f} m",
            f"heading e_psi{np.degrees(st.e_psi):+6.1f} deg",
            f"curvature  {st.kappa:6.4f} 1/m",
            f"target speed {env.filter.curve_speed_limit(st.kappa) * 3.6:5.1f} km/h",
        ]
        for i, line in enumerate(state):
            d.text((336, BOT + 24 + 17 * i), line, font=F_SM, fill=(225, 225, 225))

        d.text((336, BOT + 122), "CONTROL", font=F_MD, fill=(255, 255, 120))
        d.text((336, BOT + 142), f"steer (RL)   {action[0]:+.3f}", font=F_SM, fill=(225, 225, 225))
        d.text((336, BOT + 159), f"throttle(RL) {action[1]:+.3f}", font=F_SM, fill=(225, 225, 225))
        d.text((336, BOT + 176), f"steer (safe) {float(s.get('action', [0, 0])[0]) if 'action' in s else 0:+.3f}"
               if "action" in s else "", font=F_SM, fill=(225, 225, 225))
        d.text((336, BOT + 193), f"intervention {int(s.get('intervened', 0))}", font=F_SM,
               fill=(255, 90, 90) if s.get("intervened") else (160, 230, 160))

        d.text((336, BOT + 222), "NAVIGATION", font=F_MD, fill=(255, 255, 120))
        d.text((336, BOT + 242), f"next maneuver {man}", font=F_SM, fill=(225, 225, 225))
        d.text((336, BOT + 259), f"route {wp_i}/{wp_n} wp  ({100 * wp_i / max(wp_n, 1):4.1f} %)", font=F_SM,
               fill=(225, 225, 225))
        d.text((336, BOT + 276), f"dist to goal {wp_n - wp_i:4d} m", font=F_SM, fill=(225, 225, 225))

        # ---- safety bars
        # NOTE: keep this header short - at x=620 with F_MD the old
        # "SAFETY (CBF barriers, h >= 0 is safe)" ran into the "t = .. s"
        # label of the time-series column at x=830 and the two overlapped.
        d.text((620, BOT + 4), "SAFETY  (CBF barriers)", font=F_MD, fill=(255, 255, 120))
        names = ["lane right", "lane left ", "speed     ", "curve     ", "gap / TTC "]
        for i, (nm, hv) in enumerate(zip(names, u)):
            frac = min(hv / 4.0, 1.0) if hv < 1e2 else 1.0
            bar(d, 620, BOT + 26 + 26 * i, 130, 15, frac,
                (90, 200, 110) if hv > 0 else (235, 80, 80), nm, f"{hv:+.2f}")
        mh = s.get("min_barrier", 0.0)
        d.text((620, BOT + 162), f"min_i h_i = {mh:+.3f}", font=F_MD,
               fill=(160, 230, 160) if mh > 0 else (255, 90, 90))
        d.text((620, BOT + 184), f"filter interventions: {env.filter.num_interventions}/"
                                 f"{env.filter.num_calls}", font=F_SM, fill=(225, 225, 225))
        if disturb > 0:
            d.text((620, BOT + 204), "*** DISTURBANCE INJECTED ***", font=F_SM, fill=(255, 90, 90))
        d.text((620, BOT + 224), f"scenario: {scn.name}", font=F_SM, fill=(255, 255, 120))
        d.text((620, BOT + 246), "bar > 0 = safe, red bar = violated", font=F_SM, fill=(190, 190, 190))

        hist["steer"].append(action[0])
        hist["throttle"].append(action[1])
        hist["minh"].append(mh)
        hist["speed"].append(st.v * 3.6)
        hist["itv"].append(int(bool(s.get("intervened"))))
        win = 90
        draw_series(d, (830, BOT + 20, 1270, BOT + 90),
                    [hist["steer"][-win:], hist["throttle"][-win:]],
                    ["steer", "throttle"], -1.0, 1.0,
                    [(255, 170, 60), (80, 170, 255)])
        draw_series(d, (830, BOT + 110, 1270, BOT + 180),
                    [hist["minh"][-win:]], ["min h"], -0.6, 2.0, [(255, 90, 90)])
        draw_series(d, (830, BOT + 200, 1270, BOT + 270),
                    [hist["speed"][-win:]], ["speed km/h"], 0.0, 60.0, [(120, 230, 140)])
        d.text((830, BOT + 274), "time series (last 6 s)", font=F_SM, fill=(180, 180, 180))
        d.text((830, BOT + 4), f"t = {t:5.1f} s", font=F_MD, fill=(255, 255, 120))

        writer.write(cv2.cvtColor(np.array(canvas), cv2.COLOR_RGB2BGR))
        rows.append(dict(t=round(t, 2), speed_kmh=round(st.v * 3.6, 2),
                         lateral=round(st.e_y, 4), heading_deg=round(float(np.degrees(st.e_psi)), 2),
                         steer_rl=round(float(action[0]), 4), throttle_rl=round(float(action[1]), 4),
                         min_h=round(mh, 4), intervened=int(bool(s.get("intervened"))),
                         disturbance=int(bool(disturb))))
        if k % 30 == 0:
            print(f"  frame {k}/{n}  t={t:4.1f}s  v={st.v * 3.6:5.1f}km/h  e_y={st.e_y:+.2f}  "
                  f"minh={mh:+.3f}  int={int(bool(s.get('intervened')))}", flush=True)
        if term or trunc:
            # Make the outcome unambiguous: freeze on this frame, stamp the
            # verdict in huge letters, and stop.  (Previously the run silently
            # continued on a new route, which made the comparison unreadable.)
            # NOTE (2026-10-05): this used to print "*** COLLISION ***" for *any*
            # terminal, including the reward's stall termination - so a run in
            # which the filter had correctly stopped the car behind an obstacle
            # was labelled as a crash.  The verdict now follows the actual cause.
            collided = bool(getattr(env.unwrapped, "collision_flag", False))
            reason = ""
            for item in reversed(getattr(env.unwrapped, "extra_info", []) or []):
                if isinstance(item, str) and item and item != "Running...":
                    reason = item
                    break
            if collided:
                title, sub, colour = ("*** COLLISION ***", "the car hit an obstacle",
                                      (200, 20, 20))
            elif term:
                title, sub, colour = ("*** EPISODE ENDED ***",
                                      f"reward termination: {reason or 'unknown'}",
                                      (200, 120, 20))
            else:
                title, sub, colour = ("*** TIME LIMIT ***",
                                      "the 400-step episode budget ended", (70, 70, 90))
            # hold the verdict until the end of the clip so that the two halves of
            # the comparison stay aligned ("top frozen on the crash, bottom keeps
            # driving safely")
            for _ in range(max(int(1.5 * args.fps), n - k)):
                banner(d, 0, TOP_H // 2 - 40, W, 80, title, colour, sub)
                writer.write(cv2.cvtColor(np.array(canvas), cv2.COLOR_RGB2BGR))
            print(f"  episode ended at t={t:.1f}s (collided={collided}, reason={reason!r}, "
                  f"truncated={bool(trunc)}) -> recording stops here", flush=True)
            break
        # A physical collision event is not always raised for a soft touch, so
        # also treat "measured gap <= 0.5 m" as CONTACT: that is what the safety
        # layer exists to prevent.  The bottom panel then freezes on the verdict.
        gap_now = s.get("gap")
        if gap_now is not None:
            min_gap_seen = min(min_gap_seen, float(gap_now))
        if min_gap_seen < 0.5:
            for _ in range(max(int(1.5 * args.fps), n - k)):
                banner(d, 0, TOP_H // 2 - 40, W, 80,
                       "*** CONTACT / UNSAFE ***", (200, 20, 20),
                       f"the safety barrier was violated until the gap reached "
                       f"{min_gap_seen:.2f} m")
                writer.write(cv2.cvtColor(np.array(canvas), cv2.COLOR_RGB2BGR))
            print(f"  CONTACT at t={t:.1f}s (min gap {min_gap_seen:.2f} m) -> recording stops",
                  flush=True)
            break

    writer.release()
    cam.destroy()
    scn.destroy()
    if traffic is not None:
        traffic.destroy()
    csv_path = out.replace(".mp4", "_telemetry.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[video] {out} ({os.path.getsize(out) / 1e6:.1f} MB, {len(rows)} frames, "
          f"{time.time() - t0:.0f}s wall)")
    print(f"[video] telemetry -> {csv_path}")
    env.close()


if __name__ == "__main__":
    main()
