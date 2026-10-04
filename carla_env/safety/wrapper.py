"""Gymnasium wrapper that turns any CarlaRouteEnv policy into a *shielded* one."""
from __future__ import annotations

import csv
import os
from collections import deque

import gymnasium as gym
import numpy as np

import carla
from carla_env.safety.cbf import CBFFilterConfig, CBFSafetyFilter, EgoState
from carla_env.wrappers import vector


def _wrap_pi(angle: float) -> float:
    return float((angle + np.pi) % (2 * np.pi) - np.pi)


def _curvature(p0, p1, p2) -> float:
    """Circumscribed-circle curvature through three 2-D points."""
    pts = np.array([[p0.x, p0.y], [p1.x, p1.y], [p2.x, p2.y]], dtype=np.float64)
    a = np.linalg.norm(pts[1] - pts[2])
    b = np.linalg.norm(pts[0] - pts[2])
    c = np.linalg.norm(pts[0] - pts[1])
    if min(a, b, c) < 1e-6:
        return 0.0
    area2 = abs((pts[1][0] - pts[0][0]) * (pts[2][1] - pts[0][1])
                - (pts[2][0] - pts[0][0]) * (pts[1][1] - pts[0][1]))
    return float(2.0 * area2 / (a * b * c))


class SafetyShieldWrapper(gym.Wrapper):
    """Applies the CBF safety filter to every action and logs safety metrics.

    Parameters
    ----------
    env : CarlaRouteEnv
    config : CBFFilterConfig | None
        Physical/tuning parameters of the barrier functions.
    penalty : float
        If > 0, subtract ``penalty * max(0, -min_i h_i(x))`` from the reward so
        the learner is *also* nudged towards the safe set (reward shaping in
        addition to the hard filter).
    log_path : str | None
        Optional CSV path receiving one row per episode.
    """

    def __init__(self, env, config: CBFFilterConfig | None = None,
                 penalty: float = 0.0, log_path: str | None = None,
                 enabled: bool = True):
        super().__init__(env)
        self.config = config or CBFFilterConfig(dt=1.0 / float(env.unwrapped.fps))
        self.filter = CBFSafetyFilter(self.config)
        self.enabled = enabled
        self.penalty = penalty
        self.log_path = log_path
        self.episode_index = -1
        self._episode_rows: list[dict] = []
        self.ep_history = deque(maxlen=2000)
        self._prev_e_y: float | None = None
        self._e_y_dot: float = 0.0
        self._rate_ema: float = 0.5
        # obstacle (lead vehicle / static prop) detection cache - the actor list
        # query is an RPC, so it is refreshed every few steps instead of每步
        self._obstacle_cache: tuple[float, float] | None = None
        self._obstacle_step = 0
        self.obstacle_refresh = 5
        self.obstacle_max_dist = 60.0
        self.obstacle_half_width = 1.9
        self.obstacle_cone = 0.30
        """The gate widens with distance: |lateral| <= half_width + cone * forward.
        A *fixed* lateral gate in the ego body frame silently missed obstacles
        that were dead ahead on the route but appeared lateral because the ego
        had a heading error (measured: a roadblock 45 m ahead was only detected
        at 0.8 m, which is far too late for the barrier to act)."""
        self._registered_obstacles: list = []
        """Actors placed by a scenario.  For these we measure the gap as the
        *arc length along the route* (waypoint index difference), which is exact
        on curved roads - a local lane frame gives a bogus lateral offset for a
        point 40 m ahead that sits around a bend."""

    def register_obstacle(self, actor) -> None:
        """Tell the safety layer that this actor is a scenario obstacle."""
        if actor is not None:
            self._registered_obstacles.append(actor)

    def _arc_gap(self, actor) -> tuple[float, float] | None:
        """(gap, lead_speed) measured along the route for a registered actor."""
        env = self.env.unwrapped
        wps = env.route_waypoints
        i = int(env.current_waypoint_index)
        loc = vector(actor.get_transform().location)
        best_j, best_d = None, None
        for j in range(i, min(i + 80, len(wps))):
            d = np.linalg.norm(vector(wps[j][0].transform.location)[:2] - loc[:2])
            if best_d is None or d < best_d:
                best_d, best_j = d, j
        if best_j is None:
            return None
        # PORT FIX (2026-10-04): an actor that has just been spawned has *no*
        # transform until the world ticks once - ``get_transform()`` returns
        # (0, 0, 0).  Projecting that onto the route picks the waypoint closest
        # to the world origin and reports it as ``gap = 0.0 m``, i.e. "the
        # obstacle is on top of us".  In the demo recorder that tripped the
        # "measured gap < 0.5 m => CONTACT" rule on frame 0 and every blocking
        # scenario clip was cut down to a single frame with the car standing
        # still.  An obstacle that is not actually near the route corridor is
        # not an obstacle ahead of us, so report nothing instead.
        if best_d > 5.0:
            return None
        # waypoints are ~1 m apart; use the true accumulated length
        gap = 0.0
        prev = wps[i][0].transform.location
        for j in range(i + 1, best_j + 1):
            cur = wps[j][0].transform.location
            gap += cur.distance(prev)
            prev = cur
        lead = 0.0
        try:
            lead = max(float(np.dot(vector(actor.get_velocity()),
                                     vector(env.current_waypoint.transform.rotation.get_forward_vector()))), 0.0)
        except Exception:  # noqa: BLE001
            pass
        return gap, lead

    # ------------------------------------------------------- obstacle lookup
    def nearest_obstacle(self, force: bool = False) -> tuple[float, float] | None:
        """(gap, lead_speed) of the nearest obstacle ahead inside the corridor."""
        env = self.env.unwrapped
        # scenario obstacles are measured along the route (curve-exact)
        for actor in list(self._registered_obstacles):
            try:
                g = self._arc_gap(actor)
            except Exception:  # noqa: BLE001
                g = None
            if g is not None and 0.0 <= g[0] <= self.obstacle_max_dist:
                self._obstacle_cache = g
                return g
        if not force and self._obstacle_cache is not None and \
                self._obstacle_step % self.obstacle_refresh != 0:
            self._obstacle_step += 1
            return self._obstacle_cache
        self._obstacle_step += 1
        tf = env.vehicle.get_transform()
        # Reference frame = the LANE, not the ego heading.  Using the body frame
        # made the detection flicker (an obstacle dead ahead disappeared from the
        # corridor whenever the ego had a heading error), which is exactly when
        # the headway barrier is needed.
        lane_fwd = vector(env.current_waypoint.transform.rotation.get_forward_vector())
        fwd = np.array([lane_fwd[0], lane_fwd[1], 0.0])
        n = np.linalg.norm(fwd[:2])
        fwd = fwd / n if n > 1e-6 else np.array([1.0, 0.0, 0.0])
        right = np.array([-fwd[1], fwd[0], 0.0])
        pos = vector(tf.location)
        ego_id = env.vehicle.actor.id
        best: tuple[float, float] | None = None
        try:
            actors = env.world.get_actors()
        except Exception:  # noqa: BLE001
            return None
        for a in actors:
            tid = a.type_id
            if a.id == ego_id or not (tid.startswith("vehicle.") or tid.startswith("static.prop")):
                continue
            d = vector(a.get_transform().location) - pos
            fdist = float(np.dot(d, fwd))
            ldist = float(np.dot(d, right))
            gate = self.obstacle_half_width + self.obstacle_cone * fdist
            if 0.0 < fdist <= self.obstacle_max_dist and abs(ldist) < gate:
                lead_speed = 0.0
                if tid.startswith("vehicle."):
                    vel = a.get_velocity()
                    lead_speed = float(np.dot(vector(vel), fwd))
                if best is None or fdist < best[0]:
                    best = (fdist, max(lead_speed, 0.0))
        self._obstacle_cache = best
        return best

    # ------------------------------------------------------------- state
    def ego_state(self) -> EgoState:
        env = self.env.unwrapped
        tf = env.vehicle.get_transform()
        fwd = vector(tf.rotation.get_forward_vector())
        wp = env.current_waypoint
        wp_loc = wp.transform.location
        wp_fwd = vector(wp.transform.rotation.get_forward_vector())
        # signed lateral offset: z-component of the cross product
        d = vector(tf.location) - vector(wp_loc)
        e_y = float((wp_fwd[0] * d[1] - wp_fwd[1] * d[0]))
        # Heading error, defined so that a positive e_psi means "the vehicle
        # heading is rotated towards increasing e_y".  With this convention the
        # sign of the measured lateral rate and of the heading error agree, which
        # the one-step prediction relies on (an inverted sign made the model
        # predict that a swerving car was straightening out).
        e_psi = _wrap_pi(float(np.arctan2(fwd[1], fwd[0]) - np.arctan2(wp_fwd[1], wp_fwd[0])))
        # curvature from three route waypoints around the current one
        wps = env.route_waypoints
        i = int(env.current_waypoint_index)
        i0, i2 = max(0, i - 4), min(len(wps) - 1, i + 4)
        kappa = 0.0
        if i2 > i0:
            kappa = _curvature(wps[i0][0].transform.location,
                               wps[i][0].transform.location,
                               wps[i2][0].transform.location)
        v_ms = float(env.vehicle.get_speed()) / 3.6
        # measured lateral rate (EMA-smoothed finite difference) -> robust CBF
        dt = 1.0 / float(env.fps)
        if self._prev_e_y is not None:
            raw = (e_y - self._prev_e_y) / dt
            self._e_y_dot = self._rate_ema * raw + (1 - self._rate_ema) * self._e_y_dot
        self._prev_e_y = e_y
        obs = self.nearest_obstacle()
        gap = None if obs is None else obs[0]
        v_lead = 0.0 if obs is None else obs[1]
        return EgoState(e_y=e_y, e_psi=e_psi, v=v_ms, kappa=kappa,
                        e_y_dot=self._e_y_dot, gap=gap, v_lead=v_lead)

    # ------------------------------------------------------------- gym API
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        if self.episode_index >= 0:
            self._flush_episode()
        self.episode_index += 1
        self._ep = {"interventions": 0, "steps": 0, "violations": 0,
                    "min_barrier": float("inf"), "collision": 0,
                    "distance": 0.0, "reward": 0.0}
        self.filter = CBFSafetyFilter(self.config)
        self._prev_e_y = None
        self._e_y_dot = 0.0
        self._obstacle_cache = None
        self._obstacle_step = 0
        info["safety"] = {"intervened": False, "min_barrier": float("inf")}
        return obs, info

    def step(self, action):
        state = self.ego_state()
        if self.enabled:
            safe = self.filter.solve(state, np.asarray(action, dtype=np.float64))
        else:  # metrics-only mode: keep the raw action, still record barriers
            from carla_env.safety.cbf import SafetyStep
            h = self.filter.barriers(state)
            safe = SafetyStep(action=np.asarray(action, dtype=np.float32),
                              intervened=False, min_barrier=float(np.min(h)),
                              barriers=h, qp_success=True, fallback=False)
        obs, reward, terminated, truncated, info = self.env.step(safe.action)

        if self.penalty > 0.0:
            reward = float(reward) - self.penalty * max(0.0, -safe.min_barrier)

        violation = safe.min_barrier < 0.0
        ep = self._ep
        ep["steps"] += 1
        ep["interventions"] += int(safe.intervened)
        ep["violations"] += int(violation)
        ep["min_barrier"] = min(ep["min_barrier"], safe.min_barrier)
        ep["reward"] += float(reward)
        ep["distance"] = float(info.get("total_distance", ep["distance"]))
        if terminated:
            ep["terminations"] = ep.get("terminations", 0) + 1
        if bool(getattr(self.env.unwrapped, "collision_flag", False)):
            ep["collision"] += 1
        self.ep_history.append((state.v, safe.min_barrier, float(safe.action[0]), float(safe.action[1])))

        info["safety"] = {
            "intervened": bool(safe.intervened),
            "min_barrier": float(safe.min_barrier),
            "barriers": safe.barriers.tolist(),
            "action": safe.action.tolist(),
            "action_rl": [float(np.asarray(action)[0]), float(np.asarray(action)[1])],
            "qp_success": bool(safe.qp_success),
            "fallback": bool(safe.fallback),
            "speed_kmh": state.v * 3.6,
            "lateral_offset": float(state.e_y),
            # raw headway (None when no obstacle is inside the corridor) - the
            # demo videos use it to tell "barrier violated" from real contact
            "gap": None if state.gap is None else float(state.gap),
        }
        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------- logging
    def _flush_episode(self) -> None:
        if not hasattr(self, "_ep"):
            return
        ep = self._ep
        steps = max(ep["steps"], 1)
        row = {
            "episode": self.episode_index,
            "steps": ep["steps"],
            "reward": round(ep["reward"], 4),
            "distance_m": round(ep["distance"], 2),
            "mean_speed_kmh": round(float(np.mean([h[0] for h in self.ep_history])) * 3.6, 3)
            if self.ep_history else 0.0,
            "min_barrier": round(ep["min_barrier"], 4),
            "violation_steps": ep["violations"],
            "interventions": ep["interventions"],
            "intervention_rate": round(ep["interventions"] / steps, 4),
            "collisions": ep["collision"],
            "terminations": ep.get("terminations", 0),
        }
        self._episode_rows.append(row)
        if self.log_path:
            os.makedirs(os.path.dirname(os.path.abspath(self.log_path)), exist_ok=True)
            new_file = not os.path.exists(self.log_path)
            with open(self.log_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(row.keys()))
                if new_file:
                    writer.writeheader()
                writer.writerow(row)

    def close(self):
        self._flush_episode()
        return self.env.close()

    def safety_stats(self) -> dict:
        stats = self.filter.stats()
        if self._episode_rows:
            stats["episodes"] = len(self._episode_rows)
            stats["episode_violation_rate"] = float(np.mean(
                [r["violation_steps"] / max(r["steps"], 1) for r in self._episode_rows]))
            stats["collisions"] = int(sum(r["collisions"] for r in self._episode_rows))
        return stats
