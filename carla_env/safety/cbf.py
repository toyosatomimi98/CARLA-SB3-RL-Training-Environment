"""Control-Barrier-Function (CBF) safety filter for the lane-following agent.

The reinforcement-learning policy proposes a normalised action
``u_rl = (steer, throttle) in [-1, 1] x [0, 1]``.  A *safety filter* solves, at
every control step, the small quadratic program

    u* = argmin_u  || W (u - u_rl) ||^2
         s.t.      h_i( x_{k+1}(u) ) >= (1 - alpha_i) h_i( x_k ),   i = 1..m
                   u_min <= u <= u_max

where ``h_i`` are control barrier functions encoding the safety requirements
and ``x_{k+1}(u)`` is the one-step prediction of a kinematic bicycle model.
The filter therefore changes the learned action *as little as possible* while
keeping the closed loop inside the safe set  C = { x : h_i(x) >= 0 }.

Chosen safety requirements (the "safety constraints of my choice"):
  * h_lane^+/-  : stay inside the lane corridor      h = d_lane -/+ e_y
  * h_speed     : do not exceed the road speed limit h = v_max - v
  * h_curve     : slow down for the current curvature h = sqrt(a_lat_max/kappa) - v
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import List

import numpy as np
from scipy.optimize import minimize


@dataclass
class CBFFilterConfig:
    """Physical / tuning parameters of the filter."""

    dt: float = 1.0 / 15.0          # control period [s] (matches env fps)
    wheelbase: float = 2.40         # m (Lincoln MKZ wheelbase, CARLA default 2.4)
    steer_max: float = 0.60         # rad, steering angle at |steer| = 1
    accel_max: float = 3.0          # m/s^2 at throttle = 1
    brake_max: float = 6.0          # m/s^2 at throttle = 0 while moving
    drag: float = 0.02              # 1/s linear drag
    lane_half_width: float = 1.75   # m, lateral corridor half width
    lane_margin: float = 0.30
    """Extra conservatism: the CBF enforces ``|e_y| <= lane_half_width - margin``
    so that the true lane boundary is respected despite model/prediction error."""
    speed_limit: float = 13.9       # m/s (= 50 km/h)
    a_lat_max: float = 2.5          # m/s^2 lateral acceleration budget (curve speed)
    d_min: float = 5.0              # m, static part of the safe following gap
    tau: float = 1.2                # s, time headway (speed-dependent part)
    alpha_lane: float = 0.30        # class-K linear gains of the discrete-time CBFs
    alpha_speed: float = 0.30
    alpha_curve: float = 0.30
    alpha_gap: float = 0.30
    throttle_min: float = 0.0
    """Lower bound of the throttle channel: 0 = coast-only (no brake), -1 = the
    environment exposes a real brake channel (see CarlaRouteEnv.allow_brake)."""
    standstill_recovery_speed: float = 0.2
    """Below this speed [m/s] the lane constraint is *suspended* so the car can
    drive out of a "stopped at an angle" deadlock: while stopped, accelerating
    would increase the lateral drift, so the lane barrier would otherwise keep
    the car standing forever (measured: the vehicle halted 37 m before a
    roadblock and never moved again)."""
    slack_penalty: float = 2.0
    """Cost per unit of barrier violation when the QP is relaxed with slack
    variables.  A hard constraint is infeasible whenever the state is already
    unsafe, which made the filter fall back to full braking (the car froze)."""
    slack_max: float = 12.0
    liveness_weight: float = 5.0
    """Penalty on losing speed relative to the reference speed at the end of the
    horizon.  This is a *liveness* term: a purely safety-oriented filter can
    always choose to stand still (it is the smallest possible intervention), so
    without a progress term "safe" degenerates into "frozen"."""
    v_min_live: float = 2.0
    """The liveness term only bites below this speed [m/s] (~7 km/h): the filter
    is allowed to slow down for safety, but not to freeze."""
    liveness_clear_gap: float = 20.0
    """The liveness term is switched OFF when an obstacle is closer than this
    [m]: in front of a roadblock the safe answer IS to slow down / stop, and a
    single weight cannot encode both "keep moving" and "stop for the obstacle"
    (measured: an unconditional liveness term made the filter prefer violating
    the headway barrier over braking)."""
    steer_weight: float = 1.0       # QP weights
    throttle_weight: float = 1.0
    slack_penalty: float = 1e4      # penalty if the QP has to be relaxed
    intervention_tol: float = 0.02  # ||u* - u_rl||_inf above which we log "intervened"
    horizon: int = 15               # number of predicted steps in the CBF
    """The filter predicts ``horizon`` steps ahead with the candidate action held
    constant and requires the lane corridor to hold over the whole horizon.  A
    single-step CBF has no authority to correct an already-large heading error
    (steering only changes the *rate*), which made the shield let the car drift
    out of the lane."""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class EgoState:
    """Minimal ego state used by the barrier functions (SI units)."""

    e_y: float      # signed lateral offset from the lane centre [m]
    e_psi: float    # heading error w.r.t. the lane direction [rad]
    v: float        # longitudinal speed [m/s]
    kappa: float    # curvature of the reference path at the ego position [1/m]
    e_y_dot: float | None = None
    """Measured lateral rate d(e_y)/dt [m/s].  When provided the predictor uses
    the *measured* value instead of the model term ``v sin(e_psi)``.  This makes
    the barrier robust to the mismatch between the kinematic model and the
    full CARLA vehicle dynamics (suspension, tyre slip, action smoothing)."""
    gap: float | None = None
    """Distance to the nearest obstacle ahead inside the ego's lane corridor [m].
    ``None`` = no obstacle ahead, i.e. the headway barrier is inactive."""
    v_lead: float = 0.0
    """Longitudinal speed of that obstacle [m/s] (0 for a static prop)."""


@dataclass
class SafetyStep:
    """Result of one filter invocation."""

    action: np.ndarray
    """Filtered (safe) action handed to the simulator."""
    intervened: bool
    """True when the filter changed the proposed action beyond ``intervention_tol``."""
    min_barrier: float
    """Smallest barrier value *before* the step; < 0 means the safe set was left."""
    barriers: np.ndarray = field(default_factory=lambda: np.zeros(4))
    qp_success: bool = True
    """False when even the relaxed QP failed and the fallback was used."""
    fallback: bool = False


class CBFSafetyFilter:
    """Discrete-time CBF filter (2-D QP solved with SLSQP)."""

    def __init__(self, config: CBFFilterConfig | None = None):
        self.cfg = config or CBFFilterConfig()
        self.num_calls = 0
        self.num_interventions = 0
        self.num_fallbacks = 0
        self.last_safety_step: SafetyStep | None = None

    # ------------------------------------------------------------------ model
    def _action_to_physical(self, action: np.ndarray) -> tuple[float, float]:
        steer = float(np.clip(action[0], -1.0, 1.0))
        throttle = float(np.clip(action[1], self.cfg.throttle_min, 1.0))
        delta = steer * self.cfg.steer_max
        accel = throttle * self.cfg.accel_max if throttle >= 0.0 \
            else throttle * self.cfg.brake_max
        return delta, accel

    def _predict(self, x: EgoState, action: np.ndarray) -> EgoState:
        """One-step Euler prediction of the lateral / heading / speed dynamics.

        The lateral channel is advanced with the *measured* lateral rate when it
        is available and with the steering-induced lateral acceleration
        ``v^2/L tan(delta)`` layered on top, so the barrier reacts to the real
        drift instead of trusting the (idealised) bicycle model alone.
        """
        delta, accel = self._action_to_physical(action)
        c, dt, L = self.cfg, self.cfg.dt, self.cfg.wheelbase
        rate = x.e_y_dot if x.e_y_dot is not None else x.v * np.sin(x.e_psi)
        e_y_next = x.e_y + dt * rate + 0.5 * dt * dt * (x.v ** 2 / L) * np.tan(delta)
        e_psi_next = x.e_psi + dt * ((x.v / L) * np.tan(delta) - x.v * x.kappa)
        v_next = x.v + dt * (accel - c.drag * x.v)
        v_next = max(v_next, 0.0)
        gap_next = None
        if x.gap is not None:
            # the obstacle may be moving (v_lead); the gap closes at the relative speed
            gap_next = max(x.gap - dt * (v_next - x.v_lead), 0.0)
        return EgoState(e_y=e_y_next, e_psi=e_psi_next, v=v_next, kappa=x.kappa,
                        gap=gap_next, v_lead=x.v_lead)

    def rollout(self, x: EgoState, action: np.ndarray, horizon: int | None = None) -> List[EgoState]:
        """Predict ``horizon`` states under a constant action."""
        h = self.cfg.horizon if horizon is None else horizon
        states = [x]
        cur = x
        for _ in range(h):
            cur = self._predict(cur, action)
            states.append(cur)
        return states

    def curve_speed_limit(self, kappa: float) -> float:
        kappa_eff = max(abs(kappa), 1e-4)
        return min(float(np.sqrt(self.cfg.a_lat_max / kappa_eff)), self.cfg.speed_limit)

    def barriers(self, x: EgoState) -> np.ndarray:
        """h(x) = [lane_right, lane_left, speed, curve, gap]  (>= 0 means safe)."""
        c = self.cfg
        d = c.lane_half_width - c.lane_margin
        # Headway / time-to-collision barrier: the required gap grows with speed,
        #   h_gap(x) = gap - (d_min + tau * v)
        # It is inactive when no obstacle is inside the lane corridor ahead.
        h_gap = 1e3 if x.gap is None else x.gap - (c.d_min + c.tau * x.v)
        return np.array([
            d - x.e_y,                           # do not leave to the right
            d + x.e_y,                           # do not leave to the left
            c.speed_limit - x.v,
            self.curve_speed_limit(x.kappa) - x.v,
            h_gap,                               # do not run into the obstacle ahead
        ])

    # ------------------------------------------------------------------- QP
    def _constraints(self, x: EgoState, alpha: np.ndarray):
        h_now = self.barriers(x)
        h = self.cfg.horizon

        def fun(u: np.ndarray) -> np.ndarray:
            states = self.rollout(x, u)
            out = []
            recovering = x.v < self.cfg.standstill_recovery_speed
            for j, s in enumerate(states[1:], start=1):
                b = self.barriers(s)
                # Lane corridor, in the *discrete-time CBF* form
                #   h(x_{k+j}) >= (1-alpha)^j h(x_k)
                # instead of the hard "h >= 0".  The hard form is infeasible as
                # soon as the car is already outside the corridor, which made the
                # filter fall back to full braking and the car never recovered.
                if not recovering:
                    out.append(b[0] - (1.0 - alpha[0]) ** j * h_now[0])
                    out.append(b[1] - (1.0 - alpha[1]) ** j * h_now[1])
                # the headway constraint is only added when an obstacle is
                # actually present, otherwise the (inactive) constant barrier
                # value would destroy the conditioning of the QP
                if x.gap is not None:
                    out.append(b[4])
                # speed / curvature barriers may relax at the class-K rate
                out.append(b[2] - (1.0 - alpha[2]) ** j * h_now[2])
                out.append(b[3] - (1.0 - alpha[3]) ** j * h_now[3])
            return np.asarray(out)

        return fun

    def solve(self, x: EgoState, action_rl: np.ndarray) -> SafetyStep:
        """Project the RL action onto the safe set (slack-relaxed QP).

        Decision variables: z = [steer, throttle, s_1 ... s_H] with s_j >= 0 the
        slack allowed at horizon step j.  The objective

            ||W (u - u_rl)||^2  +  M * sum_j s_j  +  kappa * (v_H - v_ref)^2

        trades "change the policy as little as possible" against "you may
        violate a barrier, but it costs" and against "do not lose speed for
        free".  The slack variables make the problem feasible in every state
        (which removes the braking deadlock), and the liveness term stops the
        filter from choosing "stand still" as the cheapest option.
        """
        self.num_calls += 1
        u_rl = np.array([float(np.clip(action_rl[0], -1.0, 1.0)),
                         float(np.clip(action_rl[1], self.cfg.throttle_min, 1.0))],
                        dtype=np.float64)
        alpha = np.array([self.cfg.alpha_lane, self.cfg.alpha_lane,
                          self.cfg.alpha_speed, self.cfg.alpha_curve,
                          self.cfg.alpha_gap])
        cons = self._constraints(x, alpha)
        w = np.array([self.cfg.steer_weight, self.cfg.throttle_weight])
        h = self.cfg.horizon
        v_ref = min(self.cfg.speed_limit, self.curve_speed_limit(x.kappa))

        def objective(z: np.ndarray) -> float:
            u = z[:2]
            s = z[2:]
            d = w * (u - u_rl)
            cost = float(np.dot(d, d)) + self.cfg.slack_penalty * float(np.sum(s))
            v_end = self.rollout(x, u)[-1].v
            shortfall = max(0.0, self.cfg.v_min_live - v_end)
            clear = not (x.gap is not None and x.gap < self.cfg.liveness_clear_gap)
            cost += (self.cfg.liveness_weight if clear else 0.0) * shortfall ** 2
            return cost

        bounds = [(-1.0, 1.0), (self.cfg.throttle_min, 1.0)] + \
                 [(0.0, self.cfg.slack_max)] * h

        def ineq(z: np.ndarray) -> np.ndarray:
            u, s = z[:2], z[2:]
            raw = cons(u)
            # rebuild the per-step structure so each slack only relaxes its own step
            out = []
            k = 0
            for j in range(1, h + 1):
                n_j = 2 + (0 if x.gap is None else 1) + 2
                out.extend(list(raw[k:k + n_j] + s[j - 1]))
                k += n_j
            return np.asarray(out)

        z0 = np.concatenate([u_rl, np.zeros(h)])
        res = minimize(objective, z0, method="SLSQP", bounds=bounds,
                       constraints=[{"type": "ineq", "fun": ineq}],
                       options={"maxiter": 40, "ftol": 1e-6})
        feasible = bool(res.success)
        if feasible:
            u_safe = np.asarray(res.x[:2], dtype=np.float64)
            slack_used = float(np.sum(res.x[2:]))
        else:
            # The slack-relaxed problem is feasible in every state, so a solver
            # failure here means a numerical problem -> coarse grid search whose
            # objective is the same trade-off (safety + minimal change).
            cands = []
            for steer_c in np.linspace(-1.0, 1.0, 21):
                for thr_c in (self.cfg.throttle_min, 0.0, 0.15, 0.35, 0.6, 1.0):
                    cand = np.array([steer_c, thr_c])
                    worst = float(np.min(cons(cand)))
                    d = w * (cand - u_rl)
                    cost = float(np.dot(d, d)) + self.cfg.slack_penalty * max(0.0, -worst)
                    v_end = self.rollout(x, cand)[-1].v
                    clear = not (x.gap is not None and x.gap < self.cfg.liveness_clear_gap)
                    cost += (self.cfg.liveness_weight if clear else 0.0) * \
                            max(0.0, self.cfg.v_min_live - v_end) ** 2
                    cands.append((cost, cand, worst))
            cands.sort(key=lambda t: t[0])
            u_safe = cands[0][1]
            slack_used = max(0.0, -cands[0][2])
            self.num_fallbacks += 1
        fallback = not feasible

        h_now = self.barriers(x)
        diff = float(np.max(np.abs(u_safe - u_rl)))
        intervened = diff > self.cfg.intervention_tol
        if intervened:
            self.num_interventions += 1
        step = SafetyStep(action=u_safe.astype(np.float32),
                          intervened=intervened,
                          min_barrier=float(np.min(h_now)),
                          barriers=h_now,
                          qp_success=feasible,
                          fallback=fallback)
        self.last_safety_step = step
        return step

    def stats(self) -> dict:
        return {
            "filter_calls": self.num_calls,
            "interventions": self.num_interventions,
            "intervention_rate": (self.num_interventions / self.num_calls) if self.num_calls else 0.0,
            "fallbacks": self.num_fallbacks,
        }
