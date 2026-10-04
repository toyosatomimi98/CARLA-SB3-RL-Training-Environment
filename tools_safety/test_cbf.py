"""Unit tests for the CBF safety filter (no simulator required)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from carla_env.safety.cbf import CBFFilterConfig, CBFSafetyFilter, EgoState  # noqa: E402


def check(name, cond, extra=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name} {extra}")
    return bool(cond)


def main() -> None:
    cfg = CBFFilterConfig(dt=1 / 15, lane_half_width=1.75, speed_limit=13.9, a_lat_max=2.5)
    f = CBFSafetyFilter(cfg)
    ok = True

    print("== 1. barrier values ==")
    safe_state = EgoState(e_y=0.2, e_psi=0.0, v=8.0, kappa=0.0)
    h = f.barriers(safe_state)
    ok &= check("centred, below limit -> all h > 0", np.all(h > 0), f"h={np.round(h, 3)}")
    bad_state = EgoState(e_y=3.0, e_psi=0.0, v=25.0, kappa=0.0)
    h = f.barriers(bad_state)
    ok &= check("off lane + over speed -> h < 0", np.any(h < 0), f"h={np.round(h, 3)}")

    print("== 2. filter passes through a safe action ==")
    u = np.array([0.05, 0.3])
    step = f.solve(safe_state, u)
    ok &= check("safe action unchanged", not step.intervened, f"u*={np.round(step.action, 4)}")

    print("== 3. filter rejects an unsafe action (too fast) ==")
    fast = EgoState(e_y=0.0, e_psi=0.0, v=13.5, kappa=0.0)  # just under the limit
    step = f.solve(fast, np.array([0.0, 1.0]))              # full throttle would exceed it
    h_next = f.barriers(f._predict(fast, step.action))
    ok &= check("speed barrier respected after filtering",
                h_next[2] >= -1e-6, f"h_speed_next={h_next[2]:.4f}, action={np.round(step.action, 3)}")
    ok &= check("intervention logged", step.intervened)

    print("== 4. filter keeps the car inside the corridor ==")
    # NOTE: the effective corridor is lane_half_width - lane_margin = 1.45 m, so
    # the state must start *inside* it for the hard "h >= 0" constraint to be
    # satisfiable at all.
    drifting = EgoState(e_y=1.20, e_psi=0.10, v=10.0, kappa=0.0)
    step = f.solve(drifting, np.array([0.9, 0.8]))  # policy wants to steer further left
    h_next = f.barriers(f._predict(drifting, step.action))
    ok &= check("lane barrier respected after filtering",
                np.all(h_next[:2] >= -1e-6), f"h_lane_next={np.round(h_next[:2], 4)}")

    print("== 4b. already outside the corridor -> never worse than the raw action ==")
    outside = EgoState(e_y=1.60, e_psi=0.10, v=10.0, kappa=0.0)
    raw = np.array([0.9, 0.8])

    def worst_barrier(x, u):
        return float(np.min([np.min(f.barriers(s)) for s in f.rollout(x, u)]))

    step = f.solve(outside, raw)
    # A state that is already outside the safe set cannot be repaired in one
    # step (the heading error dominates), so the meaningful guarantee is that
    # the filter's choice is never *worse* than the raw policy action.
    ok &= check("fallback is at least as safe as the raw action",
                worst_barrier(outside, step.action) >= worst_barrier(outside, raw) - 1e-6,
                f"filter={worst_barrier(outside, step.action):+.4f} "
                f"raw={worst_barrier(outside, raw):+.4f} action={np.round(step.action, 3)}")

    print("== 5. curve speed limit is monotone in curvature ==")
    v_straight = f.curve_speed_limit(0.0)
    v_curve = f.curve_speed_limit(0.05)
    ok &= check("tighter curve -> lower speed", v_curve < v_straight,
                f"straight={v_straight:.2f} m/s, curve={v_curve:.2f} m/s")

    print("== 6. filter is a projection: identity for a safe action ==")
    step = f.solve(EgoState(0.0, 0.0, 5.0, 0.0), np.array([0.2, 0.5]))
    ok &= check("||u* - u_rl|| ~ 0", np.max(np.abs(step.action - np.array([0.2, 0.5]))) < 1e-6)

    print("\noverall:", "PASS" if ok else "FAIL")
    print("filter stats:", f.stats())
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
