"""Shared reference lane-following controller (sign conventions verified).

Empirically established on this CARLA 0.10.0 build (see probe_steer_sign.py):

    steer > 0              -> the car turns RIGHT
    vehicle.get_angle(wp)  -> POSITIVE when the vehicle points to the RIGHT of
                              the lane (it is computed as
                              angle_diff(waypoint_forward, velocity))

Therefore the corrective steering must be **negative** feedback:

    steer = -k * get_angle()

The first version of the demo/benchmark controller used `steer = +k * ang`,
i.e. positive feedback: any small rightward error made the car steer further
right, so every "no safety logic" run drifted to the right and off the road.
That bug inflated the apparent benefit of the safety filter, because the filter
was fighting a driver that was actively steering off the lane.
"""
import numpy as np


def signed_lateral_offset(env) -> float:
    """Signed lateral offset from the current lane centre [m] (+ = to the right)."""
    wp = env.current_waypoint
    fwd = wp.transform.rotation.get_forward_vector()
    loc = env.vehicle.get_transform().location
    d = np.array([loc.x - wp.transform.location.x, loc.y - wp.transform.location.y, 0.0])
    return float(fwd.x * d[1] - fwd.y * d[0])


def lane_follow_action(env, ref_speed_kmh: float = 16.0, steer_gain: float = 1.0,
                       lat_gain: float = 2.0, disturbance: float = 0.0) -> np.ndarray:
    """Stanley-style lane keeping: heading error **and** lateral offset.

    A pure heading controller (only the first term) cannot remove a steady
    lateral offset: the car happily drives *parallel* to the lane while sitting
    3.5 m outside it.  Measured before adding the lateral term: the car settled
    at e_y = +3.5 m with ang = 0 and steer = 0, then kept drifting off.

    Signs (verified by tools_safety/probe_steer_sign.py):
        steer > 0 -> right,   get_angle() > 0 -> pointing right
        e_y > 0   -> right of the lane
    so the corrective steering is negative feedback on both terms.
    """
    ang = float(env.vehicle.get_angle(env.current_waypoint))
    v = max(float(env.vehicle.get_speed()) / 3.6, 0.5)      # m/s, avoid /0
    e_y = signed_lateral_offset(env)
    delta = -(steer_gain * ang + np.arctan(lat_gain * e_y / v))
    steer = float(np.clip(delta, -1.0, 1.0))
    speed = float(env.vehicle.get_speed())
    throttle = float(np.clip((ref_speed_kmh - speed) / 14.0, 0.0, 1.0))
    steer += disturbance * 0.75
    throttle += disturbance * 0.35
    return np.array([float(np.clip(steer, -1.0, 1.0)),
                     float(np.clip(throttle, 0.0, 1.0))], dtype=np.float32)
