"""Interactive traffic scenarios that actually *challenge* the safety barriers.

The course's safety story only becomes meaningful when other agents are present,
so each scenario places something in the ego's path:

  roadblock   static prop across the ego lane (e.g. warningconstruction)
  lead_brake  a lead vehicle that creeps and then brakes hard   -> headway barrier
  crossing    a vehicle crossing the ego's path at a junction   -> headway barrier
  ped_cross   a pedestrian walking across the lane              -> headway barrier

All scenarios are deterministic (fixed spawn points taken from the ego's own
route, scripted controls), so they can be replayed with the same seed.
"""
from __future__ import annotations

import math

import carla
import numpy as np

from carla_env.wrappers import vector


def route_point_ahead(env, metres: float):
    """Return (location, waypoint) on the ego route `metres` ahead."""
    wps = env.route_waypoints
    i = int(env.current_waypoint_index)
    acc = 0.0
    prev = wps[i][0].transform.location
    for j in range(i, len(wps) - 1):
        loc = wps[j + 1][0].transform.location
        acc += loc.distance(prev)
        prev = loc
        if acc >= metres:
            return loc, wps[j + 1][0]
    return wps[-1][0].transform.location, wps[-1][0]


class Scenario:
    """Base class: spawn actors, then update them every frame."""

    name = "none"

    def __init__(self, env):
        self.env = env
        self.actors: list = []
        self.notes = ""

    def spawn(self):  # noqa: D401
        return self.actors

    def update(self, t: float):  # noqa: D401
        pass

    def destroy(self):
        for a in self.actors:
            try:
                a.destroy()
            except Exception:  # noqa: BLE001
                pass
        self.actors = []


class RoadblockScenario(Scenario):
    """A construction barrier placed in the middle of the ego's lane."""

    name = "roadblock"

    def __init__(self, env, ahead_m: float = 45.0):
        super().__init__(env)
        self.ahead_m = ahead_m

    def spawn(self):
        env = self.env
        loc, wp = route_point_ahead(env, self.ahead_m)
        bp = env.world.get_blueprint_library().find("static.prop.warningconstruction")
        tf = carla.Transform(loc + carla.Location(z=0.1),
                             carla.Rotation(yaw=wp.transform.rotation.yaw))
        self.actors.append(env.world.spawn_actor(bp, tf))
        self.notes = f"static roadblock {self.ahead_m:.0f} m ahead in the ego lane"
        return self.actors


class LeadBrakeScenario(Scenario):
    """A lead vehicle that creeps forward and then brakes hard."""

    name = "lead_brake"

    def __init__(self, env, ahead_m: float = 32.0, creep_kmh: float = 12.0,
                 brake_at: float = 6.0, brake_for: float = 2.5):
        super().__init__(env)
        self.ahead_m, self.creep_kmh = ahead_m, creep_kmh
        self.brake_at, self.brake_for = brake_at, brake_for
        self.lead = None
        self.braking = False

    def spawn(self):
        env = self.env
        loc, wp = route_point_ahead(env, self.ahead_m)
        bp = env.world.get_blueprint_library().find("vehicle.nissan.patrol")
        if bp.has_attribute("color"):
            bp.set_attribute("color", bp.get_attribute("color").recommended_values[0])
        self.lead = env.world.spawn_actor(
            bp, carla.Transform(loc + carla.Location(z=0.5),
                                carla.Rotation(yaw=wp.transform.rotation.yaw)))
        self.actors.append(self.lead)
        self.notes = (f"lead vehicle {self.ahead_m:.0f} m ahead, creeps at "
                      f"{self.creep_kmh:.0f} km/h then brakes at t={self.brake_at:.0f}s")
        return self.actors

    def update(self, t: float):
        if self.lead is None:
            return
        ctrl = carla.VehicleControl()
        if self.brake_at <= t < self.brake_at + self.brake_for:
            ctrl.brake = 1.0
            self.braking = True
        else:
            ctrl.throttle = self.creep_kmh / 60.0
        self.lead.apply_control(ctrl)


class CrossingScenario(Scenario):
    """A vehicle that drives across the ego's path at a junction."""

    name = "crossing"

    def __init__(self, env, ahead_m: float = 38.0, start_after: float = 2.0):
        super().__init__(env)
        self.ahead_m, self.start_after = ahead_m, start_after
        self.npc = None
        self.moving = False

    def spawn(self):
        env = self.env
        loc, wp = route_point_ahead(env, self.ahead_m)
        # put the crossing vehicle on a junction: search a nearby waypoint on a
        # different road_id, i.e. a genuinely crossing lane
        crossing_wp = None
        for cand in wp.next(12.0):
            if cand.road_id != wp.road_id:
                crossing_wp = cand
                break
        self.crossing_wp = crossing_wp or wp
        yaw = self.crossing_wp.transform.rotation.yaw
        bp = env.world.get_blueprint_library().find("vehicle.dodge.charger")
        if bp.has_attribute("color"):
            bp.set_attribute("color", bp.get_attribute("color").recommended_values[0])
        self.npc = env.world.spawn_actor(
            bp, carla.Transform(self.crossing_wp.transform.location + carla.Location(z=0.5),
                                carla.Rotation(yaw=yaw)))
        self.actors.append(self.npc)
        self.notes = (f"crossing vehicle on road {self.crossing_wp.road_id} "
                      f"(ego route road {wp.road_id}) about {self.ahead_m:.0f} m ahead")
        return self.actors

    def update(self, t: float):
        if self.npc is None:
            return
        ctrl = carla.VehicleControl()
        if t >= self.start_after:
            self.moving = True
        ctrl.throttle = 0.45 if self.moving else 0.0
        ctrl.brake = 0.0 if self.moving else 1.0
        self.npc.apply_control(ctrl)


class PedestrianCrossingScenario(Scenario):
    """A pedestrian walking across the ego lane."""

    name = "ped_cross"

    def __init__(self, env, ahead_m: float = 28.0, start_after: float = 2.0):
        super().__init__(env)
        self.ahead_m, self.start_after = ahead_m, start_after
        self.walker = None
        self.controller = None

    def spawn(self):
        env = self.env
        loc, wp = route_point_ahead(env, self.ahead_m)
        fwd = vector(wp.transform.rotation.get_forward_vector())
        right = carla.Location(x=-fwd[1], y=fwd[0], z=0.0)
        bp = env.world.get_blueprint_library().filter("walker.pedestrian.*")[0]
        # The spawn point can be occupied (it failed once with "Spawn failed
        # because of collision at spawn position"), so walk outwards a few times.
        self.walker = None
        for lateral in (4.0, 5.0, 6.0, 3.0):
            start = loc + right * lateral + carla.Location(z=0.3)
            try:
                self.walker = env.world.spawn_actor(bp, carla.Transform(start))
                break
            except RuntimeError:
                continue
        if self.walker is None:
            raise RuntimeError("could not spawn the pedestrian (all offsets occupied)")
        start = self.walker.get_transform().location
        cbp = env.world.get_blueprint_library().find("controller.ai.walker")
        self.controller = env.world.spawn_actor(cbp, carla.Transform(), attach_to=self.walker)
        self.actors += [self.walker, self.controller]
        self.target = loc + right * (-4.0)
        self.started = False
        self.notes = f"pedestrian crossing the lane {self.ahead_m:.0f} m ahead"
        return self.actors

    def update(self, t: float):
        if self.controller is None or self.started or t < self.start_after:
            return
        self.controller.start()
        self.controller.go_to_location(self.target)
        self.controller.set_max_speed(1.4)
        self.started = True


class CutInScenario(Scenario):
    """A vehicle from the adjacent lane cuts into the ego's lane and then slows
    down, forcing the ego to keep a safe headway (this is the interaction that
    actually makes the headway barrier fire).

    Timeline (defaults): 0-3 s the NPC cruises in the adjacent lane slightly
    slower than the ego (so the ego closes in); 3-6 s it steers into the ego's
    lane; after 6 s it keeps a low speed, so the ego must slow down behind it.
    """

    name = "cutin"

    def __init__(self, env, ahead_m: float = 22.0, start_at: float = 3.0,
                 cut_duration: float = 3.0, cut_steer: float = 0.32,
                 cruise_throttle: float = 0.22, slow_throttle: float = 0.16,
                 brake_check: bool = True, brake_duration: float = 3.0):
        super().__init__(env)
        self.ahead_m, self.start_at, self.cut_duration = ahead_m, start_at, cut_duration
        self.cut_steer, self.cruise_throttle, self.slow_throttle = \
            cut_steer, cruise_throttle, slow_throttle
        self.brake_check = brake_check
        self.brake_duration = brake_duration
        self.npc = None
        self.steer_sign = 1.0
        self.phase = "approach"

    def spawn(self):
        env = self.env
        # Search a few distances ahead and both sides: at some points the route
        # waypoint has no adjacent *driving* lane (road edge / junction), and the
        # first attempt failed with "no adjacent driving lane to cut in from".
        adj = None
        for ahead in (self.ahead_m, 26.0, 18.0, 30.0, 14.0, 34.0):
            _loc, wp = route_point_ahead(env, ahead)
            for getter, sign in ((wp.get_left_lane, 1.0), (wp.get_right_lane, -1.0)):
                cand = getter()
                if cand is not None and cand.lane_type == carla.LaneType.Driving:
                    adj, self.steer_sign, self.ahead_m = cand, sign, ahead
                    break
            if adj is not None:
                break
        if adj is None:
            # Fallback that always works: the NPC is scripted (not autopilot), so
            # it does not need a real adjacent lane on the map.  Place it one lane
            # width to the side of the ego lane and let it merge in.
            loc, wp = route_point_ahead(env, self.ahead_m)
            fwd = vector(wp.transform.rotation.get_forward_vector())
            right = carla.Location(x=-fwd[1], y=fwd[0], z=0.0)
            self.steer_sign = -1.0          # coming from the right -> steer left
            spawn_tf = carla.Transform(loc + right * 3.0 + carla.Location(z=0.5),
                                       carla.Rotation(yaw=wp.transform.rotation.yaw))
            self._spawn_at(env, spawn_tf)
            self.notes = (f"cut-in (synthetic): NPC spawns {self.ahead_m:.0f} m ahead, "
                          f"3.0 m to the side of the ego lane, and merges in at "
                          f"t={self.start_at:.0f}s, then crawls")
            return self.actors
        bp = env.world.get_blueprint_library().find("vehicle.dodge.charger")
        if bp.has_attribute("color"):
            bp.set_attribute("color", bp.get_attribute("color").recommended_values[0])
        self.npc = env.world.spawn_actor(
            bp, carla.Transform(adj.transform.location + carla.Location(z=0.5),
                                carla.Rotation(yaw=adj.transform.rotation.yaw)))
        self.actors.append(self.npc)
        self.notes = (f"cut-in: NPC spawns {self.ahead_m:.0f} m ahead in lane "
                      f"{adj.lane_id} and merges into the ego lane (steer "
                      f"{self.steer_sign:+.0f}) at t={self.start_at:.0f}s, then crawls")
        return self.actors

    def _spawn_at(self, env, transform):
        bp = env.world.get_blueprint_library().find("vehicle.dodge.charger")
        if bp.has_attribute("color"):
            bp.set_attribute("color", bp.get_attribute("color").recommended_values[0])
        self.npc = env.world.spawn_actor(bp, transform)
        self.actors.append(self.npc)
        return self.npc

    def update(self, t: float):
        if self.npc is None:
            return
        ctrl = carla.VehicleControl()
        if t < self.start_at:
            self.phase = "approach"          # stay in the adjacent lane, let the ego close in
            ctrl.throttle = self.cruise_throttle
        elif t < self.start_at + self.cut_duration:
            self.phase = "cutting in"        # steer across into the ego lane
            ctrl.steer = self.steer_sign * self.cut_steer
            ctrl.throttle = self.cruise_throttle
        else:
            t_block = t - (self.start_at + self.cut_duration)
            if self.brake_check and t_block < self.brake_duration:
                # BRAKE CHECK: the classic cut-in accident - the vehicle that
                # just merged in front of us slams the brakes.
                self.phase = "brake-check"
                ctrl.brake = 1.0
            else:
                self.phase = "blocking"      # stay slow in front of the ego
                ctrl.throttle = self.slow_throttle
        self.npc.apply_control(ctrl)


def build(name: str, env) -> Scenario:
    table = {
        "none": Scenario,
        "roadblock": RoadblockScenario,
        "lead_brake": LeadBrakeScenario,
        "crossing": CrossingScenario,
        "ped_cross": PedestrianCrossingScenario,
        "cutin": CutInScenario,
    }
    if name not in table:
        raise KeyError(f"unknown scenario {name!r}; choose from {sorted(table)}")
    return table[name](env)
