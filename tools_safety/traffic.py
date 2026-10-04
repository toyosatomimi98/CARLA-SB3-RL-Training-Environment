"""Generate live traffic (CARLA Traffic Manager) so that the safety layer is
challenged by *moving* agents, not only by static props.

The Traffic Manager is put into synchronous mode to stay in step with the world
(otherwise the NPCs keep simulating between our ticks and behave erratically).
"""
from __future__ import annotations

import math
import random

import carla
import numpy as np


class Traffic:
    def __init__(self, env, n_vehicles: int = 12, n_walkers: int = 8,
                 seed: int = 0, min_distance_to_ego: float = 25.0,
                 global_leader_distance: float = 2.5):
        self.env = env
        self.actors: list = []
        self.tm = None
        self.n_vehicles = n_vehicles
        self.n_walkers = n_walkers
        self.rng = random.Random(seed)
        self.min_distance_to_ego = min_distance_to_ego
        self.global_leader_distance = global_leader_distance
        self.notes = ""

    # ------------------------------------------------------------------ spawn
    def spawn(self):
        env = self.env
        world = env.world
        try:
            self.tm = env.client.get_trafficmanager()
            self.tm.set_synchronous_mode(True)
            self.tm.set_global_distance_to_leader_percentage(
                max(0, int((self.global_leader_distance - 2.0) * 100)))
        except Exception as e:  # noqa: BLE001 - TM is optional
            print("traffic manager unavailable:", e)
            self.tm = None

        spawn_points = world.map.get_spawn_points()
        ego_loc = env.vehicle.get_transform().location
        vbp = world.get_blueprint_library().filter("vehicle.*")
        vbp = [b for b in vbp if b.id != "vehicle.lincoln.mkz"]
        placed = 0
        for sp in self.rng.sample(spawn_points, len(spawn_points)):
            if placed >= self.n_vehicles:
                break
            if sp.location.distance(ego_loc) < self.min_distance_to_ego:
                continue                      # keep the ego's immediate area clear
            bp = self.rng.choice(vbp)
            if bp.has_attribute("color"):
                bp.set_attribute("color", bp.get_attribute("color").recommended_values[0])
            if bp.has_attribute("role_name"):
                bp.set_attribute("role_name", "npc")
            try:
                v = world.spawn_actor(bp, sp)
            except RuntimeError:
                continue
            v.set_autopilot(True)
            if self.tm is not None:
                try:
                    self.tm.ignore_lights_percentage(v, 0)
                    self.tm.vehicle_percentage_speed_difference(v, self.rng.uniform(-10, 10))
                except Exception:  # noqa: BLE001
                    pass
            self.actors.append(v)
            placed += 1

        # pedestrians
        wbp = world.get_blueprint_library().filter("walker.pedestrian.*")
        cbp = world.get_blueprint_library().find("controller.ai.walker")
        walkers = 0
        for _ in range(self.n_walkers * 3):
            if walkers >= self.n_walkers:
                break
            loc = world.get_random_location_from_navigation()
            if loc is None or loc.distance(ego_loc) < self.min_distance_to_ego:
                continue
            try:
                w = world.spawn_actor(self.rng.choice(wbp), carla.Transform(loc))
            except RuntimeError:
                continue
            c = world.spawn_actor(cbp, carla.Transform(), attach_to=w)
            self.actors += [w, c]
            c.start()
            c.go_to_location(world.get_random_location_from_navigation())
            c.set_max_speed(1.4)
            walkers += 1

        self.notes = (f"live traffic: {placed} autopilot vehicles + {walkers} pedestrians "
                      f"(TM sync, {self.global_leader_distance:.1f} m to leader)")
        return self.actors

    # ------------------------------------------------------------------ utils
    def update(self, t: float):
        """Autopilot drives the NPCs; nothing to script per frame."""

    def nearest_vehicle_ahead(self) -> tuple[float, float] | None:
        """(gap, relative speed) to the nearest NPC vehicle in front (debug/HUD)."""
        env = self.env
        tf = env.vehicle.get_transform()
        fwd = tf.get_forward_vector()
        fwd = np.array([fwd.x, fwd.y, 0.0])
        fwd /= max(np.linalg.norm(fwd), 1e-6)
        pos = np.array([tf.location.x, tf.location.y, 0.0])
        ego_v = env.vehicle.get_speed() / 3.6
        best = None
        for a in self.actors:
            if not a.type_id.startswith("vehicle."):
                continue
            loc = a.get_transform().location
            d = np.array([loc.x, loc.y, 0.0]) - pos
            fd = float(np.dot(d, fwd))
            lat = float(abs(d[0] * -fwd[1] + d[1] * fwd[0]))
            if 0.0 < fd < 60.0 and lat < 1.9 + 0.3 * fd:
                if best is None or fd < best[0]:
                    best = (fd, 0.0)
        return best

    def destroy(self):
        for a in self.actors:
            try:
                if a.type_id.startswith("controller."):
                    a.stop()
                a.destroy()
            except Exception:  # noqa: BLE001
                pass
        self.actors = []
        if self.tm is not None:
            try:
                self.tm.set_synchronous_mode(False)
            except Exception:  # noqa: BLE001
                pass
