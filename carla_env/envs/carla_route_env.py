import os
import subprocess
import sys
import glob
import time
import math
import gymnasium as gym  # ported from gym 0.22 -> gymnasium (SB3 >= 2.4)
import pygame
import cv2
from pygame.locals import *

from carla_env.tools.hud import HUD
from carla_env.navigation.planner import RoadOption, compute_route_waypoints
from carla_env.wrappers import *

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
import carla
from collections import deque
import itertools

# NOTE (port): the original hard-coded spawn-point index pairs are Town02
# specific. The locally available CARLA 0.10.0 map is Town10HD_Opt, so routes
# are now sampled from the current map at run time instead.
eval_routes = itertools.cycle([(48, 21), (0, 72), (28, 83), (61, 39)])

discrete_actions = {
    0: [-1, 1], 1: [0, 1], 2: [1, 1], 3: [0, 0],
}


class CarlaRouteEnv(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array", "rgb_array_no_hud", "state_pixels"]
    }

    def __init__(self, host="127.0.0.1", port=2000,
                 viewer_res=(1120, 560), obs_res=(160, 80),
                 reward_fn=None,
                 observation_space=None,
                 encode_state_fn=None, decode_vae_fn=None,
                 fps=15, action_smoothing=0.0, action_space_type="continuous",
                 activate_spectator=True,
                 activate_lidar=False,
                 start_carla=True,
                 eval=False,
                 activate_render=True,
                 town="Town10HD_Opt",
                 carla_exe=None,
                 route_rng_seed=None,
                 throttle_bias=0.0,
                 weather=None,
                 allow_brake=False,
                 no_rendering=False,
                 use_camera_obs=None):
        """
        A gym-like environment for interacting with a running CARLA environment and controlling a Lincoln MKZ2017 vehicle.

        Parameters:
            - host (str): IP address of the CARLA host
            - port (int): Port used to connect to CARLA
            - viewer_res (tuple[int, int]): Resolution of the spectator camera as a (width, height) tuple
            - obs_res (tuple[int, int]): Resolution of the observation camera as a (width, height) tuple
            - reward_fn (function): Custom reward function that is called every step. If None, no reward function is used.
            - observation_space: Custom observation space. If None, the default observation space is used.
            - encode_state_fn (function): Function that encodes the image from the observation camera to a state vector returned by step(). If None, the full image is returned.
            - decode_vae_fn (function): Function that decodes a state vector to an image. Used only if encode_state_fn is not None.
            - fps (int): FPS of the client. If fps <= 0 then use unbounded FPS.
            - action_smoothing (float): Scalar used to smooth the incoming action signal. 1.0 = max smoothing, 0.0 = no smoothing
            - action_space_type (str): Type of action space. Can be "continuous" or "discrete".
            - activate_spectator (bool): Whether to activate the spectator camera. Default is True.
            - activate_lidar (bool): Whether to activate the lidar sensor. Default is False.
            - start_carla (bool): Whether to automatically start CARLA when True. Note that you need to set the environment variable ${CARLA_ROOT} to point to the CARLA root directory for this option to work.
            - eval (bool): Whether the environment is used for evaluation or training. Default is False.
            - activate_render (bool): Whether to activate rendering. Default is True.
        """

        self.town = town
        self.route_rng = np.random.RandomState(route_rng_seed)
        self.spawn_grace_time = 1.5
        """Seconds after a (re)spawn during which collision events are ignored.
        Town10HD_Opt has parked vehicles standing on/near some spawn points, so
        the ego can start *overlapping* one: the collision sensor then fires
        ~1 s into the episode and the run is scored as a crash even though the
        policy did nothing wrong.  Ignored events are printed so this stays
        visible instead of silently changing the semantics."""
        self.episode_start_time = time.time()
        # PORT: with a plain `throttle in [0, 1]` action the agent has to discover
        # that it must give *sustained* throttle to move at all (a single step of
        # full throttle only adds ~0.2 km/h), so PPO at a small budget collapses
        # to throttle ~= 0 and the car never moves.  A bias keeps the car rolling:
        #   applied_throttle = bias + (1 - bias) * action_throttle
        self.throttle_bias = float(throttle_bias)
        self.allow_brake = bool(allow_brake)
        self.carla_process = None
        if start_carla:
            # PORT (Windows / CARLA 0.10.0): the upstream code only knew how to
            # launch the Linux `CarlaUE4.sh` helper.  On Windows the shipped
            # server is `CarlaUnreal.exe`, so accept an explicit executable and
            # fall back to CARLA_ROOT/CarlaUE4.sh for Linux users.
            if carla_exe is None:
                if os.name == "nt" and os.environ.get("CARLA_ROOT"):
                    candidate = os.path.join(
                        os.environ["CARLA_ROOT"],
                        "Carla-0.10.0-Win64-Shipping", "CarlaUnreal.exe")
                    if os.path.exists(candidate):
                        carla_exe = candidate
            if carla_exe is None:
                if "CARLA_ROOT" not in os.environ:
                    raise Exception("${CARLA_ROOT} has not been set!")
                carla_path = os.path.join(os.environ["CARLA_ROOT"], "CarlaUE4.sh")
                launch_command = [carla_path]
            else:
                launch_command = [carla_exe]
            launch_command += ['-quality_level=Low']
            launch_command += ['-benchmark']
            launch_command += ["-fps=%i" % fps]
            launch_command += ['-RenderOffScreen']
            launch_command += ['-prefernvidia']
            launch_command += ['-nosound']
            launch_command += [f'-carla-world-port={port}']
            print("Running command:")
            print(" ".join(launch_command))
            self.carla_process = subprocess.Popen(launch_command, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
            print("Waiting for CARLA to initialize\n")

            # ./CarlaUE4.sh -quality_level=Low -benchmark -fps=15 -RenderOffScreen
            time.sleep(5)

        width, height = viewer_res
        if obs_res is None:
            out_width, out_height = width, height
        else:
            out_width, out_height = obs_res
        self.activate_render = activate_render


        # Setup gym environment
        self.action_space_type = action_space_type
        if self.action_space_type == "continuous":
            # steer, throttle; when `allow_brake` the throttle channel is
            # [-1, 1] where negative values mean braking.  WITHOUT a brake
            # channel a safety filter can only cut the throttle (i.e. coast),
            # which is not enough to honour a headway constraint.
            low = np.array([-1, -1 if self.allow_brake else 0])
            self.action_space = gym.spaces.Box(low, np.array([1, 1]), dtype=np.float32)
        elif self.action_space_type == "discrete":
            self.action_space = gym.spaces.Discrete(len(discrete_actions))

        self.observation_space = observation_space
        # PORT (2026-10-04): the dashboard camera is only needed when the
        # observation actually contains an image.  With the vector-state
        # configs (`states["1"]`) the frame was fetched and thrown away every
        # single step, which also forced the server to keep rendering
        # (no_rendering_mode=False) - the main reason training ran at ~26
        # steps/s instead of a few hundred.
        _auto_camera = bool(observation_space is not None and any(
            key in observation_space.keys() for key in ("rgb_camera", "seg_camera")))
        # The demo recorder also wants the dashboard frame for its "driver view"
        # panel even though the policy only consumes the vector state, so it can
        # pass `use_camera_obs=True` explicitly.
        self.use_camera_obs = _auto_camera if use_camera_obs is None else bool(use_camera_obs)
        self.no_rendering = bool(no_rendering)

        self.fps = fps
        self.action_smoothing = action_smoothing
        self.episode_idx = -2

        self.encode_state_fn = (lambda x: x) if not callable(encode_state_fn) else encode_state_fn
        self.decode_vae_fn = None if not callable(decode_vae_fn) else decode_vae_fn
        self.reward_fn = (lambda x: 0) if not callable(reward_fn) else reward_fn
        self.max_distance = 3000  # m
        self.activate_spectator = activate_spectator
        self.activate_lidar = activate_lidar
        self.eval = eval

        self.world = None
        try:
            # Connect to carla
            self.client = carla.Client(host, port)
            self.client.set_timeout(60.0)

            # Create world wrapper
            self.world = World(self.client, town=self.town)

            settings = self.world.get_settings()
            settings.fixed_delta_seconds = 1 / self.fps
            settings.synchronous_mode = True
            # Training with vector observations does not need rendered frames.
            settings.no_rendering_mode = self.no_rendering
            self.world.apply_settings(settings)
            self.client.reload_world(False)  # reload map keeping the world settings
            # Optional deterministic lighting: pass a preset name (e.g.
            # "ClearNoon") to pin the weather, so that runs recorded at
            # different times look the same.  Default: keep the server weather.
            if weather:
                preset = getattr(carla.WeatherParameters, str(weather), None)
                if preset is not None:
                    self.world.set_weather(preset)
                    print(f"Pinned weather to {weather}")


            # Create vehicle and attach camera to it
            self.vehicle = Vehicle(self.world, self.world.map.get_spawn_points()[0],
                                   on_collision_fn=lambda e: self._on_collision(e),
                                   on_invasion_fn=lambda e: self._on_invasion(e))

            # Create hud and initialize pygame for visualization
            if self.activate_render:
                pygame.init()
                pygame.font.init()
                self.display = pygame.display.set_mode((width, height), pygame.HWSURFACE | pygame.DOUBLEBUF)
                self.clock = pygame.time.Clock()
                self.hud = HUD(width, height)
                self.hud.set_vehicle(self.vehicle)
                self.world.on_tick(self.hud.on_world_tick)

            seg_settings = {}
            if "seg_camera" in self.observation_space.keys():
                seg_settings.update({
                    'camera_type': "sensor.camera.semantic_segmentation",
                    'custom_palette': True
                })
            if self.use_camera_obs:
                self.dashcam = Camera(self.world, out_width, out_height,
                                      transform=sensor_transforms["dashboard"],
                                      attach_to=self.vehicle,
                                      on_recv_image=lambda e: self._set_observation_image(e),
                                      **seg_settings)
            else:
                self.dashcam = None

            if self.activate_spectator:
                self.camera = Camera(self.world, width, height,
                                     transform=sensor_transforms["spectator"],
                                     attach_to=self.vehicle, on_recv_image=lambda e: self._set_viewer_image(e))
            if self.activate_lidar:
                self.lidar = Lidar(self.world, transform=sensor_transforms["lidar"],
                                   attach_to=self.vehicle, on_recv_image=lambda e: self._set_lidar_data(e))
        except Exception as e:
            self.close()
            raise e
        # Reset env to set initial state
        self.reset()

    def reset(self, *, seed=None, options=None):
        """Gymnasium reset: returns (obs, info)."""
        self.episode_start_time = time.time()
        if seed is not None:
            self.route_rng = np.random.RandomState(seed)
            self._np_seed = seed
        # Create new route
        self.num_routes_completed = -1
        self.episode_idx += 1
        self.new_route()

        # Two different variables to differ between success episode and fail episode
        self.terminal_state = False  # Set to True when we want to end episode
        self.success_state = False  # Set to True when we want to end episode.
        self.collision_flag = False  # PORT: distinguishes real collisions from
        # other terminal conditions (off-track / too fast / vehicle stopped)

        self.closed = False  # Set to True when ESC is pressed
        self.extra_info = []  # List of extra info shown on the HUD
        self.observation = self.observation_buffer = None  # Last received observation
        self.viewer_image = self.viewer_image_buffer = None  # Last received image to show in the viewer
        self.lidar_data = self.lidar_data_buffer = None
        self.step_count = 0

        # Init metrics
        self.total_reward = 0.0
        self.previous_location = self.vehicle.get_transform().location
        self.distance_traveled = 0.0
        self.center_lane_deviation = 0.0
        self.speed_accum = 0.0
        self.routes_completed = 0.0
        self.world.tick()
        # Return initial observation
        time.sleep(0.2)
        obs = self._step_impl(None)[0]
        time.sleep(0.2)
        return obs, {}

    def new_route(self):
        # Do a soft reset (teleport vehicle)
        self.vehicle.control.steer = float(0.0)
        self.vehicle.control.throttle = float(0.0)
        # PORT FIX (CARLA 0.10): the upstream code disabled physics, teleported
        # and re-enabled physics.  On the locally shipped 0.10.0 build the
        # vehicle is *permanently inert* after that sequence (measured: max
        # speed 0.5 km/h with 90 % throttle for 40 steps), which silently froze
        # every training run after the first reset.  Instead we keep physics on,
        # zero the velocities and teleport.
        try:
            self.vehicle.actor.set_target_velocity(carla.Vector3D(0.0, 0.0, 0.0))
            self.vehicle.actor.set_target_angular_velocity(carla.Vector3D(0.0, 0.0, 0.0))
        except Exception:  # noqa: BLE001 - API not available on every build
            pass

        # Generate waypoints along the lap
        spawn_points = self.world.map.get_spawn_points()
        if not self.eval:
            spawn_points_list = self._sample_spawn_pair(spawn_points)
        else:
            idx_pair = next(eval_routes)
            if max(idx_pair) >= len(spawn_points):
                spawn_points_list = self._sample_spawn_pair(spawn_points)
            else:
                spawn_points_list = [spawn_points[index] for index in idx_pair]
        route_length = 1
        attempts = 0
        while True:
            attempts += 1
            self.start_wp, self.end_wp = [self.world.map.get_waypoint(spawn.location) for spawn in
                                          spawn_points_list]
            route = compute_route_waypoints(self.world.map, self.start_wp, self.end_wp, resolution=1.0)
            route, anchor = self._sanitize_route(route)
            route_length = len(route)
            # Accept only a route that is one continuous lane chain: a
            # discontinuity (see _sanitize_route / _route_has_discontinuity)
            # would make the car measure a phantom lateral offset for the whole
            # episode.  Bounded so a pathological map cannot spin for ever.
            if route_length > 1 and (self.eval or attempts >= 25
                                     or not self._route_has_discontinuity(route)):
                self.route_waypoints = route
                self.start_wp = anchor
                break
            spawn_points_list = self._sample_spawn_pair(spawn_points)

        self.distance_from_center_history = deque(maxlen=30)

        self.current_waypoint_index = 0
        self.num_routes_completed += 1
        self.vehicle.set_transform(self.start_wp.transform)
        time.sleep(0.2)

    def _sample_spawn_pair(self, spawn_points):
        """Pick two distinct spawn points (by index, numpy-2 safe)."""
        idx = self.route_rng.choice(len(spawn_points), 2, replace=False)
        return [spawn_points[int(i)] for i in idx]

    @staticmethod
    def _sanitize_route(route, max_lateral_lead_in=1.0):
        """Remove the planner's lateral lead-in so the route starts in *its own* lane.

        PORT FIX (2026-10-04).  Measured with ``tmp/hw3_route_probe.py``: on
        some spawn pairs ``compute_route_waypoints`` returns a first leg that
        is a diagonal *connector* into a neighbouring lane.  Seed 33, for
        example, gives

            wp0 = (-15.447, -68.156)   <- the car is teleported here
            wp1 = (-20.302, -64.706)   <- 4.86 m ahead AND 3.45 m to the side
            wp2 = (-21.302, -64.717)   <- now running parallel, 1 m steps

        so the route's own lane centre is a whole lane (3.5 m) away from the
        point the car is placed at.  Both ``distance_from_center`` (the
        off-track test, and the HUD's ``e_y``) and the reference controller in
        ``tools_safety/driving_control.py`` measure the offset against the
        **line through (current, next) waypoint**, so from the second index on
        the car is reported as being 3.5 m outside its lane even though it is
        sitting in the middle of a lane.  In the demo clips this showed up as
        ``e_y`` jumping ``-0.005 -> +3.494 m`` in a single 1/15 s step at
        t = 1.73 s, ``min_i h_i`` dropping ``+1.44 -> -2.04``, and the CBF
        "overriding" a policy that had done nothing wrong.

        The route geometry and the RNG stream are left untouched: we simply
        drop the leading connector, re-anchor the car to the first waypoint of
        the route's real lane chain, and (as a hardening) drop any consecutive
        duplicate waypoints, which produce a degenerate (current, next) line.
        """
        # 1) drop the lateral lead-in
        while len(route) >= 3:
            w0 = route[0][0].transform
            w1 = route[1][0].transform
            f = w0.rotation.get_forward_vector()
            right = (-f.y, f.x)                       # right vector of wp0
            dx = w1.location.x - w0.location.x
            dy = w1.location.y - w0.location.y
            lateral = dx * right[0] + dy * right[1]
            if abs(lateral) <= max_lateral_lead_in:
                break
            route = route[1:]
        # 2) drop consecutive duplicates (degenerate lines)
        out = []
        for wp in route:
            if out:
                a = out[-1][0].transform.location
                b = wp[0].transform.location
                if abs(a.x - b.x) < 1e-3 and abs(a.y - b.y) < 1e-3:
                    continue
            out.append(wp)
        return out, (out[0][0] if out else route[0][0])

    @staticmethod
    def _route_has_discontinuity(route, max_leg=2.5):
        """True if the route contains a leg much longer than its 1 m resolution.

        PORT FIX (2026-10-04).  ``compute_route_waypoints(..., resolution=1.0)``
        emits waypoints ~1 m apart (measured over the first 200 legs: seed 11
        max 1.67 m, seed 5 max 1.31 m).  On some spawn pairs the planner instead
        jumps sideways to a neighbouring lane and leaves a 5.96 - 6.95 m leg
        behind (measured: seeds 7, 33).  The car is teleported to the start
        waypoint, so from that jump onwards it is a full lane width away from
        the line it is being steered along - i.e. a permanent phantom ``e_y``.
        Such a route is rejected and a new spawn pair is drawn.
        """
        locs = [w[0].transform.location for w in route]
        for a, b in zip(locs[:-1], locs[1:]):
            if math.hypot(b.x - a.x, b.y - a.y) > max_leg:
                return True
        return False

    def close(self):
        if self.carla_process:
            self.carla_process.terminate()
        pygame.quit()
        if self.world is not None:
            self.world.destroy()
            # PORT (2026-10): the upstream code left the server in synchronous
            # mode after closing.  Any later client (or the human opening the
            # CARLA window) then sees a frozen simulator, because nobody is
            # sending `world.tick()` - this is exactly the "get_map() raises
            # UnicodeDecodeError / simulation appears wedged" symptom we hit.
            # Restore an asynchronous, variable-time-step world on close.
            try:
                settings = self.world.get_settings()
                settings.synchronous_mode = False
                settings.fixed_delta_seconds = 0.0
                self.world.apply_settings(settings)
                self.world.tick()  # let the server run freely again
                print("Restored the CARLA world to asynchronous mode")
            except Exception as e:  # noqa: BLE001
                print("Could not restore world settings:", e)
        self.closed = True

    def render(self, mode="human"):
        if mode == "rgb_array_no_hud":
            return self.viewer_image
        elif mode == "rgb_array":
            # Turn display surface into rgb_array
            return np.array(pygame.surfarray.array3d(self.display), dtype=np.uint8).transpose([1, 0, 2])
        elif mode == "state_pixels":
            return self.observation

        # Tick render clock
        self.clock.tick()
        self.hud.tick(self.world, self.clock)

        # Get maneuver name
        if self.current_road_maneuver == RoadOption.LANEFOLLOW:
            maneuver = "Follow Lane"
        elif self.current_road_maneuver == RoadOption.LEFT:
            maneuver = "Left"
        elif self.current_road_maneuver == RoadOption.RIGHT:
            maneuver = "Right"
        elif self.current_road_maneuver == RoadOption.STRAIGHT:
            maneuver = "Straight"
        else:
            maneuver = "INVALID"

        # Add metrics to HUD
        self.extra_info.extend([
            "Episode {}".format(self.episode_idx),
            "Reward: % 19.2f" % self.last_reward,
            "",
            "Maneuver:        % 11s" % maneuver,
            "Routes completed:    % 7.2f" % self.routes_completed,
            "Distance traveled: % 7d m" % self.distance_traveled,
            "Center deviance:   % 7.2f m" % self.distance_from_center,
            "Avg center dev:    % 7.2f m" % (self.center_lane_deviation / self.step_count),
            "Avg speed:      % 7.2f km/h" % (self.speed_accum / self.step_count),
            "Total reward:        % 7.2f" % self.total_reward,
        ])
        if self.activate_spectator:
            # Blit image from spectator camera
            self.viewer_image = self._draw_path(self.camera, self.viewer_image)
            self.display.blit(pygame.surfarray.make_surface(self.viewer_image.swapaxes(0, 1)), (0, 0))
            # Superimpose current observation into top-right corner
        if self.observation is None:
            # Vector-state config: there is no dashboard frame to blit.
            return
        obs_h, obs_w = self.observation.shape[:2]
        pos_observation = (self.display.get_size()[0] - obs_w - 10, 10)
        self.display.blit(pygame.surfarray.make_surface(self.observation.swapaxes(0, 1)), pos_observation)

        pos_vae_decoded = (self.display.get_size()[0] - 2 * obs_w - 10, 10)
        if self.decode_vae_fn:
            self.display.blit(pygame.surfarray.make_surface(self.observation_decoded.swapaxes(0, 1)), pos_vae_decoded)

        if self.activate_lidar:
            lidar_h, lidar_w = self.lidar_data.shape[:2]
            pos_lidar = (self.display.get_size()[0] - obs_w - 10, 100)
            self.display.blit(pygame.surfarray.make_surface(self.lidar_data.swapaxes(0, 1)), pos_lidar)

        # Render HUD
        self.hud.render(self.display, extra_info=self.extra_info)
        self.extra_info = []  # Reset extra info list

        # Render to screen
        pygame.display.flip()



    def step(self, action):
        """Gymnasium step: returns (obs, reward, terminated, truncated, info)."""
        obs, reward, terminated, truncated, info = self._step_impl(action)
        return obs, reward, terminated, truncated, info

    def _step_impl(self, action):
        if self.closed:
            raise Exception("CarlaEnv.step() called after the environment was closed." +
                            "Check for info[\"closed\"] == True in the learning loop.")
        # Take action
        if action is not None:
            # Create new route on route completion
            if self.current_waypoint_index >= len(self.route_waypoints) - 1:
                if not self.eval:
                    self.new_route()
                else:
                    self.success_state = True

            if self.action_space_type == "continuous":
                steer, throttle = [float(a) for a in action]
                if self.throttle_bias > 0.0:
                    throttle = self.throttle_bias + (1.0 - self.throttle_bias) * throttle
                if self.allow_brake and throttle < 0.0:
                    self.vehicle.control.brake = float(min(-throttle, 1.0))
                    throttle = 0.0
                else:
                    self.vehicle.control.brake = 0.0
            elif self.action_space_type == "discrete":
                steer, throttle = discrete_actions[action]

            self.vehicle.control.steer = smooth_action(self.vehicle.control.steer, steer, self.action_smoothing)
            self.vehicle.control.throttle = smooth_action(self.vehicle.control.throttle, throttle,
                                                          self.action_smoothing)
        # Tick game
        self.world.tick()

        # Get most recent observation and viewer image
        if self.use_camera_obs:
            self.observation = self._get_observation()
        if self.activate_spectator:
            self.viewer_image = self._get_viewer_image()

        if self.activate_lidar:
            self.lidar_data = self._get_lidar_data()

        # Get vehicle transform
        transform = self.vehicle.get_transform()

        # Keep track of the closest waypoint on the route.
        #
        # PORT FIX (2026-10-05): upstream advanced the index with a
        # "is the car past the waypoint's forward plane" test
        # (dot(waypoint_forward, car - waypoint) > 0).  That test is fragile:
        # as soon as the car has a lateral offset (or the route bends) the dot
        # product goes negative again, the index *sticks*, `current_waypoint`
        # becomes stale, and `distance_from_center` (measured against the line
        # through the stale waypoint) explodes -> the reward function then
        # declares "Off-track" even though the car is centred in its lane.
        # Measured: an episode ended after 1.7 s with "Off-track" while the
        # telemetry showed e_y = 0.00 m.
        #
        # The replacement is a monotone nearest-point search over the next few
        # waypoints, which is what the CARLA agents' map projection does.
        self.prev_waypoint_index = self.current_waypoint_index
        car_xy = np.array([transform.location.x, transform.location.y])
        best_index = self.current_waypoint_index
        best_dist = None
        search_end = min(self.current_waypoint_index + 15, len(self.route_waypoints))
        for j in range(self.current_waypoint_index, search_end):
            wp_loc = self.route_waypoints[j][0].transform.location
            d = float(np.hypot(wp_loc.x - car_xy[0], wp_loc.y - car_xy[1]))
            if best_dist is None or d < best_dist:
                best_dist, best_index = d, j
        self.current_waypoint_index = best_index

        # Check for route completion
        if self.current_waypoint_index < len(self.route_waypoints) - 1:
            self.next_waypoint, self.next_road_maneuver = self.route_waypoints[
                (self.current_waypoint_index + 1) % len(self.route_waypoints)]

        self.current_waypoint, self.current_road_maneuver = self.route_waypoints[
            self.current_waypoint_index % len(self.route_waypoints)]
        self.routes_completed = self.num_routes_completed + (self.current_waypoint_index + 1) / len(
            self.route_waypoints)

        # Calculate deviation from center of the lane
        self.distance_from_center = distance_to_line(vector(self.current_waypoint.transform.location),
                                                     vector(self.next_waypoint.transform.location),
                                                     vector(transform.location))
        self.center_lane_deviation += self.distance_from_center

        # Calculate distance traveled
        if action is not None:
            self.distance_traveled += self.previous_location.distance(transform.location)
        self.previous_location = transform.location

        # Accumulate speed
        self.speed_accum += self.vehicle.get_speed()

        # Terminal on max distance
        if self.distance_traveled >= self.max_distance and not self.eval:
            self.success_state = True

        self.distance_from_center_history.append(self.distance_from_center)

        # Call external reward fn
        self.last_reward = self.reward_fn(self)
        self.total_reward += self.last_reward

        # Encode the state
        encoded_state = self.encode_state_fn(self)
        if self.decode_vae_fn:
            self.observation_decoded = self.decode_vae_fn(encoded_state['vae_latent'])
        self.step_count += 1

        # DEBUG: Draw path
        # self._draw_path_server(life_time=1.0, skip=8)
        # DEBUG: Draw current waypoint
        # self.world.debug.draw_point(self.current_waypoint.transform.location + carla.Location(z=1.25), size=0.1,color=carla.Color(0, 255, 255), life_time=2.0, persistent_lines=False)

        # Check for ESC press
        if self.activate_render:
            pygame.event.pump()
            if pygame.key.get_pressed()[K_ESCAPE]:
                self.close()
                self.terminal_state = True
            self.render()

        info = {
            "closed": self.closed,
            'total_reward': self.total_reward,
            'routes_completed': self.routes_completed,
            'total_distance': self.distance_traveled,
            'avg_center_dev': (self.center_lane_deviation / self.step_count),
            'avg_speed': (self.speed_accum / self.step_count),
            'mean_reward': (self.total_reward / self.step_count)
        }
        terminated = bool(self.terminal_state)
        truncated = bool(self.success_state)
        return encoded_state, self.last_reward, terminated, truncated, info

    def _draw_path_server(self, life_time=60.0, skip=0):
        """
            Draw a connected path from start of route to end.
            Green node = start
            Red node   = point along path
            Blue node  = destination
        """
        for i in range(0, len(self.route_waypoints) - 1, skip + 1):
            z = 30.25
            w0 = self.route_waypoints[i][0]
            w1 = self.route_waypoints[i + 1][0]
            self.world.debug.draw_line(
                w0.transform.location + carla.Location(z=z),
                w1.transform.location + carla.Location(z=z),
                thickness=0.1, color=carla.Color(255, 0, 0),
                life_time=life_time, persistent_lines=False)
            self.world.debug.draw_point(
                w0.transform.location + carla.Location(z=z), 0.1,
                carla.Color(0, 255, 0) if i == 0 else carla.Color(255, 0, 0),
                life_time, False)
        self.world.debug.draw_point(
            self.route_waypoints[-1][0].transform.location + carla.Location(z=z), 0.1,
            carla.Color(0, 0, 255),
            life_time, False)

    def _draw_path(self, camera, image):
        """
            Draw a connected path from start of route to end using homography.
        """
        vehicle_vector = vector(self.vehicle.get_transform().location)
        # Get the world to camera matrix
        world_2_camera = np.array(camera.get_transform().get_inverse_matrix())

        # Get the attributes from the camera
        image_w = int(camera.actor.attributes['image_size_x'])
        image_h = int(camera.actor.attributes['image_size_y'])
        fov = float(camera.actor.attributes['fov'])
        for i in range(self.current_waypoint_index, len(self.route_waypoints)):
            waypoint_location = self.route_waypoints[i][0].transform.location + carla.Location(z=1.25)
            waypoint_vector = vector(waypoint_location)
            if not (2 < abs(np.linalg.norm(vehicle_vector - waypoint_vector)) < 50):
                continue
            # Calculate the camera projection matrix to project from 3D -> 2D
            K = build_projection_matrix(image_w, image_h, fov)
            x, y = get_image_point(waypoint_location, K, world_2_camera)
            if i == len(self.route_waypoints) - 1:
                color = (255, 0, 0)
            else:
                color = (0, 0, 255)
            image = cv2.circle(image, (x, y), radius=3, color=color, thickness=-1)
        return image

    def _get_observation(self):
        while self.observation_buffer is None:
            pass
        obs = self.observation_buffer.copy()
        self.observation_buffer = None
        return obs

    def _get_viewer_image(self):
        while self.viewer_image_buffer is None:
            pass
        image = self.viewer_image_buffer.copy()
        self.viewer_image_buffer = None
        return image

    def _get_lidar_data(self):
        while self.lidar_data_buffer is None:
            pass
        image = self.lidar_data_buffer.copy()
        self.lidar_data_buffer = None
        return image

    def _on_collision(self, event):
        during_spawn = (time.time() - self.episode_start_time) < self.spawn_grace_time
        print(f"[collision] with {event.other_actor.type_id} "
              f"({get_actor_display_name(event.other_actor)}) "
              f"{time.time() - self.episode_start_time:.2f}s after spawn"
              f"{' - ignored (spawn grace)' if during_spawn else ''}")
        if during_spawn:
            return
        if get_actor_display_name(event.other_actor) != "Road":
            self.terminal_state = True
            self.collision_flag = True
        if self.activate_render:
            self.hud.notification("Collision with {}".format(get_actor_display_name(event.other_actor)))

    def _on_invasion(self, event):
        lane_types = set(x.type for x in event.crossed_lane_markings)
        text = ["%r" % str(x).split()[-1] for x in lane_types]
        if self.activate_render:
            self.hud.notification("Crossed line %s" % " and ".join(text))

    def _set_observation_image(self, image):
        self.observation_buffer = image

    def _set_viewer_image(self, image):
        self.viewer_image_buffer = image

    def _set_lidar_data(self, image):
        self.lidar_data_buffer = image
