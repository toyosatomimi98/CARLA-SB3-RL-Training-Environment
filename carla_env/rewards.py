import numpy as np
from config import CONFIG

low_speed_timer = 0

min_speed = CONFIG["reward_params"]["min_speed"]
max_speed = CONFIG["reward_params"]["max_speed"]
target_speed = CONFIG["reward_params"]["target_speed"]
max_distance = CONFIG["reward_params"]["max_distance"]
max_std_center_lane = CONFIG["reward_params"]["max_std_center_lane"]
max_angle_center_lane = CONFIG["reward_params"]["max_angle_center_lane"]
penalty_reward = CONFIG["reward_params"]["penalty_reward"]
early_stop = CONFIG["reward_params"]["early_stop"]
# PORT (2026-10): the original reward admits a degenerate "stand still" optimum -
# staying put gives ~0 reward and never triggers the -10 terminal penalty, so
# policies collapse to not moving at all.  When this flag is set we also
# terminate episodes in which the agent never even reaches the first waypoint.
stop_if_no_progress = bool(CONFIG["reward_params"].get("stop_if_no_progress", False))
no_progress_timeout = float(CONFIG["reward_params"].get("no_progress_timeout", 12.0))
# PORT (2026-10-04): finer-grained terminations.  Upstream gated the low-speed,
# off-track and "too fast" terminations behind a single `early_stop` flag, so a
# config either forbade off-tracking or allowed unlimited speed.  For the RL
# baseline we want off-track to end the episode but want an over-speed policy to
# be *punished by the reward* (and then fixed by the CBF) instead of being
# killed - a policy that is terminated the moment it exceeds 50 km/h never gets
# to learn what overspeed costs.
stop_if_off_track = bool(CONFIG["reward_params"].get("stop_if_off_track", True))
stop_if_too_fast = bool(CONFIG["reward_params"].get("stop_if_too_fast", True))
low_speed_timeout = float(CONFIG["reward_params"].get("low_speed_timeout", 5.0))
# The stall termination is a *training* device: it exists so the learner cannot
# farm "do nothing".  While recording or evaluating, the same rule is harmful -
# a car that has correctly stopped behind an obstacle would be reported as an
# ended episode, and a recording that freezes on that looks like a crash.
# Configs used for evaluation/recording set `stall_termination=False`.
stall_termination = bool(CONFIG["reward_params"].get("stall_termination", True))
reward_functions = {}


def create_reward_fn(reward_fn):
    def func(env):
        terminal_reason = "Running..."
        if early_stop:
            # Stop if the car has been *stalled* for `low_speed_timeout` seconds.
            # NOTE (2026-10-04): upstream added 1/fps to this timer on every step
            # and reset it only on termination, so the test degenerated into
            # "speed < 1 km/h at any step after t = 5 s" and cut most episodes
            # off after ~6 s of a 26.7 s budget - the learner never saw a long
            # drive.  Now the timer is a real stall detector: any step above the
            # threshold resets it.
            global low_speed_timer
            speed = env.vehicle.get_speed()
            if speed < 1.0:
                low_speed_timer += 1.0 / env.fps
            else:
                low_speed_timer = 0.0
            if (stall_termination and low_speed_timer > low_speed_timeout
                    and env.current_waypoint_index >= 1):
                env.terminal_state = True
                terminal_reason = "Vehicle stopped"

            # Stop if distance from center > max distance
            if stop_if_off_track and env.distance_from_center > max_distance:
                env.terminal_state = True
                terminal_reason = "Off-track"

            # Stop if speed is too high
            if stop_if_too_fast and max_speed > 0 and speed > max_speed:
                env.terminal_state = True
                terminal_reason = "Too fast"

            # PORT: kill the "stand still forever" degenerate solution
            if stop_if_no_progress and not env.terminal_state:
                elapsed = env.step_count / max(env.fps, 1)
                if elapsed > no_progress_timeout and env.current_waypoint_index < 1:
                    env.terminal_state = True
                    terminal_reason = "Not moving"

        # Calculate reward
        reward = 0
        if not env.terminal_state:
            reward += reward_fn(env)
        else:
            low_speed_timer = 0.0
            reward += penalty_reward
            print(f"{env.episode_idx}| Terminal: ", terminal_reason)

        if env.success_state:
            print(f"{env.episode_idx}| Success")

        env.extra_info.extend([
            terminal_reason,
            ""
        ])
        return reward

    return func


# Reward_fn5
def reward_fn5(env):
    """
        reward = Positive speed reward for being close to target speed,
                 however, quick decline in reward beyond target speed
               * centering factor (1 when centered, 0 when not)
               * angle factor (1 when aligned with the road, 0 when more than max_angle_center_lane degress off)
               * distance_std_factor (1 when std from center lane is low, 0 when not)
    """

    angle = env.vehicle.get_angle(env.current_waypoint)
    speed_kmh = env.vehicle.get_speed()
    if speed_kmh < min_speed:  # When speed is in [0, min_speed] range
        speed_reward = speed_kmh / min_speed  # Linearly interpolate [0, 1] over [0, min_speed]
    elif speed_kmh > target_speed:  # When speed is in [target_speed, inf]
        # Interpolate from [1, 0, -inf] over [target_speed, max_speed, inf]
        speed_reward = 1.0 - (speed_kmh - target_speed) / (max_speed - target_speed)
    else:  # Otherwise
        speed_reward = 1.0  # Return 1 for speeds in range [min_speed, target_speed]

    # Interpolated from 1 when centered to 0 when 3 m from center
    centering_factor = max(1.0 - env.distance_from_center / max_distance, 0.0)

    # Interpolated from 1 when aligned with the road to 0 when +/- 20 degress of road
    angle_factor = max(1.0 - abs(angle / np.deg2rad(max_angle_center_lane)), 0.0)

    std = np.std(env.distance_from_center_history)
    distance_std_factor = max(1.0 - abs(std / max_std_center_lane), 0.0)

    # Final reward
    reward = speed_reward * centering_factor * angle_factor * distance_std_factor

    return reward


reward_functions["reward_fn5"] = create_reward_fn(reward_fn5)


def reward_fn_waypoints(env):
    """
        reward
            - Each time the vehicle overpasses a waypoint, it will receive a reward of 1.0
            - When the vehicle does not pass a waypoint, it receives a reward of 0.0
    """
    angle = env.vehicle.get_angle(env.current_waypoint)
    speed_kmh = env.vehicle.get_speed()
    if speed_kmh < min_speed:  # When speed is in [0, min_speed] range
        speed_reward = speed_kmh / min_speed  # Linearly interpolate [0, 1] over [0, min_speed]
    elif speed_kmh > target_speed:  # When speed is in [target_speed, inf]
        # Interpolate from [1, 0, -inf] over [target_speed, max_speed, inf]
        speed_reward = 1.0 - (speed_kmh - target_speed) / (max_speed - target_speed)
    else:  # Otherwise
        speed_reward = 1.0  # Return 1 for speeds in range [min_speed, target_speed]

    # Interpolated from 1 when centered to 0 when 3 m from center
    centering_factor = max(1.0 - env.distance_from_center / max_distance, 0.0)
    reward = (env.current_waypoint_index - env.prev_waypoint_index) + speed_reward * centering_factor
    return reward


reward_functions["reward_fn_waypoints"] = create_reward_fn(reward_fn_waypoints)


def reward_fn_progress(env):
    """Progress-based reward (port addition): makes *driving forward* the
    dominant objective, so that the "stand still" policy is no longer optimal.

        reward = 1.0 * (waypoints advanced this step)
               + 0.3 * speed_term * centering_factor
               - 0.5 * |heading error|

    Every route waypoint is 1 m apart, so the first term is (approximately) the
    distance driven during the step.
    """
    progress = float(env.current_waypoint_index - env.prev_waypoint_index)
    speed_kmh = env.vehicle.get_speed()
    if speed_kmh < min_speed:
        speed_term = speed_kmh / min_speed
    elif speed_kmh > target_speed:
        speed_term = max(1.0 - (speed_kmh - target_speed) / max(max_speed - target_speed, 1e-6), 0.0)
    else:
        speed_term = 1.0
    centering_factor = max(1.0 - env.distance_from_center / max_distance, 0.0)
    angle = env.vehicle.get_angle(env.current_waypoint)
    return 1.0 * progress + 0.3 * speed_term * centering_factor - 0.5 * abs(angle)


reward_functions["reward_fn_progress"] = create_reward_fn(reward_fn_progress)
