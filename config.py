import torch as th
from stable_baselines3.common.noise import NormalActionNoise
import numpy as np
from utils import lr_schedule

algorithm_params = {
    "PPO": dict(
        learning_rate=lr_schedule(1e-4, 1e-6, 2),
        gamma=0.98,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.05,
        n_epochs=10,
        n_steps=1024,
        policy_kwargs=dict(activation_fn=th.nn.ReLU,
                           net_arch=[dict(pi=[500, 300], vf=[500, 300])])
    ),
    "SAC": dict(
        learning_rate=lr_schedule(5e-4, 1e-6, 2),
        buffer_size=300000,
        batch_size=256,
        ent_coef='auto',
        gamma=0.98,
        tau=0.02,
        train_freq=64,
        gradient_steps=64,
        learning_starts=10000,
        use_sde=True,
        policy_kwargs=dict(log_std_init=-3, net_arch=[400, 300]),
    ),
    "DDPG": dict(
        gamma=0.98,
        buffer_size=200000,
        learning_starts=10000,
        action_noise=NormalActionNoise(mean=np.zeros(2), sigma=0.5 * np.ones(2)),
        gradient_steps=-1,
        learning_rate=lr_schedule(5e-4, 1e-6, 2),
        policy_kwargs=dict(net_arch=[400, 300]),
    ),
    "SAC_BEST": dict(
        learning_rate=lr_schedule(1e-4, 5e-7, 2),
        buffer_size=300000,
        batch_size=256,
        ent_coef='auto',
        gamma=0.98,
        tau=0.02,
        train_freq=64,
        gradient_steps=64,
        learning_starts=10000,
        use_sde=True,
        policy_kwargs=dict(log_std_init=-3, net_arch=[500, 300]),
    ),
}

states = {
    "1": ["steer", "throttle", "speed", "angle_next_waypoint", "maneuver"],
    "2": ["steer", "throttle", "speed", "maneuver"],
    "3": ["steer", "throttle", "speed", "waypoints"],
    "4": ["steer", "throttle", "speed", "angle_next_waypoint", "maneuver", "distance_goal"]
}

reward_params = {
    "reward_fn_5_default": dict(
        early_stop=True,
        min_speed=20.0,  # km/h
        max_speed=35.0,  # km/h
        target_speed=25.0,  # kmh
        max_distance=3.0,  # Max distance from center before terminating
        max_std_center_lane=0.4,
        max_angle_center_lane=90,
        penalty_reward=-10,
    ),
     "reward_fn_5_no_early_stop": dict(
         early_stop=False,
         min_speed=20.0,  # km/h
         max_speed=35.0,  # km/h
         target_speed=25.0,  # kmh
         max_distance=3.0,  # Max distance from center before terminating
         max_std_center_lane=0.4,
         max_angle_center_lane=90,
         penalty_reward=-10,
     ),
    "reward_fn_5_best": dict(
        early_stop=True,
        min_speed=20.0,  # km/h
        max_speed=35.0,  # km/h
        target_speed=25.0,  # kmh
        max_distance=2.0,  # Max distance from center before terminating
        max_std_center_lane=0.35,
        max_angle_center_lane=90,
        penalty_reward=-10,
    ),
    # PORT (2026-10): training-friendly params used with reward_fn_progress.
    # - larger off-track tolerance so early exploration is not punished instantly
    # - `stop_if_no_progress` terminates the degenerate "stand still" policy
    "train_progress": dict(
        early_stop=True,
        min_speed=15.0,
        max_speed=45.0,
        target_speed=25.0,
        max_distance=3.5,
        max_std_center_lane=0.6,
        max_angle_center_lane=90,
        penalty_reward=-5,
        stop_if_no_progress=True,
        no_progress_timeout=12.0,
    ),
    # v2: the first attempt still collapsed to "stand still".  Diagnosis: the
    # -5 terminal penalty made *any* movement risky early on, so the policy learnt
    # to avoid the penalty instead of driving.  v2 makes driving strictly better
    # than standing still: a small penalty, a much wider off-track tolerance and
    # a shorter no-progress timeout.
    "train_progress_v2": dict(
        early_stop=True,
        min_speed=15.0,
        max_speed=60.0,
        target_speed=25.0,
        max_distance=5.0,
        max_std_center_lane=0.8,
        max_angle_center_lane=90,
        penalty_reward=-1.0,
        stop_if_no_progress=True,
        no_progress_timeout=10.0,
    ),
    # v3 (used for the shipped drive config): keep the repo's *proven* speed
    # reward (`reward_fn5`), only soften the terminal penalty and widen the
    # off-track tolerance, and add the anti-"stand still" termination.  The
    # custom progress reward above over-weighted the heading penalty and made
    # driving look worse than standing still.
    "train_fn5_soft": dict(
        early_stop=True,
        min_speed=15.0,
        max_speed=50.0,
        target_speed=25.0,
        max_distance=4.0,
        max_std_center_lane=0.6,
        max_angle_center_lane=90,
        penalty_reward=-1.0,
        stop_if_no_progress=True,
        no_progress_timeout=10.0,
    ),
}

_CONFIG_1 = {
    "algorithm": "PPO",
    "algorithm_params": algorithm_params["PPO"],
    "state": states["3"],
    "vae_model": "vae_64",
    "action_smoothing": 0.75,
    "reward_fn": "reward_fn5",
    "reward_params": reward_params["reward_fn_5_default"],
    "obs_res": (160, 80),
    "seed": 100,
    "wrappers": []
}

_CONFIG_2 = {
    "algorithm": "SAC",
    "algorithm_params": algorithm_params["SAC"],
    "state": states["3"],
    "vae_model": "vae_64",
    "action_smoothing": 0.75,
    "reward_fn": "reward_fn5",
    "reward_params": reward_params["reward_fn_5_default"],
    "obs_res": (160, 80),
    "seed": 100,
    "wrappers": []
}

_CONFIG_3 = {
    "algorithm": "DDPG",
    "algorithm_params": algorithm_params["DDPG"],
    "state": states["3"],
    "vae_model": "vae_64",
    "action_smoothing": 0.75,
    "reward_fn": "reward_fn5",
    "reward_params": reward_params["reward_fn_5_default"],
    "obs_res": (160, 80),
    "seed": 100,
    "wrappers": []
}

_CONFIG_4 = {
    "algorithm": "PPO",
    "algorithm_params": algorithm_params["PPO"],
    "state": states["1"],
    "vae_model": "vae_64",
    "action_smoothing": 0.75,
    "reward_fn": "reward_fn5",
    "reward_params": reward_params["reward_fn_5_default"],
    "obs_res": (160, 80),
    "seed": 100,
    "wrappers": []
}

_CONFIG_5 = {
    "algorithm": "SAC",
    "algorithm_params": algorithm_params["SAC"],
    "state": states["1"],
    "vae_model": "vae_64",
    "action_smoothing": 0.75,
    "reward_fn": "reward_fn5",
    "reward_params": reward_params["reward_fn_5_default"],
    "obs_res": (160, 80),
    "seed": 100,
    "wrappers": []
}

_CONFIG_6 = {
    "algorithm": "DDPG",
    "algorithm_params": algorithm_params["DDPG"],
    "state": states["1"],
    "vae_model": "vae_64",
    "action_smoothing": 0.75,
    "reward_fn": "reward_fn5",
    "reward_params": reward_params["reward_fn_5_default"],
    "obs_res": (160, 80),
    "seed": 100,
    "wrappers": []
}

_CONFIG_7 = {
    "algorithm": "SAC",
    "algorithm_params": algorithm_params["SAC"],
    "state": states["1"],
    "vae_model": "vae_64_augmentation",
    "action_smoothing": 0.75,
    "reward_fn": "reward_fn5",
    "reward_params": reward_params["reward_fn_5_default"],
    "obs_res": (160, 80),
    "seed": 100,
    "wrappers": []
}

_CONFIG_8 = {
    "algorithm": "SAC",
    "algorithm_params": algorithm_params["SAC"],
    "state": states["1"],
    "vae_model": "vae_64",
    "action_smoothing": 0,
    "reward_fn": "reward_fn5",
    "reward_params": reward_params["reward_fn_5_default"],
    "obs_res": (160, 80),
    "seed": 100,
    "wrappers": []
}

_CONFIG_9 = {
    "algorithm": "SAC",
    "algorithm_params": algorithm_params["SAC"],
    "state": states["1"],
    "vae_model": "vae_64",
    "action_smoothing": 0.75,
    "reward_fn": "reward_fn5",
    "reward_params": reward_params["reward_fn_5_no_early_stop"],
    "obs_res": (160, 80),
    "seed": 100,
    "wrappers": []
}

_CONFIG_10 = {
    "algorithm": "SAC",
    "algorithm_params": algorithm_params["SAC"],
    "state": states["2"],
    "vae_model": "vae_64", # Cambiar mejor VAE
    "action_smoothing": 0.75,
    "reward_fn": "reward_fn5",
    "reward_params": reward_params["reward_fn_5_default"],
    "obs_res": (160, 80),
    "seed": 100,
    "wrappers": []
}

_CONFIG_11 = {
    "algorithm": "SAC",
    "algorithm_params": algorithm_params["SAC"],
    "state": states["4"],
    "vae_model": "vae_64", # Cambiar mejor VAE
    "action_smoothing": 0.75,
    "reward_fn": "reward_fn5",
    "reward_params": reward_params["reward_fn_5_default"],
    "obs_res": (160, 80),
    "seed": 100,
    "wrappers": []
}

_CONFIG_12 = {
    "algorithm": "SAC",
    "algorithm_params": algorithm_params["SAC"],
    "state": states["1"],
    "vae_model": "vae_64_augmentation",
    "action_smoothing": 0.75,
    "reward_fn": "reward_fn5",
    "reward_params": reward_params["reward_fn_5_default"],
    "obs_res": (160, 80),
    "seed": 100,
    "wrappers": []
}

_CONFIG_13 = {
    "algorithm": "SAC",
    "algorithm_params": algorithm_params["SAC"],
    "state": states["1"],
    "vae_model": "vae_64_augmentation",
    "action_smoothing": 0.75,
    "reward_fn": "reward_fn5",
    "reward_params": reward_params["reward_fn_5_default"],
    "obs_res": (160, 80),
    "seed": 100,
    "wrappers": ["HistoryWrapperObsDict_5"]
}

_CONFIG_14 = {
    "algorithm": "SAC",
    "algorithm_params": algorithm_params["SAC"],
    "state": states["1"],
    "vae_model": "vae_64_augmentation",
    "action_smoothing": 0.75,
    "reward_fn": "reward_fn5",
    "reward_params": reward_params["reward_fn_5_default"],
    "obs_res": (160, 80),
    "seed": 100,
    "wrappers": ["FrameSkip_3"]
}

_CONFIG_15 = {
    "algorithm": "SAC",
    "algorithm_params": algorithm_params["SAC"],
    "state": states["1"],
    "vae_model": "vae_64_augmentation",
    "action_smoothing": 0.75,
    "reward_fn": "reward_fn5",
    "reward_params": reward_params["reward_fn_5_default"],
    "obs_res": (160, 80),
    "seed": 100,
    "wrappers": ["FrameSkip_3", "HistoryWrapperObsDict_5"]
}

_CONFIG_BEST = {
    "algorithm": "SAC",
    "algorithm_params": algorithm_params["SAC_BEST"],
    "state": states["1"],
    "vae_model": "vae_64_augmentation",
    "action_smoothing": 0.75,
    "reward_fn": "reward_fn5",
    "reward_params": reward_params["reward_fn_5_best"],
    "obs_res": (160, 80),
    "seed": 100,
    "wrappers": []
}
CONFIGS = {
    "1": _CONFIG_1,
    "2": _CONFIG_2,
    "3": _CONFIG_3,
    "4": _CONFIG_4,
    "5": _CONFIG_5,
    "6": _CONFIG_6,
    "7": _CONFIG_7,
    "8": _CONFIG_8,
    "9": _CONFIG_9,
    "10": _CONFIG_10,
    "11": _CONFIG_11,
    "12": _CONFIG_12,
    "13": _CONFIG_13,
    "14": _CONFIG_14,
    "15": _CONFIG_15,
    "BEST": _CONFIG_BEST
}

# ---------------------------------------------------------------------------
# PORTED (2026-10) safety-constrained driving configs
#   - CARLA 0.10.0 / Town10HD_Opt / vehicle.lincoln.mkz
#   - no VAE (vector observations only) so training runs headless
#   - `safety` block configures the CBF filter used by SafetyShieldWrapper
# ---------------------------------------------------------------------------
_SAFETY_BASE = {
    "algorithm": "SAC",
    "algorithm_params": algorithm_params["SAC_BEST"],
    "state": states["1"],          # steer, throttle, speed, angle_next_waypoint, maneuver
    "vae_model": None,
    "action_smoothing": 0.75,
    "reward_fn": "reward_fn5",
    "reward_params": reward_params["reward_fn_5_best"],
    "obs_res": (160, 80),
    "seed": 100,
    "wrappers": [],
    "map": "Town10HD_Opt",
    "safety": {
        "enabled": True,
        "penalty": 2.0,            # reward shaping: penalise barrier violation
        "lane_half_width": 1.75,
        "lane_margin": 0.30,
        "speed_limit": 13.9,       # m/s (= 50 km/h)
        "a_lat_max": 2.5,
        # 2026-10-05: 10 steps = 0.67 s of preview was too myopic for a policy
        # that sits at the 50 km/h limit: by the time the lane/curve barrier was
        # tight, full braking could no longer pull the car back.  20 steps
        # (1.33 s) gives the QP time to slow down before the constraint binds.
        "horizon": 20,
        "alpha_lane": 0.30,
        "alpha_speed": 0.30,
        "alpha_curve": 0.30,
    },
}

_CONFIG_SAFETY = dict(_SAFETY_BASE)
_CONFIG_SAFETY_PPO = dict(_SAFETY_BASE, algorithm="PPO",
                          algorithm_params=algorithm_params["PPO"])
_CONFIG_SAFETY_UNSHIELDED = dict(_SAFETY_BASE)
_CONFIG_SAFETY_UNSHIELDED["safety"] = dict(_SAFETY_BASE["safety"], enabled=False, penalty=0.0)

CONFIGS["SAFETY"] = _CONFIG_SAFETY
CONFIGS["SAFETY_PPO"] = _CONFIG_SAFETY_PPO
CONFIGS["SAFETY_UNSHIELDED"] = _CONFIG_SAFETY_UNSHIELDED
CONFIGS["SAFETY_PPO_UNSHIELDED"] = dict(
    _CONFIG_SAFETY_PPO,
    safety=dict(_SAFETY_BASE["safety"], enabled=False, penalty=0.0))

# Configs that use the *progress* reward so the learned policy actually drives
# (the original reward has a degenerate "stand still" optimum, see rewards.py).
_CONFIG_DRIVE = dict(
    _SAFETY_BASE,
    algorithm="PPO",
    algorithm_params=algorithm_params["PPO"],
    # no action smoothing: with smoothing 0.75 the agent must hold a high
    # throttle for many steps before the car moves, which makes early learning
    # much harder.
    action_smoothing=0.0,
    reward_fn="reward_fn5",
    reward_params=reward_params["train_fn5_soft"],
)
CONFIGS["SAFETY_PPO_DRIVE"] = dict(
    _CONFIG_DRIVE, safety=dict(_SAFETY_BASE["safety"], enabled=True, penalty=2.0))
CONFIGS["SAFETY_PPO_DRIVE_UNSHIELDED"] = dict(
    _CONFIG_DRIVE, safety=dict(_SAFETY_BASE["safety"], enabled=False, penalty=0.0))

# SAC variant of the "drive" config: off-policy, higher sample efficiency, and a
# smaller `learning_starts` so it actually explores within a 30k-step budget.
_SAC_DRIVE_PARAMS = dict(algorithm_params["SAC_BEST"])
_SAC_DRIVE_PARAMS.update(learning_starts=2000, train_freq=8, gradient_steps=8,
                         batch_size=256)
CONFIGS["SAFETY_SAC_DRIVE"] = dict(
    _CONFIG_DRIVE,
    algorithm="SAC",
    algorithm_params=_SAC_DRIVE_PARAMS,
    safety=dict(_SAFETY_BASE["safety"], enabled=True, penalty=2.0),
)
CONFIGS["SAFETY_SAC_DRIVE_UNSHIELDED"] = dict(
    CONFIGS["SAFETY_SAC_DRIVE"],
    safety=dict(_SAFETY_BASE["safety"], enabled=False, penalty=0.0),
)

# "Speed" config used for the demo video.
# Rationale: with the safety-flavoured rewards the PPO mean action collapses to
# ~zero throttle (a policy that does not move).  Here the reward is ONLY the
# repo's speed/centering term with early stopping disabled, so the learned mean
# action actually drives - and precisely because it drives fast the CBF safety
# filter has real work to do (this is what the homework asks to illustrate).
CONFIGS["SAFETY_PPO_SPEED"] = dict(
    _SAFETY_BASE,
    algorithm="PPO",
    algorithm_params=algorithm_params["PPO"],
    action_smoothing=0.0,
    reward_fn="reward_fn5",
    reward_params=reward_params["reward_fn_5_no_early_stop"],
    safety=dict(_SAFETY_BASE["safety"], enabled=True, penalty=0.0),
)
CONFIGS["SAFETY_PPO_SPEED_UNSHIELDED"] = dict(
    CONFIGS["SAFETY_PPO_SPEED"],
    safety=dict(_SAFETY_BASE["safety"], enabled=False, penalty=0.0),
)

# Same as SAFETY_PPO_SPEED but the throttle has a floor (see CarlaRouteEnv
# `throttle_bias`), which removes the "must discover sustained throttle"
# exploration problem that made every earlier policy stand still.
CONFIGS["SAFETY_PPO_ROLL"] = dict(CONFIGS["SAFETY_PPO_SPEED"], throttle_bias=0.45)
CONFIGS["SAFETY_PPO_ROLL_UNSHIELDED"] = dict(
    CONFIGS["SAFETY_PPO_ROLL"],
    safety=dict(_SAFETY_BASE["safety"], enabled=False, penalty=0.0),
)

# ---------------------------------------------------------------------------
# RL baseline used by the Homework 3 experiments (2026-10-04)
#
# Homework 3 is about *safe RL*, so the barrier has to act on a learned policy
# rather than on a hand-written controller.  This config fixes the three things
# that stopped the earlier policies from being usable:
#
#   * `allow_brake=True` - the training action space now matches the evaluation
#     and demo environments (throttle channel [-1, 1]), so the filter has a
#     brake to use and the policy that is evaluated is the policy that was
#     trained.  Earlier runs trained with throttle in [0, 1] and were then
#     evaluated in [-1, 1].
#   * `action_smoothing=0` - the action the QP certifies is the action that is
#     executed, so the one-step barrier guarantee holds exactly instead of
#     approximately.
#   * softened `reward_fn5` - the repo reward terminates at 2 m off centre with
#     -10, which punishes driving harder than standing still: a real trajectory
#     (even the reference controller) reaches |e_y| ~ 1.2-2.8 m, while standing
#     still never triggers the low-speed termination.  The corridor here is 4 m
#     and the terminal penalty is -5, so driving is strictly better.
# ---------------------------------------------------------------------------
_RL_REWARD = dict(reward_params["train_fn5_soft"])
_RL_REWARD.update(
    max_distance=4.0, penalty_reward=-5.0,
    # do NOT kill the episode the moment the car exceeds the limit: an
    # over-speeding policy must be *punished* by the reward (and then corrected
    # by the CBF), otherwise it can never learn what overspeed costs.
    stop_if_too_fast=False,
    # a real stall detector (see carla_env/rewards.py)
    low_speed_timeout=3.0,
)
_RL_PPO = dict(algorithm_params["PPO"])
_RL_PPO.update(n_steps=2048, batch_size=512,
               # 0.05 kept the action std at ~1.0 (i.e. the policy stayed almost
               # random) after 45k steps; a smaller entropy bonus lets the mean
               # sharpen within the training budget we have.
               ent_coef=0.01)
_CONFIG_RL = dict(
    _SAFETY_BASE,
    algorithm="PPO",
    algorithm_params=_RL_PPO,
    # CRITICAL (2026-10-05): `states["1"]` (steer, throttle, speed, heading
    # error, maneuver) has **no lateral information at all**, so the policy
    # cannot tell how far off the lane centre it is - it can only drive
    # *parallel* to the lane, exactly like the heading-only reference controller
    # did before the lateral term was added.  82% of the episodes ended with
    # "Off-track" because of this.  `states["3"]` adds the 15 relative route
    # waypoints, which is what makes lane keeping learnable.
    state=states["3"],
    action_smoothing=0.0,
    reward_fn="reward_fn5",
    reward_params=_RL_REWARD,
    allow_brake=True,
)
CONFIGS["SAFETY_RL"] = dict(
    _CONFIG_RL, safety=dict(_SAFETY_BASE["safety"], enabled=True, penalty=0.0))
CONFIGS["SAFETY_RL_UNSHIELDED"] = dict(
    _CONFIG_RL, safety=dict(_SAFETY_BASE["safety"], enabled=False, penalty=0.0))
# Shield active *and* shaping the reward (the "CBF during training" arm).
CONFIGS["SAFETY_RL_SHAPED"] = dict(
    _CONFIG_RL, safety=dict(_SAFETY_BASE["safety"], enabled=True, penalty=2.0))

# Identical to SAFETY_RL except that the *training* stall termination is off.
# Recording and evaluation use this: when the filter has correctly stopped the
# car behind an obstacle, the episode must keep running so the clip shows the
# car holding its stop instead of freezing on an "episode ended" banner.  The
# filter, the action space, the observation and the checkpoint are unchanged.
_RL_DEMO_REWARD = dict(_RL_REWARD)
_RL_DEMO_REWARD.update(stall_termination=False, stop_if_no_progress=False)
CONFIGS["SAFETY_RL_DEMO"] = dict(_CONFIG_RL, reward_params=_RL_DEMO_REWARD)
CONFIGS["SAFETY_RL_DEMO_UNSHIELDED"] = dict(
    CONFIGS["SAFETY_RL_DEMO"],
    safety=dict(_SAFETY_BASE["safety"], enabled=False, penalty=0.0))

CONFIG = None


def set_config(config_name):
    global CONFIG
    CONFIG = CONFIGS[config_name]
