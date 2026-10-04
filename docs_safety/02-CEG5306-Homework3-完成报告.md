# A Control Barrier Function Safety Filter for an Autonomous Driving Agent

**CEG5306 Homework 3 · Li Ruiqin · A0356877E**

## 1. Control problem

A car in CARLA 0.10.0 (map `Town10HD_Opt`) follows a randomly drawn urban route. The policy acts at 15 Hz and outputs a normalised action \(u_{\mathrm{rl}} = (\text{steer}, \text{throttle}) \in [-1,1]^2\), where a negative throttle commands the brake. The observation holds speed \(v\), lateral offset \(e_y\), heading error \(e_\psi\), path curvature \(\kappa\) and the next manoeuvre. A trained PPO policy supplies \(u_{\mathrm{rl}}\); the filter sees only the action and the measured vehicle state.

The safety constraints are: stay inside the lane corridor; keep the speed at or below 50 km/h; keep the lateral acceleration at or below 2.5 m/s² in curves; hold a time headway behind an obstacle in the ego lane.

![Design overview: the policy proposes an action, the safety filter solves the QP and projects it onto the safe set, and CARLA executes the result; the five barriers used by the QP are listed on the right.](figures/fig0_schematic.png)

## 2. Control barrier functions

Constants: lane half-width \(w/2 = 1.75\) m, lane margin \(m = 0.30\) m, speed limit \(v_{\max} = 13.9\) m/s, lateral-acceleration limit \(a_{\mathrm{lat}} = 2.5\) m/s², minimum gap \(d_{\min} = 5\) m, time headway \(\tau = 1.2\) s.

| Constraint | \(h_i(x)\) |
| --- | --- |
| lane, right | \(h_1 = (w/2 - m) - e_y\) |
| lane, left | \(h_2 = (w/2 - m) + e_y\) |
| speed | \(h_3 = v_{\max} - v\) |
| curve | \(h_4 = \sqrt{a_{\mathrm{lat}}/\kappa} - v\) |
| headway | \(h_5 = \text{gap} - (d_{\min} + \tau v)\) |

The lane constraint is split into two barriers so that each is smooth in \(e_y\). The safe set is \(\mathcal{C} = \{x : h_i(x) \ge 0,\ \forall i\}\).

The filter enforces a discrete-time condition over a horizon of \(H = 20\) steps at \(\Delta t = 1/15\) s:

\[ h_i(x_{k+j}) \ge (1-\alpha_i)^j\, h_i(x_k), \qquad j = 1, \dots, H. \]

The first prediction step takes the lateral rate from the measured finite difference \(\dot{e}_y \approx (e_y(k) - e_y(k-1))/\Delta t\) rather than from the kinematic model. A single-step condition with only steering authority cannot pull a car that is already leaving the lane back inside, and on this vehicle the model error of the bicycle model is comparable to the lane margin.

## 3. Safety filter

At every step the filter solves a quadratic program over the two-dimensional action:

\[ \min_{u,\ s \ge 0}\ \lVert W(u - u_{\mathrm{rl}}) \rVert^2 + \rho \sum_j s_j + \ell(u), \qquad \text{s.t. the barrier conditions above, with slack } s. \]

\(\ell(u) = \mu \max(0, v_{\min} - v_H(u))^2\) keeps the predicted speed at the end of the horizon above 3 km/h, so the filter does not stop the car when a moving action is feasible; it is disabled when the measured gap is below 20 m, where stopping is the correct response. The program is solved with SLSQP, with a grid search over \((\text{steer}, \text{throttle})\) as a fallback. If \(u_{\mathrm{rl}}\) is already safe the filter returns it unchanged (\(\lVert u^\star - u_{\mathrm{rl}} \rVert_\infty < 10^{-6}\)).

The same filter is active during training: the reward adds \(\lambda \max(0, -\min_i h_i)\) and the shield is enabled inside the environment, so the policy is optimised against filtered actions.

## 4. Results

### 4.1 Unit tests

`tools_safety/test_cbf.py` runs 7 test groups (9 assertions) without the simulator: an action that is already safe passes through unchanged; an unsafe speed or lane action is projected onto the boundary; the curve speed limit is monotone in curvature; and when the car is already outside the corridor the fallback is never worse than the raw action. All pass.

### 4.2 Aggressive constant policy, filter off and on

The policy is PPO, trained in the same CARLA environment for 150 000 steps (45 min wall clock, 57 steps/s). The observation is the vehicle state (steer, throttle, speed, heading error, maneuver) plus the 15 relative route waypoints; the action is \((\text{steer}, \text{throttle}) \in [-1,1]^2\) with a negative throttle commanding the brake, which is exactly the action space the filter and the recorded episodes use. The reward is the repository's speed/centering term with two terminations changed: the corridor is 4 m instead of 2 m, and the low-speed termination is a stall detector (3 s of continuous standstill) instead of "slower than 1 km/h at any time after 5 s".

![PPO learning curve](figures/fig2_learning_curve.png)

The trained policy drives. Over the last rollouts the mean episode reward is 250 and the mean episode length is 377 of the 400 allowed steps (25 s at about 17 km/h), so the baseline is a policy that completes the route rather than one that stands still at the start line. It is also not a safe policy: the lane, speed and curve barriers are all active at different points and the car leaves the lane on some routes.

### 4.3 Same checkpoint, filter off and on

Five episodes with the same seeds and the same deterministic checkpoint; the filter is the only difference between the two runs. All values are per episode except the two totals.

| Quantity | CBF off | CBF on |
| --- | --- | --- |
| \(\min_i h_i\) | \(+0.72,\ -0.53,\ -0.52,\ +1.43,\ -0.38\) | \(+1.01,\ -0.06,\ -0.08,\ +1.43,\ -0.15\) |
| violating steps | 125 | 61 |
| \(\max\lvert e_y\rvert\) [m] | \(0.73,\ 1.98,\ 1.97,\ 0.02,\ 1.83\) | \(0.44,\ 1.51,\ 1.53,\ 0.02,\ 1.60\) |
| distance [m] | \(126,\ 126,\ 127,\ 127,\ 127\) | \(125,\ 126,\ 129,\ 128,\ 129\) |
| mean speed [km/h] | 17.1 | 17.2 |
| interventions | 0 | 732 |

![Same checkpoint with and without the filter](figures/fig1_shield_comparison.png)

Three of the five unfiltered episodes leave the lane (\(\lvert e_y\rvert\) between 1.83 and 1.98 m against a half-width of 1.75 m); with the filter none do (worst 1.60 m). The depth of the deepest remaining violation drops from 0.53 to 0.15 and the number of violating steps is halved. The car covers the same distance at the same speed, so the filter does not buy safety by slowing the policy down.

### 4.4 Scenario benchmark

Five scenarios run with the same seed and route and with the same trained checkpoint, once with and once without the filter. PASS means no collision and no freeze.

| Scenario | Filter off | Filter on |
| --- | --- | --- |
| Roadblock 45 m ahead | collision | stops, 5.13 m |
| Lead vehicle brakes | collision | stops, 5.35 m |
| Cut-in with brake check | contact, 0.00 m | stops, 5.35 m |
| Crossing vehicle | pass | pass |
| Pedestrian crossing | pass | pass |
| Total | 2 / 5 | 5 / 5 |

The three scenarios that place an obstacle in the ego lane are the ones that discriminate: the learned policy hits them without the filter and stops behind them with it. The crossing vehicle and the pedestrian clear the ego lane ahead of the car on this route, so both runs pass and those two rows are not evidence for the filter.

### 4.5 Demonstration videos

Every clip is 30 s at 1280×720, 15 fps, with the raw policy on top and the filtered policy below. Both halves share seed, route, obstacle and checkpoint. The overlay shows speed, \(e_y\), \(e_\psi\), \(\kappa\), the policy and filtered steer commands, the next manoeuvre, the barrier values with \(\min_i h_i\), and a per-step strip marking the steps where the filter overrode the action.

| Clip | Filter off | Filter on |
| --- | --- | --- |
| `SCN_cutin` | contact at 6.1 s, \(-10.28\) | no collision, stops (gap 5.35 m), \(-0.03\) |
| `SCN_roadblock` | collision at 10.1 s, \(-7.60\) | no collision, stops (5.13 m), \(-0.17\) |
| `SCN_leadbrake` | collision at 7.0 s, \(-4.98\) | no collision, stops (5.35 m), \(-0.08\) |
| `MAIN_badcmd` | 121.8 m, \(\max\lvert e_y\rvert = 0.48\) m | 121.8 m, \(0.44\) m, 34 overrides |
| `TRAFFIC` | no conflict | no conflict |

![Cut-in: contact without the filter, safe stop with it|60%](figures/fig6_cutin_comparison.png)

![The filter replacing the policy action|80%](figures/fig4_demo_intervention.png)

In the three conflict clips the unfiltered policy hits the obstacle - a real collision event in the roadblock and lead-brake scenarios and a measured 0.00 m gap in the cut-in - while with the filter the car never collides: it brakes to a stop behind the obstacle and holds that stop for the rest of the 26.7 s clip. The barrier value at the stop is between \(-0.03\) and \(-0.17\), against \(-4.98\) to \(-10.28\) at the moment the unfiltered car hits. The numbers in the table are \(\min_i h_i\) for the whole clip.

In `MAIN_badcmd` there is no obstacle: the policy drives the whole 121.8 m at 16 km/h both with and without the filter, the filter overrides 34 of 400 steps and leaves the rest untouched, and the lateral excursion is the same (0.48 m against 0.44 m). The filter does not brake by default.

### 4.6 Simulator settings

`tools_safety/check_simulator_settings.py` checks 13 settings and all pass: synchronous mode with a fixed step of 1/15 s, substepping, rendering enabled, measured displacement matching \(v\Delta t\), and bit-identical trajectories for two runs with the same seed and actions. The last check is what makes the off/on comparison a controlled experiment.

## 5. Implementation notes

Two groups of fixes were needed: making the repository run on this machine, and making the RL baseline usable.

The repository targets CARLA 0.9.13, gym 0.22 and image observations; this machine has CARLA 0.10.0, two maps and eleven vehicle blueprints.

* `gym` migrated to `gymnasium`; hard-coded `Town02` routes replaced by random spawn pairs on `Town10HD_Opt`.
* `maneuver` clamped to \([0,3]\); higher values trip a CUDA assert.
* Soft reset by `set_simulate_physics(False/True)` replaced by zeroing the velocities and teleporting.
* `close()` restores asynchronous mode and ticks, instead of leaving the world synchronous.
* Routes whose first leg merges sideways into the next lane are rejected and redrawn. Without this the car sits a full lane away from the line its lateral offset is measured against, and every run in the affected episodes reports a 3.5 m lane departure; 18 seeds produce no such route after the fix.

Each of the following made the earlier policies unusable:

* The training action space did not match the evaluation one: training did not enable the brake channel (throttle in \([0,1]\)) while the filter, the benchmark and the videos used \([-1,1]\). The policy is now trained with the brake channel.
* The training config applied action smoothing 0.75 while the benchmark used 0.0, so the action the QP certified was not the action that was executed. Smoothing is off everywhere now, which makes the one-step guarantee exact.
* The observation had no lateral information (steer, throttle, speed, heading error, maneuver), so the policy could only drive *parallel* to the lane, exactly as a heading-only controller does; 82 % of training episodes ended off-track because of it. The observation now includes the 15 relative route waypoints.
* The low-speed termination counted total episode time rather than time spent stalled, so it ended most episodes about 6 s into a 26.7 s budget. It is a stall detector now (3 s of continuous standstill).
* Off-track and over-speed terminations are now separate: an over-speeding policy is punished by the reward, and corrected by the filter, instead of being killed the moment it passes the limit.
* The environment fetched a dashboard camera frame on every step even for vector observations, which forced the server to keep rendering. Skipping it, and running the server with `no_rendering_mode`, raises training from 26 to 57 steps/s.

## 6. Limitations

* The baseline drives, but it is not a good driver: it settles at about 17 km/h, and without the filter it leaves the lane in 3 of the 5 evaluation episodes and hits the three in-lane obstacles in the scenario suite.
* The filter reduces the remaining violation but does not always remove it. In §4.3 the worst residual is \(-0.15\) against \(-0.53\) without the filter; it comes from the one-step lateral model and from the slack in the QP, which keeps the problem feasible when the car is already at the corridor edge. A longer-horizon model-based filter, or training the policy against the filter (`SAFETY_RL_SHAPED`), is the fix.
* Static obstacles have no barrier of their own. Only the headway barrier reacts to an obstacle in the lane, and it treats the obstacle as a stationary lead vehicle.
* The crossing-vehicle and pedestrian scenarios do not create a conflict on this route, so both arms pass them and those rows are not evidence for the filter.
* The lane barrier and the requirement to keep moving conflict when the car stops at an angle; the liveness term and separate lateral and longitudinal channels only work around this.
* Both training runs are 150 000 steps. The repository's own configuration uses 1 000 000.

## 7. What makes this submission special

The barrier acts on a **learned** policy trained in exactly the action space of the filter and of the recorded episodes, so the off/on comparison isolates the filter and not the driver, and the action the QP certifies is the action that is executed. Using the measured lateral rate instead of the bicycle model then lets the same checkpoint stay inside the lane (worst lateral excursion 1.98 m \(\rightarrow\) 1.60 m against a 1.75 m half-lane) at unchanged speed, and turns every collision in the scenario suite into a stop.

## 8. AI tool declaration

I used an AI coding assistant (Codex, running the DeepSeek Flash model) to write and refactor the CARLA environment code, to train and evaluate the policy, to run the experiments, to generate the figures and the design schematic, and to draft and polish the text of this report. I am responsible for the content and quality of the submitted work.
