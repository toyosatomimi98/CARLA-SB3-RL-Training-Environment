# 🚗 CARLA-SB3-RL-Training-Environment

> **This is a fork.** 上游是 Alberto Maté 在 UC3M 的本科毕业论文工作
> "Application of Deep Reinforcement Learning in autonomous driving"——一个开箱即用的
> CARLA + [Stable-Baselines3](https://stable-baselines3.readthedocs.io/en/master/) 训练/评估环境
> （上游要求 **CARLA 0.9.13**，只在 Town02 上验证过）。
>
> 本 fork 在它上面做了两件事：① 把它**移植到本机的 CARLA 0.10.0**（UE5.5 / Town10HD_Opt）；
> ② 在策略与仿真器之间加了 **CBF（control barrier function）安全滤波器**，并把 barrier
> 违反量写进 reward——这是 **CEG5306 Homework 3** 的提交。逐条改动见
> [`docs_safety/01-环境审计与移植记录.md`](docs_safety/01-环境审计与移植记录.md)，
> 作业视角的总结在上一层 [`../README.md`](../README.md)。
>
> 上游 README 的安装/配置/训练说明**保留在文末**「上游功能（保留）」一节，
> 但请以 §「本 fork 改了什么」为准——上游那套命令在本机跑不起来（版本、地图、车型都变了）。

<p align="center">
  <img width="88%" src="docs/images/rl_cbf_framework.png">
</p>

---

## 🧠 算法框架 / Algorithm framework

上图（`docs/images/rl_cbf_framework.png`，矢量版 `.svg` 同目录）就是本 fork 的完整数据流。

**闭环（15 Hz，CARLA sync 模式）**

1. **CARLA 0.10.0** 里跑 `vehicle.lincoln.mkz`（本机没有 `vehicle.tesla.model3`），
   地图 **Town10HD_Opt**（155 个 spawn point）。
2. **观测**取 `states["3"]` → **18 维**向量：`[steer, throttle, speed]` + **15 个相对路点**。
   ⚠️ 这里有个移植期的关键坑：`states["1"]` 里**完全没有横向信息**，策略只能"平行地"开，
   82% 的 episode 以 "Off-track" 结束。
3. **策略（SB3 PPO）**输出归一化动作 `u_rl = (steer, throttle)`，油门 ∈ [−1, 1]
   （`allow_brake=True`，负值 = 刹车），`action_smoothing = 0.0`。
4. **CBF 安全滤波器**每个控制步解一个小型 QP，把 `u_rl` 投影到安全集：

   ```
   u_safe = argmin_u  ‖W (u − u_rl)‖²  + ρ·Σ slack_j  − w_live·v_H
            s.t.  h_i(x_{k+1}(u)) ≥ (1 − α_i)·h_i(x_k) − slack_j ,  i = 1..5,  j = 1..H
                  u_min ≤ u ≤ u_max ,  slack_j ≥ 0
   ```

   * 预测模型：**运动学自行车**（L = 2.40 m），**时域 H = 20 步 @ 15 Hz = 1.33 s**；
     横向速率取自**实测差分**而不是理想模型（这台车上的模型误差与车道余量同量级）。
   * **slack（松弛）+ standstill recovery + liveness 项**：纯安全约束会出现"停死在原地"
     的退化解，这三样是让它"既安全、又还能走"的关键。
5. **动作真正下发给仿真器**（`VehicleControl(steer, throttle/brake)`），下一步回到 1。
6. **reward** = 任务奖励（进度 / 航向 / 车速）**+ 2.0 · Σ max(0, −h_i(x))**——
   这就是 `SAFETY_RL_SHAPED`：**CBF 既在测试期用、也在训练期用**。

**五条 barrier（安全集 C = {x : h_i(x) ≥ 0}）**

| barrier | 公式 | 含义 |
| --- | --- | --- |
| `h_lane±` | `(1.75 − 0.30) ∓ e_y` | 留在车道走廊内（半宽 1.75 m，另留 0.30 m 保守量） |
| `h_speed` | `13.9 − v` | 不超速（13.9 m/s = 50 km/h） |
| `h_curve` | `√(a_lat_max / κ) − v` | 按当前曲率 κ 限速（a_lat_max = 2.5 m/s²） |
| `h_gap` | `gap − (5.0 + 1.2·v)` | 跟车时间头距（d_min = 5 m，τ = 1.2 s） |

离散时间 CBF 的 class-K 增益统一取 α = 0.30；约束要在**整个 20 步时域**成立，
所以 QP 有时间在约束生效前先刹车（10 步 = 0.67 s 在 50 km/h 下太短，车拉不回来）。

**四套配置（`config.py`）**：`SAFETY_RL`（开滤波、无 shaping）、
`SAFETY_RL_SHAPED`（开滤波 + 2.0 barrier 惩罚）、`SAFETY_RL_UNSHIELDED`（基线：裸策略）、
`SAFETY_RL_DEMO`（关掉训练用的"停车终止"，专供录像与评测）。

---

## 🎬 效果 demo 关键帧 / Demo keyframes

以下是**压缩版 JPEG 副本**（放在 `docs/images/demo_keyframes/`，为的是 GitHub 上能直接渲染；原始 PNG 在主仓库 `../demo_keyframes/`，原视频在 `../` 与 `safety_results/demo/`，
每段自带逐帧遥测 CSV）。**对照类**的上下两半是**同一个 PPO checkpoint、同一 seed / 路线 / 障碍物**，
唯一差别是**有没有开 CBF 滤波器**。

| 关键帧 | 画面说明了什么 |
| --- | --- |
| ![](docs/images/demo_keyframes/01_CUTIN_no-shield_contact_vs_CBF_safe.jpg) | **加塞 cut-in（首推的"一成功一失败"）**：上=无滤波，NPC 切入后踩急刹，**0.00 m 接触**，红条 `*** CONTACT / UNSAFE ***`、min h = −9.93、介入 0；下=有 CBF，减速让行、**0.5 km/h 停住**、min h = +0.18、介入 **124/181**。 |
| ![](docs/images/demo_keyframes/02_ROADBLOCK_no-shield_crash_vs_CBF_stop.jpg) | **静态路障（前方 45 m）**：上=无滤波**真实碰撞**（min h = −7.60）；下=有 CBF **5.13 m 处停住**并保持到录像结束。 |
| ![](docs/images/demo_keyframes/03_LEADBRAKE_no-shield_crash_vs_CBF_stop.jpg) | **前车急刹**：上=无滤波**碰撞**（min h = −4.98）；下=有 CBF **保持 5.35 m 停住**。 |
| ![](docs/images/demo_keyframes/04_MAIN_badcmd_offlane_vs_inlane.jpg) | **注入坏指令**：上=无滤波被带出车道；下=CBF 把动作拉回、**留在车道内**——最能说明"滤波器只改它必须改的那一点"。 |
| ![](docs/images/demo_keyframes/05_TRAFFIC_live_traffic.jpg) | **动态交通**：Traffic Manager 生成 **14 辆自动驾驶车 + 6 个行人**（同步模式），BEV 与 `gap / TTC` 都能看到车流交互；⚠️ 但默认车流**不主动制造冲突**，这一段开/关滤波几乎没有差别（如实标注，不算滤波器的功劳）。 |
| ![](docs/images/demo_keyframes/06_MAIN_normal_t6.jpg) | **主视频三帧之一 · 正常行驶（t = 6 s）**：滤波不干预，策略动作原样通过（介入条带全绿）。 |
| ![](docs/images/demo_keyframes/07_MAIN_intervention_t11.jpg) | **主视频 · 介入中（t = 11 s）**：策略动作越界，滤波器改写它（介入条带转红，HUD 上 `steer (RL)` 与 `steer (safe)` 分离）。 |
| ![](docs/images/demo_keyframes/08_MAIN_recovery_t20.jpg) | **主视频 · 恢复（t = 20 s）**：车回到走廊内，滤波器把控制权交还给策略。 |

> HUD 面板从上到下是：**driver view / map BEV（正上方 45 m 俯视）/ DRIVING STATE**（车速、$e_y$、$e_\psi$、曲率、曲率允许车速）
> / **CONTROL**（策略动作 vs 滤波动作、介入计数）/ **NAVIGATION**（下一机动、路线进度、距终点）
> / **SAFETY**（五条 barrier 实时条形 + h = 0 边界 + min h + 介入次数）/ 三条时间序列
> / 底部整段介入条带（绿 = 策略原样通过，红 = 被滤波器改写）。

---

## 🔧 本 fork 改了什么（相对上游）/ What this fork adds

完整审计在 [`docs_safety/01-环境审计与移植记录.md`](docs_safety/01-环境审计与移植记录.md)，这里只列主干。

### 1. 移植：CARLA 0.9.13 → 0.10.0（不改就一行都跑不起来）

| 文件 | 改动 | 为什么 |
| --- | --- | --- |
| `carla_env/wrappers.py` | 车型 `vehicle.tesla.model3` → **`vehicle.lincoln.mkz`**；`color` 加 `has_attribute` 保护；`load_world` 参数化 `town` | 本机 0.10 构建只有 11 个车辆 blueprint，没有 tesla |
| `carla_env/envs/carla_route_env.py` | `import gym` → **`gymnasium`**、`reset/step` 改五元组；删掉 Town02 专用硬编码路线，改成**按当前地图随机采样 spawn 点对**（可复现）；新增 `collision_flag` 区分"真碰撞"与"压线 / 超速 / 停车" | SB3 ≥ 2.4 走 gymnasium API；本机只有 Town10HD_Opt，布局也不同 |
| 同上 | **`new_route()` 去掉 `set_simulate_physics(False/True)`**，改用清零速度 + `set_transform` | 0.10.0 上这套软重置会让自车**永久失去物理**（90% 油门 40 步只有 0.5 km/h）；SB3 每次 `learn()` 开头都 `reset()`，等于整轮都在跑一辆冻住的车。修后同条件 **36.1 km/h**（`tools_safety/probe_reset_physics.py` 可复现） |
| `carla_env/navigation/local_planner.py` | `RoadOption` 补 `CHANGELANELEFT = 5` / `CHANGELANERIGHT = 6` | 上游缺这两项，路线规划直接 `AttributeError` |
| `carla_env/state_commons.py` | `maneuver` 观测**夹到 [0, 3]**；`torchvision` 改可选导入 | 新增的 5 / 6 / −1 会让 SB3 的 `nn.Embedding` 下标越界 → **CUDA device-side assert** |
| `config.py` | 新增 `SAFETY*` 配置族（`vae_model=None`、`map=Town10HD_Opt`、`safety` 块） | 无 VAE 才能无头训练；安全参数集中可调 |
| 运行环境 | 仓库内 venv（`--system-site-packages` 继承本机 torch 2.8+cu128），补 SB3 2.9 / gymnasium 1.3 / opencv ≥ 4.10 | 复用已有 CUDA torch，不重装 |

### 2. 新增：CBF 安全层与配套工具（本作业的主体）

| 路径 | 内容 |
| --- | --- |
| `carla_env/safety/cbf.py` | barrier 定义、运动学预测、receding-horizon 约束、**带 slack 与 liveness 的 QP** 求解 |
| `carla_env/safety/wrapper.py` | gymnasium 安全壳：拦动作、记介入与 min h、可选的 reward shaping |
| `train_safety.py` | 带 shield 的训练入口（上游 `train.py` 没有安全层，且强依赖 VAE） |
| `tools_safety/` | 单测（`test_cbf.py`）/ 冒烟 / 场景库（`scenarios.py`）/ 评测台（`run_scenarios.py`）/ 演示录制（`make_demo_video.py`）/ 设置审计（`check_simulator_settings.py`）/ 一批探针（`probe_*.py`、`pick_best_*.py`） |
| `docs_safety/` | 审计与报告：01 移植记录、02 报告、03 仿真器设置审计、04 交互场景现状、05 参考控制器的隐藏 bug |

### 3. 实测效果（全部是"同一 checkpoint 开 / 关滤波"的对照）

| 指标 | 无滤波 | 有 CBF |
| --- | ---: | ---: |
| 场景挑战评测台（路障 / 前车急刹 / 加塞 / 路口 / 行人） | **2 / 5 通过** | **5 / 5 通过** |
| 最差 min h（5 集） | −0.53 | **−0.15**（深 −72%） |
| 最差 \|e_y\|（车道半宽 1.75 m） | 1.98 m（**出车道**） | **1.60 m（在车道内）** |
| 行驶距离 / 均速 | 126.6 m / 17.1 km/h | 127.3 m / 17.2 km/h（**没被拖慢**） |
| CBF 单测 / 仿真器设置审计 | — | **7/7** 与 **13/13** 通过 |

---

## 📥 Installation（本机实际）

```powershell
# 0) CARLA（本机固定这个版本与路径；0.9.x 在 RTX 5060 Ti / Win11 上起不来）
Start-Process "E:\carla\Carla-0.10.0-Win64-Shipping\CarlaUnreal.exe" `
  -ArgumentList "-quality_level=Low","-RenderOffScreen","-nosound","-prefernvidia","-carla-world-port=2000"

# 1) 环境：仓库内 venv（已被 .gitignore 忽略）
E:\anaconda3\envs\audiobook\python.exe -m venv --system-site-packages venv
.\venv\Scripts\python.exe -m pip install "stable-baselines3>=2.4" "gymnasium>=1.0" "opencv-python>=4.10" pygame
```

## 🛠️ Usage（本 fork 的命令）

```powershell
# 滤波器单测
.\venv\Scripts\python.exe tools_safety\test_cbf.py

# 训练 RL baseline（不带 shield；约 55 分钟 / 150k 步）
.\venv\Scripts\python.exe -u train_safety.py --config SAFETY_RL_UNSHIELDED `
  --no_render --no_rendering_mode --total_timesteps 150000 --tag rl_long

# 实验三件套：同一 checkpoint 的开/关 shield 对照 → 场景评测台 → 演示视频
powershell -NoProfile -ExecutionPolicy Bypass -File .\tools_safety\run_rl_experiments.ps1
```

> `--no_rendering_mode` 只在"观测里没有图像"的配置下可用（本作业的 RL 配置就是），
> 它把服务器渲染关掉、训练速度从 ~26 步/秒提到 **~45 步/秒**；带相机的录视频脚本不要加。
> 录像用 **`SAFETY_RL_DEMO`**：与 `SAFETY_RL` 完全相同，只关掉**训练用的"熄火终止"**——
> 否则滤波器把车正确刹停在障碍物后方之后，episode 会被提前结束，画面定格在红横幅上，看起来像撞车。

结果写在 `safety_results/`（`demo/` 视频与逐帧遥测、`shielded|unshielded/episodes.csv` 训练期逐集指标、`logs/`），
训练曲线在 `tensorboard/`。

## 📁 主要目录

| 路径 | 内容 |
| --- | --- |
| `carla_env/` | gymnasium 环境、观测 / 奖励、wrappers、**`safety/`（CBF 与安全壳）** |
| `train_safety.py` / `eval.py` / `eval_plots.py` | 训练与评估入口 |
| `tools_safety/` | 单测、场景、评测台、录制、探针（见上表） |
| `docs_safety/` | 审计记录与报告源码 / PDF |
| `docs/images/` | 上游插图 + 本 fork 的 `rl_cbf_framework.png` / `.svg` |
| `safety_results/`、`tensorboard/`、`venv/` | 运行产物与本地环境（**均不进版本库**） |

## ⚠️ 已知限制（如实说）

1. 本机 CARLA 0.10.0 构建**只有 `Town10HD_Opt` / `Mine_01` 两张地图、11 个车辆**，
   上游的 Town01/02 路线、预训练 VAE 与历史 checkpoint 都无法复用。
2. 阻挡场景里车会"**歪着停住**"：车道 barrier 与"必须动起来才能回正"互相锁死，
   现在靠 liveness 项与横向 / 纵向通道分离绕过，**根治要把 QP 写成带 slack 的软约束**。
3. **静态障碍物没有专属 barrier**：5 条 barrier 里只有 `h_gap` 在看"前方车辆"，
   路边的静止车辆 / 建筑不在安全集内。
4. 滤波器把残余越界从 −0.53 压到 −0.15 但**没完全消掉**（单步模型误差 + slack）；
   方向是更长时域的 MPC 式滤波，或训练期就把 CBF 纳进去（`SAFETY_RL_SHAPED` 已提供）。
5. RL baseline 会开车但开得一般：稳定 ~17 km/h、一次约 126 m / 26 s；不开滤波时有 3/5 个 episode 出车道。

---

## 上游功能（保留）/ Upstream features

> 以下来自上游 README（Alberto Maté, UC3M）。**本机不保证能跑**：上游要 CARLA 0.9.13，
> 且依赖 Town02 的路线与 VAE 观测。保留在此是为了说明本 fork 的起点。

**配置（`config.py`）**：`algorithm`（SB3 任意算法）、`algorithm_params`、`state`（观测项列表，
见 `carla_env/state_commons.py`）、`vae_model`（`vae_64` / `vae_64_augmentation` / `None`）、
`action_smoothing`、`reward_fn` 与 `reward_params`（见 `carla_env/reward_functions.py`）、
`obs_res`、`seed`、`wrappers`。

**训练 / 评估（上游命令）**：

```bash
python train.py --config 0 --total_timesteps 1000000     # 训练；曲线在 tensorboard/
python evaluate.py --config 0 --model <checkpoint>       # 评估；路线改 carla_env/envs/carla_env.py 的 eval_routes
python run_experiments.py                                # 批量训练 + 评估
python vae/train_vae.py --epochs <N>                     # 训练 VAE
python carla_env/envs/collect_data_manual_env.py         # 手工采集数据
python carla_env/envs/collect_data_rl_env.py             # 用早期 RL 策略自动采集数据
```

上游插图（原样保留）：`docs/images/architecture.png`（上游架构图）、`docs/images/rl_train.gif`（训练演示）、
`docs/images/vae_comparation.png`（VAE 重建对比）、`docs/images/Town02_spawnpoints.png`（Town02 出生点）。

## 出处与引用 / Attribution

* 上游仓库：**CARLA-SB3-RL-Training-Environment** —— Alberto Maté（UC3M 本科毕业论文
  *Application of Deep Reinforcement Learning in autonomous driving*）。
* 本 fork 的移植与 CBF 安全层：**Li Ruiqin · A0356877E**，CEG5306 Homework 3（2026-10）。
* 相关交付：上一层 [`../README.md`](../README.md)（作业总览）、`../CEG5306-Homework3-Report.pdf`（英文 7 页报告）、
  `../CEG5306-Homework3-Report-中文对照.pdf`（逐节中文对照，自用）、
  `../CEG5306-Homework3-Schematic.png`（设计示意图，报告 Figure 1）、
  `../CEG5306-Homework3-Framework.png`（本文算法框架图的入库副本）。
