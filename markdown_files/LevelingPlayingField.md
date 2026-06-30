---
title: "Leveling the Playing Field: Carefully Comparing Classical and Learned Controllers for Quadrotor Trajectory Tracking"
method_name: "Leveling the Playing Field"
authors: ["Pratik Kunapuli"]
year: 2025
venue: "RSS"
tags: ["robust-control", "reinforcement-learning", "agile-locomotion", "whole-body-control", "navigation"]
image_source: "online"
arxiv_html: "https://arxiv.org/html/2506.17832v1"
---
# Leveling the Playing Field
## 一句话总结

> Leveling the Playing Field: Carefully Comparing Classical and Learned Controllers for Quadrotor Trajectory Tracking 主要落在 [[aerial-robotics]]、[[agile-locomotion]]、[[motion-tracking]]、[[强化学习]] 这一类问题上;从 System0 视角看,它的价值在于把身体状态、接触变化、局部感知或动力学约束接到机器人低层闭环里,让动作生成更稳定、安全、实时。
## 核心贡献

1. 围绕 **Leveling the Playing Field: Carefully Comparing Classical and Learned Controllers for Quadrotor Trajectory Tracking** 建立了一个与 aerial-robotics、agile-locomotion、motion-tracking、强化学习、鲁棒控制 相关的机器人问题设定,重点不是高层语义规划,而是身体级闭环中的状态变化、接触变化或控制约束。
2. 方法层面的关键价值在于把感知、动力学、约束优化、策略学习或硬件反馈组织成可执行的低层动作生成机制,适合放进 System0 的快反应模块中理解。
3. 对系统集成而言,这类工作补上了 System1 技能选择与电机级控制之间的一层:它负责把上层目标变成动态可行、可恢复、可安全过滤的动作。
4. 若论文包含真实机器人、接触任务或仿真到真实迁移,核心应看它是否报告了闭环稳定性、扰动恢复、硬件延迟、失败模式和跨场景泛化。
## 问题背景

这篇论文可以放在 System0 的“具身快系统”背景下阅读:机器人需要在很短时间内根据本体状态、接触、力/触觉、局部视觉或动力学预测做出动作,而不是等待慢速规划器重新推理整个任务。aerial-robotics、agile-locomotion、motion-tracking、强化学习、鲁棒控制 相关问题的难点通常不在“知道要做什么”,而在动作执行过程中状态会快速偏离、接触会突然改变、约束会实时收紧,低层系统必须及时修正。

### System0 关联

精读时优先看三件事:第一,论文使用哪些快速反馈量,例如关节状态、接触状态、力觉、触觉、深度或局部视觉;第二,反馈如何进入 MPC、WBC、策略网络、安全过滤器或恢复控制器;第三,输出是否直接影响机器人下一步动作,而不是只服务离线分析或高层任务规划。
## 方法详解

### System0 视角解析

从 System0 角度看,这篇论文的方法部分应当围绕 aerial-robotics、agile-locomotion、motion-tracking、强化学习、鲁棒控制 来读。核心问题是:系统如何从快速变化的机器人状态中构造控制输入,如何处理接触、碰撞、约束、模型误差或执行延迟,以及如何把这些信息变成下一时刻的动作。

如果方法里有优化问题,重点看目标函数、约束和求解频率;如果方法里有学习策略,重点看观测空间、动作空间、训练扰动和部署时的闭环频率;如果方法里有硬件或传感器,重点看传感链路是否真的进入控制回路。这样读可以把论文和 System2 语言规划、System1 技能编排区分开来。

### 关键公式与机制

#### 公式 1: [[动力学或策略机制]]

$$\boldsymbol{M}=\begin{bmatrix}0,-l\cdot k_{t},0,l\cdot k_{t}\\ l\cdot k_{t},0,-l\cdot k_{t},0\\ -k_{m},k_{m},-k_{m},k_{m}\end{bmatrix}*\boldsymbol{\Omega}_{d}^{2},$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 2: [[优化目标/约束]]

$$o_{t}=\begin{bmatrix}{}^{\mathcal{B}}{\boldsymbol{e}}^{\mathcal{}}_{p}\\ {}^{\mathcal{B}}{\boldsymbol{e}}^{\mathcal{}}_{R}\\ {}^{\mathcal{B}}{\boldsymbol{g}}^{\mathcal{}}\\ {}^{\mathcal{B}}{\boldsymbol{e}}^{\mathcal{}}_{v}\\ {}^{\mathcal{B}}{\boldsymbol{e}}^{\mathcal{}}_{\omega}\end{bmatrix}$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 3: [[动力学或策略机制]]

$$r(t)=\lambda_{p}\phi(\boldsymbol{p}(t)-\boldsymbol{p}_{d}(t),\delta_{p})+\lambda_{R}(\left\lVert \psi(t)-\psi_{d}(t)\right \rVert)\\ +\lambda_{v}(\left\lVert \boldsymbol{v}(t)-\boldsymbol{v}_{d}(t)\right \rVert)+\lambda_{\omega}(\left\lVert \boldsymbol{\omega}(t)-\boldsymbol{\omega}_{d}(t)\right \rVert).$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 4: [[动力学或策略机制]]

$$\begin{multlined}\boldsymbol{\dot{\omega}}_{des}=-K_{R}(\boldsymbol{e_{R}})-K_{\omega}(\boldsymbol{\omega}-\boldsymbol{\omega}_{d})-\\ (\hat{\boldsymbol{\omega}}\boldsymbol{R}^{T}\boldsymbol{R}_{des}\boldsymbol{\omega}_{d}-\boldsymbol{R}^{T}\boldsymbol{R}_{des}\boldsymbol{\dot{\omega}}_{d}).\end{multlined}\boldsymbol{\dot{\omega}}_{des}=-K_{R}(\boldsymbol{e_{R}})-K_{\omega}(\boldsymbol{\omega}-\boldsymbol{\omega}_{d})-\\ (\hat{\boldsymbol{\omega}}\boldsymbol{R}^{T}\boldsymbol{R}_{des}\boldsymbol{\omega}_{d}-\boldsymbol{R}^{T}\boldsymbol{R}_{des}\boldsymbol{\dot{\omega}}_{d}).$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 5: [[动力学或策略机制]]

$$g_{t}=\begin{bmatrix}x_{d}\\ y_{d}\\ z_{d}\\ \psi_{d}\end{bmatrix}=\begin{bmatrix}A_{x}sin(\omega_{x}\cdot t+\phi_{x})+\delta_{x}\\ A_{y}sin(\omega_{y}\cdot t+\phi_{y})+\delta_{y}\\ A_{z}sin(\omega_{z}\cdot t+\phi_{z})+\delta_{z}\\ A_{\psi}sin(\omega_{\psi}\cdot t+\phi_{\psi})+\delta_{\psi}\end{bmatrix}$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 6: [[优化目标/约束]]

$$o_{t}=\begin{bmatrix}{}^{\mathcal{B}}{\boldsymbol{e}}^{\mathcal{}}_{p}\\ {}^{\mathcal{B}}{\boldsymbol{e}}^{\mathcal{}}_{R}\\ {}^{\mathcal{B}}{\boldsymbol{g}}^{\mathcal{}}\\ {}^{\mathcal{B}}{\boldsymbol{e}}^{\mathcal{}}_{v}\\ {}^{\mathcal{B}}{\boldsymbol{e}}^{\mathcal{}}_{\omega}\end{bmatrix}=\begin{bmatrix}{}^{\mathcal{B}}{\boldsymbol{R}}^{\mathcal{W}}({}^{\mathcal{W}}{\boldsymbol{p}}^{\mathcal{B}}-{}^{\mathcal{W}}{\boldsymbol{p}}^{\mathcal{B}}_{d})\\ ({}^{\mathcal{W}}{\boldsymbol{R}}^{\mathcal{B}})^{T}{}^{\mathcal{W}}{\boldsymbol{R}}^{\mathcal{B}}_{d}\\ {}^{\mathcal{B}}{\boldsymbol{R}}^{\mathcal{W}}(g\boldsymbol{z}_{\mathcal{W}})\\ {}^{\mathcal{B}}{\boldsymbol{R}}^{\mathcal{W}}({}^{\mathcal{W}}{\boldsymbol{v}}^{\mathcal{B}}-{}^{\mathcal{W}}{\boldsymbol{v}}^{\mathcal{B}}_{d})\\ {}^{\mathcal{B}}{\boldsymbol{R}}^{\mathcal{W}}({}^{\mathcal{W}}{\boldsymbol{\omega}}^{\mathcal{B}}-{}^{\mathcal{W}}{\boldsymbol{\omega}}^{\mathcal{B}}_{d}),\end{bmatrix}$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 7: [[优化目标/约束]]

$$\ddot{\boldsymbol{p}}_{des}=-K_{p}(\boldsymbol{p}-\boldsymbol{p}_{d})-K_{v}(\boldsymbol{v}-\boldsymbol{v}_{d})-mg\boldsymbol{z}_{\mathcal{W}}+\ddot{\boldsymbol{p}}_{d}.$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 8: [[优化目标/约束]]

$$\ddot{\boldsymbol{p}}_{des}=-K_{p}(\boldsymbol{p}-\boldsymbol{p}_{d})-K_{v}(\boldsymbol{v}-\boldsymbol{v}_{d})-mg\boldsymbol{z}_{\mathcal{W}}+\ddot{\boldsymbol{p}}_{d},$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 9: [[动力学或策略机制]]

$$\boldsymbol{\dot{\Omega}}=\frac{1}{\tau_{m}}(\boldsymbol{\Omega}_{d}-\boldsymbol{\Omega}).$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 10: [[动力学或策略机制]]

$$\boldsymbol{R}_{des}=H_{1}(\psi_{d})H_{2}(\frac{\ddot{\boldsymbol{p}}_{des}}{\left\lVert \ddot{\boldsymbol{p}}_{des}\right \rVert}),$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。
## 关键图表

### Figure 1: Pipeline / core system figure: Trajectory Tracking for a Quadrotor. Rollouts of trajectory tracking from an initial

![Figure 1](https://arxiv.org/html/2506.17832v1/extracted/6560634/figures/AM_example_motion.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Pipeline / core system figure: Trajectory Tracking for a Quadrotor. Rollouts of trajectory tracking from an initial”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 2: The Impact of Each Type of Asymmetry in RL vs. GC Comparisons. Model comparisons in b

![Figure 2](https://arxiv.org/html/2506.17832v1/x1.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“The Impact of Each Type of Asymmetry in RL vs. GC Comparisons. Model comparisons in b”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 3: Trajectory Tracking Errors. Position and yaw errors time for both the best-in-cl

![Figure 3](https://arxiv.org/html/2506.17832v1/x2.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Trajectory Tracking Errors. Position and yaw errors time for both the best-in-cl”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### 论文表格

#### Table 1: TABLE I: Trajectory Tracking Controller Comparisons in Recent Literature. Partial survey of representative works propos

| | | Geometric Control (GC) | Reinforcement Learning (RL) | | | | |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Focus | Paper | Tuned for Obj.? | Traj Data? | Feed Forward? | Tuned for Obj.? | Traj Data? | Feed Forward? |
| RL | Benchmark of Learned Policies [13 ] | - | - | - | ✓ | ✓ | ✓ |
| SimpleFlight [5 ] | - | - | - | ✓ | ✓ | ✓ | |
| Leveraging Symmetry [29 ] | - | - | - | ✓ | ✓ | ✓ | |
| Sim-2-Multi-Real [20 ] | ✗ | ✗ | ✗ | ✓ | ✗ | ✗ | |
| Power of Input [7 ] | ✗ | ✗ | ✗ | ✓ | ✗ | ✗ | |
| DATT [12 ] | ✗ | ✗ | ∼ similar-to \sim ∼ | ✓ | ✓ | ✓ | |
| Learning to Fly in Seconds [8 ] | ✗ | ✗ | ✗ | ✓ | ✗ | ✗ | |
| GC | Geometric S E (3) S E 3 SE(3) (3) Control [15 ] | ✗ | ✓ | ✓ | - | - | - |
| INDI [26 ] | ✗ | ✓ | ✓ | - | - | - | |
| NMPC vs. DFBC [25 ] | ✗ | ✓ | ✓ | - | - | - | |
| PID AutoTune [27 ] | ✓ | ✓ | ✗ | - | - | - | |
| | NonLinear PID [21 ] | ✓ | ✓ | ∼ similar-to \sim ∼ | - | - | - |
| | Ours | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

#### Table 2: TABLE II: Initial Conditions. Randomization ranges for initialization in Hover and Tracking Lissajous tasks by state co

| | Task | |
| --- | --- | --- |
| State Component | Hover | Tracking Lissajous |
| Position (m m m) | [-2, 2] | [-0.5, 0.5] |
| Velocity (m / s m s m/s /) | [0, 0] | [-0.1, 0.1] |
| Yaw (r a d r a d rad) | $\pi, π π \pi ]$ | $\pi, π π \pi ]$ |
| Angular Velocity (r a d / s r a d s rad/s /) | [0, 0] | [-0.1, 0.1] |

**说明**: TABLE II: Initial Conditions. Randomization ranges for initialization in Hover and Tracking Lissajous tasks by state component.

#### Table 3: TABLE III: Objective Hyperparameters. Parameters which define the form of the reward function (objective) for trajector

| Parameter |  p  p \lambda_{p} |  R  R \lambda_{R} |  v  v \lambda_{v} | $\omega}$ | $\delta_{p}$ |
| --- | --- | --- | --- | --- | --- |
| Value | 15.0  d t  15.0 d t 15.0\cdot dt 15.0  | - 4.0  d t  4.0 d t -4.0\cdot dt - 4.0  | - 0.05  d t  0.05 d t -0.05\cdot dt - 0.05  | - 0.01  d t  0.01 d t -0.01\cdot dt - 0.01  | 0.8 → 0.1 → 0.8 0.1 0.8 $\rightarrow 0.1 0.8 → 0.1$ |

**说明**: TABLE III: Objective Hyperparameters. Parameters which define the form of the reward function (objective) for trajectory tracking tasks.

#### Table 4: TABLE IV: Trajectory Tracking for Quadrotor and Aerial Manipulator. Comparison of the best RL controller against the be

| | Quadrotor | Aerial Manipulator | | | | |
| --- | --- | --- | --- | --- | --- | --- |
| Controller | Avg. Reward | Position RMSE (m) | Yaw RMSE (rad) | Avg. Reward | Position RMSE (m) | Yaw RMSE (rad) |
| RL-Opt.-Liss.-FF | 14.196 $\pm$ 0.48 | 0.119 $\pm$ 0.05 | 0.274 $\pm$ 0.15 | 13.621 $\pm$ 1.28 | 0.118 $\pm$ 0.05 | 0.487 $\pm$ 0.26 |
| GC-Opt.-Liss.-FF | 13.447 $\pm$ 1.61 | 0.158 $\pm$ 0.20 | 0.483 $\pm$ 0.29 | 13.792 $\pm$ 1.28 | 0.136 $\pm$ 0.10 | 0.405 $\pm$ 0.29 |

**说明**: TABLE IV: Trajectory Tracking for Quadrotor and Aerial Manipulator. Comparison of the best RL controller against the best GC controller in trajectory tracking for a Quadrotor and an Aerial Manipulator (Quadrotor with fixed-arm) 1000 trials in the Lissajous Tracking task. Rewards are averaged time per rollout, then presented as average and standard deviation the trials. Maximum reward is 15.0 15.0 15.0 15.0. RMSE of position and yaw are shown as averages with standard deviation 1000 trials. Contrary to many literature claims, performance is very similar between best-in-class methods.

#### Table 5: TABLE V: Ball Catching success rate. Percentage of catches made by the aerial manipulator with varied initial velocitie

| | Time-To-Catch (s) | | | |
| --- | --- | --- | --- | --- |
| Controller | 0.79 | 1.09 | 1.53 | 1.99 |
| RL-EE | 0.65 | 1.0 | 1.0 | 0.99 |
| RL-COM | 0.72 | 0.75 | 0.85 | 0.94 |
| GC | 0.30 | 0.37 | 0.49 | 0.97 |

**说明**: TABLE V: Ball Catching success rate. Percentage of catches made by the aerial manipulator with varied initial velocities by each controller, denoted by the time-to-catch in each setting. Results are shown as mean catch rate 100 trials.

#### Table 6: TABLE VI: Controller Performance Domain Randomization. Comparison of RL and GC controllers optimized for Lissajou

| Controller | Avg. Reward | Position RMSE (m) | Yaw RMSE (rad) |
| --- | --- | --- | --- |
| RL-0 | 13.345 $\pm$ 1.30 | 0.124 $\pm$ 0.07 | 0.281 $\pm$ 0.18 |
| RL-20 | 13.558 $\pm$ 1.08 | 0.119 $\pm$ 0.06 | 0.260 $\pm$ 0.16 |
| RL-40 | 13.506 $\pm$ 1.11 | 0.113 $\pm$ 0.06 | 0.301 $\pm$ 0.18 |
| GC-0 | 12.005 $\pm$ 2.14 | 0.163 $\pm$ 0.20 | 0.461 $\pm$ 0.28 |
| GC-20 | 12.162 $\pm$ 2.47 | 0.161 $\pm$ 0.25 | 0.485 $\pm$ 0.31 |
| GC-40 | 11.834 $\pm$ 2.74 | 0.216 $\pm$ 0.33 | 0.510 $\pm$ 0.30 |

**说明**: TABLE VI: Controller Performance Domain Randomization. Comparison of RL and GC controllers optimized for Lissajous trajectory tracking, with feed-forward terms, varied amounts of domain randomization (0-40%) of mass, inertia, and thrust-to-weight. Results are shown as averages with standard deviation 1000 trials in the 20% domain randomization setting for evaluation.

#### Table 7: TABLE VII: Controller Performance Realistic Dynamics. Comparison of RL and GC controllers optimized for Lissajous

| Controller | Avg. Reward | Position RMSE (m) | Yaw RMSE (rad) |
| --- | --- | --- | --- |
| RL-Simple | 2.941 $\pm$ 1.69 | 0.856 $\pm$ 0.20 | 1.695 $\pm$ 0.17 |
| RL-Realistic | 13.053 $\pm$ 1.17 | 0.164 $\pm$ 0.06 | 0.236 $\pm$ 0.18 |
| GC-Simple | 12.709 $\pm$ 0.72 | 0.234 $\pm$ 0.12 | 0.595 $\pm$ 0.18 |
| GC-Realistic | 12.595 $\pm$ 1.99 | 0.183 $\pm$ 0.28 | 0.406 $\pm$ 0.31 |

**说明**: TABLE VII: Controller Performance Realistic Dynamics. Comparison of RL and GC controllers optimized for Lissajous trajectory tracking, with feed-forward terms, Simple dynamics (rigid body dynamics only) or Realistic dynamics (motor dynamics and saturation). Results are shown as averages with standard deviation 1000 trials in the realistic dynamics setting for evaluation.

#### Table 8: TABLE VIII: Randomization Ranges for Tasks. Ranges used for both optimization and evaluation in Hover and Lissajous Tra

| | | Lissajous Parameters | | | |
| --- | --- | --- | --- | --- | --- |
| Task | State | A A A | $\omega$ |  \phi | $\delta$ |
| Hover | x x x | [0.0, 0.0] | [-2.0, 2.0] | $\pi, π π \pi ]$ | [-2.0, 2.0] |
| y y y | [0.0, 0.0] | [-2.0, 2.0] | $\pi, π π \pi ]$ | [-2.0, 2.0] | |
| z z z | [0.0, 0.0] | [-2.0, 2.0] | $\pi, π π \pi ]$ | [-2.0, 2.0] | |
|   \psi | [0.0, 0.0] | [-2.0, 2.0] | $\pi, π π \pi ]$ | $\pi, π π \pi ]$ | |
| Lissajous Tracking | x x x | [-2.0, 2.0] | [-3.0, 3.0] | $\pi, π π \pi ]$ | [-2.0, 2.0] |
| y y y | [-2.0, 2.0] | [-3.0, 3.0] | $\pi, π π \pi ]$ | [-2.0, 2.0] | |
| z z z | [-2.0, 2.0] | [-3.0, 3.0] | $\pi, π π \pi ]$ | [-2.0, 2.0] | |
|   \psi | [-2.0, 2.0] | [-2.0, 2.0] | $\pi, π π \pi ]$ | $\pi, π π \pi ]$ | |

**说明**: TABLE VIII: Randomization Ranges for Tasks. Ranges used for both optimization and evaluation in Hover and Lissajous Tracking tasks, listed in terms of randomized Lissajous parameters.

#### Table 9: TABLE IX: Achieved rewards, position RMSE, and yaw RMSE for controller configurations tested in Lissajous Tracking and

| | Lissajous Tracking | Hover | | | | |
| --- | --- | --- | --- | --- | --- | --- |
| Model | Reward | Position RMSE | Yaw RMSE | Reward | Position RMSE | Yaw RMSE |
| RL-Opt-Liss.-FF | 13.621 $\pm$ 0.816 | 0.117 $\pm$ 0.051 | 0.487 $\pm$ 0.265 | 12.409 $\pm$ 1.214 | 0.423 $\pm$ 0.178 | 0.606 $\pm$ 0.255 |
| GC-Man-Lissajous-FF | 13.494 $\pm$ 1.477 | 0.167 $\pm$ 0.144 | 0.44 $\pm$ 0.283 | | | |
| GC-Opt-Lissajous-FF | 13.792 $\pm$ 1.278 | 0.136 $\pm$ 0.099 | 0.404 $\pm$ 0.294 | 13.474 $\pm$ 0.692 | 0.372 $\pm$ 0.121 | 0.451 $\pm$ 0.21 |
| RL-Opt-Hover-FF | 13.494 $\pm$ 0.863 | 0.172 $\pm$ 0.079 | 0.659 $\pm$ 0.353 | 13.918 $\pm$ 0.335 | 0.327 $\pm$ 0.104 | 0.509 $\pm$ 0.285 |
| GC-Opt-Hover-FF | 13.695 $\pm$ 1.261 | 0.153 $\pm$ 0.118 | 0.416 $\pm$ 0.292 | 13.72 $\pm$ 0.477 | 0.383 $\pm$ 0.117 | 0.333 $\pm$ 0.172 |
| RL-Opt-Lissajous-None | 11.559 $\pm$ 2.172 | 0.406 $\pm$ 0.191 | 0.354 $\pm$ 0.142 | | | |
| GC-Opt-Lissajous-None | 11.937 $\pm$ 2.19 | 0.318 $\pm$ 0.29 | 0.629 $\pm$ 0.343 | | | |
| GC-Opt-Lissajous-PID | 11.524 $\pm$ 2.458 | 0.35 $\pm$ 0.235 | 0.627 $\pm$ 0.385 | | | |
| GC-Man-Hover-FF | | | | 12.953 $\pm$ 1.022 | 0.41 $\pm$ 0.153 | 0.501 $\pm$ 0.193 |
| RL-Opt-Hover-None | | | | 13.918 $\pm$ 0.335 | 0.327 $\pm$ 0.104 | 0.509 $\pm$ 0.285 |
| GC-Opt-Hover-None | | | | 13.756 $\pm$ 0.449 | 0.355 $\pm$ 0.116 | 0.421 $\pm$ 0.139 |
| GC-Opt-Hover-PID | | | | 9.955 $\pm$ 3.009 | 0.631 $\pm$ 0.351 | 0.854 $\pm$ 0.265 |

**说明**: TABLE IX: Achieved rewards, position RMSE, and yaw RMSE for controller configurations tested in Lissajous Tracking and Hover tasks. Results are shown as mean with standard deviation 1000 trials per task.
## 实验解读

- 评价重点:围绕 aerial-robotics、agile-locomotion、motion-tracking、强化学习、鲁棒控制,优先看真实机器人结果、扰动恢复、接触稳定性、控制误差、成功率、实时性和 sim-to-real 差距。
- System0 视角:实验结论不只看最终成功率,还要看闭环过程中是否能处理状态估计误差、执行延迟、碰撞/接触突变和外部扰动。
- 与上层系统的关系:如果方法能在快闭环内稳定工作,它可以作为 System1 技能或 System2 规划的底层执行保障;如果只在离线或慢速设定下成立,则还不能直接作为身体级快闭环模块。
## 批判性思考

### 优点

- 论文把 aerial-robotics、agile-locomotion、motion-tracking、强化学习、鲁棒控制 放到机器人身体级闭环里讨论,和只做高层语义推理的工作相比,更接近真实机器人部署时会遇到的状态、动力学和安全约束。
- 若论文同时给出公式、系统结构图、硬件实验或消融结果,这些证据可以帮助判断方法是否真的能支撑毫秒到百毫秒级反应。

### 局限

- 需要重点核查控制频率、推理延迟、传感延迟、失败恢复、硬件磨损、未见场景泛化和安全边界;这些因素决定它能否成为可靠的 System0 模块。
- 对学习型方法,还要区分训练环境中的鲁棒性和真实部署中的鲁棒性,尤其是接触丰富任务、动态运动任务和人形机器人任务。
## 论文定位

- 研究对象:Leveling the Playing Field: Carefully Comparing Classical and Learned Controllers for Quadrotor Trajectory Tracking。
- 关键词:aerial-robotics、agile-locomotion、motion-tracking、强化学习、鲁棒控制。
- System0 关联:关注机器人身体级快反应链路中“状态/接触/局部感知 -> 约束/优化/策略 -> 动作”的短闭环,不把问题停留在语言规划或任务分解层。

## 速查卡片

> [!summary] Leveling the Playing Field
> - **论文**: https://www.roboticsproceedings.org/rss21/p116.pdf
> - **arXiv**: http://arxiv.org/abs/2506.17832v1
> - **arXiv HTML**: https://arxiv.org/html/2506.17832v1
