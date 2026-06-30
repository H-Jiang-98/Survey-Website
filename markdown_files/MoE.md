---
title: "Toward Reliable Sim-to-Real Predictability for MoE-based Robust Quadrupedal Locomotion"
method_name: "Reliable Sim MoE"
authors: ["Tianyang Wu"]
year: 2026
venue: "RSS"
tags: ["robust-control", "legged-locomotion", "reinforcement-learning", "robot-generalization", "sim-to-real", "agile-locomotion", "state-estimation"]
image_source: "online"
arxiv_html: "https://arxiv.org/html/2602.00678v4"
---
# Reliable Sim MoE
## 一句话总结

> Toward Reliable Sim-to-Real Predictability for MoE-based Robust Quadrupedal Locomotion 主要落在 [[agile-locomotion]]、[[足式运动]]、[[proprioception]]、[[quadruped]] 这一类问题上;从 System0 视角看,它的价值在于把身体状态、接触变化、局部感知或动力学约束接到机器人低层闭环里,让动作生成更稳定、安全、实时。
## 核心贡献

1. 围绕 **Toward Reliable Sim-to-Real Predictability for MoE-based Robust Quadrupedal Locomotion** 建立了一个与 agile-locomotion、足式运动、proprioception、quadruped、强化学习、robot-generalization 相关的机器人问题设定,重点不是高层语义规划,而是身体级闭环中的状态变化、接触变化或控制约束。
2. 方法层面的关键价值在于把感知、动力学、约束优化、策略学习或硬件反馈组织成可执行的低层动作生成机制,适合放进 System0 的快反应模块中理解。
3. 对系统集成而言,这类工作补上了 System1 技能选择与电机级控制之间的一层:它负责把上层目标变成动态可行、可恢复、可安全过滤的动作。
4. 若论文包含真实机器人、接触任务或仿真到真实迁移,核心应看它是否报告了闭环稳定性、扰动恢复、硬件延迟、失败模式和跨场景泛化。
## 问题背景

这篇论文可以放在 System0 的“具身快系统”背景下阅读:机器人需要在很短时间内根据本体状态、接触、力/触觉、局部视觉或动力学预测做出动作,而不是等待慢速规划器重新推理整个任务。agile-locomotion、足式运动、proprioception、quadruped、强化学习、robot-generalization、鲁棒控制 相关问题的难点通常不在“知道要做什么”,而在动作执行过程中状态会快速偏离、接触会突然改变、约束会实时收紧,低层系统必须及时修正。

### System0 关联

精读时优先看三件事:第一,论文使用哪些快速反馈量,例如关节状态、接触状态、力觉、触觉、深度或局部视觉;第二,反馈如何进入 MPC、WBC、策略网络、安全过滤器或恢复控制器;第三,输出是否直接影响机器人下一步动作,而不是只服务离线分析或高层任务规划。
## 方法详解

### System0 视角解析

从 System0 角度看,这篇论文的方法部分应当围绕 agile-locomotion、足式运动、proprioception、quadruped、强化学习、robot-generalization、鲁棒控制、仿真到真实迁移 来读。核心问题是:系统如何从快速变化的机器人状态中构造控制输入,如何处理接触、碰撞、约束、模型误差或执行延迟,以及如何把这些信息变成下一时刻的动作。

如果方法里有优化问题,重点看目标函数、约束和求解频率;如果方法里有学习策略,重点看观测空间、动作空间、训练扰动和部署时的闭环频率;如果方法里有硬件或传感器,重点看传感链路是否真的进入控制回路。这样读可以把论文和 System2 语言规划、System1 技能编排区分开来。

### 关键公式与机制

#### 公式 1: [[动力学或策略机制]]

$$J(\pi)=\mathbb{E}_{\boldsymbol{s}_{0}\sim\rho_{0},\tau\sim\pi}\left[\sum_{t=0}^{\infty}\gamma^{t}R(\boldsymbol{s}_{t},\boldsymbol{a}_{t},\boldsymbol{s}_{t+1})\right]$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 2: [[优化目标/约束]]

$$
\mathcal{L}_{\text{load balance}}=\sum_{k=1}^{K}\left(\bar{\omega}_{k}-\frac{1}{K}\right)^{2},\quad\bar{\omega}_{k}=\frac{1}{B}\sum_{j=1}^{B}\omega_{k}^{(j)}
$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 3: [[动力学或策略机制]]

$$r^{\text{hs}}=\frac{| v_{x}^{ $\text{cmd}}$ | }{ $\lVert \boldsymbol{v}^{\text{cmd}} \rVert_{2}}\cdot\left($ | q_{ $\text{FL}}^{\text{hip}}+q_{\text{FR}}^{\text{hip}}$ |+| q_{ $\text{RL}}^{\text{hip}}+q_{\text{RR}}^{\text{hip}}$ |\right)$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 4: [[动力学或策略机制]]

$$Q_{i,j}(L)=\left(\prod_{k=1}^{K}m_{k}^{w_{k}}\right)^{1/\sum_{k=1}^{K}w_{k}}$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 5: [[动力学或策略机制]]

$$\begin{cases}\boldsymbol{M}_{y}=-x_{\text{zmp}}\boldsymbol{F}_{z}+z_{\text{zmp}}\boldsymbol{F}_{x}\\ \boldsymbol{M}_{x}=y_{\text{zmp}}\boldsymbol{F}_{z}-z_{\text{zmp}}\boldsymbol{F}_{y}\end{cases}$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 6: [[优化目标/约束]]

$$m_{\text{zmp margin}}=\max\left(0,1-\frac{|| (x_{ $\text{zmp}},y_{\text{zmp}})$ ||_{2}}{D_{\text{norm}}}\right)$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 7: [[优化目标/约束]]

$$m_{\text{friction margin}}=\sum_{i=1}^{N_{c}}w_{i}\max\left(0,1-\frac{|| f_{i}^{ $\text{tangent}}$ ||}{\mu f_{i}^{\text{normal}}}\right)$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 8: [[优化目标/约束]]

$$v^{*}:=\text{clip}\left(\frac{5-|| $\sum_{i=1}^{n_{r}}\boldsymbol{v}_{i}^{\text{cmd}}$ || _{2}T_{r}}{T_{ $\text{ep}}-n_{r}T_{r}},0,\min($ | v^{ $\text{min}}$ |,| v^{ $\text{max}}$ |)\right)$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 9: [[优化目标/约束]]

$$T^{zero}=\text{clip}\left(T_{\text{ep}}-n_{r}T_{r}-\frac{5-|| $\sum_{i=1}^{n_{r}}\boldsymbol{v}_{i}^{\text{cmd}}$ ||_{2}T_{r}}{0.8\times\max(v^{\text{max}}_{x},v^{\text{max}}_{y})},0,T_{r}\right)$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 10: [[动力学或策略机制]]

$$\ \sum_{i=1}^{N}\left[(\boldsymbol{p}_{i}\times m_{i}(\boldsymbol{g}-\boldsymbol{\ddot{p}}_{i}))-(\boldsymbol{I}_{i}\boldsymbol{\dot{\omega}}_{i}+\boldsymbol{\omega}_{i}\times(\boldsymbol{I}_{i}\boldsymbol{\omega}_{i}))\right]$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。
## 关键图表

### Figure 1: Pipeline / core system figure: Comparative analysis against one-stage proprioceptive methods including CTS, HIM, and

![Figure 1](https://arxiv.org/html/2602.00678v4/x1.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Pipeline / core system figure: Comparative analysis against one-stage proprioceptive methods including CTS, HIM, and”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 2: The RoboGauge evaluation architecture consists of three hierarchical stages. (A) Ba

![Figure 2](https://arxiv.org/html/2602.00678v4/x2.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“The RoboGauge evaluation architecture consists of three hierarchical stages. (A) Ba”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 3: The top panel shows the robot quickly adjusting its posture to safely descend when t

![Figure 3](https://arxiv.org/html/2602.00678v4/figures/real/unexpected_recovery_real.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“The top panel shows the robot quickly adjusting its posture to safely descend when t”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### 论文表格

#### Table 1: TABLE I: Domain Randomization Specifications

| Randomization Term | Range | Unit |
| --- | --- | --- |
| Friction | [0.5, 1.5 ] [0.5,1.5] | – |
| Payload mass | [- 1, 1 ] [-1,1] | kg |
| Link mass | . . Nominal Value . . \ $\text{Nominal Value}$ | kg |
| Base center of mass | [- 3, 3 ]  [- 3, 3 ]  [- 3, 3 ] [-3,3]\times[-3,3]\times[-3,3] | cm |
| Restitution | [0.0, 0.5 ] [0.0,0.5] | – |
| Proportional gain k p k_{ $\text{p}}$ | . . Nominal Value . . \ $\text{Nominal Value}$ | Nm / rad  $\text{Nm}/\text{rad}$ |
| Derivative gain k d k_{ $\text{d}}$ | . . Nominal Value . . \ $\text{Nominal Value}$ | $\text{Nm}\cdot\text{s}/\text{rad}$ |
| Actuator strength | . . Nominal Value . . \ $\text{Nominal Value}$ | – |
| Actuator offset | [- 0.035, 0.035 ] [-0.035,0.035] | rad |
| Control latency | [0, 20 ] [0,20] | ms |

**说明**: TABLE I: Domain Randomization Specifications

#### Table 2: TABLE II: Metrics for the RoboGauge Framework

| Metric | Description |
| --- | --- |
| Lin. Velocity Error | Linear velocity l 2 \ell_{2} tracking error |
| Ang. Velocity Error | Angular velocity l 2 \ell_{2} tracking error |
| Dof Power | Motor power consumption |
| Dof Limits | Joint angles exceeding soft limits |
| Orientation Stability | Gravity projection on the lateral (y y) axis |
| Torque Smoothness | Temporal smoothness of motor torques |
| ZMP Margin | Normalized Zero Moment Point deviation |
| Friction Margin | Normal-force-weighted Coulomb friction |

**说明**: TABLE II: Metrics for the RoboGauge Framework

#### Table 3: TABLE III: Metrics Error Comparison

| Env. | Cmd. | Tracking ↓ \downarrow | Safety ↓ \downarrow | Quality ↓ \downarrow |
| --- | --- | --- | --- | --- |
| MuJoCo (Ours) | Longitudinal | 0.0573 | 0.0253 | 0.0246 |
| Lateral | 0.0541 | 0.0049 | 0.0079 | |
| Angular | 0.0560 | 0.0050 | 0.0035 | |
| Average | 0.0558 | 0.0117 | 0.0120 | |
| IsaacGym (Training) | Longitudinal | 0.1365 | 0.0844 | 0.0678 |
| Lateral | 0.0572 | 0.0052 | 0.0125 | |
| Angular | 0.0713 | 0.0103 | 0.0337 | |
| Average | 0.0883 | 0.0333 | 0.0380 | |

**说明**: TABLE III: Metrics Error Comparison

#### Table 4: TABLE IV: RoboGauge results for baselines

| Model | Score | Tracking ↑ \uparrow | Safety ↑ \uparrow | Quality ↑ \uparrow | Level |
| --- | --- | --- | --- | --- | --- |
| Ours | 0.6713 | 0.6669 | 0.7857 | 0.7392 | 7.85 |
| CTS | 0.5786 | 0.5755 | 0.7066 | 0.6624 | 6.83 |
| HIM | 0.5379 | 0.5453 | 0.6476 | 0.6050 | 6.19 |
| DreamWaQ | 0.5054 | 0.5105 | 0.6149 | 0.5730 | 5.74 |

**说明**: TABLE IV: RoboGauge results for baselines

#### Table 5: TABLE V: RoboGauge Results for MoE Ablation

| Model | Score | Tracking | Safety | Quality | Level |
| --- | --- | --- | --- | --- | --- |
| MoE (Ours) | 0.6713 | 0.6669 | 0.7857 | 0.7392 | 7.85 |
| AC-MoE [14 ] | 0.6509 | 0.6442 | 0.7644 | 0.7149 | 7.52 |
| MoE-NG | 0.6519 | 0.6447 | 0.7639 | 0.7186 | 7.56 |
| MCP [40 ] | 0.6399 | 0.6355 | 0.7542 | 0.7058 | 7.41 |

**说明**: TABLE V: RoboGauge Results for MoE Ablation

#### Table 6: TABLE VI: Real-World Survival Rate Comparison

| Model | Survival Rate (%) ↑ \uparrow | | |
| --- | --- | --- | --- |
| Lat. Impulse (80–100 N) | $\mu=0.38)$ | Obstacle cm . $\mu=0.85)$ | |
| Ours | 18/20 | 85/85 | 17/20 |
| Built-in RL | 5/20 | 85/85 | 0/20 |
| CTS | 11/20 | 18/85 | 0/20 |
| HIM | 8/20 | 24/85 | 0/20 |
| DreamWaQ | 7/20 | 12/85 | 0/20 |

**说明**: TABLE VI: Real-World Survival Rate Comparison

#### Table 7: TABLE IX: Reward Function Specifications

| Reward Term | Equation | Weight |
| --- | --- | --- |
| Lin. velocity tracking | $\exp(-\sigma\lVert \rVert\boldsymbol{v}_{xy}^{\text{cmd}}-\boldsymbol{v}_{xy}\lVert \rVert_{2}^{2})$ | 1.0 / 2.0 1.0/{2.0} |
| Ang. velocity tracking | $\exp(-\sigma\lVert \omega_{z}^{\text{cmd}}-\omega_{z} \rVert^{2})$ | 0.5 0.5 |
| Lin. velocity (z z) | v z 2 v_{z}^{2} | - 2.0 -2.0 |
| Ang. velocity (x  y xy) | $\boldsymbol{\omega}_{xy}\lVert \rVert_{2}^{2}$ | - 0.05 -0.05 |
| Joint acceleration | q  ̈ 2 \ddot{q}^{2} | - 2.5  10 - 7 -2.5\times 10^{-7} |
| Joint power | $\boldsymbol{\tau}\lVert \rVert\dot{q}\lVert^{T}$ | - 2  10 - 5 -2\times 10^{-5} |
| Joint torque | $\boldsymbol{\tau}\lVert \rVert_{2}^{2}$ | - 1  10 - 4 -1\times 10^{-4} |
| Base height | (h des - h) 2 (h^{ $\text{des}}-h)^{2}$ | - 1.0 -1.0 |
| Action rate | $\boldsymbol{a}_{t}-\boldsymbol{a}_{t-1}\lVert \rVert_{2}^{2}$ | - 0.01 -0.01 |
| Action smoothness | $\boldsymbol{a}_{t}-2\boldsymbol{a}_{t-1}+\boldsymbol{a}_{t-2}\lVert \rVert_{2}^{2}$ | - 0.01 -0.01 |
| Collision | n collision n_{ $\text{collision}}$ | - 1.0 -1.0 |
| Joint limit | n limitation n_{ $\text{limitation}}$ | - 2.0 -2.0 |
| Foot regulation | r fr r^{ $\text{fr}}$ | - 0.05 -0.05 |
| Hip regulation | \| q hip - q default hip \| \| $\boldsymbol{q}^{\text{hip}}-\boldsymbol{q}_{\text{default}}^{\text{hip}}\lVert$ | - 0.05 -0.05 |
| Hip symmetry | r hs r^{ $\text{hs}}$ | - 1 -1 |

**说明**: TABLE IX: Reward Function Specifications

#### Table 8: TABLE X: Maximum Velocity Tracking Coefficients and Command Limits Across Terrains

| Terrain Type | $\text{max}}^{i}$ | v x v_{x} [m/s] | v y v_{y} [m/s] | $\omega_{z} [rad/s]$ |
| --- | --- | --- | --- | --- |
| Flat | 1/4 | ± \pm 2.0 | ± \pm 1.0 | ± \pm 2.0 |
| Wave | 5/12 | ± \pm 1.5 | ± \pm 1.0 | ± \pm 1.5 |
| Slope | 1/4 | ± \pm 1.5 | ± \pm 1.0 | ± \pm 1.5 |
| Stairs Up | 1/2 | ± \pm 1.0 | ± \pm 1.0 | ± \pm 1.5 |
| Stairs Down | 1/2 | ± \pm 1.0 | ± \pm 1.0 | ± \pm 1.5 |
| Obstacle | 3/4 | ± \pm 1.0 | ± \pm 1.0 | ± \pm 1.5 |

**说明**: TABLE X: Maximum Velocity Tracking Coefficients and Command Limits Across Terrains

#### Table 9: TABLE XI: Command Curriculum Stages and Velocity Limits

| Stage | Training Steps | v x v_{x} | v y v_{y} | $\omega_{z}$ |
| --- | --- | --- | --- | --- |
| | | [m/s] | [m/s] | [rad/s] |
| Initial | [0, 2  10 4 ] [0,2\times 10^{4}] | ± \pm 0.5 | ± \pm 0.5 | ± \pm 1.0 |
| Intermediate | [2  10 4, 5  10 4 ] [2\times 10^{4},5\times 10^{4}] | ± \pm 1.0 | ± \pm 1.0 | ± \pm 1.5 |
| Advanced | $\infty]$ | ± \pm 2.0 | ± \pm 1.0 | ± \pm 2.0 |

**说明**: TABLE XI: Command Curriculum Stages and Velocity Limits

#### Table 10: TABLE XII: Comprehensive Evaluation: Real-World Measurements, Predicted Values, and Absolute Errors

| Source | Movement | Lin. Trk. | Ang. Trk. | DOF Power | DOF Limits | Orient. | Smooth. |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Real (Ground Truth) | Linear (x = 1 x=1) | 0.9185 | 0.5808 | 0.8527 | 0.9159 | 0.9675 | 0.7739 |
| Lateral (y = 0.5 y=0.5) | 0.9552 | 0.7037 | 0.9696 | 0.9384 | 0.9661 | 0.8985 | |
| Angular (z = 1 z=1) | 0.9552 | 0.8010 | 0.9659 | 0.9439 | 0.9614 | 0.9020 | |
| Stairs (x = 1 x=1) | 0.8554 | 0.2732 | 0.6721 | 0.8395 | 0.8749 | 0.5944 | |
| RoboGauge (Predicted) | Linear (x = 1 x=1) | 0.8217 | 0.5669 | 0.8330 | 0.9386 | 0.9592 | 0.7853 |
| Lateral (y = 0.5 y=0.5) | 0.8685 | 0.6822 | 0.9704 | 0.9293 | 0.9627 | 0.8861 | |
| Angular (z = 1 z=1) | 0.8763 | 0.7679 | 0.9734 | 0.9414 | 0.9647 | 0.8983 | |
| Stairs (x = 1 x=1) | 0.7596 | 0.2507 | 0.6211 | 0.8472 | 0.8625 | 0.6606 | |
| RoboGauge (Ours) (Abs. Error) ↓ \downarrow | x = 1 x=1 (Merge) | 0.0963 | 0.0182 | 0.0354 | 0.0152 | 0.0103 | 0.0388 |
| Average | 0.0873 | 0.0243 | 0.0145 | 0.0089 | 0.0057 | 0.0183 | |
| IsaacGym (Predicted) | Linear (x = 1 x=1) | 0.8977 | 0.7826 | 0.9155 | 0.9361 | 0.9737 | 0.8289 |
| Lateral (y = 0.5 y=0.5) | 0.9694 | 0.8039 | 0.9598 | 0.9378 | 0.9707 | 0.8781 | |
| Angular (z = 1 z=1) | 0.9853 | 0.9134 | 0.9751 | 0.9325 | 0.9798 | 0.9510 | |
| Stairs (x = 1 x=1) | 0.8786 | 0.5732 | 0.8635 | 0.9027 | 0.9339 | 0.7454 | |
| IsaacGym (Abs. Error) ↓ \downarrow | x = 1 x=1 (Merge) | 0.0220 | 0.2509 | 0.1271 | 0.0417 | 0.0326 | 0.1030 |
| Average | 0.0221 | 0.1545 | 0.0487 | 0.0179 | 0.0185 | 0.0575 | |

**说明**: TABLE XII: Comprehensive Evaluation: Real-World Measurements, Predicted Values, and Absolute Errors

#### Table 11: TABLE XIII: RoboGauge detailed metrics for baselines

| Model | ang vel err | lin vel err | dof limits | dof power | | | | | | | | |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mean | mean@25 | mean@50 | mean | mean@25 | mean@50 | mean | mean@25 | mean@50 | mean | mean@25 | mean@50 | |
| Our | 0.7018 | 0.6047 | 0.6431 | 0.7394 | 0.6497 | 0.6908 | 0.8139 | 0.8029 | 0.8073 | 0.7889 | 0.7411 | 0.7642 |
| CTS | 0.6231 | 0.5253 | 0.5632 | 0.6464 | 0.5363 | 0.5878 | 0.7341 | 0.7232 | 0.7275 | 0.7113 | 0.6607 | 0.6856 |
| HIM | 0.5652 | 0.4620 | 0.5025 | 0.6297 | 0.5443 | 0.5881 | 0.6781 | 0.6645 | 0.6699 | 0.6529 | 0.5996 | 0.6253 |
| DreamWaQ | 0.5309 | 0.4305 | 0.4698 | 0.5937 | 0.5060 | 0.5512 | 0.6437 | 0.6307 | 0.6360 | 0.6200 | 0.5690 | 0.5939 |
| Model | orientation stability | torque smoothness | zmp margin | friction margin | | | | | | | | |
| mean | mean@25 | mean@50 | mean | mean@25 | mean@50 | mean | mean@25 | mean@50 | mean | mean@25 | mean@50 | |
| Our | 0.8147 | 0.7946 | 0.8040 | 0.7734 | 0.7400 | 0.7535 | 0.7933 | 0.7401 | 0.7653 | 0.6892 | 0.6051 | 0.6340 |
| CTS | 0.7346 | 0.7124 | 0.7232 | 0.6954 | 0.6594 | 0.6744 | 0.7163 | 0.6670 | 0.6900 | 0.6188 | 0.5310 | 0.5622 |
| HIM | 0.6780 | 0.6497 | 0.6643 | 0.6416 | 0.6053 | 0.6201 | 0.6604 | 0.6092 | 0.6344 | 0.5544 | 0.4706 | 0.5014 |
| DreamWaQ | 0.6443 | 0.6170 | 0.6312 | 0.6060 | 0.5709 | 0.5853 | 0.6290 | 0.5814 | 0.6049 | 0.5236 | 0.4392 | 0.4706 |

**说明**: TABLE XIII: RoboGauge detailed metrics for baselines

#### Table 12: TABLE XIV: RoboGauge detailed terrain scores for baselines

| Model | flat | wave | slope forward | slope backward | | | | | | | | |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mean | mean@25 | mean@50 | mean | mean@25 | mean@50 | mean | mean@25 | mean@50 | mean | mean@25 | mean@50 | |
| Our | 0.755 ± 0.005  $\mathbf{0.755\pm 0.005}$ | 0.559 ± 0.013  $\mathbf{0.559\pm 0.013}$ | 0.663 ± 0.009  $\mathbf{0.663\pm 0.009}$ | 0.615 ± 0.042  $\mathbf{0.615\pm 0.042}$ | 0.542 ± 0.042  $\mathbf{0.542\pm 0.042}$ | 0.579 ± 0.042  $\mathbf{0.579\pm 0.042}$ | 0.583 ± 0.104  $\mathbf{0.583\pm 0.104}$ | 0.519 ± 0.094  $\mathbf{0.519\pm 0.094}$ | 0.551 ± 0.099  $\mathbf{0.551\pm 0.099}$ | 0.584 ± 0.105 0.584\pm 0.105 | 0.522 ± 0.094 0.522\pm 0.094 | 0.553 ± 0.100 0.553\pm 0.100 |
| CTS | 0.721 ± 0.004 0.721\pm 0.004 | 0.436 ± 0.008 0.436\pm 0.008 | 0.592 ± 0.006 0.592\pm 0.006 | 0.602 ± 0.056 0.602\pm 0.056 | 0.525 ± 0.055 0.525\pm 0.055 | 0.561 ± 0.055 0.561\pm 0.055 | 0.557 ± 0.104 0.557\pm 0.104 | 0.488 ± 0.094 0.488\pm 0.094 | 0.521 ± 0.098 0.521\pm 0.098 | 0.524 ± 0.101 0.524\pm 0.101 | 0.458 ± 0.090 0.458\pm 0.090 | 0.491 ± 0.096 0.491\pm 0.096 |
| HIM | 0.739 ± 0.003 0.739\pm 0.003 | 0.486 ± 0.009 0.486\pm 0.009 | 0.621 ± 0.006 0.621\pm 0.006 | 0.569 ± 0.030 0.569\pm 0.030 | 0.490 ± 0.028 0.490\pm 0.028 | 0.525 ± 0.029 0.525\pm 0.029 | 0.492 ± 0.090 0.492\pm 0.090 | 0.425 ± 0.082 0.425\pm 0.082 | 0.457 ± 0.085 0.457\pm 0.085 | 0.571 ± 0.102 0.571\pm 0.102 | 0.504 ± 0.092 0.504\pm 0.092 | 0.537 ± 0.097 0.537\pm 0.097 |
| DreamWaQ | 0.736 ± 0.003 0.736\pm 0.003 | 0.485 ± 0.010 0.485\pm 0.010 | 0.620 ± 0.006 0.620\pm 0.006 | 0.532 ± 0.042 0.532\pm 0.042 | 0.458 ± 0.041 0.458\pm 0.041 | 0.491 ± 0.041 0.491\pm 0.041 | 0.404 ± 0.083 0.404\pm 0.083 | 0.343 ± 0.069 0.343\pm 0.069 | 0.373 ± 0.076 0.373\pm 0.076 | 0.595 ± 0.105  $\mathbf{0.595\pm 0.105}$ | 0.533 ± 0.097  $\mathbf{0.533\pm 0.097}$ | 0.564 ± 0.102  $\mathbf{0.564\pm 0.102}$ |
| Model | stairs forward | stairs backward | obstacle | | | | | | | | | |
| mean | mean@25 | mean@50 | mean | mean@25 | mean@50 | mean | mean@25 | mean@50 | | | | |
| Our | 0.826 ± 0.024  $\mathbf{0.826\pm 0.024}$ | 0.743 ± 0.021  $\mathbf{0.743\pm 0.021}$ | 0.775 ± 0.024  $\mathbf{0.775\pm 0.024}$ | 0.791 ± 0.039  $\mathbf{0.791\pm 0.039}$ | 0.710 ± 0.038  $\mathbf{0.710\pm 0.038}$ | 0.744 ± 0.039  $\mathbf{0.744\pm 0.039}$ | 0.882 ± 0.004  $\mathbf{0.882\pm 0.004}$ | 0.799 ± 0.004  $\mathbf{0.799\pm 0.004}$ | 0.835 ± 0.004  $\mathbf{0.835\pm 0.004}$ | | | |
| CTS | 0.799 ± 0.051 0.799\pm 0.051 | 0.722 ± 0.050 0.722\pm 0.050 | 0.746 ± 0.050 0.746\pm 0.050 | 0.666 ± 0.056 0.666\pm 0.056 | 0.587 ± 0.053 0.587\pm 0.053 | 0.617 ± 0.055 0.617\pm 0.055 | 0.569 ± 0.008 0.569\pm 0.008 | 0.483 ± 0.007 0.483\pm 0.007 | 0.521 ± 0.008 0.521\pm 0.008 | | | |
| HIM | 0.560 ± 0.060 0.560\pm 0.060 | 0.486 ± 0.051 0.486\pm 0.051 | 0.514 ± 0.054 0.514\pm 0.054 | 0.735 ± 0.082 0.735\pm 0.082 | 0.663 ± 0.068 0.663\pm 0.068 | 0.693 ± 0.074 0.693\pm 0.074 | 0.460 ± 0.028 0.460\pm 0.028 | 0.385 ± 0.020 0.385\pm 0.020 | 0.418 ± 0.024 0.418\pm 0.024 | | | |
| DreamWaQ | 0.588 ± 0.086 0.588\pm 0.086 | 0.515 ± 0.075 0.515\pm 0.075 | 0.542 ± 0.079 0.542\pm 0.079 | 0.680 ± 0.056 0.680\pm 0.056 | 0.602 ± 0.054 0.602\pm 0.054 | 0.633 ± 0.055 0.633\pm 0.055 | 0.355 ± 0.020 0.355\pm 0.020 | 0.280 ± 0.015 0.280\pm 0.015 | 0.315 ± 0.017 0.315\pm 0.017 | | | |

**说明**: TABLE XIV: RoboGauge detailed terrain scores for baselines
## 实验解读

- 评价重点:围绕 agile-locomotion、足式运动、proprioception、quadruped、强化学习,优先看真实机器人结果、扰动恢复、接触稳定性、控制误差、成功率、实时性和 sim-to-real 差距。
- System0 视角:实验结论不只看最终成功率,还要看闭环过程中是否能处理状态估计误差、执行延迟、碰撞/接触突变和外部扰动。
- 与上层系统的关系:如果方法能在快闭环内稳定工作,它可以作为 System1 技能或 System2 规划的底层执行保障;如果只在离线或慢速设定下成立,则还不能直接作为身体级快闭环模块。
## 批判性思考

### 优点

- 论文把 agile-locomotion、足式运动、proprioception、quadruped、强化学习 放到机器人身体级闭环里讨论,和只做高层语义推理的工作相比,更接近真实机器人部署时会遇到的状态、动力学和安全约束。
- 若论文同时给出公式、系统结构图、硬件实验或消融结果,这些证据可以帮助判断方法是否真的能支撑毫秒到百毫秒级反应。

### 局限

- 需要重点核查控制频率、推理延迟、传感延迟、失败恢复、硬件磨损、未见场景泛化和安全边界;这些因素决定它能否成为可靠的 System0 模块。
- 对学习型方法,还要区分训练环境中的鲁棒性和真实部署中的鲁棒性,尤其是接触丰富任务、动态运动任务和人形机器人任务。
## 论文定位

- 研究对象:Toward Reliable Sim-to-Real Predictability for MoE-based Robust Quadrupedal Locomotion。
- 关键词:agile-locomotion、足式运动、proprioception、quadruped、强化学习、robot-generalization、鲁棒控制、仿真到真实迁移、terrain-adaptation。
- System0 关联:关注机器人身体级快反应链路中“状态/接触/局部感知 -> 约束/优化/策略 -> 动作”的短闭环,不把问题停留在语言规划或任务分解层。

## 速查卡片

> [!summary] Reliable Sim MoE
> - **论文**: https://www.roboticsproceedings.org/rss22/p156.pdf
> - **arXiv**: http://arxiv.org/abs/2602.00678v4
> - **arXiv HTML**: https://arxiv.org/html/2602.00678v4
