---
title: "Design and Control of a Bipedal Robotic Character"
method_name: "Design Control Bipedal"
authors: ["Ruben Grandia"]
year: 2024
venue: "RSS"
tags: ["robust-control", "real-time-control", "legged-locomotion", "reinforcement-learning", "robot-generalization"]
image_source: "online"
arxiv_html: "https://arxiv.org/html/2501.05204v1"
---
# Design Control Bipedal
## 一句话总结

> Design and Control of a Bipedal Robotic Character 主要落在 [[biped]]、[[inference-time-algorithm]]、[[足式运动]]、[[实时控制]] 这一类问题上;从 System0 视角看,它的价值在于把身体状态、接触变化、局部感知或动力学约束接到机器人低层闭环里,让动作生成更稳定、安全、实时。
## 核心贡献

1. 围绕 **Design and Control of a Bipedal Robotic Character** 建立了一个与 biped、inference-time-algorithm、足式运动、实时控制、强化学习、鲁棒控制 相关的机器人问题设定,重点不是高层语义规划,而是身体级闭环中的状态变化、接触变化或控制约束。
2. 方法层面的关键价值在于把感知、动力学、约束优化、策略学习或硬件反馈组织成可执行的低层动作生成机制,适合放进 System0 的快反应模块中理解。
3. 对系统集成而言,这类工作补上了 System1 技能选择与电机级控制之间的一层:它负责把上层目标变成动态可行、可恢复、可安全过滤的动作。
4. 若论文包含真实机器人、接触任务或仿真到真实迁移,核心应看它是否报告了闭环稳定性、扰动恢复、硬件延迟、失败模式和跨场景泛化。
## 问题背景

这篇论文可以放在 System0 的“具身快系统”背景下阅读:机器人需要在很短时间内根据本体状态、接触、力/触觉、局部视觉或动力学预测做出动作,而不是等待慢速规划器重新推理整个任务。biped、inference-time-algorithm、足式运动、实时控制、强化学习、鲁棒控制、terrain-adaptation 相关问题的难点通常不在“知道要做什么”,而在动作执行过程中状态会快速偏离、接触会突然改变、约束会实时收紧,低层系统必须及时修正。

### System0 关联

精读时优先看三件事:第一,论文使用哪些快速反馈量,例如关节状态、接触状态、力觉、触觉、深度或局部视觉;第二,反馈如何进入 MPC、WBC、策略网络、安全过滤器或恢复控制器;第三,输出是否直接影响机器人下一步动作,而不是只服务离线分析或高层任务规划。
## 方法详解

### System0 视角解析

从 System0 角度看,这篇论文的方法部分应当围绕 biped、inference-time-algorithm、足式运动、实时控制、强化学习、鲁棒控制、terrain-adaptation 来读。核心问题是:系统如何从快速变化的机器人状态中构造控制输入,如何处理接触、碰撞、约束、模型误差或执行延迟,以及如何把这些信息变成下一时刻的动作。

如果方法里有优化问题,重点看目标函数、约束和求解频率;如果方法里有学习策略,重点看观测空间、动作空间、训练扰动和部署时的闭环频率;如果方法里有硬件或传感器,重点看传感链路是否真的进入控制回路。这样读可以把论文和 System2 语言规划、System1 技能编排区分开来。

### 关键公式与机制

#### 公式 1: [[动力学或策略机制]]

$$\tau_{f}=\mu_{s}\tanh{\left(\dot{q}/\dot{q}_{s}\right)}+\mu_{d}\dot{q},$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 2: [[优化目标/约束]]

$$\boldsymbol{s}_{t}=(\boldsymbol{p}^{\mathcal{P}}_{t},{\boldsymbol{\theta}}^{\mathcal{P}}_{t},{\boldsymbol{v}}^{\mathcal{T}}_{t},{\boldsymbol{\omega}}^{\mathcal{T}}_{t},\boldsymbol{q}_{t},\dot{\boldsymbol{q}}_{t},\boldsymbol{a}_{t-1},\boldsymbol{a}_{t-2})$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 3: [[动力学或策略机制]]

$$\tau=\text{clamp}_{\left[\ {\tau}(\dot{q}),\ {\tau}(\dot{q})\right]}\left(\tau_{m}\right)-\tau_{f},$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 4: [[优化目标/约束]]

$$\hat{q}=\tilde{q}+0.5\cdot b\cdot\tanh{\left(\tau_{m}/\tau_{b}\right)}+\mathcal{N}\left(0,\sigma_{q}^{2}\right),$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 5: [[优化目标/约束]]

$$\boldsymbol{g}^{\text{peri}}_{t}=(\Delta h^{\text{head}}_{t},\Delta\boldsymbol{\theta}^{\text{head}}_{t},\boldsymbol{v}^{\mathcal{P}}_{t},\omega^{\mathcal{P}}_{t}).$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 6: [[优化目标/约束]]

$$(\boldsymbol{y}_{t},\boldsymbol{v}^{\mathcal{P}}_{t},\omega^{\mathcal{P}}_{t})=\mathcal{J}^{\text{peri}}\left(\boldsymbol{y}^{\text{bld}}_{t},\boldsymbol{u}_{t}\right),$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 7: [[动力学或策略机制]]

$$\boldsymbol{x}_{t}=(\boldsymbol{p}_{t},\boldsymbol{\theta}_{t},\boldsymbol{v}_{t},\boldsymbol{\omega}_{t},\boldsymbol{q}_{t},\dot{\boldsymbol{q}}_{t},c^{L}_{t},c^{R}_{t}),$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 8: [[优化目标/约束]]

$$\boldsymbol{y}_{t}=\mathcal{J}^{\text{perp}}\left(\boldsymbol{y}^{\text{bld}}_{t},\boldsymbol{u}_{t}\right),$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 9: [[动力学或策略机制]]

$$\boldsymbol{g}^{\text{perp}}_{t}=(\Delta h^{\text{head}}_{t},\Delta\boldsymbol{\theta}^{\text{head}}_{t},h^{\text{torso}}_{t},\boldsymbol{\theta}^{\text{torso}}_{t}).$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 10: [[优化目标/约束]]

$$\boldsymbol{c}_{t}=({\boldsymbol{p}}^{\mathcal{P}}_{t},{\boldsymbol{\theta}}^{\mathcal{P}}_{t},\boldsymbol{q}_{t})$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。
## 关键图表

### Figure 1: Pipeline / core system figure: Three instances of our robotic character performing an unscripted show. Apart from the

![Figure 1](https://arxiv.org/html/2501.05204v1/x1.jpeg)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Pipeline / core system figure: Three instances of our robotic character performing an unscripted show. Apart from the”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 2: Our character design and control pipeline consists of animation, mechatronic design, r

![Figure 2](https://arxiv.org/html/2501.05204v1/x2.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Our character design and control pipeline consists of animation, mechatronic design, r”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 3: Top) The operator uses the proposed puppeteering interface to act out a scene where

![Figure 3](https://arxiv.org/html/2501.05204v1/x17.jpg)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Top) The operator uses the proposed puppeteering interface to act out a scene where”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### 论文表格

#### Table 1: TABLE I: Weighted Reward Terms

| Name | Reward Term | Weight |
| --- | --- | --- |
| Imitation | | |
| Torso position xy | $\exp\left(-200.0\cdot\lVert \boldsymbol{p}_{x,y}-\hat{\boldsymbol{p}}_{x,y} \rVert^{2}\right)$ | 1.0 1.0 1.0 1.0 |
| Torso orientation | $\exp\left(-20.0\cdot\lVert \boldsymbol{\theta}\boxminus\hat{\boldsymbol{\theta}} \rVert^{2}\right)$ | 1.0 1.0 1.0 1.0 |
| Linear velocity xy | $\exp\left(-8.0\cdot\lVert \boldsymbol{v}_{x,y}-\hat{\boldsymbol{v}}_{x,y} \rVert^{2}\right)$ | 1.0 1.0 1.0 1.0 |
| Linear velocity z | $\exp\left(-8.0\cdot\left(v_{z}-\hat{v}_{z}\right)^{2}\right)$ | 1.0 1.0 1.0 1.0 |
| Angular velocity xy | $\exp\left(-2.0\cdot\lVert \boldsymbol{\omega}_{x,y}-\hat{\boldsymbol{\omega}}_{x,y} \rVert^{2}\right)$ | 0.5 0.5 0.5 0.5 |
| Angular velocity z | $\exp\left(-2.0\cdot\left(\omega_{z}-\hat{\omega}_{z}\right)^{2}\right)$ | 0.5 0.5 0.5 0.5 |
| Leg joint positions | $$ | 15.0 15.0 15.0 15.0 |
| Neck joint positions | $$ | 100.0 100.0 100.0 100.0 |
| Leg joint velocities | $$ | 1.0  10 - 3  1.0 10 3 1.0\cdot 10^{-3} 1.0  10 - 3 |
| Neck joint velocities | $$ | 1.0 1.0 1.0 1.0 |
| Contact | ∑ i ∈ {L, R } I [c i = c ^ i ] i L R I c i ^ c i  $\sum_{i\in\{L,R\}}\text{I}\left[c_{i}=\,\hat{c}_{i}\right] ∑ ∈ {, } I [= ]$ | 1.0 1.0 1.0 1.0 |
| Regularization | | |
| Joint torques | $$ | 1.0  10 - 3  1.0 10 3 1.0\cdot 10^{-3} 1.0  10 - 3 |
| Joint accelerations | $$ | 2.5  10 - 6  2.5 10 6 2.5\cdot 10^{-6} 2.5  10 - 6 |
| Leg action rate | $$ | 1.5 1.5 1.5 1.5 |
| Neck action rate | $$ | 5.0 5.0 5.0 5.0 |
| Leg action acc. | $$ | 0.45 0.45 0.45 0.45 |
| Neck action acc. | $$ | 5.0 5.0 5.0 5.0 |
| Survival | | |
| Survival | 1.0 1.0 1.0 1.0 | 20.0 20.0 20.0 20.0 |

**说明**: TABLE I: Weighted Reward Terms

#### Table 2: TABLE II: Show Function Parameters

| Function Parameters | Dimensionality | Units |
| --- | --- | --- |
| Antenna positions | 2  1 2 1 2\times 1 2  1 | [rad times absent radian  $\text{\,}\mathrm{rad} times ]$ |
| Eye colors | 2  3 2 3 2\times 3 2  3 | [RGB] |
| Eye radii | 2  1 2 1 2\times 1 2  1 | [% times absent percent  $\text{\,}\mathrm{\char 37\relax} times ]$ |
| Head lamp brightness | 1 | [% times absent percent  $\text{\,}\mathrm{\char 37\relax} times ]$ |

**说明**: TABLE II: Show Function Parameters

#### Table 3: TABLE III: Mean Absolute Tracking Error (MAE) of Joint Positions

| Type | Name | MAE [rad times absent radian  $\text{\,}\mathrm{rad} times ]$ |
| --- | --- | --- |
| Perpetual | Standing | 0.035 |
| Periodic | Walking | 0.123 |
| Episodic | Excited Motion | 0.029 |
| | Happy Dance | 0.027 |
| | Jump | 0.043 |
| | Tantrum | 0.032 |

**说明**: TABLE III: Mean Absolute Tracking Error (MAE) of Joint Positions

#### Table 4: TABLE IV: RL Hyperparameters

| Param. | Value |
| --- | --- |
| Num. iterations | 100 000 100000 100\,000 100 000 |
| $\text{envs.}\times\text{steps})$ | 8192  24 8192 24 8192\times 24 8192  24 |
| Num. mini-batches | 4 4 4 4 |
| Num. epochs | 5 5 5 5 |
| Clip range | 0.2 0.2 0.2 0.2 |
| Entropy coefficient | 0.0 0.0 0.0 0.0 |
| Discount factor | 0.99 0.99 0.99 0.99 |
| GAE discount factor | 0.95 0.95 0.95 0.95 |
| Desired KL-divergence | 0.01 0.01 0.01 0.01 |
| Max gradient norm | 1.0 1.0 1.0 1.0 |

**说明**: TABLE IV: RL Hyperparameters

#### Table 5: TABLE V: Disturbance Parameters

| Param. | | Short / small | Long / small | Short / large |
| --- | --- | --- | --- | --- |
| Body | | Hips, Feet | Pelvis, Head | Pelvis |
| Force [N times absent newton  $\text{\,}\mathrm{N} times ]$ | XY | [0.0, 5.0] | [0.0, 5.0] | [90.0, 150.0] |
| | Z | [0.0, 5.0] | [0.0, 5.0] | [0.0, 10.0] |
| Torque [N m times absent times newton meter  $\text{\,}\mathrm{N}\text{\,}\mathrm{m} times times ]$ | XY | [0.0, 0.25] | [0.0, 0.25] | [0.0, 15.0] |
| | Z | [0.0, 0.25] | [0.0, 0.25] | [0.0, 15.0] |
| Duration [s times absent second  $\text{\,}\mathrm{s} times ]$ | On | [0.25, 2.0] | [2.0, 10.0] | [0.1, 0.1] |
| | Off | [1.0, 3.0] | [1.0, 3.0] | [12.0, 15.0] |

**说明**: TABLE V: Disturbance Parameters

#### Table 6: TABLE VI: Actuator Gains and Model Parameters

| | Unitree | Unitree | Dynamixel | |
| --- | --- | --- | --- | --- |
| Param. | A1 | Go1 | XH540-V150 | Units |
| k P k P k_{ $\text{P}} P$ | 15.0 15.0 15.0 15.0 | 10.0 10.0 10.0 10.0 | 5.0 5.0 5.0 5.0 | [N m rad - 1 times absent times newton meter radian 1  $\text{\,}\mathrm{N}\text{\,}\mathrm{m}\text{\,}{\mathrm{rad}}^{-1} times times times power - 1 ]$ |
| k D k D k_{ $\text{D}} D$ | 0.6 0.6 0.6 0.6 | 0.3 0.3 0.3 0.3 | 0.2 0.2 0.2 0.2 | [N m s rad - 1 times absent times newton meter second radian 1  $\text{\,}\mathrm{N}\text{\,}\mathrm{m}\text{\,}\mathrm{s}\text{\,}{\mathrm{rad}}^{-1} times times times times power - 1 ]$ |
| $\tau_{\text{max}} max$ | 34.0 34.0 34.0 34.0 | 23.7 23.7 23.7 23.7 | 4.8 4.8 4.8 4.8 | [N m times absent times newton meter  $\text{\,}\mathrm{N}\text{\,}\mathrm{m} times times ]$ |
| $\dot{q}_{\tau_{\text{max}}} max$ | 7.4 7.4 7.4 7.4 | 10.6 10.6 10.6 10.6 | 0.2 0.2 0.2 0.2 | [rad s - 1 times absent times radian second 1  $\text{\,}\mathrm{rad}\text{\,}{\mathrm{s}}^{-1} times times power - 1 ]$ |
| q  ̇ max  ̇ q max  $\dot{q}_{\text{max}} max$ | 20.0 20.0 20.0 20.0 | 28.8 28.8 28.8 28.8 | 7.0 7.0 7.0 7.0 | [rad s - 1 times absent times radian second 1  $\text{\,}\mathrm{rad}\text{\,}{\mathrm{s}}^{-1} times times power - 1 ]$ |
| $\mu_{s}$ | 0.45 0.45 0.45 0.45 | 0.15 0.15 0.15 0.15 | 0.05 0.05 0.05 0.05 | [N m times absent times newton meter  $\text{\,}\mathrm{N}\text{\,}\mathrm{m} times times ]$ |
| $\mu_{d}$ | 0.023 0.023 0.023 0.023 | 0.016 0.016 0.016 0.016 | 0.009 0.009 0.009 0.009 | [N m s rad - 1 times absent times newton meter second radian 1  $\text{\,}\mathrm{N}\text{\,}\mathrm{m}\text{\,}\mathrm{s}\text{\,}{\mathrm{rad}}^{-1} times times times times power - 1 ]$ |
| b min b min b_{ $\text{min}} min$ | 0.005 0.005 0.005 0.005 | 0.002 0.002 0.002 0.002 | 0.002 0.002 0.002 0.002 | [rad times absent radian  $\text{\,}\mathrm{rad} times ]$ |
| b max b max b_{ $\text{max}} max$ | 0.015 0.015 0.015 0.015 | 0.005 0.005 0.005 0.005 | 0.005 0.005 0.005 0.005 | [rad times absent radian  $\text{\,}\mathrm{rad} times ]$ |
| q max q max \epsilon q $\text{max}}, max$ | 0.02 0.02 0.02 0.02 | 0.02 0.02 0.02 0.02 | 0.02 0.02 0.02 0.02 | [rad times absent radian  $\text{\,}\mathrm{rad} times ]$ |
|  q, 0  q 0 \sigma_{q,0}, 0 | 1.80  10 - 4  1.80 10 4 1.80\cdot 10^{-4} 1.80  10 - 4 | 1.89  10 - 4  1.89 10 4 1.89\cdot 10^{-4} 1.89  10 - 4 | 4.31  10 - 4  4.31 10 4 4.31\cdot 10^{-4} 4.31  10 - 4 | [rad times absent radian  $\text{\,}\mathrm{rad} times ]$ |
|  q, 1  q 1 \sigma_{q,1}, 1 | 3.61  10 - 5  3.61 10 5 3.61\cdot 10^{-5} 3.61  10 - 5 | 5.47  10 - 5  5.47 10 5 5.47\cdot 10^{-5} 5.47  10 - 5 | 2.43  10 - 5  2.43 10 5 2.43\cdot 10^{-5} 2.43  10 - 5 | [s times absent second  $\text{\,}\mathrm{s} times ]$ |
| I m I m I_{m} | 0.011 0.011 0.011 0.011 | 0.0043 0.0043 0.0043 0.0043 | 0.0058 0.0058 0.0058 0.0058 | [kg m 2 times absent times kilogram meter 2  $\text{\,}\mathrm{kg}\text{\,}{\mathrm{m}}^{2} times times power 2 ]$ |

**说明**: TABLE VI: Actuator Gains and Model Parameters

#### Table 7: TABLE VII: Puppeteering Button Mapping

| Button | | Effect |
| --- | --- | --- |
| Menu | | Trigger a safety mode called motion stop. This forces a transition to standing and freezes the joint setpoints with high position gains after waiting 0.5 s times 0.5 second 0.5 $\text{\,}\mathrm{s} 0.5 times.$ |
| View | | Slowly move all joints to the default pose. Only available at startup or while in motion stop. |
| D-pad | ↕ ↕ \updownarrow ↕ | Move the head up-down. |
| | ↔ ↔  $\leftrightarrow ↔$ | Roll the head left-right. |
| Left Joystick | ↕ ↕ \updownarrow ↕ | During walking: Longitudinal walking velocity. During standing: Up pitches the torso forward while the head remains stationary, and Down lowers the torso height. |
| | ↔ ↔  $\leftrightarrow ↔$ | During walking: Turning rate. During standing: Torso yaw while the head remains stationary. |
| | L3 | Pressing the left joystick triggers a scanning animation. |
| Right Joystick | ↕ ↕ \updownarrow ↕ | Pitches the head. During standing, this additionally commands torso pitch. |
| | ↔ ↔  $\leftrightarrow ↔$ | Yaws the head left-right. During standing, the end of the range additionally commands torso yaw. |
| | R3 | Pressing the right joystick toggles the audio level. |
| ABXY | A | Transition to standing. |
| | B | Fully tuck the neck in, turn off the eyes, and retract the antennas. While standing, the torso height is also lowered. |
| | X | Cancel all active animations. |
| | Y | Turn on the background animation layer. |
| Left Trackpad | | Trigger an episodic motion. Each quadrant of the trackpad maps to a different motion. |
| Right Trackpad | | Like the left trackpad. Reserved to trigger four additional episodic motions. |
| Backside | L1 | Turn the head lamp on and off. |
| | R1 | Single press: Start and stop walking, Hold: increase the walking velocity gain to 100 % times 100 percent 100 $\text{\,}\mathrm{\char 37\relax} 100 times . Without holding R1, all velocity commands are scaled to 50 times 50 percent 50\text{\,}\mathrm{\char 37\relax} 50 times of the maximum.$ |
| | L2 / R2 | During walking: Lateral walking velocity. During standing: Roll the torso while the head remains stationary. |
| | L4 | Short press: trigger a happy animation. Long press: trigger an angry animation. |
| | L5 | Short press: trigger an anxious animation. Long press: trigger a curious animation. |
| | R4 | Short press: trigger a “yes” animation. Long press: trigger a “no” animation. |
| | R5 | Trigger an expressive audio clip. |

**说明**: TABLE VII: Puppeteering Button Mapping
## 实验解读

- 评价重点:围绕 biped、inference-time-algorithm、足式运动、实时控制、强化学习,优先看真实机器人结果、扰动恢复、接触稳定性、控制误差、成功率、实时性和 sim-to-real 差距。
- System0 视角:实验结论不只看最终成功率,还要看闭环过程中是否能处理状态估计误差、执行延迟、碰撞/接触突变和外部扰动。
- 与上层系统的关系:如果方法能在快闭环内稳定工作,它可以作为 System1 技能或 System2 规划的底层执行保障;如果只在离线或慢速设定下成立,则还不能直接作为身体级快闭环模块。
## 批判性思考

### 优点

- 论文把 biped、inference-time-algorithm、足式运动、实时控制、强化学习 放到机器人身体级闭环里讨论,和只做高层语义推理的工作相比,更接近真实机器人部署时会遇到的状态、动力学和安全约束。
- 若论文同时给出公式、系统结构图、硬件实验或消融结果,这些证据可以帮助判断方法是否真的能支撑毫秒到百毫秒级反应。

### 局限

- 需要重点核查控制频率、推理延迟、传感延迟、失败恢复、硬件磨损、未见场景泛化和安全边界;这些因素决定它能否成为可靠的 System0 模块。
- 对学习型方法,还要区分训练环境中的鲁棒性和真实部署中的鲁棒性,尤其是接触丰富任务、动态运动任务和人形机器人任务。
## 论文定位

- 研究对象:Design and Control of a Bipedal Robotic Character。
- 关键词:biped、inference-time-algorithm、足式运动、实时控制、强化学习、鲁棒控制、terrain-adaptation。
- System0 关联:关注机器人身体级快反应链路中“状态/接触/局部感知 -> 约束/优化/策略 -> 动作”的短闭环,不把问题停留在语言规划或任务分解层。

## 速查卡片

> [!summary] Design Control Bipedal
> - **论文**: https://www.roboticsproceedings.org/rss20/p103.pdf
> - **arXiv**: http://arxiv.org/abs/2501.05204v1
> - **arXiv HTML**: https://arxiv.org/html/2501.05204v1
