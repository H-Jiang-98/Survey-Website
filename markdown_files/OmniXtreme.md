---
title: "OmniXtreme: Breaking the Generality Barrier in High-Dynamic Humanoid Control"
method_name: "OmniXtreme"
authors: ["Yunshen Wang"]
year: 2026
venue: "RSS"
tags: ["robust-control", "legged-locomotion", "reinforcement-learning", "robot-generalization", "humanoid", "sim-to-real", "whole-body-control", "trajectory-optimization"]
image_source: "online"
arxiv_html: "https://arxiv.org/html/2602.23843v1"
---
# OmniXtreme
## 一句话总结

> OmniXtreme: Breaking the Generality Barrier in High-Dynamic Humanoid Control 主要落在 [[actuator-modeling]]、[[人形机器人]]、[[足式运动]]、[[motion-tracking]] 这一类问题上;从 System0 视角看,它的价值在于把身体状态、接触变化、局部感知或动力学约束接到机器人低层闭环里,让动作生成更稳定、安全、实时。
## 核心贡献

1. 围绕 **OmniXtreme: Breaking the Generality Barrier in High-Dynamic Humanoid Control** 建立了一个与 actuator-modeling、人形机器人、足式运动、motion-tracking、强化学习、robot-generalization 相关的机器人问题设定,重点不是高层语义规划,而是身体级闭环中的状态变化、接触变化或控制约束。
2. 方法层面的关键价值在于把感知、动力学、约束优化、策略学习或硬件反馈组织成可执行的低层动作生成机制,适合放进 System0 的快反应模块中理解。
3. 对系统集成而言,这类工作补上了 System1 技能选择与电机级控制之间的一层:它负责把上层目标变成动态可行、可恢复、可安全过滤的动作。
4. 若论文包含真实机器人、接触任务或仿真到真实迁移,核心应看它是否报告了闭环稳定性、扰动恢复、硬件延迟、失败模式和跨场景泛化。
## 问题背景

这篇论文可以放在 System0 的“具身快系统”背景下阅读:机器人需要在很短时间内根据本体状态、接触、力/触觉、局部视觉或动力学预测做出动作,而不是等待慢速规划器重新推理整个任务。actuator-modeling、人形机器人、足式运动、motion-tracking、强化学习、robot-generalization、鲁棒控制 相关问题的难点通常不在“知道要做什么”,而在动作执行过程中状态会快速偏离、接触会突然改变、约束会实时收紧,低层系统必须及时修正。

### System0 关联

精读时优先看三件事:第一,论文使用哪些快速反馈量,例如关节状态、接触状态、力觉、触觉、深度或局部视觉;第二,反馈如何进入 MPC、WBC、策略网络、安全过滤器或恢复控制器;第三,输出是否直接影响机器人下一步动作,而不是只服务离线分析或高层任务规划。
## 方法详解

### System0 视角解析

从 System0 角度看,这篇论文的方法部分应当围绕 actuator-modeling、人形机器人、足式运动、motion-tracking、强化学习、robot-generalization、鲁棒控制、scalable-robot-learning 来读。核心问题是:系统如何从快速变化的机器人状态中构造控制输入,如何处理接触、碰撞、约束、模型误差或执行延迟,以及如何把这些信息变成下一时刻的动作。

如果方法里有优化问题,重点看目标函数、约束和求解频率;如果方法里有学习策略,重点看观测空间、动作空间、训练扰动和部署时的闭环频率;如果方法里有硬件或传感器,重点看传感链路是否真的进入控制回路。这样读可以把论文和 System2 语言规划、System1 技能编排区分开来。

### 关键公式与机制

#### 公式 1: [[优化目标/约束]]

$$
\mathcal{L}_{\text{FM}}(\theta)=\mathbb{E}_{t,\epsilon,a_{\text{expert}}}\left[\lVert v_{\theta}(a_{t},t,o)-(\epsilon-a_{\text{expert}})\rVert^{2}\right],
$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 2: [[优化目标/约束]]

$$
\mathcal{L}_{\text{neg-power}}=\sum_{j\in\mathcal{J}}\left(\frac{\max(-P_{j}-P_{\text{db}},0)}{K}\right)^{2},
$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 3: [[优化目标/约束]]

$$\tau_{\max,0}=\begin{cases}\tau_{y1},&v\cdot\tau_{\text{in}}>0,\\ \tau_{y2},&v\cdot\tau_{\text{in}}\leq 0.\end{cases}$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 4: [[优化目标/约束]]

$$\mathrm{Airborne}=\frac{1}{T}\sum_{t=0}^{T-1}\mathbf{1}\!\left[\min_{b\in\mathcal{F}}z_{t}^{(b)}>h_{\text{air}}\right],$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 5: [[优化目标/约束]]

$$\mathbf{o}=[\boldsymbol{\psi},\,\mathbf{e}_{\text{torso}},\,\mathcal{V}_{\text{imu}},\,\boldsymbol{q}-\boldsymbol{q}^{0},\,\dot{\boldsymbol{q}},\,\mathbf{a}_{\text{last}}],$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 6: [[优化目标/约束]]

$$\sum\left(\frac{\max\!\bigl(0,\;-\tau\dot{q}-150\bigr)}{500}\right)^{\!2}$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 7: [[动力学或策略机制]]

$$\mathrm{MPJPE}=1000\cdot\frac{1}{T}\sum_{t=1}^{T}\left(\frac{1}{N}\sum_{i=1}^{N}\left\lVert \mathbf{p}^{\text{ref}}_{t,i}-\mathbf{p}^{\text{rob}}_{t,i}\right \rVert_{2}\right).$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 8: [[动力学或策略机制]]

$$\Delta v=1000\cdot\Delta t\cdot\frac{1}{T}\sum_{t=1}^{T}\left(\frac{1}{N}\sum_{i=1}^{N}\left\lVert \mathbf{v}^{\text{ref}}_{t,i}-\mathbf{v}^{\text{rob}}_{t,i}\right \rVert_{2}\right).$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 9: [[动力学或策略机制]]

$$e^{\text{acc}}_{t}=\frac{1}{N}\sum_{i=1}^{N}\left\lVert \mathbf{a}^{\text{ref}}_{t,i}-\mathbf{a}^{\text{rob}}_{t,i}\right \rVert_{2}.$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 10: [[优化目标/约束]]

$$c_{\text{knee}}=\sum_{j\in\mathcal{J}_{\text{knee}}}\left(\frac{\tilde{P}_{j}}{500}\right)^{\!2},$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。
## 关键图表

### Figure 1: Overview of the OmniXtreme. (a) Pretraining phase: A unified base policy is trained

![Figure 1](https://arxiv.org/html/2602.23843v1/x1.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Overview of the OmniXtreme. (a) Pretraining phase: A unified base policy is trained”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 2: Capacity scaling. Tracking fidelity and robustness as a function of model capacity. O

![Figure 2](https://arxiv.org/html/2602.23843v1/x3.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Capacity scaling. Tracking fidelity and robustness as a function of model capacity. O”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 3: Qualitative results. Representative real-world rollouts produced by OmniXtreme, exec

![Figure 3](https://arxiv.org/html/2602.23843v1/x4.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Qualitative results. Representative real-world rollouts produced by OmniXtreme, exec”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### 论文表格

#### Table 1: TABLE I: Configurations for noise, domain randomization, and termination thresholds during pre-training and post-traini

| Parameter Item | Moderate | Aggressive |
| --- | --- | --- |
| Noise and Domain Randomization | | |
| Joint Position (rad) | ± 0.01 \pm 0.01 | ± 0.01 \pm 0.01 |
| Joint Velocity (rad/s) | ± 0.5 \pm 0.5 | ± 0.5 \pm 0.5 |
| Angular Velocity (rad/s) | ± 0.2 \pm 0.2 | ± 0.2 \pm 0.2 |
| Torso 6D Rotation (rad) | ± 0.05 \pm 0.05 | ± 0.05 \pm 0.05 |
| Base CoM Offset (m) | x x: ± 0.025, y, z \pm 0.025,y,z: ± 0.05 \pm 0.05 | x x: ± 0.025, y, z \pm 0.025,y,z: ± 0.05 \pm 0.05 |
| Static Friction | [0.3, 1.6 ] [0.3,1.6] | [0.3, 1.6 ] [0.3,1.6] |
| Dynamic Friction | [0.3, 1.2 ] [0.3,1.2] | [0.3, 1.2 ] [0.3,1.2] |
| Action Delay (ms) | [0, 15 ] [0,15] | [5, 10 ]  $\mathbf{[5,10]}$ |
| Coefficient of Restitution | None | [0.0, 0.5 ]  $\mathbf{[0.0,0.5]}$ |
| Default Calib. (rad) | ± 0.01 \pm 0.01 | ± 0.01 \pm 0.01 |
| Init. Pose (rad) | ± 0.1 \pm 0.1 | ± 0.15  $\mathbf{\pm 0.15}$ |
| Init. Lin. Vel. (m/s) | x  y xy: ± 0.5, z \pm 0.5,z: ± 0.2 \pm 0.2 | xy: ± 0.75, z: ± 0.3  $\mathbf{xy:\pm 0.75,z:\pm 0.3}$ |
| Init. Ang. Vel. (rad/s) | R  P RP: ± 0.52, Y \pm 0.52,Y: ± 0.78 \pm 0.78 | RP: ± 0.78, Y: ± 1.17  $\mathbf{RP:\pm 0.78,Y:\pm 1.17}$ |
| Push Frequency (s) | 1.0 - 3.0 1.0-3.0 | 1.0 - 3.0 1.0-3.0 |
| Push Lin. Vel. (m/s) | x  y xy: ± 0.5, z \pm 0.5,z: ± 0.2 \pm 0.2 | x  y xy: ± 0.5, z \pm 0.5,z: ± 0.2 \pm 0.2 |
| Push Ang. Vel. (rad/s) | R  P RP: ± 0.52, Y \pm 0.52,Y: ± 0.78 \pm 0.78 | R  P RP: ± 0.52, Y \pm 0.52,Y: ± 0.78 \pm 0.78 |
| Terrain Surface / Step (m) | None | [0, 0.01 ] / 0.01  $\mathbf{[0,0.01]/0.01}$ |
| Termination Thresholds | | |
| Torso Pos. Z / Ori. Error | 0.25 0.25 m / 0.8 0.8 rad | 0.375  m / 1.2  rad  $\mathbf{0.375m/1.2rad}$ |
| End-Effector Z-Error (m) | 0.25 0.25 | 0.375  $\mathbf{0.375}$ |

**说明**: TABLE I: Configurations for noise, domain randomization, and termination thresholds during pre-training and post-training phases. Here ± x $\pm$ x denotes [- x, x ] [-x,x].

#### Table 2: TABLE II: Scalable high-fidelity motion tracking diverse motion sets. OmniXtreme consistently achieves lower kinem

| Method | LaFAN1+XtremeMotion | XtremeMotion | Unseen Motions | | | | | | | |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| | MPJPE ↓ \downarrow |  \Delta vel ↓ \downarrow |  \Delta acc ↓ \downarrow | Succ.(%) ↑ \uparrow | MPJPE ↓ \downarrow |  \Delta vel ↓ \downarrow |  \Delta acc ↓ \downarrow | Succ.(%) ↑ \uparrow | MPJPE ↓ \downarrow | Succ.(%) ↑ \uparrow |
| From-scratch RL [28 ] | 47.95 | 10.03 | 3.27 | 82.95 | 54.19 | 14.04 | 4.04 | 79.45 | 56.87 | 85.29 |
| Specialist →  $\rightarrow Unified MLP [58 ]$ | 33.35 | 6.70 | 2.11 | 94.91 | 43.43 | 11.38 | 2.51 | 89.22 | 58.94 | 85.95 |
| OmniXtreme (Pretrain only) | 32.65 | 6.34 | 2.04 | 97.17 | 37.11 | 10.46 | 2.39 | 95.16 | 56.25 | 89.23 |
| OmniXtreme (Pretrain + Post-train) | 30.93 | 6.19 | 2.13 | 98.54 | 36.17 | 9.94 | 2.58 | 95.64 | 56.05 | 89.54 |

**说明**: TABLE II: Scalable high-fidelity motion tracking diverse motion sets. OmniXtreme consistently achieves lower kinematic errors and higher success rates than baselines, particularly on high-dynamic and unseen motions.

#### Table 3: TABLE III: Real-world evaluation of OmniXtreme on Unitree G1. We evaluate OmniXtreme on physical hardware using motions

| Skill | #Motions | Attempts | Success (%) ↑ \uparrow |
| --- | --- | --- | --- |
| Flip | 7 | 55 | 96.36 |
| Handspring | 5 | 35 | 88.57 |
| Acrobatics | 4 | 15 | 80.00 |
| Breakdance | 5 | 22 | 86.36 |
| Martial arts | 3 | 30 | 93.33 |
| Total | 24 | 157 | 91.08 |

**说明**: TABLE III: Real-world evaluation of OmniXtreme on Unitree G1. We evaluate OmniXtreme on physical hardware using motions drawn from the XtremeMotion motion library.

#### Table 4: TABLE IV: Ablation of post-training mechanisms. Real-world executability of different skills incremental post-trai

| Skill | None | +MC | +MC+ADR | Full (+MC+ADR+PS) |
| --- | --- | --- | --- | --- |
| Flip | △ \triangle | ✓ \checkmark | ✓ \checkmark | ✓ \checkmark |
| Breakdance | △ \triangle | △ \triangle | ✓ \checkmark | ✓ \checkmark |
| Acrobatics |  \times | △ \triangle | ⊝ \circleddash | ✓ \checkmark |

**说明**: 该表格来自论文中的公式、奖励项或实验设置；为避免 PDF 抽取噪声，网页中仅保留可渲染的核心符号和数值。

#### Table 5: TABLE V: Motion information summary.

| Motion ID | Motion Source | Motion Description |
| --- | --- | --- |
| 1 | CMU-85-05 | Handstand walk. |
| 2 | CMU-85-10 | Handstand spin. |
| 3 | CMU-88-09 | Back Handspring with a full twist. |
| 4 | CMU-88-08 | Back Handspring with a half twist. |
| 5 | CMU-90-08 | Aerial Cartwheel |
| 6 | CMU-85-14 | B-boying with a rapid-fire string of back handsprings. |
| 7 | CMU-85-13 | Skip back, pivot through a hand-to-headstand, drop, flip, and bounce back. |
| 8 | CMU-90-06 | Fly kick. |
| 9 | CMU-90-34 | Forward roll. |
| 10 | CMU-90-01 | Backward roll. |
| 11 | CMU-90-28 | Backspin. |
| 12 | CMU-85-08 | Thomas Flare. |
| 13 | CMU-85-12 | Long breaking dance. |
| 14 | CMU-85-04 | Another long breaking dance. |
| 15 | CMU-85-01 | Bicycle kick flip. |
| 16 | CMU-85-02 | Another bicycle kick flip. |
| 17 | CMU-88-06 | Butterfly kick. |
| 18 | CMU-85-06 | Webster flip. |
| 19 | CMU-49-08 | Two consecutive cartwheels. |
| 20 | CMU-90-30 | Alternating pistol squats. |
| 21 | CMU-90-29 | Acrobatic gymnastics with cartwheel and back handsprings. |
| 22 | CMU-90-19 | Crawl forward and backflip. |
| 23 | GeneralA11-MilitaryCrawlForward | Stay low and crawl forward. |
| 24 | IconicHeroMotion-SwordJudgment | Spinning slash. |
| 25 | IconicHeroMotion-SwordHeroic | Another style of spinning slash. |
| 26 | HandtoHandCombat-B3AttackReverseTurningKick | Reverse turning kick. |
| 27 | HandtoHandCombat-D2AttackPunchSweepKick | Punch and sweep kick. |
| 28 | HandtoHandCombat-D4AttackReverseFrontSnapKick | Reverse and front snap kick. |
| 29 | HandtoHandCombat-D4DodgeRollBack | Two consecutive rolls in different styles. |
| 30 | HandtoHandCombat-G1GetupKipUp | Kip up. |
| 31 | HandtoHandCombat-G2GetupHandstandKipUp | Handstand kip. |
| 32 | HandtoHandCombat-KO2FalltoGroundAxelDown | Execute a downward Axel into a ground fall. |
| 33 | LadyAgent-AgentElbowStrikeSweepKick | Elbow strike and sweep kick. |
| 34 | LadyAgent-AgentHandspring | Front handspring. |
| 35 | LadyAgent-AgentRollForward | Shoulder roll. |
| 36 | LadyAgent-AgentShootSidewardRoll | Quick side roll. |
| 37 | LadyAgent-AgentSnapKick | Snapkick. |
| 38 | Mimickit-g1spinkick | Spinkick. |
| 39 | LAFAN1 -dance1subject2 [82.8,106.9]s | Constantly spin in full circles. |
| 40 | LAFAN1 -dance1subject2 [145.3,161.3]s | Play guitar while hopping on one leg. |
| 41 | LAFAN1 -dance2subject3 [160.2,224.3]s | Flutter arms and hands. |
| 42 | LAFAN1 -fightandsports1subject1 [167.0,176.9]s | Continuous long jumps. |
| 43 | LAFAN1 -fightandsports1subject1 [0.0,17.0]s | Balance on one leg. |
| 44 | LAFAN1 -dance1subject1 [104.6,119.1]s | Cartwheel twice. |
| 45 | LAFAN1 -jump1subject1 [70.3,87.6]s | Play hopscotch. |
| 46 | LAFAN1 -jump1subject1 [89.1,138.4]s | Hop on one foot. |
| 47 | LAFAN1 -jump1subject1 [0.0,72.0]s | Successive leaps. |
| 48 | LAFAN1 -fightandsports1subject4 [154.4,219.8]s | Vigorously swing the baseball bat. |
| 49 | LAFAN1 -fightandsports1subject4 [61.1,85.9]s | Lash the golf club. |
| 50 | LAFAN1 -fightandsports1subject4 [28.2,61.9]s | Diverse kicking movements. |
| 51 | LAFAN1 -run1subject2 [20.4,101.0]s | Shuttle run. |
| 52 | LAFAN1 -run1subject2 [92.0,130.6]s | Run rapidly. |
| 53 | LAFAN1 -fallandgetup2subject3 [33.8,56.1]s | Kip up twice. |
| 54 | LAFAN1 -fightandsports1subject1 [17.3,27.2]s | Roundhouse kick. |
| 55 | LAFAN1 -jump1subject2 [187.9,196.2]s | Push up. |
| 56 | LAFAN1 -jump1subject2 [196.0,205.8]s | Lateral roll. |
| 57 | LAFAN1 -jump1subject2 [205.6,244.4]s | Lateral roll and kip up. |

**说明**: TABLE V: Motion information summary.

#### Table 6: TABLE VI: Motion tracking hyperparameters for teacher training [28 ].

| Hyperparameter | Value |
| --- | --- |
| Architecture | |
| Actor MLP hidden dimensions | [512, 256, 128] |
| Critic MLP hidden dimensions | [512, 256, 128] |
| Activation function | ELU |
| Training | |
| Steps per environment | 24 |
| Clip parameter | 0.2 |
| Entropy coefficient | 0.005 |
| Value loss coefficient | 1.0 |
| Discount factor $\gamma)$ | 0.99 |
| GAE  \lambda | 0.95 |
| Desired KL | 0.01 |
| Learning epochs | 5 |
| Mini-batches | 4 |

**说明**: TABLE VI: Motion tracking hyperparameters for teacher training [28 ].

#### Table 7: TABLE VII: Reward function terms and expressions for Teacher Policy training [28 ].

| Reward Term | Weight | Expression |
| --- | --- | --- |
| Global Torso Position | 0.5 | $\exp(-\lVert p_{err} \rVert^{2}/0.3^{2})$ |
| Global Torso Orientation | 0.5 | $\exp(-\lVert \theta_{err} \rVert^{2}/0.4^{2})$ |
| Relative Body Position | 1.0 | $\exp(-\lVert p_{rel\_err} \rVert^{2}/0.3^{2})$ |
| Relative Body Orientation | 1.0 | $\exp(-\lVert \theta_{rel\_err} \rVert^{2}/0.4^{2})$ |
| Body Linear Velocity | 1.0 | $\exp(-\lVert v_{err} \rVert^{2}/1.0^{2})$ |
| Body Angular Velocity | 1.0 | $\exp(-\lVert \omega_{err} \rVert^{2}/3.14^{2})$ |
| Action Rate | -0.1 |  a t - a t - 1  2 \\|a_{t}-a_{t-1}\\|^{2} |
| Joint Limit | -10.0 | ∑ max  (0, q - q l  i  m  i  t)  $\sum\max(0,q-q_{limit})$ |
| Undesired Contacts | -0.1 | ∑ 1  (F c  o  n  t  a  c  t > 1.0)  $\sum 1(F_{contact}>1.0)$ |

**说明**: TABLE VII: Reward function terms and expressions for Teacher Policy training [28 ].

#### Table 8: TABLE VIII: Reward function terms and expressions for Residual Policy training. Refer to Sec. V-K for further details r

| Reward Term | Weight | Expression |
| --- | --- | --- |
| Global Torso Position | 0.5 | $\exp(-\lVert p_{err} \rVert^{2}/0.3^{2})$ |
| Global Torso Orientation | 0.5 | $\exp(-\lVert(\theta_{err} \rVert^{2}/0.4^{2})$ |
| Relative Body Position | 1.0 | $\exp(-\lVert p_{rel\_err} \rVert^{2}/0.3^{2})$ |
| Relative Body Orientation | 1.0 | $\exp(-\lVert \theta_{rel\_err} \rVert^{2}/0.4^{2})$ |
| Body Linear Velocity | 1.0 | $\exp(-\lVert v_{err} \rVert^{2}/1.0^{2})$ |
| Body Angular Velocity | 1.0 | $\exp(-\lVert \omega_{err} \rVert^{2}/3.14^{2})$ |
| Action Rate | -0.1 |  a t - a t - 1  2 \\|a_{t}-a_{t-1}\\|^{2} |
| Joint Limit | -10.0 | ∑ max  (0, q - q l  i  m  i  t)  $\sum\max(0,q-q_{limit})$ |
| Undesired Contacts | -0.1 | ∑ 1  (F c  o  n  t  a  c  t > 1.0)  $\sum 1(F_{contact}>1.0)$ |
| Power Safety Regularization | -10.0 | $\sum\left(\frac{\max\!\bigl(0,\;-\tau\dot{q}-150\bigr)}{500}\right)^{\!2}$ |

**说明**: TABLE VIII: Reward function terms and expressions for Residual Policy training. Refer to Sec. V-K for further details regarding power safety regularization.

#### Table 9: TABLE IX: Actuator modeling parameters.

| Actuator | $\tau_{y1}$ | $\tau_{y2}$ | v x  1 v_{x1} | v x  2 v_{x2} | $\mu_{s}$ | v a  c  t v_{act} | $\mu_{d}$ | I I |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 5020-16 | 24.8 | 31.9 | 30.86 | 40.13 | 0.6 | 0.01 | 0.06 | 3.610e-03 |
| 7520-14.3 | 71.0 | 83.3 | 22.63 | 35.52 | 1.6 | 0.01 | 0.16 | 1.018e-02 |
| 7520-22.5 | 111.0 | 131.0 | 14.5 | 22.7 | 2.4 | 0.01 | 0.24 | 2.510e-02 |
| 4010-25 | 4.8 | 8.6 | 15.3 | 24.76 | 0.6 | 0.01 | 0.06 | 4.250e-03 |

**说明**: TABLE IX: Actuator modeling parameters.

#### Table 10: TABLE X: Unitree G1 Joint-to-Motor Mapping.

| Joint Name | Motor Model |
| --- | --- |
| Hip Pitch Joint | 7520-22.5 |
| Hip Roll Joint | 7520-22.5 |
| Knee Joint | 7520-22.5 |
| Hip Yaw Joint | 7520-14.3 |
| Ankle Pitch Joint | 5020 |
| Ankle Roll Joint | 5020 |
| Waist Roll Joint | 5020 |
| Waist Pitch Joint | 5020 |
| Waist Yaw Joint | 7520-14.3 |
| Shoulder Pitch Joint | 5020 |
| Shoulder Roll Joint | 5020 |
| Shoulder Yaw Joint | 5020 |
| Elbow Joint | 5020 |
| Wrist Roll Joint | 5020 |
| Wrist Pitch Joint | 4010 |
| Wrist Yaw Joint | 4010 |

**说明**: TABLE X: Unitree G1 Joint-to-Motor Mapping.

#### Table 11: TABLE XI: Skill-level grouping of real-world evaluation motions. Each skill category groups multiple motion instances w

| Skill category | Motion IDs |
| --- | --- |
| Flip | 5, 9, 10, 15, 16, 18, 22 |
| Handspring | 2, 3, 4, 19, 23 |
| Acrobatics | 1, 6, 7, 20, 21 |
| Breakdance | 11, 12, 13, 14 |
| Martial arts | 8, 17, 38 |

**说明**: TABLE XI: Skill-level grouping of real-world evaluation motions. Each skill category groups multiple motion instances with similar semantic meaning and dynamic structure. Motion IDs correspond to retargeted motions used during hardware evaluation.

#### Table 12: TABLE XII: Motion subsets used in Q2 fidelity–scalability analysis. All evaluations are performed on the same first 10

| Training motions | Motion IDs |
| --- | --- |
| 10 | 3–10, 13, 14 |
| 20 | 2–10, 13-22 |
| 50 | All |

**说明**: TABLE XII: Motion subsets used in Q2 fidelity–scalability analysis. All evaluations are performed on the same first 10 extreme motions. Larger training sets (20 and all motions) extend this core set with additional diverse extreme motions.
## 实验解读

- 评价重点:围绕 actuator-modeling、人形机器人、足式运动、motion-tracking、强化学习,优先看真实机器人结果、扰动恢复、接触稳定性、控制误差、成功率、实时性和 sim-to-real 差距。
- System0 视角:实验结论不只看最终成功率,还要看闭环过程中是否能处理状态估计误差、执行延迟、碰撞/接触突变和外部扰动。
- 与上层系统的关系:如果方法能在快闭环内稳定工作,它可以作为 System1 技能或 System2 规划的底层执行保障;如果只在离线或慢速设定下成立,则还不能直接作为身体级快闭环模块。
## 批判性思考

### 优点

- 论文把 actuator-modeling、人形机器人、足式运动、motion-tracking、强化学习 放到机器人身体级闭环里讨论,和只做高层语义推理的工作相比,更接近真实机器人部署时会遇到的状态、动力学和安全约束。
- 若论文同时给出公式、系统结构图、硬件实验或消融结果,这些证据可以帮助判断方法是否真的能支撑毫秒到百毫秒级反应。

### 局限

- 需要重点核查控制频率、推理延迟、传感延迟、失败恢复、硬件磨损、未见场景泛化和安全边界;这些因素决定它能否成为可靠的 System0 模块。
- 对学习型方法,还要区分训练环境中的鲁棒性和真实部署中的鲁棒性,尤其是接触丰富任务、动态运动任务和人形机器人任务。
## 论文定位

- 研究对象:OmniXtreme: Breaking the Generality Barrier in High-Dynamic Humanoid Control。
- 关键词:actuator-modeling、人形机器人、足式运动、motion-tracking、强化学习、robot-generalization、鲁棒控制、scalable-robot-learning、仿真到真实迁移、轨迹优化。
- System0 关联:关注机器人身体级快反应链路中“状态/接触/局部感知 -> 约束/优化/策略 -> 动作”的短闭环,不把问题停留在语言规划或任务分解层。

## 速查卡片

> [!summary] OmniXtreme
> - **论文**: https://www.roboticsproceedings.org/rss22/p031.pdf
> - **arXiv**: http://arxiv.org/abs/2602.23843v1
> - **arXiv HTML**: https://arxiv.org/html/2602.23843v1
