---
title: "Tilde: Teleoperation for Dexterous In-Hand Manipulation Learning with a DeltaHand"
method_name: "Tilde"
authors: ["Zilin Si"]
year: 2024
venue: "RSS"
tags: ["robot-manipulation", "robust-control", "real-time-control", "reinforcement-learning", "adaptive-control", "imitation-learning", "robot-generalization", "closed-loop-control", "dexterous-manipulation"]
image_source: "online"
arxiv_html: "https://arxiv.org/html/2405.18804v2"
---
# Tilde
## 一句话总结

> Tilde: Teleoperation for Dexterous In-Hand Manipulation Learning with a DeltaHand 主要落在 [[adaptive-control]]、[[closed-loop-control]]、[[灵巧操作]]、[[diffusion-policy]] 这一类问题上;从 System0 视角看,它的价值在于把身体状态、接触变化、局部感知或动力学约束接到机器人低层闭环里,让动作生成更稳定、安全、实时。
## 核心贡献

1. 围绕 **Tilde: Teleoperation for Dexterous In-Hand Manipulation Learning with a DeltaHand** 建立了一个与 adaptive-control、closed-loop-control、灵巧操作、diffusion-policy、模仿学习、in-hand-manipulation 相关的机器人问题设定,重点不是高层语义规划,而是身体级闭环中的状态变化、接触变化或控制约束。
2. 方法层面的关键价值在于把感知、动力学、约束优化、策略学习或硬件反馈组织成可执行的低层动作生成机制,适合放进 System0 的快反应模块中理解。
3. 对系统集成而言,这类工作补上了 System1 技能选择与电机级控制之间的一层:它负责把上层目标变成动态可行、可恢复、可安全过滤的动作。
4. 若论文包含真实机器人、接触任务或仿真到真实迁移,核心应看它是否报告了闭环稳定性、扰动恢复、硬件延迟、失败模式和跨场景泛化。
## 问题背景

这篇论文可以放在 System0 的“具身快系统”背景下阅读:机器人需要在很短时间内根据本体状态、接触、力/触觉、局部视觉或动力学预测做出动作,而不是等待慢速规划器重新推理整个任务。adaptive-control、closed-loop-control、灵巧操作、diffusion-policy、模仿学习、in-hand-manipulation、实时控制 相关问题的难点通常不在“知道要做什么”,而在动作执行过程中状态会快速偏离、接触会突然改变、约束会实时收紧,低层系统必须及时修正。

### System0 关联

精读时优先看三件事:第一,论文使用哪些快速反馈量,例如关节状态、接触状态、力觉、触觉、深度或局部视觉;第二,反馈如何进入 MPC、WBC、策略网络、安全过滤器或恢复控制器;第三,输出是否直接影响机器人下一步动作,而不是只服务离线分析或高层任务规划。
## 方法详解

### System0 视角解析

从 System0 角度看,这篇论文的方法部分应当围绕 adaptive-control、closed-loop-control、灵巧操作、diffusion-policy、模仿学习、in-hand-manipulation、实时控制、robot-generalization 来读。核心问题是:系统如何从快速变化的机器人状态中构造控制输入,如何处理接触、碰撞、约束、模型误差或执行延迟,以及如何把这些信息变成下一时刻的动作。

如果方法里有优化问题,重点看目标函数、约束和求解频率;如果方法里有学习策略,重点看观测空间、动作空间、训练扰动和部署时的闭环频率;如果方法里有硬件或传感器,重点看传感链路是否真的进入控制回路。这样读可以把论文和 System2 语言规划、System1 技能编排区分开来。

### 关键公式与机制

#### 公式 1: [[动力学或策略机制]]

$$\widetilde{\mathit{Tilde}}$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 2: [[动力学或策略机制]]

$$N_{input}=(512+12)*2+12*8$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。
## 关键图表

### Figure 1: Pipeline / core system figure: a) A DeltaHand with an in-hand RGB camera. A kinematic twin teleoperation interface

![Figure 1](https://arxiv.org/html/2405.18804v2/extracted/5805420/figs/hand-teleop.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Pipeline / core system figure: a) A DeltaHand with an in-hand RGB camera. A kinematic twin teleoperation interface”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 2: Comparisons between (a) the hand from DeltaHands [38 ] and (b) our adapted hand. We

![Figure 2](https://arxiv.org/html/2405.18804v2/extracted/5805420/figs/Hand-Compare.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Comparisons between (a) the hand from DeltaHands [38 ] and (b) our adapted hand. We”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 3: Overview of the teleoperation system. The system can be modularized to the control P

![Figure 3](https://arxiv.org/html/2405.18804v2/extracted/5805420/figs/Hand-Control.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Overview of the teleoperation system. The system can be modularized to the control P”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### 论文表格

#### Table 1: TABLE I: Experimental results on six tasks. We show that with less than 50 demos, we can achieve success rates 60%

| Task | Grasp | Block Slide | Block Lift | Cap Twist | Ball Roll | Syringe Push |
| --- | --- | --- | --- | --- | --- | --- |
| # demos | 45 | 40 | 20 | 30 | 20 | 50 |
| # DAgger demos | 10 | 0 | 10 | 10 | 5 | 20 |
| # Success / # tests before DAgger | 17/20 | 10/10 | 7/10 | 8/10 | 7/10 | 6/10 |
| # Success / # tests after DAgger | 20/20 | 10/10 | 8/10 | 10/10 | 10/10 | 8/10 |

**说明**: TABLE I: Experimental results on six tasks. We show that with less than 50 demos, we can achieve success rates 60% on all tasks before DAgger. With additional DAgger demonstrations, all tasks have improved results and achieved success rates 80%.

#### Table 2: TABLE II: We evaluate the performance of three Imitation Learning methods: Behavior Cloning (BC) (Robotmimic [24 ]),

| Task | Observations | BC [24 ] | IBC [12 ] | Diffusion Policy [9 ] |
| --- | --- | --- | --- | --- |
| Cap | State | 0/10 | 0/10 | 3/10 |
| Twist | State + In-Hand | 4/10 | 0/10 | 10/10 |
| Shape Insert | State | 0/10 | 3/10 | 2/10 |
| State + In-Hand | 2/10 | 3/10 | 7/10 | |

**说明**: TABLE II: We evaluate the performance of three Imitation Learning methods: Behavior Cloning (BC) (Robotmimic [24 ]), Implicit Behavior Cloning (IBC) [12 ], and Diffusion Policy [9 ] on two tasks, Cap Twist and Shape Insert, either with joint states only or with joint states and in-hand visual observations. We show Diffusion Policy achieves the best performance.

#### Table 3: TABLE III: Frequency of sensor and control signals on ROS.

| | Joint State | Control (Desired Joint Sate) | Linear Slider | In-hand Camera | External Camera |
| --- | --- | --- | --- | --- | --- |
| Frequency (Hz) | 20 | 33 | 133 | 30 | 10 |

**说明**: TABLE III: Frequency of sensor and control signals on ROS.

#### Table 4: TABLE IV: Teleoperation interface characterization and comparison with visual tracking by using a Leap Motion camera. Ou

| Teleoperation | Communication | Demonstration Collecting Time (seconds) ↓ ↓ \downarrow ↓ | Mapping Error (mm) ↓ ↓ \downarrow ↓ | | | | |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Interfaces | Frequency (Hz) ↑ ↑ \uparrow ↑ | Block Slide | Cap Twist | Shape Insert | Block Slide | Cap Twist | Shape Insert |
| Visual Tracking | 10 | 54.05 | 81.51 | 14.47 | 0.60 | 0.34 | 0.31 |
| Ours | 133 | 16.99 | 31.11 | 11.81 | 0.23 | 0.23 | 0.20 |

**说明**: TABLE IV: Teleoperation interface characterization and comparison with visual tracking by using a Leap Motion camera. Our proposed TeleHand has better performance on all metrics when evaluating communication efficiency, demonstration collection time, and mapping errors.

#### Table 5: TABLE V: Comparison with other state-of-the-art teleoperation systems for dexterous robotic hands. We show that our syst

| Teleoperation | Robotic | Cost | # DoF | Hand | Hand | Teleoperation | Availability |
| --- | --- | --- | --- | --- | --- | --- | --- |
| System | Hand | | | Type | Material | Interface | |
| DexPilot [15 ] | Allegro | $15,000 | 16 | Anthropomorphic | Rigid | Vision | Off-the-shelf |
| | | | | | (RealSense) | | |
| DIME [6 ] | Allegro | $15,000 | 16 | Anthropomorphic | Rigid | Vision | Off-the-shelf |
| | | | | | (RealSense) | | |
| LEAP Hand [37 ] | LEAP Hand | $2,000 | 16 | Anthropomorphic | Rigid | Vision (RGB Camera) | Open-sourced |
| | | | | | Manus Meta Glove | | |
| Stewart Hand [26 ] | Stewart Hand | $600 | 6 | Underactuated | Rigid | Space Mouse | Open-sourced |
| [25 ] | DASH Hand | $1500 | 16 | Anthropomorphic | Soft | Manus Meta Glove | Open-sourced |
| [40 ] | RBO Hand 3 | $2350* | 16 | Anthropomorphic | Soft | Vision (Webcam) | Open-sourced |
| Ours | DeltaHand | $1,000 | 12 | Exactly Constrained | Soft | Kinematic Twin | Open-sourced |

**说明**: TABLE V: Comparison with other state-of-the-art teleoperation systems for dexterous robotic hands. We show that our system has a relatively lower cost while still preserving high dexterity. The DeltaHand has soft fingertips and a compliant finger structure which assists with adapting to different objects and environments and tolerating deviations from learned policies. *The estimated cost of the RBO Hand 3 is $250 for manufacturing the hand,$480 for 16 Freescale MPX4250 pressure sensors, and $1600 for 16 pneumatic Matrix Series 320 valve controllers based on [11, 31 ].

#### Table 6: TABLE VI: Comparison with other state-of-the-art policy learning systems for dexterous robotic hands. We show that our s

| Learning | Robotic | Task Types | Policy Learning | Feedback | | | | |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| System | Hand | Grasping | Translational | Rotational | Method | Environment | Proprioception | Vision |
| DIME [6 ] | Allegro | | | ✓ | IL | Sim & Real | ✓ | ✓ |
| LEAP Hand [37 ] | LEAP Hand | ✓ | | ✓ | RL | Sim-to-Real | ✓ | |
| Visual Dexterity [7 ] | D’Claw [1 ] | | | ✓ | RL | Sim-to-Real | ✓ | ✓ |
| DEFT [18 ] | DASH Hand [25 ] | ✓ | | ✓ | RL | Real | | ✓ |
| [29 ] | RBO Hand 3 [31 ] | | ✓ | ✓ | Motion | Real | | |
| | | | | Planning | | | | |
| [3 ] | Shadow Hand [10 ] | | | ✓ | RL | Sim-to-Real | | ✓ |
| [27 ] | Yale Model Q [23 ] | ✓ | ✓ | ✓ | Motion | Real | | ✓ |
| | | | | Planning | | | | |
| Ours | DeltaHand [38 ] | ✓ | ✓ | ✓ | IL | Real | ✓ | ✓ |

**说明**: TABLE VI: Comparison with other state-of-the-art policy learning systems for dexterous robotic hands. We show that our system is evaluated on a variety of tasks including grasping, in-hand translation and rotation tasks. Leveraging end-to-end real-world imitation learning has the benefits of data efficiency and less data distribution shift.
## 实验解读

- 评价重点:围绕 adaptive-control、closed-loop-control、灵巧操作、diffusion-policy、模仿学习,优先看真实机器人结果、扰动恢复、接触稳定性、控制误差、成功率、实时性和 sim-to-real 差距。
- System0 视角:实验结论不只看最终成功率,还要看闭环过程中是否能处理状态估计误差、执行延迟、碰撞/接触突变和外部扰动。
- 与上层系统的关系:如果方法能在快闭环内稳定工作,它可以作为 System1 技能或 System2 规划的底层执行保障;如果只在离线或慢速设定下成立,则还不能直接作为身体级快闭环模块。
## 批判性思考

### 优点

- 论文把 adaptive-control、closed-loop-control、灵巧操作、diffusion-policy、模仿学习 放到机器人身体级闭环里讨论,和只做高层语义推理的工作相比,更接近真实机器人部署时会遇到的状态、动力学和安全约束。
- 若论文同时给出公式、系统结构图、硬件实验或消融结果,这些证据可以帮助判断方法是否真的能支撑毫秒到百毫秒级反应。

### 局限

- 需要重点核查控制频率、推理延迟、传感延迟、失败恢复、硬件磨损、未见场景泛化和安全边界;这些因素决定它能否成为可靠的 System0 模块。
- 对学习型方法,还要区分训练环境中的鲁棒性和真实部署中的鲁棒性,尤其是接触丰富任务、动态运动任务和人形机器人任务。
## 论文定位

- 研究对象:Tilde: Teleoperation for Dexterous In-Hand Manipulation Learning with a DeltaHand。
- 关键词:adaptive-control、closed-loop-control、灵巧操作、diffusion-policy、模仿学习、in-hand-manipulation、实时控制、robot-generalization、机器人操作、鲁棒控制。
- System0 关联:关注机器人身体级快反应链路中“状态/接触/局部感知 -> 约束/优化/策略 -> 动作”的短闭环,不把问题停留在语言规划或任务分解层。

## 速查卡片

> [!summary] Tilde
> - **论文**: https://www.roboticsproceedings.org/rss20/p128.pdf
> - **arXiv**: http://arxiv.org/abs/2405.18804v2
> - **arXiv HTML**: https://arxiv.org/html/2405.18804v2
