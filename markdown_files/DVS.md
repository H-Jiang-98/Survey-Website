---
title: "Demonstrating DVS: Dynamic Virtual-Real Simulation Platform for Mobile Robotic Tasks"
method_name: "Demonstrating DVS"
authors: ["Zijie Zheng"]
year: 2025
venue: "RSS"
tags: ["robot-manipulation", "real-time-control", "legged-locomotion", "robot-generalization", "closed-loop-control", "loco-manipulation"]
image_source: "online"
arxiv_html: "https://arxiv.org/html/2504.18944v1"
---
# Demonstrating DVS
## 一句话总结

> Demonstrating DVS: Dynamic Virtual-Real Simulation Platform for Mobile Robotic Tasks 主要落在 [[benchmark-dataset]]、[[closed-loop-control]]、[[grasping]]、[[足式运动]] 这一类问题上;从 System0 视角看,它的价值在于把身体状态、接触变化、局部感知或动力学约束接到机器人低层闭环里,让动作生成更稳定、安全、实时。
## 核心贡献

1. 围绕 **Demonstrating DVS: Dynamic Virtual-Real Simulation Platform for Mobile Robotic Tasks** 建立了一个与 benchmark-dataset、closed-loop-control、grasping、足式运动、移动操作、mobile-manipulation 相关的机器人问题设定,重点不是高层语义规划,而是身体级闭环中的状态变化、接触变化或控制约束。
2. 方法层面的关键价值在于把感知、动力学、约束优化、策略学习或硬件反馈组织成可执行的低层动作生成机制,适合放进 System0 的快反应模块中理解。
3. 对系统集成而言,这类工作补上了 System1 技能选择与电机级控制之间的一层:它负责把上层目标变成动态可行、可恢复、可安全过滤的动作。
4. 若论文包含真实机器人、接触任务或仿真到真实迁移,核心应看它是否报告了闭环稳定性、扰动恢复、硬件延迟、失败模式和跨场景泛化。
## 问题背景

这篇论文可以放在 System0 的“具身快系统”背景下阅读:机器人需要在很短时间内根据本体状态、接触、力/触觉、局部视觉或动力学预测做出动作,而不是等待慢速规划器重新推理整个任务。benchmark-dataset、closed-loop-control、grasping、足式运动、移动操作、mobile-manipulation、实时控制 相关问题的难点通常不在“知道要做什么”,而在动作执行过程中状态会快速偏离、接触会突然改变、约束会实时收紧,低层系统必须及时修正。

### System0 关联

精读时优先看三件事:第一,论文使用哪些快速反馈量,例如关节状态、接触状态、力觉、触觉、深度或局部视觉;第二,反馈如何进入 MPC、WBC、策略网络、安全过滤器或恢复控制器;第三,输出是否直接影响机器人下一步动作,而不是只服务离线分析或高层任务规划。
## 方法详解

### System0 视角解析

从 System0 角度看,这篇论文的方法部分应当围绕 benchmark-dataset、closed-loop-control、grasping、足式运动、移动操作、mobile-manipulation、实时控制、机器人操作 来读。核心问题是:系统如何从快速变化的机器人状态中构造控制输入,如何处理接触、碰撞、约束、模型误差或执行延迟,以及如何把这些信息变成下一时刻的动作。

如果方法里有优化问题,重点看目标函数、约束和求解频率;如果方法里有学习策略,重点看观测空间、动作空间、训练扰动和部署时的闭环频率;如果方法里有硬件或传感器,重点看传感链路是否真的进入控制回路。这样读可以把论文和 System2 语言规划、System1 技能编排区分开来。

### 关键公式与机制

#### 公式 1: [[动力学或策略机制]]

$$T_{\text{virtual}}=R\cdot T_{\text{real}}+t$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。
## 关键图表

### Figure 1: Virtual-Real Data Synchronization Framework. The central demonstrates the synchroniza

![Figure 1](https://arxiv.org/html/2504.18944v1/x2.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Virtual-Real Data Synchronization Framework. The central demonstrates the synchroniza”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 2: The robotic arm is interrupted while executing Prompt A and is requested to execute P

![Figure 2](https://arxiv.org/html/2504.18944v1/x3.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“The robotic arm is interrupted while executing Prompt A and is requested to execute P”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 3: Visualization of pedestrian trajectory prediction, where each color represents a diff

![Figure 3](https://arxiv.org/html/2504.18944v1/x4.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Visualization of pedestrian trajectory prediction, where each color represents a diff”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### 论文表格

#### Table 1: TABLE I: Comparison of Simulation Platforms. For the sensor, S refers to semantic, L refers to Lidar

| Simulation Platform | Sensors | Dynamic Scenes | VR Interaction | ROS | |
| --- | --- | --- | --- | --- | --- |
| | Pedestrians | Objects | | | |
| Arena [12 ] | RGB-D, L | ✓ ✓ \checkmark ✓ |  \times  |  \times  | ✓ ✓ \checkmark ✓ |
| AI2THOR [15 ] | RGB-D, S |  \times  |  \times  |  \times  |  \times  |
| Gibson series [17 ] [34 ] | RGB-D, S, L |  \times  |  \times  | ✓ ✓ \checkmark ✓ | ✓ ✓ \checkmark ✓ |
| HoME [1 ] | RGB-D, S |  \times  |  \times  |  \times  |  \times  |
| Habitat [33 ] [36 ] [31 ] | RGB-D, S |  \times  |  \times  | ✓ ✓ \checkmark ✓ |  \times  |
| SAPIEN [42 ] | RGB-D, S |  \times  |  \times  | ✓ ✓ \checkmark ✓ |  \times  |
| ThreeDWorld [7 ] | RGB-D, S |  \times  |  \times  | ✓ ✓ \checkmark ✓ |  \times  |
| VirtualHome [30 ] | RGB-D, S |  \times  |  \times  |  \times  |  \times  |
| DVS(Ours) | RGB-D, S, L | ✓ ✓ \checkmark ✓ | ✓ ✓ \checkmark ✓ | ✓ ✓ \checkmark ✓ | ✓ ✓ \checkmark ✓ |

**说明**: TABLE I: Comparison of Simulation Platforms. For the sensor, S refers to semantic, L refers to Lidar

#### Table 2: TABLE II: Camera Trajectories With and Without Smoothing

| Scene | Trajectory Type | Avg. Features ↑ ↑ \uparrow ↑ |
| --- | --- | --- |
| Bedroom | Straight | 751.35 |
| Smooth | 1484.29 | |
| Livingroom | Straight | 1190.47 |
| Smooth | 1631.73 | |

**说明**: TABLE II: Camera Trajectories With and Without Smoothing

#### Table 3: TABLE III: Task Success Rates by Module and Prompt Order

| Module | Prompt Order | Success Rate (%) | | |
| --- | --- | --- | --- | --- |
| First | Second | First Task | Second Task | |
| OpenVLA-7B | A | B | 0.0 | 100.0 |
| B | A | 0.0 | 90.0 | |
| RDT-1B | A | B | 0.0 | 80.0 |
| B | A | 0.0 | 90.0 | |

**说明**: TABLE III: Task Success Rates by Module and Prompt Order

#### Table 4: TABLE IV: Comparative with Different Finetuning Data

| Task | Trials | Finetune Data | Successes |
| --- | --- | --- | --- |
| Pick the apple and place at the target point | 10 | virtual | 6 |
| 10 | virtual-real | 9 | |
| Pick the banana and place at the target point | 10 | virtual | 4 |
| 10 | virtual-real | 8 | |

**说明**: TABLE IV: Comparative with Different Finetuning Data

#### Table 5: TABLE V: Experiments on Pedestrian Trajectory Prediction. Gym, Office and Supermarket are our synthetic indoor scenes,

| Scene | Method | ADE ↓ ↓ \downarrow ↓ | FDE ↓ ↓ \downarrow ↓ |
| --- | --- | --- | --- |
| Gym | STGAT | 1.39 | 3.01 |
| Trajectron++ | 0.59 | 1.02 | |
| TUTR | 0.70 | 1.19 | |
| Office | STGAT | 1.38 | 2.75 |
| Trajectron++ | 0.89 | 1.60 | |
| TUTR | 0.81 | 1.40 | |
| Supermarket | STGAT | 1.42 | 2.88 |
| Trajectron++ | 0.96 | 1.82 | |
| TUTR | 0.83 | 1.50 | |
| ETH | STGAT | 0.79 | 1.48 |
| Trajectron++ | 0.52 | 0.97 | |
| TUTR | 0.43 | 0.83 | |

**说明**: TABLE V: Experiments on Pedestrian Trajectory Prediction. Gym, Office and Supermarket are our synthetic indoor scenes, while ETH [29 ] is the official public outdoor dataset.

#### Table 6: TABLE VI: Experiments on Social Navigation

| Scene | Metric | HumanNumber=10 / 15 / 20 | | |
| --- | --- | --- | --- | --- |
| ORCA | DS-RNN | AttnGraph | | |
| Restaurant | SuccessRate ↑ ↑ \uparrow ↑ | 0.78 / 0.74 / 0.62 | 0.82 / 0.76 / 0.68 | 0.83 / 0.77 / 0.67 |
| CollisionRate ↓ ↓ \downarrow ↓ | 0.01 / 0.06 / 0.08 | 0.01 / 0.05 / 0.06 | 0.02 / 0.06 / 0.07 | |
| NavigationTime ↓ ↓ \downarrow ↓ | 42.50 / 43.59 / 43.90 | 37.48 / 44.96 / 45.24 | 39.68 / 44.81 / 49.65 | |
| Store | SuccessRate ↑ ↑ \uparrow ↑ | 0.96 / 0.85 / 0.51 | 0.98 / 0.81 / 0.75 | 0.98 / 0.87 / 0.79 |
| CollisionRate ↓ ↓ \downarrow ↓ | 0.03 / 0.04 / 0.06 | 0.01 / 0.07 / 0.09 | 0.02 / 0.04 / 0.06 | |
| NavigationTime ↓ ↓ \downarrow ↓ | 40.39 / 47.62 / 46.47 | 34.21 / 39.51 / 39.97 | 41.56 / 43.75 / 48.22 | |

**说明**: TABLE VI: Experiments on Social Navigation
## 实验解读

- 评价重点:围绕 benchmark-dataset、closed-loop-control、grasping、足式运动、移动操作,优先看真实机器人结果、扰动恢复、接触稳定性、控制误差、成功率、实时性和 sim-to-real 差距。
- System0 视角:实验结论不只看最终成功率,还要看闭环过程中是否能处理状态估计误差、执行延迟、碰撞/接触突变和外部扰动。
- 与上层系统的关系:如果方法能在快闭环内稳定工作,它可以作为 System1 技能或 System2 规划的底层执行保障;如果只在离线或慢速设定下成立,则还不能直接作为身体级快闭环模块。
## 批判性思考

### 优点

- 论文把 benchmark-dataset、closed-loop-control、grasping、足式运动、移动操作 放到机器人身体级闭环里讨论,和只做高层语义推理的工作相比,更接近真实机器人部署时会遇到的状态、动力学和安全约束。
- 若论文同时给出公式、系统结构图、硬件实验或消融结果,这些证据可以帮助判断方法是否真的能支撑毫秒到百毫秒级反应。

### 局限

- 需要重点核查控制频率、推理延迟、传感延迟、失败恢复、硬件磨损、未见场景泛化和安全边界;这些因素决定它能否成为可靠的 System0 模块。
- 对学习型方法,还要区分训练环境中的鲁棒性和真实部署中的鲁棒性,尤其是接触丰富任务、动态运动任务和人形机器人任务。
## 论文定位

- 研究对象:Demonstrating DVS: Dynamic Virtual-Real Simulation Platform for Mobile Robotic Tasks。
- 关键词:benchmark-dataset、closed-loop-control、grasping、足式运动、移动操作、mobile-manipulation、实时控制、机器人操作、scalable-robot-learning。
- System0 关联:关注机器人身体级快反应链路中“状态/接触/局部感知 -> 约束/优化/策略 -> 动作”的短闭环,不把问题停留在语言规划或任务分解层。

## 速查卡片

> [!summary] Demonstrating DVS
> - **论文**: https://www.roboticsproceedings.org/rss21/p129.pdf
> - **arXiv**: http://arxiv.org/abs/2504.18944v1
> - **arXiv HTML**: https://arxiv.org/html/2504.18944v1
