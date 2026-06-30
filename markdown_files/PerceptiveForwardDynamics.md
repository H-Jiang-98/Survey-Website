---
title: "Learned Perceptive Forward Dynamics Model for Safe and Platform-aware Robotic Navigation"
method_name: "Perceptive Forward Dynamics"
authors: ["Pascal Roth"]
year: 2025
venue: "RSS"
tags: ["robust-control", "real-time-control", "legged-locomotion", "safe-control", "robot-generalization", "sim-to-real", "state-estimation", "navigation"]
image_source: "online"
arxiv_html: "https://arxiv.org/html/2504.19322v2"
---
# Perceptive Forward Dynamics
## 一句话总结

> Learned Perceptive Forward Dynamics Model for Safe and Platform-aware Robotic Navigation 主要落在 [[dynamics-modeling]]、[[足式运动]]、[[navigation]]、[[proprioception]] 这一类问题上;从 System0 视角看,它的价值在于把身体状态、接触变化、局部感知或动力学约束接到机器人低层闭环里,让动作生成更稳定、安全、实时。
## 核心贡献

1. 围绕 **Learned Perceptive Forward Dynamics Model for Safe and Platform-aware Robotic Navigation** 建立了一个与 dynamics-modeling、足式运动、navigation、proprioception、实时控制、robot-generalization 相关的机器人问题设定,重点不是高层语义规划,而是身体级闭环中的状态变化、接触变化或控制约束。
2. 方法层面的关键价值在于把感知、动力学、约束优化、策略学习或硬件反馈组织成可执行的低层动作生成机制,适合放进 System0 的快反应模块中理解。
3. 对系统集成而言,这类工作补上了 System1 技能选择与电机级控制之间的一层:它负责把上层目标变成动态可行、可恢复、可安全过滤的动作。
4. 若论文包含真实机器人、接触任务或仿真到真实迁移,核心应看它是否报告了闭环稳定性、扰动恢复、硬件延迟、失败模式和跨场景泛化。
## 问题背景

这篇论文可以放在 System0 的“具身快系统”背景下阅读:机器人需要在很短时间内根据本体状态、接触、力/触觉、局部视觉或动力学预测做出动作,而不是等待慢速规划器重新推理整个任务。dynamics-modeling、足式运动、navigation、proprioception、实时控制、robot-generalization、safe-control 相关问题的难点通常不在“知道要做什么”,而在动作执行过程中状态会快速偏离、接触会突然改变、约束会实时收紧,低层系统必须及时修正。

### System0 关联

精读时优先看三件事:第一,论文使用哪些快速反馈量,例如关节状态、接触状态、力觉、触觉、深度或局部视觉;第二,反馈如何进入 MPC、WBC、策略网络、安全过滤器或恢复控制器;第三,输出是否直接影响机器人下一步动作,而不是只服务离线分析或高层任务规划。
## 方法详解

### System0 视角解析

从 System0 角度看,这篇论文的方法部分应当围绕 dynamics-modeling、足式运动、navigation、proprioception、实时控制、robot-generalization、safe-control、scalable-robot-learning 来读。核心问题是:系统如何从快速变化的机器人状态中构造控制输入,如何处理接触、碰撞、约束、模型误差或执行延迟,以及如何把这些信息变成下一时刻的动作。

如果方法里有优化问题,重点看目标函数、约束和求解频率;如果方法里有学习策略,重点看观测空间、动作空间、训练扰动和部署时的闭环频率;如果方法里有硬件或传感器,重点看传感链路是否真的进入控制回路。这样读可以把论文和 System2 语言规划、System1 技能编排区分开来。

### 关键公式与机制

#### 公式 1: [[动力学或策略机制]]

$$\mathbf{m}^{1,\dots,7}=m_{t,\dots,t-n}^{1,\dots,7}$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 2: [[优化目标/约束]]

$$\theta^{*}=\arg\min_{\theta}\left(\mathcal{L}_{pose}+\mathcal{L}_{risk}\right),$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 3: [[优化目标/约束]]

$$\mathcal{R}_{risk}(\mathbf{r})=\sum_{i}^{q}\mathbf{r}^{i}\cdot\begin{cases}\lambda_{risk}&\text{if}\quad\exists r_{t}\in\mathbf{r}^{i},\;r_{t}>\delta_{risk}\\ 0&\text{else}\end{cases}$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 4: [[优化目标/约束]]

$$w_{i}=\frac{\exp\Bigl(\frac{1}{\gamma}(\mathcal{R}_{i}-\mathcal{R}_{\max})\Bigr)}{\sum_{j=1}^{C}\exp\Bigl(\frac{1}{\gamma}(\mathcal{R}_{j}-\mathcal{R}_{\max})\Bigr)},$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 5: [[优化目标/约束]]

$$\mathcal{R}_{pose}(p_{t+n},g)=\begin{cases}\lVert p_{t+n}-g \rVert_{2}\cdot\lambda_{pull},&\text{if}\lVert p_{t+n}-g \rVert_{2}<\delta_{pose}\\ \lVert p_{t+n}-g \rVert_{2},&\text{else}.\end{cases}$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 6: [[优化目标/约束]]

$$=\arg\max_{i\in[1,C]}\mathcal{R}(\tilde{f}(\mathbf{a}^{i},o_{t}),g)$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 7: [[优化目标/约束]]

$$=\arg\max_{i\in[1,C]}\mathcal{R}(\mathbf{\tilde{s}}^{i},g)$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 8: [[动力学或策略机制]]

$$\bar{\mathbf{a}}\leftarrow\bar{\mathbf{a}}+\sum_{i=1}^{C}w_{i}\delta\mathbf{a}_{i},$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 9: [[优化目标/约束]]

$$\mathcal{R}=\lambda_{pose}\cdot\mathcal{R}_{pose}+\lambda_{risk}\cdot\mathcal{R}_{risk}$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 10: [[优化目标/约束]]

$$\mathcal{L}=\epsilon_{pose}\cdot\mathcal{L}_{pose}+\epsilon_{risk}\cdot\mathcal{L}_{risk}+\epsilon_{stop}\cdot\mathcal{L}_{stop},$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。
## 关键图表

### Figure 1: Overview of the FDM training. Data is collected in a parallelized simulation setting

![Figure 1](https://arxiv.org/html/2504.19322v2/x1.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Overview of the FDM training. Data is collected in a parallelized simulation setting”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 2: Demonstration of environment- and platform-aware state predictions using the presente

![Figure 2](https://arxiv.org/html/2504.19322v2/x3.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Demonstration of environment- and platform-aware state predictions using the presente”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 3: Comparison of position error at the final prediction step across different environme

![Figure 3](https://arxiv.org/html/2504.19322v2/x15.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Comparison of position error at the final prediction step across different environme”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### 论文表格

#### Table 1: TABLE I: The observation space of the FDM combines proprioceptive information of the robot state m 2 ⋯ 4 superscrip

| # | Observation | Dimensions | Augmentation |
| --- | --- | --- | --- |
| m 1 m 1 m^{1} 1 | Twist Commands | R 3 R 3  $\mathbb{R}^{3} 3$ | - |
| m 2 m 2 m^{2} 2 | Projected Gravity | R 3 R 3  $\mathbb{R}^{3} 3$ | U [- 0.05, 0.05 ] U 0.05 0.05  $\mathcal{U}[-0.05,0.05] [- 0.05, 0.05 ]$ |
| m 3 m 3 m^{3} 3 | Base linear velocities | R 3 R 3  $\mathbb{R}^{3} 3$ | U [- 0.1, 0.1 ] U 0.1 0.1  $\mathcal{U}[-0.1,0.1] [- 0.1, 0.1 ]$ |
| m 4 m 4 m^{4} 4 | Base angular velocities | R 3 R 3  $\mathbb{R}^{3} 3$ | U [- 0.2, 0.2 ] U 0.2 0.2  $\mathcal{U}[-0.2,0.2] [- 0.2, 0.2 ]$ |
| m 5 m 5 m^{5} 5 | Joint positions | R b R b  $\mathbb{R}^{b}$ | U [- 0.01, 0.01 ] U 0.01 0.01  $\mathcal{U}[-0.01,0.01] [- 0.01, 0.01 ]$ |
| m 6 m 6 m^{6} 6 | Joint velocities | R b R b  $\mathbb{R}^{b}$ | U [- 1.5, 1.5 ] U 1.5 1.5  $\mathcal{U}[-1.5,1.5] [- 1.5, 1.5 ]$ |
| m 7 m 7 m^{7} 7 | Last Two Joint actions | R 2 b R 2 b  $\mathbb{R}^{2b} 2$ | - |
| h h h | Height Map | $\mathbb{R}^{u\times v} ×$ | U [- 0.1, 0.1 ] U 0.1 0.1  $\mathcal{U}[-0.1,0.1] [- 0.1, 0.1 ]$ |

**说明**: 该表格来自论文中的公式、奖励项或实验设置；为避免 PDF 抽取噪声，网页中仅保留可渲染的核心符号和数值。

#### Table 2: TABLE II: Comparison between the developed FDM, a perceptive FDM using a 2 D 2 D 2D 2 LiDAR by Kim et al. [| Env. | Method | Pos. Offset | Precision | Recall | Accuracy | F1 Score |
| --- | --- | --- | --- | --- | --- | --- |
| Plane | Constant Vel. | 0.32 $\pm$ 0.24 | - | - | - | - |
| | Kim et al. | 0.45 $\pm$ 0.33 | 10.84 | 90.47 | 92.92 | 0.17 |
| | Ours | 0.13 $\pm$ 0.19 | 12.73 | 9.83 | 98.32 | 0.10 |
| 2D | Constant Vel. | 1.33 $\pm$ 1.17 | - | - | - | - |
| | Kim et al. | 0.37 $\pm$ 0.41 | 80.63 | 92.68 | 86.47 | 0.86 |
| | Ours | 0.28 $\pm$ 0.34 | 93.42 | 86.57 | 89.13 | 0.90 |
| 2D-3D | Constant Vel. | 1.08 $\pm$ 1.11 | - | - | - | - |
| | Kim et al. | 0.45 $\pm$ 0.45 | 70.11 | 88.97 | 83.61 | 0.78 |
| | Ours | 0.30 $\pm$ 0.37 | 83.09 | 87.25 | 89.20 | 0.85 |
| 3D | Constant Vel. | 0.99 $\pm$ 1.06 | - | - | - | - |
| | Kim et al. | 0.44 $\pm$ 0.45 | 74.05 | 86.62 | 86.35 | 0.80 |
| | Ours | 0.28 $\pm$ 0.35 | 83.58 | 86.51 | 90.61 | 0.85 |

#### Table 3: TABLE III: Comparison of planning methods in 2 D 2 D 2D 2 and 3 D 3 D 3D 3 environments, evaluati

| Env. | Method | Success (%) | MPL (m) | MPT (s) | | |
| --- | --- | --- | --- | --- | --- | --- |
| | | | Suc. | All | Suc. | All |
| 2D | MPPI Ours | 88.33 | 4.28 | 4.02 | 9.23 | 8.92 |
| MPPI Kim et al. [5 ] | 78.33 | 9.98 | 16.23 | 25.01 | 41.88 | |
| MPPI Heuristics [30 ] | 82.50 | 4.73 | 4.79 | 11.12 | 12.68 | |
| 3D | MPPI Ours | 73.75 | 3.93 | 3.69 | 8.68 | 9.67 |
| MPPI Kim et al. [5 ] | 48.75 | 7.20 | 13.26 | 18.60 | 35.45 | |
| MPPI Heuristics [30 ] | 33.13 | 4.41 | 3.85 | 13.99 | 17.17 | |

#### Table 4: TABLE IV: Nomenclature used in this work.

| Symbol | Description | Symbol | Description |
| --- | --- | --- | --- |
| a a a | Action | a ~ ~ a  $\tilde{a}$ | Action correction predicted by FDM |
| a ^ ^ a  $\hat{a}$ | Followed action on the robot | b b b | Robot joint count |
| f f f | FDM function | g g g | Goal pose |
| h h h | Height scan | k k k | MPPI iteration count |
| m m m | Proprioceptive measurements | n n n | FDM number of prediction steps |
| o o o | Observations | p p p | Pose |
| q q q | Set of neighbors for MPPI obstacle cost | r r r | Risk |
| s s s | State | t t t | Time |
| u u u | Height-map width | v v v | Height-map length |
| w w w | Trajectory weight assigned for MPPI update | z z z | Observation probability |
| L L  $\mathcal{L}$ | FDM Loss | S S  $\mathcal{S}$ | Set of states |
| T T  $\mathcal{T}$ | Transition likelihood between states | O O  $\mathcal{O}$ | Set of observations |
| Z Z  $\mathcal{Z}$ | Observation probability | C C  $\mathcal{C}$ | Set of MPPI candidates |
| R R  $\mathcal{R}$ | MPPI Reward functions | U U  $\mathcal{U}$ | Uniform distribution |
| N N  $\mathcal{N}$ | Normal Distribution | | |
|   \beta | Time correlation factor for action sampling |   \sigma | Standard deviation for action sampling |
| $\theta$ | FDM network weights |   \lambda | MPPI reward term weights |
|  \epsilon | FDM loss term weights | $\delta_{risk}$ | Threshold for risky trajectory |
| $\delta_{pose}$ | Threshold to apply pull factor in pose reward | | |
|  t h  t h \Delta t_{h} | Time-step of the history information with frequency 1 /  t h 1  t h 1/\Delta t_{h} 1 / |  t p  t p \Delta t_{p} | Time-step of the forward predictions with frequency 1 /  t p 1  t p 1/\Delta t_{p} 1 / |

**说明**: TABLE IV: Nomenclature used in this work.

#### Table 5: TABLE V: Ablation studies evaluating the impact of removing specific input modalities across multiple environments. Omi

| Env. | Method Variation | Pos. Offset | Precision | Recall | Accuracy | F1 Score |
| --- | --- | --- | --- | --- | --- | --- |
| Plane | Ours | 0.14 $\pm$ 0.16 | 41.69 | 35.23 | 98.88 | 0.37 |
| | Ours W/o State Obs. | 0.14 $\pm$ 0.17 | 30.12 | 26.89 | 99.08 | 0.28 |
| | Ours W/o Proprio. Obs. | 1.22 $\pm$ 0.81 | 2.62 | 81.02 | 83.38 | 0.05 |
| | Ours W/o Height Scan | 0.44 $\pm$ 0.40 | 5.92 | 63.51 | 92.82 | 0.10 |
| 2D | Ours | 0.23 $\pm$ 0.28 | 93.7 | 90.14 | 91.43 | 0.92 |
| | Ours W/o State Obs. | 0.22 $\pm$ 0.27 | 89.38 | 89.23 | 90.93 | 0.89 |
| | Ours W/o Proprio. Obs. | 0.25 $\pm$ 0.32 | 86.05 | 91.00 | 89.77 | 0.88 |
| | Ours W/o Height Scan | 0.48 $\pm$ 0.51 | 89.65 | 68.89 | 82.75 | 0.78 |
| 2D-3D | Ours | 0.26 $\pm$ 0.32 | 93.61 | 91.56 | 93.24 | 0.93 |
| | Ours W/o State Obs. | 0.26 $\pm$ 0.32 | 88.18 | 85.69 | 92.07 | 0.87 |
| | Ours W/o Proprio. Obs. | 0.32 $\pm$ 0.35 | 83.64 | 87.89 | 90.73 | 0.86 |
| | Ours W/o Height Scan | 0.43 $\pm$ 0.43 | 87.35 | 72.36 | 87.39 | 0.79 |
| 3D | Ours | 0.27 $\pm$ 0.32 | 90.28 | 85.9 | 93.14 | 0.88 |
| | Ours W/o State Obs. | 0.27 $\pm$ 0.32 | 89.41 | 85.12 | 92.70 | 0.87 |
| | Ours W/o Proprio. Obs. | 0.28 $\pm$ 0.32 | 85.34 | 87.09 | 92.04 | 0.86 |
| | Ours W/o Height Scan | 0.41 $\pm$ 0.41 | 88.94 | 72.61 | 88.89 | 0.80 |

**说明**: TABLE V: Ablation studies evaluating the impact of removing specific input modalities across multiple environments. Omitting past state information reduces the F1 score across all environments. Removal of proprioceptive inputs or the height scan results in substantial declines in both pose accuracy and failure prediction, the importance of these components.

#### Table 6: TABLE VI: Influence of the risk term when planning in 2 D 2 D 2D 2 and 3 D 3 D 3D 3 environments.

| Env. | Method | Success (%) | MPL (m) | MPT (s) |
| --- | --- | --- | --- | --- |
| 2D | Ours | 88.33 | 4.28 | 9.23 |
| Ours (w/o Failure Estimation) | 85.86 | 4.50 | 7.73 | |
| 3D | Ours | 73.75 | 3.93 | 8.68 |
| Ours (w/o Failure Estimation) | 69.17 | 3.96 | 5.74 | |

**说明**: TABLE VI: Influence of the risk term when planning in 2 D 2 D 2D 2 and 3 D 3 D 3D 3 environments. By including the risk reward term, the success rate improves in both cases. While the mean path length (MPT) is approximately equal, the additional safety requirement leads to longer mean path times (MPT).
## 实验解读

- 评价重点:围绕 dynamics-modeling、足式运动、navigation、proprioception、实时控制,优先看真实机器人结果、扰动恢复、接触稳定性、控制误差、成功率、实时性和 sim-to-real 差距。
- System0 视角:实验结论不只看最终成功率,还要看闭环过程中是否能处理状态估计误差、执行延迟、碰撞/接触突变和外部扰动。
- 与上层系统的关系:如果方法能在快闭环内稳定工作,它可以作为 System1 技能或 System2 规划的底层执行保障;如果只在离线或慢速设定下成立,则还不能直接作为身体级快闭环模块。
## 批判性思考

### 优点

- 论文把 dynamics-modeling、足式运动、navigation、proprioception、实时控制 放到机器人身体级闭环里讨论,和只做高层语义推理的工作相比,更接近真实机器人部署时会遇到的状态、动力学和安全约束。
- 若论文同时给出公式、系统结构图、硬件实验或消融结果,这些证据可以帮助判断方法是否真的能支撑毫秒到百毫秒级反应。

### 局限

- 需要重点核查控制频率、推理延迟、传感延迟、失败恢复、硬件磨损、未见场景泛化和安全边界;这些因素决定它能否成为可靠的 System0 模块。
- 对学习型方法,还要区分训练环境中的鲁棒性和真实部署中的鲁棒性,尤其是接触丰富任务、动态运动任务和人形机器人任务。
## 论文定位

- 研究对象:Learned Perceptive Forward Dynamics Model for Safe and Platform-aware Robotic Navigation。
- 关键词:dynamics-modeling、足式运动、navigation、proprioception、实时控制、robot-generalization、safe-control、scalable-robot-learning、仿真到真实迁移、zero-shot。
- System0 关联:关注机器人身体级快反应链路中“状态/接触/局部感知 -> 约束/优化/策略 -> 动作”的短闭环,不把问题停留在语言规划或任务分解层。

## 速查卡片

> [!summary] Perceptive Forward Dynamics
> - **论文**: https://www.roboticsproceedings.org/rss21/p001.pdf
> - **arXiv**: http://arxiv.org/abs/2504.19322v2
> - **arXiv HTML**: https://arxiv.org/html/2504.19322v2
