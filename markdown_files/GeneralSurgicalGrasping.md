---
title: "World Models for General Surgical Grasping"
method_name: "General Surgical Grasping"
authors: ["Hongbin Lin"]
year: 2024
venue: "RSS"
tags: ["contact-reasoning", "robot-manipulation", "robust-control", "reinforcement-learning", "adaptive-control", "robot-generalization", "state-estimation"]
image_source: "online"
arxiv_html: "https://arxiv.org/html/2405.17940v1"
---
# General Surgical Grasping
## 一句话总结

> World Models for General Surgical Grasping 主要落在 [[adaptive-control]]、[[接触推理]]、[[grasping]]、[[policy-learning]] 这一类问题上;从 System0 视角看,它的价值在于把身体状态、接触变化、局部感知或动力学约束接到机器人低层闭环里,让动作生成更稳定、安全、实时。
## 核心贡献

1. 围绕 **World Models for General Surgical Grasping** 建立了一个与 adaptive-control、接触推理、grasping、policy-learning、强化学习、鲁棒控制 相关的机器人问题设定,重点不是高层语义规划,而是身体级闭环中的状态变化、接触变化或控制约束。
2. 方法层面的关键价值在于把感知、动力学、约束优化、策略学习或硬件反馈组织成可执行的低层动作生成机制,适合放进 System0 的快反应模块中理解。
3. 对系统集成而言,这类工作补上了 System1 技能选择与电机级控制之间的一层:它负责把上层目标变成动态可行、可恢复、可安全过滤的动作。
4. 若论文包含真实机器人、接触任务或仿真到真实迁移,核心应看它是否报告了闭环稳定性、扰动恢复、硬件延迟、失败模式和跨场景泛化。
## 问题背景

这篇论文可以放在 System0 的“具身快系统”背景下阅读:机器人需要在很短时间内根据本体状态、接触、力/触觉、局部视觉或动力学预测做出动作,而不是等待慢速规划器重新推理整个任务。adaptive-control、接触推理、grasping、policy-learning、强化学习、鲁棒控制、状态估计 相关问题的难点通常不在“知道要做什么”,而在动作执行过程中状态会快速偏离、接触会突然改变、约束会实时收紧,低层系统必须及时修正。

### System0 关联

精读时优先看三件事:第一,论文使用哪些快速反馈量,例如关节状态、接触状态、力觉、触觉、深度或局部视觉;第二,反馈如何进入 MPC、WBC、策略网络、安全过滤器或恢复控制器;第三,输出是否直接影响机器人下一步动作,而不是只服务离线分析或高层任务规划。
## 方法详解

### System0 视角解析

从 System0 角度看,这篇论文的方法部分应当围绕 adaptive-control、接触推理、grasping、policy-learning、强化学习、鲁棒控制、状态估计、visuomotor-policy 来读。核心问题是:系统如何从快速变化的机器人状态中构造控制输入,如何处理接触、碰撞、约束、模型误差或执行延迟,以及如何把这些信息变成下一时刻的动作。

如果方法里有优化问题,重点看目标函数、约束和求解频率;如果方法里有学习策略,重点看观测空间、动作空间、训练扰动和部署时的闭环频率;如果方法里有硬件或传感器,重点看传感链路是否真的进入控制回路。这样读可以把论文和 System2 语言规划、System1 技能编排区分开来。

### 关键公式与机制

#### 公式 1: [[动力学或策略机制]]

$$\mathbb{E}_{\pi}[\sum_{i=t}^{T}\gamma^{i-t}r_{i}]$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 2: [[动力学或策略机制]]

$$M=\{m_{i}\in\mathbb{R}^{600\times 600}\}_{i=1}^{K}$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 3: [[动力学或策略机制]]

$$M^{\prime}=\{m_{i}^{{}^{\prime}}\in\mathbb{R}^{600\times 600}\}_{i}^{2}\subset M$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 4: [[动力学或策略机制]]

$$M_{s}=\{m_{s_{i}}\in\mathbb{R}^{600\times 600}\}_{i=1}^{2}$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 5: [[动力学或策略机制]]

$$V=\{v_{i}\in\mathbb{R}^{600\times 600}\}_{i=1}^{2}$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 6: [[动力学或策略机制]]

$$a_{t}^{c}=\begin{cases}a_{t},&t\geq H_{clutch}\\ a_{idle},&t<H_{clutch},\end{cases}$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 7: [[优化目标/约束]]

$$=\mathop{\arg\min}_{m_{j}^{gt}\in M^{gt}}\ \ \mathrm{C}(m_{i}^{e},m_{j}^{gt}).$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 8: [[动力学或策略机制]]

$$f_{zoom}(\sum_{i=1}^{|M^{{}^{\prime}}|}m_{i}^{{}^{\prime}}\odot\hat{I}^{d},m_{zoom}),$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 9: [[动力学或策略机制]]

$$f_{zoom}(\sum_{i=1}^{|M^{{}^{\prime}}|}m_{i}^{{}^{\prime}}\odot v_{i},m_{zoom}),$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 10: [[动力学或策略机制]]

$$\sum_{i=1}^{|M_{s}|}m_{s_{i}}\odot v_{i}+m_{zoom}\odot v_{zoom},$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。
## 关键图表

### Figure 1: Schematic illustration and real-world visualization of our uncertainty-aware depth est

![Figure 1](https://arxiv.org/html/2405.17940v1/extracted/5625690/fig/depth_estimation.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Schematic illustration and real-world visualization of our uncertainty-aware depth est”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 2: Pipeline of Dynamic Spotlight Adaptation (DSA) for visual representation of world mode

![Figure 2](https://arxiv.org/html/2405.17940v1/extracted/5625690/fig/dsa.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Pipeline of Dynamic Spotlight Adaptation (DSA) for visual representation of world mode”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 3: Schematic illustration for virtual clutch (VC). (a) We show the first 5 timesteps of a

![Figure 3](https://arxiv.org/html/2405.17940v1/extracted/5625690/fig/virtual_clutch.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Schematic illustration for virtual clutch (VC). (a) We show the first 5 timesteps of a”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### 论文表格

#### Table 1: Table 1: Visual Masks for Surgical Grasping

| Class ID | Category | Attach Marker | 10mm Depth Accuracy |
| --- | --- | --- | --- |
| Target | Target Object | ✕ | ✓ |
| Gripper Base | Gripper | ✓ | ✓ |
| Gripper Tip | Gripper | ✓ | ✕ |

**说明**: Table 1: Visual Masks for Surgical Grasping

#### Table 2: Table 2: Value/Range for Domain Randomization

| Type | Name | Value/Range |
| --- | --- | --- |
| Camera Pose | Roll Angle | [- 11 o, 11 o ] 11 o 11 o [-11^{o},11^{o}] [- 11, 11 ] |
| Pitch Angle | [35 o, 55 o ] 35 o 55 o [35^{o},55^{o}] [35, 55 ] | |
| Yaw Angle | [- 5 o, 5 o ] 5 o 5 o [-5^{o},5^{o}] [- 5, 5 ] | |
| Target Location | [- 10 m m, 10 m m ] 10 m m 10 m m [-10mm,10mm] [- 10, 10 ] | |
| Target Distance | [250 m m, 350 m m ] 250 m m 350 m m [250mm,350mm] [250, 350 ] | |
| Depth Range | [90 m m, 110 m m ] 90 m m 110 m m [90mm,110mm] [90, 110 ] | |
| Depth Center | [290 m m, 310 m m ] 290 m m 310 m m [290mm,310mm] [290, 310 ] | |
| Dynamic Noise Ratio | [- 0.1, 0.1 ] 0.1 0.1 [-0.1,0.1] [- 0.1, 0.1 ] | |
| Target Disturbance | Dynamic Noise Ratio | [- 0.01, 0.01 ] 0.01 0.01 [-0.01,0.01] [- 0.01, 0.01 ] |
| Gripper Noise | Dynamic Noise Ratio | [- 0.04, 0.04 ] 0.04 0.04 [-0.04,0.04] [- 0.04, 0.04 ] |
| Image Noise | SnP Amount | 0.05 0.05 0.05 0.05 |
| SnP Balance | 0.5 0.5 0.5 0.5 | |
| Gaussian Blur Kernel Size | 3 3 3 3 | |
| Gaussian Blur Sigma | 0.8 0.8 0.8 0.8 | |
| RGB-D Cutout Amount | [0.0, 0.46 ] 0.0 0.46 [0.0,0.46] [0.0, 0.46 ] | |

**说明**: Table 2: Value/Range for Domain Randomization

#### Table 3: Table 3: Evaluating 40 Rollouts in Simulation Over 3 Training Seeds

| | GAS(Ours) | GAS-RawVR | GAS-NoDE | GAS-NoClutch | GAS-NoDR | PPO [31 ] | DreamerV2 [43 ] | | | | | | | | |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| | Target Object | SR | Score | SR | Score | SR | Score | SR | Score | SR | Score | SR | Score | SR | Score |
| | Needle | 86 $\pm$ 4 | 52 $\pm$ 24 | 0 ± 0 plus-or-minus 0 0 0\pm 0 0 ± 0 | 0 ± 0 plus-or-minus 0 0 0\pm 0 0 ± 0 | 0 ± 0 plus-or-minus 0 0 0\pm 0 0 ± 0 | 0 ± 0 plus-or-minus 0 0 0\pm 0 0 ± 0 | 32 ± 32 plus-or-minus 32 32 32\pm 32 32 ± 32 | 17 $\pm$ 28 | 78 $\pm$ 3 | 47 $\pm$ 31 | 0 $\pm$ 0 | 0 $\pm$ 0 | 0 $\pm$ 0 | 0 $\pm$ 0 |
| | box&rod | 89 $\pm$ 7 | 60 $\pm$ 18 | 0 ± 0 plus-or-minus 0 0 0\pm 0 0 ± 0 | 0 ± 0 plus-or-minus 0 0 0\pm 0 0 ± 0 | 0 ± 0 plus-or-minus 0 0 0\pm 0 0 ± 0 | 0 ± 0 plus-or-minus 0 0 0\pm 0 0 ± 0 | 40 $\pm$ 40 | 26 $\pm$ 32 | 85 ± 4 plus-or-minus 85 4 85\pm 4 85 ± 4 | 51 $\pm$ 24 | 0 $\pm$ 0 | 0 $\pm$ 0 | 0 $\pm$ 0 | 0 $\pm$ 0 |
| | Aggregate | 87 $\pm$ 6 | 56 $\pm$ 21 | 0 ± 0 plus-or-minus 0 0 0\pm 0 0 ± 0 | 0 ± 0 plus-or-minus 0 0 0\pm 0 0 ± 0 | 0 ± 0 plus-or-minus 0 0 0\pm 0 0 ± 0 | 0 ± 0 plus-or-minus 0 0 0\pm 0 0 ± 0 | 36 $\pm$ 36 | 22 $\pm$ 31 | 84 $\pm$ 6 | 49 $\pm$ 28 | 0 $\pm$ 0 | 0 $\pm$ 0 | 0 $\pm$ 0 | 0 $\pm$ 0 |

**说明**: Table 3: Evaluating 40 Rollouts in Simulation Over 3 Training Seeds

#### Table 4: Table 4: Studies of Performance, General and Robustness for GAS in Real Robot

| | Type | Description | Episode | SR (%) | Score (%) |
| --- | --- | --- | --- | --- | --- |
| Performance | Phantom | Rectangle | 40 | 75 | 53 |
| Phantom | Liver | 40 | 70 | 45 | |
| Generality | Needle | 20mm Needle | 20 | 75 | 53 |
| Thread | Long Thread | 10 | 70 | 37 | |
| Short Thread | 10 | 70 | 37 | | |
| Aggregate | 20 | 70 | 37 | | |
| Gauze | - | 20 | 80 | 54 | |
| Sponge | - | 20 | 60 | 33 | |
| Fragment | Raisin | 20 | 60 | 39 | |
| Gripper | - | 20 | 70 | 39 | |
| | Aggregate | - | 120 | 69 | 43 |
| Robustness | Background | Background 2 | 10 | 70 | 37 |
| Background 3 | 10 | 80 | 32 | | |
| Aggregate | 20 | 75 | 35 | | |
| Target Noise | Intermittence | 10 | 70 | 44 | |
| Back&Forth | 10 | 90 | 48 | | |
| Aggregate | 20 | 80 | 46 | | |
| Camera Pose | Random Pose | 20 | 75 | 37 | |
| Action Noise | - | 20 | 75 | 43 | |
| Image Noise | - | 20 | 75 | 35 | |
| Re-grasping | - | 20 | 55 | 12 | |

**说明**: Table 4: Studies of Performance, General and Robustness for GAS in Real Robot

#### Table 5: Table 5: Hyperparameter for GAS

| Type | Name | Symbol | Value |
| --- | --- | --- | --- |
| DreamerV2 | Prefill Demonstration Steps | - | 3  10 3 3 10 3 3\times 10^{3} 3  10 3 |
| Prefill Random Steps | - | 7  10 3 7 10 3 7\times 10^{3} 7  10 3 | |
| Replay Buffer Size | - | 5  10 5 5 10 5 5\times 10^{5} 5  10 5 | |
| Training Steps | - | 1.8  10 6 1.8 10 6 1.8\times 10^{6} 1.8  10 6 | |
| Timelimit | H | 300 300 300 300 | |
| Depth Estimation | Empirical Size of Gripper Tip | b i b i b_{i} | 7 7 7 7 mm |
| DSA | Square Size Ratio |  s  s \alpha_{s} | 0.15 0.15 0.15 0.15 |
| Zoom-In Size Ratio |  z o o m  z o o m \alpha_{zoom} | 0.3 0.3 0.3 0.3 | |
| Image Size of Scalar Encoding | - | 6  6 6 6 6\times 6 6  6 | |
| VC | Timesteps of Closing Clutch | H c l u t c h H c l u t c h H_{clutch} | 6 |
| Sparse Rewards | Successful Termination | - | 1 1 1 1 |
| Failed Termination | - | - 0.1 0.1 -0.1 - 0.1 | |
| Normal Progress | - | - 0.001 0.001 -0.001 - 0.001 | |
| Abnormal Progress 1 | - | - 0.01 0.01 -0.01 - 0.01 | |
| Abnormal Progress 2 | - | - 0.01 0.01 -0.01 - 0.01 | |
| Abnormal Progress 3 | - | - 0.05 0.05 -0.05 - 0.05 | |

**说明**: Table 5: Hyperparameter for GAS
## 实验解读

- 评价重点:围绕 adaptive-control、接触推理、grasping、policy-learning、强化学习,优先看真实机器人结果、扰动恢复、接触稳定性、控制误差、成功率、实时性和 sim-to-real 差距。
- System0 视角:实验结论不只看最终成功率,还要看闭环过程中是否能处理状态估计误差、执行延迟、碰撞/接触突变和外部扰动。
- 与上层系统的关系:如果方法能在快闭环内稳定工作,它可以作为 System1 技能或 System2 规划的底层执行保障;如果只在离线或慢速设定下成立,则还不能直接作为身体级快闭环模块。
## 批判性思考

### 优点

- 论文把 adaptive-control、接触推理、grasping、policy-learning、强化学习 放到机器人身体级闭环里讨论,和只做高层语义推理的工作相比,更接近真实机器人部署时会遇到的状态、动力学和安全约束。
- 若论文同时给出公式、系统结构图、硬件实验或消融结果,这些证据可以帮助判断方法是否真的能支撑毫秒到百毫秒级反应。

### 局限

- 需要重点核查控制频率、推理延迟、传感延迟、失败恢复、硬件磨损、未见场景泛化和安全边界;这些因素决定它能否成为可靠的 System0 模块。
- 对学习型方法,还要区分训练环境中的鲁棒性和真实部署中的鲁棒性,尤其是接触丰富任务、动态运动任务和人形机器人任务。
## 论文定位

- 研究对象:World Models for General Surgical Grasping。
- 关键词:adaptive-control、接触推理、grasping、policy-learning、强化学习、鲁棒控制、状态估计、visuomotor-policy、world-model。
- System0 关联:关注机器人身体级快反应链路中“状态/接触/局部感知 -> 约束/优化/策略 -> 动作”的短闭环,不把问题停留在语言规划或任务分解层。

## 速查卡片

> [!summary] General Surgical Grasping
> - **论文**: https://www.roboticsproceedings.org/rss20/p041.pdf
> - **arXiv**: http://arxiv.org/abs/2405.17940v1
> - **arXiv HTML**: https://arxiv.org/html/2405.17940v1
