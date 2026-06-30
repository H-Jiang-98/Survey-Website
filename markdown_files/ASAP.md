---
title: "ASAP: Aligning Simulation and Real-World Physics for Learning Agile Humanoid Whole-Body Skills"
method_name: "ASAP"
authors: ["Tairan He"]
year: 2025
venue: "RSS"
tags: ["legged-locomotion", "imitation-learning", "robot-generalization", "humanoid", "sim-to-real", "agile-locomotion", "whole-body-control"]
image_source: "online"
arxiv_html: "https://arxiv.org/html/2502.01143v3"
---
# ASAP
## 一句话总结

> ASAP: Aligning Simulation and Real-World Physics for Learning Agile Humanoid Whole-Body Skills 主要落在 [[action-model]]、[[agile-locomotion]]、[[人形机器人]]、[[足式运动]] 这一类问题上;从 System0 视角看,它的价值在于把身体状态、接触变化、局部感知或动力学约束接到机器人低层闭环里,让动作生成更稳定、安全、实时。
## 核心贡献

1. 围绕 **ASAP: Aligning Simulation and Real-World Physics for Learning Agile Humanoid Whole-Body Skills** 建立了一个与 action-model、agile-locomotion、人形机器人、足式运动、运动模仿、motion-tracking 相关的机器人问题设定,重点不是高层语义规划,而是身体级闭环中的状态变化、接触变化或控制约束。
2. 方法层面的关键价值在于把感知、动力学、约束优化、策略学习或硬件反馈组织成可执行的低层动作生成机制,适合放进 System0 的快反应模块中理解。
3. 对系统集成而言,这类工作补上了 System1 技能选择与电机级控制之间的一层:它负责把上层目标变成动态可行、可恢复、可安全过滤的动作。
4. 若论文包含真实机器人、接触任务或仿真到真实迁移,核心应看它是否报告了闭环稳定性、扰动恢复、硬件延迟、失败模式和跨场景泛化。
## 问题背景

这篇论文可以放在 System0 的“具身快系统”背景下阅读:机器人需要在很短时间内根据本体状态、接触、力/触觉、局部视觉或动力学预测做出动作,而不是等待慢速规划器重新推理整个任务。action-model、agile-locomotion、人形机器人、足式运动、运动模仿、motion-tracking、retargeting 相关问题的难点通常不在“知道要做什么”,而在动作执行过程中状态会快速偏离、接触会突然改变、约束会实时收紧,低层系统必须及时修正。

### System0 关联

精读时优先看三件事:第一,论文使用哪些快速反馈量,例如关节状态、接触状态、力觉、触觉、深度或局部视觉;第二,反馈如何进入 MPC、WBC、策略网络、安全过滤器或恢复控制器;第三,输出是否直接影响机器人下一步动作,而不是只服务离线分析或高层任务规划。
## 方法详解

### System0 视角解析

从 System0 角度看,这篇论文的方法部分应当围绕 action-model、agile-locomotion、人形机器人、足式运动、运动模仿、motion-tracking、retargeting、仿真到真实迁移 来读。核心问题是:系统如何从快速变化的机器人状态中构造控制输入,如何处理接触、碰撞、约束、模型误差或执行延迟,以及如何把这些信息变成下一时刻的动作。

如果方法里有优化问题,重点看目标函数、约束和求解频率;如果方法里有学习策略,重点看观测空间、动作空间、训练扰动和部署时的闭环频率;如果方法里有硬件或传感器,重点看传感链路是否真的进入控制回路。这样读可以把论文和 System2 语言规划、System1 技能编排区分开来。

### 关键公式与机制

#### 公式 1: [[优化目标/约束]]

$$r_{t}=\mathcal{R}\left(s_{t}^{\mathrm{p}},s_{t}^{\mathrm{g}}\right)$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 2: [[动力学或策略机制]]

$$\mathbb{E}\left[\sum_{t=1}^{T}\gamma^{t-1}r_{t}\right]$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 3: [[动力学或策略机制]]

$$\Delta a_{t}=\pi^{\Delta}_{\theta}(s_{t},a_{t})$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 4: [[动力学或策略机制]]

$$f^{\text{real}}(s,\pi(s))=f^{\text{sim}}(s,\pi(s)+\pi^{\Delta}(s,\pi(s))).$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 5: [[优化目标/约束]]

$$\mathcal{L}=\bigg{\lVert}s^{\text{real}}_{t+K}-\ {f^{\text{sim}}\big{(}\dots f^{\text{sim}}}_{K}(s_{t},a_{t})+f^{\Delta}_{\theta}(s_{t},a_{t}),\dots,a_{t+K}\big{)}\bigg{\rVert}.$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 6: [[优化目标/约束]]

$$\mathcal{D}^{\text{r}}=\{s^{\text{r}}_{0},a^{\text{r}}_{0},\dots,s^{\text{r}}_{T},a^{\text{r}}_{T}\}$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 7: [[动力学或策略机制]]

$$s_{t}^{\mathrm{p}}:=\left[{\boldsymbol{{q}}_{t-4:t}},{\boldsymbol{\dot{q}}_{t-4:t}},{\boldsymbol{\omega}^{root}_{t-4:t}},{\boldsymbol{g}_{t-4:t}},{\boldsymbol{a}_{t-5:t-1}}\right]$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 8: [[动力学或策略机制]]

$$s^{\text{real}}_{t+1}-s^{\text{sim}}_{t+1}=f^{\Delta}_{\theta}(s^{\text{real}}_{t},a^{\text{real}}_{t}),\quad\forall t.$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 9: [[动力学或策略机制]]

$$s_{t}=[p^{\text{base}}_{t},v_{t}^{\text{base}},\alpha^{\text{base}}_{t},\omega^{\text{base}}_{t},q_{t},\dot{q}_{t}],$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 10: [[动力学或策略机制]]

$$s_{t+1}=f^{\text{{ASAP}}}(s_{t},a_{t})=f^{\text{sim}}(s_{t},a_{t}+\pi^{\Delta}(s_{t},a_{t})),$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。
## 关键图表

### Figure 1: Overview of ASAP. (a) Motion Tracking Pre-training and Real Trajectory Collection: W

![Figure 1](https://arxiv.org/html/2502.01143v3/x1.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Overview of ASAP. (a) Motion Tracking Pre-training and Real Trajectory Collection: W”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 2: Retargeting Human Video Motions to Robot Motions: (a) Human motions are captured from

![Figure 2](https://arxiv.org/html/2502.01143v3/x2.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Retargeting Human Video Motions to Robot Motions: (a) Human motions are captured from”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 3: Baselines of ASAP. (a) Model-free RL training. (b) System ID from real to sim using r

![Figure 3](https://arxiv.org/html/2502.01143v3/x3.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Baselines of ASAP. (a) Model-free RL training. (b) System ID from real to sim using r”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### 论文表格

#### Table 1: TABLE I: Reward Terms for Pretraining

| Term | Weight | Term | Weight |
| --- | --- | --- | --- |
| Penalty | | | |
| DoF position limits | - 10.0 10.0 -10.0 - 10.0 | DoF velocity limits | - 5.0 5.0 -5.0 - 5.0 |
| Torque limits | - 5.0 5.0 -5.0 - 5.0 | Termination | - 200.0 200.0 -200.0 - 200.0 |
| Regularization | | | |
| Torques | - 1  10 - 6 1 10 6 -1\times 10^{-6} - 1  10 - 6 | Action rate | - 0.5 0.5 -0.5 - 0.5 |
| Feet orientation | - 2.0 2.0 -2.0 - 2.0 | Feet heading | - 0.1 0.1 -0.1 - 0.1 |
| Slippage | - 1.0 1.0 -1.0 - 1.0 | | |
| Task Reward | | | |
| Body position | 1.0 1.0 1.0 1.0 | VR 3-point | 1.6 1.6 1.6 1.6 |
| Body position (feet) | 2.1 2.1 2.1 2.1 | Body rotation | 0.5 0.5 0.5 0.5 |
| Body angular velocity | 0.5 0.5 0.5 0.5 | Body velocity | 0.5 0.5 0.5 0.5 |
| DoF position | 0.75 0.75 0.75 0.75 | DoF velocity | 0.5 0.5 0.5 0.5 |

**说明**: TABLE I: Reward Terms for Pretraining

#### Table 2: TABLE II: Reward Terms for Delta Action Learning

| Term | Weight | Term | Weight |
| --- | --- | --- | --- |
| Penalty | | | |
| DoF position limits | - 10.0 10.0 -10.0 - 10.0 | DoF velocity limits | - 5.0 5.0 -5.0 - 5.0 |
| Torque limits | - 0.1 0.1 -0.1 - 0.1 | Termination | - 200.0 200.0 -200.0 - 200.0 |
| Regularization | | | |
| Action rate | - 0.01 0.01 -0.01 - 0.01 | Action norm | - 0.2 0.2 -0.2 - 0.2 |
| Task Reward | | | |
| Body position | 1.0 1.0 1.0 1.0 | VR 3-point | 1.0 1.0 1.0 1.0 |
| Body position (feet) | 1.0 1.0 1.0 1.0 | Body rotation | 0.5 0.5 0.5 0.5 |
| Body angular velocity | 0.5 0.5 0.5 0.5 | Body velocity | 0.5 0.5 0.5 0.5 |
| DoF position | 0.5 0.5 0.5 0.5 | DoF velocity | 0.5 0.5 0.5 0.5 |

**说明**: TABLE II: Reward Terms for Delta Action Learning

#### Table 3: TABLE III: Open-loop performance comparison across simulators and motion lengths.

| Simulator & Length | IsaacSim | Genesis | | | | | | | |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Length | Method | E g-mpjpe E g-mpjpe E_{ $\text{g-mpjpe}} g-mpjpe$ | E mpjpe E mpjpe E_{ $\text{mpjpe}} mpjpe$ | E acc E acc  $\text{E}_{\text{acc}} E acc$ | E vel E vel  $\text{E}_{\text{vel}} E vel$ | E g-mpjpe E g-mpjpe E_{ $\text{g-mpjpe}} g-mpjpe$ | E mpjpe E mpjpe E_{ $\text{mpjpe}} mpjpe$ | E acc E acc  $\text{E}_{\text{acc}} E acc$ | E vel E vel  $\text{E}_{\text{vel}} E vel$ |
| 0.25s | OpenLoop | 19.5 | 15.1 | 6.44 | 5.80 | 19.8 | 15.3 | 6.53 | 5.88 |
| SysID | 19.4 | 15.0 | 6.43 | 5.74 | 19.3 | 15.0 | 6.42 | 5.73 | |
| DeltaDynamics | 24.4 | 13.6 | 9.43 | 7.85 | 20.0 | 12.4 | 8.42 | 6.89 | |
| ASAP | 19.9 | 15.6 | 6.48 | 5.86 | 19.0 | 14.9 | 6.19 | 5.59 | |
| 0.5s | OpenLoop | 33.3 | 23.2 | 6.80 | 6.84 | 33.1 | 23.0 | 6.78 | 6.82 |
| SysID | 32.1 | 22.2 | 6.57 | 6.56 | 32.2 | 22.3 | 6.57 | 6.57 | |
| DeltaDynamics | 36.5 | 16.4 | 8.89 | 7.98 | 27.8 | 14.0 | 7.63 | 6.74 | |
| ASAP | 26.8 | 19.2 | 5.09 | 5.36 | 25.9 | 18.4 | 4.93 | 5.19 | |
| 1.0s | OpenLoop | 80.8 | 43.5 | 10.6 | 11.1 | 82.5 | 44.5 | 10.8 | 11.4 |
| SysID | 77.6 | 41.5 | 10.2 | 10.7 | 76.5 | 41.6 | 10.0 | 10.5 | |
| DeltaDynamics | 68.1 | 21.5 | 9.61 | 9.14 | 50.2 | 17.2 | 8.19 | 7.62 | |
| ASAP | 37.9 | 22.9 | 4.38 | 5.26 | 36.9 | 22.6 | 4.23 | 5.10 | |

**说明**: TABLE III: Open-loop performance comparison across simulators and motion lengths.

#### Table 4: TABLE IV: Closed-loop motion imitation evaluation across different simulators. All variants are trained with identical r

| Test Environment | IsaacSim | Genesis | | | | | | | | | |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Level | Method | Succ ↑ ↑ Succ absent  $\text{Succ}\uparrow Succ ↑$ | E g-mpjpe ↓ ↓ E g-mpjpe absent E_{ $\text{g-mpjpe}}\downarrow g-mpjpe ↓$ | E mpjpe ↓ ↓ E mpjpe absent E_{ $\text{mpjpe}}\downarrow mpjpe ↓$ | E acc ↓ ↓ E acc absent  $\text{E}_{\text{acc}}\downarrow E acc ↓$ | E vel ↓ ↓ E vel absent  $\text{E}_{\text{vel}}\downarrow E vel ↓$ | Succ ↑ ↑ Succ absent  $\text{Succ}\uparrow Succ ↑$ | E g-mpjpe ↓ ↓ E g-mpjpe absent E_{ $\text{g-mpjpe}}\downarrow g-mpjpe ↓$ | E mpjpe ↓ ↓ E mpjpe absent E_{ $\text{mpjpe}}\downarrow mpjpe ↓$ | E acc ↓ ↓ E acc absent  $\text{E}_{\text{acc}}\downarrow E acc ↓$ | E vel ↓ ↓ E vel absent  $\text{E}_{\text{vel}}\downarrow E vel ↓$ |
| | Oracle (IsaacGym → →  $\rightarrow → IsaacGym)$ | 100% ± 0.000% plus-or-minus 0.000% \pm $\text{0.000\} ± 0.000$ | 97.5 ± 0.605 plus-or-minus 0.605 \pm $\text{0.605} ± 0.605$ | 43.2 ± 0.112 plus-or-minus 0.112 \pm $\text{0.112} ± 0.112$ | 2.56 ± 0.024 plus-or-minus 0.024 \pm $\text{0.024} ± 0.024$ | 4.48 ± 0.023 plus-or-minus 0.023 \pm $\text{0.023} ± 0.023$ | 100% ± 0.000% plus-or-minus 0.000% \pm $\text{0.000\} ± 0.000$ | 97.5 ± 0.605 plus-or-minus 0.605 \pm $\text{0.605} ± 0.605$ | 43.2 ± 0.112 plus-or-minus 0.112 \pm $\text{0.112} ± 0.112$ | 2.56 ± 0.024 plus-or-minus 0.024 \pm $\text{0.024} ± 0.024$ | 4.48 ± 0.023 plus-or-minus 0.023 \pm $\text{0.023} ± 0.023$ |
| | Vanilla (IsaacGym → →  $\rightarrow → TestEnv)$ | 100% ± 0.000% plus-or-minus 0.000% \pm $\text{0.000\} ± 0.000$ | 107 ± 0.578 plus-or-minus 0.578 \pm $\text{0.578} ± 0.578$ | 45.4 ± 0.169 plus-or-minus 0.169 \pm $\text{0.169} ± 0.169$ | 2.83 ± 0.012 plus-or-minus 0.012 \pm $\text{0.012} ± 0.012$ | 4.59 ± 0.021 plus-or-minus 0.021 \pm $\text{0.021} ± 0.021$ | 100% ± 0.000% plus-or-minus 0.000% \pm $\text{0.000\} ± 0.000$ | 140 ± 1.85 plus-or-minus 1.85 \pm $\text{1.85} ± 1.85$ | 70.1 ± 0.626 plus-or-minus 0.626 \pm $\text{0.626} ± 0.626$ | 2.68 ± 0.042 plus-or-minus 0.042 \pm $\text{0.042} ± 0.042$ | 4.65 ± 0.046 plus-or-minus 0.046 \pm $\text{0.046} ± 0.046$ |
| Easy | SysID | 100% ± 0.000% plus-or-minus 0.000% \pm $\text{0.000\} ± 0.000$ | 105 ± 1.35 plus-or-minus 1.35 \pm $\text{1.35} ± 1.35$ | 47.8 ± 0.970 plus-or-minus 0.970 \pm $\text{0.970} ± 0.970$ | 3.09 ± 0.011 plus-or-minus 0.011 \pm $\text{0.011} ± 0.011$ | 4.98 ± 0.020 plus-or-minus 0.020 \pm $\text{0.020} ± 0.020$ | 100% ± 0.000% plus-or-minus 0.000% \pm $\text{0.000\} ± 0.000$ | 127 ± 0.233 plus-or-minus 0.233 \pm $\text{0.233} ± 0.233$ | 79.9 ± 0.330 plus-or-minus 0.330 \pm $\text{0.330} ± 0.330$ | 2.99 ± 0.035 plus-or-minus 0.035 \pm $\text{0.035} ± 0.035$ | 4.95 ± 0.012 plus-or-minus 0.012 \pm $\text{0.012} ± 0.012$ |
| | DeltaDynamics | 100% ± 0.000% plus-or-minus 0.000% \pm $\text{0.000\} ± 0.000$ | 127 ± 2.97 plus-or-minus 2.97 \pm $\text{2.97} ± 2.97$ | 56.7 ± 0.390 plus-or-minus 0.390 \pm $\text{0.390} ± 0.390$ | 3.50 ± 0.028 plus-or-minus 0.028 \pm $\text{0.028} ± 0.028$ | 5.56 ± 0.031 plus-or-minus 0.031 \pm $\text{0.031} ± 0.031$ | 83.3% ± 0.000% plus-or-minus 0.000% \pm $\text{0.000\} ± 0.000$ | 168 ± 7.62 plus-or-minus 7.62 \pm $\text{7.62} ± 7.62$ | 87.0 ± 1.51 plus-or-minus 1.51 \pm $\text{1.51} ± 1.51$ | 3.08 ± 0.18 plus-or-minus 0.18 \pm $\text{0.18} ± 0.18$ | 5.39 ± 0.34 plus-or-minus 0.34 \pm $\text{0.34} ± 0.34$ |
| | ASAP | 100% ± 0.000% plus-or-minus 0.000% \pm $\text{0.000\} ± 0.000$ | 106 ± 0.498 plus-or-minus 0.498 \pm $\text{0.498} ± 0.498$ | 44.3 ± 0.103 plus-or-minus 0.103 \pm $\text{0.103} ± 0.103$ | 2.74 ± 0.025 plus-or-minus 0.025 \pm $\text{0.025} ± 0.025$ | 4.46 ± 0.020 plus-or-minus 0.020 \pm $\text{0.020} ± 0.020$ | 100% ± 0.000% plus-or-minus 0.000% \pm $\text{0.000\} ± 0.000$ | 125 ± 4.75 plus-or-minus 4.75 \pm $\text{4.75} ± 4.75$ | 73.5 ± 0.570 plus-or-minus 0.570 \pm $\text{0.570} ± 0.570$ | 2.10 ± 0.083 plus-or-minus 0.083 \pm $\text{0.083} ± 0.083$ | 4.11 ± 0.133 plus-or-minus 0.133 \pm $\text{0.133} ± 0.133$ |
| | Oracle (IsaacGym → →  $\rightarrow → IsaacGym)$ | 100% ± 0.000% plus-or-minus 0.000% \pm $\text{0.000\} ± 0.000$ | 111 ± 0.635 plus-or-minus 0.635 \pm $\text{0.635} ± 0.635$ | 48.8 ± 0.133 plus-or-minus 0.133 \pm $\text{0.133} ± 0.133$ | 2.63 ± 0.017 plus-or-minus 0.017 \pm $\text{0.017} ± 0.017$ | 4.82 ± 0.019 plus-or-minus 0.019 \pm $\text{0.019} ± 0.019$ | 100% ± 0.000% plus-or-minus 0.000% \pm $\text{0.000\} ± 0.000$ | 111 ± 0.635 plus-or-minus 0.635 \pm $\text{0.635} ± 0.635$ | 48.8 ± 0.133 plus-or-minus 0.133 \pm $\text{0.133} ± 0.133$ | 2.63 ± 0.017 plus-or-minus 0.017 \pm $\text{0.017} ± 0.017$ | 4.82 ± 0.019 plus-or-minus 0.019 \pm $\text{0.019} ± 0.019$ |
| | Vanilla (IsaacGym → →  $\rightarrow → TestEnv)$ | 100% ± 0.000% plus-or-minus 0.000% \pm $\text{0.000\} ± 0.000$ | 114 ± 0.720 plus-or-minus 0.720 \pm $\text{0.720} ± 0.720$ | 49.2 ± 0.104 plus-or-minus 0.104 \pm $\text{0.104} ± 0.104$ | 2.92 ± 0.021 plus-or-minus 0.021 \pm $\text{0.021} ± 0.021$ | 5.07 ± 0.016 plus-or-minus 0.016 \pm $\text{0.016} ± 0.016$ | 94.3% ± 7.00% plus-or-minus 7.00% \pm $\text{7.00\} ± 7.00$ | 169 ± 5.76 plus-or-minus 5.76 \pm $\text{5.76} ± 5.76$ | 72.0 ± 0.692 plus-or-minus 0.692 \pm $\text{0.692} ± 0.692$ | 3.26 ± 0.076 plus-or-minus 0.076 \pm $\text{0.076} ± 0.076$ | 5.86 ± 0.101 plus-or-minus 0.101 \pm $\text{0.101} ± 0.101$ |
| Medium | SysID | 100% ± 0.000% plus-or-minus 0.000% \pm $\text{0.000\} ± 0.000$ | 115 ± 1.256 plus-or-minus 1.256 \pm $\text{1.256} ± 1.256$ | 49.1 ± 0.560 plus-or-minus 0.560 \pm $\text{0.560} ± 0.560$ | 3.43 ± 0.021 plus-or-minus 0.021 \pm $\text{0.021} ± 0.021$ | 5.01 ± 0.017 plus-or-minus 0.017 \pm $\text{0.017} ± 0.017$ | 100% ± 0.000% plus-or-minus 0.000% \pm $\text{0.000\} ± 0.000$ | 138 ± 2.70 plus-or-minus 2.70 \pm $\text{2.70} ± 2.70$ | 75.4 ± 1.18 plus-or-minus 1.18 \pm $\text{1.18} ± 1.18$ | 3.14 ± 0.042 plus-or-minus 0.042 \pm $\text{0.042} ± 0.042$ | 5.50 ± 0.058 plus-or-minus 0.058 \pm $\text{0.058} ± 0.058$ |
| | DeltaDynamics | 83.3% ± 0.000% plus-or-minus 0.000% \pm $\text{0.000\} ± 0.000$ | 151 ± 2.62 plus-or-minus 2.62 \pm $\text{2.62} ± 2.62$ | 68.0 ± 0.364 plus-or-minus 0.364 \pm $\text{0.364} ± 0.364$ | 2.90 ± 0.047 plus-or-minus 0.047 \pm $\text{0.047} ± 0.047$ | 5.90 ± 0.107 plus-or-minus 0.107 \pm $\text{0.107} ± 0.107$ | 83.3% ± 0.000% plus-or-minus 0.000% \pm $\text{0.000\} ± 0.000$ | 190 ± 1.46 plus-or-minus 1.46 \pm $\text{1.46} ± 1.46$ | 89.4 ± 0.50 plus-or-minus 0.50 \pm $\text{0.50} ± 0.50$ | 3.44 ± 0.16 plus-or-minus 0.16 \pm $\text{0.16} ± 0.16$ | 7.49 ± 0.11 plus-or-minus 0.11 \pm $\text{0.11} ± 0.11$ |
| | ASAP | 100% ± 0.000% plus-or-minus 0.000% \pm $\text{0.000\} ± 0.000$ | 112 ± 1.648 plus-or-minus 1.648 \pm $\text{1.648} ± 1.648$ | 49.3 ± 0.574 plus-or-minus 0.574 \pm $\text{0.574} ± 0.574$ | 2.53 ± 0.019 plus-or-minus 0.019 \pm $\text{0.019} ± 0.019$ | 4.45 ± 0.026 plus-or-minus 0.026 \pm $\text{0.026} ± 0.026$ | 100% ± 0.000% plus-or-minus 0.000% \pm $\text{0.000\} ± 0.000$ | 126 ± 1.63 plus-or-minus 1.63 \pm $\text{1.63} ± 1.63$ | 71.2 ± 0.163 plus-or-minus 0.163 \pm $\text{0.163} ± 0.163$ | 2.81 ± 0.037 plus-or-minus 0.037 \pm $\text{0.037} ± 0.037$ | 5.13 ± 0.066 plus-or-minus 0.066 \pm $\text{0.066} ± 0.066$ |
| | Oracle (IsaacGym → →  $\rightarrow → IsaacGym)$ | 1 00% ± 0.000% plus-or-minus 0.000% \pm $\text{0.000\} ± 0.000$ | 116 ± 0.711 plus-or-minus 0.711 \pm $\text{0.711} ± 0.711$ | 52.5 ± 0.298 plus-or-minus 0.298 \pm $\text{0.298} ± 0.298$ | 3.40 ± 0.027 plus-or-minus 0.027 \pm $\text{0.027} ± 0.027$ | 6.16 ± 0.028 plus-or-minus 0.028 \pm $\text{0.028} ± 0.028$ | 100% ± 0.000% plus-or-minus 0.000% \pm $\text{0.000\} ± 0.000$ | 116 ± 0.711 plus-or-minus 0.711 \pm $\text{0.711} ± 0.711$ | 52.5 ± 0.298 plus-or-minus 0.298 \pm $\text{0.298} ± 0.298$ | 3.40 ± 0.027 plus-or-minus 0.027 \pm $\text{0.027} ± 0.027$ | 6.16 ± 0.028 plus-or-minus 0.028 \pm $\text{0.028} ± 0.028$ |
| | Vanilla (IsaacGym → →  $\rightarrow → TestEnv)$ | 100% ± 0.000% plus-or-minus 0.000% \pm $\text{0.000\} ± 0.000$ | 148 ± 0.845 plus-or-minus 0.845 \pm $\text{0.845} ± 0.845$ | 51.6 ± 0.137 plus-or-minus 0.137 \pm $\text{0.137} ± 0.137$ | 4.41 ± 0.055 plus-or-minus 0.055 \pm $\text{0.055} ± 0.055$ | 6.88 ± 0.064 plus-or-minus 0.064 \pm $\text{0.064} ± 0.064$ | 82.9% ± 5.70% plus-or-minus 5.70% \pm $\text{5.70\} ± 5.70$ | 175 ± 9.77 plus-or-minus 9.77 \pm $\text{9.77} ± 9.77$ | 80.7 ± 1.69 plus-or-minus 1.69 \pm $\text{1.69} ± 1.69$ | 3.87 ± 0.175 plus-or-minus 0.175 \pm $\text{0.175} ± 0.175$ | 7.19 ± 0.199 plus-or-minus 0.199 \pm $\text{0.199} ± 0.199$ |
| Hard | SysID | 100% ± 0.000% plus-or-minus 0.000% \pm $\text{0.000\} ± 0.000$ | 165 ± 3.83 plus-or-minus 3.83 \pm $\text{3.83} ± 3.83$ | 58.4 ± 0.229 plus-or-minus 0.229 \pm $\text{0.229} ± 0.229$ | 4.87 ± 0.197 plus-or-minus 0.197 \pm $\text{0.197} ± 0.197$ | 7.13 ± 0.131 plus-or-minus 0.131 \pm $\text{0.131} ± 0.131$ | 100% ± 0.000% plus-or-minus 0.000% \pm $\text{0.000\} ± 0.000$ | 186 ± 3.84 plus-or-minus 3.84 \pm $\text{3.84} ± 3.84$ | 93.0 ± 1.49 plus-or-minus 1.49 \pm $\text{1.49} ± 1.49$ | 4.98 ± 0.245 plus-or-minus 0.245 \pm $\text{0.245} ± 0.245$ | 8.98 ± 0.119 plus-or-minus 0.119 \pm $\text{0.119} ± 0.119$ |
| | DeltaDynamics | 66.7% ± 0.000% plus-or-minus 0.000% \pm $\text{0.000\} ± 0.000$ | 137 ± 2.59 plus-or-minus 2.59 \pm $\text{2.59} ± 2.59$ | 60.2 ± 0.477 plus-or-minus 0.477 \pm $\text{0.477} ± 0.477$ | 4.20 ± 0.041 plus-or-minus 0.041 \pm $\text{0.041} ± 0.041$ | 7.10 ± 0.024 plus-or-minus 0.024 \pm $\text{0.024} ± 0.024$ | 60.0% ± 5.70% plus-or-minus 5.70% \pm $\text{5.70\} ± 5.70$ | 190 ± 14.0 plus-or-minus 14.0 \pm $\text{14.0} ± 14.0$ | 89.6 ± 9.34 plus-or-minus 9.34 \pm $\text{9.34} ± 9.34$ | 4.29 ± 1.16 plus-or-minus 1.16 \pm $\text{1.16} ± 1.16$ | 8.70 ± 2.33 plus-or-minus 2.33 \pm $\text{2.33} ± 2.33$ |
| | ASAP | 100% ± 0.000% plus-or-minus 0.000% \pm $\text{0.000\} ± 0.000$ | 129 ± 1.57 plus-or-minus 1.57 \pm $\text{1.57} ± 1.57$ | 56.5 ± 1.15 plus-or-minus 1.15 \pm $\text{1.15} ± 1.15$ | 3.72 ± 0.036 plus-or-minus 0.036 \pm $\text{0.036} ± 0.036$ | 6.52 ± 0.042 plus-or-minus 0.042 \pm $\text{0.042} ± 0.042$ | 100% ± 0.000% plus-or-minus 0.000% \pm $\text{0.000\} ± 0.000$ | 129 ± 2.31 plus-or-minus 2.31 \pm $\text{2.31} ± 2.31$ | 77.0 ± 1.07 plus-or-minus 1.07 \pm $\text{1.07} ± 1.07$ | 2.69 ± 0.040 plus-or-minus 0.040 \pm $\text{0.040} ± 0.040$ | 5.65 ± 0.073 plus-or-minus 0.073 \pm $\text{0.073} ± 0.073$ |

**说明**: TABLE IV: Closed-loop motion imitation evaluation across different simulators. All variants are trained with identical rewards.

#### Table 5: TABLE V: Real-world closed-loop performance comparing with and without ASAP finetuning on one in-distribution motion and

| Motion | Real-World-Kick | Real-World-LeBron (OOD) | | | | | | |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Method | E g-mpjpe E g-mpjpe E_{ $\text{g-mpjpe}} g-mpjpe$ | E mpjpe E mpjpe E_{ $\text{mpjpe}} mpjpe$ | E acc E acc  $\text{E}_{\text{acc}} E acc$ | E vel E vel  $\text{E}_{\text{vel}} E vel$ | E g-mpjpe E g-mpjpe E_{ $\text{g-mpjpe}} g-mpjpe$ | E mpjpe E mpjpe E_{ $\text{mpjpe}} mpjpe$ | E acc E acc  $\text{E}_{\text{acc}} E acc$ | E vel E vel  $\text{E}_{\text{vel}} E vel$ |
| Vanilla | 61.2 | 43.5 | 2.96 | 2.91 | 159 | 55.3 | 3.43 | 6.43 |
| ASAP | 50.2 | 40.1 | 2.46 | 2.70 | 112 | 47.5 | 2.84 | 5.94 |

**说明**: TABLE V: Real-world closed-loop performance comparing with and without ASAP finetuning on one in-distribution motion and one out-of-distribution motion.

#### Table 6: TABLE VI: Domain Randomizations

| Term | Value |
| --- | --- |
| Dynamics Randomization | |
| Friction | U (0.2, 1.1) U 0.2 1.1  $\mathcal{U}(0.2,1.1)$ |
| P Gain | $\mathcal{U}(0.925,1.05)\times (0.925, 1.05) × default$ |
| Control delay | U (20, 40) ms U 20 40 ms  $\mathcal{U}(20,40)\mathrm{ms}$ |
| External Perturbation | |
| Push robot | interval = 10 s, v x y = 0.5 m / s formulae-sequence absent 10 s v x y 0.5 m s =10s,v_{xy}=0.5 $\mathrm{~{}m}/\mathrm{s} = 10, = 0.5 /$ |

**说明**: TABLE VI: Domain Randomizations

#### Table 7: TABLE VII: SysID Parameters

| Parameter | Range | Parameter | Range |
| --- | --- | --- | --- |
| c x c x c_{x} | [- 0.02, 0.0, 0.02 ] 0.02 0.0 0.02 [-0.02,0.0,0.02] [- 0.02, 0.0, 0.02 ] | c y c y c_{y} | [- 0.02, 0.0, 0.02 ] 0.02 0.0 0.02 [-0.02,0.0,0.02] [- 0.02, 0.0, 0.02 ] |
| c z c z c_{z} | [- 0.02, 0.0, 0.02 ] 0.02 0.0 0.02 [-0.02,0.0,0.02] [- 0.02, 0.0, 0.02 ] | k m k m k_{m} | [0.95, 1.0, 1.05 ] 0.95 1.0 1.05 [0.95,1.0,1.05] [0.95, 1.0, 1.05 ] |
| k p i k i p k^{i}_{p} | [0.95, 1.0, 1.05 ] 0.95 1.0 1.05 [0.95,1.0,1.05] [0.95, 1.0, 1.05 ] | k d i k i d k^{i}_{d} | [0.95, 1.0, 1.05 ] 0.95 1.0 1.05 [0.95,1.0,1.05] [0.95, 1.0, 1.05 ] |

**说明**: TABLE VII: SysID Parameters
## 实验解读

- 评价重点:围绕 action-model、agile-locomotion、人形机器人、足式运动、运动模仿,优先看真实机器人结果、扰动恢复、接触稳定性、控制误差、成功率、实时性和 sim-to-real 差距。
- System0 视角:实验结论不只看最终成功率,还要看闭环过程中是否能处理状态估计误差、执行延迟、碰撞/接触突变和外部扰动。
- 与上层系统的关系:如果方法能在快闭环内稳定工作,它可以作为 System1 技能或 System2 规划的底层执行保障;如果只在离线或慢速设定下成立,则还不能直接作为身体级快闭环模块。
## 批判性思考

### 优点

- 论文把 action-model、agile-locomotion、人形机器人、足式运动、运动模仿 放到机器人身体级闭环里讨论,和只做高层语义推理的工作相比,更接近真实机器人部署时会遇到的状态、动力学和安全约束。
- 若论文同时给出公式、系统结构图、硬件实验或消融结果,这些证据可以帮助判断方法是否真的能支撑毫秒到百毫秒级反应。

### 局限

- 需要重点核查控制频率、推理延迟、传感延迟、失败恢复、硬件磨损、未见场景泛化和安全边界;这些因素决定它能否成为可靠的 System0 模块。
- 对学习型方法,还要区分训练环境中的鲁棒性和真实部署中的鲁棒性,尤其是接触丰富任务、动态运动任务和人形机器人任务。
## 论文定位

- 研究对象:ASAP: Aligning Simulation and Real-World Physics for Learning Agile Humanoid Whole-Body Skills。
- 关键词:action-model、agile-locomotion、人形机器人、足式运动、运动模仿、motion-tracking、retargeting、仿真到真实迁移、全身控制。
- System0 关联:关注机器人身体级快反应链路中“状态/接触/局部感知 -> 约束/优化/策略 -> 动作”的短闭环,不把问题停留在语言规划或任务分解层。

## 速查卡片

> [!summary] ASAP
> - **论文**: https://www.roboticsproceedings.org/rss21/p066.pdf
> - **arXiv**: http://arxiv.org/abs/2502.01143v3
> - **arXiv HTML**: https://arxiv.org/html/2502.01143v3
