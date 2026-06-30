---
title: "Meta-Learning Online Dynamics Model Adaptation in Off-Road Autonomous Driving"
method_name: "Meta Online Dynamics"
authors: ["Jacob Levy"]
year: 2025
venue: "RSS"
tags: ["robust-control", "real-time-control", "legged-locomotion", "safe-control", "adaptive-control", "robot-generalization", "agile-locomotion", "navigation"]
image_source: "online"
arxiv_html: "https://arxiv.org/html/2504.16923v1"
---
# Meta Online Dynamics
## 一句话总结

> Meta-Learning Online Dynamics Model Adaptation in Off-Road Autonomous Driving 主要落在 [[adaptive-control]]、[[agile-locomotion]]、[[dynamics-modeling]]、[[navigation]] 这一类问题上;从 System0 视角看,它的价值在于把身体状态、接触变化、局部感知或动力学约束接到机器人低层闭环里,让动作生成更稳定、安全、实时。
## 核心贡献

1. 围绕 **Meta-Learning Online Dynamics Model Adaptation in Off-Road Autonomous Driving** 建立了一个与 adaptive-control、agile-locomotion、dynamics-modeling、navigation、实时控制、robot-generalization 相关的机器人问题设定,重点不是高层语义规划,而是身体级闭环中的状态变化、接触变化或控制约束。
2. 方法层面的关键价值在于把感知、动力学、约束优化、策略学习或硬件反馈组织成可执行的低层动作生成机制,适合放进 System0 的快反应模块中理解。
3. 对系统集成而言,这类工作补上了 System1 技能选择与电机级控制之间的一层:它负责把上层目标变成动态可行、可恢复、可安全过滤的动作。
4. 若论文包含真实机器人、接触任务或仿真到真实迁移,核心应看它是否报告了闭环稳定性、扰动恢复、硬件延迟、失败模式和跨场景泛化。
## 问题背景

这篇论文可以放在 System0 的“具身快系统”背景下阅读:机器人需要在很短时间内根据本体状态、接触、力/触觉、局部视觉或动力学预测做出动作,而不是等待慢速规划器重新推理整个任务。adaptive-control、agile-locomotion、dynamics-modeling、navigation、实时控制、robot-generalization、safe-control 相关问题的难点通常不在“知道要做什么”,而在动作执行过程中状态会快速偏离、接触会突然改变、约束会实时收紧,低层系统必须及时修正。

### System0 关联

精读时优先看三件事:第一,论文使用哪些快速反馈量,例如关节状态、接触状态、力觉、触觉、深度或局部视觉;第二,反馈如何进入 MPC、WBC、策略网络、安全过滤器或恢复控制器;第三,输出是否直接影响机器人下一步动作,而不是只服务离线分析或高层任务规划。
## 方法详解

### System0 视角解析

从 System0 角度看,这篇论文的方法部分应当围绕 adaptive-control、agile-locomotion、dynamics-modeling、navigation、实时控制、robot-generalization、safe-control、terrain-adaptation 来读。核心问题是:系统如何从快速变化的机器人状态中构造控制输入,如何处理接触、碰撞、约束、模型误差或执行延迟,以及如何把这些信息变成下一时刻的动作。

如果方法里有优化问题,重点看目标函数、约束和求解频率;如果方法里有学习策略,重点看观测空间、动作空间、训练扰动和部署时的闭环频率;如果方法里有硬件或传感器,重点看传感链路是否真的进入控制回路。这样读可以把论文和 System2 语言规划、System1 技能编排区分开来。

### 关键公式与机制

#### 公式 1: [[优化目标/约束]]

$$\xi\leftarrow\xi-\alpha\nabla_{\xi}\sum_{i=1}^{N_{B}}\mathcal{L}_{i}$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 2: [[动力学或策略机制]]

$$\dot{v_{t}}=M^{-1}X(x_{t})\left[F_{t}+\zeta(\mathbf{\eta}_{t};\phi,\theta_{t})\right]+D_{t},$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 3: [[动力学或策略机制]]

$$\zeta(\mathbf{\eta};\phi,\theta)=\left(\phi^{w}+\theta^{w}\right)^{\mkern-1.5mu\mathsf{T}}\mathbf{W}\Phi(\mathbf{\eta};\phi^{l})+\phi^{b}+\theta^{b},$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 4: [[动力学或策略机制]]

$$f(x_{t},u_{t},y_{t})=E_{A}R(p_{t})x_{t}+E_{B}\begin{bmatrix}v_{t}^{\mkern-1.5mu\mathsf{T}}&\dot{v_{t}}^{\mkern-1.5mu\mathsf{T}}&\dot{z_{t}}^{\mkern-1.5mu\mathsf{T}}\end{bmatrix}^{\mkern-1.5mu\mathsf{T}},$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 5: [[优化目标/约束]]

$$\mathtt{rollover}(x,y)=\mathcal{P}_{n}\left(\min\left(F^{L},F^{R}\right);r_{\mathtt{limit}}\right),$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 6: [[优化目标/约束]]

$$
\mathcal{L}_{i}=\frac{1}{T}\sum_{j=0}^{T}\left\lVert \hat{x}_{t+1+j}-x_{t+1+j}\right \rVert^{2}_{2},
$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 7: [[动力学或策略机制]]

$$J(\mathbf{u}_{s})=V(x_{s+1+T})+\sum_{t=s}^{s+T}\left[l(x_{t},u_{t},y_{t})+V(x_{t})\right]$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 8: [[优化目标/约束]]

$$=\mathcal{F}^{x}_{t+i}H_{t+i-1}+\mathcal{F}^{\theta}_{t+i}$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 9: [[动力学或策略机制]]

$$\gamma_{t}=\frac{\left\lVert v_{t}\right \rVert^{2}_{2}}{\left\lVert v_{t}\right \rVert^{2}_{2}+\varepsilon},$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 10: [[动力学或策略机制]]

$$S_{t+h}=C\left(H_{t+h}\bar{P}_{t+h}H_{t+h}^{\mkern-1.5mu\mathsf{T}}+R\right)C^{\mkern-1.5mu\mathsf{T}}$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。
## 关键图表

### Figure 1: Pipeline / core system figure: Trajectories for a single 3-lap run, with insets displaying video stills. The baseline

![Figure 1](https://arxiv.org/html/2504.16923v1/x1.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Pipeline / core system figure: Trajectories for a single 3-lap run, with insets displaying video stills. The baseline”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 2: Meta-learning online dynamics model adaptation. Online, a Kalman filter updates the li

![Figure 2](https://arxiv.org/html/2504.16923v1/x2.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Meta-learning online dynamics model adaptation. Online, a Kalman filter updates the li”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 3: Trajectories for all 3-lap real-world runs. The baseline configuration exhibits errati

![Figure 3](https://arxiv.org/html/2504.16923v1/x4.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Trajectories for all 3-lap real-world runs. The baseline configuration exhibits errati”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### 论文表格

#### Table 1: TABLE I: Dataset statistics.

| | Mean | Std | Min | 5th % | Median | 95th % | Max |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Fwd. vel. (m/s) | 4.55 | 2.91 | -2.79 | -0.01 | 4.56 | 9.62 | 15.25 |
| Lat. vel. (m/s) | 0.02 | 0.20 | -1.31 | -0.32 | 0.00 | 0.37 | 1.34 |
| Yaw rate (rad/s) | 0.00 | 0.19 | -1.32 | -0.32 | 0.00 | 0.31 | 1.40 |
| Pitch (deg) | -0.4 | 4.9 | -24.2 | -8.0 | -0.4 | 7.9 | 31.3 |
| Roll (deg) | 0.2 | 4.8 | -28.7 | -7.8 | 0.2 | 8.7 | 28.8 |

**说明**: TABLE I: Dataset statistics.

#### Table 2: TABLE II: Real-world validation results. The means and standard deviations 4 runs of each configuration are display

| | Completion | Average | Prediction | # times crossed limit | Time exceeds limit, s second  $\mathrm{s}$ | Cost ( 10 4 absent 10 4 \times 10^{4}  10 4) | | | |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| | time, s second  $\mathrm{s}$ | speed, m s - 1 times meter second 1  $\mathrm{m}\text{\,}{\mathrm{s}}^{-1} times power - 1$ | Error, m meter  $\mathrm{m}$ | Track | Rollover | Track | Rollover | Track | Rollover |
| Baseline (no adaptation) | 154.6 $\pm$ 16.9 | 5.06 $\pm$ 0.58 | 4.88 $\pm$ 0.47 | 8.0 $\pm$ 1.8 | 13 $\pm$ 5.4 | 5.32 $\pm$ 1.81 | 6.64 $\pm$ 2.09 | 40.8 | 20.2 |
| Meta-adaptation (ours) | 130.9 $\pm$ 7.8 | 5.84 $\pm$ 0.33 | 3.10 $\pm$ 0.18 | 3.3 $\pm$ 2.1 | 3.8 $\pm$ 1.7 | 0.57 $\pm$ 0.53 | 0.85 $\pm$ 0.59 | 1.95 | 0.36 |

**说明**: TABLE II: Real-world validation results. The means and standard deviations 4 runs of each configuration are displayed.

#### Table 3: TABLE III: Results of the simulated experiments. The means 25 runs of each configuration are displayed.

| | Completion time, s second  $\mathrm{s}$ | Average speed, m s - 1 times meter second 1  $\mathrm{m}\text{\,}{\mathrm{s}}^{-1} times power - 1$ | Prediction Error, m meter  $\mathrm{m}$ | | | | | | | | | |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Steepness | shallow | shallow | steep | steep | shallow | shallow | steep | steep | shallow | shallow | steep | steep |
| Obstacle Density | sparse | dense | sparse | dense | sparse | dense | sparse | dense | sparse | dense | sparse | dense |
| Baseline (no adaptation) | 39.2 | 44.4 | 73.0 * | 86.2 * | 5.88 * | 5.32 | 4.06 * | 3.59 * | 7.19 | 7.62 | 8.17 | 9.03 |
| Sliding LSQ | 46.7 | 46.6 | 87.7 | 102.0 | 5.04 | 5.06 | 3.43 | 3.06 | 5.71 | 5.00 | 5.21 | 5.35 |
| Adaptation | 40.4 | 43.1 * | 83.0 | 95.3 | 5.70 | 5.47 * | 3.59 | 3.15 | 4.54 | 4.45 | 4.25 | 4.64 |
| Meta-adaptation (ours) | 40.1 | 44.7 | 88.8 | 105.9 | 5.75 | 5.27 | 3.34 | 2.99 | 2.96 * | 2.19 * | 3.67 * | 4.65 |
| | # times rollover limit | Time exceeding rollover limit, s second  $\mathrm{s}$ | Rollover Cost ( 10 4 absent 10 4 \times 10^{4}  10 4) | | | | | | | | | |
| Baseline (no adaptation) | 2.7 | 3.6 | 5.4 | 9.9 | 1.30 | 5.31 | 5.09 | 7.50 | 1.09 | 1.75 | 1.82 | 8.91 |
| Sliding LSQ | 2.5 | 4.1 | 5.4 | 7.1 | 2.19 | 6.12 | 4.24 | 5.53 | 1.19 | 2.00 | 4.23 | 8.59 |
| Adaptation | 2.9 | 4.6 | 4.7 | 6.7 | 1.47 | 7.10 | 5.71 | 5.42 | 1.04 | 2.49 | 1.14 | 2.75 |
| Meta-adaptation (ours) | 1.9 | 5.3 | 3.2 * | 6.2 | 0.97 | 4.83 | 3.88 | 3.40 * | 0.99 | 1.15 * | 0.83 | 2.57 |
| * Significant best result: bootstrapped 95% confidence intervals of the mean do not with those of any other configuration. | | | | | | | | | | | | |

**说明**: TABLE III: Results of the simulated experiments. The means 25 runs of each configuration are displayed.
## 实验解读

- 评价重点:围绕 adaptive-control、agile-locomotion、dynamics-modeling、navigation、实时控制,优先看真实机器人结果、扰动恢复、接触稳定性、控制误差、成功率、实时性和 sim-to-real 差距。
- System0 视角:实验结论不只看最终成功率,还要看闭环过程中是否能处理状态估计误差、执行延迟、碰撞/接触突变和外部扰动。
- 与上层系统的关系:如果方法能在快闭环内稳定工作,它可以作为 System1 技能或 System2 规划的底层执行保障;如果只在离线或慢速设定下成立,则还不能直接作为身体级快闭环模块。
## 批判性思考

### 优点

- 论文把 adaptive-control、agile-locomotion、dynamics-modeling、navigation、实时控制 放到机器人身体级闭环里讨论,和只做高层语义推理的工作相比,更接近真实机器人部署时会遇到的状态、动力学和安全约束。
- 若论文同时给出公式、系统结构图、硬件实验或消融结果,这些证据可以帮助判断方法是否真的能支撑毫秒到百毫秒级反应。

### 局限

- 需要重点核查控制频率、推理延迟、传感延迟、失败恢复、硬件磨损、未见场景泛化和安全边界;这些因素决定它能否成为可靠的 System0 模块。
- 对学习型方法,还要区分训练环境中的鲁棒性和真实部署中的鲁棒性,尤其是接触丰富任务、动态运动任务和人形机器人任务。
## 论文定位

- 研究对象:Meta-Learning Online Dynamics Model Adaptation in Off-Road Autonomous Driving。
- 关键词:adaptive-control、agile-locomotion、dynamics-modeling、navigation、实时控制、robot-generalization、safe-control、terrain-adaptation。
- System0 关联:关注机器人身体级快反应链路中“状态/接触/局部感知 -> 约束/优化/策略 -> 动作”的短闭环,不把问题停留在语言规划或任务分解层。

## 速查卡片

> [!summary] Meta Online Dynamics
> - **论文**: https://www.roboticsproceedings.org/rss21/p139.pdf
> - **arXiv**: http://arxiv.org/abs/2504.16923v1
> - **arXiv HTML**: https://arxiv.org/html/2504.16923v1
