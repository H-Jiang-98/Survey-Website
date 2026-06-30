---
title: "X-Loco: Towards Generalist Humanoid Locomotion Control via Synergetic Policy Distillation"
method_name: "X-Loco"
authors: ["Dewei Wang"]
year: 2026
venue: "RSS"
tags: ["legged-locomotion", "adaptive-control", "robot-generalization", "humanoid", "whole-body-control", "recovery"]
image_source: "online"
arxiv_html: "https://arxiv.org/html/2603.03733"
---
# X-Loco
## 一句话总结

> X-Loco: Towards Generalist Humanoid Locomotion Control via Synergetic Policy Distillation 主要落在 [[adaptive-control]]、[[co-training]]、[[人形机器人]]、[[足式运动]] 这一类问题上;从 System0 视角看,它的价值在于把身体状态、接触变化、局部感知或动力学约束接到机器人低层闭环里,让动作生成更稳定、安全、实时。
## 核心贡献

1. 围绕 **X-Loco: Towards Generalist Humanoid Locomotion Control via Synergetic Policy Distillation** 建立了一个与 adaptive-control、co-training、人形机器人、足式运动、recovery、robot-generalization 相关的机器人问题设定,重点不是高层语义规划,而是身体级闭环中的状态变化、接触变化或控制约束。
2. 方法层面的关键价值在于把感知、动力学、约束优化、策略学习或硬件反馈组织成可执行的低层动作生成机制,适合放进 System0 的快反应模块中理解。
3. 对系统集成而言,这类工作补上了 System1 技能选择与电机级控制之间的一层:它负责把上层目标变成动态可行、可恢复、可安全过滤的动作。
4. 若论文包含真实机器人、接触任务或仿真到真实迁移,核心应看它是否报告了闭环稳定性、扰动恢复、硬件延迟、失败模式和跨场景泛化。
## 问题背景

这篇论文可以放在 System0 的“具身快系统”背景下阅读:机器人需要在很短时间内根据本体状态、接触、力/触觉、局部视觉或动力学预测做出动作,而不是等待慢速规划器重新推理整个任务。adaptive-control、co-training、人形机器人、足式运动、recovery、robot-generalization、terrain-adaptation 相关问题的难点通常不在“知道要做什么”,而在动作执行过程中状态会快速偏离、接触会突然改变、约束会实时收紧,低层系统必须及时修正。

### System0 关联

精读时优先看三件事:第一,论文使用哪些快速反馈量,例如关节状态、接触状态、力觉、触觉、深度或局部视觉;第二,反馈如何进入 MPC、WBC、策略网络、安全过滤器或恢复控制器;第三,输出是否直接影响机器人下一步动作,而不是只服务离线分析或高层任务规划。
## 方法详解

### System0 视角解析

从 System0 角度看,这篇论文的方法部分应当围绕 adaptive-control、co-training、人形机器人、足式运动、recovery、robot-generalization、terrain-adaptation、visuomotor-policy 来读。核心问题是:系统如何从快速变化的机器人状态中构造控制输入,如何处理接触、碰撞、约束、模型误差或执行延迟,以及如何把这些信息变成下一时刻的动作。

如果方法里有优化问题,重点看目标函数、约束和求解频率;如果方法里有学习策略,重点看观测空间、动作空间、训练扰动和部署时的闭环频率;如果方法里有硬件或传感器,重点看传感链路是否真的进入控制回路。这样读可以把论文和 System2 语言规划、System1 技能编排区分开来。

### 关键公式与机制

#### 公式 1: [[动力学或策略机制]]

$$J(\pi)=\mathbb{E}_{\tau\sim\pi}\left[\sum_{t=0}^{\infty}\gamma^{t}R(\boldsymbol{s}_{t},\boldsymbol{a}_{t})\right]$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 2: [[优化目标/约束]]

$$a^{*}_{t}=\sum_{(i,\pi_{i})\in\mathcal{C}}\mathbb{I}(i=f(b_{t},I_{t}))\cdot\pi_{i}(\boldsymbol{s}_{i,t}),$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 3: [[优化目标/约束]]

$$\begin{split}\mathcal{L}_{D}=&\ \mathbb{E}_{\tau\sim\mathcal{M}}[(D_{\phi}(\tau)-1)^{2}]+\mathbb{E}_{\tau\sim\mathcal{P}}[(D_{\phi}(\tau)+1)^{2}]\\ &+\frac{\alpha^{d}}{2}\mathbb{E}_{\tau\sim\mathcal{M}}[\lVert \nabla_{\phi}D_{\phi}(\tau)\rVert_{2}^{2}],\end{split}$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 4: [[优化目标/约束]]

$$\exp\left(-\left(\frac{1}{| $\mathcal{B}_{\text{target}}$ |}\sum_{b\in\mathcal{B}_{\text{target}}}\lVert \mathbf{p}_{b}^{\text{des}}-\mathbf{p}_{b} \rVert^{2}\right)/0.3^{2}\right)$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 5: [[优化目标/约束]]

$$\exp\left(-\left(\frac{1}{| $\mathcal{B}_{\text{target}}$ |}\sum_{b\in\mathcal{B}_{\text{target}}}\lVert \mathbf{v}_{b}^{\text{des}}-\mathbf{v}_{b} \rVert^{2}\right)/1.0^{2}\right)$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 6: [[优化目标/约束]]

$$\exp\left(-\left(\frac{1}{| $\mathcal{B}_{\text{target}}$ |}\sum_{b\in\mathcal{B}_{\text{target}}}\lVert \boldsymbol{\omega}_{b}^{\text{des}}-\boldsymbol{\omega}_{b} \rVert^{2}\right)/3.14^{2}\right)$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 7: [[优化目标/约束]]

$$\sum_{j=1}^{N}\left[\max(l_{j}-\theta_{j},0)+\max(\theta_{j}-u_{j},0)\right]$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 8: [[优化目标/约束]]

$$\sum_{\text{hips}}\mathbb{I}(\max| $\theta_{i}$ |>0.9\lor\min| $\theta_{i}$ |<0.8)$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 9: [[优化目标/约束]]

$$\sum_{\text{knees}}\mathbb{I}(\max| $\theta_{i}$ |>2.85\lor\min| $\theta_{i}$ |<-0.06)$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 10: [[动力学或策略机制]]

$$f(b_{t},I_{t})=\begin{cases}c_{\text{rec}},&\text{if}b_{t}<1.1\\ c_{\text{coor}}^{1},&\text{if}b_{t}\geq 1.1\text{and}I_{t}=1\\ c_{\text{coor}}^{2},&\text{if}b_{t}\geq 1.1\text{and}I_{t}=2\\ c_{\text{loco}},&\text{otherwise}\end{cases},$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。
## 关键图表

### Figure 1: Overview of X-Loco. (a) X-Loco integrates the capabilities of three specialist polici

![Figure 1](https://arxiv.org/html/2603.03733v2/x2.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Overview of X-Loco. (a) X-Loco integrates the capabilities of three specialist polici”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 2: Top: Testing the generalist policy on hybrid, challenging terrains. Bottom: Extensib

![Figure 2](https://arxiv.org/html/2603.03733v2/x4.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Top: Testing the generalist policy on hybrid, challenging terrains. Bottom: Extensib”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 3: Left: (a) fall recovery and platform traversal; (b) resilience to external disturbanc

![Figure 3](https://arxiv.org/html/2603.03733v2/x6.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Left: (a) fall recovery and platform traversal; (b) resilience to external disturbanc”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### 论文表格

#### Table 1: TABLE I: Quantitative comparison of X-Loco against baselines and specialist policies. Bold numbers indicates the best pe

| Method | Locomotion | Whole-Body Coordination | Recovery | | | | | | | | |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Slope | Pit | Stairs | | Hanging Bar | Box | | Flat | | | | |
| R succ R_{ $\text{succ}}$ | D trav D_{ $\text{trav}}$ | R succ R_{ $\text{succ}}$ | D trav D_{ $\text{trav}}$ | R succ R_{ $\text{succ}}$ | D trav D_{ $\text{trav}}$ | R  ̄ succ \bar{R}_{ $\text{succ}}$ | R succ R_{ $\text{succ}}$ | R succ R_{ $\text{succ}}$ | R  ̄ succ \bar{R}_{ $\text{succ}}$ | R succ R_{ $\text{succ}}$ | |
| BeyondMimic [28 ] | - | - | - | - | - | - | - | 1.000 ± \pm.000 | 0.916 ± \pm.008 | 0.958 ± \pm.004 | - |
| MoRE [54 ] | 0.992 ± \pm.008 | 7.863 ± \pm.018 | 0.844 ± \pm.009 | 6.833 ± \pm.114 | 0.926 ± \pm.009 | 7.387 ± \pm.013 | 0.921 ± \pm.009 | - | - | - | - |
| PPO [43 ] | 0.823 ± \pm.021 | 6.674 ± \pm.152 | 0.793 ± \pm.018 | 6.396 ± \pm.184 | 0.781 ± \pm.025 | 6.565 ± \pm.141 | 0.799 ± \pm.021 | - | - | - | - |
| AHC [69 ] | 0.968 ± \pm.005 | 7.871 ± \pm.022 | 0.403 ± \pm.031 | 3.179 ± \pm.254 | 0.278 ± \pm.042 | 2.524 ± \pm.311 | 0.550 ± \pm.026 | - | - | - | 1.000 ± \pm.000 |
| Locomotion Specialist | 0.995 ± \pm.002 | 7.989 ± \pm.011 | 0.984 ± \pm.010 | 7.914 ± \pm.135 | 0.991 ± \pm.011 | 7.963 ± \pm.004 | 0.990 ± \pm.008 | - | - | - | - |
| Whole-Body Specialist | - | - | - | - | - | - | - | 1.000 ± \pm.000 | 1.000 ± \pm.000 | 1.000 ± \pm.000 | - |
| Recovery Specialist | - | - | - | - | - | - | - | - | - | - | 1.000 ± \pm.000 |
| Ours (X-Loco) | 0.982 ± \pm.010 | 7.984 ± \pm.033 | 0.878 ± \pm.015 | 7.592 ± \pm.084 | 0.958 ± \pm.007 | 7.853 ± \pm.011 | 0.939 ± \pm.011 | 0.873 ± \pm.014 | 0.868 ± \pm.018 | 0.871 ± \pm.016 | 1.000 ± \pm.000 |

**说明**: TABLE I: Quantitative comparison of X-Loco against baselines and specialist policies. Bold numbers indicates the best performance other than the specialists, and - denotes that the method failed to complete the corresponding task.

#### Table 2: TABLE II: Terrain Configurations for Evaluation

| Skill | Obstacle Properties | Ranges | Unit |
| --- | --- | --- | --- |
| Locomotion | slope incline | [15, 20 ] [15,20] | ∘ |
| pit obstacle height | [0.30, 0.40 ] [0.30,0.40] | m | |
| stair step height | [0.10, 0.15 ] [0.10,0.15] | m | |
| Whole-Body Coordination | obstacle vertical clearance | [0.87, 0.95 ] [0.87,0.95] | m |
| obstacle height | [0.50, 0.65 ] [0.50,0.65] | m | |
| Recovery | flat ground | - | - |

**说明**: TABLE II: Terrain Configurations for Evaluation

#### Table 3: TABLE III: Quantitative results of ablation analysis on CASS and MoE architectures.

| Method | Locomotion | Whole-Body Coordination | Recovery | Average |
| --- | --- | --- | --- | --- |
| R  ̄ succ \bar{R}_{ $\text{succ}}$ | R  ̄ succ \bar{R}_{ $\text{succ}}$ | R succ R_{ $\text{succ}}$ | R  ̄ succ \bar{R}_{ $\text{succ}}$ | |
| Ours w/o CASS | 0.903 | 0.446 | 1.000 | 0.783 |
| Ours w/o MoE | 0.692 | 0.561 | 0.996 | 0.749 |
| MoE-3 | 0.628 | 0.709 | 1.000 | 0.779 |
| MoE-2 (Ours) | 0.939 | 0.845 | 1.000 | 0.928 |

**说明**: TABLE III: Quantitative results of ablation analysis on CASS and MoE architectures.

#### Table 4: TABLE V: PD Controller Parameters and Action Scaling

| Joint Names | K p K_{p} (N  \cdot m/rad) | K d K_{d} (N  \cdot m  \cdot s/rad) | Action Scale |
| --- | --- | --- | --- |
| Hip Roll, Knee | 99.10 99.10 | 6.31 6.31 | 0.35 0.35 |
| Hip Pitch, Hip Yaw, Waist Yaw | 40.18 40.18 | 2.56 2.56 | 0.55 0.55 |
| Ankle (P/R), Waist (R/P) | 28.50 28.50 | 1.81 1.81 | 0.44 0.44 |
| Shoulder (P/R/Y), Elbow | 14.25 14.25 | 0.91 0.91 | 0.44 0.44 |

**说明**: TABLE V: PD Controller Parameters and Action Scaling

#### Table 5: TABLE VI: Domain Randomization Parameters. (U  [⋅ ] $\mathcal{U}[\cdot]$: uniform distribution)

| Domain Randomization | Sampling Distribution | Unit |
| --- | --- | --- |
| Static friction | $\mu_{\text{static}}\sim\mathcal{U}[0.6,1.0]$ | - |
| Dynamic friction | dynamic ∼ U . . $\mu_{\text{dynamic}}\sim\mathcal{U}[0.4,0.8]$ | - |
| Torso payload mass | m payload ∼ U  [- 1.0, 5.0 ] m_{ $\text{payload}}\sim\mathcal{U}[-1.0,5.0]$ | kg |
| Initial joint positions | init ∼ U . . nominal $\theta_{\text{init}}\sim\mathcal{U}[0.8,1.2]\times\theta_{\text{nominal}}$ | - |
| External push interval | $\text{push}}\sim\mathcal{U}[5.0,8.0]$ | s |
| Push linear velocity | v x, v y ∼ U  [- 0.5, 0.5 ], v z ∼ U  [- 0.2, 0.2 ] v_{x},v_{y}\sim $\mathcal{U}[-0.5,0.5],v_{z}\sim\mathcal{U}[-0.2,0.2]$ | m/s |
| Push angular velocity | $\omega_{r},\omega_{p}\sim\mathcal{U}[-0.52,0.52],\omega_{y}\sim\mathcal{U}[-0.78,0.78]$ | rad/s |
| Actuator stiffness K p K_{p} | k p ∼ U . . K p nominal k p \sim $\mathcal{U}[0.8,1.2]\times K_{p,\text{nominal}}$ | - |
| Actuator damping K d K_{d} | k d ∼ U . . K d nominal k d \sim $\mathcal{U}[0.8,1.2]\times K_{d,\text{nominal}}$ | - |

**说明**: 该表格来自论文中的公式、奖励项或实验设置；为避免 PDF 抽取噪声，网页中仅保留可渲染的核心符号和数值。

#### Table 6: TABLE VII: Optimization Hyperparameters for Specialists and Generalist Policies

| Hyperparameter | Specialist Training (PPO) | Generalist Distillation |
| --- | --- | --- |
| Number of environments | 4096 | 4096 |
| Learning rate | 1.0  10 - 3 1.0\times 10^{-3} (Adaptive) | 1.0  10 - 3 1.0\times 10^{-3} (Fixed) |
| Num. epochs per iteration | 5 | 8 |
| Num. mini-batches | 4 | 12 |
| Steps per training batch | 24 | 12 |
| Discount factor $\gamma)$ | 0.99 | - |
| GAE parameter ( \lambda) | 0.95 | - |
| PPO clip parameter ( \epsilon) | 0.2 | - |
| Entropy coefficient | 0.005 | - |
| Desired KL divergence | 0.01 | - |
| Total Training Iterations | | |
| Upright Locomotion l $\pi_{\text{l}})$ | 30,000 | |
| Recovery r $\pi_{\text{r}})$ | 10,000 | |
| Whole-Body Coordination w $\pi_{\text{w}})$ | 50,000 | |
| Generalist Policy g $\pi_{\text{g}})$ | 30,000 | |

**说明**: TABLE VII: Optimization Hyperparameters for Specialists and Generalist Policies

#### Table 7: TABLE VIII: Depth Augmentation and Camera Randomization Parameters. (U  [⋅ ] $\mathcal{U}[\cdot]$: uniform distributio

| Camera Randomization | Value / Range | Unit |
| --- | --- | --- |
| Gaussian noise | $\text{noise}}=0.02$ | m |
| Gaussian filter kernel | k ∈ {3  3 } k\in\{3\times 3\} | pixel |
| Gaussian filter sigma | $\text{filter}}=1.0$ | - |
| Camera position | $\mathcal{U}[-0.05,0.05]$ | m |
| Camera rotation | $\theta_{\text{pitch}}\sim\mathcal{U}[-10,5], θ roll, θ yaw ∼ U [- 1, 1 ] \theta_{\text{roll}},\theta_{\text{yaw}}\sim\mathcal{U}[-1,1]$ | ∘ |
| Camera horizontal FOV | $\text{FOV}_{\text{h}}\sim\mathcal{U}[-10,10]$ | ∘ |

**说明**: 该表格来自论文中的公式、奖励项或实验设置；为避免 PDF 抽取噪声，网页中仅保留可渲染的核心符号和数值。

#### Table 8: TABLE IX: Reward Function Definitions for the Whole-Body Coordination Specialist (π w $\pi_{w})$.

| Reward Terms | Equation | Weight |
| --- | --- | --- |
| Body position | $\exp\left(-\left(\frac{1}{\lVert \mathcal{B}_{\text{target}} \rVert}\sum_{b\in\mathcal{B}_{\text{target}}}\lVert \mathbf{p}_{b}^{\text{des}}-\mathbf{p}_{b} \rVert^{2}\right)/0.3^{2}\right)$ | 1.0 |
| Body orientation | $\exp\left(-\left(\frac{1}{\lVert \mathcal{B}_{\text{target}} \rVert}\sum_{b\in\mathcal{B}_{\text{target}}}\lVert \log(R_{b}^{\text{des}}R_{b}^{\top})\rVert^{2}\right)/0.4^{2}\right)$ | 1.0 |
| Body linear velocity | $\exp\left(-\left(\frac{1}{\lVert \mathcal{B}_{\text{target}} \rVert}\sum_{b\in\mathcal{B}_{\text{target}}}\lVert \mathbf{v}_{b}^{\text{des}}-\mathbf{v}_{b} \rVert^{2}\right)/1.0^{2}\right)$ | 1.0 |
| Body angular velocity | $\exp\left(-\left(\frac{1}{\lVert \mathcal{B}_{\text{target}} \rVert}\sum_{b\in\mathcal{B}_{\text{target}}}\lVert \boldsymbol{\omega}_{b}^{\text{des}}-\boldsymbol{\omega}_{b} \rVert^{2}\right)/3.14^{2}\right)$ | 1.0 |
| Anchor position | $\exp\left(-\lVert \mathbf{p}_{\text{anchor}}^{\text{des}}-\mathbf{p}_{\text{anchor}} \rVert^{2}/0.3^{2}\right)$ | 0.5 |
| Anchor orientation | $\exp\left(-\lVert \log(R_{\text{anchor}}^{\text{des}}R_{\text{anchor}}^{\top})\rVert^{2}/0.4^{2}\right)$ | 0.5 |
| Action smoothness | $\mathbf{a}_{t}-\mathbf{a}_{t-1}\lVert^{2}$ | - 0.1 -0.1 |
| Joint position limit | $\sum_{j=1}^{N}\left[\max(l_{j}-\theta_{j},0)+\max(\theta_{j}-u_{j},0)\right]$ | - 10.0 -10.0 |
| Undesired self-contacts | $\sum_{b\notin\mathcal{B}_{\text{ee}}}\mathbf{1}[\lVert f_{b}^{\text{self}} \rVert>1\text{N}]$ | - 0.1 -0.1 |

**说明**: 该表格来自论文中的公式、奖励项或实验设置；为避免 PDF 抽取噪声，网页中仅保留可渲染的核心符号和数值。

#### Table 9: TABLE X: Reward Function Definitions for the Locomotion Specialist (π l $\pi_{l})$.

| Reward Terms | Equation | Weight |
| --- | --- | --- |
| Track lin. vel. | $\exp\{-\frac{\lVert \boldsymbol{v}_{\text{lin}}^{\text{cmd}}-\boldsymbol{v}_{\text{lin}} \rVert^{2}_{2}}{0.5}\}$ | 5.0 5.0 |
| Track ang. vel. | $\exp\{-\frac{(\boldsymbol{\omega}_{\text{yaw}}^{\text{cmd}}-\boldsymbol{\omega}_{\text{yaw}})^{2}}{0.5}\}$ | 5.0 5.0 |
| Joint acc. | $\theta}\lVert_{2}^{2}$ | - 5  10 - 7 -5\times 10^{-7} |
| Joint vel. | $\dot{\theta}\lVert_{2}^{2}$ | - 1  10 - 3 -1\times 10^{-3} |
| Action rate | $\boldsymbol{a}_{t}-\boldsymbol{a}_{t-1}\lVert_{2}^{2}$ | - 0.03 -0.03 |
| Action smoothness | $\boldsymbol{a}_{t}-2\boldsymbol{a}_{t-1}+\boldsymbol{a}_{t-2}\lVert_{2}^{2}$ | - 0.05 -0.05 |
| Angular vel. (x, y x,y) | $\boldsymbol{\omega}_{xy}\lVert_{2}^{2}$ | - 0.05 -0.05 |
| Orientation | $\boldsymbol{g}_{xy}^{\text{torso}}\lVert_{2}^{2}$ | - 1.5 -1.5 |
| Joint power | $\tau\lVert \rVert\dot{\theta}\lVert^{\top}$ | - 2.5  10 - 5 -2.5\times 10^{-5} |
| Feet stumble | I  (∃ i, \| F i x  y \| ≥ 3  \| F i z \|)  $\mathbb{I}(\exists i,\lVert \boldsymbol{F}_{i}^{xy} \rVert\geq 3\lVert F_{i}^{z} \rVert)$ | - 1.0 -1.0 |
| Torques | $\sum_{\text{all joints}}\tau_{i}^{2}$ | - 1  10 - 5 -1\times 10^{-5} |
| Joint deviation | $\sum_{k\in\{\text{arm, waist, hip}\}}w_{k}\sum_{j\in\mathcal{J}_{k}}\lVert \theta_{j}-\theta_{j}^{\text{def}} \rVert$ | - 0.5 -0.5 |
| Joint pos. limits | ∑ all joints o  u  t i  $\sum_{\text{all joints}}\boldsymbol{out}_{i}$ | - 2.0 -2.0 |
| Joint vel. limits | $\text{ReLU}(\dot{\theta}-\dot{\theta}^{\text{max}})$ | - 1.0 -1.0 |
| Torque limits | $\text{ReLU}(\tau-\tau^{\text{max}})$ | - 1.0 -1.0 |
| Feet lateral distance | \| (y left feet b - y right feet b) - 0.2 \| \|(y^{b}_{ $\text{left feet}}-y^{b}_{\text{right feet}})-0.2\lVert$ | 0.5 0.5 |
| Feet slippage | ∑ feet v i foot I contact $\sum_{\text{feet}}\lVert \boldsymbol{v}_{i}^{\text{foot}} \rVert\cdot\mathbb{I}_{\text{contact}}$ | - 0.25 -0.25 |
| Collision | n collision n_{ $\text{collision}}$ | - 15.0 -15.0 |
| Feet air time | ∑ foot t i air - . I first contact i $\sum_{\text{foot}}(t^{\text{air}}_{i}-0.5)\cdot\mathbb{I}(\text{first contact}_{i})$ | 1.0 1.0 |
| Stuck | $\boldsymbol{v}\lVert_{2}\leq 0.1)\cdot(\rVert\boldsymbol{c}^{v}\lVert_{2}\geq 0.2)$ | - 1.0 -1.0 |
| Feet clearance | $\sum_{\text{foot}}((z^{i}-h^{\text{target}})^{2}\cdot\lVert \boldsymbol{v}^{i}_{xy} \rVert)$ | 2.0 2.0 |
| Alive | 1 1 | 2 2 |
| AMP reward | max - D - \displaystyle $\text{max}\left[0,1-\frac{1}{4}(D_{\phi}(\tau)-1)^{2}\right]$ | 3.0 3.0 |

**说明**: 该表格来自论文中的公式、奖励项或实验设置；为避免 PDF 抽取噪声，网页中仅保留可渲染的核心符号和数值。

#### Table 10: TABLE XI: Reward Function Definitions for the Recovery Specialist (π r $\pi_{r})$. The f tol f_{$\text{tol}$} adopts a Gau

| Reward Terms | Equation | Weight |
| --- | --- | --- |
| Task Rewards | w task = 1.0 w^{ $\text{task}}=1.0$ | |
| Orientation | $\text{tol}}(-\theta^{z}_{\text{base}},[0.99,\infty],1,0.05)$ | 1.0 1.0 |
| Head height | f tol  (h head, [1, ∞ ], 1, 0.1) f_{ $\text{tol}}(h_{\text{head}},[1,\infty],1,0.1)$ | 1.0 1.0 |
| Style Rewards | w style = 1.0 w^{ $\text{style}}=1.0$ | |
| Hip joint deviation | $\theta_{i}\lVert<0.8)$ | - 10.0 -10.0 |
| Knee deviation | $\theta_{i}\lVert<-0.06)$ | - 0.25 -0.25 |
| Shoulder roll dev. | $\mathbb{I}(\theta_{\text{left}} 0.02)$ | - 2.5 -2.5 |
| Thigh orientation | $\text{tol}}(\frac{1}{2}\sum_{\text{thighs}}(\theta^{\text{z}}_{\text{thigh}}),[0.8,\infty],1,0.1)$ | 10.0 10.0 |
| Feet distance | $\mathbb{I}(\lVert \mathbf{p}^{xy}_{\text{left\_f}}-\mathbf{p}^{xy}_{\text{right\_f}} \rVert^{2}>0.9)$ | - 10.0 -10.0 |
| Angular vel. (x, y x,y) | $\exp(-2\lVert \boldsymbol{\omega}_{xy} \rVert^{2}_{2})\cdot\mathbb{I}(h_{\text{base}}>h_{\text{stage1}})$ | 25.0 25.0 |
| Foot displacement | $\exp(\text{clip}(-2\lVert \mathbf{q}^{xy}_{\text{base}}-\mathbf{q}^{xy}_{\text{foot}} \rVert^{2},0.3,\infty))$ | 2.5 2.5 |
| AMP reward | $\left[0,1-\frac{1}{4}(D_{\phi}(\tau)-1)^{2}\right]$ | 80.0 80.0 |
| Regularization Rewards | w regu = 1.0 w^{ $\text{regu}}=1.0$ | |
| Joint acc. | $\theta}\lVert_{2}^{2}$ | - 2.5  e-  7 -2.5 $\text{e-}7$ |
| Joint vel. | $\dot{\theta}\lVert_{2}^{2}$ | - 1  e-  3 -1 $\text{e-}3$ |
| Action rate | $\mathbf{a}_{t}-\mathbf{a}_{t-1}\lVert_{2}^{2}$ | - 0.01 -0.01 |
| Action smoothness | $\mathbf{a}_{t}-2\mathbf{a}_{t-1}+\mathbf{a}_{t-2}\lVert_{2}^{2}$ | - 0.05 -0.05 |
| Torques | $\sum\tau_{i}^{2}$ | - 1  10 - 5 -1\times 10^{-5} |
| Joint power | $\tau\lVert \rVert\dot{\theta}\lVert^{\top}$ | - 2.5  10 - 5 -2.5\times 10^{-5} |
| Joint pos. limits | $\sum\text{ReLU}(\lVert \theta_{i} \rVert-\theta_{i}^{\text{max}})$ | - 2.0 -2.0 |
| Joint vel. limits | $\text{ReLU}(\dot{\theta}-\dot{\theta}^{\text{max}})$ | - 1.0 -1.0 |
| Post-Task Rewards (Conditioned on h base > h stage3 h_{ $\text{base}}>h_{\text{stage3}})$ | w task = 1.0 w^{ $\text{task}}=1.0$ | |
| Tracking errors | $\exp(-2\lVert \boldsymbol{\omega}_{xy} \rVert_{2}^{2}),\exp(-5\lVert \mathbf{v}_{xy} \rVert_{2}^{2}),\exp(-5\lVert \mathbf{g}_{xy} \rVert_{2}^{2})$ | 10.0 10.0 |
| Base height | exp  (- 20  \| h base - 0.75 \|)  $\exp(-20\lVert h_{\text{base}}-0.75 \rVert)$ | 10.0 10.0 |
| Target joint dev. | $\exp(-0.1\sum(\theta_{i}-\theta_{i}^{\text{def}})^{2})$ | 10.0 10.0 |
| Target feet dist. | f tol  (y L - y R, [0.3, 0.4 ], 0.1, 0.05) f_{ $\text{tol}}(y_{\text{L}}-y_{\text{R}},[0.3,0.4],0.1,0.05)$ | - 5.0 -5.0 |

**说明**: 该表格来自论文中的公式、奖励项或实验设置；为避免 PDF 抽取噪声，网页中仅保留可渲染的核心符号和数值。
## 实验解读

- 评价重点:围绕 adaptive-control、co-training、人形机器人、足式运动、recovery,优先看真实机器人结果、扰动恢复、接触稳定性、控制误差、成功率、实时性和 sim-to-real 差距。
- System0 视角:实验结论不只看最终成功率,还要看闭环过程中是否能处理状态估计误差、执行延迟、碰撞/接触突变和外部扰动。
- 与上层系统的关系:如果方法能在快闭环内稳定工作,它可以作为 System1 技能或 System2 规划的底层执行保障;如果只在离线或慢速设定下成立,则还不能直接作为身体级快闭环模块。
## 批判性思考

### 优点

- 论文把 adaptive-control、co-training、人形机器人、足式运动、recovery 放到机器人身体级闭环里讨论,和只做高层语义推理的工作相比,更接近真实机器人部署时会遇到的状态、动力学和安全约束。
- 若论文同时给出公式、系统结构图、硬件实验或消融结果,这些证据可以帮助判断方法是否真的能支撑毫秒到百毫秒级反应。

### 局限

- 需要重点核查控制频率、推理延迟、传感延迟、失败恢复、硬件磨损、未见场景泛化和安全边界;这些因素决定它能否成为可靠的 System0 模块。
- 对学习型方法,还要区分训练环境中的鲁棒性和真实部署中的鲁棒性,尤其是接触丰富任务、动态运动任务和人形机器人任务。
## 论文定位

- 研究对象:X-Loco: Towards Generalist Humanoid Locomotion Control via Synergetic Policy Distillation。
- 关键词:adaptive-control、co-training、人形机器人、足式运动、recovery、robot-generalization、terrain-adaptation、visuomotor-policy、全身控制。
- System0 关联:关注机器人身体级快反应链路中“状态/接触/局部感知 -> 约束/优化/策略 -> 动作”的短闭环,不把问题停留在语言规划或任务分解层。

## 速查卡片

> [!summary] X-Loco
> - **论文**: https://www.roboticsproceedings.org/rss22/p022.pdf
> - **arXiv**: http://arxiv.org/abs/2603.03733
> - **arXiv HTML**: https://arxiv.org/html/2603.03733
