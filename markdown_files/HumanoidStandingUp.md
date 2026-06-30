---
title: "Learning Humanoid Standing-up Control across Diverse Postures"
method_name: "Humanoid Standing up"
authors: ["Tao Huang"]
year: 2025
venue: "RSS"
tags: ["contact-reasoning", "robot-manipulation", "robust-control", "legged-locomotion", "reinforcement-learning", "adaptive-control", "humanoid", "sim-to-real", "loco-manipulation", "recovery"]
image_source: "online"
arxiv_html: "https://arxiv.org/html/2502.08378v2"
---
# Humanoid Standing up
## 一句话总结

> Learning Humanoid Standing-up Control across Diverse Postures 主要落在 [[adaptive-control]]、[[接触推理]]、[[人形机器人]]、[[足式运动]] 这一类问题上;从 System0 视角看,它的价值在于把身体状态、接触变化、局部感知或动力学约束接到机器人低层闭环里,让动作生成更稳定、安全、实时。
## 核心贡献

1. 围绕 **Learning Humanoid Standing-up Control across Diverse Postures** 建立了一个与 adaptive-control、接触推理、人形机器人、足式运动、移动操作、recovery 相关的机器人问题设定,重点不是高层语义规划,而是身体级闭环中的状态变化、接触变化或控制约束。
2. 方法层面的关键价值在于把感知、动力学、约束优化、策略学习或硬件反馈组织成可执行的低层动作生成机制,适合放进 System0 的快反应模块中理解。
3. 对系统集成而言,这类工作补上了 System1 技能选择与电机级控制之间的一层:它负责把上层目标变成动态可行、可恢复、可安全过滤的动作。
4. 若论文包含真实机器人、接触任务或仿真到真实迁移,核心应看它是否报告了闭环稳定性、扰动恢复、硬件延迟、失败模式和跨场景泛化。
## 问题背景

这篇论文可以放在 System0 的“具身快系统”背景下阅读:机器人需要在很短时间内根据本体状态、接触、力/触觉、局部视觉或动力学预测做出动作,而不是等待慢速规划器重新推理整个任务。adaptive-control、接触推理、人形机器人、足式运动、移动操作、recovery、强化学习 相关问题的难点通常不在“知道要做什么”,而在动作执行过程中状态会快速偏离、接触会突然改变、约束会实时收紧,低层系统必须及时修正。

### System0 关联

精读时优先看三件事:第一,论文使用哪些快速反馈量,例如关节状态、接触状态、力觉、触觉、深度或局部视觉;第二,反馈如何进入 MPC、WBC、策略网络、安全过滤器或恢复控制器;第三,输出是否直接影响机器人下一步动作,而不是只服务离线分析或高层任务规划。
## 方法详解

### System0 视角解析

从 System0 角度看,这篇论文的方法部分应当围绕 adaptive-control、接触推理、人形机器人、足式运动、移动操作、recovery、强化学习、机器人操作 来读。核心问题是:系统如何从快速变化的机器人状态中构造控制输入,如何处理接触、碰撞、约束、模型误差或执行延迟,以及如何把这些信息变成下一时刻的动作。

如果方法里有优化问题,重点看目标函数、约束和求解频率;如果方法里有学习策略,重点看观测空间、动作空间、训练扰动和部署时的闭环频率;如果方法里有硬件或传感器,重点看传感链路是否真的进入控制回路。这样读可以把论文和 System2 语言规划、System1 技能编排区分开来。

### 关键公式与机制

#### 公式 1: [[动力学或策略机制]]

$$\mathbb{E}_{\pi_{\theta}}[\sum_{t=0}^{T-1}\gamma^{t}r_{t}]$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 2: [[动力学或策略机制]]

$$\tau_{t}=K_{p}\cdot(p_{t}^{d}-p_{t})-K_{d}\cdot\dot{p}_{t},$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 3: [[优化目标/约束]]

$$\mathcal{L}(\theta)=\mathbb{E}\left[\min\left(\alpha_{t}(\theta)A_{t},\mathrm{clip}(\alpha_{t}(\theta),1-\epsilon,1+\epsilon)A_{t}\right)\right],$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 4: [[优化目标/约束]]

$$
\mathcal{L}_{\mathrm{L2C2}}=\lambda_{\pi}D(\pi_{\theta}(s_{t}),\pi_{\theta}(\bar{s}_{t}))+\lambda_{V}\sum D(V_{\phi_{i}}(s_{t}),V_{\phi_{i}}(\bar{s}_{t})),
$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 5: [[动力学或策略机制]]

$$\sum_{t=0}^{h_{\mathrm{base}}<H_{\mathrm{stage2}}}| $\boldsymbol{\tau}_{t}$ |\cdot| $\dot{p}_{t}$ |^{T}\mathrm{dt}$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 6: [[优化目标/约束]]

$$\mathcal{L}(\phi_{i})=\mathbb{E}\big{[}\lVert r_{t}^{i}+\gamma V_{\phi_{i}}(s_{t})-\bar{V}_{\phi_{i}}(s_{t+1})\rVert^{2}\big{]},$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 7: [[动力学或策略机制]]

$$r_{t}=w^{\mathrm{task}}\cdot r^{\mathrm{task}}_{t}+w^{\mathrm{style}}\cdot r^{\mathrm{style}}_{t}+w^{\mathrm{regu}}\cdot r^{\mathrm{regu}}_{t}+w^{\mathrm{post}}\cdot r^{\mathrm{post}}_{t},$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 8: [[动力学或策略机制]]

$$\exp\left(-2\times\lVert \boldsymbol{q}_{\mathrm{base}}^{\mathrm{xy}}-\boldsymbol{q}_{\mathrm{foot}}^{\mathrm{xy}} \rVert^{2}.\mathrm{clip}(0.3,\mathrm{inf})\right)\times\mathbf{1}(h_{\mathrm{base}}>H_{\mathrm{stage2}})$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 9: [[优化目标/约束]]

$$\mathbf{1}(\max(\boldsymbol{q}_{\mathrm{knee}}^{\mathrm{l,r}})>2.85)\;|\;\mathbf{1}(\min(\boldsymbol{q}_{\mathrm{knee}}^{\mathrm{l,r}})<-0.06)$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 10: [[优化目标/约束]]

$$\mathbf{1}(\max(| $\boldsymbol{q}_{\mathrm{hip}}^{\mathrm{l,r}}$ |)>1.4)\;| \; $\mathbf{1}(\min($ | $\boldsymbol{q}_{\mathrm{hip}}^{\mathrm{l,r}}$ |)>0.9)$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。
## 关键图表

### Figure 1: Framework. (a) We train policies in simulation from scratch with multiple cr

![Figure 1](https://arxiv.org/html/2502.08378v2/x1.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Framework. (a) We train policies in simulation from scratch with multiple cr”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 2: Motion analysis in simulation. (Left) UMAP visualization of joint-space trajectories

![Figure 2](https://arxiv.org/html/2502.08378v2/x2.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Motion analysis in simulation. (Left) UMAP visualization of joint-space trajectories”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 3: Trade-off analysis in simulation. Trade-offs between motion speed, smoothness, and en

![Figure 3](https://arxiv.org/html/2502.08378v2/x4.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Trade-off analysis in simulation. Trade-offs between motion speed, smoothness, and en”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### 论文表格

#### Table 1: TABLE I: Comparison with existing methods on standing-up control.

| Method | Real Robot | w/o Prior Trajectory | Beyond Ground | High DoF | 1-stage Training |
| --- | --- | --- | --- | --- | --- |
| Peng et al. [36 ] | ✗ | ✗ | ✗ | ✓ | ✗ |
| Yang et al. [52 ] | ✗ | ✗ | ✗ | ✓ | ✓ |
| Tao et al. [46 ] | ✗ | ✓ | ✗ | ✓ | ✗ |
| Haarnoja et al. [12 ] | ✓ | ✗ | ✗ | ✓ | ✓ |
| Gaspard et al. [10 ] | ✓ | ✓ | ✗ | ✗ | ✓ |
| HoST (ours) | ✓ | ✓ | ✓ | ✓ | ✓ |

**说明**: TABLE I: Comparison with existing methods on standing-up control.

#### Table 2: TABLE II: Domain randomization settings for standing-up control.

| Term | Value |
| --- | --- |
| Trunk Mass | U (- 2, 5) U 2 5  $\mathcal{U}(-2,5) (- 2, 5) kg$ |
| Base CoM offset | U (- d, d) U d d  $\mathcal{U}(-d,d)$ |
| Link mass | $\mathcal{U}(-0.8,1.2)\times (- 0.8, 1.2) × default kg$ |
| Fiction | U (0.1, 1) U 0.1 1  $\mathcal{U}(0.1,1)$ |
| Restitution | U (0, 1) U 0 1  $\mathcal{U}(0,1)$ |
| P Gain | U (0.85, 1.15) U 0.85 1.15  $\mathcal{U}(0.85,1.15)$ |
| D Gain | U (0.85, 1.15) U 0.85 1.15  $\mathcal{U}(0.85,1.15)$ |
| Torque RFI [2 ] | $\mathcal{U}(-0.05,0.05)\times (- 0.05, 0.05) × torque limit N ⋅ ⋅ \cdot ⋅ m$ |
| Motor Strength | U (0.9, 1.1) U 0.9 1.1  $\mathcal{U}(0.9,1.1)$ |
| Control delay | U (0, 100) U 0 100  $\mathcal{U}(0,100) (0, 100) ms$ |
| Initial joint angle offset | U (- 0.1, 0.1) U 0.1 0.1  $\mathcal{U}(-0.1,0.1) (- 0.1, 0.1) rad$ |
| Initial joint angle scale | $\mathcal{U}(0.9,1.1)\times (0.9, 1.1) × default joint angle rad$ |

**说明**: TABLE II: Domain randomization settings for standing-up control.

#### Table 3: TABLE III: Main simulation results. We present a performance comparison between HoST and baselines for the proposed metr

| Method | | Ground | | Platform | | Wall | | Slope | | | | | | | | | | | | | |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| | E succ ↑ ↑ E succ absent E_{ $\mathrm{succ}}\uparrow ↑$ | E feet ↓ ↓ E feet absent E_{ $\mathrm{feet}}\downarrow ↓$ | E smth ↓ ↓ E smth absent E_{ $\mathrm{smth}}\downarrow ↓$ | E engy ↓ ↓ E engy absent E_{ $\mathrm{engy}}\downarrow ↓$ | | E succ E succ E_{ $\mathrm{succ}} ↑ ↑ \uparrow ↑$ | E feet E feet E_{ $\mathrm{feet}} ↑ ↑ \uparrow ↑$ | E smth E smth E_{ $\mathrm{smth}} ↓ ↓ \downarrow ↓$ | E engy E engy E_{ $\mathrm{engy}} ↓ ↓ \downarrow ↓$ | | E succ E succ E_{ $\mathrm{succ}} ↑ ↑ \uparrow ↑$ | E feet E feet E_{ $\mathrm{feet}} ↑ ↑ \uparrow ↑$ | E smth E smth E_{ $\mathrm{smth}}. ↓ ↓ \downarrow ↓$ | E engy E engy E_{ $\mathrm{engy}}. ↓ ↓ \downarrow ↓$ | | E succ E succ E_{ $\mathrm{succ}} ↑ ↑ \uparrow ↑$ | E smth E smth E_{ $\mathrm{smth}} ↑ ↑ \uparrow ↑$ | E smth E smth E_{ $\mathrm{smth}} ↓ ↓ \downarrow ↓$ | E engy E engy E_{ $\mathrm{engy}} ↓ ↓ \downarrow ↓$ | | |
| (a) Ablation on Number of Critics | | | | | | | | | | | | | | | | | | | | | |
| HoST -w/o-MuC | | 0.0 (± 0.0 plus-or-minus 0.0 \pm 0.0 ± 0.0) | / | / | / | | 0.0 (± 0.0 plus-or-minus 0.0 \pm 0.0 ± 0.0) | / | / | / | | 0.0 (± 0.0 plus-or-minus 0.0 \pm 0.0 ± 0.0) | / | / | / | | 0.0 (± 0.0 plus-or-minus 0.0 \pm 0.0 ± 0.0) | / | / | / | |
| HoST | | 99.5 (± 0.4 plus-or-minus 0.4 \pm 0.4 ± 0.4) | 1.52 (±.10 plus-or-minus.10 \pm.10 ±.10) | 2.90 (±.21 plus-or-minus.21 \pm.21 ±.21) | 1.35 (±.02 plus-or-minus.02 \pm.02 ±.02) | | 99.8 (± 0.2 plus-or-minus 0.2 \pm 0.2 ± 0.2) | 1.16 (±.04 plus-or-minus.04 \pm.04 ±.04) | 3.39 (±.39 plus-or-minus.39 \pm.39 ±.39) | 0.58 (±.01 plus-or-minus.01 \pm.01 ±.01) | | 94.2 (± 1.2 plus-or-minus 1.2 \pm 1.2 ± 1.2) | 1.14 (±.08 plus-or-minus.08 \pm.08 ±.08) | 4.66 (±.69 plus-or-minus.69 \pm.69 ±.69) | 1.08 (±.02 plus-or-minus.02 \pm.02 ±.02) | | 98.5 (± 0.4 plus-or-minus 0.4 \pm 0.4 ± 0.4) | 5.71 (±.24 plus-or-minus.24 \pm.24 ±.24) | 5.31 (±.45 plus-or-minus.45 \pm.45 ±.45) | 0.83 (±.01 plus-or-minus.01 \pm.01 ±.01) | |
| (b) Ablation on Exploration Strategy | | | | | | | | | | | | | | | | | | | | | |
| HoST -w/o-Force | | 0.0 (± 0.0 plus-or-minus 0.0 \pm 0.0 ± 0.0) | / | / | / | | 6.8 (± 2.0 plus-or-minus 2.0 \pm 2.0 ± 2.0) | 0.12 (±.02 plus-or-minus.02 \pm.02 ±.02) | 3.39 (±.40 plus-or-minus.40 \pm.40 ±.40) | 1.98 (±.02 plus-or-minus.02 \pm.02 ±.02) | | 0.0 (± 0.0 plus-or-minus 0.0 \pm 0.0 ± 0.0) | / | / | / | | 0.0 (± 0.0 plus-or-minus 0.0 \pm 0.0 ± 0.0) | / | / | / | |
| HoST -w/o-Force-RND | | 19.8 (± 1.2 plus-or-minus 1.2 \pm 1.2 ± 1.2) | 0.87 (±.11 plus-or-minus.11 \pm.11 ±.11) | 3.13 (±.18 plus-or-minus.18 \pm.18 ±.18) | 2.55 (±.03 plus-or-minus.03 \pm.03 ±.03) | | 99.5 (± 0.4 plus-or-minus 0.4 \pm 0.4 ± 0.4) | 1.66 (±.11 plus-or-minus.11 \pm.11 ±.11) | 3.55 (±.37 plus-or-minus.37 \pm.37 ±.37) | 0.78 (±.01 plus-or-minus.01 \pm.01 ±.01) | | 0.0 (± 0.0 plus-or-minus 0.0 \pm 0.0 ± 0.0) | / | / | / | | 0.0 (± 0.0 plus-or-minus 0.0 \pm 0.0 ± 0.0) | / | / | / | |
| HoST | | 99.5 (± 0.4 plus-or-minus 0.4 \pm 0.4 ± 0.4) | 1.52 (± 0.10 plus-or-minus 0.10 \pm 0.10 ± 0.10) | 2.90 (±.21 plus-or-minus.21 \pm.21 ±.21) | 1.35 (±.02 plus-or-minus.02 \pm.02 ±.02) | | 99.8 (± 0.2 plus-or-minus 0.2 \pm 0.2 ± 0.2) | 1.16 (±.04 plus-or-minus.04 \pm.04 ±.04) | 3.39 (±.39 plus-or-minus.39 \pm.39 ±.39) | 0.58 (±.01 plus-or-minus.01 \pm.01 ±.01) | | 94.2 (± 1.2 plus-or-minus 1.2 \pm 1.2 ± 1.2) | 1.14 (±.08 plus-or-minus.08 \pm.08 ±.08) | 4.66 (±.69 plus-or-minus.69 \pm.69 ±.69) | 1.08 (±.02 plus-or-minus.02 \pm.02 ±.02) | | 98.1 (± 0.4 plus-or-minus 0.4 \pm 0.4 ± 0.4) | 5.71 (±.24 plus-or-minus.24 \pm.24 ±.24) | 5.44 (±.45 plus-or-minus.45 \pm.45 ±.45) | 0.89 (±.01 plus-or-minus.01 \pm.01 ±.01) | |
| (c) Ablation on Motion Constraints | | | | | | | | | | | | | | | | | | | | | |
| HoST -w/o-Bound | | 98.8 (± 0.6 plus-or-minus 0.6 \pm 0.6 ± 0.6) | 7.27 (±.42 plus-or-minus.42 \pm.42 ±.42) | 9.52 (±.25 plus-or-minus.25 \pm.25 ±.25) | 3.59 (±.02 plus-or-minus.02 \pm.02 ±.02) | | 99.4 (± 0.8 plus-or-minus 0.8 \pm 0.8 ± 0.8) | 6.23 (±.34 plus-or-minus.34 \pm.34 ±.34) | 11.65 (±.34 plus-or-minus.34 \pm.34 ±.34) | 1.76 (±.03 plus-or-minus.03 \pm.03 ±.03) | | 99.6 (± 0.5 plus-or-minus 0.5 \pm 0.5 ± 0.5) | 5.48 (±.70 plus-or-minus.70 \pm.70 ±.70) | 8.80 (±.74 plus-or-minus.74 \pm.74 ±.74) | 1.73 (±.02 plus-or-minus.02 \pm.02 ±.02) | | 82.4 (± 4.4 plus-or-minus 4.4 \pm 4.4 ± 4.4) | 32.22 (± 2.5 plus-or-minus 2.5 \pm 2.5 ± 2.5) | 16.44 (±.86 plus-or-minus.86 \pm.86 ±.86) | 2.62 (±.07 plus-or-minus.07 \pm.07 ±.07) | |
| HoST -Bound0.25 | | 99.8 (± 0.4 plus-or-minus 0.4 \pm 0.4 ± 0.4) | 1.16 (±.08 plus-or-minus.08 \pm.08 ±.08) | 2.75 (±.19 plus-or-minus.19 \pm.19 ±.19) | 1.56 (±.01 plus-or-minus.01 \pm.01 ±.01) | | 99.8 (± 0.1 plus-or-minus 0.1 \pm 0.1 ± 0.1) | 0.68 (±.05 plus-or-minus.05 \pm.05 ±.05) | 3.17 (±.41 plus-or-minus.41 \pm.41 ±.41) | 0.79 (±.02 plus-or-minus.02 \pm.02 ±.02) | | 84.6 (± 2.5 plus-or-minus 2.5 \pm 2.5 ± 2.5) | 0.42 (±.02 plus-or-minus.02 \pm.02 ±.02) | 4.23 (±.71 plus-or-minus.71 \pm.71 ±.71) | 1.44 (±.04 plus-or-minus.04 \pm.04 ±.04) | | 98.0 (± 1.4 plus-or-minus 1.4 \pm 1.4 ± 1.4) | 2.74 (±.16 plus-or-minus.16 \pm.16 ±.16) | 4.67 (±.42 plus-or-minus.42 \pm.42 ±.42) | 0.90 (±.02 plus-or-minus.02 \pm.02 ±.02) | |
| HoST -w/o-L2C2 | | 92.3 (± 0.7 plus-or-minus 0.7 \pm 0.7 ± 0.7) | 2.29 (±.06 plus-or-minus.06 \pm.06 ±.06) | 4.05 (±.21 plus-or-minus.21 \pm.21 ±.21) | 1.43 (±.01 plus-or-minus.01 \pm.01 ±.01) | | 99.8 (± 0.0 plus-or-minus 0.0 \pm 0.0 ± 0.0) | 1.93 (±.07 plus-or-minus.07 \pm.07 ±.07) | 4.47 (±.42 plus-or-minus.42 \pm.42 ±.42) | 0.92 (±.02 plus-or-minus.02 \pm.02 ±.02) | | 97.8 (± 1.6 plus-or-minus 1.6 \pm 1.6 ± 1.6) | 1.43 (±.16 plus-or-minus.16 \pm.16 ±.16) | 5.29 (±.70 plus-or-minus.70 \pm.70 ±.70) | 1.55 (±.02 plus-or-minus.02 \pm.02 ±.02) | | 98.8 (± 0.8 plus-or-minus 0.8 \pm 0.8 ± 0.8) | 3.93 (±.24 plus-or-minus.24 \pm.24 ±.24) | 6.32 (±.46 plus-or-minus.46 \pm.46 ±.46) | 1.12 (±.02 plus-or-minus.02 \pm.02 ±.02) | |
| HoST -w/o- r style r style r^{ $\mathrm{style}}$ | | 99.2 (± 0.5 plus-or-minus 0.5 \pm 0.5 ± 0.5) | 1.36 (±.07 plus-or-minus.07 \pm.07 ±.07) | 2.83 (±.21 plus-or-minus.21 \pm.21 ±.21) | 1.67 (±.03 plus-or-minus.03 \pm.03 ±.03) | | 82.2 (± 3.5 plus-or-minus 3.5 \pm 3.5 ± 3.5) | 1.18 (±.08 plus-or-minus.08 \pm.08 ±.08) | 3.56 (±.40 plus-or-minus.40 \pm.40 ±.40) | 0.67 (±.03 plus-or-minus.03 \pm.03 ±.03) | | 0.0 (± 0.0 plus-or-minus 0.0 \pm 0.0 ± 0.0) | / | / | / | | 21.4 (± 3.2 plus-or-minus 3.2 \pm 3.2 ± 3.2) | 8.61 (±.12 plus-or-minus.12 \pm.12 ±.12) | 6.49 (±.54 plus-or-minus.54 \pm.54 ±.54) | 1.69 (±.05 plus-or-minus.05 \pm.05 ±.05) | |
| HoST | | 99.5 (± 0.4 plus-or-minus 0.4 \pm 0.4 ± 0.4) | 1.52 (±.10 plus-or-minus.10 \pm.10 ±.10) | 2.90 (±.21 plus-or-minus.21 \pm.21 ±.21) | 1.35 (±.02 plus-or-minus.02 \pm.02 ±.02) | | 99.8 (± 0.2 plus-or-minus 0.2 \pm 0.2 ± 0.2) | 1.16 (±.04 plus-or-minus.04 \pm.04 ±.04) | 3.39 (±.39 plus-or-minus.39 \pm.39 ±.39) | 0.58 (±.01 plus-or-minus.01 \pm.01 ±.01) | | 94.2 (± 1.2 plus-or-minus 1.2 \pm 1.2 ± 1.2) | 1.14 (±.08 plus-or-minus.08 \pm.08 ±.08) | 4.66 (±.69 plus-or-minus.69 \pm.69 ±.69) | 1.08 (±.02 plus-or-minus.02 \pm.02 ±.02) | | 98.5 (± 0.4 plus-or-minus 0.4 \pm 0.4 ± 0.4) | 5.71 (±.24 plus-or-minus.24 \pm.24 ±.24) | 5.31 (±.45 plus-or-minus.45 \pm.45 ±.45) | 0.83 (±.01 plus-or-minus.01 \pm.01 ±.01) | |
| (d) Ablation on Historical States | | | | | | | | | | | | | | | | | | | | | |
| HoST -History0 | | 98.1 (± 1.4 plus-or-minus 1.4 \pm 1.4 ± 1.4) | 2.11 (±.14 plus-or-minus.14 \pm.14 ±.14) | 2.72 (±.22 plus-or-minus.22 \pm.22 ±.22) | 1.27 (±.02 plus-or-minus.02 \pm.02 ±.02) | | 99.5 (± 0.5 plus-or-minus 0.5 \pm 0.5 ± 0.5) | 1.53 (±.13 plus-or-minus.13 \pm.13 ±.13) | 3.29 (±.40 plus-or-minus.40 \pm.40 ±.40) | 0.47 (±.01 plus-or-minus.01 \pm.01 ±.01) | | 64.5 (± 1.2 plus-or-minus 1.2 \pm 1.2 ± 1.2) | 1.66 (±.04 plus-or-minus.04 \pm.04 ±.04) | 4.74 (±.72 plus-or-minus.72 \pm.72 ±.72) | 1.66 (±.03 plus-or-minus.03 \pm.03 ±.03) | | 97.4 (± 2.0 plus-or-minus 2.0 \pm 2.0 ± 2.0) | 5.20 (±.24 plus-or-minus.24 \pm.24 ±.24) | 4.97 (±.48 plus-or-minus.48 \pm.48 ±.48) | 0.66 (±.02 plus-or-minus.02 \pm.02 ±.02) | |
| HoST -History2 | | 99.3 (± 0.3 plus-or-minus 0.3 \pm 0.3 ± 0.3) | 2.25 (±.13 plus-or-minus.13 \pm.13 ±.13) | 2.56 (±.19 plus-or-minus.19 \pm.19 ±.19) | 1.16 (±.01 plus-or-minus.01 \pm.01 ±.01) | | 99.4 (± 0.5 plus-or-minus 0.5 \pm 0.5 ± 0.5) | 0.77 (±.39 plus-or-minus.39 \pm.39 ±.39) | 3.27 (±.39 plus-or-minus.39 \pm.39 ±.39) | 0.60 (±.01 plus-or-minus.01 \pm.01 ±.01) | | 93.7 (± 1.4 plus-or-minus 1.4 \pm 1.4 ± 1.4) | 1.79 (±.08 plus-or-minus.08 \pm.08 ±.08) | 4.81 (±.71 plus-or-minus.71 \pm.71 ±.71) | 1.22 (±.01 plus-or-minus.01 \pm.01 ±.01) | | 98.6 (± 0.6 plus-or-minus 0.6 \pm 0.6 ± 0.6) | 5.06 (±.24 plus-or-minus.24 \pm.24 ±.24) | 5.35 (±.44 plus-or-minus.44 \pm.44 ±.44) | 0.77 (±.01 plus-or-minus.01 \pm.01 ±.01) | |
| HoST -History5 (ours) | | 99.5 (± 0.4 plus-or-minus 0.4 \pm 0.4 ± 0.4) | 1.52 (±.10 plus-or-minus.10 \pm.10 ±.10) | 2.90 (±.21 plus-or-minus.21 \pm.21 ±.21) | 1.35 (±.02 plus-or-minus.02 \pm.02 ±.02) | | 99.8 (± 0.2 plus-or-minus 0.2 \pm 0.2 ± 0.2) | 1.16 (±.04 plus-or-minus.04 \pm.04 ±.04) | 3.39 (±.39 plus-or-minus.39 \pm.39 ±.39) | 0.58 (±.01 plus-or-minus.01 \pm.01 ±.01) | | 94.2 (± 1.2 plus-or-minus 1.2 \pm 1.2 ± 1.2) | 1.14 (±.08 plus-or-minus.08 \pm.08 ±.08) | 4.66 (±.69 plus-or-minus.69 \pm.69 ±.69) | 1.08 (±.02 plus-or-minus.02 \pm.02 ±.02) | | 98.6 (± 0.4 plus-or-minus 0.4 \pm 0.4 ± 0.4) | 5.71 (±.24 plus-or-minus.24 \pm.24 ±.24) | 5.31 (±.45 plus-or-minus.45 \pm.45 ±.45) | 0.83 (±.01 plus-or-minus.01 \pm.01 ±.01) | |
| HoST -History10 | | 98.8 (± 0.8 plus-or-minus 0.8 \pm 0.8 ± 0.8) | 1.62 (±.08 plus-or-minus.08 \pm.08 ±.08) | 3.02 (±.20 plus-or-minus.20 \pm.20 ±.20) | 1.60 (±.02 plus-or-minus.02 \pm.02 ±.02) | | 99.2 (± 0.8 plus-or-minus 0.8 \pm 0.8 ± 0.8) | 0.78 (±.05 plus-or-minus.05 \pm.05 ±.05) | 3.55 (±.40 plus-or-minus.40 \pm.40 ±.40) | 0.71 (±.01 plus-or-minus.01 \pm.01 ±.01) | | 88.2 (± 2.6 plus-or-minus 2.6 \pm 2.6 ± 2.6) | 1.24 (±.06 plus-or-minus.06 \pm.06 ±.06) | 4.61 (±.72 plus-or-minus.72 \pm.72 ±.72) | 1.46 (±.05 plus-or-minus.05 \pm.05 ±.05) | | 98.6 (± 0.8 plus-or-minus 0.8 \pm 0.8 ± 0.8) | 3.93 (±.26 plus-or-minus.26 \pm.26 ±.26) | 5.41 (±.49 plus-or-minus.49 \pm.49 ±.49) | 0.91 (±.01 plus-or-minus.01 \pm.01 ±.01) | |

#### Table 4: TABLE IV: Main results for real robot experiments. We report the success rate and motion smoothness to quantitatively co

| Method | | Ground | | Platform | | Wall | | Slope | | Overall | | | | | | |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| | E succ ↑ ↑ E succ absent E_{ $\mathrm{succ}}\uparrow ↑$ | E smth ↓ ↓ E smth absent E_{ $\mathrm{smth}}\downarrow ↓$ | | E succ E succ E_{ $\mathrm{succ}} ↑ ↑ \uparrow ↑$ | E smth E smth E_{ $\mathrm{smth}} ↓ ↓ \downarrow ↓$ | | E succ E succ E_{ $\mathrm{succ}} ↑ ↑ \uparrow ↑$ | E smth E smth E_{ $\mathrm{smth}} ↓ ↓ \downarrow ↓$ | | E succ E succ E_{ $\mathrm{succ}} ↑ ↑ \uparrow ↑$ | E smth E smth E_{ $\mathrm{smth}} ↓ ↓ \downarrow ↓$ | | E succ E succ E_{ $\mathrm{succ}} ↑ ↑ \uparrow ↑$ | E smth E smth E_{ $\mathrm{smth}} ↓ ↓ \downarrow ↓$ | | |
| HoST -w/o-L2C2 | | 5 / 5 5 5  $\frac{5}{5} / 5 5$ | 2.09 | | 2 / 5 2 5  $\frac{2}{5} / 2 5$ | 7.85 | | 4 / 5 4 5  $\frac{4}{5} / 4 5$ | 13.36 | | 0 / 5 0 5  $\frac{0}{5} / 0 5$ | 2.89 | | 11 / 20 11 20  $\frac{11}{20} / 11 20$ | 6.54 | |
| HoST (ours) | | 5 / 5 5 5  $\frac{5}{5} / 5 5$ | 1.83 | | 5 / 5 5 5  $\frac{5}{5} / 5 5$ | 5.06 | | 5 / 5 5 5  $\frac{5}{5} / 5 5$ | 7.22 | | 5 / 5 5 5  $\frac{5}{5} / 5 5$ | 1.94 | | 20 / 20 20 20  $\frac{20}{20} / 20 20$ | 4.01 | |

**说明**: TABLE IV: Main results for real robot experiments. We report the success rate and motion smoothness to quantitatively compare our methods with the baseline. The results demonstrate the superiority of our method and the importance of adding smooth regularization into our method.

#### Table 5: TABLE V: Robustness to payload and random torque dropout.

| Metric | | Payload Mass | | Torque Dropout Ratio | | | | | | | | |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| | 4kg | 6kg | 8kg | 10kg | 12kg | | 0.05 | 0.1 | 0.15 | 0.2 | | |
| E smth ↓ ↓ E smth absent E_{ $\mathrm{smth}}\downarrow ↓$ | | 1.75 | 1.92 | 1.86 | 1.82 | 1.85 | | 2.00 | 2.16 | 2.61 | / | |
| E succ ↑ ↑ E succ absent E_{ $\mathrm{succ}}\uparrow ↑$ | | 3 / 3 3 3  $\frac{3}{3} / 3 3$ | 3 / 3 3 3  $\frac{3}{3} / 3 3$ | 3 / 3 3 3  $\frac{3}{3} / 3 3$ | 3 / 3 3 3  $\frac{3}{3} / 3 3$ | 2 / 3 2 3  $\frac{2}{3} / 2 3$ | | 3 / 3 3 3  $\frac{3}{3} / 3 3$ | 3 / 3 3 3  $\frac{3}{3} / 3 3$ | 3 / 3 3 3  $\frac{3}{3} / 3 3$ | 0 / 3 0 3  $\frac{0}{3} / 0 3$ | |

**说明**: TABLE V: Robustness to payload and random torque dropout.

#### Table 6: TABLE VI: Reward functions and groups used for learning standing-up control. Reward functions within the same group are

| Term | Expression | Weight | Description |
| --- | --- | --- | --- |
| (a) Task Reward | r task r task r^{ $\mathrm{task}}$ | w task = 2.5 w task 2.5 w^{ $\mathrm{task}}=2.5 = 2.5$ | It specifies the high-level task objectives. |
| Head height | f tol (h head, [1, inf ], 1, 0.1) f tol h head 1 inf 1 0.1 f_{ $\mathrm{tol}}\left(h_{\mathrm{head}},[1,\mathrm{inf}],1,0.1\right)$ | 1 | The head of robot head h head h head h_{ $\mathrm{head}} in the world frame.$ |
| Base orientation | $\mathrm{tol}}\left(-{\theta}_{\mathrm{base}}^{\mathrm{z}},[0.99,\mathrm{inf}],1,0.05\right)$ | 1 | The orientation of the robot base represented by projected gravity vector. |
| (b) Style Reward | r style r style r^{ $\mathrm{style}}$ | w style = 1 w style 1 w^{ $\mathrm{style}}=1 = 1$ | It specifies the style of standing-up motion. |
| Waist yaw deviation | 1 (\| q waist \| > 1.4) 1 q waist 1.4  $\mathbf{1}(\lVert q_{\mathrm{waist}} \rVert>1.4) (\lVert \rVert > 1.4)$ | - 10 10 -10 - 10 | It penalizes the large joint angle of the waist yaw. |
| Hip roll/yaw deviation | 1 (max (\| q hip l, r \|) > 1.4) \| 1 (min (\| q hip l, r \|) > 0.9)  $\mathbf{1}(\max(\lVert \boldsymbol{q}_{\mathrm{hip}}^{\mathrm{l,r}} \rVert)>1.4)\;\lVert \;\mathbf{1}(\min(\rVert\boldsymbol{q}_{\mathrm{hip}}^{\mathrm{l,r}}\lVert)>0.9) ((\rVert, \lVert) > 1.4) \rVert ((\lVert, \rVert) > 0.9)$ | - 10 10 -10 - 10 / - 10 10 -10 - 10 | It penalizes the large joint angle of hip roll/yaw joints. |
| Knee deviation | 1 (max  (q knee l, r) > 2.85) \| 1 (min  (q knee l, r) 2.85)\;\|\; $\mathbf{1}(\min(\boldsymbol{q}_{\mathrm{knee}}^{\mathrm{l,r}}) 2.85) \lVert$ | - 0.25 (G) - 10 (P S W) 0.25 G 10 P S W  $\frac{-0.25(G)}{-10(PSW)} divide - 0.25 () - 10 ()$ | It penalizes the large joint angle of knee joints. |
| Shoulder roll deviation | 1 (q shoulder l 0.02) conditional 1 q shoulder l 0.02 1 q shoulder r 0.02  $\mathbf{1}(q_{\mathrm{shoulder}}^{l} 0.02) (0.02)$ | - 2.5 2.5 -2.5 - 2.5 | It penalizes the large joint angle of shoulder roll joint. |
| Foot displacement | $\exp\left(-2\times\lVert \boldsymbol{q}_{\mathrm{base}}^{\mathrm{xy}}-\boldsymbol{q}_{\mathrm{foot}}^{\mathrm{xy}} \rVert^{2}.\mathrm{clip}(0.3,\mathrm{inf})\right)\times\mathbf{1}(h_{\mathrm{base}}>H_{\mathrm{stage2}})$ | 2.5 / 2.5 2.5 2.5 2.5/2.5 2.5 / 2.5 | It encourages robot CoM locates in support polygon, inspired by [11 ]. |
| Ankle parallel | (var (q left ankle z) + var (q right ankle z)) / 2 < 0.05 var q left ankle z var q right ankle z 2 0.05 ( $\mathrm{var}(\boldsymbol{q}_{\mathrm{left\;ankle}}^{z})+\mathrm{var}(\boldsymbol{q}_{\mathrm{right\;ankle}}^{z}))/2<0.05 (() + ()) / 2 < 0.05$ | 20 20 20 20 | It encourages the ankles to be parallel to the ground via ankle keypoints. |
| Foot distance | $$ | - 10 10 -10 - 10 | It penalizes a far distance between feet. |
| Feet stumble | 1 (∃ i, \| F i xy \| > 3 \| F i z \|) 1 i F i xy 3 F i z  $\mathbf{1}(\exists i,\lVert \mathbf{F}_{i}^{\mathrm{xy}} \rVert>3\lVert F_{i}^{\mathrm{z}} \rVert) (∃, \lVert \rVert > 3 \lVert \rVert)$ | 0 (G) - 25 (P S W) 0 G 25 P S W  $\frac{0(G)}{-25(PSW)} divide 0 () - 25 ()$ | It penalizes a horizontal contact force with the environment. |
| Shank orientation | $\mathrm{tol}}(\mathrm{mean}(\mathrm{\boldsymbol{\theta}_{\mathrm{shank}}^{\mathrm{l,r}}[2]}),[0.8,\mathrm{inf}],1,0.1)\times\mathbf{1}(h_{\mathrm{base}}>H_{\mathrm{stage1}})$ | 10 10 10 10 | It encourages the left/right shank to be perpendicular to the ground. |
| Base angular velocity | $\mathrm{exp}(-2\times\lVert \boldsymbol{\omega}^{\mathrm{xy}}_{\mathrm{base}} \rVert^{2})\times\mathbf{1}(h_{\mathrm{base}}>H_{\mathrm{stage1}})$ | 1 1 1 1 | It encourages low angular velocity of the during rising up. |
| (c) Regularization Reward | r regu r regu r^{ $\mathrm{regu}}$ | w regu = 0.1 w regu 0.1 w^{ $\mathrm{regu}}=0.1 = 0.1$ | It specifies the regulariztaion on standing-up motion. |
| Joint acceleration |  p  ̈  2 norm  ̈ p 2 \\|\ddot{p}\\|^{2}   2 | - 2.5 e - 7 2.5 e 7 -2.5e^{-7} - 2.5 - 7 | It penalizes the high joint accelrations. |
| Action rate |  a t - a t - 1  2 norm a t a t 1 2 \\|a_{t}-{a}_{t-1}\\|^{2}  - - 1  2 | - 1 e - 2 1 e 2 -1e^{-2} - 1 - 2 | It penalizes the high changing speed of action. |
| Smoothness |  a t - 2 a t - 1 + a t - 2  2 norm a t 2 a t 1 a t 2 2 \\|{a}_{t}-2{a}_{t-1}+{a}_{t-2}\\|^{2}  - 2 - 1 + - 2  2 | - 1 e - 2 1 e 2 -1e^{-2} - 1 - 2 | It penalizes the discrepancy between consecutive actions. |
| Torques | $$ | - 2.5 e - 6 2.5 e 6 -2.5e^{-6} - 2.5 - 6 | It penalizes the high joint torques. |
| Joint power | $$ | - 2.5 e - 5 2.5 e 5 -2.5e^{-5} - 2.5 - 5 | It penalizes the high joint power |
| Joint velocity | $$ | - 1 e - 4 1 e 4 -1e^{-4} - 1 - 4 | It penalizes the high joint velocity. |
| Joint tracking error | $$ | - 2.5 e - 1 2.5 e 1 -2.5e^{-1} - 2.5 - 1 | It penalizes the error between PD target (Eq. 1) and actual joint position. |
| Joint position limits | ∑ i [(p i - p i Lower). clip (- inf, 0) + (p i - p i Higher). clip (0, inf) ]  $\sum_{i}[(p_{i}-p_{i}^{\mathrm{Lower}}).\mathrm{clip}(-\mathrm{inf},0)+(p_{i}-p_{i}^{\mathrm{Higher}}).\mathrm{clip}(0,\mathrm{inf})] ∑ [(-). (-, 0) + (-). (0,) ]$ | - 1 e 2 1 e 2 -1e^{2} - 1 2 | It penalizes the joint position that beyond limits. |
| Joint velocity limits | ∑ i [(\| p  ̇ i \| - p  ̇ i Limit). clip (0, inf) ]  $\sum_{i}[(\lVert \dot{p}_{i} \rVert-\dot{p}_{i}^{\mathrm{Limit}}).\mathrm{clip(0,\mathrm{inf})}] ∑ [(\lVert \rVert -). (0,) ]$ | - 1 1 -1 - 1 | It penalizes the joint velocity that beyond limits. |
| (d) Post-task Reward | r post r post r^{ $\mathrm{post}}$ | w post = 1 w post 1 w^{ $\mathrm{post}}=1 = 1$ | It specifies the desired behaviors after a successful standing up. |
| Base angular velocity | $\mathrm{exp}(-2\times\lVert \boldsymbol{\omega}^{\mathrm{xy}}_{\mathrm{base}} \rVert^{2})\times\mathbf{1}(h_{\mathrm{base}}>H_{\mathrm{stage2}})$ | 10 10 10 10 | It encourages low angular velocity of robot base after standing up. |
| Base linear velocity | $\mathrm{exp}(-5\times\lVert \boldsymbol{v}^{\mathrm{xy}}_{\mathrm{base}} \rVert^{2})\times\mathbf{1}(h_{\mathrm{base}}>H_{\mathrm{stage2}})$ | 10 10 10 10 | It encourages low linear velocity of robot base after standing up. |
| Base orientation | $\exp(-5\times\lVert \boldsymbol{\theta}_{\mathrm{base}}^{\mathrm{xy}} \rVert^{2}\times\mathbf{1}(h_{\mathrm{base}}>H_{\mathrm{stage2}})$ | 10 | It encourages the robot base to be perpendicular to the ground. |
| Base height | $\exp(-20\times\lVert {h}_{\mathrm{base}}-{h}_{\mathrm{base}}^{\mathrm{target}} \rVert^{2}\times\mathbf{1}(h_{\mathrm{base}}>H_{\mathrm{stage2}})$ | 10 | It encourages the robot base to reach a target height. |
| Upper-body posture | $\exp(-0.1\times\lVert p_{\mathrm{upper}}-p_{\mathrm{upper}}^{\mathrm{target}} \rVert^{2})\times\mathbf{1}(h_{\mathrm{base}}>H_{\mathrm{stage2}})$ | 10 | It encourages the robot to track a target upper body postures. |
| Feet parallel | $\exp(-20\times\lVert h_{\mathrm{feet}}^{l}-h_{\mathrm{feet}}^{r} \rVert.\mathrm{clip}(0.02,\mathrm{inf}))\times\mathbf{1}(h_{\mathrm{base}}>H_{\mathrm{stage2}}) (- 20 × \lVert - \rVert.$ | 2.5 | In encourages the feet to be parallel to each other. |

**说明**: 该表格来自论文中的公式、奖励项或实验设置；为避免 PDF 抽取噪声，网页中仅保留可渲染的核心符号和数值。

#### Table 7: Table 7

| Joint | G1 | | H1 | | H1-2 | | | |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Kp | Kd | | Kp | Kd | | Kp | Kd | |
| Hip | 150 | 4 | | 350 | 4 | | 350 | 4 |
| Knee | 200 | 6 | | 350 | 4 | | 350 | 4 |
| Ankle | 40 | 2 | | 120 | 2 | | 120 | 2 |
| Shoulder | 100 | 4 | | 350 | 4 | | 350 | 4 |
| Elbow | 100 | 4 | | 350 | 4 | | 350 | 4 |
| Waist | 100 | 4 | | 200 | 4 | | 200 | 4 |

**说明**: Table 7

#### Table 8: Table 8

| Observation | Ang. Velocity | Pitch & Roll | DoF Position | DoF Velocity | Action Rescaler |
| --- | --- | --- | --- | --- | --- |
| Noise Scale | U (- 0.2, 0.2) U 0.2 0.2  $\mathcal{U}(-0.2,0.2)$ | U (- 0.05, 0.05) U 0.05 0.05  $\mathcal{U}(-0.05,0.05)$ | U (- 0.01, 0.01) U 0.01 0.01  $\mathcal{U}(-0.01,0.01)$ | U (- 1.5, 1.5) U 1.5 1.5  $\mathcal{U}(-1.5,1.5)$ | U (- 0.025, 0.025) U 0.025 0.025  $\mathcal{U}(-0.025,0.025)$ |

**说明**: Table 8
## 实验解读

- 评价重点:围绕 adaptive-control、接触推理、人形机器人、足式运动、移动操作,优先看真实机器人结果、扰动恢复、接触稳定性、控制误差、成功率、实时性和 sim-to-real 差距。
- System0 视角:实验结论不只看最终成功率,还要看闭环过程中是否能处理状态估计误差、执行延迟、碰撞/接触突变和外部扰动。
- 与上层系统的关系:如果方法能在快闭环内稳定工作,它可以作为 System1 技能或 System2 规划的底层执行保障;如果只在离线或慢速设定下成立,则还不能直接作为身体级快闭环模块。
## 批判性思考

### 优点

- 论文把 adaptive-control、接触推理、人形机器人、足式运动、移动操作 放到机器人身体级闭环里讨论,和只做高层语义推理的工作相比,更接近真实机器人部署时会遇到的状态、动力学和安全约束。
- 若论文同时给出公式、系统结构图、硬件实验或消融结果,这些证据可以帮助判断方法是否真的能支撑毫秒到百毫秒级反应。

### 局限

- 需要重点核查控制频率、推理延迟、传感延迟、失败恢复、硬件磨损、未见场景泛化和安全边界;这些因素决定它能否成为可靠的 System0 模块。
- 对学习型方法,还要区分训练环境中的鲁棒性和真实部署中的鲁棒性,尤其是接触丰富任务、动态运动任务和人形机器人任务。
## 论文定位

- 研究对象:Learning Humanoid Standing-up Control across Diverse Postures。
- 关键词:adaptive-control、接触推理、人形机器人、足式运动、移动操作、recovery、强化学习、机器人操作、鲁棒控制、仿真到真实迁移。
- System0 关联:关注机器人身体级快反应链路中“状态/接触/局部感知 -> 约束/优化/策略 -> 动作”的短闭环,不把问题停留在语言规划或任务分解层。

## 速查卡片

> [!summary] Humanoid Standing up
> - **论文**: https://www.roboticsproceedings.org/rss21/p064.pdf
> - **arXiv**: http://arxiv.org/abs/2502.08378v2
> - **arXiv HTML**: https://arxiv.org/html/2502.08378v2
