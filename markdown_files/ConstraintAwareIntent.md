---
title: "Constraint-Aware Intent Estimation for Dynamic Human-Robot Object Co-Manipulation"
method_name: "Constraint Aware Intent"
authors: ["Yifei Simon Shao"]
year: 2024
venue: "RSS"
tags: ["robot-manipulation", "robust-control", "real-time-control", "safe-control", "adaptive-control"]
image_source: "online"
arxiv_html: "https://arxiv.org/html/2409.00215v1"
---
# Constraint Aware Intent
## 一句话总结

> Constraint-Aware Intent Estimation for Dynamic Human-Robot Object Co-Manipulation 主要落在 [[adaptive-control]]、[[compliance-control]]、[[impedance-control]]、[[实时控制]] 这一类问题上;从 System0 视角看,它的价值在于把身体状态、接触变化、局部感知或动力学约束接到机器人低层闭环里,让动作生成更稳定、安全、实时。
## 核心贡献

1. 围绕 **Constraint-Aware Intent Estimation for Dynamic Human-Robot Object Co-Manipulation** 建立了一个与 adaptive-control、compliance-control、impedance-control、实时控制、机器人操作、safe-control 相关的机器人问题设定,重点不是高层语义规划,而是身体级闭环中的状态变化、接触变化或控制约束。
2. 方法层面的关键价值在于把感知、动力学、约束优化、策略学习或硬件反馈组织成可执行的低层动作生成机制,适合放进 System0 的快反应模块中理解。
3. 对系统集成而言,这类工作补上了 System1 技能选择与电机级控制之间的一层:它负责把上层目标变成动态可行、可恢复、可安全过滤的动作。
4. 若论文包含真实机器人、接触任务或仿真到真实迁移,核心应看它是否报告了闭环稳定性、扰动恢复、硬件延迟、失败模式和跨场景泛化。
## 问题背景

这篇论文可以放在 System0 的“具身快系统”背景下阅读:机器人需要在很短时间内根据本体状态、接触、力/触觉、局部视觉或动力学预测做出动作,而不是等待慢速规划器重新推理整个任务。adaptive-control、compliance-control、impedance-control、实时控制、机器人操作、safe-control 相关问题的难点通常不在“知道要做什么”,而在动作执行过程中状态会快速偏离、接触会突然改变、约束会实时收紧,低层系统必须及时修正。

### System0 关联

精读时优先看三件事:第一,论文使用哪些快速反馈量,例如关节状态、接触状态、力觉、触觉、深度或局部视觉;第二,反馈如何进入 MPC、WBC、策略网络、安全过滤器或恢复控制器;第三,输出是否直接影响机器人下一步动作,而不是只服务离线分析或高层任务规划。
## 方法详解

### System0 视角解析

从 System0 角度看,这篇论文的方法部分应当围绕 adaptive-control、compliance-control、impedance-control、实时控制、机器人操作、safe-control 来读。核心问题是:系统如何从快速变化的机器人状态中构造控制输入,如何处理接触、碰撞、约束、模型误差或执行延迟,以及如何把这些信息变成下一时刻的动作。

如果方法里有优化问题,重点看目标函数、约束和求解频率;如果方法里有学习策略,重点看观测空间、动作空间、训练扰动和部署时的闭环频率;如果方法里有硬件或传感器,重点看传感链路是否真的进入控制回路。这样读可以把论文和 System2 语言规划、System1 技能编排区分开来。

### 关键公式与机制

#### 公式 1: [[动力学或策略机制]]

$$\mathbf{q}=\begin{bmatrix}s\\ \boldsymbol{u}\\ \end{bmatrix}=\begin{bmatrix}\text{scalar}(\mathbf{q})\\ \text{vec}(\mathbf{q})\\ \end{bmatrix}=\begin{bmatrix}\cos(\theta/2)\\ \sin(\theta/2)\boldsymbol{n}\\ \end{bmatrix}$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 2: [[动力学或策略机制]]

$$\mathbf{M}_{\theta}(\boldsymbol{\theta})\ddot{\boldsymbol{\theta}}+\mathbf{C}_{\theta}(\boldsymbol{\theta},\dot{\boldsymbol{\theta}})\dot{\boldsymbol{\theta}}+\mathbf{g}_{\theta}(\boldsymbol{\theta})=\boldsymbol{\tau}_{\theta}{-}\mathbf{J}(\boldsymbol{\theta})^{\top}{\mathbf{u_{\text{ext}}}},$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 3: [[动力学或策略机制]]

$$=-\left[\begin{array}{cc}\mathbf{D}_{p}(c_{p}(t))&0\\ 0&\mathbf{D}_{o}(c_{o}(t))\end{array}\right]\left[\begin{array}{l}\dot{\mathbf{p}}-\widehat{\dot{\mathbf{p}}}\\ \boldsymbol{\omega}-\widehat{\boldsymbol{\omega}}\end{array}\right],$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 4: [[动力学或策略机制]]

$$\frac{\partial(c_{p}(t))}{\partial\dot{\mathbf{p}}}=\int_{-T}^{0}-\frac{\partial(e_{p}(s))}{\partial\dot{\mathbf{p}}}\mathrm{d}{s}=\int_{-T}^{0}\frac{\widehat{\dot{\mathbf{p}}}(s)-\dot{\mathbf{p}}(s)}{|| \widehat{ $\dot{\mathbf{p}}}(s)-\dot{\mathbf{p}}(s)$ ||_{2}}\mathrm{d}{s}$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 5: [[动力学或策略机制]]

$$*=\begin{bmatrix}d_{p}^{1}\frac{\partial(c_{p}(t))}{\partial\dot{\mathbf{p}}_{1}}\dot{\mathbf{p}}_{1}&d_{p}^{1}\frac{\partial(c_{p}(t))}{\partial\dot{\mathbf{p}}_{2}}\dot{\mathbf{p}}_{1}&d_{p}^{1}\frac{\partial(c_{p}(t))}{\partial\dot{\mathbf{p}}_{3}}\dot{\mathbf{p}}_{1}\\ d_{p}^{2}\frac{\partial(c_{p}(t))}{\partial\dot{\mathbf{p}}_{1}}\dot{\mathbf{p}}_{2}&d_{p}^{2}\frac{\partial(c_{p}(t))}{\partial\dot{\mathbf{p}}_{2}}\dot{\mathbf{p}}_{2}&d_{p}^{2}\frac{\partial(c_{p}(t))}{\partial\dot{\mathbf{p}}_{3}}\dot{\mathbf{p}}_{2}\\ d_{p}^{3}\frac{\partial(c_{p}(t))}{\partial\dot{\mathbf{p}}_{1}}\dot{\mathbf{p}}_{3}&d_{p}^{3}\frac{\partial(c_{p}(t))}{\partial\dot{\mathbf{p}}_{2}}\dot{\mathbf{p}}_{3}&d_{p}^{3}\frac{\partial(c_{p}(t))}{\partial\dot{\mathbf{p}}_{3}}\dot{\mathbf{p}}_{3}\\ \end{bmatrix}.$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 6: [[动力学或策略机制]]

$$=\begin{cases}\eta_{9}\left(\frac{1}{\Delta\ {\theta}}-\frac{1}{\ {\delta}}\right)\frac{1}{{\Delta\ {\theta}}^{2}},&\>\,\,\,\text{if}\Delta\ {\theta}\leq\ {\delta}\\ 0,&\>\,\,\,\text{if}\Delta\ {\theta}>\ {\delta}\end{cases},$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 7: [[动力学或策略机制]]

$$=\begin{cases}-\eta_{9}\left(\frac{1}{\Delta\ {\theta}}-\frac{1}{\ {\delta}}\right)\frac{1}{{\Delta\ {\theta}}^{2}},&\text{if}\Delta\ {\theta}\leq\ {\delta}\\ 0,&\text{if}\Delta\ {\theta}>\ {\delta}\end{cases},$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 8: [[动力学或策略机制]]

$$\dot{\mathbf{q}}=\frac{1}{2}\boldsymbol{\tilde{\omega}}\otimes\mathbf{q}\rightarrow\dot{\mathbf{q}}=\begin{bmatrix}\dot{s}\\ \boldsymbol{\dot{u}}\end{bmatrix}=\begin{bmatrix}-\frac{1}{2}\boldsymbol{u}^{T}\boldsymbol{\omega}\\ \frac{1}{2}(s\mathbf{I}-\mathbf{S}(\boldsymbol{u}))\boldsymbol{\omega}\end{bmatrix}$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 9: [[动力学或策略机制]]

$$\mathbf{M}_{r}(\boldsymbol{\theta})\ddot{\mathbf{x}}_{r}+\mathbf{C}_{r}(\boldsymbol{\theta},\dot{\boldsymbol{\theta}})\dot{\mathbf{x}}_{r}+\mathbf{g}_{r}(\boldsymbol{\theta})={\mathbf{u}_{r}}-{\mathbf{u_{\text{ext}}}},$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 10: [[动力学或策略机制]]

$$\mathbf{D}_{p}^{h}(c_{p}(t))=\frac{\partial\mathbf{u}_{h}^{\prime}}{\partial\dot{\mathbf{p}}}=\mathbf{C}+\frac{\partial\left(\mathbf{D}_{p}(c_{p}(t))\dot{\mathbf{p}}\right)}{\partial\dot{\mathbf{p}}},$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。
## 关键图表

### Figure 1: Our method uses particle filters to predict full 6 DoF intent and a variable impedanc

![Figure 1](https://arxiv.org/html/2409.00215v1/extracted/5824316/fig/fig1_new.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Our method uses particle filters to predict full 6 DoF intent and a variable impedanc”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 2: Comparison of (left) average force (N) and (right) average torque (Nm) all trial

![Figure 2](https://arxiv.org/html/2409.00215v1/x4.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Comparison of (left) average force (N) and (right) average torque (Nm) all trial”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 3: Experiment data of a single trial. Since the task is in 3D space, the left column sho

![Figure 3](https://arxiv.org/html/2409.00215v1/extracted/5824316/fig/subj_3__trial_14_data.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Experiment data of a single trial. Since the task is in 3D space, the left column sho”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### 论文表格

#### Table 1: TABLE I: Results for all subjects: Mean task completion time, linear impulse, angular impulse, average force (N) and av

| | | Method | | | |
| --- | --- | --- | --- | --- | --- |
| Metric | Subj. | Proposed | Admittance | Adaptation | Adaptation |
| F/T robot | F/T robot | F/T human | | | |
| Time | 1 | 5.6 | 6.7 | 5.7 | 5.8 |
| 2 | 5.2 | 6.2 | 4.8 | 5.5 | |
| (s) | 3 | 6.0 | 7.2 | 6.7 | 7.0 |
| Lin. Impulse | 1 | 38.9 | 58.0 | 40.9 | 26.4 |
| 2 | 44.3 | 62.6 | 42.9 | 39.6 | |
| (Ns) | 3 | 65.7 | 76.3 | 66.8 | 42.3 |
| Ang. Impulse | 1 | 7.7 | 10.1 | 6.4 | 3.9 |
| 2 | 8.3 | 13.0 | 7.6 | 7.7 | |
| (Nms) | 3 | 10.7 | 11.8 | 11.6 | 9.5 |
| Avg. Force | 1 | 6.5 | 9.1 | 7.3 | 5.0 |
| 2 | 6.5 | 9.1 | 7.3 | 5.0 | |
| (N) | 3 | 7.4 | 8.4 | 8.5 | 3.8 |
| Avg. Torque | 1 | 0.40 | 0.58 | 0.38 | 0.22 |
| 2 | 0.57 | 0.95 | 0.66 | 0.60 | |
| (Nm) | 3 | 0.66 | 0.64 | 0.62 | 0.46 |

**说明**: TABLE I: Results for all subjects: Mean task completion time, linear impulse, angular impulse, average force (N) and average torque (Nm) through all trials

#### Table 2: TABLE II: Proposed Method Hyperparameters. Fixed values are written in low column

| | Low | High |
| --- | --- | --- |
| Damping Gain Cartesian D p D p  $\mathbf{D}_{p}$ | 1 | 85 |
| Damping Gain Rotation D o D o  $\mathbf{D}_{o}$ | 1 | 13 |
| Dynamics Cartesian A p A p  $\mathbf{A}_{p}$ | -0.6 | -0.4 |
| Dynamics Rotation A o A o  $\mathbf{A}_{o}$ | -0.9 | -0.6 |
| Relative Weights in Particle Filter  1 ∼  4 similar-to  1  4 \eta_{1}\sim\eta_{4} 1 ∼ 4 | 0.5 | |
| Ellipsoid Decay Weight $\boldsymbol{\eta}}_{5} 5$ | 1.5 | |
| Noise Dynamics Cartesian $\boldsymbol{\eta}}_{6} 6$ | 3e-4 | 4e-3 |
| Noise Dynamics Rotation $\boldsymbol{\eta}}_{7} 7$ | 2e-4 | 8.5e-3 |
| Noise Goal Rotation $\boldsymbol{\eta}}_{8} 8$ | 2e-4 | 8.5e-3 |
| Joint Limit Resistance  9  9 \eta_{9} 9 | 0.1 | |
| Ascent Rate Cartesian d p d p d_{p} | 0.41 | |
| Ascent Rate Rotation d o d o d_{o} | 0.49 | |

**说明**: TABLE II: Proposed Method Hyperparameters. Fixed values are written in low column

#### Table 3: TABLE III: Baseline Parameters

| Admittance | Value | Task Adaptation | Value |
| --- | --- | --- | --- |
| Mass Cartesian | 10.0 | Tank Size | 2 |
| Mass Rotation | 2.0 | Trigger | 1 |
| Damping Cartesian | 30.0 | Force Deadzone | 15 |
| Damping Rotation | 5.0 | Torque Deadzone | 0.8 |
| Max Velocity | 0.8 | Max Velocity | 1.3 |
| Max Acceleration | 1.0 | Max Acceleration | 2.0 |

**说明**: TABLE III: Baseline Parameters
## 实验解读

- 评价重点:围绕 adaptive-control、compliance-control、impedance-control、实时控制、机器人操作,优先看真实机器人结果、扰动恢复、接触稳定性、控制误差、成功率、实时性和 sim-to-real 差距。
- System0 视角:实验结论不只看最终成功率,还要看闭环过程中是否能处理状态估计误差、执行延迟、碰撞/接触突变和外部扰动。
- 与上层系统的关系:如果方法能在快闭环内稳定工作,它可以作为 System1 技能或 System2 规划的底层执行保障;如果只在离线或慢速设定下成立,则还不能直接作为身体级快闭环模块。
## 批判性思考

### 优点

- 论文把 adaptive-control、compliance-control、impedance-control、实时控制、机器人操作 放到机器人身体级闭环里讨论,和只做高层语义推理的工作相比,更接近真实机器人部署时会遇到的状态、动力学和安全约束。
- 若论文同时给出公式、系统结构图、硬件实验或消融结果,这些证据可以帮助判断方法是否真的能支撑毫秒到百毫秒级反应。

### 局限

- 需要重点核查控制频率、推理延迟、传感延迟、失败恢复、硬件磨损、未见场景泛化和安全边界;这些因素决定它能否成为可靠的 System0 模块。
- 对学习型方法,还要区分训练环境中的鲁棒性和真实部署中的鲁棒性,尤其是接触丰富任务、动态运动任务和人形机器人任务。
## 论文定位

- 研究对象:Constraint-Aware Intent Estimation for Dynamic Human-Robot Object Co-Manipulation。
- 关键词:adaptive-control、compliance-control、impedance-control、实时控制、机器人操作、safe-control。
- System0 关联:关注机器人身体级快反应链路中“状态/接触/局部感知 -> 约束/优化/策略 -> 动作”的短闭环,不把问题停留在语言规划或任务分解层。

## 速查卡片

> [!summary] Constraint Aware Intent
> - **论文**: https://www.roboticsproceedings.org/rss20/p028.pdf
> - **arXiv**: http://arxiv.org/abs/2409.00215v1
> - **arXiv HTML**: https://arxiv.org/html/2409.00215v1
