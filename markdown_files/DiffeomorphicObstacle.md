---
title: "Diffeomorphic Obstacle Avoidance for Contractive Dynamical Systems via Implicit Representations"
method_name: "Diffeomorphic Obstacle Avoidance"
authors: ["Ken-Joel Simmoteit"]
year: 2025
venue: "RSS"
tags: ["robust-control", "safe-control", "adaptive-control", "imitation-learning", "robot-generalization", "collision-avoidance", "whole-body-control"]
image_source: "online"
arxiv_html: "https://arxiv.org/html/2504.18860v1"
---
# Diffeomorphic Obstacle Avoidance
## 一句话总结

> Diffeomorphic Obstacle Avoidance for Contractive Dynamical Systems via Implicit Representations 主要落在 [[adaptive-control]]、[[碰撞避免]]、[[flow-matching]]、[[模仿学习]] 这一类问题上;从 System0 视角看,它的价值在于把身体状态、接触变化、局部感知或动力学约束接到机器人低层闭环里,让动作生成更稳定、安全、实时。
## 核心贡献

1. 围绕 **Diffeomorphic Obstacle Avoidance for Contractive Dynamical Systems via Implicit Representations** 建立了一个与 adaptive-control、碰撞避免、flow-matching、模仿学习、鲁棒控制、safe-control 相关的机器人问题设定,重点不是高层语义规划,而是身体级闭环中的状态变化、接触变化或控制约束。
2. 方法层面的关键价值在于把感知、动力学、约束优化、策略学习或硬件反馈组织成可执行的低层动作生成机制,适合放进 System0 的快反应模块中理解。
3. 对系统集成而言,这类工作补上了 System1 技能选择与电机级控制之间的一层:它负责把上层目标变成动态可行、可恢复、可安全过滤的动作。
4. 若论文包含真实机器人、接触任务或仿真到真实迁移,核心应看它是否报告了闭环稳定性、扰动恢复、硬件延迟、失败模式和跨场景泛化。
## 问题背景

这篇论文可以放在 System0 的“具身快系统”背景下阅读:机器人需要在很短时间内根据本体状态、接触、力/触觉、局部视觉或动力学预测做出动作,而不是等待慢速规划器重新推理整个任务。adaptive-control、碰撞避免、flow-matching、模仿学习、鲁棒控制、safe-control、全身控制 相关问题的难点通常不在“知道要做什么”,而在动作执行过程中状态会快速偏离、接触会突然改变、约束会实时收紧,低层系统必须及时修正。

### System0 关联

精读时优先看三件事:第一,论文使用哪些快速反馈量,例如关节状态、接触状态、力觉、触觉、深度或局部视觉;第二,反馈如何进入 MPC、WBC、策略网络、安全过滤器或恢复控制器;第三,输出是否直接影响机器人下一步动作,而不是只服务离线分析或高层任务规划。
## 方法详解

### System0 视角解析

从 System0 角度看,这篇论文的方法部分应当围绕 adaptive-control、碰撞避免、flow-matching、模仿学习、鲁棒控制、safe-control、全身控制 来读。核心问题是:系统如何从快速变化的机器人状态中构造控制输入,如何处理接触、碰撞、约束、模型误差或执行延迟,以及如何把这些信息变成下一时刻的动作。

如果方法里有优化问题,重点看目标函数、约束和求解频率;如果方法里有学习策略,重点看观测空间、动作空间、训练扰动和部署时的闭环频率;如果方法里有硬件或传感器,重点看传感链路是否真的进入控制回路。这样读可以把论文和 System2 语言规划、System1 技能编排区分开来。

### 关键公式与机制

#### 公式 1: [[优化目标/约束]]

$$=\frac{1}{2}\left(1+\frac{\dot{\mathbf{q}}_{\text{NCDS}}\cdot\nabla_{\mathbf{q}}\Gamma_{\mathcal{R}}(\mathbf{x},\mathbf{q})}{| $\dot{\mathbf{q}}_{\text{NCDS}}$ |\ | \nabla_{ $\mathbf{q}}\Gamma_{\mathcal{R}}(\mathbf{x},\mathbf{q})$ |}\right),$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 2: [[优化目标/约束]]

$$=\frac{1}{2}\left(1-\frac{\dot{\mathbf{q}}_{\text{NCDS}}\cdot\nabla_{\mathbf{q}}\Gamma_{\mathcal{S}}(\mathbf{x},\mathbf{q})}{| $\dot{\mathbf{q}}_{\text{NCDS}}$ |\ | \nabla_{ $\mathbf{q}}\Gamma_{\mathcal{S}}(\mathbf{x},\mathbf{q})$ |}\right).$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 3: [[优化目标/约束]]

$$\begin{split}&\text{DTWD}\big{(}\tau_{\text{base}}(\mathbf{x}),\tau_{m}(\mathbf{x})\big{)}=\\ &\sum_{j\in l(\tau_{\text{base}}(\mathbf{x}))}\min_{i\in l(\tau_{m}(\mathbf{x}))}d\big{(}\tau_{i,\text{base}}(\mathbf{x}),\tau_{j,m}(\mathbf{x})\big{)}+\\ &\sum_{i\in l(\tau_{m}(\mathbf{x}))}\min_{j\in l(\tau_{\text{base}}(\mathbf{x}))}d\big{(}\tau_{i,\text{base}}(\mathbf{x}),\tau_{j,m}(\mathbf{x})\big{)},\end{split}$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 4: [[优化目标/约束]]

$$\mathcal{L}_{\text{eik}}(\Gamma_{\text{SDF}}(\mathbf{x};\boldsymbol{\theta}))=\left| $\lVert \nabla_{\mathbf{x}}\Gamma_{\text{SDF}}(\mathbf{x};\boldsymbol{\theta})\rVert-1\right$ |,$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 5: [[优化目标/约束]]

$$
\mathcal{L}_{\text{noise}}=-\frac{1}{T_{n}}\sum_{t=0}^{T_{n}-1}\left\lVert \mathbf{x}_{t+1}-\left(\tilde{\mathbf{x}}_{t}+\hat{\dot{\mathbf{x}}}_{t}\right)\right \rVert^{2},
$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 6: [[动力学或策略机制]]

$$\begin{split}\dot{\mathbf{x}}&=f_{\text{NCDS}}(\mathbf{x})\\ &=\dot{\mathbf{x}}_{0}+\int_{0}^{1}\hat{\mathbf{J}}_{f_{\text{NCDS}}}\big{(}c(\mathbf{x},t,\mathbf{x}_{0})\big{)}\dot{c}(\mathbf{x},t,\mathbf{x}_{0})\mathrm{d}t,\end{split}$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 7: [[动力学或策略机制]]

$$\begin{split}\psi(\mathbf{x},\mathbf{y})&=\gamma(\mathbf{x},\mathbf{y},t)=\mathbf{q},\\ &=\mathbf{y}-\int_{0}^{t}b\big{(}\mathbf{x},\gamma(\mathbf{x},\mathbf{y},u)\big{)}\nabla_{\mathbf{y}}\Gamma_{\text{SDF}}\big{(}\mathbf{x},\gamma(\mathbf{x},\mathbf{y},u)\big{)}du.\end{split}$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 8: [[动力学或策略机制]]

$$\begin{split}\psi(\mathbf{x},{\mathbf{q}})^{-1}&=\gamma(\mathbf{x},{\mathbf{q}},-t)^{-1}=\mathbf{y},\\ ={\mathbf{q}}-\int_{-t}^{0}b&\big{(}\mathbf{x},\gamma(\mathbf{x},{\mathbf{q}},u)^{-1}\big{)}\nabla_{\mathbf{q}}\Gamma_{\text{SDF}}\big{(}\mathbf{x},\gamma(\mathbf{x},{\mathbf{q}},u)^{-1}\big{)}du.\end{split}$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 9: [[优化目标/约束]]

$$
\mathcal{L}_{\text{grad}}(\Gamma_{\text{SDF}}(\mathbf{x};\boldsymbol{\theta}),\mathbf{g})=1-\frac{\nabla_{\mathbf{x}}\Gamma_{\text{SDF}}(\mathbf{x};\boldsymbol{\theta})\cdot\mathbf{g}}{\lVert \nabla_{\mathbf{x}}\Gamma_{\text{SDF}}(\mathbf{x};\boldsymbol{\theta})\rVert\lVert \mathbf{g} \rVert},
$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 10: [[优化目标/约束]]

$$\mathcal{L}_{\text{SDF}}(\Gamma_{\text{SDF}}(\mathbf{x};\boldsymbol{\theta}))=\left| $\text{clamp}\left(\Gamma_{\text{SDF}}(\mathbf{x};\boldsymbol{\theta}),\delta\right)-\text{clamp}(s,\delta)\right$ |,$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。
## 关键图表

### Figure 1: Overview of the proposed Signed Distance Field Diffeomorphic Transform: (1) A neural

![Figure 1](https://arxiv.org/html/2504.18860v1/x1.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Overview of the proposed Signed Distance Field Diffeomorphic Transform: (1) A neural”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 2: Obstacle avoidance with SDDC and SDC using a inverse barrier on a NCDS model trained

![Figure 2](https://arxiv.org/html/2504.18860v1/x10.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Obstacle avoidance with SDDC and SDC using a inverse barrier on a NCDS model trained”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 3: Real-world robot setup in a kitchen environment. It consists of a Franka-Emika Panda

![Figure 3](https://arxiv.org/html/2504.18860v1/x12.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Real-world robot setup in a kitchen environment. It consists of a Franka-Emika Panda”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### 论文表格

#### Table 1: Table I: Quantitative analysis of the influence of s grad s grad s_{$\text{grad}$}

| | MJ | RFC | VM | DTWD | D min D min  $\textbf{D}_{\textbf{min}} D min$ |
| --- | --- | --- | --- | --- | --- |
| s grad = 0.05 s grad 0.05  $\mathbf{s}_{\textbf{grad}}=\mathbf{0.05} grad =.05$ | | | | | |
| SDDC inv SDDC inv  $\text{SDDC}_{\text{inv}} SDDC inv$ | 0.74 e - 4 0.74 e 4 0.74e\!-\!4 0.74 - 4 | 10.19 10.19 10.19 10.19 | 0.49 0.49 0.49 0.49 | 123.8 123.8 123.8 123.8 | 0.28 0.28 0.28 0.28 |
| SDDC inv,swept SDDC inv,swept  $\text{SDDC}_{\text{inv,swept}} SDDC inv,swept$ | 0.69 e - 4 0.69 e 4 0.69e\!-\!4 0.69 - 4 | 3.80 3.80 3.80 3.80 | 0.42 0.42 0.42 0.42 | 84.9 84.9  $\mathbf{84.9}.9$ | 0.29 0.29 0.29 0.29 |
| SDC inv SDC inv  $\text{SDC}_{\text{inv}} SDC inv$ | 5.08 e - 4 5.08 e 4 5.08e\!-\!4 5.08 - 4 | 23.58 23.58 23.58 23.58 | 0.20 0.20 0.20 0.20 | 104.1 104.1 104.1 104.1 | 0.35 0.35 0.35 0.35 |
| SDC inv,swept SDC inv,swept  $\text{SDC}_{\text{inv,swept}} SDC inv,swept$ | 3.93 e - 4 3.93 e 4 3.93e\!-\!4 3.93 - 4 | 21.26 21.26 21.26 21.26 | 0.15 0.15  $\mathbf{0.15}.15$ | 99.9 99.9 99.9 99.9 | 0.34 0.34 0.34 0.34 |
| s grad = 0.10 s grad 0.10  $\mathbf{s}_{\textbf{grad}}=\mathbf{0.10} grad =.10$ | | | | | |
| SDDC inv SDDC inv  $\text{SDDC}_{\text{inv}} SDDC inv$ | 0.62 e - 4 0.62 e 4 0.62e\!-\!4 0.62 - 4 | 8.80 8.80 8.80 8.80 | 0.57 0.57 0.57 0.57 | 184.2 184.2 184.2 184.2 | 0.31 0.31 0.31 0.31 |
| SDDC inv,swept SDDC inv,swept  $\text{SDDC}_{\text{inv,swept}} SDDC inv,swept$ | 0.62 e - 4 0.62 e 4 0.62e\!-\!4 0.62 - 4 | 3.75 3.75  $\mathbf{3.75}.75$ | 0.54 0.54 0.54 0.54 | 127.8 127.8 127.8 127.8 | 0.32 0.32 0.32 0.32 |
| SDC inv SDC inv  $\text{SDC}_{\text{inv}} SDC inv$ | 4.81 e - 4 4.81 e 4 4.81e\!-\!4 4.81 - 4 | 15.50 15.50 15.50 15.50 | 0.26 0.26 0.26 0.26 | 163.4 163.4 163.4 163.4 | 0.42 0.42 0.42 0.42 |
| SDC inv,swept SDC inv,swept  $\text{SDC}_{\text{inv,swept}} SDC inv,swept$ | 3.25 e - 4 3.25 e 4 3.25e\!-\!4 3.25 - 4 | 12.59 12.59 12.59 12.59 | 0.20 0.20 0.20 0.20 | 149.9 149.9 149.9 149.9 | 0.41 0.41 0.41 0.41 |
| s grad = 0.20 s grad 0.20  $\mathbf{s}_{\textbf{grad}}=\mathbf{0.20} grad =.20$ | | | | | |
| SDDC inv SDDC inv  $\text{SDDC}_{\text{inv}} SDDC inv$ | 0.56 e - 4 0.56 e 4  $\mathbf{0.56e\!-\!4}.56 -$ | 8.14 8.14 8.14 8.14 | 0.69 0.69 0.69 0.69 | 288.5 288.5 288.5 288.5 | 0.34 0.34 0.34 0.34 |
| SDDC inv,swept SDDC inv,swept  $\text{SDDC}_{\text{inv,swept}} SDDC inv,swept$ | 0.62 e - 4 0.62 e 4 0.62e\!-\!4 0.62 - 4 | 3.99 3.99 3.99 3.99 | 0.70 0.70 0.70 0.70 | 177.2 177.2 177.2 177.2 | 0.35 0.35 0.35 0.35 |
| SDC inv SDC inv  $\text{SDC}_{\text{inv}} SDC inv$ | 3.81 e - 4 3.81 e 4 3.81e\!-\!4 3.81 - 4 | 8.58 8.58 8.58 8.58 | 0.36 0.36 0.36 0.36 | 269.6 269.6 269.6 269.6 | 0.53 0.53  $\mathbf{0.53}.53$ |
| SDC inv,swept SDC inv,swept  $\text{SDC}_{\text{inv,swept}} SDC inv,swept$ | 3.33 e - 4 3.33 e 4 3.33e\!-\!4 3.33 - 4 | 5.99 5.99 5.99 5.99 | 0.29 0.29 0.29 0.29 | 238.6 238.6 238.6 238.6 | 0.52 0.52 0.52 0.52 |

**说明**: 该表格来自论文中的公式、奖励项或实验设置；为避免 PDF 抽取噪声，网页中仅保留可渲染的核心符号和数值。

#### Table 2: Table II: Comparison of the proposed SDC and SDDC methods against state-of-the-art baselines. Three different implicit

| | GP | MLP | BP | | | | | | | | | | | | |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Methods | MJ | RFC | VM | DTWD | D min D min  $\textbf{D}_{\textbf{min}} D min$ | MJ | RFC | VM | DTWD | D min D min  $\textbf{D}_{\textbf{min}} D min$ | MJ | RFC | VM | DTWD | D min D min  $\textbf{D}_{\textbf{min}} D min$ |
| SDDC inv SDDC inv  $\text{SDDC}_{\text{inv}} SDDC inv$ | 0.62 e - 4 0.62 e 4 0.62e\!-\!4 0.62 - 4 | 8.80 8.80 8.80 8.80 | 0.57 0.57 0.57 0.57 | 184.2 184.2 184.2 184.2 | 0.31 0.31 0.31 0.31 | 4.02 e - 4 4.02 e 4 4.02e\!-\!4 4.02 - 4 | 7.78 7.78 7.78 7.78 | 0.61 0.61 0.61 0.61 | 174.2 174.2 174.2 174.2 | 0.34 0.34 0.34 0.34 | 1.51 e - 4 1.51 e 4 1.51e\!-\!4 1.51 - 4 | 4.38 4.38 4.38 4.38 | 0.74 0.74 0.74 0.74 | 222.4 222.4 222.4 222.4 | 0.31 0.31 0.31 0.31 |
| SDDC inv,swept SDDC inv,swept  $\text{SDDC}_{\text{inv,swept}} SDDC inv,swept$ | 0.62 e - 4 0.62 e 4 0.62e\!-\!4 0.62 - 4 | 3.75 3.75  $\mathbf{3.75}.75$ | 0.54 0.54 0.54 0.54 | 127.8 127.8 127.8 127.8 | 0.32 0.32 0.32 0.32 | 3.37 e - 4 3.37 e 4 3.37e\!-\!4 3.37 - 4 | 6.96 6.96  $\mathbf{6.96}.96$ | 0.47 0.47 0.47 0.47 | 129.9 129.9 129.9 129.9 | 0.33 0.33 0.33 0.33 | 1.16 e - 4 1.16 e 4 1.16e\!-\!4 1.16 - 4 | 4.14 4.14  $\mathbf{4.14}.14$ | 0.49 0.49 0.49 0.49 | 134.8 134.8 134.8 134.8 | 0.30 0.30 0.30 0.30 |
| SDC inv SDC inv  $\text{SDC}_{\text{inv}} SDC inv$ | 5.08 e - 4 5.08 e 4 5.08e\!-\!4 5.08 - 4 | 23.58 23.58 23.58 23.58 | 0.20 0.20 0.20 0.20 | 104.1 104.1 104.1 104.1 | 0.35 0.35 0.35 0.35 | 7.44 e - 4 7.44 e 4 7.44e\!-\!4 7.44 - 4 | 28.47 28.47 28.47 28.47 | 0.20 0.20 0.20 0.20 | 98.5 98.5 98.5 98.5 | 0.37 0.37 0.37 0.37 | 3.16 e - 4 3.16 e 4 3.16e\!-\!4 3.16 - 4 | 8.42 8.42 8.42 8.42 | 0.10 0.10 0.10 0.10 | 98.8 98.8 98.8 98.8 | 0.34 0.34 0.34 0.34 |
| SDC inv,swept SDC inv,swept  $\text{SDC}_{\text{inv,swept}} SDC inv,swept$ | 3.93 e - 4 3.93 e 4 3.93e\!-\!4 3.93 - 4 | 21.26 21.26 21.26 21.26 | 0.15 0.15  $\mathbf{0.15}.15$ | 99.9 99.9  $\mathbf{99.9}.9$ | 0.34 0.34 0.34 0.34 | 6.33 e - 4 6.33 e 4 6.33e\!-\!4 6.33 - 4 | 27.79 27.79 27.79 27.79 | 0.15 0.15  $\mathbf{0.15}.15$ | 97.7 97.7  $\mathbf{97.7}.7$ | 0.35 0.35 0.35 0.35 | 2.12 e - 4 2.12 e 4 2.12e\!-\!4 2.12 - 4 | 7.71 7.71 7.71 7.71 | 0.09 0.09  $\mathbf{0.09}.09$ | 95.4 95.4  $\mathbf{95.4}.4$ | 0.33 0.33 0.33 0.33 |
| DT inv DT inv  $\text{DT}_{\text{inv}} DT inv$ | 0.42 e - 4 0.42 e 4 0.42e\!-\!4 0.42 - 4 | 9.74 9.74 9.74 9.74 | 0.42 0.42 0.42 0.42 | 151.3 151.3 151.3 151.3 | 0.40 0.40  $\mathbf{0.40}.40$ | 3.27 e - 4 3.27 e 4 3.27e\!-\!4 3.27 - 4 | 12.31 12.31 12.31 12.31 | 0.34 0.34 0.34 0.34 | 132.9 132.9 132.9 132.9 | 0.41 0.41  $\mathbf{0.41}.41$ | 1.58 e - 4 1.58 e 4 1.58e\!-\!4 1.58 - 4 | 5.32 5.32 5.32 5.32 | 0.42 0.42 0.42 0.42 | 156.8 156.8 156.8 156.8 | 0.37 0.37  $\mathbf{0.37}.37$ |
| DT inv,swept DT inv,swept  $\text{DT}_{\text{inv,swept}} DT inv,swept$ | 0.52 e - 4 0.52 e 4 0.52e\!-\!4 0.52 - 4 | 8.34 8.34 8.34 8.34 | 0.40 0.40 0.40 0.40 | 143.4 143.4 143.4 143.4 | 0.38 0.38 0.38 0.38 | 2.67 e - 4 2.67 e 4 2.67e\!-\!4 2.67 - 4 | 8.95 8.95 8.95 8.95 | 0.34 0.34 0.34 0.34 | 132.9 132.9 132.9 132.9 | 0.40 0.40 0.40 0.40 | 1.10 e - 4 1.10 e 4 1.10e\!-\!4 1.10 - 4 | 5.67 5.67 5.67 5.67 | 0.28 0.28 0.28 0.28 | 147.9 147.9 147.9 147.9 | 0.34 0.34 0.34 0.34 |
| MM | 2.66 e - 4 2.66 e 4 2.66e\!-\!4 2.66 - 4 | 9.66 9.66 9.66 9.66 | 0.45 0.45 0.45 0.45 | 199.4 199.4 199.4 199.4 | 0.34 0.34 0.34 0.34 | 9.13 e - 4 9.13 e 4 9.13e\!-\!4 9.13 - 4 | 18, 63 18 63 18,63 18, 63 | 0.42 0.42 0.42 0.42 | 177.1 177.1 177.1 177.1 | 0.35 0.35 0.35 0.35 | 0.34 e - 4 0.34 e 4 0.34e\!-\!4 0.34 - 4 | 5, 13 5 13 5,13 5, 13 | 0.62 0.62 0.62 0.62 | 208.5 208.5 208.5 208.5 | 0.33 0.33 0.33 0.33 |
| ARPF | 0.15 e - 4 0.15 e 4  $\mathbf{0.15e\!-\!4}.15 -$ | 8.08 8.08 8.08 8.08 | 0.61 0.61 0.61 0.61 | 146.6 146.6 146.6 146.6 | 0.32 0.32 0.32 0.32 | 2.00 e - 4 2.00 e 4  $\mathbf{2.00e\!-\!4}.00 -$ | 22, 13 22 13 22,13 22, 13 | 0.57 0.57 0.57 0.57 | 135.1 135.1 135.1 135.1 | 0.33 0.33 0.33 0.33 | 0.02 e - 4 0.02 e 4  $\mathbf{0.02e\!-\!4}.02 -$ | 6.72 6.72 6.72 6.72 | 0.84 0.84 0.84 0.84 | 198.3 198.3 198.3 198.3 | 0.31 0.31 0.31 0.31 |

**说明**: 该表格来自论文中的公式、奖励项或实验设置；为避免 PDF 抽取噪声，网页中仅保留可渲染的核心符号和数值。

#### Table 3: Table III: Comparison of the inference time t step t step t_{$\text{step}$} step e

| Methods | GP | MLP | BP | HM |
| --- | --- | --- | --- | --- |
| SDC/SDDC/DT | 2.2 2.2 2.2 2.2 | 4.0 4.0 4.0 4.0 | 9.0 9.0 9.0 9.0 | 2.6 2.6 2.6 2.6 |
| MM | 1.7 1.7 1.7 1.7 | 3.0 3.0 3.0 3.0 | 5.7 5.7 5.7 5.7 | - |
| ARPF | 1.2 1.2 1.2 1.2 | 2.5 2.5 2.5 2.5 | 5.2 5.2 5.2 5.2 | - |

**说明**: 该表格来自论文中的公式、奖励项或实验设置；为避免 PDF 抽取噪声，网页中仅保留可渲染的核心符号和数值。

#### Table 4: Table IV: A comparison of the inference time t flow t flow t_{$\text{flow}$} flow

| | t flow t flow  $\mathbf{t}_{\textbf{flow}} flow$ | t jacobian t jacobian  $\mathbf{t}_{\textbf{jacobian}} jacobian$ | | | | | | |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Solvers | GP | MLP | BP | HM | GP | MLP | BP | HM |
| SDC/SDDC | | | | | | | | |
| Convex | 1.8 1.8 1.8 1.8 | 3.4 3.4 3.4 3.4 | 7.3 7.3 7.3 7.3 | 1.9 1.9 1.9 1.9 | 0.4 0.4 0.4 0.4 | 0.5 0.5 0.5 0.5 | 1.5 1.5 1.5 1.5 | 0.6 0.6 0.6 0.6 |
| Euler | 3.7 3.7 3.7 3.7 | 6.7 6.7 6.7 6.7 | 14.8 14.8 14.8 14.8 | 3.2 3.2 3.2 3.2 | 0.5 0.5 0.5 0.5 | 0.8 0.8 0.8 0.8 | 3.0 3.0 3.0 3.0 | 0.9 0.9 0.9 0.9 |
| Runge- Kutta | 10.2 10.2 10.2 10.2 | 18.6 18.6 18.6 18.6 | 40.5 40.5 40.5 40.5 | 10.0 10.0 10.0 10.0 | 1.0 1.0 1.0 1.0 | 1.4 1.4 1.4 1.4 | 5.7 5.7 5.7 5.7 | 2.0 2.0 2.0 2.0 |

**说明**: 该表格来自论文中的公式、奖励项或实验设置；为避免 PDF 抽取噪声，网页中仅保留可渲染的核心符号和数值。

#### Table 5: Table V: Overview of various ODE solvers

| ODE Solver | Description | Pros | Cons | Complexity |
| --- | --- | --- | --- | --- |
| Convex | One-step solution of flow | + Fastest solver | - Only for convex surfaces | O (1) O 1  $\mathcal{O}(1) (1)$ |
| Euler method | Numerical first-order ODE solver | + Any curvature + Fast | - Very imprecise | O (N) O N  $\mathcal{O}(N) ()$ |
| Runge-Kutta (RK4) | Numerical fourth-order ODE solver | + Any curvature + Precise | - Slowest method | O (4 N) O 4 N  $\mathcal{O}(4N) (4)$ |

**说明**: Table V: Overview of various ODE solvers

#### Table 6: Table VI: Comparison of the inference time t step t step t_{$\text{step}$} step en

| Solvers | RDF | CDF |
| --- | --- | --- |
| SDC | | |
| Convex | 20.9 20.9 20.9 20.9 | 2.4 2.4 2.4 2.4 |
| Euler | 41.1 41.1 41.1 41.1 | 4.5 4.5 4.5 4.5 |
| Runge-Kutta | 91.2 91.2 91.2 91.2 | 9.3 9.3 9.3 9.3 |

**说明**: 该表格来自论文中的公式、奖励项或实验设置；为避免 PDF 抽取噪声，网页中仅保留可渲染的核心符号和数值。
## 实验解读

- 评价重点:围绕 adaptive-control、碰撞避免、flow-matching、模仿学习、鲁棒控制,优先看真实机器人结果、扰动恢复、接触稳定性、控制误差、成功率、实时性和 sim-to-real 差距。
- System0 视角:实验结论不只看最终成功率,还要看闭环过程中是否能处理状态估计误差、执行延迟、碰撞/接触突变和外部扰动。
- 与上层系统的关系:如果方法能在快闭环内稳定工作,它可以作为 System1 技能或 System2 规划的底层执行保障;如果只在离线或慢速设定下成立,则还不能直接作为身体级快闭环模块。
## 批判性思考

### 优点

- 论文把 adaptive-control、碰撞避免、flow-matching、模仿学习、鲁棒控制 放到机器人身体级闭环里讨论,和只做高层语义推理的工作相比,更接近真实机器人部署时会遇到的状态、动力学和安全约束。
- 若论文同时给出公式、系统结构图、硬件实验或消融结果,这些证据可以帮助判断方法是否真的能支撑毫秒到百毫秒级反应。

### 局限

- 需要重点核查控制频率、推理延迟、传感延迟、失败恢复、硬件磨损、未见场景泛化和安全边界;这些因素决定它能否成为可靠的 System0 模块。
- 对学习型方法,还要区分训练环境中的鲁棒性和真实部署中的鲁棒性,尤其是接触丰富任务、动态运动任务和人形机器人任务。
## 论文定位

- 研究对象:Diffeomorphic Obstacle Avoidance for Contractive Dynamical Systems via Implicit Representations。
- 关键词:adaptive-control、碰撞避免、flow-matching、模仿学习、鲁棒控制、safe-control、全身控制。
- System0 关联:关注机器人身体级快反应链路中“状态/接触/局部感知 -> 约束/优化/策略 -> 动作”的短闭环,不把问题停留在语言规划或任务分解层。

## 速查卡片

> [!summary] Diffeomorphic Obstacle Avoidance
> - **论文**: https://www.roboticsproceedings.org/rss21/p162.pdf
> - **arXiv**: http://arxiv.org/abs/2504.18860v1
> - **arXiv HTML**: https://arxiv.org/html/2504.18860v1
