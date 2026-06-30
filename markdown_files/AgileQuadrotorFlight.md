---
title: "Learning Agile Quadrotor Flight in the Real World"
method_name: "Agile Quadrotor Flight"
authors: ["Yunfan Ren"]
year: 2026
venue: "RSS"
tags: ["robust-control", "real-time-control", "safe-control", "adaptive-control", "agile-locomotion", "navigation"]
image_source: "online"
arxiv_html: "https://arxiv.org/html/2602.10111v1"
---
# Agile Quadrotor Flight
## 一句话总结

> Learning Agile Quadrotor Flight in the Real World 主要落在 [[actuator-modeling]]、[[adaptive-control]]、[[aerial-robotics]]、[[agile-locomotion]] 这一类问题上;从 System0 视角看,它的价值在于把身体状态、接触变化、局部感知或动力学约束接到机器人低层闭环里,让动作生成更稳定、安全、实时。
## 核心贡献

1. 围绕 **Learning Agile Quadrotor Flight in the Real World** 建立了一个与 actuator-modeling、adaptive-control、aerial-robotics、agile-locomotion、navigation、实时控制 相关的机器人问题设定,重点不是高层语义规划,而是身体级闭环中的状态变化、接触变化或控制约束。
2. 方法层面的关键价值在于把感知、动力学、约束优化、策略学习或硬件反馈组织成可执行的低层动作生成机制,适合放进 System0 的快反应模块中理解。
3. 对系统集成而言,这类工作补上了 System1 技能选择与电机级控制之间的一层:它负责把上层目标变成动态可行、可恢复、可安全过滤的动作。
4. 若论文包含真实机器人、接触任务或仿真到真实迁移,核心应看它是否报告了闭环稳定性、扰动恢复、硬件延迟、失败模式和跨场景泛化。
## 问题背景

这篇论文可以放在 System0 的“具身快系统”背景下阅读:机器人需要在很短时间内根据本体状态、接触、力/触觉、局部视觉或动力学预测做出动作,而不是等待慢速规划器重新推理整个任务。actuator-modeling、adaptive-control、aerial-robotics、agile-locomotion、navigation、实时控制、鲁棒控制 相关问题的难点通常不在“知道要做什么”,而在动作执行过程中状态会快速偏离、接触会突然改变、约束会实时收紧,低层系统必须及时修正。

### System0 关联

精读时优先看三件事:第一,论文使用哪些快速反馈量,例如关节状态、接触状态、力觉、触觉、深度或局部视觉;第二,反馈如何进入 MPC、WBC、策略网络、安全过滤器或恢复控制器;第三,输出是否直接影响机器人下一步动作,而不是只服务离线分析或高层任务规划。
## 方法详解

### System0 视角解析

从 System0 角度看,这篇论文的方法部分应当围绕 actuator-modeling、adaptive-control、aerial-robotics、agile-locomotion、navigation、实时控制、鲁棒控制、safe-control 来读。核心问题是:系统如何从快速变化的机器人状态中构造控制输入,如何处理接触、碰撞、约束、模型误差或执行延迟,以及如何把这些信息变成下一时刻的动作。

如果方法里有优化问题,重点看目标函数、约束和求解频率;如果方法里有学习策略,重点看观测空间、动作空间、训练扰动和部署时的闭环频率;如果方法里有硬件或传感器,重点看传感链路是否真的进入控制回路。这样读可以把论文和 System2 语言规划、System1 技能编排区分开来。

### 关键公式与机制

#### 公式 1: [[动力学或策略机制]]

$$\mathbf{x}_{k+1}=\mathbf{F}_{\Delta t}(\mathbf{x}_{k},\mathbf{u}_{k};\theta):=\Phi_{\mathrm{RK4}}\!\left(\mathbf{x}_{k},\mathbf{u}_{k},\mathbf{f}_{\mathrm{hybrid}};\Delta t\right).$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 2: [[优化目标/约束]]

$$\mathcal{L}_{\text{res}}(\theta)=\frac{1}{| $\mathcal{B}$ |}\sum_{k\in\mathcal{B}}\mathcal{D}({\mathbf{x}}_{k+1},\hat{\mathbf{x}}_{k+1})+\lambda_{\text{reg}}\sum_{l}\lVert \mathbf{W}_{l} \rVert_{\sigma}.$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 3: [[动力学或策略机制]]

$${\mathbf{x}}_{k+1}=\mathbf{F}_{\Delta t}({\mathbf{x}}_{k},\mathbf{u}_{k};\theta),\quad k=0,\dots,H-1.$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 4: [[优化目标/约束]]

$$\mathcal{J}(\phi)=\sum_{k=0}^{H-1}\gamma^{k}\,r({\mathbf{x}}_{k},\mathbf{u}_{k},\mathbf{x}_{\mathrm{ref},k}),$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 5: [[动力学或策略机制]]

$$\mathbf{p}_{\text{ref}}(\tau;\alpha)=\mathbf{C}\boldsymbol{\beta}\left(\frac{\tau}{\alpha}\right).$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 6: [[优化目标/约束]]

$$\mathcal{J}_{\text{ATS}}(\alpha)=\lambda_{\mathrm{speed}}\,\alpha+\lambda_{\mathrm{safe}}\sum_{k=0}^{H-1}\Psi\!\left(\mathcal{E}_{k}(\alpha)-\mathcal{E}_{\mathrm{th}}\right),$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 7: [[优化目标/约束]]

$$\frac{{\mathrm{d}}\mathcal{J}_{\text{ATS}}}{{\mathrm{d}}\alpha}=\lambda_{\mathrm{speed}}+\lambda_{\mathrm{safe}}\sum_{k=0}^{H-1}\Psi^{\prime}\!\left(\mathcal{E}_{k}(\alpha)-\mathcal{E}_{\mathrm{th}}\right)\,\frac{{\mathrm{d}}\mathcal{E}_{k}(\alpha)}{{\mathrm{d}}\alpha},$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 8: [[优化目标/约束]]

$$\frac{{\mathrm{d}}\mathcal{E}_{k}}{{\mathrm{d}}\alpha}=\frac{\partial\mathcal{E}_{k}}{\partial\mathbf{x}_{\text{ref},k}}\frac{\partial\mathbf{x}_{\text{ref},k}}{\partial\alpha}+\frac{\partial\mathcal{E}_{k}}{\partial\hat{\mathbf{x}}_{k}}\mathbf{S}_{k}.$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 9: [[动力学或策略机制]]

$$\frac{{\mathrm{d}}\mathbf{u}_{k}}{{\mathrm{d}}\alpha}=\frac{\partial\pi_{\phi}}{\partial\mathbf{o}_{k}}\left(\frac{\partial\mathbf{o}_{k}}{\partial\hat{\mathbf{x}}_{k}}\mathbf{S}_{k}+\frac{\partial\mathbf{o}_{k}}{\partial\mathbf{x}_{\mathrm{ref},k}}\frac{\partial\mathbf{x}_{\mathrm{ref},k}}{\partial\alpha}\right),$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 10: [[优化目标/约束]]

$$\alpha\leftarrow\Pi_{[\alpha_{\min},\,\alpha_{\max}]}\!(\alpha-\eta_{\alpha}\nabla_{\alpha}\mathcal{J}_{\text{ATS}}).$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。
## 关键图表

### Figure 1: Overview of the self-adaptive autonomous flight framework. The system operates as a co

![Figure 1](https://arxiv.org/html/2602.10111v1/x1.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Overview of the self-adaptive autonomous flight framework. The system operates as a co”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 2: The optimization landscape for ATS. The heatmap visualizes the composite potential J A

![Figure 2](https://arxiv.org/html/2602.10111v1/x2.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“The optimization landscape for ATS. The heatmap visualizes the composite potential J A”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 3: Ablation study of residual dynamics and anchored rollouts. Note: Since no existing met

![Figure 3](https://arxiv.org/html/2602.10111v1/x7.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Ablation study of residual dynamics and anchored rollouts. Note: Since no existing met”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### 论文表格

#### Table 1: Table 1

| | p  ̇ \displaystyle $\dot{\mathbf{p}}$ | = v, \displaystyle= $\mathbf{v},$ | | (1a) |
| --- | --- | --- | --- | --- |
| | v  ̇ \displaystyle $\dot{\mathbf{v}}$ | c Re - g e a res \displaystyle c\ $\mathbf{R}\mathbf{e}_{3}-g\mathbf{e}_{3}+\mathbf{a}_{\mathrm{res}}(\boldsymbol{\zeta};\theta),$ | | (1b) |
| | R  ̇ \displaystyle $\dot{\mathbf{R}}$ | R cmd res ∧ \displaystyle $\mathbf{R}\Big(\boldsymbol{\omega}_{\mathrm{cmd}}+\boldsymbol{\omega}_{\mathrm{res}}(\boldsymbol{\zeta};\theta)\Big)^{\wedge},$ | | (1c) |

**说明**: Table 1

#### Table 2: Table 2

| | $\boldsymbol{\zeta}:=[\mathbf{p}^{\top},\ \mathbf{v}^{\top},\ \mathrm{vec}(\mathbf{R})^{\top},\ \mathbf{u}^{\top}]^{\top},$ | | (2) |
| --- | --- | --- | --- |

**说明**: Table 2

#### Table 3: Table 3

| | $\mathbf{x}_{k+1}=\mathbf{F}_{\Delta t}(\mathbf{x}_{k},\mathbf{u}_{k};\theta):=\Phi_{\mathrm{RK4}}\!\left(\mathbf{x}_{k},\mathbf{u}_{k},\mathbf{f}_{\mathrm{hybrid}};\Delta t\right).$ | | (3) |
| --- | --- | --- | --- |

**说明**: Table 3

#### Table 4: Table 4

| | $\mathcal{L}_{\text{res}}(\theta)=\frac{1}{\lVert \mathcal{B} \rVert}\sum_{k\in\mathcal{B}}\mathcal{D}({\mathbf{x}}_{k+1},\hat{\mathbf{x}}_{k+1})+\lambda_{\text{reg}}\sum_{l}\lVert \mathbf{W}_{l} \rVert_{\sigma}.$ | | (4) |
| --- | --- | --- | --- |

**说明**: Table 4

#### Table 5: Table 5

| | $\mathbf{x}}_{k+1}=\mathbf{F}_{\Delta t}({\mathbf{x}}_{k},\mathbf{u}_{k};\theta),\quad k=0,\dots,H-1.$ | | (5) |
| --- | --- | --- | --- |

**说明**: Table 5

#### Table 6: Table 6

| | $\mathcal{J}(\phi)=\sum_{k=0}^{H-1}\gamma^{k}\,r({\mathbf{x}}_{k},\mathbf{u}_{k},\mathbf{x}_{\mathrm{ref},k}),$ | | (6) |
| --- | --- | --- | --- |

**说明**: Table 6

#### Table 7: Table 7

| | $\mathbf{p}_{\text{ref}}(\tau;\alpha)=\mathbf{C}\boldsymbol{\beta}\left(\frac{\tau}{\alpha}\right).$ | | (7) |
| --- | --- | --- | --- |

**说明**: Table 7

#### Table 8: Table 8

| | $\mathcal{J}_{\text{ATS}}(\alpha)=\lambda_{\mathrm{speed}}\,\alpha+\lambda_{\mathrm{safe}}\sum_{k=0}^{H-1}\Psi\!\left(\mathcal{E}_{k}(\alpha)-\mathcal{E}_{\mathrm{th}}\right),$ | | (8) |
| --- | --- | --- | --- |

**说明**: Table 8

#### Table 9: Table 9

| | $\mathcal{E}_{k}(\alpha):=\mathcal{D}(\hat{\mathbf{x}}_{k}(\alpha),\,\mathbf{x}_{\mathrm{ref},k}(\alpha)).$ | | (9) |
| --- | --- | --- | --- |

**说明**: Table 9

#### Table 10: Table 10

| | $\frac{{\mathrm{d}}\mathcal{J}_{\text{ATS}}}{{\mathrm{d}}\alpha}=\lambda_{\mathrm{speed}}+\lambda_{\mathrm{safe}}\sum_{k=0}^{H-1}\Psi^{\prime}\!\left(\mathcal{E}_{k}(\alpha)-\mathcal{E}_{\mathrm{th}}\right)\,\frac{{\mathrm{d}}\mathcal{E}_{k}(\alpha)}{{\mathrm{d}}\alpha},$ | | (10) |
| --- | --- | --- | --- |

**说明**: Table 10

#### Table 11: Table 11

| | $\frac{{\mathrm{d}}\mathcal{E}_{k}}{{\mathrm{d}}\alpha}=\frac{\partial\mathcal{E}_{k}}{\partial\mathbf{x}_{\text{ref},k}}\frac{\partial\mathbf{x}_{\text{ref},k}}{\partial\alpha}+\frac{\partial\mathcal{E}_{k}}{\partial\hat{\mathbf{x}}_{k}}\mathbf{S}_{k}.$ | | (11) |
| --- | --- | --- | --- |

**说明**: Table 11

#### Table 12: Table 12

| | $\mathbf{S}_{k+1}=\mathbf{A}_{k}\,\mathbf{S}_{k}+\mathbf{B}_{k}\,\frac{{\mathrm{d}}\mathbf{u}_{k}}{{\mathrm{d}}\alpha},$ | | (12) |
| --- | --- | --- | --- |

**说明**: Table 12

#### Table 13: Table 13

| | $\mathbf{A}_{k}:=\left.\frac{\partial\mathbf{F}_{\Delta t}}{\partial\mathbf{x}}\right \rVert_{(\bar{\mathbf{x}}_{k},\bar{\mathbf{u}}_{k})},\mathbf{B}_{k}:=\left.\frac{\partial\mathbf{F}_{\Delta t}}{\partial\mathbf{u}}\right \rVert_{(\bar{\mathbf{x}}_{k},\bar{\mathbf{u}}_{k})}.$ | | (13) |
| --- | --- | --- | --- |

**说明**: Table 13

#### Table 14: Table 14

| | $\frac{{\mathrm{d}}\mathbf{u}_{k}}{{\mathrm{d}}\alpha}=\frac{\partial\pi_{\phi}}{\partial\mathbf{o}_{k}}\left(\frac{\partial\mathbf{o}_{k}}{\partial\hat{\mathbf{x}}_{k}}\mathbf{S}_{k}+\frac{\partial\mathbf{o}_{k}}{\partial\mathbf{x}_{\mathrm{ref},k}}\frac{\partial\mathbf{x}_{\mathrm{ref},k}}{\partial\alpha}\right),$ | | (14) |
| --- | --- | --- | --- |

**说明**: Table 14

#### Table 15: Table 15

| | $\leftarrow\Pi_{[\alpha_{\min},\,\alpha_{\max}]}\!(\alpha-\eta_{\alpha}\nabla_{\alpha}\mathcal{J}_{\text{ATS}}).$ | | (15) |
| --- | --- | --- | --- |

**说明**: Table 15
## 实验解读

- 评价重点:围绕 actuator-modeling、adaptive-control、aerial-robotics、agile-locomotion、navigation,优先看真实机器人结果、扰动恢复、接触稳定性、控制误差、成功率、实时性和 sim-to-real 差距。
- System0 视角:实验结论不只看最终成功率,还要看闭环过程中是否能处理状态估计误差、执行延迟、碰撞/接触突变和外部扰动。
- 与上层系统的关系:如果方法能在快闭环内稳定工作,它可以作为 System1 技能或 System2 规划的底层执行保障;如果只在离线或慢速设定下成立,则还不能直接作为身体级快闭环模块。
## 批判性思考

### 优点

- 论文把 actuator-modeling、adaptive-control、aerial-robotics、agile-locomotion、navigation 放到机器人身体级闭环里讨论,和只做高层语义推理的工作相比,更接近真实机器人部署时会遇到的状态、动力学和安全约束。
- 若论文同时给出公式、系统结构图、硬件实验或消融结果,这些证据可以帮助判断方法是否真的能支撑毫秒到百毫秒级反应。

### 局限

- 需要重点核查控制频率、推理延迟、传感延迟、失败恢复、硬件磨损、未见场景泛化和安全边界;这些因素决定它能否成为可靠的 System0 模块。
- 对学习型方法,还要区分训练环境中的鲁棒性和真实部署中的鲁棒性,尤其是接触丰富任务、动态运动任务和人形机器人任务。
## 论文定位

- 研究对象:Learning Agile Quadrotor Flight in the Real World。
- 关键词:actuator-modeling、adaptive-control、aerial-robotics、agile-locomotion、navigation、实时控制、鲁棒控制、safe-control。
- System0 关联:关注机器人身体级快反应链路中“状态/接触/局部感知 -> 约束/优化/策略 -> 动作”的短闭环,不把问题停留在语言规划或任务分解层。

## 速查卡片

> [!summary] Agile Quadrotor Flight
> - **论文**: https://www.roboticsproceedings.org/rss22/p135.pdf
> - **arXiv**: http://arxiv.org/abs/2602.10111v1
> - **arXiv HTML**: https://arxiv.org/html/2602.10111v1
