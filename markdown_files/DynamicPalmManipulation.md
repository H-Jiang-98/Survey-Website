---
title: "Dynamic On-Palm Manipulation via Controlled Sliding"
method_name: "Dynamic Palm Manipulation"
authors: ["William Yang"]
year: 2024
venue: "RSS"
tags: ["contact-reasoning", "robot-manipulation", "robust-control", "adaptive-control", "contact-rich-manipulation", "model-predictive-control"]
image_source: "online"
arxiv_html: "https://arxiv.org/html/2405.08731v2"
---
# Dynamic Palm Manipulation
## 一句话总结

> Dynamic On-Palm Manipulation via Controlled Sliding 主要落在 [[actuator-modeling]]、[[adaptive-control]]、[[contact-estimation]]、[[contact-implicit-optimization]] 这一类问题上;从 System0 视角看,它的价值在于把身体状态、接触变化、局部感知或动力学约束接到机器人低层闭环里,让动作生成更稳定、安全、实时。
## 核心贡献

1. 围绕 **Dynamic On-Palm Manipulation via Controlled Sliding** 建立了一个与 actuator-modeling、adaptive-control、contact-estimation、contact-implicit-optimization、接触推理、接触丰富操作 相关的机器人问题设定,重点不是高层语义规划,而是身体级闭环中的状态变化、接触变化或控制约束。
2. 方法层面的关键价值在于把感知、动力学、约束优化、策略学习或硬件反馈组织成可执行的低层动作生成机制,适合放进 System0 的快反应模块中理解。
3. 对系统集成而言,这类工作补上了 System1 技能选择与电机级控制之间的一层:它负责把上层目标变成动态可行、可恢复、可安全过滤的动作。
4. 若论文包含真实机器人、接触任务或仿真到真实迁移,核心应看它是否报告了闭环稳定性、扰动恢复、硬件延迟、失败模式和跨场景泛化。
## 问题背景

这篇论文可以放在 System0 的“具身快系统”背景下阅读:机器人需要在很短时间内根据本体状态、接触、力/触觉、局部视觉或动力学预测做出动作,而不是等待慢速规划器重新推理整个任务。actuator-modeling、adaptive-control、contact-estimation、contact-implicit-optimization、接触推理、接触丰富操作、grasping 相关问题的难点通常不在“知道要做什么”,而在动作执行过程中状态会快速偏离、接触会突然改变、约束会实时收紧,低层系统必须及时修正。

### System0 关联

精读时优先看三件事:第一,论文使用哪些快速反馈量,例如关节状态、接触状态、力觉、触觉、深度或局部视觉;第二,反馈如何进入 MPC、WBC、策略网络、安全过滤器或恢复控制器;第三,输出是否直接影响机器人下一步动作,而不是只服务离线分析或高层任务规划。
## 方法详解

### System0 视角解析

从 System0 角度看,这篇论文的方法部分应当围绕 actuator-modeling、adaptive-control、contact-estimation、contact-implicit-optimization、接触推理、接触丰富操作、grasping、模型预测控制 来读。核心问题是:系统如何从快速变化的机器人状态中构造控制输入,如何处理接触、碰撞、约束、模型误差或执行延迟,以及如何把这些信息变成下一时刻的动作。

如果方法里有优化问题,重点看目标函数、约束和求解频率;如果方法里有学习策略,重点看观测空间、动作空间、训练扰动和部署时的闭环频率;如果方法里有硬件或传感器,重点看传感链路是否真的进入控制回路。这样读可以把论文和 System2 语言规划、System1 技能编排区分开来。

### 关键公式与机制

#### 公式 1: [[动力学或策略机制]]

$$\dot{y}=\frac{\partial\psi}{\partial q}\dot{q}=J_{y,i}\dot{q}$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 2: [[动力学或策略机制]]

$$y_{des,1}=[1,0,0,0],\dot{y}_{des,1}=\ddot{y}_{des,1}=[0,0,0]$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 3: [[动力学或策略机制]]

$$q_{lcs}=\begin{bmatrix}ee_{x}\\ ee_{y}\\ ee_{z}\\ tray_{qw}\\ tray_{qx}\\ tray_{qy}\\ tray_{qz}\\ tray_{x}\\ tray_{y}\\ tray_{z}\end{bmatrix},v_{lcs}=\begin{bmatrix}ee_{vx}\\ ee_{vy}\\ ee_{vz}\\ tray_{wx}\\ tray_{wy}\\ tray_{wz}\\ tray_{vx}\\ tray_{vy}\\ tray_{vz}\end{bmatrix}.$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 4: [[动力学或策略机制]]

$$G=\begin{bmatrix}G_{x}&&\\ &G_{\lambda}&\\ &&G_{u}\end{bmatrix}$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 5: [[动力学或策略机制]]

$$\left\lVert \lambda-{\lambda_{ee}}\right \rVert^{2}_{W}+\sum_{i}^{N}\left\lVert(\ddot{y}-\ddot{y}_{cmd})_{i}\right \rVert^{2}_{W_{i}}$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 6: [[动力学或策略机制]]

$$M(q)\ddot{q}+C(q,\dot{q})=Bu+J(q)^{T}\lambda_{ee},$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 7: [[动力学或策略机制]]

$$\ddot{y}_{cmd,i}=\ddot{y}_{des,i}(t)+K_{p}(y_{des,i}(t)-y)+K_{d}(\dot{y}_{des,i}(t)-\dot{y}).$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 8: [[动力学或策略机制]]

$$0\leq\lambda_{k}\perp Ex_{k}+F\lambda_{k}+Hu_{k}+c\geq 0,$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 9: [[动力学或策略机制]]

$$x_{N}^{T}Q_{f}x_{N}+\sum_{k=0}^{N-1}x_{k}^{T}Qx_{k}+u_{k}^{T}Ru_{k}$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 10: [[动力学或策略机制]]

$$0\leq\mu\lambda_{n}-\lambda_{t}\perp v\geq 0,$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。
## 关键图表

### Figure 1: Pipeline / core system figure: We examine a dynamic sliding task, where the robot uses the full spectrum of contact

![Figure 1](https://arxiv.org/html/2405.08731v2/extracted/5723606/figures/hardware/figure_1_transparent.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Pipeline / core system figure: We examine a dynamic sliding task, where the robot uses the full spectrum of contact”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 2: We abstract the system into two models. The LCS model captures the contact forces

![Figure 2](https://arxiv.org/html/2405.08731v2/extracted/5723606/figures/task_models.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“We abstract the system into two models. The LCS model captures the contact forces  ”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 3: System diagram for the hardware implementation. The different colored boxes indicate

![Figure 3](https://arxiv.org/html/2405.08731v2/extracted/5723606/figures/diagrams/franka_controller_diagram.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“System diagram for the hardware implementation. The different colored boxes indicate”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### 论文表格

#### Table 1: TABLE I: Target positions for tray retrieval task. Positions are specified as meters and in the robot/world frame where

| | Tray (m) | End Effector (m) | Idle Time (s) |
| --- | --- | --- | --- |
| Initial Position | [0.7, 0.0, 0.485] | [0.55, 0, 0.45] | |
| First Target | [0.45, 0, 0.485] | [0.45, 0, 0.47 | 0.5 |
| Second Target | [0.45, 0, 0.60] | [0.45, 0, 0.585] | 3.0 |
| Third Target | [0.7, 0, 0.485] | [0.6, 0, 0.47] | |

**说明**: TABLE I: Target positions for tray retrieval task. Positions are specified as meters and in the robot/world frame where the base of the robot is at the origin [0, 0, 0]. Idle time indicates how long the robot must remain at the target before the next target is given.

#### Table 2: TABLE II: Physical Parameters

| | Value |
| --- | --- |
| Tray Mass | 1 kg |
| Tray Radius | 0.228 m |
| Tray Thickness | 0.004 m |
| Tray Height (including raised rim) | 0.022 m |
| End Effector Mass | 0.37 kg |
| End Effector Radius | 0.0725 m |
| End Effector Thickness | 0.01 m |
| Tray/Support Friction Coefficient | 0.18 |
| Tray/End Effector Friction Coefficient | 0.5 |

**说明**: TABLE II: Physical Parameters

#### Table 3: TABLE III: Full C3 parameters used across all tray retrieval experiments

| N N N | 5 |
| --- | --- |
| d t d t dt | 0.075 |
| $\mu_{tray,ee},$ | 0.6 |
| $\mu_{tray,supports},$ | 0.1 |
|   \rho | 4 |
| ADMM iterations | 2 |
| Q q Q q Q_{q} | 50 * [150, 150, 150, 0, 1, 1, 0, 15000, 15000, 15000] |
| Q v Q v Q_{v} | 50 * [5, 5, 15, 10, 10, 1, 5, 5, 5] |
| R R R | 50 * [0.15, 0.15, 0.1] |
| w G x w G x w_{G_{x}} | 0.1 |
| w G  w G  w_{G_{\lambda}} | 10 |
| w G u w G u w_{G_{u}} | 0.1 |
| w U x w U x w_{U_{x}} | 0.1 |
| w U  w U  w_{U_{\lambda}} | 10 |
| w U u w U u w_{U_{u}} | 3 |
| u m i n u m i n u_{min} | [-10, -10, 0] |
| u m a x u m a x u_{max} | [10, 10, 30] |
| q e e, m i n q e e m i n q_{ee,min}, | [0.4, -0.1, 0.35] |
| q e e, m a x q e e m a x q_{ee,max}, | [0.6, 0.1, 0.7] |

**说明**: TABLE III: Full C3 parameters used across all tray retrieval experiments

#### Table 4: TABLE IV: Full C3 parameters used for rotating with external wall experiment

| N N N | 4 |
| --- | --- |
| d t d t dt | 0.05 |
| $\mu_{tray,ee},$ | 0.8 |
| $\mu_{tray,wall},$ | 1.0 |
|   \rho | 5 |
| ADMM iterations | 3 |
| Q q Q q Q_{q} | 50 * [10, 10, 150, 1000, 1000, 1000, 1000, 25, 25, 15000] |
| Q v Q v Q_{v} | 50 * [5, 5, 5, 1, 1, 500, 5, 5, 5] |
| R R R | 75 * [1.9, 0.5, 0.05] |
| w G x w G x w_{G_{x}} | 0.5 |
| w G  w G  w_{G_{\lambda}} | 75 |
| w G u w G u w_{G_{u}} | 1.25 |
| w U x w U x w_{U_{x}} | 0.5 |
| w U  w U  w_{U_{\lambda}} | 50 |
| w U u w U u w_{U_{u}} | 15 |
| u m i n u m i n u_{min} | [-10, -10, 0] |
| u m a x u m a x u_{max} | [10, 10, 30] |
| q e e, m i n q e e m i n q_{ee,min}, | [0.45, -0.2, 0.4] |
| q e e, m a x q e e m a x q_{ee,max}, | [0.7, 0.2, 0.5] |

**说明**: TABLE IV: Full C3 parameters used for rotating with external wall experiment
## 实验解读

- 评价重点:围绕 actuator-modeling、adaptive-control、contact-estimation、contact-implicit-optimization、接触推理,优先看真实机器人结果、扰动恢复、接触稳定性、控制误差、成功率、实时性和 sim-to-real 差距。
- System0 视角:实验结论不只看最终成功率,还要看闭环过程中是否能处理状态估计误差、执行延迟、碰撞/接触突变和外部扰动。
- 与上层系统的关系:如果方法能在快闭环内稳定工作,它可以作为 System1 技能或 System2 规划的底层执行保障;如果只在离线或慢速设定下成立,则还不能直接作为身体级快闭环模块。
## 批判性思考

### 优点

- 论文把 actuator-modeling、adaptive-control、contact-estimation、contact-implicit-optimization、接触推理 放到机器人身体级闭环里讨论,和只做高层语义推理的工作相比,更接近真实机器人部署时会遇到的状态、动力学和安全约束。
- 若论文同时给出公式、系统结构图、硬件实验或消融结果,这些证据可以帮助判断方法是否真的能支撑毫秒到百毫秒级反应。

### 局限

- 需要重点核查控制频率、推理延迟、传感延迟、失败恢复、硬件磨损、未见场景泛化和安全边界;这些因素决定它能否成为可靠的 System0 模块。
- 对学习型方法,还要区分训练环境中的鲁棒性和真实部署中的鲁棒性,尤其是接触丰富任务、动态运动任务和人形机器人任务。
## 论文定位

- 研究对象:Dynamic On-Palm Manipulation via Controlled Sliding。
- 关键词:actuator-modeling、adaptive-control、contact-estimation、contact-implicit-optimization、接触推理、接触丰富操作、grasping、模型预测控制、non-prehensile-manipulation、机器人操作。
- System0 关联:关注机器人身体级快反应链路中“状态/接触/局部感知 -> 约束/优化/策略 -> 动作”的短闭环,不把问题停留在语言规划或任务分解层。

## 速查卡片

> [!summary] Dynamic Palm Manipulation
> - **论文**: https://www.roboticsproceedings.org/rss20/p012.pdf
> - **arXiv**: http://arxiv.org/abs/2405.08731v2
> - **arXiv HTML**: https://arxiv.org/html/2405.08731v2
