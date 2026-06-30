---
title: "A Unified and General Humanoid Whole-Body Controller for Fine-Grained Locomotion"
method_name: "Unified General Humanoid"
authors: ["Yufei Xue"]
year: 2025
venue: "RSS"
tags: ["robot-manipulation", "robust-control", "real-time-control", "legged-locomotion", "imitation-learning", "humanoid", "agile-locomotion", "loco-manipulation", "whole-body-control"]
image_source: "online"
arxiv_html: "https://arxiv.org/html/2502.03206"
---
# Unified General Humanoid
## 一句话总结

> A Unified and General Humanoid Whole-Body Controller for Fine-Grained Locomotion 主要落在 [[agile-locomotion]]、[[人形机器人]]、[[足式运动]]、[[移动操作]] 这一类问题上;从 System0 视角看,它的价值在于把身体状态、接触变化、局部感知或动力学约束接到机器人低层闭环里,让动作生成更稳定、安全、实时。
## 核心贡献

1. 围绕 **A Unified and General Humanoid Whole-Body Controller for Fine-Grained Locomotion** 建立了一个与 agile-locomotion、人形机器人、足式运动、移动操作、实时控制、机器人操作 相关的机器人问题设定,重点不是高层语义规划,而是身体级闭环中的状态变化、接触变化或控制约束。
2. 方法层面的关键价值在于把感知、动力学、约束优化、策略学习或硬件反馈组织成可执行的低层动作生成机制,适合放进 System0 的快反应模块中理解。
3. 对系统集成而言,这类工作补上了 System1 技能选择与电机级控制之间的一层:它负责把上层目标变成动态可行、可恢复、可安全过滤的动作。
4. 若论文包含真实机器人、接触任务或仿真到真实迁移,核心应看它是否报告了闭环稳定性、扰动恢复、硬件延迟、失败模式和跨场景泛化。
## 问题背景

这篇论文可以放在 System0 的“具身快系统”背景下阅读:机器人需要在很短时间内根据本体状态、接触、力/触觉、局部视觉或动力学预测做出动作,而不是等待慢速规划器重新推理整个任务。agile-locomotion、人形机器人、足式运动、移动操作、实时控制、机器人操作、鲁棒控制 相关问题的难点通常不在“知道要做什么”,而在动作执行过程中状态会快速偏离、接触会突然改变、约束会实时收紧,低层系统必须及时修正。

### System0 关联

精读时优先看三件事:第一,论文使用哪些快速反馈量,例如关节状态、接触状态、力觉、触觉、深度或局部视觉;第二,反馈如何进入 MPC、WBC、策略网络、安全过滤器或恢复控制器;第三,输出是否直接影响机器人下一步动作,而不是只服务离线分析或高层任务规划。
## 方法详解

### System0 视角解析

从 System0 角度看,这篇论文的方法部分应当围绕 agile-locomotion、人形机器人、足式运动、移动操作、实时控制、机器人操作、鲁棒控制、遥操作 来读。核心问题是:系统如何从快速变化的机器人状态中构造控制输入,如何处理接触、碰撞、约束、模型误差或执行延迟,以及如何把这些信息变成下一时刻的动作。

如果方法里有优化问题,重点看目标函数、约束和求解频率;如果方法里有学习策略,重点看观测空间、动作空间、训练扰动和部署时的闭环频率;如果方法里有硬件或传感器,重点看传感链路是否真的进入控制回路。这样读可以把论文和 System2 语言规划、System1 技能编排区分开来。

### 关键公式与机制

#### 公式 1: [[优化目标/约束]]

$$\pi^{*}=\mathop{\arg\max}_{\pi}\mathbb{E}_{\pi}\left[\sum_{t=0}^{\infty}\gamma^{t}r(o_{t},a_{t},c_{t})\right]$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 2: [[动力学或策略机制]]

$$=\left\{\begin{aligned} &0.5\times\frac{\phi_{i}}{\phi_{\text{stance}}},&\phi_{i}<\phi_{\text{stance}}\\ &0.5+0.5\times\frac{\phi_{i}-\phi_{\text{stance}}}{1-\phi_{\text{stance}}},&\phi_{i}\geq\phi_{\text{stance}}\end{aligned}\right.~{},$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 3: [[动力学或策略机制]]

$$l_{t}^{\text{target},i}=\left\{\begin{aligned} &l_{t}\sum_{k=0}^{5}a_{k}\left(0.25-\left| \bar{\phi}_{t,i}-0.75 $\right$ |\right)^{k},&\bar{\phi}_{t,i}>0.5\\ &0,&\bar{\phi}_{t,i}<0.5\end{aligned}\right.~{}.$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 4: [[优化目标/约束]]

$$
\mathcal{L}_{\text{sym}}=\sum_{t}\left\lVert \pi\left({o^{\pi}_{t}}\right)-\mathcal{F}_{a}\left(\pi\left(\mathcal{F}_{o}\left({o^{\pi}_{t}}\right)\right)\right)\right \rVert^{2}~{},
$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 5: [[动力学或策略机制]]

$$\begin{aligned} l_{t}^{\text{target},i}=\left\{\begin{aligned} &\frac{6(l_{t}-p_{s,z})}{(\bar{\phi}_{i}^{0.75}-\bar{\phi}_{i}^{0.5})^{5}}(\bar{\phi_{i}}-0.5)^{5}+\frac{15(p_{s,z}-l_{t})}{(\bar{\phi}_{i}^{0.75}-\bar{\phi}_{i}^{0.5})^{4}}(\bar{\phi_{i}}-0.5)^{4}+\frac{10(p_{s,z}-l_{t})}{(\bar{\phi}_{i}^{0.75}-\bar{\phi}_{i}^{0.5})^{3}}(\bar{\phi_{i}}-0.5)^{3}+p_{s,z},&0.5<\phi_{i}<0.75\\ &\frac{6(p_{e,z}-l_{t})}{(\bar{\phi}_{i}^{1.0}-\bar{\phi}_{i}^{0.75})^{5}}(1-\bar{\phi_{i}})^{5}+\frac{15(l_{t}-p_{e,z})}{(\bar{\phi}_{i}^{1.0}-\bar{\phi}_{i}^{0.75})^{4}}(1-\bar{\phi_{i}})^{4}+\frac{10(l_{t}-p_{e,z})}{(\bar{\phi}_{i}^{1,0}-\bar{\phi}_{i}^{0.75})^{3}}(1-\bar{\phi_{i}})^{3}+l_{t},&0.75<\phi_{i}<1.0\end{aligned}\right.~{},\end{aligned}$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 6: [[动力学或策略机制]]

$$V^{\text{targ}}(o,c)=\mathbb{E}_{\pi}\left[\sum_{t}\gamma^{t}r(o_{t},a_{t},c_{t})|o_{0}=o,c_{0}=c\right]$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 7: [[优化目标/约束]]

$$
\mathcal{L}_{\text{value}}=\mathbb{E}_{\pi}\left[\lVert V_{\pi}(o,c)-V^{\text{targ}}(o,c)\rVert^{2}\right]~{},
$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 8: [[优化目标/约束]]

$$
\mathcal{L}_{\text{est}}=\mathbb{E}_{\pi}\left[\lVert \mathcal{E}_{\pi}(h^{k})-s^{\text{key}} \rVert^{2}\right]
$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 9: [[优化目标/约束]]

$$
\mathcal{L}_{\text{policy}}=\mathbb{E}_{\pi}[\min(rA,\text{clip}(r,1-\epsilon,1+\epsilon)A)],
$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 10: [[优化目标/约束]]

$$\mathcal{L}^{\text{AAC}}=\mathcal{L}^{\text{value}}+\lambda^{\text{policy}}\mathcal{L}^{\text{policy}}+\lambda^{\text{est}}\mathcal{L}^{\text{est}}$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。
## 关键图表

### Figure 1: Framework of HugHBC. Illustration with the Unitree H1 robot. a): Visualization of pa

![Figure 1](https://arxiv.org/html/2502.03206/extracted/6354275/imgs/FrameworkV6.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Framework of HugHBC. Illustration with the Unitree H1 robot. a): Visualization of pa”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 2: The expected contact probability function C ( t, i) C t i C

![Figure 2](https://arxiv.org/html/2502.03206/x1.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“The expected contact probability function C ( t, i) C t i C”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 3: External disturbance tolerance. Left: A constant and continuous force is applied to

![Figure 3](https://arxiv.org/html/2502.03206/x3.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“External disturbance tolerance. Left: A constant and continuous force is applied to”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### 论文表格

#### Table 1: TABLE I: Reward definitions used in HugWBC.

| Term | Definition | Weight |
| --- | --- | --- |
| Task Reward | | |
| Linear Velocity Tracking | $\exp\left(-\left\lVert v_{\text{xy}}^{\text{target}}-v_{\text{xy}}\right \rVert^{2}/0.2\right)$ | 2 |
| Angular Velocity Tracking | $\exp\left(-\left\lVert \omega_{\text{z}}^{\text{target}}-\omega_{\text{z}}\right \rVert^{2}/0.2\right)$ | 2 |
| Behavior Reward | | |
| Body Height Tracking | $$ | -40 |
| Body Pitch Tracking | $$ | -10 |
| Waist Yaw Tracking | $$ | -2 |
| Foot Swing Tracking | $$ | -30 |
| Contact-Swing Tracking | $$ | -2 |
| Regularization Reward | | |
| R-P Angular Velocity | $$ | -0.5 |
| Vertical Body Movement | $$ | -0.1 |
| Feet Slip | $\sum_{i}\exp\left(-\left\lVert v_{\text{xy}}^{\text{foot,i}}\right \rVert^{2}\right) 1 - ∑$ | -0.2 |
| Action Rate | $$ | -0.01 |
| Action Smoothness | $$ | -0.01 |
| Joint Torque | $$ | -5e-6 |
| Joint Acceleration | $$ | -2.5e-7 |
| Upper Joint Deviation | q upper - q upper nominal q upper q nominal upper $$ | -0.5 |
| Hip Joint Deviation | q hip xz - q hip xz nominal q hip xz q nominal hip xz $$ | -2 |
| Feet Symmetry | $$ | -5 |
| Termination | 1 [Early Terminate ] 1 Early Terminate  $\mathbf{1}[\text{Early Terminate}] [Early Terminate ]$ | -200 |

**说明**: TABLE I: Reward definitions used in HugWBC.

#### Table 2: TABLE II: Command ranges. Ranges of curriculum starting, finishing, and default values of commands, for all gaits excep

| Group | Term | Default | Initial | |
| --- | --- | --- | --- | --- |
| Range | | | | |
| Finishing | | | | |
| Range | | | | |
| Task Commands | v x v x v_{x} | 0 | [- 0.6, 0.6 ] 0.6 0.6 [-0.6,0.6] [- 0.6, 0.6 ] | [- 0.6, 2.0 ] 0.6 2.0 [-0.6,2.0] [- 0.6, 2.0 ] |
| v y v y v_{y} | 0 | [- 0.6, 0.6 ] 0.6 0.6 [-0.6,0.6] [- 0.6, 0.6 ] | [- 0.6, 0.6 ] 0.6 0.6 [-0.6,0.6] [- 0.6, 0.6 ] | |
| $\omega$ | 0 | [- 0.6, 0.6 ] 0.6 0.6 [-0.6,0.6] [- 0.6, 0.6 ] | [- 1.0, 1.0 ] 1.0 1.0 [-1.0,1.0] [- 1.0, 1.0 ] | |
| Behavior Commands | f f f | 2 | [1.5, 3.5 ] 1.5 3.5 [1.5,3.5] [1.5, 3.5 ] | |
| l l l | 0.15 | [0.1, 0.35 ] 0.1 0.35 [0.1,0.35] [0.1, 0.35 ] | | |
| h h h | 0 | [- 0.3, 0 ] 0.3 0 [-0.3,0] [- 0.3, 0 ] | | |
| p p p | 0 | [0, 0.4 ] 0 0.4 [0,0.4] [0, 0.4 ] | | |
| w w w | 0 | [- 1.0, 1.0 ] 1.0 1.0 [-1.0,1.0] [- 1.0, 1.0 ] | | |

**说明**: TABLE II: Command ranges. Ranges of curriculum starting, finishing, and default values of commands, for all gaits except hopping.

#### Table 3: TABLE III: Single command tracking error. The tracking errors for foot commands are calculated a complete gait cyc

| Gait | Movement | Foot | Posture | | | | | | |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E v x low E v x low E_{v_{x}}^{ $\text{low}} low (m / s m s m/s /)$ | E v x high E v x high E_{v_{x}}^{ $\text{high}} high (m / s m s m/s /)$ | E v y E v y E_{v_{y}} (m / s m s m/s /) | $\omega} r a d / s r a d s rad/s /$ | E f E f E_{f} (H Z H Z HZ) | E l E l E_{l} (m m m) | E h E h E_{h} (m m m) | E p E p E_{p} (r a d r a d rad) | E w E w E_{w} (r a d r a d rad) | |
| Standing | - | - | - | - | - | - | 0.035 | 0.047 | 0.022 |
| Walking | 0.030 | 0.216 | 0.085 | 0.054 | 0.028 | 0.011 | 0.064 | 0.038 | 0.075 |
| Jumping | 0.090 | 0.532 | 0.069 | 0.077 | 0.027 | 0.012 | 0.058 | 0.048 | 0.022 |
| Hopping | 0.033 | - | 0.046 | 0.078 | - | - | 0.103 | - | - |

**说明**: 该表格来自论文中的公式、奖励项或实验设置；为避免 PDF 抽取噪声，网页中仅保留可渲染的核心符号和数值。

#### Table 4: TABLE IV: Tracking errors with different intervention strategies the walking gait. We evaluate three upper-body

| Training Strategy | Intervention Task | Task Commands | Behavior Commands | | | | | | |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Movement | Foot | Posture | | | | | | | |
| E v x E v x E_{v_{x}} (m / s m s m/s /) | E v y E v y E_{v_{y}} (m / s m s m/s /) | $\omega} (r a d / s r a d s rad/s /)$ | E f E f E_{f} (H z H z Hz) | E l E l E_{l} (m m m) | E h E h E_{h} (m m m) | E p E p E_{p} (r a d r a d rad) | E w E w E_{w} (r a d r a d rad) | | |
| Noise Curriculum (HugWBC) | Noise | 0.0483 | 0.0962 | 0.1879 | 0.0471 | 0.0542 | 0.0402 | 0.0432 | 0.0552 |
| AMASS | 0.0391 | 0.0920 | 0.1039 | 0.0464 | 0.0543 | 0.0387 | 0.0364 | 0.0540 | |
| None | 0.0264 | 0.0863 | 0.0543 | 0.0447 | 0.0522 | 0.0372 | 0.0375 | 0.0475 | |
| AMASS | Noise | 0.1697 | 0.1055 | 0.2156 | 0.0621 | 0.0542 | 0.0620 | 0.0812 | 0.0694 |
| AMASS | 0.0567 | 0.0965 | 0.1593 | 0.0466 | 0.0555 | 0.0579 | 0.0458 | 0.0554 | |
| None | 0.0645 | 0.0916 | 0.0802 | 0.0460 | 0.0531 | 0.0577 | 0.0455 | 0.0568 | |
| No Intervention | Noise | 0.8658 | 0.7511 | 0.9116 | 0.1930 | 0.1913 | 0.1658 | 0.3622 | 0.2241 |
| AMASS | 0.6299 | 0.4026 | 0.5758 | 0.2245 | 0.2527 | 0.1305 | 0.2367 | 0.1112 | |
| None | 0.0755 | 0.1076 | 0.1151 | 0.0450 | 0.0678 | 0.0255 | 0.0211 | 0.0380 | |

**说明**: TABLE IV: Tracking errors with different intervention strategies the walking gait. We evaluate three upper-body intervention training strategies: Noise (HugWBC), the AMASS dataset, and no intervention at all. The tracking errors across various task and behavior commands reflect the intervention tolerance, i.e., the ability of precise locomotion control external intervention.

#### Table 5: TABLE V: Averaged foot displacement intervention. We compare foot displacement D cmd D cmd D_{$\text${cm

| Training Strategy | Intervention Task | D h D h D_{h} (m / s m s m/s /) | D p D p D_{p} (m / s m s m/s /) | D w D w D_{w} (m / s m s m/s /) |
| --- | --- | --- | --- | --- |
| Noise Curriculum (HugWBC) | Noise | 0.0339 | 0.0892 | 0.0199 |
| AMASS | 0.0454 | 0.0728 | 0.0196 | |
| None | 0.0003 | 0.0016 | 0.0007 | |
| AMASS only | Noise | 2.0815 | 2.8978 | 3.2630 |
| AMASS | 0.0536 | 0.1743 | 0.0396 | |
| None | 0.0139 | 0.0160 | 0.0013 | |
| No Intervention | Noise | 17.5358 | 17.9732 | 25.7132 |
| AMASS | 25.3802 | 26.3496 | 21.3078 | |
| None | 0.0159 | 1.7065 | 1.7152 | |

**说明**: 该表格来自论文中的公式、奖励项或实验设置；为避免 PDF 抽取噪声，网页中仅保留可渲染的核心符号和数值。

#### Table 6: TABLE VI: Tracking error in real world. We conducted five tests to measure the tracking error for each command th

| Gait | E p real E p real E_{p}^{ $\text{real}} real$ | E w real E w real E_{w}^{ $\text{real}} real$ |
| --- | --- | --- |
| Standing | 0.0712 $\pm$ 0.0425 | 0.0718 $\pm$ 0.0614 |
| Walking | 0.1006 $\pm$ 0.0581 | 0.0571 $\pm$ 0.0489 |
| Jumping | 0.0674 $\pm$ 0.0569 | 0.0552 $\pm$ 0.0469 |

**说明**: TABLE VI: Tracking error in real world. We conducted five tests to measure the tracking error for each command three gaits. The tracking error for each command was calculated during each control step. The tested commands gradually increased from the minimum to the maximum values within a predefined range, while the remaining commands were kept at their default values.

#### Table 7: TABLE VII: Ranges and default values of commands for gait hopping.

| Group | Term | Default | Range |
| --- | --- | --- | --- |
| Movement | linear velocity v x v x v_{x} | 0 | [- 0.6, 0.6 ] 0.6 0.6 [-0.6,0.6] [- 0.6, 0.6 ] |
| linear velocity v y v y v_{y} | 0 | [- 0.6, 0.6 ] 0.6 0.6 [-0.6,0.6] [- 0.6, 0.6 ] | |
| angular velocity $\omega$ | 0 | [- 0.6, 0.6 ] 0.6 0.6 [-0.6,0.6] [- 0.6, 0.6 ] | |
| Posture | body height h h h | 0 | [- 0.3, 0 ] 0.3 0 [-0.3,0] [- 0.3, 0 ] |

**说明**: TABLE VII: Ranges and default values of commands for gait hopping.

#### Table 8: TABLE VIII: Boundary conditions for z z z -direction quintic polynomial trajectory.

| Time | Position | Velocity | Acceleration |
| --- | --- | --- | --- |
| t=  i 0.5  ̄  ̄ i 0.5 \bar{\phi_{i}^{0.5}} 0.5 | p s, z p s z p_{s,z}, | 0 | 0 |
| t=  i 0.75  ̄  ̄ i 0.75 \bar{\phi_{i}^{0.75}} 0.75 | l t l t l_{t} | 0 | 0 |
| t=  i 1.0  ̄  ̄ i 1.0 \bar{\phi_{i}^{1.0}} 1.0 | p e, z p e z p_{e,z}, | 0 | 0 |

**说明**: TABLE VIII: Boundary conditions for z z z -direction quintic polynomial trajectory.

#### Table 9: TABLE IX: Network architectures.

| Module | Inputs | Hidden Layers | Outputs |
| --- | --- | --- | --- |
| Historical State Encoder | o t his o t his o_{t}^{ $\text{his}} his$ | [256, 128] | z t z t z_{t} |
| State Estimator | z t z t z_{t} | [64, 32] | v t ^, l t ^, h t ^ ^ v t ^ l t ^ h t  $\hat{v_{t}},\hat{l_{t}},\hat{h_{t}},,$ |
| Low-Level Network | z t, v t ^, l t ^, h t ^ z t ^ v t ^ l t ^ h t z_{t}, $\hat{v_{t}},\hat{l_{t}},\hat{h_{t}},,,, o t pro, c t, I (t) o t pro c t I t o_{t}^{\text{pro}},c_{t},I(t) pro,, ()$ | [256, 128, 64] | a t a t a_{t} |
| Critic | o t pro, o t pri, o t ter o t pro o t pri o t ter o_{t}^{ $\text{pro}},o_{t}^{\text{pri}},o_{t}^{\text{ter}} pro, pri, ter$ | [512, 256, 128] | V t V t V_{t} |

**说明**: TABLE IX: Network architectures.

#### Table 10: TABLE X: Tracking error with different intervention strategies the standing gait and the jumping gait. We evalua

**说明**: TABLE X: Tracking error with different intervention strategies the standing gait and the jumping gait. We evaluate three upper-body intervention training strategies: noise curriculum (HUGWBC), the AMASS dataset, and no intervention at all. The tracking errors across various tasks and behavior commands reflect the intervention tolerance, i.e., the ability of precise locomotion control external intervention.

#### Table 11: TABLE XI: Single command tracking error comparison with learning based baselines.

| Methods | E v x low E v x low E_{v_{x}}^{ $\text{low}} low$ | E v x high E v x high E_{v_{x}}^{ $\text{high}} high$ | E v y E v y E_{v_{y}} | $\omega}$ | E h E h E_{h} | E p E p E_{p} | E w E w E_{w} |
| --- | --- | --- | --- | --- | --- | --- | --- |
| HOVER [20 ] | 0.559 | 1.324 | 0.328 | 0.436 | 0.270 | 0.127 | 0.082 |
| ExBody [3 ] | 0.109 | 0.242 | 0.114 | 0.587 | 0.145 | 0.122 | 0.097 |
| HugWBC (Ours) | 0.030 | 0.216 | 0.085 | 0.054 | 0.064 | 0.038 | 0.075 |

**说明**: TABLE XI: Single command tracking error comparison with learning based baselines.
## 实验解读

- 评价重点:围绕 agile-locomotion、人形机器人、足式运动、移动操作、实时控制,优先看真实机器人结果、扰动恢复、接触稳定性、控制误差、成功率、实时性和 sim-to-real 差距。
- System0 视角:实验结论不只看最终成功率,还要看闭环过程中是否能处理状态估计误差、执行延迟、碰撞/接触突变和外部扰动。
- 与上层系统的关系:如果方法能在快闭环内稳定工作,它可以作为 System1 技能或 System2 规划的底层执行保障;如果只在离线或慢速设定下成立,则还不能直接作为身体级快闭环模块。
## 批判性思考

### 优点

- 论文把 agile-locomotion、人形机器人、足式运动、移动操作、实时控制 放到机器人身体级闭环里讨论,和只做高层语义推理的工作相比,更接近真实机器人部署时会遇到的状态、动力学和安全约束。
- 若论文同时给出公式、系统结构图、硬件实验或消融结果,这些证据可以帮助判断方法是否真的能支撑毫秒到百毫秒级反应。

### 局限

- 需要重点核查控制频率、推理延迟、传感延迟、失败恢复、硬件磨损、未见场景泛化和安全边界;这些因素决定它能否成为可靠的 System0 模块。
- 对学习型方法,还要区分训练环境中的鲁棒性和真实部署中的鲁棒性,尤其是接触丰富任务、动态运动任务和人形机器人任务。
## 论文定位

- 研究对象:A Unified and General Humanoid Whole-Body Controller for Fine-Grained Locomotion。
- 关键词:agile-locomotion、人形机器人、足式运动、移动操作、实时控制、机器人操作、鲁棒控制、遥操作、全身控制。
- System0 关联:关注机器人身体级快反应链路中“状态/接触/局部感知 -> 约束/优化/策略 -> 动作”的短闭环,不把问题停留在语言规划或任务分解层。

## 速查卡片

> [!summary] Unified General Humanoid
> - **论文**: https://www.roboticsproceedings.org/rss21/p067.pdf
> - **arXiv**: http://arxiv.org/abs/2502.03206
> - **arXiv HTML**: https://arxiv.org/html/2502.03206
