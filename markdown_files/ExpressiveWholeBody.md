---
title: "Expressive Whole-Body Control for Humanoid Robots"
method_name: "Expressive Whole Body"
authors: ["Xuxin Cheng"]
year: 2024
venue: "RSS"
tags: ["robust-control", "legged-locomotion", "reinforcement-learning", "imitation-learning", "robot-generalization", "humanoid", "whole-body-control"]
image_source: "online"
arxiv_html: "https://arxiv.org/html/2402.16796v2"
---
# Expressive Whole Body
## 一句话总结

> Expressive Whole-Body Control for Humanoid Robots 主要落在 [[人形机器人]]、[[模仿学习]]、[[足式运动]]、[[运动模仿]] 这一类问题上;从 System0 视角看,它的价值在于把身体状态、接触变化、局部感知或动力学约束接到机器人低层闭环里,让动作生成更稳定、安全、实时。
## 核心贡献

1. 围绕 **Expressive Whole-Body Control for Humanoid Robots** 建立了一个与 人形机器人、模仿学习、足式运动、运动模仿、强化学习、鲁棒控制 相关的机器人问题设定,重点不是高层语义规划,而是身体级闭环中的状态变化、接触变化或控制约束。
2. 方法层面的关键价值在于把感知、动力学、约束优化、策略学习或硬件反馈组织成可执行的低层动作生成机制,适合放进 System0 的快反应模块中理解。
3. 对系统集成而言,这类工作补上了 System1 技能选择与电机级控制之间的一层:它负责把上层目标变成动态可行、可恢复、可安全过滤的动作。
4. 若论文包含真实机器人、接触任务或仿真到真实迁移,核心应看它是否报告了闭环稳定性、扰动恢复、硬件延迟、失败模式和跨场景泛化。
## 问题背景

这篇论文可以放在 System0 的“具身快系统”背景下阅读:机器人需要在很短时间内根据本体状态、接触、力/触觉、局部视觉或动力学预测做出动作,而不是等待慢速规划器重新推理整个任务。人形机器人、模仿学习、足式运动、运动模仿、强化学习、鲁棒控制、scalable-robot-learning 相关问题的难点通常不在“知道要做什么”,而在动作执行过程中状态会快速偏离、接触会突然改变、约束会实时收紧,低层系统必须及时修正。

### System0 关联

精读时优先看三件事:第一,论文使用哪些快速反馈量,例如关节状态、接触状态、力觉、触觉、深度或局部视觉;第二,反馈如何进入 MPC、WBC、策略网络、安全过滤器或恢复控制器;第三,输出是否直接影响机器人下一步动作,而不是只服务离线分析或高层任务规划。
## 方法详解

### System0 视角解析

从 System0 角度看,这篇论文的方法部分应当围绕 人形机器人、模仿学习、足式运动、运动模仿、强化学习、鲁棒控制、scalable-robot-learning、全身控制 来读。核心问题是:系统如何从快速变化的机器人状态中构造控制输入,如何处理接触、碰撞、约束、模型误差或执行延迟,以及如何把这些信息变成下一时刻的动作。

如果方法里有优化问题,重点看目标函数、约束和求解频率;如果方法里有学习策略,重点看观测空间、动作空间、训练扰动和部署时的闭环频率;如果方法里有硬件或传感器,重点看传感链路是否真的进入控制回路。这样读可以把论文和 System2 语言规划、System1 技能编排区分开来。

### 关键公式与机制

#### 公式 1: [[优化目标/约束]]

$$\mathcal{G}^{m}=\langle\mathbf{v},rpy,h\rangle$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 2: [[动力学或策略机制]]

$$s_{t}=[\omega_{t},r_{t},p_{t},\Delta y,q_{t},\dot{q}_{t},\mathbf{a}_{t-1}]^{T}$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 3: [[优化目标/约束]]

$$\mathcal{G}=\mathcal{G}^{e}\times\mathcal{G}^{m}$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 4: [[优化目标/约束]]

$$\mathcal{G}^{e}=\langle\mathbf{q},\mathbf{p}\rangle$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 5: [[动力学或策略机制]]

$$\mathbf{q^{i}_{m}}=(q_{x},q_{y},q_{z},q_{w})$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 6: [[动力学或策略机制]]

$$\mathbf{m}=[m_{1},m_{2},m_{3}]\in\mathbb{R}^{3}$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 7: [[动力学或策略机制]]

$$\mathbf{m}=\theta\mathbf{a}$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 8: [[动力学或策略机制]]

$$\theta=2\arccos(q_{w}),~{}\mathbf{a}=\frac{1}{\sqrt{1-q_{w}^{2}}}\begin{pmatrix}q_{x}\\ q_{y}\\ q_{z}\end{pmatrix}$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 9: [[动力学或策略机制]]

$$\mathbf{1}\left\{| $\textit{F}_{\textit{i}}^{\textit{z}}$ | \geq $\textit{F}_{\text{th}}\right\}*($ | $\textit{F}_{\textit{i}}^{\textit{z}}$ |-\textit{F}_{\text{th}})$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 10: [[动力学或策略机制]]

$$\mathbf{1}\left\{\exists i,\>| $\mathbf{F}_{\textit{i}}^{\textit{xy}}$ |>4| F_{ $\textit{i}}^{\textit{z}}$ |\right\}$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。
## 关键图表

### Figure 1: Overview of our framework. Our framework is able to train on data from various source

![Figure 1](https://arxiv.org/html/2402.16796v2/extracted/5451661/figures/method.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Overview of our framework. Our framework is able to train on data from various source”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 2: Random Sampling g m g m $\mathcal{\mathbf{g}}^{m}$

![Figure 2](https://arxiv.org/html/2402.16796v2/extracted/5451661/figures/comp_rand.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Random Sampling g m g m $\mathcal{\mathbf{g}}^{m}$”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 3: Text2Motion trajectories replay. A motion sequence is prompted offline with the input

![Figure 3](https://arxiv.org/html/2402.16796v2/x2.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Text2Motion trajectories replay. A motion sequence is prompted offline with the input”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### 论文表格

#### Table 1: TABLE I: Comparisons with physics-based character animation works. In PHC, the policy observes the Linear velocities an

| Metrics | Mimic WBC (Ours) | PHC [35 ] | ASE [46 ] |
| --- | --- | --- | --- |
| DoFs | 19 | 69 | 37 |
| Number of Motion Clips | 780 | 11000 | 187 |
| Total Time of Motions (h) | 3.7 | 40 | 0.5 |
| Real Robot | ✓ |  \times  |  \times  |
| Single Network | ✓ |  \times  | ✓ |
| Linear Velocities Obs |  \times  | ✓ | ✓ |
| Keypoint Positions Obs |  \times  | ✓ | ✓ |
| Robot Height Obs |  \times  |  \times  | ✓ |

**说明**: TABLE I: Comparisons with physics-based character animation works. In PHC, the policy observes the Linear velocities and keypoint positions of each rigid body, while in ASE linear velocities are for the root only. PHC and ASE both observe privileged states that are not available on the real robot.

#### Table 2: TABLE II: The details of our dataset. We select a subset from CMU MoCap dataset for training, and test on various expre

| | Category | Clips | Length (s) s (s) () |
| --- | --- | --- | --- |
| Training | Walk | 546 | 9076.6 |
| Dance | 78 | 1552.3 | |
| Basketball | 36 | 766.1 | |
| Punch | 20 | 800.0 | |
| Others | 100 | 1188.0 | |
| Total | 780 | 13383.0 | |
| Real-World Test | Punch | 1 | 18.9 |
| Wave Hello | 1 | 5.0 | |
| Mummy Walk | 1 | 22.5 | |
| Zombie Walk | 1 | 13.0 | |
| Walk, Exaggerated Stride | 1 | 2.5 | |
| High Five | 1 | 3.3 | |
| Basketball Signals | 1 | 32.6 | |
| Adjust Hair | 1 | 9.6 | |
| Drinking from Bottle | 1 | 15.2 | |
| Direct Traffic | 1 | 39.3 | |
| Hand Signal | 1 | 32.2 | |
| Russian Dance | 1 | 8.2 | |
| | Total | 11 | 202.3 |
| Additional Realworld Test (Diffusion [54 ]) | Boxing | 1 | 4.0 |
| Hug | 1 | 4.0 | |
| Shake Hands | 1 | 4.0 | |

**说明**: TABLE II: The details of our dataset. We select a subset from CMU MoCap dataset for training, and test on various expressive motions in sim and the real world.

#### Table 3: TABLE III: Expressive Rewards Specification

| Term | Expression | Weight |
| --- | --- | --- |
| Expression Goal G e G e G^{e} | | |
| DoF Position | exp (- 0.7 \| q ref - q \| (-0.7\| $\mathbf{q}_{\text{ref}}-\mathbf{q}\lVert (- 0.7 \rVert ref - \lVert)$ | 3.0 |
| Keypoint Position | exp (- \| p ref - p \| (-\| $\mathbf{p}_{\text{ref}}-\mathbf{p}\lVert (- \rVert ref - \lVert$ | 2.0 |
| Root Movement Goal G m G m G^{m} | | |
| Linear Velocity | exp (- 4.0 \| v ref - v \|) 4.0 v ref v (-4.0\| $\mathbf{v}_{\text{ref}}-\mathbf{v}\lVert) (- 4.0 \rVert ref - \lVert)$ | 6.0 |
| Roll & Pitch | $\boldsymbol{\Omega}_{\text{ref}}^{\phi\theta}-\boldsymbol{\Omega}^{\phi\theta}\lVert) (- \rVert ref - \lVert)$ | 1.0 |
| Yaw | exp (- \|  y \|)  y (-\|\Delta y\|) (- \| \|) | 1.0 |

**说明**: TABLE III: Expressive Rewards Specification

#### Table 4: TABLE IV: Comparisons with baselines. We sample 10,000 trajectories with 4096 environments in simulation and report the

| Baselines | Motion Sample | Random Sample | | | | | | |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MEL ↑ ↑ \uparrow ↑ | MELV ↑ ↑ \uparrow ↑ | MERP ↑ ↑ \uparrow ↑ | MEK ↑ ↑ \uparrow ↑ | MEL | MELV | MERP | MEK | |
| ExBody (Ours) | 16.87 | 318.67 | 754.92 | 659.78 | 13.51 | 132.14 | 523.79 | 483.67 |
| ExBody + AMP | 17.28 | 205.60 | 765.85 | 635.51 | 15.59 | 95.11 | 583.82 | 544.59 |
| ExBody + AMP NoReg | 16.16 | 87.83 | 714.74 | 561.56 | 15.40 | 36.76 | 584.23 | 515.53 |
| No RSI | 0.23 | 0.63 | 10.09 | 7.25 | 0.22 | 0.10 | 7.41 | 7.15 |
| Random Sample | 16.50 | 181.85 | 704.73 | 326.66 | 16.37 | 38.51 | 586.83 | 324.10 |
| Full Body Tracking | 13.28 | 246.11 | 584.40 | 397.25 | 10.76 | 76.46 | 407.88 | 284.69 |

**说明**: 该表格来自论文中的公式、奖励项或实验设置；为避免 PDF 抽取噪声，网页中仅保留可渲染的核心符号和数值。

#### Table 5: TABLE V: We report the mean absolute roll and pitch angle for a 10-second test in the real world for each motion.

| Motions | Ours | Ours+AMP |
| --- | --- | --- |
| Walk, Exaggerated Stride | 0.054 | 0.087 |
| Zombie Walk | 0.072 | 0.11 |
| Wave Hello | 0.062 | 0.095 |
| Walk Happily | 0.037 | 0.074 |
| Punch | 0.052 | 0.055 |
| Direct Traffic, Wave, Point | 0.037 | 0.094 |
| Highfive | 0.04 | 0.084 |
| Basketball Signals | 0.045 | 0.081 |
| Adjust Hair Walk | 0.042 | 0.09 |
| Russian Dance | 0.063 | 0.1 |
| Mummy Walk | 0.064 | 0.086 |
| Boxing | 0.075 | 0.068 |
| Hug | 0.037 | 0.086 |
| Shake Hand | 0.036 | 0.099 |
| Mean | 0.051 | 0.087 |

**说明**: TABLE V: We report the mean absolute roll and pitch angle for a 10-second test in the real world for each motion.

#### Table 6: TABLE VI: Regularization Rewards Specification

| Term | Expression | Weight |
| --- | --- | --- |
| Feet Related | | |
| Height | max(\| h feet \| - 0.2, 0 h feet 0.2 0 \| $\mathbf{h}_{\text{feet}}\lVert-0.2,0 \rVert feet \lVert - 0.2, 0)$ | 2.0 |
| Time in Air | ∑ t i air * 1 new contact t i air 1 new contact  $\sum t_{\textit{i}}^{\text{air}}*\mathbf{1}_{\text{new contact}} ∑ i air * new contact$ | 10.0 |
| Drag | ∑ \| v i foot \| * ∼ 1 new contact  $\sum\lVert \mathbf{v}_{i}^{\text{foot}} \rVert*\sim\mathbf{1}_{\text{new contact}} ∑ \lVert foot \rVert * ∼ new contact$ | -0.1 |
| Contact Force | 1 {\| F i z \| ≥ F th } * (\| F i z \| - F th) 1 F i z F th F i z F th  $\mathbf{1}\left\{\lVert \textit{F}_{\textit{i}}^{\textit{z}} \rVert\geq\textit{F}_{\text{th}}\right\}*(\lVert \textit{F}_{\textit{i}}^{\textit{z}} \rVert-\textit{F}_{\text{th}}) {\lVert F i z \rVert ≥ F th } * (\lVert F i z \rVert - F th)$ | -3e-3 |
| Stumble | 1 {∃ i, \| F i xy \| > 4 \| F i z \| } 1 i F i xy 4 F i z  $\mathbf{1}\left\{\exists i,\>\lVert \mathbf{F}_{\textit{i}}^{\textit{xy}} \rVert>4\lVert F_{\textit{i}}^{\textit{z}} \rVert\right\} {∃, \lVert i xy \rVert > 4 \lVert i z \rVert }$ | -2.0 |
| Other Items | | |
| DoF Acceleration | \| q  ̈ \| 2  ̈ q 2 \|\ddot{ $\mathbf{q}}\lVert^{2} \rVert \lVert 2$ | -3e-7 |
| Action Rate | \| a t - 1 - a t \| a t 1 a t \| $\mathbf{a}_{\textit{t}-1}-\mathbf{a}_{\textit{t}}\lVert \rVert t - 1 - t \lVert$ | -0.1 |
| Energy | \| q  ̈ \| 2  ̈ q 2 \|\ddot{ $\mathbf{q}}\lVert^{2} \rVert \lVert 2$ | -1e-3 |
| Collision | 1 collision 1 collision  $\mathbf{1}_{\text{collision}} collision$ | -0.1 |
| DoF Limit Violation | 1 q i > q max \| \| q i q_{ $\text{max}}\lVert \rVertq_{\textit{i}} max \lVert \rVert i < min$ | -10.0 |
| DoF Deviation | \| q default low - q low \| 2 q default low q low 2 \| $\mathbf{q}_{\text{default}}^{\text{low}}-\mathbf{q}^{\text{low}}\lVert^{2} \rVert default low - low \lVert 2$ | -10.0 |
| Vertical Linear Velocity | v z 2 v z 2 v_{ $\textit{z}}^{2} z 2$ | -1.0 |
| Horizontal Angular Velocity | $\boldsymbol{\omega}_{\textit{xy}}\lVert^{2} \rVert xy \lVert 2$ | -0.4 |
| Projected Gravity | \| g xy \| 2 g xy 2 \| $\mathbf{g}_{\textit{xy}}\lVert^{2} \rVert xy \lVert 2$ | -2.0 |

**说明**: TABLE VI: Regularization Rewards Specification

#### Table 7: TABLE VII: PPO hyperparameters.

**说明**: TABLE VII: PPO hyperparameters.

#### Table 8: TABLE VIII: AMP hyperparameters.

**说明**: TABLE VIII: AMP hyperparameters.
## 实验解读

- 评价重点:围绕 人形机器人、模仿学习、足式运动、运动模仿、强化学习,优先看真实机器人结果、扰动恢复、接触稳定性、控制误差、成功率、实时性和 sim-to-real 差距。
- System0 视角:实验结论不只看最终成功率,还要看闭环过程中是否能处理状态估计误差、执行延迟、碰撞/接触突变和外部扰动。
- 与上层系统的关系:如果方法能在快闭环内稳定工作,它可以作为 System1 技能或 System2 规划的底层执行保障;如果只在离线或慢速设定下成立,则还不能直接作为身体级快闭环模块。
## 批判性思考

### 优点

- 论文把 人形机器人、模仿学习、足式运动、运动模仿、强化学习 放到机器人身体级闭环里讨论,和只做高层语义推理的工作相比,更接近真实机器人部署时会遇到的状态、动力学和安全约束。
- 若论文同时给出公式、系统结构图、硬件实验或消融结果,这些证据可以帮助判断方法是否真的能支撑毫秒到百毫秒级反应。

### 局限

- 需要重点核查控制频率、推理延迟、传感延迟、失败恢复、硬件磨损、未见场景泛化和安全边界;这些因素决定它能否成为可靠的 System0 模块。
- 对学习型方法,还要区分训练环境中的鲁棒性和真实部署中的鲁棒性,尤其是接触丰富任务、动态运动任务和人形机器人任务。
## 论文定位

- 研究对象:Expressive Whole-Body Control for Humanoid Robots。
- 关键词:人形机器人、模仿学习、足式运动、运动模仿、强化学习、鲁棒控制、scalable-robot-learning、全身控制。
- System0 关联:关注机器人身体级快反应链路中“状态/接触/局部感知 -> 约束/优化/策略 -> 动作”的短闭环,不把问题停留在语言规划或任务分解层。

## 速查卡片

> [!summary] Expressive Whole Body
> - **论文**: https://www.roboticsproceedings.org/rss20/p107.pdf
> - **arXiv**: http://arxiv.org/abs/2402.16796v2
> - **arXiv HTML**: https://arxiv.org/html/2402.16796v2
