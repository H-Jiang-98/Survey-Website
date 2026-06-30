---
title: "LangWBC: Language-directed Humanoid Whole-Body Control via End-to-end Learning"
method_name: "LangWBC"
authors: ["Yiyang Shao"]
year: 2025
venue: "RSS"
tags: ["robust-control", "legged-locomotion", "reinforcement-learning", "adaptive-control", "robot-generalization", "humanoid", "agile-locomotion", "whole-body-control"]
image_source: "online"
arxiv_html: "https://arxiv.org/html/2504.21738v1"
---
# LangWBC
## 一句话总结

> LangWBC: Language-directed Humanoid Whole-Body Control via End-to-end Learning 主要落在 [[adaptive-control]]、[[agile-locomotion]]、[[cvae]]、[[人形机器人]] 这一类问题上;从 System0 视角看,它的价值在于把身体状态、接触变化、局部感知或动力学约束接到机器人低层闭环里,让动作生成更稳定、安全、实时。
## 核心贡献

1. 围绕 **LangWBC: Language-directed Humanoid Whole-Body Control via End-to-end Learning** 建立了一个与 adaptive-control、agile-locomotion、cvae、人形机器人、language-conditioned-policy、足式运动 相关的机器人问题设定,重点不是高层语义规划,而是身体级闭环中的状态变化、接触变化或控制约束。
2. 方法层面的关键价值在于把感知、动力学、约束优化、策略学习或硬件反馈组织成可执行的低层动作生成机制,适合放进 System0 的快反应模块中理解。
3. 对系统集成而言,这类工作补上了 System1 技能选择与电机级控制之间的一层:它负责把上层目标变成动态可行、可恢复、可安全过滤的动作。
4. 若论文包含真实机器人、接触任务或仿真到真实迁移,核心应看它是否报告了闭环稳定性、扰动恢复、硬件延迟、失败模式和跨场景泛化。
## 问题背景

这篇论文可以放在 System0 的“具身快系统”背景下阅读:机器人需要在很短时间内根据本体状态、接触、力/触觉、局部视觉或动力学预测做出动作,而不是等待慢速规划器重新推理整个任务。adaptive-control、agile-locomotion、cvae、人形机器人、language-conditioned-policy、足式运动、强化学习 相关问题的难点通常不在“知道要做什么”,而在动作执行过程中状态会快速偏离、接触会突然改变、约束会实时收紧,低层系统必须及时修正。

### System0 关联

精读时优先看三件事:第一,论文使用哪些快速反馈量,例如关节状态、接触状态、力觉、触觉、深度或局部视觉;第二,反馈如何进入 MPC、WBC、策略网络、安全过滤器或恢复控制器;第三,输出是否直接影响机器人下一步动作,而不是只服务离线分析或高层任务规划。
## 方法详解

### System0 视角解析

从 System0 角度看,这篇论文的方法部分应当围绕 adaptive-control、agile-locomotion、cvae、人形机器人、language-conditioned-policy、足式运动、强化学习、robot-generalization 来读。核心问题是:系统如何从快速变化的机器人状态中构造控制输入,如何处理接触、碰撞、约束、模型误差或执行延迟,以及如何把这些信息变成下一时刻的动作。

如果方法里有优化问题,重点看目标函数、约束和求解频率;如果方法里有学习策略,重点看观测空间、动作空间、训练扰动和部署时的闭环频率;如果方法里有硬件或传感器,重点看传感链路是否真的进入控制回路。这样读可以把论文和 System2 语言规划、System1 技能编排区分开来。

### 关键公式与机制

#### 公式 1: [[动力学或策略机制]]

$$\sum|| $\tau_{i}\cdot\dot{\theta}_{i}$ ||^{2}$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 2: [[优化目标/约束]]

$$\mathbb{I}_{\theta_{i}\notin[\theta_{\min},\theta_{\max}]}$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 3: [[优化目标/约束]]

$${q}_{\min}\leq{q}_{t}\leq{q}_{\max},\quad\forall\,t=1,\dots,T.$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 4: [[优化目标/约束]]

$$\mathcal{L}_{\text{sym}}=\mathbb{E}_{s_{t}\sim\mathcal{D}}\left[|| $\pi^{\text{teacher}}(s_{t})-\mathcal{M}(\pi^{\text{teacher}}(s_{t}^{m}))$ ||^{2}\right].$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 5: [[优化目标/约束]]

$$
\mathcal{L}_{\text{student}}=\lVert a^{\text{T}}_{t}-a^{\text{S}}_{t} \rVert_{2}^{2}+\lambda_{\text{KL}}\,D_{\text{KL}}(q_{\phi}(z_{t}|o_{t-20:t},v_{t}^{\text{text}})\lVert p(z_{t})),
$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 6: [[动力学或策略机制]]

$${f}_{k}=\begin{bmatrix}{x}_{\text{robot},1}({q}_{1}^{(k)})-{x}_{\text{mocap},1}\\ \sqrt{w_{\text{ori}}}\Delta{r}_{1}({q}_{1}^{(k)})\\ \vdots\\ {x}_{\text{robot},T}({q}_{T}^{(k)})-{x}_{\text{mocap},T}\\ \sqrt{w_{\text{ori}}}\Delta{r}_{T}({q}_{T}^{(k)})\\ \sqrt{w_{\text{smooth}}}({q}_{2}^{(k)}-{q}_{1}^{(k)})\\ \sqrt{w_{\text{smooth}}}({q}_{3}^{(k)}-{q}_{2}^{(k)})\\ \vdots\\ \sqrt{w_{\text{smooth}}}({q}_{T}^{(k)}-{q}_{T-1}^{(k)})\end{bmatrix}$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 7: [[动力学或策略机制]]

$$+w_{\text{ori}}\left\lVert \Delta{r}_{t}({q}_{t})\right \rVert^{2}\Bigr)+w_{\text{smooth}}\sum_{t=2}^{T}\left\lVert {q}_{t}-{q}_{t-1}\right \rVert^{2},$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 8: [[动力学或策略机制]]

$$\left({J}_{k}^{\top}{J}_{k}+\lambda{I}+w_{\text{smooth}}{S}^{\top}{S}\right)\Delta{q}=-{J}_{k}^{\top}{f}_{k}.$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 9: [[优化目标/约束]]

$$
\mathcal{L}_{\text{teacher}}=\mathcal{L}_{\text{PPO}}+\lambda_{\text{sym}}\mathcal{L}_{\text{sym}},
$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 10: [[动力学或策略机制]]

$$\sum_{t=1}^{T}\Bigl(\left\lVert {x}_{\text{robot},t}({q}_{t})-{x}_{\text{mocap},t}\right \rVert^{2}$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。
## 关键图表

### Figure 1: The Overview of the Training Framework. The training process includes a motion-trackin

![Figure 1](https://arxiv.org/html/2504.21738v1/x1.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“The Overview of the Training Framework. The training process includes a motion-trackin”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 2: Real World Demonstration. Conditioned on text commands, our framework is able to learn

![Figure 2](https://arxiv.org/html/2504.21738v1/x3.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Real World Demonstration. Conditioned on text commands, our framework is able to learn”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 3: Upper-body Motion Examples. Our framework generates diverse upper-body movements incl

![Figure 3](https://arxiv.org/html/2504.21738v1/extracted/6402247/fig/real2.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Upper-body Motion Examples. Our framework generates diverse upper-body movements incl”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### 论文表格

#### Table 1: TABLE I: Reward Function Components for Teacher Policy

| Term | Expression | Weight |
| --- | --- | --- |
| Z linear vel penalty | $\text{root}}}\lVert \rVert^{2} \lVert \rVert root \lVert \rVert 2$ | - 0.2 0.2 -0.2 - 0.2 |
| XY angular vel penalty | $\omega_{xy}^{\text{root}}}\lVert \rVert^{2} \lVert \rVert root \lVert \rVert 2$ | - 0.05 0.05 -0.05 - 0.05 |
| Joint torque penalty | $\sum\tau_{i}^{2} ∑ 2$ | - 2  10 - 6 2 10 6 -2\!\times\!10^{-6} - 2  10 - 6 |
| Joint acc penalty | $\sum\ddot{\theta}_{i}^{2} ∑ 2$ | - 1  10 - 7 1 10 7 -1\!\times\!10^{-7} - 1  10 - 7 |
| Joint action rate penalty | $\sum(\Delta a^{\text{T}}_{i})^{2} ∑ (T) 2$ | - 0.05 0.05 -0.05 - 0.05 |
| Energy cost | $\sum\lVert \rVert\tau_{i}\cdot\dot{\theta}_{i}\lVert \rVert^{2} ∑ \lVert \rVert ⋅ \lVert \rVert 2$ | - 1  10 - 6 1 10 6 -1\!\times\!10^{-6} - 1  10 - 6 |
| Termination penalty | I terminated I terminated  $\mathbb{I}_{\text{terminated}} terminated$ | - 200 200 -200 - 200 |
| Joint limit penalty | $\mathbb{I}_{\theta_{i}\notin[\theta_{\min},\theta_{\max}]} ∉ [, ]$ | - 1 1 -1 - 1 |
| Orientation penalty | $\text{proj}}}\lVert \rVert^{2} \lVert \rVert proj \lVert \rVert 2$ | - 10.0 10.0 -10.0 - 10.0 |
| Feet slide penalty | $\mathbb{I}(F_{\text{feet}}>100\text{N})\cdot\sum\lVert \rVert{v_{xy}^{\text{feet}}}\lVert \rVert^{2} (feet > 100 N) ⋅ ∑ \lVert \rVert feet \lVert \rVert 2$ | - 0.1 0.1 -0.1 - 0.1 |
| Hip joint deviation | ∑ \| hip - default \| hip default $\sum\lVert \theta_{\text{hip}}-\theta_{\text{default}} \rVert ∑ \lVert hip - default \rVert$ | - 0.03 0.03 -0.03 - 0.03 |
| Leg joint deviation | ∑ \| leg - default \| leg default $\sum\lVert \theta_{\text{leg}}-\theta_{\text{default}} \rVert ∑ \lVert leg - default \rVert$ | - 0.01 0.01 -0.01 - 0.01 |
| Keypoint tracking | $\exp\left(-\frac{\lVert \rVert{{p}_{\text{key}}-{p}_{\text{ref}}}\lVert \rVert^{2}}{2}\right) (- divide \lVert \rVert key - ref \lVert \rVert 2 2)$ | 1.0 1.0 1.0 1.0 |
| Joint tracking | $\exp\left(-\frac{\lVert \rVert{\boldsymbol{\theta}-\boldsymbol{\theta}_{\text{ref}}}\lVert \rVert^{2}}{4}\right) (- divide \lVert \rVert - ref \lVert \rVert 2 4)$ | 1.0 1.0 1.0 1.0 |
| Single stance reward | $\mathbb{I}_{\begin{subarray}{c}\Delta h^{\text{feet}}_{\text{ref}}>0.05\\ t_{\text{stance}}\in[0.1,0.5]\end{subarray}}\cdot t_{\text{stance}} feet ref > 0.05 stance ∈ [0.1, 0.5 ] ⋅ stance$ | 1.5 1.5 1.5 1.5 |

**说明**: TABLE I: Reward Function Components for Teacher Policy

#### Table 2: TABLE II: Motion Quality Metric on unseen commands.

| Model | Similar | Moderate | Different |
| --- | --- | --- | --- |
| CLIP+CVAE | 80.92% | 69.58% | 54.62% |
| CLIP+MLP | 80.62% | 64.20% | 50.28% |

**说明**: TABLE II: Motion Quality Metric on unseen commands.

#### Table 3: TABLE III: Ablation study on the proposed architecture, symmetry loss, and student tracking objectives

| Metric | CVAE | No‐Symm | No‐Rel | MLP |
| --- | --- | --- | --- | --- |
| Motion Quality ↑ ↑ \uparrow ↑ | 96.2 % | 91.6% | 93.8% | 91.9% |
| Stability ↑ ↑ \uparrow ↑ | 99.10 % | 98.50% | 96.60% | 96.81% |
| Imitation Loss ↓ ↓ \downarrow ↓ | 0.085 | 0.107 | 0.098 | 0.091 |

**说明**: TABLE III: Ablation study on the proposed architecture, symmetry loss, and student tracking objectives

#### Table 4: TABLE IV: Domain Randomization Parameters

| Mode | Component | Range |
| --- | --- | --- |
| Reset | Initial velocity (linear) | [- 0.5, 0.5 ] 0.5 0.5 [-0.5,0.5] [- 0.5, 0.5 ] m/s |
| Initial velocity (angular) | [- 0.5, 0.5 ] 0.5 0.5 [-0.5,0.5] [- 0.5, 0.5 ] rad/s | |
| Reset | Joint positions | [0.5, 1.5 ]  [0.5,1.5]\times [0.5, 1.5 ]  default |
| Joint velocities | [0.0, 0.0 ] 0.0 0.0 [0.0,0.0] [0.0, 0.0 ] rad/s | |
| Startup | Ankle friction (static) | [0.2, 0.6 ] 0.2 0.6 [0.2,0.6] [0.2, 0.6 ] |
| Ankle friction (dynamic) | [0.2, 0.6 ] 0.2 0.6 [0.2,0.6] [0.2, 0.6 ] | |
| Ankle restitution | [0.0, 0.4 ] 0.0 0.4 [0.0,0.4] [0.0, 0.4 ] | |
| Startup | Link masses | [0.9, 1.1 ]  [0.9,1.1]\times [0.9, 1.1 ]  default |
| Base mass modification | [- 1.0, 1.0 ] 1.0 1.0 [-1.0,1.0] [- 1.0, 1.0 ] kg (additive) | |
| Joint armature | [0.8, 1.2 ]  [0.8,1.2]\times [0.8, 1.2 ]  default | |
| Startup | Joint default positions | [- 0.05, 0.05 ] 0.05 0.05 [-0.05,0.05] [- 0.05, 0.05 ] rad (additive) |
| Reset | Base torque | [- 5.0, 5.0 ] 5.0 5.0 [-5.0,5.0] [- 5.0, 5.0 ] Nm |
| Interval | Push robot (every 10-15s) | [- 1.0, 1.0 ] 1.0 1.0 [-1.0,1.0] [- 1.0, 1.0 ] m/s (x,y) |

**说明**: TABLE IV: Domain Randomization Parameters
## 实验解读

- 评价重点:围绕 adaptive-control、agile-locomotion、cvae、人形机器人、language-conditioned-policy,优先看真实机器人结果、扰动恢复、接触稳定性、控制误差、成功率、实时性和 sim-to-real 差距。
- System0 视角:实验结论不只看最终成功率,还要看闭环过程中是否能处理状态估计误差、执行延迟、碰撞/接触突变和外部扰动。
- 与上层系统的关系:如果方法能在快闭环内稳定工作,它可以作为 System1 技能或 System2 规划的底层执行保障;如果只在离线或慢速设定下成立,则还不能直接作为身体级快闭环模块。
## 批判性思考

### 优点

- 论文把 adaptive-control、agile-locomotion、cvae、人形机器人、language-conditioned-policy 放到机器人身体级闭环里讨论,和只做高层语义推理的工作相比,更接近真实机器人部署时会遇到的状态、动力学和安全约束。
- 若论文同时给出公式、系统结构图、硬件实验或消融结果,这些证据可以帮助判断方法是否真的能支撑毫秒到百毫秒级反应。

### 局限

- 需要重点核查控制频率、推理延迟、传感延迟、失败恢复、硬件磨损、未见场景泛化和安全边界;这些因素决定它能否成为可靠的 System0 模块。
- 对学习型方法,还要区分训练环境中的鲁棒性和真实部署中的鲁棒性,尤其是接触丰富任务、动态运动任务和人形机器人任务。
## 论文定位

- 研究对象:LangWBC: Language-directed Humanoid Whole-Body Control via End-to-end Learning。
- 关键词:adaptive-control、agile-locomotion、cvae、人形机器人、language-conditioned-policy、足式运动、强化学习、robot-generalization、鲁棒控制、全身控制。
- System0 关联:关注机器人身体级快反应链路中“状态/接触/局部感知 -> 约束/优化/策略 -> 动作”的短闭环,不把问题停留在语言规划或任务分解层。

## 速查卡片

> [!summary] LangWBC
> - **论文**: https://www.roboticsproceedings.org/rss21/p065.pdf
> - **arXiv**: http://arxiv.org/abs/2504.21738v1
> - **arXiv HTML**: https://arxiv.org/html/2504.21738v1
