---
title: "Simulation Distillation: Pretraining World Models in Simulation for Rapid Real-World Adaptation"
method_name: "Simulation Distillation"
authors: ["Jacob Levy"]
year: 2026
venue: "RSS"
tags: ["contact-reasoning", "robot-manipulation", "real-time-control", "legged-locomotion", "reinforcement-learning", "adaptive-control", "robot-generalization", "sim-to-real", "loco-manipulation"]
image_source: "online"
arxiv_html: "https://arxiv.org/html/2603.15759v2"
---
# Simulation Distillation
## 一句话总结

> Simulation Distillation: Pretraining World Models in Simulation for Rapid Real-World Adaptation 主要落在 [[adaptive-control]]、[[接触推理]]、[[足式运动]]、[[移动操作]] 这一类问题上;从 System0 视角看,它的价值在于把身体状态、接触变化、局部感知或动力学约束接到机器人低层闭环里,让动作生成更稳定、安全、实时。
## 核心贡献

1. 围绕 **Simulation Distillation: Pretraining World Models in Simulation for Rapid Real-World Adaptation** 建立了一个与 adaptive-control、接触推理、足式运动、移动操作、quadruped、reactive-control 相关的机器人问题设定,重点不是高层语义规划,而是身体级闭环中的状态变化、接触变化或控制约束。
2. 方法层面的关键价值在于把感知、动力学、约束优化、策略学习或硬件反馈组织成可执行的低层动作生成机制,适合放进 System0 的快反应模块中理解。
3. 对系统集成而言,这类工作补上了 System1 技能选择与电机级控制之间的一层:它负责把上层目标变成动态可行、可恢复、可安全过滤的动作。
4. 若论文包含真实机器人、接触任务或仿真到真实迁移,核心应看它是否报告了闭环稳定性、扰动恢复、硬件延迟、失败模式和跨场景泛化。
## 问题背景

这篇论文可以放在 System0 的“具身快系统”背景下阅读:机器人需要在很短时间内根据本体状态、接触、力/触觉、局部视觉或动力学预测做出动作,而不是等待慢速规划器重新推理整个任务。adaptive-control、接触推理、足式运动、移动操作、quadruped、reactive-control、实时控制 相关问题的难点通常不在“知道要做什么”,而在动作执行过程中状态会快速偏离、接触会突然改变、约束会实时收紧,低层系统必须及时修正。

### System0 关联

精读时优先看三件事:第一,论文使用哪些快速反馈量,例如关节状态、接触状态、力觉、触觉、深度或局部视觉;第二,反馈如何进入 MPC、WBC、策略网络、安全过滤器或恢复控制器;第三,输出是否直接影响机器人下一步动作,而不是只服务离线分析或高层任务规划。
## 方法详解

### System0 视角解析

从 System0 角度看,这篇论文的方法部分应当围绕 adaptive-control、接触推理、足式运动、移动操作、quadruped、reactive-control、实时控制、强化学习 来读。核心问题是:系统如何从快速变化的机器人状态中构造控制输入,如何处理接触、碰撞、约束、模型误差或执行延迟,以及如何把这些信息变成下一时刻的动作。

如果方法里有优化问题,重点看目标函数、约束和求解频率;如果方法里有学习策略,重点看观测空间、动作空间、训练扰动和部署时的闭环频率;如果方法里有硬件或传感器,重点看传感链路是否真的进入控制回路。这样读可以把论文和 System2 语言规划、System1 技能编排区分开来。

### 关键公式与机制

#### 公式 1: [[动力学或策略机制]]

$$\mathbb{E}\left[\sum_{t=0}^{\infty}\gamma^{t}r(s_{t},a_{t},s_{t+1})\right]$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 2: [[动力学或策略机制]]

$$\begin{array}{ll}\text{Latent representation:}&z_{t}=E_{\theta}(o_{t})\\ \text{History representation:}&h_{t}=C_{\theta}(o_{t-H:t-1},a_{t-H:t-1})\\ \text{Latent dynamics:}&\hat{z}_{t+1:t+T}=f_{\theta}(z_{t},a_{t:t+T-1},h_{t})\\ \text{Reward Prediction:}&\hat{r}_{t:t+T-1}=R_{\theta}(\hat{z}_{t:t+T},a_{t:t+T-1})\\ \text{Value Prediction:}&\hat{v}_{t+1:t+T}=V_{\theta}(\hat{z}_{t:t+T})\\ \text{Base Policy:}&\hat{a}_{t:t+H}=\pi_{\theta}(z_{t},h_{t}).\end{array}$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 3: [[优化目标/约束]]

$$\mathcal{R}(a_{t:t+T-1})=\gamma^{T}\hat{v}_{t+T}+\sum_{s=t}^{t+T-1}\gamma^{s-t}\hat{r}_{s}$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 4: [[优化目标/约束]]

$$\mathcal{D}_{\mathtt{sim}}=\{(o_{t},a_{t},r_{t},v_{t})\}_{t=0}^{N}$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 5: [[优化目标/约束]]

$$\{(o_{t},r_{t},v_{t},a_{t})\}_{t=i-H}^{i+T}\sim\mathcal{D}_{\mathtt{sim}}$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 6: [[优化目标/约束]]

$$\{(o_{t},a_{t})\}_{t=i-H}^{i+T}\sim\mathcal{D}_{\mathtt{real}}$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 7: [[优化目标/约束]]

$$\begin{array}{l}\mathcal{L}_{t}^{\mathtt{real}}(\theta)=\sum_{i=0}^{T}\,\,\mathord{\left\lVert \hat{z}_{t+i+1}-\mathtt{sg}(E_{\theta}(o_{t+i+1}))\right \rVert}^{2}_{2},\\ \text{with}C_{\theta},E_{\theta},R_{\theta},V_{\theta},\pi_{\theta}\ \text{frozen,}f_{\theta}\text{finetunable}.\end{array}$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 8: [[优化目标/约束]]

$$
\mathcal{L}_{t}^{\mathtt{sim}}(\theta)=\sum_{i=0}^{T}\bigg(
$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 9: [[动力学或策略机制]]

$$\ {$\mathord${$\left$$\lVert$$\hat{z}_{t+i+1}$-$\mathtt{sg}$($E_{\theta}($o_{t+i+1}))$\right$$\rVert$}^{2}_{2}}_{$\text{latent dynamics}$}
$$**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。 **符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。 #### 公式 10: [[动力学或策略机制]]$$
+c_{3}\ {$\mathbf{1}_{e}$(a_{t+i})$\mathord{\left\lVert \hat{a}_{t+i}-a_{t+i}\right \rVert}^{2}_{2}$}_{$\text{behavior cloning}$}$\bigg$),
$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。
## 关键图表

### Figure 1: Pipeline / core system figure: SimDist. 1) An expert policy, policy checkpoints, and a value function are t

![Figure 1](https://arxiv.org/html/2603.15759v2/x1.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Pipeline / core system figure: SimDist. 1) An expert policy, policy checkpoints, and a value function are t”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 2: World model architecture. The most recent observation is encoded into a latent repres

![Figure 2](https://arxiv.org/html/2603.15759v2/x2.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“World model architecture. The most recent observation is encoded into a latent repres”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 3: Detailed world model architecture for the quadruped

![Figure 3](https://arxiv.org/html/2603.15759v2/x8.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Detailed world model architecture for the quadruped”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### 论文表格

#### Table 1: TABLE I: Ablation results in simulation, reporting success rates for manipulation tasks and average state-based reward

| | Peg Insertion | Table Leg | Quadruped |
| --- | --- | --- | --- |
| | (SR) | (SR) | (Reward) |
| SimDist | 0.90 | 0.85 | 22.78 |
| 50% data | 0.72 | 0.61 | 22.73 |
| 10% data | 0.06 | 0.02 | 19.38 |
| Expert Data Only | 0.10 | 0.05 | 16.68 |
| MLP Reward+Value Models | 0.82 | 0.60 | 19.47 |
| Raw Obs. Reconstruction | 0.32 | 0.21 | 23.34 |

**说明**: TABLE I: Ablation results in simulation, reporting success rates for manipulation tasks and average state-based reward per episode for the quadruped.

#### Table 2: TABLE II: World model architectural parameters for manipulation.

| Parameter | Value |
| --- | --- |
| Embedding dimension | 64 |
| All transformers MLP hidden size | 256 |
| Dynamics transformer layers | 3 |
| Dynamics transformer heads | 4 |
| Reward transformer layers | 1 |
| Reward transformer heads | 1 |
| Value transformer layers | 1 |
| Value transformer heads | 1 |
| Base policy transformer layers | 4 |
| Base policy transformer heads | 8 |

**说明**: TABLE II: World model architectural parameters for manipulation.

#### Table 3: TABLE III: MPPI parameters for manipulation.

| Parameter | Value |
| --- | --- |
| Candidate actions batch size | 250 |
| Noised base policy actions batch size | 100 |
| Solver iterations | 3 |
| Initial action standard deviation | 1.0 |
| Minimum action standard deviation | 0.05 |
| Base policy action standard deviation | 0.1 |
| Elites | 64 |
| Temperature | 0.4 |
| Momentum | 0.0 |
| Discount | 0.99 |

**说明**: TABLE III: MPPI parameters for manipulation.

#### Table 4: TABLE IV: Privileged simulator state space for the quadruped, along with domain randomization ranges, where applicable.

| State Variable | Dim. | Domain Rand. |
| --- | --- | --- |
| Base linear velocity | 3 3 | - |
| Base angular velocity | 3 3 | - |
| Projected gravity vector | 3 3 | - |
| Commanded base twist | 3 | - |
| Joint angles | 12 12 | - |
| Joint speeds | 12 12 | - |
| Previous action | 12 12 | - |
| Cosine / sine of phase | 2 | - |
| Height map | (21, 15) (21,15) | - |
| Foot force wrenches | (4, 6) (4,6) | - |
| Foot heights | 4 4 | - |
| Base mass | 1 1 | - 1.0, + 3.0 -1.0,+3.0 kg  $\mathrm{kg}$ |
| Static / dynamic friction | 2 2 | [0.2, 1.2 ] [0.2,1.2] |
| Coefficient of restitution | 1 1 | [0.0, 0.3 ] [0.0,0.3] |
| Joint stiffness | 12 12 | ± 10 % \pm 10\% |
| Joint damping | 12 12 | ± 10 % \pm 10\% |
| Joint friction | 12 12 | [0.0, 0.05 ] [0.0,0.05] |

**说明**: TABLE IV: Privileged simulator state space for the quadruped, along with domain randomization ranges, where applicable.

#### Table 5: TABLE V: State-based reward terms used for quadruped expert policy training.

| Term | Weight |
| --- | --- |
| Commanded x, y x,y -velocity tracking reward | 1.5 1.5 |
| Commanded yaw rate tracking reward | 0.75 0.75 |
| Desired gait reward | 0.05 0.05 |
| Desired gait foot height reward | 0.2 0.2 |
| Base z z -velocity penalty | - 2.0 -2.0 |
| Base angular velocity penalty | - 0.05 -0.05 |
| Base orientation penalty | - 4.0 -4.0 |
| Deviation from default hip joint angles penalty | - 0.25 -0.25 |
| Joint torque penalty | $\text{\times}{10}^{-4}$ |
| Joint acceleration penalty | $\text{\times}{10}^{-7}$ |
| Action rate penalty | - 0.01 -0.01 |

**说明**: TABLE V: State-based reward terms used for quadruped expert policy training.

#### Table 6: TABLE VI: Observation space for the quadruped.

| Observation | Dimension |
| --- | --- |
| Base linear velocity (local frame) | 3 |
| Base angular velocity (local frame) | 3 |
| Projected gravity vector | 3 |
| Joint angles | 12 |
| Joint speeds | 12 |
| Cosine and sine of phase | 2 |
| Height map | (21, 15) (21,15) |

**说明**: TABLE VI: Observation space for the quadruped.

#### Table 7: TABLE VII: World model architectural parameters for the quadruped.

| Parameter | Value |
| --- | --- |
| Embedding dimension | 64 |
| Proprioceptive observations MLP hidden dims | 128, 128 |
| CNN kernel size | 3 |
| CNN strides | 2, 2, 2 |
| CNN features | 8, 16, 32 |
| All transformers MLP hidden size | 256 |
| Dynamics transformer layers | 2 |
| Dynamics transformer heads | 8 |
| Reward transformer layers | 1 |
| Reward transformer heads | 1 |
| Value transformer layers | 1 |
| Value transformer heads | 1 |
| Base policy transformer layers | 4 |
| Base policy transformer heads | 8 |

**说明**: TABLE VII: World model architectural parameters for the quadruped.

#### Table 8: TABLE VIII: MPPI parameters for the quadruped.

| Parameter | Value |
| --- | --- |
| Candidate actions batch size | 450 |
| Noised base policy actions batch size | 22 |
| Solver iterations | 8 |
| Initial action standard deviation | 2.0 |
| Minimum action standard deviation | 0.05 |
| Base policy action standard deviation | 0.05 |
| Elites | 64 |
| Temperature | 0.25 |
| Momentum | 0.0 |
| Discount | 0.99 |

**说明**: TABLE VIII: MPPI parameters for the quadruped.

#### Table 9: TABLE IX: Real-world quadruped results for both tasks. Success is reported as successful trials out of five. Forward pr

| | Speed | Pretrained model | Single-step BC policy | SimDist (ours) | IQL | RLPD | | | | | |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| | m s - 1  $\mathrm{m}\text{\,}{\mathrm{s}}^{-1}$ | Fwd. Prog. | Success | Fwd. Prog. | Success | Fwd. Prog. | Success | Fwd. Prog. | Success | Fwd. Prog. | Success |
| Slippery | 0.1 | 0.70 ± \pm 0.56 | 0/5 | 1.49 ± \pm 0.29 | 2/5 | 1.78 ± \pm 0.08 | 4/5 | 0.00 ± \pm 0.00 | 0/5 | 0.32 ± \pm 0.01 | 0/5 |
| Slope | 0.3 | 0.43 ± \pm 0.13 | 0/5 | 1.43 ± \pm 0.25 | 1/5 | 1.82 ± \pm 0.00 | 5/5 | 0.39 ± \pm 0.56 | 0/5 | 0.34 ± \pm 0.05 | 0/5 |
| | 0.5 | 0.90 ± \pm 0.40 | 0/5 | 0.62 ± \pm 0.23 | 0/5 | 1.82 ± \pm 0.00 | 5/5 | 0.46 ± \pm 0.42 | 0/5 | 0.35 ± \pm 0.01 | 0/5 |
| Foam | 0.2 | 2.98 ± \pm 0.02 | 3/5 | 2.07 ± \pm 1.01 | 1/5 | 3.00 ± \pm 0.00 | 5/5 | 0.92 ± \pm 0.48 | 1/5 | – | – |
| | 0.7 | 2.39 ± \pm 0.71 | 2/5 | 1.54 ± \pm 0.81 | 1/5 | 3.00 ± \pm 0.00 | 5/5 | 2.25 ± \pm 0.87 | 2/5 | – | – |
| | 1.2 | 1.70 ± \pm 0.99 | 0/5 | 2.45 ± \pm 0.47 | 2/5 | 3.00 ± \pm 0.00 | 5/5 | 2.73 ± \pm 0.34 | 3/5 | – | – |

**说明**: TABLE IX: Real-world quadruped results for both tasks. Success is reported as successful trials out of five. Forward progress is reported as mean ± \pm standard deviation (meters) across trials at each commanded speed. The Pretrained model corresponds to zero-shot deployment of the simulation-trained world model. The Single-step BC policy is the behavior cloning policy used to initialize IQL and RLPD prior to finetuning. SimDist, IQL, and RLPD results reflect performance after real-world finetuning using 35.7 minutes (Slippery Slope) and 32.1 minutes (Foam) of data. RLPD results on the Foam task are not reported, as the method destabilized the robot prior to evaluation.

#### Table 10: TABLE X: Parameter values used to construct quadruped ablation environments in simulation. Each environment is defined

| | | | Terrain |
| --- | --- | --- | --- |
| Speed | Friction | Terrain | Difficulty |
| 0.2 | 0.2 | Boxes | 0.2 |
| 0.4 | 0.4 | Rough | 0.4 |
| 0.6 | 0.6 | Stairs Up | 0.6 |
| 0.8 | 0.8 | Stairs Down | 0.8 |
| 1.0 | 1 | Slope Up | 1 |
| 1.2 | 1.2 | Slope Down | |

**说明**: TABLE X: Parameter values used to construct quadruped ablation environments in simulation. Each environment is defined by a unique combination of commanded forward speed, ground friction coefficient, terrain type, and terrain difficulty. All combinations are evaluated for each model, yielding 1080 environments per model.
## 实验解读

- 评价重点:围绕 adaptive-control、接触推理、足式运动、移动操作、quadruped,优先看真实机器人结果、扰动恢复、接触稳定性、控制误差、成功率、实时性和 sim-to-real 差距。
- System0 视角:实验结论不只看最终成功率,还要看闭环过程中是否能处理状态估计误差、执行延迟、碰撞/接触突变和外部扰动。
- 与上层系统的关系:如果方法能在快闭环内稳定工作,它可以作为 System1 技能或 System2 规划的底层执行保障;如果只在离线或慢速设定下成立,则还不能直接作为身体级快闭环模块。
## 批判性思考

### 优点

- 论文把 adaptive-control、接触推理、足式运动、移动操作、quadruped 放到机器人身体级闭环里讨论,和只做高层语义推理的工作相比,更接近真实机器人部署时会遇到的状态、动力学和安全约束。
- 若论文同时给出公式、系统结构图、硬件实验或消融结果,这些证据可以帮助判断方法是否真的能支撑毫秒到百毫秒级反应。

### 局限

- 需要重点核查控制频率、推理延迟、传感延迟、失败恢复、硬件磨损、未见场景泛化和安全边界;这些因素决定它能否成为可靠的 System0 模块。
- 对学习型方法,还要区分训练环境中的鲁棒性和真实部署中的鲁棒性,尤其是接触丰富任务、动态运动任务和人形机器人任务。
## 论文定位

- 研究对象:Simulation Distillation: Pretraining World Models in Simulation for Rapid Real-World Adaptation。
- 关键词:adaptive-control、接触推理、足式运动、移动操作、quadruped、reactive-control、实时控制、强化学习、机器人操作、仿真到真实迁移。
- System0 关联:关注机器人身体级快反应链路中“状态/接触/局部感知 -> 约束/优化/策略 -> 动作”的短闭环,不把问题停留在语言规划或任务分解层。

## 速查卡片

> [!summary] Simulation Distillation
> - **论文**: https://www.roboticsproceedings.org/rss22/p017.pdf
> - **arXiv**: http://arxiv.org/abs/2603.15759v2
> - **arXiv HTML**: https://arxiv.org/html/2603.15759v2
