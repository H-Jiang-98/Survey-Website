---
title: "Tune to Learn: How Controller Gains Shape Robot Policy Learning"
method_name: "Tune to Learn"
authors: ["Antonia Bronars"]
year: 2026
venue: "RSS"
tags: ["contact-reasoning", "robot-manipulation", "robust-control", "reinforcement-learning", "imitation-learning", "closed-loop-control", "sim-to-real"]
image_source: "online"
arxiv_html: "https://arxiv.org/html/2604.02523v1"
---
# Tune to Learn
## 一句话总结

> Tune to Learn: How Controller Gains Shape Robot Policy Learning 主要落在 [[closed-loop-control]]、[[compliance-control]]、[[接触推理]]、[[模仿学习]] 这一类问题上;从 System0 视角看,它的价值在于把身体状态、接触变化、局部感知或动力学约束接到机器人低层闭环里,让动作生成更稳定、安全、实时。
## 核心贡献

1. 围绕 **Tune to Learn: How Controller Gains Shape Robot Policy Learning** 建立了一个与 closed-loop-control、compliance-control、接触推理、模仿学习、policy-learning、强化学习 相关的机器人问题设定,重点不是高层语义规划,而是身体级闭环中的状态变化、接触变化或控制约束。
2. 方法层面的关键价值在于把感知、动力学、约束优化、策略学习或硬件反馈组织成可执行的低层动作生成机制,适合放进 System0 的快反应模块中理解。
3. 对系统集成而言,这类工作补上了 System1 技能选择与电机级控制之间的一层:它负责把上层目标变成动态可行、可恢复、可安全过滤的动作。
4. 若论文包含真实机器人、接触任务或仿真到真实迁移,核心应看它是否报告了闭环稳定性、扰动恢复、硬件延迟、失败模式和跨场景泛化。
## 问题背景

这篇论文可以放在 System0 的“具身快系统”背景下阅读:机器人需要在很短时间内根据本体状态、接触、力/触觉、局部视觉或动力学预测做出动作,而不是等待慢速规划器重新推理整个任务。closed-loop-control、compliance-control、接触推理、模仿学习、policy-learning、强化学习、机器人操作 相关问题的难点通常不在“知道要做什么”,而在动作执行过程中状态会快速偏离、接触会突然改变、约束会实时收紧,低层系统必须及时修正。

### System0 关联

精读时优先看三件事:第一,论文使用哪些快速反馈量,例如关节状态、接触状态、力觉、触觉、深度或局部视觉;第二,反馈如何进入 MPC、WBC、策略网络、安全过滤器或恢复控制器;第三,输出是否直接影响机器人下一步动作,而不是只服务离线分析或高层任务规划。
## 方法详解

### System0 视角解析

从 System0 角度看,这篇论文的方法部分应当围绕 closed-loop-control、compliance-control、接触推理、模仿学习、policy-learning、强化学习、机器人操作、仿真到真实迁移 来读。核心问题是:系统如何从快速变化的机器人状态中构造控制输入,如何处理接触、碰撞、约束、模型误差或执行延迟,以及如何把这些信息变成下一时刻的动作。

如果方法里有优化问题,重点看目标函数、约束和求解频率;如果方法里有学习策略,重点看观测空间、动作空间、训练扰动和部署时的闭环频率;如果方法里有硬件或传感器,重点看传感链路是否真的进入控制回路。这样读可以把论文和 System2 语言规划、System1 技能编排区分开来。

### 关键公式与机制

#### 公式 1: [[动力学或策略机制]]

$$\mathbf{M}(\mathbf{q})\ddot{\mathbf{q}}+\mathbf{C}(\mathbf{q},\dot{\mathbf{q}})\dot{\mathbf{q}}+\mathbf{g}(\mathbf{q})=\boldsymbol{\tau}+\boldsymbol{\tau}_{\text{ext}}$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 2: [[动力学或策略机制]]

$$\boldsymbol{\tau}=\mathbf{K}_{p}(\mathbf{q}_{d}-\mathbf{q})+\mathbf{K}_{d}(\dot{\mathbf{q}}_{d}-\dot{\mathbf{q}})+\mathbf{g}(\mathbf{q})$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 3: [[动力学或策略机制]]

$$\mathbf{q}_{\text{des}}(t)=\mathbf{q}(t)+\mathbf{K}_{p}^{-1}\left(\boldsymbol{\tau}(t)+\mathbf{K}_{d}\dot{\mathbf{q}}(t)\right),$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 4: [[优化目标/约束]]

$$\psi^{\star}(\mathbf{K})=\arg\min_{\psi}\sum_{t=0}^{T}\lVert \mathbf{x}(t;\mathbf{K})-\bar{\mathbf{x}}(t;\psi)\rVert^{2}$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 5: [[优化目标/约束]]

$$\mathcal{E}=\ {\lVert \mathbf{q}_{\text{sim}}-\mathbf{q}_{\text{real}} \rVert^{2}}_{\text{position error}}+\ {\lVert \dot{\mathbf{q}}_{\text{sim}}-\dot{\mathbf{q}}_{\text{real}} \rVert^{2}}_{\text{velocity error}}$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 6: [[动力学或策略机制]]

$$\boldsymbol{\tau}=\mathbf{J}^{\top}\mathbf{M}_{x}\left(\mathbf{K}_{p}\tilde{\mathbf{x}}-\mathbf{K}_{d}\dot{\mathbf{x}}\right)+\boldsymbol{\tau}_{\text{null}},$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 7: [[动力学或策略机制]]

$$\mathbf{x}_{\text{des}}(t)=\mathbf{x}(t)+\mathbf{K}_{p}^{\prime-1}\left(\mathbf{F}(t)+\mathbf{K}_{d}^{\prime}\dot{\mathbf{x}}(t)\right),$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 8: [[优化目标/约束]]

$$\mathcal{L}(\psi)=\mathcal{L}_{\text{spec}}(\mathbf{q}^{\text{real}},\,\mathbf{q}^{\text{sim}}(\psi))+\mathcal{L}_{\text{spec}}(\dot{\mathbf{q}}^{\text{real}},\,\dot{\mathbf{q}}^{\text{sim}}(\psi))$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 9: [[优化目标/约束]]

$$\mathcal{E}_{\text{NN}}=\text{RMS}\left(\pi_{\theta}(\mathbf{s}_{t}^{\text{real}})-\pi_{\theta}(\mathbf{s}_{t}^{\text{sim}})\right)$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 10: [[优化目标/约束]]

$$\mathcal{J}=\begin{cases}1+r_{\text{success}}&\text{if all}v_{c}\leq\bar{v}_{c}\\[4.0pt] r_{\text{success}}\prod_{c\in\mathcal{C}}\phi_{c}&\text{otherwise}\end{cases}$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。
## 关键图表

### Figure 1: Pipeline / core system figure: a) BC

![Figure 1](https://arxiv.org/html/2604.02523v1/assets/main-figure-gain-setting-2.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Pipeline / core system figure: a) BC”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 2: Any teleoperation system requires a mapping φ $\phi$ from user inputs u $\mathbf{u}$ to d

![Figure 2](https://arxiv.org/html/2604.02523v1/x17.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Any teleoperation system requires a mapping φ $\phi$ from user inputs u $\mathbf{u}$ to d”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 3: End-to-end BC pipeline still favors compliant and gain regime

![Figure 3](https://arxiv.org/html/2604.02523v1/x28.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“End-to-end BC pipeline still favors compliant and gain regime”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### 论文表格

#### Table 1: TABLE I: RL solution existence across gain regimes. For each task, we verify that at least one successful policy can be

| Task / Platform | Existence Proof |
| --- | --- |
| FR3 Joint-Reach | ✓ |
| FR3 EE-Reach | ✓ |
| FR3 Lift-Cube | ✓ |
| FR3 Open-Drawer | ✓ |
| G1 Track-Velocity | ✓ |
| Allegro In-Hand Cube Manipulation | ✓ |
| FR3 Nonprehensile Cube Reorientation | ✓ |

**说明**: TABLE I: RL solution existence across gain regimes. For each task, we verify that at least one successful policy can be discovered in every gain configuration given appropriate environment shaping. A checkmark indicates that gain regimes corresponding to four corner extremes in Fig. 2 yield working controllers (99%+ success rate). Videos or live policy rollouts of discovered behaviors for each gain settings are available on our project website.

#### Table 2: TABLE II: Statistical analysis of BC results. Success rates for compliant- (G CO $\mathcal{G}^{\text{CO}}$) v

| | Success Rate | Logistic Reg. | Barnard’s | | |
| --- | --- | --- | --- | --- | --- |
| Task | G CO  $\mathcal{G}^{\text{CO}}$ | G ∖ G CO  $\mathcal{G}\setminus\mathcal{G}^{\text{CO}}$ | $\mathbf{K}_{\text{p}}}$ | $\mathbf{K}_{\text{d}}}$ | p p -value |
| Dishwasher Opening | 70.1% | 45.6% | - 0.074 -0.074 | + 0.277 +0.277 | 1.6  10 - 51 1.6\times 10^{-51} |
| Bimanual Handover | 35.1% | 16.5% | - 0.173 -0.173 | + 0.225 +0.225 | 6.2  10 - 35 6.2\times 10^{-35} |
| Mug Hanging | 62.4% | 40.9% | - 0.146 -0.146 | + 0.155 +0.155 | 1.4  10 - 40 1.4\times 10^{-40} |
| Dishrack Unloading | 12.7% | 6.9% | - 0.144 -0.144 | + 0.159 +0.159 | 1.2  10 - 5 1.2\times 10^{-5} |
| Dishrack Loading | 28.2% | 14.8% | - 0.169 -0.169 | + 0.217 +0.217 | 2.3  10 - 19 2.3\times 10^{-19} |
| Block Stacking | 85.1% | 39.0% | - 0.265 -0.265 | + 0.429 +0.429 | 6.9  10 - 231 6.9\times 10^{-231} |
| Pooled | — | - 0.160 -0.160 | + 0.220 +0.220 | — | |

**说明**: 该表格来自论文中的公式、奖励项或实验设置；为避免 PDF 抽取噪声，网页中仅保留可渲染的核心符号和数值。

#### Table 3: TABLE III: Action representation across RL tasks.

| Task | q ref  (t)  $\mathbf{q}_{\text{ref}}(t)$ | G 1  $\mathcal{G}_{1}$ | G 2  $\mathcal{G}_{2}$ | Gripper |
| --- | --- | --- | --- | --- |
| FR3 Joint-Reach | q  (t)  $\mathbf{q}(t)$ | q 0  –  3 q_{0 $\text{--}3} (elbow)$ | q 4  –  6 q_{4 $\text{--}6} (wrist)$ | – |
| FR3 EE-Reach | q  (t)  $\mathbf{q}(t)$ | q 0  –  3 q_{0 $\text{--}3} (elbow)$ | q 4  –  6 q_{4 $\text{--}6} (wrist)$ | – |
| FR3 Lift Cube | q  (t)  $\mathbf{q}(t)$ | q 0  –  3 q_{0 $\text{--}3} (elbow)$ | q 4  –  6 q_{4 $\text{--}6} (wrist)$ | binary |
| FR3 Open Drawer | q  (t)  $\mathbf{q}(t)$ | q 0  –  3 q_{0 $\text{--}3} (elbow)$ | q 4  –  6 q_{4 $\text{--}6} (wrist)$ | binary |
| G1 Locomotion | q 0  $\mathbf{q}_{0}$ | q 0  –  13 q_{0 $\text{--}13} (lower)$ | q 13  –  36 q_{13 $\text{--}36} (upper)$ | – |

**说明**: TABLE III: Action representation across RL tasks.

#### Table 4: TABLE IV: Success criteria for each RL task.

| Task | Criterion | Threshold |
| --- | --- | --- |
| FR3 Joint-Reach | $\mathbf{q}-\mathbf{q}_{\text{goal}}\lVert<\epsilon$ |  = 0.1 \epsilon=0.1 rad |
| FR3 EE-Reach | $\mathbf{p}-\mathbf{p}_{\text{goal}}\lVert<\epsilon_{p},$ |  p = 0.02 \epsilon_{p}=0.02 m |
| | $\theta\lVert<\epsilon_{r}$ |  r = 0.1 \epsilon_{r}=0.1 rad |
| FR3 Lift Cube | $\mathbf{p}_{\text{obj}}-\mathbf{p}_{\text{goal}}\lVert<\epsilon$ |  = \epsilon= 0.05m |
| FR3 Open Drawer | d drawer > d min d_{ $\text{drawer}}>d_{\text{min}}$ | d min = 0.2 d_{ $\text{min}}=0.2$ |
| G1 Locomotion | $\dot{\mathbf{q}}\lVert/\rVert\dot{\mathbf{q}}_{\text{goal}}\lVert>\rho$ |  = 0.4 \rho=0.4 |

**说明**: TABLE IV: Success criteria for each RL task.

#### Table 5: TABLE V: PPO hyperparameters shared across all tasks.

| Hyperparameter | Value |
| --- | --- |
| Algorithm | PPO (SKRL) |
| Discount factor $\gamma$ | 0.99 0.99 |
| GAE  \lambda | 0.95 0.95 |
| Learning epochs | 5 5 |
| Clip range (ratio) | 0.2 0.2 |
| Clip range (value) | 0.2 0.2 |
| Grad norm clip | 1.0 1.0 |
| LR scheduler | KL-Adaptive |
| Activation | ELU |
| Seed | 42 42 |

**说明**: TABLE V: PPO hyperparameters shared across all tasks.

#### Table 6: TABLE VI: PPO hyperparameters that vary across tasks.

| Hyperparameter | Reach | Lift | Drawer | G1 |
| --- | --- | --- | --- | --- |
| Network layers | [64,64] | [256,128,64] | [256,128,64] | [256,128,128] |
| Learning rate | 1  e-  3 1 $\text{e-}3$ | 1  e-  3 1 $\text{e-}3$ | 5  e-  4 5 $\text{e-}4$ | 1  e-  3 1 $\text{e-}3$ |
| KL threshold | 0.01 0.01 | 0.01 0.01 | 0.008 0.008 | 0.01 0.01 |
| Rollouts | 24 24 | 24 24 | 96 96 | 24 24 |
| Mini-batches | 4 4 | 4 4 | 96 96 | 4 4 |
| Entropy coeff. | 0.01 0.01 | 0.01 0.01 | 0.001 0.001 | 0.008 0.008 |
| Value loss coeff. | 1.0 1.0 | 1.0 1.0 | 2.0 2.0 | 1.0 1.0 |
| Min log std | - 3.0 -3.0 | - 3.0 -3.0 | - 20.0 -20.0 | - 20.0 -20.0 |
| State preprocess | RSS | RSS | – | – |
| Value preprocess | RSS | RSS | – | – |
| Timesteps | 24 24 k | 48 48 k | 38.4 38.4 k | 12 12 k |
| RSS = RunningStandardScaler. | | | | |

**说明**: TABLE VI: PPO hyperparameters that vary across tasks.

#### Table 7: TABLE VII: System identification parameter bounds. Parameters are optimized per-actuator.

| Parameter | Lower | Upper |
| --- | --- | --- |
| Stiffness K p K_{p} | 1 1 | 1024 1024 |
| Damping K d K_{d} | 1 1 | 1024 1024 |
| Armature | 0 | 0.5 0.5 |
| Static friction | 0.01 0.01 | 1.0 1.0 |
| Dynamic friction ratio | 0 | 1.0 1.0 |
| Viscous friction | 0 | 1.0 1.0 |

**说明**: TABLE VII: System identification parameter bounds. Parameters are optimized per-actuator.

#### Table 8: TABLE VIII: Statistical analysis of Sim2Real results. Mean trajectory error for stiff- (G SO \mathcal{G}^{$\t$

| | Mean Error | OLS Reg. | Mann-Whitney | | |
| --- | --- | --- | --- | --- | --- |
| Condition | G SO  $\mathcal{G}^{\text{SO}}$ | G ∖ G SO  $\mathcal{G}\setminus\mathcal{G}^{\text{SO}}$ | $\mathbf{K}_{\text{p}}}$ | $\mathbf{K}_{\text{d}}}$ | p p -value |
| Joint-Reach (no DR) | 0.043 | 0.010 | + 0.298 +0.298 | + 0.087 +0.087 | 1.9  10 - 36 1.9\times 10^{-36} |
| Joint-Reach (with DR) | 0.060 | 0.018 | + 0.354 +0.354 | + 0.215 +0.215 | 1.7  10 - 20 1.7\times 10^{-20} |
| EE-Reach (no DR) | 0.198 | 0.123 | + 0.063 +0.063 | + 0.127 +0.127 | 3.5  10 - 36 3.5\times 10^{-36} |

**说明**: 该表格来自论文中的公式、奖励项或实验设置；为避免 PDF 抽取噪声，网页中仅保留可渲染的核心符号和数值。
## 实验解读

- 评价重点:围绕 closed-loop-control、compliance-control、接触推理、模仿学习、policy-learning,优先看真实机器人结果、扰动恢复、接触稳定性、控制误差、成功率、实时性和 sim-to-real 差距。
- System0 视角:实验结论不只看最终成功率,还要看闭环过程中是否能处理状态估计误差、执行延迟、碰撞/接触突变和外部扰动。
- 与上层系统的关系:如果方法能在快闭环内稳定工作,它可以作为 System1 技能或 System2 规划的底层执行保障;如果只在离线或慢速设定下成立,则还不能直接作为身体级快闭环模块。
## 批判性思考

### 优点

- 论文把 closed-loop-control、compliance-control、接触推理、模仿学习、policy-learning 放到机器人身体级闭环里讨论,和只做高层语义推理的工作相比,更接近真实机器人部署时会遇到的状态、动力学和安全约束。
- 若论文同时给出公式、系统结构图、硬件实验或消融结果,这些证据可以帮助判断方法是否真的能支撑毫秒到百毫秒级反应。

### 局限

- 需要重点核查控制频率、推理延迟、传感延迟、失败恢复、硬件磨损、未见场景泛化和安全边界;这些因素决定它能否成为可靠的 System0 模块。
- 对学习型方法,还要区分训练环境中的鲁棒性和真实部署中的鲁棒性,尤其是接触丰富任务、动态运动任务和人形机器人任务。
## 论文定位

- 研究对象:Tune to Learn: How Controller Gains Shape Robot Policy Learning。
- 关键词:closed-loop-control、compliance-control、接触推理、模仿学习、policy-learning、强化学习、机器人操作、仿真到真实迁移。
- System0 关联:关注机器人身体级快反应链路中“状态/接触/局部感知 -> 约束/优化/策略 -> 动作”的短闭环,不把问题停留在语言规划或任务分解层。

## 速查卡片

> [!summary] Tune to Learn
> - **论文**: https://www.roboticsproceedings.org/rss22/p139.pdf
> - **arXiv**: http://arxiv.org/abs/2604.02523v1
> - **arXiv HTML**: https://arxiv.org/html/2604.02523v1
