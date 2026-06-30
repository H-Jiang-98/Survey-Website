---
title: "Demonstrating Agile Flight from Pixels without State Estimation"
method_name: "Agile Flight Pixels"
authors: ["Ismail Geles"]
year: 2024
venue: "RSS"
tags: ["robust-control", "reinforcement-learning", "imitation-learning", "robot-generalization", "agile-locomotion", "state-estimation", "navigation"]
image_source: "online"
arxiv_html: "https://arxiv.org/html/2406.12505v1"
---
# Agile Flight Pixels
## 一句话总结

> Demonstrating Agile Flight from Pixels without State Estimation 主要落在 [[aerial-robotics]]、[[agile-locomotion]]、[[egocentric-perception]]、[[模仿学习]] 这一类问题上;从 System0 视角看,它的价值在于把身体状态、接触变化、局部感知或动力学约束接到机器人低层闭环里,让动作生成更稳定、安全、实时。
## 核心贡献

1. 围绕 **Demonstrating Agile Flight from Pixels without State Estimation** 建立了一个与 aerial-robotics、agile-locomotion、egocentric-perception、模仿学习、navigation、强化学习 相关的机器人问题设定,重点不是高层语义规划,而是身体级闭环中的状态变化、接触变化或控制约束。
2. 方法层面的关键价值在于把感知、动力学、约束优化、策略学习或硬件反馈组织成可执行的低层动作生成机制,适合放进 System0 的快反应模块中理解。
3. 对系统集成而言,这类工作补上了 System1 技能选择与电机级控制之间的一层:它负责把上层目标变成动态可行、可恢复、可安全过滤的动作。
4. 若论文包含真实机器人、接触任务或仿真到真实迁移,核心应看它是否报告了闭环稳定性、扰动恢复、硬件延迟、失败模式和跨场景泛化。
## 问题背景

这篇论文可以放在 System0 的“具身快系统”背景下阅读:机器人需要在很短时间内根据本体状态、接触、力/触觉、局部视觉或动力学预测做出动作,而不是等待慢速规划器重新推理整个任务。aerial-robotics、agile-locomotion、egocentric-perception、模仿学习、navigation、强化学习、鲁棒控制 相关问题的难点通常不在“知道要做什么”,而在动作执行过程中状态会快速偏离、接触会突然改变、约束会实时收紧,低层系统必须及时修正。

### System0 关联

精读时优先看三件事:第一,论文使用哪些快速反馈量,例如关节状态、接触状态、力觉、触觉、深度或局部视觉;第二,反馈如何进入 MPC、WBC、策略网络、安全过滤器或恢复控制器;第三,输出是否直接影响机器人下一步动作,而不是只服务离线分析或高层任务规划。
## 方法详解

### System0 视角解析

从 System0 角度看,这篇论文的方法部分应当围绕 aerial-robotics、agile-locomotion、egocentric-perception、模仿学习、navigation、强化学习、鲁棒控制、状态估计 来读。核心问题是:系统如何从快速变化的机器人状态中构造控制输入,如何处理接触、碰撞、约束、模型误差或执行延迟,以及如何把这些信息变成下一时刻的动作。

如果方法里有优化问题,重点看目标函数、约束和求解频率;如果方法里有学习策略,重点看观测空间、动作空间、训练扰动和部署时的闭环频率;如果方法里有硬件或传感器,重点看传感链路是否真的进入控制回路。这样读可以把论文和 System2 语言规划、System1 技能编排区分开来。

### 关键公式与机制

#### 公式 1: [[动力学或策略机制]]

$$\tilde{\mathbf{R}}\in\mathbb{R}^{6}$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 2: [[优化目标/约束]]

$$\mathbf{R_{\mathcal{W}\mathcal{B}}}$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 3: [[动力学或策略机制]]

$$\mathbf{i}=\begin{bmatrix}\exp(-i_{c})+\cos(\alpha\cdot i_{c})\\ \exp(-i_{c})+\sin(\alpha\cdot i_{c})\\ \end{bmatrix}\;,$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 4: [[优化目标/约束]]

$$\dot{\boldsymbol{x}}=\begin{bmatrix}\dot{\boldsymbol{p}}_{\mathcal{W}\mathcal{B}}\\ \dot{\boldsymbol{q}}_{\mathcal{W}\mathcal{B}}\\ \dot{\boldsymbol{v}}_{\mathcal{W}}\\ \dot{\boldsymbol{\omega}}_{\mathcal{B}}\\ \dot{\boldsymbol{\Omega}}\end{bmatrix}=\begin{bmatrix}\boldsymbol{v}_{\mathcal{W}}\\ \boldsymbol{q}_{\mathcal{W}\mathcal{B}}\cdot\begin{bmatrix}0\\ \boldsymbol{\omega}_{\mathcal{B}}/2\end{bmatrix}\\ \frac{1}{m}\Big{(}\boldsymbol{q}_{\mathcal{W}\mathcal{B}}\odot(\boldsymbol{f}_{\text{prop}}+\boldsymbol{f}_{\text{aero}})\Big{)}+\boldsymbol{g}_{\mathcal{W}}\\ \boldsymbol{J}^{-1}\big{(}\boldsymbol{\tau}_{\text{prop}}+\boldsymbol{\tau}_{\text{aero}}-\boldsymbol{\omega}_{\mathcal{B}}\times\boldsymbol{J}\boldsymbol{\omega}_{\mathcal{B}}\big{)}\\ \frac{1}{k_{\text{mot}}}\big{(}\boldsymbol{\Omega}_{\text{ss}}-\boldsymbol{\Omega}\big{)}\end{bmatrix}\;,$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 5: [[动力学或策略机制]]

$$=\lambda_{2}\exp\left(-\delta_{\text{cam}}^{4}\right)$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 6: [[动力学或策略机制]]

$$=\begin{cases}1.0-d_{t},&\text{if passed gate at current timestep}\\ 0,&\text{otherwise}\end{cases}$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 7: [[动力学或策略机制]]

$$=\lambda_{1}\left(d_{t-1}-d_{t}\right)$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 8: [[动力学或策略机制]]

$$=\begin{cases}-4.0,&\text{if $p_{z}<0$ or in collision with gate}\\ 0,&\text{otherwise}\end{cases}$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 9: [[动力学或策略机制]]

$$\mathbf{s}=[\mathbf{p},\tilde{\mathbf{R}},\mathbf{v},\boldsymbol{\omega},\mathbf{i},\mathbf{d}]$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 10: [[动力学或策略机制]]

$$=\lambda_{3}|| $\boldsymbol{a}_{t}$ ||+\lambda_{4}|| $\boldsymbol{a}_{t}-\boldsymbol{a}_{t-1}$ ||^{2}$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。
## 关键图表

### Figure 1: The architecture of our method consists of a gate detector, which is trained to segmen

![Figure 1](https://arxiv.org/html/2406.12505v1/x2.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“The architecture of our method consists of a gate detector, which is trained to segmen”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 2: The top row shows the three different racetracks with trajectories flown by the pixel

![Figure 2](https://arxiv.org/html/2406.12505v1/x3.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“The top row shows the three different racetracks with trajectories flown by the pixel”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 3: Comparison of multiple rollouts with different initial conditions on an acyclic racetr

![Figure 3](https://arxiv.org/html/2406.12505v1/extracted/5675317/images/acyclic_full_comparison.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Comparison of multiple rollouts with different initial conditions on an acyclic racetr”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### 论文表格

#### Table 1: TABLE I: Simulation and HIL results. We compare the success rate (SR), mean-gate-passing-error (MGE), and the lap-time (| | | BEM | Augmented | HIL | | | | | | |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| | | SR | MGE | LT | SR | MGE | LT | SR | MGE | LT |
| Racetrack | Observation | [%] | [m] | [sec] | [%] | [m] | [sec] | [%] | [m] | [sec] |
| | State-based (Song et al. [63 ]) | 100.00 | 0.516 | 2.800 | 100.00 | 0.531 | 2.731 | 100.00 | 0.540 | 2.816 |
| | State-based (Song et al. [63 ]) +Perc | 100.00 | 0.199 | 2.810 | 100.00 | 0.196 | 2.729 | 100.00 | 0.221 | 2.817 |
| Ellipse | State-based (ours) | 100.00 | 0.296 | 2.846 | 100.00 | 0.219 | 2.756 | 100.00 | 0.389 | 2.570 |
| | Pixel-based (sym.) (ours) | 0.00 | 0.328 | - | 0.00 | 0.268 | - | 0.00 | - | - |
| | Pixel-based (asym.) (ours) | 93.75 | 0.350 | 3.072 | 90.60 | 0.154 | 2.902 | 100.00 | 0.381 | 3.318 |
| | State-based (Song et al. [63 ]) | 100.00 | 0.446 | 3.600 | 100.00 | 0.438 | 3.588 | 100.00 | 0.414 | 3.648 |
| | State-based (Song et al. [63 ]) +Perc | 100.00 | 0.190 | 4.819 | 100.00 | 0.180 | 4.727 | 100.00 | 0.184 | 4.900 |
| Figure-8 | State-based (ours) | 100.00 | 0.200 | 4.833 | 100.00 | 0.190 | 4.741 | 100.00 | 0.367 | 4.703 |
| | Pixel-based (sym.) (ours) | 0.00 | 0.409 | - | 0.00 | 0.378 | - | 0.00 | - | - |
| | Pixel-based (asym.) (ours) | 100.00 | 0.198 | 4.626 | 95.30 | 0.186 | 4.644 | 100.00 | 0.238 | 4.777 |
| | State-based (Song et al. [63 ]) | 100.00 | 0.471 | 3.479 | 100.00 | 0.451 | 3.497 | 100.00 | 0.455 | 3.570 |
| | State-based (Song et al. [63 ]) +Perc | 100.00 | 0.119 | 5.065 | 100.00 | 0.121 | 5.018 | 100.00 | 0.163 | 5.090 |
| Glasses | State-based (ours) | 100.00 | 0.151 | 5.102 | 100.00 | 0.163 | 4.985 | 100.00 | 0.191 | 5.157 |
| | Pixel-based (sym.) (ours) | 0.00 | 0.402 | - | 0.00 | 0.396 | - | 0.00 | - | - |
| | Pixel-based (asym.) (ours) | 100.00 | 0.240 | 5.267 | 89.10 | 0.209 | 5.162 | 100.00 | 0.273 | 5.586 |

**说明**: TABLE I: Simulation and HIL results. We compare the success rate (SR), mean-gate-passing-error (MGE), and the lap-time (LT) of our asymmetric pixel-based policy against four baselines. Three different racetracks are evaluated in simulation and HIL experiments, each with three laps. Pixel-based policies trained with our asymmetric actor-critic architecture perform significantly better than the symmetric architecture. We compare against the state-based approach of Song et al. [63 ] and a modified version, denoted by +Perc, which uses the same observations as [63 ] but with our perception-aware reward design. The best state-based result is and the best pixel-based result is bold.

#### Table 2: TABLE II: Results in the real world flying without state estimation for 6 trials with 3 laps each on the Figure-8 racetr

| | SR | MGE | LT |
| --- | --- | --- | --- |
| Observation | [%] | [m] | [sec] |
| State-based | 100.00 | 0.367 | 4.703 |
| Pixel-based (asym.) | 100.00 | 0.491 | 4.683 |

**说明**: TABLE II: Results in the real world flying without state estimation for 6 trials with 3 laps each on the Figure-8 racetrack. We compare the success rate (SR), mean-gate-passing-error (MGE), and the lap-time (LT) in the real-world using state-based and pixel-based observations.

#### Table 3: TABLE III: PPO hyperparameters to train state- and vision-based policies.

| Parameter | Value |
| --- | --- |
| learning rate | 3e-4 linear decay to 1e-5 |
| discount factor | 0.995 |
| GAE-   \lambda | 0.95 |
| learning epochs | 10 |
| clip range | 0.2 |
| entropy coefficient | 0.001 |
| batch size | 25000 |
| policy network MLP | [512, 512] |
| value network MLP | [512, 512] |
| CNN-encoder latent dimension | 256 |

**说明**: TABLE III: PPO hyperparameters to train state- and vision-based policies.
## 实验解读

- 评价重点:围绕 aerial-robotics、agile-locomotion、egocentric-perception、模仿学习、navigation,优先看真实机器人结果、扰动恢复、接触稳定性、控制误差、成功率、实时性和 sim-to-real 差距。
- System0 视角:实验结论不只看最终成功率,还要看闭环过程中是否能处理状态估计误差、执行延迟、碰撞/接触突变和外部扰动。
- 与上层系统的关系:如果方法能在快闭环内稳定工作,它可以作为 System1 技能或 System2 规划的底层执行保障;如果只在离线或慢速设定下成立,则还不能直接作为身体级快闭环模块。
## 批判性思考

### 优点

- 论文把 aerial-robotics、agile-locomotion、egocentric-perception、模仿学习、navigation 放到机器人身体级闭环里讨论,和只做高层语义推理的工作相比,更接近真实机器人部署时会遇到的状态、动力学和安全约束。
- 若论文同时给出公式、系统结构图、硬件实验或消融结果,这些证据可以帮助判断方法是否真的能支撑毫秒到百毫秒级反应。

### 局限

- 需要重点核查控制频率、推理延迟、传感延迟、失败恢复、硬件磨损、未见场景泛化和安全边界;这些因素决定它能否成为可靠的 System0 模块。
- 对学习型方法,还要区分训练环境中的鲁棒性和真实部署中的鲁棒性,尤其是接触丰富任务、动态运动任务和人形机器人任务。
## 论文定位

- 研究对象:Demonstrating Agile Flight from Pixels without State Estimation。
- 关键词:aerial-robotics、agile-locomotion、egocentric-perception、模仿学习、navigation、强化学习、鲁棒控制、状态估计、visuomotor-policy。
- System0 关联:关注机器人身体级快反应链路中“状态/接触/局部感知 -> 约束/优化/策略 -> 动作”的短闭环,不把问题停留在语言规划或任务分解层。

## 速查卡片

> [!summary] Agile Flight Pixels
> - **论文**: https://www.roboticsproceedings.org/rss20/p082.pdf
> - **arXiv**: http://arxiv.org/abs/2406.12505v1
> - **arXiv HTML**: https://arxiv.org/html/2406.12505v1
