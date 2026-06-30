---
title: "FlashSAC: Fast and Stable Off-Policy Reinforcement Learning for High-Dimensional Robot Control"
method_name: "FlashSAC"
authors: ["Donghu Kim"]
year: 2026
venue: "RSS"
tags: ["legged-locomotion", "reinforcement-learning", "imitation-learning", "robot-generalization", "humanoid", "sim-to-real"]
image_source: "online"
arxiv_html: "https://arxiv.org/html/2604.04539v2"
---
# FlashSAC
## 一句话总结

> FlashSAC: Fast and Stable Off-Policy Reinforcement Learning for High-Dimensional Robot Control 主要落在 [[人形机器人]]、[[模仿学习]]、[[足式运动]]、[[policy-learning]] 这一类问题上;从 System0 视角看,它的价值在于把身体状态、接触变化、局部感知或动力学约束接到机器人低层闭环里,让动作生成更稳定、安全、实时。
## 核心贡献

1. 围绕 **FlashSAC: Fast and Stable Off-Policy Reinforcement Learning for High-Dimensional Robot Control** 建立了一个与 人形机器人、模仿学习、足式运动、policy-learning、强化学习、scalable-robot-learning 相关的机器人问题设定,重点不是高层语义规划,而是身体级闭环中的状态变化、接触变化或控制约束。
2. 方法层面的关键价值在于把感知、动力学、约束优化、策略学习或硬件反馈组织成可执行的低层动作生成机制,适合放进 System0 的快反应模块中理解。
3. 对系统集成而言,这类工作补上了 System1 技能选择与电机级控制之间的一层:它负责把上层目标变成动态可行、可恢复、可安全过滤的动作。
4. 若论文包含真实机器人、接触任务或仿真到真实迁移,核心应看它是否报告了闭环稳定性、扰动恢复、硬件延迟、失败模式和跨场景泛化。
## 问题背景

这篇论文可以放在 System0 的“具身快系统”背景下阅读:机器人需要在很短时间内根据本体状态、接触、力/触觉、局部视觉或动力学预测做出动作,而不是等待慢速规划器重新推理整个任务。人形机器人、模仿学习、足式运动、policy-learning、强化学习、scalable-robot-learning、仿真到真实迁移 相关问题的难点通常不在“知道要做什么”,而在动作执行过程中状态会快速偏离、接触会突然改变、约束会实时收紧,低层系统必须及时修正。

### System0 关联

精读时优先看三件事:第一,论文使用哪些快速反馈量,例如关节状态、接触状态、力觉、触觉、深度或局部视觉;第二,反馈如何进入 MPC、WBC、策略网络、安全过滤器或恢复控制器;第三,输出是否直接影响机器人下一步动作,而不是只服务离线分析或高层任务规划。
## 方法详解

### System0 视角解析

从 System0 角度看,这篇论文的方法部分应当围绕 人形机器人、模仿学习、足式运动、policy-learning、强化学习、scalable-robot-learning、仿真到真实迁移、visuomotor-policy 来读。核心问题是:系统如何从快速变化的机器人状态中构造控制输入,如何处理接触、碰撞、约束、模型误差或执行延迟,以及如何把这些信息变成下一时刻的动作。

如果方法里有优化问题,重点看目标函数、约束和求解频率;如果方法里有学习策略,重点看观测空间、动作空间、训练扰动和部署时的闭环频率;如果方法里有硬件或传感器,重点看传感链路是否真的进入控制回路。这样读可以把论文和 System2 语言规划、System1 技能编排区分开来。

### 关键公式与机制

#### 公式 1: [[优化目标/约束]]

$$
\mathcal{L}_{Q}=\mathbb{E}_{(s,a,r,s^{\prime})\sim\mathcal{D}}\left[\left(Q_{\theta}(s,a)-\left(r+\gamma Q_{\theta}(s^{\prime},a^{\prime})\right)\right)^{2}\right],
$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 2: [[优化目标/约束]]

$$
\mathcal{L}_{\pi}(\theta)=\mathbb{E}_{s\sim\mathcal{D},\,a\sim\pi_{\theta}}\big(\alpha\log\pi_{\theta}(a|s)-\min_{i=1,2}Q_{\phi_{i}}(s,a)\big),
$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 3: [[优化目标/约束]]

$$
\mathcal{L}_{Q}(\phi_{i})=\mathbb{E}_{(s,a,r,s^{\prime})\sim\mathcal{D}}\left[Q_{\phi_{i}}(s,a)-y\right]^{2},
$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 4: [[优化目标/约束]]

$$y=r+\gamma\big(\min_{j=1,2}Q_{\bar{\phi}_{j}}(s^{\prime},a^{\prime})-\alpha\log\pi_{\theta}(a^{\prime}| s^{\prime}) $\big),a^{\prime}\sim\pi_{\theta}(\cdot$ |s^{\prime}).$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 5: [[优化目标/约束]]

$$\bar{r}_{t}=\frac{r_{t}}{\max\!\left(\sqrt{\sigma_{t,G}^{2}+\epsilon},\;G_{t,\max}/G_{\max}\right)}.$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 6: [[优化目标/约束]]

$$\bar{\mathcal{H}}=\tfrac{1}{2}| $\mathcal{A}$ |\log\left(2\pi e\,\sigma_{\text{tgt}}^{2}\right),$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 7: [[动力学或策略机制]]

$$\textbf{o}_{t}=\begin{bmatrix}\boldsymbol{\omega}_{t}&\textbf{g}_{t}&\textbf{c}_{t}&\boldsymbol{q}_{t}&\boldsymbol{\dot{q}}_{t}&\textbf{a}_{t-1}\end{bmatrix}^{\top},$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 8: [[动力学或策略机制]]

$$\sum_{i=1}^{2}\lVert \boldsymbol{v}_{xy}^{\mathrm{ft}_{i}} \rVert\cdot\mathbf{1}[c^{\mathrm{ft}_{i}}]$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 9: [[动力学或策略机制]]

$$\sum_{i=1}^{2}\lvert\boldsymbol{\omega}_{z}^{\mathrm{ft}_{i}}\rvert\cdot\mathbf{1}[c^{\mathrm{ft}_{i}}]$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 10: [[优化目标/约束]]

$$\sum_{i=1}^{2}\min\bigl(\Delta\boldsymbol{v}_{z,\mathrm{ft}_{i}}^{2},\;1\bigr)$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。
## 关键图表

### Figure 1: $\textbf$ FlashSAC Architecture. The architecture consists of stacked inverted residua

![Figure 1](https://arxiv.org/html/2604.04539v2/x2.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“$\textbf$ FlashSAC Architecture. The architecture consists of stacked inverted residua”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 2: $\textbf$ Results on State-Based RL, CPU-based Simulators. Learning curves on select t

![Figure 2](https://arxiv.org/html/2604.04539v2/x4.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“$\textbf$ Results on State-Based RL, CPU-based Simulators. Learning curves on select t”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 3: $\textbf$ Results on Vision-Based RL. Learning curves on selected tasks from vision-ba

![Figure 3](https://arxiv.org/html/2604.04539v2/x5.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“$\textbf$ Results on Vision-Based RL. Learning curves on selected tasks from vision-ba”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### 论文表格

#### Table 1: Table 1: $\textbf$ IsaacLab environments. We evaluate 12 tasks from IsaacLab spanning gripper manipulation, dexterous ma

**说明**: 该表格来自论文中的公式、奖励项或实验设置；为避免 PDF 抽取噪声，网页中仅保留可渲染的核心符号和数值。

#### Table 2: Table 2: $\textbf$ MuJoCo Playground environments. We evaluate four humanoid locomotion tasks from MuJoCo Playground. No

**说明**: 该表格来自论文中的公式、奖励项或实验设置；为避免 PDF 抽取噪声，网页中仅保留可渲染的核心符号和数值。

#### Table 3: Table 3: $\textbf$ ManiSkill Environments. We evaluate 6 ManiSkill gripper-based manipulation environments. Normalized s

**说明**: 该表格来自论文中的公式、奖励项或实验设置；为避免 PDF 抽取噪声，网页中仅保留可渲染的核心符号和数值。

#### Table 4: Table 4: $\textbf$ Genesis environments. We evaluate 3 reinforcement learning tasks from Genesis. Normalized scores are

**说明**: 该表格来自论文中的公式、奖励项或实验设置；为避免 PDF 抽取噪声，网页中仅保留可渲染的核心符号和数值。

#### Table 5: Table 5: $\textbf$ MuJoCo Environments. We evaluate five standard MuJoCo environments. Normalized scores are computed re

**说明**: 该表格来自论文中的公式、奖励项或实验设置；为避免 PDF 抽取噪声，网页中仅保留可渲染的核心符号和数值。

#### Table 6: Table 6: $\textbf$ DMC Environments. We evaluate 10 DeepMind Control Suite tasks, including humanoid and dog embodiments

**说明**: 该表格来自论文中的公式、奖励项或实验设置；为避免 PDF 抽取噪声，网页中仅保留可渲染的核心符号和数值。

#### Table 7: Table 7: $\textbf$ HumanoidBench Environments. We evaluate 14 humanoid locomotion tasks without hand control from Humano

**说明**: 该表格来自论文中的公式、奖励项或实验设置；为避免 PDF 抽取噪声，网页中仅保留可渲染的核心符号和数值。

#### Table 8: Table 8: $\textbf$ Myosuite Environments. We evaluate 10 Myosuite environments. Normalized scores correspond to the maxi

**说明**: 该表格来自论文中的公式、奖励项或实验设置；为避免 PDF 抽取噪声，网页中仅保留可渲染的核心符号和数值。

#### Table 9: Table 9: $\textbf$ Hyperparameters (GPU-based Simulators). FlashSAC hyperparameters used in benchmarks that support mass

**说明**: 该表格来自论文中的公式、奖励项或实验设置；为避免 PDF 抽取噪声，网页中仅保留可渲染的核心符号和数值。

#### Table 10: Table 10: $\textbf$ Hyperparameters (CPU-based Simulators). FlashSAC hyperparameters used in benchmarks with no parallel

**说明**: 该表格来自论文中的公式、奖励项或实验设置；为避免 PDF 抽取噪声，网页中仅保留可渲染的核心符号和数值。

#### Table 11: Table 11: $\textbf$ Hyperparameters (Vision-Based RL). FlashSAC hyperparameters used in vision-based tasks. We list only

**说明**: 该表格来自论文中的公式、奖励项或实验设置；为避免 PDF 抽取噪声，网页中仅保留可渲染的核心符号和数值。

#### Table 12: Table 12: $\textbf$ Joint Information of Unitree G1 humanoid. Joint list of Unitree G1 29-DoF with default angle, stiffn

**说明**: 该表格来自论文中的公式、奖励项或实验设置；为避免 PDF 抽取噪声，网页中仅保留可渲染的核心符号和数值。

#### Table 13: Table 13: $\textbf$ Observation Space for Sim-to-Real Experiments. Our observation space combines proprioceptive informa

**说明**: 该表格来自论文中的公式、奖励项或实验设置；为避免 PDF 抽取噪声，网页中仅保留可渲染的核心符号和数值。

#### Table 14: Table 14: $\textbf$ Reward Configurations for Sim-to-Real Experiments. Both methods share the same reward structure, inc

**说明**: 该表格来自论文中的公式、奖励项或实验设置；为避免 PDF 抽取噪声，网页中仅保留可渲染的核心符号和数值。

#### Table 15: Table 15: $\textbf$ Notation for Reward Terms. Symbol definitions of the reward terms in Table 14.

**说明**: 该表格来自论文中的公式、奖励项或实验设置；为避免 PDF 抽取噪声，网页中仅保留可渲染的核心符号和数值。
## 实验解读

- 评价重点:围绕 人形机器人、模仿学习、足式运动、policy-learning、强化学习,优先看真实机器人结果、扰动恢复、接触稳定性、控制误差、成功率、实时性和 sim-to-real 差距。
- System0 视角:实验结论不只看最终成功率,还要看闭环过程中是否能处理状态估计误差、执行延迟、碰撞/接触突变和外部扰动。
- 与上层系统的关系:如果方法能在快闭环内稳定工作,它可以作为 System1 技能或 System2 规划的底层执行保障;如果只在离线或慢速设定下成立,则还不能直接作为身体级快闭环模块。
## 批判性思考

### 优点

- 论文把 人形机器人、模仿学习、足式运动、policy-learning、强化学习 放到机器人身体级闭环里讨论,和只做高层语义推理的工作相比,更接近真实机器人部署时会遇到的状态、动力学和安全约束。
- 若论文同时给出公式、系统结构图、硬件实验或消融结果,这些证据可以帮助判断方法是否真的能支撑毫秒到百毫秒级反应。

### 局限

- 需要重点核查控制频率、推理延迟、传感延迟、失败恢复、硬件磨损、未见场景泛化和安全边界;这些因素决定它能否成为可靠的 System0 模块。
- 对学习型方法,还要区分训练环境中的鲁棒性和真实部署中的鲁棒性,尤其是接触丰富任务、动态运动任务和人形机器人任务。
## 论文定位

- 研究对象:FlashSAC: Fast and Stable Off-Policy Reinforcement Learning for High-Dimensional Robot Control。
- 关键词:人形机器人、模仿学习、足式运动、policy-learning、强化学习、scalable-robot-learning、仿真到真实迁移、visuomotor-policy。
- System0 关联:关注机器人身体级快反应链路中“状态/接触/局部感知 -> 约束/优化/策略 -> 动作”的短闭环,不把问题停留在语言规划或任务分解层。

## 速查卡片

> [!summary] FlashSAC
> - **论文**: https://www.roboticsproceedings.org/rss22/p099.pdf
> - **arXiv**: http://arxiv.org/abs/2604.04539v2
> - **arXiv HTML**: https://arxiv.org/html/2604.04539v2
