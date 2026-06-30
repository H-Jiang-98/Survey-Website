---
title: "Distilling Contact Planning for Fast Trajectory Optimization in Robot Air Hockey"
method_name: "Distilling Contact Planning"
authors: ["Julius Jankowski"]
year: 2025
venue: "RSS"
tags: ["contact-reasoning", "real-time-control", "agile-locomotion", "model-predictive-control", "trajectory-optimization"]
image_source: "online"
arxiv_html: "https://arxiv.org/html/2407.03705v2"
---
# Distilling Contact Planning
## 一句话总结

> Distilling Contact Planning for Fast Trajectory Optimization in Robot Air Hockey 主要落在 [[agile-locomotion]]、[[接触推理]]、[[模型预测控制]]、[[实时控制]] 这一类问题上;从 System0 视角看,它的价值在于把身体状态、接触变化、局部感知或动力学约束接到机器人低层闭环里,让动作生成更稳定、安全、实时。
## 核心贡献

1. 围绕 **Distilling Contact Planning for Fast Trajectory Optimization in Robot Air Hockey** 建立了一个与 agile-locomotion、接触推理、模型预测控制、实时控制、轨迹优化 相关的机器人问题设定,重点不是高层语义规划,而是身体级闭环中的状态变化、接触变化或控制约束。
2. 方法层面的关键价值在于把感知、动力学、约束优化、策略学习或硬件反馈组织成可执行的低层动作生成机制,适合放进 System0 的快反应模块中理解。
3. 对系统集成而言,这类工作补上了 System1 技能选择与电机级控制之间的一层:它负责把上层目标变成动态可行、可恢复、可安全过滤的动作。
4. 若论文包含真实机器人、接触任务或仿真到真实迁移,核心应看它是否报告了闭环稳定性、扰动恢复、硬件延迟、失败模式和跨场景泛化。
## 问题背景

这篇论文可以放在 System0 的“具身快系统”背景下阅读:机器人需要在很短时间内根据本体状态、接触、力/触觉、局部视觉或动力学预测做出动作,而不是等待慢速规划器重新推理整个任务。agile-locomotion、接触推理、模型预测控制、实时控制、轨迹优化 相关问题的难点通常不在“知道要做什么”,而在动作执行过程中状态会快速偏离、接触会突然改变、约束会实时收紧,低层系统必须及时修正。

### System0 关联

精读时优先看三件事:第一,论文使用哪些快速反馈量,例如关节状态、接触状态、力觉、触觉、深度或局部视觉;第二,反馈如何进入 MPC、WBC、策略网络、安全过滤器或恢复控制器;第三,输出是否直接影响机器人下一步动作,而不是只服务离线分析或高层任务规划。
## 方法详解

### System0 视角解析

从 System0 角度看,这篇论文的方法部分应当围绕 agile-locomotion、接触推理、模型预测控制、实时控制、轨迹优化 来读。核心问题是:系统如何从快速变化的机器人状态中构造控制输入,如何处理接触、碰撞、约束、模型误差或执行延迟,以及如何把这些信息变成下一时刻的动作。

如果方法里有优化问题,重点看目标函数、约束和求解频率;如果方法里有学习策略,重点看观测空间、动作空间、训练扰动和部署时的闭环频率;如果方法里有硬件或传感器,重点看传感链路是否真的进入控制回路。这样读可以把论文和 System2 语言规划、System1 技能编排区分开来。

### 关键公式与机制

#### 公式 1: [[优化目标/约束]]

$$\mathrm{Pr}_{3}\left({}^{c}\dot{\boldsymbol{x}}_{k^{+}}^{p}|{}^{c}\dot{\boldsymbol{x}}_{k^{-}}^{p},{}^{c}\dot{\boldsymbol{x}}_{k}^{m}\right)=\mathcal{N}\left(\boldsymbol{\Theta}_{3}^{p}{}^{c}\dot{\boldsymbol{x}}_{k^{-}}^{p}\!+\!\boldsymbol{\Theta}_{3}^{m}{}^{c}\dot{\boldsymbol{x}}_{k}^{m}\!+\!\boldsymbol{\theta}_{3},\boldsymbol{\Sigma}_{3}\right).$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 2: [[优化目标/约束]]

$$\mathrm{Pr}\left(\mathrm{G}=1|\hat{\boldsymbol{s}}_{0^{-}},\boldsymbol{x}_{0}^{m},\dot{\boldsymbol{x}}_{0}^{m}\right)=\int_{\mathcal{X}_{\mathrm{goal}}}\mathrm{Pr}\left(\boldsymbol{x}^{p}_{k_{\mathrm{goal}}}\right)d\boldsymbol{x}^{p}_{k_{\mathrm{goal}}}.$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 3: [[优化目标/约束]]

$$\mathrm{Pr}_{2}\left({}^{c}\dot{\boldsymbol{x}}_{k+1}^{p}|{}^{c}\dot{\boldsymbol{x}}_{k}^{p}\right)=\mathcal{N}\left(\boldsymbol{\Theta}_{2}{}^{c}\dot{\boldsymbol{x}}_{k}^{p}+\boldsymbol{\theta}_{2},\boldsymbol{\Sigma}_{2}\right),$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 4: [[优化目标/约束]]

$$\mathrm{Pr}_{1}\left(\dot{\boldsymbol{x}}_{k+1}^{p}|\dot{\boldsymbol{x}}_{k}^{p}\right)=\mathcal{N}\left(\boldsymbol{\Theta}_{1}\dot{\boldsymbol{x}}_{k}^{p}+\boldsymbol{\theta}_{1},\boldsymbol{\Sigma}_{1}\right),$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 5: [[优化目标/约束]]

$$\mathrm{Pr}_{i}(\boldsymbol{y}_{i},\boldsymbol{\xi}_{i})=\mathcal{N}\left(\begin{pmatrix}\boldsymbol{\mu}_{\boldsymbol{y}_{i}}\\ \boldsymbol{\mu}_{\boldsymbol{\xi}_{i}}\end{pmatrix},\begin{pmatrix}\boldsymbol{\Sigma}_{\boldsymbol{y}_{i}}&\boldsymbol{\Sigma}_{\boldsymbol{y}_{i}\boldsymbol{\xi}_{i}}\\ \boldsymbol{\Sigma}_{\boldsymbol{y}_{i}\boldsymbol{\xi}_{i}}^{\scriptscriptstyle\top}&\boldsymbol{\Sigma}_{\boldsymbol{\xi}_{i}}\end{pmatrix}\right).$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 6: [[动力学或策略机制]]

$$\begin{split}\boldsymbol{A}_{i}=\begin{pmatrix}\boldsymbol{A}_{i,\boldsymbol{x}\boldsymbol{x}}&\boldsymbol{A}_{i,\boldsymbol{x}\dot{\boldsymbol{x}}}\\ \boldsymbol{0}&\boldsymbol{\Theta}_{i}\end{pmatrix};\;\boldsymbol{b}_{i}=\begin{pmatrix}\boldsymbol{0}\\ \boldsymbol{\theta}_{i}\end{pmatrix};\;\boldsymbol{Q}_{i}=\begin{pmatrix}\boldsymbol{0}&\boldsymbol{0}\\ \boldsymbol{0}&\boldsymbol{\Sigma}_{i}\end{pmatrix}.\end{split}$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 7: [[动力学或策略机制]]

$$\mathrm{Pr}\left(\mathrm{G}=1| $\hat{\boldsymbol{s}}_{0^{-}},\boldsymbol{x}_{0}^{m},\dot{\boldsymbol{x}}_{0}^{m}\right)\approx\frac{1}{N_{\mathrm{G}}}\sum_{n=1}^{N_{\mathrm{G}}}\mathrm{Pr}\left(\mathrm{G}=1$ |\boldsymbol{x}^{p}_{k_{\mathrm{goal}},n}\right),$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 8: [[优化目标/约束]]

$$\begin{split}v^{*}=&\max v\\ \text{s.t.}\quad&v\boldsymbol{e}_{u}=\boldsymbol{J}(\boldsymbol{q}_{0})\dot{\boldsymbol{q}}_{0},\\ &\dot{\boldsymbol{q}}_{0}\in\mathcal{\dot{Q}}.\end{split}$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 9: [[优化目标/约束]]

$$\mathrm{Pr}\left(\mathrm{G}=1|\boldsymbol{x}^{p}_{k}\right)=\begin{cases}1,\quad\mathrm{if}\;\boldsymbol{x}^{p}_{k}\in\mathcal{X}_{\mathrm{goal}}\\ 0,\quad\mathrm{else.}\end{cases}$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 10: [[动力学或策略机制]]

$$\begin{split}\boldsymbol{\Theta}_{i}&=\boldsymbol{\Sigma}_{\boldsymbol{y}_{i}\boldsymbol{\xi}_{i}}\boldsymbol{\Sigma}_{\boldsymbol{\xi}_{i}}^{-1},\\ \boldsymbol{\theta}_{i}&=\boldsymbol{\mu}_{\boldsymbol{y}_{i}}-\boldsymbol{\Sigma}_{\boldsymbol{y}_{i}\boldsymbol{\xi}_{i}}\boldsymbol{\Sigma}_{\boldsymbol{\xi}_{i}}^{-1}\boldsymbol{\mu}_{\boldsymbol{\xi}_{i}},\\ \boldsymbol{\Sigma}_{i}&=\boldsymbol{\Sigma}_{\boldsymbol{y}_{i}}-\boldsymbol{\Sigma}_{\boldsymbol{y}_{i}\boldsymbol{\xi}_{i}}\boldsymbol{\Sigma}_{\boldsymbol{\xi}_{i}}^{-1}\boldsymbol{\Sigma}_{\boldsymbol{y}_{i}\boldsymbol{\xi}_{i}}^{\scriptscriptstyle\top}.\end{split}$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。
## 关键图表

### Figure 1: The proposed control framework enables our robot to autonomously play matches of air h

![Figure 1](https://arxiv.org/html/2407.03705v2/extracted/6445524/figures/title_airhockey_figure_adj.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“The proposed control framework enables our robot to autonomously play matches of air h”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 2: Overview of the interplay between puck state estimation ∙ ∙ $\bullet$ ∙ and robot contro

![Figure 2](https://arxiv.org/html/2407.03705v2/extracted/6445524/figures/ah_diagram.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Overview of the interplay between puck state estimation ∙ ∙ $\bullet$ ∙ and robot contro”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 3: The automated experimental setup consists of a robot placing the puck at a pre-defined

![Figure 3](https://arxiv.org/html/2407.03705v2/extracted/6445524/figures/ah_exp_setup.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“The automated experimental setup consists of a robot placing the puck at a pre-defined”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### 论文表格

#### Table 1: TABLE I: Results of the simulated experiments.

| | Score | Puck Speed [m s ] m s \left[\frac{\text{m}}{\text{s}}\right] [divide m s ] (mean $\pm$ std.) | Num. Banks (mean) |
| --- | --- | --- | --- |
| CB | 0.51 | 0.52 $\pm$ 0.24 | 0.00 |
| Atacom | 0.90 | 0.55 $\pm$ 0.05 | 0.00 |
| Ours #1 | 0.93 | 1.00 $\pm$ 0.20 | 0.00 |
| Ours #2 | 0.80 | 1.44 $\pm$ 0.63 | 0.53 |
| Ours #3 | 0.61 | 1.97 $\pm$ 0.49 | 1.13 |

**说明**: TABLE I: Results of the simulated experiments.

#### Table 2: TABLE II: Results of the real-world experiments.

| | Score | Puck Speed [m s ] m s \left[\frac{\text{m}}{\text{s}}\right] [divide m s ] (mean $\pm$ std.) | Num. Banks (mean) |
| --- | --- | --- | --- |
| CB | 0.49 | 1.09 $\pm$ 0.24 | 0.00 |
| Atacom | 0.13 | 0.66 $\pm$ 0.15 | 0.31 |
| Ours #1 | 0.78 | 1.72 $\pm$ 0.20 | 0.00 |
| Ours #2 | 0.60 | 2.02 $\pm$ 0.35 | 0.37 |
| Ours #3 | 0.31 | 2.37 $\pm$ 0.50 | 0.90 |

**说明**: TABLE II: Results of the real-world experiments.

#### Table 3: TABLE III: Reduced Action Space (R. Act.) v. Full Action Space (F. Act.).

| | Score | Puck Speed [m s ] m s \left[\frac{\text{m}}{\text{s}}\right] [divide m s ] (mean $\pm$ std.) | Num. Banks (mean) |
| --- | --- | --- | --- |
| R. Act. | 0.80 | 1.44 $\pm$ 0.63 | 0.53 |
| F. Act. | 0.51 | 0.96 $\pm$ 0.13 | 0.00 |

**说明**: TABLE III: Reduced Action Space (R. Act.) v. Full Action Space (F. Act.).
## 实验解读

- 评价重点:围绕 agile-locomotion、接触推理、模型预测控制、实时控制、轨迹优化,优先看真实机器人结果、扰动恢复、接触稳定性、控制误差、成功率、实时性和 sim-to-real 差距。
- System0 视角:实验结论不只看最终成功率,还要看闭环过程中是否能处理状态估计误差、执行延迟、碰撞/接触突变和外部扰动。
- 与上层系统的关系:如果方法能在快闭环内稳定工作,它可以作为 System1 技能或 System2 规划的底层执行保障;如果只在离线或慢速设定下成立,则还不能直接作为身体级快闭环模块。
## 批判性思考

### 优点

- 论文把 agile-locomotion、接触推理、模型预测控制、实时控制、轨迹优化 放到机器人身体级闭环里讨论,和只做高层语义推理的工作相比,更接近真实机器人部署时会遇到的状态、动力学和安全约束。
- 若论文同时给出公式、系统结构图、硬件实验或消融结果,这些证据可以帮助判断方法是否真的能支撑毫秒到百毫秒级反应。

### 局限

- 需要重点核查控制频率、推理延迟、传感延迟、失败恢复、硬件磨损、未见场景泛化和安全边界;这些因素决定它能否成为可靠的 System0 模块。
- 对学习型方法,还要区分训练环境中的鲁棒性和真实部署中的鲁棒性,尤其是接触丰富任务、动态运动任务和人形机器人任务。
## 论文定位

- 研究对象:Distilling Contact Planning for Fast Trajectory Optimization in Robot Air Hockey。
- 关键词:agile-locomotion、接触推理、模型预测控制、实时控制、轨迹优化。
- System0 关联:关注机器人身体级快反应链路中“状态/接触/局部感知 -> 约束/优化/策略 -> 动作”的短闭环,不把问题停留在语言规划或任务分解层。

## 速查卡片

> [!summary] Distilling Contact Planning
> - **论文**: https://www.roboticsproceedings.org/rss21/p115.pdf
> - **arXiv**: http://arxiv.org/abs/2407.03705v2
> - **arXiv HTML**: https://arxiv.org/html/2407.03705v2
