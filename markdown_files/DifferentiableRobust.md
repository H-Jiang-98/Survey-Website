---
title: "Differentiable Robust Model Predictive Control"
method_name: "Differentiable Robust Predictive"
authors: ["Alex Oshin"]
year: 2024
venue: "RSS"
tags: ["robust-control", "real-time-control", "model-predictive-control", "trajectory-optimization"]
image_source: "online"
arxiv_html: "https://arxiv.org/html/2308.08426v3"
---
# Differentiable Robust Predictive
## 一句话总结

> Differentiable Robust Model Predictive Control 主要落在 [[模型预测控制]]、[[实时控制]]、[[鲁棒控制]]、[[轨迹优化]] 这一类问题上;从 System0 视角看,它的价值在于把身体状态、接触变化、局部感知或动力学约束接到机器人低层闭环里,让动作生成更稳定、安全、实时。
## 核心贡献

1. 围绕 **Differentiable Robust Model Predictive Control** 建立了一个与 模型预测控制、实时控制、鲁棒控制、轨迹优化 相关的机器人问题设定,重点不是高层语义规划,而是身体级闭环中的状态变化、接触变化或控制约束。
2. 方法层面的关键价值在于把感知、动力学、约束优化、策略学习或硬件反馈组织成可执行的低层动作生成机制,适合放进 System0 的快反应模块中理解。
3. 对系统集成而言,这类工作补上了 System1 技能选择与电机级控制之间的一层:它负责把上层目标变成动态可行、可恢复、可安全过滤的动作。
4. 若论文包含真实机器人、接触任务或仿真到真实迁移,核心应看它是否报告了闭环稳定性、扰动恢复、硬件延迟、失败模式和跨场景泛化。
## 问题背景

这篇论文可以放在 System0 的“具身快系统”背景下阅读:机器人需要在很短时间内根据本体状态、接触、力/触觉、局部视觉或动力学预测做出动作,而不是等待慢速规划器重新推理整个任务。模型预测控制、实时控制、鲁棒控制、轨迹优化 相关问题的难点通常不在“知道要做什么”,而在动作执行过程中状态会快速偏离、接触会突然改变、约束会实时收紧,低层系统必须及时修正。

### System0 关联

精读时优先看三件事:第一,论文使用哪些快速反馈量,例如关节状态、接触状态、力觉、触觉、深度或局部视觉;第二,反馈如何进入 MPC、WBC、策略网络、安全过滤器或恢复控制器;第三,输出是否直接影响机器人下一步动作,而不是只服务离线分析或高层任务规划。
## 方法详解

### System0 视角解析

从 System0 角度看,这篇论文的方法部分应当围绕 模型预测控制、实时控制、鲁棒控制、轨迹优化 来读。核心问题是:系统如何从快速变化的机器人状态中构造控制输入,如何处理接触、碰撞、约束、模型误差或执行延迟,以及如何把这些信息变成下一时刻的动作。

如果方法里有优化问题,重点看目标函数、约束和求解频率;如果方法里有学习策略,重点看观测空间、动作空间、训练扰动和部署时的闭环频率;如果方法里有硬件或传感器,重点看传感链路是否真的进入控制回路。这样读可以把论文和 System2 语言规划、System1 技能编排区分开来。

### 关键公式与机制

#### 公式 1: [[动力学或策略机制]]

$$\begin{aligned} \boldsymbol{\bar{\tau}}(\bar{\theta})=\operatorname*{arg\,min}_{\boldsymbol{\tau}}\bar{J}(\boldsymbol{\tau},\bar{\theta}):={}&\sum_{k=0}^{N-1}\bar{\ell}(x_{k},u_{k},\bar{\theta})+\bar{\phi}(x_{N},\bar{\theta}),\end{aligned}$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 2: [[动力学或策略机制]]

$$\begin{aligned} \boldsymbol{\tau}^{*}(\theta)=\operatorname*{arg\,min}_{\boldsymbol{\tau}}J(\boldsymbol{\tau},\theta,t):=&\sum_{k=0}^{N-1}\ell(x_{k},u_{k},\bar{x}_{k},\bar{u}_{k},\theta)\\ &\quad+\phi(x_{N},\bar{x}_{N},\theta),\end{aligned}$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 3: [[优化目标/约束]]

$$\begin{aligned} \mathcal{L}(\boldsymbol{z},\theta)={}&\sum_{k=0}^{N-1}\ell(x_{k},u_{k},\theta)+\lambda_{k+1}^{\top}(f(x_{k},u_{k},\theta)-x_{k+1})\\ &+\lambda_{0}^{\top}(\xi(\theta)-x_{0})+\phi(x_{N},\theta),\end{aligned}$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 4: [[优化目标/约束]]

$$\begin{aligned} \mathcal{L}(\boldsymbol{z},\theta)={}&\sum_{t=0}^{N-1}\ell(x_{k},u_{k},\theta)+\lambda_{k+1}^{\top}(f(x_{k},u_{k},\theta)-x_{k+1})\\ &+\lambda_{0}^{\top}(\xi(\theta)-x_{0})+\phi(x_{N},\theta),\end{aligned}$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 5: [[优化目标/约束]]

$$\begin{split}-\frac{\partial x_{0}}{\partial \theta}&=-\xi_{\theta},\\ -\frac{\partial \lambda_{k}}{\partial \theta}+\mathcal{L}_{xx}^{(k)}\frac{\partial x_{k}}{\partial \theta}+\mathcal{L}_{xu}^{(k)}\frac{\partial u_{k}}{\partial \theta}+f_{x_{k}}^{\top}\frac{\partial \lambda_{k+1}}{\partial \theta}&=-\mathcal{L}_{x\theta}^{(k)},\\ \mathcal{L}_{ux}^{(k)}\frac{\partial x_{k}}{\partial \theta}+\mathcal{L}_{uu}^{(k)}\frac{\partial u_{k}}{\partial \theta}+f_{u_{k}}^{\top}\frac{\partial \lambda_{k+1}}{\partial \theta}&=-\mathcal{L}_{u\theta}^{(k)},\\ f_{x_{k}}\frac{\partial x_{k}}{\partial \theta}+f_{u_{k}}\frac{\partial u_{k}}{\partial \theta}-\frac{\partial x_{k+1}}{\partial \theta}&=-f_{\theta_{k}},\\ -\frac{\partial \lambda_{N}}{\partial \theta}+\phi_{xx}\frac{\partial x_{N}}{\partial \theta}&=-\phi_{x\theta},\end{split}$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 6: [[动力学或策略机制]]

$$\begin{aligned} \leq{}&\zeta\sum_{t=0}^{k-1}D_{k,t}\lVert \hat{\tau}_{t}-\tau_{t}^{*} \rVert+\eta\sum_{t=0}^{k}C_{k,t}\lVert \hat{\tau}_{t}-\tau_{t}^{*} \rVert\\ &+(\nu_{k}A+\mu_{k}B+N)\lVert \hat{\tau}_{k}-\tau_{k}^{*} \rVert\end{aligned}$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 7: [[动力学或策略机制]]

$$\begin{aligned} \leq{}&\sum_{t=0}^{k-1}\ {(\zeta D_{k,t}+\eta C_{k,t})}_{:=D_{k+1,t}}\lVert \hat{\tau}_{t}-\tau_{t}^{*} \rVert\\ &+\ {(\nu_{k}A+\mu_{k}B+N+\eta C_{k,k})}_{D_{k+1,k}}\lVert \hat{\tau}_{k}-\tau_{k}^{*} \rVert\end{aligned}$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 8: [[动力学或策略机制]]

$$\begin{aligned} \boldsymbol{\tau}^{*}=\operatorname*{arg\,min}_{\boldsymbol{\tau}}J(\boldsymbol{\tau},t):={}&\sum_{k=0}^{N-1}\ell(x_{k},u_{k},\bar{x}_{k},\bar{u}_{k})\\ &\quad+\phi(x_{N},\bar{x}_{N}),\end{aligned}$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 9: [[动力学或策略机制]]

$$\begin{aligned} ={}&\hat{Q}_{uu}^{-1}\left((Q_{u\theta}-\hat{Q}_{u\theta})+\hat{Q}_{ux}(\partial x_{k}^{*}-\partial\hat{x}_{k})+(Q_{ux}-\hat{Q}_{ux})\partial x_{k}^{*}\right)\\ &\quad+(Q_{uu}^{-1}-\hat{Q}_{uu}^{-1})(Q_{u\theta}+Q_{ux}\partial x_{k}^{*}),\end{aligned}$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 10: [[动力学或策略机制]]

$$\begin{aligned} \lVert \partial\hat{u}_{k}-\partial u_{k}^{*} \rVert\leq{}&\ {(\frac{M}{\alpha}+\frac{\nu_{k}L}{\alpha}+\frac{K(\gamma+\beta\nu_{k})}{\alpha^{2}})}_{=:= C_{k,k}}\lVert \hat{\tau}_{k}-\tau_{k}^{*} \rVert\\ &\quad+\frac{\beta}{\alpha}\lVert \partial\hat{x}_{k}-\partial x_{k}^{*} \rVert\end{aligned}$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。
## 关键图表

### Figure 1: Proposed differentiable robust MPC architecture. Orange dashed arrows show how gradie

![Figure 1](https://arxiv.org/html/2308.08426v3/extracted/5757086/figures/architecture.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Proposed differentiable robust MPC architecture. Orange dashed arrows show how gradie”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 2: Controlled quadrotor trajectories subject to large disturbances. 50 trajectories are

![Figure 2](https://arxiv.org/html/2308.08426v3/extracted/5757086/figures/quadrotor_comparison.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Controlled quadrotor trajectories subject to large disturbances. 50 trajectories are”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 3: Controlled Dubins vehicle trajectories subject to large noise. NT-MPC trajectories di

![Figure 3](https://arxiv.org/html/2308.08426v3/extracted/5757086/figures/dubins_comparison.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Controlled Dubins vehicle trajectories subject to large noise. NT-MPC trajectories di”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### 论文表格

#### Table 1: TABLE I: Success and safety violation percentage for each algorithm 50 trials per task. For successes, higher is b

| | Dubins Vehicle | Quadrotor | Robot Arm | Cheetah | Quadruped | | | | | |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| | Successes | Violations | Successes | Violations | Successes | Violations | Successes | Violations | Successes | Violations |
| NT-MPC | 14% | 0% | 14% | 20% | 0% | 56% | 26% | 4% | 20% | 0% |
| DT-MPC (ours) | 100% | 0% | 76% | 4% | 78% | 10% | 70% | 0% | 64% | 0% |

**说明**: TABLE I: Success and safety violation percentage for each algorithm 50 trials per task. For successes, higher is better, while for violations, lower is better. Success is defined as arriving close to a target state, while a violation is defined as colliding with an obstacle or violating the safety constraints. The magnitude of disturbances are upper-bounded by 0.05 0.05 0.05 0.05, 0.1 0.1 0.1 0.1, 0.1 0.1 0.1 0.1, 0.05 0.05 0.05 0.05 and 0.05 0.05 0.05 0.05 for each system, respectively.
## 实验解读

- 评价重点:围绕 模型预测控制、实时控制、鲁棒控制、轨迹优化,优先看真实机器人结果、扰动恢复、接触稳定性、控制误差、成功率、实时性和 sim-to-real 差距。
- System0 视角:实验结论不只看最终成功率,还要看闭环过程中是否能处理状态估计误差、执行延迟、碰撞/接触突变和外部扰动。
- 与上层系统的关系:如果方法能在快闭环内稳定工作,它可以作为 System1 技能或 System2 规划的底层执行保障;如果只在离线或慢速设定下成立,则还不能直接作为身体级快闭环模块。
## 批判性思考

### 优点

- 论文把 模型预测控制、实时控制、鲁棒控制、轨迹优化 放到机器人身体级闭环里讨论,和只做高层语义推理的工作相比,更接近真实机器人部署时会遇到的状态、动力学和安全约束。
- 若论文同时给出公式、系统结构图、硬件实验或消融结果,这些证据可以帮助判断方法是否真的能支撑毫秒到百毫秒级反应。

### 局限

- 需要重点核查控制频率、推理延迟、传感延迟、失败恢复、硬件磨损、未见场景泛化和安全边界;这些因素决定它能否成为可靠的 System0 模块。
- 对学习型方法,还要区分训练环境中的鲁棒性和真实部署中的鲁棒性,尤其是接触丰富任务、动态运动任务和人形机器人任务。
## 论文定位

- 研究对象:Differentiable Robust Model Predictive Control。
- 关键词:模型预测控制、实时控制、鲁棒控制、轨迹优化。
- System0 关联:关注机器人身体级快反应链路中“状态/接触/局部感知 -> 约束/优化/策略 -> 动作”的短闭环,不把问题停留在语言规划或任务分解层。

## 速查卡片

> [!summary] Differentiable Robust Predictive
> - **论文**: https://www.roboticsproceedings.org/rss20/p003.pdf
> - **arXiv**: http://arxiv.org/abs/2308.08426v3
> - **arXiv HTML**: https://arxiv.org/html/2308.08426v3
