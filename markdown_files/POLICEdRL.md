---
title: "POLICEd RL: Learning Closed-Loop Robot Control Policies with Provable Satisfaction of Hard Constraints"
method_name: "POLICEd RL"
authors: ["Jean-Baptiste Bouvier"]
year: 2024
venue: "RSS"
tags: ["reinforcement-learning", "safe-control", "closed-loop-control"]
image_source: "online"
arxiv_html: "https://arxiv.org/html/2403.13297v3"
---
# POLICEd RL
## 一句话总结

> POLICEd RL: Learning Closed-Loop Robot Control Policies with Provable Satisfaction of Hard Constraints 主要落在 [[certified-control]]、[[closed-loop-control]]、[[policy-learning]]、[[强化学习]] 这一类问题上;从 System0 视角看,它的价值在于把身体状态、接触变化、局部感知或动力学约束接到机器人低层闭环里,让动作生成更稳定、安全、实时。
## 核心贡献

1. 围绕 **POLICEd RL: Learning Closed-Loop Robot Control Policies with Provable Satisfaction of Hard Constraints** 建立了一个与 certified-control、closed-loop-control、policy-learning、强化学习、safe-control 相关的机器人问题设定,重点不是高层语义规划,而是身体级闭环中的状态变化、接触变化或控制约束。
2. 方法层面的关键价值在于把感知、动力学、约束优化、策略学习或硬件反馈组织成可执行的低层动作生成机制,适合放进 System0 的快反应模块中理解。
3. 对系统集成而言,这类工作补上了 System1 技能选择与电机级控制之间的一层:它负责把上层目标变成动态可行、可恢复、可安全过滤的动作。
4. 若论文包含真实机器人、接触任务或仿真到真实迁移,核心应看它是否报告了闭环稳定性、扰动恢复、硬件延迟、失败模式和跨场景泛化。
## 问题背景

这篇论文可以放在 System0 的“具身快系统”背景下阅读:机器人需要在很短时间内根据本体状态、接触、力/触觉、局部视觉或动力学预测做出动作,而不是等待慢速规划器重新推理整个任务。certified-control、closed-loop-control、policy-learning、强化学习、safe-control 相关问题的难点通常不在“知道要做什么”,而在动作执行过程中状态会快速偏离、接触会突然改变、约束会实时收紧,低层系统必须及时修正。

### System0 关联

精读时优先看三件事:第一,论文使用哪些快速反馈量,例如关节状态、接触状态、力觉、触觉、深度或局部视觉;第二,反馈如何进入 MPC、WBC、策略网络、安全过滤器或恢复控制器;第三,输出是否直接影响机器人下一步动作,而不是只服务离线分析或高层任务规划。
## 方法详解

### System0 视角解析

从 System0 角度看,这篇论文的方法部分应当围绕 certified-control、closed-loop-control、policy-learning、强化学习、safe-control 来读。核心问题是:系统如何从快速变化的机器人状态中构造控制输入,如何处理接触、碰撞、约束、模型误差或执行延迟,以及如何把这些信息变成下一时刻的动作。

如果方法里有优化问题,重点看目标函数、约束和求解频率;如果方法里有学习策略,重点看观测空间、动作空间、训练扰动和部署时的闭环频率;如果方法里有硬件或传感器,重点看传感链路是否真的进入控制回路。这样读可以把论文和 System2 语言规划、System1 技能编排区分开来。

### 关键公式与机制

#### 公式 1: [[优化目标/约束]]

$$\theta^{*}=\ {\theta}{\arg\max}\ \mathcal{G}(\mu_{\theta})\quad\text{s.t.}\quad\dot{s}(t)=f\big{(}s(t),\mu_{\theta}(s(t))\big{)},\quad s(0)\sim\rho_{0},\quad Cs(t)\leq d,\quad\text{for all}\ t\geq 0.$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 2: [[优化目标/约束]]

$$\ {\theta}{\max}\,\mathcal{G}(\mu_{\theta}):=\ {s_{0}\sim\rho_{0}}{\mathbb{E}}\int_{0}^{\infty}\gamma^{t}R\big{(}s(t),\mu_{\theta}(s(t))\big{)}dt\quad\text{s.t.}\,$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 3: [[动力学或策略机制]]

$$\mu_{\theta}(s)=D_{\theta}s+e_{\theta}=D_{\theta}\left(\sum_{k=1}^{N}\alpha_{k}v_{k}\right)+e_{\theta}\left(\sum_{k=1}^{N}\alpha_{k}\right)=\sum_{k=1}^{N}\alpha_{k}(D_{\theta}v_{k}+e_{\theta}).$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 4: [[优化目标/约束]]

$$\varepsilon^{*}_{ABc}:=\min\left\{\varepsilon:\big{| }Cf(s,a)-C(As+Ba+c) $\big{$ |}\leq\varepsilon,\quad\text{for all}\ a\in\mathcal{A},\ \text{and all}\ s\in\mathcal{B}\right\}.$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 5: [[优化目标/约束]]

$$\max\big{\{}C\big{(}s_{j+1}-s_{j}\big{)}:s_{j}\in\mathcal{S}_{s},\ a_{j}=\mu_{\theta}(s_{j})\big{\}}\leq r,$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 6: [[优化目标/约束]]

$$\tau_{d}^{\mathcal{S}}(s_{0},\mu_{\theta}):=\big{(}s_{0},s_{1},\ldots\big{)}\in\mathcal{S}^{\mathbb{N}},$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 7: [[优化目标/约束]]

$$f_{\theta}(x)=\sum_{j=1}^{M}\big{(}D_{j}x+e_{j}\big{)}\mathbf{1}_{x\,\in\,\mathcal{R}_{j}},$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 8: [[优化目标/约束]]

$$\mathcal{B}=\big{\{}(x,\theta,\dot{x},\dot{\theta}):x\in[-0.9,0.9],\ \theta\in[0.1,0.2],\ \dot{x}\in[-1,1],\ \dot{\theta}\in[-r,0]\big{\}}$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 9: [[动力学或策略机制]]

$$=\sum_{k=1}^{N}\alpha_{k}C\big{(}Av_{k}+B\mu_{\theta}(v_{k})+c\big{)}\leq\sum_{k=1}^{N}\alpha_{k}(-\varepsilon)=-\varepsilon,$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 10: [[动力学或策略机制]]

$$\frac{\partial}{\partial a}\frac{\partial y(t)}{\partial t}=\frac{\partial}{\partial a}\frac{\partial Cs(t)}{\partial t}=\frac{\partial}{\partial a}C\dot{s}(t)=\frac{\partial}{\partial a}C\big{(}As(t)+Ba(t)+c\big{)}=CB.$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。
## 关键图表

### Figure 1: Pipeline / core system figure: The three categories of constraint satisfaction with increasing guarantees of satisfa

![Figure 1](https://arxiv.org/html/2403.13297v3/extracted/5980082/figures/ConstraintDefinitions.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Pipeline / core system figure: The three categories of constraint satisfaction with increasing guarantees of satisfa”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 2: Classification task of orange versus purple by a learned decision boundary (red) wh

![Figure 2](https://arxiv.org/html/2403.13297v3/extracted/5980082/figures/sine_wave.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Classification task of orange versus purple by a learned decision boundary (red) wh”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 3: State space S S $\mathcal{S}$ with arrows denoting state transitions unde

![Figure 3](https://arxiv.org/html/2403.13297v3/extracted/5980082/figures/toy2.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“State space S S $\mathcal{S}$ with arrows denoting state transitions unde”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### 论文表格

#### Table 1: Table 1: Metrics comparison for different methods based on a 500 episode deployment with the fully-trained policies on

| Models | Completion % | Completion % | | |
| --- | --- | --- | --- | --- |
| w/o Violation | | | | |
| Average reward | | | | |
| ± 95 plus-or-minus 95 \pm 95 ± 95 % CI | | | | |
| Average Constraint | | | | |
| Satisfaction ± 95 plus-or-minus 95 \pm 95 ± 95 % CI | | | | |
| TD3 trained and | | | | |
| evaluated w/o penalty | | | | |
| | | | | |
| POLICEd (ours) | 93.4 93.4 93.4 93.4 | 93.4 93.4  $\mathbf{93.4}.4$ | - 16.22 ± 0.68 plus-or-minus 16.22 0.68  $\mathbf{-16.22\pm 0.68} -.22 ±.68$ | 100 ± 0.0 plus-or-minus 100 0.0  $\mathbf{100\pm 0.0} ±.0$ |
| TD3 | 75.8 75.8 75.8 75.8 | 12.0 12.0 12.0 12.0 | - 45.20 ± 3.23 plus-or-minus 45.20 3.23 -45.20\pm 3.23 - 45.20 ± 3.23 | 28.4 ± 3.9 plus-or-minus 28.4 3.9 28.4\pm 3.9 28.4 ± 3.9 |
| CPO | 2.0 2.0 2.0 2.0 | 2.0 2.0 2.0 2.0 | - 96.71 ± 3.45 plus-or-minus 96.71 3.45 -96.71\pm 3.45 - 96.71 ± 3.45 | 89.9 ± 2.7 plus-or-minus 89.9 2.7 89.9\pm 2.7 89.9 ± 2.7 |
| PPO-Barrier | 100 100  $\mathbf{100}$ | 86.2 86.2 86.2 86.2 | - 41.26 ± - 2.30 -41.26\pm-2.30 - 41.26 ± - 2.30 | 86.2 ± 3.0 plus-or-minus 86.2 3.0 86.2\pm 3.0 86.2 ± 3.0 |
| [0.9pt/3pt] | | | | |
| POLICEd trained | | | | |
| w/o penalty | | | | |
| TD3 trained w/o penalty | 99.8 99.8 99.8 99.8 | 48.8 48.8 48.8 48.8 | - 45.69 ± 16.61 plus-or-minus 45.69 16.61 -45.69\pm 16.61 - 45.69 ± 16.61 | 53.4 ± 4.4 plus-or-minus 53.4 4.4 53.4\pm 4.4 53.4 ± 4.4 |
## 实验解读

- 评价重点:围绕 certified-control、closed-loop-control、policy-learning、强化学习、safe-control,优先看真实机器人结果、扰动恢复、接触稳定性、控制误差、成功率、实时性和 sim-to-real 差距。
- System0 视角:实验结论不只看最终成功率,还要看闭环过程中是否能处理状态估计误差、执行延迟、碰撞/接触突变和外部扰动。
- 与上层系统的关系:如果方法能在快闭环内稳定工作,它可以作为 System1 技能或 System2 规划的底层执行保障;如果只在离线或慢速设定下成立,则还不能直接作为身体级快闭环模块。
## 批判性思考

### 优点

- 论文把 certified-control、closed-loop-control、policy-learning、强化学习、safe-control 放到机器人身体级闭环里讨论,和只做高层语义推理的工作相比,更接近真实机器人部署时会遇到的状态、动力学和安全约束。
- 若论文同时给出公式、系统结构图、硬件实验或消融结果,这些证据可以帮助判断方法是否真的能支撑毫秒到百毫秒级反应。

### 局限

- 需要重点核查控制频率、推理延迟、传感延迟、失败恢复、硬件磨损、未见场景泛化和安全边界;这些因素决定它能否成为可靠的 System0 模块。
- 对学习型方法,还要区分训练环境中的鲁棒性和真实部署中的鲁棒性,尤其是接触丰富任务、动态运动任务和人形机器人任务。
## 论文定位

- 研究对象:POLICEd RL: Learning Closed-Loop Robot Control Policies with Provable Satisfaction of Hard Constraints。
- 关键词:certified-control、closed-loop-control、policy-learning、强化学习、safe-control。
- System0 关联:关注机器人身体级快反应链路中“状态/接触/局部感知 -> 约束/优化/策略 -> 动作”的短闭环,不把问题停留在语言规划或任务分解层。

## 速查卡片

> [!summary] POLICEd RL
> - **论文**: https://www.roboticsproceedings.org/rss20/p104.pdf
> - **arXiv**: http://arxiv.org/abs/2403.13297v3
> - **arXiv HTML**: https://arxiv.org/html/2403.13297v3
