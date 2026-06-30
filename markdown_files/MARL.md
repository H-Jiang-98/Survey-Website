---
title: "Solving Multi-Agent Safe Optimal Control with Distributed Epigraph Form MARL"
method_name: "MARL"
authors: ["Songyuan Zhang"]
year: 2025
venue: "RSS"
tags: ["reinforcement-learning", "safe-control", "robot-generalization", "trajectory-optimization"]
image_source: "online"
arxiv_html: "https://arxiv.org/html/2504.15425v1"
---
# MARL
## 一句话总结

> Solving Multi-Agent Safe Optimal Control with Distributed Epigraph Form MARL 主要落在 [[强化学习]]、[[safe-control]]、[[scalable-robot-learning]]、[[轨迹优化]] 这一类问题上;从 System0 视角看,它的价值在于把身体状态、接触变化、局部感知或动力学约束接到机器人低层闭环里,让动作生成更稳定、安全、实时。
## 核心贡献

1. 围绕 **Solving Multi-Agent Safe Optimal Control with Distributed Epigraph Form MARL** 建立了一个与 强化学习、safe-control、scalable-robot-learning、轨迹优化 相关的机器人问题设定,重点不是高层语义规划,而是身体级闭环中的状态变化、接触变化或控制约束。
2. 方法层面的关键价值在于把感知、动力学、约束优化、策略学习或硬件反馈组织成可执行的低层动作生成机制,适合放进 System0 的快反应模块中理解。
3. 对系统集成而言,这类工作补上了 System1 技能选择与电机级控制之间的一层:它负责把上层目标变成动态可行、可恢复、可安全过滤的动作。
4. 若论文包含真实机器人、接触任务或仿真到真实迁移,核心应看它是否报告了闭环稳定性、扰动恢复、硬件延迟、失败模式和跨场景泛化。
## 问题背景

这篇论文可以放在 System0 的“具身快系统”背景下阅读:机器人需要在很短时间内根据本体状态、接触、力/触觉、局部视觉或动力学预测做出动作,而不是等待慢速规划器重新推理整个任务。强化学习、safe-control、scalable-robot-learning、轨迹优化 相关问题的难点通常不在“知道要做什么”,而在动作执行过程中状态会快速偏离、接触会突然改变、约束会实时收紧,低层系统必须及时修正。

### System0 关联

精读时优先看三件事:第一,论文使用哪些快速反馈量,例如关节状态、接触状态、力觉、触觉、深度或局部视觉;第二,反馈如何进入 MPC、WBC、策略网络、安全过滤器或恢复控制器;第三,输出是否直接影响机器人下一步动作,而不是只服务离线分析或高层任务规划。
## 方法详解

### System0 视角解析

从 System0 角度看,这篇论文的方法部分应当围绕 强化学习、safe-control、scalable-robot-learning、轨迹优化 来读。核心问题是:系统如何从快速变化的机器人状态中构造控制输入,如何处理接触、碰撞、约束、模型误差或执行延迟,以及如何把这些信息变成下一时刻的动作。

如果方法里有优化问题,重点看目标函数、约束和求解频率;如果方法里有学习策略,重点看观测空间、动作空间、训练扰动和部署时的闭环频率;如果方法里有硬件或传感器,重点看传感链路是否真的进入控制回路。这样读可以把论文和 System2 语言规划、System1 技能编排区分开来。

### 关键公式与机制

#### 公式 1: [[优化目标/约束]]

$$
\mathcal{L}_{\pi}(\theta)=\frac{1}{MN}\sum_{k=1}^{M}\sum_{i=1}^{N}\Bigg{[}\min\Bigg{\{}\frac{\pi_{\theta}(o_{i}^{k},z^{k})}{\pi_{\mathrm{old}}(o_{i}^{k},z^{k})}A_{i}(x^{k},z^{k}),\mathrm{clip}\left(\frac{\pi_{\theta}(o_{i}^{k},z^{k})}{\pi_{\mathrm{old}}(o_{i}^{k},z^{k})},1-\epsilon_{\mathrm{clip}},1+\epsilon_{\mathrm{clip}}\right)A_{i}(x^{k},z^{k})\Bigg{\}}\Bigg{]}.
$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 2: [[优化目标/约束]]

$$l(x,u)=\frac{1}{N}\sum_{j=1}^{N}\min_{i\in\mathcal{V}_{a}}\left(0.01\lVert p_{i}-p_{j}^{\mathrm{goal}} \rVert+0.001\mathrm{sign}\left(\mathrm{ReLU}(\lVert p_{i}-p_{j}^{\mathrm{goal}} \rVert-0.01)\right)+0.0001\lVert u_{j} \rVert^{2}\right).$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 3: [[优化目标/约束]]

$$\max_{\lambda\geq 0}\min_{\pi}\quad J_{\lambda}(\pi,\lambda):= J(\pi)+\lambda\sum_{k=0}^{\infty}\max\{h(x^{k}),0\},$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 4: [[优化目标/约束]]

$$=\max\left\{\max\{h(x^{k}),\max_{p\geq k+1}h(x^{p})\},\sum_{p\geq k+1}l(x^{p},\pi(x^{p}))-{\ {\Big{[}z^{k}-l(x^{k},\pi(x^{k}))\Big{]}}_{:= z^{k+1}}}\right\}$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 5: [[优化目标/约束]]

$$h_{c}(o_{i})=\max_{i}\min_{j\in\mathcal{N}_{i}^{o}}\lVert p_{i}-p_{j} \rVert-R^{\prime}+\nu\mathrm{sign}\left(\max_{i}\min_{j\in\mathcal{N}_{i}^{o}}\lVert p_{i}-p_{j} \rVert-R^{\prime}\right),$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 6: [[优化目标/约束]]

$$\quad\min_{\{\pi_{i}\}_{i=1}^{N}}\,\ {\max\big{\{}\max_{i}V_{i}^{h}(o_{i}^{\tau};\pi),V^{l}(x^{\tau};\pi)-z\big{\}}}_{:= V(x^{0},z;\pi)}\leq 0.$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 7: [[优化目标/约束]]

$$=\max\left\{\max\{h(x^{k}),\max_{p\geq k+1}h(x^{p})\},\sum_{p\geq k+1}l(x^{p},\pi(x^{p}))+l(x^{k},\pi(x^{k}))-z^{k}\right\}$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 8: [[优化目标/约束]]

$$=\max\left\{h(x^{k}),\max\left\{\max_{p\geq k+1}h(x^{p}),\sum_{p\geq k+1}l(x^{p},\pi(x^{p}))-z^{k+1}\right\}\right\}$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 9: [[优化目标/约束]]

$$\begin{split}z_{i}&=\min_{z^{\prime}}\quad z^{\prime}\\ &\mathrm{s.t.}\quad V_{i}^{h}(o_{i};\pi(\cdot,z^{\prime}))\leq 0,\quad i=1,\cdots,N.\end{split}$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 10: [[优化目标/约束]]

$$\frac{\partial}{\partial\lambda}J_{\lambda}({\pi},\lambda)=\sum_{k=0}^{\infty}\max\{h(x^{k}),0\}\geq 0$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。
## 关键图表

### Figure 1: Pipeline / core system figure: Def-MARL algorithm. Randomly sampled initial states and z 0 z 0 z^{0} ita

![Figure 1](https://arxiv.org/html/2504.15425v1/x1.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Pipeline / core system figure: Def-MARL algorithm. Randomly sampled initial states and z 0 z 0 z^{0} ita”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 2: Hardware Results on Corridor (N = 3 N 3 N=3 = 3). Left to right: key frame

![Figure 2](https://arxiv.org/html/2504.15425v1/x22.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Hardware Results on Corridor (N = 3 N 3 N=3 = 3). Left to right: key frame”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 3: Hardware Results on Inspect. The CF drone with the yellow / green sphere

![Figure 3](https://arxiv.org/html/2504.15425v1/x24.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Hardware Results on Inspect. The CF drone with the yellow / green sphere”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### 论文表格

#### Table 1: TABLE I: Policy Generalization. Testing Def-MARL on Target with more agents after training with N = 8 N 8 N=8

| # Agent | 32 | 128 | 512 |
| --- | --- | --- | --- |
| Safety rate | 99.8 ± 0.2 plus-or-minus 99.8 0.2 99.8\pm 0.2 99.8 ± 0.2 | 99.6 ± 0.4 plus-or-minus 99.6 0.4 99.6\pm 0.4 99.6 ± 0.4 | 99.5 ± 0.3 plus-or-minus 99.5 0.3 99.5\pm 0.3 99.5 ± 0.3 |
| Cost | - 0.387 ± 0.029 plus-or-minus 0.387 0.029 -0.387\pm 0.029 - 0.387 ± 0.029 | - 0.408 ± 0.015 plus-or-minus 0.408 0.015 -0.408\pm 0.015 - 0.408 ± 0.015 | - 0.410 ± 0.009 plus-or-minus 0.410 0.009 -0.410\pm 0.009 - 0.410 ± 0.009 |

**说明**: TABLE I: Policy Generalization. Testing Def-MARL on Target with more agents after training with N = 8 N 8 N=8 = 8 agents.

#### Table 2: TABLE II: Effect of z i z i z_{i} communication (Sec

| Environment | No communication (z ← z i) ← z z i (z $\leftarrow z_{i}) (←)$ | Communication (z = max i  z i z i z i z=\max_{i}z_{i} =) | | |
| --- | --- | --- | --- | --- |
| Safety rate | Cost | Safety rate | Cost | |
| Target | 97.9 ± 1.5 plus-or-minus 97.9 1.5 97.9\pm 1.5 97.9 ± 1.5 | 0.196 ± 0.108 plus-or-minus 0.196 0.108 0.196\pm 0.108 0.196 ± 0.108 | 96.9 ± 3.0 plus-or-minus 96.9 3.0 96.9\pm 3.0 96.9 ± 3.0 | 0.214 ± 0.141 plus-or-minus 0.214 0.141 0.214\pm 0.141 0.214 ± 0.141 |
| Spread | 99.0 ± 0.9 plus-or-minus 99.0 0.9 99.0\pm 0.9 99.0 ± 0.9 | 0.162 ± 0.144 plus-or-minus 0.162 0.144 0.162\pm 0.144 0.162 ± 0.144 | 98.6 ± 1.3 plus-or-minus 98.6 1.3 98.6\pm 1.3 98.6 ± 1.3 | 0.171 ± 0.128 plus-or-minus 0.171 0.128 0.171\pm 0.128 0.171 ± 0.128 |
| Formation | 98.3 ± 1.0 plus-or-minus 98.3 1.0 98.3\pm 1.0 98.3 ± 1.0 | 0.123 ± 0.940 plus-or-minus 0.123 0.940 0.123\pm 0.940 0.123 ± 0.940 | 98.3 ± 1.8 plus-or-minus 98.3 1.8 98.3\pm 1.8 98.3 ± 1.8 | 0.126 ± 0.100 plus-or-minus 0.126 0.100 0.126\pm 0.100 0.126 ± 0.100 |
| Line | 98.6 ± 0.5 plus-or-minus 98.6 0.5 98.6\pm 0.5 98.6 ± 0.5 | 0.117 ± 0.540 plus-or-minus 0.117 0.540 0.117\pm 0.540 0.117 ± 0.540 | 98.3 ± 0.5 plus-or-minus 98.3 0.5 98.3\pm 0.5 98.3 ± 0.5 | 0.121 ± 0.630 plus-or-minus 0.121 0.630 0.121\pm 0.630 0.121 ± 0.630 |
| Corridor | 97.9 ± 1.8 plus-or-minus 97.9 1.8 97.9\pm 1.8 97.9 ± 1.8 | 0.247 ± 0.390 plus-or-minus 0.247 0.390 0.247\pm 0.390 0.247 ± 0.390 | 98.6 ± 1.9 plus-or-minus 98.6 1.9 98.6\pm 1.9 98.6 ± 1.9 | 0.255 ± 0.470 plus-or-minus 0.255 0.470 0.255\pm 0.470 0.255 ± 0.470 |
| ConnectSpread | 97.9 ± 1.7 plus-or-minus 97.9 1.7 97.9\pm 1.7 97.9 ± 1.7 | 0.324 ± 0.187 plus-or-minus 0.324 0.187 0.324\pm 0.187 0.324 ± 0.187 | 99.0 ± 0.8 plus-or-minus 99.0 0.8 99.0\pm 0.8 99.0 ± 0.8 | 0.339 ± 0.201 plus-or-minus 0.339 0.201 0.339\pm 0.201 0.339 ± 0.201 |

**说明**: TABLE II: Effect of z i z i z_{i} communication (Section IV-C) in different environments.

#### Table 3: TABLE IV: Shared hyperparameters of Def-MARL, Penalty, and Lagr.

| Hyperparameter | Value | Hyperparameter | Value |
| --- | --- | --- | --- |
| policy GNN layers | 2 | RNN type | GRU |
| massage passing dimension | 32 | RNN data chunk length | 16 |
| GNN output dimension | 64 | RNN layers | 1 |
| number of attention heads | 3 | number of sampling environments | 128 |
| activation functions | ReLU | gradient clip norm | 2 |
| GNN head layers | (32, 32) | entropy coefficient | 0.01 |
| optimizer | Adam | GAE   \lambda | 0.95 |
| discount $\gamma$ | 0.99 | clip  \epsilon | 0.25 |
| policy learning rate | 3e-4 | PPO epoch | 1 |
| V l V l V^{l} learning rate | 1e-3 | batch size | 16384 |
| network initialization | Orthogonal | layer normalization | True |

**说明**: TABLE IV: Shared hyperparameters of Def-MARL, Penalty, and Lagr.

#### Table 4: TABLE V: Hyperparameters of Def-MARL.

| Hyperparameter | Value |
| --- | --- |
| V h V h V^{h} GNN layers | 2 for ConnectSpread, 1 for others |
| z z z encoding dimension | 8 |
| outer problem solver | Chandrupatla’s method [12 ] |

**说明**: TABLE V: Hyperparameters of Def-MARL.

#### Table 5: TABLE VI: Safety and Cost of Def-MARL policies trained with different z max z max z_{$\mathrm{max}$} s

| z max / z max, orig z max z max orig z_{ $\mathrm{max}}/z_{\mathrm{max,orig}} /,$ | Safety rate | Cost |
| --- | --- | --- |
| 0.25 0.25 0.25 0.25 | 93.8 ± 2.4 plus-or-minus 93.8 2.4 93.8\pm 2.4 93.8 ± 2.4 | 0.152 ± 0.100 plus-or-minus 0.152 0.100 0.152\pm 0.100 0.152 ± 0.100 |
| 0.5 0.5 0.5 0.5 | 98.0 ± 1.4 plus-or-minus 98.0 1.4 98.0\pm 1.4 98.0 ± 1.4 | 0.155 ± 0.104 plus-or-minus 0.155 0.104 0.155\pm 0.104 0.155 ± 0.104 |
| 1.0 1.0 1.0 1.0 | 99.0 ± 0.9 plus-or-minus 99.0 0.9 99.0\pm 0.9 99.0 ± 0.9 | 0.162 ± 0.144 plus-or-minus 0.162 0.144 0.162\pm 0.144 0.162 ± 0.144 |
| 1.5 1.5 1.5 1.5 | 99.0 ± 0.0 plus-or-minus 99.0 0.0 99.0\pm 0.0 99.0 ± 0.0 | 0.165 ± 0.100 plus-or-minus 0.165 0.100 0.165\pm 0.100 0.165 ± 0.100 |
| 2.0 2.0 2.0 2.0 | 99.0 ± 0.1 plus-or-minus 99.0 0.1 99.0\pm 0.1 99.0 ± 0.1 | 0.228 ± 0.109 plus-or-minus 0.228 0.109 0.228\pm 0.109 0.228 ± 0.109 |

**说明**: 该表格来自论文中的公式、奖励项或实验设置；为避免 PDF 抽取噪声，网页中仅保留可渲染的核心符号和数值。

#### Table 6: TABLE VII: Effect of z i z i z_{i} communication in l

| N N N | No communication (z ← z i) ← z z i (z $\leftarrow z_{i}) (←)$ | Communication (z = max i  z i z i z i z=\max_{i}z_{i} =) | | |
| --- | --- | --- | --- | --- |
| Safety rate | Cost | Safety rate | Cost | |
| 32 | 99.8 ± 0.2 plus-or-minus 99.8 0.2 99.8\pm 0.2 99.8 ± 0.2 | - 0.387 ± 0.029 plus-or-minus 0.387 0.029 -0.387\pm 0.029 - 0.387 ± 0.029 | 99.8 ± 0.2 plus-or-minus 99.8 0.2 99.8\pm 0.2 99.8 ± 0.2 | - 0.416 ± 0.052 plus-or-minus 0.416 0.052 -0.416\pm 0.052 - 0.416 ± 0.052 |
| 128 | 99.6 ± 0.4 plus-or-minus 99.6 0.4 99.6\pm 0.4 99.6 ± 0.4 | - 0.408 ± 0.015 plus-or-minus 0.408 0.015 -0.408\pm 0.015 - 0.408 ± 0.015 | 99.8 ± 0.3 plus-or-minus 99.8 0.3 99.8\pm 0.3 99.8 ± 0.3 | - 0.491 ± 0.090 plus-or-minus 0.491 0.090 -0.491\pm 0.090 - 0.491 ± 0.090 |
| 512 | 99.5 ± 0.3 plus-or-minus 99.5 0.3 99.5\pm 0.3 99.5 ± 0.3 | - 0.410 ± 0.009 plus-or-minus 0.410 0.009 -0.410\pm 0.009 - 0.410 ± 0.009 | 99.9 ± 0.1 plus-or-minus 99.9 0.1 99.9\pm 0.1 99.9 ± 0.1 | - 0.608 ± 0.210 plus-or-minus 0.608 0.210 -0.608\pm 0.210 - 0.608 ± 0.210 |

**说明**: TABLE VII: Effect of z i z i z_{i} communication in larger scale environments.
## 实验解读

- 评价重点:围绕 强化学习、safe-control、scalable-robot-learning、轨迹优化,优先看真实机器人结果、扰动恢复、接触稳定性、控制误差、成功率、实时性和 sim-to-real 差距。
- System0 视角:实验结论不只看最终成功率,还要看闭环过程中是否能处理状态估计误差、执行延迟、碰撞/接触突变和外部扰动。
- 与上层系统的关系:如果方法能在快闭环内稳定工作,它可以作为 System1 技能或 System2 规划的底层执行保障;如果只在离线或慢速设定下成立,则还不能直接作为身体级快闭环模块。
## 批判性思考

### 优点

- 论文把 强化学习、safe-control、scalable-robot-learning、轨迹优化 放到机器人身体级闭环里讨论,和只做高层语义推理的工作相比,更接近真实机器人部署时会遇到的状态、动力学和安全约束。
- 若论文同时给出公式、系统结构图、硬件实验或消融结果,这些证据可以帮助判断方法是否真的能支撑毫秒到百毫秒级反应。

### 局限

- 需要重点核查控制频率、推理延迟、传感延迟、失败恢复、硬件磨损、未见场景泛化和安全边界;这些因素决定它能否成为可靠的 System0 模块。
- 对学习型方法,还要区分训练环境中的鲁棒性和真实部署中的鲁棒性,尤其是接触丰富任务、动态运动任务和人形机器人任务。
## 论文定位

- 研究对象:Solving Multi-Agent Safe Optimal Control with Distributed Epigraph Form MARL。
- 关键词:强化学习、safe-control、scalable-robot-learning、轨迹优化。
- System0 关联:关注机器人身体级快反应链路中“状态/接触/局部感知 -> 约束/优化/策略 -> 动作”的短闭环,不把问题停留在语言规划或任务分解层。

## 速查卡片

> [!summary] MARL
> - **论文**: https://www.roboticsproceedings.org/rss21/p027.pdf
> - **arXiv**: http://arxiv.org/abs/2504.15425v1
> - **arXiv HTML**: https://arxiv.org/html/2504.15425v1
