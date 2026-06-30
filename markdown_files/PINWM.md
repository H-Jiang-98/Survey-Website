---
title: "PIN-WM: Learning Physics-INformed World Models for Non-Prehensile Manipulation"
method_name: "PIN-WM"
authors: ["Wenxuan Li"]
year: 2025
venue: "RSS"
tags: ["contact-reasoning", "robot-manipulation", "robust-control", "reinforcement-learning", "contact-rich-manipulation", "robot-generalization", "state-estimation"]
image_source: "online"
arxiv_html: "https://arxiv.org/html/2504.16693v2"
---
# PIN-WM
## 一句话总结

> PIN-WM: Learning Physics-INformed World Models for Non-Prehensile Manipulation 主要落在 [[接触推理]]、[[dynamics-modeling]]、[[non-prehensile-manipulation]]、[[policy-learning]] 这一类问题上;从 System0 视角看,它的价值在于把身体状态、接触变化、局部感知或动力学约束接到机器人低层闭环里,让动作生成更稳定、安全、实时。
## 核心贡献

1. 围绕 **PIN-WM: Learning Physics-INformed World Models for Non-Prehensile Manipulation** 建立了一个与 接触推理、dynamics-modeling、non-prehensile-manipulation、policy-learning、强化学习、robot-generalization 相关的机器人问题设定,重点不是高层语义规划,而是身体级闭环中的状态变化、接触变化或控制约束。
2. 方法层面的关键价值在于把感知、动力学、约束优化、策略学习或硬件反馈组织成可执行的低层动作生成机制,适合放进 System0 的快反应模块中理解。
3. 对系统集成而言,这类工作补上了 System1 技能选择与电机级控制之间的一层:它负责把上层目标变成动态可行、可恢复、可安全过滤的动作。
4. 若论文包含真实机器人、接触任务或仿真到真实迁移,核心应看它是否报告了闭环稳定性、扰动恢复、硬件延迟、失败模式和跨场景泛化。
## 问题背景

这篇论文可以放在 System0 的“具身快系统”背景下阅读:机器人需要在很短时间内根据本体状态、接触、力/触觉、局部视觉或动力学预测做出动作,而不是等待慢速规划器重新推理整个任务。接触推理、dynamics-modeling、non-prehensile-manipulation、policy-learning、强化学习、robot-generalization、机器人操作 相关问题的难点通常不在“知道要做什么”,而在动作执行过程中状态会快速偏离、接触会突然改变、约束会实时收紧,低层系统必须及时修正。

### System0 关联

精读时优先看三件事:第一,论文使用哪些快速反馈量,例如关节状态、接触状态、力觉、触觉、深度或局部视觉;第二,反馈如何进入 MPC、WBC、策略网络、安全过滤器或恢复控制器;第三,输出是否直接影响机器人下一步动作,而不是只服务离线分析或高层任务规划。
## 方法详解

### System0 视角解析

从 System0 角度看,这篇论文的方法部分应当围绕 接触推理、dynamics-modeling、non-prehensile-manipulation、policy-learning、强化学习、robot-generalization、机器人操作、鲁棒控制 来读。核心问题是:系统如何从快速变化的机器人状态中构造控制输入,如何处理接触、碰撞、约束、模型误差或执行延迟,以及如何把这些信息变成下一时刻的动作。

如果方法里有优化问题,重点看目标函数、约束和求解频率;如果方法里有学习策略,重点看观测空间、动作空间、训练扰动和部署时的闭环频率;如果方法里有硬件或传感器,重点看传感链路是否真的进入控制回路。这样读可以把论文和 System2 语言规划、System1 技能编排区分开来。

### 关键公式与机制

#### 公式 1: [[优化目标/约束]]

$$I_{t+1}=\mathcal{I}(g(\mathbf{x}_{t},\mathbf{a}_{t},\boldsymbol{\theta}),\boldsymbol{\alpha}),$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 2: [[优化目标/约束]]

$$\hat{I}(x,y)=\sum_{i=1}\boldsymbol{\alpha}^{c}_{i}\boldsymbol{\alpha}^{o}_{i}\mathcal{G}_{i}(\mathbf{u}(x,y))\prod_{j=1}^{i-1}\left(1-\boldsymbol{\alpha}^{o}_{j}\mathcal{G}_{j}(\mathbf{u}(x,y))\right),$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 3: [[优化目标/约束]]

$$\min_{\boldsymbol{\theta}}\quad\mathcal{L}_{r}(\boldsymbol{\theta})=\sum_{i=1}^{n}\lVert \mathcal{I}(g(\mathbf{x}_{t+i-1},\mathbf{a}_{t+i-1},\boldsymbol{\theta}),\boldsymbol{\alpha}^{*})-I^{d}_{t+i} \rVert_{2}^{2}.$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 4: [[动力学或策略机制]]

$$\boldsymbol{\theta}^{\mathbf{M}}\boldsymbol{\xi}_{t+1}=\boldsymbol{\theta}^{\mathbf{M}}\boldsymbol{\xi}_{t}+\mathbf{f}^{\text{g}}\cdot H+\mathbf{J}_{e}\boldsymbol{\lambda}_{e}+\mathbf{J}_{c}\boldsymbol{\lambda}_{c}+\mathbf{J}_{f}\boldsymbol{\lambda}_{f},$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 5: [[动力学或策略机制]]

$$\mathbf{H}_{0}=\begin{bmatrix}s_{u}\mathbf{t}_{u}&s_{v}\mathbf{t}_{v}&\mathbf{0}&\mathbf{p}_{k}\\ 0&0&0&1\end{bmatrix}=\begin{bmatrix}\mathbf{RS}&\mathbf{p}_{k}\\ \mathbf{0}&1\end{bmatrix}.$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 6: [[动力学或策略机制]]

$$\mathbf{J}_{f}\boldsymbol{\xi}_{t+1}+\mathbf{E}\boldsymbol{\gamma}\geq 0,\quad\boldsymbol{\theta}^{\mu}\boldsymbol{\lambda}_{c}\geq\mathbf{E}^{\top}\boldsymbol{\lambda}_{f},\quad\text{(Friction Constraints)}$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 7: [[优化目标/约束]]

$$\cdot(\frac{\partial\mathcal{I}}{\partial g}\frac{\partial g}{\partial\boldsymbol{\theta}}+\frac{\partial\mathcal{I}}{\partial g}\frac{\partial g}{\partial\mathbf{x}_{t+i-1}}\frac{\partial\mathbf{x}_{t+i-1}}{\partial\boldsymbol{\theta}}).$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 8: [[优化目标/约束]]

$$\frac{d\mathcal{L}_{r}(\boldsymbol{\theta})}{d\boldsymbol{\theta}}=\sum_{i=1}^{n}(\mathcal{I}$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 9: [[动力学或策略机制]]

$$\mathbf{J}_{c}\boldsymbol{\xi}_{t+1}\geq-\boldsymbol{\theta}^{\mathbf{k}}\mathbf{J}_{c}\boldsymbol{\xi}_{t}\geq-\mathbf{c},\quad\quad\quad\quad\quad\,\,\,\text{(Contact Constraints)}$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 10: [[动力学或策略机制]]

$$\boldsymbol{\theta}=\{\boldsymbol{\theta}^{\mathbf{M}},\boldsymbol{\theta}^{\mathbf{k}},\boldsymbol{\theta}^{\mu}\}$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。
## 关键图表

### Figure 1: Pipeline / core system figure: PIN-WM is learned from few-shot and task-agnostic physical interaction trajectories (r

![Figure 1](https://arxiv.org/html/2504.16693v2/x1.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Pipeline / core system figure: PIN-WM is learned from few-shot and task-agnostic physical interaction trajectories (r”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 2: Our Real2Sim2Real framework for learning non-prehensile manipulation policies. (a) The

![Figure 2](https://arxiv.org/html/2504.16693v2/x2.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Our Real2Sim2Real framework for learning non-prehensile manipulation policies. (a) The”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 3: Our real-world experiment setup

![Figure 3](https://arxiv.org/html/2504.16693v2/x5.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Our real-world experiment setup”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### 论文表格

#### Table 1: TABLE I: Comparisons on policy performance in the target domain.

| Methods | Tasks | | | |
| --- | --- | --- | --- | --- |
| Push | Flip | | | |
| S u c c % S u c percent c Succ\,\% % | # S t e p s # S t e p s \#Steps # | S u c c % S u c percent c Succ\,\% % | # S t e p s # S t e p s \#Steps # | |
| Random | 0% | 100.0 | 0% | 25.0 |
| Dreamer V2 [27 ] | 1% | 99.9 | 0% | 25.0 |
| Diffusion Policy [11 ] | 13% | 91.1 | 10% | 23.3 |
| RoboGSim [45 ] | 19% | 82.6 | 21% | 20.5 |
| Domain Rand [60 ] + I I  $\mathcal{I}$ | 33% | 73.6 | 32% | 20.0 |
| 2D Physics [67 ] + I I  $\mathcal{I}$ | 55% | 60.6 | 8% | 22.6 |
| ASID [53 ] + I I  $\mathcal{I}$ | 58% | 57.6 | 11% | 22.2 |
| PIN-WM w/o PADC | 92% | 32.1 | 70% | 12.0 |
| PIN-WM w/ PADC | 97 % percent 97  $\mathbf{97}\$ | 30.1 30.1  $\mathbf{30.1}.1$ | 83 % percent 83  $\mathbf{83}\$ | 11.4 11.4  $\mathbf{11.4}.4$ |

**说明**: TABLE I: Comparisons on policy performance in the target domain.

#### Table 2: TABLE II: Comparisons on system identification accuracy across different methods, using one-step error of the predicted

| Methods | Tasks | | | |
| --- | --- | --- | --- | --- |
| Push | Flip | | | |
| Trans. | Ori. | Trans. | Ori. | |
| Dreamer V2 [27 ] | 7.6  10 - 2 absent 10 2 \times 10^{-2}  10 - 2 | 5.6  10 - 1 absent 10 1 \times 10^{-1}  10 - 1 | 2.3  10 - 1 absent 10 1 \times 10^{-1}  10 - 1 | 2.0 |
| 2D Physics [67 ] | 4.2  10 - 2 absent 10 2 \times 10^{-2}  10 - 2 | 5.4  10 - 1 absent 10 1 \times 10^{-1}  10 - 1 | 2.1  10 - 1 absent 10 1 \times 10^{-1}  10 - 1 | 1.6 |
| ASID [53 ] | 3.0  10 - 2 absent 10 2 \times 10^{-2}  10 - 2 | 4.0  10 - 1 absent 10 1 \times 10^{-1}  10 - 1 | 1.4  10 - 1 absent 10 1 \times 10^{-1}  10 - 1 | 1.6 |
| PIN-WM | $\mathbf{1.7\times 10^{-2}}.7 × -$ | $\mathbf{1.3\times 10^{-1}}.3 × -$ | $\mathbf{1.4\times 10^{-2}}.4 × -$ | 0.3 0.3  $\mathbf{0.3}.3$ |

**说明**: TABLE II: Comparisons on system identification accuracy across different methods, using one-step error of the predicted trajectory. “Trans.” and “Rot.” are translation and rotation errors, respectively.

#### Table 3: TABLE III: Real-world deployment performance.

| Methods | Tasks | | | |
| --- | --- | --- | --- | --- |
| Push | Flip | | | |
| S u c c % S u c percent c Succ\% % | # S t e p s # S t e p s \#Steps # | S u c c % S u c percent c Succ\% % | # S t e p s # S t e p s \#Steps # | |
| Random | 0% | 100.0 | 0% | 25.0 |
| Domain Rand [60 ] + I I  $\mathcal{I}$ | 10% | 87.5 | 25% | 18.5 |
| RoboGSim [45 ] | 30% | 72.7 | 15% | 21.2 |
| 2D Physics [67 ] + I I  $\mathcal{I}$ | 35% | 70.7 | 5% | 24.1 |
| ASID [53 ] + I I  $\mathcal{I}$ | 40% | 64.6 | 10% | 22.4 |
| PIN-WM w/o PADC | 65% | 45.2 | 60% | 13.2 |
| PIN-WM w/ PADC | 75 % | 37.5 | 65 % | 11.3 |

**说明**: TABLE III: Real-world deployment performance.

#### Table 4: TABLE IV: Success rates across different ranges of randomization.

| Tasks | GT | PIN-WM w/ PADC | DR (R R  $\mathbf{R} /4)$ | DR (R R  $\mathbf{R} /2)$ | DR (R R  $\mathbf{R})$ | |
| --- | --- | --- | --- | --- | --- | --- |
| Push | 98 % percent 98  $\mathbf{98\}$ | 97% | 78% | 56% | 33% | |
| Flip | 89 % percent 89  $\mathbf{89\}$ | 83% | 61% | 43% | 32% | |

**说明**: TABLE IV: Success rates across different ranges of randomization.

#### Table 5: TABLE V: Identified physical parameters.

| Methods | Push | Flip | | | | |
| --- | --- | --- | --- | --- | --- | --- |
| Friction | Mass (kg) | Restitution | Friction | Mass (kg) | Restitution | |
| GT | 3.00  10 - 2 3.00 10 2 3.00\times 10^{-2} 3.00  10 - 2 | 1.00 | 0.00 | 2.00  10 - 1 2.00 10 1 2.00\times 10^{-1} 2.00  10 - 1 | 1.00 | 0.00 |
| ASID | 4.45  10 - 2 4.45 10 2 4.45\times 10^{-2} 4.45  10 - 2 | 0.75 | 1.12  10 - 2 1.12 10 2 1.12\times 10^{-2} 1.12  10 - 2 | 2.59  10 - 1 2.59 10 1 2.59\times 10^{-1} 2.59  10 - 1 | 1.36 | 1.12  10 - 2 1.12 10 2 1.12\times 10^{-2} 1.12  10 - 2 |
| 2D Physics | – | 19.25 | – | – | 29.60 | – |
| PIN-WM | $\mathbf{3.05\times 10^{-2}}.05 × -$ | 0.76 0.76  $\mathbf{0.76}.76$ | $\mathbf{4.18\times 10^{-4}}.18 × -$ | $\mathbf{2.11\times 10^{-1}}.11 × -$ | 1.19 1.19  $\mathbf{1.19}.19$ | $\mathbf{1.69\times 10^{-5}}.69 × -$ |

**说明**: TABLE V: Identified physical parameters.

#### Table 6: TABLE VI: One-step errors across different noise levels.

| Methods | Push | Flip | | |
| --- | --- | --- | --- | --- |
| Trans. | Ori. | Trans. | Ori. | |
| PIN-WM ( = 0.0  0.0 \sigma=0.0 = 0.0) | $\mathbf{1.7\times 10^{-2}}.7 × -$ | $\mathbf{1.3\times 10^{-1}}.3 × -$ | $\mathbf{1.4\times 10^{-2}}.4 × -$ | $\mathbf{2.7\times 10^{-1}}.7 × -$ |
| PIN-WM ( = 0.5  0.5 \sigma=0.5 = 0.5) | 2.1  10 - 2 2.1 10 2 2.1\times 10^{-2} 2.1  10 - 2 | 2.1  10 - 1 2.1 10 1 2.1\times 10^{-1} 2.1  10 - 1 | 4.2  10 - 2 4.2 10 2 4.2\times 10^{-2} 4.2  10 - 2 | 9.6  10 - 1 9.6 10 1 9.6\times 10^{-1} 9.6  10 - 1 |
| PIN-WM ( = 1.0  1.0 \sigma=1.0 = 1.0) | 2.3  10 - 2 2.3 10 2 2.3\times 10^{-2} 2.3  10 - 2 | 1.6  10 - 1 1.6 10 1 1.6\times 10^{-1} 1.6  10 - 1 | 4.2  10 - 2 4.2 10 2 4.2\times 10^{-2} 4.2  10 - 2 | 9.5  10 - 1 9.5 10 1 9.5\times 10^{-1} 9.5  10 - 1 |
| PIN-WM ( = 3.0  3.0 \sigma=3.0 = 3.0) | 2.9  10 - 2 2.9 10 2 2.9\times 10^{-2} 2.9  10 - 2 | 3.0  10 - 1 3.0 10 1 3.0\times 10^{-1} 3.0  10 - 1 | 7.7  10 - 2 7.7 10 2 7.7\times 10^{-2} 7.7  10 - 2 | 1.5 1.5 1.5 1.5 |
| ASID | 3.0  10 - 2 3.0 10 2 3.0\times 10^{-2} 3.0  10 - 2 | 4.0  10 - 1 4.0 10 1 4.0\times 10^{-1} 4.0  10 - 1 | 1.4  10 - 1 1.4 10 1 1.4\times 10^{-1} 1.4  10 - 1 | 1.6 |

**说明**: TABLE VI: One-step errors across different noise levels.

#### Table 7: TABLE VII: Success rate comparisons on different real-world tasks.

| Methods | Push T (Slippery) | Push Cube (Slippery) | Flip Cube |
| --- | --- | --- | --- |
| ASID | 5 % percent 5 5\% 5 % | 0 % percent 0 0\% 0 % | 5 % percent 5 5\% 5 % |
| PIN-WM | 45 % percent 45  $\mathbf{45\}$ | 40 % percent 40  $\mathbf{40\}$ | 60 % percent 60  $\mathbf{60\}$ |

**说明**: TABLE VII: Success rate comparisons on different real-world tasks.
## 实验解读

- 评价重点:围绕 接触推理、dynamics-modeling、non-prehensile-manipulation、policy-learning、强化学习,优先看真实机器人结果、扰动恢复、接触稳定性、控制误差、成功率、实时性和 sim-to-real 差距。
- System0 视角:实验结论不只看最终成功率,还要看闭环过程中是否能处理状态估计误差、执行延迟、碰撞/接触突变和外部扰动。
- 与上层系统的关系:如果方法能在快闭环内稳定工作,它可以作为 System1 技能或 System2 规划的底层执行保障;如果只在离线或慢速设定下成立,则还不能直接作为身体级快闭环模块。
## 批判性思考

### 优点

- 论文把 接触推理、dynamics-modeling、non-prehensile-manipulation、policy-learning、强化学习 放到机器人身体级闭环里讨论,和只做高层语义推理的工作相比,更接近真实机器人部署时会遇到的状态、动力学和安全约束。
- 若论文同时给出公式、系统结构图、硬件实验或消融结果,这些证据可以帮助判断方法是否真的能支撑毫秒到百毫秒级反应。

### 局限

- 需要重点核查控制频率、推理延迟、传感延迟、失败恢复、硬件磨损、未见场景泛化和安全边界;这些因素决定它能否成为可靠的 System0 模块。
- 对学习型方法,还要区分训练环境中的鲁棒性和真实部署中的鲁棒性,尤其是接触丰富任务、动态运动任务和人形机器人任务。
## 论文定位

- 研究对象:PIN-WM: Learning Physics-INformed World Models for Non-Prehensile Manipulation。
- 关键词:接触推理、dynamics-modeling、non-prehensile-manipulation、policy-learning、强化学习、robot-generalization、机器人操作、鲁棒控制、状态估计、world-model。
- System0 关联:关注机器人身体级快反应链路中“状态/接触/局部感知 -> 约束/优化/策略 -> 动作”的短闭环,不把问题停留在语言规划或任务分解层。

## 速查卡片

> [!summary] PIN-WM
> - **论文**: https://www.roboticsproceedings.org/rss21/p153.pdf
> - **arXiv**: http://arxiv.org/abs/2504.16693v2
> - **arXiv HTML**: https://arxiv.org/html/2504.16693v2
