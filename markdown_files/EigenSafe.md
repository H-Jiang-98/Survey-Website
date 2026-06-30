---
title: "EigenSafe: A Spectral Framework for Learning-Based Probabilistic Safety Assessment"
method_name: "EigenSafe"
authors: ["Inkyu Jang"]
year: 2026
venue: "RSS"
tags: ["contact-reasoning", "robot-manipulation", "robust-control", "reinforcement-learning", "safe-control", "imitation-learning", "closed-loop-control"]
image_source: "online"
arxiv_html: "https://arxiv.org/html/2509.17750"
---
# EigenSafe
## 一句话总结

> EigenSafe: A Spectral Framework for Learning-Based Probabilistic Safety Assessment 主要落在 [[closed-loop-control]]、[[接触推理]]、[[模仿学习]]、[[reachability]] 这一类问题上;从 System0 视角看,它的价值在于把身体状态、接触变化、局部感知或动力学约束接到机器人低层闭环里,让动作生成更稳定、安全、实时。
## 核心贡献

1. 围绕 **EigenSafe: A Spectral Framework for Learning-Based Probabilistic Safety Assessment** 建立了一个与 closed-loop-control、接触推理、模仿学习、reachability、强化学习、机器人操作 相关的机器人问题设定,重点不是高层语义规划,而是身体级闭环中的状态变化、接触变化或控制约束。
2. 方法层面的关键价值在于把感知、动力学、约束优化、策略学习或硬件反馈组织成可执行的低层动作生成机制,适合放进 System0 的快反应模块中理解。
3. 对系统集成而言,这类工作补上了 System1 技能选择与电机级控制之间的一层:它负责把上层目标变成动态可行、可恢复、可安全过滤的动作。
4. 若论文包含真实机器人、接触任务或仿真到真实迁移,核心应看它是否报告了闭环稳定性、扰动恢复、硬件延迟、失败模式和跨场景泛化。
## 问题背景

这篇论文可以放在 System0 的“具身快系统”背景下阅读:机器人需要在很短时间内根据本体状态、接触、力/触觉、局部视觉或动力学预测做出动作,而不是等待慢速规划器重新推理整个任务。closed-loop-control、接触推理、模仿学习、reachability、强化学习、机器人操作、鲁棒控制 相关问题的难点通常不在“知道要做什么”,而在动作执行过程中状态会快速偏离、接触会突然改变、约束会实时收紧,低层系统必须及时修正。

### System0 关联

精读时优先看三件事:第一,论文使用哪些快速反馈量,例如关节状态、接触状态、力觉、触觉、深度或局部视觉;第二,反馈如何进入 MPC、WBC、策略网络、安全过滤器或恢复控制器;第三,输出是否直接影响机器人下一步动作,而不是只服务离线分析或高层任务规划。
## 方法详解

### System0 视角解析

从 System0 角度看,这篇论文的方法部分应当围绕 closed-loop-control、接触推理、模仿学习、reachability、强化学习、机器人操作、鲁棒控制、safe-control 来读。核心问题是:系统如何从快速变化的机器人状态中构造控制输入,如何处理接触、碰撞、约束、模型误差或执行延迟,以及如何把这些信息变成下一时刻的动作。

如果方法里有优化问题,重点看目标函数、约束和求解频率;如果方法里有学习策略,重点看观测空间、动作空间、训练扰动和部署时的闭环频率;如果方法里有硬件或传感器,重点看传感链路是否真的进入控制回路。这样读可以把论文和 System2 语言规划、System1 技能编排区分开来。

### 关键公式与机制

#### 公式 1: [[动力学或策略机制]]

$$\begin{aligned} | T_ $\pi \beta(x)$ | &=  $\left$ | \int_C \beta(y) p_ $\pi (y$ | x) dy $\right$ | \leq \left\lVert \beta \right \rVert_\infty. \end{aligned}$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 2: [[优化目标/约束]]

$$\begin{multlined} \mathcal{J}_\text{eig}[\gamma, \psi]:= \\ \frac{W_\text{eig}}{| $\mathcal{D}$ |}\sum_{(x,u,x') \in \mathcal{D}} \left(\begin{aligned} 1[x \in C \wedge x' \in C]\cdot \psi(x', u') \quad \quad \\ {} - \gamma \cdot 1[x \in C] \cdot \psi(x,u) \end{aligned}\right)^2 \\ {} + W_n \cdot \left(\max_{(x,u,\cdot) \in \mathcal{D}} \psi(x,u) - 1 \right)^2, \end{multlined}$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 3: [[优化目标/约束]]

$$\mathcal{J}_+[\psi] = \frac{W_+}{| $\mathcal{D}$ |}\sum_{(x,u,\cdot) \in \mathcal{D}} \operatorname{ReLU}\left(-\psi(x,u)\right)$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 4: [[优化目标/约束]]

$$\begin{aligned} \max_{\pi} \quad & \mathcal{J}_\text{RL} [\pi]:= \mathbb{E}_{(x,\cdot, \cdot) \sim \mathcal{D}, u \sim \pi(\cdot| x)}  $\left[Q_\pi (x, u)\right] \\ \operatorname{s.t.}\quad & \mathbb{E}_{x' \sim P(\cdot$ | x,u), u' \sim  $\pi(\cdot$ |x')} \psi_\pi (x',u') \geq \gamma_0 \psi_\pi(x,u), \end{aligned}$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 5: [[动力学或策略机制]]

$$\begin{aligned} | T_ $\pi \beta(x_1) - T_\pi \beta(x_2)$ | &=  $\left$ | \int_C \beta(y)  $\left(p_\pi(y$ | x_1) - p_ $\pi(y$ | x_2)  $\right) dy\right$ | \\ &\leq \int_C |\beta(y)| \cdot  $\left$ | p_ $\pi(y$ | x_1) - p_ $\pi(y$ | x_2) $\right$ |dy \\ &\leq \left\lVert \beta \right \rVert_\infty \cdot \frac{\epsilon}{\operatorname{Vol}(C)} \cdot \operatorname{Vol}(C) \leq \epsilon, \end{aligned}$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 6: [[动力学或策略机制]]

$$\begin{aligned} \lambda^n \phi(x) &= T_\pi^n \phi(x) \\ & = \mathbb{E}_\pi [\phi(s_n) | s_0 = x] \\ &= \int_C \phi (y) p^n_ $\pi(y$ | x) dy \\ &= \ {\int_U \phi (y) p^n_ $\pi(y$ | x) dy}_{ $\text{(A)}} + \ {\int_{C \setminus U}\phi (y) p^n_\pi(y$ |x) dy}_{\text{(B)}}. \end{aligned}$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 7: [[动力学或策略机制]]

$$\begin{aligned} \gamma_\pi |\phi | (x) &= |\lambda||\phi(x)| = | T_ $\pi\phi(x)$ | =  $\left$ | \int_C p_ $\pi (y$ | x) \phi(y) dy  $\right$ | \\ &\leq \int_C p_ $\pi(y$ |x) |\phi(y)| dy = T_ $\pi$ |\phi|(x), \end{aligned}$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 8: [[动力学或策略机制]]

$$\begin{multlined} \left| \int_C p_ $\pi^n (y$ | x) \phi(y) dy  $\right$ | = | T_ $\pi^n \phi (x)$ | = |\lambda^n \phi (x)| =  $\gamma_\pi^n \phi_\pi(x) \\ = T_\pi^n \phi_\pi (x) = T_\pi^n$ |\phi| (x) = \int_C p_ $\pi^n (y$ |x) |\phi (y)| dy. \end{multlined}$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 9: [[优化目标/约束]]

$$\mathcal{J}_+[\psi] = \frac{W_+}{| $\mathcal{D}$ |}\sum_{(x,u,\cdot) \in \mathcal{D}} \operatorname{ReLU}\left(-\psi(x,u)\right)$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 10: [[动力学或策略机制]]

$$\lambda^n \phi(x) = T_\pi^n \phi(x) \geq T_\pi^n \phi_\pi(x) = \gamma_\pi^n \phi_\pi(x), \; \forall n \in \mathbb{N}, x \in C.$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。
## 关键图表

### Figure 1: Pipeline / core system figure: EigenSafe project figure: reinforcement-learning comparison and safety assessment visualization

![Figure 1](https://eigen-safe.github.io/files/rl_comparison_web.svg)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Pipeline / core system figure: EigenSafe project figure: reinforcement-learning comparison and safety assessment visualization”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 2: EigenSafe project figure: imitation-learning success-rate visualization

![Figure 2](https://eigen-safe.github.io/files/il_success_rate.svg)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“EigenSafe project figure: imitation-learning success-rate visualization”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### 论文表格

#### Table 1: Practical Implications of the Theorems for EigenSafe

| tabular{m{3.5cm} m{4.5cm}} 1{c}{Mathematical Guarantee} | 1{c}{Practical Implication} |
| --- | --- |
| thm: compactness of T: The operator $T_{}$ is compact. | Enables the subsequent theoretical analyses by guaranteeing that $T_$ has similar properties to finite matrices. |
| thm: dominant eigenfunction positivity: The dominant eigenfunction $_{}$ is strictly positive everywhere in the safe set. | Supports that the dominant eigenfunction plays a role of a calibrated safety critic, (the safety probability is always positive). |
| thm: uniqueness: The dominant eigenpair $(_{}, _{})$ is unique up to scaling. | Supports that the learning algorithm has a stable, unique target to converge to, avoiding mode collapse. |
| thm: spectral gap: A strictly positive gap exists between $_{}$ and the rest of the spectrum $\lVert \rVert < _{}$. | Ensures that the non-dominant eigenmodes of $T_$ decay faster than the dominant mode, explains the convergence of power iteration. |
| thm: probability approximation: The true safety probability $Z_{}(t,x)$ decays asymptotically as $c _{}(x) _{}^t$. | Justifies the use of the dominant eigenvalue $_{}$ and the dominant eigenfunction $_$ as global and local safety scores, respectively. |
| thm: non-dominant: All real, non-dominant eigenfunctions must take negative values. | Justifies the auxiliary positivity loss $J_{+}$ introduced in eq: positivity loss. This loss penalizes converging to any non-dominant mode. |
| thm: global-local equivalence: The global spectral constraint $_{} _0$ is equivalent to a local condition on $T_{}$. | Enables the implementation of the global safety constraint using only local transitions sampled from the replay buffer. |
| tabular | |

**说明**: Practical Implications of the Theorems for EigenSafe

#### Table 2: Hyperparameters for Safe RL with EigenSafe.

| tabular{cc} Parameter | Value |
| --- | --- |
| Learning rate | \(5 10^{-4}\) |
| Discount factor for SAC Q function $_RL$ | \(0.99 \) |
| Optimizer | Adam kingma2014adam |
| \# of episodes per epoch | 10 |
| \# of gradient steps per epoch | 64 |
| Replay buffer size | \(5 10^4\) |
| Minibatch size | 512 |
| Target smoothing coefficient | \(0.995 \) |
| Entropy coefficient | Auto-tuned haarnoja2018soft |
| Activation function | ReLU |
| \# of hidden layers | 2 |
| \# of hidden units per layer | 512 |
| Eigenfunction loss weight $W_eig$ | 1.0 |
| Normalization loss weight $W_{n}$ | 1.0 |
| Positivity loss weight $W_+$ | 1.0 |
| Target eigenvalue $_0$ | 1.0 |
| Lagrange multiplier relaxation constant $$ | \(1 10^{-3}\) |
| tabular | |

**说明**: Hyperparameters for Safe RL with EigenSafe.

#### Table 3: Hyperparameters for Safe IL with EigenSafe.

| tabular{cc} Parameter | Value |
| --- | --- |
| Learning rate | \(1 10^{-4}\) |
| Optimizer | Adam kingma2014adam |
| Minibatch size | 32 |
| Activation function | ELU |
| \# of hidden layers | 2 |
| \# of hidden units per layer | 512 |
| \# of LayerNorm layers | 2 |
| Eigenfunction loss weight $W_eig$ | 1.0 |
| Normalization loss weight $W_{n}$ | 1.0 |
| Positivity loss weight $W_+$ | 1.0 |
| tabular | |

**说明**: Hyperparameters for Safe IL with EigenSafe.
## 实验解读

- 评价重点:围绕 closed-loop-control、接触推理、模仿学习、reachability、强化学习,优先看真实机器人结果、扰动恢复、接触稳定性、控制误差、成功率、实时性和 sim-to-real 差距。
- System0 视角:实验结论不只看最终成功率,还要看闭环过程中是否能处理状态估计误差、执行延迟、碰撞/接触突变和外部扰动。
- 与上层系统的关系:如果方法能在快闭环内稳定工作,它可以作为 System1 技能或 System2 规划的底层执行保障;如果只在离线或慢速设定下成立,则还不能直接作为身体级快闭环模块。
## 批判性思考

### 优点

- 论文把 closed-loop-control、接触推理、模仿学习、reachability、强化学习 放到机器人身体级闭环里讨论,和只做高层语义推理的工作相比,更接近真实机器人部署时会遇到的状态、动力学和安全约束。
- 若论文同时给出公式、系统结构图、硬件实验或消融结果,这些证据可以帮助判断方法是否真的能支撑毫秒到百毫秒级反应。

### 局限

- 需要重点核查控制频率、推理延迟、传感延迟、失败恢复、硬件磨损、未见场景泛化和安全边界;这些因素决定它能否成为可靠的 System0 模块。
- 对学习型方法,还要区分训练环境中的鲁棒性和真实部署中的鲁棒性,尤其是接触丰富任务、动态运动任务和人形机器人任务。
## 论文定位

- 研究对象:EigenSafe: A Spectral Framework for Learning-Based Probabilistic Safety Assessment。
- 关键词:closed-loop-control、接触推理、模仿学习、reachability、强化学习、机器人操作、鲁棒控制、safe-control、安全过滤。
- System0 关联:关注机器人身体级快反应链路中“状态/接触/局部感知 -> 约束/优化/策略 -> 动作”的短闭环,不把问题停留在语言规划或任务分解层。

## 速查卡片

> [!summary] EigenSafe
> - **论文**: https://www.roboticsproceedings.org/rss22/p146.pdf
> - **arXiv**: http://arxiv.org/abs/2509.17750v2
> - **arXiv HTML**: https://arxiv.org/html/2509.17750
> - **项目页**: https://eigen-safe.github.io/
