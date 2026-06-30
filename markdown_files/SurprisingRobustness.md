---
title: "On the Surprising Robustness of Sequential Convex Optimization for Contact-Implicit Motion Planning"
method_name: "Surprising Robustness Sequential"
authors: ["Yulin Li"]
year: 2025
venue: "RSS"
tags: ["contact-reasoning", "robust-control", "real-time-control", "adaptive-control"]
image_source: "online"
arxiv_html: "https://arxiv.org/html/2502.01055v3"
---
# Surprising Robustness Sequential
## 一句话总结

> On the Surprising Robustness of Sequential Convex Optimization for Contact-Implicit Motion Planning 主要落在 [[adaptive-control]]、[[contact-implicit-optimization]]、[[接触推理]]、[[实时控制]] 这一类问题上;从 System0 视角看,它的价值在于把身体状态、接触变化、局部感知或动力学约束接到机器人低层闭环里,让动作生成更稳定、安全、实时。
## 核心贡献

1. 围绕 **On the Surprising Robustness of Sequential Convex Optimization for Contact-Implicit Motion Planning** 建立了一个与 adaptive-control、contact-implicit-optimization、接触推理、实时控制、鲁棒控制 相关的机器人问题设定,重点不是高层语义规划,而是身体级闭环中的状态变化、接触变化或控制约束。
2. 方法层面的关键价值在于把感知、动力学、约束优化、策略学习或硬件反馈组织成可执行的低层动作生成机制,适合放进 System0 的快反应模块中理解。
3. 对系统集成而言,这类工作补上了 System1 技能选择与电机级控制之间的一层:它负责把上层目标变成动态可行、可恢复、可安全过滤的动作。
4. 若论文包含真实机器人、接触任务或仿真到真实迁移,核心应看它是否报告了闭环稳定性、扰动恢复、硬件延迟、失败模式和跨场景泛化。
## 问题背景

这篇论文可以放在 System0 的“具身快系统”背景下阅读:机器人需要在很短时间内根据本体状态、接触、力/触觉、局部视觉或动力学预测做出动作,而不是等待慢速规划器重新推理整个任务。adaptive-control、contact-implicit-optimization、接触推理、实时控制、鲁棒控制 相关问题的难点通常不在“知道要做什么”,而在动作执行过程中状态会快速偏离、接触会突然改变、约束会实时收紧,低层系统必须及时修正。

### System0 关联

精读时优先看三件事:第一,论文使用哪些快速反馈量,例如关节状态、接触状态、力觉、触觉、深度或局部视觉;第二,反馈如何进入 MPC、WBC、策略网络、安全过滤器或恢复控制器;第三,输出是否直接影响机器人下一步动作,而不是只服务离线分析或高层任务规划。
## 方法详解

### System0 视角解析

从 System0 角度看,这篇论文的方法部分应当围绕 adaptive-control、contact-implicit-optimization、接触推理、实时控制、鲁棒控制 来读。核心问题是:系统如何从快速变化的机器人状态中构造控制输入,如何处理接触、碰撞、约束、模型误差或执行延迟,以及如何把这些信息变成下一时刻的动作。

如果方法里有优化问题,重点看目标函数、约束和求解频率;如果方法里有学习策略,重点看观测空间、动作空间、训练扰动和部署时的闭环频率;如果方法里有硬件或传感器,重点看传感链路是否真的进入控制回路。这样读可以把论文和 System2 语言规划、System1 技能编排区分开来。

### 关键公式与机制

#### 公式 1: [[动力学或策略机制]]

$$LFC=\left\{d\in\mathbb{R}^{2}\;\bigg{|}\;\begin{array}{l}\nabla g_{i}^{\top}d\geq 0,i=1,2\\ \nabla g_{j}^{\top}d=0,j=3\end{array}\right\}.$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 2: [[动力学或策略机制]]

$$D(q_{\mu,n_{k}}(x_{n_{k}+1});p)\xrightarrow[k\to\infty]{}\nabla J(x^{\star})^{\top}p+\sum_{i=1}^{m_{1}}\mu_{i}\nabla c_{i}(x^{\star})^{\top}p-\sum_{i=m_{1}+1}^{m_{1}+m_{2}}\mu_{i}\nabla c_{i}(x^{\star})^{\top}p+\sum_{i=m_{1}+m_{2}+1}^{m}\mu_{i}|(\nabla c_{i}(x^{\star})^{\top}p)|\geq 0.$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 3: [[动力学或策略机制]]

$$\geq\nabla J(x^{\star})^{\top}p+\sum_{i=1}^{m_{1}}\mu_{i}\nabla c_{i}(x^{\star})^{\top}p-\sum_{i=m_{1}+1}^{m_{1}+m_{2}}\mu_{i}\nabla c_{i}(x^{\star})^{\top}p+\sum_{i=m_{1}+m_{2}+1}^{m_\mathcal{E}}\mu_{i}|\nabla c_{i}(x^{\star})^{\top}p|.$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 4: [[优化目标/约束]]

$$x_{k+1}=\operatorname*{arg\,min}_{x}\{q_{\mu,k}(x)=J(x)+\sum_{i\in\mathcal{E}}\mu_{i}\lvert c_{i}(x_{k})+\nabla c_{i}(x_{k})^{\top}(x-x_{k})\rvert\}.$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 5: [[动力学或策略机制]]

$$w_{k}=\left[\begin{array}{ccc}w_{k,1}&\dots&w_{k,m_{\mathcal{E}}}\end{array}\right]\in{{\mathbb{R}}^{m_\mathcal{E}}}$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 6: [[动力学或策略机制]]

$$=\nabla J(x_{n_{k}+1})^{\top}p+\sum_{i=1}^{m_{1}}\mu_{i}\nabla c_{i}(x_{n_{k}})^{\top}p-\sum_{i=m_{1}+1}^{m_{1}+m_{2}}\mu_{i}\nabla c_{i}(x_{n_{k}})^{\top}p+\sum_{i=m_{1}+m_{2}+1}^{m}\mu_{i}|\nabla c_{i}(x_{n_{k}})^{\top}p|.$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 7: [[动力学或策略机制]]

$$-\sum_{i=m_{1}+1}^{m_{1}+m_{2,-}}\mu_{i}\nabla c_{i}(x^{\star})^{\top}p+\sum_{i=m_{1}+m_{2,-}+1}^{m_{1}+m_{2}}\mu_{i}|\nabla c_{i}(x^{\star})^{\top}p| + $\sum_{i=m_{1}+m_{2}+1}^{m_\mathcal{E}}\mu_{i}$ |\nabla c_{i}(x^{\star})^{\top}p|.$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 8: [[动力学或策略机制]]

$$=\frac{1}{\mu mg}.\left[(\lambda_{2}+\lambda_{4}+\lambda_{6}+\lambda_{8})\cos\theta-(\lambda_{1}+\lambda_{3}+\lambda_{5}+\lambda_{7})\sin\theta\right],$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 9: [[动力学或策略机制]]

$$=\frac{1}{\mu mg}.\left[(\lambda_{2}+\lambda_{4}+\lambda_{6}+\lambda_{8})\sin\theta+(\lambda_{1}+\lambda_{3}+\lambda_{5}+\lambda_{7})\cos\theta\right],$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 10: [[动力学或策略机制]]

$$\begin{split}TC=\{d\in\mathbb{R}^{2}\mid d_{1}\geq 0,d_{2}=0\}\ \cup\ \\ \{d\in{{\mathbb{R}}^{2}}\mid d_{1}=0,d_{2}\geq 0\}.\end{split}$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。
## 关键图表

### Figure 1: Pipeline / core system figure: Geometric intuition of MPCC through Example 1

![Figure 1](https://arxiv.org/html/2502.01055v3/x2.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Pipeline / core system figure: Geometric intuition of MPCC through Example 1”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 2: Schematic of the contact-implicit motion planning tasks considered in the ex

![Figure 2](https://arxiv.org/html/2502.01055v3/x4.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Schematic of the contact-implicit motion planning tasks considered in the ex”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 3: Comparison of optimized trajectories for the hopper and waiter problems. The left and

![Figure 3](https://arxiv.org/html/2502.01055v3/x7.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Comparison of optimized trajectories for the hopper and waiter problems. The left and”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### 论文表格

#### Table 1: TABLE I: Definition of Hyperparameters in CRISP

| Paramaters | Descriptions |
| --- | --- |
| k m a x k m a x k_{max} | max iteration numbers: 1000 |
|  0  0 \Delta_{0} 0 | initial trust region radius: 1 |
|  m a x  m a x \Delta_{max} | max trust region radius: 10 |
| $\mu_{0} 0$ | initial penalty: 10 |
| $\mu_{max}$ | max penalty: 1 e 6 1 e 6 1e^{6} 1 6 |
|  l o w  l o w \eta_{low} | reduction ratio lower bound: 0.25 |
|  h i g h  h i g h \eta_{high} | reduction ratio upper bound: 0.75 |
| $\gamma_{shrink}$ | trust region shrink factor: 0.25 |
| $\gamma_{expand}$ | trust region expand factor: 2 |
|  c c \epsilon_{c} | tolerance of constraint violation: 1 e 6 1 e 6 1e^{6} 1 6 |
|  p p \epsilon_{p} | tolerance of trial step norm: 1 e - 3 1 e 3 1e^{-3} 1 - 3 |
|  r r \epsilon_{r} | tolerance of trust region radius: 1 e - 3 1 e 3 1e^{-3} 1 - 3 |

**说明**: TABLE I: Definition of Hyperparameters in CRISP

#### Table 2: TABLE II: Comparison of CRISP with benchmark solvers across different tasks.

| Methods | Cartpole with Soft Walls | Push Box | Transport | Push T | | | | | | | | | | | | | | | | |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Rate [%]Success | ErrorTracking | Violation | Iterations | Time | Rate [%]Success | ErrorTracking | Violation | Iterations | Time | Rate [%]Success | ErrorTracking | Violation | Iterations | Time | Rate [%]Success | ErrorTracking | Violation | Iterations | Time | |
| Snopt | 63.33 63.33 63.33 63.33 | 0.08 0.08 0.08 0.08 | 2.0  10 - 3 2.0 10 3 2.0\times 10^{-3} 2.0  10 - 3 | 4552.50 4552.50 4552.50 4552.50 | 12.97 12.97 12.97 12.97 | 0 0 | 4.25 4.25 4.25 4.25 | 1.0  10 - 11 1.0 10 11 1.0\times 10^{-11} 1.0  10 - 11 | 3303.00 3303.00 3303.00 3303.00 | 25.22 25.22 25.22 25.22 | 100 100  $\mathbf{100}$ | 1.7  10 - 4 1.7 10 4 1.7\times 10^{-4} 1.7  10 - 4 | 5.5  10 - 9 5.5 10 9 5.5\times 10^{-9} 5.5  10 - 9 | 5782.30 5782.30 5782.30 5782.30 | 43.98 43.98 43.98 43.98 | 33.33 33.33 33.33 33.33 | 0.67 0.67 0.67 0.67 | $\mathbf 6.4\times 10^{-8}.4 × -$ | 2257.50 2257.50 2257.50 2257.50 | 210.92 210.92 210.92 210.92 |
| Ipopt | 86.67 86.67 86.67 86.67 | 0.04 0.04 0.04 0.04 | 8.9  10 - 7 8.9 10 7 8.9\times 10^{-7} 8.9  10 - 7 | 288.21 288.21  $\mathbf 288.21.21$ | 0.51 0.51  $\mathbf 0.51.51$ | 94.44 94.44 94.44 94.44 | 0.03 0.03 0.03 0.03 | $\mathbf 1.3\times 10^{-11}.3 × -$ | 407.69 407.69  $\mathbf 407.69.69$ | 0.92 0.92 0.92 0.92 | 87.50 87.50 87.50 87.50 | 0.02 0.02 0.02 0.02 | 1.6  10 - 8 1.6 10 8 1.6\times 10^{-8} 1.6  10 - 8 | 404.96 404.96 404.96 404.96 | 1.36 1.36 1.36 1.36 | 72.22 72.22 72.22 72.22 | 0.26 0.26 0.26 0.26 | 4.9  10 - 7 4.9 10 7 4.9\times 10^{-7} 4.9  10 - 7 | 1560.20 1560.20 1560.20 1560.20 | 39.49 39.49 39.49 39.49 |
| PROXNLP | 60 60 60 60 | 0.032 0.032 0.032 0.032 | 1.4  10 - 3 1.4 10 3 1.4\times 10^{-3} 1.4  10 - 3 | 716.25 716.25 716.25 716.25 | 140.92 140.92 140.92 140.92 | 66.67 66.67 66.67 66.67 | 0.03 0.03 0.03 0.03 | 3.6  10 - 5 3.6 10 5 3.6\times 10^{-5} 3.6  10 - 5 | 2000.00 2000.00 2000.00 2000.00 | 1277.21 1277.21 1277.21 1277.21 | 59.26 59.26 59.26 59.26 | 0.01 0.01 0.01 0.01 | 1.1  10 - 3 1.1 10 3 1.1\times 10^{-3} 1.1  10 - 3 | 361.92 361.92 361.92 361.92 | 1070.01 1070.01 1070.01 1070.01 | - - - | - - - | - - - | - - - | - - - |
| CRISP (ours) | 100 100  $\mathbf{100}$ | 0.02 0.02  $\mathbf 0.02.02$ | $\mathbf 1.4\times 10^{-7}.4 × -$ | 415.39 415.39 415.39 415.39 | 1.56 1.56 1.56 1.56 | 100 100  $\mathbf{100}$ | 0.02 0.02  $\mathbf 0.02.02$ | 8.3  10 - 10 8.3 10 10 8.3\times 10^{-10} 8.3  10 - 10 | 420.69 420.69 420.69 420.69 | 0.74 0.74  $\mathbf{0.74}.74$ | 100 100  $\mathbf{100}$ | $\mathbf 1.1\times 10^{-4}.1 × -$ | $\mathbf 1.1\times 10^{-11}.1 × -$ | 286.20 286.20  $\mathbf{286.20}.20$ | 0.77 0.77  $\mathbf 0.77.77$ | 100 100  $\mathbf{100}$ | 0.01 0.01  $\mathbf 0.01.01$ | 8.7  10 - 7 8.7 10 7 8.7\times 10^{-7} 8.7  10 - 7 | 311.35 311.35  $\mathbf 311.35.35$ | 3.54 3.54  $\mathbf 3.54.54$ |

**说明**: TABLE II: Comparison of CRISP with benchmark solvers across different tasks.
## 实验解读

- 评价重点:围绕 adaptive-control、contact-implicit-optimization、接触推理、实时控制、鲁棒控制,优先看真实机器人结果、扰动恢复、接触稳定性、控制误差、成功率、实时性和 sim-to-real 差距。
- System0 视角:实验结论不只看最终成功率,还要看闭环过程中是否能处理状态估计误差、执行延迟、碰撞/接触突变和外部扰动。
- 与上层系统的关系:如果方法能在快闭环内稳定工作,它可以作为 System1 技能或 System2 规划的底层执行保障;如果只在离线或慢速设定下成立,则还不能直接作为身体级快闭环模块。
## 批判性思考

### 优点

- 论文把 adaptive-control、contact-implicit-optimization、接触推理、实时控制、鲁棒控制 放到机器人身体级闭环里讨论,和只做高层语义推理的工作相比,更接近真实机器人部署时会遇到的状态、动力学和安全约束。
- 若论文同时给出公式、系统结构图、硬件实验或消融结果,这些证据可以帮助判断方法是否真的能支撑毫秒到百毫秒级反应。

### 局限

- 需要重点核查控制频率、推理延迟、传感延迟、失败恢复、硬件磨损、未见场景泛化和安全边界;这些因素决定它能否成为可靠的 System0 模块。
- 对学习型方法,还要区分训练环境中的鲁棒性和真实部署中的鲁棒性,尤其是接触丰富任务、动态运动任务和人形机器人任务。
## 论文定位

- 研究对象:On the Surprising Robustness of Sequential Convex Optimization for Contact-Implicit Motion Planning。
- 关键词:adaptive-control、contact-implicit-optimization、接触推理、实时控制、鲁棒控制。
- System0 关联:关注机器人身体级快反应链路中“状态/接触/局部感知 -> 约束/优化/策略 -> 动作”的短闭环,不把问题停留在语言规划或任务分解层。

## 速查卡片

> [!summary] Surprising Robustness Sequential
> - **论文**: https://www.roboticsproceedings.org/rss21/p047.pdf
> - **arXiv**: http://arxiv.org/abs/2502.01055v3
> - **arXiv HTML**: https://arxiv.org/html/2502.01055v3
