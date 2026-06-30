---
title: "Safe Planning for Articulated Robots Using Reachability-based Obstacle Avoidance With Spheres"
method_name: "Safe Planning Articulated"
authors: ["Jonathan Michaux"]
year: 2024
venue: "RSS"
tags: ["real-time-control", "safe-control", "adaptive-control", "robot-generalization", "collision-avoidance", "trajectory-optimization"]
image_source: "online"
arxiv_html: "https://arxiv.org/html/2402.08857v1"
---
# Safe Planning Articulated
## 一句话总结

> Safe Planning for Articulated Robots Using Reachability-based Obstacle Avoidance With Spheres 主要落在 [[adaptive-control]]、[[certified-control]]、[[碰撞避免]]、[[inference-time-algorithm]] 这一类问题上;从 System0 视角看,它的价值在于把身体状态、接触变化、局部感知或动力学约束接到机器人低层闭环里,让动作生成更稳定、安全、实时。
## 核心贡献

1. 围绕 **Safe Planning for Articulated Robots Using Reachability-based Obstacle Avoidance With Spheres** 建立了一个与 adaptive-control、certified-control、碰撞避免、inference-time-algorithm、reachability、实时控制 相关的机器人问题设定,重点不是高层语义规划,而是身体级闭环中的状态变化、接触变化或控制约束。
2. 方法层面的关键价值在于把感知、动力学、约束优化、策略学习或硬件反馈组织成可执行的低层动作生成机制,适合放进 System0 的快反应模块中理解。
3. 对系统集成而言,这类工作补上了 System1 技能选择与电机级控制之间的一层:它负责把上层目标变成动态可行、可恢复、可安全过滤的动作。
4. 若论文包含真实机器人、接触任务或仿真到真实迁移,核心应看它是否报告了闭环稳定性、扰动恢复、硬件延迟、失败模式和跨场景泛化。
## 问题背景

这篇论文可以放在 System0 的“具身快系统”背景下阅读:机器人需要在很短时间内根据本体状态、接触、力/触觉、局部视觉或动力学预测做出动作,而不是等待慢速规划器重新推理整个任务。adaptive-control、certified-control、碰撞避免、inference-time-algorithm、reachability、实时控制、safe-control 相关问题的难点通常不在“知道要做什么”,而在动作执行过程中状态会快速偏离、接触会突然改变、约束会实时收紧,低层系统必须及时修正。

### System0 关联

精读时优先看三件事:第一,论文使用哪些快速反馈量,例如关节状态、接触状态、力觉、触觉、深度或局部视觉;第二,反馈如何进入 MPC、WBC、策略网络、安全过滤器或恢复控制器;第三,输出是否直接影响机器人下一步动作,而不是只服务离线分析或高层任务规划。
## 方法详解

### System0 视角解析

从 System0 角度看,这篇论文的方法部分应当围绕 adaptive-control、certified-control、碰撞避免、inference-time-algorithm、reachability、实时控制、safe-control、轨迹优化 来读。核心问题是:系统如何从快速变化的机器人状态中构造控制输入,如何处理接触、碰撞、约束、模型误差或执行延迟,以及如何把这些信息变成下一时刻的动作。

如果方法里有优化问题,重点看目标函数、约束和求解频率;如果方法里有学习策略,重点看观测空间、动作空间、训练扰动和部署时的闭环频率;如果方法里有硬件或传感器,重点看传感链路是否真的进入控制回路。这样读可以把论文和 System2 语言规划、System1 技能编排区分开来。

### 关键公式与机制

#### 公式 1: [[优化目标/约束]]

$$s_{d}(c;\mathcal{O})=\begin{cases}\max(Ac-b)&\text{if}c\in\mathcal{O}\\ (Ac-b)_{i_{A}}&\text{if}c\not\in\mathcal{O},\max(Ap_{\text{face}}-b)\leq 0\\ \ {E_{i_{E}}\in E}{\min}d(c;E_{i_{E}})&\text{otherwise}\end{cases},$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 2: [[动力学或策略机制]]

$${\mathrm{\text{{slice}}}}(\mathbf{P},x_{j},\sigma)\subset\mathbf{P}=\left\{z\in\mathbf{P}\,\mid\,z=\sum_{i=0}^{{n_{g}}}g_{i}x^{\alpha_{i}},\,x_{j}=\sigma\right\}.$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 3: [[动力学或策略机制]]

$$q(t;k)=\begin{cases}q_{0}+\dot{q}_{0}t+\frac{1}{2}k\,t^{2},&t\in[0,t_{p})\\ q_{0}+\dot{q}_{0}t_{p}+\frac{1}{2}k\,t_{p}^{2}+\\ (\dot{q}_{0}t_{p}+kt_{p})\frac{(2t_{\text{f}}-t_{p}-t)(t-t_{p})}{2(t_{\text{f}}-t_{p})},&t\in[t_{p},t_{\text{f}}].\end{cases}$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 4: [[动力学或策略机制]]

$$\mathbf{FK_{j}}(\mathbf{q}(\mathbf{T_{i}};\mathbf{K}))=\begin{bmatrix}\mathbf{R_{j}}(\mathbf{q}(\mathbf{T_{i}};\mathbf{K}))&\mathbf{p_{j}}(\mathbf{q}(\mathbf{T_{i}};\mathbf{K}))\\ \mathbf{0}&1\\ \end{bmatrix},$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 5: [[优化目标/约束]]

$$=\mathcal{PZ}\left(g_{i},\alpha_{i},x,\mathbf{0},\mathbf{0}\right)\oplus\mathcal{PZ}\left(\mathbf{0},\mathbf{0},\mathbf{0},h_{j},y_{j}\right)$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 6: [[动力学或策略机制]]

$$\mathbf{P}=\big{\{}z\in\mathbb{R}^{n}\,\mid\,z=c+\sum_{i=1}^{{n_{g}}}g_{i}x^{\alpha_{i}}+\sum_{j=1}^{n_{h}}h_{j}y_{j},\,x\in[-1,1]^{n_{g}},$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 7: [[优化目标/约束]]

$$\mathcal{SFO}_{j}(\mathbf{q}(\mathbf{T_{i}};k))=\left\{\bar{S}_{j,i,m}(\mathbf{q}(\mathbf{T_{i}};k))\,:\,1\leq m\leq n_{s}\right\},$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 8: [[优化目标/约束]]

$$\mathcal{SFO}_{j}(\mathbf{q}(\mathbf{T_{i}};k))=\left\{{\bar{S}_{j,i,m}(\mathbf{q}(\mathbf{T_{i}};k))}\,:\,m\in N_{s}\right\},$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 9: [[动力学或策略机制]]

$$=\biggl{(}\frac{\left\lVert \mathbf{C_{j+1}}(\mathbf{T_{i}};k)-\mathbf{C_{j}}(\mathbf{T_{i}};k)\right \rVert}{2(n_{s}-2)}-(\frac{r_{j+1,i}-r_{j,i}}{2(n_{s}-2)})\biggr{)}^{\frac{1}{2}}$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 10: [[优化目标/约束]]

$$\mathcal{SJO}=\left\{{S_{j}(\mathbf{q}(\mathbf{T_{i}};\mathbf{K}))}\,:\,\forall i\in N_{t},j\in N_{q}\right\}.$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。
## 关键图表

### Figure 1: Pipeline / core system figure: This paper presents SPARROWS, a method that is capable of generating safe motion plan

![Figure 1](https://arxiv.org/html/2402.08857v1/x1.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Pipeline / core system figure: This paper presents SPARROWS, a method that is capable of generating safe motion plan”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 2: A visualization of the robot arm and its environment. The obstacles are shown in red a

![Figure 2](https://arxiv.org/html/2402.08857v1/x2.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“A visualization of the robot arm and its environment. The obstacles are shown in red a”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 3: A visualization of the Spherical Forward Occupancy construction for a robotic arm in 3

![Figure 3](https://arxiv.org/html/2402.08857v1/x3.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“A visualization of the Spherical Forward Occupancy construction for a robotic arm in 3”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### 论文表格

#### Table 1: TABLE I: Summary of polynomial zonotope operations.

| Operation | Computation |
| --- | --- |
| P 1 ⊕ P 2 direct-sum P 1 P 2  $\mathbf{P}_{1}\oplus\mathbf{P}_{2} 1 ⊕ 2 (Minkowski Sum) [9, eq. (10)]$ | Exact |
| P 1 P 2 P 1 P 2  $\mathbf{P}_{1}\mathbf{P}_{2} 1 2 (PZ Multiplication) [9, eq. (11)]$ | Exact |
| $\mathrm{\text{{slice}}}}(\mathbf{P},x_{j},\sigma) slice$ | Exact |
| sup (P) sup P { $\mathrm{\text{{sup}}}}(\mathbf{P}) sup () (5) and inf (P) (6) inf P (6) {\mathrm{\text{{inf}}}}(\mathbf{P}) inf () ()$ | Overapproximative |

**说明**: TABLE I: Summary of polynomial zonotope operations.

#### Table 2: TABLE II: Mean runtime for constraint and constraint gradient evaluation for SPARROWS and ARMTD in Kinova planning exper

| Methods | mean constraint evaluation time [ms] | | |
| --- | --- | --- | --- |
| # Obstacles (s) | 10 | 20 | 40 |
| SPARROWS $\pi/24) (/ 24)$ | 3.1 ± 0.2 | 3.8 ± 0.1 | 5.1 ± 0.1 |
| $\pi/24) (/ 24)$ | 4.1 ± 0.3 | 5.4 ± 0.5 | 8.3 ± 0.7 |
| SPARROWS $\pi/6) (/ 6)$ | 3.1 ± 0.1 | 3.8 ± 0.1 | 5.2 ± 0.1 |
| $\pi/6) (/ 6)$ | 4.1 ± 0.3 | 5.4 ± 0.5 | 8.1 ± 0.7 |

**说明**: TABLE II: Mean runtime for constraint and constraint gradient evaluation for SPARROWS and ARMTD in Kinova planning experiments with 10, 20, and 40 obstacles 0.5s time limit ↓ ↓ $\downarrow$ ↓

#### Table 3: TABLE III: Mean per-step planning time for SPARROWS and ARMTD in Kinova planning experiments with 10, 20, and 40 obstacl

| Methods | mean planning time [s] | | |
| --- | --- | --- | --- |
| # Obstacles (s) | 10 | 20 | 40 |
| SPARROWS $\pi/24) (/ 24)$ | 0.14 ± 0.07 | 0.16 ± 0.08 | 0.20 ± 0.09 |
| $\pi/24) (/ 24)$ | 0.18 ± 0.08 | 0.30 ± 0.10 | 0.45 ± 0.08 |
| SPARROWS $\pi/6) (/ 6)$ | 0.16 ± 0.08 | 0.18 ± 0.08 | 0.24 ± 0.09 |
| $\pi/6) (/ 6)$ | 0.24 ± 0.09 | 0.38 ± 0.10 | 0.51 ± 0.04 |

**说明**: TABLE III: Mean per-step planning time for SPARROWS and ARMTD in Kinova planning experiments with 10, 20, and 40 obstacles 0.5s time limit ↓ ↓ $\downarrow$ ↓. Red indicates that the average planning time limit has been exceeded.

#### Table 4: TABLE IV: Mean runtime for constraint and constraint gradient evaluation for SPARROWS and ARMTD in 2 and 3 Kinvoa arms p

| Methods | mean constraint evaluation time [ms] | | | | | |
| --- | --- | --- | --- | --- | --- | --- |
| # DOF | 14 | 21 | | | | |
| # Obstacles | 5 | 10 | 15 | 5 | 10 | 15 |
| SPARROWS $\pi/24) (/ 24)$ | 3.9 ± 0.2 | 4.6 ± 0.2 | 5.4 ± 0.1 | 4.8 ± 0.2 | 6.0 ± 0.1 | 7.2 ± 0.1 |
| $\pi/24) (/ 24)$ | 8.4 ± 0.6 | 9.4 ± 0.7 | 10.7 ± 0.8 | 12.9 ± 0.9 | 14.5 ± 0.7 | 16.1 ± 0.8 |
| SPARROWS $\pi/6) (/ 6)$ | 3.9 ± 0.3 | 4.6 ± 0.2 | 5.4 ± 0.1 | 4.8 ± 0.1 | 6.0 ± 0.1 | 7.2 ± 0.1 |
| $\pi/6) (/ 6)$ | 8.4 ± 0.6 | 9.4 ± 0.7 | 10.6 ± 0.7 | 12.8 ± 0.8 | 14.5 ± 0.7 | 16.7 ± 1.4 |

**说明**: TABLE IV: Mean runtime for constraint and constraint gradient evaluation for SPARROWS and ARMTD in 2 and 3 Kinvoa arms planning experiment with 5, 10, 15 obstacles 1.0s time limit ↓ ↓ $\downarrow$ ↓

#### Table 5: TABLE V: Mean per-step planning time for SPARROWS and ARMTD in 2 and 3 Kinvoa arms planning experiments with 5, 10, 15 o

| Methods | mean planning time [ms] | | | | | |
| --- | --- | --- | --- | --- | --- | --- |
| # DOF | 14 | 21 | | | | |
| # Obstacles | 5 | 10 | 15 | 5 | 10 | 15 |
| SPARROWS $\pi/24) (/ 24)$ | 0.33 ± 0.10 | 0.37 ± 0.14 | 0.41 ± 0.16 | 0.63 ± 0.14 | 0.67 ± 0.16 | 0.71 ± 0.16 |
| $\pi/24) (/ 24)$ | 0.38 ± 0.10 | 0.57 ± 0.16 | 0.76 ± 0.17 | 0.70 ± 0.13 | 0.98 ± 0.07 | 1.07 ± 0.03 |
| SPARROWS $\pi/6) (/ 6)$ | 0.38 ± 0.12 | 0.46 ± 0.16 | 0.50 ± 0.17 | 0.72 ± 0.16 | 0.79 ± 0.17 | 0.85 ± 0.16 |
| $\pi/6) (/ 6)$ | 0.49 ± 0.16 | 0.72 ± 0.18 | 0.91 ± 0.15 | 0.81 ± 0.15 | 1.03 ± 0.03 | 1.06 ± 0.03 |

**说明**: TABLE V: Mean per-step planning time for SPARROWS and ARMTD in 2 and 3 Kinvoa arms planning experiments with 5, 10, 15 obstacles 1.0s time limit ↓ ↓ $\downarrow$ ↓

#### Table 6: TABLE VI: Number of successes for SPARROWS, ARMTD, MPOT, CHOMP, and Trajopt in Kinova planning experiment with 10, 2

| Methods | # Successes | | |
| --- | --- | --- | --- |
| # Obstacles | 10 | 20 | 40 |
| SPARROWS $\pi/24) (/ 24)$ | 79 | 61 | 34 |
| $\pi/24) (/ 24)$ | 79 | 50 | 2 |
| SPARROWS $\pi/6) (/ 6)$ | 87 | 62 | 40 |
| $\pi/6) (/ 6)$ | 56 | 17 | 0 |
| CHOMP [2 ] | 30 | 9 | 4 |
| TrajOpt [3 ] | 33 (67) | 9 (91) | 6 (94) |
| MPOT [29 ] | 57 (43) | 22 (78) | 9 (91) |

**说明**: TABLE VI: Number of successes for SPARROWS, ARMTD, MPOT, CHOMP, and Trajopt in Kinova planning experiment with 10, 20, and 40 Random Obstacles ↑ ↑ $\uparrow$ ↑. Red indicates the number of failures due to collision.

#### Table 7: TABLE VII: Number of successes for SPARROWS and ARMTD in 2 and 3 Kinvoa arms planning experiment with 5, 10, 15 obstacle

| Methods | # Successes | | | | | |
| --- | --- | --- | --- | --- | --- | --- |
| # DOF | 14 | 21 | | | | |
| # Obstacles | 5 | 10 | 15 | 5 | 10 | 15 |
| SPARROWS $\pi/24) (/ 24)$ | 92 | 79 | 62 | 73 | 46 | 29 |
| $\pi/24) (/ 24)$ | 96 | 75 | 34 | 73 | 0 | 0 |
| SPARROWS $\pi/6) (/ 6)$ | 94 | 83 | 68 | 67 | 36 | 13 |
| $\pi/6) (/ 6)$ | 83 | 39 | 6 | 41 | 0 | 0 |

**说明**: TABLE VII: Number of successes for SPARROWS and ARMTD in 2 and 3 Kinvoa arms planning experiment with 5, 10, 15 obstacles 1.0s time limit ↑ ↑ $\uparrow$ ↑

#### Table 8: TABLE VIII: Number of success for SPARROWS, ARMTD, MPOT, CHOMP, and TrajOpt in Kinova planning experiment on 14 Hard

| Methods | # Successes |
| --- | --- |
| SPARROWS $\pi/24) (/ 24)$ | 3 |
| $\pi/24) (/ 24)$ | 3 |
| SPARROWS $\pi/6) (/ 6)$ | 5 |
| $\pi/6) (/ 6)$ | 4 |
| CHOMP [2 ] | 2 |
| TrajOpt [3 ] | 0 (14) |
| MPOT [29 ] | 6 (8) |

**说明**: TABLE VIII: Number of success for SPARROWS, ARMTD, MPOT, CHOMP, and TrajOpt in Kinova planning experiment on 14 Hard Scenarios ↑ ↑ $\uparrow$ ↑. Red indicates the number of failures due to collision.

#### Table 9: TABLE IX: Mean runtime for constraint and constraint gradient evaluation for SPARROWS and ARMTD in Kinova planning exper

| Methods | mean constraint evaluation time [ms] | | |
| --- | --- | --- | --- |
| # Obstacles (s) | 10 | 20 | 40 |
| SPARROWS $\pi/24) (/ 24)$ | 3.1 ± 0.1 | 3.8 ± 0.1 | 5.1 ± 0.1 |
| $\pi/24) (/ 24)$ | 4.1 ± 0.3 | 5.4 ± 0.5 | 8.1 ± 0.6 |
| SPARROWS $\pi/6) (/ 6)$ | 3.1 ± 0.1 | 3.8 ± 0.1 | 5.1 ± 0.1 |
| $\pi/6) (/ 6)$ | 4.1 ± 0.3 | 5.4 ± 0.4 | 7.6 ± 0.3 |

**说明**: TABLE IX: Mean runtime for constraint and constraint gradient evaluation for SPARROWS and ARMTD in Kinova planning experiments with 10, 20, and 40 obstacles 0.25s time limit ↓ ↓ $\downarrow$ ↓

#### Table 10: TABLE X: Mean runtime for constraint and constraint gradient evaluation for SPARROWS and ARMTD in Kinova planning experi

| Methods | mean constraint evaluation time [ms] | | |
| --- | --- | --- | --- |
| # Obstacles (s) | 10 | 20 | 40 |
| SPARROWS $\pi/24) (/ 24)$ | 3.1 ± 0.1 | 3.8 ± 0.1 | 5.1 ± 0.1 |
| $\pi/24) (/ 24)$ | 4.1 ± 0.3 | 5.3 ± 0.1 | 7.8 ± 0.4 |
| SPARROWS $\pi/6) (/ 6)$ | 3.1 ± 0.1 | 3.8 ± 0.1 | 5.1 ± 0.1 |
| $\pi/6) (/ 6)$ | 4.0 ± 0.2 | 5.2 ± 0.2 | 8.8 ± 0.7 |

**说明**: TABLE X: Mean runtime for constraint and constraint gradient evaluation for SPARROWS and ARMTD in Kinova planning experiments with 10, 20, and 40 obstacles 0.15s time limit ↓ ↓ $\downarrow$ ↓

#### Table 11: TABLE XI: Mean per-step planning time for SPARROWS and ARMTD in Kinova planning experiment with 10, 20, and 40 obstacles

| Methods | mean planning time [s] | | |
| --- | --- | --- | --- |
| # Obstacles (s) | 10 | 20 | 40 |
| SPARROWS $\pi/24) (/ 24)$ | 0.12 ± 0.04 | 0.14 ± 0.05 | 0.17 ± 0.05 |
| $\pi/24) (/ 24)$ | 0.16 ± 0.04 | 0.24 ± 0.03 | 0.27 ± 0.02 |
| SPARROWS $\pi/6) (/ 6)$ | 0.14 ± 0.04 | 0.16 ± 0.04 | 0.20 ± 0.04 |
| $\pi/6) (/ 6)$ | 0.20 ± 0.04 | 0.26 ± 0.02 | 0.27 ± 0.02 |

**说明**: TABLE XI: Mean per-step planning time for SPARROWS and ARMTD in Kinova planning experiment with 10, 20, and 40 obstacles 0.25s time limit ↓ ↓ $\downarrow$ ↓. Red indicates that the average planning time limit has been exceeded.

#### Table 12: TABLE XII: Mean per-step planning time for SPARROWS and ARMTD in Kinova planning experiment with 10, 20, and 40 obstacle

| Methods | mean planning time [s] | | |
| --- | --- | --- | --- |
| # Obstacles (s) | 10 | 20 | 40 |
| SPARROWS $\pi/24) (/ 24)$ | 0.11 ± 0.02 | 0.13 ± 0.02 | 0.14 ± 0.02 |
| $\pi/24) (/ 24)$ | 0.14 ± 0.02 | 0.16 ± 0.03 | 0.17 ± 0.03 |
| SPARROWS $\pi/6) (/ 6)$ | 0.13 ± 0.02 | 0.14 ± 0.02 | 0.15 ± 0.02 |
| $\pi/6) (/ 6)$ | 0.16 ± 0.02 | 0.16 ± 0.03 | 0.18 ± 0.03 |

**说明**: TABLE XII: Mean per-step planning time for SPARROWS and ARMTD in Kinova planning experiment with 10, 20, and 40 obstacles 0.15s time limit ↓ ↓ $\downarrow$ ↓. Red indicates that the average planning time limit has been exceeded.

#### Table 13: TABLE XIII: Number of successes for SPARROWS and ARMTD in 2 and 3 Kinvoa arms planning experiment with 5, 10, 15 obstacl

| Methods | # Successes | | | | | |
| --- | --- | --- | --- | --- | --- | --- |
| # DOF | 14 | 21 | | | | |
| # Obstacles | 5 | 10 | 15 | 5 | 10 | 15 |
| SPARROWS $\pi/24) (/ 24)$ | 74 | 47 | 27 | 0 | 0 | 0 |
| $\pi/24) (/ 24)$ | 70 | 3 | 0 | 0 | 0 | 0 |
| SPARROWS $\pi/6) (/ 6)$ | 55 | 42 | 12 | 0 | 0 | 0 |
| $\pi/6) (/ 6)$ | 35 | 0 | 0 | 0 | 0 | 0 |

**说明**: TABLE XIII: Number of successes for SPARROWS and ARMTD in 2 and 3 Kinvoa arms planning experiment with 5, 10, 15 obstacles 0.5s time limit ↑ ↑ $\uparrow$ ↑

#### Table 14: TABLE XIV: Mean runtime for a constraint and constraint gradient evaluation for SPARROWS and ARMTD in 2 and 3 Kinvoa arm

| Methods | mean constraint evaluation time [ms] | | | | | |
| --- | --- | --- | --- | --- | --- | --- |
| # DOF | 14 | 21 | | | | |
| # Obstacles | 5 | 10 | 15 | 5 | 10 | 15 |
| SPARROWS $\pi/24) (/ 24)$ | 3.9 ± 0.2 | 4.7 ± 0.1 | 5.4 ± 0.2 | 4.9 ± 0.2 | 6.0 ± 0.1 | 7.2 ± 0.1 |
| $\pi/24) (/ 24)$ | 8.4 ± 0.6 | 9.3 ± 0.6 | 10.6 ± 0.4 | 12.6 ± 0.6 | 14.3 ± 0.5 | 17.1 ± 1.3 |
| SPARROWS $\pi/6) (/ 6)$ | 3.9 ± 0.1 | 4.6 ± 0.1 | 5.4 ± 0.1 | 4.8 ± 0.1 | 6.0 ± 0.2 | 7.2 ± 0.2 |
| $\pi/6) (/ 6)$ | 8.3 ± 0.6 | 9.0 ± 0.2 | 10.6 ± 0.1 | 13.0 ± 0.7 | 14.6 ± 0.8 | 16.4 ± 0.9 |

**说明**: TABLE XIV: Mean runtime for a constraint and constraint gradient evaluation for SPARROWS and ARMTD in 2 and 3 Kinvoa arms planning experiment with 5, 10, 15 obstacles 0.5s time limit ↓ ↓ $\downarrow$ ↓

#### Table 15: TABLE XV: Mean per-step planning time for SPARROWS and ARMTD in 2 and 3 Kinvoa arms planning experiment with 5, 10, 15 o

| Methods | mean planning time [s] | | | | | |
| --- | --- | --- | --- | --- | --- | --- |
| # DOF | 14 | 21 | | | | |
| # Obstacles | 5 | 10 | 15 | 5 | 10 | 15 |
| SPARROWS $\pi/24) (/ 24)$ | 0.32 ± 0.06 | 0.35 ± 0.07 | 0.37 ± 0.08 | 0.51 ± 0.02 | 0.51 ± 0.01 | 0.52 ± 0.02 |
| $\pi/24) (/ 24)$ | 0.37 ± 0.06 | 0.49 ± 0.04 | 0.52 ± 0.02 | 0.54 ± 0.02 | 0.55 ± 0.03 | 0.55 ± 0.03 |
| SPARROWS $\pi/6) (/ 6)$ | 0.37 ± 0.08 | 0.40 ± 0.08 | 0.44 ± 0.07 | 0.53 ± 0.02 | 0.52 ± 0.01 | 0.53 ± 0.01 |
| $\pi/6) (/ 6)$ | 0.42 ± 0.07 | 0.52 ± 0.02 | 0.54 ± 0.14 | 0.53 ± 0.02 | 0.56 ± 0.02 | 0.54 ± 0.02 |

**说明**: TABLE XV: Mean per-step planning time for SPARROWS and ARMTD in 2 and 3 Kinvoa arms planning experiment with 5, 10, 15 obstacles 0.5s time limit ↓ ↓ $\downarrow$ ↓
## 实验解读

- 评价重点:围绕 adaptive-control、certified-control、碰撞避免、inference-time-algorithm、reachability,优先看真实机器人结果、扰动恢复、接触稳定性、控制误差、成功率、实时性和 sim-to-real 差距。
- System0 视角:实验结论不只看最终成功率,还要看闭环过程中是否能处理状态估计误差、执行延迟、碰撞/接触突变和外部扰动。
- 与上层系统的关系:如果方法能在快闭环内稳定工作,它可以作为 System1 技能或 System2 规划的底层执行保障;如果只在离线或慢速设定下成立,则还不能直接作为身体级快闭环模块。
## 批判性思考

### 优点

- 论文把 adaptive-control、certified-control、碰撞避免、inference-time-algorithm、reachability 放到机器人身体级闭环里讨论,和只做高层语义推理的工作相比,更接近真实机器人部署时会遇到的状态、动力学和安全约束。
- 若论文同时给出公式、系统结构图、硬件实验或消融结果,这些证据可以帮助判断方法是否真的能支撑毫秒到百毫秒级反应。

### 局限

- 需要重点核查控制频率、推理延迟、传感延迟、失败恢复、硬件磨损、未见场景泛化和安全边界;这些因素决定它能否成为可靠的 System0 模块。
- 对学习型方法,还要区分训练环境中的鲁棒性和真实部署中的鲁棒性,尤其是接触丰富任务、动态运动任务和人形机器人任务。
## 论文定位

- 研究对象:Safe Planning for Articulated Robots Using Reachability-based Obstacle Avoidance With Spheres。
- 关键词:adaptive-control、certified-control、碰撞避免、inference-time-algorithm、reachability、实时控制、safe-control、轨迹优化。
- System0 关联:关注机器人身体级快反应链路中“状态/接触/局部感知 -> 约束/优化/策略 -> 动作”的短闭环,不把问题停留在语言规划或任务分解层。

## 速查卡片

> [!summary] Safe Planning Articulated
> - **论文**: https://www.roboticsproceedings.org/rss20/p035.pdf
> - **arXiv**: http://arxiv.org/abs/2402.08857v1
> - **arXiv HTML**: https://arxiv.org/html/2402.08857v1
