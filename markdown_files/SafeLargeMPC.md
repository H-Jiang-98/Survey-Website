---
title: "Safe Large-Scale Robust Nonlinear MPC in Milliseconds via Reachability-Constrained System Level Synthesis on the GPU"
method_name: "Safe Large MPC"
authors: ["Jeffrey Fang"]
year: 2026
venue: "RSS"
tags: ["robust-control", "real-time-control", "legged-locomotion", "safe-control", "adaptive-control", "robot-generalization", "closed-loop-control", "model-predictive-control", "whole-body-control"]
image_source: "online"
arxiv_html: "https://arxiv.org/html/2604.07644v1"
---
# Safe Large MPC
## 一句话总结

> Safe Large-Scale Robust Nonlinear MPC in Milliseconds via Reachability-Constrained System Level Synthesis on the GPU 主要落在 [[adaptive-control]]、[[certified-control]]、[[closed-loop-control]]、[[足式运动]] 这一类问题上;从 System0 视角看,它的价值在于把身体状态、接触变化、局部感知或动力学约束接到机器人低层闭环里,让动作生成更稳定、安全、实时。
## 核心贡献

1. 围绕 **Safe Large-Scale Robust Nonlinear MPC in Milliseconds via Reachability-Constrained System Level Synthesis on the GPU** 建立了一个与 adaptive-control、certified-control、closed-loop-control、足式运动、模型预测控制、quadruped 相关的机器人问题设定,重点不是高层语义规划,而是身体级闭环中的状态变化、接触变化或控制约束。
2. 方法层面的关键价值在于把感知、动力学、约束优化、策略学习或硬件反馈组织成可执行的低层动作生成机制,适合放进 System0 的快反应模块中理解。
3. 对系统集成而言,这类工作补上了 System1 技能选择与电机级控制之间的一层:它负责把上层目标变成动态可行、可恢复、可安全过滤的动作。
4. 若论文包含真实机器人、接触任务或仿真到真实迁移,核心应看它是否报告了闭环稳定性、扰动恢复、硬件延迟、失败模式和跨场景泛化。
## 问题背景

这篇论文可以放在 System0 的“具身快系统”背景下阅读:机器人需要在很短时间内根据本体状态、接触、力/触觉、局部视觉或动力学预测做出动作,而不是等待慢速规划器重新推理整个任务。adaptive-control、certified-control、closed-loop-control、足式运动、模型预测控制、quadruped、reachability 相关问题的难点通常不在“知道要做什么”,而在动作执行过程中状态会快速偏离、接触会突然改变、约束会实时收紧,低层系统必须及时修正。

### System0 关联

精读时优先看三件事:第一,论文使用哪些快速反馈量,例如关节状态、接触状态、力觉、触觉、深度或局部视觉;第二,反馈如何进入 MPC、WBC、策略网络、安全过滤器或恢复控制器;第三,输出是否直接影响机器人下一步动作,而不是只服务离线分析或高层任务规划。
## 方法详解

### System0 视角解析

从 System0 角度看,这篇论文的方法部分应当围绕 adaptive-control、certified-control、closed-loop-control、足式运动、模型预测控制、quadruped、reachability、实时控制 来读。核心问题是:系统如何从快速变化的机器人状态中构造控制输入,如何处理接触、碰撞、约束、模型误差或执行延迟,以及如何把这些信息变成下一时刻的动作。

如果方法里有优化问题,重点看目标函数、约束和求解频率;如果方法里有学习策略,重点看观测空间、动作空间、训练扰动和部署时的闭环频率;如果方法里有硬件或传感器,重点看传感链路是否真的进入控制回路。这样读可以把论文和 System2 语言规划、System1 技能编排区分开来。

### 关键公式与机制

#### 公式 1: [[动力学或策略机制]]

$$\begin{array}{l@{\quad}l}\hat{Q}_{k}=Q_{k}+\rho C_{k}^{\top}C_{k},&\hat{q}_{k}=q_{k}+C_{k}^{\top}(\lambda_{k}-\rho z_{k}),\\ \hat{R}_{k}=R_{k}+\rho D_{k}^{\top}D_{k},&\hat{r}_{k}=r_{k}+D_{k}^{\top}(\lambda_{k}-\rho z_{k}),\\ \hat{S}_{k}=S_{k}+\rho C_{k}^{\top}D_{k},&\forall k\in[N],\\ \hat{Q}_{N}=Q_{N}+\rho C_{N}^{\top}C_{N},&\hat{q}_{N}=q_{N}+C_{N}^{\top}(\lambda_{N}-\rho z_{N}).\end{array}$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 2: [[优化目标/约束]]

$$\begin{aligned} &\mathbf{\Phi}^{\text{x}}_{j+1,j}=E_{j},\\ &\mathbf{\Phi}^{u}_{k,j}=\mathcal{K}_{k}^{(j)}\mathbf{\Phi}^{\text{x}}_{k,j},&\mathbf{\Phi}^{\text{x}}_{k+1,j}=(A^{(s)}_{k}+B^{(s)}_{k}\mathcal{K}_{k}^{(j)})\mathbf{\Phi}^{\text{x}}_{k,j},\\ \end{aligned}$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 3: [[优化目标/约束]]

$$=\sum_{j=0}^{N-1}\Big(\sum_{k=j}^{N-1}\!\left(\lVert \bar{Q}^{\frac{1}{2}}\mathbf{\Phi}^{\mathrm{x}}_{k,j} \rVert_{\mathcal{F}}^{2}+\lVert \bar{R}^{\frac{1}{2}}\mathbf{\Phi}^{\mathrm{u}}_{k,j} \rVert_{\mathcal{F}}^{2}\right)$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 4: [[优化目标/约束]]

$$\mathcal{Q}_{k,j}=\left(\text{diag}\left(\sqrt{\tau_{k,j}}\right)\begin{bmatrix}C_{k}^{(s)}&D_{k}^{(s)}\end{bmatrix},\begin{bmatrix}\bar{Q}^{\frac{1}{2}}&0\\ 0&\bar{R}^{\frac{1}{2}}\end{bmatrix}\right),$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 5: [[优化目标/约束]]

$$\sum_{j=0}^{N-1}\left(\sum_{k=j}^{N-1}\lVert \mathcal{Q}_{k,j}\mathbf{\Phi}_{k,j} \rVert^{2}_{\mathcal{F}}+\lVert \mathcal{Q}_{N,j}\mathbf{\Phi}^{\text{x}}_{N,j} \rVert^{2}_{\mathcal{F}}\right)$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 6: [[优化目标/约束]]

$$\begin{bmatrix}Q_{k}&S_{k}^{\top}\\ S_{k}&R_{k}\end{bmatrix}=\nabla^{2}_{(x_{k},u_{k})}\mathcal{L}_{k}\mid_{\xi_{k}^{(s)}},\quad Q_{N}=\nabla^{2}_{x_{k}}\mathcal{L}_{N}\mid_{x_{N}^{(s)}},$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 7: [[动力学或策略机制]]

$$J_{\text{QP}}(\delta x,\delta u):=\sum_{k=0}^{N-1}\frac{1}{2}\begin{bmatrix}\delta x_{k}\\ \delta u_{k}\end{bmatrix}^{\top}\begin{bmatrix}Q_{k}&S_{k}^{\top}\\ S_{k}&R_{k}\end{bmatrix}\begin{bmatrix}\delta x_{k}\\ \delta u_{k}\end{bmatrix}$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 8: [[动力学或策略机制]]

$$\sum_{k=0}^{N-1}\frac{1}{2}\begin{bmatrix}\delta x_{k}\\ \delta u_{k}\end{bmatrix}^{\top}\begin{bmatrix}\hat{Q}_{k}&\hat{S}_{k}^{\top}\\ \hat{S}_{k}&\hat{R}_{k}\end{bmatrix}\begin{bmatrix}\delta x_{k}\\ \delta u_{k}\end{bmatrix}+\begin{bmatrix}\hat{q}_{k}\\ \hat{r}_{k}\end{bmatrix}^{\top}\begin{bmatrix}\delta x_{k}\\ \delta u_{k}\end{bmatrix}$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 9: [[优化目标/约束]]

$$\mathcal{V}^{(j)}_{i\rightarrow l}(\mathbf{\Phi}_{i,j},\mathbf{\Phi}_{l,j})=\max_{\theta}$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 10: [[动力学或策略机制]]

$$h_{k}(\mathbf{\Phi})=\textstyle\sum_{j=0}^{k-1}\big\lVert \big(C^{(s)}_{k}\big)\mathbf{\Phi}^{\mathrm{x}}_{k,j}+\big(D^{(s)}_{k}\big)\mathbf{\Phi}^{\mathrm{u}}_{k,j}\big \rVert_{2,\mathrm{row}}$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。
## 关键图表

### Figure 1: Schematic of our method, GPU-SLS. At each SQP iteration (s) (s), we compute a nomin

![Figure 1](https://arxiv.org/html/2604.07644v1/x2.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Schematic of our method, GPU-SLS. At each SQP iteration (s) (s), we compute a nomin”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 2: a): Comparison of average solve-time scaling with horizon length for various NMPC so

![Figure 2](https://arxiv.org/html/2604.07644v1/x3.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“a): Comparison of average solve-time scaling with horizon length for various NMPC so”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 3: Simulated rollouts of 8 adversarial disturbances. Using the controller (14) th

![Figure 3](https://arxiv.org/html/2604.07644v1/x9.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Simulated rollouts of 8 adversarial disturbances. Using the controller (14) th”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### 论文表格

#### Table 1: Table 1

| | x k + 1 = f  (x k, u k) + E  (x k)  w k, x_{k+1}=f(x_{k},u_{k})+E(x_{k})w_{k}, | | (1) |
| --- | --- | --- | --- |

**说明**: Table 1

#### Table 2: Table 2

| | min \displaystyle\min $\pi(\cdot)}\quad$ | J x ̄ \displaystyle J $\left(\bar{x},\pi(\cdot)\right)$ | | (2a) |
| --- | --- | --- | --- | --- |
| | s.t. | x k + 1 = f  (x k, u k) + E  (x k)  w k, ∀ k ∈ [N ], \displaystyle x_{k+1}=f(x_{k},u_{k})+E(x_{k})w_{k},\quad\forall k\in[N], | | (2b) |
| | | x 0 = x  ̄ 0, \displaystyle x_{0}=\bar{x}_{0}, | | (2c) |
| | | u k k x k ∀ k ∈ N \displaystyle u k $\pi_{k}(x_{0:k}),\quad\forall k\in[N],$ | | (2d) |
| | | g  (x k, u k) ≤ 0, ∀ k ∈ [N ], ∀ w k ∈ E n x, \displaystyle g(x_{k},u_{k})\leq 0,\quad\forall k\in[N],\quad\forall w_{k}\in $\mathcal{E}_{n_{x}},$ | | (2e) |
| | | g f  (x N) ≤ 0, ∀ w N ∈ E n x, \displaystyle g^{f}(x_{N})\leq 0,\quad\forall w_{N}\in $\mathcal{E}_{n_{x}},$ | | (2f) |

**说明**: Table 2

#### Table 3: Table 3

| | min X, U \displaystyle\min_{\begin{subarray}{c}X,\,U\end{subarray}}\quad | J  (X, U):= l f  (x N) + ∑ k = 0 N - 1 l  (x k, u k) \displaystyle J(X,U):=\ell_{f}(x_{N})+ $\textstyle\sum_{k=0}^{N-1}\ell(x_{k},u_{k})$ | | (3a) |
| --- | --- | --- | --- | --- |
| | s.t. | x k + 1 = f  (x k, u k), ∀ k ∈ [N ], \displaystyle x_{k+1}=f(x_{k},u_{k}),\quad\quad\forall k\in[N], | | (3b) |
| | | x 0 = x  ̄ 0, \displaystyle x_{0}=\bar{x}_{0}, | | (3c) |
| | | g  (x k, u k) ≤ 0, ∀ k ∈ [N ], g f  (x N) ≤ 0, \displaystyle g(x_{k},u_{k})\leq 0,\forall k\in[N],\quad g^{f}(x_{N})\leq 0, | | (3d) |

**说明**: Table 3

#### Table 4: Table 4

| | L X U \displaystyle $\mathcal{L}(X,U,\mu,\gamma,\nu)$ | J X U x ̄ - x \displaystyle J X U $\mu_{0}^{\top}(\bar{x}_{0}-x_{0})$ | | (4) |
| --- | --- | --- | --- | --- |
| | | ∑ k N - k f x k u k - x k \displaystyle\quad $\textstyle\sum^{N-1}_{k=0}\mu_{k+1}^{\top}\left(f(x_{k},u_{k})-x_{k+1}\right)$ | | |
| | | ∑ k N - k g x k u k g f x N \displaystyle\quad $\textstyle\sum^{N-1}_{k=0}\gamma_{k}^{\top}g(x_{k},u_{k})+\nu^{\top}g^{f}(x_{N}),$ | | |

**说明**: Table 4

#### Table 5: Table 5

| | min x u \displaystyle\min $\delta x,\delta u}\quad$ | J QP x u ∑ k N - x k u k Q k S k S k R k x k u k \displaystyle J $\text{QP}}(\delta x,\delta u):=\sum_{k=0}^{N-1}\frac{1}{2}\begin{bmatrix}\delta x_{k}\\ \delta u_{k}\end{bmatrix}^{\top}\begin{bmatrix}Q_{k}&S_{k}^{\top}\\ S_{k}&R_{k}\end{bmatrix}\begin{bmatrix}\delta x_{k}\\ \delta u_{k}\end{bmatrix}$ | | |
| --- | --- | --- | --- | --- |
| | | q k r k x k u k x N Q N x N q N x N \displaystyle \begin bmatrix q k \\ r k \end bmatrix \top \begin bmatrix $\delta x_{k}\\ \delta u_{k}\end{bmatrix}+\frac{1}{2}\delta x_{N}^{\top}Q_{N}\delta x_{N}+q^{\top}_{N}\delta x_{N}$ | | (5a) |
| | s.t. | x k A k x k B k u k b k ∀ k ∈ N \displaystyle $\delta x_{k+1}=A_{k}\delta x_{k}+B_{k}\delta u_{k}+b_{k},\quad\forall k\in[N],$ | | (5b) |
| | | x x ̄ - x s \displaystyle $\delta x_{0}=\bar{x}_{0}-x^{(s)}_{0},$ | | (5c) |
| | | C k x k D k u k ≤ f k ∀ k ∈ N \displaystyle C k $\delta x_{k}+D_{k}\delta u_{k}\leq f_{k},\quad\forall k\in[N],$ | | (5d) |
| | | C N x N ≤ f N \displaystyle C N $\delta x_{N}\leq f_{N},$ | | (5e) |

**说明**: Table 5

#### Table 6: Table 6

| | x k x k - x k s u k u k - u k s \displaystyle $\delta x_{k}=x_{k}-x_{k}^{(s)}\qquad\delta u_{k}=u_{k}-u_{k}^{(s)}$ | | (6a) |
| --- | --- | --- | --- |
| | A k = ∇ x f ∣  k (s), B k = ∇ u f ∣  k (s), \displaystyle A_{k}=\nabla_{x}f\mid_{\xi_{k}^{(s)}},\qquad B_{k}=\nabla_{u}f\mid_{\xi_{k}^{(s)}}, | | (6b) |
| | C k = ∇ x g ∣  k (s), D k = ∇ u g ∣  k (s), \displaystyle C_{k}=\nabla_{x}g\mid_{\xi_{k}^{(s)}},\qquad D_{k}=\nabla_{u}g\mid_{\xi_{k}^{(s)}}, | | (6c) |
| | C N = ∇ x g f ∣ x N (s), \displaystyle C_{N}=\nabla_{x}g_{f}\mid_{x_{N}^{(s)}}, | | (6d) |
| | f k = - g  (x k (s), u k (s)), f N = - g f  (x N (s)). \displaystyle f_{k}=-g(x_{k}^{(s)},u_{k}^{(s)}),\quad f_{N}=-g_{f}(x_{N}^{(s)}). | | (6e) |

**说明**: Table 6

#### Table 7: Table 7

| | Q k S k S k R k ∇ x k u k L k ∣ k s Q N ∇ x k L N ∣ x N s \displaystyle\begin bmatrix Q k &S k \top \\ S k &R k \end bmatrix \nabla x k u k $\mathcal{L}_{k}\mid_{\xi_{k}^{(s)}},\quad Q_{N}=\nabla^{2}_{x_{k}}\mathcal{L}_{N}\mid_{x_{N}^{(s)}},$ | | (7a) |
| --- | --- | --- | --- |
| | q k ∇ x k L k \| k s r k ∇ u k L k \| k s \displaystyle q k \nabla x k $\mathcal{L}_{k}\lVert_{\xi_{k}^{(s)}}\qquad r_{k}=\nabla_{u_{k}}\mathcal{L}_{k} \rVert_{\xi_{k}^{(s)}}$ | | (7b) |
| | q N = ∇ x N L N \| x N (s). \displaystyle q_{N}=\nabla_{x_{N}} $\mathcal{L}_{N}\lVert_{x_{N}^{(s)}}.$ | | (7c) |

**说明**: Table 7

#### Table 8: Table 8

| | min x ∈ C \displaystyle\min_{x\in $\mathcal{C}}$ | f  (x) \displaystyle f(x) | | (8) |
| --- | --- | --- | --- | --- |

**说明**: Table 8

#### Table 9: Table 9

| | min x, z f  (x) + I C  (z) s.t. x = z, \displaystyle\min_{x,z}\quad f(x)+I_{ $\mathcal{C}}(z)\qquad\text{s.t.}\quad x=z,$ | | (9a) |
| --- | --- | --- | --- |
| | I C  (z) = {0 z ∈ C + ∞ otherwise, \displaystyle I_{ $\mathcal{C}}(z)=\begin{cases}0&z\in\mathcal{C}\\ +\infty&\mathrm{otherwise},\end{cases}$ | | (9b) |

**说明**: Table 9

#### Table 10: Table 10

| | $\mathcal{L}_{A}(x,z,\lambda):=f(x)+I_{\mathcal{C}}(z)+\lambda^{\top}(x-z)+\textstyle\frac{\rho}{2}\lVert x-z \rVert_{2}^{2},$ | | (10) |
| --- | --- | --- | --- |

**说明**: Table 10

#### Table 11: Table 11

| | x (t + 1):= \displaystyle x^{(t+1)}:= | arg min x L A x z t t \displaystyle\arg\min x $\mathcal{L}_{A}(x,z^{(t)},\lambda^{(t)}),$ | | (11) |
| --- | --- | --- | --- | --- |
| | z (t + 1):= \displaystyle z^{(t+1)}:= | arg min z L A x t z t \displaystyle\arg\min z $\mathcal{L}_{A}(x^{(t+1)},z,\lambda^{(t)}),$ | | (12) |
| |  (t + 1):= \displaystyle\lambda^{(t+1)}:= |  (t) +   (x (t + 1) - z (t + 1)). \displaystyle\lambda^{(t)}+\rho(x^{(t+1)}-z^{(t+1)}). | | (13) |

**说明**: Table 11

#### Table 12: Table 12

| | $\textstyle\sum_{j=0}^{k-1}\mathbf{\Phi}_{k,j}^{u}w_{j},$ | | (14) |
| --- | --- | --- | --- |

**说明**: Table 12

#### Table 13: Table 13

| | x k + 1 = A k  x k + B k  u k + E k  w k, x_{k+1}=A_{k}x_{k}+B_{k}u_{k}+E_{k}w_{k}, | | (15) |
| --- | --- | --- | --- |

**说明**: Table 13

#### Table 14: Table 14

| | $\textstyle\sum_{j=0}^{k-1}\mathbf{\Phi}_{k,j}^{x}w_{j},~z_{0}=\bar{x}_{0},$ | | (16) |
| --- | --- | --- | --- |

**说明**: Table 14

#### Table 15: Table 15

| | k j x A k k j x B k k j u \displaystyle $\mathbf{\Phi}_{k+1,j}^{x}=A_{k}\mathbf{\Phi}_{k,j}^{x}+B_{k}\mathbf{\Phi}_{k,j}^{u},$ | | (17) |
| --- | --- | --- | --- |

**说明**: Table 15

#### Table 16: Table 16

| | min X U \displaystyle\min \begin subarray c X \ U $\mathbf{\Phi}\end{subarray}}\ \$ | J X U H ~ \displaystyle J X U $\tilde{H}_{0}(\mathbf{\Phi})$ | | (18a) |
| --- | --- | --- | --- | --- |
| | s.t. | x k + 1 = f  (x k, u k), ∀ k ∈ [N ], x 0 = x  ̄ 0, \displaystyle x_{k+1}=f(x_{k},u_{k}),\quad\forall k\in[N],\quad x_{0}=\bar{x}_{0}, | | (18b) |
| | | k j x A k s k j x B k s k j u \displaystyle $\mathbf{\Phi}^{\text{x}}_{k+1,j}=A^{(s)}_{k}\mathbf{\Phi}^{\text{x}}_{k,j}+B^{(s)}_{k}\mathbf{\Phi}^{\text{u}}_{k,j},$ | | (18c) |
| | | ∀ j ∈ [N ], ∀ k ∈ [j + 1, N - 1 ], \displaystyle\quad\quad\forall j\in[N],\forall k\in[j+1,N-1], | | |
| | | j j x E x j \displaystyle $\mathbf{\Phi}^{\text{x}}_{j+1,j}=E(x_{j}),$ | | (18d) |
| | | g x k u k h k ≤ ∀ k ∈ N \displaystyle g x k u k h k $\mathbf{\Phi})\leq 0,\qquad\forall k\in[N],$ | | (18e) |
| | | g f x N h f ≤ \displaystyle g f x N h f $\mathbf{\Phi})\leq 0,$ | | (18f) |

**说明**: Table 16

#### Table 17: Table 17

| | h k ∑ j k - C k s k j x D k s k j u row \displaystyle h k $\mathbf{\Phi})=\textstyle\sum_{j=0}^{k-1}\big\lVert \big(C^{(s)}_{k}\big)\mathbf{\Phi}^{\mathrm{x}}_{k,j}+\big(D^{(s)}_{k}\big)\mathbf{\Phi}^{\mathrm{u}}_{k,j}\big \rVert_{2,\mathrm{row}}$ | | (19a) |
| --- | --- | --- | --- |
| | h f ∑ j N - C N s N j x row. \displaystyle h f $\mathbf{\Phi})=\textstyle\sum_{j=0}^{N-1}\big\lVert \big(C^{(s)}_{N}\big)\mathbf{\Phi}^{\mathrm{x}}_{N,j}\big \rVert_{2,\mathrm{row}}.$ | | (19b) |

**说明**: Table 17

#### Table 18: Table 18

| | H ~ \displaystyle $\tilde{H}_{0}(\mathbf{\Phi})$ | ∑ j N - ∑ k j N - Q ̄ k j x F R ̄ k j u F \displaystyle $\sum_{j=0}^{N-1}\Big(\sum_{k=j}^{N-1}\!\left(\lVert \bar{Q}^{\frac{1}{2}}\mathbf{\Phi}^{\mathrm{x}}_{k,j} \rVert_{\mathcal{F}}^{2}+\lVert \bar{R}^{\frac{1}{2}}\mathbf{\Phi}^{\mathrm{u}}_{k,j} \rVert_{\mathcal{F}}^{2}\right)$ | | (20) |
| --- | --- | --- | --- | --- |
| | | Q ̄ N N j x F \displaystyle \\|\bar Q N $\frac{1}{2}}\mathbf{\Phi}^{\mathrm{x}}_{N,j}\lVert_{\mathcal{F}}^{2}\Big),$ | | |

**说明**: Table 18

#### Table 19: Table 19

| | min X, U \displaystyle\min_{\begin{subarray}{c}X,\,U\end{subarray}}\quad | J  (X, U) \displaystyle J(X,U) | | (21a) |
| --- | --- | --- | --- | --- |
| | s.t. | x k + 1 = f  (x k, u k), ∀ k ∈ [N ], x 0 = x  ̄ 0, \displaystyle x_{k+1}=f(x_{k},u_{k}),\forall k\in[N],\qquad x_{0}=\bar{x}_{0}, | | (21b) |
| | | g x k u k h k ≤ ∀ k ∈ N \displaystyle g x k u k h k $\mathbf{\Phi})\leq 0,\qquad\forall k\in[N],$ | | (21c) |
| | | g f x N h f ≤ . \displaystyle g f x N h f $\mathbf{\Phi})\leq 0.$ | | (21d) |

**说明**: Table 19

#### Table 20: Table 20

| | min x u \displaystyle\min $\mathbf{\Phi}^{\text{x}},\mathbf{\Phi}^{\text{u}}}\$ | ∑ j N - ∑ k j N - Q k j k j F Q N j N j x F \displaystyle $\sum_{j=0}^{N-1}\left(\sum_{k=j}^{N-1}\lVert \mathcal{Q}_{k,j}\mathbf{\Phi}_{k,j} \rVert^{2}_{\mathcal{F}}+\lVert \mathcal{Q}_{N,j}\mathbf{\Phi}^{\text{x}}_{N,j} \rVert^{2}_{\mathcal{F}}\right)$ | | (22a) |
| --- | --- | --- | --- | --- |
| | s.t. | k j x A k s k j x B k s k j u \displaystyle $\mathbf{\Phi}^{\text{x}}_{k+1,j}=A^{(s)}_{k}\mathbf{\Phi}^{x}_{k,j}+B_{k}^{(s)}\mathbf{\Phi}^{\text{u}}_{k,j},$ | | (22b) |
| | | j j x E j ∀ j ∈ N ∀ k ∈ j N - \displaystyle $\mathbf{\Phi}^{\text{x}}_{j+1,j}=E_{j},\ \ \forall j\in[N],\ \ \forall k\in[j+1,N-1],$ | | (22c) |

**说明**: Table 20

#### Table 21: Table 21

| | | Q k j diag k j C k s D k s Q ̄ R ̄ \displaystyle $\mathcal{Q}_{k,j}=\left(\text{diag}\left(\sqrt{\tau_{k,j}}\right)\begin{bmatrix}C_{k}^{(s)}&D_{k}^{(s)}\end{bmatrix},\begin{bmatrix}\bar{Q}^{\frac{1}{2}}&0\\ 0&\bar{R}^{\frac{1}{2}}\end{bmatrix}\right),$ | | (23) |
| --- | --- | --- | --- | --- |
| | | Q N j diag N j C N s Q ̄ N \displaystyle $\mathcal{Q}_{N,j}=\left(\text{diag}\left(\sqrt{\tau_{N,j}}\right)C_{N}^{(s)},\bar{Q}_{N}^{\frac{1}{2}}\right),$ | | |
| | | Q k j x Q k j xu Q k j ux Q k j u ≔ Q k j Q k j. \displaystyle\begin bmatrix $\mathcal{Q}^{\text{x}}_{k,j}&\mathcal{Q}^{\text{xu}}_{k,j}\\ \mathcal{Q}^{\text{ux}}_{k,j}&\mathcal{Q}^{\text{u}}_{k,j}\end{bmatrix}:=\mathcal{Q}^{\top}_{k,j}\mathcal{Q}_{k,j}.$ | | |

**说明**: Table 21

#### Table 22: Table 22

| | z k C k x k D k u k ∀ k ∈ N z N C N x N. \displaystyle z k C k $\delta x_{k}+D_{k}\delta u_{k},\quad\forall k\in[N];\quad z_{N}=C_{N}\delta x_{N}.$ | | (24) |
| --- | --- | --- | --- |

**说明**: Table 22

#### Table 23: Table 23

| | G G x u G ... G N with \displaystyle G $\delta\xi)=G(\delta x,\delta u)=(G_{0},\ldots,G_{N}),\quad\mathrm{with}$ | | (25) |
| --- | --- | --- | --- |
| | G k k ≔ C k x k D k u k G N x N ≔ C N N. \displaystyle G k $\delta\xi_{k}):= C_{k}\delta x_{k}+D_{k}\delta u_{k},\quad G_{N}(\delta x_{N}):= C_{N}\delta_{N}.$ | | |

**说明**: Table 23

#### Table 24: Table 24

| | min x u \displaystyle\min $\delta x,\delta u}$ | J QP x u I C z \displaystyle J $\text{QP}}(\delta x,\delta u)+I_{\mathcal{C}}(z)$ | | (26) |
| --- | --- | --- | --- | --- |
| | s.t. | x k A k x k B k u k b k ∀ k ∈ N \displaystyle $\delta x_{k+1}=A_{k}\delta x_{k}+B_{k}\delta u_{k}+b_{k},\quad\forall k\in[N],$ | | |
| | | G x u z. \displaystyle G $\delta x,\delta u)=z.$ | | |

**说明**: Table 24

#### Table 25: Table 25

| | | L A J QP I C z T G - z \displaystyle $\mathcal{L}_{A}(\delta\xi,\lambda,\rho)=J_{\text{QP}}(\delta\xi)+I_{\mathcal{C}}(z)+\lambda^{T}(G(\delta\xi)-z)$ | | (27) |
| --- | --- | --- | --- | --- |
| | | G - z \displaystyle $\textstyle\frac{\rho}{2}\lVert G(\delta\xi)-z \rVert_{2}^{2}$ | | |
| | | s.t. x k A k x k B k u k b k x x ̄ - x s \displaystyle $\text{s.t.}\delta x_{k+1}=A_{k}\delta x_{k}+B_{k}\delta u_{k}+b_{k},\delta x_{0}=\bar{x}_{0}-x^{(s)}_{0},$ | | |

**说明**: Table 25

#### Table 26: Table 26

| | min x u \displaystyle\min $\delta x,\delta u}\quad$ | ∑ k N - x k u k Q k S k S k R k x k u k q k r k x k u k \displaystyle $\sum_{k=0}^{N-1}\frac{1}{2}\begin{bmatrix}\delta x_{k}\\ \delta u_{k}\end{bmatrix}^{\top}\begin{bmatrix}\hat{Q}_{k}&\hat{S}_{k}^{\top}\\ \hat{S}_{k}&\hat{R}_{k}\end{bmatrix}\begin{bmatrix}\delta x_{k}\\ \delta u_{k}\end{bmatrix}+\begin{bmatrix}\hat{q}_{k}\\ \hat{r}_{k}\end{bmatrix}^{\top}\begin{bmatrix}\delta x_{k}\\ \delta u_{k}\end{bmatrix}$ | | |
| --- | --- | --- | --- | --- |
| | | x N Q N x N q N x N \displaystyle $\frac{1}{2}\delta x_{N}^{\top}\hat{Q}_{N}\delta x_{N}+\hat{q}^{\top}_{N}\delta x_{N}$ | | (28a) |
| | s.t. | x k A k x k B k u k b k ∀ k ∈ N \displaystyle $\delta x_{k+1}=A_{k}\delta x_{k}+B_{k}\delta u_{k}+b_{k},\ \forall k\in[N],$ | | (28b) |
| | | x x ̄ - x s \displaystyle $\delta x_{0}=\bar{x}_{0}-x^{(s)}_{0},$ | | (28c) |

**说明**: Table 26

#### Table 27: Table 27

| | $\hat{Q}_{k}=Q_{k}+\rho C_{k}^{\top}C_{k},&\hat{q}_{k}=q_{k}+C_{k}^{\top}(\lambda_{k}-\rho z_{k}),\\ \hat{R}_{k}=R_{k}+\rho D_{k}^{\top}D_{k},&\hat{r}_{k}=r_{k}+D_{k}^{\top}(\lambda_{k}-\rho z_{k}),\\ \hat{S}_{k}=S_{k}+\rho C_{k}^{\top}D_{k},&\forall k\in[N],\\ \hat{Q}_{N}=Q_{N}+\rho C_{N}^{\top}C_{N},&\hat{q}_{N}=q_{N}+C_{N}^{\top}(\lambda_{N}-\rho z_{N}).\end{array}$ | | (29) |
| --- | --- | --- | --- |

**说明**: Table 27

#### Table 28: Table 28

| | q k q k C k y k - z k r k r k D k y k - z k q N q N C N y N - z N . \begin aligned & $\hat{q}_{k}=q_{k}+\rho C^{\top}_{k}(y_{k}-z_{k}),\quad\hat{r}_{k}=r_{k}+\rho D^{\top}_{k}(y_{k}-z_{k}),\\ &\hat{q}_{N}=q_{N}+\rho C^{\top}_{N}(y_{N}-z_{N}).\end{aligned}$ | | (30) |
| --- | --- | --- | --- |

**说明**: Table 28

#### Table 29: Table 29

| | z t min G x t u t y t f \displaystyle z t \min $\big(G(\delta x^{(t+1)},\delta u^{(t+1)})+y^{(t)},f\big),$ | | (31a) |
| --- | --- | --- | --- |
| | t t G x t u t - z t . \displaystyle\lambda t \lambda t \rho $\big(G(\delta x^{(t+1)},\delta u^{(t+1)})-z^{(t+1)}\big).$ | | (31b) |

**说明**: Table 29

#### Table 30: Table 30

| | s 1: N:= (a 1 ⊗ a 2 ⊗ ⋯ ⊗ a N, a 2 ⊗ ⋯ ⊗ a N, ..., a N), s_{1:N}:=(a_{1}\otimes a_{2}\otimes\cdots\otimes a_{N},\ a_{2}\otimes\cdots\otimes a_{N},\ \ldots\,a_{N}), | | (32) |
| --- | --- | --- | --- |

**说明**: Table 30

#### Table 31: Table 31

| | V i → j x i x j max \displaystyle V i $\rightarrow j}(x_{i},x_{j})=\max_{\eta}$ | x i P ~ i j x i p ~ i j x i - C ~ i j \displaystyle $\frac{1}{2}x_{i}^{\top}\tilde{P}_{i,j}x_{i}+\tilde{p}_{i,j}^{\top}x_{i}-\frac{1}{2}\eta^{\top}\tilde{C}_{i,j}\eta$ | | (33) |
| --- | --- | --- | --- | --- |
| | | - x j - A ~ i j x i - b ~ i j \displaystyle-\eta \top $\big(x_{j}-\tilde{A}_{i,j}x_{i}-\tilde{b}_{i,j}\big),$ | | |

**说明**: Table 31

#### Table 32: Table 32

| | P ~ i, j \displaystyle $\tilde{P}_{i,j}$ | P ~ i k ⊗ P ~ k j ≔ i j P ~ k j A ~ i k P ~ i k \displaystyle $\tilde{P}_{i,k}\otimes\tilde{P}_{k,j}:=\Upsilon_{i,j}\tilde{P}_{k,j}\tilde{A}_{i,k}+\tilde{P}_{i,k},$ | | (34a) |
| --- | --- | --- | --- | --- |
| | p ~ i, j \displaystyle $\tilde{p}_{i,j}$ | p ~ i k ⊗ p ~ k j ≔ i j p ~ k j - P ~ k j b ~ i k p ~ i k \displaystyle $\tilde{p}_{i,k}\otimes\tilde{p}_{k,j}:=\Upsilon_{i,j}\big(\tilde{p}_{k,j}-\tilde{P}_{k,j}\tilde{b}_{i,k}\big)+\tilde{p}_{i,k},$ | | (34b) |
| | A ~ i, j \displaystyle $\tilde{A}_{i,j}$ | A ~ i k ⊗ A ~ k j ≔ i j A ~ i k \displaystyle $\tilde{A}_{i,k}\otimes\tilde{A}_{k,j}:=\Psi_{i,j}\tilde{A}_{i,k},$ | | (34c) |
| | C ~ i, j \displaystyle $\tilde{C}_{i,j}$ | C ~ i k ⊗ C ~ k j ≔ i j C ~ i k A ~ k j C ~ k j \displaystyle $\tilde{C}_{i,k}\otimes\tilde{C}_{k,j}:=\Psi_{i,j}\tilde{C}_{i,k}\tilde{A}_{k,j}^{\top}+\tilde{C}_{k,j},$ | | (34d) |
| | b ~ i, j \displaystyle $\tilde{b}_{i,j}$ | b ~ i k ⊗ b ~ k j ≔ i j b ~ i k - C ~ i k p ~ k j b ~ k j \displaystyle $\tilde{b}_{i,k}\otimes\tilde{b}_{k,j}:=\Psi_{i,j}\big(\tilde{b}_{i,k}-\tilde{C}_{i,k}\tilde{p}_{k,j}\big)+\tilde{b}_{k,j},$ | | (34e) |
| | where | i j A ~ i k I P ~ k j C ~ i k - \displaystyle\quad\Upsilon i j $\tilde{A}_{i,k}^{\top}\big(I+\tilde{P}_{k,j}\tilde{C}_{i,k}\big)^{-1},$ | | (34f) |
| | | i j A ~ k j I C ~ i k P ~ k j - . \displaystyle\quad\Psi i j $\tilde{A}_{k,j}\big(I+\tilde{C}_{i,k}\tilde{P}_{k,j}\big)^{-1}.$ | | (34g) |

**说明**: Table 32

#### Table 33: Table 33

| | P ~ i i Q i - S i R i - S i \displaystyle $\tilde{P}_{i,i+1}=\hat{Q}_{i}-\hat{S}_{i}^{\top}\hat{R}_{i}^{-1}\hat{S}_{i},$ | p ~ i i q i - S i i \displaystyle\quad $\tilde{p}_{i,i+1}=\hat{q}_{i}-\hat{S}_{i}^{\top}\Omega_{i}$ | | (35a) |
| --- | --- | --- | --- | --- |
| | A ~ i, i + 1 = A i - B i  R ^ i - 1  S ^ i, \displaystyle $\tilde{A}_{i,i+1}=A_{i}-B_{i}\hat{R}_{i}^{-1}\hat{S}_{i},$ | C ~ i i B i R i - B i \displaystyle\quad $\tilde{C}_{i,i+1}=B_{i}\hat{R}_{i}^{-1}B_{i}^{\top},$ | | (35b) |
| | b ~ i i b i - B i i \displaystyle $\tilde{b}_{i,i+1}=b_{i}-B_{i}\Omega_{i},$ | i R i - r i ∀ i ∈ N \displaystyle\quad\Omega i $\hat{R}_{i}^{-1}\hat{r}_{i}\quad\forall i\in[N]$ | | (35c) |
| | P ~ N, N + 1 = Q ^ N, \displaystyle $\tilde{P}_{N,N+1}=\hat{Q}_{N},$ | p ~ N, N + 1 = q ^ N, \displaystyle\quad $\tilde{p}_{N,N+1}=\hat{q}_{N},$ | | (35d) |
| | A ~ N, N + 1 = 0, C ~ \displaystyle $\tilde{A}_{N,N+1}=0,\quad\tilde{C}$ | = N, N + 1 0, b ~ N, N + 1 = 0. {}_{N,N+1}=0,\quad $\tilde{b}_{N,N+1}=0.$ | | (35e) |

**说明**: Table 33

#### Table 34: Table 34

| | K i \displaystyle K_{i} | - i S i B i P i A i \displaystyle -\Gamma i $\big(\hat{S}_{i}+B_{i}^{\top}P_{i+1}A_{i}\big),$ | | (36a) |
| --- | --- | --- | --- | --- |
| | k i \displaystyle k_{i} | - i B i p i P i b i r i \displaystyle -\Gamma i $\left(B_{i}^{\top}(p_{i+1}+P_{i+1}b_{i})+\hat{r}_{i}\right),$ | | (36b) |
| | where | i R i B i P i B i - . \displaystyle\quad\Gamma i $\hat{R}_{i}+B_{i}^{\top}P_{i+1}B_{i})^{-1}.$ | | (36c) |

**说明**: Table 34

#### Table 35: Table 35

| | $\rightarrow j}(\delta x_{i},\delta x_{j})=\bar{A}_{i,j}\delta x_{i}+\bar{b}_{i,j}$ | | (37) |
| --- | --- | --- | --- |

**说明**: Table 35

#### Table 36: Table 36

| | $\delta x_{0}+b_{0},\end{array}$ | | (38) |
| --- | --- | --- | --- |

**说明**: Table 36

#### Table 37: Table 37

| | | x i h ̄ → ⊗ h ̄ → ⊗ ⋯ ⊗ h ̄ i - → i \displaystyle $\delta x_{i}=\bar{h}_{0\rightarrow 1}\otimes\bar{h}_{1\rightarrow 2}\otimes\cdots\otimes\bar{h}_{i-1\rightarrow i}$ | | (39) |
| --- | --- | --- | --- | --- |
| | | u i K i x i k i. \displaystyle $\delta u_{i}=K_{i}\delta x_{i}+k_{i}.$ | | |

**说明**: Table 37

#### Table 38: Table 38

| | p ~ i, j \displaystyle $\tilde{p}_{i,j}$ | i j cache p ~ k j - P ~ k j cache b ~ i j \displaystyle \Upsilon i j $\text{cache}}\big(\tilde{p}_{k,j}-\tilde{P}_{k,j}^{\text{cache}}\tilde{b}_{i,j}\big),$ | | (40) |
| --- | --- | --- | --- | --- |
| | b ~ i, j \displaystyle $\tilde{b}_{i,j}$ | i j cache b ~ i k - C ~ i k cache p ~ i j b ~ j k \displaystyle \Psi i j $\text{cache}}\big(\tilde{b}_{i,k}-\tilde{C}_{i,k}^{\text{cache}}\tilde{p}_{i,j}\big)+\tilde{b}_{j,k},$ | | |

**说明**: Table 38

#### Table 39: Table 39

| | $\tilde{p}_{i,i+1}=\hat{q}_{i}-\hat{S}_{i}\Omega_{i}^{\text{cache}},&\tilde{b}_{i,i+1}=b_{i}-B_{i}\Omega_{i}^{\text{cache}},&\forall i\in[N]\\ \tilde{p}_{N,N+1}=\hat{q}_{N},&\tilde{b}_{N,N+1}=0,\end{array}$ | | (41) |
| --- | --- | --- | --- |

**说明**: Table 39

#### Table 40: Table 40

| | | K i K i cache k i - i cache B i p i P i cache b i r i \displaystyle K i K i $\text{cache}},\ \ k_{i}=-\Gamma_{i}^{\text{cache}}\left(B_{i}^{\top}(p_{i+1}+P_{i+1}^{\text{cache}}b_{i})+\hat{r}_{i}\right)$ | | (42) |
| --- | --- | --- | --- | --- |

**说明**: Table 40

#### Table 41: Table 41

| | G N (j) \displaystyle $\mathcal{G}_{N}^{(j)}$ | = Q N, j x, K k (j) = - G k (j)  B k (j), \displaystyle= $\mathcal{Q}^{\text{x}}_{N,j},\quad\quad\mathcal{K}_{k}^{(j)}=-\mathcal{G}_{k}^{(j)}\mathcal{B}_{k}^{(j)},$ | | (43) |
| --- | --- | --- | --- | --- |
| | P k (j) \displaystyle $\mathcal{P}_{k}^{(j)}$ | Q k j x A k s P k j A k s K k j B k j \displaystyle $\mathcal{Q}^{\text{x}}_{k,j}+\big(A^{(s)}_{k}\big)^{\top}\mathcal{P}_{k+1}^{(j)}A^{(s)}_{k}+\big(\mathcal{K}_{k}^{(j)}\big)^{\top}\mathcal{B}_{k}^{(j)},$ | | |
| | G k (j) \displaystyle $\mathcal{G}_{k}^{(j)}$ | Q k j u B k s P k j B k s - \displaystyle $\big(\mathcal{Q}^{\text{u}}_{k,j}+\big(B^{(s)}_{k}\big)^{\top}\mathcal{P}_{k+1}^{(j)}B^{(s)}_{k}\big)^{-1},$ | | |
| | B k (j) \displaystyle $\mathcal{B}_{k}^{(j)}$ | Q k j ux B k s P k j A k s \displaystyle $\mathcal{Q}^{\text{ux}}_{k,j}+\big(B_{k}^{(s)}\big)^{\top}\mathcal{P}_{k+1}^{(j)}A^{(s)}_{k},$ | | |

**说明**: Table 41

#### Table 42: Table 42

| | j j x E j k j u K k j k j x k j x A k s B k s K k j k j x \begin aligned & $\mathbf{\Phi}^{\text{x}}_{j+1,j}=E_{j},\\ &\mathbf{\Phi}^{u}_{k,j}=\mathcal{K}_{k}^{(j)}\mathbf{\Phi}^{\text{x}}_{k,j},&\mathbf{\Phi}^{\text{x}}_{k+1,j}=(A^{(s)}_{k}+B^{(s)}_{k}\mathcal{K}_{k}^{(j)})\mathbf{\Phi}^{\text{x}}_{k,j},\\ \end{aligned}$ | | (44) |
| --- | --- | --- | --- |

**说明**: Table 42

#### Table 43: Table 43

| | V i → l j i j l j max \displaystyle $\mathcal{V}^{(j)}_{i\rightarrow l}(\mathbf{\Phi}_{i,j},\mathbf{\Phi}_{l,j})=\max_{\theta}$ | i j P ~ i l j i j - D ~ i l j \displaystyle $\frac{1}{2}\mathbf{\Phi}_{i,j}^{\top}\tilde{\mathcal{P}}_{i,l}^{(j)}\mathbf{\Phi}_{i,j}-\frac{1}{2}\theta^{\top}\tilde{\mathcal{D}}_{i,l}^{(j)}\theta$ | | (45) |
| --- | --- | --- | --- | --- |
| | | - l j - A ~ i l j i j \displaystyle- $\theta^{\top}\big(\mathbf{\Phi}_{l,j}-\tilde{\mathcal{A}}_{i,l}^{(j)}\mathbf{\Phi}_{i,j}\big),$ | | |

**说明**: Table 43

#### Table 44: Table 44

| | P ~ i, l (j) \displaystyle $\tilde{\mathcal{P}}^{(j)}_{i,l}$ | A ~ i k j I P ~ k l j D ~ i k j - P ~ k l j A ~ i k j P ~ i k j \displaystyle $\big(\tilde{\mathcal{A}}^{(j)}_{i,k}\big)^{\top}\big(I+\tilde{\mathcal{P}}^{(j)}_{k,l}\tilde{\mathcal{D}}^{(j)}_{i,k}\big)^{-1}\tilde{\mathcal{P}}^{(j)}_{k,l}\tilde{\mathcal{A}}^{(j)}_{i,k}+\tilde{\mathcal{P}}^{(j)}_{i,k},$ | | (46) |
| --- | --- | --- | --- | --- |
| | A ~ i, l (j) \displaystyle $\tilde{\mathcal{A}}^{(j)}_{i,l}$ | = A ~ k, l (j)  (I + D ~ i, k (j)  P ~ k, l (j)) - 1  A ~ i, k (j), \displaystyle= $\tilde{\mathcal{A}}^{(j)}_{k,l}\big(I+\tilde{\mathcal{D}}^{(j)}_{i,k}\tilde{P}^{(j)}_{k,l}\big)^{-1}\tilde{A}^{(j)}_{i,k},$ | | |
| | D ~ i, l (j) \displaystyle $\tilde{\mathcal{D}}^{(j)}_{i,l}$ | A ~ k l j I D ~ i k j P ~ k l j - D ~ i k j A ~ k l j D ~ k l j . \displaystyle $\tilde{\mathcal{A}}^{(j)}_{k,l}\big(I+\tilde{\mathcal{D}}^{(j)}_{i,k}\tilde{P}^{(j)}_{k,l}\big)^{-1}\tilde{\mathcal{D}}^{(j)}_{i,k}\big(\tilde{\mathcal{A}}^{(j)}_{k,l}\big)^{\top}+\tilde{\mathcal{D}}^{(j)}_{k,l}.$ | | |

**说明**: Table 44

#### Table 45: Table 45

| | A ~ i, i + 1 (j) \displaystyle $\tilde{\mathcal{A}}_{i,i+1}^{(j)}$ | = A i (s) - B i (s)  (Q i, j u) - 1  Q i, j xu, \displaystyle=A^{(s)}_{i}-B^{(s)}_{i} $\big(\mathcal{Q}_{i,j}^{\text{u}}\big)^{-1}\mathcal{Q}^{\text{xu}}_{i,j},$ | | (47) |
| --- | --- | --- | --- | --- |
| | D ~ i, i + 1 (j) \displaystyle $\tilde{\mathcal{D}}_{i,i+1}^{(j)}$ | B i s Q i j u - B i s \displaystyle B s i $\big(\mathcal{Q}_{i,j}^{\text{u}}\big)^{-1}\big(B^{(s)}_{i}\big)^{\top},$ | | |
| | P ~ i, i + 1 (j) \displaystyle $\tilde{\mathcal{P}}_{i,i+1}^{(j)}$ | = Q i, j x - Q i, j ux  (Q i, j u) - 1  Q i, j xu, \displaystyle= $\mathcal{Q}^{\text{x}}_{i,j}-\mathcal{Q}^{\text{ux}}_{i,j}\big(\mathcal{Q}^{\text{u}}_{i,j}\big)^{-1}\mathcal{Q}^{\text{xu}}_{i,j},$ | | |
| | P ~ N, N + 1 (j) = \displaystyle $\tilde{\mathcal{P}}^{(j)}_{N,N+1}=$ | Q N + 1 x A ~ N, N + 1 (j) = 0, D ~ N, N + 1 (j) = 0. \displaystyle $\mathcal{Q}^{\text{x}}_{N+1}\quad\tilde{\mathcal{A}}^{(j)}_{N,N+1}=0,\quad\tilde{\mathcal{D}}^{(j)}_{N,N+1}=0.$ | | |

**说明**: Table 45

#### Table 46: Table 46

| | A  ̄ i, l (j) = A  ̄ i, k (j)  A  ̄ k, l (j), A  ̄ i, i + 1 (j) = A i (s) + B i (s)  K i, j, \displaystyle\bar{ $\mathcal{A}}_{i,l}^{(j)}=\bar{\mathcal{A}}_{i,k}^{(j)}\bar{\mathcal{A}}_{k,l}^{(j)},\qquad\bar{\mathcal{A}}_{i,i+1}^{(j)}=A_{i}^{(s)}+B_{i}^{(s)}\mathcal{K}_{i,j},$ | | (48) |
| --- | --- | --- | --- |

**说明**: Table 46

#### Table 47: Table 47

| | i j x \displaystyle $\mathbf{\Phi}^{\text{x}}_{i,j}$ | N ̄ → ⊗ ... ⊗ N ̄ i - → i i j u K i j i j x. \displaystyle \bar $\mathcal{N}}_{0\rightarrow 1}\otimes\ldots\otimes\bar{\mathcal{N}}_{i-1\rightarrow i},\qquad\mathbf{\Phi}^{\text{u}}_{i,j}=\mathcal{K}_{i,j}\mathbf{\Phi}^{\text{x}}_{i,j}.$ | | (49) |
| --- | --- | --- | --- | --- |

**说明**: Table 47

#### Table 48: Table 48

| | s 1 \displaystyle s_{1} | = a 1, \displaystyle=a_{1}, | | (50) |
| --- | --- | --- | --- | --- |
| | s 2 \displaystyle s_{2} | = a 1 ⊗ a 2, \displaystyle=a_{1}\otimes a_{2}, | | |
| | s 3 \displaystyle s_{3} | = a 1 ⊗ a 2 ⊗ a 3, \displaystyle=a_{1}\otimes a_{2}\otimes a_{3}, | | |
| | ⋮ \displaystyle\vdots | | | |
| | s n \displaystyle s_{n} | = a 1 ⊗ a 2 ⊗ ... ⊗ a n. \displaystyle=a_{1}\otimes a_{2}\otimes\ldots\otimes a_{n}. | | |

**说明**: Table 48

#### Table 49: Table 49

| | p i ≔ a 1 ⊗ a 2 ⊗ ⋯ ⊗ a i - 1, p_{i}:= a_{1}\otimes a_{2}\otimes\cdots\otimes a_{i-1}, | | (51) |
| --- | --- | --- | --- |

**说明**: Table 49

#### Table 50: Table 50

| | | pref(root) = e \displaystyle $\text{pref(root)}=e$ | | (52) |
| --- | --- | --- | --- | --- |
| | | pref  (L) = pref  (P) \displaystyle $\text{pref}(L)=\text{pref}(P)$ | | |
| | | pref  (R) = pref  (P) ⊗ val  (L) \displaystyle $\text{pref}(R)=\text{pref}(P)\otimes\text{val}(L)$ | | |

**说明**: Table 50

#### Table 51: Table 51

| |  k, j \displaystyle\beta_{k,j} | C k s k j x D k s k j u row \displaystyle $\big\lVert \big(C^{(s)}_{k}\big)\mathbf{\Phi}^{\mathrm{x}}_{k,j}+\big(D^{(s)}_{k}\big)\mathbf{\Phi}^{\mathrm{u}}_{k,j}\big \rVert_{2,\mathrm{row}}^{2}$ | | (53) |
| --- | --- | --- | --- | --- |
| | | ∀ j ∈ [N ] ∀ k ∈ [j, N ] \displaystyle\forall j\in[N]\qquad\forall k\in[j,N] | | |
| |  N, j \displaystyle\beta_{N,j} | C N s N j x row \displaystyle \\| $\left(C_{N}^{(s)}\right)\mathbf{\Phi}^{\mathrm{x}}_{N,j}\lVert^{2}_{2,\text{row}}$ | | |
| | | ∀ j ∈ [N ] \displaystyle\forall j\in[N] | | |

**说明**: Table 51

#### Table 52: Table 52

| | $\tau_{k,j}=\frac{\lambda_{k}}{\sqrt{\beta_{k,j}+\epsilon}}\qquad\forall j\in[N]\quad\forall k\in[j,N]$ | | (54) |
| --- | --- | --- | --- |

**说明**: Table 52

#### Table 53: Table 53

| | x k ≔ p x k p y k k x k \begin bmatrix p x k \\ p y k \\ $\theta_{k}\end{bmatrix}$ | | (55) |
| --- | --- | --- | --- |

**说明**: Table 53

#### Table 54: Table 54

| | $\omega_{k}$ | | (56) |
| --- | --- | --- | --- |

**说明**: Table 54

#### Table 55: Table 55

| | x k f x k u k p x k v cos k t p y k v sin k t k k t x k f x k u k \begin bmatrix p x k v\cos $\theta_{k}\Delta t\\ p_{y,k}+v\sin\theta_{k}\Delta t\\ \theta_{k}+\omega_{k}\Delta t\end{bmatrix}$ | | (57) |
| --- | --- | --- | --- |

**说明**: Table 55

#### Table 56: Table 56

| | $\omega_{\text{min}}\leq w_{k}\leq\omega_{\text{max}}$ | | (58) |
| --- | --- | --- | --- |

**说明**: Table 56

#### Table 57: Table 57

| | r 2 - (p x - c x) 2 - (p y - c y) 2 ≤ 0 r^{2}-(p_{x}-c_{x})^{2}-(p_{y}-c_{y})^{2}\leq 0 | | (59) |
| --- | --- | --- | --- |

**说明**: Table 57

#### Table 58: Table 58

| | x  ̇ \displaystyle $\dot{x}$ | v x v y ̇ - m u u sin m u u cos - g L J u - u \displaystyle \begin bmatrix v x \\ v y \\ $\dot{\phi}\\ -\frac{1}{m}(u_{1}+u_{2})\sin(\phi)\\ \frac{1}{m}(u_{1}+u_{2})\cos(\phi)-g\\ \frac{L}{J}(u_{2}-u_{1})\end{bmatrix}$ | | (60a) |
| --- | --- | --- | --- | --- |

**说明**: Table 58

#### Table 59: Table 59

| | x 0: 2 \displaystyle x_{0:2} | : Center of Mass (CoM) position \displaystyle: $\text{Center of Mass (CoM) position}$ | |
| --- | --- | --- | --- |
| | x 3: 6 \displaystyle x_{3:6} | : Floating base orientation (quaternion) \displaystyle: $\text{Floating base orientation (quaternion)}$ | |
| | x 7: 18 \displaystyle x_{7:18} | : joint angles \displaystyle: $\text{joint angles}$ | |
| | x 19: 21 \displaystyle x_{19:21} | : CoM linear velocity \displaystyle: $\text{CoM linear velocity }$ | |
| | x 22: 24 \displaystyle x_{22:24} | : Floating base angular velocity \displaystyle: $\text{Floating base angular velocity}$ | |
| | x 25: 36 \displaystyle x_{25:36} | : joint angular velocity \displaystyle: $\text{joint angular velocity}$ | |
| | x 37: 48 \displaystyle x_{37:48} | : contact positions (world frame) \displaystyle: $\text{contact positions (world frame)}$ | |
| | x 49: 60 \displaystyle x_{49:60} | : ground reaction forces at each feet. \displaystyle: $\text{ground reaction forces at each feet}.$ | |

**说明**: Table 59

#### Table 60: Table 60

| | x 0: 2 \displaystyle x_{0:2} | : CoM position \displaystyle: $\text{CoM position}$ | |
| --- | --- | --- | --- |
| | x 3: 6 \displaystyle x_{3:6} | : Floating base orientation (quaternion) \displaystyle: $\text{Floating base orientation (quaternion)}$ | |
| | x 7: 25 \displaystyle x_{7:25} | : joint angles \displaystyle: $\text{joint angles}$ | |
| | x 26: 28 \displaystyle x_{26:28} | : CoM linear velocity \displaystyle: $\text{CoM linear velocity }$ | |
| | x 29: 31 \displaystyle x_{29:31} | : Floating base angular velocity \displaystyle: $\text{Floating base angular velocity}$ | |
| | x 32: 50 \displaystyle x_{32:50} | : joint angular velocity \displaystyle: $\text{joint angular velocity}$ | |
| | x 51: 62 \displaystyle x_{51:62} | : contact positions (world frame) \displaystyle: $\text{contact positions (world frame)}$ | |
| | x 63: 74 \displaystyle x_{63:74} | : ground reaction forces at each feet. \displaystyle: $\text{ground reaction forces at each feet}.$ | |

**说明**: Table 60

#### Table 61: Table 61

| | E \displaystyle E | = diag  (e 1, ..., e n), where \displaystyle= $\operatorname{diag}(e_{1},\dots,e_{n}),\quad\text{where}$ | | (61a) |
| --- | --- | --- | --- | --- |
| | e i \displaystyle e_{i} | = {0.025, i ∈ {0, 1, 2 }, 0.5, i ∈ {26, 27, 28 }, 0, otherwise, \displaystyle=\begin{cases}0.025,&i\in\{0,1,2\},\\ 0.5,&i\in\{26,27,28\},\\ 0,& $\text{otherwise},\end{cases}$ | | (61b) |

**说明**: Table 61
## 实验解读

- 评价重点:围绕 adaptive-control、certified-control、closed-loop-control、足式运动、模型预测控制,优先看真实机器人结果、扰动恢复、接触稳定性、控制误差、成功率、实时性和 sim-to-real 差距。
- System0 视角:实验结论不只看最终成功率,还要看闭环过程中是否能处理状态估计误差、执行延迟、碰撞/接触突变和外部扰动。
- 与上层系统的关系:如果方法能在快闭环内稳定工作,它可以作为 System1 技能或 System2 规划的底层执行保障;如果只在离线或慢速设定下成立,则还不能直接作为身体级快闭环模块。
## 批判性思考

### 优点

- 论文把 adaptive-control、certified-control、closed-loop-control、足式运动、模型预测控制 放到机器人身体级闭环里讨论,和只做高层语义推理的工作相比,更接近真实机器人部署时会遇到的状态、动力学和安全约束。
- 若论文同时给出公式、系统结构图、硬件实验或消融结果,这些证据可以帮助判断方法是否真的能支撑毫秒到百毫秒级反应。

### 局限

- 需要重点核查控制频率、推理延迟、传感延迟、失败恢复、硬件磨损、未见场景泛化和安全边界;这些因素决定它能否成为可靠的 System0 模块。
- 对学习型方法,还要区分训练环境中的鲁棒性和真实部署中的鲁棒性,尤其是接触丰富任务、动态运动任务和人形机器人任务。
## 论文定位

- 研究对象:Safe Large-Scale Robust Nonlinear MPC in Milliseconds via Reachability-Constrained System Level Synthesis on the GPU。
- 关键词:adaptive-control、certified-control、closed-loop-control、足式运动、模型预测控制、quadruped、reachability、实时控制、鲁棒控制、safe-control。
- System0 关联:关注机器人身体级快反应链路中“状态/接触/局部感知 -> 约束/优化/策略 -> 动作”的短闭环,不把问题停留在语言规划或任务分解层。

## 速查卡片

> [!summary] Safe Large MPC
> - **论文**: https://www.roboticsproceedings.org/rss22/p103.pdf
> - **arXiv**: http://arxiv.org/abs/2604.07644v1
> - **arXiv HTML**: https://arxiv.org/html/2604.07644v1
