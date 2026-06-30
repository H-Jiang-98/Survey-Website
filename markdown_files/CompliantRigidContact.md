---
title: "From Compliant to Rigid Contact Simulation: a Unified and Efficient Approach"
method_name: "Compliant Rigid Contact"
authors: ["Justin Carpentier"]
year: 2024
venue: "RSS"
tags: ["contact-reasoning", "robust-control", "real-time-control", "adaptive-control", "dexterous-manipulation"]
image_source: "online"
arxiv_html: "https://arxiv.org/html/2405.17020v1"
---
# Compliant Rigid Contact
## 一句话总结

> From Compliant to Rigid Contact Simulation: a Unified and Efficient Approach 主要落在 [[adaptive-control]]、[[compliance-control]]、[[contact-implicit-optimization]]、[[接触推理]] 这一类问题上;从 System0 视角看,它的价值在于把身体状态、接触变化、局部感知或动力学约束接到机器人低层闭环里,让动作生成更稳定、安全、实时。
## 核心贡献

1. 围绕 **From Compliant to Rigid Contact Simulation: a Unified and Efficient Approach** 建立了一个与 adaptive-control、compliance-control、contact-implicit-optimization、接触推理、灵巧操作、实时控制 相关的机器人问题设定,重点不是高层语义规划,而是身体级闭环中的状态变化、接触变化或控制约束。
2. 方法层面的关键价值在于把感知、动力学、约束优化、策略学习或硬件反馈组织成可执行的低层动作生成机制,适合放进 System0 的快反应模块中理解。
3. 对系统集成而言,这类工作补上了 System1 技能选择与电机级控制之间的一层:它负责把上层目标变成动态可行、可恢复、可安全过滤的动作。
4. 若论文包含真实机器人、接触任务或仿真到真实迁移,核心应看它是否报告了闭环稳定性、扰动恢复、硬件延迟、失败模式和跨场景泛化。
## 问题背景

这篇论文可以放在 System0 的“具身快系统”背景下阅读:机器人需要在很短时间内根据本体状态、接触、力/触觉、局部视觉或动力学预测做出动作,而不是等待慢速规划器重新推理整个任务。adaptive-control、compliance-control、contact-implicit-optimization、接触推理、灵巧操作、实时控制、鲁棒控制 相关问题的难点通常不在“知道要做什么”,而在动作执行过程中状态会快速偏离、接触会突然改变、约束会实时收紧,低层系统必须及时修正。

### System0 关联

精读时优先看三件事:第一,论文使用哪些快速反馈量,例如关节状态、接触状态、力觉、触觉、深度或局部视觉;第二,反馈如何进入 MPC、WBC、策略网络、安全过滤器或恢复控制器;第三,输出是否直接影响机器人下一步动作,而不是只服务离线分析或高层任务规划。
## 方法详解

### System0 视角解析

从 System0 角度看,这篇论文的方法部分应当围绕 adaptive-control、compliance-control、contact-implicit-optimization、接触推理、灵巧操作、实时控制、鲁棒控制 来读。核心问题是:系统如何从快速变化的机器人状态中构造控制输入,如何处理接触、碰撞、约束、模型误差或执行延迟,以及如何把这些信息变成下一时刻的动作。

如果方法里有优化问题,重点看目标函数、约束和求解频率;如果方法里有学习策略,重点看观测空间、动作空间、训练扰动和部署时的闭环频率;如果方法里有硬件或传感器,重点看传感链路是否真的进入控制回路。这样读可以把论文和 System2 语言规划、System1 技能编排区分开来。

### 关键公式与机制

#### 公式 1: [[动力学或策略机制]]

$$\begin{pmatrix}M&J_{c}^{\top}\\ J_{c}&0_{m\times m}\end{pmatrix}\begin{pmatrix}\boldsymbol{\dot{v}}\\ \boldsymbol{-\lambda}\end{pmatrix}=\begin{pmatrix}\boldsymbol{\tau}-b(\boldsymbol{q},\boldsymbol{v})\\ -\gamma(\boldsymbol{q},\boldsymbol{v})\end{pmatrix}.$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 2: [[动力学或策略机制]]

$$\begin{pmatrix}M&J_{c}^{\top}\\ J_{c}&-R\end{pmatrix}\begin{pmatrix}\boldsymbol{\dot{v}}\\ \boldsymbol{-\lambda}\end{pmatrix}=\begin{pmatrix}\boldsymbol{\tau}-b(\boldsymbol{q},\boldsymbol{v})\\ -\gamma\end{pmatrix},$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 3: [[动力学或策略机制]]

$$\rho^{\text{new}}=\begin{cases}\tau^{\text{inc}}\rho&\text{if}\lVert \text{r}_{k}^{\text{prim}} \rVert_{\infty}\geq\alpha\lVert \text{r}_{k}^{\text{dual}} \rVert_{\infty}\\ {\rho}/{\tau^{\text{dec}}}&\text{if}\lVert \text{r}_{k}^{\text{dual}} \rVert_{\infty}\geq\alpha\lVert \text{r}_{k}^{\text{prim}} \rVert_{\infty}\\ \rho,&\text{otherwise}\,,\end{cases}$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 4: [[优化目标/约束]]

$$\mathcal{K}_{\mu^{(i)}}=\{\lambda^{(i)}\in\mathbb{R}^{3}|\ \lVert \lambda^{(i)}_{T} \rVert_{2}\leq\mu^{(i)}\lambda_{N}^{(i)}\}$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 5: [[优化目标/约束]]

$$
\mathcal{L}_{\rho,\eta}^{A}(\boldsymbol{f},\boldsymbol{y},\boldsymbol{z})=h_{1}(\boldsymbol{f})+\frac{\eta}{2}\lVert \boldsymbol{f}-\boldsymbol{f}^{-} \rVert^{2}_{2}+h_{2}(\boldsymbol{y})\\ +\frac{\rho}{2}\left\lVert \boldsymbol{f}-\boldsymbol{y}-\frac{\boldsymbol{z}}{\rho}\right \rVert^{2}_{2}-\frac{1}{\rho}\left\lVert \boldsymbol{z}\right \rVert^{2}_{2},
$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 6: [[动力学或策略机制]]

$$=\begin{cases}p+p^{\text{inc}}&\text{if}\lVert \text{r}_{k}^{\text{prim}} \rVert_{\infty}\geq\alpha\lVert \text{r}_{k}^{\text{dual}} \rVert_{\infty},\\ p-p^{\text{dec}}&\text{if}\lVert \text{r}_{k}^{\text{dual}} \rVert_{\infty}\geq\alpha\lVert \text{r}_{k}^{\text{prim}} \rVert_{\infty},\\ p,&\text{otherwise}\,,\end{cases}$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 7: [[动力学或策略机制]]

$$\Gamma(\boldsymbol{\sigma})=\begin{pmatrix}\Gamma^{(1)}(\boldsymbol{\sigma}^{(1)})&\dots&\Gamma^{(n_{c})}(\boldsymbol{\sigma}^{(n_{c})})\end{pmatrix}\in\mathbb{R}^{3n_{c}}$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 8: [[优化目标/约束]]

$$\boldsymbol{y}_{k}=\operatorname*{arg\,min}_{\boldsymbol{y}\in\mathcal{K}_{\mu}}\frac{\rho}{2}\left\lVert \boldsymbol{f}_{k}-\frac{\boldsymbol{z}_{k-1}}{\rho}-\boldsymbol{y}\right \rVert^{2}_{2}\,.$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 9: [[优化目标/约束]]

$$=\operatorname*{arg\,min}_{\boldsymbol{f}\in\mathcal{K}_{\mu}}\frac{1}{2}\boldsymbol{f}^{\top}(G+R)\boldsymbol{f}+\boldsymbol{f}^{\top}\left(\Gamma(\boldsymbol{\sigma})+g\right)\,,$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 10: [[优化目标/约束]]

$$\mathcal{L}(\boldsymbol{\dot{v}},\boldsymbol{\lambda})=\frac{1}{2}\lVert \boldsymbol{\dot{v}}-\boldsymbol{\dot{v}}_{f} \rVert^{2}_{M}-\boldsymbol{\lambda}^{\top}(J_{c}\boldsymbol{\dot{v}}+\gamma)$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。
## 关键图表

### Figure 1: Pipeline / core system figure: Simulation of an ill-conditioned stack of cubes. Our approach robustly solves high-dim

![Figure 1](https://arxiv.org/html/2405.17020v1/extracted/5623191/figures/garde_boum.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Pipeline / core system figure: Simulation of an ill-conditioned stack of cubes. Our approach robustly solves high-dim”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 2: Timings on robotics systems. Thanks to its combination of automatic spectral adaptatio

![Figure 2](https://arxiv.org/html/2405.17020v1/x5.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Timings on robotics systems. Thanks to its combination of automatic spectral adaptatio”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 3: Inverse dynamics is used to slide the end-effector of a UR5 on a wall (top left). Th

![Figure 3](https://arxiv.org/html/2405.17020v1/x12.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Inverse dynamics is used to slide the end-effector of a UR5 on a wall (top left). Th”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### 论文表格

#### Table 1: TABLE I: Characteristics of the contact solvers in robotics.

| Physics engine | Complementarity problem | Contact type | Algorithm |
| --- | --- | --- | --- |
| ODE [48 ], PhysX [36 ], DART [33 ] | LCP | hard | PGS |
| Bullet [13 ] | NCP | hard | PGS |
| MuJoCo [53 ], Drake [52 ] | CCP | soft | non-smooth Newton |
| RaiSim [26 ] | - ∗ | hard | per-contact bisection |
| Dojo [25 ] | NCP | hard | Interior Point |
| Ours | NCP | hard & soft | ADMM |

**说明**: TABLE I: Characteristics of the contact solvers in robotics.

#### Table 2: TABLE II: Problems of the FCLIB dataset [1 ].

| Category | # Problems | n c n c n_{c} | Dofs | friction $\mu$ |
| --- | --- | --- | --- | --- |
| BoxesStack | 255 | [0:200] | [6:300] | 0.7 |
| Chain | 242 | [8:28] | [48:60] | 0.3 |
| Capsules | 249 | [0:200] | [6:300] | 0.7 |

**说明**: TABLE II: Problems of the FCLIB dataset [1 ].

#### Table 3: TABLE III: Comparison of the number of Cholesky updates between Linear and Spectral update strategies

| | Linear | Spectral | | | | | |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Category | $\tau=2 = 2$ | $\tau=4 = 4$ | $\tau=8 = 8$ | $\tau=16 = 16$ | p = 0.01 | p = 0.05 | p = 0.08 |
| BoxesStack | 9.20 ± 2.48 plus-or-minus 9.20 2.48 9.20\pm 2.48 9.20 ± 2.48 | 6.12 ± 1.75 plus-or-minus 6.12 1.75 6.12\pm 1.75 6.12 ± 1.75 | 5.85 ± 2.47 plus-or-minus 5.85 2.47 5.85\pm 2.47 5.85 ± 2.47 | 6.12 ± 2.40 plus-or-minus 6.12 2.40 6.12\pm 2.40 6.12 ± 2.40 | 11.26 ± 4.99 plus-or-minus 11.26 4.99 11.26\pm 4.99 11.26 ± 4.99 | 5.02 ± 2.40 plus-or-minus 5.02 2.40 5.02\pm 2.40 5.02 ± 2.40 | 3.97 ± 1.76 plus-or-minus 3.97 1.76  $\boldsymbol{3.97\pm 1.76}.97.76$ |
| Chain | 5.74 ± 1.53 plus-or-minus 5.74 1.53 5.74\pm 1.53 5.74 ± 1.53 | 8.76 ± 69.93 plus-or-minus 8.76 69.93 8.76\pm 69.93 8.76 ± 69.93 | 33.7 ± 212.77 plus-or-minus 33.7 212.77 33.7\pm 212.77 33.7 ± 212.77 | 25.48 ± 154.88 plus-or-minus 25.48 154.88 25.48\pm 154.88 25.48 ± 154.88 | 7.87 ± 6.55 plus-or-minus 7.87 6.55 7.87\pm 6.55 7.87 ± 6.55 | 2.76 ± 1.65 plus-or-minus 2.76 1.65  $\boldsymbol{2.76\pm 1.65}.76.65$ | 3.61 ± 20.48 plus-or-minus 3.61 20.48 3.61\pm 20.48 3.61 ± 20.48 |
| Capsules | 4.58 ± 1.71 plus-or-minus 4.58 1.71 4.58\pm 1.71 4.58 ± 1.71 | 3.09 ± 2.01 plus-or-minus 3.09 2.01 3.09\pm 2.01 3.09 ± 2.01 | 3.60 ± 2.94 plus-or-minus 3.60 2.94 3.60\pm 2.94 3.60 ± 2.94 | 4.6 ± 2.06 plus-or-minus 4.6 2.06 4.6\pm 2.06 4.6 ± 2.06 | 5.57 ± 4.85 plus-or-minus 5.57 4.85 5.57\pm 4.85 5.57 ± 4.85 | 2.30 ± 1.50 plus-or-minus 2.30 1.50 2.30\pm 1.50 2.30 ± 1.50 | 2.12 ± 1.01 plus-or-minus 2.12 1.01  $\boldsymbol{2.12\pm 1.01}.12.01$ |

**说明**: TABLE III: Comparison of the number of Cholesky updates between Linear and Spectral update strategies

#### Table 4: TABLE IV: Comparison of the mean timings, in ms, between Linear and Spectral update strategies

| | Linear | Spectral | | | | | |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Category | $\tau=2 = 2$ | $\tau=4 = 4$ | $\tau=8 = 8$ | $\tau=16 = 16$ | p = 0.01 | p = 0.05 | p = 0.08 |
| BoxesStack | 2.16 ± 1.80 plus-or-minus 2.16 1.80 2.16\pm 1.80 2.16 ± 1.80 | 1.73 ± 1.55 plus-or-minus 1.73 1.55 1.73\pm 1.55 1.73 ± 1.55 | 1.76 ± 1.80 plus-or-minus 1.76 1.80 1.76\pm 1.80 1.76 ± 1.80 | 1.95 ± 1.87 plus-or-minus 1.95 1.87 1.95\pm 1.87 1.95 ± 1.87 | 2.56 ± 2.03 plus-or-minus 2.56 2.03 2.56\pm 2.03 2.56 ± 2.03 | 1.55 ± 1.60 plus-or-minus 1.55 1.60 1.55\pm 1.60 1.55 ± 1.60 | 1.51 ± 1.76 plus-or-minus 1.51 1.76  $\boldsymbol{1.51\pm 1.76}.51.76$ |
| Chain | 0.527 ± 0.417 plus-or-minus 0.527 0.417 0.527\pm 0.417 0.527 ± 0.417 | 0.637 ± 0.373 plus-or-minus 0.637 0.373 0.637\pm 0.373 0.637 ± 0.373 | 1.61 ± 7.84 plus-or-minus 1.61 7.84 1.61\pm 7.84 1.61 ± 7.84 | 1.34 ± 6.82 plus-or-minus 1.34 6.82 1.34\pm 6.82 1.34 ± 6.82 | 0.445 ± 0.53 plus-or-minus 0.445 0.53 0.445\pm 0.53 0.445 ± 0.53 | 0.321 ± 0.304 plus-or-minus 0.321 0.304  $\boldsymbol{0.321\pm 0.304}.321.304$ | 0.494 ± 0.301 plus-or-minus 0.494 0.301 0.494\pm 0.301 0.494 ± 0.301 |
| Capsules | 2.86 ± 1.90 plus-or-minus 2.86 1.90 2.86\pm 1.90 2.86 ± 1.90 | 2.50 ± 1.92 plus-or-minus 2.50 1.92 2.50\pm 1.92 2.50 ± 1.92 | 1.95 ± 1.35 plus-or-minus 1.95 1.35 1.95\pm 1.35 1.95 ± 1.35 | 1.69 ± 1.20 plus-or-minus 1.69 1.20 1.69\pm 1.20 1.69 ± 1.20 | 2.17 ± 1.38 plus-or-minus 2.17 1.38 2.17\pm 1.38 2.17 ± 1.38 | 1.72 ± 1.09 plus-or-minus 1.72 1.09 1.72\pm 1.09 1.72 ± 1.09 | 1.63 ± 1.10 plus-or-minus 1.63 1.10  $\boldsymbol{1.63\pm 1.10}.63.10$ |

**说明**: TABLE IV: Comparison of the mean timings, in ms, between Linear and Spectral update strategies
## 实验解读

- 评价重点:围绕 adaptive-control、compliance-control、contact-implicit-optimization、接触推理、灵巧操作,优先看真实机器人结果、扰动恢复、接触稳定性、控制误差、成功率、实时性和 sim-to-real 差距。
- System0 视角:实验结论不只看最终成功率,还要看闭环过程中是否能处理状态估计误差、执行延迟、碰撞/接触突变和外部扰动。
- 与上层系统的关系:如果方法能在快闭环内稳定工作,它可以作为 System1 技能或 System2 规划的底层执行保障;如果只在离线或慢速设定下成立,则还不能直接作为身体级快闭环模块。
## 批判性思考

### 优点

- 论文把 adaptive-control、compliance-control、contact-implicit-optimization、接触推理、灵巧操作 放到机器人身体级闭环里讨论,和只做高层语义推理的工作相比,更接近真实机器人部署时会遇到的状态、动力学和安全约束。
- 若论文同时给出公式、系统结构图、硬件实验或消融结果,这些证据可以帮助判断方法是否真的能支撑毫秒到百毫秒级反应。

### 局限

- 需要重点核查控制频率、推理延迟、传感延迟、失败恢复、硬件磨损、未见场景泛化和安全边界;这些因素决定它能否成为可靠的 System0 模块。
- 对学习型方法,还要区分训练环境中的鲁棒性和真实部署中的鲁棒性,尤其是接触丰富任务、动态运动任务和人形机器人任务。
## 论文定位

- 研究对象:From Compliant to Rigid Contact Simulation: a Unified and Efficient Approach。
- 关键词:adaptive-control、compliance-control、contact-implicit-optimization、接触推理、灵巧操作、实时控制、鲁棒控制。
- System0 关联:关注机器人身体级快反应链路中“状态/接触/局部感知 -> 约束/优化/策略 -> 动作”的短闭环,不把问题停留在语言规划或任务分解层。

## 速查卡片

> [!summary] Compliant Rigid Contact
> - **论文**: https://www.roboticsproceedings.org/rss20/p108.pdf
> - **arXiv**: http://arxiv.org/abs/2405.17020v1
> - **arXiv HTML**: https://arxiv.org/html/2405.17020v1
