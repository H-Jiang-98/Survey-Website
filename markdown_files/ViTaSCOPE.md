---
title: "ViTaSCOPE: Visuo-tactile Implicit Representation for In-hand Pose and Extrinsic Contact Estimation"
method_name: "ViTaSCOPE"
authors: ["Jayjun Lee"]
year: 2025
venue: "RSS"
tags: ["contact-reasoning", "robot-manipulation", "contact-rich-manipulation", "robot-generalization", "closed-loop-control", "dexterous-manipulation", "sim-to-real", "collision-avoidance", "tactile-feedback", "state-estimation"]
image_source: "online"
arxiv_html: "https://arxiv.org/html/2506.12239v1"
---
# ViTaSCOPE
## 一句话总结

> ViTaSCOPE: Visuo-tactile Implicit Representation for In-hand Pose and Extrinsic Contact Estimation 主要落在 [[closed-loop-control]]、[[碰撞避免]]、[[contact-estimation]]、[[接触推理]] 这一类问题上;从 System0 视角看,它的价值在于把身体状态、接触变化、局部感知或动力学约束接到机器人低层闭环里,让动作生成更稳定、安全、实时。
## 核心贡献

1. 围绕 **ViTaSCOPE: Visuo-tactile Implicit Representation for In-hand Pose and Extrinsic Contact Estimation** 建立了一个与 closed-loop-control、碰撞避免、contact-estimation、接触推理、接触丰富操作、灵巧操作 相关的机器人问题设定,重点不是高层语义规划,而是身体级闭环中的状态变化、接触变化或控制约束。
2. 方法层面的关键价值在于把感知、动力学、约束优化、策略学习或硬件反馈组织成可执行的低层动作生成机制,适合放进 System0 的快反应模块中理解。
3. 对系统集成而言,这类工作补上了 System1 技能选择与电机级控制之间的一层:它负责把上层目标变成动态可行、可恢复、可安全过滤的动作。
4. 若论文包含真实机器人、接触任务或仿真到真实迁移,核心应看它是否报告了闭环稳定性、扰动恢复、硬件延迟、失败模式和跨场景泛化。
## 问题背景

这篇论文可以放在 System0 的“具身快系统”背景下阅读:机器人需要在很短时间内根据本体状态、接触、力/触觉、局部视觉或动力学预测做出动作,而不是等待慢速规划器重新推理整个任务。closed-loop-control、碰撞避免、contact-estimation、接触推理、接触丰富操作、灵巧操作、in-hand-manipulation 相关问题的难点通常不在“知道要做什么”,而在动作执行过程中状态会快速偏离、接触会突然改变、约束会实时收紧,低层系统必须及时修正。

### System0 关联

精读时优先看三件事:第一,论文使用哪些快速反馈量,例如关节状态、接触状态、力觉、触觉、深度或局部视觉;第二,反馈如何进入 MPC、WBC、策略网络、安全过滤器或恢复控制器;第三,输出是否直接影响机器人下一步动作,而不是只服务离线分析或高层任务规划。
## 方法详解

### System0 视角解析

从 System0 角度看,这篇论文的方法部分应当围绕 closed-loop-control、碰撞避免、contact-estimation、接触推理、接触丰富操作、灵巧操作、in-hand-manipulation、机器人操作 来读。核心问题是:系统如何从快速变化的机器人状态中构造控制输入,如何处理接触、碰撞、约束、模型误差或执行延迟,以及如何把这些信息变成下一时刻的动作。

如果方法里有优化问题,重点看目标函数、约束和求解频率;如果方法里有学习策略,重点看观测空间、动作空间、训练扰动和部署时的闭环频率;如果方法里有硬件或传感器,重点看传感链路是否真的进入控制回路。这样读可以把论文和 System2 语言规划、System1 技能编排区分开来。

### 关键公式与机制

#### 公式 1: [[优化目标/约束]]

$$\nabla\mathcal{O}(\boldsymbol{q}_{j})=\partial\mathcal{O}(\boldsymbol{q}_{j})/\partial\boldsymbol{q}_{j}$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 2: [[优化目标/约束]]

$$\mathcal{L}_{\text{sdf}}=\frac{1}{|\Omega| } $\sum_{j=1}^{$ |\Omega|}| $\mathcal{O}(\boldsymbol{q}_{j})-s_{j}$ | +\lambda $\frac{1}{$ |\Omega_{s}| } $\sum_{j=1}^{$ |\Omega_{s}|}(1-\langle\nabla\mathcal{O}(\boldsymbol{q}_{j}),\boldsymbol{n}_{j}^{*}\rangle)$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 3: [[优化目标/约束]]

$$\ {\boldsymbol{\psi}}{\operatorname*{arg\,min}}\ \frac{1}{| $\boldsymbol{\phi}_{i}$ | } $\sum_{j=1}^{$ | $\boldsymbol{\phi}_{i}$ | } $\left$ | $\mathcal{T}(\boldsymbol{g}_{j}\mid\boldsymbol{\xi},\boldsymbol{\psi})-[u_{j},v_{j}]^{\top}\right$ |$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 4: [[优化目标/约束]]

$$
\mathcal{L}_{\text{train}}=\lambda_{\text{shear}}\mathcal{L}_{\text{shear}}+\lambda_{\text{emb}}\mathcal{L}_{\text{emb}}+\lambda_{\text{hyper}}\mathcal{L}_{\text{hyper}}+\lambda_{\text{contact}}\mathcal{L}_{\text{contact}}
$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 5: [[动力学或策略机制]]

$${}^{w}\mathbf{X}^{\text{o}}_{\text{SE(2)}}={}^{w}\mathbf{X}^{\text{ee}}\ \cdot\ \text{Proj}_{\mathrm{SE}(2)}(\left[{}^{w}\mathbf{X}^{\text{ee}}\right]^{-1}\ \cdot\ \left[{}^{w}\mathbf{X}^{\text{o}}_{\text{SE(3)}}\right])$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 6: [[优化目标/约束]]

$$\ {{}^{w}\mathbf{X}^{\text{o}}}{\operatorname*{arg\,min}}\ \frac{1}{|\Omega_{s}| } $\sum_{\Omega_{s}}\left$ | $\mathcal{O}\left(S\cdot[^{w}\mathbf{X}^{o}]^{-1}\cdot{}^{w}\boldsymbol{q}^{o}_{s};z_{o}\right)\right$ |$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 7: [[优化目标/约束]]

$$\mathcal{L}_{\text{shear}}=\frac{1}{|\Phi_{i}| } $\sum_{j=1}^{$ |\Phi_{i}|}| $\boldsymbol{\phi}(\boldsymbol{g}_{j})-\left[u_{j}^{*},v_{j}^{*}\right]^{T}$ |$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 8: [[优化目标/约束]]

$$\mathcal{L}_{\text{contact}}=\frac{1}{|\Omega_{i,S}| } $\sum_{j=1}^{$ |\Omega_{i,S}| } $\text{BCE}(C(\boldsymbol{q}\oplus\boldsymbol{z}_{\mathcal{O}}$ |\cdot),c^{*}_{j})$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 9: [[动力学或策略机制]]

$$\boldsymbol{g}\in G=\{(s_{x},s_{y})\ |\ s_{x}\in\left[s_{x_{\text{min}}},s_{x_{\text{max}}}\right],s_{x}\in\left[s_{y_{\text{min}}},s_{y_{\text{max}}}\right]\}$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 10: [[优化目标/约束]]

$$c=C(\boldsymbol{q}\oplus\boldsymbol{z}_{\mathcal{O}},\mathcal{T}(\boldsymbol{g}\mid\boldsymbol{\xi},\boldsymbol{\psi})\mid H_{\mathcal{C}}(\boldsymbol{\xi},\boldsymbol{\psi}))$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。
## 关键图表

### Figure 1: Pipeline / core system figure: ViTaSCOPE training and inference. ViTaSCOPE is composed of three modules: Object modul

![Figure 1](https://arxiv.org/html/2506.12239v1/x1.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Pipeline / core system figure: ViTaSCOPE training and inference. ViTaSCOPE is composed of three modules: Object modul”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 2: Simulation Environment for Visuo-tactile Data Generation. A visualization of simulated

![Figure 2](https://arxiv.org/html/2506.12239v1/x3.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Simulation Environment for Visuo-tactile Data Generation. A visualization of simulated”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 3: Extrinsic Contact Estimation. Results with varying Chamfer distances and different con

![Figure 3](https://arxiv.org/html/2506.12239v1/x4.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Extrinsic Contact Estimation. Results with varying Chamfer distances and different con”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### 论文表格

#### Table 1: TABLE I: Results on Geometry Reconstruction, In-hand Pose Estimation, and Extrinsic Contact Prediction in Simulation and

| | | Simulation | Real world | | | | |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Geometries | SDF CD ↓ ↓ \downarrow ↓ [m 2 m 2  $\text{m}^{2} m 2 ]$ | Pose Est. | Ext. Contact Patch | Pose Est. | Ext. Contact Patch | | |
| | | Trans. err [mm] | Rot. err [deg] | CD [m 2 m 2  $\text{m}^{2} m 2 ]$ | Trans. err [mm] | Rot. err [deg] | CD [m 2 m 2  $\text{m}^{2} m 2 ]$ |
| mountain | 0.072 | 3.639 | 2.739 | 0.0015 | 3.110 | 1.286 | 0.0394 |
| rectangle | 0.057 | 2.504 | 4.833 | 0.2724 | 2.162 | 0.210 | 0.2216 |
| pyramid | 0.065 | 3.567 | 0.703 | 0.0676 | 3.203 | 1.487 | 0.0565 |
| hex | 0.041 | 3.415 | 0.872 | 0.0841 | 2.286 | 0.790 | 0.0596 |
| cylinder | 0.039 | 2.924 | 1.660 | 0.1534 | 3.893 | 1.011 | 0.1716 |
| semisphere | 0.040 | 8.571 | 0.371 | 0.0017 | 9.637 | 0.657 | 0.0031 |
| average | 0.052 | 4.103 | 1.863 | 0.0968 | 4.049 | 0.907 | 0.0920 |

**说明**: TABLE I: Results on Geometry Reconstruction, In-hand Pose Estimation, and Extrinsic Contact Prediction in Simulation and Real-world. CD results are based on normalized geometries.

#### Table 2: TABLE II: Pose Estimation Results. Translational and rotational errors are shown for the ICP baseline and ViTaSCOPE.

| | ICP | ViTaSCOPE (Ours) | | | | | | | | | | |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Geometries | Vision | Tactile | Vision+Tactile | Vision | Tactile | Vision+Tactile | | | | | | |
| | Trans. err [mm] | Rot. err [deg] | Trans. | Rot. | Trans. | Rot. | Trans. | Rot. | Trans. | Rot. | Trans. | Rot. |
| mountain | 81.4 | 134.8 | 81.6 | 70.9 | 81.8 | 49.22 | 16.6 | 9.96 | 16.5 | 21.4 | 3.11 | 1.29 |
| rectangle | 17.5 | 82.05 | 21.9 | 102.1 | 4.68 | 69.1 | 1.91 | 0.27 | 26.7 | 10.3 | 2.16 | 0.21 |
| pyramid | 129.2 | 85.4 | 119.6 | 160.9 | 122.7 | 58.4 | 3.35 | 1.51 | 26.4 | 9.60 | 3.20 | 1.49 |
| hex | 5.37 | 127.4 | 28.3 | 69.9 | 2.64 | 82.33 | 3.75 | 3.56 | 32.0 | 6.14 | 2.29 | 0.79 |
| cylinder | 4.77 | 130.7 | 21.8 | 146.7 | 4.43 | 155.9 | 4.86 | 8.98 | 8.54 | 0.03 | 3.89 | 1.01 |
| semisphere | 39.7 | 72.0 | 41.6 | 118.9 | 41.8 | 56.5 | 12.3 | 4.35 | 32.4 | 1.51 | 9.64 | 0.66 |
| average | 46.3 | 105.4 | 52.5 | 111.6 | 43.0 | 78.6 | 7.13 | 4.77 | 23.8 | 8.84 | 4.04 | 0.91 |

**说明**: TABLE II: Pose Estimation Results. Translational and rotational errors are shown for the ICP baseline and ViTaSCOPE.

#### Table 3: TABLE III: Ablation and baseline results for real-world extrinsic contact estimation. We report the performance measured

| Ablation | Real World Contact Patch Chamfer Distance (CD) ↓ ↓ \downarrow ↓ [m 2 m 2  $\text{m}^{2} m 2 ]$ | | | | | | |
| --- | --- | --- | --- | --- | --- | --- | --- |
| | mount. | rectangle | pyramid | hex | cylinder | semisphere | mean |
| vitascope | 0.0394 | 0.2216 | 0.0565 | 0.0596 | 0.1716 | 0.0031 | 0.0920 |
| wo acts | 0.0872 | 0.3274 | 0.0992 | 0.0953 | 0.2033 | 0.0044 | 0.1361 |
| wo obj pose | 0.1650 | 0.2900 | 0.1335 | 0.1711 | 0.1101 | 0.0073 | 0.1462 |
| shear → →  $\rightarrow → rgb$ | 0.0551 | 0.2891 | 0.1175 | 0.0611 | 0.0511 | 0.0034 | 0.0962 |
| V pcd | 0.0397 | 0.2227 | 0.0570 | 0.0654 | 0.2580 | 0.0074 | 0.1084 |
| T pcd | 0.1386 | 0.3720 | 0.0889 | 0.0735 | 0.2145 | 0.0031 | 0.1484 |
| ncf baseline | 1.2068 | 0.2452 | 0.6300 | 1.1434 | 0.5930 | 1.5240 | 0.8904 |

**说明**: TABLE III: Ablation and baseline results for real-world extrinsic contact estimation. We report the performance measured by CD between predicted and GT contact patches.

#### Table 4: TABLE IV: Mean computation time [s] per inference stage.

| Stage | Pose Est. | Trial Code Inf. | Contact Pred. |
| --- | --- | --- | --- |
| Time / Steps | 2.5 / 250 | 0.7 / 60 | 0.08 / 1 |

**说明**: TABLE IV: Mean computation time [s] per inference stage.
## 实验解读

- 评价重点:围绕 closed-loop-control、碰撞避免、contact-estimation、接触推理、接触丰富操作,优先看真实机器人结果、扰动恢复、接触稳定性、控制误差、成功率、实时性和 sim-to-real 差距。
- System0 视角:实验结论不只看最终成功率,还要看闭环过程中是否能处理状态估计误差、执行延迟、碰撞/接触突变和外部扰动。
- 与上层系统的关系:如果方法能在快闭环内稳定工作,它可以作为 System1 技能或 System2 规划的底层执行保障;如果只在离线或慢速设定下成立,则还不能直接作为身体级快闭环模块。
## 批判性思考

### 优点

- 论文把 closed-loop-control、碰撞避免、contact-estimation、接触推理、接触丰富操作 放到机器人身体级闭环里讨论,和只做高层语义推理的工作相比,更接近真实机器人部署时会遇到的状态、动力学和安全约束。
- 若论文同时给出公式、系统结构图、硬件实验或消融结果,这些证据可以帮助判断方法是否真的能支撑毫秒到百毫秒级反应。

### 局限

- 需要重点核查控制频率、推理延迟、传感延迟、失败恢复、硬件磨损、未见场景泛化和安全边界;这些因素决定它能否成为可靠的 System0 模块。
- 对学习型方法,还要区分训练环境中的鲁棒性和真实部署中的鲁棒性,尤其是接触丰富任务、动态运动任务和人形机器人任务。
## 论文定位

- 研究对象:ViTaSCOPE: Visuo-tactile Implicit Representation for In-hand Pose and Extrinsic Contact Estimation。
- 关键词:closed-loop-control、碰撞避免、contact-estimation、接触推理、接触丰富操作、灵巧操作、in-hand-manipulation、机器人操作、scalable-robot-learning、仿真到真实迁移。
- System0 关联:关注机器人身体级快反应链路中“状态/接触/局部感知 -> 约束/优化/策略 -> 动作”的短闭环,不把问题停留在语言规划或任务分解层。

## 速查卡片

> [!summary] ViTaSCOPE
> - **论文**: https://www.roboticsproceedings.org/rss21/p054.pdf
> - **arXiv**: http://arxiv.org/abs/2506.12239v1
> - **arXiv HTML**: https://arxiv.org/html/2506.12239v1
