---
title: "Vysics: Object Reconstruction Under Occlusion by Fusing Vision and Contact-Rich Physics"
method_name: "Vysics"
authors: ["Bibit Bianchini"]
year: 2025
venue: "RSS"
tags: ["contact-reasoning", "robust-control", "contact-rich-manipulation", "robot-generalization", "collision-avoidance", "tactile-feedback", "state-estimation"]
image_source: "online"
arxiv_html: "https://arxiv.org/html/2504.18719v1"
---
# Vysics
## 一句话总结

> Vysics: Object Reconstruction Under Occlusion by Fusing Vision and Contact-Rich Physics 主要落在 [[碰撞避免]]、[[接触推理]]、[[接触丰富操作]]、[[dynamics-modeling]] 这一类问题上;从 System0 视角看,它的价值在于把身体状态、接触变化、局部感知或动力学约束接到机器人低层闭环里,让动作生成更稳定、安全、实时。
## 核心贡献

1. 围绕 **Vysics: Object Reconstruction Under Occlusion by Fusing Vision and Contact-Rich Physics** 建立了一个与 碰撞避免、接触推理、接触丰富操作、dynamics-modeling、proprioception、状态估计 相关的机器人问题设定,重点不是高层语义规划,而是身体级闭环中的状态变化、接触变化或控制约束。
2. 方法层面的关键价值在于把感知、动力学、约束优化、策略学习或硬件反馈组织成可执行的低层动作生成机制,适合放进 System0 的快反应模块中理解。
3. 对系统集成而言,这类工作补上了 System1 技能选择与电机级控制之间的一层:它负责把上层目标变成动态可行、可恢复、可安全过滤的动作。
4. 若论文包含真实机器人、接触任务或仿真到真实迁移,核心应看它是否报告了闭环稳定性、扰动恢复、硬件延迟、失败模式和跨场景泛化。
## 问题背景

这篇论文可以放在 System0 的“具身快系统”背景下阅读:机器人需要在很短时间内根据本体状态、接触、力/触觉、局部视觉或动力学预测做出动作,而不是等待慢速规划器重新推理整个任务。碰撞避免、接触推理、接触丰富操作、dynamics-modeling、proprioception、状态估计、tactile-feedback 相关问题的难点通常不在“知道要做什么”,而在动作执行过程中状态会快速偏离、接触会突然改变、约束会实时收紧,低层系统必须及时修正。

### System0 关联

精读时优先看三件事:第一,论文使用哪些快速反馈量,例如关节状态、接触状态、力觉、触觉、深度或局部视觉;第二,反馈如何进入 MPC、WBC、策略网络、安全过滤器或恢复控制器;第三,输出是否直接影响机器人下一步动作,而不是只服务离线分析或高层任务规划。
## 方法详解

### System0 视角解析

从 System0 角度看,这篇论文的方法部分应当围绕 碰撞避免、接触推理、接触丰富操作、dynamics-modeling、proprioception、状态估计、tactile-feedback、visuomotor-policy 来读。核心问题是:系统如何从快速变化的机器人状态中构造控制输入,如何处理接触、碰撞、约束、模型误差或执行延迟,以及如何把这些信息变成下一时刻的动作。

如果方法里有优化问题,重点看目标函数、约束和求解频率;如果方法里有学习策略,重点看观测空间、动作空间、训练扰动和部署时的闭环频率;如果方法里有硬件或传感器,重点看传感链路是否真的进入控制回路。这样读可以把论文和 System2 语言规划、System1 技能编排区分开来。

### 关键公式与机制

#### 公式 1: [[优化目标/约束]]

$$\mathcal{L}_{\text{bsdf}}=\frac{1}{| $\mathcal{V}$ |}\sum_{\mathbf{s}^{v}\in\mathcal{V}}\left\lVert \nabla\text{DSF}(\mathbf{\hat{n}}^{p^{\prime}})-\mathbf{s}^{v}\right \rVert$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 2: [[优化目标/约束]]

$$
\mathcal{L}_{\text{cvx}}=\max\left(0,\text{SDF}(\mathbf{\mathbf{s}}^{v})-\left(l\ \text{SDF}(\mathbf{s}_{1})+(1-l)\ \ {\text{SDF}}(\mathbf{s}_{2})\right)\right).
$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 3: [[优化目标/约束]]

$$=\arg\max_{\mathbf{s}_{i}\in\mathcal{S}}\,\mathbf{s}_{i}\cdot\mathbf{\hat{n}}=:\mathbf{s}.$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 4: [[优化目标/约束]]

$$=\min\left(0,\,\text{SDF}(\mathbf{s}^{v^{\prime}})-(\mathbf{s}^{v^{\prime}}-\mathbf{s}^{p})\cdot\mathbf{\hat{n}}^{p}\right)^{2}.$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 5: [[优化目标/约束]]

$$
\mathcal{L}_{\text{sp}}=\left(\ {\text{SDF}}(\mathbf{s}^{v})-\text{SDF}(\mathbf{s}^{v})\right)^{2},
$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 6: [[优化目标/约束]]

$$\mathbf{S}^{p_{0}}=\{\mathbf{s}^{v}|\mathbf{s}^{v}\in\mathbf{S}_{\text{sp}}(\mathbf{\hat{n}}^{p},\mathbf{s}^{p}),(\mathbf{\hat{n}}^{p},\mathbf{s}^{p})\in\mathcal{P}\}$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 7: [[动力学或策略机制]]

$$\mathbf{S}^{v_{0}}=\{\mathbf{s}^{v}\in\mathbb{R}^{3}\ |\ | $\text{SDF}(\mathbf{s}^{v})$ |\leq\epsilon\}$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 8: [[优化目标/约束]]

$$\mathcal{P}_{\text{sp}}(\mathbf{\hat{n}}^{p},\mathbf{s}^{p})=\{\mathbf{s}^{v}|\mathbf{s}^{v}=\mathbf{s}^{p}+l\mathbf{\hat{n}}^{p},\ \ l\in[-\epsilon,\infty)\}$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 9: [[动力学或策略机制]]

$$\mathbf{\hat{n}}^{p^{\prime}}=\frac{\mathbf{s}^{v}-\mathbf{s}^{p^{\prime}}}{\left\lVert \mathbf{s}^{v}-\mathbf{s}^{p^{\prime}}\right \rVert}$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 10: [[优化目标/约束]]

$$=\max_{\mathbf{s}_{i}\in\mathcal{S}}\,\mathbf{s}_{i}\cdot\mathbf{\hat{n}},$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。
## 关键图表

### Figure 1: Pipeline / core system figure: a) BundleSDF [66 ]

![Figure 1](https://arxiv.org/html/2504.18719v1/extracted/6390545/Figures/title_bsdf.jpg)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Pipeline / core system figure: a) BundleSDF [66 ]”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 2: The quantitative comparison of the geometric reconstruction accuracy. Each dot is one

![Figure 2](https://arxiv.org/html/2504.18719v1/extracted/6390545/Figures/scatter_chamfer_distance_v_data_allobj_full_geometry_true_units.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“The quantitative comparison of the geometric reconstruction accuracy. Each dot is one”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 3: For quantifying dynamics prediction performance, we compare how far into an open-loo

![Figure 3](https://arxiv.org/html/2504.18719v1/x1.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“For quantifying dynamics prediction performance, we compare how far into an open-loo”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### 论文表格

#### Table 1: TABLE I: Average chamfer distance (unit: cm) of shape completion baselines compared with BundleSDF and our method.

| Method | bakingbox | bottle | egg | milk | oatly | styrofoam | toblerone | all |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 3DSGrasp [46 ] | 3.83 | 2.80 | 3.78 | 3.15 | 2.51 | 2.66 | 2.77 | 3.06 |
| IPoD [70 ] | 3.25 | 1.80 | 2.16 | 2.37 | 2.73 | 1.93 | 1.97 | 2.47 |
| V-PRISM [68 ] | 3.52 | 2.47 | 2.31 | 3.33 | 2.30 | 2.54 | 2.48 | 2.80 |
| OctMAE [30 ] | 3.11 | 2.22 | 1.52 | 2.93 | 2.13 | 2.00 | 2.36 | 2.45 |
| BundleSDF [66 ] | 3.84 | 2.65 | 3.70 | 3.17 | 2.45 | 2.55 | 2.44 | 2.98 |
| Vysics (ours) | 1.83 | 1.36 | 1.05 | 1.53 | 1.25 | 1.45 | 1.02 | 1.45 |

**说明**: TABLE I: Average chamfer distance (unit: cm) of shape completion baselines compared with BundleSDF and our method.

#### Table 2: TABLE II: Hyperparameters for modified BundleSDF [66 ].

| Description | Symbol in [66 ] | Value |
| --- | --- | --- |
| RGB loss weight | w c w c w_{c} | 100 |
| Uncertain free space loss weight | w u w u w_{u} | 50 |
| Uncertain free space SDF target |  \epsilon | 0.1 |
| Empty space loss weight | w e w e w_{e} | 1 |
| Near-surface space loss weight | w s u r f w s u r f w_{surf} | 3,000 |
| Truncation distance of the near-surface space | N/A | 0.01 m |
| SDF target corresponding to the truncation distance |   \lambda | 1 |
| Eikonal loss weight | w e i k w e i k w_{eik} | 0 |
| (Ours) Support point loss weight on (8) | N/A | 2 |
| (Ours) Hyperplane-constrained loss weight on (11) | N/A | 1 |
| (Ours) Convexity loss weight on (13) | N/A | 1 |

**说明**: TABLE II: Hyperparameters for modified BundleSDF [66 ].

#### Table 3: TABLE III: Loss term weights for modified PLL [9, 53 ].

| Description | Symbol in [9 ] | Value |
| --- | --- | --- |
| Prediction loss weight | w pred w pred w_{ $\text{pred}} pred$ | 4 |
| Complementarity loss weight | w comp w comp w_{ $\text{comp}} comp$ | 1 |
| Dissipation loss weight | w diss w diss w_{ $\text{diss}} diss$ | 5000 |
| Penetration loss weight | w pen w pen w_{ $\text{pen}} pen$ | 0.5 |
| (Ours) BundleSDF loss weight on (6) | N/A | 0.04 |

**说明**: TABLE III: Loss term weights for modified PLL [9, 53 ].
## 实验解读

- 评价重点:围绕 碰撞避免、接触推理、接触丰富操作、dynamics-modeling、proprioception,优先看真实机器人结果、扰动恢复、接触稳定性、控制误差、成功率、实时性和 sim-to-real 差距。
- System0 视角:实验结论不只看最终成功率,还要看闭环过程中是否能处理状态估计误差、执行延迟、碰撞/接触突变和外部扰动。
- 与上层系统的关系:如果方法能在快闭环内稳定工作,它可以作为 System1 技能或 System2 规划的底层执行保障;如果只在离线或慢速设定下成立,则还不能直接作为身体级快闭环模块。
## 批判性思考

### 优点

- 论文把 碰撞避免、接触推理、接触丰富操作、dynamics-modeling、proprioception 放到机器人身体级闭环里讨论,和只做高层语义推理的工作相比,更接近真实机器人部署时会遇到的状态、动力学和安全约束。
- 若论文同时给出公式、系统结构图、硬件实验或消融结果,这些证据可以帮助判断方法是否真的能支撑毫秒到百毫秒级反应。

### 局限

- 需要重点核查控制频率、推理延迟、传感延迟、失败恢复、硬件磨损、未见场景泛化和安全边界;这些因素决定它能否成为可靠的 System0 模块。
- 对学习型方法,还要区分训练环境中的鲁棒性和真实部署中的鲁棒性,尤其是接触丰富任务、动态运动任务和人形机器人任务。
## 论文定位

- 研究对象:Vysics: Object Reconstruction Under Occlusion by Fusing Vision and Contact-Rich Physics。
- 关键词:碰撞避免、接触推理、接触丰富操作、dynamics-modeling、proprioception、状态估计、tactile-feedback、visuomotor-policy。
- System0 关联:关注机器人身体级快反应链路中“状态/接触/局部感知 -> 约束/优化/策略 -> 动作”的短闭环,不把问题停留在语言规划或任务分解层。

## 速查卡片

> [!summary] Vysics
> - **论文**: https://www.roboticsproceedings.org/rss21/p034.pdf
> - **arXiv**: http://arxiv.org/abs/2504.18719v1
> - **arXiv HTML**: https://arxiv.org/html/2504.18719v1
