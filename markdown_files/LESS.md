---
title: "More with LESS – Local Scene Representations for Tactile Imaging"
method_name: "LESS"
authors: ["Zohar Rimon"]
year: 2026
venue: "RSS"
tags: ["contact-reasoning", "robot-manipulation", "robust-control", "robot-generalization", "tactile-feedback", "whole-body-control", "state-estimation"]
image_source: "online"
arxiv_html: "https://arxiv.org/html/2606.14344v1"
---
# LESS
## 一句话总结

> More with LESS – Local Scene Representations for Tactile Imaging 主要落在 [[接触推理]]、[[deformable-object]]、[[motion-tracking]]、[[robot-generalization]] 这一类问题上;从 System0 视角看,它的价值在于把身体状态、接触变化、局部感知或动力学约束接到机器人低层闭环里,让动作生成更稳定、安全、实时。
## 核心贡献

1. 围绕 **More with LESS – Local Scene Representations for Tactile Imaging** 建立了一个与 接触推理、deformable-object、motion-tracking、robot-generalization、机器人操作、鲁棒控制 相关的机器人问题设定,重点不是高层语义规划,而是身体级闭环中的状态变化、接触变化或控制约束。
2. 方法层面的关键价值在于把感知、动力学、约束优化、策略学习或硬件反馈组织成可执行的低层动作生成机制,适合放进 System0 的快反应模块中理解。
3. 对系统集成而言,这类工作补上了 System1 技能选择与电机级控制之间的一层:它负责把上层目标变成动态可行、可恢复、可安全过滤的动作。
4. 若论文包含真实机器人、接触任务或仿真到真实迁移,核心应看它是否报告了闭环稳定性、扰动恢复、硬件延迟、失败模式和跨场景泛化。
## 问题背景

这篇论文可以放在 System0 的“具身快系统”背景下阅读:机器人需要在很短时间内根据本体状态、接触、力/触觉、局部视觉或动力学预测做出动作,而不是等待慢速规划器重新推理整个任务。接触推理、deformable-object、motion-tracking、robot-generalization、机器人操作、鲁棒控制、状态估计 相关问题的难点通常不在“知道要做什么”,而在动作执行过程中状态会快速偏离、接触会突然改变、约束会实时收紧,低层系统必须及时修正。

### System0 关联

精读时优先看三件事:第一,论文使用哪些快速反馈量,例如关节状态、接触状态、力觉、触觉、深度或局部视觉;第二,反馈如何进入 MPC、WBC、策略网络、安全过滤器或恢复控制器;第三,输出是否直接影响机器人下一步动作,而不是只服务离线分析或高层任务规划。
## 方法详解

### System0 视角解析

从 System0 角度看,这篇论文的方法部分应当围绕 接触推理、deformable-object、motion-tracking、robot-generalization、机器人操作、鲁棒控制、状态估计、tactile-feedback 来读。核心问题是:系统如何从快速变化的机器人状态中构造控制输入,如何处理接触、碰撞、约束、模型误差或执行延迟,以及如何把这些信息变成下一时刻的动作。

如果方法里有优化问题,重点看目标函数、约束和求解频率;如果方法里有学习策略,重点看观测空间、动作空间、训练扰动和部署时的闭环频率;如果方法里有硬件或传感器,重点看传感链路是否真的进入控制回路。这样读可以把论文和 System2 语言规划、System1 技能编排区分开来。

### 关键公式与机制

#### 公式 1: [[动力学或策略机制]]

$$\bar{z}_{t}=\left\{\bar{z}^{\bar{x}_{1}}_{t},\dots,\bar{z}^{\bar{x}_{N}}_{t}\right\}$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 2: [[优化目标/约束]]

$$\mathcal{X}_{i}=\left\{x:D(x,\bar{x}_{i})<\beta\right\}$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 3: [[优化目标/约束]]

$$
\mathcal{L}_{rec}=\frac{1}{2NKK^{\prime}}\sum_{n=1}^{N}\sum_{k=1}^{K}\sum_{k^{\prime}=1}^{K^{\prime}}\left\lVert FD\left(z_{t_{k}^{n}}^{\bar{x}_{n}},x_{t^{\prime n}_{k^{\prime}}}-\bar{x}_{n}\right)-f_{t^{\prime n}_{k^{\prime}}}\right \rVert^{2},
$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 4: [[动力学或策略机制]]

$$\mathbf{x}_{A}=\mathbf{T}_{A\leftarrow B}\,\mathbf{x}_{B}$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 5: [[动力学或策略机制]]

$$\mathbf{T}^{-1}_{C\leftarrow G.tag}=\mathbf{T}_{G.tag\leftarrow C}$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 6: [[动力学或策略机制]]

$$\mathbf{T}_{{G.tag}\leftarrow{S}}=\mathbf{T}_{{G.tag}\leftarrow{C}}\cdot\mathbf{T}_{{C}\leftarrow{S.tag}}\cdot\mathbf{T}_{{S.tag}\leftarrow S}$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 7: [[动力学或策略机制]]

$$\hat{\sigma}_{positional}=\sqrt{\frac{1}{3N}\sum_{i=1}^{N}(e^{2}_{i,x}+e^{2}_{i,y}+e^{2}_{i,z})}$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 8: [[优化目标/约束]]

$$=\lambda\cdot\mathcal{L}_{\text{Dice}}+(1-\lambda)\cdot\eta\cdot\mathcal{L}_{\text{Focal}},$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 9: [[动力学或策略机制]]

$$=1-\frac{1}{C}\sum_{c=1}^{C}\frac{2\sum_{i,j,k}p_{ijk,c}y_{ijk,c}+\epsilon}{\sum_{i,j,k}p_{ijk,c}+\sum_{i,j,k}y_{ijk,c}+\epsilon}$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 10: [[动力学或策略机制]]

$$=-\frac{1}{HWD}\sum_{i,j,k}\sum_{c=1}^{C}\alpha_{c}y_{ijk,c}(1-p_{ijk,c})^{\gamma}\log(p_{ijk,c}),$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。
## 关键图表

### Figure 1: Pipeline / core system figure: LESS representation learning: A set of localized representations for processing a tac

![Figure 1](https://arxiv.org/html/2606.14344v1/x1.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Pipeline / core system figure: LESS representation learning: A set of localized representations for processing a tac”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 2: The monitor presented to the operator at different steps while using our hand-held sy

![Figure 2](https://arxiv.org/html/2606.14344v1/x6.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“The monitor presented to the operator at different steps while using our hand-held sy”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 3: Overview of our pre-processing pipeline: (a) A subset of slices from the original MR

![Figure 3](https://arxiv.org/html/2606.14344v1/images/3D/Pipeline.jpg)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Overview of our pre-processing pipeline: (a) A subset of slices from the original MR”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### 论文表格

#### Table 1: TABLE I: Comparison of global representation training on 2D and 3D labels. The first three rows are for 2D slice metrics

| | | 2D Training | 3D Training | Average Pred. |
| --- | --- | --- | --- | --- |
| 2D | Diameter \Delta $\text{Diameter}} [mm] ↓ \downarrow$ | 2.7 ± 0.1 2.7\pm 0.1 | 1.8 ± 0.0 1.8\pm 0.0 | 2.02 2.02 |
| $\text{CoM}} [mm] ↓ \downarrow$ | 2.1 ± 0.1 2.1\pm 0.1 | 1.7 ± 0.1 1.7\pm 0.1 | 12.1 12.1 | |
| F1 Score [%] ↑ \uparrow | 79.9 ± 0.8 79.9\pm 0.8 | 85.2 ± 0.8 85.2\pm 0.8 | – | |
| 3D | Diameter \Delta $\text{Diameter}} [mm] ↓ \downarrow$ | – | 0.8 ± 0.0 0.8\pm 0.0 | 1.44 1.44 |
| $\text{CoM}} [mm] ↓ \downarrow$ | – | 1.7 ± 0.1 1.7\pm 0.1 | 12.09 12.09 | |
| F1 Score [%] ↑ \uparrow | – | 83.5 ± 0.7 83.5\pm 0.7 | – | |

**说明**: TABLE I: Comparison of global representation training on 2D and 3D labels. The first three rows are for 2D slice metrics and the rest are 3D metrics.

#### Table 2: TABLE II: Results of GLOBAL and LESS on in-distribution single inclusion and out-of-distribution multiple inclusions an

| | | GLOBAL | LESS (ours) |
| --- | --- | --- | --- |
| Single | Diameter \Delta $\text{Diameter}} [mm] ↓ \downarrow$ | 1.8 ± 0.0 1.8\pm 0.0 | 1.5 ± 0.1 1.5\pm 0.1 |
| $\text{CoM}} [mm] ↓ \downarrow$ | 1.7 ± 0.1 1.7\pm 0.1 | 1.7 ± 0.1 1.7\pm 0.1 | |
| F1 Score [%] ↑ \uparrow | 85.2 ± 0.8 85.2\pm 0.8 | 83.5 ± 0.6 83.5\pm 0.6 | |
| Multiple | $\text{Area}} [mm 2 ] ↓ \downarrow$ | 78.4 ± 5.1 78.4\pm 5.1 | 46.1 ± 2.9 46.1\pm 2.9 |
| $\text{CoM}} [mm] ↓ \downarrow$ | 7.4 ± 0.4 7.4\pm 0.4 | 4.0 ± 0.0 4.0\pm 0.0 | |
| F1 Score [%] ↑ \uparrow | 44.6 ± 1.3 44.6\pm 1.3 | 71.1 ± 0.0 71.1\pm 0.0 | |
| Large | Diameter \Delta $\text{Diameter}} [mm] ↓ \downarrow$ | 9.0 ± 1.8 9.0\pm 1.8 | 2.7 ± 1.1 2.7\pm 1.1 |
| $\text{CoM}} [mm] ↓ \downarrow$ | 26.8 ± 2.8 26.8\pm 2.8 | 2.3 ± 0.2 2.3\pm 0.2 | |
| F1 Score [%] ↑ \uparrow | 1.1 ± 0.5 1.1\pm 0.5 | 71.3 ± 3.6 71.3\pm 3.6 | |

**说明**: TABLE II: Results of GLOBAL and LESS on in-distribution single inclusion and out-of-distribution multiple inclusions and large phantom. In the multiple set, we report the total area error instead of the diameter error. Due to their different inclusion composition, the metrics of different insert sets are not comparable. We clarify that in the large set, the diameter and CoM errors are expected to be larger.

#### Table 3: TABLE III: Training LESS on data-primitive. We trained on only data-poke (Poke), only data-primitive (PRI) and the

| Method | Diameter \Delta $\text{Diameter}} [mm] ↓ \downarrow$ | $\text{CoM}} [mm] ↓ \downarrow$ | F1 Score [%] ↑ \uparrow |
| --- | --- | --- | --- |
| Poke | 6.3 ± 0.6 6.3\pm 0.6 | 6.0 ± 0.5 6.0\pm 0.5 | 45.8 ± 0.6 45.8\pm 0.6 |
| Pri | 2.8 ± 0.3 2.8\pm 0.3 | 2.3 ± 0.1 2.3\pm 0.1 | 74.7 ± 1.2 74.7\pm 1.2 |
| Pri + Poke | 2.0 ± 0.2 2.0\pm 0.2 | 1.6 ± 0.1 1.6\pm 0.1 | 80.9 ± 0.8 80.9\pm 0.8 |

**说明**: TABLE III: Training LESS on data-primitive. We trained on only data-poke (Poke), only data-primitive (PRI) and the union of data-poke, and data-primitive (PRI + Poke). All methods were tested on data-primitive only.

#### Table 4: TABLE IV: Testing LESS on data-handheld.

| Method | Diameter \Delta $\text{Diameter}} [mm] ↓ \downarrow$ | $\text{CoM}} [mm] ↓ \downarrow$ | F1 Score [%] ↑ \uparrow |
| --- | --- | --- | --- |
| w/ Pri, | 2.8 ± 0.8 2.8\pm 0.8 | 8.4 ± 0.7 8.4\pm 0.7 | 41.3 ± 2.9 41.3\pm 2.9 |
| w/o Pri, | 5.2 ± 0.5 5.2\pm 0.5 | 9.5 ± 0.8 9.5\pm 0.8 | 34.7 ± 4.2 34.7\pm 4.2 |

**说明**: TABLE IV: Testing LESS on data-handheld.

#### Table 5: TABLE V: Per-Pixel Class Distribution

| Class | 2D (%) | 3D (%) |
| --- | --- | --- |
| Background | 47.40 ± 2.04 47.40\pm 2.04 | 50.97 ± 2.17 50.97\pm 2.17 |
| Phantom | 49.78 ± 2.07 49.78\pm 2.07 | 47.71 ± 2.17 47.71\pm 2.17 |
| Pillar | — | 0.33 ± 0.04 0.33\pm 0.04 |
| Inclusion | 2.82 ± 1.01 2.82\pm 1.01 | 0.99 ± 0.48 0.99\pm 0.48 |

**说明**: TABLE V: Per-Pixel Class Distribution
## 实验解读

- 评价重点:围绕 接触推理、deformable-object、motion-tracking、robot-generalization、机器人操作,优先看真实机器人结果、扰动恢复、接触稳定性、控制误差、成功率、实时性和 sim-to-real 差距。
- System0 视角:实验结论不只看最终成功率,还要看闭环过程中是否能处理状态估计误差、执行延迟、碰撞/接触突变和外部扰动。
- 与上层系统的关系:如果方法能在快闭环内稳定工作,它可以作为 System1 技能或 System2 规划的底层执行保障;如果只在离线或慢速设定下成立,则还不能直接作为身体级快闭环模块。
## 批判性思考

### 优点

- 论文把 接触推理、deformable-object、motion-tracking、robot-generalization、机器人操作 放到机器人身体级闭环里讨论,和只做高层语义推理的工作相比,更接近真实机器人部署时会遇到的状态、动力学和安全约束。
- 若论文同时给出公式、系统结构图、硬件实验或消融结果,这些证据可以帮助判断方法是否真的能支撑毫秒到百毫秒级反应。

### 局限

- 需要重点核查控制频率、推理延迟、传感延迟、失败恢复、硬件磨损、未见场景泛化和安全边界;这些因素决定它能否成为可靠的 System0 模块。
- 对学习型方法,还要区分训练环境中的鲁棒性和真实部署中的鲁棒性,尤其是接触丰富任务、动态运动任务和人形机器人任务。
## 论文定位

- 研究对象:More with LESS – Local Scene Representations for Tactile Imaging。
- 关键词:接触推理、deformable-object、motion-tracking、robot-generalization、机器人操作、鲁棒控制、状态估计、tactile-feedback。
- System0 关联:关注机器人身体级快反应链路中“状态/接触/局部感知 -> 约束/优化/策略 -> 动作”的短闭环,不把问题停留在语言规划或任务分解层。

## 速查卡片

> [!summary] LESS
> - **论文**: https://www.roboticsproceedings.org/rss22/p167.pdf
> - **arXiv**: http://arxiv.org/abs/2606.14344v1
> - **arXiv HTML**: https://arxiv.org/html/2606.14344v1
