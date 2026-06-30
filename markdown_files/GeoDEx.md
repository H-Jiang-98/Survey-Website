---
title: "GeoDEx: A Unified Geometric Framework for Tactile Dexterous and Extrinsic Manipulation under Force Uncertainty"
method_name: "GeoDEx"
authors: ["Sirui Chen"]
year: 2025
venue: "RSS"
tags: ["contact-reasoning", "robot-manipulation", "robust-control", "dexterous-manipulation", "agile-locomotion", "tactile-feedback"]
image_source: "online"
arxiv_html: "https://arxiv.org/html/2505.00647v1"
---
# GeoDEx
## 一句话总结

> GeoDEx: A Unified Geometric Framework for Tactile Dexterous and Extrinsic Manipulation under Force Uncertainty 主要落在 [[agile-locomotion]]、[[接触推理]]、[[灵巧操作]]、[[力控制]] 这一类问题上;从 System0 视角看,它的价值在于把身体状态、接触变化、局部感知或动力学约束接到机器人低层闭环里,让动作生成更稳定、安全、实时。
## 核心贡献

1. 围绕 **GeoDEx: A Unified Geometric Framework for Tactile Dexterous and Extrinsic Manipulation under Force Uncertainty** 建立了一个与 agile-locomotion、接触推理、灵巧操作、力控制、grasping、机器人操作 相关的机器人问题设定,重点不是高层语义规划,而是身体级闭环中的状态变化、接触变化或控制约束。
2. 方法层面的关键价值在于把感知、动力学、约束优化、策略学习或硬件反馈组织成可执行的低层动作生成机制,适合放进 System0 的快反应模块中理解。
3. 对系统集成而言,这类工作补上了 System1 技能选择与电机级控制之间的一层:它负责把上层目标变成动态可行、可恢复、可安全过滤的动作。
4. 若论文包含真实机器人、接触任务或仿真到真实迁移,核心应看它是否报告了闭环稳定性、扰动恢复、硬件延迟、失败模式和跨场景泛化。
## 问题背景

这篇论文可以放在 System0 的“具身快系统”背景下阅读:机器人需要在很短时间内根据本体状态、接触、力/触觉、局部视觉或动力学预测做出动作,而不是等待慢速规划器重新推理整个任务。agile-locomotion、接触推理、灵巧操作、力控制、grasping、机器人操作、鲁棒控制 相关问题的难点通常不在“知道要做什么”,而在动作执行过程中状态会快速偏离、接触会突然改变、约束会实时收紧,低层系统必须及时修正。

### System0 关联

精读时优先看三件事:第一,论文使用哪些快速反馈量,例如关节状态、接触状态、力觉、触觉、深度或局部视觉;第二,反馈如何进入 MPC、WBC、策略网络、安全过滤器或恢复控制器;第三,输出是否直接影响机器人下一步动作,而不是只服务离线分析或高层任务规划。
## 方法详解

### System0 视角解析

从 System0 角度看,这篇论文的方法部分应当围绕 agile-locomotion、接触推理、灵巧操作、力控制、grasping、机器人操作、鲁棒控制、tactile-feedback 来读。核心问题是:系统如何从快速变化的机器人状态中构造控制输入,如何处理接触、碰撞、约束、模型误差或执行延迟,以及如何把这些信息变成下一时刻的动作。

如果方法里有优化问题,重点看目标函数、约束和求解频率;如果方法里有学习策略,重点看观测空间、动作空间、训练扰动和部署时的闭环频率;如果方法里有硬件或传感器,重点看传感链路是否真的进入控制回路。这样读可以把论文和 System2 语言规划、System1 技能编排区分开来。

### 关键公式与机制

#### 公式 1: [[动力学或策略机制]]

$$\boldsymbol{f}_{fe}=\sum_{i}^{d_{fe}}w_{i}\boldsymbol{b}_{i}+\boldsymbol{f}_{0}$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 2: [[动力学或策略机制]]

$$\begin{split}\boldsymbol{x}_{\text{est}}=\boldsymbol{P}_{fe}(\boldsymbol{f}_{m}-\boldsymbol{f}_{0})\\ \boldsymbol{f}_{\text{est}}=\boldsymbol{P}_{fe}^{T}\boldsymbol{x}_{\text{est}}+\boldsymbol{f}_{0}\end{split}$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 3: [[动力学或策略机制]]

$$\boldsymbol{f}_{\text{est}}=\boldsymbol{P}_{fe}\sum_{i}^{n_{e}}w_{i}\boldsymbol{f}^{i}_{\text{sub}}$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 4: [[动力学或策略机制]]

$$\begin{split}\boldsymbol{m}^{T}\boldsymbol{D}\boldsymbol{m}\leq 1\\ \boldsymbol{D}=\text{Diag}(1/\sigma^{2}_{0},...,1/\sigma^{2}_{n_{i}+n_{e}})\end{split}$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 5: [[动力学或策略机制]]

$$\begin{split}(\boldsymbol{x}_{fe}-\boldsymbol{c})^{T}\boldsymbol{M}(\boldsymbol{x}_{fe}-\boldsymbol{c})\\ \boldsymbol{M}=(\boldsymbol{E}^{\dagger})^{T}\boldsymbol{D}\boldsymbol{E}^{\dagger}\\ \end{split}$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 6: [[动力学或策略机制]]

$$\begin{split}\boldsymbol{x}_{fe}=\boldsymbol{P}_{fe}\boldsymbol{N}_{M-cone}\boldsymbol{m}\\ \boldsymbol{x}_{fe}=\hat{\boldsymbol{E}}\boldsymbol{m}\\ \end{split}$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 7: [[优化目标/约束]]

$$\begin{split}\min_{\boldsymbol{w}}|| $\boldsymbol{f}_{\text{est}}$ ||+|| $\boldsymbol{B}_{\text{sub}}\boldsymbol{f}_{\text{est}}$ ||\\ s.t.\boldsymbol{f}_{\text{est}}=\boldsymbol{P}_{fe}\sum_{i}^{n_{e}}w_{i}\boldsymbol{f}^{i}_{\text{sub}}\end{split}$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 8: [[优化目标/约束]]

$$\mathcal{M}=\{\boldsymbol{f}:\exists\boldsymbol{m},\boldsymbol{f}=\sum_{j}^{n_{i}+n_{e}}m_{j}\textbf{Extend}(\boldsymbol{n}_{j})\}$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 9: [[动力学或策略机制]]

$$\boldsymbol{B}_{fe}=\{\boldsymbol{b}_{1},...,\boldsymbol{b}_{d_{fe}}\},\boldsymbol{b}_{i}\in\mathbb{R}^{(n_{i}+n_{e})\times 3}$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 10: [[动力学或策略机制]]

$$\boldsymbol{f}=\sum^{n_{i}+n_{e}}_{i}m_{i}\textbf{Extend}(\boldsymbol{n}_{i})=\boldsymbol{N}_{M-cone}\boldsymbol{m}$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。
## 关键图表

### Figure 1: System diagram of our proposed method

![Figure 1](https://arxiv.org/html/2505.00647v1/x1.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“System diagram of our proposed method”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 2: Block diagram of the control system

![Figure 2](https://arxiv.org/html/2505.00647v1/x2.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Block diagram of the control system”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 3: Successfully grasping different objects using the proposed method

![Figure 3](https://arxiv.org/html/2505.00647v1/x5.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Successfully grasping different objects using the proposed method”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### 论文表格

#### Table 1: TABLE I: Nomenclature of the Proposed Framework

| Name | Notation |
| --- | --- |
| Number of extrinsic contacts | n e n e n_{ $\text{e}} e$ |
| Number of intrinsic contacts | n i n i n_{ $\text{i}} i$ |
| i t h i t h i^{th} contact normal | n i ∈ R 3 n i R 3  $\boldsymbol{n}_{i}\in\mathbb{R}^{3} ∈ 3$ |
| All contact forces | $\boldsymbol{f}\in\mathbb{R}^{(n_{\text{i}}+n_{\text{e}})\times 3} ∈ (i + e) × 3$ |
| Gravity wrench | g ∈ R 6 g R 6  $\boldsymbol{g}\in\mathbb{R}^{6} ∈ 6$ |
| Extrinsic contact set | C ext C ext  $\mathcal{C}_{\text{ext}} ext$ |
| Intrinsic contact set | C int C int  $\mathcal{C}_{\text{int}} int$ |
| Space of contact forces | F F  $\mathcal{F}$ |
| Force equilibrium constraints matrix | A f e A f e  $\boldsymbol{A}_{fe}$ |
| FE-basis | B f e B f e  $\boldsymbol{B}_{fe}$ |
| Coordinate in FE-basis | x f e x f e  $\boldsymbol{x}_{fe}$ |
| Constraint matrix | C C  $\boldsymbol{C}$ |
| Measurement cone edge weight | m m  $\boldsymbol{m}$ |
| Coordinate transformation matrix | P f e P f e  $\boldsymbol{P}_{fe}$ |

**说明**: TABLE I: Nomenclature of the Proposed Framework

#### Table 2: TABLE II: Execution time comparison between SOCP and Geometric optimization.

| Method | 100 steps | 300 steps |
| --- | --- | --- |
| SOCP | 4.81 s 4.81 s 4.81s 4.81 | 13.36 s 13.36 s 13.36s 13.36 |
| Geometric | 0.31 s 0.31 s 0.31s 0.31 | 0.97 0.97 0.97 0.97 |

**说明**: TABLE II: Execution time comparison between SOCP and Geometric optimization.

#### Table 3: TABLE III: Success rate for wrench and cylinder grasp experiments with the mean and std of the force error of the grasp

| Object | Control | Success rate | Mean & std force error | |
| --- | --- | --- | --- | --- |
| wrench | f e s t f e s t f_{est} | 80 % percent 80 80\% 80 % | success: | 0.09 ± 0.04 N plus-or-minus 0.09 0.04 N 0.09\pm 0.04N 0.09 ± 0.04 |
| failure: | 0.73 ± 0.26 N plus-or-minus 0.73 0.26 N 0.73\pm 0.26N 0.73 ± 0.26 | | | |
| f r a w f r a w f_{raw} | 20 % percent 20 20\% 20 % | success: | 0.19 ± 0.16 N plus-or-minus 0.19 0.16 N 0.19\pm 0.16N 0.19 ± 0.16 | |
| failure: | 0.83 ± 0.57 N plus-or-minus 0.83 0.57 N 0.83\pm 0.57N 0.83 ± 0.57 | | | |
| cylinder | f e s t f e s t f_{est} | 60 % percent 60 60\% 60 % | success: | 0.35 ± 0.17 N plus-or-minus 0.35 0.17 N 0.35\pm 0.17N 0.35 ± 0.17 |
| failure: | 0.57 ± 0.31 N plus-or-minus 0.57 0.31 N 0.57\pm 0.31N 0.57 ± 0.31 | | | |
| f r a w f r a w f_{raw} | 0 % percent 0 0\% 0 % | success: | N/A | |
| failure: | 0.82 ± 0.45 N plus-or-minus 0.82 0.45 N 0.82\pm 0.45N 0.82 ± 0.45 | | | |

**说明**: TABLE III: Success rate for wrench and cylinder grasp experiments with the mean and std of the force error of the grasps when it was successful and when it failed.
## 实验解读

- 评价重点:围绕 agile-locomotion、接触推理、灵巧操作、力控制、grasping,优先看真实机器人结果、扰动恢复、接触稳定性、控制误差、成功率、实时性和 sim-to-real 差距。
- System0 视角:实验结论不只看最终成功率,还要看闭环过程中是否能处理状态估计误差、执行延迟、碰撞/接触突变和外部扰动。
- 与上层系统的关系:如果方法能在快闭环内稳定工作,它可以作为 System1 技能或 System2 规划的底层执行保障;如果只在离线或慢速设定下成立,则还不能直接作为身体级快闭环模块。
## 批判性思考

### 优点

- 论文把 agile-locomotion、接触推理、灵巧操作、力控制、grasping 放到机器人身体级闭环里讨论,和只做高层语义推理的工作相比,更接近真实机器人部署时会遇到的状态、动力学和安全约束。
- 若论文同时给出公式、系统结构图、硬件实验或消融结果,这些证据可以帮助判断方法是否真的能支撑毫秒到百毫秒级反应。

### 局限

- 需要重点核查控制频率、推理延迟、传感延迟、失败恢复、硬件磨损、未见场景泛化和安全边界;这些因素决定它能否成为可靠的 System0 模块。
- 对学习型方法,还要区分训练环境中的鲁棒性和真实部署中的鲁棒性,尤其是接触丰富任务、动态运动任务和人形机器人任务。
## 论文定位

- 研究对象:GeoDEx: A Unified Geometric Framework for Tactile Dexterous and Extrinsic Manipulation under Force Uncertainty。
- 关键词:agile-locomotion、接触推理、灵巧操作、力控制、grasping、机器人操作、鲁棒控制、tactile-feedback、tool-manipulation。
- System0 关联:关注机器人身体级快反应链路中“状态/接触/局部感知 -> 约束/优化/策略 -> 动作”的短闭环,不把问题停留在语言规划或任务分解层。

## 速查卡片

> [!summary] GeoDEx
> - **论文**: https://www.roboticsproceedings.org/rss21/p057.pdf
> - **arXiv**: http://arxiv.org/abs/2505.00647v1
> - **arXiv HTML**: https://arxiv.org/html/2505.00647v1
