---
title: "Collision-Affording Point Trees: SIMD-Amenable Nearest Neighbors for Fast Motion Planning with Pointclouds"
method_name: "Collision-Affording Point Trees"
authors: ["Clayton Ramsey"]
year: 2024
venue: "RSS"
tags: ["contact-reasoning", "real-time-control", "collision-avoidance"]
image_source: "online"
arxiv_html: "https://arxiv.org/html/2406.02807"
---
# Collision-Affording Point Trees
## 一句话总结

> Collision-Affording Point Trees: SIMD-Amenable Nearest Neighbors for Fast Motion Planning with Pointclouds 主要落在 [[碰撞避免]]、[[接触推理]]、[[实时控制]] 这一类问题上;从 System0 视角看,它的价值在于把身体状态、接触变化、局部感知或动力学约束接到机器人低层闭环里,让动作生成更稳定、安全、实时。
## 核心贡献

1. 围绕 **Collision-Affording Point Trees: SIMD-Amenable Nearest Neighbors for Fast Motion Planning with Pointclouds** 建立了一个与 碰撞避免、接触推理、实时控制 相关的机器人问题设定,重点不是高层语义规划,而是身体级闭环中的状态变化、接触变化或控制约束。
2. 方法层面的关键价值在于把感知、动力学、约束优化、策略学习或硬件反馈组织成可执行的低层动作生成机制,适合放进 System0 的快反应模块中理解。
3. 对系统集成而言,这类工作补上了 System1 技能选择与电机级控制之间的一层:它负责把上层目标变成动态可行、可恢复、可安全过滤的动作。
4. 若论文包含真实机器人、接触任务或仿真到真实迁移,核心应看它是否报告了闭环稳定性、扰动恢复、硬件延迟、失败模式和跨场景泛化。
## 问题背景

这篇论文可以放在 System0 的“具身快系统”背景下阅读:机器人需要在很短时间内根据本体状态、接触、力/触觉、局部视觉或动力学预测做出动作,而不是等待慢速规划器重新推理整个任务。碰撞避免、接触推理、实时控制 相关问题的难点通常不在“知道要做什么”,而在动作执行过程中状态会快速偏离、接触会突然改变、约束会实时收紧,低层系统必须及时修正。

### System0 关联

精读时优先看三件事:第一,论文使用哪些快速反馈量,例如关节状态、接触状态、力觉、触觉、深度或局部视觉;第二,反馈如何进入 MPC、WBC、策略网络、安全过滤器或恢复控制器;第三,输出是否直接影响机器人下一步动作,而不是只服务离线分析或高层任务规划。
## 方法详解

### System0 视角解析

从 System0 角度看,这篇论文的方法部分应当围绕 碰撞避免、接触推理、实时控制 来读。核心问题是:系统如何从快速变化的机器人状态中构造控制输入,如何处理接触、碰撞、约束、模型误差或执行延迟,以及如何把这些信息变成下一时刻的动作。

如果方法里有优化问题,重点看目标函数、约束和求解频率;如果方法里有学习策略,重点看观测空间、动作空间、训练扰动和部署时的闭环频率;如果方法里有硬件或传感器,重点看传感链路是否真的进入控制回路。这样读可以把论文和 System2 语言规划、System1 技能编排区分开来。

### 关键公式与机制

#### 公式 1: [[动力学或策略机制]]

$$r_{\text{filter}}=2$\mathrm{cm}$$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 2: [[动力学或策略机制]]

$$\exists p\in\mathit{PC}:\lVert x-p \rVert\leq r\rightarrow\exists q\in P_{i}:\lVert x-q \rVert\leq r$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 3: [[优化目标/约束]]

$$P_{i}=\{p\in\mathit{PC}:\min_{x_{c}\in c}\lVert x_{c}-p \rVert\leq r_{\text{max}}\}$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 4: [[优化目标/约束]]

$$\min_{x_{c}\in c}\lVert x_{c}-p \rVert\leq r\leq r_{\text{max}}$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 5: [[优化目标/约束]]

$$\delta(O,\mathit{PC})=\sup_{x\in O}\min_{p\in PC}\lVert x-p \rVert$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 6: [[动力学或策略机制]]

$$\lVert x-p^{*} \rVert\leq r_{\text{filter}}+\delta(O,\mathit{PC})\leq r_{\text{min}}$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 7: [[动力学或策略机制]]

$$r_{q}^{\prime}=r_{q}+r_{\text{filter}}$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 8: [[动力学或策略机制]]

$$\exists p^{*}\in\mathit{PC}^{\prime}:\lVert x-p^{*} \rVert\leq r_{\text{filter}}+\delta(O,\mathit{PC})$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 9: [[动力学或策略机制]]

$$z_{1}\leftarrow\{p\in z\cup B_{2}:\text{$c_{1}$ affords $p$ at $r_{\text{max}}$}\}$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 10: [[动力学或策略机制]]

$$z_{2}\leftarrow\{p\in z\cup B_{1}:\text{$c_{2}$ affords $p$ at $r_{\text{max}}$}\}$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。
## 关键图表

### Figure 1: Pipeline / core system figure: a

![Figure 1](https://arxiv.org/html/2406.02807/extracted/5641460/figures/realsense_color.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Pipeline / core system figure: a”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 2: b

![Figure 2](https://arxiv.org/html/2406.02807/extracted/5641460/figures/realsense_raw_pc.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“b”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 3: Construction times and average query throughput for pointcloud collision-checking app

![Figure 3](https://arxiv.org/html/2406.02807/extracted/5641460/figures/throughput_fig.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Construction times and average query throughput for pointcloud collision-checking app”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### 论文表格

#### Table 1: TABLE I: Statistics the MotionBenchMaker [chamzas2021mbm ] dataset. We compare parallel (capt simd) and sequent

| | Backend | Mean Filter | Mean Build | Med. Build | 95% Build | Mean Plan | Med. Plan | 95% Plan | Mean Simpl. | Mean Total | Med. Total | 95% Total | Succ. |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| UR5 | OctoMap | 2.897 | 120.935 | 104.892 | 218.075 | 78.894 | 24.736 | 375.782 | 23.290 | 220.906 | 179.170 | 521.604 | 96.5% |
| nanoflann | 0.346 | 0.339 | 0.596 | 15.526 | 4.797 | 78.979 | 7.810 | 26.730 | 15.806 | 92.221 | 100.0% | | |
| capt Seq. | 5.859 | 5.608 | 10.467 | 0.833 | 0.273 | 4.226 | 0.368 | 9.935 | 8.829 | 16.279 | 100.0% | | |
| capt simd | 0.490 | 0.146 | 2.542 | 0.225 | 9.499 | 8.630 | 15.541 | 100.0% | | | | | |
| Primitives | - | - | - | - | 0.204 | 0.051 | 1.039 | 0.067 | 0.272 | 0.117 | 1.145 | 100.0% | |
| Panda | OctoMap | 2.712 | 66.127 | 49.503 | 122.043 | 25.579 | 9.363 | 109.210 | 24.424 | 118.844 | 103.436 | 223.829 | 99.6% |
| nanoflann | 0.315 | 0.233 | 0.611 | 5.958 | 3.241 | 20.352 | 7.423 | 16.607 | 13.726 | 39.647 | 100.0% | | |
| capt Seq. | 4.269 | 3.227 | 8.758 | 0.342 | 0.182 | 1.166 | 0.477 | 7.781 | 6.430 | 15.078 | 100.0% | | |
| capt simd | 0.198 | 0.102 | 0.728 | 0.261 | 7.445 | 6.055 | 14.704 | 100.0% | | | | | |
| Primitives | - | - | - | - | 0.078 | 0.034 | 0.363 | 0.070 | 0.148 | 0.100 | 0.468 | 100.0% | |
| Fetch | OctoMap | 3.345 | 160.976 | 124.376 | 335.959 | 309.302 | 212.953 | 894.198 | 121.892 | 429.362 | 375.223 | 900.221 | 96.6% |
| nanoflann | 0.469 | 0.391 | 0.863 | 177.912 | 70.510 | 661.252 | 33.131 | 204.500 | 102.742 | 680.011 | 99.9% | | |
| capt Seq. | 6.073 | 5.145 | 10.508 | 18.331 | 3.863 | 76.251 | 1.831 | 29.624 | 16.015 | 88.011 | 99.7% | | |
| capt simd | 10.736 | 2.159 | 42.974 | 0.983 | 21.213 | 12.648 | 58.239 | 99.7% | | | | | |
| Primitives | - | - | - | - | 3.873 | 0.779 | 16.290 | 0.262 | 4.136 | 0.966 | 16.744 | 99.3% | |

**说明**: TABLE I: Statistics the MotionBenchMaker [chamzas2021mbm ] dataset. We compare parallel (capt simd) and sequential (capt Seq.) collision-affording point tree collision checking against other collision checking backends. All collision-checking backends except for the primitive geometry used the same filtered point clouds for planning. We report the mean, median, and 95th percentile times spent constructing each collision-checking data structure, planning, simplifying the path, and the total time spent from observation to completed plan for each robot and collision-checking backend. All times are in milliseconds.

#### Table 2: TABLE II: Empirical measurements of point cloud dispersion before (left grouping) and after (right grouping) applying t

| | $\delta(O,\mathit{PC})$ | $\delta(O,\mathit{PC}^{\prime})$ | | | | |
| --- | --- | --- | --- | --- | --- | --- |
| r filter r filter r_{ $\text{filter}} filter$ | Mean | Median | 95% | Mean | Median | 95% |
| 0.5 | 0.00336 | 0.00198 | 0.01075 | 0.00763 | 0.00675 | 0.01314 |
| 1 | 0.01253 | 0.01182 | 0.01774 | | | |
| 1.5 | 0.01758 | 0.01696 | 0.02314 | | | |
| 1.8 | 0.02055 | 0.01995 | 0.02663 | | | |
| 1.9 | 0.02158 | 0.02097 | 0.02790 | | | |
| 2 | 0.02261 | 0.02199 | 0.02919 | | | |

**说明**: 该表格来自论文中的公式、奖励项或实验设置；为避免 PDF 抽取噪声，网页中仅保留可渲染的核心符号和数值。

#### Table 3: TABLE III: Impact of r filter r filter r_{$\text{filter}$} filter

| r filter r filter r_{ $\text{filter}} filter$ | \| P C ′ \| P C ′ \|PC^{\prime}\| \| ′ \| | Filter | Build | Plan | Simpl. | Total |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 16225 | 6.944 | 30.373 | 0.168 | 0.275 | 37.761 |
| 1.5 | 7872 | 5.682 | 7.669 | 0.079 | 0.085 | 13.517 |
| 2 | 4614 | 4.990 | 3.964 | 0.087 | 0.089 | 9.131 |

**说明**: 该表格来自论文中的公式、奖励项或实验设置；为避免 PDF 抽取噪声，网页中仅保留可渲染的核心符号和数值。

#### Table 4: TABLE IV: Effect of r filter r filter r_{$\text{filter}$} filter

| r filter r filter r_{ $\text{filter}} filter$ | Mean Filter | Mean Build | Med. Build | 95% Build | Mean Plan | Med. Plan | 95% Plan | Mean Simpl. | Mean Total | Med. Total | 95% Total | Succ. |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.5 | 4.544 | 62.560 | 56.294 | 122.966 | 0.427 | 0.210 | 1.581 | 0.480 | 68.012 | 61.546 | 132.708 | 100.0 |
| 1 | 3.307 | 19.339 | 13.286 | 41.888 | 0.288 | 0.136 | 1.136 | 0.344 | 23.280 | 16.514 | 49.667 | 100.0 |
| 1.5 | 2.742 | 8.069 | 5.954 | 16.595 | 0.228 | 0.112 | 0.923 | 0.288 | 11.329 | 8.753 | 22.851 | 100.0 |
| 1.8 | 2.514 | 5.378 | 4.278 | 10.895 | 0.212 | 0.103 | 0.923 | 0.265 | 8.370 | 6.905 | 16.565 | 100.0 |
| 1.9 | 2.454 | 4.701 | 3.613 | 9.760 | 0.207 | 0.102 | 0.734 | 0.265 | 7.628 | 6.281 | 15.149 | 100.0 |
| 2 | 2.400 | 4.098 | 3.125 | 8.467 | 0.192 | 0.099 | 0.702 | 0.253 | 6.945 | 5.658 | 13.812 | 100.0 |

**说明**: 该表格来自论文中的公式、奖励项或实验设置；为避免 PDF 抽取噪声，网页中仅保留可渲染的核心符号和数值。
## 实验解读

- 评价重点:围绕 碰撞避免、接触推理、实时控制,优先看真实机器人结果、扰动恢复、接触稳定性、控制误差、成功率、实时性和 sim-to-real 差距。
- System0 视角:实验结论不只看最终成功率,还要看闭环过程中是否能处理状态估计误差、执行延迟、碰撞/接触突变和外部扰动。
- 与上层系统的关系:如果方法能在快闭环内稳定工作,它可以作为 System1 技能或 System2 规划的底层执行保障;如果只在离线或慢速设定下成立,则还不能直接作为身体级快闭环模块。
## 批判性思考

### 优点

- 论文把 碰撞避免、接触推理、实时控制 放到机器人身体级闭环里讨论,和只做高层语义推理的工作相比,更接近真实机器人部署时会遇到的状态、动力学和安全约束。
- 若论文同时给出公式、系统结构图、硬件实验或消融结果,这些证据可以帮助判断方法是否真的能支撑毫秒到百毫秒级反应。

### 局限

- 需要重点核查控制频率、推理延迟、传感延迟、失败恢复、硬件磨损、未见场景泛化和安全边界;这些因素决定它能否成为可靠的 System0 模块。
- 对学习型方法,还要区分训练环境中的鲁棒性和真实部署中的鲁棒性,尤其是接触丰富任务、动态运动任务和人形机器人任务。
## 论文定位

- 研究对象:Collision-Affording Point Trees: SIMD-Amenable Nearest Neighbors for Fast Motion Planning with Pointclouds。
- 关键词:碰撞避免、接触推理、实时控制。
- System0 关联:关注机器人身体级快反应链路中“状态/接触/局部感知 -> 约束/优化/策略 -> 动作”的短闭环,不把问题停留在语言规划或任务分解层。

## 速查卡片

> [!summary] Collision-Affording Point Trees
> - **论文**: https://www.roboticsproceedings.org/rss20/p038.pdf
> - **arXiv**: http://arxiv.org/abs/2406.02807
> - **arXiv HTML**: https://arxiv.org/html/2406.02807
