---
title: "DOGlove: Dexterous Manipulation with a Low-Cost Open-Source Haptic Force Feedback Glove"
method_name: "DOGlove"
authors: ["Han Zhang"]
year: 2025
venue: "RSS"
tags: ["contact-reasoning", "robot-manipulation", "robust-control", "imitation-learning", "contact-rich-manipulation", "closed-loop-control", "dexterous-manipulation", "tactile-feedback"]
image_source: "online"
arxiv_html: "https://arxiv.org/html/2502.07730v1"
---
# DOGlove
## 一句话总结

> DOGlove: Dexterous Manipulation with a Low-Cost Open-Source Haptic Force Feedback Glove 主要落在 [[actuator-modeling]]、[[closed-loop-control]]、[[接触推理]]、[[接触丰富操作]] 这一类问题上;从 System0 视角看,它的价值在于把身体状态、接触变化、局部感知或动力学约束接到机器人低层闭环里,让动作生成更稳定、安全、实时。
## 核心贡献

1. 围绕 **DOGlove: Dexterous Manipulation with a Low-Cost Open-Source Haptic Force Feedback Glove** 建立了一个与 actuator-modeling、closed-loop-control、接触推理、接触丰富操作、灵巧操作、力控制 相关的机器人问题设定,重点不是高层语义规划,而是身体级闭环中的状态变化、接触变化或控制约束。
2. 方法层面的关键价值在于把感知、动力学、约束优化、策略学习或硬件反馈组织成可执行的低层动作生成机制,适合放进 System0 的快反应模块中理解。
3. 对系统集成而言,这类工作补上了 System1 技能选择与电机级控制之间的一层:它负责把上层目标变成动态可行、可恢复、可安全过滤的动作。
4. 若论文包含真实机器人、接触任务或仿真到真实迁移,核心应看它是否报告了闭环稳定性、扰动恢复、硬件延迟、失败模式和跨场景泛化。
## 问题背景

这篇论文可以放在 System0 的“具身快系统”背景下阅读:机器人需要在很短时间内根据本体状态、接触、力/触觉、局部视觉或动力学预测做出动作,而不是等待慢速规划器重新推理整个任务。actuator-modeling、closed-loop-control、接触推理、接触丰富操作、灵巧操作、力控制、haptic-feedback 相关问题的难点通常不在“知道要做什么”,而在动作执行过程中状态会快速偏离、接触会突然改变、约束会实时收紧,低层系统必须及时修正。

### System0 关联

精读时优先看三件事:第一,论文使用哪些快速反馈量,例如关节状态、接触状态、力觉、触觉、深度或局部视觉;第二,反馈如何进入 MPC、WBC、策略网络、安全过滤器或恢复控制器;第三,输出是否直接影响机器人下一步动作,而不是只服务离线分析或高层任务规划。
## 方法详解

### System0 视角解析

从 System0 角度看,这篇论文的方法部分应当围绕 actuator-modeling、closed-loop-control、接触推理、接触丰富操作、灵巧操作、力控制、haptic-feedback、模仿学习 来读。核心问题是:系统如何从快速变化的机器人状态中构造控制输入,如何处理接触、碰撞、约束、模型误差或执行延迟,以及如何把这些信息变成下一时刻的动作。

如果方法里有优化问题,重点看目标函数、约束和求解频率;如果方法里有学习策略,重点看观测空间、动作空间、训练扰动和部署时的闭环频率;如果方法里有硬件或传感器,重点看传感链路是否真的进入控制回路。这样读可以把论文和 System2 语言规划、System1 技能编排区分开来。

### 关键公式与机制

#### 公式 1: [[动力学或策略机制]]

$$\alpha_{\text{joint}}=\frac{\text{V}_{\text{ADC}}}{\text{V}_{\text{CC}}}\cdot 360$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。
## 关键图表

### Figure 1: Pipeline / core system figure: Teleoperation demos. a) While squeezing condensed milk, the operator regulates the flo

![Figure 1](https://arxiv.org/html/2502.07730v1/x2.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Pipeline / core system figure: Teleoperation demos. a) While squeezing condensed milk, the operator regulates the flo”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 2: Teleoperation experiments and quantitative results. a) Without visual feedback, force

![Figure 2](https://arxiv.org/html/2502.07730v1/x9.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Teleoperation experiments and quantitative results. a) Without visual feedback, force”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 3: The imitation learning experiment. (a) The robot must first locate the correct positi

![Figure 3](https://arxiv.org/html/2502.07730v1/x10.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“The imitation learning experiment. (a) The robot must first locate the correct positi”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### 论文表格

#### Table 1: Table I: The combination strategy for haptic force feedback in DOGlove.

| Force Sensor Readings (g) | Haptic Feedback | Force Feedback |
| --- | --- | --- |
| <10 | ✗ | ✗ |
| 10–50 | ✓ | ✗ |
| 50–100 | ✓ | ✓ |
| >100 | ✗ | ✓ |

**说明**: Table I: The combination strategy for haptic force feedback in DOGlove.

#### Table 2: Table II: Success rates in the user study. All feedback modes perform well for the basic pairs. For the challenging pair

| | Pair 1 | Pair 2 | Pair 3 | Pair 4 | Pair 5 |
| --- | --- | --- | --- | --- | --- |
| Only Force | 5/5 | 5/5 | 5/5 | 4/5 | 0/5 |
| Only Haptic | 5/5 | 5/5 | 5/5 | 3/5 | 3/5 |
| Haptic+Force | 5/5 | 5/5 | 5/5 | 3/5 | 2/5 |

**说明**: Table II: Success rates in the user study. All feedback modes perform well for the basic pairs. For the challenging pairs, force feedback is more sensitive to softness, while haptic feedback is more sensitive to shape.

#### Table 3: Table III: Quantitative experiment results. Haptic force feedback enables operators to achieve a higher success rate and

| | Success Rate | Average Completion Time (s) |
| --- | --- | --- |
| Only Force | 9/10 | 18.92 |
| Only Haptic | 9/10 | 21.16 |
| Haptic+Force | 10/10 | 19.89 |
| No Haptic/Force | 4/10 | 24.76 |
| AnyTeleop | 1/10 | 54.85 |

**说明**: Table III: Quantitative experiment results. Haptic force feedback enables operators to achieve a higher success rate and a faster average completion time, as haptic feedback provides contact information, while force feedback indicates the proper timing for in-hand rotation.
## 实验解读

- 评价重点:围绕 actuator-modeling、closed-loop-control、接触推理、接触丰富操作、灵巧操作,优先看真实机器人结果、扰动恢复、接触稳定性、控制误差、成功率、实时性和 sim-to-real 差距。
- System0 视角:实验结论不只看最终成功率,还要看闭环过程中是否能处理状态估计误差、执行延迟、碰撞/接触突变和外部扰动。
- 与上层系统的关系:如果方法能在快闭环内稳定工作,它可以作为 System1 技能或 System2 规划的底层执行保障;如果只在离线或慢速设定下成立,则还不能直接作为身体级快闭环模块。
## 批判性思考

### 优点

- 论文把 actuator-modeling、closed-loop-control、接触推理、接触丰富操作、灵巧操作 放到机器人身体级闭环里讨论,和只做高层语义推理的工作相比,更接近真实机器人部署时会遇到的状态、动力学和安全约束。
- 若论文同时给出公式、系统结构图、硬件实验或消融结果,这些证据可以帮助判断方法是否真的能支撑毫秒到百毫秒级反应。

### 局限

- 需要重点核查控制频率、推理延迟、传感延迟、失败恢复、硬件磨损、未见场景泛化和安全边界;这些因素决定它能否成为可靠的 System0 模块。
- 对学习型方法,还要区分训练环境中的鲁棒性和真实部署中的鲁棒性,尤其是接触丰富任务、动态运动任务和人形机器人任务。
## 论文定位

- 研究对象:DOGlove: Dexterous Manipulation with a Low-Cost Open-Source Haptic Force Feedback Glove。
- 关键词:actuator-modeling、closed-loop-control、接触推理、接触丰富操作、灵巧操作、力控制、haptic-feedback、模仿学习、运动模仿、retargeting。
- System0 关联:关注机器人身体级快反应链路中“状态/接触/局部感知 -> 约束/优化/策略 -> 动作”的短闭环,不把问题停留在语言规划或任务分解层。

## 速查卡片

> [!summary] DOGlove
> - **论文**: https://www.roboticsproceedings.org/rss21/p104.pdf
> - **arXiv**: http://arxiv.org/abs/2502.07730v1
> - **arXiv HTML**: https://arxiv.org/html/2502.07730v1
