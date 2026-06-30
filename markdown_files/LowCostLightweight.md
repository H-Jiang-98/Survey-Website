---
title: "A low-cost and lightweight 6 DoF bimanual arm for dynamic and contact-rich manipulation"
method_name: "low cost lightweight"
authors: ["Jaehyung Kim"]
year: 2025
venue: "RSS"
tags: ["contact-reasoning", "robot-manipulation", "robust-control", "legged-locomotion", "reinforcement-learning", "imitation-learning", "contact-rich-manipulation", "robot-generalization", "humanoid"]
image_source: "online"
arxiv_html: "https://ar5iv.labs.arxiv.org/html/2502.16908"
---
# low cost lightweight
## 一句话总结

> A low-cost and lightweight 6 DoF bimanual arm for dynamic and contact-rich manipulation 主要落在 [[actuator-modeling]]、[[assembly]]、[[bimanual-manipulation]]、[[compliance-control]] 这一类问题上;从 System0 视角看,它的价值在于把身体状态、接触变化、局部感知或动力学约束接到机器人低层闭环里,让动作生成更稳定、安全、实时。
## 核心贡献

1. 围绕 **A low-cost and lightweight 6 DoF bimanual arm for dynamic and contact-rich manipulation** 建立了一个与 actuator-modeling、assembly、bimanual-manipulation、compliance-control、接触推理、接触丰富操作 相关的机器人问题设定,重点不是高层语义规划,而是身体级闭环中的状态变化、接触变化或控制约束。
2. 方法层面的关键价值在于把感知、动力学、约束优化、策略学习或硬件反馈组织成可执行的低层动作生成机制,适合放进 System0 的快反应模块中理解。
3. 对系统集成而言,这类工作补上了 System1 技能选择与电机级控制之间的一层:它负责把上层目标变成动态可行、可恢复、可安全过滤的动作。
4. 若论文包含真实机器人、接触任务或仿真到真实迁移,核心应看它是否报告了闭环稳定性、扰动恢复、硬件延迟、失败模式和跨场景泛化。
## 问题背景

这篇论文可以放在 System0 的“具身快系统”背景下阅读:机器人需要在很短时间内根据本体状态、接触、力/触觉、局部视觉或动力学预测做出动作,而不是等待慢速规划器重新推理整个任务。actuator-modeling、assembly、bimanual-manipulation、compliance-control、接触推理、接触丰富操作、人形机器人 相关问题的难点通常不在“知道要做什么”,而在动作执行过程中状态会快速偏离、接触会突然改变、约束会实时收紧,低层系统必须及时修正。

### System0 关联

精读时优先看三件事:第一,论文使用哪些快速反馈量,例如关节状态、接触状态、力觉、触觉、深度或局部视觉;第二,反馈如何进入 MPC、WBC、策略网络、安全过滤器或恢复控制器;第三,输出是否直接影响机器人下一步动作,而不是只服务离线分析或高层任务规划。
## 方法详解

### System0 视角解析

从 System0 角度看,这篇论文的方法部分应当围绕 actuator-modeling、assembly、bimanual-manipulation、compliance-control、接触推理、接触丰富操作、人形机器人、足式运动 来读。核心问题是:系统如何从快速变化的机器人状态中构造控制输入,如何处理接触、碰撞、约束、模型误差或执行延迟,以及如何把这些信息变成下一时刻的动作。

如果方法里有优化问题,重点看目标函数、约束和求解频率;如果方法里有学习策略,重点看观测空间、动作空间、训练扰动和部署时的闭环频率;如果方法里有硬件或传感器,重点看传感链路是否真的进入控制回路。这样读可以把论文和 System2 语言规划、System1 技能编排区分开来。

### 关键公式与机制

#### 公式 1: [[动力学或策略机制]]

$$P_{i}\in\mathbb{R}^{3},i=1,2,\cdots,N$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 2: [[动力学或策略机制]]

$$\bar{P}=\frac{1}{N}\sum_{i=1}^{N}P_{i}$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 3: [[动力学或策略机制]]

$$\mu=\frac{1}{N}\sum_{i=1}^{N}\lVert P_{i}-\bar{P} \rVert,\quad\sigma=\sqrt{\frac{1}{N-1}\sum_{i=1}^{N}(\lVert P_{i}-\bar{P} \rVert)^{2}}$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。
## 关键图表

### Figure 1: Pipeline / core system figure: Comparison of gearboxes in low-gear ratio actuators (b) vs. high-gear ratio actuators

![Figure 1](https://ar5iv.labs.arxiv.org/html/2502.16908/assets/fig/gear_unncertainty_modi2.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Pipeline / core system figure: Comparison of gearboxes in low-gear ratio actuators (b) vs. high-gear ratio actuators”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 2: Hammering experiment setup. A nail is fixed on a wooden board with a height of 20 mm

![Figure 2](https://ar5iv.labs.arxiv.org/html/2502.16908/assets/fig/hammering_fig.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Hammering experiment setup. A nail is fixed on a wooden board with a height of 20 mm”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 3: a) Scenario 1: The robot moves the object from the right to the left of the bump

![Figure 3](https://ar5iv.labs.arxiv.org/html/2502.16908/assets/fig/bump_scenario_V2.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“a) Scenario 1: The robot moves the object from the right to the left of the bump”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### 论文表格

#### Table 1: TABLE I: Comparison of manipulators

| | ARMADA (Ours) | Franka Panda [21 ] | KUKA iiwa 7 R800 | Quigley et al. [46 ] | LIMS [36 ] | Nishii et al. [44 ] | PAMY2 [20 ] | BLUE [18 ] |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DoF (one arm) | 6 | 7 | 7 | 7 | 7 | 6 | 4 | 7 |
| Inertia (kg  \cdot m 2) | 0.234 | large | large | 0.083 | 0.599 | ? | ? | 0.75 |
| Moving mass 1 (kg) | 1.09 | 18 | 22.3 | 2 | 2.24 | 0.176 | 1.3 | 8.7 |
| End-effector speed (m/s) | 6.16 | 1.7 | 3.2 | 1.5 | 5.35 | ? | 12 | 2.1 |
| Total cost ($, one arm) | 3,040 | expensive | expensive | 4,135 | ? | ? | 14,540 | <5,000 |
| Open-source | O | X | X | O | X | X | O | O |
| Payload (kg) | 2.5 | 3 | 7 | 2 | 3 | 3 | ? | 2 |

**说明**: TABLE I: Comparison of manipulators

#### Table 2: TABLE II: ISO 9283 repeatability experiment results

| Point | Average distance (mm) | Std dev. (mm) | Repeatability (mm) |
| --- | --- | --- | --- |
| P 1 P_{1} | 1.048 | 0.546 | 2.687 |
| P 2 P_{2} | 1.042 | 0.409 | 2.269 |
| P 3 P_{3} | 0.939 | 0.689 | 3.006 |
| P 4 P_{4} | 0.751 | 0.520 | 2.311 |
| P 5 P_{5} | 1.104 | 0.584 | 2.857 |
| Average | 0.977 | 0.550 | 2.626 |
| Franka Panda [21 ] | 0.1 | | |
| KUKA iiwa 7 R800 | 0.1 | | |
| Quigley et al. [46 ] | *3 | | |
| LIMS [36 ] | 0.43 | | |
| Nishii et al. [44 ] | *2.2 | | |
| BLUE [18 ] | *3.7 | | |

**说明**: TABLE II: ISO 9283 repeatability experiment results

#### Table 3: TABLE III: Snatching experiment results

| Object | Glue | Threadlocker | Plastic box | Bracket | Tumbler | Total |
| --- | --- | --- | --- | --- | --- | --- |
| Success/Trial | 10/10 | 10/10 | 10/10 | 10/10 | 0/10 | 40/50 |

**说明**: TABLE III: Snatching experiment results

#### Table 4: TABLE IV: Box driving distance

| Trial | 1 | 2 | 3 | 4 | 5 | Avg. | Std dev. |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Distance (m) | 1.80 | 2.07 | 1.81 | 2.20 | 2.25 | 2.03 | 0.19 |

**说明**: TABLE IV: Box driving distance
## 实验解读

- 评价重点:围绕 actuator-modeling、assembly、bimanual-manipulation、compliance-control、接触推理,优先看真实机器人结果、扰动恢复、接触稳定性、控制误差、成功率、实时性和 sim-to-real 差距。
- System0 视角:实验结论不只看最终成功率,还要看闭环过程中是否能处理状态估计误差、执行延迟、碰撞/接触突变和外部扰动。
- 与上层系统的关系:如果方法能在快闭环内稳定工作,它可以作为 System1 技能或 System2 规划的底层执行保障;如果只在离线或慢速设定下成立,则还不能直接作为身体级快闭环模块。
## 批判性思考

### 优点

- 论文把 actuator-modeling、assembly、bimanual-manipulation、compliance-control、接触推理 放到机器人身体级闭环里讨论,和只做高层语义推理的工作相比,更接近真实机器人部署时会遇到的状态、动力学和安全约束。
- 若论文同时给出公式、系统结构图、硬件实验或消融结果,这些证据可以帮助判断方法是否真的能支撑毫秒到百毫秒级反应。

### 局限

- 需要重点核查控制频率、推理延迟、传感延迟、失败恢复、硬件磨损、未见场景泛化和安全边界;这些因素决定它能否成为可靠的 System0 模块。
- 对学习型方法,还要区分训练环境中的鲁棒性和真实部署中的鲁棒性,尤其是接触丰富任务、动态运动任务和人形机器人任务。
## 论文定位

- 研究对象:A low-cost and lightweight 6 DoF bimanual arm for dynamic and contact-rich manipulation。
- 关键词:actuator-modeling、assembly、bimanual-manipulation、compliance-control、接触推理、接触丰富操作、人形机器人、足式运动、运动模仿、non-prehensile-manipulation。
- System0 关联:关注机器人身体级快反应链路中“状态/接触/局部感知 -> 约束/优化/策略 -> 动作”的短闭环,不把问题停留在语言规划或任务分解层。

## 速查卡片

> [!summary] low cost lightweight
> - **论文**: https://www.roboticsproceedings.org/rss21/p055.pdf
> - **arXiv**: https://arxiv.org/abs/2502.16908
> - **arXiv HTML**: https://ar5iv.labs.arxiv.org/html/2502.16908
