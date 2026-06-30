---
title: "Adaptive Locomotion on Mud through Proprioceptive Sensing of Substrate Properties"
method_name: "Adaptive Locomotion Mud"
authors: ["Shipeng Liu"]
year: 2025
venue: "RSS"
tags: ["contact-reasoning", "robot-manipulation", "robust-control", "real-time-control", "legged-locomotion", "adaptive-control", "closed-loop-control", "loco-manipulation", "state-estimation"]
image_source: "online"
arxiv_html: "https://arxiv.org/html/2504.19607v2"
---
# Adaptive Locomotion Mud
## 一句话总结

> Adaptive Locomotion on Mud through Proprioceptive Sensing of Substrate Properties 主要落在 [[actuator-modeling]]、[[adaptive-control]]、[[closed-loop-control]]、[[contact-estimation]] 这一类问题上;从 System0 视角看,它的价值在于把身体状态、接触变化、局部感知或动力学约束接到机器人低层闭环里,让动作生成更稳定、安全、实时。
## 核心贡献

1. 围绕 **Adaptive Locomotion on Mud through Proprioceptive Sensing of Substrate Properties** 建立了一个与 actuator-modeling、adaptive-control、closed-loop-control、contact-estimation、接触推理、足式运动 相关的机器人问题设定,重点不是高层语义规划,而是身体级闭环中的状态变化、接触变化或控制约束。
2. 方法层面的关键价值在于把感知、动力学、约束优化、策略学习或硬件反馈组织成可执行的低层动作生成机制,适合放进 System0 的快反应模块中理解。
3. 对系统集成而言,这类工作补上了 System1 技能选择与电机级控制之间的一层:它负责把上层目标变成动态可行、可恢复、可安全过滤的动作。
4. 若论文包含真实机器人、接触任务或仿真到真实迁移,核心应看它是否报告了闭环稳定性、扰动恢复、硬件延迟、失败模式和跨场景泛化。
## 问题背景

这篇论文可以放在 System0 的“具身快系统”背景下阅读:机器人需要在很短时间内根据本体状态、接触、力/触觉、局部视觉或动力学预测做出动作,而不是等待慢速规划器重新推理整个任务。actuator-modeling、adaptive-control、closed-loop-control、contact-estimation、接触推理、足式运动、移动操作 相关问题的难点通常不在“知道要做什么”,而在动作执行过程中状态会快速偏离、接触会突然改变、约束会实时收紧,低层系统必须及时修正。

### System0 关联

精读时优先看三件事:第一,论文使用哪些快速反馈量,例如关节状态、接触状态、力觉、触觉、深度或局部视觉;第二,反馈如何进入 MPC、WBC、策略网络、安全过滤器或恢复控制器;第三,输出是否直接影响机器人下一步动作,而不是只服务离线分析或高层任务规划。
## 方法详解

### System0 视角解析

从 System0 角度看,这篇论文的方法部分应当围绕 actuator-modeling、adaptive-control、closed-loop-control、contact-estimation、接触推理、足式运动、移动操作、mobile-manipulation 来读。核心问题是:系统如何从快速变化的机器人状态中构造控制输入,如何处理接触、碰撞、约束、模型误差或执行延迟,以及如何把这些信息变成下一时刻的动作。

如果方法里有优化问题,重点看目标函数、约束和求解频率;如果方法里有学习策略,重点看观测空间、动作空间、训练扰动和部署时的闭环频率;如果方法里有硬件或传感器,重点看传感链路是否真的进入控制回路。这样读可以把论文和 System2 语言规划、System1 技能编排区分开来。

### 关键公式与机制

#### 公式 1: [[动力学或策略机制]]

$$\tau_{\text{sense}}\approx k_{t}I$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 2: [[动力学或策略机制]]

$$\mathbf{F}_{\text{sense}}=[f_{x},f_{y},f_{z}]^{T}$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 3: [[动力学或策略机制]]

$$\mathbf{\tau}_{\text{sense}}=[\tau_{1},\tau_{2}]^{T}$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 4: [[动力学或策略机制]]

$$k_{p}=\operatorname*{arg\,min}_{k_{p}}\left(\hat{f}_{z}(k_{p})-f_{z}\right)^{2}$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 5: [[动力学或策略机制]]

$$\hat{f_{z}}=k_{p}\int A(z)\,dz$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 6: [[动力学或策略机制]]

$$d\hat{f_{z}}=k_{p}\cdot A(z)\cdot dz$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 7: [[动力学或策略机制]]

$$k_{s}=\operatorname*{arg\,min}_{k_{s}}\left(\hat{f_{x}}(k_{s})-f_{x}\right)^{2}$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 8: [[动力学或策略机制]]

$$\hat{f_{x}}=k_{s}\int bz\,dz$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 9: [[动力学或策略机制]]

$$\begin{bmatrix}f_{x}\\ f_{z}\end{bmatrix}=\begin{bmatrix}\frac{\cos\alpha}{l}&0\\ -\frac{\sin\beta\cdot\sin\alpha}{l}&\frac{\cos\beta\cdot\cos\alpha}{l}\end{bmatrix}\begin{bmatrix}\tau_{1}\\ \tau_{2}\end{bmatrix}$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 10: [[动力学或策略机制]]

$$\mathbf{F}_{\text{sense}}\approx\mathbf{J}^{\mathrm{-T}}\mathbf{\tau}_{\text{sense}}$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。
## 关键图表

### Figure 1: Pipeline / core system figure: Natural mud mixtures vary significantly in water content and particle size. Inspired b

![Figure 1](https://arxiv.org/html/2504.19607v2/x1.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Pipeline / core system figure: Natural mud mixtures vary significantly in water content and particle size. Inspired b”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 2: Robots and Experiment Setup. (A) A bio-inspired flipper-based robot with a direct-driv

![Figure 2](https://arxiv.org/html/2504.19607v2/x2.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Robots and Experiment Setup. (A) A bio-inspired flipper-based robot with a direct-driv”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 3: Measurements from the single flipper experiment. (A) During one full gait cycle, the r

![Figure 3](https://arxiv.org/html/2504.19607v2/x4.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Measurements from the single flipper experiment. (A) During one full gait cycle, the r”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### 论文表格

#### Table 1: TABLE I: Average forward velocity on different terrains of three different gait configurations.

| | w = 48.5 % w percent 48.5 w=48.5\% = 48.5 % | w = 53.1 % w percent 53.1 w=53.1\% = 53.1 % | w = 45.3 % w percent 45.3 w=45.3\% = 45.3 % |
| --- | --- | --- | --- |
| v (cm / s) v cm s v\,( $\mathrm{cm/s}) (/) z = 3 z 3 z=3 = 3$ | 7.6 $\pm$ 0.5 | 2.1 $\pm$ 0.8 | 7.2 $\pm$ 0.5 |
| v (cm / s) v cm s v\,( $\mathrm{cm/s}) (/) z = 5 z 5 z=5 = 5$ | 6.6 $\pm$ 0.3 | 5.5 $\pm$ 0.7 | 3.0 $\pm$ 1.0 |
| v (cm / s) v cm s v\,( $\mathrm{cm/s}) (/) adaptive z z z$ | 6.4 $\pm$ 0.4 | 5.1 $\pm$ 0.3 | 7.2 $\pm$ 0.1 |

**说明**: TABLE I: Average forward velocity on different terrains of three different gait configurations.
## 实验解读

- 评价重点:围绕 actuator-modeling、adaptive-control、closed-loop-control、contact-estimation、接触推理,优先看真实机器人结果、扰动恢复、接触稳定性、控制误差、成功率、实时性和 sim-to-real 差距。
- System0 视角:实验结论不只看最终成功率,还要看闭环过程中是否能处理状态估计误差、执行延迟、碰撞/接触突变和外部扰动。
- 与上层系统的关系:如果方法能在快闭环内稳定工作,它可以作为 System1 技能或 System2 规划的底层执行保障;如果只在离线或慢速设定下成立,则还不能直接作为身体级快闭环模块。
## 批判性思考

### 优点

- 论文把 actuator-modeling、adaptive-control、closed-loop-control、contact-estimation、接触推理 放到机器人身体级闭环里讨论,和只做高层语义推理的工作相比,更接近真实机器人部署时会遇到的状态、动力学和安全约束。
- 若论文同时给出公式、系统结构图、硬件实验或消融结果,这些证据可以帮助判断方法是否真的能支撑毫秒到百毫秒级反应。

### 局限

- 需要重点核查控制频率、推理延迟、传感延迟、失败恢复、硬件磨损、未见场景泛化和安全边界;这些因素决定它能否成为可靠的 System0 模块。
- 对学习型方法,还要区分训练环境中的鲁棒性和真实部署中的鲁棒性,尤其是接触丰富任务、动态运动任务和人形机器人任务。
## 论文定位

- 研究对象:Adaptive Locomotion on Mud through Proprioceptive Sensing of Substrate Properties。
- 关键词:actuator-modeling、adaptive-control、closed-loop-control、contact-estimation、接触推理、足式运动、移动操作、mobile-manipulation、proprioception、实时控制。
- System0 关联:关注机器人身体级快反应链路中“状态/接触/局部感知 -> 约束/优化/策略 -> 动作”的短闭环,不把问题停留在语言规划或任务分解层。

## 速查卡片

> [!summary] Adaptive Locomotion Mud
> - **论文**: https://www.roboticsproceedings.org/rss21/p126.pdf
> - **arXiv**: http://arxiv.org/abs/2504.19607v2
> - **arXiv HTML**: https://arxiv.org/html/2504.19607v2
