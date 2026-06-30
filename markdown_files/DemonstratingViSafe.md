---
title: "Demonstrating ViSafe: Vision-enabled Safety for High-speed Detect and Avoid"
method_name: "Demonstrating ViSafe"
authors: ["Parv Kapoor"]
year: 2025
venue: "RSS"
tags: ["safe-control", "robot-generalization", "agile-locomotion", "collision-avoidance", "navigation"]
image_source: "online"
arxiv_html: "https://arxiv.org/html/2505.03694v2"
---
# Demonstrating ViSafe
## 一句话总结

> Demonstrating ViSafe: Vision-enabled Safety for High-speed Detect and Avoid 主要落在 [[aerial-robotics]]、[[agile-locomotion]]、[[certified-control]]、[[碰撞避免]] 这一类问题上;从 System0 视角看,它的价值在于把身体状态、接触变化、局部感知或动力学约束接到机器人低层闭环里,让动作生成更稳定、安全、实时。
## 核心贡献

1. 围绕 **Demonstrating ViSafe: Vision-enabled Safety for High-speed Detect and Avoid** 建立了一个与 aerial-robotics、agile-locomotion、certified-control、碰撞避免、inference-time-algorithm、navigation 相关的机器人问题设定,重点不是高层语义规划,而是身体级闭环中的状态变化、接触变化或控制约束。
2. 方法层面的关键价值在于把感知、动力学、约束优化、策略学习或硬件反馈组织成可执行的低层动作生成机制,适合放进 System0 的快反应模块中理解。
3. 对系统集成而言,这类工作补上了 System1 技能选择与电机级控制之间的一层:它负责把上层目标变成动态可行、可恢复、可安全过滤的动作。
4. 若论文包含真实机器人、接触任务或仿真到真实迁移,核心应看它是否报告了闭环稳定性、扰动恢复、硬件延迟、失败模式和跨场景泛化。
## 问题背景

这篇论文可以放在 System0 的“具身快系统”背景下阅读:机器人需要在很短时间内根据本体状态、接触、力/触觉、局部视觉或动力学预测做出动作,而不是等待慢速规划器重新推理整个任务。aerial-robotics、agile-locomotion、certified-control、碰撞避免、inference-time-algorithm、navigation、safe-control 相关问题的难点通常不在“知道要做什么”,而在动作执行过程中状态会快速偏离、接触会突然改变、约束会实时收紧,低层系统必须及时修正。

### System0 关联

精读时优先看三件事:第一,论文使用哪些快速反馈量,例如关节状态、接触状态、力觉、触觉、深度或局部视觉;第二,反馈如何进入 MPC、WBC、策略网络、安全过滤器或恢复控制器;第三,输出是否直接影响机器人下一步动作,而不是只服务离线分析或高层任务规划。
## 方法详解

### System0 视角解析

从 System0 角度看,这篇论文的方法部分应当围绕 aerial-robotics、agile-locomotion、certified-control、碰撞避免、inference-time-algorithm、navigation、safe-control、安全过滤 来读。核心问题是:系统如何从快速变化的机器人状态中构造控制输入,如何处理接触、碰撞、约束、模型误差或执行延迟,以及如何把这些信息变成下一时刻的动作。

如果方法里有优化问题,重点看目标函数、约束和求解频率;如果方法里有学习策略,重点看观测空间、动作空间、训练扰动和部署时的闭环频率;如果方法里有硬件或传感器,重点看传感链路是否真的进入控制回路。这样读可以把论文和 System2 语言规划、System1 技能编排区分开来。

### 关键公式与机制

#### 公式 1: [[动力学或策略机制]]

$$\dot{x_{own}}=\begin{bmatrix}v_{own}\cos{(\chi_{own})}\\ v_{own}\sin{(\chi_{own})}\\ 0\\ 0\end{bmatrix}+\begin{bmatrix}0&0\\ 0&0\\ 1&0\\ 0&1\end{bmatrix}u$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 2: [[动力学或策略机制]]

$$\dot{x_{int}}=\begin{bmatrix}v_{int}\cos{(\chi_{int})}\\ v_{int}\sin{(\chi_{int})}\\ \end{bmatrix}$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 3: [[优化目标/约束]]

$$\mathcal{C}:=\{x\in\mathcal{X}|h(x)\leq 0\}$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 4: [[动力学或策略机制]]

$$u=\begin{bmatrix}\dot{v}_{own}\\ \dot{\chi}_{own}\\ \end{bmatrix}$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 5: [[动力学或策略机制]]

$${v_{int}}_{North}={v_{own}}_{North}+\dot{d}\cos{\theta}+d\dot{\theta}\sin{\theta}$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 6: [[动力学或策略机制]]

$${v_{int}}_{East}={v_{own}}_{East}+\dot{d}\sin{\theta}+d\dot{\theta}\cos{\theta}$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 7: [[动力学或策略机制]]

$$\ddot{d}=\begin{bmatrix}\cos{(\alpha-\chi_{own})}\\ v_{own}\sin{(\alpha-\chi_{own})}\\ \end{bmatrix}^{\top}u$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 8: [[动力学或策略机制]]

$$\theta=\arctan\left(\frac{y_{t}}{x_{t}}\right)$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 9: [[动力学或策略机制]]

$$\ddot{d}=\dot{v}_{own}\cos{(\alpha-\chi_{own})}+\dot{\chi}_{own}v_{own}\sin{(\alpha-\chi_{own})}$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 10: [[动力学或策略机制]]

$${\chi_{int}}=\arctan\left(\frac{{v_{int}}_{East}}{{v_{int}}_{North}}\right)$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。
## 关键图表

### Figure 1: Overview of our ViSafe framework for real-world testing and hardware-in-the-loop simul

![Figure 1](https://arxiv.org/html/2505.03694v2/x1.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Overview of our ViSafe framework for real-world testing and hardware-in-the-loop simul”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 2: Diversity of the airborne collision testing scenarios: (a) The various encounter geome

![Figure 2](https://arxiv.org/html/2505.03694v2/x3.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Diversity of the airborne collision testing scenarios: (a) The various encounter geome”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 3: Average horizontal rate of closure comparisons across different encounter geometries i

![Figure 3](https://arxiv.org/html/2505.03694v2/x4.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Average horizontal rate of closure comparisons across different encounter geometries i”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### 论文表格

#### Table 1: TABLE I: Experimental Configurations

| Scenario ID | Intruder | Ownship | Collision Geometry | Hor. Closure Rate | Location |
| --- | --- | --- | --- | --- | --- |
| E1 | Multirotor (10 m/s) | Aurelia X6 (10 m/s) | Head-On | 20 m/s (72 km/hr) | Nardo Airfield |
| E2 | Multirotor (5 m/s) | Aurelia X6 (10 m/s) | Overtake | 5 m/s (18 km/hr) | Nardo Airfield |
| E3 | Multirotor (5 m/s) | Aurelia X6 (10 m/s) | Crossing | 11.18 m/s (40 km/hr) | Nardo Airfield |
| E4 | VTOL (30 m/s) | Aurelia X6 (10 m/s) | Head-On | 40 m/s (144 km/hr) | Nardo Airfield |
| E5 | Multirotor (10 m/s) | Aurelia X6 (10 m/s) | Head-On | 20 m/s (72 km/hr) | Leesburg, VA |

**说明**: TABLE I: Experimental Configurations

#### Table 2: TABLE II: Digital Twin & Hardware-in-the-Loop Benchmarking TABLE III: Real world benchmarking TABLE IV: ViSafe System Pr

| | Separation Minima (m) ↑ ↑ \uparrow ↑ | P(NMAC) ↓ ↓ \downarrow ↓ | Risk Ratio ↓ ↓ \downarrow ↓ | Number of violations ↓ ↓ \downarrow ↓ | | | | |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Scenario | Nominal | ViSafe | Nominal | ViSafe | Nominal | ViSafe | Nominal | ViSafe |
| E1 | 19.9 $\pm$ 2.15 | 35.55 $\pm$ 14.00 | 1.0 | 0.55 | 1.0 | 0.55 | 4000 | 2213 |
| E2 | 22.72 $\pm$ 2.63 | 27.635 $\pm$ 6.27 | 1.0 | 0.439 | 1.0 | 0.439 | 4000 | 1759 |
| E3 | 49.14 $\pm$ 0.7 | 59.04 $\pm$ 11.37 | 1.0 | 0.5248 | 1.0 | 0.5248 | 4000 | 2099 |
| Above Horizon - E1, E2, E3 | 30.58 $\pm$ 3.02 | 50.90 $\pm$ 4.72 | 1.0 | 0.1 | 1.0 | 0.1 | 6000 | 605 |
| Below Horizon - E1, E2, E3 | 28.6 $\pm$ 2.73 | 30.46 $\pm$ 3.26 | 1.0 | 0.91 | 1.0 | 0.91 | 6000 | 5466 |
| | Separation Minima (m) ↑ ↑ \uparrow ↑ | P(NMAC) ↓ ↓ \downarrow ↓ | Risk Ratio ↓ ↓ \downarrow ↓ | Number of violations ↓ ↓ \downarrow ↓ | | | | |
| Scenario | Nominal | ViSafe | Nominal | ViSafe | Nominal | ViSafe | Nominal | ViSafe |
| E1 | 41.28 $\pm$ 0.75 | 51.58 $\pm$ 12.76 | 1.0 | 0.33 | 1.0 | 0.5 | 4 | 2 |
| E2 | 44.25 $\pm$ 0.64 | 56.22 $\pm$ 2.88 | 1.0 | 0.25 | 1.0 | 0.25 | 4 | 1 |
| E3 | 47.45 $\pm$ 0.22 | 71.69 $\pm$ 11.65 | 1.0 | 0.5 | 1.0 | 0.25 | 4 | 1 |
| E4 | 30.54 $\pm$ 2.07 | 46.05 $\pm$ 8.29 | 1.0 | 0.0 | 1.0 | 0.0 | 2 | 0 |
| E5 | 27.39 $\pm$ 5.03 | 38.94 $\pm$ 7.83 | 1.0 | 0.0 | 1.0 | 0.0 | 3 | 0 |
| Above Horizon - E1, E2, E3 | 44.43 $\pm$ 3.02 | 59.43 $\pm$ 4.72 | 1.0 | 0.0 | 1.0 | 0.0 | 6 | 0 |
| Below Horizon - E1, E2, E3 | 44.59 $\pm$ 2.73 | 60.23 $\pm$ 18.76 | 1.0 | 0.667 | 1.0 | 0.667 | 6 | 4 |
| Component | Performance | | | | | | | |
| AirTrack Detection & Tracking (GPU) | 8 Hz | | | | | | | |
| Multi-View Fusion & State Estimation (CPU) | 24 Hz | | | | | | | |
| CBF-based Avoidance Control (CPU) | 50 Hz (solving a QP) | | | | | | | |
| Mean CPU Utilization | 55.21 % | | | | | | | |
| Mean GPU Utilization | 50.02 % | | | | | | | |
| Peak Memory Consumption | 15.96 GB | | | | | | | |

**说明**: TABLE II: Digital Twin & Hardware-in-the-Loop Benchmarking TABLE III: Real world benchmarking TABLE IV: ViSafe System Profiling
## 实验解读

- 评价重点:围绕 aerial-robotics、agile-locomotion、certified-control、碰撞避免、inference-time-algorithm,优先看真实机器人结果、扰动恢复、接触稳定性、控制误差、成功率、实时性和 sim-to-real 差距。
- System0 视角:实验结论不只看最终成功率,还要看闭环过程中是否能处理状态估计误差、执行延迟、碰撞/接触突变和外部扰动。
- 与上层系统的关系:如果方法能在快闭环内稳定工作,它可以作为 System1 技能或 System2 规划的底层执行保障;如果只在离线或慢速设定下成立,则还不能直接作为身体级快闭环模块。
## 批判性思考

### 优点

- 论文把 aerial-robotics、agile-locomotion、certified-control、碰撞避免、inference-time-algorithm 放到机器人身体级闭环里讨论,和只做高层语义推理的工作相比,更接近真实机器人部署时会遇到的状态、动力学和安全约束。
- 若论文同时给出公式、系统结构图、硬件实验或消融结果,这些证据可以帮助判断方法是否真的能支撑毫秒到百毫秒级反应。

### 局限

- 需要重点核查控制频率、推理延迟、传感延迟、失败恢复、硬件磨损、未见场景泛化和安全边界;这些因素决定它能否成为可靠的 System0 模块。
- 对学习型方法,还要区分训练环境中的鲁棒性和真实部署中的鲁棒性,尤其是接触丰富任务、动态运动任务和人形机器人任务。
## 论文定位

- 研究对象:Demonstrating ViSafe: Vision-enabled Safety for High-speed Detect and Avoid。
- 关键词:aerial-robotics、agile-locomotion、certified-control、碰撞避免、inference-time-algorithm、navigation、safe-control、安全过滤。
- System0 关联:关注机器人身体级快反应链路中“状态/接触/局部感知 -> 约束/优化/策略 -> 动作”的短闭环,不把问题停留在语言规划或任务分解层。

## 速查卡片

> [!summary] Demonstrating ViSafe
> - **论文**: https://www.roboticsproceedings.org/rss21/p002.pdf
> - **arXiv**: http://arxiv.org/abs/2505.03694v2
> - **arXiv HTML**: https://arxiv.org/html/2505.03694v2
