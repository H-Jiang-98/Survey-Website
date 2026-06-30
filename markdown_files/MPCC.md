---
title: "MPCC++: Model Predictive Contouring Control for Time-Optimal Flight with Safety Constraints"
method_name: "MPCC++"
authors: ["Maria Krinner"]
year: 2024
venue: "RSS"
tags: ["reinforcement-learning", "safe-control", "contact-rich-manipulation", "collision-avoidance", "navigation"]
image_source: "online"
arxiv_html: "https://arxiv.org/html/2403.17551v2"
---
# MPCC++
## 一句话总结

> MPCC++: Model Predictive Contouring Control for Time-Optimal Flight with Safety Constraints 主要落在 [[aerial-robotics]]、[[certified-control]]、[[碰撞避免]]、[[non-prehensile-manipulation]] 这一类问题上;从 System0 视角看,它的价值在于把身体状态、接触变化、局部感知或动力学约束接到机器人低层闭环里,让动作生成更稳定、安全、实时。
## 核心贡献

1. 围绕 **MPCC++: Model Predictive Contouring Control for Time-Optimal Flight with Safety Constraints** 建立了一个与 aerial-robotics、certified-control、碰撞避免、non-prehensile-manipulation、强化学习、safe-control 相关的机器人问题设定,重点不是高层语义规划,而是身体级闭环中的状态变化、接触变化或控制约束。
2. 方法层面的关键价值在于把感知、动力学、约束优化、策略学习或硬件反馈组织成可执行的低层动作生成机制,适合放进 System0 的快反应模块中理解。
3. 对系统集成而言,这类工作补上了 System1 技能选择与电机级控制之间的一层:它负责把上层目标变成动态可行、可恢复、可安全过滤的动作。
4. 若论文包含真实机器人、接触任务或仿真到真实迁移,核心应看它是否报告了闭环稳定性、扰动恢复、硬件延迟、失败模式和跨场景泛化。
## 问题背景

这篇论文可以放在 System0 的“具身快系统”背景下阅读:机器人需要在很短时间内根据本体状态、接触、力/触觉、局部视觉或动力学预测做出动作,而不是等待慢速规划器重新推理整个任务。aerial-robotics、certified-control、碰撞避免、non-prehensile-manipulation、强化学习、safe-control 相关问题的难点通常不在“知道要做什么”,而在动作执行过程中状态会快速偏离、接触会突然改变、约束会实时收紧,低层系统必须及时修正。

### System0 关联

精读时优先看三件事:第一,论文使用哪些快速反馈量,例如关节状态、接触状态、力觉、触觉、深度或局部视觉;第二,反馈如何进入 MPC、WBC、策略网络、安全过滤器或恢复控制器;第三,输出是否直接影响机器人下一步动作,而不是只服务离线分析或高层任务规划。
## 方法详解

### System0 视角解析

从 System0 角度看,这篇论文的方法部分应当围绕 aerial-robotics、certified-control、碰撞避免、non-prehensile-manipulation、强化学习、safe-control 来读。核心问题是:系统如何从快速变化的机器人状态中构造控制输入,如何处理接触、碰撞、约束、模型误差或执行延迟,以及如何把这些信息变成下一时刻的动作。

如果方法里有优化问题,重点看目标函数、约束和求解频率;如果方法里有学习策略,重点看观测空间、动作空间、训练扰动和部署时的闭环频率;如果方法里有硬件或传感器,重点看传感链路是否真的进入控制回路。这样读可以把论文和 System2 语言规划、System1 技能编排区分开来。

### 关键公式与机制

#### 公式 1: [[动力学或策略机制]]

$$\dot{\boldsymbol{\Omega}}=\frac{1}{\tau_{mot}}(\boldsymbol{\Omega_{des}}-\boldsymbol{\Omega})$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 2: [[动力学或策略机制]]

$$\begin{aligned} \dot{\boldsymbol{p}}&=\boldsymbol{v}&\dot{\boldsymbol{v}}&=\boldsymbol{g}+\frac{\mathbf{R}(\boldsymbol{q})\boldsymbol{f}_{T}}{m}\\ \dot{\boldsymbol{q}}&=\frac{\boldsymbol{q}}{2}\odot[0~{}~{}\boldsymbol{\omega}]^{T}&\dot{\boldsymbol{\omega}}&=\mathbf{J}^{-1}\left(\boldsymbol{\tau}_{T}-\boldsymbol{\omega}\times\mathbf{J}\boldsymbol{\omega}\right)\end{aligned}$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 3: [[动力学或策略机制]]

$$\begin{multlined}\sum_{k=0}^{N}\lVert \boldsymbol{e^{l}}(\theta_{k})\rVert_{Q_{l}}^{2}+\lVert \boldsymbol{e^{c}}(\theta_{k})\rVert_{Q_{c}}^{2}+\lVert \boldsymbol{\omega_{k}} \rVert_{Q_{\omega}}^{2}\\ +\lVert v_{\theta_{k}} \rVert_{R_{v_{\theta}}}^{2}+\lVert \Delta\boldsymbol{f_{k}} \rVert_{R_{\Delta f}}^{2}-\mu v_{\theta_{k}}\end{multlined}\sum_{k=0}^{N}\lVert \boldsymbol{e^{l}}(\theta_{k})\rVert_{Q_{l}}^{2}+\lVert \boldsymbol{e^{c}}(\theta_{k})\rVert_{Q_{c}}^{2}+\lVert \boldsymbol{\omega_{k}} \rVert_{Q_{\omega}}^{2}\\ +\lVert v_{\theta_{k}} \rVert_{R_{v_{\theta}}}^{2}+\lVert \Delta\boldsymbol{f_{k}} \rVert_{R_{\Delta f}}^{2}-\mu v_{\theta_{k}}$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 4: [[动力学或策略机制]]

$$=\begin{bmatrix}l/\sqrt{2}(f_{1}+f_{2}-f_{3}-f_{4})\\ l/\sqrt{2}(-f_{1}+f_{2}+f_{3}-f_{4})\\ c_{\tau}(f_{1}-f_{2}+f_{3}-f_{4})\end{bmatrix}$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 5: [[动力学或策略机制]]

$$J_{MPCC}(\boldsymbol{x})=\sum_{k=0}^{N}\lVert \boldsymbol{e^{l}}(\theta_{k})\rVert_{Q_{l}}^{2}+\lVert \boldsymbol{e^{c}}(\theta_{k})\rVert_{Q_{c}}^{2}-\mu{v_{\theta_{k}}}$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 6: [[动力学或策略机制]]

$$=\boldsymbol{C}_{\tau_{x}}\begin{bmatrix}v_{y}&\Omega^{2}&v_{y}\Omega^{2}\end{bmatrix}^{T}$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 7: [[动力学或策略机制]]

$$=\boldsymbol{C}_{\tau_{y}}\begin{bmatrix}v_{x}&\Omega^{2}&v_{x}\Omega^{2}\end{bmatrix}^{T}$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 8: [[动力学或策略机制]]

$$=\boldsymbol{C}_{f_{z}}\begin{bmatrix}v_{z}&v_{z}^{3}&v_{xy}&v_{xy}^{2}&v_{xy}\Omega^{2}&v_{z}\Omega^{2}&v_{xy}v_{z}\Omega^{2}\end{bmatrix}^{T}$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 9: [[动力学或策略机制]]

$$=\boldsymbol{C}_{\tau_{z}}\begin{bmatrix}v_{x}&v_{y}\end{bmatrix}^{T}$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 10: [[动力学或策略机制]]

$$\boldsymbol{p}_{0}(\theta_{k})=\boldsymbol{p}^{d}(\theta_{k})-W(\theta_{k})\cdot\boldsymbol{n}(\theta_{k})-H(\theta_{k})\cdot\boldsymbol{b}(\theta_{k})$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。
## 关键图表

### Figure 1: Pipeline / core system figure: Spatial constraint forming a tunnel around the centerline. The width and height are p

![Figure 1](https://arxiv.org/html/2403.17551v2/x1.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Pipeline / core system figure: Spatial constraint forming a tunnel around the centerline. The width and height are p”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 2: Simulation experiments of MPCC with the proposed MPCC++, both tuned using TuRBO

![Figure 2](https://arxiv.org/html/2403.17551v2/x3.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Simulation experiments of MPCC with the proposed MPCC++, both tuned using TuRBO”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 3: Real world flight trajectories on the Split-S track for the baseline MPCC [14 ], MP

![Figure 3](https://arxiv.org/html/2403.17551v2/x4.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Real world flight trajectories on the Split-S track for the baseline MPCC [14 ], MP”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### 论文表格

#### Table 1: TABLE I: Results for MPCC, MPCC++, and RL across three simulation environments - Simple, BEM, and Residual - as well as

| Category | Methods | Tuning | Environments | | | | | | | |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Simple | BEM | Residual | Real World | | | | | | | |
| Lap Time [s] | SR[%] | Lap Time [s] | SR[%] | Lap Time [s] | SR[%] | Lap Time [s] | SR[%] | | | |
| MPCC [14 ] | Nominal | WML [16 ] | 5.38 $\pm$ 0.1 | 100 | 5.51 $\pm$ 0.06 | 100 | 5.51 $\pm$ 0.13 | 83.3 | - | - |
| TuRBO | 5.65 $\pm$ 1.07 | 89.7 | 5.37 $\pm$ 0.06 | 100 | 5.62 $\pm$ 0.23 | 96.7 | 5.67 $\pm$ 1.06 | 59.3 | | |
| MPCC++ (ours) | Nominal | TuRBO | 5.16 $\pm$ 0.02 | 100 | 5.30 $\pm$ 0.02 | 100 | 5.37 $\pm$ 0.09 | 100 | 5.41 $\pm$ 0.14 | 100 |
| w/ augment. | TuRBO | 5.09 $\pm$ 0.10 | 100 | 5.15 $\pm$ 0.03 | 100 | 5.19 $\pm$ 0.03 | 100 | 5.38 $\pm$ 0.26 | 100 | |
| w/ random. | TuRBO | 5.20 $\pm$ 0.13 | 100 | 5.37 $\pm$ 0.08 | 100 | 5.26 $\pm$ 0.27 | 100 | - | - | |
| RL [7 ] | - | - | 5.14 $\pm$ 0.09 | 100 | - | - | 5.26 $\pm$ 0.32 | 100 | 5.35 $\pm$ 0.15 | 85.0 |

**说明**: TABLE I: Results for MPCC, MPCC++, and RL across three simulation environments - Simple, BEM, and Residual - as well as for real-world experiments. MPCC++ achieves a 100% success rate across all simulation and real-world environments.

#### Table 2: TABLE II: BO Training Success Rate (TSR): percentage of episodes during training completed successfully without gate co

| Category | Methods | Tuning | Environments | | |
| --- | --- | --- | --- | --- | --- |
| Simple | BEM | Residual | | | |
| MPCC [14 ] | Nominal | WML [16 ] | 62.9 | 75.7 | 64.4 |
| TuRBO | 69.2 | 70.3 | 53.0 | | |
| MPCC++ (ours) | Nominal | TuRBO | 99.5 | 100 | 99.8 |
| w/ augment. | TuRBO | 100 | 100 | 100 | |
| w/ random. | TuRBO | 91.3 | 93.9 | 89.0 | |

**说明**: TABLE II: BO Training Success Rate (TSR): percentage of episodes during training completed successfully without gate collisions.
## 实验解读

- 评价重点:围绕 aerial-robotics、certified-control、碰撞避免、non-prehensile-manipulation、强化学习,优先看真实机器人结果、扰动恢复、接触稳定性、控制误差、成功率、实时性和 sim-to-real 差距。
- System0 视角:实验结论不只看最终成功率,还要看闭环过程中是否能处理状态估计误差、执行延迟、碰撞/接触突变和外部扰动。
- 与上层系统的关系:如果方法能在快闭环内稳定工作,它可以作为 System1 技能或 System2 规划的底层执行保障;如果只在离线或慢速设定下成立,则还不能直接作为身体级快闭环模块。
## 批判性思考

### 优点

- 论文把 aerial-robotics、certified-control、碰撞避免、non-prehensile-manipulation、强化学习 放到机器人身体级闭环里讨论,和只做高层语义推理的工作相比,更接近真实机器人部署时会遇到的状态、动力学和安全约束。
- 若论文同时给出公式、系统结构图、硬件实验或消融结果,这些证据可以帮助判断方法是否真的能支撑毫秒到百毫秒级反应。

### 局限

- 需要重点核查控制频率、推理延迟、传感延迟、失败恢复、硬件磨损、未见场景泛化和安全边界;这些因素决定它能否成为可靠的 System0 模块。
- 对学习型方法,还要区分训练环境中的鲁棒性和真实部署中的鲁棒性,尤其是接触丰富任务、动态运动任务和人形机器人任务。
## 论文定位

- 研究对象:MPCC++: Model Predictive Contouring Control for Time-Optimal Flight with Safety Constraints。
- 关键词:aerial-robotics、certified-control、碰撞避免、non-prehensile-manipulation、强化学习、safe-control。
- System0 关联:关注机器人身体级快反应链路中“状态/接触/局部感知 -> 约束/优化/策略 -> 动作”的短闭环,不把问题停留在语言规划或任务分解层。

## 速查卡片

> [!summary] MPCC++
> - **论文**: https://www.roboticsproceedings.org/rss20/p109.pdf
> - **arXiv**: http://arxiv.org/abs/2403.17551v2
> - **arXiv HTML**: https://arxiv.org/html/2403.17551v2
