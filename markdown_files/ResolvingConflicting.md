---
title: "Resolving Conflicting Constraints in Multi-Agent Reinforcement Learning with Layered Safety"
method_name: "Resolving Conflicting"
authors: ["Jason Jangho Choi"]
year: 2025
venue: "RSS"
tags: ["reinforcement-learning", "safe-control", "collision-avoidance", "navigation"]
image_source: "online"
arxiv_html: "https://arxiv.org/html/2505.02293v1"
---
# Resolving Conflicting
## 一句话总结

> Resolving Conflicting Constraints in Multi-Agent Reinforcement Learning with Layered Safety 主要落在 [[aerial-robotics]]、[[certified-control]]、[[碰撞避免]]、[[navigation]] 这一类问题上;从 System0 视角看,它的价值在于把身体状态、接触变化、局部感知或动力学约束接到机器人低层闭环里,让动作生成更稳定、安全、实时。
## 核心贡献

1. 围绕 **Resolving Conflicting Constraints in Multi-Agent Reinforcement Learning with Layered Safety** 建立了一个与 aerial-robotics、certified-control、碰撞避免、navigation、reachability、强化学习 相关的机器人问题设定,重点不是高层语义规划,而是身体级闭环中的状态变化、接触变化或控制约束。
2. 方法层面的关键价值在于把感知、动力学、约束优化、策略学习或硬件反馈组织成可执行的低层动作生成机制,适合放进 System0 的快反应模块中理解。
3. 对系统集成而言,这类工作补上了 System1 技能选择与电机级控制之间的一层:它负责把上层目标变成动态可行、可恢复、可安全过滤的动作。
4. 若论文包含真实机器人、接触任务或仿真到真实迁移,核心应看它是否报告了闭环稳定性、扰动恢复、硬件延迟、失败模式和跨场景泛化。
## 问题背景

这篇论文可以放在 System0 的“具身快系统”背景下阅读:机器人需要在很短时间内根据本体状态、接触、力/触觉、局部视觉或动力学预测做出动作,而不是等待慢速规划器重新推理整个任务。aerial-robotics、certified-control、碰撞避免、navigation、reachability、强化学习、safe-control 相关问题的难点通常不在“知道要做什么”,而在动作执行过程中状态会快速偏离、接触会突然改变、约束会实时收紧,低层系统必须及时修正。

### System0 关联

精读时优先看三件事:第一,论文使用哪些快速反馈量,例如关节状态、接触状态、力觉、触觉、深度或局部视觉;第二,反馈如何进入 MPC、WBC、策略网络、安全过滤器或恢复控制器;第三,输出是否直接影响机器人下一步动作,而不是只服务离线分析或高层任务规划。
## 方法详解

### System0 视角解析

从 System0 角度看,这篇论文的方法部分应当围绕 aerial-robotics、certified-control、碰撞避免、navigation、reachability、强化学习、safe-control、安全过滤 来读。核心问题是:系统如何从快速变化的机器人状态中构造控制输入,如何处理接触、碰撞、约束、模型误差或执行延迟,以及如何把这些信息变成下一时刻的动作。

如果方法里有优化问题,重点看目标函数、约束和求解频率;如果方法里有学习策略,重点看观测空间、动作空间、训练扰动和部署时的闭环频率;如果方法里有硬件或传感器,重点看传感链路是否真的进入控制回路。这样读可以把论文和 System2 语言规划、System1 技能编排区分开来。

### 关键公式与机制

#### 公式 1: [[优化目标/约束]]

$$\begin{aligned} &\hat{\mathcal{S}}^{(i)}:=\Big{\{}\{s^{(j)}\}_{j\in I(i)}\;|\;\forall j\in I(i),V(s^{(ij)})\geq r_{\mathrm{safety}},\&\\ &\nexists j_{1},j_{2}\in I(i)\;\text{s.t.}\ V_{\mathrm{worst}}(s^{(ij_{1})})\!<\!r_{\mathrm{safety}}\;\&\;V_{\mathrm{worst}}(s^{(ij_{2})})\!<\!r_{\mathrm{safety}}\Big{\}}\end{aligned}$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 2: [[优化目标/约束]]

$$\begin{aligned} &\tilde{\mathcal{S}}^{(i)}:=\Big{\{}\{s^{(j)}\}_{j\in I(i)}\;|\;\forall j\in I(i),V(s^{(ij)})\geq r_{\mathrm{safety}},\&\\ &\nexists j_{1},j_{2}\in I(i)\;\text{s.t.}\;\mathrm{dist}(s^{(ij_{1})})\!<\!r_{\mathrm{conflict}}\;\&\;\mathrm{dist}(s^{(ij_{2})})\!<\!r_{\mathrm{conflict}}\Big{\}}\end{aligned}$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 3: [[优化目标/约束]]

$$,\;a_{\mathrm{safe}}^{(j)})=\arg\!\!\!\!\!\!\!\min_{(a^{(i)},a^{(j)})\in\mathcal{A}}|| a^{(i)}\!-\!a_{ $\mathrm{marl}}^{(i)}$ ||^{2}+|| a^{(j)}\!-\!a_{ $\mathrm{marl}}^{(j)}$ ||^{2}$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 4: [[优化目标/约束]]

$$V_{$\mathrm{worst}$}(s^{(ij)}_{0}):=[\;$\min$\max_{\{a^{(j)}_{k},a^{(i)}_{k}\}_{k\geq 0}\;\;\;\;\;\;\;\;}]\;J($s^{(ij)}_{0},\boldsymbol{a}^{(i)},\boldsymbol{a}^{(j)})$.$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 5: [[优化目标/约束]]

$$$\mathcal{BRT}(\mathcal{L}^{(ij)}):=$\{s^{(ij)}_{0}\;|\;$\forall\boldsymbol{a}^{(i)},\boldsymbol{a}^{(j)},\exists$t$\geq$0\;$\text{s.t.}$\;s^{(ij)}($t)\in\mathcal{L}^{(ij)}$\},$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 6: [[优化目标/约束]]

$$$\mathcal{R}_{\mathrm{total}}$(o^{(i)}_{k},a^{(i)}_{k})={$\mathcal{R}$}_{$\mathrm{tracking}$}(o^{(i)}_{k},a^{(i)}_{k})\\ +$\rho_{\mathrm{goal}}$$\mathcal{R}_{\mathrm{goal}}$(o^{(i)}_{k},a^{(i)}_{k})-$\rho_{\mathrm{conflict}}$$\mathcal{C}_{\mathrm{conflict}}$$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 7: [[优化目标/约束]]

$$$\mathcal{S}^{(ij)}:=$\{s^{(ij)}_{0}\;|\;$\exists\boldsymbol{a}^{(i)},\boldsymbol{a}^{(j)}$,\;$\text{s.t.}$\;$\forall$t$\geq$0,s^{(ij)}($t)\notin\mathcal{L}^{(ij)}$\},$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 8: [[优化目标/约束]]

$$$\min_{a^{(j)}\in\mathcal{A}}$$\nabla$B(s^{(ij)})\!$\cdot$\!$f^{(ij)}\big{(}$s^{(ij)},a^{(i)},$a^{(j)}\big{)}$\!+\!$\gamma$B($s^{(ij)})\geq$ 0,
$$**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。 **符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。 #### 公式 9: [[优化目标/约束]]$$
$\mathcal{A}=[-\omega_{\max},\omega_{\max}]\times[\mathrm{a}_{\min},\mathrm{a}_{\max}]$
$$**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。 **符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。 #### 公式 10: [[优化目标/约束]]$$
$=\arg$\!\!$\min_{a^{(i)}\in\mathcal{A}}$||a^{(i)}\!-\!a_{$\mathrm{marl}$}^{(i)}||^{2}
$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。
## 关键图表

### Figure 1: Pipeline / core system figure: The figure shows our approach using an example scenario of four agents. Agent i i i it

![Figure 1](https://arxiv.org/html/2505.02293v1/x1.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Pipeline / core system figure: The figure shows our approach using an example scenario of four agents. Agent i i i it”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 2: Crazyflie hardware experiment with the MARL policy learned by our method. The three dr

![Figure 2](https://arxiv.org/html/2505.02293v1/extracted/6411078/Images/hardware_safemarl.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Crazyflie hardware experiment with the MARL policy learned by our method. The three dr”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 3: We compare the recorded Crazyflie hardware experiment trajectories our method an

![Figure 3](https://arxiv.org/html/2505.02293v1/x4.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“We compare the recorded Crazyflie hardware experiment trajectories our method an”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### 论文表格

#### Table 1: TABLE I: Parameter Summary for Different Vehicle Dynamics

| Parameter | Air taxi (Sim) | Crazyflie |
| --- | --- | --- |
| Groundspeed | | |
| v min v v_{\min} | 60 knot (30 m/s) | -1.0 m/s |
| v max v v_{\max} | 175 knot (90 m/s) | 1.0 m/s |
| v nominal v nominal v_{ $\text{nominal}} nominal$ | 110 knot (57 m/s) | 0.5 m/s |
| Acceleration | | |
| a min a a_{\min} | -3.3 ft/s 2 (-1.0 m/s 2) | -0.5 m/s 2 |
| a max a a_{\max} | 6.6 ft/s 2 (2.0 m/s 2) | 0.5 m/s 2 |
| Angular Rate max $\omega_{\max}) (rad/s)$ | 0.1 | - |
| Sampling Rate (s) | 1.0 | 0.1 |
| Waypoint Thresholds ($\pm$) | | |
| Distance to Goal | 0.186 miles (0.3 km) | 0.2 m |
| Heading | 45° | 45° |
| Speed | 38.9 knot (20 m/s) | 0.1 m/s |
| Observation Range (r obs r obs r_{ $\mathrm{obs}} obs)$ | 3.1 mi. (5.0 km) | 4.0 m |
| Safety Distance (r safety r safety r_{ $\mathrm{safety}} safety)$ | 500 - 2200 ft (0.152 - 0.671 km) | 0.5 m |
| Potential Conflict Range (r conflict r conflict r_{ $\mathrm{conflict}} conflict)$ | 4600 ft (for r safety r safety r_{ $\mathrm{safety}} safety =2200 ft)$ | 1.0 m |

**说明**: TABLE I: Parameter Summary for Different Vehicle Dynamics

#### Table 2: TABLE II: Simulation results for Crazyflie dynamics with N N N =4, with time horizon 51.2s. We evaluate goal r

| Methods | Goal reach(%) | Near collision(%) |
| --- | --- | --- |
| DG-PPO | 96 $\pm$ 11.8 | 0.04 $\pm$ 0.16 |
| Exponential CBF | 100 $\pm$ 0 | 0.0 $\pm$ 0.0 |
| Our Method | 100 $\pm$ 0 | 0.0 $\pm$ 0.0 |

**说明**: 该表格来自论文中的公式、奖励项或实验设置；为避免 PDF 抽取噪声，网页中仅保留可渲染的核心符号和数值。

#### Table 3: TABLE III: Simulation results with N N N =8, and initial & goal positions arranged in lines random order

| Methods | Goal reach(%) | Near collision(%) |
| --- | --- | --- |
| DG-PPO | 100 $\pm$ 0 | 9.1 $\pm$ 2.7 |
| Exponential CBF | 93 $\pm$ 8.9 | 8.8 $\pm$ 10.7 |
| Our Method | 100 $\pm$ 0 | 0.0 $\pm$ 0.0 |

**说明**: TABLE III: Simulation results with N N N =8, and initial & goal positions arranged in lines random order. Videos are available in the supplementary material.

#### Table 4: TABLE IV: Simulation results of air taxi operations emulating potential peak traffic around the Bay Area—a scenario in w

| Methods | Merging Scenario (N N N =8, M M M =5) | | |
| --- | --- | --- | --- |
| Travel t(s)(↓ ↓ \downarrow ↓) | Near collision(%)(↓ ↓ \downarrow ↓) | Conflict(%)(↓ ↓ \downarrow ↓) | |
| Safety-blind | 675.6 | 0.055 | 2.4 |
| No penalty | 617.9 | 0.042 | 5.5 |
| Proposed | 450.5 | 0.021 | 3.2 |

**说明**: 该表格来自论文中的公式、奖励项或实验设置；为避免 PDF 抽取噪声，网页中仅保留可渲染的核心符号和数值。

#### Table 5: TABLE V: Simulation results of air taxi operations—a scenario in which two air corridors intersect with each other.

| Methods | Intersection Scenario (N N N =16, M M M =6) | | |
| --- | --- | --- | --- |
| Travel t(s)(↓ ↓ \downarrow ↓) | Near collision(%)(↓ ↓ \downarrow ↓) | Conflict(%)(↓ ↓ \downarrow ↓) | |
| Safety blind | 987.4 | 0.058 | 2.1 |
| No penalty | 780.5 | 0.129 | 3.8 |
| Proposed | 660.8 | 0.056 | 1.6 |

**说明**: TABLE V: Simulation results of air taxi operations—a scenario in which two air corridors intersect with each other.

#### Table 6: TABLE VI: Results of policies trained various methods for Crazyflie dynamics: We evaluate mean travel time (s) and

| Methods | Scenario 1 (Training) (N N N =4, M M M =2, L = 4 L 4 L=4 = 4) | Scenario 2 (N N N =6, M M M =3, L = 6 L 6 L=6 = 6) | Scenario 3 (N N N =3, M M M =3, L = 3 L 3 L=3 = 3) | | | | | | |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Travel time(s)(↓ ↓ \downarrow ↓) | Waypoint#(↑ ↑ \uparrow ↑) | Conflict(%)(↓ ↓ \downarrow ↓) | Travel t | Waypoint # | Conflict | Travel t | Waypoint # | Conflict | |
| 1 (safety-blind) | 18.33 18.33 18.33 18.33 | 1.62 ± 0.25 plus-or-minus 1.62 0.25 1.62\pm 0.25 1.62 ± 0.25 | 8.7 | 29.18 29.18 29.18 29.18 | 2.11 ± 0.24 plus-or-minus 2.11 0.24 2.11\pm 0.24 2.11 ± 0.24 | 21.9 | 19.08 19.08 19.08 19.08 | 2.14 ± 0.53 plus-or-minus 2.14 0.53 2.14\pm 0.53 2.14 ± 0.53 | 16.2 |
| 2 | 18.21 18.21 18.21 18.21 | 1.67 ± 0.19 plus-or-minus 1.67 0.19 1.67\pm 0.19 1.67 ± 0.19 | 7.1 | 29.77 | 2.06 ± 0.24 plus-or-minus 2.06 0.24 2.06\pm 0.24 2.06 ± 0.24 | 19.7 | 18.92 18.92 18.92 18.92 | 2.40 ± 0.47 plus-or-minus 2.40 0.47 2.40\pm 0.47 2.40 ± 0.47 | 16.0 |
| 3 | 17.73 17.73 17.73 17.73 | 1.76 ± 0.16 plus-or-minus 1.76 0.16 1.76\pm 0.16 1.76 ± 0.16 | 6.6 | 28.59 28.59 28.59 28.59 | 2.26 ± 0.21 plus-or-minus 2.26 0.21 2.26\pm 0.21 2.26 ± 0.21 | 19.7 | 18.44 18.44 18.44 18.44 | 2.41 ± 0.46 plus-or-minus 2.41 0.46 2.41\pm 0.46 2.41 ± 0.46 | 14.9 |
| 4 | 18.73 18.73 18.73 18.73 | 1.58 ± 0.23 plus-or-minus 1.58 0.23 1.58\pm 0.23 1.58 ± 0.23 | 7.2 | 29.11 29.11 29.11 29.11 | 2.20 ± 0.21 plus-or-minus 2.20 0.21 2.20\pm 0.21 2.20 ± 0.21 | 17.1 | 18.79 18.79 18.79 18.79 | 2.17 ± 0.47 plus-or-minus 2.17 0.47 2.17\pm 0.47 2.17 ± 0.47 | 13.4 |
| 5 (no penalty) | 17.56 17.56 17.56 17.56 | 1.75 ± 0.17 plus-or-minus 1.75 0.17 1.75\pm 0.17 1.75 ± 0.17 | 5.3 | 28.46 28.46 28.46 28.46 | 2.33 ± 0.20 plus-or-minus 2.33 0.20 2.33\pm 0.20 2.33 ± 0.20 | 16.9 | 16.09 16.09 16.09 16.09 | 2.78 ± 0.28 plus-or-minus 2.78 0.28 2.78\pm 0.28 2.78 ± 0.28 | 11.8 |
| 6 | 18.31 18.31 18.31 18.31 | 1.67 ± 0.17 plus-or-minus 1.67 0.17 1.67\pm 0.17 1.67 ± 0.17 | 5.9 | 28.92 28.92 28.92 28.92 | 2.27 ± 0.21 plus-or-minus 2.27 0.21 2.27\pm 0.21 2.27 ± 0.21 | 17.0 | 17.69 17.69 17.69 17.69 | 2.42 ± 0.42 plus-or-minus 2.42 0.42 2.42\pm 0.42 2.42 ± 0.42 | 15.0 |
| 7 | 17.90 17.90 17.90 17.90 | 1.71 ± 0.18 plus-or-minus 1.71 0.18 1.71\pm 0.18 1.71 ± 0.18 | 5.3 | 28.98 28.98 28.98 28.98 | 2.32 ± 0.18 plus-or-minus 2.32 0.18 2.32\pm 0.18 2.32 ± 0.18 | 15.8 | 17.59 17.59 17.59 17.59 | 2.54 ± 0.34 plus-or-minus 2.54 0.34 2.54\pm 0.34 2.54 ± 0.34 | 11.2 |
| 8 | 20.52 20.52 20.52 20.52 | 1.26 ± 0.21 plus-or-minus 1.26 0.21 1.26\pm 0.21 1.26 ± 0.21 | 2.7 | 31.25 31.25 31.25 31.25 | 1.74 ± 0.27 plus-or-minus 1.74 0.27 1.74\pm 0.27 1.74 ± 0.27 | 6.9 | 18.36 18.36 18.36 18.36 | 2.20 ± 0.36 plus-or-minus 2.20 0.36 2.20\pm 0.36 2.20 ± 0.36 | 8.6 |
| 9 (proposed) | 17.81 | 1.78 $\pm$ 0.18 | 5.4 | 28.59 | 2.42 $\pm$ 0.20 | 15.1 | 16.91 | 2.71 $\pm$ 0.24 | 10.8 |

**说明**: TABLE VI: Results of policies trained various methods for Crazyflie dynamics: We evaluate mean travel time (s) and number of reached waypoints (Waypoint #) for performance, and the percentage of the events involving multiple agents encountered within the potential conflict range in the trajectory data (Conflict %) for safety risk. Note that in these simulations, the agent never violated safety for all methods due to our safety filter, except in the training scenario when the agent is initialized at the safety-violating states. (N N N =number of agents, M M M =number of waypoints, L L L =world size)
## 实验解读

- 评价重点:围绕 aerial-robotics、certified-control、碰撞避免、navigation、reachability,优先看真实机器人结果、扰动恢复、接触稳定性、控制误差、成功率、实时性和 sim-to-real 差距。
- System0 视角:实验结论不只看最终成功率,还要看闭环过程中是否能处理状态估计误差、执行延迟、碰撞/接触突变和外部扰动。
- 与上层系统的关系:如果方法能在快闭环内稳定工作,它可以作为 System1 技能或 System2 规划的底层执行保障;如果只在离线或慢速设定下成立,则还不能直接作为身体级快闭环模块。
## 批判性思考

### 优点

- 论文把 aerial-robotics、certified-control、碰撞避免、navigation、reachability 放到机器人身体级闭环里讨论,和只做高层语义推理的工作相比,更接近真实机器人部署时会遇到的状态、动力学和安全约束。
- 若论文同时给出公式、系统结构图、硬件实验或消融结果,这些证据可以帮助判断方法是否真的能支撑毫秒到百毫秒级反应。

### 局限

- 需要重点核查控制频率、推理延迟、传感延迟、失败恢复、硬件磨损、未见场景泛化和安全边界;这些因素决定它能否成为可靠的 System0 模块。
- 对学习型方法,还要区分训练环境中的鲁棒性和真实部署中的鲁棒性,尤其是接触丰富任务、动态运动任务和人形机器人任务。
## 论文定位

- 研究对象:Resolving Conflicting Constraints in Multi-Agent Reinforcement Learning with Layered Safety。
- 关键词:aerial-robotics、certified-control、碰撞避免、navigation、reachability、强化学习、safe-control、安全过滤。
- System0 关联:关注机器人身体级快反应链路中“状态/接触/局部感知 -> 约束/优化/策略 -> 动作”的短闭环,不把问题停留在语言规划或任务分解层。

## 速查卡片

> [!summary] Resolving Conflicting
> - **论文**: https://www.roboticsproceedings.org/rss21/p094.pdf
> - **arXiv**: http://arxiv.org/abs/2505.02293v1
> - **arXiv HTML**: https://arxiv.org/html/2505.02293v1
