---
title: "Provably-Safe, Online System Identification"
method_name: "Provably Safe Online"
authors: ["Bohao Zhang"]
year: 2025
venue: "RSS"
tags: ["contact-reasoning", "robot-manipulation", "robust-control", "real-time-control", "safe-control", "collision-avoidance", "state-estimation"]
image_source: "online"
arxiv_html: "https://arxiv.org/html/2504.21486v1"
---
# Provably Safe Online
## 一句话总结

> Provably-Safe, Online System Identification 主要落在 [[certified-control]]、[[碰撞避免]]、[[接触推理]]、[[inertial-estimation]] 这一类问题上;从 System0 视角看,它的价值在于把身体状态、接触变化、局部感知或动力学约束接到机器人低层闭环里,让动作生成更稳定、安全、实时。
## 核心贡献

1. 围绕 **Provably-Safe, Online System Identification** 建立了一个与 certified-control、碰撞避免、接触推理、inertial-estimation、实时控制、机器人操作 相关的机器人问题设定,重点不是高层语义规划,而是身体级闭环中的状态变化、接触变化或控制约束。
2. 方法层面的关键价值在于把感知、动力学、约束优化、策略学习或硬件反馈组织成可执行的低层动作生成机制,适合放进 System0 的快反应模块中理解。
3. 对系统集成而言,这类工作补上了 System1 技能选择与电机级控制之间的一层:它负责把上层目标变成动态可行、可恢复、可安全过滤的动作。
4. 若论文包含真实机器人、接触任务或仿真到真实迁移,核心应看它是否报告了闭环稳定性、扰动恢复、硬件延迟、失败模式和跨场景泛化。
## 问题背景

这篇论文可以放在 System0 的“具身快系统”背景下阅读:机器人需要在很短时间内根据本体状态、接触、力/触觉、局部视觉或动力学预测做出动作,而不是等待慢速规划器重新推理整个任务。certified-control、碰撞避免、接触推理、inertial-estimation、实时控制、机器人操作、鲁棒控制 相关问题的难点通常不在“知道要做什么”,而在动作执行过程中状态会快速偏离、接触会突然改变、约束会实时收紧,低层系统必须及时修正。

### System0 关联

精读时优先看三件事:第一,论文使用哪些快速反馈量,例如关节状态、接触状态、力觉、触觉、深度或局部视觉;第二,反馈如何进入 MPC、WBC、策略网络、安全过滤器或恢复控制器;第三,输出是否直接影响机器人下一步动作,而不是只服务离线分析或高层任务规划。
## 方法详解

### System0 视角解析

从 System0 角度看,这篇论文的方法部分应当围绕 certified-control、碰撞避免、接触推理、inertial-estimation、实时控制、机器人操作、鲁棒控制、safe-control 来读。核心问题是:系统如何从快速变化的机器人状态中构造控制输入,如何处理接触、碰撞、约束、模型误差或执行延迟,以及如何把这些信息变成下一时刻的动作。

如果方法里有优化问题,重点看目标函数、约束和求解频率;如果方法里有学习策略,重点看观测空间、动作空间、训练扰动和部署时的闭环频率;如果方法里有硬件或传感器,重点看传感链路是否真的进入控制回路。这样读可以把论文和 System2 语言规划、System1 技能编排区分开来。

### 关键公式与机制

#### 公式 1: [[动力学或策略机制]]

$$p(q(t_{2}),\dot{q}(t_{2}),\theta)-p(q(t_{1}),\dot{q}(t_{1}),\theta)=\\ =\int_{t_{1}}^{t_{2}}(C^{T}(q(t),\dot{q}(t),\theta)\dot{q}(t)-g(q(t),\theta)\\ -F(\dot{q}(t),\ddot{q}(t),\theta)+\tau(t))\text{d}t,$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 2: [[动力学或策略机制]]

$$\mathbf{W}(k):=\\ \begin{bmatrix}W_{e}(q(t_{1};k),\dot{q}(t_{1};k),\ddot{q}(t_{1};k))\\ \vdots\\ W_{e}(q(t_{N_{s}};k),\dot{q}(t_{N_{s}};k),\ddot{q}(t_{N_{s}};k))\end{bmatrix},$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 3: [[动力学或策略机制]]

$$\mathbf{U}(\mathbf{q}_{1:N},\dot{\mathbf{q}}_{1:N},\boldsymbol{\tau}_{1:N},\theta_{\text{{r}}}):=\begin{bmatrix}U_{1:1+h}\\ U_{1+h:1+2h}\\ \vdots\\ U_{1+(N_{h}-1)h:1+N_{h}h}\end{bmatrix}\in\mathbb{R}^{N_{h}n},$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 4: [[动力学或策略机制]]

$$\big{(}W_{m}(q(t_{2}),\dot{q}(t_{2}))-W_{m}(q(t_{1}),\dot{q}(t_{1}))\big{)}\theta=\\ \left(\int_{t_{1}}^{t_{2}}W_{c}(q(t),\dot{q}(t))\text{d}t\right)\theta-\\ I_{a}\circ(\dot{q}(t_{2})-\dot{q}(t_{1}))+\int_{t_{1}}^{t_{2}}\tau(t)\text{d}t$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 5: [[优化目标/约束]]

$$\min_{\eta_{e}\in\mathbb{R}^{10}}\lVert \mathbf{Y}(\mathbf{q}_{1:N},\dot{\mathbf{q}}_{1:N})P(\eta_{e})-\mathbf{U}(\mathbf{q}_{1:N},\dot{\mathbf{q}}_{1:N},\boldsymbol{\tau}_{1:N},\theta_{\text{{r}},0})\rVert,$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 6: [[动力学或策略机制]]

$$U_{i:i+h}=-(W_{m,r}(q(t_{i+h}),\dot{q}(t_{i+h}))-W_{m,r}(q(t_{i}),\dot{q}(t_{i}))+\\ +\sum_{j=i}^{i+h}W_{c,r}(q(t),\dot{q}(t))(t_{j+1}-t_{j}))\theta_{r}-\\ -I_{a}\circ(\dot{q}(t_{i+h})-\dot{q}(t_{i}))+\sum_{j=i}^{i+h}\tau(t_{j})(t_{j+1}-t_{j}).$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 7: [[动力学或策略机制]]

$$\text{LMI}(\theta_{\text{{ip}},j}):=\begin{bmatrix}\left(\frac{\text{tr}(I_{j})}{2}\mathbb{1}_{3\times 3}-I_{j}\right)\ &p_{j}m_{j}\\ p_{j}^{T}m_{j}&m_{j}\end{bmatrix}\succ\mathbb{0}_{4\times 4},$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 8: [[动力学或策略机制]]

$$\mathbf{Y}(\mathbf{q}_{1:N},\dot{\mathbf{q}}_{1:N}):=\begin{bmatrix}Y_{1:1+h}\\ Y_{1+h:1+2h}\\ \vdots\\ Y_{1+(N_{h}-1)h:1+N_{h}h}\end{bmatrix}\in\mathbb{R}^{N_{h}n\times 10},$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 9: [[动力学或策略机制]]

$$\mathbf{Y}(\mathbf{q}_{1:N},\dot{\mathbf{q}}_{1:N})\theta_{e}^{*}=\mathbf{U}(\mathbf{q}_{1:N},\dot{\mathbf{q}}_{1:N},\boldsymbol{\tau}_{1:N},\theta_{\text{{r}},0})$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 10: [[动力学或策略机制]]

$$\frac{\partial\theta^{*}_{e}}{\partial\mathbf{m}}(\mathbf{m})=-\frac{\partial P}{\partial\eta_{e}}(\eta_{e}^{*}(\mathbf{m}))\left(\frac{\partial^{2}J}{\partial\eta_{e}^{2}}(\mathbf{m},\eta_{e}^{*}(\mathbf{m}))\right)^{-1}\\ \cdot\frac{\partial^{2}J}{\partial\mathbf{m}\partial\eta_{e}}(\mathbf{m},\eta_{e}^{*}(\mathbf{m})).$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。
## 关键图表

### Figure 1: This figure summarizes the proposed framework. Initially, the approach assumes an ove

![Figure 1](https://arxiv.org/html/2504.21486v1/x1.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“This figure summarizes the proposed framework. Initially, the approach assumes an ove”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 2: This figure illustrates the tracking error of our method and all the comparisons on

![Figure 2](https://arxiv.org/html/2504.21486v1/extracted/6401389/Figures/full_results_0.25.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“This figure illustrates the tracking error of our method and all the comparisons on”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 3: This figure illustrates the tracking error of our method and all the comparisons on

![Figure 3](https://arxiv.org/html/2504.21486v1/extracted/6401389/Figures/full_results_0.50.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“This figure illustrates the tracking error of our method and all the comparisons on”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### 论文表格

#### Table 1: TABLE I: The conservative nominal parameters and interval uncertainties of the end-effector assigned to the planner and

| Inertial parameters | nominal e e $\theta_{\text{{e}},0} e, 0$ | interval e e $\theta_{\text{{e}}}] [e ]$ |
| --- | --- | --- |
| m m m (kg) | 3.2 | [1.2, 5.2] |
| p x  m  p x m p_{x}\cdot m  (kg   \cdot  m) | 0.0 | [-0.4, 0.4] |
| p y  m  p y m p_{y}\cdot m  (kg   \cdot  m) | 0.0 | [-0.4, 0.4] |
| p z  m  p z m p_{z}\cdot m  (kg   \cdot  m) | -0.5 | [-1.0, -0.1] |
| X X X X XX (kg   \cdot  m 2) | 0.0 | [-0.2, 0.2] |
| Y Y Y Y YY (kg   \cdot  m 2) | 0.0 | [-0.2, 0.2] |
| Z Z Z Z ZZ (kg   \cdot  m 2) | 0.0 | [-0.2, 0.2] |
| X Y X Y XY (kg   \cdot  m 2) | 0.0 | [-0.2, 0.2] |
| Y Z Y Z YZ (kg   \cdot  m 2) | 0.0 | [-0.2, 0.2] |
| X Z X Z XZ (kg   \cdot  m 2) | 0.0 | [-0.2, 0.2] |

**说明**: 该表格来自论文中的公式、奖励项或实验设置；为避免 PDF 抽取噪声，网页中仅保留可渲染的核心符号和数值。

#### Table 2: TABLE II: All comparisons evaluated in the hardware experiments, categorized by the controller type, the robot model us

| methods | controller | robot model | robot model uncertainties | exciting trajectories |
| --- | --- | --- | --- | --- |
| ours | ARMOUR robust [28 ] | $\theta_{0} 0 from Algorithm. 1$ | $\theta] [] from Algorithm. 1$ | Algorithm. 2 |
| wrong | ARMOUR robust [28 ] | assume no payload at end-effector | 0% | N/A |
| conservative | ARMOUR robust [28 ] | nominal in TABLE I | interval in TABLE I | N/A |
| random | ARMOUR robust [28 ] | $\theta_{0} 0 from Algorithm. 1$ | $\theta] [] from Algorithm. 1$ | random |
| adap-1 | adaptive [38 ] | identified by adaptive controller | N/A | N/A |
| adap-1-excit | adaptive [38 ] | identified by adaptive controller | N/A | Algorithm. 2 |
| adap-2 | adaptive [1 ] | identified by adaptive controller | N/A | N/A |
| adap-2-excit | adaptive [1 ] | identified by adaptive controller | N/A | Algorithm. 2 |
| grav-pid | gravity compensated PID [37, (6.19)] | assume no payload at end-effector | N/A | N/A |
| grav-pid-ours | gravity compensated PID [37, (6.19)] | $\theta_{0} 0 from Algorithm. 1$ | N/A | Algorithm. 2 |
| grav-pid-excit | gravity compensated PID [37, (6.19)] | $\theta_{0} 0 from Algorithm. 1$ | N/A | [5 ] |

**说明**: TABLE II: All comparisons evaluated in the hardware experiments, categorized by the controller type, the robot model used in the controller, the robot model uncertainties used in the controller, and whether exciting trajectories were used for system identification prior to task execution. Methods marked ”N/A” in the “exciting trajectories” column skip the identification phase and directly execute the task.

#### Table 3: TABLE III: Results of our method and baseline comparisons across three hardware experiments. Each experiment was repeat

| Methods | Experiment (a) | Experiment (b) | Experiment (c) |
| --- | --- | --- | --- |
| ours | success | success | success |
| wrong | fail at 8lb | fail at 8lb | collide |
| conservative | fail at 8lb | fail at 8lb | collide |
| random | success | fail at 8lb | collide |
| adap-1 | success | fail at 8lb | collide |
| adap-1-excit | success | success | collide |
| adap-2 | fail at 8lb | fail at 4lb | collide |
| adap-2-excit | fail at 8lb | fail at 4lb | collide |
| grav-pid | success | fail at 6lb | collide |
| grav-pid-ours | success | success | collide |
| grav-pid-excit | fail at 4lb | fail at 4lb | collide |

**说明**: TABLE III: Results of our method and baseline comparisons across three hardware experiments. Each experiment was repeated five times and returned consistent results.

#### Table 4: TABLE IV: The Pearson correlation coefficients between the condition number of the standard dynamics regressor W W $\mat$

| forward integration horizon h h h | Pearson correlation coefficients |
| --- | --- |
| 1 | 1.0000 |
| 5 | 1.0000 |
| 10 | 0.9994 |
| 20 | 0.9934 |
| 50 | 0.9540 |
| 100 | 0.6709 |
| 200 | 0.3537 |
| 500 | 0.2789 |

**说明**: 该表格来自论文中的公式、奖励项或实验设置；为避免 PDF 抽取噪声，网页中仅保留可渲染的核心符号和数值。

#### Table 5: TABLE V: This table reports the 2-norm condition number of Y Y $\mathbf{Y}$ based on data collected during the ent

| dumbbell | ours | random |
| --- | --- | --- |
| 4lb | 274.030 | 479.573 |
| 5lb | 282.840 | 418.078 |
| 6lb | 250.704 | 373.448 |
| 7lb | 258.517 | 312.352 |
| 8lb | 233.539 | 413.381 |

**说明**: 该表格来自论文中的公式、奖励项或实验设置；为避免 PDF 抽取噪声，网页中仅保留可渲染的核心符号和数值。
## 实验解读

- 评价重点:围绕 certified-control、碰撞避免、接触推理、inertial-estimation、实时控制,优先看真实机器人结果、扰动恢复、接触稳定性、控制误差、成功率、实时性和 sim-to-real 差距。
- System0 视角:实验结论不只看最终成功率,还要看闭环过程中是否能处理状态估计误差、执行延迟、碰撞/接触突变和外部扰动。
- 与上层系统的关系:如果方法能在快闭环内稳定工作,它可以作为 System1 技能或 System2 规划的底层执行保障;如果只在离线或慢速设定下成立,则还不能直接作为身体级快闭环模块。
## 批判性思考

### 优点

- 论文把 certified-control、碰撞避免、接触推理、inertial-estimation、实时控制 放到机器人身体级闭环里讨论,和只做高层语义推理的工作相比,更接近真实机器人部署时会遇到的状态、动力学和安全约束。
- 若论文同时给出公式、系统结构图、硬件实验或消融结果,这些证据可以帮助判断方法是否真的能支撑毫秒到百毫秒级反应。

### 局限

- 需要重点核查控制频率、推理延迟、传感延迟、失败恢复、硬件磨损、未见场景泛化和安全边界;这些因素决定它能否成为可靠的 System0 模块。
- 对学习型方法,还要区分训练环境中的鲁棒性和真实部署中的鲁棒性,尤其是接触丰富任务、动态运动任务和人形机器人任务。
## 论文定位

- 研究对象:Provably-Safe, Online System Identification。
- 关键词:certified-control、碰撞避免、接触推理、inertial-estimation、实时控制、机器人操作、鲁棒控制、safe-control。
- System0 关联:关注机器人身体级快反应链路中“状态/接触/局部感知 -> 约束/优化/策略 -> 动作”的短闭环,不把问题停留在语言规划或任务分解层。

## 速查卡片

> [!summary] Provably Safe Online
> - **论文**: https://www.roboticsproceedings.org/rss21/p121.pdf
> - **arXiv**: http://arxiv.org/abs/2504.21486v1
> - **arXiv HTML**: https://arxiv.org/html/2504.21486v1
