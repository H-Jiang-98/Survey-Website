---
title: "MOBIUS: A Multi-Modal Bipedal Robot that can Walk, Crawl, Climb, and Roll"
method_name: "MOBIUS"
authors: ["Alexander Schperberg"]
year: 2026
venue: "RSS"
tags: ["contact-reasoning", "robot-manipulation", "robust-control", "legged-locomotion", "reinforcement-learning", "safe-control", "loco-manipulation", "whole-body-control"]
image_source: "online"
arxiv_html: "https://arxiv.org/html/2511.01774v3"
---
# MOBIUS
## 一句话总结

> MOBIUS: A Multi-Modal Bipedal Robot that can Walk, Crawl, Climb, and Roll 主要落在 [[biped]]、[[compliance-control]]、[[接触推理]]、[[grasping]] 这一类问题上;从 System0 视角看,它的价值在于把身体状态、接触变化、局部感知或动力学约束接到机器人低层闭环里,让动作生成更稳定、安全、实时。
## 核心贡献

1. 围绕 **MOBIUS: A Multi-Modal Bipedal Robot that can Walk, Crawl, Climb, and Roll** 建立了一个与 biped、compliance-control、接触推理、grasping、足式运动、移动操作 相关的机器人问题设定,重点不是高层语义规划,而是身体级闭环中的状态变化、接触变化或控制约束。
2. 方法层面的关键价值在于把感知、动力学、约束优化、策略学习或硬件反馈组织成可执行的低层动作生成机制,适合放进 System0 的快反应模块中理解。
3. 对系统集成而言,这类工作补上了 System1 技能选择与电机级控制之间的一层:它负责把上层目标变成动态可行、可恢复、可安全过滤的动作。
4. 若论文包含真实机器人、接触任务或仿真到真实迁移,核心应看它是否报告了闭环稳定性、扰动恢复、硬件延迟、失败模式和跨场景泛化。
## 问题背景

这篇论文可以放在 System0 的“具身快系统”背景下阅读:机器人需要在很短时间内根据本体状态、接触、力/触觉、局部视觉或动力学预测做出动作,而不是等待慢速规划器重新推理整个任务。biped、compliance-control、接触推理、grasping、足式运动、移动操作、强化学习 相关问题的难点通常不在“知道要做什么”,而在动作执行过程中状态会快速偏离、接触会突然改变、约束会实时收紧,低层系统必须及时修正。

### System0 关联

精读时优先看三件事:第一,论文使用哪些快速反馈量,例如关节状态、接触状态、力觉、触觉、深度或局部视觉;第二,反馈如何进入 MPC、WBC、策略网络、安全过滤器或恢复控制器;第三,输出是否直接影响机器人下一步动作,而不是只服务离线分析或高层任务规划。
## 方法详解

### System0 视角解析

从 System0 角度看,这篇论文的方法部分应当围绕 biped、compliance-control、接触推理、grasping、足式运动、移动操作、强化学习、机器人操作 来读。核心问题是:系统如何从快速变化的机器人状态中构造控制输入,如何处理接触、碰撞、约束、模型误差或执行延迟,以及如何把这些信息变成下一时刻的动作。

如果方法里有优化问题,重点看目标函数、约束和求解频率;如果方法里有学习策略,重点看观测空间、动作空间、训练扰动和部署时的闭环频率;如果方法里有硬件或传感器,重点看传感链路是否真的进入控制回路。这样读可以把论文和 System2 语言规划、System1 技能编排区分开来。

### 关键公式与机制

#### 公式 1: [[动力学或策略机制]]

$$\boldsymbol{\Theta}_{\mathrm{action}}=\boldsymbol{\Theta}_{\mathrm{default}}+\left(\frac{1}{n_{\mathrm{actions}}}\sum_{i=1}^{n_{\mathrm{actions}}}\mathbf{A}_{t-i}\right)a_{\mathrm{scale}},$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 2: [[优化目标/约束]]

$$\mathcal{W}=[\mathbf{f}^{\top},\boldsymbol{\tau}^{\top}]^{\top}$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 3: [[优化目标/约束]]

$$o^{*}=\arg\min_{o\in\mathcal{O}_{\infty}}\lVert o-o_{q} \rVert,$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 4: [[动力学或策略机制]]

$$\sum_{i=1}^{N_{j}}\left| $\boldsymbol{\theta}_{i}-\boldsymbol{\theta}_{i}^{\mathrm{ref}}\right$ | $\left($ || $\mathbf{v}^{\mathrm{des}}$ ||<0.1\right)$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 5: [[动力学或策略机制]]

$$\sum_{i=1}^{N_{f}}\left| \Delta T_{air} $\mathbf{C}_{i}\right$ | $\left($ || $\mathbf{v}^{\mathrm{des}}$ ||>0.05\right)$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 6: [[动力学或策略机制]]

$$\sum_{i=1}^{N_{f}}\mathbf{C}_{i}\left(T_{\text{contact}}\geq 0.10\right)$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 7: [[动力学或策略机制]]

$$\sum_{i=1}^{N_{f}}\left| $\mathbf{p}_{i}\mathbf{C}_{i}\right$ | ^{2} $\left($ || $\mathbf{v}^{\mathrm{des}}$ ||>0.05\right)$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 8: [[动力学或策略机制]]

$$\mathbf{k}_{\text{kick}}\leftarrow\begin{bmatrix}\cos(\theta_{\text{kick}})\\ \sin(\theta_{\text{kick}})\end{bmatrix}$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 9: [[动力学或策略机制]]

$$\begin{bmatrix}p_{\text{angular}}^{x}\\ p_{\text{angular}}^{y}\end{bmatrix}=R(\theta)\begin{bmatrix}\bar{p}^{x}\\ \bar{p}^{y}\end{bmatrix}$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 10: [[动力学或策略机制]]

$$W_{\text{exp}}\sum_{i=0}^{N_{\text{grid}}-1}\sum_{j=0}^{N_{\text{grid}}-1}V_{i,j}-W_{\text{goal}}\left\lVert x_{T}-x_{\text{goal}}\right \rVert^{2}$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。
## 关键图表

### Figure 1: Pipeline / core system figure: MOBIUS: M ulti-modal O perations B ipedal I ntelligent U rban S cout

![Figure 1](https://arxiv.org/html/2511.01774v3/x1.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Pipeline / core system figure: MOBIUS: M ulti-modal O perations B ipedal I ntelligent U rban S cout”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 2: We show different planning approaches depending on the secondary mode of our robot

![Figure 2](https://arxiv.org/html/2511.01774v3/x14.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“We show different planning approaches depending on the secondary mode of our robot”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 3: A visual servo pipeline is used to track the handle bars of a slide before initiatin

![Figure 3](https://arxiv.org/html/2511.01774v3/x17.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“A visual servo pipeline is used to track the handle bars of a slide before initiatin”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### 论文表格

#### Table 1: TABLE I: MIQCP Constraints (1)-(20)

| Visiting Grid Constraints | |
| --- | --- |
| (1) | z  (t, i, j) ∈ {0, 1 } z(t,i,j)\in\{0,1\} |
| (2) | z sum  (i, j) = ∑ t = 0 T - 1 z  (t, i, j), ∀ i, j ∈ {0, 1, ..., N grid - 1 } z_{ $\text{sum}}(i,j)=\sum_{t=0}^{T-1}z(t,i,j),\quad\forall i,j\in\{0,1,\dots,N_{\text{grid}}-1\}$ |
| (3) | V  (i, j) ≤ z sum  (i, j), ∀ i, j V(i,j)\leq z_{ $\text{sum}}(i,j),\quad\forall i,j$ |
| (4) | V  (i, j) ≤ T  V  (i, j), ∀ i, j V(i,j)\leq T\cdot V(i,j),\quad\forall i,j |
| (5) | x  (t, 0) - i ≥ - M  (1 - z  (t, i, j)) x(t,0)-i\geq-M(1-z(t,i,j)) |
| (6) | x  (t, 1) - j ≥ - M  (1 - z  (t, i, j)) x(t,1)-j\geq-M(1-z(t,i,j)) |
| Propagation Update | |
| (7) | x  (t + 1) = x  (t) + d  (t) x(t+1)=x(t)+d(t) |
| Mode Types Constraints | |
| (8) | m 1  (t) = 1 ⟹ d  (t) ∈ {- 1, 1 } m_{1}(t)=1\implies d(t)\in\{-1,1\} |
| (9) | m 2  (t) = 1 ⟹ d  (t) ∈ {- 2, - 1, 1, 2 } m_{2}(t)=1\implies d(t)\in\{-2,-1,1,2\} |
| (10) | m 3  (t) = 1 ⟹ d  (t) ∈ {- 3, 3 } m_{3}(t)=1\implies d(t)\in\{-3,3\} |
| (11) | m 1  (t) + m 2  (t) + m 3  (t) = 1 m_{1}(t)+m_{2}(t)+m_{3}(t)=1 |
| Terrain Type Constraints | |
| (12) | d circle = (x  (t, 0) - x center) 2 + (x  (t, 1) - y center) 2 d_{ $\text{circle}}=(x(t,0)-x_{\text{center}})^{2}+(x(t,1)-y_{\text{center}})^{2}$ |
| (13) | d circle ≤ r circle 2 + M  (1 - b circle k  (t)) d_{ $\text{circle}}\leq r_{\text{circle}}^{2}+M(1-b_{\text{circle}}^{k}(t))$ |
| (14) | $\text{circle}}\geq r_{\text{circle}}^{2}+\epsilon-Mb_{\text{circle}}^{k}(t)$ |
| (15) | m k  (t) ≥ b circle k  (t), b circle k ∈ {0, 1 } m_{k}(t)\geq b_{ $\text{circle}}^{k}(t),\quad b_{\text{circle}}^{k}\in\{0,1\}$ |
| Obstacle Constraints | |
| (16) | x  (t, 0) ≥ x min + M  (1 - b 1 rect  (t)) x(t,0)\geq x_{ $\text{min}}+M(1-b_{1}^{\text{rect}}(t))$ |
| (17) | x  (t, 0) ≤ x max - M  (1 - b 2 rect  (t)) x(t,0)\leq x_{ $\text{max}}-M(1-b_{2}^{\text{rect}}(t))$ |
| (18) | x  (t, 1) ≥ y min + M  (1 - b 3 rect  (t)) x(t,1)\geq y_{ $\text{min}}+M(1-b_{3}^{\text{rect}}(t))$ |
| (19) | x  (t, 1) ≤ y max - M  (1 - b 4 rect  (t)) x(t,1)\leq y_{ $\text{max}}-M(1-b_{4}^{\text{rect}}(t))$ |
| (20) | b 1 rect  (t) + b 2 rect  (t) + b 3 rect  (t) + b 4 rect  (t) ≤ 3 b_{1}^{ $\text{rect}}(t)+b_{2}^{\text{rect}}(t)+b_{3}^{\text{rect}}(t)+b_{4}^{\text{rect}}(t)\leq 3$ |

**说明**: TABLE I: MIQCP Constraints (1)-(20)

#### Table 2: TABLE II: Maximum Locomotion Velocities

| Mode | Biped Mode | Crawling Mode | Rolling Mode |
| --- | --- | --- | --- |
| Max velocity (m/s) | 0.1 | 0.4 | 0.7 |

**说明**: TABLE II: Maximum Locomotion Velocities

#### Table 3: TABLE III: Energy and Cost of Transport per Mode

| Mode | Energy (J/cell) | CoT |
| --- | --- | --- |
| Biped | 26.89 | 0.548 |
| Crawl | 5.11 | 0.104 |
| Roll | 21.96 | 0.448 |

**说明**: TABLE III: Energy and Cost of Transport per Mode

#### Table 4: TABLE IV: Maximum supporting force of the spine gripper on a slippery aluminum bar in Fig. 1 e.

| Direction | Maximum Supporting Force |
| --- | --- |
| x x -axis | 23.6 23.6 N ± 4.1 \pm 4.1 N |
| z z -axis | 124.2 124.2 N ± 24.5 \pm 24.5 N |

**说明**: TABLE IV: Maximum supporting force of the spine gripper on a slippery aluminum bar in Fig. 1 e.

#### Table 5: TABLE V: Hyperparameters and settings for RL training

| Parameter Name | Value |
| --- | --- |
| Mujoco solver time step | 0.005 |
| Linear search iteration | 3 |
| Constrain iteration | 3 |
| Network parameters | (128,128,128,128) |
| Batch size | 256 |
| Learning rate | 3e-4 |
| Discounting | 0.97 |
| Entropy cost | 1e-2 |
| Minibatches | 32 |
| Update per batch | 4 |
| Num of time steps (flat) | 300,000,000 |
| Num of time steps (rough) | 100,000,000 |
| Max time step in episode | 2000 |
| Max motor RPM | 50 (rpm) |
| Action scale (a scale a_{ $\mathrm{scale}})$ | 0.3 |

**说明**: TABLE V: Hyperparameters and settings for RL training

#### Table 6: TABLE VI: Reward and penalty formulation for the RL planning and control pipeline.

| Reward/Penalty Description | Reward Equation |
| --- | --- |
| Penalize Vertical Velocity (R v z R_{v_{z}}) | v z 2 {v_{z}}^{2} |
| Penalize Angular Velocity R x y R $\omega_{xy}})$ | $\boldsymbol{\omega}_{xy}\lVert \rVert^{2}$ |
| Penalize Joint Torques R R $\tau})$ | $\sum_{i=1}^{N_{j}}\lVert \boldsymbol{\tau}_{i} \rVert\lVert \boldsymbol{\omega}_{i} \rVert$ |
| Penalize Change in Actions (R a R_{a}) | ∑ i = 1 N j (a t i - a t - 1 i)  $\sum_{i=1}^{N_{j}}(\mathbf{a}_{t}^{i}-\mathbf{a}_{t-1}^{i})$ |
| Reward Linear Velocity (R v x  y R_{v_{xy}}) | $\exp\left(-\frac{\lVert \rVert\mathbf{v}_{xy}^{des}-\mathbf{v}_{xy}\lVert \rVert^{2}}{\sigma}\right)$ |
| Reward Angular Velocity R z R $\omega_{z}})$ | $\exp\left(-\frac{\lVert \rVert\boldsymbol{\omega}_{z}^{des}-\boldsymbol{\omega}_{z}\lVert \rVert^{2}}{\sigma}\right)$ |
| Reward Standing Still (R stand R_{ $\text{stand}})$ | $\sum_{i=1}^{N_{j}}\left\lVert \boldsymbol{\theta}_{i}-\boldsymbol{\theta}_{i}^{\mathrm{ref}}\right \rVert\left(\lVert \rVert\mathbf{v}^{\mathrm{des}}\lVert \rVert<0.1\right)$ |
| Reward Air Time (R air R_{ $\text{air}})$ | $\sum_{i=1}^{N_{f}}\left\lVert \Delta T_{air}\mathbf{C}_{i}\right \rVert\left(\lVert \rVert\mathbf{v}^{\mathrm{des}}\lVert \rVert>0.05\right)$ |
| Reward Contact (R contact R_{ $\text{contact}})$ | ∑ i = 1 N f C i  (T contact ≥ 0.10)  $\sum_{i=1}^{N_{f}}\mathbf{C}_{i}\left(T_{\text{contact}}\geq 0.10\right)$ |
| Penalize Foot Slip (R slip R_{ $\text{slip}})$ | $\sum_{i=1}^{N_{f}}\left\lVert \mathbf{p}_{i}\mathbf{C}_{i}\right \rVert^{2}\left(\lVert \rVert\mathbf{v}^{\mathrm{des}}\lVert \rVert>0.05\right)$ |
| Penalize Termination (R term R_{ $\text{term}})$ | Done t  e  r  m  (t < 500)  $\text{Done}_{term}(t<500)$ |

**说明**: TABLE VI: Reward and penalty formulation for the RL planning and control pipeline.

#### Table 7: TABLE VII: Domain randomization parameters.

| Description | Symbol | Range of Values |
| --- | --- | --- |
| Orientation (x, y, z) | $\boldsymbol{\theta}$ | $\boldsymbol{\theta}\sim\mathcal{U}(\boldsymbol{-\pi},\boldsymbol{\pi})^{\circ}$ |
| Angular velocity (yaw) | $\omega_{\text{yaw}}$ | $\omega_{\text{yaw}}\sim\mathcal{U}(-0.15,0.15)\text{rad/s}$ |
| Linear velocity (x, y, z) | v linear  $\mathbf{v}_{\text{linear}}$ | v linear ∼ U  (- 0.05, 0.05)  m/s  $\mathbf{v}_{\text{linear}}\sim\mathcal{U}(\mathbf{-0.05},\mathbf{0.05})\text{m/s}$ |
| Initial starting state | default $\delta\boldsymbol{\Theta}_{\text{default}}$ | default ∼ U . . ∘ $\delta\boldsymbol{\Theta}_{\text{default}}\sim\mathcal{U}(\mathbf{0.01},\mathbf{0.01})^{\circ}$ |
| Selection of Past Observation | N delay N_{ $\mathrm{delay}}$ | N delay ∼ U  (1, 3) N_{ $\mathrm{delay}}\sim\mathcal{U}(1,3)$ |
| Prior Selection Chance | p p | 0.3 |
| Kick Interval | P int P_{ $\mathrm{int}}$ | P int ∼ U  (10, 30) P_{ $\mathrm{int}}\sim\mathcal{U}(10,30)$ |
| Kick angle | $\theta_{\text{kick}}$ | $\theta_{\text{kick}}\sim\mathcal{U}(0,2\pi)^{\circ}$ |
| Body mass | m body m_{ $\text{body}}$ | m body ∼ U  (- 5.0, 5.0)  kg m_{ $\text{body}}\sim\mathcal{U}(-5.0,5.0)\text{kg}$ |
| Body location (x, y, z) | l body  $\mathbf{l}_{\text{body}}$ | l body ∼ U  (- 0.1, 0.1)  m  $\mathbf{l}_{\text{body}}\sim\mathcal{U}(\mathbf{-0.1},\mathbf{0.1})\text{m}$ |
| Surface friction | $\delta\mu$ | $\delta\mu\sim\mathcal{U}(-0.3,0.5)$ |
| Actuator gain | $\delta\mathbf{k}_{\text{act}}$ | k act ∼ U  (- 20, 20)  $\mathbf{k}_{\text{act}}\sim\mathcal{U}(\mathbf{-20},\mathbf{20})$ |
| Actuator bias | $\delta\mathbf{b}_{\text{act}}$ | b act ∼ U  (- 20, 20)  $\mathbf{b}_{\text{act}}\sim\mathcal{U}(\mathbf{-20},\mathbf{20})$ |

**说明**: TABLE VII: Domain randomization parameters.

#### Table 8: TABLE VIII: MIQCP Constraints (1)-(20)

| Visiting Grid Constraints | |
| --- | --- |
| (1) | z  (t, i, j) ∈ {0, 1 } z(t,i,j)\in\{0,1\} |
| (2) | z sum  (i, j) = ∑ t = 0 T - 1 z  (t, i, j), ∀ i, j ∈ {0, 1, ..., N grid - 1 } z_{ $\text{sum}}(i,j)=\sum_{t=0}^{T-1}z(t,i,j),\quad\forall i,j\in\{0,1,\dots,N_{\text{grid}}-1\}$ |
| (3) | V  (i, j) ≤ z sum  (i, j), ∀ i, j V(i,j)\leq z_{ $\text{sum}}(i,j),\quad\forall i,j$ |
| (4) | V  (i, j) ≤ T  V  (i, j), ∀ i, j V(i,j)\leq T\cdot V(i,j),\quad\forall i,j |
| (5) | x  (t, 0) - i ≥ - M  (1 - z  (t, i, j)) x(t,0)-i\geq-M(1-z(t,i,j)) |
| (6) | x  (t, 1) - j ≥ - M  (1 - z  (t, i, j)) x(t,1)-j\geq-M(1-z(t,i,j)) |
| Propagation Update | |
| (7) | x  (t + 1) = x  (t) + d  (t) x(t+1)=x(t)+d(t) |
| Mode Types Constraints | |
| (8) | m 1  (t) = 1 ⟹ d  (t) ∈ {- 1, 1 } m_{1}(t)=1\implies d(t)\in\{-1,1\} |
| (9) | m 2  (t) = 1 ⟹ d  (t) ∈ {- 2, - 1, 1, 2 } m_{2}(t)=1\implies d(t)\in\{-2,-1,1,2\} |
| (10) | m 3  (t) = 1 ⟹ d  (t) ∈ {- 3, 3 } m_{3}(t)=1\implies d(t)\in\{-3,3\} |
| (11) | m 1  (t) + m 2  (t) + m 3  (t) = 1 m_{1}(t)+m_{2}(t)+m_{3}(t)=1 |
| Terrain Type Constraints | |
| (12) | d circle = (x  (t, 0) - x center) 2 + (x  (t, 1) - y center) 2 d_{ $\text{circle}}=(x(t,0)-x_{\text{center}})^{2}+(x(t,1)-y_{\text{center}})^{2}$ |
| (13) | d circle ≤ r circle 2 + M  (1 - b circle k  (t)) d_{ $\text{circle}}\leq r_{\text{circle}}^{2}+M(1-b_{\text{circle}}^{k}(t))$ |
| (14) | $\text{circle}}\geq r_{\text{circle}}^{2}+\epsilon-Mb_{\text{circle}}^{k}(t)$ |
| (15) | m k  (t) ≥ b circle k  (t), b circle k ∈ {0, 1 } m_{k}(t)\geq b_{ $\text{circle}}^{k}(t),\quad b_{\text{circle}}^{k}\in\{0,1\}$ |
| Obstacle Constraints | |
| (16) | x  (t, 0) ≥ x min + M  (1 - b 1 rect  (t)) x(t,0)\geq x_{ $\text{min}}+M(1-b_{1}^{\text{rect}}(t))$ |
| (17) | x  (t, 0) ≤ x max - M  (1 - b 2 rect  (t)) x(t,0)\leq x_{ $\text{max}}-M(1-b_{2}^{\text{rect}}(t))$ |
| (18) | x  (t, 1) ≥ y min + M  (1 - b 3 rect  (t)) x(t,1)\geq y_{ $\text{min}}+M(1-b_{3}^{\text{rect}}(t))$ |
| (19) | x  (t, 1) ≤ y max - M  (1 - b 4 rect  (t)) x(t,1)\leq y_{ $\text{max}}-M(1-b_{4}^{\text{rect}}(t))$ |
| (20) | b 1 rect  (t) + b 2 rect  (t) + b 3 rect  (t) + b 4 rect  (t) ≤ 3 b_{1}^{ $\text{rect}}(t)+b_{2}^{\text{rect}}(t)+b_{3}^{\text{rect}}(t)+b_{4}^{\text{rect}}(t)\leq 3$ |

**说明**: TABLE VIII: MIQCP Constraints (1)-(20)

#### Table 9: TABLE IX: Top table shows the optimal parameter selection, from a combination of parameters (324 in total) shown in Tab

#### Table 10: TABLE X: Energy comparison per meter of travel. COM heights are 0.35 m (biped) and 0.20 m (crawling). Biped incurs high

| Locomotion Type | KE (J) | PE (J) | Energy per 1 m (J/m) |
| --- | --- | --- | --- |
| Biped | 1.35 | 102.82 | 104.17 |
| Crawling | 6.00 | 58.86 | 64.86 |
| Rolling | 70.00 | 0.00 | 70.00 |
| Pull-up | 3.00 | 58.86 | 61.86 |
| Climbing | 0.60 | 294.30 | 294.90 |

**说明**: TABLE X: Energy comparison per meter of travel. COM heights are 0.35 m (biped) and 0.20 m (crawling). Biped incurs higher potential energy costs due to its elevated posture, while crawling distributes effort across more limbs. Rolling takes a lot of initial effort as it has to go kick back with as much force as possible to roll backwards.
## 实验解读

- 评价重点:围绕 biped、compliance-control、接触推理、grasping、足式运动,优先看真实机器人结果、扰动恢复、接触稳定性、控制误差、成功率、实时性和 sim-to-real 差距。
- System0 视角:实验结论不只看最终成功率,还要看闭环过程中是否能处理状态估计误差、执行延迟、碰撞/接触突变和外部扰动。
- 与上层系统的关系:如果方法能在快闭环内稳定工作,它可以作为 System1 技能或 System2 规划的底层执行保障;如果只在离线或慢速设定下成立,则还不能直接作为身体级快闭环模块。
## 批判性思考

### 优点

- 论文把 biped、compliance-control、接触推理、grasping、足式运动 放到机器人身体级闭环里讨论,和只做高层语义推理的工作相比,更接近真实机器人部署时会遇到的状态、动力学和安全约束。
- 若论文同时给出公式、系统结构图、硬件实验或消融结果,这些证据可以帮助判断方法是否真的能支撑毫秒到百毫秒级反应。

### 局限

- 需要重点核查控制频率、推理延迟、传感延迟、失败恢复、硬件磨损、未见场景泛化和安全边界;这些因素决定它能否成为可靠的 System0 模块。
- 对学习型方法,还要区分训练环境中的鲁棒性和真实部署中的鲁棒性,尤其是接触丰富任务、动态运动任务和人形机器人任务。
## 论文定位

- 研究对象:MOBIUS: A Multi-Modal Bipedal Robot that can Walk, Crawl, Climb, and Roll。
- 关键词:biped、compliance-control、接触推理、grasping、足式运动、移动操作、强化学习、机器人操作、鲁棒控制、safe-control。
- System0 关联:关注机器人身体级快反应链路中“状态/接触/局部感知 -> 约束/优化/策略 -> 动作”的短闭环,不把问题停留在语言规划或任务分解层。

## 速查卡片

> [!summary] MOBIUS
> - **论文**: https://www.roboticsproceedings.org/rss22/p024.pdf
> - **arXiv**: http://arxiv.org/abs/2511.01774v3
> - **arXiv HTML**: https://arxiv.org/html/2511.01774v3
