---
title: "Bridging the Sim-to-Real Gap for Athletic Loco-Manipulation"
method_name: "Bridging Sim Gap"
authors: ["Nolan Fey"]
year: 2025
venue: "RSS"
tags: ["robot-manipulation", "robust-control", "legged-locomotion", "sim-to-real", "agile-locomotion", "loco-manipulation"]
image_source: "online"
arxiv_html: "https://arxiv.org/html/2502.10894v1"
---
# Bridging Sim Gap
## 一句话总结

> Bridging the Sim-to-Real Gap for Athletic Loco-Manipulation 主要落在 [[actuator-modeling]]、[[agile-locomotion]]、[[足式运动]]、[[移动操作]] 这一类问题上;从 System0 视角看,它的价值在于把身体状态、接触变化、局部感知或动力学约束接到机器人低层闭环里,让动作生成更稳定、安全、实时。
## 核心贡献

1. 围绕 **Bridging the Sim-to-Real Gap for Athletic Loco-Manipulation** 建立了一个与 actuator-modeling、agile-locomotion、足式运动、移动操作、mobile-manipulation、机器人操作 相关的机器人问题设定,重点不是高层语义规划,而是身体级闭环中的状态变化、接触变化或控制约束。
2. 方法层面的关键价值在于把感知、动力学、约束优化、策略学习或硬件反馈组织成可执行的低层动作生成机制,适合放进 System0 的快反应模块中理解。
3. 对系统集成而言,这类工作补上了 System1 技能选择与电机级控制之间的一层:它负责把上层目标变成动态可行、可恢复、可安全过滤的动作。
4. 若论文包含真实机器人、接触任务或仿真到真实迁移,核心应看它是否报告了闭环稳定性、扰动恢复、硬件延迟、失败模式和跨场景泛化。
## 问题背景

这篇论文可以放在 System0 的“具身快系统”背景下阅读:机器人需要在很短时间内根据本体状态、接触、力/触觉、局部视觉或动力学预测做出动作,而不是等待慢速规划器重新推理整个任务。actuator-modeling、agile-locomotion、足式运动、移动操作、mobile-manipulation、机器人操作、鲁棒控制 相关问题的难点通常不在“知道要做什么”,而在动作执行过程中状态会快速偏离、接触会突然改变、约束会实时收紧,低层系统必须及时修正。

### System0 关联

精读时优先看三件事:第一,论文使用哪些快速反馈量,例如关节状态、接触状态、力觉、触觉、深度或局部视觉;第二,反馈如何进入 MPC、WBC、策略网络、安全过滤器或恢复控制器;第三,输出是否直接影响机器人下一步动作,而不是只服务离线分析或高层任务规划。
## 方法详解

### System0 视角解析

从 System0 角度看,这篇论文的方法部分应当围绕 actuator-modeling、agile-locomotion、足式运动、移动操作、mobile-manipulation、机器人操作、鲁棒控制、仿真到真实迁移 来读。核心问题是:系统如何从快速变化的机器人状态中构造控制输入,如何处理接触、碰撞、约束、模型误差或执行延迟,以及如何把这些信息变成下一时刻的动作。

如果方法里有优化问题,重点看目标函数、约束和求解频率;如果方法里有学习策略,重点看观测空间、动作空间、训练扰动和部署时的闭环频率;如果方法里有硬件或传感器,重点看传感链路是否真的进入控制回路。这样读可以把论文和 System2 语言规划、System1 技能编排区分开来。

### 关键公式与机制

#### 公式 1: [[动力学或策略机制]]

$$\lvert\boldsymbol{\tau}\rvert^{\top}\lvert\dot{\mathbf{q}}\rvert\leq P_{\mathrm{max}},$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 2: [[优化目标/约束]]

$$\geq-\boldsymbol{\tau}_{\max}\left(1+\max\left(\min\left(\frac{\dot{\mathbf{q}}}{\dot{\mathbf{q}}_{\max}},0\right),-1\right)\right),$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 3: [[优化目标/约束]]

$$\leq\boldsymbol{\tau}_{\max}\left(1-\max\left(\min\left(\frac{\dot{\mathbf{q}}}{\dot{\mathbf{q}}_{\max}},1\right),0\right)\right).$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 4: [[动力学或策略机制]]

$$=-5\left\lvert v^{\mathrm{cmd}}_{x,t}-v_{x,t}\right\rvert^{2}_{2}-\delta y_{t}-\delta\theta^{\mathrm{yaw}}_{t}-\sum_{i}^{\{4,5,6\}}q_{i}^{2},$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 5: [[优化目标/约束]]

$$\min_{\pi_{\mathrm{UAN}}}|| f_{ $\mathrm{real}}\big{(}\mathbf{s},\boldsymbol{\tau}\big{)}-f_{\mathrm{sim}}\left(\mathbf{s},\boldsymbol{\tau}+\pi_{\mathrm{UAN}}\left(\mathbf{e}\right)\right)$ ||.$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 6: [[优化目标/约束]]

$$F^{\mathrm{pull}}_{t}=\max\left(1000\left\lvert d-l\right\rvert_{2}^{2}-\left\lvert\dot{d}\right\rvert^{2}_{2},0\right),$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 7: [[动力学或策略机制]]

$$\{\left(\mathbf{s}_{t},\boldsymbol{\tau}_{t},\mathbf{s}_{t+1}\right)_{i}\}_{i=0}^{N}$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 8: [[动力学或策略机制]]

$$=5\sum_{i}^{3}\frac{1}{3}{\mathrm{exp}}\left(-2\left\lvert p^{\mathrm{cmd}}_{i,t}-p_{i,t}\right\rvert\right),$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 9: [[动力学或策略机制]]

$$=5\sum_{i}^{3}\frac{1}{3}{\mathrm{exp}}\left(-2\left\lvert p^{\mathrm{cmd}}_{i,t}-p_{i,t}\right\rvert\right)$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 10: [[动力学或策略机制]]

$$\mathbf{v}_{t}^{\mathrm{cmd}}=\left[v^{\mathrm{cmd}}_{x,t},v^{\mathrm{cmd}}_{y,t},\omega^{\mathrm{cmd}}_{z,t}\right]$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。
## 关键图表

### Figure 1: Pipeline / core system figure: Sim-to-real transfer of athletic loco-manipulation. We reduce the sim-to-real gap for

![Figure 1](https://arxiv.org/html/2502.10894v1/extracted/6206771/throw_sequence_transparent.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Pipeline / core system figure: Sim-to-real transfer of athletic loco-manipulation. We reduce the sim-to-real gap for”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 2: Sim-to-real transfer of athletic loco-manipulation. We reduce the sim-to-real gap for

![Figure 2](https://arxiv.org/html/2502.10894v1/extracted/6206771/lift_sequence_transparent.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Sim-to-real transfer of athletic loco-manipulation. We reduce the sim-to-real gap for”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 3: Unsupervised Actuator Network (UAN) approach for real-to-sim-to-real. Our training pi

![Figure 3](https://arxiv.org/html/2502.10894v1/x1.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Unsupervised Actuator Network (UAN) approach for real-to-sim-to-real. Our training pi”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### 论文表格

#### Table 1: TABLE I: PPO Hyperparameters

| Hyperparameter | UAN Value | Pre-train Value | Fine-tune Value |
| --- | --- | --- | --- |
| Discount Factor | 0.995 | 0.99 | 0.99 |
| GAE Parameter | 0.95 | 0.95 | 0.95 |
| Entropy Coefficient | 0.0 | 0.01 | 0.0 |
| Actor Learning Rate | Adaptive | Adaptive | Adaptive |
| Critic Learning Rate | 5.e-4 | 5.e-4 | 5.e-4 |
| KL Threshold | 0.01 | 0.01 | 0.01 |
| Horizon | 96 | 24 | 24 |
| Number of Environments | 4096 | 4096 | 4096 |
| Actor Minibatch Size | 98304 | 24576 | 24576 |
| Critic Minibatch Size | 393216 | 98304 | 98304 |
| # of Mini Epochs | 5 | 5 | 5 |
| Optimizer | AdamW | AdamW | AdamW |
| Weight Decay | 0.01 | 0.01 | 0.01 |

**说明**: TABLE I: PPO Hyperparameters

#### Table 2: TABLE II: WBC & Fine-tune Domain Randomization

| Parameter | Min | Max |
| --- | --- | --- |
| Terrain Friction | 0.5 | 4.0 |
| Terrain Restitution | 0.0 | 0.5 |
| Terrain Roughness [cm times absent centimeter  $\text{\,}\mathrm{cm} times ]$ | 0 | 2.5 |
| Leg Joint Stiffness Scale | 0.9 | 1.1 |
| Leg Joint Damping Scale | 0.5 | 1.5 |
| Leg Stall Torque Scale | 0.9 | 1.1 |
| Link Mass Scale | 0.5 | 1.5 |
| Link Center of Mass Offsets [cm times absent centimeter  $\text{\,}\mathrm{cm} times ]$ | -2 | 2 |
| Encoder Offset [rad times absent radian  $\text{\,}\mathrm{rad} times ]$ | -0.05 | -0.05 |
| Policy Lag Timesteps (5 ms times 5 millisecond 5 $\text{\,}\mathrm{ms} 5 times)$ | 0 | 6 |

**说明**: TABLE II: WBC & Fine-tune Domain Randomization

#### Table 3: TABLE III: WBC Rewards

| Reward Component | Term | Scale |
| --- | --- | --- |
| EE Pose Tracking | ∑ i 3 1 3 exp (- 2 \| p i, t cmd - p i, t \|) i 3 1 3 exp 2 p cmd i t p i t  $\sum_{i}^{3}\frac{1}{3}{\mathrm{exp}}\left(-2\left\lvert p^{\mathrm{cmd}}_{i,t}-p_{i,t}\right\rvert\right) ∑ 3 divide 1 3 (- 2 \lVert, -, \rVert)$ | 5.0 |
| Linear Velocity Tracking | exp (- 4 \| [v x, t cmd, v y, t cmd ] - [v x, t, v y, t ] \| 2 2) exp 4 v cmd x t v cmd y t v x t v y t 2 2 { $\mathrm{exp}}\left(-4\left\lvert\left[v^{\mathrm{cmd}}_{x,t},v^{\mathrm{cmd}}_{y,t}\right]-\left[v_{x,t},v_{y,t}\right]\right\rvert^{2}_{2}\right) (- 4 \lVert [,,, ] - [,,, ] \rVert 2 2)$ | 2.0 |
| Angular Velocity Tracking | $\mathrm{exp}}\left(-4\left(\omega^{\mathrm{cmd}}_{z,t}-\omega_{z,t}\right)^{2}\right)$ | 1.0 |
| Gait | ∑ i ∈ {FR, FL, RR, RL } {∼ c i } 1 {p z, t i < 0.043 }  $\sum_{i\in\{\mathrm{FR,FL,RR,RL}\}}\{\sim c_{i}\}~{}\mathbf{1}\{p^{i}_{z,t}<0.043\} ∑ ∈ {,,, } {∼ } {, < 0.043 }$ | -0.5 |
| No Slip | ∑ i ∈ {FR, FL, RR, RL } {c i } exp  (- 0.1 \| v t i \| 2 2) i FR FL RR RL c i 0.1 v i t 2 2  $\sum_{i\in\{{\mathrm{FR},FL,RR,RL}\}}\{c_{i}\}~{}\exp\left(-0.1\left\lvert\boldsymbol{v}^{i}_{t}\right\rvert^{2}_{2}\right) ∑ ∈ {,,, } {} (- 0.1 \lVert \rVert 2 2)$ | -0.5 |
| Foot Clearance | ∑ i ∈ {FR, FL, RR, RL } {∼ c i } \| p z, t cmd, i - p z, t i \| 2 2  $\sum_{i\in\{{\mathrm{FR},FL,RR,RL}\}}\{\sim c_{i}\}~{}\left\lvert p^{{\mathrm{cmd}},i}_{z,t}-p^{i}_{z,t}\right\rvert^{2}_{2} ∑ ∈ {,,, } {∼ } \lVert,, -, \rVert 2 2$ | -40.0 |
| Mechanical Power | $\left\lvert\boldsymbol{\tau}_{t}\cdot\dot{\boldsymbol{q}}_{t}\right\rvert \lVert ⋅ \rVert$ | -0.0001 |
| Action Smoothness | \| a t - a t - 1 \| 2 2 + 1 2 \| a t - 2 a t - 1 + a t - 2 \| 2 2 a t a t 1 2 2 1 2 a t 2 a t 1 a t 2 2 2  $\left\lvert\mathbf{a}_{t}-\mathbf{a}_{t-1}\right\rvert^{2}_{2}+\frac{1}{2}\left\lvert\mathbf{a}_{t}-2\mathbf{a}_{t-1}+\mathbf{a}_{t-2}\right\rvert^{2}_{2} \lVert - - 1 \rVert 2 2 + divide 1 2 \lVert - 2 - 1 + - 2 \rVert 2 2$ | -0.05 |
| Linear Velocity Z | \| v z, t \| 2 v z t 2  $\left\lvert v_{z,t}\right\rvert^{2} \lVert, \rVert 2$ | -2.0 |
| Angular Velocity XY | $\left\lvert\left[\omega_{x,t},\omega_{y,t}\right]\right\rvert^{2}_{2} \lVert [,,, ] \rVert 2 2$ | -0.05 |
| Joint Positions | \| q t default - q t \| q default t q t  $\left\lvert\boldsymbol{q}^{\mathrm{default}}_{t}-\boldsymbol{q}_{t}\right\rvert \lVert - \rVert$ | -0.25 |
| Collision | {\| F t arm \| 2 2 > 0.1 or \| F t leg \| 2 2 > 0.1 } F arm t 2 2 0.1 or F leg t 2 2 0.1 \{ $\left\lvert F^{\mathrm{arm}}_{t}\right\rvert^{2}_{2}>0.1\text{or }\left\lvert F^{\mathrm{leg}}_{t}\right\rvert^{2}_{2}>0.1\} {\lVert \rVert 2 2 > 0.1 or \lVert \rVert 2 2 > 0.1 }$ | -5.0 |
| Joint Position Limits | ∑ i - min  (q i, t - q i, t min, 0) + max  (q i, t - q i, t max, 0) i q i t q min i t 0 q i t q max i t 0  $\sum_{i}-\min(q_{i,t}-q^{\mathrm{min}}_{i,t},0)+\max(q_{i,t}-q^{\mathrm{max}}_{i,t},0) ∑ -$ | -10.0 |
| Contact Force | ∑ i ∈ {FR, FL, RR, RL } \| F t i \| 2 2 i FR FL RR RL F i t 2 2  $\sum_{i\in\{{\mathrm{FR},FL,RR,RL}\}}\left\lvert F^{i}_{t}\right\rvert^{2}_{2} ∑ ∈ {,,, } \lVert \rVert 2 2$ | -0.000004 |

**说明**: TABLE III: WBC Rewards

#### Table 4: TABLE IV: Unsupervised Actuator Net Rewards

| Reward Component | Term | Scale |
| --- | --- | --- |
| Joint Positions (L1) | \| q t real - q t sim \| q real t q sim t  $\left\lvert\boldsymbol{q}^{\mathrm{real}}_{t}-\boldsymbol{q}^{\mathrm{sim}}_{t}\right\rvert \lVert - \rVert$ | -1.5 |
| Joint Positions (Relaxed) | exp (- 100 \| q t real - q t sim \| 2 2) exp 100 q real t q sim t 2 2 { $\mathrm{exp}}\left(-100\left\lvert\boldsymbol{q}^{\mathrm{real}}_{t}-\boldsymbol{q}^{\mathrm{sim}}_{t}\right\rvert^{2}_{2}\right) (- 100 \lVert - \rVert 2 2)$ | 4.0 |
| Joint Positions (Moderate) | exp (- 300 \| q t real - q t sim \| 2 2) exp 300 q real t q sim t 2 2 { $\mathrm{exp}}\left(-300\left\lvert\boldsymbol{q}^{\mathrm{real}}_{t}-\boldsymbol{q}^{\mathrm{sim}}_{t}\right\rvert^{2}_{2}\right) (- 300 \lVert - \rVert 2 2)$ | 4.0 |
| Joint Positions (Strict) | exp (- 1000 \| q t real - q t sim \| 2 2) exp 1000 q real t q sim t 2 2 { $\mathrm{exp}}\left(-1000\left\lvert\boldsymbol{q}^{\mathrm{real}}_{t}-\boldsymbol{q}^{\mathrm{sim}}_{t}\right\rvert^{2}_{2}\right) (- 1000 \lVert - \rVert 2 2)$ | 5.0 |
| Action Smoothness | exp (- 0.5 \| a t - a t - 1 \|) exp 0.5 a t a t 1 { $\mathrm{exp}}\left(-0.5\left\lvert\mathbf{a}_{t}-\mathbf{a}_{t-1}\right\rvert\right) (- 0.5 \lVert - - 1 \rVert)$ | 0.5 |

**说明**: TABLE IV: Unsupervised Actuator Net Rewards
## 实验解读

- 评价重点:围绕 actuator-modeling、agile-locomotion、足式运动、移动操作、mobile-manipulation,优先看真实机器人结果、扰动恢复、接触稳定性、控制误差、成功率、实时性和 sim-to-real 差距。
- System0 视角:实验结论不只看最终成功率,还要看闭环过程中是否能处理状态估计误差、执行延迟、碰撞/接触突变和外部扰动。
- 与上层系统的关系:如果方法能在快闭环内稳定工作,它可以作为 System1 技能或 System2 规划的底层执行保障;如果只在离线或慢速设定下成立,则还不能直接作为身体级快闭环模块。
## 批判性思考

### 优点

- 论文把 actuator-modeling、agile-locomotion、足式运动、移动操作、mobile-manipulation 放到机器人身体级闭环里讨论,和只做高层语义推理的工作相比,更接近真实机器人部署时会遇到的状态、动力学和安全约束。
- 若论文同时给出公式、系统结构图、硬件实验或消融结果,这些证据可以帮助判断方法是否真的能支撑毫秒到百毫秒级反应。

### 局限

- 需要重点核查控制频率、推理延迟、传感延迟、失败恢复、硬件磨损、未见场景泛化和安全边界;这些因素决定它能否成为可靠的 System0 模块。
- 对学习型方法,还要区分训练环境中的鲁棒性和真实部署中的鲁棒性,尤其是接触丰富任务、动态运动任务和人形机器人任务。
## 论文定位

- 研究对象:Bridging the Sim-to-Real Gap for Athletic Loco-Manipulation。
- 关键词:actuator-modeling、agile-locomotion、足式运动、移动操作、mobile-manipulation、机器人操作、鲁棒控制、仿真到真实迁移、torque-control。
- System0 关联:关注机器人身体级快反应链路中“状态/接触/局部感知 -> 约束/优化/策略 -> 动作”的短闭环,不把问题停留在语言规划或任务分解层。

## 速查卡片

> [!summary] Bridging Sim Gap
> - **论文**: https://www.roboticsproceedings.org/rss21/p125.pdf
> - **arXiv**: http://arxiv.org/abs/2502.10894v1
> - **arXiv HTML**: https://arxiv.org/html/2502.10894v1
