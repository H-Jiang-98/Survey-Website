---
title: "Perceptive Humanoid Parkour: Chaining Dynamic Human Skills via Motion Matching"
method_name: "Perceptive Humanoid Parkour"
authors: ["Zhen Wu"]
year: 2026
venue: "RSS"
tags: ["robust-control", "real-time-control", "legged-locomotion", "reinforcement-learning", "adaptive-control", "imitation-learning", "robot-generalization", "closed-loop-control", "humanoid", "agile-locomotion"]
image_source: "online"
arxiv_html: "https://arxiv.org/html/2602.15827v2"
---
# Perceptive Humanoid Parkour
## 一句话总结

> Perceptive Humanoid Parkour: Chaining Dynamic Human Skills via Motion Matching 主要落在 [[adaptive-control]]、[[agile-locomotion]]、[[closed-loop-control]]、[[人形机器人]] 这一类问题上;从 System0 视角看,它的价值在于把身体状态、接触变化、局部感知或动力学约束接到机器人低层闭环里,让动作生成更稳定、安全、实时。
## 核心贡献

1. 围绕 **Perceptive Humanoid Parkour: Chaining Dynamic Human Skills via Motion Matching** 建立了一个与 adaptive-control、agile-locomotion、closed-loop-control、人形机器人、足式运动、运动模仿 相关的机器人问题设定,重点不是高层语义规划,而是身体级闭环中的状态变化、接触变化或控制约束。
2. 方法层面的关键价值在于把感知、动力学、约束优化、策略学习或硬件反馈组织成可执行的低层动作生成机制,适合放进 System0 的快反应模块中理解。
3. 对系统集成而言,这类工作补上了 System1 技能选择与电机级控制之间的一层:它负责把上层目标变成动态可行、可恢复、可安全过滤的动作。
4. 若论文包含真实机器人、接触任务或仿真到真实迁移,核心应看它是否报告了闭环稳定性、扰动恢复、硬件延迟、失败模式和跨场景泛化。
## 问题背景

这篇论文可以放在 System0 的“具身快系统”背景下阅读:机器人需要在很短时间内根据本体状态、接触、力/触觉、局部视觉或动力学预测做出动作,而不是等待慢速规划器重新推理整个任务。adaptive-control、agile-locomotion、closed-loop-control、人形机器人、足式运动、运动模仿、实时控制 相关问题的难点通常不在“知道要做什么”,而在动作执行过程中状态会快速偏离、接触会突然改变、约束会实时收紧,低层系统必须及时修正。

### System0 关联

精读时优先看三件事:第一,论文使用哪些快速反馈量,例如关节状态、接触状态、力觉、触觉、深度或局部视觉;第二,反馈如何进入 MPC、WBC、策略网络、安全过滤器或恢复控制器;第三,输出是否直接影响机器人下一步动作,而不是只服务离线分析或高层任务规划。
## 方法详解

### System0 视角解析

从 System0 角度看,这篇论文的方法部分应当围绕 adaptive-control、agile-locomotion、closed-loop-control、人形机器人、足式运动、运动模仿、实时控制、强化学习 来读。核心问题是:系统如何从快速变化的机器人状态中构造控制输入,如何处理接触、碰撞、约束、模型误差或执行延迟,以及如何把这些信息变成下一时刻的动作。

如果方法里有优化问题,重点看目标函数、约束和求解频率;如果方法里有学习策略,重点看观测空间、动作空间、训练扰动和部署时的闭环频率;如果方法里有硬件或传感器,重点看传感链路是否真的进入控制回路。这样读可以把论文和 System2 语言规划、System1 技能编排区分开来。

### 关键公式与机制

#### 公式 1: [[优化目标/约束]]

$$i_{t}^{\star}=\arg\min_{i\in\mathcal{C}_{t}}\;\lVert \hat{\boldsymbol{x}}_{t}-\boldsymbol{x}_{i} \rVert^{2},$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 2: [[优化目标/约束]]

$$\mathcal{L}=\lambda_{\text{PPO}}\,\mathcal{L}_{\text{PPO}}+\lambda_{D}\,\mathcal{L}_{D},\qquad\lambda_{\text{PPO}}+\lambda_{D}=1,$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 3: [[优化目标/约束]]

$$\lambda_{D}(k)=\max\left(0.1,1-\frac{k}{K/2}\right)$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 4: [[动力学或策略机制]]

$$\boldsymbol{p}(\tau)=\boldsymbol{p}_{0}-\frac{\boldsymbol{j}_{1}}{y^{2}}e^{-y\tau}+\frac{-\boldsymbol{j}_{0}-\tau\boldsymbol{j}_{1}}{y}e^{-y\tau}+\frac{\boldsymbol{j}_{1}}{y^{2}}+\frac{\boldsymbol{j}_{0}}{y}+\boldsymbol{u}^{\text{cmd}}_{t}\,\tau,$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 5: [[优化目标/约束]]

$$\exp\!\Big(-\big(\tfrac{1}{| $\mathcal{B}_{\mathrm{target}}$ |}\sum_{b\in\mathcal{B}_{\mathrm{target}}}\lVert \mathbf{p}^{\mathrm{des}}_{b}-\mathbf{p}_{b} \rVert^{2}\big)/0.3^{2}\Big)$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 6: [[优化目标/约束]]

$$\exp\!\Big(-\big(\tfrac{1}{| $\mathcal{B}_{\mathrm{target}}$ |}\sum_{b\in\mathcal{B}_{\mathrm{target}}}\lVert \log(R^{\mathrm{des}}_{b}R_{b}^{\top})\rVert^{2}\big)/0.4^{2}\Big)$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 7: [[优化目标/约束]]

$$\exp\!\Big(-\big(\tfrac{1}{| $\mathcal{B}_{\mathrm{target}}$ |}\sum_{b\in\mathcal{B}_{\mathrm{target}}}\lVert \mathbf{v}^{\mathrm{des}}_{b}-\mathbf{v}_{b} \rVert^{2}\big)/1.0^{2}\Big)$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 8: [[优化目标/约束]]

$$\exp\!\Big(-\big(\tfrac{1}{| $\mathcal{B}_{\mathrm{target}}$ |}\sum_{b\in\mathcal{B}_{\mathrm{target}}}\lVert \boldsymbol{\omega}^{\mathrm{des}}_{b}-\boldsymbol{\omega}_{b} \rVert^{2}\big)/3.14^{2}\Big)$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 9: [[优化目标/约束]]

$$\sum_{j=1}^{N}\big[\max(l_{j}-\theta_{j},0)+\max(\theta_{j}-u_{j},0)\big]$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 10: [[优化目标/约束]]

$$\sum_{b\notin\mathcal{B}_{\mathrm{ee}}}\mathbf{1}\!\left[\lVert f^{\mathrm{self}}_{b} \rVert>1\,\text{N}\right]$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。
## 关键图表

### Figure 1: Pipeline / core system figure: Perceptive Humanoid Parkour. Atomic parkour skills are composed into long-hor

![Figure 1](https://arxiv.org/html/2602.15827v2/x2.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Pipeline / core system figure: Perceptive Humanoid Parkour. Atomic parkour skills are composed into long-hor”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 2: Diverse variations of composed parkour skills synthesized via motion matching. (a) Dif

![Figure 2](https://arxiv.org/html/2602.15827v2/x3.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Diverse variations of composed parkour skills synthesized via motion matching. (a) Dif”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 3: Hardware results demonstrating agile, long-horizon parkour behaviors, including (a) a

![Figure 3](https://arxiv.org/html/2602.15827v2/x5.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Hardware results demonstrating agile, long-horizon parkour behaviors, including (a) a”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### 论文表格

#### Table 1: TABLE I: Baseline success rate on parkour tasks with different commanded speeds and obstacle heights.

| Commanded Velocity | 1.0 m/s | 2.0 m/s | | | | |
| --- | --- | --- | --- | --- | --- | --- |
| | 36 cm | 58 cm | 76 cm | 36 cm | 58 cm | 76 cm |
| Velocity Tracking | 1.00 | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 |
| Uncomposed Data | 0.06 | 0.02 | 0.00 | 0.37 | 0.27 | 0.07 |
| End-to-end Depth | 0.95 | 0.07 | 0.08 | 0.78 | 0.19 | 0.14 |
| Ours | 1.00 | 0.99 | 0.95 | 1.00 | 0.99 | 0.95 |

**说明**: TABLE I: Baseline success rate on parkour tasks with different commanded speeds and obstacle heights.

#### Table 2: TABLE II: Success rate on parkour tasks with different motion matching densities and RL strategies during distillation.

| Method | 1.0 m/s | 2.0 m/s | | | | |
| --- | --- | --- | --- | --- | --- | --- |
| | 58 cm | 76 cm | 94 cm | 36 cm | 58 cm | 76 cm |
| Extreme Distances | 0.99 | 0.62 | 0.64 | 0.98 | 0.60 | 0.58 |
| Half Density | 0.95 | 0.32 | 0.57 | 0.99 | 0.85 | 0.81 |
| DAgger Only | 0.16 | 0.03 | 0.12 | 0.63 | 0.09 | 0.10 |
| DAgger & Alive Reward | 1.00 | 0.90 | 0.96 | 0.94 | 0.91 | 0.84 |
| DAgger & Root Tracking | 1.00 | 0.79 | 0.75 | 1.00 | 0.92 | 0.87 |
| 1/4 Training Envs | 0.97 | 0.00 | 0.59 | 0.94 | 0.65 | 0.58 |
| 1/2 Training Envs | 0.94 | 0.60 | 0.68 | 0.97 | 0.79 | 0.75 |
| 3-layer MLP | 0.99 | 0.02 | 0.00 | 0.98 | 0.89 | 0.81 |
| 4-layer MLP | 1.00 | 0.94 | 0.08 | 1.00 | 0.94 | 0.88 |
| Ours | 0.99 | 0.95 | 1.00 | 1.00 | 0.98 | 0.90 |

**说明**: TABLE II: Success rate on parkour tasks with different motion matching densities and RL strategies during distillation.

#### Table 3: TABLE III: Motion clips used in our motion library.

| Skill | Duration (s) |
| --- | --- |
| Locomotion | |
| Locomotion | 495.5 |
| Parkour skills @ 1.0 m/s | |
| Step (36 cm) | 2.2 |
| Climb (58 cm) | 12.1 |
| Climb (76 cm) | 8.8 |
| Climb (94 cm) | 10.3 |
| Parkour skills @ 2.0 m/s | |
| Step (36 cm) | 1.6 |
| Climb (58 cm) | 6.1 |
| Climb (76 cm) | 4.4 |
| Climb (94 cm) | 5.2 |
| Climb (125 cm) | 5.9 |
| Dash Vault | 5.0 |
| Speed Vault | 3.1 |
| Parkour skills @ 3.0 m/s | |
| Cat Vault | 1.5 |

**说明**: TABLE III: Motion clips used in our motion library.

#### Table 4: TABLE IV: Reward formulation using Gaussian-shaped tracking scores.

| Reward Terms | Equation | Weight |
| --- | --- | --- |
| Task (Tracking) | | |
| Body Position | $\exp\!\Big(-\big(\tfrac{1}{\lVert \mathcal{B}_{\mathrm{target}} \rVert}\sum_{b\in\mathcal{B}_{\mathrm{target}}}\lVert \mathbf{p}^{\mathrm{des}}_{b}-\mathbf{p}_{b} \rVert^{2}\big)/0.3^{2}\Big)$ | 1.0 1.0 |
| Body Orientation | $\exp\!\Big(-\big(\tfrac{1}{\lVert \mathcal{B}_{\mathrm{target}} \rVert}\sum_{b\in\mathcal{B}_{\mathrm{target}}}\lVert \log(R^{\mathrm{des}}_{b}R_{b}^{\top})\rVert^{2}\big)/0.4^{2}\Big)$ | 1.0 1.0 |
| Body Linear velocity | $\exp\!\Big(-\big(\tfrac{1}{\lVert \mathcal{B}_{\mathrm{target}} \rVert}\sum_{b\in\mathcal{B}_{\mathrm{target}}}\lVert \mathbf{v}^{\mathrm{des}}_{b}-\mathbf{v}_{b} \rVert^{2}\big)/1.0^{2}\Big)$ | 1.0 1.0 |
| Body Angular velocity | $\exp\!\Big(-\big(\tfrac{1}{\lVert \mathcal{B}_{\mathrm{target}} \rVert}\sum_{b\in\mathcal{B}_{\mathrm{target}}}\lVert \boldsymbol{\omega}^{\mathrm{des}}_{b}-\boldsymbol{\omega}_{b} \rVert^{2}\big)/3.14^{2}\Big)$ | 1.0 1.0 |
| Anchor Position | $\exp\!\Big(-\lVert \mathbf{p}^{\mathrm{des}}_{\text{anchor}}-\mathbf{p}_{\text{anchor}} \rVert^{2}/0.3^{2}\Big)$ | 1.0 1.0 |
| Anchor Orientation | $\exp\!\Big(-\lVert \log(R^{\mathrm{des}}_{\text{anchor}}R_{\text{anchor}}^{\top})\rVert^{2}/0.4^{2}\Big)$ | 1.0 1.0 |
| Regularization | | |
| Action smoothness | $\mathbf{a}_{t}-\mathbf{a}_{t-1}\lVert^{2}$ | - 0.1 -0.1 |
| Joint position limit | $\sum_{j=1}^{N}\big[\max(l_{j}-\theta_{j},0)+\max(\theta_{j}-u_{j},0)\big]$ | - 10.0 -10.0 |
| Undesired self-contacts | $\sum_{b\notin\mathcal{B}_{\mathrm{ee}}}\mathbf{1}\!\left[\lVert f^{\mathrm{self}}_{b} \rVert>1\,\text{N}\right]$ | - 0.5 -0.5 |

**说明**: TABLE IV: Reward formulation using Gaussian-shaped tracking scores.

#### Table 5: TABLE V: Domain randomization parameters. (U  [⋅ ] $\mathcal{U}[\cdot]$: uniform distribution)

| Domain Randomization | Sampling Distribution |
| --- | --- |
| Physical parameters | |
| Static friction coefficients | $\mu_{\text{static}}\sim\mathcal{U}[0.4,\,1.3]$ |
| Dynamic friction coefficients | dynamic ∼ U . . $\mu_{\text{dynamic}}\sim\mathcal{U}[0.4,\,1.1]$ |
| Restitution coefficient | e rest ∼ U  [0, 0.5 ] e_{ $\text{rest}}\sim\mathcal{U}[0,\,0.5]$ |
| Default joint positions (except ankle) [rad] | $\theta^{0}_{j}\sim\mathcal{U}[-0.01,\,0.01]$ |
| Default ankle joint positions [rad] | $\theta^{0}_{j}\sim\mathcal{U}[-0.03,\,0.03]$ |
| Torso COM offset [m] | $\mathcal{U}[-0.025,0.025],\ \Delta y,\Delta z\!\sim\!\mathcal{U}[-0.05,0.05]$ |
| Root velocity perturbations | |
| Root linear vel [m/s] | v x, v y ∼ U  [- 0.1, 0.1 ], v z ∼ U  [- 0.05, 0.05 ] v_{x},v_{y}\!\sim\! $\mathcal{U}[-0.1,0.1],\ v_{z}\!\sim\!\mathcal{U}[-0.05,0.05]$ |
| Push duration [s] | $\mathcal{U}[1.0,\,3.0]$ |
| Root angular vel [rad/s] | $\omega_{x},\omega_{y},\omega_{z}\!\sim\!\mathcal{U}[-0.1,0.1]$ |

**说明**: 该表格来自论文中的公式、奖励项或实验设置；为避免 PDF 抽取噪声，网页中仅保留可渲染的核心符号和数值。

#### Table 6: TABLE VI: Training hyperparameters.

| Hyperparameter | Motion Tracking | Distillation |
| --- | --- | --- |
| Architecture | | |
| Actor / Student MLP hidden dims | [512, 256, 128] | [2048, 1024, 512, 256, 128] |
| Critic MLP hidden dims | [512, 256, 128] | [512, 256, 128] |
| Activation function | ELU | ELU |
| Init noise std | 1.0 | 0.01 |
| Depth backbone | – | 3-layer CNN + GAP |
| Depth input resolution | – | 58  87 58\times 87 |
| Depth output dim | – | 32 |
| Training | | |
| Steps per environment | 24 | 24 |
| Max iterations | 20,000 | 20,000 |
| Learning rate | 1  10 - 3 1\times 10^{-3} | 3  10 - 4 3\times 10^{-4} |
| Schedule | adaptive | adaptive after 1000 iterations |
| Clip parameter | 0.2 | 0.2 |
| Entropy coefficient | 0.005 | 0.001 |
| Discount factor $\gamma)$ | 0.99 | 0.99 |
| GAE  \lambda | 0.95 | 0.95 |
| Desired KL | 0.01 | 0.01 |
| Learning epochs | 5 | 2 |
| Mini-batches | 4 | 96 |
| Max grad norm | 1.0 | 1.0 |
| Distillation-Specific | | |
| Curriculum end epoch | – | 10,000 |
| Distill loss type | – | mse |
| DAgger loss coefficient | – | 10.0 |

**说明**: TABLE VI: Training hyperparameters.
## 实验解读

- 评价重点:围绕 adaptive-control、agile-locomotion、closed-loop-control、人形机器人、足式运动,优先看真实机器人结果、扰动恢复、接触稳定性、控制误差、成功率、实时性和 sim-to-real 差距。
- System0 视角:实验结论不只看最终成功率,还要看闭环过程中是否能处理状态估计误差、执行延迟、碰撞/接触突变和外部扰动。
- 与上层系统的关系:如果方法能在快闭环内稳定工作,它可以作为 System1 技能或 System2 规划的底层执行保障;如果只在离线或慢速设定下成立,则还不能直接作为身体级快闭环模块。
## 批判性思考

### 优点

- 论文把 adaptive-control、agile-locomotion、closed-loop-control、人形机器人、足式运动 放到机器人身体级闭环里讨论,和只做高层语义推理的工作相比,更接近真实机器人部署时会遇到的状态、动力学和安全约束。
- 若论文同时给出公式、系统结构图、硬件实验或消融结果,这些证据可以帮助判断方法是否真的能支撑毫秒到百毫秒级反应。

### 局限

- 需要重点核查控制频率、推理延迟、传感延迟、失败恢复、硬件磨损、未见场景泛化和安全边界;这些因素决定它能否成为可靠的 System0 模块。
- 对学习型方法,还要区分训练环境中的鲁棒性和真实部署中的鲁棒性,尤其是接触丰富任务、动态运动任务和人形机器人任务。
## 论文定位

- 研究对象:Perceptive Humanoid Parkour: Chaining Dynamic Human Skills via Motion Matching。
- 关键词:adaptive-control、agile-locomotion、closed-loop-control、人形机器人、足式运动、运动模仿、实时控制、强化学习、retargeting、鲁棒控制。
- System0 关联:关注机器人身体级快反应链路中“状态/接触/局部感知 -> 约束/优化/策略 -> 动作”的短闭环,不把问题停留在语言规划或任务分解层。

## 速查卡片

> [!summary] Perceptive Humanoid Parkour
> - **论文**: https://www.roboticsproceedings.org/rss22/p020.pdf
> - **arXiv**: http://arxiv.org/abs/2602.15827v2
> - **arXiv HTML**: https://arxiv.org/html/2602.15827v2
