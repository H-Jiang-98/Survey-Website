---
title: "Advancing Humanoid Locomotion: Mastering Challenging Terrains with Denoising World Model Learning"
method_name: "Advancing Humanoid Locomotion"
authors: ["Xinyang Gu"]
year: 2024
venue: "RSS"
tags: ["robot-manipulation", "robust-control", "legged-locomotion", "reinforcement-learning", "robot-generalization", "humanoid", "sim-to-real", "loco-manipulation"]
image_source: "online"
arxiv_html: "https://arxiv.org/html/2408.14472v1"
---
# Advancing Humanoid Locomotion
## 一句话总结

> Advancing Humanoid Locomotion: Mastering Challenging Terrains with Denoising World Model Learning 主要落在 [[人形机器人]]、[[足式运动]]、[[移动操作]]、[[强化学习]] 这一类问题上;从 System0 视角看,它的价值在于把身体状态、接触变化、局部感知或动力学约束接到机器人低层闭环里,让动作生成更稳定、安全、实时。
## 核心贡献

1. 围绕 **Advancing Humanoid Locomotion: Mastering Challenging Terrains with Denoising World Model Learning** 建立了一个与 人形机器人、足式运动、移动操作、强化学习、robot-generalization、机器人操作 相关的机器人问题设定,重点不是高层语义规划,而是身体级闭环中的状态变化、接触变化或控制约束。
2. 方法层面的关键价值在于把感知、动力学、约束优化、策略学习或硬件反馈组织成可执行的低层动作生成机制,适合放进 System0 的快反应模块中理解。
3. 对系统集成而言,这类工作补上了 System1 技能选择与电机级控制之间的一层:它负责把上层目标变成动态可行、可恢复、可安全过滤的动作。
4. 若论文包含真实机器人、接触任务或仿真到真实迁移,核心应看它是否报告了闭环稳定性、扰动恢复、硬件延迟、失败模式和跨场景泛化。
## 问题背景

这篇论文可以放在 System0 的“具身快系统”背景下阅读:机器人需要在很短时间内根据本体状态、接触、力/触觉、局部视觉或动力学预测做出动作,而不是等待慢速规划器重新推理整个任务。人形机器人、足式运动、移动操作、强化学习、robot-generalization、机器人操作、鲁棒控制 相关问题的难点通常不在“知道要做什么”,而在动作执行过程中状态会快速偏离、接触会突然改变、约束会实时收紧,低层系统必须及时修正。

### System0 关联

精读时优先看三件事:第一,论文使用哪些快速反馈量,例如关节状态、接触状态、力觉、触觉、深度或局部视觉;第二,反馈如何进入 MPC、WBC、策略网络、安全过滤器或恢复控制器;第三,输出是否直接影响机器人下一步动作,而不是只服务离线分析或高层任务规划。
## 方法详解

### System0 视角解析

从 System0 角度看,这篇论文的方法部分应当围绕 人形机器人、足式运动、移动操作、强化学习、robot-generalization、机器人操作、鲁棒控制、仿真到真实迁移 来读。核心问题是:系统如何从快速变化的机器人状态中构造控制输入,如何处理接触、碰撞、约束、模型误差或执行延迟,以及如何把这些信息变成下一时刻的动作。

如果方法里有优化问题,重点看目标函数、约束和求解频率;如果方法里有学习策略,重点看观测空间、动作空间、训练扰动和部署时的闭环频率;如果方法里有硬件或传感器,重点看传感链路是否真的进入控制回路。这样读可以把论文和 System2 语言规划、System1 技能编排区分开来。

### 关键公式与机制

#### 公式 1: [[优化目标/约束]]

$$\mathcal{M}=\langle\mathcal{S},\mathcal{A},T,\mathcal{O},R,\gamma\rangle$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 2: [[动力学或策略机制]]

$$J=\mathbb{E}[R_{t}]=\mathbb{E}\left[\sum_{t}\gamma^{t}r_{t}\right]$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 3: [[动力学或策略机制]]

$$\phi(e,w):=\exp\left(-w\cdot\lVert e \rVert^{2}\right)$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 4: [[动力学或策略机制]]

$$f(t)=\sum_{k\leq 5}a_{k}t^{k}$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 5: [[动力学或策略机制]]

$$P(\tilde{\mathbf{s}}_{t})=\mathrm{E}_{o_{\leq t}}\left[\int_{z}P_{\text{Decoder}}(\tilde{\mathbf{s}}_{t}| z_{t})\cdot P_{ $\text{Encoder}}(z_{t}$ |o_{\leq t})\right]$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 6: [[优化目标/约束]]

$$
\mathcal{L}_{\mathrm{denoise}}=\lVert \tilde{\mathbf{s}}_{t}-\mathbf{s}_{t} \rVert_{2}+\lambda_{r}\lVert \boldsymbol{z}_{t} \rVert_{1}
$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 7: [[优化目标/约束]]

$$
\mathcal{L}_{\text{DWL}}=\mathcal{L}_{\text{denoise}}+\lambda_{\pi}\mathcal{L}_{\pi}+\lambda_{v}\mathcal{L}_{v},
$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 8: [[优化目标/约束]]

$$=\min\left[\frac{\pi(a_{t}\mid o_{\leq t})}{\pi_{b}(a_{t}\mid o_{\leq t})}A^{\pi_{b}}(o_{\leq t},a_{t}),\right.$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 9: [[动力学或策略机制]]

$$r_{\text{velocity}}^{\text{periodic}}(t)=(1-I_{L}(t))\cdot\dot{P}^{f}_{L}+(1-I_{R}(t))\cdot\dot{P}^{f}_{R}$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 10: [[动力学或策略机制]]

$$\quad\left.\text{clip}\left(\frac{\pi(a_{t}\mid o_{\leq t})}{\pi_{b}(a_{t}\mid o_{\leq t})},c_{1},c_{2}\right)A^{\pi_{b}}(o_{\leq t},a_{t})\right]$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。
## 关键图表

### Figure 1: Pipeline / core system figure: Illustration of the humanoid robot’s hardware structure and the Closed Kinematic Chai

![Figure 1](https://arxiv.org/html/2408.14472v1/x2.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Pipeline / core system figure: Illustration of the humanoid robot’s hardware structure and the Closed Kinematic Chai”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 2: Illustration of the Denoising World Model Learning Framework. This diagram details th

![Figure 2](https://arxiv.org/html/2408.14472v1/x3.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Illustration of the Denoising World Model Learning Framework. This diagram details th”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 3: Dynamic adaptation of the ankle control mechanism. A) The top image demonstrates the

![Figure 3](https://arxiv.org/html/2408.14472v1/x4.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Dynamic adaptation of the ankle control mechanism. A) The top image demonstrates the”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### 论文表格

#### Table 1: TABLE I: Summary of Observation Space. The table categorizes the components of the observation space into observation a

| Components | Dims | Observation | State |
| --- | --- | --- | --- |
| Clock Input | 2 | ✓ | ✓ |
| Commands | 3 | ✓ | ✓ |
| Joint Position | 12 | ✓ | ✓ |
| Joint Velocity | 12 | ✓ | ✓ |
| Angular Velocity | 3 | ✓ | ✓ |
| Orientation | 3 | ✓ | ✓ |
| Last Actions | 12 | ✓ | ✓ |
| Base Linear Velocity | 3 | | ✓ |
| Frictions | 1 | | ✓ |
| Push Force&Torques | 6 | | ✓ |
| Cycle Time | 1 | | ✓ |
| Periodic Stance Mask | 2 | | ✓ |
| Feet movement | 12 | | ✓ |
| Feet Contact | 2 | | ✓ |
| Body Mass | 1 | | ✓ |
| Current Reward | 1 | | ✓ |
| Torques | 12 | | ✓ |
| Height Scan | 96 | | ✓ |

**说明**: TABLE I: Summary of Observation Space. The table categorizes the components of the observation space into observation and state. The table also details their dimensions.

#### Table 2: TABLE II: Overview of Domain Randomization. Presented are the domain randomization terms and the associated parameter r

| Parameter | Unit | Range | Operator |
| --- | --- | --- | --- |
| Joint Position | rad | [-0.3, 0.3] | additive |
| Joint Velocity | rad/s | [-1, 1] | additive |
| Angular Velocity | rad/s | [-0.1, 0.1] | additive |
| Orientation | rad | [-0.1, 0.1] | additive |
| System Delay | ms | [0, 10] | - |
| Friction | - | [0.2, 2.0] | - |
| Motor Offset | rad | [-0.05, 0.05] | additive |
| Motor Strength | % | [90, 110] | scaling |
| Payload | kg | [-5, 20] | additive |
| PD Factors | % | [80, 120] | scaling |

**说明**: TABLE II: Overview of Domain Randomization. Presented are the domain randomization terms and the associated parameter ranges. Additive randomization increments the parameter by a value within the specified range while scaling randomization adjusts it by a multiplicative factor from the same range.

#### Table 3: TABLE III: Real robot testing across various terrains. Bold values is our DWL with ankle control, DWL p DWL p

| Algorithm | Slope | Stair-up | Stair-down | Irregular |
| --- | --- | --- | --- | --- |
| PPO | 80% | 20% | 60% | 20% |
| DWL p DWL p  $\text{DWL}_{p} DWL$ | 80% | 20% | 100% | 40% |
| DWL | 100% | 100% | 100% | 100% |

**说明**: 该表格来自论文中的公式、奖励项或实验设置；为避免 PDF 抽取噪声，网页中仅保留可渲染的核心符号和数值。

#### Table 4: TABLE IV: Quintic Polynomial Foot Trajectory Parameters f (t) = ∑ k ≤ 5 a k t k f t k 5 a k s

| Trajectory Parameters | Optimization constraints | | |
| --- | --- | --- | --- |
| Coefficient | Value | Objection | Value |
| a 5 a 5 a_{5} 5 | 9.6 | h 0 h 0 h_{0} 0 | 0.0 |
| a 4 a 4 a_{4} 4 | 12.0 | h T h T h_{T} | 0.0 |
| a 3 a 3 a_{3} 3 | -18.8 | v 0 v 0 v_{0} 0 | 0.1 |
| a 2 a 2 a_{2} 2 | 5.0 | v T v T v_{T} | 0.0 |
| a 1 a 1 a_{1} 1 | 0.1 | h m a x h m a x h_{max} | 0.1 |
| a 0 a 0 a_{0} 0 | 0.0 | T T T | 0.5 |

**说明**: 该表格来自论文中的公式、奖励项或实验设置；为避免 PDF 抽取噪声，网页中仅保留可渲染的核心符号和数值。

#### Table 5: TABLE V: In defining the reward function, we use a tracking error metric denoted by  (e, w) e w $\phi($e,w

| Reward | Equation (r i r i r_{i}) | $\mu_{i})$ |
| --- | --- | --- |
| Lin. velocity tracking | $\dot{P}^{b}_{xyz}-\text{CMD}_{xyz},5)$ | 1.0 |
| Ang. velocity tracking | $\dot{P}^{b}_{\alpha\beta\gamma}-\text{CMD}_{\alpha\beta\gamma},7)$ | 1.0 |
| Orientation tracking |  (P   b, 5) P b   5 \phi(P^{b}_{\alpha\beta},5) (, 5) | 1.0 |
| Base height tracking |  (P z b - 0.7, 10) P b z 0.7 10 \phi(P^{b}_{z}-0.7,10) (- 0.7, 10) | 0.5 |
| Periodic Force | r F o r c e p e r i o d i c (t) r F o r c e p e r i o d i c t r_{Force}^{periodic}(t) () | 1.0 |
| Periodic Velocity | r v e l o c i t y p e r i o d i c (t) r v e l o c i t y p e r i o d i c t r_{velocity}^{periodic}(t) () | 1.0 |
| Foot height tracking |  (P z f - f t, 5) P f z f t 5 \phi(P^{f}_{z}-f_{t},5) (-, 5) | 1.0 |
| Foot vel tracking | $\dot{P}^{f}_{z}-\dot{f}_{t},3)$ | 0.5 |
| Default Joint | $\theta_{t}-\theta_{0},2)$ | 0.2 |
| Energy Cost | $\tau\lVert \rVert\dot{\theta}\lVert \rVert \lVert \rVert \lVert$ | -0.0001 |
| Action Smoothness |  a t - 2 a t - 1 + a t - 2  2 norm a t 2 a t 1 a t 2 2 \\|a_{t}-2a_{t-1}+a_{t-2}\\|_{2}  - 2 - 1 + - 2  2 | -0.01 |
| Feet movements | $$ | -0.01 |
| Large contact | CLIP (F L, R - 400, 0, 100) CLIP F L R 400 0 100  $\text{CLIP}(F_{L,R}-400,0,100) CLIP$ | -0.01 |

**说明**: 该表格来自论文中的公式、奖励项或实验设置；为避免 PDF 抽取噪声，网页中仅保留可渲染的核心符号和数值。

#### Table 6: TABLE VI: DWL Network Architecture Details

| Component | Configuration |
| --- | --- |
| Encoder | |
| RNN_memory(0) | GRU(47 → →  $\rightarrow → 256)$ |
| emb_model (1) | Linear(256 → →  $\rightarrow → 256)$ |
| emb_model (2) | ELU(alpha=1.0) |
| emb_model (3) | Linear(256 → →  $\rightarrow → 24)$ |
| Decoder | |
| denoise_net (0) | Linear(24 → →  $\rightarrow → 64)$ |
| denoise_net (1) | ELU(alpha=1.0) |
| denoise_net (2) | Linear(64 → →  $\rightarrow → 184)$ |
| Actor | |
| policy_net (0) | Linear(24 → →  $\rightarrow → 48)$ |
| policy_net (1) | ELU(alpha=1.0) |
| policy_net (2) | Linear(48 → →  $\rightarrow → 12)$ |
| Critic | |
| Critic_Net (0) | Linear(184 → →  $\rightarrow → 512)$ |
| Critic_Net (1) | ELU(alpha=1.0) |
| Critic_Net (2) | Linear(512 → →  $\rightarrow → 512)$ |
| Critic_Net (3) | ELU(alpha=1.0) |
| Critic_Net (4) | Linear(512 → →  $\rightarrow → 256)$ |
| Critic_Net (5) | ELU(alpha=1.0) |
| Critic_Net (6) | Linear(256 → →  $\rightarrow → 1)$ |

**说明**: TABLE VI: DWL Network Architecture Details

#### Table 7: TABLE VII: PPO Network Architecture Details

| Component | Configuration |
| --- | --- |
| Actor | |
| RNN_memory(0) | GRU(47 → →  $\rightarrow → 256)$ |
| policy_net (1) | Linear(256 → →  $\rightarrow → 256)$ |
| policy_net (2) | ELU(alpha=1.0) |
| policy_net (3) | Linear(256 → →  $\rightarrow → 128)$ |
| policy_net (4) | ELU(alpha=1.0) |
| policy_net (5) | Linear(128 → →  $\rightarrow → 12)$ |
| Critic | |
| Critic_Net (0) | Linear(184 → →  $\rightarrow → 512)$ |
| Critic_Net (1) | ELU(alpha=1.0) |
| Critic_Net (2) | Linear(512 → →  $\rightarrow → 512)$ |
| Critic_Net (3) | ELU(alpha=1.0) |
| Critic_Net (4) | Linear(512 → →  $\rightarrow → 256)$ |
| Critic_Net (5) | ELU(alpha=1.0) |
| Critic_Net (6) | Linear(256 → →  $\rightarrow → 1)$ |

**说明**: TABLE VII: PPO Network Architecture Details

#### Table 8: TABLE VIII: Hyperparameters of DWL.

| Parameter | Value |
| --- | --- |
| Number of Environments | 12288 |
| Number Training Epochs | 2 |
| Batch size | 12288  24 12288 24 12288\times 24 12288  24 |
| Episode Length | 2400 steps |
| Discount Factor | 0.995 |
| GAE discount factor | 0.95 |
| Entropy Regularization Coefficient | 0.005 |
| c 1 c 1 c1 1 | 0.8 |
| c 2 c 2 c2 2 | 1.2 |
| Learning rate | 1e-5 |
| regularization coefficient  r  r \lambda_{r} | 0.002 |
| policy coefficient \lambda $\pi}$ | 5 |
| value coefficient  v  v \lambda_{v} | 5 |

**说明**: TABLE VIII: Hyperparameters of DWL.
## 实验解读

- 评价重点:围绕 人形机器人、足式运动、移动操作、强化学习、robot-generalization,优先看真实机器人结果、扰动恢复、接触稳定性、控制误差、成功率、实时性和 sim-to-real 差距。
- System0 视角:实验结论不只看最终成功率,还要看闭环过程中是否能处理状态估计误差、执行延迟、碰撞/接触突变和外部扰动。
- 与上层系统的关系:如果方法能在快闭环内稳定工作,它可以作为 System1 技能或 System2 规划的底层执行保障;如果只在离线或慢速设定下成立,则还不能直接作为身体级快闭环模块。
## 批判性思考

### 优点

- 论文把 人形机器人、足式运动、移动操作、强化学习、robot-generalization 放到机器人身体级闭环里讨论,和只做高层语义推理的工作相比,更接近真实机器人部署时会遇到的状态、动力学和安全约束。
- 若论文同时给出公式、系统结构图、硬件实验或消融结果,这些证据可以帮助判断方法是否真的能支撑毫秒到百毫秒级反应。

### 局限

- 需要重点核查控制频率、推理延迟、传感延迟、失败恢复、硬件磨损、未见场景泛化和安全边界;这些因素决定它能否成为可靠的 System0 模块。
- 对学习型方法,还要区分训练环境中的鲁棒性和真实部署中的鲁棒性,尤其是接触丰富任务、动态运动任务和人形机器人任务。
## 论文定位

- 研究对象:Advancing Humanoid Locomotion: Mastering Challenging Terrains with Denoising World Model Learning。
- 关键词:人形机器人、足式运动、移动操作、强化学习、robot-generalization、机器人操作、鲁棒控制、仿真到真实迁移、terrain-adaptation、world-model。
- System0 关联:关注机器人身体级快反应链路中“状态/接触/局部感知 -> 约束/优化/策略 -> 动作”的短闭环,不把问题停留在语言规划或任务分解层。

## 速查卡片

> [!summary] Advancing Humanoid Locomotion
> - **论文**: https://www.roboticsproceedings.org/rss20/p058.pdf
> - **arXiv**: http://arxiv.org/abs/2408.14472v1
> - **arXiv HTML**: https://arxiv.org/html/2408.14472v1
