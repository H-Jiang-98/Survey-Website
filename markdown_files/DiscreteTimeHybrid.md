---
title: "Discrete-Time Hybrid Automata Learning: Legged Locomotion Meets Skateboarding"
method_name: "Discrete Time Hybrid"
authors: ["Hang Liu"]
year: 2025
venue: "RSS"
tags: ["contact-reasoning", "robot-manipulation", "robust-control", "legged-locomotion", "reinforcement-learning", "agile-locomotion", "loco-manipulation"]
image_source: "online"
arxiv_html: "https://arxiv.org/html/2503.01842v2"
---
# Discrete Time Hybrid
## 一句话总结

> Discrete-Time Hybrid Automata Learning: Legged Locomotion Meets Skateboarding 主要落在 [[agile-locomotion]]、[[接触推理]]、[[足式运动]]、[[移动操作]] 这一类问题上;从 System0 视角看,它的价值在于把身体状态、接触变化、局部感知或动力学约束接到机器人低层闭环里,让动作生成更稳定、安全、实时。
## 核心贡献

1. 围绕 **Discrete-Time Hybrid Automata Learning: Legged Locomotion Meets Skateboarding** 建立了一个与 agile-locomotion、接触推理、足式运动、移动操作、mobile-manipulation、quadruped 相关的机器人问题设定,重点不是高层语义规划,而是身体级闭环中的状态变化、接触变化或控制约束。
2. 方法层面的关键价值在于把感知、动力学、约束优化、策略学习或硬件反馈组织成可执行的低层动作生成机制,适合放进 System0 的快反应模块中理解。
3. 对系统集成而言,这类工作补上了 System1 技能选择与电机级控制之间的一层:它负责把上层目标变成动态可行、可恢复、可安全过滤的动作。
4. 若论文包含真实机器人、接触任务或仿真到真实迁移,核心应看它是否报告了闭环稳定性、扰动恢复、硬件延迟、失败模式和跨场景泛化。
## 问题背景

这篇论文可以放在 System0 的“具身快系统”背景下阅读:机器人需要在很短时间内根据本体状态、接触、力/触觉、局部视觉或动力学预测做出动作,而不是等待慢速规划器重新推理整个任务。agile-locomotion、接触推理、足式运动、移动操作、mobile-manipulation、quadruped、强化学习 相关问题的难点通常不在“知道要做什么”,而在动作执行过程中状态会快速偏离、接触会突然改变、约束会实时收紧,低层系统必须及时修正。

### System0 关联

精读时优先看三件事:第一,论文使用哪些快速反馈量,例如关节状态、接触状态、力觉、触觉、深度或局部视觉;第二,反馈如何进入 MPC、WBC、策略网络、安全过滤器或恢复控制器;第三,输出是否直接影响机器人下一步动作,而不是只服务离线分析或高层任务规划。
## 方法详解

### System0 视角解析

从 System0 角度看,这篇论文的方法部分应当围绕 agile-locomotion、接触推理、足式运动、移动操作、mobile-manipulation、quadruped、强化学习、机器人操作 来读。核心问题是:系统如何从快速变化的机器人状态中构造控制输入,如何处理接触、碰撞、约束、模型误差或执行延迟,以及如何把这些信息变成下一时刻的动作。

如果方法里有优化问题,重点看目标函数、约束和求解频率;如果方法里有学习策略,重点看观测空间、动作空间、训练扰动和部署时的闭环频率;如果方法里有硬件或传感器,重点看传感链路是否真的进入控制回路。这样读可以把论文和 System2 语言规划、System1 技能编排区分开来。

### 关键公式与机制

#### 公式 1: [[动力学或策略机制]]

$$\begin{array}{l}\mathbb{E}[g_{\text{clip}}]-\nabla_{\theta}J(\pi_{\theta})=\\ \mathbb{E}_{s}\Bigg{[}\int_{\lvert a\rvert>h}\pi_{\theta}(a| s)(\nabla_{ $\theta}\log\pi_{\theta}(\pm h$ | s)A^{ $\pi}(s,\pm h)\\ -\nabla_{\theta}\log\pi_{\theta}(a$ |s)A^{\pi}(s,a))\,da\Bigg{]}\neq 0.\end{array}$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 2: [[动力学或策略机制]]

$$\nabla_{\sigma_{\theta}}J(\theta)=\frac{1}{\sigma_{\theta}^{3}}\mathbb{E}_{a\sim\pi_{\theta}}\left[\left((a-\mu_{\theta})^{2}-\sigma_{\theta}^{2}\right)A^{\pi}(s,a)\right].$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 3: [[动力学或策略机制]]

$$\mathbb{E}\big{[}\nabla_{\theta}\log\pi_{\theta}(a\mid s)Q^{\pi}(s,a)\big{]}=\nabla_{\theta}\int_{-h}^{+h}\pi_{\theta}(a\mid s)Q^{\pi}(s,a)\,da$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 4: [[动力学或策略机制]]

$$\int_{-h}^{+h}\nabla_{\theta}\pi_{\theta}(a\mid s)\,da=0,\text{i.e.,}\int_{-h}^{+h}\pi_{\theta}(a\mid s)\nabla_{\theta}\log\pi_{\theta}(a\mid s)\,da=0.$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 5: [[优化目标/约束]]

$$L^{\text{surrogate}}=\mathbb{E}_{t}\left[\min\left(\alpha_{t}\tilde{A}_{t},\,\text{clip}(\alpha_{t},1-\epsilon,1+\epsilon)\tilde{A}_{t}\right)\right]$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 6: [[优化目标/约束]]

$$L_{\mathcal{P}}=\mathbb{E}_{t}\left[\left\lVert r_{{\mathcal{P}},t}+\gamma V_{\mathcal{P}}(s_{t+1})-V_{\mathcal{P}}(s_{t})\right \rVert^{2}\right]$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 7: [[优化目标/约束]]

$$L_{\mathcal{G}}=\mathbb{E}_{t}\left[\left\lVert r_{{\mathcal{G}},t}+\gamma V_{\mathcal{G}}(s_{t+1})-V_{\mathcal{G}}(s_{t})\right \rVert^{2}\right]$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 8: [[优化目标/约束]]

$$L_{\mathcal{S}}=\mathbb{E}_{t}\left[\left\lVert r_{{\mathcal{S}},t}+\gamma V_{\mathcal{S}}(s_{t+1})-V_{\mathcal{S}}(s_{t})\right \rVert^{2}\right]$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 9: [[动力学或策略机制]]

$$\int_{-h}^{+h}\pi_{\theta}(a\mid s)\nabla_{\theta}\log\pi_{\theta}(a\mid s)\,da=\int_{-h}^{+h}\nabla_{\theta}\pi_{\theta}(a\mid s)\,da.$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 10: [[动力学或策略机制]]

$$\nabla_{\theta}J(\pi_{\theta})=\mathbb{E}_{a\sim\pi_{\theta}}\bigl{[}\nabla_{\theta}\log\pi_{\theta}(a\mid s)\,A^{\pi}(s,a)\bigr{]}.$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。
## 关键图表

### Figure 1: Pipeline / core system figure: The potholes on the downhill slope caused the robot dog’s right front leg to get stuck

![Figure 1](https://arxiv.org/html/2503.01842v2/x1.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Pipeline / core system figure: The potholes on the downhill slope caused the robot dog’s right front leg to get stuck”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 2: Discrete-time Hybrid Dynamics Learning (DHAL) Framework: (a) During training, the netw

![Figure 2](https://arxiv.org/html/2503.01842v2/x2.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Discrete-time Hybrid Dynamics Learning (DHAL) Framework: (a) During training, the netw”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 3: Real-world Experiments in Skateboard Park. For additional demonstrations, please refe

![Figure 3](https://arxiv.org/html/2503.01842v2/x7.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Real-world Experiments in Skateboard Park. For additional demonstrations, please refe”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### 论文表格

#### Table 1: TABLE I: Important symbols and abbreviations

| Meaning | Symbol |
| --- | --- |
| POMDP | |
| Full State | s t s t s_{t} |
| Partial Observation | o t o t o_{t} |
| Action | a t a t a_{t} |
| State Space | S S  $\mathcal{S}$ |
| Action Space | A A  $\mathcal{A}$ |
| Discount Factor | $\gamma$ |
| Hybrid Dynamics System | |
| Discrete-time dynamics | f i t f i t f^{i_{t}} |
| Mode index | i t i t i_{t} |
| Number of modes | K K K |
| Jacobian | J J J |
| Probability of each mode | p p p |
| mode indicator vector | $\delta$ |
| maximum number of modes | $\delta\lVert = \rVert \lVert$ |
| Environment | |
| Joint Position | q = [q H i p, T h i g h, C a l f F L / F R / R L / R R ] q q H i p T h i g h C a l f F L F R R L R R q= $\left[q_{Hip,Thigh,Calf}^{FL/FR/RL/RR}\right] = [,, / / / ]$ |
| Joint Velocity | q  ̇ = [q  ̇ H i p, T h i g h, C a l f F L / F R / R L / R R ]  ̇ q  ̇ q H i p T h i g h C a l f F L F R R L R R  $\dot{q}=\left[\dot{q}_{Hip,Thigh,Calf}^{FL/FR/RL/RR}\right] = [,, / / / ]$ |
| Go1 Base Roll, Pitch,Yaw | $\theta,\psi,,$ |
| Go1 Base angular velocity | $\omega_{x},\omega_{y},\omega_{z},,$ |
| Gravity | g x, y, z g x y z g_{x,y,z},, |
| Action | a = [a H i p, T h i g h, C a l f F L / F R / R L / R R ] a a H i p T h i g h C a l f F L F R R L R R a= $\left[a_{Hip,Thigh,Calf}^{FL/FR/RL/RR}\right] = [,, / / / ]$ |
| Phase |   \Phi |
| Command | c m d = [c x, c y a w ] c m d c x c y a w cmd=[c_{x},c_{yaw}] = [, ] |
| Proprioception | $\dot{q},g_{x,y,z},\phi,\theta,\psi,\omega_{x},\omega_{y},\omega_{z}] = [,,,,,,,,,, ]$ |
| Contact | c c c |
| Torque | $\tau$ |

**说明**: TABLE I: Important symbols and abbreviations

#### Table 2: TABLE II: Summary of Reward Terms and Their Expressions. Each term is multiplied by its phase coefficient ( glide subs

| Gliding Critic Reward | Expression |
| --- | --- |
| Feet on board | $\delta_{\mathrm{glide}}\sum_{i=1}^{4}\!\Bigl(\lVert \mathbf{p}_{\mathrm{feet},i}-\mathbf{p}_{\mathrm{glide},i} \rVert<0.05\Bigr) ∑ = 1 4$ |
| Contact number | $\delta_{\mathrm{glide}}\ R_{contact\_num} _$ |
| Feet distance | $\delta_{\mathrm{glide}}\exp\!\Bigl(-\,\sum_{i=1}^{4}\lVert \mathbf{p}_{\mathrm{glide},j}-\mathbf{p}_{\mathrm{feet},j} \rVert\Bigr)$ |
| Joint positions | $\delta_{\mathrm{glide}}\exp\!\Bigl(-\!\sum_{i=1}^{12}(q_{i}-q^{\mathrm{glide}}_{i})^{2}\Bigr)$ |
| Hip positions | $\delta_{\mathrm{glide}}\exp\!\Bigl(-\!\sum_{i\in{Hip}}(q_{i}-q^{\mathrm{glide}}_{i})^{2}\Bigr)$ |
| Pushing Critic Reward | Expression |
| Tracking linear velocity | $\delta_{\mathrm{push}}\exp\!\Bigl(-\tfrac{1}{\sigma}\bigl{\lVert}\mathbf{v}^{\mathrm{cmd}}_{x}-\mathbf{v}_{x}\bigr{\rVert}^{2}\Bigr)$ |
| Tracking angular velocity | $\delta_{\mathrm{push}}\exp\!\Bigl(-\tfrac{(\omega^{\mathrm{cmd}}_{z}-\omega_{z})^{2}}{\sigma_{\mathrm{yaw}}}\Bigr)$ |
| Hip positions | $\delta_{\mathrm{push}}\exp\!\Bigl(-\!\sum_{i\in{Hip}}(q_{i}-q^{\mathrm{push}}_{i})^{2}\Bigr)$ |
| Orientation | $$ |
| Sim2Real Critic Reward | Expression |
| Wheel contact number | (∑ i ∈ wheels c i = 4) i wheels c i 4 \;\! $\Bigl(\textstyle{\sum_{i\in\mathrm{wheels}}c_{i}}\;=\;4\Bigr)$ |
| Board–body height | exp  (- 4 \| (z body - z board) - 0.15 \|) 4 z body z board 0.15  $\exp\!\Bigl(-4\,\bigl{\lVert}\,(z_{\mathrm{body}}-z_{\mathrm{board}})\;-\;0.15\bigr{\rVert}\Bigr) (- 4 \lVert (-) - 0.15 \rVert)$ |
| Joint acceleration | $\sum_{i=1}^{12}\biggl{(}\frac{\mathrm{clip}\bigl(\dot{q}_{i}^{(t-1)}-\dot{q}_{i}^{(t)},\,{-10},\,10\bigr)}{\Delta t}\biggr{)}^{2} ∑ = 1 12 (divide ((- 1) - (), - 10, 10)) 2$ |
| Collisions | $\sum_{i\in\mathcal{P}}\!\Bigl(\lVert \mathbf{f}_{i} \rVert>0.1\Bigr) ∑ ∈$ |
| Action rate | $$ |
| Delta torques | $$ |
| Torques | $$ |
| Linear velocity (z-axis) | $$ |
| Angular velocity (x/y) | $$ |
| Base orientation | $$ |
| Cycle Calculation | Expression |
| Cycle | T T T |
| Phase | $\leftarrow\;\sin\!\bigl(2\pi\,t/T\bigr) ← (2 /)$ |
| Still Indicator | $\delta_{still}$ |
| Glide Indicator | $\delta_{\mathrm{glide}}^{(t)}\;=\;LPF\Bigl(\bigl{[}\phi<0.5\bigr{]}\;\lor\;\delta_{\mathrm{still}}\Bigr)$ |
| Push Indicator | $\delta_{\mathrm{push}}^{(t)}\;=\;LPF\Bigl(\bigl{[}\phi\geq 0.5\bigr{]}\;\land\;\neg\,\delta_{\mathrm{still}}\Bigr)$ |
| Low pass filter | L P F L P F LPF |

**说明**: 该表格来自论文中的公式、奖励项或实验设置；为避免 PDF 抽取噪声，网页中仅保留可渲染的核心符号和数值。

#### Table 3: TABLE III: Succes Rate Comparison: We deployed each method on a real robot to evaluate the success rate. Each method was

| Method | Ceramic | Carpet | Disturbance |
| --- | --- | --- | --- |
| Ours | 100% | 100% | 100% |
| Our-wo-MC(transfered) | 100% | 100% | 60% |
| Our-wo-MC | 60% | 60% | 40% |
| Ours-wo-Beta | ✗ | ✗ | ✗ |
| DreamWaq [27 ] | ✗ | ✗ | ✗ |
| Method | Slope | Single-step | Uneven |
| Ours | 80% | 100% | 60% |
| Our-wo-MC(transfered) | 60% | 40% | 60% |
| Our-wo-MC | 0% | 40% | 40% |
| Ours-wo-Beta | ✗ | ✗ | ✗ |
| DreamWaq [27 ] | ✗ | ✗ | ✗ |

**说明**: TABLE III: Succes Rate Comparison: We deployed each method on a real robot to evaluate the success rate. Each method was tested five times per scenarios. Success was defined as completing at least one full up-board and down-board motions, traverse a distance of more than 5 meters, and avoiding abrupt movements or detachment from the skateboard. ✗ indicates complete failure(Massive torque caused the joints protection state, for hardware protection, we first test torque value in simulation to make sure it will no exceed safty range).

#### Table 4: TABLE IV: Reward weights and Advantage weights for skateboarding environment design

| Gliding Critic Reward | Weight |
| --- | --- |
| Feet on board | 0.3 |
| Contact number | 0.3 |
| Feet distance | 1.8 |
| Joint positions | 1.2 |
| Hip positions | 1.2 |
| Pushing Critic Reward | Weight |
| Tracking linear velocity | 1.6 |
| Tracking angular velocity | 0.8 |
| Hip positions | 0.6 |
| Orientation | -2 |
| Sim2Real Critic Reward | Weight |
| Wheel contact number | 0.8 |
| Board–body height | 1 |
| Joint acceleration | -2.5e-7 |
| Collisions | -1 |
| Action rate | -0.22 |
| Delta torques | -1.0e-7 |
| Torques | -1.0e-5 |
| Linear velocity (z-axis) | -0.1 |
| Angular velocity (x/y) | -0.01 |
| Base orientation | -25 |
| Advantage | Weight |
| Gliding Critic Advantage | 0.35 |
| Pushing Critic Advantage | 0.4 |
| Sim2Real Critic Advantage | 0.25 |

**说明**: TABLE IV: Reward weights and Advantage weights for skateboarding environment design

#### Table 5: TABLE V: Network Architecture and Training Hyper-parameter

| Network Hyperparameters | value |
| --- | --- |
| DHA Architecture | MLP |
| DHA Hidden Dims | [256, 64, 32] |
| VAE Encoder Architecture | 1-D CNN |
| VAE Encoder time steps | 20 |
| VAE Encoder Convolutional Layers | Input channel = [30, 20] |
| VAE Encoder Convolutional Layers | Kernel=(6,4), Stride=(2,2) |
| VAE Decoder Hidden Dims | [256, 128, 64] |
| VAE Latent Dims | 20 |
| VAE KL Divergence Weight(  \beta) | 1e-2 |
| Actor Hidden Dims | [512, 256, 128] |
| Gliding Critic Hidden Dims | [512, 256, 128] |
| Pushing Critic Hidden Dims | [512, 256, 128] |
| Sim2Real Critic Hidden Dims | [512, 256, 128] |
| PPO HyperParameters | Weight |
| Environments | 4096 |
| Collection Steps | 24 |
| Discount Factor | 0.99 |
| GAE Parameter | 0.9 |
| Target KL Divergence | 0.01 |
| Learning Rate Schedule | adaptive |
| Number of Mini-batches | 4 |
| Clipping Paramete | 0.2 |

**说明**: TABLE V: Network Architecture and Training Hyper-parameter

#### Table 6: TABLE VI: Randomization and Noise

| Property Randomization | value |
| --- | --- |
| Friction | [0.6, 2.] |
| Added Mass | [0, 3]kg |
| Added COM | [-0.2, 0.2] |
| Push robot | 0.5m/s per 8s |
| Delay | [0, 20]ms |
| Sensor Noise | Weight |
| Euler Angle | N ∗ 0.08 N 0.08  $\mathcal{N}*0.08 ∗ 0.08$ |
| Angular Velocity | N ∗ 0.4 N 0.4  $\mathcal{N}*0.4 ∗ 0.4$ |
| Projected Gravity | N ∗ 0.05 N 0.05  $\mathcal{N}*0.05 ∗ 0.05$ |
| Joint Position | N ∗ 0.05 N 0.05  $\mathcal{N}*0.05 ∗ 0.05$ |
| Joint Velocity | N ∗ 0.1 N 0.1  $\mathcal{N}*0.1 ∗ 0.1$ |

**说明**: TABLE VI: Randomization and Noise

#### Table 7: TABLE VII: Reward weights for two single-critic method(w-transfer/ wo-transfer)

| Gliding Critic Reward | Weight | Weight(Transfer) |
| --- | --- | --- |
| Feet on board | 0.3 | 0.35 ∗ 0.3 0.35 0.3 0.35*0.3 0.35 ∗ 0.3 |
| Contact number | 0.3 | 0.35 ∗ 0.3 0.35 0.3 0.35*0.3 0.35 ∗ 0.3 |
| Feet distance | 1.8 | 0.35 ∗ 1.8 0.35 1.8 0.35*1.8 0.35 ∗ 1.8 |
| Joint positions | 1.2 | 0.35 ∗ 1.2 0.35 1.2 0.35*1.2 0.35 ∗ 1.2 |
| Hip positions | 1.2 | 0.35 ∗ 1.2 0.35 1.2 0.35*1.2 0.35 ∗ 1.2 |
| Pushing Critic Reward | Weight | Weight(Transfer) |
| Tracking linear velocity | 1.6 | 0.4 ∗ 1.6 0.4 1.6 0.4*1.6 0.4 ∗ 1.6 |
| Tracking angular velocity | 0.8 | 0.4 ∗ 0.8 0.4 0.8 0.4*0.8 0.4 ∗ 0.8 |
| Hip positions | 0.6 | 0.4 ∗ 0.6 0.4 0.6 0.4*0.6 0.4 ∗ 0.6 |
| Orientation | -2 | 0.4 ∗ - 2 0.4*-2 0.4 ∗ - 2 |
| Sim2Real Critic Reward | Weight | Weight(Transfer) |
| Wheel contact number | 0.8 | 0.25 ∗ 0.8 0.25 0.8 0.25*0.8 0.25 ∗ 0.8 |
| Board–body height | 1 | 0.25 ∗ 1 0.25 1 0.25*1 0.25 ∗ 1 |
| Joint acceleration | -2.5e-7 | 0.25 ∗ - 2.5 e - 7 0.25*-2.5e-7 0.25 ∗ - 2.5 - 7 |
| Collisions | -1 | 0.25 ∗ - 1 0.25*-1 0.25 ∗ - 1 |
| Action rate | -0.22 | 0.25 ∗ - 0.22 0.25*-0.22 0.25 ∗ - 0.22 |
| Delta torques | -1.0e-7 | 0.25 ∗ - 1.0 e - 7 0.25*-1.0e-7 0.25 ∗ - 1.0 - 7 |
| Torques | -1.0e-5 | 0.25 ∗ - 1.0 e - 5 0.25*-1.0e-5 0.25 ∗ - 1.0 - 5 |
| Linear velocity (z-axis) | -0.1 | 0.25 ∗ - 0.1 0.25*-0.1 0.25 ∗ - 0.1 |
| Angular velocity (x/y) | -0.01 | 0.25 ∗ - 0.01 0.25*-0.01 0.25 ∗ - 0.01 |
| Base orientation | -25 | 0.25 ∗ - 25 0.25*-25 0.25 ∗ - 25 |

**说明**: TABLE VII: Reward weights for two single-critic method(w-transfer/ wo-transfer)
## 实验解读

- 评价重点:围绕 agile-locomotion、接触推理、足式运动、移动操作、mobile-manipulation,优先看真实机器人结果、扰动恢复、接触稳定性、控制误差、成功率、实时性和 sim-to-real 差距。
- System0 视角:实验结论不只看最终成功率,还要看闭环过程中是否能处理状态估计误差、执行延迟、碰撞/接触突变和外部扰动。
- 与上层系统的关系:如果方法能在快闭环内稳定工作,它可以作为 System1 技能或 System2 规划的底层执行保障;如果只在离线或慢速设定下成立,则还不能直接作为身体级快闭环模块。
## 批判性思考

### 优点

- 论文把 agile-locomotion、接触推理、足式运动、移动操作、mobile-manipulation 放到机器人身体级闭环里讨论,和只做高层语义推理的工作相比,更接近真实机器人部署时会遇到的状态、动力学和安全约束。
- 若论文同时给出公式、系统结构图、硬件实验或消融结果,这些证据可以帮助判断方法是否真的能支撑毫秒到百毫秒级反应。

### 局限

- 需要重点核查控制频率、推理延迟、传感延迟、失败恢复、硬件磨损、未见场景泛化和安全边界;这些因素决定它能否成为可靠的 System0 模块。
- 对学习型方法,还要区分训练环境中的鲁棒性和真实部署中的鲁棒性,尤其是接触丰富任务、动态运动任务和人形机器人任务。
## 论文定位

- 研究对象:Discrete-Time Hybrid Automata Learning: Legged Locomotion Meets Skateboarding。
- 关键词:agile-locomotion、接触推理、足式运动、移动操作、mobile-manipulation、quadruped、强化学习、机器人操作、鲁棒控制。
- System0 关联:关注机器人身体级快反应链路中“状态/接触/局部感知 -> 约束/优化/策略 -> 动作”的短闭环,不把问题停留在语言规划或任务分解层。

## 速查卡片

> [!summary] Discrete Time Hybrid
> - **论文**: https://www.roboticsproceedings.org/rss21/p127.pdf
> - **arXiv**: http://arxiv.org/abs/2503.01842v2
> - **arXiv HTML**: https://arxiv.org/html/2503.01842v2
