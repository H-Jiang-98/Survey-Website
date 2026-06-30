---
title: "USER: A Unified and Extensible System for Online Real-World Policy Learning in Embodied AI"
method_name: "USER"
authors: ["Hongzhi Zang"]
year: 2026
venue: "RSS"
tags: ["contact-reasoning", "robust-control", "real-time-control", "reinforcement-learning", "adaptive-control", "robot-generalization", "state-estimation", "recovery"]
image_source: "online"
arxiv_html: "https://arxiv.org/html/2602.07837"
---
# USER
## 一句话总结

> USER: A Unified and Extensible System for Online Real-World Policy Learning in Embodied AI 主要落在 [[adaptive-control]]、[[接触推理]]、[[policy-learning]]、[[实时控制]] 这一类问题上;从 System0 视角看,它的价值在于把身体状态、接触变化、局部感知或动力学约束接到机器人低层闭环里,让动作生成更稳定、安全、实时。
## 核心贡献

1. 围绕 **USER: A Unified and Extensible System for Online Real-World Policy Learning in Embodied AI** 建立了一个与 adaptive-control、接触推理、policy-learning、实时控制、recovery、强化学习 相关的机器人问题设定,重点不是高层语义规划,而是身体级闭环中的状态变化、接触变化或控制约束。
2. 方法层面的关键价值在于把感知、动力学、约束优化、策略学习或硬件反馈组织成可执行的低层动作生成机制,适合放进 System0 的快反应模块中理解。
3. 对系统集成而言,这类工作补上了 System1 技能选择与电机级控制之间的一层:它负责把上层目标变成动态可行、可恢复、可安全过滤的动作。
4. 若论文包含真实机器人、接触任务或仿真到真实迁移,核心应看它是否报告了闭环稳定性、扰动恢复、硬件延迟、失败模式和跨场景泛化。
## 问题背景

这篇论文可以放在 System0 的“具身快系统”背景下阅读:机器人需要在很短时间内根据本体状态、接触、力/触觉、局部视觉或动力学预测做出动作,而不是等待慢速规划器重新推理整个任务。adaptive-control、接触推理、policy-learning、实时控制、recovery、强化学习、鲁棒控制 相关问题的难点通常不在“知道要做什么”,而在动作执行过程中状态会快速偏离、接触会突然改变、约束会实时收紧,低层系统必须及时修正。

### System0 关联

精读时优先看三件事:第一,论文使用哪些快速反馈量,例如关节状态、接触状态、力觉、触觉、深度或局部视觉;第二,反馈如何进入 MPC、WBC、策略网络、安全过滤器或恢复控制器;第三,输出是否直接影响机器人下一步动作,而不是只服务离线分析或高层任务规划。
## 方法详解

### System0 视角解析

从 System0 角度看,这篇论文的方法部分应当围绕 adaptive-control、接触推理、policy-learning、实时控制、recovery、强化学习、鲁棒控制、scalable-robot-learning 来读。核心问题是:系统如何从快速变化的机器人状态中构造控制输入,如何处理接触、碰撞、约束、模型误差或执行延迟,以及如何把这些信息变成下一时刻的动作。

如果方法里有优化问题,重点看目标函数、约束和求解频率;如果方法里有学习策略,重点看观测空间、动作空间、训练扰动和部署时的闭环频率;如果方法里有硬件或传感器,重点看传感链路是否真的进入控制回路。这样读可以把论文和 System2 语言规划、System1 技能编排区分开来。

### 关键公式与机制

#### 公式 1: [[优化目标/约束]]

$$J(\pi)=\sum_{t=0}^{T}\mathbb{E}_{(o_{t},a_{t})\sim\rho_{\pi}}\big[r(o_{t},a_{t})+\alpha\mathcal{H}(\pi(\cdot\mid o_{t}))\big],$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 2: [[优化目标/约束]]

$$L_{\pi}(\theta)=\mathbb{E}_{o\sim\mathcal{B},\,a\sim\pi_{\theta}(\cdot\mid o)}\big[\alpha\log\pi_{\theta}(a\mid o)-Q_{\psi}(o,a)\big].$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 3: [[优化目标/约束]]

$$A_{t_{i+1}}=A_{t_{i}}+\Delta t_{i}\,v_{\theta}(t_{i},A_{t_{i}},o),\quad A_{t_{0}}\sim\mathcal{N}(0,I).$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 4: [[优化目标/约束]]

$$A_{t_{i+1}}=A_{t_{i}}+v_{\theta}(t_{i},A_{t_{i}},o)\Delta t_{i}+\sigma\sqrt{\Delta t_{i}}\epsilon_{i},\quad\epsilon_{i}\sim\mathcal{N}(0,I),$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 5: [[优化目标/约束]]

$$L_{\pi}(\theta)=\mathbb{E}_{\mathcal{A}\sim\pi_{\theta}}\big[\alpha\log p_{c}(\mathcal{A}\mid o)-Q_{\psi}(o,\tanh(A_{t_{K}}))\big],$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 6: [[动力学或策略机制]]

$$a_{t}=\begin{cases}a_{t}^{\text{human}},&\text{if human intervention is active},\\ \pi_{\theta}(o_{t}),&\text{otherwise}.\end{cases}$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 7: [[优化目标/约束]]

$$
\mathcal{L}_{BC}(\theta)=\mathbb{E}_{(o,a^{h})\sim\mathcal{D}_{\text{intervene}}\cup\mathcal{D}_{\text{demo}}}\big[\lVert \pi_{\theta}(o)-a^{h} \rVert^{2}\big].
$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 8: [[优化目标/约束]]

$$Q_{target}=r+\gamma(\min_{j=1,\ldots,M}Q_{\bar{\psi}_{j}}(s^{\prime},a^{\prime})-\alpha\log\pi_{\theta}(a^{\prime}|s^{\prime}))$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 9: [[优化目标/约束]]

$$L_{Q}(\psi)=\mathbb{E}_{(o,a,r,o^{\prime})\sim\mathcal{B}}\Big[\big(Q_{\psi}(o,a)$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 10: [[动力学或策略机制]]

$$(r+\gamma\mathbb{E}_{a^{\prime}\sim\pi_{\theta}(\cdot\mid o^{\prime})}[Q_{\bar{\psi}}(o^{\prime},a^{\prime})-\alpha\log\pi_{\theta}(a^{\prime}\mid o^{\prime})])\big)^{2}\Big],$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。
## 关键图表

### Figure 1: Overview of learning framework design: a fully asynchronous real-world learning pipeli

![Figure 1](https://arxiv.org/html/2602.07837/x3.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Overview of learning framework design: a fully asynchronous real-world learning pipeli”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 2: Parallel training on two Franka robot arms. The unified hardware layer enables distri

![Figure 2](https://arxiv.org/html/2602.07837/x7.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Parallel training on two Franka robot arms. The unified hardware layer enables distri”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 3: Parallel training on two Franka robot arms. The unified hardware layer enables distri

![Figure 3](https://arxiv.org/html/2602.07837/figures/2arms.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Parallel training on two Franka robot arms. The unified hardware layer enables distri”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### 论文表格

#### Table 1: TABLE I: Experiment setting summary.

| Task Name | Model | Algorithm | Reward Type | Demo (trajs) |
| --- | --- | --- | --- | --- |
| Peg Insertion | CNN | SAC | rule-based, dense | / |
| RLPD | rule-based, sparse | 20 | | |
| Flow | SAC Flow | | | |
| Charger | CNN | SAC | rule-based, dense | / |
| RLPD | rule-based, sparse | 20 | | |
| Flow | SAC Flow | rule-based, sparse | | |
| Cap Tightening | CNN | RLPD | human reward, sparse | 20 |
| Pick-and-Place | CNN | RLPD | human-reward, sparse | 40 |
| $\pi_{0}$ | HG-DAgger | / | 40 | |
| Table Clean-up | $\pi_{0}$ | HG-DAgger | / | 40 |

**说明**: TABLE I: Experiment setting summary.

#### Table 2: TABLE II: USER enables significant performance gains after online training for foundation VLA models.

| | Before online training | After online training |
| --- | --- | --- |
| Pick-and-Place | 39/60 | 58/60 |
| Table Clean-up | 9/20 | 16/20 |

**说明**: TABLE II: USER enables significant performance gains after online training for foundation VLA models.

#### Table 3: TABLE III: Communication performance of distributed channels cross-domain and same-domain network settings. Enabli

| Domain | Distributed | Rollout (s/chunk) | Interact (s/chunk) | Send Obs (s/data) | Send Action (s/data) | Total Generation Time ↓ \downarrow (s/episode) | |
| --- | --- | --- | --- | --- | --- | --- | --- |
| cross | w/ | 0.002(± \pm 0.000) | 0.106(± \pm 0.001) | 0.042 (± \pm 0.031) | 0.070 (± \pm 0.017) | 21.979 (± \pm 0.435) | |
| w/o | 0.002(± \pm 0.000) | 0.107(± \pm 0.001) | 0.671(± \pm 0.012) | 0.270(± \pm 0.009) | 69.265(± \pm 1.905) | | |
| same | w/ | 0.006(± \pm 0.001) | 0.106(± \pm 0.002) | 0.025 (± \pm 0.006) | 0.028 (± \pm 0.008) | 17.304 (± \pm 0.001) | |
| w/o | 0.007(± \pm 0.001) | 0.106(± \pm 0.002) | 0.169(± \pm 0.018) | 0.021(± \pm 0.008) | 18.696(± \pm 0.710) | | |

**说明**: 该表格来自论文中的公式、奖励项或实验设置；为避免 PDF 抽取噪声，网页中仅保留可渲染的核心符号和数值。

#### Table 4: TABLE IV: Profiling results of synchronous and asynchronous training pipelines. Generation and training periods denote t

| | | Rollout (s/chunk) | Interact (s/chunk) | Send data | Train (s/update) | Sync weights | Generation Period ↓ \downarrow (s/episode) | Training Period ↓ \downarrow (s/update) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| $\pi_{0} + HG-Dagger$ | Sync | 0.214(± \pm 0.001) | 1.093(± \pm 0.013) | 0.606(± \pm 0.028) | 6.128(± \pm 0.247) | 0.816(± \pm 0.024) | 45.068(± \pm 0.304) | 45.011(± \pm 0.251) |
| Async | 0.213(± \pm 0.016) | 1.091(± \pm 0.010) | 7.903(± \pm) | 5.969(± \pm 0.196) | 0.789(± \pm 0.052) | 37.538(± \pm 0.363) | 7.903(± \pm 0.234) | |
| Speed Up | / | / | / | / | / | 1.20  \times | 5.70  \times | |
| CNN + SAC | Sync | 0.006(± \pm 0.002) | 0.108(± \pm 0.002) | 0.144(± \pm 0.008) | 0.108(± \pm 0.002) | 0.162(± \pm 0.004) | 20.291(± \pm 0.632) | 0.643(± \pm 2.984) |
| Async | 0.004(± \pm 0.004) | 0.107(± \pm 0.002) | 0.004(± \pm 0.001) | 0.123(± \pm 0.005) | 0.174(± \pm 0.003) | 13.108(± \pm 0.218) | 0.135(± \pm 0.034) | |
| Speed Up | / | / | / | / | / | 1.55  \times | 4.61  \times | |

**说明**: TABLE IV: Profiling results of synchronous and asynchronous training pipelines. Generation and training periods denote the intervals between consecutive episode generation and training executions, respectively.

#### Table 5: TABLE V: Hyperparameters of SAC

| Parameter | Value |
| --- | --- |
| Critic Network | ResNet10 encoder (shared) + MLP |
| Actor Network | ResNet10 encoder (shared) + MLP |
| Buffer size | 20000 |
| Train start step | 200 |
| Discount Factor $\gamma)$ | 0.96 |
| Batch Size | 256 |
| Critic learning rate | 3e-4 |
| Actor learning rate | 3e-4 |
|  \alpha learning rate | 3e-4 |
| Target Entropy | -3 |
| Temperature ( \alpha) | Auto tune (init 0.01) |
| $\tau$ | 0.005 |
| Target update frequency | 1 |
| Actor update frequency | 4 |
| Weight sync. frequency | 32 |

**说明**: TABLE V: Hyperparameters of SAC

#### Table 6: TABLE VI: Hyperparameters of SAC-Flow

| Parameter | Value |
| --- | --- |
| Critic Network | ResNet10 encoder (shared) + MLP |
| Actor Network | ResNet10 encoder (shared) + Flow-T |
| Denoising Steps (N N) | 4 |
| Decoder dimension | 256 |
| Attention head | 4 |
| Decoder layer | 2 |
| Log std range | [-5, 2] |
| Buffer size | 20000 |
| Train start step | 200 |
| Discount Factor $\gamma)$ | 0.96 |
| Batch Size | 256 |
| Critic learning rate | 3e-4 |
| Actor learning rate | 3e-4 |
|  \alpha learning rate | 3e-4 |
| Target Entropy | -3 |
| Temperature ( \alpha) | Auto tune (init 0.01) |
| $\tau$ | 0.005 |
| Target update frequency | 1 |
| Actor update frequency | 4 |
| Weight sync. frequency | 30 |

**说明**: TABLE VI: Hyperparameters of SAC-Flow

#### Table 7: TABLE VII: Hyperparameters of RLPD

| Parameter | Value |
| --- | --- |
| Critic Network | ResNet10 encoder (shared) + MLP |
| Actor Network | ResNet10 encoder (shared) + MLP |
| Buffer size | 20000 |
| Train start step | 200 |
| Critic ensemble size | 10 |
| Critic sub-sample size | 2 |
| Demo sampling ratio | 50% |
| Discount Factor $\gamma)$ | 0.96 |
| Batch Size | 256 |
| Critic learning rate | 3e-4 |
| Actor learning rate | 3e-4 |
|  \alpha learning rate | 3e-4 |
| Target Entropy | -3 |
| Temperature ( \alpha) | Auto tune (init 0.01) |
| $\tau$ | 0.005 |
| Target update frequency | 1 |
| Actor update frequency | 4 |
| Weight sync. frequency | 32 |

**说明**: TABLE VII: Hyperparameters of RLPD

#### Table 8: TABLE VIII: Hyperparameters of HG-DAgger

| Parameter | Value |
| --- | --- |
| Network | $\pi_{0} (∼ \sim 3B)$ |
| Action chunk size | 10 |
| SFT learning rate | 2.5e-5 |
| SFT decay learning rate | 2.5e-6 |
| SFT learning rate scheduler | cosine |
| SFT weight decay | 1e-10 |
| SFT batch size | 1e-5 |
| SFT epoch | 1e-5 |
| HG-DAgger learning rate | 1e-5 |
| HG-DAgger batch size | 64 |
| HG-DAgger weight decay | 1e-8 |
| Intervention sampling ratio | 50% |
| Weight sync. frequency | 1 |

**说明**: TABLE VIII: Hyperparameters of HG-DAgger

#### Table 9: TABLE IX: Detailed settings of peg insertion

| Parameter | Value |
| --- | --- |
| Image shape | 1  \times (3, 128, 128) |
| State dim | 19 |
| Obs coordinate | Reset pose |
| Action space | Delta EE pose (6 dim) |
| Step frequency | 10 Hz |
| Max episode length | 100 steps |
| Bounding box (xyz) | [0.05, 0.05, 0.1] (m) |
| Bounding box (rpy) | $\pi /6 (rad)$ |
| RLPD Specific: | |
| Demos | 20 |
| Demo Sampling Ratio | 50% |
| Reward | Rule-based, sparse |
| SAC Specific: | |
| Demos | / |
| Reward | Rule-based, dense |
| SAC Flow Specific: | |
| Demos | 20 |
| Reward | Rule-based, sparse |

**说明**: TABLE IX: Detailed settings of peg insertion

#### Table 10: TABLE X: Detailed settings of charger plugging

| Parameter | Value |
| --- | --- |
| Image shape | 1  \times (3, 128, 128) |
| State dim | 19 |
| Obs coordinate | Reset pose |
| Action space | Delta EE pose (6 dim) |
| Step frequency | 10 Hz |
| Max episode length | 100 |
| Bounding box (xyz) | [0.02, 0.02, 0.055] (m) |
| Bounding box (rpy) | $\pi /9 (rad)$ |
| RLPD Specific: | |
| Demos | 20 |
| Demo Sampling Ratio | 50% |
| Reward | Rule-based, sparse |
| SAC Specific: | |
| Demos | / |
| Reward | Rule-based, dense |
| SAC Flow Specific: | |
| Demos | 20 |
| Reward | Rule-based, sparse |

**说明**: TABLE X: Detailed settings of charger plugging

#### Table 11: TABLE XI: Detailed settings of cap tightening

| Parameter | Value |
| --- | --- |
| Image shape | 2  \times (3, 128, 128) |
| State dim | 20 |
| Obs coordinate | Target pose |
| Action space | Delta EE pose + gripper (7 dim) |
| Step frequency | 10 Hz |
| Max episode length | 240 |
| Bounding box (xyz) | [0.01, 0.01, 0.02] (m) |
| Bounding box (rpy) | $\pi /6 (rad)$ |
| RLPD Specific: | |
| Demos | 20 |
| Demo Sampling Ratio | 50% |
| Reward | Human-provided, sparse |

**说明**: TABLE XI: Detailed settings of cap tightening

#### Table 12: TABLE XII: Deatiled settings of Pick-and-Place

| Parameter | Value |
| --- | --- |
| Image shape | 2  \times (3, 128, 128) |
| State dim | 20 |
| Obs coordinate | Reset pose |
| Action space | Delta EE pose + gripper (7 dim) |
| Step frequency | 10 Hz |
| Max episode length | 240 |
| Bounding box (xyz) | [0.3, 0.3, 0.15] (m) |
| Bounding box (rpy) | $\pi /6 (rad)$ |
| RLPD Specific: | |
| Demos | 40 |
| Demo Sampling Ratio | 50% |
| Reward | Human-provided, sparse |
| HG-DAgger Specific: | |
| Demos | 40 |
| SFT epoch | 5000 |

**说明**: TABLE XII: Deatiled settings of Pick-and-Place

#### Table 13: TABLE XIII: Policy training details for Table Clean-up (HG-DAgger).

| Parameter | Value |
| --- | --- |
| Image shape | 2  \times (3, 128, 128) |
| State dim | 20 |
| Obs coordinate | Reset pose |
| Action space | Delta EE pose + gripper (7 dim) |
| Step frequency | 10 Hz |
| Max episode length | 360 |
| Bounding box (xyz) | [0.4, 0.4, 0.25] (m) |
| Bounding box (rpy) | $\pi /3$ |
| HG-DAgger Specific: | |
| Demos | 40 |
| SFT epoch | 15000 |

**说明**: TABLE XIII: Policy training details for Table Clean-up (HG-DAgger).

#### Table 14: TABLE XIV: Parameters of the impedance controller

| Translational Parameter | Value | Rotational Parameter | Value |
| --- | --- | --- | --- |
| Stiffness | 2500 | Stiffness | 150 |
| Damping | 100 | Damping | 7 |
| K i K_{i} | 0 | K i K_{i} | 0 |
| Clip x | [-0.007, 0.007] | Clip x | [-0.07, 0.07] |
| Clip y | [-0.007, 0.007] | Clip y | [-0.07, 0.07] |
| Clip z | [-0.007, 0.007] | Clip z | [-0.06, 0.06] |

**说明**: TABLE XIV: Parameters of the impedance controller
## 实验解读

- 评价重点:围绕 adaptive-control、接触推理、policy-learning、实时控制、recovery,优先看真实机器人结果、扰动恢复、接触稳定性、控制误差、成功率、实时性和 sim-to-real 差距。
- System0 视角:实验结论不只看最终成功率,还要看闭环过程中是否能处理状态估计误差、执行延迟、碰撞/接触突变和外部扰动。
- 与上层系统的关系:如果方法能在快闭环内稳定工作,它可以作为 System1 技能或 System2 规划的底层执行保障;如果只在离线或慢速设定下成立,则还不能直接作为身体级快闭环模块。
## 批判性思考

### 优点

- 论文把 adaptive-control、接触推理、policy-learning、实时控制、recovery 放到机器人身体级闭环里讨论,和只做高层语义推理的工作相比,更接近真实机器人部署时会遇到的状态、动力学和安全约束。
- 若论文同时给出公式、系统结构图、硬件实验或消融结果,这些证据可以帮助判断方法是否真的能支撑毫秒到百毫秒级反应。

### 局限

- 需要重点核查控制频率、推理延迟、传感延迟、失败恢复、硬件磨损、未见场景泛化和安全边界;这些因素决定它能否成为可靠的 System0 模块。
- 对学习型方法,还要区分训练环境中的鲁棒性和真实部署中的鲁棒性,尤其是接触丰富任务、动态运动任务和人形机器人任务。
## 论文定位

- 研究对象:USER: A Unified and Extensible System for Online Real-World Policy Learning in Embodied AI。
- 关键词:adaptive-control、接触推理、policy-learning、实时控制、recovery、强化学习、鲁棒控制、scalable-robot-learning、状态估计、vision-language-action。
- System0 关联:关注机器人身体级快反应链路中“状态/接触/局部感知 -> 约束/优化/策略 -> 动作”的短闭环,不把问题停留在语言规划或任务分解层。

## 速查卡片

> [!summary] USER
> - **论文**: https://www.roboticsproceedings.org/rss22/p037.pdf
> - **arXiv**: http://arxiv.org/abs/2602.07837
> - **arXiv HTML**: https://arxiv.org/html/2602.07837
