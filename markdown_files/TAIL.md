---
title: "TAIL-Safe: Task-Agnostic Safety Monitoring for Imitation Learning Policies"
method_name: "TAIL-Safe"
authors: ["Riad Ahmed"]
year: 2026
venue: "RSS"
tags: ["contact-reasoning", "robot-manipulation", "reinforcement-learning", "safe-control", "imitation-learning", "recovery"]
image_source: "online"
arxiv_html: "https://arxiv.org/html/2605.01195v2"
---
# TAIL-Safe
## 一句话总结

> TAIL-Safe: Task-Agnostic Safety Monitoring for Imitation Learning Policies 主要落在 [[certified-control]]、[[接触推理]]、[[diffusion-policy]]、[[grasping]] 这一类问题上;从 System0 视角看,它的价值在于把身体状态、接触变化、局部感知或动力学约束接到机器人低层闭环里,让动作生成更稳定、安全、实时。
## 核心贡献

1. 围绕 **TAIL-Safe: Task-Agnostic Safety Monitoring for Imitation Learning Policies** 建立了一个与 certified-control、接触推理、diffusion-policy、grasping、模仿学习、recovery 相关的机器人问题设定,重点不是高层语义规划,而是身体级闭环中的状态变化、接触变化或控制约束。
2. 方法层面的关键价值在于把感知、动力学、约束优化、策略学习或硬件反馈组织成可执行的低层动作生成机制,适合放进 System0 的快反应模块中理解。
3. 对系统集成而言,这类工作补上了 System1 技能选择与电机级控制之间的一层:它负责把上层目标变成动态可行、可恢复、可安全过滤的动作。
4. 若论文包含真实机器人、接触任务或仿真到真实迁移,核心应看它是否报告了闭环稳定性、扰动恢复、硬件延迟、失败模式和跨场景泛化。
## 问题背景

这篇论文可以放在 System0 的“具身快系统”背景下阅读:机器人需要在很短时间内根据本体状态、接触、力/触觉、局部视觉或动力学预测做出动作,而不是等待慢速规划器重新推理整个任务。certified-control、接触推理、diffusion-policy、grasping、模仿学习、recovery、强化学习 相关问题的难点通常不在“知道要做什么”,而在动作执行过程中状态会快速偏离、接触会突然改变、约束会实时收紧,低层系统必须及时修正。

### System0 关联

精读时优先看三件事:第一,论文使用哪些快速反馈量,例如关节状态、接触状态、力觉、触觉、深度或局部视觉;第二,反馈如何进入 MPC、WBC、策略网络、安全过滤器或恢复控制器;第三,输出是否直接影响机器人下一步动作,而不是只服务离线分析或高层任务规划。
## 方法详解

### System0 视角解析

从 System0 角度看,这篇论文的方法部分应当围绕 certified-control、接触推理、diffusion-policy、grasping、模仿学习、recovery、强化学习、机器人操作 来读。核心问题是:系统如何从快速变化的机器人状态中构造控制输入,如何处理接触、碰撞、约束、模型误差或执行延迟,以及如何把这些信息变成下一时刻的动作。

如果方法里有优化问题,重点看目标函数、约束和求解频率;如果方法里有学习策略,重点看观测空间、动作空间、训练扰动和部署时的闭环频率;如果方法里有硬件或传感器,重点看传感链路是否真的进入控制回路。这样读可以把论文和 System2 语言规划、System1 技能编排区分开来。

### 关键公式与机制

#### 公式 1: [[优化目标/约束]]

$$s_{rec}(s)=\sigma\left(-d_{M}(\phi(o),\mu_{\mathcal{D}},\Sigma_{\mathcal{D}})\right)$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 2: [[优化目标/约束]]

$$s_{grasp}(s,a)=\max_{g\in\mathcal{G}}\;\text{sim}(T_{ee}(s,a),T_{g})$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 3: [[优化目标/约束]]

$$Q(s,a)=\mathbb{E}_{\pi}\left[\min_{k\geq 0}\gamma^{k}\mathcal{R}(s_{t+k},a_{t+k})\mid s_{t}=s,a_{t}=a\right]$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 4: [[优化目标/约束]]

$$\mathcal{C}_{safe}=\{(s,a)\in\mathcal{S}\times\mathcal{A}\mid Q(s,a)\geq 0\}$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 5: [[优化目标/约束]]

$$Q^{\pi}(s_{t},a_{t})=\min\left(h(s_{t},a_{t}),\;\gamma\cdot Q^{\pi}(s_{t+1},\pi(s_{t+1}))\right)$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 6: [[优化目标/约束]]

$$
\mathcal{L}_{hill}=\mathbb{E}_{(s,a^{*})\sim\mathcal{D}}\left[\max(0,Q(s,a)-(1-\alpha\lVert a-a^{*} \rVert_{2}))\right]
$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 7: [[优化目标/约束]]

$$
\mathcal{L}_{Q}=\mathcal{L}_{anchor}+\lambda_{hill}\mathcal{L}_{hill}
$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 8: [[优化目标/约束]]

$$a^{(k+1)}=\text{Proj}_{\mathcal{A}}\left(a^{(k)}+\eta\frac{\nabla_{a}Q(s_{t},a^{(k)})}{\lVert \nabla_{a}Q(s_{t},a^{(k)})\rVert_{2}}\right)$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 9: [[优化目标/约束]]

$$y_{t}=\min\left(h(s_{t},a_{t}),\;\gamma\cdot y_{t+1}\right),\quad y_{T}=\begin{cases}+1&\text{task success}\\ -1&\text{task failure}\end{cases}$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 10: [[优化目标/约束]]

$$
\mathcal{L}_{anchor}=\mathbb{E}[(Q_{\phi}(s_{t},a_{t})-y_{t})^{2}]
$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。
## 关键图表

### Figure 1: Pipeline / core system figure: Q-Function Landscape. Multi-view 2D projections show the bounded safe set (green, Q ≥

![Figure 1](https://arxiv.org/html/2605.01195v2/x1.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Pipeline / core system figure: Q-Function Landscape. Multi-view 2D projections show the bounded safe set (green, Q ≥”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 2: Safe Set and Q-Value Propagation. (a) Without TAIL-Safe: Q-value remains negative, ta

![Figure 2](https://arxiv.org/html/2605.01195v2/x9.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Safe Set and Q-Value Propagation. (a) Without TAIL-Safe: Q-value remains negative, ta”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 3: WeightNet Learned Weights Across Trajectory Phases. Dynamic weight distribution acro

![Figure 3](https://arxiv.org/html/2605.01195v2/x13.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“WeightNet Learned Weights Across Trajectory Phases. Dynamic weight distribution acro”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### 论文表格

#### Table 1: TABLE I: Task Performance: Simulation and Real Robot

| Task | Condition | N | Success | Steps | Recov. |
| --- | --- | --- | --- | --- | --- |
| Simulation | | | | | |
| Candy Pick | Unsafe (No Safety) | 200 | 20.0% | 94.2 ± \pm 7.8 | — |
| Safe (With Safety) | 200 | 100% | 74.3 ± \pm 5.9 | 0.63 ± \pm 0.7 | |
| Pick-Place | Unsafe (No Safety) | 200 | 23.3% | 92.8 ± \pm 8.6 | — |
| Safe (With Safety) | 200 | 100% | 76.9 ± \pm 6.8 | 0.70 ± \pm 0.9 | |
| Real Robot | | | | | |
| Candy Pick | Unsafe (No Safety) | 50 | 25.0% | 91.5 ± \pm 9.1 | — |
| Safe (With Safety) | 50 | 100% | 78.2 ± \pm 7.3 | 0.80 ± \pm 0.8 | |
| Pick-Place | Unsafe (No Safety) | 50 | 20.0% | 95.3 ± \pm 8.4 | — |
| Safe (With Safety) | 50 | 100% | 81.4 ± \pm 7.9 | 0.85 ± \pm 1.0 | |

**说明**: TABLE I: Task Performance: Simulation and Real Robot

#### Table 2: TABLE II: WeightNet vs Equal Weights for Safety Detection. WeightNet achieves near-perfect detection at both trajectory

| | Trajectory-Level | State-Level | | |
| --- | --- | --- | --- | --- |
| Method | Safe Recall | Unsafe Recall | Safe Recall | Unsafe Recall |
| WeightNet | 100.0% | 98.3% | 98.4% | 100.0% |
| Equal Weights | 0.0% | 100.0% | 1.9% | 100.0% |

**说明**: TABLE II: WeightNet vs Equal Weights for Safety Detection. WeightNet achieves near-perfect detection at both trajectory and state levels, while Equal Weights fails to identify safe regions (0% trajectory, 1.9% state recall). See Table VII for full details.

#### Table 3: TABLE III: Baseline comparison on 480 episodes (59,779 state-action pairs). Per-step detection (AUROC), episode-level d

| Method | AUROC | Ep. AUC | Flat ∇ \nabla † | Recov. ‡ | Per-crit. | Latency |
| --- | --- | --- | --- | --- | --- | --- |
| Ensemble [15, 10 ] | 0.525 | 0.525 | N/A a | 0.0% b |  \times | 15.0 ms |
| Learned CBF [27 ] | 0.987 | 0.989 | 97.4% | 6.9% |  \times | ∼ {\sim} 3 ms |
| h  (s, a) h(s,a) only | 0.999 | 0.999 | N/A c | N/A c | ✓ \checkmark | ∼ {\sim} 1 ms |
| TAIL-Safe Q  (s, a) Q(s,a) | 0.999 | 1.000 | 1.1% | 100% | ✓ \checkmark | 2.8 ms |

**说明**: TABLE III: Baseline comparison on 480 episodes (59,779 state-action pairs). Per-step detection (AUROC), episode-level detection (Ep. AUC), fraction of near-zero recovery gradients (Flat ∇ $\nabla)$, oracle-true recovery rate, per-criterion attribution, and inference latency.

#### Table 4: TABLE IV: Ablation: Effect of Lipschitz and Energy Shaping on Recovery. We measure gradient-based recovery success from

| Method | Recovery Success (%) | Mean Gradient ∇ Q \nabla Q | Avg. Step to Correct |
| --- | --- | --- | --- |
| 1. No Constraints | 20 | 0.036 | 170 |
| 2. + Lipschitz | 35 | 0.041 | 67 |
| ⌞ \llcorner Improvement | +15% | | |
| 3. + Energy | 100 | 0.802 | 2 |
| ⌞ \llcorner Improvement | +65% | | |

**说明**: TABLE IV: Ablation: Effect of Lipschitz and Energy Shaping on Recovery. We measure gradient-based recovery success from 200 unsafe initial states. Progressive Improvement Analysis (Max Step number = 200, Step size = 0.2)

#### Table 5: TABLE V: Q-Function Calibration Metrics

| Metric | Value | Interpretation |
| --- | --- | --- |
| AUROC | 99.3% | Excellent Discrimination |
| AUPRC | 99.7% | High Precision-Recall |
| False Safe Rate | 0.84% | Critical Safety Metric |
| False Unsafe Rate | 1.58% | Conservatism Penalty |
| ECE | 0.37 | Calibration Error |

**说明**: TABLE V: Q-Function Calibration Metrics

#### Table 6: TABLE VI: FlowPolicy Training Hyperparameters

| Parameter | Value | Parameter | Value |
| --- | --- | --- | --- |
| Architecture | | | |
| Point cloud input | 8192 pts  \times 6 | Diffusion embed dim | 128 |
| State dimension | 21 (proprio.) | Kernel size | 5 |
| Action dimension | 7 (xyz, rot, grip) | Conditioning | FiLM (global) |
| Encoder output dim | 64 | PointNet type | MLP + LayerNorm |
| Down dims | [256, 512, 1024] | | |
| CFM Parameters | | | |
| Start time  \epsilon | 5  10 - 3 5\!\times\!10^{-3} | Num segments | 3 |
| $\delta$ | 0.7 | Inference steps | 50 |
| Velocity weight  \alpha | 0.8 | Integration | RK4 |
| Training | | | |
| Optimizer | AdamW | Batch size | 32 |
| Learning rate | 5  10 - 4 5\!\times\!10^{-4} | Epochs | 300 |
| Weight decay | 10 - 5 10^{-5} | LR scheduler | Cosine + warmup |
| EMA power | 0.75 | Early-stop patience | 10 epochs |
| Loss Weights / Normalization / Dataset | | | |
| CFM loss | 1.0 | Translation (xyz) | [- 1, 1 ] [-1,1] min-max |
| Multi-step consistency | 0.5 | Rotation (rxryrz) | Identity |
| Velocity regularization | 0.5 | Gripper | [0, 1 ] [0,1] binary |
| Action MSE supervision | 0.1 | Horizon / obs / act steps | 4 / 2 / 4 |
| Train/val split | 90% / 10% | | |

**说明**: TABLE VI: FlowPolicy Training Hyperparameters

#### Table 7: TABLE VII: Ablation Study — Ground Truth Distribution and Trajectory-Level Prediction (WeightNet vs Equal Weights).

| Ground Truth Distribution | | | | |
| --- | --- | --- | --- | --- |
| Trajectories: 270 | States: 26,977 | | | |
| 152 safe / 118 unsafe | 8,133 safe / 18,844 unsafe | | | |
| H-Score Trajectory Prediction Performance | | | | |
| Method | Acc. | AUROC | Safe | Unsafe |
| WeightNet h  (reward) h( $\text{reward})$ | 99.3% | 100.0% | 152/152 | 116/118 |
| Equal Weights h  (reward) h( $\text{reward})$ | 43.7% | 93.5% | 0/152 | 118/118 |

**说明**: TABLE VII: Ablation Study — Ground Truth Distribution and Trajectory-Level Prediction (WeightNet vs Equal Weights).

#### Table 8: TABLE VIII: Ablation Study — Q-Function State Labeling (Recall).

| Method | Correct Safe | Correct Unsafe |
| --- | --- | --- |
| WeightNet-based Q Q | 8,005/8,133 (98.4%) | 18,844/18,844 (100.0%) |
| Equal Weights Q Q | 154/8,133 (1.9%) | 18,844/18,844 (100.0%) |

**说明**: TABLE VIII: Ablation Study — Q-Function State Labeling (Recall).

#### Table 9: TABLE IX: Ablation Study — Q-Function Prediction Quality (Precision).

| Method | Labeled | Correct | Precision |
| --- | --- | --- | --- |
| WeightNet Safe (Q > 0 Q\!>\!0) | 8,005 | 8,005/8,005 | 100.0% |
| WeightNet Unsafe (Q < 0 Q\!<\!0) | 18,972 | 18,844/18,972 | 99.3% |
| Equal Weights Safe (Q > 0 Q\!>\!0) | 154 | 154/154 | 100.0% |
| Equal Weights Unsafe (Q < 0 Q\!<\!0) | 26,823 | 18,844/26,823 | 70.3% |

**说明**: TABLE IX: Ablation Study — Q-Function Prediction Quality (Precision).
## 实验解读

- 评价重点:围绕 certified-control、接触推理、diffusion-policy、grasping、模仿学习,优先看真实机器人结果、扰动恢复、接触稳定性、控制误差、成功率、实时性和 sim-to-real 差距。
- System0 视角:实验结论不只看最终成功率,还要看闭环过程中是否能处理状态估计误差、执行延迟、碰撞/接触突变和外部扰动。
- 与上层系统的关系:如果方法能在快闭环内稳定工作,它可以作为 System1 技能或 System2 规划的底层执行保障;如果只在离线或慢速设定下成立,则还不能直接作为身体级快闭环模块。
## 批判性思考

### 优点

- 论文把 certified-control、接触推理、diffusion-policy、grasping、模仿学习 放到机器人身体级闭环里讨论,和只做高层语义推理的工作相比,更接近真实机器人部署时会遇到的状态、动力学和安全约束。
- 若论文同时给出公式、系统结构图、硬件实验或消融结果,这些证据可以帮助判断方法是否真的能支撑毫秒到百毫秒级反应。

### 局限

- 需要重点核查控制频率、推理延迟、传感延迟、失败恢复、硬件磨损、未见场景泛化和安全边界;这些因素决定它能否成为可靠的 System0 模块。
- 对学习型方法,还要区分训练环境中的鲁棒性和真实部署中的鲁棒性,尤其是接触丰富任务、动态运动任务和人形机器人任务。
## 论文定位

- 研究对象:TAIL-Safe: Task-Agnostic Safety Monitoring for Imitation Learning Policies。
- 关键词:certified-control、接触推理、diffusion-policy、grasping、模仿学习、recovery、强化学习、机器人操作、safe-control。
- System0 关联:关注机器人身体级快反应链路中“状态/接触/局部感知 -> 约束/优化/策略 -> 动作”的短闭环,不把问题停留在语言规划或任务分解层。

## 速查卡片

> [!summary] TAIL-Safe
> - **论文**: https://www.roboticsproceedings.org/rss22/p207.pdf
> - **arXiv**: http://arxiv.org/abs/2605.01195v2
> - **arXiv HTML**: https://arxiv.org/html/2605.01195v2
