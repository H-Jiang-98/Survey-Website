---
title: "TouchGuide: Inference-Time Steering of Visuomotor Policies via Touch Guidance"
method_name: "TouchGuide"
authors: ["Zhemeng Zhang"]
year: 2026
venue: "RSS"
tags: ["contact-reasoning", "robot-manipulation", "reinforcement-learning", "imitation-learning", "contact-rich-manipulation", "robot-generalization", "closed-loop-control", "tactile-feedback"]
image_source: "online"
arxiv_html: "https://arxiv.org/html/2601.20239v6"
---
# TouchGuide
## 一句话总结

> TouchGuide: Inference-Time Steering of Visuomotor Policies via Touch Guidance 主要落在 [[closed-loop-control]]、[[接触推理]]、[[接触丰富操作]]、[[diffusion-policy]] 这一类问题上;从 System0 视角看,它的价值在于把身体状态、接触变化、局部感知或动力学约束接到机器人低层闭环里,让动作生成更稳定、安全、实时。
## 核心贡献

1. 围绕 **TouchGuide: Inference-Time Steering of Visuomotor Policies via Touch Guidance** 建立了一个与 closed-loop-control、接触推理、接触丰富操作、diffusion-policy、模仿学习、inference-time-algorithm 相关的机器人问题设定,重点不是高层语义规划,而是身体级闭环中的状态变化、接触变化或控制约束。
2. 方法层面的关键价值在于把感知、动力学、约束优化、策略学习或硬件反馈组织成可执行的低层动作生成机制,适合放进 System0 的快反应模块中理解。
3. 对系统集成而言,这类工作补上了 System1 技能选择与电机级控制之间的一层:它负责把上层目标变成动态可行、可恢复、可安全过滤的动作。
4. 若论文包含真实机器人、接触任务或仿真到真实迁移,核心应看它是否报告了闭环稳定性、扰动恢复、硬件延迟、失败模式和跨场景泛化。
## 问题背景

这篇论文可以放在 System0 的“具身快系统”背景下阅读:机器人需要在很短时间内根据本体状态、接触、力/触觉、局部视觉或动力学预测做出动作,而不是等待慢速规划器重新推理整个任务。closed-loop-control、接触推理、接触丰富操作、diffusion-policy、模仿学习、inference-time-algorithm、policy-learning 相关问题的难点通常不在“知道要做什么”,而在动作执行过程中状态会快速偏离、接触会突然改变、约束会实时收紧,低层系统必须及时修正。

### System0 关联

精读时优先看三件事:第一,论文使用哪些快速反馈量,例如关节状态、接触状态、力觉、触觉、深度或局部视觉;第二,反馈如何进入 MPC、WBC、策略网络、安全过滤器或恢复控制器;第三,输出是否直接影响机器人下一步动作,而不是只服务离线分析或高层任务规划。
## 方法详解

### System0 视角解析

从 System0 角度看,这篇论文的方法部分应当围绕 closed-loop-control、接触推理、接触丰富操作、diffusion-policy、模仿学习、inference-time-algorithm、policy-learning、机器人操作 来读。核心问题是:系统如何从快速变化的机器人状态中构造控制输入,如何处理接触、碰撞、约束、模型误差或执行延迟,以及如何把这些信息变成下一时刻的动作。

如果方法里有优化问题,重点看目标函数、约束和求解频率;如果方法里有学习策略,重点看观测空间、动作空间、训练扰动和部署时的闭环频率;如果方法里有硬件或传感器,重点看传感链路是否真的进入控制回路。这样读可以把论文和 System2 语言规划、System1 技能编排区分开来。

### 关键公式与机制

#### 公式 1: [[优化目标/约束]]

$$
\mathcal{L}_{\text{CPM}}(\mathbf{O}_{t},\mathbf{a}_{t})=\frac{1}{2}(\mathcal{L}_{\mathbf{O}\to\mathbf{a}}+\mathcal{L}_{\mathbf{a}\to\mathbf{O}}),
$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 2: [[动力学或策略机制]]

$$\hat{\epsilon}_{\theta}=\epsilon_{\theta}(\mathbf{A}_{t}^{k},\mathbf{V}_{t})-\eta\sqrt{1-\bar{\alpha}_{k}}\nabla_{\mathbf{A}_{t}^{k}}s_{\phi}(\mathbf{V}_{t},\mathbf{T}_{t},\mathbf{A}_{t}^{k}),$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 3: [[动力学或策略机制]]

$$\hat{u}_{\theta}=u_{\theta}(\mathbf{A}_{t}^{k},\mathbf{V}_{t})-\eta\frac{k}{1-k}\nabla_{\mathbf{A}_{t}^{k}}s_{\phi}(\mathbf{V}_{t},\mathbf{T}_{t},\mathbf{A}_{t}^{k}),$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 4: [[动力学或策略机制]]

$$\log p(\mathbf{a}_{\text{real}}|\mathbf{a})\approx-\frac{1}{2\sigma}\lVert \mathbf{O}-\mathbf{a} \rVert_{2}^{2}+C=-\frac{1}{2\sigma}\left(\lVert \mathbf{O} \rVert_{2}^{2}+\lVert \mathbf{a} \rVert_{2}^{2}-2\mathbf{O}^{\top}\mathbf{a}\right)+C.$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 5: [[动力学或策略机制]]

$$=-\frac{1}{M}\sum_{i=1}^{M}\log\frac{\exp(\mathbf{O}_{i}^{\top}\mathbf{a}_{i}/\tau)}{\sum_{j=1}^{M}\exp(\mathbf{O}_{i}^{\top}\mathbf{a}_{j}/\tau)},$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 6: [[动力学或策略机制]]

$$=-\frac{1}{M}\sum_{i=1}^{M}\log\frac{\exp(\mathbf{a}_{i}^{\top}\mathbf{O}_{i}/\tau)}{\sum_{j=1}^{M}\exp(\mathbf{a}_{i}^{\top}\mathbf{O}_{j}/\tau)},$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 7: [[动力学或策略机制]]

$$\nabla_{\mathbf{A}}\log p(\mathbf{a}_{\text{real}}|\mathbf{a})=\frac{1}{\sigma}\nabla_{\mathbf{A}}\mathbf{O}^{\top}\mathbf{a}=\frac{1}{\sigma}\nabla_{\mathbf{A}}\mathbf{s}.$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 8: [[动力学或策略机制]]

$$=a_{t}x\frac{\int p_{t}(x| x_{0})q(x_{0})dx_{0}}{p_{t}(x)}+b_{t}\int $\frac{\nabla p_{t}(x$ |x_{0})}{p_{t}(x| x_{0})} $\frac{p_{t}(x$ |x_{0})q(x_{0})}{p_{t}(x)}dx_{0}$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 9: [[动力学或策略机制]]

$$=a_{t}x\int\frac{p_{t}(x| x_{0})q(x_{0})}{p_{t}(x)}dx_{0}+b_{t}\int\nabla $\log p_{t}(x$ | x_{0}) $\frac{p_{t}(x$ |x_{0})q(x_{0})}{p_{t}(x)}dx_{0}$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 10: [[动力学或策略机制]]

$$\hat{\epsilon}_{\theta}(x_{t})=\epsilon_{\theta}(x_{t})-\eta\sqrt{1-\bar{\alpha}_{t}}\nabla_{x_{t}}\log p_{\phi}(y|x_{t}).$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。
## 关键图表

### Figure 1: Overview of TacUMI data collection system. (a) TacUMI (Collection-side) uses a Vive t

![Figure 1](https://arxiv.org/html/2601.20239v6/x2.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Overview of TacUMI data collection system. (a) TacUMI (Collection-side) uses a Vive t”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 2: Overview of TouchGuide framework. (a) The architecture of the task-specific Contact P

![Figure 2](https://arxiv.org/html/2601.20239v6/x3.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Overview of TouchGuide framework. (a) The architecture of the task-specific Contact P”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 3: Trajectory visualization comparing SLAM-based UMI and TacUMI. (a)-(c) show three ra

![Figure 3](https://arxiv.org/html/2601.20239v6/x11.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Trajectory visualization comparing SLAM-based UMI and TacUMI. (a)-(c) show three ra”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### 论文表格

#### Table 1: TABLE I: Policy performance comparison on five challenging fine-grained, contact-rich tasks.

| | | Bi-Arx5 (Dual-arm) | Flexiv Rizon4 (Single-arm) | | | | |
| --- | --- | --- | --- | --- | --- | --- | --- |
| | | Shoe Lacing (100 Demos) | Chip Handover (50 Demos) | Cucum. Peeling (50 Demos) | Vase Wiping (30 Demos) | Lock Opening (20 Demos) | Average |
| DP | Diffusion Policy | 0% | 5% | 0.500 | 0.265 | 0% | 16.3% |
| DP w/ Tactile Observation | 0% | 15% | 0.520 | 0.240 | 5% | 19.2% | |
| SafeDiff | 0% | 10% | 0.635 | 0.325 | 5% | 22.2% | |
| RDP | 0% | 20% | 0.740 | 0.475 | 10% | 30.3% | |
| Policy Consensus | 0% | 15% | 0.630 | 0.405 | 5% | 24.7% | |
| Tactile Dynamics | 0% | 15% | 0.695 | 0.335 | 0% | 23.6% | |
| TouchGuide (Force) | 0% | 30% | 0.805 | 0.510 | 15% | 35.3% | |
| TouchGuide (Tactile Img.) | 0% | 25% | 0.810 | 0.550 | 20% | 36.2% | |
| $\pi_{0.5}$ | $\pi_{0.5}$ | 20% | 25% | 0.785 | 0.360 | 20% | 35.9% |
| $\pi_{0.5} w/ Tactile Observation$ | 20% | 15% | 0.760 | 0.305 | 10% | 30.3% | |
| Tactile Dynamics | 10% | 30% | 0.815 | 0.395 | 20% | 36.2% | |
| TouchGuide (Force) | 25% | 40% | 0.955 | 0.590 | 30% | 49.9% | |
| TouchGuide (Tactile Img.) | 35% | 60% | 0.975 | 0.675 | 30% | 58.0% | |

**说明**: TABLE I: Policy performance comparison on five challenging fine-grained, contact-rich tasks.

#### Table 2: TABLE II: Ablation study on noise pretraining.

| | Chip | Cucumber | Lock | |
| --- | --- | --- | --- | --- |
| | Handover | Peeling | Opening | Average |
| Ours w/o Noise Pretraining | 30% | 0.725 | 15% | 39.17% |
| Ours w/ Noise Pretraining | 60% | 0.975 | 30% | 62.50% |

**说明**: TABLE II: Ablation study on noise pretraining.

#### Table 3: TABLE III: Ablation study on CPM input modalities.

| | Chip | Cucumber | Lock | |
| --- | --- | --- | --- | --- |
| | Handover | Peeling | Opening | Average |
| Ours w/o Vision | 30% | 0.850 | 15% | 43.33% |
| Ours w/o Touch | 35% | 0.755 | 20% | 43.50% |
| Ours w/ Vision & Touch | 60% | 0.975 | 30% | 62.50% |

**说明**: TABLE III: Ablation study on CPM input modalities.

#### Table 4: TABLE IV: Robustness evaluation visual occlusion at different contact phases in the Cucumber Peeling task.

| | None | Before | During | After | Average |
| --- | --- | --- | --- | --- | --- |
| TouchGuide . $\pi_{0.5})$ | 0.975 | 0.955 | 0.910 | 0.960 | 0.950 |

**说明**: TABLE IV: Robustness evaluation visual occlusion at different contact phases in the Cucumber Peeling task.

#### Table 5: TABLE V: Success rate comparison of data collection systems on the Lock Opening task.

| Method | TacUMI | UMI (SLAM-based) | VR Teleop |
| --- | --- | --- | --- |
| $\pi_{0.5}$ | 20% | 0% | 5% |
| TouchGuide | 30% | 5% | 15% |

**说明**: TABLE V: Success rate comparison of data collection systems on the Lock Opening task.

#### Table 6: TABLE VI: User study on data collection systems.

| Method | Attempts ↓ \downarrow | Length ↓ \downarrow | Valid Rate ↑ \uparrow | Satisfaction ↑ \uparrow |
| --- | --- | --- | --- | --- |
| VR Teleop | 100 | 23.0 min | 100.00% | 7.8 |
| SLAM-based | 143 | 15.3 min | 69.93% | 7.2 |
| VR Motion Tracker | 104 | 14.7 min | 96.15% | 9.0 |
| TacUMI (Ours) | 100 | 13.5 min | 100.00% | 9.6 |

**说明**: TABLE VI: User study on data collection systems.

#### Table 7: TABLE VII: Hyperparameters for TouchGuide across tasks.

| | | Shoe Lacing | Chip Handover | Cucumber Peeling | Vase Wiping | Lock Opening |
| --- | --- | --- | --- | --- | --- | --- |
| Diffusion Policy | K TouchGuide K_{ $\text{TouchGuide}}$ | 15 | 20 | 20 | 10 | 10 |
|  \eta | 3 | 4 | 4 | 4 | 4 | |
| $\pi_{0.5}$ | K TouchGuide K_{ $\text{TouchGuide}}$ | 0.2 | 0.3 | 0.3 | 0.3 | 0.3 |
|  \eta | 10 | 10 | 10 | 10 | 10 | |

**说明**: TABLE VII: Hyperparameters for TouchGuide across tasks.

#### Table 8: TABLE VIII: Comparison of CPM performance different visual and tactile encoders.

| Visual / Tactile Encoder | Trainable Parameters (CPM) | Chip Handover | Cucumber Peeling | Lock Opening | Average |
| --- | --- | --- | --- | --- | --- |
| ResNet18 / ResNet18 | 141 M | 50% | 0.820 | 15% | 49.0% |
| DINOv2 / ResNet18 | 130 M | 60% | 0.965 | 25% | 60.5% |
| DINOv2 / DINOv2 (Ours) | 118 M | 60% | 0.975 | 30% | 62.5% |

**说明**: TABLE VIII: Comparison of CPM performance different visual and tactile encoders.

#### Table 9: TABLE IX: A detailed comparison of existing data collection systems in terms of tactile feedback, precision, cost, and

| | | | | Low-cost | Lightweight |
| --- | --- | --- | --- | --- | --- |
| Method | Category | Tactile Feedback | High-precision | (< < $1000) | (< < 1000g) |
| GELLO [60 ] | Teleop (Leader-follower) | None | ✓ | ✓ | ✓ |
| ALOHA [75 ] | Teleop (Leader-follower) | None | ✓ | ✗ | ✗ |
| Bi-ACT [6 ] | Teleop (Leader-follower) | Indirect (Force) | ✓ | ✓ | ✓ |
| Bunny-VisionPro [20 ] | Teleop (Hand Retargeting) | Indirect (Vibration) | ✓ | ✓ | ✓ |
| TactAR [65 ] | Teleop (VR Controller) | Indirect (Visual) | ✓ | ✓ | ✓ |
| UMI [15 ] | Handheld (SLAM-based) | Semi-direct (Linkage) | ✗ | ✓ | ✓ |
| FastUMI [76 ] | Handheld (SLAM-based) | Semi-direct (Linkage) | ✗ | ✓ | ✓ |
| Touch in the Wild [79 ] | Handheld (SLAM-based) | Semi-direct (Linkage) | ✗ | ✓ | ✓ |
| ViTaMIn [42 ] | Handheld (SLAM-based) | Semi-direct (Linkage) | ✗ | ✓ | ✓ |
| exUMI [64 ] | Handheld (VR Motion Tracker) | Semi-direct (Linkage) | ✓ | ✓ | ✗ |
| FARM [28 ] | Handheld (Motion Capture) | Semi-direct (Linkage) | ✓ | ✗ | ✓ |
| UMI-FT [16 ] | Handheld (VIO ARKit) | Semi-direct (Linkage) | ✗ | ✓ | ✓ |
| ViTaMIn-B [39 ] | Handheld (VR Motion Tracker) | Direct (Rigid) | ✓ | ✓ | ✗ |
| FreeTacMan [59 ] | Handheld (Motion Capture) | Direct (Rigid) | ✓ | ✗ | ✓ |
| TacUMI (Ours) | Handheld (Vive Tracker) | Direct (Rigid) | ✓ | ✓ | ✓ |

**说明**: TABLE IX: A detailed comparison of existing data collection systems in terms of tactile feedback, precision, cost, and weight.

#### Table 10: TABLE X: TacUMI hardware bill of materials, including model numbers and quantities.

| | Component | Model | Quantity |
| --- | --- | --- | --- |
| Collection-side | Tracker | Vive Tracker | 1 |
| Base Station | Vive Lighthouse | 2 | |
| Magnetic Encoder | MT6835 | 1 | |
| Rigid Fingertip | PLA 3D-Printed | 2 | |
| Execution-side | Gripper Motor | GIM3505-8 | 1 |
| Both | Wrist Camera | USB Camera | 2 |
| Onboard Compute Module | RK3576 Core Board | 2 | |
| Tactile Sensor | Xense | 4 / 2 (Shared) | |

**说明**: TABLE X: TacUMI hardware bill of materials, including model numbers and quantities.

#### Table 11: TABLE XI: TouchGuide performance comparison on NVIDIA RTX PRO 6000 Blackwell.

| | $\pi_{0.5} (JAX)$ | TouchGuide . $\pi_{0.5})$ | DP | TouchGuide (DP) |
| --- | --- | --- | --- | --- |
| Inference Speed ↑ \uparrow | 18.52 fps | 17.24 fps (- - 6.91%) | 12.82 fps | 12.35 fps (- - 3.67%) |
| Average Success Rate ↑ \uparrow | 35.9% | 58.0% (+ + 61.56%) | 16.3% | 36.2% (+ + 122.09%) |

**说明**: TABLE XI: TouchGuide performance comparison on NVIDIA RTX PRO 6000 Blackwell.

#### Table 12: TABLE XII: Hyperparameters for DP training on NVIDIA RTX PRO 6000 Blackwell.

| Hyperparameters | Shoe Lacing | Chip Handover | Cucumber Peeling | Vase Wiping | Lock Opening |
| --- | --- | --- | --- | --- | --- |
| Image Shape (Resolution  \times Views) | (224, 224, 3)  \times 3 | (224, 224, 3)  \times 3 | (224, 224, 3)  \times 3 | (224, 224, 3)  \times 1 | (224, 224, 3)  \times 1 |
| Action Shape | 14 | 14 | 14 | 10 | 10 |
| Batch Size | 32 | 32 | 32 | 32 | 32 |
| Learning Rate | 1.0e-4 | 1.0e-4 | 1.0e-4 | 1.0e-4 | 1.0e-4 |
| Warm-up Steps | 500 | 500 | 500 | 500 | 500 |
| Betas | [0.95, 0.999] | [0.95, 0.999] | [0.95, 0.999] | [0.95, 0.999] | [0.95, 0.999] |
| Weight Decay | 1.0e-6 | 1.0e-6 | 1.0e-6 | 1.0e-6 | 1.0e-6 |
| Epsilon | 1.0e-8 | 1.0e-8 | 1.0e-8 | 1.0e-8 | 1.0e-8 |
| Epochs | 300 | 300 | 300 | 300 | 300 |

**说明**: TABLE XII: Hyperparameters for DP training on NVIDIA RTX PRO 6000 Blackwell.

#### Table 13: TABLE XIII: Hyperparameters for π 0.5 $\pi_{0.5}$ training NVIDIA RTX PRO 6000 Blackwell.

| Hyperparameters | Shoe Lacing | Chip Handover | Cucumber Peeling | Vase Wiping | Lock Opening |
| --- | --- | --- | --- | --- | --- |
| Image Shape (Resolution  \times Views) | (224, 224, 3)  \times 3 | (224, 224, 3)  \times 3 | (224, 224, 3)  \times 3 | (224, 224, 3)  \times 1 | (224, 224, 3)  \times 1 |
| Action Shape | 14 | 14 | 14 | 10 | 10 |
| Batch Size | 64 | 64 | 64 | 64 | 64 |
| Paligemma | gemma_2b_lora | gemma_2b_lora | gemma_2b_lora | gemma_2b_lora | gemma_2b_lora |
| Action Expert | gemma_300m_lora | gemma_300m_lora | gemma_300m_lora | gemma_300m_lora | gemma_300m_lora |
| Weight | pi05_base | pi05_base | pi05_base | pi05_base | pi05_base |
| EMA Decay | 0.99 | 0.99 | 0.99 | 0.99 | 0.99 |
| Steps | 40,000 | 20,000 | 20,000 | 20,000 | 20,000 |

**说明**: 该表格来自论文中的公式、奖励项或实验设置；为避免 PDF 抽取噪声，网页中仅保留可渲染的核心符号和数值。

#### Table 14: TABLE XIV: Hyperparameters for CPM training NVIDIA RTX PRO 6000 Blackwell.

| Hyperparameters | Shoe Lacing | Chip Handover | Cucumber Peeling | Vase Wiping | Lock Opening |
| --- | --- | --- | --- | --- | --- |
| Image Shape (Resolution  \times Views) | (224, 224, 3)  \times 3 | (224, 224, 3)  \times 3 | (224, 224, 3)  \times 3 | (224, 224, 3)  \times 1 | (224, 224, 3)  \times 1 |
| Tactile Shape (Resolution  \times Views) | (224, 224, 3)  \times 2 | (224, 224, 3)  \times 2 | (224, 224, 3)  \times 2 | (224, 224, 3)  \times 2 | (224, 224, 3)  \times 2 |
| Action Shape | 14 | 14 | 14 | 10 | 10 |
| Batch Size | 64 | 64 | 64 | 64 | 64 |
| Learning Rate | 1.0e-5 | 1.0e-5 | 1.0e-5 | 1.0e-5 | 1.0e-5 |
| Warm-up Ratio | 0.05 | 0.05 | 0.05 | 0.05 | 0.05 |
| Epochs | 200 | 200 | 200 | 200 | 200 |

**说明**: TABLE XIV: Hyperparameters for CPM training NVIDIA RTX PRO 6000 Blackwell.
## 实验解读

- 评价重点:围绕 closed-loop-control、接触推理、接触丰富操作、diffusion-policy、模仿学习,优先看真实机器人结果、扰动恢复、接触稳定性、控制误差、成功率、实时性和 sim-to-real 差距。
- System0 视角:实验结论不只看最终成功率,还要看闭环过程中是否能处理状态估计误差、执行延迟、碰撞/接触突变和外部扰动。
- 与上层系统的关系:如果方法能在快闭环内稳定工作,它可以作为 System1 技能或 System2 规划的底层执行保障;如果只在离线或慢速设定下成立,则还不能直接作为身体级快闭环模块。
## 批判性思考

### 优点

- 论文把 closed-loop-control、接触推理、接触丰富操作、diffusion-policy、模仿学习 放到机器人身体级闭环里讨论,和只做高层语义推理的工作相比,更接近真实机器人部署时会遇到的状态、动力学和安全约束。
- 若论文同时给出公式、系统结构图、硬件实验或消融结果,这些证据可以帮助判断方法是否真的能支撑毫秒到百毫秒级反应。

### 局限

- 需要重点核查控制频率、推理延迟、传感延迟、失败恢复、硬件磨损、未见场景泛化和安全边界;这些因素决定它能否成为可靠的 System0 模块。
- 对学习型方法,还要区分训练环境中的鲁棒性和真实部署中的鲁棒性,尤其是接触丰富任务、动态运动任务和人形机器人任务。
## 论文定位

- 研究对象:TouchGuide: Inference-Time Steering of Visuomotor Policies via Touch Guidance。
- 关键词:closed-loop-control、接触推理、接触丰富操作、diffusion-policy、模仿学习、inference-time-algorithm、policy-learning、机器人操作、tactile-feedback、visual-tactile。
- System0 关联:关注机器人身体级快反应链路中“状态/接触/局部感知 -> 约束/优化/策略 -> 动作”的短闭环,不把问题停留在语言规划或任务分解层。

## 速查卡片

> [!summary] TouchGuide
> - **论文**: https://www.roboticsproceedings.org/rss22/p078.pdf
> - **arXiv**: http://arxiv.org/abs/2601.20239v6
> - **arXiv HTML**: https://arxiv.org/html/2601.20239v6
