---
title: "$\\\\Psi_0$: An Open Foundation Model Towards Universal Humanoid Loco-Manipulation"
method_name: "$\\\\Psi_0$"
authors: ["Songlin Wei"]
year: 2026
venue: "RSS"
tags: ["robot-manipulation", "real-time-control", "legged-locomotion", "robot-generalization", "humanoid", "loco-manipulation"]
image_source: "online"
arxiv_html: "https://arxiv.org/html/2603.12263"
---
# $\Psi_0$
## 一句话总结

> $\\Psi_0$: An Open Foundation Model Towards Universal Humanoid Loco-Manipulation 主要落在 [[co-training]]、[[cross-embodiment]]、[[egocentric-perception]]、[[flow-matching]] 这一类问题上;从 System0 视角看,它的价值在于把身体状态、接触变化、局部感知或动力学约束接到机器人低层闭环里,让动作生成更稳定、安全、实时。
## 核心贡献

1. 围绕 **$\\Psi_0$: An Open Foundation Model Towards Universal Humanoid Loco-Manipulation** 建立了一个与 co-training、cross-embodiment、egocentric-perception、flow-matching、foundation-model、人形机器人 相关的机器人问题设定,重点不是高层语义规划,而是身体级闭环中的状态变化、接触变化或控制约束。
2. 方法层面的关键价值在于把感知、动力学、约束优化、策略学习或硬件反馈组织成可执行的低层动作生成机制,适合放进 System0 的快反应模块中理解。
3. 对系统集成而言,这类工作补上了 System1 技能选择与电机级控制之间的一层:它负责把上层目标变成动态可行、可恢复、可安全过滤的动作。
4. 若论文包含真实机器人、接触任务或仿真到真实迁移,核心应看它是否报告了闭环稳定性、扰动恢复、硬件延迟、失败模式和跨场景泛化。
## 问题背景

这篇论文可以放在 System0 的“具身快系统”背景下阅读:机器人需要在很短时间内根据本体状态、接触、力/触觉、局部视觉或动力学预测做出动作,而不是等待慢速规划器重新推理整个任务。co-training、cross-embodiment、egocentric-perception、flow-matching、foundation-model、人形机器人、language-conditioned-policy 相关问题的难点通常不在“知道要做什么”,而在动作执行过程中状态会快速偏离、接触会突然改变、约束会实时收紧,低层系统必须及时修正。

### System0 关联

精读时优先看三件事:第一,论文使用哪些快速反馈量,例如关节状态、接触状态、力觉、触觉、深度或局部视觉;第二,反馈如何进入 MPC、WBC、策略网络、安全过滤器或恢复控制器;第三,输出是否直接影响机器人下一步动作,而不是只服务离线分析或高层任务规划。
## 方法详解

### System0 视角解析

从 System0 角度看,这篇论文的方法部分应当围绕 co-training、cross-embodiment、egocentric-perception、flow-matching、foundation-model、人形机器人、language-conditioned-policy、足式运动 来读。核心问题是:系统如何从快速变化的机器人状态中构造控制输入,如何处理接触、碰撞、约束、模型误差或执行延迟,以及如何把这些信息变成下一时刻的动作。

如果方法里有优化问题,重点看目标函数、约束和求解频率;如果方法里有学习策略,重点看观测空间、动作空间、训练扰动和部署时的闭环频率;如果方法里有硬件或传感器,重点看传感链路是否真的进入控制回路。这样读可以把论文和 System2 语言规划、System1 技能编排区分开来。

### 关键公式与机制

#### 公式 1: [[动力学或策略机制]]

$$\mathbf{q}_{hand}\in\mathbb{R}^{14}$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 2: [[动力学或策略机制]]

$$\mathbf{torso}_{rpy}\in\mathbb{R}^{3}$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 3: [[动力学或策略机制]]

$$\mathbf{q}_{lower}\in\mathbb{R}^{15}$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 4: [[动力学或策略机制]]

$$\{\mathbf{T}_{wrist},\mathbf{P}_{thumb},\mathbf{P}_{index},\mathbf{P}_{middle},\mathbf{P}_{ring},\mathbf{P}_{pinky}\}$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 5: [[动力学或策略机制]]

$$p_{\theta}(\mathbf{a})=\prod^{N}_{t=1}p_{\theta}(\mathbf{a}_{t}|\mathbf{a}_{<t},\ell,\mathbf{o}_{t}).$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 6: [[动力学或策略机制]]

$$\mathbf{z}_{t}\!=\!f_{\theta}^{vlm}(\mathbf{o}_{t},\ell)$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 7: [[优化目标/约束]]

$$
\mathcal{L}_{fm}=\mathbb{E}\left[\lVert v_{\rho}^{flow}(\mathbf{z}_{t},\mathbf{a}_{t}^{\tau},\tau)-(\boldsymbol{\epsilon}-\mathbf{a}_{t})\rVert\right]
$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 8: [[动力学或策略机制]]

$$\mathbf{a}_{t}^{\tau}=\tau\mathbf{a}_{t}+(1-\tau)\boldsymbol{\epsilon}$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 9: [[优化目标/约束]]

$$d\!=\!\texttt{uniform}(0,d_{\max})$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 10: [[动力学或策略机制]]

$$a\!=\!\{\mathbf{q}_{hand},\mathbf{q}_{arm},\mathbf{torso}_{rpy},h_{b},v_{x},v_{y},v_{yaw},p_{yaw}\}$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。
## 关键图表

### Figure 1: Pipeline / core system figure: Model Training and Deployment: First, we pre-train the VLM on the EgoDex [20 ] datas

![Figure 1](https://arxiv.org/html/2603.12263v1/x2.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Pipeline / core system figure: Model Training and Deployment: First, we pre-train the VLM on the EgoDex [20 ] datas”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 2: MM-DiT for VLA: Comparison of MM-DiT architecture with naive DiT. τ $\tau$ is the flow t

![Figure 2](https://arxiv.org/html/2603.12263v1/x3.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“MM-DiT for VLA: Comparison of MM-DiT architecture with naive DiT. τ $\tau$ is the flow t”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 3: Single Operator Teleoperation Framework. Our framework maps human upper-body motions

![Figure 3](https://arxiv.org/html/2603.12263v1/x9.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Single Operator Teleoperation Framework. Our framework maps human upper-body motions”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### 论文表格

#### Table 1: TABLE I: Ablation Studies. We study the effects of pre-training, post-training, and real-time chunking on a dual-arm lo

| Pre-Training | Post-Training (On HE) | Real-Time Chunking | MM-DiT Action Head | Naive DiT Action Head | Right-Arm Pick-n-Place | Left-Arm Pick-n-Place | Dual-Arm Carry | Overall Success Rate | |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EgoDex | HE | | | | | | | | |
| ✗ | ✗ | ✗ | ✗ | ✗ | ✓ | 1/10 | 1/10 | 1/10 | 0/10 |
| ✗ | ✗ | ✗ | ✗ | ✓ | ✗ | 9/10 | 2/10 | 3/10 | 2/10 |
| ✓ | ✗ | ✗ | ✗ | ✓ | ✗ | 8/10 | 6/10 | 6/10 | 6/10 |
| ✓ | ✓ | ✗ | ✗ | ✓ | ✗ | 8/10 | 8/10 | 9/10 | 8/10 |
| ✓ | ✓ | ✓ | ✗ | ✓ | ✗ | 9/10 | 9/10 | 10/10 | 9/10 |
| ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | 9/10 | 9/10 | 9/10 | 9/10 |

**说明**: TABLE I: Ablation Studies. We study the effects of pre-training, post-training, and real-time chunking on a dual-arm long-horizon task which consists of three steps: right-arm pick and place, left-arm pick-and-place and dual-arm lift.

#### Table 2: TABLE II: Fast Tokenizer. Comparison of reconstruction loss and average token length before and after training. Boldface

| | Reconstruction L1 Loss | Avg Token Length |
| --- | --- | --- |
| Before | $\mathrm{e}{-}4$ | 2.08 |
| After | $\textbf{1.95}\times 1\mathrm{e}{-}4$ | 13.04 |

**说明**: TABLE II: Fast Tokenizer. Comparison of reconstruction loss and average token length before and after training. Boldface indicates the best performance.

#### Table 3: TABLE III: Real-World Benchmarking: We provide a detailed report of real-world benchmarking results, including sub-task

| Task 3 Task 2 | | | | | | | | | | | | |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Descriptions | Pick the bottle, turn around, and pour into cup | Spray the bowl with water, wipe clean, and fold it up | | | | | | | | | | |
| | Grasp | Move | Pour | Place | | Overall | Grasp | Pull | Spray | Put | | Overall |
| Diffusion Policy | 0/10 | 0/10 | 0/10 | 0/10 | | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | | 0/10 |
| ACT | 0/10 | 0/10 | 0/10 | 0/10 | | 0/10 | 3/10 | 2/10 | 4/10 | 3/10 | | 1/10 |
| InternVLA-M1 | 0/10 | 0/10 | 0/10 | 0/10 | | 0/10 | 0/10 | 0/10 | 2/10 | 4/10 | | 0/10 |
| EgoVLA | 4/10 | 6/10 | 1/10 | 2/10 | | 1/10 | 0/10 | 0/10 | 0/10 | 0/10 | | 0/10 |
| H-RDT | 1/10 | 0/10 | 0/10 | 1/10 | | 0/10 | 0/10 | 1/10 | 0/10 | 0/10 | | 0/10 |
| $\pi 0.5$ | 10/10 | 6/10 | 3/10 | 2/10 | | 2/10 | 9/10 | 7/10 | 5/10 | 7/10 | | 3/10 |
| GR00T N1.6 | 3/10 | 5/10 | 5/10 | 4/10 | | 4/10 | 5/10 | 5/10 | 9/10 | 7/10 | | 4/10 |
|  0 \Psi_{0} (Ours) | 9/10 | 8/10 | 8/10 | 8/10 | | 8/10 | 10/10 | 10/10 | 9/10 | 7/10 | | 7/10 |
| Task 4 Task 1 | | | | | | | | | | | | |
| Descriptions | Grab the can, turn and pour onto plate, push the cart forward | Remove the lid, turn on the faucet, and fill with water | | | | | | | | | | |
| | Grasp | Rotate | Pour | Grab | Push | Overall | Grasp | Remove | Turn | Put | | Overall |
| Diffusion Policy | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | | 0/10 |
| ACT | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 7/10 | 0/10 | 0/10 | 0/10 | | 0/10 |
| InternVLA-M1 | 2/10 | 0/10 | 1/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | | 0/10 |
| EgoVLA | 0/10 | 0/10 | 1/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | | 0/10 |
| H-RDT | 3/10 | 1/10 | 1/10 | 0/10 | 0/10 | 0/10 | 7/10 | 0/10 | 0/10 | 1/10 | | 0/10 |
| $\pi 0.5$ | 2/10 | 5/10 | 5/10 | 8/10 | 1/10 | 1/10 | 4/10 | 4/10 | 8/10 | 2/10 | | 2/10 |
| GR00T N1.6 | 5/10 | 7/10 | 5/10 | 4/10 | 3/10 | 3/10 | 10/10 | 3/10 | 2/10 | 3/10 | | 2/10 |
|  0 \Psi_{0} (Ours) | 10/10 | 9/10 | 7/10 | 10/10 | 10/10 | 7/10 | 10/10 | 10/10 | 6/10 | 10/10 | | 6/10 |
| Task 5 Task 6 | | | | | | | | | | | | |
| Descriptions | Put toy into basket, walk to human, hand it | Push the cart, grab the grapes, and place on the plate | | | | | | | | | | |
| | Grasp | Hook | Walk | Hand | | Overall | Handle | Push | Grasp | Place | | Overall |
| Diffusion Policy | 0/10 | 0/10 | 0/10 | 0/10 | | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | | 0/10 |
| ACT | 3/10 | 0/10 | 5/10 | 5/10 | | 0/10 | 2/10 | 2/10 | 0/10 | 0/10 | | 0/10 |
| InternVLA-M1 | 2/10 | 3/10 | 1/10 | 1/10 | | 1/10 | 8/10 | 8/10 | 5/10 | 5/10 | | 5/10 |
| EgoVLA | 0/10 | 0/10 | 10/10 | 1/10 | | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | | 0/10 |
| H-RDT | 2/10 | 0/10 | 0/10 | 0/10 | | 0/10 | 0/10 | 0/10 | 6/10 | 1/10 | | 0/10 |
| $\pi 0.5$ | 9/10 | 8/10 | 5/10 | 5/10 | | 5/10 | 8/10 | 9/10 | 3/10 | 3/10 | | 3/10 |
| GR00T N1.6 | 8/10 | 5/10 | 0/10 | 0/10 | | 0/10 | 7/10 | 9/10 | 8/10 | 7/10 | | 4/10 |
|  0 \Psi_{0} (Ours) | 9/10 | 9/10 | 10/10 | 10/10 | | 9/10 | 9/10 | 9/10 | 7/10 | 7/10 | | 6/10 |
| Task 8 Task 7 | | | | | | | | | | | | |
| Descriptions | Pull out the tray and turn to throw the chip can into the trash | Hold the lunch bag and squat down to place on the table | | | | | | | | | | |
| | Grasp | Pull | Walk | Drop | | Overall | Hold | Turn | Squat | Put | | Overall |
| Diffusion Policy | 0/10 | 0/10 | 0/10 | 0/10 | | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | | 0/10 |
| ACT | 7/10 | 1/10 | 7/10 | 0/10 | | 0/10 | 6/10 | 8/10 | 5/10 | 5/10 | | 5/10 |
| InternVLA-M1 | 8/10 | 5/10 | 1/10 | 1/10 | | 1/10 | 0/10 | 0/10 | 0/10 | 0/10 | | 0/10 |
| H-RDT | 8/10 | 4/10 | 2/10 | 0/10 | | 0/10 | 9/10 | 9/10 | 7/10 | 6/10 | | 6/10 |
| EgoVLA | 0/10 | 0/10 | 0/10 | 0/10 | | 0/10 | 3/10 | 4/10 | 2/10 | 2/10 | | 2/10 |
| $\pi 0.5$ | 9/10 | 3/10 | 3/10 | 3/10 | | 1/10 | 3/10 | 9/10 | 2/10 | 2/10 | | 2/10 |
| GR00T N1.6 | 10/10 | 1/10 | 10/10 | 3/10 | | 1/10 | 5/10 | 10/10 | 5/10 | 5/10 | | 5/10 |
|  0 \Psi_{0} (Ours) | 10/10 | 5/10 | 10/10 | 9/10 | | 5/10 | 10/10 | 9/10 | 9/10 | 9/10 | | 9/10 |

**说明**: TABLE III: Real-World Benchmarking: We provide a detailed report of real-world benchmarking results, including sub-task progress. Each task consists of three to five subtasks, and a trial is counted as successful only if all subtasks are completed. Boldface indicates the best performance, while denotes the second-best performance.

#### Table 4: TABLE IV: GR00T with RTC. We study the effect of RTC on the GR00T baseline. The task consists of three steps. It achieve

| GR00T-N1.6 | Pick the dumpling | Pick the hippo | Carry the box | Overall SR |
| --- | --- | --- | --- | --- |
| w/o RTC | 10/10 | 7/10 | 9/10 | 7/10 |
| w/ RTC | 6/10 | 7/10 | 10/10 | 6/10 |

**说明**: TABLE IV: GR00T with RTC. We study the effect of RTC on the GR00T baseline. The task consists of three steps. It achieves comparable performance on GR00T with and without RTC.

#### Table 5: TABLE V: Ablation of Pre-Training on 10% EgoDex. We found that using 10% of EgoDex perform worse than the baseline  0 \

| Experiment 1 | Pick the dumpling | Pick the hippo | Carry the box | Overall SR |
| --- | --- | --- | --- | --- |
| Baseline ( 0 \Psi_{0}) | 9/10 | 9/10 | 10/10 | 8/10 |
| Variant (10% EgoDex) | 6/10 | 1/10 | 5/10 | 1/10 |
| Experiment 2 | Grasp bottle | Wipe the bowl | Stack up | Overall SR |
| Baseline ( 0 \Psi_{0}) | 10/10 | 9/10 | 7/10 | 7/10 |
| Variant (10% EgoDex) | 9/10 | 10/10 | 7/10 | 6/10 |

**说明**: 该表格来自论文中的公式、奖励项或实验设置；为避免 PDF 抽取噪声，网页中仅保留可渲染的核心符号和数值。

#### Table 6: TABLE VI: Ablation of Pre-Training on HE. We discover that the HE variant achieves high performance on tasks that do not

| Experiment 1 | Pick the dumpling | Pick the hippo | Carry the box | Overall SR |
| --- | --- | --- | --- | --- |
| Baseline ( 0 \Psi_{0}) | 9/10 | 9/10 | 10/10 | 8/10 |
| Variant (HE) | 9/10 | 4/10 | 10/10 | 4/10 |
| Experiment 2 | Grasp bottle | Wipe the bowl | Stack up | Overall SR |
| Baseline ( 0 \Psi_{0}) | 10/10 | 9/10 | 7/10 | 7/10 |
| Variant (HE) | 10/10 | 9/10 | 4/10 | 4/10 |

**说明**: TABLE VI: Ablation of Pre-Training on HE. We discover that the HE variant achieves high performance on tasks that do not require fine-grained manipulation; however, it still lags behind our baseline on subtasks requiring more precise manipulation.
## 实验解读

- 评价重点:围绕 co-training、cross-embodiment、egocentric-perception、flow-matching、foundation-model,优先看真实机器人结果、扰动恢复、接触稳定性、控制误差、成功率、实时性和 sim-to-real 差距。
- System0 视角:实验结论不只看最终成功率,还要看闭环过程中是否能处理状态估计误差、执行延迟、碰撞/接触突变和外部扰动。
- 与上层系统的关系:如果方法能在快闭环内稳定工作,它可以作为 System1 技能或 System2 规划的底层执行保障;如果只在离线或慢速设定下成立,则还不能直接作为身体级快闭环模块。
## 批判性思考

### 优点

- 论文把 co-training、cross-embodiment、egocentric-perception、flow-matching、foundation-model 放到机器人身体级闭环里讨论,和只做高层语义推理的工作相比,更接近真实机器人部署时会遇到的状态、动力学和安全约束。
- 若论文同时给出公式、系统结构图、硬件实验或消融结果,这些证据可以帮助判断方法是否真的能支撑毫秒到百毫秒级反应。

### 局限

- 需要重点核查控制频率、推理延迟、传感延迟、失败恢复、硬件磨损、未见场景泛化和安全边界;这些因素决定它能否成为可靠的 System0 模块。
- 对学习型方法,还要区分训练环境中的鲁棒性和真实部署中的鲁棒性,尤其是接触丰富任务、动态运动任务和人形机器人任务。
## 论文定位

- 研究对象:$\\Psi_0$: An Open Foundation Model Towards Universal Humanoid Loco-Manipulation。
- 关键词:co-training、cross-embodiment、egocentric-perception、flow-matching、foundation-model、人形机器人、language-conditioned-policy、足式运动、移动操作、实时控制。
- System0 关联:关注机器人身体级快反应链路中“状态/接触/局部感知 -> 约束/优化/策略 -> 动作”的短闭环,不把问题停留在语言规划或任务分解层。

## 速查卡片

> [!summary] $\\Psi_0$
> - **论文**: https://www.roboticsproceedings.org/rss22/p021.pdf
> - **arXiv**: http://arxiv.org/abs/2603.12263
> - **arXiv HTML**: https://arxiv.org/html/2603.12263
