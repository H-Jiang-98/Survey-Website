---
title: "CoRAL: Contact-Rich Adaptive LLM-based Control for Robotic Manipulation"
method_name: "CoRAL"
authors: ["Berk Cicek"]
year: 2026
venue: "RSS"
tags: ["contact-reasoning", "robot-manipulation", "real-time-control", "adaptive-control", "contact-rich-manipulation", "robot-generalization", "closed-loop-control", "sim-to-real"]
image_source: "online"
arxiv_html: "https://arxiv.org/html/2605.02600v2"
---
# CoRAL
## 一句话总结

> CoRAL: Contact-Rich Adaptive LLM-based Control for Robotic Manipulation 主要落在 [[adaptive-control]]、[[closed-loop-control]]、[[接触推理]]、[[接触丰富操作]] 这一类问题上;从 System0 视角看,它的价值在于把身体状态、接触变化、局部感知或动力学约束接到机器人低层闭环里,让动作生成更稳定、安全、实时。
## 核心贡献

1. 围绕 **CoRAL: Contact-Rich Adaptive LLM-based Control for Robotic Manipulation** 建立了一个与 adaptive-control、closed-loop-control、接触推理、接触丰富操作、language-conditioned-policy、reactive-control 相关的机器人问题设定,重点不是高层语义规划,而是身体级闭环中的状态变化、接触变化或控制约束。
2. 方法层面的关键价值在于把感知、动力学、约束优化、策略学习或硬件反馈组织成可执行的低层动作生成机制,适合放进 System0 的快反应模块中理解。
3. 对系统集成而言,这类工作补上了 System1 技能选择与电机级控制之间的一层:它负责把上层目标变成动态可行、可恢复、可安全过滤的动作。
4. 若论文包含真实机器人、接触任务或仿真到真实迁移,核心应看它是否报告了闭环稳定性、扰动恢复、硬件延迟、失败模式和跨场景泛化。
## 问题背景

这篇论文可以放在 System0 的“具身快系统”背景下阅读:机器人需要在很短时间内根据本体状态、接触、力/触觉、局部视觉或动力学预测做出动作,而不是等待慢速规划器重新推理整个任务。adaptive-control、closed-loop-control、接触推理、接触丰富操作、language-conditioned-policy、reactive-control、实时控制 相关问题的难点通常不在“知道要做什么”,而在动作执行过程中状态会快速偏离、接触会突然改变、约束会实时收紧,低层系统必须及时修正。

### System0 关联

精读时优先看三件事:第一,论文使用哪些快速反馈量,例如关节状态、接触状态、力觉、触觉、深度或局部视觉;第二,反馈如何进入 MPC、WBC、策略网络、安全过滤器或恢复控制器;第三,输出是否直接影响机器人下一步动作,而不是只服务离线分析或高层任务规划。
## 方法详解

### System0 视角解析

从 System0 角度看,这篇论文的方法部分应当围绕 adaptive-control、closed-loop-control、接触推理、接触丰富操作、language-conditioned-policy、reactive-control、实时控制、机器人操作 来读。核心问题是:系统如何从快速变化的机器人状态中构造控制输入,如何处理接触、碰撞、约束、模型误差或执行延迟,以及如何把这些信息变成下一时刻的动作。

如果方法里有优化问题,重点看目标函数、约束和求解频率;如果方法里有学习策略,重点看观测空间、动作空间、训练扰动和部署时的闭环频率;如果方法里有硬件或传感器,重点看传感链路是否真的进入控制回路。这样读可以把论文和 System2 语言规划、System1 技能编排区分开来。

### 关键公式与机制

#### 公式 1: [[优化目标/约束]]

$$U^{*}=\arg\min_{U}\mathbb{E}\left[\phi(x_{H})+\sum_{t=0}^{H-1}q(x_{t},u_{t})\right]$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 2: [[动力学或策略机制]]

$$\begin{split}U_{new}&=U_{prev}+\sum_{k=1}^{K}w_{k}\delta U_{k},\\ \text{where}\quad w_{k}&=\frac{\exp\left(-\frac{1}{\lambda}S(V_{k})\right)}{\sum_{j=1}^{K}\exp\left(-\frac{1}{\lambda}S(V_{j})\right)}\end{split}$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 3: [[动力学或策略机制]]

$$J(\mathbf{U})=\mathbb{E}\left[\phi(\mathbf{x}_{H})+\sum_{t=0}^{H-1}q(\mathbf{x}_{t},\mathbf{u}_{t})\right]$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 4: [[动力学或策略机制]]

$$U^{(n)}=\big(U^{(n)}_{0},\dots,U^{(n)}_{H-1}\big),\qquad U^{(n)}_{t}\in\mathbb{R}^{d},$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 5: [[优化目标/约束]]

$$u_{i,t}=\mathrm{clip}\!\left(U^{(n)}_{t}+\varepsilon_{i,t},-u_{\max},u_{\max}\right),$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 6: [[动力学或策略机制]]

$$w_{i}(\lambda)=\exp\!\left(-\frac{S_{i}}{\lambda}\right),\qquad W_{i}(\lambda)=\frac{w_{i}(\lambda)}{\sum_{j=1}^{K}w_{j}(\lambda)}.$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 7: [[动力学或策略机制]]

$$\mathrm{ESS}(\lambda)=\frac{1}{\sum_{i=1}^{K}W_{i}(\lambda)^{2}}\in[1,K],\qquad E_{\text{tgt}}=\phi K,$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 8: [[优化目标/约束]]

$$\lambda_{\mathrm{lo}}=10^{-8},\qquad\lambda_{\mathrm{hi}}=5\cdot\max\!\big(10^{-3},\max_{i}S_{i}\big),$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 9: [[优化目标/约束]]

$$\Delta U_{t}=\sum_{i=1}^{K}W_{i}(\lambda^{(n)})\,\varepsilon_{i,t},\qquad U^{(n+1)}_{t}=\mathrm{clip}\!\left(U^{(n)}_{t}+\beta\,\Delta U_{t},-u_{\max},u_{\max}\right).$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 10: [[动力学或策略机制]]

$$\lambda=\tfrac{1}{2}(\lambda_{\mathrm{lo}}+\lambda_{\mathrm{hi}}),\quad\text{if}\mathrm{ESS}(\lambda)<E_{\text{tgt}}\Rightarrow\lambda_{\mathrm{lo}}\leftarrow\lambda,\;\text{else}\lambda_{\mathrm{hi}}\leftarrow\lambda.$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。
## 关键图表

### Figure 1: The conceptual workflow of CoRAL, illustrated with the “pick the cutting board” task

![Figure 1](https://arxiv.org/html/2605.02600v2/x1.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“The conceptual workflow of CoRAL, illustrated with the “pick the cutting board” task”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 2: The architecture of the CoRAL framework. Given an input image I I, object mo

![Figure 2](https://arxiv.org/html/2605.02600v2/x2.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“The architecture of the CoRAL framework. Given an input image I I, object mo”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 3: Figure S1: End-to-end workflow execution trace. This figure illustrates the data flow using rep

![Figure 3](https://arxiv.org/html/2605.02600v2/x3.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Figure S1: End-to-end workflow execution trace. This figure illustrates the data flow using rep”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### 论文表格

#### Table 1: TABLE I: Comparison against the baselines and ablation study. Performance is measured by success rate (x/10 trials).

| Method | T1: Push+Pick | T2: Pick+Place | T3: Clutter | T4: Const. Force | T5: Flip Box | T6: Flip w/ Wall |
| --- | --- | --- | --- | --- | --- | --- |
| State-of-the-Art Baseline | | | | | | |
| OpenVLA-OFT [12 ] | 0/10 | 10/10 | 9/10 | 0/10 | 1/10 | 0/10 |
| $\pi_{0.5} [2 ]$ | 0/10 | 10/10 | 8/10 | 0/10 | 3/10 | 0/10 |
| L2R [31 ] | 0/10 | 10/10 | 9/10 | 5/10 | 4/10 | 1/10 |
| Human Expert-Designed Cost Baselines | | | | | | |
| Expert (single-stage) | 0/10 | 10/10 | 10/10 | 9/10 | 9/10 | 3/10 |
| Expert (FSM) | 8/10 | 10/10 | 10/10 | 10/10 | 10/10 | 9/10 |
| Our Method (Ablation Study) | | | | | | |
| CoRAL (Ours) | 5/10 | 10/10 | 10/10 | 9/10 | 9/10 | 7/10 |
| CoRAL (w/o Memory) | 2/10 | 10/10 | 9/10 | 9/10 | 7/10 | 5/10 |
| CoRAL (w/o Refinement) | 0/10 | 10/10 | 3/10 | 6/10 | 4/10 | 2/10 |
| CoRAL (Unified VLM) | 0/10 | 2/10 | 0/10 | 1/10 | 0/10 | 0/10 |
| CoRAL (w/o Pose Tracking) | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 |

**说明**: TABLE I: Comparison against the baselines and ablation study. Performance is measured by success rate (x/10 trials).

#### Table 2: TABLE II: Cutting-board success rates different pose and physical-parameter settings.

| Condition | Success Rate |
| --- | --- |
| VLM-estimated parameters, no parameter refinement | 2 / 10 2/10 |
| VLM-estimated parameters, with parameter refinement | 5 / 10 5/10 |
| Ground-truth physical parameters | 5 / 10 5/10 |
| Ground-truth poses | 7 / 10 7/10 |
| Ground-truth poses + ground-truth physical parameters | 8 / 10 8/10 |

**说明**: TABLE II: Cutting-board success rates different pose and physical-parameter settings.

#### Table 3: TABLE III: Real-World Experimental Results. CoRAL is evaluated on the physical robot across all six tasks without real-

| Task | Success Rate | Avg. Exec. Time (s) |
| --- | --- | --- |
| T1: Push and Pick Cutting Board | 4/10 | 21.6 ± \pm 5.5 |
| T2: Pick Box | 10/10 | 16.7 ± \pm 1.1 |
| T3: Pick and Place in Clutter | 10/10 | 22.0 ± \pm 1.5 |
| T4: Push with Constant Force | 9/10 | 11.7 ± \pm 1.9 |
| T5: Flip Box | 7/10 | 9.2 ± \pm 2.6 |
| T6: Flip with Wall | 6/10 | 25.3 ± \pm 4.9 |

**说明**: TABLE III: Real-World Experimental Results. CoRAL is evaluated on the physical robot across all six tasks without real-world fine-tuning.

#### Table 4: TABLE S1: Comparison with State-of-the-Art VLA Manipulation Frameworks

| Framework | Primary Modality | Planning & Control Strategy | Reasoning Mechanism | Data Requirement |
| --- | --- | --- | --- | --- |
| OpenVLA [13 ] | Vision, Language | End-to-End Learned Policy (Action Token Prediction) | Implicit (in VLM backbone) | Large-scale Imitation Learning Demos |
| $\pi_{0.5} [2 ]$ | Vision, Language | End-to-End Learned Policy (Flow Matching) | Implicit (in VLM backbone) | Large-scale Imitation Learning Demos |
| ForceVLA [30 ], TLA [7 ] | Vision, Language, Tactile/Force | End-to-End Learned Policy | Implicit (in network weights) | Large-scale Tactile/Force Demos |
| VLA-Touch [1 ] | Vision, Language, Tactile | VLA Policy + Tactile-based Refinement Controller | Explicit VLM Planning + Semantic Tactile Feedback | Leverages pretrained models; no VLA fine-tuning |
| ThinkAct [8 ], ECoT [32 ] | Vision, Language | End-to-End Learned Policy | Explicit LLM Reasoning (Chain-of-Thought) | Large-scale Imitation Learning Demos |
| OneTwoVLA [16 ] | Vision, Language | Unified Policy (Adaptive Acting & Reasoning) | Explicit LLM Reasoning (System 2) | Imitation Demos + Synthetic Reasoning Data |
| MolmoAct [15 ] | Vision, Language | Multi-stage Pipeline (Perception, Spatial Plan, Action) | Explicit Spatial Reasoning (Trajectory Traces) | Large-scale Imitation Learning Demos |
| IMPACT [17 ] | Vision, Language | VLM-based Static Cost Map + RRT* | Implicit (semantic object labeling) | N/A (Planner-based) |
| VLMPC [33 ] | Vision, Language | VLM-guided Model Predictive Control (MPC) | Explicit VLM Reasoning (for cost & sampling) | N/A (Planner-based) |
| L2R [31 ] | Language | LLM-generated Reward Function + MPC | Explicit LLM Reasoning (Static Code Gen.) | N/A (Planner-based) |
| CoRAL (Ours) | Vision, Language, Tactile/Force | LLM-guided MPPI + Reactive Control | Explicit LLM Reasoning (Strategy + Adaptation) | Zero-Shot (No Demos) |

**说明**: TABLE S1: Comparison with State-of-the-Art VLA Manipulation Frameworks
## 实验解读

- 评价重点:围绕 adaptive-control、closed-loop-control、接触推理、接触丰富操作、language-conditioned-policy,优先看真实机器人结果、扰动恢复、接触稳定性、控制误差、成功率、实时性和 sim-to-real 差距。
- System0 视角:实验结论不只看最终成功率,还要看闭环过程中是否能处理状态估计误差、执行延迟、碰撞/接触突变和外部扰动。
- 与上层系统的关系:如果方法能在快闭环内稳定工作,它可以作为 System1 技能或 System2 规划的底层执行保障;如果只在离线或慢速设定下成立,则还不能直接作为身体级快闭环模块。
## 批判性思考

### 优点

- 论文把 adaptive-control、closed-loop-control、接触推理、接触丰富操作、language-conditioned-policy 放到机器人身体级闭环里讨论,和只做高层语义推理的工作相比,更接近真实机器人部署时会遇到的状态、动力学和安全约束。
- 若论文同时给出公式、系统结构图、硬件实验或消融结果,这些证据可以帮助判断方法是否真的能支撑毫秒到百毫秒级反应。

### 局限

- 需要重点核查控制频率、推理延迟、传感延迟、失败恢复、硬件磨损、未见场景泛化和安全边界;这些因素决定它能否成为可靠的 System0 模块。
- 对学习型方法,还要区分训练环境中的鲁棒性和真实部署中的鲁棒性,尤其是接触丰富任务、动态运动任务和人形机器人任务。
## 论文定位

- 研究对象:CoRAL: Contact-Rich Adaptive LLM-based Control for Robotic Manipulation。
- 关键词:adaptive-control、closed-loop-control、接触推理、接触丰富操作、language-conditioned-policy、reactive-control、实时控制、机器人操作、仿真到真实迁移、vision-language-action。
- System0 关联:关注机器人身体级快反应链路中“状态/接触/局部感知 -> 约束/优化/策略 -> 动作”的短闭环,不把问题停留在语言规划或任务分解层。

## 速查卡片

> [!summary] CoRAL
> - **论文**: https://www.roboticsproceedings.org/rss22/p054.pdf
> - **arXiv**: http://arxiv.org/abs/2605.02600v2
> - **arXiv HTML**: https://arxiv.org/html/2605.02600v2
