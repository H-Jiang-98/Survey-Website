---
title: "Learning to Evolve: Multi-modal Interactive Fields for Robust Humanoid Navigation in Dynamic Environments"
method_name: "Learning to Evolve"
authors: ["Peifeng Jiang"]
year: 2026
venue: "RSS"
tags: ["robot-manipulation", "robust-control", "legged-locomotion", "safe-control", "adaptive-control", "robot-generalization", "closed-loop-control", "humanoid", "navigation"]
image_source: "online"
arxiv_html: "https://arxiv.org/html/2605.21935v1"
---
# Learning to Evolve
## 一句话总结

> Learning to Evolve: Multi-modal Interactive Fields for Robust Humanoid Navigation in Dynamic Environments 主要落在 [[adaptive-control]]、[[closed-loop-control]]、[[flow-matching]]、[[人形机器人]] 这一类问题上;从 System0 视角看,它的价值在于把身体状态、接触变化、局部感知或动力学约束接到机器人低层闭环里,让动作生成更稳定、安全、实时。
## 核心贡献

1. 围绕 **Learning to Evolve: Multi-modal Interactive Fields for Robust Humanoid Navigation in Dynamic Environments** 建立了一个与 adaptive-control、closed-loop-control、flow-matching、人形机器人、足式运动、navigation 相关的机器人问题设定,重点不是高层语义规划,而是身体级闭环中的状态变化、接触变化或控制约束。
2. 方法层面的关键价值在于把感知、动力学、约束优化、策略学习或硬件反馈组织成可执行的低层动作生成机制,适合放进 System0 的快反应模块中理解。
3. 对系统集成而言,这类工作补上了 System1 技能选择与电机级控制之间的一层:它负责把上层目标变成动态可行、可恢复、可安全过滤的动作。
4. 若论文包含真实机器人、接触任务或仿真到真实迁移,核心应看它是否报告了闭环稳定性、扰动恢复、硬件延迟、失败模式和跨场景泛化。
## 问题背景

这篇论文可以放在 System0 的“具身快系统”背景下阅读:机器人需要在很短时间内根据本体状态、接触、力/触觉、局部视觉或动力学预测做出动作,而不是等待慢速规划器重新推理整个任务。adaptive-control、closed-loop-control、flow-matching、人形机器人、足式运动、navigation、机器人操作 相关问题的难点通常不在“知道要做什么”,而在动作执行过程中状态会快速偏离、接触会突然改变、约束会实时收紧,低层系统必须及时修正。

### System0 关联

精读时优先看三件事:第一,论文使用哪些快速反馈量,例如关节状态、接触状态、力觉、触觉、深度或局部视觉;第二,反馈如何进入 MPC、WBC、策略网络、安全过滤器或恢复控制器;第三,输出是否直接影响机器人下一步动作,而不是只服务离线分析或高层任务规划。
## 方法详解

### System0 视角解析

从 System0 角度看,这篇论文的方法部分应当围绕 adaptive-control、closed-loop-control、flow-matching、人形机器人、足式运动、navigation、机器人操作、鲁棒控制 来读。核心问题是:系统如何从快速变化的机器人状态中构造控制输入,如何处理接触、碰撞、约束、模型误差或执行延迟,以及如何把这些信息变成下一时刻的动作。

如果方法里有优化问题,重点看目标函数、约束和求解频率;如果方法里有学习策略,重点看观测空间、动作空间、训练扰动和部署时的闭环频率;如果方法里有硬件或传感器,重点看传感链路是否真的进入控制回路。这样读可以把论文和 System2 语言规划、System1 技能编排区分开来。

### 关键公式与机制

#### 公式 1: [[优化目标/约束]]

$$v_{i}=\{\mathcal{S}_{i},\Omega_{i},\mathcal{A}_{i},\mathbf{c}_{i}\}$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 2: [[优化目标/约束]]

$$\mathbf{c}_{i}=\frac{\sum_{j\in\mathcal{N}_{i}}C_{j}\alpha_{j}\mathbf{x}_{j}}{\sum_{j\in\mathcal{N}_{i}}C_{j}\alpha_{j}},$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 3: [[动力学或策略机制]]

$$\delta_{i}=\Omega_{i}\cdot\left(w_{pos}\lVert \mathbf{c}_{i}^{loc}-\mathbf{c}_{j}^{glob} \rVert_{2}+w_{sem}(1-\cos(\hat{\mathbf{f}}_{i},\hat{\mathbf{f}}_{j}))\right).$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 4: [[优化目标/约束]]

$$\mathcal{D}=\frac{1}{| $\mathcal{V}_{loc}$ | } $\sum_{i\in\mathcal{V}_{loc}}\delta_{i}+w_{rel}\frac{$ | $\mathcal{E}_{loc}\Delta\mathcal{E}_{spat}$ |}{| $\mathcal{E}_{loc}\cup\mathcal{E}_{spat}$ |}.$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 5: [[优化目标/约束]]

$$Q(k)=\exp(-\lVert \mathbf{c}_{i}-\mathbf{p}_{k} \rVert_{2}^{2}/2\sigma_{d}^{2})\cdot(\mathbf{d}_{k}\cdot\mathbf{v}_{ik})^{\gamma}\cdot\frac{1}{| $\mathcal{N}_{i}$ |}\sum_{j\in\mathcal{N}_{i}}\Omega_{j},$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 6: [[优化目标/约束]]

$$\min_{\mathbf{R},\mathbf{t},s}\sum_{m=1}^{M}\rho\left(\lVert s\mathbf{R}\mathbf{x}_{gen}^{m}+\mathbf{t}-\mathbf{x}_{app}^{NN(m)} \rVert\right).$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 7: [[优化目标/约束]]

$$S_{IPS}=\mathbf{I}_{col}(\mathbf{p}_{se3},\mathcal{M}_{gen})\land\mathbf{I}_{ik}(\mathbf{p}_{se3},\mathbf{t}_{obj})\land\mathbf{I}_{stab}(\mathbf{p}_{se3}),$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 8: [[优化目标/约束]]

$$\mathbf{c}_{goal}\leftarrow\text{RefineCentroid}(\mathbf{v}_{target},\mathcal{F}_{app})$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 9: [[优化目标/约束]]

$$\mathcal{M}_{gen}\leftarrow\text{FlowMatching}(I_{best},\text{cond}=\mathbf{v}_{target}.\text{sem})$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 10: [[优化目标/约束]]

$$\mathbf{F}^{distill}(\mathbf{p})=\sum_{i\in\mathcal{N}}C_{i}\mathbf{f}_{i}^{distill}\alpha_{i}T_{i},\quad\text{where}T_{i}=\prod_{j=1}^{i-1}(1-C_{j}\alpha_{j}).$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。
## 关键图表

### Figure 1: Overview of the Multi-modal Interactive Fields (MIF) framework. (Red) Incremental Appe

![Figure 1](https://arxiv.org/html/2605.21935v1/images/RSS_PIPELINE2.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Overview of the Multi-modal Interactive Fields (MIF) framework. (Red) Incremental Appe”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 2: Humanoid navigation control via Pure Pursuit. The schematic illustrates the geometric

![Figure 2](https://arxiv.org/html/2605.21935v1/x1.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Humanoid navigation control via Pure Pursuit. The schematic illustrates the geometric”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 3: Figure S.2: System architecture and dataflow. The architecture is divided between the Onboard Ro

![Figure 3](https://arxiv.org/html/2605.21935v1/images/SYSTEM_PIPELINE.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Figure S.2: System architecture and dataflow. The architecture is divided between the Onboard Ro”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### 论文表格

#### Table 1: TABLE I: Quantitative results of semantic grounding locomotion-induced distortion.

| Method | Success Rate | MDE (m) | Latency (s) |
| --- | --- | --- | --- |
| Feature-Splatting [36 ] | 65% | 0.42 | 3.2 |
| HOV-SG [46 ] | 74% | 0.35 | 2.8 |
| MIF (Ours) | 92% | 0.18 | 1.6 |

**说明**: TABLE I: Quantitative results of semantic grounding locomotion-induced distortion.

#### Table 2: TABLE II: Rendering results slow (S-) and fast (F-) walk.

| Method | S-PSNR ↑ \uparrow | S-SSIM ↑ \uparrow | F-PSNR ↑ \uparrow | F-SSIM ↑ \uparrow |
| --- | --- | --- | --- | --- |
| w/o Confidence | 28.4 | 0.88 | 21.6 | 0.72 |
| Deblur-GS [32 ] | 25.14 | 0.80 | 24.67 | 0.70 |
| PUP-3DGS [17 ] | 20.44 | 0.67 | 17.89 | 0.53 |
| MIF C i C_{i} (Ours) | 31.2 | 0.93 | 29.8 | 0.89 |

**说明**: TABLE II: Rendering results slow (S-) and fast (F-) walk.

#### Table 3: TABLE III: Semantic object localization on scene graphs constructed from rendered vs. raw frames.

| Input Source | Recall ↑ \uparrow | Precision ↑ \uparrow | mAP ↑ \uparrow |
| --- | --- | --- | --- |
| Raw frames (Slow) | 0.87 | 0.91 | 0.84 |
| Raw frames (Fast) | 0.61 | 0.79 | 0.58 |
| MIF Rendered (Slow) | 0.94 | 0.95 | 0.93 |
| MIF Rendered (Fast) | 0.92 | 0.94 | 0.91 |
| Static camera (Ref.) | ∼ \sim 1.00 | ∼ \sim 1.00 | ∼ \sim 1.00 |

**说明**: TABLE III: Semantic object localization on scene graphs constructed from rendered vs. raw frames.

#### Table 4: TABLE IV: Interaction Feasibility Comparison. We compare our generative approach against a Pt-Cloud Baseline and a Stati

| Method | IPS Success (%) | Nav. Err. (m) | Collision Rate |
| --- | --- | --- | --- |
| Pt-Cloud | 62% | 0.28 | 32% |
| Static Mesh | 78% | 0.22 | 14% |
| MIF (Ours) | 94% | 0.12 | 0% |

**说明**: TABLE IV: Interaction Feasibility Comparison. We compare our generative approach against a Pt-Cloud Baseline and a Static Mesh Baseline.

#### Table 5: TABLE V: ROC-based threshold selection in real scenarios.

| Threshold $\boldsymbol{\tau}$ | True Positive Rate ↑ \uparrow | False Positive Rate ↓ \downarrow | F1 ↑ \uparrow |
| --- | --- | --- | --- |
| 0.25 | 99.2% | 18.7% | 0.817 |
| 0.35 | 97.8% | 9.3% | 0.893 |
| 0.45 (Ours) | 95.1% | 4.2% | 0.934 |
| 0.55 | 89.6% | 1.8% | 0.913 |
| 0.65 | 78.3% | 0.7% | 0.866 |

**说明**: TABLE V: ROC-based threshold selection in real scenarios.

#### Table 6: TABLE VI: Scene-change task success without a separate global rescan before task execution. Reloc/Remov/Addit denote rel

| Method | Reloc SR ↑ \uparrow | Remov SR ↑ \uparrow | Addit SR ↑ \uparrow |
| --- | --- | --- | --- |
| HOV-SG [46 ] | 12% | 10% | 0% |
| ConceptGraphs + Rescan [16 ] | 38% | 35% | 12% |
| Khronos [41 ] | ∼ \sim 18% | 61% | ∼ \sim 12% |
| MIF (Initial) | 64% | 68% | 42% |
| MIF (Post-Update) | 94% | 98% | 86% |

**说明**: TABLE VI: Scene-change task success without a separate global rescan before task execution. Reloc/Remov/Addit denote relocated (> 1.5 >1.5 m), removed, and newly added objects. MIF (Initial) uses local observations near the obsolete memory location without local memory revision, while MIF (Post-Update) reports final task success after active scanning, local graph revision, and replanning are enabled.

#### Table 7: TABLE VII: Semantic localization slow (S-) and fast (F-) walk conditions, memory usage, and latency.

| Method | VRAM ↓ \downarrow (GB) | Latency ↓ \downarrow (s) | S-SR ↑ \uparrow | F-SR ↑ \uparrow |
| --- | --- | --- | --- | --- |
| LangSplat [35 ] | 18.45 | 2.3 | 95% | 72% |
| LatentBKI-64D [47 ] | 23.5 | 7.1 | 98% | 74% |
| FeatSplat-32D [36 ] | 4.8 | 3.2 | 81% | 59% |
| MIF 64D (Ours) | 18.0 | 4.6 | 92% | 88% |
| MIF 32D (Ours) | < < 4.0 | 1.6 | 90% | 86% |

**说明**: TABLE VII: Semantic localization slow (S-) and fast (F-) walk conditions, memory usage, and latency.

#### Table 8: TABLE S.1: Parameters for confidence-aware filtering.

| Symbol | Value | Description |
| --- | --- | --- |
|  \beta | 5.0 | Controls sensitivity to normalized optimization instability. |
| $\gamma$ | 2.0 | Scaling factor for opacity-based reliability. |
| $\tau_{conf}$ | 0.6 | Minimum confidence threshold for VLM rendering. |

**说明**: TABLE S.1: Parameters for confidence-aware filtering.

#### Table 9: TABLE S.2: Parameters for discrepancy detection.

| Symbol | Value | Description of Weight or Threshold |
| --- | --- | --- |
| w p  o  s w_{pos} | 1.0 | Centroid Euclidean distance error (in meters). |
| w s  e  m w_{sem} | 0.5 | CLIP/DINO semantic feature cosine distance. |
| w r  e  l w_{rel} | 0.8 | Jaccard distance of graph edge sets. |
| $\tau$ | 0.45 | Threshold of D  $\mathcal{D} for local memory update and replanning.$ |

**说明**: TABLE S.2: Parameters for discrepancy detection.

#### Table 10: TABLE S.3: Extended ablation of confidence gating walking motion.

| Method | Slow Walk (0.2  m/s 0.2\, $\text{m/s})$ | Fast Walk (0.5  m/s 0.5\, $\text{m/s})$ | Change Det. | | |
| --- | --- | --- | --- | --- | --- |
| PSNR ↑ \uparrow | SSIM ↑ \uparrow | PSNR ↑ \uparrow | SSIM ↑ \uparrow | FPR ↓ \downarrow | |
| w/o Gating | 28.4 | 0.88 | 21.6 | 0.72 | 46.5% |
| MIF (Ours) | 31.2 | 0.93 | 29.8 | 0.89 | 4.2% |

**说明**: TABLE S.3: Extended ablation of confidence gating walking motion.
## 实验解读

- 评价重点:围绕 adaptive-control、closed-loop-control、flow-matching、人形机器人、足式运动,优先看真实机器人结果、扰动恢复、接触稳定性、控制误差、成功率、实时性和 sim-to-real 差距。
- System0 视角:实验结论不只看最终成功率,还要看闭环过程中是否能处理状态估计误差、执行延迟、碰撞/接触突变和外部扰动。
- 与上层系统的关系:如果方法能在快闭环内稳定工作,它可以作为 System1 技能或 System2 规划的底层执行保障;如果只在离线或慢速设定下成立,则还不能直接作为身体级快闭环模块。
## 批判性思考

### 优点

- 论文把 adaptive-control、closed-loop-control、flow-matching、人形机器人、足式运动 放到机器人身体级闭环里讨论,和只做高层语义推理的工作相比,更接近真实机器人部署时会遇到的状态、动力学和安全约束。
- 若论文同时给出公式、系统结构图、硬件实验或消融结果,这些证据可以帮助判断方法是否真的能支撑毫秒到百毫秒级反应。

### 局限

- 需要重点核查控制频率、推理延迟、传感延迟、失败恢复、硬件磨损、未见场景泛化和安全边界;这些因素决定它能否成为可靠的 System0 模块。
- 对学习型方法,还要区分训练环境中的鲁棒性和真实部署中的鲁棒性,尤其是接触丰富任务、动态运动任务和人形机器人任务。
## 论文定位

- 研究对象:Learning to Evolve: Multi-modal Interactive Fields for Robust Humanoid Navigation in Dynamic Environments。
- 关键词:adaptive-control、closed-loop-control、flow-matching、人形机器人、足式运动、navigation、机器人操作、鲁棒控制、safe-control。
- System0 关联:关注机器人身体级快反应链路中“状态/接触/局部感知 -> 约束/优化/策略 -> 动作”的短闭环,不把问题停留在语言规划或任务分解层。

## 速查卡片

> [!summary] Learning to Evolve
> - **论文**: https://www.roboticsproceedings.org/rss22/p023.pdf
> - **arXiv**: http://arxiv.org/abs/2605.21935v1
> - **arXiv HTML**: https://arxiv.org/html/2605.21935v1
