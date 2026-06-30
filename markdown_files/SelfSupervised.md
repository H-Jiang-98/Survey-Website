---
title: "Self-Supervised Bootstrapping of Action-Predictive Embodied Reasoning"
method_name: "Self Supervised Bootstrapping"
authors: ["Milan Ganai"]
year: 2026
venue: "RSS"
tags: ["contact-reasoning", "robot-manipulation", "robust-control", "legged-locomotion", "robot-generalization", "collision-avoidance", "navigation"]
image_source: "online"
arxiv_html: "https://arxiv.org/html/2602.08167v2"
---
# Self Supervised Bootstrapping
## 一句话总结

> Self-Supervised Bootstrapping of Action-Predictive Embodied Reasoning 主要落在 [[action-model]]、[[biped]]、[[碰撞避免]]、[[接触推理]] 这一类问题上;从 System0 视角看,它的价值在于把身体状态、接触变化、局部感知或动力学约束接到机器人低层闭环里,让动作生成更稳定、安全、实时。
## 核心贡献

1. 围绕 **Self-Supervised Bootstrapping of Action-Predictive Embodied Reasoning** 建立了一个与 action-model、biped、碰撞避免、接触推理、language-conditioned-policy、足式运动 相关的机器人问题设定,重点不是高层语义规划,而是身体级闭环中的状态变化、接触变化或控制约束。
2. 方法层面的关键价值在于把感知、动力学、约束优化、策略学习或硬件反馈组织成可执行的低层动作生成机制,适合放进 System0 的快反应模块中理解。
3. 对系统集成而言,这类工作补上了 System1 技能选择与电机级控制之间的一层:它负责把上层目标变成动态可行、可恢复、可安全过滤的动作。
4. 若论文包含真实机器人、接触任务或仿真到真实迁移,核心应看它是否报告了闭环稳定性、扰动恢复、硬件延迟、失败模式和跨场景泛化。
## 问题背景

这篇论文可以放在 System0 的“具身快系统”背景下阅读:机器人需要在很短时间内根据本体状态、接触、力/触觉、局部视觉或动力学预测做出动作,而不是等待慢速规划器重新推理整个任务。action-model、biped、碰撞避免、接触推理、language-conditioned-policy、足式运动、navigation 相关问题的难点通常不在“知道要做什么”,而在动作执行过程中状态会快速偏离、接触会突然改变、约束会实时收紧,低层系统必须及时修正。

### System0 关联

精读时优先看三件事:第一,论文使用哪些快速反馈量,例如关节状态、接触状态、力觉、触觉、深度或局部视觉;第二,反馈如何进入 MPC、WBC、策略网络、安全过滤器或恢复控制器;第三,输出是否直接影响机器人下一步动作,而不是只服务离线分析或高层任务规划。
## 方法详解

### System0 视角解析

从 System0 角度看,这篇论文的方法部分应当围绕 action-model、biped、碰撞避免、接触推理、language-conditioned-policy、足式运动、navigation、quadruped 来读。核心问题是:系统如何从快速变化的机器人状态中构造控制输入,如何处理接触、碰撞、约束、模型误差或执行延迟,以及如何把这些信息变成下一时刻的动作。

如果方法里有优化问题,重点看目标函数、约束和求解频率;如果方法里有学习策略,重点看观测空间、动作空间、训练扰动和部署时的闭环频率;如果方法里有硬件或传感器,重点看传感链路是否真的进入控制回路。这样读可以把论文和 System2 语言规划、System1 技能编排区分开来。

### 关键公式与机制

#### 公式 1: [[优化目标/约束]]

$$\log p(A)=\log\operatorname*{\mathbb{E}}\left[\frac{1}{K}\sum_{k=1}^{K}w_{k}\right]\geq\operatorname*{\mathbb{E}}\left[\log\frac{1}{K}\sum_{k=1}^{K}w_{k}\right]=\mathcal{L}_{K},$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 2: [[优化目标/约束]]

$$\mathbb{E}_{A\sim p_{\text{data}}}\left[\log\frac{\mathbb{E}_{Z\sim q}[w(Z_{\mathbf{R}})\mid Z_{\mathbf{R}}\in\mathcal{Z}_{\mathbf{R}}]}{\mathbb{E}_{Z\sim q}[w(Z_{\cancel{\mathbf{R}}})\mid Z_{\cancel{\mathbf{R}}}\in\mathcal{Z}_{\cancel{\mathbf{R}}}]}\right]=\Delta\mathcal{I}_{\mathbf{R}}.$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 3: [[优化目标/约束]]

$$\frac{1}{K}\sum_{k=1}^{K}w(Z_{k})\cdot\mathbf{1}_{Z_{k}\in\mathcal{Z}_{\mathbf{R}}}\xrightarrow{K\to\infty}\operatorname*{\mathbb{E}}_{Z\sim q}[w(Z)\cdot\mathbf{1}_{Z\in\mathcal{Z}_{\mathbf{R}}}].$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 4: [[优化目标/约束]]

$$=\ {\left(\int_{\mathcal{Z}_{\mathbf{R}}}q(z| C,A)\,dz $\right)}_{q(\mathcal{Z}_{\mathbf{R}}$ | C,A)}\cdot\ { $\frac{\int_{\mathcal{Z}_{\mathbf{R}}}w(z)\,q(z\mid C,A)\,dz}{\int_{\mathcal{Z}_{\mathbf{R}}}q(z$ | C,A)\,dz}}_{ $\operatorname*{\mathbb{E}}_{Z\sim q}[w(Z)$ |Z\in\mathcal{Z}_{\mathbf{R}}]}$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 5: [[优化目标/约束]]

$$=\operatorname*{\mathbb{E}}_{Z_{1},\ldots,Z_{K}\sim q(Z|C,A)}\left[\frac{\frac{1}{K}\sum_{k=1}^{K}w(Z_{k})\cdot\mathbf{1}_{Z_{k}\in\mathcal{Z}_{\mathbf{R}}}}{\frac{1}{K}\sum_{k^{\prime}=1}^{K}w(Z_{k^{\prime}})}\right].$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 6: [[优化目标/约束]]

$$=\operatorname*{\mathbb{E}}_{Z_{1},\ldots,Z_{K}\sim q(Z|C,A)}\left[\sum_{k=1}^{K}\frac{w(Z_{k})}{\sum_{k^{\prime}=1}^{K}w(Z_{k^{\prime}})}\cdot\mathbf{1}_{Z_{k}\in\mathcal{Z}_{\mathbf{R}}}\right]$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 7: [[优化目标/约束]]

$$=\operatorname*{\mathbb{E}}_{A\sim p_{\text{data}}}\left[\log\frac{\operatorname*{\mathbb{E}}_{Z\sim q}[w(Z)| Z\in $\mathcal{Z}_{\mathbf{R}}]}{\operatorname*{\mathbb{E}}_{Z\sim q}[w(Z)$ |Z\in\mathcal{Z}_{\cancel{\mathbf{R}}}]}+\log\frac{1-d}{d}\right]$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 8: [[优化目标/约束]]

$$=\ {\operatorname*{\mathbb{E}}_{A\sim p_{\text{data}}}\left[\log\frac{\operatorname*{\mathbb{E}}_{Z\sim q}[w(Z)| Z\in $\mathcal{Z}_{\mathbf{R}}]}{\operatorname*{\mathbb{E}}_{Z\sim q}[w(Z)$ |Z\in\mathcal{Z}_{\cancel{\mathbf{R}}}]}\right]}_{\Delta\mathcal{I}_{\mathbf{R}}\text{(by Proposition of Sec~)}}+\ {\log\frac{1-d}{d}}_{\text{Warmstart Preference}}$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 9: [[优化目标/约束]]

$$\textbf{Refined Preference for}\mathbf{R}:=\operatorname*{\mathbb{E}}_{A\sim p_{\text{data}}}\left[\log\frac{\tilde{q}(\mathcal{Z}_{\mathbf{R}}| C,A)}{ $\tilde{q}(\mathcal{Z}_{\cancel{\mathbf{R}}}$ |C,A)}\right].$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 10: [[动力学或策略机制]]

$$\operatorname*{\mathbb{E}}_{Z_{1:K}\sim q}\left[\frac{1}{K}\sum_{k=1}^{K}\frac{p(Z_{k},A)}{q(Z_{k}\mid A)}\right]=\frac{1}{K}\sum_{k=1}^{K}\operatorname*{\mathbb{E}}_{Z_{k}\sim q}\left[\frac{p(A,Z_{k})}{q(Z_{k}\mid A)}\right]$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。
## 关键图表

### Figure 1: Pipeline / core system figure: We generate diverse embodied reasoning primitives and refine them based on action-pred

![Figure 1](https://arxiv.org/html/2602.08167v2/x1.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Pipeline / core system figure: We generate diverse embodied reasoning primitives and refine them based on action-pred”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 2: Overview of R&B-EnCoRe. (a) We generate diverse reasoning primitives (e.g., Plan, Vis

![Figure 2](https://arxiv.org/html/2602.08167v2/x3.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Overview of R&B-EnCoRe. (a) We generate diverse reasoning primitives (e.g., Plan, Vis”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 3: Prior and Posterior architecture. The prior architecture is the same as the standard

![Figure 3](https://arxiv.org/html/2602.08167v2/x15.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Prior and Posterior architecture. The prior architecture is the same as the standard”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### 论文表格

#### Table 1: TABLE I: R&B-EnCoRe autonomously identifies critical objects in LIBERO-90 data that are most predictive of expert action

| Training Strategy | Success Rate | Object Criticality Rate |
| --- | --- | --- |
| No Reasoning | 75.9% | N/A |
| List of All Objects | 76.1% | 0.03% |
| List of Random Objects | 77.0% | 3.43% |
| R&B-EnCoRe on Object List | 80.3% | 25.02% |

**说明**: TABLE I: R&B-EnCoRe autonomously identifies critical objects in LIBERO-90 data that are most predictive of expert actions, pruning irrelevant objects and improving task success.

#### Table 2: TABLE II: Results on analyzing reasoning primitives in LIBERO-90: Plan, Visible Objects, Subtask, Subtask Explain, Move,

| Training Strategy | Success Rate | Avg. # Gen. Token |
| --- | --- | --- |
| No Reasoning | 75.9% | 10.0 |
| All Primitives | 78.6% | 256.8 |
| Random Primitives | 76.5% | 256.6 |
| R&B-EnCoRe | 79.5% | 129.3 |

**说明**: TABLE II: Results on analyzing reasoning primitives in LIBERO-90: Plan, Visible Objects, Subtask, Subtask Explain, Move, Move Explain, and Gripper Position. R&B-EnCoRe refines action-predictive reasoning, resulting in shorter traces and improved success rate.

#### Table 3: TABLE III: Success rates on WidowX hardware Bridgev2 setup. Reasoning Models are prompted and/or trained with Action for

| Category | Task | No Reason | All Primitives | Random | R&B-EnCoRe |
| --- | --- | --- | --- | --- | --- |
| In Distrib. | put red pepper in yellow basket | 69.2% | 100.0% | 92.3% | 100.0% |
| put red pepper on black stove |.8% | 92.3% |.2% |.6% | |
| put orange carrot on pink plate | 61.5% | 76.9% | 84.6% | 84.6% | |
| OOD Target Object | put blue peacock in sink |.2% |.5% |.2% | 76.9% |
| put orange tape on green towel (include orange carrot) | 38.5% | 38.5% | 38.5% | 76.9% | |
| put pink pepto on green towel |.2% |.8% |.8% | 69.2% | |
| OOD Scene with Distracting Objects | put yellow corn on blue plate (include pink plate, carrot) | 53.8% | 69.2% | 38.5% | 76.9% |
| put orange carrot in yellow basket (include distraction objects in basket, sink) |.5% | 92.3% |.6% | 92.3% | |
| first put yellow corn in yellow basket then put red pepper in yellow basket (human takes corn) | 46.2% | 30.8% | 46.2% | 61.5% | |

**说明**: 该表格来自论文中的公式、奖励项或实验设置；为避免 PDF 抽取噪声，网页中仅保留可渲染的核心符号和数值。

#### Table 4: TABLE IV: Performance comparison of AV with UniAD metrics. R&B-EnCoRe is able to refine reasoning traces and reduce devi

| | L2 Path Error (m) | Collision Rate (%) | | | | | | |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Method | 1s | 2s | 3s | Avg | 1s | 2s | 3s | Avg |
| No Reasoning | 0.22 | 0.62 | 1.34 | 0.72 | 0.10 | 0.25 | 1.11 | 0.49 |
| Full Reasoning | 0.22 | 0.64 | 1.33 | 0.73 | 0.07 | 0.17 | 0.91 | 0.38 |
| Random | 0.22 | 0.62 | 1.31 | 0.72 | 0.05 | 0.20 | 0.81 | 0.35 |
| R&B-EnCoRe | 0.21 | 0.59 | 1.25 | 0.68 | 0.05 | 0.17 | 0.70 | 0.30 |

**说明**: TABLE IV: Performance comparison of AV with UniAD metrics. R&B-EnCoRe is able to refine reasoning traces and reduce deviation from ground truth path and collision rate. For comparison, zero-shot native chain-of-thought achieves suboptimal 10.35m average L2 error.

#### Table 5: TABLE V: Notation and Symbols used in paper

| Symbol | Description |
| --- | --- |
| Core Variables | |
| C C | Context (observation image and task description) |
| A A | Action (discrete action tokens, 2D waypoints, etc) |
| Z Z | Latent reasoning trace (unobserved strategy) |
| R R | Reasoning primitve (e.g., Affordance reasoning) |
|  \rho | Number of reasoning primitive types |
| R  $\mathcal{R}$ | Set of reasoning primitives {R 1, ..., R  } \{R_{1},\ldots,R_{\rho}\} |
| R  $\mathbf{R}$ | Reasoning Strategy, subset of reasoning primitives |
| z R z_{R} | Textual content for reasoning primitive R ∈ R R\in $\mathcal{R}$ |
| Probabilistic Models | |
| p  (A ∣ C) p(A\mid C) | Marginal action distribution given context |
| p  (Z, A ∣ C) p(Z,A\mid C) | Prior distribution (joint reasoning and action) |
| p  (A ∣ C, Z) p(A\mid C,Z) | Action likelihood given context and reasoning |
| p  (Z ∣ C) p(Z\mid C) | Prior reasoning distribution |
| q  (Z ∣ C, A) q(Z\mid C,A) | Posterior distribution (reasoning given action) |
| p data  (A ∣ C) p_{ $\text{data}}(A\mid C)$ | Expert action distribution |
| Variational Inference | |
| K K | Number of posterior samples in IWAE |
| Z 1, ..., Z K Z_{1},\ldots,Z_{K} | K K i.i.d. samples from posterior q  (Z ∣ C, A) q(Z\mid C,A) |
| w k, w  (Z k) w_{k},w(Z_{k}) | Importance weight p  (Z k, A ∣ C) q  (Z k ∣ C, A)  $\frac{p(Z_{k},A\mid C)}{q(Z_{k}\mid C,A)}$ |
| L K  $\mathcal{L}_{K}$ | K K -sample importance-weighted lower bound |
| ELBO VAE  $\text{ELBO}_{\text{VAE}}$ | Evidence lower bound for standard VAE |
| $\text{KL}}(\cdot\lVert \cdot)$ | Kullback-Leibler divergence |
| Datasets and Training | |
| D  $\mathcal{D}$ | Original dataset: {(C i, A i) } i = 1 N \{(C^{i},A^{i})\}_{i=1}^{N} |
| N N | Number of demonstrations in dataset |
| D warm  $\mathcal{D}_{\text{warm}}$ | Warmstart dataset with diverse reasoning traces |
| D refined  $\mathcal{D}_{\text{refined}}$ | Refined dataset with action-predictive reasoning |
| M M | Number of warmstarting traces per demonstration |
| d d | Reasoning dropout rate |
| Z j i Z^{i}_{j} | j j -th synthetic reasoning trace for demonstration i i |
| Z i ⁣ ∗ Z^{i*} | Refined, importance-sampled reasoning trace |
| Models | |
| M  $\mathcal{M}$ | Base vision-language model |
| M p  q  $\mathcal{M}_{pq}$ | Jointly trained prior-posterior model |
| M VLA  $\mathcal{M}_{\text{VLA}}$ | Final vision-language-action model |
| FM | Foundation model for generated reasoning content |
| Information Benefit | |
|   I R \Delta I_{R} | Information benefit of reasoning strategy R R |
| Z R  $\mathcal{Z}_{R}$ | Set of reasoning traces containing strategy R R |
| Z R  $\mathcal{Z}_{R}$ | Set of reasoning traces not containing strategy R R |
| E Z ∼ q [w  (Z) ∣ Z ∈ Z R ]  $\operatorname*{\mathbb{E}}_{Z\sim q}[w(Z)\mid Z\in\mathcal{Z}_{\mathbf{R}}]$ | Expected importance weight of reasoning traces in set Z R  $\mathcal{Z}_{R}$ |

**说明**: TABLE V: Notation and Symbols used in paper

#### Table 6: TABLE VI: Experimental Configuration Details Across Embodiment Domains

| Configuration | LIBERO-90 | WidowX Hardware | Legged Robot | Autonomous Vehicle |
| --- | --- | --- | --- | --- |
| M M (warmstart traces) | 10 | 4 | 8 | 32 |
| K K (posterior samples) | 8 | 4 | 4 | 8,12,16,32 |
|  \rho (reasoning primitives) | 7 | 7 | 7,8 (w/ or w/o Weather) | 6 |
| Dropout rate d d | 0.2 | 0.2 | 0.1,0.3,0.5.0.7,0.9 | 0.5 |
| Posterior sampling temp. | 1.0 | 1.0 | 0.7 | 1.0 |
| Expert demo source | LIBERO-90 [50 ] | Bridgev2 [84 ] | NaviTrace [90 ] | nuScenes [14 ] |
| Primary reasoning | Llama 2 [79 ], Molmo [21 ] | Gemini 1.0 [29 ] | Qwen3-VL 30B MoE [3 ] | Human Annotations [56, 57 ] |
| primitive content source | | | | |
| | MiniVLA [5 ]: Qwen2.5 0.5B LLM [66 ], | OpenVLA [43 ]: Llama 2 7B LLM [79 ], | Qwen3-VL 30B MoE [3 ] | Qwen3-VL 4B Dense [3 ] |
| VLA architecture | DINOv2 [63 ] +SigLIP [98 ],VQ-VAE [82 ] | DINOv2 [63 ] +SigLIP [98 ] | | |

**说明**: TABLE VI: Experimental Configuration Details Across Embodiment Domains

#### Table 7: TABLE VII: WidowX Task Success Rate with Test-Time Reasoning Enabled. Evaluation includes 13 trials per task and model t

| Category | Task | All Primitives | R&B-EnCoRe |
| --- | --- | --- | --- |
| In Distrib. | put red pepper in yellow basket | 84.6% | 92.3% |
| OOD Scene w/Distr. | put red pepper in yellow basket | 53.8% | 84.6% |
| (include distraction objects in basket, sink) | | | |

**说明**: TABLE VII: WidowX Task Success Rate with Test-Time Reasoning Enabled. Evaluation includes 13 trials per task and model to compare performance when the model generates full chain-of-thought traces.
## 实验解读

- 评价重点:围绕 action-model、biped、碰撞避免、接触推理、language-conditioned-policy,优先看真实机器人结果、扰动恢复、接触稳定性、控制误差、成功率、实时性和 sim-to-real 差距。
- System0 视角:实验结论不只看最终成功率,还要看闭环过程中是否能处理状态估计误差、执行延迟、碰撞/接触突变和外部扰动。
- 与上层系统的关系:如果方法能在快闭环内稳定工作,它可以作为 System1 技能或 System2 规划的底层执行保障;如果只在离线或慢速设定下成立,则还不能直接作为身体级快闭环模块。
## 批判性思考

### 优点

- 论文把 action-model、biped、碰撞避免、接触推理、language-conditioned-policy 放到机器人身体级闭环里讨论,和只做高层语义推理的工作相比,更接近真实机器人部署时会遇到的状态、动力学和安全约束。
- 若论文同时给出公式、系统结构图、硬件实验或消融结果,这些证据可以帮助判断方法是否真的能支撑毫秒到百毫秒级反应。

### 局限

- 需要重点核查控制频率、推理延迟、传感延迟、失败恢复、硬件磨损、未见场景泛化和安全边界;这些因素决定它能否成为可靠的 System0 模块。
- 对学习型方法,还要区分训练环境中的鲁棒性和真实部署中的鲁棒性,尤其是接触丰富任务、动态运动任务和人形机器人任务。
## 论文定位

- 研究对象:Self-Supervised Bootstrapping of Action-Predictive Embodied Reasoning。
- 关键词:action-model、biped、碰撞避免、接触推理、language-conditioned-policy、足式运动、navigation、quadruped、机器人操作、鲁棒控制。
- System0 关联:关注机器人身体级快反应链路中“状态/接触/局部感知 -> 约束/优化/策略 -> 动作”的短闭环,不把问题停留在语言规划或任务分解层。

## 速查卡片

> [!summary] Self Supervised Bootstrapping
> - **论文**: https://www.roboticsproceedings.org/rss22/p064.pdf
> - **arXiv**: http://arxiv.org/abs/2602.08167v2
> - **arXiv HTML**: https://arxiv.org/html/2602.08167v2
