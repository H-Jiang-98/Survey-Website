---
title: "Semantic Contact Fields for Category-Level Generalizable Tool Manipulation"
method_name: "Semantic Contact Fields"
authors: ["Kevin Yuchen Ma"]
year: 2026
venue: "RSS"
tags: ["contact-reasoning", "robot-manipulation", "robust-control", "real-time-control", "reinforcement-learning", "contact-rich-manipulation", "robot-generalization", "sim-to-real", "tactile-feedback"]
image_source: "online"
arxiv_html: "https://arxiv.org/html/2602.13833"
---
# Semantic Contact Fields
## 一句话总结

> Semantic Contact Fields for Category-Level Generalizable Tool Manipulation 主要落在 [[接触推理]]、[[接触丰富操作]]、[[diffusion-policy]]、[[foundation-model]] 这一类问题上;从 System0 视角看,它的价值在于把身体状态、接触变化、局部感知或动力学约束接到机器人低层闭环里,让动作生成更稳定、安全、实时。
## 核心贡献

1. 围绕 **Semantic Contact Fields for Category-Level Generalizable Tool Manipulation** 建立了一个与 接触推理、接触丰富操作、diffusion-policy、foundation-model、haptic-feedback、language-conditioned-policy 相关的机器人问题设定,重点不是高层语义规划,而是身体级闭环中的状态变化、接触变化或控制约束。
2. 方法层面的关键价值在于把感知、动力学、约束优化、策略学习或硬件反馈组织成可执行的低层动作生成机制,适合放进 System0 的快反应模块中理解。
3. 对系统集成而言,这类工作补上了 System1 技能选择与电机级控制之间的一层:它负责把上层目标变成动态可行、可恢复、可安全过滤的动作。
4. 若论文包含真实机器人、接触任务或仿真到真实迁移,核心应看它是否报告了闭环稳定性、扰动恢复、硬件延迟、失败模式和跨场景泛化。
## 问题背景

这篇论文可以放在 System0 的“具身快系统”背景下阅读:机器人需要在很短时间内根据本体状态、接触、力/触觉、局部视觉或动力学预测做出动作,而不是等待慢速规划器重新推理整个任务。接触推理、接触丰富操作、diffusion-policy、foundation-model、haptic-feedback、language-conditioned-policy、实时控制 相关问题的难点通常不在“知道要做什么”,而在动作执行过程中状态会快速偏离、接触会突然改变、约束会实时收紧,低层系统必须及时修正。

### System0 关联

精读时优先看三件事:第一,论文使用哪些快速反馈量,例如关节状态、接触状态、力觉、触觉、深度或局部视觉;第二,反馈如何进入 MPC、WBC、策略网络、安全过滤器或恢复控制器;第三,输出是否直接影响机器人下一步动作,而不是只服务离线分析或高层任务规划。
## 方法详解

### System0 视角解析

从 System0 角度看,这篇论文的方法部分应当围绕 接触推理、接触丰富操作、diffusion-policy、foundation-model、haptic-feedback、language-conditioned-policy、实时控制、robot-generalization 来读。核心问题是:系统如何从快速变化的机器人状态中构造控制输入,如何处理接触、碰撞、约束、模型误差或执行延迟,以及如何把这些信息变成下一时刻的动作。

如果方法里有优化问题,重点看目标函数、约束和求解频率;如果方法里有学习策略,重点看观测空间、动作空间、训练扰动和部署时的闭环频率;如果方法里有硬件或传感器,重点看传感链路是否真的进入控制回路。这样读可以把论文和 System2 语言规划、System1 技能编排区分开来。

### 关键公式与机制

#### 公式 1: [[优化目标/约束]]

$$
\mathcal{L}_{total}=\lambda_{prob}\mathcal{L}_{prob}+\lambda_{force}\mathcal{L}_{force}
$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 2: [[优化目标/约束]]

$$c_{i}=P(contact|d_{i})=\exp\left(-\left(\frac{\max(-d_{i},0)}{\lambda}\right)^{k}\right)$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 3: [[动力学或策略机制]]

$$\mathbf{f}^{ext}_{i}=S(d_{i})\cdot\frac{\sum_{j}w_{ij}(F_{j}\mathbf{n}_{j})}{\sum_{j}w_{ij}}$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 4: [[动力学或策略机制]]

$$w_{ij}=\frac{1}{1+(\lambda_{dist}\lVert \mathbf{x}_{j}-p_{i} \rVert)^{2}}$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 5: [[动力学或策略机制]]

$$S(d_{i})=\sqrt{\text{ReLU}\left(1-\frac{d_{i}}{d_{thresh}}\right)}$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 6: [[优化目标/约束]]

$$
\mathcal{L}_{total}=\lambda_{prob}\mathcal{L}_{prob}+\lambda_{force}(\lambda_{mag}\mathcal{L}_{mag}+\lambda_{dir}\mathcal{L}_{dir})
$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 7: [[优化目标/约束]]

$$
\mathcal{L}_{mag}=w_{i}(\lVert \mathbf{f}^{pred} \rVert-\lVert \mathbf{f}^{gt} \rVert)^{2}
$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 8: [[优化目标/约束]]

$$
\mathcal{L}_{dir}=1-\frac{\mathbf{f}^{pred}\cdot\mathbf{f}^{gt}}{\max(\lVert \mathbf{f}^{pred} \rVert\lVert \mathbf{f}^{gt} \rVert,\epsilon)}
$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 9: [[优化目标/约束]]

$$\text{Eff Norm}=\min\left(1,\frac{\text{Eff}}{L_{\text{ref}}}\right)$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 10: [[动力学或策略机制]]

$$\left\lVert \mathbf{G}\mathbf{f}-\mathbf{W}_{tac}\right \rVert_{2}^{2}+\lambda\sum_{i\in C_{candidate}}\frac{\lVert \mathbf{f}_{i} \rVert_{2}^{2}}{c_{i}+\epsilon}$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。
## 关键图表

### Figure 1: Method Overview. Left: Contact Field Learning (III-B) Stage 1 learns general geome

![Figure 1](https://arxiv.org/html/2602.13833v2/x1.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Method Overview. Left: Contact Field Learning (III-B) Stage 1 learns general geome”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 2: Contact field model architecture. The network fuses tactile markers and force arrays

![Figure 2](https://arxiv.org/html/2602.13833v2/figures/Network_architecture_v2.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Contact field model architecture. The network fuses tactile markers and force arrays”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 3: Qualitative results in simulation. The predicted contact probabilities (bottom row

![Figure 3](https://arxiv.org/html/2602.13833v2/figures/sim_contact_field_viz.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Qualitative results in simulation. The predicted contact probabilities (bottom row”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### 论文表格

#### Table 1: TABLE I: Sim Evaluation: Architecture Capacity

| Model | F1 Score ↑ \uparrow | Force MSE ↓ \downarrow |
| --- | --- | --- |
| NCF [7 ] | 0.043 | N/A |
| No-Tactile | 0.539 | 0.0146 |
| Ablation - 2D Tactile Enc. | 0.531 | 0.0147 |
| Ablation - BCE Loss | 0.123 | 0.0146 |
| Ablation - No Cont. Prob. | N/A | 0.0158 |
| Ours (Sim-Only) | 0.587 | 0.0147 |

**说明**: TABLE I: Sim Evaluation: Architecture Capacity

#### Table 2: TABLE III: Task 1: Scraper Performance. Ours outperforms baselines on unseen tools, demonstrating robust generalization

| | Seen Tools | Unseen Tools | | | | |
| --- | --- | --- | --- | --- | --- | --- |
| Method | SR (%) | Eff (%) | Eff Norm (%) | SR (%) | Eff (%) | Eff Norm (%) |
| Vision-Only (GenDP) | 39.1 | 10.1 | 30.5 | 35.1 | 25.4 | 35.1 |
| Raw Tactile | 35.1 | 26.9 | 35.7 | 50.0 | 23.3 | 27.3 |
| Ours (SCFields) | 73.5 | 61.8 | 85.2 | 79.6 | 73.5 | 84.7 |
| Ablation: Sim-Only CF | 34.5 | 20.6 | 33.1 | 55.6 | 37.0 | 45.2 |
| Ablation: Real-Only CF | 45.8 | 32.3 | 44.5 | 50.0 | 47.5 | 54.2 |
| Ablation: No Force | 31.3 | 22.2 | 29.9 | 26.0 | 24.3 | 27.6 |

**说明**: TABLE III: Task 1: Scraper Performance. Ours outperforms baselines on unseen tools, demonstrating robust generalization.

#### Table 3: TABLE IV: Task 2: Crayon Drawing Consistency (Score 0-1).

| Method | Seen Crayons | Unseen Crayons |
| --- | --- | --- |
| Vision-Only (GenDP) | 0.81 | 0.60 |
| Raw Tactile | 0.76 | 0.61 |
| Ours (SCFields) | 0.86 | 0.78 |
| Ablation: Sim-Only CF | 0.81 | 0.60 |
| Ablation: Real-Only CF | 0.68 | 0.74 |
| Ablation: No Force | 0.80 | 0.76 |

**说明**: TABLE IV: Task 2: Crayon Drawing Consistency (Score 0-1).

#### Table 4: TABLE V: Task 3: Peeler Results. Aligned model performance validates the pipeline.

| | Seen Peelers | Unseen Peelers | | | | |
| --- | --- | --- | --- | --- | --- | --- |
| Method | Contact (%) | Cut-in (%) | Avg Peel Length (cm) | Contact (%) | Cut-in (%) | Avg Peel Length (cm) |
| Vision-Only | 45.0 | 30.0 | 1.50 | 50.0 | 33.3 | 1.12 |
| Raw Tactile | 45.0 | 20.0 | 1.05 | 40.0 | 30.0 | 0.85 |
| Ours (SCField) | 80.0 | 70.0 | 4.73 | 90.0 | 73.3 | 4.52 |
| Ablation: Sim-Only CF | 60.0 | 30.0 | 2.00 | 50.0 | 46.7 | 3.05 |
| Ablation: Real-Only CF | 60.0 | 15.0 | 0.93 | 57.5 | 50.0 | 1.80 |
| Ablation: No Force | 65.0 | 30.0 | 1.95 | 40.0 | 13.3 | 1.08 |

**说明**: TABLE V: Task 3: Peeler Results. Aligned model performance validates the pipeline.

#### Table 5: TABLE VI: Contact Field Model Training Hyperparameters

| Parameter | Stage 1 (Sim) | Stage 2 (Real) |
| --- | --- | --- |
| Optimizer | AdamW | AdamW |
| Learning Rate | 1  e - 4 1e^{-4} | 5  e - 6 5e^{-6} |
| LR Scheduler | ReduceLROnPlateau | ReduceLROnPlateau |
| Batch Size | 320 | 128 |
| Epochs | 400 | 60 |
| Point Translation | ± 0.1 \pm 0.1 m | ± 0.05 \pm 0.05 m |
| Point Rotation | ± 30 ∘ \pm 30^{\circ} (Z-axis) | ± 15 ∘ \pm 15^{\circ} (Z-axis) |
| Jitter Noise ( \sigma) | 0.01 | 0.01 |
| Tactile Noise ( \sigma) | 0.001 | 0.001 |

**说明**: TABLE VI: Contact Field Model Training Hyperparameters

#### Table 6: TABLE VII: Diffusion Policy Hyperparameters

| Parameter | Value |
| --- | --- |
| Policy Configuration | |
| Observation Horizon (T o  b  s T_{obs}) | 3 |
| Action Execution (T a  c  t  i  o  n T_{action}) | 8 |
| Prediction Horizon (T T) | 16 |
| Action Type | Delta End-Effector Pose |
| Optimization | |
| Optimizer | AdamW |
| Learning Rate | 1  e - 4 1e^{-4} |
| Weight Decay | 1  e - 6 1e^{-6} |
| LR Scheduler | Cosine w/ 500 warmup steps |
| Batch Size | 128 |
| Epochs | 1000 |
| EMA | Enabled (Power 0.75) |

**说明**: TABLE VII: Diffusion Policy Hyperparameters

#### Table 7: TABLE VIII: Crayon Picking Experimental Results. We report the Directional Accuracy (Dir. Acc.) and Grasp Success Rate

| | Seen Objects | Unseen Objects | | |
| --- | --- | --- | --- | --- |
| Method | Dir. Acc. | Success | Dir. Acc. | Success |
| Baseline (w/o Semantic) | 63.3% | 33.3% | 40.0% | 23.3% |
| Ours (w/ Semantic) | 93.3% | 76.7% | 93.3% | 73.3% |

**说明**: TABLE VIII: Crayon Picking Experimental Results. We report the Directional Accuracy (Dir. Acc.) and Grasp Success Rate (Success) on seen and unseen objects.

#### Table 8: TABLE IX: Contact-field evaluation with confidence intervals and selected pairwise tests. F1 scores are computed per-fr

| Setting | Model | F1 Score ↑ \uparrow | F1 p p -value | Force MSE ↓ \downarrow | Force MSE p p -value |
| --- | --- | --- | --- | --- | --- |
| Sim | No-Tactile | 0.322 [0.317, 0.327] | < < 0.001 | 0.0146 [0.0141, 0.0151] | 1.000 |
| 2D Tactile Encoder | 0.303 [0.298, 0.308] | < < 0.001 | 0.0147 [0.0142, 0.0152] | 0.011 | |
| BCE Loss | 0.022 [0.021, 0.024] | < < 0.001 | 0.0146 [0.0142, 0.0151] | 1.000 | |
| No Contact Prob. | N/A | N/A | 0.0158 [0.0153, 0.0162] | < < 0.001 | |
| Ours | 0.462 [0.457, 0.467] | – | 0.0147 [0.0142, 0.0151] | – | |
| Real Scraper | Sim-Only | 0.002 [0.001, 0.004] | < < 0.001 | 0.0435 [0.0408, 0.0463] | < < 0.001 |
| Real-Only | 0.445 [0.435, 0.455] | < < 0.001 | 0.0221 [0.0204, 0.0238] | 1.000 | |
| No-Tactile | 0.397 [0.385, 0.408] | < < 0.001 | 0.0432 [0.0406, 0.0458] | < < 0.001 | |
| No Contact Prob. | N/A | N/A | 0.0377 [0.0353, 0.0402] | < < 0.001 | |
| Ours | 0.518 [0.506, 0.529] | – | 0.0254 [0.0236, 0.0273] | – | |
| Real Crayon | Sim-Only | 0.005 [0.002, 0.008] | < < 0.001 | 0.0284 [0.0258, 0.0311] | < < 0.001 |
| Real-Only | 0.485 [0.456, 0.513] | < < 0.001 | 0.0106 [0.0089, 0.0124] | 0.003 | |
| No-Tactile | 0.423 [0.395, 0.451] | < < 0.001 | 0.0115 [0.0089, 0.0143] | 0.004 | |
| No Contact Prob. | N/A | N/A | 0.0089 [0.0070, 0.0111] | 0.308 | |
| Ours | 0.524 [0.496, 0.552] | – | 0.0085 [0.0069, 0.0102] | – | |

**说明**: TABLE IX: Contact-field evaluation with confidence intervals and selected pairwise tests. F1 scores are computed per-frame to enable bootstrap significance testing, distinct from the aggregate scores in Table II and Table II. No Contact Prob. does not output contact probability, so F1 is not applicable. p p -values compare each method against Ours.

#### Table 9: TABLE X: Scraper policy evaluation statistics. Success is computed individual scrape attempts and reported as perc

| Method | Seen Tools | Unseen Tools | | | | | | |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Success [95% CI] | p p | Eff. Norm. [95% CI] | p p | Success [95% CI] | p p | Eff. Norm. [95% CI] | p p | |
| GenDP | 39.1 [26.4, 53.5] | < < 0.001 | 30.5 [13.3, 48.9] | < < 0.001 | 35.1 [24.0, 48.1] | < < 0.001 | 35.1 [21.2, 49.2] | < < 0.001 |
| Raw Tactile | 35.1 [24.0, 48.1] | < < 0.001 | 35.7 [16.0, 56.5] | 0.001 | 50.0 [36.4, 63.6] | 0.002 | 27.3 [10.2, 46.7] | < < 0.001 |
| Sim-Only CF | 34.6 [23.4, 47.8] | < < 0.001 | 33.1 [14.5, 53.6] | < < 0.001 | 55.6 [44.1, 66.5] | 0.004 | 45.2 [26.9, 64.3] | 0.001 |
| Real-Only CF | 45.8 [34.8, 57.3] | 0.001 | 44.5 [23.9, 65.6] | 0.003 | 50.0 [38.1, 61.9] | 0.001 | 54.2 [34.2, 73.3] | 0.009 |
| No Force | 31.3 [21.2, 43.4] | < < 0.001 | 29.9 [12.8, 49.0] | < < 0.001 | 26.0 [15.9, 39.6] | < < 0.001 | 27.6 [11.4, 45.7] | < < 0.001 |
| Ours | 73.5 [62.0, 82.6] | – | 85.2 [67.3, 98.1] | – | 79.6 [67.1, 88.2] | – | 84.7 [70.2, 95.6] | – |

**说明**: TABLE X: Scraper policy evaluation statistics. Success is computed individual scrape attempts and reported as percentage with Wilson score intervals. Normalized cleaning efficiency is also reported as percentage. p p denotes the raw test p p -value comparing each method against Ours.

#### Table 10: TABLE XI: Crayon drawing consistency statistics. p p denotes the raw test p p -value comparing each method against Ours

| Method | Seen Crayons | Unseen Crayons | | | | |
| --- | --- | --- | --- | --- | --- | --- |
| Score | 95% CI | p p | Score | 95% CI | p p | |
| GenDP | 0.81 | [0.71, 0.89] | 0.161 | 0.60 | [0.43, 0.77] | 0.126 |
| Raw Tactile | 0.76 | [0.67, 0.86] | 0.066 | 0.61 | [0.48, 0.74] | 0.016 |
| Sim-Only CF | 0.80 | [0.72, 0.88] | 0.085 | 0.76 | [0.65, 0.85] | 0.313 |
| Real-Only CF | 0.68 | [0.49, 0.85] | 0.087 | 0.74 | [0.56, 0.89] | 0.604 |
| No Force | 0.77 | [0.59, 0.92] | 0.297 | 0.76 | [0.58, 0.91] | 0.724 |
| Ours | 0.86 | [0.77, 0.94] | – | 0.78 | [0.66, 0.89] | – |

**说明**: TABLE XI: Crayon drawing consistency statistics. p p denotes the raw test p p -value comparing each method against Ours.

#### Table 11: TABLE XII: Peeler policy evaluation statistics. Contact and cut-in values are success percentages with Wilson score int

| Split | Method | Contact [95% CI] | p p | Cut-in [95% CI] | p p | Peel Length [95% CI] | p p |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Seen | GenDP | 45.0 [25.8, 65.8] | 0.024 | 30.0 [14.6, 51.9] | 0.013 | 1.50 [0.33, 3.13] | 0.023 |
| Raw Tactile | 45.0 [25.8, 65.8] | 0.024 | 20.0 [8.1, 41.6] | 0.002 | 1.05 [0.03, 2.53] | 0.003 | |
| Sim-Only CF | 60.0 [38.7, 78.1] | 0.150 | 30.0 [14.6, 51.9] | 0.013 | 2.00 [0.60, 3.65] | 0.024 | |
| Real-Only CF | 60.0 [38.7, 78.1] | 0.150 | 15.0 [5.2, 36.0] | < < 0.001 | 0.93 [0.00, 2.38] | 0.003 | |
| No Force | 65.0 [43.3, 81.9] | 0.240 | 30.0 [14.6, 51.9] | 0.013 | 1.95 [0.43, 3.90] | 0.032 | |
| Ours | 80.0 [58.4, 91.9] | – | 70.0 [48.1, 85.5] | – | 4.73 [2.43, 7.10] | – | |
| Unseen | GenDP | 50.0 [33.2, 66.9] | 0.001 | 33.3 [19.2, 51.2] | 0.002 | 1.12 [0.40, 2.02] | 0.001 |
| Raw Tactile | 40.0 [24.6, 57.7] | < < 0.001 | 30.0 [16.7, 47.9] | 0.001 | 0.85 [0.20, 1.78] | < < 0.001 | |
| Sim-Only CF | 50.0 [33.2, 66.9] | 0.001 | 46.7 [30.2, 63.9] | 0.032 | 3.05 [1.67, 4.55] | 0.080 | |
| Real-Only CF | 56.7 [39.2, 72.6] | 0.004 | 26.7 [14.2, 44.5] | < < 0.001 | 1.78 [0.70, 3.07] | 0.003 | |
| No Force | 40.0 [24.6, 57.7] | < < 0.001 | 13.3 [5.3, 29.7] | < < 0.001 | 1.08 [0.23, 2.18] | < < 0.001 | |
| Ours | 90.0 [74.4, 96.5] | – | 73.3 [55.6, 85.8] | – | 4.52 [3.00, 6.05] | – | |

**说明**: TABLE XII: Peeler policy evaluation statistics. Contact and cut-in values are success percentages with Wilson score intervals. Peel length is reported in centimeters. p p denotes the raw test p p -value comparing each method against Ours.
## 实验解读

- 评价重点:围绕 接触推理、接触丰富操作、diffusion-policy、foundation-model、haptic-feedback,优先看真实机器人结果、扰动恢复、接触稳定性、控制误差、成功率、实时性和 sim-to-real 差距。
- System0 视角:实验结论不只看最终成功率,还要看闭环过程中是否能处理状态估计误差、执行延迟、碰撞/接触突变和外部扰动。
- 与上层系统的关系:如果方法能在快闭环内稳定工作,它可以作为 System1 技能或 System2 规划的底层执行保障;如果只在离线或慢速设定下成立,则还不能直接作为身体级快闭环模块。
## 批判性思考

### 优点

- 论文把 接触推理、接触丰富操作、diffusion-policy、foundation-model、haptic-feedback 放到机器人身体级闭环里讨论,和只做高层语义推理的工作相比,更接近真实机器人部署时会遇到的状态、动力学和安全约束。
- 若论文同时给出公式、系统结构图、硬件实验或消融结果,这些证据可以帮助判断方法是否真的能支撑毫秒到百毫秒级反应。

### 局限

- 需要重点核查控制频率、推理延迟、传感延迟、失败恢复、硬件磨损、未见场景泛化和安全边界;这些因素决定它能否成为可靠的 System0 模块。
- 对学习型方法,还要区分训练环境中的鲁棒性和真实部署中的鲁棒性,尤其是接触丰富任务、动态运动任务和人形机器人任务。
## 论文定位

- 研究对象:Semantic Contact Fields for Category-Level Generalizable Tool Manipulation。
- 关键词:接触推理、接触丰富操作、diffusion-policy、foundation-model、haptic-feedback、language-conditioned-policy、实时控制、robot-generalization、机器人操作、鲁棒控制。
- System0 关联:关注机器人身体级快反应链路中“状态/接触/局部感知 -> 约束/优化/策略 -> 动作”的短闭环,不把问题停留在语言规划或任务分解层。

## 速查卡片

> [!summary] Semantic Contact Fields
> - **论文**: https://www.roboticsproceedings.org/rss22/p004.pdf
> - **arXiv**: http://arxiv.org/abs/2602.13833
> - **arXiv HTML**: https://arxiv.org/html/2602.13833
