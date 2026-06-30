---
title: "Real-Time Anomaly Detection and Reactive Planning with Large Language Models"
method_name: "Time Anomaly Detection"
authors: ["Rohan Sinha"]
year: 2024
venue: "RSS"
tags: ["real-time-control", "safe-control", "robot-generalization", "closed-loop-control", "model-predictive-control", "recovery"]
image_source: "online"
arxiv_html: "https://arxiv.org/html/2407.08735v1"
---
# Time Anomaly Detection
## 一句话总结

> Real-Time Anomaly Detection and Reactive Planning with Large Language Models 主要落在 [[closed-loop-control]]、[[foundation-model]]、[[inference-time-algorithm]]、[[language-conditioned-policy]] 这一类问题上;从 System0 视角看,它的价值在于把身体状态、接触变化、局部感知或动力学约束接到机器人低层闭环里,让动作生成更稳定、安全、实时。
## 核心贡献

1. 围绕 **Real-Time Anomaly Detection and Reactive Planning with Large Language Models** 建立了一个与 closed-loop-control、foundation-model、inference-time-algorithm、language-conditioned-policy、模型预测控制、reactive-control 相关的机器人问题设定,重点不是高层语义规划,而是身体级闭环中的状态变化、接触变化或控制约束。
2. 方法层面的关键价值在于把感知、动力学、约束优化、策略学习或硬件反馈组织成可执行的低层动作生成机制,适合放进 System0 的快反应模块中理解。
3. 对系统集成而言,这类工作补上了 System1 技能选择与电机级控制之间的一层:它负责把上层目标变成动态可行、可恢复、可安全过滤的动作。
4. 若论文包含真实机器人、接触任务或仿真到真实迁移,核心应看它是否报告了闭环稳定性、扰动恢复、硬件延迟、失败模式和跨场景泛化。
## 问题背景

这篇论文可以放在 System0 的“具身快系统”背景下阅读:机器人需要在很短时间内根据本体状态、接触、力/触觉、局部视觉或动力学预测做出动作,而不是等待慢速规划器重新推理整个任务。closed-loop-control、foundation-model、inference-time-algorithm、language-conditioned-policy、模型预测控制、reactive-control、实时控制 相关问题的难点通常不在“知道要做什么”,而在动作执行过程中状态会快速偏离、接触会突然改变、约束会实时收紧,低层系统必须及时修正。

### System0 关联

精读时优先看三件事:第一,论文使用哪些快速反馈量,例如关节状态、接触状态、力觉、触觉、深度或局部视觉;第二,反馈如何进入 MPC、WBC、策略网络、安全过滤器或恢复控制器;第三,输出是否直接影响机器人下一步动作,而不是只服务离线分析或高层任务规划。
## 方法详解

### System0 视角解析

从 System0 角度看,这篇论文的方法部分应当围绕 closed-loop-control、foundation-model、inference-time-algorithm、language-conditioned-policy、模型预测控制、reactive-control、实时控制、recovery 来读。核心问题是:系统如何从快速变化的机器人状态中构造控制输入,如何处理接触、碰撞、约束、模型误差或执行延迟,以及如何把这些信息变成下一时刻的动作。

如果方法里有优化问题,重点看目标函数、约束和求解频率;如果方法里有学习策略,重点看观测空间、动作空间、训练扰动和部署时的闭环频率;如果方法里有硬件或传感器,重点看传感链路是否真的进入控制回路。这样读可以把论文和 System2 语言规划、System1 技能编排区分开来。

### 关键公式与机制

#### 公式 1: [[优化目标/约束]]

$$\mathcal{D}_{\mathrm{nom}}=\{\boldsymbol{o}_{i}\}_{i=1}^{N}$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 2: [[优化目标/约束]]

$$\boldsymbol{w}(\boldsymbol{o}_{t_{\mathrm{anom}}},\mathcal{Y}_{t_{\mathrm{anom}}})=0$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 3: [[优化目标/约束]]

$$y=\boldsymbol{w}(\boldsymbol{o}_{t_{\mathrm{anom}}},\mathcal{Y}_{t_{\mathrm{anom}}})$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 4: [[动力学或策略机制]]

$$\boldsymbol{h}(o)=\begin{cases}\mathrm{anomaly}&\text{if $\boldsymbol{s}(o)>\tau$}\\ \mathrm{nominal}&\text{if $\boldsymbol{s}(o)\leq\tau$}\end{cases}.$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 5: [[优化目标/约束]]

$$\mathcal{Y}_{t_{\mathrm{anom}}+K^{\prime}}=\mathcal{Y}_{t_{\mathrm{anom}}}$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 6: [[优化目标/约束]]

$$\tau=\inf\bigg{\{}q\in\mathbb{R}\:\ \frac{| { $\boldsymbol{e}_{i}\in\mathcal{D}_{e}:\boldsymbol{s}(\boldsymbol{e}_{i};\mathcal{D}_{e}\setminus\{\boldsymbol{e}_{i}\})\leq q}$ |}{N}\geq\alpha\bigg{\}},$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 7: [[优化目标/约束]]

$$\begin{aligned} \operatorname*{\mathrm{minimize}}_{\{\boldsymbol{x}_{t:t+T+1| t}^{i}, $\boldsymbol{u}_{t:t+T$ | t}^{i}\}_{i\in $\mathcal{Y}\cup\{0\}}}&\enspace C(\boldsymbol{x}_{t:t+T+1$ | t}^{0}, $\boldsymbol{u}_{t:t+T$ | t}^{0})\\  $\operatorname*{\mathrm{s.t.}}\enspace&\boldsymbol{x}_{t+k+1$ | t}^{i}= $\boldsymbol{f}(\boldsymbol{x}_{t+k$ | t}^{i}, $\boldsymbol{u}_{t+k$ | t}^{i})\\ & $\boldsymbol{u}_{t+k$ | t}^{i}\in $\mathcal{U}\quad\boldsymbol{x}_{t+k$ | t}^{i}\in $\mathcal{X}\\ &\boldsymbol{x}_{t$ | t}^{i}= $\boldsymbol{x}_{t}\\ &\boldsymbol{x}_{t+T+1$ | t}^{i}\in $\mathcal{X}_{R}^{i}\quad\forall i\in\mathcal{Y}\\ &\boldsymbol{u}_{t$ | t}^{i}= $\boldsymbol{u}_{t$ | t}^{0}\quad\forall i\in $\mathcal{Y}\\ &\boldsymbol{u}_{t:t+K$ | t}^{i}= $\boldsymbol{u}_{t:t+K$ |t}^{j}\ \forall i,j\in\mathcal{Y}\end{aligned}$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 8: [[优化目标/约束]]

$$\boldsymbol{s}(\boldsymbol{e}_{t};\mathcal{D}_{e}):=-\max_{\boldsymbol{e}_{i}\in\mathcal{D}_{e}}{\frac{\boldsymbol{e}_{i}^{T}\boldsymbol{e}_{t}}{\lVert \boldsymbol{e}_{i} \rVert\lVert \boldsymbol{e}_{t} \rVert}},$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 9: [[动力学或策略机制]]

$$\boldsymbol{x}_{t+1:t+T+2| t+2}^{i}:=[ $\boldsymbol{x}_{t+1};\boldsymbol{x}_{t+2:t+T+1$ | t}^{i,\star}; $\boldsymbol{f}(\boldsymbol{x}_{t+T+1$ | t}^{i,\star}, $\boldsymbol{u}_{t+T+1$ |t+1}^{i})]$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 10: [[优化目标/约束]]

$$J_{t}(\{\boldsymbol{w}(\boldsymbol{o}_{t_{\mathrm{anom}}},\mathcal{Y}_{t_{\mathrm{anom}}})\},0,\max\{0,T-k\})$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。
## 关键图表

### Figure 1: Pipeline / core system figure: We present an embedding-based runtime monitoring scheme using fast and slow language

![Figure 1](https://arxiv.org/html/2407.08735v1/x1.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Pipeline / core system figure: We present an embedding-based runtime monitoring scheme using fast and slow language”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 2: Closed-loop trajectory of a quadrotor using the AESOP algorithm. The figure represent

![Figure 2](https://arxiv.org/html/2407.08735v1/x3.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Closed-loop trajectory of a quadrotor using the AESOP algorithm. The figure represent”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 3: d) Closed loop trajectory of the AESOP algorithm (black). In this trajectory, the fast anomaly

![Figure 3](https://arxiv.org/html/2407.08735v1/extracted/5725866/figures/quadrotors/supplement/aesop_abort.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“d) Closed loop trajectory of the AESOP algorithm (black). In this trajectory, the fast anomaly”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### 论文表格

#### Table 1: TABLE I: Slow Generative Reasoning for Anomaly Assessment in VTOL. Best scores are bolded; second best are.

| Method | TPR | FPR | Accuracy |
| --- | --- | --- | --- |
| Llama 2 (7B) | 0.52 | 0.46 | 0.52 |
| GPT-3.5 Turbo | 0.97 | 0.54 | 0.73 |
| GPT-3.5 Turbo CoT | 0.82 | 0.28 | 0.77 |
| GPT-4 | 0.65 | 0.06 | 0.79 |
| GPT-4 CoT | 0.89 | 0.10 | 0.90 |

**说明**: TABLE I: Slow Generative Reasoning for Anomaly Assessment in VTOL. Best scores are bolded; second best are.

#### Table 2: TABLE II: Percentage of trajectories where the quadrotor successfully recovered to the LLM’s choice of recovery region.

| | Naive MPC | FS-MPC [46 ] | AESOP |
| --- | --- | --- | --- |
| Successful Recovery Rate | 15% | 23% | 100% |

**说明**: TABLE II: Percentage of trajectories where the quadrotor successfully recovered to the LLM’s choice of recovery region.

#### Table 3: TABLE III: Inference times for the OWL-ViT object detector, LLM embedding models (MPNet, Mistral), and cloud-queryin

| Component | Mean (s) | Standard Deviation (s) |
| --- | --- | --- |
| MPC solve of Section IV-B | 0.023 | 0.019 |
| OWL-ViT | 0.025 | 0.002 |
| MPNet | 0.028 | 0.005 |
| Mistral | 0.32 | 0.08 |
| GPT-3-Turbo CoT | 3.10 | 0.85 |
| GPT-4 CoT | 18.88 | 3.923 |

**说明**: TABLE III: Inference times for the OWL-ViT object detector, LLM embedding models (MPNet, Mistral), and cloud-querying GPT-3/4 on a Jetson AGX Orin module.

#### Table 4: TABLE IV: CARLA Evaluation. Text and Vision-based Anomaly Detection.

| | Method | TPR | FPR | Bal. Accuracy |
| --- | --- | --- | --- | --- |
| Text | GPT-4 | 0.74 | 0.19 | 0.78 |
| GPT-3 CoT [12 ] | 0.89 | 0.26 | 0.82 | |
| MPNet (Ours) | 0.69 | 0.05 | 0.82 | |
| Mistral (Ours) | 0.95 | 0.05 | 0.95 | |
| Vision | SCOD | 0.40 | 0.06 | 0.67 |
| Mahal. | 0.40 | 0.13 | 0.64 | |
| GPT-4V | 0.97 | 0.27 | 0.85 | |
| GPT-4V CoT | 0.89 | 0.10 | 0.90 | |
| CLIP (Ours) | 0.86 | 0.05 | 0.90 | |
| CLIP (Ours) Abl. | 0.99 | 0.57 | 0.71 | |

**说明**: TABLE IV: CARLA Evaluation. Text and Vision-based Anomaly Detection.

#### Table 5: TABLE V: Glossary of notation and symbols used in this paper.

| | Symbol | Description |
| --- | --- | --- |
| | x x x | unless explicitly defined otherwise, scalar variables are lowercase |
| | x x  $\mathbf{x}$ | vectors are boldfaced |
| | X X  $\mathcal{X}$ | sets are caligraphic |
| | x t x t x_{t} | time-varying quantities are indexed with a t ∈ N ≥ 0 t N absent 0 t\in $\mathbb{N}_{\geq 0} ∈ ≥ 0$ |
| Notation and conventions | x 0: t x: 0 t  $\boldsymbol{x}_{0:t} 0:$ | Shorthand to index subsequences: x 0: t:= {x 0, ..., x t } assign x: 0 t x 0 ... x t  $\boldsymbol{x}_{0:t}:=\{\boldsymbol{x}_{0},\dots,\boldsymbol{x}_{t}\} 0::= {0, ..., }$ |
| | $\delta,\epsilon,\theta,,,$ | hyperparameters (regardless of their type) are lowercase Greek characters |
| | x t + k \| t x t conditional k t  $\boldsymbol{x}_{t+k\lVert t} + \rVert$ | Predicted quantities at k k k time steps into the future computed at time step t t t. Read x t + k \| t x t conditional k t  $\boldsymbol{x}_{t+k\lVert t} + \rVert as “the predicted value of x x \boldsymbol{x} at time t + k t k t+k + given time t t t.”$ |
| | x x  $\boldsymbol{x}$ | System state |
| | u u  $\boldsymbol{u}$ | Input |
| | o o  $\boldsymbol{o}$ | Observation |
| | f f  $\boldsymbol{f}$ | Dynamics |
| | h h  $\boldsymbol{h}$ | Anomaly Detector |
| | w w  $\boldsymbol{w}$ | Generative reasoner |
| | e e  $\boldsymbol{e}$ | Embedding vector |
| |  \phi | Embedding model |
| | X X  $\mathcal{X}$ | State constraint set |
| | U U  $\mathcal{U}$ | Input constraint set |
| | O O  $\mathcal{O}$ | Observation space |
| | $\tau$ | Anomaly detection threshold |
| | s s  $\boldsymbol{s}$ | Anomaly score function |
| | C C C | control objective function for the MPC Section IV-B |
| | D nom D nom  $\mathcal{D}_{\mathrm{nom}}$ | Dataset of nominal observations |
| | N N N | Number of nominal observations in D nom D nom  $\mathcal{D}_{\mathrm{nom}}$ |
| Variables | X R i X R i  $\mathcal{X}_{R}^{i}$ | the i i i ’th recovery region |
| | d d d | Number of recovery regions |
| | D e D e  $\mathcal{D}_{e}$ | Embedding vector cache constructed from D nom D nom  $\mathcal{D}_{\mathrm{nom}}$ |
| |   \alpha | Quantile hyperparameter to select the anomaly detector threshold |
| | q q q | Optimization variable used to define the empirical   \alpha -quantile |
| | Y Y  $\mathcal{Y}$ | Subset of {1, ..., d } 1 ... d \{1, $\dots,d\} {1, ..., } indicating a selection of recovery regions$ |
| | K K K | Upper bound on the latency of the slow reasoner |
| | T T T | Time horizon of the MPC Section IV-B |
| | J J J | Objective value associated with the solution of the MPC problem Section IV-B |
| | x ⋆, u ⋆ x ⋆ u ⋆  $\boldsymbol{x}^{\star},\ \boldsymbol{u}^{\star} ⋆, ⋆$ | Starred quantities denote the optimal values of the decision variables in the MPC problem Section IV-B |
| | t anom t anom t_{ $\mathrm{anom}}$ | Time step at which the anomaly detector triggers |

**说明**: TABLE V: Glossary of notation and symbols used in this paper.

#### Table 6: TABLE VI: Calibration results for a selection of embedding models in the VTOL domain. The table reports the mean anomal

| | Embedding | | Generative | | | | |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Threshold: | 75-Quantile | 85-Quantile | 90-Quantile | 95-Quantile | | GPT-4 | GPT-4 CoT |
| OpenAI Ada 002 | 0.8 (0.002) | 0.84 (0.002) | 0.85 (0.002) | 0.85 (0.002) | | 0.75 | 0.86 |
| MPNet | 0.83 (0.002) | 0.89 (0.002) | 0.91 (0.002) | 0.93 (0.001) | | | |
| Mistral (7B) | 0.83 (0.001) | 0.89 (0.001) | 0.91 (0.001) | 0.94 (0.001) | | | |

**说明**: TABLE VI: Calibration results for a selection of embedding models in the VTOL domain. The table reports the mean anomaly detection accuracy thresholding the top-5 score function at different quantile thresholds. Standard deviations are provided in parentheses. Statistics were computed multiple samplings of 80% of the available nominal data samples. Accuracies for GPT-4 single-token and CoT queries are provided for comparison.

#### Table 7: TABLE VII: Slow Generative Reasoning for Anomaly Assessment in the Warehouse Manipulation Domain.

| Domain | Method | TPR | FPR | Accuracy |
| --- | --- | --- | --- | --- |
| Manip. | GPT-3.5 Turbo | 1.0 | 0.73 | 0.64 |
| GPT-3.5 Turbo CoT | 1.0 | 0.52 | 0.74 | |
| GPT-4 | 1.0 | 0.0 | 1.0 | |
| GPT-4 CoT | 1.0 | 0.0 | 1.0 | |

**说明**: TABLE VII: Slow Generative Reasoning for Anomaly Assessment in the Warehouse Manipulation Domain.

#### Table 8: TABLE VIII: Percentage of trajectories where the quadrotor successfully recovered to the LLM’s choice of recovery regio

| | Naive MPC | FS-MPC [46 ] | AESOP |
| --- | --- | --- | --- |
| Successful Recovery Rate | 15% | 23% | 100% |

**说明**: TABLE VIII: Percentage of trajectories where the quadrotor successfully recovered to the LLM’s choice of recovery region.

#### Table 9: TABLE IX: Accuracy of embedding detectors when withholding nominal data from CARLA routes with anomalies. All methods a

| Method | TPR | FPR | Bal. Accuracy |
| --- | --- | --- | --- |
| (Lang.) MPNet Abl. | 0.55 | 0.11 | 0.72 |
| (Lang.) Mistral Abl. | 0.96 | 0.19 | 0.89 |
| (Vision) CLIP Abl. | 0.99 | 0.57 | 0.71 |

**说明**: TABLE IX: Accuracy of embedding detectors when withholding nominal data from CARLA routes with anomalies. All methods are our own.
## 实验解读

- 评价重点:围绕 closed-loop-control、foundation-model、inference-time-algorithm、language-conditioned-policy、模型预测控制,优先看真实机器人结果、扰动恢复、接触稳定性、控制误差、成功率、实时性和 sim-to-real 差距。
- System0 视角:实验结论不只看最终成功率,还要看闭环过程中是否能处理状态估计误差、执行延迟、碰撞/接触突变和外部扰动。
- 与上层系统的关系:如果方法能在快闭环内稳定工作,它可以作为 System1 技能或 System2 规划的底层执行保障;如果只在离线或慢速设定下成立,则还不能直接作为身体级快闭环模块。
## 批判性思考

### 优点

- 论文把 closed-loop-control、foundation-model、inference-time-algorithm、language-conditioned-policy、模型预测控制 放到机器人身体级闭环里讨论,和只做高层语义推理的工作相比,更接近真实机器人部署时会遇到的状态、动力学和安全约束。
- 若论文同时给出公式、系统结构图、硬件实验或消融结果,这些证据可以帮助判断方法是否真的能支撑毫秒到百毫秒级反应。

### 局限

- 需要重点核查控制频率、推理延迟、传感延迟、失败恢复、硬件磨损、未见场景泛化和安全边界;这些因素决定它能否成为可靠的 System0 模块。
- 对学习型方法,还要区分训练环境中的鲁棒性和真实部署中的鲁棒性,尤其是接触丰富任务、动态运动任务和人形机器人任务。
## 论文定位

- 研究对象:Real-Time Anomaly Detection and Reactive Planning with Large Language Models。
- 关键词:closed-loop-control、foundation-model、inference-time-algorithm、language-conditioned-policy、模型预测控制、reactive-control、实时控制、recovery、robot-generalization、safe-control。
- System0 关联:关注机器人身体级快反应链路中“状态/接触/局部感知 -> 约束/优化/策略 -> 动作”的短闭环,不把问题停留在语言规划或任务分解层。

## 速查卡片

> [!summary] Time Anomaly Detection
> - **论文**: https://www.roboticsproceedings.org/rss20/p114.pdf
> - **arXiv**: http://arxiv.org/abs/2407.08735v1
> - **arXiv HTML**: https://arxiv.org/html/2407.08735v1
