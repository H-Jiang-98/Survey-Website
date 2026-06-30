---
title: "CordViP: Correspondence-based Visuomotor Policy for Dexterous Manipulation in Real-World"
method_name: "CordViP"
authors: ["Yankai Fu"]
year: 2025
venue: "RSS"
tags: ["contact-reasoning", "robot-manipulation", "robust-control", "reinforcement-learning", "imitation-learning", "robot-generalization", "dexterous-manipulation", "state-estimation"]
image_source: "online"
arxiv_html: "https://arxiv.org/html/2502.08449v2"
---
# CordViP
## 一句话总结

> CordViP: Correspondence-based Visuomotor Policy for Dexterous Manipulation in Real-World 主要落在 [[接触推理]]、[[灵巧操作]]、[[模仿学习]]、[[policy-learning]] 这一类问题上;从 System0 视角看,它的价值在于把身体状态、接触变化、局部感知或动力学约束接到机器人低层闭环里,让动作生成更稳定、安全、实时。
## 核心贡献

1. 围绕 **CordViP: Correspondence-based Visuomotor Policy for Dexterous Manipulation in Real-World** 建立了一个与 接触推理、灵巧操作、模仿学习、policy-learning、proprioception、robot-generalization 相关的机器人问题设定,重点不是高层语义规划,而是身体级闭环中的状态变化、接触变化或控制约束。
2. 方法层面的关键价值在于把感知、动力学、约束优化、策略学习或硬件反馈组织成可执行的低层动作生成机制,适合放进 System0 的快反应模块中理解。
3. 对系统集成而言,这类工作补上了 System1 技能选择与电机级控制之间的一层:它负责把上层目标变成动态可行、可恢复、可安全过滤的动作。
4. 若论文包含真实机器人、接触任务或仿真到真实迁移,核心应看它是否报告了闭环稳定性、扰动恢复、硬件延迟、失败模式和跨场景泛化。
## 问题背景

这篇论文可以放在 System0 的“具身快系统”背景下阅读:机器人需要在很短时间内根据本体状态、接触、力/触觉、局部视觉或动力学预测做出动作,而不是等待慢速规划器重新推理整个任务。接触推理、灵巧操作、模仿学习、policy-learning、proprioception、robot-generalization、机器人操作 相关问题的难点通常不在“知道要做什么”,而在动作执行过程中状态会快速偏离、接触会突然改变、约束会实时收紧,低层系统必须及时修正。

### System0 关联

精读时优先看三件事:第一,论文使用哪些快速反馈量,例如关节状态、接触状态、力觉、触觉、深度或局部视觉;第二,反馈如何进入 MPC、WBC、策略网络、安全过滤器或恢复控制器;第三,输出是否直接影响机器人下一步动作,而不是只服务离线分析或高层任务规划。
## 方法详解

### System0 视角解析

从 System0 角度看,这篇论文的方法部分应当围绕 接触推理、灵巧操作、模仿学习、policy-learning、proprioception、robot-generalization、机器人操作、鲁棒控制 来读。核心问题是:系统如何从快速变化的机器人状态中构造控制输入,如何处理接触、碰撞、约束、模型误差或执行延迟,以及如何把这些信息变成下一时刻的动作。

如果方法里有优化问题,重点看目标函数、约束和求解频率;如果方法里有学习策略,重点看观测空间、动作空间、训练扰动和部署时的闭环频率;如果方法里有硬件或传感器,重点看传感链路是否真的进入控制回路。这样读可以把论文和 System2 语言规划、System1 技能编排区分开来。

### 关键公式与机制

#### 公式 1: [[优化目标/约束]]

$$\mathbf{P}_{hand}\in\mathbb{R}^{N_{\mathcal{P}}\times 3}$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 2: [[优化目标/约束]]

$$\mathbf{P}_{hand}=\mathcal{F}_{pc}(q,\{P_{\ell_{i}}\}_{i=1}^{N_{\ell}}),$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 3: [[优化目标/约束]]

$$\mathcal{D_{(O,H)}}=\min_{v_{h}\in\mathbf{P}_{hand}}e^{\gamma(1-| |)}\lVert v_{o}-v_{h} \rVert_{2},$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 4: [[优化目标/约束]]

$$\mathcal{C}=1-2(\text{Sigmod}(\theta\cdot\mathcal{{D_{(O,H)}}})-0.5)$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 5: [[动力学或策略机制]]

$$f_{\theta_{O}}(\mathbf{P}_{obj})$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 6: [[动力学或策略机制]]

$$f_{\theta_{H}}(\mathbf{P}_{hand})$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 7: [[动力学或策略机制]]

$$\begin{split}\phi^{H}&=g_{\theta_{H}}(f_{\theta_{H}}(\mathbf{P}_{hand}),f_{\theta_{O}}(\mathbf{P}_{obj}))+f_{\theta_{H}}(\mathbf{P}_{hand})\\ \phi^{O}&=g_{\theta_{O}}(f_{\theta_{O}}(\mathbf{P}_{obj}),f_{\theta_{H}}(\mathbf{P}_{hand}))+f_{\theta_{O}}(\mathbf{P}_{obj})\end{split}$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 8: [[优化目标/约束]]

$$\ {\mathcal{E}}{\min}\mathcal{L}=\mathcal{L}_{contact}+\lambda\mathcal{L}_{coordination},$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 9: [[优化目标/约束]]

$$\mathcal{L}=MSE(\varepsilon^{k},\varepsilon_{\theta}(\bar{\alpha_{k}}A^{0}+\bar{\beta}\varepsilon^{k},\phi^{H,O},\psi^{A,H},k))$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 10: [[优化目标/约束]]

$$A^{k-1}=\alpha_{k}(A^{k}-\gamma_{k}\varepsilon_{\theta}(\phi^{H,O},\psi^{A,H},A^{k},k))+\sigma_{k}\mathcal{N}(0,I),$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。
## 关键图表

### Figure 1: Overview Framework (a) We first employ TripoSR to generate the initial object point c

![Figure 1](https://arxiv.org/html/2502.08449v2/x2.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Overview Framework (a) We first employ TripoSR to generate the initial object point c”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 2: Real robot system. Our system consists of a Leap Hand and a UR5 Arm, with a fixed Rea

![Figure 2](https://arxiv.org/html/2502.08449v2/x4.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Real robot system. Our system consists of a Leap Hand and a UR5 Arm, with a fixed Rea”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 3: Experimental results of efficiency. We train ACT, DP, DP3, and CordViP on the PickPla

![Figure 3](https://arxiv.org/html/2502.08449v2/x7.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Experimental results of efficiency. We train ACT, DP, DP3, and CordViP on the PickPla”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### 论文表格

#### Table 1: TABLE I: Main results of four base real-world tasks. Each experiment is evaluated with 20 trials.

| Method | PickPlace | FlipCup | Assembly | ArtiManip | Avg |
| --- | --- | --- | --- | --- | --- |
| State-based MLP | 0% | 0% | 0% | 0% | 0% |
| BCRNN | 0% | 0% | 0% | 5% | 1% |
| BCRNN+3D | 0% | 5% | 0% | 10% | 4% |
| ACT | 45% | 70% | 25% | 65% | 51% |
| ACT+3D | 70% | 80% | 35% | 70% | 64% |
| DP | 55% | 65% | 20% | 35% | 44% |
| DP3 | 30% | 20% | 0% | 40% | 23% |
| State-based DP | 40% | 35% | 40% | 40% | 39% |
| G3Flow | 65% | 65% | 80% | 85% | 74% |
| CordViP(Ours) | 85% | 90% | 90% | 95% | 90% |

**说明**: TABLE I: Main results of four base real-world tasks. Each experiment is evaluated with 20 trials.

#### Table 2: TABLE II: Main results of advanced tasks. Each experiment is evaluated with 10 trials.

| Method | FlipCap | LongHoriManip | | | |
| --- | --- | --- | --- | --- | --- |
| Pull | Pick | Place | Push | | |
| ACT | 30% | 100% | 40% | 40% | 30% |
| ACT+3D | 40% | 100% | 40% | 20% | 20% |
| DP | 30% | 70% | 30% | 10% | 10% |
| State-based DP | 20% | 80% | 30% | 10% | 0% |
| G3Flow | 60% | 100% | 80% | 50% | 30% |
| CordViP(Ours) | 80% | 100% | 100% | 70% | 60% |

**说明**: TABLE II: Main results of advanced tasks. Each experiment is evaluated with 10 trials.

#### Table 3: TABLE III: Generalization results on different lighting conditions. We evaluate the policy three lighting scenari

**说明**: TABLE III: Generalization results on different lighting conditions. We evaluate the policy three lighting scenarios: dim light(dim), white light(white) and colored lighting(colored).

#### Table 4: TABLE IV: Generalization results in diverse scenarios, including varying visual appearances and challenging cluttered

**说明**: TABLE IV: Generalization results in diverse scenarios, including varying visual appearances and challenging cluttered environments.

#### Table 5: TABLE V: Generalization results on unseen objects. For PickPlace and FlipCup tasks, we chose three previously unseen ob

**说明**: TABLE V: Generalization results on unseen objects. For PickPlace and FlipCup tasks, we chose three previously unseen objects, varying in color, shape, and dynamics.

#### Table 6: TABLE VI: Generalization results to different viewpoints.

**说明**: TABLE VI: Generalization results to different viewpoints.

#### Table 7: TABLE VII: Ablation experiments on the Effectiveness of different components. DP3 + Interaction-aware PC refers to usin

| Ablation | PickPlace | FlipCup | Assembly | ArtiManip | Avg |
| --- | --- | --- | --- | --- | --- |
| DP3 + Interaction-aware PC | 65% | 75% | 60% | 85% | 71% |
| CordViP w/o. contact and coordination pretrain | 75% | 80% | 75% | 85% | 79% |
| CordViP w/o. contact pretrain | 85% | 75% | 85% | 90% | 84% |
| CordViP w/o. coordination pretrain | 80% | 85% | 80% | 85% | 83% |
| CordViP(Ours) | 85% | 90% | 90% | 95% | 90% |

**说明**: TABLE VII: Ablation experiments on the Effectiveness of different components. DP3 + Interaction-aware PC refers to using the Interaction-aware point clouds as the visual input for DP3. W/o. contact and coordination pretrain means that the encoder was not pre-trained with contact and coordination data.

#### Table 8: TABLE VIII: Ablation experiments on point cloud encoders. * indicates that the encoder is frozen during the training ph

| Encoders | PickPlace | FlipCup | Assembly | ArtiManip | Avg |
| --- | --- | --- | --- | --- | --- |
| PointNet | 85% | 90% | 90% | 95% | 90% |
| PointNet++ | 5% | 30% | 30% | 40% | 26% |
| PointNeXt | 0% | 0% | 0% | 0% | 0% |
| PointNet* | 70% | 90% | 75% | 80% | 79% |
| PointNet++* | 5% | 15% | 5% | 30% | 14% |

**说明**: TABLE VIII: Ablation experiments on point cloud encoders. * indicates that the encoder is frozen during the training phase of the correspondence-based diffusion policy.

#### Table 9: TABLE IX: Transferability to different backbone.

| Method | PickPlace | FlipCup | Assembly | ArtiManip |
| --- | --- | --- | --- | --- |
| Diffusion-based | 85% | 90% | 90% | 95% |
| Transformer-based | 90% | 75% | 90% | 80% |

**说明**: TABLE IX: Transferability to different backbone.

#### Table 10: TABLE X: Parameters of expert demonstrations for real-world tasks. “Demo" refers to the number of demonstrations, “Epis

| Task Name | Demos | Episode Length | Teleop. Times(s) | Max steps |
| --- | --- | --- | --- | --- |
| PickPlace | 50 | 150 | 30 | 400 |
| Flipcup | 50 | 150 | 30 | 300 |
| Assembly | 50 | 175 | 35 | 500 |
| ArtiManip | 50 | 190 | 38 | 600 |
| Flipcap | 50 | 150 | 30 | 400 |
| LongHoriManip | 50 | 250 | 50 | 800 |

**说明**: TABLE X: Parameters of expert demonstrations for real-world tasks. “Demo" refers to the number of demonstrations, “Episode Length" denotes the duration of each episode in a task, “Teleop. Times" indicates the teleoperation time per demonstration, and “Max Steps" represents the maximum execution time for a task during evaluation.

#### Table 11: TABLE XI: Training hyperparameters in CordViP.

| Hyperparameters | Value |
| --- | --- |
| Robot point cloud size | 1024*3 |
| Object point cloud size | 1024*3 |
| Contact map scaling factor $\gamma$ | 1 |
| Contact map scaling factor $\theta$ | 10 |
| Contact map size | 1024*1 |
| Loss weight   \lambda | 1 |
| h o r i z o n h o r i z o n horizon | 12 |
| n _ o b s _ s t e p s n _ o b s _ s t e p s n\_obs\_steps _ _ | 4 |
| n _ a c t i o n _ s t e p s n _ a c t i o n _ s t e p s n\_action\_steps _ _ | 6 |
| Optimizer | AdamW |

**说明**: TABLE XI: Training hyperparameters in CordViP.
## 实验解读

- 评价重点:围绕 接触推理、灵巧操作、模仿学习、policy-learning、proprioception,优先看真实机器人结果、扰动恢复、接触稳定性、控制误差、成功率、实时性和 sim-to-real 差距。
- System0 视角:实验结论不只看最终成功率,还要看闭环过程中是否能处理状态估计误差、执行延迟、碰撞/接触突变和外部扰动。
- 与上层系统的关系:如果方法能在快闭环内稳定工作,它可以作为 System1 技能或 System2 规划的底层执行保障;如果只在离线或慢速设定下成立,则还不能直接作为身体级快闭环模块。
## 批判性思考

### 优点

- 论文把 接触推理、灵巧操作、模仿学习、policy-learning、proprioception 放到机器人身体级闭环里讨论,和只做高层语义推理的工作相比,更接近真实机器人部署时会遇到的状态、动力学和安全约束。
- 若论文同时给出公式、系统结构图、硬件实验或消融结果,这些证据可以帮助判断方法是否真的能支撑毫秒到百毫秒级反应。

### 局限

- 需要重点核查控制频率、推理延迟、传感延迟、失败恢复、硬件磨损、未见场景泛化和安全边界;这些因素决定它能否成为可靠的 System0 模块。
- 对学习型方法,还要区分训练环境中的鲁棒性和真实部署中的鲁棒性,尤其是接触丰富任务、动态运动任务和人形机器人任务。
## 论文定位

- 研究对象:CordViP: Correspondence-based Visuomotor Policy for Dexterous Manipulation in Real-World。
- 关键词:接触推理、灵巧操作、模仿学习、policy-learning、proprioception、robot-generalization、机器人操作、鲁棒控制、状态估计、visuomotor-policy。
- System0 关联:关注机器人身体级快反应链路中“状态/接触/局部感知 -> 约束/优化/策略 -> 动作”的短闭环,不把问题停留在语言规划或任务分解层。

## 速查卡片

> [!summary] CordViP
> - **论文**: https://www.roboticsproceedings.org/rss21/p110.pdf
> - **arXiv**: http://arxiv.org/abs/2502.08449v2
> - **arXiv HTML**: https://arxiv.org/html/2502.08449v2
