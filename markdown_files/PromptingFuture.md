---
title: "Prompting with the Future: Open-World Model Predictive Control with Interactive Digital Twins"
method_name: "Prompting with the Future"
authors: ["Chuanruo Ning"]
year: 2025
venue: "RSS"
tags: ["contact-reasoning", "robot-manipulation", "robust-control", "adaptive-control", "contact-rich-manipulation", "robot-generalization", "dexterous-manipulation", "model-predictive-control"]
image_source: "online"
arxiv_html: "https://arxiv.org/html/2506.13761v1"
---
# Prompting with the Future
## 一句话总结

> Prompting with the Future: Open-World Model Predictive Control with Interactive Digital Twins 主要落在 [[adaptive-control]]、[[接触推理]]、[[接触丰富操作]]、[[in-hand-manipulation]] 这一类问题上;从 System0 视角看,它的价值在于把身体状态、接触变化、局部感知或动力学约束接到机器人低层闭环里,让动作生成更稳定、安全、实时。
## 核心贡献

1. 围绕 **Prompting with the Future: Open-World Model Predictive Control with Interactive Digital Twins** 建立了一个与 adaptive-control、接触推理、接触丰富操作、in-hand-manipulation、language-conditioned-policy、模型预测控制 相关的机器人问题设定,重点不是高层语义规划,而是身体级闭环中的状态变化、接触变化或控制约束。
2. 方法层面的关键价值在于把感知、动力学、约束优化、策略学习或硬件反馈组织成可执行的低层动作生成机制,适合放进 System0 的快反应模块中理解。
3. 对系统集成而言,这类工作补上了 System1 技能选择与电机级控制之间的一层:它负责把上层目标变成动态可行、可恢复、可安全过滤的动作。
4. 若论文包含真实机器人、接触任务或仿真到真实迁移,核心应看它是否报告了闭环稳定性、扰动恢复、硬件延迟、失败模式和跨场景泛化。
## 问题背景

这篇论文可以放在 System0 的“具身快系统”背景下阅读:机器人需要在很短时间内根据本体状态、接触、力/触觉、局部视觉或动力学预测做出动作,而不是等待慢速规划器重新推理整个任务。adaptive-control、接触推理、接触丰富操作、in-hand-manipulation、language-conditioned-policy、模型预测控制、robot-generalization 相关问题的难点通常不在“知道要做什么”,而在动作执行过程中状态会快速偏离、接触会突然改变、约束会实时收紧,低层系统必须及时修正。

### System0 关联

精读时优先看三件事:第一,论文使用哪些快速反馈量,例如关节状态、接触状态、力觉、触觉、深度或局部视觉;第二,反馈如何进入 MPC、WBC、策略网络、安全过滤器或恢复控制器;第三,输出是否直接影响机器人下一步动作,而不是只服务离线分析或高层任务规划。
## 方法详解

### System0 视角解析

从 System0 角度看,这篇论文的方法部分应当围绕 adaptive-control、接触推理、接触丰富操作、in-hand-manipulation、language-conditioned-policy、模型预测控制、robot-generalization、机器人操作 来读。核心问题是:系统如何从快速变化的机器人状态中构造控制输入,如何处理接触、碰撞、约束、模型误差或执行延迟,以及如何把这些信息变成下一时刻的动作。

如果方法里有优化问题,重点看目标函数、约束和求解频率;如果方法里有学习策略,重点看观测空间、动作空间、训练扰动和部署时的闭环频率;如果方法里有硬件或传感器,重点看传感链路是否真的进入控制回路。这样读可以把论文和 System2 语言规划、System1 技能编排区分开来。

### 关键公式与机制

#### 公式 1: [[优化目标/约束]]

$$I_{g}=\text{Render}(G(\mathcal{T}),C)$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 2: [[动力学或策略机制]]

$$I_{r}=\text{Render}(S(U,a))$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 3: [[动力学或策略机制]]

$$I={I_{i}\in\mathbb{R}^{W\times H\times 3}}$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 4: [[动力学或策略机制]]

$$\tau={\tau_{1},\tau_{2},\dots}$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。
## 关键图表

### Figure 1: Pipeline / core system figure: Model Predictive Control through Simulation-Informed Prompting. Given a free-form inst

![Figure 1](https://arxiv.org/html/2506.13761v1/x3.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Pipeline / core system figure: Model Predictive Control through Simulation-Informed Prompting. Given a free-form inst”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 2: Example trajectories. Example trajectories planned by our framework in both digital tw

![Figure 2](https://arxiv.org/html/2506.13761v1/x5.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Example trajectories. Example trajectories planned by our framework in both digital tw”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 3: Example trajectories. Example trajectories planned by our framework in both the digita

![Figure 3](https://arxiv.org/html/2506.13761v1/x8.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Example trajectories. Example trajectories planned by our framework in both the digita”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### 论文表格

#### Table 1: TABLE I: Definition of tasks. We list the text instructions for prompting VLM and the success criteria of each task.

| Tasks | Instruction | Success Criteria |
| --- | --- | --- |
| Water plant | Water the plant with the cup | The cup is the plant pot with tilting action |
| Play drum | Hit the drum with the drum stick | The head of the drum stick contacts the drum |
| Clean up | Wipe the tea with the sponge | The sponge makes contact with the spilled tea |
| Press spacebar | Press the space on the keyboard | The spacebar is pressed |
| Cucumber basket | Put the green cucumber into the basket | The cucumber is in the basket |
| Pair up shoes | Pair up the shoes | Corresponding shoes are next to each other in parallel |
| Unplug charger | Unplug the charger | The charger is no longer plugged into the power strip |
| Lowest tune | Play the lowest pitch with the drum stick | The head of the drum stick contacts the lowest key |

**说明**: TABLE I: Definition of tasks. We list the text instructions for prompting VLM and the success criteria of each task.

#### Table 2: TABLE II: Comparison against baselines. PWTF better leverages the reasoning ability of VLM and improves the performance

| Methods | Water plant | Play drum | Press spacebar | Pair up shoes | Cucumber basket | Lowest tune | Unplug charger | Clean up |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Voxposer 1 1 footnotemark: 1 [21 ] | 6/10 | 2/10 | 0/10 | 2/10 | 8/10 | 0/10 | 4/10 | 0/10 |
| MOKA [14 ] | 4/10 | 5/10 | 0/10 | 2/10 | 5/10 | 0/10 | 0/10 | 5/10 |
| OpenVLA [26 ] | 1/10 | 0/10 | 1/10 | 0/10 | 4/10 | 0/10 | 0/10 | 0/10 |
| $\pi_{0} 0 [6 ]$ | 0/10 | 0/10 | 0/10 | 1/10 | 3/10 | 0/10 | 2/10 | 0/10 |
| OpenVLA-finetuned | 0/10 | 0/10 | 0/10 | 0/10 | 5/10 | 0/10 | 0/10 | 0/10 |
| $\pi_{0} 0 -finetuned$ | 0/10 | 0/10 | 0/10 | 2/10 | 2/10 | 0/10 | 3/10 | 2/10 |
| Ours | 5/10 | 6/10 | 5/10 | 6/10 | 9/10 | 4/10 | 7/10 | 5/10 |

**说明**: TABLE II: Comparison against baselines. PWTF better leverages the reasoning ability of VLM and improves the performance on most of the tasks. Besides, our image-based evaluation on results provides more flexibility than value map or key points, enabling challenging tasks that are hard to be parametrized by previous representations (e.g., play the lowest tune).

#### Table 3: TABLE III: Ablation study. We validate the effectiveness of our components. Multi-view observations are essential for ta

| Methods | Water plant | Play drum | Press spacebar | Pair up shoes | Cucumber basket | Lowest tune | Unplug charger | Clean up |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| w/o Views | 1/10 | 0/10 | 0/10 | 3/10 | 8/10 | 0/10 | 3/10 | 1/10 |
| w/o Subtasks | 5/10 | 0/10 | 6/10 | 3/10 | 8/10 | 2/10 | 7/10 | 0/10 |
| w/o CEM | 0/10 | 2/10 | 2/10 | 2/10 | 1/10 | 0/10 | 4/10 | 0/10 |
| Ours Full Method | 5/10 | 6/10 | 5/10 | 6/10 | 9/10 | 4/10 | 7/10 | 5/10 |

**说明**: TABLE III: Ablation study. We validate the effectiveness of our components. Multi-view observations are essential for tasks that are sensitive to perspectives. Subtask division improves the performance on tasks that require multi-stage planning. Most importantly, CEM significantly increases the sampling efficiency, facilitates effective planning.

#### Table 4: TABLE IV: Success rate with Sparse Voxels Rasterization. Our method is agnostic and robust to reconstruction methods.

| Methods | Water plant | Clean up | Cucumber in basket |
| --- | --- | --- | --- |
| SVRaster [45 ] | 4/10 | 6/10 | 7/10 |
| Gaussian Splatting | 5/10 | 5/10 | 9/10 |

**说明**: TABLE IV: Success rate with Sparse Voxels Rasterization. Our method is agnostic and robust to reconstruction methods.

#### Table 5: TABLE V: Success rate with different configurations. We evaluate on Water plant, Clean up, and Cucumber in basket.

| Configurations | 1/3 training | 2/3 training | Full training |
| --- | --- | --- | --- |
| 1/2 images | 0.3 / 0.4 / 0.8 | 0.3 / 0.3 / 0.8 | 0.2 / 0.4 / 0.9 |
| Full images | 0.5 / 0.6 / 0.9 | 0.5 / 0.4 / 0.8 | 0.5 / 0.5 / 0.9 |

**说明**: TABLE V: Success rate with different configurations. We evaluate on Water plant, Clean up, and Cucumber in basket.
## 实验解读

- 评价重点:围绕 adaptive-control、接触推理、接触丰富操作、in-hand-manipulation、language-conditioned-policy,优先看真实机器人结果、扰动恢复、接触稳定性、控制误差、成功率、实时性和 sim-to-real 差距。
- System0 视角:实验结论不只看最终成功率,还要看闭环过程中是否能处理状态估计误差、执行延迟、碰撞/接触突变和外部扰动。
- 与上层系统的关系:如果方法能在快闭环内稳定工作,它可以作为 System1 技能或 System2 规划的底层执行保障;如果只在离线或慢速设定下成立,则还不能直接作为身体级快闭环模块。
## 批判性思考

### 优点

- 论文把 adaptive-control、接触推理、接触丰富操作、in-hand-manipulation、language-conditioned-policy 放到机器人身体级闭环里讨论,和只做高层语义推理的工作相比,更接近真实机器人部署时会遇到的状态、动力学和安全约束。
- 若论文同时给出公式、系统结构图、硬件实验或消融结果,这些证据可以帮助判断方法是否真的能支撑毫秒到百毫秒级反应。

### 局限

- 需要重点核查控制频率、推理延迟、传感延迟、失败恢复、硬件磨损、未见场景泛化和安全边界;这些因素决定它能否成为可靠的 System0 模块。
- 对学习型方法,还要区分训练环境中的鲁棒性和真实部署中的鲁棒性,尤其是接触丰富任务、动态运动任务和人形机器人任务。
## 论文定位

- 研究对象:Prompting with the Future: Open-World Model Predictive Control with Interactive Digital Twins。
- 关键词:adaptive-control、接触推理、接触丰富操作、in-hand-manipulation、language-conditioned-policy、模型预测控制、robot-generalization、机器人操作、鲁棒控制、world-model。
- System0 关联:关注机器人身体级快反应链路中“状态/接触/局部感知 -> 约束/优化/策略 -> 动作”的短闭环,不把问题停留在语言规划或任务分解层。

## 速查卡片

> [!summary] Prompting with the Future
> - **论文**: https://www.roboticsproceedings.org/rss21/p145.pdf
> - **arXiv**: http://arxiv.org/abs/2506.13761v1
> - **arXiv HTML**: https://arxiv.org/html/2506.13761v1
