---
title: "Contact-Anchored Policies: Contact Conditioning Creates Strong Robot Utility Models"
method_name: "Contact-Anchored Policies"
authors: ["Zichen Jeff Cui"]
year: 2026
venue: "RSS"
tags: ["contact-reasoning", "robot-manipulation", "robust-control", "real-time-control", "imitation-learning", "robot-generalization", "sim-to-real"]
image_source: "online"
arxiv_html: "https://arxiv.org/html/2602.09017v1"
---
# Contact-Anchored Policies
## 一句话总结

> Contact-Anchored Policies: Contact Conditioning Creates Strong Robot Utility Models 主要落在 [[接触推理]]、[[模仿学习]]、[[inference-time-algorithm]]、[[reactive-control]] 这一类问题上;从 System0 视角看,它的价值在于把身体状态、接触变化、局部感知或动力学约束接到机器人低层闭环里,让动作生成更稳定、安全、实时。
## 核心贡献

1. 围绕 **Contact-Anchored Policies: Contact Conditioning Creates Strong Robot Utility Models** 建立了一个与 接触推理、模仿学习、inference-time-algorithm、reactive-control、robot-generalization、机器人操作 相关的机器人问题设定,重点不是高层语义规划,而是身体级闭环中的状态变化、接触变化或控制约束。
2. 方法层面的关键价值在于把感知、动力学、约束优化、策略学习或硬件反馈组织成可执行的低层动作生成机制,适合放进 System0 的快反应模块中理解。
3. 对系统集成而言,这类工作补上了 System1 技能选择与电机级控制之间的一层:它负责把上层目标变成动态可行、可恢复、可安全过滤的动作。
4. 若论文包含真实机器人、接触任务或仿真到真实迁移,核心应看它是否报告了闭环稳定性、扰动恢复、硬件延迟、失败模式和跨场景泛化。
## 问题背景

这篇论文可以放在 System0 的“具身快系统”背景下阅读:机器人需要在很短时间内根据本体状态、接触、力/触觉、局部视觉或动力学预测做出动作,而不是等待慢速规划器重新推理整个任务。接触推理、模仿学习、inference-time-algorithm、reactive-control、robot-generalization、机器人操作、鲁棒控制 相关问题的难点通常不在“知道要做什么”,而在动作执行过程中状态会快速偏离、接触会突然改变、约束会实时收紧,低层系统必须及时修正。

### System0 关联

精读时优先看三件事:第一,论文使用哪些快速反馈量,例如关节状态、接触状态、力觉、触觉、深度或局部视觉;第二,反馈如何进入 MPC、WBC、策略网络、安全过滤器或恢复控制器;第三,输出是否直接影响机器人下一步动作,而不是只服务离线分析或高层任务规划。
## 方法详解

### System0 视角解析

从 System0 角度看,这篇论文的方法部分应当围绕 接触推理、模仿学习、inference-time-algorithm、reactive-control、robot-generalization、机器人操作、鲁棒控制、仿真到真实迁移 来读。核心问题是:系统如何从快速变化的机器人状态中构造控制输入,如何处理接触、碰撞、约束、模型误差或执行延迟,以及如何把这些信息变成下一时刻的动作。

如果方法里有优化问题,重点看目标函数、约束和求解频率;如果方法里有学习策略,重点看观测空间、动作空间、训练扰动和部署时的闭环频率;如果方法里有硬件或传感器,重点看传感链路是否真的进入控制回路。这样读可以把论文和 System2 语言规划、System1 技能编排区分开来。

### 关键公式与机制

#### 公式 1: [[优化目标/约束]]

$$\mathcal{D}\subset\mathcal{O}\times\mathcal{A}$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 2: [[优化目标/约束]]

$$\mathcal{L}(\mathcal{D})$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 3: [[动力学或策略机制]]

$$p_{t}=A_{t}^{-1}A_{c}p_{c}$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 4: [[动力学或策略机制]]

$$\pi(a_{t:t+h}|o_{t-k:t},p_{t-k:t})$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 5: [[动力学或策略机制]]

$$z_{v}\in\mathbb{R}^{256}$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 6: [[动力学或策略机制]]

$$z_{c}\in\mathbb{R}^{256}$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 7: [[动力学或策略机制]]

$$p_{0}=d_{u,v}K^{-1}[u,v,1]^{T}$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 8: [[动力学或策略机制]]

$$p_{t}=A_{t}^{-1}A_{0}p_{0}$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 9: [[动力学或策略机制]]

$$\pi_{0.5\texttt{-DROID}}$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 10: [[动力学或策略机制]]

$$\pi_{0.5\texttt{-DROID}}$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。
## 关键图表

### Figure 1: Pipeline / core system figure: The process of data labeling, training, and inference for Contact-Anchored Policies

![Figure 1](https://arxiv.org/html/2602.09017v1/x2.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Pipeline / core system figure: The process of data labeling, training, and inference for Contact-Anchored Policies”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 2: Left: Sim-to-real correlation for single-blind EgoGym-Pick evaluations. Right: Analys

![Figure 2](https://arxiv.org/html/2602.09017v1/x9.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Left: Sim-to-real correlation for single-blind EgoGym-Pick evaluations. Right: Analys”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 3: Pipeline for Extracting Gripper Label By Using SAM2

![Figure 3](https://arxiv.org/html/2602.09017v1/x11.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Pipeline for Extracting Gripper Label By Using SAM2”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### 论文表格

#### Table 1: Table 1: Evaluation results of CAP and baselines on our three different tasks and four different robot embodiments.

| Task | Robot | Model | Success rate |
| --- | --- | --- | --- |
| Pick | Stretch | CAP + Retry | 90.4 % ± 6.0 % 90.4\%\pm 6.0\% |
| Pick | Stretch | CAP | 83.2 % ± 7.9 % 83.2\%\pm 7.9\% |
| Pick | Stretch | AnyGrasp | 46.7 % ± 7.9 % 46.7\%\pm 7.9\% |
| Pick | Franka | CAP | 79.0 % ± 10.9 % 79.0\%\pm 10.9\% |
| Pick | Franka | CAP-VLM | 81.0 % ± 9.2 % 81.0\%\pm 9.2\% |
| Pick | Franka | $\pi_{0.5\texttt{-DROID}}$ | 25.0 % ± 15.2 % 25.0\%\pm 15.2\% |
| Pick | XArm | CAP | 83.0 % ± 17.9 % 83.0\%\pm 17.9\% |
| Pick | UR3e | CAP | 70.0 % ± 15.2 % 70.0\%\pm 15.2\% |
| Open | Stretch | CAP + Retry | 91.0 % ± 5.3 % 91.0\%\pm 5.3\% |
| Open | Stretch | CAP | 81.0 % ± 10.7 % 81.0\%\pm 10.7\% |
| Open | Stretch | Stretch-Open | 58.0 % ± 29.3 % 58.0\%\pm 29.3\% |
| Close | Stretch | CAP + Retry | 98.0 % ± 3.0 % 98.0\%\pm 3.0\% |
| Close | Stretch | CAP | 96.0 % ± 3.5 % 96.0\%\pm 3.5\% |
| EgoGym-Pick | Sim Gripper | CAP | 79.88 % ± 1.1 % 79.88\%\pm 1.1\% |
| EgoGym-Pick | Sim Franka | $\pi_{0.5\texttt{-DROID}}$ | 20.9 % ± 1.1 % 20.9\%\pm 1.1\% |

**说明**: Table 1: Evaluation results of CAP and baselines on our three different tasks and four different robot embodiments.

#### Table 2: Table 2: Success rate by stages for long-horizon tasks

| Get coffee beans | |
| --- | --- |
| Stage | Success |
| Open cabinet | 10/10 |
| Pick bag | 7/10 |
| Drop bag | 7/10 |
| Close cabinet | 6/10 |
| Clear table | |
| Stage | Success |
| 1st Object | 10/10 |
| 2nd Object | 10/10 |
| 3rd Object | 10/10 |
| 4th Object | 10/10 |
| 5th Object | 10/10 |

**说明**: Table 2: Success rate by stages for long-horizon tasks

#### Table 3: Table 3: Ablation results of CAP on the Close task.

| Model | Success rate |
| --- | --- |
| CAP - RGB Only Ablation | 58 % ± 28.2 % 58\%\pm 28.2\% |
| CAP | 96 % ± 3.5 % 96\%\pm 3.5\% |

**说明**: Table 3: Ablation results of CAP on the Close task.

#### Table 4: Table 4: Hyperparameters used for Pick, Open, and Close tasks

| Hyperparameter | Pick | Open | Close |
| --- | --- | --- | --- |
| Obs window size | 3 | 3 | 3 |
| Training Steps | 308,565 | 364,811 | 277,522 |
| Batch Size | 256 | 200 | 200 |
| Learning Rate | 3e-4 | 1e-4 | 2.7e-4 |
| Transformer Depth | 8 | 8 | 8 |
| Attn Heads | 8 | 8 | 8 |
| Embedding Dim | 512 | 512 | 256 |
| VQ-VAE codebook size | 16 | 32 | 32 |
| VQ-VAE embedding dim | 512 | 512 | 512 |

**说明**: Table 4: Hyperparameters used for Pick, Open, and Close tasks
## 实验解读

- 评价重点:围绕 接触推理、模仿学习、inference-time-algorithm、reactive-control、robot-generalization,优先看真实机器人结果、扰动恢复、接触稳定性、控制误差、成功率、实时性和 sim-to-real 差距。
- System0 视角:实验结论不只看最终成功率,还要看闭环过程中是否能处理状态估计误差、执行延迟、碰撞/接触突变和外部扰动。
- 与上层系统的关系:如果方法能在快闭环内稳定工作,它可以作为 System1 技能或 System2 规划的底层执行保障;如果只在离线或慢速设定下成立,则还不能直接作为身体级快闭环模块。
## 批判性思考

### 优点

- 论文把 接触推理、模仿学习、inference-time-algorithm、reactive-control、robot-generalization 放到机器人身体级闭环里讨论,和只做高层语义推理的工作相比,更接近真实机器人部署时会遇到的状态、动力学和安全约束。
- 若论文同时给出公式、系统结构图、硬件实验或消融结果,这些证据可以帮助判断方法是否真的能支撑毫秒到百毫秒级反应。

### 局限

- 需要重点核查控制频率、推理延迟、传感延迟、失败恢复、硬件磨损、未见场景泛化和安全边界;这些因素决定它能否成为可靠的 System0 模块。
- 对学习型方法,还要区分训练环境中的鲁棒性和真实部署中的鲁棒性,尤其是接触丰富任务、动态运动任务和人形机器人任务。
## 论文定位

- 研究对象:Contact-Anchored Policies: Contact Conditioning Creates Strong Robot Utility Models。
- 关键词:接触推理、模仿学习、inference-time-algorithm、reactive-control、robot-generalization、机器人操作、鲁棒控制、仿真到真实迁移、zero-shot。
- System0 关联:关注机器人身体级快反应链路中“状态/接触/局部感知 -> 约束/优化/策略 -> 动作”的短闭环,不把问题停留在语言规划或任务分解层。

## 速查卡片

> [!summary] Contact-Anchored Policies
> - **论文**: https://www.roboticsproceedings.org/rss22/p141.pdf
> - **arXiv**: http://arxiv.org/abs/2602.09017v1
> - **arXiv HTML**: https://arxiv.org/html/2602.09017v1
