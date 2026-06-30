---
title: "Reconciling Reality through Simulation: A Real-To-Sim-to-Real Approach for Robust Manipulation"
method_name: "Reconciling Reality Simulation"
authors: ["Marcel Torne Villasevil"]
year: 2024
venue: "RSS"
tags: ["contact-reasoning", "robot-manipulation", "robust-control", "reinforcement-learning", "safe-control", "imitation-learning", "sim-to-real"]
image_source: "online"
arxiv_html: "https://arxiv.org/html/2403.03949v3"
---
# Reconciling Reality Simulation
## 一句话总结

> Reconciling Reality through Simulation: A Real-To-Sim-to-Real Approach for Robust Manipulation 主要落在 [[接触推理]]、[[模仿学习]]、[[强化学习]]、[[机器人操作]] 这一类问题上;从 System0 视角看,它的价值在于把身体状态、接触变化、局部感知或动力学约束接到机器人低层闭环里,让动作生成更稳定、安全、实时。
## 核心贡献

1. 围绕 **Reconciling Reality through Simulation: A Real-To-Sim-to-Real Approach for Robust Manipulation** 建立了一个与 接触推理、模仿学习、强化学习、机器人操作、鲁棒控制、safe-control 相关的机器人问题设定,重点不是高层语义规划,而是身体级闭环中的状态变化、接触变化或控制约束。
2. 方法层面的关键价值在于把感知、动力学、约束优化、策略学习或硬件反馈组织成可执行的低层动作生成机制,适合放进 System0 的快反应模块中理解。
3. 对系统集成而言,这类工作补上了 System1 技能选择与电机级控制之间的一层:它负责把上层目标变成动态可行、可恢复、可安全过滤的动作。
4. 若论文包含真实机器人、接触任务或仿真到真实迁移,核心应看它是否报告了闭环稳定性、扰动恢复、硬件延迟、失败模式和跨场景泛化。
## 问题背景

这篇论文可以放在 System0 的“具身快系统”背景下阅读:机器人需要在很短时间内根据本体状态、接触、力/触觉、局部视觉或动力学预测做出动作,而不是等待慢速规划器重新推理整个任务。接触推理、模仿学习、强化学习、机器人操作、鲁棒控制、safe-control、仿真到真实迁移 相关问题的难点通常不在“知道要做什么”,而在动作执行过程中状态会快速偏离、接触会突然改变、约束会实时收紧,低层系统必须及时修正。

### System0 关联

精读时优先看三件事:第一,论文使用哪些快速反馈量,例如关节状态、接触状态、力觉、触觉、深度或局部视觉;第二,反馈如何进入 MPC、WBC、策略网络、安全过滤器或恢复控制器;第三,输出是否直接影响机器人下一步动作,而不是只服务离线分析或高层任务规划。
## 方法详解

### System0 视角解析

从 System0 角度看,这篇论文的方法部分应当围绕 接触推理、模仿学习、强化学习、机器人操作、鲁棒控制、safe-control、仿真到真实迁移 来读。核心问题是:系统如何从快速变化的机器人状态中构造控制输入,如何处理接触、碰撞、约束、模型误差或执行延迟,以及如何把这些信息变成下一时刻的动作。

如果方法里有优化问题,重点看目标函数、约束和求解频率;如果方法里有学习策略,重点看观测空间、动作空间、训练扰动和部署时的闭环频率;如果方法里有硬件或传感器,重点看传感链路是否真的进入控制回路。这样读可以把论文和 System2 语言规划、System1 技能编排区分开来。

### 关键公式与机制

#### 公式 1: [[优化目标/约束]]

$$\{\mathcal{G}_{i}\}_{i=1}^{M}$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 2: [[优化目标/约束]]

$$\mathcal{S}=\{\{\mathcal{G}_{i}\}_{i=1}^{M},\mathcal{K},\mathcal{P}\}$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 3: [[优化目标/约束]]

$$\mathcal{G}_{i}\}_{i=1}^{M}$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 4: [[动力学或策略机制]]

$$\begin{split}total\_active\_time=t_{\textit{scan scene}}\\ +t_{\textit{scan object}}\cdot N_{\textit{objects}}\\ +t_{\textit{cut object}}\cdot N_{\textit{cut objects}}\\ +t_{\textit{add joint}}\cdot N_{joints}\end{split}$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 5: [[动力学或策略机制]]

$$t_{\textit{scan object}}=4:50$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 6: [[动力学或策略机制]]

$$t_{\textit{scan scene}}=3:14$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 7: [[优化目标/约束]]

$$\begin{split}\max_{\theta,\phi}\alpha\sum_{(s_{t},a_{t},r_{t})\in\tau_{\pi_{\theta_{\text{old}}}}}\text{min}(\frac{\pi_{\theta}(a_{t}| s_{t})}{ $\pi_{\theta_{\text{old}}(a_{t}$ | s_{t})}} $\hat{A}_{t},\\ \text{clip}(\frac{\pi_{\theta}(a_{t}$ | s_{t})}{ $\pi_{\theta_{\text{old}}(a_{t}$ | s_{t})}},1-\epsilon,1+\epsilon) $\hat{A}_{t})\\ +\beta\sum_{(s_{t},V_{t}^{\text{targ}})\in\tau_{\pi_{\theta_{\text{old}}}}}(V_{\phi}(s_{t})-V_{t}^{\text{targ}})^{2}\\ +\gamma\sum_{(s_{i},a_{i})\in\mathcal{D}_{\text{sim}}}\frac{\pi_{\theta}(a_{i}$ | s_{i})}{ $\sum_{a_{c}}\pi_{\theta}(a_{c}$ |s_{i})}\end{split}$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 8: [[优化目标/约束]]

$$\begin{split}\max_{\theta}\alpha\sum_{(s_{i},o_{i},a_{i})\sim\tau_{\pi_{\theta}}}\frac{\pi_{\theta}(\pi_{\text{teacher}}(s_{i})| o_{i})}{ $\sum_{a_{c}}\pi_{\theta}(a_{c}$ | o_{i})}\\ +\beta $\sum_{(o_{i},a_{i})\in\mathcal{D}_{\text{real}}}\frac{\pi_{\theta}(a_{i}$ | o_{i})}{ $\sum_{a_{c}}\pi_{\theta}(a_{c}$ |o_{i})}\end{split}$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 9: [[优化目标/约束]]

$$\mathcal{D}_{\text{sim}}=\{(o_{1}^{i},a_{1}^{i},s_{1}^{i})\dots,(o_{H}^{i},a_{H}^{i},s_{H}^{i})\}_{i=1}^{M}$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 10: [[优化目标/约束]]

$$\mathcal{D}_{\text{real}}=\{(o_{1}^{i},a_{1}^{i}),\dots,(o_{H}^{i},a_{H}^{i})\}_{i=1}^{N}$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。
## 关键图表

### Figure 1: Overview of the disturbances that RialTo is robust to in the different tasks that we

![Figure 1](https://arxiv.org/html/2403.03949v3/x10.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Overview of the disturbances that RialTo is robust to in the different tasks that we”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 2: Overview of the scenes generated using our GUI and used for evaluating RialTo

![Figure 2](https://arxiv.org/html/2403.03949v3/x11.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Overview of the scenes generated using our GUI and used for evaluating RialTo”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 3: Overview of the hardware setup used for evaluating RialTo. left: used for the kitche

![Figure 3](https://arxiv.org/html/2403.03949v3/x13.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Overview of the hardware setup used for evaluating RialTo. left: used for the kitche”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### 论文表格

#### Table 1: TABLE I: RialTo and imitation learning on placing a book on the shelf.

| | Only randomization | Distractors | Disturbances |
| --- | --- | --- | --- |
| BC (15 demos) | 10 $\pm$ 9% | 0 $\pm$ 0% | 0 $\pm$ 0% |
| BC (50 demos) | 40 $\pm$ 15% | 30 $\pm$ 16% | 20 $\pm$ 13% |
| RialTo (15 demos) | 90 $\pm$ 9% | 70 $\pm$ 14% | 60 $\pm$ 16% |

**说明**: TABLE I: RialTo and imitation learning on placing a book on the shelf.

#### Table 2: TABLE II: Real-World performance of policies trained with and without distractors on the task of placing a mug on a shel

| | Pose Randomization | Distractors |
| --- | --- | --- |
| RialTo without distractor training | 60 $\pm$ 15% | 30 $\pm$ 15% |
| RialTo with distractor training | 100 $\pm$ 0% | 70 $\pm$ 15% |

**说明**: TABLE II: Real-World performance of policies trained with and without distractors on the task of placing a mug on a shelf.

#### Table 3: TABLE III: Comparison of training RL from scratch against RL from real and sim demos. RL from sim and real demos seem to

| | Open | Book on | Plate on | Mug on | Open |
| --- | --- | --- | --- | --- | --- |
| | toaster | shelf | rack | shelf | drawer |
| RL from scratch with 0 demos | 62 $\pm$ 2% | 0 $\pm$ 0% | 2 $\pm$ 0% | 0 $\pm$ 0% | 0 $\pm$ 0% |
| RL fine-tuning from 15 real demos | 91 $\pm$ 1% | 90 $\pm$ 1% | 81 $\pm$ 2% | 81 $\pm$ 2% | 96 $\pm$ 1% |
| RL fine-tuning from 15 sim demos | 96 $\pm$ 1% | 89 $\pm$ 1% | 82 $\pm$ 2% | 82 $\pm$ 2% | 95 $\pm$ 1% |

**说明**: TABLE III: Comparison of training RL from scratch against RL from real and sim demos. RL from sim and real demos seem to be equivalent in most cases, but RL from scratch barely solves the task.

#### Table 4: TABLE IV: Specific parameters for each one of the tasks.

| Task | USD Name | Episode length | Randomized | Position | Position | Orientation | Orientation |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Parameters | | | Object Ids | Min (x,y,z) | Max (x,y,z) | Min (z-axis) | Max (z-axis) |
| Kitchen toaster | kitchentoaster3.usd | 130 | [267] | [0.3,-0.2,-0.2] | [0.7,0.1,0.2] | [-0.1] | [0.1] |
| Plate on rack | dishinrackv3.usd | 150 | [278, [270,287]] | [-0.4,-0.035,0] | [0,0.25,0] | [-0.52,0] | [0.52,0] |
| Mug on shelf | mugandshelf2.usd | 150 | [267,263] | [[-0.3,0,0], [-0.1,0.25,0]] | [[0.25,0.3,0.07], [0.4,0.4,0]] | [-0.52,-0.54] | [0.52, 0.54] |
| Book on shelf | booknshelve.usd | 130 | [277, [268,272]] | [[-0.25,-0.12,0], [-0.15,-0.05,0]] | [[0.15,0.28,0], [0.15,0.15,0]] | [-0.52,0] | [0.52,0] |
| Open cabinet | cabinet.usd | 90 | [268] | [-0.5,-0.1,0.1] | [0,0.3,-0.1] | [-0.52] | [0.52] |
| Open drawer | drawerbiggerhandle.usd | 80 | [268] | [-0.26,-0.07,-0.05] | [0.16,0.27,0] | -0.5 | 0.5 |
| Cup in trash | cupntrash.usd | 90 | [263, 266] | [[[-0.2, -0.3, -0.2], [-0.2,-0.12,0]]] | [[[0.2, 0.1, 0.2], [0.2,0.2,0]]] | [0,0] | [0,0] |
| Plate on rack from kitchen | dishsinklab.usd | 110 | [[263, 278, 270]] | [[[-0.25, -0.1, -0.1], [-0.1,0.05,0], [-0.2,0,0]]] | [[[0.1, 0.2, 0.1], [0.1,0.15,0], [0,0,0]]] | [0,-0.3,0] | [0,0.3,0] |

**说明**: TABLE IV: Specific parameters for each one of the tasks.

#### Table 5: TABLE V: Camera parameters for each task.

| Task | Position (x,y,z) | Rotation (quat) | Crop Min | Crop Max | Size |
| --- | --- | --- | --- | --- | --- |
| Parameters | Camera | Camera | Camera | Camera | Image |
| Kitchen toaster | [0.0, -0.37, 0.68] | [0.82,0.34,-0.20, -0.41] | [-0.8,-0.8,-0.8] | [0.8,0.8,0.8] | (640,480) |
| Plate on rack | [0.95,-0.4,0.68] | [0.78,0.36, 0.21, 0.46] | [-0.3,-0.6,0.02] | [0.9,0.6,1] | (640,480) |
| Mug on shelf | [0.95,-0.4,0.68] | [0.78,0.36, 0.21, 0.46] | [-0.3,-0.6,0.02] | [0.9,0.6,1] | (640,480) |
| Book on shelf | [0.95,-0.4,0.68] | [0.78,0.36, 0.21, 0.46] | [-0.3,-0.6,0.02] | [0.9,0.6,1] | (640,480) |
| Open cabinet | [0.95,-0.4,0.68] | [0.78,0.36, 0.21, 0.46] | [-0.3,-0.6,0.02] | [0.9,0.6,1] | (640,480) |
| Open drawer | [0.95,-0.4,0.68] | [0.78,0.36, 0.21, 0.46] | [-0.3,-0.6,0.02] | [0.9,0.6,1] | (640,480) |
| Cup in trash | [0.0, -0.37, 0.68] | [0.82,0.34,-0.20, -0.41] | [-1,-1,-1] | [1,1,1] | (640,480) |
| Plate on rack from kitchen | [0.0, -0.37, 0.68] | [0.82,0.34,-0.20, -0.41] | [-0.8,-0.8,-0.8] | [0.8,0.8,0.8] | (640,480) |

**说明**: TABLE V: Camera parameters for each task.

#### Table 6: TABLE VI: State-based policy training parameters. The rest of the parameters are the default as described in Stable Base

| MLP layers | PPO n_steps | PPO batch size | PPO BC batch size | PPO BC weight | Gradient Clipping |
| --- | --- | --- | --- | --- | --- |
| 256,256 | episode length | 31257 | 32 | 0.1 | 5 |

**说明**: TABLE VI: State-based policy training parameters. The rest of the parameters are the default as described in Stable Baselines 3 [53 ].

#### Table 7: TABLE VII: Point cloud generation and randomization parameters.

| Total pcd | Sample Arm | Dropout | Jitter | Jitter | Sample Object | Pcd | Pcd | Grid |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| points | Points (#) | ratio | ratio | noise | Meshes Points | Normalization | Scale | Size |
| 6000 | 3000 | [0.1,0.3] | 0.3 | N (0, 0.01) N 0 0.01  $\mathcal{N}(0,0.01)$ | 1000 | [0,0,0] (toaster) [0.35,0,0.4] (others) | 0.625 (toaster) 1 (others) | 32x32x32 |

**说明**: TABLE VII: Point cloud generation and randomization parameters.

#### Table 8: TABLE VIII: Point cloud teacher-student distillation parameters.

| MLP layers | lr | Optimizer | Batch Size | Nb full pcd traj | Nb simulated pcd traj | Nb simulated pcd traj (distractors) | Nb real traj |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 256,256 | 0.0003 | AdamW | 32-64 | 15000 | 5000 | 1000 | 15 |

**说明**: TABLE VIII: Point cloud teacher-student distillation parameters.

#### Table 9: TABLE IX: Comparison of the plain imitation learning baseline (IL) against adding new distractors (IL with distractors)

| | Pose | Distractors | Disturbances |
| --- | --- | --- | --- |
| | randomization | | |
| IL | 40 $\pm$ 15% | 50 $\pm$ 17% | 10 $\pm$ 9% |
| IL with distractors | 50 $\pm$ 17% | 20 $\pm$ 13% | 10 $\pm$ 9% |

**说明**: TABLE IX: Comparison of the plain imitation learning baseline (IL) against adding new distractors (IL with distractors) on the task of opening the drawer. No improvement is observed.

#### Table 10: TABLE X: Comparison of training RL from different amounts of real-world demos.

| | Book on | Open |
| --- | --- | --- |
| | shelf | drawer |
| RL fine-tuning from 0 real demos | 0 $\pm$ 0% | 0 $\pm$ 0% |
| RL fine-tuning from 5 real demos | 0 $\pm$ 0% | 89 $\pm$ 1% |
| RL fine-tuning from 10 real demos | 0 $\pm$ 0% | 96 $\pm$ 1% |
| RL fine-tuning from 15 real demos | 90 $\pm$ 2% | 96 $\pm$ 1% |

**说明**: TABLE X: Comparison of training RL from different amounts of real-world demos.

#### Table 11: TABLE XI: Comparison of using RialTo with added synthetic assets against standard RialTo on the task of opening the draw

| | Pose | Distractors |
| --- | --- | --- |
| | randomization | |
| RialTo | 90 $\pm$ 9% | 90 $\pm$ 9% |
| RialTo + synthetic assets | 90 $\pm$ 9% | 80 $\pm$ 13% |

**说明**: TABLE XI: Comparison of using RialTo with added synthetic assets against standard RialTo on the task of opening the drawer in the real world. No improvement is observed.

#### Table 12: TABLE XII: Comparison of training RialTo on multiple tasks against single-task RialTo. No improvement is observed.

| | Open | Mug |
| --- | --- | --- |
| | drawer | on shelf |
| Imitation learning | 40 $\pm$ 17% | 10 $\pm$ 9% |
| RialTo | 90 $\pm$ 9% | 100 $\pm$ 0% |
| RialTo multitask | 90 $\pm$ 9% | 80 $\pm$ 15% |

**说明**: TABLE XII: Comparison of training RialTo on multiple tasks against single-task RialTo. No improvement is observed.

#### Table 13: TABLE XIII: Comparison of performance in simulation (top) and the real world (bottom).

| | Kitchen | Book on | Plate on | Mug on | Open | Open |
| --- | --- | --- | --- | --- | --- | --- |
| | toaster | shelf | rack | shelf | drawer | cabinet |
| Performance in simulation | 90 $\pm$ 4% | 84 $\pm$ 5% | 80 $\pm$ 6% | 72 $\pm$ 6% | 95 $\pm$ 3% | 92 $\pm$ 4% |
| Performance in the real world | 90 $\pm$ 9% | 90 $\pm$ 9% | 90 $\pm$ 9% | 100 $\pm$ 0% | 90 $\pm$ 9% | 85 $\pm$ 8% |

**说明**: TABLE XIII: Comparison of performance in simulation (top) and the real world (bottom).

#### Table 14: TABLE XIV: Detailed time spent by each user in the user study, see Section VI.

| | Scan | Process + Upload 1st Scan (idle) | Cut | Joint | 2nd Scan | Process + Upload 2nd Scan (idle) | Total time | Total active time |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| User 1 | 2:25 | 5:41 | 4:15 | 4:56 | 8:10 | 10:45 | 36:12 | 19:46 |
| User 2 | 6:30 | 12:57 | 3:32 | 3:51 | 2:37 | 4:19 | 33:46 | 16:30 |
| User 3 | 3:52 | 5:52 | 4:35 | 4:14 | 3:26 | 4:15 | 26:14 | 16:07 |
| User 4 | 2:34 | 2:06 | 2:48 | 1:41 | 5:14 | 4:33 | 19:06 | 12:27 |
| User 5 | 1:32 | 2:33 | 4:43 | 1:28 | 4:34 | 3:50 | 18:40 | 12:17 |
| User 6 | 2:30 | 3:52 | 2:08 | 1:17 | 4:59 | 2:26 | 17:12 | 10:54 |

**说明**: TABLE XIV: Detailed time spent by each user in the user study, see Section VI.
## 实验解读

- 评价重点:围绕 接触推理、模仿学习、强化学习、机器人操作、鲁棒控制,优先看真实机器人结果、扰动恢复、接触稳定性、控制误差、成功率、实时性和 sim-to-real 差距。
- System0 视角:实验结论不只看最终成功率,还要看闭环过程中是否能处理状态估计误差、执行延迟、碰撞/接触突变和外部扰动。
- 与上层系统的关系:如果方法能在快闭环内稳定工作,它可以作为 System1 技能或 System2 规划的底层执行保障;如果只在离线或慢速设定下成立,则还不能直接作为身体级快闭环模块。
## 批判性思考

### 优点

- 论文把 接触推理、模仿学习、强化学习、机器人操作、鲁棒控制 放到机器人身体级闭环里讨论,和只做高层语义推理的工作相比,更接近真实机器人部署时会遇到的状态、动力学和安全约束。
- 若论文同时给出公式、系统结构图、硬件实验或消融结果,这些证据可以帮助判断方法是否真的能支撑毫秒到百毫秒级反应。

### 局限

- 需要重点核查控制频率、推理延迟、传感延迟、失败恢复、硬件磨损、未见场景泛化和安全边界;这些因素决定它能否成为可靠的 System0 模块。
- 对学习型方法,还要区分训练环境中的鲁棒性和真实部署中的鲁棒性,尤其是接触丰富任务、动态运动任务和人形机器人任务。
## 论文定位

- 研究对象:Reconciling Reality through Simulation: A Real-To-Sim-to-Real Approach for Robust Manipulation。
- 关键词:接触推理、模仿学习、强化学习、机器人操作、鲁棒控制、safe-control、仿真到真实迁移。
- System0 关联:关注机器人身体级快反应链路中“状态/接触/局部感知 -> 约束/优化/策略 -> 动作”的短闭环,不把问题停留在语言规划或任务分解层。

## 速查卡片

> [!summary] Reconciling Reality Simulation
> - **论文**: https://www.roboticsproceedings.org/rss20/p015.pdf
> - **arXiv**: http://arxiv.org/abs/2403.03949v3
> - **arXiv HTML**: https://arxiv.org/html/2403.03949v3
