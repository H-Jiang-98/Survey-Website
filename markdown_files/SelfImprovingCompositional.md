---
title: "Self-Improving Robot Policy with Compositional World Model"
method_name: "Self Improving Compositional"
authors: ["Jiazhi Yang"]
year: 2026
venue: "RSS"
tags: ["contact-reasoning", "robot-manipulation", "robust-control", "reinforcement-learning", "safe-control", "contact-rich-manipulation", "robot-generalization", "closed-loop-control"]
image_source: "online"
arxiv_html: "https://arxiv.org/html/2602.11075v2"
---
# Self Improving Compositional
## 一句话总结

> Self-Improving Robot Policy with Compositional World Model 主要落在 [[closed-loop-control]]、[[接触推理]]、[[接触丰富操作]]、[[dynamics-modeling]] 这一类问题上;从 System0 视角看,它的价值在于把身体状态、接触变化、局部感知或动力学约束接到机器人低层闭环里,让动作生成更稳定、安全、实时。
## 核心贡献

1. 围绕 **Self-Improving Robot Policy with Compositional World Model** 建立了一个与 closed-loop-control、接触推理、接触丰富操作、dynamics-modeling、policy-learning、强化学习 相关的机器人问题设定,重点不是高层语义规划,而是身体级闭环中的状态变化、接触变化或控制约束。
2. 方法层面的关键价值在于把感知、动力学、约束优化、策略学习或硬件反馈组织成可执行的低层动作生成机制,适合放进 System0 的快反应模块中理解。
3. 对系统集成而言,这类工作补上了 System1 技能选择与电机级控制之间的一层:它负责把上层目标变成动态可行、可恢复、可安全过滤的动作。
4. 若论文包含真实机器人、接触任务或仿真到真实迁移,核心应看它是否报告了闭环稳定性、扰动恢复、硬件延迟、失败模式和跨场景泛化。
## 问题背景

这篇论文可以放在 System0 的“具身快系统”背景下阅读:机器人需要在很短时间内根据本体状态、接触、力/触觉、局部视觉或动力学预测做出动作,而不是等待慢速规划器重新推理整个任务。closed-loop-control、接触推理、接触丰富操作、dynamics-modeling、policy-learning、强化学习、机器人操作 相关问题的难点通常不在“知道要做什么”,而在动作执行过程中状态会快速偏离、接触会突然改变、约束会实时收紧,低层系统必须及时修正。

### System0 关联

精读时优先看三件事:第一,论文使用哪些快速反馈量,例如关节状态、接触状态、力觉、触觉、深度或局部视觉;第二,反馈如何进入 MPC、WBC、策略网络、安全过滤器或恢复控制器;第三,输出是否直接影响机器人下一步动作,而不是只服务离线分析或高层任务规划。
## 方法详解

### System0 视角解析

从 System0 角度看,这篇论文的方法部分应当围绕 closed-loop-control、接触推理、接触丰富操作、dynamics-modeling、policy-learning、强化学习、机器人操作、鲁棒控制 来读。核心问题是:系统如何从快速变化的机器人状态中构造控制输入,如何处理接触、碰撞、约束、模型误差或执行延迟,以及如何把这些信息变成下一时刻的动作。

如果方法里有优化问题,重点看目标函数、约束和求解频率;如果方法里有学习策略,重点看观测空间、动作空间、训练扰动和部署时的闭环频率;如果方法里有硬件或传感器,重点看传感链路是否真的进入控制回路。这样读可以把论文和 System2 语言规划、System1 技能编排区分开来。
## 关键图表

### Figure 1: Workflow of compositional world model. Top: Training recipe upon proper model initia

![Figure 1](https://arxiv.org/html/2602.11075v2/x4.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Workflow of compositional world model. Top: Training recipe upon proper model initia”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 2: Self-improving loop of RISE. Our learning pipeline encompasses two stages. Top: Roll

![Figure 2](https://arxiv.org/html/2602.11075v2/x5.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Self-improving loop of RISE. Our learning pipeline encompasses two stages. Top: Roll”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 3: Visual ablation study on training strategies. Compared to the other baselines, which

![Figure 3](https://arxiv.org/html/2602.11075v2/x12.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Visual ablation study on training strategies. Compared to the other baselines, which”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### 论文表格

#### Table 1: TABLE I: Performance comparisons on real-world tasks. We evaluate success rates and scores across three diverse tasks,

| Method | Dynamic Brick Sorting | Backpack Packing | Box Closing | | | |
| --- | --- | --- | --- | --- | --- | --- |
| Succ. (%) | Score | Succ. (%) | Score | Succ. (%) | Score | |
| $\pi_{0.5} [8 ]$ | 35.00 | 8.28 | 30.00 | 4.25 | 35.00 | 7.50 |
| $\pi_{0.5} +DAgger [73, 45 ]$ | 15.00 | 6.10 | 50.00 | 7.00 | 40.00 | 7.50 |
| $\pi_{0.5} +PPO [75 ]$ | 10.00 | 7.68 | 35.00 | 5.88 | 10.00 | 4.75 |
| $\pi_{0.5} +DSRL [80 ]$ | 10.00 | 6.65 | 10.00 | 3.50 | 10.00 | 7.63 |
| RECAP [2 ] | 50.00 | 9.00 | 40.00 | 6.13 | 60.00 | 8.13 |
| RISE (Ours) | 85.00 | 9.78 | 85.00 | 9.50 | 95.00 | 9.88 |

**说明**: TABLE I: Performance comparisons on real-world tasks. We evaluate success rates and scores across three diverse tasks, ranging from dynamic sorting to precise packing. RISE exhibits superior performance compared to baselines in all scenarios.

#### Table 2: TABLE II: Ablation on offline data ratio. Overall performance peaks at 0.6, indicating that balanced offline data is cr

| Ratio | Pick&Place | Sort | Complete | |
| --- | --- | --- | --- | --- |
| Succ. (%) | Acc. (%) | Succ. (%) | Score | |
| 0.1 | 15.00 | 83.33 | 5.00 | 1.35 |
| 0.3 | 78.75 | 80.95 | 25.00 | 7.03 |
| 0.6 | 90.00 | 87.50 | 50.00 | 8.32 |
| 0.9 | 90.00 | 80.56 | 30.00 | 7.90 |

**说明**: TABLE II: Ablation on offline data ratio. Overall performance peaks at 0.6, indicating that balanced offline data is crucial for complex generalization.

#### Table 3: TABLE III: Ablation on online action and state integration. Results demonstrate the necessity of incorporating both onl

| Online Action | Online State | Pick&Place | Sort | Complete | |
| --- | --- | --- | --- | --- | --- |
| Succ. (%) | Acc. (%) | Succ. (%) | Score | | |
| ✗ | ✗ | 80.00 | 76.56 | 35.00 | 6.98 |
| ✓ | ✗ | 96.25 | 84.42 | 40.00 | 8.73 |
| ✓ | ✓ | 98.75 | 92.41 | 70.00 | 9.43 |

**说明**: TABLE III: Ablation on online action and state integration. Results demonstrate the necessity of incorporating both online action proposed by the rollout policy and the online state generated by the dynamics model.

#### Table 4: TABLE IV: Ablations on the modular designs of dynamics and value models. “w/o Progress” indicates that the value model

| Module Variants | Pick&Place | Sort | Complete | | |
| --- | --- | --- | --- | --- | --- |
| Succ. (%) | Acc. (%) | Succ. (%) | Score | | |
| Dynamics | w/o Pre-train | 97.50 | 60.26 | 15.00 | 7.43 |
| w/o Task-Centric | 93.75 | 89.33 | 40.00 | 8.78 | |
| Value | w/o Progress | 95.00 | 86.84 | 50.00 | 8.78 |
| w/o TD Learning | 98.75 | 72.15 | 35.00 | 8.38 | |
| RISE (Ours) | w/ all designs | 98.75 | 92.41 | 70.00 | 9.43 |

**说明**: TABLE IV: Ablations on the modular designs of dynamics and value models. “w/o Progress” indicates that the value model is trained without the auxiliary progress loss. Our full architecture proves to be the most effective across all metrics.

#### Table 5: TABLE V: Quantitative comparison of dynamics models. ↑ $\uparrow$ (↓ $\downarrow)$ denotes higher (lower) is better. Our

| Method | PSNR ↑ \uparrow | LPIPS ↓ \downarrow | SSIM ↑ \uparrow | FVD ↓ \downarrow | EPE ↓ \downarrow |
| --- | --- | --- | --- | --- | --- |
| Experiment #1: Fine-tuning on our real world tasks | | | | | |
| Cosmos | 21.17 | 0.14 | 0.79 | 97.90 | 1.21 |
| GE | 21.16 | 0.11 | 0.79 | 85.72 | 1.05 |
| RISE (w/o Task-Centric) | 22.67 | 0.08 | 0.80 | 61.22 | 0.68 |
| RISE (Ours) | 23.90 | 0.07 | 0.82 | 66.84 | 0.54 |
| Experiment #2: Fine-tuning on Bridge dataset [81 ] | | | | | |
| Cosmos | 21.32 | 0.14 | 0.80 | 73.21 | 1.18 |
| GE | 21.47 | 0.12 | 0.79 | 64.55 | 0.96 |
| RISE (w/o Task-Centric) | 22.61 | 0.10 | 0.78 | 49.07 | 0.72 |
| RISE (Ours) | 23.68 | 0.10 | 0.82 | 45.21 | 0.64 |

**说明**: TABLE V: Quantitative comparison of dynamics models. ↑ $\uparrow$ (↓ $\downarrow)$ denotes higher (lower) is better. Our method shows superior motion accuracy (EPE) and perceptual quality across both real-world tasks in Fig. 2 and the Bridge dataset [81 ].

#### Table 6: TABLE VI: Quantitative ablation on the pre-training of our dynamics model.

| Method | PSNR ↑ \uparrow | LPIPS ↓ \downarrow | SSIM ↑ \uparrow | FVD ↓ \downarrow | EPE ↓ \downarrow |
| --- | --- | --- | --- | --- | --- |
| RISE (w/o pre-train) | 20.95 | 0.11 | 0.78 | 83.36 | 1.09 |
| RISE (Ours) | 23.90 | 0.07 | 0.82 | 66.84 | 0.54 |

**说明**: TABLE VI: Quantitative ablation on the pre-training of our dynamics model.

#### Table 7: TABLE VII: Task evaluation standard.

| Task | Sub-goals | Total | Score |
| --- | --- | --- | --- |
| Conveyor | Grasp brick | 10 | 1.0 each |
| Place in matched bin | 1.5 each | | |
| Workspace cleared | 10.0 max | | |
| Backpack | Open bag & Insert items | 10 | 2.5 |
| Lift to settle contents | 5.0 | | |
| Zip halfway | 7.5 | | |
| Zip fully | 10.0 max | | |
| Box | Load cup | 10 | 2.5 |
| Fold side flaps | 5.0 | | |
| Fold rear flap | 7.5 | | |
| Tuck locking tab | 10.0 max | | |

**说明**: TABLE VII: Task evaluation standard.

#### Table 8: TABLE VIII: Hyper-parameters of dynamics model.

| Hyperparameter | Value |
| --- | --- |
| Basics | |
| Model initialization | GE-Base [59 ] |
| Input / Prediction frames | 4 / 25 |
| Number of views | 3 |
| Sampling frequency (pre-train / Fine-tune) | 30 / 15 Hz |
| Optimization | |
| Training steps (pre-train / Fine-tune) | 120k / 50k |
| Batch size (pre-train / Fine-tune) | 512 / 64 |
| Optimizer | AdamW |
| Learning rate | 1  10 - 4 1\times 10^{-4} |
| Conditioned noise level  \sigma | 0.2 |

**说明**: TABLE VIII: Hyper-parameters of dynamics model.

#### Table 9: TABLE IX: Hyper-parameters of value model.

| Hyperparameter | Value |
| --- | --- |
| Basics | |
| Model initialization | $\pi_{0.5} [8 ]$ |
| Input frames | 1 |
| Number of views | 3 |
| Optimization | |
| Training steps | 50k |
| Batch size | 64 |
| Optimizer | AdamW |
| Learning rate | 2.5  10 - 5 2.5\times 10^{-5} |
| Value discount factor | 0.995 0.995 |

**说明**: TABLE IX: Hyper-parameters of value model.

#### Table 10: TABLE X: Hyper-parameters of policy self-improving.

| Hyperparameter | Value |
| --- | --- |
| Batch size | 64 |
| Optimizer | cosine |
| Learning rate | 1  10 - 4 1\times 10^{-4} |
| Minimum learning rate ratio | 0.1 |
| Rollout ema decay rate | 0.995 |
| Action chunk size | 50 |
| Action dimension | 14 |

**说明**: TABLE X: Hyper-parameters of policy self-improving.
## 实验解读

- 评价重点:围绕 closed-loop-control、接触推理、接触丰富操作、dynamics-modeling、policy-learning,优先看真实机器人结果、扰动恢复、接触稳定性、控制误差、成功率、实时性和 sim-to-real 差距。
- System0 视角:实验结论不只看最终成功率,还要看闭环过程中是否能处理状态估计误差、执行延迟、碰撞/接触突变和外部扰动。
- 与上层系统的关系:如果方法能在快闭环内稳定工作,它可以作为 System1 技能或 System2 规划的底层执行保障;如果只在离线或慢速设定下成立,则还不能直接作为身体级快闭环模块。
## 批判性思考

### 优点

- 论文把 closed-loop-control、接触推理、接触丰富操作、dynamics-modeling、policy-learning 放到机器人身体级闭环里讨论,和只做高层语义推理的工作相比,更接近真实机器人部署时会遇到的状态、动力学和安全约束。
- 若论文同时给出公式、系统结构图、硬件实验或消融结果,这些证据可以帮助判断方法是否真的能支撑毫秒到百毫秒级反应。

### 局限

- 需要重点核查控制频率、推理延迟、传感延迟、失败恢复、硬件磨损、未见场景泛化和安全边界;这些因素决定它能否成为可靠的 System0 模块。
- 对学习型方法,还要区分训练环境中的鲁棒性和真实部署中的鲁棒性,尤其是接触丰富任务、动态运动任务和人形机器人任务。
## 论文定位

- 研究对象:Self-Improving Robot Policy with Compositional World Model。
- 关键词:closed-loop-control、接触推理、接触丰富操作、dynamics-modeling、policy-learning、强化学习、机器人操作、鲁棒控制、safe-control、scalable-robot-learning。
- System0 关联:关注机器人身体级快反应链路中“状态/接触/局部感知 -> 约束/优化/策略 -> 动作”的短闭环,不把问题停留在语言规划或任务分解层。

## 速查卡片

> [!summary] Self Improving Compositional
> - **论文**: https://www.roboticsproceedings.org/rss22/p012.pdf
> - **arXiv**: http://arxiv.org/abs/2602.11075v2
> - **arXiv HTML**: https://arxiv.org/html/2602.11075v2
