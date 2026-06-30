---
title: "ViTacFormer: Learning Cross-Modal Representation for Visuo-Tactile Dexterous Manipulation"
method_name: "ViTacFormer"
authors: ["Liang Heng"]
year: 2026
venue: "RSS"
tags: ["contact-reasoning", "robot-manipulation", "robust-control", "real-time-control", "adaptive-control", "imitation-learning", "robot-generalization", "dexterous-manipulation", "tactile-feedback"]
image_source: "online"
arxiv_html: "https://arxiv.org/html/2506.15953v2"
---
# ViTacFormer
## 一句话总结

> ViTacFormer: Learning Cross-Modal Representation for Visuo-Tactile Dexterous Manipulation 主要落在 [[adaptive-control]]、[[接触推理]]、[[灵巧操作]]、[[模仿学习]] 这一类问题上;从 System0 视角看,它的价值在于把身体状态、接触变化、局部感知或动力学约束接到机器人低层闭环里,让动作生成更稳定、安全、实时。
## 核心贡献

1. 围绕 **ViTacFormer: Learning Cross-Modal Representation for Visuo-Tactile Dexterous Manipulation** 建立了一个与 adaptive-control、接触推理、灵巧操作、模仿学习、reactive-control、机器人操作 相关的机器人问题设定,重点不是高层语义规划,而是身体级闭环中的状态变化、接触变化或控制约束。
2. 方法层面的关键价值在于把感知、动力学、约束优化、策略学习或硬件反馈组织成可执行的低层动作生成机制,适合放进 System0 的快反应模块中理解。
3. 对系统集成而言,这类工作补上了 System1 技能选择与电机级控制之间的一层:它负责把上层目标变成动态可行、可恢复、可安全过滤的动作。
4. 若论文包含真实机器人、接触任务或仿真到真实迁移,核心应看它是否报告了闭环稳定性、扰动恢复、硬件延迟、失败模式和跨场景泛化。
## 问题背景

这篇论文可以放在 System0 的“具身快系统”背景下阅读:机器人需要在很短时间内根据本体状态、接触、力/触觉、局部视觉或动力学预测做出动作,而不是等待慢速规划器重新推理整个任务。adaptive-control、接触推理、灵巧操作、模仿学习、reactive-control、机器人操作、鲁棒控制 相关问题的难点通常不在“知道要做什么”,而在动作执行过程中状态会快速偏离、接触会突然改变、约束会实时收紧,低层系统必须及时修正。

### System0 关联

精读时优先看三件事:第一,论文使用哪些快速反馈量,例如关节状态、接触状态、力觉、触觉、深度或局部视觉;第二,反馈如何进入 MPC、WBC、策略网络、安全过滤器或恢复控制器;第三,输出是否直接影响机器人下一步动作,而不是只服务离线分析或高层任务规划。
## 方法详解

### System0 视角解析

从 System0 角度看,这篇论文的方法部分应当围绕 adaptive-control、接触推理、灵巧操作、模仿学习、reactive-control、机器人操作、鲁棒控制、tactile-feedback 来读。核心问题是:系统如何从快速变化的机器人状态中构造控制输入,如何处理接触、碰撞、约束、模型误差或执行延迟,以及如何把这些信息变成下一时刻的动作。

如果方法里有优化问题,重点看目标函数、约束和求解频率;如果方法里有学习策略,重点看观测空间、动作空间、训练扰动和部署时的闭环频率;如果方法里有硬件或传感器,重点看传感链路是否真的进入控制回路。这样读可以把论文和 System2 语言规划、System1 技能编排区分开来。

### 关键公式与机制

#### 公式 1: [[优化目标/约束]]

$$\mathcal{D}=\{\tau_{i}\}_{i=1}^{N}$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 2: [[动力学或策略机制]]

$$\tau_{i}=\{(o_{t}^{i},a_{t}^{i})\}_{t=1}^{T_{i}}$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 3: [[动力学或策略机制]]

$$a_{t}=\pi_{\theta}(o_{t})$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 4: [[优化目标/约束]]

$$\mathcal{L}=w_{1}\cdot\mathcal{L}_{KL}+w_{2}\cdot\mathcal{L}_{JA}+w_{3}\cdot\mathcal{L}_{tactile}+w_{4}\cdot\mathcal{L}_{arm},$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 5: [[优化目标/约束]]

$$
\mathcal{L}_{arm}=\lambda_{1}\cdot\mathcal{L}_{position}+\lambda_{2}\cdot\mathcal{L}_{rotation},
$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 6: [[动力学或策略机制]]

$$\text{HNS}=\frac{\sum_{i=1}^{N}w_{i}\cdot s_{i}}{3*\sum_{i=1}^{N}w_{i}},$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。
## 关键图表

### Figure 1: Pipeline / core system figure: An of our system hardware and teleoperation setup. (a) Our hardware system se

![Figure 1](https://arxiv.org/html/2506.15953v2/x1.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Pipeline / core system figure: An of our system hardware and teleoperation setup. (a) Our hardware system se”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 2: The neural network architecture for ViTacFormer is a conditional variational auto-enco

![Figure 2](https://arxiv.org/html/2506.15953v2/x2.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“The neural network architecture for ViTacFormer is a conditional variational auto-enco”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 3: Four short-horizon visuo-tactile manipulation tasks: Peg Insertion, Cap Twist, Vase Wi

![Figure 3](https://arxiv.org/html/2506.15953v2/img/experiments_simple.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Four short-horizon visuo-tactile manipulation tasks: Peg Insertion, Cap Twist, Vase Wi”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### 论文表格

#### Table 1: TABLE I: Success rate comparison on four short-horizon dexterous manipulation tasks. Our ViTacFormer achieves 50 %

| Task | Peg Insertion | Cap Twist | Vase Wipe | Book Flip |
| --- | --- | --- | --- | --- |
| DP | 2 / 10 2/10 | 0 / 10 0/10 | 3 / 10 3/10 | 1 / 10 1/10 |
| ACT | 4 / 10 4/10 | 4 / 10 4/10 | 3 / 10 3/10 | 2 / 10 2/10 |
| HATO | 4 / 10 4/10 | 1 / 10 1/10 | 4 / 10 4/10 | 3 / 10 3/10 |
| ACTw/T | 6 / 10 6/10 | 6 / 10 6/10 | 4 / 10 4/10 | 4 / 10 4/10 |
| Ours | 10 / 10  $\textbf{10}/10$ | 10 / 10  $\textbf{10}/10$ | 9 / 10  $\textbf{9}/10$ | 9 / 10  $\textbf{9}/10$ |

**说明**: TABLE I: Success rate comparison on four short-horizon dexterous manipulation tasks. Our ViTacFormer achieves 50 % 50\% success rates compared to the baselines.

#### Table 2: TABLE II: Human evaluation score comparison on a very long-horizon dexterous manipulation task. ViTacFormer shows promis

| Stage | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | Overall |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ACT | 2.4 | 2.5 | 1.9 | 2.0 | 0.7 | 2.2 | 1.6 | 2.8 | 2.2 | 2.2 | 0.7 | 0.61 |
| ACT w./T | 2.6 | 3.0 | 1.8 | 1.6 | 2.0 | 2.3 | 2.2 | 2.9 | 2.0 | 1.8 | 1.4 | 0.72 |
| Ours | 2.9 | 3.0 | 1.9 | 1.8 | 2.7 | 2.9 | 2.0 | 2.8 | 2.4 | 2.5 | 3.0 | 0.88 |

**说明**: TABLE II: Human evaluation score comparison on a very long-horizon dexterous manipulation task. ViTacFormer shows promising results on this long-horizon task.

#### Table 3: TABLE III: Scoring criteria for Peg Insertion.

| Stage 1: Grasp (weight 1) | Description |
| --- | --- |
| 0 | No grasp |
| 1 | Grasped but slipped or dropped |
| 2 | Poor or tilted grasp |
| 3 | Stable grasp |
| Stage 2: Insertion (weight 2) | Description |
| 0 | No insertion |
| 1 | Misaligned, dropped |
| 2 | Partial insertion |
| 3 | Fully inserted |

**说明**: TABLE III: Scoring criteria for Peg Insertion.

#### Table 4: TABLE IV: Peg Insertion: inference results across models.

| Model | Stage 1 | Stage 2 | HNS | Success Rate |
| --- | --- | --- | --- | --- |
| DP | 1.6 | 0.9 | 0.37 | 20% |
| ACT | 2.6 | 1.1 | 0.53 | 40% |
| HATO | 2.4 | 1.1 | 0.51 | 40% |
| ACT w/T | 2.6 | 1.8 | 0.68 | 60% |
| Ours w/o CrossAttention | 2.9 | 2.2 | 0.81 | 90% |
| Ours w/o AutoRegressive | 3.0 | 2.1 | 0.80 | 70% |
| Ours w/o Two-Stage | 2.8 | 2.0 | 0.75 | 70% |
| Ours | 3.0 | 2.7 | 0.93 | 100% |

**说明**: TABLE IV: Peg Insertion: inference results across models.

#### Table 5: TABLE V: Scoring criteria for Cap Twist.

| Stage 1: Rotate (weight 2) | Description |
| --- | --- |
| 0 | No contact with the cap |
| 1 | Rotated 0–50° |
| 2 | Rotated 50–100°, or |
| 3 | Fully unscrewed, cap held securely |
| Stage 2: Place (weight 2) | Description |
| 0 | Dropped immediately or stuck on bottle |
| 1 | Released before full separation |
| 2 | Partially placed or fell off |
| 3 | Stably placed on the table |

**说明**: TABLE V: Scoring criteria for Cap Twist.

#### Table 6: TABLE VI: Cap Twist: inference results across models.

| Model | Stage 1 | Stage 2 | HNS | Success Rate |
| --- | --- | --- | --- | --- |
| DP | 1.1 | 0.3 | 0.23 | 0% |
| ACT | 2.4 | 1.1 | 0.58 | 40% |
| HATO | 1.8 | 0.5 | 0.38 | 10% |
| ACT w/T | 2.6 | 1.8 | 0.73 | 60% |
| Ours w/o CrossAttention | 2.9 | 2.2 | 0.85 | 70% |
| Ours w/o AutoRegressive | 3.0 | 2.3 | 0.88 | 70% |
| Ours w/o Two-Stage | 2.7 | 2.0 | 0.78 | 60% |
| Ours | 3.0 | 2.9 | 0.98 | 100% |

**说明**: TABLE VI: Cap Twist: inference results across models.

#### Table 7: TABLE VII: Scoring criteria for Vase Wipe.

| Stage 1: Pick (weight 1) | Description |
| --- | --- |
| 0 | Failed to grasp the sponge |
| 1 | Grasped only a corner of sponge |
| 2 | Unstable grasp with partial control |
| 3 | Firm 3-finger grasp with full control |
| Stage 2: Wipe (weight 2) | Description |
| 0 | No contact with the ink mark |
| 1 | Wiped less than 50% |
| 2 | Wiped 50–90%, some ink remains |
| 3 | Fully wiped the ink area clean |

**说明**: TABLE VII: Scoring criteria for Vase Wipe.

#### Table 8: TABLE VIII: Vase Wipe: inference results across models.

| Model | Stage 1 | Stage 2 | HNS | Success Rate |
| --- | --- | --- | --- | --- |
| DP | 1.8 | 1.3 | 0.49 | 30% |
| ACT | 2.0 | 1.5 | 0.56 | 30% |
| HATO | 2.5 | 1.7 | 0.65 | 40% |
| ACT w/T | 3.0 | 1.9 | 0.75 | 40% |
| Ours w/o CrossAttention | 3.0 | 2.5 | 0.89 | 70% |
| Ours w/o AutoRegressive | 3.0 | 2.5 | 0.89 | 60% |
| Ours w/o Two-Stage | 2.1 | 1.6 | 0.59 | 40% |
| Ours | 3.0 | 2.9 | 0.98 | 90% |

**说明**: TABLE VIII: Vase Wipe: inference results across models.

#### Table 9: TABLE IX: Scoring criteria for Book Flip.

| Stage 1: Flip (weight 2) | Description |
| --- | --- |
| 0 | No contact with the page |
| 1 | Touched but failed to lift / flipped multiple pages |
| 2 | Lifted halfway but stopped |
| 3 | Fully flipped one page |
| Stage 2: Press (weight 2) | Description |
| 0 | No contact with the page |
| 1 | Insufficient force, page rebounds |
| 2 | Pressed down, but misaligned |
| 3 | Fully and correctly pressed the page down |

**说明**: TABLE IX: Scoring criteria for Book Flip.

#### Table 10: TABLE X: Book Flip: inference results across models.

| Model | Stage 1 | Stage 2 | HNS | Success Rate |
| --- | --- | --- | --- | --- |
| DP | 1.5 | 0.5 | 0.35 | 10% |
| ACT | 1.9 | 0.7 | 0.43 | 20% |
| HATO | 2.0 | 0.6 | 0.43 | 30% |
| ACT w/T | 2.3 | 0.9 | 0.53 | 40% |
| Ours w/o CrossAttention | 2.7 | 2.1 | 0.80 | 70% |
| Ours w/o AutoRegressive | 2.7 | 1.9 | 0.77 | 70% |
| Ours w/o Two-Stage | 2.7 | 1.2 | 0.65 | 60% |
| Ours | 3.0 | 2.6 | 0.93 | 90% |

**说明**: TABLE X: Book Flip: inference results across models.

#### Table 11: TABLE XI: Scoring criteria for the long-horizon hamburger task.

| Stage | Action | Weight | Score Description |
| --- | --- | --- | --- |
| 1 | Flip sign (start) | 2 | 0: miss/fail; 1–2: partial (0–180°); 3: clean flip |
| 2 | Grab spatula | 2 | 0: miss; 1–2: unstable grasp; 3: secure grasp |
| 3 | Lift meat patty | 1 | 0: failed; 1–2: partial lift; 3: stable lift |
| 4 | Place meat patty | 1 | 0: miss; 1–2: partial/inaccurate; 3: centered |
| 5 | Grasp lettuce | 2 | 0: miss; 1–2: loose grasp; 3: stable placement |
| 6 | Lift top bread | 2 | 0: failed; 1–2: unstable or too forceful; 3: correct |
| 7 | Place top bread | 1 | 0: miss; 1–2: inaccurate; 3: clean stack |
| 8 | Lift hamburger | 1 | 0: failed; 1–2: unstable; 3: correct lift |
| 9 | Place on plate | 1 | 0: miss; 1–2: off-center; 3: perfect placement |
| 10 | Return spatula | 1 | 0: drop/fail; 1–2: misaligned; 3: accurate return |
| 11 | Flip sign (end) | 2 | 0: fail; 1–2: partial rotation; 3: clean close flip |

**说明**: TABLE XI: Scoring criteria for the long-horizon hamburger task.
## 实验解读

- 评价重点:围绕 adaptive-control、接触推理、灵巧操作、模仿学习、reactive-control,优先看真实机器人结果、扰动恢复、接触稳定性、控制误差、成功率、实时性和 sim-to-real 差距。
- System0 视角:实验结论不只看最终成功率,还要看闭环过程中是否能处理状态估计误差、执行延迟、碰撞/接触突变和外部扰动。
- 与上层系统的关系:如果方法能在快闭环内稳定工作,它可以作为 System1 技能或 System2 规划的底层执行保障;如果只在离线或慢速设定下成立,则还不能直接作为身体级快闭环模块。
## 批判性思考

### 优点

- 论文把 adaptive-control、接触推理、灵巧操作、模仿学习、reactive-control 放到机器人身体级闭环里讨论,和只做高层语义推理的工作相比,更接近真实机器人部署时会遇到的状态、动力学和安全约束。
- 若论文同时给出公式、系统结构图、硬件实验或消融结果,这些证据可以帮助判断方法是否真的能支撑毫秒到百毫秒级反应。

### 局限

- 需要重点核查控制频率、推理延迟、传感延迟、失败恢复、硬件磨损、未见场景泛化和安全边界;这些因素决定它能否成为可靠的 System0 模块。
- 对学习型方法,还要区分训练环境中的鲁棒性和真实部署中的鲁棒性,尤其是接触丰富任务、动态运动任务和人形机器人任务。
## 论文定位

- 研究对象:ViTacFormer: Learning Cross-Modal Representation for Visuo-Tactile Dexterous Manipulation。
- 关键词:adaptive-control、接触推理、灵巧操作、模仿学习、reactive-control、机器人操作、鲁棒控制、tactile-feedback、visual-tactile、visuomotor-policy。
- System0 关联:关注机器人身体级快反应链路中“状态/接触/局部感知 -> 约束/优化/策略 -> 动作”的短闭环,不把问题停留在语言规划或任务分解层。

## 速查卡片

> [!summary] ViTacFormer
> - **论文**: https://www.roboticsproceedings.org/rss22/p129.pdf
> - **arXiv**: http://arxiv.org/abs/2506.15953v2
> - **arXiv HTML**: https://arxiv.org/html/2506.15953v2
