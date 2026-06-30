---
title: "Human2LocoMan: Learning Versatile Quadrupedal Manipulation with Human Pretraining"
method_name: "Human2LocoMan"
authors: ["Yaru Niu"]
year: 2025
venue: "RSS"
tags: ["robot-manipulation", "legged-locomotion", "imitation-learning", "robot-generalization", "loco-manipulation"]
image_source: "online"
arxiv_html: "https://arxiv.org/html/2506.16475v2"
---
# Human2LocoMan
## 一句话总结

> Human2LocoMan: Learning Versatile Quadrupedal Manipulation with Human Pretraining 主要落在 [[bimanual-manipulation]]、[[co-training]]、[[cross-embodiment]]、[[模仿学习]] 这一类问题上;从 System0 视角看,它的价值在于把身体状态、接触变化、局部感知或动力学约束接到机器人低层闭环里,让动作生成更稳定、安全、实时。
## 核心贡献

1. 围绕 **Human2LocoMan: Learning Versatile Quadrupedal Manipulation with Human Pretraining** 建立了一个与 bimanual-manipulation、co-training、cross-embodiment、模仿学习、足式运动、移动操作 相关的机器人问题设定,重点不是高层语义规划,而是身体级闭环中的状态变化、接触变化或控制约束。
2. 方法层面的关键价值在于把感知、动力学、约束优化、策略学习或硬件反馈组织成可执行的低层动作生成机制,适合放进 System0 的快反应模块中理解。
3. 对系统集成而言,这类工作补上了 System1 技能选择与电机级控制之间的一层:它负责把上层目标变成动态可行、可恢复、可安全过滤的动作。
4. 若论文包含真实机器人、接触任务或仿真到真实迁移,核心应看它是否报告了闭环稳定性、扰动恢复、硬件延迟、失败模式和跨场景泛化。
## 问题背景

这篇论文可以放在 System0 的“具身快系统”背景下阅读:机器人需要在很短时间内根据本体状态、接触、力/触觉、局部视觉或动力学预测做出动作,而不是等待慢速规划器重新推理整个任务。bimanual-manipulation、co-training、cross-embodiment、模仿学习、足式运动、移动操作、mobile-manipulation 相关问题的难点通常不在“知道要做什么”,而在动作执行过程中状态会快速偏离、接触会突然改变、约束会实时收紧,低层系统必须及时修正。

### System0 关联

精读时优先看三件事:第一,论文使用哪些快速反馈量,例如关节状态、接触状态、力觉、触觉、深度或局部视觉;第二,反馈如何进入 MPC、WBC、策略网络、安全过滤器或恢复控制器;第三,输出是否直接影响机器人下一步动作,而不是只服务离线分析或高层任务规划。
## 方法详解

### System0 视角解析

从 System0 角度看,这篇论文的方法部分应当围绕 bimanual-manipulation、co-training、cross-embodiment、模仿学习、足式运动、移动操作、mobile-manipulation、quadruped 来读。核心问题是:系统如何从快速变化的机器人状态中构造控制输入,如何处理接触、碰撞、约束、模型误差或执行延迟,以及如何把这些信息变成下一时刻的动作。

如果方法里有优化问题,重点看目标函数、约束和求解频率;如果方法里有学习策略,重点看观测空间、动作空间、训练扰动和部署时的闭环频率;如果方法里有硬件或传感器,重点看传感链路是否真的进入控制回路。这样读可以把论文和 System2 语言规划、System1 技能编排区分开来。

### 关键公式与机制

#### 公式 1: [[优化目标/约束]]

$$
\mathcal{L}_{\text{human}}(B)=\sum_{i}\mathcal{L}_{\text{human},m_{i}}(B)
$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 2: [[优化目标/约束]]

$$
\mathcal{L}_{\text{LocoMan}}(B)=\sum_{i}\mathcal{L}_{\text{LocoMan},m_{i}}(B)
$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 3: [[优化目标/约束]]

$$
\mathcal{L}_{e,m_{i}}(B_{e})=\frac{1}{n}\sum_{j=1}^{n}\left[\frac{1}{h}\sum_{l=1}^{h}\ell_{1}\left(\boldsymbol{a}_{j,l}\left[m_{i}\right],\hat{\boldsymbol{a}}_{j,l}\left[m_{i}\right]\right)\right],
$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 4: [[动力学或策略机制]]

$$\boldsymbol{p}_{0}=(\boldsymbol{x}^{\text{torso}}_{\text{uni, 0}},\boldsymbol{R}^{\text{torso}}_{\text{uni, 0}},\boldsymbol{x}^{\text{r-eef}}_{\text{uni, 0}},\boldsymbol{R}^{\text{r-eef}}_{\text{uni, 0}},\boldsymbol{x}^{\text{l-eef}}_{\text{uni, 0}},\boldsymbol{R}^{\text{l-eef}}_{\text{uni, 0}},\boldsymbol{\theta}^{\text{gripper}}_{0})$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 5: [[动力学或策略机制]]

$$\boldsymbol{p}^{\text{t}}_{t}=(\boldsymbol{x}^{\text{torso,t}}_{\text{uni},t},\boldsymbol{R}^{\text{torso,t}}_{\text{uni},t},\boldsymbol{x}^{\text{r-eef,t}}_{\text{uni},t},\boldsymbol{R}^{\text{r-eef,t}}_{\text{uni},t},\boldsymbol{x}^{\text{l-eef,t}}_{\text{uni},t},\boldsymbol{R}^{\text{l-eef,t}}_{\text{uni},t},\boldsymbol{\theta}^{\text{gripper,t}}_{t})$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 6: [[动力学或策略机制]]

$$=\frac{\boldsymbol{\theta}^{\text{gripper}}_{\text{max}}-\boldsymbol{\theta}^{\text{gripper}}_{\text{min}}}{d^{\text{tip}}_{\text{max}}}\circ\boldsymbol{d}^{\text{tip}}_{t}+\boldsymbol{\theta}^{\text{gripper}}_{\text{min}}$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 7: [[优化目标/约束]]

$$
\mathcal{L}_{e}(\theta)=\sum_{i=1}^{k}\mathcal{L}_{e,m_{i}}(\theta),
$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 8: [[动力学或策略机制]]

$$\hat{\boldsymbol{a}}_{j,l}\left[m_{i}\right]=\left[\pi_{\theta}(\boldsymbol{o}_{j})\right]_{l}\left[m_{i}\right]$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 9: [[动力学或策略机制]]

$$(\boldsymbol{x}^{\cdot}_{\text{uni}},\boldsymbol{R}^{\cdot}_{\text{uni}})=(\boldsymbol{R}^{\text{vr}}_{\text{uni}}\boldsymbol{x}^{\cdot}_{\text{vr}},\boldsymbol{R}^{\text{vr}}_{\text{uni}}\boldsymbol{R}^{\cdot}_{\text{vr}})$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 10: [[动力学或策略机制]]

$$:=[\boldsymbol{x}^{\text{r-eef, t}}_{\text{uni},t},\boldsymbol{R}^{\text{r-eef, t}}_{\text{uni},t},\boldsymbol{x}^{\text{l-eef, t}}_{\text{uni},t},\boldsymbol{R}^{\text{l-eef,t}}_{\text{uni},t}],$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。
## 关键图表

### Figure 1: Human2LocoMan framework. Our system uses an XR headset for data collection, capturing

![Figure 1](https://arxiv.org/html/2506.16475v2/x1.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Human2LocoMan framework. Our system uses an XR headset for data collection, capturing”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 2: Modularized Cross-embodiment Transformer (MXT) architecture. The inputs are organized

![Figure 2](https://arxiv.org/html/2506.16475v2/x2.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Modularized Cross-embodiment Transformer (MXT) architecture. The inputs are organized”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 3: Best validation loss of our method and HIT on all our tasks. MXT-Pretrained: MXT pre

![Figure 3](https://arxiv.org/html/2506.16475v2/x6.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Best validation loss of our method and HIT on all our tasks. MXT-Pretrained: MXT pre”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### 论文表格

#### Table 1: TABLE I: Human2LocoMan embodiments (R = = = Right, L = = = Left).

| Embodiments | Head | Wrist | Body | R-EEF | L-EEF | Body | R-EEF | L-EEF | R-Grasp | L-Grasp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Images | Image | Priop. | Priop. | Priop. | Pose | Pose | Pose | Action | Action | |
| Human-Unimanual (R) | ✓ |  \times  | ✓ | ✓ |  \times  | ✓ | ✓ |  \times  | ✓ |  \times  |
| Human-Unimanual (L) | ✓ |  \times  | ✓ |  \times  | ✓ | ✓ |  \times  | ✓ |  \times  | ✓ |
| Human-Bimanual | ✓ |  \times  | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| LocoMan-Unimanual (R) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |  \times  | ✓ |  \times  |
| LocoMan-Unimanual (L) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |  \times  | ✓ |  \times  | ✓ |
| LocoMan-Bimanual | ✓ |  \times  | ✓ | ✓ | ✓ |  \times  | ✓ | ✓ | ✓ | ✓ |

**说明**: TABLE I: Human2LocoMan embodiments (R = = = Right, L = = = Left).

#### Table 2: TABLE II: Result Summary. We report success rate (SR) ↑ ↑ \uparrow ↑ in % and task score (TS) ↑ ↑ \uparrow ↑ for each t

**说明**: TABLE II: Result Summary. We report success rate (SR) ↑ ↑ \uparrow ↑ in % and task score (TS) ↑ ↑ \uparrow ↑ for each task. We highlight the best performance in bold and the second best in. ID results are based on 24 trials, and OOD results on 12 trials.

#### Table 3: TABLE III: Records of data collection for different tasks.

| Task | # human traj. | human time (min) | # robot traj. | robot time (min) |
| --- | --- | --- | --- | --- |
| TC-Uni | 300 | 25 | 150 | 15 |
| TC-Bi | 315 | 22 | 70 | 7 |
| SO-Uni | 240 | 34 | 90 | 23 |
| SO-Bi | 200 | 20 | 92 | 12 |
| Scoop-Uni | 340 | 96 | 66 | 22 |
| Pour-Bi | 210 | 35 | 64 | 22 |

**说明**: TABLE III: Records of data collection for different tasks.

#### Table 4: TABLE IV: MXT trunk and training hyperparameters

| Hyperparameters | Value |
| --- | --- |
| optimizer | AdamW |
| learning rate | 5e-5 (finetuning/from scratch) |
| 1e-4 (pretraining) | |
| scheduler | constant |
| weight decay | 1e-4 |
| trunk encoder layers | 4 |
| trunk decoder layers | 4 |
| hidden dim. | 128 |
| Transformer feedforward dim. | 256 |
| #attention heads | 16 |

**说明**: TABLE IV: MXT trunk and training hyperparameters

#### Table 5: TABLE V: MXT tokenizer hyperparameters

| Modality | Input dimensions | #tokens | MLP widths |
| --- | --- | --- | --- |
| main image | (3, 480 1280) | 16 | N/A |
| wrist image | (3, 480, 640) | 8 | |
| body pose | (6,) | 4 | [128, 128] |
| EEF pose | (12,) | 4 | |
| EEF-to-body pose | (12,) | 4 | |
| gripper angles | (2,) | 4 | |

**说明**: TABLE V: MXT tokenizer hyperparameters

#### Table 6: TABLE VI: MXT detokenizer hyperparameters

| Modalities | Output dimensions | #tokens |
| --- | --- | --- |
| body pose | (6,) | 6 |
| EEF pose | (12,) | 6 |
| gripper angle | (2,) | 6 |

**说明**: TABLE VI: MXT detokenizer hyperparameters

#### Table 7: TABLE VII: HIT hyperparameters

| Hyperparameters | Value |
| --- | --- |
| optimizer | AdamW |
| learning rate | 2e-5 |
| scheduler | constant |
| weight decay | 1e-4 |
| encoder layers | 4 |
| decoder layers | 4 |
| hidden dim | 128 |
| #attention heads | 8 |
| feature loss weight | 0.001 |
| image backbone | ResNet18 |

**说明**: TABLE VII: HIT hyperparameters

#### Table 8: TABLE VIII: HPT hyperparameters

| Hyperparameters | Value |
| --- | --- |
| optimizer | AdamW |
| learning rate | 5e-5 (finetuning/from scratch) |
| 1e-4 (pretraining) | |
| scheduler | constant |
| weight decay | 1e-4 |
| trunk | |
| #Transformer blocks | 16 |
| hidden dim | 128 |
| feedforward dim | 256 |
| #attention heads | 8 |
| action head | |
| #attention heads | 8 |
| head dim | 64 |
| dropout | 0.1 |
| output dim | 20 |
| image stem | |
| encoder | ResNet18 |
| MLP widths | [128] |
| #tokens | 16 |
| state stem | |
| MLP widths | [128] |
| #tokens | 16 |

**说明**: TABLE VIII: HPT hyperparameters

#### Table 9: TABLE IX: Global training parameters for each task

| Task | Mode | Batch Size | Training Steps | Chunk Size |
| --- | --- | --- | --- | --- |
| Toy Collection | Unimanual | 16 | 60000 | 60 |
| Bimanual | 16 | 60000 | 60 | |
| Shoe Organization | Unimanual | 24 | 80000 | 180 |
| Bimanual | 24 | 100000 | 120 | |
| Scooping | Unimanual | 24 | 100000 | 120 |
| Pouring | Bimanual | 24 | 80000 | 180 |

**说明**: TABLE IX: Global training parameters for each task
## 实验解读

- 评价重点:围绕 bimanual-manipulation、co-training、cross-embodiment、模仿学习、足式运动,优先看真实机器人结果、扰动恢复、接触稳定性、控制误差、成功率、实时性和 sim-to-real 差距。
- System0 视角:实验结论不只看最终成功率,还要看闭环过程中是否能处理状态估计误差、执行延迟、碰撞/接触突变和外部扰动。
- 与上层系统的关系:如果方法能在快闭环内稳定工作,它可以作为 System1 技能或 System2 规划的底层执行保障;如果只在离线或慢速设定下成立,则还不能直接作为身体级快闭环模块。
## 批判性思考

### 优点

- 论文把 bimanual-manipulation、co-training、cross-embodiment、模仿学习、足式运动 放到机器人身体级闭环里讨论,和只做高层语义推理的工作相比,更接近真实机器人部署时会遇到的状态、动力学和安全约束。
- 若论文同时给出公式、系统结构图、硬件实验或消融结果,这些证据可以帮助判断方法是否真的能支撑毫秒到百毫秒级反应。

### 局限

- 需要重点核查控制频率、推理延迟、传感延迟、失败恢复、硬件磨损、未见场景泛化和安全边界;这些因素决定它能否成为可靠的 System0 模块。
- 对学习型方法,还要区分训练环境中的鲁棒性和真实部署中的鲁棒性,尤其是接触丰富任务、动态运动任务和人形机器人任务。
## 论文定位

- 研究对象:Human2LocoMan: Learning Versatile Quadrupedal Manipulation with Human Pretraining。
- 关键词:bimanual-manipulation、co-training、cross-embodiment、模仿学习、足式运动、移动操作、mobile-manipulation、quadruped、机器人操作、scalable-robot-learning。
- System0 关联:关注机器人身体级快反应链路中“状态/接触/局部感知 -> 约束/优化/策略 -> 动作”的短闭环,不把问题停留在语言规划或任务分解层。

## 速查卡片

> [!summary] Human2LocoMan
> - **论文**: https://www.roboticsproceedings.org/rss21/p122.pdf
> - **arXiv**: http://arxiv.org/abs/2506.16475v2
> - **arXiv HTML**: https://arxiv.org/html/2506.16475v2
