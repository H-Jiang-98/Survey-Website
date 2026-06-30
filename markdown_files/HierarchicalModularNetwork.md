---
title: "Hierarchical and Modular Network on Non-prehensile Manipulation in General Environments"
method_name: "Hierarchical Modular Network"
authors: ["Yoonyoung Cho"]
year: 2025
venue: "RSS"
tags: ["contact-reasoning", "robot-manipulation", "reinforcement-learning", "adaptive-control", "contact-rich-manipulation", "robot-generalization"]
image_source: "online"
arxiv_html: "https://arxiv.org/html/2502.20843v2"
---
# Hierarchical Modular Network
## 一句话总结

> Hierarchical and Modular Network on Non-prehensile Manipulation in General Environments 主要落在 [[adaptive-control]]、[[接触推理]]、[[grasping]]、[[non-prehensile-manipulation]] 这一类问题上;从 System0 视角看,它的价值在于把身体状态、接触变化、局部感知或动力学约束接到机器人低层闭环里,让动作生成更稳定、安全、实时。
## 核心贡献

1. 围绕 **Hierarchical and Modular Network on Non-prehensile Manipulation in General Environments** 建立了一个与 adaptive-control、接触推理、grasping、non-prehensile-manipulation、强化学习、robot-generalization 相关的机器人问题设定,重点不是高层语义规划,而是身体级闭环中的状态变化、接触变化或控制约束。
2. 方法层面的关键价值在于把感知、动力学、约束优化、策略学习或硬件反馈组织成可执行的低层动作生成机制,适合放进 System0 的快反应模块中理解。
3. 对系统集成而言,这类工作补上了 System1 技能选择与电机级控制之间的一层:它负责把上层目标变成动态可行、可恢复、可安全过滤的动作。
4. 若论文包含真实机器人、接触任务或仿真到真实迁移,核心应看它是否报告了闭环稳定性、扰动恢复、硬件延迟、失败模式和跨场景泛化。
## 问题背景

这篇论文可以放在 System0 的“具身快系统”背景下阅读:机器人需要在很短时间内根据本体状态、接触、力/触觉、局部视觉或动力学预测做出动作,而不是等待慢速规划器重新推理整个任务。adaptive-control、接触推理、grasping、non-prehensile-manipulation、强化学习、robot-generalization、机器人操作 相关问题的难点通常不在“知道要做什么”,而在动作执行过程中状态会快速偏离、接触会突然改变、约束会实时收紧,低层系统必须及时修正。

### System0 关联

精读时优先看三件事:第一,论文使用哪些快速反馈量,例如关节状态、接触状态、力觉、触觉、深度或局部视觉;第二,反馈如何进入 MPC、WBC、策略网络、安全过滤器或恢复控制器;第三,输出是否直接影响机器人下一步动作,而不是只服务离线分析或高层任务规划。
## 方法详解

### System0 视角解析

从 System0 角度看,这篇论文的方法部分应当围绕 adaptive-control、接触推理、grasping、non-prehensile-manipulation、强化学习、robot-generalization、机器人操作、zero-shot 来读。核心问题是:系统如何从快速变化的机器人状态中构造控制输入,如何处理接触、碰撞、约束、模型误差或执行延迟,以及如何把这些信息变成下一时刻的动作。

如果方法里有优化问题,重点看目标函数、约束和求解频率;如果方法里有学习策略,重点看观测空间、动作空间、训练扰动和部署时的闭环频率;如果方法里有硬件或传感器,重点看传感链路是否真的进入控制回路。这样读可以把论文和 System2 语言规划、System1 技能编排区分开来。

### 关键公式与机制

#### 公式 1: [[动力学或策略机制]]

$$\theta=\sum_{i=1}^{M}w_{i}\theta_{i}$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 2: [[动力学或策略机制]]

$$\tau=k_{p}\Delta q-k_{d}\dot{q}$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 3: [[动力学或策略机制]]

$$r=r_{s}+\lambda_{r}r_{r}+\lambda_{c}r_{c}$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 4: [[动力学或策略机制]]

$$w=\{w_{i,j}\}_{j=1}^{L}\in\mathbb{R}^{L\times M}$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 5: [[动力学或策略机制]]

$$g=\{{g_{j}}\}_{j=1}^{L}\in\mathbb{R}^{L\times D_{j}}$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 6: [[动力学或策略机制]]

$$\theta=\{(\sum_{i=1}^{M}w_{i,j}\theta_{i,j})\odot g_{j}\}_{j=1}^{L}$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 7: [[动力学或策略机制]]

$$|| $\sum_{i=1}^{7}\tau_{i}\dot{q}_{i}$ ||_{2}$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 8: [[动力学或策略机制]]

$$\theta^{(a)}_{j}=\sum_{i=1}^{M}w_{i,j}^{(a)}\theta_{i,j}\odot g^{(a)}_{j}\}_{j=1}^{L}$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 9: [[动力学或策略机制]]

$$\theta^{(v)}_{j}=\sum_{i=1}^{M}w_{i,j}^{(v)}\theta_{i,j}\odot g^{(v)}_{j}\}_{j=1}^{L}$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 10: [[动力学或策略机制]]

$$R_{t}=\mathbb{E}_{a_{t}\sim\pi(\cdot|s_{t})}[\sum\gamma^{t}r(s_{t},a_{t},s_{t+1})]$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。
## 关键图表

### Figure 1: Pipeline / core system figure: Illustration of computational structure for biological motor control (left) and our a

![Figure 1](https://arxiv.org/html/2502.20843v2/extracted/6553794/fig/02intro/bio-figure-v4.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Pipeline / core system figure: Illustration of computational structure for biological motor control (left) and our a”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 2: Overall method. Our framework consists of four main components: a modular po

![Figure 2](https://arxiv.org/html/2502.20843v2/x5.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Overall method. Our framework consists of four main components: a modular po”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 3: Illustration of student policy architecture. As in the teacher network, the student

![Figure 3](https://arxiv.org/html/2502.20843v2/x18.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Illustration of student policy architecture. As in the teacher network, the student”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### 论文表格

#### Table 1: TABLE I: Comparison of generalization capabilities for non-prehensile manipulation.

| | Object generalization | General action space | Environment generalization |
| --- | --- | --- | --- |
| HACMAN [122 ] | O | X | X |
| CORN [17 ] | O | O | X |
| Wu et al. [109 ] | △ △ \triangle △ | △ △ \triangle △ | △ △ \triangle △ |
| Ours | O | O | O |

**说明**: TABLE I: Comparison of generalization capabilities for non-prehensile manipulation.

#### Table 2: TABLE II: Comparison between baselines regarding architecture and representation.

| Model Name | Model Architecture | Representation |
| --- | --- | --- |
| UniCORN-HAMnet (ours) | HAMnet | UniCORN |
| UniCORN-Hyper | Hypernetwork | UniCORN |
| UniCORN-SM | Soft-Modularization [112 ] | UniCORN |
| UniCORN-Transformer | Transformer | UniCORN |
| UniCORN-Mono | MLP | UniCORN |
| PointGPT-HAMnet | HAMnet | PointGPT [11 ] |
| E2E-HAMnet | HAMnet | End-to-end |

**说明**: TABLE II: Comparison between baselines regarding architecture and representation.

#### Table 3: TABLE III: Results on 9 unseen real-world domains.

| Domain | Object | Success rate | Domain | Object | Success rate |
| --- | --- | --- | --- | --- | --- |
| Cabinet | Bulldozer | 4/5 | Top of cabinet | Bulldozer | 3/5 |
| Heart-Box | 3/5 | Crab | 4/5 | | |
| Sink | Bulldozer | 5/5 | Basket | Bulldozer | 3/5 |
| Angled Cup | 4/5 | Heart-Box | 5/5 | | |
| Drawer | Bulldozer | 4/5 | Grill | Bulldozer | 5/5 |
| Pencil case | 3/5 | Dino | 4/5 | | |
| Circular bin | Bulldozer | 4/5 | Flat | Bulldozer | 5/5 |
| Pineapple | 3/5 | Nutella | 3/5 | | |
| Suitcase | Bulldozer | 4/5 | Total | | 78.9% |
| Candy Jar | 5/5 | | | | |

**说明**: TABLE III: Results on 9 unseen real-world domains.

#### Table 4: TABLE IV: Pretraining pipeline hyperparameters.

| Hyperparameter | Value |
| --- | --- |
| Batch size | 1024 |
| Optimizer | SAM [32 ] |
| Learning rate schedule | cosine |
| Base learning rate | 0.0002 |
| Min. learning rate | 1e-6 |
| Max. gradient norm | 1,000 |
| Weight decay | 0.001 |
| Rotational augmentation | $\pi -, + π π +\pi +)$ |
| Translational augmentation | (-0.1, +0.1) |
| Scale augmentation | (e - 1, e + 1) e 1 e 1 (e^{-1},e^{+1}) (- 1, + 1) |
| Noise augmentation | 0.01 |
| Positive patch fraction | 0.5 |
| Decoder size | (128, 128) |

**说明**: TABLE IV: Pretraining pipeline hyperparameters.

#### Table 5: TABLE V: Scene parameters and their ranges in our procedural generation pipeline. All angles are in degrees, and dimens

| Parameter | Value |
| --- | --- |
| table_dim.x | U(0.255, 0.51) |
| table_dim.y | U(0.325, 0.65) |
| table_dim.z | U(0.2, 0.4) |
| table_pos.x | U(0.0, 0.1) |
| table_pos.y | U(-0.15, 0.15) |
| table_pos.z | U(0.1, 0.8) |
| ramp_angle | U(0.0, 30)  \times  4 |
| plate_height | U(0.0, 0.15)  \times  6 |
| ceiling_height | U(0.3, 0.5) |
| gap_height | U(0.03, 0.05) |
| ceil_mask | B(0.5) |
| wall_mask | B(0.5)  \times  4 |
| table_friction | U(0.2, 0.6) |

**说明**: TABLE V: Scene parameters and their ranges in our procedural generation pipeline. All angles are in degrees, and dimensions are in meters. U: uniform distribution; B: Bernoulli distribution.

#### Table 6: TABLE VI: Sim2real hyperparameters.

| Hyperparameter | Value |
| --- | --- |
|  ∗  \xi^{*} ∗ (large joint) | 0.16 |
| $\text{max}} max (large joint)$ | 0.26 |
|  ∗  \xi^{*} ∗ (small joint) | 0.08 |
| $\text{max}} max (small joint)$ | 0.21 |
| N s N s N_{s} | 1024 |
| N t N t N_{t} | 2e6 |
|  x x \epsilon_{x} | 0.12 |
| x max x max \epsilon x $\text{max}} max$ | 0.24 |
|   \alpha | 0.8 |

**说明**: TABLE VI: Sim2real hyperparameters.

#### Table 7: TABLE VII: Summary of our MDP, in terms of state, action, and reward components. Each component is denoted by its name,

| State Component | Symbol | Dimension | Description |
| --- | --- | --- | --- |
| Object state † Object state †  $\text{Object state}^{{\dagger}} Object state †$ | x t o x t o x_{t}^{o} | R 15 R 15  $\mathbb{R}^{15} 15$ | Object pose and velocity |
| Robot state | x t q x t q x_{t}^{q} | R 14 R 14  $\mathbb{R}^{14} 14$ | Joint positions and velocities |
| End-effector pose | x t E E x t E E x_{t}^{EE} | R 9 R 9  $\mathbb{R}^{9} 9$ | Pose of the robot’s end-effector |
| Physics parameters † Physics parameters †  $\text{Physics parameters}^{{\dagger}} Physics parameters †$ |   \nu | R 6 R 6  $\mathbb{R}^{6} 6$ | Mass, friction, restitution of object and friction of robot and environment |
| Object geometry | G o G o G_{o} | $\mathbb{R}^{512\times 3} 512 × 3$ | Surface-sampled point cloud of the object |
| Environment geometry | G e G e G_{e} | $\mathbb{R}^{512\times 3} 512 × 3$ | Surface-sampled point cloud of the environment |
| Goal pose | T g T g T_{g} | R 9 R 9  $\mathbb{R}^{9} 9$ | Target pose for the object, relative to current pose |
| Action Component | Symbol | Dimension | Description |
| Joint-space subgoal residuals |  q  q \Delta q | R 7 R 7  $\mathbb{R}^{7} 7$ | Desired changes in joint positions |
| Proportional gains | k p k p k_{p} | R 7 R 7  $\mathbb{R}^{7} 7$ | Joint-space proportional gains |
| Damping factors |   \rho | R 7 R 7  $\mathbb{R}^{7} 7$ | Factors for computing damping terms |
| Reward Component | Symbol | Dimension | Description |
| Task success reward | r s r s r_{ $\text{s}} s$ | R 1 R 1  $\mathbb{R}^{1} 1$ | Reward for task success |
| Goal-reaching reward | r r r r r_{ $\text{r}} r$ | R 1 R 1  $\mathbb{R}^{1} 1$ | Reward for moving object towards goal |
| Contact-inducing reward | r c r c r_{ $\text{c}} c$ | R 1 R 1  $\mathbb{R}^{1} 1$ | Reward for moving gripper towards object |

**说明**: TABLE VII: Summary of our MDP, in terms of state, action, and reward components. Each component is denoted by its name, shorthand symbol, dimensionality and a brief description. † † ${\dagger}$ †: only used in simulation.

#### Table 8: TABLE VIII: Hyperparmeters for the reward terms.

| Parameter | Value | Description |
| --- | --- | --- |
|  r  r \lambda_{r} | 0.15 | Goal-reaching reward coefficient |
|  c  c \lambda_{c} | 0.03 | Contact-inducing reward coefficient |
| c g c g c_{g} | 3.0 | Scale for goal-reaching distance potential |
| c r c r c_{r} | 3.0 | Scale for contact-inducing distance potential |

**说明**: TABLE VIII: Hyperparmeters for the reward terms.

#### Table 9: TABLE IX: Network Hyperparameters.

| Hyperparameter | Value | Hyperparameter | Value | Hyperparameter | Value |
| --- | --- | --- | --- | --- | --- |
| Num. points | 512 | Num. encoder layers | 4 | Modulation | |
| Network | | | | | |
| Num. patches | 16 | Num. self-attn heads | 4 | Actor | MLP (256, 128, 128, 64) |
| Patch size | 32 | Cross-attn embedding dim. | 64 | Critic | MLP (256, 128, 128, 64) |
| Embedding dim. | 128 | Num. cross-attn heads | | | |
| (object / others) | | | | | |
| Num. modules | | | | | |

**说明**: TABLE IX: Network Hyperparameters.

#### Table 10: TABLE X: PPO Hyperparameters.

| Hyperparameter | Value | Hyperparameter | Value |
| --- | --- | --- | --- |
| Max Num. epoch | 8 | Base learning rate | 0.0003 |
| Early-stopping KL target | 0.024 | Adaptive-LR KL target | 0.016 |
| Entropy regularization | 0 | Learning rate schedule | KL-adaptive |
| Initial log std. | -0.4 | log std. decay factor | -0.000367 |
| Policy loss coeff. | 2 | Value loss coeff. | 0.5 |
| GAE parameter | 0.95 | Num. environment | 1024 |
| Discount factor | 0.99 | Episode length | 300 |
| PPO clip range | 0.3 | Update frequency | 8 |
| Bound loss coeff. | 0.02 | Energy loss coeff. | 8e-5 |

**说明**: TABLE X: PPO Hyperparameters.

#### Table 11: TABLE XI: Content of EnvCode.

| Parameter | Dimensions | Description |
| --- | --- | --- |
| ramp position | $\mathbb{R}^{2\times 2} 2 × 2$ | Position of each ramp |
| ramp slope | $\mathbb{R}^{2\times 2} 2 × 2$ | Angle of each ramp |
| plate elevations | $\mathbb{R}^{2\times 3} 2 × 3$ | Height of each base plate |
| wall heights | R 4 R 4  $\mathbb{R}^{4} 4$ | Height of each wall |
| ceiling height | R 1 R 1  $\mathbb{R}^{1} 1$ | Height of the ceiling |
| scene dimension | R 3 R 3  $\mathbb{R}^{3} 3$ | Overall scene dimension |
| scene position | R 3 R 3  $\mathbb{R}^{3} 3$ | Overall scene position |

**说明**: TABLE XI: Content of EnvCode.

#### Table 12: TABLE XII: MLP baseline configurations scaled to match the parameter count of the HAMnet network for each module count.

| Corresponding | | |
| --- | --- | --- |
| Num. modules | | |
| 1 | [256, 128, 128, 64] | 0.36 M |
| 2 | [304, 144, 144, 64] | 0.43 M |
| 4 | [512, 256, 256, 128] | 0.94 M |
| 8 | [768, 384, 384, 192] | 1.76 M |

**说明**: TABLE XII: MLP baseline configurations scaled to match the parameter count of the HAMnet network for each module count. Network size denotes the dimensions of hidden layers.
## 实验解读

- 评价重点:围绕 adaptive-control、接触推理、grasping、non-prehensile-manipulation、强化学习,优先看真实机器人结果、扰动恢复、接触稳定性、控制误差、成功率、实时性和 sim-to-real 差距。
- System0 视角:实验结论不只看最终成功率,还要看闭环过程中是否能处理状态估计误差、执行延迟、碰撞/接触突变和外部扰动。
- 与上层系统的关系:如果方法能在快闭环内稳定工作,它可以作为 System1 技能或 System2 规划的底层执行保障;如果只在离线或慢速设定下成立,则还不能直接作为身体级快闭环模块。
## 批判性思考

### 优点

- 论文把 adaptive-control、接触推理、grasping、non-prehensile-manipulation、强化学习 放到机器人身体级闭环里讨论,和只做高层语义推理的工作相比,更接近真实机器人部署时会遇到的状态、动力学和安全约束。
- 若论文同时给出公式、系统结构图、硬件实验或消融结果,这些证据可以帮助判断方法是否真的能支撑毫秒到百毫秒级反应。

### 局限

- 需要重点核查控制频率、推理延迟、传感延迟、失败恢复、硬件磨损、未见场景泛化和安全边界;这些因素决定它能否成为可靠的 System0 模块。
- 对学习型方法,还要区分训练环境中的鲁棒性和真实部署中的鲁棒性,尤其是接触丰富任务、动态运动任务和人形机器人任务。
## 论文定位

- 研究对象:Hierarchical and Modular Network on Non-prehensile Manipulation in General Environments。
- 关键词:adaptive-control、接触推理、grasping、non-prehensile-manipulation、强化学习、robot-generalization、机器人操作、zero-shot。
- System0 关联:关注机器人身体级快反应链路中“状态/接触/局部感知 -> 约束/优化/策略 -> 动作”的短闭环,不把问题停留在语言规划或任务分解层。

## 速查卡片

> [!summary] Hierarchical Modular Network
> - **论文**: https://www.roboticsproceedings.org/rss21/p154.pdf
> - **arXiv**: http://arxiv.org/abs/2502.20843v2
> - **arXiv HTML**: https://arxiv.org/html/2502.20843v2
