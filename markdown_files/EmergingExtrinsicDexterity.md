---
title: "Emerging Extrinsic Dexterity in Cluttered Scenes via Dynamics-aware Policy Learning"
method_name: "Emerging Extrinsic Dexterity"
authors: ["Yixin Zheng"]
year: 2026
venue: "RSS"
tags: ["contact-reasoning", "robot-manipulation", "robust-control", "reinforcement-learning", "imitation-learning", "contact-rich-manipulation", "robot-generalization", "dexterous-manipulation", "sim-to-real", "recovery"]
image_source: "online"
arxiv_html: "https://arxiv.org/html/2603.09882v2"
---
# Emerging Extrinsic Dexterity
## 一句话总结

> Emerging Extrinsic Dexterity in Cluttered Scenes via Dynamics-aware Policy Learning 主要落在 [[接触推理]]、[[灵巧操作]]、[[non-prehensile-manipulation]]、[[policy-learning]] 这一类问题上;从 System0 视角看,它的价值在于把身体状态、接触变化、局部感知或动力学约束接到机器人低层闭环里,让动作生成更稳定、安全、实时。
## 核心贡献

1. 围绕 **Emerging Extrinsic Dexterity in Cluttered Scenes via Dynamics-aware Policy Learning** 建立了一个与 接触推理、灵巧操作、non-prehensile-manipulation、policy-learning、recovery、强化学习 相关的机器人问题设定,重点不是高层语义规划,而是身体级闭环中的状态变化、接触变化或控制约束。
2. 方法层面的关键价值在于把感知、动力学、约束优化、策略学习或硬件反馈组织成可执行的低层动作生成机制,适合放进 System0 的快反应模块中理解。
3. 对系统集成而言,这类工作补上了 System1 技能选择与电机级控制之间的一层:它负责把上层目标变成动态可行、可恢复、可安全过滤的动作。
4. 若论文包含真实机器人、接触任务或仿真到真实迁移,核心应看它是否报告了闭环稳定性、扰动恢复、硬件延迟、失败模式和跨场景泛化。
## 问题背景

这篇论文可以放在 System0 的“具身快系统”背景下阅读:机器人需要在很短时间内根据本体状态、接触、力/触觉、局部视觉或动力学预测做出动作,而不是等待慢速规划器重新推理整个任务。接触推理、灵巧操作、non-prehensile-manipulation、policy-learning、recovery、强化学习、机器人操作 相关问题的难点通常不在“知道要做什么”,而在动作执行过程中状态会快速偏离、接触会突然改变、约束会实时收紧,低层系统必须及时修正。

### System0 关联

精读时优先看三件事:第一,论文使用哪些快速反馈量,例如关节状态、接触状态、力觉、触觉、深度或局部视觉;第二,反馈如何进入 MPC、WBC、策略网络、安全过滤器或恢复控制器;第三,输出是否直接影响机器人下一步动作,而不是只服务离线分析或高层任务规划。
## 方法详解

### System0 视角解析

从 System0 角度看,这篇论文的方法部分应当围绕 接触推理、灵巧操作、non-prehensile-manipulation、policy-learning、recovery、强化学习、机器人操作、鲁棒控制 来读。核心问题是:系统如何从快速变化的机器人状态中构造控制输入,如何处理接触、碰撞、约束、模型误差或执行延迟,以及如何把这些信息变成下一时刻的动作。

如果方法里有优化问题,重点看目标函数、约束和求解频率;如果方法里有学习策略,重点看观测空间、动作空间、训练扰动和部署时的闭环频率;如果方法里有硬件或传感器,重点看传感链路是否真的进入控制回路。这样读可以把论文和 System2 语言规划、System1 技能编排区分开来。

### 关键公式与机制

#### 公式 1: [[优化目标/约束]]

$$
\mathcal{L}_{\text{dyn}}=\sum_{i}\left\lVert \hat{\mathbf{p}}_{i}^{t+1}-\mathbf{p}_{i}^{t+1}\right \rVert_{2}^{2}+\lambda\left\lVert \hat{\mathbf{v}}_{i}^{t+1}-\mathbf{v}_{i}^{t+1}\right \rVert_{2}^{2}.
$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 2: [[优化目标/约束]]

$$
\mathcal{L}_{\text{var}}=\left\lVert \mathrm{Std}\!\left(\{\hat{\mathbf{v}}_{i}^{t+1}\}_{i}\right)-\mathrm{Std}\!\left(\{\mathbf{v}_{i}^{t+1}\}_{i}\right)\right \rVert_{2},
$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 3: [[动力学或策略机制]]

$$r_{\text{goal}}=\mathbb{I}(d_{\text{oe}}<\tau_{d})\left(1-\tanh(d_{\text{og}})\right).$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 4: [[优化目标/约束]]

$$\mathcal{L}=\lambda_{\text{pos}}\mathcal{L}_{\text{pos}}+\lambda_{\text{vel}}\mathcal{L}_{\text{vel}}+\lambda_{\text{var}}\mathcal{L}_{\text{var}},$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 5: [[优化目标/约束]]

$$d_{\text{ee}}=\min\left(\lVert p_{\text{obj}}-p_{\text{ee,L}} \rVert,\lVert p_{\text{obj}}-p_{\text{ee,R}} \rVert\right).$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 6: [[动力学或策略机制]]

$$\mathbb{I}_{\text{near}}=\begin{cases}1,&d_{\text{ee}}<d_{\text{th}},\\ 0,&\text{otherwise},\end{cases}$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 7: [[动力学或策略机制]]

$$r_{\text{goal}}=\mathbb{I}_{\text{near}}\left(1-\tanh\left(\frac{d}{\sigma_{\text{coarse}}}\right)\right),$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 8: [[动力学或策略机制]]

$$r_{\text{goal-fine}}=\mathbb{I}_{\text{near}}\left(1-\tanh\left(\frac{d}{\sigma_{\text{fine}}}\right)\right),$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 9: [[动力学或策略机制]]

$$\bar{d}=\frac{1}{N_{\text{obs}}}\sum_{i=1}^{N_{\text{obs}}}d_{\text{obs},i},\quad\bar{\theta}=\frac{1}{N_{\text{obs}}}\sum_{i=1}^{N_{\text{obs}}}\theta_{\text{obs},i}.$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 10: [[优化目标/约束]]

$$\hat{d}=\text{clip}\left(\frac{\bar{d}}{d_{\max}},0,1\right),\quad\hat{\theta}=\text{clip}\left(\frac{\bar{\theta}}{\theta_{\max}},0,1\right).$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。
## 关键图表

### Figure 1: Overview of the proposed two-stage learning framework. Stage 1 (World Model Pretrainin

![Figure 1](https://arxiv.org/html/2603.09882v2/x1.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Overview of the proposed two-stage learning framework. Stage 1 (World Model Pretrainin”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 2: Overview of our proposed Clutter6D Benchmark and real-world setup. (a) Representative

![Figure 2](https://arxiv.org/html/2603.09882v2/x2.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Overview of our proposed Clutter6D Benchmark and real-world setup. (a) Representative”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 3: Training efficiency and convergence comparison. Our method (blue) achieves significant

![Figure 3](https://arxiv.org/html/2603.09882v2/x3.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Training efficiency and convergence comparison. Our method (blue) achieves significant”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### 论文表格

#### Table 1: TABLE I: Quantitative results measured by success rate in the simulation benchmark. Note that Mean Offset (M.O.) for Gra

| Methods | Action Type | Sparse | Moderate | Dense | | | |
| --- | --- | --- | --- | --- | --- | --- | --- |
| S.R. ↑ \uparrow | M.O. ↓ \downarrow | S.R. ↑ \uparrow | M.O. ↓ \downarrow | S.R. ↑ \uparrow | M.O. ↓ \downarrow | | |
| Teleoperation [38 ] | Mixed | 50.0 | 3.13 | 40.0 | 7.49 | 20.0 | 21.34 |
| GraspGen + CuRobo [28, 34 ] | Prehensile | 26.6 | – | 15.6 | – | 3.13 | – |
| Point2Vec [22 ] | Non-prehensile | 6.89 | 5.09 | 1.95 | 3.36 | 0.78 | 5.35 |
| Concerto [42 ] | Non-prehensile | 3.13 | 1.65 | 1.56 | 2.90 | 0.39 | 7.56 |
| CORN [8 ] | Non-prehensile | 46.63 | 3.15 | 45.83 | 5.51 | 22.22 | 17.43 |
| CORN-multi | Non-prehensile | 35.93 | 2.73 | 15.38 | 3.92 | 11.83 | 12.06 |
| UniCORN [9 ] | Non-prehensile | 20.61 | 1.71 | 11.67 | 4.13 | 5.81 | 9.79 |
| Ours | Non-prehensile | 71.88 | 2.59 | 51.04 | 2.7 | 44.56 | 12.65 |

**说明**: TABLE I: Quantitative results measured by success rate in the simulation benchmark. Note that Mean Offset (M.O.) for GraspGen + CuRobo is omitted because it relies on the CuRobo motion planner, which strictly prioritizes collision-free trajectories. Our method significantly outperforms all baselines across all clutter densities, maintaining high success rates even in dense environments where baselines experience a sharp performance decline.

#### Table 2: TABLE II: Ablation Study on Pre-training Objectives and Input Modalities (Sparse Track). P.E.: Chamfer Distance predicti

| Pretrain Task | Granularity | Velocity | Phys. | P.E. ↓ \downarrow | S.R. ↑ \uparrow | M.O. ↓ \downarrow |
| --- | --- | --- | --- | --- | --- | --- |
| Recons. | Point-level | ✗ | ✗ | - | 11.75 | 1.31 |
| Recons. | Point-level | ✓ | ✓ | - | 29.63 | 2.63 |
| World Model | Object-level | ✗ | ✗ | 3.1 | 14.13 | 3.27 |
| World Model | Object-level | ✓ | ✓ | 3.2 | 16.88 | 3.84 |
| World Model | Point-level | ✗ | ✗ | 4.1 | 42.00 | 4.91 |
| World Model | Point-level | ✓ | ✗ | 5.1 | 58.25 | 4.86 |
| World Model | Point-level | ✓ | ✓ | 4.6 | 71.88 | 2.59 |

**说明**: TABLE II: Ablation Study on Pre-training Objectives and Input Modalities (Sparse Track). P.E.: Chamfer Distance prediction error (cm). Results highlight the indispensability of velocity and physical features, while further establishing that point-level dynamics modeling is a more effective pretext task than either object-level relative pose prediction or simple reconstruction.

#### Table 3: TABLE III: Behavior analysis 50 sampled trajectories per method. Unnec. Contacts and Object Disturbance metrics are

| Method | Unnec. Contacts | Object Disturbance | Dexterous Behaviors (%) | | | | | | |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Small Trans. | Small Rot. | Large Trans. | Large Rot. | Avoid | Traverse | Leverage | Simple | | |
| CORN | 3.6 | 2.8 | 2.6 | 4.4 | 4.8 | 22 | 14 | 6 | 62 |
| Ours | 2.4 | 3.2 | 3.0 | 1.8 | 1.2 | 52 | 38 | 42 | 28 |

**说明**: TABLE III: Behavior analysis 50 sampled trajectories per method. Unnec. Contacts and Object Disturbance metrics are reported as average counts per trajectory. Small/Large Trans. count non-target objects with translations below/above 5 cm, while Small/Large Rot. count non-target objects with rotations below/above 45 ∘ $45^{\circ}$. Avoid, Traverse, and Leverage correspond to the dexterous behavior types illustrated in Fig. LABEL:first_figure. Multiple behavior types may occur in one trajectory; Simple denotes trajectories in which none of the three is observed.

#### Table 4: TABLE IV: Sensitivity to noisy physical and goal inputs controlled simulation perturbations. Noise levels are repo

| Noise Source | 0.5% | 5% | 25% | 50% | | | | |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| S.R. | M.O. | S.R. | M.O. | S.R. | M.O. | S.R. | M.O. | |
| Mass | 66.9 | 3.1 | 67.3 | 3.4 | 66.0 | 4.1 | 62.4 | 5.9 |
| Velocity | 64.0 | 3.6 | 64.0 | 3.2 | 61.1 | 3.9 | 58.5 | 4.4 |
| Goal Pose | 66.3 | 3.1 | 65.0 | 3.3 | 48.1 | 3.5 | 15.1 | 4.0 |
| All Inputs | 65.7 | 3.7 | 66.6 | 4.9 | 45.1 | 6.0 | 5.9 | 8.4 |

**说明**: TABLE IV: Sensitivity to noisy physical and goal inputs controlled simulation perturbations. Noise levels are reported as percentages of the corresponding input mean. S.R. denotes success rate and M.O. denotes mean offset (cm).

#### Table 5: TABLE V: Quantitative results across 10 scenes. Mean (last column) denotes average performance across all scenes. SR ↑ \

| Methods | S1 | S2 | S3 | S4 | | | | |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| | S.R. ↑ \uparrow | M.T. ↓ \downarrow | S.R. ↑ \uparrow | M.T. ↓ \downarrow | S.R. ↑ \uparrow | M.T. ↓ \downarrow | S.R. ↑ \uparrow | M.T. ↓ \downarrow |
| Teleop. [25 ] | 3/5 | 77 | 4/5 | 59 | 3/5 | 39 | 1/5 | 81 |
| Ours | 2/5 | 51 | 1/5 | 26 | 3/5 | 33 | 1/5 | 73 |
| Methods | S5 | S6 | S7 | S8 | | | | |
| | S.R. ↑ \uparrow | M.T. ↓ \downarrow | S.R. ↑ \uparrow | M.T. ↓ \downarrow | S.R. ↑ \uparrow | M.T. ↓ \downarrow | S.R. ↑ \uparrow | M.T. ↓ \downarrow |
| Teleop. [25 ] | 3/5 | 28 | 2/5 | 39 | 4/5 | 20 | 0/5 | 90 |
| Ours | 4/5 | 37 | 3/5 | 59 | 2/5 | 24 | 2/5 | 28 |
| Methods | S9 | S10 | Avg. | | | | | |
| | S.R. ↑ \uparrow | M.T. ↓ \downarrow | S.R. ↑ \uparrow | M.T. ↓ \downarrow | S.R. ↑ \uparrow | M.T. ↓ \downarrow | | |
| Teleop. [25 ] | 5/5 | 62 | 1/5 | 64 | 52% | 55.9 | | |
| Ours | 5/5 | 45 | 1/5 | 50 | 48% | 42.6 | | |

**说明**: TABLE V: Quantitative results across 10 scenes. Mean (last column) denotes average performance across all scenes. SR ↑ $\uparrow$ indicates success rate and MT ↓ $\downarrow$ indicates mean execution time.

#### Table 6: TABLE I: Summary of our MDP components. † $\dagger$: simulation only.

| State Component | Symbol | Dimension | Description |
| --- | --- | --- | --- |
| Object point cloud | P o P_{o} | $\mathbb{R}^{512\times 7}$ | Surface-sampled object point cloud in the environment frame, with per-point mass and contact-conditioned velocity (x, y, z, m, v x, v y, v z) (x,y,z,m,v_{x},v_{y},v_{z}) |
| Environment point cloud | P e P_{e} | $\mathbb{R}^{512\times 7}$ | Surface-sampled obstacle point cloud in the environment frame, with per-point mass and velocity (x, y, z, m, v x, v y, v z) (x,y,z,m,v_{x},v_{y},v_{z}) |
| End-effector point cloud | P e  e P_{ee} | $\mathbb{R}^{256\times 7}$ | Surface-sampled end-effector point cloud in the environment frame, with per-point mass and velocity (x, y, z, m, v x, v y, v z) (x,y,z,m,v_{x},v_{y},v_{z}) |
| Hand state | s t E  E s_{t}^{EE} | R 9  $\mathbb{R}^{9}$ | End-effector position and orientation in 6D rotation representation (3D position + 6D orientation) |
| Robot state | s t q s_{t}^{q} | R 14  $\mathbb{R}^{14}$ | Joint positions and velocities of the 7-DoF arm |
| Relative goal pose | T g T_{g} | R 9  $\mathbb{R}^{9}$ | Target object pose expressed relative to the current object pose (3D translation + 6D orientation) |
| Physics parameters † |  \rho | R 5  $\mathbb{R}^{5}$ | Object mass, object static friction, hand friction, ground friction, and object restitution |
| Previous action | a t - 1 a_{t-1} | R 7  $\mathbb{R}^{7}$ | Previous-step joint-space residual action applied to the arm |
| Action Component | Symbol | Dimension | Description |
| Joint-space residuals |   q t \Delta q_{t} | R 7  $\mathbb{R}^{7}$ | Relative joint position commands q target q t q t q $\text{target}}=q_{t}+\Delta q_{t} for the 7 arm joints$ |

**说明**: TABLE I: Summary of our MDP components. † $\dagger$: simulation only.

#### Table 7: TABLE II: PPO Algorithm Hyperparameters

| Parameter | Value |
| --- | --- |
| Value loss coefficient | 0.5 |
| Use clipped value loss | True |
| Clip parameter ( \epsilon) | 0.3 |
| Entropy coefficient | 0.006 |
| Learning epochs | 8 |
| Mini-batches | 8 |
| Learning rate | 5.0  10 - 5 5.0\times 10^{-5} |
| LR Schedule | Adaptive |
| Discount factor $\gamma)$ | 0.99 |
| GAE parameter ( \lambda) | 0.95 |
| Desired KL divergence | 0.016 |
| Max gradient norm | 1.0 |

**说明**: TABLE II: PPO Algorithm Hyperparameters

#### Table 8: TABLE III: Reward terms and weights used for RL training.

| Reward term | Weight |
| --- | --- |
| r contact r_{ $\text{contact}}$ | 1.0 |
| r goal r_{ $\text{goal}}$ | 5.0 |
| r goal-fine r_{ $\text{goal-fine}}$ | 16.0 |
| r success r_{ $\text{success}}$ | 2000.0 |
| $\text{coarse}}$ | 0.6 |
| $\text{fine}}$ | 0.3 |

**说明**: TABLE III: Reward terms and weights used for RL training.

#### Table 9: TABLE IV: Difficulty levels and procedural object counts.

| Difficulty | Large Obstacles | Small Obstacles | Target |
| --- | --- | --- | --- |
| Sparse | 1 | 2 | 1 |
| Moderate | 3 | 4 | 1 |
| Dense | 5 | 6 | 1 |

**说明**: TABLE IV: Difficulty levels and procedural object counts.
## 实验解读

- 评价重点:围绕 接触推理、灵巧操作、non-prehensile-manipulation、policy-learning、recovery,优先看真实机器人结果、扰动恢复、接触稳定性、控制误差、成功率、实时性和 sim-to-real 差距。
- System0 视角:实验结论不只看最终成功率,还要看闭环过程中是否能处理状态估计误差、执行延迟、碰撞/接触突变和外部扰动。
- 与上层系统的关系:如果方法能在快闭环内稳定工作,它可以作为 System1 技能或 System2 规划的底层执行保障;如果只在离线或慢速设定下成立,则还不能直接作为身体级快闭环模块。
## 批判性思考

### 优点

- 论文把 接触推理、灵巧操作、non-prehensile-manipulation、policy-learning、recovery 放到机器人身体级闭环里讨论,和只做高层语义推理的工作相比,更接近真实机器人部署时会遇到的状态、动力学和安全约束。
- 若论文同时给出公式、系统结构图、硬件实验或消融结果,这些证据可以帮助判断方法是否真的能支撑毫秒到百毫秒级反应。

### 局限

- 需要重点核查控制频率、推理延迟、传感延迟、失败恢复、硬件磨损、未见场景泛化和安全边界;这些因素决定它能否成为可靠的 System0 模块。
- 对学习型方法,还要区分训练环境中的鲁棒性和真实部署中的鲁棒性,尤其是接触丰富任务、动态运动任务和人形机器人任务。
## 论文定位

- 研究对象:Emerging Extrinsic Dexterity in Cluttered Scenes via Dynamics-aware Policy Learning。
- 关键词:接触推理、灵巧操作、non-prehensile-manipulation、policy-learning、recovery、强化学习、机器人操作、鲁棒控制、仿真到真实迁移、遥操作。
- System0 关联:关注机器人身体级快反应链路中“状态/接触/局部感知 -> 约束/优化/策略 -> 动作”的短闭环,不把问题停留在语言规划或任务分解层。

## 速查卡片

> [!summary] Emerging Extrinsic Dexterity
> - **论文**: https://www.roboticsproceedings.org/rss22/p149.pdf
> - **arXiv**: http://arxiv.org/abs/2603.09882v2
> - **arXiv HTML**: https://arxiv.org/html/2603.09882v2
