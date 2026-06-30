---
title: "SimToolReal: An Object-Centric Policy for Zero-Shot Dexterous Tool Manipulation"
method_name: "SimToolReal"
authors: ["Kushal Kedia"]
year: 2026
venue: "RSS"
tags: ["contact-reasoning", "robot-manipulation", "reinforcement-learning", "imitation-learning", "robot-generalization", "dexterous-manipulation", "sim-to-real"]
image_source: "online"
arxiv_html: "https://arxiv.org/html/2602.16863v2"
---
# SimToolReal
## 一句话总结

> SimToolReal: An Object-Centric Policy for Zero-Shot Dexterous Tool Manipulation 主要落在 [[接触推理]]、[[灵巧操作]]、[[grasping]]、[[in-hand-manipulation]] 这一类问题上;从 System0 视角看,它的价值在于把身体状态、接触变化、局部感知或动力学约束接到机器人低层闭环里,让动作生成更稳定、安全、实时。
## 核心贡献

1. 围绕 **SimToolReal: An Object-Centric Policy for Zero-Shot Dexterous Tool Manipulation** 建立了一个与 接触推理、灵巧操作、grasping、in-hand-manipulation、inference-time-algorithm、运动模仿 相关的机器人问题设定,重点不是高层语义规划,而是身体级闭环中的状态变化、接触变化或控制约束。
2. 方法层面的关键价值在于把感知、动力学、约束优化、策略学习或硬件反馈组织成可执行的低层动作生成机制,适合放进 System0 的快反应模块中理解。
3. 对系统集成而言,这类工作补上了 System1 技能选择与电机级控制之间的一层:它负责把上层目标变成动态可行、可恢复、可安全过滤的动作。
4. 若论文包含真实机器人、接触任务或仿真到真实迁移,核心应看它是否报告了闭环稳定性、扰动恢复、硬件延迟、失败模式和跨场景泛化。
## 问题背景

这篇论文可以放在 System0 的“具身快系统”背景下阅读:机器人需要在很短时间内根据本体状态、接触、力/触觉、局部视觉或动力学预测做出动作,而不是等待慢速规划器重新推理整个任务。接触推理、灵巧操作、grasping、in-hand-manipulation、inference-time-algorithm、运动模仿、强化学习 相关问题的难点通常不在“知道要做什么”,而在动作执行过程中状态会快速偏离、接触会突然改变、约束会实时收紧,低层系统必须及时修正。

### System0 关联

精读时优先看三件事:第一,论文使用哪些快速反馈量,例如关节状态、接触状态、力觉、触觉、深度或局部视觉;第二,反馈如何进入 MPC、WBC、策略网络、安全过滤器或恢复控制器;第三,输出是否直接影响机器人下一步动作,而不是只服务离线分析或高层任务规划。
## 方法详解

### System0 视角解析

从 System0 角度看,这篇论文的方法部分应当围绕 接触推理、灵巧操作、grasping、in-hand-manipulation、inference-time-algorithm、运动模仿、强化学习、retargeting 来读。核心问题是:系统如何从快速变化的机器人状态中构造控制输入,如何处理接触、碰撞、约束、模型误差或执行延迟,以及如何把这些信息变成下一时刻的动作。

如果方法里有优化问题,重点看目标函数、约束和求解频率;如果方法里有学习策略,重点看观测空间、动作空间、训练扰动和部署时的闭环频率;如果方法里有硬件或传感器,重点看传感链路是否真的进入控制回路。这样读可以把论文和 System2 语言规划、System1 技能编排区分开来。

### 关键公式与机制

#### 公式 1: [[动力学或策略机制]]

$$\boldsymbol{a}_{t}=\pi_{\theta}\!\left(\boldsymbol{s}_{t},\,\boldsymbol{o}_{t},\,\boldsymbol{\phi},\,\boldsymbol{g}\right)$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 2: [[动力学或策略机制]]

$$r_{\mathrm{goal}}=\mathrm{max}(d^{*}-d(\boldsymbol{o}_{t},\boldsymbol{g}),0)+B_{\mathrm{succ}}\,\mathbb{I}\!\left[d(\boldsymbol{o}_{t},\boldsymbol{g})<\epsilon\right],$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 3: [[动力学或策略机制]]

$$r_{\text{smooth}}=-\lambda_{\text{arm}}\lVert \dot{\mathbf{q}}_{\text{arm}} \rVert_{1}-\lambda_{\text{hand}}\lVert \dot{\mathbf{q}}_{\text{hand}} \rVert_{1}$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 4: [[优化目标/约束]]

$$r_{\text{approach}}=\lambda_{\text{approach}}\max(\bar{d}^{*}_{\text{ft}}-\bar{d}_{\text{ft}},0)$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 5: [[优化目标/约束]]

$$r_{\text{lift}}=\lambda_{\text{lift}}\max(z-z_{\text{init}},0)+\mathbb{I}[z\geq z_{\text{lifted}}]B_{\text{lifted}}$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 6: [[优化目标/约束]]

$$d(\boldsymbol{o}_{t},\boldsymbol{g})=\max_{i}\left\lVert \boldsymbol{o}_{t,i}-\boldsymbol{g}_{i}\right \rVert,$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 7: [[动力学或策略机制]]

$$\mathbf{s}^{\text{rew}}=[s_{x}^{\text{rew}},s_{y}^{\text{rew}},s_{z}^{\text{rew}}]=[0.14,0.03,0.03]$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 8: [[动力学或策略机制]]

$$\mathbf{k}\in\left\{\begin{bmatrix}s_{x}^{\text{rew}}/2\\ s_{y}^{\text{rew}}/2\\ s_{z}^{\text{rew}}/2\end{bmatrix},\begin{bmatrix}s_{x}^{\text{rew}}/2\\ s_{y}^{\text{rew}}/2\\ -s_{z}^{\text{rew}}/2\end{bmatrix},\begin{bmatrix}-s_{x}^{\text{rew}}/2\\ -s_{y}^{\text{rew}}/2\\ s_{z}^{\text{rew}}/2\end{bmatrix},\begin{bmatrix}-s_{x}^{\text{rew}}/2\\ -s_{y}^{\text{rew}}/2\\ -s_{z}^{\text{rew}}/2\end{bmatrix}\right\}$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 9: [[动力学或策略机制]]

$$\hat{\boldsymbol{q}}^{\text{hand}}_{t}=\frac{\boldsymbol{a}^{\text{hand}}_{t}+1}{2}\odot(\boldsymbol{q}^{\text{hand}}_{\text{upper}}-\boldsymbol{q}^{\text{hand}}_{\text{lower}})+\boldsymbol{q}^{\text{hand}}_{\text{lower}}$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 10: [[动力学或策略机制]]

$$\mathbf{k}_{\text{obs}}\in\left\{\begin{bmatrix}s_{x}/2\\ s_{y}/2\\ s_{z}/2\end{bmatrix},\begin{bmatrix}s_{x}/2\\ s_{y}/2\\ -s_{z}/2\end{bmatrix},\begin{bmatrix}-s_{x}/2\\ -s_{y}/2\\ s_{z}/2\end{bmatrix},\begin{bmatrix}-s_{x}/2\\ -s_{y}/2\\ -s_{z}/2\end{bmatrix}\right\}$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。
## 关键图表

### Figure 1: Overview of SimToolReal. (Top) Training in Simulation: We train a goal-conditioned RL

![Figure 1](https://arxiv.org/html/2602.16863v2/x2.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Overview of SimToolReal. (Top) Training in Simulation: We train a goal-conditioned RL”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 2: Real-World Deployment. (Left) Human Video Processing: We collect an RGB-D human video

![Figure 2](https://arxiv.org/html/2602.16863v2/x3.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Real-World Deployment. (Left) Human Video Processing: We collect an RGB-D human video”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 3: Kinematic Retargeting Pipeline. From the RGB-D human video, we use SAM 2 [65 ] for h

![Figure 3](https://arxiv.org/html/2602.16863v2/x10.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Kinematic Retargeting Pipeline. From the RGB-D human video, we use SAM 2 [65 ] for h”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### 论文表格

#### Table 1: TABLE I: Simulation environment and SAPG training hyperparameters.

| Parameter | Value | Parameter | Value |
| --- | --- | --- | --- |
| Environment & Control | SAPG Hyperparameters | | |
| Simulation / Control Frequency | 120 / 60 Hz | Actor Network | LSTM[1024] + MLP[1024,1024,512,512] |
| Num. Environments | 24576 | Critic Network | MLP[1024,1024,512,512] |
| Episode Length | 600 steps | Learning Rate | 1  10 - 4 1\times 10^{-4} |
| Obj. Pos. Range (x, y x,y) | ± 10 \pm\,10 cm | Minibatch Size | 98,304 |
| Table Height Range (z z) | ± 1 \pm\,1 cm | SAPG Block Size | 4096 |
| Robot Joint Pos. Range | ± 0.1 \pm\,0.1 rad | Entropy Bonus Scale | 0.005 |
| Success Tolerance ( \epsilon) | 1 cm | Discount Factor $\gamma)$ | 0.99 |
| Initial Height (z init z_{ $\text{init}})$ | 0.63 m | GAE Parameter $\tau)$ | 0.95 |
| Lift Threshold (z lifted z_{ $\text{lifted}})$ | 0.73 m | Clip Range | 0.1 |
| Domain Randomization | Reward Coefficients | | |
| Obj. Pose Noise (Trans.) | 1 cm | Arm Action Penalty arm \lambda $\text{arm}})$ | 0.03 |
| Obj. Pose Noise (Rot.) | 5.0 ∘ 5.0^{\circ} | Hand Action Penalty hand \lambda $\text{hand}})$ | 0.003 |
| Obj. Pose Delay Max | 10 steps | Approach Scale approach \lambda $\text{approach}})$ | 50.0 |
| Action/Obs Delay Max | 3 steps | Lifting Scale lift \lambda $\text{lift}})$ | 20.0 |
| Joint Vel. Obs Noise ( \sigma) | 0.1 rad/s | Lifting Bonus (B lifted B_{ $\text{lifted}})$ | 300.0 |
| Perturb. Force Scale | 5.0 N | $\text{goal}})$ | 200.0 |
| Perturb. Torque Scale | 0.5 Nm | Success Bonus (B succ) B_{ $\text{succ}})$ | 1000.0 |

**说明**: TABLE I: Simulation environment and SAPG training hyperparameters.

#### Table 2: TABLE II: Detailed Real-World Evaluation Results. We report the Task Progress (%) for each of the 5 rollouts across all

| Category | Instance | Trajectory | R1 | R2 | R3 | R4 | R5 | Avg |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Hammer | Claw | Swing Down | 100 | 100 | 100 | 100 | 100 | 100 |
| Swing Side | 100 | 100 | 77.5 | 100 | 100 | 95.5 | | |
| Mallet | Swing Down | 100 | 100 | 33.3 | 100 | 88.9 | 84.4 | |
| Swing Side | 65.6 | 81.3 | 100 | 78.1 | 62.5 | 77.5 | | |
| Marker | Sharpie | Draw Smile | 100 | 100 | 100 | 100 | 0.0 | 80.0 |
| Write C | 100 | 100 | 100 | 32.0 | 56.0 | 77.6 | | |
| Staples | Draw Smile | 100 | 53.1 | 100 | 100 | 100 | 90.6 | |
| Write C | 31.0 | 0.0 | 100 | 100 | 100 | 66.2 | | |
| Eraser | Handle | Wipe Smile | 100 | 100 | 100 | 100 | 100 | 100 |
| Wipe C | 100 | 100 | 100 | 100 | 100 | 100 | | |
| Flat | Wipe Smile | 100 | 100 | 100 | 100 | 100 | 100 | |
| Wipe C | 100 | 100 | 100 | 100 | 100 | 100 | | |
| Brush | Blue | Sweep Fwd | 47.8 | 47.8 | 40.0 | 60.0 | 60.0 | 51.1 |
| Sweep Right | 100 | 100 | 100 | 100 | 100 | 100 | | |
| Red | Sweep Fwd | 100 | 100 | 13.5 | 100 | 100 | 82.7 | |
| Sweep Right | 43.2 | 62.2 | 100 | 42.2 | 59.5 | 61.4 | | |
| Spatula | Spoon | Serve Plate | 100 | 76.5 | 100 | 48.5 | 100 | 85.0 |
| Flip Over | 77.5 | 100 | 100 | 100 | 100 | 95.5 | | |
| Flat | Serve Plate | 100 | 75.0 | 78.1 | 56.3 | 78.1 | 77.5 | |
| Flip Over | 0.0 | 52.2 | 0.0 | 78.3 | 100 | 46.1 | | |
| Screwdriver | Long | Spin Vert | 20.9 | 72.1 | 53.5 | 74.4 | 58.1 | 55.8 |
| Spin Horiz | 48.4 | 86.7 | 83.3 | 3.3 | 86.7 | 61.7 | | |
| Short | Spin Vert | 15.4 | 61.5 | 12.8 | 100 | 0.0 | 37.9 | |
| Spin Horiz | 100 | 100 | 100 | 77.8 | 0.0 | 75.6 | | |

**说明**: TABLE II: Detailed Real-World Evaluation Results. We report the Task Progress (%) for each of the 5 rollouts across all 24 object-task variations. The specific tools correspond to the instances shown in Fig. 4.

#### Table 3: TABLE III: Specialist Objects and Trajectories. Objects and trajectories used for specialist policy training and evaluat

| Category | Obj A | Traj A | Obj B | Traj B |
| --- | --- | --- | --- | --- |
| Brush | Red | Sweep | Blue | Sweep |
| Brush | Forward | Brush | Right | |
| Eraser | Flat | Wipe | Handle | Wipe |
| Eraser | Smile | Eraser | C | |
| Hammer | Mallet | Swing | Claw | Swing |
| Hammer | Down | Hammer | Side | |
| Marker | Staples | Draw | Sharpie | Write |
| Marker | Smile | Marker | C | |
| Screwdriver | Long | Spin | Short | Spin |
| Screwdriver | Vertical | Screwdriver | Horizontal | |
| Spatula | Flat | Serve | Spoon | Flip |
| Spatula | Plate | Spatula | Over | |

**说明**: TABLE III: Specialist Objects and Trajectories. Objects and trajectories used for specialist policy training and evaluation.
## 实验解读

- 评价重点:围绕 接触推理、灵巧操作、grasping、in-hand-manipulation、inference-time-algorithm,优先看真实机器人结果、扰动恢复、接触稳定性、控制误差、成功率、实时性和 sim-to-real 差距。
- System0 视角:实验结论不只看最终成功率,还要看闭环过程中是否能处理状态估计误差、执行延迟、碰撞/接触突变和外部扰动。
- 与上层系统的关系:如果方法能在快闭环内稳定工作,它可以作为 System1 技能或 System2 规划的底层执行保障;如果只在离线或慢速设定下成立,则还不能直接作为身体级快闭环模块。
## 批判性思考

### 优点

- 论文把 接触推理、灵巧操作、grasping、in-hand-manipulation、inference-time-algorithm 放到机器人身体级闭环里讨论,和只做高层语义推理的工作相比,更接近真实机器人部署时会遇到的状态、动力学和安全约束。
- 若论文同时给出公式、系统结构图、硬件实验或消融结果,这些证据可以帮助判断方法是否真的能支撑毫秒到百毫秒级反应。

### 局限

- 需要重点核查控制频率、推理延迟、传感延迟、失败恢复、硬件磨损、未见场景泛化和安全边界;这些因素决定它能否成为可靠的 System0 模块。
- 对学习型方法,还要区分训练环境中的鲁棒性和真实部署中的鲁棒性,尤其是接触丰富任务、动态运动任务和人形机器人任务。
## 论文定位

- 研究对象:SimToolReal: An Object-Centric Policy for Zero-Shot Dexterous Tool Manipulation。
- 关键词:接触推理、灵巧操作、grasping、in-hand-manipulation、inference-time-algorithm、运动模仿、强化学习、retargeting、robot-generalization、机器人操作。
- System0 关联:关注机器人身体级快反应链路中“状态/接触/局部感知 -> 约束/优化/策略 -> 动作”的短闭环,不把问题停留在语言规划或任务分解层。

## 速查卡片

> [!summary] SimToolReal
> - **论文**: https://www.roboticsproceedings.org/rss22/p151.pdf
> - **arXiv**: http://arxiv.org/abs/2602.16863v2
> - **arXiv HTML**: https://arxiv.org/html/2602.16863v2
