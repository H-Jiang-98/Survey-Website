---
title: "When Life Gives You BC, Make Q-functions: Extracting Q-values from Behavior Cloning for On-Robot Reinforcement Learning"
method_name: "When Life BC"
authors: ["Lakshita Dodeja"]
year: 2026
venue: "RSS"
tags: ["contact-reasoning", "robot-manipulation", "robust-control", "real-time-control", "reinforcement-learning", "imitation-learning", "contact-rich-manipulation"]
image_source: "online"
arxiv_html: "https://arxiv.org/html/2605.05172v2"
---
# When Life BC
## 一句话总结

> When Life Gives You BC, Make Q-functions: Extracting Q-values from Behavior Cloning for On-Robot Reinforcement Learning 主要落在 [[assembly]]、[[behavior-cloning]]、[[接触推理]]、[[接触丰富操作]] 这一类问题上;从 System0 视角看,它的价值在于把身体状态、接触变化、局部感知或动力学约束接到机器人低层闭环里,让动作生成更稳定、安全、实时。
## 核心贡献

1. 围绕 **When Life Gives You BC, Make Q-functions: Extracting Q-values from Behavior Cloning for On-Robot Reinforcement Learning** 建立了一个与 assembly、behavior-cloning、接触推理、接触丰富操作、模仿学习、实时控制 相关的机器人问题设定,重点不是高层语义规划,而是身体级闭环中的状态变化、接触变化或控制约束。
2. 方法层面的关键价值在于把感知、动力学、约束优化、策略学习或硬件反馈组织成可执行的低层动作生成机制,适合放进 System0 的快反应模块中理解。
3. 对系统集成而言,这类工作补上了 System1 技能选择与电机级控制之间的一层:它负责把上层目标变成动态可行、可恢复、可安全过滤的动作。
4. 若论文包含真实机器人、接触任务或仿真到真实迁移,核心应看它是否报告了闭环稳定性、扰动恢复、硬件延迟、失败模式和跨场景泛化。
## 问题背景

这篇论文可以放在 System0 的“具身快系统”背景下阅读:机器人需要在很短时间内根据本体状态、接触、力/触觉、局部视觉或动力学预测做出动作,而不是等待慢速规划器重新推理整个任务。assembly、behavior-cloning、接触推理、接触丰富操作、模仿学习、实时控制、强化学习 相关问题的难点通常不在“知道要做什么”,而在动作执行过程中状态会快速偏离、接触会突然改变、约束会实时收紧,低层系统必须及时修正。

### System0 关联

精读时优先看三件事:第一,论文使用哪些快速反馈量,例如关节状态、接触状态、力觉、触觉、深度或局部视觉;第二,反馈如何进入 MPC、WBC、策略网络、安全过滤器或恢复控制器;第三,输出是否直接影响机器人下一步动作,而不是只服务离线分析或高层任务规划。
## 方法详解

### System0 视角解析

从 System0 角度看,这篇论文的方法部分应当围绕 assembly、behavior-cloning、接触推理、接触丰富操作、模仿学习、实时控制、强化学习、机器人操作 来读。核心问题是:系统如何从快速变化的机器人状态中构造控制输入,如何处理接触、碰撞、约束、模型误差或执行延迟,以及如何把这些信息变成下一时刻的动作。

如果方法里有优化问题,重点看目标函数、约束和求解频率;如果方法里有学习策略,重点看观测空间、动作空间、训练扰动和部署时的闭环频率;如果方法里有硬件或传感器,重点看传感链路是否真的进入控制回路。这样读可以把论文和 System2 语言规划、System1 技能编排区分开来。

### 关键公式与机制

#### 公式 1: [[优化目标/约束]]

$$\mathcal{D}=\left\{\tau_{j}\right\}^{M}_{j=1},\qquad\tau_{j}=\left\{(o_{j,t},a_{j,t})\right\}^{T_{j}}_{t=1},$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 2: [[优化目标/约束]]

$$\mathcal{M}=\left\{\mathcal{S},\mathcal{A},\mathcal{R},\mathcal{T},\rho_{0},\gamma\right\}$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 3: [[动力学或策略机制]]

$$Q_{\pi}(s,a)=\mathbb{E}_{\tau\sim\pi}\left[\sum^{\infty}_{t=0}\gamma^{t}r(s_{t},a_{t})\mid s_{0}=s,a_{0}=a\right].$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 4: [[优化目标/约束]]

$$\hat{Q}_{\mathrm{BC}}=V_{\mathrm{BC}}(s)+\alpha\log\pi_{\mathrm{BC}}(a\mid s)+\alpha\mathcal{H}[\pi_{\mathrm{BC}}(\cdot\mid s)].$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 5: [[动力学或策略机制]]

$$\log\pi(a\mid s)=\frac{1}{2}\sum^{d}_{i=1}\left[\frac{(a_{i}-\mu_{i}(s))^{2}}{\sigma^{2}_{i}(s)}+\log(2\pi\sigma^{2}_{i}(s))\right],$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 6: [[优化目标/约束]]

$$\mathcal{H}\!\left[\pi(\cdot\mid s)\right]=\frac{1}{2}\,\log\!\left(2\pi e\,\sigma^{2}(s)\right).$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 7: [[动力学或策略机制]]

$$a=\begin{cases}a_{\mathrm{BC}}\sim\pi_{\mathrm{BC}}(s),&\hat{Q}_{BC}(s,a_{\mathrm{BC}})>Q_{RL}(s,a_{\mathrm{RL}})\\ a_{\mathrm{RL}}\sim\pi_{\mathrm{RL}}(s_{t}),&\mathrm{otherwise}.\end{cases}$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 8: [[优化目标/约束]]

$$\mathcal{H}[\pi(\cdot\mid s)\mid C]=\sum_{i}c_{i}\mathcal{H}[\pi_{i}(\cdot\mid s)]$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 9: [[动力学或策略机制]]

$$\hat{V}_{BC}(s_{t}^{(i)})=\frac{1}{N}\sum_{i=1}^{N}\sum_{k=t}^{\tau_{i}}\gamma^{k}r_{t+k}^{(i)},\ \text{for}t\in\{1,\dots,\tau_{i}\}.$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 10: [[动力学或策略机制]]

$$=-\mathbb{E}_{a\sim\pi_{BC}(\cdot\mid s)}\left[\frac{1}{\alpha}Q(s,a)-\log Z(s)\right]$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。
## 关键图表

### Figure 1: Pipeline / core system figure: Tasks from D4RL (Kitchen-Complete, Adroit-Pen, Adroit-Door) and robomimic (Lift

![Figure 1](https://arxiv.org/html/2605.05172v2/figs/Task_Vizulaizations.jpg)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Pipeline / core system figure: Tasks from D4RL (Kitchen-Complete, Adroit-Pen, Adroit-Door) and robomimic (Lift”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 2: Results on D4RL. Plots begin when a method starts online interaction. The dotted blue

![Figure 2](https://arxiv.org/html/2605.05172v2/figs/d4rl.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Results on D4RL. Plots begin when a method starts online interaction. The dotted blue”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 3: On-Robot RL Results. Task success is reported 20 trials per method. Q2RL signifi

![Figure 3](https://arxiv.org/html/2605.05172v2/figs/real_world_results.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“On-Robot RL Results. Task success is reported 20 trials per method. Q2RL signifi”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### 论文表格

#### Table 1: TABLE I: BC Epochs and Auxiliary Loss Weights for RL

| | BC Loss Weight | BC Epoch Used |
| --- | --- | --- |
| Kitchen | 0.3 | 200k |
| Pen | 0.3 | 200k |
| Door | 0.3 | 100k |
| Lift-State | 0.3 | 350 |
| Can-State | 0.3 | 250 |
| Square-State | 0.2 | 1000 |
| Lift-Image | 0.2 | 20 |
| Can-Image | 0.2 | 15 |
| Peg Insertion | 0.1 | 600 |
| Pipe Assembly | 0.1 | 600 |
| Kitting (Modified) | 0.1 | 600 |

**说明**: TABLE I: BC Epochs and Auxiliary Loss Weights for RL

#### Table 2: TABLE II: Reinforcement Learning Agent Hyperparameters

| | Simulation (State) | Simulation (Image) | Real-world |
| --- | --- | --- | --- |
| Batch Size | 256 (Kitchen: 1024) | 256 | 256 |
| Hidden Dimensions (Actor and Critic) | [512, 512, 512] | [1024, 1024] | [1024, 1024] |
| Learning Rate (Actor and Critic) | 3e-4 | 1e-4 | 1e-4 |
| Reward Scale | 5 (Adroit: 10, Kitchen: 4) | 5 | 10 |
| Num. Rollouts for Q-Estimation Phase | 50 (D4RL), 100 (Robomimic) | 100 | See Table V |
| Reward Bias | -1 (Adroit: 5) | | |
| Num. Q-Estimation Training Steps | 20k (50k for Square) | | |
| UTD Ratio | 4 | | |
| Critic Ensemble Size | 10 | | |
| Critic Subsample Size | 2 | | |
| Discount Factor | 0.99 | | |
| Soft Target Update Rate | 0.005 | | |
| Layer Normalization | Yes | | |
| Replay Buffer Size | 2e6 | | |

**说明**: TABLE II: Reinforcement Learning Agent Hyperparameters

#### Table 3: TABLE III: Simulation Results - State

| Without Data | Kitchen | Pen | Door | | | | | | |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Method | 100k | 200k | 300k | 0 | 150k | 300k | 100k | 200k | 300k |
| BC Policy | 0.69 | 0.69 | 0.69 | 0.9 | 0.9 | 0.9 | 0.5 | 0.5 | 0.5 |
| WSRL | 0.31 ± 0.09 0.31\pm 0.09 | 0.65 ± 0.09 0.65\pm 0.09 | 0.64 ± 0.08 0.64\pm 0.08 | 0.78 ± 0.06 0.78\pm 0.06 | 0.92 ± 0.05  $\mathbf{0.92\pm 0.05}$ | 0.98 ± 0.01  $\mathbf{0.98\pm 0.01}$ | 0.91 ± 0.11  $\mathbf{0.91\pm 0.11}$ | 0.98 ± 0.01  $\mathbf{0.98\pm 0.01}$ | 1.0  $\mathbf{1.0}$ |
| CQL | 0.0 | 0.0 |.06 ± 0.08.06\pm 0.08 | 0.7 ± 0.08 0.7\pm 0.08 | 0.12 ± 0.03 0.12\pm 0.03 | 0.05 ± 0.03 0.05\pm 0.03 | 0.0 | 0.02 ± 0.03 0.02\pm 0.03 | 0.31 ± 0.33 0.31\pm 0.33 |
| CalQL | 0.07 ± 0.07 0.07\pm 0.07 | 0.19 ± 0.07 0.19\pm 0.07 | 0.23 ± 0.02 0.23\pm 0.02 | 0.71 ± 0.11 0.71\pm 0.11 | 0.87 ± 0.02  $\mathbf{0.87\pm 0.02}$ | 0.93 ± 0.03  $\mathbf{0.93\pm 0.03}$ | 0.0 0.0 | 0.0 0.0 | 0.15 ± 0.15 0.15\pm 0.15 |
| IBRL | 0.35 ± 0.14 0.35\pm 0.14 | 0.43 ± 0.18 0.43\pm 0.18 | 0.50 ± 0.15 0.50\pm 0.15 | 0.1 ± 0.15 0.1\pm 0.15 | 0.82 ± 0.02 0.82\pm 0.02 | 0.95 ± 0.05  $\mathbf{0.95\pm 0.05}$ | 0.0 0.0 | 0.0 0.0 | 0.0 0.0 |
| Q2RL (Ours) | 0.85 ± 0.06  $\mathbf{0.85\pm 0.06}$ | 0.87 ± 0.03  $\mathbf{0.87\pm 0.03}$ | 0.91 ± 0.01  $\mathbf{0.91\pm 0.01}$ | 0.88 ± 0.04  $\mathbf{0.88\pm 0.04}$ | 0.91 ± 0.05  $\mathbf{0.91\pm 0.05}$ | 0.93 ± 0.03  $\mathbf{0.93\pm 0.03}$ | 0.55 ± 0.15 0.55\pm 0.15 | 0.73 ± 0.12 0.73\pm 0.12 | 0.87 ± 0.08 0.87\pm 0.08 |
| With Data | Lift-State | Can-State | Square-State | | | | | | |
| Method | 100k | 200k | 300k | 100k | 300k | 500k | 100k | 300k | 500k |
| BC Policy | 0.58 0.58 | 0.58 0.58 | 0.58 0.58 | 0.6 0.6 | 0.6 0.6 | 0.6 0.6 | 0.58  $\mathbf{0.58}$ | 0.58 0.58 | 0.58 0.58 |
| RLPD | 0.87 ± 0.07 0.87\pm 0.07 | 0.98 ± 0.02  $\mathbf{0.98\pm 0.02}$ | 0.99 ± 0.01  $\mathbf{0.99\pm 0.01}$ | 0.0 0.0 | 0.0 0.0 | 0.0 0.0 | 0.0 0.0 | 0.0 0.0 | 0.0 0.0 |
| IBRL | 0.97 ± 0.03  $\mathbf{0.97\pm 0.03}$ | 1.0  $\mathbf{1.0}$ | 0.98 ± 0.02  $\mathbf{0.98\pm 0.02}$ | 0.02 ± 0.02 0.02\pm 0.02 | 0.32 ± 0.16 0.32\pm 0.16 | 0.54 ± 0.18 0.54\pm 0.18 | 0.0 0.0 | 0.72 ± 0.11  $\mathbf{0.72\pm 0.11}$ | 0.94 ± 0.01  $\mathbf{0.94\pm 0.01}$ |
| Q2RL (Ours) | 0.88 ± 0.06 0.88\pm 0.06 | 1.0  $\mathbf{1.0}$ | 1.0  $\mathbf{1.0}$ | 0.76 ± 0.05  $\mathbf{0.76\pm 0.05}$ | 0.86 ± 0.03  $\mathbf{0.86\pm 0.03}$ | 0.85 ± 0.03  $\mathbf{0.85\pm 0.03}$ | 0.60 ± 0.07  $\mathbf{0.60\pm 0.07}$ | 0.78 ± 0.06  $\mathbf{0.78\pm 0.06}$ | 0.81 ± 0.08 0.81\pm 0.08 |
| Without Data | Lift-State | Can-State | Square-State | | | | | | |
| Method | 100k | 200k | 300k | 100k | 300k | 550k | 100k | 300k | 550k |
| BC Policy | 0.58 0.58 | 0.58 0.58 | 0.58 0.58 | 0.6  $\mathbf{0.6}$ | 0.6 0.6 | 0.6 0.6 | 0.58  $\mathbf{0.58}$ | 0.58 0.58 | 0.58 0.58 |
| WSRL | 0.01 ± 0.01 0.01\pm 0.01 | 0.0 0.0 | 0.0 0.0 | 0.0 0.0 | 0.0 0.0 | 0.0 0.0 | 0.0 0.0 | 0.0 0.0 | 0.0 0.0 |
| CQL | 0.0 0.0 | 0.0 0.0 | 0.0 0.0 | 0.0 0.0 | 0.0 0.0 | 0.0 0.0 | 0.0 0.0 | 0.0 0.0 | 0.0 0.0 |
| CalQL | 0.0 0.0 | 0.0 0.0 | 0.0 0.0 | 0.0 0.0 | 0.0 0.0 | 0.0 0.0 | 0.0 0.0 | 0.0 0.0 | 0.0 0.0 |
| IBRL | 0.01 ± 0.01 0.01\pm 0.01 | 0.0 0.0 | 0.0 0.0 | 0.0 0.0 | 0.0 0.0 | 0.0 0.0 | 0.0 0.0 | 0.0 0.0 | 0.0 0.0 |
| Q2RL (Ours) | 0.86 ± 0.08  $\mathbf{0.86\pm 0.08}$ | 0.96 ± 0.05  $\mathbf{0.96\pm 0.05}$ | 1.0  $\mathbf{1.0}$ | 0.57 ± 0.12  $\mathbf{0.57\pm 0.12}$ | 0.75 ± 0.13  $\mathbf{0.75\pm 0.13}$ | 0.82 ± 0.10  $\mathbf{0.82\pm 0.10}$ | 0.51 ± 0.15  $\mathbf{0.51\pm 0.15}$ | 0.68 ± 0.06  $\mathbf{0.68\pm 0.06}$ | 0.76 ± 0.06  $\mathbf{0.76\pm 0.06}$ |

**说明**: TABLE III: Simulation Results - State

#### Table 4: TABLE IV: Simulation Results - Image

| With Data | Lift-Image | Can-Image | | | |
| --- | --- | --- | --- | --- | --- |
| Method | 100k | 200k | 100k | 300k | 500k |
| BC Policy | 0.6 0.6 | 0.6 0.6 | 0.45 0.45 | 0.45 0.45 | 0.45 0.45 |
| RLPD | 0.0 0.0 | 0.0 0.0 | 0.0 0.0 | 0.0 0.0 | 0.0 0.0 |
| IBRL | 1.0  $\mathbf{1.0}$ | 1.0  $\mathbf{1.0}$ | 0.12 ± 0.13 0.12\pm 0.13 | 0.01 ± 0.01 0.01\pm 0.01 | 0.03 ± 0.05 0.03\pm 0.05 |
| Q2RL (Ours) | 0.93 ± 0.05 0.93\pm 0.05 | 1.0  $\mathbf{1.0}$ | 0.53 ± 0.07  $\mathbf{0.53\pm 0.07}$ | 0.63 ± 0.02  $\mathbf{0.63\pm 0.02}$ | 0.73 ± 0.06  $\mathbf{0.73\pm 0.06}$ |
| Without Data | Lift-Image | Can-Image | | | |
| Method | 100k | 200k | 100k | 300k | 550k |
| BC Policy | 0.6 0.6 | 0.6 0.6 | 0.45  $\mathbf{0.45}$ | 0.45 0.45 | 0.45 0.45 |
| IBRL | 0 | 0.01 ± 0.01 0.01\pm 0.01 | 0.0 0.0 | 0.0 0.0 | 0.0 0.0 |
| Q2RL (Ours) | 0.87 ± 0.05  $\mathbf{0.87\pm 0.05}$ | 0.98 ± 0.01  $\mathbf{0.98\pm 0.01}$ | 0.45 ± 0.07  $\mathbf{0.45\pm 0.07}$ | 0.57 ± 0.10  $\mathbf{0.57\pm 0.10}$ | 0.63 ± 0.05  $\mathbf{0.63\pm 0.05}$ |

**说明**: TABLE IV: Simulation Results - Image

#### Table 5: TABLE V: Real World Experiment Configurations

| | Peg Insertion | Pipe Assembly | Kitting (Modified) |
| --- | --- | --- | --- |
| Initial Pose Distribution ( $\mathrm{unit}, \mathrm{unit})$ | ± \pm 0.045, ± \pm 0.07875 | fixed | fixed |
| Delta Action Range ( $\mathrm{unit}, \mathrm{unit})$ | ± \pm 0.015, ± \pm 0.02625 | ± \pm 0.015, ± \pm 0.02625 | ± \pm 0.015, ± \pm 0.02625 |
| Safety Limit XYZ ( $\mathrm{unit})$ | [-0.02, 0.05], [-0.1, 0.02], [-0.13, 0.02] | [-0.2, 0.1], [-0.2, 0.2], [-0.4, 0.05] | [-0.2, 0.3], [-0.2, 0.2], [-0.18, 0.05] |
| Safety Limit RPY ( $\mathrm{unit})$ | [-5, 5], [-5, 5], [-20, 20] | [-5, 5], [-5, 5], [-20, 20] | [-5, 5], [-5, 5], [-20, 20] |
| Num. Demos for BC Training | 50 | 100 | 50 |
| Num. Rollouts for Q-Estimation Phase | 100 | 100 | 30 |
| Max Episode Length | 200 | 200 | 500 |
| Cartesian Stiffness (XYZRPY) | | [1024, 1024, 1024, 100, 100, 100] | |
| Cartesian Damping (XYZRPY) | | [64, 64, 64, 10, 10, 10] | |
| Policy Hz | | 10 | |

**说明**: TABLE V: Real World Experiment Configurations

#### Table 6: TABLE VI: Real-World Evaluation Results Success Rate 20 trials. Results are reported for the best evaluated checkp

| | Peg Insertion | Pipe Assembly | Kitting-Modified | | | |
| --- | --- | --- | --- | --- | --- | --- |
| Method | Success Rate | Online Learner Steps | Success Rate | Online Learner Steps | Success Rate | Online Learner Steps |
| BC Policy | 0.70 | – | 0.20 | – | 0.35 | – |
| IBRL [2 ] | 0.95 | 40k | 0.0 | 90k | 0.0 | 165k |
| Q2RL (Ours) | 1.0 | 60k | 0.75 | 90k | 0.70 | 165k |

**说明**: TABLE VI: Real-World Evaluation Results Success Rate 20 trials. Results are reported for the best evaluated checkpoint.

#### Table 7: TABLE VII: Peg Insertion without Seeded Replay Buffer All methods either do not do online learning or start with no dat

| | Peg Insertion, No Data | |
| --- | --- | --- |
| Method | Success Rate | Online Learner Steps |
| BC Policy | 0.70 | – |
| CalQL [6 ] Deterministic | 0.10 | – |
| CalQL [6 ] Stochastic | 0.20 | – |
| IBRL [2 ] | 0.0 | 20k |
| Q2RL (Ours) | 1.00 | 20k |

**说明**: TABLE VII: Peg Insertion without Seeded Replay Buffer All methods either do not do online learning or start with no data in the online replay buffer. Success rate is 20 trials. Results are reported for the best evaluated checkpoint.
## 实验解读

- 评价重点:围绕 assembly、behavior-cloning、接触推理、接触丰富操作、模仿学习,优先看真实机器人结果、扰动恢复、接触稳定性、控制误差、成功率、实时性和 sim-to-real 差距。
- System0 视角:实验结论不只看最终成功率,还要看闭环过程中是否能处理状态估计误差、执行延迟、碰撞/接触突变和外部扰动。
- 与上层系统的关系:如果方法能在快闭环内稳定工作,它可以作为 System1 技能或 System2 规划的底层执行保障;如果只在离线或慢速设定下成立,则还不能直接作为身体级快闭环模块。
## 批判性思考

### 优点

- 论文把 assembly、behavior-cloning、接触推理、接触丰富操作、模仿学习 放到机器人身体级闭环里讨论,和只做高层语义推理的工作相比,更接近真实机器人部署时会遇到的状态、动力学和安全约束。
- 若论文同时给出公式、系统结构图、硬件实验或消融结果,这些证据可以帮助判断方法是否真的能支撑毫秒到百毫秒级反应。

### 局限

- 需要重点核查控制频率、推理延迟、传感延迟、失败恢复、硬件磨损、未见场景泛化和安全边界;这些因素决定它能否成为可靠的 System0 模块。
- 对学习型方法,还要区分训练环境中的鲁棒性和真实部署中的鲁棒性,尤其是接触丰富任务、动态运动任务和人形机器人任务。
## 论文定位

- 研究对象:When Life Gives You BC, Make Q-functions: Extracting Q-values from Behavior Cloning for On-Robot Reinforcement Learning。
- 关键词:assembly、behavior-cloning、接触推理、接触丰富操作、模仿学习、实时控制、强化学习、机器人操作、鲁棒控制。
- System0 关联:关注机器人身体级快反应链路中“状态/接触/局部感知 -> 约束/优化/策略 -> 动作”的短闭环,不把问题停留在语言规划或任务分解层。

## 速查卡片

> [!summary] When Life BC
> - **论文**: https://www.roboticsproceedings.org/rss22/p153.pdf
> - **arXiv**: http://arxiv.org/abs/2605.05172v2
> - **arXiv HTML**: https://arxiv.org/html/2605.05172v2
