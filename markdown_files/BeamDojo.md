---
title: "BeamDojo: Learning Agile Humanoid Locomotion on Sparse Footholds"
method_name: "BeamDojo"
authors: ["Huayi Wang"]
year: 2025
venue: "RSS"
tags: ["robust-control", "legged-locomotion", "reinforcement-learning", "humanoid", "agile-locomotion"]
image_source: "online"
arxiv_html: "https://arxiv.org/html/2502.10363v3"
---
# BeamDojo
## 一句话总结

> BeamDojo: Learning Agile Humanoid Locomotion on Sparse Footholds 主要落在 [[agile-locomotion]]、[[foothold-tracking]]、[[人形机器人]]、[[足式运动]] 这一类问题上;从 System0 视角看,它的价值在于把身体状态、接触变化、局部感知或动力学约束接到机器人低层闭环里,让动作生成更稳定、安全、实时。
## 核心贡献

1. 围绕 **BeamDojo: Learning Agile Humanoid Locomotion on Sparse Footholds** 建立了一个与 agile-locomotion、foothold-tracking、人形机器人、足式运动、强化学习、鲁棒控制 相关的机器人问题设定,重点不是高层语义规划,而是身体级闭环中的状态变化、接触变化或控制约束。
2. 方法层面的关键价值在于把感知、动力学、约束优化、策略学习或硬件反馈组织成可执行的低层动作生成机制,适合放进 System0 的快反应模块中理解。
3. 对系统集成而言,这类工作补上了 System1 技能选择与电机级控制之间的一层:它负责把上层目标变成动态可行、可恢复、可安全过滤的动作。
4. 若论文包含真实机器人、接触任务或仿真到真实迁移,核心应看它是否报告了闭环稳定性、扰动恢复、硬件延迟、失败模式和跨场景泛化。
## 问题背景

这篇论文可以放在 System0 的“具身快系统”背景下阅读:机器人需要在很短时间内根据本体状态、接触、力/触觉、局部视觉或动力学预测做出动作,而不是等待慢速规划器重新推理整个任务。agile-locomotion、foothold-tracking、人形机器人、足式运动、强化学习、鲁棒控制、terrain-adaptation 相关问题的难点通常不在“知道要做什么”,而在动作执行过程中状态会快速偏离、接触会突然改变、约束会实时收紧,低层系统必须及时修正。

### System0 关联

精读时优先看三件事:第一,论文使用哪些快速反馈量,例如关节状态、接触状态、力觉、触觉、深度或局部视觉;第二,反馈如何进入 MPC、WBC、策略网络、安全过滤器或恢复控制器;第三,输出是否直接影响机器人下一步动作,而不是只服务离线分析或高层任务规划。
## 方法详解

### System0 视角解析

从 System0 角度看,这篇论文的方法部分应当围绕 agile-locomotion、foothold-tracking、人形机器人、足式运动、强化学习、鲁棒控制、terrain-adaptation 来读。核心问题是:系统如何从快速变化的机器人状态中构造控制输入,如何处理接触、碰撞、约束、模型误差或执行延迟,以及如何把这些信息变成下一时刻的动作。

如果方法里有优化问题,重点看目标函数、约束和求解频率;如果方法里有学习策略,重点看观测空间、动作空间、训练扰动和部署时的闭环频率;如果方法里有硬件或传感器,重点看传感链路是否真的进入控制回路。这样读可以把论文和 System2 语言规划、System1 技能编排区分开来。

### 关键公式与机制

#### 公式 1: [[动力学或策略机制]]

$$-\sum_{i=1}^{2}\mathbb{C}_{i}\sum_{j=1}^{n}\mathbf{1}\{d_{ij}<\epsilon\}$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 2: [[优化目标/约束]]

$$\mathcal{L}(\theta)=\mathbb{E}\left[\min\left(\alpha_{t}(\theta)\hat{A}_{t},\text{clip}(\alpha_{t}(\theta),1-\epsilon,1+\epsilon)\hat{A}_{t}\right)\right],$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 3: [[优化目标/约束]]

$$\max_{\pi}J(\mathcal{M},\pi)=\mathbb{E}\left[\sum_{t=0}^{\infty}\gamma^{t}r(s_{t},a_{t})\right].$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 4: [[动力学或策略机制]]

$$r_{\text{foothold}-p\}=-\sum_{i=1}^{2}\mathbb{C}_{i}\cdot\mathbf{1}\left\{\left(\sum_{j=1}^{n}\mathbf{1}\{d_{ij}<\epsilon\}\right)\geq p\\cdot n\right\}.$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 5: [[动力学或策略机制]]

$$\sum_{i=1}^{2}\left(\mathbf{p}_{z,i}-\mathbf{p}_{z}^{\text{target}}\right)^{2}\cdot\dot{\mathbf{p}}_{xy,i}$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 6: [[优化目标/约束]]

$$\mathcal{L}(\phi_{i})=\mathbb{E}\left[\left\lVert R_{i,t}+\gamma V_{\phi_{i}}(s_{t+1})-V_{\phi_{i}}(s_{t})\right \rVert^{2}\right],$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 7: [[动力学或策略机制]]

$${| $\boldsymbol{\tau}\lVert \dot{\boldsymbol{\theta}}$ |^{T}}/\left(\rVert\mathbf{v}\lVert_{2}^{2}+0.2*\rVert\boldsymbol{\omega}\lVert_{2}^{2}\right)$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 8: [[动力学或策略机制]]

$$\sum_{i=1}^{2}\left(t_{\text{air},i}-t_{\text{air}}^{\text{target}}\right)\cdot\mathbb{F}_{i}$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 9: [[动力学或策略机制]]

$$\mathbf{o}_{t}=\left[\mathbf{c}_{t},\mathbf{o}^{\text{proprio}}_{t},\mathbf{o}_{t}^{\text{percept}},\mathbf{a}_{t-1}\right].$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 10: [[动力学或策略机制]]

$$r_{\text{foothold}}=-\sum_{i=1}^{2}\mathbb{C}_{i}\sum_{j=1}^{n}\mathbf{1}\{d_{ij}<\epsilon\},$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。
## 关键图表

### Figure 1: Pipeline / core system figure: Foothold Reward. We sample n n n points the foot. Green points indicat

![Figure 1](https://arxiv.org/html/2502.10363v3/extracted/6391328/figures/foothold_reward.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Pipeline / core system figure: Foothold Reward. We sample n n n points the foot. Green points indicat”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 2: Overview of BeamDojo. (a) Training in Simulation: In stage 1, proprioceptive and per

![Figure 2](https://arxiv.org/html/2502.10363v3/extracted/6391328/figures/framework.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Overview of BeamDojo. (a) Training in Simulation: In stage 1, proprioceptive and per”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 3: Real-world Experiments. We build terrains in the real world similar to those in simul

![Figure 3](https://arxiv.org/html/2502.10363v3/extracted/6391328/figures/real_results/gaps.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Real-world Experiments. We build terrains in the real world similar to those in simul”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### 论文表格

#### Table 1: TABLE I: Commands Sampled in Two Stage RL Training

| Term | Value (stage 1) | Value (stage 2) |
| --- | --- | --- |
| v x c v x c  $\mathbf{v}_{x}^{c}$ | U (- 1.0, 1.0) U 1.0 1.0  $\mathcal{U}(-1.0,1.0) (- 1.0, 1.0) m/s$ | U (- 1.0, 1.0) U 1.0 1.0  $\mathcal{U}(-1.0,1.0) (- 1.0, 1.0) m/s$ |
| v y c v y c  $\mathbf{v}_{y}^{c}$ | U (- 1.0, 1.0) U 1.0 1.0  $\mathcal{U}(-1.0,1.0) (- 1.0, 1.0) m/s$ | U (0.0, 0.0) U 0.0 0.0  $\mathcal{U}(0.0,0.0) (0.0, 0.0) m/s$ |
| $\boldsymbol{\omega}_{\text{yaw}}^{c} yaw$ | U (- 1.0, 1.0) U 1.0 1.0  $\mathcal{U}(-1.0,1.0) (- 1.0, 1.0) rad/s$ | U (0.0, 0.0) U 0.0 0.0  $\mathcal{U}(0.0,0.0) (0.0, 0.0) m/s$ |

**说明**: TABLE I: Commands Sampled in Two Stage RL Training

#### Table 2: TABLE II: Benchmarked Comparison in Simulation.

| | Stepping Stones | Balancing Beams | Stepping Beams | Gaps | | | | |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R succ R succ R_{ $\mathrm{succ}} (, ↑ \,\uparrow , ↑)$ | R trav R trav R_{ $\mathrm{trav}} (, ↑ \,\uparrow , ↑)$ | R succ R succ R_{ $\mathrm{succ}} (, ↑ \,\uparrow , ↑)$ | R trav R trav R_{ $\mathrm{trav}} (, ↑ \,\uparrow , ↑)$ | R succ R succ R_{ $\mathrm{succ}} (, ↑ \,\uparrow , ↑)$ | R trav R trav R_{ $\mathrm{trav}} (, ↑ \,\uparrow , ↑)$ | R succ R succ R_{ $\mathrm{succ}} (, ↑ \,\uparrow , ↑)$ | R trav R trav R_{ $\mathrm{trav}} (, ↑ \,\uparrow , ↑)$ | |
| Medium Terrain Difficulty | | | | | | | | |
| PIM | 71.00 71.00 71.00 71.00 (± 1.53 plus-or-minus 1.53 \pm 1.53 ± 1.53) | 78.29 78.29 78.29 78.29 (± 2.49 plus-or-minus 2.49 \pm 2.49 ± 2.49) | 74.67 74.67 74.67 74.67 (± 2.08 plus-or-minus 2.08 \pm 2.08 ± 2.08) | 82.19 82.19 82.19 82.19 (± 4.96 plus-or-minus 4.96 \pm 4.96 ± 4.96) | 88.33 88.33 88.33 88.33 (± 3.61 plus-or-minus 3.61 \pm 3.61 ± 3.61) | 93.16 93.16 93.16 93.16 (± 4.78 plus-or-minus 4.78 \pm 4.78 ± 4.78) | 98.00 98.00  $\mathbf{98.00}.00 (± 0.57 plus-or-minus 0.57 \pm 0.57 ± 0.57)$ | 99.16 99.16 99.16 99.16 (± 0.75 plus-or-minus 0.75 \pm 0.75 ± 0.75) |
| Naive | 48.33 48.33 48.33 48.33 (± 6.11 plus-or-minus 6.11 \pm 6.11 ± 6.11) | 47.79 47.79 47.79 47.79 (± 5.76 plus-or-minus 5.76 \pm 5.76 ± 5.76) | 57.00 57.00 57.00 57.00 (± 7.81 plus-or-minus 7.81 \pm 7.81 ± 7.81) | 71.59 71.59 71.59 71.59 (± 8.14 plus-or-minus 8.14 \pm 8.14 ± 8.14) | 92.00 92.00 92.00 92.00 (± 2.52 plus-or-minus 2.52 \pm 2.52 ± 2.52) | 92.67 92.67 92.67 92.67 (± 3.62 plus-or-minus 3.62 \pm 3.62 ± 3.62) | 95.33 95.33 95.33 95.33 (± 1.53 plus-or-minus 1.53 \pm 1.53 ± 1.53) | 98.41 98.41 98.41 98.41 (± 0.67 plus-or-minus 0.67 \pm 0.67 ± 0.67) |
| Ours w/o Soft Dyn | 65.33 65.33 65.33 65.33 (± 2.08 plus-or-minus 2.08 \pm 2.08 ± 2.08) | 74.62 74.62 74.62 74.62 (± 1.37 plus-or-minus 1.37 \pm 1.37 ± 1.37) | 79.00 79.00 79.00 79.00 (± 2.64 plus-or-minus 2.64 \pm 2.64 ± 2.64) | 82.67 82.67 82.67 82.67 (± 2.92 plus-or-minus 2.92 \pm 2.92 ± 2.92) | 98.67 98.67  $\mathbf{98.67}.67 (± 2.31 plus-or-minus 2.31 \pm 2.31 ± 2.31)$ | 99.64 99.64  $\mathbf{99.64}.64 (± 0.62 plus-or-minus 0.62 \pm 0.62 ± 0.62)$ | 96.33 96.33 96.33 96.33 (± 1.53 plus-or-minus 1.53 \pm 1.53 ± 1.53) | 98.60 98.60 98.60 98.60 (± 1.15 plus-or-minus 1.15 \pm 1.15 ± 1.15) |
| Ours w/o Double Critic | 83.00 83.00 83.00 83.00 (± 2.00 plus-or-minus 2.00 \pm 2.00 ± 2.00) | 86.64 86.64 86.64 86.64 (± 1.96 plus-or-minus 1.96 \pm 1.96 ± 1.96) | 88.67 88.67 88.67 88.67 (± 2.65 plus-or-minus 2.65 \pm 2.65 ± 2.65) | 90.21 90.21 90.21 90.21 (± 1.95 plus-or-minus 1.95 \pm 1.95 ± 1.95) | 96.33 96.33 96.33 96.33 (± 1.15 plus-or-minus 1.15 \pm 1.15 ± 1.15) | 98.88 98.88 98.88 98.88 (± 1.21 plus-or-minus 1.21 \pm 1.21 ± 1.21) | 98.00 98.00  $\mathbf{98.00}.00 (± 1.00 plus-or-minus 1.00 \pm 1.00 ± 1.00)$ | 99.33 99.33  $\mathbf{99.33}.33 (± 0.38 plus-or-minus 0.38 \pm 0.38 ± 0.38)$ |
| BeamDojo | 95.67 95.67  $\mathbf{95.67}.67 (± 1.53 plus-or-minus 1.53 \pm 1.53 ± 1.53)$ | 96.11 96.11  $\mathbf{96.11}.11 (± 1.22 plus-or-minus 1.22 \pm 1.22 ± 1.22)$ | 98.00 98.00  $\mathbf{98.00}.00 (± 2.00 plus-or-minus 2.00 \pm 2.00 ± 2.00)$ | 99.91 99.91  $\mathbf{99.91}.91 (± 0.07 plus-or-minus 0.07 \pm 0.07 ± 0.07)$ | 98.33 98.33 98.33 98.33 (± 1.15 plus-or-minus 1.15 \pm 1.15 ± 1.15) | 99.28 99.28 99.28 99.28 (± 0.65 plus-or-minus 0.65 \pm 0.65 ± 0.65) | 98.00 98.00  $\mathbf{98.00}.00 (± 2.65 plus-or-minus 2.65 \pm 2.65 ± 2.65)$ | 99.21 99.21 99.21 99.21 (± 1.24 plus-or-minus 1.24 \pm 1.24 ± 1.24) |
| Hard Terrain Difficulty | | | | | | | | |
| PIM | 46.67 46.67 46.67 46.67 (± 2.31 plus-or-minus 2.31 \pm 2.31 ± 2.31) | 52.88 52.88 52.88 52.88 (± 2.86 plus-or-minus 2.86 \pm 2.86 ± 2.86) | 33.00 33.00 33.00 33.00 (± 2.31 plus-or-minus 2.31 \pm 2.31 ± 2.31) | 45.2 45.2 45.2 45.2 8 (± 3.64 plus-or-minus 3.64 \pm 3.64 ± 3.64) | 82.67 82.67 82.67 82.67 (± 2.31 plus-or-minus 2.31 \pm 2.31 ± 2.31) | 90.68 90.68 90.68 90.68 (± 1.79 plus-or-minus 1.79 \pm 1.79 ± 1.79) | 96.00 96.00  $\mathbf{96.00}.00 (± 1.00 plus-or-minus 1.00 \pm 1.00 ± 1.00)$ | 98.27 98.27  $\mathbf{98.27}.27 (± 3.96 plus-or-minus 3.96 \pm 3.96 ± 3.96)$ |
| Naive | 00.33 00.33 00.33 00.33 (± 0.57 plus-or-minus 0.57 \pm 0.57 ± 0.57) | 21.17 21.17 21.17 21.17 (± 1.71 plus-or-minus 1.71 \pm 1.71 ± 1.71) | 00.67 00.67 00.67 00.67 (± 1.15 plus-or-minus 1.15 \pm 1.15 ± 1.15) | 36.25 36.25 36.25 36.25 (± 7.85 plus-or-minus 7.85 \pm 7.85 ± 7.85) | 82.00 82.00 82.00 82.00 (± 3.61 plus-or-minus 3.61 \pm 3.61 ± 3.61) | 88.91 88.91 88.91 88.91 (± 3.75 plus-or-minus 3.75 \pm 3.75 ± 3.75) | 31.00 31.00 31.00 31.00 (± 3.61 plus-or-minus 3.61 \pm 3.61 ± 3.61) | 62.70 62.70 62.70 62.70 (± 4.08 plus-or-minus 4.08 \pm 4.08 ± 4.08) |
| Ours w/o Soft Dyn | 42.00 42.00 42.00 42.00 (± 6.56 plus-or-minus 6.56 \pm 6.56 ± 6.56) | 47.09 47.09 47.09 47.09 (± 6.97 plus-or-minus 6.97 \pm 6.97 ± 6.97) | 51.00 51.00 51.00 51.00 (± 4.58 plus-or-minus 4.58 \pm 4.58 ± 4.58) | 72.93 72.93 72.93 72.93 (± 4.38 plus-or-minus 4.38 \pm 4.38 ± 4.38) | 87.33 87.33 87.33 87.33 (± 2.08 plus-or-minus 2.08 \pm 2.08 ± 2.08) | 89.41 89.41 89.41 89.41 (± 1.75 plus-or-minus 1.75 \pm 1.75 ± 1.75) | 93.00 93.00 93.00 93.00 (± 1.00 plus-or-minus 1.00 \pm 1.00 ± 1.00) | 95.62 95.62 95.62 95.62 (± 2.50 plus-or-minus 2.50 \pm 2.50 ± 2.50) |
| Ours w/o Double Critic | 55.67 55.67 55.67 55.67 (± 3.61 plus-or-minus 3.61 \pm 3.61 ± 3.61) | 60.95 60.95 60.95 60.95 (± 2.67 plus-or-minus 2.67 \pm 2.67 ± 2.67) | 70.33 70.33 70.33 70.33 (± 3.06 plus-or-minus 3.06 \pm 3.06 ± 3.06) | 85.64 85.64 85.64 85.64 (± 3.24 plus-or-minus 3.24 \pm 3.24 ± 3.24) | 94.67 94.67 94.67 94.67 (± 1.53 plus-or-minus 1.53 \pm 1.53 ± 1.53) | 96.57 96.57 96.57 96.57 (± 1.42 plus-or-minus 1.42 \pm 1.42 ± 1.42) | 94.33 94.33 94.33 94.33 (± 3.06 plus-or-minus 3.06 \pm 3.06 ± 3.06) | 95.62 95.62 95.62 95.62 (± 2.50 plus-or-minus 2.50 \pm 2.50 ± 2.50) |
| BeamDojo | 91.67 91.67  $\mathbf{91.67}.67 (± 1.33 plus-or-minus 1.33 \pm 1.33 ± 1.33)$ | 94.26 94.26  $\mathbf{94.26}.26 (± 2.08 plus-or-minus 2.08 \pm 2.08 ± 2.08)$ | 94.33 94.33  $\mathbf{94.33}.33 (± 1.53 plus-or-minus 1.53 \pm 1.53 ± 1.53)$ | 95.15 95.15  $\mathbf{95.15}.15 (± 1.82 plus-or-minus 1.82 \pm 1.82 ± 1.82)$ | 97.67 97.67  $\mathbf{97.67}.67 (± 2.08 plus-or-minus 2.08 \pm 2.08 ± 2.08)$ | 98.54 98.54  $\mathbf{98.54}.54 (± 1.43 plus-or-minus 1.43 \pm 1.43 ± 1.43)$ | 94.33 94.33 94.33 94.33 (± 1.15 plus-or-minus 1.15 \pm 1.15 ± 1.15) | 97.00 97.00 97.00 97.00 (± 1.30 plus-or-minus 1.30 \pm 1.30 ± 1.30) |

**说明**: TABLE II: Benchmarked Comparison in Simulation.

#### Table 3: TABLE III: Gait Regularization. We conduct experiments on stepping stones and evaluate three representative gait regula

| Designs | Smoothness (↓ ↓ \downarrow ↓) | Feet Air Time (↑ ↑ \uparrow ↑) |
| --- | --- | --- |
| Naive | 1.7591 1.7591 1.7591 1.7591 (± 0.1316 plus-or-minus 0.1316 \pm 0.1316 ± 0.1316) | - 0.0319 0.0319 -0.0319 - 0.0319 (± 0.0028 plus-or-minus 0.0028 \pm 0.0028 ± 0.0028) |
| Ours w/o Soft Dyn | 0.9633 0.9633 0.9633 0.9633 (± 0.0526 plus-or-minus 0.0526 \pm 0.0526 ± 0.0526) | - 0.0169 0.0169  $\mathbf{-0.0169} -.0169 (± 0.0014 plus-or-minus 0.0014 \pm 0.0014 ± 0.0014)$ |
| Ours w/o Double Critic | 1.2705 1.2705 1.2705 1.2705 (± 0.1168 plus-or-minus 0.1168 \pm 0.1168 ± 0.1168) | - 0.0229 0.0229 -0.0229 - 0.0229 (± 0.0033 plus-or-minus 0.0033 \pm 0.0033 ± 0.0033) |
| BeamDojo | 0.7603 0.7603  $\mathbf{0.7603}.7603 (± 0.0315 plus-or-minus 0.0315 \pm 0.0315 ± 0.0315)$ | - 0.0182 0.0182 -0.0182 - 0.0182 (± 0.0027 plus-or-minus 0.0027 \pm 0.0027 ± 0.0027) |

**说明**: TABLE III: Gait Regularization. We conduct experiments on stepping stones and evaluate three representative gait regularization reward metrics: smoothness, feet air time, and feet clearance. Detailed definitions of the reward functions can be found in Table VII.

#### Table 4: TABLE IV: Agility Test. We evaluate the agility of the humanoid robot on stepping stones with a total length of 2.8m.

| v x c v x c  $\mathbf{v}_{x}^{c} (m/s)$ | Time Cost (s) | Average Speed (m/s) | Error Rate (%, ↓ ↓ \downarrow ↓) |
| --- | --- | --- | --- |
| 0.5 0.5 0.5 0.5 | 6.33 6.33 6.33 6.33 (± 0.15 plus-or-minus 0.15 \pm 0.15 ± 0.15) | 0.45 0.45 0.45 0.45 (± 0.05 plus-or-minus 0.05 \pm 0.05 ± 0.05) | 10.67 10.67 10.67 10.67 (± 4.54 plus-or-minus 4.54 \pm 4.54 ± 4.54) |
| 0.75 0.75 0.75 0.75 | 4.33 4.33 4.33 4.33 (± 0.29 plus-or-minus 0.29 \pm 0.29 ± 0.29) | 0.65 0.65 0.65 0.65 (± 0.05 plus-or-minus 0.05 \pm 0.05 ± 0.05) | 13.53 13.53 13.53 13.53 (± 6.52 plus-or-minus 6.52 \pm 6.52 ± 6.52) |
| 1.0 1.0 1.0 1.0 | 3.17 3.17 3.17 3.17 (± 0.58 plus-or-minus 0.58 \pm 0.58 ± 0.58) | 0.88 0.88 0.88 0.88 (± 0.04 plus-or-minus 0.04 \pm 0.04 ± 0.04) | 11.83 11.83 11.83 11.83 (± 8.08 plus-or-minus 8.08 \pm 8.08 ± 8.08) |
| 1.25 1.25 1.25 1.25 | 2.91 2.91 2.91 2.91 (± 0.63 plus-or-minus 0.63 \pm 0.63 ± 0.63) | 0.96 0.96 0.96 0.96 (± 0.03 plus-or-minus 0.03 \pm 0.03 ± 0.03) | 22.74 22.74 22.74 22.74 (± 5.32 plus-or-minus 5.32 \pm 5.32 ± 5.32) |
| 1.5 1.5 1.5 1.5 | 2.69 2.69 2.69 2.69 (± 0.42 plus-or-minus 0.42 \pm 0.42 ± 0.42) | 1.04 1.04 1.04 1.04 (± 0.05 plus-or-minus 0.05 \pm 0.05 ± 0.05) | 30.68 30.68 30.68 30.68 (± 6.17 plus-or-minus 6.17 \pm 6.17 ± 6.17) |

**说明**: TABLE IV: Agility Test. We evaluate the agility of the humanoid robot on stepping stones with a total length of 2.8m.

#### Table 5: TABLE V: Comparison of Different Foothold Reward Designs. The success rate and foothold error for each foothold reward

| Designs | R succ R succ R_{ $\mathrm{succ}} (, ↑ \,\uparrow , ↑)$ | E foot E foot E_{ $\mathrm{foot}} (, ↓ \,\downarrow , ↓)$ |
| --- | --- | --- |
| foothold- 30 % percent 30 30\% 30 % | 93.67 93.67 93.67 93.67 (± 1.96 plus-or-minus 1.96 \pm 1.96 ± 1.96) | 11.43 11.43 11.43 11.43 (± 0.81 plus-or-minus 0.81 \pm 0.81 ± 0.81) |
| foothold- 50 % percent 50 50\% 50 % | 92.71 92.71 92.71 92.71 (± 1.06 plus-or-minus 1.06 \pm 1.06 ± 1.06) | 10.78 10.78 10.78 10.78 (± 1.94 plus-or-minus 1.94 \pm 1.94 ± 1.94) |
| foothold- 70 % percent 70 70\% 70 % | 91.94 91.94 91.94 91.94 (± 2.08 plus-or-minus 2.08 \pm 2.08 ± 2.08) | 14.35 14.35 14.35 14.35 (± 2.61 plus-or-minus 2.61 \pm 2.61 ± 2.61) |
| BeamDojo | 95.67 95.67  $\mathbf{95.67}.67 (± 1.53 plus-or-minus 1.53 \pm 1.53 ± 1.53)$ | 7.79 7.79  $\mathbf{7.79}.79 (± 1.33 plus-or-minus 1.33 \pm 1.33 ± 1.33)$ |

**说明**: TABLE V: Comparison of Different Foothold Reward Designs. The success rate and foothold error for each foothold reward design are evaluated on stepping stones with medium terrain difficulty.

#### Table 6: TABLE VI: Comparison of Different Curriculum Designs. The success rate and traverse rate for each curriculum design are

| Designs | Medium Difficulty | Hard Difficulty | | |
| --- | --- | --- | --- | --- |
| R succ R succ R_{ $\mathrm{succ}}$ | R trav R trav R_{ $\mathrm{trav}}$ | R succ R succ R_{ $\mathrm{succ}}$ | R trav R trav R_{ $\mathrm{trav}}$ | |
| w/o curriculum-medium | 88.33 88.33 88.33 88.33 | 90.76 90.76 90.76 90.76 | 2.00 2.00 2.00 2.00 | 18.36 18.36 18.36 18.36 |
| w/o curriculum-hard | 40.00 40.00 40.00 40.00 | 52.49 52.49 52.49 52.49 | 23.67 23.67 23.67 23.67 | 39.94 39.94 39.94 39.94 |
| BeamDojo | 95.67 95.67  $\mathbf{95.67}.67$ | 96.11 96.11  $\mathbf{96.11}.11$ | 82.33 82.33  $\mathbf{82.33}.33$ | 86.87 86.87  $\mathbf{86.87}.87$ |

**说明**: TABLE VI: Comparison of Different Curriculum Designs. The success rate and traverse rate for each curriculum design are evaluated on stepping stones with medium and hard terrain difficulty respectively.

#### Table 7: TABLE VII: Reward Functions

| Term | Equation | Weight |
| --- | --- | --- |
| Group 1: Locomotion Reward Group | | |
| xy velocity tracking | $$ | 1.0 1.0 1.0 1.0 |
| yaw velocity tracking | $\exp\left\{-{\left(\boldsymbol{\omega}_{\text{yaw}}-\boldsymbol{\omega}_{\text{yaw}}^{c}\right)^{2}}/{\sigma}\right\} {- (yaw - yaw) 2 / }$ | 1.0 1.0 1.0 1.0 |
| base height | (h - h target) 2 h h target 2  $\left(h-h^{\text{target}}\right)^{2} (- target) 2$ | - 10.0 10.0 -10.0 - 10.0 |
| orientation | $$ | - 2.0 2.0 -2.0 - 2.0 |
| z velocity | v z 2 v z 2  $\mathbf{v}_{z}^{2} 2$ | - 2.0 2.0 -2.0 - 2.0 |
| roll-pitch velocity | $$ | - 0.05 0.05 -0.05 - 0.05 |
| action rate | $$ | - 0.01 0.01 -0.01 - 0.01 |
| smoothness | $$ | - 1 e - 3 1 e 3 -1e-3 - 1 - 3 |
| stand still | $$ | - 0.05 0.05 -0.05 - 0.05 |
| joint velocities | $$ | - 1 e - 4 1 e 4 -1e-4 - 1 - 4 |
| joint accelerations | $$ | - 2.5 e - 8 2.5 e 8 -2.5e-8 - 2.5 - 8 |
| joint position limits | $\text{ReLU}({\boldsymbol{\theta}}-{\boldsymbol{\theta}}_{\text{max}})+ ReLU (- max) + ReLU (θ min - θ) ReLU θ min θ \text{ReLU}({\boldsymbol{\theta}_{\text{min}}}-{\boldsymbol{\theta}}) ReLU (min -)$ | - 5.0 5.0 -5.0 - 5.0 |
| joint velocity limits | $\text{ReLU}(\lVert \dot{\boldsymbol{\theta}} \rVert-\lVert \dot{\boldsymbol{\theta}}_{\text{max}} \rVert) ReLU (\lVert \rVert - \lVert max \rVert)$ | - 1 e - 3 1 e 3 -1e-3 - 1 - 3 |
| joint power | $$ | - 2 e - 5 2 e 5 -2e-5 - 2 - 5 |
| feet ground parallel | ∑ i = 1 2 Var (p z, i) i 1 2 Var p z i  $\sum_{i=1}^{2}\text{Var}(\mathbf{p}_{z,i}) ∑ = 1 2 Var$ | - 0.02 0.02 -0.02 - 0.02 |
| feet distance | ReLU (\| p y, 1 - p y, 2 \| - d min) ReLU p y 1 p y 2 d min  $\text{ReLU}\left(\lVert \mathbf{p}_{y,1}-\mathbf{p}_{y,2} \rVert-d_{\text{min}}\right) ReLU (\lVert, 1 -, 2 \rVert - min)$ | 0.5 0.5 0.5 0.5 |
| feet air time | $\sum_{i=1}^{2}\left(t_{\text{air},i}-t_{\text{air}}^{\text{target}}\right)\cdot\mathbb{F}_{i} ∑ = 1 2 (air, - air target) ⋅$ | 1.0 1.0 1.0 1.0 |
| feet clearance | $\sum_{i=1}^{2}\left(\mathbf{p}_{z,i}-\mathbf{p}_{z}^{\text{target}}\right)^{2}\cdot\dot{\mathbf{p}}_{xy,i} ∑ = 1 2 (, - target) 2 ⋅,$ | - 1.0 1.0 -1.0 - 1.0 |
| Group 2: Foothold Reward Group | | |
| foothold | $\sum_{i=1}^{2}\mathbb{C}_{i}\sum_{j=1}^{n}\mathbf{1}\{d_{ij}<\epsilon\} - ∑ = 1 2 ∑ = 1 {< }$ | 1.0 1.0 1.0 1.0 |

**说明**: TABLE VII: Reward Functions

#### Table 8: TABLE VIII: Used Symbols

| Symbols | Description |
| --- | --- |
|   \sigma | Tracking shape scale, set to 0.25 0.25 0.25 0.25. |
|  \epsilon | Threshold for determining zero-command in stand still reward, set to 0.1 0.1 0.1 0.1. |
| $\boldsymbol{\tau}$ | Computed joint torques. |
| h target h target h^{ $\text{target}} target$ | Desired base height relative to the ground, set to 0.725 0.725 0.725 0.725. |
| ReLU(  \cdot ) | Function that clips negative values to zero [14 ]. |
| p i, p  ̇ i p i  ̇ p i  $\mathbf{p}_{i},\dot{\mathbf{p}}_{i},$ | Spatial position and velocity of all sampled points on the i i i -th foot respectively. |
| p z target p z target  $\mathbf{p}_{z}^{\text{target}} target$ | Target foot-lift height, set to 0.1 0.1 0.1 0.1. |
| t air, i t air i t_{ $\text{air},i} air,$ | Air time of the i i i -th foot. |
| t air target t air target t_{ $\text{air}}^{\text{target}} air target$ | Desired feet air time, set to 0.5 0.5 0.5 0.5. |
| F i F i  $\mathbb{F}_{i}$ | Indicator specifying whether foot i i i makes first ground contact. |
| d min d min d_{ $\text{min}} min$ | Minimum allowable distance between two feet, set to 0.18 0.18 0.18 0.18. |

**说明**: TABLE VIII: Used Symbols

#### Table 9: TABLE IX: Domain Randomization Setting

| Term | Value |
| --- | --- |
| Observations | |
| angular velocity noise | U (- 0.5, 0.5) U 0.5 0.5  $\mathcal{U}(-0.5,0.5) (- 0.5, 0.5) rad/s$ |
| joint position noise | U (- 0.05, 0.05) U 0.05 0.05  $\mathcal{U}(-0.05,0.05) (- 0.05, 0.05) rad/s$ |
| joint velocity noise | U (- 2.0, 2.0) U 2.0 2.0  $\mathcal{U}(-2.0,2.0) (- 2.0, 2.0) rad/s$ |
| projected gravity noise | U (- 0.05, 0.05) U 0.05 0.05  $\mathcal{U}(-0.05,0.05) (- 0.05, 0.05) rad/s$ |
| Humanoid Physical Properties | |
| actuator offset | U (- 0.05, 0.05) U 0.05 0.05  $\mathcal{U}(-0.05,0.05) (- 0.05, 0.05) rad$ |
| motor strength noise | U (0.9, 1.1) U 0.9 1.1  $\mathcal{U}(0.9,1.1)$ |
| payload mass | U (- 2.0, 2.0) U 2.0 2.0  $\mathcal{U}(-2.0,2.0) (- 2.0, 2.0) kg$ |
| center of mass displacement | U (- 0.05, 0.05) U 0.05 0.05  $\mathcal{U}(-0.05,0.05) (- 0.05, 0.05) m$ |
| Kp, Kd noise factor | U (0.85, 1.15) U 0.85 1.15  $\mathcal{U}(0.85,1.15)$ |
| Terrain Dynamics | |
| friction factor | U (0.4, 1.0) U 0.4 1.0  $\mathcal{U}(0.4,1.0)$ |
| restitution factor | U (0.0, 1.0) U 0.0 1.0  $\mathcal{U}(0.0,1.0)$ |
| terrain height noise | U (- 0.02, 0.02) U 0.02 0.02  $\mathcal{U}(-0.02,0.02) (- 0.02, 0.02) m$ |
| Elevation Map | |
| vertical offset | U (- 0.03, 0.03) U 0.03 0.03  $\mathcal{U}(-0.03,0.03) (- 0.03, 0.03) m$ |
| vertical noise | U (- 0.03, 0.03) U 0.03 0.03  $\mathcal{U}(-0.03,0.03) (- 0.03, 0.03) m$ |
| map roll, pitch rotation noise | U (- 0.03, 0.03) U 0.03 0.03  $\mathcal{U}(-0.03,0.03) (- 0.03, 0.03) m$ |
| map yaw rotation noise | U (- 0.2, 0.2) U 0.2 0.2  $\mathcal{U}(-0.2,0.2) (- 0.2, 0.2) rad$ |
| foothold extension probability | 0.6 0.6 0.6 0.6 |
| map repeat probability | 0.2 0.2 0.2 0.2 |

**说明**: TABLE IX: Domain Randomization Setting

#### Table 10: TABLE X: Hyperparameters

| Hyperparameter | Value |
| --- | --- |
| General | |
| num of robots | 4096 |
| num of steps per iteration | 100 |
| num of epochs | 5 |
| gradient clipping | 1.0 |
| adam epsilon | 1 e - 8 1 e 8 1e-8 1 - 8 |
| PPO | |
| clip range | 0.2 |
| entropy coefficient | 0.01 |
| discount factor $\gamma$ | 0.99 |
| GAE balancing factor   \lambda | 0.95 |
| desired KL-divergence | 0.01 |
| actor and double critic NN | MLP, hidden units [512, 216, 128] |
| BeamDojo | |
| w 1, w 2 w 1 w 2 w_{1},w_{2} 1, 2 | 1.0, 0.25 |

**说明**: TABLE X: Hyperparameters
## 实验解读

- 评价重点:围绕 agile-locomotion、foothold-tracking、人形机器人、足式运动、强化学习,优先看真实机器人结果、扰动恢复、接触稳定性、控制误差、成功率、实时性和 sim-to-real 差距。
- System0 视角:实验结论不只看最终成功率,还要看闭环过程中是否能处理状态估计误差、执行延迟、碰撞/接触突变和外部扰动。
- 与上层系统的关系:如果方法能在快闭环内稳定工作,它可以作为 System1 技能或 System2 规划的底层执行保障;如果只在离线或慢速设定下成立,则还不能直接作为身体级快闭环模块。
## 批判性思考

### 优点

- 论文把 agile-locomotion、foothold-tracking、人形机器人、足式运动、强化学习 放到机器人身体级闭环里讨论,和只做高层语义推理的工作相比,更接近真实机器人部署时会遇到的状态、动力学和安全约束。
- 若论文同时给出公式、系统结构图、硬件实验或消融结果,这些证据可以帮助判断方法是否真的能支撑毫秒到百毫秒级反应。

### 局限

- 需要重点核查控制频率、推理延迟、传感延迟、失败恢复、硬件磨损、未见场景泛化和安全边界;这些因素决定它能否成为可靠的 System0 模块。
- 对学习型方法,还要区分训练环境中的鲁棒性和真实部署中的鲁棒性,尤其是接触丰富任务、动态运动任务和人形机器人任务。
## 论文定位

- 研究对象:BeamDojo: Learning Agile Humanoid Locomotion on Sparse Footholds。
- 关键词:agile-locomotion、foothold-tracking、人形机器人、足式运动、强化学习、鲁棒控制、terrain-adaptation。
- System0 关联:关注机器人身体级快反应链路中“状态/接触/局部感知 -> 约束/优化/策略 -> 动作”的短闭环,不把问题停留在语言规划或任务分解层。

## 速查卡片

> [!summary] BeamDojo
> - **论文**: https://www.roboticsproceedings.org/rss21/p068.pdf
> - **arXiv**: http://arxiv.org/abs/2502.10363v3
> - **arXiv HTML**: https://arxiv.org/html/2502.10363v3
