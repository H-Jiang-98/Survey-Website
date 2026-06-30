---
title: "ViserDex: Visual Sim-to-Real for Robust Dexterous In-hand Reorientation"
method_name: "ViserDex"
authors: ["Arjun Bhardwaj"]
year: 2026
venue: "RSS"
tags: ["contact-reasoning", "robot-manipulation", "robust-control", "reinforcement-learning", "dexterous-manipulation", "sim-to-real", "whole-body-control", "state-estimation"]
image_source: "online"
arxiv_html: "https://arxiv.org/html/2604.11138v1"
---
# ViserDex
## 一句话总结

> ViserDex: Visual Sim-to-Real for Robust Dexterous In-hand Reorientation 主要落在 [[接触推理]]、[[灵巧操作]]、[[in-hand-manipulation]]、[[motion-tracking]] 这一类问题上;从 System0 视角看,它的价值在于把身体状态、接触变化、局部感知或动力学约束接到机器人低层闭环里,让动作生成更稳定、安全、实时。
## 核心贡献

1. 围绕 **ViserDex: Visual Sim-to-Real for Robust Dexterous In-hand Reorientation** 建立了一个与 接触推理、灵巧操作、in-hand-manipulation、motion-tracking、强化学习、机器人操作 相关的机器人问题设定,重点不是高层语义规划,而是身体级闭环中的状态变化、接触变化或控制约束。
2. 方法层面的关键价值在于把感知、动力学、约束优化、策略学习或硬件反馈组织成可执行的低层动作生成机制,适合放进 System0 的快反应模块中理解。
3. 对系统集成而言,这类工作补上了 System1 技能选择与电机级控制之间的一层:它负责把上层目标变成动态可行、可恢复、可安全过滤的动作。
4. 若论文包含真实机器人、接触任务或仿真到真实迁移,核心应看它是否报告了闭环稳定性、扰动恢复、硬件延迟、失败模式和跨场景泛化。
## 问题背景

这篇论文可以放在 System0 的“具身快系统”背景下阅读:机器人需要在很短时间内根据本体状态、接触、力/触觉、局部视觉或动力学预测做出动作,而不是等待慢速规划器重新推理整个任务。接触推理、灵巧操作、in-hand-manipulation、motion-tracking、强化学习、机器人操作、鲁棒控制 相关问题的难点通常不在“知道要做什么”,而在动作执行过程中状态会快速偏离、接触会突然改变、约束会实时收紧,低层系统必须及时修正。

### System0 关联

精读时优先看三件事:第一,论文使用哪些快速反馈量,例如关节状态、接触状态、力觉、触觉、深度或局部视觉;第二,反馈如何进入 MPC、WBC、策略网络、安全过滤器或恢复控制器;第三,输出是否直接影响机器人下一步动作,而不是只服务离线分析或高层任务规划。
## 方法详解

### System0 视角解析

从 System0 角度看,这篇论文的方法部分应当围绕 接触推理、灵巧操作、in-hand-manipulation、motion-tracking、强化学习、机器人操作、鲁棒控制、仿真到真实迁移 来读。核心问题是:系统如何从快速变化的机器人状态中构造控制输入,如何处理接触、碰撞、约束、模型误差或执行延迟,以及如何把这些信息变成下一时刻的动作。

如果方法里有优化问题,重点看目标函数、约束和求解频率;如果方法里有学习策略,重点看观测空间、动作空间、训练扰动和部署时的闭环频率;如果方法里有硬件或传感器,重点看传感链路是否真的进入控制回路。这样读可以把论文和 System2 语言规划、System1 技能编排区分开来。

### 关键公式与机制

#### 公式 1: [[动力学或策略机制]]

$$z=f_{\phi}({o}_{\text{prop}}^{\text{noisy}},{o}_{\text{exte}}^{\text{noisy}})$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 2: [[动力学或策略机制]]

$$(\tilde{o}_{\text{exte}},\tilde{o}_{\text{priv}})=h_{\psi}(z,{o}_{\text{exte}}^{\text{noisy}})$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 3: [[动力学或策略机制]]

$$\tilde{a}=g_{\rho}(z,{o}_{\text{prop}}^{\text{noisy}},{o}_{\text{exte}}^{\text{noisy}})$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 4: [[优化目标/约束]]

$$\mathcal{L}=\mathcal{L}_{\text{BC}}+\lambda\mathcal{L}_{\text{recon}}$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 5: [[动力学或策略机制]]

$$c(\mathbf{d})=\text{Sigmoid}\left(\sum_{\ell=0}^{L}\sum_{m=-\ell}^{\ell}k_{\ell}^{m}Y_{\ell}^{m}(\mathbf{d})\right)$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 6: [[动力学或策略机制]]

$$\mathbb{I}(d(\theta)\leq\epsilon_{\text{success}})$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 7: [[动力学或策略机制]]

$$d(\theta)\leq\epsilon_{\text{success}}=0.1$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 8: [[优化目标/约束]]

$$\delta\sim\mathcal{U}(\delta_{\min},\delta_{\max})$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 9: [[优化目标/约束]]

$$\mathcal{I}_{k}\leftarrow\{i\mid C_{i}=k\}$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 10: [[优化目标/约束]]

$$\mathbf{S}^{\prime}[\mathcal{I}_{k}]\leftarrow\mathbf{S}^{\prime}[\mathcal{I}_{k}]\;\oplus\;\delta$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。
## 关键图表

### Figure 1: Overview of our sim-to-real in-hand reorientation pipeline. We first train a teacher

![Figure 1](https://arxiv.org/html/2604.11138v1/x1.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Overview of our sim-to-real in-hand reorientation pipeline. We first train a teacher”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 2: Left: The experimental setup with an RGB camera, an Allegro Hand, and a multi-colored

![Figure 2](https://arxiv.org/html/2604.11138v1/x3.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Left: The experimental setup with an RGB camera, an Allegro Hand, and a multi-colored”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 3: Impact of performance-based curricula on training efficiency. Curves show learning pr

![Figure 3](https://arxiv.org/html/2604.11138v1/x4.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Impact of performance-based curricula on training efficiency. Curves show learning pr”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### 论文表格

#### Table 1: TABLE I: Pre-Rasterization Augmentation Parameters

| Augmentation | Targets | Probability | Fraction | Range |
| --- | --- | --- | --- | --- |
| Random Noise | | | | |
| Additive | SH0, SHN | 0.2 | 1.0 | [- 0.1, 0.1 ] [-0.1,0.1] |
| Scaling | SH0, SHN | 0.2 | 1.0 | [0.8, 1.2 ] [0.8,1.2] |
| Spatial Cluster | | | | |
| Additive | SH0, SHN | 0.8 | 0.10 | [- 0.1, 0.1 ] [-0.1,0.1] |
| Scaling | SH0, SHN | 0.8 | 0.20 | [0.9, 1.1 ] [0.9,1.1] |
| Color Cluster | | | | |
| Additive | SH0 | 0.8 | 0.10 | [- 0.2, 0.2 ] [-0.2,0.2] |
| Additive | SHN | 0.8 | 0.10 | [- 0.1, 0.1 ] [-0.1,0.1] |
| Scaling | SH0, SHN | 0.8 | 0.10 | [0.6, 1.4 ] [0.6,1.4] |
| Global Shift | | | | |
| Additive | SHN | 0.2 | 1.0 | [- 0.1, 0.1 ] [-0.1,0.1] |
| Scaling | SH0, SHN | 0.2 | 1.0 | [0.6, 1.4 ] [0.6,1.4] |
| Uniform Additive | SH0, SHN | 0.8 | 1.0 | [- 0.2, 0.2 ] [-0.2,0.2] |
| Uniform Scaling | SH0 | 0.8 | 1.0 | [0.9, 1.4 ] [0.9,1.4] |

**说明**: TABLE I: Pre-Rasterization Augmentation Parameters

#### Table 2: TABLE II: Evaluation of learned pose estimator on real-world data nominal and adversarial lighting. Our method is

| Objects | Cube | 3D Printed Toy | Rubber Duck | Tablet Bottle | Globe | Mean | | | | | | |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Method | ADD | Accuracy | ADD | Accuracy | ADD | Accuracy | ADD | Accuracy | ADD | Accuracy | ADD | Accuracy |
| Nominal Conditions | | | | | | | | | | | | |
| Standard Tiled | 11.9 ± 0.86 11.9_{\pm 0.86} | 57.2 ± 8.13 57.2_{\pm 8.13} | 14.1 ± 0.92 14.1_{\pm 0.92} | 37.7 ± 2.84 37.7_{\pm 2.84} | 9.5 ± 0.23 9.5_{\pm 0.23} | 67.0 ± 1.47 67.0_{\pm 1.47} | 10.0 ± 0.65 10.0_{\pm 0.65} | 45.7 ± 4.78 45.7_{\pm 4.78} | 15.0 ± 2.41 15.0_{\pm 2.41} | 59.1 ± 3.68 59.1_{\pm 3.68} | 12.1 ± 1.01 12.1_{\pm 1.01} | 53.3 ± 4.18 53.3_{\pm 4.18} |
| DR Tiled | 10.5 ± 0.75 10.5_{\pm 0.75} | 69.7 ± 9.05 69.7_{\pm 9.05} | 15.9 ± 0.64 15.9_{\pm 0.64} | 32.5 ± 2.78 32.5_{\pm 2.78} | 9.1 ± 0.23 9.1_{\pm 0.23} | 73.7 ± 2.22 73.7_{\pm 2.22} | 9.0 ± 0.52  $\mathbf{9.0_{\pm 0.52}}$ | 50.7 ± 3.56 50.7_{\pm 3.56} | 16.7 ± 1.20 16.7_{\pm 1.20} | 51.3 ± 4.21 51.3_{\pm 4.21} | 12.2 ± 0.67 12.2_{\pm 0.67} | 55.6 ± 4.36 55.6_{\pm 4.36} |
| Naïve GS | 10.4 ± 1.35 10.4_{\pm 1.35} | 59.0 ± 4.66 59.0_{\pm 4.66} | 18.7 ± 0.95 18.7_{\pm 0.95} | 23.3 ± 3.18 23.3_{\pm 3.18} | 11.4 ± 0.26 11.4_{\pm 0.26} | 44.3 ± 4.88 44.3_{\pm 4.88} | 11.2 ± 0.63 11.2_{\pm 0.63} | 48.3 ± 4.59 48.3_{\pm 4.59} | 20.2 ± 1.45 20.2_{\pm 1.45} | 17.2 ± 4.61 17.2_{\pm 4.61} | 14.4 ± 0.93 14.4_{\pm 0.93} | 38.4 ± 4.38 38.4_{\pm 4.38} |
| Ours | 9.1 ± 0.51  $\mathbf{9.1_{\pm 0.51}}$ | 73.1 ± 4.73  $\mathbf{73.1_{\pm 4.73}}$ | 11.3 ± 0.82  $\mathbf{11.3_{\pm 0.82}}$ | 56.5 ± 5.77  $\mathbf{56.5_{\pm 5.77}}$ | 7.9 ± 0.24  $\mathbf{7.9_{\pm 0.24}}$ | 78.0 ± 3.97  $\mathbf{78.0_{\pm 3.97}}$ | 10.2 ± 0.84 10.2_{\pm 0.84} | 51.7 ± 5.06  $\mathbf{51.7_{\pm 5.06}}$ | 12.3 ± 0.89  $\mathbf{12.3_{\pm 0.89}}$ | 67.7 ± 3.06  $\mathbf{67.7_{\pm 3.06}}$ | 10.2 ± 0.66  $\mathbf{10.2_{\pm 0.66}}$ | 65.4 ± 4.52  $\mathbf{65.4_{\pm 4.52}}$ |
| Adversarial Conditions | | | | | | | | | | | | |
| Standard Tiled | 12.6 ± 0.74 12.6_{\pm 0.74} | 56.8 ± 4.83 56.8_{\pm 4.83} | 20.3 ± 2.10 20.3_{\pm 2.10} | 29.1 ± 7.05 29.1_{\pm 7.05} | 15.6 ± 0.86 15.6_{\pm 0.86} | 51.3 ± 4.52 51.3_{\pm 4.52} | 24.1 ± 1.58 24.1_{\pm 1.58} | 25.0 ± 3.95 25.0_{\pm 3.95} | 18.7 ± 2.04 18.7_{\pm 2.04} | 41.9 ± 2.68 41.9_{\pm 2.68} | 18.3 ± 1.46 18.3_{\pm 1.46} | 40.8 ± 4.61 40.8_{\pm 4.61} |
| DR Tiled | 11.5 ± 0.89 11.5_{\pm 0.89} | 57.6 ± 4.69 57.6_{\pm 4.69} | 13.8 ± 0.54  $\mathbf{13.8_{\pm 0.54}}$ | 39.0 ± 2.54 39.0_{\pm 2.54} | 11.5 ± 0.61  $\mathbf{11.5_{\pm 0.61}}$ | 57.7 ± 3.89 57.7_{\pm 3.89} | 14.4 ± 1.68 14.4_{\pm 1.68} | 39.1 ± 5.14 39.1_{\pm 5.14} | 18.6 ± 1.07 18.6_{\pm 1.07} | 42.7 ± 4.44 42.7_{\pm 4.44} | 14.0 ± 0.96 14.0_{\pm 0.96} | 47.2 ± 4.14 47.2_{\pm 4.14} |
| Naïve GS | 14.3 ± 0.85 14.3_{\pm 0.85} | 44.3 ± 2.30 44.3_{\pm 2.30} | 21.9 ± 0.85 21.9_{\pm 0.85} | 27.9 ± 4.85 27.9_{\pm 4.85} | 13.9 ± 1.25 13.9_{\pm 1.25} | 57.3 ± 4.29 57.3_{\pm 4.29} | 22.1 ± 1.39 22.1_{\pm 1.39} | 25.0 ± 3.35 25.0_{\pm 3.35} | 21.0 ± 1.52 21.0_{\pm 1.52} | 28.1 ± 5.95 28.1_{\pm 5.95} | 18.6 ± 1.17 18.6_{\pm 1.17} | 36.5 ± 4.15 36.5_{\pm 4.15} |
| Ours | 10.6 ± 0.38  $\mathbf{10.6_{\pm 0.38}}$ | 60.6 ± 5.47  $\mathbf{60.6_{\pm 5.47}}$ | 14.4 ± 0.77 14.4_{\pm 0.77} | 45.9 ± 4.58  $\mathbf{45.9_{\pm 4.58}}$ | 12.2 ± 0.53 12.2_{\pm 0.53} | 62.7 ± 3.09  $\mathbf{62.7_{\pm 3.09}}$ | 13.5 ± 0.99  $\mathbf{13.5_{\pm 0.99}}$ | 46.5 ± 3.43  $\mathbf{46.5_{\pm 3.43}}$ | 13.8 ± 0.78  $\mathbf{13.8_{\pm 0.78}}$ | 65.6 ± 2.52  $\mathbf{65.6_{\pm 2.52}}$ | 12.9 ± 0.69  $\mathbf{12.9_{\pm 0.69}}$ | 56.3 ± 3.82  $\mathbf{56.3_{\pm 3.82}}$ |

**说明**: 该表格来自论文中的公式、奖励项或实验设置；为避免 PDF 抽取噪声，网页中仅保留可渲染的核心符号和数值。

#### Table 3: TABLE III: Ablation study of pre-rasterization augmentations nominal and adversarial conditions for pose estimati

| Objects | Cube | 3D Printed Toy | Rubber Duck | Tablet Bottle | Globe | Mean | | | | | | |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Method | ADD | Accuracy | ADD | Accuracy | ADD | Accuracy | ADD | Accuracy | ADD | Accuracy | ADD | Accuracy |
| Nominal Conditions | | | | | | | | | | | | |
| w/o Random Noise | 11.9 ± 3.24 11.9_{\pm 3.24} | 61.5 ± 11.21 61.5_{\pm 11.21} | 13.5 ± 1.14 13.5_{\pm 1.14} | 37.2 ± 9.13 37.2_{\pm 9.13} | 8.2 ± 0.39 8.2_{\pm 0.39} | 80.9 ± 2.88  $\mathbf{80.9_{\pm 2.88}}$ | 10.7 ± 0.84 10.7_{\pm 0.84} | 46.7 ± 3.33 46.7_{\pm 3.33} | 14.5 ± 1.17 14.5_{\pm 1.17} | 66.5 ± 5.32 66.5_{\pm 5.32} | 11.8 ± 1.36 11.8_{\pm 1.36} | 58.6 ± 6.37 58.6_{\pm 6.37} |
| w/o Spatial Clustering | 9.2 ± 0.91 9.2_{\pm 0.91} | 74.9 ± 8.49  $\mathbf{74.9_{\pm 8.49}}$ | 12.9 ± 0.86 12.9_{\pm 0.86} | 44.7 ± 7.06 44.7_{\pm 7.06} | 8.0 ± 0.22 8.0_{\pm 0.22} | 73.5 ± 3.27 73.5_{\pm 3.27} | 16.3 ± 1.38 16.3_{\pm 1.38} | 39.3 ± 5.64 39.3_{\pm 5.64} | 13.8 ± 2.15 13.8_{\pm 2.15} | 51.3 ± 5.73 51.3_{\pm 5.73} | 12.0 ± 1.10 12.0_{\pm 1.10} | 56.7 ± 6.04 56.7_{\pm 6.04} |
| w/o Color Clustering | 9.2 ± 0.41 9.2_{\pm 0.41} | 71.0 ± 5.94 71.0_{\pm 5.94} | 13.6 ± 1.60 13.6_{\pm 1.60} | 49.4 ± 2.42 49.4_{\pm 2.42} | 8.2 ± 0.59 8.2_{\pm 0.59} | 77.4 ± 5.64 77.4_{\pm 5.64} | 20.4 ± 0.56 20.4_{\pm 0.56} | 34.3 ± 4.03 34.3_{\pm 4.03} | 14.5 ± 1.69 14.5_{\pm 1.69} | 64.8 ± 3.09 64.8_{\pm 3.09} | 13.2 ± 0.97 13.2_{\pm 0.97} | 59.4 ± 4.22 59.4_{\pm 4.22} |
| w/o Global Shift | 11.2 ± 1.26 11.2_{\pm 1.26} | 67.2 ± 3.85 67.2_{\pm 3.85} | 10.9 ± 1.13  $\mathbf{10.9_{\pm 1.13}}$ | 55.7 ± 6.58 55.7_{\pm 6.58} | 9.1 ± 0.41 9.1_{\pm 0.41} | 70.9 ± 3.85 70.9_{\pm 3.85} | 39.5 ± 2.84 39.5_{\pm 2.84} | 10.3 ± 2.45 10.3_{\pm 2.45} | 13.9 ± 0.25 13.9_{\pm 0.25} | 52.1 ± 5.17 52.1_{\pm 5.17} | 16.9 ± 1.18 16.9_{\pm 1.18} | 51.2 ± 4.38 51.2_{\pm 4.38} |
| Ours | 9.1 ± 0.51  $\mathbf{9.1_{\pm 0.51}}$ | 73.1 ± 4.73 73.1_{\pm 4.73} | 11.3 ± 0.82 11.3_{\pm 0.82} | 56.5 ± 5.77  $\mathbf{56.5_{\pm 5.77}}$ | 7.9 ± 0.24  $\mathbf{7.9_{\pm 0.24}}$ | 78.0 ± 3.97 78.0_{\pm 3.97} | 10.2 ± 0.84  $\mathbf{10.2_{\pm 0.84}}$ | 51.7 ± 5.06  $\mathbf{51.7_{\pm 5.06}}$ | 12.3 ± 0.89  $\mathbf{12.3_{\pm 0.89}}$ | 67.7 ± 3.06  $\mathbf{67.7_{\pm 3.06}}$ | 10.2 ± 0.66  $\mathbf{10.2_{\pm 0.66}}$ | 65.4 ± 4.52  $\mathbf{65.4_{\pm 4.52}}$ |
| Adversarial Conditions | | | | | | | | | | | | |
| w/o Random Noise | 13.2 ± 0.77 13.2_{\pm 0.77} | 49.7 ± 5.46 49.7_{\pm 5.46} | 15.8 ± 1.34 15.8_{\pm 1.34} | 40.7 ± 4.62 40.7_{\pm 4.62} | 12.9 ± 0.39 12.9_{\pm 0.39} | 52.7 ± 4.29 52.7_{\pm 4.29} | 13.9 ± 0.78 13.9_{\pm 0.78} | 41.5 ± 1.10 41.5_{\pm 1.10} | 16.3 ± 0.86 16.3_{\pm 0.86} | 58.4 ± 3.83 58.4_{\pm 3.83} | 14.4 ± 0.83 14.4_{\pm 0.83} | 48.6 ± 3.86 48.6_{\pm 3.86} |
| w/o Spatial Clustering | 13.5 ± 1.40 13.5_{\pm 1.40} | 46.2 ± 7.00 46.2_{\pm 7.00} | 16.4 ± 1.20 16.4_{\pm 1.20} | 39.3 ± 7.13 39.3_{\pm 7.13} | 14.0 ± 0.69 14.0_{\pm 0.69} | 51.7 ± 4.22 51.7_{\pm 4.22} | 17.6 ± 0.84 17.6_{\pm 0.84} | 35.6 ± 3.00 35.6_{\pm 3.00} | 16.2 ± 1.59 16.2_{\pm 1.59} | 39.6 ± 5.92 39.6_{\pm 5.92} | 15.5 ± 1.14 15.5_{\pm 1.14} | 42.5 ± 5.45 42.5_{\pm 5.45} |
| w/o Color Clustering | 14.4 ± 0.68 14.4_{\pm 0.68} | 47.3 ± 6.63 47.3_{\pm 6.63} | 16.7 ± 1.60 16.7_{\pm 1.60} | 41.2 ± 4.03 41.2_{\pm 4.03} | 14.8 ± 0.80 14.8_{\pm 0.80} | 46.7 ± 3.16 46.7_{\pm 3.16} | 19.9 ± 1.15 19.9_{\pm 1.15} | 29.4 ± 4.46 29.4_{\pm 4.46} | 16.2 ± 1.46 16.2_{\pm 1.46} | 58.9 ± 3.93 58.9_{\pm 3.93} | 16.4 ± 1.14 16.4_{\pm 1.14} | 44.7 ± 4.44 44.7_{\pm 4.44} |
| w/o Global Shift | 24.9 ± 1.29 24.9_{\pm 1.29} | 17.8 ± 3.16 17.8_{\pm 3.16} | 22.0 ± 1.54 22.0_{\pm 1.54} | 25.4 ± 2.54 25.4_{\pm 2.54} | 20.0 ± 1.89 20.0_{\pm 1.89} | 27.7 ± 1.70 27.7_{\pm 1.70} | 30.2 ± 1.63 30.2_{\pm 1.63} | 12.6 ± 2.39 12.6_{\pm 2.39} | 17.5 ± 0.52 17.5_{\pm 0.52} | 34.4 ± 2.42 34.4_{\pm 2.42} | 22.9 ± 1.37 22.9_{\pm 1.37} | 23.6 ± 2.44 23.6_{\pm 2.44} |
| Ours | 10.6 ± 0.38  $\mathbf{10.6_{\pm 0.38}}$ | 60.6 ± 5.47  $\mathbf{60.6_{\pm 5.47}}$ | 14.4 ± 0.77  $\mathbf{14.4_{\pm 0.77}}$ | 45.9 ± 4.58  $\mathbf{45.9_{\pm 4.58}}$ | 12.2 ± 0.53  $\mathbf{12.2_{\pm 0.53}}$ | 62.7 ± 3.09  $\mathbf{62.7_{\pm 3.09}}$ | 13.5 ± 0.99  $\mathbf{13.5_{\pm 0.99}}$ | 46.5 ± 3.43  $\mathbf{46.5_{\pm 3.43}}$ | 13.8 ± 0.78  $\mathbf{13.8_{\pm 0.78}}$ | 65.6 ± 2.52  $\mathbf{65.6_{\pm 2.52}}$ | 12.9 ± 0.69  $\mathbf{12.9_{\pm 0.69}}$ | 56.3 ± 3.82  $\mathbf{56.3_{\pm 3.82}}$ |

**说明**: 该表格来自论文中的公式、奖励项或实验设置；为避免 PDF 抽取噪声，网页中仅保留可渲染的核心符号和数值。

#### Table 4: TABLE IV: Real-world deployment on different objects.

| | Consecutive Successes (Over Five Runs) | | |
| --- | --- | --- | --- |
| Objects | DeXtreme [7 ] | Ours (Nominal) | Ours (Adversarial) |
| Cube | 27.8 ± 19.0 27.8_{\pm 19.0} | 35.4 ± 13.8 {35.4_{\pm 13.8}} | 25.6 ± 8.9 25.6_{\pm 8.9} |
| 3D Printed Toy | - | 28.2 ± 12.6 {28.2_{\pm 12.6}} | 12.0 ± 6.9 12.0_{\pm 6.9} |
| Rubber Duck | - | 24.2 ± 15.3 {24.2_{\pm 15.3}} | 9.0 ± 5.0 9.0_{\pm 5.0} |
| Tablet Bottle | - | 12.6 ± 8.8 {12.6_{\pm 8.8}} | 4.2 ± 0.7 4.2_{\pm 0.7} |
| Globe | - | 87.6 ± 41.4 {87.6_{\pm 41.4}} | 76.2 ± 66.2 76.2_{\pm 66.2} |
| Mean | - | 37.6 ± 21.8 {37.6_{\pm 21.8}} | 25.4 ± 30.1 25.4_{\pm 30.1} |

**说明**: TABLE IV: Real-world deployment on different objects.

#### Table 5: TABLE V: Simulation results on different objects. We evaluate the policy across 256 environments with randomized physic

| | Consecutive Successes | | |
| --- | --- | --- | --- |
| Objects | Teacher Policy (w/o obs noise) | Student Policy (w/o obs noise) | Student Policy (with obs noise) |
| Cube | 111.4 ± 24.7 111.4_{\pm 24.7} | 92.1 ± 5.7 {92.1_{\pm 5.7}} | 82.3 ± 8.4 82.3_{\pm 8.4} |
| 3D Printed Toy | 106.0 ± 25.0 106.0_{\pm 25.0} | 74.9 ± 4.7 {74.9_{\pm 4.7}} | 74.6 ± 6.2 74.6_{\pm 6.2} |
| Rubber Duck | 97.1 ± 14.1 97.1_{\pm 14.1} | 46.2 ± 1.8 {46.2_{\pm 1.8}} | 41.6 ± 2.3 41.6_{\pm 2.3} |
| Tablet Bottle | 118.4 ± 2.8 118.4_{\pm 2.8} | 77.0 ± 4.2 {77.0_{\pm 4.2}} | 69.4 ± 2.9 69.4_{\pm 2.9} |
| Globe | 163.6 ± 3.4 163.6_{\pm 3.4} | 138.1 ± 2.4 {138.1_{\pm 2.4}} | 129.5 ± 5.9 129.5_{\pm 5.9} |
| Mean | 119.3 ± 17.05 119.3_{\pm 17.05} | 85.7 ± 4.03 85.7_{\pm 4.03} | 79.5 ± 5.61 79.5_{\pm 5.61} |

**说明**: TABLE V: Simulation results on different objects. We evaluate the policy across 256 environments with randomized physical conditions (DR) and report the mean and standard deviation five episodes.

#### Table 6: TABLE VI: Pose estimation during real-world hardware deployment. We report the translation error (in mm) and the rotati

| Objects | Trans Error | Rot Error | Trans Error Correlation | Rot Error Correlation |
| --- | --- | --- | --- | --- |
| Cube | 9.05 9.05 | 14.6 14.6 | 0.4 0.4 | 0.38 0.38 |
| 3D Printed Toy | 11.14 11.14 | 33.59 33.59 | 0.42 0.42 | 0.12 0.12 |
| Rubber Duck | 8.85 8.85 | 18.84 18.84 | 0.02 0.02 | 0.17 0.17 |
| Tablet Bottle | 10.9 10.9 | 38.27 38.27 | 0.14 0.14 | 0.08 0.08 |
| Globe | 12.01 12.01 | 32.42 32.42 | 0.04 0.04 | 0.20 0.20 |
| Mean | 10.39 10.39 | 27.54 27.54 | 0.20 0.20 | 0.19 0.19 |

**说明**: TABLE VI: Pose estimation during real-world hardware deployment. We report the translation error (in mm) and the rotation error (in degrees). We also report Pearson correlation coefficient between the occlusion ratio and errors.

#### Table 7: TABLE VII: Evaluation of learned pose estimator on real-world data nominal and adversarial lighting. Our method i

| Objects | Cube | 3D Printed Toy | Rubber Duck | Tablet Bottle | Globe | Mean | | | | | | |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Method | Trans Error | Rot Error | Trans Error | Rot Error | Trans Error | Rot Error | Trans Error | Rot Error | Trans Error | Rot Error | Trans Error | Rot Error |
| Nominal Conditions | | | | | | | | | | | | |
| Standard Tiled | 10.8 ± 0.78 10.8_{\pm 0.78} | 5.6 ± 1.57 5.6_{\pm 1.57} | 11.2 ± 0.32 11.2_{\pm 0.32} | 19.2 ± 3.79 19.2_{\pm 3.79} | 8.1 ± 0.22 8.1_{\pm 0.22} | 7.5 ± 0.75 7.5_{\pm 0.75} | 8.2 ± 0.39 8.2_{\pm 0.39} | 13.3 ± 1.41 13.3_{\pm 1.41} | 13.9 ± 2.51 13.9_{\pm 2.51} | 8.4 ± 0.59 8.4_{\pm 0.59} | 10.4 ± 1.20 10.4_{\pm 1.20} | 10.8 ± 1.99 10.8_{\pm 1.99} |
| Domain Randomized | 10.0 ± 0.80 10.0_{\pm 0.80} | 4.0 ± 0.15  $\mathbf{4.0_{\pm 0.15}}$ | 15.4 ± 0.74 15.4_{\pm 0.74} | 8.8 ± 0.78  $\mathbf{8.8_{\pm 0.78}}$ | 8.3 ± 0.26 8.3_{\pm 0.26} | 5.0 ± 0.20 5.0_{\pm 0.20} | 7.8 ± 0.13 7.8_{\pm 0.13} | 10.2 ± 1.29  $\mathbf{10.2_{\pm 1.29}}$ | 14.5 ± 1.29 14.5_{\pm 1.29} | 14.5 ± 2.31 14.5_{\pm 2.31} | 11.2 ± 0.77 11.2_{\pm 0.77} | 8.5 ± 1.24 8.5_{\pm 1.24} |
| Naive GS | 8.7 ± 1.25 8.7_{\pm 1.25} | 6.9 ± 1.22 6.9_{\pm 1.22} | 15.3 ± 0.96 15.3_{\pm 0.96} | 23.2 ± 2.04 23.2_{\pm 2.04} | 10.6 ± 0.32 10.6_{\pm 0.32} | 6.0 ± 0.27 6.0_{\pm 0.27} | 8.6 ± 0.54 8.6_{\pm 0.54} | 16.0 ± 3.54 16.0_{\pm 3.54} | 18.8 ± 1.50 18.8_{\pm 1.50} | 11.7 ± 1.38 11.7_{\pm 1.38} | 12.4 ± 1.01 12.4_{\pm 1.01} | 12.8 ± 2.01 12.8_{\pm 2.01} |
| Ours | 8.2 ± 0.64  $\mathbf{8.2_{\pm 0.64}}$ | 4.8 ± 0.18 4.8_{\pm 0.18} | 10.2 ± 1.01  $\mathbf{10.2_{\pm 1.01}}$ | 8.8 ± 0.82  $\mathbf{8.8_{\pm 0.82}}$ | 7.5 ± 0.26  $\mathbf{7.5_{\pm 0.26}}$ | 4.2 ± 0.08  $\mathbf{4.2_{\pm 0.08}}$ | 7.5 ± 0.74  $\mathbf{7.5_{\pm 0.74}}$ | 10.3 ± 1.93 10.3_{\pm 1.93} | 11.7 ± 0.83  $\mathbf{11.7_{\pm 0.83}}$ | 5.3 ± 0.55  $\mathbf{5.3_{\pm 0.55}}$ | 9.0 ± 0.74  $\mathbf{9.0_{\pm 0.74}}$ | 6.7 ± 0.97  $\mathbf{6.7_{\pm 0.97}}$ |
| Adversarial Conditions | | | | | | | | | | | | |
| Standard Tiled | 9.0 ± 0.95  $\mathbf{9.0_{\pm 0.95}}$ | 12.0 ± 1.74 12.0_{\pm 1.74} | 15.9 ± 1.49 15.9_{\pm 1.49} | 33.4 ± 5.14 33.4_{\pm 5.14} | 9.1 ± 0.53 9.1_{\pm 0.53} | 25.1 ± 1.83 25.1_{\pm 1.83} | 12.4 ± 1.03 12.4_{\pm 1.03} | 57.2 ± 4.08 57.2_{\pm 4.08} | 15.3 ± 2.38 15.3_{\pm 2.38} | 20.7 ± 1.39 20.7_{\pm 1.39} | 12.3 ± 1.42 12.3_{\pm 1.42} | 29.7 ± 3.21 29.7_{\pm 3.21} |
| Domain Randomized | 9.1 ± 0.64 9.1_{\pm 0.64} | 9.2 ± 1.63 9.2_{\pm 1.63} | 12.2 ± 0.58  $\mathbf{12.2_{\pm 0.58}}$ | 13.3 ± 1.79  $\mathbf{13.3_{\pm 1.79}}$ | 9.1 ± 0.38 9.1_{\pm 0.38} | 10.7 ± 1.31  $\mathbf{10.7_{\pm 1.31}}$ | 8.7 ± 0.72 8.7_{\pm 0.72} | 28.8 ± 4.55 28.8_{\pm 4.55} | 14.4 ± 1.37 14.4_{\pm 1.37} | 26.5 ± 2.03 26.5_{\pm 2.03} | 10.7 ± 0.81 10.7_{\pm 0.81} | 17.7 ± 2.55 17.7_{\pm 2.55} |
| Naive GS | 10.9 ± 0.79 10.9_{\pm 0.79} | 13.5 ± 1.21 13.5_{\pm 1.21} | 17.4 ± 0.66 17.4_{\pm 0.66} | 37.6 ± 2.73 37.6_{\pm 2.73} | 8.3 ± 0.43  $\mathbf{8.3_{\pm 0.43}}$ | 22.3 ± 4.35 22.3_{\pm 4.35} | 9.5 ± 0.76 9.5_{\pm 0.76} | 56.3 ± 4.37 56.3_{\pm 4.37} | 15.9 ± 1.84 15.9_{\pm 1.84} | 31.1 ± 1.80 31.1_{\pm 1.80} | 12.4 ± 1.02 12.4_{\pm 1.02} | 32.2 ± 3.17 32.2_{\pm 3.17} |
| Ours | 9.1 ± 0.46 9.1_{\pm 0.46} | 6.6 ± 0.78  $\mathbf{6.6_{\pm 0.78}}$ | 12.7 ± 0.61 12.7_{\pm 0.61} | 17.2 ± 1.67 17.2_{\pm 1.67} | 9.5 ± 0.32 9.5_{\pm 0.32} | 12.0 ± 2.07 12.0_{\pm 2.07} | 7.7 ± 0.35  $\mathbf{7.7_{\pm 0.35}}$ | 26.7 ± 2.66  $\mathbf{26.7_{\pm 2.66}}$ | 12.3 ± 0.71  $\mathbf{12.3_{\pm 0.71}}$ | 10.7 ± 2.20  $\mathbf{10.7_{\pm 2.20}}$ | 10.3 ± 0.51  $\mathbf{10.3_{\pm 0.51}}$ | 14.6 ± 1.98  $\mathbf{14.6_{\pm 1.98}}$ |

**说明**: TABLE VII: Evaluation of learned pose estimator on real-world data nominal and adversarial lighting. Our method is compared against three baselines: Standard Tiled, Randomized Tiled, and Naive GS rendering. We report the translation error (in mm) and the rotation error (in degrees), averaged 5 random seeds.

#### Table 8: TABLE VIII: Ablation study of pre-rasterization augmentations nominal and adversarial conditions for pose estimat

| Objects | Cube | 3D Printed Toy | Rubber Duck | Tablet Bottle | Globe | Mean | | | | | | |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Method | Trans Error | Rot Error | Trans Error | Rot Error | Trans Error | Rot Error | Trans Error | Rot Error | Trans Error | Rot Error | Trans Error | Rot Error |
| Nominal Conditions | | | | | | | | | | | | |
| w/o Random Noise | 10.8 ± 3.05 10.8_{\pm 3.05} | 6.8 ± 2.27 6.8_{\pm 2.27} | 12.5 ± 0.91 12.5_{\pm 0.91} | 10.0 ± 2.63 10.0_{\pm 2.63} | 7.7 ± 0.44 7.7_{\pm 0.44} | 4.7 ± 0.29 4.7_{\pm 0.29} | 8.3 ± 0.62 8.3_{\pm 0.62} | 12.6 ± 2.20 12.6_{\pm 2.20} | 14.0 ± 1.16 14.0_{\pm 1.16} | 5.5 ± 0.59 5.5_{\pm 0.59} | 10.7 ± 1.55 10.7_{\pm 1.55} | 7.9 ± 1.86 7.9_{\pm 1.86} |
| w/o Spatial Clustering | 8.1 ± 1.23  $\mathbf{8.1_{\pm 1.23}}$ | 5.6 ± 3.46 5.6_{\pm 3.46} | 11.7 ± 0.93 11.7_{\pm 0.93} | 10.9 ± 1.48 10.9_{\pm 1.48} | 7.2 ± 0.28  $\mathbf{7.2_{\pm 0.28}}$ | 5.8 ± 0.24 5.8_{\pm 0.24} | 10.7 ± 0.71 10.7_{\pm 0.71} | 26.7 ± 3.23 26.7_{\pm 3.23} | 13.1 ± 2.36 13.1_{\pm 2.36} | 6.6 ± 0.86 6.6_{\pm 0.86} | 10.2 ± 1.31 10.2_{\pm 1.31} | 11.1 ± 2.25 11.1_{\pm 2.25} |
| w/o Color Clustering | 8.1 ± 0.62  $\mathbf{8.1_{\pm 0.62}}$ | 5.4 ± 0.54 5.4_{\pm 0.54} | 12.2 ± 1.78 12.2_{\pm 1.78} | 11.7 ± 1.61 11.7_{\pm 1.61} | 7.6 ± 0.65 7.6_{\pm 0.65} | 4.7 ± 0.16 4.7_{\pm 0.16} | 14.0 ± 0.71 14.0_{\pm 0.71} | 33.8 ± 2.11 33.8_{\pm 2.11} | 13.7 ± 1.81 13.7_{\pm 1.81} | 6.5 ± 0.98 6.5_{\pm 0.98} | 11.1 ± 1.25 11.1_{\pm 1.25} | 12.4 ± 1.29 12.4_{\pm 1.29} |
| w/o Global Shift | 8.3 ± 0.60 8.3_{\pm 0.60} | 11.0 ± 4.64 11.0_{\pm 4.64} | 9.9 ± 1.10  $\mathbf{9.9_{\pm 1.10}}$ | 9.5 ± 1.69 9.5_{\pm 1.69} | 8.1 ± 0.34 8.1_{\pm 0.34} | 6.6 ± 0.53 6.6_{\pm 0.53} | 29.5 ± 3.50 29.5_{\pm 3.50} | 67.9 ± 1.83 67.9_{\pm 1.83} | 11.9 ± 0.43 11.9_{\pm 0.43} | 12.4 ± 1.10 12.4_{\pm 1.10} | 13.5 ± 1.68 13.5_{\pm 1.68} | 21.5 ± 2.42 21.5_{\pm 2.42} |
| Ours | 8.2 ± 0.64 8.2_{\pm 0.64} | 4.8 ± 0.18  $\mathbf{4.8_{\pm 0.18}}$ | 10.2 ± 1.01 10.2_{\pm 1.01} | 8.8 ± 0.82  $\mathbf{8.8_{\pm 0.82}}$ | 7.5 ± 0.26 7.5_{\pm 0.26} | 4.2 ± 0.08  $\mathbf{4.2_{\pm 0.08}}$ | 7.5 ± 0.74  $\mathbf{7.5_{\pm 0.74}}$ | 10.3 ± 1.93  $\mathbf{10.3_{\pm 1.93}}$ | 11.7 ± 0.83  $\mathbf{11.7_{\pm 0.83}}$ | 5.3 ± 0.55  $\mathbf{5.3_{\pm 0.55}}$ | 9.0 ± 0.74  $\mathbf{9.0_{\pm 0.74}}$ | 6.7 ± 0.97  $\mathbf{6.7_{\pm 0.97}}$ |
| Adversarial Conditions | | | | | | | | | | | | |
| w/o Random Noise | 11.1 ± 0.73 11.1_{\pm 0.73} | 9.3 ± 1.68 9.3_{\pm 1.68} | 14.2 ± 1.24 14.2_{\pm 1.24} | 16.8 ± 1.90  $\mathbf{16.8_{\pm 1.90}}$ | 10.4 ± 0.45 10.4_{\pm 0.45} | 12.2 ± 1.18 12.2_{\pm 1.18} | 8.7 ± 0.60 8.7_{\pm 0.60} | 26.1 ± 1.83  $\mathbf{26.1_{\pm 1.83}}$ | 15.0 ± 0.85 15.0_{\pm 0.85} | 11.5 ± 1.15 11.5_{\pm 1.15} | 11.9 ± 0.82 11.9_{\pm 0.82} | 15.2 ± 1.58 15.2_{\pm 1.58} |
| w/o Spatial Clustering | 11.2 ± 1.34 11.2_{\pm 1.34} | 9.7 ± 1.11 9.7_{\pm 1.11} | 13.5 ± 1.39 13.5_{\pm 1.39} | 23.6 ± 1.15 23.6_{\pm 1.15} | 10.0 ± 0.30 10.0_{\pm 0.30} | 17.3 ± 2.21 17.3_{\pm 2.21} | 8.9 ± 0.72 8.9_{\pm 0.72} | 39.8 ± 2.90 39.8_{\pm 2.90} | 13.7 ± 1.85 13.7_{\pm 1.85} | 17.1 ± 1.51 17.1_{\pm 1.51} | 11.5 ± 1.25 11.5_{\pm 1.25} | 21.5 ± 1.90 21.5_{\pm 1.90} |
| w/o Color Clustering | 10.9 ± 0.84 10.9_{\pm 0.84} | 13.2 ± 2.58 13.2_{\pm 2.58} | 14.4 ± 1.45 14.4_{\pm 1.45} | 22.3 ± 2.46 22.3_{\pm 2.46} | 10.8 ± 0.44 10.8_{\pm 0.44} | 17.9 ± 2.85 17.9_{\pm 2.85} | 11.2 ± 0.71 11.2_{\pm 0.71} | 41.8 ± 4.49 41.8_{\pm 4.49} | 14.0 ± 1.85 14.0_{\pm 1.85} | 13.8 ± 3.40 13.8_{\pm 3.40} | 12.3 ± 1.18 12.3_{\pm 1.18} | 21.8 ± 3.24 21.8_{\pm 3.24} |
| w/o Global Shift | 16.0 ± 0.93 16.0_{\pm 0.93} | 35.4 ± 2.40 35.4_{\pm 2.40} | 16.6 ± 1.14 16.6_{\pm 1.14} | 41.2 ± 4.81 41.2_{\pm 4.81} | 14.7 ± 1.71 14.7_{\pm 1.71} | 25.2 ± 3.11 25.2_{\pm 3.11} | 10.3 ± 0.60 10.3_{\pm 0.60} | 71.7 ± 6.55 71.7_{\pm 6.55} | 13.9 ± 0.54 13.9_{\pm 0.54} | 21.2 ± 0.66 21.2_{\pm 0.66} | 14.3 ± 1.07 14.3_{\pm 1.07} | 38.9 ± 4.05 38.9_{\pm 4.05} |
| Ours | 9.1 ± 0.46  $\mathbf{9.1_{\pm 0.46}}$ | 6.6 ± 0.78  $\mathbf{6.6_{\pm 0.78}}$ | 12.7 ± 0.61  $\mathbf{12.7_{\pm 0.61}}$ | 17.2 ± 1.67 17.2_{\pm 1.67} | 9.5 ± 0.32  $\mathbf{9.5_{\pm 0.32}}$ | 12.0 ± 2.07  $\mathbf{12.0_{\pm 2.07}}$ | 7.7 ± 0.35  $\mathbf{7.7_{\pm 0.35}}$ | 26.7 ± 2.66 26.7_{\pm 2.66} | 12.3 ± 0.71  $\mathbf{12.3_{\pm 0.71}}$ | 10.7 ± 2.20  $\mathbf{10.7_{\pm 2.20}}$ | 10.3 ± 0.51  $\mathbf{10.3_{\pm 0.51}}$ | 14.6 ± 1.98  $\mathbf{14.6_{\pm 1.98}}$ |

**说明**: TABLE VIII: Ablation study of pre-rasterization augmentations nominal and adversarial conditions for pose estimation. We compare our method with models trained with specific augmentation groups removed. We report the translation error (in mm) and the rotation error (in degrees), averaged 5 random seeds.

#### Table 9: TABLE IX: Reward Terms

| Term | Weight | Equation | Description |
| --- | --- | --- | --- |
| Task Rewards | | | |
| Orientation Tracking | 1.0 | $\theta)+\epsilon)^{-1}$ | $\theta): orientation error, ε = 0.1 \epsilon=0.1.$ |
| Success Bonus | 250.0 | I d ≤ success $\mathbb{I}(d(\theta)\leq\epsilon_{\text{success}})$ | Bonus when d ≤ success . d $\theta)\leq\epsilon_{\text{success}}=0.1 rad.$ |
| Termination Penalties | | | |
| Object Dropped | - 10.0 -10.0 | I  (drop)  $\mathbb{I}(\text{drop})$ | Penalty when the object falls. |
| Curriculum Penalties (Weights are gradually increased during training) | | | |
| Object Distance | -20.0 | $\text{robot}}-p_{\text{obj}}\lVert_{2}$ | Object-robot distance. |
| Object Velocity | -1e-3 | $\text{obj}}\lVert_{2}$ | L 2 L_{2} norm of object linear velocity. |
| Joint Velocity | -8e-2 | $\dot{q}\lVert_{2}$ | L 2 L_{2} norm of joint velocities. |
| Action Magnitude | -0.80 | -  a t  2 -\\|a_{t}\\|_{2} | L 2 L_{2} norm of the action vector. |
| Action Rate | -0.12 | -  a t - a t - 1  2 -\\|a_{t}-a_{t-1}\\|_{2} | L 2 L_{2} norm of diff. b/w consecutive actions. |
| Joint Work | -0.12 | $\sum\lVert \tau\cdot\dot{q} \rVert$ | Mechanical work. |
| Joint Torques | -50.0 | $\tau\lVert_{2}$ | L 2 L_{2} norm of applied joint torques. |

**说明**: TABLE IX: Reward Terms

#### Table 10: TABLE X: Observation Space

| Term | Dim. | Description |
| --- | --- | --- |
| Proprioceptive Observations O prop  $\boldsymbol{\mathcal{O}_{\text{prop}}}$ | | |
| Joint Positions | 16 | Measured robot joint angles. |
| Action History | 64 | Joint position commands from the last four time steps. |
| Goal Orientation | 4 | Target orientation (quaternion). |
| Palm Link Position | 3 | Position of the palm link (constant) |
| Remaining Time | 1 | Normalized time remaining in the episode. |
| Group Total | 88 | |
| Exteroceptive Observations O extero  $\boldsymbol{\mathcal{O}_{\text{extero}}}$ | | |
| Object Pose | 7 | Ground-truth object position and orientation (quaternion). |
| Goal Quaternion Diff. | 4 | Quaternion representing rotation from object to goal. |
| Group Total | 11 | |
| Privileged Observations O priv  $\boldsymbol{\mathcal{O}_{\text{priv}}}$ | | |
| Joint Velocities | 16 | Angular velocities of robot joints. |
| Joint Torques | 16 | Actuator-applied joint torques. |
| Fingertip Forces | 12 | Net contact forces at the four fingertips. |
| Object Velocities | 6 | Linear and angular velocities of the object. |
| Physical Properties | 7 | Randomized object/robot scale and object mass. |
| Scene Gravity | 3 | Gravity vector. |
| Actuator Gains | 32 | Randomized joint stiffness and damping. |
| Action Properties | 2 | Randomized EMA parameter  \alpha and delay |
| Random Forces | 6 | External forces and torques applied to the object. |
| Group Total | 100 | |
| Total Observation Dim. | 199 | |

**说明**: TABLE X: Observation Space

#### Table 11: TABLE XI: Domain Randomization Parameters

| Parameter | Type | Distribution | Range / Details |
| --- | --- | --- | --- |
| Startup Randomization | | | |
| Robot Link Mass | Scaling | Log-Uniform |  [0.75, 1.5 ] \times[0.75,1.5] |
| Robot Link Friction | Absolute | Uniform | Static/Dynamic: [0.0, 0.3 ] [0.0,0.3] |
| Fingertip Friction | Absolute | Uniform | Static/Dynamic: [0.3, 0.8 ] [0.3,0.8] |
| Robot Restitution | Absolute | Uniform | [0.0, 0.4 ] [0.0,0.4] |
| Object Scale | Scaling | Uniform |  [0.8, 1.2 ] \times[0.8,1.2] (per axis) |
| Object Mass | Scaling | Uniform |  [0.5, 1.5 ] \times[0.5,1.5] |
| Object Friction | Absolute | Uniform | Static/Dynamic: [0.3, 0.8 ] [0.3,0.8] |
| Object Restitution | Absolute | Uniform | [0.0, 0.4 ] [0.0,0.4] |
| Reset Randomization | | | |
| Joint Stiffness | Scaling | Log-Uniform |  [0.75, 1.5 ] \times[0.75,1.5] |
| Joint Damping | Scaling | Log-Uniform |  [0.75, 1.5 ] \times[0.75,1.5] |
| Joint Friction | Scaling | Log-Uniform |  [0.75, 1.5 ] \times[0.75,1.5] |
| Joint Armature | Scaling | Log-Uniform |  [0.75, 1.5 ] \times[0.75,1.5] |
| Joint Limits | Scaling | Log-Uniform | Lower/Upper:  [0.95, 1.05 ] \times[0.95,1.05] |
| Interval Randomization | | | |
| External Forces | Additive | Impulse | Magnitude: 2.0  2.0\times, Prob.: 0.1 0.1 |
| Gravity Vector | Additive | Uniform | ± 0.5 \pm 0.5 m/s 2 every 0 – 15 15 s |

**说明**: TABLE XI: Domain Randomization Parameters

#### Table 12: TABLE XII: Teacher Policy Architecture and PPO Training Configuration

| Parameter | Value |
| --- | --- |
| Policy Architecture | |
| Actor Hidden Layers | [1024, 1024, 1024, 512 ] [1024,1024,1024,512] |
| Critic Hidden Layers | [1024, 1024, 1024, 512 ] [1024,1024,1024,512] |
| Exte. Encoder Hidden Layers | [64, 64 ] [64,64] |
| Exte. Encoder Latent Dimension | 24 24 |
| Priv. Encoder Hidden Layers | [256, 256 ] [256,256] |
| Priv. Encoder Latent Dimension | 128 128 |
| Activation Function | ELU |
| PPO Training Parameters | |
| Steps per Environment | 24 24 |
| Discount Factor $\gamma$ | 0.998 0.998 |
| GAE Parameter  \lambda | 0.95 0.95 |
| Learning Rate | 1  10 - 3 1\times 10^{-3} |
| Learning Rate Schedule | Adaptive |
| Clip Range | 0.2 0.2 |
| Value Loss Coefficient | 0.5 0.5 |
| Entropy Coefficient | 0.002 0.002 |
| Learning Epochs per Iteration | 5 5 |
| Mini-batches | 12 12 |
| Target KL Divergence | 0.01 0.01 |

**说明**: TABLE XII: Teacher Policy Architecture and PPO Training Configuration

#### Table 13: TABLE XIII: Student Policy and Distillation Hyperparameters

| Parameter | Value |
| --- | --- |
| Student Architecture | |
| Actor MLP | [1024, 1024, 512, 512 ] [1024,1024,512,512] |
| Exteroceptive MLP | [256, 256 ] [256,256] |
| Exteroceptive Latent Dim | 64 64 |
| Privileged Latent Dim | 256 256 |
| Activation Function | ELU |
| Initial Action Noise Std. | 0.02 0.02 |
| Belief Encoder and Decoder | |
| RNN Hidden Dimension | 256 256 |
| Number of RNN Layers | 2 2 |
| Latent Hidden Dimensions | [256, 256 ] [256,256] |
| Attention Gate Dimensions | [128, 128 ] [128,128] |
| Exteroceptive Decoder MLP | [256, 256 ] [256,256] |
| Privileged Decoder MLP | [256, 256 ] [256,256] |
| Distillation Algorithm | |
| Optimizer | AdamW |
| Learning Rate | 3.0  10 - 4 3.0\times 10^{-4} |
| Number of Learning Epochs | 32 32 |
| Mini-batches | 1 1 |
| Backpropagation Length | 45 45 steps |
| DAgger Mixing Ratio | 0.9 0.9 |
| Mixing Ratio Decay | 0.95 0.95 |
| Reconstruction Loss Coefficient | 0.2 0.2 |
| Decoder L1 Loss Coefficient | 0.2 0.2 |
| Decoder Exteroceptive Loss Coefficient | 2.0 2.0 |

**说明**: TABLE XIII: Student Policy and Distillation Hyperparameters

#### Table 14: TABLE XIV: Perception Noise Generator Parameters

| Noise Component | Distribution / Type | Parameters / Range |
| --- | --- | --- |
| Object Position | | |
| Temporal Downsampling | Discrete Sampling (k k) | Update period: 1 1 – 3 3 steps |
| Stochastic Jitter | Bernoulli Delay (p p) | Delay probability: 0.0 0.0 – 0.1 0.1 |
| Tracking Failure | Random Replacement | Failure probability: 0.0 0.0 – 0.3 0.3 |
| Biased Noise | Additive Uniform | Noise: U  [- 12, 12 ]  $\mathcal{U}[-12,12] mm Bias: U [- 12, 12 ] \mathcal{U}[-12,12] mm$ |
| Object Orientation | | |
| Temporal Downsampling | Discrete Sampling (k k) | Update period: 1 1 – 3 3 steps |
| Stochastic Jitter | Bernoulli Delay (p p) | Delay probability: 0.0 0.0 – 0.1 0.1 |
| Tracking Failure | Random Replacement | Failure probability: 0.0 0.0 – 0.3 0.3 |
| Biased Noise | Additive Uniform | Noise: U  [- 1, 1 ]  $\mathcal{U}[-1,1] deg Bias: U [- 0.1, 0.1 ] \mathcal{U}[-0.1,0.1] deg$ |

**说明**: TABLE XIV: Perception Noise Generator Parameters

#### Table 15: TABLE XV: Image Augmentation Operators and Parameters

| Augmentation | Probability | Range |
| --- | --- | --- |
| Photometric Augmentations | | |
| Color Jitter | 0.2 0.2 | [0.8, 1.2 ] [0.8,1.2] |
| Hue Shift | 0.2 0.2 | [- 0.2, 0.2 ] [-0.2,0.2] |
| Brightness Scaling | 0.5 0.5 | [0.5, 1.5 ] [0.5,1.5] |
| Contrast Scaling | 0.5 0.5 | [0.5, 1.5 ] [0.5,1.5] |
| Gamma | 0.5 0.5 | [0.5, 1.5 ] [0.5,1.5] |
| Saturation Scaling | 0.5 0.5 | [0.5, 1.5 ] [0.5,1.5] |
| Sensor and Noise Augmentations | | |
| ISO-like Noise | 0.25 0.25 | |
| Motion Blur | 0.5 0.5 | Kernel size: 3 3 – 17 17 |
| Blur and Filtering | | |
| Box Blur | 0.5 0.5 | Kernel Size: 3 3 – 5 5 |
| Binary Opening | 1.0 1.0 | Kernel size: 3 3 |

**说明**: TABLE XV: Image Augmentation Operators and Parameters
## 实验解读

- 评价重点:围绕 接触推理、灵巧操作、in-hand-manipulation、motion-tracking、强化学习,优先看真实机器人结果、扰动恢复、接触稳定性、控制误差、成功率、实时性和 sim-to-real 差距。
- System0 视角:实验结论不只看最终成功率,还要看闭环过程中是否能处理状态估计误差、执行延迟、碰撞/接触突变和外部扰动。
- 与上层系统的关系:如果方法能在快闭环内稳定工作,它可以作为 System1 技能或 System2 规划的底层执行保障;如果只在离线或慢速设定下成立,则还不能直接作为身体级快闭环模块。
## 批判性思考

### 优点

- 论文把 接触推理、灵巧操作、in-hand-manipulation、motion-tracking、强化学习 放到机器人身体级闭环里讨论,和只做高层语义推理的工作相比,更接近真实机器人部署时会遇到的状态、动力学和安全约束。
- 若论文同时给出公式、系统结构图、硬件实验或消融结果,这些证据可以帮助判断方法是否真的能支撑毫秒到百毫秒级反应。

### 局限

- 需要重点核查控制频率、推理延迟、传感延迟、失败恢复、硬件磨损、未见场景泛化和安全边界;这些因素决定它能否成为可靠的 System0 模块。
- 对学习型方法,还要区分训练环境中的鲁棒性和真实部署中的鲁棒性,尤其是接触丰富任务、动态运动任务和人形机器人任务。
## 论文定位

- 研究对象:ViserDex: Visual Sim-to-Real for Robust Dexterous In-hand Reorientation。
- 关键词:接触推理、灵巧操作、in-hand-manipulation、motion-tracking、强化学习、机器人操作、鲁棒控制、仿真到真实迁移、状态估计。
- System0 关联:关注机器人身体级快反应链路中“状态/接触/局部感知 -> 约束/优化/策略 -> 动作”的短闭环,不把问题停留在语言规划或任务分解层。

## 速查卡片

> [!summary] ViserDex
> - **论文**: https://www.roboticsproceedings.org/rss22/p150.pdf
> - **arXiv**: http://arxiv.org/abs/2604.11138v1
> - **arXiv HTML**: https://arxiv.org/html/2604.11138v1
