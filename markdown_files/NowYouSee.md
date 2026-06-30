---
title: "Now You See That: Learning End-to-End Humanoid Locomotion from Raw Pixels"
method_name: "Now You See That"
authors: ["Wandong Sun"]
year: 2026
venue: "RSS"
tags: ["robust-control", "legged-locomotion", "adaptive-control", "imitation-learning", "robot-generalization", "humanoid", "sim-to-real"]
image_source: "online"
arxiv_html: "https://arxiv.org/html/2602.06382v2"
---
# Now You See That
## 一句话总结

> Now You See That: Learning End-to-End Humanoid Locomotion from Raw Pixels 主要落在 [[adaptive-control]]、[[人形机器人]]、[[足式运动]]、[[运动模仿]] 这一类问题上;从 System0 视角看,它的价值在于把身体状态、接触变化、局部感知或动力学约束接到机器人低层闭环里,让动作生成更稳定、安全、实时。
## 核心贡献

1. 围绕 **Now You See That: Learning End-to-End Humanoid Locomotion from Raw Pixels** 建立了一个与 adaptive-control、人形机器人、足式运动、运动模仿、鲁棒控制、仿真到真实迁移 相关的机器人问题设定,重点不是高层语义规划,而是身体级闭环中的状态变化、接触变化或控制约束。
2. 方法层面的关键价值在于把感知、动力学、约束优化、策略学习或硬件反馈组织成可执行的低层动作生成机制,适合放进 System0 的快反应模块中理解。
3. 对系统集成而言,这类工作补上了 System1 技能选择与电机级控制之间的一层:它负责把上层目标变成动态可行、可恢复、可安全过滤的动作。
4. 若论文包含真实机器人、接触任务或仿真到真实迁移,核心应看它是否报告了闭环稳定性、扰动恢复、硬件延迟、失败模式和跨场景泛化。
## 问题背景

这篇论文可以放在 System0 的“具身快系统”背景下阅读:机器人需要在很短时间内根据本体状态、接触、力/触觉、局部视觉或动力学预测做出动作,而不是等待慢速规划器重新推理整个任务。adaptive-control、人形机器人、足式运动、运动模仿、鲁棒控制、仿真到真实迁移、terrain-adaptation 相关问题的难点通常不在“知道要做什么”,而在动作执行过程中状态会快速偏离、接触会突然改变、约束会实时收紧,低层系统必须及时修正。

### System0 关联

精读时优先看三件事:第一,论文使用哪些快速反馈量,例如关节状态、接触状态、力觉、触觉、深度或局部视觉;第二,反馈如何进入 MPC、WBC、策略网络、安全过滤器或恢复控制器;第三,输出是否直接影响机器人下一步动作,而不是只服务离线分析或高层任务规划。
## 方法详解

### System0 视角解析

从 System0 角度看,这篇论文的方法部分应当围绕 adaptive-control、人形机器人、足式运动、运动模仿、鲁棒控制、仿真到真实迁移、terrain-adaptation、visuomotor-policy 来读。核心问题是:系统如何从快速变化的机器人状态中构造控制输入,如何处理接触、碰撞、约束、模型误差或执行延迟,以及如何把这些信息变成下一时刻的动作。

如果方法里有优化问题,重点看目标函数、约束和求解频率;如果方法里有学习策略,重点看观测空间、动作空间、训练扰动和部署时的闭环频率;如果方法里有硬件或传感器,重点看传感链路是否真的进入控制回路。这样读可以把论文和 System2 语言规划、System1 技能编排区分开来。

### 关键公式与机制

#### 公式 1: [[动力学或策略机制]]

$$\Phi_{\text{amp}}=[\mathbf{q}_{\text{rel}},\dot{\mathbf{q}}_{\text{rel}},\mathbf{v}_{\text{torso}}^{b},\omega_{\text{torso}}^{b},\mathbf{g}^{b},\mathbf{p}_{\text{body}}^{b},\mathbf{q}_{\text{body}}^{b}]$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 2: [[优化目标/约束]]

$$
\mathcal{L}_{\text{behavior}}=\mathbb{E}_{s_{t}}\left[\left\lVert \mu_{\text{deploy}}(s_{t})-\mu_{\text{priv}}(s_{t})\right \rVert_{2}^{2}\right]
$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 3: [[优化目标/约束]]

$$
\mathcal{L}_{\text{denoise}}=\mathbb{E}_{d,\tilde{d}}\left[\left\lVert E(d)-E(\tilde{d})\right \rVert_{2}^{2}\right]
$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 4: [[优化目标/约束]]

$$
\mathcal{L}_{\text{kl}}=\mathrm{KL}\!\left(\mathcal{N}(\boldsymbol{\mu},\mathrm{diag}(\boldsymbol{\sigma}^{2}))\,\lVert \,\mathcal{N}(\mathbf{0},\mathbf{I})\right)
$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 5: [[优化目标/约束]]

$$
\mathcal{L}_{\text{total}}=\mathcal{L}_{\text{behavior}}+\lambda_{\text{denoise}}\mathcal{L}_{\text{denoise}}+\lambda_{\text{kl}}\mathcal{L}_{\text{kl}}
$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 6: [[动力学或策略机制]]

$$P=\frac{1}{T}\sum_{t}\lVert \tau_{t}\odot\dot{q}_{t} \rVert_{2}$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 7: [[动力学或策略机制]]

$$r_{\text{vel}}^{\text{exp}}=\exp\left(-\frac{\lVert \mathbf{v}_{xy}^{\text{cmd}}-\mathbf{v}_{xy}^{\text{robot}} \rVert^{2}}{\sigma^{2}}\right)$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 8: [[优化目标/约束]]

$$r_{\text{vel}}^{\text{dir}}=\frac{\min\left(\mathbf{v}_{xy}^{\text{robot}}\cdot\hat{\mathbf{d}}_{\text{cmd}},\,\lVert \mathbf{v}_{xy}^{\text{cmd}} \rVert\right)}{\lVert \mathbf{v}_{xy}^{\text{cmd}} \rVert+\epsilon}$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 9: [[优化目标/约束]]

$$r_{\text{contact}}=\sum_{f\in\{\text{left},\text{right}\}}\mathbb{1}_{\text{contact}}^{f}\cdot\text{std}\left(\text{clip}(h_{f}^{\text{scan}},-h_{\max},h_{\max})\right)$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 10: [[动力学或策略机制]]

$$d_{\text{fused}}(u,v)=\begin{cases}d_{\text{left}}(u,v)&\text{if}| d_{ $\text{left}}-d_{\text{right}}(u_{r},v)$ |<\tau\cdot d_{\text{left}}\\ 0&\text{otherwise}\end{cases}$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。
## 关键图表

### Figure 1: Pipeline / core system figure: Diverse terrain types used during training. Each terrain type contains 20 difficulty l

![Figure 1](https://arxiv.org/html/2602.06382v2/x2.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Pipeline / core system figure: Diverse terrain types used during training. Each terrain type contains 20 difficulty l”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 2: Visualization of the depth augmentation pipeline. Starting from clean left and right d

![Figure 2](https://arxiv.org/html/2602.06382v2/x3.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Visualization of the depth augmentation pipeline. Starting from clean left and right d”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 3: Method Overview. Our framework consists of two stages: (1) Privileged RL Training: A

![Figure 3](https://arxiv.org/html/2602.06382v2/x4.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Method Overview. Our framework consists of two stages: (1) Privileged RL Training: A”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### 论文表格

#### Table 1: TABLE I: Comparison of perceptive humanoid locomotion methods. Representation indicates the terrain perception approach.

| Method | Representation | End-to-End | Noise Modeling | Long-Term Deploy | Fine Locomotion | Extreme Parkour |
| --- | --- | --- | --- | --- | --- | --- |
| Long et al. [26 ] | Elevation Map | ✗ | Moderate | ✗ | ✓ | ✗ |
| Sun et al. [47 ] | Elevation Map | ✗ | Moderate | ✗ | ✓ | ✗ |
| He et al. [12 ] | Elevation Map | ✗ | Moderate | ✗ | ✓ | ✗ |
| Ben et al. [2 ] | Voxel | ✗ | Moderate | ✓ | ✓ | ✗ |
| Zhuang et al. [60 ] | End-to-End Vision | ✓ | Moderate | ✓ | ✗ | ✓ |
| Song et al. [45 ] | Vision-to-Elevation | ✗ | Moderate | ✓ | ✓ | ✗ |
| Ours | End-to-End Vision | ✓ | Comprehensive | ✓ | ✓ | ✓ |

**说明**: TABLE I: Comparison of perceptive humanoid locomotion methods. Representation indicates the terrain perception approach. Noise Modeling indicates the comprehensiveness of depth sensor simulation. Long-Term Deploy indicates drift-free operation capability. Fine Locomotion indicates support for precise movements like stair climbing. Extreme Parkour indicates support for dynamic maneuvers across challenging obstacles.

#### Table 2: TABLE II: Vision Augmentation Pipeline: Operations, Schedule, Parameters and Descriptions

| Order | Operation | Schedule | Parameters | Description |
| --- | --- | --- | --- | --- |
| Preparation Stage | | | | |
| 1 | Camera intrinsics | Once at startup | s h, s v ∼ U  (0.90, 1.10) s_{ $\text{h}},s_{\text{v}}\sim\mathcal{U}(0.90,1.10)$ | Focal length variations |
| 2 | Camera extrinsics | Once at startup | $\mathbf{p}\sim\mathcal{U}(-0.05,0.05)^{3} m$ | Mounting position offset |
| | | | $\boldsymbol{\theta}\sim\mathcal{U}(-0.10,0.10)^{3} rad$ | Mounting orientation offset |
| 3 | Observation delay | Once at startup | d frame ∼ U  [2, 4 ] d_{ $\text{frame}}\sim\mathcal{U}[2,4] frames$ | Processing pipeline latency |
| Processing Pipeline | | | | |
| 1 | Stereo fusion | Per frame | $\tau\sim\mathcal{U}[0.05,0.20]$ | Disparity consistency check |
| 2 | Random convolution | Per frame | w i, j, k, l ∼ U  (- 0.05, 0.05) w_{i,j,k,l}\sim $\mathcal{U}(-0.05,0.05)$ | Optical aberrations |
| 3 | Gaussian noise | Per frame | c 0, c 1, c 2 ∼ U  (- 0.03, 0.03) c_{0},c_{1},c_{2}\sim $\mathcal{U}(-0.03,0.03)$ | Distance-dependent noise |
| 4 | Perlin noise | Per frame | c 0 p, c 1 p, c 2 p ∼ U  (- 0.02, 0.02) c_{0}^{ $\text{p}},c_{1}^{\text{p}},c_{2}^{\text{p}}\sim\mathcal{U}(-0.02,0.02)$ | Time-dependent noise |
| 5 | Scale randomization | Per frame | s i ∼ U  (0.90, 1.10) s_{i}\sim $\mathcal{U}(0.90,1.10)$ | Calibration errors |
| 6 | Pixel failures | Per frame | p zero = p max = 0.001 p_{ $\text{zero}}=p_{\text{max}}=0.001$ | Dead/saturated pixels |
| 7 | Depth clipping | Per frame | [d min, d max ] = [0.3, 2.0 ] [d_{\min},d_{\max}]=[0.3,2.0] m | Valid sensing range |
| 8 | Spatial cropping | Per frame | (t, b, l, r) = (3, 3, 4, 4) (t,b,l,r)=(3,3,4,4) pixels | Edge distortion removal |

**说明**: TABLE II: Vision Augmentation Pipeline: Operations, Schedule, Parameters and Descriptions

#### Table 3: TABLE III: Robot Dynamics Domain Randomization

| Parameter | Range |
| --- | --- |
| Contact Properties | |
| Static/Dynamic friction | (0.4, 1.2) (0.4,1.2) |
| Restitution | (0.0, 0.4) (0.0,0.4) |
| Body Properties | |
| Mass scaling | (0.8, 1.2) (0.8,1.2) |
| COM offset | ± 0.03 \pm 0.03 m |
| Default joint offset | ± 0.03 \pm 0.03 rad |
| Actuator Dynamics | |
| Armature scaling | (0.5, 1.5) (0.5,1.5) |
| Stiffness/Damping scaling | (0.9, 1.1) (0.9,1.1) |
| External Disturbances | |
| Push velocity | ± 1.2 \pm 1.2 m/s |
| IMU bias | ± 0.04 \pm 0.04 rad/s |

**说明**: TABLE III: Robot Dynamics Domain Randomization

#### Table 4: TABLE IV: Performance on RDT-Bench across four terrain configurations. SR: Success Rate (%), P: Average Power ( 10 1 \

| | Stairs Up | Stairs Down | Gaps | Platform | Average | | | | | | | | | | |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Method | SR | P ↓ \downarrow | PDR ↓ \downarrow | SR | P ↓ \downarrow | PDR ↓ \downarrow | SR | P ↓ \downarrow | PDR ↓ \downarrow | SR | P ↓ \downarrow | PDR ↓ \downarrow | SR | P ↓ \downarrow | PDR ↓ \downarrow |
| No Augmentation | 45.3 ±1.8 | 52.3 ±1.2 | 68.4 ±2.1 | 42.1 ±2.0 | 48.7 ±1.1 | 71.2 ±2.3 | 44.8 ±1.7 | 51.2 ±1.0 | 69.5 ±1.9 | 39.7 ±2.2 | 54.6 ±1.3 | 74.3 ±2.5 | 43.0 ±1.9 | 51.7 ±1.2 | 70.9 ±2.2 |
| Partial Aug. | 61.2 ±1.5 | 43.8 ±0.9 | 42.6 ±1.4 | 57.4 ±1.6 | 41.2 ±0.8 | 45.3 ±1.5 | 59.8 ±1.4 | 42.5 ±0.7 | 43.8 ±1.3 | 53.6 ±1.8 | 46.1 ±1.0 | 48.7 ±1.6 | 58.0 ±1.6 | 43.4 ±0.9 | 45.1 ±1.5 |
| Standard DR | 65.4 ±1.3 | 41.5 ±0.8 | 38.2 ±1.2 | 62.1 ±1.4 | 39.4 ±0.7 | 40.6 ±1.3 | 63.7 ±1.2 | 40.2 ±0.6 | 39.1 ±1.1 | 56.8 ±1.6 | 44.3 ±0.9 | 45.2 ±1.4 | 62.0 ±1.4 | 41.4 ±0.8 | 40.8 ±1.3 |
| Humanoid Parkour | 74.2 ±1.1 | 38.2 ±0.7 | 28.5 ±1.0 | 71.5 ±1.2 | 36.5 ±0.6 | 30.2 ±1.1 | 72.8 ±1.0 | 37.1 ±0.5 | 29.4 ±0.9 | 65.5 ±1.4 | 40.8 ±0.8 | 35.6 ±1.2 | 71.0 ±1.2 | 38.2 ±0.7 | 30.9 ±1.1 |
| Direct RL | 57.3 ±1.9 | 48.6 ±1.1 | 55.2 ±1.8 | 53.8 ±2.1 | 45.8 ±1.0 | 58.4 ±2.0 | 55.6 ±1.8 | 47.2 ±0.9 | 56.8 ±1.7 | 49.3 ±2.3 | 51.4 ±1.2 | 62.1 ±2.2 | 54.0 ±2.0 | 48.3 ±1.1 | 58.1 ±1.9 |
| Single Critic/Disc. | 85.6 ±0.8 | 34.6 ±0.5 | 18.3 ±0.6 | 82.3 ±0.9 | 32.8 ±0.4 | 20.1 ±0.7 | 83.8 ±0.7 | 33.5 ±0.4 | 19.2 ±0.5 | 76.3 ±1.1 | 37.8 ±0.6 | 25.4 ±0.8 | 82.0 ±0.9 | 34.7 ±0.5 | 20.8 ±0.7 |
| BC Only | 89.2 ±0.7 | 32.4 ±0.4 | 15.6 ±0.5 | 86.5 ±0.8 | 30.6 ±0.4 | 17.2 ±0.6 | 87.8 ±0.6 | 31.2 ±0.3 | 16.3 ±0.4 | 80.5 ±0.9 | 35.2 ±0.5 | 21.8 ±0.7 | 86.0 ±0.8 | 32.4 ±0.4 | 17.7 ±0.6 |
| Ours | 99.2 ±0.3 | 28.5 ±0.3 | 5.2 ±0.2 | 98.6 ±0.4 | 27.2 ±0.3 | 6.1 ±0.3 | 99.3 ±0.2 | 25.8 ±0.2 | 4.5 ±0.2 | 98.4 ±0.5 | 29.4 ±0.4 | 7.2 ±0.3 | 98.9 ±0.4 | 27.7 ±0.3 | 5.8 ±0.3 |

**说明**: 该表格来自论文中的公式、奖励项或实验设置；为避免 PDF 抽取噪声，网页中仅保留可渲染的核心符号和数值。

#### Table 6: TABLE VI: Ablation study on multi-critic and multi-discriminator architecture. Stairs column reports the average of asce

| | Stairs | Gaps | Platforms | | | |
| --- | --- | --- | --- | --- | --- | --- |
| Method | SR | PDR ↓ \downarrow | SR | PDR ↓ \downarrow | SR | PDR ↓ \downarrow |
| Single Critic/Disc. | 84.0 ±0.5 | 19.2 ±0.3 | 83.8 ±0.4 | 19.2 ±0.3 | 76.3 ±0.6 | 25.4 ±0.4 |
| Multi-Critic/Disc. (Ours) | 98.9 ±0.4 | 5.7 ±0.2 | 99.3 ±0.2 | 4.5 ±0.2 | 98.4 ±0.5 | 7.2 ±0.3 |
|  \Delta | +14.9 | -13.5 | +15.5 | -14.7 | +22.1 | -18.2 |

**说明**: 该表格来自论文中的公式、奖励项或实验设置；为避免 PDF 抽取噪声，网页中仅保留可渲染的核心符号和数值。

| Configuration | SR (%) | P ( 10 1 \times 10^{1} W) ↓ \downarrow | PDR (%) ↓ \downarrow |
| --- | --- | --- | --- |
| Full | 98.9 ±0.4 | 27.7 ±0.3 | 5.8 ±0.2 |
| w/o L denoise  $\mathcal{L}_{\text{denoise}}$ | 93.4 ±0.5 | 32.3 ±0.4 | 13.3 ±0.3 |
| w/o L kl  $\mathcal{L}_{\text{kl}}$ | 96.1 ±0.4 | 29.5 ±0.3 | 8.9 ±0.3 |
| BC Only | 86.0 ±0.6 | 32.4 ±0.4 | 17.7 ±0.4 |

**说明**: 该表格来自论文中的公式、奖励项或实验设置；为避免 PDF 抽取噪声，网页中仅保留可渲染的核心符号和数值。

#### Table 8: TABLE VIII: Real-world deployment success rates across 15 trials per scenario.

| Scenario | Success | Rate (%) |
| --- | --- | --- |
| Outdoor Stairs (up) | 15/15 | 100.0 |
| Outdoor Stairs (down) | 13/15 | 86.7 |
| Outdoor Platform (up) | 15/15 | 100.0 |
| Outdoor Platform (down) | 15/15 | 100.0 |
| Extended Staircase | 15/15 | 100.0 |
| Wide Gap Crossing | 15/15 | 100.0 |
| Overall | 88/90 | 97.8 |

**说明**: TABLE VIII: Real-world deployment success rates across 15 trials per scenario.

#### Table 9: TABLE IX: Teacher Policy Network Architecture

| Component | Configuration |
| --- | --- |
| Height Scan Encoder | |
| Input Layer | Height Scan (21  33 = 693) |
| Hidden Layer 1 | Linear(693 → 512) + SiLU |
| Hidden Layer 2 | Linear(512 → 256) + SiLU |
| Output Layer | Linear(256 → 128) |
| Recurrent Module | |
| Input | Concat(Proprio[96], Encoder[128]) = 224 |
| RNN Type | GRU (1 layer) |
| Hidden Dimension | 256 |
| Actor MLP | |
| Input Layer | GRU Output (256) |
| Hidden Layer 1 | Linear(256 → 512) + SiLU |
| Hidden Layer 2 | Linear(512 → 256) + SiLU |
| Hidden Layer 3 | Linear(256 → 128) + SiLU |
| Output Layer | Linear(128 → 29) |
| Policy Distribution | |
| Distribution Type | Gaussian |
| Initial Noise Std | 1.0 |

**说明**: TABLE IX: Teacher Policy Network Architecture

#### Table 10: TABLE X: Student Policy Network Architecture

| Component | Configuration |
| --- | --- |
| Depth Image Encoder (CNN) | |
| Input Layer | Depth Image (1  24  32) |
| Conv Layer 1 | Conv2d(1 → 32, k=3, s=2, p=1) + SiLU |
| Conv Layer 2 | Conv2d(32 → 64, k=3, s=2, p=1) + SiLU |
| Conv Layer 3 | Conv2d(64 → 128, k=3, s=2, p=1) + SiLU |
| Flatten | 128  3  4 = 1536 |
| Output Layer | Linear(1536 → 128) |
| Recurrent Module | |
| Input | Concat(Proprio[96], Encoder[128]) = 224 |
| RNN Type | GRU (1 layer) |
| Hidden Dimension | 256 |
| Actor MLP | |
| Input Layer | GRU Output (256) |
| Hidden Layer 1 | Linear(256 → 512) + SiLU |
| Hidden Layer 2 | Linear(512 → 256) + SiLU |
| Hidden Layer 3 | Linear(256 → 128) + SiLU |
| Output Layer | Linear(128 → 29) |
| Policy Distribution | |
| Distribution Type | Gaussian |
| Constant Noise Std | 0.1 |

**说明**: TABLE X: Student Policy Network Architecture

#### Table 11: TABLE XI: Multi-Critic Network Architecture

| Component | Configuration |
| --- | --- |
| Height Scan Encoder (Shared) | |
| Input Layer | Height Scan (21  33 = 693) |
| Hidden Layer 1 | Linear(693 → 512) + SiLU |
| Hidden Layer 2 | Linear(512 → 256) + SiLU |
| Output Layer | Linear(256 → 128) |
| Recurrent Module (Shared) | |
| Input | Concat(Proprio[96], Encoder[128]) = 224 |
| RNN Type | GRU (1 layer) |
| Hidden Dimension | 256 |
| Critic MLP (Shared Backbone) | |
| Input Layer | GRU Output (256) |
| Hidden Layer 1 | Linear(256 → 512) + SiLU |
| Hidden Layer 2 | Linear(512 → 256) + SiLU |
| Hidden Layer 3 | Linear(256 → 128) + SiLU |
| Terrain-Specific Output Heads | |
| Stair Head | Linear(128 → 1) |
| Gap Head | Linear(128 → 1) |
| General Head | Linear(128 → 1) |

**说明**: TABLE XI: Multi-Critic Network Architecture

#### Table 12: TABLE XII: Policy Distillation Configuration

| Parameter | Value |
| --- | --- |
| (a) Training Hyperparameters | |
| Steps per environment per iteration | 800 |
| Total training iterations | 4000 |
| Learning rate schedule | One Cycle [44 ] |
| Initial learning rate | 1  10 - 3 1\times 10^{-3} |
| Div factor (peak/init) | 10.0 |
| Final div factor (final/init) | 50.0 |
| Gradient accumulation steps | 10 |
| Max gradient norm | 1.0 |
| (b) Loss Coefficients | |
| Behavior loss weight | 1.0 1.0 |
| Denoising loss coeff. denoise \lambda $\text{denoise}}$ | 0.1 0.1 |
| $\text{kl}}$ | 0.1 0.1 |
| EMA decay | 0.997 0.997 |

**说明**: TABLE XII: Policy Distillation Configuration

#### Table 13: TABLE XIII: CycleGAN Translation Quality Metrics. Translation Quality: FID and KID measure distributional distance betw

| Metric | Sim →  $\rightarrow Real$ | Real →  $\rightarrow Sim$ |
| --- | --- | --- |
| Translation Quality | | |
| FID ↓ \downarrow (no translation baseline) | 67.2 | |
| FID ↓ \downarrow (after CycleGAN) | 23.4 | 21.7 |
| KID ( 10 - 3 \times 10^{-3}) ↓ \downarrow | 8.2 | 7.6 |
| Cycle Consistency | | |
| SSIM ↑ \uparrow | 0.89 | 0.91 |
| PSNR ↑ \uparrow | 28.3 dB | 29.1 dB |
| LPIPS ↓ \downarrow | 0.12 | 0.11 |

**说明**: 该表格来自论文中的公式、奖励项或实验设置；为避免 PDF 抽取噪声，网页中仅保留可渲染的核心符号和数值。

#### Table 14: TABLE XIV: CycleGAN Training Configuration

| Parameter | Value |
| --- | --- |
| (a) Dataset Statistics | |
| Total real-world recording time | ∼ \sim 2 hours |
| Total depth frames | ∼ \sim 200,000 |
| Frame resolution | 24  \times 32 |
| Train/Val split | 90% / 10% |
| (b) Network Architecture | |
| Generator architecture | ResNet (9 blocks) |
| Discriminator architecture | PatchGAN (70  \times 70) |
| Number of filters (first layer) | 64 |
| Normalization | Instance Normalization |
| (c) Training Hyperparameters | |
| Batch size | 64 |
| Total epochs | 200 |
| Optimizer | Adam |
| Learning rate | 2  10 - 4 2\times 10^{-4} |
|  1 \beta_{1},  2 \beta_{2} | 0.5, 0.999 |
| Learning rate decay | Linear (after epoch 100) |
| (d) Loss Weights | |
| Adversarial loss weight | 1.0 |
| Cycle consistency loss weight  c  y  c \lambda_{cyc} | 10.0 |
| Identity loss weight  i  d  t \lambda_{idt} | 0.5 |

**说明**: TABLE XIV: CycleGAN Training Configuration

#### Table 15: TABLE XV: Terrain-Specific Reward Configuration

| Reward | Stairs/Platforms | Gaps | Rough |
| --- | --- | --- | --- |
| r vel exp r_{ $\text{vel}}^{\text{exp}}$ | ✓ | – | ✓ |
| r vel dir r_{ $\text{vel}}^{\text{dir}}$ | – | ✓ | – |
| r contact r_{ $\text{contact}}$ | ✓ | – | – |

**说明**: TABLE XV: Terrain-Specific Reward Configuration

#### Table 16: TABLE XVI: Cross-Platform Validation on Unitree G1

| Scenario | Success | Rate (%) |
| --- | --- | --- |
| Stair Ascending | 15/15 | 100.0 |

**说明**: TABLE XVI: Cross-Platform Validation on Unitree G1
## 实验解读

- 评价重点:围绕 adaptive-control、人形机器人、足式运动、运动模仿、鲁棒控制,优先看真实机器人结果、扰动恢复、接触稳定性、控制误差、成功率、实时性和 sim-to-real 差距。
- System0 视角:实验结论不只看最终成功率,还要看闭环过程中是否能处理状态估计误差、执行延迟、碰撞/接触突变和外部扰动。
- 与上层系统的关系:如果方法能在快闭环内稳定工作,它可以作为 System1 技能或 System2 规划的底层执行保障;如果只在离线或慢速设定下成立,则还不能直接作为身体级快闭环模块。
## 批判性思考

### 优点

- 论文把 adaptive-control、人形机器人、足式运动、运动模仿、鲁棒控制 放到机器人身体级闭环里讨论,和只做高层语义推理的工作相比,更接近真实机器人部署时会遇到的状态、动力学和安全约束。
- 若论文同时给出公式、系统结构图、硬件实验或消融结果,这些证据可以帮助判断方法是否真的能支撑毫秒到百毫秒级反应。

### 局限

- 需要重点核查控制频率、推理延迟、传感延迟、失败恢复、硬件磨损、未见场景泛化和安全边界;这些因素决定它能否成为可靠的 System0 模块。
- 对学习型方法,还要区分训练环境中的鲁棒性和真实部署中的鲁棒性,尤其是接触丰富任务、动态运动任务和人形机器人任务。
## 论文定位

- 研究对象:Now You See That: Learning End-to-End Humanoid Locomotion from Raw Pixels。
- 关键词:adaptive-control、人形机器人、足式运动、运动模仿、鲁棒控制、仿真到真实迁移、terrain-adaptation、visuomotor-policy。
- System0 关联:关注机器人身体级快反应链路中“状态/接触/局部感知 -> 约束/优化/策略 -> 动作”的短闭环,不把问题停留在语言规划或任务分解层。

## 速查卡片

> [!summary] Now You See That
> - **论文**: https://www.roboticsproceedings.org/rss22/p027.pdf
> - **arXiv**: http://arxiv.org/abs/2602.06382v2
> - **arXiv HTML**: https://arxiv.org/html/2602.06382v2
