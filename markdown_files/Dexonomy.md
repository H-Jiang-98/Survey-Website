---
title: "Dexonomy: Synthesizing All Dexterous Grasp Types in a Grasp Taxonomy"
method_name: "Dexonomy"
authors: ["Jiayi Chen"]
year: 2025
venue: "RSS"
tags: ["contact-reasoning", "robot-manipulation", "contact-rich-manipulation", "robot-generalization", "dexterous-manipulation"]
image_source: "online"
arxiv_html: "https://arxiv.org/html/2504.18829v2"
---
# Dexonomy
## 一句话总结

> Dexonomy: Synthesizing All Dexterous Grasp Types in a Grasp Taxonomy 主要落在 [[接触推理]]、[[接触丰富操作]]、[[dexterous-grasping]]、[[灵巧操作]] 这一类问题上;从 System0 视角看,它的价值在于把身体状态、接触变化、局部感知或动力学约束接到机器人低层闭环里,让动作生成更稳定、安全、实时。
## 核心贡献

1. 围绕 **Dexonomy: Synthesizing All Dexterous Grasp Types in a Grasp Taxonomy** 建立了一个与 接触推理、接触丰富操作、dexterous-grasping、灵巧操作、grasping、robot-generalization 相关的机器人问题设定,重点不是高层语义规划,而是身体级闭环中的状态变化、接触变化或控制约束。
2. 方法层面的关键价值在于把感知、动力学、约束优化、策略学习或硬件反馈组织成可执行的低层动作生成机制,适合放进 System0 的快反应模块中理解。
3. 对系统集成而言,这类工作补上了 System1 技能选择与电机级控制之间的一层:它负责把上层目标变成动态可行、可恢复、可安全过滤的动作。
4. 若论文包含真实机器人、接触任务或仿真到真实迁移,核心应看它是否报告了闭环稳定性、扰动恢复、硬件延迟、失败模式和跨场景泛化。
## 问题背景

这篇论文可以放在 System0 的“具身快系统”背景下阅读:机器人需要在很短时间内根据本体状态、接触、力/触觉、局部视觉或动力学预测做出动作,而不是等待慢速规划器重新推理整个任务。接触推理、接触丰富操作、dexterous-grasping、灵巧操作、grasping、robot-generalization、机器人操作 相关问题的难点通常不在“知道要做什么”,而在动作执行过程中状态会快速偏离、接触会突然改变、约束会实时收紧,低层系统必须及时修正。

### System0 关联

精读时优先看三件事:第一,论文使用哪些快速反馈量,例如关节状态、接触状态、力觉、触觉、深度或局部视觉;第二,反馈如何进入 MPC、WBC、策略网络、安全过滤器或恢复控制器;第三,输出是否直接影响机器人下一步动作,而不是只服务离线分析或高层任务规划。
## 方法详解

### System0 视角解析

从 System0 角度看,这篇论文的方法部分应当围绕 接触推理、接触丰富操作、dexterous-grasping、灵巧操作、grasping、robot-generalization、机器人操作、scalable-robot-learning 来读。核心问题是:系统如何从快速变化的机器人状态中构造控制输入,如何处理接触、碰撞、约束、模型误差或执行延迟,以及如何把这些信息变成下一时刻的动作。

如果方法里有优化问题,重点看目标函数、约束和求解频率;如果方法里有学习策略,重点看观测空间、动作空间、训练扰动和部署时的闭环频率;如果方法里有硬件或传感器,重点看传感链路是否真的进入控制回路。这样读可以把论文和 System2 语言规划、System1 技能编排区分开来。

### 关键公式与机制

#### 公式 1: [[动力学或策略机制]]

$$\mathbf{n}_{i}=\mathbf{d}_{i}\times\mathbf{c}_{i}$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 2: [[动力学或策略机制]]

$$\mathbf{w}_{i}=\mathbf{J}_{o,i}^{T}\mathbf{x}_{i}$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 3: [[动力学或策略机制]]

$$e=\lVert \sum_{i=1}^{m}\mathbf{J}_{o,i}^{T}\mathbf{f}_{i}-\mathbf{g} \rVert^{2}$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 4: [[动力学或策略机制]]

$$L=k_{p}\sum_{i=1}^{m}\lVert \mathbf{p}^{h}_{i}-\mathbf{p}^{o}_{i} \rVert^{2}+k_{n}\sum_{i=1}^{m}\lVert \mathbf{n}^{h}_{i}-\mathbf{n}^{o}_{i} \rVert^{2}$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 5: [[动力学或策略机制]]

$$\mathbf{f}_{i}=k_{f}(\mathbf{p}^{h}_{i}-\mathbf{p}^{o}_{i}),~~~\mathbf{\tau}=\sum_{i=1}^{m}\mathbf{J}_{h,i}^{T}\mathbf{f}_{i}$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 6: [[动力学或策略机制]]

$$=\begin{bmatrix}\mathbf{n}_{i}&\mathbf{d}_{i}&\mathbf{c}_{i}\\ \mathbf{p}_{i}\times\mathbf{n}_{i}&\mathbf{p}_{i}\times\mathbf{d}_{i}&\mathbf{p}_{i}\times\mathbf{c}_{i}\\ \end{bmatrix}\in\mathbb{R}^{6\times 3}$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 7: [[动力学或策略机制]]

$$=\left\{\mathbf{x}_{i}\in\mathbb{R}^{3}~|~0\leq x_{i,1}\leq 1,x_{i,2}^{2}+x_{i,3}^{2}\leq\mu^{2}x_{i,1}^{2}\right\}$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 8: [[优化目标/约束]]

$$(\mathbf{f}_{1},...,\mathbf{f}_{m})=\ {(\mathbf{x}_{1},...,\mathbf{x}_{m})}{\arg\min}~~~$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 9: [[动力学或策略机制]]

$$\lVert \sum_{i=1}^{m}\mathbf{J}_{o,i}^{T}\mathbf{x}_{i}-\mathbf{g} \rVert^{2}$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 10: [[动力学或策略机制]]

$$\Sigma_{i=1}^{m}x_{i,1}\geq\lambda$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。
## 关键图表

### Figure 1: The pipeline of Dexonomy. (1) Grasp Template Library initially requires one human-ann

![Figure 1](https://arxiv.org/html/2504.18829v2/figures/pipeline.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“The pipeline of Dexonomy. (1) Grasp Template Library initially requires one human-ann”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 2: Visualization of Type-Unaware Grasps. Our method synthesizes human-like and stable gr

![Figure 2](https://arxiv.org/html/2504.18829v2/figures/fingertip_baseline.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Visualization of Type-Unaware Grasps. Our method synthesizes human-like and stable gr”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 3: Robustness to Initial Templates. Our method generates good grasps even for very nois

![Figure 3](https://arxiv.org/html/2504.18829v2/x2.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Robustness to Initial Templates. Our method generates good grasps even for very nois”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### 论文表格

#### Table 1: TABLE I: Dexterous Grasp Dataset Comparison. Our large-scale dataset aims to support the study of data-driven methods f

| Dataset | Hand | Sim./Real | Objects | Grasps | Grasp Types | Force Closure | Data Type | Method |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DexGraspNet [47 ] | Shadow | IsaacGym | 5.4k | 1.32M | Random | ✓ | Grasp pose | Optimization |
| RealDex [29 ] | Shadow | Real | 52 | 59k | Random | ✗ | Motion | Teleoperation |
| GraspXL [59 ] | Multiple | RaiSim | 500k | 10M | Random | ✗ | Motion | RL |
| BODex [7 ] | Shadow | MuJoCo | 2.4k | 3.62M | Fingertip | ✓ | Pre-grasp, grasp poses | Optimization |
| Dexonomy (Ours) | Shadow | MuJoCo | 10.7k | 9.5M | 31 types | ✓ | Pre-grasp, grasp, squeeze poses | Sampling+opt. |

**说明**: TABLE I: Dexterous Grasp Dataset Comparison. Our large-scale dataset aims to support the study of data-driven methods for type-aware grasp synthesis.

#### Table 2: TABLE II: Comparison with Type-Unaware Grasp Synthesis Baselines for Allegro Hand. Most baselines, except DexGraspNet,

| | GSR (% \%) ↑ \uparrow | OSR (% \%) ↑ \uparrow | S (s - 1 s^{-1}) ↑ \uparrow | CLN ↑ \uparrow | CDC (m  m mm) ↓ \downarrow | PD (m  m mm) ↓ \downarrow | SPD (m  m mm) ↓ \downarrow | D (% \%) ↓ \downarrow |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DexGraspNet [47 ] | 12.10 | 57.01 | 3.25 | 3.22 | 7.58 | 4.85 | 1.20 | 29.03 |
| FRoGGeR [25 ] | 10.34 | 55.70 | 2.98 | 2.51 | 4.95 | 0.22 | 0.00 | 27.01 |
| SpringGrasp [8 ] | 7.83 | 35.44 | 5.47 | 2.79 | 23.59 | 16.58 | 1.06 | 70.18 |
| BODex [7 ] | 49.23 | 96.56 | 403.9 | 3.85 | 3.03 | 0.63 | 0.02 | 32.50 |
| Ours | 60.50 | 96.53 | 323.4 | 4.38 | 0.21 | 0.00 | 0.00 | 34.17 |

**说明**: TABLE II: Comparison with Type-Unaware Grasp Synthesis Baselines for Allegro Hand. Most baselines, except DexGraspNet, only synthesize fingertip grasps, so we also synthesize fingertip grasps for a fair comparison.

#### Table 3: TABLE III: A Harder Benchmark for Fingertip Grasp Synthesis. This benchmark uses smaller friction coefficients and more

| Method | Attempt | DGN object [47 ] | Objaverse [18 ] | | |
| --- | --- | --- | --- | --- | --- |
| Number | GSR ↑ \uparrow | OSR ↑ \uparrow | GSR ↑ \uparrow | OSR ↑ \uparrow | |
| BODex | 20 | 14.79 | 71.30 | 6.92 | 43.48 |
| BODex | 100 | 14.80 | 89.84 | 6.91 | 73.53 |
| Ours | 20 | 27.16 | 91.28 | 18.25 | 84.17 |
| Ours | 100 | 27.18 | 95.13 | 18.34 | 94.63 |

**说明**: TABLE III: A Harder Benchmark for Fingertip Grasp Synthesis. This benchmark uses smaller friction coefficients and more diverse objects, and our method consistently outperforms the baseline. DGN indicates DexGraspNet.

#### Table 4: TABLE IV: Statistics of Grasp Synthesis for the GRASP Taxonomy. The success rate is lower than fingertip grasps because

| | GSR(% \%) ↑ \uparrow | OSR(% \%) ↑ \uparrow | CLN ↑ \uparrow | D(% \%) ↓ \downarrow | | |
| --- | --- | --- | --- | --- | --- | --- |
| | Normal | Hard | Normal | Hard | | |
| Power | 24.2 | 12.8 | 81.9 | 68.3 | 9.1 | 24.7 |
| Intermediate | 23.0 | 6.6 | 79.9 | 69.4 | 4.8 | 27.6 |
| Precision | 36.0 | 11.4 | 95.9 | 85.6 | 4.2 | 25.8 |

**说明**: TABLE IV: Statistics of Grasp Synthesis for the GRASP Taxonomy. The success rate is lower than fingertip grasps because many flexible grasp types are suitable only for specific objects, e.g., Lateral (#  16 \#16) grasps for flat objects.

#### Table 5: TABLE V: Ablation Study on Pipeline Modules for Fingertip Grasp Synthesis of Allegro Hand.

| | | GSR ↑ \uparrow | OSR ↑ \uparrow | CDC ↓ \downarrow | PD ↓ \downarrow |
| --- | --- | --- | --- | --- | --- |
| Global stage | w/o opt. | 45.7 | 88.4 | 0.22 | 0.00 |
| w/o filter | 18.6 | 80.3 | 0.34 | 0.00 | |
| Local stage | w/o opt. | 17.8 | 72.6 | 10.24 | 4.95 |
| w/o filter | 62.8 | 96.7 | 0.82 | 0.34 | |
| Template library | w/o update | 41.5 | 88.8 | 0.24 | 0.00 |
| Ours | 60.5 | 96.5 | 0.21 | 0.00 | |

**说明**: TABLE V: Ablation Study on Pipeline Modules for Fingertip Grasp Synthesis of Allegro Hand.

#### Table 6: TABLE VI: Ablation on Control Strategy for Simulation Validation. Power grasps are robust likely due to rich contacts.

| GSR (% \%) ↑ \uparrow | Shadow | Allegro | | |
| --- | --- | --- | --- | --- |
| Power | Intermediate | Precision | Fingertip | |
| Grasp only | 17.64 | 9.81 | 13.79 | 24.60 |
| w/ pre-grasp | 25.85 | 19.93 | 34.56 | 45.08 |
| Ours | 24.23 | 23.03 | 36.00 | 60.50 |

**说明**: TABLE VI: Ablation on Control Strategy for Simulation Validation. Power grasps are robust likely due to rich contacts.

#### Table 7: TABLE VII: Robustness to Initial Templates. Our strategy, adding new templates from successful grasps, is the key to r

| Grasp Success Rate(% \%) ↑ \uparrow | 6.Prismatic 4 Finger | 1.Large Diameter | | | | |
| --- | --- | --- | --- | --- | --- | --- |
| Template ID | 1 | 2 | 3 | 1 | 2 | 3 |
| w/o new templates | 45.6 | 12.0 | 1.7 | 22.9 | 22.5 | 7.2 |
| w/ new templates (Ours) | 57.1 | 56.7 | 46.2 | 30.1 | 34.7 | 26.8 |

**说明**: TABLE VII: Robustness to Initial Templates. Our strategy, adding new templates from successful grasps, is the key to robustness.

#### Table 8: TABLE VIII: Learning-based Grasp Synthesis from Single-View Object Point Clouds in Simulation. Our type-conditional mod

| Method | Dataset | GSR ↑ \uparrow | OSR ↑ \uparrow | CDC ↓ \downarrow | PD ↓ \downarrow | D ↓ \downarrow |
| --- | --- | --- | --- | --- | --- | --- |
| Type-uncond. | DGN [47 ] | 8.32 | 44.3 | 20.5 | 15.9 | 29.1 |
| BODex [7 ] | 54.0 | 84.4 | 11.7 | 6.2 | 32.0 | |
| Ours-type1 | 55.5 | 85.9 | 10.8 | 8.4 | 31.5 | |
| Ours-all | 24.5 | 73.2 | 15.6 | 11.6 | 28.0 | |
| Type-cond. | Ours-all | 63.9 | 91.3 | 13.9 | 8.6 | 25.7 |

**说明**: TABLE VIII: Learning-based Grasp Synthesis from Single-View Object Point Clouds in Simulation. Our type-conditional model trained on our Dexonomy dataset significantly outperforms baselines.

#### Table 9: TABLE IX: Different Learning Architectures Trained on Grasp Type 1. Flow denotes for mobius normalizing flow. NaN mean

| | Diffusion | Diffusion + MLP | Flow | Flow + MLP |
| --- | --- | --- | --- | --- |
| GSR (% \%) ↑ \uparrow | 16.0 | 41.8 | NaN | 55.5 |

**说明**: TABLE IX: Different Learning Architectures Trained on Grasp Type 1. Flow denotes for mobius normalizing flow. NaN means that the training fails with NaN gradients.

#### Table 10: TABLE X: Type-unconditional Model Trained on High-Quality Grasp Types. The selected grasp types exhibit high success r

| Dataset | GSR ↑ \uparrow | OSR ↑ \uparrow | CDC ↓ \downarrow | PD ↓ \downarrow | D ↓ \downarrow |
| --- | --- | --- | --- | --- | --- |
| Ours-type1 | 55.5 | 85.9 | 10.8 | 8.4 | 31.5 |
| Ours-type6 | 48.8 | 83.8 | 11.4 | 7.0 | 34.1 |
| Ours-type9 | 48.0 | 79.8 | 15.5 | 6.6 | 34.8 |
| Ours-type18 | 61.2 | 87.0 | 13.3 | 8.6 | 34.0 |
| Ours-type22 | 52.0 | 79.1 | 12.6 | 9.0 | 35.4 |
| Ours-type26 | 51.3 | 82.0 | 14.0 | 10.3 | 33.5 |
| Ours-type31 | 47.8 | 81.8 | 14.9 | 6.9 | 32.9 |
| Ours-type33 | 46.3 | 77.7 | 15.7 | 6.7 | 33.8 |

**说明**: TABLE X: Type-unconditional Model Trained on High-Quality Grasp Types. The selected grasp types exhibit high success rates during the synthesis of the Dexonomy dataset.

#### Table 11: TABLE XI: Type-Conditional Model with Different Testing Methods. The performances of Top 1 oracle and Classifier B outp

| | Average | Top 1 (oracle) | Classifier A | Classifier B |
| --- | --- | --- | --- | --- |
| GSR (% \%) | 28.9 | 78.2 | 46.1 | 63.9 |

**说明**: TABLE XI: Type-Conditional Model with Different Testing Methods. The performances of Top 1 oracle and Classifier B outperform type-unconditional models in Table X, highlighting the potential of studying different types for grasping.
## 实验解读

- 评价重点:围绕 接触推理、接触丰富操作、dexterous-grasping、灵巧操作、grasping,优先看真实机器人结果、扰动恢复、接触稳定性、控制误差、成功率、实时性和 sim-to-real 差距。
- System0 视角:实验结论不只看最终成功率,还要看闭环过程中是否能处理状态估计误差、执行延迟、碰撞/接触突变和外部扰动。
- 与上层系统的关系:如果方法能在快闭环内稳定工作,它可以作为 System1 技能或 System2 规划的底层执行保障;如果只在离线或慢速设定下成立,则还不能直接作为身体级快闭环模块。
## 批判性思考

### 优点

- 论文把 接触推理、接触丰富操作、dexterous-grasping、灵巧操作、grasping 放到机器人身体级闭环里讨论,和只做高层语义推理的工作相比,更接近真实机器人部署时会遇到的状态、动力学和安全约束。
- 若论文同时给出公式、系统结构图、硬件实验或消融结果,这些证据可以帮助判断方法是否真的能支撑毫秒到百毫秒级反应。

### 局限

- 需要重点核查控制频率、推理延迟、传感延迟、失败恢复、硬件磨损、未见场景泛化和安全边界;这些因素决定它能否成为可靠的 System0 模块。
- 对学习型方法,还要区分训练环境中的鲁棒性和真实部署中的鲁棒性,尤其是接触丰富任务、动态运动任务和人形机器人任务。
## 论文定位

- 研究对象:Dexonomy: Synthesizing All Dexterous Grasp Types in a Grasp Taxonomy。
- 关键词:接触推理、接触丰富操作、dexterous-grasping、灵巧操作、grasping、robot-generalization、机器人操作、scalable-robot-learning。
- System0 关联:关注机器人身体级快反应链路中“状态/接触/局部感知 -> 约束/优化/策略 -> 动作”的短闭环,不把问题停留在语言规划或任务分解层。

## 速查卡片

> [!summary] Dexonomy
> - **论文**: https://www.roboticsproceedings.org/rss21/p105.pdf
> - **arXiv**: http://arxiv.org/abs/2504.18829v2
> - **arXiv HTML**: https://arxiv.org/html/2504.18829v2
