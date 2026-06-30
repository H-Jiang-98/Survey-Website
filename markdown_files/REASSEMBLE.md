---
title: "Demonstrating REASSEMBLE: A Multimodal Dataset for Contact-rich Robotic Assembly and Disassembly"
method_name: "Demonstrating REASSEMBLE"
authors: ["Daniel Sliwowski"]
year: 2025
venue: "RSS"
tags: ["contact-reasoning", "robot-manipulation", "robust-control", "imitation-learning", "contact-rich-manipulation", "robot-generalization", "recovery"]
image_source: "online"
arxiv_html: "https://arxiv.org/html/2502.05086"
---
# Demonstrating REASSEMBLE
## 一句话总结

> Demonstrating REASSEMBLE: A Multimodal Dataset for Contact-rich Robotic Assembly and Disassembly 主要落在 [[assembly]]、[[benchmark-dataset]]、[[接触推理]]、[[接触丰富操作]] 这一类问题上;从 System0 视角看,它的价值在于把身体状态、接触变化、局部感知或动力学约束接到机器人低层闭环里,让动作生成更稳定、安全、实时。
## 核心贡献

1. 围绕 **Demonstrating REASSEMBLE: A Multimodal Dataset for Contact-rich Robotic Assembly and Disassembly** 建立了一个与 assembly、benchmark-dataset、接触推理、接触丰富操作、模仿学习、recovery 相关的机器人问题设定,重点不是高层语义规划,而是身体级闭环中的状态变化、接触变化或控制约束。
2. 方法层面的关键价值在于把感知、动力学、约束优化、策略学习或硬件反馈组织成可执行的低层动作生成机制,适合放进 System0 的快反应模块中理解。
3. 对系统集成而言,这类工作补上了 System1 技能选择与电机级控制之间的一层:它负责把上层目标变成动态可行、可恢复、可安全过滤的动作。
4. 若论文包含真实机器人、接触任务或仿真到真实迁移,核心应看它是否报告了闭环稳定性、扰动恢复、硬件延迟、失败模式和跨场景泛化。
## 问题背景

这篇论文可以放在 System0 的“具身快系统”背景下阅读:机器人需要在很短时间内根据本体状态、接触、力/触觉、局部视觉或动力学预测做出动作,而不是等待慢速规划器重新推理整个任务。assembly、benchmark-dataset、接触推理、接触丰富操作、模仿学习、recovery、机器人操作 相关问题的难点通常不在“知道要做什么”,而在动作执行过程中状态会快速偏离、接触会突然改变、约束会实时收紧,低层系统必须及时修正。

### System0 关联

精读时优先看三件事:第一,论文使用哪些快速反馈量,例如关节状态、接触状态、力觉、触觉、深度或局部视觉;第二,反馈如何进入 MPC、WBC、策略网络、安全过滤器或恢复控制器;第三,输出是否直接影响机器人下一步动作,而不是只服务离线分析或高层任务规划。
## 方法详解

### System0 视角解析

从 System0 角度看,这篇论文的方法部分应当围绕 assembly、benchmark-dataset、接触推理、接触丰富操作、模仿学习、recovery、机器人操作、torque-control 来读。核心问题是:系统如何从快速变化的机器人状态中构造控制输入,如何处理接触、碰撞、约束、模型误差或执行延迟,以及如何把这些信息变成下一时刻的动作。

如果方法里有优化问题,重点看目标函数、约束和求解频率;如果方法里有学习策略,重点看观测空间、动作空间、训练扰动和部署时的闭环频率;如果方法里有硬件或传感器,重点看传感链路是否真的进入控制回路。这样读可以把论文和 System2 语言规划、System1 技能编排区分开来。

### 关键公式与机制

#### 公式 1: [[动力学或策略机制]]

$$\boldsymbol{\tau}_{\mathrm{n}}=\left(\mathbf{I}-\mathbf{J}^{\top}(\boldsymbol{\theta})\left[\mathbf{J}(\boldsymbol{\theta})\mathbf{J}^{\top}(\boldsymbol{\theta})\right]^{-1}\mathbf{J}(\boldsymbol{\theta})\right)\frac{\partial V(\boldsymbol{\theta})}{\partial\boldsymbol{\theta}},$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 2: [[动力学或策略机制]]

$$V(\boldsymbol{\theta})=\alpha\det\left(\mathbf{J}(\boldsymbol{\theta})\mathbf{J}^{\top}(\boldsymbol{\theta})\right)+\beta\mathrm{J_{\mathrm{L}}}(\boldsymbol{\theta})+\gamma\dot{\boldsymbol{\theta}},$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 3: [[动力学或策略机制]]

$$\mathrm{J_{\mathrm{L}}}(\boldsymbol{\theta})=\tanh\left(a(\boldsymbol{\theta}-\mathbf{c})+b\right)+\tanh\left(-a(\boldsymbol{\theta}-\mathbf{c})+b\right)+2,$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 4: [[动力学或策略机制]]

$$\mathbf{e}_{\dot{\xi}}=\dot{\boldsymbol{\xi}}_{\mathrm{r,cmd}}-\dot{\boldsymbol{\xi}}_{\mathrm{r}}\in\mathbb{R}^{6}$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 5: [[动力学或策略机制]]

$$\tau_{\mathrm{imp}}=\mathbf{J}^{\top}(\mathbf{K}_{x}\mathbf{e}_{\xi}+\mathbf{D}\mathbf{e}_{\dot{\xi}}),$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 6: [[动力学或策略机制]]

$$\mathbf{f}(\mathbf{x})=\frac{\sum_{i=1}^{N}\mathbf{w}_{i}\psi_{i}(\mathbf{x})}{\sum_{i=1}^{N}\psi_{i}(\mathbf{x})}\mathbf{x},$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 7: [[动力学或策略机制]]

$$=\alpha_{z}\left(\beta_{z}(\mathbf{g}-\mathbf{y})-\mathbf{z}\right)+\hat{\mathbf{f}}(\mathbf{x}),$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 8: [[动力学或策略机制]]

$$\mathbf{q}_{1}\otimes\mathbf{q}_{2}:=\begin{bmatrix}w_{1}+w_{2}-\mathbf{v}_{1}^{T}\mathbf{v}_{2}\\ w_{2}\mathbf{v}_{2}+w_{2}\mathbf{v}_{1}+\mathbf{v}_{1}\times\mathbf{v}_{2}\end{bmatrix}.$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 9: [[动力学或策略机制]]

$$\mathbf{f}_{m}=-\mathbf{B}_{m}\dot{\mathbf{x}}_{m}+0.35\mathbf{f}_{r}^{\mathrm{r0}},$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 10: [[动力学或策略机制]]

$$\mathbf{f}_{r}^{\mathrm{m0}}=\mathbf{R}_{\mathrm{m0}}^{\mathrm{r0}^{-1}}\mathbf{R}_{\mathrm{r,Sensor}}^{\mathrm{r0}}\mathbf{f}_{r},$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。
## 关键图表

### Figure 1: Overview of the sensor placement. We use two external and one wrist-mounted RGB camer

![Figure 1](https://arxiv.org/html/2502.05086/x1.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Overview of the sensor placement. We use two external and one wrist-mounted RGB camer”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 2: Overview of the teleoperation control system. The operator controls the robot’s motio

![Figure 2](https://arxiv.org/html/2502.05086/x2.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Overview of the teleoperation control system. The operator controls the robot’s motio”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 3: Large Gear assembly & disassembly The figure illustrates the trajectories generated b

![Figure 3](https://arxiv.org/html/2502.05086/extracted/6394582/images/MLP_ecperiments_new.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Large Gear assembly & disassembly The figure illustrates the trajectories generated b”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### 论文表格

#### Table 1: TABLE I: Datasets comparison. We compare several commonly used datasets based on the number of demonstrations, the numb

| Dataset | # Demos | # Verbs | Sensors | Robot | Collection method | Tasks |
| --- | --- | --- | --- | --- | --- | --- |
| BC-Z [14 ] | 263k | 8 | 1 RGB Camera, Robot Proprioception | Everyday Robots | VR teleoperation | MPL |
| RT-1 [4 ] | 130k | 8 | 1 RGB Camera, Robot Proprioception | Everyday Robots | VR teleoperation | MPL |
| Language Table | 594k | n/r | 1 RGB Camera, Robot Proprioception | XArm | VR teleoperation | MPL |
| BridgeDatav2 [15 ] | 60.1k | 14 | 4 RGB Cameras, 1 Depth Camera, Robot Proprioception | WidowX | VR teleoperation | MPL |
| RoboSet | 98.5k | 14 | 4 RGBD Cameras, Robot Proprioception | Franka Emika Panda Reaserach 3 | Kinesthetic, VR teleoperation, Autonomous | MPL |
| FurnitureBench [16 ] | 5.1k | 9 | 2 RGB Cameras, Robot Proprioception | Franka | VR teleoperation | MPL |
| DROID [17 ] | 76k | 86 | 3 RGBD CAmeras, Robot Proprioception | Franka Emika Panda Reaserach 3 | VR teleoperation | MPL |
| RH20T [18 ] | 110k | 42 | 8-10 RGBD Cameras, 1 Microphone, F/T Sensor | Multiple | haptic teleoperation | MPL |
| JIGSAWS [19 ] | 103 | 11 | 1 RGB Camera, Robot Proprioception | DaVinci | Haptic Teleoperation | TAS |
| 50SALADS [20 ] | 50 | 6 | 1 RGBD Camera, accelerometers | N/A | Human demonstration | TAS |
| Assembly101 [21 ] | 4.3k | 24 | 8 RGB Cameras, 4 Mono Cameras | N/A | Human demonstration | TAS |
| FALIURE [22 ] | 229 | 6 | 1 RGBD Camera, 1 Microphone | Baxter | Scripted | AD |
| (Im)PerfectPour [23 ] | 554 | 4 | 2 RGB Cameras | Franka Emika Panda Reaserach 3 | VR teleoperation | AD |
| REASSEMBLE | 4k | 4 actions and 9 skills | 3 RGB cameras, Robot Proprioception, 1 DAVIS Event camera, 3 Microphones, F/T Sensor | Franka Emika Panda Reaserach 3 | Haptic teleoperation | TAS, MPL, AD, TIL |

**说明**: TABLE I: Datasets comparison. We compare several commonly used datasets based on the number of demonstrations, the number of verbs they contain, the sensors used during data collection, the robotic platform, the data collection method, and the tasks that can be learned from the dataset. The tasks include TAS (Temporal Action Segmentation), MPL (Motion Policy Learning), AD (Anomaly Detection), and TIL (Task Inversion Learning). We use "n/r" to denote information that is not reported by the works and "N/A" for cases where the category is not applicable.
## 实验解读

- 评价重点:围绕 assembly、benchmark-dataset、接触推理、接触丰富操作、模仿学习,优先看真实机器人结果、扰动恢复、接触稳定性、控制误差、成功率、实时性和 sim-to-real 差距。
- System0 视角:实验结论不只看最终成功率,还要看闭环过程中是否能处理状态估计误差、执行延迟、碰撞/接触突变和外部扰动。
- 与上层系统的关系:如果方法能在快闭环内稳定工作,它可以作为 System1 技能或 System2 规划的底层执行保障;如果只在离线或慢速设定下成立,则还不能直接作为身体级快闭环模块。
## 批判性思考

### 优点

- 论文把 assembly、benchmark-dataset、接触推理、接触丰富操作、模仿学习 放到机器人身体级闭环里讨论,和只做高层语义推理的工作相比,更接近真实机器人部署时会遇到的状态、动力学和安全约束。
- 若论文同时给出公式、系统结构图、硬件实验或消融结果,这些证据可以帮助判断方法是否真的能支撑毫秒到百毫秒级反应。

### 局限

- 需要重点核查控制频率、推理延迟、传感延迟、失败恢复、硬件磨损、未见场景泛化和安全边界;这些因素决定它能否成为可靠的 System0 模块。
- 对学习型方法,还要区分训练环境中的鲁棒性和真实部署中的鲁棒性,尤其是接触丰富任务、动态运动任务和人形机器人任务。
## 论文定位

- 研究对象:Demonstrating REASSEMBLE: A Multimodal Dataset for Contact-rich Robotic Assembly and Disassembly。
- 关键词:assembly、benchmark-dataset、接触推理、接触丰富操作、模仿学习、recovery、机器人操作、torque-control。
- System0 关联:关注机器人身体级快反应链路中“状态/接触/局部感知 -> 约束/优化/策略 -> 动作”的短闭环,不把问题停留在语言规划或任务分解层。

## 速查卡片

> [!summary] Demonstrating REASSEMBLE
> - **论文**: https://www.roboticsproceedings.org/rss21/p059.pdf
> - **arXiv**: http://arxiv.org/abs/2502.05086
> - **arXiv HTML**: https://arxiv.org/html/2502.05086
