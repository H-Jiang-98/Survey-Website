---
title: "From Reaction to Anticipation: Proactive Failure Recovery through Agentic Task Graph for Robotic Manipulation"
method_name: "From Reaction to Anticipation"
authors: ["Sheng Xu"]
year: 2026
venue: "RSS"
tags: ["robot-manipulation", "robust-control", "real-time-control", "closed-loop-control", "recovery"]
image_source: "online"
arxiv_html: "https://arxiv.org/html/2605.11951v1"
---
# From Reaction to Anticipation
## 一句话总结

> From Reaction to Anticipation: Proactive Failure Recovery through Agentic Task Graph for Robotic Manipulation 主要落在 [[bimanual-manipulation]]、[[closed-loop-control]]、[[reactive-control]]、[[实时控制]] 这一类问题上;从 System0 视角看,它的价值在于把身体状态、接触变化、局部感知或动力学约束接到机器人低层闭环里,让动作生成更稳定、安全、实时。
## 核心贡献

1. 围绕 **From Reaction to Anticipation: Proactive Failure Recovery through Agentic Task Graph for Robotic Manipulation** 建立了一个与 bimanual-manipulation、closed-loop-control、reactive-control、实时控制、recovery、机器人操作 相关的机器人问题设定,重点不是高层语义规划,而是身体级闭环中的状态变化、接触变化或控制约束。
2. 方法层面的关键价值在于把感知、动力学、约束优化、策略学习或硬件反馈组织成可执行的低层动作生成机制,适合放进 System0 的快反应模块中理解。
3. 对系统集成而言,这类工作补上了 System1 技能选择与电机级控制之间的一层:它负责把上层目标变成动态可行、可恢复、可安全过滤的动作。
4. 若论文包含真实机器人、接触任务或仿真到真实迁移,核心应看它是否报告了闭环稳定性、扰动恢复、硬件延迟、失败模式和跨场景泛化。
## 问题背景

这篇论文可以放在 System0 的“具身快系统”背景下阅读:机器人需要在很短时间内根据本体状态、接触、力/触觉、局部视觉或动力学预测做出动作,而不是等待慢速规划器重新推理整个任务。bimanual-manipulation、closed-loop-control、reactive-control、实时控制、recovery、机器人操作、鲁棒控制 相关问题的难点通常不在“知道要做什么”,而在动作执行过程中状态会快速偏离、接触会突然改变、约束会实时收紧,低层系统必须及时修正。

### System0 关联

精读时优先看三件事:第一,论文使用哪些快速反馈量,例如关节状态、接触状态、力觉、触觉、深度或局部视觉;第二,反馈如何进入 MPC、WBC、策略网络、安全过滤器或恢复控制器;第三,输出是否直接影响机器人下一步动作,而不是只服务离线分析或高层任务规划。
## 方法详解

### System0 视角解析

从 System0 角度看,这篇论文的方法部分应当围绕 bimanual-manipulation、closed-loop-control、reactive-control、实时控制、recovery、机器人操作、鲁棒控制 来读。核心问题是:系统如何从快速变化的机器人状态中构造控制输入,如何处理接触、碰撞、约束、模型误差或执行延迟,以及如何把这些信息变成下一时刻的动作。

如果方法里有优化问题,重点看目标函数、约束和求解频率;如果方法里有学习策略,重点看观测空间、动作空间、训练扰动和部署时的闭环频率;如果方法里有硬件或传感器,重点看传感链路是否真的进入控制回路。这样读可以把论文和 System2 语言规划、System1 技能编排区分开来。

### 关键公式与机制

#### 公式 1: [[优化目标/约束]]

$$\mathcal{F}^{i\rightarrow j}=\{f^{i\rightarrow j}_{1},f^{i\rightarrow j}_{2},\dots\}.$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 2: [[优化目标/约束]]

$$V_{\text{rec}}=\left\{v_{m}^{\text{rec}(i,j)}\;\middle|\;f_{m}^{i\rightarrow j}\in\mathcal{F}^{i\rightarrow j}\right\}.$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 3: [[优化目标/约束]]

$$\mathcal{M}^{i\rightarrow j}_{m}(t)=\mathbb{I}\!\left[f^{i\rightarrow j}_{m}(z_{t^{\prime}})>\epsilon,\ \forall t^{\prime}\in\{t\!\!-\!\!K\!\!+\!\!1,\dots,t\}\right],$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 4: [[动力学或策略机制]]

$$\rho\!\left(\varepsilon^{i\rightarrow j},f^{i\rightarrow j}_{m}\right)=\varepsilon^{i\rightarrow\text{rec}(i,j,m)},$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 5: [[优化目标/约束]]

$$\mathcal{G}=(V,E)\leftarrow\text{Structure}(\mathcal{I},o_{0};\mathcal{F}_{\text{mllm}})$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 6: [[优化目标/约束]]

$$\mathcal{F}^{i\rightarrow j}=\{f^{i\rightarrow j}_{1},f^{i\rightarrow j}_{2},\dots\}$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 7: [[优化目标/约束]]

$$f_{\text{tilt}}^{k}(z_{t})=\arccos\!\left(\left| {u^{k}_{t}}^{\top} $\hat{\mathbf{g}}\right$ |\right)-\theta_{\max}\leq 0.$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 8: [[优化目标/约束]]

$$=\arg\min_{\mathbf{e}_{t:t+H}}\ \lambda_{\text{path}}^{i\rightarrow j}(\mathbf{e}_{t:t+H};\hat{x}_{t})$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 9: [[优化目标/约束]]

$$=\arg\min_{e}\ \lambda_{\text{sub}}^{i}(e;\hat{x}^{i})$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 10: [[优化目标/约束]]

$$\cup\left\{\varepsilon^{\text{rec}(i,j,m)\rightarrow k}\;\middle|\;f_{m}^{i\rightarrow j}\in\mathcal{F}^{i\rightarrow j},\ v^{k}\in K_{m}^{i\rightarrow j}\right\}.$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。
## 关键图表

### Figure 1: The framework of AgentChord with an example. It involves three agents: the Task Struct

![Figure 1](https://arxiv.org/html/2605.11951v1/x1.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“The framework of AgentChord with an example. It involves three agents: the Task Struct”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 2: Illustration of the failure recovery process in AgentChord for six real-world tasks

![Figure 2](https://arxiv.org/html/2605.11951v1/x3.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Illustration of the failure recovery process in AgentChord for six real-world tasks”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 3: Overview of the hardware system, consisting of an AgileX CobotMagic dual-arm robot and

![Figure 3](https://arxiv.org/html/2605.11951v1/figures/devices.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Overview of the hardware system, consisting of an AgileX CobotMagic dual-arm robot and”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### 论文表格

#### Table 1: TABLE I: Evaluation results on three simulation tasks with different degrees of disturbances. The bolded values indicate

| Tasks with disturbance | Success Rate (%) ↑ \uparrow | Execution Time (s) ↓ \downarrow | Episode Steps ↓ \downarrow | | | | | | | | | | | |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| IM | DRM | ReKep | CaM | AgentChord | DRM | ReKep | CaM | AgentChord | DRM | ReKep | CaM | AgentChord | | |
| Single-arm pour water with drop p p | p p =0.05 | 85 | 95 | 100 | 100 | 100 | 96.7 | 38.2 | 72.4 | 33.1 | 356 | 362 | 338 | 340 |
| p p =0.10 | 75 | 90 | 95 | 100 | 100 | 126.0 | 46.1 | 98.1 | 38.8 | 389 | 380 | 361 | 355 | |
| Dual-arm pour water with drop p p | p p =0.05 | 75 | 90 | 85 | 95 | 100 | 103.8 | 69.9 | 83.9 | 50.2 | 418 | 438 | 409 | 392 |
| p p =0.10 | 70 | 80 | 75 | 90 | 95 | 145.3 | 93.1 | 110.0 | 63.4 | 512 | 531 | 492 | 483 | |
| Rearrange table with drop p p | p p =0.05 | 90 | 100 | 100 | 100 | 100 | 64.2 | 34.2 | 43.1 | 29.6 | 267 | 298 | 257 | 255 |
| p p =0.10 | 80 | 100 | 100 | 100 | 100 | 99.2 | 45.0 | 61.2 | 33.9 | 296 | 337 | 280 | 285 | |
| Average | 79.2 | 92.5 | 92.5 | 97.5 | 99.2 | 105.9 | 54.4 | 78.1 | 41.5 | 373 | 391 | 356 | 352 | |

**说明**: TABLE I: Evaluation results on three simulation tasks with different degrees of disturbances. The bolded values indicate the best results (highest success rate, lowest execution time, and lowest episode steps) for each setting. In cases of tied primary metrics, bolding is determined by the better secondary metrics.

#### Table 2: TABLE II: Evaluation results on six real-world tasks with disturbances. Bold values indicate the best results. In cases

| Tasks with disturbance | Success Rate (%) ↑ \uparrow | Execution Time (s) ↓ \downarrow | | | | | | | |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| IM | DRM | ReKep | CaM | AgentChord | DRM | ReKep | CaM | AgentChord | |
| Single-arm pour water | 70 | 80 | 85 | 90 | 95 | 130.5 | 87.0 | 119.5 | 75.9 |
| Dual-arm pour water | 60 | 65 | 60 | 70 | 80 | 185.0 | 130.8 | 167.2 | 110.7 |
| Rearrange table | 80 | 85 | 95 | 90 | 90 | 113.4 | 85.4 | 99.3 | 71.0 |
| Handover block | 60 | 70 | 60 | 75 | 85 | 178.6 | 149.5 | 170.2 | 130.1 |
| Fold towel | 40 | 50 | 50 | 50 | 55 | 117.5 | 83.7 | 108.1 | 78.4 |
| Setup coffee tray | 45 | 50 | 40 | 60 | 60 | 135.8 | 105.9 | 121.3 | 87.3 |
| Average | 59.2 | 66.7 | 65.0 | 72.5 | 77.5 | 143.5 | 107.1 | 130.9 | 92.2 |

**说明**: TABLE II: Evaluation results on six real-world tasks with disturbances. Bold values indicate the best results. In cases of tied primary metrics, bolding is determined by the better secondary metrics.

#### Table 3: TABLE III: Success rates of different policies on the simulated single-arm pour water task. Bold values indicate the bes

| Method | Success Rate |
| --- | --- |
| RDT [36 ] | 16/50 |
| $\pi_{0} [5 ]$ | 20/50 |
| GR00T N1.5 [4 ] | 15/50 |
| Sim2Real-VLA [52 ] | 26/50 |
| Sim2Real-VLA (rec) | 39/50 |

**说明**: TABLE III: Success rates of different policies on the simulated single-arm pour water task. Bold values indicate the best result.

#### Table 4: TABLE IV: Atomic actions used to compose task-graph edges. Each action is parameterized by robot_name and (when applicab

| Atomic action | Brief description |
| --- | --- |
| drive | Dual-arm wrapper executing left_arm_action and right_arm_action (either can be None). |
| grasp | Approach and grasp a target object with a pre-grasp offset (e.g., pre_grasp_dis); supports optional grasp offsets for alignment. |
| open_gripper | Open the gripper to release; optional sample_num controls actuation steps. |
| close_gripper | Close the gripper to secure a hold; optional sample_num controls actuation steps. |
| move_to_obj | Move to a pose defined by offsets relative to a referenced object (e.g., x_offset, y_offset, z_offset); supports orientation hints and mask-conditioned targeting. |
| move_to_target | Move to an absolute target pose (extrinsic frame). |
| move_by_offset | Apply a relative Cartesian displacement in intrinsic/extrinsic frame (e.g., dx, dy, dz) or angles. |
| rotate_eef | Rotate the end-effector by a specified angle; used for tilt-to-pour and restoring upright orientation. |
| place | Place a grasped object at a target table location (x,y) with optional z_offset, then release. |
| back | Return the arm to a predefined home configuration. |

**说明**: TABLE IV: Atomic actions used to compose task-graph edges. Each action is parameterized by robot_name and (when applicable) obj_name.

#### Table 5: TABLE V: Ablation study on the recovery-augmented graph. We use AC to denote AgentChord and AC-BT to denote its backtrac

| Task | Success Rate (%) ↑ \uparrow | Execution Time (s) ↓ \downarrow | | | | |
| --- | --- | --- | --- | --- | --- | --- |
| ReKep | AC-BT | AC | ReKep | AC-BT | AC | |
| Single-arm pour water | 85 | 85 | 95 | 87.0 | 92.3 | 75.9 |
| Dual-arm pour water | 60 | 65 | 80 | 130.8 | 139.5 | 110.7 |
| Handover block | 60 | 70 | 85 | 149.5 | 153.4 | 130.1 |
| Average | 68.3 | 73.3 | 86.7 | 122.4 | 128.4 | 105.6 |

**说明**: TABLE V: Ablation study on the recovery-augmented graph. We use AC to denote AgentChord and AC-BT to denote its backtracking variant without pre-compiled recovery edges.
## 实验解读

- 评价重点:围绕 bimanual-manipulation、closed-loop-control、reactive-control、实时控制、recovery,优先看真实机器人结果、扰动恢复、接触稳定性、控制误差、成功率、实时性和 sim-to-real 差距。
- System0 视角:实验结论不只看最终成功率,还要看闭环过程中是否能处理状态估计误差、执行延迟、碰撞/接触突变和外部扰动。
- 与上层系统的关系:如果方法能在快闭环内稳定工作,它可以作为 System1 技能或 System2 规划的底层执行保障;如果只在离线或慢速设定下成立,则还不能直接作为身体级快闭环模块。
## 批判性思考

### 优点

- 论文把 bimanual-manipulation、closed-loop-control、reactive-control、实时控制、recovery 放到机器人身体级闭环里讨论,和只做高层语义推理的工作相比,更接近真实机器人部署时会遇到的状态、动力学和安全约束。
- 若论文同时给出公式、系统结构图、硬件实验或消融结果,这些证据可以帮助判断方法是否真的能支撑毫秒到百毫秒级反应。

### 局限

- 需要重点核查控制频率、推理延迟、传感延迟、失败恢复、硬件磨损、未见场景泛化和安全边界;这些因素决定它能否成为可靠的 System0 模块。
- 对学习型方法,还要区分训练环境中的鲁棒性和真实部署中的鲁棒性,尤其是接触丰富任务、动态运动任务和人形机器人任务。
## 论文定位

- 研究对象:From Reaction to Anticipation: Proactive Failure Recovery through Agentic Task Graph for Robotic Manipulation。
- 关键词:bimanual-manipulation、closed-loop-control、reactive-control、实时控制、recovery、机器人操作、鲁棒控制。
- System0 关联:关注机器人身体级快反应链路中“状态/接触/局部感知 -> 约束/优化/策略 -> 动作”的短闭环,不把问题停留在语言规划或任务分解层。

## 速查卡片

> [!summary] From Reaction to Anticipation
> - **论文**: https://www.roboticsproceedings.org/rss22/p180.pdf
> - **arXiv**: http://arxiv.org/abs/2605.11951v1
> - **arXiv HTML**: https://arxiv.org/html/2605.11951v1
