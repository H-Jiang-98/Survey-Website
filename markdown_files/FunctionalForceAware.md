---
title: "Functional Force-Aware Retargeting from Virtual Human Demos to Soft Robot Policies"
method_name: "Functional Force Aware"
authors: ["Uksang Yoo"]
year: 2026
venue: "RSS"
tags: ["contact-reasoning", "robot-manipulation", "robust-control", "real-time-control", "imitation-learning", "contact-rich-manipulation", "robot-generalization", "dexterous-manipulation", "whole-body-control", "state-estimation", "recovery"]
image_source: "online"
arxiv_html: "https://arxiv.org/html/2604.01224v1"
---
# Functional Force Aware
## 一句话总结

> Functional Force-Aware Retargeting from Virtual Human Demos to Soft Robot Policies 主要落在 [[compliance-control]]、[[接触推理]]、[[接触丰富操作]]、[[灵巧操作]] 这一类问题上;从 System0 视角看,它的价值在于把身体状态、接触变化、局部感知或动力学约束接到机器人低层闭环里,让动作生成更稳定、安全、实时。
## 核心贡献

1. 围绕 **Functional Force-Aware Retargeting from Virtual Human Demos to Soft Robot Policies** 建立了一个与 compliance-control、接触推理、接触丰富操作、灵巧操作、力控制、模仿学习 相关的机器人问题设定,重点不是高层语义规划,而是身体级闭环中的状态变化、接触变化或控制约束。
2. 方法层面的关键价值在于把感知、动力学、约束优化、策略学习或硬件反馈组织成可执行的低层动作生成机制,适合放进 System0 的快反应模块中理解。
3. 对系统集成而言,这类工作补上了 System1 技能选择与电机级控制之间的一层:它负责把上层目标变成动态可行、可恢复、可安全过滤的动作。
4. 若论文包含真实机器人、接触任务或仿真到真实迁移,核心应看它是否报告了闭环稳定性、扰动恢复、硬件延迟、失败模式和跨场景泛化。
## 问题背景

这篇论文可以放在 System0 的“具身快系统”背景下阅读:机器人需要在很短时间内根据本体状态、接触、力/触觉、局部视觉或动力学预测做出动作,而不是等待慢速规划器重新推理整个任务。compliance-control、接触推理、接触丰富操作、灵巧操作、力控制、模仿学习、运动模仿 相关问题的难点通常不在“知道要做什么”,而在动作执行过程中状态会快速偏离、接触会突然改变、约束会实时收紧,低层系统必须及时修正。

### System0 关联

精读时优先看三件事:第一,论文使用哪些快速反馈量,例如关节状态、接触状态、力觉、触觉、深度或局部视觉;第二,反馈如何进入 MPC、WBC、策略网络、安全过滤器或恢复控制器;第三,输出是否直接影响机器人下一步动作,而不是只服务离线分析或高层任务规划。
## 方法详解

### System0 视角解析

从 System0 角度看,这篇论文的方法部分应当围绕 compliance-control、接触推理、接触丰富操作、灵巧操作、力控制、模仿学习、运动模仿、motion-tracking 来读。核心问题是:系统如何从快速变化的机器人状态中构造控制输入,如何处理接触、碰撞、约束、模型误差或执行延迟,以及如何把这些信息变成下一时刻的动作。

如果方法里有优化问题,重点看目标函数、约束和求解频率;如果方法里有学习策略,重点看观测空间、动作空间、训练扰动和部署时的闭环频率;如果方法里有硬件或传感器,重点看传感链路是否真的进入控制回路。这样读可以把论文和 System2 语言规划、System1 技能编排区分开来。

### 关键公式与机制

#### 公式 1: [[优化目标/约束]]

$$d_{\mathrm{geo}}(x,y)=\min_{P\in\mathcal{P}(\hat{v}(x),\hat{v}(y))}\sum_{(i,j)\in P}\lVert v_{i}-v_{j} \rVert_{2},$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 2: [[优化目标/约束]]

$$h_{v}^{t}=\sum_{(c,f)\in\mathcal{C}_{t}}\lVert f \rVert_{2}\exp\!\left(-\lambda\,d_{\mathrm{geo}}(v,c)\right).$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 3: [[优化目标/约束]]

$$\min_{\{n_{r}\}}\;\max_{r}\frac{F_{r}}{n_{r}}\quad\text{s.t.}\quad\sum_{r}n_{r}=N_{f},$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 4: [[优化目标/约束]]

$$\delta_{i}^{t}=\frac{\sum_{j}w_{ij}^{t}(c_{j}^{t}-s_{i}^{t})}{\sum_{j}w_{ij}^{t}+\epsilon},\qquad\lVert \delta_{i}^{t} \rVert_{2}\leq\delta_{\max}.$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 5: [[优化目标/约束]]

$$\mathbf{p}^{\mathrm{ref}}=\arg\min_{\mathbf{p}\in[p_{\min},p_{\max}]^{3}}\left\lVert \Pi_{xy}\!\left(f_{\mathrm{MLP}}(\mathbf{p})\right)-\Pi_{xy}\!\left(\mathbf{d}^{\mathrm{ref}}\right)\right \rVert_{2}^{2},$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 6: [[优化目标/约束]]

$$\mathbf{p}(\mathbf{u})=p_{\min}+(p_{\max}-p_{\min})\,\sigma(\mathbf{u}),$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 7: [[优化目标/约束]]

$$\hat{v}(x)=\arg\min_{v_{i}\in\mathcal{V}}\lVert x-v_{i} \rVert_{2}.$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 8: [[优化目标/约束]]

$$h_{v}^{t}=\sum_{(c,f)\in\mathcal{C}_{t}}\lVert f \rVert_{2}\exp\!\left(-\lambda\,d_{\mathrm{geo}}(v,c)\right),$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 9: [[优化目标/约束]]

$$C_{ir}=\lVert \mu_{i}-\mu_{r} \rVert_{2}+\beta\,\mathbb{I}\!\left[\mathcal{W}_{i}\cap\mathcal{W}_{r}=\varnothing\right].$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 10: [[动力学或策略机制]]

$$d_{ij}^{t}=d_{\mathrm{geo}}(s_{i}^{t},c_{j}^{t}),\quad w_{ij}^{t}=\lVert f_{j}^{t} \rVert_{2}\exp\!\left(-\lambda\,d_{ij}^{t}\right).$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。
## 关键图表

### Figure 1: Pipeline / core system figure: Design of the pneumatic soft manipulator. The manipulator features a rigid–soft hybrid

![Figure 1](https://arxiv.org/html/2604.01224v1/x1.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Pipeline / core system figure: Design of the pneumatic soft manipulator. The manipulator features a rigid–soft hybrid”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 2: Retargeting Stages. Overview of the two-stage force-aware retargeting pipeline. Stage

![Figure 2](https://arxiv.org/html/2604.01224v1/x5.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Retargeting Stages. Overview of the two-stage force-aware retargeting pipeline. Stage”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 3: Real-World Manipulation Results. Sequential frames showing zero-shot real-world execut

![Figure 3](https://arxiv.org/html/2604.01224v1/figures/realworld.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Real-World Manipulation Results. Sequential frames showing zero-shot real-world execut”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### 论文表格

#### Table 1: TABLE I: Fingertip trajectory tracking performance for all controllers. Lower is better.

| Controller | RMSE | Mean ± \pm Std | Max |
| --- | --- | --- | --- |
| | (mm) | (mm) | (mm) |
| Direct KNN | 8.74 | 5.33 ± \pm 6.93 | 34.34 |
| Direct MLP | 6.11 | 4.78 ± \pm 3.80 | 13.85 |
| Direct Linear | 5.05 | 3.97 ± \pm 3.13 | 10.89 |
| SoftAct | 2.28 | 2.06 ± \pm 0.98 | 5.47 |

**说明**: TABLE I: Fingertip trajectory tracking performance for all controllers. Lower is better.

#### Table 2: TABLE II: Object trajectory tracking error in simulation experiments. Lower is better.

| Task | Method | Pos. (cm) | Rot. (deg) |
| --- | --- | --- | --- |
| Light Bulb Insertion | Kinematic | 1.52 ± 0.61 1.52\pm 0.61 | 11.8 ± 3.4 11.8\pm 3.4 |
| SoftAct -stage 2 | 0.97 ± 0.42 0.97\pm 0.42 | 25.1 ± 7.9 25.1\pm 7.9 | |
| SoftAct | 0.48 ± 0.21  $\mathbf{0.48\pm 0.21}$ | 12.4 ± 8.1  $\mathbf{12.4\pm 8.1}$ | |
| Light Bulb Twisting | Kinematic | 0.05 ± 0.07 0.05\pm 0.07 | 18.2 ± 6.5 18.2\pm 6.5 |
| SoftAct -stage 2 | 0.07 ± 0.03 0.07\pm 0.03 | 9.4 ± 3.4 9.4\pm 3.4 | |
| SoftAct | 0.02 ± 0.005  $\mathbf{0.02\pm 0.005}$ | 2.9 ± 1.3  $\mathbf{2.9\pm 1.3}$ | |
| Cup Pouring | Kinematic | 1.29 ± 0.50 1.29\pm 0.50 | 12.5 ± 4.1 12.5\pm 4.1 |
| SoftAct -stage 2 | 0.81 ± 0.35 0.81\pm 0.35 | 6.4 ± 2.2 6.4\pm 2.2 | |
| SoftAct | 0.51 ± 0.21  $\mathbf{0.51\pm 0.21}$ | 3.5 ± 1.3  $\mathbf{3.5\pm 1.3}$ | |
| Marker Grasping | Kinematic | 0.97 ± 0.41 0.97\pm 0.41 | 14.3 ± 5.0 14.3\pm 5.0 |
| SoftAct -stage 2 | 0.56 ± 0.25 0.56\pm 0.25 | 7.3 ± 2.5 7.3\pm 2.5 | |
| SoftAct | 0.31 ± 0.13  $\mathbf{0.31\pm 0.13}$ | 3.9 ± 1.5  $\mathbf{3.9\pm 1.5}$ | |
| Bottle Unscrewing | Kinematic | 0.26 ± 0.11 0.26\pm 0.11 | 21.3 ± 7.2 21.3\pm 7.2 |
| SoftAct -stage 2 | 0.11 ± 0.05 0.11\pm 0.05 | 11.1 ± 3.8 11.1\pm 3.8 | |
| SoftAct | 0.07 ± 0.12  $\mathbf{0.07\pm 0.12}$ | 6.7 ± 2.6  $\mathbf{6.7\pm 2.6}$ | |
| Box Reorienting | Kinematic | 1.34 ± 0.54 1.34\pm 0.54 | 17.6 ± 6.0 17.6\pm 6.0 |
| SoftAct -stage 2 | 0.89 ± 0.37 0.89\pm 0.37 | 9.0 ± 3.1 9.0\pm 3.1 | |
| SoftAct | 0.62 ± 0.26  $\mathbf{0.62\pm 0.26}$ | 5.1 ± 2.0  $\mathbf{5.1\pm 2.0}$ | |

**说明**: TABLE II: Object trajectory tracking error in simulation experiments. Lower is better.

#### Table 3: TABLE III: Task success rates in simulation and the real world for SoftAct compared to a kinematic-only baseline. For ea

| Task | Domain | SoftAct | Kinematic Baseline |
| --- | --- | --- | --- |
| Paper Cup Pouring | Simulation | 90% | 40% |
| Light Bulb Insertion | Simulation | 47% | 0% |
| Marker Grasping | Simulation | 97% | 73% |
| Bottle Unscrewing | Simulation | 80% | 47% |
| Box Reorienting | Simulation | 90% | 30% |
| Light Bulb Screwing | Simulation | 100% | 77% |
| Paper Cup Pouring | Real | 85% | 35% |
| Light Bulb Screwing | Real | 95% | 30% |
| Box Reorienting | Real | 70% | 10% |

**说明**: TABLE III: Task success rates in simulation and the real world for SoftAct compared to a kinematic-only baseline. For each baseline per task, we do 30 rollouts in simulation, and 20 in the real world. Higher is better.

#### Table 4: TABLE IV: Simulation parameters (representative values).

| Parameter | Value |
| --- | --- |
| Young’s modulus | 127 kPa |
| Poisson’s ratio | 0.48 |
| Material density | 1000 kg/m 3 |
| Spine capsule length | 10 mm |
| Number of spine segments | 8 |
| Joint stiffness | 0.2 N  \cdot m/rad |
| Joint damping | 0.02 N  \cdot m  \cdot s/rad |
| Soft–rigid spring stiffness | 200 N/m |
| Simulation timestep | 1 ms |
| Contact friction coefficient | 0.6 |

**说明**: TABLE IV: Simulation parameters (representative values).

#### Table 5: TABLE V: Diffusion policy training hyperparameters.

| Parameter | Value |
| --- | --- |
| Horizon | 16 |
| Observation steps | 2 |
| Action steps | 8 |
| Diffusion steps | 100 |
| Batch size | 256 |
| Learning rate | 1  10 - 4 1\times 10^{-4} |
| EMA decay | 0.999 |
| Training epochs | 3000 |

**说明**: TABLE V: Diffusion policy training hyperparameters.
## 实验解读

- 评价重点:围绕 compliance-control、接触推理、接触丰富操作、灵巧操作、力控制,优先看真实机器人结果、扰动恢复、接触稳定性、控制误差、成功率、实时性和 sim-to-real 差距。
- System0 视角:实验结论不只看最终成功率,还要看闭环过程中是否能处理状态估计误差、执行延迟、碰撞/接触突变和外部扰动。
- 与上层系统的关系:如果方法能在快闭环内稳定工作,它可以作为 System1 技能或 System2 规划的底层执行保障;如果只在离线或慢速设定下成立,则还不能直接作为身体级快闭环模块。
## 批判性思考

### 优点

- 论文把 compliance-control、接触推理、接触丰富操作、灵巧操作、力控制 放到机器人身体级闭环里讨论,和只做高层语义推理的工作相比,更接近真实机器人部署时会遇到的状态、动力学和安全约束。
- 若论文同时给出公式、系统结构图、硬件实验或消融结果,这些证据可以帮助判断方法是否真的能支撑毫秒到百毫秒级反应。

### 局限

- 需要重点核查控制频率、推理延迟、传感延迟、失败恢复、硬件磨损、未见场景泛化和安全边界;这些因素决定它能否成为可靠的 System0 模块。
- 对学习型方法,还要区分训练环境中的鲁棒性和真实部署中的鲁棒性,尤其是接触丰富任务、动态运动任务和人形机器人任务。
## 论文定位

- 研究对象:Functional Force-Aware Retargeting from Virtual Human Demos to Soft Robot Policies。
- 关键词:compliance-control、接触推理、接触丰富操作、灵巧操作、力控制、模仿学习、运动模仿、motion-tracking、实时控制、recovery。
- System0 关联:关注机器人身体级快反应链路中“状态/接触/局部感知 -> 约束/优化/策略 -> 动作”的短闭环,不把问题停留在语言规划或任务分解层。

## 速查卡片

> [!summary] Functional Force Aware
> - **论文**: https://www.roboticsproceedings.org/rss22/p202.pdf
> - **arXiv**: http://arxiv.org/abs/2604.01224v1
> - **arXiv HTML**: https://arxiv.org/html/2604.01224v1
