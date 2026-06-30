---
title: "HiWET: Hierarchical World-Frame End-Effector Tracking for Long-Horizon Humanoid Loco-Manipulation"
method_name: "HiWET"
authors: ["Zhanxiang Cao"]
year: 2026
venue: "RSS"
tags: ["contact-reasoning", "robot-manipulation", "legged-locomotion", "reinforcement-learning", "robot-generalization", "humanoid", "sim-to-real", "loco-manipulation", "whole-body-control"]
image_source: "online"
arxiv_html: "https://arxiv.org/html/2602.06341v1"
---
# HiWET
## 一句话总结

> HiWET: Hierarchical World-Frame End-Effector Tracking for Long-Horizon Humanoid Loco-Manipulation 主要落在 [[接触推理]]、[[人形机器人]]、[[足式运动]]、[[移动操作]] 这一类问题上;从 System0 视角看,它的价值在于把身体状态、接触变化、局部感知或动力学约束接到机器人低层闭环里,让动作生成更稳定、安全、实时。
## 核心贡献

1. 围绕 **HiWET: Hierarchical World-Frame End-Effector Tracking for Long-Horizon Humanoid Loco-Manipulation** 建立了一个与 接触推理、人形机器人、足式运动、移动操作、motion-tracking、强化学习 相关的机器人问题设定,重点不是高层语义规划,而是身体级闭环中的状态变化、接触变化或控制约束。
2. 方法层面的关键价值在于把感知、动力学、约束优化、策略学习或硬件反馈组织成可执行的低层动作生成机制,适合放进 System0 的快反应模块中理解。
3. 对系统集成而言,这类工作补上了 System1 技能选择与电机级控制之间的一层:它负责把上层目标变成动态可行、可恢复、可安全过滤的动作。
4. 若论文包含真实机器人、接触任务或仿真到真实迁移,核心应看它是否报告了闭环稳定性、扰动恢复、硬件延迟、失败模式和跨场景泛化。
## 问题背景

这篇论文可以放在 System0 的“具身快系统”背景下阅读:机器人需要在很短时间内根据本体状态、接触、力/触觉、局部视觉或动力学预测做出动作,而不是等待慢速规划器重新推理整个任务。接触推理、人形机器人、足式运动、移动操作、motion-tracking、强化学习、机器人操作 相关问题的难点通常不在“知道要做什么”,而在动作执行过程中状态会快速偏离、接触会突然改变、约束会实时收紧,低层系统必须及时修正。

### System0 关联

精读时优先看三件事:第一,论文使用哪些快速反馈量,例如关节状态、接触状态、力觉、触觉、深度或局部视觉;第二,反馈如何进入 MPC、WBC、策略网络、安全过滤器或恢复控制器;第三,输出是否直接影响机器人下一步动作,而不是只服务离线分析或高层任务规划。
## 方法详解

### System0 视角解析

从 System0 角度看,这篇论文的方法部分应当围绕 接触推理、人形机器人、足式运动、移动操作、motion-tracking、强化学习、机器人操作、scalable-robot-learning 来读。核心问题是:系统如何从快速变化的机器人状态中构造控制输入,如何处理接触、碰撞、约束、模型误差或执行延迟,以及如何把这些信息变成下一时刻的动作。

如果方法里有优化问题,重点看目标函数、约束和求解频率;如果方法里有学习策略,重点看观测空间、动作空间、训练扰动和部署时的闭环频率;如果方法里有硬件或传感器,重点看传感链路是否真的进入控制回路。这样读可以把论文和 System2 语言规划、System1 技能编排区分开来。

### 关键公式与机制

#### 公式 1: [[动力学或策略机制]]

$$s_{t}=\big[\boldsymbol{\omega}_{t},\;\mathbf{g}_{t},\;\mathbf{q}_{t},\;\dot{\mathbf{q}}_{t},\;\mathbf{a}_{t-1}^{L}\big],$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 2: [[动力学或策略机制]]

$$\mathbf{u}_{t}=\big[\mathbf{v}_{b}^{des},\;h^{des},\;{}^{b}\mathbf{T}_{L}^{des},\;{}^{b}\mathbf{T}_{R}^{des},\;\alpha_{t}\big],$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 3: [[动力学或策略机制]]

$$a_{t}^{L}=\mathbf{q}_{t}^{des}=\big[\mathbf{q}_{t,up}^{des},\;\mathbf{q}_{t,low}^{des}\big]\sim\pi^{L}(\mathbf{q}_{t}^{des}\mid s_{t},\mathbf{u}_{t}),$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 4: [[优化目标/约束]]

$$\mathcal{H}_{t}=\{(s_{t-i},\mathbf{u}_{t-i})\}_{i=0}^{H-1}$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 5: [[动力学或策略机制]]

$$\mathbf{o}_{t}^{actor}=\big[s_{t},\;\mathbf{u}_{t},\;\hat{\mathbf{p}}_{t},\;\mathbf{e}_{t},\;\hat{\mathbf{q}}_{t,up}\big].$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 6: [[动力学或策略机制]]

$$\mathbf{o}_{t}^{critic}=\big[s_{t},\;\mathbf{u}_{t},\;\mathbf{p}_{t},\;\mathbf{h}_{t},\;\hat{\mathbf{q}}_{t,up}\big],$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 7: [[动力学或策略机制]]

$$r_{kmp,t}=\exp\left(-\frac{1}{N_{up}\sigma_{kmp}^{2}}\lVert \mathbf{q}_{t,up}-\hat{\mathbf{q}}_{t,up} \rVert_{2}^{2}\right),$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 8: [[动力学或策略机制]]

$$\mathbf{o}_{t}^{H,\text{critic}}=\big[s_{t}^{H},\;\mathbf{v}_{b}^{w,t},\;{}^{w}\mathbf{T}_{L}^{t},\;{}^{w}\mathbf{T}_{R}^{t}\big],$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 9: [[动力学或策略机制]]

$$a_{t}^{H}=\mathbf{u}_{t}=\Big[\mathbf{v}_{b}^{des},\;h^{des},\;{}^{b}\mathbf{T}_{L}^{des},\;{}^{b}\mathbf{T}_{R}^{des},\;\alpha_{t}\Big]\sim\pi^{H}(\mathbf{u}_{t}\mid s_{t}^{H}).$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 10: [[动力学或策略机制]]

$$e_{h,t}=\begin{cases}(h_{t}-h_{t}^{des})\cdot\bar{w}_{knee}^{flex}&\text{if}h_{t}>h_{t}^{des},\\ (h_{t}-h_{t}^{des})\cdot\bar{w}_{knee}^{ext}&\text{if}h_{t}<h_{t}^{des},\end{cases}$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。
## 关键图表

### Figure 1: Pipeline / core system figure: HiWET capabilities in simulation and real-world deployment. (a)-(c) Whole-body redunda

![Figure 1](https://arxiv.org/html/2602.06341v1/x1.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Pipeline / core system figure: HiWET capabilities in simulation and real-world deployment. (a)-(c) Whole-body redunda”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 2: HiWET architecture and two-stage training procedure. Stage 1: Tracker (blue): The trac

![Figure 2](https://arxiv.org/html/2602.06341v1/x2.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“HiWET architecture and two-stage training procedure. Stage 1: Tracker (blue): The trac”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 3: Snapshots of real-world deployment on the Unitree G1 robot. The low-level policy maint

![Figure 3](https://arxiv.org/html/2602.06341v1/x7.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Snapshots of real-world deployment on the Unitree G1 robot. The low-level policy maint”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### 论文表格

#### Table 1: TABLE I: Command Tracking Performance Comparison

| Method | Lin. Vel. Error | Ang. Vel. Error | Height Error | EE Pos. Error |
| --- | --- | --- | --- | --- |
| | (m/s) | (rad/s) | (m) | (mm) |
| HiWET | 0.157 ± \pm 0.003 | 0.461 ± \pm 0.006 | 0.018 ± \pm 0.012 | 12.4 ± \pm 2.4 |
| HiWET w/o IS | 0.165 ± \pm 0.005 | 0.472 ± \pm 0.006 | 0.018 ± \pm 0.014 | 16.1 ± \pm 5.3 |
| HiWET w/o State Est. | 0.169 ± \pm 0.003 | 0.459 ± \pm 0.004 | 0.018 ± \pm 0.016 | 23.0 ± \pm 7.2 |
| HiWET w/o KMP | 0.149 ± \pm 0.004 | 0.423 ± \pm 0.005 | 0.015 ± \pm 0.010 | 25.2 ± \pm 12.8 |
| HOMIE [2 ] | 0.194 ± \pm 0.003 | 0.451 ± \pm 0.006 | 0.022 ± \pm 0.019 | - |

**说明**: TABLE I: Command Tracking Performance Comparison

#### Table 2: TABLE II: Real-world end-effector tracking errors for circle and square trajectory tasks.

| Method | Circle RMSE (m) | Square RMSE (m) |
| --- | --- | --- |
| HiWET | 0.012 ± \pm 0.005 | 0.015 ± \pm 0.007 |
| HiWET w/ Fixed  \alpha | 0.018 ± \pm 0.008 | 0.019 ± \pm 0.009 |
| HiWET w/o State Est. | 0.024 ± \pm 0.011 | 0.028 ± \pm 0.011 |
| HiWET w/o KMP | 0.032 ± \pm 0.013 | 0.039 ± \pm 0.015 |

**说明**: TABLE II: Real-world end-effector tracking errors for circle and square trajectory tasks.
## 实验解读

- 评价重点:围绕 接触推理、人形机器人、足式运动、移动操作、motion-tracking,优先看真实机器人结果、扰动恢复、接触稳定性、控制误差、成功率、实时性和 sim-to-real 差距。
- System0 视角:实验结论不只看最终成功率,还要看闭环过程中是否能处理状态估计误差、执行延迟、碰撞/接触突变和外部扰动。
- 与上层系统的关系:如果方法能在快闭环内稳定工作,它可以作为 System1 技能或 System2 规划的底层执行保障;如果只在离线或慢速设定下成立,则还不能直接作为身体级快闭环模块。
## 批判性思考

### 优点

- 论文把 接触推理、人形机器人、足式运动、移动操作、motion-tracking 放到机器人身体级闭环里讨论,和只做高层语义推理的工作相比,更接近真实机器人部署时会遇到的状态、动力学和安全约束。
- 若论文同时给出公式、系统结构图、硬件实验或消融结果,这些证据可以帮助判断方法是否真的能支撑毫秒到百毫秒级反应。

### 局限

- 需要重点核查控制频率、推理延迟、传感延迟、失败恢复、硬件磨损、未见场景泛化和安全边界;这些因素决定它能否成为可靠的 System0 模块。
- 对学习型方法,还要区分训练环境中的鲁棒性和真实部署中的鲁棒性,尤其是接触丰富任务、动态运动任务和人形机器人任务。
## 论文定位

- 研究对象:HiWET: Hierarchical World-Frame End-Effector Tracking for Long-Horizon Humanoid Loco-Manipulation。
- 关键词:接触推理、人形机器人、足式运动、移动操作、motion-tracking、强化学习、机器人操作、scalable-robot-learning、仿真到真实迁移、zero-shot。
- System0 关联:关注机器人身体级快反应链路中“状态/接触/局部感知 -> 约束/优化/策略 -> 动作”的短闭环,不把问题停留在语言规划或任务分解层。

## 速查卡片

> [!summary] HiWET
> - **论文**: https://www.roboticsproceedings.org/rss22/p030.pdf
> - **arXiv**: http://arxiv.org/abs/2602.06341v1
> - **arXiv HTML**: https://arxiv.org/html/2602.06341v1
