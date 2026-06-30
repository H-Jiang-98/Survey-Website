---
title: "HoMMI: Learning Whole-Body Mobile Manipulation from Human Demonstrations"
method_name: "HoMMI"
authors: ["Xiaomeng Xu"]
year: 2026
venue: "RSS"
tags: ["robot-manipulation", "reinforcement-learning", "imitation-learning", "robot-generalization", "whole-body-control", "navigation"]
image_source: "online"
arxiv_html: "https://arxiv.org/html/2603.03243v2"
---
# HoMMI
## 一句话总结

> HoMMI: Learning Whole-Body Mobile Manipulation from Human Demonstrations 主要落在 [[bimanual-manipulation]]、[[cross-embodiment]]、[[egocentric-perception]]、[[模仿学习]] 这一类问题上;从 System0 视角看,它的价值在于把身体状态、接触变化、局部感知或动力学约束接到机器人低层闭环里,让动作生成更稳定、安全、实时。
## 核心贡献

1. 围绕 **HoMMI: Learning Whole-Body Mobile Manipulation from Human Demonstrations** 建立了一个与 bimanual-manipulation、cross-embodiment、egocentric-perception、模仿学习、mobile-manipulation、navigation 相关的机器人问题设定,重点不是高层语义规划,而是身体级闭环中的状态变化、接触变化或控制约束。
2. 方法层面的关键价值在于把感知、动力学、约束优化、策略学习或硬件反馈组织成可执行的低层动作生成机制,适合放进 System0 的快反应模块中理解。
3. 对系统集成而言,这类工作补上了 System1 技能选择与电机级控制之间的一层:它负责把上层目标变成动态可行、可恢复、可安全过滤的动作。
4. 若论文包含真实机器人、接触任务或仿真到真实迁移,核心应看它是否报告了闭环稳定性、扰动恢复、硬件延迟、失败模式和跨场景泛化。
## 问题背景

这篇论文可以放在 System0 的“具身快系统”背景下阅读:机器人需要在很短时间内根据本体状态、接触、力/触觉、局部视觉或动力学预测做出动作,而不是等待慢速规划器重新推理整个任务。bimanual-manipulation、cross-embodiment、egocentric-perception、模仿学习、mobile-manipulation、navigation、policy-learning 相关问题的难点通常不在“知道要做什么”,而在动作执行过程中状态会快速偏离、接触会突然改变、约束会实时收紧,低层系统必须及时修正。

### System0 关联

精读时优先看三件事:第一,论文使用哪些快速反馈量,例如关节状态、接触状态、力觉、触觉、深度或局部视觉;第二,反馈如何进入 MPC、WBC、策略网络、安全过滤器或恢复控制器;第三,输出是否直接影响机器人下一步动作,而不是只服务离线分析或高层任务规划。
## 方法详解

### System0 视角解析

从 System0 角度看,这篇论文的方法部分应当围绕 bimanual-manipulation、cross-embodiment、egocentric-perception、模仿学习、mobile-manipulation、navigation、policy-learning、retargeting 来读。核心问题是:系统如何从快速变化的机器人状态中构造控制输入,如何处理接触、碰撞、约束、模型误差或执行延迟,以及如何把这些信息变成下一时刻的动作。

如果方法里有优化问题,重点看目标函数、约束和求解频率;如果方法里有学习策略,重点看观测空间、动作空间、训练扰动和部署时的闭环频率;如果方法里有硬件或传感器,重点看传感链路是否真的进入控制回路。这样读可以把论文和 System2 语言规划、System1 技能编排区分开来。

### 关键公式与机制

#### 公式 1: [[动力学或策略机制]]

$$R_{t}^{\text{cur}}=\begin{bmatrix}x_{t}&y_{t}&z_{t}\end{bmatrix}\in\mathbb{R}^{3\times 3}$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 2: [[动力学或策略机制]]

$$x_{t}^{\prime}=x_{t}-(x_{t}^{\top}\hat{d}_{t})\hat{d}_{t},\hat{x}_{t}=\frac{x_{t}^{\prime}}{\lVert x_{t}^{\prime} \rVert},$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 3: [[动力学或策略机制]]

$$R_{t}=\begin{bmatrix}\hat{x}_{t}&\hat{y}_{t}&\hat{d}_{t}\end{bmatrix}.$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 4: [[动力学或策略机制]]

$$f(\Delta q)=C_{\text{ee}}(\Delta q)+C_{\text{nominal}}(\Delta q)+C_{\text{current}}(\Delta q)+C_{\text{com}}(\Delta q)$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 5: [[动力学或策略机制]]

$$\mathbf{p}_{\text{interp}}(t)=(1-\alpha)\mathbf{p}_{\text{prev}}+\alpha\mathbf{p}_{\text{cmd}},$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 6: [[动力学或策略机制]]

$$C_{\text{ee}}=\sum_{i\in\{\mathrm{L},\mathrm{R}\}}\left\lVert \mathbf{J}_{i}\Delta q-\mathbf{v}_{i}\right \rVert_{\mathbf{W}_{\text{ee}}}^{2},$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 7: [[动力学或策略机制]]

$$C_{\text{nominal}}=\left\lVert(q+\Delta q)-q_{\text{nom}}\right \rVert_{\mathbf{W}_{\text{nom}}}^{2},$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 8: [[动力学或策略机制]]

$$C_{\text{current}}=\left\lVert \Delta q\right \rVert_{\mathbf{W}_{\text{curr}}}^{2},$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 9: [[动力学或策略机制]]

$$=\left\lVert p^{xy}_{\text{torso}}(q+\Delta q)-p^{xy}_{\text{base}}(q+\Delta q)-r^{xy}_{\star}\right \rVert_{\mathbf{W}_{\text{com}}}^{2}$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 10: [[动力学或策略机制]]

$$| p^{x}_{ $\text{torso}}-p^{x}_{\text{base}}-r^{x}_{\star}$ |\leq b_{x},\quad| p^{y}_{ $\text{torso}}-p^{y}_{\text{base}}-r^{y}_{\star}$ |\leq b_{y},$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。
## 关键图表

### Figure 1: HoMMI System Overview. We learn whole-body mobile manipulation from human demonstrati

![Figure 1](https://arxiv.org/html/2603.03243v2/x2.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“HoMMI System Overview. We learn whole-body mobile manipulation from human demonstrati”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 2: HoMMI Whole-Body Controller is designed to achieve precise end-effector tracking for

![Figure 2](https://arxiv.org/html/2603.03243v2/x5.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“HoMMI Whole-Body Controller is designed to achieve precise end-effector tracking for”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 3: Hardware Schematic

![Figure 3](https://arxiv.org/html/2603.03243v2/fig/hardware_schematic.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Hardware Schematic”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### 论文表格

#### Table 1: TABLE I: Whole-body IK parameters for the three evaluation tasks

| Symbol | Laundry | Delivery | Tablescape |
| --- | --- | --- | --- |
| w p w_{p} | 10000 | 10000 | 10000 |
| w o w_{o} | 10000 | 10000 | 10000 |
| w nom,torso w_{ $\text{nom,torso}}$ | 50 | 1000 | 200 |
| w nom,arm w_{ $\text{nom,arm}}$ | 50 | 1000 | 10 |
| w curr w_{ $\text{curr}}$ | 50 | 1000 | 10 |
| w base,pos w_{ $\text{base,pos}}$ | 50 | 50 | 5000 |
| w base,ori w_{ $\text{base,ori}}$ | 50 | 50 | 5000 |
| w com w_{ $\text{com}}$ | 100000 | 100000 | 100000 |
| b x, b y b_{x},b_{y} | 0.08 m 0.08 $\text{\,}\mathrm{m}$ | 0.08 m 0.08 $\text{\,}\mathrm{m}$ | 0.08 m 0.08 $\text{\,}\mathrm{m}$ |

**说明**: TABLE I: Whole-body IK parameters for the three evaluation tasks

#### Table 2: TABLE II: Depth quality sensitivity.

| noise std (mm) | 0 | 2 | 10 | 20 |
| --- | --- | --- | --- | --- |
| success (%) | 90 | 90 | 90 | 50 |

**说明**: TABLE II: Depth quality sensitivity.

#### Table 3: TABLE III: Generalization to unseen objects.

| s een and u nseen object | s1 | s2 | u1 | u2 | u3 | u4 | u5 | u6 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| success (%) | 100 | 75 | 100 | 75 | 100 | 100 | 100 | 75 |

**说明**: TABLE III: Generalization to unseen objects.

#### Table 4: TABLE IV: Robustness to lighting variations.

| light (lux) | 4300 | 3370 | 1840 | 810 |
| --- | --- | --- | --- | --- |
| success (%) | 100 | 100 | 100 | 75 |

**说明**: TABLE IV: Robustness to lighting variations.
## 实验解读

- 评价重点:围绕 bimanual-manipulation、cross-embodiment、egocentric-perception、模仿学习、mobile-manipulation,优先看真实机器人结果、扰动恢复、接触稳定性、控制误差、成功率、实时性和 sim-to-real 差距。
- System0 视角:实验结论不只看最终成功率,还要看闭环过程中是否能处理状态估计误差、执行延迟、碰撞/接触突变和外部扰动。
- 与上层系统的关系:如果方法能在快闭环内稳定工作,它可以作为 System1 技能或 System2 规划的底层执行保障;如果只在离线或慢速设定下成立,则还不能直接作为身体级快闭环模块。
## 批判性思考

### 优点

- 论文把 bimanual-manipulation、cross-embodiment、egocentric-perception、模仿学习、mobile-manipulation 放到机器人身体级闭环里讨论,和只做高层语义推理的工作相比,更接近真实机器人部署时会遇到的状态、动力学和安全约束。
- 若论文同时给出公式、系统结构图、硬件实验或消融结果,这些证据可以帮助判断方法是否真的能支撑毫秒到百毫秒级反应。

### 局限

- 需要重点核查控制频率、推理延迟、传感延迟、失败恢复、硬件磨损、未见场景泛化和安全边界;这些因素决定它能否成为可靠的 System0 模块。
- 对学习型方法,还要区分训练环境中的鲁棒性和真实部署中的鲁棒性,尤其是接触丰富任务、动态运动任务和人形机器人任务。
## 论文定位

- 研究对象:HoMMI: Learning Whole-Body Mobile Manipulation from Human Demonstrations。
- 关键词:bimanual-manipulation、cross-embodiment、egocentric-perception、模仿学习、mobile-manipulation、navigation、policy-learning、retargeting、机器人操作、scalable-robot-learning。
- System0 关联:关注机器人身体级快反应链路中“状态/接触/局部感知 -> 约束/优化/策略 -> 动作”的短闭环,不把问题停留在语言规划或任务分解层。

## 速查卡片

> [!summary] HoMMI
> - **论文**: https://www.roboticsproceedings.org/rss22/p205.pdf
> - **arXiv**: http://arxiv.org/abs/2603.03243v2
> - **arXiv HTML**: https://arxiv.org/html/2603.03243v2
