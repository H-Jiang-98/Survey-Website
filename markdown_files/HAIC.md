---
title: "HAIC: Humanoid Agile Object Interaction Control via Dynamics-Aware World Model"
method_name: "HAIC"
authors: ["Dongting Li"]
year: 2026
venue: "RSS"
tags: ["contact-reasoning", "robot-manipulation", "robust-control", "legged-locomotion", "adaptive-control", "contact-rich-manipulation", "robot-generalization", "humanoid", "agile-locomotion", "collision-avoidance", "whole-body-control", "state-estimation"]
image_source: "online"
arxiv_html: "https://arxiv.org/html/2602.11758v2"
---
# HAIC
## 一句话总结

> HAIC: Humanoid Agile Object Interaction Control via Dynamics-Aware World Model 主要落在 [[adaptive-control]]、[[agile-locomotion]]、[[碰撞避免]]、[[接触推理]] 这一类问题上;从 System0 视角看,它的价值在于把身体状态、接触变化、局部感知或动力学约束接到机器人低层闭环里,让动作生成更稳定、安全、实时。
## 核心贡献

1. 围绕 **HAIC: Humanoid Agile Object Interaction Control via Dynamics-Aware World Model** 建立了一个与 adaptive-control、agile-locomotion、碰撞避免、接触推理、人形机器人、inertial-estimation 相关的机器人问题设定,重点不是高层语义规划,而是身体级闭环中的状态变化、接触变化或控制约束。
2. 方法层面的关键价值在于把感知、动力学、约束优化、策略学习或硬件反馈组织成可执行的低层动作生成机制,适合放进 System0 的快反应模块中理解。
3. 对系统集成而言,这类工作补上了 System1 技能选择与电机级控制之间的一层:它负责把上层目标变成动态可行、可恢复、可安全过滤的动作。
4. 若论文包含真实机器人、接触任务或仿真到真实迁移,核心应看它是否报告了闭环稳定性、扰动恢复、硬件延迟、失败模式和跨场景泛化。
## 问题背景

这篇论文可以放在 System0 的“具身快系统”背景下阅读:机器人需要在很短时间内根据本体状态、接触、力/触觉、局部视觉或动力学预测做出动作,而不是等待慢速规划器重新推理整个任务。adaptive-control、agile-locomotion、碰撞避免、接触推理、人形机器人、inertial-estimation、足式运动 相关问题的难点通常不在“知道要做什么”,而在动作执行过程中状态会快速偏离、接触会突然改变、约束会实时收紧,低层系统必须及时修正。

### System0 关联

精读时优先看三件事:第一,论文使用哪些快速反馈量,例如关节状态、接触状态、力觉、触觉、深度或局部视觉;第二,反馈如何进入 MPC、WBC、策略网络、安全过滤器或恢复控制器;第三,输出是否直接影响机器人下一步动作,而不是只服务离线分析或高层任务规划。
## 方法详解

### System0 视角解析

从 System0 角度看,这篇论文的方法部分应当围绕 adaptive-control、agile-locomotion、碰撞避免、接触推理、人形机器人、inertial-estimation、足式运动、non-prehensile-manipulation 来读。核心问题是:系统如何从快速变化的机器人状态中构造控制输入,如何处理接触、碰撞、约束、模型误差或执行延迟,以及如何把这些信息变成下一时刻的动作。

如果方法里有优化问题,重点看目标函数、约束和求解频率;如果方法里有学习策略,重点看观测空间、动作空间、训练扰动和部署时的闭环频率;如果方法里有硬件或传感器,重点看传感链路是否真的进入控制回路。这样读可以把论文和 System2 语言规划、System1 技能编排区分开来。

### 关键公式与机制

#### 公式 1: [[优化目标/约束]]

$$
\mathcal{L}_{WM}=\lambda_{\text{obj}}\lVert \hat{s}^{obj}_{t}-s^{obj}_{t} \rVert^{2}+\lambda_{\text{priv}}\lVert \hat{z}^{priv}_{t}-z^{priv}_{t} \rVert^{2}
$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 2: [[优化目标/约束]]

$$
\mathcal{L}_{Stage1}=\mathcal{L}_{PPO}(\pi_{T},V_{T})+\mathcal{L}_{WM}+\lambda_{\text{distill}}D_{KL}(\pi_{S}||\pi_{T})
$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 3: [[优化目标/约束]]

$$r_{\text{contact}}=\frac{1}{| $\mathcal{O}$ | } $\sum_{o\in\mathcal{O}}\left(\frac{1}{$ | $\mathcal{E}_{o}$ |}\sum_{e\in\mathcal{E}_{o}}\mathbb{I}_{o,e}\cdot r_{\text{pos}}^{o,e}\cdot r_{\text{force}}^{o,e}\right).$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 4: [[优化目标/约束]]

$$r_{\text{pos}}^{o,e}=\exp\left(-{\max(0,\lVert \boldsymbol{p}_{e}-\boldsymbol{p}_{\text{tgt}} \rVert-\epsilon_{\text{tol}})}/{\sigma_{p}}\right).$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 5: [[优化目标/约束]]

$$\frac{1}{| $\mathcal{O}$ | } $\sum_{o\in\mathcal{O}}\left(\frac{1}{$ | $\mathcal{E}_{o}$ |}\sum_{e\in\mathcal{E}_{o}}\mathbb{I}_{o,e}\cdot r_{\text{pos}}^{o,e}\cdot r_{\text{force}}^{o,e}\right)$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 6: [[动力学或策略机制]]

$$E_{\mathrm{mpbpe}}=\mathbb{E}\left[\big\lVert(\boldsymbol{p}_{t}-\boldsymbol{p}_{\mathrm{root},t})-(\boldsymbol{p}_{t}^{\mathrm{ref}}-\boldsymbol{p}_{\mathrm{root},t}^{\mathrm{ref}})\big \rVert_{2}\right].$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 7: [[动力学或策略机制]]

$$E_{\mathrm{mpjpe}}=\mathbb{E}\left[\big\lVert \boldsymbol{\theta}_{t}-\boldsymbol{\theta}_{t}^{\mathrm{ref}}\big \rVert_{1}\right].$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 8: [[动力学或策略机制]]

$$E_{\mathrm{mpjve}}=\mathbb{E}\left[\big\lVert \Delta\boldsymbol{\theta}_{t}-\Delta\boldsymbol{\theta}_{t}^{\mathrm{ref}}\big \rVert_{1}\right],$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 9: [[动力学或策略机制]]

$$E_{\mathrm{mpooe}}=\mathbb{E}\left[2\arccos\left(| $\langle\boldsymbol{q}_{t}^{\mathrm{obj}},\boldsymbol{q}_{t}^{\mathrm{obj,ref}}\rangle$ |\right)\right].$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 10: [[动力学或策略机制]]

$$E_{\mathrm{mpoae}}=\mathbb{E}\left[\big\lVert \Delta^{2}\boldsymbol{p}_{t}^{\mathrm{obj}}-\Delta^{2}\boldsymbol{p}_{t}^{\mathrm{obj,ref}}\big \rVert_{2}\right],$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。
## 关键图表

### Figure 1: Pipeline / core system figure: a) Types of interacted objects

![Figure 1](https://arxiv.org/html/2602.11758v2/x2.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Pipeline / core system figure: a) Types of interacted objects”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 2: Overview of our Dynamics-aware World Model. It predicts object dynamics from proprioc

![Figure 2](https://arxiv.org/html/2602.11758v2/x4.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Overview of our Dynamics-aware World Model. It predicts object dynamics from proprioc”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 3: Framework. We train policies in the simulation from scratch. The framework

![Figure 3](https://arxiv.org/html/2602.11758v2/x5.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Framework. We train policies in the simulation from scratch. The framework”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### 论文表格

#### Table 1: TABLE I: Real-world results for Skateboarding. We report the success rate for both Gliding and Completing. HAIC achiev

| | Success Rate | Robot State Metrics | | | | | | |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Method | Glide | Comp. | E mpbpe E_{ $\mathrm{mpbpe}}$ | E mpboe E_{ $\mathrm{mpboe}}$ | E mpjpe E_{ $\mathrm{mpjpe}}$ | E mpbve E_{ $\mathrm{mpbve}}$ | E mpbae E_{ $\mathrm{mpbae}}$ | E mpjve E_{ $\mathrm{mpjve}}$ |
| HDMI ∗ | 20% | 0% | 132.1 ±35.7 | 666.7 ±213.6 | 188.7 ±12.6 | 5.39 ±0.62 | 4.06 ±0.04 | 10.40 ±0.19 |
| HAIC | 100% | 60% | 81.5 ±15.4 | 453.5 ±88.2 | 127.1 ±11.5 | 4.72 ±0.75 | 4.48 ±0.50 | 12.15 ±1.87 |

**说明**: TABLE I: Real-world results for Skateboarding. We report the success rate for both Gliding and Completing. HAIC achieves stable gliding and significantly higher completion rates compared to the baseline.

#### Table 2: TABLE II: Real-world results for Cart Manipulation. HAIC achieves 100% success in both tasks, whereas the baseline fai

| Task | Method | SR ↑ \uparrow | E mpbpe E_{ $\mathrm{mpbpe}} ↓ \downarrow$ | E mpboe E_{ $\mathrm{mpboe}} ↓ \downarrow$ | E mpjpe E_{ $\mathrm{mpjpe}} ↓ \downarrow$ | E mpbve E_{ $\mathrm{mpbve}} ↓ \downarrow$ | E mpbae E_{ $\mathrm{mpbae}} ↓ \downarrow$ | E mpjve E_{ $\mathrm{mpjve}} ↓ \downarrow$ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Pull Cart | HDMI ∗ | 40% | 118.0 ±5.8 | 556.7 ±43.0 | 165.0 ±2.3 | 6.11 ±0.13 | 3.73 ±0.14 | 10.32 ±1.00 |
| HAIC | 100% | 76.2 ±6.8 | 425.0 ±41.7 | 132.3 ±6.4 | 5.08 ±0.53 | 3.66 ±0.21 | 9.50 ±1.10 | |
| Push Cart | HDMI ∗ | 0% | 136.2 ±23.3 | 778.5 ±218.6 | 158.9 ±4.7 | 7.36 ±0.48 | 5.32 ±0.21 | 15.62 ±0.74 |
| HAIC | 100% | 82.0 ±6.3 | 418.7 ±36.6 | 150.0 ±9.3 | 5.29 ±0.17 | 4.47 ±0.09 | 11.61 ±0.80 | |

**说明**: TABLE II: Real-world results for Cart Manipulation. HAIC achieves 100% success in both tasks, whereas the baseline fails in “Push”.

#### Table 3: TABLE III: Real-world results for Sequential Interaction. The baseline is omitted as it failed to complete the prerequ

| Task | Method | SR ↑ \uparrow | E mpbpe E_{ $\mathrm{mpbpe}} ↓ \downarrow$ | E mpboe E_{ $\mathrm{mpboe}} ↓ \downarrow$ | E mpjpe E_{ $\mathrm{mpjpe}} ↓ \downarrow$ | E mpbve E_{ $\mathrm{mpbve}} ↓ \downarrow$ | E mpbae E_{ $\mathrm{mpbae}} ↓ \downarrow$ | E mpjve E_{ $\mathrm{mpjve}} ↓ \downarrow$ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Pull Cart w/ Box | HDMI ∗ | 0% | - | - | - | - | - | - |
| HAIC | 40% | 88.2 ±8.2 | 420.7 ±16.7 | 157.2 ±8.2 | 5.91 ±0.07 | 5.02 ±0.09 | 12.20 ±0.31 | |
| Push Cart w/ Box | HDMI ∗ | 0% | - | - | - | - | - | - |
| HAIC | 100% | 70.5 ±5.7 | 381.7 ±24.7 | 143.6 ±4.1 | 5.27 ±0.26 | 3.92 ±0.18 | 11.10 ±0.85 | |

**说明**: TABLE III: Real-world results for Sequential Interaction. The baseline is omitted as it failed to complete the prerequisite cart tasks. HAIC demonstrates capability in sequential object transport.

#### Table 4: TABLE IV: Real-world results for Multi-terrain Interaction. HAIC succeeds on the complex slope-stair terrain where bas

| Task | Method | SR ↑ \uparrow | E mpbpe E_{ $\mathrm{mpbpe}} ↓ \downarrow$ | E mpboe E_{ $\mathrm{mpboe}} ↓ \downarrow$ | E mpjpe E_{ $\mathrm{mpjpe}} ↓ \downarrow$ | E mpbve E_{ $\mathrm{mpbve}} ↓ \downarrow$ | E mpbae E_{ $\mathrm{mpbae}} ↓ \downarrow$ | E mpjve E_{ $\mathrm{mpjve}} ↓ \downarrow$ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Carry Box | HDMI ∗ | 100% | 66.6 ±2.0 | 344.0 ±19.8 | 171.7 ±1.7 | 4.69 ±0.10 | 3.14 ±0.01 | 8.10 ±0.13 |
| HAIC | 100% | 58.8 ±6.1 | 305.6 ±20.8 | 162.0 ±6.8 | 4.21 ±0.13 | 3.03 ±0.05 | 7.75 ±0.18 | |
| w/ Stair | HDMI ∗ | 100% | 66.2 ±4.1 | 316.2 ±2.8 | 158.9 ±13.8 | 4.79 ±0.36 | 4.35 ±0.24 | 10.10 ±0.98 |
| HAIC | 100% | 55.5 ±3.9 | 331.2 ±23.1 | 118.4 ±0.6 | 4.06 ±0.02 | 4.05 ±0.01 | 8.66 ±0.01 | |
| w/ Slope | HDMI ∗ | 100% | 53.5 ±0.7 | 289.0 ±1.7 | 130.6 ±1.9 | 3.68 ±0.03 | 3.39 ±0.01 | 7.41 ±0.02 |
| HAIC | 100% | 60.5 ±4.3 | 324.7 ±28.0 | 120.2 ±1.6 | 8.77 ±0.11 | 4.60 ±0.13 | 3.66 ±0.05 | |
| w/ Stair + Slope | HDMI ∗ | 0% | 126.8 ±4.4 | 701.0 ±32.1 | 162.3 ±2.0 | 5.98 ±0.07 | 4.03 ±0.10 | 10.87 ±0.27 |
| HAIC | 100% | 60.6 ±13.4 | 286.5 ±59.8 | 128.5 ±16.2 | 4.74 ±0.17 | 4.05 ±0.04 | 10.67 ±0.52 | |

**说明**: TABLE IV: Real-world results for Multi-terrain Interaction. HAIC succeeds on the complex slope-stair terrain where baseline fails.

#### Table 5: TABLE V: Ablation results for Skateboarding. Vec-Pose can balance but fails to dismount due to a lack of acceleration

| | SR ↑ \uparrow | Robot State Metrics | Object State Metrics | | | | | | | | | | |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Method | Glide | Comp. | E mpbpe E_{ $\mathrm{mpbpe}} ↓ \downarrow$ | E g  -  mpbpe E_{ $\mathrm{g\text{-}mpbpe}} ↓ \downarrow$ | E mpboe E_{ $\mathrm{mpboe}} ↓ \downarrow$ | E mpjpe E_{ $\mathrm{mpjpe}} ↓ \downarrow$ | E mpbve E_{ $\mathrm{mpbve}} ↓ \downarrow$ | E mpbae E_{ $\mathrm{mpbae}} ↓ \downarrow$ | E mpjve E_{ $\mathrm{mpjve}} ↓ \downarrow$ | E mpope E_{ $\mathrm{mpope}} ↓ \downarrow$ | E mpooe E_{ $\mathrm{mpooe}} ↓ \downarrow$ | E mpove E_{ $\mathrm{mpove}} ↓ \downarrow$ | E mpoae E_{ $\mathrm{mpoae}} ↓ \downarrow$ |
| Proprio | 40% | 0% | 91.1 ±9.6 | 548.1 ±186.4 | 490.6 ±58.0 | 163.3 ±3.4 | 5.08 ±0.28 | 4.18 ±0.10 | 12.45 ±0.55 | 549.9 ±167.5 | 1219 ±179 | 9.63 ±0.64 | 11.02 ±0.25 |
| Vec-Pose | 100% | 0% | 77.1 ±12.7 | 430.3 ±248.2 | 394.0 ±55.0 | 167.0 ±4.7 | 4.25 ±0.20 | 3.94 ±0.09 | 10.14 ±0.10 | 434.8 ±259.9 | 1192 ±147 | 8.78 ±1.25 | 10.75 ±0.10 |
| Vec-Dyn | 100% | 100% | 74.6 ±0.8 | 444.7 ±13.3 | 416.7 ±6.8 | 147.7 ±2.3 | 4.87 ±0.16 | 4.11 ±0.16 | 12.20 ±0.56 | 432.1 ±9.1 | 1038 ±137 | 8.64 ±0.36 | 10.90 ±0.28 |
| Geo-Pose | 100% | 100% | 60.2 ±2.5 | 513.0 ±86.2 | 354.5 ±18.3 | 121.6 ±1.8 | 4.55 ±0.08 | 4.01 ±0.04 | 12.42 ±0.22 | 565.8 ±83.2 | 1071 ±72 | 8.77 ±0.98 | 10.74 ±0.11 |
| HAIC | 100% | 100% | 57.8 ±2.8 | 417.8 ±36.4 | 335.5 ±19.7 | 117.6 ±0.7 | 4.54 ±0.11 | 4.04 ±0.05 | 12.30 ±0.31 | 473.8 ±43.1 | 1035 ±57 | 8.08 ±0.32 | 10.66 ±0.01 |

**说明**: TABLE V: Ablation results for Skateboarding. Vec-Pose can balance but fails to dismount due to a lack of acceleration prediction. HAIC achieves a high completion rate with the lowest robot tracking errors and the most stable object acceleration control.

#### Table 6: TABLE VI: Ablation results for Cart Manipulation. HAIC shows consistently high performance, achieving either the best

| | | | Robot State Metrics | Object State Metrics | | | | | | | | | |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Task | Method | SR ↑ \uparrow | E mpbpe E_{ $\mathrm{mpbpe}} ↓ \downarrow$ | E g  -  mpbpe E_{ $\mathrm{g\text{-}mpbpe}} ↓ \downarrow$ | E mpboe E_{ $\mathrm{mpboe}} ↓ \downarrow$ | E mpjpe E_{ $\mathrm{mpjpe}} ↓ \downarrow$ | E mpbve E_{ $\mathrm{mpbve}} ↓ \downarrow$ | E mpbae E_{ $\mathrm{mpbae}} ↓ \downarrow$ | E mpjve E_{ $\mathrm{mpjve}} ↓ \downarrow$ | E mpope E_{ $\mathrm{mpope}} ↓ \downarrow$ | E mpooe E_{ $\mathrm{mpooe}} ↓ \downarrow$ | E mpove E_{ $\mathrm{mpove}} ↓ \downarrow$ | E mpoae E_{ $\mathrm{mpoae}} ↓ \downarrow$ |
| Pull Cart | Proprio | 60% | 97.9 ±34.7 | 341.9 ±114.6 | 575.4 ±188 | 137.4 ±6.5 | 5.58 ±0.73 | 3.75 ±0.22 | 11.39 ±1.69 | 448.8 ±190 | 1551 ±149 | 5.90 ±2.41 | 3.59 ±0.06 |
| Vec-Pose | 60% | 100.0 ±26.6 | 388.0 ±221.0 | 589.6 ±141 | 145.3 ±13.5 | 5.86 ±0.74 | 3.83 ±0.31 | 11.96 ±2.06 | 437.7 ±223 | 1530 ±212 | 5.78 ±1.96 | 3.55 ±0.03 | |
| Vec-Dyn | 100% | 71.3 ±0.5 | 244.1 ±22.3 | 406.9 ±9.3 | 133.7 ±1.0 | 5.12 ±0.04 | 3.54 ±0.01 | 9.88 ±0.18 | 230.1 ±16.0 | 1640 ±108 | 4.05 ±0.02 | 3.52 ±0.01 | |
| Geo-Pose | 100% | 66.1 ±0.3 | 261.5 ±32.7 | 384.1 ±21.7 | 127.3 ±4.4 | 5.29 ±0.66 | 3.65 ±0.19 | 10.85 ±1.52 | 320.3 ±89.5 | 1210 ±11.6 | 4.09 ±0.11 | 3.57 ±0.04 | |
| HAIC | 100% | 64.8 ±2.8 | 225.7 ±3.4 | 372.7 ±34.1 | 123.1 ±0.3 | 4.75 ±0.01 | 3.48 ±0.01 | 9.39 ±0.09 | 219.5 ±6.6 | 1241 ±247 | 4.19 ±0.02 | 3.55 ±0.01 | |
| Push Cart | Proprio | 0% | 112.9 ±9.2 | 276.6 ±80.5 | 538.4 ±129.2 | 172.1 ±14.2 | 6.50 ±0.41 | 4.78 ±0.14 | 15.38 ±0.85 | 460.7 ±286.2 | 1406 ±16 | 4.61 ±0.91 | 3.75 ±0.04 |
| Vec-Pose | 100% | 61.5 ±1.8 | 451.8 ±82.9 | 287.7 ±3.1 | 125.3 ±2.0 | 4.65 ±0.17 | 4.16 ±0.10 | 11.20 ±0.91 | 338.7 ±63.9 | 1152 ±58 | 4.25 ±0.43 | 3.71 ±0.01 | |
| Vec-Dyn | 100% | 62.3 ±2.5 | 439.0 ±75.7 | 299.9 ±16.3 | 126.4 ±2.2 | 4.58 ±0.05 | 4.12 ±0.04 | 10.86 ±0.39 | 358.4 ±73.1 | 1172 ±26 | 4.63 ±1.03 | 3.70 ±0.01 | |
| Geo-Pose | 100% | 64.0 ±4.8 | 249.1 ±43.3 | 346.8 ±28.5 | 141.0 ±3.1 | 4.93 ±0.06 | 4.28 ±0.03 | 12.70 ±0.16 | 259.8 ±66.8 | 1153 ±107 | 3.65 ±0.41 | 3.70 ±0.01 | |
| HAIC | 100% | 61.3 ±2.1 | 221.0 ±74.3 | 319.9 ±13.0 | 145.9 ±5.5 | 5.00 ±0.10 | 4.23 ±0.08 | 12.36 ±0.39 | 267.9 ±85.6 | 1262 ±96 | 4.04 ±1.07 | 3.70 ±0.01 | |

**说明**: TABLE VI: Ablation results for Cart Manipulation. HAIC shows consistently high performance, achieving either the best or second-best results across most metrics.

#### Table 7: TABLE VII: Compared to the RMA baseline, HAIC achieves superior or competitive success rates and significantly lower tr

| Task | Method | SR ↑ \uparrow | E mpbpe E_{ $\mathrm{mpbpe}} ↓ \downarrow$ | E g  -  mpbpe E_{ $\mathrm{g\text{-}mpbpe}} ↓ \downarrow$ | E mpboe E_{ $\mathrm{mpboe}} ↓ \downarrow$ | E mpjpe E_{ $\mathrm{mpjpe}} ↓ \downarrow$ | E mpope E_{ $\mathrm{mpope}} ↓ \downarrow$ | E mpooe E_{ $\mathrm{mpooe}} ↓ \downarrow$ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Pull Cart | RMA | 70% | 99.1 ±38.2 | 323.1 ±110.8 | 530.6 ±168.6 | 139.9 ±5.6 | 530.6 ±168.6 | 1474 ±178.5 |
| HAIC | 100% | 64.8 ±2.8 | 225.7 ±3.4 | 372.7 ±34.1 | 123.1 ±0.3 | 219.5 ±6.6 | 1241 ±246.5 | |
| Push Cart | RMA | 100% | 74.1 ±6.3 | 267.0 ±52.8 | 340.8 ±30.2 | 140.1 ±6.0 | 519.9 ±92.9 | 1120 ±31.0 |
| HAIC | 100% | 61.3 ±2.1 | 221.0 ±74.3 | 319.9 ±13.0 | 145.9 ±5.5 | 267.9 ±85.6 | 1262 ±96.2 | |

**说明**: TABLE VII: Compared to the RMA baseline, HAIC achieves superior or competitive success rates and significantly lower tracking errors.

#### Table 8: TABLE VIII: Ablation study on contact reward thresholds. We varied the tolerance (ε tol $\epsilon_{\text{tol}}$) and fo

| tol \epsilon $\text{tol}}$ | F thr F_{ $\text{thr}}$ | SR ↑ \uparrow | E mpbpe E_{ $\mathrm{mpbpe}} ↓ \downarrow$ | E g  -  mpbpe E_{ $\mathrm{g\text{-}mpbpe}} ↓ \downarrow$ | E mpboe E_{ $\mathrm{mpboe}} ↓ \downarrow$ | E mpjpe E_{ $\mathrm{mpjpe}} ↓ \downarrow$ | E mpope E_{ $\mathrm{mpope}} ↓ \downarrow$ | E mpooe E_{ $\mathrm{mpooe}} ↓ \downarrow$ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.1 | 0 | 100% | 77.6 ±15.1 | 511.7 ±525.8 | 372.7 ±59.3 | 157.3 ±16.5 | 795.7 ±722.9 | 1254 ±59.4 |
| 0.05 | 5 | 100% | 68.4 ±2.4 | 388.4 ±37.6 | 334.7 ±16.9 | 137 ±1.4 | 401.5 ±37.9 | 1147 ±227.4 |
| 0 | 10 | 100% | 61.3 ±2.1 | 221.0 ±74.3 | 319.9 ±13.0 | 145.9 ±5.5 | 267.9 ±85.6 | 1262 ±96.2 |

**说明**: 该表格来自论文中的公式、奖励项或实验设置；为避免 PDF 抽取噪声，网页中仅保留可渲染的核心符号和数值。

#### Table 9: TABLE IX: Details of the Observation Space and Privileged Information.

| Category | Observation Term | Noise ( \sigma) | Description |
| --- | --- | --- | --- |
| Proprioception | Base Angular Velocity | 0.05 | History of base angular velocity in base frame (steps: [0]). |
| Projected Gravity | 0.05 | History of gravity vector projected to base frame (steps: [0]). | |
| Joint Positions | 0.015 | History of joint positions (steps: [0, 1, 2, 3, 4, 8]). | |
| Previous Actions | - | Joint target history from previous 3 steps. | |
| Ref. Motion Phase | - | Phase variable  \phi of the reference motion. | |
| Ref. Joint Positions | - | Future reference joint positions. | |
| Ref. Body Pos (Local) | - | Future reference body positions in robot local frame. | |
| Ref. Contact 1 Pos | 0.01 | Reference contact positions on Object 1. | |
| | Ref. Contact 2 Pos | 0.01 | Reference contact positions on Object 2 (if enabled). |
| | Objects Point Cloud | - | Canonical point clouds of Object 1 & Object 2 (if enabled). |
| Privileged | Clean Proprioception | 0.0 | Noise-free history (steps: 0-8) of all proprioceptive states. |
| Root Linear Velocity | 0.0 | Linear velocity of the robot base in local frame. | |
| Body Velocity | 0.0 | Linear velocity of key bodies (e.g., ankles). | |
| Body Height | 0.0 | Height of pelvis, torso, and feet relative to ground. | |
| Ref. Root State (Global) | 0.0 | Future reference root position/orientation in global frame. | |
| Ref. Diff (Local) | 0.0 | Difference between current and ref (pos, ori, ...) in local frame. | |
| Obj. 1 State | 0.01 | Relative state (pos, ori, ...) of Object 1 in local frame. | |
| Obj. 2 State | 0.01 | Relative state (pos, ori, ...) of Object 2 (if enabled) in local frame. | |
| Applied Forces | - | Real applied actions and torques. | |
| Dynamics Randomization Parameters (Implicitly observed by Critic): | | | |
| Body/Object Mass, Friction, Restitution, Object Scale, Joint Armature/Damping. | | | |

**说明**: TABLE IX: Details of the Observation Space and Privileged Information.

#### Table 10: TABLE X: Reward Functions for the Humanoid-Object Interaction Task.

| Term | Expression | Weight | Description |
| --- | --- | --- | --- |
| (a) Motion Tracking Reward | | | |
| Joint Position | $\exp(-\lVert \boldsymbol{q}-\boldsymbol{q}^{\text{ref}} \rVert^{2}/\sigma_{q})$ | 0.5 | Tracks reference joint positions. |
| Joint Velocity | $\exp(-\lVert \dot{\boldsymbol{q}}-\dot{\boldsymbol{q}}^{\text{ref}} \rVert^{2}/\sigma_{\dot{q}})$ | 0.5 | Tracks reference joint velocities. |
| Upper Body Pos. | $\exp(-\lVert \boldsymbol{p}_{\text{up}}-\boldsymbol{p}_{\text{up}}^{\text{ref}} \rVert^{2}/\sigma_{p})$ | 0.5 | Tracks pos. of shoulder, elbow, wrist. |
| Upper Body Ori. | $\exp(-\lVert \boldsymbol{\theta}_{\text{up}}\ominus\boldsymbol{\theta}_{\text{up}}^{\text{ref}} \rVert^{2}/\sigma_{\theta})$ | 0.5 | Tracks ori. of upper body links. |
| Lower Body Pos. | $\exp(-\lVert \boldsymbol{p}_{\text{low}}-\boldsymbol{p}_{\text{low}}^{\text{ref}} \rVert^{2}/\sigma_{p})$ | 0.5 | Tracks pos. of hip, knee, ankle links. |
| Lower Body Ori. | $\exp(-\lVert \boldsymbol{\theta}_{\text{low}}\ominus\boldsymbol{\theta}_{\text{low}}^{\text{ref}} \rVert^{2}/\sigma_{\theta})$ | 0.5 | Tracks ori. of lower body links. |
| Root Position | $\exp(-\lVert \boldsymbol{p}_{\text{root}}-\boldsymbol{p}_{\text{root}}^{\text{ref}} \rVert^{2}/\sigma_{p})$ | 0.5 | Tracks pelvis position in world frame. |
| Root Orientation | $\exp(-\lVert \boldsymbol{\theta}_{\text{root}}\ominus\boldsymbol{\theta}_{\text{root}}^{\text{ref}} \rVert^{2}/\sigma_{\theta})$ | 0.5 | Tracks pelvis orientation. |
| Body Lin/Ang Vel. | $\exp(-\lVert \boldsymbol{v}-\boldsymbol{v}^{\text{ref}} \rVert^{2}/\sigma_{v})$ | 0.5 | Tracks lin/ang velocities of bodies. |
| (b) Object Interaction Reward | | | |
| Object Position | $\exp(-\lVert \boldsymbol{p}_{\text{obj}}-\boldsymbol{p}_{\text{obj}}^{\text{ref}} \rVert^{2}/\sigma_{p})$ | 1.0 | Tracks global position of object(s). |
| Object Orientation | $\exp(-\lVert \boldsymbol{\theta}_{\text{obj}}\ominus\boldsymbol{\theta}_{\text{obj}}^{\text{ref}} \rVert^{2}/\sigma_{\theta})$ | 1.0 | Tracks global orientation of object(s). |
| Multiple Objects Contact | $\frac{1}{\lVert \mathcal{O} \rVert}\sum_{o\in\mathcal{O}}\left(\frac{1}{\lVert \mathcal{E}_{o} \rVert}\sum_{e\in\mathcal{E}_{o}}\mathbb{I}_{o,e}\cdot r_{\text{pos}}^{o,e}\cdot r_{\text{force}}^{o,e}\right)$ | 1.0 | Align active body-object pairs & forces. |
| (c) Foot Constraints | | | |
| Feet Air Time | $\exp(\text{clip}(t_{\text{air}}-t_{\text{thr}})/\sigma)\cdot\mathbb{I}_{\text{step}}$ | 0.5 | Encourages longer swing phases. |
| Feet Slip | $\boldsymbol{v}_{\text{foot}}^{xy}\lVert \cdot\mathbb{I}_{\text{ground}}$ | 0.5 | Penalizes sliding velocity when grounded. |
| Feet Contact Match | $\exp(-\lVert \mathbb{I}_{\text{con}}^{\text{real}}-\mathbb{I}_{\text{con}}^{\text{ref}} \rVert^{2}/\sigma)$ | 0.5 | Matches reference contact states. |
| Feet Air Lift | $\sum(\boldsymbol{h}_{\text{foot}}<h_{\text{min}})\cdot\mathbb{I}_{\text{swing}}$ | 0.5 | Penalizes tripping during swing. |
| Impact Force | $\boldsymbol{F}_{\text{impact}}\lVert^{2}$ | 1.0 | Penalizes large impact forces. |
| (d) Regularization | | | |
| Action Rate | $\boldsymbol{a}_{t}-\boldsymbol{a}_{t-1}\lVert^{2}$ | 0.1 | Penalizes rapid action changes. |
| Joint Velocity L2 | $\dot{\boldsymbol{q}}\lVert^{2}$ | 5e-4 | Penalizes high velocity (energy). |
| Joint Limits | - ∑ clip  (q - q limit) - $\sum\text{clip}(\boldsymbol{q}-\boldsymbol{q}_{\text{limit}})$ | 10.0 | Penalizes exceeding physical limits. |
| Torque Limits | $\sum\text{clip}(\boldsymbol{\tau}-\boldsymbol{\tau}_{\text{limit}})$ | 0.01 | Penalizes torque saturation. |
| Survival | 1.0 1.0 | 1.0 | Reward for not terminating early. |

**说明**: TABLE X: Reward Functions for the Humanoid-Object Interaction Task.

#### Table 11: TABLE XI: Domain Randomization Parameters for Robot and Objects.

| Category | Parameter | Range / Distribution |
| --- | --- | --- |
| Robot Dynamics | | |
| Properties | Link Mass Scale | U . . default $\mathcal{U}(0.9,1.1)\times\text{default}$ |
| Center of Mass Offset | U  (- 0.02, 0.02)  $\mathcal{U}(-0.02,0.02) m$ | |
| Static Friction | U  (0.3, 1.6)  $\mathcal{U}(0.3,1.6)$ | |
| Dynamic Friction | U  (0.3, 1.2)  $\mathcal{U}(0.3,1.2)$ | |
| Actuation | Joint Position Offset | U  (- 0.01, 0.01)  $\mathcal{U}(-0.01,0.01) rad$ |
| Motor Stiffness Scale | U  (0.9, 1.1)  $\mathcal{U}(0.9,1.1)$ | |
| Motor Damping Scale | U  (0.9, 1.1)  $\mathcal{U}(0.9,1.1)$ | |
| Action Delay | U  [40, 120 ]  $\mathcal{U}[40,120] ms$ | |
| Object Interaction | | |
| Surface | Dynamic Friction | U  (0.3, 0.8)  $\mathcal{U}(0.3,0.8)$ |
| Static-to-Dynamic Ratio | U  (1.0, 2.0)  $\mathcal{U}(1.0,2.0)$ | |
| Restitution | U  (0.0, 0.2)  $\mathcal{U}(0.0,0.2)$ | |
| Box | Mass | U  (1.0, 2.0)  $\mathcal{U}(1.0,2.0) kg$ |
| Scale | U  (0.9, 1.1)  $\mathcal{U}(0.9,1.1)$ | |
| Cart | Body Mass | U  (11.0, 13.0)  $\mathcal{U}(11.0,13.0) kg$ |
| Wheel Mass | U  (0.2, 0.4)  $\mathcal{U}(0.2,0.4) kg$ | |
| Wheel Joint Friction | U  (0.01, 0.1)  $\mathcal{U}(0.01,0.1) N ⋅ \cdot m$ | |
| Wheel Joint Damping | U  (0.01, 0.1)  $\mathcal{U}(0.01,0.1) N ⋅ \cdot m ⋅ \cdot s/rad$ | |
| Scale | U  (0.9, 1.1)  $\mathcal{U}(0.9,1.1)$ | |
| Skateboard | Body Mass | U  (2.0, 5.0)  $\mathcal{U}(2.0,5.0) kg$ |
| Wheel Mass | U  (0.1, 0.2)  $\mathcal{U}(0.1,0.2) kg$ | |
| Wheel Armature | U  (0.0, 1  e-  4)  $\mathcal{U}(0.0,1\text{e-}4) kg ⋅ \cdot m 2$ | |
| Wheel Joint Damping | U  (0.0, 1  e-  3)  $\mathcal{U}(0.0,1\text{e-}3) N ⋅ \cdot m ⋅ \cdot s/rad$ | |
| Scale | U  (0.9, 1.1)  $\mathcal{U}(0.9,1.1)$ | |
| Slope / Stair | Scale | U  (0.98, 1.02)  $\mathcal{U}(0.98,1.02)$ |
| External Perturbation | | |
| Push | Push Force | $\mathcal{U}(0.2,0.5)\times\text{weight}$ |
| Push Min Interval | 2s | |

**说明**: TABLE XI: Domain Randomization Parameters for Robot and Objects.

#### Table 12: TABLE XII: Hyperparameters related to PPO and Network Architecture.

| Hyperparameter | Value |
| --- | --- |
| Optimizer | Adam |
| Number of Environments | 4096 |
| Rollout Steps (Horizon) | 32 |
| Mini-batches | 8 |
| Learning Epochs | 3 |
| Discount Factor $\gamma)$ | 0.99 |
| GAE Parameter ( \lambda) | 0.95 |
| Clip Parameter ( \epsilon) | 0.2 |
| Entropy Coefficient | 0.001 |
| Max Gradient Norm | 1.0 |
| Desired KL | 0.01 |
| Learning Rate | 3  10 - 4 3\times 10^{-4} |
| Initial Noise Std | 1.0 |
| Loss Coefficients | |
| Value Loss Coefficient value \lambda $\text{value}})$ | 1.0 |
| Object Loss Coefficient obj \lambda $\text{obj}})$ | 1.0 |
| Privileged Loss Coefficient priv \lambda $\text{priv}})$ | 1.0 |
| Distillation Loss Coefficient distill \lambda $\text{distill}})$ | 1.0 |
| Network Architecture | |
| Actor MLP Size | [512, 256, 256 ] [512,256,256] |
| Critic MLP Size | [512, 256, 128 ] [512,256,128] |
| Adapter MLP Size | [256, 256 ] [256,256] |
| Activation Function | ELU |

**说明**: TABLE XII: Hyperparameters related to PPO and Network Architecture.

#### Table 13: TABLE XIII: PD controller gains for the G1 Humanoid.

| Joint Name | Stiffness (k p k_{p}) | Damping (k d k_{d}) |
| --- | --- | --- |
| Legs | | |
| Left/Right Hip Pitch/Yaw | 40.18 | 2.558 |
| Left/Right Hip Roll | 99.10 | 6.309 |
| Left/Right Knee | 99.10 | 6.309 |
| Left/Right Ankle Pitch/Roll | 28.50 | 1.814 |
| Waist | | |
| Waist Yaw | 40.18 | 2.558 |
| Waist Roll/Pitch | 28.50 | 1.814 |
| Arms | | |
| Left/Right Shoulder Pitch/Roll/Yaw | 14.25 | 0.9072 |
| Left/Right Elbow Pitch/Roll | 14.25 | 0.9072 |
| Left/Right Wrist Roll/Pitch | 16.78 | 1.068 |

**说明**: TABLE XIII: PD controller gains for the G1 Humanoid.

#### Table 14: TABLE XIV: Ablation results for Sequential Interaction. The complexity of sequentially loading and manipulating object

| | | | Robot State Metrics | Object State Metrics | | | | | | | | | |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Task | Method | SR ↑ \uparrow | E mpbpe E_{ $\mathrm{mpbpe}} ↓ \downarrow$ | E g  -  mpbpe E_{ $\mathrm{g\text{-}mpbpe}} ↓ \downarrow$ | E mpboe E_{ $\mathrm{mpboe}} ↓ \downarrow$ | E mpjpe E_{ $\mathrm{mpjpe}} ↓ \downarrow$ | E mpbve E_{ $\mathrm{mpbve}} ↓ \downarrow$ | E mpbae E_{ $\mathrm{mpbae}} ↓ \downarrow$ | E mpjve E_{ $\mathrm{mpjve}} ↓ \downarrow$ | E mpope E_{ $\mathrm{mpope}} ↓ \downarrow$ | E mpooe E_{ $\mathrm{mpooe}} ↓ \downarrow$ | E mpove E_{ $\mathrm{mpove}} ↓ \downarrow$ | E mpoae E_{ $\mathrm{mpoae}} ↓ \downarrow$ |
| Pull Cart w/ Box | Proprio | 0% | 117.9 ±31.3 | 354.7 ±93.0 | 590.8 ±124 | 162.7 ±14.1 | 5.67 ±0.24 | 4.64 ±0.04 | 12.21 ±0.15 | 501.6 ±47.3 | 454.4 ±13.5 | 4.63 ±1.04 | 2.87 ±0.01 |
| Vec-Pose | 0% | 138.7 ±2.5 | 444.8 ±118 | 780.9 ±75.7 | 155.8 ±1.4 | 5.25 ±0.09 | 4.55 ±0.02 | 10.57 ±0.25 | 550.6 ±218 | 438.7 ±77.6 | 4.83 ±0.92 | 2.87 ±0.01 | |
| Vec-Dyn | 100% | 59.7 ±2.7 | 204.7 ±21.2 | 333.5 ±15.0 | 130.0 ±2.0 | 5.18 ±0.14 | 4.64 ±0.10 | 11.88 ±0.42 | 300.6 ±26.1 | 499.3 ±66.3 | 4.06 ±0.39 | 2.88 ±0.01 | |
| Geo-Pose | 100% | 69.8 ±2.0 | 302.2 ±65.9 | 376.6 ±14.5 | 138.4 ±0.9 | 5.63 ±0.13 | 4.75 ±0.02 | 12.45 ±0.19 | 414.7 ±64.7 | 635.5 ±68.1 | 4.47 ±0.37 | 2.88 ±0.01 | |
| HAIC | 100% | 64.4 ±2.1 | 196.5 ±29.5 | 343.7 ±20.3 | 135.8 ±5.2 | 5.24 ±0.14 | 4.65 ±0.06 | 12.17 ±0.46 | 332.9 ±30.0 | 561.3 ±22.6 | 4.35 ±0.26 | 2.86 ±0.01 | |
| Push Cart w/ Box | Proprio | 0% | 99.7 ±39.2 | 520.7 ±240 | 605.0 ±263 | 167.0 ±9.6 | 5.67 ±0.64 | 3.84 ±0.27 | 12.23 ±1.42 | 536.7 ±171 | 362.2 ±78.5 | 4.43 ±1.01 | 3.03 ±0.03 |
| Vec-Pose | 80% | 66.2 ±2.0 | 451.9 ±69.3 | 393.6 ±13.4 | 175.2 ±3.5 | 4.97 ±0.15 | 3.52 ±0.11 | 11.42 ±0.52 | 499.3 ±79.4 | 422.3 ±41.9 | 4.22 ±0.28 | 3.02 ±0.01 | |
| Vec-Dyn | 100% | 68.3 ±5.0 | 490.2 ±46.3 | 407.7 ±21.9 | 156.2 ±6.1 | 5.00 ±0.24 | 3.48 ±0.09 | 11.68 ±0.51 | 519.0 ±47.1 | 386.3 ±72.6 | 4.06 ±0.22 | 3.01 ±0.01 | |
| Geo-Pose | 100% | 58.2 ±1.0 | 360.2 ±9.1 | 324.8 ±6.0 | 141.6 ±2.0 | 4.95 ±0.13 | 3.54 ±0.08 | 11.39 ±0.42 | 491.3 ±68.1 | 419.1 ±63.7 | 4.11 ±0.12 | 3.02 ±0.02 | |
| HAIC | 100% | 60.2 ±3.2 | 326.5 ±11.3 | 327.7 ±4.3 | 140.8 ±0.8 | 4.75 ±0.03 | 3.42 ±0.02 | 10.73 ±0.05 | 386.7 ±18.0 | 326.9 ±7.3 | 4.00 ±0.13 | 3.00 ±0.01 | |

**说明**: TABLE XIV: Ablation results for Sequential Interaction. The complexity of sequentially loading and manipulating objects causes error accumulation in Proprio. Proprio and Vec-Pose fail completely on “Pull Cart w/ Box”. HAIC maintains a high success rate across both tasks and achieves the lowest object orientation error in the “Push” phase, confirming that the dynamic-aware world model effectively mitigates drift long horizons.

#### Table 15: TABLE XV: Ablation results for Multi-terrain Interaction. HAIC achieves the most consistent success rate across all te

| | | | Robot State Metrics | Object State Metrics | | | | | | | | | |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Task | Method | SR ↑ \uparrow | E mpbpe E_{ $\mathrm{mpbpe}}$ | E g  -  mpbpe E_{ $\mathrm{g\text{-}mpbpe}} ↓ \downarrow$ | E mpboe E_{ $\mathrm{mpboe}} ↓ \downarrow$ | E mpjpe E_{ $\mathrm{mpjpe}} ↓ \downarrow$ | E mpbve E_{ $\mathrm{mpbve}} ↓ \downarrow$ | E mpbae E_{ $\mathrm{mpbae}} ↓ \downarrow$ | E mpjve E_{ $\mathrm{mpjve}} ↓ \downarrow$ | E mpope E_{ $\mathrm{mpope}} ↓ \downarrow$ | E mpooe E_{ $\mathrm{mpooe}} ↓ \downarrow$ | E mpove E_{ $\mathrm{mpove}} ↓ \downarrow$ | E mpoae E_{ $\mathrm{mpoae}} ↓ \downarrow$ |
| Carry Box | Proprio | 100% | 61.8 ±4.8 | 166.3 ±10.3 | 323.6 ±16.7 | 181.6 ±5.0 | 5.34 ±0.23 | 3.34 ±0.07 | 11.08 ±0.50 | 180.1 ±12.0 | 129.0 ±7.3 | 7.23 ±0.09 | 1.84 ±0.05 |
| Vec-Pose | 100% | 59.3 ±4.4 | 155.7 ±17.6 | 315.9 ±19.3 | 194.5 ±1.4 | 5.78 ±0.12 | 3.72 ±0.08 | 13.49 ±0.28 | 161.8 ±24.6 | 149.9 ±22.6 | 6.92 ±0.22 | 2.23 ±0.11 | |
| Vec-Dyn | 100% | 73.2 ±5.3 | 220.2 ±52.1 | 340.2 ±24.1 | 198.2 ±2.9 | 5.19 ±0.06 | 3.32 ±0.06 | 11.20 ±0.28 | 197.5 ±50.4 | 517.4 ±45.8 | 7.24 ±0.35 | 1.99 ±0.08 | |
| Geo-Pose | 100% | 69.4 ±1.6 | 221.8 ±13.8 | 318.7 ±6.9 | 209.3 ±1.2 | 4.82 ±0.12 | 3.18 ±0.07 | 10.20 ±0.18 | 184.6 ±22.5 | 258.3 ±8.7 | 6.26 ±0.25 | 1.71 ±0.04 | |
| HAIC | 100% | 64.3 ±2.9 | 173.2 ±34.4 | 320.4 ±18.7 | 182.5 ±0.8 | 5.17 ±0.05 | 3.23 ±0.04 | 10.29 ±0.15 | 176.0 ±30.9 | 169.2 ±5.5 | 6.42 ±0.23 | 1.62 ±0.02 | |
| w/ Stair | Proprio | 100% | 66.1 ±4.6 | 234.7 ±28.8 | 351.5 ±15.5 | 166.5 ±10.8 | 4.95 ±0.20 | 4.14 ±0.06 | 12.02 ±0.36 | 148.2 ±20.9 | 193.9 ±100.8 | 3.88 ±0.40 | 2.57 ±0.09 |
| Vec-Pose | 100% | 55.3 ±1.3 | 218.0 ±11.4 | 300.6 ±9.8 | 132.3 ±1.5 | 4.34 ±0.05 | 3.95 ±0.02 | 9.41 ±0.17 | 138.4 ±35.5 | 390.9 ±101.2 | 3.79 ±0.23 | 2.40 ±0.06 | |
| Vec-Dyn | 100% | 57.8 ±4.4 | 216.4 ±9.9 | 317.3 ±30.3 | 132.1 ±0.5 | 4.39 ±0.02 | 3.97 ±0.03 | 9.61 ±0.01 | 149.5 ±29.0 | 294.8 ±145.7 | 3.73 ±0.06 | 2.38 ±0.02 | |
| Geo-Pose | 100% | 53.7 ±1.0 | 283.0 ±30.8 | 302.9 ±1.7 | 132.6 ±1.3 | 3.91 ±0.01 | 3.84 ±0.03 | 8.91 ±0.11 | 168.9 ±7.9 | 82.0 ±2.6 | 3.17 ±0.05 | 2.32 ±0.04 | |
| HAIC | 100% | 53.5 ±0.9 | 281.5 ±72.9 | 290.5 ±0.8 | 127.5 ±2.5 | 4.34 ±0.04 | 3.95 ±0.03 | 9.67 ±0.08 | 154.5 ±29.8 | 71.0 ±7.4 | 3.41 ±0.03 | 2.40 ±0.02 | |
| w/ Slope | Proprio | 100% | 54.6 ±1.1 | 186.6 ±14.4 | 317.7 ±3.6 | 138.4 ±2.4 | 3.72 ±0.04 | 3.30 ±0.03 | 8.78 ±0.08 | 105.4 ±5.1 | 108.4 ±17.0 | 3.06 ±0.06 | 1.71 ±0.02 |
| Vec-Pose | 100% | 60.7 ±2.5 | 235.6 ±16.5 | 340.8 ±9.8 | 150.4 ±2.0 | 3.91 ±0.03 | 3.35 ±0.04 | 9.20 ±0.09 | 130.5 ±9.5 | 109.5 ±18.6 | 3.29 ±0.09 | 1.74 ±0.01 | |
| Vec-Dyn | 100% | 59.4 ±1.2 | 229.4 ±12.1 | 322.9 ±7.0 | 129.7 ±0.5 | 3.93 ±0.04 | 3.33 ±0.02 | 9.30 ±0.13 | 134.4 ±3.0 | 75.7 ±5.1 | 3.37 ±0.14 | 1.74 ±0.01 | |
| Geo-Pose | 100% | 70.3 ±1.9 | 198.9 ±28.8 | 404.3 ±3.4 | 175.9 ±3.1 | 4.39 ±0.04 | 3.43 ±0.03 | 9.96 ±0.10 | 198.9 ±28.8 | 145.1 ±24.1 | 3.31 ±0.16 | 1.84 ±0.02 | |
| HAIC | 100% | 58.5 ±1.2 | 163.1 ±31.0 | 314.4 ±8.0 | 131.8 ±4.0 | 3.90 ±0.08 | 3.32 ±0.02 | 8.94 ±0.26 | 99.0 ±11.3 | 80.4 ±9.4 | 2.93 ±0.08 | 1.73 ±0.03 | |
| w/ Slope + Stair | Proprio | 60% | 91.0 ±36.9 | 702.9 ±203.9 | 462.4 ±192.2 | 185.3 ±16.5 | 5.22 ±0.53 | 3.86 ±0.21 | 13.01 ±1.17 | 358.9 ±72.4 | 337.1 ±117.8 | 4.33 ±0.45 | 2.30 ±0.05 |
| Vec-Pose | 60% | 93.2 ±52.6 | 448.1 ±122.3 | 503.2 ±312.7 | 165.5 ±26.4 | 4.84 ±0.32 | 3.66 ±0.14 | 11.87 ±0.84 | 222.1 ±37.6 | 183.7 ±88.7 | 4.49 ±0.17 | 2.30 ±0.04 | |
| Vec-Dyn | 100% | 65.3 ±6.0 | 336.2 ±69.8 | 367.8 ±44.9 | 151.5 ±1.1 | 4.29 ±0.12 | 3.47 ±0.03 | 10.75 ±0.28 | 202.8 ±30.6 | 130.1 ±27.2 | 3.92 ±0.26 | 2.26 ±0.02 | |
| Geo-Pose | 100% | 62.3 ±6.9 | 320.3 ±129.0 | 321.0 ±61.0 | 144.1 ±8.2 | 4.63 ±0.44 | 3.83 ±0.05 | 11.92 ±1.25 | 149.7 ±5.6 | 179.1 ±173.9 | 3.53 ±0.63 | 1.72 ±0.47 | |
| HAIC | 100% | 67.7 ±1.9 | 250.1 ±29.4 | 327.8 ±7.5 | 154.2 ±4.2 | 4.75 ±0.16 | 3.56 ±0.07 | 11.98 ±0.61 | 157.2 ±3.8 | 121.4 ±7.8 | 4.01 ±0.02 | 2.29 ±0.01 | |

**说明**: TABLE XV: Ablation results for Multi-terrain Interaction. HAIC achieves the most consistent success rate across all terrains, while Geo-Pose demonstrates superior stability in velocity and acceleration metrics on complex stair terrains due to the explicit geometric projection.
## 实验解读

- 评价重点:围绕 adaptive-control、agile-locomotion、碰撞避免、接触推理、人形机器人,优先看真实机器人结果、扰动恢复、接触稳定性、控制误差、成功率、实时性和 sim-to-real 差距。
- System0 视角:实验结论不只看最终成功率,还要看闭环过程中是否能处理状态估计误差、执行延迟、碰撞/接触突变和外部扰动。
- 与上层系统的关系:如果方法能在快闭环内稳定工作,它可以作为 System1 技能或 System2 规划的底层执行保障;如果只在离线或慢速设定下成立,则还不能直接作为身体级快闭环模块。
## 批判性思考

### 优点

- 论文把 adaptive-control、agile-locomotion、碰撞避免、接触推理、人形机器人 放到机器人身体级闭环里讨论,和只做高层语义推理的工作相比,更接近真实机器人部署时会遇到的状态、动力学和安全约束。
- 若论文同时给出公式、系统结构图、硬件实验或消融结果,这些证据可以帮助判断方法是否真的能支撑毫秒到百毫秒级反应。

### 局限

- 需要重点核查控制频率、推理延迟、传感延迟、失败恢复、硬件磨损、未见场景泛化和安全边界;这些因素决定它能否成为可靠的 System0 模块。
- 对学习型方法,还要区分训练环境中的鲁棒性和真实部署中的鲁棒性,尤其是接触丰富任务、动态运动任务和人形机器人任务。
## 论文定位

- 研究对象:HAIC: Humanoid Agile Object Interaction Control via Dynamics-Aware World Model。
- 关键词:adaptive-control、agile-locomotion、碰撞避免、接触推理、人形机器人、inertial-estimation、足式运动、non-prehensile-manipulation、proprioception、机器人操作。
- System0 关联:关注机器人身体级快反应链路中“状态/接触/局部感知 -> 约束/优化/策略 -> 动作”的短闭环,不把问题停留在语言规划或任务分解层。

## 速查卡片

> [!summary] HAIC
> - **论文**: https://www.roboticsproceedings.org/rss22/p013.pdf
> - **arXiv**: http://arxiv.org/abs/2602.11758v2
> - **arXiv HTML**: https://arxiv.org/html/2602.11758v2
