---
title: "PP-Tac: Paper Picking Using Omnidirectional Tactile Feedback in Dexterous Robotic Hands"
method_name: "PP-Tac"
authors: ["Pei Lin"]
year: 2025
venue: "RSS"
tags: ["contact-reasoning", "robot-manipulation", "robust-control", "real-time-control", "reinforcement-learning", "closed-loop-control", "dexterous-manipulation", "tactile-feedback", "state-estimation"]
image_source: "online"
arxiv_html: "https://arxiv.org/html/2504.16649v2"
---
# PP-Tac
## 一句话总结

> PP-Tac: Paper Picking Using Omnidirectional Tactile Feedback in Dexterous Robotic Hands 主要落在 [[closed-loop-control]]、[[contact-estimation]]、[[接触推理]]、[[灵巧操作]] 这一类问题上;从 System0 视角看,它的价值在于把身体状态、接触变化、局部感知或动力学约束接到机器人低层闭环里,让动作生成更稳定、安全、实时。
## 核心贡献

1. 围绕 **PP-Tac: Paper Picking Using Omnidirectional Tactile Feedback in Dexterous Robotic Hands** 建立了一个与 closed-loop-control、contact-estimation、接触推理、灵巧操作、diffusion-policy、力控制 相关的机器人问题设定,重点不是高层语义规划,而是身体级闭环中的状态变化、接触变化或控制约束。
2. 方法层面的关键价值在于把感知、动力学、约束优化、策略学习或硬件反馈组织成可执行的低层动作生成机制,适合放进 System0 的快反应模块中理解。
3. 对系统集成而言,这类工作补上了 System1 技能选择与电机级控制之间的一层:它负责把上层目标变成动态可行、可恢复、可安全过滤的动作。
4. 若论文包含真实机器人、接触任务或仿真到真实迁移,核心应看它是否报告了闭环稳定性、扰动恢复、硬件延迟、失败模式和跨场景泛化。
## 问题背景

这篇论文可以放在 System0 的“具身快系统”背景下阅读:机器人需要在很短时间内根据本体状态、接触、力/触觉、局部视觉或动力学预测做出动作,而不是等待慢速规划器重新推理整个任务。closed-loop-control、contact-estimation、接触推理、灵巧操作、diffusion-policy、力控制、grasping 相关问题的难点通常不在“知道要做什么”,而在动作执行过程中状态会快速偏离、接触会突然改变、约束会实时收紧,低层系统必须及时修正。

### System0 关联

精读时优先看三件事:第一,论文使用哪些快速反馈量,例如关节状态、接触状态、力觉、触觉、深度或局部视觉;第二,反馈如何进入 MPC、WBC、策略网络、安全过滤器或恢复控制器;第三,输出是否直接影响机器人下一步动作,而不是只服务离线分析或高层任务规划。
## 方法详解

### System0 视角解析

从 System0 角度看,这篇论文的方法部分应当围绕 closed-loop-control、contact-estimation、接触推理、灵巧操作、diffusion-policy、力控制、grasping、实时控制 来读。核心问题是:系统如何从快速变化的机器人状态中构造控制输入,如何处理接触、碰撞、约束、模型误差或执行延迟,以及如何把这些信息变成下一时刻的动作。

如果方法里有优化问题,重点看目标函数、约束和求解频率;如果方法里有学习策略,重点看观测空间、动作空间、训练扰动和部署时的闭环频率;如果方法里有硬件或传感器,重点看传感链路是否真的进入控制回路。这样读可以把论文和 System2 语言规划、System1 技能编排区分开来。

### 关键公式与机制

#### 公式 1: [[动力学或策略机制]]

$$D(u,v)=\begin{bmatrix}Z_{c}K^{-1}\begin{bmatrix}u\\ v\\ 1\end{bmatrix}\end{bmatrix}_{[3,:]},$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 2: [[动力学或策略机制]]

$$\begin{bmatrix}u\\ v\\ 1\end{bmatrix}=K\begin{bmatrix}\frac{(A[x,y,z]^{T}+b)_{x}}{Z_{c}}\\ \frac{(A[x,y,z]^{T}+b)_{y}}{Z_{c}}\\ 1\end{bmatrix},Z_{c}=(A[x,y,z]^{T}+b)_{z}.$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 3: [[优化目标/约束]]

$$=\arg\min_{\boldsymbol{\gamma}}\left(L_{\boldsymbol{ee}}+L_{\Delta}+L_{\boldsymbol{R},\boldsymbol{p}_{wrist}}\right),$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 4: [[动力学或策略机制]]

$${}^{\mathit{c^{\prime}}}_{\mathit{c^{\prime\prime}}}\hat{T}=\begin{bmatrix}R(\theta^{\prime},\boldsymbol{v_{2}})&0\\ 0&1\end{bmatrix},$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 5: [[动力学或策略机制]]

$$=w_{\boldsymbol{R},\boldsymbol{p}_{wrist}}\ \mathbf{MSE}\left((\boldsymbol{\bar{R}},\boldsymbol{\bar{p}}_{wrist}),\right.$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 6: [[动力学或策略机制]]

$${}^{\mathit{c}}_{\mathit{c^{\prime}}}\hat{T}=\begin{bmatrix}R(\theta,\boldsymbol{v_{1}})&0\\ 0&1\end{bmatrix},$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 7: [[动力学或策略机制]]

$${}^{\mathit{b}}_{\mathit{ee^{\prime\prime}}}T=^{\mathit{b}}_{\mathit{ee^{\prime}}}T\ ^{\mathit{ee^{\prime}}}_{\mathit{c^{\prime}}}T\,^{\mathit{c^{\prime}}}_{\mathit{c^{\prime\prime}}}\hat{T}\,{{}^{\mathit{c^{\prime\prime}}}_{\mathit{ee^{\prime\prime}}}T},$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 8: [[动力学或策略机制]]

$$=w_{\Delta}\ \mathbf{MSE}\left(\bar{\boldsymbol{\gamma}},\boldsymbol{\gamma}\right),$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 9: [[动力学或策略机制]]

$${}^{\mathit{b}}_{\mathit{ee^{\prime}}}T=^{\mathit{b}}_{\mathit{ee}}T\ ^{\mathit{ee}}_{\mathit{c}}T\,^{\mathit{c}}_{\mathit{c^{\prime}}}\hat{T}\,{{}^{\mathit{c^{\prime}}}_{\mathit{ee^{\prime}}}T},$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 10: [[动力学或策略机制]]

$$=w_{\boldsymbol{ee}}\ \mathbf{MSE}(\mathbf{fk}(\boldsymbol{\gamma}),\boldsymbol{ee}_{target}),$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。
## 关键图表

### Figure 1: Pipeline / core system figure: The hardware design of the R-Tac and its integration into the four-fingered dexterous

![Figure 1](https://arxiv.org/html/2504.16649v2/extracted/6551759/figure/hardware_new.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Pipeline / core system figure: The hardware design of the R-Tac and its integration into the four-fingered dexterous”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 2: Inference pipeline of the proposed PP-Tac policy. Conditioned on robot proprioception

![Figure 2](https://arxiv.org/html/2504.16649v2/x1.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Inference pipeline of the proposed PP-Tac policy. Conditioned on robot proprioception”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 3: Experimental Results. Evaluations were conducted to quantify the success rate of gras

![Figure 3](https://arxiv.org/html/2504.16649v2/extracted/6551759/figure/experiment-01.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Experimental Results. Evaluations were conducted to quantify the success rate of gras”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### 论文表格

#### Table 1: TABLE I: Experimental results for varying paper quantities: The system’s performance was evaluated on paper materials w

| Paper Layers | No. Slip | Succ. Rate (%) |
| --- | --- | --- |
| 1 | 0.2 | 90 |
| 3 | 2.9 | 75 |
| 5 | 13.3 | 30 |
| 7 | 18.2 | 5 |

**说明**: TABLE I: Experimental results for varying paper quantities: The system’s performance was evaluated on paper materials with different buckling strengths, achieved by bonding 1, 3, 5, and 7 layers of paper with adhesive. For each configuration, 20 trials of grasps were conducted. The average number of slip events detected (No. Slip) and the final success rate (Succ. Rate) were recorded.

#### Table 2: TABLE II: Summary of symbols and notations.

| Symbols | Descriptions |
| --- | --- |
| u, v u v u,v, | Pixel coordinates in VBTS. |
| X c, Y c, Z c X c Y c Z c X_{c},Y_{c},Z_{c},, | Camera coordinates in VBTS. |
| x, y, z x y z x,y,z,, | Sensor coordinates in VBTS. |
| K K K | The intrinsic parameters of the camera in VBTS. |
| A, b A b A,b, | The extrinsic parameters of the camera in VBTS. |
| D D D | Sensor surface reference projection in VBTS. |
| M M M | Depth mapping function in VBTS. |
| q q  $\boldsymbol{q}$ | Rotation angle of controllable hand joints. |
| q  ̇  ̇ q  $\dot{\boldsymbol{q}}$ | Angular velocity of controllable hand joints. |
| p p  $\boldsymbol{p}$ | Positional coordinate of hand joints in arm’s base axis. |
| p  ̇  ̇ p  $\dot{\boldsymbol{p}}$ | Linear velocity of hand joints in arm’s base axis. |
| R R R | Wrist’s (end effector of arm) 6D rotation. |
|   \Omega | Angular velocity of hand pose. |
| p w r i s t p w r i s t p_{wrist} | Wrist (end-effector of arm)’s height along arm’s z z z -axis. |
| p  ̇ w r i s t  ̇ p w r i s t  $\dot{p}_{wrist}$ | Linear velocity of p e e p e e  $\boldsymbol{p}_{ee}.$ |
| d t a c d t a c  $\boldsymbol{d}_{tac}$ | The deformation depth readings from four fingertip tactile sensors. |
| d  ̄ t a c  ̄ d t a c \bar{ $\boldsymbol{d}}_{tac}$ | The target deformation depth. |
| D D  $\mathcal{D}$ | State variable’s dimension. |
| $\gamma$ | Hand joint angles q 1: N d a t a q: 1 N d a t a  $\boldsymbol{q}^{1:N_{data}} 1:, wrist’s (end effector of arm) 6D rotation R 1: N d a t a R: 1 N d a t a R^{1:N_{data}} 1: and wrist’s translation along z-axis p e e 1: N d a t a p e e: 1 N d a t a p_{ee}^{1:N_{data}} 1: for trajectory.$ |
| N d a t a N d a t a N_{data} | Length of synthesis motion sequence. |
| N p r e d N p r e d N_{pred} | Length of predicted actions. |
| x p r e d x p r e d x^{pred} | Future motion predicted by PP-Tac policy. |
| N p r e f i x N p r e f i x N_{prefix} | Length of historical actions. |
| x p r e f i x x p r e f i x x^{prefix} | The historical action sequence. |
| t t t | Diffusion step. |

**说明**: TABLE II: Summary of symbols and notations.
## 实验解读

- 评价重点:围绕 closed-loop-control、contact-estimation、接触推理、灵巧操作、diffusion-policy,优先看真实机器人结果、扰动恢复、接触稳定性、控制误差、成功率、实时性和 sim-to-real 差距。
- System0 视角:实验结论不只看最终成功率,还要看闭环过程中是否能处理状态估计误差、执行延迟、碰撞/接触突变和外部扰动。
- 与上层系统的关系:如果方法能在快闭环内稳定工作,它可以作为 System1 技能或 System2 规划的底层执行保障;如果只在离线或慢速设定下成立,则还不能直接作为身体级快闭环模块。
## 批判性思考

### 优点

- 论文把 closed-loop-control、contact-estimation、接触推理、灵巧操作、diffusion-policy 放到机器人身体级闭环里讨论,和只做高层语义推理的工作相比,更接近真实机器人部署时会遇到的状态、动力学和安全约束。
- 若论文同时给出公式、系统结构图、硬件实验或消融结果,这些证据可以帮助判断方法是否真的能支撑毫秒到百毫秒级反应。

### 局限

- 需要重点核查控制频率、推理延迟、传感延迟、失败恢复、硬件磨损、未见场景泛化和安全边界;这些因素决定它能否成为可靠的 System0 模块。
- 对学习型方法,还要区分训练环境中的鲁棒性和真实部署中的鲁棒性,尤其是接触丰富任务、动态运动任务和人形机器人任务。
## 论文定位

- 研究对象:PP-Tac: Paper Picking Using Omnidirectional Tactile Feedback in Dexterous Robotic Hands。
- 关键词:closed-loop-control、contact-estimation、接触推理、灵巧操作、diffusion-policy、力控制、grasping、实时控制、机器人操作、鲁棒控制。
- System0 关联:关注机器人身体级快反应链路中“状态/接触/局部感知 -> 约束/优化/策略 -> 动作”的短闭环,不把问题停留在语言规划或任务分解层。

## 速查卡片

> [!summary] PP-Tac
> - **论文**: https://www.roboticsproceedings.org/rss21/p056.pdf
> - **arXiv**: http://arxiv.org/abs/2504.16649v2
> - **arXiv HTML**: https://arxiv.org/html/2504.16649v2
