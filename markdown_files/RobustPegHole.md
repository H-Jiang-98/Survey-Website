---
title: "Robust Peg-in-Hole Assembly under Uncertainties via Compliant and Interactive Contact-Rich Manipulation"
method_name: "Robust Peg Hole"
authors: ["Yiting Chen"]
year: 2025
venue: "RSS"
tags: ["contact-reasoning", "robot-manipulation", "robust-control", "adaptive-control", "contact-rich-manipulation", "robot-generalization", "collision-avoidance"]
image_source: "online"
arxiv_html: "https://arxiv.org/html/2506.22766v1"
---
# Robust Peg Hole
## 一句话总结

> Robust Peg-in-Hole Assembly under Uncertainties via Compliant and Interactive Contact-Rich Manipulation 主要落在 [[adaptive-control]]、[[assembly]]、[[碰撞避免]]、[[compliance-control]] 这一类问题上;从 System0 视角看,它的价值在于把身体状态、接触变化、局部感知或动力学约束接到机器人低层闭环里,让动作生成更稳定、安全、实时。
## 核心贡献

1. 围绕 **Robust Peg-in-Hole Assembly under Uncertainties via Compliant and Interactive Contact-Rich Manipulation** 建立了一个与 adaptive-control、assembly、碰撞避免、compliance-control、接触推理、接触丰富操作 相关的机器人问题设定,重点不是高层语义规划,而是身体级闭环中的状态变化、接触变化或控制约束。
2. 方法层面的关键价值在于把感知、动力学、约束优化、策略学习或硬件反馈组织成可执行的低层动作生成机制,适合放进 System0 的快反应模块中理解。
3. 对系统集成而言,这类工作补上了 System1 技能选择与电机级控制之间的一层:它负责把上层目标变成动态可行、可恢复、可安全过滤的动作。
4. 若论文包含真实机器人、接触任务或仿真到真实迁移,核心应看它是否报告了闭环稳定性、扰动恢复、硬件延迟、失败模式和跨场景泛化。
## 问题背景

这篇论文可以放在 System0 的“具身快系统”背景下阅读:机器人需要在很短时间内根据本体状态、接触、力/触觉、局部视觉或动力学预测做出动作,而不是等待慢速规划器重新推理整个任务。adaptive-control、assembly、碰撞避免、compliance-control、接触推理、接触丰富操作、robot-generalization 相关问题的难点通常不在“知道要做什么”,而在动作执行过程中状态会快速偏离、接触会突然改变、约束会实时收紧,低层系统必须及时修正。

### System0 关联

精读时优先看三件事:第一,论文使用哪些快速反馈量,例如关节状态、接触状态、力觉、触觉、深度或局部视觉;第二,反馈如何进入 MPC、WBC、策略网络、安全过滤器或恢复控制器;第三,输出是否直接影响机器人下一步动作,而不是只服务离线分析或高层任务规划。
## 方法详解

### System0 视角解析

从 System0 角度看,这篇论文的方法部分应当围绕 adaptive-control、assembly、碰撞避免、compliance-control、接触推理、接触丰富操作、robot-generalization、机器人操作 来读。核心问题是:系统如何从快速变化的机器人状态中构造控制输入,如何处理接触、碰撞、约束、模型误差或执行延迟,以及如何把这些信息变成下一时刻的动作。

如果方法里有优化问题,重点看目标函数、约束和求解频率;如果方法里有学习策略,重点看观测空间、动作空间、训练扰动和部署时的闭环频率;如果方法里有硬件或传感器,重点看传感链路是否真的进入控制回路。这样读可以把论文和 System2 语言规划、System1 技能编排区分开来。

### 关键公式与机制

#### 公式 1: [[优化目标/约束]]

$$\Gamma(x,y,z)=\left\{\begin{array}{lr}-1,&z 0\\ 1,&z>0\lor(z\leq 0\land f_{T_{\{OH\}}\mathcal{A}_{h}}(x,y)<0)\\ 0,&\text{Otherwise}\\ \end{array}\right.$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 2: [[优化目标/约束]]

$$\mathbf{g}^{*}=\arg\max_{\mathbf{g}\in\mathcal{G}}\Delta\mathcal{H}(\mathbf{g})$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 3: [[优化目标/约束]]

$$g_{t}(\widetilde{T}_{\{OH\}})=\left\{\begin{array}{lr}f_{\widetilde{T}_{\{OH\}}\mathcal{A}_{h}}(p_{o}),| $\mathcal{P}_{o,\text{footprint},t}$ |=1&\\ -\max f_{\widetilde{T}_{\{OH\}}\mathcal{A}_{h}}(p_{o}),\text{Otherwise}&\end{array}\right.$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 4: [[优化目标/约束]]

$$=\arg\min\sum_{i=0}^{T-1}(\mathcal{J}(\Pi(\mathbf{x}_{t+i},\mathbf{x}_{t+i}^{\text{d}}),\mathbf{x}^{*})+\mathbf{u}_{t+i})$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 5: [[动力学或策略机制]]

$$\Delta\mathbf{s}=\ {\text{translation deviation:}\Delta v}{\left\lVert v_{\text{lateral}}-T_{\{OH\}}v_{h}^{\text{corner}}\right \rVert}+\ {\text{inclined deviation:}\Delta\theta}{| $\theta^{\text{corner}}-\theta_{\text{lateral}}$ |}$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 6: [[优化目标/约束]]

$$h(\widetilde{T}_{\{OH\}},\mathbf{g})=\left\{\begin{array}{lr}1,&\mathbf{g}\in\widetilde{T}_{\{OH\}}\mathcal{A}_{h}\\ 0,&\text{Otherwise}\end{array}\right.$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 7: [[优化目标/约束]]

$$P_{t}(\widetilde{T}_{\{OH\}})=\left\{\begin{array}{lr}\frac{1}{| $\mathcal{X}_{t}$ |},&\widetilde{T}_{\{OH\}}\in\mathcal{X}_{t}\\ 0,&\widetilde{T}_{\{OH\}}\notin\mathcal{X}_{t}\end{array}\right.$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 8: [[优化目标/约束]]

$$\mathcal{A}_{\text{basin}}^{v_{d}}=\{v\in\mathbb{R}^{3}\mid\angle v_{i}v_{d}v\leq\angle v_{i}v_{d}v_{k}\cap\angle v_{i}v_{j}v\leq\angle v_{i}v_{j}v_{k}\}$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 9: [[优化目标/约束]]

$$\mathcal{A}_{\text{well}}=\{v\in\mathbb{R}^{3}\mid\frac{\pi}{2}\leq\angle v_{i}v_{j}v\leq(\frac{3\pi}{2}-\angle v_{i}v_{j}v_{k})\}$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 10: [[动力学或策略机制]]

$$\mathbf{M}(\mathbf{q}_{t})\ddot{\mathbf{q}_{t}}+\mathbf{C}(\mathbf{q}_{t},\dot{\mathbf{q}}_{t})\dot{\mathbf{q}}_{t}+\mathbf{g}(\mathbf{q}_{t})=\mathbf{J}(\mathbf{q}_{t})^{\top}\mathbf{F}$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。
## 关键图表

### Figure 1: Pipeline / core system figure: a) The peg-in-hole problem is considered as inserting a peg into its matching hole on

![Figure 1](https://arxiv.org/html/2506.22766v1/x2.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Pipeline / core system figure: a) The peg-in-hole problem is considered as inserting a peg into its matching hole on”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 2: a) Overview of the System Setup; (b) Ablation study on the perception manipulation fu

![Figure 2](https://arxiv.org/html/2506.22766v1/x7.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“a) Overview of the System Setup; (b) Ablation study on the perception manipulation fu”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 3: Overview of the Peg-in-Hole Tasks in Real-world Experiments

![Figure 3](https://arxiv.org/html/2506.22766v1/x8.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Overview of the Peg-in-Hole Tasks in Real-world Experiments”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### 论文表格

#### Table 1: TABLE I: Experimental Result of Real-world Insertion Tasks

| Name | Clearance | Position-based 1 | Funnel-based 2 |
| --- | --- | --- | --- |
| Round 8 | ∼ 0.8 similar-to absent 0.8 \sim 0.8 ∼ 0.8 | 4/10 | 10/10 |
| Round 12 | ∼ 0.8 similar-to absent 0.8 \sim 0.8 ∼ 0.8 | 5/10 | 10/10 |
| Round 16 | ∼ 0.8 similar-to absent 0.8 \sim 0.8 ∼ 0.8 | 5/10 | 10/10 |
| Rectangle 8 | ∼ 0.6 similar-to absent 0.6 \sim 0.6 ∼ 0.6 | 3/10 | 10/10 |
| Rectangle 12 | ∼ 0.7 similar-to absent 0.7 \sim 0.7 ∼ 0.7 | 5/10 | 10/10 |
| Rectangle 16 | ∼ 0.8 similar-to absent 0.8 \sim 0.8 ∼ 0.8 | 4/10 | 10/10 |
| Random #1 | ∼ 0.4 similar-to absent 0.4 \sim 0.4 ∼ 0.4 | 1/10 | 9/10 |
| Random #2 | ∼ 0.4 similar-to absent 0.4 \sim 0.4 ∼ 0.4 | 0/10 | 8/10 |
| Random #3 | ∼ 0.4 similar-to absent 0.4 \sim 0.4 ∼ 0.4 | 0/10 | 10/10 |
| Average | \ \ \backslash \ | 3.0/10 | 9.7/10 |

**说明**: TABLE I: Experimental Result of Real-world Insertion Tasks

#### Table 2: TABLE II: System Performance Different Prior Knowledge on the Target Hole

| | Partially Inside the Hole | Bounded Area | Overall | | | | |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Object | Success | Interactions ($\pm$ std) | Uncertainty($\pm$ std) | Success | Interactions ($\pm$ std) | Uncertainty($\pm$ std) | Success |
| Round 8 | 5 / 5 | 4.0 $\pm$ 0.7 | 0.152 $\pm$ 0.029 | 5/5 | 4.6 $\pm$ 1.5 | 0.148 $\pm$ 0.040 | 10/10 |
| Round 12 | 5 / 5 | 4.4 $\pm$ 1.1 | 0.155 $\pm$ 0.048 | 5/5 | 5.0 $\pm$ 1.7 | 0.150 $\pm$ 0.036 | 10/10 |
| Round 16 | 5 / 5 | 4.6 $\pm$ 0.9 | 0.163 $\pm$ 0.060 | 5/5 | 5.6 $\pm$ 1.5 | 0.162 $\pm$ 0.063 | 10/10 |
| Rectangle 8 | 5 / 5 | 6.8 $\pm$ 1.6 | 0.178 $\pm$ 0.031 | 4/5 | 7.8 $\pm$ 1.9 | 0.174 $\pm$ 0.043 | 9/10 |
| Rectangle 12 | 5 / 5 | 6.8 $\pm$ 1.3 | 0.174 $\pm$ 0.046 | 5/5 | 7.6 $\pm$ 1.5 | 0.170 $\pm$ 0.033 | 10/10 |
| Rectangle 16 | 5 / 5 | 7.2 $\pm$ 0.8 | 0.178 $\pm$ 0.031 | 5/5 | 7.8 $\pm$ 2.1 | 0.193 $\pm$ 0.055 | 10/10 |
| Random #1 | 5 / 5 | 9.2 $\pm$ 2.2 | 0.172 $\pm$ 0.029 | 4/5 | 9.0 $\pm$ 2.9 | 0.158 $\pm$ 0.036 | 9/10 |
| Random #2 | 4 / 5 | 9.4 $\pm$ 2.7 | 0.164 $\pm$ 0.062 | 4/5 | 10.2 $\pm$ 1.9 | 0.181 $\pm$ 0.065 | 8/10 |
| Random #3 | 4 / 5 | 9.0 $\pm$ 2.9 | 0.170 $\pm$ 0.045 | 5/5 | 9.6 $\pm$ 1.8 | 0.178 $\pm$ 0.061 | 9/10 |
| Average | 4.78 / 5 | 6.8 $\pm$ 2.6 | 0.167 $\pm$ 0.041 | 4.67/5 | 7.5 $\pm$ 2.6 | 0.168 $\pm$ 0.046 | 9.44/10 |

**说明**: TABLE II: System Performance Different Prior Knowledge on the Target Hole
## 实验解读

- 评价重点:围绕 adaptive-control、assembly、碰撞避免、compliance-control、接触推理,优先看真实机器人结果、扰动恢复、接触稳定性、控制误差、成功率、实时性和 sim-to-real 差距。
- System0 视角:实验结论不只看最终成功率,还要看闭环过程中是否能处理状态估计误差、执行延迟、碰撞/接触突变和外部扰动。
- 与上层系统的关系:如果方法能在快闭环内稳定工作,它可以作为 System1 技能或 System2 规划的底层执行保障;如果只在离线或慢速设定下成立,则还不能直接作为身体级快闭环模块。
## 批判性思考

### 优点

- 论文把 adaptive-control、assembly、碰撞避免、compliance-control、接触推理 放到机器人身体级闭环里讨论,和只做高层语义推理的工作相比,更接近真实机器人部署时会遇到的状态、动力学和安全约束。
- 若论文同时给出公式、系统结构图、硬件实验或消融结果,这些证据可以帮助判断方法是否真的能支撑毫秒到百毫秒级反应。

### 局限

- 需要重点核查控制频率、推理延迟、传感延迟、失败恢复、硬件磨损、未见场景泛化和安全边界;这些因素决定它能否成为可靠的 System0 模块。
- 对学习型方法,还要区分训练环境中的鲁棒性和真实部署中的鲁棒性,尤其是接触丰富任务、动态运动任务和人形机器人任务。
## 论文定位

- 研究对象:Robust Peg-in-Hole Assembly under Uncertainties via Compliant and Interactive Contact-Rich Manipulation。
- 关键词:adaptive-control、assembly、碰撞避免、compliance-control、接触推理、接触丰富操作、robot-generalization、机器人操作、鲁棒控制。
- System0 关联:关注机器人身体级快反应链路中“状态/接触/局部感知 -> 约束/优化/策略 -> 动作”的短闭环,不把问题停留在语言规划或任务分解层。

## 速查卡片

> [!summary] Robust Peg Hole
> - **论文**: https://www.roboticsproceedings.org/rss21/p060.pdf
> - **arXiv**: http://arxiv.org/abs/2506.22766v1
> - **arXiv HTML**: https://arxiv.org/html/2506.22766v1
