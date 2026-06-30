---
title: "Latent Diffeomorphic Co-Design of End-Effectors for Deformable and Fragile Object Manipulation"
method_name: "Latent Diffeomorphic Co"
authors: ["Kei Ikemura"]
year: 2026
venue: "RSS"
tags: ["contact-reasoning", "robot-manipulation", "contact-rich-manipulation", "robot-generalization", "agile-locomotion"]
image_source: "online"
arxiv_html: "https://arxiv.org/html/2602.17921v1"
---
# Latent Diffeomorphic Co
## 一句话总结

> Latent Diffeomorphic Co-Design of End-Effectors for Deformable and Fragile Object Manipulation 主要落在 [[agile-locomotion]]、[[接触推理]]、[[grasping]]、[[non-prehensile-manipulation]] 这一类问题上;从 System0 视角看,它的价值在于把身体状态、接触变化、局部感知或动力学约束接到机器人低层闭环里,让动作生成更稳定、安全、实时。
## 核心贡献

1. 围绕 **Latent Diffeomorphic Co-Design of End-Effectors for Deformable and Fragile Object Manipulation** 建立了一个与 agile-locomotion、接触推理、grasping、non-prehensile-manipulation、机器人操作、zero-shot 相关的机器人问题设定,重点不是高层语义规划,而是身体级闭环中的状态变化、接触变化或控制约束。
2. 方法层面的关键价值在于把感知、动力学、约束优化、策略学习或硬件反馈组织成可执行的低层动作生成机制,适合放进 System0 的快反应模块中理解。
3. 对系统集成而言,这类工作补上了 System1 技能选择与电机级控制之间的一层:它负责把上层目标变成动态可行、可恢复、可安全过滤的动作。
4. 若论文包含真实机器人、接触任务或仿真到真实迁移,核心应看它是否报告了闭环稳定性、扰动恢复、硬件延迟、失败模式和跨场景泛化。
## 问题背景

这篇论文可以放在 System0 的“具身快系统”背景下阅读:机器人需要在很短时间内根据本体状态、接触、力/触觉、局部视觉或动力学预测做出动作,而不是等待慢速规划器重新推理整个任务。agile-locomotion、接触推理、grasping、non-prehensile-manipulation、机器人操作、zero-shot 相关问题的难点通常不在“知道要做什么”,而在动作执行过程中状态会快速偏离、接触会突然改变、约束会实时收紧,低层系统必须及时修正。

### System0 关联

精读时优先看三件事:第一,论文使用哪些快速反馈量,例如关节状态、接触状态、力觉、触觉、深度或局部视觉;第二,反馈如何进入 MPC、WBC、策略网络、安全过滤器或恢复控制器;第三,输出是否直接影响机器人下一步动作,而不是只服务离线分析或高层任务规划。
## 方法详解

### System0 视角解析

从 System0 角度看,这篇论文的方法部分应当围绕 agile-locomotion、接触推理、grasping、non-prehensile-manipulation、机器人操作、zero-shot 来读。核心问题是:系统如何从快速变化的机器人状态中构造控制输入,如何处理接触、碰撞、约束、模型误差或执行延迟,以及如何把这些信息变成下一时刻的动作。

如果方法里有优化问题,重点看目标函数、约束和求解频率;如果方法里有学习策略,重点看观测空间、动作空间、训练扰动和部署时的闭环频率;如果方法里有硬件或传感器,重点看传感链路是否真的进入控制回路。这样读可以把论文和 System2 语言规划、System1 技能编排区分开来。

### 关键公式与机制

#### 公式 1: [[动力学或策略机制]]

$$v(x)=\sum_{i=1}^{K}\lambda_{i}\exp\!\left(-\frac{\lVert x-c_{i} \rVert^{2}}{2\sigma_{i}^{2}}\right),$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 2: [[优化目标/约束]]

$$p^{*}=\arg\max_{p}f(p;d,\xi)=\arg\max_{p}\sum_{m\in\mathcal{M}}\lambda_{m}\,f_{m}(p;\rho_{d},\phi_{\xi}),$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 3: [[动力学或策略机制]]

$$J_{e}(d)=\lambda_{\text{prog}}\,q_{\text{prog}}+\lambda_{\text{succ}}\,q_{\text{succ}}-\lambda_{\bar{\sigma}}\,\bar{\sigma}-\lambda_{\mathrm{max}}\,\sigma^{\mathrm{max}},$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 4: [[动力学或策略机制]]

$$\bar{\sigma}=P_{s}\!\left(\{\bar{\sigma}_{k}\}_{k=1}^{K}\right),\quad\sigma^{\mathrm{max}}=P_{s}\!\left(\{\sigma^{\mathrm{max}}_{k}\}_{k=1}^{K}\right).$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 5: [[优化目标/约束]]

$$\{p_{e,i}\}_{i=1}^{I}\leftarrow\text{CMA-ES}\!\left(f(\cdot;d_{t},\xi_{e}),\mathcal{X},B_{p}\right)$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 6: [[优化目标/约束]]

$$p^{\star}=\arg\max_{p}\;\;f_{\text{pen}}(p)+f_{\text{near}}(p)+\sum_{m\in\mathcal{M}_{\text{task}}}f_{m}(p).$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 7: [[动力学或策略机制]]

$$q_{\text{succ}}:=\mathbf{1}\!\left[z_{k}-z_{1}\geq\tau_{z},\ \forall k\in\{K-K_{s}+1,\dots,K\}\right],$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 8: [[动力学或策略机制]]

$$q_{\text{succ}}:=\mathbf{1}\left[l_{k}\leq\tau_{l},\ \forall k\in\{K-K_{s}+1,\dots,K\}\right],$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 9: [[优化目标/约束]]

$$=\arg\max_{d\in\mathcal{D}}\mathbb{E}_{\xi}\left[J_{\xi}(d,\pi^{*}_{d})\right],$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 10: [[优化目标/约束]]

$$=\arg\max_{\pi}\mathbb{E}_{\xi}\left[J_{\xi}(d,\pi)\right].$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。
## 关键图表

### Figure 1: Pipeline / core system figure: We jointly optimize end-effector morphology and motion-adaptive control to enable safe

![Figure 1](https://arxiv.org/html/2602.17921v1/x1.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Pipeline / core system figure: We jointly optimize end-effector morphology and motion-adaptive control to enable safe”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 2: Overview of the proposed co-design framework for deformable and fragile object manipul

![Figure 2](https://arxiv.org/html/2602.17921v1/x2.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Overview of the proposed co-design framework for deformable and fragile object manipul”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 3: a) Task-specific end-effector designs optimized, fabricated, and mounted on the robot

![Figure 3](https://arxiv.org/html/2602.17921v1/x7.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“a) Task-specific end-effector designs optimized, fabricated, and mounted on the robot”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### 论文表格

#### Table 1: TABLE I: Optimal design performance of the jelly grasping task

| | cylinder | onigiri | large cube | | | | | | |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| | score ↑ \uparrow | success ↑ \uparrow | stress ↓ \downarrow | score ↑ \uparrow | success ↑ \uparrow | stress ↓ \downarrow | score ↑ \uparrow | success ↑ \uparrow | stress ↓ \downarrow |
| Ours | 143 ± \pm 7 | 0.97 ± \pm 0.03 | 9274 ± \pm 124 | 64 ± \pm 18 | 0.73 ± \pm 0.08 | 13973 ± \pm 490 | 19 ± \pm 48 | 0.51 ± \pm 0.21 | 12050 ± \pm 1116 |
| PJ | 79 ± \pm 12 | 0.63 ± \pm 0.05 | 9366 ± \pm 120 | -33 ± \pm 3 | 0.12 ± \pm 0.04 | 10224 ± \pm 332 | -133 ± \pm 17 | 0.18 ± \pm 0.04 | 18893 ± \pm 620 |
| BO | 106 ± \pm 28 | 0.80 ± \pm 0.11 | 9998 ± \pm 1089 | 30 ± \pm 66 | 0.59 ± \pm 0.27 | 14946 ± \pm 1658 | 33 ± \pm 53 | 0.45 ± \pm 0.40 | 13189 ± \pm 1030 |
| RL | -110 ± \pm 62 | 0.04 ± \pm 0.02 | 22015 ± \pm 12105 | -244 ± \pm 97 | 0.09 ± \pm 0.06 | 38107 ± \pm 19327 | - | - | - |

**说明**: TABLE I: Optimal design performance of the jelly grasping task

#### Table 2: TABLE II: Optimal design performance of the jelly pushing task

| | cylinder | onigiri | | | | |
| --- | --- | --- | --- | --- | --- | --- |
| | score J J | success rate q succ q_{ $\text{succ}}$ | $\mathrm{max}}$ | score J J | success rate q succ q_{ $\text{succ}}$ | $\mathrm{max}}$ |
| Ours | 85 ± \pm 5 | 0.760 ± \pm 0.000 | 1236 ± \pm 108 | 70 ± \pm 16 | 0.627 ± \pm 0.076 | 1177 ± \pm 85 |
| BO | 63 ± \pm 16 | 0.633 ± \pm 0.064 | 1289 ± \pm 46 | 65 ± \pm 23 | 0.627 ± \pm 0.115 | 1295 ± \pm 64 |
| RL | -39 ± \pm 31 | 0.280 ± \pm 0.100 | 2432 ± \pm 522 | -81 ± \pm 33 | 0.287 ± \pm 0.163 | 5032 ± \pm 659 |

**说明**: TABLE II: Optimal design performance of the jelly pushing task

#### Table 3: TABLE III: Optimal design performance of the fillet scooping task

| | score J ↑ J\uparrow | success rate q succ ↑ q_{ $\text{succ}}\uparrow$ | $\mathrm{max}}\downarrow$ |
| --- | --- | --- | --- |
| Ours | 293 ± \pm 97 | 0.779 ± \pm 0.203 | 17154 ± \pm 469 |
| BO | 282 ± \pm 146 | 0.683 ± \pm 0.322 | 13586 ± \pm 2857 |

**说明**: TABLE III: Optimal design performance of the fillet scooping task

#### Table 4: TABLE IV: Comparison of design spaces

| | score J J | success rate q succ q_{ $\text{succ}}$ | $\mathrm{max}}$ |
| --- | --- | --- | --- |
| Ours | 64 ± \pm 18 | 0.727 ± \pm 0.081 | 13973 ± \pm 490 |
| Cubic | 10 ± \pm 17 | 0.473 ± \pm 0.083 | 14894 ± \pm 306 |
| Spherical | 13 ± \pm 23 | 0.540 ± \pm 0.106 | 15321 ± \pm 386 |

**说明**: TABLE IV: Comparison of design spaces

#### Table 5: TABLE V: Real-World Experiments

| Task | Method | success rate ↑ \uparrow | object damage ↓ \downarrow |
| --- | --- | --- | --- |
| Grasp | Ours | 9 /9 | 1 /9 |
| BO | 5/9 | 1 /9 | |
| Push | Ours | 10 /12 | 0 /12 |
| BO | 9/12 | 0 /12 | |

**说明**: TABLE V: Real-World Experiments

#### Table 6: TABLE VI: Comparison of co-design methods by design and action space dimensionality and paradigm.

| Paper | Design Space Dimension | Action Space Dimension | Co-Design Paradigm |
| --- | --- | --- | --- |
| SERL Co-Design [8 ] | 2 | 9 | Two-stage: RL + GA |
| Meta-RL Co-Design [4 ] | 4 | 16 | Two-stage: RL + CMA-ES |
| Object Adaptation [18 ] | 4 | 7 | Single-stage: Dual MDP |
| Object Adaptation [18 ] | 4 | 7 | Single-stage: Dual MDP |
| Cross-Embodiment Hand Co-Design [14 ] | 5 | 20 | Two-stage with Design-Conditioned Policy |
| CageCoOpt [12 ] | 6 | 4 | Two-stage: PPO + BO/GA |
| Evolving Robot Hand [51 ] | 8 | 26 | Two-stage: RL + Gradient-based Optimization |
| MORPH [22 ] | 10 | 11 | Single-stage RL + Neural Surrogate |
| Tool Design and Use [31 ] | 12 | 6 | Single-stage: PPO |
| HWasP [7 ] | 12 | 9 | Single-stage: TRPO |
| EvoGym [5 ] | 15 | 35 | Two-stage: PPO + BO/GA |
| Co-Design with CBD [49 ] | 17 | 8 | End-to-end Differentiable Co-Design |
| Soft Gripper Co-Design [53 ] | 22 | 8 | Neural-physics surrogate |
| NGE [43 ] | > 2 >2 | 10 | Two-stage with Evolving Controller |

**说明**: TABLE VI: Comparison of co-design methods by design and action space dimensionality and paradigm.

#### Table 7: TABLE VII: Design objective weights used for different tasks.

| Task | $\text{prog}}$ | $\text{succ}}$ | $\mathrm{max}}$ | $\mathrm{max}}$ |    ̄ \lambda_{\bar{\sigma}} |
| --- | --- | --- | --- | --- | --- |
| Grasping | 1  10 2 1\times 10^{2} | 2  10 2 2\times 10^{2} | 2  10 - 3 2\times 10^{-3} | 8  10 - 4 8\times 10^{-4} | 1  10 - 2 1\times 10^{-2} |
| Pushing | 3  10 2 3\times 10^{2} | 5  10 2 5\times 10^{2} | 2  10 - 3 2\times 10^{-3} | 8  10 - 4 8\times 10^{-4} | 1  10 - 2 1\times 10^{-2} |
| Scooping | 1  10 2 1\times 10^{2} | 2  10 2 2\times 10^{2} | 2  10 - 3 2\times 10^{-3} | 8  10 - 4 8\times 10^{-4} | 1  10 - 2 1\times 10^{-2} |

**说明**: TABLE VII: Design objective weights used for different tasks.
## 实验解读

- 评价重点:围绕 agile-locomotion、接触推理、grasping、non-prehensile-manipulation、机器人操作,优先看真实机器人结果、扰动恢复、接触稳定性、控制误差、成功率、实时性和 sim-to-real 差距。
- System0 视角:实验结论不只看最终成功率,还要看闭环过程中是否能处理状态估计误差、执行延迟、碰撞/接触突变和外部扰动。
- 与上层系统的关系:如果方法能在快闭环内稳定工作,它可以作为 System1 技能或 System2 规划的底层执行保障;如果只在离线或慢速设定下成立,则还不能直接作为身体级快闭环模块。
## 批判性思考

### 优点

- 论文把 agile-locomotion、接触推理、grasping、non-prehensile-manipulation、机器人操作 放到机器人身体级闭环里讨论,和只做高层语义推理的工作相比,更接近真实机器人部署时会遇到的状态、动力学和安全约束。
- 若论文同时给出公式、系统结构图、硬件实验或消融结果,这些证据可以帮助判断方法是否真的能支撑毫秒到百毫秒级反应。

### 局限

- 需要重点核查控制频率、推理延迟、传感延迟、失败恢复、硬件磨损、未见场景泛化和安全边界;这些因素决定它能否成为可靠的 System0 模块。
- 对学习型方法,还要区分训练环境中的鲁棒性和真实部署中的鲁棒性,尤其是接触丰富任务、动态运动任务和人形机器人任务。
## 论文定位

- 研究对象:Latent Diffeomorphic Co-Design of End-Effectors for Deformable and Fragile Object Manipulation。
- 关键词:agile-locomotion、接触推理、grasping、non-prehensile-manipulation、机器人操作、zero-shot。
- System0 关联:关注机器人身体级快反应链路中“状态/接触/局部感知 -> 约束/优化/策略 -> 动作”的短闭环,不把问题停留在语言规划或任务分解层。

## 速查卡片

> [!summary] Latent Diffeomorphic Co
> - **论文**: https://www.roboticsproceedings.org/rss22/p195.pdf
> - **arXiv**: http://arxiv.org/abs/2602.17921v1
> - **arXiv HTML**: https://arxiv.org/html/2602.17921v1
