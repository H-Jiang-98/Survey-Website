---
title: "DexEvolve: Evolutionary Optimization for Robust and Diverse Dexterous Grasp Synthesis"
method_name: "DexEvolve"
authors: ["René Zurbrügg"]
year: 2026
venue: "RSS"
tags: ["contact-reasoning", "robot-manipulation", "robust-control", "reinforcement-learning", "robot-generalization", "dexterous-manipulation"]
image_source: "online"
arxiv_html: "https://arxiv.org/html/2602.15201v1"
---
# DexEvolve
## 一句话总结

> DexEvolve: Evolutionary Optimization for Robust and Diverse Dexterous Grasp Synthesis 主要落在 [[接触推理]]、[[dexterous-grasping]]、[[灵巧操作]]、[[diffusion-policy]] 这一类问题上;从 System0 视角看,它的价值在于把身体状态、接触变化、局部感知或动力学约束接到机器人低层闭环里,让动作生成更稳定、安全、实时。
## 核心贡献

1. 围绕 **DexEvolve: Evolutionary Optimization for Robust and Diverse Dexterous Grasp Synthesis** 建立了一个与 接触推理、dexterous-grasping、灵巧操作、diffusion-policy、grasping、机器人操作 相关的机器人问题设定,重点不是高层语义规划,而是身体级闭环中的状态变化、接触变化或控制约束。
2. 方法层面的关键价值在于把感知、动力学、约束优化、策略学习或硬件反馈组织成可执行的低层动作生成机制,适合放进 System0 的快反应模块中理解。
3. 对系统集成而言,这类工作补上了 System1 技能选择与电机级控制之间的一层:它负责把上层目标变成动态可行、可恢复、可安全过滤的动作。
4. 若论文包含真实机器人、接触任务或仿真到真实迁移,核心应看它是否报告了闭环稳定性、扰动恢复、硬件延迟、失败模式和跨场景泛化。
## 问题背景

这篇论文可以放在 System0 的“具身快系统”背景下阅读:机器人需要在很短时间内根据本体状态、接触、力/触觉、局部视觉或动力学预测做出动作,而不是等待慢速规划器重新推理整个任务。接触推理、dexterous-grasping、灵巧操作、diffusion-policy、grasping、机器人操作、鲁棒控制 相关问题的难点通常不在“知道要做什么”,而在动作执行过程中状态会快速偏离、接触会突然改变、约束会实时收紧,低层系统必须及时修正。

### System0 关联

精读时优先看三件事:第一,论文使用哪些快速反馈量,例如关节状态、接触状态、力觉、触觉、深度或局部视觉;第二,反馈如何进入 MPC、WBC、策略网络、安全过滤器或恢复控制器;第三,输出是否直接影响机器人下一步动作,而不是只服务离线分析或高层任务规划。
## 方法详解

### System0 视角解析

从 System0 角度看,这篇论文的方法部分应当围绕 接触推理、dexterous-grasping、灵巧操作、diffusion-policy、grasping、机器人操作、鲁棒控制、scalable-robot-learning 来读。核心问题是:系统如何从快速变化的机器人状态中构造控制输入,如何处理接触、碰撞、约束、模型误差或执行延迟,以及如何把这些信息变成下一时刻的动作。

如果方法里有优化问题,重点看目标函数、约束和求解频率;如果方法里有学习策略,重点看观测空间、动作空间、训练扰动和部署时的闭环频率;如果方法里有硬件或传感器,重点看传感链路是否真的进入控制回路。这样读可以把论文和 System2 语言规划、System1 技能编排区分开来。

### 关键公式与机制

#### 公式 1: [[优化目标/约束]]

$$\mathcal{F}_{i}^{\prime}=\mathcal{F}(G_{i})\cdot\left(\sum_{G_{j}\in\mathcal{B}_{r}(G_{i})}\left(1-\frac{d(G_{i},G_{j})^{p}}{r^{p}}\right)\right)^{-1},$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 2: [[优化目标/约束]]

$${\Delta q}_{cmd}\leftarrow\operatorname*{arg\,min}_{\Delta\tilde{q}}\lVert J(\mathcal{C}^{\prime})\,\Delta\tilde{q}-N_{o}(\mathcal{C}^{\prime})\rVert_{2}$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 3: [[优化目标/约束]]

$$\mathcal{G}_{\text{succ}}\leftarrow\{G\in\mathcal{P}:E_{\text{lifetime}}(G)\geq E_{\min}\}$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 4: [[优化目标/约束]]

$$
\mathcal{L}_{\text{total}}=\lambda_{\epsilon}\mathcal{L}_{\epsilon}+\lambda_{k}\mathcal{L}_{k}+\lambda_{p}\mathcal{L}_{p}\cdot e^{-\frac{t}{T_{\text{max}}/2}}
$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 5: [[优化目标/约束]]

$$\hat{\boldsymbol{\epsilon}}_{\theta}\leftarrow f_{\theta}(\mathbf{x}_{t}^{\text{norm}},t;\mathcal{P}_{\text{obs}})$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 6: [[动力学或策略机制]]

$$\hat{\mathbf{x}}_{0}^{\text{norm}}\leftarrow\frac{1}{\sqrt{\bar{\alpha}_{t}}}\mathbf{x}_{t}^{\text{norm}}-\frac{\sqrt{1-\bar{\alpha}_{t}}}{\sqrt{\bar{\alpha}_{t}}}\hat{\boldsymbol{\epsilon}}_{\theta}$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 7: [[优化目标/约束]]

$$\mathcal{L}_{p}\leftarrow\sum_{i<| $\mathcal{P}_{\text{gt}}$ |}\left[\max(0,\mathbf{d_{i}})\cdot\exp\left(-\frac{t}{T/2}\right)\right]$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 8: [[优化目标/约束]]

$$
\mathcal{L}_{k}\leftarrow\frac{1}{K}\sum_{j=1}^{K}\lVert \hat{\mathbf{k}}_{j}-\mathbf{k}_{\text{gt},j} \rVert_{2}
$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 9: [[优化目标/约束]]

$$
\mathcal{L}_{\text{sim}}=\mathbb{E}\!\left[(R_{\psi}(G_{2},\mathcal{P})-R_{\psi}(G_{1},\mathcal{P}))^{2}\right].
$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 10: [[动力学或策略机制]]

$$=\lambda_{\text{pos}}\left\lVert G_{\text{pos}}-G^{\prime}_{\text{pos}}\right \rVert_{2}^{2}+\lambda_{\text{orient}}\left\lVert G_{\text{orient}}-G^{\prime}_{\text{orient}}\right \rVert_{2}^{2}$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。
## 关键图表

### Figure 1: Pipeline / core system figure: Visualization of Generated Grasps using our Evolutionary Refinement visualized for tw

![Figure 1](https://arxiv.org/html/2602.15201v1/data/figures/teaser.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Pipeline / core system figure: Visualization of Generated Grasps using our Evolutionary Refinement visualized for tw”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 2: Method Overview. Given object meshes, we first generate an initial set of analytical

![Figure 2](https://arxiv.org/html/2602.15201v1/x1.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Method Overview. Given object meshes, we first generate an initial set of analytical”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 3: Real-world point cloud reconstructions. We capture depth data for objects used in ha

![Figure 3](https://arxiv.org/html/2602.15201v1/data/figures/pcs.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Real-world point cloud reconstructions. We capture depth data for objects used in ha”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### 论文表格

#### Table 1: Table 1

| | j ⋆ = arg  min j  d  (G, G j), j^{\star}\;=\; $\operatorname*{arg\,min}_{j}d(G,G_{j}),$ | | (1) |
| --- | --- | --- | --- |

**说明**: Table 1

#### Table 2: Table 2

| | d  (G, G ′) \displaystyle d(G,G^{\prime}) | pos G pos - G pos ′ orient G orient - G orient ′ \displaystyle \lambda $\text{pos}}\left\lVert G_{\text{pos}}-G^{\prime}_{\text{pos}}\right \rVert_{2}^{2}+\lambda_{\text{orient}}\left\lVert G_{\text{orient}}-G^{\prime}_{\text{orient}}\right \rVert_{2}^{2}$ | | (2) |
| --- | --- | --- | --- | --- |
| | | joints G joints - G joints ′ . \displaystyle\quad \lambda $\text{joints}}\left\lVert G_{\text{joints}}-G^{\prime}_{\text{joints}}\right \rVert_{2}^{2}.$ | | |

**说明**: Table 2

#### Table 3: Table 3

| | G j ⋆ ← G if F  (G) > F  (G j ⋆). G_{j^{\star}} $\leftarrow G\quad\text{if}\quad\mathcal{F}(G)>\mathcal{F}(G_{j^{\star}}).$ | | (3) |
| --- | --- | --- | --- |

**说明**: Table 3

#### Table 4: Table 4

| | F  (G) = E l  i  f  e  t  i  m  e - w d  i  s  E d  i  s - w p  e  n  E p  e  n,  $\mathcal{F}(G)={E}_{lifetime}-w_{dis}{E}_{dis}-w_{pen}{E}_{pen},$ | | (4) |
| --- | --- | --- | --- |

**说明**: Table 4

#### Table 5: Table 5

| | $\mathcal{F}_{i}^{\prime}=\mathcal{F}(G_{i})\cdot\left(\sum_{G_{j}\in\mathcal{B}_{r}(G_{i})}\left(1-\frac{d(G_{i},G_{j})^{p}}{r^{p}}\right)\right)^{-1},$ | | (5) |
| --- | --- | --- | --- |

**说明**: Table 5

#### Table 6: Table 6

| | $\operatorname*{arg\,min}_{\Delta\tilde{q}}\lVert \rVertJ\cdot\Delta\tilde{q}-{N_{o}}\lVert \rVert_{2},$ | | (6) |
| --- | --- | --- | --- |

**说明**: Table 6

#### Table 7: Table 7

| | d  (G, G ′) \displaystyle d(G,G^{\prime}) | mean G - G ′ \displaystyle \tfrac $\sqrt{\mathrm{mean}\!\left((\phi(G)_{1:7}-\phi(G^{\prime})_{1:7})^{2}\right)}$ | | (7) |
| --- | --- | --- | --- | --- |
| | | mean G - G ′ . \displaystyle\quad \tfrac $\sqrt{\mathrm{mean}\!\left((\phi(G)_{8:}-\phi(G^{\prime})_{8:})^{2}\right)}.$ | | |

**说明**: Table 7

#### Table 8: Table 8

| | $\text{pen}}=\sum_{j}\max\bigl(0,\;\delta-p_{j}\bigr)+\sum_{i}\max\bigl(0,\;\delta-s_{i}\bigr),$ | | (8) |
| --- | --- | --- | --- |

**说明**: Table 8

#### Table 9: Table 9

| | x = [x, y, z, r 11, r 21, r 31, r 12, r 22, r 32, q 1, ..., q 12 ].  $\mathbf{x}=[x,y,z,\;r_{11},r_{21},r_{31},\;r_{12},r_{22},r_{32},\;q_{1},\ldots,q_{12}].$ | | (9) |
| --- | --- | --- | --- |

**说明**: Table 9

#### Table 10: Table 10

| | $\frac{t-1}{T-1}(\beta_{\max}-\beta_{\min}),\quad t\in\{1,\ldots,T\},$ | | (10) |
| --- | --- | --- | --- |

**说明**: Table 10

#### Table 11: Table 11

| | $\mathcal{L}_{\text{pref}}=-\log\sigma\!\bigl(R_{\psi}(G_{a},\mathcal{P})-R_{\psi}(G_{b},\mathcal{P})\bigr).$ | |
| --- | --- | --- |

**说明**: Table 11

#### Table 12: Table 12

| | $\mathcal{L}_{\text{sim}}=\mathbb{E}\!\left[(R_{\psi}(G_{2},\mathcal{P})-R_{\psi}(G_{1},\mathcal{P}))^{2}\right].$ | |
| --- | --- | --- |

**说明**: Table 12

#### Table 13: Table 13

| | F  (G) = w lifetime  E lifetime - w pen  E pen - w dis  E dis + w reward  E reward.  $\mathcal{F}(G)=w_{\text{lifetime}}E_{\text{lifetime}}-w_{\text{pen}}E_{\text{pen}}-w_{\text{dis}}E_{\text{dis}}+w_{\text{reward}}E_{\text{reward}}.$ | |
| --- | --- | --- |

**说明**: Table 13
## 实验解读

- 评价重点:围绕 接触推理、dexterous-grasping、灵巧操作、diffusion-policy、grasping,优先看真实机器人结果、扰动恢复、接触稳定性、控制误差、成功率、实时性和 sim-to-real 差距。
- System0 视角:实验结论不只看最终成功率,还要看闭环过程中是否能处理状态估计误差、执行延迟、碰撞/接触突变和外部扰动。
- 与上层系统的关系:如果方法能在快闭环内稳定工作,它可以作为 System1 技能或 System2 规划的底层执行保障;如果只在离线或慢速设定下成立,则还不能直接作为身体级快闭环模块。
## 批判性思考

### 优点

- 论文把 接触推理、dexterous-grasping、灵巧操作、diffusion-policy、grasping 放到机器人身体级闭环里讨论,和只做高层语义推理的工作相比,更接近真实机器人部署时会遇到的状态、动力学和安全约束。
- 若论文同时给出公式、系统结构图、硬件实验或消融结果,这些证据可以帮助判断方法是否真的能支撑毫秒到百毫秒级反应。

### 局限

- 需要重点核查控制频率、推理延迟、传感延迟、失败恢复、硬件磨损、未见场景泛化和安全边界;这些因素决定它能否成为可靠的 System0 模块。
- 对学习型方法,还要区分训练环境中的鲁棒性和真实部署中的鲁棒性,尤其是接触丰富任务、动态运动任务和人形机器人任务。
## 论文定位

- 研究对象:DexEvolve: Evolutionary Optimization for Robust and Diverse Dexterous Grasp Synthesis。
- 关键词:接触推理、dexterous-grasping、灵巧操作、diffusion-policy、grasping、机器人操作、鲁棒控制、scalable-robot-learning。
- System0 关联:关注机器人身体级快反应链路中“状态/接触/局部感知 -> 约束/优化/策略 -> 动作”的短闭环,不把问题停留在语言规划或任务分解层。

## 速查卡片

> [!summary] DexEvolve
> - **论文**: https://www.roboticsproceedings.org/rss22/p059.pdf
> - **arXiv**: http://arxiv.org/abs/2602.15201v1
> - **arXiv HTML**: https://arxiv.org/html/2602.15201v1
