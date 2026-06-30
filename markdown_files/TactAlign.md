---
title: "TactAlign: Human-to-Robot Policy Transfer via Tactile Alignment"
method_name: "TactAlign"
authors: ["Youngsun Wi"]
year: 2026
venue: "RSS"
tags: ["contact-reasoning", "robot-manipulation", "reinforcement-learning", "imitation-learning", "contact-rich-manipulation", "robot-generalization", "closed-loop-control", "dexterous-manipulation", "tactile-feedback"]
image_source: "online"
arxiv_html: "https://arxiv.org/html/2602.13579v1"
---
# TactAlign
## 一句话总结

> TactAlign: Human-to-Robot Policy Transfer via Tactile Alignment 主要落在 [[closed-loop-control]]、[[接触推理]]、[[接触丰富操作]]、[[cross-embodiment]] 这一类问题上;从 System0 视角看,它的价值在于把身体状态、接触变化、局部感知或动力学约束接到机器人低层闭环里,让动作生成更稳定、安全、实时。
## 核心贡献

1. 围绕 **TactAlign: Human-to-Robot Policy Transfer via Tactile Alignment** 建立了一个与 closed-loop-control、接触推理、接触丰富操作、cross-embodiment、灵巧操作、egocentric-perception 相关的机器人问题设定,重点不是高层语义规划,而是身体级闭环中的状态变化、接触变化或控制约束。
2. 方法层面的关键价值在于把感知、动力学、约束优化、策略学习或硬件反馈组织成可执行的低层动作生成机制,适合放进 System0 的快反应模块中理解。
3. 对系统集成而言,这类工作补上了 System1 技能选择与电机级控制之间的一层:它负责把上层目标变成动态可行、可恢复、可安全过滤的动作。
4. 若论文包含真实机器人、接触任务或仿真到真实迁移,核心应看它是否报告了闭环稳定性、扰动恢复、硬件延迟、失败模式和跨场景泛化。
## 问题背景

这篇论文可以放在 System0 的“具身快系统”背景下阅读:机器人需要在很短时间内根据本体状态、接触、力/触觉、局部视觉或动力学预测做出动作,而不是等待慢速规划器重新推理整个任务。closed-loop-control、接触推理、接触丰富操作、cross-embodiment、灵巧操作、egocentric-perception、模仿学习 相关问题的难点通常不在“知道要做什么”,而在动作执行过程中状态会快速偏离、接触会突然改变、约束会实时收紧,低层系统必须及时修正。

### System0 关联

精读时优先看三件事:第一,论文使用哪些快速反馈量,例如关节状态、接触状态、力觉、触觉、深度或局部视觉;第二,反馈如何进入 MPC、WBC、策略网络、安全过滤器或恢复控制器;第三,输出是否直接影响机器人下一步动作,而不是只服务离线分析或高层任务规划。
## 方法详解

### System0 视角解析

从 System0 角度看,这篇论文的方法部分应当围绕 closed-loop-control、接触推理、接触丰富操作、cross-embodiment、灵巧操作、egocentric-perception、模仿学习、policy-learning 来读。核心问题是:系统如何从快速变化的机器人状态中构造控制输入,如何处理接触、碰撞、约束、模型误差或执行延迟,以及如何把这些信息变成下一时刻的动作。

如果方法里有优化问题,重点看目标函数、约束和求解频率;如果方法里有学习策略,重点看观测空间、动作空间、训练扰动和部署时的闭环频率;如果方法里有硬件或传感器,重点看传感链路是否真的进入控制回路。这样读可以把论文和 System2 语言规划、System1 技能编排区分开来。

### 关键公式与机制

#### 公式 1: [[优化目标/约束]]

$$\min_{v_{\theta}}\sum_{(h_{i}^{\ast},r_{j}^{\ast})\in P}\int_{0}^{1}\left\lVert(h_{i}^{\ast}-r_{j}^{\ast})-{v_{\theta}}(x_{t},t)\right \rVert^{2}dt.$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 2: [[优化目标/约束]]

$$\mathcal{P}_{\tau}^{r}=\{p_{t,k}^{r}\mid(F_{t}^{r},P_{t}^{r},w_{t}^{r},o_{t}^{r})\in A_{\tau}^{r}\}$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 3: [[优化目标/约束]]

$$\mathcal{S}_{\tau}^{r}=\{s_{t}^{r}\mid(F_{t}^{r},P_{t}^{r},w_{t}^{r},o_{t}^{r})\in A_{\tau}^{r}\}.$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 4: [[优化目标/约束]]

$$\mu_{s}(\tau)=\frac{1}{| $\mathcal{S}_{\tau}^{r}$ |}\sum_{s\in\mathcal{S}_{\tau}^{r}}s,$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 5: [[优化目标/约束]]

$$\sigma_{s,\max}(\tau)=\max_{d\in\{x,y,z\}}\operatorname{std}\!\left(\{s_{d}:s\in\mathcal{S}_{\tau}^{r}\}\right).$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 6: [[优化目标/约束]]

$$\mathcal{Q}_{\tau}^{r}=\{q_{t}^{r}\mid(F_{t}^{r},P_{t}^{r},w_{t}^{r},o_{t}^{r})\in A_{\tau}^{r}\}.$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 7: [[优化目标/约束]]

$$\mu_{q}(\tau)=\frac{1}{| $\mathcal{Q}_{\tau}^{r}$ |}\sum_{q\in\mathcal{Q}_{\tau}^{r}}q,$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 8: [[优化目标/约束]]

$$\sigma_{q,\max}(\tau)=\max_{d\in\{x,y,z\}}\operatorname{std}\!\left(\{q_{d}:q\in\mathcal{Q}_{\tau}^{r}\}\right).$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 9: [[优化目标/约束]]

$$\sigma_{p,\max}(\tau)=\max_{d\in\{x,y,z\}}\operatorname{std}\!\left(\{p_{d}:p\in\mathcal{P}_{\tau}^{r}\}\right).$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 10: [[优化目标/约束]]

$$\mu_{p}(\tau)=\frac{1}{| $\mathcal{P}_{\tau}^{r}$ |}\sum_{p\in\mathcal{P}_{\tau}^{r}}p,$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。
## 关键图表

### Figure 1: Tactile Alignment Overview. Our method consists of two stages: self-supervised represe

![Figure 1](https://arxiv.org/html/2602.13579v1/x1.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Tactile Alignment Overview. Our method consists of two stages: self-supervised represe”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 2: Pivoting Task. The task begins in a non-contact state and transitions to pivoting upon

![Figure 2](https://arxiv.org/html/2602.13579v1/x5.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Pivoting Task. The task begins in a non-contact state and transitions to pivoting upon”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 3: l 1 $\ell_{1}$ force prediction error (mean ± $\pm$ std) along each axis, averaged f

![Figure 3](https://arxiv.org/html/2602.13579v1/x8.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“l 1 $\ell_{1}$ force prediction error (mean ± $\pm$ std) along each axis, averaged f”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### 论文表格

#### Table 1: TABLE I: H2R Co-training. We evaluate human-to-robot transfer on tasks used for alignment (pivoting and insertion) as we

| | | Seen for Alignment v v $\theta}, Seen-by-both only)$ | Unseen for Alignment v v $\theta})$ | | | | | | | |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Task | All | Pivoting | Insertion | Lid Closing | | | | | | |
| Method [%] ↑ \uparrow | Avg. ↑ \uparrow | Seen-by-both | Human-only | Unseen-by-both | Seen-by-both | Human-only | Unseen-by-both | Seen-by-both | Human-only | Unseen-by-both |
| Robot Only | 38 | 100 | 0 | 0 | 70 | 0 | 33 | 100 | 35 | 0 |
| TactAlign w/o Tactile | 21 | 0 | 0 | 0 | 50 | 35 | 43 | 20 | 0 | 40 |
| TactAlign w/o Align | 28 | 0 | 63 | 13 | 10 | 5 | 0 | 70 | 40 | 55 |
| TactAlign | 79 | 100 | 83 | 60 | 100 | 65 | 67 | 100 | 65 | 70 |

**说明**: TABLE I: H2R Co-training. We evaluate human-to-robot transfer on tasks used for alignment (pivoting and insertion) as well as on an unseen task (lid closing). Performance is reported across three object categories: seen-by-both, human-only, and held-out objects. Incorporating human data enables zero-shot generalization to objects not observed during robot training and beyond those present in the human dataset. We compare against three baselines: robot-only training, TactAlign without tactile input, and TactAlign without tactile alignment.

#### Table 2: TABLE II: H2R Co-training Results. Success rates (%) averaged 10 rollouts per object for pivoting, insertion, and l

| | Pivoting | | Seen-by-both | Human-only | Unseen-by-both | | | | |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| | Object | Avg. ↑ \uparrow | Cheezit | Z | Tissue | Realsense | Tape | Spray | CAN Box |
| Specs | Length [mm] | 149 | 210 | 140 | 130 | 146 | 100 | 200 | 120 |
| Weight [g] | 224 | 558 | 117 | 191 | 143 | 146 | 366 | 45 | |
| Methods ↑ \uparrow | Robot Only | 14 | 100 | 0 | 0 | 0 | 0 | 0 | 0 |
| Ours w/o Tactile | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | |
| Ours w/o Align | 34 | 0 | 80 | 80 | 30 | 40 | 10 | 0 | |
| Ours | 76 | 100 | 100 | 90 | 60 | 30 | 70 | 80 | |
| | Insertion | | Seen-by-both | Human-only | Unseen-by-both | | | | |
| | Object | Avg. ↑ \uparrow | RVP+ | Mac | Cana | Belkin | TKDY | FNRSi | |
| Specs | Height [mm] | 64 | 66 | 80 | 50 | 66 | 50 | 70 | |
| Weight [g] | 114 | 83 | 165 | 90 | 80 | 88 | 176 | | |
| Methods ↑ \uparrow | Robot Only | 28 | 70 | 0 | 0 | 50 | 10 | 40 | |
| Ours w/o Tactile | 42 | 50 | 20 | 50 | 40 | 50 | 40 | | |
| Ours w/o Align | 12 | 0 | 10 | 0 | 10 | 50 | 0 | | |
| Ours | 72 | 100 | 60 | 70 | 60 | 50 | 90 | | |
| | Lid Closing | | Seen-by-both | Human-only | Unseen-by-both | | | | |
| | Object | Avg. ↑ \uparrow | Haweek | Thermos | Cyxw | Energify | Haers | | |
| Specs | Height [mm] | 115 | 95 | 100 | 140 | 130 | 110 | | |
| Width [mm] | 95 | 100 | 85 | 95 | 93 | 100 | | | |
| Methods ↑ \uparrow | Robot Only | 34 | 100 | 70 | 0 | 0 | 0 | | |
| Ours w/o Tactile | 20 | 20 | 0 | 0 | 40 | 40 | | | |
| Ours w/o Align | 52 | 70 | 70 | 10 | 60 | 50 | | | |
| Ours | 74 | 100 | 90 | 40 | 70 | 70 | | | |

**说明**: TABLE II: H2R Co-training Results. Success rates (%) averaged 10 rollouts per object for pivoting, insertion, and lid closing. Avg. reports mean performance across all objects for each task. Each object is shown in Sec. A-E.

#### Table 3: TABLE III: Light Bulb Screwing Result. Success rates on the light bulb screwing task, evaluated 10 rollouts.

| Light Bulb Screwing | Ours w/o Tactile | Ours w/o Align | Ours |
| --- | --- | --- | --- |
| Success rate ↑ \uparrow [%] | 0 | 0 | 100 |

**说明**: TABLE III: Light Bulb Screwing Result. Success rates on the light bulb screwing task, evaluated 10 rollouts.

#### Table 4: TABLE IV: EMD reduction rate (EMD Red.) before and after alignment for different values of  $\lambda$. Higher values ind

|  \lambda | 0.8 | 0.9 | 1.0 | 1.1 | 1.2 |
| --- | --- | --- | --- | --- | --- |
| EMD Red. [%] ↑ \uparrow | 80.5 | 82.5 | 83.2 | 80.6 | 79.2 |

**说明**: 该表格来自论文中的公式、奖励项或实验设置；为避免 PDF 抽取噪声，网页中仅保留可渲染的核心符号和数值。

#### Table 5: TABLE V: EMD reduction rate (EMD Red.) before and after alignment for different values of  $\lambda$ for different  $\del$

| $\delta$ | 1.5 | 2.0 | 2.5 | 3.0 | 3.5 |
| --- | --- | --- | --- | --- | --- |
| EMD Red. [%] ↑ \uparrow | 82.7 | 83.2 | 83.9 | 80.7 | 83.9 |

**说明**: 该表格来自论文中的公式、奖励项或实验设置；为避免 PDF 抽取噪声，网页中仅保留可渲染的核心符号和数值。
## 实验解读

- 评价重点:围绕 closed-loop-control、接触推理、接触丰富操作、cross-embodiment、灵巧操作,优先看真实机器人结果、扰动恢复、接触稳定性、控制误差、成功率、实时性和 sim-to-real 差距。
- System0 视角:实验结论不只看最终成功率,还要看闭环过程中是否能处理状态估计误差、执行延迟、碰撞/接触突变和外部扰动。
- 与上层系统的关系:如果方法能在快闭环内稳定工作,它可以作为 System1 技能或 System2 规划的底层执行保障;如果只在离线或慢速设定下成立,则还不能直接作为身体级快闭环模块。
## 批判性思考

### 优点

- 论文把 closed-loop-control、接触推理、接触丰富操作、cross-embodiment、灵巧操作 放到机器人身体级闭环里讨论,和只做高层语义推理的工作相比,更接近真实机器人部署时会遇到的状态、动力学和安全约束。
- 若论文同时给出公式、系统结构图、硬件实验或消融结果,这些证据可以帮助判断方法是否真的能支撑毫秒到百毫秒级反应。

### 局限

- 需要重点核查控制频率、推理延迟、传感延迟、失败恢复、硬件磨损、未见场景泛化和安全边界;这些因素决定它能否成为可靠的 System0 模块。
- 对学习型方法,还要区分训练环境中的鲁棒性和真实部署中的鲁棒性,尤其是接触丰富任务、动态运动任务和人形机器人任务。
## 论文定位

- 研究对象:TactAlign: Human-to-Robot Policy Transfer via Tactile Alignment。
- 关键词:closed-loop-control、接触推理、接触丰富操作、cross-embodiment、灵巧操作、egocentric-perception、模仿学习、policy-learning、retargeting、robot-generalization。
- System0 关联:关注机器人身体级快反应链路中“状态/接触/局部感知 -> 约束/优化/策略 -> 动作”的短闭环,不把问题停留在语言规划或任务分解层。

## 速查卡片

> [!summary] TactAlign
> - **论文**: https://www.roboticsproceedings.org/rss22/p006.pdf
> - **arXiv**: http://arxiv.org/abs/2602.13579v1
> - **arXiv HTML**: https://arxiv.org/html/2602.13579v1
