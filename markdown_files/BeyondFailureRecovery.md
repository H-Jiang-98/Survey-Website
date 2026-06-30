---
title: "Beyond Failure Recovery: An Engagement-Aware Human-in-the-loop Framework for Robotic Systems"
method_name: "Beyond Failure Recovery"
authors: ["Jiaying Fang"]
year: 2026
venue: "RSS"
tags: ["robot-manipulation", "robust-control", "model-predictive-control", "recovery"]
image_source: "online"
arxiv_html: "https://arxiv.org/html/2606.18189v1"
---
# Beyond Failure Recovery
## 一句话总结

> Beyond Failure Recovery: An Engagement-Aware Human-in-the-loop Framework for Robotic Systems 主要落在 [[dynamics-modeling]]、[[模型预测控制]]、[[recovery]]、[[鲁棒控制]] 这一类问题上;从 System0 视角看,它的价值在于把身体状态、接触变化、局部感知或动力学约束接到机器人低层闭环里,让动作生成更稳定、安全、实时。
## 核心贡献

1. 围绕 **Beyond Failure Recovery: An Engagement-Aware Human-in-the-loop Framework for Robotic Systems** 建立了一个与 dynamics-modeling、模型预测控制、recovery、鲁棒控制、tool-manipulation 相关的机器人问题设定,重点不是高层语义规划,而是身体级闭环中的状态变化、接触变化或控制约束。
2. 方法层面的关键价值在于把感知、动力学、约束优化、策略学习或硬件反馈组织成可执行的低层动作生成机制,适合放进 System0 的快反应模块中理解。
3. 对系统集成而言,这类工作补上了 System1 技能选择与电机级控制之间的一层:它负责把上层目标变成动态可行、可恢复、可安全过滤的动作。
4. 若论文包含真实机器人、接触任务或仿真到真实迁移,核心应看它是否报告了闭环稳定性、扰动恢复、硬件延迟、失败模式和跨场景泛化。
## 问题背景

这篇论文可以放在 System0 的“具身快系统”背景下阅读:机器人需要在很短时间内根据本体状态、接触、力/触觉、局部视觉或动力学预测做出动作,而不是等待慢速规划器重新推理整个任务。dynamics-modeling、模型预测控制、recovery、鲁棒控制、tool-manipulation 相关问题的难点通常不在“知道要做什么”,而在动作执行过程中状态会快速偏离、接触会突然改变、约束会实时收紧,低层系统必须及时修正。

### System0 关联

精读时优先看三件事:第一,论文使用哪些快速反馈量,例如关节状态、接触状态、力觉、触觉、深度或局部视觉;第二,反馈如何进入 MPC、WBC、策略网络、安全过滤器或恢复控制器;第三,输出是否直接影响机器人下一步动作,而不是只服务离线分析或高层任务规划。
## 方法详解

### System0 视角解析

从 System0 角度看,这篇论文的方法部分应当围绕 dynamics-modeling、模型预测控制、recovery、鲁棒控制、tool-manipulation 来读。核心问题是:系统如何从快速变化的机器人状态中构造控制输入,如何处理接触、碰撞、约束、模型误差或执行延迟,以及如何把这些信息变成下一时刻的动作。

如果方法里有优化问题,重点看目标函数、约束和求解频率;如果方法里有学习策略,重点看观测空间、动作空间、训练扰动和部署时的闭环频率;如果方法里有硬件或传感器,重点看传感链路是否真的进入控制回路。这样读可以把论文和 System2 语言规划、System1 技能编排区分开来。

### 关键公式与机制

#### 公式 1: [[优化目标/约束]]

$$\mathcal{S}=\{\phi^{1},\phi^{2},\dots,\phi^{N}\}$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 2: [[优化目标/约束]]

$$c_{t}=\mathcal{C}(\phi^{i}_{t},\mathbf{o}_{t})\in[0,1]$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 3: [[优化目标/约束]]

$$\mathbf{0}=[0,0]\in\mathcal{Q}$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 4: [[动力学或策略机制]]

$$g_{t}=e^{-\lambda_{g}\delta_{t}}g_{t_{k}}+(1-e^{-\lambda_{g}\delta_{t}})k_{q_{rd}}q_{rd,t},$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 5: [[优化目标/约束]]

$$\mathbf{q}_{t}=[q_{d,t},q_{rd,t}]\in\mathcal{Q}$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 6: [[动力学或策略机制]]

$$w_{t}=\gamma w_{\mathrm{init}}+\sum_{j=0}^{H-1}{\mathbf{w}_{j}}^{T}\mathbf{q}_{t-j}+w_{0}$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 7: [[优化目标/约束]]

$$\mathcal{J}_{w}=\lambda_{w}\big[\max(0,\,w_{t+l}-\tau_{w})\big]^{2},$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 8: [[优化目标/约束]]

$$\mathbf{q}_{t+l}\in\begin{cases}\mathcal{Q}^{\mathrm{assist},\phi^{i}_{t+l}},&\text{if}\mathcal{C}(\phi^{i}_{t+l},\mathbf{o}_{t+l})\leq\tau_{c}\\ &\text{and}r^{\phi^{i}_{t+l}}_{t}=\tau_{r},\\ \mathcal{Q}^{\mathrm{feas},\phi^{i}_{t+l}}\cup\{\mathbf{0}\}\,&\text{otherwise}.\end{cases}$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 9: [[动力学或策略机制]]

$$\mathrm{Satisfaction}=\begin{cases}1-|g_{t}-g_{des}|,&\begin{aligned} &\text{if}w_{t}\leq\tau_{w}\text{and skill}\\ &\text{succeeds in $\tau_{r}$ retries},\\ \end{aligned}\\[6.0pt] 0,&\text{otherwise}.\end{cases}$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 10: [[优化目标/约束]]

$$\sum_{l=0}^{L-1}\Big((g_{\mathrm{des}}-g_{t+l})^{2}+\lambda_{w}\,\big[\max(0,\,w_{t+l}-\tau_{w})\big]^{2}\Big)$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。
## 关键图表

### Figure 1: E-MPC framework. E-MPC proactively issues user queries to regulate engagement g t g_

![Figure 1](https://arxiv.org/html/2606.18189v1/figures/pipeline_figure.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“E-MPC framework. E-MPC proactively issues user queries to regulate engagement g t g_”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 2: Effect of “Fake” Queries and Heuristic Baselines. (a) Satisfaction for three personas

![Figure 2](https://arxiv.org/html/2606.18189v1/x2.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Effect of “Fake” Queries and Heuristic Baselines. (a) Satisfaction for three personas”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 3: User Study Results. (a) Visualization of a rollout from the user study. The user ini

![Figure 3](https://arxiv.org/html/2606.18189v1/x4.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“User Study Results. (a) Visualization of a rollout from the user study. The user ini”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### 论文表格

#### Table 1: TABLE I: Performance of E-MPC and its ablations across three desired engagement levels (pre-query skill success rate p

| g d  e  s g_{des} | Mode | Engagement Acc. ↑ \uparrow | Workload Compliance (%) ↑ \uparrow | Satisfaction ↑ \uparrow |
| --- | --- | --- | --- | --- |
| 0.2 {0.2} | E-MPC | 0.910 ± 0.021 0.910\!\pm\!0.021 | 100.0 ± 0.000 100.0\!\pm\!0.000 | 0.933 ± 0.012  $\mathbf{0.933\!\pm\!0.012}$ |
| NoWorkloadCst | 0.910 ± 0.021 0.910\!\pm\!0.021 | 100.0 ± 0.000 100.0\!\pm\!0.000 | 0.933 ± 0.012  $\mathbf{0.933\!\pm\!0.012}$ | |
| NoEngagementTrack | 0.862 ± 0.013 0.862\!\pm\!0.013 | 100.0 ± 0.000 100.0\!\pm\!0.000 | 0.880 ± 0.013 0.880\!\pm\!0.013 | |
| 0.5 {0.5} | E-MPC | 0.917 ± 0.003 0.917\!\pm\!0.003 | 100.0 ± 0.000 100.0\!\pm\!0.000 | 0.942 ± 0.007  $\mathbf{0.942\!\pm\!0.007}$ |
| NoWorkloadCst | 0.919 ± 0.001 0.919\!\pm\!0.001 | 97.62 ± 4.124 97.62\!\pm\!4.124 | 0.922 ± 0.035 0.922\!\pm\!0.035 | |
| NoEngagementTrack | 0.594 ± 0.025 0.594\!\pm\!0.025 | 100.0 ± 0.000 100.0\!\pm\!0.000 | 0.607 ± 0.024 0.607\!\pm\!0.024 | |
| 0.8 {0.8} | E-MPC | 0.720 ± 0.013 0.720\!\pm\!0.013 | 97.22 ± 4.811 97.22\!\pm\!4.811 | 0.730 ± 0.028  $\mathbf{0.730\!\pm\!0.028}$ |
| NoWorkloadCst | 0.786 ± 0.000 0.786\!\pm\!0.000 | 50.00 ± 0.000 50.00\!\pm\!0.000 | 0.382 ± 0.000 0.382\!\pm\!0.000 | |
| NoEngagementTrack | 0.300 ± 0.027 0.300\!\pm\!0.027 | 100.0 ± 0.000 100.0\!\pm\!0.000 | 0.307 ± 0.024 0.307\!\pm\!0.024 | |

**说明**: 该表格来自论文中的公式、奖励项或实验设置；为避免 PDF 抽取噪声，网页中仅保留可渲染的核心符号和数值。

#### Table 2: TABLE A1: Subjective user study results. Values are reported as mean ± $\pm$ standard deviation.

| Metric | WorkloadAware | E-MPC | p p -value |
| --- | --- | --- | --- |
| Engagement Tracking Accuracy | 2.40 ± 1.07 2.40\pm 1.07 | 4.00 ± 0.67  $\mathbf{4.00\pm 0.67}$ | 0.003 0.003 |
| Task Success | 4.70 ± 0.48  $\mathbf{4.70\pm 0.48}$ | 4.50 ± 0.97 4.50\pm 0.97 | 0.925 0.925 |
| Interaction Satisfaction | 2.80 ± 1.14 2.80\pm 1.14 | 4.00 ± 1.05  $\mathbf{4.00\pm 1.05}$ | 0.032 0.032 |
| Agency Satisfaction | 2.60 ± 1.26 2.60\pm 1.26 | 4.30 ± 0.95  $\mathbf{4.30\pm 0.95}$ | 0.007 0.007 |

**说明**: TABLE A1: Subjective user study results. Values are reported as mean ± $\pm$ standard deviation.

#### Table 3: TABLE A2: Gaze hit rates Close to g d  e  s g_{des} vs. Deviated Engagement.

| Category | Close to g d  e  s (%) g_{des}(\%) | Deviated (%) |
| --- | --- | --- |
| Robot | 64.43 | 73.35 |
| Food plate | 11.51 | 4.33 |
| iPad | 9.27 | 4.14 |
| No hit | 14.79 | 18.18 |

**说明**: TABLE A2: Gaze hit rates Close to g d  e  s g_{des} vs. Deviated Engagement.

#### Table 4: TABLE A3: Head pose statistics Close to g d  e  s g_{des} vs. Deviated Engagement.

| | Close to g d  e  s g_{des} | Deviated | | |
| --- | --- | --- | --- | --- |
| Metric | Mean (∘) | Std (∘) | Mean (∘) | Std (∘) |
| Pitch (deg) | -38.28 | 28.06 | -30.57 | 27.68 |
| Yaw (deg) | -58.17 | 21.71 | -53.25 | 41.17 |
| Roll (deg) | 41.18 | 30.39 | 32.60 | 32.50 |

**说明**: TABLE A3: Head pose statistics Close to g d  e  s g_{des} vs. Deviated Engagement.
## 实验解读

- 评价重点:围绕 dynamics-modeling、模型预测控制、recovery、鲁棒控制、tool-manipulation,优先看真实机器人结果、扰动恢复、接触稳定性、控制误差、成功率、实时性和 sim-to-real 差距。
- System0 视角:实验结论不只看最终成功率,还要看闭环过程中是否能处理状态估计误差、执行延迟、碰撞/接触突变和外部扰动。
- 与上层系统的关系:如果方法能在快闭环内稳定工作,它可以作为 System1 技能或 System2 规划的底层执行保障;如果只在离线或慢速设定下成立,则还不能直接作为身体级快闭环模块。
## 批判性思考

### 优点

- 论文把 dynamics-modeling、模型预测控制、recovery、鲁棒控制、tool-manipulation 放到机器人身体级闭环里讨论,和只做高层语义推理的工作相比,更接近真实机器人部署时会遇到的状态、动力学和安全约束。
- 若论文同时给出公式、系统结构图、硬件实验或消融结果,这些证据可以帮助判断方法是否真的能支撑毫秒到百毫秒级反应。

### 局限

- 需要重点核查控制频率、推理延迟、传感延迟、失败恢复、硬件磨损、未见场景泛化和安全边界;这些因素决定它能否成为可靠的 System0 模块。
- 对学习型方法,还要区分训练环境中的鲁棒性和真实部署中的鲁棒性,尤其是接触丰富任务、动态运动任务和人形机器人任务。
## 论文定位

- 研究对象:Beyond Failure Recovery: An Engagement-Aware Human-in-the-loop Framework for Robotic Systems。
- 关键词:dynamics-modeling、模型预测控制、recovery、鲁棒控制、tool-manipulation。
- System0 关联:关注机器人身体级快反应链路中“状态/接触/局部感知 -> 约束/优化/策略 -> 动作”的短闭环,不把问题停留在语言规划或任务分解层。

## 速查卡片

> [!summary] Beyond Failure Recovery
> - **论文**: https://www.roboticsproceedings.org/rss22/p118.pdf
> - **arXiv**: http://arxiv.org/abs/2606.18189v1
> - **arXiv HTML**: https://arxiv.org/html/2606.18189v1
