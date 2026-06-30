---
title: "Safety with Agency: Human-Centered Safety Filter with Application to AI-Assisted Motorsports"
method_name: "Safety with Agency"
authors: ["Donggeon David Oh"]
year: 2025
venue: "RSS"
tags: ["robust-control", "safe-control", "robot-generalization"]
image_source: "online"
arxiv_html: "https://ar5iv.labs.arxiv.org/html/2504.11717v4"
---
# Safety with Agency
## 一句话总结

> Safety with Agency: Human-Centered Safety Filter with Application to AI-Assisted Motorsports 主要落在 [[inference-time-algorithm]]、[[鲁棒控制]]、[[safe-control]]、[[安全过滤]] 这一类问题上;从 System0 视角看,它的价值在于把身体状态、接触变化、局部感知或动力学约束接到机器人低层闭环里,让动作生成更稳定、安全、实时。
## 核心贡献

1. 围绕 **Safety with Agency: Human-Centered Safety Filter with Application to AI-Assisted Motorsports** 建立了一个与 inference-time-algorithm、鲁棒控制、safe-control、安全过滤 相关的机器人问题设定,重点不是高层语义规划,而是身体级闭环中的状态变化、接触变化或控制约束。
2. 方法层面的关键价值在于把感知、动力学、约束优化、策略学习或硬件反馈组织成可执行的低层动作生成机制,适合放进 System0 的快反应模块中理解。
3. 对系统集成而言,这类工作补上了 System1 技能选择与电机级控制之间的一层:它负责把上层目标变成动态可行、可恢复、可安全过滤的动作。
4. 若论文包含真实机器人、接触任务或仿真到真实迁移,核心应看它是否报告了闭环稳定性、扰动恢复、硬件延迟、失败模式和跨场景泛化。
## 问题背景

这篇论文可以放在 System0 的“具身快系统”背景下阅读:机器人需要在很短时间内根据本体状态、接触、力/触觉、局部视觉或动力学预测做出动作,而不是等待慢速规划器重新推理整个任务。inference-time-algorithm、鲁棒控制、safe-control、安全过滤 相关问题的难点通常不在“知道要做什么”,而在动作执行过程中状态会快速偏离、接触会突然改变、约束会实时收紧,低层系统必须及时修正。

### System0 关联

精读时优先看三件事:第一,论文使用哪些快速反馈量,例如关节状态、接触状态、力觉、触觉、深度或局部视觉;第二,反馈如何进入 MPC、WBC、策略网络、安全过滤器或恢复控制器;第三,输出是否直接影响机器人下一步动作,而不是只服务离线分析或高层任务规划。
## 方法详解

### System0 视角解析

从 System0 角度看,这篇论文的方法部分应当围绕 inference-time-algorithm、鲁棒控制、safe-control、安全过滤 来读。核心问题是:系统如何从快速变化的机器人状态中构造控制输入,如何处理接触、碰撞、约束、模型误差或执行延迟,以及如何把这些信息变成下一时刻的动作。

如果方法里有优化问题,重点看目标函数、约束和求解频率;如果方法里有学习策略,重点看观测空间、动作空间、训练扰动和部署时的闭环频率;如果方法里有硬件或传感器,重点看传感链路是否真的进入控制回路。这样读可以把论文和 System2 语言规划、System1 技能编排区分开来。

### 关键公式与机制

#### 公式 1: [[优化目标/约束]]

$${V}({x})=\min\{g({x}),\max_{{u}\in{\mathcal{U}}}{V}({f}({x},{u}))\},$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 2: [[优化目标/约束]]

$${\mathcal{Q}}({x},{u})=\min\{g({x}),\max_{{u}^{\prime}\in{\mathcal{U}}}{\mathcal{Q}}({f}({x},{u}),{u}^{\prime})\},$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 3: [[优化目标/约束]]

$${u}({x})=\begin{cases}{{\pi}^{\text{task}}}({x}),&\forall{x}\in{\mathcal{X}}\quad\text{s.t.}\quad{V}({x})>0\\ {{\pi}^{\text{shield*}}}({x}),&\text{otherwise},\end{cases}$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 4: [[优化目标/约束]]

$$\min\{g({x}),\max_{{u}^{\prime}\in{\mathcal{U}}}{\mathcal{Q}}(f({x},{u}),{u}^{\prime})\}-{V}({x})\geq(\gamma-1){V}({x})$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 5: [[优化目标/约束]]

$${V}(f({x},{u}))=\max_{{u}^{\prime}\in{\mathcal{U}}}{\mathcal{Q}}(f({x},{u}),{u}^{\prime})$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 6: [[优化目标/约束]]

$$\mathcal{U}_{\text{FT}}=\{({u}_{\text{steer}},{u}_{\text{throttle}},{u}_{\text{brake}})\mid{u}_{\text{steer}}\in[-1,1],{u}_{\text{throttle}}=1,{u}_{\text{brake}}=-1\}$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 7: [[优化目标/约束]]

$$V_{\phi}(x)=(1-{\gamma_{\text{ENV}}})g(x)+{\gamma_{\text{ENV}}}\min\{g(x),\max_{u}Q_{\phi}(x,u))\}.$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 8: [[优化目标/约束]]

$$y:=(1-{\gamma_{\text{ENV}}})\,g^{\prime}\;+\;{\gamma_{\text{ENV}}}\,\min\!\bigl{\{}g^{\prime},\,Q_{\phi}\bigl(x^{\prime},\,u^{\prime}\bigr)\bigr{\}}$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 9: [[优化目标/约束]]

$$L(\theta):=\mathbb{E}^{\mathcal{B},\pi}[-Q_{\phi}(x,u)+\alpha\log\pi_{\theta}(u|x)].$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 10: [[优化目标/约束]]

$$\max_{{u}^{\prime}\in{\mathcal{U}}}{\mathcal{Q}}(f({x},{u})),{u}^{\prime})-{V}({x})\geq(\gamma-1){V}({x})\quad\text{(Condition 2)}$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。
## 关键图表

### Figure 1: Pipeline / core system figure: a) Last-Resort Safety Filter

![Figure 1](https://ar5iv.labs.arxiv.org/html/2504.11717/assets/figs/teaser/saferacing_rss25_figures_half.001.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Pipeline / core system figure: a) Last-Resort Safety Filter”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 2: A diagram describing the interaction between a human operator, our proposed HCSF, an

![Figure 2](https://ar5iv.labs.arxiv.org/html/2504.11717/assets/figs/system_diagram/saferacing_rss25_figures.005.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“A diagram describing the interaction between a human operator, our proposed HCSF, an”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 3: Histogram of I.M. for the HCSF and LRSF groups all timesteps, including those wh

![Figure 3](https://ar5iv.labs.arxiv.org/html/2504.11717/assets/x5.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Histogram of I.M. for the HCSF and LRSF groups all timesteps, including those wh”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### 论文表格

#### Table 1: TABLE I: Number of participants in and the average initial skill level of each group.

| | HCSF | LRSF | None |
| --- | --- | --- | --- |
| Number of Participants | 29 29 | 29 29 | 25 25 |
| Average Initial Skill level | 2.17 ± 0.54 2.17\pm 0.54 | 2.21 ± 0.86 2.21\pm 0.86 | 2.28 ± 0.68 2.28\pm 0.68 |

**说明**: TABLE I: Number of participants in and the average initial skill level of each group.

#### Table 2: TABLE II: Duration of and type of assistance provided in each session.

| Driving Session | Time | Assistance |
| --- | --- | --- |
| Session 1 | 5 minutes | None |
| Session 2 | 10 minutes | LRSF, HCSF, or None |
| Session 3 | 5 minutes | None |

**说明**: TABLE II: Duration of and type of assistance provided in each session.

#### Table 3: TABLE III: 4 core metrics and 4 filter-specific metrics together with their corresponding affirmative/negated questions

| Metric | Related Questions |
| --- | --- |
| Robustness | affirmative: I felt confident that I could drive safely throughout the race. negated: I felt nervous about whether I would be able to finish the race without accidents. |
| Agency | affirmative: I felt I was in control of the vehicle throughout the race. negated: There were times when the vehicle wasn’t doing what I wanted. |
| Comfort | affirmative: The driving experience felt smooth. negated: There were times when the drive felt jerky. |
| Satisfaction | affirmative: Overall, I am happy with how the race went. negated: This race did not go as well as I thought it could have. |
| Trustworthiness | affirmative: I trusted the AI Assistant to keep me safe throughout the race. negated: I felt uneasy about whether the AI Assistant would get me into an accident. |
| Predictability | affirmative: The AI Assistant’s actions were predictable. negated: The AI Assistant’s actions surprised me at times. |
| Interpretability | affirmative: The AI Assistant’s actions made sense to me. negated: Sometimes, I couldn’t figure out why the AI Assistant was taking actions. |
| Competence | affirmative: The AI Assistant seemed to have a good grasp of the situation. negated: Sometimes, the AI Assistant didn’t seem to know what it was doing. |

**说明**: TABLE III: 4 core metrics and 4 filter-specific metrics together with their corresponding affirmative/negated questions.

#### Table 4: TABLE IV: Hyperparameters for the Warmup & Initialization Phases

| Observation | Dimensionality | Past Timesteps |
| --- | --- | --- |
| Ego vehicle speed | 1 | 4 |
| Gap to the reference path | 1 | 4 |
| Force feedback | 1 | 4 |
| RPM | 1 | 4 |
| Acceleration | 2 | 4 |
| Gear | 1 | 4 |
| Angular velocity | 1 | 4 |
| Local velocity | 2 | 4 |
| Slip Angle | 4 | 4 |
| Distance to track boundary | 11 | 4 |
| Out-of-track | 1 | 1 |
| Look ahead curvature | 12 | 1 |
| Control input | 3 | 4 |
| Ego vehicle heading | 1 | 1 |
| Ego vehicle yaw | 1 | 1 |
| Opponent distance | 1 | 1 |
| Opponent direction | 1 | 1 |
| Opponent speed | 1 | 1 |
| Opponent heading | 1 | 1 |
| Opponent yaw | 1 | 1 |
| Opponent brake | 1 | 1 |

**说明**: TABLE IV: Hyperparameters for the Warmup & Initialization Phases

#### Table 5: TABLE V: Hyperparameters for the Warmup & Initialization Phases

| Hyperparameter | Value | Description |
| --- | --- | --- |
| T warmup, max T^{{ $\text{warmup}},\max}$ | 25  s 25 $\text{s}$ | Max duration of warmup phase. |
| P warmup P^{ $\text{warmup}}_{\text{}}$ | 0.6 0.6 | Probability of using policy during warmup. |
| P oppo warmup P^{ $\text{warmup}}_{\text{oppo}}$ | 0.25 0.25 | Probability of early termination if near an opponent. |
| d oppo warmup, min d^{{ $\text{warmup}},\min}_{\text{oppo}}$ | 6  m 6 $\text{m}$ | Min distance threshold for opponent proximity. |
| d oppo warmup, max d^{{ $\text{warmup}},\max}_{\text{oppo}}$ | 36  m 36 $\text{m}$ | Max distance threshold for opponent proximity. |
| P brake, epi warmup P^{ $\text{warmup}}_{\text{brake, epi}}$ | 0.4 0.4 | Probability that heavy braking triggers warmup termination. |
| P brake, step warmup P^{ $\text{warmup}}_{\text{brake, step}}$ | 0.25 0.25 | Probability of termination on each timestep of heavy braking. |
| u brake warmup {u}^{ $\text{warmup}}_{\text{brake}}$ | 0.6 0.6 | Threshold for heavy braking input. |
| v warmup v^{ $\text{warmup}}$ | 40  m/s 40~ $\text{m/s}$ | Speed threshold which braking triggers termination. |
| P term init P^{ $\text{init}}_{\text{term}}$ | 0.2 0.2 | Probability of ending initialization once Q { $\mathcal{Q}} -value falls below threshold.$ |
| Q term init { $\mathcal{Q}}^{\text{init}}_{\text{term}}$ | 2 2 | Q { $\mathcal{Q}} -value threshold for indicating dangerous scenarios.$ |
| P FT init P^{ $\text{init}}_{\text{FT}}$ | 0.4 0.4 | Probability of enabling full-throttle mode. |
| P adv init P^{ $\text{init}}_{\text{adv}}$ | 0.3 0.3 | Probability of adversarial initialization. |
| P rand init P^{ $\text{init}}_{\text{rand}}$ | 0.3 0.3 | Probability of random initialization. |
| P mix init P^{ $\text{init}}_{\text{mix}}$ | 0.4 0.4 | Probability of mixed initialization. |
| c 1 c_{1} | 1 / 12 1/12 | Design parameter for training the nominal warmup policy. |
| c 2 c_{2} | 300 300 | Design parameter for training the nominal warmup policy. |
| d oppo d^{ $\text{}}_{\text{oppo}}$ | 100  m 100 $\text{m}$ | Distance threshold for rewards. |
| c 3 c_{3} | 600 600 | Design parameter for training the policy. |

**说明**: TABLE V: Hyperparameters for the Warmup & Initialization Phases

#### Table 6: TABLE VI: Hyperparameters for SAC Training

| Hyperparameter | Value | Description |
| --- | --- | --- |
|  \eta | 3  10 - 4 3\times 10^{-4} | Actor and critic learning rate. |
| $\gamma_{\text{ENV}}}$ | 0.992 0.992 | Discount factor. |
| \| B \| \| $\mathcal{B}\lVert$ | 2  10 7 2\times 10^{7} | Replay buffer size. |
| Batch Size | 256 256 | Training batch size. |
| $\tau$ | 0.005 0.005 | Target network update rate. |
|  \alpha | Learned | Entropy temperature. |
| N U  T  D N_{UTD} | 1 1 | Gradient steps per environment step. |

**说明**: TABLE VI: Hyperparameters for SAC Training

#### Table 7: TABLE VII: Cronbach’s Alpha Test Results

| Metric | Cronbach’s Alpha Value |
| --- | --- |
| Robustness | 0.673 0.673 |
| Agency | 0.716 0.716 |
| Comfort | 0.775 0.775 |
| Satisfaction | 0.710 0.710 |
| Trustworthiness | 0.683 0.683 |
| Predictability | 0.175 0.175 |
| Interpretability | 0.562 0.562 |
| Competence | 0.540 0.540 |

**说明**: TABLE VII: Cronbach’s Alpha Test Results
## 实验解读

- 评价重点:围绕 inference-time-algorithm、鲁棒控制、safe-control、安全过滤,优先看真实机器人结果、扰动恢复、接触稳定性、控制误差、成功率、实时性和 sim-to-real 差距。
- System0 视角:实验结论不只看最终成功率,还要看闭环过程中是否能处理状态估计误差、执行延迟、碰撞/接触突变和外部扰动。
- 与上层系统的关系:如果方法能在快闭环内稳定工作,它可以作为 System1 技能或 System2 规划的底层执行保障;如果只在离线或慢速设定下成立,则还不能直接作为身体级快闭环模块。
## 批判性思考

### 优点

- 论文把 inference-time-algorithm、鲁棒控制、safe-control、安全过滤 放到机器人身体级闭环里讨论,和只做高层语义推理的工作相比,更接近真实机器人部署时会遇到的状态、动力学和安全约束。
- 若论文同时给出公式、系统结构图、硬件实验或消融结果,这些证据可以帮助判断方法是否真的能支撑毫秒到百毫秒级反应。

### 局限

- 需要重点核查控制频率、推理延迟、传感延迟、失败恢复、硬件磨损、未见场景泛化和安全边界;这些因素决定它能否成为可靠的 System0 模块。
- 对学习型方法,还要区分训练环境中的鲁棒性和真实部署中的鲁棒性,尤其是接触丰富任务、动态运动任务和人形机器人任务。
## 论文定位

- 研究对象:Safety with Agency: Human-Centered Safety Filter with Application to AI-Assisted Motorsports。
- 关键词:inference-time-algorithm、鲁棒控制、safe-control、安全过滤。
- System0 关联:关注机器人身体级快反应链路中“状态/接触/局部感知 -> 约束/优化/策略 -> 动作”的短闭环,不把问题停留在语言规划或任务分解层。

## 速查卡片

> [!summary] Safety with Agency
> - **论文**: https://www.roboticsproceedings.org/rss21/p093.pdf
> - **arXiv**: http://arxiv.org/abs/2504.11717v4
> - **arXiv HTML**: https://ar5iv.labs.arxiv.org/html/2504.11717v4
