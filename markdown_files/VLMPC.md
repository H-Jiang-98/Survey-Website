---
title: "VLMPC: Vision-Language Model Predictive Control for Robotic Manipulation"
method_name: "VLMPC"
authors: ["Wentao Zhao"]
year: 2024
venue: "RSS"
tags: ["robot-manipulation", "robot-generalization", "model-predictive-control"]
image_source: "online"
arxiv_html: "https://arxiv.org/html/2407.09829v1"
---
# VLMPC
## 一句话总结

> VLMPC: Vision-Language Model Predictive Control for Robotic Manipulation 主要落在 [[language-conditioned-policy]]、[[模型预测控制]]、[[机器人操作]] 这一类问题上;从 System0 视角看,它的价值在于把身体状态、接触变化、局部感知或动力学约束接到机器人低层闭环里,让动作生成更稳定、安全、实时。
## 核心贡献

1. 围绕 **VLMPC: Vision-Language Model Predictive Control for Robotic Manipulation** 建立了一个与 language-conditioned-policy、模型预测控制、机器人操作 相关的机器人问题设定,重点不是高层语义规划,而是身体级闭环中的状态变化、接触变化或控制约束。
2. 方法层面的关键价值在于把感知、动力学、约束优化、策略学习或硬件反馈组织成可执行的低层动作生成机制,适合放进 System0 的快反应模块中理解。
3. 对系统集成而言,这类工作补上了 System1 技能选择与电机级控制之间的一层:它负责把上层目标变成动态可行、可恢复、可安全过滤的动作。
4. 若论文包含真实机器人、接触任务或仿真到真实迁移,核心应看它是否报告了闭环稳定性、扰动恢复、硬件延迟、失败模式和跨场景泛化。
## 问题背景

这篇论文可以放在 System0 的“具身快系统”背景下阅读:机器人需要在很短时间内根据本体状态、接触、力/触觉、局部视觉或动力学预测做出动作,而不是等待慢速规划器重新推理整个任务。language-conditioned-policy、模型预测控制、机器人操作 相关问题的难点通常不在“知道要做什么”,而在动作执行过程中状态会快速偏离、接触会突然改变、约束会实时收紧,低层系统必须及时修正。

### System0 关联

精读时优先看三件事:第一,论文使用哪些快速反馈量,例如关节状态、接触状态、力觉、触觉、深度或局部视觉;第二,反馈如何进入 MPC、WBC、策略网络、安全过滤器或恢复控制器;第三,输出是否直接影响机器人下一步动作,而不是只服务离线分析或高层任务规划。
## 方法详解

### System0 视角解析

从 System0 角度看,这篇论文的方法部分应当围绕 language-conditioned-policy、模型预测控制、机器人操作 来读。核心问题是:系统如何从快速变化的机器人状态中构造控制输入,如何处理接触、碰撞、约束、模型误差或执行延迟,以及如何把这些信息变成下一时刻的动作。

如果方法里有优化问题,重点看目标函数、约束和求解频率;如果方法里有学习策略,重点看观测空间、动作空间、训练扰动和部署时的闭环频率;如果方法里有硬件或传感器,重点看传感链路是否真的进入控制回路。这样读可以把论文和 System2 语言规划、System1 技能编排区分开来。

### 关键公式与机制

#### 公式 1: [[优化目标/约束]]

$$\mathcal{S}_{t}=\{S^{1}_{t},S^{2}_{t},...,S^{N}_{t}\}$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 2: [[动力学或策略机制]]

$$V^{n}_{t}=\{\widehat{O}^{n}_{\tau}(a^{n}_{\tau})|\tau\in\{t+1,...,t+T\}$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 3: [[动力学或策略机制]]

$$\begin{split}O_{t-1}^{\prime}&=[O_{t-1}\cdot a_{t-1}^{\prime}\cdot a_{t}^{\prime}\cdot{a^{n}_{t+1}}^{\prime}],\\ O_{t}^{\prime}&=[O_{t}\cdot a_{t-1}^{\prime}\cdot a_{t}^{\prime}\cdot{a^{n}_{t+1}}^{\prime}]\end{split}$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 4: [[动力学或策略机制]]

$$\begin{split}\text{VT}(V_{t}| e_{t},s_{t},I_{t})=\{&e_{ $\tau}^{n},s_{\tau}^{n},I_{\tau}^{n}$ |n\in\{1,2,...,N\},\\ &\tau\in\{t+1,t+2,...,t+T\}\}\end{split}$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 5: [[动力学或策略机制]]

$$C_{\text{VLM}}^{n}(t)=\sum\limits_{\tau=t+1}^{t+T}(|| c(e_{ $\tau}^{n})-c(s_{\tau}^{n})$ ||_{2}-|| c(e_{ $\tau}^{n})-c(I_{\tau}^{n})$ ||_{2}),$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 6: [[动力学或策略机制]]

$$C_{P}^{n}(t)=\sum\limits_{\tau=t+1}^{t+T}|| \widehat{O}^{n}_{ $\tau}(a^{n}_{\tau})-G$ ||_{2}$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 7: [[动力学或策略机制]]

$$\mu^{\text{VLM}}_{t}=w_{m}*\{\widehat{d}^{x}_{t},\widehat{d}^{y}_{t},\widehat{d}^{z}_{t}\}\cup w_{r}*\{\widehat{r}^{x}_{t},\widehat{r}^{y}_{t},\widehat{r}^{z}_{t}\}\cup\{g_{t}\}$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 8: [[动力学或策略机制]]

$$\text{VLM}(O_{t},G\lor L|\phi_{s})=\{\widehat{d}^{x}_{t},\widehat{d}^{y}_{t},\widehat{d^{z}_{t}},\widehat{r^{x}_{t}},\widehat{r^{y}_{t}},\widehat{r^{z}_{t}},g_{t}\}$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 9: [[动力学或策略机制]]

$$V^{n}_{t}=\{\widehat{O}^{n}_{t+1}(a^{n}_{t+1}),\widehat{O}^{n}_{t+2}(a^{n}_{t+2}),...,\widehat{O}^{n}_{t+T}(a^{n}_{t+T})\}$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 10: [[动力学或策略机制]]

$$\widehat{O}^{n}_{t+1}(a^{n}_{t+1})=\text{DMVFN-Act}(O_{t-1}^{\prime},O_{t}^{\prime})$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。
## 关键图表

### Figure 1: Pipeline / core system figure: VLMPC takes as input either a goal image or a language instruction. It first prompts V

![Figure 1](https://arxiv.org/html/2407.09829v1/x1.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Pipeline / core system figure: VLMPC takes as input either a goal image or a language instruction. It first prompts V”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 2: The real-world experimental platform includes a UR5 robot arm and a monocular RGB came

![Figure 2](https://arxiv.org/html/2407.09829v1/x6.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“The real-world experimental platform includes a UR5 robot arm and a monocular RGB came”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 3: Visualization of the action execution for two challenging real-world manipulation task

![Figure 3](https://arxiv.org/html/2407.09829v1/x7.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Visualization of the action execution for two challenging real-world manipulation task”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### 论文表格

#### Table 1: TABLE I: Comparison with existing methods on the task of move to area in the Language Table environment.

| Method | Success Rate(%) | Reward |
| --- | --- | --- |
| UniPi [18 ] | 0 | 30.8 |
| LAVA [41 ] | 22 | 59.8 |
| PALM-E [17 ] | 0 | 36.5 |
| RT-2 [10 ] | 0 | 18.5 |
| VLP [19 ] | 64 | 87.3 |
| VLMPC | 70 | 89.3 |

**说明**: TABLE I: Comparison with existing methods on the task of move to area in the Language Table environment.

#### Table 2: TABLE II: Results of VLMPC using goal image or language instruction as input in real-world experiments.

| Tasks | Goal Image | Language Instruction | | |
| --- | --- | --- | --- | --- |
| Success Rate(%) | Time(s) | Success Rate(%) | Time(s) | |
| grasp towel | 76.67 | 162.4 | 73.33 | 184.6 |
| put banana | 60.00 | 203.9 | 46.67 | 230.7 |
| turn on lamp | 83.33 | 128.4 | 86.67 | 142.8 |
| wipe water | 36.67 | 289.3 | 23.33 | 331.9 |

**说明**: TABLE II: Results of VLMPC using goal image or language instruction as input in real-world experiments.

#### Table 3: TABLE III: Ablation study using the variants of VLMPC on different tasks in real-world environments.

| VLMPC Variant | grasp towel | put banana | turn on lamp | wipe water | | | | |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Success Rate(%) | Time(s) | Success Rate(%) | Time(s) | Success Rate(%) | Time(s) | Success Rate(%) | Time(s) | |
| VLMPC-rs | 63.33 | 302.5 | 40 | 389.5 | 73.33 | 256.7 | 13.33 | 573.9 |
| VLMPC-PD | 26.67 | 178.3 | 0 | - | 60.00 | 123.6 | 0 | - |
| VLMPC-VS | 56.67 | 201.5 | 46.67 | 297.3 | 56.67 | 243.7 | 10.00 | 543.9 |
| VLMPC-MCVD | 33.33 | 509.3 | 23.33 | 689.4 | 46.67 | 553.8 | 6.67 | 803.5 |
| VLMPC | 76.67 | 162.4 | 60.00 | 203.9 | 83.33 | 128.4 | 36.67 | 289.3 |

**说明**: TABLE III: Ablation study using the variants of VLMPC on different tasks in real-world environments.
## 实验解读

- 评价重点:围绕 language-conditioned-policy、模型预测控制、机器人操作,优先看真实机器人结果、扰动恢复、接触稳定性、控制误差、成功率、实时性和 sim-to-real 差距。
- System0 视角:实验结论不只看最终成功率,还要看闭环过程中是否能处理状态估计误差、执行延迟、碰撞/接触突变和外部扰动。
- 与上层系统的关系:如果方法能在快闭环内稳定工作,它可以作为 System1 技能或 System2 规划的底层执行保障;如果只在离线或慢速设定下成立,则还不能直接作为身体级快闭环模块。
## 批判性思考

### 优点

- 论文把 language-conditioned-policy、模型预测控制、机器人操作 放到机器人身体级闭环里讨论,和只做高层语义推理的工作相比,更接近真实机器人部署时会遇到的状态、动力学和安全约束。
- 若论文同时给出公式、系统结构图、硬件实验或消融结果,这些证据可以帮助判断方法是否真的能支撑毫秒到百毫秒级反应。

### 局限

- 需要重点核查控制频率、推理延迟、传感延迟、失败恢复、硬件磨损、未见场景泛化和安全边界;这些因素决定它能否成为可靠的 System0 模块。
- 对学习型方法,还要区分训练环境中的鲁棒性和真实部署中的鲁棒性,尤其是接触丰富任务、动态运动任务和人形机器人任务。
## 论文定位

- 研究对象:VLMPC: Vision-Language Model Predictive Control for Robotic Manipulation。
- 关键词:language-conditioned-policy、模型预测控制、机器人操作。
- System0 关联:关注机器人身体级快反应链路中“状态/接触/局部感知 -> 约束/优化/策略 -> 动作”的短闭环,不把问题停留在语言规划或任务分解层。

## 速查卡片

> [!summary] VLMPC
> - **论文**: https://www.roboticsproceedings.org/rss20/p106.pdf
> - **arXiv**: http://arxiv.org/abs/2407.09829v1
> - **arXiv HTML**: https://arxiv.org/html/2407.09829v1
