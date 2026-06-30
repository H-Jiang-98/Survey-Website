---
title: "ConRFT: A Reinforced Fine-tuning Method for VLA Models via Consistency Policy"
method_name: "ConRFT"
authors: ["Yuhui Chen"]
year: 2025
venue: "RSS"
tags: ["contact-reasoning", "robot-manipulation", "robust-control", "real-time-control", "reinforcement-learning", "safe-control", "imitation-learning", "contact-rich-manipulation", "robot-generalization"]
image_source: "online"
arxiv_html: "https://arxiv.org/html/2502.05450v2"
---
# ConRFT
## 一句话总结

> ConRFT: A Reinforced Fine-tuning Method for VLA Models via Consistency Policy 主要落在 [[behavior-cloning]]、[[接触推理]]、[[接触丰富操作]]、[[模仿学习]] 这一类问题上;从 System0 视角看,它的价值在于把身体状态、接触变化、局部感知或动力学约束接到机器人低层闭环里,让动作生成更稳定、安全、实时。
## 核心贡献

1. 围绕 **ConRFT: A Reinforced Fine-tuning Method for VLA Models via Consistency Policy** 建立了一个与 behavior-cloning、接触推理、接触丰富操作、模仿学习、language-conditioned-policy、实时控制 相关的机器人问题设定,重点不是高层语义规划,而是身体级闭环中的状态变化、接触变化或控制约束。
2. 方法层面的关键价值在于把感知、动力学、约束优化、策略学习或硬件反馈组织成可执行的低层动作生成机制,适合放进 System0 的快反应模块中理解。
3. 对系统集成而言,这类工作补上了 System1 技能选择与电机级控制之间的一层:它负责把上层目标变成动态可行、可恢复、可安全过滤的动作。
4. 若论文包含真实机器人、接触任务或仿真到真实迁移,核心应看它是否报告了闭环稳定性、扰动恢复、硬件延迟、失败模式和跨场景泛化。
## 问题背景

这篇论文可以放在 System0 的“具身快系统”背景下阅读:机器人需要在很短时间内根据本体状态、接触、力/触觉、局部视觉或动力学预测做出动作,而不是等待慢速规划器重新推理整个任务。behavior-cloning、接触推理、接触丰富操作、模仿学习、language-conditioned-policy、实时控制、强化学习 相关问题的难点通常不在“知道要做什么”,而在动作执行过程中状态会快速偏离、接触会突然改变、约束会实时收紧,低层系统必须及时修正。

### System0 关联

精读时优先看三件事:第一,论文使用哪些快速反馈量,例如关节状态、接触状态、力觉、触觉、深度或局部视觉;第二,反馈如何进入 MPC、WBC、策略网络、安全过滤器或恢复控制器;第三,输出是否直接影响机器人下一步动作,而不是只服务离线分析或高层任务规划。
## 方法详解

### System0 视角解析

从 System0 角度看,这篇论文的方法部分应当围绕 behavior-cloning、接触推理、接触丰富操作、模仿学习、language-conditioned-policy、实时控制、强化学习、机器人操作 来读。核心问题是:系统如何从快速变化的机器人状态中构造控制输入,如何处理接触、碰撞、约束、模型误差或执行延迟,以及如何把这些信息变成下一时刻的动作。

如果方法里有优化问题,重点看目标函数、约束和求解频率;如果方法里有学习策略,重点看观测空间、动作空间、训练扰动和部署时的闭环频率;如果方法里有硬件或传感器,重点看传感链路是否真的进入控制回路。这样读可以把论文和 System2 语言规划、System1 技能编排区分开来。

### 关键公式与机制

#### 公式 1: [[优化目标/约束]]

$$
\mathcal{L}_{\pi}^{Q}=-\mathbb{E}_{s\sim\mathcal{D},a\sim\pi_{\psi}}[Q(s,a)]
$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 2: [[优化目标/约束]]

$$=\mathbb{E}_{(s,a,s^{\prime})\sim(\mathcal{D}\cup\mathcal{R})}[(Q_{\theta}(s,a)-\mathcal{B}^{\pi}\ {Q}(s,a))^{2}]$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 3: [[优化目标/约束]]

$$\mathcal{B}^{\pi}\ {Q}(s,a)=r(s,a)+\gamma\mathbb{E}_{a^{\prime}\sim\pi(\cdot|s^{\prime})}(\ {Q}(s^{\prime},a^{\prime}))$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 4: [[优化目标/约束]]

$$
\mathcal{L}_{\pi}^{BC}=\mathbb{E}_{(s,a)\sim(\mathcal{D}\cup\mathcal{R}),m\sim\mathcal{U}[1,M-1]}[d(f_{\psi}(a+k_{m}z,k_{m}|E(s)),a)]
$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 5: [[动力学或策略机制]]

$$V^{\pi}(s)=\mathbb{E}_{\pi}[\sum_{t=0}^{H}\gamma^{t}r(s_{t},a_{t})| s_{0}=s,a_{t}\sim $\pi(s_{t}),s_{t+1}\sim p(\cdot$ |s_{t},a_{t})]$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 6: [[动力学或策略机制]]

$$Q^{\pi}(s,a)=\mathbb{E}_{\pi}[\sum_{t=0}^{H}\gamma^{t}r(s_{t},a_{t})|s_{0}=s,a_{0}=a,s_{t+1}\sim p(\cdot|s_{t},a_{t})]$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 7: [[优化目标/约束]]

$$
\mathcal{L}_{\pi}^{BC}=\mathbb{E}_{(s,a)\sim\mathcal{D},m\sim\mathcal{U}[1,M-1]}[d(f_{\psi}(a+k_{m}z,k_{m}|E(s)),a)]
$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 8: [[优化目标/约束]]

$$\alpha(\mathbb{E}_{s\sim\mathcal{D},a\sim\pi(\cdot|s)}[\max(Q_{\theta}(s,a),V^{\mu}(s))]$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 9: [[优化目标/约束]]

$$+\frac{1}{2}\mathbb{E}_{(s,a,s^{\prime})\sim\mathcal{D}}[(Q_{\theta}(s,a)-\mathcal{B}^{\pi}\ {Q}_{\ {\theta}}(s,a))^{2}]$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 10: [[优化目标/约束]]

$$
\mathcal{L}_{\pi}^{Q}=-\mathbb{E}_{s\sim(\mathcal{D}\cup\mathcal{R}),a\sim\pi_{\psi}}[Q(s,a)]
$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。
## 关键图表

### Figure 1: Overview of ConRFT. This figure illustrates the architecture of our reinforced fine-t

![Figure 1](https://arxiv.org/html/2502.05450v2/extracted/6358946/method.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Overview of ConRFT. This figure illustrates the architecture of our reinforced fine-t”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 2: Overview of all real-world experimental tasks. The real-world tasks include picking a

![Figure 2](https://arxiv.org/html/2502.05450v2/extracted/6358946/tasks.jpg)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Overview of all real-world experimental tasks. The real-world tasks include picking a”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 3: Learning curves during online training. This figure presents the success rates, inter

![Figure 3](https://arxiv.org/html/2502.05450v2/extracted/6358946/online_result.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Learning curves during online training. This figure presents the success rates, inter”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### 论文表格

#### Table 1: TABLE I: All experiment results for various offline and online fine-tuning methods. We report the policy performance ag

| | Training Time (mins) | Success Rate (%) | Episode length | | | | | | | | |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Task | SFT [47 ] | Cal- ConRFT | HG- DAgger [19 ] | PA-RL [14 ] | HIL- ConRFT | SFT [47 ] | Cal- ConRFT | HG- DAgger [19 ] | PA-RL [14 ] | HIL- ConRFT | |
| Pick Banana | 45 | 40 | 50 | 60 (+50%) | 80 (+100%) | 90 (+80%) | 63.7 | 57.8 | 67.5 (0.9x) | 56.1 (1.1x) | 51.2 (1.1x) |
| Put Spoon | 45 | 50 | 55 | 90 (+80%) | 80 (+60%) | 100 (+82%) | 49.9 | 57.2 | 50.9 (1.0x) | 45.3 (1.1x) | 22.6 (2.5x) |
| Open Drawer | 15 | 35 | 30 | 80 (+129%) | 60 (+71%) | 100 (+233%) | 63.6 | 61.7 | 48.4 (1.3x) | 57.1 (1.1x) | 32.4 (1.8x) |
| Pick Bread | 45 | 65 | 55 | 65 (+0%) | 80 (+23%) | 100 (+82%) | 53.2 | 49.1 | 65.6 (0.8x) | 51.7 (1.0x) | 31.6 (1.6x) |
| Open Toaster | 30 | 30 | 30 | 75 (+116%) | 100 (+233%) | 100 (+233%) | 51.2 | 50.7 | 43.4 (1.2x) | 34.3 (1.5x) | 22.1 (2.3x) |
| Put Bread | 60 | 5 | 20 | 60 (+1100%) | 75 (+1400%) | 100 (+400%) | 102 | 84.8 | 74.2 (1.4x) | 72.1 (1.4x) | 36.6 (2.3x) |
| Insert Wheel | 60 | 35 | 35 | 40 (+14%) | 30 (-14%) | 80 (+129%) | 42.7 | 43.4 | 53.0 (0.8x) | 47.4 (0.9x) | 21.9 (2.0x) |
| Hang Chinese Knot | 90 | 55 | 40 | 50 (-10%) | 65 (+18%) | 100 (+150%) | 52.6 | 54.9 | 47.5 (1.1x) | 44.4 (1.3x) | 26.8 (2.0x) |
| Average | 48.8 | 39.4 | 39.4 | 65 (+65%) | 71.3 (+81%) | 96.3 (+144%) | 59.9 | 57.5 | 56.3 (1.1x) | 51.1 (1.2x) | 30.7 (1.9x) |

#### Table 2: TABLE II: Experiment results for training from scratch (HIL-SERL [20 ]) and fine-tuning VLA (HIL-ConRFT). Policies ar

| | Training Time (mins) | Success Rate (%) | Episode length | | |
| --- | --- | --- | --- | --- | --- |
| Task | HIL- SERL [20 ] | HIL- ConRFT | HIL- SERL [20 ] | HIL- ConRFT | |
| Pick Banana | 45 | 0 → →  $\rightarrow → 15$ | 50 → →  $\rightarrow → 90$ | 30.6 | 51.2 |
| Put Spoon | 45 | 0 → →  $\rightarrow → 60$ | 55 → →  $\rightarrow → 100$ | 56.1 | 22.6 |
| Open Drawer | 15 | 0 → →  $\rightarrow → 10$ | 30 → →  $\rightarrow → 100$ | 67.5 | 32.4 |
| Pick Bread | 45 | 0 → →  $\rightarrow → 45$ | 55 → →  $\rightarrow → 100$ | 22.0 | 31.6 |
| Open Toaster | 30 | 0 → →  $\rightarrow → 100$ | 30 → →  $\rightarrow → 100$ | 28.1 | 22.1 |
| Put Bread | 60 | 0 → →  $\rightarrow → 5$ | 20 → →  $\rightarrow → 100$ | 62.0 | 36.6 |
| Insert Wheel | 60 | 0 → →  $\rightarrow → 5$ | 35 → →  $\rightarrow → 80$ | 42.0 | 21.9 |
| Hang Chinese Knot | 90 | 0 → →  $\rightarrow → 15$ | 40 → →  $\rightarrow → 100$ | 57.3 | 26.8 |
| Average | 48.8 | 0 → →  $\rightarrow → 31.9$ | 39.4 → →  $\rightarrow → 96.3$ | 45.7 | 30.7 |

**说明**: TABLE II: Experiment results for training from scratch (HIL-SERL [20 ]) and fine-tuning VLA (HIL-ConRFT). Policies are trained using the same number of episodes with human interventions. All metrics are reported 20 trials per task.

#### Table 3: TABLE III: Experimental comparisons with various demonstrations. Diffusion Policy (DP) [50 ] and SFT [47 ] are traine

| | Success Rate (%) | | | | |
| --- | --- | --- | --- | --- | --- |
| Task | DP [50 ] | SFT [47 ] | RLDG [6 ] | Cal-ConRFT | HIL-ConRFT |
| Put Spoon | 60 | 70 | 100 | 55 | 100 |
| Put Bread | 30 | 65 | 100 | 20 | 100 |
| Insert Wheel | 35 | 40 | 50 | 35 | 80 |
| Average | 41.7 | 58.3 | 83.3 | 36.7 | 93.3 |

**说明**: TABLE III: Experimental comparisons with various demonstrations. Diffusion Policy (DP) [50 ] and SFT [47 ] are trained with 150 demonstrations collected by human teleoperation, while RLDG [6 ] is trained with 150 demonstrations collected by RL policy. Cal-ConRFT is trained with 20 demonstrations collected by human teleoperation, and HIL-ConRFT is trained with 20 demonstrations as well as 80-120 policy-generated rollout trajectories. All metrics are reported 20 trials per task.

#### Table 4: TABLE IV: Experimental results of ConRFT on different VLA models. We fine-tune RoboVLM [51 ] with two VLM backbones us

| | Success Rate (%) | |
| --- | --- | --- |
| Task | Kosmos-2(1.6B) | PaliGemma(3B) |
| Pick Banana | 60 → →  $\rightarrow → 100$ | 65 → →  $\rightarrow → 100$ |
| Put Spoon | 55 → →  $\rightarrow → 100$ | 30 → →  $\rightarrow → 100$ |
| Hang Chinese Knot | 45 → →  $\rightarrow → 100$ | 60 → →  $\rightarrow → 100$ |
| Average | 53.3 → →  $\rightarrow → 100$ | 51.7 → →  $\rightarrow → 100$ |

**说明**: TABLE IV: Experimental results of ConRFT on different VLA models. We fine-tune RoboVLM [51 ] with two VLM backbones using our method. Specifically, we fine-tune only the action head while keeping the visual encoders and transformer backbone frozen. All metrics are reported 20 trials per task.

#### Table 5: TABLE V: Policy training details for the Pick Banana task.

| Parameter | Value |
| --- | --- |
| Action space | 7-dimensional |
| Initial offline demonstrations | 20 |
| Max episode length | 100 |
| Reset method | Human reset |
| Randomization range | 3 cm in x and y |
| (, , )    (\alpha,\beta,\eta) (,,) for offline fine-tuning | (0.01, 1.0, 0.1) 0.01 1.0 0.1 (0.01,1.0,0.1) (0.01, 1.0, 0.1) |
| (, )   (\beta,\eta) (,) for online fine-tuning | (0.5, 1.0) 0.5 1.0 (0.5,1.0) (0.5, 1.0) |

**说明**: TABLE V: Policy training details for the Pick Banana task.

#### Table 6: TABLE VI: Policy training details for the Put Spoon task.

| Parameter | Value |
| --- | --- |
| Action space | 7-dimensional |
| Initial offline demonstrations | 20 |
| Max episode length | 100 |
| Reset method | Human reset |
| Randomization range | 3 cm in x and y |
| (, , )    (\alpha,\beta,\eta) (,,) for offline fine-tuning | (0.01, 1.0, 0.1) 0.01 1.0 0.1 (0.01,1.0,0.1) (0.01, 1.0, 0.1) |
| (, )   (\beta,\eta) (,) for online fine-tuning | (0.5, 1.0) 0.5 1.0 (0.5,1.0) (0.5, 1.0) |

**说明**: TABLE VI: Policy training details for the Put Spoon task.

#### Table 7: TABLE VII: Policy training details for the Open Drawer task.

| Parameter | Value |
| --- | --- |
| Action space | 6-dimensional |
| Initial offline demonstrations | 20 |
| Max episode length | 100 |
| Reset method | Script reset |
| Randomization range | 3 cm in y and x |
| (, , )    (\alpha,\beta,\eta) (,,) for offline fine-tuning | (0.01, 1.0, 0.1) 0.01 1.0 0.1 (0.01,1.0,0.1) (0.01, 1.0, 0.1) |
| (, )   (\beta,\eta) (,) for online fine-tuning | (0.5, 1.0) 0.5 1.0 (0.5,1.0) (0.5, 1.0) |

**说明**: TABLE VII: Policy training details for the Open Drawer task.

#### Table 8: TABLE VIII: Policy training details for the Pick Bread task.

| Parameter | Value |
| --- | --- |
| Action space | 7-dimensional |
| Initial offline demonstrations | 30 |
| Max episode length | 100 |
| Reset method | Human reset |
| Randomization range | 2 cm in x and y |
| (, , )    (\alpha,\beta,\eta) (,,) for offline fine-tuning | (0.01, 1.0, 0.1) 0.01 1.0 0.1 (0.01,1.0,0.1) (0.01, 1.0, 0.1) |
| (, )   (\beta,\eta) (,) for online fine-tuning | (0.5, 1.0) 0.5 1.0 (0.5,1.0) (0.5, 1.0) |

**说明**: TABLE VIII: Policy training details for the Pick Bread task.

#### Table 9: TABLE IX: Policy training details for the Open Toaster task.

| Parameter | Value |
| --- | --- |
| Action space | 6-dimensional |
| Initial offline demonstrations | 20 |
| Max episode length | 100 |
| Reset method | Script reset |
| Randomization range | 2 cm in y and z |
| (, , )    (\alpha,\beta,\eta) (,,) for offline fine-tuning | (0.01, 1.0, 0.1) 0.01 1.0 0.1 (0.01,1.0,0.1) (0.01, 1.0, 0.1) |
| (, )   (\beta,\eta) (,) for online fine-tuning | (0.5, 1.0) 0.5 1.0 (0.5,1.0) (0.5, 1.0) |

**说明**: TABLE IX: Policy training details for the Open Toaster task.

#### Table 10: TABLE X: Policy training details for the Put Bread task.

| Parameter | Value |
| --- | --- |
| Action space | 7-dimensional |
| Initial offline demonstrations | 30 |
| Max episode length | 120 |
| Reset method | Human reset |
| Randomization range | 2 cm in x and y |
| (, , )    (\alpha,\beta,\eta) (,,) for offline fine-tuning | (0.01, 1.0, 0.1) 0.01 1.0 0.1 (0.01,1.0,0.1) (0.01, 1.0, 0.1) |
| (, )   (\beta,\eta) (,) for online fine-tuning | (0.5, 1.0) 0.5 1.0 (0.5,1.0) (0.5, 1.0) |

**说明**: TABLE X: Policy training details for the Put Bread task.

#### Table 11: TABLE XI: Policy training details for the Insert Wheel task.

| Parameter | Value |
| --- | --- |
| Action space | 7-dimensional |
| Initial offline demonstrations | 30 |
| Max episode length | 100 |
| Reset method | Human reset |
| Randomization range | 2 cm in x and y |
| (, , )    (\alpha,\beta,\eta) (,,) for offline fine-tuning | (0.01, 1.0, 0.1) 0.01 1.0 0.1 (0.01,1.0,0.1) (0.01, 1.0, 0.1) |
| (, )   (\beta,\eta) (,) for online fine-tuning | (0.5, 1.0) 0.5 1.0 (0.5,1.0) (0.5, 1.0) |

**说明**: TABLE XI: Policy training details for the Insert Wheel task.

#### Table 12: TABLE XII: Policy training details for the Hang Chinese Knot task.

| Parameter | Value |
| --- | --- |
| Action space | 7-dimensional |
| Initial offline demonstrations | 30 |
| Max episode length | 100 |
| Reset method | Human reset |
| Randomization range | 3 cm in y and z |
| (, , )    (\alpha,\beta,\eta) (,,) for offline fine-tuning | (0.01, 1.0, 0.1) 0.01 1.0 0.1 (0.01,1.0,0.1) (0.01, 1.0, 0.1) |
| (, )   (\beta,\eta) (,) for online fine-tuning | (0.5, 1.0) 0.5 1.0 (0.5,1.0) (0.5, 1.0) |

**说明**: TABLE XII: Policy training details for the Hang Chinese Knot task.
## 实验解读

- 评价重点:围绕 behavior-cloning、接触推理、接触丰富操作、模仿学习、language-conditioned-policy,优先看真实机器人结果、扰动恢复、接触稳定性、控制误差、成功率、实时性和 sim-to-real 差距。
- System0 视角:实验结论不只看最终成功率,还要看闭环过程中是否能处理状态估计误差、执行延迟、碰撞/接触突变和外部扰动。
- 与上层系统的关系:如果方法能在快闭环内稳定工作,它可以作为 System1 技能或 System2 规划的底层执行保障;如果只在离线或慢速设定下成立,则还不能直接作为身体级快闭环模块。
## 批判性思考

### 优点

- 论文把 behavior-cloning、接触推理、接触丰富操作、模仿学习、language-conditioned-policy 放到机器人身体级闭环里讨论,和只做高层语义推理的工作相比,更接近真实机器人部署时会遇到的状态、动力学和安全约束。
- 若论文同时给出公式、系统结构图、硬件实验或消融结果,这些证据可以帮助判断方法是否真的能支撑毫秒到百毫秒级反应。

### 局限

- 需要重点核查控制频率、推理延迟、传感延迟、失败恢复、硬件磨损、未见场景泛化和安全边界;这些因素决定它能否成为可靠的 System0 模块。
- 对学习型方法,还要区分训练环境中的鲁棒性和真实部署中的鲁棒性,尤其是接触丰富任务、动态运动任务和人形机器人任务。
## 论文定位

- 研究对象:ConRFT: A Reinforced Fine-tuning Method for VLA Models via Consistency Policy。
- 关键词:behavior-cloning、接触推理、接触丰富操作、模仿学习、language-conditioned-policy、实时控制、强化学习、机器人操作、鲁棒控制、safe-control。
- System0 关联:关注机器人身体级快反应链路中“状态/接触/局部感知 -> 约束/优化/策略 -> 动作”的短闭环,不把问题停留在语言规划或任务分解层。

## 速查卡片

> [!summary] ConRFT
> - **论文**: https://www.roboticsproceedings.org/rss21/p019.pdf
> - **arXiv**: http://arxiv.org/abs/2502.05450v2
> - **arXiv HTML**: https://arxiv.org/html/2502.05450v2
