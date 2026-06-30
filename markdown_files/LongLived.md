---
title: "Towards Long-Lived Robots: Continual Learning VLA Models via Reinforcement Fine-Tuning"
method_name: "Towards Long-Lived Robots"
authors: ["Yuan Liu"]
year: 2026
venue: "RSS"
tags: ["real-time-control", "reinforcement-learning", "safe-control", "adaptive-control", "robot-generalization", "closed-loop-control"]
image_source: "online"
arxiv_html: "https://arxiv.org/html/2602.10503v2"
---
# Towards Long-Lived Robots
## 一句话总结

> Towards Long-Lived Robots: Continual Learning VLA Models via Reinforcement Fine-Tuning 主要落在 [[action-chunking]]、[[adaptive-control]]、[[certified-control]]、[[closed-loop-control]] 这一类问题上;从 System0 视角看,它的价值在于把身体状态、接触变化、局部感知或动力学约束接到机器人低层闭环里,让动作生成更稳定、安全、实时。
## 核心贡献

1. 围绕 **Towards Long-Lived Robots: Continual Learning VLA Models via Reinforcement Fine-Tuning** 建立了一个与 action-chunking、adaptive-control、certified-control、closed-loop-control、实时控制、强化学习 相关的机器人问题设定,重点不是高层语义规划,而是身体级闭环中的状态变化、接触变化或控制约束。
2. 方法层面的关键价值在于把感知、动力学、约束优化、策略学习或硬件反馈组织成可执行的低层动作生成机制,适合放进 System0 的快反应模块中理解。
3. 对系统集成而言,这类工作补上了 System1 技能选择与电机级控制之间的一层:它负责把上层目标变成动态可行、可恢复、可安全过滤的动作。
4. 若论文包含真实机器人、接触任务或仿真到真实迁移,核心应看它是否报告了闭环稳定性、扰动恢复、硬件延迟、失败模式和跨场景泛化。
## 问题背景

这篇论文可以放在 System0 的“具身快系统”背景下阅读:机器人需要在很短时间内根据本体状态、接触、力/触觉、局部视觉或动力学预测做出动作,而不是等待慢速规划器重新推理整个任务。action-chunking、adaptive-control、certified-control、closed-loop-control、实时控制、强化学习、robot-generalization 相关问题的难点通常不在“知道要做什么”,而在动作执行过程中状态会快速偏离、接触会突然改变、约束会实时收紧,低层系统必须及时修正。

### System0 关联

精读时优先看三件事:第一,论文使用哪些快速反馈量,例如关节状态、接触状态、力觉、触觉、深度或局部视觉;第二,反馈如何进入 MPC、WBC、策略网络、安全过滤器或恢复控制器;第三,输出是否直接影响机器人下一步动作,而不是只服务离线分析或高层任务规划。
## 方法详解

### System0 视角解析

从 System0 角度看,这篇论文的方法部分应当围绕 action-chunking、adaptive-control、certified-control、closed-loop-control、实时控制、强化学习、robot-generalization、scalable-robot-learning 来读。核心问题是:系统如何从快速变化的机器人状态中构造控制输入,如何处理接触、碰撞、约束、模型误差或执行延迟,以及如何把这些信息变成下一时刻的动作。

如果方法里有优化问题,重点看目标函数、约束和求解频率;如果方法里有学习策略,重点看观测空间、动作空间、训练扰动和部署时的闭环频率;如果方法里有硬件或传感器,重点看传感链路是否真的进入控制回路。这样读可以把论文和 System2 语言规划、System1 技能编排区分开来。

### 关键公式与机制

#### 公式 1: [[动力学或策略机制]]

$$A_{i}=\frac{r_{i}-\mathrm{mean}(\{r_{1},\dots,r_{G}\})}{\mathrm{std}(\{r_{1},\dots,r_{G}\})}.$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 2: [[优化目标/约束]]

$$\text{QACR}\leftarrow\dfrac{\sum_{\ell=1}^{\min(U,V)}\mathbb{I}(a_{\ell}=\tilde{a}_{\ell})}{\max(U,V)}$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 3: [[动力学或策略机制]]

$$\tilde{\mathbf{y}}:=(\tilde{\mathbf{y}}^{\text{pose}},\tilde{\mathbf{y}}^{\text{grip}})\leftarrow\text{Decode}(\tilde{\mathbf{a}})$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 4: [[动力学或策略机制]]

$$d_{t}\leftarrow\dfrac{1}{\text{dim}(\mathbf{y}_{t}^{\text{pose}})}\lVert \mathbf{y}_{t}^{\text{pose}}-\tilde{\mathbf{y}}_{t}^{\text{pose}} \rVert_{1}$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 5: [[动力学或策略机制]]

$$r_{t}^{\text{grip}}\leftarrow\mathbb{I}(\mathbf{y}_{t}^{\text{grip}}=\tilde{\mathbf{y}}_{t}^{\text{grip}})$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 6: [[优化目标/约束]]

$$\text{QACR}=\begin{cases}\frac{\sum_{\ell=1}^{\min(U,V)}\mathbb{I}(a_{\ell}=\tilde{a}_{\ell})}{\max(U,V)},&\text{if valid}\\[15.00002pt] 0,&\text{otherwise}\end{cases}$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 7: [[动力学或策略机制]]

$$\text{CTAR}=\begin{cases}\frac{1}{H}\sum_{t=1}^{H}\left(\beta\cdot r_{t}^{\text{pose}}+(1-\beta)\cdot r_{t}^{\text{grip}}\right),&\text{if valid}\\[11.99998pt] 0,&\text{otherwise}\end{cases}$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 8: [[优化目标/约束]]

$$\quad\frac{1}{G}\sum_{i=1}^{G}\{\min[\frac{\pi_{\theta}(\mathbf{a}_{i}| o,l)}{ $\pi_{\theta_{\text{old}}}(\mathbf{a}_{i}$ |o,l)}A_{i},$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 9: [[优化目标/约束]]

$$=\mathbb{E}_{(o,l)\sim\mathcal{B},\{\mathbf{a}_{i}\}_{i=1}^{G}\sim\pi_{\theta_{\text{old}}}(\cdot|o,l)}$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 10: [[动力学或策略机制]]

$$\quad\text{clip}\left(\frac{\pi_{\theta}(\mathbf{a}_{i}| o,l)}{ $\pi_{\theta_{\text{old}}}(\mathbf{a}_{i}$ |o,l)},1-\epsilon,1+\epsilon\right)A_{i}]$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。
## 关键图表

### Figure 1: Overview of the proposed LifeLong-RFT. This strategy integrates the chunking-level on

![Figure 1](https://arxiv.org/html/2602.10503v2/x2.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Overview of the proposed LifeLong-RFT. This strategy integrates the chunking-level on”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 2: Overview of real-world experimental tasks: Pick & Place (Banana, Bread), Pull Drawer

![Figure 2](https://arxiv.org/html/2602.10503v2/x3.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Overview of real-world experimental tasks: Pick & Place (Banana, Bread), Pull Drawer”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 3: A representative execution of the Hang Chinese Knot task

![Figure 3](https://arxiv.org/html/2602.10503v2/x10.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“A representative execution of the Hang Chinese Knot task”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### 论文表格

#### Table 1: TABLE I: Multi-Task learning performance on SimplerEnv. Figure 3: Overview of real-world experimental tasks: Pick & Plac

| Method | Training Strategy | WidowX (Visual Matching) | Google Robot (Visual Matching) | | | | | | | |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Put Carrot | Stack | Put Spoon | Put Eggplant | Avg | Pick Coke | Move | Open/Close | Avg | | |
| on Plate | Blocks | on Towel | in Basket | Can | Near | Drawer | | | | |
| Continuous Action Models | | | | | | | | | | |
| Octo-Base [65 ] | SFT | 8.3 | 0.0 | 12.5 | 43.1 | 16.0 | 17.0 | 4.2 | 22.7 | 16.8 |
| RoboVLM [41 ] | SFT | 25.0 | 12.5 | 29.2 | 58.3 | 31.3 | 77.3 | 61.7 | 43.5 | 63.4 |
| GR00T N1.5 [52 ] | SFT | – | – | – | – | – | 69.3 | 68.7 | 35.8 | 52.4 |
| $\pi_{0} [7 ]$ | SFT | 58.8 | 21.3 | 63.3 | 79.2 | 55.7 | 72.7 | 65.3 | 38.3 | 58.7 |
| ThinkAct [23 ] | SFT + RFT | 37.5 | 8.7 | 58.3 | 70.8 | 43.8 | 92.0 | 72.4 | 50.0 | 71.5 |
| NORA-1.5 [25 ] | SFT | – | – | – | – | – | 92.8 | 78.7 | 62.2 | 77.9 |
| NORA-1.5 [25 ] (DPO) | SFT+RFT | – | – | – | – | – | 94.0 | 88.0 | 66.4 | 82.8 |
| Discrete Action Models | | | | | | | | | | |
| TraceVLA [79 ] | SFT | – | – | – | – | – | 28.0 | 53.7 | 57.0 | 42.0 |
| RT-1-X [8 ] | SFT | 4.2 | 0.0 | 0.0 | 0.0 | 1.1 | 56.7 | 31.7 | 59.7 | 53.4 |
| OpenVLA [30 ] | SFT | 0.0 | 0.0 | 0.0 | 4.1 | 1.0 | 16.3 | 46.2 | 35.6 | 27.7 |
| SpatialVLA [56 ] | SFT | 25.0 | 29.2 | 16.7 | 100.0 | 42.7 | 86.0 | 77.9 | 57.4 | 73.7 |
| $\pi_{0} -FAST [55 ]$ | SFT | 22.0 | 83.0 | 29.0 | 48.0 | 45.5 | 75.3 | 67.5 | 42.6 | 61.9 |
| NORA-1.5-FAST [25 ] | SFT | – | – | – | – | – | 88.6 | 86.4 | 41.2 | 72.1 |
| NORA-Long [26 ] (Baseline) | SFT | 46.0 | 60.3 | 80.2 | 75.7 | 65.5 | 86.0 | 82.3 | 56.0 | 74.7 |
| NORA-Long [26 ] | RFT (Ours) | 50.2 | 64.4 | 84.3 | 77.0 | 69.0 | 94.0 | 84.7 | 58.5 | 79.1 |
| $\boldsymbol{\Delta}$ | – | +4.2 | +4.1 | +4.1 | +1.3 | +3.5 | +8.0 | +2.4 | +2.5 | + 4.4 |
| | FCR = {1, if valid 0, otherwise  $\text{FCR}=\begin{cases}1,&\text{if valid}\\ 0,&\text{otherwise}\end{cases}$ | | (5) | | | | | | | |
| | $\omega\cdot\text{QACR}+(1-\omega)\cdot\text{CTAR}+\lambda\cdot\text{FCR},$ | | (6) | | | | | | | |

**说明**: TABLE I: Multi-Task learning performance on SimplerEnv. Figure 3: Overview of real-world experimental tasks: Pick & Place (Banana, Bread), Pull Drawer, and Hang Chinese Knot.

#### Table 2: TABLE II: Multi-Task learning performance on LIBERO.

| Method | Training Strategy | LIBERO | Avg | | | |
| --- | --- | --- | --- | --- | --- | --- |
| Object | Spatial | Goal | Long | | | |
| Continuous Action Models | | | | | | |
| Octo-Base [65 ] | SFT | 85.7 | 78.9 | 84.6 | 51.1 | 75.1 |
| GR00T N1 [5 ] | SFT | 97.6 | 94.4 | 93.0 | 90.6 | 93.9 |
| $\pi_{0} [7 ]$ | SFT | 98.8 | 96.8 | 95.8 | 85.2 | 94.2 |
| OpenVLA-OFT [29 ] | SFT | 98.1 | 96.9 | 95.5 | 91.1 | 95.4 |
| ThinkAct [23 ] | SFT + RFT | 91.4 | 88.3 | 87.1 | 70.9 | 84.4 |
| VLA-RFT [37 ] | SFT + RFT | 94.4 | 94.4 | 95.4 | 80.2 | 91.1 |
| NORA-1.5 [25 ] | SFT | 96.4 | 97.3 | 94.5 | 89.6 | 94.5 |
| NORA-1.5 [25 ] (DPO) | SFT + RFT | 96.0 | 98.0 | 95.4 | 90.5 | 95.0 |
| Discrete Action Models | | | | | | |
| TraceVLA [79 ] | SFT | 85.2 | 84.6 | 75.1 | 54.1 | 74.8 |
| OpenVLA [30 ] | SFT | 88.4 | 84.7 | 79.2 | 53.7 | 76.5 |
| SpatialVLA [56 ] | SFT | 89.9 | 88.2 | 78.6 | 55.5 | 78.1 |
| CoT-VLA [77 ] | SFT | 91.6 | 87.5 | 87.6 | 69.0 | 83.9 |
| WorldVLA [9 ] | SFT | 96.2 | 87.6 | 83.4 | 60.0 | 79.1 |
| $\pi_{0} -Fast [55 ]$ | SFT | 96.8 | 96.4 | 88.6 | 60.2 | 85.5 |
| MolmoAct-7B-D [33 ] | SFT | 95.4 | 87.0 | 87.6 | 77.2 | 86.6 |
| TGRPO [17 ] | SFT + RFT | 92.2 | 90.4 | 81.0 | 59.2 | 80.7 |
| NORA-Long [26 ] (Baseline) | SFT | 97.5 | 96.4 | 91.0 | 82.4 | 91.8 |
| NORA-Long [26 ] | RFT (Ours) | 99.2 | 98.2 | 95.8 | 89.0 | 95.6 |
| $\boldsymbol{\Delta}$ | – | +1.7 | +1.8 | +4.8 | +6.6 | +3.8 |

**说明**: TABLE II: Multi-Task learning performance on LIBERO.

#### Table 3: TABLE III: Multi-Task learning performance on real-world.

| Task Split | $\boldsymbol{\pi_{0}} [7 ]$ | OpenVLA [30 ] | NORA-Long [25 ] | | |
| --- | --- | --- | --- | --- | --- |
| SFT | SFT | SFT | RFT (Ours) | $\boldsymbol{\Delta}$ | |
| Pick Banana | 90.0 | 75.0 | 85.0 | 90.0 | +5.0 |
| Pick Bread | 75.0 | 70.0 | 75.0 | 85.0 | +10.0 |
| Pull Drawer | 95.0 | 85.0 | 95.0 | 100.0 | +5.0 |
| Hang Chinese Knot | 65.0 | 55.0 | 60.0 | 75.0 | +15.0 |
| Overall | 81.3 | 71.3 | 78.8 | 87.5 | +8.7 |

**说明**: TABLE III: Multi-Task learning performance on real-world.

#### Table 4: TABLE IV: Continual learning performance on LIBERO.

| Task Split | Metrics | BUDS [81 ] | LOTUS [67 ] | SPECI [71 ] | $\boldsymbol{\pi_{0}} [7 ]$ | OpenVLA [30 ] | OpenVLA-OFT [29 ] | NORA-Long [26 ] | | |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BC | BC | BC | SFT | SFT | SFT | SFT | RFT (Ours) | $\boldsymbol{\Delta}$ | | |
| LIBERO-Object | FWT (↑ \uparrow) | 52.0 | 74.0 | 83.0 | 73.0 | 59.4 | 89.8 | 84.8 | 96.0 | +11.2 |
| NBT (↓ \downarrow) | 21.0 | 11.0 | 10.0 | 16.2 | 17.9 | 3.1 | 6.8 | 1.5 | -5.3 | |
| | AUC (↑ \uparrow) | 47.0 | 65.0 | 78.0 | 59.3 | 45.1 | 87.4 | 79.7 | 94.8 | +15.1 |
| LIBERO-Spatial | FWT (↑ \uparrow) | – | – | 67.0 | 74.4 | 64.2 | 88.6 | 82.8 | 94.0 | +11.2 |
| NBT (↓ \downarrow) | – | – | 6.0 | 23.7 | 17.6 | 9.4 | 14.0 | 3.7 | -10.3 | |
| | AUC (↑ \uparrow) | – | – | 66.0 | 55.5 | 50.8 | 81.7 | 71.7 | 91.2 | +19.5 |
| | FWT (↑ \uparrow) | 50.0 | 61.0 | 74.0 | 74.6 | 58.6 | 90.2 | 72.8 | 92.4 | +19.6 |
| LIBERO-Goal | NBT (↓ \downarrow) | 39.0 | 30.0 | 20.0 | 23.9 | 5.8 | 13.8 | 25.2 | 3.1 | -22.1 |
| | AUC (↑ \uparrow) | 42.0 | 56.0 | 65.0 | 56.3 | 53.5 | 79.2 | 54.4 | 90.3 | +35.9 |
| LIBERO-Long | FWT (↑ \uparrow) | – | – | 58.0 | 53.8 | 32.0 | 64.0 | 61.0 | 74.2 | +13.2 |
| NBT (↓ \downarrow) | – | – | 21.0 | 14.2 | 14.1 | 31.4 | 17.3 | 12.8 | -4.5 | |
| | AUC (↑ \uparrow) | – | – | 46.0 | 42.5 | 20.8 | 38.7 | 47.3 | 64.5 | +17.2 |

**说明**: TABLE IV: Continual learning performance on LIBERO.

#### Table 5: TABLE V: Continual learning performance on real-world.

| Task Split | Metrics | $\boldsymbol{\pi_{0}} [7 ]$ | OpenVLA [30 ] | NORA-Long [26 ] | | |
| --- | --- | --- | --- | --- | --- | --- |
| SFT | SFT | SFT | RFT (Ours) | $\boldsymbol{\Delta}$ | | |
| Real-World | FWT (↑ \uparrow) | 58.8 | 46.3 | 56.3 | 80.0 | +23.7 |
| NBT (↓ \downarrow) | 16.3 | 17.8 | 18.3 | 6.1 | -12.2 | |
| | AUC (↑ \uparrow) | 47.9 | 35.1 | 44.2 | 75.9 | +31.7 |

**说明**: TABLE V: Continual learning performance on real-world.

#### Table 6: TABLE VI: Ablation of the multi-dimensional process reward.

| Settings | Object | Spatial | Goal | Long | Avg | | | | | |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SR | $\boldsymbol{\Delta}$ | SR | $\boldsymbol{\Delta}$ | SR | $\boldsymbol{\Delta}$ | SR | $\boldsymbol{\Delta}$ | SR | $\boldsymbol{\Delta}$ | |
| w/o QACR | 97.0 | -2.2 | 96.4 | -1.8 | 92.2 | -3.6 | 85.6 | -3.4 |.8 | -2.8 |
| w/o CTAR | 8.0 | -91.2 | 6.2 | -92.0 | 2.4 | -93.4 | 2.0 | -87.0 |.7 | -90.9 |
| w/o FCR | 98.0 | -1.2 | 96.2 | -2.0 | 93.2 | -2.6 | 84.6 | -4.4 |.0 | -2.6 |
| RFT (Ours) | 99.2 | - | 98.2 | - | 95.8 | - | 89.0 | - | 95.6 | - |

**说明**: TABLE VI: Ablation of the multi-dimensional process reward.

#### Table 7: TABLE VII: Multi-Task learning settings on SimplerEnv.

| Hyperparameter | WidowX | Google Robot |
| --- | --- | --- |
| Platform-Specific Settings | | |
| Global Batch Size | 512 | 1024 |
| Epochs | 30 | 40 |
| Shared Settings | | |
| Learning Rate | 1  10 - 6 1\times 10^{-6} | |
| Optimizer | AdamW [44 ] | |
| Group Size | 8 | |
| Temperature | 0.8 | |
| $\omega,\lambda,\gamma)$ | (5, 0.8, 0.7, 0.1, 0.001) (5,0.8,0.7,0.1,0.001) | |

**说明**: TABLE VII: Multi-Task learning settings on SimplerEnv.

#### Table 8: TABLE VIII: Multi-Task learning settings on LIBERO.

| Hyperparameter | Object / Spatial / Goal | Long |
| --- | --- | --- |
| Task-Specific Settings | | |
| Global Batch Size | 128 | 256 |
| Epochs | 15 | 35 |
| Shared Settings | | |
| Learning Rate | 1  10 - 6 1\times 10^{-6} | |
| Optimizer | AdamW [44 ] | |
| Group Size | 8 | |
| Temperature | 0.8 | |
| $\omega,\lambda,\gamma)$ | (5, 0.8, 0.7, 0.1, 0.001) (5,0.8,0.7,0.1,0.001) | |

**说明**: TABLE VIII: Multi-Task learning settings on LIBERO.

#### Table 9: TABLE IX: Multi-Task learning settings on real-world tasks.

| Hyperparameter | Real-World |
| --- | --- |
| Shared Settings | |
| Global Batch Size | 128 |
| Epochs | 20 |
| Learning Rate | 1  10 - 6 1\times 10^{-6} |
| Optimizer | AdamW [44 ] |
| Group Size | 8 |
| Temperature | 0.8 |
| $\omega,\lambda,\gamma)$ | (5, 0.8, 0.7, 0.1, 0.001) (5,0.8,0.7,0.1,0.001) |

**说明**: TABLE IX: Multi-Task learning settings on real-world tasks.

#### Table 10: TABLE X: Continual learning settings for LIBERO and real-world experiments.

| Hyperparameter | LIBERO / Real-World |
| --- | --- |
| Shared Settings | |
| Global Batch Size | 32 |
| Epochs | 10 |
| Learning Rate | 1  10 - 6 1\times 10^{-6} |
| Optimizer | AdamW [44 ] |
| Group Size | 8 |
| Temperature | 0.8 |
| $\omega,\lambda,\gamma)$ | (5, 0.8, 0.7, 0.1, 0.001) (5,0.8,0.7,0.1,0.001) |

**说明**: TABLE X: Continual learning settings for LIBERO and real-world experiments.

#### Table 11: TABLE XI: Detailed continual learning results on four LIBERO task suites (Object, Spatial, Goal, and Long).

| Task Split | LIBERO-Object | LIBERO-Spatial | | | | | | | | | | | | | | | | | | |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| T-1 | T-2 | T-3 | T-4 | T-5 | T-6 | T-7 | T-8 | T-9 | T-10 | T-1 | T-2 | T-3 | T-4 | T-5 | T-6 | T-7 | T-8 | T-9 | T-10 | |
| Base Task Stage | | | | | | | | | | | | | | | | | | | | |
| Base Task 1-6 | 100% | 100% | 100% | 98% | 98% | 100% | – | – | – | – | 90% | 100% | 98% | 98% | 96% | 84% | – | – | – | – |
| LifeLong Learning Stage | | | | | | | | | | | | | | | | | | | | |
| New Task 7 | 92% | 96% | 98% | 96% | 98% | 100% | 96% | – | – | – | 94% | 92% | 98% | 84% | 94% | 96% | 100% | – | – | – |
| New Task 8 | 98% | 100% | 94% | 98% | 96% | 100% | 100% | 82% | – | – | 100% | 97% | 100% | 94% | 86% | 92% | 98% | 90% | – | – |
| New Task 9 | 96% | 96% | 96% | 86% | 96% | 100% | 98% | 92% | 96% | – | 70% | 80% | 98% | 92% | 92% | 88% | 96% | 94% | 90% | – |
| New Task 10 | 94% | 100% | 100% | 96% | 96% | 94% | 100% | 76% | 92% | 90% | 78% | 98% | 98% | 88% | 88% | 92% | 80% | 62% | 92% | 94% |
| Task Split | LIBERO-Goal | LIBERO-Long | | | | | | | | | | | | | | | | | | |
| T-1 | T-2 | T-3 | T-4 | T-5 | T-6 | T-7 | T-8 | T-9 | T-10 | T-1 | T-2 | T-3 | T-4 | T-5 | T-6 | T-7 | T-8 | T-9 | T-10 | |
| Base Task Stage | | | | | | | | | | | | | | | | | | | | |
| Base Task 1-6 | 100% | 98% | 94% | 86% | 94% | 96% | – | – | – | – | 78% | 86% | 92% | 96% | 88% | 92% | – | – | – | – |
| LifeLong Learning Stage | | | | | | | | | | | | | | | | | | | | |
| New Task 7 | 90% | 90% | 86% | 88% | 98% | 94% | 72% | – | – | – | 58% | 78% | 74% | 94% | 44% | 86% | 36% | – | – | – |
| New Task 8 | 88% | 96% | 90% | 76% | 96% | 90% | 80% | 100% | – | – | 52% | 70% | 60% | 84% | 30% | 80% | 44% | 82% | – | – |
| New Task 9 | 94% | 94% | 98% | 80% | 94% | 96% | 82% | 98% | 100% | – | 60% | 70% | 82% | 88% | 44% | 94% | 50% | 80% | 34% | – |
| New Task 10 | 86% | 100% | 92% | 80% | 98% | 90% | 78% | 96% | 86% | 84% | 58% | 80% | 70% | 82% | 38% | 88% | 38% | 76% | 18% | 58% |

**说明**: TABLE XI: Detailed continual learning results on four LIBERO task suites (Object, Spatial, Goal, and Long).

#### Table 12: TABLE XII: Detailed continual learning results in real-world experiments.

| Task Split | Pick Banana | Pick Bread | Pull Drawer | Hang Chinese Knot |
| --- | --- | --- | --- | --- |
| LifeLong Learning Stage | | | | |
| New Task 1 | 85% | – | – | – |
| New Task 2 | 80% | 75% | – | – |
| New Task 3 | 70% | 65% | 100% | – |
| New Task 4 | 70% | 70% | 95% | 60% |

**说明**: TABLE XII: Detailed continual learning results in real-world experiments.

#### Table 13: TABLE XIII: Continual learning performance on LIBERO-Goal during the lifelong learning stage.

| Task Split | T-1 | T-2 | T-3 | T-4 | T-5 | T-6 | T-7 | T-8 | T-9 | T-10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LifeLong Learning Stage | | | | | | | | | | |
| New Task 1 | 48% | – | – | – | – | – | – | – | – | – |
| New Task 2 | 44% | 76% | – | – | – | – | – | – | – | – |
| New Task 3 | 30% | 48% | 94% | – | – | – | – | – | – | – |
| New Task 4 | 54% | 56% | 96% | 86% | – | – | – | – | – | – |
| New Task 5 | 48% | 56% | 98% | 82% | 98% | – | – | – | – | – |
| New Task 6 | 38% | 74% | 88% | 76% | 72% | 90% | – | – | – | – |
| New Task 7 | 40% | 72% | 54% | 78% | 76% | 76% | 54% | – | – | – |
| New Task 8 | 44% | 76% | 68% | 62% | 80% | 72% | 60% | 100% | – | – |
| New Task 9 | 26% | 84% | 88% | 74% | 96% | 86% | 60% | 100% | 96% | – |
| New Task 10 | 34% | 76% | 88% | 70% | 94% | 80% | 64% | 100% | 98% | 70% |

**说明**: TABLE XIII: Continual learning performance on LIBERO-Goal during the lifelong learning stage.

#### Table 14: TABLE XIV: Continual learning performance of RoboBrain-X0 on LIBERO.

| Task Split | Metrics | RoboBrain-X0 | | |
| --- | --- | --- | --- | --- |
| SFT | RFT (Ours) | $\boldsymbol{\Delta}$ | | |
| LIBERO-Object | FWT (↑ \uparrow) | 86.6 | 94.0 | +7.4 |
| NBT (↓ \downarrow) | 16.1 | 1.5 | -14.6 | |
| | AUC (↑ \uparrow) | 74.7 | 92.4 | +17.7 |
| LIBERO-Spatial | FWT (↑ \uparrow) | 80.4 | 92.6 | +12.2 |
| NBT (↓ \downarrow) | 21.0 | 7.7 | -13.3 | |
| | AUC (↑ \uparrow) | 64.5 | 86.9 | +22.4 |
| LIBERO-Goal | FWT (↑ \uparrow) | 68.8 | 84.0 | +15.2 |
| NBT (↓ \downarrow) | 16.0 | 13.1 | -2.9 | |
| | AUC (↑ \uparrow) | 56.5 | 73.8 | +17.3 |
| LIBERO-Long | FWT (↑ \uparrow) | 62.8 | 64.4 | +1.6 |
| NBT (↓ \downarrow) | 28.1 | 15.5 | -12.6 | |
| | AUC (↑ \uparrow) | 40.8 | 52.0 | +11.2 |

**说明**: TABLE XIV: Continual learning performance of RoboBrain-X0 on LIBERO.
## 实验解读

- 评价重点:围绕 action-chunking、adaptive-control、certified-control、closed-loop-control、实时控制,优先看真实机器人结果、扰动恢复、接触稳定性、控制误差、成功率、实时性和 sim-to-real 差距。
- System0 视角:实验结论不只看最终成功率,还要看闭环过程中是否能处理状态估计误差、执行延迟、碰撞/接触突变和外部扰动。
- 与上层系统的关系:如果方法能在快闭环内稳定工作,它可以作为 System1 技能或 System2 规划的底层执行保障;如果只在离线或慢速设定下成立,则还不能直接作为身体级快闭环模块。
## 批判性思考

### 优点

- 论文把 action-chunking、adaptive-control、certified-control、closed-loop-control、实时控制 放到机器人身体级闭环里讨论,和只做高层语义推理的工作相比,更接近真实机器人部署时会遇到的状态、动力学和安全约束。
- 若论文同时给出公式、系统结构图、硬件实验或消融结果,这些证据可以帮助判断方法是否真的能支撑毫秒到百毫秒级反应。

### 局限

- 需要重点核查控制频率、推理延迟、传感延迟、失败恢复、硬件磨损、未见场景泛化和安全边界;这些因素决定它能否成为可靠的 System0 模块。
- 对学习型方法,还要区分训练环境中的鲁棒性和真实部署中的鲁棒性,尤其是接触丰富任务、动态运动任务和人形机器人任务。
## 论文定位

- 研究对象:Towards Long-Lived Robots: Continual Learning VLA Models via Reinforcement Fine-Tuning。
- 关键词:action-chunking、adaptive-control、certified-control、closed-loop-control、实时控制、强化学习、robot-generalization、scalable-robot-learning、vision-language-action、vla。
- System0 关联:关注机器人身体级快反应链路中“状态/接触/局部感知 -> 约束/优化/策略 -> 动作”的短闭环,不把问题停留在语言规划或任务分解层。

## 速查卡片

> [!summary] Towards Long-Lived Robots
> - **论文**: https://www.roboticsproceedings.org/rss22/p086.pdf
> - **arXiv**: http://arxiv.org/abs/2602.10503v2
> - **arXiv HTML**: https://arxiv.org/html/2602.10503v2
