---
title: "Contact-Grounded Policy: Dexterous Visuotactile Policy with Generative Contact Grounding"
method_name: "Contact-Grounded Policy"
authors: ["Zhengtong Xu"]
year: 2026
venue: "RSS"
tags: ["contact-reasoning", "robot-manipulation", "robust-control", "reinforcement-learning", "contact-rich-manipulation", "robot-generalization", "closed-loop-control", "dexterous-manipulation", "tactile-feedback"]
image_source: "online"
arxiv_html: "https://arxiv.org/html/2603.05687v3"
---
# Contact-Grounded Policy
## 一句话总结

> Contact-Grounded Policy: Dexterous Visuotactile Policy with Generative Contact Grounding 主要落在 [[closed-loop-control]]、[[compliance-control]]、[[contact-estimation]]、[[接触推理]] 这一类问题上;从 System0 视角看,它的价值在于把身体状态、接触变化、局部感知或动力学约束接到机器人低层闭环里,让动作生成更稳定、安全、实时。
## 核心贡献

1. 围绕 **Contact-Grounded Policy: Dexterous Visuotactile Policy with Generative Contact Grounding** 建立了一个与 closed-loop-control、compliance-control、contact-estimation、接触推理、接触丰富操作、灵巧操作 相关的机器人问题设定,重点不是高层语义规划,而是身体级闭环中的状态变化、接触变化或控制约束。
2. 方法层面的关键价值在于把感知、动力学、约束优化、策略学习或硬件反馈组织成可执行的低层动作生成机制,适合放进 System0 的快反应模块中理解。
3. 对系统集成而言,这类工作补上了 System1 技能选择与电机级控制之间的一层:它负责把上层目标变成动态可行、可恢复、可安全过滤的动作。
4. 若论文包含真实机器人、接触任务或仿真到真实迁移,核心应看它是否报告了闭环稳定性、扰动恢复、硬件延迟、失败模式和跨场景泛化。
## 问题背景

这篇论文可以放在 System0 的“具身快系统”背景下阅读:机器人需要在很短时间内根据本体状态、接触、力/触觉、局部视觉或动力学预测做出动作,而不是等待慢速规划器重新推理整个任务。closed-loop-control、compliance-control、contact-estimation、接触推理、接触丰富操作、灵巧操作、diffusion-policy 相关问题的难点通常不在“知道要做什么”,而在动作执行过程中状态会快速偏离、接触会突然改变、约束会实时收紧,低层系统必须及时修正。

### System0 关联

精读时优先看三件事:第一,论文使用哪些快速反馈量,例如关节状态、接触状态、力觉、触觉、深度或局部视觉;第二,反馈如何进入 MPC、WBC、策略网络、安全过滤器或恢复控制器;第三,输出是否直接影响机器人下一步动作,而不是只服务离线分析或高层任务规划。
## 方法详解

### System0 视角解析

从 System0 角度看,这篇论文的方法部分应当围绕 closed-loop-control、compliance-control、contact-estimation、接触推理、接触丰富操作、灵巧操作、diffusion-policy、grasping 来读。核心问题是:系统如何从快速变化的机器人状态中构造控制输入,如何处理接触、碰撞、约束、模型误差或执行延迟,以及如何把这些信息变成下一时刻的动作。

如果方法里有优化问题,重点看目标函数、约束和求解频率;如果方法里有学习策略,重点看观测空间、动作空间、训练扰动和部署时的闭环频率;如果方法里有硬件或传感器,重点看传感链路是否真的进入控制回路。这样读可以把论文和 System2 语言规划、System1 技能编排区分开来。

### 关键公式与机制

#### 公式 1: [[动力学或策略机制]]

$$\mathbf{o}_{t}=\{\mathbf{i}_{t},\mathbf{u}_{t},\mathbf{x}_{t}\}.$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 2: [[优化目标/约束]]

$${\mathbf{a}}_{t}=\mathcal{M}_{\phi}\big(\mathbf{x}_{t},\mathbf{u}_{t}\big).$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 3: [[动力学或策略机制]]

$$\mathbf{O}_{t}=\{\mathbf{o}_{t-T_{o}+1},\ldots,\mathbf{o}_{t}\}.$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 4: [[动力学或策略机制]]

$$\hat{\mathbf{X}}_{t}=\{\hat{\mathbf{x}}_{t+1},\ldots,\hat{\mathbf{x}}_{t+T}\}$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 5: [[动力学或策略机制]]

$$\hat{\mathbf{U}}_{t}=\{\hat{\mathbf{u}}_{t+1},\ldots,\hat{\mathbf{u}}_{t+T}\}$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 6: [[优化目标/约束]]

$$\hat{\mathbf{a}}_{t+k}=\mathcal{M}_{\phi}(\hat{\mathbf{x}}_{t+k},\hat{\mathbf{u}}_{t+k}),\quad k=1,\ldots,T.$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 7: [[动力学或策略机制]]

$$\mathbf{h}_{t}=E(\mathbf{u}_{t}),\hat{\mathbf{u}}_{t}=G(\mathbf{h}_{t}),$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 8: [[动力学或策略机制]]

$$\mathbf{Y}_{t}=\big[\mathbf{x}_{t+1:t+T},~\mathbf{h}_{t+1:t+T}\big],$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 9: [[优化目标/约束]]

$$\mathbf{Y}_{t}^{j}=\alpha_{j}\mathbf{Y}_{t}^{0}+\sigma_{j}\boldsymbol{\epsilon},\boldsymbol{\epsilon}\sim\mathcal{N}(\mathbf{0},\mathbf{I}),$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 10: [[优化目标/约束]]

$$
\mathcal{L}_{\text{diff}}(\theta)=\mathbb{E}_{(\mathbf{O}_{t},\mathbf{Y}_{t}^{0}),\boldsymbol{\epsilon},j}\Big[\lVert \boldsymbol{\epsilon}-\pi_{\theta}(\mathbf{O}_{t},\mathbf{Y}_{t}^{j},j)\rVert^{2}\Big].
$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。
## 关键图表

### Figure 1: Schematic of contact grounding using a 3-DoF revolute finger, illustrating the actual

![Figure 1](https://arxiv.org/html/2603.05687v3/x1.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Schematic of contact grounding using a 3-DoF revolute finger, illustrating the actual”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 2: Overview of Contact-Grounded Policy (CGP). CGP grounds multi-point contacts by predict

![Figure 2](https://arxiv.org/html/2603.05687v3/x2.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Overview of Contact-Grounded Policy (CGP). CGP grounds multi-point contacts by predict”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 3: Teleoperation pipeline. We use a Meta Quest 3 headset for VR-based hand tracking in si

![Figure 3](https://arxiv.org/html/2603.05687v3/x3.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Teleoperation pipeline. We use a Meta Quest 3 headset for VR-based hand tracking in si”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### 论文表格

#### Table 1: TABLE I: Comparison of representative policy paradigms with executable contact modeling.

| Method Category | Executable Contact Modeling | Multi-Finger Hand | Extensible to Distributed Contacts |
| --- | --- | --- | --- |
| Adaptive Compliance Policies [12, 31, 5 ] | ✓ | ✗ | ✗ |
| Sparse Fingertip Force Policies [39, 3 ] | ✓ | ✓ | ✗ |
| Contact-Grounded Policy (Ours) | ✓ | ✓ | ✓ |

**说明**: TABLE I: Comparison of representative policy paradigms with executable contact modeling.

#### Table 2: TABLE II: Success rate 5 challenging contact-rich, dexterous manipulation tasks. DP is short for diffusion policy.

| | Sim | Real | | | |
| --- | --- | --- | --- | --- | --- |
| Method | In-Hand Box Flipping 60 demos | Fragile Egg Grasping 100 demos | Dish Wiping 100 demos | Jar Opening 45 demos | Real In-Hand Box Flipping 90 demos |
| Contact-Grounded Policy | 66.0% | 74.8% | 58.4% | 93.3% | 80.0% |
| Visuotactile DP | 58.0% | 70.0% | 43.6% | 66.7% | 60.0% |
| Visuomotor DP | 53.2% | 53.2% | 42.4% | 73.3% | 60.0% |

**说明**: TABLE II: Success rate 5 challenging contact-rich, dexterous manipulation tasks. DP is short for diffusion policy. For the simulation tasks (left three), we report the mean success rate the last 5 checkpoints, evaluated with 250 rollouts in total. For the real-world tasks, we report the success rate 15 consecutive rollouts.

#### Table 3: TABLE III: Hand configuration prediction results (Section V-B). MAE denotes mean absolute error. Values are reported in

| Input Modality | Tactile Encoder | Abs. Mode ↓ \downarrow | Residual Mode ↓ \downarrow |
| --- | --- | --- | --- |
| State + Tactile | ResNet1D | 8.80 ± 0.24 \pm\,0.24 | 5.94 ± 0.20 \pm\,0.20 |
| | MLP | 12.50 ± 0.32 \pm\,0.32 | 8.33 ± 0.32 \pm\,0.32 |
| | Transformer | 14.39 ± 0.38 \pm\,0.38 | 9.58 ± 0.48 \pm\,0.48 |
| State Only | - | 16.05 ± 0.39 \pm\,0.39 | 10.64 ± 0.38 \pm\,0.38 |
| Tactile Only | ResNet1D | 35.93 ± 0.89 \pm\,0.89 | 12.15 ± 0.20 \pm\,0.20 |
| | MLP | 36.86 ± 0.25 \pm\,0.25 | 12.72 ± 0.25 \pm\,0.25 |
| | Transformer | 43.11 ± 0.91 \pm\,0.91 | 14.62 ± 0.43 \pm\,0.43 |

**说明**: 该表格来自论文中的公式、奖励项或实验设置；为避免 PDF 抽取噪声，网页中仅保留可渲染的核心符号和数值。

#### Table 4: TABLE IV: Tactile reconstruction and compression results on the validation set. MAE stands for mean absolute error. MAE

| | Box | Egg | Dish | | | | | | |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Method | MAE ↓ \downarrow | Active MAE ↓ \downarrow | KL ↓ \downarrow | MAE ↓ \downarrow | Active MAE ↓ \downarrow | KL ↓ \downarrow | MAE ↓ \downarrow | Active MAE ↓ \downarrow | KL ↓ \downarrow |
| MLP (w/ KL) | 1.37 | 12.24 | 0.21 | 0.77 | 6.70 | 0.15 | 1.86 | 8.14 | 0.15 |
| MLP (w/o KL) | 1.12 | 9.93 | 0.59 | 0.54 | 4.51 | 0.40 | 1.40 | 6.20 | 0.42 |
| ResNet1D (w/ KL) | 1.26 | 12.07 | 0.12 | 0.69 | 5.95 | 0.22 | 1.54 | 6.80 | 0.24 |
| ResNet1D (w/o KL) | 0.97 | 9.91 | 0.73 | 0.45 | 3.92 | 0.43 | 1.02 | 4.49 | 0.45 |
| Transformer (w/ KL) | 1.66 | 14.88 | 0.19 | 1.58 | 10.58 | 0.28 | 3.19 | 13.57 | 0.21 |
| Transformer (w/o KL) | 1.69 | 14.60 | 0.97 | 1.18 | 9.08 | 0.61 | 3.28 | 13.57 | 0.53 |

**说明**: 该表格来自论文中的公式、奖励项或实验设置；为避免 PDF 抽取噪声，网页中仅保留可渲染的核心符号和数值。

#### Table 5: TABLE V: Summaries of the task, training, and inference specifications. Here, T T is the prediction horizon, T o T_{o} i

| Task | Domain | Demos | T {T} | T o {T_{o}} | T a {T_{a}} | Frequency | Action | State | Tactile | Vision | Latent | KL Weight |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| In-Hand Box Flipping | Sim | 60 | 16 | 2 | 8 | 5 Hz 8 DDIM steps | 29 | 27 | 748  3 748{\times}3 (Tactile Array) | 2  2{\times} RGB 180  320 180{\times}320 | 32 | 1  10 - 5 1{\times}10^{-5} |
| Fragile Egg Grasping | Sim | 100 | 16 | 2 | 8 | 5 Hz 8 DDIM steps | 29 | 27 | 748  3 748{\times}3 (Tactile Array) | 2  2{\times} RGB 180  320 180{\times}320 | 32 | 1  10 - 5 1{\times}10^{-5} |
| Dish Wiping | Sim | 100 | 16 | 2 | 8 | 5 Hz 8 DDIM steps | 29 | 27 | 748  3 748{\times}3 (Tactile Array) | 2  2{\times} RGB 180  320 180{\times}320 | 32 | 1  10 - 4 1{\times}10^{-4} |
| Jar Opening | Real | 45 | 16 | 2 | 8 | 5 Hz 8 DDIM steps | 25 | 23 | 4  4{\times} RGB 72  72 72{\times}72 (Digit360) | 2  2{\times} RGB 240  320 240{\times}320 | 80 | 5  10 - 5 5{\times}10^{-5} |
| Real In-Hand Box Flipping | Real | 90 | 16 | 2 | 8 | 5 Hz 8 DDIM steps | 25 | 23 | 4  4{\times} RGB 72  72 72{\times}72 (Digit360) | 2  2{\times} RGB 240  320 240{\times}320 | 80 | 5  10 - 5 5{\times}10^{-5} |

#### Table 6: TABLE VI: Validation results for tactile compression on real in-hand box flipping and jar opening tasks. MAE denotes mea

| Latent Dim | Model | Box | Jar | | | | | | |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MAE ↓ \downarrow | KL Loss ↓ \downarrow | PSNR (dB) ↑ \uparrow | SSIM ↑ \uparrow | MAE ↓ \downarrow | KL Loss ↓ \downarrow | PSNR (dB) ↑ \uparrow | SSIM ↑ \uparrow | | |
| 80 | w/ KL |.01 |.0694 |.2050 |.9870 |.10 |.0647 |.9445 |.9883 |
| w/o KL | 4.55 | 0.7781 | 40.4590 | 0.9939 | 3.88 | 0.6247 | 42.9845 | 0.9957 | |
| 160 | w/ KL | 9.41 | 0.0341 | 35.2768 | 0.9874 | 8.40 | 0.0337 | 36.5394 | 0.9891 |
| w/o KL | 4.66 | 0.7902 | 40.4800 | 0.9941 | 3.98 | 0.6899 | 42.9143 | 0.9957 | |
| 320 | w/ KL | 9.02 | 0.0171 | 35.2693 | 0.9870 | 8.29 | 0.0155 | 36.1177 | 0.9885 |
| w/o KL | 4.56 | 0.7539 | 40.5191 | 0.9941 | 3.84 | 0.6787 | 43.0426 | 0.9957 | |

**说明**: 该表格来自论文中的公式、奖励项或实验设置；为避免 PDF 抽取噪声，网页中仅保留可渲染的核心符号和数值。
## 实验解读

- 评价重点:围绕 closed-loop-control、compliance-control、contact-estimation、接触推理、接触丰富操作,优先看真实机器人结果、扰动恢复、接触稳定性、控制误差、成功率、实时性和 sim-to-real 差距。
- System0 视角:实验结论不只看最终成功率,还要看闭环过程中是否能处理状态估计误差、执行延迟、碰撞/接触突变和外部扰动。
- 与上层系统的关系:如果方法能在快闭环内稳定工作,它可以作为 System1 技能或 System2 规划的底层执行保障;如果只在离线或慢速设定下成立,则还不能直接作为身体级快闭环模块。
## 批判性思考

### 优点

- 论文把 closed-loop-control、compliance-control、contact-estimation、接触推理、接触丰富操作 放到机器人身体级闭环里讨论,和只做高层语义推理的工作相比,更接近真实机器人部署时会遇到的状态、动力学和安全约束。
- 若论文同时给出公式、系统结构图、硬件实验或消融结果,这些证据可以帮助判断方法是否真的能支撑毫秒到百毫秒级反应。

### 局限

- 需要重点核查控制频率、推理延迟、传感延迟、失败恢复、硬件磨损、未见场景泛化和安全边界;这些因素决定它能否成为可靠的 System0 模块。
- 对学习型方法,还要区分训练环境中的鲁棒性和真实部署中的鲁棒性,尤其是接触丰富任务、动态运动任务和人形机器人任务。
## 论文定位

- 研究对象:Contact-Grounded Policy: Dexterous Visuotactile Policy with Generative Contact Grounding。
- 关键词:closed-loop-control、compliance-control、contact-estimation、接触推理、接触丰富操作、灵巧操作、diffusion-policy、grasping、in-hand-manipulation、机器人操作。
- System0 关联:关注机器人身体级快反应链路中“状态/接触/局部感知 -> 约束/优化/策略 -> 动作”的短闭环,不把问题停留在语言规划或任务分解层。

## 速查卡片

> [!summary] Contact-Grounded Policy
> - **论文**: https://www.roboticsproceedings.org/rss22/p005.pdf
> - **arXiv**: http://arxiv.org/abs/2603.05687v3
> - **arXiv HTML**: https://arxiv.org/html/2603.05687v3
