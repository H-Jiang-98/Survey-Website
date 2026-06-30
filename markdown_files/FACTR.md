---
title: "FACTR: Force-Attending Curriculum Training for Contact-Rich Policy Learning"
method_name: "FACTR"
authors: ["Jason Jingzhou Liu"]
year: 2025
venue: "RSS"
tags: ["contact-reasoning", "robust-control", "reinforcement-learning", "imitation-learning", "contact-rich-manipulation", "robot-generalization", "closed-loop-control", "tactile-feedback"]
image_source: "online"
arxiv_html: "https://arxiv.org/html/2502.17432v2"
---
# FACTR
## 一句话总结

> FACTR: Force-Attending Curriculum Training for Contact-Rich Policy Learning 主要落在 [[closed-loop-control]]、[[接触推理]]、[[接触丰富操作]]、[[external-force-perception]] 这一类问题上;从 System0 视角看,它的价值在于把身体状态、接触变化、局部感知或动力学约束接到机器人低层闭环里,让动作生成更稳定、安全、实时。
## 核心贡献

1. 围绕 **FACTR: Force-Attending Curriculum Training for Contact-Rich Policy Learning** 建立了一个与 closed-loop-control、接触推理、接触丰富操作、external-force-perception、力控制、模仿学习 相关的机器人问题设定,重点不是高层语义规划,而是身体级闭环中的状态变化、接触变化或控制约束。
2. 方法层面的关键价值在于把感知、动力学、约束优化、策略学习或硬件反馈组织成可执行的低层动作生成机制,适合放进 System0 的快反应模块中理解。
3. 对系统集成而言,这类工作补上了 System1 技能选择与电机级控制之间的一层:它负责把上层目标变成动态可行、可恢复、可安全过滤的动作。
4. 若论文包含真实机器人、接触任务或仿真到真实迁移,核心应看它是否报告了闭环稳定性、扰动恢复、硬件延迟、失败模式和跨场景泛化。
## 问题背景

这篇论文可以放在 System0 的“具身快系统”背景下阅读:机器人需要在很短时间内根据本体状态、接触、力/触觉、局部视觉或动力学预测做出动作,而不是等待慢速规划器重新推理整个任务。closed-loop-control、接触推理、接触丰富操作、external-force-perception、力控制、模仿学习、policy-learning 相关问题的难点通常不在“知道要做什么”,而在动作执行过程中状态会快速偏离、接触会突然改变、约束会实时收紧,低层系统必须及时修正。

### System0 关联

精读时优先看三件事:第一,论文使用哪些快速反馈量,例如关节状态、接触状态、力觉、触觉、深度或局部视觉;第二,反馈如何进入 MPC、WBC、策略网络、安全过滤器或恢复控制器;第三,输出是否直接影响机器人下一步动作,而不是只服务离线分析或高层任务规划。
## 方法详解

### System0 视角解析

从 System0 角度看,这篇论文的方法部分应当围绕 closed-loop-control、接触推理、接触丰富操作、external-force-perception、力控制、模仿学习、policy-learning、robot-generalization 来读。核心问题是:系统如何从快速变化的机器人状态中构造控制输入,如何处理接触、碰撞、约束、模型误差或执行延迟,以及如何把这些信息变成下一时刻的动作。

如果方法里有优化问题,重点看目标函数、约束和求解频率;如果方法里有学习策略,重点看观测空间、动作空间、训练扰动和部署时的闭环频率;如果方法里有硬件或传感器,重点看传感链路是否真的进入控制回路。这样读可以把论文和 System2 语言规划、System1 技能编排区分开来。

### 关键公式与机制

#### 公式 1: [[优化目标/约束]]

$$k_{\infty}(x_{i},x_{j})=\mathbb{E}_{\theta\sim\mathcal{N}(0,I)}\left[\langle\nabla_{\theta}f_{\theta}(x_{i}),\nabla_{\theta}f_{\theta}(x_{j})\rangle\right],$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 2: [[动力学或策略机制]]

$$\tau_{null}=\left(\mathbf{I}-\mathbf{J}^{\dagger}\mathbf{J}\right)\left(-\mathbf{K}_{n,p}\left(\mathbf{q}-\mathbf{q}_{rest}\right)-\mathbf{K}_{n,d}\mathbf{\dot{q}}\right)$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 3: [[动力学或策略机制]]

$$\mathbf{\tau}_{grav}=\mathbf{M}(\mathbf{q})\mathbf{\ddot{q}}+\mathbf{C}(\mathbf{q},\mathbf{\dot{q}})\mathbf{\dot{q}}+\mathbf{g}(\mathbf{q})=\text{RNEA}(\mathbf{q},\mathbf{\dot{q}},\mathbf{\ddot{q}})$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 4: [[优化目标/约束]]

$$=\mathbb{E}_{W\sim\mathcal{N}(0,I)}\left[\langle\nabla_{W}f(x),\nabla_{W}f(x^{\prime})\rangle\right]$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 5: [[动力学或策略机制]]

$$\tau_{ss}^{(i)}=\begin{cases}\mu_{s}^{(i)}\cos(\frac{\pi t}{f})&\text{if}\dot{q}^{(i)}<\dot{q}^{(i)}_{s},\\ 0&\text{otherwise}\end{cases}$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 6: [[动力学或策略机制]]

$$\mathbf{\tau}_{feedback}=\mu_{f}\mathbf{K}_{f,p}\mathbf{\tau}_{ext}-\mathbf{K}_{f,d}\mathbf{\dot{q}}$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 7: [[动力学或策略机制]]

$$\mathbf{\tau}=\mathbf{\tau}_{feedback}+\mathbf{\tau}_{null}+\mathbf{\tau}_{grav}+\mathbf{\tau}_{friction}+\mathbf{\tau}_{limit}$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 8: [[动力学或策略机制]]

$$\mathbf{X}_{t}\;=\;\bigl{[}\mathbf{z}_{t}^{V};\,\mathbf{z}_{t}^{F}\bigr{]}\;\in\;\mathbb{R}^{(M_{v}+1)\times d}.$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 9: [[动力学或策略机制]]

$$\mathbf{H}_{t}^{E}\;=\;\mathrm{Enc}\bigl(\mathbf{X}_{t}\bigr)\;\in\;\mathbb{R}^{(M_{v}+1)\times d}.$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 10: [[动力学或策略机制]]

$$\hat{q}_{t:t+k}=\mathrm{MLP}\bigl(\mathbf{H}_{t}^{D}\bigr)\;\in\;\mathbb{R}^{l\times d_{a}}.$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。
## 关键图表

### Figure 1: Pipeline / core system figure: Our low-cost bimanual teleoperation system with force-feedback. The system features tw

![Figure 1](https://arxiv.org/html/2502.17432v2/x1.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Pipeline / core system figure: Our low-cost bimanual teleoperation system with force-feedback. The system features tw”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 2: Tasks. We evaluate our leader-follower teleoperation system and autonomous policies tr

![Figure 2](https://arxiv.org/html/2502.17432v2/x4.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Tasks. We evaluate our leader-follower teleoperation system and autonomous policies tr”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 3: Policies trained with FACTR learns to identify mode switching. We visualize the averag

![Figure 3](https://arxiv.org/html/2502.17432v2/x7.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Policies trained with FACTR learns to identify mode switching. We visualize the averag”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### 论文表格

#### Table 1: Table 1

| Decay Type | Scheduler Equation |
| --- | --- |
| Constant |  n =  0  n  0 \sigma_{n}=\sigma_{0} = 0 |
| Linear | $\left(1-\frac{n}{N}\right) = 0 (1 - divide)$ |
| Cosine | $\frac{\sigma_{0}}{2}\Bigl(1+\cos\Bigl(\tfrac{n\pi}{N}\Bigr)\Bigr) = divide 0 2 (1 + (divide))$ |
| Exponential |  n =  0   n,  > 1 formulae-sequence  n   0  n  1 \sigma_{n}=\sigma_{0}\cdot\alpha^{n},\quad\alpha>1 = 0 , > 1 |
| Step | n - d steps ⌊ n N d steps ⌋ d steps formulae-sequence n d steps n N d steps d steps \sigma n \sigma $\Bigl(1-\frac{1}{d_{\text{steps}}}\lfloor\frac{n}{N/d_{\text{steps}}}\rfloor\Bigr),\quad d_{\text{steps}}>1 = 0 (1 - divide 1 steps ⌊ divide / steps ⌋), steps > 1$ |

**说明**: Table 1

#### Table 2: TABLE I: Evaluation of recovery behaviors for box lifting.

| | Train Objects | Test Objects |
| --- | --- | --- |
| ACT (Vision-Only) | 4/5 | 4/30 |
| ACT (Vision+Force) | 5/5 | 16/30 |
| FACTR | 3/3 | 27/30 |

**说明**: TABLE I: Evaluation of recovery behaviors for box lifting.

#### Table 3: TABLE II: Curriculum ablation.

| | Pixel Space | Latent Space | | |
| --- | --- | --- | --- | --- |
| | Blur | Downsample | Blur | Downsample |
| Constant | 16/25 | 15/25 | 17/25 | 16/25 |
| Linear | 19/25 | 18/25 | 19/25 | 18/25 |
| Cosine | 20/25 | 19/25 | 17/25 | 19/25 |
| Exp | 19/25 | 21/25 | 20/25 | 19/25 |
| Step | 19/25 | 18/25 | 20/25 | 19/25 |

**说明**: TABLE II: Curriculum ablation.

#### Table 4: TABLE III: We present the bill of materials of one leader arm teleoperation device with force feedback. The total cost i

| Object | Quantity | Total |
| --- | --- | --- |
| Dynamixel XM430-W210-T | 2 | $539.80 |
| Dynamixel XC330-T288-T | 6 | $539.40 |
| U2D2 Control PCB | 1 | $32.10 |
| 12V 20A Power Supply | 1 | $24.99 |
| FPX330-S102 Servo Bracket | 1 | $8.70 |
| Polymaker PLA PRO Filament | 1 | $24.99 |
| U2D2 Power Hub Board | 1 | $19.99 |
| 14AWG Cable | 1 | $23.99 |
| 3/4” Bearing | 1 | $6.99 |
| Screws | - | $15.99 |
| Total | | $1229.95 |

**说明**: TABLE III: We present the bill of materials of one leader arm teleoperation device with force feedback. The total cost is around $1229.95.

#### Table 5: TABLE IV: Policy Architecture and Training Hyperparameters

| Hyperparameter | Value |
| --- | --- |
| Behavior Policy Training | |
| Optimizer | AdamW |
| Base Learning Rate | 3e-4 |
| Weight Decay | 0.05 |
| Optimizer Momentum |  1,  2 = 0.9, 0.95 formulae-sequence  1  2 0.9 0.95 \beta_{1},\beta_{2}=0.9,0.95 1, 2 = 0.9, 0.95 |
| Batch Size | 128 |
| Learning Rate Schedule | Cosine Decay |
| Total Steps | 20000-50000 |
| Warmup Steps | 500 |
| Augmentation | RandomResizeCrop |
| GPU | RTX4090 (24 gb) |
| Wall-Clock Time | 2-6 hours |
| Visual Backbone ViT Architecture | |
| Patch Size | 16 |
| # Layers | 12 |
| # MHSA Heads | 12 |
| Hidden Dim | 768 |
| Class Token | Yes |
| Positional Encoding | sin cos |
| Action Chunking Transformer Architecture | |
| # Encoder Layers | 6 |
| # Decoder Layers | 6 |
| # MHSA Heads | 8 |
| Hidden Dim | 512 |
| Feed-Forward Dim | 2048 |
| Dropout | 0.1 |
| Positional Encoding | sin cos |
| Action Chunk | 100 |

**说明**: TABLE IV: Policy Architecture and Training Hyperparameters

#### Table 6: TABLE V: Pivot task training and testing performance.

| | FACTR | AdaNorm | NoiseAug |
| --- | --- | --- | --- |
| Train (%) | 90.0 | 25.0 | 85.0 |
| Test (%) | 77.7 | 6.1 | 65.0 |

**说明**: TABLE V: Pivot task training and testing performance.

#### Table 7: TABLE VI: Policy evaluation on unseen objects across 3 tasks.

| | Box Lift | Pivot | Rolling Dough |
| --- | --- | --- | --- |
| ACT (Vision-Only) | 35/120 | 30/130 | 0/60 |
| Bi-ACT | 68/120 | 76/130 | 41/60 |
| FACTR (Ours) | 105/120 | 101/130 | 46/60 |

**说明**: TABLE VI: Policy evaluation on unseen objects across 3 tasks.

#### Table 8: TABLE VII: Comparison of methods for Box Lift task.

| | Train | Test | Train Avg | Test Avg | | | | | |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| | Box1 | Box2 | Box3 | Box4 | Box5 | Box6 | Box7 | | |
| ACT (Vision-Only) | 10/10 | 7/10 | 1/10 | 1/10 | 6/10 | 3/10 | 1/10 | 100.0% | 31.7% |
| ACT (Vision+Force) | 10/10 | 2/10 | 4/10 | 4/10 | 10/10 | 10/10 | 5/10 | 100.0% | 58.3% |
| FACTR | 10/10 | 8/10 | 7/10 | 10/10 | 10/10 | 10/10 | 10/10 | 100.0% | 91.7% |

**说明**: TABLE VII: Comparison of methods for Box Lift task.

#### Table 9: TABLE VIII: Comparison of methods for Non-Prehensile Pivot task.

| | Train | Test | Train Avg | Test Avg | | | | | |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| | Box1 | Box2 | Box3 | Box4 | Box5 | Box6 | Box7 | | |
| ACT (Vision-Only) | 10/10 | 9/10 | 3/10 | 0/10 | 7/10 | 2/10 | 1/10 | 95.0% | 26.0% |
| ACT (Vision+Force) | 9/10 | 9/10 | 1/10 | 2/10 | 9/10 | 4/10 | 5/10 | 90.0% | 42.0% |
| FACTR | 9/10 | 9/10 | 6/10 | 5/10 | 10/10 | 7/10 | 10/10 | 90.0% | 76.0% |

**说明**: TABLE VIII: Comparison of methods for Non-Prehensile Pivot task.

#### Table 10: TABLE IX: Comparison of methods for Fruit Pick-Place task.

| | Train | Test | Train Avg | Test Avg | | |
| --- | --- | --- | --- | --- | --- | --- |
| | Obj1 | Obj2 | Obj3 | Obj4 | | |
| ACT (Vision-Only) | 5/5 | 0/5 | 4/5 | 0/5 | 100.0% | 26.7% |
| ACT (Vision+Force) | 5/5 | 3/5 | 4/5 | 4/5 | 100.0% | 73.3% |
| FACTR | 5/5 | 4/5 | 5/5 | 5/5 | 100.0% | 93.3% |

**说明**: TABLE IX: Comparison of methods for Fruit Pick-Place task.

#### Table 11: TABLE X: Comparison of methods for Rolling Dough task.

| | Train | Test | Train Avg | Test Avg | | |
| --- | --- | --- | --- | --- | --- | --- |
| | Obj1 | Obj2 | Obj3 | Obj4 | | |
| ACT (Vision-Only) | 0/5 | 0/5 | 0/5 | 0/5 | 0.0% | 0.0% |
| ACT (Vision+Force) | 4/5 | 4/5 | 3/5 | 4/5 | 80.0% | 70.0% |
| FACTR | 5/5 | 4/5 | 4/5 | 4/5 | 90.0% | 80.0% |

**说明**: TABLE X: Comparison of methods for Rolling Dough task.
## 实验解读

- 评价重点:围绕 closed-loop-control、接触推理、接触丰富操作、external-force-perception、力控制,优先看真实机器人结果、扰动恢复、接触稳定性、控制误差、成功率、实时性和 sim-to-real 差距。
- System0 视角:实验结论不只看最终成功率,还要看闭环过程中是否能处理状态估计误差、执行延迟、碰撞/接触突变和外部扰动。
- 与上层系统的关系:如果方法能在快闭环内稳定工作,它可以作为 System1 技能或 System2 规划的底层执行保障;如果只在离线或慢速设定下成立,则还不能直接作为身体级快闭环模块。
## 批判性思考

### 优点

- 论文把 closed-loop-control、接触推理、接触丰富操作、external-force-perception、力控制 放到机器人身体级闭环里讨论,和只做高层语义推理的工作相比,更接近真实机器人部署时会遇到的状态、动力学和安全约束。
- 若论文同时给出公式、系统结构图、硬件实验或消融结果,这些证据可以帮助判断方法是否真的能支撑毫秒到百毫秒级反应。

### 局限

- 需要重点核查控制频率、推理延迟、传感延迟、失败恢复、硬件磨损、未见场景泛化和安全边界;这些因素决定它能否成为可靠的 System0 模块。
- 对学习型方法,还要区分训练环境中的鲁棒性和真实部署中的鲁棒性,尤其是接触丰富任务、动态运动任务和人形机器人任务。
## 论文定位

- 研究对象:FACTR: Force-Attending Curriculum Training for Contact-Rich Policy Learning。
- 关键词:closed-loop-control、接触推理、接触丰富操作、external-force-perception、力控制、模仿学习、policy-learning、robot-generalization、遥操作。
- System0 关联:关注机器人身体级快反应链路中“状态/接触/局部感知 -> 约束/优化/策略 -> 动作”的短闭环,不把问题停留在语言规划或任务分解层。

## 速查卡片

> [!summary] FACTR
> - **论文**: https://www.roboticsproceedings.org/rss21/p079.pdf
> - **arXiv**: http://arxiv.org/abs/2502.17432v2
> - **arXiv HTML**: https://arxiv.org/html/2502.17432v2
