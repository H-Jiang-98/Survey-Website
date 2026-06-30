---
title: "Generalizing Safety Beyond Collision-Avoidance via Latent-Space Reachability Analysis"
method_name: "Generalizing Safety Beyond"
authors: ["Kensuke Nakamura"]
year: 2025
venue: "RSS"
tags: ["contact-reasoning", "real-time-control", "safe-control", "imitation-learning", "robot-generalization", "collision-avoidance", "recovery"]
image_source: "online"
arxiv_html: "https://arxiv.org/html/2502.00935v3"
---
# Generalizing Safety Beyond
## 一句话总结

> Generalizing Safety Beyond Collision-Avoidance via Latent-Space Reachability Analysis 主要落在 [[碰撞避免]]、[[接触推理]]、[[模仿学习]]、[[reachability]] 这一类问题上;从 System0 视角看,它的价值在于把身体状态、接触变化、局部感知或动力学约束接到机器人低层闭环里,让动作生成更稳定、安全、实时。
## 核心贡献

1. 围绕 **Generalizing Safety Beyond Collision-Avoidance via Latent-Space Reachability Analysis** 建立了一个与 碰撞避免、接触推理、模仿学习、reachability、实时控制、recovery 相关的机器人问题设定,重点不是高层语义规划,而是身体级闭环中的状态变化、接触变化或控制约束。
2. 方法层面的关键价值在于把感知、动力学、约束优化、策略学习或硬件反馈组织成可执行的低层动作生成机制,适合放进 System0 的快反应模块中理解。
3. 对系统集成而言,这类工作补上了 System1 技能选择与电机级控制之间的一层:它负责把上层目标变成动态可行、可恢复、可安全过滤的动作。
4. 若论文包含真实机器人、接触任务或仿真到真实迁移,核心应看它是否报告了闭环稳定性、扰动恢复、硬件延迟、失败模式和跨场景泛化。
## 问题背景

这篇论文可以放在 System0 的“具身快系统”背景下阅读:机器人需要在很短时间内根据本体状态、接触、力/触觉、局部视觉或动力学预测做出动作,而不是等待慢速规划器重新推理整个任务。碰撞避免、接触推理、模仿学习、reachability、实时控制、recovery、robot-generalization 相关问题的难点通常不在“知道要做什么”,而在动作执行过程中状态会快速偏离、接触会突然改变、约束会实时收紧,低层系统必须及时修正。

### System0 关联

精读时优先看三件事:第一,论文使用哪些快速反馈量,例如关节状态、接触状态、力觉、触觉、深度或局部视觉;第二,反馈如何进入 MPC、WBC、策略网络、安全过滤器或恢复控制器;第三,输出是否直接影响机器人下一步动作,而不是只服务离线分析或高层任务规划。
## 方法详解

### System0 视角解析

从 System0 角度看,这篇论文的方法部分应当围绕 碰撞避免、接触推理、模仿学习、reachability、实时控制、recovery、robot-generalization、safe-control 来读。核心问题是:系统如何从快速变化的机器人状态中构造控制输入,如何处理接触、碰撞、约束、模型误差或执行延迟,以及如何把这些信息变成下一时刻的动作。

如果方法里有优化问题,重点看目标函数、约束和求解频率;如果方法里有学习策略,重点看观测空间、动作空间、训练扰动和部署时的闭环频率;如果方法里有硬件或传感器,重点看传感链路是否真的进入控制回路。这样读可以把论文和 System2 语言规划、System1 技能编排区分开来。

### 关键公式与机制

#### 公式 1: [[优化目标/约束]]

$$V(s)=\min\Big{\{}\ell(s),~{}\max_{a\in\mathcal{A}}V(f(s,a))\Big{\}},$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 2: [[优化目标/约束]]

$$\pi^{\text{shield*}}(s):=\arg\max_{a\in\mathcal{A}}V(f(s,a)).$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 3: [[优化目标/约束]]

$$V_{\mathrm{latent}}(z)=(1-\gamma)\ell_{\mu}(z)\\ +\gamma\min\Big{\{}\ell_{\mu}(z),\max_{a\in\mathcal{A}}\mathbb{E}_{\hat{z}^{\prime}\sim p_{\phi}(\cdot\;|\;z,a)}[V_{\mathrm{latent}}(\hat{z}^{\prime})]\Big{\}}$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 4: [[优化目标/约束]]

$$V_{\mathrm{latent}}(z)=\min\Big{\{}\ell_{\mu}(z),~{}\max_{a\in\mathcal{A}}\mathbb{E}_{\hat{z}^{\prime}\sim p_{\phi}(\cdot\;|\;z,a)}\Big{[}V_{\mathrm{latent}}(\hat{z}^{\prime})\Big{]}\Big{\}}.$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 5: [[优化目标/约束]]

$$\pi^{\text{shield*}}_{\mathrm{latent}}(z)=\arg\max_{a\in\mathcal{A}}Q^{\text{shield*}}_{\mathrm{latent}}(z,a)$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 6: [[优化目标/约束]]

$$\pi^{\text{shield*}}_{\mathrm{priv}}(s)=\arg\max_{a\in\mathcal{A}}Q^{\text{shield*}}_{\mathrm{priv}}(s,a)$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 7: [[优化目标/约束]]

$$\pi^{\mathrm{risk}}_{\mathrm{latent}}(z):=\arg\min_{a\in\mathcal{A}}Q^{\mathrm{risk}}(z,a)$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 8: [[动力学或策略机制]]

$$a^{\text{exec}}_{t}=\begin{cases}\pi^{\text{task}}(z_{t}),&\text{if}V^{\mathrm{risk}}(\hat{z}_{t+1})<\epsilon_{\mathrm{risk}},\\ \pi^{\mathrm{risk}}_{\mathrm{latent}}(z_{t}),&\text{otherwise.}\\ \end{cases}$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 9: [[动力学或策略机制]]

$$a^{\text{exec}}_{t}=\begin{cases}\pi^{\text{task}}(z_{t}),&\text{if}V(\hat{z}_{t+1})>\epsilon,\\ \pi^{\text{shield*}}_{\mathrm{latent}}(z_{t}),&\text{otherwise.}\\ \end{cases}$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 10: [[动力学或策略机制]]

$$a^{\text{exec}}_{t}=\begin{cases}\pi^{\text{task}}(z_{t}),&\text{if}V(\hat{z}_{t+1})>\epsilon\\ \pi^{\text{shield*}}_{\mathrm{latent}}(z_{t}),&\mathrm{otherwise}\\ \end{cases}$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。
## 关键图表

### Figure 1: Pipeline / core system figure: Latent Safety vs. Privileged Safety. Dubins’ car collision-avoidance qualitative resul

![Figure 1](https://arxiv.org/html/2502.00935v3/x1.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Pipeline / core system figure: Latent Safety vs. Privileged Safety. Dubins’ car collision-avoidance qualitative resul”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 2: Ablation: Latent Safety with Incomplete WM. Unsafe set approximated by LatentSafe usin

![Figure 2](https://arxiv.org/html/2502.00935v3/x2.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Ablation: Latent Safety with Incomplete WM. Unsafe set approximated by LatentSafe usin”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 3: Far Left: Without a safety filter, a teleoperator lifts the closed-end of the bag too

![Figure 3](https://arxiv.org/html/2502.00935v3/x4.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Far Left: Without a safety filter, a teleoperator lifts the closed-end of the bag too”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### 论文表格

#### Table 1: TABLE I: Quality of the Runtime Monitor. Performance of latent (LatentSafe) and privileged (PrivilegedSafe) safety v

| Method | True Safe | False Safe | False Unsafe | True Unsafe | F 1 F 1 F_{1} 1 -score |
| --- | --- | --- | --- | --- | --- |
| PrivilegedSafe | 0.771 | 0.011 | 0.008 | 0.209 | 0.987 |
| LatentSafe | 0.759 | 0.003 | 0.021 | 0.217 | 0.984 |

**说明**: TABLE I: Quality of the Runtime Monitor. Performance of latent (LatentSafe) and privileged (PrivilegedSafe) safety value functions. Note that this is computed all three dimensions of the Dubins’ car state.

#### Table 2: TABLE II: Visual Manipulation: Simulation. Success at the task without any safety violations, constraint violations, and

| Method | Safe Success % percent \% % (↑ ↑ \uparrow ↑) | Constraint Violation % percent \% % (↓ ↓ \downarrow ↓) | Incompletion % percent \% % (↓ ↓ \downarrow ↓) |
| --- | --- | --- | --- |
| Dreamer | 64 | 36 | 0 |
| SQRL risk . risk . \epsilon $\mathrm{risk}}=0.1 risk = 0.1)$ | 68 | 28 | 4 |
| SQRL risk . risk . \epsilon $\mathrm{risk}}=0.05 risk = 0.05)$ | 8 | 22 | 70 |
| LatentSafe | 80 | 20 | 0 |

**说明**: TABLE II: Visual Manipulation: Simulation. Success at the task without any safety violations, constraint violations, and incompletion rates across 50 rollouts corresponding to 50 random initial conditions of the blocks. Task success is picking up the green block; constraint violation is where either of the red blocks fall down on the table.

#### Table 3: TABLE III: Hyperparameters for Dreamer

| Hyperparameter | Values |
| --- | --- |
| Image size | 128 |
| Optimizer | Adam |
| Learning rate (lr) | 1 e - 4 1 e 4 1e-4 1 - 4 |
| Hidden dim | 512 |
| Dyn deterministic | 512 |
| Activation fn | SiLU |
| CNN depth | 32 |
| Batch size | 16 |
| Batch Length | 64 |
| Recon loss scale | 1 |
| Dyn loss scale | 0.5 |
| Representation loss scale | 0.1 |

**说明**: TABLE III: Hyperparameters for Dreamer

#### Table 4: TABLE IV: Hyperparameters for DINO-WM

| Hyperparameter | Values |
| --- | --- |
| Image size | 224 |
| DINOv2 patch size | (14  14, 384) 14 14 384 (14\times 14,384) (14  14, 384) |
| Optimizer | AdamW |
| Predictor lr | 5e-5 |
| Decoder lr | 3e-4 |
| Action Encoder lr | 5e-4 |
| Action emb dim | 10 |
| Proprioception emb dim | 10 |
| Batch size | 16 |
| Training iterations | 100000 |
| ViT depth | 6 |
| ViT attention heads | 16 |
| ViT MLP dim | 2048 |

**说明**: TABLE IV: Hyperparameters for DINO-WM

#### Table 5: TABLE V: Hyperparameters for DDQN HJ Reachability

| Hyperparameter | Values |
| --- | --- |
| Optimizer | AdamW |
| Learning rate | 1e-3 |
| Learning rate decay | 0.8 |
| Hidden dims | [100, 100 ] 100 100 [100,100] [100, 100 ] |
| Time discount $\gamma$ | 0.9999 |
| Activations | Tanh |
| Batch size | 64 |
| Training iterations | 400000 |

**说明**: TABLE V: Hyperparameters for DDQN HJ Reachability

#### Table 6: TABLE VI: Hyperparameters for DDPG HJ Reachability

| Hyperparameter | Values |
| --- | --- |
| Optimizer | AdamW |
| Actor lr | 1e-4 |
| Critic lr | 1e-3 |
| Actor + Critic hidden dims | [512, 512, 512, 512 ] 512 512 512 512 [512,512,512,512] [512, 512, 512, 512 ] |
| Time discount $\gamma$ | 0.9999 |
| Activations | ReLU |
| Batch size | 512 |
| Epochs | 50 |

**说明**: TABLE VI: Hyperparameters for DDPG HJ Reachability

#### Table 7: TABLE VII: Hyperparameters for Diffusion Policy

| Hyperparameter | Values |
| --- | --- |
| State Normalization | Yes |
| Action Normalization | Yes |
| Action Space | End Effector Delta Position |
| Rotation Representation | Axis Angle |
| Action Chunk | 16 |
| Image Chunk | 2 |
| Image Size | 256 |
| Batch size | 100 |
| Training Iterations | 500000 |
| Learning Rate | 1e-4 |
| Learning Rate Schedule | Cosine |
| Optimizer | AdamW |

**说明**: TABLE VII: Hyperparameters for Diffusion Policy
## 实验解读

- 评价重点:围绕 碰撞避免、接触推理、模仿学习、reachability、实时控制,优先看真实机器人结果、扰动恢复、接触稳定性、控制误差、成功率、实时性和 sim-to-real 差距。
- System0 视角:实验结论不只看最终成功率,还要看闭环过程中是否能处理状态估计误差、执行延迟、碰撞/接触突变和外部扰动。
- 与上层系统的关系:如果方法能在快闭环内稳定工作,它可以作为 System1 技能或 System2 规划的底层执行保障;如果只在离线或慢速设定下成立,则还不能直接作为身体级快闭环模块。
## 批判性思考

### 优点

- 论文把 碰撞避免、接触推理、模仿学习、reachability、实时控制 放到机器人身体级闭环里讨论,和只做高层语义推理的工作相比,更接近真实机器人部署时会遇到的状态、动力学和安全约束。
- 若论文同时给出公式、系统结构图、硬件实验或消融结果,这些证据可以帮助判断方法是否真的能支撑毫秒到百毫秒级反应。

### 局限

- 需要重点核查控制频率、推理延迟、传感延迟、失败恢复、硬件磨损、未见场景泛化和安全边界;这些因素决定它能否成为可靠的 System0 模块。
- 对学习型方法,还要区分训练环境中的鲁棒性和真实部署中的鲁棒性,尤其是接触丰富任务、动态运动任务和人形机器人任务。
## 论文定位

- 研究对象:Generalizing Safety Beyond Collision-Avoidance via Latent-Space Reachability Analysis。
- 关键词:碰撞避免、接触推理、模仿学习、reachability、实时控制、recovery、robot-generalization、safe-control、安全过滤、遥操作。
- System0 关联:关注机器人身体级快反应链路中“状态/接触/局部感知 -> 约束/优化/策略 -> 动作”的短闭环,不把问题停留在语言规划或任务分解层。

## 速查卡片

> [!summary] Generalizing Safety Beyond
> - **论文**: https://www.roboticsproceedings.org/rss21/p113.pdf
> - **arXiv**: http://arxiv.org/abs/2502.00935v3
> - **arXiv HTML**: https://arxiv.org/html/2502.00935v3
