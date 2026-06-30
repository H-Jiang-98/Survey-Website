---
title: "Distributionally Robust Control via Stein Variational Inference for Contact-rich Manipulation"
method_name: "Distributionally Robust Control"
authors: ["Hrishikesh Sathyanarayan"]
year: 2026
venue: "RSS"
tags: ["contact-reasoning", "robot-manipulation", "robust-control", "adaptive-control", "contact-rich-manipulation", "robot-generalization"]
image_source: "online"
arxiv_html: "https://arxiv.org/html/2605.19029v1"
---
# Distributionally Robust Control
## 一句话总结

> Distributionally Robust Control via Stein Variational Inference for Contact-rich Manipulation 主要落在 [[adaptive-control]]、[[接触推理]]、[[接触丰富操作]]、[[机器人操作]] 这一类问题上;从 System0 视角看,它的价值在于把身体状态、接触变化、局部感知或动力学约束接到机器人低层闭环里,让动作生成更稳定、安全、实时。
## 核心贡献

1. 围绕 **Distributionally Robust Control via Stein Variational Inference for Contact-rich Manipulation** 建立了一个与 adaptive-control、接触推理、接触丰富操作、机器人操作、鲁棒控制、scalable-robot-learning 相关的机器人问题设定,重点不是高层语义规划,而是身体级闭环中的状态变化、接触变化或控制约束。
2. 方法层面的关键价值在于把感知、动力学、约束优化、策略学习或硬件反馈组织成可执行的低层动作生成机制,适合放进 System0 的快反应模块中理解。
3. 对系统集成而言,这类工作补上了 System1 技能选择与电机级控制之间的一层:它负责把上层目标变成动态可行、可恢复、可安全过滤的动作。
4. 若论文包含真实机器人、接触任务或仿真到真实迁移,核心应看它是否报告了闭环稳定性、扰动恢复、硬件延迟、失败模式和跨场景泛化。
## 问题背景

这篇论文可以放在 System0 的“具身快系统”背景下阅读:机器人需要在很短时间内根据本体状态、接触、力/触觉、局部视觉或动力学预测做出动作,而不是等待慢速规划器重新推理整个任务。adaptive-control、接触推理、接触丰富操作、机器人操作、鲁棒控制、scalable-robot-learning 相关问题的难点通常不在“知道要做什么”,而在动作执行过程中状态会快速偏离、接触会突然改变、约束会实时收紧,低层系统必须及时修正。

### System0 关联

精读时优先看三件事:第一,论文使用哪些快速反馈量,例如关节状态、接触状态、力觉、触觉、深度或局部视觉;第二,反馈如何进入 MPC、WBC、策略网络、安全过滤器或恢复控制器;第三,输出是否直接影响机器人下一步动作,而不是只服务离线分析或高层任务规划。
## 方法详解

### System0 视角解析

从 System0 角度看,这篇论文的方法部分应当围绕 adaptive-control、接触推理、接触丰富操作、机器人操作、鲁棒控制、scalable-robot-learning 来读。核心问题是:系统如何从快速变化的机器人状态中构造控制输入,如何处理接触、碰撞、约束、模型误差或执行延迟,以及如何把这些信息变成下一时刻的动作。

如果方法里有优化问题,重点看目标函数、约束和求解频率;如果方法里有学习策略,重点看观测空间、动作空间、训练扰动和部署时的闭环频率;如果方法里有硬件或传感器,重点看传感链路是否真的进入控制回路。这样读可以把论文和 System2 语言规划、System1 技能编排区分开来。

### 关键公式与机制

#### 公式 1: [[优化目标/约束]]

$$\begin{split}&\min_{\tau=\{(x_{k},u_{k},c_{k})\forall k\in[0,t_{h}]\}}\mathcal{J}_{\theta}(\tau)\\ \mathrm{s.t}&\begin{cases}x_{0}\in\mathcal{X}&\mathrm{(init. state)}\\ x_{k+1}=f_{\theta}(x_{k},u_{k})&\mathrm{(dynamical model)}\\ c_{k+1}\in\mathcal{C}(x_{k})&\mathrm{(contact model)}\end{cases}\end{split}$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 2: [[优化目标/约束]]

$$\begin{split}\mathbb{E}_{q}\big[\mathcal{L}(\tau,\theta)\big]&\approx\mathcal{L}(\tau,\theta)\Big| _{ $\theta=\bar{\theta}}\\ \mathbb{V}_{q}\big[\mathcal{L}(\tau,\theta)\big]&\approx\nabla_{\theta}\mathcal{L}(\tau,\theta)\Big$ | _{ $\theta=\bar{\theta}}^{\top}\Sigma_{\theta\theta}\nabla_{\theta}\mathcal{L}(\tau,\theta)\Big$ |_{\theta=\bar{\theta}}\end{split}$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 3: [[优化目标/约束]]

$$\begin{split}p^{*}(\theta)&=\arg\min_{p\in\mathcal{P}}\{\mathbb{D}_{KL}(p\lVert q)\\ &\equiv\mathbb{E}_{\theta\sim p(\theta)}\big[\log p(\theta)\big]-\mathbb{E}_{\theta\sim p(\theta)}\big[\log q(\theta)\big]+\log z\}\end{split}$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 4: [[优化目标/约束]]

$$\begin{split}\phi_{p,q}^{*}(\cdot)&=\arg\min_{\phi\in\mathcal{H}^{d}}\{-\nabla_{\theta}D_{KL}(p\lVert q)\ |\ \rVert\phi\lVert_{\mathcal{H}^{d}}\leq 1\}\\ &=\mathbb{E}_{\theta_{t}\sim q}\big[\mathcal{A}_{q}k(\theta_{t},\cdot)\big]\end{split}$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 5: [[动力学或策略机制]]

$$\begin{split}\phi_{p,q}^{*}(\cdot)=\mathbb{E}_{\theta_{t}\sim p}\big[k(\theta_{t},\cdot)\nabla_{\theta}\log q(\theta_{t})+\nabla_{\theta}k(\theta_{t},\cdot)\big]\end{split}$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 6: [[动力学或策略机制]]

$$\begin{split}\phi^{*}(\cdot)&=\mathbb{E}_{\theta\sim p(\theta)}\big[k(\theta^{i},\cdot)\nabla_{\theta}\log p^{\prime}(\theta^{i})+\nabla_{\theta}k(\theta^{i},\cdot)\big]\\ &\approx\frac{1}{N}\sum_{i=1}^{N}k(\theta_{t}^{i},\cdot)\nabla_{\theta}\log p^{\prime}(\theta_{t}^{i})+\nabla_{\theta}k(\theta_{t}^{i},\cdot)\end{split}$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 7: [[优化目标/约束]]

$$\begin{split}\min_{\tau=\{(x_{k},u_{k},c_{k})\forall k\in[t,t+t_{h}]\}}&\mathbb{E}_{q}\big[\mathcal{L}(\tau,\theta)\big]\approx\frac{1}{N}\sum_{i=1}^{N}\mathcal{L}(\tau,\theta^{i})\\ &=\mathcal{L}(\tau,\theta)\big|_{\theta=\bar{\theta}}+\gamma\ \frac{1}{N}\sum_{i=1}^{N}\delta\mathcal{L}(\tau,\theta^{i}_{t})\end{split}$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 8: [[优化目标/约束]]

$$\tau^{*}\leftarrow\arg\min_{\tau}\mathcal{L}(\tau,\theta)\big|_{\theta=\bar{\theta}}+\gamma\frac{1}{N}\sum_{i=1}^{N}\delta\mathcal{L}(\tau,\theta_{t}^{i})$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 9: [[优化目标/约束]]

$$\tau^{*}=\arg\min_{\tau\in\mathcal{T}}\tilde{\mathcal{L}}(\tau,\theta)$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 10: [[优化目标/约束]]

$$\begin{split}&\min_{\tau=\{(x_{k},u_{k},c_{k})\forall k\in[t,t+t_{h}]\}}m(x_{t_{h}},u_{t_{h}},c_{t_{h}})+\sum_{k=t}^{t+t_{h}}\ell(x_{k},u_{k},c_{k})\\ \mathrm{s.t.}&\begin{cases}x_{0}\in\mathcal{X}&\mathrm{(init. state)}\\ x_{k+1}=f_{\theta}(x_{k},u_{k},c_{k})&\mathrm{(dynamical model)}\end{cases}\end{split}$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。
## 关键图表

### Figure 1: Pipeline / core system figure: Within-hand dynamic positioning of a cup with unknown mass distribution and friction c

![Figure 1](https://arxiv.org/html/2605.19029v1/x1.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Pipeline / core system figure: Within-hand dynamic positioning of a cup with unknown mass distribution and friction c”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 2: Hardware demonstration of within-hand dynamic serving with a top-heavy object. The rob

![Figure 2](https://arxiv.org/html/2605.19029v1/x4.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Hardware demonstration of within-hand dynamic serving with a top-heavy object. The rob”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 3: Controller convergence based on prior distribution for Within-hand Positioning. The nu

![Figure 3](https://arxiv.org/html/2605.19029v1/x5.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Controller convergence based on prior distribution for Within-hand Positioning. The nu”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### 论文表格

#### Table 1: TABLE I: Simulated performance comparison 32 trials of the bimanual Push-T and within-hand dynamic positioning expe

| | Success % / Time to Completion (s) | | | |
| --- | --- | --- | --- | --- |
| | Bimanual Push-T | Within-Hand Dynamic Positioning | | |
| Method | ≤ 10 \leq 10 cm | ≤ 1 \leq 1 cm | ≤ 10 \leq 10 cm | ≤ 1 \leq 1 cm |
| SV-DRO (Ours) | 93.25% / 16.45 ± \pm 2.45 | 84.38% / 14.21 ± \pm 2.22 | 100% / 12.00 ± \pm 4.78 | 100% / 12.00 ± \pm 4.78 |
| DuST-MPC [3 ] | 71.88% / 14.41 ± \pm 3.58 | 59.38% / 13.32 ± \pm 2.67 | 84.38% / 20.21 ± \pm 5.62 | 46.88% / 20.18 ± \pm 1.24 |
| EMPPI [1 ] | 43.75% / 18.88 ± \pm 6.29 | 31.25% / 14.89 ± \pm 3.12 | 59.38% / 22.44 ± \pm 3.22 | 40.63% / 21.78 ± \pm 3.43 |
| MPC | 28.13% / 19.56 ± \pm 6.21 | 18.75% / 16.56 ± \pm 2.21 | 31.25% / 18.40 ± \pm 5.60 | 25% / 16.80 ± \pm 5.14 |
| DRO [18 ] | 3.13% / 19.12 ± \pm 3.13 | 0% / ⋆ \star | 34.38% / 17.79 ± \pm 5.00 | 28.13% / 18.55 ± \pm 5.15 |

**说明**: TABLE I: Simulated performance comparison 32 trials of the bimanual Push-T and within-hand dynamic positioning experiments unknown bounded uniform parameter distribution prior initialization. Results report success rate and completion time as mean ± $\pm$ standard deviation. ⋆ $\star$ denotes insufficient completed trials to obtain completion-time statistics.

#### Table 2: TABLE II: Effect of kernel choice on SV-DRO performance 32 trials for within-hand object positioning problem. We co

| Kernel | Success Rate % \% (< 1 <1 cm) | Completion Time (s) |
| --- | --- | --- |
| RBF | 100.0 % 100.0\% | 12.93 ± 4.78 12.93\pm 4.78 |
| k = 1 k=1 | 100.0 % 100.0\% | 16.54 ± 2.12 16.54\pm 2.12 |
| IMQ | 100.0 % 100.0\% | 8.78 ± 1.34 8.78\pm 1.34 |

**说明**: TABLE II: Effect of kernel choice on SV-DRO performance 32 trials for within-hand object positioning problem. We compare the standard RBF kernel, a constant kernel, and the inverse multiquadric (IMQ) kernel. The IMQ kernel achieves the fastest completion time, suggesting that its heavier-tailed distribution better preserves particle diversity task-sensitive regions of the posterior and avoids premature mode collapse of Stein particles.

#### Table 3: TABLE III: Experiment Parameters. Here we outline the experiment details for both the Bimanual push-T and the Within-Han

| Term | Bimanual Push-T | Within-Hand Dynamic Positioning |
| --- | --- | --- |
| robot pose | x ee y ee ⏟ end effector x ee y ee ⏟ end effector x block y block block ⏟ T block \ x $\mathrm{ee1}},y_{\mathrm{ee1}})}_{\mathrm{end effector 1}},\ {(x_{\mathrm{ee2}},y_{\mathrm{ee2}})}_{\mathrm{end effector 2}},\ {(x_{\mathrm{block}},y_{\mathrm{block}},\theta_{\mathrm{block}})}_{\mathrm{T block}}]^{\top}$ | $\mathrm{tray}},y_{\mathrm{tray}},\theta_{\mathrm{tray}})}_{\mathrm{tray}},\ {(x_{\mathrm{object}},y_{\mathrm{object}},\theta_{\mathrm{object}})}_{\mathrm{object}}]^{\top}$ |
| control (torque) | $\mathrm{ee1}},u_{y,\mathrm{ee1}}),(u_{x,\mathrm{ee2}},u_{y,\mathrm{ee2}})]^{\top}$ | $\mathrm{tray}},u_{y,\mathrm{tray}},u_{\theta,\mathrm{tray}}]^{\top}$ |
| proprioceptive sensor noise | 0.001 m | 0.01 m |
| uniform sampling bounds of params. | mass: (- 1  e-  4, 10.0)  kg (-1 $\mathrm{e-}4,\,10.0)\,\mathrm{kg}$ | mass: (5  e-  2, 1.0)  kg (5 $\mathrm{e-}2,\,1.0)\,\mathrm{kg}$ |
| inertia: (- 1  e-  4, 10.0)  kg m 2 (-1 $\mathrm{e-}4,\,10.0)\,\mathrm{kg m}^{2}$ | inertia: (- 1  e-  4, 1.0)  kg m 2 (-1 $\mathrm{e-}4,\,1.0)\,\mathrm{kg m}^{2}$ | |
| — | x CoM x_{ $\mathrm{CoM}}: (- 1.2 e- 2, 1.2 e- 2) m (⋆) (-1.2\mathrm{e-}2,\,1.2\mathrm{e-}2)\,\mathrm{m}(\star)$ | |
| — | y CoM y_{ $\mathrm{CoM}}: (- 2.5 e- 2, 2.5 e- 2) m (⋆) (-2.5\mathrm{e-}2,\,2.5\mathrm{e-}2)\,\mathrm{m}(\star)$ | |
| planning horizon | 0.375 seconds | 0.375 seconds |
| total time | 10 seconds | 12.5 seconds |
| spatial bounds | -0.75 m  \times 0.75 m | -0.125 m  \times 0.125 m |
| control (torque) bounds [u min, u max  $\mathrm{u}_{\mathrm{min}},\mathrm{u}_{\mathrm{max}} ]$ | [(-1.0, -1.0), (1.0, 1.0)] N | [(-5.0, -0.2, -0.5), (5.0, 0.2, 0.5)] N |
| object dimensions | (T height, T width, T thickness) = (0.5, 0.6, 0.2) (T_{ $\mathrm{height}},T_{\mathrm{width}},T_{\mathrm{thickness}})=(0.5,0.6,0.2) m$ | (radius obj., height obj.) = (0.05, 0.25) ( $\mathrm{radius}_{\mathrm{obj.}},\mathrm{height}_{\mathrm{obj.}})=(0.05,0.25) m$ |
| MPC tuning weights | diag(Q  $\mathbf{Q}) = [5.0, 5.0, 2.0, 0.0, 0.0, 0.0]$ | diag(Q  $\mathbf{Q}) = [15.0, 0.0, 5.0, 0.01, 0.01, 0.1, 0.0, 0.0, 5.0, 0.01, 0.01, 0.1]$ |
| diag(R  $\mathbf{R}) = [0.0001, 0.0001]$ | diag(R  $\mathbf{R}) = [15.0, 20.0, 20.0]$ | |
| diag(N  $\mathbf{N}) = [0.1, 0.1]$ | diag(N  $\mathbf{N}) = [0.001, 0.001]$ | |
| diag(Q f  $\mathbf{Q_{f}}) = [5.0, 5.0, 2.0, 0.5, 0.5, 0.5]$ | diag(Q f  $\mathbf{Q_{f}}) = [15.0, 0.0, 5.0, 0.01, 0.01, 0.1, 0.0, 0.0, 5.0, 0.01, 0.01, 0.1]$ | |
| $\theta Particle Count$ | 5 | 5 |
| SVGD Step Size | 0.0001 | 0.0001 |
| RBF Kernel bandwidth h h | 0.75 | 0.75 |
| num. control steps per planning cycle H H | 10 | 10 |

**说明**: TABLE III: Experiment Parameters. Here we outline the experiment details for both the Bimanual push-T and the Within-Hand Dynamic Positioning experiments. We outline the state-control space of each experiment as well as uncertain model parameters. Each parameter of interest is sampled from a uniform distribution with indicated bounds. ⋆ $\star$ indicates that the boundaries for the sample space is with respect to the object body coordinates, not world coordinates.

#### Table 4: TABLE IV: Compute parity across baselines for a single MPC loop. All methods are evaluated the same compute settin

| Method | Wall-Clock / Step (s) |
| --- | --- |
| SV-DRO (Ours) | 0.130 ± 0.008 0.130\pm 0.008 |
| DuST-MPC | 0.151 ± 0.009 0.151\pm 0.009 |
| EMPPI | 0.087 ± 0.007 0.087\pm 0.007 |
| MPC | 0.068 ± 0.011 0.068\pm 0.011 |
| DRO | 0.123 ± 0.008 0.123\pm 0.008 |

**说明**: TABLE IV: Compute parity across baselines for a single MPC loop. All methods are evaluated the same compute setting with an MPC planning horizon of 0.375 0.375 s on an NVIDIA GeForce RTX 3080 GPU, and particle-based methods use N = 5 N=5 parameter particles. SV-DRO has a wall-clock time of 0.130 ± 0.008 0.130$\pm$ 0.008 s per step, which is comparable to DRO and remains below the MPC planning horizon despite the additional SVGD posterior update.
## 实验解读

- 评价重点:围绕 adaptive-control、接触推理、接触丰富操作、机器人操作、鲁棒控制,优先看真实机器人结果、扰动恢复、接触稳定性、控制误差、成功率、实时性和 sim-to-real 差距。
- System0 视角:实验结论不只看最终成功率,还要看闭环过程中是否能处理状态估计误差、执行延迟、碰撞/接触突变和外部扰动。
- 与上层系统的关系:如果方法能在快闭环内稳定工作,它可以作为 System1 技能或 System2 规划的底层执行保障;如果只在离线或慢速设定下成立,则还不能直接作为身体级快闭环模块。
## 批判性思考

### 优点

- 论文把 adaptive-control、接触推理、接触丰富操作、机器人操作、鲁棒控制 放到机器人身体级闭环里讨论,和只做高层语义推理的工作相比,更接近真实机器人部署时会遇到的状态、动力学和安全约束。
- 若论文同时给出公式、系统结构图、硬件实验或消融结果,这些证据可以帮助判断方法是否真的能支撑毫秒到百毫秒级反应。

### 局限

- 需要重点核查控制频率、推理延迟、传感延迟、失败恢复、硬件磨损、未见场景泛化和安全边界;这些因素决定它能否成为可靠的 System0 模块。
- 对学习型方法,还要区分训练环境中的鲁棒性和真实部署中的鲁棒性,尤其是接触丰富任务、动态运动任务和人形机器人任务。
## 论文定位

- 研究对象:Distributionally Robust Control via Stein Variational Inference for Contact-rich Manipulation。
- 关键词:adaptive-control、接触推理、接触丰富操作、机器人操作、鲁棒控制、scalable-robot-learning。
- System0 关联:关注机器人身体级快反应链路中“状态/接触/局部感知 -> 约束/优化/策略 -> 动作”的短闭环,不把问题停留在语言规划或任务分解层。

## 速查卡片

> [!summary] Distributionally Robust Control
> - **论文**: https://www.roboticsproceedings.org/rss22/p061.pdf
> - **arXiv**: http://arxiv.org/abs/2605.19029v1
> - **arXiv HTML**: https://arxiv.org/html/2605.19029v1
