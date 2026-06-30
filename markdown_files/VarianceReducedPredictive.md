---
title: "Variance-Reduced Model Predictive Path Integral via Quadratic Model Approximation"
method_name: "Variance Reduced Predictive"
authors: ["Fabian Schramm"]
year: 2026
venue: "RSS"
tags: ["contact-reasoning", "robot-manipulation", "real-time-control", "contact-rich-manipulation"]
image_source: "online"
arxiv_html: "https://arxiv.org/html/2602.03639v2"
---
# Variance Reduced Predictive
## 一句话总结

> Variance-Reduced Model Predictive Path Integral via Quadratic Model Approximation 主要落在 [[接触推理]]、[[接触丰富操作]]、[[实时控制]]、[[机器人操作]] 这一类问题上;从 System0 视角看,它的价值在于把身体状态、接触变化、局部感知或动力学约束接到机器人低层闭环里,让动作生成更稳定、安全、实时。
## 核心贡献

1. 围绕 **Variance-Reduced Model Predictive Path Integral via Quadratic Model Approximation** 建立了一个与 接触推理、接触丰富操作、实时控制、机器人操作 相关的机器人问题设定,重点不是高层语义规划,而是身体级闭环中的状态变化、接触变化或控制约束。
2. 方法层面的关键价值在于把感知、动力学、约束优化、策略学习或硬件反馈组织成可执行的低层动作生成机制,适合放进 System0 的快反应模块中理解。
3. 对系统集成而言,这类工作补上了 System1 技能选择与电机级控制之间的一层:它负责把上层目标变成动态可行、可恢复、可安全过滤的动作。
4. 若论文包含真实机器人、接触任务或仿真到真实迁移,核心应看它是否报告了闭环稳定性、扰动恢复、硬件延迟、失败模式和跨场景泛化。
## 问题背景

这篇论文可以放在 System0 的“具身快系统”背景下阅读:机器人需要在很短时间内根据本体状态、接触、力/触觉、局部视觉或动力学预测做出动作,而不是等待慢速规划器重新推理整个任务。接触推理、接触丰富操作、实时控制、机器人操作 相关问题的难点通常不在“知道要做什么”,而在动作执行过程中状态会快速偏离、接触会突然改变、约束会实时收紧,低层系统必须及时修正。

### System0 关联

精读时优先看三件事:第一,论文使用哪些快速反馈量,例如关节状态、接触状态、力觉、触觉、深度或局部视觉;第二,反馈如何进入 MPC、WBC、策略网络、安全过滤器或恢复控制器;第三,输出是否直接影响机器人下一步动作,而不是只服务离线分析或高层任务规划。
## 方法详解

### System0 视角解析

从 System0 角度看,这篇论文的方法部分应当围绕 接触推理、接触丰富操作、实时控制、机器人操作 来读。核心问题是:系统如何从快速变化的机器人状态中构造控制输入,如何处理接触、碰撞、约束、模型误差或执行延迟,以及如何把这些信息变成下一时刻的动作。

如果方法里有优化问题,重点看目标函数、约束和求解频率;如果方法里有学习策略,重点看观测空间、动作空间、训练扰动和部署时的闭环频率;如果方法里有硬件或传感器,重点看传感链路是否真的进入控制回路。这样读可以把论文和 System2 语言规划、System1 技能编排区分开来。

### 关键公式与机制

#### 公式 1: [[优化目标/约束]]

$$\mu^{*}\in\ {\begin{subarray}{c}\mu\\ \text{supp}(\mu)\subseteq\Omega\end{subarray}}{\arg\min}\,\mathcal{J}(\mu)\quad\text{with}\quad\mathcal{J}(\mu)=\int_{\Omega}f(x)\mu(x)\,dx.$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 2: [[优化目标/约束]]

$$\mu^{k+1}=\ {\mu}{\arg\min}\Big(\mathcal{J}(\mu)+\lambda\cdot\text{KL}(\mu\,||\,\mu^{k})\Big).$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 3: [[动力学或策略机制]]

$$\tilde{x}^{k}=\tilde{\Sigma}^{k}\left((\Sigma^{k})^{-1}\bar{x}^{k}+\tfrac{1}{\lambda}\left(H^{k}\bar{x}^{k}-g^{k}\right)\right).$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 4: [[优化目标/约束]]

$$p^{*}(x)\;\propto\;\ {\mathcal{N}(x\,|\,\tilde{x}^{k},\tilde{\Sigma}^{k})}_{\tilde{p}_{\theta^{k}}(x)}\cdot\exp\!\left(-r^{k}(x)/\lambda\right).$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 5: [[动力学或策略机制]]

$$\Delta\bar{x}^{k}=\frac{\mathbb{E}_{\tilde{p}_{\theta^{k}}}\!\left[\epsilon\,\exp\!\left(-r^{k}(\bar{x}^{k}+\epsilon)/\lambda\right)\right]}{\mathbb{E}_{\tilde{p}_{\theta^{k}}}\!\left[\exp\!\left(-r^{k}(\bar{x}^{k}+\epsilon)/\lambda\right)\right]}.$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 6: [[动力学或策略机制]]

$$\Delta\bar{x}^{k}\approx\sum_{i=1}^{N}w_{i}\,\epsilon_{i},\;\text{with}\;w_{i}=\frac{\exp\!\left(-r^{k}(\bar{x}^{k}+\epsilon_{i})/\lambda\right)}{\sum_{j=1}^{N}\exp\!\left(-r^{k}(\bar{x}^{k}+\epsilon_{j})/\lambda\right)}.$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 7: [[动力学或策略机制]]

$$\frac{1}{\tilde{\sigma}_{i}^{2}}=\frac{1}{\sigma_{0k}^{2}}+\frac{\kappa_{i}}{\lambda}\quad\Rightarrow\quad\tilde{\sigma}_{i}^{2}=\frac{\lambda\,\sigma_{0k}^{2}}{\lambda+\sigma_{0k}^{2}\,\kappa_{i}}.$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 8: [[动力学或策略机制]]

$$\bar{x}^{k+1}\approx\sum_{i=1}^{N}\tilde{w}^{(i)}\,x^{(i)},\;\text{with}\tilde{w}^{(i)}=\frac{\exp\left(-r^{k}(x^{(i)})/\lambda\right)}{\sum_{j=1}^{N}\exp\left(-r^{k}(x^{(j)})/\lambda\right)}.$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 9: [[动力学或策略机制]]

$$p^{*}(x)\propto\ {\left[p_{\theta^{k}}(x)\exp\left(-m^{k}(x)/\lambda\right)\right]}_{\text{model-guided prior}\tilde{p}_{\theta^{k}}(x)}\cdot\exp\left(-r^{k}(x)/\lambda\right).$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 10: [[动力学或策略机制]]

$$\approx\frac{1}{M\sigma^{4}}\sum_{j=1}^{M}\left(f(\bar{x}^{k}+z_{j})-f(\bar{x}^{k})\right)(z_{j}z_{j}^{\top}-\sigma^{2}I),$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。
## 关键图表

### Figure 1: Pipeline / core system figure: Illustration of covariance adaptation via Newton-like approximation. Model-guided MPP

![Figure 1](https://arxiv.org/html/2602.03639v2/x1.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Pipeline / core system figure: Illustration of covariance adaptation via Newton-like approximation. Model-guided MPP”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 2: Comparison coarse vs fine model. (a) A large smoothing kernel  = 1.0 $\sigma=$1.0 supp

![Figure 2](https://arxiv.org/html/2602.03639v2/x2.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Comparison coarse vs fine model. (a) A large smoothing kernel  = 1.0 $\sigma=$1.0 supp”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 3: Convergence comparison of Hessian approximations on the cart-pole swing-up task. We c

![Figure 3](https://arxiv.org/html/2602.03639v2/x5.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Convergence comparison of Hessian approximations on the cart-pole swing-up task. We c”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### 论文表格

#### Table 1: TABLE I: Mean iterations ± $\pm$ std (failures) 100 seeds.

| Function | Model-Guided MPPI (ours) | Vanilla MPPI | CMA-ES |
| --- | --- | --- | --- |
| Rosenbrock | 6.9 ± 2.5 6.9\pm 2.5 — | 21.4 ± 7.3 21.4\pm 7.3 — | 12.0 ± 1.3 12.0\pm 1.3 — |
| Styblinski–Tang | 2.1 ± 0.4 2.1\pm 0.4 — | 9.2 ± 0.8 9.2\pm 0.8 (14) (14) | 7.8 ± 2.9 7.8\pm 2.9 — |
| Rastrigin | 3.0 ± 1.1 3.0\pm 1.1 — | 6.9 ± 6.4 6.9\pm 6.4 — | 5.5 ± 4.6 5.5\pm 4.6 — |
| Ackley | 3.3 ± 1.6 3.3\pm 1.6 (1) (1) | 3.8 ± 2.1 3.8\pm 2.1 — | 3.5 ± 1.4 3.5\pm 1.4 — |

**说明**: TABLE I: Mean iterations ± $\pm$ std (failures) 100 seeds.

#### Table 2: TABLE II: Per-iteration timings ± $\pm$ std 10 seeds using N N samples and percentages denote share of total time. S

| Task - (N) (N) | Model Construction [ms] | | Sampling r k r^{k} or f f [ms] | |
| --- | --- | --- | --- | --- |
| SB - (100) (100) | 0.2 ± 0.0 0.2\pm 0.0 | (12%) | 1.4 ± 0.2 1.4\pm 0.2 | (88%) |
| CP - (2) (2) | 3.7 ± 0.5 3.7\pm 0.5 | (75%) | 1.2 ± 1.8 1.2\pm 1.8 | (25%) |
| CP - (8) (8) | — | (70%) | 1.5 ± 3.5 1.5\pm 3.5 | (30%) |
| CP - (64) (64) | — | (52%) | 3.6 ± 2.1 3.6\pm 2.1 | (48%) |
| CP - (256) (256) | — | (26%) | 10.8 ± 3.0 10.8\pm 3.0~\, | (74%) |
| CP - (1024) (1024) | — | (0 9%) | 39.5 ± 4.5 39.5\pm 4.5~\, | (91%) |
| SF - (64) (64) | 1412.1 ± 156.6 1412.1\pm 156.6~ | (68%) | 706.4 ± 78.1 706.4\pm 78.1~\, | (32%) |

**说明**: TABLE II: Per-iteration timings ± $\pm$ std 10 seeds using N N samples and percentages denote share of total time. Static benchmark (SB) uses analytical derivatives, cart-pole (CP) uses AD-based exact derivatives and non-smooth single-finger (SF) manipulation uses randomized smoothing.
## 实验解读

- 评价重点:围绕 接触推理、接触丰富操作、实时控制、机器人操作,优先看真实机器人结果、扰动恢复、接触稳定性、控制误差、成功率、实时性和 sim-to-real 差距。
- System0 视角:实验结论不只看最终成功率,还要看闭环过程中是否能处理状态估计误差、执行延迟、碰撞/接触突变和外部扰动。
- 与上层系统的关系:如果方法能在快闭环内稳定工作,它可以作为 System1 技能或 System2 规划的底层执行保障;如果只在离线或慢速设定下成立,则还不能直接作为身体级快闭环模块。
## 批判性思考

### 优点

- 论文把 接触推理、接触丰富操作、实时控制、机器人操作 放到机器人身体级闭环里讨论,和只做高层语义推理的工作相比,更接近真实机器人部署时会遇到的状态、动力学和安全约束。
- 若论文同时给出公式、系统结构图、硬件实验或消融结果,这些证据可以帮助判断方法是否真的能支撑毫秒到百毫秒级反应。

### 局限

- 需要重点核查控制频率、推理延迟、传感延迟、失败恢复、硬件磨损、未见场景泛化和安全边界;这些因素决定它能否成为可靠的 System0 模块。
- 对学习型方法,还要区分训练环境中的鲁棒性和真实部署中的鲁棒性,尤其是接触丰富任务、动态运动任务和人形机器人任务。
## 论文定位

- 研究对象:Variance-Reduced Model Predictive Path Integral via Quadratic Model Approximation。
- 关键词:接触推理、接触丰富操作、实时控制、机器人操作。
- System0 关联:关注机器人身体级快反应链路中“状态/接触/局部感知 -> 约束/优化/策略 -> 动作”的短闭环,不把问题停留在语言规划或任务分解层。

## 速查卡片

> [!summary] Variance Reduced Predictive
> - **论文**: https://www.roboticsproceedings.org/rss22/p108.pdf
> - **arXiv**: http://arxiv.org/abs/2602.03639v2
> - **arXiv HTML**: https://arxiv.org/html/2602.03639v2
