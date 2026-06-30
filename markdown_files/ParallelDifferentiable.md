---
title: "Parallel Differentiable Reachability for Learning and Planning with Certified Neural Dynamics and Controllers"
method_name: "Parallel Differentiable"
authors: ["Keyi Shen"]
year: 2026
venue: "RSS"
tags: ["robot-manipulation", "robust-control", "real-time-control", "safe-control", "contact-rich-manipulation", "closed-loop-control", "model-predictive-control", "navigation"]
image_source: "online"
arxiv_html: "https://arxiv.org/html/2605.25346v1"
---
# Parallel Differentiable
## 一句话总结

> Parallel Differentiable Reachability for Learning and Planning with Certified Neural Dynamics and Controllers 主要落在 [[aerial-robotics]]、[[certified-control]]、[[closed-loop-control]]、[[dynamics-modeling]] 这一类问题上;从 System0 视角看,它的价值在于把身体状态、接触变化、局部感知或动力学约束接到机器人低层闭环里,让动作生成更稳定、安全、实时。
## 核心贡献

1. 围绕 **Parallel Differentiable Reachability for Learning and Planning with Certified Neural Dynamics and Controllers** 建立了一个与 aerial-robotics、certified-control、closed-loop-control、dynamics-modeling、模型预测控制、non-prehensile-manipulation 相关的机器人问题设定,重点不是高层语义规划,而是身体级闭环中的状态变化、接触变化或控制约束。
2. 方法层面的关键价值在于把感知、动力学、约束优化、策略学习或硬件反馈组织成可执行的低层动作生成机制,适合放进 System0 的快反应模块中理解。
3. 对系统集成而言,这类工作补上了 System1 技能选择与电机级控制之间的一层:它负责把上层目标变成动态可行、可恢复、可安全过滤的动作。
4. 若论文包含真实机器人、接触任务或仿真到真实迁移,核心应看它是否报告了闭环稳定性、扰动恢复、硬件延迟、失败模式和跨场景泛化。
## 问题背景

这篇论文可以放在 System0 的“具身快系统”背景下阅读:机器人需要在很短时间内根据本体状态、接触、力/触觉、局部视觉或动力学预测做出动作,而不是等待慢速规划器重新推理整个任务。aerial-robotics、certified-control、closed-loop-control、dynamics-modeling、模型预测控制、non-prehensile-manipulation、reachability 相关问题的难点通常不在“知道要做什么”,而在动作执行过程中状态会快速偏离、接触会突然改变、约束会实时收紧,低层系统必须及时修正。

### System0 关联

精读时优先看三件事:第一,论文使用哪些快速反馈量,例如关节状态、接触状态、力觉、触觉、深度或局部视觉;第二,反馈如何进入 MPC、WBC、策略网络、安全过滤器或恢复控制器;第三,输出是否直接影响机器人下一步动作,而不是只服务离线分析或高层任务规划。
## 方法详解

### System0 视角解析

从 System0 角度看,这篇论文的方法部分应当围绕 aerial-robotics、certified-control、closed-loop-control、dynamics-modeling、模型预测控制、non-prehensile-manipulation、reachability、实时控制 来读。核心问题是:系统如何从快速变化的机器人状态中构造控制输入,如何处理接触、碰撞、约束、模型误差或执行延迟,以及如何把这些信息变成下一时刻的动作。

如果方法里有优化问题,重点看目标函数、约束和求解频率;如果方法里有学习策略,重点看观测空间、动作空间、训练扰动和部署时的闭环频率;如果方法里有硬件或传感器,重点看传感链路是否真的进入控制回路。这样读可以把论文和 System2 语言规划、System1 技能编排区分开来。

### 关键公式与机制

#### 公式 1: [[优化目标/约束]]

$$u_{t}\in{\mathcal{U}}:=\{u\mid\ {u}\leq u\leq\ {u}\}\subset{\mathbb{R}}^{k}$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 2: [[动力学或策略机制]]

$$\textstyle g_{j+1}(x_{0},\tau)=\text{Trunc}_{j}\left(x_{0}+\int_{0}^{\tau}f_{\text{ct,dyn}}\bigl(g_{j}(x_{0},s)\bigr)\,ds\right),$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 3: [[优化目标/约束]]

$${\mathcal{L}}_{\text{pred}}=\textstyle\frac{1}{M^{\prime}T_{h}}\sum_{m=0}^{M^{\prime}-1}\sum_{t=0}^{T_{h}-1}w_{t}\bigl\lVert x_{t+1}^{m}-\hat{x}_{t+1}^{m}\bigr \rVert_{2}^{2},$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 4: [[优化目标/约束]]

$${\mathcal{L}}_{\text{track}}=\frac{1}{M^{\prime}T_{t}}\sum_{m=0}^{M^{\prime}-1}\sum_{t=0}^{T_{t}-1}w_{t}\Bigl(\lVert u_{t}^{m}-\hat{u}_{t}^{m} \rVert_{2}^{2}+\gamma\lVert x_{t+1}^{m}-\hat{x}_{t+1}^{m} \rVert_{2}^{2}\Bigr),$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 5: [[优化目标/约束]]

$$O=\textstyle\sum_{t=t_{0}}^{t_{0}+H}c(\hat{x}_{t},u_{t})+C\max(0,-G\!\left({\mathcal{R}}_{t}({\boldsymbol{u}},{\mathcal{B}}_{\epsilon}(\hat{x}_{t_{0}}))\right))$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 6: [[优化目标/约束]]

$$\{{\mathcal{X}}_{t}^{m}\}_{t=1}^{T_{h}}\leftarrow\mathtt{DTReach}\big(f_{\text{dt,dyn}}^{\theta},{\mathcal{X}}_{0}^{m},u_{0:T_{h}-1}^{m}\big)$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 7: [[优化目标/约束]]

$$V_{{\mathcal{R}},\epsilon}^{m}\leftarrow\sum_{t=1}^{T_{h}}\sum_{j=0}^{n-1}\mathtt{width}({\mathcal{X}}_{t,j}^{m})$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 8: [[动力学或策略机制]]

$$x=\begin{bmatrix}x\quad y\quad z\quad\dot{x}\quad\dot{y}\quad\dot{z}\quad\phi\quad\theta\quad\psi\quad p\quad q\quad r\end{bmatrix}^{\top}\in\mathbb{R}^{12},$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 9: [[动力学或策略机制]]

$$b_{3}=R_{WB}e_{3}=\begin{bmatrix}\cos\phi\sin\theta\cos\psi+\sin\phi\sin\psi\\ \cos\phi\sin\theta\sin\psi-\sin\phi\cos\psi\\ \cos\phi\cos\theta\end{bmatrix}.$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 10: [[优化目标/约束]]

$$G\!\left({\mathcal{R}}_{t}({\boldsymbol{u}},{\mathcal{B}}_{\epsilon}(\hat{x}_{t_{0}}))\right)\geq 0,\quad t=t_{0},\dots,t_{0}+H,$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。
## 关键图表

### Figure 1: Pipeline / core system figure: Reachability-guided planning. We visualize three successful model predictive control

![Figure 1](https://arxiv.org/html/2605.25346v1/x1.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Pipeline / core system figure: Reachability-guided planning. We visualize three successful model predictive control”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 2: Overview of the proposed framework. (a) Our parallel differentiable reachability engin

![Figure 2](https://arxiv.org/html/2605.25346v1/x2.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Overview of the proposed framework. (a) Our parallel differentiable reachability engin”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 3: Benchmarking our JAX reachability tool. Top: Compared with DT CROWN reachability, our

![Figure 3](https://arxiv.org/html/2605.25346v1/x3.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Benchmarking our JAX reachability tool. Top: Compared with DT CROWN reachability, our”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### 论文表格

#### Table 1: TABLE I: Refinement strategies.

| Method | Volume | Runtime (s) |
| --- | --- | --- |
| Baseline | 91.94 | 0.0021 |
| Input splitting (8) | 27.56 | 0.0048 |
| Gradient (20 iters) | 18.18 | 0.1084 |

**说明**: TABLE I: Refinement strategies.

#### Table 2: TABLE III: Reachability on a 10-joint arm.

| Method | EEF X width | EEF Y width | Runtime (s) |
| --- | --- | --- | --- |
| Sampling (100k) | 0.4799 | 0.6256 | 1.0944 |
| Ours (no split) | 0.8414 | 1.2264 | 0.0531 |
| Ours (64 splits) | 0.8118 | 0.9756 | 0.1186 |
| Ours (512 splits) | 0.7353 | 0.8999 | 0.3854 |

**说明**: TABLE III: Reachability on a 10-joint arm.

#### Table 3: TABLE IV: Tightness long horizons on quadrotor.

| Horizon | 1  1\times | 2  2\times | 3  3\times | 4  4\times | 5  5\times |
| --- | --- | --- | --- | --- | --- |
| DT Dyn (no split) | 1.3221 | 1.3227 | 1.8782 | 1.8872 | 1.8898 |
| CT Ctl (no split) | 1.8648 | 2.1591 | 3.8261 | 3.8737 | 18.6154 |
| CT Ctl (8 splits, rpy) | 1.9171 | 2.9817 | 3.8261 | 3.0396 | 6.2407 |

**说明**: TABLE IV: Tightness long horizons on quadrotor.
## 实验解读

- 评价重点:围绕 aerial-robotics、certified-control、closed-loop-control、dynamics-modeling、模型预测控制,优先看真实机器人结果、扰动恢复、接触稳定性、控制误差、成功率、实时性和 sim-to-real 差距。
- System0 视角:实验结论不只看最终成功率,还要看闭环过程中是否能处理状态估计误差、执行延迟、碰撞/接触突变和外部扰动。
- 与上层系统的关系:如果方法能在快闭环内稳定工作,它可以作为 System1 技能或 System2 规划的底层执行保障;如果只在离线或慢速设定下成立,则还不能直接作为身体级快闭环模块。
## 批判性思考

### 优点

- 论文把 aerial-robotics、certified-control、closed-loop-control、dynamics-modeling、模型预测控制 放到机器人身体级闭环里讨论,和只做高层语义推理的工作相比,更接近真实机器人部署时会遇到的状态、动力学和安全约束。
- 若论文同时给出公式、系统结构图、硬件实验或消融结果,这些证据可以帮助判断方法是否真的能支撑毫秒到百毫秒级反应。

### 局限

- 需要重点核查控制频率、推理延迟、传感延迟、失败恢复、硬件磨损、未见场景泛化和安全边界;这些因素决定它能否成为可靠的 System0 模块。
- 对学习型方法,还要区分训练环境中的鲁棒性和真实部署中的鲁棒性,尤其是接触丰富任务、动态运动任务和人形机器人任务。
## 论文定位

- 研究对象:Parallel Differentiable Reachability for Learning and Planning with Certified Neural Dynamics and Controllers。
- 关键词:aerial-robotics、certified-control、closed-loop-control、dynamics-modeling、模型预测控制、non-prehensile-manipulation、reachability、实时控制、机器人操作、鲁棒控制。
- System0 关联:关注机器人身体级快反应链路中“状态/接触/局部感知 -> 约束/优化/策略 -> 动作”的短闭环,不把问题停留在语言规划或任务分解层。

## 速查卡片

> [!summary] Parallel Differentiable
> - **论文**: https://www.roboticsproceedings.org/rss22/p191.pdf
> - **arXiv**: http://arxiv.org/abs/2605.25346v1
> - **arXiv HTML**: https://arxiv.org/html/2605.25346v1
