---
title: "PRIME: Physically-consistent Robotic Inertial and Motion Estimation for Legged and Humanoid Robots"
method_name: "PRIME"
authors: ["Jiarong Kang"]
year: 2026
venue: "RSS"
tags: ["contact-reasoning", "robust-control", "legged-locomotion", "contact-rich-manipulation", "robot-generalization", "humanoid", "state-estimation", "recovery"]
image_source: "online"
arxiv_html: "https://arxiv.org/html/2605.17681"
---
# PRIME
## 一句话总结

> PRIME: Physically-consistent Robotic Inertial and Motion Estimation for Legged and Humanoid Robots 主要落在 [[actuator-modeling]]、[[contact-estimation]]、[[contact-implicit-optimization]]、[[接触推理]] 这一类问题上;从 System0 视角看,它的价值在于把身体状态、接触变化、局部感知或动力学约束接到机器人低层闭环里,让动作生成更稳定、安全、实时。
## 核心贡献

1. 围绕 **PRIME: Physically-consistent Robotic Inertial and Motion Estimation for Legged and Humanoid Robots** 建立了一个与 actuator-modeling、contact-estimation、contact-implicit-optimization、接触推理、接触丰富操作、foundation-model 相关的机器人问题设定,重点不是高层语义规划,而是身体级闭环中的状态变化、接触变化或控制约束。
2. 方法层面的关键价值在于把感知、动力学、约束优化、策略学习或硬件反馈组织成可执行的低层动作生成机制,适合放进 System0 的快反应模块中理解。
3. 对系统集成而言,这类工作补上了 System1 技能选择与电机级控制之间的一层:它负责把上层目标变成动态可行、可恢复、可安全过滤的动作。
4. 若论文包含真实机器人、接触任务或仿真到真实迁移,核心应看它是否报告了闭环稳定性、扰动恢复、硬件延迟、失败模式和跨场景泛化。
## 问题背景

这篇论文可以放在 System0 的“具身快系统”背景下阅读:机器人需要在很短时间内根据本体状态、接触、力/触觉、局部视觉或动力学预测做出动作,而不是等待慢速规划器重新推理整个任务。actuator-modeling、contact-estimation、contact-implicit-optimization、接触推理、接触丰富操作、foundation-model、人形机器人 相关问题的难点通常不在“知道要做什么”,而在动作执行过程中状态会快速偏离、接触会突然改变、约束会实时收紧,低层系统必须及时修正。

### System0 关联

精读时优先看三件事:第一,论文使用哪些快速反馈量,例如关节状态、接触状态、力觉、触觉、深度或局部视觉;第二,反馈如何进入 MPC、WBC、策略网络、安全过滤器或恢复控制器;第三,输出是否直接影响机器人下一步动作,而不是只服务离线分析或高层任务规划。
## 方法详解

### System0 视角解析

从 System0 角度看,这篇论文的方法部分应当围绕 actuator-modeling、contact-estimation、contact-implicit-optimization、接触推理、接触丰富操作、foundation-model、人形机器人、inertial-estimation 来读。核心问题是:系统如何从快速变化的机器人状态中构造控制输入,如何处理接触、碰撞、约束、模型误差或执行延迟,以及如何把这些信息变成下一时刻的动作。

如果方法里有优化问题,重点看目标函数、约束和求解频率;如果方法里有学习策略,重点看观测空间、动作空间、训练扰动和部署时的闭环频率;如果方法里有硬件或传感器,重点看传感链路是否真的进入控制回路。这样读可以把论文和 System2 语言规划、System1 技能编排区分开来。

### 关键公式与机制

#### 公式 1: [[动力学或策略机制]]

$$\mathbf{q}^{+}-\mathbf{q} \quad =\Delta t\,\mathbf{v}^{+}, \quad \mathbf{M}(\mathbf{q})({\mathbf{v}}^{+}-{\mathbf{v}}) \quad =\Delta t\big(\mathbf{B}\mathbf{u}-\mathbf{h}(\mathbf{q},\mathbf{v})+\textstyle\sum_{i}\mathbf{J}_{i}(\mathbf{q})^{\intercal}\mathbf{f}_{i}\big),$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 2: [[优化目标/约束]]

$$\mathcal{I} \quad =\mathbf{U}\mathbf{U}^{\intercal},\quad\mathbf{U}=e^{\alpha}\begin{bmatrix}e^{d_{1}}&s_{12}&s_{13}&t_{1}\\ 0&e^{d_{2}}&s_{23}&t_{2}\\ 0&0&e^{d_{3}}&t_{3}\\ 0&0&0&1\end{bmatrix}, \quad \boldsymbol{\theta} \quad =\bigl[\alpha\ d_{1}\ d_{2}\ d_{3}\ s_{12}\ s_{23}\ s_{13}\ t_{1}\ t_{2}\ t_{3}\bigr]^{\intercal}\in\mathbb{R}^{10}.$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 3: [[优化目标/约束]]

$$\min_{\mathbf{x}_{[0,T]},\boldsymbol{\delta}_{[0,T]}}\quad\Gamma(\mathbf{x}_{0}) \quad +\sum_{k=0}^{T-1}|| $\boldsymbol{\delta}^{\mathbf{x}}_{k}$ || ^{2}_{ $\text{Cov}_{\mathbf{x}}^{-1}}+\sum_{k=0}^{T}$ || $\boldsymbol{\delta}^{\mathbf{y}}_{k}$ ||^{2}_{\text{Cov}_{\mathbf{y}}^{-1}} \quad \mathbf{x}_{k+1}\leftarrow{\mathcal{F}}(\mathbf{x}_{k},\mathbf{u}_{k})\oplus\boldsymbol{\delta}^{\mathbf{x}}_{k},\ \forall k\in\{0,...,T-1\} \quad \mathbf{y}_{k}=\textbf{Meas}(\mathbf{x}_{k})\oplus\boldsymbol{\delta}^{\mathbf{y}}_{k},\ \forall k\in\{0,...,T\}, \quad \mathbf{0}\leq\textbf{Constr}(\mathbf{x}_{k}),\ \forall k\in\{0,...,T\},$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 4: [[优化目标/约束]]

$$\min_{\mathbf{x}_{[0,T]},\boldsymbol{\delta}_{[0,T]},\boldsymbol{\theta}}\quad|| $\boldsymbol{\theta}-\hat{\boldsymbol{\theta}}$ || ^{2}_{ $\text{Cov}_{\boldsymbol{\theta}}^{-1}}+\Gamma(\mathbf{x}_{0}) \quad +\sum_{k=0}^{T-1}$ || $\boldsymbol{\delta}^{\mathbf{x}}_{k}$ || ^{2}_{ $\text{Cov}_{\mathbf{x}}^{-1}}+\sum_{k=0}^{T}$ || $\boldsymbol{\delta}^{\mathbf{y}}_{k}$ ||^{2}_{\text{Cov}_{\mathbf{y}}^{-1}} \quad \mathbf{x}_{k+1}\leftarrow\mathcal{F}(\mathbf{x}_{k},\mathbf{u}_{k},\boldsymbol{\theta})\oplus\boldsymbol{\delta}^{\mathbf{x}}_{k},\ \forall k\in\{0,...,T-1\} \quad \mathbf{y}_{k}=\textbf{Meas}(\mathbf{x}_{k})\oplus\boldsymbol{\delta}^{\mathbf{y}}_{k},\ \forall k\in\{0,...,T\}.$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 5: [[优化目标/约束]]

$$A \quad =\partial_{\bar{\mathbf{x}}}\mathcal{F}(\tilde{\mathbf{x}},\tilde{\boldsymbol{\delta}},\tilde{\boldsymbol{\theta}}), \quad B \quad =\partial_{\boldsymbol{\delta}}\mathcal{F}(\tilde{\mathbf{x}},\tilde{\boldsymbol{\delta}},\tilde{\boldsymbol{\theta}}), \quad \nabla_{\bar{\mathbf{x}}}Q \quad =\nabla_{\bar{\mathbf{x}}}L+A^{\intercal}\nabla_{\bar{\mathbf{x}}}{V}, \quad \nabla_{\boldsymbol{\delta}}Q \quad =\nabla_{\boldsymbol{\delta}}L+B^{\intercal}\nabla_{\bar{\mathbf{x}}}{V}, \quad \nabla_{\bar{\mathbf{x}}}^{2}Q \quad =\nabla_{\bar{\mathbf{x}}}^{2}L+A^{\intercal}(\nabla_{\bar{\mathbf{x}}}^{2}{V})A, \quad \nabla_{\boldsymbol{\delta}}^{2}Q \quad =\nabla_{\boldsymbol{\delta}}^{2}L+B^{\intercal}(\nabla_{\bar{\mathbf{x}}}^{2}{V})B, \quad \nabla_{\boldsymbol{\delta}{\bar{\mathbf{x}}}}Q \quad =\nabla_{\boldsymbol{\delta}\bar{\mathbf{x}}}L+B^{\intercal}(\nabla_{\bar{\mathbf{x}}}^{2}{V})A.$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 6: [[动力学或策略机制]]

$$k_{i}=-\nabla_{\boldsymbol{\delta}}^{2}Q^{-1}\nabla_{\boldsymbol{\delta}}Q \quad K_{i}=-\nabla_{\boldsymbol{\delta}}^{2}Q^{-1}\nabla_{\boldsymbol{\delta}{\bar{\mathbf{x}}}}Q, \quad \nabla_{\bar{\mathbf{x}}}{V}=\nabla_{\bar{\mathbf{x}}}Q+K_{i}^{\intercal}\nabla_{\boldsymbol{\delta}}Q, \quad \nabla_{\bar{\mathbf{x}}}^{2}{V}=\nabla_{\bar{\mathbf{x}}}^{2}Q+K_{i}^{\intercal}\nabla_{\boldsymbol{\delta}\bar{\mathbf{x}}}Q.$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 7: [[优化目标/约束]]

$$\boldsymbol{\delta}_{i}^{{}^{\prime}} \quad =\boldsymbol{\delta}_{i}+\alpha k_{i}+K_{i}\delta\bar{\mathbf{x}}_{i}, \quad \bar{\mathbf{x}}_{i+1}^{{}^{\prime}} \quad =\begin{bmatrix}\mathcal{F}(\mathbf{x}_{i}^{{}^{\prime}},\boldsymbol{\delta}_{i}^{{}^{\prime}},\boldsymbol{\theta}^{{}^{\prime}})\\ \boldsymbol{\theta}^{{}^{\prime}}\end{bmatrix}, \quad \delta\bar{\mathbf{x}}_{i+1} \quad =\bar{\mathbf{x}}_{i+1}^{{}^{\prime}}-\begin{bmatrix}\mathbf{x}_{i}\\ \boldsymbol{\theta}\end{bmatrix}, \quad V^{{}^{\prime}} \quad =V^{{}^{\prime}}+L_{i}(\bar{\mathbf{x}}_{i}^{{}^{\prime}},\boldsymbol{\delta}_{i}^{{}^{\prime}}).$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 8: [[动力学或策略机制]]

$$Q(\mathbf{x},\boldsymbol{\delta}) \quad \approx Q(\hat{\mathbf{x}},\hat{\boldsymbol{\delta}})+\Delta Q, \quad \Delta Q \quad =\frac{1}{2}\begin{bmatrix}1\\ \delta\mathbf{x}\\ \delta\boldsymbol{\delta}\end{bmatrix}^{\intercal}\begin{bmatrix}0&\nabla_{\mathbf{x}}Q^{\intercal}&\nabla_{\boldsymbol{\delta}}Q^{\intercal}\\ \nabla_{\mathbf{x}}Q&\nabla_{\mathbf{x}}^{2}Q&\nabla_{\mathbf{x}\boldsymbol{\delta}}Q\\ \nabla_{\boldsymbol{\delta}}Q&\nabla_{\boldsymbol{\delta}\mathbf{x}}Q&\nabla_{\boldsymbol{\delta}}^{2}Q\end{bmatrix}\begin{bmatrix}1\\ \delta\mathbf{x}\\ \delta\boldsymbol{\delta}\end{bmatrix}.$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 9: [[优化目标/约束]]

$$\bar{\mathbf{x}}^{+}=\begin{bmatrix}\mathbf{x}^{+}\\ \boldsymbol{\theta}^{+}\end{bmatrix}\leftarrow\begin{bmatrix}\mathcal{F}(\mathbf{x},\mathbf{u},\boldsymbol{\theta})\oplus\boldsymbol{\delta}^{\mathbf{x}}\\ \boldsymbol{\theta}\end{bmatrix}.$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 10: [[动力学或策略机制]]

$$\mathbf{M}(\mathbf{q})\dot{\mathbf{v}}+\mathbf{h}(\mathbf{q},\mathbf{v})=\mathbf{B}^{\top}\boldsymbol{\tau}+\textstyle\sum_{i}\mathbf{J}_{i}(\mathbf{q})^{\intercal}\mathbf{w}_{i},$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。
## 关键图表

### Figure 1: Overview of PRIME: PRIME reconstructs physically consistent robot trajectories and hardware-matc

![Figure 1](https://arxiv.org/html/2605.17681v1/figure/FirstPlot_new.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Overview of PRIME: PRIME reconstructs physically consistent robot trajectories and hardware-matc”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 2: a Robot experimental configurations for Unitree Go2 and G1. Point contact model is used to appro

![Figure 2](https://arxiv.org/html/2605.17681v1/figure/Configurations_forceplate.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“a Robot experimental configurations for Unitree Go2 and G1. Point contact model is used to appro”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 3: a Force-estimation comparison between our method and the contact-constrained estimation baseline

![Figure 3](https://arxiv.org/html/2605.17681v1/x2.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“a Force-estimation comparison between our method and the contact-constrained estimation baseline”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。
## 实验解读

- 评价重点:围绕 actuator-modeling、contact-estimation、contact-implicit-optimization、接触推理、接触丰富操作,优先看真实机器人结果、扰动恢复、接触稳定性、控制误差、成功率、实时性和 sim-to-real 差距。
- System0 视角:实验结论不只看最终成功率,还要看闭环过程中是否能处理状态估计误差、执行延迟、碰撞/接触突变和外部扰动。
- 与上层系统的关系:如果方法能在快闭环内稳定工作,它可以作为 System1 技能或 System2 规划的底层执行保障;如果只在离线或慢速设定下成立,则还不能直接作为身体级快闭环模块。
## 批判性思考

### 优点

- 论文把 actuator-modeling、contact-estimation、contact-implicit-optimization、接触推理、接触丰富操作 放到机器人身体级闭环里讨论,和只做高层语义推理的工作相比,更接近真实机器人部署时会遇到的状态、动力学和安全约束。
- 若论文同时给出公式、系统结构图、硬件实验或消融结果,这些证据可以帮助判断方法是否真的能支撑毫秒到百毫秒级反应。

### 局限

- 需要重点核查控制频率、推理延迟、传感延迟、失败恢复、硬件磨损、未见场景泛化和安全边界;这些因素决定它能否成为可靠的 System0 模块。
- 对学习型方法,还要区分训练环境中的鲁棒性和真实部署中的鲁棒性,尤其是接触丰富任务、动态运动任务和人形机器人任务。
## 论文定位

- 研究对象:PRIME: Physically-consistent Robotic Inertial and Motion Estimation for Legged and Humanoid Robots。
- 关键词:actuator-modeling、contact-estimation、contact-implicit-optimization、接触推理、接触丰富操作、foundation-model、人形机器人、inertial-estimation、足式运动、proprioception。
- System0 关联:关注机器人身体级快反应链路中“状态/接触/局部感知 -> 约束/优化/策略 -> 动作”的短闭环,不把问题停留在语言规划或任务分解层。

## 速查卡片

> [!summary] PRIME
> - **论文**: https://www.roboticsproceedings.org/rss22/p029.pdf
> - **arXiv**: https://arxiv.org/abs/2605.17681
> - **arXiv HTML**: https://arxiv.org/html/2605.17681
