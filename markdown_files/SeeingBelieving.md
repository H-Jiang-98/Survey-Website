---
title: "Seeing is Believing: Certified Perception-Based Control from Learned Visual Representations via System Level Synthesis"
method_name: "Seeing is Believing"
authors: ["Antoine Leeman"]
year: 2026
venue: "RSS"
tags: ["robust-control", "real-time-control", "legged-locomotion", "safe-control", "robot-generalization", "closed-loop-control", "humanoid", "navigation"]
image_source: "online"
arxiv_html: "https://arxiv.org/html/2604.24894"
---
# Seeing is Believing
## 一句话总结

> Seeing is Believing: Certified Perception-Based Control from Learned Visual Representations via System Level Synthesis 主要落在 [[aerial-robotics]]、[[certified-control]]、[[closed-loop-control]]、[[人形机器人]] 这一类问题上;从 System0 视角看,它的价值在于把身体状态、接触变化、局部感知或动力学约束接到机器人低层闭环里,让动作生成更稳定、安全、实时。
## 核心贡献

1. 围绕 **Seeing is Believing: Certified Perception-Based Control from Learned Visual Representations via System Level Synthesis** 建立了一个与 aerial-robotics、certified-control、closed-loop-control、人形机器人、足式运动、实时控制 相关的机器人问题设定,重点不是高层语义规划,而是身体级闭环中的状态变化、接触变化或控制约束。
2. 方法层面的关键价值在于把感知、动力学、约束优化、策略学习或硬件反馈组织成可执行的低层动作生成机制,适合放进 System0 的快反应模块中理解。
3. 对系统集成而言,这类工作补上了 System1 技能选择与电机级控制之间的一层:它负责把上层目标变成动态可行、可恢复、可安全过滤的动作。
4. 若论文包含真实机器人、接触任务或仿真到真实迁移,核心应看它是否报告了闭环稳定性、扰动恢复、硬件延迟、失败模式和跨场景泛化。
## 问题背景

这篇论文可以放在 System0 的“具身快系统”背景下阅读:机器人需要在很短时间内根据本体状态、接触、力/触觉、局部视觉或动力学预测做出动作,而不是等待慢速规划器重新推理整个任务。aerial-robotics、certified-control、closed-loop-control、人形机器人、足式运动、实时控制、鲁棒控制 相关问题的难点通常不在“知道要做什么”,而在动作执行过程中状态会快速偏离、接触会突然改变、约束会实时收紧,低层系统必须及时修正。

### System0 关联

精读时优先看三件事:第一,论文使用哪些快速反馈量,例如关节状态、接触状态、力觉、触觉、深度或局部视觉;第二,反馈如何进入 MPC、WBC、策略网络、安全过滤器或恢复控制器;第三,输出是否直接影响机器人下一步动作,而不是只服务离线分析或高层任务规划。
## 方法详解

### System0 视角解析

从 System0 角度看,这篇论文的方法部分应当围绕 aerial-robotics、certified-control、closed-loop-control、人形机器人、足式运动、实时控制、鲁棒控制、safe-control 来读。核心问题是:系统如何从快速变化的机器人状态中构造控制输入,如何处理接触、碰撞、约束、模型误差或执行延迟,以及如何把这些信息变成下一时刻的动作。

如果方法里有优化问题,重点看目标函数、约束和求解频率;如果方法里有学习策略,重点看观测空间、动作空间、训练扰动和部署时的闭环频率;如果方法里有硬件或传感器,重点看传感链路是否真的进入控制回路。这样读可以把论文和 System2 语言规划、System1 技能编排区分开来。

### 关键公式与机制

#### 公式 1: [[动力学或策略机制]]

$$\begin{bmatrix}\Delta\mathbf{x}\\ \Delta\mathbf{u}\end{bmatrix}=\ {\begin{bmatrix}\mathbf{\Phi}^{\mathrm{xw}}&\mathbf{\Phi}^{\mathrm{xe}}\\ \mathbf{\Phi}^{\mathrm{uw}}&\mathbf{\Phi}^{\mathrm{ue}}\end{bmatrix}}_{\mathbf{\Phi}}\begin{bmatrix}\mathbf{E}&\mathbf{0}\\ \mathbf{0}&\mathbf{F}\end{bmatrix}\begin{bmatrix}\mathbf{w}\\ \mathbf{e}\end{bmatrix},$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 2: [[优化目标/约束]]

$$\ell(\theta_{p},\theta_{h})=\lVert \operatorname{p}(\psi^{\mathrm{dino}}(y_{i});\theta_{p})-h^{r}(x_{i};\theta_{h})\rVert-\lambda\sigma_{\text{min}}(\mathcal{O}(\hat{x},\boldsymbol{u}))$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 3: [[动力学或策略机制]]

$$J_{\mathrm{error}}=\mathbb{E}\!\left[\sum_{k=0}^{T-1}\begin{bmatrix}\Delta x_{k}\\ \Delta u_{k}\end{bmatrix}^{\!\top}\!\begin{bmatrix}Q&0\\ 0&R\end{bmatrix}\!\begin{bmatrix}\Delta x_{k}\\ \Delta u_{k}\end{bmatrix}+\Delta x_{T}^{\top}P\Delta x_{T}\right].$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 4: [[优化目标/约束]]

$$\max_{w_{j},e_{j}}\hat{e}_{\ell}^{\top}\!\begin{bmatrix}\Delta x_{k}\\ \Delta u_{k}\end{bmatrix}\leq\sum_{j=0}^{k}\Big(\lVert \hat{e}_{\ell}^{\top}\Phi^{\mathrm{w}}_{k,j}\Sigma_{j} \rVert_{1}+\lVert \hat{e}_{\ell}^{\top}\Phi^{\mathrm{e}}_{k,j}\Upsilon_{j} \rVert_{1}\Big),$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 5: [[动力学或策略机制]]

$${\begin{bmatrix}\dot{p}_{x}\\ \dot{p}_{y}\\ \dot{p}_{z}\\ \dot{v}_{x}\\ \dot{v}_{y}\\ \dot{v}_{z}\\ \dot{\theta}_{x}\\ \dot{\theta}_{y}\\ \dot{\omega}_{x}\\ \dot{\omega}_{y}\end{bmatrix}}=\begin{bmatrix}v_{x}\\ v_{y}\\ v_{z}\\ g\tan(\theta_{x})-3v_{x}\\ g\tan(\theta_{y})-3v_{y}\\ u_{z}-g-v_{z}\\ -10\theta_{x}+\omega_{x}\\ -10\theta_{y}+\omega_{y}\\ -10\theta_{x}+50u_{x}\\ -10\theta_{y}+50u_{y}\end{bmatrix}$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 6: [[动力学或策略机制]]

$${\begin{bmatrix}\dot{p}_{x}\\ \dot{p}_{y}\\ \dot{\theta}\\ \dot{v}\\ \dot{\omega}\end{bmatrix}}=\begin{bmatrix}v\cos\theta\\ v\sin\theta\\ \omega\\ u_{1}\\ u_{2}\end{bmatrix}$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 7: [[动力学或策略机制]]

$$:=\mathbb{E}\!\left[\sum_{k=0}^{T-1}\begin{bmatrix}\Delta\hat{x}_{k}\\ \Delta u_{k}\end{bmatrix}^{\!\top}\!\begin{bmatrix}Q&0\\ 0&R\end{bmatrix}\!\begin{bmatrix}\Delta\hat{x}_{k}\\ \Delta u_{k}\end{bmatrix}+\Delta\hat{x}_{T}^{\top}P\Delta\hat{x}_{T}\right],$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 8: [[动力学或策略机制]]

$$$\sum_{j=0}^{k}\Bigl$\lVert c_{i}^{$\top$}$\boldsymbol{\Phi}^{\mathrm{w}}_{k,j}$\,$\Sigma_{j}\Bigr$\rVert$_{1}+\Bigl$\lVert c_{i}^{$\top$}$\boldsymbol{\Phi}^{\mathrm{e}}_{k,j}$\,$\Upsilon_{j}\Bigr$$\|_{1}$+$c_{i}^{\top}\begin{bmatrix}$z_{k}\\$v_{k}\end{bmatrix}+$$b_{i}\leq$ 0,$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 9: [[动力学或策略机制]]

$$$\textstyle\sum_{j=0}^{k}\big$\lVert$\hat{e}_{l}^{\top}$$\boldsymbol{\Phi}^{\mathrm{w}}_{k,j}$$\Sigma_{j}$$\big$\lVert$_{1}+\big$\lVert$\hat{e}_{l}^{\top}$$\boldsymbol{\Phi}^{\mathrm{e}}_{k,j}$$\Upsilon_{j}$$\big$\lVert$_{1}\leq\tau_{k}$,$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 10: [[动力学或策略机制]]

$$$\sum_{j=0}^{k}\Bigl$\lVert$\hat{e}_{l}^{\top}$$\boldsymbol{\Phi}^{\mathrm{w}}_{k,j}$\,$\Sigma_{j}\Bigr$\lVert$_{1}+\Bigl$\lVert$\hat{e}_{l}^{\top}$$\boldsymbol{\Phi}^{\mathrm{e}}_{k,j}$\,$\Upsilon_{j}\Bigr$\lVert$_{1}\leq\tau_{k}$,$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。
## 关键图表

### Figure 1: Pipeline / core system figure: We stabilize a Unitree G1 humanoid (59 states) around a backflip trajectory that is jo

![Figure 1](https://arxiv.org/html/2604.24894v1/x1.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Pipeline / core system figure: We stabilize a Unitree G1 humanoid (59 states) around a backflip trajectory that is jo”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 2: Online SLS execution: Images are mapped to low-dimensional observations p  ( dino

![Figure 2](https://arxiv.org/html/2604.24894v1/x2.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Online SLS execution: Images are mapped to low-dimensional observations p  ( dino ”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 3: On the classic light-dark domain, our method optimally visits the low perception-error

![Figure 3](https://arxiv.org/html/2604.24894v1/x3.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“On the classic light-dark domain, our method optimally visits the low perception-error”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### 论文表格

#### Table 1: TABLE I: Comparison between our method, certainty equivalent (CE) and non-robust (NR) baseline in terms of success rate

| | Light-dark | 4D Car | | | | |
| --- | --- | --- | --- | --- | --- | --- |
| | Ours | CE | NR | Ours | CE | NR |
| SR | 100% | 90.13% | 87.73% | 98.53% | 93.47% | 48.13% |
| CVR | 0% | 9.87% | 12.27% | 1.6% | 6.53% | 55.87% |
| | Quadrotor | Humanoid | | | | |
| | Ours | CE | NR | Ours | CE | NR |
| SR | 100% | 100% | 29.47% | 100% | 100% | 2% |
| CVR | 0% | 0% | 92.8% | 0% | 0% | 98% |

**说明**: TABLE I: Comparison between our method, certainty equivalent (CE) and non-robust (NR) baseline in terms of success rate (SR) and constraint violation rate (CVR).

#### Table 2: TABLE II: Error bound coverage vs. number of calibration points.

| calibration points | 50 | 100 | 250 | 500 | 1500 |
| --- | --- | --- | --- | --- | --- |
| empirical coverage | 91.4% | 92.2% | 97.4% | 99% | 99% |

**说明**: TABLE II: Error bound coverage vs. number of calibration points.
## 实验解读

- 评价重点:围绕 aerial-robotics、certified-control、closed-loop-control、人形机器人、足式运动,优先看真实机器人结果、扰动恢复、接触稳定性、控制误差、成功率、实时性和 sim-to-real 差距。
- System0 视角:实验结论不只看最终成功率,还要看闭环过程中是否能处理状态估计误差、执行延迟、碰撞/接触突变和外部扰动。
- 与上层系统的关系:如果方法能在快闭环内稳定工作,它可以作为 System1 技能或 System2 规划的底层执行保障;如果只在离线或慢速设定下成立,则还不能直接作为身体级快闭环模块。
## 批判性思考

### 优点

- 论文把 aerial-robotics、certified-control、closed-loop-control、人形机器人、足式运动 放到机器人身体级闭环里讨论,和只做高层语义推理的工作相比,更接近真实机器人部署时会遇到的状态、动力学和安全约束。
- 若论文同时给出公式、系统结构图、硬件实验或消融结果,这些证据可以帮助判断方法是否真的能支撑毫秒到百毫秒级反应。

### 局限

- 需要重点核查控制频率、推理延迟、传感延迟、失败恢复、硬件磨损、未见场景泛化和安全边界;这些因素决定它能否成为可靠的 System0 模块。
- 对学习型方法,还要区分训练环境中的鲁棒性和真实部署中的鲁棒性,尤其是接触丰富任务、动态运动任务和人形机器人任务。
## 论文定位

- 研究对象:Seeing is Believing: Certified Perception-Based Control from Learned Visual Representations via System Level Synthesis。
- 关键词:aerial-robotics、certified-control、closed-loop-control、人形机器人、足式运动、实时控制、鲁棒控制、safe-control、visuomotor-policy。
- System0 关联:关注机器人身体级快反应链路中“状态/接触/局部感知 -> 约束/优化/策略 -> 动作”的短闭环,不把问题停留在语言规划或任务分解层。

## 速查卡片

> [!summary] Seeing is Believing
> - **论文**: https://www.roboticsproceedings.org/rss22/p172.pdf
> - **arXiv**: http://arxiv.org/abs/2604.24894
> - **arXiv HTML**: https://arxiv.org/html/2604.24894
