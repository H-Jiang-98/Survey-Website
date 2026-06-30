---
title: "TinySDP: Real Time Semidefinite Optimization for Certifiable and Agile Edge Robotics"
method_name: "TinySDP"
authors: ["Ishaan Mahajan"]
year: 2026
venue: "RSS"
tags: ["real-time-control", "reinforcement-learning", "safe-control", "agile-locomotion", "collision-avoidance", "model-predictive-control", "navigation"]
image_source: "online"
arxiv_html: "https://arxiv.org/html/2605.13748v1"
---
# TinySDP
## 一句话总结

> TinySDP: Real Time Semidefinite Optimization for Certifiable and Agile Edge Robotics 主要落在 [[aerial-robotics]]、[[agile-locomotion]]、[[certified-control]]、[[碰撞避免]] 这一类问题上;从 System0 视角看,它的价值在于把身体状态、接触变化、局部感知或动力学约束接到机器人低层闭环里,让动作生成更稳定、安全、实时。
## 核心贡献

1. 围绕 **TinySDP: Real Time Semidefinite Optimization for Certifiable and Agile Edge Robotics** 建立了一个与 aerial-robotics、agile-locomotion、certified-control、碰撞避免、模型预测控制、navigation 相关的机器人问题设定,重点不是高层语义规划,而是身体级闭环中的状态变化、接触变化或控制约束。
2. 方法层面的关键价值在于把感知、动力学、约束优化、策略学习或硬件反馈组织成可执行的低层动作生成机制,适合放进 System0 的快反应模块中理解。
3. 对系统集成而言,这类工作补上了 System1 技能选择与电机级控制之间的一层:它负责把上层目标变成动态可行、可恢复、可安全过滤的动作。
4. 若论文包含真实机器人、接触任务或仿真到真实迁移,核心应看它是否报告了闭环稳定性、扰动恢复、硬件延迟、失败模式和跨场景泛化。
## 问题背景

这篇论文可以放在 System0 的“具身快系统”背景下阅读:机器人需要在很短时间内根据本体状态、接触、力/触觉、局部视觉或动力学预测做出动作,而不是等待慢速规划器重新推理整个任务。aerial-robotics、agile-locomotion、certified-control、碰撞避免、模型预测控制、navigation、实时控制 相关问题的难点通常不在“知道要做什么”,而在动作执行过程中状态会快速偏离、接触会突然改变、约束会实时收紧,低层系统必须及时修正。

### System0 关联

精读时优先看三件事:第一,论文使用哪些快速反馈量,例如关节状态、接触状态、力觉、触觉、深度或局部视觉;第二,反馈如何进入 MPC、WBC、策略网络、安全过滤器或恢复控制器;第三,输出是否直接影响机器人下一步动作,而不是只服务离线分析或高层任务规划。
## 方法详解

### System0 视角解析

从 System0 角度看,这篇论文的方法部分应当围绕 aerial-robotics、agile-locomotion、certified-control、碰撞避免、模型预测控制、navigation、实时控制、强化学习 来读。核心问题是:系统如何从快速变化的机器人状态中构造控制输入,如何处理接触、碰撞、约束、模型误差或执行延迟,以及如何把这些信息变成下一时刻的动作。

如果方法里有优化问题,重点看目标函数、约束和求解频率;如果方法里有学习策略,重点看观测空间、动作空间、训练扰动和部署时的闭环频率;如果方法里有硬件或传感器,重点看传感链路是否真的进入控制回路。这样读可以把论文和 System2 语言规划、System1 技能编排区分开来。

### 关键公式与机制

#### 公式 1: [[优化目标/约束]]

$$\ {\ell_{f}(x_{N})+\sum_{k=0}^{N-1}\left(\ell(x_{k},u_{k})+\mathcal{I}_{\mathcal{U}}(z_{k})+\frac{\rho}{2}\lVert u_{k}-z_{k}+y_{k} \rVert_{2}^{2}\right).}_{\mathcal{L}_{\rho}}$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 2: [[优化目标/约束]]

$$\mathcal{O}_{k,j}:=\{p\in\mathbb{R}^{2}:\lVert p-c_{k,j} \rVert_{2}^{2}\leq r_{k,j}^{2}\},$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 3: [[动力学或策略机制]]

$$\bar{x}_{k}:=\begin{bmatrix}x_{k}\\ \mathrm{vec}(X_{k})\end{bmatrix},\;\bar{u}_{k}:=\begin{bmatrix}u_{k}\\ \mathrm{vec}(XU_{k})\\ \mathrm{vec}(UX_{k})\\ \mathrm{vec}(UU_{k})\end{bmatrix},\;\bar{x}_{0}=\begin{bmatrix}x_{0}\\ \mathrm{vec}(x_{0}x_{0}^{\top})\end{bmatrix},$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 4: [[动力学或策略机制]]

$${$\bar{A}=\begin{bmatrix}A&0\\ 0&A\otimes A\end{bmatrix},\;\bar{B}=\begin{bmatrix}B&0&0&0\\ 0&B\otimes A&A\otimes B&B\otimes B\end{bmatrix}$}.$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 5: [[动力学或策略机制]]

$$M_{k}(\bar{x}_{k},\bar{u}_{k}):=\begin{bmatrix}1&x_{k}^{\top}&u_{k}^{\top}\\ x_{k}&X_{k}&XU_{k}\\ u_{k}&UX_{k}&UU_{k}\end{bmatrix}\in\mathbb{S}^{1+n_{x}+n_{u}}.$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 6: [[优化目标/约束]]

$$\Pi_{\mathbb{S}_{+}^{p}}(Y):=\arg\min_{X\succeq 0}\ \lVert X-Y \rVert_{F}^{2}.$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 7: [[优化目标/约束]]

$$\Pi_{\mathbb{S}_{+}^{p}}(Y)=V\operatorname{diag}\!\big(\max(\lambda,0)\big)V^{\top},$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 8: [[动力学或策略机制]]

$$=\operatorname*{arg\,min}_{\bar{x},u}\sum_{k=0}^{N-1}\tilde{\ell}_{k}(\bar{x}_{k},u_{k};S_{k},H_{k}),$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 9: [[优化目标/约束]]

$$\ell_{f}(x_{N})+\sum_{k=0}^{N-1}\ell(x_{k},u_{k})+\mathcal{I}_{\mathcal{U}}(z_{k})$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 10: [[优化目标/约束]]

$$=\operatorname*{arg\,min}_{z}\mathcal{L}_{\rho}(x^{+},u^{+},z,y)$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。
## 关键图表

### Figure 1: Pipeline / core system figure: Front (top) and side (bottom) views of the Crazyflie running TinySDP online to avoid a

![Figure 1](https://arxiv.org/html/2605.13748v1/images/hardware-tinysdp.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Pipeline / core system figure: Front (top) and side (bottom) views of the Crazyflie running TinySDP online to avoid a”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 2: Static U-shape benchmark. Trajectories for TinySDP (blue), RPCBF (red), TinyMPC-LIN (g

![Figure 2](https://arxiv.org/html/2605.13748v1/images/static_experiments.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Static U-shape benchmark. Trajectories for TinySDP (blue), RPCBF (red), TinyMPC-LIN (g”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 3: Dynamic moving-gap benchmark. Time-lapse comparison of TinySDP (blue) against RPCBF (r

![Figure 3](https://arxiv.org/html/2605.13748v1/images/dynamic_experiments.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Dynamic moving-gap benchmark. Time-lapse comparison of TinySDP (blue) against RPCBF (r”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### 论文表格

#### Table 1: TABLE I: Static U-shape results for four initial conditions.

| Start | Method | Path Len | Goal Dist | Safe |
| --- | --- | --- | --- | --- |
| Inside | TinySDP (ours) | 17.95 | 0.006 | ✓ |
| | RPCBF | 26.03 | 0.091 | ✓ |
| | TinyMPC-LIN (m=3.1m) | 18.38 | 1.400 | ✓ |
| | TinyMPC-LIN (m=0m) | – | – | ✗ |
| | TinyMPC-HOCBF (m=3m) | – | – | ✗ |
| | TinyMPC-HOCBF (m=0m) | – | – | ✗ |
| Outside | TinySDP (ours) | 10.15 | 0.021 | ✓ |
| Center | RPCBF | 31.03 | 0.093 | ✓ |
| | TinyMPC-LIN (m=3.1m) | 24.58 | 1.400 | ✓ |
| | TinyMPC-LIN (m=0m) | – | – | ✗ |
| | TinyMPC-HOCBF (m=3m) | – | – | ✗ |
| | TinyMPC-HOCBF (m=0m) | – | – | ✗ |
| Edge Up | TinySDP (ours) | 9.93 | 0.023 | ✓ |
| | RPCBF | 36.81 | 0.132 | ✓ |
| | TinyMPC-LIN (m=3.1m) | 18.58 | 1.400 | ✓ |
| | TinyMPC-LIN (m=0m) | – | – | ✗ |
| | TinyMPC-HOCBF (m=3m) | – | – | ✗ |
| | TinyMPC-HOCBF (m=0m) | – | – | ✗ |
| Edge Down | TinySDP (ours) | 9.93 | 0.023 | ✓ |
| | RPCBF | 36.25 | 0.108 | ✓ |
| | TinyMPC-LIN (m=3.1m) | 23.64 | 1.400 | ✓ |
| | TinyMPC-LIN (m=0m) | – | – | ✗ |
| | TinyMPC-HOCBF (m=3m) | – | – | ✗ |
| | TinyMPC-HOCBF (m=0m) | – | – | ✗ |

**说明**: TABLE I: Static U-shape results for four initial conditions.

#### Table 2: TABLE II: Dynamic obstacle scenario results.

| Method | Path Len | Goal Dist | Safe |
| --- | --- | --- | --- |
| TinySDP (ours) | 19.11 | 0.018 | ✓ |
| RPCBF | 27.33 | 0.069 | ✓ |
| TinyMPC-LIN (m=1.5m) | 13.61 | 0.023 | ✓ |
| TinyMPC-LIN (m=0m) | – | – | ✗ |
| TinyMPC-HOCBF (m=3m) | 33.80 | 0.077 | ✓ |
| TinyMPC-HOCBF (m=0m) | – | – | ✗ |

**说明**: TABLE II: Dynamic obstacle scenario results.
## 实验解读

- 评价重点:围绕 aerial-robotics、agile-locomotion、certified-control、碰撞避免、模型预测控制,优先看真实机器人结果、扰动恢复、接触稳定性、控制误差、成功率、实时性和 sim-to-real 差距。
- System0 视角:实验结论不只看最终成功率,还要看闭环过程中是否能处理状态估计误差、执行延迟、碰撞/接触突变和外部扰动。
- 与上层系统的关系:如果方法能在快闭环内稳定工作,它可以作为 System1 技能或 System2 规划的底层执行保障;如果只在离线或慢速设定下成立,则还不能直接作为身体级快闭环模块。
## 批判性思考

### 优点

- 论文把 aerial-robotics、agile-locomotion、certified-control、碰撞避免、模型预测控制 放到机器人身体级闭环里讨论,和只做高层语义推理的工作相比,更接近真实机器人部署时会遇到的状态、动力学和安全约束。
- 若论文同时给出公式、系统结构图、硬件实验或消融结果,这些证据可以帮助判断方法是否真的能支撑毫秒到百毫秒级反应。

### 局限

- 需要重点核查控制频率、推理延迟、传感延迟、失败恢复、硬件磨损、未见场景泛化和安全边界;这些因素决定它能否成为可靠的 System0 模块。
- 对学习型方法,还要区分训练环境中的鲁棒性和真实部署中的鲁棒性,尤其是接触丰富任务、动态运动任务和人形机器人任务。
## 论文定位

- 研究对象:TinySDP: Real Time Semidefinite Optimization for Certifiable and Agile Edge Robotics。
- 关键词:aerial-robotics、agile-locomotion、certified-control、碰撞避免、模型预测控制、navigation、实时控制、强化学习、safe-control。
- System0 关联:关注机器人身体级快反应链路中“状态/接触/局部感知 -> 约束/优化/策略 -> 动作”的短闭环,不把问题停留在语言规划或任务分解层。

## 速查卡片

> [!summary] TinySDP
> - **论文**: https://www.roboticsproceedings.org/rss22/p110.pdf
> - **arXiv**: http://arxiv.org/abs/2605.13748v1
> - **arXiv HTML**: https://arxiv.org/html/2605.13748v1
