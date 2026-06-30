---
title: "IMPACT: An Implicit Active-Set Augmented Lagrangian for Fast Contact-Implicit Trajectory Optimization"
method_name: "IMPACT"
authors: ["Jiayun Li"]
year: 2026
venue: "RSS"
tags: ["contact-reasoning", "robot-manipulation", "legged-locomotion", "safe-control", "contact-rich-manipulation", "robot-generalization", "dexterous-manipulation", "loco-manipulation", "model-predictive-control", "trajectory-optimization"]
image_source: "online"
arxiv_html: "https://arxiv.org/html/2605.09127v3"
---
# IMPACT
## 一句话总结

> IMPACT: An Implicit Active-Set Augmented Lagrangian for Fast Contact-Implicit Trajectory Optimization 主要落在 [[certified-control]]、[[contact-implicit-optimization]]、[[接触推理]]、[[接触丰富操作]] 这一类问题上;从 System0 视角看,它的价值在于把身体状态、接触变化、局部感知或动力学约束接到机器人低层闭环里,让动作生成更稳定、安全、实时。
## 核心贡献

1. 围绕 **IMPACT: An Implicit Active-Set Augmented Lagrangian for Fast Contact-Implicit Trajectory Optimization** 建立了一个与 certified-control、contact-implicit-optimization、接触推理、接触丰富操作、灵巧操作、足式运动 相关的机器人问题设定,重点不是高层语义规划,而是身体级闭环中的状态变化、接触变化或控制约束。
2. 方法层面的关键价值在于把感知、动力学、约束优化、策略学习或硬件反馈组织成可执行的低层动作生成机制,适合放进 System0 的快反应模块中理解。
3. 对系统集成而言,这类工作补上了 System1 技能选择与电机级控制之间的一层:它负责把上层目标变成动态可行、可恢复、可安全过滤的动作。
4. 若论文包含真实机器人、接触任务或仿真到真实迁移,核心应看它是否报告了闭环稳定性、扰动恢复、硬件延迟、失败模式和跨场景泛化。
## 问题背景

这篇论文可以放在 System0 的“具身快系统”背景下阅读:机器人需要在很短时间内根据本体状态、接触、力/触觉、局部视觉或动力学预测做出动作,而不是等待慢速规划器重新推理整个任务。certified-control、contact-implicit-optimization、接触推理、接触丰富操作、灵巧操作、足式运动、移动操作 相关问题的难点通常不在“知道要做什么”,而在动作执行过程中状态会快速偏离、接触会突然改变、约束会实时收紧,低层系统必须及时修正。

### System0 关联

精读时优先看三件事:第一,论文使用哪些快速反馈量,例如关节状态、接触状态、力觉、触觉、深度或局部视觉;第二,反馈如何进入 MPC、WBC、策略网络、安全过滤器或恢复控制器;第三,输出是否直接影响机器人下一步动作,而不是只服务离线分析或高层任务规划。
## 方法详解

### System0 视角解析

从 System0 角度看,这篇论文的方法部分应当围绕 certified-control、contact-implicit-optimization、接触推理、接触丰富操作、灵巧操作、足式运动、移动操作、模型预测控制 来读。核心问题是:系统如何从快速变化的机器人状态中构造控制输入,如何处理接触、碰撞、约束、模型误差或执行延迟,以及如何把这些信息变成下一时刻的动作。

如果方法里有优化问题,重点看目标函数、约束和求解频率;如果方法里有学习策略,重点看观测空间、动作空间、训练扰动和部署时的闭环频率;如果方法里有硬件或传感器,重点看传感链路是否真的进入控制回路。这样读可以把论文和 System2 语言规划、System1 技能编排区分开来。

### 关键公式与机制

#### 公式 1: [[优化目标/约束]]

$$\mathcal{C}:=\{(Y,Z):0\leq Y\perp Z\geq 0\}.$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 2: [[动力学或策略机制]]

$$N_{C_{i}}(y_{i},z_{i})=\begin{cases}\{0\}\times\mathbb{R},&y_{i}>0,\ z_{i}=0,\\[2.0pt] \mathbb{R}\times\{0\},&y_{i}=0,\ z_{i}>0,\\[2.0pt] \mathbb{R}_{-}^{2}\cup(\mathbb{R}\times\{0\})\cup(\{0\}\times\mathbb{R}),&y_{i}=0,\ z_{i}=0.\end{cases}$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 3: [[优化目标/约束]]

$$r_{\mathrm{nc}}(w)\doteq\operatorname{dist}_{\infty}\!\left(0,\,\nabla\Phi(w)+\{0\}\times N_{\mathcal{C}}(Y,Z)\right),\qquad w=(X,Y,Z),\quad(Y,Z)\in\mathcal{C}.$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 4: [[优化目标/约束]]

$$r_{\mathrm{in}}(w;u,v)\doteq\max\Bigl\{\lVert G^{X}(w)\rVert_{\infty},\,\lVert \nabla_{Y}\Phi(w)-u \rVert_{\infty},\,\lVert \nabla_{Z}\Phi(w)-v \rVert_{\infty},\,r_{\mathrm{pri}}(w)\Bigr\}.$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 5: [[优化目标/约束]]

$$\operatorname{dist}\!\left(0,\,\nabla\Phi^{k}(w^{k})+\{0\}\times N_{\mathcal{C}}(Y^{k},Z^{k})\right)\leq O(\varepsilon_{k}),$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 6: [[动力学或策略机制]]

$$c_{X}\sum_{j=0}^{N}\bigl\lVert G^{X}(w^{(j)})\bigr \rVert_{\infty}^{2}\leq\Phi(w^{(0)})-\Phi(w^{(N+1)}).$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 7: [[优化目标/约束]]

$$\min_{0\leq j\leq N}\bigl\lVert G^{X}(w^{(j)})\bigr \rVert_{\infty}\leq\sqrt{\frac{\Phi(w^{(0)})-\inf_{w:\,(Y,Z)\in\mathcal{C}}\Phi(w)}{c_{X}(N+1)}}.$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 8: [[优化目标/约束]]

$$d_{i}(w)\doteq\begin{cases}|a_{i}(w)|,&i\in I_{+0}(w),\\[2.0pt] |b_{i}(w)| ,&i\in I_{0+}(w),\\[2.0pt] \min $\Bigl\{\max\{(-a_{i}(w))_{+},(-b_{i}(w))_{+}\},\,$ |a_{i}(w)|,\,|b_{i}(w)|\Bigr\},&i\in I_{00}(w).\end{cases}$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 9: [[优化目标/约束]]

$$=\inf_{(u,v)\in\mathcal{M}_{M}(w)}\max\Bigl\{\lVert \nabla_{X}\Phi(w)\rVert_{\infty},\,\lVert \nabla_{Y}\Phi(w)-u \rVert_{\infty},\,\lVert \nabla_{Z}\Phi(w)-v \rVert_{\infty}\Bigr\}.$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 10: [[动力学或策略机制]]

$$\frac{1}{2}\sum_{t=1}^{T}r_{t}^{\top}r_{t}+\frac{\rho_{\bar{h}}}{2}\Bigl\lVert \bar{h}_{t}+\tfrac{\kappa_{t}}{\rho_{\bar{h}}}\Bigr \rVert^{2}+\frac{\rho_{g}}{2}\Bigl\lVert(g_{t}+\tfrac{\mu_{t}}{\rho_{g}})_{+}\Bigr \rVert^{2}$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。
## 关键图表

### Figure 1: Pipeline / core system figure: IMPACT demonstrations in simulation and hardware. Top: Allegro Hand reorients a rubbe

![Figure 1](https://arxiv.org/html/2605.09127v3/x1.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Pipeline / core system figure: IMPACT demonstrations in simulation and hardware. Top: Allegro Hand reorients a rubbe”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 2: IMPACT planning demos on three CITO tasks. The green dashed box marks the start pose

![Figure 2](https://arxiv.org/html/2605.09127v3/x3.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“IMPACT planning demos on three CITO tasks. The green dashed box marks the start pose”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 3: Allegro-Hand CI-MPC benchmark results on 17 objects. We compare IMPACT against cfree

![Figure 3](https://arxiv.org/html/2605.09127v3/x4.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Allegro-Hand CI-MPC benchmark results on 17 objects. We compare IMPACT against cfree”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### 论文表格

#### Table 1: TABLE I: CITO benchmark results reported as mean ± $\pm$ 95% confidence interval (CI). For each task and metric, the best

| Task | Solver | Success (%) | Track. Err. ↓ \downarrow | Iters ↓ \downarrow | Time (s) ↓ \downarrow |
| --- | --- | --- | --- | --- | --- |
| Box | SR | 100.0 100.0 | 19.46 ± 2.30  $\boldsymbol{19.46\pm 2.30}$ | 164 ± 12  $\boldsymbol{164\pm 12}$ | 1.27 ± 0.10 1.27\pm 0.10 |
| PM | 98.0 98.0 | 58.00 ± 5.11 58.00\pm 5.11 | 66 ± 13  $\boldsymbol{66\pm 13}$ | 0.85 ± 0.17  $\boldsymbol{0.85\pm 0.17}$ | |
| CRISP | 100.0 100.0 | 51.90 ± 4.08 51.90\pm 4.08 | 641 ± 358 641\pm 358 | 2.52 ± 1.34 2.52\pm 1.34 | |
| IMPACT | 100.0 100.0 | 26.99 ± 2.81  $\boldsymbol{26.99\pm 2.81}$ | 219 ± 49 219\pm 49 | 0.15 ± 0.03  $\boldsymbol{0.15\pm 0.03}$ | |
| T | SR | 100.0 100.0 | 1.24 ± 0.13  $\boldsymbol{1.24\pm 0.13}$ | 202 ± 11  $\boldsymbol{202\pm 11}$ | 5.50 ± 0.38 5.50\pm 0.38 |
| PM | 90.0 90.0 | 21.99 ± 2.57 21.99\pm 2.57 | 117 ± 14  $\boldsymbol{117\pm 14}$ | 2.96 ± 0.39  $\boldsymbol{2.96\pm 0.39}$ | |
| CRISP | 100.0 100.0 | 8.03 ± 0.71  $\boldsymbol{8.03\pm 0.71}$ | 1243 ± 407 1243\pm 407 | 25.73 ± 6.92 25.73\pm 6.92 | |
| IMPACT | 100.0 100.0 | 8.48 ± 1.94 8.48\pm 1.94 | 606 ± 225 606\pm 225 | 1.03 ± 0.44  $\boldsymbol{1.03\pm 0.44}$ | |
| Cart | SR | 100.0 100.0 | 433.08 ± 44.92 433.08\pm 44.92 | 261 ± 22 261\pm 22 | 5.63 ± 0.57 5.63\pm 0.57 |
| PM | 100.0 100.0 | 408.65 ± 42.32 408.65\pm 42.32 | 87 ± 14  $\boldsymbol{87\pm 14}$ | 2.00 ± 0.32  $\boldsymbol{2.00\pm 0.32}$ | |
| CRISP | 100.0 100.0 | 235.10 ± 21.70  $\boldsymbol{235.10\pm 21.70}$ | 501 ± 381 501\pm 381 | 2.72 ± 2.05 2.72\pm 2.05 | |
| IMPACT | 100.0 100.0 | 381.89 ± 32.35  $\boldsymbol{381.89\pm 32.35}$ | 101 ± 15  $\boldsymbol{101\pm 15}$ | 0.08 ± 0.01  $\boldsymbol{0.08\pm 0.01}$ | |

**说明**: TABLE I: CITO benchmark results reported as mean ± $\pm$ 95% confidence interval (CI). For each task and metric, the best method is highlighted in red and the second best in blue.

#### Table 2: TABLE II: Hyperparameters for the three benchmark tasks.

| Parameter | Push Box | Push T | Cart Transporter |
| --- | --- | --- | --- |
| Physical Parameters | | | |
| Mass m m (kg) | 0.1 | 0.1 | m 1 = 0.1 m_{1}=0.1, m 2 = 0.2 m_{2}=0.2 |
| Gravity g g (m/s 2) | 9.81 | 9.8 | 9.81 |
| Friction coefficient $\mu$ | 0.5 | 0.4 | 0.2 |
| Characteristic length (m) | a = 0.3 a=0.3, b = 0.4 b=0.4 | l = 0.05 l=0.05 | l = 1.0 l=1.0 |
| Time step   t \Delta t (s) | 0.05 | 0.05 | 0.02 |
| Problem Dimensions | | | |
| State dimension n x n_{x} | 3 | 3 | 4 |
| Control dimension n u n_{u} | 6 | 24 | 4 |
| Complementarity pairs n c n_{c} | 10 | 43 | 3 |
| Equality constraints n e n_{e} | 0 | 7 | 1 |
| Inequality constraints n i n_{i} | 0 | 4 | 4 |
| Optimization Parameters | | | |
| Horizon | 50 | 50 | 300 |
| Stage control cost weight | 0.001 | 0.01 | 10 - 6 10^{-6} |
| Final cost weight | 100.0 | 100.0 | 5000.0 |
| Multiplier Safeguard | 10 6 10^{6} | 10 6 10^{6} | 10 6 10^{6} |
|  \rho scaling factor | 1.1 | 1.1 | 1.5 |
| Convergence Criteria | | | |
| Max outer iterations | 1000 | 1000 | 1000 |
| Outer tolerance  h \epsilon_{h} | 10 - 5 10^{-5} | 10 - 5 10^{-5} | 10 - 5 10^{-5} |
| Outer tolerance comp \epsilon $\text{comp}}$ | 10 - 5 10^{-5} | 10 - 5 10^{-5} | 10 - 5 10^{-5} |
| Max inner iterations | 50 | 50 | 10 |
| Inner tolerance | 10 - 3 10^{-3} | 10 - 3 10^{-3} | 10 - 3 10^{-3} |
| Newton Solver Parameters | | | |
| Max Newton iterations | 50 | 200 | 100 |
| Newton tolerance | 10 - 6 10^{-6} | 10 - 6 10^{-6} | 10 - 6 10^{-6} |
| Regularization | 2  10 - 5 2\times 10^{-5} | 5  10 - 5 5\times 10^{-5} | 10 - 5 10^{-5} |

**说明**: TABLE II: Hyperparameters for the three benchmark tasks.

#### Table 3: TABLE III: Hyperparameters comparison for Allegro Hand in-hand manipulation: IMPACT vs. C-Free

| Parameter | IMPACT | C-Free | Notes |
| --- | --- | --- | --- |
| System Dimensions | | | |
| State dimension n q n_{q} | 23 | 23 | Same (obj: 7, robot: 16) |
| Velocity dimension n v n_{v} | 22 | 22 | Same (obj: 6, robot: 16) |
| Command dimension n u n_{u} | 16 | 16 | Same (joint position cmds) |
| Max contacts n con n_{ $\text{con}}$ | 20 | 20 | Same |
| Simulation Parameters (Mujoco) | | | |
| Time step h h (s) | 0.1 | 0.1 | Same |
| Frame skip | 50 | 50 | Same |
| Friction coefficient $\mu$ | 0.5 | 0.5 | Same |
| MPC Parameters | | | |
| Horizon N N | 4 | 4 | Same |
| Control bound | ± 0.1 \pm 0.1 | ± 0.1  (0.2) \pm 0.1(0.2) | |
| Cost Function Weights (Path Cost) | | | |
| Position cost weight | 0.0 | 0.0 | Same |
| Quaternion cost weight | 0.0 | 0.0 | Same |
| Contact cost weight | 1.0 | 1.0 | Encourage contact |
| Grasp closure weight | 0.0 | 0.0 | Same |
| Control cost weight | 0.1 | 0.1 | Same |
| Velocity penalty | 0.1 | – | Object Damping |
| Cost Function Weights (Final Cost) | | | |
| Final position weight | 1000.0 | 1000.0 | Same |
| Final quaternion weight | 90.0 | 50.0 | scale match |
| Solver-Specific Parameters | | | |
| IMPACT Parameters | | | |
|  max \rho_{\max} | 10 3 10^{3} | – | |
|  \rho scale factor | 5.0 | – | |
| Max outer iterations | 10 | – | |
| Outer tol ( h \epsilon_{h}) | 10 - 3 10^{-3} | – | |
| Outer tol comp \epsilon $\text{comp}})$ | 10 - 3 10^{-3} | – | |
| Max inner iterations | 5 | – | |
| Newton max iterations | 30 | – | |
| Newton step tolerance | 10 - 5 10^{-5} | – | |
| Newton obj tolerance | 10 - 6 10^{-6} | – | |
| C-Free (IPOPT-based NLP) | | | |
| IPOPT max iterations | – | 50 | |
| Complementarity relaxation | – | > 0 >0 | Relaxed LCP |
| Success Criteria | | | |
| Position tolerance (m) | 0.02 | 0.02 | Same |
| Quaternion tolerance | 0.04 | 0.04 | Same |
| Consecutive success steps | 20 | 20 | Same |
| Max rollout steps | 1000 | 1000 | Same |

**说明**: TABLE III: Hyperparameters comparison for Allegro Hand in-hand manipulation: IMPACT vs. C-Free
## 实验解读

- 评价重点:围绕 certified-control、contact-implicit-optimization、接触推理、接触丰富操作、灵巧操作,优先看真实机器人结果、扰动恢复、接触稳定性、控制误差、成功率、实时性和 sim-to-real 差距。
- System0 视角:实验结论不只看最终成功率,还要看闭环过程中是否能处理状态估计误差、执行延迟、碰撞/接触突变和外部扰动。
- 与上层系统的关系:如果方法能在快闭环内稳定工作,它可以作为 System1 技能或 System2 规划的底层执行保障;如果只在离线或慢速设定下成立,则还不能直接作为身体级快闭环模块。
## 批判性思考

### 优点

- 论文把 certified-control、contact-implicit-optimization、接触推理、接触丰富操作、灵巧操作 放到机器人身体级闭环里讨论,和只做高层语义推理的工作相比,更接近真实机器人部署时会遇到的状态、动力学和安全约束。
- 若论文同时给出公式、系统结构图、硬件实验或消融结果,这些证据可以帮助判断方法是否真的能支撑毫秒到百毫秒级反应。

### 局限

- 需要重点核查控制频率、推理延迟、传感延迟、失败恢复、硬件磨损、未见场景泛化和安全边界;这些因素决定它能否成为可靠的 System0 模块。
- 对学习型方法,还要区分训练环境中的鲁棒性和真实部署中的鲁棒性,尤其是接触丰富任务、动态运动任务和人形机器人任务。
## 论文定位

- 研究对象:IMPACT: An Implicit Active-Set Augmented Lagrangian for Fast Contact-Implicit Trajectory Optimization。
- 关键词:certified-control、contact-implicit-optimization、接触推理、接触丰富操作、灵巧操作、足式运动、移动操作、模型预测控制、non-prehensile-manipulation、机器人操作。
- System0 关联:关注机器人身体级快反应链路中“状态/接触/局部感知 -> 约束/优化/策略 -> 动作”的短闭环,不把问题停留在语言规划或任务分解层。

## 速查卡片

> [!summary] IMPACT
> - **论文**: https://www.roboticsproceedings.org/rss22/p163.pdf
> - **arXiv**: http://arxiv.org/abs/2605.09127v3
> - **arXiv HTML**: https://arxiv.org/html/2605.09127v3
