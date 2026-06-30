---
title: "HUSKY: Humanoid Skateboarding System via Physics-Aware Whole-Body Control"
method_name: "HUSKY"
authors: ["Jinrui Han"]
year: 2026
venue: "RSS"
tags: ["contact-reasoning", "robust-control", "legged-locomotion", "imitation-learning", "contact-rich-manipulation", "humanoid", "agile-locomotion", "whole-body-control"]
image_source: "online"
arxiv_html: "https://arxiv.org/html/2602.03205v2"
---
# HUSKY
## 一句话总结

> HUSKY: Humanoid Skateboarding System via Physics-Aware Whole-Body Control 主要落在 [[agile-locomotion]]、[[接触推理]]、[[人形机器人]]、[[足式运动]] 这一类问题上;从 System0 视角看,它的价值在于把身体状态、接触变化、局部感知或动力学约束接到机器人低层闭环里,让动作生成更稳定、安全、实时。
## 核心贡献

1. 围绕 **HUSKY: Humanoid Skateboarding System via Physics-Aware Whole-Body Control** 建立了一个与 agile-locomotion、接触推理、人形机器人、足式运动、运动模仿、non-prehensile-manipulation 相关的机器人问题设定,重点不是高层语义规划,而是身体级闭环中的状态变化、接触变化或控制约束。
2. 方法层面的关键价值在于把感知、动力学、约束优化、策略学习或硬件反馈组织成可执行的低层动作生成机制,适合放进 System0 的快反应模块中理解。
3. 对系统集成而言,这类工作补上了 System1 技能选择与电机级控制之间的一层:它负责把上层目标变成动态可行、可恢复、可安全过滤的动作。
4. 若论文包含真实机器人、接触任务或仿真到真实迁移,核心应看它是否报告了闭环稳定性、扰动恢复、硬件延迟、失败模式和跨场景泛化。
## 问题背景

这篇论文可以放在 System0 的“具身快系统”背景下阅读:机器人需要在很短时间内根据本体状态、接触、力/触觉、局部视觉或动力学预测做出动作,而不是等待慢速规划器重新推理整个任务。agile-locomotion、接触推理、人形机器人、足式运动、运动模仿、non-prehensile-manipulation、鲁棒控制 相关问题的难点通常不在“知道要做什么”,而在动作执行过程中状态会快速偏离、接触会突然改变、约束会实时收紧,低层系统必须及时修正。

### System0 关联

精读时优先看三件事:第一,论文使用哪些快速反馈量,例如关节状态、接触状态、力觉、触觉、深度或局部视觉;第二,反馈如何进入 MPC、WBC、策略网络、安全过滤器或恢复控制器;第三,输出是否直接影响机器人下一步动作,而不是只服务离线分析或高层任务规划。
## 方法详解

### System0 视角解析

从 System0 角度看,这篇论文的方法部分应当围绕 agile-locomotion、接触推理、人形机器人、足式运动、运动模仿、non-prehensile-manipulation、鲁棒控制、全身控制 来读。核心问题是:系统如何从快速变化的机器人状态中构造控制输入,如何处理接触、碰撞、约束、模型误差或执行延迟,以及如何把这些信息变成下一时刻的动作。

如果方法里有优化问题,重点看目标函数、约束和求解频率;如果方法里有学习策略,重点看观测空间、动作空间、训练扰动和部署时的闭环频率;如果方法里有硬件或传感器,重点看传感链路是否真的进入控制回路。这样读可以把论文和 System2 语言规划、System1 技能编排区分开来。

### 关键公式与机制

#### 公式 1: [[动力学或策略机制]]

$$\boldsymbol{o}^{\text{prop}}_{t}=[\boldsymbol{c}_{t},\boldsymbol{\omega}_{t},\boldsymbol{g}_{t},\boldsymbol{\theta}_{t},\dot{\boldsymbol{\theta}}_{t},\boldsymbol{a}_{t-1},\Phi],$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 2: [[动力学或策略机制]]

$$\boldsymbol{o}^{\text{priv}}_{t}=[\boldsymbol{v}_{t},\boldsymbol{p}^{b}_{t},\boldsymbol{r}^{b}_{t},\boldsymbol{v}^{b}_{t},\boldsymbol{\omega}^{b}_{t},\boldsymbol{\theta}^{{b}}_{t},\boldsymbol{f}^{g}_{t},\boldsymbol{f}^{b}_{t}],$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 3: [[动力学或策略机制]]

$$r_{t}=\mathbb{I}^{\text{push}}\cdot r_{t}^{\text{push}}+\mathbb{I}^{\text{steer}}\cdot r_{t}^{\text{steer}}+\mathbb{I}^{\text{trans}}\cdot r_{t}^{\text{trans}}+r_{t}^{\text{reg}},$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 4: [[优化目标/约束]]

$$r^{\text{style}}(s_{t})=\alpha\cdot\max\left(0,\ 1-\frac{1}{4}(d-1)^{2}\right),$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 5: [[动力学或策略机制]]

$$\gamma_{\text{ref}}=\arcsin\!\left(\frac{L\,\Delta\psi}{v\,\Delta t\,\tan\lambda}\right),$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 6: [[优化目标/约束]]

$$\boldsymbol{p}^{\mathcal{K}}(t)=\sum_{i=0}^{n}\binom{n}{i}(1-s)^{\,n-i}s^{i}\,\boldsymbol{p}^{\mathcal{K}}_{i},\quad s=\frac{t-t_{0}}{t_{f}-t_{0}},$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 7: [[优化目标/约束]]

$$\boldsymbol{q}^{\mathcal{K}}(t)=\frac{\sin((1-s)\Omega)}{\sin\Omega}\,\boldsymbol{q}^{\mathcal{K}}_{\text{end}}+\frac{\sin(s\,\Omega)}{\sin\Omega}\,\boldsymbol{q}^{\mathcal{K}}_{\text{ref}},$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 8: [[动力学或策略机制]]

$$E_{\text{vel}}=\mathbb{E}\left[\left| v_{ $\text{cmd}}-v_{\text{board}}\right$ |\right]$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 9: [[优化目标/约束]]

$$\mathbb{I}(T_{\text{air}}^{\mathrm{left}\_foot}\in[T_{\text{air}}^{\min},T_{\text{air}}^{\max}])\cdot\mathbb{I}(v_{\mathrm{cmd}}>v_{\mathrm{th}})$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 10: [[优化目标/约束]]

$$+\frac{\alpha^{d}}{2}\mathbb{E}_{\tau\sim\mathcal{M}}[\lVert \nabla_{\phi}D_{\phi}(\tau)\rVert_{2}].$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。
## 关键图表

### Figure 1: Pipeline / core system figure: Skateboard Model. We analyze the skateboard kinematic structure and derive the couplin

![Figure 1](https://arxiv.org/html/2602.03205v2/x2.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Pipeline / core system figure: Skateboard Model. We analyze the skateboard kinematic structure and derive the couplin”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 2: Framework of HUSKY. (a) We first analyze and model the humanoid–skateboard system, de

![Figure 2](https://arxiv.org/html/2602.03205v2/figures/method.jpg)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Framework of HUSKY. (a) We first analyze and model the humanoid–skateboard system, de”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 3: Humanoid Skateboarding on Diverse Boards. HUSKY enables the humanoid to perform skate

![Figure 3](https://arxiv.org/html/2602.03205v2/x7.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Humanoid Skateboarding on Diverse Boards. HUSKY enables the humanoid to perform skate”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### 论文表格

#### Table 1: TABLE I: Comparison of Humanoid and Quadrupedal Skateboarding Tasks.

| Term | Humanoid (G1) | Quadruped (Go1) [19 ] |
| --- | --- | --- |
| Degrees of Freedom | 23 | 12 |
| Contact Points | 1–2 | 2–4 |
| CoM Height | 0.6–0.8 m m | 0.2–0.3 m m |
| Support Polygon | 0.05–0.10 m 2 m^{2} | 0.10–0.15 m 2 m^{2} |
| Side-on Steering | ✓ \checkmark |  \times |
| Leg Reorientation | ✓ \checkmark |  \times |

**说明**: TABLE I: Comparison of Humanoid and Quadrupedal Skateboarding Tasks.

#### Table 2: TABLE II: Domain Randomization Parameters.

| DR terms | Range | Unit |
| --- | --- | --- |
| Robot Center of Mass | U  (- 2.5, 2.5)  $\mathcal{U}(-2.5,\ 2.5)$ | c  m cm |
| Skateboard Center of Mass | U  (- 2.5, 2.5)  $\mathcal{U}(-2.5,\ 2.5)$ | c  m cm |
| Default Root Position | U  (- 2.0, 2.0)  $\mathcal{U}(-2.0,\ 2.0)$ | c  m cm |
| Default Joint Position | U  (- 0.01, 0.01)  $\mathcal{U}(-0.01,\ 0.01)$ | r  a  d rad |
| Push Robot Base | U  (- 0.5, 0.5)  $\mathcal{U}(-0.5,\ 0.5)$ | m / s m/s |
| Robot Body Friction | U  (0.3, 1.6)  $\mathcal{U}(0.3,\ 1.6)$ | - |
| Skateboard Deck Friction | U  (0.8, 2.0)  $\mathcal{U}(0.8,\ 2.0)$ | - |

**说明**: TABLE II: Domain Randomization Parameters.

#### Table 3: TABLE III: Simulation Results.

| Method | E succ ↑ E_{ $\text{succ}}\!\uparrow$ | E vel ↓ E_{ $\text{vel}}\!\downarrow$ | E yaw ↓ E_{ $\text{yaw}}\!\downarrow$ | E smth ↓ E_{ $\text{smth}}\!\downarrow$ | E contact ↓ E_{ $\text{contact}}\!\downarrow$ | |
| --- | --- | --- | --- | --- | --- | --- |
| Ablation on Pushing Style | | | | | | |
| HUSKY -Tracking-Based | 11.12 ± 3.86 11.12\!\pm\!{3.86} | 0.435 ± 0.101 0.435\!\pm\!{0.101} | 0.568 ± 0.092 {0.568}\!\pm\!{0.092} | 0.044 ± 0.025 {0.044}\!\pm\!{0.025} | 0.015 ± 0.010 {0.015}\!\pm\!{0.010} | |
| HUSKY -Gait-Based | 82.38 ± 7.25 82.38\!\pm\!{7.25} | 0.102 ± 0.035 0.102\!\pm\!{0.035} | 0.302 ± 0.041 {0.302}\!\pm\!{0.041} | 0.043 ± 0.011 {0.043}\!\pm\!{0.011} | 0.130 ± 0.072 {0.130}\!\pm\!{0.072} | |
| HUSKY (ours) | 100.00 ± 0.00  $\textbf{{100.00}}\!\pm\!{0.00}$ | 0.056 ± 0.013  $\textbf{{0.056}}\!\pm\!{0.013}$ | 0.208 ± 0.014  $\textbf{{0.208}}\!\pm\!{0.014}$ | 0.033 ± 0.005  $\textbf{0.033}\!\pm\!{0.005}$ | 0.001 ± 0.001  $\textbf{0.001}\!\pm\!{0.001}$ | |
| Ablation on Steering Strategy | | | | | | |
| HUSKY -w/o-Tilt Guidance | 96.72 ± 2.10 96.72\!\pm\!{2.10} | 0.071 ± 0.010 0.071\!\pm\!{0.010} | 0.233 ± 0.027 {0.233}\!\pm\!{0.027} | 0.035 ± 0.017 0.035\!\pm\!{0.017} | 0.002 ± 0.002 0.002\!\pm\!{0.002} | |
| HUSKY (ours) | 100.00 ± 0.00  $\textbf{{100.00}}\!\pm\!{0.00}$ | 0.056 ± 0.013  $\textbf{{0.056}}\!\pm\!{0.013}$ | 0.208 ± 0.014  $\textbf{{0.208}}\!\pm\!{0.014}$ | 0.033 ± 0.005  $\textbf{0.033}\!\pm\!{0.005}$ | 0.001 ± 0.001  $\textbf{0.001}\!\pm\!{0.001}$ | |
| Ablation on Transition Mechanism | | | | | | |
| HUSKY -AMP Transition | 85.12 ± 4.11 85.12\!\pm\!{4.11} | 0.053 ± 0.025  $\textbf{0.053}\!\pm\!{0.025}$ | 0.265 ± 0.050 {0.265}\!\pm\!{0.050} | 0.040 ± 0.007 0.040\!\pm\!{0.007} | 0.394 ± 0.015 0.394\!\pm\!{0.015} | |
| HUSKY -Translation-only | 89.55 ± 2.30 {89.55}\!\pm\!{2.30} | 0.064 ± 0.020 0.064\!\pm\!{0.020} | 0.294 ± 0.075 {0.294}\!\pm\!{0.075} | 0.039 ± 0.012 0.039\!\pm\!{0.012} | 0.038 ± 0.012 0.038\!\pm\!{0.012} | |
| HUSKY -Mixed Initialization | 86.03 ± 1.70 {86.03}\!\pm\!{1.70} | 0.067 ± 0.024 0.067\!\pm\!{0.024} | 0.278 ± 0.092 {0.278}\!\pm\!{0.092} | 0.037 ± 0.015 0.037\!\pm\!{0.015} | 0.371 ± 0.025 0.371\!\pm\!{0.025} | |
| HUSKY (ours) | 100.00 ± 0.00  $\textbf{{100.00}}\!\pm\!{0.00}$ | 0.056 ± 0.013 0.056\!\pm\!{0.013} | 0.208 ± 0.014  $\textbf{{0.208}}\!\pm\!{0.014}$ | 0.033 ± 0.005  $\textbf{0.033}\!\pm\!{0.005}$ | 0.001 ± 0.001  $\textbf{0.001}\!\pm\!{0.001}$ | |

**说明**: TABLE III: Simulation Results.

#### Table 4: TABLE IV: Skateboard Bodies

| Name | Position | Type | Size |
| --- | --- | --- | --- |
| Skateboard deck | 0 0 0 | Box | 0.8 0.2 0.02 |
| Front truck | 0 0 -0.09 | Box | 0.1 0.02 0.02 |
| Front left wheel | 0 0.07 0 | Cylinder | 0.03 0.02 |
| Front right wheel | 0 -0.07 0 | Cylinder | 0.03 0.02 |
| Rear truck | 0 0 -0.09 | Box | 0.1 0.02 0.02 |
| Rear left wheel | 0 0.07 0 | Cylinder | 0.03 0.02 |
| Rear right wheel | 0 -0.07 0 | Cylinder | 0.03 0.02 |

**说明**: TABLE IV: Skateboard Bodies

#### Table 5: TABLE V: Skateboard Joints

| Name | Type | Joint Axis | Range |
| --- | --- | --- | --- |
| Board tilt joint | Hinge | -1 0 0 | (-0.2, 0.2) |
| Front truck joint | Hinge | 0 0 1 | (-0.1, 0.1) |
| Rear truck joint | Hinge | 0 0 -1 | (-0.1, 0.1) |
| Wheel joints | Hinge | 0 1 0 | Continuous |

**说明**: TABLE V: Skateboard Joints

#### Table 6: TABLE VI: Task Reward terms and weights.

| Term | Expression | Weight | Meaning |
| --- | --- | --- | --- |
| Pushing Phase r t push r_{t}^{ $\text{push}}$ | | | |
| Linear velocity tracking | $\exp(-\lVert v_{\mathrm{board}}-v_{\mathrm{cmd}} \rVert^{2}/\sigma_{v}^{2})$ | 3.0 | Track commanded forward velocity |
| Yaw alignment | $\exp(-\lvert\psi_{\mathrm{robot}}-\psi_{\mathrm{board}}\rvert^{2}/\sigma_{\mathrm{yaw}}^{2})$ | 1.0 | Align robot yaw with skateboard during pushing |
| Feet air time | $\mathbb{I}(T_{\text{air}}^{\mathrm{left}\_foot}\in[T_{\text{air}}^{\min},T_{\text{air}}^{\max}])\cdot\mathbb{I}(v_{\mathrm{cmd}}>v_{\mathrm{th}})$ | 3.0 | Encourage proper left-foot lift timing during pushing |
| Ankle parallel [14 ] | $\mathbb{I}(\mathrm{Var}(z_{\mathrm{left}\_ankle})<0.05\)\cdot\mathbb{I}(\text{left foot on ground}))$ | 0.5 | Encourage left foot to remain parallel during pushing |
| AMP style reward | $\left(0,\ 1-{1}/{4}(d-1)^{2}\right)$ | 5.0 | Encourage human-like natural pushing behavior |
| Steering Phase r t steer r_{t}^{ $\text{steer}}$ | | | |
| Steer feet contact | 2 ∗ I  (both feet on board) - I  (left foot on ground) 2*\, $\mathbb{I}(\text{both feet on board})-\mathbb{I}(\text{left foot on ground})$ | 3.0 | Encourage both feet on board and avoid ground contact |
| Joint position deviation | $\exp(-\lVert \boldsymbol{\theta}_{t}-\hat{\boldsymbol{\theta}}_{t} \rVert_{2}^{2}/\sigma_{\mathrm{jpos}}^{2})$ | 1.5 | Maintain nominal humanoid steering pose |
| Heading tracking | $\exp(-(\psi_{\mathrm{board}}-\psi)^{2}/\sigma_{\psi}^{2})$ | 5.0 | Track desired board direction |
| Board tilt tracking | $\exp(-(\gamma-\gamma_{\mathrm{ref}})^{2}/\sigma_{\gamma}^{2})$ | 4.0 | Align board lean with physics-guided reference |
| Feet marker distance | $\exp(-\lVert \boldsymbol{p}_{\mathrm{foot}}-\boldsymbol{p}_{\mathrm{marker}} \rVert_{2}^{2}/\sigma_{m}^{2})$ | 1.0 | Encourage feet near preferred foot markers |
| Transition r t trans r_{t}^{ $\text{trans}}$ | | | |
| Keybody position tracking | $\exp(-\lVert \boldsymbol{p}^{\mathcal{K}}_{t}-\hat{\boldsymbol{p}}^{\mathcal{K}}_{t} \rVert_{2}^{2}/\sigma_{\mathrm{pos}}^{2})$ | 10.0 | Follow trajectory-planned keybody positions |
| Keybody orientation tracking | $\exp(-\lVert \boldsymbol{q}^{\mathcal{K}}_{t}\ominus\boldsymbol{q}^{\mathcal{K}}_{t} \rVert_{2}^{2}/\sigma_{\mathrm{rot}}^{2})$ | 10.0 | Follow trajectory-planned keybody orientations |
| Regularization r t reg r_{t}^{ $\text{reg}}$ | | | |
| Skateboard wheel contact | I  (∑ i = 1 4 c i = 4)  $\mathbb{I}\big(\sum_{i=1}^{4}c_{i}=4\big)$ | 0.5 | Reward full wheel contact, avoid unrealistic detachment |
| Joint position limits | I  (q t ∉ [q min, q max ])  $\mathbb{I}(\boldsymbol{q}_{t}\notin[\boldsymbol{q}_{\mathrm{min}},\boldsymbol{q}_{\mathrm{max}}])$ | -5.0 | Keep joints within safe limits |
| Joint velocity | $\lVert \dot{\boldsymbol{q}}_{t} \rVert_{2}^{2}$ | -1e-3 | Penalize high joint speeds |
| Joint acceleration | $\lVert \ddot{\boldsymbol{q}}_{t} \rVert_{2}^{2}$ | -2.5e-7 | Penalize abrupt joint accelerations |
| Joint torque | $\lVert \boldsymbol{\tau}_{t} \rVert_{2}^{2}$ | -1e-6 | Penalize excessive torque |
| Action rate | $\lVert \boldsymbol{a}_{t}-\boldsymbol{a}_{t-1} \rVert_{2}^{2}$ | -0.1 | Encourage smooth actions |
| Action smoothness | $\lVert \boldsymbol{a}_{t}-2\boldsymbol{a}_{t-1}+\boldsymbol{a}_{t-2} \rVert_{2}^{2}$ | -0.1 | Reduce oscillations in control commands |
| Collision | I collision  $\mathbb{I}_{\mathrm{collision}}$ | -10.0 | Penalize self-collisions |

**说明**: TABLE VI: Task Reward terms and weights.

#### Table 7: TABLE VII: Hyperparameters related to PPO.

| Hyperparameter | Value |
| --- | --- |
| Optimizer | Adam |
| Batch size | 4096 |
| Mini Batches | 4 |
| Learning epoches | 5 |
| Entropy coefficient | 0.005 |
| Value loss coefficient | 1.0 |
| Clip param | 0.2 |
| Max grad norm | 1.0 |
| Init noise std | 1.0 |
| Learning rate | 1e-3 |
| Desired KL | 0.01 |
| GAE decay factor( \lambda) | 0.95 |
| GAE discount factor $\gamma)$ | 0.99 |
| Actor MLP size | [512, 256, 128] |
| Critic MLP size | [512, 256, 128] |
| MLP Activation | ELU |

**说明**: TABLE VII: Hyperparameters related to PPO.

#### Table 8: TABLE VIII: Evaluations on More Terrains

| Terrain Settings | E succ ↑ E_{ $\text{succ}}\!\uparrow$ | E smth ↓ E_{ $\text{smth}}\!\downarrow$ | E contact ↓ E_{ $\text{contact}}\!\downarrow$ |
| --- | --- | --- | --- |
| HUSKY-Slope | 95.02 ± 0.14 {{95.02}}\!\pm\!{0.14} | 0.038 ± 0.006 {0.038}\!\pm\!{0.006} | 0.001 ± 0.001 {0.001}\!\pm\!{0.001} |
| HUSKY-Steps | 89.21 ± 0.25 {{89.21}}\!\pm\!{0.25} | 0.042 ± 0.009 {0.042}\!\pm\!{0.009} | 0.003 ± 0.002 {0.003}\!\pm\!{0.002} |
| HUSKY-Flat Ground | 100.00 ± 0.00 {{100.00}}\!\pm\!{0.00} | 0.033 ± 0.005 {0.033}\!\pm\!{0.005} | 0.001 ± 0.001 {0.001}\!\pm\!{0.001} |

**说明**: TABLE VIII: Evaluations on More Terrains

#### Table 9: TABLE IX: Sensitivity Analysis of Transition

| Transition Settings | E succ ↑ E_{ $\text{succ}}\!\uparrow$ | E smth ↓ E_{ $\text{smth}}\!\downarrow$ | E contact ↓ E_{ $\text{contact}}\!\downarrow$ |
| --- | --- | --- | --- |
| Pose-Perturbed | 99.63 ± 0.12 {{99.63}}\!\pm\!{0.12} | 0.037 ± 0.007 {0.037}\!\pm\!{0.007} | 0.002 ± 0.001 {0.002}\!\pm\!{0.001} |
| Time-Scaled | 100.00 ± 0.00 {{100.00}}\!\pm\!{0.00} | 0.035 ± 0.003 {0.035}\!\pm\!{0.003} | 0.001 ± 0.002 {0.001}\!\pm\!{0.002} |
| Path-Varied | 98.54 ± 0.93 {{98.54}}\!\pm\!{0.93} | 0.033 ± 0.004 {0.033}\!\pm\!{0.004} | 0.002 ± 0.002 {0.002}\!\pm\!{0.002} |
| HUSKY | 100.00 ± 0.00 {{100.00}}\!\pm\!{0.00} | 0.033 ± 0.005 {0.033}\!\pm\!{0.005} | 0.001 ± 0.001 {0.001}\!\pm\!{0.001} |

**说明**: TABLE IX: Sensitivity Analysis of Transition
## 实验解读

- 评价重点:围绕 agile-locomotion、接触推理、人形机器人、足式运动、运动模仿,优先看真实机器人结果、扰动恢复、接触稳定性、控制误差、成功率、实时性和 sim-to-real 差距。
- System0 视角:实验结论不只看最终成功率,还要看闭环过程中是否能处理状态估计误差、执行延迟、碰撞/接触突变和外部扰动。
- 与上层系统的关系:如果方法能在快闭环内稳定工作,它可以作为 System1 技能或 System2 规划的底层执行保障;如果只在离线或慢速设定下成立,则还不能直接作为身体级快闭环模块。
## 批判性思考

### 优点

- 论文把 agile-locomotion、接触推理、人形机器人、足式运动、运动模仿 放到机器人身体级闭环里讨论,和只做高层语义推理的工作相比,更接近真实机器人部署时会遇到的状态、动力学和安全约束。
- 若论文同时给出公式、系统结构图、硬件实验或消融结果,这些证据可以帮助判断方法是否真的能支撑毫秒到百毫秒级反应。

### 局限

- 需要重点核查控制频率、推理延迟、传感延迟、失败恢复、硬件磨损、未见场景泛化和安全边界;这些因素决定它能否成为可靠的 System0 模块。
- 对学习型方法,还要区分训练环境中的鲁棒性和真实部署中的鲁棒性,尤其是接触丰富任务、动态运动任务和人形机器人任务。
## 论文定位

- 研究对象:HUSKY: Humanoid Skateboarding System via Physics-Aware Whole-Body Control。
- 关键词:agile-locomotion、接触推理、人形机器人、足式运动、运动模仿、non-prehensile-manipulation、鲁棒控制、全身控制。
- System0 关联:关注机器人身体级快反应链路中“状态/接触/局部感知 -> 约束/优化/策略 -> 动作”的短闭环,不把问题停留在语言规划或任务分解层。

## 速查卡片

> [!summary] HUSKY
> - **论文**: https://www.roboticsproceedings.org/rss22/p019.pdf
> - **arXiv**: http://arxiv.org/abs/2602.03205v2
> - **arXiv HTML**: https://arxiv.org/html/2602.03205v2
