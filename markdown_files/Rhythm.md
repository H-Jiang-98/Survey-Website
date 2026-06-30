---
title: "Rhythm: Learning Interactive Whole-Body Control for Dual Humanoids"
method_name: "Rhythm"
authors: ["Hongjin Chen"]
year: 2026
venue: "RSS"
tags: ["contact-reasoning", "robust-control", "legged-locomotion", "reinforcement-learning", "imitation-learning", "humanoid", "whole-body-control"]
image_source: "online"
arxiv_html: "https://arxiv.org/html/2603.02856v2"
---
# Rhythm
## 一句话总结

> Rhythm: Learning Interactive Whole-Body Control for Dual Humanoids 主要落在 [[接触推理]]、[[人形机器人]]、[[足式运动]]、[[运动模仿]] 这一类问题上;从 System0 视角看,它的价值在于把身体状态、接触变化、局部感知或动力学约束接到机器人低层闭环里,让动作生成更稳定、安全、实时。
## 核心贡献

1. 围绕 **Rhythm: Learning Interactive Whole-Body Control for Dual Humanoids** 建立了一个与 接触推理、人形机器人、足式运动、运动模仿、强化学习、retargeting 相关的机器人问题设定,重点不是高层语义规划,而是身体级闭环中的状态变化、接触变化或控制约束。
2. 方法层面的关键价值在于把感知、动力学、约束优化、策略学习或硬件反馈组织成可执行的低层动作生成机制,适合放进 System0 的快反应模块中理解。
3. 对系统集成而言,这类工作补上了 System1 技能选择与电机级控制之间的一层:它负责把上层目标变成动态可行、可恢复、可安全过滤的动作。
4. 若论文包含真实机器人、接触任务或仿真到真实迁移,核心应看它是否报告了闭环稳定性、扰动恢复、硬件延迟、失败模式和跨场景泛化。
## 问题背景

这篇论文可以放在 System0 的“具身快系统”背景下阅读:机器人需要在很短时间内根据本体状态、接触、力/触觉、局部视觉或动力学预测做出动作,而不是等待慢速规划器重新推理整个任务。接触推理、人形机器人、足式运动、运动模仿、强化学习、retargeting、鲁棒控制 相关问题的难点通常不在“知道要做什么”,而在动作执行过程中状态会快速偏离、接触会突然改变、约束会实时收紧,低层系统必须及时修正。

### System0 关联

精读时优先看三件事:第一,论文使用哪些快速反馈量,例如关节状态、接触状态、力觉、触觉、深度或局部视觉;第二,反馈如何进入 MPC、WBC、策略网络、安全过滤器或恢复控制器;第三,输出是否直接影响机器人下一步动作,而不是只服务离线分析或高层任务规划。
## 方法详解

### System0 视角解析

从 System0 角度看,这篇论文的方法部分应当围绕 接触推理、人形机器人、足式运动、运动模仿、强化学习、retargeting、鲁棒控制、全身控制 来读。核心问题是:系统如何从快速变化的机器人状态中构造控制输入,如何处理接触、碰撞、约束、模型误差或执行延迟,以及如何把这些信息变成下一时刻的动作。

如果方法里有优化问题,重点看目标函数、约束和求解频率;如果方法里有学习策略,重点看观测空间、动作空间、训练扰动和部署时的闭环频率;如果方法里有硬件或传感器,重点看传感链路是否真的进入控制回路。这样读可以把论文和 System2 语言规划、System1 技能编排区分开来。

### 关键公式与机制

#### 公式 1: [[优化目标/约束]]

$$q^{*}=\arg\min_{q}\left(E_{self}(q)+E_{inter}(q)\right)\quad\text{s.t.}\quad q\in\mathcal{C}_{phy},$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 2: [[优化目标/约束]]

$$\begin{split}E_{self}(q)=&\sum_{a\in\{1,2\}}\sum_{p_{i}\in\mathcal{V}_{a}}\lVert \mathcal{L}(p_{i})-\mathcal{L}(p_{i}^{ind})\rVert^{2}\\ &+\lambda_{rot}\sum_{a\in\{1,2\}}\sum_{k\in\mathcal{B}_{a}}\lVert \theta_{k}\ominus\hat{\theta}_{k}^{src} \rVert^{2},\end{split}$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 3: [[优化目标/约束]]

$$E_{inter}(q)=\sum_{(i,j)\in\mathcal{E}_{inter}}\omega_{ij}(d_{ij})\cdot\lVert(p_{i}-p_{j})-(\hat{p}_{i}^{uni}-\hat{p}_{j}^{uni})\rVert^{2},$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 4: [[优化目标/约束]]

$$r_{inter}=\exp\left(-\frac{1}{\sigma_{inter}}\sum_{(i,j)\in\mathcal{E}_{inter}}\omega_{ij}\lVert d_{ij}^{sim}-\hat{d}_{ij}^{ref} \rVert^{2}\right),$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 5: [[优化目标/约束]]

$$\begin{split}\mathcal{J}_{self}=\sum_{k\in\{1,2\}}\Big(&\lVert \mathcal{L}(f(q_{t}^{(k)}))-\mathcal{L}(P_{ind}^{(k)})\rVert^{2}\\ &+\lambda_{rot}\sum_{b\in\mathcal{B}_{k}}\lVert \theta_{b}(q_{t}^{(k)})\ominus\hat{\theta}_{b}^{src} \rVert^{2}\Big).\end{split}$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 6: [[优化目标/约束]]

$$\begin{split}\mathcal{J}_{inter}=\sum_{(i,j)\in\mathcal{E}_{inter}}\omega_{ij}\big\lVert&(f_{i}(q_{t}^{(1)})-f_{j}(q_{t}^{(2)}))\\ &-(p_{t,i}^{uni,(1)}-p_{t,j}^{uni,(2)})\big \rVert^{2}.\end{split}$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 7: [[优化目标/约束]]

$$=w_{self}(\lVert \mathcal{L}(q)\!-\!L_{ref} \rVert^{2}\!+\!\lambda_{rot}\lVert \Delta\theta \rVert^{2})$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 8: [[优化目标/约束]]

$$r_{inter}=\exp\left(-\frac{1}{\sigma_{inter}}\sum_{(i,j)\in\mathcal{E}}w_{ij}\cdot\lVert p_{ij}^{sim}-p_{ij}^{ref} \rVert^{2}\right).$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 9: [[优化目标/约束]]

$$E_{act}=\sum_{k\in\mathcal{V}_{act}}\left(\beta\lVert C_{k}^{sim}-1 \rVert+(1-\beta)\mathcal{L}_{force}(f_{k}^{sim})\right).$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 10: [[优化目标/约束]]

$$
\mathcal{L}_{force}(f)=\begin{cases}1.0-f/F_{min}&\text{if}f F_{max}\text{(Too Strong)}\\ 0&\text{otherwise (Valid)}\end{cases}.
$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。
## 关键图表

### Figure 1: Overview of Rhythm. IAMR utilizes decoupled optimization to generate high-quality huma

![Figure 1](https://arxiv.org/html/2603.02856v2/figure/pipeline_new.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Overview of Rhythm. IAMR utilizes decoupled optimization to generate high-quality huma”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 2: Overview of MAGIC. MAGIC contains ∼ $\sim$ 3 hours of high-fidelity interaction data bal

![Figure 2](https://arxiv.org/html/2603.02856v2/figure/rss_dataset.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Overview of MAGIC. MAGIC contains ∼ $\sim$ 3 hours of high-fidelity interaction data bal”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 3: Qualitative Visualization of Policy. Single Agent (blue) drifts into collisions. w/o C

![Figure 3](https://arxiv.org/html/2603.02856v2/figure/policy_demo.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Qualitative Visualization of Policy. Single Agent (blue) drifts into collisions. w/o C”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### 论文表格

#### Table 1: TABLE I: Quantitative Results of Retargeting. Comparison across four interaction categories. Metrics include Safety (IPR

| | Safety | Fidelity | Utility | | | |
| --- | --- | --- | --- | --- | --- | --- |
| IPR (%) ↓ \downarrow | MPD (cm) ↓ \downarrow | IEE (%) ↓ \downarrow | F1-S ↑ \uparrow | F1-L ↑ \uparrow | DSR (%) ↑ \uparrow | |
| MAGIC: Collaborate | | | | | | |
| GMR | 0.14 | 1.2 | 4.3 | 0.602 | 0.804 | 85.5 |
| OR | 0.20 | 1.4 | 4.1 | 0.747 | 0.902 | 87.4 |
| DOR | 0.00 | 0.0 | 3.9 | 0.711 | 0.899 | 89.5 |
| IAMR | 0.00 | 0.0 | 3.7 | 0.785 | 0.936 | 89.0 |
| MAGIC: Light Contact | | | | | | |
| GMR | 2.18 | 3.3 | 4.6 | 0.738 | 0.893 | 48.5 |
| OR | 7.62 | 5.9 | 3.6 | 0.844 | 0.912 | 63.1 |
| DOR | 0.00 | 0.0 | 3.6 | 0.810 | 0.918 | 69.4 |
| IAMR | 0.00 | 0.0 | 3.1 | 0.905 | 0.935 | 75.3 |
| MAGIC: Intensive Contact | | | | | | |
| GMR | 35.2 | 3.8 | 9.6 | 0.864 | 0.928 | 45.5 |
| OR | 47.3 | 5.3 | 8.0 | 0.884 | 0.929 | 56.5 |
| DOR | 0.00 | 0.0 | 7.8 | 0.883 | 0.925 | 63.3 |
| IAMR | 0.00 | 0.0 | 6.6 | 0.932 | 0.941 | 78.3 |
| Inter-X | | | | | | |
| GMR | 11.7 | 1.7 | 8.0 | 0.598 | 0.752 | 31.7 |
| OR | 18.4 | 2.6 | 6.8 | 0.587 | 0.791 | 46.3 |
| DOR | 0.00 | 0.0 | 6.7 | 0.589 | 0.795 | 52.9 |
| IAMR | 0.00 | 0.0 | 4.9 | 0.843 | 0.860 | 69.9 |

**说明**: TABLE I: Quantitative Results of Retargeting. Comparison across four interaction categories. Metrics include Safety (IPR, MPD), Fidelity (IEE, F1), and Utility (DSR). IAMR achieves the best balance, strictly eliminating penetration (IPR=0) while maximizing contact F1 scores.

#### Table 2: TABLE II: Quantitative Results of Policy. We evaluate the contribution of each component. Our full method achieves the m

| | Interaction | Contact | | |
| --- | --- | --- | --- | --- |
| ISR (%) ↑ \uparrow | IEE (%) ↓ \downarrow | CSR (%) ↑ \uparrow | CER ↓ \downarrow | |
| MAGIC: Collaborate | | | | |
| Single Agent | 18.7 | 38.9 | 100.0 | 0.000 |
| w/o Peer Obs | 19.5 | 47.0 | 100.0 | 0.000 |
| w/o Contact Rew | 93.4 | 4.7 | 100.0 | 0.000 |
| w/o Interact Rew | 58.1 | 15.1 | 100.0 | 0.000 |
| Ours (Full) | 92.9 | 4.8 | 100.0 | 0.000 |
| MAGIC: Light Contact | | | | |
| Single Agent | 34.3 | 19.9 | 24.1 | 0.283 |
| w/o Peer Obs | 48.9 | 13.9 | 18.6 | 0.268 |
| w/o Contact Rew | 85.9 | 5.4 | 52.1 | 0.203 |
| w/o Interact Rew | 48.7 | 19.3 | 28.1 | 0.243 |
| Ours (Full) | 90.0 | 4.2 | 78.0 | 0.120 |
| MAGIC: Intensive Contact | | | | |
| Single Agent | 24.0 | 29.9 | 37.5 | 0.312 |
| w/o Peer Obs | 34.1 | 21.4 | 43.7 | 0.280 |
| w/o Contact Rew | 77.3 | 7.7 | 70.6 | 0.174 |
| w/o Interact Rew | 51.3 | 17.0 | 56.8 | 0.211 |
| Ours (Full) | 75.2 | 7.9 | 78.8 | 0.159 |
| Inter-X | | | | |
| Single Agent | 25.7 | 26.9 | 57.4 | 0.256 |
| w/o Peer Obs | 73.2 | 7.6 | 75.3 | 0.143 |
| w/o Contact Rew | 95.1 | 3.4 | 68.3 | 0.208 |
| w/o Interact Rew | 63.2 | 9.7 | 78.6 | 0.110 |
| Ours (Full) | 92.8 | 3.5 | 77.4 | 0.125 |

**说明**: TABLE II: Quantitative Results of Policy. We evaluate the contribution of each component. Our full method achieves the most robust balance, effectively integrating coarse-grained geometric alignment (low IEE) with fine-grained physical contact fidelity (high CSR).

#### Table 3: TABLE III: Main Results for Real Robot Experiments. We conducted 10 trials for each task and evaluated success based on

| Task | Method | Success / Total | Rate (%) |
| --- | --- | --- | --- |
| Hug | Single Agent | 8 / 30 8~/~30 | 26.7% |
| Ours | 26 / 30 | 86.7% | |
| Shoulder | Single Agent | 6 / 30 6~/~30 | 20.0% |
| Ours | 24 / 30 | 80.0% | |
| Greeting | Single Agent | 11 / 90 11~/~90 | 12.2% |
| Ours | 74 / 90 | 82.2% | |

**说明**: TABLE III: Main Results for Real Robot Experiments. We conducted 10 trials for each task and evaluated success based on contact establishment at specific keyframes (K K frames per trial).

#### Table 4: TABLE IV: Reward Terms and Weights used in IGRL

| Term | Weight | Equation | Description |
| --- | --- | --- | --- |
| Interaction Graph Objectives | | | |
| Interact Edge | 1.5 1.5 | $\exp\left(-\frac{1}{\sigma_{i}}\sum_{(i,j)\in\mathcal{E}}w_{ij}\lVert p_{ij}^{sim}-p_{ij}^{ref} \rVert^{2}\right)$ | Enforces geometric consistency of interaction edges. |
| Contact | 1.0 1.0 | $$ | Balances active contact enforcement (force constrained) and ghost interaction suppression. |
| Motion Tracking Objectives | | | |
| Upper Pos | 1.0 1.0 | $\exp(-\frac{1}{N_{u}}\sum_{k\in Upper}\lVert p_{k}^{sim}-p_{k}^{ref} \rVert^{2}/\sigma^{2}_{pos})$ | Tracks Euclidean positions of upper body links. |
| Upper Ori | 1.0 1.0 | $\exp(-\frac{1}{N_{u}}\sum_{k\in Upper}\lVert \log((R_{k}^{sim})^{\top}R_{k}^{ref})\rVert^{2}/\sigma^{2}_{ori})$ | Tracks orientation of upper body links. |
| Upper Lin Vel | 1.0 1.0 | $\exp(-\frac{1}{N_{u}}\sum_{k\in Upper}\lVert v_{k}^{sim}-v_{k}^{ref} \rVert^{2}/\sigma^{2}_{vel})$ | Matches linear velocities of upper body. |
| Upper Ang Vel | 1.0 1.0 | $\exp(-\frac{1}{N_{u}}\sum_{k\in Upper}\lVert \omega_{k}^{sim}-\omega_{k}^{ref} \rVert^{2}/\sigma^{2}_{ang})$ | Matches angular velocities of upper body. |
| Lower Pos | 0.5 0.5 | $\exp(-\frac{1}{N_{l}}\sum_{k\in Lower}\lVert p_{k}^{sim}-p_{k}^{ref} \rVert^{2}/\sigma^{2}_{pos})$ | Tracks Euclidean positions of lower body links. |
| Lower Ori | 0.5 0.5 | $\exp(-\frac{1}{N_{l}}\sum_{k\in Lower}\lVert \log((R_{k}^{sim})^{\top}R_{k}^{ref})\rVert^{2}/\sigma^{2}_{ori})$ | Tracks orientation of lower body links. |
| Lower Lin Vel | 0.5 0.5 | $\exp(-\frac{1}{N_{l}}\sum_{k\in Lower}\lVert v_{k}^{sim}-v_{k}^{ref} \rVert^{2}/\sigma^{2}_{vel})$ | Matches linear velocities of lower body. |
| Lower Ang Vel | 0.5 0.5 | $\exp(-\frac{1}{N_{l}}\sum_{k\in Lower}\lVert \omega_{k}^{sim}-\omega_{k}^{ref} \rVert^{2}/\sigma^{2}_{ang})$ | Matches angular velocities of lower body. |
| Anchor Pos | 0.3 0.3 | $\exp(-\lVert p_{root}^{sim}-p_{root}^{ref} \rVert^{2}/\sigma^{2}_{root})$ | Tracks root position in world frame to prevent global drift. |
| Anchor Ori | 0.5 0.5 | $\exp(-\lVert \log((R_{root}^{sim})^{\top}R_{root}^{ref})\rVert^{2}/\sigma^{2}_{root\_ori})$ | Tracks root heading in world frame. |
| Regularization & Penalties | | | |
| Action Rate | - 0.3 -0.3 | $\mathbf{a}_{t}-\mathbf{a}_{t-1}\lVert^{2}$ | Penalizes action changes to ensure smooth control. |
| Feet Slip | - 0.5 -0.5 | $\sum_{k\in Feet}\mathbb{I}(contact_{k})\cdot\lVert v_{k,xy} \rVert^{2}$ | Penalizes foot sliding velocity during ground contact. |
| Joint Limit | - 10.0 -10.0 | ∑ j max  (0, \| q j \| - q l  i  m  i  t)  $\sum_{j}\max(0,\lVert q_{j} \rVert-q_{limit})$ | Penalizes violations of physical joint limits. |
| Torque | 10 - 4 10^{-4} | $\tau\lVert^{2}$ | Prevents excessive torques. |

**说明**: TABLE IV: Reward Terms and Weights used in IGRL

#### Table 5: TABLE V: Domain Randomization Parameters

| Term | Value |
| --- | --- |
| Dynamics Randomization | |
| Link Mass | U . . default $\mathcal{U}[0.9,1.1]\times\text{default} (per link)$ |
| CoM Offset (Torso) | $\mathcal{U}[-0.05,0.05] m$ |
| Joint Friction | Static: U  [0.3, 2.0 ]  $\mathcal{U}[0.3,2.0], Dynamic: U [0.3, 1.6 ] \mathcal{U}[0.3,1.6]$ |
| Actuator Gains | Stiffness/Damping: U  [0.9, 1.1 ]  $\mathcal{U}[0.9,1.1]$ |
| Restitution | U  [0.0, 0.8 ]  $\mathcal{U}[0.0,0.8] (Ground contact)$ |
| Default Joint Pos | $\theta_{0}\sim\mathcal{U}[-0.01,0.01] rad (Calibration)$ |
| Control Delay | U  [0, 15 ]  $\mathcal{U}[0,15] ms$ |
| External Perturbations | |
| Robot Push (Linear) | v x, y ∼ U  [- 0.4, 0.4 ] v_{x,y}\sim $\mathcal{U}[-0.4,0.4] m/s, v z ∼ U [- 0.16, 0.16 ] v_{z}\sim\mathcal{U}[-0.16,0.16] m/s$ |
| Robot Push (Angular) | $\omega_{x,y}\sim\mathcal{U}[-0.4,0.4] rad/s, ω z ∼ U [- 0.64, 0.64 ] \omega_{z}\sim\mathcal{U}[-0.64,0.64] rad/s$ |
| Push Interval | Applied every 1.0 ∼ 3.0 1.0\sim 3.0 s |
| Initial Pose Offset |   p  o  s ∈ [- 5, 5 ] \Delta pos\in[-5,5] cm,   y  a  w ∈ [- 0.2, 0.2 ] \Delta yaw\in[-0.2,0.2] rad |
| Communication Degradation | |
| Peer Latency | Proprioception & Relocalization: U  [20, 60 ]  $\mathcal{U}[20,60] ms$ |

**说明**: TABLE V: Domain Randomization Parameters
## 实验解读

- 评价重点:围绕 接触推理、人形机器人、足式运动、运动模仿、强化学习,优先看真实机器人结果、扰动恢复、接触稳定性、控制误差、成功率、实时性和 sim-to-real 差距。
- System0 视角:实验结论不只看最终成功率,还要看闭环过程中是否能处理状态估计误差、执行延迟、碰撞/接触突变和外部扰动。
- 与上层系统的关系:如果方法能在快闭环内稳定工作,它可以作为 System1 技能或 System2 规划的底层执行保障;如果只在离线或慢速设定下成立,则还不能直接作为身体级快闭环模块。
## 批判性思考

### 优点

- 论文把 接触推理、人形机器人、足式运动、运动模仿、强化学习 放到机器人身体级闭环里讨论,和只做高层语义推理的工作相比,更接近真实机器人部署时会遇到的状态、动力学和安全约束。
- 若论文同时给出公式、系统结构图、硬件实验或消融结果,这些证据可以帮助判断方法是否真的能支撑毫秒到百毫秒级反应。

### 局限

- 需要重点核查控制频率、推理延迟、传感延迟、失败恢复、硬件磨损、未见场景泛化和安全边界;这些因素决定它能否成为可靠的 System0 模块。
- 对学习型方法,还要区分训练环境中的鲁棒性和真实部署中的鲁棒性,尤其是接触丰富任务、动态运动任务和人形机器人任务。
## 论文定位

- 研究对象:Rhythm: Learning Interactive Whole-Body Control for Dual Humanoids。
- 关键词:接触推理、人形机器人、足式运动、运动模仿、强化学习、retargeting、鲁棒控制、全身控制。
- System0 关联:关注机器人身体级快反应链路中“状态/接触/局部感知 -> 约束/优化/策略 -> 动作”的短闭环,不把问题停留在语言规划或任务分解层。

## 速查卡片

> [!summary] Rhythm
> - **论文**: https://www.roboticsproceedings.org/rss22/p034.pdf
> - **arXiv**: http://arxiv.org/abs/2603.02856v2
> - **arXiv HTML**: https://arxiv.org/html/2603.02856v2
