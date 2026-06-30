---
title: "Opening the Sim-to-Real Door for Humanoid Pixel-to-Action Policy Transfer"
method_name: "DoorMan"
authors: [Haoru Xue, Tairan He, Zi Wang, Qingwei Ben, Wenli Xiao, Zhengyi Luo, Xingye Da, Fernando Castañeda, Guanya Shi, Shankar Sastry, Linxi "Jim" Fan, Yuke Zhu]
year: 2026
venue: CVPR
tags: [humanoid, loco-manipulation, sim-to-real, teacher-student, GRPO, domain-randomization, RGB-policy, articulated-object]
zotero_collection: _inbox
image_source: online
arxiv_html: https://arxiv.org/html/2512.01061v1
created: 2026-06-29
---

# Opening the Sim-to-Real Door for Humanoid Pixel-to-Action Policy Transfer

## 元信息

| 项目 | 内容 |
|------|------|
| 作者 | Haoru Xue, Tairan He, Zi Wang, Qingwei Ben, Wenli Xiao, Zhengyi Luo, Xingye Da, Fernando Castañeda, Guanya Shi, Shankar Sastry, Linxi "Jim" Fan, Yuke Zhu |
| 机构 | NVIDIA / UC Berkeley / CMU 等（作者多隶属 NVIDIA GEAR 与高校实验室） |
| 会议 | CVPR 2026 |
| 类别 | Humanoid / 视觉 loco-manipulation / Sim-to-Real RL |
| 日期 | 2025-12（arXiv v1） |
| 项目主页 | （论文未给出公开主页） |
| 链接 | [arXiv](https://arxiv.org/abs/2512.01061) / [PDF](https://arxiv.org/pdf/2512.01061) / [CVPR Poster](https://cvpr.thecvf.com/virtual/2026/poster/39500) |

---

## 一句话总结

> 用「教师-学生-自举（teacher-student-bootstrap）」三阶段流水线，在 IsaacLab 里靠大规模物理+视觉随机化训练出**纯 RGB**的人形全身 loco-manipulation 策略，零样本迁移到真实世界开各种门，并在完成时间上比人类遥操作快最高 31.7%。

---

## 核心贡献

1. **首个纯 RGB 人形 sim-to-real 关节物体 loco-manipulation 策略**: 不依赖深度、物体位姿先验或硬编码运动基元，仅用 RGB + 本体感知即可零样本泛化到多种门型（拉/推、杠杆/推杠）。
2. **Teacher-Student-Bootstrap 三阶段流水线**: 含一套针对长程任务的 [[Staged-Reset Exploration|分阶段重置探索]] 机制（稳定特权教师训练），以及一个 [[GRPO]] 微调阶段（缓解学生策略的部分可观测性）。
3. **IsaacLab 原生程序化造门管线**: 物理精确、视觉多样、可大规模并行/分布式 RL，覆盖门尺寸、铰链阻尼、闩锁机制、材质、灯光、相机内外参等的随机化。
4. **超越人类遥操作**: 在同一全身控制器（[[HOMIE]]）下，自主策略成功率 83% 对比专家 80% / 非专家 60%，并把交互完成时间缩短 23.1%–31.7%。

---

## 问题背景

### 要解决的问题
如何让人形机器人**只用一个移动的第一人称 RGB 相机 + 本体感知**，就能完成接触丰富、需要全身协调与平衡的日常 loco-manipulation 任务。论文把**开门**作为代表性高难度基准：机器人要从晃动的 egocentric 相机里识别把手位置、旋转弹簧加载的把手、跟踪门板的圆弧运动、并在铰链反力下维持平衡。

### 现有方法的局限
- 专门针对门的系统大多依赖**深度传感、物体中心特征、或硬编码运动基元**，且常跑在轮式平台上（calvert2025, xiong2024, weng2025 HDMI）。
- 部分工作**简化接触力学或要求精确物体定位**（traverse-doors-2025）。
- DARPA Robotics Challenge 时代系统严重依赖脚本与操作员介入；近期遥操作流水线（StageACT）脆弱、且评测被限定在与采集数据**完全相同**的场景/背景/光照/时段。
- 行为克隆（BC）路线的性能上限被人类遥操作数据质量卡死，而全身遥操作本身不直观，效率与成功率都有天花板。

### 本文的动机
- 视觉 sim-to-real 在 locomotion、运动模仿、灵巧操作上已有强结果，但 loco-manipulation（感知+平衡+接触+导航交织）仍欠探索。论文识别两大根本挑战：**(i)** 算法须简单、可扩展、对部分可观测鲁棒，能协调视觉与全身控制；**(ii)** 视觉 sim-to-real gap 横跨巨大的外观与物理变化空间，需要**广而异质**的数据而非少量精心场景。
- 关键思想：**不复刻真实场景**，而是用程序化随机化覆盖一个宽广的可变性包络（variability envelope），让策略在零样本下也能泛化到训练中未见的真实门。

---

## 方法详解

### 模型架构

DoorMan 在预训练全身控制器 [[HOMIE]]（ben2025homie）之上构建，采用经典 **teacher-student 蒸馏 + RL 自举** 的三阶段管线（见 Figure 2）。问题建模为部分可观测 MDP（[[POMDP]]）$\mathcal{P}=(\mathcal{S},\mathcal{A},\mathcal{O},T,\mathcal{R},\mathcal{O},\gamma,\rho_0)$。

- **平台/动作空间**: Unitree G1，29 个身体关节 + 14 个手部关节；策略输出目标关节角，动作维度高达 33，由底层电机以 PD 控制律跟踪，**策略需在 50 Hz 一致推理**。
- **Phase 1 — 教师策略 $\pi_T(a\mid s)$**: 用特权观测，[[PPO]] 训练，配 [[Staged-Reset Exploration|分阶段重置探索]]。
- **Phase 2 — 学生蒸馏**: 用 [[DAgger]] 把教师蒸馏成 RGB 学生策略，加重度视觉随机化。
- **Phase 3 — GRPO 自举微调**: 用 [[GRPO]] + 二值成功信号细化学生，弥补部分可观测性。

### 核心模块

#### 模块1: Teacher Policy（特权教师策略）

**设计动机**: 用仿真中才有的**特权信息**先学会高质量开门行为，作为学生的监督来源；目标是在部署时**彻底去掉这些先验**，由纯 RGB 学生接管。

**具体实现**: 教师 $\pi_T(a\mid s)$ 在时刻 $t$ 可访问的特权观测 $o_T$ 包括：
- 机器人根到门的真值变换 $\xi_{\text{RD}}$；
- 左右手到门把手的变换 $\xi_{\text{LD}},\xi_{\text{RD}}$；
- 18 个手部 body 上的净接触力旋量 $\tau_H\in\mathbb{R}^{18\times6}$；
- 根线速度 $v_R\in\mathbb{R}^3$。

教师用标准 [[PPO]]（rsl_rl）训练，奖励 shaping 见 Table 2（六阶段、分阶段塑形）。

#### 模块2: Student Distillation（RGB 学生策略）

**设计动机**: 学生只有非特权本体感知 + RGB 图像 + 时序上下文，需要在自己的输入分布上被监督，因此用 [[DAgger]] 而非纯 BC（BC 只覆盖教师分布）。

**具体实现**:
- 学生输入：关节角 $q$、关节速度 $\dot q$、根角速度 $\dot\omega\in\mathbb{R}^3$，以及 RGB 观测；
- 视觉用 [[ResNet]] 编码器，latent 与本体感知特征拼接后送入**两层 LSTM（各 512 单元）**；
- 再经**三层 MLP（512, 256, 128）**映射到目标关节角；
- 视觉编码器**与策略联合微调**；
- 用 [[DAgger]] 交互式蒸馏，直接在学生输入分布上监督。

#### 模块3: Multi-Stage 全身 loco-manipulation 与 Staged-Reset Exploration

**设计动机**: 接触丰富的精细操作（如开门）在 on-policy RL 里很难稳定探索并推进到后期阶段——抓住把手却不会正确旋转，会因电机扭矩过大、接触力峰值、甚至摔倒而被惩罚，导致策略「unlearn」抓取行为、回避进入下一阶段。

**具体实现**:
- 类似 WoCoCo，设计**分阶段奖励**把任务拆成原子阶段，注入人类先验（用把手与铰链状态区分 approaching / opening / traversing）。开门任务被拆成**六个阶段**：(0) 走向门、(1) 预抓取、(2) 抓取、(3) 开门、(4) 摆门、(5) 穿过门。
- 受 Go-Explore（first_return2021）启发，利用物理仿真器的**完全可恢复性**：当某环境推进到下一阶段时，一个滚动缓冲区缓存该步最近 **100 个**机器人+环境（门）状态快照（含所有关节/刚体的广义坐标）；reset 时以非零概率把机器人随机重置到初始阶段或某个中间阶段（见 Figure 3）。
- 形式化：把状态空间分成不相交子集 $\{S_1,\dots,S_K\}$，由窄「桥」$\mathcal{B}_{y,y+1}$ 连接，跨桥探索概率 $p_{\text{bridge}}\ll1$，导致从 $\rho_0$ 训练的策略早期难达下游阶段。分阶段重置律把占用测度**重新加权到后期区域**，提升这些状态的梯度更新频率与有效幅度。

#### 模块4: RL Finetuning for Partial Observability（GRPO 自举）

**设计动机**: 学生因遮挡而缺失关键特征，单靠 BC 损失达不到最优；学生需要在**自己的 rollout** 上自举，发现弥补部分可观测的新策略（如调整身位让操作区域保持在相机视野内）。

**具体实现**:
- 用 [[GRPO]]（一种省去价值函数、由分组轨迹得分估计 baseline 的 actor-only PPO 变体）微调学生；
- 一个 batch 采 $G$ 条 rollout，各自回报 $R_i$，用组内归一化优势 $\hat A_i$（公式4）+ clipped PPO surrogate（公式5）更新；
- 微调主要用**二值任务成功信号**，加上关节速度/加速度/动作率等简单 shaping 惩罚做正则；因此可作为**即插即用**手段去改进任何已有非零成功率的 loco-manipulation 任务。
- 实测：学生学会了教师从未演示的**补偿行为**（把操作物体保持在画面中心、调整末端位姿维持可见性）。

#### 模块5: Massive-Scale Simulation Randomization（程序化造门）

**设计动机**: 把视觉与动力学多样性扩到「前所未有」的规模，且**不复刻真实场景**——所有评测真实场景在训练中均未见。

**具体实现**:
- **物理变化**: 5 种门型，覆盖 3 大类（带旋转把手的推门、带旋转把手的拉门、带推杠的推门）；随机化门尺寸、把手位置、铰链阻尼、把手阻力矩等；尤为关键的是用**真实闩锁机制**捕捉开门瞬间全身动力学的突变。
- **视觉变化**: 从 IsaacLab 的 [[PBR]] 材质随机抽取贴图并应用到所有表面；用 **5233 张 dome-light 贴图**模拟不同地点与时段；用 RTX 实时渲染器的 performance 模式（开运动模糊、自动白平衡），相机内外参对齐并轻微随机化——这些对再现腿式机器人持续接触切换下的相机抖动至关重要。

### 关键公式与机制

#### 公式1: [[Staged-Reset Exploration|分阶段重置律]]（混合系数）

$$
\alpha=(\alpha_1,\dots,\alpha_K),\qquad \sum_{y=1}^{K}\alpha_y=1
$$

**含义**: 指定从各阶段重置分布 $\rho_y$ 初始化的 rollout 比例。

**符号说明**:
- $\alpha_y$: 从第 $y$ 阶段重置分布采样的占比，所有阶段占比之和为 1；
- $K$: 阶段总数。

#### 公式2: 混合初始分布

$$
\tilde{\rho}_\alpha=\sum_{y=1}^{K}\alpha_y\,\rho_y
$$

**含义**: 由分阶段重置律得到的实际初始状态分布，是各阶段重置分布的凸组合。

**符号说明**:
- $\rho_y$: 第 $y$ 阶段的重置（快照）分布；
- $\tilde\rho_\alpha$: 混合后的初始分布（替换原始 $\rho_0$）。

#### 公式3: 更新后的折扣占用测度

$$
d_\pi^\alpha(s)=(1-\gamma)\sum_{t=0}^{\infty}\gamma^{t}\,\Pr\!\left(s_t=s \mid s_0\sim\tilde{\rho}_\alpha,\ \pi\right)
$$

**含义**: 在混合初始分布下策略 $\pi$ 的折扣状态占用测度。论文据此论证分阶段重置把占用测度**重新加权到后期阶段区域**，提高对这些状态的梯度更新频率与有效幅度，从而改善长程信用分配。

**符号说明**:
- $\gamma\in[0,1)$: 折扣因子；
- $\Pr(s_t=s\mid\cdot)$: 边际概率；
- $d_\pi^\alpha(s)$: 状态 $s$ 的折扣占用测度。

#### 公式4: [[GRPO]] 组内归一化优势

$$
\hat{A}_i=\frac{R_i-\operatorname{mean}(R)}{\operatorname{std}(R)}
$$

**含义**: 用一组 rollout 回报的均值/标准差对单条轨迹回报标准化，作为 actor-only 的优势估计（省去价值网络）。

**符号说明**:
- $R_i$: 第 $i$ 条 rollout 的回报；
- $\operatorname{mean}(R),\operatorname{std}(R)$: 同一组 $G$ 条 rollout 回报的均值与标准差。

#### 公式5: [[GRPO]] 裁剪目标

$$
\mathcal{L}_{\text{GRPO}}(\theta)=\mathbb{E}_{i,t}\!\left[\min\!\Big(r_{i,t}(\theta)\,\hat{A}_i,\ \operatorname{clip}\big(r_{i,t}(\theta),\,1-\epsilon,\,1+\epsilon\big)\,\hat{A}_i\Big)\right]
$$

其中重要性比为

$$
r_{i,t}(\theta)=\frac{\pi_\theta(a_{i,t}\mid o_{i,t})}{\pi_{\text{old}}(a_{i,t}\mid o_{i,t})}
$$

**含义**: 标准 PPO 风格的裁剪 surrogate，用组相对优势 $\hat A_i$ 指导更新，把学生策略在**自己的部分观测**下做强化细化。

**符号说明**:
- $r_{i,t}(\theta)$: 新旧策略在 $(a_{i,t},o_{i,t})$ 上的概率比；
- $\epsilon$: 裁剪范围；
- $\mathbb{E}_{i,t}$: 对组内样本与时间步取期望。

---

## 关键图表

<!-- 图片均使用 arXiv HTML 在线链接，未本地化 -->

### Figure 1: Real-world generalization of DoorMan / 真实世界泛化

![Figure 1](https://arxiv.org/html/2512.01061v1/x1.png)

**说明**: DoorMan 在真实世界的泛化展示。上排为多样的把手视觉与物理形状；中排为多样的墙板/门板外观；下排为自然地推开与拉开门。佐证策略在训练未见的真实门上零样本工作。

### Figure 2: DoorMan training pipeline / 三阶段训练流水线

![Figure 2](https://arxiv.org/html/2512.01061v1/x2.png)

**说明**: 全流程均在 IsaacLab 中交互完成。Phase 1 用特权观测训练教师；Phase 2 用 [[DAgger]] 蒸馏成 RGB 学生；Phase 3 用 [[GRPO]] + 二值成功信号进一步训练学生。这是全文方法的总览图。

### Figure 3: Staged-reset exploration scheme / 分阶段重置探索

![Figure 3](https://arxiv.org/html/2512.01061v1/x3.png)

**说明**: 进入新阶段时把仿真快照缓存进 buffer（最近 100 个）；任务 reset 时以非零概率从某个先前阶段加载快照来初始化。这正是稳定长程教师训练的核心机制，对应公式 1–3 的占用测度重加权。

### Figure 4: Procedurally generated doors / 程序化生成的门

![Figure 4](https://arxiv.org/html/2512.01061v1/x4.png)

**说明**: 用于训练的程序化门，覆盖门板设计、闩锁机制、灯光、材质等；每个并行环境训练在一组独特的门参数上；最后一张为无材质（裸几何）的门。直观体现「大规模物理+视觉随机化」。

### Figure 5: Average performance vs human teleop / 与人类遥操作的平均性能对比

![Figure 5](https://arxiv.org/html/2512.01061v1/x5.png)

**说明**: 所有开门任务的平均表现。左：成功率（越高越好）；右：任务流畅度即完成耗时（越低越好）。DoorMan 真实成功率与专家遥操作持平、比非专家高 28%；流畅度上比专家快 23.8%、比非专家快 31.7%。

### Figure 6(a): Student success during GRPO / GRPO 训练中的学生成功率

![Figure 6a](https://arxiv.org/html/2512.01061v1/x6.png)

**说明**: 三个开门子任务上 GRPO 自举阶段的进展，虚线为教师成功率上界。教师稳定在 80–90% 时，初始学生仅 50–70%（不可恢复的可观测性 gap）；自举结束后学生达 80.8–85.8%，曲线明显趋近教师上界，说明 GRPO 有效缩小了部分可观测性带来的差距。

### Figure 6(b): Teacher training with reset buffer sizes / 不同重置 buffer 大小的教师训练

![Figure 6b](https://arxiv.org/html/2512.01061v1/x7.png)

**说明**: buffer=0/10/100 三种设置下的教师训练进度。buffer=100 时约 500 迭代即达大多数阶段、约 1700 迭代达全部阶段；buffer=10 需 4000+ 迭代；buffer=0（不用重置缓冲）则探索失败——策略难以进入 Stage 2（抓把手），因抓取失败会招致额外碰撞惩罚而「unlearn」。

### Table 1: Success rates under visual randomization / 视觉随机化设置下的成功率（%）

每个配置（实验 1–6）在 120 次未见门试验上评测。Appearance 为视觉变化类型（Solid-color = 仅均匀重着色无纹理；+10%/100% Texture = 纹理随机化比例）；DL 为 dome-light（穹顶光）随机化（✓ 开 / ✗ 关）。

| Experiment | Appearance | DL | Push Lever | Pull Lever | Push Bar |
|---|---|---|---|---|---|
| 1 | No Rand. | ✗ | 10.8 | 5.0 | 20.0 |
| 2 | Solid-color Rand. | ✓ | 67.5 | 65.8 | 70.0 |
| 3 | +10% Texture Rand. | ✗ | 58.3 | 50.8 | 76.7 |
| 4 | +10% Texture Rand. | ✓ | 79.2 | 77.5 | 77.5 |
| 5 | +100% Texture Rand. | ✗ | 73.3 | 55.8 | 76.7 |
| **6** | **+100% Texture Rand.** | **✓** | **85.8** | **80.8** | **85.0** |

**说明**: 用全部纹理 + dome light（实验 6）泛化最好（81–86%）。去掉 dome light 随机化掉 15–30%，对最长程、最难的「拉杆门（pull lever）」影响最大。仅 10% 纹理（实验 4）已接近 100%（仅差 4–8%）。完全不随机化（实验 1）成功率骤降到 5–20%。仅纯色材质 + dome light（实验 2，对应 Tobin2017/Zhu2018 等早期 RGB sim-to-real 设置）也能达 65.8–70%，剩余差距体现了现代高保真渲染带来的更强视觉泛化。

### Table 2: Reward components for door opening / 开门任务奖励项（教师）

Track$(x,\mu,\sigma)$ 表示高斯跟踪奖励 $\exp(-(x-\mu)^2/(2\sigma^2))$。任务分六阶段：(0) 走向门、(1) 预抓取、(2) 抓取、(3) 开门、(4) 摆门、(5) 穿过门。

| 类别 | 奖励项 | 权重 | 阶段 |
|---|---|---|---|
| **终止/通用惩罚** | Termination $\mathbb{1}_{\{\text{termination}\}}$ | −1000.0 | 0–5 |
| | Delta action rate $\|\Delta a_t\|_2^2$ | −0.01 | 0–5 |
| | DoF velocity $\|\dot{\mathbf{q}}_{\text{upper, non-finger}}\|_2^2$ | −1.0e−3 | 0–5 |
| | DoF acceleration $\|\ddot{\mathbf{q}}_{\text{upper, non-finger}}\|_2^2$ | −1.0e−5 | 0–5 |
| | DoF position limits | −5.0 | 0–5 |
| | Finger primitive limits | −1.0 | 0–5 |
| | Humanly DoF limit | −1.0 | 0–5 |
| | DoF overspeed $\sum\max(0,|\dot{\mathbf{q}}_i|-2.0)^2$ | −0.1 | 0–5 |
| | Undesired contact | −0.2 | 0–5 |
| | Door frame contact $\sum\|\mathbf{f}_{\text{door frame}}\|_2$ | −0.1 | 0–5 |
| | Door panel contact $\sum\|\mathbf{f}_{\text{door panel}}\|_2$ | −0.1 | 0–5 |
| | Upright penalty | −1.0 | 0–5 |
| | HOMIE action limit | −1.0 | 0–5 |
| **Stage 0 走向门** | Walk to door（高斯速度跟踪, $\sigma$=0.15） | 5.0 | 0 |
| | Upper body deviation $\|\mathbf{q}_{\text{upper,non-finger}}-\mathbf{q}_{\text{resting}}\|_1$ | −1.0 | 0, 5 |
| | Face door | −1.0 | 0–2, 5 |
| **Stage 1 预抓取** | Hand-handle orientation（高斯, $\sigma$=0.6） | 3.0 | 1–4 |
| | Pregrasp finger pose（track 位置+速度） | 1.5 | 0–1, 5 |
| | Unused arm deviation | −1.0 | 1–4 |
| | Pre-grasp target distance（track 距离+速度） | 6.0 | 1 |
| | Penalty not standing still $\|\mathbf{u}_{\text{HOMIE},[0:3]}\|_2$ | −15.0 | 1–3 |
| **Stage 2 抓取** | Grasp finger DoF pose（track） | 3.0 | 2–4 |
| | Grasp target distance（高斯, $\sigma$=0.1） | 3.0 | 2–4 |
| | Grasp force $\sum(-|\mathbf{f}_{\text{palm},y,z}|+f_{\text{palm},x})$ | 0.2 | 1–4 |
| **Stage 3 开门** | Push door handle $\dot\theta_{\text{handle}}+\operatorname{clip}(\theta_{\text{handle}},0,45^\circ)/45^\circ$ | 6.0 | 3 |
| | Push door hinge $10\dot\theta_{\text{hinge}}+\operatorname{clip}(\theta_{\text{hinge}},0,90^\circ)/90^\circ$ | 6.0 | 3–4 |
| | Push door force $\operatorname{clip}(\mathbf{f}_{\text{hand},x},0,20)$ | 0.3 | 3 |
| **Stage 4 摆门 & Stage 5 穿门** | Don't push door handle $-\dot\theta_{\text{handle}}+(45^\circ-\theta_{\text{handle}})/45^\circ$ | 3.0 | 4–5 |
| | Target root distance（track 速度+距离） | 12.0 | 4–5 |
| | Penalty standing still（高斯, $\sigma$=0.05） | −1.0 | 4 |
| **始终启用** | Stage progress $\text{stage}_{\text{current}}$ | 1.0 | 0–5 |
| | Task completion $\mathbb{1}_{\{\text{complete}\}}$ | 4.0 | 0–5 |
| | Success save time $\mathbb{1}_{\{\text{success}\}}\cdot$ 剩余时间比 | 0.5 | 0–5 |

**说明**: 分阶段奖励把长程开门拆成六个原子阶段，每阶段独立塑形（含高斯跟踪 + 大量物理安全惩罚）。注意「Success save time」鼓励更快完成——这正是策略能在流畅度上超越人类遥操作的奖励来源之一。

### Table 3: Physical property randomization range / 门的物理属性随机化范围（IsaacLab）

| Property | Range | Unit |
|---|---|---|
| Panel Width | 0.8–1.1 | m |
| Panel Height | 1.9–2.2 | m |
| Handle Height | 0.85–0.95 | m |
| Handle to Edge Distance | 0.04–0.1 | m |
| Handle Type | {knob, lever, pushbar, handle, flat} | — |
| Open Handedness | {left, right} | — |
| Open Direction | {in, out} | — |
| Weight | 80–120 | kg |
| Hinge Max Force | 20–30 | Nm |
| Hinge Damping | 5–10 | (kg·m²)/(s²·°) |
| Hinge Stiffness | 10–20 | (kg·m²)/(s²·°) |
| Handle Max Force | 1–3 | Nm |
| Handle Damping | 0.1–0.6 | (kg·m²)/(s²·°) |
| Handle Stiffness | 30–50 | (kg·m²)/(s²·°) |

**说明**: 程序化造门的物理随机化范围。闩锁被建模为依附把手关节角的 mimic joint；把手执行器设 −5°（向上）目标位以模拟弹簧加载的张力。这套大范围物理随机化是零样本物理 sim-to-real 的基础。

---

## 实验

### 数据集 / 评测设置

| 设置 | 规模/构成 | 特点 | 用途 |
|------|------|------|------|
| 仿真（IsaacLab） | 程序化海量门，每并行环境一套独特参数 | 物理+视觉大规模随机化；评测用 holdout 纹理 | 训练/测试 |
| 真实世界 | 多样真实门（推/拉、杠杆/推杠） | **训练中完全未见** | 零样本测试 |
| 三类门任务 | Push Lever / Pull Lever / Push Bar | 分别为最简单 / 受限空间长程 / 需克服弹簧的强力交互 | 训练+评测 |
| 视觉随机化消融 | 6 个配置 ×120 未见门试验 | 见 Table 1 | 测试 |

**评测协议**: 机器人随机放在门前 1 米、面向门中心，yaw 扰动 ±0.3 rad；当机器人穿过门并到达门框另一侧 1 米处判定成功，同时记录完成时间。

### 实现细节

- **机器人**: 29-DoF Unitree G1 + 两只 7-DoF 三指灵巧手；感知用 Intel RealSense D435i（**不用深度输出**）。
- **底层控制**: 预训练全身控制器 [[HOMIE]]，策略以 50 Hz 输出目标关节角，由 PD 跟踪。
- **教师**: [[PPO]] + [[Staged-Reset Exploration|分阶段重置]]（buffer=100），特权观测。
- **学生**: [[ResNet]] 视觉编码器 → 两层 LSTM(512) → 三层 MLP(512,256,128)；[[DAgger]] 蒸馏；再 [[GRPO]] 微调。
- **仿真**: IsaacLab，RTX 实时渲染（performance 模式），5233 张 dome-light 贴图 + PBR 材质随机化。
- **推理硬件**: Intel i9-14900K + NVIDIA RTX 4090 桌面工作站。
- **遥操作基线**: PICO 4 Ultra 头显 + 双手柄，输出 3 个上半身 SE(3) 位姿 + 手指角 + 腰高 + 平面导航命令；用 Pinocchio 解 IK。专家 = 全职 3 个月以上经验；非专家 = 不到 1 天经验。

### 关键实验结论

- **超越人类遥操作（Fig 5）**: 真实成功率 83%（专家 80% / 非专家 60%）；完成时间比专家快 23.8%、比非专家快 31.7%。定性观察：遥操作者常误判弹簧门的反力、不会用合适身位维持平稳开门速度、难以跟踪门的旋转轨迹——这类反馈信息超出当前 VR 遥操作能力，但能在仿真中交互学到。
- **视觉随机化（Table 1）**: 全纹理 + dome light 最优（81–86%）；去 dome light 掉 15–30%；10% 纹理已接近全量；无随机化骤降到 5–20%；纯色 + dome light 仍达 65.8–70%。
- **GRPO 微调（Fig 6a）**: 学生从 50–70% 提升到 80.8–85.8%，曲线趋近教师上界，证明有效弥合部分可观测性 gap。
- **分阶段重置（Fig 6b）**: buffer=100 约 1700 迭代达全部阶段；buffer=10 需 4000+；buffer=0 直接探索失败（卡在 Stage 2 抓把手）。

---

## 批判性思考

### 优点
1. **真正的纯 RGB 全身 sim-to-real**: 去掉深度/位姿先验与硬编码基元，零样本迁移到未见真实门，且首次在人形 + 关节物体 loco-manipulation 上做到，难度与完成度都高。
2. **两个机制都有干净的消融支撑**: 分阶段重置（Fig 6b）与 GRPO 自举（Fig 6a）各自有对照实验，且 GRPO 阶段被论证为「即插即用」可改进任何非零成功率任务。
3. **直接对标人类遥操作并取胜**: 在**同一全身控制器**下比较，控制了变量，结论（自主策略更快更稳）有说服力，并定性解释了遥操作的物理感知短板。
4. **可扩展数据管线**: IsaacLab 原生程序化造门 + 5233 dome light + 真实闩锁建模，物理与视觉多样性规模大且面向并行 RL。

### 局限性
1. **任务单一**: 仅以「开门」为代表，虽涵盖三类门，但抽屉、旋钮、柜门、阀门等更广的关节物体未实证，"diverse loco-manipulation" 的普适性仍待验证。
2. **重度奖励工程**: Table 2 含 30+ 条精细塑形奖励与六阶段人类先验划分，复现与迁移成本高；作者也在结论里把「减少对任务特定奖励工程的依赖」列为未来方向。
3. **样本与基线有限**: 真实评测每子任务样本量、绝对失败模式分析较少；视觉消融在 120 次未见门试验上，统计稳健性可更强。
4. **无公开代码/权重信息**: 论文未给出项目主页或开源承诺，复现门槛高（依赖 IsaacLab + 特定 G1 + HOMIE 控制器栈）。

### 潜在改进方向
1. 把六阶段人类先验与分阶段重置推广到更通用的关节物体（抽屉/旋钮/阀门），验证管线的任务无关性。
2. 用高容量 BC 教师或语言条件减少手工奖励 shaping（作者亦提及）。
3. 引入更系统的真实世界统计评测（更多 trial、失败归因、跨光照/材质的细粒度泛化曲线）。
4. 探索 GRPO 自举在其他全身 loco-manipulation 任务上的「drop-in」可迁移性，量化其对部分可观测性的普适收益。

### 可复现性评估
- [ ] 代码开源（论文未提供链接/承诺）
- [ ] 预训练模型（未提及）
- [x] 训练细节较完整（奖励表 Table 2、物理随机化 Table 3、网络结构、超参分阶段给出）
- [ ] 数据集可获取（程序化造门管线为自研，依赖 IsaacLab + 私有资产，未声明发布）

---

## 速查卡片

> [!summary] DoorMan: RGB 人形 loco-manipulation 的 Sim-to-Real
> - **核心**: 教师-学生-自举三阶段，靠 IsaacLab 大规模物理+视觉随机化，训出纯 RGB 人形开门策略并零样本迁移真实世界。
> - **方法**: PPO 特权教师（+ 分阶段重置探索，buffer=100）→ DAgger 蒸馏 RGB 学生（ResNet+LSTM+MLP）→ GRPO + 二值成功信号自举微调，缓解部分可观测性。
> - **结果**: 真实成功率 83%（>专家 80% / 非专家 60%），完成时间比人类快 23.1%–31.7%；视觉随机化全开达 81–86%。
> - **平台**: Unitree G1（29-DoF + 双 7-DoF 灵巧手）、RealSense D435i（不用深度）、50 Hz、HOMIE 全身控制器。

---

*笔记创建时间: 2026-06-29*
