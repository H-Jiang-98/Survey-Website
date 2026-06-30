---
title: "Devil is in Narrow Policy: Unleashing Exploration in Driving VLA Models"
method_name: "Curious-VLA"
authors: [Canyu Chen, Yuguang Yang, Zhewen Tan, Yizhi Wang, Ruiyi Zhan, Haiyan Liu, Xuanyao Mao, Jason Bao, Xinyue Tang, Linlin Yang, Bingchuan Sun, Yan Wang, Baochang Zhang]
year: 2026
venue: CVPR
tags: [VLA, autonomous-driving, reinforcement-learning, GRPO, imitation-learning, exploration, narrow-policy]
zotero_collection: _inbox
image_source: online
arxiv_html: https://arxiv.org/html/2603.06049v1
created: 2026-06-29
---

# Devil is in Narrow Policy: Unleashing Exploration in Driving VLA Models

## 元信息

| 项目 | 内容 |
|------|------|
| 作者 | Canyu Chen, Yuguang Yang, Zhewen Tan, Yizhi Wang, Ruiyi Zhan 等 13 人 |
| 机构 | 北京航空航天大学（国家卓越工程师学院 / 电子信息工程学院 / 计算机 / 网络空间 / 人工智能学院）、清华大学智能产业研究院 (AIR)、联想集团、中国传媒大学 |
| 会议 | CVPR 2026 |
| 类别 | VLA（Vision-Language-Action）/ 自动驾驶 |
| 日期 | 2026-03（arXiv v1） |
| 项目主页 | https://github.com/Mashiroln/curious_vla.git |
| 链接 | [arXiv](https://arxiv.org/abs/2603.06049) / [Code](https://github.com/Mashiroln/curious_vla.git) |

---

## 一句话总结

> 揭示驾驶 VLA 的"窄策略 (Narrow Policy)"问题——模仿学习把探索压缩成单一模式、导致后续 GRPO 强化学习因奖励无方差而提前饱和——并用 IL 端的可行轨迹扩展 + RL 端的多样性感知采样与跨度驾驶奖励双管齐下，在 Navsim 上刷新 SOTA。

---

## 核心贡献

1. **首次定义并理论分析"窄策略 (Narrow Policy)"问题**: 指出 [[Imitation Learning|模仿学习]]（SFT）过度拟合真值轨迹会让策略坍缩成单模式（低多样性、低质量），进而使 [[GRPO]] 这类无评论家 RL 因组内奖励方差趋零、优势 $A_i\to 0$ 而梯度消失、过早饱和。配套提出三项**行为诊断 (Behavioral Diagnostics)** 指标（Diversity / Quality / Performance）量化该现象。
2. **IL 端 Feasible Trajectory Expansion (FTE)**: 用扩散模型生成多条**物理可行**轨迹做数据扩展 (DE)、四阶段 [[Chain-of-Thought|CoT]] 推理合成、以及**逐步归一化 (Step-wise Normalization)** 解决远近时域物理尺度失衡，把多样数据真正变成可学习的知识。
3. **RL 端 Diversity-Aware RL**: 提出 **Adaptive Diversity-Aware Sampling (ADAS)** 用伯努利试验统计筛选高多样性场景以保住奖励方差，并提出 **Spanning Driving Reward (SDR)** 用 focal 风格加权放大奖励值跨度、提升对驾驶质量的敏感度。Navsim v1 PDMS 90.3、v2 EPDMS 85.3、Best-of-N PDMS 94.8（达人类 GT 水平 94.8）。

---

## 问题背景

### 要解决的问题
驾驶 [[VLA]] 普遍采用 **先 IL（SFT）后 RL（GRPO）** 的两阶段训练管线。本文指出该管线存在一个被长期忽视的**根本缺陷——窄策略 (Narrow Policy, NP)**：IL 阶段过度利用 (over-exploit) 真值轨迹，导致探索坍缩；探索一旦坍缩，后续 RL 因反馈多样性不足而过早饱和，难以真正提升性能。如何打破这个"利用-探索 (exploit-explore)"困境是核心问题。

### 现有方法的局限
作者把当前驾驶 VLA 分为两类，二者都受 NP 困扰：
- **VLA-Planner**（如 [[ReCogDrive]]、ORION、ImagiDrive、DriveVLA-W0）: VLM 作语义推理器、外接一个轨迹规划模块预测未来运动分布。
- **VLA-Token**（如 [[AutoVLA]]、AdaThinkDrive、EMMA、Poutine）: VLM 解码器直接输出离散动作 token 或文本 waypoint。

作者在 Navsim navtrain 上对两个代表基线 Qwen2.5-VL（VLA-Token）与 ReCogDrive（VLA-Planner）做诊断（Figure 1）：每个模型独立推理 $k=8$ 次采样 8 条轨迹，结果**多样性极低**（mean-pFDE 仅 0.20m / 0.33m）、**质量受限**（min-FDE 1.05m），采样轨迹塌缩到单一模式甚至出现不安全行为。

### 本文的动机
NP 的根源被拆成三处可分析的失配（见 3.2）：
1. **优化目标失配**: [[Cross Entropy Loss|交叉熵损失]]把所有非真值 token 视作"等价错误"，缺乏轨迹 token 间的空间/物理邻近性度量，鼓励对单一专家模式过度自信；
2. **时域物理尺度失配**: ego-centric 坐标下远时域（$t=4$s）方差比近时域（$t=0.5$s）大几个数量级，远时域损失主导 SFT，近时域（决定转向精度）几乎无贡献；
3. **RL 中的优势坍缩**: 策略坍缩 → 组内奖励近乎相同 → $\sigma_R\to0$ → $A_i\to0$ → GRPO 梯度消失。

因此必须**在 IL 与 RL 两端同时注入多样性**：IL 端造可行且多样的数据并修正尺度失衡，RL 端筛选高多样性场景并放大奖励跨度。

---

## 方法详解

### 模型架构

Curious-VLA 是一个**纯 VLM**，从 [[Qwen2.5-VL]]-3B 初始化，直接自回归输出**文本 waypoint** 轨迹，**不外接任何规划模块**（属 VLA-Token 范式）。整体管线（见 Figure 2）由两阶段构成：

- **输入**: 多模态观测 $\mathcal{X}$ = 多视角相机图像 $\mathcal{C}$（本文仅用 CAMERA-FRONT 单前视）+ 文本指令 $\mathcal{I}$（如 "turn left"）+ ego 状态 $\mathcal{S}$（速度/加速度/历史控制）
- **策略**: 统一生成式策略 $\pi_\theta$，把 $\mathcal{X}$ 映射为长度 $T$ 的动作序列 $\tau=\{w_1,\dots,w_T\}$
- **IL 阶段**: 标准 SFT + **Feasible Trajectory Expansion (FTE)** = 数据扩展 (DE) + 四阶段 CoT 合成 + 逐步归一化 (SN)
- **RL 阶段**: GRPO + **Adaptive Diversity-Aware Sampling (ADAS)** + **Spanning Driving Reward (SDR)**
- **基座**: Qwen2.5-VL-3B（比同类 7B/8B 更小更早）

### 核心模块

#### 模块1: Feasible Trajectory Expansion（IL 阶段可行轨迹扩展）

**设计动机**: 直击 NP 在 IL 端的根源——既要造出**多样**轨迹打破单模式，又要保证它们**物理可行/安全**，还要修正时域尺度失衡使多样数据可学。

**具体实现**（三个子模块）:
- **Exploratory Data Expansion (DE)**: 先用 [[Qwen2.5-VL]]-72B 从 103k NavTrain 中筛出 **12k 个挑战性片段**（多车道、路口、遮挡）；再用扩散式 [[ReCogDrive]] **扰动扩散隐变量**生成多样轨迹；全部候选用 **PDMS scorer 过滤**保证安全。扩展同时覆盖 **within-intent**（同一目标周围采样）与 **across-intent**（改变路线级决策），最终得到 **142k 安全且多样**的样本。
- **Chain-of-Thought 合成 (CoT)**: 把驾驶推理结构化为单轮对话的**四阶段链**：(i) 关键物体感知 → (ii) 驾驶解释 → (iii) 元行为描述 → (iv) 轨迹预测；用 Qwen2.5-VL-72B 自动为整个扩展数据集生成结构化推理序列。
- **Step-wise Normalization (SN)**: 对每个预测步 $t$ **独立归一化**（见公式7），均衡各时域梯度幅度，从而提升轨迹模式的可分性——是把多样数据"转化为可学习知识"的关键催化剂。

#### 模块2: Adaptive Diversity-Aware Sampling（RL 阶段多样性感知采样）

**设计动机**: GRPO 需要组内奖励有方差才能产生非零优势/有效梯度。ADAS 动态挑选"在随机策略下能产生多样 rollout"的场景，维持足够奖励方差以稳定 GRPO（对应 3.2 第 3 点的优势坍缩）。

**具体实现**:
- 把每个场景的结果变异建模为**简化伯努利过程**：每次 rollout 是成功（高 PDMS）/失败（低 PDMS）的二元试验，概率 $p$，刻画奖励分布的最极端情况。
- 每个外循环开始时从全量训练数据**重采样**一个新的活动训练集 $\mathcal{D}_{active}$；对每个场景 $x$ 周期性做 $M$ 次离线 rollout（$M\gg G$）估计经验奖励分布，平均归一化 PDMS 作为成功概率估计 $\hat p$。
- 仅当场景同时满足两个多样性条件（见公式9-10）才纳入 $\mathcal{D}_{active}$：第一项限制 $G$ 次在线 rollout 全部同结果（全成功/全失败）的概率，保证样本间变异；第二项要求经验标准差 $\sigma_R$ 与理论伯努利方差一致（置信裕度 $\epsilon_{conf}$ 内），过滤掉不稳定/噪声场景。
- 算法见 **Algorithm 1**。

#### 模块3: Spanning Driving Reward（RL 阶段跨度驾驶奖励）

**设计动机**: 原始 Navsim PDMS/EPDMS 奖励值跨度小，难以区分次优与最优行为。SDR 用 focal 风格加权**放大奖励的值跨度**，提升对驾驶质量的敏感度，进一步放大探索信号。

**具体实现**:
- 原始指标（公式11）为安全约束 $C$ 与加权目标 $M$ 的乘积：$C=\{\text{NC, DAC}\}$（无碰撞、可行驶区域合规），$M=\{\text{EP, TTC, C}\}$（ego 进度、碰撞时间、舒适），权重 $w_m=\{5,5,2\}$。
- 重构为 focal-style 跨度目标（公式12），用 $1-(1-m)^{\gamma_m}$ 替换线性项 $m$，$\gamma_m$ 为超参，放大次优与最优之间的差异。
- EPDMS 复用同结构：约束扩展 $\{\text{DDC, TLC}\}$（行驶方向、红绿灯合规），目标扩展 $\{\text{LK, EC}\}$（车道保持、两帧扩展舒适），额外权重 $\{2,2\}$。

### 关键公式与机制

#### 公式1: [[Imitation Learning|SFT 模仿学习损失]]

$$
\mathcal{L}_{\text{SFT}}(\theta)=-\mathop{\mathbb{E}}_{(\mathcal{X},\mathbf{y}^{*})\sim\mathcal{D}}\left[\frac{1}{L}\sum_{t=1}^{L}\log\pi_{\theta}(y_{t}^{*}\mid\mathbf{y}^{*}_{<t},\mathcal{X})\right]
$$

**含义**: 在 VLA-Token 范式下，用交叉熵最大化生成的文本轨迹 token $\mathbf{y}^*$ 的似然，建立基础规划能力。

**符号说明**:
- $\mathcal{X}$: 多模态输入；$\mathbf{y}^*$: 真值轨迹 token 序列；$L$: token 长度
- $\pi_\theta$: VLA 生成式策略；$\mathbf{y}^*_{<t}$: 因果上下文

#### 公式2: [[GRPO]] 训练目标

$$
\mathcal{J}_{\text{GRPO}}(\theta)=\mathop{\mathbb{E}}_{\mathcal{X},\{\mathbf{y}_{i}\}\sim\pi_{\theta_{\text{old}}}}\bigg[\frac{1}{G}\sum_{i=1}^{G}\Big(\hat{\mathcal{J}}_{i}(\theta)-\beta\,\mathbb{D}_{\text{KL}}(\pi_{\theta}\,\|\,\pi_{\text{sft}})\Big)\bigg]
$$

**含义**: 对每个输入 $\mathcal{X}$ 从旧策略采样 $G$ 个输出 $\{\mathbf{y}_i\}$，组内归一化优势，结合裁剪损失与对 SFT 策略的 KL 约束做策略更新（无需价值网络）。

**符号说明**:
- $G$: 组大小（rollout 数）；$\beta$: KL 系数；$\pi_{\text{sft}}$: SFT 初始策略
- $\hat{\mathcal{J}}_i(\theta)$: 第 $i$ 个样本的裁剪优势项（公式3）

#### 公式3: 裁剪优势项

$$
\hat{\mathcal{J}}_{i}(\theta)=\min\!\left(\rho_{i}(\theta)A_{i},\ \text{clip}\big(\rho_{i}(\theta),1-\epsilon,1+\epsilon\big)A_{i}\right),\quad \rho_{i}(\theta)=\frac{\pi_{\theta}(\mathbf{y}_{i}\mid\mathcal{X})}{\pi_{\theta_{\text{old}}}(\mathbf{y}_{i}\mid\mathcal{X})}
$$

**含义**: PPO 式裁剪，限制似然比 $\rho_i$ 偏离 1 的幅度以稳定更新。

**符号说明**:
- $\rho_i(\theta)$: 新旧策略似然比；$\epsilon$: 裁剪范围；$A_i$: 标准化优势

#### 公式4: 组内标准化优势

$$
A_{i}=\frac{R(\mathbf{y}_{i})-\mu_{R}}{\sigma_{R}+\xi}
$$

**含义**: 用组内奖励的均值/标准差把奖励标准化为优势，是 GRPO 去价值网络的关键。

**符号说明**:
- $R(\mathbf{y}_i)$: 第 $i$ 个样本的奖励；$\mu_R,\sigma_R$: 组内奖励均值/标准差；$\xi$: 数值稳定项

#### 公式5: 交叉熵梯度（NP 的优化目标失配）

$$
\frac{\partial\mathcal{L}_{\text{SFT}}}{\partial z_{t}}=\pi_{\theta}(y_{k}\mid\mathbf{y}^{*}_{<t},\mathcal{X})-\mathbb{I}(\hat{y}_{t}=y_{t}^{*})
$$

**含义**: 揭示 CE 损失的梯度只看"是否等于真值 token"，对"接近正确"（如 31.4）与"明显错误"（如 21.4）不做区分，因而鼓励对单一专家模式过度自信、压缩多样性。

**符号说明**:
- $z_t$: 输出 logit；$\hat y_t$: 预测 token；$y_t^*$: 真值 token；$\mathbb{I}(\cdot)$: 指示函数

#### 公式6: 优势坍缩（NP 的 RL 失配）

$$
\lim_{\sigma_{R}\to 0}A_{i}=\frac{R(\mathbf{y}_{i})-\mu_{R}}{\sigma_{R}+\xi}\to 0
$$

**含义**: 当策略坍缩到单模式、$R(\mathbf{y}_i)\approx\mu_R$ 时 $\sigma_R\to0$，优势趋零，GRPO 梯度消失——这正是 RL 过早饱和的数学根源。

#### 公式7: Step-wise Normalization（逐步归一化）

$$
\tilde{w}_{t}=\frac{w_{t}-\mu_{t}}{\sigma_{t}},\qquad \hat{w}_{t}=\hat{\tilde{w}}_{t}\,\sigma_{t}+\mu_{t}
$$

**含义**: 对每个时域步 $t$ 用训练集逐步统计量独立归一化（训练用 $\tilde w_t$），测试时反归一化还原 $\hat w_t$。均衡各时域梯度幅度，解决远近时域物理尺度失配。

**符号说明**:
- $w_t$: 第 $t$ 步 waypoint；$(\mu_t,\sigma_t)$: 训练集第 $t$ 步统计量

#### 公式8: ADAS 入选场景的多样性条件（散度项）

$$
\hat{p}^{G}+(1-\hat{p})^{G}<\epsilon_{\text{div}}
$$

**含义**: 限制 $G$ 次在线 rollout 全部成功或全部失败的概率上界，确保该场景能产生足够变异的结果（非全 0 优势）。

**符号说明**:
- $\hat p$: 离线估计的成功概率；$G$: 组大小；$\epsilon_{\text{div}}$: 散度阈值

#### 公式9: ADAS 入选场景的一致性条件（置信项）

$$
\big|\sigma_{R}-\sqrt{\hat{p}(1-\hat{p})}\,R_{\text{range}}\big|<\epsilon_{\text{conf}}
$$

**含义**: 要求经验奖励标准差 $\sigma_R$ 与理论伯努利方差一致，过滤不稳定/噪声场景。满足两项后将 $x$ 加入活动集 $\mathcal{D}_{active}\leftarrow\mathcal{D}_{active}\cup\{x\}$。

**符号说明**:
- $R_{\text{range}}$: 奖励取值范围；$\epsilon_{\text{conf}}$: 置信裕度

#### 公式10: 原始 PDMS 奖励

$$
\text{PDMS}=\prod_{c\in C}c\times\frac{\sum_{m\in M}w_{m}\cdot m}{\sum_{m\in M}w_{m}}
$$

**含义**: Navsim 指标 = 安全约束乘积 × 加权目标的归一化加权和。

**符号说明**:
- $C=\{\text{NC, DAC}\}$: 硬约束（任一为 0 则整体为 0）；$M=\{\text{EP, TTC, C}\}$: 软目标；$w_m=\{5,5,2\}$

#### 公式11: Spanning Driving Reward（跨度驱动奖励）

$$
R_{\text{span}}=\prod_{c\in C}c\cdot\frac{\sum_{m\in M}w^{\prime}_{m}\cdot\big(1-(1-m)^{\gamma_{m}}\big)}{\sum_{m\in M}w^{\prime}_{m}}
$$

**含义**: 用 focal 风格 $1-(1-m)^{\gamma_m}$ 替换线性目标项，放大次优/最优行为之间的奖励差异，提升对驾驶质量的敏感度并强化探索信号。

**符号说明**:
- $\gamma_m$: 第 $m$ 个目标的 focal 超参；$w'_m$: 重加权权重

---

## 关键图表

<!-- 图片均使用 arXiv HTML 在线链接，未本地化 -->

### Figure 1: Visualization of Narrow Policy / 窄策略可视化

![Figure 1a Behavioral Diagnostics](https://arxiv.org/html/2603.06049v1/x1.png)
![Figure 1b Multi-infer Trajectories](https://arxiv.org/html/2603.06049v1/x2.png)

**说明**: (a) 行为诊断定量对比：基线多样性极低（mean-pFDE：Qwen2.5-VL 0.20m、ReCogDrive 0.33m）、质量受限（min-FDE：Qwen2.5-VL 1.05m），而 Curious-VLA 在多样性、质量、性能上全优。(b) Qwen2.5-VL（$k=8$ 多次推理）轨迹塌缩成单模式甚至不安全行为，Curious-VLA 则保持多模式。这是全文"窄策略"问题的核心证据图。

### Figure 2: Overall Pipeline of Curious-VLA / 整体管线

![Figure 2](https://arxiv.org/html/2603.06049v1/x3.png)

**说明**: Curious-VLA 整体架构。中间面板示意现有 IL-RL 管线产生重叠/不安全行为的"窄策略"；左面板为 IL 端 FTE（多样轨迹生成 + 四阶段 CoT + 逐步归一化），右面板为 RL 端多样性感知（ADAS 自适应采样 + SDR 跨度奖励）。展示了方法如何在 IL 与 RL 两端同时缓解窄策略。

### Figure 3: Horizon Physical Scale Mismatch / 时域物理尺度失配

![Figure 3](https://arxiv.org/html/2603.06049v1/x4.png)

**说明**: 用 waypoints 分布可视化时域物理尺度失配。远时域（$t=4$s）坐标方差比近时域（$t=0.5$s）大几个数量级，导致远时域损失主导 SFT；逐步归一化 (SN) 后各时域梯度幅度被均衡，轨迹模式可分性提升。直接佐证 3.2(2) 与公式7。

### Figure 4: Qualitative Comparison (BEV + Camera) / 定性对比

![Figure 4](https://arxiv.org/html/2603.06049v1/x5.png)

**说明**: 与基线在 BEV 与相机视角下的定性对比，Curious-VLA 产生更可行 (feasible) 的轨迹。

### Figure 6: RL Training Curves / RL 训练曲线

![Figure 6](https://arxiv.org/html/2603.06049v1/x6.png)

**说明**: ADAS(3x) 下 130 步的验证奖励 (Val Reward) 与测试 PDMS 曲线。Random Sample 基线出现 **RL 坍缩**（奖励停滞/下降），而 ADAS 持续上升，直观说明多样性感知采样对维持 RL 学习信号的必要性。

### Figure 7: Stability Analysis / 稳定性分析

![Figure 7](https://arxiv.org/html/2603.06049v1/x7.png)

**说明**: ADAS(1x) 下 $k=4$ 次独立训练的 Critic 与 Val Reward 曲线，呈现一致提升且**低方差**，说明方法训练稳定、可复现。

### Figure 8: More Visualization (Curious-VLA vs Qwen2.5-VL) / 更多可视化（附录）

![Figure 8a](https://arxiv.org/html/2603.06049v1/figures/sv1.png)
![Figure 8b](https://arxiv.org/html/2603.06049v1/figures/sv2.png)
![Figure 8c](https://arxiv.org/html/2603.06049v1/figures/sv3.png)

**说明**: 上排 Curious-VLA、下排 Qwen2.5-VL 的更多场景轨迹对比，补充展示前者多样且安全、后者塌缩或不安全。

### Figure 5 (附录): Prompt Template for Scenario Filtering / 场景筛选提示模板

**说明**: 用于筛选"具备多个可行驾驶意图的语义挑战场景"的提示模板与代表性回答示例（即 DE 中 Qwen2.5-VL-72B 的筛选 prompt，无独立图片资源，文中以排版块呈现）。

### Table 1: Navsim V1 Benchmark / Navsim v1 开环结果

| Method | Base Model | Sensors | Trajectory | NC↑ | DAC↑ | EP↑ | TTC↑ | C↑ | PDMS↑ |
|--------|-----------|---------|-----------|-----|------|-----|------|----|-------|
| Human GT [10] | - | - | - | 100.0 | 100.0 | 87.5 | 100.0 | 99.9 | 94.8 |
| Ego-MLP [29] | - | 6×C | Continuous | 93.1 | 78.3 | 63.2 | 84.0 | 100.0 | 66.4 |
| UniAD [16] | - | 3×C+L | Continuous | 97.7 | 92.8 | 79.2 | 92.8 | 100.0 | 84.0 |
| DiffusionDrive [30] | - | 3×C+L | Continuous | 98.2 | 96.2 | 82.2 | 94.7 | 100.0 | 88.1 |
| WoTE [26] | - | 3×C+L | Continuous | 98.5 | 96.8 | 81.9 | 94.4 | 99.9 | 88.3 |
| ImagiDrive-A [23] | InternVL2.5-4B | 1×C | Continuous | 98.1 | 96.2 | 80.1 | 94.4 | 100.0 | 86.9 |
| ImagiDrive-S | InternVL2.5-4B | 1×C | Continuous | 98.6 | 96.2 | 80.5 | 94.5 | 100.0 | 87.4 |
| ReCogDrive [27] | InternVL2-8B | 3×C | Continuous | 98.2 | 97.8 | 83.5 | 95.2 | 100.0 | 89.6 |
| DriveVLA-W0 [25] | Emu-3-8B | 1×C | Continuous | 98.7 | 99.1 | 93.3 | 95.3 | 99.3 | 90.2 |
| DriveVLA-W0† | Emu-3-8B | 1×C | Continuous | 99.3 | 97.4 | 88.3 | 97.0 | 99.9 | 93.0 |
| Qwen2.5-VL [27] | Qwen2.5-VL-7B | 1×C | Text Waypoint | 97.8 | 92.1 | 78.3 | 92.8 | 100.0 | 83.3 |
| InternVL2 [27] | InternVL2-8B | 1×C | Text Waypoint | 97.0 | 92.4 | 78.9 | 91.8 | 100.0 | 83.3 |
| AutoVLA [54] | Qwen2.5-VL-3B | 3×C | Discrete Action | 98.4 | 95.6 | 81.9 | 98.0 | 99.9 | 89.1 |
| AutoVLA† | Qwen2.5-VL-3B | 3×C | Discrete Action | 99.1 | 98.8 | 87.9 | 97.2 | 100.0 | 92.1 |
| AdaThinkDrive [33] | InternVL3-8B | 1×C | Text Waypoint | 98.4 | 97.8 | 84.4 | 95.2 | 100.0 | 90.3 |
| AdaThinkDrive† | InternVL3-8B | 1×C | Text Waypoint | 99.1 | 98.8 | 87.9 | 95.2 | 100.0 | 93.0 |
| **Curious-VLA** | **Qwen2.5-VL-3B** | **1×C** | **Text Waypoint** | 98.4 | 96.9 | **88.5** | 97.9 | 98.1 | **90.3** |
| **Curious-VLA†** | **Qwen2.5-VL-3B** | **1×C** | **Text Waypoint** | **99.5** | **99.0** | **91.8** | **99.3** | 98.4 | **94.8** |

**说明**: 单前视相机输入下 Curious-VLA PDMS 90.3 创 SOTA，超过同基座 AutoVLA（89.1，且后者用 3×相机）+1.2，与更大模型 DriveVLA-W0(90.2)/AdaThinkDrive(90.3) 相当；尤其 EP(88.5) 与 TTC(97.9) 同步提升（Comfort 略降）。Best-of-N（N=6）下 Curious-VLA† 达 **94.8**，超 AdaThinkDrive†(93.0) +1.8，**追平人类 GT 水平 (94.8)**——多样且正确的轨迹能力的直接证据。

### Table 2: Navsim V2 (Extended PDMS) / Navsim v2 开环结果

| Method | NC↑ | DAC↑ | DDC↑ | TLC↑ | EP↑ | TTC↑ | LK↑ | C↑ | EC↑ | EPDMS↑ |
|--------|-----|------|------|------|-----|------|-----|----|-----|--------|
| Ego-MLP [29] | 93.1 | 77.9 | 92.7 | 99.6 | 86.0 | 91.5 | 89.4 | 98.3 | 85.4 | 64.0 |
| VADv2 [20] | 97.3 | 91.7 | 98.2 | 99.9 | 77.6 | 92.7 | 66.0 | 100.0 | 97.4 | 76.6 |
| TransFuser [5] | 96.9 | 89.9 | 97.8 | 99.7 | 87.1 | 95.4 | 92.7 | 98.3 | 87.2 | 76.7 |
| HydraMDP++ [28] | 97.2 | 97.5 | 99.4 | 99.6 | 83.1 | 96.5 | 94.4 | 98.2 | 70.9 | 81.4 |
| DriveSuprim [47] | 97.5 | 96.5 | 99.4 | 99.6 | 88.4 | 96.6 | 95.5 | 98.3 | 77.0 | 83.1 |
| ARTEMIS [12] | 98.3 | 95.1 | 98.6 | 99.8 | 81.5 | 97.4 | 96.5 | 98.3 | – | 83.1 |
| ReCogDrive [27] | 98.3 | 95.2 | 99.5 | 99.8 | 87.1 | 97.5 | 96.6 | 98.3 | 86.5 | 83.6 |
| DiffusionDrive [30] | 98.2 | 95.9 | 99.4 | 99.8 | 87.5 | 97.3 | 96.8 | 98.3 | 87.7 | 84.5 |
| **Curious-VLA** | 98.4 | **96.9** | 99.2 | 99.8 | **88.5** | **97.9** | **96.9** | 98.1 | 81.5 | **85.3** |

**说明**: navtest split 上 EPDMS 综合 85.3 创 SOTA，较 DiffusionDrive(84.5) +0.8，主要由 DAC、EP、TTC 三项支撑（EC 偏低）。（注：摘要写 85.4，正文/表为 85.3，存在小不一致。）

### Table 3: nuScenes Benchmark / nuScenes 开环结果

| Method | ST-P3 L2(m)↓ | ST-P3 Coll.(%)↓ | UniAD L2(m)↓ | UniAD Coll.(%)↓ |
|--------|--------------|-----------------|--------------|------------------|
| ST-P3 [15] | 2.11 | 0.71 | – | – |
| VAD [20] | 0.37 | 0.14 | – | – |
| UniAD [16] | 0.69 | 0.12 | 1.03 | 0.31 |
| EMMA [18] | 0.32 | – | – | – |
| OpenEMMA [45] | 2.81 | – | – | – |
| OpenDriveVLA [53] | 0.33 | 0.10 | 0.67 | 0.30 |
| AutoVLA [54] | 0.48 | 0.13 | 0.86 | 0.35 |
| Impromptu VLA [4] | 0.33 | 0.13 | 0.67 | 0.38 |
| **Curious-VLA** | **0.31** | **0.10** | **0.60** | 0.33 |

**说明**: 用 28k nuScenes 子集 + ADE-based 奖励做 RL，验证真实世界泛化：L2(3s) 与碰撞率均优于现有 VLA 与 E2E（ST-P3 L2 0.31、UniAD L2 0.60 最低）。

### Table 4: Exploration Analysis (@k=8) / 探索分析

| Method | Stage | Quality (minADE/FDE↓) | Diversity (mean-pADE/FDE↑) | Perf. (mean-PDMS↑) |
|--------|-------|----------------------|----------------------------|--------------------|
| ReCogDrive | IL+RL | 0.295 / 0.621 | 0.148 / 0.325 | 90.95 |
| Qwen2.5-VL | IL | 0.481 / 1.052 | 0.090 / 0.200 | 90.69 |
| + FTE (w/o SN) | IL | 0.513 / 1.129 | 0.170 / 0.381 | 90.65 |
| + FTE | IL | 0.480 / 1.078 | 0.346 / 0.803 | 91.31 |
| **+ FTE + RL** | IL+RL | **0.269 / 0.547** | **0.641 / 1.415** | **91.55** |

**说明**: 逐步拆解方法如何缓解窄策略。仅加扩展数据 (+FTE w/o SN) 反而降质量；加入 SN (+FTE) 后多样性大涨（mean-pFDE 0.200→0.803）且性能保持（91.31）——SN 是从多样数据学习的关键；再叠加 RL 把多样性与质量同时推到最优（mean-pFDE 1.415、minFDE 0.547）。

### Table 5: Ablation on Feasible Trajectory Expansion (IL) / IL 端消融

| DE | CoT | SN | NC↑ | DAC↑ | EP↑ | TTC↑ | C↑ | PDMS↑ |
|----|-----|----|-----|------|-----|------|----|-------|
| ✗ | ✗ | ✗ | 97.7 | 91.8 | 85.8 | 96.8 | 98.4 | 83.9 |
| ✗ | ✓ | ✗ | 98.2 | 93.2 | 85.8 | 97.3 | 98.4 | 85.6 |
| ✓ | ✓ | ✗ | 98.0 | 93.0 | 85.9 | 97.2 | 98.4 | 85.2 |
| ✗ | ✓ | ✓ | 98.2 | 94.3 | 86.7 | 97.3 | 98.4 | 86.9 |
| **✓** | **✓** | **✓** | 98.3 | **95.1** | 86.5 | 97.6 | 98.3 | **87.6** |

**说明**: 加 CoT 比裸 SFT +1.7 PDMS（83.9→85.6）作为强基线。**只加 DE 不加 SN 反而退化**（85.2 < 85.6），说明朴素数据扩展难以奏效；SN 是必要催化剂——DE+SN 才取得最佳 SFT 策略 87.6 PDMS，把多样探索样本转化为可执行知识。

### Table 6: Ablation on Diversity-Aware RL / RL 端消融

| Sampling Strategy | SDR | NC↑ | DAC↑ | EP↑ | TTC↑ | C↑ | PDMS↑ |
|-------------------|-----|-----|------|-----|------|----|-------|
| Human Difficulty | ✗ | 73.9 | 43.7 | 94.9 | 70.3 | 97.1 | 35.2 |
| Reject Unimodal | ✗ | 98.4 | 96.0 | 87.0 | 97.8 | 98.4 | 88.8 |
| ADAS(1×) | ✗ | 98.2 | 96.3 | 88.6 | 97.6 | 98.2 | 89.6 |
| ADAS(3×) | ✗ | 98.4 | 96.8 | 88.5 | 97.8 | 98.1 | 90.1 |
| **ADAS(3×)** | **✓** | 98.4 | **96.9** | 88.5 | **97.9** | 98.1 | **90.3** |

**说明**: 难度感知采样 (Human Difficulty) 与 Random/Full 一样**导致训练坍缩**（仅 35.2）——避开零优势场景才是关键。按奖励多样性过滤（Reject Unimodal）避免坍缩达 88.8；ADAS 用伯努利检验统计验证多样性更优（89.6），3 外循环达 90.1；叠加 SDR 取得最优 90.3。

### Table 7: SFT Training Config / SFT 训练配置（附录）

| Setting | SFT |
|---------|-----|
| Base Model | Qwen2.5-VL-3B |
| Max Pixels | 262144 |
| Global Batch Size | 128 |
| Epochs | 3 (align) / 3 (fine-tune) |
| Trainset Samples | 103k (navtrain) + 39k (DE) |
| Learning Rate | 4e-5 / 5e-6 |
| Weight Decay | 0.05 |
| Warmup Ratio | 0.10 |
| bfloat16 | ✓ |
| GPUs | 8 |

**说明**: 两阶段 SFT（对齐 3 epoch + 微调 3 epoch），扩展数据 (DE) 提供约 39k 额外样本。

### Table 8: GRPO Training Config / GRPO 训练配置（附录）

| Setting | GRPO |
|---------|------|
| Max Pixels | 262144 |
| Rollout Batch Size | 256 |
| Actor Global Batch Size | 256 |
| N Rollout (Group Size) | 8 |
| Outer Loops | 3 |
| Total Steps | 130 |
| Active Samples | 6k / 3k / 1k（3 外循环递减） |
| bfloat16 | ✓ |
| GPUs | 8 |

**说明**: ADAS 的活动样本量随外循环递减 (6k→3k→1k)，体现"逐步聚焦高多样性场景"。

### Table 9: Inference Latency / 推理延迟（附录）

| Method | Setting | Latency (s) |
|--------|---------|-------------|
| AutoVLA | Dual-Sys (Text) | 9.31 |
| AutoVLA | Dual-Sys (Action) | 3.95 |
| AutoVLA | Dual-Sys (Action + RFT) | 1.31 |
| **Curious-VLA** | **Slow Think only (Text)** | **1.57** |

**说明**: Curious-VLA 仅用"慢思考(Text)"单系统即 1.57s，优于 AutoVLA 的双系统 Text(9.31)/Action(3.95)，仅略慢于其 Action+RFT(1.31)，效率有竞争力。

### Table 10: Extended Exploration Analysis (@k=8) / 扩展探索分析（附录）

| Method | Output | Quality (ADE/FDE) | Diversity (ADE/FDE) | Perf. (mean-PDMS) |
|--------|--------|-------------------|---------------------|-------------------|
| DiffusionDrive | Diff.@all | 0.218 / 0.430 | 0.571 / 1.175 | 87.60 |
| DiffusionDrive | Diff.@1 | 0.350 / 0.720 | 0.037 / 0.076 | 88.10 |
| **Curious-VLA** | AR@1 | 0.269 / 0.547 | **0.641 / 1.415** | **91.55** |

**说明**: 与扩散式 DiffusionDrive 对比：其 20 个去噪候选 (@all) 多样性尚可但质量/性能不及 Curious-VLA，置信 Top-1 (@1) 选择则多样性骤降。Curious-VLA 单条自回归 (AR@1) 在多样性 (1.415) 与性能 (91.55) 上全面领先。

---

## 实验

### 数据集 / 基准

| 基准 | 规模 | 特点 | 用途 |
|------|------|------|------|
| Navsim v1 [10] | navtrain ~103k 样本，navtest 评测 | 基于 OpenScene/nuPlan 的非反应式仿真，8 环视相机 + 5 LiDAR（本文仅用前视单相机） | 训练/评测（PDMS） |
| Navsim v2 [3] | navtest split | 更难，扩展 PDMS (EPDMS) | 评测（EPDMS） |
| nuScenes | 1000×20s 场景，~1.4M 图像 / ~390k LiDAR；训练用 28k 子集 | 真实世界泛化验证 | 训练/评测（L2、碰撞率） |

### 实现细节

- **基座**: [[Qwen2.5-VL]]-3B，纯 VLM 直接自回归文本 waypoint（无外接规划器）
- **IL**: 用 LLaMA-Factory，6 epoch（对齐 3 + 微调 3），global batch 128，lr 4e-5/5e-6，weight decay 0.05，warmup 0.10
- **RL**: 基于 VeRL + EasyR1 改造 GRPO，130 步、3 个 ADAS 外循环，rollout 数 8，actor global batch 256，活动样本 6k/3k/1k
- **硬件**: 8× NVIDIA H100，DeepSpeed ZeRO-1，bfloat16
- **数据**: DE 从 103k 筛 12k 挑战片段 → 扩展为 142k 安全多样样本（附录表注 +39k DE）

### 关键实验结论

- **Navsim v1**: PDMS 90.3（单前视 SOTA），Best-of-N(N=6) 94.8 追平人类 GT。
- **Navsim v2**: EPDMS 85.3 SOTA（较 DiffusionDrive +0.8）。
- **nuScenes**: L2/碰撞率均优于现有 VLA 与 E2E，验证真实世界泛化。
- **探索分析 (Table 4/10)**: SN 是把多样数据转为知识的关键；RL 把多样性与质量同推至最优；AR@1 多样性超 DiffusionDrive 多候选。
- **消融 (Table 5/6)**: 朴素 DE 无效、需 SN 催化；难度采样会坍缩、需 ADAS 的伯努利多样性筛选 + SDR。

---

## 批判性思考

### 优点
1. **问题定义新且有理论支撑**: "窄策略"把 IL 过拟合 → 探索坍缩 → RL 优势坍缩 ($\sigma_R\to0$, $A_i\to0$) 串成一条清晰因果链（公式5/6），并配套行为诊断指标量化，不只是经验观察。
2. **方案系统、覆盖 IL+RL 双端**: 从数据 (DE/CoT/SN)、采样 (ADAS)、奖励 (SDR) 三个层面同时注入多样性，消融充分（Table 4/5/6/10）证明各组件不可或缺，尤其 SN 与 ADAS 的"催化/防坍缩"作用论证扎实。
3. **小模型高性价比**: 仅 3B 基座、单前视相机即超越 7B/8B 与 3×相机方法，Best-of-N 追平人类 GT，且推理延迟有竞争力（1.57s 单系统）。

### 局限性
1. **全程开环评测**: Navsim v1/v2 与 nuScenes 均为开环 (open-loop) 指标，未做闭环/反应式仿真或真车实验，"探索多样性"对实际闭环安全的收益缺乏直接验证。
2. **数字小不一致 + 强依赖外部组件**: 摘要 EPDMS 85.4 vs 正文/表 85.3；DE 依赖 ReCogDrive 扩散生成 + Qwen2.5-VL-72B 筛选/CoT 合成，数据管线重、对外部大模型与扩散先验依赖强，复现成本高。
3. **超参与阈值偏经验**: SDR 的 focal $\gamma_m$、ADAS 的 $\epsilon_{div}/\epsilon_{conf}$、伯努利近似（把复杂奖励分布简化为二元）等关键设计缺少敏感度分析；Best-of-N 的"质量"提升部分依赖选择器，单条策略 (90.3) 与 Best-of-N (94.8) 差距大。

### 潜在改进方向
1. 引入**闭环评测**（如 Bench2Drive/nuPlan 闭环）量化多样性→安全的因果收益。
2. 把伯努利近似换成更细的奖励分布建模，或让 $\gamma_m$、$\epsilon_{div}$ 等阈值可学习/自动搜索，减少手工调参。
3. 探索去除对扩散生成器与 72B 教师的依赖（如自蒸馏式数据扩展），降低数据管线复杂度与复现门槛。

### 可复现性评估
- [x] 代码开源（https://github.com/Mashiroln/curious_vla.git）
- [ ] 预训练模型（论文未明确声明 release 权重）
- [x] 训练细节完整（附录 Table 7/8 给出 SFT/GRPO 超参与样本量）
- [x] 数据集可获取（Navsim v1/v2、nuScenes 公开；扩展数据依赖外部模型生成）

---

## 速查卡片

> [!summary] Curious-VLA: Unleashing Exploration in Driving VLA
> - **核心**: 揭示驾驶 VLA 的"窄策略"——IL 过拟合塌缩探索、致 GRPO 优势坍缩 ($\sigma_R\to0$) 提前饱和；在 IL+RL 两端注入多样性破局。
> - **方法**: IL 端 FTE（扩散扩展 DE + 四阶段 CoT + 逐步归一化 SN）；RL 端 ADAS（伯努利多样性筛场景）+ SDR（focal 跨度奖励）；基座 Qwen2.5-VL-3B、单前视、文本 waypoint。
> - **结果**: Navsim v1 PDMS 90.3 / Best-of-N 94.8（追平人类 GT）、v2 EPDMS 85.3、nuScenes L2/碰撞 SOTA。
> - **代码**: https://github.com/Mashiroln/curious_vla.git

---

*笔记创建时间: 2026-06-29*
