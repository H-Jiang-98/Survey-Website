---
title: "Drive My Way: Preference Alignment of Vision-Language-Action Model for Personalized Driving"
method_name: "Drive My Way (DMW)"
authors: [Zehao Wang, Huaide Jiang, Shuaiwu Dong, Yuping Wang, Hang Qiu, Jiachen Li]
year: 2026
venue: CVPR
tags: [VLA, autonomous-driving, personalization, preference-alignment, contrastive-learning, GRPO, reward-shaping]
zotero_collection: _inbox
image_source: online
arxiv_html: https://arxiv.org/html/2603.25740v1
created: 2026-06-29
---

# Drive My Way: Preference Alignment of Vision-Language-Action Model for Personalized Driving

## 元信息

| 项目 | 内容 |
|------|------|
| 作者 | Zehao Wang, Huaide Jiang, Shuaiwu Dong, Yuping Wang, Hang Qiu, Jiachen Li |
| 机构 | University of California, Riverside（DMW 项目主页域名 dmw-cvpr.github.io）等 |
| 会议 | CVPR 2026 |
| 类别 | VLA（Vision-Language-Action）/ 端到端自动驾驶 / 个性化 |
| 日期 | 2026-03（arXiv v1） |
| 项目主页 | https://dmw-cvpr.github.io/ |
| 链接 | [arXiv](https://arxiv.org/abs/2603.25740) / [Code](https://dmw-cvpr.github.io/) |

---

## 一句话总结

> 在预训练 [[VLA]] 驾驶主干上，用对比学习从真人数据里学出"驾驶员身份嵌入"做长期偏好对齐，再用 LLM 生成的风格化奖励 + [[GRPO]] 微调残差解码器做短期指令适配，让一辆车开出"每个司机自己的味道"。

---

## 核心贡献

1. **个性化驾驶数据集（PDD）**: 采集 **30 名真实驾驶员**在 [[CARLA]] 中用 Logitech 方向盘踏板完成 20 个标准化场景的驾驶轨迹 + 结构化"驾驶员画像"问卷，并用 [[PDM-Lite]] 专家目标速度作为风格的密集描述子，填补"既有长期习惯又有显式画像"的数据空白。
2. **长期偏好对齐机制**: 提出"画像编码器 $f_p$ + 路线处理器 $f_b$"双塔，用 [[InfoNCE]] 对比损失把"司机画像"与"实际行为窗口"拉到同一隐空间，得到可冻结、可泛化到陌生司机（OOD）的**用户嵌入 $z_p^m$**，并以它条件化策略。
3. **短期指令适配（风格感知奖励 + GRPO）**: 用 [[GPT-5]] 从场景描述与自然语言指令推断**奖励权重与阈值**（安全/效率/舒适三项），经专家复核后驱动 [[GRPO]] 微调一个**离散残差解码器**，在不破坏底层安全策略的前提下实时调节驾驶激进/保守程度。

---

## 问题背景

### 要解决的问题
人类驾驶本质上是"私人化"的——同一情境下不同人在加速、刹车、并线、让行、超车上的取舍各异，且受长期习惯与短期意图共同塑造。如何让端到端自动驾驶系统**既贴合某位驾驶员的长期习惯、又能听懂实时自然语言指令做短期风格调整**，是本文要解决的核心问题。

### 现有方法的局限
1. **通用目标或固定模式**: 现有端到端系统要么只优化通用目标（安全/效率），要么只提供少数固定"驾驶模式"，无法适配个体偏好。
2. **VLA 的个性化几乎空白**: 现有 [[VLA]] 驾驶模型能把传感输入+语言指令翻译成动作，但"个性化、适配个人偏好"这一维度基本未被探索。
3. **风格条件粒度太粗**: 已有风格条件策略（如 [[StyleDrive]]）只学到粗粒度风格，忽略细粒度个体差异。
4. **不建模长期习惯**: 基于 LLM/VLM 的个性化方法不考虑驾驶员长期习惯与累积经验，而这些恰恰是多样且持续演化的。

### 本文的动机
- 长期习惯应当从**真人历史驾驶数据 $\mathcal D^m$**中学习，并与其**画像 $P^m$**关联，而非靠少量风格标签；
- 短期意图应当通过**自然语言指令 $I_t$**实时注入；
- 关键设计假设：把"画像"和"行为"对齐到同一嵌入空间后，用户嵌入既能复现已知司机风格，又能**零样本泛化到未见司机**（只需其画像即可推断行为倾向）。

---

## 方法详解

### 模型架构

DMW 采用 **感知-语言-动作（VLA）+ 个性化条件 + 残差控制** 的模块化流水线（见 Figure 3）：
- **输入**: 前视相机图像 $\mathbf I_t$ + 自车状态 $\mathbf q_t$ + 短期语言指令 $I_t$ + 导航目标 $g_t$ + 驾驶员画像标识 $P^m$
- **Backbone**: [[SimLingo]]（基于 [[InternVL2]]-1B = [[InternViT]]-300M-448px 视觉编码器 + [[Qwen2]]-0.5B 语言模型，Qwen2 用 [[LoRA]] 适配）
- **核心模块**: [[长期偏好编码器]]（画像编码器 $f_p$ + 路线处理器 $f_b$，[[InfoNCE]] 对齐）→ [[残差解码器]]（离散动作头，[[GRPO]] 微调）
- **输出**: 连续动作 $a_t\in\mathbb R^3$（throttle, brake, steering）

整体被建模为一个 [[MDP]] $(\mathcal S,\mathcal A,\mathcal O,\mathcal T,\mathcal R,\gamma)$，其中观测 $o_t=(\mathbf I_t,\mathbf q_t,I_t,g_t,P^m)$，动作空间 $\mathcal A\subset\mathbb R^3$。策略需从历史数据 $\mathcal D^m$ 推断驾驶员长期倾向，并与其画像 $P^m$ 关联。优化目标：

$$
\max_{\theta}\ \mathbb E_{\pi_\theta}\!\left[\sum_{t=0}^{T}\gamma^{t}\,\mathcal R(s_t,a_t)\right]
$$

**符号说明**: $\pi_\theta$ 为策略；$\gamma$ 折扣因子；$\mathcal R$ 为下文风格感知/偏好对齐奖励；$T$ 为回合长度。

### 核心模块

#### 模块1: VLA Backbone（SimLingo 感知-规划主干）

**设计动机**: 复用一个已在大规模驾驶数据上预训练好的 VLA，作为稳定的"基础驾驶能力"提供者，个性化只做增量。

**具体实现**:
- 主干为 [[SimLingo]]，由 [[PDM-Lite]] 特权专家示范 + 辅助推理信号监督；
- 主干预测**时序与几何 waypoints**（路线点 + 速度点），再转换为目标速度与转向指令，经 [[PID]] 控制器得到**基础动作 $a_t^{base}$**；
- 个性化层不直接重写动作，而是输出一个**残差 $a_t^\Delta$** 叠加在基础动作上（见公式"动作合成"）。

#### 模块2: Long-Term Preference Encoder（长期偏好编码器，双塔 + 对比对齐）

**设计动机**: 把"司机是谁（画像）"和"司机怎么开（行为）"绑定到同一隐空间，从而由画像即可推断长期行为倾向，支持对未见司机的零样本个性化。

**具体实现**（见 Figure 4）:
- **画像编码器 $f_p(\cdot)$**: 输入结构化画像 $P^m$（人口学、驾龄、驾驶目的等问卷），用 [[DeBERTaV3]] 文本编码器 + 投影头，输出用户嵌入 $z_p^m\in\mathcal Z$；
- **路线处理器 $f_b(\cdot)$**: 输入过去长度为 $k$ 的轨迹窗口 $\xi_t^m$，用带多头自注意力的时序编码器，输出行为嵌入 $z_{b,t}^m\in\mathcal Z$；
- 用 [[InfoNCE]] 对比损失把同一司机的 $z_p^m$ 与 $z_{b,t}^m$ 拉近、不同司机推远；
- **Adaptive Average Pooling (AAP)**: 消融（Table 4）证明对行为窗口做自适应平均池化是对齐成败的关键，去掉后对齐分从 0.92 跌到 0.67；
- 训练收敛后**冻结画像编码器**，用 [[AdamW]]（weight decay 1e-3, lr 1e-4）。

#### 模块3: Residual Decoder（残差解码器，GRPO 强化微调）

**设计动机**: 在不破坏预训练主干安全性的前提下，注入个性化与风格意图——只学"相对基础动作的微调量"。

**具体实现**（见 Figure 5）:
- 输入 token：视觉、语言、导航、**用户嵌入 $z_p^m$**、运动 query token，再加**可学习残差 query token**；用户嵌入与其它模态**拼接（concatenate）**完成条件化；
- 解码器为 MLP + 分类动作头，输出**两个离散残差**：(i) 速度变化、(ii) 转向变化；
- 用 [[GRPO]]（Group Relative Policy Optimization）微调，每个输入采样 **4 个 response** 做策略梯度更新；硬件为 **8×NVIDIA RTX A6000**，per-GPU batch=8。

### 关键公式与机制

#### 公式1: [[Action Chunking|动作合成]]（残差叠加）

$$
a_t = a_t^{base} + a_t^{\Delta}
$$

**含义**: 最终动作 = 主干 waypoints 经 PID 得到的基础动作 + 残差解码器的个性化离散残差。

**符号说明**:
- $a_t^{base}$: 由预测 waypoints + [[PID]] 控制器得到的基础动作
- $a_t^{\Delta}$: 残差解码器输出的离散残差（速度变化 + 转向变化）

#### 公式2: [[InfoNCE]] 对比对齐损失

$$
\mathcal L_t^{m} = -\log\frac{\exp\!\big(\operatorname{sim}(z_p^{m},\,z_{b,t}^{m})/\tau\big)}{\sum_{j=1}^{M}\exp\!\big(\operatorname{sim}(z_p^{j},\,z_{b,t}^{m})/\tau\big)}
$$

**含义**: 让同一司机 $m$ 的画像嵌入 $z_p^m$ 与其当前行为嵌入 $z_{b,t}^m$ 相似度最高，把其它司机 $j$ 的画像嵌入作为负样本推远，从而把"画像↔行为"对齐到共享隐空间 $\mathcal Z$。

**符号说明**:
- $z_p^m$: 司机 $m$ 的画像（用户）嵌入，来自 $f_p$
- $z_{b,t}^m$: 司机 $m$ 在 $t$ 时刻的行为嵌入，来自 $f_b$
- $\operatorname{sim}(\cdot,\cdot)$: 余弦相似度；$\tau$: 温度系数；$M$: 司机总数

#### 公式3: 行为窗口输入

$$
\xi_t^{m} = \big\{(\mathbf I_{t-k:t}^{m},\,\mathbf q_{t-k:t}^{m},\,a_{t-k:t}^{m})\big\}
$$

**含义**: 路线处理器 $f_b$ 的输入是过去 $k$ 步的图像、自车状态、动作三元组窗口，刻画"最近这段怎么开"。

**符号说明**: $k$ 为轨迹窗口长度；$\mathbf I,\mathbf q,a$ 分别为图像、状态、动作序列。

#### 公式4: 最不相似司机选择（数据增强）

$$
u = \arg\min_{x\in\mathcal M}\ \operatorname{sim}(z_p^{m},\,z_p^{x})
$$

**含义**: 为做"反风格"增强，挑选与司机 $m$ 画像最不相似的司机 $u$，用其平均动作幅度去缩放 $m$ 的动作，构造对比性的偏好样本。

**符号说明**: $\mathcal M$ 为司机集合；$z_p^x$ 为司机 $x$ 的画像嵌入。

#### 公式5: 增强动作缩放

$$
\tilde a_t^{m} = \frac{\bar a^{m}}{\bar a^{u}}\cdot a_t^{m}
$$

**含义**: 用司机 $m$ 与最不相似司机 $u$ 的路线级平均动作之比，对 $m$ 的动作整体缩放，得到"偏向 $u$ 风格"的增强动作 $\tilde a_t^m$。

**符号说明**: $\bar a^m,\bar a^u$ 分别为 $m,u$ 的路线级平均动作。

#### 公式6: 偏好对齐奖励

$$
\mathcal R(s_t^{m}, a_t) =
\begin{cases}
d(a_t,\ a_t^{m}), & \text{条件于目标画像 } P^{m}\\[4pt]
d(a_t,\ \tilde a_t^{m}), & \text{条件于替代画像 } P^{u}
\end{cases}
$$

**含义**: 当策略条件于目标司机画像时，奖励鼓励动作接近该司机真实动作 $a_t^m$；条件于替代画像 $P^u$ 时，鼓励接近增强动作 $\tilde a_t^m$，以此区分不同画像应导出不同行为。

**符号说明**: $d(\cdot)$ 为动作空间相似度度量。

#### 公式7: 风格感知奖励（短期指令适配）

$$
\mathcal R(s_t, a_t) = w_s\, R_{safety} + w_e\, R_{efficiency} + w_c\, R_{comfort}
$$

**含义**: 短期指令通过调整三项权重把"风格意图"注入奖励；激进指令对应更高 $w_e$ 与偏好速度，保守指令则提高 $w_c/w_s$。权重与阈值由 [[GPT-5]] 从场景描述+指令推断、再经专家复核。

**符号说明**: $w_s,w_e,w_c$ 为指令相关的安全/效率/舒适权重。

#### 公式8: 安全奖励

$$
R_{safety} = \mathbb I_{safety}\big(\mathrm{TTC}_t \ge \beta_{safety}\big)
$$

**含义**: 当碰撞时间 $\mathrm{TTC}_t$ 不低于阈值 $\beta_{safety}$ 时给予安全奖励（二值指示）。

#### 公式9: 效率奖励

$$
R_{efficiency} = \exp\!\big(-\alpha\,|v_t - v_{pref}|\big)
$$

**含义**: 实际速度 $v_t$ 越接近指令相关偏好速度 $v_{pref}$，效率奖励越高；$\alpha$ 为惩罚系数。

#### 公式10: 舒适奖励

$$
R_{comfort} = \mathbb I_{comf}\big(|a_t^{steer}| < \beta_{lat}\ \text{and}\ |a_t^{acc}| < \beta_{long}\big)
$$

**含义**: 当转向幅度与纵向加速度同时低于横/纵向阈值 $\beta_{lat},\beta_{long}$ 时给予舒适奖励（二值指示）。

---

## 关键图表

<!-- 图片均使用 arXiv HTML 在线链接，未本地化 -->

### Figure 1: Teaser / 长期偏好 + 短期指令双适配

![Figure 1](https://arxiv.org/html/2603.25740v1/x1.png)

**说明**: DMW 同时实现**长期偏好对齐**（贴合某司机习惯）与**短期风格指令适配**（实时听懂自然语言改风格）的端到端个性化驾驶，奠定全文"长期+短期"双轴主张。

### Figure 2: Personalized Driving Dataset (PDD) / 数据集概览

![Figure 2](https://arxiv.org/html/2603.25740v1/x2.png)

**说明**: PDD 由两部分组成——真人**驾驶数据**（自车运动、周围车/行人/骑行者/路侧危险的感知、信号灯/停止标志/路线几何/限速等交通上下文）与**结构化驾驶员画像**（问卷）。30 名司机各跑 20 个 CARLA 场景，5 Hz 采样，相机 1024×512、110° FoV。

### Figure 3: DMW Framework / 整体框架

![Figure 3](https://arxiv.org/html/2603.25740v1/x3.png)

**说明**: 预训练 [[VLA]] 主干吃进前视图像、指令、路线目标点、用户画像；运动预测器输出路线与速度 waypoints 推出**基础动作**（throttle/steer），**残差解码器**输出离散残差叠加到基础动作上得最终个性化动作。这是公式 $a_t=a_t^{base}+a_t^\Delta$ 的结构图。

### Figure 4: Contrastive Learning / 对比对齐机制

![Figure 4](https://arxiv.org/html/2603.25740v1/x4.png)

**说明**: 长期偏好编码器（画像编码器 $f_p$）与路线处理器（$f_b$）的 [[InfoNCE]] 对比学习：同一司机的画像嵌入 $z_p^m$ 与行为嵌入 $z_{b,t}^m$ 拉近、异司机推远，对应公式2。

### Figure 5: Fine-tuning & Reward Generation / 短期指令对齐流程

![Figure 5](https://arxiv.org/html/2603.25740v1/x5.png)

**说明**: 短期指令适配的微调与奖励生成流程：自然语言指令 + 场景描述 → [[GPT-5]] 推断 $(w_s,w_e,w_c)$ 与阈值 → 专家复核 → 风格感知奖励驱动 [[GRPO]] 微调残差解码器（公式7–10）。

### Figure 6: Driving Preference Visualization (1) / 激进 vs 保守轨迹

![Figure 6](https://arxiv.org/html/2603.25740v1/x6.png)

**说明**: 安全关键场景下激进/保守指令产生的轨迹对比。红色 waypoints 为按距离参数化（每 1 m）的导航路径，绿色 waypoints 为按时间参数化（每 0.25 s）的轨迹——绿点越疏代表速度越高，直观体现风格差异。

### Figure 7: Driving Preference Visualization (2) / 更多场景轨迹对比

![Figure 7a](https://arxiv.org/html/2603.25740v1/x7.png)
![Figure 7b](https://arxiv.org/html/2603.25740v1/x8.png)

**说明**: 跨多场景的激进 vs 保守指令轨迹补充示例，红/绿 waypoints 含义同 Figure 6，进一步佐证 DMW 能按指令拉开行为差异。

### Table 1: Bench2Drive Closed-Loop Metrics by Style / 风格指令下闭环驾驶指标

| Method | Style | DS | SR | Efficiency | Comfort | Speed | Acc. | LC | Headway | TT |
|--------|-------|-----|-----|-----------|---------|-------|------|-----|---------|-----|
| SimLingo | Aggressive | 78.56 | 65.83 | 247.60 | 18.61 | 7.66 | 5.39 | 0.75 | 25.99 | 25.35 |
| SimLingo | Neutral | 78.15 | 65.85 | 241.44 | 24.67 | 7.37 | 5.22 | 0.75 | 27.81 | 31.41 |
| SimLingo | Conservative | 78.18 | 65.56 | 238.77 | 26.99 | 7.21 | 5.29 | 0.70 | 29.12 | 33.02 |
| StyleDrive | Aggressive | 75.68 | 60.89 | 256.71 | 16.79 | 7.23 | 5.59 | 0.74 | 24.95 | 27.76 |
| StyleDrive | Neutral | 76.26 | 62.13 | 249.07 | 21.35 | 6.98 | 5.43 | 0.66 | 23.62 | 29.12 |
| StyleDrive | Conservative | 77.02 | 61.96 | 242.18 | 23.67 | 6.82 | 5.39 | 0.70 | 27.19 | 29.98 |
| DMW-Vanilla | Aggressive | 82.19 | 70.97 | 253.10 | 15.86 | 7.86 | 5.29 | 0.78 | 26.46 | 19.69 |
| DMW-Vanilla | Neutral | 81.96 | 70.63 | 247.77 | 19.21 | 7.66 | 5.17 | 0.77 | 26.63 | 23.16 |
| DMW-Vanilla | Conservative | 81.48 | 71.05 | 246.80 | 21.87 | 7.75 | 5.37 | 0.75 | 26.90 | 22.51 |
| **DMW** | **Aggressive** | 79.50 | 67.36 | **281.56** | 21.62 | 7.72 | **6.01** | 0.70 | 26.37 | 26.93 |
| **DMW** | **Neutral** | 82.03 | 70.95 | 244.98 | 28.67 | 6.34 | 5.43 | 0.61 | 27.60 | 40.75 |
| **DMW** | **Conservative** | **82.72** | **71.56** | 237.06 | **34.62** | 6.18 | 5.26 | 0.60 | 30.05 | 47.38 |

**说明**: DMW 把风格"拉开"得最明显——激进时效率分冲到 281.56（远超 SimLingo 247.60）、加速度最高 6.01；保守时舒适分最高 34.62、车头时距最大 30.05、行程时间显著拉长（47.38 s）。[[StyleDrive]] 因固定风格条件，指标位移小且 DS 更低；DMW-Vanilla（固定权重 $w_s{=}0.35,w_e{=}0.35,w_c{=}0.30$）DS 略高但风格区分度不足，证明**风格感知奖励权重**是拉开风格的关键。

### Table 2: Per-Driver Metrics With/Without Style Instruction / 各司机有无风格指令对比

| Driver | Scenario | Style (None→) | DS | Speed | Efficiency | Acc. | Headway | AS | Ratings |
|--------|----------|---------------|-----|-------|-----------|------|---------|-----|---------|
| D1 | Emergency Brake | →Aggressive | 95.10→84.02 | 8.06→9.58 | 150.94→167.12 | 6.31→6.39 | 18.94→17.11 | 1.00 | 8.8 |
| D2 | Emergency Brake | →Aggressive | 98.64→96.71 | 4.61→4.97 | 96.02→99.21 | 4.94→5.35 | 23.10→21.18 | 0.67 | 8.3 |
| D3 | Emergency Brake | →Aggressive | 97.88→95.93 | 5.39→6.34 | 109.83→121.06 | 5.63→6.16 | 21.61→19.22 | 1.00 | 8.0 |
| D4 | Emergency Brake | →Aggressive | 94.02→93.15 | 7.76→8.35 | 145.22→159.56 | 6.03→6.92 | 21.03→18.19 | 0.67 | 7.8 |
| D1 | Merging | →Conservative | 90.05→94.38 | 9.32→8.76 | 270.84→260.10 | 5.15→4.68 | 96.44→101.16 | 0.67 | 8.4 |
| D2 | Merging | →Conservative | 86.94→91.72 | 6.37→5.76 | 196.82→175.31 | 4.29→4.16 | 118.92→121.03 | 1.00 | 8.1 |
| D3 | Merging | →Conservative | 96.85→95.94 | 6.98→6.20 | 205.41→195.77 | 4.46→4.24 | 111.26→114.05 | 0.67 | 7.6 |
| D4 | Merging | →Conservative | 97.31→97.88 | 8.78→8.63 | 261.52→259.52 | 5.01→4.91 | 88.76→100.93 | 1.00 | 8.2 |
| D1 | Overtaking | →Neutral | 97.56→97.94 | 7.70→7.94 | 220.19→223.13 | 7.28→7.37 | 14.69→14.53 | 1.00 | 9.0 |
| D2 | Overtaking | →Neutral | 98.23→98.41 | 5.50→5.56 | 167.40→167.80 | 5.65→5.78 | 22.23→22.04 | 1.00 | 8.6 |
| D3 | Overtaking | →Neutral | 97.14→96.62 | 6.34→6.13 | 191.02→183.84 | 6.18→6.01 | 20.79→22.19 | 0.67 | 7.5 |
| D4 | Overtaking | →Neutral | 80.61→82.19 | 9.63→9.51 | 271.98→265.84 | 7.11→6.82 | 17.81→18.03 | 1.00 | 8.1 |
| D1 | Traffic Sign | →Conservative | 90.82→96.67 | 9.08→8.49 | 408.74→401.28 | 7.12→6.37 | 31.44→32.52 | 1.00 | 8.7 |
| D2 | Traffic Sign | →Conservative | 97.92→98.35 | 5.79→4.91 | 193.26→183.11 | 5.72→5.51 | 35.02→36.91 | 1.00 | 8.2 |
| D3 | Traffic Sign | →Conservative | 89.31→92.02 | 6.88→5.37 | 237.05→231.46 | 5.96→5.91 | 34.54→36.51 | 1.00 | 8.1 |
| D4 | Traffic Sign | →Conservative | 95.91→97.32 | 8.91→6.74 | 418.02→379.68 | 6.54→5.81 | 31.19→31.77 | 0.67 | 7.9 |

**说明**: 逐司机看，加上风格指令后速度/效率/车头时距随风格语义合理移动（如 Emergency Brake→Aggressive 时速度普遍上升），且各司机的 [[Alignment Score|对齐分(AS)]] 多为 0.67–1.00、人评 7.5–9.0，说明在个体层面既能保长期风格、又能响应短期指令。

### Table 3: Long-Term Preference Alignment (vs MORL-PD) / 长期偏好对齐

| Methods | AS-ID D1 | D2 | D3 | D4 | Ratings-ID D1 | D2 | D3 | D4 |
|---------|------|------|------|------|------|------|------|------|
| MORL-PD | 0.42 | 0.58 | 0.25 | 0.33 | 5.1 | 6.2 | 3.9 | 3.5 |
| **DMW** | **0.92** | **0.92** | **0.83** | **0.83** | **8.7** | **8.3** | **7.8** | **8.0** |

**说明**: 长期对齐上 DMW 全面碾压基线 [[MORL-PD]]：对齐分从 0.25–0.58 提升到 0.83–0.92，人评从 3.5–6.2 提升到 7.8–8.7，说明生成行为可被识别为"某司机自己的风格"。（OOD 列在原表中为空占位。）

### Table 4: Ablation — Adaptive Average Pooling (AAP) / 自适应平均池化消融

| Driver | Encoder | DS | Speed | Effic. | Acc. | Headway | AS | Ratings |
|--------|---------|-----|-------|--------|------|---------|-----|---------|
| D1 | w/o AAP | 82.74 | 4.52 | 161.38 | 5.84 | 50.11 | 0.67 | 6.1 |
| D2 | w/o AAP | 88.02 | 4.21 | 163.07 | 5.88 | 47.56 | 0.58 | 5.9 |
| D3 | w/o AAP | 87.65 | 4.49 | 133.12 | 5.81 | 41.63 | 0.25 | 4.6 |
| D4 | w/o AAP | 80.13 | 4.46 | 172.92 | 5.79 | 38.94 | 0.50 | 5.3 |
| **D1** | **w/ AAP** | **93.38** | **8.54** | **262.68** | 6.46 | 40.38 | **0.92** | **8.7** |
| **D2** | **w/ AAP** | **95.43** | 5.57 | 163.38 | 5.15 | 49.82 | **0.92** | **8.3** |
| **D3** | **w/ AAP** | **95.30** | 6.40 | 185.83 | 5.56 | 47.05 | **0.83** | **7.8** |
| **D4** | **w/ AAP** | **91.96** | 8.77 | **274.19** | 6.17 | 39.70 | **0.83** | **8.0** |

**说明**: 去掉 AAP 后行为坍缩——速度被压到 ~4.2–4.5、对齐分跌到 0.25–0.67、人评跌到 4.6–6.1；加上 AAP 后对齐分回到 0.83–0.92、速度与效率恢复正常。**AAP 是长期偏好编码器对齐成败的关键组件**。

### Table 5: Per-Driver Metrics Across All Scenarios (D1–D10) / 跨场景逐司机指标

| Driver | DS | Speed | Effic. | Acc. | Headway | AS | Ratings |
|--------|-----|-------|--------|------|---------|-----|---------|
| D1 | 93.38 | 8.54 | 262.68 | 6.46 | 40.38 | 0.92 | 8.7 |
| D2 | 95.43 | 5.57 | 163.38 | 5.15 | 49.82 | 0.92 | 8.3 |
| D3 | 95.30 | 6.40 | 185.83 | 5.56 | 47.05 | 0.83 | 7.8 |
| D4 | 91.96 | 8.77 | 274.19 | 6.17 | 39.70 | 0.83 | 8.0 |
| D5 | 92.58 | 7.48 | 229.93 | 5.58 | 39.08 | 0.83 | 7.9 |
| D6 | 97.71 | 6.16 | 190.74 | 6.08 | 42.41 | 0.75 | 7.4 |
| D7 | 96.11 | 7.16 | 218.25 | 6.36 | 40.12 | 0.83 | 7.9 |
| D8 | 95.28 | 5.82 | 188.27 | 5.54 | 43.95 | 0.92 | 8.2 |
| D9 | 96.16 | 5.60 | 178.97 | 5.84 | 45.07 | 0.67 | 7.0 |
| D10 | 94.20 | 8.38 | 258.28 | 6.89 | 40.26 | 1.00 | 8.6 |

**说明**: 在更多司机（D1–D10，含 OOD）上对齐分稳定在 0.67–1.00、人评 7.0–8.7，显示用户嵌入对**陌生司机的零样本泛化**能力（只凭画像即可推断风格）。

### Table 6: User Study Ratings for Style Instructions / 风格指令人评（E1–E5）

> 五位评测者（E1–E5）对 SimLingo / StyleDrive / DMW 在 4 场景 × 3 风格下的轨迹打分（0–10），评分越高表示越贴合指令意图。

| Scenario | Model | Style | E1 | E2 | E3 | E4 | E5 |
|----------|-------|-------|-----|-----|-----|-----|-----|
| Emergency Brake | SimLingo | Conservative | 7.4 | 6.7 | 6.4 | 7.3 | 6.5 |
| Emergency Brake | SimLingo | Neutral | 7.0 | 7.3 | 7.5 | 7.0 | 7.1 |
| Emergency Brake | SimLingo | Aggressive | 6.5 | 7.6 | 7.2 | 6.6 | 7.4 |
| Emergency Brake | StyleDrive | Conservative | 8.2 | 7.5 | 7.3 | 8.0 | 7.2 |
| Emergency Brake | StyleDrive | Neutral | 7.8 | 8.0 | 8.2 | 7.6 | 7.9 |
| Emergency Brake | StyleDrive | Aggressive | 7.2 | 8.4 | 7.9 | 7.1 | 8.2 |
| Emergency Brake | **DMW** | Conservative | **9.0** | 8.1 | 8.0 | **8.8** | 7.9 |
| Emergency Brake | **DMW** | Neutral | 8.4 | 8.6 | **9.1** | 8.1 | 8.5 |
| Emergency Brake | **DMW** | Aggressive | 7.8 | **9.2** | 8.3 | 7.6 | **9.0** |
| Merging | SimLingo | Conservative | 7.2 | 6.4 | 6.3 | 7.2 | 6.3 |
| Merging | SimLingo | Neutral | 6.8 | 7.0 | 7.3 | 6.7 | 6.8 |
| Merging | SimLingo | Aggressive | 6.2 | 7.5 | 6.9 | 6.1 | 7.3 |
| Merging | StyleDrive | Conservative | 8.0 | 7.3 | 7.2 | 7.9 | 7.1 |
| Merging | StyleDrive | Neutral | 7.6 | 7.8 | 8.1 | 7.4 | 7.7 |
| Merging | StyleDrive | Aggressive | 7.0 | 8.3 | 7.6 | 6.9 | 8.1 |
| Merging | **DMW** | Conservative | **8.9** | 7.9 | 7.7 | **8.7** | 7.6 |
| Merging | **DMW** | Neutral | 8.2 | 8.4 | **9.0** | 8.0 | 8.3 |
| Merging | **DMW** | Aggressive | 7.5 | **9.1** | 8.1 | 7.4 | **8.9** |
| Overtaking | SimLingo | Conservative | 7.3 | 6.3 | 6.2 | 7.2 | 6.2 |
| Overtaking | SimLingo | Neutral | 6.9 | 7.1 | 7.4 | 6.8 | 6.9 |
| Overtaking | SimLingo | Aggressive | 6.1 | 7.5 | 7.0 | 6.0 | 7.3 |
| Overtaking | StyleDrive | Conservative | 8.1 | 7.2 | 7.1 | 8.0 | 7.0 |
| Overtaking | StyleDrive | Neutral | 7.7 | 7.9 | 8.2 | 7.5 | 7.8 |
| Overtaking | StyleDrive | Aggressive | 6.9 | 8.4 | 7.7 | 6.8 | 8.2 |
| Overtaking | **DMW** | Conservative | **9.1** | 7.7 | 7.5 | **8.8** | 7.6 |
| Overtaking | **DMW** | Neutral | 8.3 | 8.5 | **9.0** | 8.0 | 8.4 |
| Overtaking | **DMW** | Aggressive | 7.4 | **9.3** | 8.2 | 7.2 | **9.2** |
| Traffic Sign | SimLingo | Conservative | 7.4 | 7.0 | 6.9 | 7.3 | 6.8 |
| Traffic Sign | SimLingo | Neutral | 7.1 | 7.3 | 7.5 | 7.0 | 7.1 |
| Traffic Sign | SimLingo | Aggressive | 6.8 | 7.6 | 7.2 | 6.7 | 7.4 |
| Traffic Sign | StyleDrive | Conservative | 8.3 | 7.8 | 7.6 | 8.1 | 7.7 |
| Traffic Sign | StyleDrive | Neutral | 7.9 | 8.1 | 8.3 | 7.7 | 8.0 |
| Traffic Sign | StyleDrive | Aggressive | 7.4 | 8.6 | 8.0 | 7.2 | 8.4 |
| Traffic Sign | **DMW** | Conservative | **8.9** | 8.3 | 8.1 | **8.6** | 8.2 |
| Traffic Sign | **DMW** | Neutral | 8.5 | 8.7 | **8.8** | 8.2 | 8.6 |
| Traffic Sign | **DMW** | Aggressive | 8.0 | **9.0** | 8.4 | 7.8 | **8.9** |

**说明**: 在所有场景×风格下，DMW 的人评（多数 7.5–9.3）一致高于 [[StyleDrive]] 与 [[SimLingo]]，定量印证其生成轨迹最贴合指令意图。

---

## 实验

### 数据集 / 基准

| 名称 | 规模 | 特点 | 用途 |
|------|------|------|------|
| Personalized Driving Dataset (PDD) | 30 司机 × 20 场景，CARLA Town 12，5 Hz，相机 1024×512/110° | 真人方向盘驾驶 + 结构化画像问卷 + [[PDM-Lite]] 目标速度做风格描述子 | 训练（25 司机）/ OOD 评测（5 司机） |
| 风格指令集 | 20 场景 × 9 条指令（3 风格 × 3 直接程度） | 含风格意图与场景语义 | 短期指令微调/评测 |
| [[Bench2Drive]] | 每场景 3 条测试路线 | 闭环评测（**碰撞即终止**，强调安全） | 评测 |

### 实现细节

- **Backbone**: [[SimLingo]] = [[InternVL2]]-1B（[[InternViT]]-300M-448px + [[Qwen2]]-0.5B，Qwen2 用 [[LoRA]]）
- **偏好编码器**: 画像编码器 = [[DeBERTaV3]] + 投影头；路线处理器 = 多头自注意力时序编码器 + [[Adaptive Average Pooling|AAP]]；对齐用 [[InfoNCE]]
- **偏好编码器训练**: [[AdamW]]，weight decay 1e-3，lr 1e-4，收敛后冻结画像编码器
- **残差解码器**: MLP + 分类动作头，输出离散（速度变化 + 转向变化）残差；[[GRPO]] 微调，每输入采 4 个 response
- **奖励生成**: [[GPT-5]] 从场景+指令推断 $(w_s,w_e,w_c)$ 与阈值（$\beta_{safety},v_{pref},\beta_{lat},\beta_{long}$），专家复核
- **硬件**: 8×NVIDIA RTX A6000，per-GPU batch=8
- **评测指标**: 驾驶分 DS、成功率 SR、效率、舒适、速度、加速度、变道数 LC、车头时距 Headway、行程时间 TT；个性化指标 [[Alignment Score|对齐分 AS]]（基于聚类的行为分类准确率）、人评 Ratings（1–10）

### 关键实验结论

- **短期风格适配（Table 1）**: DMW 把激进/保守风格拉得最开（激进效率 281.56、加速度 6.01；保守舒适 34.62、车头时距 30.05、行程 47.38 s），且 DS 高于 SimLingo/StyleDrive；风格感知奖励优于固定权重的 DMW-Vanilla。
- **长期偏好对齐（Table 3）**: 对齐分 0.83–0.92、人评 7.8–8.7，全面超过 [[MORL-PD]]（0.25–0.58 / 3.5–6.2）。
- **泛化（Table 5）**: D1–D10（含 OOD）对齐分稳定 0.67–1.00，仅凭画像即可零样本个性化未见司机。
- **消融（Table 4）**: 去掉 [[Adaptive Average Pooling|AAP]] 行为坍缩、对齐分跌到 0.25，证明其关键性。
- **用户研究（Table 6）**: 4 场景 × 3 风格下 DMW 人评一致最高。

---

## 批判性思考

### 优点
1. **"长期习惯 + 短期指令"两轴个性化设计完整**: 长期靠对比学习的用户嵌入、短期靠 LLM 生成的风格化奖励，结构清晰且各有可验证证据（Table 3 长期、Table 1/6 短期）。
2. **零样本个性化有亮点**: 画像↔行为对齐到同一空间后，仅凭新司机画像即可推断风格（OOD 对齐分 0.83），降低了"为每个新用户重训"的成本。
3. **不破坏底层安全策略**: 个性化只做**离散残差**叠加在预训练主干基础动作上，且 GRPO 奖励显式含 TTC 安全项，工程上更稳妥；Bench2Drive 还改成"碰撞即终止"以强调安全。

### 局限性
1. **纯仿真验证，无真车**: 全部在 [[CARLA]]/[[Bench2Drive]] 闭环，作者自认 sim-to-real gap 是未来工作；个性化在真实噪声/多传感器下能否成立未知。
2. **奖励依赖外部 LLM + 人工复核**: 风格权重与阈值由 [[GPT-5]] 推断再经专家复核，引入了不可完全自动化、且可能不一致的外部依赖；论文也未给出温度 $\tau$ 等若干超参的具体值。
3. **画像与对齐指标偏主观/弱**: 对齐分基于聚类行为分类、人评为小样本（5–10 评测者），且仅 4–10 名司机进入主表，统计力有限；"画像问卷→行为"的因果链更多是相关性证据。

### 潜在改进方向
1. 推进真车部署与多模态传感（非仅前视），量化 sim-to-real 下个性化的退化。
2. 把奖励权重/阈值的推断做成可学习、可校准的内生模块，减少对 GPT-5 + 人工复核的依赖。
3. 扩大司机规模与场景多样性，引入更客观的风格度量（如轨迹分布距离、表征探针）替代/补充聚类对齐分与小样本人评。

### 可复现性评估
- [x] 代码开源（项目主页 https://dmw-cvpr.github.io/ 声明 data & code available）
- [ ] 预训练模型（未明确声明权重 release）
- [x] 训练细节较完整（优化器/lr/硬件/GRPO 配置给出；个别超参如 $\tau$ 缺失）
- [x] 数据集可获取（PDD 声明随项目开放；Bench2Drive/CARLA 公开）

---

## 速查卡片

> [!summary] Drive My Way (DMW): Preference Alignment of VLA for Personalized Driving
> - **核心**: 在预训练 VLA 驾驶主干（SimLingo）上，用对比学习学"驾驶员身份嵌入"做长期偏好对齐 + LLM 生成风格化奖励 & GRPO 微调残差解码器做短期指令适配。
> - **方法**: 画像编码器(DeBERTaV3) + 路线处理器(自注意力+AAP) 经 InfoNCE 对齐出 $z_p^m$ → 拼接条件化策略；残差解码器输出离散速度/转向残差叠加到基础动作；风格感知奖励 $w_sR_{safety}+w_eR_{eff}+w_cR_{comf}$ 由 GPT-5 推断权重。
> - **结果**: Bench2Drive 闭环，DMW 拉开风格差异且 DS 最高；长期对齐分 0.83–0.92、人评 7.8–8.7（碾压 MORL-PD）；OOD 司机零样本对齐 0.83。
> - **代码/主页**: https://dmw-cvpr.github.io/

---

*笔记创建时间: 2026-06-29*
