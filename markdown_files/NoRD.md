---
title: "NoRD: A Data-Efficient Vision-Language-Action Model that Drives without Reasoning"
method_name: "NoRD"
authors: [Ishaan Rawal, Shubh Gupta, Yihan Hu, Wei Zhan]
year: 2026
venue: CVPR
tags: [VLA, autonomous-driving, reasoning-free, data-efficient, GRPO, Dr-GRPO, difficulty-bias, RL-post-training]
zotero_collection: _inbox
image_source: online
arxiv_html: https://arxiv.org/html/2602.21172v3
created: 2026-06-29
---

# NoRD: A Data-Efficient Vision-Language-Action Model that Drives without Reasoning

## 元信息

| 项目 | 内容 |
|------|------|
| 作者 | Ishaan Rawal, Shubh Gupta, Yihan Hu, Wei Zhan |
| 机构 | UC Berkeley 等（结合 Wei Zhan / Yihan Hu 的研究背景，自动驾驶方向） |
| 会议 | CVPR 2026 |
| 类别 | VLA（Vision-Language-Action）/ 端到端自动驾驶 |
| 日期 | 2026（arXiv v3） |
| 项目主页 | https://nord-vla-ai.github.io/ |
| 链接 | [arXiv](https://arxiv.org/abs/2602.21172) / [Project](https://nord-vla-ai.github.io/) |

---

## 一句话总结

> 在不要任何推理标注、仅用现有数据 60% 以下的设定下训练自动驾驶 [[VLA]]，并指出标准 [[GRPO]] 因"难度偏置"无法优化弱 SFT 策略，改用 [[Dr. GRPO]] 后在 Waymo/NAVSIM 上取得有竞争力的成绩。

---

## 核心贡献

1. **首次定位"弱 SFT + 复杂驾驶指标"下 RL 失效的根因**: 指出在小数据、无推理标注上训练的弱 SFT 策略，配合 [[PDM Score]] 这类复杂稀疏奖励时，[[GRPO]] 的失效是一种 [[Difficulty Bias|难度偏置]] 现象（此前主要在 LLM 数学推理中被观察到）。
2. **实证刻画失效机制**: 弱 SFT 策略会诱导出**两极化的组内奖励分布**——大量样本落在中等均值、**高方差**区域，而 GRPO 恰恰在这一区域几乎不更新策略，因此整体只能提升 +0.67%。
3. **用 [[Dr. GRPO]] 作为 drop-in 替换**: 去掉组内优势的标准差归一化项，让"难"场景贡献足够梯度信号；这是该算法**首次在自动驾驶域**被验证，使弱 SFT 策略提升 +11.68%。
4. **数据/Token 双高效且无推理**: 在 [[NAVSIM]] 与 [[WaymoE2E]] 上，用至少少 60% 的数据、3 倍更少 token、无推理标注即达到与 SOTA 竞争的水平，并显著降低推理时延。

---

## 问题背景

### 要解决的问题
当前自动驾驶 [[VLA]] 的主流范式是两阶段：(1) 在带 [[Chain-of-Thought|CoT]] 推理标注的大规模数据上做 SFT；(2) 用 [[GRPO]] 做 RL 后训练对齐驾驶指标。本文要回答：**能不能既"无推理（reasoning-free）"又"数据高效（data-efficient）"地训练 VLA，还能在 NAVSIM/WaymoE2E 这类困难基准上保持竞争力？**

### 现有方法的局限
作者把推理中心范式的代价归纳为三类**不可扩展成本**：
1. **数据成本**: 采集与清洗海量专业驾驶场景；
2. **标注成本**: 用 teacher 模型生成高质量推理轨迹（reasoning traces），token 量、训练时间、算力都暴涨；
3. **训练/推理成本**: 推理 token 在推理阶段带来延迟，实车部署不现实。

而一个看似简单的反面证据是：作者先用 8 万 NAVSIM 样本（比 SOTA 减少 >60% 数据）训了无推理的 **NoRD-base**，再用 GRPO 后训练优化 [[PDM Score]]——结果比基于推理的基线低 >12 分，且 GRPO 只带来 +0.67% 的"聊胜于无"提升。**这容易让人误判"推理数据是必需的"。**

### 本文的动机
作者主张这个结论**为时过早**：失效不在"无推理 SFT 策略"本身，而在**策略优化方法（GRPO）与奖励地形（reward landscape）的相互作用**。两条相关工作支撑了"无推理"假设：(1) 理论上有人质疑推理的必要性，提出"推理-规划解耦假设"（[[Reasoning-Planning Disconnect]]），认为推理可能是规划的副产品而非因果决定因素；(2) 经验上 [[EMMA]]、[[SimLingo]] 等无推理端到端模型已在常规基准上表现强劲。本文把这条路线推进到更难的 NAVSIM/WaymoE2E。

---

## 方法详解

### 模型架构

NoRD（**No R**easoning for **D**riving）是一个直接预测**动作 token**、不输出任何推理轨迹的自动驾驶 [[VLA]]（见 Figure 5）：
- **Backbone**: [[Qwen2.5-VL]]-3B-Instruct（容量与计算成本的折中）
- **输入**: 过去 ego 轨迹 + 当前速度/加速度 + 高层驾驶命令 + 前 / 前左 / 前右三路 RGB 图像
- **轨迹表示**: 用 [[k-disc Tokenization|k-disc 轨迹 token 化]]，词表大小 2048
- **输出**: 未来 ego 轨迹（10 Hz），以离散轨迹 token 形式 next-token 预测
- **训练**: 两阶段——(1) 有限数据 SFT 得到弱策略 NoRD-base；(2) 用 [[Dr. GRPO]] 做 RL 后训练得到 NoRD

整体把"学习负担"从 SFT **前移到 RL 后训练**：故意只用很少数据做 SFT，再靠 RL 把性能拉上来。

### 核心模块

#### 模块1: k-disc 轨迹 Token 化（Trajectory Tokenization）

**设计动机**: 用离散 token 直接表示轨迹，避免推理 token，最大化 token 效率，并复用语言模型的 next-token 预测范式。

**具体实现**:
- 训练集所有未来轨迹先插值到 10 Hz，再按 **0.5 秒**为单位切片；
- 这些片段依据片段间的**轮廓距离（contour distance）**聚成 **2048 个簇**，簇心构成离散轨迹码本（codebook），可重建任意轨迹；
- 这些轨迹 token 追加到 Qwen 原词表，新 embedding 用**现有 token embedding 的均值与协方差所参数化的多元正态分布**采样初始化；
- 输出形如 `TRAJ_0242 TRAJ_0150 TRAJ_0172` 的 token 串，解码为 10 Hz 的 $[x,y,\text{yaw}]$ 路点序列；消融（Table 5）显示词表 512→2048 提升明显，小词表表达不了急转弯等复杂机动。

#### 模块2: 有限数据 SFT（Supervised Fine-Tuning with Limited Data）

**设计动机**: 把学习主要交给 RL 阶段，所以**刻意**只用很少数据做 SFT，得到一个"弱"但表达力足够的 NoRD-base。

**具体实现**:
- 把轨迹预测建模为 next-token 预测，模型在输入条件下输出轨迹 token；
- NAVSIM 仅用约 8 万样本；WaymoE2E 仅约 1.2 万样本做 SFT；
- 代价是初始性能低（Table 1，NoRD-base PDMS 仅 76.66），这正是后续要靠 RL 修复的"弱策略"。

#### 模块3: 面向弱策略的 RL 后训练（RL Post-Training with Dr. GRPO）

**设计动机**: 标准 [[GRPO]] 无法优化弱 SFT 策略——根因是 [[Difficulty Bias|难度偏置]]。弱策略在大多数样本上产生**高组内方差**的 rollout，而 GRPO 的优势项分母含组内标准差，会**反向抑制**这些高方差（即"难"）样本的梯度。

**具体实现**:
- 改用 [[Dr. GRPO]]：**去掉组相对优势中的标准差归一化项**（见公式2），让难场景贡献足够梯度；
- 配合 **DAPO 式非对称裁剪**（high clip 0.1 / low clip -0.2）防止熵塌缩，并按 Liu et al. 的做法**不使用 KL 正则**；
- 与 VD-GRPO 等保留奖励绝对量级以平衡目标（安全 vs 舒适）的变体不同，Dr. GRPO 侧重保证"难"场景的梯度信号；
- 结果：NoRD-base 从中等方差样本中学到东西，整体相对基模型 +11.68%（GRPO 仅 +0.67%），甚至能学会急转弯、变道等复杂机动（Fig 4 / Fig 3(b)）。

### 关键公式与机制

#### 公式1: [[GRPO]] 组相对优势（失效的来源）

$$
\hat{A}_{i,t}^{\text{GRPO}} \coloneqq \frac{r(o_{i}\mid x) - \frac{1}{G}\sum_{j=1}^{G} r(o_{j}\mid x)}{\operatorname{std}_{j=1,\ldots,G}\big(r(o_{j}\mid x)\big)}
$$

**含义**: GRPO 用组内归一化的相对奖励作为优势。问题在于**分母的组内标准差**：当某组方差很小时优势被放大，方差大时优势被压缩——于是高方差（难）样本几乎得不到学习信号，形成 [[Difficulty Bias|难度偏置]]。

**符号说明**:
- $r(o_{i}\mid x)$: 输入 $x$ 下第 $i$ 个 rollout $o_i$ 的奖励
- $G$: 组大小（本文 rollout 数 = 8）
- $\operatorname{std}_{j=1,\ldots,G}$: 组内奖励的标准差（正是被诟病的归一化项）

#### 公式2: [[Dr. GRPO]] 优势与目标（去掉标准差项）

$$
\hat{A}_{i,t}^{\text{DrGRPO}} = r(o_{i}\mid x) - \frac{1}{G}\sum_{j=1}^{G} r(o_{j}\mid x)
$$

$$
L_{\text{DrGRPO}} = \sum_{t=1}^{|o_{i}|} \min\!\Bigg( \frac{\pi_{\theta}(o_{i,t}\mid q,o_{i,<t})}{\pi_{\theta_{\text{old}}}(o_{i,t}\mid q,o_{i,<t})}\,\hat{A}_{i,t}^{\text{DrGRPO}},\ \operatorname{clip}\!\Big(\frac{\pi_{\theta}(o_{i,t}\mid q,o_{i,<t})}{\pi_{\theta_{\text{old}}}(o_{i,t}\mid q,o_{i,<t})},\, 1-\epsilon_{\text{l}},\, 1+\epsilon_{\text{h}}\Big)\,\hat{A}_{i,t}^{\text{DrGRPO}} \Bigg)
$$

**含义**: 优势项**移除组内标准差**，使难场景不再被惩罚；目标函数沿用 PPO 式裁剪的 surrogate，但用**非对称裁剪界** $1-\epsilon_{\text l}$ / $1+\epsilon_{\text h}$（DAPO 风格）以稳定更新、抑制熵塌缩。

**符号说明**:
- $\pi_{\theta}$ / $\pi_{\theta_{\text{old}}}$: 当前 / 旧策略
- $q$: 输入查询（多视角图像 + 历史状态等）；$o_{i,<t}$: 已生成的前缀 token
- $\epsilon_{\text l}=0.2$, $\epsilon_{\text h}=0.1$: 非对称裁剪的下/上界

#### 公式3: NAVSIM 的 [[PDM Score]]（数据集奖励）

$$
\text{PDM Score} = \text{NC}\times\text{DAC}\times\frac{5\cdot\text{TTC} + 2\cdot\text{C} + 5\cdot\text{EP}}{12}
$$

**含义**: NAVSIM 的核心奖励，综合安全/合规/进度/舒适等高层驾驶指标，范围 $[0,1]$。

**符号说明**:
- $\text{NC}$: No at-fault Collision（无责碰撞）
- $\text{DAC}$: Drivable Area Compliance（可行驶区域合规）
- $\text{TTC}$: Time-to-Collision（碰撞时间）
- $\text{C}$: Comfort（舒适）；$\text{EP}$: Ego Progress（自车进度）；均 $\in[0,1]$

#### 公式4: WaymoE2E 的 Normalized [[RFS]]（数据集奖励）

$$
\text{Normalized RFS} = \frac{\max\big(\max_{r}(s_{r}),\, 4\big) - 4}{6}
$$

**含义**: 把原始 RFS（衡量预测轨迹 $\hat T$ 与三条人类评分参考轨迹的对齐度，原始范围 $[3,10]$，先取 $\max_r$、按 $t\in\{3,5\}$ 秒平均、并 $\min(\cdot,4)$ 截断后）归一化到 $[0,1]$，作为 WaymoE2E 的奖励。

**符号说明**:
- $s_{r}\in[3,10]$: 第 $r$ 条参考轨迹基于纵/横向信任域阈值打的分
- $\max_{r}(s_r)$: 取三条参考中的最高分

#### 公式5: 总奖励（组合）

$$
r = \frac{r_{f} + r_{l} + r_{d}}{1.5}
$$

**含义**: RL 总奖励由格式奖励、长度奖励、数据集专属奖励三部分组成后归一化到 $[0,1]$。

**符号说明**:
- $r_f\in\{0,0.25\}$: 格式奖励（token 须为合法 `TRAJ_i`，$i$ 为 $[0,2047]$ 的 4 位补零整数）
- $r_l\in\{0,0.25\}$: 长度奖励（NAVSIM 须 8 个 token / WaymoE2E 须 10 个）
- $r_d$: 数据集专属奖励（NAVSIM 用 PDM Score，WaymoE2E 用 Normalized RFS）

---

## 关键图表

<!-- 图片均使用 arXiv HTML 在线链接，未本地化 -->

### Figure 1: Comparison of VLA Training Pipelines / 训练范式对比

![Figure 1](https://arxiv.org/html/2602.21172v3/x1.png)

**说明**: (a) 现有范式依赖大规模推理数据生成 + 大量 SFT + RL；(b) NoRD 直接用小规模驾驶数据做 SFT，再用**面向弱 SFT 策略定制**的 RL（Dr. GRPO）后训练，实现无推理监督的数据高效学习。这张图点明了全文的"对照结构"。

### Figure 2: Reward Distribution in Weak SFT Model / 弱策略的奖励分布

![Figure 2](https://arxiv.org/html/2602.21172v3/figs/difficulty_plot.png)

**说明**: NoRD-base 的组均 PDM 分布，band 表示组内标准差均值。**关键观察**：低方差出现在组均极高（$\ge 0.8$）或极低（$\le 0.15$）的样本；**高方差出现在中等组均 $[0.2,0.65]$ 的样本，且占绝大多数**。GRPO 恰恰优化不了这片高方差区域——这是难度偏置的直接证据（绿色为真值轨迹、红色为 NoRD-base 预测）。

### Figure 3: Evolution of Group-mean PDM during RLFT (GRPO vs Dr. GRPO) / 训练中奖励分布演化

![Figure 3a GRPO](https://arxiv.org/html/2602.21172v3/figs/grpo_steps.png)
![Figure 3b Dr. GRPO](https://arxiv.org/html/2602.21172v3/figs/drgrpo_steps.png)

**说明**: (a) GRPO 训练全程，高方差区 $[0.2,0.65]$ 的密度几乎不变，只有接近 1 的最低方差区密度稳步上升——解释了 GRPO 整体仅 +0.67%；(b) Dr. GRPO 能有效优化高方差样本，带来显著整体提升。这两图是"GRPO 失效 / Dr. GRPO 修复"的机制级证据。

### Figure 4: Qualitative GRPO vs Dr. GRPO / 定性对比

![Figure 4](https://arxiv.org/html/2602.21172v3/figs/comparison_figure.png)

**说明**: 用 Dr. GRPO，NoRD 学会急转弯、变道等复杂机动且无碰撞；用 GRPO 则无法优化弱 SFT 策略并发生碰撞（红色）。直观佐证机制结论。

### Figure 5: Model Architecture of NoRD / 模型架构

![Figure 5](https://arxiv.org/html/2602.21172v3/figs/nord.png)

**说明**: NoRD 以 Qwen2.5-VL-3B 为主干，输入三路 RGB + 过去轨迹 + 速度/加速度/驾驶命令，**直接预测轨迹 token，无任何推理轨迹**，从而大幅提升训练与推理效率。

### Figure 6: Pareto-optimal Curves on Two Benchmarks / 性能-数据效率帕累托前沿

![Figure 6a NAVSIM](https://arxiv.org/html/2602.21172v3/x2.png)
![Figure 6b WaymoE2E](https://arxiv.org/html/2602.21172v3/x3.png)

**说明**: (a) NAVSIM 上 NoRD 是唯一在"高性能 + 高数据效率"区、且仅用 RGB 输入的 VLA；(b) WaymoE2E 上 NoRD 用极少训练数据、无 ensembling、无推理监督即达竞争性 RFS。阴影区为性能/效率的定性分区。这是"数据高效"主张的核心论据图。

### Figure 7: Qualitative Results on NAVSIM / NAVSIM 定性结果

![Figure 7](https://arxiv.org/html/2602.21172v3/figs/navsim_examples.png)

**说明**: NoRD 在 navtest 上安全完成急转弯、遵守红绿灯、避免碰撞（预测轨迹为红色），展示稳健驾驶行为。

### Figure 8: Qualitative Results on WaymoE2E / WaymoE2E 定性结果

![Figure 8](https://arxiv.org/html/2602.21172v3/figs/waymo_results.png)

**说明**: NoRD 在 OOD 困难场景（行人危险横穿、施工区）下安全行驶（预测轨迹红色，拼接并中心裁剪后可视化），佐证其泛化能力。

### Figure 9: Token and Runtime Efficiency / Token 与运行时效率

![Figure 9](https://arxiv.org/html/2602.21172v3/figs/nord_efficient.png)

**说明**: 因直接预测轨迹 token、无推理 token，NoRD 是 (a) token 最省、(b) 运行时最快的 VLA。支撑"3 倍更少 token、更低推理时延"的效率主张。

### Figure 10: Training Improvement Patterns across Variance Levels / 不同方差组的提升模式（附录）

![Figure 10](https://arxiv.org/html/2602.21172v3/figs/contour_plots.png)

**说明**: 相对 SFT（step 0）的组均 PDM 变化，按组内方差三分位分组。$y=x$ 表示无变化。(a) 低方差、初始分 $[0.8,1.0]$ 时 GRPO 反而占优；(b)(c) 中/高方差时 Dr. GRPO 明显在 $y=x$ 之上更密集，且**方差越高优势越大**，与"GRPO 抑制高方差更新"的论断一致。

### Figure 11: Training/Validation Curves (GRPO vs Dr. GRPO) / 训练与验证曲线（附录）

![Figure 11](https://arxiv.org/html/2602.21172v3/x4.png)

**说明**: Dr. GRPO（红）在训练集 (a) 与验证集 (b) 上都以明显幅度持续优于 GRPO（蓝）。

### Figure 12: Example of NoRD Inference (Prompt) / 推理与 prompt 示例（附录）

![Figure 12](https://arxiv.org/html/2602.21172v3/figs/prompt_example.png)

**说明**: 给定多视角图像、过去轨迹、当前速度/加速度与驾驶命令，NoRD 直接预测轨迹 token，无显式推理——展示其 token/推理效率的来源。

### Figure 13: Failure Cases / 失败案例（附录）

![Figure 13](https://arxiv.org/html/2602.21172v3/figs/failure_cases.png)

**说明**: NoRD 的代表性失败（预测轨迹红色，违规处红圈标注）。作者归因于 Dr. GRPO 仍残留难度偏置，提示用更强的"难度感知"干预可继续推高上限。

### Table 1: GRPO vs Dr. GRPO on NAVSIM (PDMS) / 核心动机实验

| Model | PDMS ↑ |
|-------|--------|
| NoRD-base | 76.66 |
| NoRD-base + GRPO | 77.18 (+0.67%) |
| **NoRD-base + Dr. GRPO** | **85.62 (+11.68%)** |

**说明**: 全文的"决定性证据"——同一弱 SFT 策略，GRPO 几乎无效（+0.67%），换成 Dr. GRPO 后大涨 +11.68%。证明瓶颈不在无推理 SFT，而在优化算法与难度偏置。

### Table 2: WaymoE2E Test Results / Waymo 端到端基准

| Model | w/o Reason | w/o Ensemble | RFS ↑ | ADE@3 ↓ |
|-------|------------|--------------|-------|---------|
| Poutine | ✗ | ✓ | **7.986** | 1.2055 |
| HMVLM | ✗ | ✓ | 7.736 | 1.3269 |
| DiffusionLTF | ✓ | ✗ | 7.717 | 1.3561 |
| UniPlan | ✓ | ✗ | 7.692 | 1.3083 |
| AutoVLA | ✗ | ✓ | 7.556 | 1.3507 |
| **NoRD** | ✓ | ✓ | 7.709 | **1.2504** |

**说明**: NoRD 是榜上唯一**既无推理又无 ensembling**的顶尖模型，RFS 7.709 排第三，且 ADE 优于所有竞品（仅 Poutine 的 RFS 更高，但用了 17× 数据）。NoRD 仅用 1.2 万 SFT + 450 RLFT 样本。

### Table 3: NAVSIM Test Results (navtest) / NAVSIM 基准

| Method | w/o R | w/o L | C | PDMS ↑ | Collision ↑ | DAC ↑ | Direction ↑ | Progress ↑ | TTC ↑ | Comfort ↑ |
|--------|-------|-------|---|--------|------------|-------|-------------|-----------|-------|-----------|
| UniAD | ✓ | ✓ | 32 | 83.4 | 97.7 | 91.9 | - | 78.8 | 92.9 | 100 |
| Transfuser | ✓ | ✗ | 3 | 84.0 | 97.7 | 92.8 | 97.9 | 79.2 | 92.8 | 100 |
| Hydra-MDP | ✓ | ✗ | 3 | 86.5 | 98.2 | 96.2 | 95.8 | 78.7 | 94.6 | 100 |
| DiffusionDrive | ✓ | ✗ | 3 | 88.1 | 98.2 | 96.2 | - | 82.2 | 94.7 | 88.1 |
| AutoVLA | ✗ | ✓ | 12 | 89.1 | 98.4 | 95.6 | 95.4 | 81.9 | 98.0 | 99.9 |
| AutoVLA-BoN* | ✗ | ✓ | 12 | 92.1 | 99.1 | 97.1 | 95.5 | 87.6 | 97.1 | 100 |
| RecogDrive | ✗ | ✓ | 12 | 89.6 | 98.2 | 97.9 | - | 83.5 | 95.2 | 99.8 |
| NoRD | ✓ | ✓ | 3 | 85.6 | 97.6 | 94.9 | 95.9 | 79.3 | 93.5 | 100 |
| **NoRD-BoN*** | ✓ | ✓ | 3 | **92.4** | **99.2** | **98.3** | 95.9 | 86.4 | 97.8 | 99.9 |

**说明**: NoRD 是唯一**无推理、仅 3 帧 RGB、无 LiDAR/HD Map**的方法。单样本 PDMS 85.6 已具竞争力；BoN（6 选 1 oracle）下 NoRD-BoN 达 92.4，**超过基于推理的 AutoVLA-BoN(92.1)**，而 AutoVLA/RecogDrive 分别用了 1.6× / 34× 更多数据。（w/o R=无推理数据，w/o L=无 LiDAR，C=RGB 帧数，*BoN=6 个随机种子里每样本取最优分再平均。）

### Table 4: Component-wise GRPO vs Dr. GRPO on NAVSIM / 分项对比（附录）

| Method | PDMS ↑ | Collision ↑ | DAC ↑ | Direction ↑ | Progress ↑ | TTC ↑ | Comfort ↑ |
|--------|--------|------------|-------|-------------|-----------|-------|-----------|
| NoRD-base | 76.66 | 96.45 | 86.37 | 94.62 | 71.58 | 90.37 | 99.97 |
| NoRD-base + GRPO | 77.18 | 91.89 | 90.12 | 91.84 | **80.06** | 80.13 | 99.96 |
| **NoRD-base + Dr. GRPO** | **85.62** | **97.56** | **94.92** | **95.94** | 79.30 | **93.53** | **100** |

**说明**: 除 Ego Progress 外，Dr. GRPO 在所有分项上都明显优于 GRPO；尤其 GRPO 把 Collision/TTC 训得更差（91.89/80.13），说明它对难场景的优化是反效果的。

### Table 5: Effect of k-disc Vocabulary Size / 词表大小消融（附录）

| Vocabulary Size | PDMS ↑ |
|-----------------|--------|
| 512 | 83.07 |
| **2048** | **85.62** |

**说明**: 词表 512→2048 提升约 2.55 分；小词表无法忠实表示急转弯等复杂机动，故采用 2048。

### Table 6: Detailed WaymoE2E Test Scores / Waymo 分场景细分（附录）

| Metric | Value ↑ |
|--------|---------|
| Construction Score | 8.073 |
| Intersection Score | 7.925 |
| Pedestrian Score | 7.778 |
| Cyclist Score | 7.806 |
| Multi Lane Maneuver Score | 7.826 |
| Single Lane Maneuver Score | 8.309 |
| Cut In Score | 7.735 |
| Foreign Object Debris Score | 7.699 |
| Special Vehicle Score | 7.796 |
| Spotlight Score | 6.531 |
| Others Score | 7.323 |
| ADE @ 3 s | 1.250 |
| ADE @ 5 s | 2.893 |
| **Average Score** | **7.709** |

**说明**: 分场景看，NoRD 在多车道/单车道机动、施工区、交叉口等都较强；Spotlight 场景（6.531）最弱，是后续改进空间。

---

## 实验

### 数据集 / 基准

| 基准 | 规模/划分 | 特点 | NoRD 用量 |
|------|-----------|------|-----------|
| [[NAVSIM]] | OpenScene 重整，120 小时驾驶；预测未来 4 秒 @2Hz | 360° 相机+LiDAR+HD Map+动态体框；模拟器执行 [[PDM Score]] 打分 | SFT ~8 万；RLFT 过滤+均衡（直/左/右）后子集 |
| [[WaymoE2E]] | 4021 段（2037 训/479 验/1505 测）；长尾 | 360° 相机；三条人类评分参考轨迹，用 [[RFS]] 评分 | SFT ~1.2 万（20% 采样）；RLFT 450 |

### 实现细节

- **Backbone**: Qwen2.5-VL-3B-Instruct，全层微调（vision encoder + 多模态 MLP + LM）
- **SFT**: 16× A100，bf16 混合精度 + gradient checkpointing；lr $5\times10^{-5}$、cosine + warmup 0.03、grad clip 1；每设备 batch 8 × 4 梯度累积；WaymoE2E 用 DeepSpeed ZeRO-3；每 50 步按最低验证 loss 选模型
- **RL（Dr. GRPO）**: NAVSIM 30× A100 / 160 步 / lr $5\times10^{-6}$；WaymoE2E 32× A100 / 150 步 / lr $1\times10^{-6}$；group size 8、采样温度 1.0（验证 0.01）；非对称裁剪 high 0.1 / low -0.2；无 KL 正则；batch 128(NAVSIM)/256(WaymoE2E)
- **框架**: RL 在 [[verl]] 实现，FSDP 省显存，[[vLLM]] 做 rollout
- **输入配置**: NAVSIM 三相机帧 + 3 个过去轨迹 token(1.5s) + 速度/加速度/驾驶命令，预测 8 个 token(4s@10Hz)；WaymoE2E 6 个过去 token(3s)，预测 10 个 token(5s)

### 关键实验结论

- **NAVSIM**: NoRD 单样本 PDMS 85.6（唯一无推理/仅 3 RGB/无 LiDAR），NoRD-BoN 92.4 超过 AutoVLA-BoN(92.1)，而对手用 1.6×–34× 更多数据。
- **WaymoE2E**: RFS 7.709（第三），ADE 最优，唯一无推理+无 ensembling 的顶尖模型，数据量仅同类的约 1/6–1/17。
- **效率（Fig 9）**: token 与运行时双第一。
- **机制（Table 1/4，Fig 2/3/10/11）**: GRPO 仅 +0.67%、Dr. GRPO +11.68%；失效根因是难度偏置，高方差样本得不到学习信号。
- **消融（Table 5）**: 词表 2048 优于 512。

---

## 批判性思考

### 优点
1. **机制诊断扎实**: 不止"换个算法有效"，而是用奖励分布（Fig 2）、训练演化（Fig 3）、方差分组的提升模式（Fig 10）层层论证 GRPO 失效=难度偏置，论证链完整且可证伪。
2. **效率主张有多维支撑**: 数据量（<60%、3×–34× 更少）、token/推理时延（Fig 9）、性能（Table 2/3）三者同时给出，Pareto 前沿（Fig 6）把"高效"具体化为可比的坐标。
3. **结论有反直觉价值**: BoN 下无推理的 NoRD 反超基于推理的 AutoVLA，对"驾驶 VLA 必须大数据+推理"的惯例构成实证挑战。

### 局限性
1. **靠 Dr. GRPO 是 drop-in，新意主要在"诊断+迁移"**: Dr. GRPO 本身来自 LLM 推理领域，本文的算法贡献偏"应用与验证"，而非新算法；作者也承认 Dr. GRPO 仍残留难度偏置（Conclusion / Fig 13）。
2. **BoN 比较的公平性**: NAVSIM 上最亮眼的 92.4 来自 6 选 1 的 oracle BoN，依赖仿真器可拿到 PDM 真分来选轨迹，实车不可得；单样本 85.6 其实低于多个推理基线（AutoVLA 89.1 等）。
3. **数据规模对比为"估算"**: 各基线训练样本数由作者按论文配置**估算**（附录 12），3×/34× 等倍数结论对估算口径敏感。
4. **机构/泛化范围有限**: 仅前视三路相机、无 LiDAR/HD Map 是效率优势也是上限来源；Spotlight 等场景明显偏弱（Table 6），长尾鲁棒性仍有缺口。

### 潜在改进方向
1. 设计**更彻底的难度感知优化**（如显式按组内方差/难度重加权、课程式 RL），把 Fig 13 的残留失败补齐。
2. 把"无推理"与"轻量推理"做可控对比，量化推理-规划解耦假设在困难基准上的边界。
3. 引入 LiDAR/HD Map 或更多视角做"效率-性能"可调档位，并在闭环/实车上验证（当前为非反应式模拟评测）。
4. 用更严格统一的数据规模口径替代估算，巩固数据效率结论。

### 可复现性评估
- [ ] 代码开源（仅给出项目主页 https://nord-vla-ai.github.io/，未在论文中确认放出代码/权重）
- [ ] 预训练模型（未明确）
- [x] 训练细节完整（附录 11 给出 SFT/RL 全套超参、GPU 数、裁剪界、奖励函数）
- [x] 数据集可获取（NAVSIM、WaymoE2E 均为公开基准）

---

## 速查卡片

> [!summary] NoRD: Data-Efficient VLA that Drives without Reasoning
> - **核心**: 无推理标注 + <60% 数据训练驾驶 VLA；指出 GRPO 因难度偏置无法优化弱 SFT 策略，改用 Dr. GRPO 修复。
> - **方法**: Qwen2.5-VL-3B + k-disc 轨迹 token(2048) → 有限数据 SFT 得弱策略 NoRD-base → Dr. GRPO（去标准差归一 + DAPO 非对称裁剪 + 无 KL）后训练。
> - **结果**: NAVSIM PDMS 85.6 / BoN 92.4（超 AutoVLA-BoN），WaymoE2E RFS 7.709 + ADE 最优；GRPO +0.67% vs Dr. GRPO +11.68%；token/推理时延双第一。
> - **项目**: https://nord-vla-ai.github.io/

---

*笔记创建时间: 2026-06-29*
