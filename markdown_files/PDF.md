---
title: "Test-Time Perturbation Learning with Delayed Feedback for Vision-Language-Action Models"
method_name: "PDF"
authors: [Zehua Zang, Xi Wang, Fuchun Sun, Xiao Xu, Lixiang Lium, Jiahuan Zhou, Jiangmeng Li]
year: 2026
venue: CVPR
tags: [VLA, test-time-adaptation, trajectory-overfitting, data-augmentation, REINFORCE, delayed-feedback]
zotero_collection: _inbox
image_source: online
arxiv_html: https://arxiv.org/html/2604.18107v1
created: 2026-06-29
---

# Test-Time Perturbation Learning with Delayed Feedback for Vision-Language-Action Models

## 元信息

| 项目 | 内容 |
|------|------|
| 作者 | Zehua Zang, Xi Wang, Fuchun Sun, Xiao Xu, Lixiang Lium, Jiahuan Zhou, Jiangmeng Li |
| 机构 | 论文未在正文显式标注；通讯/代码归属 zhoujiahuan1991（北京大学方向团队） |
| 会议 | CVPR 2026 |
| 类别 | VLA（Vision-Language-Action）/ 测试时自适应 |
| 日期 | 2026（arXiv v1） |
| 项目主页 | https://github.com/zhoujiahuan1991/CVPR2026-PDF |
| 链接 | [arXiv](https://arxiv.org/abs/2604.18107) / [Code](https://github.com/zhoujiahuan1991/CVPR2026-PDF) |

---

## 一句话总结

> 提出无需验证器、无需微调基座的测试时自适应框架 PDF：用不确定性驱动的数据增强+动作投票打散“轨迹过拟合”的伪相关，再用延迟反馈训练一个仅 9M 的轻量扰动头修正过自信，在 LIBERO(+7.4% SR) 与 Atari(+10.3 HNS) 上稳定提升。

---

## 核心贡献

1. **系统刻画 [[Trajectory Overfitting|轨迹过拟合]] 现象**: 指出 VLA 脆弱性的根因不是噪声而是依赖“动作-实体”的伪相关（spurious correlation）——模型记住训练轨迹里与成功相关的视觉/上下文模式（夹爪外观、背景纹理），在相似输入下机械复现记忆动作，而非基于任务语义决策。即使目标物体被遮挡，机械臂仍重复同样的运动轨迹。
2. **提出 [[PDF]]（Perturbation learning with Delayed Feedback）**: 一个 **verifier-free** 的测试时自适应（[[Test-Time Adaptation|TTA]]）框架，**冻结全部 VLA 参数**，只通过(a) 基于不确定性的动作投票 与 (b) 延迟反馈引导的扰动学习提升决策性能，仅训练 9M 参数的扰动头，且不需访问基座梯度。
3. **跨基准一致增益**: 在 [[LIBERO]] 机器人操作（平均 SR 0.77，mean rank 2.5，长程任务 +4.1）与 [[Atari]]-57 视觉控制（HNS 1.07 vs 基座 0.97，47/57 游戏提升）上均超过 vanilla VLA 与现有 TTA 基线。

---

## 问题背景

### 要解决的问题
VLA 在序列决策上表现强，但对**微小、语义无关的环境扰动**（如物体位姿轻微变化）极度脆弱，常导致性能骤降或任务失败。如何在**不微调基座模型**的前提下，于测试时在线提升 VLA 的决策鲁棒性？

### 现有方法的局限
作者把脆弱性归因于 [[Trajectory Overfitting|轨迹过拟合]]：VLA 过度关注“动作-实体”的伪相关，复现记忆中的动作链而非真正理解任务语义。针对性的 TTA 路线各有缺陷：
1. **验证器式 TTA（verifier-based）**: 用预训练验证器给候选动作打分、best-of-$N$ 投票选优。效果好，但验证器预训练与反复 rollout 计算开销巨大，且难以泛化到动态/视觉统计不同的未见环境。
2. **无验证器 TTA（verifier-free）**: 仅用无标注测试样本实时自适应。但缺乏真值反馈，在严重分布偏移下**自适应不稳定、精度欠佳**。其核心问题在于依赖**自监督置信度指标**（如熵最小化）——当模型预测本身被错误校准（miscalibrated）时，这类指标变得不可靠。

如 Figure 2 所示，当 VLA 对**错误行为过自信**时，基于熵最小化的 TTA 反而会进一步抬高错误 logits，放大错误。

### 本文的动机
- 既然轨迹过拟合源于对固定输入分布的记忆，**扰动输入分布**（数据增强）可让夹爪重新聚焦目标、恢复正确执行（Figure 1d 给出经验证据）——这启发“用扰动学习替代昂贵微调”。
- 增强有计算开销，应**按需分配**：用决策不确定性决定增强预算，兼顾性能与效率。
- 自监督信号不可靠 → 改用环境天然给出的**延迟反馈**（episode 结束后的成功/失败/累计奖励）作为监督，训练一个轻量扰动头**事后修正** logits，纠正过自信，同时保持基座冻结以稳定、高效。

---

## 方法详解

### 模型架构

[[PDF]] 是一个即插即用（plug-and-play）的 TTA 策略，整体框架见 Figure 3：
- **输入**: 像素观测 $o_t$ + 文本指令 $c_t$，构成多模态状态 $s_t=(o_t,c_t)$
- **基座（全程冻结）**: 视觉编码器 + token 嵌入层 + causal transformer + **LM head** $h_\phi(\cdot)$
- **新增可学习模块**: **扰动头（P head）** $h_\theta(\cdot)$，仅 9M 参数，输出对 logits 的可学习扰动
- **核心机制**: [[Uncertainty-Based Action Voting|基于不确定性的动作投票]] + [[Delayed Feedback|延迟反馈]]引导的扰动学习
- **输出**: 经多数投票（majority voting）选出的最终动作 $a_t$
- **缓冲**: rollout buffer $\mathcal{D}$ 存储原始/增强观测的特征与投票后 logits，episode 结束有反馈后再优化

整个流程**只更新 P head**，VLA（含视觉编码器、嵌入层、transformer、LM head、P head 的其余部分）保持冻结。

### 核心模块

#### 模块1: Uncertainty-Based Action Voting（基于不确定性的动作投票）

**设计动机**: 用数据增强打散轨迹过拟合与注意力失衡，提升动作生成可靠性；但增强开销大，需按不确定性**自适应分配预算**，避免无谓计算。

**具体实现**:
- 把 [[MDP]] 形式化为 $M=(\mathcal{S},\mathcal{A},P,R)$，状态空间 $\mathcal{S}\subseteq\mathbb{R}^n$、动作空间 $\mathcal{A}\subseteq\mathbb{R}^m$ 均连续；每步观测多模态状态 $s_t=(o_t,c_t)$，任务完成后才收到延迟反馈 $r_i\in\mathcal{R}$。
- 像素 $o_t$ 经视觉编码器得 $e_{o_t}$，指令 $c_t$ 分词嵌入得 $e_{c_t}$，拼接投影为特征 $f_t$，再经 LM head $h_\phi(\cdot)$ 产出每个 token 的 logits $z_t$。
- 用预测动作分布的**归一化香农熵**估计决策不确定性 $\mathcal{U}_t$（公式1）。
- 增强预算 $N_t$ 与 $\mathcal{U}_t$ **成正比**（公式2）：越不确定，越多增强视图。对 $o_t$ 施加变换集合 $\mathcal{T}=\{T_1,\dots,T_N\}$ 得增强观测 $\{T_j(o_t)\}$。
- 原始与增强观测的特征都经 LM head 与 P head 双头融合得最终 logits $\tilde z_t$（公式3），反 token 化为候选动作集，最终动作由**多数投票**（实验表明 **dim-wise 维度级投票**最优，见 4.6）选出。

#### 模块2: Delayed Feedback-Guided Adaptation（延迟反馈引导的自适应）

**设计动机**: 无真值时自监督置信度不可靠，转而用环境在 episode 结束才给的**延迟反馈**作监督，训练 P head 修正过自信、抬高正确动作 logits。

**具体实现**:
- episode 结束收到延迟反馈 $r\in\mathcal{R}$（成功/失败/累计奖励）后，从 buffer $\mathcal{D}$ 采一批特征 $f_b$，带梯度计算最终 logits $\tilde z_b$（公式4），softmax 得策略 $\tilde\pi_b$。
- 用 PDF 损失（公式5）优化 P head：第一项是 **REINFORCE-style** 项，当反馈 $r$ 超过基线 $b$ 时提升扰动策略动作的似然；第二项是 **KL 正则**，由 $\mathbb{I}[r>b]$ 门控——**仅在正反馈时启用**以稳定更新、加速收敛，负反馈时放开探索。
- 扰动因此被导向成功动作方向，并在失败方向被抑制。**只更新 $h_\theta$**，其余全冻结，实现高效稳定的 TTA。

### 关键公式与机制

#### 公式1: [[Uncertainty Estimation|动作分布不确定性（归一化熵）]]

$$
\mathcal{U}_{t}=-\frac{1}{\log K}\sum_{k=1}^{K}p(a_{k}\mid s_{k})\log p(a_{k}\mid s_{k})
$$

**含义**: 用预测动作分布的归一化香农熵量化当前状态下决策的不确定性，作为后续分配增强预算的依据。

**符号说明**:
- $K$: 可能的动作 token 数量
- $p(a_t=a_k\mid s_t)=\mathrm{softmax}(z_t)_k$: 第 $k$ 个动作 token 的预测概率
- $\frac{1}{\log K}$: 归一化因子，把熵约束到 $[0,1]$

#### 公式2: [[Adaptive Augmentation Budget|自适应增强预算]]

$$
N_{t}=N_{\max}\cdot\mathcal{U}_{t}
$$

**含义**: 增强视图数量随不确定性线性增长——越不确定的决策分配越多增强以更可靠地投票，越确定则省算力。

**符号说明**:
- $N_{\max}$: 最大增强预算（实验设为 3）
- $\mathcal{U}_t\in[0,1]$: 公式1 的不确定性

#### 公式3: [[Logit Perturbation|双头融合最终 logits（推理）]]

$$
\tilde{z}_{t}=h_{\phi}(f_{t})+\lambda\, h_{\theta}(f_{t})
$$

**含义**: 在原决策 logits 上叠加可学习扰动，得到修正后的最终 logits，再反 token 化为候选动作。

**符号说明**:
- $h_\phi(\cdot)$: 冻结的 LM head，产生决策 logits
- $h_\theta(\cdot)$: 可学习 P head，产生 logit 扰动
- $\lambda$: 控制扰动强度的超参数；$f_t$: 多模态融合特征

#### 公式4: [[Logit Perturbation|带梯度的批最终 logits（优化）]]

$$
\tilde{z}_{b}=h_{\phi}(f_{b})+\lambda\, h_{\theta}(f_{b})
$$

**含义**: 与公式3 同构，但作用于从 buffer 采样的一批特征 $f_b$ 且**保留梯度**，用于 episode 结束后反传更新 P head。

**符号说明**:
- $f_b$: 从 rollout buffer $\mathcal{D}$ 采样的一批特征
- $\tilde z_b$: 该批的最终 logits，softmax 后得策略 $\tilde\pi_b$

#### 公式5: [[PDF Loss|PDF 损失（REINFORCE + 门控 KL）]]

$$
\mathcal{L}_{\mathrm{PDF}}=-(r-b)\log\pi_{\phi}+\lambda_{\mathrm{KL}}\,\mathbb{I}[r>b]\,\operatorname{KL}(\pi_{\phi}\parallel\tilde{\pi})
$$

**含义**: 第一项为 REINFORCE 式策略梯度，当延迟反馈优于基线时提升对应动作似然；第二项 KL 正则仅在正反馈（$r>b$）时启用，约束扰动策略与基策略不过度偏离，稳定收敛、加速并允许 $r\le b$ 时灵活探索。

**符号说明**:
- $r$: episode 延迟反馈；$b$: 基线
- $\pi_\phi$: 基策略（LM head softmax）；$\tilde\pi$: 扰动后策略
- $\mathbb{I}[r>b]$: 仅正反馈门控的指示函数；$\lambda_{\mathrm{KL}}$: KL 权重
- $\operatorname{KL}(\cdot\parallel\cdot)$: KL 散度

---

## 关键图表

<!-- 图片均使用 arXiv HTML 在线链接，未本地化 -->

### Figure 1: Evidence of Trajectory Overfitting / 轨迹过拟合证据

![Figure 1](https://arxiv.org/html/2604.18107v1/x1.png)

**说明**: 轨迹过拟合的诊断证据与数据增强的有效性。(a) 让抓碗时机械臂无视任务成败仍朝盘子运动；(b) 目标碗被遮挡，仍复现相似动作，说明策略未 grounding 到目标视觉语义；(c) 注意力图显示夹爪忽视目标，记住的是夹爪外观/背景纹理等与成功相关的伪相关模式；(d) 施加数据增强后夹爪重新聚焦目标并做出正确决策——这是“扰动输入分布可缓解轨迹过拟合”的核心直觉来源。

### Figure 2: TTA vs PDF on Logits / 传统 TTA 与 PDF 的 logits 对比

![Figure 2](https://arxiv.org/html/2604.18107v1/x2.png)

**说明**: 虚线条=自适应前 logits，实线条=自适应后；红=错误动作 logits，绿=正确动作 logits。当 VLA 对错误行为过自信时，熵最小化 TTA 会进一步抬高错误 logits、放大错误；PDF 则同时抑制过自信并抬高正确动作 logits，引导模型走向正确决策。直观说明本文为何弃用自监督置信度、改用延迟反馈。

### Figure 3: Overall Framework of PDF / PDF 整体框架

![Figure 3](https://arxiv.org/html/2604.18107v1/x3.png)

**说明**: 测试时 VLA 接收 $o_t,c_t$，估计动作-logit 不确定性 $\mathcal{U}_t$ 以分配自适应增强预算 $N_t$；原始与增强观测共同编码为特征 $f_t$，分别经 LM head 产生决策 logits、经 P head $h_\theta(\cdot)$ 产生 logit 扰动；最终 logits 给出候选动作，经多数投票选定 $a_t$。特征与投票后 logits 存入 buffer $\mathcal{D}$；每个 episode 后采样特征+反馈计算 PDF 损失（公式5），**仅更新 P head**。

### Figure 4: Atari-57 Human Normalized Score Changes / Atari 57 游戏 HNS 变化

![Figure 4](https://arxiv.org/html/2604.18107v1/x4.png)

**说明**: 57 个 Atari 游戏的 HNS 变化（蓝=提升、橙=退化，按提升幅度排序）。BOXING 最大增益 +60.25%，BATTLE ZONE 最大下降 -10.72%；**47/57 游戏正向提升**，平均提升 11.28%。佐证 PDF 在离散视觉控制上的普适增益。

### Figure 5: Effect of Augmentation Budget / 增强预算消融

![Figure 5](https://arxiv.org/html/2604.18107v1/x5.png)

**说明**: 五个基准（LIBERO-Spatial/Object/Goal/Long、Atari）在增强预算从 0 增到 4 时性能普遍下降，说明**过度数据增强反而有害**（噪声累积、低质量视图）。据此把最大增强预算设为 3，并建议按基准自适应设上限而非一味最大化。

### Figure 6: Loss Function Ablation / 损失项消融

![Figure 6](https://arxiv.org/html/2604.18107v1/x6.png)

**说明**: 对比基座 OpenVLA、去 KL（w/o KL）、去 REINFORCE 项（w/o RE）与完整 PDF。完整目标得分最高，Atari/LIBERO-Object 增益最大（约 +0.1）；KL 项与 REINFORCE 项提供互补收益（稳定+校准探索 vs 奖励对齐的动作修正），组合带来可加、非冗余的提升，且 KL 项对最终结果影响更大。

### Figure 7: Visualization & Case Study / 可视化与案例分析

![Figure 7](https://arxiv.org/html/2604.18107v1/x7.png)

**说明**: 三个 LIBERO 任务上 OpenVLA 与 PDF 的对比（绿/红拇指=动作正确/错误，方框=目标实体）。OpenVLA 表现轨迹过拟合：执行示范式动作却无视是否抓住目标而失败；PDF 更具状态感知、可靠完成任务；在 Put alphabet soup 上 PDF 初次失败后能立即重试并成功，说明其是**闭环决策而非开环回放**。

### Table 1: LIBERO Benchmark Comparison / LIBERO 基准对比

| Method | Pub. | Param. | Spatial SR | Object SR | Goal SR | Long SR | Avg. SR | Mean Rank |
|--------|------|--------|-----------|-----------|---------|---------|---------|-----------|
| PackNet | CVPR'18 | - | 0.63 | 0.60 | 0.75 | 0.25 | 0.56 | 10.2 |
| ER | Arxiv'19 | - | 0.56 | 0.44 | 0.49 | 0.32 | 0.45 | 11.5 |
| SeqL | NeurIPS'23 | - | 0.20 | 0.26 | 0.22 | 0.15 | 0.21 | 12.8 |
| MTL | NeurIPS'23 | - | 0.83 | 0.54 | 0.80 | 0.48 | 0.66 | 7.2 |
| ATM | RSS'23 | - | 0.69 | 0.68 | 0.78 | 0.39 | 0.63 | 10.5 |
| OpenVLA | CoRL'24 | - | 0.79 | 0.86 | 0.85 | 0.51 | 0.75 | 5.2 |
| OpenVLA† | CoRL'24 | - | 0.85 | 0.64 | 0.76 | 0.53 | 0.69 | 5.8 |
| DP | IJRR'25 | - | 0.78 | **0.92** | 0.68 | 0.51 | 0.72 | 7.0 |
| OCTO | RSS'24 | 93M | 0.79 | 0.86 | 0.85 | 0.51 | 0.75 | 5.3 |
| TraceVLA | ICLR'25 | 130M | 0.85 | 0.85 | 0.75 | 0.54 | 0.75 | 6.5 |
| OpenVLA-DPO | Arxiv'25 | 130M | 0.84 | 0.89 | 0.79 | 0.53 | 0.76 | 4.0 |
| SFT-4LIBERO | Arxiv'25 | 130M | 0.85 | 0.87 | 0.77 | 0.55 | 0.76 | 3.5 |
| MG-Select | Arxiv'25 | 130M | 0.82 | 0.73 | 0.73 | 0.55 | 0.71 | 6.0 |
| **PDF (Ours)** | - | **9M** | **0.90** | 0.72 | **0.86** | **0.59** | **0.77** | **2.5** |

**说明**: PDF 以仅 **9M** 可训练参数（基线 93–130M 且需基座梯度）取得最佳平均 SR(0.77) 与 mean rank(2.5)；Spatial 0.90、Goal 0.86、Long 0.59（长程 +4.1 over best baseline）均居首；相比 MG-Select 平均 SR 约 +6 个点。Object 上略逊于 DP/OpenVLA-DPO，但整体最稳。

### Table 2: Ablation of DF & DA across LIBERO Suites / DF 与 DA 消融（各 10 任务逐任务 SR）

| Suite | Method | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | Avg. |
|-------|--------|---|---|---|---|---|---|---|---|---|---|------|
| Spatial | OpenVLA | 0.94 | 0.94 | 0.86 | 0.96 | 0.64 | 0.88 | 0.92 | 0.86 | 0.74 | 0.76 | 0.85 |
| Spatial | PDF w/o DF | 0.78 | 0.84 | 0.90 | 0.90 | 0.78 | 0.68 | 0.92 | 0.70 | 0.80 | 0.74 | 0.80 |
| Spatial | PDF w/o DA | 0.94 | 0.94 | 0.88 | 0.94 | 0.68 | 0.88 | 0.92 | 0.88 | 0.76 | 0.76 | 0.86 |
| Spatial | **PDF** | 0.94 | 0.96 | 0.98 | 0.98 | 0.80 | 0.88 | 0.92 | 0.88 | 0.80 | 0.80 | **0.89** |
| Object | OpenVLA | 0.64 | 0.68 | 0.78 | 0.34 | 0.90 | 0.66 | 0.50 | 0.62 | 0.36 | 0.74 | 0.62 |
| Object | PDF w/o DF | 0.86 | 0.50 | 0.74 | 0.62 | 0.68 | 0.16 | 0.10 | 0.28 | 0.36 | 0.74 | 0.50 |
| Object | PDF w/o DA | 0.88 | 0.68 | 0.82 | 0.40 | 0.94 | 0.70 | 0.48 | 0.66 | 0.36 | 0.76 | 0.67 |
| Object | **PDF** | 0.96 | 0.74 | 0.82 | 0.66 | 0.96 | 0.66 | 0.56 | 0.66 | 0.40 | 0.80 | **0.72** |
| Goal | OpenVLA | 0.62 | 0.90 | 0.84 | 0.66 | 0.94 | 0.90 | 0.78 | 1.00 | 0.86 | 0.68 | 0.81 |
| Goal | PDF w/o DF | 0.70 | 0.96 | 0.88 | 0.34 | 0.94 | 0.78 | 0.82 | 0.98 | 0.68 | 0.62 | 0.77 |
| Goal | PDF w/o DA | 0.62 | 0.94 | 0.86 | 0.60 | 0.92 | 0.94 | 0.80 | 1.00 | 0.76 | 0.66 | 0.81 |
| Goal | **PDF** | 0.72 | 0.98 | 0.90 | 0.66 | 0.96 | 0.96 | 0.86 | 1.00 | 0.82 | 0.70 | **0.85** |
| Long | OpenVLA | 0.62 | 0.64 | 0.58 | 0.50 | 0.44 | 0.86 | 0.50 | 0.66 | 0.20 | 0.58 | 0.56 |
| Long | PDF w/o DF | 0.70 | 0.64 | 0.58 | 0.54 | 0.52 | 0.82 | 0.48 | 0.62 | 0.20 | 0.52 | 0.56 |
| Long | PDF w/o DA | 0.72 | 0.60 | 0.58 | 0.50 | 0.46 | 0.88 | 0.50 | 0.66 | 0.20 | 0.56 | 0.57 |
| Long | **PDF** | 0.72 | 0.64 | 0.58 | 0.56 | 0.56 | 0.88 | 0.50 | 0.68 | 0.20 | 0.58 | **0.59** |

**说明**: “w/o DF”=仅多视图投票不含延迟反馈；“w/o DA”=仅用原始观测学扰动不含数据增强。完整 PDF 在四个 suite 平均 SR 全部最高。**DF 更关键**：去掉 DF 后 Object 跌到 0.50、Goal 跌到 0.77；去掉 DA 仍多数任务优于 OpenVLA。两者协同（DF+DA）带来最稳、最高的增益。

### Table 3: Dim-wise vs Action-wise Voting / 维度级与动作级投票对比

| Task | Dim-wise | Action-wise |
|------|----------|-------------|
| Alien (Atari) | 0.26 | - |
| Black Bowl | 0.68 | **0.88** |
| Alphabet Soup | **0.86** | 0.64 |
| Bowl on Stove | **0.96** | 0.90 |
| Moka Pot | 0.58 | 0.58 |

**说明**: 在 Atari(Alien) 动作近一维，两种投票等价。LIBERO 上 **dim-wise（逐动作维度多数投票）** 总体更可靠：允许各视图在单个动作分量上达成一致，能更灵活地偏离原策略以缓解轨迹过拟合；action-wise（对完整动作元组投票）因各视图很难在整元组上一致，常回退到基线、结果参差。综合 dim-wise 提供最佳鲁棒性-可塑性权衡。

### Table 4: Atari-57 Full Results / Atari 57 全量结果（附录）

**说明**: 附录 Table 4 给出 57 个 Atari 游戏的逐游戏评测分数（对应 Figure 4 的 per-game HNS 变化的数值来源）。整体 PDF 取得 HNS 1.07，相对 Jat 基座 0.97 提升约 +0.10，47/57 游戏正向提升。

### Algorithm 1: Perturbation Learning with Delayed Feedback / PDF 伪代码（附录）

**说明**: 附录第 6 节给出 PDF 完整伪代码：测试时对每步估计不确定性→自适应增强→双头融合 logits→多数投票得动作→特征/logits 入 buffer；episode 结束收到延迟反馈后，采样 batch 按公式5 仅更新 P head $h_\theta$，全程冻结 VLA 其余参数。

---

## 实验

### 数据集 / 基准

| 基准 | 规模 | 特点 | 用途 |
|------|------|------|------|
| [[LIBERO]] | 4 个 10-task 套件（Spatial/Object/Goal/Long），每任务 50 次评测 rollout | Franka Panda 仿真，RGB 视图+机器人状态+任务文本+delta 末端执行器动作；成功率为指标 | 训练/测试（评测前用 OpenVLA 训 50 episode） |
| [[Atari]] | 全部 57 个 Atari 2600 游戏，每游戏 50 次评测 episode | 像素输入，4–18 个离散动作；episode 在丢命/通关/108000 帧结束；HNS 为指标 | 训练/测试（基座 Jat 每游戏训 50 episode） |

HNS 定义: $HNS=(\mathrm{Score}-\mathrm{RandomScore})/(\mathrm{HumanScore}-\mathrm{RandomScore})$。

### 实现细节

- **基座**: LIBERO 用 [[OpenVLA]]，Atari 用 [[Jat]]；两者参数全程冻结。
- **可训练**: 仅 P head（9M 参数），无需访问基座梯度（对比基线 93–130M 且需基座梯度）。
- **关键超参**: 最大增强预算 $N_{\max}=3$（>3 性能下降）；增强预算随不确定性自适应分配。
- **优化触发**: 每个 episode 结束、收到延迟反馈后，从 rollout buffer 采 batch 按公式5 更新 P head。
- **硬件**: 单卡 Tesla V100-PCIE-32GB（LIBERO 与 Atari 均同）。

### 关键实验结论

- **LIBERO（Table 1）**: 平均 SR 0.77、mean rank 2.5 居首；Spatial 0.90、Goal 0.86、Long 0.59 最佳；相比 MG-Select 平均约 +6 点；无需任何微调即匹敌/超过 TraceVLA、SFT-4LIBERO 等强预训练模型。
- **Atari-57**: HNS 1.07 vs Jat 基座 0.97（+0.10），47/57 游戏提升，BOXING +0.60、TIME PILOT +0.53、ATLANTIS +0.49、ASSAULT +0.45。
- **组件消融（Table 2）**: DF 是鲁棒性关键（去掉后 Object 跌至 0.50），DA 普遍小幅有益，DF+DA 协同最优、最稳。
- **增强预算（Fig 5）**: 预算 0→4 性能普降，max=3 为最佳权衡（噪声累积导致过度增强有害）。
- **损失消融（Fig 6）**: REINFORCE 项与 KL 项互补可加，KL 影响更大。
- **投票机制（Table 3）**: dim-wise 维度级投票优于 action-wise，鲁棒性-可塑性权衡最佳。
- **案例（Fig 7）**: PDF 表现闭环、状态感知决策，失败后能重试纠正，而非开环回放示范。

---

## 批判性思考

### 优点
1. **轻量且解耦基座**: 仅训练 9M 的 P head、不微调也不需基座梯度，相比 93–130M 且需梯度的基线在部署上极友好，真正意义上的“即插即用 TTA”。
2. **诊断扎实、动机清晰**: 用遮挡实验（Fig 1b）、注意力图（Fig 1c）等定性证据系统刻画“轨迹过拟合=动作-实体伪相关”，并以此推出数据增强+延迟反馈两条对策，逻辑链完整。
3. **延迟反馈优于自监督置信度**: Fig 2 直观说明熵最小化在过自信错误上会“火上浇油”，而用真实延迟反馈的 REINFORCE+门控 KL 能定向纠错，方法论上更稳健。
4. **跨域验证**: 同时覆盖连续控制（LIBERO 机器人）与离散视觉控制（Atari 57 全套），泛化证据较广。

### 局限性
1. **依赖延迟反馈信号**: PDF 本质需要 episode 级反馈（成功/失败/奖励）来训练 P head，在**无任何反馈**的纯开放部署中不可用；作者自承“verifier-free”但仍需环境反馈，与完全无监督 TTA 有别。
2. **未根治轨迹过拟合**: 作者明确承认 PDF 只是**缓解负面影响**而非根本解决；绝对数值上 Object 套件仍逊于若干基线。
3. **增益幅度有限/不均**: Atari 仅 +0.10 HNS 且 10/57 游戏退化；LIBERO 各 suite 提升在 +0.03~+0.10 之间，Long 仅 +0.03。
4. **机构/作者信息缺失**: arXiv HTML 正文未给机构，第一作者列表中疑有拼写（Lixiang Lium），可复现性的归属信息不完整。
5. **超参敏感性分析不足**: $\lambda$、$\lambda_{\mathrm{KL}}$、基线 $b$ 的设置与敏感度未系统给出。

### 潜在改进方向
1. 把“延迟反馈”放宽为**稀疏/带噪/中间过程奖励**，或引入自学习的代理反馈，扩大到无显式反馈的部署场景。
2. 将增强预算上限、$\lambda/\lambda_{\mathrm{KL}}$ 做成可学习/自动搜索，替代人工设 max=3 的经验值。
3. 探索更结构化的增强（语义保持型）以避免“增强越多越差”的噪声累积，针对结构依赖任务定制增强分布。
4. 把 P head 思路迁移到连续动作的扩散/流匹配动作头，验证 logit 扰动范式在非离散 token 化 VLA 上的可移植性。

### 可复现性评估
- [x] 代码开源（https://github.com/zhoujiahuan1991/CVPR2026-PDF）
- [ ] 预训练模型（未明确声明 release P head 权重）
- [x] 训练细节较完整（基座、9M P head、增强预算、损失、硬件 V100；附录给伪代码与 Atari 全表）
- [x] 数据集可获取（LIBERO、Atari 均为公开基准）

---

## 速查卡片

> [!summary] PDF: Perturbation Learning with Delayed Feedback for VLAs
> - **核心**: 把 VLA 脆弱性归因于“轨迹过拟合”（动作-实体伪相关），提出冻结基座、仅训 9M 扰动头的 verifier-free 测试时自适应。
> - **方法**: 不确定性(熵)→自适应增强预算→多头(LM+P head)融合 logits→dim-wise 多数投票选动作；episode 后用延迟反馈按 REINFORCE+门控 KL 损失（公式5）仅更新 P head。
> - **结果**: LIBERO 平均 SR 0.77/rank 2.5（+7.4% SR、Long +4.1），Atari-57 HNS 1.07 vs 0.97（+0.10，47/57 提升）；仅 9M 参数、无需基座梯度、单卡 V100。
> - **代码**: https://github.com/zhoujiahuan1991/CVPR2026-PDF

---

*笔记创建时间: 2026-06-29*
