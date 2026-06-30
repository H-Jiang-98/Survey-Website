---
title: "OmniMotion-X: Versatile Multimodal Whole-Body Motion Generation"
method_name: "OmniMotion-X"
authors: [Guowei Xu, Yuxuan Bian, Ailing Zeng, Mingyi Shi, Shaoli Huang, Wen Li, Lixin Duan, Qiang Xu]
year: 2026
venue: CVPR
tags: [motion-generation, whole-body, multimodal, diffusion-transformer, autoregressive, SMPL-X, text-to-motion, music-to-dance, speech-to-gesture, dataset]
zotero_collection: _inbox
image_source: online
arxiv_html: https://arxiv.org/html/2510.19789v1
created: 2026-06-29
---

# OmniMotion-X: Versatile Multimodal Whole-Body Motion Generation

## 元信息

| 项目 | 内容 |
|------|------|
| 作者 | Guowei Xu, Yuxuan Bian, Ailing Zeng, Mingyi Shi, Shaoli Huang, Wen Li, Lixin Duan, Qiang Xu（前两位共同一作，Ailing Zeng 与 Wen Li 为通讯） |
| 机构 | 电子科技大学（UESTC）、香港中文大学（CUHK）、香港大学（HKU）、腾讯（相关合作） |
| 会议 | CVPR 2026 |
| 类别 | 全身人体动作生成 / 多模态生成（Humanoid Motion） |
| 日期 | 2025-10（arXiv v1） |
| 项目主页 | https://cvpr.thecvf.com/virtual/2026/poster/40868 |
| 链接 | [arXiv](https://arxiv.org/abs/2510.19789) / [PDF](https://arxiv.org/pdf/2510.19789) |

---

## 一句话总结

> 用一个统一的[[自回归扩散 Transformer]]，把文本、音乐、语音、全局轨迹与“参考动作”等多模态条件以前缀 token 形式拼接，再配合 weak-to-strong 渐进训练，在自建的最大统一 MoCap 数据集 OmniMoCap-X 上一次性支持十类全身动作生成任务并刷新 SOTA。

---

## 核心贡献

1. **统一多模态自回归扩散框架**: 提出 [[OmniMotion-X]]——单个 [[Diffusion Transformer|DiT]] 主干，把 T2M、M2D、S2G、全局时空可控生成（预测/补帧/补全/关节-轨迹引导）以及它们的任意组合统一为序列到序列建模；并**首次引入“参考动作”（reference motion）作为一种特殊条件**，显著提升生成内容、风格与时序动态的一致性，从而实现 clip-by-clip 的交互式自回归生成。
2. **weak-to-strong 渐进混合条件训练**: 针对多模态条件“粒度冲突”（文本是高层语义、时空控制是物理约束、音频是节奏约束）会相互压制、导致优化困难的问题，提出由弱（文本）到强（参考动作→全局动作→音频）逐步加入条件的训练策略。
3. **OmniMoCap-X——迄今最大的统一多模态 MoCap 数据集**: 整合 **28 个公开高质量 MoCap 数据集**、横跨 10 类任务，统一到 [[SMPL-X]] 格式、30 fps、世界坐标系；共 **64.3M 帧 / 286.2 小时**，并用 [[GPT-4o]] 对渲染视频做结构化、分层（low-level 动作 + high-level 语义）的一致性字幕标注。
4. **全面 SOTA**: 在自建（更难）测试集上，T2M、GSTC、M2D、S2G 四类任务均显著超越现有方法。

---

## 问题背景

### 要解决的问题
如何用**一个模型**支持**多模态、多任务、多粒度**的全身（whole-body，含身体+手+脸）人体动作生成，并在加入“参考动作”这一新条件后实现高一致性、可交互、长时程的动作合成。

### 现有方法的局限
作者在 Table 1 / Table 2 中系统对比，指出**建模**与**数据**两条线的痛点：

建模层面（与既有多模态动作模型相比）：
1. **独立模型训练（Separate Model Training）**: 如 AMD 为不同模态各训一个模型，根本无法做同时多模态控制；
2. **额外控制分支（Additional Control Branch）**: 如 MotionCraft、MCM 为每个条件加独立分支，条件之间缺乏交互；
3. **粒度冲突训练（Conflict Granularity Training）**: 如 MotionVerse、M³GPT、MotionLLaMA 把高层语义条件与低层控制混在一起训练，模型过拟合强约束的低层信号（如稠密关节控制、细粒度参考动作），压制文本等弱信号，造成可控性下降与优化困难（视频生成里也有类似现象）。

数据层面（与既有合并数据集相比）：
1. **动作质量低**: Motion-X、MotionUnion、MotionVerse 等大量掺入非 MoCap 的估计动作，"garbage in, garbage out"；
2. **文本不一致**: 用 LLM 在“看不到动作”的情况下扩写文本，质量参差、易幻觉；
3. **任务覆盖有限**: 普遍缺 HOI / HSI / HHI 等交互任务。

### 本文的动机
- 既然问题出在“分支割裂”和“粒度冲突”，那就用**统一的前缀条件拼接**让所有模态在同一 DiT 里交互，并用**渐进式 weak-to-strong**调度避免强条件压制弱条件；
- 既然问题出在“数据脏、文本糊、任务窄”，那就**只收 MoCap 级高质量数据**、统一到 SMPL-X、并用**能“看见动作”的 [[GPT-4o]]（视频+原始文本输入）**生成分层字幕；
- 关键 insight：把**参考动作**当成一种条件，既能注入其它模态给不了的细粒度时空细节，又天然支撑“上一段生成结果→下一段条件”的自回归长序列生成。

---

## 方法详解

### 模型架构

[[OmniMotion-X]] 整体是一个**统一的、自回归的、基于 Transformer 的扩散模型**（见 Figure 1），核心思路是“多模态条件 token 化 → 作为前缀上下文拼到带噪动作 token 前 → DiT 去噪预测动作”。

- **输入条件**:
  - 文本 $\mathbf{c}_t$（语义指导）
  - 全局动作 $\mathbf{c}_g$（spatial-temporal 控制，含轨迹/关节）
  - 语音 $\mathbf{c}_s$（手势-唇动与节奏同步）
  - 音乐 $\mathbf{c}_m$（节拍与风格，用于舞蹈）
  - **参考动作 $\mathbf{c}_r$**（来自用户设计或模型上一段生成，提供其它条件给不出的细粒度动作先验）
- **条件编码器**（modality-specific encoders）:
  - 文本: [[T5-XXL]]
  - 语音: wav 编码器（沿用 BEAT2/EMAGE 一系）
  - 音乐: [[Librosa]] 特征
  - 动作（参考/全局）: body-wise 编码（沿用 MotionCraft）
  - 各模态特征经**可学习线性投影**对齐到 motion embedding 维度后拼接
- **Backbone**: [[Diffusion Transformer|DiT]]（8 层、8 头），把拼接后的条件作为**前缀上下文**与带噪动作 token 一起送入
- **输出**: 统一的全身 [[SMPL-X]] 动作序列（默认参考/预测长度 150 帧）
- **隐藏维度**: $d_{\text{model}}=1536=128\times12$（每个 body part 一个 128 维 embedding，共 12 个 part），FFN 维度 3072

### 核心模块

#### 模块1: 统一动作表示（Unified Motion Representation）

**设计动机**: 28 个数据集格式各异（BVH / FBX / SMPL(-H) / 关键点），必须先统一成同一表示，否则无法联合训练。

**具体实现**:
- 全部先转成 [[Motion-X]] 的 [[SMPL-X]] 参数（root 朝向、body/hand/jaw pose、表情、面部 shape、平移、body shape），再**归一化平移尺度与初始 root 朝向**建立一致坐标系，并统一**重采样到 30 fps**。
- 把常用的 body-only 表示扩展为**全身**表示：第 $i$ 帧姿态写作一个元组（见公式1），同时引入足-地接触二值特征与 100 维面部表情（[[FLAME]] 格式）。
- $N=127$ 为 SMPL-X 全身关节，$N'=53$ 为 body+hands+jaw 关节。

#### 模块2: 一致的视觉-文本动作字幕（Consistent Visual-textual Motion Captions）

**设计动机**: 现有三类标注（纯 LLM 扩写 / 人工 / motion-to-text 模型）分别有幻觉、不可扩展、缺高层语义的问题。

**具体实现**:
- **把动作渲染成视频**，连同已有文本标注（描述、动作标签、任务类别）一起喂给能“看视频”的 [[GPT-4o]]；
- 由此产出**结构化、分层（low-level 动作细节 + high-level 语义）、精确**的字幕；
- 工程优化：增加输入帧数（更好时序理解）、提升渲染质量、做大量 prompt engineering（示例见 Figure 4）。

#### 模块3: 统一多模态建模与前缀条件拼接（Unified Multimodal Modeling）

**设计动机**: 让所有模态在同一注意力空间内**充分交互**（区别于"额外分支"路线），并便于灵活组合/缺省任意条件。

**具体实现**:
- 各模态经编码器 $f_\bullet$ + 投影 $h_\bullet$ 后拼成统一条件 $c$（公式1），作为 DiT 的**前缀上下文（prefix context）**；
- 扩散目标遵循“**直接预测干净动作 $\hat x_0$ 而非噪声**”（沿用 MDM），以更好约束动作的物理属性（公式2）；
- 带噪动作 $x_t=[p_0^t,\dots,\mathbf{p}_N^t]$ 与条件前缀一起去噪。

#### 模块4: weak-to-strong 渐进混合条件训练（Progressive Training）

**设计动机**: 单阶段同时上所有条件时，模型会过拟合“强约束低层信号”（参考动作、稠密关节控制），压制文本等弱信号，可控性变差、优化困难。

**具体实现**（见 Figure 2 与实现细节）:
- **阶段1**: 仅文本条件，建立动作-语义对齐（460K 步）；
- **阶段2**: 加入参考动作（再 460K 步）；
- **阶段3**: 加入全局时空控制（230K 步）；
- **阶段4**: 加入完整音频条件（语音+音乐，920K 步）；
- 每新增条件**重置学习率**（$1\times10^{-4}$ → cosine 衰减到 $1\times10^{-5}$），由弱到强逐级收紧约束（从高层语义到稠密时空对齐）。

### 关键公式与机制

#### 公式1: 统一全身动作表示（逐帧姿态元组）

$$
\mathbf{p}_{i}=(\dot{r}^{a},\dot{r}^{x},\dot{r}^{z},r^{y},\mathbf{j}^{p},\mathbf{j}^{v},\mathbf{c}^{f},\mathbf{f})
$$

**含义**: 把第 $i$ 帧的全身姿态编码为一个包含根运动、关节位置/速度、足接触与面部表情的紧凑向量，作为模型生成与去噪的基本单元。

**符号说明**:
- $\dot{r}^{a}\in\mathbb{R}$: 绕 Y 轴的根角速度
- $\dot{r}^{x},\dot{r}^{z}\in\mathbb{R}$: XZ 平面内的根线速度
- $r^{y}\in\mathbb{R}$: 根高度
- $\mathbf{j}^{p}\in\mathbb{R}^{3N-1}$: 局部关节位置；$\mathbf{j}^{r}\in\mathbb{R}^{6N'}$: SMPL-X 关节 6D 旋转；$\mathbf{j}^{v}\in\mathbb{R}^{3N}$: 关节速度
- $\mathbf{c}^{f}$: 由脚跟/脚尖关节速度阈值化得到的足-地接触二值特征
- $\mathbf{f}\in\mathbb{R}^{100}$: [[FLAME]] 面部表情特征
- $N=127$（全身关节）、$N'=53$（body+hands+jaw）

#### 公式2: 统一多模态条件拼接

$$
c=\big[h_{t}(f_{t}(c_{t})),\,h_{g}(f_{g}(c_{g})),\,h_{s}(f_{s}(c_{s})),\,h_{m}(f_{m}(c_{m})),\,h_{r}(f_{r}(c_{r}))\big]
$$

**含义**: 文本/全局动作/语音/音乐/参考动作五路条件，分别经各自编码器 $f$ 与投影层 $h$ 后拼接成统一表示 $c$，作为 DiT 去噪的前缀上下文。任意条件可缺省，从而支持灵活的单任务或混合任务。

**符号说明**:
- $f_\bullet$: modality-specific 编码器（T5-XXL / wav encoder / Librosa / body-wise 编码）
- $h_\bullet$: 把各模态特征对齐到 motion embedding 维度的可学习线性投影
- 下标 $t,g,s,m,r$ 分别对应 text / global motion / speech / music / reference motion

#### 公式3: 扩散去噪目标（预测 $x_0$）

$$
\mathcal{L}_{\text{simple}}=\mathbb{E}_{x_{0}\sim q(x_{0}\mid c),\,t\sim[1,T]}\left[\left\|x_{0}-G(x_{t},t,c)\right\|_{2}^{2}\right]
$$

**含义**: 训练去噪网络 $G$ **直接预测干净动作 $x_0$**（而非噪声），更利于约束动作的物理属性。

**符号说明**:
- $q(x_{0}\mid c)$: 给定条件 $c$ 的动作数据分布；$T$: 最大扩散步数
- $G$: 学习到的去噪函数；$x_t=[p_0^t,\dots,\mathbf{p}_N^t]$: 第 $t$ 步带噪动作，$\mathbf{p}_i^t$ 为该步第 $i$ 帧姿态
- $\mathbb{E}$: 对数据分布与扩散步取期望

#### 公式4: 评测用对比损失（重训文本-动作特征提取器，附录）

由于本数据集与既有工作在规模、分布、表示上差异大，作者用对比学习**重训**文本/动作特征提取器，再用它计算 FID、R-Precision 等指标。先定义欧氏距离：

$$
D_{s_{t},s_{m}}=\|s_{t}-s_{m}\|_{2}
$$

再用 contrastive（margin）损失拉近匹配对、推开非匹配对：

$$
\mathcal{L}_{Cta}=(1-y)\,\big(D_{s_{t},s_{m}}\big)^{2}+y\,\big\{\max(0,\,d-D_{s_{t},s_{m}})\big\}^{2}
$$

**含义**: 让匹配的文本-动作特征对 $(s_t,s_m)$ 在嵌入空间靠近，非匹配对至少分离 margin $d$，从而得到适配本数据集的评测特征空间。

**符号说明**:
- $s_t,s_m$: 文本/动作语义向量（文本编码器输入来自冻结的 [[DistilBERT]]，动作编码器直接处理 150 帧以内原始动作）
- $y$: 二值标签，匹配对 $y=0$、非匹配对 $y=1$
- $d$: 非匹配对的 margin，实验中设为 10

---

## 关键图表

<!-- 图片均使用 arXiv HTML 在线链接，未本地化 -->

### Figure 0 (Teaser): Versatile Generation Showcase / 任务全景

![Teaser](https://arxiv.org/html/2510.19789v1/x1.png)

**说明**: 论文首图（teaser），直观展示 OmniMotion-X 在 text-to-motion、music-to-dance、speech-to-gesture 及全局时空可控生成等多任务上的统一序列到序列生成能力，是全文“一个模型干所有任务”主张的视觉概览。

### Figure 1: Overview / 整体架构

![Figure 1](https://arxiv.org/html/2510.19789v1/x2.png)

**说明**: OmniMotion-X 总体框架。文本、全局动作、语音、音乐、参考动作五路条件经各自编码器映射到统一空间，作为前缀上下文与带噪动作 token 一起送入自回归 DiT；时空引导（spatial-temporal guidance）保证全局动作特征一致。这是理解“前缀条件拼接 + 直接预测 $x_0$”建模范式的核心图。

### Figure 2: Weak-to-Strong Progressive Training / 渐进训练策略

![Figure 2](https://arxiv.org/html/2510.19789v1/x3.png)

**说明**: weak-to-strong 渐进训练示意。先用文本建立动作-语义对齐（弱条件），再依次叠加参考动作、全局动作、音频等更强信号，逐级提升生成质量与可控性。这是论文区别于“单阶段混合训练”的关键机制，对应 Table 7 的消融。

### Figure 3: Diverse Motion Synthesis Capabilities / 多任务定性结果

![Fig3a Text-to-motion](https://arxiv.org/html/2510.19789v1/x4.png)
![Fig3b Speech-to-gesture](https://arxiv.org/html/2510.19789v1/x5.png)
![Fig3c Music-to-dance](https://arxiv.org/html/2510.19789v1/x6.png)
![Fig3d Trajectory-guided](https://arxiv.org/html/2510.19789v1/x7.png)
![Fig3e Motion In-between](https://arxiv.org/html/2510.19789v1/x8.png)
![Fig3f Motion Prediction](https://arxiv.org/html/2510.19789v1/x9.png)

**说明**: 六个子图分别展示 (a) 文本到动作、(b) 语音到手势、(c) 音乐到舞蹈、(d) 轨迹引导合成、(e) 动作补帧（in-betweening）、(f) 动作预测，覆盖论文宣称的全部任务类型；加入参考动作时，生成动作能与参考保持一致。

### Figure 4: Example of GPT-4o Captioning Quality / 字幕标注质量示例

![Figure 4](https://arxiv.org/html/2510.19789v1/x10.png)

**说明**: GPT-4o 基于渲染视频生成的分层字幕示例，佐证“视觉+文本”联合标注相比纯 LLM 扩写更精确、更少幻觉，且能同时覆盖低层动作细节与高层语义。

### Figure 5: Interaction Types in Dataset (HHI / HOI / HSI) / 交互类型可视化

![Fig5a HHI](https://arxiv.org/html/2510.19789v1/x11.png)
![Fig5b HOI](https://arxiv.org/html/2510.19789v1/x12.png)
![Fig5c HSI](https://arxiv.org/html/2510.19789v1/x13.png)

**说明**: 从左到右为 OmniMoCap-X 中的人-人交互（HHI，如对话/握手/拥抱）、人-物交互（HOI，如抓取/操作/用工具）、人-场景交互（HSI，如坐沙发脚踩地毯）。展示数据集对交互任务的覆盖度——这正是 Motion-X 等既有数据集所缺的。

### Table 1: Method Comparison / 与现有动作生成方法对比

| Type | Method | T2M | M2D | S2G | GSTC(S) | GSTC(D) | Reference | Mixed-cond | Whole-Body | #Datasets | Hours |
|------|--------|-----|-----|-----|---------|---------|-----------|------------|------------|-----------|-------|
| DiT | MDM | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | 2 | 28.6 |
| DiT | MCM | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | 3 | 109.8 |
| DiT | LMM | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ | ✓ | ✓ | 16 | - |
| DiT | MotionCraft | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ | ✓ | 3 | 48.4 |
| AR | MoMask | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | 2 | 28.6 |
| AR | MotionGPT | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | 2 | 28.6 |
| AR | M³GPT | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ | ✓ | ✗ | 3 | 164 |
| AR | MotionLLaMA | ✓ | ✓ | ✓ | ✗ | ✓ | ✗ | ✓ | ✗ | 11 | 70.1 |
| AR-DiT | AMD | ✓ | ✓ | ✗ | ✗ | ✗ | ✓ | ✗ | ✗ | 4 | 85.87 |
| AR-DiT | DART | ✓ | ✗ | ✗ | ✓ | ✓ | ✓ | ✓ | ✗ | 2 | 43.5 |
| **AR-DiT** | **OmniMotion-X (Ours)** | **✓** | **✓** | **✓** | **✓** | **✓** | **✓** | **✓** | **✓** | **28** | **286.2** |

**说明**: OmniMotion-X 是唯一在全部任务列（T2M/M2D/S2G/稀疏与稠密全局控制）、参考动作、混合条件、全身、以及训练数据规模（28 个数据集 / 286.2 小时）上**全部打勾且数据量最大**的方法，直观佐证“最全能”定位。

### Table 2: Dataset Comparison / 与现有合并数据集对比

| Dataset | T2M | M2D | S2G | HOI | HSI | HHI | Whole-Body | Mocap/Total | Caption Src | Hierarchical | Frames | Hours |
|---------|-----|-----|-----|-----|-----|-----|------------|-------------|-------------|--------------|--------|-------|
| Motion-X (2023-07) | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ | 2 / 9 | T (9) | ✓ | 15.6M | 144.2 |
| OMG (2024-03) | ✓ | ✗ | ✗ | ✗ | ✗ | ✓ | ✗ | 9 / 13 | - | ✗ | 22.3M | 206.5 |
| MotionUnion (2024-11) | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | 4 / 15 | - | ✗ | 30M | 260 |
| MotionVerse (2024-04) | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ | ✓ | 5 / 16 | - | ✗ | 100M | - |
| MotionHub (2024-11) | ✓ | ✓ | ✓ | ✗ | ✗ | ✓ | ✓ | 8 / 19 | V(3)/T(7) | ✗ | - | 70.1 |
| **OmniMoCap-X (Ours)** | **✓** | **✓** | **✓** | **✓** | **✓** | **✓** | **✓** | **21 / 28** | **V + T (28)** | **✓** | **64.3M** | **286.2** |

**说明**: OmniMoCap-X 是唯一**同时覆盖 HOI/HSI/HHI 三类交互**、且**全部 28 个来源都用“视觉+文本”联合补全字幕并提供分层标注**的数据集；MoCap 占比最高（21/28）。注意它并非帧数最多（MotionVerse 100M），但强调"质量优先（纯 MoCap）+ 任务最全 + 字幕一致"。

### Table 3: Composition of OmniMoCap-X / 数据集组成（28 个来源）

| Task | Dataset | Frames | Hours | MoCap | Format |
|------|---------|--------|-------|-------|--------|
| T2M | Mixamo | 0.4M | 1.9 | Marker-M | BVH |
| T2M | KIT | 2.3M | 6.4 | Marker-V | SMPL |
| T2M | OMOMO | 1.1M | 2.5 | Marker-V | SMPL-X |
| T2M | IDEA400 | 1.7M | 15.7 | SV-RGB | SMPL-X |
| T2M | 100Style | 4.8M | 22.1 | IMU | BVH |
| T2M | HumanML3D | 26.5M | 65.7 | Marker-V | SMPL |
| M2D | Choreomaster | 0.1M | 1.2 | Marker-M | FBX |
| M2D | Finedance | 0.8M | 7.7 | Marker-V | SMPL-H |
| M2D | Phantomdance | 1.0M | 9.5 | Marker-M | SMPL |
| M2D | AIST++ | 1.1M | 5.2 | MV-RGB | SMPL |
| M2D | Motorica | 2.7M | 12.4 | Marker-V | BVH |
| M2D | AIOZ | 6.6M | 60.8 | SV-RGB | SMPL |
| S2G | BEAT2 | 6.9M | 64.2 | Marker-V | SMPL-X |
| HHI | Humansc3d | 0.2M | 1.1 | Marker-V | SMPL-X |
| HHI | InterHuman | 4.5M | 20.8 | MV-RGB | SMPL |
| HHI | Inter-X | 16.2M | 37.5 | Marker-V | SMPL-X |
| HOI | Arctic | 0.2M | 2.0 | Marker-V | SMPL-X |
| HOI | TACO | 0.4M | 3.3 | Marker-V | MANO |
| HOI | Fit3d | 0.4M | 2.5 | Marker-V | SMPL-X |
| HOI | Behave | 0.4M | 4.1 | MV-RGB | SMPL-X |
| HOI | Chairs | 1.0M | 9.5 | Marker-V | SMPL-X |
| HOI | HOI-M3 | 2.3M | 10.7 | MV-RGB | SMPL |
| HOI | Oaklnkv2 | 4.0M | 36.8 | Marker-V | SMPL-X |
| HSI | EMDB | 0.03M | 0.3 | IMU | SMPL |
| HSI | Rich | 0.1M | 0.8 | MV-RGB | SMPL-X |
| HSI | Lafan1 | 1.0M | 4.5 | Marker-V | BVH |
| HSI | Trumans | 1.2M | 11.2 | Marker-V | SMPL-X |
| HSI | Circle | 4.4M | 10.1 | Marker-V | SMPL-X |
| **All** | **OmniMoCap-X** | **64.3M** | **286.2** | **Mixed** | **SMPL-X** |

**说明**: 完整列出 28 个来源（按 6 类任务分组）及其帧数、时长、采集方式与原始格式。采集质量排序为 Marker-M > Marker-V > IMU > MV-RGB > SV-RGB；最终全部统一到 SMPL-X、30 fps。最大单源为 HumanML3D（65.7h）与 BEAT2（64.2h）。

### Table 4: Text-to-Motion Results / 文本到动作（OmniMoCap-X 测试集）

| Method | R-Top1 ↑ | R-Top2 ↑ | R-Top3 ↑ | FID ↓ | MM-Dist ↓ | Diversity → | MModality ↑ |
|--------|----------|----------|----------|-------|-----------|-------------|-------------|
| GT | 0.535 | 0.725 | 0.821 | 0.013 | 2.493 | 9.194 | - |
| MDM | 0.063 | 0.121 | 0.169 | 72.928 | 8.376 | 1.981 | 0.394 |
| MLD | 0.084 | 0.152 | 0.209 | 70.082 | 8.281 | 1.998 | 0.439 |
| MoMask | 0.104 | 0.163 | 0.199 | 69.361 | 8.341 | 2.123 | 0.482 |
| MoMask* | 0.267 | 0.414 | 0.530 | 17.428 | 5.661 | 6.772 | 0.811 |
| MotionCraft | 0.176 | 0.259 | 0.319 | 63.049 | 7.936 | 2.325 | 0.557 |
| MotionCraft* | 0.236 | 0.370 | 0.489 | 47.428 | 7.424 | 2.820 | 0.863 |
| **OmniMotion-X (Ours)** | **0.303** | **0.464** | **0.571** | **5.040** | **4.678** | **8.650** | **1.696** |
| **OmniMotion-X (Ours + RM)** | **0.346** | **0.511** | **0.629** | **3.199** | **4.106** | **8.009** | **1.143** |

**说明**: `*` 表示在 OmniMoCap-X 上重训的版本，`+RM` 表示加入参考动作。OmniMotion-X 在 R-Precision、FID、MM-Dist、Diversity 上全面领先，FID 从最强基线 MoMask* 的 17.4 降到 5.04，**加参考动作后进一步降到 3.20、R-Top3 升到 0.629**，直接量化了"reference motion"条件的增益。（评测重复 20 次，报告均值 ± 95% 置信区间。）

### Table 5: Global Spatiotemporal Controllable Generation / 全局时空稠密控制

| Method | FID ↓ | R-Precision Top-3 ↑ | Multimodal Distance ↓ |
|--------|-------|---------------------|-----------------------|
| GT | 0.013 | 0.821 | 2.493 |
| OmniControl | 63.725 | 0.392 | 8.011 |
| **OmniMotion-X** | **4.224** | **0.682** | **4.377** |

**说明**: 采用 OmniControl 的 cross-joint 设置、控制全部关节模拟稠密控制。因数据规模小，OmniControl 几乎无法泛化（FID 63.7）；OmniMotion-X 在稠密关节控制下 FID 仅 4.22，体现统一框架能稳定跟随稠密时空控制信号。

### Table 6: Speech-to-Gesture & Music-to-Dance / 语音手势与音乐舞蹈

| Method | S2G FID(WB) ↓ | S2G FID(Hands) ↓ | S2G Face_MSE ↓ | S2G Div ↑ | M2D FID(WB) ↓ | M2D FID(Hands) ↓ | M2D Div ↑ |
|--------|---------------|------------------|----------------|-----------|---------------|------------------|-----------|
| MotionCraft | 3.422 | 5.370 | 0.182 | 1.003 | 9.875 | 7.099 | 3.798 |
| **OmniMotion-X (Ours)** | **2.641** | 9.095 | **0.045** | **1.664** | **16.209** | **5.827** | **4.716** |

**说明**: S2G 在 BEAT2、M2D 在 AIST++/FineDance/PhantomDance 上评测。OmniMotion-X 在 S2G 全身 FID、面部 MSE、多样性，以及 M2D 的手部 FID、多样性上更优；作者解释 M2D 全身 FID 偏高（16.2）主要因为 S2G/M2D 测试集小、且本模型在更大 OmniMoCap-X 上训练带来分布差异，加上更高多样性也会推高 FID。

### Table 7: Ablation on Training Strategy / 训练策略消融

| Task | Method | FID ↓ | R-Precision Top-3 ↑ | MM-Dist ↓ | Diversity → |
|------|--------|-------|---------------------|-----------|-------------|
| T2M | w/o TrSt | 9.574 | 0.232 | 6.853 | 3.118 |
| T2M | **Ours** | **5.040** | **0.571** | **4.678** | **8.650** |
| GSTC | w/o TrSt | 10.247 | 0.491 | 6.130 | 2.438 |
| GSTC | **Ours** | **4.224** | **0.686** | **4.377** | **6.292** |

**说明**: `w/o TrSt` 指去掉 weak-to-strong 渐进训练（改为单阶段混合训练）。两项任务上 FID、R-Precision、多样性全部明显变差，验证两点结论：(1) 去掉 coarse-to-fine 训练会让细粒度控制压制文本语义对齐；(2) 混合条件训练因物理约束引入优化冲突、损害时空控制，**必须用 weak-to-strong 调度**。

### Table 8: Dataset Text Quality Statistics / 文本质量统计（附录，摘选）

| Task | Dataset | Text Count | Avg. Len | TTR Range | Top Verbs（示例） |
|------|---------|-----------|----------|-----------|-------------------|
| T2M | HumanML3D | 48,398 | 232.44 | 0.139-0.817 | stand, bend, walk, lean, turn... |
| T2M | IDEA400 | 14,753 | 279.69 | 0.288-1.000 | walk, step, lift, raise, stand... |
| M2D | AIOZ | 74,649 | 262.65 | 0.311-1.000 | extend, shift, leave, weight... |
| S2G | BEAT2 | 21,576 | 233.63 | 0.331-0.942 | emphasize, stand, move, gesture... |
| HHI | InterHuman | 23,072 | 285.82 | 0.206-0.756 | extend, leave, stand, step... |
| HOI | Oaklnkv2 | 24,854 | 266.42 | 0.280-1.000 | focus, stand, extend, adjust... |
| **All** | **OmniMoCap-X** | **321,967** | **276.78** | **0.139-1.000** | **stand, lean, bend, lift, raise...** |

**说明**: 全数据集共 **321,967 条文本**、平均句长 276.78（词数）、TTR（Type-Token Ratio，词汇多样性）范围 0.139–1.000。Top 动词分布显示数据覆盖站/弯/走/抬/举等丰富日常动作，支撑“字幕一致且词汇丰富”的主张。（完整 29 行见原文 Table 8。）

---

## 实验

### 数据集 / 基准

| 基准 | 规模 | 特点 | 用途 |
|------|------|------|------|
| OmniMoCap-X | 28 源 / 64.3M 帧 / 286.2 h / 321,967 条文本 | 纯 MoCap、统一 SMPL-X@30fps、GPT-4o 分层字幕、10 任务 | 训练 + 自建测试集 |
| OmniMoCap-X test | 每源采 10 样本（T2M/GSTC 共 280） | 跨全部数据集均匀采样、更难 | T2M / GSTC 测试 |
| AIST++ / FineDance / PhantomDance | M2D 测试序列 | 舞蹈 | M2D 测试 |
| BEAT2 | S2G 测试序列 | 语音-手势（含面部） | S2G 测试 |

### 实现细节

- **架构**: Transformer Encoder（沿用 Vaswani 2017），8 层、8 头，$d_{\text{model}}=1536=128\times12$（12 个 body part），FFN 3072
- **条件编码器**: 文本 T5-XXL、语音 wav encoder、音乐 Librosa、动作 body-wise 编码；线性投影对齐后做前缀拼接
- **扩散**: 直接预测 $x_0$，简单 L2 目标
- **渐进训练步数**: 文本 460K → +参考动作 460K → +全局时空控制 230K → +完整音频 920K；对应 batch size 48 / 48 / 48 / 16
- **优化**: AdamW，初始 lr $1\times10^{-4}$（新条件加入时重置），cosine 衰减到 $1\times10^{-5}$（前 460K 步内）
- **默认动作长度**: 参考/预测 150 帧
- **硬件**: 单张 H800 GPU
- **评测特征器**: 因分布差异，按公式3/4 用对比学习在本数据集上**重训**文本/动作特征提取器（文本侧基于冻结 DistilBERT，结构类 ACTOR 但去掉概率分布建模）

### 关键实验结论

- **T2M（Table 4）**: 全面 SOTA，FID 5.04（基线最佳 17.4），加参考动作后降到 3.20；强文本编码器（T5-XXL > CLIP）+ 参考动作 + 统一框架共同带来增益。
- **GSTC（Table 5）**: 稠密关节控制下 FID 4.22，远胜 OmniControl（63.7）。
- **M2D / S2G（Table 6）**: S2G 多项指标领先；M2D 手部 FID 与多样性更优，全身 FID 受小测试集与高多样性影响偏高。
- **消融（Table 7）**: weak-to-strong 渐进训练在 T2M 与 GSTC 上均显著优于单阶段混合训练，证明“按粒度由弱到强加条件”是必要的。
- **定性（Figure 3）**: 六类任务均可生成，结合参考动作时与参考一致。

---

## 批判性思考

### 优点
1. **“最全能”定位有硬证据**: Table 1/2 把任务覆盖、条件类型、数据规模列得清清楚楚，OmniMotion-X 与 OmniMoCap-X 在“全勾选”这件事上确实是最全的；不是空喊 unified。
2. **“参考动作”是简洁而有效的新条件**: 仅把它当成另一路前缀条件，就同时拿到了 (a) 细粒度时空一致性、(b) clip-by-clip 自回归长序列生成两个好处，Table 4 的 `+RM` 增益（FID 5.04→3.20）把价值量化了出来。
3. **数据工程扎实**: 坚持纯 MoCap、统一 SMPL-X、用“能看视频的 GPT-4o”做分层字幕，直击既有数据集“脏/糊/窄”的痛点；Table 8 给了 32 万条文本的统计佐证。
4. **消融指向明确**: weak-to-strong 的必要性有 Table 7 双任务支撑，而非仅凭直觉。

### 局限性
1. **缺交互约束**: 作者自己承认模型当前**没有 scene/object/human 交互的物理约束**——尽管数据集含 HOI/HSI/HHI，但论文正文的定量评测只覆盖 T2M/GSTC/M2D/S2G 四类，**交互任务（HOI/HSI/HHI）没有定量结果**，"支持 10 任务"的主张在交互这一块更多停留在数据与定性层面。
2. **评测自洽性存疑**: 测试集、特征提取器都是作者自建/自训（自建 280 样本测试集 + 重训 FID 特征器），与社区标准评测口径不同，**跨论文可比性弱**；M2D/S2G 还用“测试集太小、分布差异、多样性高”来解释 FID 偏高，说服力有限。
3. **推理慢**: 采样空间去噪导致推理速度慢（作者亦承认），不利于实时/交互部署。
4. **可控性证据偏弱**: GSTC 仅与 OmniControl 一个基线比，M2D/S2G 仅与 MotionCraft 比，基线偏单一。

### 潜在改进方向
1. 补齐 HOI/HSI/HHI 的**定量评测**与对应基线，让“10 任务”名副其实；并引入接触/穿透等物理约束。
2. 用 latent-space 扩散或一致性/蒸馏加速采样，缓解推理慢。
3. 采用社区公认的评测协议或公开重训特征器权重，增强跨工作可比性与可复现性。
4. 对“参考动作”做更系统的消融（不同来源/长度/噪声水平）以厘清其增益边界。

### 可复现性评估
- [ ] 代码开源（v1 论文未见明确开源链接）
- [ ] 预训练模型（未声明 release）
- [x] 训练细节较完整（步数/batch/lr/调度/架构维度均给出，含附录评测器细节）
- [x] 数据来源可获取（28 个均为公开数据集；但合并后的 OmniMoCap-X 与字幕是否 release 未明确）

---

## 速查卡片

> [!summary] OmniMotion-X: Versatile Multimodal Whole-Body Motion Generation
> - **核心**: 一个自回归 DiT，把文本/全局动作/语音/音乐/**参考动作**五路条件作为前缀上下文统一拼接，直接预测干净动作 $x_0$，配 weak-to-strong 渐进训练。
> - **方法**: 统一 SMPL-X 全身表示 → 多模态条件编码+线性投影+前缀拼接（公式2）→ DiT 去噪（公式3）；训练分文本→参考动作→全局控制→音频四阶段，新条件重置 lr。
> - **数据**: OmniMoCap-X，28 个 MoCap 源 / 64.3M 帧 / 286.2 h / 32 万条 GPT-4o 分层字幕，统一 SMPL-X@30fps，覆盖 T2M/M2D/S2G/HOI/HSI/HHI。
> - **结果**: T2M FID 5.04（+参考动作 3.20）、GSTC FID 4.22、S2G/M2D 多项领先；消融证明渐进训练必要。
> - **不足**: 交互任务无定量结果、推理慢、评测口径自定。

---

*笔记创建时间: 2026-06-29*
