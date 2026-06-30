---
title: "Mantis: A Versatile Vision-Language-Action Model with Disentangled Visual Foresight"
method_name: "Mantis"
authors: [Yi Yang, Xueqi Li, Yiyang Chen, Jin Song, Yihan Wang, Zipeng Xiao, Jiadi Su, You Qiaoben, Pengfei Liu, Zhijie Deng]
year: 2026
venue: CVPR
tags: [VLA, visual-foresight, latent-action, diffusion-transformer, meta-queries, language-supervision, temporal-ensemble]
zotero_collection: _inbox
image_source: online
arxiv_html: https://arxiv.org/html/2511.16175v2
created: 2026-06-29
---

# Mantis: A Versatile Vision-Language-Action Model with Disentangled Visual Foresight

## 元信息

| 项目 | 内容 |
|------|------|
| 作者 | Yi Yang, Xueqi Li, Yiyang Chen, Jin Song, Yihan Wang, Zipeng Xiao, Jiadi Su, You Qiaoben, Pengfei Liu, Zhijie Deng |
| 机构 | 上海交通大学（SJTU）等（通讯作者 Zhijie Deng / Pengfei Liu） |
| 会议 | CVPR 2026 |
| 类别 | VLA（Vision-Language-Action） |
| 日期 | 2025-11（arXiv v2） |
| 项目主页 | （代码与权重已开源，见论文声明） |
| 链接 | [arXiv](https://arxiv.org/abs/2511.16175) / [HTML](https://arxiv.org/html/2511.16175v2) |

---

## 一句话总结

> 用「解耦式视觉前瞻（DVF）」把未来帧预测从主干剥离到独立的 meta-query + DiT 头，让潜在动作隐式辅助显式动作学习，同时把主干算力让给语言监督以保住理解与推理能力，在 LIBERO 上拿到 96.7% 并在真机指令跟随/泛化上超过 $\pi_{0.5}$。

---

## 核心贡献

1. **解耦式视觉前瞻 DVF**：提出 [[Disentangled Visual Foresight|DVF]]，用 [[Meta Queries|meta queries]] + [[Diffusion Transformer|DiT]] 头把「未来帧预测」从 [[VLA]] 主干中解耦出来。配合把当前帧经[[残差连接|residual connection]]送入 DiT，一个简单的 next-state 预测目标即可让 meta queries 自动捕获刻画视觉轨迹的[[Latent Action|潜在动作]]，从而为显式动作预测提供简洁而有针对性的先验。
2. **渐进式训练配方（Progressive Training Recipe）**：分三阶段逐步引入视觉→动作→语言模态，缓解多模态学习信号之间的竞争，实现稳定优化；尤其把主干算力从「重建像素」中解放出来用于语言监督，保住 VLM 的理解与推理能力。
3. **强性能 + 高效推理**：在 [[LIBERO]] 上 96.7% 平均成功率、收敛远快于纠缠式视觉前瞻方法；真机上指令跟随、OOD 泛化、推理能力均超过 $\pi_{0.5}$；并提出 [[Adaptive Temporal Ensemble|自适应时间集成（ATE）]]，推理调用次数减半而性能基本不降。

---

## 问题背景

### 要解决的问题
[[VLA]] 的低维动作信号过于稀疏，难以充分监督处理高维感知输入的大模型，导致大量表征容量被浪费、性能受限。如何用视觉信号有效补充稀疏动作监督，同时**不破坏主干的语言理解与推理能力**，是本文要解决的核心问题。

### 现有方法的局限（见 Figure 1）
作者把「视觉增强的动作学习」归纳为三类，逐一指出短板：
1. **视觉前瞻（Visual Foresight）**：让 VLA 直接预测高维未来帧（显式自回归离散图像 token 或隐式联合训练视频生成+动作）。但像素级前瞻引入大量冗余、分散动作预测注意力，**训练成本高、收敛慢**；还可能把物理运动错误关联到纹理/光照等视觉外观，诱发幻觉。
2. **轨迹引导（Track Guidance）**：把视觉状态压缩成关键点轨迹等紧凑控制表征。但压缩造成**信息瓶颈**，且从视频中提取点轨迹精度有限，误差累积导致动作预测更差（论文里 [[ATM]] 表现最差即佐证）。
3. **潜在动作监督（Latent Action Supervision）**：先训一个动作量化模型从帧间差学离散潜在动作，再让 VLA 预测潜在动作后微调。但**需额外训练量化模型**，计算复杂度上升。

此外，多数方法**忽视语言监督**：机器人专项训练会覆盖预训练阶段获得的视觉-文本对齐，损害指令跟随与推理。

### 本文的动机
- 把前瞻预测**解耦**到独立的 meta-query + DiT 头，主干不再直接吐出冗余视觉信息，前瞻信息既紧凑又精确；
- 经残差连接喂入当前帧，使潜在动作 query 聚焦于**帧间动态**而非重建整帧，天然贴合「显式机器人动作」；
- 解耦降低主干表征负担，**腾出容量给语言监督**，从而在动作学习后仍保住语义理解与推理。核心假设：解耦前瞻 + 语言监督 = 又准又泛化。

---

## 方法详解

### 模型架构

Mantis（见 Figure 2 中部）由以下组件构成：
- **输入**：语言指令 $l$ + 当前视觉状态（原始图像帧）$\mathbf{o}_t$
- **Backbone** $\mathcal{P}$：[[Qwen2.5-VL]]（3.7B），原生支持灵活分辨率——主相机高分辨率、腕部相机低分辨率
- **核心模块**：[[Disentangled Visual Foresight|DVF]] 头 $\mathcal{D}$（[[Sana]] DiT，1.4B）+ 连接器 $\mathcal{C}$（12 层 transformer encoder + 投影层）+ 动作头 $\pi$（DiT-based，0.3B）
- **可学习 token**：[[Latent Action|潜在动作 query]] `[LAT]`（9 个）、动作 query `[ACT]`（6 个）、多间隔 query `[GAP]`（$6\times3$，对应间隔 1–6）
- **输出**：未来帧 $\mathbf{o}_{t+n}$（训练时）+ [[Action Chunking|动作块]] $\mathbf{a}_{t:t+n}$
- **总参数**：5.8B（主干 3.7B + DVF 1.4B + 动作头 0.3B + VAE 0.3B）

### 核心模块

#### 模块1: DVF 头（解耦式视觉前瞻）

**设计动机**：把未来帧预测从主干剥离，避免主干直接生成高维冗余视觉信息；并让 `[LAT]` 自动学到刻画视觉轨迹的潜在动作。

**具体实现**：
- 主干把 $\mathbf{o}_t,\,l,\,\texttt{[LAT]}$ 打包成序列编码为隐状态 $\mathbf{h}_t$（公式1）；
- $\mathbf{h}_t$ 与 $\mathbf{o}_t$ 拼接后送入连接器 $\mathcal{C}$，投影成 DiT 头 $\mathcal{D}$ 的条件输入，生成间隔为 $n$ 的未来帧 $\mathbf{o}_{t+n}$（公式2）；
- 关键点：**当前帧 $\mathbf{o}_t$ 经残差连接喂入 $\mathcal{D}$**——这样 `[LAT]` 不需要重建整帧，只需编码帧间动态（即潜在动作），故称 latent-action queries；
- DVF 头用 [[Sana]]（集成深度压缩 autoencoder 的高效 DiT）；推理执行时 **DVF 头被省略**以降开销（机器人执行不需要预测视觉状态）。

#### 模块2: 动作头（Action Head）

**设计动机**：把 `[LAT]` 提供的潜在动作转成显式动作轨迹。

**具体实现**：
- 沿用 [[DreamVLA]] 的 DiT-based 动作头；
- 可学习 `[ACT]` query 先通过**因果注意力**从输入与 `[LAT]` 中聚合信息（公式3 把 `[LAT]` 加入上下文、`[ACT]` 做信息抽取）；
- 再由动作头把高斯噪声去噪成 $n$ 步动作轨迹。

#### 模块3: 多间隔 query `[GAP]`

**设计动机**：训练时产生更密集的视觉预测，并适配下游不同任务的时间尺度。

**具体实现**：`[GAP]` 插在 `[LAT]` 之前，引导在不同时间步间隔下生成未来帧（间隔 1–6）；Figure 3 可视化多间隔未来帧生成。

### 渐进式训练配方（三阶段）

直接把视觉/语言/动作一起塞进预训练会偏向最易学的信号（如动作）或被主导模态（如语言）过拟合，造成跨模态竞争与不稳定。于是分阶段引入（Figure 2 左）：

- **Stage 1 多间隔视觉训练**：在无动作标注的人类操作视频（[[SSV2]]）上预测未来帧。解冻 DVF 头、`[LAT]`、`[GAP]`，优化扩散损失 $\mathcal{L}_{\text{DVF}}$，**冻结主干**以保住预训练语言表征。
- **Stage 2 视觉-动作联合训练**：引入机器人示范（[[DROID]]）的动作模态，时间步间隔固定为动作 chunk 大小以对齐视觉与动作流。解冻 `[ACT]`、仍冻主干，目标 $\alpha\mathcal{L}_{\text{DVF}}+\mathcal{L}_{\text{action}}$。
- **Stage 3 语言监督混合训练**：联合 38 个多模态数据集 + DROID 训练，**解冻主干**，对语言输出加交叉熵 $\mathcal{L}_{\text{lang}}$，总目标 $\alpha\mathcal{L}_{\text{DVF}}+\mathcal{L}_{\text{action}}+\beta\mathcal{L}_{\text{lang}}$。

### Adaptive Temporal Ensemble（ATE）

推理时 Mantis 用 [[Temporal Ensemble]] 提升运动稳定性，但开销大。ATE 按当前时刻的运动稳定性需求动态调整集成强度（Figure 2 右、Figure 4）：
- **Target patches**：与指令最相关的区域——用主干 cross-attention 的 text-to-vision 注意力分数取 top $\tau_{\text{target}}\%$ 视觉 token；
- **Dynamic patches**：视觉变化显著区域——把当前与上一帧切块，按像素空间余弦相似度取最低的 top $\tau_{\text{dynamic}}\%$；
- 二者**重叠 = 精细操作**（如抓取），此时启用 Temporal Ensemble 增稳；否则关闭以省算力。该变体记作 **Mantis-ATE**。

### 关键公式与机制

#### 公式1: [[VLA]] 主干编码（含潜在动作 query）

$$
\mathbf{h}_{t}=\mathcal{P}(\mathbf{o}_{t},\,l,\,\texttt{[LAT]})
$$

**含义**：主干 $\mathcal{P}$ 把当前帧、指令与潜在动作 query 打包编码为隐状态 $\mathbf{h}_t$。

**符号说明**：
- $\mathbf{o}_t$：$t$ 时刻视觉状态（原始图像帧）；$l$：语言指令
- `[LAT]`：可学习潜在动作 query；$\mathbf{h}_t$：主干输出隐状态

#### 公式2: 解耦式未来帧生成（DVF）

$$
\mathbf{o}_{t+n}=\mathcal{D}\!\left(\mathcal{C}(\mathbf{o}_{t},\,\mathbf{h}_{t})\right)
$$

**含义**：连接器 $\mathcal{C}$ 把当前帧与隐状态投影为 DiT 头 $\mathcal{D}$ 的条件输入，生成间隔 $n$ 的未来帧。其中 $\mathbf{o}_t$ 经残差连接进入 $\mathcal{D}$，迫使 `[LAT]` 只编码帧间动态而非整帧。

**符号说明**：
- $\mathcal{D}$：DVF 头（Sana DiT）；$\mathcal{C}$：连接器（12 层 encoder + 投影）
- $n$：当前帧与未来帧之间的时间步间隔；$\mathbf{o}_{t+n}$：预测的未来帧

#### 公式3: 动作生成

$$
\mathbf{a}_{t:t+n}=\pi\!\left(\mathcal{P}(\mathbf{o}_{t},\,l,\,\texttt{[LAT]},\,\texttt{[ACT]})\right)
$$

**含义**：把 `[LAT]` 加入上下文、`[ACT]` 通过因果注意力抽取信息，动作头 $\pi$ 去噪生成未来 $n$ 步动作块。

**符号说明**：
- `[ACT]`：可学习动作 query；$\pi$：DiT-based 动作头
- $\mathbf{a}_{t:t+n}$：未来 $n$ 步连续动作轨迹

#### 公式4: 单层主干 FLOPs（ATE 复杂度分析，附录 A）

$$
\text{FLOPs}\approx 4LD^{2}+2L^{2}D+2LDM
$$

**含义**：标准 Mantis 推理时每个主干层的理论计算量。ATE 额外开销——动态块识别 $\mathcal{O}(L_v D_{\text{patch}})$、目标块跨模态注意力 $\mathcal{O}(L_t L_v D)$、阈值排序 $\mathcal{O}(L_v\log L_v)$——相对该式可忽略，故 ATE 能有效加速（经验上推理次数减少 >40%）。

**符号说明**：
- $L$：首层输入 token 数；$D$：隐状态维度；$M$：FFN 中间维度
- $L_v$：图像 patch 数；$D_{\text{patch}}$：每个 patch 维度；$L_t$：文本 token 数

---

## 关键图表

<!-- 图片均使用 arXiv HTML 在线链接，未本地化 -->

### Figure 1: Vision-augmented action learning paradigms / 三类视觉增强动作学习范式

![Figure 1](https://arxiv.org/html/2511.16175v2/x1.png)

**说明**：对比三类已有范式——(a) Visual Foresight 通过预测未来帧增强动作；(b) Track Guidance 用压缩的视觉状态表征（如关键点轨迹）引导动作；(c) Latent Action Supervision 用辅助潜在动作改进动作学习。三者分别对应「冗余/高成本」「信息瓶颈」「额外量化模型」三种痛点，引出 Mantis 的解耦方案。

### Figure 2: Overview of Mantis / 整体架构与训练配方

![Figure 2](https://arxiv.org/html/2511.16175v2/x2.png)

**说明**：左=渐进式训练配方（视觉→动作→语言三阶段逐步融合）；中=Mantis 总览，由主干、DVF 头、动作头组成，DVF 头预测未来帧以促进潜在动作学习、进而改进动作预测，语言监督帮助主干保住理解与推理；右=自适应时间集成（ATE），按 target token 与 dynamic token 的重叠动态调整集成强度。这是全文的核心架构图。

### Figure 3: Multi-gap future frame generation / 多间隔未来帧生成

![Figure 3](https://arxiv.org/html/2511.16175v2/x3.png)

**说明**：Stage 1 中 `[GAP]` query 引导在不同时间步间隔（1–6）下生成的未来帧，验证多间隔机制能产生密集且时间尺度可变的视觉预测。

### Figure 4: Visualization of ATE / ATE 的注意力与相似度可视化

![Figure 4](https://arxiv.org/html/2511.16175v2/x4.png)

**说明**：注意力热图越深值越高、余弦相似度热图反之（参数 $\tau_{\text{target}}=1,\ \tau_{\text{dynamic}}=12$）。展示 target patches（指令相关物体）与 dynamic patches（机械臂/末端运动）的分布；二者重叠区域即精细操作触发时间集成的依据。

### Figure 5: Convergence speed comparison / 收敛速度对比

![Figure 5](https://arxiv.org/html/2511.16175v2/x5.png)

**说明**：LIBERO Spatial 上 epoch-SR 曲线。Mantis 收敛速度与非视觉增强的 [[OpenVLA]]、潜在动作监督 [[UniVLA]] 相当；而纠缠式视觉前瞻 [[UnifiedVLA]] 收敛最慢、前十个 epoch 成功率为零，凸显「解耦前瞻」对高效优化的必要性。

### Figure 6: Real World Experiments / 真机实验

![Figure 6](https://arxiv.org/html/2511.16175v2/x6.png)

**说明**：(a) Agilex 平台；(b) 三个场景设置与示例指令（每场景一条 ID 与对应 OOD 指令）；(c) Mantis 与 $\pi_{0.5}$ 在 ID/OOD 任务上的平均成功次数；(d) 场景 1 的逐任务成功次数。整体显示 Mantis 在 ID 与 OOD 上均胜出，$\pi_{0.5}$ 几乎无 OOD 泛化能力。

### Figure 7: Visualization of Generated Future Frames / 生成的未来帧

![Figure 7](https://arxiv.org/html/2511.16175v2/x7.png)

**说明**：真机实验中激活 DVF 头采集的生成未来帧序列。最后一帧与真实最终状态高度一致，证明 DVF 确实在为动作预测提炼有效的前瞻线索。

### Figure 8: Mantis (TE) vs Mantis-ATE / 标准 TE 与自适应 ATE 对比

![Figure 8](https://arxiv.org/html/2511.16175v2/x8.png)

**说明**：主纵轴=成功率（SR），次纵轴=推理调用次数（IC）。四个 LIBERO 套件上 Mantis-ATE 把 IC 降低近 50% 而 SR 基本持平，显著提升推理效率。

### Figure 9: Mantis vs Mantis-LU / 语言监督消融

![Figure 9](https://arxiv.org/html/2511.16175v2/x9.png)

**说明**：对比有语言监督的 Mantis 与无语言监督变体 Mantis-LU 在三场景 ID/OOD 上的表现。Mantis-LU 在 ID 上尚可，但 OOD 明显变差，说明语言监督对指令泛化至关重要。

### Figure 10: Execution examples on real-world tasks / 真机任务执行示例

![Figure 10](https://arxiv.org/html/2511.16175v2/x10.png)

**说明**：真实世界任务的执行帧序列（附录 C），定性展示 Mantis 在 Agilex 平台上的操作过程。

### Table 1: Comparison on the LIBERO benchmark / LIBERO 主结果

| 类别 | Method | Spatial | Object | Goal | Long | Avg. |
|------|--------|---------|--------|------|------|------|
| 非视觉增强 | Diffusion Policy | 78.3 | 92.5 | 68.3 | 50.5 | 72.4 |
| 非视觉增强 | OpenVLA | 84.7 | 88.4 | 79.2 | 53.7 | 76.5 |
| 非视觉增强 | $\pi_{0}$ | 96.8 | 98.8 | 95.8 | 85.2 | 94.2 |
| 非视觉增强 | $\pi_{0}$-FAST | 96.4 | 96.8 | 88.6 | 60.2 | 85.5 |
| 非视觉增强 | NORA | 92.2 | 95.4 | 89.4 | 74.6 | 87.9 |
| 视觉增强 | ATM | 68.5 | 68.0 | 77.8 | 39.3 | 63.4 |
| 视觉增强 | CoT-VLA | 87.5 | 91.6 | 87.6 | 69.0 | 81.1 |
| 视觉增强 | WorldVLA | 87.6 | 96.2 | 83.4 | 60.0 | 81.8 |
| 视觉增强 | UniVLA | 96.5 | 96.8 | 95.6 | 92.0 | 95.2 |
| 视觉增强 | UnifiedVLA | 95.4 | 98.8 | 93.6 | 94.0 | 95.5 |
| 视觉增强 | DreamVLA | 97.5 | 94.0 | 89.5 | 89.5 | 92.6 |
| 视觉增强 | $\mathcal{F}_{1}$ | 98.2 | 97.8 | 95.4 | 91.3 | 95.7 |
| 视觉增强 | **Mantis (Ours)** | **98.8** | **99.2** | 94.4 | **94.2** | **96.7** |

**说明**：Mantis 在 Spatial/Object/Long 三个套件最优、平均 96.7% SOTA（Goal 上 94.4% 略低于 $\pi_0$ 的 95.8% 等）。视觉增强方法整体优于非视觉增强，印证「稠密视觉状态可补充稀疏动作信号」；[[ATM]] 因点轨迹精度有限误差累积而垫底。

### Table 2: Comparison of four DVF variants / DVF 四变体消融

| 配置 | Spatial | Object | Goal | Long | Avg. |
|------|---------|--------|------|------|------|
| vanilla-DVF | 98.2 | 98.8 | 93.6 | 92.2 | 95.7 |
| flawed-DVF（去残差连接） | 96.8 | 97.0 | 93.8 | 89.8 | 94.4 |
| no-DVF（仅动作头） | 93.4 | 93.0 | 90.4 | 88.2 | 91.3 |
| **pretrained-DVF（视频预训练）** | **98.4** | **99.0** | **94.0** | **93.2** | **96.2** |

**关键发现**：pretrained-DVF > vanilla-DVF > flawed-DVF > no-DVF，三点结论：(a) DVF 促进动作学习；(b) 残差连接帮助 DVF 更好捕获潜在动作（去掉即降）；(c) 视频预训练进一步增益。

### Table 3: VQA and multimodal understanding benchmarks / 理解与推理基准（附录 D）

| Method | MME | OCRBench | RealWorldQA |
|--------|-----|----------|-------------|
| Qwen2.5-VL（原始主干，参考） | **2217.3** | **807** | **62.1** |
| ECoT | 0 | 12 | 0 |
| ChatVLA | 1435.2 | 729 | 57.0 |
| **Mantis (Ours)** | **2070.2** | **757** | 56.9 |

**说明**：Mantis 在 3 个基准中 2 个（MME、OCRBench）领先其他 VLA，且相对原始 Qwen2.5-VL 仅小幅下降，证明语言监督有效保住了主干的理解能力；ECoT 几近全崩，凸显机器人专项训练对语言能力的破坏。

### Table 4: 38 language-supervision datasets / 语言监督数据集（附录 B，节选）

| 来源 | 示例数据集 |
|------|-----------|
| LLaVA-OneVision-1.5-Instruct（38 个） | alfworld, allava_instruct_laion4v/vflan4v, cambrian, coco, gqa, llava_cot_100k, magpie_pro/ultra, open_orca, orca_994k, sharegpt4o/4v(part 00–08), CLEVR-Math, Super-CLEVR, textocr_gpt4v, wikipedia_2m 等 |

**说明**：覆盖 VQA、OCR、具身规划等通用任务，剔除 chart QA、医学影像等高度专门领域，用于 Stage 3 的语言监督。

### Table 5: ID / OOD instructions / 三场景指令设计（附录 C）

| 场景 | ID 指令示例 | OOD 指令示例 |
|------|------------|-------------|
| 场景1（世界知识） | Put the cup on the female singer | Put the cup on Taylor Swift |
| 场景1 | Put the cup on the basketball player | Put the cup on Michael Jordan |
| 场景1 | Put the cup on the Marvel superhero | Put the cup on Iron Man |
| 场景1 | Put the cup on the English playwright | Put the cup on Shakespeare |
| 场景2（算术逻辑） | Put the bear on the number 8 | Put the bear on the number (3+5) |
| 场景2 | Put the bear on the letter A | Put the bear on the first letter |
| 场景2 | Put the bear on the letter Z | Put the bear on the last letter |
| 场景2 | Put the bear on the number 7 | Put the bear on the number (9-2) |
| 场景3（人类意图） | Put the bottle in the basket | Put a thing that can quench thirst in the basket |
| 场景3 | Put the Rubik's Cube in the basket | I want to play with something. Can you help? |
| 场景3 | Put the bread in the basket | I'm hungry. Can you help? |
| 场景3 | Put the watch in the basket | Put a thing that can tell the time in the basket |

**说明**：每场景 4 条 ID + 4 条对应 OOD。OOD 分别考察世界知识、基础推理、人类意图理解，比单纯指代替换更需要语义理解与推理。

### Table 6: Task-level success counts / 真机逐任务成功次数（附录 C）

| 模型 | 类型 | 场景1 Avg. | 场景2 Avg. | 场景3 Avg. |
|------|------|-----------|-----------|-----------|
| **Mantis** | ID | **8.25** | **9.25** | **6.00** |
| **Mantis** | OOD | **8.25** | **7.75** | **3.75** |
| $\pi_{0.5}$ | ID | 7.75 | 8.25 | 5.75 |
| $\pi_{0.5}$ | OOD | 3.5 | 2.5 | 2.5 |

（逐任务原始值，每条指令最多 10 次：Mantis 场景1 ID=9/8/9/7、OOD=10/8/9/6；场景2 ID=9/10/9/9、OOD=9/8/9/6；场景3 ID=6/7/6/5、OOD=6/0/6/3。$\pi_{0.5}$ 场景1 ID=10/9/6/6、OOD=10/4/0/0；场景2 ID=10/6/8/9、OOD=10/0/0/0；场景3 ID=5/7/5/6、OOD=0/10/0/0。）

**说明**：Mantis 在 ID 与 OOD 上整体均超 $\pi_{0.5}$，尤其 OOD 上 $\pi_{0.5}$ 大面积归零（几无泛化），佐证语言监督带来的指令跟随与泛化优势。

---

## 实验

### 数据集 / 基准

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| [[SSV2]] | ~220K 视频 | 人类操作视频，无动作标注 | Stage 1 预训练 |
| [[DROID]] | 76K 机器人 episode | 含视频与动作 | Stage 2/3 预训练 |
| 38 个多模态数据集 | 源自 LLaVA-OneVision-1.5-Instruct | VQA/OCR/具身规划等 | Stage 3 语言监督 |
| [[LIBERO]] | 4 套件 ×10 任务 ×50 试 | Spatial/Object/Goal/Long | 下游微调/评测 |
| 真机（Agilex） | 3 场景，每场景 100 遥操作 episode（每任务 25） | ID/OOD 各 4 条指令 | 真机评测 |

### 实现细节

- **总参数 5.8B**：主干 [[Qwen2.5-VL]] 3.7B + DVF 头 [[Sana]] 1.4B + 动作头 0.3B + VAE 0.3B
- **token 配置**：`[LAT]`=9、`[ACT]`=6、`[GAP]`=$6\times3$（间隔 1–6）；扩散步数 DVF 头 30、动作头 10
- **优化**：AdamW，weight decay 0.1，grad clip 0.5，DeepSpeed 分布式
- **分辨率**：视频/主相机裁剪 resize 到 $512\times512$，腕部相机 $256\times256$
- **学习率**：Stage 1/2 cosine 调度、500 warmup、base lr 1e-4、min lr 1e-5、各训 1 epoch；Stage 3 固定 lr 1e-5、$\alpha=0.1$、$\beta=0.005$、训 1.5 epoch
- **LIBERO 微调**：沿用 Stage 1/2 学习率，训 30 epoch，**不加语言监督**，$\alpha=0.1$，选验证 SR 最高的 checkpoint
- **ATE**：图像切 $18\times18$ patch，$\tau_{\text{target}}=1,\ \tau_{\text{dynamic}}=12$

### 关键实验结论

- **仿真（Table 1）**：LIBERO 平均 96.7% SOTA，Spatial/Object/Long 三套件最优；视觉增强整体优于非视觉增强。
- **收敛（Figure 5）**：解耦前瞻使 Mantis 收敛接近 OpenVLA/UniVLA，远快于纠缠式 UnifiedVLA。
- **真机（Figure 6 / Table 6）**：ID/OOD 均超 $\pi_{0.5}$，OOD 上 $\pi_{0.5}$ 几乎归零。
- **DVF 消融（Table 2）**：pretrained > vanilla > flawed（去残差）> no-DVF，验证 DVF、残差连接、视频预训练各自的增益。
- **ATE 消融（Figure 8）**：推理次数降近 50%，SR 基本不变。
- **语言监督消融（Table 3 / Figure 9）**：保住主干理解能力（相对 Qwen2.5-VL 仅小降），且去掉语言监督的 Mantis-LU 在 OOD 上明显退化。

---

## 批判性思考

### 优点
1. **解耦思想干净有效**：用 meta-query + DiT 头把前瞻从主干剥离，既避免像素级前瞻的冗余/慢收敛，又规避轨迹引导的信息瓶颈和潜在动作监督的额外量化模型，三类痛点一并绕开；残差连接这一细节用消融（去残差即降）给出了实证。
2. **语言能力保持有可验证证据**：不是空喊「保住理解」，而是用 MME/OCRBench/RealWorldQA（Table 3）+ Mantis-LU OOD 退化（Figure 9）+ 真机 OOD 指令（含 Taylor Swift、算术、人类意图）多角度佐证，对比 ECoT 几近全崩很有说服力。
3. **工程实用**：推理省略 DVF 头、ATE 推理次数减半，兼顾性能与效率；代码与权重开源。

### 局限性
1. **真机评测规模小、噪声大**：逐任务仅 10 次执行、成功次数为整数计数，方差大；OOD 成功次数出现 0 与 10 的极端值（如 $\pi_{0.5}$ 场景2 OOD 单任务 10、其余 0），结论稳健性需更大样本。
2. **作者自承运动回退**：因缺少机器人状态输入，真机存在轻微 motion rollback；当前仅用单帧 RGB 作视觉状态，未引入本体状态/3D 点云。
3. **参数量偏大**：5.8B 总参数（主干 3.7B），相比 LIBERO 上同样强的轻量方法在部署成本上无优势；论文未报告推理频率/显存等效率绝对值。
4. **部分对比口径不一**：真机仅与 $\pi_{0.5}$ 比；附录中 Mantis 与 $\pi_{0.5}$ 的语言损失权重 $\beta$ 设置不同（真机 $\beta=0.1$，$\pi_{0.5}$ 无语言监督），属于「带语言监督 vs 不带」的对比，OOD 优势部分来自这一设定本身。

### 潜在改进方向
1. 引入机器人本体状态、3D 点云等更丰富输入以消除运动回退（作者已列为 future work）。
2. 扩大真机评测规模与基线集合，用更细粒度指标（而非整数成功计数）量化指令跟随/泛化。
3. 探索更轻量主干或对 DVF 头进一步压缩，给出推理频率/显存等部署侧绝对指标。
4. 把 ATE 的 $\tau_{\text{target}}/\tau_{\text{dynamic}}$ 做成自适应/可学习，减少手工阈值依赖。

### 可复现性评估
- [x] 代码开源（论文声明 release code）
- [x] 预训练模型（论文声明 release weights）
- [x] 训练细节完整（正文 + 附录 A/B 给出阶段、超参、分辨率、损失权重）
- [x] 数据集可获取（SSV2 / DROID / LIBERO / LLaVA-OneVision 公开；真机数据自采）

---

## 速查卡片

> [!summary] Mantis: Versatile VLA with Disentangled Visual Foresight
> - **核心**：用 meta-query + DiT 头把「未来帧预测」从主干解耦（DVF），残差喂入当前帧让 `[LAT]` 学潜在动作辅助显式动作；主干算力让给语言监督保住理解推理。
> - **方法**：Qwen2.5-VL 主干 + Sana DVF 头 + DiT 动作头；三阶段渐进训练（视觉 SSV2 → 动作 DROID → 语言混训）；推理省略 DVF 头，ATE 按 target/dynamic patch 重叠动态开关时间集成。
> - **结果**：LIBERO 96.7% SOTA、收敛远快于 UnifiedVLA；真机 ID/OOD 均超 $\pi_{0.5}$；Mantis-ATE 推理次数减半；总参 5.8B。
> - **代码**：已开源（code & weights released）

---

*笔记创建时间: 2026-06-29*
