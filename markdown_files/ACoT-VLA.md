---
title: "ACoT-VLA: Action Chain-of-Thought for Vision-Language-Action Models"
method_name: "ACoT-VLA"
authors: [Linqing Zhong, Yi Liu, Yifei Wei, Ziyu Xiong, Maoqing Yao, Si Liu, Guanghui Ren]
year: 2026
venue: CVPR
tags: [VLA, action-chain-of-thought, reasoning, flow-matching, self-conditioning, kv-cache]
zotero_collection: _inbox
image_source: online
arxiv_html: https://arxiv.org/html/2601.11404v2
created: 2026-06-29
---

# ACoT-VLA: Action Chain-of-Thought for Vision-Language-Action Models

## 元信息

| 项目 | 内容 |
|------|------|
| 作者 | Linqing Zhong, Yi Liu, Yifei Wei, Ziyu Xiong, Maoqing Yao, Si Liu, Guanghui Ren |
| 机构 | 北京航空航天大学（Beihang University）、智元机器人（AgiBot） |
| 会议 | CVPR 2026 |
| 类别 | VLA（Vision-Language-Action） |
| 日期 | 2026-03（arXiv v2） |
| 项目主页 | https://github.com/AgibotTech/ACoT-VLA |
| 链接 | [arXiv](https://arxiv.org/abs/2601.11404) / [Code](https://github.com/AgibotTech/ACoT-VLA) |

---

## 一句话总结

> 把 [[Chain-of-Thought|思维链]] 的"思考"从语言/视觉空间搬到**动作空间**：用显式动作推理器（EAR）生成粗粒度参考轨迹、用隐式动作推理器（IAR）从 VLM 的 KV cache 抽取潜在动作先验，二者共同构成 Action-CoT 来引导动作头，在多个仿真与真机基准上刷新 SOTA。

---

## 核心贡献

1. **提出 Action Chain-of-Thought (ACoT) 新范式**: 据作者所知，这是首个把策略的"深思熟虑"过程定义为**结构化的显式动作意图链**（而非抽象的语言子目标或视觉子目标图像）的工作，让推理直接发生在与执行同构的动作空间，缓解"语义-运动学鸿沟"。
2. **设计互补的 EAR 与 IAR 两个动作推理器**: [[Explicit Action Reasoner|EAR]] 以轻量 Transformer 自合成可执行的参考轨迹（显式动作引导），[[Implicit Action Reasoner|IAR]] 以可学习 query 对 VLM 的 [[KV Cache|键值缓存]] 做交叉注意力抽取潜在动作先验（隐式动作引导），二者从两种互补形态提供动作空间指导。
3. **构建统一框架 ACoT-VLA 并验证 SOTA**: 通过 [[Action-Guided Prediction|动作引导预测（AGP）]] 头把显式/隐式引导经双路交叉注意力融合后条件化去噪，在 [[LIBERO]]、[[LIBERO-Plus]]、[[VLABench]] 三个仿真基准与 AgiBot G1 / AgileX 真机上全面超越 $\pi_0$、$\pi_{0.5}$ 等强基线。

---

## 问题背景

### 要解决的问题
现有通用机器人策略（[[VLA]]）几乎都在**视觉-语言（输入）空间**"思考"，难以弥合"丰富语义表征"与"精确低层动作执行（输出）"之间的内在落差。如何**直接在动作空间**高效合成引导推理所需的高维运动线索，从而得到真正"接地（grounded）"的策略，是本文要解决的核心问题。

### 现有方法的局限
作者把已有"中间推理"路线归为两类，并指出其共同缺陷：
1. **语言 CoT（Language CoT）**: 如预测子任务、用 LLM 推理能力生成中间步骤（见 Figure 1(a)）。其引导 $g_{\text{lang}}$ 停留在语义层面，对动作执行而言是**间接**的。
2. **视觉 CoT / 世界模型（Visual CoT）**: 如 [[CoT-VLA]] 合成子目标图像、[[DreamVLA]]/[[WorldVLA]] 预测未来视觉状态（见 Figure 1(b)）。其引导 $g_{\text{vis}}$ 仍绑定在视觉表征上，同样**间接**。
3. **根本症结**: VLM 主干的知识来自 web 规模、面向语义对齐与问答的预训练，表征**偏语言理解而非物理动力学**；因此无论语义还是视觉形式的推理，都只能提供**次优、间接**的动作引导，信息通道天然受限，难以传递动作空间所需的细粒度知识。

### 本文的动机
作者主张：要跨越"语义-运动学鸿沟（semantic-kinematic gap）"，引导必须是**运动学连贯（kinematically coherent）**的，而非纯语义或纯视觉。于是把"思想"重新定义为一串**显式、运动学接地的动作意图**（Figure 1(c)），如同"从物理示范中学习"，直接以动作空间信息为条件，使策略学习更高效、更接地。进一步观察到动作相关信息有两种互补形态——**显式**（可观测运动轨迹，如人类示范）与**隐式**（潜藏在 "reach out"/"grasp" 等语言表达与视觉交互意图中的动作分布），据此设计 EAR + IAR。

---

## 方法详解

### 模型架构

ACoT-VLA 在 $\pi_{0.5}$ 之上构建，整体由**共享 VLM 主干 + 三个核心组件**组成（见 Figure 2）：
- **输入**: 语言指令 $l$ + 当前视觉观测 $o_t$
- **Backbone**: VLM = [[SigLIP]] 视觉编码器 + [[Gemma]]-2B 语言主干（$N=18$ 层，hidden size $d=2048$）；输入帧 resize 到 $224\times224$
- **核心模块**: [[Explicit Action Reasoner|EAR]]（合成参考轨迹，显式引导 $g_{\text{action}}^{\text{ex}}$）+ [[Implicit Action Reasoner|IAR]]（从 KV cache 抽取隐式先验 $g_{\text{action}}^{\text{im}}$）+ [[Action-Guided Prediction|AGP]] 头（融合两路引导并去噪）
- **输出**: 连续 [[Action Chunking|动作块]] $a_{t:t+H-1}$，默认 $H=10$
- **训练范式**: 全程 [[Flow Matching|流匹配]] MSE 目标；EAR 用 [[Teacher Forcing|teacher forcing]] 稳定，推理时切换为自条件（self-conditioned）

ACoT 的核心思想是把通用的"输入空间引导"扩展出**动作空间引导** $g_{\text{action}}$，并把它拆成显式与隐式两支，分别由 EAR/IAR 生成、由 AGP 融合后条件化最终动作头。

### 核心模块

#### 模块1: Explicit Action Reasoner（EAR，显式动作推理器）

**设计动机**: 让模型**自主合成参考动作序列**作为内部引导。作者把它类比为生成模型中 [[Self-Conditioning|自条件（self-conditioning）]] 在动作空间的迁移——把"先验估计"注入生成过程能显著提升样本质量。

**具体实现**:
- 预训练 VLM 先把 $(o_t,l)$ 编码成逐层的 KV cache（公式3）。
- EAR 实例化为**轻量 Transformer**（同样 $N=18$ 层），输入一段噪声参考动作 $\tilde a_{t:t+H^{ref}-1}$（$H^{ref}$ 为参考动作 horizon，默认 15），嵌入为初始隐状态 $h_0^{\text{ref}}$。
- 每层做 **self-attention（捕捉动作序列内部时序依赖）+ cross-attention（向对应 VLM 层的 $K_i^{\text{VLM}},V_i^{\text{VLM}}$ 注入多模态上下文先验）**（公式4），再经残差并行的 FFN 更新（公式5）。
- 经流匹配训练，EAR（记 $\pi_\theta^{\text{ref}}$）学到动作轨迹分布，输出去噪后的参考序列 $a^{ref}$（公式6），再经 MLP 投影得到显式动作嵌入 $Z^{\text{ex}}$，即 $g_{\text{action}}^{\text{ex}}$。

#### 模块2: Implicit Action Reasoner（IAR，隐式动作推理器）

**设计动机**: VLM 的多模态潜空间还编码了**隐式运动线索**（如视觉 affordance、动作相关语义）。直接从 KV cache 抽取这些动作相关表征可提供与 EAR 互补的引导。

**具体实现**:
- 对每个 VLM 层 $i\in[1,N]$，初始化可学习矩阵 $Q_i\in\mathbb{R}^{M\times d}$（$M$ 为超参，论文设 $M=1$）。
- 考虑 KV cache 信息冗余与计算效率，先把该层 $K_i^{\text{VLM}},V_i^{\text{VLM}}$ 用可学习线性投影 **下采样到低维** $d'\ll d$（论文 $d'=128$）（公式7）。
- 在下采样后的空间做 cross-attention，再 **average pooling + MLP**，得到该层隐式动作语义 $z_i^{\text{im}}$（公式8）。
- 跨层聚合得隐式动作特征 $Z^{\text{im}}$，即 $g_{\text{action}}^{\text{im}}$，补足 EAR 的显式运动先验。
- 消融（Table 6）显示"先下采样再聚合（Downsample）"优于直接 query 或 attention pooling，提示 VLM 特征对动作预测**含噪**，需要恰当的交互机制做对齐。

#### 模块3: Action-Guided Prediction（AGP，动作引导预测）

**设计动机**: 把 EAR 的显式嵌入 $Z^{\text{ex}}$ 与 IAR 的隐式特征 $Z^{\text{im}}$ 一并融入策略学习。

**具体实现**:
- 噪声动作段 $\tilde a_{t:t+H-1}$ 经 MLP 编码后，**不直接喂动作头**，而是当作 action query $Q_{action}$。
- 用 $Q_{action}$ 分别对 $Z^{\text{ex}}$、$Z^{\text{im}}$ 做**双路交叉注意力**，得到显式/隐式引导下的注意表征 $S^{\text{ex}}$、$S^{\text{im}}$（公式9、10）。二者强调运动的不同侧面：显式给**运动学线索**，隐式给**潜在动作倾向**。
- 拼接 $[S^{\text{ex}};S^{\text{im}}]$ 后经 **self-attention 融合块** 得到统一表征 $\bar h$（公式11），最终送入动作头 $\pi_\theta^{\text{head}}$ 预测去噪动作序列。
- **Teacher Forcing 稳定化**: 训练时 $\pi_\theta^{\text{ref}}$ 输出不稳定，故 $Z^{\text{ex}}$ **直接由真值参考轨迹算得**（而非用 EAR 预测），避免对 $\pi_\theta^{\text{head}}$ 的优化干扰；推理时切换为完全自条件模式，由 $\pi_\theta^{\text{ref}}$ 自主生成参考动作。

### 关键公式与机制

#### 公式1: [[VLA]] 基本策略映射

$$
a_{t:t+H-1}=\pi_{\theta}(o_{t},l)
$$

**含义**: 通用机器人策略 $\pi_\theta$ 把当前观测与语言指令映射为长度 $H$ 的动作序列。

**符号说明**:
- $o_t$: $t$ 时刻视觉观测；$l$: 语言指令
- $H$: 动作 horizon；$a_{t:t+H-1}$: 输出动作块

#### 公式2: 引导信号的链式分解

$$
\pi_{\theta}(a_{t:t+H-1},g\mid o_{t},l)=\pi_{\theta}(a_{t:t+H-1}\mid o_{t},l,g)\,\pi_{\theta}(g\mid o_{t},l)
$$

**含义**: 把"额外引导 $g$"显式纳入策略，先由观测/指令生成引导 $g$，再以 $g$ 为条件预测动作。本文据此把 $g$ 从 $\{g_{\text{lang}},g_{\text{vis}}\}$ **扩展到包含 $g_{\text{action}}$**，并将其拆为显式 $g_{\text{action}}^{\text{ex}}$ 与隐式 $g_{\text{action}}^{\text{im}}$。

**符号说明**:
- $g\in\{g_{\text{lang}},g_{\text{vis}},g_{\text{action}}\}$: 语言/视觉/动作三类引导
- $g_{\text{action}}^{\text{ex}}$: 参考动作序列形式的直接先验；$g_{\text{action}}^{\text{im}}$: 来自上下文（如语言隐含的动作分布）的隐式先验

#### 公式3: VLM 编码为 KV cache

$$
(K^{\text{VLM}}_{1:N},V^{\text{VLM}}_{1:N})=\text{VLM}(o_{t},l)
$$

**含义**: 预训练 VLM 把观测与指令编码为逐层（共 $N$ 层）的键值缓存，供 EAR/IAR 取用。

**符号说明**:
- $N$: VLM 层数（论文为 18）；$K^{\text{VLM}}_i,V^{\text{VLM}}_i$: 第 $i$ 层的键/值

#### 公式4: EAR 的注意力层

$$
\tilde{h}_{i}^{\text{ref}}=\text{Self-Attn}(h_{i-1}^{\text{ref}})+\text{CrossAttn}(h_{i-1}^{\text{ref}},K^{\text{VLM}}_{i},V^{\text{VLM}}_{i})
$$

**含义**: EAR 第 $i$ 层并行做 self-attention（抓动作序列内时序依赖）与对 VLM 第 $i$ 层 KV 的 cross-attention（注入多模态上下文先验）。

**符号说明**:
- $h_{i-1}^{\text{ref}}$: 上一层参考动作隐状态；$\tilde h_i^{\text{ref}}$: 注意力后中间表征

#### 公式5: EAR 的 FFN 残差更新

$$
h_{i}^{\text{ref}}=h_{i-1}^{\text{ref}}+\text{FFN}(\tilde{h}_{i}^{\text{ref}})
$$

**含义**: 以残差并行方式经前馈网络更新到第 $i$ 层参考表征 $h_i^{\text{ref}}$。

#### 公式6: EAR 输出参考动作序列

$$
a^{ref}_{t:t+H^{ref}-1}=\pi^{\text{ref}}_{\theta}(\tilde{a}_{t:t+H^{ref}-1},K^{\text{VLM}}_{1:N},V^{\text{VLM}}_{1:N})
$$

**含义**: 经流匹配训练的 EAR（$\pi_\theta^{\text{ref}}$）以噪声参考动作与 VLM KV cache 为条件，去噪出可执行的参考轨迹；再经 MLP 投影得显式嵌入 $Z^{\text{ex}}$。

**符号说明**:
- $H^{ref}$: 参考动作 horizon（默认 15，可大于策略输出 horizon $H$）

#### 公式7: IAR 的下采样投影

$$
Q_{i}^{\prime}=Q_{i}W_{Q}^{(i)},\quad K_{i}^{\prime}=K_{i}^{\text{VLM}}W_{K}^{(i)},\quad V_{i}^{\prime}=V_{i}^{\text{VLM}}W_{V}^{(i)}
$$

**含义**: 把可学习 query 与该层 VLM 的键/值同时线性投影到低维空间，降冗余、提效率。

**符号说明**:
- $Q_i\in\mathbb{R}^{M\times d}$: 第 $i$ 层可学习查询（$M=1$）
- $W_Q^{(i)},W_K^{(i)},W_V^{(i)}\in\mathbb{R}^{d\times d'}$: 可学习投影矩阵；$d'\ll d$（$d'=128$）

#### 公式8: IAR 的层内隐式动作语义

$$
z^{\text{im}}_{i}=\text{MLP}(\text{Pool}(\text{CrossAttn}(Q_{i}^{\prime},K_{i}^{\prime},V_{i}^{\prime})))
$$

**含义**: 在下采样空间做交叉注意力、平均池化、MLP，得到第 $i$ 层的隐式动作语义；跨层聚合后得 $Z^{\text{im}}$。

**符号说明**:
- $\text{Pool}$: average pooling；$z_i^{\text{im}}$: 第 $i$ 层隐式动作语义表征

#### 公式9–10: AGP 的双路交叉注意力

$$
S^{\text{ex}}=\text{CrossAttn}(Q_{action},Z^{\text{ex}},Z^{\text{ex}})
$$

$$
S^{\text{im}}=\text{CrossAttn}(Q_{action},Z^{\text{im}},Z^{\text{im}})
$$

**含义**: 把噪声动作编码成 action query $Q_{action}$，分别向显式嵌入 $Z^{\text{ex}}$ 与隐式特征 $Z^{\text{im}}$ 检索互补先验，得到两路注意表征。

**符号说明**:
- $Q_{action}$: 由噪声动作段经 MLP 得到的查询
- $S^{\text{ex}}$: 受显式（运动学）先验引导的表征；$S^{\text{im}}$: 受隐式（潜在动作倾向）先验引导的表征

#### 公式11: AGP 的自注意力融合

$$
\bar{h}=\text{Self-Attn}([S^{\text{ex}};\,S^{\text{im}}])
$$

**含义**: 拼接两路注意表征后经 self-attention 融合为统一表征 $\bar h$，再送入动作头 $\pi_\theta^{\text{head}}$ 预测去噪动作。

**符号说明**:
- $[\,\cdot\,;\,\cdot\,]$: 沿特征维拼接；$\bar h$: 融合后的统一动作条件表征

#### 公式12: 总训练目标

$$
\mathcal{L}_{\text{total}}=\lambda_{1}\mathcal{L}_{\pi_{\theta}^{\text{ref}}}+\lambda_{2}\mathcal{L}_{\pi_{\theta}^{\text{head}}}
$$

**含义**: 整个框架在标准 [[Flow Matching|流匹配]] MSE 下优化，损失由 EAR（$\pi_\theta^{\text{ref}}$）与动作头（$\pi_\theta^{\text{head}}$）两部分加权组成。

**符号说明**:
- $\mathcal{L}_{\pi_{\theta}^{\text{ref}}}$: EAR 参考轨迹的流匹配损失；$\mathcal{L}_{\pi_{\theta}^{\text{head}}}$: 动作头的流匹配损失
- $\lambda_1,\lambda_2$: 平衡因子（论文均设 0.5）

---

## 关键图表

<!-- 图片均使用 arXiv HTML 在线链接，未本地化 -->

### Figure 1: Chain-of-Thought in Different Space / 不同空间中的思维链

![Figure 1](https://arxiv.org/html/2601.11404v2/x1.png)

**说明**: 对比三类 CoT 范式。(a) 语言 CoT 预测子任务作为中间推理；(b) 视觉 CoT 合成目标图像引导动作策略；(c) 本文的 Action CoT **直接在动作空间运作**，给出与执行同构（homogeneous）的动作引导。此图直观地点出本文与语言/视觉中间推理路线的根本区别——把"思考"放进动作空间，弥合语义-运动学鸿沟。

### Figure 2: Architectural Overview of ACoT-VLA / 整体架构

![Figure 2](https://arxiv.org/html/2601.11404v2/x2.png)

**说明**: ACoT-VLA 三大组件均作用在共享 VLM 主干的特征上。(a) [[Explicit Action Reasoner|EAR]]：Transformer 模块合成粗粒度参考轨迹，提供显式动作空间引导；(b) [[Implicit Action Reasoner|IAR]]：用可学习 query 对 VLM 内部表征做交叉注意力，抽取潜在动作先验；(c) [[Action-Guided Prediction|AGP]] 头：经交叉注意力把显式与隐式引导协同融合，条件化最终去噪过程，产出可执行动作序列。这张图是理解 EAR/IAR/AGP 数据流（对应公式3–11）的关键。

### Figure 3: Real-World Task Visualization / 三个真机操作任务

![Figure 3](https://arxiv.org/html/2601.11404v2/x3.png)

**说明**: AgiBot G1 上的三个真机任务可视化——"Wipe Stain"（擦污渍，考接触丰富的操作）、"Pour Water"（倒水，考精细物体操控）、"Open-set Pick"（开放集抓取，考指令遵循）。佐证方法在真实传感条件下的可用性，并通过在 AgileX 上重做 Open-set Pick 验证跨本体适配性。

### Figure 4: Real-World Evaluation Results / 真机评测结果

![Figure 4](https://arxiv.org/html/2601.11404v2/figures/fig4.png)

**说明**: 三个真机任务的成功率对比。ACoT-VLA 平均成功率 **66.7%**，高于 $\pi_{0.5}$（61.0%）与 $\pi_0$（33.8%）；在 AgiBot G1 与 AgileX 上一致提升，说明方法在真实环境与不同本体上均有效。

### Table 1: LIBERO Benchmark / LIBERO 基准（成功率 SR% 与排名 Rank）

按 Guidance 类型分组（–/Visual/Linguistics/Action），仅列代表性行，加粗本文与最佳。

| Method | Guidance | Spatial | Object | Goal | Long | Avg. |
|--------|----------|---------|--------|------|------|------|
| Diffusion Policy | – | 78.3 | 92.5 | 68.3 | 50.5 | 72.4 |
| Octo | – | 78.9 | 85.7 | 84.6 | 51.1 | 75.1 |
| CoT-VLA | Visual | 87.5 | 91.6 | 87.6 | 69.0 | 81.1 |
| WorldVLA (512²) | Visual | 87.6 | 96.2 | 83.4 | 60.0 | 81.8 |
| DreamVLA | Visual | 97.5 | 94.0 | 89.5 | 89.5 | 92.6 |
| UniVLA | Visual | 95.4 | 98.8 | 93.6 | 94.0 | 95.5 |
| F1 | Visual | 98.2 | 97.8 | 95.4 | 91.3 | 95.7 |
| GE-Act | Visual | 98.2 | 97.6 | 95.8 | 94.4 | 96.5 |
| OpenVLA | Linguistics | 84.7 | 88.4 | 79.2 | 53.7 | 76.5 |
| $\pi_0$-FAST | Linguistics | 96.4 | 96.8 | 88.6 | 60.2 | 85.5 |
| SmolVLA | Linguistics | 93.0 | 94.0 | 91.0 | 77.0 | 88.8 |
| GR00T-N1 | Linguistics | 94.4 | 97.6 | 93.0 | 90.6 | 93.9 |
| $\pi_0$ | Linguistics | 96.8 | 98.8 | 95.8 | 85.2 | 94.1 |
| GO-1 | Linguistics | 96.2 | 97.8 | 96.0 | 89.2 | 94.8 |
| MemoryVLA | Linguistics | 98.4 | 98.4 | 96.4 | 93.4 | 96.7 |
| $\pi_{0.5}$ | Linguistics | 98.8 | 98.2 | 98.0 | 92.4 | 96.9 |
| OpenVLA-OFT | Linguistics | 97.6 | 98.4 | 97.9 | 94.5 | 97.1 |
| VLA-Adapter | Linguistics | 97.8 | 99.2 | 97.2 | 95.0 | 97.3 |
| **Ours $\diamond$（冻结 LLM）** | **Action** | **99.4** | **99.6** | 98.8 | 96.0 | **98.5** |
| **Ours** | **Action** | 98.6 | 99.0 | **99.4** | **97.0** | **98.5** |

**说明**: ACoT-VLA 在四个 track 上整体 **平均排名第 1（98.5%）**，较此前最佳 $\pi_{0.5}$（96.9%）绝对提升 **+1.6%**；在最难的 **Long（长程）** track 上提升尤其明显（97.0%，Rank 1），印证"动作作为中间推理"对长程任务鲁棒性的增益。注意 $\diamond$ 表示训练时冻结 LLM 主干，仍达 98.5% 平均，说明增益主要来自 EAR/IAR 而非 LLM 微调。

### Table 2: LIBERO-Plus Benchmark / 分布偏移鲁棒性（成功率 %）

7 维扰动；分零样本迁移与监督微调两种协议。

**Zero-Shot Transfer（LIBERO 训练 → LIBERO-Plus 测试）**

| Method | Guidance | Camera | Robot | Language | Light | Background | Noise | Layout | Avg. |
|--------|----------|--------|-------|----------|-------|------------|-------|--------|------|
| OpenVLA | Linguistics | 0.8 | 3.5 | 23.0 | 8.1 | 34.8 | 15.2 | 28.5 | 15.6 |
| $\pi_0$-Fast | Linguistics | 65.1 | 21.6 | 61.0 | 73.2 | 73.2 | 74.4 | 68.8 | 61.6 |
| OpenVLA-OFT | Linguistics | 56.4 | 31.9 | 79.5 | 88.7 | 93.3 | 75.8 | 74.2 | 69.6 |
| $\pi_0^{*}$ | Linguistics | 61.0 | 40.8 | 63.5 | 89.3 | 84.1 | 80.1 | 76.4 | 69.4 |
| $\pi_{0.5}^{*}$ | Linguistics | 75.8 | 79.4 | 83.3 | 95.5 | 95.0 | 89.6 | 87.0 | 85.7 |
| Ours $\diamond$ | Action | 68.9 | 80.3 | 84.1 | 95.6 | 93.1 | 81.5 | 88.3 | 83.6 |
| **Ours** | **Action** | 72.6 | **82.6** | **87.5** | **97.7** | **96.5** | 87.8 | **88.1** | **86.6** |

**Supervised Fine-Tuning（直接在 LIBERO-Plus 训练）**

| Method | Guidance | Camera | Robot | Language | Light | Background | Noise | Layout | Avg. |
|--------|----------|--------|-------|----------|-------|------------|-------|--------|------|
| $\pi_0^{\diamond}$ | Linguistics | 79.6 | 21.1 | 72.5 | 84.7 | 86.2 | 68.3 | 69.4 | 67.4 |
| $\pi_{0.5}^{\diamond}$ | Linguistics | 70.3 | 41.7 | 81.1 | 97.3 | 94.6 | 71.8 | 84.9 | 75.7 |
| Ours $\diamond$ | Action | 91.2 | 62.5 | 80.3 | 95.1 | 91.5 | 88.3 | 84.9 | 84.1 |
| **Ours** | **Action** | **96.6** | **70.4** | 79.7 | 95.1 | **97.1** | **95.9** | 85.0 | **88.0** |

**说明**: 两种协议下均全面超越基线。零样本下对**机器人初始状态扰动 +3.2%**、**语言变化 +4.2%** 提升最明显——而语言/视觉引导策略在这些维度退化严重，凸显动作空间推理对分布偏移的鲁棒性。

### Table 3: VLABench Benchmark / VLABench（IS 意图分 / PS 进度分 %）

| Method | Guidance | In-dist. (IS/PS) | Category | Commonsense | Instruction | Texture | Avg. (IS/PS) |
|--------|----------|------------------|----------|-------------|-------------|---------|--------------|
| $\pi_0^{\diamond}$ | Linguistics | 67.8 / 62.7 | 44.0 / 33.6 | 54.9 / 43.0 | 58.0 / 38.7 | 50.6 / 42.5 | 55.0 / 44.1 |
| $\pi_{0.5}^{\diamond}$ | Linguistics | 75.0 / 60.8 | 49.6 / 35.3 | 57.5 / 41.6 | 57.1 / 30.3 | 62.0 / 47.4 | 60.2 / 43.1 |
| **Ours $\diamond$** | **Action** | **79.8 / 66.1** | **54.1 / 38.9** | 52.3 / 37.8 | 56.8 / 39.6 | **74.6 / 54.6** | **63.5 / 47.4** |

**说明**: 统一训练设置下，本文 IS（63.5%）/PS（47.4%）双指标最优；在最难的 **unseen-texture（未见纹理）** track 上 **IS +12.6%、PS +7.2%**，再次显示对分布偏移的强鲁棒性。

### Table 4: Module Ablations (LIBERO) / 模块消融

| Name | EAR | IAR | Spatial | Object | Goal | Long | Avg. |
|------|-----|-----|---------|--------|------|------|------|
| Baseline ($\pi_{0.5}$) |  |  | 98.8 | 98.2 | 98.0 | 92.4 | 96.9 |
| #1 | ✓ |  | 99.0 | 99.4 | 98.0 | 96.6 | 98.3 |
| #2 |  | ✓ | 99.2 | 99.2 | 98.2 | 95.6 | 98.1 |
| **#3** | **✓** | **✓** | **99.4** | **99.6** | **98.8** | 96.0 | **98.5** |

**关键发现**: 单加 EAR（96.9→98.3）或单加 IAR（96.9→98.1）都涨，二者**互补**，合用达 98.5%。显式与隐式动作引导分别注入"行为归纳偏置/降低观测→动作的歧义"和"贴近可行动作分布"的先验。

### Table 5: Reference Action Parameter Ablation (EAR) / 参考动作配置消融

| Action shift | Action horizon | Equi. horizon | Spatial | Object | Goal | Long | Avg. |
|--------------|----------------|---------------|---------|--------|------|------|------|
| Baseline (1 / 10 / 10) | – | – | 98.6 | 99.0 | 96.4 | 92.2 | 96.6 |
| 1 | 10 | 10 | 99.4 | 99.4 | 98.8 | 95.0 | **98.2** |
| 2 | 5 | 10 | 99.6 | 99.6 | 98.4 | 94.4 | 98.0 |
| 1 | 30 | 30 | 99.2 | 99.2 | 97.6 | 95.6 | 97.9 |
| **2 | 15 | 30 (+EAR 默认)** | 99.0 | 99.4 | 98.0 | **96.6** | **98.3** |
| 2 | 30 | 60 | 99.4 | 99.0 | 98.2 | 95.0 | 97.9 |
| 3 | 30 | 90 | 98.8 | 99.4 | 97.4 | 96.2 | 98.0 |

**关键发现**: 各种 (action shift, horizon) 组合都优于 baseline，说明"提供动作线索"普遍有益；**较短 horizon + 适中 shift** 收益相对更强。

### Table 6: KV-cache Interaction Strategies in IAR / IAR 的 KV 交互策略消融

| Strategy | Spatial | Object | Goal | Long | Avg. |
|----------|---------|--------|------|------|------|
| Baseline | 98.8 | 98.2 | 98.0 | 92.4 | 96.9 |
| Query | 98.8 | 99.0 | 97.2 | 92.8 | 97.0 |
| Attention Pooling | 99.4 | 98.6 | 98.2 | 92.8 | 97.3 |
| **Downsample** | 99.2 | 99.2 | 98.2 | **95.6** | **98.1** |

**关键发现**: 三种策略都优于 baseline；**先下采样再聚合（Downsample）最优**，暗示 VLM 特征对动作预测含噪、需恰当的视觉语言-动作对齐机制。

### Table 7: Dataset Statistics / 数据集统计（附录）

| Type | Dataset | Embodiment | DoF | Episodes | Frames | FPS |
|------|---------|------------|-----|----------|--------|-----|
| Sim | LIBERO | Franka | 7 | 1,693 | 273,465 | 10 |
| Sim | LIBERO-Plus | Franka | 7 | 14,347 | 2,238,036 | 20 |
| Sim | VLABench | Franka | 7 | 4,713 | 528,398 | 10 |
| Real | Wipe Stain | AgiBot G1 | 22 | 177 | 356,316 | 30 |
| Real | Pour Water | AgiBot G1 | 22 | 1,821 | 5,062,506 | 30 |
| Real | Open-set Pick | AgiBot G1 | 22 | 1,936 | 219,824 | 30 |
| Real | Open-set Pick | AgileX | 14 | 962 | 251,283 | 30 |

**说明**: 三个仿真基准（Franka 7-DoF）+ 真机（AgiBot G1 22-DoF、AgileX 14-DoF）。Pour Water 单任务超 500 万帧，数据规模可观。

### Table 8: Training Details / 训练细节（附录）

| Task | Action Space | Action Horizon | State | Batch | Steps |
|------|--------------|----------------|-------|-------|-------|
| LIBERO | Delta EEF | 10 | ✗ | 128 | 40K |
| LIBERO-Plus | Delta EEF | 10 | ✗ | 128 | 100K |
| VLABench | Abs EEF | 10 | ✓ | 128 | 60K |
| Wipe Stain | Abs Joint | 30 | ✓ | 128 | 50K |
| Pour Water | Abs Joint | 30 | ✓ | 128 | 240K |
| Open-set Pick | Abs Joint | 30 | ✓ | 128 | 50K |
| Open-set Pick† (AgileX) | Abs Joint | 30 | ✓ | 128 | 50K |

**说明**: 不同任务的动作空间/horizon/状态输入与训练步数配置；真机用绝对关节空间、horizon 30。

### Table 9: Module Ablations on LIBERO-Plus / LIBERO-Plus 上模块消融（附录，监督微调、冻结 LLM）

| Name | EAR | IAR | Camera | Robot | Language | Light | Background | Noise | Layout | Avg. |
|------|-----|-----|--------|-------|----------|-------|------------|-------|--------|------|
| Baseline |  |  | 70.3 | 41.7 | 81.1 | 97.3 | 94.6 | 71.8 | 84.9 | 75.7 |
| #1 | ✓ |  | 88.7 | 63.5 | 80.4 | 94.0 | 90.2 | 89.5 | 84.2 | 83.7 |
| #2 |  | ✓ | 80.7 | 48.7 | 82.6 | 97.7 | 90.9 | 84.3 | 86.0 | 80.4 |
| **#3** | **✓** | **✓** | **91.2** | 62.5 | 80.3 | 95.1 | 91.5 | 88.3 | 84.9 | **84.1** |

**说明**: 在更难的 LIBERO-Plus 上，EAR 的增益（+8.0% 平均）尤为突出（Camera 70.3→88.7、Robot 41.7→63.5），显式动作引导对扰动鲁棒性贡献巨大；EAR+IAR 合用最佳。

### Table 10: Action-head / EAR Param & Denoise Steps / 参数量与去噪步数影响（附录，未加 IAR）

| Name | Head Param/Denoise | EAR Param/Denoise | LIBERO Avg. | LIBERO-Plus Avg. |
|------|--------------------|--------------------|-------------|------------------|
| Baseline | 300M / 10 | – | 96.6 | 75.7 |
| #1 | 600M / 10 | – | 97.6 | 74.9 |
| #2 | 600M / 20 | – | 97.5 | 75.1 |
| #3 | 300M / 5 | 300M / 5 | 97.9 | 83.9 |
| #4 (默认) | 300M / 10 | 300M / 10 | 98.3 | 83.7 |
| #5 | 300M / 10 | 150M / 10 | 97.6 | 81.7 |
| #6 | 300M / 10 | 250M / 10 | 97.5 | 83.1 |
| #7 | 300M / 10 | 500M / 10 | 97.0 | 80.9 |

**说明**: 单纯放大动作头/去噪步数（#1、#2）收益有限甚至倒退；而加入 EAR（#3–#7）在 LIBERO-Plus 上从 ~75% 跃升到 ~81–84%，说明增益来自**动作空间推理结构**而非简单堆参数；EAR 约 300M 时性价比最佳。

### Table 11: Genie Sim 3.0 + Real-World Transfer / Genie Sim 仿真与真机迁移（附录，成功率 %）

| Task | Sim ($\pi_{0.5}$) | Sim (Ours) | Real ($\pi_{0.5}$) | Real (Ours) |
|------|-------------------|------------|--------------------|-------------|
| Select Color | 86.0 | **98.8** | 85.0 | **94.0** |
| Recognize Size | 93.0 | **96.0** | **94.0** | **94.0** |
| Grasp Targets | **71.7** | 68.0 | 70.8 | **75.0** |
| Organize Objects | 52.0 | **74.0** | 60.0 | **68.4** |
| **Avg.** | 75.7 | **84.2** | 77.5 | **82.9** |

**说明**: 在 Genie Sim 3.0 仿真（平均 84.2% vs 75.7%）与真机迁移（82.9% vs 77.5%）上均超 $\pi_{0.5}$，验证 sim-to-real 迁移能力。

### Table 12: Efficiency vs Performance Ablation / 效率-性能消融（附录）

| Name | EAR | IAR | Param. | Latency | LIBERO Avg. SR | LIBERO-Plus Avg. SR |
|------|-----|-----|--------|---------|----------------|----------------------|
| Baseline |  |  | 3.35B | 91ms | 96.9 | 75.7 |
| #1 | ✓ |  | 3.80B | 110ms | 98.3 | 83.7 |
| #2 |  | ✓ | 3.36B | 93ms | 98.1 | 80.4 |
| **#3** | **✓** | **✓** | **3.81B** | **112ms** | **98.5** | **84.1** |

**说明**: 完整模型 3.81B 参数、单帧延迟 112ms（RTX 4090）。IAR 几乎不增延迟（+2ms）；EAR 增 ~19ms 但带来最大性能跃升。作者认为相对收益而言额外开销"较为温和"。

---

## 实验

### 数据集 / 基准

| 基准 | 规模 | 特点 | 用途 |
|------|------|------|------|
| [[LIBERO]] | 4 套件×10 任务，每任务 50 示范，每任务 50 trial（共 2,000 rollout） | 单臂 Franka，考空间/物体/目标/长程四种能力 | 训练/测试 |
| [[LIBERO-Plus]] | 10,030 评测 episode；7 维扰动 | 系统评估分布偏移鲁棒性（相机/机器人/语言/光照/背景/噪声/布局） | 训练/测试 |
| [[VLABench]] | 5 公开 track | 基于 ManiSkill3，用 Intention/Progress Score 评估，含 commonsense、未见纹理等 | 训练/测试 |
| 真实世界 | AgiBot G1 / AgileX，三任务 | 接触丰富（擦污）、精细操控（倒水）、指令遵循（开放集抓取）、跨本体 | 训练/测试 |

### 实现细节

- **基线框架**: 在 $\pi_{0.5}$ 之上实现；视觉编码器 [[SigLIP]]，LLM 主干 [[Gemma]]-2B（$N=18$ 层，$d=2048$），输入帧 $224\times224$
- **EAR**: 18 层轻量 Transformer；**IAR**: 每层 query 行维 $M=1$，下采样维 $d'=128$
- **horizon**: 参考动作 $H^{ref}=15$、策略输出 $H=10$；action shift 分别为 2 和 1；损失平衡因子 $\lambda_1=\lambda_2=0.5$
- **训练**: AdamW + cosine 衰减，warmup 10K 步，peak lr $5\mathrm{e}{-5}$，grad clip 1.0，EMA decay 0.999；bfloat16
- **硬件**: 单节点 8× NVIDIA H100 训练；推理用单卡 RTX 4090（完整模型 3.81B / 112ms）

### 关键实验结论

- **仿真**: LIBERO 平均 98.5%（SOTA，较 $\pi_{0.5}$ +1.6%，Long track 提升最显著）；LIBERO-Plus 监督微调 88.0%、零样本 86.6%（均 SOTA）；VLABench IS 63.5% / PS 47.4%（SOTA，未见纹理大涨）。
- **真机**: AgiBot G1 + AgileX 三任务平均 66.7%，高于 $\pi_{0.5}$（61.0%）、$\pi_0$（33.8%），跨本体一致提升（另见 Table 11 的 sim/real 迁移）。
- **消融**: EAR、IAR 单独有效且互补（Table 4/9）；IAR 用 Downsample 交互最优（Table 6）；增益来自动作空间推理结构而非堆参数（Table 10）；冻结 LLM（$\diamond$）仍达 SOTA。

---

## 批判性思考

### 优点
1. **范式新颖且定位清晰**: 把 CoT 从语言/视觉空间显式搬到动作空间，理论动机（语义-运动学鸿沟）明确，并用 Figure 1 的三路对比让贡献一目了然；"显式 + 隐式"双形态的拆解自洽。
2. **实证扎实**: 跨 3 个仿真基准（含专门测分布偏移的 LIBERO-Plus，10K+ episode 统计可靠）+ 真机 + 跨本体，对照 20+ 强基线；消融充分（模块、参考动作配置、KV 交互、参数 vs 步数、效率-延迟）。尤其 Table 10 用"堆参数无效、加结构有效"有力支持其增益归因。
3. **工程友好**: 冻结 LLM 主干仍达 SOTA，IAR 几乎零延迟开销，整体在 RTX 4090 上 112ms 可推理；EAR 用 teacher forcing 稳定训练、推理切自条件，细节考虑周到。

### 局限性
1. **额外计算开销**: 作者自承推理模块增计算成本（EAR +~19ms / +0.45B 参数），对资源受限平台部署不利。
2. **动作表征受限**: 当前以 action chunk（关节角/末端位姿序列）为动作表征，缺乏显式几何结构，难支撑物体中心协调、接触几何等高层空间推理，ACoT 潜力未完全释放（作者列为未来方向：转向几何可解释的 3D 动作空间）。
3. **超参解释偏经验**: $M=1$、$d'=128$、$H^{ref}=15$、action shift=2 等关键设置主要靠消融经验选定，缺系统敏感度分析；teacher forcing 使训练/推理分布存在 gap（训练用真值参考、推理用 EAR 自生成），其影响未深入量化。
4. **真机任务范围**: 真机仅 3 类桌面任务、成功率绝对值仍偏低（平均 66.7%），长程、双臂、可变形等更难场景未覆盖。

### 潜在改进方向
1. 把动作表征升级为带几何结构的 3D 表示（如关键点/物体中心坐标），让 ACoT 在几何可解释空间推理（作者已点出）。
2. 量化 teacher forcing 的训练-推理 gap，探索 scheduled sampling 或在线自条件训练以闭合差距。
3. 引入更难的长程/接触/双臂真机任务与更强扰动，验证 ACoT 在复杂物理交互下的可扩展性；并对 EAR/IAR 的超参做自动搜索以减少经验依赖。

### 可复现性评估
- [x] 代码开源（https://github.com/AgibotTech/ACoT-VLA，论文声明）
- [ ] 预训练模型（论文未明确声明 release 权重）
- [x] 训练细节完整（附录给出 backbone、超参、训练配置表 7/8）
- [x] 数据集可获取（LIBERO/LIBERO-Plus/VLABench 公开；真机数据自采，未声明 release）

---

## 速查卡片

> [!summary] ACoT-VLA: Action Chain-of-Thought for VLA
> - **核心**: 把 CoT 推理从语言/视觉空间搬到**动作空间**，用"显式参考轨迹 + 隐式动作先验"双形态引导动作头，弥合语义-运动学鸿沟。
> - **方法**: 共享 VLM 主干（SigLIP+Gemma-2B）→ EAR（轻量 Transformer 经流匹配自合成参考轨迹 $Z^{\text{ex}}$）+ IAR（下采样 VLM KV cache、交叉注意力抽取 $Z^{\text{im}}$）→ AGP 头双路交叉注意力融合后去噪；teacher forcing 稳定训练、推理自条件。
> - **结果**: LIBERO 98.5%（+1.6% vs $\pi_{0.5}$）、LIBERO-Plus 88.0%/86.6%、VLABench IS 63.5%/PS 47.4%、真机平均 66.7%（vs $\pi_{0.5}$ 61.0%）；3.81B / 112ms@4090，冻结 LLM 亦达 SOTA。
> - **代码**: https://github.com/AgibotTech/ACoT-VLA

---

*笔记创建时间: 2026-06-29*
