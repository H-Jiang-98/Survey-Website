---
title: "ConsisVLA-4D: Advancing Spatiotemporal Consistency in Efficient 3D-Perception and 4D-Reasoning for Robotic Manipulation"
method_name: "ConsisVLA-4D"
authors: [Wei Li, Jizhihui Liu, Li Yixing, Junwen Tong, Rui Shao, Liqiang Nie]
year: 2026
venue: CVPR
tags: [VLA, 3D-perception, 4D-reasoning, spatiotemporal-consistency, multi-view, token-sparsification, robotic-manipulation]
zotero_collection: _inbox
image_source: online
arxiv_html: https://arxiv.org/html/2605.05126v1
created: 2026-06-29
---

# ConsisVLA-4D: Advancing Spatiotemporal Consistency in Efficient 3D-Perception and 4D-Reasoning for Robotic Manipulation

## 元信息

| 项目 | 内容 |
|------|------|
| 作者 | Wei Li, Jizhihui Liu, Li Yixing, Junwen Tong, Rui Shao, Liqiang Nie |
| 机构 | 哈尔滨工业大学（深圳）等（通讯/主要单位，Rui Shao、Liqiang Nie 团队） |
| 会议 | CVPR 2026 |
| 类别 | VLA（Vision-Language-Action）/ 3D 感知 + 4D 推理 |
| 日期 | 2026-05（arXiv v1） |
| 项目主页 | （论文声明开源，地址在 arXiv 摘要末尾给出） |
| 链接 | [arXiv](https://arxiv.org/abs/2605.05126) / [HTML](https://arxiv.org/html/2605.05126v1) |

---

## 一句话总结

> 用 [[CV-Aligner]]（跨视图语义对齐）+ [[CO-Fuser]]（跨物体几何聚合）从 2D 观测高效构造 3D 表征，再用 [[CS-Thinker]] 把"预测未来动态物体 / 全局深度"的隐式知识注入动作生成，仅用约 1/8 的视觉 token 就在 [[LIBERO]] 上比 [[OpenVLA]] 提升 21.6%、推理加速 2.3×。

---

## 核心贡献

1. **统一高效的 3D-感知到 4D-推理框架**: 提出 [[ConsisVLA-4D]]，把"从 2D 观测构造 3D 表征"与"面向未来场景的 4D 视觉推理"统一进一个框架，在不引入额外深度/点云传感器、不付出昂贵算力的前提下，提升时空一致性（spatiotemporal consistency）。
2. **两个空间一致性模块**: [[CV-Aligner]] 保证**跨视图物体语义一致性**（cross-view object semantic consistency），[[CO-Fuser]] 保证**跨物体空间几何一致性**（cross-object spatial geometric consistency），共同改善对复杂 3D 场景的理解。
3. **跨场景时空一致性推理**: 设计 [[CS-Thinker]] + [[Spatiotemporal Consistency Attention|SC-Attn]]，在训练期学习"动作发生后未来动态物体 / 全局深度"的隐式知识，推理期无需显式生成这些中间量即可输出稳定、可靠的动作。
4. **性能与效率双 SOTA**: 大量仿真与真实实验证明，相比 OpenVLA 在 LIBERO / 真实平台上分别取得 **21.6% / 41.5%** 的性能提升与 **2.3× / 2.4×** 的推理加速。

---

## 问题背景

### 要解决的问题
当前 [[VLA]] 模型主要把 **2D 观测映射到动作**，在**时空感知与推理**上有两大短板，本文要在不增加传感器与算力负担的前提下同时解决它们：
1. **3D 空间理解不足**：要么靠点云/深度等额外传感器（贵、平台受限），要么做粗糙的 2D→3D 映射（物体错位、空间关系歧义）。
2. **4D 时空推理欠发达**：动作执行中场景动态变化，现有方法缺乏对当前空间状态的完整理解和对场景动态演化的知识，难以与"预测的未来场景"建立一致关联，导致视觉推理受损、动作生成不稳。

### 现有方法的局限
- **显式 3D/4D 输入类**（[[3D-VLA]]、[[PointVLA]]、[[GeoVLA]]、[[Lift3D]]、[[4D-VLA]]、[[TraceVLA]]）：依赖专用传感器，平台灵活性差、算力开销大。
- **2D→3D 投影类**（[[SpatialVLA]]、[[BridgeVLA]]、[[Evo-0]]、[[GeoAware-VLA]]）：从 2D 推断 3D，但有投影偏差、几何不一致、遮挡误差。
- **世界模型类**（[[WorldVLA]]、[[World4Omni]]、[[V-JEPA 2]]）：主要预测**未来图像帧**，并未实现真正的 3D 空间理解或 4D 推理——这正是 [[CoT-VLA]] 等"未来帧预测"路线被本文批评的核心：未来帧 ≠ 指令对齐的未来场景，破坏时空一致性。

### 本文的动机
作者从**人类操作行为**取灵感：人靠双目视觉/移动在不同视角间维持一致的空间感知（物体位置与关系），并基于这种稳定感知预测未来空间状态，从而在执行中保持时间稳定。由此引出两个核心问题并对应设计：
1. 如何在不增加算力的情况下，从 2D 观测**高效生成 3D 表征**？→ E3D（Efficient 3D-Perception，CV-Aligner + CO-Fuser）。
2. 如何通过 **4D 视觉推理增强时空一致性**来优化动作预测？→ CS-Thinker（4D-Reasoning）。

作者把这一机制抽象为：**2D --construction--> 3D --prediction/refinement--> 4D**（见公式2）。

---

## 方法详解

### 模型架构

[[ConsisVLA-4D]] 以 [[OpenVLA]]（7B，LoRA 微调）为主干，多视角观测 $\mathcal{I}=\{M,L,R\}$（Main/Left/Right）下分两个阶段、三个核心模块组织（见 Figure 1、Figure 2、Figure 3、Figure 4）：

- **输入**: 多视角 RGB 观测 $\mathcal{I}=\{M,L,R\}$ + 语言指令 $\mathbf{t}$ + 机器人状态
- **三类视觉编码器**:
  - [[SigLIP]] → 语义表征 $\mathbf{z}^{\text{sem}}$（token 携带语言语义）
  - [[DINOv2]] → 几何表征 $\mathbf{z}^{\text{geo}}$（捕捉几何一致性）
  - [[VGGT]] → 3D 空间表征 $\mathbf{z}^{\text{3D}}$（输出深度图 $D_i$、点图 $P_i$、特征网格 $G_i$，含点追踪先验）
- **E3D（Efficient 3D-Perception）阶段**: [[CV-Aligner]]（红）+ [[CO-Fuser]]（橙）
- **4D-Reasoning 阶段**: [[CS-Thinker]] + [[Spatiotemporal Consistency Attention|SC-Attn]]
- **输出**: 并行解码的 [[Action Chunking|动作块]] $\hat{\mathbf{A}}$
- **token 压缩**: 仅用约 **1/8**（CV-Aligner）与 **1/12~1/8**（CO-Fuser）的原始视觉 token；推理期隐式知识 < 观测-指令序列的 10%

### 核心模块

#### 模块1: CV-Aligner（Cross-View Aligner，跨视图对齐器）

**设计动机**: 过滤冗余视觉、只保留**与指令相关**的物体 token，并在多视角间对齐**同一物体身份**，从而以极少 token 维持跨视图语义一致性。

**具体实现**（三步，见 Figure 2 红色路径）:
- **Step 1 — FiLM 调制**: 对 SigLIP 每层视觉特征施加由指令 $\mathbf{t}$ 生成的逐层缩放/平移向量 $\gamma_i(\mathbf{t}),\beta_i(\mathbf{t})$（[[FiLM]]），逐 token 增强观测-指令语义对齐（公式8）。
- **Step 2 — ES-Selection（显式语义物体选择）**: 用文本编码器 $f_t^{\text{SigLIP}}$ 编码指令，与每个视觉 token 算余弦相似度（公式10），取 **Top-K（默认 K=32，约 256→32，即每视图保留 1/8）** 最相关 token 作为物体 token $\mathbf{z}_i^{\text{obj}}$（公式11–12）。
- **Step 3 — Single-Fusion（逐帧融合）**: 以 $\mathbf{z}_i^{\text{obj}}$ 为 Query、VGGT 的 $\mathbf{z}_i^{\text{3D}}$ 为 Key/Value，叠 $N$ 层交叉注意力 + 残差（公式13），利用 VGGT 点追踪先验把 3D 信息注入物体 token，得到 $\mathbf{z}_i^{\text{obj-3D}}$，实现跨视图物体语义一致性（仅 1/8 token）。

#### 模块2: CO-Fuser（Cross-Object Fuser，跨物体融合器）

**设计动机**: 单视图深度有尺度歧义、物体间空间关系含糊；通过多视角联合处理，把跨视图几何关系压进一小撮**聚合 token**，消除空间歧义。

**具体实现**（三步，见 Figure 2 橙色路径）:
- **Step 1 — Group-Fusion（逐块多帧融合）**: 在 [[DINOv2]] 与 [[VGGT]] 编码器的每个 block，把几何特征 $\mathbf{z}_l^{\text{geo}}$ 与按层余弦衰减加权的 3D 特征 $\mathbf{z}_l^{\text{3D}}$ 逐元素融合（公式14），权重 $\alpha_l$ 随层深做**余弦衰减**（公式15，$\alpha_0=\psi=0.2$，$\alpha_{\mathcal{L}'}=\psi\cdot\delta=0.01$，$\mathcal{L}'=24$）。冻结的 VGGT 先验随深度淡出，学得的融合特征逐渐占主导。
- **Step 2 — 聚合 token 拼接**: 初始化一组聚合 token $\mathbf{z}_0^{\text{agg-3D}}$（固定 64 个，单臂双视压 1/8、双臂三视压 1/12）拼到几何特征前。
- **Step 3 — IG-Aggregation（隐式几何关系聚合）**: 用 **Block-wise Causal Self-Attention（BC-Attn，见 Figure 8）**——$\mathbf{z}_l^{\text{geo-3D}}$ 与 $\mathbf{z}_l^{\text{agg-3D}}$ 之间用因果注意力、各自集合内用双向注意力（公式16），让跨视图信息汇入聚合 token；只取最后一层 $\mathbf{z}_{\mathcal{L}'}^{\text{agg-3D}}$，维持跨物体空间几何一致性（1/12~1/8 token）。

#### 模块3: CS-Thinker（Cross-Scene Thinker，跨场景思考器）

**设计动机**: 动作展开使场景持续变化，把"单场景内空间一致性"扩展为"跨场景时空一致性"；通过**训练期学隐式知识、推理期免显式生成**，兼顾效果与效率。

**具体实现**（见 Figure 4，含 [[Spatiotemporal Consistency Attention|SC-Attn]]）:
- **多视图物体 → 单视图动态物体**: 为每个视角初始化一组动态 token $\mathbf{0}_i^{\text{dyn-4D}}$（3 组 ×4=12 个），由对应 $\mathbf{z}_i^{\text{obj-3D}}$ 与指令 $\mathbf{t}$ 引导（公式17），预测动作发生后固定视角 $i^*$ 的动态物体，监督来自 [[CoTracker]]（公式18，带物体掩码 $\mathbf{m}_{i^*}^{\text{obj-3D}}$）。
- **抽象关系 → 具体全局深度**: 初始化一组深度 token $\mathbf{0}^{\text{dep-4D}}$（1 组 ×4=4 个），由 $\mathbf{z}_{\mathcal{L}'}^{\text{agg-3D}}$ 与 $\mathbf{t}$ 引导（公式19），解码三视角全局深度，监督来自 [[Depth-Anything]]（公式20）。
- **并行动作解码**: 把动态/深度预测与动作 token $\mathbf{0}^A$ 放进**同一上下文窗口**，SC-Attn 并行解码出 $\hat{\mathbf{A}}$，动作用 L1 损失 $\mathcal{L}_{\text{action}}$。**推理期不显式生成动态物体/深度**，只靠学到的隐式语义/几何知识。
- 解码器结构: 1 个动态解码器 + 3 个独立深度解码器，每个 8 层 Transformer、hidden 1024、16 头、FFN ratio 4。

### 关键公式与机制

#### 公式1: [[VGGT]] 的 3D 表征提取

$$
\text{DPT}\!\left(f^{\text{VGGT}}_{v}(\mathbf{x}_{i})_{i=1}^{M}\right)=(D_{i},P_{i},G_{i})_{i=1}^{M}
$$

**含义**: VGGT 对同一 3D 场景的 $M$ 张 RGB 输出深度图、点图、特征网格；其潜在特征 $\mathbf{z}^{\text{3D}}=f_v^{\text{VGGT}}(\mathbf{x})$ 为后续局部语义过滤与全局几何建模提供 3D 先验。

**符号说明**:
- $D_i,P_i,G_i$: 第 $i$ 视图的深度图 / 点图 / 用于点追踪的特征网格
- $\text{DPT}(\cdot)$: Dense Prediction 头；$M$: 同场景视图数

#### 公式2: 人类对齐的时空一致性建模范式

$$
\text{2D} \xrightarrow{\text{construction}} \text{3D} \xrightarrow{\text{prediction}} \text{4D} \xrightarrow{\text{refinement}}
$$

**含义**: 类比人眼成 2D 像、人脑整合空间线索并预测未来——从 2D 构造时空一致的 3D 表征（跨视图身份匹配 + 视角互补消歧），再推理动态空间场景以增强时间稳定性。

#### 公式3: E3D — CV-Aligner 的物体-3D 表征

$$
\mathbf{z}_{\{M,L,R\}}^{\text{obj-3D}}=f_{\text{SF}}\!\left(f_{\text{ES-S}}(\mathbf{z}_{\{M,L,R\}}^{\text{sem}},\mathbf{t}),\ \mathbf{z}_{\{M,L,R\}}^{\text{3D}}\right)
$$

**含义**: 多视角语义特征先经 ES-Selection 按指令选 token，再经 Single-Fusion 与 VGGT 3D 特征逐帧融合，得到指令相关的物体-3D 表征。

**符号说明**:
- $f_{\text{ES-S}}(\cdot)$: 显式语义物体选择；$f_{\text{SF}}(\cdot)$: 逐帧 Single-Fusion；$\mathbf{t}$: 指令嵌入

#### 公式4: E3D — CO-Fuser 的几何关系聚合（逐块）

$$
(\mathbf{z}_{l+1}^{\text{geo-3D}},\mathbf{z}_{l+1}^{\text{agg-3D}})=f_{\text{IG-A}}\!\left(f_{\text{GF}}(\mathbf{z}_{l}^{\text{geo}},\mathbf{z}_{l}^{\text{3D}}),\ \mathbf{z}_{l}^{\text{agg-3D}}\right)
$$

**含义**: 在 DINOv2/VGGT 编码器每个 block 内先 Group-Fusion 融合几何与 3D 特征，再做 Implicit Geometric Relation Aggregation 把跨视图关系汇入聚合 token。

**符号说明**:
- $f_{\text{GF}}$: Group-Fusion；$f_{\text{IG-A}}$: IG-Aggregation；$\mathbf{z}^{\text{agg-3D}}$: 聚合 token

#### 公式5–6: 4D-Reasoning（动态物体 / 全局深度）

$$
\hat{\mathbf{z}}_{M}^{\text{dyn-4D}}=\text{4D-Reasoning}\!\left(\mathbf{z}_{\{M,L,R\}}^{\text{obj-3D}},\mathbf{t},\mathbf{0}_{\{M,L,R\}}^{\text{dyn-4D}}\right)
$$

$$
\hat{\mathbf{z}}_{\{M,L,R\}}^{\text{dep-4D}}=\text{4D-Reasoning}\!\left(\mathbf{z}_{\mathcal{L}^{\prime}}^{\text{agg-3D}},\mathbf{t},\mathbf{0}^{\text{dep-4D}}\right)
$$

**含义**: 用 CV-Aligner 的物体语义 token 预测动作发生后单视图**未来动态物体**；用 CO-Fuser 的聚合几何 token 推理多视角**未来全局深度**。

**符号说明**:
- $\mathbf{0}_{\{M,L,R\}}^{\text{dyn-4D}},\mathbf{0}^{\text{dep-4D}}$: 初始化的动态 / 深度 token；$\mathcal{L}'$: 编码器层数

#### 公式7: SC-Attn 并行动作解码

$$
\hat{\mathbf{A}}=\text{SC-Attn}\!\left(\mathbf{z}_{\{M,L,R\}}^{\text{obj-3D}},\ \mathbf{z}_{\mathcal{L}^{\prime}}^{\text{agg-3D}},\ \mathbf{t},\ \mathbf{0}_{\{M,L,R\}}^{\text{dyn-4D}},\ \mathbf{0}^{\text{dep-4D}},\ \mathbf{0}^{A}\right)
$$

**含义**: 把物体-3D、聚合几何、指令、动态/深度初始 token 与动作初始 token $\mathbf{0}^A$ 放进同一上下文，由 SC-Attn 并行解出动作块 $\hat{\mathbf{A}}$；动态/深度预测充当动作生成的中间视觉推理。

#### 公式8: ES-Selection 的 FiLM 调制

$$
\tilde{\mathbf{z}}_{i,l}^{\text{sem}}=(\mathbf{1}+\gamma(\mathbf{t}))\odot\text{Self-Attn}(\mathbf{z}_{i,l}^{\text{sem}})+\beta(\mathbf{t})
$$

**含义**: 在 SigLIP 第 $l$ 层用指令生成的 $\gamma,\beta$ 对自注意力输出做逐 token 缩放/平移，强化视觉-语言语义对齐。

**符号说明**:
- $\gamma(\mathbf{t}),\beta(\mathbf{t})$: 由指令生成的 [[FiLM]] 缩放/平移向量；$\odot$: 逐元素乘

#### 公式9–12: Top-K 语义物体选择

$$
s_{i,j}=\operatorname{sim}\!\left(\mathbf{z}_{i}^{\text{sem},j},\mathbf{W_t}\!\cdot\!\mathbf{t}\right)=\frac{\mathbf{z}_{i}^{\text{sem},j}(\mathbf{W_t}\!\cdot\!\mathbf{t})^{\top}}{\|\mathbf{z}_{i}^{\text{sem},j}\|_{2}\cdot\|\mathbf{W_t}\!\cdot\!\mathbf{t}\|_{2}}
$$

$$
\mathcal{S}_{i}=\operatorname{Top\text{-}K}\!\left(\{s_{i,1},\ldots,s_{i,N_i}\},K\right),\qquad
\mathbf{z}_{i}^{\text{obj}}=\{\mathbf{z}_{i}^{\text{sem},j}\mid j\in\mathcal{S}_i\}=f_{\text{ES-S}}(\mathbf{z}_{i}^{\text{sem}},\mathbf{t})
$$

**含义**: 每个视觉 token 与指令算余弦相似度 $s_{i,j}$，保留 Top-K（默认 32）最相关 token 作为物体 token，去除大量视觉冗余。

**符号说明**:
- $\mathbf{W_t}$: 文本到视觉维度的映射矩阵；$\mathcal{S}_i$: 第 $i$ 视图选中 token 索引集；$N_i$: 该视图 token 数

#### 公式13: Single-Fusion（注入 3D 信息）

$$
\mathbf{z}_{i}^{\text{obj-3D}}=\Big(\text{FFN}\big(\text{Cross-Attn}(\mathbf{z}_{i}^{\text{obj}},\mathbf{z}_{i}^{\text{3D}})\big)+\text{Res}(\mathbf{z}_{i}^{\text{obj}})\Big)\Big|_{\text{Layer}=1,\ldots,N}=f_{\text{SF}}(\mathbf{z}_{i}^{\text{obj}},\mathbf{z}_{i}^{\text{3D}})
$$

**含义**: 物体 token 作 Query、VGGT 3D 特征作 K/V，经 $N$ 层交叉注意力 + 残差融合，利用点追踪先验建立跨视图同一物体身份关联。

**符号说明**:
- $\text{Res}(\cdot)$: 残差连接；$N$: Transformer 层数

#### 公式14: Group-Fusion（几何-3D 加权融合）

$$
\mathbf{z}_{l}^{\text{geo-3D}}=(1-\alpha_{l})\odot\mathbf{z}_{l}^{\text{geo}}+\alpha_{l}\odot\mathbf{z}_{l}^{\text{3D}}=f_{\text{GF}}(\mathbf{z}_{l}^{\text{geo}},\mathbf{z}_{l}^{\text{3D}})
$$

**含义**: 第 $l$ block 内按权重 $\alpha_l$ 把几何特征与 3D 特征逐元素融合。

**符号说明**:
- $\alpha_l$: 随层深余弦衰减的融合权重；$\odot$: 逐元素乘

#### 公式15: 余弦衰减权重 $\alpha_l$

$$
\alpha_{l}=\psi\cdot\left(\delta+(1-\delta)\cdot\frac{1+\cos\!\left(\dfrac{l\pi}{\mathcal{L}^{\prime}}\right)}{2}\right)
$$

**含义**: 浅层（$l\to 0$）权重高且斜率近 0，让模型吸收 VGGT 几何先验；深层（$l\to\mathcal{L}'$）平滑降到最小、退出先验约束；变化集中在中层（$l\approx\mathcal{L}'/2$）以整合高低层信息。消融（Table 8）证明这优于线性衰减。

**符号说明**:
- $\psi$: 最大权重（=0.2）；$\delta$: 控制最小权重（$\alpha_{\mathcal{L}'}=\psi\delta=0.01$）；$\mathcal{L}'$: 层数（=24）

#### 公式16: IG-Aggregation（BC-Attn）

$$
(\mathbf{z}_{l+1}^{\text{geo-3D}},\mathbf{z}_{l+1}^{\text{agg-3D}})=\text{BC-Attn}\!\left(\mathbf{z}_{l}^{\text{geo-3D}}\oplus\mathbf{z}_{l}^{\text{agg-3D}}\right)=f_{\text{IG-A}}(\mathbf{z}_{l}^{\text{geo-3D}},\mathbf{z}_{l}^{\text{agg-3D}})
$$

**含义**: 块级因果自注意力——跨集合用因果注意力、集合内用双向注意力，把多视图几何信息有序汇入聚合 token。

**符号说明**:
- $\oplus$: token 集拼接；$\text{BC-Attn}$: Block-wise Causal Self-Attention（见 Figure 8）

#### 公式17–20: 4D-Reasoning 的引导与损失

$$
\forall i\in\mathcal{I},\quad \mathbf{0}_{i}^{\text{dyn-4D}}\xleftarrow{\text{guide}}(\mathbf{z}_{i}^{\text{obj-3D}},\mathbf{t}),\qquad
\mathbf{0}^{\text{dep-4D}}\xleftarrow{\text{guide}}(\mathbf{z}_{\mathcal{L}^{\prime}}^{\text{agg-3D}},\mathbf{t})
$$

$$
\mathcal{L}_{\text{dyn-4D}}=\big\|(\hat{\mathbf{z}}_{i^{*}}^{\text{dyn-4D}}\odot\mathbf{m}_{i^{*}}^{\text{obj-3D}})-({\mathbf{z}}_{i^{*}}^{\text{dyn-4D}}\odot\mathbf{m}_{i^{*}}^{\text{obj-3D}})\big\|_{2}^{2},\qquad
\mathcal{L}_{\text{dep-4D}}=\sum_{i=1}^{N_i}\big\|\hat{\mathbf{z}}_{i}^{\text{dep-4D}}-{\mathbf{z}}_{i}^{\text{dep-4D}}\big\|_{2}^{2}
$$

**含义**: 动态/深度 token 分别由物体表征、聚合几何表征 + 指令引导；动态损失在物体掩码区域做 L2，深度损失对各视角 L2 求和。这些只在训练期学习，推理期靠隐式知识。

**符号说明**:
- $\mathbf{m}_{i^*}^{\text{obj-3D}}$: 定位物体位置的掩码；$i^*$: 固定预测视角；$N_i$: 视角数

#### 公式21: 总训练目标

$$
\mathcal{L}_{\text{total}}=\mathcal{L}_{\text{action}}+\mathcal{L}_{\text{dyn-4D}}+\mathcal{L}_{\text{dep-4D}}
$$

**含义**: 动作 L1 损失 + 动态物体损失 + 全局深度损失三者相加，端到端联合优化，使隐式时空知识服务于动作生成。

---

## 关键图表

<!-- 图片均使用 arXiv HTML 在线链接，未本地化 -->

### Figure 1: Comparison with Existing Paradigms / 与现有范式对比

![Figure 1](https://arxiv.org/html/2605.05126v1/fig/intro.png)

**说明**: 对比四类范式。Para. A 用显式 3D/4D 输入（点云、深度图、历史帧），Para. B 把 2D 投影到 3D，Para. C 从 2D 预测 3D 表征；本文 Para. D 在统一框架内把 **3D-感知扩展到 4D-推理**：CV-Aligner 抽取指令相关、跨视图相关的空间物体；CO-Fuser 聚合多视图几何关系；CS-Thinker 基于未来动态物体与全局深度的隐式知识推理动作。仅用约 1/8 原始视觉输入即达成时空一致性。右侧示意鲁棒高效的操作效果。这张图直接点明本文相对四类已有路线的定位。

### Figure 2: Efficient 3D-Perception / 高效 3D 感知（CV-Aligner + CO-Fuser）

![Figure 2](https://arxiv.org/html/2605.05126v1/fig/3d.png)

**说明**: E3D 的双路设计。红色 [[CV-Aligner]] = 显式语义物体选择 + 逐帧 Single-Fusion，保证跨视图物体语义一致性；橙色 [[CO-Fuser]] = 隐式几何关系聚合 + 多帧 Group-Fusion，保证跨物体空间几何一致性。两条路对应公式3 与公式4，是理解整个 3D 感知阶段的主图。

### Figure 3: Mechanism from 3D-Perception to 4D-Reasoning / 3D 感知到 4D 推理的机制

![Figure 3](https://arxiv.org/html/2605.05126v1/fig/pipline.png)

**说明**: CV-Aligner 跨视图选出身份匹配的空间物体，经 4D-Reasoning 进一步预测动作发生后从一视角到另一视角的**同身份动态物体**；CO-Fuser 聚合全局几何关系消歧，经 4D-Reasoning 进一步从同一几何表征预测不同视角的**全局深度**。这一推进既增强时空一致性，又减少视觉输入、提升训练/推理效率——是连接感知与推理的总流程图。

### Figure 4: Efficient 4D-Reasoning / 高效 4D 推理（CS-Thinker + SC-Attn）

![Figure 4](https://arxiv.org/html/2605.05126v1/fig/4d.png)

**说明**: CS-Thinker 配 [[Spatiotemporal Consistency Attention|SC-Attn]]：1) 三组动态 token 在不同视角物体特征引导下解码单视角动态物体（[[CoTracker]] 监督）；2) 一组深度 token 在多视图几何关系引导下解码三视角全局深度（[[Depth-Anything]] 监督）；3) 这些预测作为中间视觉推理，与动作 token 并行解码。IK = 隐式知识（implicit knowledge）。对应公式17–21。

### Figure 5: Simulation Results on RoboTwin 2.0 / RoboTwin 2.0 仿真结果

![Figure 5](https://arxiv.org/html/2605.05126v1/fig/vis0.png)

**说明**: RoboTwin 2.0 上 7 个 ALOHA 双臂任务（Click Alarmclock、Turn Switch、Put Bottles Dustbin、Open Laptop、Press Stapler、Place Empty Cup、Blocks Ranking RGB）的成功率柱状图，每任务 100 trials。ConsisVLA-4D（最右深色柱）在多数任务上领先 Diffusion Policy / ACT / RDT-1B / $\pi_0$，如 Open Laptop 0.90、Place Empty Cup 0.70、Click Alarmclock 0.87，验证细粒度空间感知与鲁棒双臂操作能力。

### Figure 8: Block-Wise Causal Self-Attention / 块级因果自注意力（附录）

![Figure 8](https://arxiv.org/html/2605.05126v1/fig/supp_1.png)
![Figure 8 (b)](https://arxiv.org/html/2605.05126v1/fig/supp_A.png)

**说明**: CO-Fuser 中 IG-Aggregation 的 [[BC-Attn]] 细节。多视图观测 $\mathcal{I}=\{M,L,R\}$，几何-3D token 与聚合 token 之间用因果注意力、各自集合内双向注意力，压缩比可达 **1/12**；高亮块表示 Aggregation Token。对应公式16。

### Figure 9: Real-World Task 1 & 2 Execution / 真实任务 1、2 执行可视化（附录）

![Figure 9](https://arxiv.org/html/2605.05126v1/fig/freecompress-supp_task1_2.png)

**说明**: Task 1（Microwave Operation：把面包放入碗、碗送进微波炉、关门）与 Task 2（Banana Peeling：一臂稳定、一臂剥皮并放盘）的关键执行帧，展示长程、嵌套约束与双臂协调能力。

### Figure 10: Additional CV-Aligner Visualizations / CV-Aligner 注意力可视化（附录）

![Figure 10](https://arxiv.org/html/2605.05126v1/fig/freecompress-supp_vis1.png)

**说明**: 四条指令（pick up alphabet soup / butter / cream cheese / milk）下 Main View 与 Wrist View 随时间的注意力热图。CV-Aligner 能从冗余背景中过滤出与指令语义高度匹配的目标物体，佐证 Top-K 选择（公式11–12）去冗余的有效性。

### Figure 11: Additional CO-Fuser Visualizations / CO-Fuser 注意力可视化（附录）

![Figure 11](https://arxiv.org/html/2605.05126v1/fig/freecompress-supp_vis2.png)

**说明**: 聚合 token 与原始视觉 patch token 的注意力热图。不同于 CV-Aligner 的单点聚焦，CO-Fuser 呈**分布式注意力**，覆盖任务相关的多个空间节点（如物体与篮筐的相对距离），用 1/12~1/8 token 隐式捕捉跨视图几何关系，与 CV-Aligner 互补。

### Table 1: Simulation Results on LIBERO / LIBERO 仿真（成功率 %）

| Method | Spatial | Object | Goal | Long | Avg. |
|--------|---------|--------|------|------|------|
| Diffusion Policy [RSS'23] | 78.3 | 92.5 | 68.3 | 50.5 | 72.4 |
| Octo [RSS'24] | 78.9 | 85.7 | 84.6 | 51.1 | 75.1 |
| OpenVLA [CoRL'24] | 84.7 | 88.4 | 79.2 | 83.7 | 76.5 |
| OpenVLA-OFT [RSS'25] | 97.6 | 98.4 | 97.9 | 94.5 | 97.1 |
| $\pi_0$ [RSS'25] | 96.8 | 98.8 | 95.8 | 85.2 | 94.2 |
| $\pi_0$-Fast [RSS'25] | 96.4 | 96.8 | 88.6 | 60.2 | 85.5 |
| $\pi_{0.5}$ [arXiv'25] | 98.8 | 98.2 | 98.0 | 92.4 | 96.9 |
| TraceVLA [ICLR'25] | 84.6 | 85.2 | 75.1 | 54.1 | 74.8 |
| CoT-VLA [CVPR'25] | 87.5 | 91.6 | 87.6 | 69.0 | 83.9 |
| SpatialVLA [RSS'25] | 88.2 | 89.9 | 78.6 | 55.5 | 78.1 |
| **ConsisVLA-4D** | **98.8** | **99.8** | **98.0** | **95.6** | **98.1** |

**说明**: 四个套件均最优，整体 98.1%，超过擅长空间建模的 SpatialVLA 与擅长视觉推理的 CoT-VLA 分别约 **20% / 14.2%**；相对基线 OpenVLA（76.5%）提升约 21.6%。

### Table 2: Simulation Results on ManiSkill2 / ManiSkill2 仿真（成功率，†=同设置复现）

| Method | PickC. | StackC. | PushC. | Avg. |
|--------|--------|---------|--------|------|
| Octo [RSS'24] | 86% | 76% | - | 81.0% |
| OpenVLA [CoRL'24] † | 67% | 64% | 71% | 67.3% |
| CogACT [arXiv'24] | 95% | 90% | - | 92.5% |
| GeoVLA [arXiv'25] | 90% | 90% | - | 90.0% |
| Dita [ICCV'25] | 79% | 80% | - | 79.5% |
| OpenVLA-OFT [RSS'25] † | 85% | 93% | 88% | 88.7% |
| **ConsisVLA-4D** | **93%** | **95%** | **95%** | **94.3%** |

**说明**: PickCube/StackCube/PushCube 三任务均领先，平均 94.3%，体现精细空间感知。

### Table 3: Efficiency Optimization Results / 效率优化（†=同设置复现）

**Simulation（单臂任务）**

| Method | Latency ↓ | T-put ↑ | FLOPs ↓ | Cost ↓ |
|--------|-----------|---------|---------|--------|
| RT-2-X [PMLR'23] | 0.200 s | 5.0 Hz | - | - |
| ManipLLM [CVPR'24] | - | 2.2 Hz | - | - |
| CogACT [arXiv'24] | - | 9.8 Hz | - | - |
| OpenVLA [CoRL'24] † | 0.254 s | 3.9 Hz | 8.48 T | 11.7 h |
| TraceVLA [ICLR'25] | 0.192 s | 5.2 Hz | - | - |
| SpatialVLA [RSS'25] | 0.199 s | 20.1 Hz | - | - |
| OpenVLA-OFT [RSS'25] † | 0.137 s | 58.4 Hz | 8.45 T | 12.3 h |
| FiS-VLA [NeurIPS'25] | - | 21.9 Hz | - | - |
| **ConsisVLA-4D** | **0.110 s** | **72.7 Hz** | **4.59 T** | **8.6 h** |
| w/o E3D | 0.204 s | 39.2 Hz | 16.83 T | 22.3 h |

**Real World（双臂任务）**

| Method | Latency ↓ | T-put ↑ | FLOPs ↓ | Cost ↓ |
|--------|-----------|---------|---------|--------|
| OpenVLA [CoRL'24] † | 0.552 s | 1.8 Hz | 16.30 T | 12.8 h |
| OpenVLA-OFT [RSS'25] † | 0.334 s | 74.8 Hz | 14.95 T | 13.7 h |
| **ConsisVLA-4D** | **0.231 s** | **108.2 Hz** | **9.68 T** | **10.1 h** |
| w/o E3D | 0.398 s | 62.8 Hz | 25.70 T | 22.0 h |

**说明**: 尽管引入约 2B 参数（主要来自 VGGT）与 4D 推理，ConsisVLA-4D 反而比最高效的 7B 模型 OpenVLA-OFT 更快——单臂相对 OpenVLA / OpenVLA-OFT 提速 **2.31× / 1.25×**、训练成本省 **1.36× / 1.43×**。**w/o E3D** 行（去掉高效 3D 感知阶段）效率大幅恶化（FLOPs 翻数倍、Cost 升至 22 h+），证明 E3D 的 token 压缩是效率关键。

### Table 4: Real-World Task Results / 真实世界长程任务（子阶段成功率，†=复现，每任务 15 trials）

**Galaxea R1 Lite platform**

| Method | Micro. Put | +Place | +Close | Banana Pick | +Peel | +Place | Drawer Pull | +Place | +Push | T-shirt Step1 | +Step2 | +Step3 | Avg. |
|--------|-----------|--------|--------|-------------|-------|--------|-------------|--------|-------|---------------|--------|--------|------|
| OpenVLA † | 6.7 | 5.3 | 4.7 | 8.0 | 0.7 | 0.7 | 5.3 | 3.3 | 2.0 | 4.7 | 4.0 | 4.0 | 28.5% |
| SpatialVLA † | 6.0 | 5.3 | 4.7 | - | - | - | 6.0 | 3.3 | 2.7 | - | - | - | 36.7% |
| OpenVLA-OFT † | 8.0 | 6.7 | 6.7 | 9.3 | 4.0 | 3.3 | 7.3 | 6.7 | 6.0 | 6.7 | 6.0 | 4.7 | 51.8% |
| **ConsisVLA-4D** | **9.3** | **8.7** | **8.0** | **10** | **6.0** | **6.0** | **8.7** | **8.0** | **6.7** | **9.3** | **8.7** | **7.3** | **70.0%** |

**AgileX Cobot Magic platform**

| Method | Micro. Put | +Place | +Close | Banana Pick | +Peel | +Place | Drawer Pull | +Place | +Push | T-shirt Step1 | +Step2 | +Step3 | Avg. |
|--------|-----------|--------|--------|-------------|-------|--------|-------------|--------|-------|---------------|--------|--------|------|
| OpenVLA † | 6.0 | 5.3 | 4.7 | 8.0 | 1.3 | 0.7 | 5.3 | 3.3 | 2.7 | 4.7 | 4.7 | 4.0 | 30.0% |
| OpenVLA-OFT † | 8.0 | 6.7 | 6.7 | 9.3 | 3.3 | 2.7 | 7.3 | 6.7 | 6.0 | 6.7 | 5.3 | 4.7 | 50.3% |
| **ConsisVLA-4D** | **9.3** | **8.7** | **8.0** | **9.3** | **6.0** | **6.0** | **8.7** | **7.3** | **6.0** | **9.3** | **8.7** | **7.3** | **68.3%** |

**说明**: 四个长程双臂任务（Microwave / Banana Peeling / Drawer / T-shirt Folding），数值为分阶段（progressive stage）平均、Avg. 为完整完成率。两平台上 ConsisVLA-4D 均大幅领先，且跨平台波动仅 **±1.7%**（70.0% vs 68.3%），表明强 sim-to-real 与跨本体稳定性。相对 OpenVLA（约 28.5–30%）提升约 41.5%。

### Table 5: Ablation — CV-Aligner & CO-Fuser / 模块消融

| CV-Aligner ES-Sel. | CV-Aligner S-Fus. | CO-Fuser G-Fus. | CO-Fuser IG-Agg. | LIBERO SR ↑ | Real-World SR ↑ |
|:---:|:---:|:---:|:---:|------|------|
| | ✓ | ✓ | ✓ | 93.9 (-4.2) | 71.7 (-6.6) |
| ✓ | | ✓ | ✓ | 95.5 (-2.6) | 73.3 (-5.0) |
| ✓ | ✓ | | ✓ | 95.6 (-2.5) | 70.0 (-8.3) |
| ✓ | ✓ | ✓ | | 91.7 (-6.4) | 68.3 (-10.0) |
| **✓** | **✓** | **✓** | **✓** | **98.1** | **78.3** |

**说明**: 去掉 CV-Aligner 的 ES-Selection/Single-Fusion 使 SR 降 7.0%/10.0%（真实世界）；去掉 CO-Fuser 的 Group-Fusion/IG-Aggregation 降 8.2%/13.3%。证明每个一致性模块都关键，且把 SigLIP/DINOv2 换成其他编码器会退化。

### Table 6: Ablation — CS-Thinker / CS-Thinker 消融

| Dyn. O. | Glob. D. | Attention | LIBERO SR ↑ | Real-World SR ↑ |
|:---:|:---:|:---:|------|------|
| | ✓ | SC-Attn | 93.3 (-4.8) | 66.7 (-11.6) |
| ✓ | | SC-Attn | 95.4 (-2.7) | 73.3 (-5.0) |
| ✓ | ✓ | Causal | 90.9 (-7.2) | 66.7 (-11.6) |
| ✓ | ✓ | Bidirectional | 92.2 (-5.9) | 68.3 (-10.0) |
| **✓** | **✓** | **SC-Attn** | **98.1** | **78.3** |

**说明**: 去掉动态物体（Dyn. O.）或全局深度（Glob. D.）的隐式知识使 SR 降 2.7–4.8%（仿真）/ 5.7–11.6%（真实）；把 SC-Attn 换成纯因果/双向注意力均明显变差，说明 SC-Attn 的定向调制对时空一致性建模不可或缺。

### Table 7: Ablation — Sparsification Ratio / 稀疏化比例消融（†=复现）

| Spf. Ratio | $\mathbf{z}_{\mathcal{I}}^{\text{obj-3D}}$ | $\mathbf{z}_{\mathcal{L}'}^{\text{agg-3D}}$ | $\mathbf{0}^{\text{4D}}$ | LIBERO SR ↑ | Real-World SR ↑ |
|:---:|:---:|:---:|:---:|------|------|
| ≈ 1/4 | 128 | 128 | 30 | 98.0 (-0.1) | 80.0 (+0.7) |
| **≈ 1/8** | **64** | **64** | **18** | **98.1** | **78.3** |
| ≈ 1/16 | 32 | 32 | 12 | 94.9 (-3.2) | 68.3 (-10.0) |
| ≈ 1/8 | FastV † | | | 88.8 (-9.3) | 50.0 (-28.3) |
| ≈ 1/8 | SliME † | | | 85.6 (-12.5) | 46.7 (-31.6) |

**说明**: $R=1/8$ 在性能与效率间最佳（1/4 几乎无增益、1/16 明显掉点）。在同样 8× 压缩下，ConsisVLA-4D 远胜通用稀疏化方法 [[FastV]]（50.0% 真实）与 [[SliME]]（46.7%），证明面向任务的语义/几何稀疏化优于通用 token 裁剪。$\mathbf{0}^{4D}$ 为动态+深度 token 总数。

### Table 8: $\alpha_l$ Setting vs Linear Decay / 权重调度消融

| $d\alpha_l/dl$（衰减方式） | LIBERO SR ↑ | Real-World SR ↑ |
|---|------|------|
| **$-\frac{\psi(1-\delta)}{2}\cdot\frac{\pi}{\mathcal{L}'}\sin(\frac{l\pi}{\mathcal{L}'})$（余弦，本文）** | **98.1** | **78.3** |
| 1.0（线性，恒定斜率） | 94.4 (-3.7) | 73.3 (-5.0) |
| 0.1（线性，恒定斜率） | 95.9 (-2.2) | 75.0 (-3.3) |

**说明**: 余弦衰减（中层斜率最大、两端近 0）优于线性恒定斜率衰减，因为线性会在整个深度上均匀剥离几何先验、造成优化中的突变。

---

## 实验

### 数据集 / 基准

| 基准 | 规模 | 特点 | 用途 |
|------|------|------|------|
| [[LIBERO]] | 4 套件 × 10 任务 × 50 trials | 单臂，Spatial/Object/Goal/Long，强调空间感知 | 训练/测试 |
| [[ManiSkill2]] | PickCube/StackCube/PushCube | 单臂 pick-and-place，精细空间 | 训练/测试 |
| [[RoboTwin]] 2.0 | 7 个 ALOHA 双臂任务 × 100 trials | 双臂，细空间感知 + 鲁棒协调 | 训练/测试 |
| 真实世界 | 4 长程任务（Microwave/Banana/Drawer/T-shirt），60/60/60/45 示范，每任务 15 trials | 双平台（Galaxea R1 Lite、AgileX Cobot Magic） | 训练/测试 |

### 实现细节

- **Backbone**: [[OpenVLA]]（7B），[[LoRA]] 微调（rank 32，$\alpha=64$）
- **单臂训练**: 动作块 $K=8$；80K 步、batch 64、初始 lr $5\times10^{-4}$；每 10K 步评估取最优
- **双臂训练**: 动作块 $K=25$；80K 步、batch 32、lr $5\times10^{-4}$（50K 步后降到 $5\times10^{-5}$）
- **视觉编码器**: SigLIP（语义）+ DINOv2（几何）+ VGGT（3D，冻结先验）；CO-Fuser 在 DINOv2/VGGT 共 24 层做逐块融合
- **4D 监督**: 动态物体用 [[CoTracker]]、全局深度用 [[Depth-Anything]]
- **硬件**: 训练 4× A800 GPU；真实推理单张 RTX 5090

### 关键实验结论

- **仿真**: LIBERO 98.1%（四套件全 SOTA，超 SpatialVLA/CoT-VLA 约 20%/14.2%）、ManiSkill2 94.3%、RoboTwin 2.0 多任务领先。
- **效率**: 相对 OpenVLA / OpenVLA-OFT 推理提速 2.31× / 1.25×、训练成本省 1.36× / 1.43×；真实双臂吞吐 108.2 Hz（+33.4 Hz），可实时部署。去掉 E3D 效率显著恶化。
- **真实世界**: 两平台 70.0% / 68.3%，跨平台波动 ±1.7%，相对 OpenVLA 约 +41.5%，sim-to-real 与跨本体稳定。
- **消融**: CV-Aligner、CO-Fuser、CS-Thinker 三模块及 SC-Attn 缺一不可（Table 5/6）；稀疏比 1/8 最佳且远胜通用裁剪 FastV/SliME（Table 7）；余弦权重优于线性（Table 8）。
- **定性**: Figure 10/11 显示 CV-Aligner 单点聚焦指令物体、CO-Fuser 分布式覆盖空间几何节点，二者互补。

---

## 批判性思考

### 优点
1. **效率与性能同时拿满**: 在引入 VGGT（约 +2B 参数）与 4D 推理的情况下，靠"任务相关 token 稀疏化（1/8）+ 推理期免显式生成中间量"反而比最快的 7B 基线更快，工程吸引力强；w/o E3D 消融直接量化了这一收益来源。
2. **"时空一致性"被拆成可验证的三层一致性**: cross-view 语义、cross-object 几何、cross-scene 时空，分别由 CV-Aligner/CO-Fuser/CS-Thinker 落实，并各有消融与可视化支撑，论证链条清晰。
3. **隐式知识蒸馏思路新颖**: 用 CoTracker/Depth-Anything 作监督，在训练期把"未来动态物体 + 全局深度"压成隐式知识，推理期零额外生成开销，规避了"未来帧预测"路线的算力与一致性问题。

### 局限性
1. **强依赖多个重型预训练编码器**: SigLIP + DINOv2 + VGGT 三套编码器与 OpenVLA-7B 主干堆叠，虽推理被稀疏化抵消，但训练对显存/工程的门槛仍高，对资源受限场景的"高效"主要体现在推理而非训练成本绝对值。
2. **真实评测样本偏少**: 每任务仅 15 trials（消融扩到 30），4 个真实任务集中在桌面长程抓放/折叠，缺乏接触丰富、强遮挡、大幅分布外扰动的系统性泛化实验。
3. **关键超参偏经验**: $K=32$ Top-K、64 个聚合 token、$\psi=0.2/\delta$、24 层等多为经验设定；余弦衰减的优势有消融但缺少更细的敏感度分析；4D 监督质量直接受 CoTracker/Depth-Anything 上限制约。

### 潜在改进方向
1. 把稀疏比例、Top-K、聚合 token 数做成自适应/可学习，按任务复杂度动态调整，而非固定 1/8。
2. 引入更强的泛化压力测试（未见物体、背景/光照剧变、遮挡、跨形态迁移）与更多基线，量化"三层一致性 → 鲁棒性"的因果。
3. 探索更轻量的 3D 先验来源（蒸馏 VGGT 或换更小的几何编码器），进一步压低训练成本，让"高效"覆盖训练侧。

### 可复现性评估
- [x] 代码开源（论文声明 open-sourced，地址见 arXiv 摘要末尾）
- [ ] 预训练模型（未明确说明权重释放）
- [x] 训练细节较完整（附录 G 给出 LoRA、步数、batch、lr、解码器结构等）
- [x] 数据集可获取（LIBERO / ManiSkill2 / RoboTwin 2.0 公开；真实数据基于 AgileX、Galaxea 平台）

---

## 速查卡片

> [!summary] ConsisVLA-4D: Efficient 3D-Perception & 4D-Reasoning for Manipulation
> - **核心**: 从 2D 高效构造 3D 表征（CV-Aligner 跨视图语义 + CO-Fuser 跨物体几何），再用 CS-Thinker/SC-Attn 注入"未来动态物体 + 全局深度"的隐式知识，达成跨视图/跨物体/跨场景三层时空一致性。
> - **方法**: SigLIP+DINOv2+VGGT 编码 → Top-K(1/8) 选物体 token + 64 聚合 token → CoTracker/Depth-Anything 监督的 4D 隐式知识 → SC-Attn 并行解码动作；OpenVLA-7B + LoRA 主干。
> - **结果**: LIBERO 98.1% / ManiSkill2 94.3% / 真实 70.0%(R1)·68.3%(AgileX)；相对 OpenVLA 提升 21.6%(LIBERO)·41.5%(真实)，推理提速 2.3×/2.4×，真实吞吐 108.2 Hz。
> - **代码**: 开源（地址见 arXiv 2605.05126 摘要）

---

*笔记创建时间: 2026-06-29*
