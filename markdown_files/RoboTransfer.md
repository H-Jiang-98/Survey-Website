---
title: "RoboTransfer: Controllable Geometry-Consistent Video Diffusion for Manipulation Policy Transfer"
method_name: "RoboTransfer"
authors: [Liu Liu, Xiaofeng Wang, Guosheng Zhao, Keyu Li, Wenkang Qin, Jiagang Zhu, Jiaxiong Qiu, Zheng Zhu, Guan Huang, Zhizhong Su]
year: 2026
venue: CVPR
tags: [video-diffusion, sim-to-real, data-synthesis, imitation-learning, multi-view-consistency, manipulation]
zotero_collection: _inbox
image_source: online
arxiv_html: https://arxiv.org/html/2505.23171v2
created: 2026-06-29
---

# RoboTransfer: Controllable Geometry-Consistent Video Diffusion for Manipulation Policy Transfer

## 元信息

| 项目 | 内容 |
|------|------|
| 作者 | Liu Liu, Xiaofeng Wang, Guosheng Zhao, Keyu Li, Wenkang Qin, Jiagang Zhu, Jiaxiong Qiu, Zheng Zhu, Guan Huang, Zhizhong Su |
| 机构 | Horizon Robotics（地平线）、GigaAI、CASIA（中科院自动化所） |
| 会议 | CVPR 2026 |
| 类别 | 视频扩散 / 机器人数据合成 / sim-to-real / 模仿学习 |
| 日期 | 2025-05（arXiv v2） |
| 项目主页 | https://horizonrobotics.github.io/robot_lab/robotransfer |
| 链接 | [arXiv](https://arxiv.org/abs/2505.23171) / [PDF](https://arxiv.org/pdf/2505.23171) |

---

## 一句话总结

> 用一个几何（depth+normal）与外观（背景/物体参考图）解耦的多视角视频扩散模型，把真实机器人演示数据"重渲染"成几何一致、可精细编辑的合成视频，再以 50/50 比例混入真实数据训练策略，显著提升视觉策略对未见物体与场景的泛化。

---

## 核心贡献

1. **首个多视角一致的机器人数据合成框架**: 提出 [[RoboTransfer]]，在保证**时序一致**的同时实现**多视角几何一致**（左/中/右三相机），并对**背景与物体外观解耦地精细控制**——这是 [[ROSIE]]、[[RoboEngine]]、[[Cosmos-Transfer]] 等已有方法都不能同时满足的（见 Table 1）。
2. **自动化数据构造管线**: 提出把真实机器人演示**自动分解**为「几何条件（metric depth + surface normal）+ 外观条件（背景参考图 + 物体参考图）+ ground-truth 图像」三元组的管线，依赖 [[Video Depth Anything]]、[[MoGe]]、[[LOTUS]]、[[Grounding-SAM]]、[[CLIP]] 等现成模型，无需人工标注。
3. **合成数据切实提升策略泛化**: 在真实 [[Agilex Cobot Magic]] 双臂平台上，用 RoboTransfer 合成数据增广后，最难的 Diff-All 设定下 spoon pick-and-place 成功率从 13.3% 提升到 46.7%（相对提升 251%），验证「几何一致的视觉多样性」对 [[模仿学习]]策略的价值。

---

## 问题背景

### 要解决的问题
[[模仿学习]]（Imitation Learning, IL）是机器人操作的主流范式，但**大规模真实演示采集极其昂贵**；仿真虽便宜，但 **sim-to-real gap**（物理建模、渲染保真度、场景构成差异）让规模化迁移困难。如何**低成本地合成既逼真又几何一致、且可精细编辑**的机器人训练数据，是核心问题。

### 现有方法的局限
- **逐帧图像增广**（[[ROSIE]]）：基于 text-to-image 做单帧增广，丢失时序与空间一致性。
- **背景 inpainting 类**（[[ReBot]]、[[RoboEngine]]）：只换背景纹理，缺乏对前景/背景的细粒度控制，且无时序/多视角一致性，常出现噪点与抖动。
- **视频到视频翻译**（[[Cosmos-Transfer]]）：用 segmentation+depth 条件保持单视角几何一致，但**固定视角下尚可、动态视角下显著退化**，且**无法保证多视角一致**。
- **纯文本驱动的视频生成**（[[UniSim]]、[[UniPi]]、[[RoboDreamer]]）：仅靠文本难以精确控制复杂交互式操作，且常有时空不一致。

两个关键挑战始终未解决：**(1)** 机器人多依赖多视角观测，而视频生成模型难以产出多视角一致结果；**(2)** 操作任务复杂、交互性强，仅靠文本输入难以精确控制。

### 本文的动机
作者主张**几何与外观解耦**：用**全局一致的 3D 几何（metric depth + normal）**锚定"内容渲染在哪、怎么渲染"，保证多视角与时序一致；用**参考图（背景图 + 物体图）**提供外观风格，且参考图**不必与每帧对齐**，由扩散模型借几何 anchor 合成跨视角/跨时间一致的外观。这样只需"多样的参考图"而非"逐场景 3D 扫描或严格对齐输入"，可规模化。多视角一致则通过**把多视角视频沿宽度拼接、复用预训练视频扩散主干的多视角 in-context 能力**实现，无需改结构。

---

## 方法详解

### 模型架构

RoboTransfer 是一个**条件视频扩散框架**，从预训练的 [[Stable Video Diffusion|SVD]] 微调而来：
- **输入条件**: 几何条件 $y_s$（spatially-aligned，metric depth 视频 + surface normal 视频）+ 外观条件（背景参考图 $C_b$ + 物体参考图 $C_o$，后者为 unstructured 条件 $y_u$）
- **主干**: 预训练 [[Stable Video Diffusion|SVD]]（视频扩散），用 [[EDM]] 训练目标
- **核心机制**: [[Multi-View Consistent Modeling|多视角一致建模]]（宽度拼接）+ 几何条件通道拼接注入 + 外观条件（背景 VAE latent 拼接 / 物体 [[CLIP]] embedding 交叉注意力）
- **输出**: $N$ 路视角同步、几何一致、外观可控的视频

整体框架见 Figure 2，数据构造管线见 Figure 3。

### 核心模块

#### 模块1: Multi-view Consistent Modeling（多视角一致建模）

**设计动机**: 机器人常用多视角相机并行观测，但视频生成模型难以产出多视角一致结果。作者不引入额外 cross-view 模块，而是**把多视角一致性转化为单图内的全局空间一致性**。

**具体实现**:
- 给定 $N$ 路同步视频 $\{V_1,\dots,V_N\}$，**沿宽度维拼接**后用 [[VAE]] 编码器 $\mathcal{E}$ 联合编码（公式5）。
- 这样直接**复用预训练单视角视频扩散主干的空间推理能力**，无需结构改动，且可**直接加载预训练单视角权重**，收敛快、视角一致性高。

#### 模块2: Geometry Conditions Injection（几何条件注入）

**设计动机**: 现有视频生成靠 temporal/full attention 学时空连贯，但**缺乏对底层 3D 几何的显式理解**，不适合机器人场景。

**具体实现**:
- 用 **depth（相机到表面的空间距离）+ surface normal（局部表面朝向）**两类互补几何线索，组合得到更完整的几何描述。
- depth/normal 序列与 RGB **空间对齐**，用堆叠卷积层的 [[VAE]] 编码器联合下采样编码，得到 geometry-aware 表征。
- 该表征沿**通道维与噪声 latent 拼接**，让扩散生成全程被一致、物理合理的 3D 线索引导。

#### 模块3: Appearance Conditions Injection（外观条件注入）

**设计动机**: 在不破坏几何一致的前提下，对纹理做细粒度控制；分背景与物体两个互补视角。

**具体实现**:
- **背景**: 用 [[VAE]] 编码器把背景参考图 $C_b$ 压成与生成 latent **空间对齐**的 latent，再与 latent 拼接，控制全局纹理/背景风格。
- **物体**: 物体数量与分布可变，故作为 **unstructured 条件**——每个物体用 [[CLIP]] 编码成全局 embedding，经**交叉注意力**注入，灵活可扩展。
- **关键坑**: 外观与几何**朴素组合会冲突**（如 depth 与 texture 不匹配），因此需精心 curate 参考图（如对背景先 inpainting）使其不违背几何先验。

#### 模块4: 数据构造管线（真实演示 → 三元组）

**几何条件构造**: 起点是带两腕相机+一头相机的演示数据。由于部分机器人只有 RGB、RGB-D 原始深度噪声大，用 SOTA 深度估计器（[[Video Depth Anything|VAD]]）产相对深度，再以**原始 sensor depth 作稀疏 metric 锚点、VAD 输出作稠密结构先验**，用鲁棒多帧最小二乘拟合对齐（Algorithm 1/2），得到既有 metric 精度又完整的深度；normal 用 [[LOTUS]] 逐帧估计；无多视角 RGB-D 的数据（如 [[AgiBot-World]]）用 [[MoGe]] 同时估深度与法线。

**外观条件构造**: 采样多视角 RGB 关键帧；用 [[VLM]] 描述器生成物体/场景描述，引导 [[Grounding-SAM]]（Grounding DINO + SAM2）做逐物体 mask；分割出的物体经 inpainting 得到干净背景参考（如空桌面），物体 mask 再过 [[CLIP]] 得语义 embedding。

### 关键公式与机制

#### 公式1: [[EDM]] 去噪训练目标

$$
\mathcal{L}(D_{\theta},\sigma)=\mathbb{E}_{\mathbf{x_0},y_s,y_u}\left[\left\|\mathbf{x_0}-D_{\theta}\!\left(\mathcal{E}(\mathbf{x_0}+\mathbf{n}),\,\tau_u(y_u),\,\tau_s(y_s),\,\sigma\right)\right\|_2^2\right]
$$

**含义**: 在 [[EDM]] 框架下，去噪网络 $D_{\theta}$ 在给定噪声水平 $\sigma$ 与编码后的条件下，从加噪 latent 中重建干净视频 latent $\mathbf{x_0}$。

**符号说明**:
- $\mathbf{x_0}\sim p_{\text{data}}$: 干净样本；$\mathbf{n}\sim\mathcal{N}(\mathbf{0},\sigma^2\mathbf{I})$: 高斯噪声
- $\mathcal{E}$: [[VAE]] 编码器
- $y_s$: 空间对齐条件（如 depth map）；$y_u$: 非结构化条件（如 [[CLIP]] embedding）
- $\tau_s(\cdot),\tau_u(\cdot)$: 对两类条件的编码函数；$\sigma$: 噪声水平

#### 公式2 & 3: 跨噪声水平的加权总目标

$$
\mathcal{L}(D_{\theta})=\mathbb{E}_{\sigma}\!\left[\frac{\lambda(\sigma)}{\exp(u(\sigma))}\,\mathcal{L}(D_{\theta},\sigma)+u(\sigma)\right]
$$

$$
\lambda(\sigma)=\frac{\sigma^2+\sigma_{\text{data}}^2}{(\sigma\cdot\sigma_{\text{data}})^2}
$$

**含义**: 为在不同噪声幅度下稳定学习，对 per-noise 损失做加权期望聚合；权重 $\lambda(\sigma)$ 让训练初期各噪声水平**等比例**贡献。

**符号说明**:
- $\sigma_{\text{data}}$: 数据经验标准差
- $\lambda(\sigma)$: 噪声水平加权函数
- $u(\sigma)$: 可学习的不确定性/加权调制项

#### 公式4: 噪声水平采样分布

$$
\ln(\sigma)\sim\mathcal{N}\!\left(P_{\text{mean}},\,P_{\text{std}}^2\right)
$$

**含义**: 训练时噪声水平 $\sigma$ 的对数服从正态分布，由超参 $P_{\text{mean}},P_{\text{std}}$ 控制其分布（[[EDM]] 标准做法）。

**符号说明**:
- $P_{\text{mean}},P_{\text{std}}$: 控制 $\ln\sigma$ 分布的均值与标准差超参

#### 公式5: 多视角拼接编码

$$
\mathbf{x}_0=\mathcal{E}\!\left([V_1,V_2,\ldots,V_N]\right)
$$

**含义**: 把 $N$ 路同步视角视频**沿宽度维拼接**为单个序列后用 [[VAE]] 编码，从而把"视角间一致性"转化为"单图内全局空间一致性"，复用预训练主干的空间推理能力。

**符号说明**:
- $V_i$: 第 $i$ 路视角视频；$[\,\cdot\,]$: 沿宽度维拼接
- $\mathbf{x}_0$: 拼接后联合编码的 latent

#### Algorithm 1: Dynamic Mask Alignment（深度尺度对齐 + 迭代离群剔除）

把相对深度 $\mathbf{D}_{\text{pred}}$ 对齐到稀疏 metric sensor 深度 $\mathbf{D}_{\text{sensor}}$：先以两者均 $>\epsilon$ 初始化 mask $\mathcal{M}$，迭代 2 次：用最小二乘拟合 $s,b$ → $\mathbf{D}_{\text{metric}}\leftarrow s\cdot\mathbf{D}_{\text{pred}}+b$ → 计算误差 $\mathcal{E}=|\mathbf{D}_{\text{pred}}-\mathbf{D}_{\text{sensor}}|\odot\mathcal{M}$，以 80 百分位为阈值 $\tau$ **剔除高误差区域（sensor 噪声/空洞）**，更新 mask。最终深度兼具 sensor 的 metric 精度与 VAD 的完整性。

#### Algorithm 2: Scale Fitting（最小二乘闭式解）

$$
\min_{s,b}\ \|s\mathbf{p}+b\mathbf{1}-\mathbf{s}\|^2
$$

闭式解：

$$
\begin{bmatrix}s\\ b\end{bmatrix}=\frac{1}{\Delta}\begin{bmatrix}N(\mathbf{p}^\top\mathbf{s})-(\mathbf{1}^\top\mathbf{p})(\mathbf{1}^\top\mathbf{s})\\ (\mathbf{p}^\top\mathbf{p})(\mathbf{1}^\top\mathbf{s})-(\mathbf{p}^\top\mathbf{s})(\mathbf{1}^\top\mathbf{p})\end{bmatrix},\quad \Delta=N(\mathbf{p}^\top\mathbf{p})-(\mathbf{1}^\top\mathbf{p})^2
$$

**含义**: 在有效像素子集（mask 内）上对预测深度 $\mathbf{p}$ 与 sensor 深度 $\mathbf{s}$ 做线性拟合，解出尺度 $s$ 与偏移 $b$。

**符号说明**:
- $\mathbf{p}=\mathbf{D}_{\text{pred}}[\mathcal{M}]$、$\mathbf{s}=\mathbf{D}_{\text{sensor}}[\mathcal{M}]$: mask 内的有效像素向量
- $N$: 有效像素数；$\mathbf{1}$: 全 1 向量；$\Delta$: 法方程行列式

---

## 关键图表

<!-- 仅 Figure 1 在 arXiv HTML 渲染为在线 PNG（已外链）。Figure 2-14 在 arXiv HTML 中为未编译的 LaTeX overpic 占位块（无可用在线 PNG），项目主页只提供 mp4 视频，故以下据论文 figcaption + 正文如实描述其内容与意义。 -->

### Figure 1: Overview of RoboTransfer / 总览

![Figure 1](https://arxiv.org/html/2505.23171v2/x1.png)

**说明**: RoboTransfer 总览。真实数据采集昂贵、仿真数据缺乏视觉保真度，RoboTransfer 在**严格多视角一致**下合成高质量数据；实验证明该合成数据显著提升真机策略性能。这是全文唯一在 arXiv HTML 成功渲染的在线图。

### Figure 2: RoboTransfer Framework / 框架（arXiv HTML 未渲染，依据 caption+正文）

**说明**: RoboTransfer 框架做**多视角一致建模**以跨视角联合推理：几何用 metric depth + normal 表示（沿通道与噪声 latent 拼接注入），外观用背景参考图（VAE latent 拼接）与物体参考图（[[CLIP]] embedding 交叉注意力）编码，实现对外观的细粒度控制。是理解整个条件注入机制的核心结构图。

### Figure 3: Data Construction Pipeline / 数据构造管线（arXiv HTML 未渲染）

**说明**: 把真实演示视频自动分解为「几何条件（左：metric depth + normal）+ 外观条件（右：取自关键帧）+ ground-truth」配对。[[VLM]] 描述器生成物体描述，交给 [[Grounding-SAM]] 产逐物体 mask。是贡献2"自动化三元组构造"的图示。

### Figure 4: Different Background Reference Images / 换背景（arXiv HTML 未渲染）

**说明**: 固定前景与场景几何不变，仅灵活编辑背景属性（纹理、颜色）。证明背景外观可**独立可控**。

### Figure 5: Different Object Reference Images / 换物体（arXiv HTML 未渲染）

**说明**: 前景外观可独立于背景被替换，证明物体外观的解耦控制。

### Figure 6: Joint Background+Object Editing / 联合编辑（arXiv HTML 未渲染）

**说明**: 同时编辑背景与前景，展示双重解耦控制能力。Fig 4-6 共同支撑"Real-to-Real"增广能力，从结构化输入生成多样变体以丰富训练数据。

### Figure 7: Sim-to-Real Generation / 仿真到真实（arXiv HTML 未渲染）

**说明**: 在仿真中 per-view 几何条件零成本可得，RoboTransfer 由仿真几何输入生成逼真视频（含分布外样例），降低对真实几何数据的依赖。

### Figure 8: Diverse Scene Synthesis / 多样场景生成（arXiv HTML 未渲染）

**说明**: 借助 [[AgiBot-World]] 等更多训练数据，RoboTransfer 在复杂（移动操作）场景下生成更丰富多样的场景，增强合成数据的多样性与真实感。

### Figure 9: Comparison with Cosmos-Transfer & RoboEngine / 对比（arXiv HTML 未渲染）

**说明**: [[Cosmos-Transfer]] 固定视角尚可、动态视角显著退化；[[RoboEngine]]（基于图像 inpainting）有噪点抖动、缺时序一致、无法精确控背景/物体；RoboTransfer 保持强多视角一致并产出逼真连贯的新视角合成。是定性对比的关键图。

### Figure 10: Synthetic Data Mixing Ratios / 合成数据混合比例（arXiv HTML 未渲染）

**说明**: 在 spoon pick-and-place 的 Diff-All 设定下，成功率与 stage score **均在 50/50 比例处达峰**（SR 13.3%→46.7%，score ≈1.6→3.0）；比例过高反而下降（合成数据缺接触动力学等细粒度物理保真）；纯合成数据仍达 40% 成功率，超过纯真实基线。支撑"50/50 最优"这一关键结论。

### Figure 11: Robot Platform / 机器人平台（附录，arXiv HTML 未渲染）

**说明**: [[Agilex Cobot Magic]] 平台可视化（双 PIPER 臂 + 三路 RealSense D435i：两腕 + 一头顶）。

### Figure 12: Dataset Collected with Agilex / 数据集（附录，arXiv HTML 未渲染）

**说明**: 用 Agilex 机器人采集的数据集——12 个任务、每任务 100 段演示 × 10 种物体配置（共 1000 样本/任务），背景从纹理桌面到杂乱表面，物体形状/尺寸/材质多样。

### Figure 13: Depth Scale Alignment Algorithms / 深度尺度对齐（附录，arXiv HTML 未渲染）

**说明**: 即 Algorithm 1/2 的图示——迭代鲁棒最小二乘：Algorithm 1 通过迭代过滤高误差区（离群）把相对深度对齐到稀疏 metric sensor 深度；Algorithm 2 给出线性对齐参数 $s$（scale）、$b$（shift）的解析闭式解。

### Figure 14: Visual Description Prompt Template / 描述器提示模板（附录，arXiv HTML 未渲染）

**说明**: 物体描述器的结构化 system prompt 模板（要求 100% 检测保证、列出所有可移动物体并按 `A [color] [object], [shape], located [region](x,y)` 模板输出），用于驱动 [[Grounding-SAM]]。

### Table 1: Comparison of Data Generation Methods / 数据生成方法对比

| Method | Model Type | Temporal | Multi-View | BG Ctrl | Object Ctrl | Env Ctrl |
|--------|-----------|----------|------------|---------|-------------|----------|
| ROSIE | Image Diffusion | ✗ | ✗ | ✗ | ✔ | ✔ |
| RoboEngine | Image Diffusion | ✗ | ✗ | ✗ | ✔ | ✗ |
| Cosmos-Transfer1 | Video Diffusion | ✔ | ✗ | ✗ | ✗ | ✔ |
| **RoboTransfer (Ours)** | **Video Diffusion** | **✔** | **✔** | **✔** | **✔** | **✔** |

**说明**: RoboTransfer 是**唯一**同时具备时序一致、多视角一致，且在背景/物体/环境三个层面都可细粒度控制的方法。这是定位全文新颖性的总览表。

### Table 2: Ablation on Geometry Conditions / 几何条件消融

> 缩写: D.S.=raw Depth Sensor；D.P.=model-predicted Depth；Metric D.P.=metric 对齐的预测深度；+N.=再加 Normal。Pix.Mat.（匹配像素，↑越好）；其余 RMSE/Abs.Rel./Sq.Rel./Mean Err./Med.Err./FVD 均↓越好。

| Model | Camera | RMSE↓ | Abs.Rel.↓ | Sq.Rel.↓ | Mean Err.↓ | Med.Err.↓ | Pix.Mat.↑ | FVD↓ |
|-------|--------|-------|-----------|----------|------------|-----------|-----------|------|
| RoboTransfer [D.S.] | left | 0.074 | 0.124 | 0.020 | 4.86 | 2.88 | 142.90 | 218.51 |
| RoboTransfer [D.P.] | left | 0.054 | 0.090 | 0.010 | 3.91 | 2.28 | 149.68 | 123.31 |
| RoboTransfer [Metric D.P.] | left | 0.049 | 0.081 | 0.008 | 3.48 | 1.99 | 183.26 | 112.43 |
| **RoboTransfer [Metric D.P. + N.]** | **left** | **0.047** | **0.079** | **0.008** | **3.31** | **1.92** | **202.03** | **107.43** |
| RoboTransfer [D.S.] | head | 0.182 | 0.074 | 0.031 | 3.51 | 1.58 | – | 153.76 |
| RoboTransfer [D.P.] | head | 0.132 | 0.053 | 0.015 | 3.05 | 1.43 | – | **95.89** |
| RoboTransfer [Metric D.P.] | head | 0.134 | 0.054 | 0.015 | 2.93 | 1.40 | – | 103.32 |
| **RoboTransfer [Metric D.P. + N.]** | **head** | 0.133 | 0.054 | 0.015 | **2.86** | **1.39** | – | 101.17 |
| RoboTransfer [D.S.] | right | 0.090 | 0.137 | 0.025 | 4.93 | 2.91 | 40.70 | 396.33 |
| RoboTransfer [D.P.] | right | 0.072 | 0.103 | 0.016 | 4.14 | 2.36 | 56.02 | 262.96 |
| RoboTransfer [Metric D.P.] | right | 0.064 | 0.090 | 0.011 | 3.74 | 2.11 | 65.45 | 226.76 |
| **RoboTransfer [Metric D.P. + N.]** | **right** | **0.058** | **0.087** | **0.009** | **3.55** | **2.00** | **75.67** | **220.12** |

**说明**: 原始 sensor depth (D.S.) 噪声大、各视角全面最差；改用模型预测深度 (D.P.) 后中视角 RMSE/Mean Err. 相对改善 27.4%/14.2%，中视角与左右视角 Pix.Mat. 相对 +4.7%/+37.6%；再做 metric 对齐 (Metric D.P.) 使左右视角 Pix.Mat. 再 +22.4%/+16.8%（head 静止故 Pix.Mat. 省略）；最终 **depth+normal 组合 (Metric D.P.+N.) 在几乎所有几何指标上最优**，是几何条件设计的核心证据。

### Table 3: Ablation on Appearance Conditions / 外观条件消融

> Bg. Inpaint=背景先 inpainting；Obj. Split=逐物体分别编码 [[CLIP]]。BG.Sim./Obj.Sim.（[[CLIP]] 相似度，↑越好）。

| Bg. Inpaint | Obj. Split | Camera | BG.Sim.↑ | Obj.Sim.↑ | RMSE↓ | Med.Err.↓ | Pix.Mat.↑ | FVD↓ |
|:-----------:|:----------:|--------|----------|-----------|-------|-----------|-----------|------|
|  |  | left | 0.796 | – | 0.051 | 1.94 | 191.59 | 117.25 |
| ✓ |  | left | 0.802 | – | 0.049 | 1.92 | 198.32 | 119.65 |
|  | ✓ | left | 0.797 | – | 0.050 | 1.93 | 196.85 | 118.22 |
| **✓** | **✓** | **left** | **0.805** | – | **0.047** | **1.92** | **202.03** | **107.43** |
|  |  | head | 0.712 | 0.847 | 0.137 | 1.46 | – | 108.65 |
| ✓ |  | head | 0.719 | 0.845 | 0.135 | 1.41 | – | 105.66 |
|  | ✓ | head | 0.712 | 0.855 | 0.136 | 1.42 | – | 105.54 |
| **✓** | **✓** | **head** | **0.720** | **0.858** | **0.133** | **1.39** | – | **101.17** |
|  |  | right | 0.753 | – | 0.064 | 2.12 | 71.71 | 226.84 |
| ✓ |  | right | 0.762 | – | 0.060 | 2.04 | 72.44 | 223.91 |
|  | ✓ | right | 0.754 | – | 0.061 | 2.05 | 72.68 | 222.11 |
| **✓** | **✓** | **right** | **0.764** | – | **0.059** | **2.00** | **75.67** | **220.12** |

**说明**: 不做 inpainting 直接拿背景图作条件会**破坏几何线索**；先 inpainting 既保结构又提升外观一致性（RMSE/中值误差/背景相似度约 +1%）；逐物体分别 [[CLIP]] 编码（Obj. Split）使物体相似度 +1%；**背景 inpainting + 逐物体编码联合（每相机第 4 行）前后景外观控制均最佳**，验证结构化与非结构化视觉成分联合建模的有效性。

### Table 4: Effectiveness of Synthetic Data Augmentation / 合成数据增广效果（真机）

> 两任务（Spoon Pick&Place、Towel Folding）× 两泛化设定（Diff-Obj=换物体；Diff-All=换物体+环境），报告成功率 SR 与 Stage Score。

| Data Composition | Spoon Diff-Obj SR | Spoon Diff-Obj Score | Spoon Diff-All SR | Spoon Diff-All Score | Towel Diff-Obj SR | Towel Diff-Obj Score | Towel Diff-All SR | Towel Diff-All Score |
|------------------|------|------|------|------|------|------|------|------|
| Real only | 33.3% | 2.67 | 13.3% | 1.56 | 16.7% | 1.81 | 12% | 1.08 |
| Domain Random Aug | 44.4% | 2.89 | 11.1% | 1.58 | 16.7% | 1.92 | 12% | 1.12 |
| Real + Obj Aug | 44.4% | 3.00 | 22.2% | 2.04 | 16.7% | 2.00 | 12% | 1.24 |
| **Real + Obj&Bg Aug** | **66.7%** | **3.56** | **46.7%** | **2.98** | **50.0%** | **2.60** | **28%** | **1.92** |

**说明**: **物体+背景联合增广（最后一行）在全部 8 个指标上最优**。最难的 Spoon Diff-All 成功率 13.3%→46.7%（相对 +251%）；Domain Randomization 仅能处理颜色扰动、对结构性变化无效（Diff-All 甚至略降）；逐步叠加（Obj→Obj+Bg）呈**累积增益**（如 Towel Diff-Obj score 1.81→2.0→2.6），说明同时增广物体与背景外观是应对大视觉域偏移的关键。

---

## 实验

### 数据集 / 平台

| 数据集/平台 | 规模 | 特点 | 用途 |
|------|------|------|------|
| RoboTransfer 训练集（自采，Cobot Magic） | 12 任务 ×100 段 ×10 配置；分 10Hz/30 帧约 **24k clips**（评测另 1.6k clips） | 双臂、三相机（2 腕 + 1 头顶）RGB-D | 训练/评测视频生成 |
| [[AgiBot-World]] | 额外引入 | 增强背景多样性（复杂移动操作场景） | 训练补充 |
| 真机策略数据（[[ALOHA]] 遥操） | 每任务 100 段演示；与生成训练集**互斥** | RGB 1280×720→640×360，30Hz 采 10Hz；state 200Hz 降 50Hz | 策略训练/评测 |

### 实现细节

- **视频生成**: 从 [[Stable Video Diffusion|SVD]] 微调；分辨率 $640\times384$；[[AdamW]]，lr $3\times10^{-5}$，global batch 8，训练 70K 步；推理用 [[EDM]] scheduler 30 步去噪 + classifier-free guidance。物体图 resize $224\times224$ 过 [[CLIP]] 取全局 embedding 后拼接。
- **几何标注**: depth 用 [[Video Depth Anything|VAD]]（+sensor 锚点鲁棒最小二乘对齐），normal 用 [[LOTUS]]；无多视角 RGB-D 的数据用 [[MoGe]] 同时估深度与法线。
- **策略模型**: [[ACT]]（Action Chunking Transformer）无改动；输入 3 相机各 1 RGB 帧 + 机器人状态，预测未来 100 个状态（50Hz、2s horizon）。**pretrain-then-finetune**：真实数据预训练 100k 步（batch 512，lr $1\times10^{-4}$），再用合成数据微调 50k 步（lr $1\times10^{-5}$）。
- **硬件**: 8× NVIDIA H20，预训练约 24h、微调约 12h；部署在 [[Agilex Cobot Magic]]，推理延迟约 10ms，动作 50Hz 执行（同步执行完 100 个动作再取新观测）。

### 关键实验结论

- **几何一致（Table 2）**: depth+normal + metric 对齐组合在几乎所有几何/多视角指标上最优；原始 sensor depth 噪声严重拖累生成。
- **外观控制（Table 3）**: 背景先 inpainting + 逐物体 CLIP 编码联合，前后景外观一致性最佳，且不破坏几何。
- **定性（Fig 4-9）**: 背景/物体可独立与联合编辑；Sim-to-Real 与多样场景生成逼真；相对 [[Cosmos-Transfer]]/[[RoboEngine]] 在动态视角与多视角一致性上明显更优。
- **真机策略（Table 4 + Fig 10）**: 物体+背景联合增广全面最优，Spoon Diff-All +251%；合成/真实 **50/50 混合**为最优比例，纯合成也超纯真实基线，体现"合成提供视觉多样性、真实提供物理 grounding"的互补性。

---

## 批判性思考

### 优点
1. **几何-外观解耦的设计干净有效**: 用全局 3D 几何（depth+normal）锚定一致性、用参考图供外观，既保证多视角/时序一致，又支持背景与物体的独立编辑；且参考图无需逐帧对齐，工程上可规模化。
2. **不改结构复用预训练**: 多视角"宽度拼接"把跨视角一致性转化为单图全局一致性，可直接加载 [[Stable Video Diffusion|SVD]] 权重、收敛快——相对引入 cross-view 模块的方法更省。
3. **闭环到真机的证据**: 不止比生成指标，还在真实双臂平台上用 [[ACT]] 验证策略泛化（251% 相对提升、50/50 最优比例），并诚实指出"纯合成数据缺接触动力学"的边界。
4. **数据管线全自动**: 依赖 VAD/MoGe/LOTUS/Grounding-SAM/CLIP 等现成模型自动构造三元组，声明将开源代码，复现门槛相对低。

### 局限性
1. **物理保真度不足**: 作者自承合成视频缺"接触动力学、材质属性"等细粒度物理，导致合成比例过高反而掉点——本质上仍是"视觉重渲染"而非物理一致的世界模型。
2. **真机评测规模偏小**: 仅 2 个任务、每任务真机试验次数有限（SR 以 33.3%/46.7% 等小分母呈现，约 9-25 次量级），Towel Diff-All 最优也仅 28%，统计稳健性与任务覆盖面有限。
3. **依赖几何估计质量**: 整套方法高度依赖 depth/normal 估计与 sensor 对齐；低反射/遮挡区域的几何误差会直接传导到生成，论文未系统分析几何估计失败时的退化。
4. **对比基线较窄**: 生成质量主要与 [[Cosmos-Transfer]]、[[RoboEngine]] 比，且部分为定性对比；缺与更多近期多视角/4D 生成方法（如 [[TesserAct]]）的定量对照。
5. **图表可读性受限于 HTML 渲染**: arXiv HTML 仅渲染了 Figure 1，其余图为未编译占位（本笔记据 caption 描述），定性结论需对照 PDF 原图核验。

### 潜在改进方向
1. 引入物理/接触一致的条件或后处理（如可微仿真、力/触觉先验），缓解高合成比例下的性能回落。
2. 扩大真机任务与试验规模，加入更多基线与可变形/接触丰富任务的定量评测，提升结论普适性。
3. 把几何估计的不确定性显式建模进条件注入（如置信度加权），并分析几何失败模式。
4. 探索可学习的最优真实/合成混合比例（而非固定 50/50），或按任务/阶段自适应。

### 可复现性评估
- [x] 代码开源（论文声明"code will be released"，截至记录尚需确认实际可得）
- [ ] 预训练模型（未明确声明权重发布）
- [x] 训练细节完整（附录给出生成与策略两套完整超参、硬件、步数）
- [x] 数据集可获取（自采集 Cobot Magic 数据 + 公开 [[AgiBot-World]]；自采数据是否开源待确认）

---

## 速查卡片

> [!summary] RoboTransfer: Controllable Geometry-Consistent Video Diffusion for Manipulation Policy Transfer
> - **核心**: 几何（metric depth+normal）与外观（背景/物体参考图）解耦的多视角视频扩散，把真实演示"重渲染"为几何一致、可精细编辑的合成数据。
> - **方法**: 从 SVD 微调；多视角宽度拼接复用预训练主干保一致；几何沿通道拼接注入、背景 VAE-latent 拼接、物体 CLIP-embedding 交叉注意力；自动管线（VAD/MoGe/LOTUS/Grounding-SAM/CLIP）构造三元组。
> - **结果**: 生成几何/外观一致性 SOTA（depth+normal 最优）；真机 ACT 策略用 50/50 合成混合，Spoon Diff-All 成功率 13.3%→46.7%（+251%）。
> - **主页/代码**: https://horizonrobotics.github.io/robot_lab/robotransfer

---

*笔记创建时间: 2026-06-29*
