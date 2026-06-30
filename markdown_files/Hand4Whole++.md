---
title: "Enhancing Hands in 3D Whole-Body Pose Estimation with Conditional Hands Modulator"
method_name: "Hand4Whole++"
authors: [Gyeongsik Moon]
year: 2026
venue: CVPR
tags: [3D-whole-body-pose, hand-pose-estimation, SMPL-X, conditional-modulation, ControlNet, MANO, mesh-recovery]
zotero_collection: _inbox
image_source: online
arxiv_html: https://arxiv.org/abs/2603.14726
created: 2026-06-29
---

# Enhancing Hands in 3D Whole-Body Pose Estimation with Conditional Hands Modulator

> [!warning] 关于图片来源的说明
> 本文 arXiv HTML 页面（`https://arxiv.org/html/2603.14726`）因 arXiv 渲染串档，实际展示的是另一篇论文（ConsisVLA-4D），其 `xN.png` 与本文无关、不可用。正文与数据均依据**官方 PDF**（`https://arxiv.org/pdf/2603.14726`）逐页精读得到；图片采用**项目主页**的在线图床链接（`https://mks0601.github.io/Hand4Whole-plus-plus/images/`，已验证 HTTP 200）。少数仅出现在 PDF 中的定性大图（Fig 6/7、附录 Fig S1/S2）无对应在线图床，已在对应位置标注来源。

## 元信息

| 项目 | 内容 |
|------|------|
| 作者 | Gyeongsik Moon（单一作者） |
| 机构 | Korea University（高丽大学） |
| 会议 | CVPR 2026 |
| 类别 | 3D Whole-Body Pose Estimation / Human Mesh Recovery |
| 日期 | 2026-03（arXiv v1） |
| 项目主页 | https://mks0601.github.io/Hand4Whole-plus-plus |
| 链接 | [arXiv](https://arxiv.org/abs/2603.14726) / [Code](https://github.com/mks0601/Hand4Whole-plus-plus_RELEASE) / [Video](https://youtu.be/iUM7O0r59S0) |

---

## 一句话总结

> 冻结预训练的全身 [[SMPL-X]] 估计器与手部估计器，仅训练一个轻量的条件调制模块 [[CHAM]] 把手部特征注入全身特征流以修正腕部朝向，再通过可微刚性对齐把手指与手形从手模型迁移到全身网格，从而兼得"全身一致"与"手部精细"。

---

## 核心贡献

1. **Hand4Whole++ 模块化框架**: 把一个预训练**全身姿态估计器**（[[SMPLer-X]]-L32）与一个预训练**手部姿态估计器**（[[WiLoR]]）拼装在一起，**无需重训任一主干**，桥接"全身数据集手部多样性不足"与"手部数据集缺乏全身上下文"这一**监督鸿沟**。
2. **CHAM（Conditional Hands Modulator）**: 受 [[ControlNet]] 启发的轻量特征调制模块，把手部特定特征注入**冻结的**全身特征流，让全身模型预测出**既准确、又与上半身运动学链一致**的腕部朝向；仅增加约 10ms（总耗时的约 10%）。
3. **手指关节与手形迁移**: 直接采用手部估计器预测的 [[MANO]] 手指关节与手形，经**可微刚性对齐（differentiable rigid alignment）**贴合到全身网格的腕部，再做 [[Laplacian smoothing]] 消除接缝，从而把手部模型的精细表达力引入全身估计。

---

## 问题背景

### 要解决的问题
在 **3D 全身姿态估计**（whole-body / SMPL-X 网格恢复）中，如何在**全身上下文里**准确恢复手部姿态——既要手指关节精细、手形表达力强，又要腕部朝向与肩-肘-腕运动学链协调一致。

### 现有方法的局限
作者把根因归结为一个**监督鸿沟（supervision gap）**：
- **全身估计器**（[[SMPL-X]] 系，如 [[SMPLer-X]]、OSX、PyMAF-X、HybrIK-X、AiOS）在带全身标注的数据集（AGORA、ARCTIC 等）上训练，但这些数据集**手部多样性有限**，导致手部粗糙；
- **手部估计器**（如 [[WiLoR]]、HaMeR、METRO）在以手为中心的数据集（IH26M、ReIH、HIC 等）上训练，手指/手形很强，但**完全没有全身意识**，在双手交互或需与全身协调时会失败；
- **朴素拼接**（FrankMocap 式直接把手模型输出贴到身体上）：手部估计器独立预测的**腕部朝向不感知全身上下文**，常产生**解剖学上不可能的腕部姿态**（与肩、肘冲突），见 Figure 1 底部。
- 既有的 HMR-Adapter 用全身内部特征插值出手部特征来 refine，但这些特征在手部数据集上"信息量不足"；本文则**从外部手部模型注入信息特征**。

### 本文的动机
- 两类预训练模型各有专长，**与其重训不如组合**：把手部估计器的手部信息**作为条件信号**注入冻结的全身模型，既省训练成本，又保住两个主干在大规模数据上学到的能力与泛化；
- 关键区分：**腕部朝向**需要全身运动学协调（交给 CHAM 调制全身特征流来修正），而**手指关节/手形**是局部细节（直接从手模型迁移），二者**解耦**处理。

---

## 方法详解

### 模型架构

Hand4Whole++ 是一个**全前馈（feed-forward）的模块化框架**（见 Figure 2），由四部分组成：
- **输入**: 单张 RGB 图像 $I$
- **预训练手部估计器**: [[WiLoR]]（[[ViT]] backbone），输入裁剪到 $256\times256$ 的左右手图，输出 [[MANO]] 手指姿态 $\theta_{\text{rh}},\theta_{\text{lh}}$、手形 $\beta_{\text{rh}},\beta_{\text{lh}}$，并提供最后一层 ViT 手部特征（**冻结**）
- **预训练全身估计器**: [[SMPLer-X]]-L32（[[ViT]]-based），输出全套 [[SMPL-X]] 参数（身体/手/脸姿态与形状）（**冻结**）
- **核心可训练模块**: [[CHAM]]——把手部 ViT 特征作为条件，调制全身特征流（**唯一被训练的部分**）
- **手指与手形迁移模块**: 把 MANO 手网格经可微刚性对齐贴到 SMPL-X 腕部
- **输出**: 一个 3D 全身人体网格（body + hands + face）

训练时**只优化 CHAM**，两个姿态估计器全程冻结（图中雪花/火焰标记）。整体运行 10 FPS（单张 RTX A6000）。

### 核心模块

#### 模块1: CHAM（Conditional Hands Modulator）

**设计动机**: 受 [[ControlNet]] 启发，用轻量模块向冻结的预训练全身模型注入手部条件，在**不破坏**其大规模预训练能力的前提下，改善腕部朝向预测。CHAM 作用于 transformer token，**不依赖具体架构**，可与其它 ViT-based 全身模型搭配。

**具体实现**（见 Figure 3）:
- **特征来源**: 从 WiLoR 取左右手最后一层 ViT 特征作为条件输入；**未检测到的手**其 ViT 特征置零。
- **双手交互建模（可选分支）**: 当**两只手都检测到**时，先对手部特征加 **2D 位置编码**（基于原始全身图像空间，用手部裁剪框做 crop-and-resize 以保留手在全身中的空间位置），再过一个**三层交叉注意力 Transformer 编码器**建模左右手相互关系；若只检测到一只手或都没有，则**跳过 2D 位置编码与交叉注意力**，直接用可得的 ViT 特征。
- **左右手分支**: 处理后的手部特征送入两条独立分支——左手分支、右手分支，每条含 **24 个独立的 $1\times1$ 卷积层**，对应 [[SMPL-X]] 的 24 个 Transformer block；左手特征只走左手分支卷积，右手特征只走右手分支卷积，逐 block 产生各自的调制特征。
- **零初始化**: 所有卷积层**零初始化**（follow [[ControlNet]]），保证 CHAM 从中性态出发、**只在有益时**才学习调制，避免破坏预训练。
- **空间对齐与融合**: 对每个特征施加**逆仿射变换**撤销 crop-and-resize，非手部区域**零填充**；左右手分支输出做**逐元素取最大（element-wise maximum）**合成"双手特征"列表（24 个），再在 SMPL-X 各对应 block 处**加性融合（additive fusion）**到 ViT 特征流的空间 token 上，完成全身表征的调制。
- **作用范围**: 把身体特征注入手流只会动到腕部，其它关节不变——这种**受限调制**会导致相对手部定位变差、性能次优；而 CHAM 调制全身特征流可同时改善整条上半身运动学链。

#### 模块2: 手指关节与手形迁移（Finger Articulation & Hand Shape Transfer）

**设计动机**: CHAM 只管腕部朝向，不建模手指关节与手形；这部分**直接迁移**手部估计器的高精度结果（见 Figure 4）。

**具体实现**:
- 采用 WiLoR 预测的 [[MANO]] 参数——**手指姿态** $\theta_{\text{th}},\theta_{\text{lh}}$、**手形** $\beta_{\text{th}},\beta_{\text{lh}}$，**忽略其预测的腕部朝向**（腕部朝向完全交给被 CHAM 调制后的 SMPL-X 决定，以保证与全身一致）。
- 用这些参数在**规范（零）腕部朝向空间**生成手网格与关键点；
- 基于**腕关节 + 四个 MCP 关节（食指、中指、无名指、小指）**做**刚性对齐（rigid alignment）**，把规范手网格对齐到 SMPL-X 全身网格的腕部，再用对齐后的 MANO 手网格**替换**SMPL-X 对应手部顶点；
- **全可微**：对齐过程可微，梯度可回传到 CHAM 用于腕部朝向估计；
- **Laplacian smoothing**: 对齐后对手部边界顶点做拉普拉斯平滑，缓解两模型腕部几何差异造成的接缝伪影。

#### 模块3: 脸部分支

为下颌姿态与表情，复用 [[Hand4Whole]] 的预训练 **FaceNet**（ResNet-18），输入由全身估计器面部关键点裁剪出的人脸图；因 CHAM 训练集面部多样性有限，轻量 FaceNet 比全身估计器的脸分支表现更好。

### 关键公式与机制

> 本文以工程模块与损失为主，公式主要体现在**损失函数**与几个**正则项**上。

#### 公式1: [[CHAM]] 加性调制（机制式）

$$
F'_{\text{wb}}^{(k)} = F_{\text{wb}}^{(k)} + \max\!\big(C_{\text{lh}}^{(k)},\, C_{\text{rh}}^{(k)}\big),\quad k=1,\dots,24
$$

**含义**: 在 SMPL-X 第 $k$ 个 transformer block，把 CHAM 左右手分支输出的调制特征逐元素取最大后，**加性融合**进全身特征流。

**符号说明**:
- $F_{\text{wb}}^{(k)}$: 第 $k$ block 的全身（whole-body）空间 token 特征
- $C_{\text{lh}}^{(k)},C_{\text{rh}}^{(k)}$: 左/右手分支第 $k$ 个 $1\times1$ 卷积（零初始化）经逆仿射对齐、零填充后的调制特征
- $\max(\cdot)$: 逐元素取最大，合成双手特征（论文文字描述，未给编号方程）

#### 公式2: 姿态损失（Pose loss）

$$
\mathcal{L}_{\text{pose}} = \big\|\, \hat{\theta} - \theta^{\text{GT}} \,\big\|_1
$$

**含义**: 对提供 [[SMPL-X]] 姿态标注的全身数据集，监督估计的 3D 关节旋转。对**只有腕部朝向标注的手部数据集**，先把局部腕部姿态经**前向运动学（forward kinematics）**转换为全局腕部朝向再监督，并辅以关键点损失。

**符号说明**:
- $\hat{\theta}$: 估计的 3D 关节旋转；$\theta^{\text{GT}}$: 真值旋转
- $\|\cdot\|_1$: $\ell_1$ 距离

#### 公式3: 形状损失（Shape loss）

$$
\mathcal{L}_{\text{shape}} = \big\|\, \hat{\beta} - \beta^{\text{GT}} \,\big\|_1 \;+\; \lambda_{\beta}\,\big\|\hat{\beta}\big\|_2^2
$$

**含义**: 对带 SMPL-X 形状标注的全身数据集监督形状参数；对**无形状标注的手部数据集**，施加 $\ell_2$ 正则惩罚以避免不真实的形状预测。

**符号说明**:
- $\hat{\beta},\beta^{\text{GT}}$: 估计/真值 [[SMPL-X]] 形状参数
- $\lambda_{\beta}\|\hat\beta\|_2^2$: 防止形状离群的 $\ell_2$ 正则（手部数据集上启用）

#### 公式4: 2D / 3D 关键点损失（Keypoint loss）

$$
\mathcal{L}_{\text{kp}} = \big\|\,\Pi(\hat{J}_{3D}) - J_{2D}^{\text{GT}}\,\big\|_1 \;+\; \big\|\,\hat{J}_{3D} - J_{3D}^{\text{GT}}\,\big\|_1
$$

**含义**: 2D 损失把估计的 3D 关键点投影到图像平面与 GT 2D 比较；3D 损失按数据集采用不同参考系——(1) 全身数据集用 **pelvis-relative**，(2) 交互手数据集用 **right-wrist-relative**，(3) 手部数据集用 **wrist-relative**。

**符号说明**:
- $\hat{J}_{3D}$: 估计 3D 关键点；$\Pi(\cdot)$: 投影算子
- $J_{2D}^{\text{GT}},J_{3D}^{\text{GT}}$: 真值 2D/3D 关键点

#### 公式5: 身体根姿态正则（Body root pose regularization）

$$
\big\langle \hat{d}_{\text{up}},\, d_{\text{up}}^{\text{world}} \big\rangle \;\to\; 1
$$

**含义**: 对**缺乏全身标注的手部样本**，约束估计身体的"竖直向上方向"与世界坐标系上向的点积趋于 1（即保持竖直站立先验），缓解裁剪手图导致的根姿态歧义、稳定上半身预测。

**符号说明**:
- $\hat{d}_{\text{up}}$: 估计人体的 up 方向；$d_{\text{up}}^{\text{world}}$: 世界坐标 up 方向
- $\langle\cdot,\cdot\rangle$: 点积，目标值 1（向量平行）

---

## 关键图表

<!-- 图片使用项目主页在线图床（已验证 200）；Fig 6/7、S1/S2 仅在 PDF 中，无在线图床。 -->

### Figure 1: Comparison with Existing Paradigms / 与现有范式对比

![Figure 1](https://mks0601.github.io/Hand4Whole-plus-plus/images/compare_intro.png)

**说明**: (a) 输入图像；(b) 既有方法——手部估计器([[WiLoR]]/HaMeR)单手准但交互时因缺全身上下文失败，全身估计器([[SMPLer-X]])手部受限于训练多样性而粗糙，**朴素拼接**两者会得到不合理手部（尤其遮挡时）；(c) Hand4Whole++ 在全身上下文中恢复出**既准确又合理**的手。这张图直接点明本文要弥合的"监督鸿沟"。

### Figure 2: Overview of Hand4Whole++ / 整体架构

![Figure 2](https://mks0601.github.io/Hand4Whole-plus-plus/images/architecture.png)

**说明**: 框架含预训练手部估计器（绿，输出 $\theta_{\text{rh}}/\beta_{\text{rh}}$ 等）、预训练全身估计器（红）、[[CHAM]]（火焰=可训练）与手指/手形迁移模块。**仅 CHAM 更新，两个主干冻结（雪花）**。手部特征经 CHAM 调制全身特征，再由 SMPL-X 输出全身姿态，最后做 Finger/shape transfer 得到 3D 全身人。

### Figure 3: Architecture of CHAM / CHAM 内部结构

![Figure 3](https://mks0601.github.io/Hand4Whole-plus-plus/images/cham.png)

**说明**: 灰色虚线框（2D 位置编码 + 交叉注意力）**仅在双手都检出时启用**，建模左右手相互关系；否则手特征直通对应分支。左/右手分支各含一串**零初始化（zero conv）+ 逆仿射变换（inv affine transform）**层（图中示意 3 层，实际 24 层对应 SMPL-X 24 个 block）；两分支经**逐元素取最大**合成双手特征，再**加性融合**进全身特征流。零初始化保证从中性态学起、不破坏预训练。

### Figure 4: Pipeline of Finger & Shape Transfer / 手指与手形迁移流程

![Figure 4](https://mks0601.github.io/Hand4Whole-plus-plus/images/transfer.png)

**说明**: 用 [[MANO]] 在规范空间生成左右手网格 → 基于**腕 + 4 个 MCP 关节**做**刚性对齐(Wrist+MCPs)** → 把对齐手网格替换到被 CHAM 调制后的初始全身网格腕部 → 再加 **Laplacian smoothing** 消除接缝，得到最终"手部增强"的 3D 全身人。展示了"腕部朝向归 CHAM、手指/手形归迁移"的解耦设计。

### Figure 5: Effectiveness of the proposed CHAM / CHAM 有效性消融

![Figure 5](https://mks0601.github.io/Hand4Whole-plus-plus/images/ablation.png)

**说明**: (a) 输入；(b) 微调全身模型——在受限（多为手部）数据上**过拟合**，在未见全身图(EHF)上手部对齐扭曲；(c) 去掉 CHAM 的版本；(d) Hand4Whole++ 完整版——**既保泛化又改善手部精度**。佐证 CHAM 在不动主干的前提下提升手部并隐式修正上半身关节。

### Figure 6: Qualitative Comparison on In-the-Wild Images / 真实场景对比（来源: PDF p.7）

> 在线图床无此图；见 PDF 第 7 页。与 [[SMPLer-X]] 和 Multi-HMR 比较：Hand4Whole++ 产生更好的手与上半身运动学，Multi-HMR 在上半身被截断时不稳定。

### Figure 7: Qualitative Comparison (OSX / SMPLer-X / Multi-HMR) / 复杂双手交互对比（来源: PDF p.7）

> 在线图床无此图；见 PDF 第 7 页。在复杂双手交互图像上，Hand4Whole++ 的手与全身一致性显著优于 OSX、SMPLer-X、Multi-HMR。

### Figure S1: MANO vs SMPL-X Hand Shape Expressiveness / 手形表达力对比（附录，来源: PDF p.9）

> 在线图床无此图；见 PDF 第 9 页。展示 [[MANO]] 手网格比 [[SMPL-X]] 手更贴合 3D 扫描，支撑"为何要从手模型迁移手形"。

### Figure S2: Effectiveness of Body Root Pose Regularizer / 根姿态正则有效性（附录，来源: PDF p.9）

> 在线图床无此图；见 PDF 第 9 页。无根姿态正则时，仅有手标注的样本会得到错误的根姿态；竖直先验约束(公式5)使人体保持直立。

### Table 1: Baselines vs Hand4Whole++ / 全身与手部数据集主对比

| Settings | AGORA (Full/hands) | ARCTIC | EHF | IH26M (MPVPE/MRRPE) | ReIH | HIC |
|----------|--------------------|--------|-----|---------------------|------|-----|
| Original whole-body [SMPLer-X] | 85.61/52.31 | 56.06/31.48 | 63.26/46.21 | 38.64/119.56 | 58.86/101.82 | 32.43/67.68 |
| Fine-tuned whole-body [SMPLer-X] | 90.77/55.91 | 67.52/29.03 | 126.34/57.35 | 20.00/47.89 | 24.87/28.32 | 32.30/62.11 |
| Hand-only [WiLoR] | - /99.11 | - /46.79 | - /46.28 | 11.17/94877.41 | 8.09/3094.65 | **15.44**/848.23 |
| **Hand4Whole++ (Ours)** | **76.84/49.71** | **45.95/25.03** | **61.24/33.43** | **9.40/32.30** | **7.98/16.37** | 17.72/**29.09** |

**说明**: MPVPE/MRRPE 均为毫米，越低越好。Hand4Whole++ 在全身数据集（AGORA/ARCTIC/EHF）的全身与手部误差几乎全面最低；在手部数据集上手部精度逼近甚至超过纯手模型，且 **MRRPE（相对腕位误差）远低于手模型**（如 IH26M 上 32.30 vs WiLoR 的天文数字 94877），说明全身上下文带来正确的双手相对定位。微调全身模型反而在 EHF 上恶化（过拟合），凸显"冻结+CHAM"的价值。

### Table 2: Combining Strategies Ablation / 全身+手模型组合策略

| Wrist copy | CHAM | Full-body errors (root-rel) | Hands errors (wrist-rel) |
|:----------:|:----:|:---------------------------:|:------------------------:|
| ✗ | ✗ | 84.76 | 52.31 |
| ✓ | ✗ | 90.70 | 100.59 |
| ✗ | **✓** | **76.88** | **50.56** |

**说明**: 在 AGORA 上对比。直接**复制手模型腕部朝向（wrist copy）**会大幅增大误差（手部 52→100），因其不感知全身；**CHAM** 同时降低全身误差(84.76→76.88)与手部误差(52.31→50.56)——它不仅改善腕部，还隐式修正肩-肘-腕整条上半身链。（数值与 Tab.1 略异因此处统一关闭 shape transfer 以保公平）

### Table 3: Finger & Hand Shape Transfer Ablation / 手指与手形迁移消融

| Finger | Shape | IH26M | ReIH | HIC |
|:------:|:-----:|:-----:|:----:|:---:|
| ✗ | ✗ | 14.69 | 18.13 | 21.68 |
| ✓ | ✗ | 12.26 | 15.24 | 19.61 |
| **✓** | **✓** | **9.40** | **7.98** | **17.72** |

**说明**: 手部数据集上的手部 MPVPE。第一行用全身模型自带手指（CHAM 调制后）；第二行迁移手指姿态即明显降误差（强手指表达力）；第三行**再迁移手形**进一步降误差——验证 [[MANO]] 比 [[SMPL-X]] 共享 latent 更具手形表达力（呼应 Fig S1）。

### Table 4: Comparison with Full-body Pose Estimators / 与全身估计器对比

| Methods | AGORA (Full/hands) | ARCTIC | EHF | IH26M (MPVPE/MRRPE) | ReIH | HIC |
|---------|--------------------|--------|-----|---------------------|------|-----|
| Hand4Whole | 185.18/74.55 | 151.47/47.79 | 76.84/39.82 | 30.65*/219.26 | 71.57/310.69 | 22.73/101.72 |
| OSX | 178.28/76.37 | 111.42/50.70 | 70.82/53.73 | 38.47*/173.56 | 71.10/221.83 | 35.51/94.57 |
| SMPLer-X | 85.61/52.31 | 56.06/31.48 | 63.26/46.21 | 38.64*/119.56 | 58.86/101.82 | 32.43/67.68 |
| **Hand4Whole++ (Ours)** | **76.84/49.71** | **45.95/25.03** | **61.24/33.43** | **9.40*/32.30** | **7.98/16.37** | **17.72/29.09** |

**说明**: "*"表示按既有协议用手部 GT 尺度评估。Hand4Whole++ 在所有全身与手部数据集上全面领先，尤其手部 MPVPE/MRRPE 大幅降低。

### Table 5: Comparison with Hand Pose Estimators / 与手部估计器对比（手部数据集）

| Methods | IH26M | ReIH | HIC |
|---------|-------|------|-----|
| IntagHand | **9.03***/48.04 | 19.31*/33.37 | 20.08*/52.46 |
| InterWild | 10.28*/44.75 | 13.99/22.38 | 15.68/31.35 |
| HaMeR | 9.53*/594.84 | 17.29/644.51 | 16.15/526.94 |
| WiLoR | 11.17*/94877.41 | 8.09/3094.65 | **15.44**/848.23 |
| **Hand4Whole++ (Ours)** | 9.40*/**32.30** | **7.98/16.37** | 17.72/**29.09** |

**说明**: 即便与专为 3D 交互手设计的 IntagHand/InterWild 比，Hand4Whole++ 的 MPVPE 相当或更优，而 **MRRPE 显著最低**（双手相对定位更准，得益于上半身运动学）；纯手模型(HaMeR/WiLoR)因无全身上下文 MRRPE 极差。且这些手模型**无全身输出**，适用范围受限。

### Table S1: Point-to-point Error (MANO vs SMPL-X hands) / 手形拟合误差（附录）

| Settings | Point-to-point error (mm) |
|----------|---------------------------|
| SMPL-X hands | 1.98 |
| **MANO hands (Ours)** | **1.34** |

**说明**: 把两种模型拟合到 MANO 测试集的 3D 扫描，[[MANO]] 手点对点误差更低，定量支撑"迁移 MANO 手形"的合理性。

### Table S2: With/without Cross-attention in CHAM / CHAM 交叉注意力消融（附录）

| Settings | IH26M | ReIH | HIC |
|----------|-------|------|-----|
| Without cross attention | 9.77/35.36 | 9.12/19.42 | 18.25/30.44 |
| **With cross attention (Ours)** | **9.40/32.30** | **7.98/16.37** | **17.72/29.09** |

**说明**: 双手交互场景下，CHAM 的 2D 位置编码 + 三层交叉注意力同时降低 MPVPE 与 MRRPE，验证建模左右手相互关系的价值。

### Table S3: Running Time per Image (RTX A6000) / 单图各组件耗时（附录）

| Hand detector | Hand pose estimator [WiLoR] | CHAM | Whole-body estimator [SMPLer-X] | Total |
|:-------------:|:---------------------------:|:----:|:-------------------------------:|:-----:|
| 0.01 | 0.05 | **0.01** | 0.03 | **0.1** |

**说明**: 单位秒。整体约 10 FPS；最耗时是手部估计器(WiLoR, 0.05s)，**CHAM 仅 0.01s**，印证其轻量。

---

## 实验

### 数据集

| 数据集 | 类型 | 用途 | 特点 |
|--------|------|------|------|
| [[InterHand2.6M]] (IH26M) | 手部（交互） | 训练+测试 | 丰富 3D 手标注、无全身 |
| ReInterHand (ReIH) | 手部（交互） | 训练+测试 | 交互双手 |
| ARCTIC | 全身 | 训练+测试 | 含全身标注 |
| AGORA | 全身 | 训练+测试 | 多样全身姿态/形状 |
| EHF | 全身 | **仅测试** | 验证对未见域泛化 |
| HIC | 手部 | **仅测试** | 验证对未见域泛化 |

### 实现细节
- **全身估计器**: [[SMPLer-X]]-L32（冻结）；**手部估计器**: [[WiLoR]]（ViT，冻结）；脸: Hand4Whole 的 FaceNet(ResNet-18)
- **唯一可训练**: [[CHAM]]（零初始化的 24 层 $1\times1$ 卷积双分支 + 可选三层交叉注意力）
- **训练**: PyTorch；4 epochs；batch size 32；Adam，初始 lr 1e-4，第 3 epoch 降 10×；所有损失等权(权重 1)
- **硬件/耗时**: 单张 NVIDIA RTX A6000，约 20 小时训练；推理 10 FPS，单图 0.1s（CHAM 仅 +10ms ≈ 总耗时 10%）
- **评估指标**: MPVPE（mean per-vertex position error）、MRRPE（mean relative-root position error），单位 mm；全身按 pelvis 对齐、手按 wrist 对齐评 MPVPE；手部 MRRPE 衡量左右腕相对位置。**不报 PA-MPVPE**（因其会消去全局旋转误差、掩盖腕部朝向，而腕部朝向正是 CHAM 的核心目标）

### 关键实验结论
- **主结果(Tab 1/4/5)**: 在 6 个数据集上全身与手部误差几乎全面 SOTA；手部 MRRPE 相对纯手模型有数量级改善（全身上下文带来正确双手相对定位）。
- **组合策略(Tab 2)**: 直接复制腕部朝向(wrist copy)有害；CHAM 同时降全身与手误差，证明它改善整条上半身运动学链而非仅腕部。
- **迁移消融(Tab 3 + Fig S1/Tab S1)**: 迁移手指→再迁移手形逐步降误差；MANO 手形表达力优于 SMPL-X。
- **CHAM 交叉注意力(Tab S2)**: 建模双手相互关系在交互场景有效。
- **泛化(Fig 5)**: 微调全身模型在未见 EHF 上过拟合扭曲，Hand4Whole++ 保泛化且改善手部。

---

## 批判性思考

### 优点
1. **设计干净、训练便宜**: 冻结两大预训练主干、只训一个轻量 CHAM，单卡 20 小时、推理 10 FPS、CHAM 仅 +10ms，工程落地友好；直面并量化了"监督鸿沟"这一真实痛点。
2. **解耦得当且有据**: "腕部朝向 = 需全身协调 → CHAM 调制"、"手指/手形 = 局部细节 → 可微刚性对齐迁移"的拆分，由 Tab 2（wrist copy vs CHAM）、Tab 3（finger/shape transfer）、Tab S1（MANO vs SMPL-X）三组消融分别坐实。
3. **MRRPE 的提升有说服力**: 不靠 PA-MPVPE 掩盖腕部朝向，反而专门报 MRRPE 并取得数量级改善，证据指向"全身上下文修正双手相对定位"这一核心主张。

### 局限性
1. **依赖两个外部预训练模型**: 性能上限受 [[SMPLer-X]] 与 [[WiLoR]] 制约，且双模型串联增加运行时（作者在 Limitation 中自承）；CHAM 的提升本质是"组合 + 修正"，而非新的表征学习。
2. **非手关节弱监督**: 手部数据集缺全身标注，导致非手关节弱监督、可能与图像错位（即便手预测准），靠竖直先验(公式5)只是缓解。
3. **HIC 上并非全面最优**: Tab 1/4 中 HIC 的 MPVPE(17.72) 不及 WiLoR(15.44)，说明在某些纯手分布上"全身约束"未必占优；交互手专用方法(IntagHand)在 IH26M MPVPE 上也略好。
4. **单作者、单 backbone 验证**: 虽声称 CHAM 与架构无关，但实证只在 SMPLer-X 上验证，跨主干可移植性尚缺实验。

### 潜在改进方向
1. 把 CHAM 接到更多 ViT-based 全身/手模型上，验证"架构无关"主张并探索更强主干组合。
2. 为非手关节引入更强的弱监督/伪标注（如用全身先验或时序约束），缓解手部数据集上的全身错位。
3. 探索端到端联合微调一小部分主干（而非全冻结），在过拟合与表征更新间取折中；或把竖直先验扩展为更一般的场景先验以支持 egocentric 等非直立场景（作者已提及 egocentric 为 future work）。

### 可复现性评估
- [x] 代码开源（https://github.com/mks0601/Hand4Whole-plus-plus_RELEASE）
- [x] 预训练模型（依赖公开的 SMPLer-X、WiLoR；声明用 official released code & weights）
- [x] 训练细节完整（附录 S2 给出 epochs/lr/batch/硬件/耗时）
- [x] 数据集可获取（IH26M/ReIH/ARCTIC/AGORA/EHF/HIC 均公开）

---

## 速查卡片

> [!summary] Hand4Whole++ — Enhancing Hands in 3D Whole-Body Pose with CHAM
> - **核心**: 冻结全身([[SMPLer-X]])与手部([[WiLoR]])两预训练模型，只训轻量 [[CHAM]]，用手部特征调制全身特征流修正腕部朝向，再可微刚性对齐迁移 [[MANO]] 手指/手形。
> - **方法**: CHAM（受 [[ControlNet]] 启发、零初始化 24 层双分支 + 可选交叉注意力，加性融合）改腕部朝向；Finger/Shape transfer（Wrist+4MCP 刚性对齐 + Laplacian 平滑）补手指与手形；腕部朝向归 SMPL-X、不取手模型腕部。
> - **结果**: AGORA/ARCTIC/EHF/IH26M/ReIH 全身+手部误差几乎全面 SOTA，手部 MRRPE 数量级下降；CHAM 仅 +10ms，整体 10 FPS。
> - **代码**: https://github.com/mks0601/Hand4Whole-plus-plus_RELEASE

---

*笔记创建时间: 2026-06-29*
