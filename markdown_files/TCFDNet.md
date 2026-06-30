---
title: "Text-guided Feature Disentanglement for Cross-modal Gait Recognition"
method_name: "TCFDNet"
authors: [Zhiyang Lu, Ming Cheng]
year: 2026
venue: CVPR
tags: [gait-recognition, cross-modal, feature-disentanglement, CLIP, vision-language, LiDAR-camera]
zotero_collection: _inbox
image_source: online
arxiv_html: https://arxiv.org/html/2605.30784v1
created: 2026-06-29
---

# Text-guided Feature Disentanglement for Cross-modal Gait Recognition

## 元信息

| 项目 | 内容 |
|------|------|
| 作者 | Zhiyang Lu, Ming Cheng |
| 机构 | （论文未在正文显式给出，作者 2 人） |
| 会议 | CVPR 2026 |
| 类别 | 跨模态步态识别（LiDAR-Camera Cross-modal Gait Recognition, LCCGR） |
| 日期 | 2026（arXiv v1） |
| 项目主页 | — |
| 链接 | [arXiv](https://arxiv.org/abs/2605.30784) / [HTML](https://arxiv.org/html/2605.30784v1) |

---

## 一句话总结

> 用 LLM 生成的「模态-视角感知」文本字典作为语义锚点，在 [[CLIP]] 共享视觉-语言空间里通过残差分解 + 正交约束把步态特征显式拆成「模态专有」与「模态共享」两部分，再用稳定性增强模块加固共享特征，实现 LiDAR 与相机之间的跨模态步态检索 SOTA。

---

## 核心贡献

1. **文本引导的跨模态解耦框架**: 首次把 [[大语言模型|LLM]] 生成的模态-视角文本先验当作语义锚点，借 [[CLIP]] 投影到视觉空间来「可解释地」指导特征解耦——区别于以往把解耦网络当黑盒的做法。为此设计了 CLIP-based 多粒度编码器与 [[Multi-grained Fusion Module|多粒度融合（MF）模块]] 来同时整合全局与局部信息。
2. **Text-guided Feature Disentanglement (TFD) 模块**: 用 top-$k_t$ 匹配文本重建「模态专有语义」，再通过**残差分解 + 正交约束**导出「模态共享表征」，得到结构化、显式解耦的特征空间。
3. **Feature Stability Enhancement (FSE) 模块**: 针对仅靠残差分解得到的共享特征「脆弱、对扰动敏感」的问题，从**空间依赖**与**通道相关**两个互补视角加固鲁棒性。
4. **SOTA 与跨模态 Patch Exchange 增强**: 在 SUSTech1K 与 FreeGait 上全面刷新跨模态 SOTA；另引入跨模态局部块交换的数据增强进一步提升泛化。

---

## 问题背景

### 要解决的问题

[[步态识别|Gait Recognition]] 在远距离、非接触、难伪装方面优势明显。但真实场景常同时部署 LiDAR 与 RGB 相机等异构传感器，于是出现 **LiDAR-Camera 跨模态步态识别（LCCGR）**：用一种模态的样本去另一种模态的库里检索同一身份。其两大核心难点：

1. **弥合 2D 与 3D 的模态鸿沟**——2D 相机视频与 3D LiDAR 点云序列异质性极大，模态间差异常常超过类内差异，导致特征对齐困难；
2. **在模态共享空间里抽取判别性步态特征**。

### 现有方法的局限

- **CL-Gait**（合成数据对比预训练，ECCV'24）: 依赖 RGB 合成的 2D-3D 数据做对比学习预训练再微调，但合成与真实之间的 domain gap 给模型带来偏差，**额外数据需求大**（表中以 ♠ 标记）。
- **CrossGait**（可学习共享原型，IJCB'24）: 学习模态共享原型 + 注意力自适应加权，但原型**泛化能力不足**；直接把不同模态特征硬拉到一起易导致**类塌缩（class collapse）**、判别性下降。
- **通用特征解耦网络**: 本质上「不可靠、不可解释」，是黑盒。

### 本文的动机

[[CLIP]] 这类视觉-语言模型提供了新视角：**用文本描述模态专有信息并投影到视觉空间，从而可靠地指导特征解耦**。作者据此构建 LLM 生成的「步态模态文本字典」作为语义锚点，让解耦过程有显式、可解释的文本监督，而不是盲目地对齐。核心思路是：先用文本「定义」每种模态/视角各自该长什么样（模态专有），把它从视觉特征里减掉，剩下的残差即为跨模态共享的身份线索。

---

## 方法详解

### 模型架构

TCFDNet 整体流水线（见 Figure 3）：

- **输入**: 相机轮廓图（silhouette）与 LiDAR 深度图（depth），均 resize 到 $64\times64$；样本表示为序列。
- **文本侧**: 用 [[大语言模型|LLM]]（ChatGPT）按 8 个预定义视角、跨模态生成 [[Gait Modality Text Dictionary|步态模态文本字典（GMTD）]]，再经多轮交互扩展为 $l$ 组，提升语义多样性。
- **编码器**: [[Multi-grained Feature Encoder|多粒度特征编码器]]——冻结的 CLIP 视觉/文本编码器 + 轻量 [[ResNet]]-9 旁路适配器 + [[Multi-grained Fusion Module|MF 模块]] + [[Spatial Weighting Module|空间加权（SW）模块]]。
- **核心解耦**: [[Text-guided Feature Disentanglement|TFD 模块]]——top-$k_t$ 文本重建模态专有特征 + 残差分解出模态共享特征。
- **稳定加固**: [[Feature Stability Enhancement|FSE 模块]]——空间依赖（$3\times3$ 卷积）+ 通道相关（bottleneck + 1D 卷积重加权）。
- **输出/损失**: HPP + Separated FC 抽取 part-based 判别特征；三个解耦损失（MA / MO / HSIC）+ 三元组 + 交叉熵联合优化。
- **额外增强**: 跨模态 **Patch Exchange**，在输入层交换异构模态的局部块以提升跨模态感知（细节在补充材料）。

### 核心模块

#### 模块1: Multi-grained Feature Encoder（多粒度特征编码器）

**设计动机**: CLIP 提供了天然对齐的视觉-语言共享空间，但其全局特征对**细粒度步态线索**捕捉不足；需要同时拿到「与文本对齐的全局粗粒度」与「细粒度局部」表征。

**具体实现**:
- **Visual Encoder**: 冻结 CLIP 视觉编码器得到全局特征 $g^m_i$（含 `[CLS]` token 与 $o$ 个剩余 token，公式2），再经只含两层 MLP 的 Adapter 做任务微调（公式3），并用 `Maxpool` 沿时间维聚合（公式4）。同时并行一条可训练的 [[ResNet]] 分支补偿 CLIP 的细粒度短板，得到局部时空表征 $\tilde f^m_i$。
- **Multi-grained Fusion (MF) Module**: 用 [[Multi-head Cross-Attention|多头交叉注意力（MCA）]]（公式5-6）在第 $l$ 层做全局-局部互融合，并采用**双向融合**策略：全局用局部更新、局部用全局更新（公式7-8），最后逐层聚合 + Linear + Reshape 得到融合特征 $u^m_i$（公式9）。见 Figure 4。
- **Spatial Weighting (SW) Module**: 自适应强调身份判别区域、抑制无关区域（公式10-11）。

#### 模块2: Gait Modality Text Dictionary（步态模态文本字典 GMTD）

**设计动机**: 把「模态专有信息」用语言显式化、并嵌入 CLIP 共享空间，作为指导解耦的语义锚点。

**具体实现**:
- 设计 prompt 让 LLM 生成相机/LiDAR 两模态的原始步态描述；按惯例把视角分成 **8 个方向**以缓解视角变化；用 ChatGPT 给描述补充显式视角信息。
- 用**多轮（$l$ 轮）交互**把原始集扩展为 $l$ 组，增大规模并贴合 CLIP 预训练文本格式（见 Figure 1、Figure 2）。GMTD 共 $m\times8\times l$ 条（公式12）。
- 文本特征 $t^m_i$ 经**冻结 CLIP 文本编码器 + 两层 MLP 适配器**得到模态专有文本嵌入 $v^m_j$，构成 GMTF（公式13）。

#### 模块3: Text-guided Feature Disentanglement（TFD）

**设计动机**: 用文本重建「模态专有」分量，从视觉特征里减掉它即得到「模态共享」身份线索；用正交/独立约束保证两者解耦。

**具体实现**:
- 用 CLIP 的 `[CLS]` 特征 ${\tilde g*}^m_i$ 与 GMTF 各文本嵌入算**余弦相似度**（公式14），选 top-$k_t$ 语义原型 $V^m_i$。
- 各原型投影到共享潜空间 + L2 归一化得 $\widehat V^m_i$；$\tilde u^m_i$ 展平归一化得 $\hat u^m_i$；按余弦相似度 + Softmax 得到注意力图 $\Omega$（公式15），加权融合重建出视觉空间里的**模态专有特征** $F_{(mod)}^m_i$（公式16）。
- **门控机制**: 训练早期用 Sigmoid 门控因子 $\alpha$ 调制 $F_{(mod)}$ 的贡献，防止发散（公式17-18）。
- **残差分解**: 模态共享特征 = 加权后的视觉特征 − 模态专有特征（公式19）。流程见 Figure 5。

#### 模块4: Feature Stability Enhancement（FSE）

**设计动机**: 仅由残差分解得到的共享特征仍残留模态专有噪声与局部扰动，鲁棒性差；需从空间与通道两个互补视角加固。

**具体实现**（见 Figure 6）:
- **空间依赖**: $3\times3$ 卷积建模局部空间相关（公式20）。
- **通道相关**: bottleneck 先把展平特征压到低维 $d'\ll d$ 再升回 $d$（公式21）；再用 kernel=3 的 1D 卷积 + Softmax 算每个通道维的权重 $\beta$（公式22），重加权得到基于全局依赖的稳健共享特征（公式23）。
- **part 抽取**: HPP + Separated FC 得到 part-based 判别特征 $F_{(*)}^m_i\in\mathbb{R}^{n_p\times d}$。

### 关键公式与机制

#### 公式1: 输入定义

$$
X=\left\{x^{2d}_{i},x^{3d}_{i},y_{i}\mid i=1,2,\dots,n\right\}
$$

**含义**: 训练集由配对的 2D（相机轮廓）与 3D（LiDAR 深度）序列及其身份标签组成。

**符号说明**:
- $x^m_i\in\mathbb{R}^{s\times h\times w\times c}$: 第 $i$ 个样本，$s$ 帧、$h\times w$ 空间分辨率、$c$ 通道；$m\in\{2d,3d\}$
- $y_i$: 身份标签；$n$: 序列数

#### 公式2-4: 全局视觉特征提取

$$
g^{m}_{i}=\mathrm{CLIP}_{v}\left(x^{m}_{i}\right)
$$

$$
\hat{g}^{m}_{i}=\mathrm{Adapter}\left(\mathrm{LN}\left(g^{m}_{i}\right)\right)
$$

$$
\tilde{g}^{m}_{i}=\mathop{\mathrm{Maxpool}}\limits_{j=1,2,\dots,s}\left(\hat{g}^{m}_{i,j}\right)
$$

**含义**: 冻结 CLIP 视觉编码器抽全局特征 → Adapter（LN + 两层 MLP）微调 → 沿时间维 Maxpool 聚合。

**符号说明**:
- $g^m_i\in\mathbb{R}^{s\times(1+o)\times d}$: 含 `[CLS]` 与 $o$ 个 token；$d$ 为特征维
- $\tilde g^m_i\in\mathbb{R}^{(1+o)\times d}$: 时间聚合后的全局表征

#### 公式5-6: 多头交叉注意力（MF 核心）

$$
\text{MCA}(\tilde{g}^{m}_{i(l-1)},\bar{f}^{m}_{i(l-1)})=\text{Concat}(\text{head}_{1},\dots,\text{head}_{H})W^{O}
$$

$$
\text{head}_{j}=\varphi\!\left(\frac{(\tilde{g}^{m}_{i(l-1)}W^{Q}_{j})(\bar{f}^{m}_{i(l-1)}W^{K}_{j})^{\top}}{\sqrt{d_{h}}}\right)(\bar{f}^{m}_{i(l-1)}W^{V}_{j})
$$

**含义**: 全局特征作 Query、局部特征作 Key/Value，做多头交叉注意力融合。

**符号说明**:
- $\varphi$: Softmax；$W^Q_j,W^K_j,W^V_j\in\mathbb{R}^{d\times d_h}$: 第 $j$ 头投影；$d_h=d/H$；$W^O\in\mathbb{R}^{Hd_h\times d}$: 输出聚合
- $\tilde g^m_{i(l-1)}$ / $\bar f^m_{i(l-1)}$: 第 $(l{-}1)$ 层全局 / 局部特征

#### 公式7-9: 双向融合与多层聚合

$$
\tilde{g}^{m}_{i(l)}=\text{MCA}(\tilde{g}^{m}_{i(l-1)},\bar{f}^{m}_{i(l-1)}),\quad
\bar{f}^{m}_{i(l)}=\text{MCA}(\bar{f}^{m}_{i(l-1)},\tilde{g}^{m}_{i(l-1)})
$$

$$
u^{m}_{i}=\mathrm{Reshape}\!\left(\mathrm{Linear}\!\left(\mathrm{LN}\!\left(\sum_{l=1}^{n_{l}}\bar{f}^{m}_{i(l)}\right)\right)\right)
$$

**含义**: 全局/局部互为条件做双向更新（公式7-8）；逐层局部特征求和后 LN + Linear + Reshape 得最终融合特征。

**符号说明**:
- $n_l$: MF 层数；$u^m_i\in\mathbb{R}^{h'\times w'\times d}$: 融合特征

#### 公式10-11: 空间加权（SW）

$$
w^{m}_{i}=\mathrm{C}_{1\times 1}\!\left(\sigma\!\left(\mathrm{BN}\!\left(\mathrm{C}_{1\times 1}\!\left(u^{m}_{i}\right)\right)\right)\right),\qquad
\tilde{u}^{m}_{i}=w^{m}_{i}\odot u^{m}_{i}
$$

**含义**: 用两层 $1\times1$ 卷积（中间 BN + LeakyReLU）算空间注意力图，广播后逐元素重标定特征。

**符号说明**:
- $\mathrm{C}_{1\times1}$: $1\times1$ 卷积；$\sigma$: LeakyReLU；$w^m_i\in\mathbb{R}^{h'\times w'\times1}$；$\odot$: Hadamard 积

#### 公式12-13: GMTD / GMTF

$$
\text{GMTD}=\left\{t^{m}_{j}\mid m\in\{2d,3d\},\ j=1,2,\dots,8l\right\}
$$

$$
\text{GMTF}=\left\{v^{m}_{j}\mid m\in\{2d,3d\},\ j=1,2,\dots,8l\right\}
$$

**含义**: 文本字典共 $m\times8\times l$ 条（8 视角 × $l$ 组 × 模态数）；GMTF 是其 CLIP 文本嵌入集合。

**符号说明**:
- $t^m_j$: 模态 $m$ 第 $j$ 条文本描述；$v^m_j\in\mathbb{R}^{1\times d}$: 对应文本嵌入

#### 公式14-16: 文本检索与模态专有特征重建（TFD 核心）

$$
\cos({\tilde{g*}}^{m}_{i},v^{m}_{j})=\frac{{\tilde{g*}}^{m}_{i}\cdot v^{m}_{j}}{\|{\tilde{g*}}^{m}_{i}\|_{2}\cdot\|v^{m}_{j}\|_{2}}
$$

$$
\Omega=\varphi\!\left(\cos\!\left(\hat{u}^{m}_{i},\widehat{V}_{i}^{m}\right)\right),\qquad
{F_{(mod)}}^{m}_{i}=\text{Reshape}\!\left(\mathrm{Linear}\!\left(\Omega\widehat{V}_{i}^{m}\right)\right)
$$

**含义**: 先用 `[CLS]` 与文本嵌入的余弦相似度选 top-$k_t$ 原型；再以视觉特征与原型的相似度（Softmax 归一）为注意力图 $\Omega$，加权融合重建模态专有特征。

**符号说明**:
- $\widehat V^m_i\in\mathbb{R}^{k_t\times d}$: top-$k_t$ 原型投影+L2 归一化；$\hat u^m_i$: 归一化后的展平视觉特征
- $\Omega\in\mathbb{R}^{h'w'\times k_t}$: 亲和注意力图；$F_{(mod)}^m_i\in\mathbb{R}^{h'\times w'\times d}$: 重建的模态专有特征

#### 公式17-19: 门控调制与残差分解

$$
\alpha=\eta\!\left(\mathrm{MLP}\!\left(\mathop{\mathrm{Avgpool}}\limits_{h'\times w'}\left(\tilde{u}^{m}_{i}\right)\right)\right),\qquad
{\widetilde{F}_{(mod)}}{}^{m}_{i}=\alpha\odot{F_{(mod)}}^{m}_{i}
$$

$$
{F}_{(shared)}{}^{m}_{i}=\tilde{u}^{m}_{i}-{\widetilde{F}_{(mod)}}{}^{m}_{i}
$$

**含义**: 用 Sigmoid 门控 $\alpha$ 控制模态专有特征在训练早期的贡献以防发散；模态共享特征 = 视觉特征减去（门控后的）模态专有特征。

**符号说明**:
- $\eta$: Sigmoid；$\alpha\in\mathbb{R}^{1\times d}$: 通道级调制因子
- $F_{(shared)}^m_i\in\mathbb{R}^{h'\times w'\times d}$: 模态共享步态表征

#### 公式20-23: 特征稳定增强（FSE）

$$
\hat{F}_{(shared)}{}^{m}_{i}=\mathrm{C}_{3\times 3}\!\left({F}_{(shared)}{}^{m}_{i}\right)
$$

$$
\bar{F}_{(shared)}{}^{m}_{i}=W_{2}\!\left(\sigma\!\left(W_{1}\!\left(\mathrm{Flatten}\!\left(\hat{F}_{(shared)}{}^{m}_{i}\right)\right)\right)\right)
$$

$$
\beta=\eta\!\left(\mathrm{C}_{3}\!\left(\bar{F}_{(shared)}{}^{m}_{i}\right)\right),\qquad
\widetilde{F}_{(shared)}{}^{m}_{i}=\beta\odot{F}_{(shared)}{}^{m}_{i}
$$

**含义**: $3\times3$ 卷积建模空间局部依赖（公式20）；bottleneck（$W_1$ 降维到 $d'$、$W_2$ 升回）压缩通道（公式21）；1D 卷积 + Sigmoid 算通道权重 $\beta$（公式22）并重加权得稳健共享特征（公式23）。

**符号说明**:
- $W_1\in\mathbb{R}^{h''w''d\times d'}$、$W_2\in\mathbb{R}^{d'\times d}$，$d'\ll d$；$\mathrm{C}_3$: kernel=3 的 1D 卷积；$\beta\in\mathbb{R}^{1\times d}$

#### 公式24: Modality Alignment Loss（MA Loss）

$$
\mathcal{L}^{m}_{align}=1-\frac{1}{N}\sum_{i=1}^{N}\left(\cos\!\left({\bar{F}_{(mod)}}{}^{m}_{i},\bar{V}{}_{i}^{m}\right)\right)
$$

**含义**: 让重建的模态专有特征与其对应文本嵌入语义对齐，保证 $F_{(mod)}$ 确实承载文本先验编码的模态语义。

**符号说明**:
- $N$: batch size；$\bar F_{(mod)}^m_i\in\mathbb{R}^{1\times d}$: $\widetilde F_{(mod)}$ 的全局平均池化；$\bar V^m_i\in\mathbb{R}^{1\times d}$: 对应文本嵌入的均值

#### 公式25: Modality Orthogonality Loss（MO Loss）

$$
\mathcal{L}^{m}_{ortho}=\frac{1}{N}\sum_{i=1}^{N}\left(\frac{\left\langle\widetilde{F}_{(shared)}{}^{m}_{i},{\widetilde{F}_{(mod)}}{}^{m}_{i}\right\rangle}{\|\widetilde{F}_{(shared)}{}^{m}_{i}\|\,\|{\widetilde{F}_{(mod)}}{}^{m}_{i}\|}\right)
$$

**含义**: 用归一化内积（余弦）惩罚共享特征与专有特征的相关性，鼓励两者在嵌入空间相互独立（正交）。

**符号说明**:
- $\langle\cdot,\cdot\rangle$: 内积；分母为两特征范数乘积

#### 公式26: HSIC Independence Loss

$$
\mathcal{L}^{m}_{\text{HSIC}}=\mathrm{HSIC}\!\left({\widetilde{F}_{(mod)}}{}^{m}_{i},\widetilde{F}_{(shared)}{}^{m}_{i}\right)=\frac{\mathrm{tr}\!\left(K_{c}L_{c}\right)}{(N-1)^{2}}
$$

**含义**: 从**统计依赖**角度（[[HSIC|Hilbert-Schmidt 独立性准则]]）进一步解相关两个特征分布，与正交约束互补。

**符号说明**:
- $K_c,L_c\in\mathbb{R}^{N\times N}$: $\widetilde F_{(mod)}$ 与 $\widetilde F_{(shared)}$ 的中心化线性核 Gram 矩阵；$\mathrm{tr}$: 迹

#### 公式27: 总损失

$$
\mathcal{L}_{all}=\gamma_{1}\left(\mathcal{L}_{tri}+\mathcal{L}_{ce}\right)+\gamma_{2}\left(\mathcal{L}^{m}_{align}\right)+\gamma_{3}\left(\mathcal{L}^{m}_{ortho}+\mathcal{L}^{m}_{\text{HSIC}}\right)
$$

**含义**: 在「语义对齐、特征解耦、统计去相关」三个互补维度联合约束网络。三元组 + 交叉熵做身份判别基线。

**符号说明**:
- 默认权重 $\gamma_1=1.0$、$\gamma_2=0.5$、$\gamma_3=0.1$

---

## 关键图表

<!-- 图片均使用 arXiv HTML 在线链接，未本地化 -->

### Figure 1: Details of GMTD Construction / GMTD 构建细节

![Figure 1](https://arxiv.org/html/2605.30784v1/x1.png)

**说明**: 展示步态模态文本字典（GMTD）的构建流程：以 LLM 为核心、用 $l$ 轮多轮交互把原始描述扩展为更大规模、覆盖 8 视角与两模态的语义描述集合。这是「文本作为语义锚点」的数据来源，决定了后续解耦的可解释性与多样性。

### Figure 2: Instruction for GMTD / GMTD 的指令设计

![Figure 2](https://arxiv.org/html/2605.30784v1/x2.png)

**说明**: GMTD 的 prompt 指令由 formulation、protocol、examples 三部分组成，借此让 LLM 同时做 instruction-following、chain-of-thought 与 in-context generation。说明文本先验不是随手生成的，而是结构化提示工程的产物。

### Figure 3: Overall Framework / 整体框架

![Figure 3](https://arxiv.org/html/2605.30784v1/x3.png)

**说明**: TCFDNet 全流程图。输入经 Patch Exchange 增强 → CLIP 多粒度编码器抽取文本/视觉特征 → TFD 用文本先验重建并残差分解出模态共享表征 → FSE 加固脆弱的共享特征 → 跨视角/跨模态可靠识别。是理解全文的主图。

### Figure 4: Details of the MF Module / 多粒度融合模块细节

![Figure 4](https://arxiv.org/html/2605.30784v1/x4.png)

**说明**: MF 模块用多头交叉注意力做全局（CLIP）与局部（ResNet）特征的双向互融合（对应公式5-9），解释了如何弥补 CLIP 细粒度不足、把粗细两种粒度统一到与文本对齐的空间。

### Figure 5: Flowchart of the TFD Module / 文本引导解耦模块

![Figure 5](https://arxiv.org/html/2605.30784v1/x5.png)

**说明**: TFD 的核心流程——top-$k_t$ 文本检索 → 注意力加权重建模态专有特征 $F_{(mod)}$ → 门控调制 → 残差分解得到模态共享特征 $F_{(shared)}$（公式14-19）。本文「文本引导解耦」主张的可视化落点。

### Figure 6: Illustration of the FSE Module / 特征稳定增强模块

![Figure 6](https://arxiv.org/html/2605.30784v1/x6.png)

**说明**: FSE 从空间（$3\times3$ 卷积）与通道（bottleneck + 1D 卷积重加权）两路加固共享特征（公式20-23），对应「残差分解出的共享特征脆弱」这一动机。

### Figure 7: t-SNE Visualization / 跨模态特征 t-SNE

![Figure 7](https://arxiv.org/html/2605.30784v1/x7.png)

**说明**: 不同方法的 2D/3D 跨模态特征 t-SNE。TCFDNet 的同类样本更紧凑、不同类分离更大，定性佐证其学到了模态不变且判别性强的表征。

### Figure 8: Intra/Inter-class Cosine Similarity / 类内-类间余弦相似度分布

![Figure 8](https://arxiv.org/html/2605.30784v1/x8.png)

**说明**: 跨模态类内/类间余弦相似度分布对比。TCFDNet 的类内相似度更高、类间更低，量化体现更紧的类内距与更大的类间隔。

### Figure 9: Rank-1 Accuracy vs top-$k_t$ / top-$k_t$ 选择数的影响

![Figure 9](https://arxiv.org/html/2605.30784v1/x9.png)

**说明**: 消融 GMTD 中 top-$k_t$ 文本原型选取数量对 Rank-1 的影响，$k_t=16$ 时最佳——在语义多样性与对齐特异性之间取得平衡。

### Table 1: Camera(2D)→LiDAR(3D) on SUSTech1K / 2D→3D 跨模态检索（Rank-1 %）

♠ 表示该方法需大量额外数据预训练。最佳加粗。

| Methods | Venue | Normal | Bag | Clothing | Carrying | Umbrella | Uniform | Occlusion | Night | Overall R-1 | Overall R-5 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| CAJ | ICCV'21 | 16.4 | - | 7.5 | - | 7.4 | - | - | 2.4 | 11.3 | 30.1 |
| SAAI | ICCV'23 | 22.4 | - | 14.3 | - | 14.0 | - | - | 5.3 | 23.1 | 49.5 |
| LidarGait | CVPR'23 | 18.2 | - | 3.4 | - | 3.4 | - | - | 4.7 | 9.6 | 28.1 |
| CL-Gait ♠ | ECCV'24 | - | - | - | - | - | - | - | - | 55.1 | 77.3 |
| CrossGait | IJCB'24 | 63.2 | - | 30.6 | - | 38.5 | - | - | 11.8 | 53.6 | 77.0 |
| IDKL | CVPR'24 | 60.3 | 49.8 | 29.2 | 48.5 | 36.9 | 50.7 | 64.2 | 9.4 | 52.2 | 75.2 |
| TVI-LFM | NeurIPS'24 | 61.0 | 50.3 | 30.1 | 50.2 | 37.5 | 51.0 | 66.5 | 10.0 | 53.0 | 76.1 |
| TSKD | PR'25 | 52.1 | 43.6 | 27.9 | 48.0 | 32.7 | 41.1 | 55.6 | 6.3 | 42.6 | 65.8 |
| SCR | IF'25 | 61.3 | 52.9 | 29.6 | 53.0 | 39.1 | 53.7 | 69.4 | 10.3 | 54.9 | **78.1** |
| **TCFDNet (Ours)** | - | **67.6** | **60.8** | **36.1** | **55.4** | **39.1** | 52.8 | **71.2** | 11.2 | **55.9** | **78.1** |

**说明**: 2D→3D 设置下 TCFDNet 在 Overall R-1（55.9）领先，且 Normal/Bag/Clothing/Carrying/Occlusion 等多数协变量子集最优；在易混的 Clothing 与 Occlusion 场景仍稳健。Night 条件下所有方法都退化（仅 11.2），作者指出这是「昼夜跨域 + 跨模态」叠加的新难题。

### Table 2: LiDAR(3D)→Camera(2D) on SUSTech1K / 3D→2D 跨模态检索（Rank-1 %）

♠ 表示在合成数据上预训练。最佳加粗。

| Methods | Venue | Normal | Bag | Clothing | Carrying | Umbrella | Uniform | Occlusion | Night | Overall R-1 | Overall R-5 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| CAJ | ICCV'21 | 15.3 | - | 6.4 | - | 13.0 | - | - | 2.3 | 12.3 | 32.3 |
| SAAI | ICCV'23 | 26.5 | - | 21.9 | - | 23.2 | - | - | 3.2 | 26.1 | 54.1 |
| LidarGait | CVPR'23 | 23.2 | - | 14.2 | - | 24.7 | - | - | 2.4 | 18.3 | 39.6 |
| CL-Gait ♠ | ECCV'24 | - | - | - | - | - | - | - | - | 53.3 | 75.6 |
| CrossGait | IJCB'24 | 62.2 | - | 35.4 | - | 57.8 | - | - | 10.3 | 56.4 | 79.8 |
| IDKL | CVPR'24 | 59.6 | 52.3 | 31.0 | 49.5 | 55.2 | 56.1 | 65.3 | 7.9 | 54.8 | 77.1 |
| TVI-LFM | NeurIPS'24 | 60.4 | 53.0 | 32.7 | 51.6 | 56.4 | 55.8 | 69.2 | 9.1 | 55.7 | 78.5 |
| TSKD | PR'25 | 50.1 | 41.3 | 27.7 | 42.8 | 45.9 | 46.2 | 52.5 | 7.8 | 47.2 | 68.1 |
| SCR | IF'25 | 61.6 | 54.1 | 35.8 | 52.0 | 58.1 | 55.9 | 72.6 | 10.2 | 57.7 | 79.5 |
| **TCFDNet (Ours)** | - | **70.9** | **64.8** | **36.7** | **59.1** | **63.3** | **64.3** | **78.7** | **11.1** | **61.7** | **82.5** |

**说明**: 3D→2D 方向优势更明显——Overall R-1 达 61.7（较次优 SCR 57.7 高 4 个点），且**全部协变量子集均为最优**，R-5 也最高（82.5）。说明文本引导解耦在两个检索方向上都有效，反向检索增益更大。

### Table 3: Cross-modal Results on FreeGait / FreeGait 跨模态结果（%）

| Methods | Venue | 2D→3D R-1 | 2D→3D R-5 | 3D→2D R-1 | 3D→2D R-5 |
|---|---|---|---|---|---|
| HMRNet | MM'24 | 23.5 | 55.7 | 25.1 | 57.0 |
| CrossGait | IJCB'24 | 29.6 | 60.8 | 32.3 | 65.9 |
| IDKL | CVPR'24 | 36.7 | 67.4 | 39.5 | 70.3 |
| TVI-LFM | NeurIPS'24 | 38.9 | 69.1 | 41.0 | 71.8 |
| TSKD | PR'25 | 25.1 | 57.9 | 26.7 | 60.8 |
| SCR | IF'25 | 40.1 | 72.0 | 43.3 | 75.9 |
| **TCFDNet (Ours)** | - | **52.1** | **85.3** | **57.9** | **87.2** |

**说明**: 在户外 in-the-wild 的 FreeGait 上，TCFDNet 大幅领先——2D→3D R-1 比次优 SCR 高 12 个点（52.1 vs 40.1），3D→2D R-1 高 14.6 个点（57.9 vs 43.3），验证了在真实复杂场景的强泛化。

### Table 4: Ablation Study on SUSTech1K (LiDAR→Camera) / 模块消融

每步只改动一个功能组，其余保持完整集成（full integration）。

| 配置（Text / Visual Backbone / Decoupling） | R-1 | R-5 | 说明 |
|---|---|---|---|
| w/o GMTD（去掉文本，其余全集成） | 56.2 | 77.3 | 去掉文本先验，R-1 掉 5.5 个点 → 文本锚点对解耦至关重要 |
| GMTD + ViT only（去 ResNet/MF/SW 等局部） | 54.9 | 74.6 | 仅 CLIP-ViT、缺细粒度局部，最差 |
| 去 MF / 仅部分视觉组件（含 SW） | 58.4 | 78.9 | MF 缺失影响融合 |
| 去 MF 与 SW（仅 ResNet 局部） | 56.7 | 76.5 | 局部融合不足 |
| 视觉骨干 full integration、去 TFD（仅 SW 等） | 59.8 | 80.6 | 缺解耦模块 |
| 去 FSE（无稳定增强） | 58.9 | 79.3 | 共享特征脆弱，R-1 降 |
| **Full Model（GMTD+ViT+ResNet+MF+SW+TFD+FSE）** | **61.7** | **82.5** | 各组件叠加达最优 |

**说明**: 原表存在跨列合并，行含义按「每步只改一个功能组」的描述还原。最关键结论：**移除 GMTD 导致显著退化（61.7→56.2）**，凸显模态专有文本先验对跨模态解耦的核心作用；TFD、FSE、MF/SW 各组件移除均带来下降，完整模型最优。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| SUSTech1K | 25,239 序列 / 1,050 人；12 视角、8 行走条件 | RGB 相机 + LiDAR 同步采集（换衣、携物等协变量） | 训练 6,011 序列/250 人；测试 19,228 序列/800 未见人 |
| FreeGait | 11,921 序列 / 1,195 人；3 视角、25m 半径 | 户外无约束、128 线 OUSTR LiDAR + RGB；含 51 人低光场景 | 训练 5,000 序列/500 人；测试 6,921 序列/695 人 |

跨模态评测：一种模态的协变量序列作 probe，另一种模态的正常条件序列作 gallery。

### 实现细节

- **输入**: 相机轮廓图 + LiDAR 深度图，均 $64\times64$；每序列随机采 10 帧；batch = 8 (id) × 8 (序列)。
- **网络**: 可训练 [[ResNet]]-9 作 MFE 适配器；嵌入维 $d=512$；part token 数 $o=16$。
- **优化**: [[Adam]]，初始 lr $3\times10^{-4}$，共训练 25,000 epoch；MultiStepLR 在 10k/20k epoch 各 ×0.1 衰减。
- **损失权重**: $\gamma_1=1.0,\ \gamma_2=0.5,\ \gamma_3=0.1$。
- **框架/硬件**: 基于 [[OpenGait]]；采用 Patch Exchange 增强（细节在补充）；2× NVIDIA RTX 3090。

### 关键实验结论

- **SUSTech1K**: 2D→3D Overall R-1 55.9、3D→2D Overall R-1 61.7，两方向均 SOTA，反向检索增益更大；Clothing/Occlusion 仍稳健，Night 是共性短板。
- **FreeGait（in-the-wild）**: 2D→3D R-1 52.1、3D→2D R-1 57.9，较次优大幅领先 12–15 个点，泛化突出。
- **可视化（Fig 7/8）**: t-SNE 更紧凑、余弦相似度类内高类间低，佐证模态不变 + 判别性。
- **消融（Table 4）**: 去 GMTD 退化最大；TFD/FSE/MF/SW 各有贡献。
- **超参（Fig 9）**: top-$k_t=16$ 最优。

---

## 批判性思考

### 优点

1. **可解释的解耦范式**: 把「模态专有」用 LLM 文本显式定义、经 CLIP 投影后从视觉特征里减掉，相比黑盒解耦网络更可解释；MA/MO/HSIC 三损失从对齐/正交/统计独立三个互补维度约束，设计自洽。
2. **效果显著且全面**: 两数据集、两检索方向均 SOTA，FreeGait 上领先幅度尤其大（>12 点），且无需 CL-Gait 那样的大规模合成预训练。
3. **针对性补强**: 注意到「残差分解出的共享特征脆弱」并用 FSE 从空间/通道两路加固，问题-方法对应清晰；消融证实 GMTD 是性能关键来源。

### 局限性

1. **Night 条件全面失效**: 所有方法（含本文）在夜间 R-1 仅约 11，文本先验未能弥补昼夜跨域，作者自己也承认这是开放难题。
2. **大量关键细节在补充材料**: Patch Exchange、文本特征提取流程、损失分析、LLM 具体 prompt 与 GMTD 规模（$l$ 取值）都放在补充，正文可复现性打折；GMTD 的生成依赖 ChatGPT，文本质量/可复现性难保证。
3. **消融表语义模糊**: Table 4 含跨列合并、行配置标注不清晰，难以精确归因每个组件的独立增益；缺少对损失权重 $\gamma$、$3\times3$/bottleneck 维度 $d'$ 等超参的敏感度分析。
4. **机构/开源信息缺失**: 正文未见作者机构与代码链接，复现门槛偏高（仅声明基于 OpenGait）。

### 潜在改进方向

1. 针对夜间引入低光增强、红外或时序光照建模，把「昼夜跨域」单独作为一类文本先验纳入 GMTD。
2. 用更可控/可复现的文本生成（固定模板或开源 LLM）替代 ChatGPT，并公开 GMTD 与 $l$ 的取值，量化文本多样性对性能的影响。
3. 把「文本引导解耦」推广到更多模态对（如红外-可见光、骨架-点云），验证范式的可迁移性。

### 可复现性评估

- [ ] 代码开源（正文未给出链接，仅声明基于 OpenGait）
- [ ] 预训练模型（未提及 release）
- [x] 训练细节部分完整（优化器/lr/调度/batch 在正文，余在补充材料）
- [x] 数据集可获取（SUSTech1K、FreeGait 为公开基准）

---

## 速查卡片

> [!summary] TCFDNet: Text-guided Cross-modal Feature Disentanglement for Gait Recognition
> - **核心**: LLM 生成模态-视角文本字典（GMTD）作语义锚点 → CLIP 共享空间内用 top-$k_t$ 文本重建「模态专有」特征 → 残差分解得「模态共享」身份线索 → FSE 加固 + MA/MO/HSIC 三损失解耦。
> - **方法**: CLIP 多粒度编码器（冻结 CLIP + ResNet-9 旁路 + MF/SW）→ TFD（公式14-19）→ FSE（公式20-23，空间+通道）→ HPP + Separated FC。
> - **结果**: SUSTech1K 2D→3D R-1 55.9 / 3D→2D R-1 61.7，FreeGait R-1 52.1/57.9，两数据集两方向均 SOTA；去 GMTD 掉 5.5 点，top-$k_t=16$ 最优。
> - **短板**: 夜间全面失效（R-1≈11）；关键细节多在补充；正文未给代码链接。

---

*笔记创建时间: 2026-06-29*
