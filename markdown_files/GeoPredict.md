---
title: "GeoPredict: Leveraging Predictive Kinematics and 3D Gaussian Geometry for Precise VLA Manipulation"
method_name: "GeoPredict"
authors: [Jingjing Qian, Boyao Han, Chen Shi, Lei Xiao, Long Yang, Shaoshuai Shi, Li Jiang]
year: 2026
venue: CVPR
tags: [VLA, 3D-Gaussian-Splatting, predictive-prior, kinematic-trajectory, depth-rendering, flow-matching]
zotero_collection: _inbox
image_source: online
arxiv_html: https://arxiv.org/html/2512.16811v2
created: 2026-06-29
---

# GeoPredict: Leveraging Predictive Kinematics and 3D Gaussian Geometry for Precise VLA Manipulation

## 元信息

| 项目 | 内容 |
|------|------|
| 作者 | Jingjing Qian, Boyao Han, Chen Shi, Lei Xiao, Long Yang, Shaoshuai Shi, Li Jiang |
| 机构 | （论文未在正文显式列出，作者团队） |
| 会议 | CVPR 2026 |
| 类别 | VLA（Vision-Language-Action） |
| 日期 | 2025-12（arXiv v2） |
| 项目主页 | （无） |
| 链接 | [arXiv](https://arxiv.org/abs/2512.16811) / [PDF](https://arxiv.org/pdf/2512.16811) |

---

## 一句话总结

> 在 $\pi_0$ 连续动作策略上挂两个"只在训练时用"的预测分支——预测机器人关键点未来 3D 轨迹、以及用轨迹引导细化的预测式 3D Gaussian 几何（深度渲染监督），把运动学与几何先验注入 Transformer 表征，推理时零额外开销却显著提升精细 3D 操作。

---

## 核心贡献

1. **几何感知的 VLA 框架 GeoPredict**: 在强连续动作 [[VLA]] 策略（[[π0|$\pi_0$]]）上注入"面向未来"的**运动学先验**与**几何先验**，让本来 2D-centric、纯反应式的策略具备对长程 3D 动态的推理能力。
2. **两个互补的预测模块**: (a) **轨迹级运动学预测器**——预测机器人关键点的多步未来 3D 运动；(b) **预测式 3D Gaussian 几何模块**——用 [[Track-guided Refinement|轨迹引导细化]]把几何建模容量集中分配到任务相关的交互区域。
3. **训练时监督、推理时零成本**: 两个预测模块**仅作为训练期监督信号**（通过基于深度的可微渲染），推理时只需若干轻量 query token、**不调用任何 3D 解码**，因此部署效率与基座 VLA 完全一致。在 RoboCasa Human-50、LIBERO 与真实世界任务上一致且大幅超越强基线，尤其在几何密集、空间要求高的场景。

---

## 问题背景

### 要解决的问题
现有 [[VLA]] 模型虽然借助大规模 [[VLM]] 预训练获得了很强的语义泛化，但它们**主要在 2D 图像空间工作**，把当前观测**反应式（reactive）**地映射到动作，缺乏显式的 3D 空间建模。这种"2D 中心 + 短视（myopic）"的形式在需要**精确 3D 推理**与**物理一致的长程控制**的任务上表现不可靠（如判断物体位姿、间隙 clearance、末端执行器运动）。

### 现有方法的局限
作者把现有"引入预测结构"的工作分为两类并指出其不足：
1. **学习潜在动力学 / 时间抽象**（latent dynamics）：提供时间信号，但通常 **view-independent**，不强制多视角或 3D 几何一致性，难以在工作空间坐标下推理物体位姿与末端运动。
2. **预测未来观测**（RGB 帧、深度、点表征等）：同样缺乏严格的 3D 几何一致约束。
3. **计算开销**：把高容量预测模块与大 VLA 主干**紧耦合**，若推理时还要做复杂 3D 预测，会带来不可忽视的算力负担，难以实时部署。

### 本文的动机
作者主张对机器人操作而言有两类预测能力特别有价值：
- **预测式运动学先验**：总结机器人未来几步"大概会怎么动"，而不是只依赖瞬时关节状态；
- **预测式 3D Gaussian 几何**：用一个**显式、可微**、与机器人工作空间对齐、可被深度/多视角监督的表征来推理场景几何。

关键约束是：这些预测信号必须**无缝嵌入 VLA 架构**，且**推理开销要足够小**以支持实时控制。GeoPredict 的核心设计哲学正是"**训练时学几何与运动学先验、推理时只跑原版策略**"。

---

## 方法详解

### 模型架构

GeoPredict 建立在 [[π0|$\pi_0$]] 之上（[[PaliGemma]] VLM + [[SigLIP]] 视觉编码器 + 基于 [[Flow Matching|条件流匹配]]的动作专家），并新增两条预测支路（见 Figure 1）：

- **输入**: 语言指令 $\mathbf{L}$ + 多视角当前图像 $\mathbf{I}_t$ + 本体状态 $\mathbf{Q}_t$ + 经 [[Track Encoder|Track Encoder]] 编码的运动历史。
- **中央主干**: 一个共享的 LLM Transformer，承担两大预测任务 + 动作生成。
- **支路1（运动学）**: 用可学习的 [[Future Track Query|Future Track Query]] 预测多步 3D 关键点轨迹。
- **支路2（几何）**: 用 [[3D Spatial Query|3D Spatial Query]] 经 [[Voxel Decoder|Voxel Decoder]] 预测未来工作空间几何，输出预测式 [[3D Gaussian Splatting|3D 高斯]]；并用预测轨迹做 track-guided 细化。
- **输出**: 由 [[Action Expert|Action Expert]] 生成连续 [[Action Chunking|动作块]] $\mathbf{A}_t=[\mathbf{a}_t,\dots,\mathbf{a}_{t+H-1}]$，$H=50$。

每个动作 $\mathbf{a}_t\in\mathbb{R}^7$ 是 7-DoF 末端命令 $\mathbf{a}_t=\{\Delta\mathbf{x},\Delta\bm\theta,g\}$，其中 $\Delta\mathbf{x},\Delta\bm\theta\in\mathbb{R}^3$ 为平移/旋转增量，$g\in\mathbb{R}$ 为夹爪开合。

### 核心模块

#### 模块1: Trajectory-Level Kinematic Prediction（轨迹级运动学预测）

**设计动机**: 关节运动具有**惯性**，且动作生成本质上就是预测未来轨迹；把过去动力学编码进来能让模型做出更物理一致的运动预测。轨迹级建模同时提供了"过去运动的紧凑摘要"和"未来轨迹的显式预测"，后者还充当后续 3D 几何细化的**空间引导**。

**具体实现**:
- **Track Encoder（历史编码）**: 跟踪 $K$ 个 3D 关键点（关节 + 末端点）。对关键点 $k$，把从时刻 $0$ 到 $t{-}1$ 的 3D 坐标聚成轨迹 $\bm{\mathcal{T}}_k\in\mathbb{R}^{(t-1)\times3}$；一个共享可学习 history query $\mathbf{Q}^{\text{hist}}$ 对嵌入轨迹做交叉注意力，压成**单个 history track token** $\mathbf{Z}_k^{\text{hist}}$（公式1）。这 $K$ 个 token 编码了惯性、关节限位与运动规律，附加到 Transformer 输入。
- **Future Track Query（未来预测）**: 引入 $K$ 个可学习 future track query $\{\mathbf{q}_k^{\text{fut}}\}$，与指令、当前图像、history token 一起被处理，得到 future track embedding $\mathbf{e}_k^{\text{fut}}$；再用共享 MLP + 1D 正弦时间编码 $\mathbf{PE}^{\text{time}}$ 解码出 $H{+}1$ 个时刻（当前 + 未来 $H$ 步）的显式 3D 轨迹点 $\hat{\mathbf{p}}_{k,t+\tau}$（公式2），用 MSE 监督（公式3）。

#### 模块2: Predictive 3D Gaussian Geometry（预测式 3D 高斯几何）

**设计动机**: 通过**预测场景几何如何演化**赋予策略更强的空间推理能力——这对需要预判未来物体构型的操作任务至关重要。模块含四部分：3D 空间 query、体素解码器、轨迹引导细化、深度渲染监督。

**具体实现**:
- **3D Spatial Query**: 把 $H\times W\times D$（米）的工作空间按体素 $v$ 离散化为 $(H/v)\times(W/v)\times(D/v)$ 网格，再沿各轴下采样 $4\times$ 得粗网格 $N_x\times N_y\times N_z$。每个粗体素一个可学习 $C$ 维嵌入构成 $\mathbf{Q}^{\text{init}}$，加上 3D 正弦位置编码 $\mathbf{PE}^{\text{spatial}}$（由 x/y/z 三轴 1D 编码拼接而成）得到 $\mathbf{Q}^{\text{spatial}}$（公式4）；展平为 $N_xN_yN_z$ 个 token 送入 Transformer。
- **Voxel Decoder**: 多层注意力后得空间嵌入 $\mathbf{E}^{\text{spatial}}$；复用时间编码 $\mathbf{PE}^{\text{time}}$ 构造各未来时刻的偏移嵌入 $\mathbf{E}^{\text{spatial}}_{t+\tau}$（公式5），reshape 回网格后过转置卷积/上采样恢复原始体素分辨率得稠密体素特征 $\mathbf{F}^{\text{voxel}}$（公式6）；最后一层 3D 卷积把每个体素映射为 $N_G$ 个 [[3D Gaussian Splatting|3DGS]] 基元 $\mathbf{g}=\{\bm\mu,\alpha,\bm\Sigma\}$（中心、不透明度、协方差，**省略颜色**，只关注几何），并为初始全局表征 $\mathbf{G}^{\text{init}}_{t+\tau}$。
- **Track-based Gaussian Refinement（轨迹引导细化）**: 精细操作需要交互区域（末端、关节、目标物）附近的高保真几何。用模块1预测的关键点位置 $\mathbf{P}_{t+\tau}$ 对每个体素算二值细化掩码 $\mathbf{M}^{\text{refine}}$（公式7，体素内若有预测关键点则置1）；被选中的体素再由共享 MLP 生成 $N_G'$（$N_G'>N_G$）个更精细基元 $\mathbf{G}^{\text{refine}}$，与初始集合并成 $\mathbf{G}^{\text{total}}$（公式8）。这样把建模容量集中到预测的交互路径上，而无需全局高分辨率场景模型。
- **Future Depth Rendering（深度渲染监督）**: 对 $\mathbf{G}^{\text{total}}_{t+\tau}$ 在全部 $H{+}1$ 时刻用可微 alpha 合成渲染深度图。沿像素射线 $\mathbf{r}$ 收集相交高斯按前后排序，计算透射率 $T_i$（公式9）与渲染深度 $\hat{\mathbf{D}}(\mathbf{r})$（公式10）；再对落在预定义工作空间内的射线置空间掩码 $\mathbf{M}^{\text{spatial}}=1$，用 masked L1 深度损失监督（公式11，仅对两路 $224\times224$ 环境相机做监督）。

#### 模块3: Block-wise Causal Attention（块级因果注意力，融合机制）

**设计动机**: 让"感知 → 预测 → 控制"形成层次化信息流，把运动与几何先验注入 Transformer 而不破坏原 $\pi_0$ 的因果结构。

**具体实现**（见 Figure 2）: 把全部 token 划成五个有序块——(1) **2D Token**（文本+图像）、(2) **3D Token**（history track token）、(3) **3D Query**（future track query + spatial query）、(4) **State Token**（本体）、(5) **Action Noise**（流匹配用）。块内注意力**全双向**，块间注意力**严格因果**（只能看本块及之前块）。由此预测模块处于中间阶段，向 Transformer 注入结构化的运动与几何先验来塑造控制。

### 关键公式与机制

#### 公式1: [[Track Encoder|历史轨迹 token]]（History Track Token）

$$
\mathbf{Z}_{k}^{\text{hist}}=\text{CrossAttn}\big(\text{query}=\mathbf{Q}^{\text{hist}},\ \text{key}=\text{MLP}(\bm{\mathcal{T}}_{k}),\ \text{value}=\text{MLP}(\bm{\mathcal{T}}_{k})\big)
$$

**含义**: 用一个共享可学习 query 对第 $k$ 个关键点的历史轨迹做交叉注意力，压成单个 token，编码该关节的运动惯性与规律。

**符号说明**:
- $\bm{\mathcal{T}}_k\in\mathbb{R}^{(t-1)\times3}$: 关键点 $k$ 从时刻 $0$ 到 $t{-}1$ 的 3D 轨迹
- $\mathbf{Q}^{\text{hist}}$: 跨关键点共享的历史 query
- $\mathbf{Z}_k^{\text{hist}}$: 输出的单个 history track token

#### 公式2: [[Future Track Query|未来轨迹解码]]

$$
\hat{\mathbf{p}}_{k,t+\tau}=\text{MLP}\big(\mathbf{e}_{k}^{\text{fut}}+\mathbf{PE}^{\text{time}}[\tau]\big),\qquad\tau=0,\ldots,H
$$

**含义**: 把每个关键点的 future track embedding 加上 1D 正弦时间编码后，用共享 MLP 解码出当前及未来 $H$ 步的显式 3D 位置，得到时空轨迹。

**符号说明**:
- $\mathbf{e}_k^{\text{fut}}$: 关键点 $k$ 的未来轨迹潜在嵌入
- $\mathbf{PE}^{\text{time}}\in\mathbb{R}^{(H+1)\times C}$: 1D 正弦时间编码
- $\hat{\mathbf{p}}_{k,t+\tau}$: 预测的关键点 3D 坐标；$\tau$: 时间偏移（$0$ 为当前，$1\dots H$ 为未来）

#### 公式3: 轨迹预测损失

$$
\mathcal{L}_{\text{track}}=\frac{1}{K(H+1)}\sum_{k=1}^{K}\sum_{\tau=0}^{H}\big\|\hat{\mathbf{p}}_{k,t+\tau}-\mathbf{p}^{\text{gt}}_{k,t+\tau}\big\|_{2}^{2}
$$

**含义**: 对所有 $K$ 个关键点、所有 $H{+}1$ 个时刻的预测位置与真值做 MSE，迫使共享 Transformer 学到预测式、动力学一致的运动表征，同时为 3DGS 细化提供显式未来轨迹。

**符号说明**:
- $K$: 关键点数（仿真 8，真实 7）；$H=50$: 预测视界
- $\mathbf{p}^{\text{gt}}_{k,t+\tau}$: 关键点真值 3D 位置

#### 公式4: 3D 空间 query（含位置编码）

$$
\mathbf{Q}^{\text{spatial}}[i,j,k]=\mathbf{Q}^{\text{init}}[i,j,k]+\mathbf{PE}^{\text{spatial}}[i,j,k]
$$

其中 $\mathbf{PE}^{\text{spatial}}[i,j,k]=\text{Concat}\big(\mathbf{PE}^{\text{x}}[i],\,\mathbf{PE}^{\text{y}}[j],\,\mathbf{PE}^{\text{z}}[k]\big)$。

**含义**: 给每个粗体素的可学习嵌入加上 3 轴拼接的正弦位置编码，引入显式几何结构、降低学习难度。

**符号说明**:
- $\mathbf{Q}^{\text{init}}\in\mathbb{R}^{N_x\times N_y\times N_z\times C}$: 粗体素可学习嵌入
- $[i,j,k]$: 体素在三轴上的索引

#### 公式5: 时间偏移空间嵌入（预测未来几何）

$$
\mathbf{E}^{\text{spatial}}_{t+\tau}=\mathbf{E}^{\text{spatial}}+\mathbf{PE}^{\text{time}}[\tau],\quad\tau=0,\ldots,H
$$

**含义**: 复用同一套时间编码，把"当前工作空间表征"平移到各未来时刻，从而用同一表征预测未来几何演化。

**符号说明**:
- $\mathbf{E}^{\text{spatial}}\in\mathbb{R}^{(N_xN_yN_z)\times C}$: 经多层注意力后的当前空间嵌入

#### 公式6: 稠密体素特征

$$
\mathbf{F}^{\text{voxel}}\in\mathbb{R}^{(H/v)\times(W/v)\times(D/v)\times C^{\prime}}
$$

**含义**: 体素解码器（转置卷积+上采样）把粗网格恢复到原始体素分辨率，得到稠密特征体，供后续映射为高斯基元。

**符号说明**:
- $C'$: 体素特征维度（实现中 $C'=256$）；$v$: 体素边长（0.04m）

#### 公式7: 轨迹引导细化掩码

$$
\mathbf{M}^{\text{refine}}[i,j,k]=\begin{cases}1,&\text{if }\exists\,\mathbf{p}\in\mathbf{P}_{t+\tau}\ \text{s.t.}\ \mathbf{p}\in\bm{\mathcal{V}}[i,j,k]\\ 0,&\text{otherwise}\end{cases}
$$

**含义**: 若体素 $\bm{\mathcal{V}}[i,j,k]$ 内落有任意预测关键点，则标记为需细化，从而把高保真几何集中到交互区域。

**符号说明**:
- $\mathbf{P}_{t+\tau}=\{\hat{\mathbf{p}}_{k,t+\tau}\}_{k=1}^K$: 时刻 $t+\tau$ 的预测关键点集合
- $\bm{\mathcal{V}}[i,j,k]$: 第 $(i,j,k)$ 个体素

#### 公式8: 完整高斯表征

$$
\mathbf{G}^{\text{total}}_{t+\tau}=\mathbf{G}^{\text{init}}_{t+\tau}\cup\mathbf{G}^{\text{refine}}_{t+\tau}
$$

**含义**: 把全局初始高斯与被细化体素新增的精细高斯并集，形成该时刻完整的预测式 3D 场景表征。

**符号说明**:
- $\mathbf{G}^{\text{init}}$: 每体素 $N_G$ 个基元的全局表征
- $\mathbf{G}^{\text{refine}}$: 被掩码选中体素新增的 $N_G'$ 个精细基元（$N_G'>N_G$）

#### 公式9: 累积透射率（3DGS 渲染）

$$
T_{i}=\prod_{j=1}^{i-1}(1-\alpha_{j})
$$

**含义**: 射线穿过排在第 $i$ 个高斯之前所有基元的概率，是 alpha 合成的标准透射率项。

**符号说明**:
- $\alpha_j$: 第 $j$ 个高斯的不透明度；基元已按前后（front-to-back）排序

#### 公式10: 渲染深度

$$
\hat{\mathbf{D}}(\mathbf{r})=\sum_{i\in\mathcal{N}}T_{i}\,\alpha_{i}\,d_{i}
$$

**含义**: 沿射线对所有相交高斯按透射率与不透明度加权其中心深度，得到该像素的可微渲染深度（不渲染颜色）。

**符号说明**:
- $\mathcal{N}$: 与射线 $\mathbf{r}$ 相交的高斯集合
- $d_i$: 第 $i$ 个高斯中心 $\bm\mu_i$ 的深度

#### 公式11: 掩码深度损失

$$
\mathcal{L}_{\text{depth}}=\frac{1}{\sum\mathbf{M}^{\text{spatial}}}\sum_{\tau=0}^{H}\sum_{c=1}^{N_{cam}}\sum_{\mathbf{r}\in\text{pixels}}\mathbf{M}^{\text{spatial}}(\mathbf{r})\,\big|\hat{\mathbf{D}}_{c,t+\tau}(\mathbf{r})-\mathbf{D}^{\text{gt}}_{c,t+\tau}(\mathbf{r})\big|
$$

**含义**: 仅在落入工作空间的射线上做 L1 深度回归（跨所有相机、所有时刻），把监督限制在与操作相关的区域，鼓励预测高斯准确刻画工作空间几何的演化。

**符号说明**:
- $\mathbf{M}^{\text{spatial}}(\mathbf{r})$: 射线 back-project 的 3D 点是否在工作空间内（1/0）
- $N_{cam}$: 相机数；$\mathbf{D}^{\text{gt}}$: 真值深度

#### 公式12: 总训练目标

$$
\mathcal{L}_{\text{total}}=\lambda_{1}\mathcal{L}_{\text{action}}+\lambda_{2}\mathcal{L}_{\text{track}}+\lambda_{3}\mathcal{L}_{\text{depth}}
$$

**含义**: 端到端联合优化动作流匹配损失、轨迹预测损失与深度渲染损失三项。实现中三个权重全取 1.0。

**符号说明**:
- $\mathcal{L}_{\text{action}}$: 来自 $\pi_0$ 的连续条件流匹配动作损失
- $\lambda_1,\lambda_2,\lambda_3$: 平衡动作/运动学/几何监督的权重系数

---

## 关键图表

<!-- 图片均使用 arXiv HTML 在线链接，未本地化 -->

### Figure 1: Overview of GeoPredict / 整体架构

![Figure 1](https://arxiv.org/html/2512.16811v2/x1.png)

**说明**: GeoPredict 总览。给定指令、多视角图像与经 Track Encoder 编码的运动历史，中央 LLM Transformer 同时学两个任务：用 Future Track Query 预测多步 3D 关键点轨迹；用 3D Spatial Query 经 Voxel Decoder 预测未来工作空间几何（预测式 3D Gaussian），并用 track-guided refinement 将几何容量分配到交互区域。最终由 Action Expert 生成动作。**关键点**：两条预测支路仅在训练期作为监督，推理时不调用，从而保持效率。这张图是理解全文"训练时学先验、推理时跑原策略"思想的核心。

### Figure 2: Block-wise Causal Attention Mechanism / 块级因果注意力

![Figure 2](https://arxiv.org/html/2512.16811v2/x2.png)

**说明**: 五个有序 token 块（2D / 3D Token / 3D Query / State / Action Noise）的注意力结构：块内全双向、块间严格因果。它把"感知→预测→控制"组织成层次，预测模块居中，向 Transformer 注入运动与几何先验。图中为简洁省略了 3D Token 与 State Token 到其他块的部分注意力路径。

### Figure 3: Real-world Evaluation Suite / 真实世界评测套件

![Figure 3](https://arxiv.org/html/2512.16811v2/x3.png)

**说明**: 真实世界三类评测设置——空间泛化（plate 放在训练未见位置）、几何泛化（小/中/大立方体与长方体的多稳定朝向、含训练未见尺寸）、视觉鲁棒性（背景加入未见干扰物）。每列是同一任务的不同试次，用于检验空间/几何泛化与抗干扰能力。

### Figure 4: Qualitative Comparisons of Future Depth Rendering / 未来深度渲染定性对比

![Figure 4](https://arxiv.org/html/2512.16811v2/x4.png)

**说明**: 在 $t{+}1, t{+}10, t{+}20$ 三个时刻对比预测深度。初始高斯 $\mathbf{G}^{\text{init}}$ 只捕捉粗略布局，细化后的 $\mathbf{G}^{\text{total}}$（红框处）在机械臂周围呈现明显更锐利的几何细节，直观佐证轨迹引导细化能产出更精确、几何更忠实的未来预测，为动作专家提供更优的条件先验。

### Table 1: RoboCasa Simulation Benchmark Results / RoboCasa 仿真基准（24 子任务）

任务成功率（%），$\ast$ 为作者微调结果，加粗为最佳。

**子任务块 1（PnP 系列 + 操作类）**

| Method | CTC1 | CTC2 | CTS1 | STC1 | CTS2 | STC2 | CTM | MTC | SU(CM) | SV(CM) | PS(CB) | TSS |
|--------|------|------|------|------|------|------|-----|-----|--------|--------|--------|-----|
| BC-Transformer | 6.0 | 2.0 | 2.0 | 8.0 | 2.0 | 6.0 | 2.0 | 2.0 | 0.0 | 22.0 | 48.0 | 54.0 |
| GWM | 4.0 | 18.0 | 20.0 | 22.0 | 2.0 | 18.0 | 14.0 | 20.0 | 16.0 | 36.0 | **76.0** | **72.0** |
| $\pi_0^{\ast}$ (Baseline) | 24.0 | 6.8 | 26.8 | 15.6 | 11.2 | 12.8 | 6.8 | 7.6 | 18.4 | 33.6 | 74.0 | 69.6 |
| **GeoPredict (Ours)** | **32.4** | **8.8** | **27.6** | **31.2** | **20.4** | **28.8** | **14.4** | 18.0 | **28.4** | **49.2** | 67.6 | 70.8 |

**子任务块 2（开关/抽屉/门 + 旋钮等）与平均**

| Method | Avg. SR | OSD | CSD | ODD | CDD | OD | CD | TNSF | TFSF | TNS | TFS | TNM | TFM |
|--------|---------|-----|-----|-----|-----|----|----|------|------|-----|-----|-----|-----|
| BC-Transformer | 28.8 | 46.0 | 56.0 | 28.0 | 28.0 | 42.0 | 80.0 | 38.0 | 50.0 | 32.0 | 4.0 | 62.0 | 70.0 |
| GWM | 39.2 | **58.0** | 54.0 | 28.0 | 50.0 | 56.0 | 80.0 | 52.0 | 44.0 | 46.0 | 22.0 | 64.0 | 70.0 |
| $\pi_0^{\ast}$ (Baseline) | 42.3 | 40.8 | **82.0** | 55.2 | 69.2 | 66.4 | 96.0 | 43.6 | 86.0 | 43.2 | 6.0 | 59.6 | 60.0 |
| **GeoPredict (Ours)** | **52.4** | 40.4 | 78.8 | **87.2** | **70.0** | **77.6** | **96.8** | **72.4** | **94.8** | **60.0** | 13.2 | **84.8** | **82.8** |

**说明**: 在 RoboCasa Human-50 少样本设置（每任务仅 50 条人类示范）下，GeoPredict 平均成功率 **52.4%**，比 $\pi_0$ 基线 42.3% 高 **10.1%**，远超未来预测方法 GWM(39.2%) 与 2D 策略 BC-Transformer(28.8%)。改善集中在需要精确空间/几何推理的子任务（如 ODD 55.2→87.2、TFSF 43.6→72.4、TNS 86.0→94.8），印证预测式运动学+几何先验的价值。

### Table 2: LIBERO Simulation Benchmark Results / LIBERO 仿真基准

任务成功率（%），$\ast$ 为复现结果，$\dagger$ 表示无标准差数据；加粗为最佳，下划线为次优（此处用斜体近似）。

| Method | Spatial | Object | Goal | Long | Average |
|--------|---------|--------|------|------|---------|
| Diffusion Policy | 78.3 ± 1.1 | 92.5 ± 0.7 | 68.3 ± 1.2 | 50.5 ± 1.3 | 72.4 ± 0.7 |
| TraceVLA | 84.6 ± 0.2 | 85.2 ± 0.4 | 75.1 ± 0.3 | 54.1 ± 1.0 | 74.8 ± 0.5 |
| Octo | 78.9 ± 1.0 | 85.7 ± 0.9 | 84.6 ± 0.9 | 51.1 ± 1.3 | 75.1 ± 0.6 |
| OpenVLA | 84.7 ± 0.9 | 88.4 ± 0.8 | 79.2 ± 1.0 | 53.7 ± 1.3 | 76.5 ± 0.6 |
| SpatialVLA | 88.2 ± 0.5 | 89.9 ± 0.7 | 78.6 ± 0.6 | 55.5 ± 1.0 | 78.1 ± 0.7 |
| WorldVLA † | 87.6 | 96.2 | 83.4 | 60.0 | 81.8 |
| 4D-VLA | 88.9 ± 0.5 | 95.2 ± 0.3 | 90.9 ± 0.4 | 79.1 ± 1.2 | 88.6 ± 0.3 |
| DreamVLA † | 97.5 | 94.0 | 89.5 | 89.5 | 92.6 |
| $\pi_0^{\dagger}$ | 96.8 | **98.8** | **95.8** | 85.2 | 94.2 |
| OpenVLA-OFT † | 95.2 | 94.2 | 95.2 | 93.2 | 94.5 |
| UniVLA † | 96.5 | 96.8 | 95.6 | 92.0 | 95.2 |
| $\pi_0^{\ast}$ (Baseline) | 96.6 ± 0.6 | 97.2 ± 0.8 | 94.2 ± 0.7 | 87.6 ± 1.1 | 93.9 ± 0.4 |
| **GeoPredict (Ours)** | **98.0 ± 0.7** | 98.2 ± 0.7 | 95.7 ± 0.2 | **94.0 ± 1.0** | **96.5 ± 0.6** |

**说明**: GeoPredict 平均 **96.5%**，超过此前 SOTA 的 UniVLA(95.2%) 与 $\pi_0$ 基线(93.9%)，在四个套件上均衡领先，最难的 Long 套件达 94.0%（基线 87.6%）。相比 OpenVLA(76.5%) 高出 20.0%，说明几何感知训练的有效性与跨任务鲁棒性。

### Table 3: Ablation Study on RoboCasa / 组件消融

Future Depth 用初始全局高斯 $\mathbf{G}^{init}$ 渲染；Future Depth$^{\ast}$ 用细化后的 $\mathbf{G}^{total}$。数值为全子任务平均成功率（%）。

| History Track | Future Track | Future Depth | Future Depth$^{\ast}$ | Average SR |
|:---:|:---:|:---:|:---:|:---:|
| ✘ | ✘ | ✘ | ✘ | 42.3 |
| ✔ | ✘ | ✘ | ✘ | 44.8 |
| ✔ | ✔ | ✘ | ✘ | 47.2 |
| ✘ | ✘ | ✔ | ✘ | 49.4 |
| ✔ | ✔ | ✔ | ✘ | 50.5 |
| ✔ | ✔ | ✘ | ✔ | **52.4** |

**说明**: 自底向上逐件加：history Track Encoder 42.3→44.8（运动学先验），加 Future Track Query 47.2，加初始高斯深度监督 49.4，联合 track+depth(未细化) 50.5，**开启 track-guided 细化达峰值 52.4**。最后 50.5→52.4 的 +1.9% 直接验证"用运动学预测去细化 3DGS 能给策略更优的几何先验"这一核心假设。

### Table 4: Ablation Study on Depth Rendering / 深度渲染消融

$N_G,N_G'$ 为初始/细化阶段每体素高斯基元数；Color 表示是否做 RGB 重建；Time/Epoch 为每轮训练小时数。

| $N_G$ | $N_G'$ | Color | Time / Epoch (h) | Average SR |
|:---:|:---:|:---:|:---:|:---:|
| 4 | ✘ | ✔ | 12.3 | 49.2 |
| 4 | ✘ | ✘ | 12.0 | 49.4 |
| 8 | ✘ | ✘ | 19.1 | 51.4 |
| 4 | 8 | ✘ | 15.5 | 51.1 |
| 4 | **64** | ✘ | 15.7 | **52.4** |

**说明**: (1) 加颜色重建(49.2)不如纯深度(49.4)，证明只需几何信息。(2) 全局把 $N_G$ 从 4 提到 8(51.4) 会让训练时间从 12.0→19.1 h/epoch，代价大。(3) **轨迹引导细化更高效**：$N_G{=}4,N_G'{=}8$ 仅 15.5h 即 51.1；因为交互区域只占体积一小部分，可几乎无额外开销地把 $N_G'$ 提到 64，得到峰值 52.4% 且仅 15.7h。故最终采用 $N_G=4,N_G'=64$。

### Table 5: Real-World Experiment Results / 真实世界实验

任务成功率（%），三类设置：Spatial、Geometry、Robustness。

| Method | Spatial | Geometry | Robustness |
|--------|---------|----------|------------|
| $\pi_0$ (Baseline) | 60.0 | 50.0 | 35.0 |
| **GeoPredict (Ours)** | **85.0** | **95.0** | **90.0** |

**说明**: 在 DISCOVER 机械臂上，GeoPredict 全面碾压 $\pi_0$ 基线：Geometry 任务（含训练未见物体尺寸）从 50.0% 跃升到 95.0%（+45.0%），表明预测式 3DGS 赋予策略可泛化的 3D 几何理解与自适应抓取能力；Robustness（背景干扰物）35.0→90.0，体现抗干扰；Spatial（未见目标位置）60.0→85.0。

---

## 实验

### 数据集 / 基准

| 基准 | 规模 | 特点 | 用途 |
|------|------|------|------|
| [[RoboCasa]] Human-50 | 24 任务，每任务 50 人类示范；每任务 50 试 ×5 场景 | 厨房长程任务，**仅在未见物体实例/风格上评测**，少样本 | 训练/测试 |
| [[LIBERO]] | 4 套件（Spatial/Object/Goal/Long），每任务 50 示范，每任务 50 试（每套件 500 试） | 评测知识迁移与策略泛化 | 训练/测试 |
| 真实世界（DISCOVER 臂） | 3 类任务，每类 50 专家轨迹，每类 20 试 | Spatial/Geometry/Robustness 三种泛化场景 | 训练/测试 |

### 实现细节

- **基座**: [[π0|$\pi_0$]]（[[PaliGemma]] VLM + [[SigLIP]] 视觉编码器 + 条件流匹配动作专家）
- **预测视界**: $H=50$（所有任务）
- **观测**: 2 个环境相机 + 1 个腕部相机；深度监督仅施加于两个 $224\times224$ 环境相机
- **关键点数**: LIBERO/RoboCasa $K=8$（7 关节 + 1 末端），真实世界 $K=7$（6 关节 + 1 末端）
- **3DGS 工作空间**: $1.6\text{m}\times1.6\text{m}\times1.0\text{m}$，体素 $v=0.04$m
- **维度**: token 维 $C=2048$，体素解码器特征维 $C'=256$
- **高斯基元**: 初始 $N_G=4$/体素，细化体素新增 $N_G'=64$
- **损失权重**: $\lambda_1=\lambda_2=\lambda_3=1.0$
- **优化**: AdamW，LR 2.5e-5，40,000 iters，8× NVIDIA H20，总 batch size 32
- **主指标**: Task Success Rate (%)

### 关键实验结论

- **RoboCasa（Table 1）**: 52.4% 平均，较 $\pi_0$ 基线 +10.1%，少样本厨房长程任务上对几何密集子任务提升尤为明显。
- **LIBERO（Table 2）**: 96.5% 平均，超 UniVLA(95.2%) 创 SOTA，Long 套件 94.0%。
- **组件消融（Table 3）**: 每个模块都正贡献，track-guided 细化带来最后关键的 +1.9%。
- **渲染消融（Table 4）**: 纯深度优于带颜色；轨迹引导细化以更低训练成本超过全局加密。
- **定性（Figure 4）**: 细化后未来深度在机械臂周围明显更锐利。
- **真实世界（Table 5）**: 三类任务全面领先，Geometry +45.0% 最突出，验证 3D 几何泛化。

---

## 批判性思考

### 优点
1. **"训练时学先验、推理时零成本"的范式清晰且实用**: 把昂贵的 3D 预测（轨迹 + 高斯几何 + 体素解码 + 深度渲染）全部限制在训练期，推理只加轻量 query token、不跑任何 3D 解码，部署效率与 $\pi_0$ 一致，工程落地友好。
2. **轨迹引导细化把"运动学预测"与"几何建模"耦合得自洽**: 用预测关键点轨迹定位"哪里需要高保真几何"，避免全局高分辨率的算力浪费；Table 4 用训练时长 + 成功率双轴定量证明了该效率优势。
3. **真实世界几何泛化证据强**: Geometry 任务（未见物体尺寸/朝向）50→95% 的大幅提升，是对"显式 3D 几何先验带来可泛化抓取"主张的有力支撑，而非仅靠仿真。

### 局限性
1. **依赖深度/多视角真值监督**: $\mathcal{L}_{\text{depth}}$ 需要真值深度图（仿真天然有，真实场景需 RGB-D 或多视角重建），跨平台采集成本与质量会直接影响几何先验质量；论文未充分讨论深度噪声/缺失时的鲁棒性。
2. **真实世界基线单一、样本小**: Table 5 仅与 $\pi_0$ 比，每类仅 20 试、3 类任务，缺少与其他 3D/预测式 VLA（如 SpatialVLA、DreamVLA）的真实对比，结论普适性需更多基线与更大样本支撑。
3. **关键超参偏经验**: 工作空间尺寸、体素 $v=0.04$、$4\times$ 下采样、$N_G/N_G'$、关键点选取等多为经验设定；损失权重一律取 1.0 也未做敏感性分析，对不同本体/场景的可迁移性存疑。
4. **"先验如何帮助控制"仍偏间接**: 预测模块通过塑造 Transformer 表征间接影响动作，缺少对"表征里到底学到了什么 3D 结构"的探针式分析（如表征探针、几何一致性度量），机制解释偏定性。

### 潜在改进方向
1. 引入对**带噪/稀疏深度**与**单目深度估计**的鲁棒性研究，降低对高质量深度真值的依赖，扩大真实部署面。
2. 在真实世界补齐更多强基线与更大评测规模，并扩展到接触丰富、可变形物体、双臂等更难场景。
3. 把工作空间分辨率、关键点选择、$N_G'$ 与损失权重做成可学习/自动搜索，减少手工调参。
4. 用表征探针、CKA、几何一致性指标量化"预测先验 → 策略提升"的因果链，超越注意力/深度可视化的定性证据。

### 可复现性评估
- [ ] 代码开源（正文未给出代码/项目链接）
- [ ] 预训练模型（未提及）
- [x] 训练细节完整（关键点数、工作空间、$N_G/N_G'$、优化器、硬件、迭代数等齐全）
- [x] 数据集可获取（RoboCasa、LIBERO 公开；真实数据为自采）

---

## 速查卡片

> [!summary] GeoPredict: Predictive Kinematics + 3D Gaussian Geometry for Precise VLA
> - **核心**: 在 $\pi_0$ 上加两条"仅训练期"的预测支路——多步 3D 关键点轨迹 + 轨迹引导细化的预测式 3D 高斯几何（深度渲染监督），把运动学/几何先验注入 Transformer 表征，推理时零额外 3D 解码。
> - **方法**: Track Encoder 压历史轨迹 → Future Track Query 预测未来轨迹（$\mathcal{L}_{\text{track}}$）→ 3D Spatial Query + Voxel Decoder 出 3DGS、用预测轨迹做 track-guided 细化 → masked 深度渲染监督（$\mathcal{L}_{\text{depth}}$）；块级因果注意力组织"感知→预测→控制"；$\mathcal{L}_{\text{total}}=\lambda_1\mathcal{L}_{\text{action}}+\lambda_2\mathcal{L}_{\text{track}}+\lambda_3\mathcal{L}_{\text{depth}}$。
> - **结果**: RoboCasa Human-50 52.4%（+10.1% vs $\pi_0$）、LIBERO 96.5%（SOTA）、真实世界 Spatial/Geometry/Robustness 85/95/90%（基线 60/50/35）。
> - **代码**: 正文未提供。

---

*笔记创建时间: 2026-06-29*
