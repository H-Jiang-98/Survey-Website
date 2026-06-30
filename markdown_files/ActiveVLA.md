---
title: "ActiveVLA: Injecting Active Perception into Vision-Language-Action Models for Precise 3D Robotic Manipulation"
method_name: "ActiveVLA"
authors: [Zhenyang Liu, Yongchong Gu, Yikai Wang, Xiangyang Xue, Yanwei Fu]
year: 2026
venue: CVPR
tags: [VLA, active-perception, 3D-manipulation, coarse-to-fine, viewpoint-selection, heatmap]
zotero_collection: _inbox
image_source: online
arxiv_html: https://arxiv.org/html/2601.08325v1
created: 2026-06-29
---

# ActiveVLA: Injecting Active Perception into Vision-Language-Action Models for Precise 3D Robotic Manipulation

## 元信息

| 项目 | 内容 |
|------|------|
| 作者 | Zhenyang Liu, Yongchong Gu, Yikai Wang, Xiangyang Xue, Yanwei Fu |
| 机构 | 复旦大学（Fudan University）等 |
| 会议 | CVPR 2026 |
| 类别 | VLA（Vision-Language-Action）/ 3D 机器人操作 |
| 日期 | 2026-01（arXiv v1） |
| 项目主页 | （未提供） |
| 链接 | [arXiv](https://arxiv.org/abs/2601.08325) / [CVPR Poster](https://cvpr.thecvf.com/virtual/2026/poster/36656) |

---

## 一句话总结

> 给 [[VLA]] 注入"主动感知"：用 coarse-to-fine 两阶段先在多视角正交投影上定位 3D 关键区域，再主动选最优视角并做 3D zoom-in，从而在精细、长程、严重遮挡的操作任务上大幅超越固定视角基线。

---

## 核心贡献

1. **把主动感知引入 VLA**: 提出 [[ActiveVLA]]，一个 coarse-to-fine 的 3D [[VLA]] 框架，让机器人能在执行过程中**自适应地选择视角与分辨率**，打破现有方法依赖固定、腕装、以末端为中心的相机这一根本局限。
2. **关键区域定位 + 主动视角选择**: 把 3D 输入投影成多视角 2D 正交图，用 VLM 主干预测 [[Heatmap|热力图]] 反投影回 3D 定位关键区域；再用一个**基于假设检验的多目标打分**视角选择策略，在最大化"非模态可见性（amodal relevance）"与多样性、最小化遮挡之间权衡。
3. **主动 3D Zoom-in**: 在选定视角上以缩小 FoV 的方式在虚拟渲染空间做光学式 zoom-in，在不损失几何信息的前提下提升关键区域分辨率，从而显著改善精细 6-DoF 位姿预测。在 RLBench / COLOSSEUM / GemBench 三个基准均取得 SOTA，并无缝迁移到真实机器人。

---

## 问题背景

### 要解决的问题
如何让 [[VLA]] 在**长程（long-horizon）**与**精细（fine-grained）**操作中获得足够的感知信息——尤其是在小物体、严重遮挡、复杂空间结构的场景下，固定视角往往看不全、看不清，导致动作预测精度不足。

### 现有方法的局限
作者指出当前 3D VLA 普遍忽视"主动感知（active perception）"：
1. 通常依赖**静态、腕装相机**，提供的是以末端执行器为中心的固定视角；
2. 无法在任务执行中**自适应选择最优视角或分辨率**；
3. 因此在**长程任务**与**精细操作**场景下性能受限——遮挡、视角歧义、小物体细节缺失都会直接拖垮策略。

形式化地，标准 VLA 设置中观测 $\mathbf{o}$ 来自一个或多个固定、以末端为中心的 RGB-D 视角，这从根本上**限制了感知灵活性**，既阻碍上下文获取，也削弱策略泛化。

### 本文的动机
- 人在操作时会**主动调整观察姿态**（凑近看、换角度看）。把这种能力显式建模进 VLA：先粗看全局定位关键区域，再细看（换视角 + 放大）。
- 用 **coarse-to-fine** 把"看哪里（探索 / view selection）"与"看多近（利用 / zoom-in）"解耦成层次化感知策略；
- 沿用 [[BridgeVLA]] 把 3D 模态投影成 2D、用 VLM 预测 heatmap 再反投影的范式，便于复用强 VLM 先验（[[PaliGemma]]），并保证 2D-3D 几何一致。

---

## 方法详解

### 模型架构

[[ActiveVLA]] 采用 **coarse-to-fine（粗到细）两阶段** 流水线（见 Figure 1）：

- **输入**: 多路标定相机的 RGB-D 图像（重建点云）+ 语言指令 $l$
- **Backbone**: [[PaliGemma]]（[[SigLIP]] 视觉编码器 + [[Gemma]] 解码器），采用 [[BridgeVLA]] 在 120K RoboPoint 子集上预训练的权重
- **粗阶段（3.1 关键区域定位）**: 把点云渲染成 top/front/right 三视角正交投影 → VLM 预测 heatmap → 反投影定位 3D 关键区域 $p_f$
- **细阶段（3.2 主动感知）**: 主动视角选择（Active View Selection, A-VS）+ 主动 3D Zoom-in（A-3Z）
- **动作预测（3.3）**: 多视角 score volume 取平移 + 离散 Euler 角取旋转，global-local 融合预测旋转/夹爪/碰撞
- **输出**: 关键帧动作 $\mathbf{a}$ = 6-DoF 末端位姿 $T\in SE(3)$ + 夹爪状态 $g\in\{0,1\}$ + 碰撞标志 $c\in\{0,1\}$

VLA 策略形式化为从观测与指令到动作的映射：

$$
\pi:(\mathbf{o},l)\mapsto\mathbf{a}
$$

其中专家示范集 $\mathcal{D}=\{\tau^{i}\}_{i=1}^{N}$，每条轨迹 $\tau^{i}=\{l^{i},(\mathbf{o}_{1}^{i},\mathbf{a}_{1}^{i}),\dots,(\mathbf{o}_{H}^{i},\mathbf{a}_{H}^{i})\}$。

### 核心模块

#### 模块1: 3D Crucial Area Perception（关键区域定位，粗阶段）

**设计动机**: VLM 的全局表征不足以做精确空间定位，需要恢复细粒度空间注意力来锁定关键 3D 区域。

**具体实现**:
- **多视角渲染**: 用标定相机重建点云，再用 [[PyTorch3D]] 渲染 top/front/right 三张正交投影；每个视角渲 **7 通道**（RGB 3 + depth 1 + 世界系坐标 $(x,y,z)$ 3）。坐标通道用于建立跨视角像素对应：不同视角若像素共享同一 $(x,y,z)$ 即对应同一 3D 点。
- 每个像素取**最小深度**的点着色，正确处理遮挡。
- **关键区域提取**: 把 VLM 输出的 patch token $\{\mathbf{t}_i\}_{i=1}^M$ 按空间位置重排成特征网格，经 **convex upsampling block**（学习像素级权重，优于固定插值）上采样到输入分辨率得 heatmap；用**交叉熵损失**训练，再把各视角 heatmap 反投影到 3D 锁定关键区域。

#### 模块2: 3D Active Perception（主动感知，细阶段）

包含两个组件：**Active View Selection（A-VS）** 与 **Active 3D Zoom-in（A-3Z）**。

**A-VS 设计动机**: 在复杂场景中提升感知完整性——围绕粗阶段锁定的关键区域选最优相机视角，最大化非模态可见性（确保目标完整可见）与视角多样性，减少遮挡与歧义。

**A-VS 具体实现**:
- 在以关键区域 $p_f\in\mathbb{R}^3$ 为球心的球面上**均匀采候选相机位**；用**正二十面体递归细分（geodesic / icosahedron subdivision）**得到近似均匀分布，避免经纬度参数化的采样偏差。
- 对每个候选 $c_i$ 用**多目标打分函数**评估三个准则：**可见性**（KDTree 沿视线采点查最近表面距离，全部 $\geq r$ 则不遮挡）、**距离**（标准化后偏好中等观察距离）、**多样性**（与其它候选视向的总角度间隔）。
- 三项 Z-normalize 后加权求和，取 top-$K$ 作为下一观测姿态；相机用 look-at 配置（eye $c_i$、target $p_f$、动态 up 向量）。最具信息量的视角作为 zoom-in 基础。

**A-3Z 设计动机**: 固定视角对小物体（如"用焊枪焊孔"）细节不足；需在关键区域自适应放大。

**A-3Z 具体实现**: 在选定相机位姿上**缩小 FoV 重渲染**，在保持像素分辨率的同时放大局部区域，模拟光学 zoom；基于 3D 点云做尺度不变的视图合成，无几何损失，从而提升夹爪位姿精度。把探索（视角选择）与利用（zoom-in）分离，形成层次化感知策略。

#### 模块3: 3D Action Prediction（动作预测）

**设计动机**: 把主动选取并放大的多视角统一起来，预测精确且安全的 6-DoF 动作。

**具体实现**:
- **平移**: 各视角 heatmap 反投影累积到离散网格 $\mathcal{G}$ 形成多视角 score volume，取最大值点作为平移目标 $\mathbf{t}^*=\arg\max_{\mathbf{g}}S(\mathbf{g})$。
- **旋转**: 用 Euler 角 $(\phi,\theta,\psi)$ 表示，每个角离散为 **72 bins**。
- **层次化特征融合**: Global Context（对各正交投影的视觉编码 max-pool 得 3 个全局 token）+ Local Context（ROI-aware sampler 取局部 token），全部拼接经 MLP 头预测旋转、夹爪、碰撞标志。

### 关键公式与机制

#### 公式1: [[VLA]] 策略映射

$$
\pi:(\mathbf{o},l)\mapsto\mathbf{a}
$$

**含义**: 策略 $\pi$ 把观测 $\mathbf{o}$ 与语言指令 $l$ 映射到关键帧动作 $\mathbf{a}$。

**符号说明**:
- $\mathbf{o}$: 一路或多路固定视角的 RGB-D 观测；$l$: 语言指令
- $\mathbf{a}=(T,g,c)$: 6-DoF 位姿 $T\in SE(3)$、夹爪状态 $g\in\{0,1\}$、碰撞标志 $c\in\{0,1\}$

#### 公式2: 正交投影渲染

$$
I^{(v)}(u_{x},u_{y})=\sum_{i=1}^{N}\mathbf{c}_{i}\cdot\delta\big((u_{x},u_{y})-\pi^{(v)}(\mathbf{p}_{i})\big)
$$

**含义**: 在视角 $v$ 下，像素 $(u_x,u_y)$ 的颜色由所有投影到该像素的 3D 点贡献；实际取最小深度点以正确处理遮挡。

**符号说明**:
- $\pi^{(v)}(\cdot)$: 视角 $v$ 的正交投影；$\mathbf{p}_i,\mathbf{c}_i$: 第 $i$ 个点的坐标与颜色
- $\delta(\cdot)$: 指示像素与投影是否重合；$N$: 点数

#### 公式3: Heatmap 预测（convex upsampling）

$$
\mathbf{H}=\mathcal{U}\Big(\mathrm{Rearrange}\big(\{\mathbf{t}_{i}\}_{i=1}^{M}\big)\Big)
$$

**含义**: 把 VLM patch token 按空间重排成特征网格，再经 convex upsampling 块恢复到输入分辨率的热力图。

**符号说明**:
- $\mathcal{U}(\cdot)$: convex upsampling 块（学习像素级权重）
- $\mathrm{Rearrange}(\cdot)$: 重排成 $H_p\times W_p$ 特征网格；$\{\mathbf{t}_i\}_{i=1}^M$: 输出 patch token

#### 公式4: 球面候选视角采样数

$$
V(k)=12+30k+\frac{20}{3}\left(4^{k}-1\right)
$$

**含义**: 正二十面体经 $k$ 级递归细分后球面上的候选相机点数，控制视角密度。

**符号说明**:
- $k\in\mathbb{N}$: 细分级数，$k=0$ 即原始二十面体（12 顶点）
- 更大的 $k$ 对应更密的近似均匀采样

#### 公式5: 视线遮挡检测距离

$$
d_{k}=\min_{s\in\mathcal{S}}\|q_{k}-s\|
$$

**含义**: 沿候选视线均匀采点 $q_k$，计算其到观测点云最近表面的距离；若所有 $d_k\geq r$ 则该视角不遮挡（$v(c_i,p_f)=1$）。

**符号说明**:
- $\mathcal{S}$: 观测点云；$q_k$: 视线上第 $k$ 个采样点；$r$: 遮挡阈值

#### 公式6: 视角多样性得分

$$
S_{\text{div}}(c_{i})=\sum_{j\neq i}\arccos(\mathbf{v}_{i}\cdot\mathbf{v}_{j})
$$

**含义**: 候选 $c_i$ 与其它所有候选视向的总角度间隔，值越大表示视角越多样、分布越分散。

**符号说明**:
- $\mathbf{v}_i\in\mathbb{S}^2$: 候选 $c_i$ 的单位视向向量；$\arccos(\mathbf{v}_i\cdot\mathbf{v}_j)$: 两视向夹角

#### 公式7: 多目标统一打分

$$
s_{i}=w_{\text{vis}}\cdot s_{\text{vis}}+w_{\text{dis}}\cdot s_{\text{dis}}+w_{\text{div}}\cdot s_{\text{div}}
$$

**含义**: 把可见性、距离、多样性三项 Z-normalize 后加权求和，取 top-$K$ 为下一观测姿态。

**符号说明**:
- $w_{\text{vis}}+w_{\text{dis}}+w_{\text{div}}=1$: 权重系数
- $s_{\text{vis}},s_{\text{dis}},s_{\text{div}}$: 可见性 / 距离 / 多样性得分

#### 公式8: Zoom-in 空间覆盖宽度

$$
W(z)=2d\tan\left(\frac{\alpha}{2z}\right)
$$

**含义**: 缩小 FoV 实现虚拟光学 zoom；$W(z)$ 随 $z$ 增大而减小，而像素分辨率 $R=\dfrac{\text{image width (pixels)}}{W(z)}$ 因此提升，关键区域更清晰。

**符号说明**:
- $\alpha$: 原始 FoV（弧度）；$z>1$: zoom-in 因子；$d$: 相机到关键区域距离
- $W(z)$: 渲染图垂直于视向的空间覆盖宽度

#### 公式9: 多视角 score volume（平移预测）

$$
S(\mathbf{g})=\sum_{v=1}^{3}w_{v}\,h_{v}\!\left(\pi_{v}(\mathbf{g})\right)
$$

**含义**: 把三视角的 heatmap 反投影累积到离散网格上，得分最高的网格点即平移目标 $\mathbf{t}^*=\arg\max_{\mathbf{g}}S(\mathbf{g})$。

**符号说明**:
- $h_v$: 视角 $v$ 的 heatmap；$\pi_v(\mathbf{g})$: 网格点 $\mathbf{g}$ 在视角 $v$ 的 2D 投影；$w_v$: 视角权重

---

## 关键图表

<!-- 图片均使用 arXiv HTML 在线链接，未本地化 -->

### Figure 1: Pipeline of ActiveVLA / 整体流水线

![Figure 1](https://arxiv.org/html/2601.08325v1/x1.png)

**说明**: ActiveVLA 的 coarse-to-fine 两阶段框架。**粗阶段**: 3D 场景的三张正交投影 + 语言指令送入 [[PaliGemma]] 主干生成 2D heatmap，反投影定位最相关 3D 区域。**细阶段**: 主动感知模块选新视角并在该区域做 3D zoom-in；refined PaliGemma 预测末端关键位置 heatmap，action decoder 输出最终 3D 动作。这张图是理解全文"先粗看定位、再细看精操"逻辑的总图。

### Figure 2: Qualitative Results of Fine-grained Manipulation / 精细操作定性结果

![Figure 2](https://arxiv.org/html/2601.08325v1/x2.png)

**说明**: 虚线左侧（粗阶段）：(a) 把 3D 模态投影成正交图、(b) 预测 heatmap 标关键区域；虚线右侧（细阶段）：(c) 主动视角选择、(d) 主动 3D zoom-in。直观展示四个子步骤如何在复杂场景中协同实现精细操作。

### Figure 3: Visualization in Complex Manipulation Tasks / 复杂任务可视化

![Figure 3](https://arxiv.org/html/2601.08325v1/x3.png)

**说明**: ActiveVLA 在真实复杂任务中的执行序列。即便目标被严重遮挡、空间结构复杂，它仍能主动感知并精确完成，佐证主动感知对真实世界鲁棒性的价值。

### Figure 4: Hyperparameter Analysis / 超参分析

![Figure 4](https://arxiv.org/html/2601.08325v1/x4.png)

**说明**: RLBench 上的超参敏感度。(a) 选取视角数：从 1 视角 82.2% 提升到 3 视角 91.8%，>3 后饱和而算力上升；(b) zoom-in 因子：1→4 逐步提升精度，过大则丢上下文。故默认取 **3 视角、zoom-in 因子 4**。

### Figure 5: 18 RLBench Tasks / RLBench 18 任务可视化（附录）

![Figure 5](https://arxiv.org/html/2601.08325v1/x5.png)

**说明**: 评测覆盖的 18 个 RLBench 任务，涵盖抓取、工具使用、复杂空间交互等多样挑战。

### Figure 6: Direct Manipulation Tasks / 直接操作任务可视化（附录）

![Figure 6](https://arxiv.org/html/2601.08325v1/x6.png)

**说明**: 评估基础视觉运动技能的直接操作任务（含布料操作），体现不同物理/空间挑战。

### Figure 7: Occlusion-heavy & Spatially Complex Tasks / 遮挡-复杂空间任务可视化（附录）

![Figure 7](https://arxiv.org/html/2601.08325v1/x7.png)

**说明**: 重点考验主动视角选择与 3D zoom-in 的难任务：多层抽屉取毛巾（towel）、部分遮挡下高精度堆叠（red to green block）、杂乱中避碰抓取（occluded banana）、挂架窄缝取杯（purple cup）——强调严重遮挡、复杂几何与精细 6D 位姿推理。

### Table 1: Results on RLBench / RLBench 结果（18 任务，节选）

| Method | Avg. SR (%) ↑ | Avg. Rank ↓ | Insert Peg | Place Cups | Stack Cups | Sort Shape | Screw Bulb |
|--------|------|------|------|------|------|------|------|
| Image-BC (CNN) | 1.3 | 11.56 | 0.0 | 4.0 | 0.0 | 0.0 | 0.0 |
| C2F-ARM-BC | 20.1 | 9.89 | 4.0 | 0.0 | 0.0 | 8.0 | 8.0 |
| PerAct | 49.4 | 7.33 | 5.6 | 2.4 | 2.4 | 16.8 | 17.6 |
| Act3D | 65.0 | 5.28 | 27.0 | 3.0 | 9.0 | 8.0 | 47.0 |
| RVT | 62.9 | 5.39 | 11.2 | 4.0 | 26.4 | 36.0 | 48.0 |
| 3D Diffuser Actor | 81.3 | 3.39 | 65.6 | 24.0 | 47.2 | 44.0 | 82.4 |
| RVT-2 | 81.4 | 3.00 | 40.0 | 38.0 | 69.0 | 35.0 | 88.0 |
| BridgeVLA | 88.2 | 2.44 | 88.0 | 58.4 | 81.6 | 60.8 | 87.2 |
| **ActiveVLA (Ours)** | **91.8** | **1.22** | **92.4** | **65.6** | **84.8** | **63.3** | **89.3** |

**说明**: ActiveVLA 以 **91.8% 平均成功率、平均排名 1.22** 刷新 RLBench SOTA，在 18 个任务中 **10 个夺冠**。在精度要求高、接触丰富的任务（Insert Peg、Open Drawer 均最优）及遮挡任务（Place Cups 65.6%，远超次优 BridgeVLA 58.4%）上优势尤为明显——这正是主动感知发挥作用的场景。

### Table 2: Results on COLOSSEUM / COLOSSEUM 鲁棒性结果（14 类扰动，节选）

| Method | Avg. SR (%) ↑ | Avg. Rank ↓ | MO-SIZE | RO-SIZE | Table Color | Camera Pose | Distractor |
|--------|------|------|------|------|------|------|------|
| PerAct | 27.9 | 4.79 | 35.6 | 29.3 | 30.4 | 36.3 | 27.1 |
| RVT | 35.4 | 4.36 | 35.3 | 40.5 | 30.0 | 42.2 | 18.8 |
| RVT-2 | 56.7 | 2.86 | 60.9 | 53.4 | 62.6 | 64.4 | 60.8 |
| BridgeVLA | 64.0 | 2.07 | 69.3 | 61.7 | 75.7 | 73.8 | 51.8 |
| **ActiveVLA (Ours)** | **65.9** | **1.07** | **72.4** | **64.4** | **78.3** | **76.3** | **54.3** |

**说明**: COLOSSEUM 用 12+ 类物体/场景/相机扰动测鲁棒性。ActiveVLA 平均 **65.9%、排名 1.07**，比此前最佳（BridgeVLA 64.0%）高 **1.9 个百分点**，在尺寸、颜色、光照、纹理变化下都稳；Table Color(78.3%) 与 Camera Pose(76.3%) 表现最好，说明主动感知带来更强视觉泛化与不变表征。

### Table 3: Results on GemBench / GemBench 分层泛化结果

| Method | Average | L1 | L2 | L3 | L4 |
|--------|------|------|------|------|------|
| Hiveformer | 30.4 | 60.3 | 26.1 | 35.1 | 0.0 |
| PolarNet | 38.4 | 77.7 | 37.1 | 38.5 | 0.1 |
| 3D Diffuser Actor | 44.0 | 91.9 | 43.4 | 37.0 | 0.0 |
| RVT-2 | 44.0 | 89.1 | 51.0 | 36.0 | 0.0 |
| 3D-LOTUS | 45.7 | 94.3 | 49.9 | 38.1 | 0.3 |
| 3D-LOTUS++ | 48.0 | 68.7 | 64.5 | 41.5 | **17.4** |
| BridgeVLA | 50.0 | 91.1 | 65.0 | 43.8 | 0.0 |
| **ActiveVLA (Ours)** | **51.3** | **92.4** | **66.3** | **45.1** | 1.2 |

**说明**: GemBench 是分层（L1-L4）组合泛化基准。ActiveVLA 在 L1-L3 全面最优，总均 **51.3%** 比此前 SOTA 高 1.3 个百分点。但最难的 L4 仅 1.2%（远逊于 3D-LOTUS++ 的 17.4%），暴露其在极端长程组合任务上仍有短板。

### Table 4: Ablation on Key Components / 核心组件消融（成功率% / 推理时间 s，100 试）

| A-VS | A-3Z | RLBench | COLOSSEUM | GemBench |
|------|------|---------|-----------|----------|
| | | 87.6 / 0.26 | 63.6 / 0.33 | 48.9 / 0.21 |
| ✔ | | 89.4 / 0.45 | 64.5 / 0.51 | 49.4 / 0.48 |
| **✔** | **✔** | **91.8 / 0.53** | **65.9 / 0.62** | **51.3 / 0.59** |

**说明**: 固定视角基线 RLBench 87.6%（0.26s）；加 A-VS（主动选视角、改善覆盖、减遮挡）→ 89.4%（0.45s）；再加 A-3Z（虚拟光学 zoom、高分辨率特写）→ 91.8%（0.53s）。两者各司其职——**A-VS 决定看哪里、A-3Z 决定看多近**，构成层次化感知，但代价是推理时间约翻倍。

### Table 5: Real-World Experiment / 真实世界成功率（%）

| Method | Retrieving Towel | Red→Green Block | Occluded Banana | Occluded Purple Cup | Overall |
|--------|------|------|------|------|------|
| Diffusion Policy | 26 | 38 | 42 | 35 | 35.3 |
| VPP | 52 | 48 | 58 | 64 | 55.5 |
| TriVLA | 68 | 54 | 62 | 72 | 64.0 |
| RVT-2 | 77 | 63 | 72 | 78 | 72.5 |
| **ActiveVLA (Ours)** | **92** | **95** | **91** | **89** | **91.8** |

**说明**: 真机（KINOVA GEN2 + RealSense D455，eye-to-hand）四个高遮挡任务。ActiveVLA 总体 **91.8%**，大幅领先次优 RVT-2(72.5%)；在多层抽屉取毛巾(92)、部分遮挡堆叠(95)、杂乱避碰抓香蕉(91)、挂架窄缝取杯(89) 上全面碾压，验证主动感知在真实遮挡场景的实用性。

### Table 6: Per-task COLOSSEUM Perturbation Results / COLOSSEUM 逐任务扰动结果（附录，节选）

| Task | Original | All Pert. | MO-SIZE | Table Color | Distractor | Camera Pose |
|------|------|------|------|------|------|------|
| basketball_in_hoop | 100.0 | 6.2 | 100.0 | 100.0 | 38.7 | 100.0 |
| close_box | 100.0 | 73.5 | 94.8 | 100.0 | 99.0 | 100.0 |
| insert_onto_square_peg | 94.7 | 24.5 | 86.7 | 89.5 | 45.5 | 96.0 |
| open_drawer | 97.0 | 61.5 | 91.5 | 94.5 | 91.5 | 96.5 |
| slide_block_to_target | 100.0 | 25.5 | - | 100.0 | 85.5 | 100.0 |
| turn_oven_on | 94.5 | 86.5 | 91.5 | 95.5 | 97.0 | 100.0 |
| empty_dishwasher | 0.0 | 0.5 | 4.5 | 0.0 | 0.5 | 0.0 |
| wipe_desk | 0.0 | 0.5 | 0.5 | 0.0 | 0.0 | 0.0 |
| **Task Mean** | **74.5** | **20.0** | **71.0** | **77.0** | **53.0** | **75.0** |

**说明**: 逐任务（均值±标准差，此处略去标准差）展示 ActiveVLA 对各类视觉/空间扰动的稳定性。多数任务在颜色/纹理/相机位姿扰动下仍保持高成功率（任务均值 74.5%→各扰动 71-77%），但 "All Perturbations"（多扰动叠加）骤降到 20.0%，且 empty_dishwasher / wipe_desk 几乎全为 0——说明同时叠加多种扰动与少数复杂任务仍是难点。

---

## 实验

### 数据集 / 基准

| 基准 | 规模 | 特点 | 用途 |
|------|------|------|------|
| [[RLBench]] | 18 任务，每任务 100 示范，Franka Panda，4 标定相机 RGB-D | 长程 + 精细操作 | 训练/测试 |
| [[COLOSSEUM]] | 在 RLBench 上加 12 类扰动（物体/场景/相机变化） | 鲁棒性 / 泛化评测 | 测试 |
| [[GemBench]] | 16 训练 + 44 测试任务，7 个核心动作基元，L1-L4 分层 | 组合性 / 泛化 | 训练/测试 |
| 真实世界 | KINOVA GEN2 + RealSense D455（eye-to-hand），4 个高遮挡任务 | sim-to-real | 训练/测试 |

### 实现细节

- **Backbone**: [[BridgeVLA]] 预训练的 [[PaliGemma]]（[[SigLIP]] 编码器 + [[Gemma]] 解码器），在 120K-image RoboPoint 子集上预训练
- **渲染**: [[PyTorch3D]]，top/front/right 三正交投影，每视角 7 通道（RGB+depth+world coord）
- **关键超参**: 选 **3 个视角**、zoom-in 因子 **4**；旋转 Euler 角每轴离散 **72 bins**；视角采样用二十面体测地细分
- **训练目标**: heatmap 预测用交叉熵损失
- **硬件**: 8× NVIDIA H100 GPU，192-vCPU Intel Xeon Platinum 8468

### 关键实验结论

- **RLBench**: 91.8%（SOTA，rank 1.22），10/18 任务夺冠，精细/接触/遮挡任务优势最大。
- **COLOSSEUM**: 65.9%（rank 1.07），比 BridgeVLA 高 1.9pt，对尺寸/颜色/光照/纹理/相机扰动鲁棒。
- **GemBench**: 51.3% 总均（+1.3pt），L1-L3 全优；L4 仅 1.2% 为短板。
- **真实世界**: 总体 91.8%，远超 RVT-2(72.5%)，验证遮挡场景的主动感知收益。
- **消融**: A-VS（+1.8pt RLBench）与 A-3Z（再+2.4pt）累计收益显著，推理时间从 0.26s 增至 0.53s。
- **超参**: 3 视角、zoom 因子 4 为甜点；过多视角饱和、过度 zoom 丢上下文。

---

## 批判性思考

### 优点
1. **概念切中要害**: "主动感知"显式解决了固定腕装相机看不全/看不清的根本问题，三个基准 + 真机一致 SOTA，遮挡与精细任务上的提升尤其有说服力（Place Cups +7pt、真机 +19pt）。
2. **方法工程化、几何严谨**: 多视角打分（可见性/距离/多样性）、二十面体测地采样、虚拟 zoom 的 FoV-分辨率公式都给了清晰的几何定义，可解释性强且无几何损失。
3. **解耦设计干净**: 把"看哪里（A-VS）"与"看多近（A-3Z）"解耦为层次化感知，消融清楚地量化了各自贡献。

### 局限性
1. **极端长程/多扰动仍弱**: GemBench L4 仅 1.2%（被 3D-LOTUS++ 17.4% 碾压），COLOSSEUM "All Perturbations" 仅 20.0%，empty_dishwasher/wipe_desk 近乎 0——主动感知未必能救复杂长程组合任务。
2. **推理开销翻倍**: A-VS+A-3Z 把 RLBench 推理从 0.26s 增至 0.53s，且每步需重渲染点云、KDTree 遮挡检测、多候选打分，实时性与可扩展性存疑，论文未报控制频率。
3. **依赖精确点云与标定**: 整套流程建立在多标定相机重建点云之上，真机用 eye-to-hand，对深度噪声、标定误差、动态/可变形物体的敏感度未系统评估；权重/打分系数（$w_{vis},w_{dis},w_{div}$、阈值 $r$）取值与敏感度也未充分给出。

### 潜在改进方向
1. 把视角选择从"启发式多目标打分"升级为可学习/信息增益驱动（如 next-best-view 的 RL 或预测式不确定性），并联合优化 zoom 因子。
2. 针对长程 L4 与多扰动叠加，结合显式子目标分解 / 记忆，缓解主动感知在组合泛化上的瓶颈。
3. 用可微渲染或缓存复用降低每步重渲染+KDTree 的开销，报告真实控制频率，做实时性-精度权衡分析。

### 可复现性评估
- [ ] 代码开源（论文未给出项目主页/代码链接）
- [x] 预训练模型（基于公开的 [[BridgeVLA]] / [[PaliGemma]] 权重）
- [x] 训练细节较完整（基准、超参、硬件、离散化粒度均有交代）
- [x] 数据集可获取（RLBench / COLOSSEUM / GemBench 均公开）

---

## 速查卡片

> [!summary] ActiveVLA: 把主动感知注入 3D VLA
> - **核心**: coarse-to-fine 两阶段——先在多视角正交投影上预测 heatmap 定位 3D 关键区域，再主动选最优视角 + 3D zoom-in 做精细操作。
> - **方法**: PaliGemma(BridgeVLA 权重) 主干；三正交投影(7 通道) → convex upsampling heatmap → 反投影定位 $p_f$ → 二十面体测地采样 + 可见性/距离/多样性打分选 top-K 视角 → 缩 FoV 虚拟 zoom → score volume 取平移 + 72-bin Euler 取旋转 + global-local 融合。
> - **结果**: RLBench 91.8%(rank 1.22) / COLOSSEUM 65.9%(+1.9pt) / GemBench 51.3% / 真机 91.8%；3 视角、zoom 因子 4；A-VS+A-3Z 累计 +4.2pt 但推理 0.26→0.53s。
> - **代码**: 未公开（基于 BridgeVLA / PaliGemma）

---

*笔记创建时间: 2026-06-29*
