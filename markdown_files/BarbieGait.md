---
title: "BarbieGait: An Identity-Consistent Synthetic Human Dataset with Versatile Cloth-Changing for Gait Recognition"
method_name: "BarbieGait"
authors: [Qingyuan Cai, Saihui Hou, Xuecai Hu, Yongzhen Huang]
year: 2026
venue: CVPR
tags: [gait-recognition, synthetic-dataset, cloth-changing, normalization, biometrics, locomotion]
zotero_collection: _inbox
image_source: online
arxiv_html: https://arxiv.org/html/2604.12221v1
created: 2026-06-29
---

# BarbieGait: An Identity-Consistent Synthetic Human Dataset with Versatile Cloth-Changing for Gait Recognition

## 元信息

| 项目 | 内容 |
|------|------|
| 作者 | Qingyuan Cai, Saihui Hou, Xuecai Hu, Yongzhen Huang |
| 机构 | 北京师范大学人工智能学院、阿里巴巴 AMAP、WATRIX.AI |
| 会议 | CVPR 2026 |
| 类别 | 步态识别 / 合成数据集 / 跨衣识别（locomotion） |
| 日期 | 2026（arXiv v1，论文标注 2025） |
| 项目主页 | https://github.com/BarbieGait/BarbieGait |
| 链接 | [arXiv](https://arxiv.org/abs/2604.12221) / [Code](https://github.com/BarbieGait/BarbieGait) |

---

## 一句话总结

> 把真实受试者用 3D 骨骼/体型+运动学匹配映射进虚拟引擎，为每人合成 100 套换衣序列且保持步态身份不变，并提出以 [[Gait-Oriented Normalization|GON]] 归一化去除衣物统计的跨衣基线 [[GaitCLIF]]。

---

## 核心贡献

1. **首个身份一致的大规模换衣合成步态数据集**: 提出 [[BarbieGait]]，含 521 名真实受试者，每人 100 套随机全身换衣，通过"骨骼长度+体型匹配"与"运动学运动匹配"双重对齐，确保每条换衣序列都保留真实受试者的**步态身份信息**。在序列数量与服装多样性上均刷新现有数据集。
2. **跨衣不变特征基线 GaitCLIF**: 针对换衣带来的巨大类内方差，从"去除衣物统计"与"保留细粒度运动细节"两个视角出发，提出 [[Gait-Oriented Normalization|GON]] 归一化与 GON-P3D/GON-3D/GON-FC 模块，构成强基线 [[GaitCLIF]]。
3. **真实基准全面提升 + 上下游迁移价值**: 在 BarbieGait 以及 CCPG、SUSTech1K、Gait3D、GREW 上一致提升并达 SOTA；且 BarbieGait 既能作为**下游步态识别**的预训练源，又能改进**上游 2D 姿态估计**（ViTPose）对真实数据的泛化。

---

## 问题背景

### 要解决的问题
[[步态识别|Gait Recognition]] 作为一种可远距离、无需交互的生物特征技术，受**衣物/携带物**等协变量影响极大。但现实中根本**缺乏每个受试者的大量换衣数据**：在实验室数据集（CASIA-B、OU-MVLP）和野外数据集（Gait3D、GREW）里换衣样式都很有限；专门的跨衣数据集 CCPG 投入巨大，每人也只有 7 种换衣状态。没有海量换衣数据，就无法验证步态识别在"极端服装变化"下是否可靠、是否还有提升空间。而真实采集跨民族、跨季节、复杂样式的换衣步态数据**成本极高且涉及隐私，几乎不可行**。

### 现有方法的局限
现有的人体合成数据集主要服务于姿态估计、网格恢复、行人重识别：
- SURREAL 把衣物渲染到 [[SMPL]] 身体上，但衣物-人体对齐不准；
- AGORA 提供高质量真实扫描，但因扫描成本高**缺乏连续运动**；
- SynBody、GTA-Human 用参数化模型驱动生成视频，但**强调动作多样性、忽略步态身份**；
- RandPerson/UnrealPerson/SynPerson 面向合成 Re-ID 训练；
- VersatileGait 针对步态，展示了合成数据潜力。

它们普遍存在两类**破坏身份一致性**的问题：(1) 用同一段动作序列驱动不同受试者；(2) 用差异很大的步态模式驱动同一受试者。结果是同一人的合成序列无法维持身份一致。

### 本文的动机
作者提出核心问题：**能否用生成范式合成换衣步态数据、同时保住真实受试者的步态可判别性？** 为此 BarbieGait 用高精度 3D 人体姿态与网格，在**静态（骨骼长度/体型）+动态（运动学运动匹配）**两个层面对齐真实与虚拟受试者，杜绝"动作复用"与"步态漂移"。在建模侧，作者认为换衣的根源问题是**衣物相关统计量**带来的类内方差，且人体不同部位受衣物影响程度不同——因此应**去除衣物统计 + 保留细粒度运动细节**来学习跨衣不变特征。

---

## 方法详解

整个工作分两部分：**数据生成系统 BarbieGait** 与 **识别模型 GaitCLIF**。

### 一、BarbieGait 数据生成系统

#### 基石: 原始数据采集（Raw Data Collection）
部署 6 路同步相机阵列采集 521 名受试者真实步态，每人记录 3 段序列。先用预训练 [[HRNet]] 从多视角估计 2D 姿态，再用三角化（triangulation）+ [[EasyMoCap]] 估计 3D 人体姿态与网格。高质量 3D 姿态/网格是把身份信息从真人复制到虚拟人的"基石"。

整体生成系统（见 Figure 2）由五个组件构成：骨骼长度与体型匹配、随机换衣、运动学运动匹配、场景构建、渲染。

#### 模块1: Skeleton Length and Body Shape Matching（骨骼长度与体型匹配）
**设计动机**: 让 [[MakeHuman]] 生成的虚拟人与真实受试者在**静态身体属性**上对齐。

**具体实现**:
- **骨骼长度**: 3D 骨骼长度对一个受试者基本稳定，基于 3D 姿态对齐虚拟人的大腿、小腿等骨骼长度；
- **体型**: 用 EasyMoCap 逐帧估计 [[SMPL]] 网格；为减小网格恢复误差，定义 **12 个静态围度参数**（颈、胸、腰、臀等），取逐帧均值对齐每个受试者的虚拟身体。

#### 模块2: Random Dressing（随机换衣）
**设计动机**: 在可控方式下产生多样换衣。

**具体实现**: 用 MakeHuman 从多样的衣橱中为每个受试者随机选 **100 套服装**，遵循季节与日常实用的搭配策略（见 Figure 6 展示的发型/衣物/鞋/携带物多样性）。

#### 模块3: Kinematic Motion Matching（运动学运动匹配）
**设计动机**: 这是**保住步态身份的关键**——确保真实受试者与其在各种换衣下的虚拟角色之间**走路模式一致**。

**具体实现**（详见 Algorithm 1）:
- 借鉴角色动画里的 rig-to-rig 运动迁移：从原始 3D 姿态为每根骨骼构建局部坐标系，并以**四元数**形式计算其旋转；
- 这些局部空间旋转按预定义关节对应关系迁移到目标骨架，实现稳定、**无万向锁（gimbal-lock-free）**的受试者特定运动复现；
- 这种基于映射的对齐保证了所有换衣变体下的忠实运动迁移。

#### 模块4: Scene Construction（场景构建）
**设计动机**: 模拟真实采集条件、增强真实性与鲁棒性。

**具体实现**: 收集 **20 个室内外环境**，每个场景在 2.5m 高、相距 45°、半径 4m 的圆周上放置 **8 个相机**多角度拍摄；并通过摆放椅子/墙等障碍物引入**遮挡**、模拟日夜**光照变化**。

#### 模块5: Rendering（渲染）
**设计动机**: 高效的真实感渲染。

**具体实现**: 用 NVIDIA GPU 集群渲染。与 SynBody 渲染全幅 $1920\times1080$ 不同，BarbieGait **只聚焦人体与阴影区域**、静态区域只渲染一次，从而提升人体区域聚焦度、简化后处理，并把渲染速度**提升 5–6 倍**。

#### 数据处理与统计
- **多模态**: 除合成 RGB，用 [[PaddleSeg]] 得到分割[[轮廓图|silhouette]]，用 HRNet 预测逐帧 2D 姿态。
- **统计**: 521 人（174 男 / 347 女），年龄 5–80 岁、身高 110–192cm、体重 15–115kg；261 人训练（602,508 序列）、260 人测试（600,816 序列）。
- **衣物厚度协议（THK）**: 渲染无衣轮廓（Figure 3a）与穿衣轮廓（Figure 3b），取两者**非重叠区域**（Figure 3c）作为衣物复杂度，用无衣轮廓归一化得到相对厚度，分为 **THK0–THK9 共 10 档（每档增厚 15%）**。评测以 THK0 作 gallery，THK1–THK9 作 probe。指标用 Rank-1（R1）与 mAP。
- **隐私**: 获得受试者研究授权，原始数据不公开，只发布无人脸/环境/RGB 个人视觉信息的合成数据。

### 二、GaitCLIF 模型

GaitCLIF 从两个视角学习跨衣不变特征：**(1) 去除衣物特定统计；(2) 保留细粒度运动细节。**

#### 去除衣物统计（4.1）
衣物相关统计量制造了大类内方差、形成与衣物相关的"子域"，阻碍模型提取身份特征。作者发现域不变学习常用的 [[Instance Normalization|IN]] 因轮廓特征各通道含噪而**不适合步态**；遂提出 **Gait-Oriented Normalization（GON）**——受 [[Layer Normalization|LN]] 启发，但针对步态数据特性设计，逐帧跨通道去除衣物特定统计（Figure 4a）。

#### 保留细粒度运动细节（4.2）
**帧级建模（Frame-Level）**: 衣物变化在不同身体区域不均匀（头部变化小、下身因紧身裤/阔腿裤/裙子等变化大），全局归一化无法刻画这种细粒度差异。GON 因此把整张特征 $\text{X}\in\mathbb{R}^{N\times C\times H\times W}$ 按**水平分块** $\{x_0,\dots,x_i,\dots,x_m\}$ 分别归一化，再拼接，增强同一受试者跨衣的类内紧致性。为捕捉时序动态，扩展出 **GON-P3D** 与 **GON-3D** 两个时序变体（Figure 4b、c），用时序卷积增强运动表征。

**序列级建模（Sequence-Level）**: 主流架构中时序池化后的 Separate FC 层不足以应对大幅换衣。作者提出 **GON-FC**——两层 FC、每层后接 GON（Figure 4d），增强各细粒度区域的非线性表达并进一步压制衣物方差。

#### 整体框架（4.3）
GaitCLIF（Figure 4e）= 四个视觉 stage + 时序池化（TP）+ 水平池化（HP）+ 线性 Head。两个变体 **GaitCLIF-P3D** / **GaitCLIF-3D** 的视觉 stage 分别用 GON-P3D / GON-3D，二者 Head 都用 GON-FC。

### 关键公式与机制

#### 公式1: GON 的水平分块归一化

$$
\text{X}' = \text{GON}(\text{X}) = \text{Cat}\big(\text{GON}(x_0),\dots,\text{GON}(x_m)\big)
$$

**含义**: 把特征沿高度切成 $m{+}1$ 个水平区域，对每块**独立做 GON 归一化**后再拼接，得到跨衣不变特征 $\text{X}'$，使不同部位按各自衣物变化尺度去统计。

**符号说明**:
- $\text{X}\in\mathbb{R}^{N\times C\times H\times W}$: 输入特征（批量、通道、高、宽）
- $x_i$: 第 $i$ 个水平分块；$\text{GON}(x_i)\in\mathbb{R}^{N\times C\times h_i\times W}$
- $\text{Cat}(\cdot)$: 沿高度维拼接

#### 公式2: GON 归一化算子

$$
\text{GON}(x_i) = \gamma\left(\frac{x_i-\mu(x_i)}{\sigma(x_i)}\right)+\beta
$$

**含义**: 对每个分块用其自身的均值/标准差做标准化，再用可学习仿射参数缩放平移——去除该区域的"衣物风格统计"，保留身份相关结构。

**符号说明**:
- $\mu(x_i),\sigma(x_i)$: 分块 $x_i$ 在**所有通道 $C$ 与空间维 $h_i,W$** 上计算的均值与标准差（跨通道统计以压低轮廓噪声）
- $\gamma,\beta$: 可学习缩放/偏移参数

#### 公式3: 分块均值

$$
\mu(x_i)=\frac{1}{C h_i W}\sum_{c=1}^{C}\sum_{h=1}^{h_i}\sum_{w=1}^{W} x_{chw}
$$

**含义**: 跨通道与空间的均值，作为该分块衣物统计的中心。

**符号说明**: $C$ 通道数、$h_i$ 分块高、$W$ 宽；$x_{chw}$ 为对应位置特征值。

#### 公式4: 分块标准差

$$
\sigma(x_i)=\sqrt{\frac{1}{C h_i W}\sum_{c=1}^{C}\sum_{h=1}^{h_i}\sum_{w=1}^{W}\big(x_{chw}-\mu(x_i)\big)^2}
$$

**含义**: 同范围的标准差，度量该分块特征的离散程度，用于标准化分母。

**符号说明**: 同公式 3；$\mu(x_i)$ 为公式 3 的均值。

#### 机制: Algorithm 1 — Kinematic Motion Matching（运动学运动匹配）

- **输入**: 初始 A-Pose、第 $t$ 帧 3D 姿态 $P_t$、根关节 $root$。
- **函数 $CalQ$**: 计算把**局部坐标系（基于每根骨骼及其父骨骼）**相对世界坐标系的旋转参数化的**单位四元数**。
- **输出**: 第 $t$ 帧的动画姿态（Animated Pose）。
- **核心步骤**（据正文与算法行）:

$$
Q_s \leftarrow CalQ(A),\qquad Q_t \leftarrow CalQ(P_t)
$$

$$
Q \leftarrow Q_s(k)\times Q_t(k),\qquad Q \leftarrow Q_s(k)\times \Delta Q_t(k_p)\times Q_t(k)
$$

其中对根关节用 $v=J_c-J_p$（子关节减父关节得到骨向量）构建局部坐标系，$\Delta Q_t(k_p)$ 为父骨骼的相对旋转增量，$k$ 为关节索引、$k_p$ 为其父关节。该过程把真实 3D 姿态的局部骨骼旋转逐帧、无万向锁地迁移到虚拟骨架，保证换衣后步态身份一致。

---

## 关键图表

<!-- 图片均使用 arXiv HTML 在线链接，未本地化 -->

### Figure 1: BarbieGait Overview / 数据集总览

![Figure 1](https://arxiv.org/html/2604.12221v1/x1.png)

**说明**: BarbieGait 是身份一致的合成人体数据集，每个受试者有 **100 种服装组合**（含发型、上下装、携带物变化），并在多种环境下模拟多样光照，生成 RGB 图像并抽取多模态数据（2D 姿态、轮廓）用于步态识别。这是论文"身份一致 + 极致换衣多样性"主张的直观展示。

### Figure 2: BarbieGait Data Generation System / 数据生成系统

![Figure 2](https://arxiv.org/html/2604.12221v1/x2.png)

**说明**: 五阶段流水线。(a) 基于 3D 骨骼与体型用 [[MakeHuman]] 做真实人到虚拟人的**骨骼长度与体型匹配**；(b) **随机换衣**；(c) **运动学运动匹配**——跨同一真实受试者的不同服装对齐步态身份；(d) 用 [[Blender]] 构建多场景、多视角采集；(e) 渲染。这张图是整篇数据贡献的方法核心。

### Figure 3: Clothing Complexity and Thickness / 衣物复杂度与厚度协议

![Figure 3](https://arxiv.org/html/2604.12221v1/x3.png)

**说明**: (a) 无衣轮廓；(b) 穿衣轮廓；(c) 两者非重叠区域 = 衣物复杂度；(d) 受试者在各厚度档（THK0–THK9）的分布。这定义了论文用来分级评测难度的 **THK 协议**——衣物越厚（probe 厚度档越高），识别越难。

### Figure 4: Overview of GaitCLIF / 模型总览

![Figure 4](https://arxiv.org/html/2604.12221v1/x4.png)

**说明**: (a) **GON** 核心归一化单元；(b) **GON-P3D**、(c) **GON-3D** 两种基于 GON 的视觉块；(d) **GON-FC**（GON 增强的 FC 块，用于 Head）；(e) 整体 GaitCLIF 框架（四视觉 stage + TP + HP + 线性 Head）。这张图把"去衣物统计 + 保运动细节"的全部模块组织在一起。

### Figure 5: Pose Format / 姿态格式（附录）

![Figure 5](https://arxiv.org/html/2604.12221v1/x5.png)

**说明**: 实验使用的姿态格式：(a) COCO-17、(b) Barbie-17。后者是作者为步态场景选取的、与 COCO 语义/数量一致（覆盖头、躯干、下肢）的关键点定义，用于重训 ViTPose。

### Figure 6: Diverse Clothing Illustration / 服装多样性示例（附录）

![Figure 6](https://arxiv.org/html/2604.12221v1/x6.png)

**说明**: BarbieGait 涵盖多样发型、衣物、鞋、携带物，为换衣步态识别引入显著服装变化，佐证"100 套/人"的多样性主张。

### Figure 7: Synthesized Images / 合成图像示例（附录）

![Figure 7](https://arxiv.org/html/2604.12221v1/x7.png)

**说明**: 合成图在不同场景、真实光照、多样服装与自然遮挡下渲染，展示数据的真实感与覆盖度。

### Figure 8: Heatmap Visualization / 注意力热图可视化（附录）

![Figure 8](https://arxiv.org/html/2604.12221v1/x8.png)

**说明**: 轮廓法 (a)-(c) 与姿态法 (d)-(f) 的热图对比。(b)/(e) 为 DeepGaitV2/SkeletonGait 叠加在轮廓上的激活图，(c)/(f) 为加入 GaitCLIF 的效果——后者激活更聚焦于身份相关的运动区域，定性印证 GON 去衣物统计的有效性。

### Table 1: Dataset Comparison / 与现有数据集对比

| Type | Dataset | Year | Type | #Subj. | #Mesh | #Views | #Seq | Cloth-Changing | #Cloth/Subj. | GT Format |
|------|---------|------|------|--------|-------|--------|------|----------------|--------------|-----------|
| Motion | SURREAL [60] | 2017 | Synthetic | — | 145 | 1 | NA | ✗ | — | 3DM |
| Motion | AGORA [51] | 2021 | Synthetic | — | >350 | 1 | NA | ✗ | — | 3DM, silh. |
| Motion | GTA-Human [8] | 2021 | Synthetic | — | >600 | 1 | 20K | ✗ | — | 3DM |
| Motion | SynBody [73] | 2023 | Synthetic | — | 10,000 | 4 | 40K | ✗ | — | 3DM, silh. |
| Re-ID | RandPerson [64] | 2020 | Synthetic | — | 8,000 | 19 | NA | ✗ | — | NA |
| Re-ID | UnrealPerson [80] | 2021 | Synthetic | — | 3,000 | 34 | NA | ✗ | — | NA |
| Re-ID | SynPerson [69] | 2022 | Synthetic | — | 5,345 | 36 | NA | ✗ | — | NA |
| Gait | CASIA-B [76] | 2006 | Real | 124 | — | 11 | 13,640 | ✓ | 3 | NA |
| Gait | OU-MVLP [57] | 2018 | Real | 10,307 | — | 14 | 288,596 | ✗ | — | NA |
| Gait | GREW [82] | 2021 | Real | 26,345 | — | 882 | 128,671 | ✓ | 6 | NA |
| Gait | Gait3D [81] | 2022 | Real | 4,000 | — | 39 | 25,309 | ✗ | — | NA |
| Gait | CASIA-E [56] | 2023 | Real | 1,014 | — | 26 | 778,752 | ✓ | 3 | NA |
| Gait | VersatileGait [79] | 2023 | Synthetic | — | 10,000 | 44 | 1,320,000 | ✗ | — | NA |
| Gait | CCPG [40] | 2023 | Real | 200 | — | 10 | 16,566 | ✓ | 7 | NA |
| Gait | **BarbieGait (Ours)** | 2025 | Synthetic | **521** | **521,000** | 8 | **1,203,324** | **✓** | **100** | **3DJ, 3DM, silh.** |

**说明**: BarbieGait 拥有最丰富的服装多样性（**100 套/人**，远超 CCPG 的 7、GREW 的 6）、超百万级序列与最全 GT 格式（3D 关节 + 3D 网格 + 轮廓）。这是其作为"换衣步态基准"的核心定位证据。

### Table 2: Dataset-specific Configurations / 各数据集训练配置

| Dataset | Batch Size | Blocks | Milestones | Total |
|---------|-----------|--------|------------|-------|
| BarbieGait | (8,8) | [1,1,1,1] | (20k,40k,50k) | 60k |
| CCPG [40] | (8,16) | [1,1,1,1] | (20k,40k,50k) | 60k |
| SUSTech1K [55] | (8,8) | [1,1,1,1] | (20k,40k,40k) | 50k |
| Gait3D [81] | (32,4) | [1,4,4,1] | (20k,40k,50k) | 60k |
| GREW [82] | (32,4) | [1,4,4,1] | (80k,120k,150k) | 180k |

**说明**: 各数据集的 batch、各视觉 stage 的 block 数、学习率里程碑与总迭代数，便于复现。

### Table 3: Performance on BarbieGait / BarbieGait 主结果（预测轮廓 + 2D 姿态输入）

THK0 为 gallery，THK1–THK9 为 probe；AVG 为各厚度平均（R1 / mAP）。

| Method | THK1 (R1/mAP) | THK5 (R1/mAP) | THK9 (R1/mAP) | AVG (R1/mAP) |
|--------|---------------|---------------|---------------|--------------|
| GaitSet [10] | 20.6 / 23.3 | 7.4 / 10.9 | 3.2 / 6.2 | 9.7 / 12.8 |
| GaitPart [21] | 50.4 / 47.7 | 24.8 / 26.8 | 11.3 / 14.3 | 27.6 / 28.7 |
| GaitGL [42] | 44.7 / 34.0 | 22.3 / 19.5 | 13.8 / 13.8 | 25.2 / 21.3 |
| GaitBase [19] | 32.1 / 32.9 | 16.2 / 19.2 | 9.6 / 12.6 | 18.3 / 20.7 |
| DeepGaitV2-2D [18] | 59.6 / 57.9 | 36.4 / 38.2 | 18.0 / 21.9 | 36.8 / 38.3 |
| DeepGaitV2-3D [18] | 87.3 / 73.8 | 76.8 / 62.7 | 53.4 / 45.9 | 71.7 / 60.2 |
| DeepGaitV2-P3D [18] | 85.4 / 72.4 | 72.8 / 59.9 | 47.9 / 42.3 | 67.7 / 57.6 |
| **GaitCLIF-P3D (ours)** | 88.1 / 74.2 | 80.1 / 65.1 | 61.1 / 51.9 | 75.6 / 63.2 |
| **GaitCLIF-3D (ours)** | **90.7 / 75.3** | **84.5 / 67.2** | **68.5 / 55.9** | **80.4 / 65.7** |
| GaitGraph [59] | 14.0 / 18.0 | 9.8 / 13.6 | 6.5 / 10.3 | 10.0 / 14.0 |
| GaitGraph2 [58] | 36.5 / 37.2 | 28.9 / 31.4 | 20.8 / 23.8 | 27.7 / 30.1 |
| GaitTR [77] | 63.3 / 59.0 | 54.3 / 52.1 | 36.1 / 37.1 | 51.2 / 49.6 |
| GPGait [23] | 79.4 / 74.8 | 61.8 / 60.4 | 36.9 / 39.3 | 59.1 / 57.9 |
| SkeletonGait [20] | 91.4 / 85.5 | 81.2 / 75.7 | 56.1 / 54.5 | 77.1 / 72.3 |
| **GaitCLIF-P3D\* (ours, heatmap)** | **92.1 / 86.1** | **82.3 / 76.8** | **56.7 / 55.1** | **78.1 / 73.3** |

**说明**: 上半为外观法（轮廓输入），GaitCLIF-3D 取得 AVG 80.4/65.7，相对最强基线 DeepGaitV2-3D（71.7/60.2）大幅提升，且厚衣档（THK9）优势最明显（68.5 vs 53.4 R1）。下半为姿态法（heatmap 输入），GaitCLIF-P3D\* 以 78.1/73.3 超过 SkeletonGait（77.1/72.3）。注意姿态法整体 mAP（73.3）反超外观法（65.7），印证"姿态对外观变化更鲁棒"。

### Table 4: Ablation on BarbieGait / 模块消融

| GON-P3D | GON-FC | AVG-R1 (%) | AVG-mAP (%) |
|---------|--------|------------|-------------|
| ✗ | ✗ | 67.7 | 57.6 |
| ✓ | ✗ | 69.8 | 57.6 |
| ✗ | ✓ | 69.2 | 59.1 |
| **✓** | **✓** | **75.6** | **63.2** |

**说明**: GON-P3D 主要在帧级抑制衣物波动（提 R1），GON-FC 主要在序列级稳定身份线索（提 mAP），两者**互补**，组合后 R1/mAP 同时大幅跳升至 75.6/63.2。

### Table 5: Performance on CCPG and SUSTech1K / 真实数据集（外观法 P3D）

| Method | CCPG CL (R1/mAP) | CCPG MEAN (R1/mAP) | SUSTech1K NM | SUSTech1K Overall R1 | SUSTech1K Overall R5 |
|--------|------------------|--------------------|--------------|----------------------|----------------------|
| GaitSet [10] | 77.7 / 46.4 | 81.5 / 55.8 | 69.1 | 65.0 | 84.8 |
| GaitPart [21] | 77.8 / 45.5 | 81.9 / 56.2 | 62.2 | 59.2 | 80.8 |
| GaitBase [19] | 88.5 / — | 91.5 / — | 81.5 | 76.1 | 89.4 |
| SkeletonGait [20] | 52.4 / 20.8 | 63.5 / 32.3 | 67.9 | 63.0 | 83.5 |
| DeepGaitV2 [18] | 90.5 / 63.6 | 92.9 / 74.7 | 87.4 | 82.3 | 92.5 |
| **GaitCLIF (ours)** | **93.4 / 66.4** | **94.8 / 77.0** | **89.2** | **84.7** | **93.6** |

**说明**: GaitCLIF 在 CCPG 全部换衣条件均提升，ReID 协议下 MEAN 的 R1/mAP 较 DeepGaitV2 提升 **+1.9% / +2.3%**；SUSTech1K Overall R1/R5 提升 **+2.4% / +1.1%**。(CCPG 含 CL/UP/DN/MEAN 四档；SUSTech1K 含 NM/BG/CL/CA/UM/UN/OC/NG 多协变量，此处摘录代表列。)

### Table 6: Performance on Gait3D and GREW / 野外数据集（外观法 P3D）

| Method | Gait3D R1 | Gait3D R5 | Gait3D mAP | GREW R1 | GREW R5 |
|--------|-----------|-----------|------------|---------|---------|
| GaitSet [10] | 36.7 | 58.3 | 30.0 | 46.3 | 63.6 |
| GaitPart [21] | 28.2 | 47.6 | 21.6 | 44.0 | 60.7 |
| GaitBase [19] | 60.1 | — | — | 64.6 | — |
| GaitMoE [30] | 73.7 | — | 66.2 | 79.6 | 89.1 |
| DeepGaitV2 [18] | 74.4 | 88.0 | 65.8 | 77.7 | 88.9 |
| **GaitCLIF (ours)** | **76.5** | **88.5** | **67.9** | **80.2** | **89.2** |

**说明**: 野外数据集衣物变化有限，直接全量套用 GON 会造成类内过度发散，故**只用 GON-FC** 增强非线性映射。即便如此，GaitCLIF 在 Gait3D（76.5/67.9）与 GREW（80.2/89.2）上仍达 SOTA。

### Table 7: Cross-domain (BarbieGait Pretrain) on CCPG / 跨域预训练

| Model | CCPG CL (R1/mAP) | CCPG MEAN (R1/mAP) |
|-------|------------------|--------------------|
| DeepGaitV2 | 90.5 / 63.6 | 92.9 / 74.7 |
| GaitCLIF (Scratch) | 93.4 / 66.4 | 94.8 / 77.0 |
| **GaitCLIF (Pretrain)** | **93.9 / 69.8** | **95.8 / 79.5** |

**说明**: 用 BarbieGait 预训练再在 CCPG 小学习率微调，MEAN R1/mAP 由 Scratch 的 94.8/77.0 提升到 **95.8/79.5**，验证 BarbieGait 可作为真实步态识别的**通用预训练源**。

### Table 8: Upstream Pose Estimation Benefit on CCPG / 改进上游姿态估计

| Methods | CL (R1/mAP) | UP (R1/mAP) | DN (R1/mAP) | MEAN (R1/mAP) |
|---------|-------------|-------------|-------------|---------------|
| SkeletonGait [20] | 52.4 / 20.8 | 65.4 / 35.8 | 72.8 / 40.3 | 63.5 / 32.3 |
| DPGait [37] | 70.7 / — | 82.4 / — | 84.2 / — | 79.1 / — |
| **Ours (Barbie-17 + ViTPose)** | **76.9 / 40.0** | **84.9 / 56.7** | **87.2 / 57.7** | **83.0 / 51.5** |

**说明**: 用 BarbieGait 的"身份一致、服装多样、姿态精配"数据重训 [[ViTPose]]，CCPG 上 MEAN R1/mAP 达 83.0/51.5，较 SkeletonGait **+19.5/+19.2**、较 DPGait **R1 +3.9**。关键区别：DPGait 用大规模动作数据但不强制身份一致，而 BarbieGait 提供**跨衣的身份一致监督**，得到更判别、更鲁棒的姿态表征。

### Table 9: Additional Cloth-Changing Benchmarks / 更多换衣基准（附录）

| Methods | HybridGait | OU-ISIR | CCVID | MEVID |
|---------|-----------|---------|-------|-------|
| DeepGaitV2 [18] | 55.34 | 90.54 | 93.18 | 69.94 |
| **GaitCLIF (ours)** | **60.11** | **94.04** | **97.08** | **74.05** |

**说明**: 在步态识别（HybridGait、OU-ISIR）与视频 Re-ID（CCVID、MEVID）四个额外换衣基准上 GaitCLIF 全面领先，进一步证明跨衣泛化性。

### Table 10: Per-thickness Module Ablation / 各厚度档下逐模块消融（附录）

| GON-P3D | GON-FC | THK1 (R1/mAP) | THK5 (R1/mAP) | THK9 (R1/mAP) | AVG (R1/mAP) |
|---------|--------|---------------|---------------|---------------|--------------|
| ✗ | ✗ | 85.4 / 72.4 | 72.8 / 59.9 | 47.9 / 42.3 | 67.7 / 57.6 |
| ✓ | ✗ | 86.1 / 71.3 | 75.3 / 59.8 | 51.2 / 43.6 | 69.8 / 57.6 |
| ✗ | ✓ | 83.8 / 71.7 | 73.1 / 60.7 | 53.3 / 46.4 | 69.2 / 59.1 |
| **✓** | **✓** | **88.1 / 74.2** | **80.1 / 65.1** | **61.1 / 51.9** | **75.6 / 63.2** |

**说明**: Table 4 的逐厚度展开版。可见越厚的衣物（THK9）两模块组合带来的相对增益越大（R1 47.9→61.1），说明 GON 设计在**极端换衣**下尤其关键。

### Table 11: Normalization Comparison (BN/IN/LN vs GON) / 归一化方法对比（附录）

| Norm Type | THK1 (R1/mAP) | THK5 (R1/mAP) | THK9 (R1/mAP) | AVG (R1/mAP) |
|-----------|---------------|---------------|---------------|--------------|
| BN | 83.2 / 71.8 | 69.1 / 58.3 | 45.9 / 41.6 | 65.3 / 56.7 |
| IN | 75.4 / 67.2 | 61.0 / 55.5 | 38.0 / 37.5 | 57.6 / 53.1 |
| LN | 76.9 / 61.7 | 61.9 / 50.4 | 41.1 / 35.8 | 59.1 / 49.0 |
| **GON** | **88.1 / 74.2** | **80.1 / 65.1** | **61.1 / 51.9** | **75.6 / 63.2** |

**说明**: 直接验证 GON 设计动机——[[Instance Normalization|IN]]（域不变常用）在步态上最差（57.6），LN/BN 也不如 GON。GON 以 75.6/63.2 全面胜出，证明"为步态量身定制的跨通道分块归一化"有效。

### Table 12: Upstream Pose Benefit on CCPG & SUSTech1K / 上游姿态增益（附录，Barbie-17）

| Methods | Pose Type | Upstream | CCPG MEAN (R1/mAP) | SUSTech1K Overall R1 | SUSTech1K Overall R5 |
|---------|-----------|----------|--------------------|----------------------|----------------------|
| GaitTR [77] | COCO-17 | HRNet | 28.0 / 14.1 | 44.9 | 72.6 |
| GaitGraph [59] | COCO-17 | HRNet | 6.0 / 3.5 | 27.3 | 40.2 |
| SkeletonGait [20] | COCO-17 | HRNet | 63.5 / 32.3 | 62.8 | 72.6 |
| DPGait [37] | COCO-17 | ViTPose | 79.1 / — | — | — |
| **Ours** | **Barbie-17** | **ViTPose** | **83.0 / 51.5** | **69.8** | **80.4** |

**说明**: Table 8 的扩展版（含 SUSTech1K），并标注上游模型与姿态格式。用 BarbieGait 重训的 ViTPose（Barbie-17 格式）在两个真实数据集上均最优，凸显身份一致监督对**上游姿态估计**的迁移价值。

---

## 实验

### 数据集 / 基准

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| BarbieGait | 521 人（261 训/260 测），约 120 万序列，100 套衣/人 | **合成**、身份一致、THK0–THK9 厚度分级 | 训练/测试/预训练源 |
| CCPG | 200 人，7 种换衣 | 真实、换衣（CL/UP/DN）| 训练/测试 |
| SUSTech1K | 多协变量（NM/BG/CL/CA/UM/UN/OC/NG）| 真实、多变量 | 训练/测试 |
| Gait3D | 4,000 人 | 真实、野外 | 训练/测试 |
| GREW | 26,345 人 | 真实、野外、大规模 | 训练/测试 |

### 实现细节
- **采集**: 6 路同步相机采 521 人真实步态；HRNet 估 2D 姿态、triangulation + EasyMoCap 估 3D 姿态/网格。
- **生成**: MakeHuman 做体型/骨骼匹配与随机换衣，四元数运动学运动匹配，Blender 20 场景 × 8 相机渲染（聚焦人体+阴影，提速 5–6×）。
- **数据处理**: PaddleSeg 分割轮廓，HRNet 出 2D 姿态。
- **模型**: GaitCLIF-P3D / GaitCLIF-3D，视觉 4 stage（block 数见 Table 2）+ TP + HP + GON-FC head；野外数据集仅用 GON-FC。
- **训练配置**: 见 Table 2（如 BarbieGait batch (8,8)、60k 迭代）。
- **对齐质量**: 平均关节位置误差 **12.2 mm**（主要来自层级累积误差），关节角误差仅 **0.02°**，表明真实-虚拟空间对应高度精确（身份一致性的定量保证）。

### 关键实验结论
- **BarbieGait 主结果（Table 3）**: 理想轮廓下 DeepGaitV2-P3D 即达 R1 91.2%/mAP 83.4%，但**真实带噪轮廓骤降到 67.7%/57.6%**，凸显换衣场景的挑战；GaitCLIF-3D 推到 80.4/65.7，姿态法 GaitCLIF-P3D\* 达 78.1/73.3。
- **真实数据集（Table 5/6）**: CCPG、SUSTech1K、Gait3D、GREW 全面 SOTA。
- **跨域预训练（Table 7）**: BarbieGait 预训练 + CCPG 微调 → 95.8/79.5，优于从零训练。
- **上游姿态（Table 8/12）**: 重训 ViTPose 在 CCPG MEAN 达 83.0/51.5，远超基线。
- **消融（Table 4/10/11）**: GON-P3D + GON-FC 互补且在厚衣档增益最大；GON 全面优于 BN/IN/LN。

---

## 批判性思考

### 优点
1. **数据贡献新颖且规模大**: 首个"身份一致 + 100 套换衣/人"的合成步态数据集，用 3D 姿态/网格做静态+动态双重对齐，并以 12.2mm/0.02° 的对齐误差量化身份一致性，填补了真实换衣数据稀缺的空白。
2. **方法动机-证据闭环清晰**: "去衣物统计"由归一化对比（Table 11，IN/LN/BN < GON）支撑，"保运动细节"由分块设计与逐厚度消融（Table 10，厚衣增益最大）支撑，并有热图可视化（Figure 8）定性佐证。
3. **价值面广**: 不仅是基准，还验证了对下游识别（预训练）与上游姿态估计（重训 ViTPose）的双向迁移收益。

### 局限性
1. **合成-真实域差距**: 理想轮廓 91.2% vs 真实带噪轮廓 67.7% 的巨大落差说明，合成数据的"理想性"与真实部署间仍有显著 gap，GaitCLIF 在真实数据上的增益（CCPG +1.9~2.3）远小于在 BarbieGait 上的增益。
2. **GON 在野外数据需"半量"使用**: 在 Gait3D/GREW 上必须只用 GON-FC、否则类内过度发散，说明该归一化方案对"衣物变化是否充足"较敏感，普适性受限、缺少自适应机制。
3. **运动学匹配细节偏简略**: Algorithm 1 的四元数迁移与误差来源（层级累积）描述较概括，缺少对匹配失败案例、误差对识别影响的系统分析；衣物随机选取的真实分布合理性也未充分论证。

### 潜在改进方向
1. 引入显式的**合成-真实域自适应**（如风格迁移/对抗对齐），缩小理想轮廓与真实带噪轮廓的差距。
2. 把 GON 的"是否分块、分块数、是否全量启用"做成**数据驱动可学习/自适应**，避免野外场景的手工取舍。
3. 利用 BarbieGait 的可控性扩展到**跨视角、跨域**研究，并把身份一致监督迁移到更多上游任务（网格恢复、神经渲染）。

### 可复现性评估
- [x] 代码开源（https://github.com/BarbieGait/BarbieGait）
- [x] 数据集发布（声明仅发布无个人视觉信息的合成数据）
- [x] 训练细节较完整（Table 2 给出 batch/block/里程碑/迭代；上游姿态细节在补充材料）
- [x] 评测协议清晰（THK0 gallery / THK1–THK9 probe，R1 + mAP）

---

## 速查卡片

> [!summary] BarbieGait: Identity-Consistent Synthetic Cloth-Changing Gait Dataset + GaitCLIF
> - **核心**: 把真人用 3D 骨骼/体型 + 四元数运动学匹配映射进虚拟引擎，为每人合成 100 套换衣且保持步态身份（对齐误差 12.2mm/0.02°）。
> - **方法**: BarbieGait 五阶段生成流水线（匹配→换衣→运动匹配→建场景→渲染）；GaitCLIF 用 GON（水平分块跨通道归一化）去衣物统计 + GON-P3D/3D/FC 保运动细节。
> - **结果**: BarbieGait 上 GaitCLIF-3D 80.4/65.7（R1/mAP）；CCPG/SUSTech1K/Gait3D/GREW 全 SOTA；可作下游预训练源（CCPG 95.8/79.5）与改进上游 ViTPose（CCPG 83.0/51.5）。
> - **代码**: https://github.com/BarbieGait/BarbieGait

---

*笔记创建时间: 2026-06-29*
