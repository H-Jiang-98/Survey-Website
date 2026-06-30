---
title: "Learning Surgical Robotic Manipulation with 3D Spatial Priors"
method_name: "SST"
authors: [Yu Sheng, Lidian Wang, Xiaomeng Chu, Jiajun Deng, Min Cheng, Yanyong Zhang, Bei Hua, Houqiang Li, Jianmin Ji]
year: 2026
venue: CVPR
tags: [surgical-robotics, visuomotor-policy, 3D-spatial-prior, imitation-learning, MASt3R, endoscope]
zotero_collection: _inbox
image_source: online
arxiv_html: https://arxiv.org/html/2603.03798v1
created: 2026-06-29
---

# Learning Surgical Robotic Manipulation with 3D Spatial Priors

## 元信息

| 项目 | 内容 |
|------|------|
| 作者 | Yu Sheng, Lidian Wang, Xiaomeng Chu, Jiajun Deng, Min Cheng, Yanyong Zhang, Bei Hua, Houqiang Li, Jianmin Ji |
| 机构 | 中国科学技术大学（USTC）等（da Vinci/Torin 手术机器人实验环境） |
| 会议 | CVPR 2026 |
| 类别 | 手术机器人操作 / Visuomotor Policy / 3D 空间先验 |
| 日期 | 2026 |
| 项目主页 | （数据集与代码声明将开源） |
| 链接 | [arXiv](https://arxiv.org/abs/2603.03798) / Code: 待发布 |

---

## 一句话总结

> 直接从立体内窥镜图像中挖掘隐式 3D 先验：用在自建 Surgical3D 上微调的几何 Transformer 提取 3D latent，经轻量多层空间特征连接器（MSFC）对齐到内窥镜坐标系动作空间，端到端学习手术操作策略，在打结、离体器官切除等真机任务上达到 SOTA。

---

## 核心贡献

1. **Surgical3D 数据集**: 用 NVIDIA Omniverse 构建 **30K 张照片级真实感立体内窥镜图像对**，附带精确 3D 几何标注（深度图、点云、相机外参），填补手术场景缺乏 3D 重建数据的空白。
2. **Spatial Surgical Transformer（SST）**: 一个由学习到的 [[3D 空间先验]]驱动的端到端 [[Visuomotor Policy|视觉运动策略]]；以在 Surgical3D 上微调的[[几何 Transformer]]产出高质量 3D latent embedding，并用轻量连接器 **MSFC** 将多层空间特征对齐到机器人动作空间。
3. **真机验证**: 在 **Torin 手术机器人**上部署，跨三个真实手术场景（peg pickup、knot tying、离体胆囊切除）评测，取得 SOTA 成功率与强空间泛化，且**无需腕部相机**、贴合临床约束。

---

## 问题背景

### 要解决的问题
手术机器人需要对针、组织等极小且精细的结构做**毫米级精度**操作，因此 [[Visuomotor Policy|视觉运动策略]]必须具备 **3D 空间感知**能力。如何在仅有立体内窥镜、缺乏 3D 标注数据、且不能加装额外硬件的临床约束下，赋予策略可靠的 3D 空间感知，是核心难题。

### 现有方法的局限
作者指出两类既有范式各有硬伤：
1. **显式重建范式**: 先用基于优化的技术从立体 ECM（内窥镜相机臂）图像重建 3D 手术场景，再在重建结果上学习/执行技能。这种**多阶段**流程会**累积重建误差**，且无法端到端优化。
2. **腕部相机范式**: 在 PSM（患者侧机械臂）上加装腕部相机补充内窥镜视角。但：(a) 没有 3D 监督或几何先验时，多视角特征**捕捉不到有意义的几何线索**；(b) 更关键的是 **trocar（套管）对 PSM 插入路径有严格空间约束**，带相机的器械根本无法通过，临床上几乎不可用；(c) 液体（水/血）还会损坏腕部相机。

### 本文的动机
- 前馈式几何模型（[[MASt3R]]、[[VGGT]] 等）能快速 3D 重建，其 **latent embedding 蕴含丰富一致的几何信息**——直接利用这些隐式嵌入做 3D 先验，可同时规避显式重建的低效与额外传感器的硬件约束。
- 但直接套用仍非平凡：(1) 当前无公开的带 3D 标注的大规模手术数据集，现成重建模型几乎没在手术场景训练过，存在严重 domain gap；(2) 把高容量预训练编码器**朴素地塞进**视觉运动策略往往因表征与任务目标不对齐而**反而掉点**。
- 因此本文用 **Surgical3D 微调几何 Transformer**弥合 domain gap，用 **MSFC** 做有原则的特征对齐，并配合**内窥镜中心动作空间**保证感知与动作坐标系一致。

---

## 方法详解

### 模型架构

SST 的目标是学习一个由 $\theta$ 参数化的策略 $\pi_\theta$，把当前观测 $o_t$ 与本体状态 $x_t$ 映射到下一步动作 $a_t$，训练数据为 $n$ 条示范轨迹 $\mathcal D=\{\tau_1,\dots,\tau_n\}$，每条 $\tau_i=\{(o_1,x_1,a_1),\dots,(o_T,x_T,a_T)\}$。整体由三部分组成（见 Figure 2）：

- **输入**: 立体内窥镜观测 $o_t$ + 本体状态 $x_t$
- **几何 Transformer**（§3.2）: 在 Surgical3D 上以 3D 重建目标微调，从立体内窥镜观测提取稳健的 3D latent embedding
- **核心模块**: [[MSFC|多层空间特征连接器]]（§3.3）把多层 3D latent 按任务需求对齐到动作空间
- **策略解码器**（§3.4）: [[内窥镜中心坐标系]]下预测机器人**相对动作**
- **输出**: [[Action Chunking|动作块]] $a_{t:t+k}$（每臂 7 维）

### 核心模块

#### 模块1: Surgical3D 数据集构建

**设计动机**: 手术环境中器官-相机距离常 <10cm，超出多数 3D 传感器量程，导致带 3D 标注的手术数据极度稀缺。

**具体实现**:
- 用 NVIDIA Omniverse 合成数据，做 **domain randomization**：变化立体内窥镜基线、相机内外参、光照、组织纹理。
- 整合两类 3D 资产：(1) 开源人体全身器官模型（8 类）与手术器械（多样但解剖真实感有限）；(2) **用 iPad 扫描真实器官得到的 10 个真实感 3D mesh**。
- 共生成 **30K 张高分辨率（1920×1080）立体图像对**及对应深度图。
- **缝合 sim-to-real gap**: 仅用合成数据训练对真实场景泛化差（见 Figure 3）。于是对未标注真实数据（如真实手术录像）生成**伪标签**——先在合成数据上微调 [[VGGT]] 的点预测头，再用精炼后的模型推断真实数据深度，**只保留高置信度区域**。混合数据集显著提升几何 Transformer 在合成与真实两域的鲁棒性与泛化。

#### 模块2: Surgical Geometry Transformer（手术几何 Transformer）

**设计动机**: 手术影像独特——器官表面常**无纹理或高度重复**，传统特征匹配不可靠；内窥镜双目**基线极窄**，微小像素错位即引入显著深度误差。需要不依赖相机参数或特征匹配的前馈重建。

**具体实现**:
- 以 [[MASt3R]] 为原型（ViT-Large，patch=16，载入 MASt3R 预训练权重），从成对图像直接推断稠密 3D 点，且**复用其互联网规模预训练**获得稳健 3D latent。
- 之所以不选 [[VGGT]]：VGGT 几何更细但**架构过重**，难以实时部署且会引入运动抖动；MASt3R 更轻量，在 Surgical3D 微调后足以处理手术原始立体图。
- 结构纯 ViT：立体图先 patchify、flatten 成 token，经**共享编码器**跨视角处理；再送入 ViT 解码器，用 **cross-attention** 捕捉空间关系并跨视角聚合。
- 微调阶段：解码器 token 经 [[DPT]] head 回归内窥镜坐标系下的稠密 3D 点图，用回归损失（公式1）+ **置信度感知**目标（公式2）训练。

#### 模块3: Multi-Level Spatial Feature Connector（MSFC，多层空间特征连接器）

**设计动机**: 直接用显式 3D 点图作策略输入，会因重建不完美与尺度歧义引入大误差；而朴素地用强预训练视觉编码器替换原图像编码器，收益有限甚至掉点。需要一种**有原则的几何先验-动作对齐策略**。

**具体实现**:
- 动机源自 [[DPT]] 的观察：**不同 Transformer 层捕获不同抽象层级**——低层编码细粒度局部细节，高层捕获全局上下文；手术操作恰需同时推理精确物体位置与整体运动方向。
- 取几何 Transformer **四个解码器层**的 latent embedding，先投影到低维以压缩；
- 沿特征维**拼接（concatenate）**不同层嵌入，用轻量 MLP 对齐到动作空间；
- 对齐后的 latent 与位置嵌入做 **cross-attention** 生成机器人动作。
- 关键：**选取的四层正是 in-domain 微调时喂给 DPT head 的层**——它们负责预测精确点图，几何线索最丰富。
- 消融（Table 3）证实 MSFC 明显优于 Last-Layer Feature Connector（LFC）与 Multi-Layer Separate Connector（MSC）。

#### 模块4: Endoscope-Centric Policy Decoder（内窥镜中心策略解码器）

**设计动机**: §3.2 学到的 3D latent 定义在内窥镜坐标系，因此动作空间也变换到同一坐标系，使策略完全在统一的内窥镜中心表征下运作。

**具体实现**:
- **动作空间**: 手术机器人与通用机器人的关键差异在于**缺乏精确正运动学**——PSM 的 Set-Up Joints（SUJ）仅靠电位计测关节，本质不精确。故不可用绝对关节状态/末端位姿，采用**相对位姿表示**（公式3）。旋转差用 **Euler 角**表示（实验中比旋转矩阵更易学习）；夹爪用**绝对张角**（与正运动学无关）。每臂 7 维：平移[3] + 旋转[3] + 夹爪角[1]，全部在内窥镜坐标系下表示。
- **策略解码器**: $L$ 个 Transformer block，每块含 self-attention + cross-attention，融合 3D latent 与 action token。输入为固定数量的**可学习位置嵌入**；3D latent 的位置编码由 reshape 成图像状张量、把两幅立体图沿宽度拼接、再加 2D 固定位置编码得到。
- **平滑动作**: 直接预测下一动作会抖；采用 [[ACT|Action Chunk Transformer]] 预测未来 $k$ 个动作，执行动作为加权平均，权重 $w_i=\exp(-m\cdot i)$（指数衰减，$m$ 控制后续动作影响）。端到端用 MSE 训练（公式4）。

### 关键公式与机制

#### 公式1: [[几何 Transformer]] 3D 点图回归损失

$$
L_{reg}(v,i)=\sum_{v=\{1,2\}}\sum_{i\in D^{v}}\left\|\frac{1}{z}X^{v,1}_{i}-\frac{1}{\hat{z}}\hat{x}^{v,1}_{i}\right\|
$$

**含义**: 在内窥镜坐标系下，把预测点图与 GT 点图各自做尺度归一化后回归，消除预测与真值之间的**尺度歧义**。

**符号说明**:
- $v\in\{1,2\}$: 立体两个视角；$i\in D^{v}$: 视角 $v$ 中有效像素，$D^{v}\in\{1\dots W\}\times\{1\dots H\}$
- $X^{v,1}_i$: GT 点图；$\hat{x}^{v,1}_i$: 预测点图
- $z,\hat z$: 分别用于归一化 GT 与预测点图的尺度因子

#### 公式2: 置信度感知训练目标

$$
L_{conf}=\sum_{v=\{1,2\}}\sum_{i\in D^{v}}C^{v,1}_{i}\,L_{reg}(v,i)-\alpha\log C^{v,1}
$$

**含义**: 针对手术场景常见的**少纹理/难回归区域**，引入逐像素置信度 $C^{v,1}_i$ 加权回归损失，并用 $-\alpha\log C$ 项防止置信度坍塌，让模型自适应地降低不可靠区域的权重。

**符号说明**:
- $C^{v,1}_i$: 视角 $v$ 像素 $i$ 的预测置信度
- $\alpha$: 置信度正则项权重

#### 公式3: 相对动作表示（内窥镜坐标系）

$$
\begin{aligned}
a_{t}&=\{E^{i}_{t+1}\ominus E^{i}_{t}\}\\
&=\{(tr^{i}_{t+1}-tr^{i}_{t},\,(R^{i}_{t})^{T}R^{i}_{t+1})\},\quad i\in\{left,right\}
\end{aligned}
$$

**含义**: 因缺乏精确正运动学，不学绝对位姿，而学**相邻帧末端位姿之差**：平移取向量差，旋转取相对旋转 $(R^i_t)^T R^i_{t+1}$（后转 Euler 角）。

**符号说明**:
- $E^i=(R^i,tr^i)\in SE(3)$: 第 $i$ 臂末端位姿；$R^i\in SO(3)$ 旋转矩阵，$tr^i\in\mathbb R^3$ 平移
- $\ominus$: $SE(3)$ 上的相对位姿运算；$i\in\{left,right\}$ 左右两臂

#### 公式4: 策略训练损失（动作块 MSE）

$$
L_{MSE}=MSE\big(\hat{a}_{t},\,\pi_{\theta}(o_{t},x_{t})\big)
$$

**含义**: 端到端最小化预测动作与 GT 动作的均方误差；结合 ACT 框架预测未来 $k$ 个动作并以指数权重 $w_i=\exp(-m\cdot i)$ 做加权平均执行，保证轨迹平滑稳定。

**符号说明**:
- $\hat a_t$: GT 动作；$\pi_\theta(o_t,x_t)$: 策略预测；$o_t,x_t$: 观测与本体状态

---

## 关键图表

<!-- 图片均使用 arXiv HTML 在线链接，未本地化 -->

### Figure 1: SST Overview / 方法概览与真机平台

![Figure 1](https://arxiv.org/html/2603.03798v1/x1.png)

**说明**: SST 总览。左侧示意 trocar 对 PSM 插入路径的空间约束（这正是腕部相机不可行的根因）；中间为方法核心——几何 Transformer 在 Surgical3D 上微调，从立体内窥镜输入提取稳健 3D latent，MSFC 把捕获细粒度细节与全局上下文的多层 3D latent 整合进策略解码器；右侧展示部署于带立体 ECM + 双 PSM 的真实手术机器人，跨三个真实手术场景达成 SOTA。

### Figure 2: Pipeline of Our Method / 方法流水线

![Figure 2](https://arxiv.org/html/2603.03798v1/x2.png)

**说明**: 两阶段流程。**上**：几何 Transformer 先在 Surgical3D 上用 3D 重建目标微调，学会从内窥镜图提取稳健 3D latent。**下**：冻结几何 Transformer，仅训练其余组件从示范学策略；MSFC 聚合来自几何 Transformer 多个 block 的 3D latent 并对齐到动作空间；内窥镜中心策略解码器在 3D 先验引导下生成内窥镜坐标系的相对动作。这张图阐明了"先学几何先验、再冻结接策略"的核心设计。

### Figure 3: Surgical3D Samples & MASt3R Finetuning / 数据样本与微调消融

![Figure 3](https://arxiv.org/html/2603.03798v1/x3.png)

**说明**: 左为 Surgical3D 样本（多样 3D 手术资产生成的高真实感合成场景）。右为 MASt3R 在三种微调配置下的重建对比：(a) 真实在体手术立体图输入；(b) 原始 MASt3R **同时重建不出 PSM 与器官表面**；(c) 仅用合成数据微调，器官重建粗糙、PSM 几何仍不完整；(d) 用合成+真实混合数据微调，PSM 器械与器官完整度都明显更准。直接佐证 sim-to-real 伪标签策略的必要性。

### Figure 4: Visualization of Experimental Settings / 三个真机任务设置

![Figure 4](https://arxiv.org/html/2603.03798v1/x4.png)

**说明**: 黄/蓝箭头分别表示右/左臂的大致运动方向。**上**：Peg pickup，采集 180 条轨迹（约 120 条绿区 + 60 条蓝区），Test1/Test2 对应这两区评测——用于检验**空间泛化**。**中**：Knot tying，缝线尾随机置于蓝区，(b1)(b3) 为抓取与绕环动作。**下**：离体胆囊切除，抓取点在蓝区随机采样，(c1)(c3) 为抓取与切割动作。

### Figure 5: Qualitative Results of Intermediate 3D Reconstruction / 中间 3D 重建定性结果

![Figure 5](https://arxiv.org/html/2603.03798v1/x5.png)

**说明**: 图像左下角下标表示操作步。把微调后几何 Transformer 的 token 送入 DPT head 生成中间 3D 重建。结果显示几何 Transformer 在各测试任务上**无需任务特定训练**即可从当前内窥镜观测有效提取 3D 线索——证明所学 3D 先验是通用且可迁移的。

### Figure 6: Alternative Designs of Spatial Connectors / 连接器对比设计

![Figure 6](https://arxiv.org/html/2603.03798v1/x6.png)

**说明**: 两种被比较的连接器变体：(a) Last-Layer Feature Connector（LFC，仅用末层特征）；(b) Multi-Layer Separate Connector（MSC，多层但分离处理）。与本文 MSFC（多层拼接对齐）对照，支撑 Table 3 的消融结论。

### Table 1: Success Rates on Three Surgical Tasks / 三任务成功率主表

| Methods | Setting | Peg-Test1 | Peg-Test2 | KT-Grasp | KT-Loop | KT-Whole | GD-Grasp | GD-Dissect | GD-Whole |
|---------|---------|-----------|-----------|----------|---------|----------|----------|------------|----------|
| SRT | w/. wrist cams | 10/10 | 6/10 | 10/10 | 3/10 | 2/10 | - | - | - |
| ACT | w/o. wrist cams | 9/10 | 2/10 | 4/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 |
| DP | w/o. wrist cams | 10/10 | 1/10 | 5/10 | 1/10 | 1/10 | 0/10 | 0/10 | 0/10 |
| **SST (Ours)** | **w/o. wrist cams** | **10/10** | **8/10** | **10/10** | **7/10** | **7/10** | **10/10** | **6/10** | **6/10** |

**说明**: KT=Knot Tying，GD=Gallbladder Dissection。SST **不用腕部相机**即全面领先：在需要空间泛化的 Peg-Test2 上 8/10（远超 ACT 2/10、DP 1/10，甚至超过用腕部相机的 SRT 6/10）；打结整体任务 7/10、胆囊切除整体 6/10，而 ACT/DP 在这两项几乎全 0。SRT 因腕部相机会被液体损坏而不参与胆囊切除。

### Table 2: Effectiveness of Finetuning on Surgical3D / 几何 Transformer 微调消融

| Config | Peg-test1 | Peg-test2 | KT-Grasp | KT-Loop | KT-Whole |
|--------|-----------|-----------|----------|---------|----------|
| w/o ToS | 2/10 | 0/10 | 0/10 | 0/10 | 0/10 |
| **w/ ToS (Ours)** | **10/10** | **8/10** | **10/10** | **7/10** | **7/10** |

**说明**: ToS = 几何 Transformer 在 Surgical3D 上训练。不在 Surgical3D 微调时性能几乎崩溃（Peg-test1 仅 2/10、其余全 0），微调后大幅跃升——证明**弥合 domain gap 的 in-domain 微调是 3D 先验可用的前提**。

### Table 3: Effectiveness of MSFC / 连接器设计消融

| Config | Peg-test1 | Peg-test2 | KT-Grasp | KT-Loop | KT-Whole |
|--------|-----------|-----------|----------|---------|----------|
| LFC | 1/10 | 0/10 | 0/10 | 0/10 | 0/10 |
| MSC | 10/10 | 3/10 | 5/10 | 0/10 | 0/10 |
| **MSFC (Ours)** | **10/10** | **8/10** | **10/10** | **7/10** | **7/10** |

**说明**: LFC=仅末层、MSC=多层分离、MSFC=多层拼接对齐。LFC 几乎不可用；MSC 虽简单任务可用但在泛化（Peg-test2 3/10）与精细子任务（KT-Loop 0/10）上失败；MSFC 全面最优，证明**多层特征拼接对齐**对兼顾局部细节与全局上下文至关重要。

---

## 实验

### 数据集 / 任务

| 任务 | 轨迹数 | 环境 | 评测协议 |
|------|--------|------|----------|
| Peg pickup | 180（约 120 绿区 + 60 蓝区） | 模拟人体腹腔的穹顶内 | Test1/Test2 分区评测，抓起并举升 peg ≥3s 为成功 |
| Knot tying | 200 | 穹顶内 | 分 Grasp（双臂抓缝线）与 Loop（右臂绕左臂一圈）两子任务 |
| Ex-vivo 胆囊切除 | 200 | 开放环境（离体真实器官） | 先抓取并左拉胆囊，再将右臂定位到胆囊-肝交界做切割动作（仅复现切割、不真切组织） |

合成数据集 **Surgical3D**：30K 立体图像对（1920×1080）+ 深度/点云/外参，用于训练几何 Transformer。

### 实现细节

- **几何 Transformer**: ViT-Large，patch=16，载入 [[MASt3R]] 预训练权重；训练阶段经 DPT head 回归点图，部署时冻结。
- **策略解码器**: 12 层 Transformer decoder，hidden=768。
- **基线**: SRT、ACT 保留原架构（4 层 encoder + 7 层 decoder，hidden=768），动作 chunk=100，权重方案 $m=0.1$；Diffusion Policy 用 [7] 默认配置。**仅 SRT 加腕部相机**（紧凑设计以贴近其原系统），其余方法仅用内窥镜图。
- **训练**: 全部训 100 epoch，用最终 checkpoint 评测；每任务 10 次独立试验。
- **平台**: Torin 手术机器人（立体 ECM + 双 PSM）。

### 关键实验结论

- **主结果（Table 1）**: SST 不用腕部相机即在三任务全面 SOTA，尤其在需空间泛化的 Peg-Test2 与精细打结/切除子任务上大幅领先。
- **几何微调消融（Table 2）**: 不在 Surgical3D 微调几何 Transformer，性能近乎归零——in-domain 微调是关键前提。
- **连接器消融（Table 3）**: MSFC > MSC > LFC，多层拼接对齐显著优于单层或分离设计。
- **中间重建（Figure 5）**: 几何 Transformer 无需任务特定训练即能从内窥镜观测提取稳定 3D 线索，说明 3D 先验通用可迁移。

---

## 批判性思考

### 优点
1. **直击临床痛点**: 抛弃临床不可行的腕部相机，仅凭标配立体内窥镜挖掘**隐式 3D 先验**，并以内窥镜中心相对动作绕开手术机器人**无精确正运动学**的固有难题，工程与临床落地性强。
2. **设计-证据闭环清晰**: 两个核心主张（in-domain 微调、MSFC 多层对齐）都有干净的消融（Table 2/3）支撑，且 Figure 3/5 的定性重建从感知侧佐证 3D 先验确实学到位。
3. **补齐数据短板**: Surgical3D（30K，含 iPad 扫描真实器官 mesh + sim-to-real 伪标签）填补手术 3D 标注数据空白，并承诺开源数据与代码。

### 局限性
1. **评测规模偏小**: 每任务仅 10 次试验、3 个任务，统计置信度有限；离体胆囊切除还只"复现切割动作而不真切组织"，与真实临床仍有距离。
2. **基线与公平性存疑**: 唯一带腕部相机的 SRT 在胆囊切除中被排除，主表对比口径不完全一致；ACT/DP 在打结/切除上近乎全 0，差距过大也可能反映基线调参不足。
3. **依赖合成数据质量**: 性能高度依赖 Surgical3D 的真实感与伪标签可靠性；domain gap 的弥合主要靠"高置信度区域保留"这一启发式，缺乏对伪标签噪声影响的定量分析。
4. **关键超参偏经验**: "取四个解码器层""Euler 角优于旋转矩阵""$m=0.1$"等多为经验选择，缺系统敏感度分析。

### 潜在改进方向
1. 扩大任务与试验规模、引入真实切割/缝合的完整闭环评测，并补充更多强基线与统一对比口径。
2. 对 MSFC 选层、动作表示、置信度阈值做系统消融或自动搜索，降低手工经验依赖。
3. 量化伪标签噪声与合成-真实分布差异对策略的影响（如表征探针、重建误差与成功率的相关性分析）。

### 可复现性评估
- [x] 代码开源（声明将发布）
- [x] 预训练模型（基于公开 MASt3R 权重；自训权重声明发布）
- [x] 训练细节完整（架构、chunk、epoch、超参均给出，更多细节在补充材料）
- [x] 数据集可获取（Surgical3D 声明开源；真机任务为自建无公开 benchmark）

---

## 速查卡片

> [!summary] SST: Learning Surgical Robotic Manipulation with 3D Spatial Priors
> - **核心**: 从立体内窥镜图挖隐式 3D 先验，端到端学手术操作策略，无需临床不可行的腕部相机。
> - **方法**: Surgical3D（30K 合成+伪标签）微调 MASt3R 几何 Transformer → MSFC 多层 3D latent 拼接对齐 → 内窥镜中心相对动作 + ACT 解码器，两阶段（冻结几何 Transformer 训策略）。
> - **结果**: Torin 真机三任务（peg pickup / knot tying / 离体胆囊切除）SOTA；无腕部相机即超带腕部相机的 SRT；消融证实 in-domain 微调与 MSFC 均关键。
> - **代码/数据**: 声明将开源。

---

*笔记创建时间: 2026-06-29*
