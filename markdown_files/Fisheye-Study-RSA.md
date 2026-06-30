---
title: "Rethinking Camera Choice: An Empirical Study on Fisheye Camera Properties in Robotic Manipulation"
method_name: "Fisheye-Study-RSA"
authors: [Han Xue, Nan Min, Xiaotong Liu, Wendi Chen, Yuan Fang, Jun Lv, Cewu Lu, Chuan Wen]
year: 2026
venue: CVPR
tags: [fisheye-camera, imitation-learning, robotic-manipulation, data-augmentation, scene-generalization, hardware-generalization, diffusion-policy]
zotero_collection: _inbox
image_source: online
arxiv_html: https://arxiv.org/abs/2603.02139
created: 2026-06-29
---

# Rethinking Camera Choice: An Empirical Study on Fisheye Camera Properties in Robotic Manipulation

## 元信息

| 项目 | 内容 |
|------|------|
| 作者 | Han Xue, Nan Min, Xiaotong Liu, Wendi Chen, Yuan Fang, Jun Lv, Cewu Lu, Chuan Wen（前三人同等贡献，Cewu Lu / Chuan Wen 通讯） |
| 机构 | 上海交通大学、东南大学、中国科学技术大学(USTC)、上海创智学院(Shanghai Innovation Institute)、Noematrix Ltd. |
| 会议 | CVPR 2026 |
| 类别 | 鱼眼相机 / 模仿学习 / 机器人操作 / 数据增强 |
| 日期 | 2026-03（arXiv v1） |
| 项目主页 | https://robo-fisheye.github.io/ |
| 链接 | [arXiv](https://arxiv.org/abs/2603.02139) / [PDF](https://arxiv.org/pdf/2603.02139) |

> 说明：arXiv HTML 版本不可用（"No HTML"），本笔记基于 **PDF 全文**精读，图片采用**项目主页在线链接**（`https://robo-fisheye.github.io/static/images/`）。

---

## 一句话总结

> 首个系统研究腕部鱼眼相机对模仿学习影响的实证工作：宽 FoV 的空间定位增益依赖场景视觉复杂度、鱼眼策略需场景多样性才能解锁强泛化，并指出跨相机失败的根因是"尺度过拟合"、提出简单的随机尺度增强(RSA)加以缓解。

---

## 核心贡献

1. **首个鱼眼相机操作学习系统实证研究**：在仿真（MuJoCo + Robomimic / MimicGen）与真实世界两端，围绕三大研究问题（[[Spatial Localization|空间定位]]、[[Scene Generalization|场景泛化]]、[[Hardware Generalization|硬件泛化]]）对比腕部鱼眼(235°/180°)与针孔(90°/60°)相机，得到可操作的数据采集指南。
2. **揭示"宽 FoV 增益的环境依赖性"**：宽 FoV 确实捕获更多背景静态特征、增强空间定位，但该收益**严重依赖场景视觉复杂度**——在纯色/特征贫乏背景中优势消失；同时鱼眼策略在简单场景易过拟合，但只要训练场景足够多样，其**场景泛化能力显著超过针孔**。
3. **诊断跨相机失败 + 提出 RSA**：将朴素跨鱼眼镜头迁移的灾难性失败根因定位为 **[[Scale Overfitting|尺度过拟合]]**（策略记忆物体的绝对像素尺度），提出 **[[Random Scale Augmentation|随机尺度增强 (RSA)]]**——训练时对每张图随机采样尺度因子做中心裁剪/缩放，迫使网络学习相对尺度关系，在仿真与真机零样本跨镜头迁移上大幅恢复性能。

---

## 问题背景

### 要解决的问题
鱼眼相机因极宽 FoV（常 >180°）在机器人操作（如 [[UMI]]、[[RDT2]]、[[GEN-0]]、$\pi_0$）中被快速采用，并被用于构建大规模数据集训练 [[VLA]]。但**这种采用速度远超对其下游影响的系统理解**：腕部鱼眼相机对策略学习究竟有什么具体的收益与风险，此前缺乏严格的实证分析。

本文聚焦鱼眼镜头的两个本质属性及其对策略学习的直接影响：
- **(1) 宽 FoV 的收益**：巨大视场带来更丰富的上下文，如何改善策略能力？
- **(2) 畸变的挑战**：宽 FoV 由严重径向畸变换来（针孔无此特性），跨不同硬件（不同内参）泛化困难。

由此形成三个研究问题：
1. **RQ1 空间定位**：宽 FoV 在多大程度上增强策略的空间推理与定位能力？
2. **RQ2 场景泛化**：鱼眼能否提升策略对新颖/干扰背景的鲁棒性与泛化？
3. **RQ3 硬件泛化**：在一种鱼眼镜头上训练的策略，迁移到内参不同的新镜头表现如何？

### 现有方法的局限
- 现有鱼眼数据集（自动驾驶、SLAM、监控、航拍）**缺乏机器人操作任务**；流行机器人 benchmark（Robomimic / RLBench / ManiSkill / DexMimicGen）**不含鱼眼流**——量化研究鱼眼光学属性对策略学习影响的工作缺失。
- 标准机器人仿真器**缺乏鱼眼渲染**，限制了鱼眼策略的 benchmark 化研究。
- 跨相机/跨硬件泛化是非平凡挑战，但缺乏对失败机理的诊断与对策。

### 本文的动机
- 用 **OmniCV-Lib** 思路在 MuJoCo 中实现稳定/精确/高效的两阶段鱼眼渲染管线，填补仿真空白；
- 设计严格对照实验（鱼眼 vs 针孔为唯一变量），系统隔离并量化鱼眼属性的影响；
- 针对跨相机失败，提出**数据中心**的解决方案 RSA，证明问题"可解、非不可逾越"。

---

## 方法详解

本文是**实证研究**而非单一模型，其方法学贡献由三部分组成：(A) MuJoCo 鱼眼仿真管线、(B) 受控的模仿学习框架、(C) 提出的 RSA 增强。

### 模型架构 / 实验框架

研究统一搭建在 **state-free（无本体输入）** 的 [[Diffusion Policy|扩散策略]] 之上，以纯视觉直接检验 FoV 的定位优势：
- **输入**：仅视觉。腕部相机 RGB（无第三视角，对齐 UMI/RDT2/GEN-0 设定）。
- **核心算法**：[[Diffusion Policy]]（[[U-Net]] 噪声预测网络 + [[DDIM]] 调度器）。
- **视觉编码器**：仿真用 [[ResNet-18]]（无预训练，保证与现有 benchmark 可比）；真实世界用 [[CLIP]] [[ViT]]-B/16（增强对新纹理/光照的鲁棒性）。
- **无本体输入 (No Proprioception)**：刻意去除 end-effector pose / 关节状态，逼迫模型完全依赖视觉做空间定位，从而**纯净地隔离鱼眼 FoV 的定位贡献**（引用 [58] 表明去本体可抑制对简单状态向量的过拟合）。
- **动作空间**：仿真用 Robomimic 的 **delta action（相对变换）**；真机用 UMI 的 **relative action**，相对动作空间相比绝对动作更利于空间泛化。
- **输出**：[[Action Chunking|动作块]]（预测视界 $T_{pred}=16$，执行视界 $T_{act}=8$）。

### 核心模块

#### 模块1: MuJoCo 鱼眼相机仿真（两阶段投影管线）

**设计动机**：标准仿真器无原生鱼眼渲染。3D Gaussian Splatting / 扩散模型可合成鱼眼但太慢、资源密集，不适合交互式仿真；经典投影模型（OmniCV-Lib）更稳定高效。

**具体实现**（见 Figure 2）：
- 在场景中沿六个主方向（前/后/左/右/上/下）放置六个虚拟相机，将六张图拼成 **cubemap（立方体贴图）**；
- **第一阶段**：将 cubemap 六面投影并缝合为 **equirectangular（等距柱状）全景图**，把球面视图"展开"到 2D 平面；
- **第二阶段**：对等距柱状图施加**特定鱼眼投影模型**，得到最终鱼眼图，可模拟不同镜头特性。
- 整体借鉴 OmniCV-Lib 实现全向相机功能。

#### 模块2: 受控对照实验设计（四类核心因子）

**设计动机**：将"鱼眼 vs 针孔"作为唯一主变量，系统隔离各因素影响（见 Figure 1）。

**四类核心因子（自变量）**：
- **Camera Model（相机模型）**：鱼眼 vs 针孔，主对比维度；聚焦腕部安装（近距观测放大两者差异）。
- **Scene Complexity（场景复杂度）**：feature-poor（纯色）vs feature-rich（纹理丰富），用于 RQ1 空间定位。
- **Scene Diversity（场景多样性）**：训练场景从单一背景扩展到 $N$ 个不同场景，用于 RQ2 场景泛化（零样本测试 unseen scenes）。
- **Camera Parameters（相机参数）**：变化 FoV/畸变内参，用于 RQ3 硬件泛化（零样本部署到 unseen 内参）。

#### 模块3: Random Scale Augmentation (RSA) — 本文提出的核心方法

**设计动机**：不同镜头内参会使**同一物体投影成不同尺度**；标准固定尺度裁剪（如 scale=0.95）会让策略**过拟合绝对像素尺度**，换镜头后绝对尺度先验崩溃 → 灾难性失败（见 Figure 9）。RSA 直接迫使策略对尺度变化鲁棒，学习**相对空间关系**（物体相对末端执行器的尺度），这是跨相机更可泛化的线索。

**具体实现**（见 Figure 3）：
- 不再用固定裁剪比例，而是对**每张训练图**从宽均匀分布采样随机尺度因子 $s\sim\mathcal{U}(0.7,1.3)$（如 RSA 训练区间用 $0.6$–$1.4$）；
- 按 $s$ 对图像做中心裁剪后缩放回网络标准输入尺寸；
- 当 $s>1.0$ 时构成"zoom-out"：源图被缩小、周边画布用黑色填充；
- 这阻止网络记忆绝对像素尺度，转而学习"目标相对夹爪的尺度"这一跨相机一致的相对线索。

### 关键公式与机制

本文以实证为主，关键的可形式化机制为 RSA 的尺度采样与归一化评分。

#### 公式1: [[Random Scale Augmentation|RSA 尺度采样]]

$$
s \sim \mathcal{U}(0.7,\ 1.3)
$$

**含义**：对每张训练图独立采样一个尺度因子 $s$，再做中心裁剪/缩放，使策略对尺度（即等效 FoV/焦距变化）保持不变。

**符号说明**：
- $s$：尺度因子；$s<1$ 等效"zoom-in"（物体变大），$s>1$ 等效"zoom-out"（物体变小、周边黑边填充）；
- $\mathcal{U}(\cdot,\cdot)$：均匀分布（论文也提到 RSA 训练用 $0.6$–$1.4$ 的更宽区间）；$s=1.0$ 对应训练镜头原尺度。

#### 公式2: [[Normalized Score|真机归一化评分]]

$$
\text{Normalized Score} = \frac{\text{Total points earned}}{\text{Total number of stages}}
$$

**含义**：真机复杂操作中二元成功率过于稀疏，难以刻画策略细微能力差异；本文遵循 [18] 将每个任务拆成若干关键阶段（通常 2–3），每完成一个阶段记一分，归一化为 $[0,1]$ 评分，提供更细粒度的能力信号。

**符号说明**：
- Total points earned：单次 rollout 完成的阶段累计得分（各阶段有不同分值，如 Pick Cup 的 0/0.5/1.0）；
- Total number of stages：该任务的总阶段数；真机每任务做 $N=20$ 次试验并报告累计/平均评分。

**机制要点（无公式但关键）**：
- **空间定位机制**：宽 FoV 捕获更多静态背景特征点作为"视觉锚点"，因此其定位增益**依赖背景视觉丰富度**（特征贫乏背景 → 无锚点 → 增益消失）。
- **场景泛化机制**：腕部相机运动天然引入背景位移，可视为隐式数据增强；鱼眼宽 FoV **放大**该效应、引入更强的鱼眼畸变增强，从而在足够多样的训练场景下解锁更强泛化（"宽 FoV ≈ 隐式数据增强"）。
- **硬件泛化机制（尺度-深度歧义）**：换镜头改变物体绝对像素尺度，策略误判深度（看大→以为更近→抓浅；看小→以为更远→抓深），导致 Figure 9/S10 中"move closer/move away"失败；RSA 通过尺度不变性打破这一依赖。

---

## 关键图表

<!-- 图片均使用项目主页在线链接，未本地化；arXiv HTML 不可用故采用项目主页图 -->

### Figure 1: Overview of Four Core Factors / 四大核心因子与研究问题总览

![Figure 1](https://robo-fisheye.github.io/static/images/figure1.png)

**说明**：研究框架总览。(a) **Camera Model**（鱼眼 vs 针孔）为主对比；(b) **Scene Complexity**（poor vs rich）对应 RQ1 空间定位；(c) **Scene Diversity**（1 vs N 个场景）对应 RQ2 场景泛化；(d) **Camera Parameters**（变内参）对应 RQ3 硬件泛化。这张图定义了全文的自变量与实验骨架。

### Figure 2: Fisheye Simulation Pipeline in MuJoCo / 鱼眼仿真两阶段管线

![Figure 2](https://robo-fisheye.github.io/static/images/figure2.png)

**说明**：MuJoCo 鱼眼渲染管线。Pinhole → Cube Map（六向相机拼接）→ Spherical/Equirectangular（球面展开为等距柱状全景）→ Fisheye Image（施加鱼眼投影模型）。该管线填补了主流机器人仿真器无鱼眼渲染的空白，是 RQ1/RQ3 仿真实验的基础设施。

### Figure 3: Random Crop vs Random Scale Augmentation (RSA) / RSA 增强示意

![Figure 3](https://robo-fisheye.github.io/static/images/figure3.png)

**说明**：上排为传统固定尺度随机裁剪（scale=0.9）；下排为本文 **RSA**（scale 在 0.6–1.4 区间随机），同一物体被呈现为多种尺度。直观展示 RSA 如何打破策略对绝对像素尺度的记忆，是本文核心方法的可视化。

### Figure 4: Six Simulation Tasks / 六个仿真任务

![Figure 4](https://robo-fisheye.github.io/static/images/figure4.png)

**说明**：仿真六任务——(a) Square、(b) Tool Hang、(c) Coffee、(d) Threading、(e) Assembly、(f) Mug Cleanup，覆盖高精度操作(Tool Hang/Threading)、空间泛化(Square/Assembly)、长程任务(Coffee/Mug Cleanup)三类能力。

### Figure 5: Real-World Setup & Tasks / 真机平台与任务

![Figure 5](https://robo-fisheye.github.io/static/images/figure5.png)

**说明**：(a) 真机平台：Flexiv Rizon 4 机械臂 + DH AG-160-95 夹爪，腕部相机 + 第三视角评估相机 + 可更换背景（用于 RQ1 复杂度与 RQ2 泛化），Meta Quest 3 遥操作采集。(b) 三个真机任务：Pick Cup（空间泛化）、Fold Towel（可变形物体）、Hang Chinese Knot（高精度旋转）。

### 任务示意（项目主页补充图）

![Pick Cup](https://robo-fisheye.github.io/static/images/task_pick_cup.png)
![Fold Towel](https://robo-fisheye.github.io/static/images/task_fold_towel.png)
![Hang Chinese Knot](https://robo-fisheye.github.io/static/images/task_hang_knot.png)

**说明**：三个真机任务的单独示意图（Pick Cup / Fold Towel / Hang Chinese Knot），对应真机归一化评分的多阶段定义。

### Figure 6 (PDF): Real-World RQ1 Performance / 真机场景复杂度对比

**说明**（PDF 图，无在线外链）：三个真机任务下 feature-poor vs feature-rich 的归一化评分柱状图。关键发现：rich 背景下性能增益**对鱼眼更显著**（真机平均 +0.39 vs 针孔 +0.18），归因于 CLIP 编码器 + 真实纹理被鱼眼宽 FoV 充分利用。证实"鱼眼定位增益依赖场景丰富度"。

### Figure 8 (PDF): Scene Diversity Scaling (RQ2) / 场景多样性扩展曲线

**说明**（PDF 图）：(a) 仿真 Coffee 任务、(b) 真机 Pick Cup 任务下，unseen 场景成功率随训练场景数 $N$ 的变化。鱼眼曲线**斜率显著陡于针孔**；真机仅 8 个多样场景训练，鱼眼零样本成功率即 >95%。仿真曲线较缓（ResNet-18 vs CLIP、仿真背景视觉复杂度较低）。

### Figure 9 / S10 (PDF): Cross-Hardware Failure Cases / 跨硬件失败的尺度-深度歧义

**说明**（PDF 图）：换镜头导致物体绝对尺度变化 → 策略误判深度：物体看起来更大→"以为更近"→抓得太浅("move closer"失败)；更小→"以为更远"→抓得太深("move away"失败)；唯有训练尺度(Crop=1.0)成功。直接证明跨相机失败根因是**对绝对像素尺度的过拟合**。

### Figure 10 (PDF): RSA vs Standard Aug across Unseen Params (RQ3) / RSA 跨内参鲁棒性

**说明**（PDF 图）：六任务平均，不同 unseen 相机参数(Param1–5)下的成功率。基线(标准增强)在偏离训练内参时(Param3/4)严重掉点；**RSA 在全部配置上保持更高且稳定**的成功率，验证"学相对尺度 = 跨相机鲁棒"。

### Table 1: RQ1 Simulation — Scene Complexity / 场景复杂度仿真成功率

括号内为 rich 相对 poor（同相机）的绝对差异；粗体为该列最优。

| Camera | Feature | Square | Tool Hang | Coffee | Threading | Assembly | Mug Clean | Average |
|--------|---------|--------|-----------|--------|-----------|----------|-----------|---------|
| Pinhole(Single) | Poor | 0.40 | 0.52 | 0.36 | 0.14 | 0.14 | 0.40 | 0.31 |
| Pinhole(Single) | Rich | 0.48 (+0.08) | 0.56 (+0.04) | 0.34 (-0.02) | 0.18 (+0.14) | 0.12 (-0.02) | 0.38 (-0.02) | 0.34 (+0.03) |
| Fisheye(Single) | Poor | 0.68 | 0.80 | **0.80** | 0.30 | 0.24 | 0.58 | 0.57 |
| Fisheye(Single) | Rich | **0.74 (+0.06)** | **0.84 (+0.04)** | 0.76 (-0.04) | **0.56 (+0.26)** | **0.48 (+0.24)** | **0.60 (+0.02)** | **0.66 (+0.09)** |
| Pinhole(Double) | Poor | 0.50 | 0.44 | 0.26 | 0.22 | 0.44 | 0.40 | 0.38 |
| Pinhole(Double) | Rich | 0.70 (+0.20) | 0.34 (-0.10) | 0.36 (+0.10) | 0.38 (+0.16) | 0.34 (-0.10) | 0.56 (+0.16) | 0.45 (+0.07) |
| Fisheye(Double) | Poor | 0.86 | 0.84 | 0.74 | **0.68** | **0.56** | 0.66 | 0.72 |
| Fisheye(Double) | Rich | **0.88 (+0.02)** | **0.88 (+0.04)** | **0.86 (+0.12)** | 0.66 (-0.02) | 0.44 (-0.12) | **0.80 (+0.14)** | **0.75 (+0.03)** |

**说明**：鱼眼整体远超针孔（单相机 0.66 vs 0.34，双相机 0.75 vs 0.45）；feature-rich 背景下鱼眼增益更大（尤其 Threading +0.26、Assembly +0.24），印证"宽 FoV 定位增益依赖场景视觉复杂度"。

### Table 2: RQ1 Proprioception Probing / 本体预测探针（视觉编码器空间感知）

在预训练视觉编码器上微调轻量 MLP 头回归本体状态，误差越低表示空间表征越好。

| Task | Camera | Feature | Trans. Err (cm) ↓ | Rot. Err (°) ↓ |
|------|--------|---------|-------------------|----------------|
| Pick Cup | Pinhole | Poor | 12.309 | 15.345 |
| Pick Cup | Pinhole | Rich | 5.367 | 7.612 |
| Pick Cup | Fisheye | Poor | 3.369 | 3.677 |
| Pick Cup | Fisheye | Rich | **2.362** | **3.394** |
| Fold Towel | Pinhole | Poor | 4.204 | 6.829 |
| Fold Towel | Pinhole | Rich | 5.329 | 6.434 |
| Fold Towel | Fisheye | Poor | 3.837 | 3.398 |
| Fold Towel | Fisheye | Rich | **2.908** | **2.952** |
| Hang Chinese Knot | Pinhole | Poor | 14.168 | 12.464 |
| Hang Chinese Knot | Pinhole | Rich | 7.683 | 9.377 |
| Hang Chinese Knot | Fisheye | Poor | 8.804 | 7.256 |
| Hang Chinese Knot | Fisheye | Rich | **5.143** | **4.887** |

**说明**：鱼眼编码器在所有任务上本体预测误差更低，feature-rich 鱼眼最优（如 Pick Cup 平移误差仅 1.73cm 级别），直接证明鱼眼学到更准确的空间表征——是 RQ1"鱼眼增强空间定位"的内部机理证据。

### Table S1: Simulation Tasks Overview / 仿真任务概览（附录）

| Task | Trajectory Counts | Data Source | Data Type |
|------|-------------------|-------------|-----------|
| Square | 200 | RoboMimic | PH |
| Tool Hang | 200 | RoboMimic | PH |
| Coffee | 500 | MimicGen | D1 |
| Threading | 500 | MimicGen | D0 |
| Assembly | 500 | MimicGen | D0 |
| Mug Cleanup | 500 | MimicGen | D0 |

**说明**：PH=Proficient-Human；D0=默认 reset 分布；D1=拓宽 reset 分布。RoboMimic 任务 200 条、MimicGen 任务 500 条轨迹。

### Table S2: Detailed Hyperparameters / 详细超参（附录）

| Config | Simulation | Real-World |
|--------|-----------|------------|
| Visual Backbone | ResNet-18 (No Pretrain) | CLIP ViT-B/16 |
| Pooling Method | Spatial Softmax | Spatial Softmax |
| Denoising Network | Conditional U-Net1D | Conditional U-Net1D |
| Action Space | Relative Action | Relative Action |
| Action Horizon $T_{act}$ | 8 | 8 |
| Observation Horizon $T_{obs}$ | 2 | 2 |
| Prediction Horizon $T_{pred}$ | 16 | 16 |
| Image Resolution | 128×128 | 224×224 |
| Image Preprocessing | Random Crop | Random Crop |
| Proprioceptive Input | None (State-free) | None (State-free) |
| Optimizer | AdamW | AdamW |
| Weight Decay | 1e-6 | 1e-6 |
| LR Schedule | Cosine Decay | Cosine Decay |
| Learning Rate (UNet) | 1e-4 | 1e-4 |
| Learning Rate (Encoder) | 1e-4 | 1e-5 |
| Batch Size | 16 | 64 |
| Training Epochs | 2000 | 500 |
| EMA Decay | 0.75 | 0.9999 |

**说明**：仿真与真机的完整训练配置；二者均 state-free，仅编码器/分辨率/batch/epoch 不同，保证鱼眼-针孔公平对比。

### Table S3: RQ1 Proprioception Ablation in Simulation / 去本体消融（附录）

括号内为 w/o State 相对 w/ State 的相对降幅；粗体为 w/o State 下更优者。

| Camera | State | Square | Tool Hang | Coffee | Threading | Assembly | Mug Clean | Average |
|--------|-------|--------|-----------|--------|-----------|----------|-----------|---------|
| Pinhole(Single) | w/ State | 0.82 | 0.68 | 0.38 | 0.72 | 0.46 | 0.66 | 0.62 |
| Pinhole(Single) | w/o State | 0.48 (-41%) | 0.56 (-18%) | 0.34 (-11%) | 0.18 (-75%) | 0.12 (-74%) | 0.38 (-42%) | 0.34 (-45%) |
| Fisheye(Single) | w/ State | 0.86 | 0.88 | 0.88 | 0.72 | 0.58 | 0.60 | 0.75 |
| Fisheye(Single) | w/o State | **0.74 (-14%)** | **0.84 (-5%)** | **0.76 (-14%)** | **0.56 (-22%)** | **0.48 (-17%)** | **0.60 (0%)** | **0.66 (-12%)** |
| Pinhole(Double) | w/ State | 0.92 | 0.54 | 0.36 | 0.76 | 0.58 | 0.66 | 0.64 |
| Pinhole(Double) | w/o State | 0.70 (-24%) | 0.34 (-37%) | 0.36 (-5%) | 0.38 (-50%) | 0.34 (-41%) | 0.56 (-15%) | 0.45 (-30%) |
| Fisheye(Double) | w/ State | 0.94 | 0.88 | 0.88 | 0.78 | 0.56 | 0.80 | 0.81 |
| Fisheye(Double) | w/o State | **0.88 (-6%)** | **0.88 (0%)** | **0.86 (-2%)** | **0.66 (-15%)** | **0.44 (-21%)** | **0.80 (0%)** | **0.75 (-7%)** |

**说明**：去掉本体后针孔崩溃（平均 -45% / -30%），鱼眼几乎不受影响（-12% / -7%）。证明鱼眼宽 FoV 已隐式编码机器人空间关系，使显式状态输入"近乎冗余"，是 RQ1 结论的强力支撑。

### Table S4: Real-World Proprioception Ablation / 真机去本体消融（附录）

feature-rich 设置下三任务归一化评分；括号为 w/o State 相对降幅。

| Camera | State | Pick Cup | Fold Towel | Hang Knot | Average |
|--------|-------|----------|------------|-----------|---------|
| Pinhole | w/ State | 0.75 | 0.37 | 0.45 | 0.52 |
| Pinhole | w/o State | 0.65 (-13%) | 0.32 (-14%) | 0.15 (-67%) | 0.37 (-29%) |
| Fisheye | w/ State | 0.98 | 0.92 | 0.70 | 0.87 |
| Fisheye | w/o State | **0.80 (-18%)** | **0.70 (-24%)** | **0.50 (-29%)** | **0.67 (-23%)** |

**说明**：真机更明显——纯视觉(w/o State)下鱼眼(0.67)甚至超过带本体的针孔(0.52)；Fold Towel 鱼眼 0.70 vs 针孔 0.32（近翻倍）。再证鱼眼空间定位优势。

### Table S5: Third-View Camera Ablation in Simulation / 第三视角消融（附录）

双腕相机 + 第三视角配置，六任务成功率。

| Config (Double Cam + 3rd) | Square | Tool Hang | Coffee | Threading | Assembly | Mug Clean | Mean |
|---------------------------|--------|-----------|--------|-----------|----------|-----------|------|
| Pinhole baseline | 0.94 | 0.78 | 0.78 | 0.80 | 0.56 | 0.66 | 0.75 |
| Fisheye (Ours) | **0.96** | **0.84** | **0.82** | **0.82** | **0.66** | **0.72** | **0.80** |
| Improvement | +0.02 | +0.06 | +0.04 | +0.02 | +0.10 | +0.06 | **+0.05** |

**说明**：即便加入第三视角，鱼眼仍保持 +5% 平均增益（Assembly 高精度任务 +10%），说明鱼眼提供的精细局部上下文（夹爪-物体相对位姿）不会被全局视角冗余替代。

### Table S6: Performance in Different Real-World Environments / 不同真实环境（附录）

按 ORB 特征密度划分视觉丰富度。

| Typical Test Scene | Feature Density (ORB pts/frame) | Pinhole Score | Fisheye Score | Fisheye Improvement |
|--------------------|---------------------------------|---------------|---------------|---------------------|
| Textureless (Poor) | 1299.13 ± 176.81 (Low) | 0.1250 | 0.5250 | +0.4000 |
| Wooden Desk (Typical/Median) | 2111.43 ± 341.87 (Median) | 0.1250 | 0.5500 | +0.4250 |
| Highly Textured (Rich) | 3574.99 ± 102.54 (High) | 0.1813 | 0.9875 | +0.8062 |

**说明**：即便在中位特征密度的"普通木桌"场景，鱼眼也大幅领先(+0.4250)；高纹理场景增益达 +0.8062。证明鱼眼优势在日常部署场景普遍成立，并非仅极端富纹理情形。

### Table S7: Ablation on Visual Encoders / 视觉编码器消融（附录）

| Domain | Sim Feat. Density | Real Feat. Density | | | |
|--------|-------------------|--------------------|---|---|---|
| ORB pts/frame | 268.77 ± 30.27 | 3574.99 ± 102.54 | | | |

| Encoder | ResNet-18 (Sim) | CLIP (Sim) | ResNet-18 (Real) | CLIP (Real) |
|---------|-----------------|------------|------------------|-------------|
| Pinhole | 0.4467 | 0.5333 | 0.4250 | 0.7000 |
| Fisheye | **0.7533** | **0.77** | **0.7000** | **0.8875** |

**说明**：真实场景特征密度约为仿真的 13 倍(3574.99 vs 268.77)，故真机需 CLIP 这类强编码器。无论 ResNet-18 还是 CLIP，鱼眼都稳定胜出，说明**鱼眼的 FoV 收益独立于具体神经网络架构**。

### Table S8: Simulation Camera Parameters for RQ3 / RQ3 仿真相机参数（附录）

| Config Name | Method | Focal Length | Distortion | Scale |
|-------------|--------|--------------|------------|-------|
| Seen Param | EUCM | 45 | a=0.4, b=2.0 | 0.9 |
| Param 1 | EUCM | 60 | a=0.5, b=2.0 | 1.0 |
| Param 2 | DS | 50 | a=0.5, xi=0.1 | 1.0 |
| Param 3 | EUCM | 45 | a=0.4, b=2.0 | 1.0 |
| Param 4 | EUCM | 45 | a=0.4, b=2.5 | 1.0 |
| Param 5 | EUCM | 35 | a=0.4, b=1.2 | 1.0 |

**说明**：EUCM(Extended Unified Camera Model) 与 DS(Double Sphere) 两类投影模型，变化焦距/畸变/尺度构造 1 个 seen + 5 个 unseen 内参配置，模拟跨硬件的几何域移。

### Table S9: Normalized Scores under Simulated Scale Shifts (RSA) / 尺度移位下评分（附录）

Pick-and-Place 任务，标准增强 vs RSA 在不同尺度因子 $S$ 下的归一化评分；粗体为更优。

| Policy Model | Aug. Strategy | S=0.70 (Zoom-in) | Param1 | S=1.0 (Seen) | S=1.15 | S=1.30 (Zoom-out) |
|--------------|---------------|------------------|--------|--------------|--------|-------------------|
| Diffusion Policy | Standard Aug | 0.000 | 0.350 | 0.750 | 0.750 | 0.500 |
| Diffusion Policy | **RSA (Ours)** | **0.725** | **0.950** | **1.000** | **0.750** | **0.650** |
| $\pi_{0.5}$ | Standard Aug | 0.375 | 0.875 | 1.000 | 0.600 | 0.150 |
| $\pi_{0.5}$ | **RSA (Ours)** | **0.900** | **1.000** | **1.000** | **0.975** | **1.000** |

**说明**：标准增强呈"倒 V"曲线——偏离训练尺度即崩溃（Diffusion Policy 在 S=0.70 时为 0）；RSA 在全尺度范围保持高分（S=0.85 时 RSA 0.950 vs 标准 0.350）。RSA 接入 $\pi_{0.5}$ 大架构后甚至在 S=1.30 仍达 1.000，证明 RSA 与大模型正交且有效。

### Table S10: Real-World Cross-Camera Generalization with RSA / 真机跨镜头泛化（附录）

策略在标准 180° 镜头训练，零样本部署到不同物理镜头（用 $\pi_{0.5}$ 架构）。

| Test Camera | FoV Angle | Induced Scale Shift | Baseline (Standard Aug.) | RSA (Ours) |
|-------------|-----------|---------------------|--------------------------|------------|
| Seen Camera | 180° | 1.0× (Seen) | 1.0000 | **1.0000** |
| Narrow Lens | 150° | ~1.2× (Zoom In) | 0.5000 | **0.9500** |
| Wide Lens | 220° | ~0.8× (Zoom Out) | 0.0025 | **0.6000** |

**说明**：真机零样本换镜头时，基线在 Wide Lens(220°) 几乎归零(0.0025)，RSA 恢复到 0.6000；Narrow Lens 从 0.5000 提升到 0.9500。这是 RSA 在真实硬件迁移上的决定性证据。

---

## 实验

### 数据集 / 基准

| 基准 / 平台 | 规模 | 特点 | 用途 |
|-------------|------|------|------|
| Robomimic (MuJoCo) | Square/Tool Hang 各 200 条(PH) | 高精度、空间任务 | 训练/测试 |
| MimicGen (MuJoCo) | Coffee/Threading/Assembly/Mug 各 500 条 | 长程、空间泛化任务 | 训练/测试 |
| 仿真鱼眼渲染 | 自建两阶段管线(OmniCV-Lib 风格) | 235° FoV 鱼眼 / 90° 针孔 | RQ1–RQ3 |
| 真实世界 | 3 任务，每任务 $N=20$ 试 | Flexiv Rizon 4 + DH 夹爪，Meta Quest 3 遥操作 | 训练/测试 |
| 场景多样性数据 | MimicLab 32 训练纹理 + 5 unseen(仿真)；8 训练 + 4 unseen(真机) | 固定总数据量协议 | RQ2 |

### 实现细节

- **算法**：Diffusion Policy（Conditional U-Net1D + DDIM），state-free，相对动作空间，$T_{pred}=16/T_{act}=8/T_{obs}=2$。
- **编码器**：仿真 ResNet-18(无预训练，128×128)；真机 CLIP ViT-B/16(224×224)。
- **优化**：AdamW，cosine LR，UNet lr 1e-4，weight decay 1e-6；仿真 batch 16/2000 epoch/EMA 0.75，真机 batch 64/500 epoch/EMA 0.9999。
- **相机配置**：针孔 90°(仿真)/60°(真机)；鱼眼 235°(仿真)/180°(真机)；均腕部安装、无第三视角（消融除外）。
- **评估**：仿真用成功率(SR，50 rollouts，取训练中最高 checkpoint)；真机用多阶段归一化评分(N=20)。
- **RQ3 内参**：EUCM/DS 投影模型，1 seen + 5 unseen；真机 Narrow(150°)/Wide(220°) 物理镜头。

### 关键实验结论

- **RQ1 空间定位**：鱼眼显著优于针孔（Table 1：单相机 0.66 vs 0.34），但增益**依赖场景视觉丰富度**（Table S6：特征贫乏时几乎无优势）；本体探针(Table 2)与去本体消融(Table S3/S4)证明鱼眼学到更准空间表征、几乎不依赖显式状态。**指南：在视觉复杂、特征丰富的环境采集数据。**
- **RQ2 场景泛化**：固定数据量下，鱼眼策略随训练场景数 $N$ 的泛化曲线远陡于针孔(Figure 8)，真机 8 个场景即 >95% 零样本成功率；鱼眼在简单场景易过拟合，但多样性足够时泛化反超。**指南：最大化采集时的环境多样性。**
- **RQ3 硬件泛化**：朴素跨镜头迁移灾难性失败，根因为**尺度过拟合**(Figure 9/S10)；RSA 在仿真(Table S9)与真机(Table S10)大幅恢复性能，且接入 $\pi_{0.5}$ 仍有效。**指南：训练时使用强尺度增强(如 RSA)以提升跨相机迁移能力。**

---

## 批判性思考

### 优点
1. **填补真实空白且系统严谨**：首个针对腕部鱼眼相机的操作学习实证研究，仿真+真机双端、三大 RQ、对照变量干净，并自建 MuJoCo 鱼眼渲染管线，基础设施贡献明确。
2. **机理诊断到位**：不止给现象，还用本体探针(Table 2)、去本体消融(Table S3/S4)、尺度-深度歧义可视化(Figure 9)把"为什么"讲清楚，把跨相机失败精准定位为尺度过拟合。
3. **方案简单可落地**：RSA 仅是数据增强、零额外推理成本，且在 Diffusion Policy 与 $\pi_{0.5}$ 上均有效，对大规模鱼眼数据采集与 VLA 训练有直接指导价值，结论以"actionable guidance"形式给出。

### 局限性
1. **策略骨架较窄**：主体实验用 Diffusion Policy(state-free)，虽附录在 $\pi_{0.5}$ 上验证 RSA，但 RQ1/RQ2 的核心结论是否在大规模 VLA/带本体设定下同样成立缺乏完整验证。
2. **任务与规模有限**：仿真 6 任务、真机仅 3 任务(20 试)，且偏桌面抓放/插入/折叠；接触丰富、双手协调、长时序等复杂操作覆盖不足，统计规模(N=20)偏小。
3. **RSA 的尺度区间为经验值**：$\mathcal{U}(0.7,1.3)$ / $0.6$–$1.4$ 等区间靠经验设定，缺乏对采样分布、与畸变模型耦合关系的系统敏感度分析；RSA 仅处理"尺度"维度，未直接建模畸变本身的跨镜头差异。

### 潜在改进方向
1. 将结论扩展到带本体输入、更大 VLA 主干与跨本体设置，验证"鱼眼定位/泛化优势"与 RSA 的可移植性。
2. 把 RSA 从单纯尺度增强推广为**畸变感知增强**（联合采样 FoV/畸变内参），或与显式去畸变/可学习投影对齐结合。
3. 引入更难任务（接触丰富、双臂、长程）与更多物理镜头，量化"FoV / 特征密度 / 多样性"对性能的因果贡献(如更系统的 ORB/CKA 等指标)。

### 可复现性评估
- [ ] 代码开源（论文/主页未明确给出代码仓库链接，仅项目主页含视频与图）
- [ ] 预训练模型（未声明 release）
- [x] 训练细节完整（附录 Table S2 给出仿真/真机完整超参，S1 给数据规模）
- [x] 数据集可获取（Robomimic/MimicGen 公开；真机平台 Flexiv Rizon 4 + DH 夹爪、MimicLab 纹理库可复现，真机数据未明确 release）

---

## 速查卡片

> [!summary] Rethinking Camera Choice: Fisheye Camera Properties in Robotic Manipulation
> - **核心**：首个腕部鱼眼相机模仿学习系统实证研究，回答空间定位/场景泛化/硬件泛化三问，并提出 RSA 解决跨相机迁移。
> - **方法**：MuJoCo 两阶段鱼眼渲染 + state-free Diffusion Policy 对照实验(鱼眼 vs 针孔) + [[Random Scale Augmentation|RSA]]（随机尺度增强 $s\sim\mathcal{U}(0.7,1.3)$，逼策略学相对尺度）。
> - **结果**：鱼眼空间定位强但依赖场景丰富度(Table 1 0.66 vs 0.34)；多样性足够时鱼眼泛化反超(真机 8 场景 >95%)；RSA 把真机换镜头零样本从 0.0025/0.50 恢复到 0.60/0.95(Table S10)。
> - **指南**：富纹理场景采集 + 最大化场景多样性 + 强尺度增强。
> - **主页**：https://robo-fisheye.github.io/

---

*笔记创建时间: 2026-06-29*
