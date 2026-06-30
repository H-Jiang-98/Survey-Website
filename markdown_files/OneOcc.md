---
title: "OneOcc: Semantic Occupancy Prediction for Legged Robots with a Single Panoramic Camera"
method_name: "OneOcc"
authors: [Hao Shi, Ze Wang, Shangwei Guo, Mengfei Duan, Song Wang, Teng Chen, Kailun Yang, Lin Wang, Kaiwei Wang]
year: 2026
venue: CVPR
tags: [semantic-occupancy, semantic-scene-completion, panoramic, legged-robot, vision-only, occupancy-prediction, mixture-of-experts]
zotero_collection: _inbox
image_source: online
arxiv_html: https://arxiv.org/html/2511.03571v2
created: 2026-06-29
---

# OneOcc: Semantic Occupancy Prediction for Legged Robots with a Single Panoramic Camera

## 元信息

| 项目 | 内容 |
|------|------|
| 作者 | Hao Shi, Ze Wang, Shangwei Guo, Mengfei Duan, Song Wang, Teng Chen, Kailun Yang, Lin Wang, Kaiwei Wang |
| 机构 | 浙江大学、湖南大学、香港科技大学（广州）等（PAL 全景相机 / 具身感知方向） |
| 会议 | CVPR 2026 |
| 类别 | 语义占据预测 / 语义场景补全（SSC）· 全景视觉 · 足式/人形机器人 |
| 日期 | 2025-11（arXiv v2） |
| 项目主页 | https://github.com/MasterHow/OneOcc |
| 链接 | [arXiv](https://arxiv.org/abs/2511.03571) / [Code](https://github.com/MasterHow/OneOcc) |

---

## 一句话总结

> 用**单个全景相机**为足式/人形机器人做 360° 纯视觉语义占据补全：通过双投影编码、笛卡尔-柱坐标双网格体素化、步态位移补偿与分层 AMoE-3D 解码器，缓解步态抖动与全景畸变，在两个新基准上刷新视觉 SOTA 并逼近 LiDAR。

---

## 核心贡献

1. **面向足式平台的全景纯视觉 SSC 框架**: 提出 [[OneOcc]]，针对足式/人形机器人特有的**步态抖动（gait jitter）**与 **360° 环形连续性**设计，仅用一台全景相机即可输出全周向（full-surround）3D 语义占据，无需 LiDAR。
2. **四个轻量即插即用模块**: (i) 双投影融合 [[DP-ER]]（环形原图 + 等距柱状展开）保 360° 连续性与栅格对齐；(ii) 双网格体素化 [[BGV]]（笛卡尔 + 柱/极坐标）降低离散化偏差、锐化自由/占据边界；(iii) 分层 [[AMoE-3D]] 解码器做动态多尺度融合与长程/遮挡推理；(iv) 步态位移补偿 [[GDC]]，**无需额外传感器**在特征层做运动校正。
3. **两个全景占据基准**: 发布 [[QuadOcc]]（真实四足、第一人称 360°）与 [[Human360Occ|H3O]]（CARLA 人本视角 360°，含 RGB/深度/语义占据，标准化的同城/跨城划分）。OneOcc 在 QuadOcc 刷新 SOTA、与经典 LiDAR 基线竞争；H3O 同城 +3.83 mIoU、跨城 +8.08 mIoU。

---

## 问题背景

### 要解决的问题
绝大多数语义场景补全（[[Semantic Scene Completion|SSC]]）系统是为**轮式平台 + 前向传感器**设计的（如 SemanticKITTI 的前向驾驶锥）。但足式/人形机器人需要的是：**全周向（360°）**的鲁棒 3D 语义占据，且要在**步态引入的机身抖动**下工作。如何只用**一台全景相机**（不依赖 LiDAR）得到可部署的全周向语义占据，是本文要解决的核心问题。

### 现有方法的局限
作者在 3.2 节归纳了三点关键困难：
1. **单一投影无法兼顾两端**: 360° 全景同时具有**环形连续性**与**强投影畸变**——单一投影在极区 vs. 赤道区无法同时平衡分辨率与感受野。
2. **步态抖动伤害"特征→体素"提升**: 足式运动带来冲击性触地、小幅 roll/pitch 抖动，对 feature→voxel 的 lifting 步骤伤害比 image→feature 更大。
3. **近/远场需求冲突**: 全周向 SSC 必须同时调和**近场接触几何**（脚落点、障碍物）与**远场环形上下文**。

此外现有视觉 SSC（MonoScene、VoxFormer、SGN、OccFormer）与 LiDAR SSC（SSCNet、LMSCNet、OccRWKV）都不是为全景 + 足式设定设计的。

### 本文的动机
- 既然 ER（等距柱状）展开对卷积友好、保方位连续性，而原始环形（raw annulus）保留原生几何与细纹理，那就**两条投影并行**互补极区/赤道权衡。
- 笛卡尔体素近场度量精确、柱坐标体素与全景成像天然对齐（方位角 $\varphi\approx\phi$），那就**双网格融合**取两者之长。
- 步态抖动若在 lifting 之后再纠正，作用的是已经体素量化过的证据；那就**在 lifting 之前、于采样坐标处**做补偿，更便宜也避免量化损失。
- 全景场景在方位/垂直方向**各向异性**且近远尺度差异大，那就用**分层专家混合 + 梯度能量门控**做尺度感知的各向异性融合。

---

## 方法详解

### 模型架构

[[OneOcc]] 是一个**纯视觉、单帧、无里程计**的全周向 SSC 流水线（见 Figure 2），输入一张全景 RGB，输出稠密 3D 语义占据：

- **输入**: 全景 RGB 图像 $\mathbf{I}\in\mathbb{R}^{H\times W\times 3}$ + 标定内外参 $\boldsymbol{\kappa}$
- **前端**: 标定展开（[[PAL]] 环形 → ER）+ **双 2D 编码器**（[[DP-ER]]：ER 路 + raw 路）
- **核心模块**: [[GDC]] 步态位移补偿（lifting 前）→ [[BGV]] 双网格体素化 + View2View 采样 → 分层 [[AMoE-3D]] 轻量 3D 解码器
- **输出**: 每体素 logits $\mathbf{Z}\in\mathbb{R}^{X\times Y\times Z\times C}$，取 $\arg\max$ 得语义占据 $\mathbf{S}\in\{0,\dots,C\}^{X\times Y\times Z}$
- **参数/效率**: 101.76M（H3O 单投影变体约 100M；QuadOcc 双投影 189.8M），RTX 4090 上 14.3 FPS（FP32）/ 18.9 FPS（混精度），峰值 ~1.82 GB；Jetson Orin 上 13.64 FPS

整体目标（问题表述，3.1）：给定全景图与标定，预测稠密 3D 语义占据，最终标签为

$$
\hat{s}_{x,y,z}=\arg\max_{c}\mathbf{Z}_{x,y,z,c}
$$

### 核心模块

#### 模块1: Unwrapping + Dual-Projection Encoders（解卷绕 + 双投影编码器，3.3）

**设计动机**: ER 视图保方位连续、卷积友好；raw 环形保原生几何与细纹理。两路并行可补偿赤道-极区权衡。

**具体实现**:
- 用 **Taylor 多项式模型**（[[OCam]] 标定）标定 [[PAL|全景环形透镜]]，按标定逆映射把 raw 环形采样到 ER（公式1）；ER 射线的球面角 $(\phi,\theta)$、半径 $r(\theta)$ 与 raw 像素坐标由公式2-3 给出。
- 两个编码器分别产出多尺度特征 $\{\mathbf{F}^{\mathrm{equi}}_{1/s}\}$ 与 $\{\mathbf{F}^{\mathrm{raw}}_{1/s}\}$（$s\in\{1,4,8,16\}$，公式4），用 GroupNorm + $1\times1$ bottleneck 对齐通道。
- **H3O 上禁用 raw 分支**（H3O 本就是原生 ER），仅保留 ER 路，使模型显著更轻。

#### 模块2: Bi-Grid Voxelization + View2View Sampling（双网格体素化，3.4）

**设计动机**: 笛卡尔网格近场接触几何精确，柱/极坐标网格保方位连续、匹配全景成像（ER 水平轴线性对应方位角，柱坐标 $(r,\varphi,z)$ 中 $\varphi\approx\phi$、$r$ 关联深度），两者互补可降量化偏差、平衡近/远场。

**具体实现**:
- 定义笛卡尔与极坐标两套体素质心 $\mathbf{c}^{\mathrm{Ca}}_{ijk}$、$\mathbf{c}^{\mathrm{Po}}_{pqk}$（公式5）。
- 对每个视图 $v\in\{\mathrm{equi},\mathrm{raw}\}$ 与尺度 $s$，在投影像素 $\pi_v(\cdot)$ 处**双线性采样**提升特征（公式6，View2View）。
- 用**逐体素凸权重**跨尺度融合（公式7，$\sum_s\alpha_s=1$）。
- 预计算跨网格索引 $\{\mathcal{J}_\ell\}$，把极坐标上下文注入笛卡尔流（公式8，对齐后拼接）。

#### 模块3: Gait Displacement Compensation（步态位移补偿，3.5）

**设计动机**: 步态冲击产生相位误差；在 lifting 之后纠正作用于已量化的体素证据，而在 lifting 之前、于采样坐标纠正既避免量化又更便宜。

**具体实现**:
- 对每尺度每投影，用**零初始化头**（cf. zero-conv）从 GAP 特征回归 2D 位移 $\Delta_s=(d_x,d_y)$（公式9）——零初始化使初始 warp 为恒等，小运动时不扰动特征。
- lifting 前先校正采样像素 $\widehat{\mathbf{p}}=\pi_v(\mathbf{c};\boldsymbol{\kappa})+\Delta_s$ 再双线性重采样（公式10-11）。
- 顺带把 lifting 算子从整数索引 gather 升级为双线性采样，减少投影混叠。GDC 主要沿**垂直方向**自适应（步态抖动表现为竖直运动模糊，见 Figure 8）。

#### 模块4: Hierarchical AMoE-3D（分层注意力-专家混合 3D 解码器，3.6）

**设计动机**: 全景场景**各向异性**（方位强变 vs. 垂直结构、近远尺度差大）。故 (i) 分层（粗到细）注入极坐标线索，(ii) 逐体素专家选择，避免平坦区过平滑、同时增强对运动关键的边缘/接触。

**具体实现**:
- 3D 解码器是**深度可分离 UNet**（DWLite3D），三级三线性上采样（L1/L2/L3）。
- **双路体积显著性**（通道门 $\mathbf{A}_c$ + 空间门 $\mathbf{A}_s$，公式12）即 AMoE-3D 中的 "A"（attention）。
- **MoE-Fuse3D + 梯度能量门控**（公式13）：用 3D 梯度能量 GradEnergy3D 做 softmax 路由，$K$ 个 $1{\times}1{\times}1$ Conv-GELU-Conv 专家残差融合；默认 $K{=}4$。

#### 模块5: Segmentation Head + Loss（3.7）

- $1{\times}1{\times}1$ 头出每体素 logits $\mathbf{Z}$，仅监督可见有效体素，stride $\{1,2,4\}$ 深监督。
- 损失沿用 [[MonoScene]]：交叉熵 + 场景-类别亲和（SCAL）语义/几何项 + frustum 比例（FP）损失（公式14）；**刻意去掉 relation loss**——它鼓励共现先验、过平滑方位边界、压制小近场类，全景足式设定下无增益甚至退化。

### 关键公式与机制

#### 公式1: [[PAL]] 解卷绕（raw → ER）

$$
\mathbf{I}^{\mathrm{equi}}(u,v)\;=\;\mathbf{I}^{\mathrm{raw}}\!\left(\mathcal{U}^{-1}(u,v;\boldsymbol{\kappa})\right)
$$

**含义**: 按标定的逆映射，把原始环形图重采样为等距柱状（ER）全景。

**符号说明**:
- $\boldsymbol{\kappa}=\{a_i,u_0,v_0,\mathbf{A}\}$: Taylor 系数、主点、$2\times2$ 仿射（尺度/skew）
- $(u,v)\in[0,W)\times[0,H)$: ER 像素坐标

#### 公式2-3: ER 射线球面角与 raw 坐标映射

$$
\phi=\tfrac{2\pi}{W}u-\pi,\qquad \theta=\tfrac{\pi}{2}-\tfrac{\pi}{H}v
$$

$$
r(\theta)=\sum_{i=0}^{N}a_i\,\theta^{\,i},\qquad
\begin{bmatrix}u_{\text{raw}}\\ v_{\text{raw}}\end{bmatrix}
=\begin{bmatrix}u_0\\ v_0\end{bmatrix}+\mathbf{A}\,r(\theta)\begin{bmatrix}\cos\phi\\ \sin\phi\end{bmatrix}
$$

**含义**: 由 ER 像素得到球面方位/俯仰角，再经 Taylor 半径多项式映回 raw 环形像素，实现尊重 PAL 光学的标定展开。

**符号说明**: $(\phi,\theta)$ 方位/俯仰角；$r(\theta)$ Taylor 半径多项式。

#### 公式4: 双投影多尺度特征

$$
\{\mathbf{F}^{\mathrm{equi}}_{1/s}\}_{s\in\{1,4,8,16\}},\qquad \{\mathbf{F}^{\mathrm{raw}}_{1/s}\}_{s\in\{1,4,8,16\}}
$$

**含义**: 两条编码器各产 4 个尺度特征，GroupNorm + $1\times1$ 对齐通道。

#### 公式5: 双网格体素质心

$$
\mathbf{c}^{\mathrm{Ca}}_{ijk}=(x_i,y_j,z_k),\qquad
\mathbf{c}^{\mathrm{Po}}_{pqk}=\big(r_p\cos\varphi_q,\;r_p\sin\varphi_q,\;z_k\big)
$$

**含义**: 同时定义笛卡尔与柱/极坐标两套体素中心，后续分别采样、互注入。

#### 公式6: View2View 提升（双线性采样）

$$
\mathbf{V}^{(\cdot)}_{s}(\mathbf{c})=\mathrm{bilinear}\!\left(\mathbf{F}^{v}_{1/s},\,\pi_v(\mathbf{c};\boldsymbol{\kappa})\right)
$$

**含义**: 把 3D 体素质心投到 2D，在对应像素处双线性采样 2D 特征，提升为 3D 体素特征（笛卡尔/极坐标各一份）。

#### 公式7: 多尺度凸权重融合

$$
\mathbf{V}^{\mathrm{Ca}}=\sum_{s}\alpha^{\mathrm{Ca}}_{s}\odot\mathbf{V}^{\mathrm{Ca}}_{s},\quad\sum_s\alpha^{\mathrm{Ca}}_s=1;\qquad
\mathbf{V}^{\mathrm{Po}}=\sum_{s}\alpha^{\mathrm{Po}}_{s}\odot\mathbf{V}^{\mathrm{Po}}_{s},\quad\sum_s\alpha^{\mathrm{Po}}_s=1
$$

**含义**: 逐体素的凸组合权重把各尺度体素特征自适应融合，$\odot$ 为逐元素乘。

#### 公式8: 极坐标→笛卡尔上下文注入

$$
\widetilde{\mathbf{V}}^{\mathrm{Ca}}_{\ell}=\mathrm{Align}_{1\times1\times1}\!\big(\mathbf{V}^{\mathrm{Po}}_{\ell}[\mathcal{J}_{\ell}]\big)\;\|\;\mathbf{V}^{\mathrm{Ca}}_{\ell}
$$

**含义**: 用预计算跨网格索引 $\mathcal{J}_\ell$ 把极坐标体取到笛卡尔栅格、对齐通道后拼接（$\|$），将方位连续上下文注入笛卡尔流。

#### 公式9-11: GDC 位移回归与校正采样

$$
\Delta_s=\mathrm{Linear}_0\!\big(\mathrm{GAP}(\mathbf{F}^v_{1/s})\big)
$$

$$
\widehat{\mathbf{p}}=\pi_v(\mathbf{c};\boldsymbol{\kappa})+\Delta_s,\qquad
\mathbf{V}^{(\cdot)}_s(\mathbf{c})=\mathrm{bilinear}\!\big(\mathbf{F}^v_{1/s},\widehat{\mathbf{p}}\big)
$$

**含义**: 零初始化头回归 2D 位移 $\Delta_s=(d_x,d_y)$；在 lifting 前把投影像素校正为 $\widehat{\mathbf{p}}$ 再重采样，把步态相位误差"退回"到 2D 解决。

**符号说明**:
- $\mathrm{Linear}_0$: 权重/偏置全零初始化（初始恒等 warp）
- $\pi_v(\mathbf{c};\boldsymbol{\kappa})$: 用标定 Taylor 相机把体素质心 $\mathbf{c}$ 投到像素

#### 公式12: 双路体积显著性（通道 + 空间门）

$$
\mathbf{A}_c=\sigma\!\big(\mathrm{MLP}(\mathrm{GAP}(\mathbf{X}))+\mathrm{MLP}(\mathrm{GMP}(\mathbf{X}))\big),\quad
\mathbf{A}_s=\sigma\!\big(g^{7\times7\times7}([\mathrm{Avg}(\mathbf{X});\mathrm{Max}(\mathbf{X})])\big),\quad
\mathbf{Y}=\mathbf{X}\odot\mathbf{A}_c\odot\mathbf{A}_s
$$

**含义**: 对 3D 体特征做通道门 $\mathbf{A}_c$ 与空间门 $\mathbf{A}_s$ 的注意力调制，即 AMoE-3D 的 "A"。

**符号说明**: $\mathbf{X}\in\mathbb{R}^{C\times D\times H\times W}$ 当前级笛卡尔特征；$\sigma$ sigmoid；$[\,;\,]$ 通道拼接；$g^{7\times7\times7}$ 大核 3D 卷积。

#### 公式13: MoE-Fuse3D + 梯度能量门控

$$
\mathrm{GradEnergy3D}(\mathbf{Y})=\sum_{a\in\{x,y,z\}}\big\|\nabla_a\mathbf{Y}\big\|_2^2,\qquad
\boldsymbol{\alpha}=\operatorname{softmax}\!\big(W_g*\mathrm{GradEnergy3D}(\mathbf{Y})\big)
$$

$$
\widetilde{\mathbf{Y}}=\mathbf{Y}+\sum_{k=1}^{K}\alpha_k\,E_k(\mathbf{Y})
$$

**含义**: 以 3D 梯度能量（沿 $x,y,z$ 的离散梯度平方和）做门控，对高梯度区（类别边界、细结构）优先分配专家；$K$ 个 $1{\times}1{\times}1$ Conv-GELU-Conv 专家残差加权融合。

**符号说明**: $\nabla_a$ 沿轴 $a$ 的离散 3D 梯度；$*$ 3D 卷积；$E_k$ 第 $k$ 个专家；$K{=}4$ 为默认。

#### 公式14: 总损失

$$
\mathcal{L}_{\mathrm{total}}=\mathcal{L}_{\mathrm{CE}}+\mathcal{L}^{\mathrm{sem}}_{\mathrm{SCAL}}+\mathcal{L}^{\mathrm{geo}}_{\mathrm{SCAL}}+\mathcal{L}_{\mathrm{FP}}
$$

**含义**: 交叉熵 + 语义/几何场景-类别亲和（SCAL）+ frustum 比例（FP）损失；**不含 relation loss**。$\mathcal{L}_{\mathrm{CE}}$ 按标准 SSC 做类别重加权。

---

## 关键图表

<!-- 图片均使用 arXiv HTML 在线链接，未本地化。Figure 9/13 为 TikZ 矢量流水线图，无独立位图资源，故以文字说明。 -->

### Figure 1: Teaser / 任务与结果概览

![Figure 1](https://arxiv.org/html/2511.03571v2/x1.png)

**说明**: (a) 单传感器全景 SSC：四足机器人载全向相机看全 360°（含步态运动），OneOcc 输出体素化语义占据。(b) 结果摘要：QuadOcc 上 OneOcc 20.56 mIoU，超过最佳 LiDAR 基线 LMSCNet(18.44) 与最佳视觉基线 MonoScene(19.19)；H3O 同城/跨城 37.29/32.23 mIoU，超过最佳视觉基线(33.46/24.15)。一图点明"纯视觉单相机即可逼近/超越 LiDAR"的核心卖点。

### Figure 2: OneOcc Pipeline / 整体流水线

![Figure 2](https://arxiv.org/html/2511.03571v2/x2.png)

**说明**: 标定展开把 raw 全景映到 ER；两个 2D 编码器（[[DP-ER]]）喂给 [[GDC]] 做特征级步态补偿（H3O 可选）；三尺度上 [[BGV]] 建笛卡尔与极/柱坐标体素，质心特征采样到各视图（Ca/Po→Equi/Raw）后融合；轻量 3D 解码器 + 分层 [[AMoE-3D]] 聚合多尺度上下文，最后头部预测全周向占据。是理解整套数据流的主图。

### Figure 3: GDC and AMoE-3D Fusion Module / GDC 与融合模块细节

![Figure 3](https://arxiv.org/html/2511.03571v2/x3.png)

**说明**: (a) [[GDC]] 从 2D 特征经 GAP+Linear（零初始化）回归 $\Delta=(dx,dy)$，对多尺度视图坐标 $\{C_s\}$ 做 scale/shift 用于体素质心采样，得 $\{F_{3d}\}_{V2V}$。(b) [[AMoE-3D]] 把 Po2Ca 采样的极坐标特征与笛卡尔特征通道对齐、拼接、约简，施加 3D 通道+空间注意力，再做 $K$ 专家 + GradEnergy3D-Softmax 门控的 MoE-Fuse3D（$y=\sum_k P_k\odot E_k(x)$），融合后 $F^{\mathrm{Fusion}}_{3d}$ 喂入解码器。对应公式9-13。

### Figure 4: Qualitative Comparisons / 定性对比（QuadOcc + H3O）

![Figure 4](https://arxiv.org/html/2511.03571v2/x4.png)

**说明**: 从左到右：参考全景、GT 占据、各基线、OneOcc。真实(QuadOcc)与仿真(H3O)全景下，OneOcc 更好地保全局布局与 360° 连续性、产生更干净的类别边界、恢复被遮挡结构。黄色虚线圈标出相对 LMSCNet/SGN-S/MonoScene 的典型改进（前景-背景消歧、远程布局）。

### Figure 5: Cartesian vs. Cylindrical Voxel Projections / 两种体素投影

![Figure 5](https://arxiv.org/html/2511.03571v2/x5.png)

**说明**: 可视化两种体素参数化投到 ER 全景：彩点为体素质心投影，色带(0-1)编码沿方位射线占据体素的归一化深度分布。(a) 轴对齐**笛卡尔**网格投到全景成扇形足迹、远程结构被压进赤道窄带——度量均匀但破坏全景角度规律；(b) **柱坐标**网格按等方位/等高步进，投影质心近似水平行、贴合 ER 参数化、保远程环形连续。这正是 [[BGV]] 双网格互补的动机图。

### Figure 6: GT Occupancy at Different Resolutions / 不同分辨率 GT 投影

![Figure 6](https://arxiv.org/html/2511.03571v2/x6.png)

**说明**: GT 语义体素从 $64\times64\times8$（左）与 $128\times128\times16$（右）投到全景图。高分辨率更密、边界更锐；但 $64\times64\times8$ 已足够让足式机器人感知可通行面与近障。考虑具身算力/载荷约束，主结果默认 $64\times64\times8$。

### Figure 7: Resolution vs. Throughput/Accuracy / 分辨率-效率折中

![Figure 7](https://arxiv.org/html/2511.03571v2/x7.png)

**说明**: 固定 3D 边界与体素大小，仅变栅格分辨率：BEV $64{\times}64\to128{\times}128$、垂直 bin $8\to16$。提分辨率仅 +3.41M 参数(101.76→105.17M)，却让 FPS 14.30→5.01、峰值显存 1.82→10.71 GB，IoU/mIoU 反降(49.58/32.23→29.98/21.16)。佐证低分辨率才是具身部署的甜点。

### Figure 8: Panoramic Gait Displacement Compensation / GDC 真实效果

![Figure 8](https://arxiv.org/html/2511.03571v2/x17.png)

**说明**: 真实四足数据上 GDC 的效果。每行一张不同步态强度的全景：左为输入图(含远处结构放大)，右为 GDC 前(蓝)/后(红)的 FLoSP 采样位置叠加在 ER 特征图上。从上到下机身振荡由轻到重，估计的竖直偏移 $dy$ 随之增大、而水平 $dx$ 较小——说明步态抖动产生特征性的**竖直运动模糊**，GDC 主要沿竖直方向自适应校正；配合双线性插值实现亚像素位移，恢复更锐利特征。

### Figure 10: QuadOcc GT Construction / QuadOcc 真值构建（附录）

![Figure 10](https://arxiv.org/html/2511.03571v2/x9.png)

**说明**: (a) PAL 相机采的 raw 全景环；(b) 经 OCam/Taylor 标定展开为 ER；(c) 在 ER 上做开放词表分割（Grounded-SAM）得每像素类别分数 $\mathcal{S}_t$、$\mathbf{Q}_t$；(d) 单时刻体素化等。展示 QuadOcc 半自动标注流程的关键中间结果。

### Figure 11: Dataset Statistics / 数据集统计（附录）

![Figure 11](https://arxiv.org/html/2511.03571v2/x10.png)

**说明**: (a) QuadOcc 语义频率（非空体素）：强头部类如 road≈52.34%，长尾为 pedestrian/vehicle；(b) H3O 语义频率：多数在 road/sidewalk/vegetation/building，交通类稀少；(c) QuadOcc 一天内时段分布等。揭示两基准都存在严重类别不平衡。

### Figure 12: QuadOcc Camera Pose Statistics / 相机位姿统计（附录）

![Figure 12](https://arxiv.org/html/2511.03571v2/x11.png)

**说明**: QuadOcc ego 相机 6-DoF 位姿在规范栅格系(X 前/Y 左/Z 上)的分布：俯视轨迹、朝向统计、高度/朝向分布等。覆盖集中在 12.8m 半径、$64\times64\times8$ 栅格内。

### Figure 14: H3O Weather and Lighting Diversity / H3O 天气光照多样性（附录）

![Figure 14](https://arxiv.org/html/2511.03571v2/x12.png)

**说明**: H3O 的代表性 ER 全景：(a) 晴-白天、(b) 晴-夜、(c) 雾、(d) 雨/阴(都市)。均为直接用于训练/评测的原生 ER。说明 H3O 跨 Clear/Fog/Rain × Day/Dusk/Night 的覆盖度。

### Figure 15: H3O Ego Pose Distribution / H3O ego 位姿分布（附录）

![Figure 15](https://arxiv.org/html/2511.03571v2/x13.png)

**说明**: H3O ego 位姿统计（与 QuadOcc 同栅格系），聚合 16 张 CARLA 地图、160 序列、多天气光照下的轨迹与 6-DoF 位姿。

### Figure 16: Failure Case on H3O-Heter / 失败案例（附录）

![Figure 16](https://arxiv.org/html/2511.03571v2/x14.png)

**说明**: H3O-Heter 跨城、雨天黄昏、植被主导场景下的失败案例。左到右：GT、MonoScene、SGN-S、OneOcc。OneOcc 缓解但未消除严重误分类（植被被预测为人行道），凸显罕见布局+天气+时段组合下的难度。

### Figure 17: Qualitative on QuadOcc by Time-of-Day / 不同时段定性（附录）

![Figure 17](https://arxiv.org/html/2511.03571v2/x15.png)

**说明**: QuadOcc 上 day/dusk/night 三时段定性对比（参考全景、GT、LMSCNet、MonoScene、OneOcc）。所有方法从白天到夜间逐步退化（弱光/眩光带来歧义），OneOcc 相对更稳。

### Figure 18: Distance-binned mIoU / 距离分箱 mIoU（附录）

![Figure 18](https://arxiv.org/html/2511.03571v2/x16.png)

**说明**: 在 Far(±12.8m)/Mid(±6.4m)/Near(±3.2m) 三个嵌套范围评非空体素。OneOcc 在两个基准、所有距离段都优于 MonoScene，包括对局部导航最关键的近场。

### Figure 9 / Figure 13: 数据集构建流水线（附录，矢量图无位图）

**说明**: Figure 9 为 QuadOcc 真值构建流水线（Livox Mid-360 LiDAR + PAL 同步 → OCam 展开 → Grounded-SAM 开放词表分割 → LiDAR-图像标签转移 → 语义引导动态建图 → 体素化与多数投票 Eq.18 → 时序聚合 Eq.19-20 与门控 3D mode Eq.21 → 精化体 $\mathbf{V}_{\mathrm{ref}}$）。Figure 13 为 H3O 流水线（cubemap 全景拼接成 ER → 速度条件竖直 bob 模拟足式运动 Eq.24 → ER 深度/语义反投影到相机系 Eq.25-26 → 对齐到规范栅格 → 体素化标注）。

### Table 2: QuadOcc Validation Results / QuadOcc 验证集结果

| Method | Input | vehicle | pedestrian | road | building | vegetation | terrain | Prec.↑ | Recall↑ | IoU↑ | mIoU↑ |
|--------|-------|---------|-----------|------|----------|-----------|---------|--------|---------|------|-------|
| *LiDAR-based* | | | | | | | | | | | |
| SSCNet | L | 0.00 | 0.04 | 44.34 | 16.06 | 22.41 | 4.77 | 63.42 | 65.82 | 47.71 | 14.60 |
| SSCNet-full | L | 0.71 | 0.04 | 55.14 | 14.93 | 19.66 | 7.76 | 78.81 | 60.03 | 51.69 | 16.38 |
| LMSCNet | L | 0.88 | 0.20 | 57.02 | 16.45 | 24.39 | 11.70 | 79.70 | 64.49 | **55.40** | 18.44 |
| OccRWKV | L | 2.05 | 0.35 | 54.17 | 10.29 | 24.92 | 8.38 | 71.98 | 62.80 | 50.47 | 16.69 |
| *Vision-based* | | | | | | | | | | | |
| VoxFormer-S | P+D | 0.31 | 1.67 | 36.83 | 6.48 | 7.09 | 3.81 | 45.96 | 69.95 | 38.38 | 9.36 |
| SGN | P+D | 7.39 | 1.88 | 53.35 | 9.06 | 21.95 | 9.39 | 59.23 | 67.41 | 46.05 | 17.17 |
| VoxFormer-S † | P+L | 0.24 | 1.66 | 38.51 | 10.90 | 11.88 | 4.45 | 51.29 | 75.84 | 44.08 | 11.27 |
| VoxFormer-T † | P+L | 0.30 | 1.65 | 40.21 | 14.48 | 15.86 | 4.64 | 56.94 | 77.10 | 48.70 | 12.86 |
| SGN † | P+L | 11.62 | 2.47 | 53.06 | 15.60 | 25.67 | 9.91 | 73.70 | 59.62 | 49.16 | 19.72 |
| OccFormer | P | 0.29 | 0.37 | 49.46 | 10.36 | 15.00 | 2.64 | 45.79 | 76.99 | 40.28 | 13.02 |
| MonoScene | P | 8.15 | 1.59 | 55.66 | 12.88 | 26.10 | 10.78 | 62.10 | 69.28 | 48.69 | 19.19 |
| **OneOcc (Ours)** | **P** | **12.16** | **2.86** | 54.41 | 16.03 | 24.91 | **13.01** | 66.69 | 64.74 | 48.92 | **20.56** |

**说明**: 纯相机的 OneOcc 取得 **20.56 mIoU**，超过最强 LiDAR 基线 LMSCNet(18.44) 与最强视觉基线 MonoScene(19.19)，甚至超过加了 LiDAR 的 SGN†(19.72)。SGN†(P+L) 凭 LiDAR 拿到最高几何 IoU(49.16)，但相机-only 的 OneOcc 仍在 mIoU 上居视觉方法之首，证明任务对齐的全景融合(DP-ER/BGV/AMoE-3D + GDC)能让纯视觉匹敌乃至超越 LiDAR 栈。(†=用了与原论文不同的传感器模态，此处加 LiDAR。)

### Table 3: H3O Results (HOMO within-city / HETER cross-city) / H3O 同城与跨城

**HOMO（同城）mIoU 与几何指标**

| Method | Input | road | sidewalk | building | vegetation | car | truck | bus | two_wheeler | person | pole | Prec. | Recall | IoU | mIoU |
|--------|-------|------|----------|----------|-----------|-----|-------|-----|------------|--------|------|-------|--------|-----|------|
| VoxFormer-S | P+D pred. | 22.27 | 17.44 | 2.86 | 5.65 | 0.87 | 0.14 | 0.00 | 0.00 | 61.46 | 0.18 | 40.21 | 59.49 | 31.57 | 11.09 |
| VoxFormer-S | P+D gt | 35.84 | 25.86 | 12.32 | 9.90 | 3.73 | 0.70 | 0.19 | 0.17 | 64.79 | 0.93 | 59.91 | 71.12 | 48.19 | 15.44 |
| SGN-T | P+D pred. | 54.55 | 44.61 | 15.51 | 35.16 | 23.91 | 7.68 | 4.52 | 15.77 | 65.24 | 19.46 | 61.82 | 64.36 | 46.05 | 28.64 |
| SGN-S | P+D pred. | 54.81 | 47.44 | 15.64 | 33.22 | 24.23 | 8.53 | 2.62 | 13.51 | 65.88 | 19.72 | 62.28 | 63.72 | 45.98 | 28.56 |
| OccFormer | P | 54.11 | 46.41 | 15.14 | 32.84 | 17.57 | 4.30 | 0.00 | 1.65 | 65.08 | 11.72 | 57.30 | 69.94 | 45.98 | 24.88 |
| MonoScene | P | 62.45 | 57.18 | 19.97 | 38.91 | 29.76 | 13.77 | 5.03 | 16.76 | 70.15 | 20.67 | 68.78 | 71.66 | 54.08 | 33.46 |
| **OneOcc (Ours)** | **P** | **63.82** | **62.92** | **22.86** | **41.03** | **33.74** | **17.79** | **6.47** | **26.58** | **71.13** | **26.55** | 69.14 | **74.18** | **55.73** | **37.29** |

**HETER（跨城）**

| Method | Input | road | sidewalk | building | vegetation | car | truck | bus | two_wheeler | person | pole | Prec. | Recall | IoU | mIoU |
|--------|-------|------|----------|----------|-----------|-----|-------|-----|------------|--------|------|-------|--------|-----|------|
| VoxFormer-S | P+D pred. | 19.02 | 15.82 | 2.84 | 5.86 | 0.95 | 0.10 | 0.00 | 0.01 | 61.62 | 0.10 | 36.68 | 52.00 | 27.40 | 10.63 |
| VoxFormer-S | P+D gt | 32.55 | 25.95 | 8.98 | 9.93 | 4.68 | 0.44 | 0.00 | 0.00 | 65.45 | 0.26 | 59.08 | 61.21 | 42.99 | 14.82 |
| SGN-T | P+D pred. | 42.57 | 23.94 | 9.42 | 15.84 | 18.22 | 10.21 | 1.55 | 3.11 | 63.44 | 11.93 | 45.84 | 47.11 | 30.26 | 20.02 |
| SGN-S | P+D pred. | 43.47 | 24.70 | 9.19 | 20.27 | 17.21 | 8.21 | 1.85 | 5.08 | 61.10 | 14.89 | 47.82 | 50.98 | 32.76 | 20.60 |
| OccFormer | P | 47.00 | 39.01 | 10.14 | 24.85 | 11.84 | 1.52 | 0.05 | 0.77 | 66.60 | 6.93 | 53.12 | 63.57 | 40.73 | 20.87 |
| MonoScene | P | 49.68 | 41.41 | 11.04 | 20.57 | 20.34 | 10.36 | 1.98 | 2.79 | 69.68 | 13.69 | 67.39 | 55.00 | 43.44 | 24.15 |
| **OneOcc (Ours)** | **P** | **58.60** | **48.28** | **16.34** | **25.98** | **30.68** | **19.35** | **12.87** | **14.16** | **72.89** | **23.11** | 67.89 | **64.77** | **49.58** | **32.23** |

**说明**: 同城 OneOcc 37.29 mIoU（+3.83 vs. MonoScene），跨城 32.23 mIoU（**+8.08**）。跨城的更大优势凸显其分布鲁棒性——DP-ER 与 AMoE-3D 的畸变感知先验跨地图/天气迁移、抑制全景混叠。注意 OneOcc 是**纯全景(P)**，却超过了用上 GT 深度的 VoxFormer。

### Table 4: Ablations on QuadOcc / 逐模块消融

| Variant | GDC | DP-ER | BGV | AMoE-3D | vehicle | pedestrian | road | building | vegetation | terrain | Prec. | Recall | IoU | mIoU |
|---------|-----|-------|-----|---------|---------|-----------|------|----------|-----------|---------|-------|--------|-----|------|
| (Q0) baseline | ✗ | ✗ | ✗ | ✗ | 8.15 | 1.59 | 55.66 | 12.88 | 26.10 | 10.78 | 62.10 | 69.28 | 48.69 | 19.19 |
| (Q1) + GDC | ✓ | ✗ | ✗ | ✗ | 11.76 | 2.48 | 53.73 | 15.07 | 22.95 | 11.51 | 64.05 | 66.08 | 48.78 | 19.58 |
| (Q2) + DP-ER | ✓ | ✓ | ✗ | ✗ | 11.82 | 2.51 | 54.16 | 15.39 | 23.40 | 12.08 | 65.42 | 65.01 | 48.38 | 19.89 |
| (Q3) + BGV | ✓ | ✓ | ✓ | ✗ | 12.07 | 2.77 | 54.29 | 15.76 | 24.19 | 12.74 | 66.56 | 64.70 | 48.68 | 20.30 |
| **(Q4) + AMoE-3D (full)** | ✓ | ✓ | ✓ | ✓ | **12.16** | **2.86** | 54.41 | **16.03** | 24.91 | **13.01** | 66.69 | 64.74 | 48.92 | **20.56** |

**关键发现**: 增益**可加**：+GDC(+0.39)、+DP-ER(+0.31)、+BGV(+0.41)、+AMoE-3D(+0.26)，共 19.19→20.56。GDC 稳采样、DP-ER 提供互补线索、BGV 降离散化偏差、AMoE-3D 锐化边缘/接触而不过平滑平坦区。

### Table 5: QuadOcc by Lighting (Day/Dusk/Night) / 不同光照

| Method | Day P | R | IoU | mIoU | Dusk P | R | IoU | mIoU | Night P | R | IoU | mIoU |
|--------|-------|---|-----|------|--------|---|-----|------|---------|---|-----|------|
| *LiDAR* | | | | | | | | | | | | |
| SSCNet | 67.12 | 62.95 | 48.12 | 14.03 | 52.97 | 76.34 | 45.50 | 14.88 | 57.81 | 75.30 | 48.60 | 9.51 |
| SSCNet-full | 80.56 | 58.52 | 51.28 | 16.04 | 73.79 | 64.34 | 52.37 | 17.14 | 73.64 | 67.31 | 54.24 | 11.56 |
| LMSCNet | 81.18 | 62.50 | 54.59 | 17.33 | 75.34 | 70.92 | 57.56 | 18.91 | 75.68 | 72.67 | 58.91 | 13.40 |
| OccRWKV | 72.84 | 63.83 | 51.56 | 16.25 | 66.43 | 59.30 | 45.63 | 15.42 | 74.10 | 58.31 | 48.44 | 12.61 |
| *Vision* | | | | | | | | | | | | |
| OccFormer | 46.23 | 77.08 | 40.64 | 12.82 | 41.79 | 79.45 | 37.72 | 14.46 | 49.49 | 72.18 | 41.56 | 10.21 |
| VoxFormer-S † | 55.80 | 73.96 | 46.64 | 11.44 | 37.42 | 82.81 | 34.73 | 8.95 | 47.10 | 82.41 | 42.80 | 11.84 |
| VoxFormer-T † | 61.77 | 75.02 | 51.24 | 12.91 | 42.25 | 84.68 | 39.25 | 11.06 | 51.93 | 84.62 | 47.45 | 12.88 |
| SGN | 61.54 | 68.71 | 48.06 | 17.94 | 54.07 | 69.06 | 43.53 | 15.19 | 46.83 | 52.29 | 32.81 | 10.58 |
| MonoScene | 65.47 | 69.94 | 51.09 | 18.58 | 47.62 | 77.10 | 41.72 | 15.14 | 51.35 | 66.47 | 40.78 | **14.20** |
| **OneOcc (Ours)** | 69.10 | 64.66 | 50.15 | **21.15** | 57.46 | 71.98 | 46.96 | **19.86** | 63.56 | 53.75 | 41.09 | 13.50 |

**说明**: OneOcc 在白天(21.15 vs. 18.58)、黄昏(19.86 vs. 15.14)领先最佳视觉基线；夜间 mIoU(13.50)略低于 MonoScene(14.20) 但精度更高（疑因 frustum 伪影抑制）。低光是纯视觉的共同短板。

### Table 6: A1 — AMoE-3D Experts $K$ (H3O-Heter) / 专家数消融

| $K$ | Gate | P | R | IoU | mIoU |
|-----|------|---|---|-----|------|
| 1 | – | 65.38 | 61.02 | 46.12 | 29.68 |
| 2 | GradEnergy3D | 66.79 | 63.69 | 48.37 | 31.25 |
| **4** | **GradEnergy3D** | 67.89 | **64.77** | **49.58** | **32.23** |
| 8 | GradEnergy3D | **68.08** | 64.39 | 49.46 | 32.03 |
| 4 | Uniform | 66.21 | 63.04 | 47.70 | 31.13 |
| 4 | Top-$k$ | 67.18 | 64.06 | 48.79 | 31.84 |

**说明**: $K{=}1\to4$ 持续提升；$K{=}8$ 精度略升但召回降（专家碎片化、路由噪声）。GradEnergy3D 门控优于 Uniform/Top-$k$。**$K{=}4$ + GradEnergy3D 为默认**。

### Table 7: A2 — FoV from 90° to 360° (H3O-Heter) / 视场角消融（训练一次，裁剪测试）

| FoV | Visible | Method | #Params | FPS↑ | Mem↓ | P↑ | R↑ | IoU↑ | mIoU↑ |
|-----|---------|--------|---------|------|------|----|----|------|-------|
| 90° | 25% | MonoScene | 146.13M | 13.07 | 1.72GB | 56.14 | 13.59 | 12.28 | 4.65 |
| 90° | 25% | **OneOcc** | **101.76M** | **24.19** | **1.65GB** | 58.65 | 15.43 | 13.92 | **7.35** |
| 180° | 50% | MonoScene | 146.13M | 12.63 | 1.78GB | 63.40 | 29.64 | 25.31 | 13.74 |
| 180° | 50% | **OneOcc** | **101.76M** | **19.24** | **1.70GB** | 62.68 | 31.62 | 26.61 | **17.27** |
| 270° | 75% | MonoScene | 146.13M | 10.08 | 2.09GB | 65.46 | 41.19 | 33.84 | 18.20 |
| 270° | 75% | **OneOcc** | **101.76M** | **16.13** | **1.76GB** | 65.79 | 47.53 | 38.11 | **24.28** |
| 360° | 100% | MonoScene | 146.13M | 8.29 | 2.60GB | 67.39 | 55.00 | 43.44 | 24.15 |
| 360° | 100% | **OneOcc** | **101.76M** | **14.30** | **1.82GB** | 67.89 | **64.77** | **49.58** | **32.23** |

**说明**: OneOcc mIoU 随 FoV 近单调增长（90°→360° 共 +24.88，相对 +77.2%），且 270°→360° 这"合环"增量尤大(+7.95)，证明 **360° 环形连续性的非线性收益**。精度相当但**召回优势主导**（360° 时 +9.77）。OneOcc@270°(24.28 mIoU,16.13 FPS,1.76GB) 即匹敌 MonoScene@360°(24.15,8.29,2.60GB)——约 2× 吞吐、少 0.84GB。

### Table 8: A3 — Single-grid vs. Bi-Grid (H3O-Heter) / 网格消融

| Voxelization | #Params | FPS↑ | Mem↓ | P↑ | R↑ | IoU↑ | Near mIoU↑ | Far mIoU↑ |
|--------------|---------|------|------|----|----|------|-----------|-----------|
| Cartesian-only | 101.73M | **14.32** | **1.70GB** | 62.54 | 63.18 | 45.84 | 36.42 | 30.56 |
| Cylindrical-only | 101.73M | 14.33 | 1.78GB | 63.65 | **65.30** | 47.57 | 34.15 | 31.00 |
| **Bi-Grid** | 101.76M | 14.30 | 1.82GB | **67.89** | 64.77 | **49.58** | **37.35** | **32.23** |

**说明**: 笛卡尔近场强(near 36.42)、柱坐标远场/召回强(R 65.30)。Bi-Grid 取两者之长：较笛卡尔 +5.35 Prec/+3.74 IoU、近/远场同涨(+0.93/+1.67)，开销极小(+0.03M 参数、~14.3 FPS、+0.12GB)。质心投影预计算于 dataloader 是开销小的关键。

### Table 9: A4 — Projection Path (QuadOcc) / 投影路径消融

| Projection Path | #Params | FPS↑ | Mem↓ | P↑ | R↑ | IoU↑ | mIoU↑ |
|-----------------|---------|------|------|----|----|------|-------|
| ER-only | 101.76M | 18.15 | 1.71GB | 65.77 | **64.81** | 48.47 | 20.03 |
| Raw-only | 101.97M | **25.18** | **1.55GB** | 66.42 | 62.86 | 47.70 | 19.76 |
| **Dual (ER+Raw)** | 189.83M | 15.46 | 2.14GB | **66.69** | 64.74 | **48.92** | **20.56** |

**说明**: 真实 PAL 光学下 Dual 最优(+0.53 mIoU vs. ER-only)，但参数 +86.5%、显存 +25.1%、吞吐 -14.8%；Raw-only 最快最省。**H3O 原生 ER 时 ER-only 即最佳折中**，故 H3O 默认禁 raw 分支。

### Table 10: A5 — Resolution Scaling (H3O-Heter) / 分辨率缩放

| Resolution | #Params | FPS↑ | Mem↓ | P↑ | R↑ | IoU↑ | mIoU↑ |
|------------|---------|------|------|----|----|------|-------|
| **64×64×8** | 101.76M | **14.30** | **1.82GB** | **67.89** | **64.77** | **49.58** | **32.23** |
| 128×128×16 | 105.17M | 5.01 | 10.71GB | 44.69 | 47.66 | 29.98 | 21.16 |

**说明**: 提分辨率仅 +3.41M 参数，却 FPS ~3× 降、显存 ~5.9× 升，且 IoU/mIoU 反而**下降**（更密离散化加剧类别不平衡与边界敏感）。具身约束下 $64\times64\times8$ 是甜点。

### Table 11: A6 — Calibration Robustness (H3O-Heter, mIoU) / 标定鲁棒性

| Method | Clean 0% | Both 1% | 2% | 5% | Intr. 1% | 2% | 5% | Extr. 1% | 2% | 5% |
|--------|----------|---------|----|----|----------|----|----|----------|----|----|
| MonoScene | 24.15 | 21.99 | 19.58 | 13.96 | 22.78 | 21.00 | 16.30 | 22.75 | 20.69 | 15.90 |
| **OneOcc** | **32.23** | **27.26** | **23.67** | **16.75** | **27.88** | **24.80** | **19.15** | **29.42** | **25.28** | **18.95** |
| MonoScene † | 18.97 | 18.86 | 18.64 | 16.48 | 18.71 | 18.28 | 15.74 | 19.01 | 19.17 | 18.92 |
| OneOcc † | 23.00 | 22.97 | 23.03 | 22.84 | 22.86 | 22.84 | 22.48 | 22.94 | 23.15 | 23.16 |

**说明**: 各噪声等级 OneOcc 都稳超 MonoScene（联合噪声 1/2/5% 各 +5.27/+4.09/+2.79）。用 5% 标定噪声增强重训(†)可几乎拉平退化曲线(OneOcc† 各档 ~23 mIoU)，但牺牲干净集精度(32.23→23.00)——经典鲁棒性-精度折中。

### Table 12: A7 — Temporal Aggregation / 时序聚合

| Setting | QuadOcc-val mIoU↑ | H3O-Heter mIoU↑ | Latency(ms)↓ | Memory(GB)↓ |
|---------|-------------------|------------------|--------------|-------------|
| OneOcc (single-frame) | 20.56 | 32.23 | **69.93** | **1.82** |
| + Temporal avg by GT pose (3 帧) | 20.92 | 33.74 | 69.93 | 1.82 |
| + Temporal BEVFormer-like attn (3 帧) | **21.18** | **34.25** | 78.60 | 2.35 |

**说明**: 3 帧时序聚合都涨点（GT 位姿对齐的均值为上界参考），BEVFormer 式注意力涨最多但延迟/显存升。因目标是**免里程计、低延迟、易部署**，默认仍用单帧（避免漂移累积、无需运动历史）。

### Table 13/14: Parameter Breakdown / 参数分解（附录）

| Method | 2D encoder(s) | 3D decoder | GDC+fusion+gates | Total |
|--------|---------------|-----------|------------------|-------|
| MonoScene (QuadOcc) | 1×132M | 16.9M | – | 148.9M |
| **OneOcc (QuadOcc)** | 2×87.8M | **12.3M** | ≈0.27M | 189.8M |
| MonoScene (H3O) | 1×132M | 16.9M | – | 148.9M |
| **OneOcc (H3O)** | 1×87.8M | **12.3M** | ≈0.27M | **101.7M** |

**说明**: OneOcc 单个 DP-ER 编码器比 MonoScene 的 2D 编码器轻 33%(132M→87.8M)，DWLite3D 解码器轻 27%(16.9M→12.3M)；几何模块(GDC/BGV 融合/AMoE 门控)仅 ~0.27M(<0.2%)。QuadOcc 双投影使总参增约 26%（来自两条轻编码器并行，可被 GPU 并行摊销）；H3O 去掉 raw 分支后反而比 MonoScene 轻 33%(→101.7M)。

### Table 15: Jetson AGX Orin Runtime / 边端运行时（附录）

| Method | 2D Enc.+Dec. | Lift2Cart | Cart2Polar | 3D Dec. | Total(ms) | FPS |
|--------|--------------|-----------|------------|---------|-----------|-----|
| MonoScene | 61.97 | 2.46 | n.a. | 33.58 | 98.01 | 10.20 |
| **OneOcc** | **53.86** | 2.48 | 2.40 | **14.57** | **73.32** | **13.64** |

**说明**: Jetson Orin(MAX, INT8 2D + FP16 其余) 上，尽管多了 Cart2Polar 重采样(2.40ms)，OneOcc 仍 1.34× 更快：2D 分支更轻(53.86 vs. 61.97)，3D 解码器大降(14.57 vs. 33.58ms)。佐证其边端可部署性。

---

## 实验

### 数据集 / 基准

| 基准 | 规模 | 特点 | 用途 |
|------|------|------|------|
| [[QuadOcc]] | ~24K 连续帧（stride-5 训练），20 试 | **真实**四足、第一人称 360°、校园 day/dusk/night、序列异质划分；6 类、$64{\times}64{\times}8$ 栅格(0.4m)；Livox Mid-360 LiDAR + PAL 同步建真值 | 训练/测试 |
| [[Human360Occ\|H3O]] | 16 张 CARLA 地图、160 序列、两分辨率 | **仿真**人本视角 360°、模拟步态、多天气光照；含 RGB/深度/占据/位姿；同城(Homo)/跨城(Heter)标准划分；10 类 | 训练/测试 |

### 实现细节

- **前端**: PAL Taylor/OCam 标定展开；DP-ER 双编码器（H3O 仅 ER）；ER 路 $352\times1216$、Raw 路 $512\times512$；主结果 $64\times64\times8$@0.4m 栅格，垂直范围 $z\in[-2,1.2]$m
- **3D 解码器**: 深度可分离 DWLite3D UNet（三级），分层 AMoE-3D（$K{=}4$，GradEnergy3D 门控）
- **损失**: CE + SCAL(sem/geo) + FP（沿用 MonoScene，去 relation loss）；stride{1,2,4} 深监督
- **基线**: 重训 LiDAR SSC（SSCNet/LMSCNet/OccRWKV）、适配视觉 SSC（OccFormer/VoxFormer/SGN/MonoScene 经标定展开到 ER）
- **硬件/效率**: 单卡 RTX 4090 训练；推理 608×1216、batch 1，69.93ms(14.30 FPS, FP32) / 52.84ms(18.92 FPS, 混精度)，峰值 1.82GB；Jetson AGX Orin 13.64 FPS

### 关键实验结论

- **QuadOcc(真实)**: OneOcc 20.56 mIoU，纯相机超过最佳 LiDAR(LMSCNet 18.44)与最佳视觉(MonoScene 19.19)甚至加 LiDAR 的 SGN†(19.72)。
- **H3O(仿真)**: 同城 37.29(+3.83)、跨城 32.23(**+8.08**) mIoU，跨城优势更大 → 分布鲁棒。
- **光照**: 白天/黄昏领先，夜间略逊 MonoScene（纯视觉低光短板）。
- **消融**: GDC/DP-ER/BGV/AMoE-3D 增益可加(19.19→20.56)；$K{=}4$ + GradEnergy3D 最优；Bi-Grid 近乎免费地同时改善近/远场；360° 全环最关键；$64\times64\times8$ 是效率甜点；对标定噪声更鲁棒。
- **效率**: 比 MonoScene 少 30.3% 参数、更高 mIoU/GB、桌面与边端均更快。

---

## 批判性思考

### 优点
1. **问题定义新且贴具身**: 把 SSC 从"轮式前向"推广到"足式全周向 + 步态抖动"，并配套发布真实(QuadOcc)与仿真(H3O)两个 360° 占据基准，填补了该设定的数据空白，社区价值高。
2. **模块设计动机-机制-证据闭环清晰**: DP-ER/BGV/GDC/AMoE-3D 每个都有明确动机、公式与对应消融（Table 4/6/8/9），且 FoV 实验(Table 7)、Bi-Grid 近/远场拆分等给出了细粒度证据，论证扎实。
3. **效率友好、可部署**: 纯相机、单帧、免里程计，101.76M 参数、~14-19 FPS、<2GB 显存，且在 Jetson Orin 上实测 1.34× 快于 MonoScene，落地性强；代码与数据承诺开源。

### 局限性
1. **绝对精度仍偏低**: QuadOcc mIoU 仅 20.56、H3O 跨城 32.23，远未到稠密可靠占据；稀有/小目标(pedestrian、pole、two_wheeler)IoU 极低，长尾问题严重。
2. **更高分辨率反而退化**: $128^2{\times}16$ 下 mIoU 大跌(32.23→21.16)且显存暴涨 ~6×，说明方法对细粒度几何的可扩展性受限，目前靠"低分辨率甜点"回避而非真正解决。
3. **依赖准确标定与单帧**: 标定噪声下虽更鲁棒但仍明显退化(5% 联合噪声降到 16.75)；单帧设计放弃了时序信息，时序聚合(Table 12)虽能涨点却被以"部署复杂度"为由排除，留有性能余量未取。夜间纯视觉仍逊于 LiDAR/MonoScene。
4. **QuadOcc 真值半自动且规模有限**: 真值由 LiDAR-图像标签转移 + 开放词表分割构建，标注噪声不可忽视；真实数据仅校园单场景、6 类。

### 潜在改进方向
1. 引入轻量时序/world-model 把 Table 12 的时序增益以免里程计方式纳入，缓解步态抖动与遮挡。
2. 针对长尾小目标做类别均衡/边界感知损失或显式实例先验，提升 pedestrian/pole 等稀有类。
3. 让分辨率自适应（近场细、远场粗）或八叉树/稀疏体素，解决高分辨率退化与显存爆炸。
4. 扩展到更多本体（人形/轮足）、室内外极端条件与多模态（轻 LiDAR/深度）补强夜间与远场。

### 可复现性评估
- [x] 代码开源（承诺，https://github.com/MasterHow/OneOcc）
- [ ] 预训练模型（未明确声明）
- [x] 训练细节完整（正文 + 附录给出网格/分辨率/损失/超参与逐模块消融）
- [x] 数据集可获取（QuadOcc 与 H3O 承诺公开，含标准划分）

---

## 速查卡片

> [!summary] OneOcc: 单全景相机为足式机器人做 360° 语义占据
> - **核心**: 纯视觉、单帧、免里程计的全周向 SSC；用 DP-ER 双投影 + BGV 双网格 + GDC 步态补偿 + 分层 AMoE-3D，克服全景畸变与步态抖动。
> - **方法**: PAL Taylor 标定展开 → ER/Raw 双编码器 → GDC（lifting 前零初始化位移校正）→ 笛卡尔+柱坐标双网格 View2View 提升与融合 → DWLite3D + AMoE-3D($K{=}4$,GradEnergy 门控) → CE+SCAL+FP 损失。
> - **结果**: QuadOcc 20.56 mIoU（超 LiDAR LMSCNet 18.44）；H3O 同城 37.29(+3.83)/跨城 32.23(+8.08)；101.76M 参数、~14-19 FPS、<2GB，Jetson 1.34× 快于 MonoScene。
> - **数据**: 新发布 QuadOcc（真实四足 360°）与 Human360Occ/H3O（CARLA 人本 360°）。
> - **代码**: https://github.com/MasterHow/OneOcc

---

*笔记创建时间: 2026-06-29*
