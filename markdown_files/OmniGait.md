---
title: "MMGait: Towards Multi-Modal Gait Recognition"
method_name: "OmniGait"
authors: [Chenye Wang, Qingyuan Cai, Saihui Hou, Aoqi Li, Yongzhen Huang]
year: 2026
venue: CVPR
tags: [gait-recognition, multi-modal, benchmark, cross-modal-retrieval, biometrics, LiDAR, radar, omni-modal]
zotero_collection: _inbox
image_source: online
arxiv_html: https://arxiv.org/html/2604.15979v1
created: 2026-06-29
---

# MMGait: Towards Multi-Modal Gait Recognition

## 元信息

| 项目 | 内容 |
|------|------|
| 作者 | Chenye Wang†, Qingyuan Cai†, Saihui Hou\*, Aoqi Li, Yongzhen Huang（†共同一作，\*通讯） |
| 机构 | 北京师范大学人工智能学院、WATRIX.AI |
| 会议 | CVPR 2026 |
| 类别 | 步态识别 / 多模态基准（locomotion） |
| 日期 | 2026-04（arXiv v1） |
| 项目主页 | https://github.com/BNU-IVC/MMGait |
| 链接 | [arXiv](https://arxiv.org/abs/2604.15979) / [Code](https://github.com/BNU-IVC/MMGait) |

---

## 一句话总结

> 构建覆盖 5 类异构传感器、12 种模态、725 人、33.4 万序列的多模态步态基准 MMGait，并提出统一处理单模态/跨模态/多模态三类任务的新任务 Omni Multi-Modal Gait Recognition 与简洁强基线 OmniGait。

---

## 核心贡献

1. **MMGait 大规模多模态步态基准**: 整合 RGB、深度、红外、LiDAR、4D 毫米波雷达共 **5 种异构传感器**，处理为 **12 种模态**，含 **725 个受试者、334,060 条序列**，是迄今规模与模态最全面的步态数据集之一，覆盖几何、光度、运动三大域。
2. **三范式系统评测**: 在 [[单模态识别|Single-Modal]]、[[跨模态识别|Cross-Modal]]、[[多模态识别|Multi-Modal]] 三种设定下做全面 benchmark，定量刻画各模态的鲁棒性与互补性（含 intra-sensor 与 inter-sensor 融合）。
3. **新任务 OMGR + 基线 OmniGait**: 提出 [[Omni Multi-Modal Gait Recognition|OMGR]] 任务——单一模型接受任意模态查询、检索任意模态目标；并给出简洁但强力的基线 [[OmniGait]]，在共享嵌入空间中统一三种范式，部分设定下甚至超过专用模型。

---

## 问题背景

### 要解决的问题
现实步态识别系统需要**多传感器协作**与**跨模态检索**（如用 RGB 查 LiDAR），但绝大多数现有方法只聚焦 RGB 衍生模态（[[silhouette|轮廓]]、[[pose|姿态]]），无法支撑实际部署中"任意模态查询、任意模态检索"的灵活需求。

### 现有方法的局限
- **RGB 衍生模态固有缺陷**: 缺乏 3D 感知，对遮挡、雨雾、低照度等条件脆弱。
- **现有多模态基准范围窄**: 即便如 [[SUSTech1K]]、[[FreeGait]] 也基本只含 **RGB + LiDAR**，无法研究异构模态交互与统一跨传感器检索。
- **三种范式被割裂研究**: 单模态、跨模态、多模态各自为政，模态组合受限，单一用途模型无法满足真实系统的灵活查询。

### 本文的动机
- 用 **5 种横跨可见光—红外、2D 外观—3D 几何、低成本—高成本的传感器**采集互补线索，把步态识别推向真实多传感器场景。
- 提出 **OMGR**：一个统一框架同时具备三种能力——通用模态识别（任意单模态可靠识别）、自适应多模态融合（利用成对模态互补）、模态无关检索（任意模态对双向匹配）。
- 假设：**学习跨模态共享嵌入空间**即可在一个轻量模型中兼顾三种范式，避免为每种组合训练独立模型。

---

## 方法详解

本文有两条主线：(A) **MMGait 基准的构建与评测协议**；(B) **OmniGait 统一框架**。

### MMGait 基准构建

**传感器配置（5 类，统一 10 Hz 采样）**
- **RGB 相机**: 可见光纹理外观，支持 silhouette、2D/3D pose、event 等表征；作为大多数研究的参考模态。
- **IR 红外相机**: 捕捉 >780 nm 红外（设备用 940 nm 窄带以避开 LiDAR 激光干扰），低光/夜间可靠，对光照不变，利于 RGB-IR 跨模态检索。
- **深度相机**: ToF 原理测量细粒度 3D 几何，提供稠密身体轮廓、弱纹理依赖，桥接 2D 与 3D 表征。
- **LiDAR 扫描仪**: 激光反射测距，输出稠密高精度点云，3D 结构对光照/背景鲁棒。
- **4D 毫米波雷达**: 发射 FMCW 调频连续波，时频域分析反射信号，远距运动感知、强穿透（雾/雨/部分遮挡），低成本但稀疏。

> 硬件：RGB 与深度共用 Orbbec Gemini 2 XL（天然同步）；IR 为 940 nm 工业相机；LiDAR 为 Ouster OS0-7；4D 雷达为 GPAL Ares-R7861。

**采集设置（见 Figure 2）**: 受试者沿**五角星路线**行走，提供全圆周覆盖，从 $0^{\circ}$ 到 $360^{\circ}$ 每 $36^{\circ}$ 共 **10 个视角**。采集横跨整月、晨至晚、晴/阴天，自然引入多样光照。每人三种条件：正常 (NM) 两次、背包 (BG) 一次、换装 (CL) 一次，且**要求用自己的背包、自行更换上下装**，增强跨条件真实性。理想每人 480 序列（10 视角 × (2+1+1) 条件 × 12 模态），过滤低质/失效后得 725 人、334,060 序列。

**评测协议**: 训练集 200 人、测试集 525 人（身份完全不重叠以严格评测泛化）。
- **单模态**: gallery 与 query 同模态，NM-01 为 gallery，NM-02/BG-01/CL-01 为 query。
- **跨模态**: gallery 与 query 不同模态（NM-01 模态A 为 gallery，另一模态 NM-02/BG-01/CL-01 为 query）。
- **多模态**: 与单模态同样的 gallery-query 划分，但每个表征由两种模态特征融合得到。
- 指标：Rank-1 (R1) 与 mAP；结果在全部 $10\times10$ 跨视角对上平均（排除同视角对）。

### OmniGait 框架

[[OmniGait]] 在统一架构内灵活处理单模态、跨模态、多模态识别，考虑 **9 种图像化模态**（RGB、RGB silhouette、2D pose 热图、event、IR、IR silhouette、depth、LiDAR 投影深度、radar 投影深度）。多模态融合实验以 **RGB silhouette 为锚定模态**，逐一与其余模态融合。框架由三部分组成（见 Figure 5）：

#### 模块1: Modal-Specific Encoding（模态专属编码）

**设计动机**: 不同传感器捕捉人体运动的不同物理线索，需在对齐/融合前保留各自特性。

**具体实现**: 对每个模态 $m\in\{1,\dots,M\}$ 用独立编码器 $\varepsilon_m(\cdot)$ 把原始输入 $\mathbf{X}_m$ 映射为特征图（公式1），保留模态专属特征。

#### 模块2: Cross-Modal Fusion（跨模态融合）

**设计动机**: 轻量地整合多模态互补线索，并通过门控加权自适应平衡各模态贡献。

**具体实现**:
- 把两个模态特征 $f_i,f_j$ 沿通道拼接后用 $1\times1$ 卷积得融合表征（公式2）。
- 门控 $G(\cdot)$（全局平均池化 + 两层 $1\times1$ 卷积 + 非线性）生成自适应权重 $\mathbf{w}\in\mathbb{R}^2$（公式3）。
- 残差结合门控加权特征得最终融合特征（公式4）。该融合模块**在所有模态组合间共享**，动态重标定各模态贡献的同时保留共享空间语义。

#### 模块3: Shared Representation Learning（共享表示学习）

**设计动机**: 在统一嵌入空间内对齐单模态与多模态特征，支持多样识别任务。

**具体实现**:
- 把单模态特征 $\{f_m\}$ 与融合双模态特征 $\{f_{i,j}^{\text{fused}}\}$ 一起送入**共享残差骨干** $E(\cdot)$ 学习模态不变表征（公式5）。
- **关键机制**: 所有单模态与融合特征**在同一 batch 内共同处理**，使 $E(\cdot)$ 内部的 [[BatchNorm|批归一化]] 层学到跨模态更鲁棒、更泛化的统计量——这正是跨模态对齐隐式发生的关键（§3.2 也指出共享层 BN 统计强烈影响跨模态检索）。
- 特征经 [[Temporal Pooling|时间池化 (TP)]] 与 [[Horizontal Pyramid Pooling|水平金字塔池化 (HPP)]] 聚合，再过 FC 与 [[BNNeck]]，得最终推理特征 $\in\mathbb{R}^{C_3\times P}$（$P$ 为 HPP 划分数）。
- 训练用 [[Cross-Entropy Loss|分类损失]] $L_{CE}$ + [[Triplet Loss|三元组损失]] $L_{triplet}$ 联合优化，兼顾判别性与跨模态对齐。

**推理流水线**: 训练时所有模态联合参与，推理时只激活所需的模态专属编码器 $\varepsilon_m$，共享编码器 $E$ 接受任意模态特征，使 OmniGait 无额外开销即可兼容单/跨/多模态任务。

**实现配置**: 序列长 $T=16$，分辨率统一 $64\times64$；模态编码后 $C_1\times H_1\times W_1=128\times64\times64$，共享骨干后 $512\times16\times16$，最终推理特征 $256\times16$；batch 为 $8\times4$（8 身份 × 4 序列）。

### 关键公式与机制

#### 公式1: [[Modal-Specific Encoding|模态专属编码]]

$$
f_{m}=\varepsilon_{m}(\mathbf{X}_{m}),\quad f_{m}\in\mathbb{R}^{T\times C_{1}\times H_{1}\times W_{1}}
$$

**含义**: 每个模态用独立编码器把原始输入映射为时空特征图，保留模态固有物理特性。

**符号说明**:
- $\mathbf{X}_m$: 模态 $m$ 的原始输入；$\varepsilon_m$: 模态专属编码器
- $T$: 帧数；$C_1,H_1,W_1$: 特征图通道/高/宽

#### 公式2: 通道拼接融合

$$
f_{i,j}=\text{Conv}_{1\times 1}\big(\text{Concat}([f_{i},f_{j}],\text{dim}=1)\big)
$$

**含义**: 把两个模态特征沿通道维拼接后经 $1\times1$ 卷积压缩为统一融合表征。

**符号说明**:
- $f_i,f_j$: 两个模态的专属特征；$\text{Concat}(\cdot,\text{dim}=1)$: 通道维拼接

#### 公式3: 门控权重

$$
\mathbf{w}=\text{Softmax}\big(G(f_{i,j})\big),\quad\mathbf{w}\in\mathbb{R}^{2}
$$

**含义**: 门控网络生成两个模态的自适应贡献权重，刻画不同空间上下文中各模态的相对重要性。

**符号说明**:
- $G(\cdot)$: 全局平均池化 + 两层 $1\times1$ 卷积 + 非线性
- $\mathbf{w}$: 归一化后的两模态权重

#### 公式4: 门控残差融合

$$
f_{i,j}^{\text{fused}}=f_{i,j}+\sum_{m=1}^{2}\mathbf{w}_{m}f_{m},\quad f_{i,j}^{\text{fused}}\in\mathbb{R}^{T\times C_{1}\times H_{1}\times W_{1}}
$$

**含义**: 用门控权重对两模态特征加权求和，再以残差方式加回融合特征，动态重标定模态贡献且保留共享空间语义。

**符号说明**:
- $\mathbf{w}_m$: 模态 $m$ 的门控权重；$f_m$: 模态 $m$ 的专属特征

#### 公式5: 共享表示学习

$$
\mathbf{F}=E\big(\{f_{m},\,f_{i,j}^{\text{fused}}\}\big),\quad\mathbf{F}\in\mathbb{R}^{T\times C_{2}\times H_{2}\times W_{2}}
$$

**含义**: 单模态特征与融合双模态特征一并送入共享残差骨干，学习模态不变表征；同 batch 联合处理让 BN 学到跨模态鲁棒统计。

**符号说明**:
- $E(\cdot)$: 共享残差骨干；$C_2,H_2,W_2$: 共享骨干输出维度

#### 公式6: [[Cross-Modal Triplet Loss|跨模态三元组损失]]（跨模态检索基线）

$$
L_{\text{cross-triplet}}=\tfrac{1}{2}\Big(L_{\text{triplet}}(A_{\text{modal1}},P_{\text{modal2}},N_{\text{modal2}})+L_{\text{triplet}}(A_{\text{modal2}},P_{\text{modal1}},N_{\text{modal1}})\Big)
$$

**含义**: 锚点取自一个模态、正负样本取自另一模态，双向对称地拉近跨模态同身份、推远异身份，促进模态不变表征。

**符号说明**:
- $A,P,N$: 锚点 / 正样本 / 负样本；下标表示所属模态

#### 公式7-9: 分类损失与总损失

$$
L_{\text{ce}}^{\text{modal}}=-\sum_{i=1}^{c}y_{i}\log(\hat{y}_{i})
$$

$$
L_{\text{ce}}=\tfrac{1}{2}\big(L_{\text{ce}}^{\text{modal1}}+L_{\text{ce}}^{\text{modal2}}\big)
$$

$$
L_{\text{total}}=\tfrac{1}{2}\big(L_{\text{cross-triplet}}+L_{\text{ce}}\big)
$$

**含义**: 每个模态的身份分类交叉熵（公式7），两模态平均（公式8），最终与跨模态三元组损失等权组合为总损失（公式9）。

**符号说明**:
- $c$: 身份类别数；$y_i,\hat{y}_i$: 第 $i$ 类的真值与预测概率

---

## 关键图表

<!-- 图片均使用 arXiv HTML 在线链接，未本地化 -->

### Figure 1: Highlights of MMGait / 基准亮点

![Figure 1](https://arxiv.org/html/2604.15979v1/x1.png)

**说明**: MMGait 整合 5 种传感器、12 种模态，横跨多视角、多模态表征、多协变量条件（NM/BG/CL）。直观展示数据集"全面、精确、可扩展"三大特性。

### Figure 2: Collection Setup / 采集设置

![Figure 2](https://arxiv.org/html/2604.15979v1/x2.png)

**说明**: 五角星行走路线提供全圆周覆盖，10 个均匀视角（$0^{\circ}\sim360^{\circ}$，$36^{\circ}$ 间隔），保证丰富多视角信息以支撑跨视角评测。

### Figure 3: Sequence Length Distribution / 序列长度分布

![Figure 3](https://arxiv.org/html/2604.15979v1/x3.png)

**说明**: 各传感器的序列长度分布，反映不同模态时序覆盖差异，是评估时序建模的重要参考。

### Figure 4: Pairwise Cross-Modal Retrieval (baseline) / 五传感器两两跨模态检索

![Figure 4](https://arxiv.org/html/2604.15979v1/x4.png)

**说明**: 训练 10 个双向跨模态模型，对角线为单模态结果。跨模态模型在两方向无明显不对称，但条件越难差距越大：NM-R1 均值 55.5%、BG-R1 降至 49.5%、CL-R1 仅 28.4%——遮挡与换装显著放大模态间差异。

### Figure 5: OmniGait Framework / OmniGait 整体框架

![Figure 5](https://arxiv.org/html/2604.15979v1/x5.png)

**说明**: OmniGait 三段式结构——模态专属编码（保留物理特性）→ 跨模态融合（门控加权聚合）→ 共享表示学习（统一嵌入空间），灵活支撑单/跨/多模态识别。

### Figure 6: Pairwise Cross-Modal Retrieval (OmniGait) / OmniGait 两两跨模态检索

![Figure 6](https://arxiv.org/html/2604.15979v1/x6.png)

**说明**: OmniGait 在 NM/BG 设定下跨模态检索优于专用跨模态基线（如 RGB→Depth 在 NM 达 87.0% R1，远超专用模型 77.5%），得益于共享骨干学到的模态鲁棒统计；但 CL 条件下仍偏低。

### Figure 7-8: Visualization of Gait Sequences / 步态序列可视化（附录）

![Figure 7](https://arxiv.org/html/2604.15979v1/x7.png)

![Figure 8](https://arxiv.org/html/2604.15979v1/x8.png)

**说明**: 展示全部传感器模态、10 个视角、三种行走条件（NM/BG/CL）下的步态序列样例，直观呈现各模态的外观与几何差异。

### Table 1: Sensor Specifications / 传感器与模态规格

| Sensor | Range (m) | Resolution | Modality |
|--------|-----------|------------|----------|
| RGB Camera | 0.4–10 | 1280×800 | RGB, Silhouette, 2D Pose, 3D Pose, Event |
| IR Camera | 0.4–10 | 1280×700 | IR, Silhouette |
| Depth Camera | 0.4–10 | 1280×800 | Depth |
| LiDAR Scanner | 0.5–100 | 128-beam | Point Cloud, Projected Depth |
| 4D Radar System | 0.2–170 | – | Point Cloud, Projected Depth |

**说明**: 5 类传感器及其衍生的 12 种模态。LiDAR/雷达感知距离远（最高 170 m），RGB 衍生模态最丰富（5 种）。

### Table 2: Dataset Comparison / 与现有步态数据集对比

| Dataset | #Subject | #Seq. | #View | Sensor |
|---------|----------|-------|-------|--------|
| CASIA-B (2006) | 124 | 13,640 | 11 | RGB Camera |
| SZTAKI-LGA (2016) | 28 | 11 | 1 | LiDAR |
| PCG (2020) | 30 | 60 | 1 | LiDAR |
| GREW (2021) | 26,345 | 128,671 | 882 | RGB Camera |
| Gait3D (2022) | 4,000 | 25,309 | 39 | RGB Camera |
| CCPG (2023) | 200 | 16,566 | 10 | RGB Camera |
| CCGR (2024) | 970 | 1,580,617 | 33 | RGB Camera |
| CASIA-C (2006) | 153 | 1,530 | 1 | RGB, IR Camera |
| TUM-GAID (2012) | 305 | 3,370 | 1 | RGB, Depth, Audio |
| SUSTech1K (2023) | 1,050 | 25,239 | 12 | RGB, LiDAR |
| FreeGait (2024) | 1,195 | 11,950 | 1 | RGB, LiDAR |
| **MMGait (Ours)** | **725** | **334,060** | **10** | **RGB, Depth, IR, LiDAR, 4D Radar** |

**说明**: 在多模态步态数据集中，MMGait 序列数(334,060)远超其余多模态集，且**唯一同时覆盖 RGB/深度/IR/LiDAR/4D 雷达 5 类传感器**，模态多样性最高。

### Table 3: Silhouette & Pose Baselines / 轮廓与姿态基线

| Modality | Method | NM R1 | NM mAP | BG R1 | BG mAP | CL R1 | CL mAP |
|----------|--------|-------|--------|-------|--------|-------|--------|
| Silhouette | GaitSet | 95.2 | 96.8 | 90.4 | 93.4 | 48.9 | 60.7 |
| Silhouette | GaitPart | 96.4 | 97.6 | 92.2 | 94.9 | 52.0 | 63.5 |
| Silhouette | GaitGL | 93.1 | 95.1 | 89.2 | 92.4 | 49.5 | 60.8 |
| Silhouette | **GaitBase** | **98.5** | **99.0** | 96.4 | 97.6 | **61.0** | **71.3** |
| Silhouette | **DeepGaitV2-P3D** | **98.7** | **99.1** | **97.0** | **98.1** | 58.7 | 69.1 |
| 2D Pose | GaitGraph | 31.5 | 44.8 | 23.5 | 36.0 | 13.7 | 24.4 |
| 2D Pose | GaitGraph2 | 15.8 | 23.4 | 12.0 | 19.1 | 7.1 | 13.6 |
| 2D Pose | GaitTR | 72.6 | 80.3 | 56.6 | 67.0 | 38.1 | 50.3 |
| 2D Pose | GPGait | 80.4 | 85.7 | 66.0 | 74.0 | 40.9 | 52.1 |
| 2D Pose | **GPGait++** | **84.7** | **89.3** | **71.6** | **79.1** | **51.2** | **61.6** |
| 2D Pose | SkeletonGait | 82.7 | 87.5 | 71.0 | 78.2 | 43.4 | 54.2 |

**说明**: silhouette 全面优于 2D pose；GaitBase / DeepGaitV2-P3D 在 silhouette 内最强，CL（换装）条件普遍掉到 ~60%，是最难协变量。pose 类中 GPGait++ 最优。

### Table 4: Emerging Modalities / 各传感器模态识别性能

| Sensor | Modality | NM R1 | NM mAP | BG R1 | BG mAP | CL R1 | CL mAP |
|--------|----------|-------|--------|-------|--------|-------|--------|
| RGB | **RGB** | **99.7** | **99.8** | **99.1** | **99.4** | 60.7 | 71.5 |
| RGB | Silhouette | 98.5 | 99.0 | 96.4 | 97.6 | **61.0** | 71.3 |
| RGB | 2D Pose | 84.7 | 89.3 | 71.6 | 79.1 | 51.2 | 61.6 |
| RGB | 3D Pose | 42.8 | 54.3 | 30.2 | 41.6 | 22.2 | 33.0 |
| RGB | Event | 70.1 | 79.0 | 62.0 | 72.1 | 13.5 | 23.1 |
| IR | **IR** | **99.7** | **99.8** | **99.0** | **99.3** | **78.8** | **85.7** |
| IR | Silhouette | 96.8 | 97.9 | 93.1 | 95.4 | 46.7 | 58.8 |
| Depth | **Depth** | **95.1** | **96.9** | **91.0** | **94.2** | **70.3** | **79.7** |
| LiDAR | Projected Depth | 94.1 | 96.2 | 90.8 | 94.0 | 74.5 | 82.6 |
| LiDAR | **Point Cloud** | **97.0** | **98.2** | **94.8** | **96.9** | **76.6** | **84.7** |
| 4D Radar | Projected Depth | 22.4 | 36.3 | 17.6 | 30.7 | 16.0 | 28.8 |
| 4D Radar | **Point Cloud** | **39.5** | **53.9** | **31.6** | **45.6** | **28.3** | **43.3** |

**说明**: RGB/silhouette 总体最高；**IR 在换装(CL)条件下最强(78.8%)**——红外抑制纹理/颜色变化、保留结构线索。Depth/LiDAR 在 CL 下也稳健。4D 雷达因稀疏性能最弱，更适合做辅助模态。IR silhouette 相对 RGB silhouette 偏低，源于分割器在 RGB 上训练、对红外存在域偏移。

### Table 5: RGB-Centered Cross-Modal Retrieval / RGB 为中心的跨模态检索

| Probe → Gallery | NM R1 | NM mAP | BG R1 | BG mAP | CL R1 | CL mAP |
|-----------------|-------|--------|-------|--------|-------|--------|
| RGB(Sil.) → IR(Sil.) | 95.7 | 97.1 | 90.8 | 93.8 | 45.3 | 57.5 |
| IR(Sil.) → RGB(Sil.) | 95.6 | 97.0 | 90.6 | 93.6 | 45.9 | 57.8 |
| RGB(Sil.) → Depth | 76.1 | 83.1 | 67.3 | 76.1 | 30.1 | 43.3 |
| Depth → RGB(Sil.) | 77.5 | 84.5 | 60.7 | 71.1 | 29.3 | 42.5 |
| RGB(Sil.) → LiDAR(Proj.) | 76.8 | 80.8 | 68.4 | 74.6 | 35.6 | 47.5 |
| LiDAR(Proj.) → RGB(Sil.) | 75.8 | 80.6 | 64.8 | 72.2 | 35.3 | 47.1 |
| RGB(Sil.) → 4D Radar(Proj.) | 3.6 | 8.9 | 2.8 | 7.6 | 2.2 | 6.7 |
| 4D Radar(Proj.) → RGB(Sil.) | 2.3 | 6.8 | 1.6 | 5.5 | 1.6 | 5.4 |

**说明**: RGB↔IR 检索最易（视觉相似，~95%）；RGB↔Depth、RGB↔LiDAR 较难（尤其 CL）；**RGB↔4D 雷达几乎失效（<4%）**，雷达稀疏不适合做主检索源，宜作辅助。

### Table 6: Intra-/Inter-Sensor Fusion (baseline) / 多模态融合（基线）

| Modality | NM R1 | NM mAP | BG R1 | BG mAP | CL R1 | CL mAP |
|----------|-------|--------|-------|--------|-------|--------|
| **Intra-Sensor** RGB(Sil.) | 98.5 | 99.0 | 96.1 | 97.6 | 61.0 | 71.3 |
| +RGB(Event) | 98.9 | 99.1 | 98.1 | 98.6 | 62.1 (+1.1) | 72.3 (+2.0) |
| +RGB(Pose) | 98.6 | 99.0 | 97.1 | 98.0 | 64.0 (+3.0) | 73.5 (+2.2) |
| **Inter-Sensor** RGB(Sil.) | 98.5 | 99.0 | 96.1 | 97.6 | 61.0 | 71.3 |
| +Depth | 99.5 | 99.7 | 98.8 | 99.2 | **80.3 (+19.3)** | 86.7 (+15.4) |
| +LiDAR(Proj.) | 99.7 | 99.8 | 98.8 | 99.3 | **80.7 (+19.7)** | 87.1 (+15.8) |
| LiDAR(Point Clouds) | 97.0 | 98.2 | 94.8 | 96.9 | 76.6 | 84.7 |
| +Radar(Point Clouds) | 96.3 | 97.9 | 93.9 | 96.3 | 80.4 (+3.8) | 87.3 (+2.6) |

**说明**: **跨传感器(inter-sensor)融合收益最大**——RGB silhouette + 深度/LiDAR 在 CL 上 R1 暴涨 +19.3/+19.7%，几何深度线索弥补 RGB 外观歧义；intra-sensor（同 RGB 内加 event/pose）收益温和(+1~3%)；LiDAR 点云 + 雷达点云在 CL 上再 +3.8%，互补 3D 表征。

### Table 7: OmniGait Single-Modal / OmniGait 单模态识别

| Sensor | Modality | NM R1 | NM mAP | BG R1 | BG mAP | CL R1 | CL mAP |
|--------|----------|-------|--------|-------|--------|-------|--------|
| RGB | **RGB** | **99.7** | **99.8** | **99.0** | **99.3** | 58.5 | 69.9 |
| RGB | Silhouette | 98.1 | 98.7 | 95.7 | 97.2 | 44.9 | 57.6 |
| RGB | 2D Pose | 73.8 | 80.8 | 58.8 | 68.5 | 26.8 | 38.0 |
| RGB | Event | 87.3 | 91.5 | 78.7 | 85.0 | 15.2 | 24.9 |
| IR | **IR** | **99.4** | **99.6** | **97.6** | **98.4** | **59.2** | **70.7** |
| IR | Silhouette | 95.4 | 96.9 | 90.7 | 93.7 | 32.4 | 45.1 |
| Depth | **Depth** | **90.1** | **93.2** | **80.5** | **86.2** | **36.8** | **49.5** |
| LiDAR | **Projected Depth** | **93.1** | **95.5** | **88.6** | **92.4** | **43.3** | **57.0** |
| 4D Radar | **Projected Depth** | **15.8** | **27.7** | **12.7** | **23.8** | **11.5** | **22.6** |

**说明**: 统一模型 OmniGait 在 9 模态上与专用模型（Table 4）相当，但 **CL 条件下普遍掉点**（单一模型为兼顾多模态牺牲了单模态最优性）。

### Table 8: OmniGait Intra-/Inter-Sensor Fusion / OmniGait 融合性能

| Modality | NM R1 | NM mAP | BG R1 | BG mAP | CL R1 | CL mAP |
|----------|-------|--------|-------|--------|-------|--------|
| **Intra-Sensor** RGB(Sil.) | 98.1 | 98.7 | 95.7 | 97.2 | 44.9 | 57.6 |
| +RGB(Event) | 98.6 | 99.1 | 96.8 | 97.9 | 45.4 (+0.5) | 58.1 (+0.5) |
| +RGB(Pose) | 98.0 | 98.7 | 95.8 | 97.2 | 46.1 (+1.2) | 58.6 (+1.0) |
| +RGB(RGB) | 99.8 | 99.9 | 99.5 | 99.7 | **64.7 (+19.8)** | **74.9 (+17.3)** |
| **Inter-Sensor** RGB(Sil.) | 98.1 | 98.7 | 95.7 | 97.2 | 44.9 | 57.6 |
| +Depth | 98.3 | 98.9 | 96.4 | 97.6 | 49.1 (+4.2) | 61.5 (+3.9) |
| +Radar(Proj.) | 98.4 | 98.9 | 96.4 | 97.7 | 50.9 (+6.0) | 63.1 (+5.5) |
| +LiDAR(Proj.) | 98.5 | 99.1 | 96.7 | 97.9 | **51.9 (+7.0)** | 63.9 (+6.3) |

**说明**: 以 RGB silhouette 为锚，融合互补信号一致提升（CL 尤甚）：+RGB 原图带来 +19.8% R1；inter-sensor 加 LiDAR 投影深度 +7.0%。验证 OmniGait 模块化设计能捕捉多源互补线索。

### Table 9: Parameters & FLOPs / 参数与计算量

| Task Setting | Modality (示例) | Params (M) | FLOPs (G) |
|--------------|------------------|-----------|-----------|
| Single-Modal (GaitBase) | RGB (Sil.) | 7.82 | 51.68 |
| Cross-Modal (Two-Stream) | RGB(Sil.) ↔ Depth | 10.82 | 103.49 |
| Multi-Modal (MultiGait++) | RGB(Sil.) + Depth | 11.92 | 106.29 |
| **Omni Multi-Modal (OmniGait)** | 9 模态统一 | **9.96** | 1054.11 |

**说明**: OmniGait 参数(9.96M)介于单模态与多模态专用模型之间，**用一个模型替代多个专用模型**；FLOPs 较高(1054G)因训练时同 batch 联合处理全部模态——这是统一性的代价。

### Table 10: Cross-Dataset on SUSTech1K / 跨数据集泛化（零微调）

| Input Modality | Normal | Bag | Clothing | Carrying | Umbrella | Uniform | Occlusion | Night | Overall R1 | Overall R5 |
|----------------|--------|-----|----------|----------|----------|---------|-----------|-------|-----------|-----------|
| Lidar Depth | 13.67 | 10.81 | 4.88 | 7.29 | 1.14 | 7.06 | 10.88 | 9.55 | 7.77 | 17.38 |
| RGB | 43.15 | 30.07 | 20.52 | 32.45 | 27.93 | 24.33 | 32.31 | 44.66 | 32.03 | 53.21 |
| Silhouette | 47.24 | 44.68 | 26.28 | 41.78 | 39.42 | 42.04 | 42.98 | 18.08 | 42.65 | 62.15 |
| **RGB+Silhouette** | **63.93** | **54.39** | **35.98** | **52.71** | **50.98** | **48.86** | **46.97** | 32.86 | **53.07** | **70.96** |

**说明**: OmniGait 在 MMGait 训练、**直接在 SUSTech1K 上零微调评测**。多模态融合(RGB+Silhouette)显著优于单模态(Overall R1 53.07% vs 单模态 ≤42.65%)，验证 MMGait 训练表征的跨数据集泛化性与融合增益。

---

## 实验

### 数据集 / 基准

| 基准 | 规模 | 特点 | 用途 |
|------|------|------|------|
| MMGait（本文） | 725 人 / 334,060 序列 / 12 模态 / 10 视角 / NM·BG·CL | 5 类异构传感器、五角星全圆周采集、跨月多光照 | 训练(200人)/测试(525人) |
| SUSTech1K | 1,050 人 / 25,239 序列 | RGB + LiDAR，多协变量（伞/夜/遮挡等） | 跨数据集零微调测试 |

### 实现细节

- **单模态基线**: 视觉模态(RGB/Event/IR/投影深度)用 [[GaitBase]]；2D/3D pose 用 [[GPGait++]]；点云(LiDAR/4D 雷达)用 [[LidarGait++]]。
- **跨模态基线**: 两流网络（模态专属浅层 + 共享深层），交叉熵 + 跨模态三元组损失联合优化；共享层 BN 统计混合多域特征以隐式对齐。
- **多模态基线**: 遵循 [[MultiGait++]] 两流融合策略。
- **跨模态统一表征**: RGB/IR 用 silhouette，深度用 depth map，LiDAR/雷达用投影深度图，以稳定特征统计。
- **OmniGait**: $T=16$、$64\times64$、batch $8\times4$；9 种图像化模态；训练全模态联合、推理仅激活所需编码器。

### 关键实验结论

- **单模态（Table 3/4）**: silhouette/RGB 总体最强；**IR 在换装(CL)下意外最优(78.8%)**；4D 雷达最弱（稀疏），宜作辅助。
- **跨模态（Table 5 / Fig 4）**: RGB↔IR 易、RGB↔深度类难、RGB↔雷达近乎失效；**CL 条件使跨模态 R1 从 55.5% 跌至 28.4%**，是核心难点。
- **多模态（Table 6）**: **跨传感器融合收益最大**，RGB silhouette + LiDAR 在 CL 上 +19.7% R1；几何深度弥补外观歧义。
- **OmniGait（Table 7/8 / Fig 6）**: 统一模型单模态接近专用模型，跨模态在 NM/BG 反超专用基线（RGB→Depth NM 87.0% vs 77.5%），多模态融合一致增益；CL 仍是短板。
- **跨数据集（Table 10）**: 零微调迁移到 SUSTech1K，融合表征(53.07%)显著优于单模态，验证泛化。

### 讨论要点（§9）

1. **跨模态检索仍极具挑战**：大域差、分布差异、模态特定噪声使统一表征既判别又模态不变很难。
2. **多模态融合收益显著**：RGB silhouette + LiDAR 投影深度在跨换装下 +19.7%，源于 LiDAR 稳定几何 + RGB 丰富形状的互补。
3. **OMGR 机遇与挑战并存**：统一模型常牺牲单模态最优性，跨协变量鲁棒性受限；当前把 3D 点云投影成深度图以减域差、便于共享骨干，但**丢失部分内在几何结构**——直接在统一架构内建模原始 3D 点云是有前景的未来方向。

---

## 批判性思考

### 优点
1. **基准全面且规模大**: 5 传感器 / 12 模态 / 725 人 / 33.4 万序列，是同类多模态步态集中模态最全、序列最多的之一；含 10 视角与 NM/BG/CL 协变量，支撑严格的跨视角、跨协变量评测。
2. **系统化的三范式实证洞见**: 不止给数据，还系统量化了各模态鲁棒性与互补性（如 IR 抗换装、跨传感器融合在 CL 上 +19.7%、雷达适合辅助），对后续研究有很强参考价值。
3. **新任务 + 强基线**: OMGR 任务设定贴近真实部署（任意模态查询/检索），OmniGait 用单一轻量模型(9.96M)统一三范式，跨模态部分甚至反超专用模型，开源代码/权重/数据。

### 局限性
1. **统一模型牺牲单模态最优性**: OmniGait 在 CL 条件下普遍低于专用模型（Table 7 vs 4），单一模型的"通用性"以"峰值性能"为代价，差距尚未弥合。
2. **跨模态/跨换装仍是硬骨头**: CL 条件下跨模态 R1 仅 ~28%，RGB↔雷达几乎不可检索；OMGR 的实用性受真实协变量制约。
3. **几何信息损失与同步妥协**: 为统一处理把 3D 点云投影成深度图，丢弃内在几何；且**未强制跨模态时序严格同步**（仅 RGB-Depth 帧级同步，其余靠时间戳序列级近似对齐），可能影响融合上限。
4. **OmniGait 训练计算量偏高**: 同 batch 联合处理全部模态导致 FLOPs 高达 1054G，训练成本不低。

### 潜在改进方向
1. 在统一 omni 架构内**直接建模原始 3D 点云**（而非投影深度），保留几何线索，提升 3D 模态与融合上限。
2. 针对 **CL/跨模态短板**设计更强的解耦/对齐机制（如显式去外观、模态不变约束），缩小统一模型与专用模型的差距。
3. 引入**跨模态时序同步**或时序对齐模块，挖掘真正的时空互补；并探索降低 OmniGait 训练 FLOPs 的高效联合训练策略。

### 可复现性评估
- [x] 代码开源（https://github.com/BNU-IVC/MMGait）
- [x] 预训练模型（声明公开 pretrained checkpoints）
- [x] 训练细节完整（正文 + 附录给出网络配置、损失、超参）
- [x] 数据集可获取（MMGait 声明公开；SUSTech1K 公开）

---

## 速查卡片

> [!summary] MMGait / OmniGait: Towards Multi-Modal Gait Recognition
> - **核心**: 5 传感器 / 12 模态 / 725 人 / 334,060 序列的多模态步态基准 MMGait + 新任务 OMGR + 统一基线 OmniGait。
> - **方法**: OmniGait = 模态专属编码 → 门控跨模态融合 → 共享残差骨干（同 batch 联合处理使 BN 学跨模态鲁棒统计），CE + 三元组损失统一单/跨/多模态。
> - **结果**: 跨传感器融合 CL 条件 +19.7% R1；IR 抗换装最强(78.8%)；OmniGait 跨模态反超专用模型(RGB→Depth NM 87.0%)；4D 雷达适合辅助；CL 与跨模态仍是难点。
> - **代码**: https://github.com/BNU-IVC/MMGait

---

*笔记创建时间: 2026-06-29*
