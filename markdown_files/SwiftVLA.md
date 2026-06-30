---
title: "SwiftVLA: Unlocking Spatiotemporal Dynamics for Lightweight VLA Models at Minimal Overhead"
method_name: "SwiftVLA"
authors: [Chaojun Ni, Cheng Chen, Xiaofeng Wang, Zheng Zhu, Wenzhao Zheng, Boyuan Wang, Tianrun Chen, Guosheng Zhao, Haoyun Li, Zhehao Dong, Qiang Zhang, Yun Ye, Yang Wang, Guan Huang, Wenjun Mei]
year: 2026
venue: CVPR
tags: [VLA, lightweight-vla, 4D-feature, spatiotemporal, mask-reconstruct, fusion-tokens]
zotero_collection: _inbox
image_source: online
arxiv_html: https://arxiv.org/html/2512.00903v1
created: 2026-06-29
---

# SwiftVLA: Unlocking Spatiotemporal Dynamics for Lightweight VLA Models at Minimal Overhead

## 元信息

| 项目 | 内容 |
|------|------|
| 作者 | Chaojun Ni, Cheng Chen, Xiaofeng Wang, Zheng Zhu, Wenzhao Zheng 等 15 人 |
| 机构 | （arXiv 预印本，作者团队含 GigaAI / 北京大学等，详见原文）|
| 会议 | CVPR 2026 |
| 类别 | VLA（Vision-Language-Action） |
| 日期 | 2025-12（arXiv v1） |
| 项目主页 | （未公开）|
| 链接 | [arXiv](https://arxiv.org/abs/2512.00903) / [PDF](https://arxiv.org/pdf/2512.00903) |

---

## 一句话总结

> 给轻量 VLM 配一个**冻结的 4D 几何 Transformer（带时序 cache）**作为辅助输入，用 [[Fusion Tokens]] 做跨模态融合、用 **mask-and-reconstruct** 把 4D 时空知识蒸馏进网络，从而在推理时**丢掉 4D 分支**仍保住时空推理能力——边缘设备上比 $\pi_0$ 快 18×、省 12× 显存且成功率反超。

---

## 核心贡献

1. **最小开销注入 4D 时空信息**: 提出 SwiftVLA，在轻量 [[VLM]]（[[SmolVLM]]-0.5B）基础上引入 **4D 特征 + mask-and-reconstruct 训练**，把 4D 知识蒸馏进 VLA；推理时**仅用 2D 输入**即可达到与带 4D 输入近似的性能，几乎零额外开销。
2. **Fusion Tokens 跨模态融合**: 引入一组可学习 [[Fusion Tokens]]，在轻量 VLM 内通过交叉注意力融合 2D 与 4D 特征，并用**机器人末端执行器未来轨迹**作监督，产出统一、面向动作的表征。
3. **小模型对标大模型**: 仿真与真机大量实验证明 SwiftVLA 性能可比肩参数量大 **7×** 的 VLA；在 NVIDIA Jetson Orin 上推理快 **18×**、显存少 **12×**（对比 $\pi_0$）。

---

## 问题背景

### 要解决的问题
如何让**轻量级 [[VLA]]**（参数小、能在边缘设备实时跑）同时具备**强时空推理能力**（精准定位、合理轨迹规划），而**不引入大模型或额外传感器/分支带来的开销**。

### 现有方法的局限
作者用 Figure 1 / Figure 2 概括了现有路线的三类痛点：
1. **纯压缩主干（Fig 2a）**: [[SmolVLA]]、[[TinyVLA]]、MiniVLA 等通过缩小 VLM 或砍层来轻量化，但**空间推理被削弱**——例如 SmolVLM-0.5B 在"最左边碗是什么颜色"这类空间问答上明显不如 PaliGemma-3B，导致复杂操作成功率低。
2. **大 VLM 内直接融合 3D（Fig 2b）**: [[3D-VLA]]、[[SpatialVLA]]、[[Evo-0]] 把 3D 特征塞进 VLM，2D 像素与 3D 几何域差大，**必须依赖重型 VLM**才能对齐融合。
3. **解耦的独立 3D 分支（Fig 2c）**: [[PointVLA]]、[[GeoVLA]] 加一条专用空间分支，**参数开销大**且只关注 3D、忽略时序动态；4D-VLA 虽引入时序但多帧采样又带来推理开销。

### 本文的动机
- **4D 特征来自普通图像**：用预训练的 [[4D Visual Geometry Transformer|4D 视觉几何 Transformer]]（[[StreamVGGT]]）从 2D 图像增量提取带时序的 4D 特征，**无需深度相机/LiDAR**。
- **训练时用、推理时丢**：关键洞察是 4D 既然主要用于"教会"模型几何/动态先验，那就可以通过**重建目标把它蒸馏进权重**，推理时整条 4D 链路（提取器、重建头、轨迹头）全部移除，保住轻量。
- **小模型不会用多模态**：纯把 4D 喂进小 VLM 收益有限，于是用 **Fusion Tokens + 未来轨迹监督**主动引导小模型把 2D/4D 线索用起来。

---

## 方法详解

### 模型架构

SwiftVLA 由两个相连组件构成（见 Figure 3）：**轻量 VLM $\mathcal{V}$**（[[SmolVLM]]）+ **动作专家 $\mathcal{A}$**（条件扩散模型）。

- **输入**: 时刻 $t$ 给定有序视角集 $S=[left,right,front]$、语言指令 $l$、多视角观测 $o_t=\{o_t^v\}_{v\in S}$、本体状态 $s_t$。
- **2D 特征**: 各视角用图像编码器（[[SigLIP]]）提取 $F_{2D}^t=\{F_{2D}^{t,v}\}_{v\in S}$。
- **4D 特征**: 冻结的 [[StreamVGGT]] 从图像增量提取 $F_{4D}^t$（带时序 cache）。
- **核心模块**: [[Fusion Tokens]] $Q_f$ 在 VLM 内与 2D/4D/语言/状态 token 交互 → 统一表征 $Z_f^t$；VLM 中间隐状态 $\{h_{\mathcal V}^{(i)}\}$ 作为动作专家条件。
- **输出**: [[Action Chunking|动作块]]（扩散噪声预测）。
- **部署参数**: VLM + 动作专家 = **0.45B**（带 4D 输入时为 1.65B，仅训练/可选推理用）。

整体感知映射（Fusion Tokens 融合）：

$$
Z_{f}^{t}=\mathcal{V}\!\left(Q_{f},E_{s}^{t},E_{l}^{t},F_{4D}^{t},F_{2D}^{t}\right)
$$

动作专家以噪声 $\epsilon$ 为输入、VLM 分层特征为条件，产出动作潜变量：

$$
Z_{\mathcal{A}}^{t}=\mathcal{A}\!\left(\epsilon\,\middle|\,\{h_{\mathcal{V}}^{(i)}\}\right)
$$

$Z_{\mathcal A}^t$ 由两个互补头解码：一个预测动作的扩散噪声，另一个重建被 mask 的特征（推理时丢弃）。

### 核心模块

#### 模块1: Incremental 4D Feature Extraction（增量 4D 特征提取）

**设计动机**: 用预训练的 [[4D Visual Geometry Transformer|4D 视觉几何 Transformer]] 从普通图像里榨出空间+时序线索，避免额外传感器，并充分利用时序建模能力（见 Figure 4）。

**具体实现**:
- 模块由 **encoder + decoder + 时序 cache** 组成，**权重全程冻结**。
- 各视角观测先编码：$F_e^{t,v}=Encoder(o_t^v)$（公式3）。
- 解码器按视角顺序 $S$ 用空间+时序注意力提取 4D 表征；时序阶段当前特征与 cache 做交叉注意力注入时序上下文。cache 初始化为 $C^{t,0}=C^{t-1}$，逐视角更新（公式4）。
- 处理完所有视角后 $C^t=C^{t,3}$；cache 采用 **FIFO** 策略，只保留最近 $K$ 个 4D 表征。
- **关键工程取舍**: 喂给 VLM 的 4D 特征**只取 front 视角** $F_{4D}^t=F_{4D}^{t,front}$，left/right 仅用于更新 cache 以提供时空上下文，节省训练成本（公式5）。

#### 模块2: Fusion Tokens（融合 token）

**设计动机**: 小 VLM 难以自行把 2D 与 4D 融成连贯、3D-aware 的隐空间，于是引入可学习 token 主动撮合，并用**末端轨迹**作为下游任务锚点。

**具体实现**:
- $Q_f$ 在 VLM $\mathcal{V}$ 内通过交叉注意力与聚合多模态序列（$F_{2D}^t,F_{4D}^t,E_l^t,E_s^t$）交互，得融合表征 $Z_f^t$（公式1）。
- Fusion Tokens 产生的 key/value 连同其他 token 一起构成动作专家的条件 $h_{\mathcal V}^{(i)}$。
- $Z_f^t$ 中对应 Fusion Tokens 的部分解码出**末端执行器未来轨迹**，用 GT 轨迹监督（公式6），使中间隐状态学到**轨迹感知的跨模态对齐**。

#### 模块3: Mask-and-Reconstruct Strategy（掩码-重建策略）

**设计动机**: 既要 4D 带来的几何增益，又不想把 4D 开销带到推理；于是在训练时用 4D 监督建立几何感知表征，推理时丢掉 4D，性能损失最小（这是"训练时用、推理时丢"的核心机制）。

**具体实现**:
- **训练**: 以一定概率随机 mask 掉 2D 或 4D 特征（Figure 3 可视化了 4D 被 mask 时的注意力掩码：灰/白为固定可见/不可见 token，粉色为随机由可见转不可见）。VLA 须用剩余模态预测动作，同时重建被 mask 的特征（公式7 的 $\mathcal L_{2D},\mathcal L_{4D}$）。
- 动作头预测扩散噪声（公式8）；总损失为四项加权和（公式9）。
- **推理**: 仅保留 2D 分支，移除 4D 提取器、重建头、轨迹头；部署模型 = VLM + 动作专家。掩码训练让模型即便无显式 4D 输入也能**隐式重建并推理 4D 空间结构**。
- 额外发现：**适度 mask 2D** 反而促使模型更多依赖 4D 几何线索、增强跨模态一致性（Table 6）。

### 关键公式与机制

#### 公式1: [[Fusion Tokens]] 跨模态融合

$$
Z_{f}^{t}=\mathcal{V}\!\left(Q_{f},E_{s}^{t},E_{l}^{t},F_{4D}^{t},F_{2D}^{t}\right)
$$

**含义**: 可学习 Fusion Tokens $Q_f$ 在 VLM 内与状态、语言、4D、2D 嵌入交互，产出统一融合表征。

**符号说明**:
- $Q_f$: 可学习 Fusion Tokens；$\mathcal V$: 轻量 VLM
- $E_s^t,E_l^t$: 本体状态/语言嵌入；$F_{4D}^t,F_{2D}^t$: 4D/2D 视觉特征
- $Z_f^t$: 融合后的统一表征（感知输出，供轨迹预测与动作条件）

#### 公式2: 动作专家（条件扩散）

$$
Z_{\mathcal{A}}^{t}=\mathcal{A}\!\left(\epsilon\,\middle|\,\{h_{\mathcal{V}}^{(i)}\}\right)
$$

**含义**: 动作专家以噪声 $\epsilon$ 为输入、以 VLM 分层隐状态为条件，生成动作潜变量。

**符号说明**:
- $\mathcal A$: 条件扩散动作专家；$\epsilon$: 输入噪声
- $\{h_{\mathcal V}^{(i)}\}$: VLM 各层中间隐状态（分层条件特征）；$Z_{\mathcal A}^t$: 动作潜变量

#### 公式3: 单视角特征编码

$$
F_{e}^{t,v}=Encoder(o_{t}^{v})
$$

**含义**: 4D 几何 Transformer 的编码器把视角 $v$ 的观测编码为特征嵌入。

**符号说明**: $o_t^v$: 视角 $v$ 的观测；$F_e^{t,v}$: 编码后的特征嵌入。

#### 公式4: 时序解码与 cache 更新

$$
\left(F_{4D}^{t,v},\,C^{t,k}\right)=\mathrm{Decoder}\!\big(\mathrm{CrossAttn}(F_{e}^{t,v},\,C^{t,k-1})\big)
$$

**含义**: 当前视角特征与 cache 做交叉注意力注入时序上下文，解码出该视角 4D 特征并更新 cache。

**符号说明**:
- $k\in\{1,2,3\}$ 对应视角 $v=S_k$；$C^{t,k}$: 第 $k$ 步后的时序 cache
- $\mathrm{CrossAttn}$: 时序交叉注意力；初始化 $C^{t,0}=C^{t-1}$，结束 $C^t=C^{t,3}$（FIFO 保留最近 $K$ 个）

#### 公式5: VLM 视觉输入定义

$$
F_{2D}^{t}=\{F_{2D}^{t,v}\}_{v\in S},\qquad F_{4D}^{t}=F_{4D}^{t,front}
$$

**含义**: 2D 特征用全部三视角，而喂入 VLM 的 4D 特征**只取 front 视角**（其余视角仅更新 cache），权衡精度与训练成本。

#### 公式6: 末端轨迹预测（Fusion Tokens 监督）

$$
\hat{\tau}_{t}=h_{traj}\!\left(Z_{f}^{t}\right),\qquad\mathcal{L}_{traj}=\left\lVert\hat{\tau}_{t}-\tau_{t}\right\rVert_{2}^{2}
$$

**含义**: 由融合表征解码末端执行器未来轨迹，用 GT 轨迹做 L2 监督，使隐状态学到轨迹感知的跨模态对齐。

**符号说明**: $h_{traj}$: 轨迹预测头；$\hat\tau_t$: 预测轨迹；$\tau_t$: GT 轨迹。

#### 公式7: 特征重建损失

$$
\begin{split}
\mathcal{L}_{2D}&=\left\|h_{2D}\!\left(Z_{\mathcal{A}}^{t}\right)-F_{2D}^{t}\right\|_{2},\\
\mathcal{L}_{4D}&=\left\|h_{4D}\!\left(Z_{\mathcal{A}}^{t}\right)-F_{4D}^{t}\right\|_{2}.
\end{split}
$$

**含义**: 从动作潜变量分别重建被 mask 的 2D / 4D 特征，把几何-动态知识蒸馏进网络。

**符号说明**: $h_{2D},h_{4D}$: 特征重建头（推理时丢弃）；$Z_{\mathcal A}^t$: 动作潜变量（公式2）。

#### 公式8: 动作扩散损失

$$
\mathcal{L}_{action}=\mathbb{E}_{\epsilon\sim\mathcal{N}(0,I)}\!\left[\left\|h_{action}\!\left(Z_{\mathcal{A}}^{t}\right)-\epsilon\right\|_{2}^{2}\right]
$$

**含义**: 动作头从动作潜变量预测前向过程噪声 $\epsilon$，即标准扩散去噪目标。

**符号说明**: $\epsilon\sim\mathcal N(0,I)$: 前向噪声；$h_{action}$: 动作预测头。

#### 公式9: 总损失

$$
\begin{split}
\mathcal{L}_{total}&=\lambda_{2D}\,\mathcal{L}_{2D}+\lambda_{4D}\,\mathcal{L}_{4D}\\
&\quad+\lambda_{action}\,\mathcal{L}_{action}+\lambda_{traj}\,\mathcal{L}_{traj}
\end{split}
$$

**含义**: 重建、动作预测、轨迹三类目标的加权和，鼓励模型学习全面、几何感知的 4D 表征，而非依赖单一模态。

**符号说明**: 各 $\lambda$ 为平衡系数。

---

## 关键图表

<!-- 图片均使用 arXiv HTML 在线链接，未本地化 -->

### Figure 1: Motivation / 大小 VLM 的空间推理与速度权衡

![Figure 1](https://arxiv.org/html/2512.00903v1/x1.png)

**说明**: 大 VLM（PaliGemma-3B）空间推理强（绿色正确、红色错误），使基于它的 $\pi_0$ 成功率更高但推理慢；基于小 VLM 的 [[SmolVLA]] 快但成功率低。SwiftVLA 在保住速度优势的同时增强了小 VLA 的时空动态能力（在 Jetson Orin 上测速与成功率）。这张图直接点出"速度 vs 时空推理"的核心矛盾。

### Figure 2: Design Comparison (a–d) / 四类 3D-4D 融合范式对比

![Figure 2](https://arxiv.org/html/2512.00903v1/x2.png)

**说明**: (a) 仅 2D 输入 → 时空感知受限；(b) 大 VLM 内直接融合 3D 与 2D → 依赖重型 VLM；(c) 解耦的独立空间分支 → 参数开销大；(d) **SwiftVLA**：用预训练模型（[[StreamVGGT]]）提 4D 特征 + 特征重建对齐 2D/4D + Fusion Tokens 与未来预测目标强化融合，**推理时移除 4D 输入与辅助头**保效率。这是全文方法定位图。

### Figure 3: SwiftVLA Pipeline / 整体流水线与掩码

![Figure 3](https://arxiv.org/html/2512.00903v1/x3.png)

**说明**: 先从图像提 2D 与 4D 特征，轻量 VLM 用 Fusion Tokens 做跨模态整合，Fusion Tokens 输出由末端未来轨迹监督；训练时随机 mask 2D 或 4D，要求动作专家在生成动作的同时重建被 mask 的特征。图中展示了 mask 4D 时的注意力掩码：4D 被排除出 VLM 注意力，需从其余 token 重建。

### Figure 4: 4D Feature Extraction / 增量 4D 特征提取

![Figure 4](https://arxiv.org/html/2512.00903v1/x4.png)

**说明**: 每步顺序处理多视角观测，从 cache 载入上下文做时序注意力；生成的 4D 特征更新回 cache 并送入 VLM。体现"增量 + 时序 cache（FIFO 保留最近 $K$ 帧）"的设计。

### Figure 5: Real-World Qualitative Comparison / 真机抓取对比

![Figure 5](https://arxiv.org/html/2512.00903v1/x5.png)

**说明**: 相同初始位姿下，[[SmolVLA]] 因几何理解不足抓取不准，末端偏离并撞偏目标物（存在安全隐患）；SwiftVLA 凭更强空间感知与控制实现稳定精准抓取。定性佐证 4D 几何先验的价值。

### Figure 6: LIBERO Task Suites / LIBERO 任务示例（附录）

![Figure 6](https://arxiv.org/html/2512.00903v1/x6.png)

**说明**: LIBERO-Spatial / Object / Goal / Long 四个任务套件的示例。

### Figure 7: RoboTwin 2.0 Tasks / RoboTwin 双臂任务示例（附录）

![Figure 7](https://arxiv.org/html/2512.00903v1/x7.png)

**说明**: [[RoboTwin]] 2.0 任务示例，含 move Stapler Pad、Place A2B Left、Place Bread Basket、Place Dual Shoes、Dump Bin Bigbin、Handover Block。

### Figure 8: Real-World Tasks / 真机操作任务（附录）

![Figure 8](https://arxiv.org/html/2512.00903v1/x8.png)

**说明**: 真机实验任务，从上到下为 Clean the Desk、Throw the Bottle、Stack Bowls。

### Figure 9: Fold the Cloth Execution / 叠衣服执行序列（附录）

![Figure 9](https://arxiv.org/html/2512.00903v1/x9.png)

**说明**: 真机 "Fold the Cloth" 任务全过程，需要长程推理与对可变形物体的精细操作，是最具挑战性的任务。

### Table 1: Simulation Results (RoboTwin 2.0) / 仿真成功率与轨迹长度

| Method | Short SR↑ | Short Len↓ | Med SR↑ | Med Len↓ | Long SR↑ | Long Len↓ | Avg SR↑ | Avg Len↓ |
|--------|----------|-----------|---------|----------|----------|-----------|---------|----------|
| $\pi_0$ | 0.42 | 120 | 0.46 | 150 | 0.52 | 187 | 0.47 | 152 |
| GO-1 | 0.40 | 124 | 0.44 | 160 | 0.54 | 190 | 0.46 | 158 |
| TinyVLA | 0.08 | 183 | 0.08 | 240 | 0.06 | 236 | 0.07 | 220 |
| SmolVLA | 0.28 | 152 | 0.32 | 178 | 0.28 | 234 | 0.29 | 188 |
| SmolVLA† | 0.38 | 130 | 0.36 | 165 | 0.34 | 195 | 0.36 | 163 |
| **SwiftVLA** | **0.56** | 115 | 0.48 | 156 | **0.56** | 180 | 0.53 | 150 |
| **SwiftVLA (w/ 4D)** | **0.56** | **100** | **0.50** | **145** | **0.58** | 185 | **0.55** | **143** |

**说明**: RoboTwin 2.0（双臂，按 short/medium/long-horizon 分组）。SwiftVLA（仅 2D 推理）平均 SR 0.53，已超 $\pi_0$(0.47)、GO-1(0.46) 与同尺寸 SmolVLA†(0.36)；带 4D 输入再升到 0.55 且轨迹最短。†表示与 SwiftVLA 用相同配置预训练+微调。

### Table 2: Real-World Results / 真机成功率与轨迹长度

| Methods | Clean SR↑ | Clean Len↓ | Throw SR↑ | Throw Len↓ | Stack SR↑ | Stack Len↓ | Avg SR↑ | Avg Len↓ |
|---------|----------|-----------|-----------|-----------|-----------|-----------|---------|----------|
| $\pi_0$ | 0.60 | 1220 | 0.66 | 980 | 0.56 | 840 | 0.61 | 1013 |
| SmolVLA | 0.32 | 1640 | 0.40 | 1360 | 0.30 | 1360 | 0.34 | 1453 |
| SmolVLA† | 0.52 | 1360 | 0.54 | 1140 | 0.52 | 860 | 0.53 | 1120 |
| **SwiftVLA** | **0.86** | 1140 | 0.80 | 980 | 0.74 | **800** | 0.80 | 973 |
| **SwiftVLA (w/ 4D)** | **0.86** | **1090** | **0.82** | **960** | **0.78** | 810 | **0.82** | **953** |

**说明**: AgileX PiPER 6-DoF 真机三任务。SwiftVLA 平均 0.80（带 4D 0.82），大幅超过 $\pi_0$(0.61) 与同尺寸 SmolVLA†(0.53)，且轨迹更短（控制更高效）。

### Table 3: LIBERO Results / LIBERO 成功率对比

| Methods | Size | Spatial | Object | Goal | Long | Avg |
|---------|------|---------|--------|------|------|-----|
| *Spatio-Temporal Enhanced VLA* | | | | | | |
| SpatialVLA | 4B | 88.2 | 89.9 | 78.6 | 55.5 | 78.1 |
| 4D-VLA | 4B | 88.9 | 95.2 | 90.9 | 79.1 | 88.6 |
| QDepth-VLA | 4B | 97.6 | 96.6 | 95.2 | 90.0 | 94.9 |
| *Small VLA* | | | | | | |
| SmolVLA | 0.45B | 90.0 | 96.0 | 92.0 | 71.0 | 87.3 |
| SmolVLA† | 0.45B | 93.5 | 96.5 | 95.4 | 83.4 | 92.2 |
| UniAct | 0.5B | 77.0 | 87.0 | 77.0 | 70.0 | 77.8 |
| VLA-OS | 0.5B | 87.0 | 96.5 | 92.7 | 66.0 | 85.6 |
| SmolVLA | 2B | 93.0 | 94.0 | 91.0 | 77.0 | 88.8 |
| *Large VLA* | | | | | | |
| GR00T-N1 | 3B | 94.4 | 97.6 | 93.0 | 90.6 | 93.9 |
| $\pi_0$ | 3B | 96.8 | **98.8** | 95.8 | 85.2 | 94.1 |
| $\pi_0$+FAST | 3B | 96.4 | 96.8 | 88.6 | 60.2 | 85.5 |
| OpenVLA | 7B | 84.7 | 88.4 | 79.2 | 53.7 | 76.5 |
| OpenVLA-OFT | 7B | **97.6** | 98.4 | **97.9** | **94.5** | **97.1** |
| DD-VLA | 7B | 97.2 | 98.6 | 97.4 | 92.0 | 96.3 |
| UniVLA | 9B | 95.4 | **98.8** | 93.6 | 94.0 | 95.4 |
| *Spatio-Temporal Enhanced Small VLA* | | | | | | |
| **SwiftVLA** | **0.45B** | 97.0 | 96.4 | 96.8 | 88.4 | 94.7 |
| **SwiftVLA (w/ 4D)** | 1.65B | 97.2 | 96.8 | 97.4 | 89.0 | 95.1 |

**说明**: SwiftVLA 仅 **0.45B** 即取得 94.7% 平均成功率，超过 $\pi_0$(3B, 94.1%)、GR00T-N1(3B, 93.9%) 等大模型，逼近 7B 的 DD-VLA/OpenVLA-OFT；以同尺寸碾压 SmolVLA(0.45B, 87.3%)。粗体为各列全局最优（OpenVLA-OFT 在多列居首，但其参数量是 SwiftVLA 的 15×）。

### Table 4: Edge Device Efficiency (Jetson Orin) / 边缘设备效率

| Methods | Inference Time (s) | Memory (MB) | Average SR |
|---------|-------------------|-------------|------------|
| $\pi_0$ | 2.966 | 16236.2 | 0.48 |
| SmolVLA | 0.166 | 1397.5 | 0.30 |
| **SwiftVLA** | **0.167** | **1398.4** | **0.76** |

**说明**: NVIDIA Jetson Orin 上，SwiftVLA 推理 0.167s（比 $\pi_0$ 的 2.966s 快约 **18×**）、显存 1398MB（比 $\pi_0$ 的 16236MB 省约 **12×**），延迟与 SmolVLA 几乎一致，但成功率 0.76 远超 SmolVLA(0.30) 与 $\pi_0$(0.48)。这是全文最有力的"效率-性能"证据。

### Table 5: Ablation — 4D Input & Fusion Tokens / 4D 与 Fusion Tokens 消融

| Input Feature Type | Fusion Tokens | Average SR |
|--------------------|:-------------:|:----------:|
| 2D | | 0.36 |
| 2D & 4D | | 0.40 |
| 2D & 4D | ✓ | **0.50** |

**说明**: RoboTwin 2.0 上。仅 2D（0.36）→ 加 4D（0.40，证明 4D 有用）→ 再加 Fusion Tokens（0.50，证明小模型需要 Fusion Tokens 才能真正用好 4D/2D）。（Q1+Q2）

### Table 6: Ablation — Training Strategy / 训练策略消融

| 4D Feature Mask | 2D Feature Mask | Feature Reconstruction | SwiftVLA (2D infer) | SwiftVLA (w/ 4D) |
|:---------------:|:---------------:|:----------------------:|:-------------------:|:----------------:|
| | | | 0.02 | 0.50 |
| ✓ | | | 0.40 | 0.48 |
| ✓ | ✓ | | 0.50 | 0.52 |
| ✓ | ✓ | ✓ | **0.53** | **0.55** |

**说明**: 全部训练时用 2D+4D，推理仅用 2D。**不做任何处理直接推理时丢 4D → 暴跌到 0.02**（过度依赖 4D）；加 4D mask 缓解依赖（0.40）；再加 2D mask（0.50）；最后加特征重建蒸馏（0.53，逼近带 4D 的 0.55）。完整验证 mask-and-reconstruct 的必要性。（Q3）

### Table 7: Ablation — Temporal Cache Size $K$ / 时序 cache 大小消融

| Size | SwiftVLA (2D infer) | SwiftVLA (w/ 4D) |
|------|:-------------------:|:----------------:|
| $K{=}3$ | 0.47 | 0.49 |
| $K{=}4$ | 0.48 | 0.52 |
| $K{=}5$ | 0.50 | 0.51 |
| $K{=}6$ | 0.52 | 0.55 |
| **Random** | **0.53** | **0.55** |

**说明**: 训练时随机采样 $K\in\{3,4,5,6\}$ 优于所有固定长度，说明暴露于可变时序跨度显著增强适应性。（Q4）

### Table 8: RoboTwin 2.0 Task Horizon Categories / 任务步长分组（附录）

| Category | Task | Steps |
|----------|------|-------|
| Short-Horizon | Move Stapler Pad | 112 |
| Short-Horizon | Place A2B Left | 113 |
| Medium-Horizon | Place Bread Basket | 151 |
| Medium-Horizon | Place Dual Shoes | 155 |
| Long-Horizon | Dump Bin Bigbin | 283 |
| Long-Horizon | Handover Block | 313 |

**说明**: 按完成所需平均步数把评测任务分为短(<120 步，局部空间推理)、中(~150 步，多物体序列规划)、长(>280 步，强时序依赖与组合复杂度)，用于系统分析跨步长泛化。

### Table 9: Challenging Real-World Task — Fold the Cloth / 叠衣服（附录）

| Methods | SR↑ | Length↓ |
|---------|-----|---------|
| $\pi_0$ | 0.45 | 2550 |
| SmolVLA | 0.05 | 3200 |
| SmolVLA† | 0.30 | 2600 |
| **SwiftVLA** | 0.60 | 2100 |
| **SwiftVLA (w/ 4D)** | **0.65** | **2010** |

**说明**: 长程 + 可变形物体的高难任务。SwiftVLA 0.60（带 4D 0.65）远超 $\pi_0$(0.45) 与 SmolVLA(0.05)，凸显 4D 特征对处理可变形物体与长程操作的价值。

---

## 实验

### 数据集 / 基准

| 基准 | 平台 | 特点 | 用途 |
|------|------|------|------|
| [[RoboTwin]] 2.0 | 双臂仿真 | 6 任务，按 short/medium/long-horizon 分组 | 训练/测试/消融 |
| [[LIBERO]] | 单臂仿真 | Spatial/Object/Goal/Long 四套件 | 训练/测试 |
| 真机 (Clean/Throw/Stack) | AgileX PiPER 6-DoF | 桌面清理/扔瓶/叠碗 | 真机评测 |
| 真机 Fold the Cloth | AgileX PiPER 6-DoF | 长程 + 可变形物体 | 挑战性评测 |
| 边缘部署 | NVIDIA Jetson Orin | 效率评测 | 推理时延/显存 |

### 实现细节

- **Backbone**: [[SmolVLM]]（[[SigLIP]] 编码视觉），pixel-shuffle 把每帧视觉 token 限到 **64**；只用前 **16 层** 加速；注意力**自注意力与交叉注意力交替**（参照 SmolVLA / GR00T 设计）；4D 特征也限到 64 token。
- **4D 提取器**: 预训练 [[StreamVGGT]]（encoder+decoder+时序 cache），权重**冻结**。
- **动作专家**: 条件扩散模型（噪声预测）。
- **预训练（两阶段）**: 数据集 [bu2025agibot / wu2024robomind 等公开数据]。Stage1 不用 4D/Fusion Tokens/mask，仅动作监督，100k 步、batch 256、lr 1e-4 cosine 衰减到 2.5e-6、warmup 200、AdamW($\beta_1{=}0.85,\beta_2{=}0.9$)，图像 512×512；Stage2 启用 4D+Fusion Tokens+mask-reconstruct，再训 50k 步、lr 5e-5。
- **微调（两阶段）**: 各 baseline 训 30k 步、超参沿用原实现以公平比较；SwiftVLA 前 10k 步仅动作监督稳定适配，之后启用 4D+Fusion Tokens+mask-reconstruct。
- **硬件**: 真机用 NVIDIA RTX 4090；边缘部署用 NVIDIA Jetson Orin。

### 关键实验结论

- **仿真 LIBERO（Table 3）**: 0.45B 取 94.7%，超 $\pi_0$(3B)/GR00T-N1(3B)，逼近 7B 模型，对标参数大 7× 的 VLA。
- **仿真 RoboTwin（Table 1）**: 平均 SR 0.53/0.55，超 $\pi_0$、GO-1、SmolVLA†。
- **真机（Table 2 + Table 9）**: 三任务平均 0.80–0.82、叠衣服 0.60–0.65，均显著领先。
- **边缘效率（Table 4）**: 比 $\pi_0$ 快 18×、省 12× 显存，延迟≈SmolVLA 但 SR 远高。
- **消融**: 4D 有用、Fusion Tokens 关键（Table 5）；mask-and-reconstruct 是"丢 4D 仍保性能"的核心（Table 6，0.02→0.53）；随机 cache 大小最佳（Table 7）。

---

## 批判性思考

### 优点
1. **"训练时用、推理时丢"的范式干净有效**: mask-and-reconstruct 把 4D 几何知识蒸馏进权重，Table 6 用 0.02→0.53 的对比强证据支撑，效率-性能权衡（Table 4）极具说服力。
2. **不依赖额外传感器**: 4D 特征直接从 RGB 图像增量提取（StreamVGGT + 时序 cache），落地门槛低，且时序 cache 的 FIFO/随机长度设计有实验支撑。
3. **小模型对标大模型且可上边缘**: 0.45B 在 LIBERO 超 3B 的 $\pi_0$，Jetson Orin 上 18× 加速、12× 省显存，工程价值高。

### 局限性
1. **绝对成功率在难任务上仍有限**: RoboTwin 平均仅 0.53–0.55、叠衣服 0.60–0.65；双臂长程与可变形操作离实用尚远。
2. **机构/开源信息缺失**: v1 未给项目主页/代码链接，复现需依赖文中超参；4D 提取器、数据集来自第三方但具体配置（mask 概率、各 $\lambda$ 取值）未在正文完整给出。
3. **基线公平性存疑点**: 大量对比依赖 †（用 SwiftVLA 同配置重训的 SmolVLA），自训基线与原报告差异大（如 LIBERO SmolVLA 87.3 vs SmolVLA† 92.2），需谨慎解读"对标 7× 大模型"的结论；LIBERO 上 OpenVLA-OFT/DD-VLA(7B) 仍明显更高。
4. **4D 收益偏小**: 推理带不带 4D 的差距通常仅 1–2 个百分点（如 LIBERO 94.7 vs 95.1），说明 4D 主要价值在训练蒸馏而非推理增益本身，"4D 必要性"更多体现在难任务（叠衣服 +5 点）。

### 潜在改进方向
1. 公布代码/权重与完整超参（mask 概率、$\lambda$、StreamVGGT 配置），并补充更强的可变形/长程任务与更多同尺寸基线以巩固结论。
2. 把"仅取 front 视角 4D""前 16 层 VLM""token 限 64"等手工取舍做敏感度分析或自动搜索。
3. 探索将 mask-and-reconstruct 蒸馏范式迁移到更大主干或跨本体设置，验证范式可移植性；并用表征探针量化"4D 知识是否真的蒸馏进权重"。

### 可复现性评估
- [ ] 代码开源（v1 未提供链接）
- [ ] 预训练模型（未声明）
- [x] 训练细节较完整（附录给出两阶段预训练/微调超参）
- [x] 数据集可获取（LIBERO/RoboTwin 2.0 公开；预训练用公开机器人数据）

---

## 速查卡片

> [!summary] SwiftVLA: Unlocking Spatiotemporal Dynamics for Lightweight VLA Models
> - **核心**: 轻量 VLM + 冻结 4D 几何 Transformer（时序 cache）+ Fusion Tokens + mask-and-reconstruct，把 4D 知识蒸馏进权重，**推理时丢 4D** 仍保时空推理。
> - **方法**: SmolVLM(前16层) 提 2D，StreamVGGT 增量提 4D（front 视角喂 VLM）；Fusion Tokens 用末端未来轨迹监督做融合；训练随机 mask 2D/4D 并重建；动作专家用条件扩散；两阶段预训练+微调。
> - **结果**: 0.45B，LIBERO 94.7%（超 3B 的 $\pi_0$）、RoboTwin 0.53、真机 0.80；Jetson Orin 上比 $\pi_0$ 快 18×、省 12× 显存，对标 7× 大模型。
> - **代码**: 暂未公开（arXiv 2512.00903）。

---

*笔记创建时间: 2026-06-29*
