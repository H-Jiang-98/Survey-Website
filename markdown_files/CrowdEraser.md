---
title: "Generating Humanless Environment Walkthroughs from Egocentric Walking Tour Videos"
method_name: "CrowdEraser"
authors: [Yujin Ham, Junho Kim, Vivek Boominathan, Guha Balakrishnan]
year: 2026
venue: CVPR
tags: [video-inpainting, video-diffusion, object-removal, omnimatte, egocentric-video, dataset, 3d-reconstruction]
zotero_collection: _inbox
image_source: online
arxiv_html: https://arxiv.org/html/2603.29036v1
created: 2026-06-29
---

# Generating Humanless Environment Walkthroughs from Egocentric Walking Tour Videos

## 元信息

| 项目 | 内容 |
|------|------|
| 作者 | Yujin Ham, Junho Kim, Vivek Boominathan, Guha Balakrishnan |
| 机构 | Rice University（莱斯大学） |
| 会议 | CVPR 2026 |
| 类别 | 视频补全 / 物体与效果移除 / 数据集构建 / 场景建模 |
| 日期 | 2026（arXiv v1） |
| 项目主页 | （未提供） |
| 链接 | [arXiv](https://arxiv.org/abs/2603.29036) / [PDF](https://arxiv.org/pdf/2603.29036) |

---

## 一句话总结

> 用半合成数据集 EgoCrowds 微调视频扩散模型 Casper，得到 CrowdEraser，能从第一人称"城市漫步"视频中真实抹除行人及其阴影，把人群密集场景变为可用于 3D/4D 重建的干净环境视频。

---

## 核心贡献

1. **半合成配对数据集 EgoCrowds**: 提出一个由真实"walking tour"视频合成的 1,000 对（含人群 vs. 纯背景）7 秒视频片段数据集。绕过"同一场景有人/无人对照不可拍摄"的难题，用前景行人 + 模拟阴影叠加到独立采集的空背景上，得到带 ground-truth 监督的视频对。
2. **CrowdEraser 模型**: 在 EgoCrowds 上微调 SOTA 物体-效果移除扩散模型 [[Casper]]（来自 [[Generative Omnimatte]]），并加入一个时序运动损失，使其在**人群密集、背景复杂**的城市漫步视频上显著超越 Casper、[[ProPainter]]、[[DiffuEraser]]，定性与定量均更优。
3. **下游 3D/4D 建模验证**: 证明抹除人群后的干净视频可用 [[SpatialTrackerV2]] 重建出更稳定的相机位姿与更稠密一致的点云，说明"去人"是把海量漫步视频转化为大规模城市场景建模数据的关键一步。

---

## 问题背景

### 要解决的问题
YouTube 等平台上有数千小时、覆盖全球的第一人称"城市漫步"（walking tour）视频，是构建多样化城市环境视觉模型（3D 神经渲染、机器人、自动驾驶等）的最丰富图像来源之一。但这些视频有一个致命缺陷：**画面中存在大量"人类瞬态"（human transients）**——人群占据大量像素并遮挡场景结构。由于地面高度的第一人称视角，即便单个行人贴近拍摄者也能霸占很大像素面积。如何**真实地移除这些行人及其关联效果（阴影、随身物）**，恢复干净静态环境，是本文要解决的核心问题。

### 现有方法的局限
作者把相关工作分为三类并指出其局限：
1. **Omnimatte / 视频抠像系**: 把视频分解为前景物体层 + 独立背景层。早期 [[Omnimatte]] 依赖光流优化、假设静态背景并用单应性逼近相机运动；[[Omnimatte3D]]、[[OmnimatteRF]] 放松了平面假设（预测视差图 / 用辐射场建模 3D 背景）；最新的 [[Generative Omnimatte]]（其去除模型即 Casper）用强大的视频扩散先验，但**在大面积人类遮挡区域仍难恢复干净背景**。
2. **视频补全（inpainting）系**: [[ProPainter]]（光流传播 + 时空 Transformer）和基于 Stable Diffusion 的 [[DiffuEraser]] 等，**在大遮挡掩码区域容易过度平滑/丢失细节，且难以处理锐利投影阴影**。
3. **直接套用 Casper 的问题**: 作者实测 Casper 在"人少且离相机远"的视频上表现尚可，但面对**人群密集 + 复杂室外背景**时会产生不合理伪影（大掩码内常幻觉出物体或人）。作者假设：这一性能差距主要源于 **Casper 训练数据与漫步视频之间的域偏移（domain shift）**，而非扩散模型设计本身的缺陷。

### 本文的动机
既然瓶颈是**缺少针对漫步视频的监督训练数据**，那就把精力放在**精心构造一个丰富的有监督人物（含阴影）补全数据集**上。但真实采集"同场景有人/无人"配对需要演员 + 相机轨迹精确复制的高度受控装置，既不可行也难以保证视觉多样性。受光流估计 [[FlowNet]]、图像分割等任务中**半合成数据集**成功经验启发，作者改用半合成视频生成管线：从真实漫步视频里分别抠出"空背景"和"行人前景"，再随机合成、叠加规则化模拟阴影，从而**既保留真实视频的外观/遮挡/相机运动真实感，又提供训练所需的 ground-truth 监督**。

---

## 方法详解

方法分两大块：**(A) EgoCrowds 半合成数据集的构造**（核心贡献），**(B) 在其上微调 Casper 得到 CrowdEraser**。

### EgoCrowds 数据集构造

每个视频片段固定为 **7 秒（197 帧 @ 16 fps）**。整体管线见 Figure 3：背景与前景均来自真实漫步视频，前景按 `Crowd%` 等级近似均匀采样，并对每个人施加随机化的软阴影。

#### 模块1: 背景片段抽取（Background Clip Extraction）

**设计动机**: 需要尽量"空"的背景片段作为 ground-truth 干净环境。

**具体实现**:
- 用"empty / early morning / deserted downtown / lockdown street"等暗示空旷的关键词从 YouTube 检索全长视频；统一标准化到 $720\times1280$、最高 30 fps。
- 为保证结构多样性，从全球 **50 个城市**采集 **64 个背景视频（57 训练 / 7 测试）**（见 Figure 2）。
- 切成不重叠 7 秒片段。理想空片段应无人/无动态物体，但该条件太严苛且难量化，故**放松为只统计人数**：用 [[Grounded-SAM-2]]（"person" 提示）检测，设每帧最大允许人数 $P=5$，仅保留"超过该阈值的帧占比小于容差 $\tau$ "的片段，$\tau=10\%$（软容差用于容忍检测噪声及远处的极小人影）。
- **质量控制**:
  - 亮度过滤：计算所有帧平均 Y 通道值 $\bar{Y}$，只保留 $\bar{Y}\in[50,200]$ 的片段。
  - 场景切换检测：相邻帧计算结构相似度 $\operatorname{SSIM}(I_{t},I_{t+1})$ 与归一化灰度直方图相关 $\rho(H_{t},H_{t+1})$；若 $\operatorname{SSIM}(I_{t},I_{t+1})<0.3$ 且 $\rho(H_{t},H_{t+1})<0.5$ 则判为场景转换并剔除。
  - 用 OCR/检测过滤含字幕、贴纸、叠加文字的片段，最后人工复核。
  - 最终得 **1,000 训练 + 35 测试**片段，训练与测试来自完全不同的原始视频。

#### 模块2: 前景片段抽取（Foreground Clip Extraction）

**设计动机**: 提供从少数行人到密集人群、覆盖不同"人占比"的真实前景。

**具体实现**:
- 从 10 个城市的 10 个含行人视频抽取不重叠 7 秒前景片段。
- 用 [[Grounded-SAM-2]]（"person, bag, backpack" 提示）检测，设最小有效帧数阈值 $M=138$（占 7 秒片段时长的 $70\%$，即至少 70% 的帧检出掩码）。
- 计算每个片段的 **`Crowd%`**（= 所有帧的平均掩码面积占比），划分到 5 个区间：0–10%、10–20%、20–30%、30–40%、40–50%；每区间随机采 200 个，得 **1,000 个前景片段，人群规模均匀分布**。
- 评估集：另取 7 城 7 个测试背景视频，合成覆盖所有 `Crowd%` 区间的前景，每城每区间人工挑 1 个，共 **35 个测试片段**。

#### 模块3: 合成场景生成（Composite Scene Generation）

**设计动机**: 仅"抠人贴到背景"会丢失阴影等关联效果，导致不真实；需注入阴影并构造训练三元组。

**具体实现（阴影模拟，见 Figure 3 / Figure 4）**:
- 对每个分割出的人，先估计其与地面接触的**枢轴点（pivot point）**。
- 以枢轴为基准生成阴影几何：依太阳/角度做水平翻转（角度 $<90^{\circ}$ 不翻、$\geq 90^{\circ}$ 翻转）并施加仿射变换，缩放比例从 $s_{x}\sim\mathcal{U}(0.15,0.35)$、$s_{y}\sim\mathcal{U}(0.8,0.95)$ 采样，得到随机强度与角度的软阴影。
- **合成与三元组构造**:
  - 先用生成的阴影图、以随机阴影强度 $\alpha\in[0.2,0.8]$ 压暗背景（见 Figure 4，$\alpha$ 越大阴影越深）。
  - 再以全不透明（$\alpha=1$）把分割人物前景叠加到带阴影背景上。
  - 配对成三元组 **(input 合成图, mask 人物掩码, ground truth 干净背景)**。关键设计：**前景掩码不包含阴影区域**，迫使模型在训练中**隐式学习"人—投影阴影"的关联**（即模型必须自己推断该连阴影一起去除）。

### CrowdEraser 模型

**设计动机**: 在已具备"物体-效果关联"能力的 [[Casper]]（基于 [[CogVideoX]] 的视频扩散模型）上做领域适配微调，专门用于漫步视频去人。

**具体实现**:
- 输入：一段第一人称视频片段 + 对应的逐帧人物掩码片段 + 文本提示（所有实验统一用 "A video of a beautiful empty, human-free scene."）。Casper 本可同时生成背景层和各物体层，本任务**只取生成的背景**。
- 冻结 Casper 的 VAE 与文本编码器，**仅微调其 3D Transformer 的部分层**，在 EgoCrowds 的"有人/无人"配对上训练。
- 损失：在基础去噪损失外，**新增时序运动损失**约束相邻帧噪声残差的时间差，鼓励平滑时序动态。

### 关键公式与机制

#### 公式1: [[Diffusion]] 基础去噪损失

$$
\mathcal{L}_{\text{base}}=\|\hat{\epsilon}_{t}-\epsilon_{t}\|_{2}^{2}
$$

**含义**: 标准扩散去噪目标——让模型在帧 $t$ 预测的噪声逼近真值噪声。

**符号说明**:
- $\hat{\epsilon}_{t}$: 模型在帧 $t$ 预测的噪声
- $\epsilon_{t}$: 帧 $t$ 的 ground-truth 噪声

#### 公式2: 时序运动（sub）损失

$$
\mathcal{L}_{\text{sub}}=\|(\hat{\epsilon}_{t+1}-\hat{\epsilon}_{t})-(\epsilon_{t+1}-\epsilon_{t})\|_{2}^{2}
$$

**含义**: 约束**相邻帧噪声残差的时间差**，即让预测噪声沿帧轴的时间导数与真值一致，从而抑制帧间抖动、强化时序平滑。

**符号说明**:
- $\hat{\epsilon}_{t+1}-\hat{\epsilon}_{t}$: 预测噪声沿帧轴的一阶时间差分
- $\epsilon_{t+1}-\epsilon_{t}$: 真值噪声的一阶时间差分

#### 公式3: 组合损失

$$
\mathcal{L}=(1-\alpha)\,\mathcal{L}_{\text{base}}+\alpha\,\mathcal{L}_{\text{sub}}
$$

**含义**: 基础去噪损失与时序运动损失的加权和。

**符号说明**:
- $\alpha$: 运动 sub 损失的权重比例，本文取 $\alpha=0.25$（注意：此处 $\alpha$ 是损失权重，与数据合成阴影强度的 $\alpha\in[0.2,0.8]$ 含义不同，符号复用需留意）

---

## 关键图表

<!-- 图片均使用 arXiv HTML 在线链接，未本地化 -->

### Figure 1: Example Results of CrowdEraser / 方法效果总览

![Figure 1](https://arxiv.org/html/2603.29036v1/assets/figures/teaser-figure2-compressed.png)

**说明**: CrowdEraser 在三段第一人称漫步视频上的去人效果。输入为视频 + 前景人物掩码（上），输出为去除行人及其阴影后的干净视频。证明其在 (a,c) 人群拥挤、(b) 行人贴近拍摄者这两类"高人类存在"场景下都能良好工作。

### Figure 2: Locations in EgoCrowds / 背景片段地理分布

![Figure 2](https://arxiv.org/html/2603.29036v1/assets/figures/temp-location-map-viz.png)

**说明**: EgoCrowds 背景片段来源城市的世界地图。绿色为训练地点、红色为测试地点，覆盖大城市、大学城与不同建筑风格的小城，体现数据的全球结构多样性。

### Figure 3: Data Construction Pipeline / 数据构造管线

![Figure 3](https://arxiv.org/html/2603.29036v1/assets/figures/qual-grid3-compressed2.png)

> 注：该图实际渲染为 `qual-grid3-compressed2.png`（HTML 中 Figure 3 对应数据管线示意图 `x1.png`）。

![Figure 3 pipeline](https://arxiv.org/html/2603.29036v1/x1.png)

**说明**: 数据构造管线。背景与前景片段均取自真实漫步视频；前景按 `Crowd%` 等级近似均匀分布选取；对每个实例用对人物掩码做仿射变换生成随机强度/角度的软阴影（红点为枢轴点）。这是本文"零真实配对也能训"的关键。

### Figure 4: Shadow Injection with Varying α / 阴影强度注入

![Figure 4](https://arxiv.org/html/2603.29036v1/x2.png)

**说明**: 阴影注入随强度 $\alpha$ 从 0.2 (a) 到 0.8 (d) 变化，阴影逐渐变深。展示合成阴影的可控随机增强，使前景叠加更真实。

### Figure 5: Qualitative Comparison / 定性对比

![Figure 5](https://arxiv.org/html/2603.29036v1/assets/figures/qual-grid3-compressed2.png)

**说明**: 与基线的定性对比。红框=未能去除人/阴影的失败；黄框=背景被过度平滑而非填入合理内容。人少、背景简单时（如 (a) Jakarta）各方法都还行；但掩码越大、背景越复杂性能越差：[[ProPainter]] 与 [[DiffuEraser]] 在锐利投影阴影上失败，[[Casper]] 效果关联能力强但大掩码时常幻觉出物体/人。CrowdEraser 在大掩码下更鲁棒、更好保留背景结构。

### Figure 6: Ablation Across Temporal Frames / 时序帧上的消融

![Figure 6](https://arxiv.org/html/2603.29036v1/x3.png)

**说明**: 跨时序帧的消融可视化。红框=阴影捕捉失败，黄框=补全细节放大对比。(a) 原始 Casper 基线；(b) 用本文数据微调但**无阴影模拟**；(c) 仅用**小掩码**（Crowd% 0–10）训练；(d) **完整模型**（阴影注入 + 跨 Crowd% 均匀掩码分布）。完整模型在长时遮挡（如黄衣人长期挡住画面中心）下避免了色彩渗漏。

### Figure 7: SpatialTrackerV2 4D Reconstruction / 4D 重建对比

![Figure 7](https://arxiv.org/html/2603.29036v1/x4.png)

**说明**: 用 [[SpatialTrackerV2]] 对三个场景做 4D 重建。上=原始漫步视频输入，下=CrowdEraser 去人版本。每图显示从末帧相机视角看到的 3D 点云，叠加彩色圈表示点轨迹（静态物体上的点本应不动）。对比红圈静态区域：原始视频的点轨迹移动更大（误差更大），去人后背景细节更好（如 Marrakech 墙面纹理）、点云更稠密（如 Istanbul 右侧）。

### Figure 8: Difficult Cases (Limitations) / 失败/局限案例

![Figure 8](https://arxiv.org/html/2603.29036v1/x5.png)

**说明**: CrowdEraser 的两类困难案例。(a) 长视频、持续大遮挡时，补全区域在整段视频上缺乏时序一致性（受扩散主干固定 85 帧片段长度所限）；(b) 高 `Crowd%` 时难恢复细粒度高频细节，产生伪影。

### Figure 9: Full Baseline Comparison (Fig.1 scenes) / 完整基线对比（附录）

![Figure 9](https://arxiv.org/html/2603.29036v1/assets/figures/teaser-grid-suppl_box-compressed2.png)

**说明**: 针对 Figure 1 三个场景的完整基线对比。红框=前景去除/阴影处理失败，黄框=模糊补全伪影。ProPainter 与 DiffuEraser 在锐利投影阴影上失败、大掩码区域模糊丢细节；Casper 效果关联可靠但大掩码时明显幻觉；CrowdEraser 在大掩码下保持鲁棒、背景结构更可信。

### Figure 10: Baseline Comparison — Marrakech / 附录时序对比（Marrakech）

![Figure 10](https://arxiv.org/html/2603.29036v1/x6.png)

**说明**: 针对 Figure 5 "Marrakech" 场景的跨帧基线对比。Casper 大掩码时明显幻觉，DiffuEraser 倾向把遮挡区纹理过度平滑。

### Figure 11: Baseline Comparison — Jakarta / 附录时序对比（Jakarta）

![Figure 11](https://arxiv.org/html/2603.29036v1/x7.png)

**说明**: 针对 "Jakarta" 场景的跨帧对比。DiffuEraser 难以捕捉阴影，导致路面上出现"漂浮阴影"。

### Figure 12: Baseline Comparison — Istanbul / 附录时序对比（Istanbul）

![Figure 12](https://arxiv.org/html/2603.29036v1/x8.png)

**说明**: 针对 "Istanbul" 场景的跨帧对比。Casper 大掩码时幻觉，DiffuEraser 难处理阴影并难以连带去除关联物体。

### Figure 13: Baseline Comparison — Chicago / 附录时序对比（Chicago）

![Figure 13](https://arxiv.org/html/2603.29036v1/x9.png)

**说明**: 针对 "Chicago" 场景的跨帧对比。DiffuEraser 难以关联阴影与物体，导致漂浮的阴影和物体。

### Figure 14: SpatialTrackerV2 4D Reconstruction (more) / 附录 4D 重建补充

![Figure 14](https://arxiv.org/html/2603.29036v1/x10.png)

**说明**: 更多 4D 重建结果（初/中/末帧视角）。去人后静态背景区域跟踪点更稳定，点云更稠密一致，有利于场景建模与 3D 新视角合成。

### Figure 15: SpatialTrackerV2 4D Reconstruction (more) / 附录 4D 重建补充

![Figure 15](https://arxiv.org/html/2603.29036v1/x11.png)

**说明**: 同 Figure 14，另一组场景的 4D 重建补充，结论一致：去人显著提升重建稳定性与点云密度。

### Table 1: Quantitative Comparison Across Crowd% Levels / 按人群占比的定量对比

在 7 城 35 个合成测试视频上评估去除-补全质量。PSNR 越高越好，LPIPS、DreamSim 越低越好。**加粗 = 最优**。

| Method | (0–10) PSNR↑ | LPIPS↓ | DreamSim↓ | (10–20) PSNR↑ | LPIPS↓ | DreamSim↓ | (20–30) PSNR↑ | LPIPS↓ | DreamSim↓ | (30–40) PSNR↑ | LPIPS↓ | DreamSim↓ | (40–50) PSNR↑ | LPIPS↓ | DreamSim↓ | Avg PSNR↑ | Avg LPIPS↓ | Avg DreamSim↓ |
|--------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|
| ProPainter [48] | 31.81 | **0.058** | 0.011 | 28.87 | 0.088 | 0.019 | 25.50 | 0.128 | 0.056 | 21.98 | 0.192 | 0.114 | 20.93 | 0.238 | 0.134 | 25.82 | 0.141 | 0.067 |
| DiffuEraser [18] | 31.91 | **0.055** | 0.009 | 28.73 | **0.079** | 0.014 | 24.68 | 0.112 | 0.035 | 22.18 | 0.171 | 0.067 | 22.27 | 0.181 | 0.062 | 25.95 | **0.120** | 0.037 |
| Casper [17] | 31.41 | 0.073 | 0.010 | 28.72 | 0.092 | 0.014 | 23.97 | 0.118 | 0.025 | 19.88 | 0.185 | 0.052 | 19.24 | 0.185 | 0.052 | 24.64 | 0.130 | 0.029 |
| **Ours (CrowdEraser)** | **32.39** | 0.065 | **0.007** | **29.36** | 0.086 | **0.011** | **26.31** | **0.108** | **0.020** | **22.34** | **0.170** | **0.039** | **23.31** | **0.161** | **0.031** | **26.74** | **0.118** | **0.022** |

**说明**: CrowdEraser 在所有 Crowd% 区间的 **PSNR 与 DreamSim 全部最优**，平均 PSNR 26.74、DreamSim 0.022 明显领先；基线仅在低 Crowd% 时勉强可比，高人群密度时性能急剧下降。LPIPS 上偶有 DiffuEraser/ProPainter 在极低 Crowd% 时略低，但平均仍由本文领先。

### Table 2: Ablation on Data Construction Components / 数据构造组件消融

第一行为未微调的原始 Casper。"Shadow" = 是否注入模拟阴影；"Full Crowd%" = 是否使用跨 0–50% 均匀掩码分布（✗ 表示仅用 0–10 的小掩码）。**加粗 = 最优**。

| Shadow | Full Crowd% | PSNR↑ | SSIM↑ | LPIPS↓ | DreamSim↓ |
|--------|-------------|-------|-------|--------|-----------|
| ✗ | ✗ (原始 Casper) | 24.64 | 0.868 | 0.130 | 0.029 |
| ✗ | ✓ | 25.09 | 0.870 | 0.128 | 0.028 |
| ✓ | ✗ | 26.60 | **0.882** | 0.120 | 0.024 |
| ✓ | ✓ (完整模型) | **26.74** | 0.881 | **0.118** | **0.022** |

**说明**: 去掉阴影注入时模型能填大掩码但无法正确关联阴影；只用小掩码训练时遇到"人群长期占据画面"会失败（色彩渗漏，见 Figure 6）。两个组件叠加给出最佳综合指标——阴影模拟贡献最大（PSNR 24.64→26.60），均匀 Crowd% 分布进一步小幅提升并稳住高密度场景。

### Table 3: Quantitative Comparison Across Cities / 按城市的定量对比

**加粗 = 最优**（PSNR↑、DreamSim↓）。

| Method | Birmingham PSNR↑ | DreamSim↓ | Boston PSNR↑ | DreamSim↓ | Capetown PSNR↑ | DreamSim↓ | Chicago PSNR↑ | DreamSim↓ | Dubai PSNR↑ | DreamSim↓ | Rome PSNR↑ | DreamSim↓ | Zurich PSNR↑ | DreamSim↓ |
|--------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|
| ProPainter [48] | **27.77** | 0.052 | 26.00 | 0.049 | **25.86** | 0.059 | 25.91 | 0.053 | 22.34 | 0.050 | 29.97 | 0.019 | 25.76 | 0.059 |
| DiffuEraser [18] | 27.25 | 0.034 | 25.49 | 0.042 | 25.43 | 0.050 | 25.59 | 0.036 | 22.65 | 0.042 | 29.86 | 0.013 | 25.14 | 0.049 |
| Casper [17] | 26.15 | 0.028 | 25.12 | 0.031 | 24.20 | 0.035 | 26.02 | 0.022 | 23.14 | 0.027 | 29.44 | 0.012 | 25.21 | 0.027 |
| **Ours (CrowdEraser)** | 26.58 | **0.022** | **26.00** | **0.025** | 25.50 | **0.026** | **27.08** | **0.019** | **24.81** | **0.019** | **30.31** | **0.009** | **25.99** | **0.022** |

**说明**: 跨 7 城对比，CrowdEraser 在**所有城市的 DreamSim（感知质量）均最优**，且多数城市 PSNR 也领先（个别如 Birmingham/Capetown 的 PSNR 被 ProPainter 略高，但其感知相似度 DreamSim 远差）。说明本文方法的优势是更"感知可信"的补全，而非单纯像素拟合。

### Table 4–6: Dataset Sources / 数据来源（附录）

| 表 | 内容 | 要点 |
|----|------|------|
| Table 4 | 训练数据来源 | 训练背景视频来自全球 50 城（非洲 Cairo、亚洲 Beijing/Shanghai/Seoul、欧洲 Paris/Berlin/London、北美众多城市等）；训练前景来自 Varanasi、Seoul、Hamburg、Amsterdam、Anaheim、Honolulu、NYC 等含行人视频。 |
| Table 5 | 定量测试来源 | 背景 7 城：Cape Town、Dubai、Rome、Zurich、Birmingham、Boston、Chicago；前景 5 处：Hong Kong、Zurich、Vienna、Houston、Los Angeles。 |
| Table 6 | 定性测试来源 | 真实人群漫步视频 18 城：Mumbai、Tokyo、Kyoto、Bangkok、Jakarta、Cape Town、Marrakech、Lagos、NYC、Denver、Chicago、Cancun、Rio、Buenos Aires、Dubrovnik、Stockholm、Seville、Istanbul。 |

**说明**: 三表逐城列出 YouTube 源 URL，体现 EgoCrowds 的全球多样性与可追溯性（背景/前景/测试集来自完全不同的原始视频以避免泄漏）。

---

## 实验

### 数据集 / 基准

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| EgoCrowds（训练） | 1,000 对 7 秒片段（197 帧@16fps） | 半合成：真实空背景 + 真实行人前景 + 模拟阴影；前景跨 5 个 Crowd% 区间均匀分布 | 微调训练 |
| EgoCrowds（测试） | 35 个合成视频（7 城 × 5 Crowd% 区间） | 含 ground-truth 干净背景，可算 PSNR/LPIPS/DreamSim | 定量评估 |
| 真实人群漫步视频 | 18 城（Table 6） | 真实高人群存在、无 ground-truth | 定性评估 |

### 实现细节

- **基础模型**: [[Casper]]（来自 Generative Omnimatte，基于 [[CogVideoX]] 扩散模型）的公开实现。
- **微调策略**: 冻结 VAE 与文本编码器，微调 3D Transformer 的若干层，训练 **100 epochs**。
- **损失**: $\mathcal{L}=(1-\alpha)\mathcal{L}_{\text{base}}+\alpha\mathcal{L}_{\text{sub}}$，$\alpha=0.25$。
- **推理**: 输入分辨率 $720\times1080$，统一文本提示 "A video of a beautiful empty, human-free scene."
- **硬件/时长**: 4 × H200 GPU，约 **15 小时**。
- **基线**: Casper、[[ProPainter]]、[[DiffuEraser]]。
- **指标**: PSNR（重建质量）、LPIPS / DreamSim（感知相似度）。

### 关键实验结论

- **定量（Table 1/3）**: CrowdEraser 在所有 Crowd% 区间的 PSNR/DreamSim 全面最优（平均 PSNR 26.74、DreamSim 0.022），跨 7 城 DreamSim 全部最优；优势随人群密度增大而扩大。
- **定性（Figure 5/9–13）**: 大掩码、复杂背景下显著优于基线——ProPainter/DiffuEraser 在锐利阴影与大掩码处过度平滑/丢细节，Casper 大掩码时幻觉物体，本文更好保留背景结构。
- **消融（Table 2 / Figure 6）**: 阴影模拟与跨 Crowd% 均匀掩码分布缺一不可，前者使模型学会"人—阴影关联"，后者强化结构补全、避免高密度场景模糊/变暗。
- **下游 3D（Figure 7/14/15）**: 去人后用 SpatialTrackerV2 重建，静态区域点轨迹更稳、点云更稠密一致，验证"去人"是漫步视频用于城市建模的关键预处理。

---

## 批判性思考

### 优点
1. **问题切入精准、动机扎实**: 把性能瓶颈定位为"域偏移"而非模型设计，并据此把全部精力投在数据集构造上——这一判断由"换数据微调即大幅超越原 Casper"的结果有力支撑。
2. **半合成数据范式可复用**: "真实空背景 + 真实前景 + 规则化模拟阴影 + 掩码不含阴影"的构造既保真又自带监督，且掩码故意排除阴影以隐式学习人-阴影关联，设计巧妙；EgoCrowds 全球 50+ 城来源公开可追溯。
3. **闭环到下游价值**: 不止做去人指标，还用 SpatialTrackerV2 证明去人能让 4D 重建更稳更稠密，把"视频清洗"与"大规模城市场景建模"这一真实需求接上。

### 局限性
1. **时序一致性受架构限制**: 扩散主干处理固定 85 帧片段，长视频持续大遮挡时补全缺乏全局时序一致性（作者自己在 Figure 8a 承认），缺少长程记忆/跨片段条件机制。
2. **高 Crowd% 高频细节差**: 高人群密度下难恢复细粒度高频纹理、产生伪影（Figure 8b）；且评测上限仅到 Crowd% 40–50%，更极端拥堵未覆盖。
3. **评测仍偏定性 + 半合成**: 定量指标全部建立在**半合成测试集**（ground-truth 本身是合成）上，对真实人群视频只能定性评估；阴影模拟为规则化仿射近似，未必匹配真实光照几何，可能引入合成域偏置。
4. **未与 Omnimatte3D/OmnimatteRF 等几何方法直接定量对比**，基线集中在补全类方法。

### 潜在改进方向
1. 引入**时序记忆/滑窗重叠或跨片段条件**以解决长视频一致性；作者亦提出引入 **3D 几何约束**提升高密度场景结构保真。
2. 用**物理/可学习阴影渲染**替代规则仿射阴影，缩小合成-真实域差距；并探索在真实视频上构建/借用弱监督评测。
3. 扩展到更高 Crowd%、动态非人物体（车辆、鸟群）与夜间/恶劣光照场景的去除。

### 可复现性评估
- [ ] 代码开源（论文未给出代码链接）
- [ ] 预训练模型（未明确 release CrowdEraser 权重）
- [x] 训练细节较完整（基础模型、微调层、损失/权重、epoch、硬件、推理分辨率与提示词均给出）
- [x] 数据集可获取（基于公开 Casper 实现 + Table 4/5/6 逐城列出所有 YouTube 源 URL，可按管线复现 EgoCrowds）

---

## 速查卡片

> [!summary] CrowdEraser / EgoCrowds — 从第一人称漫步视频去人
> - **核心**: 把"漫步视频去人"瓶颈定位为域偏移，构造半合成配对数据集 EgoCrowds 微调视频扩散模型 Casper，得到 CrowdEraser。
> - **数据**: 真实空背景 + 真实行人前景 + 规则化模拟软阴影，1,000 训练对（跨 5 个 Crowd% 区间均匀）；掩码故意不含阴影以隐式学人-阴影关联。
> - **模型**: 冻结 VAE/文本编码器，微调 Casper(CogVideoX) 3D Transformer，损失 = 去噪 + 0.25×时序运动；4×H200 训 15 小时。
> - **结果**: 平均 PSNR 26.74 / DreamSim 0.022，全 Crowd% 区间 PSNR&DreamSim 最优；消融证明阴影模拟 + 均匀 Crowd% 缺一不可；去人后 SpatialTrackerV2 4D 重建更稳更稠密。
> - **局限**: 长视频时序一致性、高密度高频细节、半合成评测。

---

*笔记创建时间: 2026-06-29*
