---
title: "E3AD: An Emotion-Aware Vision-Language-Action Model for Human-Centric End-to-End Autonomous Driving"
method_name: "E3AD"
authors: [Yihong Tang, Haicheng Liao, Tong Nie, Junlin He, Ao Qu, Kehua Chen, Wei Ma, Zhenning Li, Lijun Sun, Chengzhong Xu]
year: 2026
venue: CVPR
tags: [VLA, autonomous-driving, emotion-aware, VAD-emotion, spatial-reasoning, DPO, end-to-end-driving]
zotero_collection: _inbox
image_source: online
arxiv_html: https://arxiv.org/html/2512.04733v3
created: 2026-06-29
---

# E3AD: An Emotion-Aware Vision-Language-Action Model for Human-Centric End-to-End Autonomous Driving

## 元信息

| 项目 | 内容 |
|------|------|
| 作者 | Yihong Tang, Haicheng Liao, Tong Nie, Junlin He, Ao Qu, Kehua Chen, Wei Ma, Zhenning Li, Lijun Sun, Chengzhong Xu |
| 机构 | McGill University、澳门大学（University of Macau）、香港理工大学、MIT 等（跨校合作） |
| 会议 | CVPR 2026 |
| 类别 | VLA（Vision-Language-Action）/ 端到端自动驾驶 |
| 日期 | 2025-12（arXiv v3） |
| 项目主页 | （论文未给出公开仓库） |
| 链接 | [arXiv](https://arxiv.org/abs/2512.04733) / [CVPR poster](https://cvpr.thecvf.com/virtual/2026/poster/36906) |

---

## 一句话总结

> 在 VLA 端到端驾驶中显式注入“乘客情绪”——用连续的 [[VAD 情绪模型]] 解析指令语气、用[[自我中心-非自我中心双通路]]做类人空间推理、再用一致性导向的 [[DPO]] 对齐情绪与轨迹，使接地、规划与反馈都更贴合人类意图。

---

## 核心贡献

1. **提出 Open-Domain End-to-End（OD-E2E）驾驶任务**: 让自动驾驶车（AV）从自由形式自然语言指令中**统一推理语义、情绪与空间**，定位被指代目标并规划物理可行轨迹，把“语言接地”从辅助感知提升为端到端决策目标的内在组成部分。
2. **提出 E3AD 情绪感知 VLA 框架**: 在统一流水线中集成**连续情绪建模**（[[VAD]] 空间）与**双系统空间推理**（egocentric + allocentric），实现情绪接地的响应与规划，而非仅做理性的序列预测。
3. **一致性导向训练 + 全面 SOTA**: 用“模态预训练 → 联合微调 → 情绪-动作对齐（DPO）”三阶段训练，在视觉接地、情绪估计、轨迹规划三类任务、四个真实世界基准上超过强基线，**在情绪敏感与 corner-case 场景上增益尤为显著**。

---

## 问题背景

### 要解决的问题
端到端（E2E）自动驾驶日益采用 [[VLA]] 模型，但它们普遍是**情绪无关（emotion-agnostic）**的：只把多视角图像直接映射到规划输出，既不理解乘客意图的“语气/紧迫度”，也没有人在回路的交互与反馈。然而乘客对“把决策交给不透明算法”的不安，恰恰是 AV 公众信任与接受度的核心瓶颈——作者称之为自动驾驶的 **emotion gap（情绪鸿沟）**。

举例：“stop here” 与 “stop here now!” 语义相同，但语气携带的情绪差异应当改变车辆响应方式。一个智能系统不仅要理解乘客“说了什么（what）”，还要理解“怎么说的（how）”。

### 现有方法的局限
作者把现有 VLA-for-AD 归为三类范式，均有缺陷：
1. **“解说员”范式**（DriveGPT-4、OpenEMMA、CoT-Drive）: 用 QA 式提示产生场景级解释，**缺乏精确空间接地与控制保真度**；
2. **“元行为”范式**（Senna、VLP、LMDrive）: VLM 产生离散 meta-behavior 引导底层控制器，**指导稀疏、缺乏连续空间推理**，驾驶增益有限；
3. **“VLM 感知 + 专用规划”范式**（Simlingo、AutoVLA、FSDrive）: 直接输出轨迹/控制信号、性能最强，但仍有两个核心问题——**空间理解弱**（大多在 2D 操作，缺乏显式 3D 或 map-based 的 allocentric 推理）、**纯理性序列预测**（完全忽略对行为对齐至关重要的乘客情绪）。

情绪计算在 AD 中过往工作多为**被动检测 + 离散标签**，且把情绪估计与下游控制**解耦**，无法捕捉细微但行为上有意义的语气变化。

### 本文的动机
- 沿用第三类范式但加以补强：用**连续 [[VAD]] 空间**（而非离散标签）表征情绪，把情绪**嵌入同一生成式推理过程**，让它既能解析“歧义指令”，又能直接条件化轨迹生成；
- 借鉴人类空间感知的**双系统模型**，用 egocentric（第一人称、动作导向）+ allocentric（世界中心、地图结构）双通路融合局部感官线索与世界先验；
- 用**一致性导向训练**强制“情绪意图 ↔ 驾驶动作”的相干性，从“情绪识别”推进到真正的人本（human-centric）自动驾驶系统。核心思想：**把情绪从被动标签变成连续控制信号**，在固定意图下选择行为上不同但物理可行的方案。

---

## 方法详解

### 模型架构

E3AD 是一个认知启发的 VLA 系统，**主干为 [[Qwen2.5-VL]]-7B-Instruct**，在统一流水线中集成三大能力（见 Figure 2）：
- **输入**: 多视角观测 $I=\{I_{\text{ego}}, I_{\text{allo}}\}$（自我中心视图 + 非自我中心 BEV 视图）+ 自然语言指令 $C$
- **核心模块**: [[Emotion Modeling|情绪建模]]（把指令编码到连续 VAD 空间）+ [[Two-System Spatial Reasoning|双系统空间推理]]（egocentric / allocentric 双通路）+ [[Action Decoder|动作解码器]] + [[Human-centric Verbal Feedback|人本语言反馈]]
- **输出**: 情绪状态 $\hat e$、接地目标 $\hat b$、未来轨迹路点 $\hat\tau=\{y_t\}_{t=1}^T$、以及面向乘客的语言反馈 $\hat r$
- **训练**: 三阶段——模态预训练 → 联合微调 → 情绪-动作对齐（[[DPO]]）；全程冻结 Qwen2.5-VL-7B 主干、仅训 [[LoRA]] 低秩适配器（rank 16, scale 32）

### 核心模块

#### 模块1: Emotion Modeling（连续情绪建模）

**设计动机**: 多数系统用 happy/angry/sad 这类离散标签建模情绪，无法捕捉“细微却影响执行方式”的语气变化。E3AD 改用连续 [[VAD]]（Valence-Arousal-Dominance，效价-唤醒-支配）模型，把情绪表示为 $e\in\mathbb{R}^3$。在驾驶语境下三轴对应：**态度**（平静 vs 焦虑）、**警觉度**（疲惫 vs 警惕）、**控制感**（自信 vs 不知所措）。

**具体实现**:
- **VAD 标签生成（句级 + 词级双源）**: 对每条指令 $C$，先用 [[GoEmotions]] 分类器得到离散情绪分布，再经 label-VAD 字典映射到**句级 VAD**；并行地去除停用词、对剩余 token 的词典分数取平均得到**词级 VAD**；二者结合作为最终标签 $e$，兼顾全局理解与情绪承载短语。
- **情绪感知指令增广（emotion-aware command augmentation）**: 驾驶指令常情绪中性，naive 训练会让模型忽略情绪。于是用 Qwen2.5-VL 对每条指令 $C^{(i)}$ 生成 $K$ 条**保持驾驶目标不变、但改变态度/强度**的改写 $C^{(i)}_{\text{aug}}=\{C^{(i)}_1,\dots,C^{(i)}_K\}$，各自打 VAD 标签，构成增广集 $\mathcal{C}^*$（见 Figure 3）。这迫使模型把 $e$ 的变化归因于**语气**而非意图。
- **情绪回归即条件生成**: 不加单独情感头，而是用指令式模板把情绪预测当作量化 VAD token 的条件生成（公式2），与其他输出共用同一生成式推理过程。

#### 模块2: Spatial Reasoning（双系统空间推理）

**设计动机**: 借鉴人类空间感知的**双系统模型**——用 egocentric 帧做即时、动作导向的感知，用 allocentric 帧做全局、地图式结构推理，让模型像人一样把第一人称观测与内部认知地图结合。

**具体实现**:
- **Egocentric Pathway（自我中心通路）**: 从 $(I_{\text{ego}}, C)$ 预测 (i) 到被指代物的相对 3D 方向、(ii) 距离、(iii) 图像坐标中的接地位置；提供细粒度、短时程空间线索。用 **30K 样本**建立稳健的自我中心推理基础。
- **Allocentric Pathway（非自我中心通路）**: 给定 BEV 输入 $I_{\text{allo}}$，预测 (i) BEV 坐标下的目标位置、(ii) 从自车位姿到目标的粗轨迹 $\tau=\{y_t\}_{t=1}^T$；提供长时程结构、道路拓扑、遮挡与多智能体布局等 map-consistent 先验。用约 **17K 样本**训练。
- 两通路互补的局部/全局空间线索，供下游接地与路点生成在杂乱、部分可观场景中使用。

#### 模块3: Action Decoder + Verbal Feedback（动作解码与语言反馈）

**设计动机**: 把 VLA 高层输出翻译成精确、物理可行的轨迹；并通过语言反馈缓解乘客的“黑箱焦虑”。

**具体实现**:
- **Action Decoder**: 在 VLA 主干后接一个轻量动作解码器 $f_{\text{act}}$，条件于接地目标 $\hat b$、粗轨迹 $\widetilde\tau$、视觉观测 $I$，输出最终轨迹 $\hat\tau=f_{\text{act}}(\hat b,\widetilde\tau,I)$，其中 $\hat\tau\in\mathbb{R}^{T\times 2}$ 为路点空间坐标。
- **Human-centric Verbal Feedback**: 用训练好的 Qwen2.5-VL 主干，在结构化提示引导下，条件于完整流水线输出 $(\hat e,\hat b,\hat\tau)$ 生成语言响应 $\hat r$。**响应策略随情绪/紧迫度自适应**：平静态给简短确认，高唤醒态给直接、时间敏感的引导，将 AV 从不透明工具变为人本智能体。

### 关键公式与机制

#### 公式1: [[OD-E2E]] 任务映射

$$
f_{\theta}:(I,C)\rightarrow\hat{\mathcal{Y}}=\{\hat{b},\hat{\tau}\}
$$

**含义**: 学习单一策略 $f_\theta$，在统一目标下由观测与指令**联合**预测接地目标与未来轨迹，而非把指代定位与运动规划拆成独立模块。

**符号说明**:
- $I=\{I_{\text{ego}},I_{\text{allo}}\}$: 多视角观测（自我中心 + 非自我中心）
- $C$: 乘客自然语言指令
- $\hat b$: 场景中被接地的目标；$\hat\tau=\{y_t\}_{t=1}^T$: 未来路点序列

#### 公式2: [[情绪回归损失]]（条件生成）

$$
\mathcal{L}_{\text{emo}}=-\mathbb{E}_{(C^{(i)}_{k},e^{(i)}_{k})\sim\mathcal{C}^{*}}\big[\log p_{\theta}(e^{(i)}_{k}\mid C^{(i)}_{k})\big]
$$

**含义**: 在增广指令-情绪对上做监督微调，让 VLM 把（量化的）VAD 标签作为条件生成目标，从而具备显式情绪理解。

**符号说明**:
- $\mathcal{C}^*$: 情绪感知增广后的指令集合
- $(C^{(i)}_k,e^{(i)}_k)$: 第 $i$ 条指令的第 $k$ 个增广改写及其 VAD 标签
- $p_\theta$: 模型在量化 VAD token 上的分布

#### 公式3: [[联合微调损失]]（自回归链式预测）

$$
\mathcal{L}_{\text{Joint}}=-\mathbb{E}_{(I,C,T)}\sum_{t=1}^{|T|}\log p_{\theta}(T_{t}\mid T_{<t},I,C)
$$

**含义**: 在单次前向中自回归预测完整输出序列 $\mathcal{T}=(\hat e,\hat b,\hat\tau)$，形成**情绪感知的思维链**——预测的情绪 $\hat e$ 与接地 $\hat b$ 直接为后续路点 $\hat\tau$ 的生成提供条件。

**符号说明**:
- $T=(\hat e,\hat b,\hat\tau)$: 由情绪、接地、轨迹拼成的输出 token 序列；$T_t$ 为第 $t$ 个 token，$T_{<t}$ 为其前缀
- $p_\theta$: 条件 token 分布

#### 公式4: [[伪偏好对构造]]（情绪扰动负样本）

$$
C^{(i)}_{k^{-}}=\arg\max_{k}\|e^{(i)}_{k}-e^{(i)}\|_{2},\quad \widetilde{\tau}^{(i)}_{k^{-}}\sim p_{\theta}(\tau\mid C^{(i)}_{k^{-}},I^{(i)})
$$

**含义**: AD 数据通常每条指令只有**单一真实轨迹**，无法直接做偏好排序。E3AD 取“VAD 偏离原指令最大”的增广变体 $C^{(i)}_{k^-}$ 作为负指令，用它生成一条**情绪偏移的 dispreferred 轨迹** $\widetilde\tau^{(i)}_{k^-}$，从而构造伪偏好对 $(\tau^{(i)}\succ\widetilde\tau^{(i)}_{k^-})$。

**符号说明**:
- $e^{(i)}$: 原指令的 VAD 向量；$e^{(i)}_k$: 第 $k$ 个增广的 VAD 向量
- $\arg\max_k\|\cdot\|_2$: 选取 VAD 欧氏偏离最大的增广作为负样本
- $\tau^{(i)}$: 原始真实（preferred）轨迹

#### 公式5: [[DPO|Emotion-Action Alignment 损失]]

$$
\mathcal{L}_{\text{dpo}}=-\mathbb{E}_{i}\Big[\log\sigma\!\Big(\beta\big(\log p_{\theta}(\tau^{(i)}\mid C^{(i)})-\log p_{\theta}(\widetilde{\tau}^{(i)}_{k^{-}}\mid C^{(i)})\big)\Big)\Big]
$$

**含义**: 用 [[DPO]] 优化上述伪偏好对——提升与原指令意图一致轨迹的似然、压低情绪扰动后的备选轨迹，得到**稳定且情绪感知**的驾驶行为。这是 $\mathcal{L}_{\text{Joint}}$ 无法显式保证的“行为-情绪一致性”。

**符号说明**:
- $\sigma$: Sigmoid 函数；$\beta$: DPO 温度/缩放系数
- $\tau^{(i)}$（preferred）、$\widetilde\tau^{(i)}_{k^-}$（dispreferred）: 偏好对中的正/负轨迹

---

## 关键图表

<!-- 图片均使用 arXiv HTML v3 在线链接，未本地化 -->

### Figure 1: Overview of OD-E2E / 任务范式对比

![Figure 1](https://arxiv.org/html/2512.04733v3/x1.png)

**说明**: (a) 现有 VLA 是情绪无关系统，把多视角图像直接映射到规划输出，无人在回路、无情绪理解；(b) E3AD 加入**显式情绪建模与闭环反馈**，能推断意图强度、更可靠地接地指代物并据此调整规划；(c) 由此得到 **OD-E2E** 任务：智能体在语言、情绪、感知、导航上联合推理，实现人本、情境感知的自治。是全文动机的总览图。

### Figure 2: E3AD Architecture & Training/Inference Pipeline / 总体架构与训练推理流程

![Figure 2](https://arxiv.org/html/2512.04733v3/x2.png)

**说明**: 给定 ego/allo 视图 + 语言指令 (a)，E3AD 经两大核心模块输出情绪、接地、路点 token——Emotion Modeling (b) 把指令编码到连续 VAD 空间 (c)，Spatial Reasoning 融合 ego/allo 通路线索。训练从模态预训练 (d) → 联合微调单链自回归预测 $(\hat e,\hat b,\hat\tau)$ (e) → 情绪-动作对齐 (f)；推理 (g) 端到端估计 $\hat e$、接地 $\hat b$、规划 $\hat\tau$。是方法的核心结构图。

### Figure 3: Emotion Distributions Before/After Augmentation / 情绪增广前后分布

![Figure 3](https://arxiv.org/html/2512.04733v3/Figures/vad_commands.png)

**说明**: (a) Talk2Car 各划分上 GoEmotion 类别占比；(b) GoEmotion 的 VAD 分布；(c) 引入驾驶指令后情绪多样性更丰富；(d) **情绪感知增广扩展并平滑了 VAD 分布**，提供更宽、连续的情绪监督。佐证“增广迫使模型把情绪变化归因于语气”的设计。

### Figure 4: DPO Effect on Emotion-Trajectory Consistency / DPO 对情绪-轨迹一致性的影响

![Figure 4](https://arxiv.org/html/2512.04733v3/x3.png)

**说明**: DPO 提升“直线度/曲折度”的 Spearman 相关，并稳定转弯平滑度与角度变化，产生几何相干、行为一致的路径。规律：**高唤醒 → 更直、更平滑的运动；低唤醒 → 更谨慎、更弯曲**。说明即便数值增益不大，DPO 也强化了情绪-轨迹一致性。

### Figure 5: Qualitative Comparison (E3AD vs FSDrive-FT) / 定性对比

![Figure 5](https://arxiv.org/html/2512.04733v3/x4.png)

**说明**: 三类挑战场景的对比——(a) 情绪丰富指令：E3AD 凭 VAD 情绪建模推断紧迫度、及时变道，无情绪模块的变体则犹豫滞后；(b) 多智能体遮挡：E3AD 融合 allo 拓扑与 ego 证据准确接地并给出安全轨迹，FSDrive-FT 接地偏离、规划不安全；(c) 歧义指令：E3AD 结合语气与空间上下文给出可行、map-consistent 的泊车路径，FSDrive-FT 越界进入不可行驶区域。

### Figure 6: Case Study — Neutral vs Cautious Command / 情绪指令案例分析

![Figure 6](https://arxiv.org/html/2512.04733v3/x5.png)

**说明**: 同一意图下，中性指令映射到 VAD $(0.60,0.39,0.45)$，加入“Be more cautious”后 VAD 升至 $(0.60,0.49,0.51)$（唤醒、支配上升）。中性 VAD 下 E3AD 规划标准变道；谨慎 VAD 下 DPO 对齐策略**直接放弃变道**。语言反馈也随 $\hat e$ 调整（EmoThink 块给出安抚性解释）。证明 **VAD 是连续控制信号**，在物理可行方案间做选择，而非二值开关或风格 token。

### Figure 7: User Study / 用户研究

![Figure 7](https://arxiv.org/html/2512.04733v3/Figures/user_study_main.png)

**说明**: 217 名参与者对五个系统的匿名片段在 Command Compliance、Emotion Alignment、Safety、Comfort、Overall Preference 上打分/排序。（左）E3AD 在各年龄组都拿到高 Likert 分；（右）Rank-1 投票对比，E3AD 在多数维度占主导，超过所有基线。无情绪建模的系统（FSDrive-FT、通用 VLM）常被认为犹豫或不安全。

### Table 1: End-to-End Performance / 端到端轨迹规划性能

Best 加粗，括号内为相对最强基线（PTPC）的相对提升；$\pm$ 为标准差。

| Model | ADE ↓ | Fréchet ↓ | SSPD ↓ | DTW ↓ | FDE ↓ | PA₂ ↑ | PA₄ ↑ |
|-------|------|-----------|--------|-------|-------|-------|-------|
| A*-ROL | 5.63 | 10.22 | 3.06 | 93.35 | 9.34 | 5.45 | 25.13 |
| A*-PE (PDPC) | 5.35 | 9.38 | 2.80 | 86.78 | 8.22 | 25.95 | 46.41 |
| GoalGAN | 5.89 | 10.86 | 3.09 | 98.75 | 10.08 | 17.32 | 33.82 |
| PECNet | 4.78 | 8.84 | 2.54 | 76.87 | 8.22 | 15.59 | 37.68 |
| Y-net | 5.28 | 10.01 | 2.50 | 85.25 | 8.98 | 2.49 | 30.84 |
| TTST | 5.24 | 9.79 | 2.47 | 84.02 | 8.50 | 13.98 | 37.54 |
| CWS | 4.82 | 9.30 | 2.46 | 78.69 | 8.59 | 3.27 | 34.66 |
| TTST + CWS | 4.76 | 8.95 | 2.41 | 76.84 | 8.08 | 17.54 | 40.79 |
| PTPC | 4.54 | 8.55 | 2.18 | 72.09 | 7.75 | 24.46 | 45.55 |
| Qwen2.5-VL-72B | 12.51 | 26.15 | 5.87 | 206.99 | 25.85 | 1.18 | 3.13 |
| Qwen3-VL-8B | 14.07 | 28.38 | 6.65 | 234.81 | 27.96 | 1.89 | 4.50 |
| FSDrive-Finetuned | 5.02 | 10.98 | 2.28 | 74.50 | 10.45 | 17.10 | 27.85 |
| CAVG (+Planner) | 4.88 | 9.23 | 2.20 | 73.51 | 9.23 | 20.10 | 43.25 |
| **E3AD (Ours)** | **3.88** | **7.23** | **1.86** | **60.07** | **6.64** | **36.21** | **55.62** |
| *相对 PTPC 提升* | *17.01%↑* | *18.26%↑* | *17.20%↑* | *16.67%↑* | *20.00%↑* | *16.71%↑* | *18.10%↑* |

**说明**: E3AD 在全部七项轨迹指标上超过所有基线：相对最强基线 PTPC，ADE/Fréchet/FDE 分别降低 17.01%/18.26%/20.00%，SSPD/DTW 改善 >16%，规划准确率 PA₂/PA₄ 提升 +16.71%/+18.10%。通用 VLM（Qwen-VL）表现极差（FDE >25），微调过的 FSDrive 也明显落后（FDE 10.45 vs 6.64）。

### Table 2: Visual Grounding / 视觉接地（IoU%，跨多基准与 corner-case）

| Model | Backbone | T2C test | MoCAD test | MoCAD val | DrivePilot test | DrivePilot val | Vis. Constr. | Multi-agent | Ambiguous | Long-text val |
|-------|----------|----------|------------|-----------|-----------------|----------------|--------------|-------------|-----------|---------------|
| AttnGrounder | ResNet-50 | 61.32 | 62.34 | 64.35 | 62.31 | 64.57 | 62.74 | 64.82 | 64.31 | 57.25 |
| CMSVG | EfficientNet | 68.61 | 67.66 | 68.47 | 68.87 | 69.93 | 69.39 | 66.77 | 67.83 | 62.21 |
| TransVG | ResNet-101 | 65.83 | 68.14 | 70.85 | 66.52 | 68.42 | 68.12 | 66.34 | 69.25 | 65.45 |
| CMRT | ResNet-152 | 69.11 | 69.42 | 68.83 | 69.54 | 70.37 | 67.12 | 66.20 | 62.23 | 64.25 |
| MDERT | ResNet-101 | 70.52 | 66.74 | 70.23 | 71.35 | 72.15 | 68.35 | 65.37 | 68.38 | 62.72 |
| VL-BERT | ResNet-101 | 70.03 | 71.42 | 70.54 | 71.47 | 72.36 | 70.29 | 70.14 | 69.84 | 66.70 |
| RSD-LXMERT | ResNet-101 | 72.64 | 72.35 | 71.46 | 73.37 | 74.52 | 70.22 | 71.87 | 63.44 | 65.80 |
| VLTVG | ResNet-101 | 63.33 | 67.14 | 68.26 | 65.37 | 68.49 | 68.51 | 66.22 | 70.24 | 68.80 |
| Grounding-DINO | ViT | 68.15 | 67.92 | 68.48 | 69.50 | 70.10 | 66.17 | 65.85 | 67.24 | 63.15 |
| UNINEXT | ResNet-50 | 70.87 | 70.62 | 71.34 | 71.35 | 73.47 | 69.26 | 68.78 | 71.29 | 65.32 |
| CAVG | ViT | 74.62 | 72.44 | 73.25 | 75.52 | 76.48 | 68.39 | 67.36 | 69.45 | 64.36 |
| Qwen2.5-VL-7B | VLM | 47.31 | 48.20 | 49.10 | 50.06 | 50.84 | 45.12 | 46.37 | 47.05 | 41.92 |
| Qwen2.5-VL-72B | VLM | 56.17 | 57.10 | 57.85 | 58.92 | 59.74 | 53.43 | 54.25 | 55.17 | 49.83 |
| Qwen3-VL-8B | VLM | 56.19 | 57.25 | 58.16 | 59.05 | 59.85 | 53.55 | 54.49 | 55.25 | 50.13 |
| **E3AD (Ours)** | **VLM** | **80.12** | **80.94** | **79.64** | **81.02** | **82.56** | **76.62** | **77.24** | **77.05** | **77.86** |

**说明**: E3AD 全面超过最强基线 CAVG，绝对增益：Talk2Car +6.86%、MoCAD test/val +10.50%/+8.72%、DrivePilot test/val +6.79%/+7.36%；在 corner-case（遮挡/多智能体/歧义）上 +8.26%/+6.95%/+7.48%，Long-text 上更高达 +11.63%。值得注意的是**通用 VLM（含 72B）在接地任务上反而大幅落后**，说明任务对齐的结构与目标比模型规模更关键。

### Table 3: Emotion Prediction (VAD Correlation) / 情绪预测（Spearman ρ / Kendall τ）

| Model | Valence ρ | Valence τ | Arousal ρ | Arousal τ | Dominance ρ | Dominance τ |
|-------|-----------|-----------|-----------|-----------|-------------|-------------|
| BERT + Ridge | 0.78 | 0.59 | 0.75 | 0.56 | 0.74 | 0.55 |
| RoBERTa + Ridge | 0.80 | 0.61 | 0.77 | 0.59 | 0.78 | 0.58 |
| DistilBERT + Ridge | 0.82 | 0.64 | 0.79 | 0.61 | 0.79 | 0.60 |
| Qwen2.5-7B-Instruct | 0.11 | 0.08 | 0.02 | 0.02 | 0.04 | 0.03 |
| Qwen3-Emb.-4B + Ridge | 0.83 | 0.64 | 0.79 | 0.61 | 0.82 | 0.63 |
| **E3AD (Ours)** | **0.95** | **0.84** | **0.94** | **0.82** | **0.94** | **0.81** |

**说明**: VAD 表示相对情绪量级，故用秩相关（Spearman ρ、Kendall τ）评估。E3AD 在三轴上都取得最高相关（valence 0.95/0.84 等），而直接用 Qwen2.5-7B-Instruct 输出近乎随机（0.11/0.08），说明 E3AD 真正建模了连续情绪空间——这是其人本规划与语言反馈的基础。

### Table 4: Spatial Reasoning / 空间推理（vs 通用 VLM）

| Model | T2C IoU50 | Loc. MAE ↓ | Loc. PA₂ | Loc. PA₄ | Depth MAE ↓ | Depth PA₂ | Depth PA₄ |
|-------|-----------|-----------|----------|----------|-------------|-----------|-----------|
| Qwen2.5-VL-7B | 40.23 | 3.49 | 39.5 | 71.1 | 22.92 | 1.3 | 4.2 |
| Qwen2.5-VL-72B | 51.42 | 10.1 | 38.5 | 77.9 | 22.68 | 1.5 | 4.5 |
| Qwen3-VL-8B | 52.68 | 3.71 | 32.8 | 71.5 | 18.89 | 13.7 | 26.5 |
| **E3AD (Ours)** | **79.32** | **0.47** | **97.7** | **98.8** | **4.25** | **53.1** | **71.2** |

**说明**: 通用 VLM（即便 72B）在基础空间推理上表现挣扎（Loc. MAE 10.1、Depth MAE 22.68），E3AD 把 Loc. MAE 降到 0.47、Depth MAE 降到 4.25，Loc. PA₂ 达 97.7%，在 3D 空间感知与定位上建立新 SOTA。佐证双通路空间推理设计的有效性。

### Table 5: Ablation — Visual Grounding / 消融（视觉接地 IoU%）

| Ego. | Allo. | Emo. | DPO | T2C ↑ | Constr. ↑ | Ambg. ↑ | Long ↑ |
|------|-------|------|-----|-------|-----------|---------|--------|
| ✗ | ✓ | ✓ | ✓ | 74.48 | 71.60 | 72.24 | 72.47 |
| ✓ | ✗ | ✓ | ✓ | 76.48 | 73.92 | 74.65 | 74.76 |
| ✓ | ✓ | ✗ | ✓ | 78.78 | 74.41 | 73.57 | 74.12 |
| ✓ | ✓ | ✓ | ✗ | 79.55 | 75.58 | 77.09 | 76.44 |
| **✓** | **✓** | **✓** | **✓** | **80.12** | **76.62** | **77.05** | **77.86** |

**说明**: 去掉 egocentric 通路对接地伤害最大（T2C ↓7.0%、Vision-Constraint ↓6.6%），印证其“把语言指代对齐到观测目标”的第一人称接地作用；去掉 allocentric 削弱全局推理；**Emotion Modeling 在歧义/长文指令上收益最大（+4.5%/+4.8%）**，提升对细微语言线索与情绪表达的敏感度；DPO 提供温和增益。

### Table 6: Ablation — Waypoint Planning / 消融（轨迹规划）

| Ego. | Allo. | Emo. | DPO | ADE ↓ | SSPD ↓ | Fréchet ↓ | FDE ↓ |
|------|-------|------|-----|-------|--------|-----------|-------|
| ✗ | ✓ | ✓ | ✓ | 4.12 | 2.06 | 7.61 | 7.02 |
| ✓ | ✗ | ✓ | ✓ | 4.27 | 2.15 | 7.86 | 7.31 |
| ✓ | ✓ | ✗ | ✓ | 3.93 | 1.91 | 7.36 | 6.80 |
| ✓ | ✓ | ✓ | ✗ | 3.96 | 1.89 | 7.31 | 6.86 |
| **✓** | **✓** | **✓** | **✓** | **3.88** | **1.86** | **7.23** | **6.64** |

**说明**: 对路点规划而言 **allocentric 地图最关键**——移除它使 ADE/FDE 恶化约 10.0%/10.1%，体现全局先验对空间感知与路线一致性的价值；egocentric 通路锚定局部运动线索；Emotion Modeling 与 DPO 进一步提升情绪一致性。

---

## 实验

### 数据集 / 基准

| 基准 | 特点 | 用途 |
|------|------|------|
| [[Talk2Car]] | 自然语言指令 + 目标接地，配 C4AV 协议 | 接地/端到端训练测试 |
| [[Talk2Car-Trajectory]] | Talk2Car 的轨迹扩展，用于端到端规划评测 | 轨迹规划 |
| [[DrivePilot]] | 真实世界驾驶，含 ThinkDeeper 协议下的 Long-Text / Corner-Case 子集 | 接地/规划/鲁棒性 |
| [[MoCAD]] | 真实世界驾驶基准（含情绪/指令） | 接地/情绪 |

- 数据规模：egocentric 通路 ~30K 样本，allocentric 通路 ~17K 样本；按 ThinkDeeper 协议为 DrivePilot/MoCAD 构造 Long-Text 与 Corner-Case 两类专门子集。

### 评测指标
- **端到端轨迹**: ADE、FDE、Fréchet、DTW、SSPD，以及 $g$ 米内规划准确率 PA$_g$（PA₂/PA₄）。
- **子任务**: 视觉接地用 IoU（Talk2Car C4AV 协议）；空间推理用 MAE 与 IoU；情绪用 Spearman ρ 与 Kendall τ。

### 实现细节
- **主干**: [[Qwen2.5-VL]]-7B-Instruct，**冻结主干**、仅训 [[LoRA]] 低秩适配器（rank 16、scale 32），可训参数预算 ≤ 基线，保证“增益来自方法而非模型规模”。
- **优化**: 用 [[MS-Swift]] 库；恒定学习率 $10^{-4}$；per-device batch 16；全量数据训 1 epoch。
- **训练流程**: (i) 模态预训练 + 统一微调使主干适配领域数据；(ii) DPO 行为对齐细化响应。
- **硬件**: 8 × NVIDIA H200 GPU。
- **额外发现**: 更大的通用 VLM（Qwen2.5-VL-72B、Qwen3-VL-8B）仍逊于 E3AD，**任务对齐的结构与目标比原始容量更重要**。

### 关键实验结论
- **端到端（Table 1）**: 七项轨迹指标全面 SOTA，相对 PTPC 在 ADE/FDE 上降 17–20%。
- **视觉接地（Table 2）**: 全基准超 CAVG，corner-case/Long-text 增益最大（+8~12%）。
- **情绪估计（Table 3）**: VAD 三轴秩相关达 0.94–0.95，远超 Ridge 系与直接 prompt 的 LLM。
- **空间推理（Table 4）**: Loc./Depth MAE 大幅领先通用 VLM，建立 3D 定位新 SOTA。
- **消融（Table 5/6）**: 接地最依赖 egocentric、规划最依赖 allocentric；Emotion 在歧义/长文上贡献突出；DPO 提供一致性增益（Fig 4）。
- **案例（Fig 6）**: VAD 作为连续控制信号——加“cautious”使车放弃变道，证明情绪不仅改语言也改运动几何。
- **用户研究（Fig 7）**: 217 人评测，E3AD 在合规、情绪对齐、安全、舒适、总体偏好上多数维度居首。

---

## 批判性思考

### 优点
1. **问题定义新颖且有现实意义**: 把“乘客情绪”作为一等公民引入 E2E 驾驶（OD-E2E 任务），并用连续 VAD 而非离散标签，抓住了 AV 接受度这一真实痛点。
2. **情绪→行为的因果有可验证证据**: 案例研究（Fig 6）展示 VAD 改变直接改变规划（变道 vs 放弃），Fig 4 给出唤醒度与轨迹几何统计的相关，DPO 伪偏好对的构造也巧妙绕开了“AD 只有单一真值轨迹”的难题。
3. **公平性控制到位**: 冻主干只训 LoRA、参数预算 ≤ 基线，且 72B 通用 VLM 仍落后，较有说服力地把增益归于方法而非规模；多基准 + corner-case + 用户研究，证据链较完整。

### 局限性
1. **情绪标签来自语言的间接构造**: VAD 真值由 GoEmotions + 词典映射 + 指令增广合成，**并非真实乘客生理/主观情绪**，"SOTA VAD 相关"实际是与这套合成标签的相关，可能存在标注循环偏置；缺乏真实人类情绪 ground-truth 校验。
2. **开环 / 离线评测为主**: 所有结果基于真实世界数据集的离线轨迹指标与一次性用户片段评分，**没有闭环或高保真仿真**，情绪-动作策略在真实交互、长时程闭环下是否稳定未知（作者也将闭环列为 future work）。
3. **情绪信号单一**: 情绪完全从**语言文本**推断，未利用语音韵律、面部、生理等多模态情绪线索；现实中“stop here now!”的紧迫感更多藏在语音/表情而非纯文本。
4. **DPO 负样本可能制造伪相关**: 用“VAD 偏离最大的增广指令”生成 dispreferred 轨迹来构造偏好对，负轨迹本身由模型自采样，存在自我强化与分布漂移风险，"高唤醒→更直更平滑"这一规律是否普适、是否安全可取值得商榷。

### 潜在改进方向
1. 引入真实乘客情绪标注（语音/生理/事后主观评分）做校验，把 VAD 监督从“语言合成”升级为多模态真值。
2. 接入高保真仿真（CARLA 等）做闭环评测，量化情绪感知策略对安全裕度与乘坐舒适的长期影响。
3. 把情绪从语言扩展到语音韵律与多模态，验证 emotion gap 在更真实交互通道下的弥合效果。
4. 对 DPO 伪偏好的安全边界做约束（如把安全包络与情绪满足显式解耦，正文已提此思想但未量化）。

### 可复现性评估
- [ ] 代码开源（论文未给出公开仓库链接）
- [ ] 预训练模型（未声明 release）
- [x] 训练细节较完整（MS-Swift + LoRA rank/scale、lr、batch、8×H200、三阶段流程；更多在附录 A.1/D）
- [x] 数据集可获取（Talk2Car / Talk2Car-Trajectory / MoCAD / DrivePilot 多为公开基准；情绪增广为合成）

---

## 速查卡片

> [!summary] E3AD: Emotion-Aware VLA for Human-Centric End-to-End AD
> - **核心**: 把乘客情绪显式注入 VLA 端到端驾驶——连续 VAD 情绪建模 + ego/allo 双系统空间推理 + 一致性导向（SFT→Joint→DPO）训练，让接地、规划、反馈都更贴合人类意图。
> - **方法**: 冻结 Qwen2.5-VL-7B 主干 + LoRA；GoEmotions/词典构造 VAD 标签并做情绪感知指令增广；单链自回归预测 $(\hat e,\hat b,\hat\tau)$；用“VAD 偏离最大”的增广指令造伪偏好对做 DPO 对齐情绪与轨迹。
> - **结果**: 4 个真实基准全面 SOTA——端到端 ADE/FDE 较 PTPC 降 17%/20%，接地超 CAVG（corner-case +8~12%），VAD 相关 0.94–0.95，217 人用户研究偏好居首；8×H200。
> - **任务**: 提出 Open-Domain End-to-End（OD-E2E）驾驶。
> - **代码**: 暂未公开。

---

*笔记创建时间: 2026-06-29*
