---
title: "ColaVLA: Leveraging Cognitive Latent Reasoning for Hierarchical Parallel Trajectory Planning in Autonomous Driving"
method_name: "ColaVLA"
authors: [Qihang Peng, Xuesong Chen, Chenye Yang, Shaoshuai Shi, Hongsheng Li]
year: 2026
venue: CVPR
tags: [VLA, autonomous-driving, latent-reasoning, end-to-end-driving, trajectory-planning, parallel-decoding]
zotero_collection: _inbox
image_source: online
arxiv_html: https://arxiv.org/html/2512.22939v3
created: 2026-06-29
---

# ColaVLA: Leveraging Cognitive Latent Reasoning for Hierarchical Parallel Trajectory Planning in Autonomous Driving

## 元信息

| 项目 | 内容 |
|------|------|
| 作者 | Qihang Peng, Xuesong Chen, Chenye Yang, Shaoshuai Shi, Hongsheng Li |
| 机构 | 清华大学、香港中文大学 MMLab、滴滴 Voyager Research |
| 会议 | CVPR 2026 |
| 类别 | VLA（Vision-Language-Action）/ 端到端自动驾驶 |
| 日期 | 2026-02（arXiv v3） |
| 项目主页 | https://github.com/pqh22/ColaVLA |
| 链接 | [arXiv](https://arxiv.org/abs/2512.22939) / [Code](https://github.com/pqh22/ColaVLA) |

---

## 一句话总结

> 把驾驶 VLM 的"思维链"从离散文本搬到统一隐空间，仅用两次 VLM 前向得到紧致的元动作先验，再由一个保因果的层次化并行规划器在单次前向中解码多尺度轨迹，在 nuScenes 开环/闭环上同时刷 SOTA 且推理快 5 倍以上。

---

## 核心贡献

1. **统一 VLA 框架直接产连续轨迹**: 提出 [[ColaVLA]]——面向端到端自动驾驶的统一 [[VLA]] 框架，**直接在连续轨迹空间作业**，既规避了"文本 token 与连续几何/动力学不匹配"的模态错配，又保留 VLM 的世界知识与先验。
2. **认知隐式推理（Cognitive Latent Reasoner）**: 把推理从文本 [[Chain-of-Thought|思维链]] 迁移到统一隐空间，通过 ego-自适应路由与元信息压缩，模仿人类驾驶员"广泛观察→选择性聚焦→审慎复盘→高效决策"四步，**仅两次 VLM 前向**即给出可解释的决策导向元表征。
3. **层次化并行规划器（Hierarchical Parallel Planner）**: 在**单次前向**中并行解码所有时间尺度与多个驾驶模式，配合"保因果混合注意力掩码"，实现高效、合理且安全的轨迹生成。
4. **全面 SOTA**: 在 [[nuScenes]] 开环与 [[NeuroNCAP]] 闭环评测上同时取得新 SOTA，并保持强可解释性与计算效率（推理延迟较文本 CoT 方法低 5 倍以上）。

---

## 问题背景

### 要解决的问题
自动驾驶需要从复杂多模态输入中生成**安全、可靠**的轨迹。模块化流水线把感知/预测/规划拆开，端到端（E2E）则联合学习，而 [[VLM]] 进一步引入跨模态先验与常识推理。但当前**基于 VLM 的规划器**面临三大痛点：(i) 离散文本推理与连续控制不匹配；(ii) 自回归思维链解码延迟高；(iii) 规划器低效或非因果，难以实时部署。

### 现有方法的局限
- **模块化系统**: 组件可解释、几何先验强，但接口脆弱、误差会沿管线传播，全局优化困难。
- **端到端系统**: 减少人工接口、开环精度高，但依赖单条 GT 轨迹的**稀疏监督**，把感知与控制纠缠在一起、因果结构被遮蔽，分布外泛化差。
- **文本式 VLM 规划器**: 先验强，但 (1) **模态错配**——离散文本 token 与轨迹的连续几何/动力学不对齐，易产生格式违规或物理不一致的 waypoint；(2) **CoT 推理延迟**——自回归逐 token 生成随序列增长，推理延迟显著。
- **双系统设计**（VLM + 轻量 E2E 规划器）: 仍依赖显式文本级推理，或在 VLM 与下游动作规划器之间存在**特征鸿沟**。

### 本文的动机
重新审视 VLM 在驾驶任务中的角色，**从显式文本思维链转向统一隐式推理**：完全在统一隐空间执行推理，并配一个"保因果且能并行解码"的规划器。这样既保住 VLM 的知识先验与推理能力，又避免冗长自回归推理及其延迟。核心受 LLM 隐空间推理（[[Coconut]]，Hao et al. 2024）启发——把"思考"放进连续隐空间而非文本。

---

## 方法详解

### 模型架构

[[ColaVLA]] 采用 **统一 VLA（感知-认知-控制）** 架构（见 Figure 2），由两大可学习组件构成（沿用论文对一般规划系统的抽象）：
- **输入**: 多模态表征 $\mathbf{S}_t$（多视角图像、可选 LiDAR/radar、文本 prompt、ego 状态）
- **感知前端**: 图像 backbone（[[EVA-02]]-L）+ [[Q-Former]]（实为 SQ-Former），产出 3D 物体与矢量化地图的视觉 token
- **Reasoner $\mathcal{R}_\theta$**: [[Cognitive Latent Reasoner|认知隐式推理器]]，共享一个 VLM Transformer（[[LLaVA]] v1.5 / [[LLaMA]]-7B + [[LoRA]]）
- **Planner $\mathcal{P}_\phi$**: [[Hierarchical Parallel Planner|层次化并行规划器]]
- **输出**: $K$ 步连续轨迹 $\widehat{\mathbf{Y}}_t=[(x_{t+1},y_{t+1}),\dots,(x_{t+K},y_{t+K})]$

论文用一个统一抽象概括了"模块化 / 无显式推理 E2E / 文本 CoT VLM"三大范式（见公式1）。ColaVLA 的关键在于：**Reasoner 只跑两次 VLM 前向**得到元动作先验，**Planner 只跑一次前向**并行解码多尺度轨迹——总计三次前向（场景理解、隐式复盘、并行动作解码），如 Figure 1(b) 所示。

### 核心模块

#### 模块1: Cognitive Latent Reasoning（认知隐式推理器）

**设计动机**: 把人类驾驶员"观察→识别→复盘→决策"的认知过程搬到**统一隐空间**，保住语言式推理的语义丰富度，同时避免逐 token 生成的开销。整个推理只需 **两次 VLM 前向**。分四个阶段（对应 Figure 2 左侧 Understand / Recognize / Rethink / Decide）：

1. **Driving Scene Comprehension（场景理解，第一次前向）**: 把固定驾驶 prompt 的文本嵌入 $\mathbf{T}$、多视角视觉嵌入 $\mathbf{V}$、ego token $\mathbf{E}$ 拼成序列送入共享 VLM，**只保留视觉切片** $\mathbf{Q}_\textsc{V}$（见公式2）；更新后的文本/ego 嵌入被丢弃，使 prompt 保持不变、不引入冗余。
2. **Critical Entity Recognition（关键实体识别）**: 先用 [[FiLM]] 条件化把视觉 token 对齐到当前 ego 状态（公式3），再由轻量 router 评分并 [[Top-K]] 选出最具信息量的 $K$ 个安全关键 token（公式4）。训练时用 [[Gumbel-Softmax]] 松弛使选择可微（形成 $K$-hot mask），推理时直接取 top-$K$。这一步充当**信息瓶颈**，留下车道、临近车辆、行人、信号灯等关键线索 $\mathbf{Q}^*$。
3. **Latent Rethinking（隐式复盘，第二次前向）**: 把固定 prompt $\mathbf{T}$、$K$ 个显著视觉 token $\mathbf{Q}^*$、ego token $\mathbf{E}$ 与一组 $C$ 个可学习 **meta-query** $\mathbf{M}$ 拼接，再过一次共享 VLM（公式5），得到代表不同驾驶策略的元嵌入 $\mathbf{Q}_\textsc{M}$。共享 VLM 保证时序/语义一致，且 token 预算小（$C\ll L_v$）保持高效。每个 $\mathbf{m}_c$ 初始化为某个元动作（如直行巡航、无保护左转、急刹），由聚类训练轨迹得到。
4. **Strategic Decision Synthesis（策略决策合成）**: $\mathbf{Q}_\textsc{M}$ 经 FiLM 调制适配 ego 状态，交叉注意到关键视觉 token $\mathbf{Q}^*$，再过一层 meta token 间自注意力；共享两层 MLP 把每个 meta token 映射成机动 logit，用 [[Focal Loss]] 训练以强调困难/安全关键样本。把推理空间约束到 $C$ 个元动作 token 实现**熵减**，产出多个可能驾驶策略作为后续结构化先验。

#### 模块2: Hierarchical Parallel Planner（层次化并行规划器）

**设计动机**: 通过"意图→运动"的多阶段解码生成轨迹，三大性质：(i) **意图到运动的逐步细化**（先定终点意图，再细化运动）；(ii) **阶段独立性**（不同时间尺度可并行解码）；(iii) **置信度感知的多样性**（保留多模式防 mode collapse）。

**具体实现**:
- **Stage-Aware Trajectory Querying（阶段感知轨迹查询）**: 把 $T$ 步预测域 $\mathcal{T}=\{0,\dots,T{-}1\}$ 划成 $S$ 个**嵌套**尺度 $\mathcal{I}_1\subset\dots\subset\mathcal{I}_S=\mathcal{T}$（粗→细）。对每个尺度 $s$，用认知推理器选出的元动作 query $\mathbf{A}$ 经时间嵌入扩成轨迹目标 $\mathbf{F}$，按预定义顺序重采样成多尺度子集 $\mathbf{F}_s$。把剪枝后的上下文与各尺度目标按时序拼成完整输入流 $\mathbf{X}$（公式6）。
- **Causality-Preserving Hybrid Attention（保因果混合注意力）**: 设计混合掩码 $\mathcal{M}$（Figure 3、公式7）调控信息流，满足：(i) 同类内双向交互、(ii) 全局上下文聚合（每个轨迹 token 都能看全部上下文）、(iii) **因果保持**——尺度 $s$ 的 token 只能访问更粗的前一尺度 $s{-}1$，禁止泄漏未来更细尺度，保证粗→细物理一致解码。
- **Confidence-Guided Parallel Decoding（置信度引导并行解码）**: 同时处理多个候选驾驶策略，两个轻量 MLP 头分别估置信度与回归多尺度轨迹；训练用 one-hot 监督——只有最接近 GT 的假设获得直接回归监督。所有候选在**单次前向**内并行解码，高效且通过保留多样性防 mode collapse。

### 关键公式与机制

#### 公式1: [[VLA]] 规划系统的统一抽象

$$
\mathbf{Z}_{t}=\mathcal{R}_{\theta}(\mathbf{S}_{t}),\qquad \widehat{\mathbf{Y}}_{t}=\mathcal{P}_{\phi}(\mathbf{Z}_{t},\mathbf{A})
$$

**含义**: 把轨迹规划抽象为"Reasoner 提取/融合多模态特征 → Planner 注入动作查询并回归连续 waypoint"两步，统一涵盖模块化、无推理 E2E、文本 CoT VLM 三大范式。

**符号说明**:
- $\mathbf{S}_t$: $t$ 时刻多模态表征（图像/点云/雷达/prompt/ego 状态或子集）
- $\mathbf{Z}_t\in\mathbb{R}^{L\times D}$: 隐 token 序列；$\mathbf{A}$: 可学习动作库（action bank）
- $\mathcal{R}_\theta$: 推理器（ResNet/PointNet/Q-Former/VLM）；$\mathcal{P}_\phi$: 规划器（MLP/扩散/Transformer 解码器）
- $\widehat{\mathbf{Y}}_t$: 预测的 $K$ 步轨迹

#### 公式2: 场景理解（第一次 VLM 前向，仅留视觉切片）

$$
\mathbf{Q}_{\textsc{V}}=\mathcal{D}_{\textsc{vlm}}\bigl([\ \mathbf{T};\mathbf{V};\mathbf{E}\ ]\bigr)\in\mathbb{R}^{L_{v}\times D}
$$

**含义**: 把固定 prompt、多视角视觉、ego token 拼起来过共享 VLM，只保留视觉部分的隐状态作为"全局连贯、时序因果"的空间语义表征。

**符号说明**:
- $\mathbf{T}\in\mathbb{R}^{L_t\times D}$: 文本 prompt 嵌入；$\mathbf{V}\in\mathbb{R}^{L_v\times D}$: 多视角视觉嵌入；$\mathbf{E}\in\mathbb{R}^{1\times D}$: ego 状态 token
- $\mathcal{D}_\textsc{vlm}$: 共享 VLM Transformer；$\mathbf{Q}_\textsc{V}$: 保留的视觉隐状态（文本/ego 更新被丢弃以保持 prompt 不变）

#### 公式3: Ego-自适应 FiLM 条件化

$$
\tilde{\mathbf{Q}}_{\textsc{V}}=\bigl(1+\gamma_{\textsc{Re}}(\mathbf{E})\bigr)\odot\mathbf{Q}_{\textsc{V}}+\beta_{\textsc{Re}}(\mathbf{E})\in\mathbb{R}^{L_{v}\times D}
$$

**含义**: 用从 ego token 生成的缩放/平移参数对视觉 token 做 [[FiLM]] 调制，让特征突出与本车速度、航向、曲率一致的场景元素（碰撞锥内的动态体与车道边界），抑制无关背景。

**符号说明**:
- $\gamma_{\textsc{Re}}(\mathbf{E}),\beta_{\textsc{Re}}(\mathbf{E})$: 由 ego token 经两个独立线性投影生成的缩放与平移
- $\odot$: 逐元素乘；$\tilde{\mathbf{Q}}_\textsc{V}$: ego 调制后的视觉 token

#### 公式4: Top-K 关键 token 选择

$$
\mathbf{w}=\mathcal{H}_{\phi}\bigl(\tilde{\mathbf{Q}}_{\textsc{V}}\bigr)\in\mathbb{R}^{L_{v}},\qquad \mathbf{Q}^{*}=\operatorname{TopK}\bigl(\tilde{\mathbf{Q}}_{\textsc{V}},\,\mathbf{w},\,K\bigr)
$$

**含义**: 轻量 router $\mathcal{H}_\phi$ 给每个调制后 token 打分，选出 top-$K$ 形成紧致安全关键集合 $\mathbf{Q}^*$，充当后续隐式推理的**信息瓶颈**。

**符号说明**:
- $\mathbf{w}$: router 输出的各 token 重要性分数；$K$: 保留 token 数（默认 256）
- $\operatorname{TopK}$: 训练时用 [[Gumbel-Softmax]] 松弛成可微 $K$-hot mask，推理时硬选 top-$K$
- $\mathbf{Q}^*\in\mathbb{R}^{K\times D}$: 剪枝后的关键视觉 token

#### 公式5: 隐式复盘（第二次 VLM 前向）

$$
\mathbf{Q}_{\textsc{M}}=\mathcal{D}_{\textsc{vlm}}\bigl([\ \mathbf{T};\mathbf{Q}^{*};\mathbf{E};\mathbf{M}\ ]\bigr)\in\mathbb{R}^{C\times D}
$$

**含义**: 把 prompt、关键视觉 token、ego token、$C$ 个可学习 meta-query 拼接，再过一次共享 VLM，让每个元动作嵌入通过上下文交互得到代表不同驾驶策略的表征 $\mathbf{Q}_\textsc{M}$。

**符号说明**:
- $\mathbf{M}=[\mathbf{m}_1,\dots,\mathbf{m}_C]\in\mathbb{R}^{C\times D}$: 可学习元动作查询（由聚类训练轨迹初始化）
- $C\ll L_v$: 元 token 预算远小于原视觉 token 数，保证高效；$\mathbf{Q}_\textsc{M}$: 候选驾驶策略表征

#### 公式6: 多尺度输入流构造

$$
\mathbf{X}=[\mathbf{Q}^{*};\mathbf{F}_{1};\dots;\mathbf{F}_{S}]\in\mathbb{R}^{L\times D},\qquad L=K+\sum_{s=1}^{S}|\mathcal{I}_{s}|
$$

**含义**: 把剪枝上下文 $\mathbf{Q}^*$ 与各尺度轨迹目标 $\mathbf{F}_s$ 按时序拼成规划器的完整输入流，保证"粗轨迹先于细化"，为保因果掩码奠基。

**符号说明**:
- $\mathbf{F}_s\in\mathbb{R}^{|\mathcal{I}_s|\times D}$: 第 $s$ 尺度的轨迹目标（由元动作 query 经时间嵌入扩展并重采样）
- $\mathcal{I}_s$: 第 $s$ 个嵌套时间尺度的索引集合；$S$: 尺度数（默认 6）；$K$: 上下文 token 数

#### 公式7: 保因果混合注意力掩码

$$
\mathcal{M}(i,j)=\begin{cases}
0,& j\leq L_{c},\\[4pt]
0,& i\geq L_{c}\ \text{且}\ \mathbf{X}[j]\in\mathcal{I}_{s-1}\cup\mathcal{I}_{s},\\[4pt]
-\infty,& \text{其他}
\end{cases}
$$

**含义**: 让每个尺度 $s$ 的 token 既能注意到全部剪枝上下文，又只能访问紧邻的更粗尺度 $\mathcal{I}_{s-1}$，禁止访问未来更细尺度 $\mathcal{I}_{s+1},\dots,\mathcal{I}_S$，实现"全局聚合 + 严格因果细化"。

**符号说明**:
- $\mathcal{M}(i,j)$: query $i$ 对 key $j$ 的注意力掩码加性偏置（0 允许、$-\infty$ 屏蔽）
- $L_c$: 剪枝上下文长度；$s$: 当前尺度索引；$\mathbf{X}$: 拼接序列

---

## 关键图表

<!-- 图片均使用 arXiv HTML 在线链接，未本地化 -->

### Figure 1: Inference Paradigms / 推理范式对比

![Figure 1](https://arxiv.org/html/2512.22939v3/x1.png)

**说明**: (a) 既有驾驶 VLM 用**文本思维链**，逐子任务自回归吐出中间文本，反复解码导致 token 成本与误差累积、延迟高；(b) 本文在 [[VLA]] 隐空间做隐式推理，仅三次前向（场景理解、隐式复盘、并行动作解码），去掉自回归文本、削减延迟，同时保留决策级可解释性。这张图直观点明了 ColaVLA 的核心卖点——"把 CoT 搬进隐空间"。

### Figure 2: ColaVLA Framework Overview / 整体框架

![Figure 2](https://arxiv.org/html/2512.22939v3/x2.png)

**说明**: 多视角图像序列先经图像 backbone 与 [[Q-Former]] 感知 3D 物体与矢量化地图，产出视觉 token。**左侧** [[Cognitive Latent Reasoner|认知隐式推理]] 通过 Understand→Recognize→Rethink→Decide 四阶段隐式推理得出驾驶策略；**右侧**据策略从 action bank 选对应元动作 query，转成多尺度目标，与剪枝上下文一起送入 [[Hierarchical Parallel Planner|层次化并行规划器]] 单次并行解码出多尺度轨迹。是理解全文数据流的主图。

### Figure 3: Causality-Preserving Hybrid Mask / 保因果混合掩码

![Figure 3](https://arxiv.org/html/2512.22939v3/x3.png)

**说明**: 规划器多尺度目标的注意力掩码设计。它允许信息从剪枝上下文流向所有时间尺度，同时在相邻尺度间维持时序因果（尺度 $s$ 只看 $s{-}1$，不看更细的未来尺度）。这是公式7的可视化，解释了"并行解码却不破坏因果"的关键机制。

### Figure 4: Multi-scale Trajectory Predictions / 多尺度轨迹定性可视化

![Figure 4](https://arxiv.org/html/2512.22939v3/x4.png)

**说明**: 红、黄、紫曲线分别表示从"仅终点"到"完整轨迹"的多尺度预测，绿色为 GT。右侧为含本车、他车与轨迹的 BEV 可视化。直行与转弯场景下，粗轨迹（红）抓全局意图，细尺度（黄/紫）逐步细化空间细节与曲率，收敛到 GT，佐证"层次化解码在单次前向内产出平滑准确的计划"。

### Figure 5: Closed-loop Qualitative Comparison (NeuroNCAP) / 闭环定性对比（附录）

![Figure 5](https://arxiv.org/html/2512.22939v3/img/appendix_visualization.jpg)

**说明**: [[NeuroNCAP]] 模拟器三类代表场景（静态障碍、正面交互、侧向冲突）下 ColaVLA 与对比规划器的预测轨迹。ColaVLA 持续把本车引离潜在碰撞，产生更安全、更稳定的运动，定性印证其更高 NeuroNCAP 分与更低延迟。

### Table 1: Open-loop Planning on nuScenes / nuScenes 开环规划

文本式驾驶模型（上）与动作式驾驶模型（下）。Ego 列表示是否使用 ego 状态。

| Method | Reference | Ego | L2 1s | L2 2s | L2 3s | L2 Avg | Col 1s | Col 2s | Col 3s | Col Avg |
|--------|-----------|-----|------|------|------|-------|-------|-------|-------|--------|
| *Text-Based* | | | | | | | | | | |
| DriveVLM | CoRL 2024 | ✓ | 0.18 | 0.34 | 0.68 | 0.40 | – | – | – | – |
| DriveVLM-Dual | CoRL 2024 | ✓ | 0.15 | 0.29 | 0.48 | 0.31 | – | – | – | – |
| OmniDrive | CVPR 2025 | ✓ | 0.14 | 0.29 | 0.55 | 0.33 | 0.00 | 0.13 | 0.78 | 0.30 |
| EMMA | TMLR | ✓ | 0.14 | 0.29 | 0.54 | 0.32 | – | – | – | – |
| EMMA+ | TMLR | ✓ | 0.13 | 0.27 | 0.48 | 0.29 | – | – | – | – |
| ImpromptuVLA | NeurIPS 2025 | ✓ | 0.13 | 0.27 | 0.53 | 0.30 | – | – | – | – |
| SOLVE-VLM | CVPR 2025 | ✓ | 0.13 | 0.25 | 0.47 | 0.28 | 0.00 | 0.16 | 0.43 | 0.20 |
| *Action-Based* | | | | | | | | | | |
| UniAD | CVPR 2023 | – | 0.59 | 1.01 | 1.48 | 1.03 | 0.16 | 0.51 | 1.64 | 0.77 |
| VAD-Base | ICCV 2023 | – | 0.69 | 1.22 | 1.83 | 1.25 | 0.06 | 0.68 | 2.52 | 1.09 |
| BEV-Planner | CVPR 2024 | – | 0.30 | 0.52 | 0.83 | 0.55 | 0.10 | 0.37 | 1.30 | 0.59 |
| UniAD | CVPR 2023 | ✓ | 0.20 | 0.42 | 0.75 | 0.46 | 0.02 | 0.25 | 0.84 | 0.37 |
| VAD-Base | ICCV 2023 | ✓ | 0.17 | 0.34 | 0.60 | 0.37 | 0.04 | 0.27 | 0.67 | 0.33 |
| AD-MLP | arXiv 2023 | ✓ | 0.15 | 0.32 | 0.59 | 0.35 | 0.00 | 0.27 | 0.85 | 0.37 |
| BEV-Planner++ | CVPR 2024 | ✓ | 0.16 | 0.32 | 0.57 | 0.35 | 0.00 | 0.29 | 0.73 | 0.34 |
| SOLVE-E2E | CVPR 2025 | ✓ | 0.14 | 0.28 | 0.50 | 0.31 | 0.04 | 0.17 | 0.68 | 0.30 |
| **ColaVLA** | – | ✓ | **0.14** | **0.27** | **0.50** | **0.30** | 0.04 | 0.17 | **0.47** | **0.23** |

**说明**: 在动作式方法中 ColaVLA 取得最低平均 L2（0.30m）与最低平均碰撞率（0.23%）；相较最强动作式基线 SOLVE-E2E（L2 0.31m / Col 0.30%），L2 降 3%、碰撞率降 23%。同时它在不做自回归文本解码的前提下逼近最新文本式 VLM，且 VLM 前向次数比典型文本管线少 5 倍以上。

### Table 2: Closed-loop Simulation on NeuroNCAP / NeuroNCAP 闭环仿真

$\dagger$ 表示用了额外训练数据且为文本式驾驶 VLM；$\ddagger$ 表示用了轨迹后处理。本评测只采用 top-1 驾驶策略以贴近真实决策。

| Method | Reference | Score Avg | Score Stat. | Score Frontal | Score Side | Col Avg | Col Stat. | Col Frontal | Col Side |
|--------|-----------|-----------|-------------|---------------|-----------|---------|-----------|-------------|----------|
| UniAD | CVPR 2023 | 0.73 | 0.84 | 0.10 | 1.26 | 88.6 | 87.8 | 98.4 | 79.6 |
| VAD | ICCV 2023 | 0.66 | 0.47 | 0.04 | 1.45 | 92.5 | 96.2 | 99.6 | 81.6 |
| SparseDrive | ICRA 2025 | 0.92 | – | – | – | 93.9 | – | – | – |
| BridgeAD-S | CVPR 2025 | 1.52 | – | – | – | 76.2 | – | – | – |
| BridgeAD-B | CVPR 2025 | 1.60 | – | – | – | 72.6 | – | – | – |
| ImpromptuVLA $\dagger$ | NeurIPS 2025 | 2.06 | 2.55 | 1.86 | 1.78 | 65.1 | 54.8 | 72.8 | 67.6 |
| BridgeAD-S $\ddagger$ | CVPR 2025 | 2.98 | – | – | – | 46.1 | – | – | – |
| BridgeAD-B $\ddagger$ | CVPR 2025 | 3.06 | – | – | – | 44.3 | – | – | – |
| **ColaVLA** | – | **3.48** | **3.54** | **3.16** | **3.75** | **36.8** | **32.3** | **51.6** | **26.4** |

**说明**: ColaVLA 取得平均 NeuroNCAP 分 3.48（正文按 Frontal 口径对比时记为 3.16，超最强 ImpromptuVLA 的 2.06，绝对 +1.10、相对 +53%），并把平均碰撞率从 65.1% 降到 36.8%，静态碰撞降幅尤大、侧向碰撞最优。关键是 ImpromptuVLA 用了额外精调数据且为文本式，而 ColaVLA **无文本 CoT、无额外安全关键数据**就更强，验证隐式推理 + 单次并行解码在闭环安全性上的优势。

### Table 3: Inference Latency / 推理延迟（单卡 H20，无 FlashAttention）

| Method | Action-based | Latency (ms) ↓ |
|--------|--------------|----------------|
| OmniDrive | ✗ | 3727 |
| SOLVE-VLM | ✗ | 3719 |
| **Ours (ColaVLA)** | ✓ | **727** |

**说明**: 端到端单帧延迟。相比依赖文本级自回归 CoT 的 OmniDrive / SOLVE-VLM，ColaVLA 的隐式推理 + 单次层次化解码快 **5 倍以上**（727ms vs ~3720ms），是"可实时部署"主张的直接证据。

### Table 4: Ablation on Latent Reasoning / 隐式推理消融

| Reasoning | Rethink | L2 1s | L2 2s | L2 3s | L2 Avg |
|-----------|---------|-------|-------|-------|--------|
| ✗ | ✗ | 14.1 | 28.5 | 54.2 | 32.2 |
| ✓ | ✗ | 14.5 | 28.5 | 50.7 | 31.3 |
| ✓ | ✓ | **14.0** | **27.1** | **50.2** | **30.4** |

**说明**: L2 单位 cm。引入隐式推理把平均 L2 从 32.2→31.3；再加 Rethink（复盘）阶段进一步降到 30.4，说明"重新评估压缩后的关键信息"能细化视觉理解、改善后续决策，尤其在长时距（3s）与复杂动态场景。

### Table 5: Ablation on Action Planner (closed-loop) / 规划器消融（闭环）

为隔离规划器贡献，本对比关闭推理模块。

| Planner | Score Static | Score Frontal | Score Side | Score Avg |
|---------|--------------|---------------|-----------|-----------|
| MLP-based | 1.18 | 0.65 | 1.31 | 1.05 |
| Diffusion-based | 1.05 | 0.58 | 1.43 | 1.02 |
| **Ours** | **2.58** | **1.10** | 0.82 | **1.50** |

**说明**: 动作式规划器常有相近开环指标，故这里只看闭环 NeuroNCAP 分。本文层次化并行规划器显著超确定性 MLP 与随机扩散头（平均 1.50 vs 1.05/1.02），在静态与正面场景提升尤大——粗到细且保因果的解码能更稳更安全地细化中间预测。

### Table 6: Ablation on Retained Token Number $K$ / 保留关键 token 数

原始物体 query 900、车道 query 300。

| $K$ | L2 1s | L2 2s | L2 3s | L2 Avg |
|-----|-------|-------|-------|--------|
| 128 | 14.4 | 28.3 | 50.8 | 31.2 |
| 192 | 14.2 | 28.1 | 50.5 | 30.9 |
| **256** | **14.0** | **27.1** | **50.2** | **30.4** |
| 320 | 14.6 | 28.8 | 51.8 | 31.7 |

**说明**: 太少 token 信息损失大、太多引入冗余增开销。$K{=}256$ 取得最佳折中（语义覆盖足且高效），定为默认配置。

### Table 7: Ablation on Hierarchical Regression Strategy / 层次回归策略消融

各变体共用并行解码框架，仅尺度子集选择顺序不同。

| Strategy Type | L2 1s | L2 2s | L2 3s | L2 Avg |
|---------------|-------|-------|-------|--------|
| Single scale | **13.8** | 28.4 | 53.4 | 31.9 |
| Multi-scale (Sequential) | 14.5 | 28.7 | 52.2 | 31.8 |
| Multi-scale (Reverse) | 14.7 | 28.5 | 51.1 | 31.4 |
| **Multi-scale (Interpolate)** | 14.0 | **27.1** | **50.2** | **30.4** |

**说明**: Single scale 直接回归终轨迹（非因果参照）。所有多尺度都优于单尺度，证明时间抽象有益；本文 **Interpolate**（先预测关键端点再跨尺度填中间点）最优，契合驾驶运动的因果结构。

### Table 8: Ablation on Planner Context Token Selection / 规划器上下文 token 选择（附录）

| Visual tokens | L2 1s | L2 2s | L2 3s | L2 Avg |
|---------------|-------|-------|-------|--------|
| Full tokens | 14.8 | 28.9 | 52.4 | 32.0 |
| **Pruned tokens (Ours)** | **14.0** | **27.1** | **50.2** | **30.4** |

**说明**: 给规划器喂"剪枝后的关键 token"比喂全部视觉 token 的 L2 更低（30.4 vs 32.0），说明 ego-自适应 router 能准确挑出当前最相关线索、过滤冗余背景，同时缩短序列、降算力——精度与效率双赢。

### Table 9: Implementation & Training Hyperparameters / 实现与训练超参（附录）

| 类别 | 设置 |
|------|------|
| Backbone VLM | LLaVA v1.5（LLaMA-7B）+ LoRA adapters |
| 图像编码器 | EVA-02-L（EVAViT，24 层，1024-d，window 16） |
| 视觉推理 | SQ-Former + 时序 PETR 风格 transformer |
| 物体 query | 900（3D 检测头） |
| 车道 query | 300（地图/车道头） |
| 输入模态 | 6 路相机，camera-only，无 LiDAR/radar |
| 图像分辨率 | 900×1600 → 增广后 320×640 |
| 点云范围 | [-51.2,-51.2,-5.0, 51.2,51.2,3.0]（仅定义 BEV 网格） |
| 体素大小 | [0.2,0.2,8.0] |
| 规划尺度 | $S=6$ 层次尺度 |
| 多尺度损失权重 | [0.5,0.7,1.0,1.2,1.5,1.8] |
| Top-$K$ 假设 | $K=3$ 元动作模式 |
| Batch | 2/GPU ×16 GPU（有效 32） |
| Epochs（Stage3） | nuScenes 上 10 epoch |
| 优化器 | AdamW，lr 1e-4，wd 1e-4，(0.9,0.999) |
| LR 调度 | Cosine Annealing + 500 步线性 warmup，min lr ratio 1e-3 |
| 精度/裁剪 | FP16 动态 loss scaling；grad clip 全局范数 10 |
| 效率技巧 | 梯度检查点、FlashAttention、混合精度 |

**说明**: 完整复现配置。注意训练用 **16 张 H20**、三阶段（VLM 适配 → 元动作库预训练 → 端到端微调），LLaMA-7B 主干始终冻结、只更新 LoRA 等组件。

---

## 实验

### 数据集 / 基准

| 基准 | 规模 | 特点 | 用途 |
|------|------|------|------|
| [[nuScenes]] | 1000 个 ~20s 场景，6 相机 + LiDAR + 语义图 + 3D 框 | 开环精度/安全（L2、碰撞率） | 训练/测试 |
| OmniDrive-nuScenes | nuScenes 的 QA 式扩展（感知/预测/规划问答） | 提供推理监督与在线空间推理 | 训练（VLM 适配） |
| [[NeuroNCAP]] | 基于 nuScenes 的照片级闭环模拟器 | 重建安全关键城市交互；五星 NeuroNCAP 分 + 碰撞率 | 闭环测试 |

### 实现细节

- **框架/主干**: [[LLaVA]] v1.5（[[LLaMA]]-7B 语言模型 + [[LoRA]]）；图像编码 [[EVA-02]]-L（EVAViT 24 层）；视觉推理 SQ-Former；检测/车道解码沿用 [[StreamPETR]]（900 物体 query + 300 车道 query）；camera-only 6 视图。
- **三阶段训练**: ① 在 OmniDrive QA 上适配 VLM（仅训 LoRA）；② 元动作库预训练（k-means 聚类 nuScenes 轨迹得元动作原型初始化 action bank，冻 VLM 训分类/轨迹头）；③ nuScenes 端到端微调（冻 LLaMA，联合优化 LoRA、视觉编码器、检测/地图头、router、meta-query、规划器；多尺度轨迹损失 + top-$K$ 假设）。
- **元动作库（附录9）**: 由"路径模式 × 速度模式"组合，结合未来 ego 轨迹与 CAN bus 信号，最终保留 **8 类**——4 种直行（停车/匀速/加速/减速）+ 4 种转弯或变道（大左、大右、小左、小右），分类流程大体沿用 [[Senna]]。
- **优化**: AdamW，lr 1e-4，wd 1e-4，Cosine Annealing + 500 步 warmup；FP16；grad clip 10；16×H20，有效 batch 32，Stage3 训 10 epoch。

### 关键实验结论

- **开环（Table 1）**: 动作式方法中最优，平均 L2 0.30m、碰撞率 0.23%，较 SOLVE-E2E 碰撞率降 23%。
- **闭环（Table 2）**: NeuroNCAP 分大幅领先（超 ImpromptuVLA 绝对 +1.10），碰撞率显著更低，且无需文本 CoT 与额外数据。
- **效率（Table 3）**: 727ms/帧，比文本 CoT 方法快 5 倍以上。
- **消融**: 推理 + Rethink 都有增益（T4）；本文规划器闭环远超 MLP/扩散头（T5）；$K{=}256$ 最优（T6）；Interpolate 多尺度策略最优（T7）；喂剪枝 token 优于喂全部 token（T8）。

---

## 批判性思考

### 优点
1. **"把 CoT 搬进隐空间"的范式干净且有效**: 用两次 VLM 前向 + 一次并行解码替代逐 token 自回归，既保 VLM 先验又把延迟压到 727ms，开环/闭环双 SOTA，且效率证据（Table 3）直接硬核。
2. **闭环安全性提升显著**: NeuroNCAP 上对静态/侧向碰撞的大幅改善（Table 2/5），比单看开环 L2 更能反映真实驾驶价值；并行解码 + 保因果掩码的设计与闭环安全性指标对齐得好。
3. **模块贡献可分离验证**: 推理、Rethink、规划器、$K$、回归策略、上下文 token 都有独立消融，论证较扎实；元动作库的构造（附录9）也交代得细，复现路径清晰。

### 局限性
1. **开环 L2 提升边际**: 相对 SOLVE-E2E 仅 L2 降 3%（0.31→0.30），且 1s/2s 段与多基线持平，"开环 SOTA"主要靠碰撞率；众所周知 nuScenes 开环 L2 受 ego 状态主导（BEV-Planner 论文已指出），其区分度本就有限。
2. **闭环分数口径略含糊**: 正文写 NeuroNCAP 3.16、平均碰撞 42.5%，而 Table 2 表内是 3.48 与 36.8%（含 Stat./Frontal/Side 细分），两处口径需读者自行对齐，易混淆。
3. **依赖额外 QA 监督与重型主干**: 强依赖 OmniDrive-nuScenes 的 QA 标注做 Stage1 适配，且用 LLaMA-7B + 16×H20 训练，"高效"主要体现在**推理**端而非训练端；camera-only、未用 LiDAR/radar 也限制了在恶劣感知条件下的结论外推。
4. **元动作库为规则手工设计**: 8 类元动作由启发式阈值（yaw、横向位移、速度统计）切分，且"speed 仅对 heavy path 细分"带经验性，可能在长尾机动（如复杂博弈、倒车）上覆盖不足。

### 潜在改进方向
1. 用更具区分度的闭环基准（如 nuPlan / Bench2Drive）与更多强基线补充，量化"隐式推理→安全"的因果，而非主要靠 NeuroNCAP 单一闭环。
2. 把元动作库从规则切分改为可学习/数据驱动的码本，减少手工阈值依赖，并验证对长尾机动的覆盖。
3. 探索更轻量主干替代 LLaMA-7B 以降低训练成本，并补 LiDAR/radar 多模态以检验在退化感知下的鲁棒性。

### 可复现性评估
- [x] 代码开源（https://github.com/pqh22/ColaVLA）
- [ ] 预训练模型（文中未明确声明发布权重）
- [x] 训练细节完整（附录 Table 9 + 三阶段流程 + 元动作库构造）
- [x] 数据集可获取（nuScenes / OmniDrive-nuScenes / NeuroNCAP 均公开）

---

## 速查卡片

> [!summary] ColaVLA: Cognitive Latent Reasoning + Hierarchical Parallel Planning
> - **核心**: 把驾驶 VLM 的思维链从文本搬到统一隐空间，两次 VLM 前向得元动作先验，一次并行前向解码多尺度保因果轨迹。
> - **方法**: Cognitive Latent Reasoner（Understand→Recognize→Rethink→Decide，FiLM + Top-K=256 路由 + meta-query 复盘）→ Hierarchical Parallel Planner（嵌套 $S{=}6$ 尺度 + 保因果混合掩码 + 置信度引导并行多模解码）；主干 LLaVA-1.5/LLaMA-7B + LoRA，三阶段训练。
> - **结果**: nuScenes 开环 L2 0.30m / 碰撞 0.23%；NeuroNCAP 闭环分 3.48（超 ImpromptuVLA +1.10）；延迟 727ms（比文本 CoT 快 5×+），开环闭环双 SOTA。
> - **代码**: https://github.com/pqh22/ColaVLA

---

*笔记创建时间: 2026-06-29*
