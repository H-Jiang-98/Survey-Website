---
title: "SaPaVe: Towards Active Perception and Manipulation in Vision-Language-Action Models for Robotics"
method_name: "SaPaVe"
authors: [Mengzhen Liu, Enshen Zhou, Cheng Chi, Yi Han, Shanyu Rong, Liming Chen, Pengwei Wang, Zhongyuan Wang, Shanghang Zhang]
year: 2026
venue: CVPR
tags: [VLA, active-perception, active-manipulation, decoupled-action, camera-control, 3D-geometry, two-stage-training]
zotero_collection: _inbox
image_source: online
arxiv_html: https://arxiv.org/html/2603.12193v1
created: 2026-06-29
---

# SaPaVe: Towards Active Perception and Manipulation in Vision-Language-Action Models for Robotics

## 元信息

| 项目 | 内容 |
|------|------|
| 作者 | Mengzhen Liu, Enshen Zhou, Cheng Chi, Yi Han, Shanyu Rong, Liming Chen, Pengwei Wang, Zhongyuan Wang, Shanghang Zhang |
| 机构 | 北京大学、北京智源人工智能研究院（BAAI）等（多机构合作） |
| 会议 | CVPR 2026 |
| 类别 | VLA（Vision-Language-Action）/ 主动感知与操作 |
| 日期 | 2026-03（arXiv v1） |
| 项目主页 | https://lmzpai.github.io/SaPaVe |
| 链接 | [arXiv](https://arxiv.org/abs/2603.12193) / [Project](https://lmzpai.github.io/SaPaVe) |

---

## 一句话总结

> 把相机运动与机械臂动作**解耦**到两个独立动作头，先用 20 万规模语义相机控制数据学会"该往哪看"，再混合数据联合微调并注入通用 3D 几何知识，从而在动态视角下实现数据高效、可泛化的主动操作。

---

## 核心贡献

1. **解耦动作空间 + 自底向上两阶段训练**: 提出 [[SaPaVe]]，首个端到端的[[Active Manipulation|主动操作]]框架，把**相机运动**与**机械臂操作**解耦到两个动作头，先学[[Semantic Active Perception|语义主动感知]]先验、再学主动视角执行，数据高效。
2. **ActiveViewPose-200K 数据集 + ActiveManip-Bench 基准**: 构建 20 万规模的"图像-语言-相机运动"配对数据集用于语义相机控制学习；并提出**首个**评测主动操作（超越固定视角设定）的仿真基准 ActiveManip-Bench（12 任务 / 100 物体 / 20 场景）。
3. **超越现有 VLA 的强性能**: 在仿真与真实环境中全面超越 [[GR00T N1]] 与 [[π0|$\pi_0$]]，真实任务成功率最高领先 31.25%（相对 $\pi_0$ 领先 40%），验证"紧耦合感知-执行 + 解耦协同训练"的有效性。

---

## 问题背景

### 要解决的问题
机器人在杂乱、遮挡、目标超出视野的真实场景中操作时，需要**主动改变视角**去揭示任务关键线索（[[Active Perception|主动感知]]），并把新获得的观测立刻落地为动作（[[Active-View Execution|主动视角执行]]）。如何在一个 [[VLA]] 框架里**统一**"语义驱动的主动感知"与"鲁棒的、视角不变的执行"，且**数据高效**，是本文要解决的核心问题。

### 现有方法的局限
1. **现有主动操作方法不支持语义输入**：只能机械地移动相机，无法根据语言指令"理解该看哪里"。
2. **直接扩展现有 VLA 动作空间会破坏先验**：把相机运动直接塞进已有 VLA 的统一动作空间，会**破坏大规模固定视角操作数据学到的先验**，并放大数据需求——而真实数据采集昂贵、耗时。
3. **缺乏视角不变的 3D 几何一致性**：现有 VLA 在动态变化的第一人称视角下难以生成时序稳定的动作块。
4. **缺乏评测基准**：现有仿真基准（如 LIBERO 类）全部限定在固定视角，无法评测主动操作能力。

### 本文的动机
- 作者发现一个关键观察（Figure 1）：抓取被遮挡的白碗（$\mathcal{O}_1$）需要旋转第一人称视角，而抓油烟机把手（$\mathcal{O}_5$）只需小幅上移即可——说明**不同任务对视角调整的语义需求差异很大**，必须由语言+图像语义驱动。
- 相机运动本质上是**与本体无关（embodiment-agnostic）**的，因此应当与本体相关的操作动作**解耦**：解耦后既能用海量无本体相机数据高效学习语义感知先验，又不破坏已有操作知识。
- 核心假设："自底向上"——**先学会感知（往哪看），再学会基于该感知去执行**，比把两者塞进同一动作空间更稳、更省数据。

---

## 方法详解

### 模型架构

[[SaPaVe]] 采用 **VLM 主干 + 解耦双动作头 + 通用 3D 几何注入** 的架构（见 Figure 2）：

- **输入**: RGB 图像 $I_t$ + 语言指令 $L$ + 可选的 3D 几何信息 $G_t$（深度图、相机内外参等）
- **Backbone**: [[Eagle-2]] VLM（[[SigLIP-2]] 图像编码器 + [[SmolLM2]] LLM），在其 LLM 注意力层上挂 [[LoRA]] 形式的 **Camera Adapter**
- **核心模块**:
  - [[Decoupled Action Heads|解耦动作头]]（Camera Action Decoder + Manipulation Action Decoder）
  - [[Camera Adapter|相机适配器]]（LoRA，学语义主动感知先验、不动 VLM 原权重）
  - [[Universal Spatial Knowledge Injection|通用空间知识注入]]（USKI，注入任意 3D 几何配置）
- **输出**: 解耦的动作块——头部相机动作 $A_{\text{head}}$（2-DoF pitch/yaw）+ 本体操作动作 $A_{\text{other}}$（26-DoF 关节增量）
- **本体平台**: [[Unitree G1]] 人形机器人，双 7-DoF 臂 + 双 6-DoF [[Inspire Hand]]，头部 2-DoF 主动相机

### 问题形式化

把主动操作建模为策略 $\pi_\theta:\mathcal{O}\times\mathcal{L}\rightarrow\mathcal{A}$：给定观测 $O_t$ 与语言指令 $L$，预测**联合动作轨迹** $A_t=\{A_{\text{head},t},A_{\text{other},t}\}$。采用 [[Action Chunking|动作分块]]，在 horizon $k$ 上预测序列：

$$
A_{\text{head},t}=\{a_{\text{head}}^{\tau}\}_{\tau=t}^{t+k-1},\quad a_{\text{head}}^{\tau}\in\mathbb{R}^{2}
$$

$$
A_{\text{other},t}=\{a_{\text{other}}^{\tau}\}_{\tau=t}^{t+k-1},\quad a_{\text{other}}^{\tau}\in\mathbb{R}^{26}
$$

**符号说明**:
- $a_{\text{head}}^{\tau}\in\mathbb{R}^2$: 机器人头部的相对 pitch / yaw 调整（2 自由度）
- $a_{\text{other}}^{\tau}\in\mathbb{R}^{26}$: 双臂 + 双灵巧手的 26-DoF 关节位置增量（相对角度变化）
- 与既有工作把相机运动和操作塞进**同一**动作空间不同，本文**解耦**二者并设计自底向上的两阶段训练。

### 核心模块

#### 模块1: Decoupled Action Heads（解耦动作头）

**设计动机**: 相机运动与本体无关、动态频率高（视觉伺服），操作动作复杂、维度高（规划）。把二者放进同一解码器会互相干扰，并把两个训练阶段在动作空间里耦合，破坏语义感知先验。

**具体实现**:
- 用一个 [[Diffusion Transformer|DiT]]（类比 [[GR00T N1]] 的 System 1）作为联合策略，预测加在动作潜变量上的噪声 $\epsilon$。
- DiT 主干由**自注意力（处理动作时序依赖）+ 交叉注意力（条件于多模态嵌入）交替**堆叠组成。
- 最后解码阶段分裂为两个专用 MLP 解码器：**Camera Action Decoder**（预测 2-DoF 头部 pitch/yaw，$\hat{A}^{cam}_t\in\mathbb{R}^{T\times2}$）与 **Manipulation Action Decoder**（预测 26-DoF 关节，$\hat{A}^{body}_t\in\mathbb{R}^{T\times26}$）。
- 这一"分叉"让智能体能**独立**学习高频视觉伺服（头）与复杂操作规划（体）。

#### 模块2: Camera Adapter with Eagle-2 and LoRA（相机适配器）

**设计动机**: 既要保留 VLM 的高层语义信息，又要高效学到"高层语义 → 低层相机运动"的对齐。全量微调十亿级 VLM 计算昂贵且有**灾难性遗忘**风险。

**具体实现**:
- 主干用 [[Eagle-2]]（SigLIP-2 + SmolLM2），为指令到视觉观测的 grounding 提供强先验。
- 用 [[LoRA]] 在 LLM 注意力块的线性层注入可训练低秩分解矩阵，**冻结预训练权重** $W_0$。
- 仅用不到 **2%** 的可训练参数即可把 Eagle-2 的通用语义特征对齐到主动机器人感知需求，桥接"互联网规模知识"与"机器人规模控制"。

#### 模块3: Universal Spatial Knowledge Injection（通用空间知识注入，USKI）

**设计动机**: 现有 VLA 缺乏视角不变的 3D 几何一致性。增强 3D 空间感知的最直接办法就是**尽可能多地引入准确 3D 信息**。

**具体实现**:
- 采用继承自强大前馈式 3D 几何模型 [[MapAnything]] 的 **Universal Spatial Encoder**，**无需重训或改结构**即可支持任意 3D 几何配置（深度、内外参等）作为可选输入。
- 编码得到的**空间 token 与 VLM 输出 token 逐元素相加**，混合 token 在动作去噪过程中注入 [[Decoupled Action Heads|解耦动作头]]。
- 既强化语义相机运动，又增强主动视角执行的空间精度。

### 关键公式与机制

#### 公式1: [[Diffusion Transformer|DiT]] 解耦去噪损失

$$
\mathcal{L}_{diff}=\mathbb{E}_{\tau,\epsilon}\Big[\lambda_1\big\|V_{\theta}^{cam}(\cdot)-\epsilon^{cam}\big\|^2+\lambda_2\big\|V_{\theta}^{body}(\cdot)-\epsilon^{body}\big\|^2\Big]
$$

**含义**: 在扩散时间步 $\tau$、噪声 $\epsilon$ 下，网络 $V_\theta$ 分别为相机分支与本体分支逼近各自加入的噪声，**两个分支的去噪损失分开计**，体现解耦。

**符号说明**:
- $V_{\theta}^{cam},V_{\theta}^{body}$: 相机/本体两个分支的噪声预测网络，条件于融合嵌入 $\phi$、带噪动作块 $A_t^\tau$、观测状态 $q_t$
- $\epsilon^{cam},\epsilon^{body}$: 加在相机/本体动作潜变量上的真值噪声
- $\lambda_1,\lambda_2$: 两分支权重

#### 公式2: [[LoRA]] 适配前向

$$
h=W_{0}x+\frac{\alpha}{r}BAx
$$

**含义**: 冻结预训练线性投影 $W_0$，仅训练低秩增量 $BA$，实现参数高效的相机适配。

**符号说明**:
- $W_0$: 冻结的预训练权重；$x$: 输入；$h$: 输出
- $B\in\mathbb{R}^{d\times r},\ A\in\mathbb{R}^{r\times k}$: 可训练低秩矩阵，秩 $r\ll d$
- $\alpha$: 缩放因子

#### 公式3: Stage 1 语义主动感知对齐损失

$$
\mathcal{L}_{\text{Stage1}}=\mathcal{L}_{\text{MSE}}(\hat{A}_{\text{head}},A^{*}_{\text{head}})=\frac{1}{T}\sum_{t=1}^{T}\big\|\hat{A}_{\text{head}}^{t}-A_{\text{head}}^{*t}\big\|^{2}
$$

**含义**: 仅监督预测的头部相机运动 $\hat{A}_{\text{head}}$ 与真值 $A^{*}_{\text{head}}$ 的 MSE，让模型先学会"基于语言该往哪看"，成为一个"具身摄影师"。

**符号说明**:
- $\hat{A}_{\text{head}}^{t}$: 第 $t$ 步预测的 2-DoF 相机运动；$A_{\text{head}}^{*t}$: 真值
- $T$: 动作块长度（horizon）
- 本阶段**冻结** Universal Spatial Encoder 与主操作解码器，**只训** Camera Adapter（LoRA）与 Camera Action Decoder。

#### 公式4: Stage 2 主动操作微调总损失

$$
\mathcal{L}_{\text{Stage2}}=\lambda_{\text{head}}\,\mathcal{L}_{\text{head}}+\lambda_{\text{other}}\,\mathcal{L}_{\text{other}}
$$

其中：

$$
\mathcal{L}_{\text{head}}=\big\|\hat{A}_{\text{head}}-A^{*}_{\text{head}}\big\|^{2},\qquad \mathcal{L}_{\text{other}}=\big\|\hat{A}_{\text{body}}-A^{*}_{\text{body}}\big\|^{2}
$$

**含义**: 第二阶段解锁解耦动作头，联合训练相机与本体两个解码器，损失为头部与本体的加权 MSE。

**符号说明**:
- $\mathcal{L}_{\text{head}}$: 头部相机跟踪损失；$\mathcal{L}_{\text{other}}$: 26-DoF 臂/手监督损失
- $\lambda_{\text{head}}=1.0,\ \lambda_{\text{other}}=10.0$: 实践取值，**侧重高维操作空间**同时维持视觉稳定
- 采用**数据混合策略**（ActiveViewPose-200K + 主动操作机器人数据 + 真实数据），防止遗忘已学到的主动感知技能。

---

## 关键图表

<!-- 图片均使用 arXiv HTML 在线链接，未本地化 -->

### Figure 1: Motivation & Overview / 动机与总览

![Figure 1](https://arxiv.org/html/2603.12193v1/x1.png)

**说明**: 提出 [[SaPaVe]]——同时整合语义主动感知（选择性移动视角以揭示杂乱场景中的任务关键线索）与主动视角执行（把新观测落地为即时动作，即便从次优视角也能成功）。(a) 抓 $\mathcal{O}_1$ 白碗需旋转第一人称视角（ego 与第三人称视角都被遮挡），而抓 $\mathcal{O}_5$ 油烟机把手只需小幅上移——直观体现视角调整的**语义差异性**。(b) 引入 ActiveManip-Bench（12 任务 / 100 物体 / 20 场景）以摆脱固定视角基准的局限和真实试验成本。(c) SaPaVe 在该基准上以 **75.2%** 平均成功率超越所有基线。

### Figure 2: Overview of SaPaVe / SaPaVe 整体架构

![Figure 2](https://arxiv.org/html/2603.12193v1/x2.png)

**说明**: SaPaVe 处理 RGB 图像与任务指令，在**解耦动作空间**中输出相机运动与操作动作。该解耦设计支持自底向上的两阶段训练：先用大规模**无本体**相机控制数据培育语义主动感知（编码为 Camera Adapter 中的先验），再用混合数据 + 通用空间知识注入（USKI）灵活吸纳各种几何配置（绝对深度、相机内参等），提升主动视角执行的空间精度。

### Figure 3: Overview of ActiveViewPose-200K / 数据集总览

![Figure 3](https://arxiv.org/html/2603.12193v1/x3.png)

**说明**: ActiveViewPose-200K 为 20 万规模的"图像-语言-相机运动"配对高质量数据集，带有细致语义标注，用于语义相机运动学习。构建流程：从 [[Objaverse]] 精选 4k 高质量带语义标注资产 + 500 个场景 → 启发式算法生成大量"图像→相机运动"对 → 半自动构建 3000 个任务模板 → 连同图像送入 [[GPT-4o]] 生成指令并人工精修。

### Figure 4: Overview of ActiveManip-Bench / 基准总览

![Figure 4](https://arxiv.org/html/2603.12193v1/x4.png)

**说明**: 首个评测主动操作（超越传统固定视角）的仿真基准，基于 [[NVIDIA Isaac Sim]] 构建，G1 人形 + Inspire 双手 + 主动头戴相机；含 12 个带丰富标注的语义主动操作任务，覆盖 100 物体、20 场景，且易扩展。

### Figure 5: Real-world Execution Roll-outs / 真实执行序列

![Figure 5](https://arxiv.org/html/2603.12193v1/x5.png)

**说明**: 真实世界执行 roll-out（第一人称 ego 视图与第三人称视图并列），展示主动头部相机如何配合手-物交互。

### Figure 6: Asset Curation Process / 资产筛选流程（附录）

![Figure 6](https://arxiv.org/html/2603.12193v1/x6.png)

**说明**: 3D 资产筛选与策展示例，对应 ActiveViewPose-200K 的 4k 高质量资产收集环节。

### Figure 7: Synthetic Environments (Infinigen) / 合成环境（附录）

![Figure 7](https://arxiv.org/html/2603.12193v1/x7.png)

**说明**: 用 [[Infinigen]] 结合自建资产库程序化生成的合成环境，含不同光照、家具布局与杂乱分布，支撑数据多样性。

### Figure 8: Atomic Scene Tasks & Viewpoint Movement / 原子任务与视角运动示例（附录）

![Figure 8](https://arxiv.org/html/2603.12193v1/x8.png)

**说明**: 原子场景任务及视角运动示例，演示语义相机运动的标注形式。

### Figure 9: Two-stage Scene Task & Viewpoint Movement / 两阶段任务示例（附录）

![Figure 9](https://arxiv.org/html/2603.12193v1/x9.png)

**说明**: 两阶段场景任务与视角运动示例。

### Figure 10: Real-Robot Hardware Setup / 真机硬件平台（附录）

![Figure 10](https://arxiv.org/html/2603.12193v1/x10.png)

**说明**: 真实平台基于 [[Unitree G1]] 人形 + Inspire 3 灵巧手；自研 3D 打印主动头部，由高精度舵机（Dynamixel XC330-M288-T）驱动独立 pitch/yaw；头部搭载 RealSense D455 RGB-D 相机作为主第一人称传感器。

### Figure 11: Real-world Task Categories / 真实任务类别（附录）

![Figure 11](https://arxiv.org/html/2603.12193v1/x11.png)

**说明**: 真实实验四类任务（遮挡/超视野 × 抓放/铰接操作）及对应场景、资产与不同光照设定。

### Figure 12–13: Simulation Demonstrations / 仿真执行示例（附录）

![Figure 12](https://arxiv.org/html/2603.12193v1/x12.png)
![Figure 13](https://arxiv.org/html/2603.12193v1/x13.png)

**说明**: 12 个主动操作任务的时序帧序列。体现两种关键行为：(1) **主动找视角**——在 Out-of-View 初始化下，智能体先根据语义指令快速旋转相机扫描环境（如"底部柜子"则向下看）再最小化臂动作；(2) **处理遮挡**——FetchFrom 任务中主动探头窥视已打开的抽屉/柜内以定位物体。

### Figure 14–15: Real-World Robot Deployment / 真机部署示例（附录）

![Figure 14](https://arxiv.org/html/2603.12193v1/x14.png)
![Figure 15](https://arxiv.org/html/2603.12193v1/x15.png)

**说明**: 在 [[Unitree G1]] 上的真实执行轨迹（超视野抓放、遮挡铰接操作）。关键展示**抗扰动恢复**：当目标被人为移出当前帧时，SaPaVe 能重新发起主动搜索重获目标，验证主动感知管线的闭环特性。

### Table 1: Semantic Active Perception Evaluation / 语义主动感知评测（成功率 %）

| Method | Val | Test1 | Test2 | Avg. |
|--------|-----|-------|-------|------|
| Qwen2.5-VL-72B | 63.9 | 65.1 | 58.0 | 62.3 |
| Multi-SpatialMLLM | 72.8 | 74.3 | 63.6 | 70.2 |
| Gemini-2.5-Pro | 73.3 | 76.5 | 68.2 | 72.7 |
| **Ours (Stage 1)** | **85.5** | **89.1** | **78.3** | **84.3** |

**说明**: 在 ActiveViewPose-200K 测试集上，SaPaVe（仅 **2B 参数**）全面超越通用 VLM 与专用空间 VLM，平均比 Gemini-2.5-Pro 高 **11.6%**；尤其在 Test2（指令省略明确方向、需从图文推断）优势最大。证明**语义主动感知不是通用 VLM 的涌现能力**，必须在专用数据上直接训练。

### Table 2: Fixed vs Dynamic Cameras (ActiveManip-Bench, sim) / 固定 vs 动态相机（成功率 %）

| Method | Unocc. P.a.P | Occ. P.a.P | OoV P.a.P | Unocc. A.M | Occ. A.M | OoV A.M | Avg. |
|--------|------|------|------|------|------|------|------|
| Fixed Camera | 74 | 46 | 11 | 52 | 27 | 7 | 36.17 |
| Fixed Camera + Wrist Camera | 83 | 62 | 28 | 66 | 51 | 24 | 52.33 |
| Active Camera + Wrist Camera | 86 | 75 | 70 | 74 | 68 | 66 | 73.16 |
| **Ours (Active Camera)** | **85** | **78** | **72** | **76** | **70** | **68** | **74.83** |

**说明**: P.a.P = Pick-and-Place，A.M = articulated Manipulation，OoV = Out-of-View。结论：(1) 固定视角下成功率大幅下滑，Out-of-View 任务暴跌 60%+；(2) 固定头部相机 + 腕部相机仍不足以应对主动操作（尤其 OoV）；(3) **主动相机已足够**——再叠加腕部相机增益甚微甚至有损（多视角不必然更好，关键在动态感知与 3D 理解）。

### Table 3: Real-World Active Manipulation / 真实主动操作（成功率 %）

| Method | Occ. P.a.P | OoV P.a.P | Occ. A.M | OoV A.M | Avg. |
|--------|------|------|------|------|------|
| $\pi_0$ | 55 | 45 | 45 | 35 | 45.00 |
| GR00T-N1 | 60 | 55 | 50 | 50 | 53.75 |
| **Ours** | **90** | **85** | **85** | **80** | **85.00** |

**说明**: 把 $\pi_0$ / [[GR00T N1]] 的动作空间扩展加入相机运动并微调后，二者在几乎所有任务上仍显著落后。SaPaVe 平均 **85%**，相对 GR00T-N1 领先 **31.25%**、相对 $\pi_0$ 领先 **40%**。原因：(1) 直接微调缺乏足够主动感知先验；(2) 现有 VLA 缺专门增强主动视角操作的模块。

### Table 4: Generalization Ability / 泛化能力评测（成功率 %）

| Task Name | Object 1 | Object 2 | Light 1 | Light 2 | Scene 1 | Scene 2 | Original |
|-----------|---------|---------|---------|---------|---------|---------|---------|
| Occluded Pick-and-Place | 85 | 90 | 90 | 95 | 90 | 85 | 90 |
| Out-of-View Pick-and-Place | 85 | 90 | 80 | 90 | 85 | 85 | 85 |
| Occluded Arti-Manip | 80 | 85 | 80 | 85 | 85 | 80 | 85 |
| Out-of-View Arti-Manip | 75 | 80 | 80 | 85 | 80 | 75 | 80 |

**说明**: 在未见物体、不同光照、不同场景/背景三类变化下，SaPaVe 成功率均与 Original 接近甚至更高（最高 95%）。说明：高层语义理解使其能解释分布外物体并正确执行；注入 3D 信息增强了 3D 空间感知，保证多样环境下的鲁棒操作。

### Table 5: Ablation Study / 消融实验（平均成功率 %）

| Ablation | Occ. P.a.P | OoV P.a.P | Occ. A.M | OoV A.M | Avg. |
|----------|------|------|------|------|------|
| w/o. Stage 1 | 65 | 55 | 50 | 45 | 53.75 |
| w/o. Stage 2 | 75 | 60 | 70 | 60 | 66.25 |
| w/o. D.A.H. (解耦动作头) | 80 | 70 | 70 | 65 | 71.25 |
| w/o. C.A. (相机适配器,改全量微调) | 80 | 75 | 70 | 70 | 73.75 |
| w/o. U.S.K.I. (空间知识注入) | 75 | 75 | 65 | 60 | 68.75 |
| **Full SaPaVe** | **90** | **85** | **85** | **80** | **85.00** |

**说明**: 逐项验证（Full 取自 Table 3）。(1) **Stage 1 至关重要**——去掉后 Out-of-View 铰接任务成功率近乎腰斩，说明先建立语义感知先验是完成主动操作的前提；(2) 去 Stage 2 整体下滑，主动操作微调必要；(3) **解耦正确**——统一单解码器会同时学相机/操作，破坏语义感知先验并削弱操作；(4) **轻量 Camera Adapter 优于全量微调**——全量微调四项全降，适配器在学相机旋转的同时保住 VLM 通用高层语义；(5) **USKI 大幅提鲁棒性**——去掉后即便在简单 Occluded P.a.P 上也掉 15%，说明一致的 3D 空间理解对主动操作很关键。

### Table 6: Active Manipulation Task Suite / 12 个任务的成功判据（附录）

| Category | Task Name | Success Criteria (Goal State) |
|----------|-----------|-------------------------------|
| Atomic | Pick | 物体离桌高度 >5cm；速度 ≈ 0 |
| Atomic | Reorient | 相对目标的朝向误差达标 |
| Atomic | OpenDrawer | 平移关节开度 >90% 解析上限 |
| Atomic | CloseDrawer | 平移关节完全闭合 |
| Atomic | OpenCabinet | 旋转关节开度 >80° |
| Atomic | CloseCabinet | 旋转关节完全闭合 |
| Short-Horizon | PickAndPlace | 物体在目标处位置误差达标；稳定放置 |
| Short-Horizon | OpenCloseDrawer | 序列：完全打开(>90%) → 完全关闭 |
| Short-Horizon | OpenCloseCabinet | 序列：完全打开(>80°) → 完全关闭 |
| Long-Horizon | FetchFromDrawer | 序列：开抽屉→取物→放台面→关抽屉（满足全部前置） |
| Long-Horizon | FetchFromCabinet | 序列：开柜→取物→放台面→关柜（满足全部前置） |
| Long-Horizon | PourLiquid | 目标容器液体 >80% 源体积；溢出达标 |

**说明**: 12 个任务按 horizon 长度与复杂度分为 Atomic / Short-Horizon / Long-Horizon 三档；成功由几何阈值（位置、旋转、关节状态）或物理量（液体体积）严格定义，并要求维持目标状态一段稳定期（如 2 秒）。

---

## 实验

### 数据集 / 基准

| 数据集/基准 | 规模 | 特点 | 用途 |
|------|------|------|------|
| ActiveViewPose-200K | 20 万 图像-语言-相机运动对（4k 资产/500 场景/3k 模板） | 语义相机控制；分 Train/Val/Test1(显式方位)/Test2(需推断) | 语义主动感知训练+评测 |
| ActiveManip-Bench | 12 任务 / 100 物体 / 20 场景 | 首个主动操作仿真基准（Isaac Sim，G1+Inspire+主动头相机），6 类任务（遮挡/超视野 × 抓放/铰接） | 主动操作训练+评测 |
| 真实机器人遥操作数据 | 4 类任务（遮挡/超视野 × 抓放/铰接） | Unitree G1 + Inspire 3 + 自研主动头(D455) | 真实训练+评测 |

### 实现细节

- **Backbone**: [[Eagle-2]]（SigLIP-2 + SmolLM2），约 2B 参数；LLM 注意力层挂 [[LoRA]] Camera Adapter（<2% 可训练参数）
- **动作头**: [[Diffusion Transformer|DiT]]（自注意力+交叉注意力交替），末端分裂为 Camera Decoder(2-DoF) 与 Manipulation Decoder(26-DoF)
- **3D 注入**: [[MapAnything]] Universal Spatial Encoder（冻结），空间 token 与 VLM token 逐元素相加后注入动作头
- **训练**: 两阶段——Stage1 仅训 Camera Adapter + Camera Decoder（MSE），Stage2 解锁解耦动作头联合训练（数据混合 ActiveViewPose + 仿真[[DexFlyWheel]]生成的操作数据 + 真实数据），$\lambda_{\text{head}}=1.0,\ \lambda_{\text{other}}=10.0$
- **仿真**: [[NVIDIA Isaac Sim]]；真实数据经 VR 遥操作采集 + 可扩展数据增强 + 半自动语义富化
- **评测指标**: 成功率；语义感知实验中预测落在真值 pitch/yaw 容差内即算成功

### 关键实验结论

- **语义主动感知（Table 1）**: 2B 模型平均 84.3%，超 Gemini-2.5-Pro 11.6%，证明该具身技能需专用数据训练而非通用 VLM 涌现。
- **固定 vs 动态视角（Table 2）**: 动态视角对主动操作至关重要；主动相机即足够，叠加腕部相机增益甚微。
- **对比 VLA（Table 3）**: 真实任务 85%，相对 GR00T-N1 领先 31.25%、相对 $\pi_0$ 领先 40%。
- **泛化（Table 4）**: 未见物体/光照/场景下均保持高成功率（多达 95%）。
- **消融（Table 5）**: Stage1（语义感知先验）最关键；解耦动作头、轻量相机适配器、USKI 各自均带来明显增益。

---

## 批判性思考

### 优点
1. **"解耦 + 自底向上"的洞察清晰且被实验支撑**: 相机运动是本体无关的，独立学习其语义先验既省数据又不破坏操作先验；消融（去 Stage1 / 改统一动作头）双重佐证。
2. **数据-基准双补全**: 同时给出首个语义相机控制数据集（20 万）与首个主动操作基准（12 任务），填补了固定视角评测的空白，对社区有基础设施价值。
3. **真实落地与抗扰动**: 在 Unitree G1 真机上做出闭环主动搜索（目标被移出帧后能重新找回），且真实成功率大幅领先 $\pi_0$/GR00T-N1。

### 局限性
1. **静态基座的工作空间受限**: 作者自陈机器人基座固定，只能靠头部 2-DoF + 双臂在可达范围内主动操作，无法移动机身去够取更远目标（未来转向移动主动操作）。
2. **泛化与对比基线偏窄**: 真实对比仅 $\pi_0$/GR00T-N1 两个基线；泛化扰动为物体/光照/场景的有限变体，缺乏更系统、更强的分布外压力测试。
3. **数据高度合成依赖**: ActiveViewPose-200K 的相机运动由启发式算法 + GPT-4o 指令生成，"最优视角"标签的质量与真实人类注视行为的一致性缺乏定量验证；操作数据部分由 DexFlyWheel 增强，sim-to-real gap 的量化分析不足。

### 潜在改进方向
1. 引入移动底盘，把"主动相机"扩展为"主动全身/导航"，实现真正的移动主动操作。
2. 用更多更强基线与更激进的分布外扰动量化"语义感知先验 → 泛化"的因果，而非仅成功率对比。
3. 对"最优视角"标签做人因/几何一致性校验，并系统分析 $\lambda_{\text{head}}/\lambda_{\text{other}}$、LoRA 秩、3D 几何组合等超参的敏感度。

### 可复现性评估
- [ ] 代码开源（论文仅给项目主页 https://lmzpai.github.io/SaPaVe，未见明确代码 release 声明）
- [ ] 预训练模型（未明确声明）
- [x] 训练细节较完整（附录 E 给出两阶段优化策略、损失、$\lambda$ 取值、LoRA、骨干等）
- [x] 数据集/基准可获取（声明发布 ActiveViewPose-200K 与 ActiveManip-Bench）

---

## 速查卡片

> [!summary] SaPaVe: Active Perception & Manipulation in VLA
> - **核心**: 把相机运动与机械臂动作**解耦**到两个动作头，自底向上先学"往哪看"(语义主动感知先验)、再学"基于该感知去做"(主动视角执行)，并注入通用 3D 几何知识。
> - **方法**: Eagle-2(2B) + LoRA Camera Adapter；DiT 解耦双解码器(2-DoF 相机 / 26-DoF 本体)；MapAnything 编码的空间 token 逐元素加到 VLM token 再注入动作头；Stage1 仅训相机(MSE)，Stage2 混合数据联合训练($\lambda_{\text{head}}{=}1,\lambda_{\text{other}}{=}10$)。
> - **数据/基准**: ActiveViewPose-200K(20 万图像-语言-相机运动对) + ActiveManip-Bench(12 任务/100 物体/20 场景，首个主动操作基准)。
> - **结果**: 语义感知超 Gemini-2.5-Pro 11.6%；真实主动操作 85%，相对 GR00T-N1 +31.25%、相对 $\pi_0$ +40%。
> - **平台**: Unitree G1 + Inspire 双手 + 自研 2-DoF 主动头(RealSense D455)。
> - **主页**: https://lmzpai.github.io/SaPaVe

---

*笔记创建时间: 2026-06-29*
