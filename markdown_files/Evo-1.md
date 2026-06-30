---
title: "Evo-1: Lightweight Vision-Language-Action Model with Preserved Semantic Alignment"
method_name: "Evo-1"
authors: [Tao Lin, Yilei Zhong, Yuxin Du, Jingjing Zhang, Jiting Liu, Yinxinyu Chen, Encheng Gu, Ziyan Liu, Hongyi Cai, Yanwen Zou, Lixing Zou, Zhaoye Zhou, Gen Li, Bo Zhao]
year: 2026
venue: CVPR
tags: [VLA, lightweight-vla, flow-matching, diffusion-transformer, semantic-preservation]
zotero_collection: _inbox
image_source: online
arxiv_html: https://arxiv.org/html/2511.04555v2
created: 2026-06-29
---

# Evo-1: Lightweight Vision-Language-Action Model with Preserved Semantic Alignment

## 元信息

| 项目 | 内容 |
|------|------|
| 作者 | Tao Lin, Yilei Zhong, Yuxin Du, Jingjing Zhang, Jiting Liu 等 14 人 |
| 机构 | 上海交通大学人工智能学院、EvoMind Tech、IAAR-Shanghai、SII、CMU、剑桥大学、南洋理工 |
| 会议 | CVPR 2026 |
| 类别 | VLA（Vision-Language-Action） |
| 日期 | 2025-11（arXiv v2） |
| 项目主页 | https://github.com/MINT-SJTU/Evo-1 |
| 链接 | [arXiv](https://arxiv.org/abs/2511.04555) / [Code](https://github.com/MINT-SJTU/Evo-1) |

---

## 一句话总结

> 用 0.77B 的原生多模态 VLM 当主干、配合纯交叉注意力的扩散动作专家与两阶段训练，在不做机器人数据预训练的前提下保住 VLM 语义、刷新多个操作基准并实现高频实时部署。

---

## 核心贡献

1. **轻量高效架构**: 提出仅 **0.77B 参数**的 [[VLA]] 架构 Evo-1，显著降低训练成本、提升推理速度，可在消费级 GPU（RTX 4090d）上以 16.4 Hz 实时部署。
2. **语义保持以增强泛化**: 提出**两阶段训练范式**，在“保留 VLM 原有多模态理解”与“适配下游动作生成”之间取得平衡，缓解端到端微调对预训练表征空间的破坏，从而提升跨任务泛化。
3. **无预训练即强性能**: 在仿真与真实世界大量实验中，**不依赖大规模机器人数据预训练**即达到 SOTA，省去昂贵的数据采集，挑战了“VLA 必须大规模 robot-pretrain”的惯例。

---

## 问题背景

### 要解决的问题
如何让 [[VLA]] 模型在**参数小、训练便宜、推理高频**的同时，保持强操作性能与泛化能力，并且**不需要大规模机器人数据预训练**。

### 现有方法的局限
作者指出当前 VLA 的四个痛点：
1. **参数巨大**（动辄数十亿），训练/推理都吃显存、算力高；
2. **控制频率低**，大计算开销拖慢实时响应；
3. **端到端训练破坏 VLM 表征空间** —— 视觉-语言主干被动作任务“带偏”，导致下游过拟合、泛化差；
4. **强依赖大规模机器人数据集**（如 OXE、DROID），采集劳动密集且昂贵。

轻量化路线（[[TinyVLA]]、[[SmolVLA]]）虽减参数，但在复杂操作上鲁棒性与成功率仍不足。

### 本文的动机
- 选用**原生多模态 VLM**（[[InternVL3]]-1B）而非“事后对齐”的 text-LLM+视觉，天然具备紧致的跨模态对齐，主干小但表征强；
- 用**纯交叉注意力的扩散 Transformer**做动作专家，提升紧凑性与推理频率；
- 用**两阶段训练**先对齐再微调，避免随机初始化的动作专家把噪声梯度回传、污染预训练语义空间。核心假设：**保住 VLM 语义空间 = 保住泛化**。

---

## 方法详解

### 模型架构

Evo-1 采用模块化的 **感知-语言-动作（perception-language-action）** 流水线（见 Figure 1），由三部分组成：
- **输入**: 多视角 RGB 观测 $\{I_t^i\}_{i=1}^N$ + 语言指令 $L_t$ + 机器人本体状态 $s_t$
- **Backbone**: [[InternVL3]]-1B（视觉 [[InternViT]]-300M + 语言 [[Qwen2.5]]-0.5B）
- **核心模块**: [[Cross-modulated Diffusion Transformer|交叉调制扩散 Transformer]] 作为动作专家 + [[Integration Module|集成模块]] 桥接感知与控制
- **输出**: 连续 [[Action Chunking|动作块]] $\hat A_t=[\hat a_t,\dots,\hat a_{t+H-1}]$
- **总参数**: 0.77B

整体映射写作：

$$
a_t = f_{\text{Evo-1}}\!\left(\{I_t^i\}_{i=1}^N, L_t, s_t;\ \theta\right)
$$

其中 $a_t\in\mathbb{R}^{d_a}$ 为 $t$ 时刻执行的连续动作向量，$\theta$ 为全模型可学习参数。

### 核心模块

#### 模块1: Vision-Language Backbone（视觉-语言主干）

**设计动机**: 用[[原生多模态预训练|native multimodal]]的 [[InternVL3]]-1B 取代“事后对齐”管线，让视觉与语言在同一空间联合学习，跨模态对齐更紧、特征融合更省。

**具体实现**:
- 视觉端用 [[InternViT]]-300M（从 InternViT-6B 经逐层负余弦相似度蒸馏而来）；每张 RGB 缩放到 $448\times448$，经 **pixel-unshuffle** 下采样把视觉 token 数减少 $4\times$，在保留空间粒度的同时压缩序列长度。
- 语言端用 [[Qwen2.5]]-0.5B 解码器，小而强，能捕捉指令 $L_t$ 的空间/逻辑/时序语义。
- 融合方式：把 patch 级图像嵌入替换序列中的 `<img>` 占位 token，再由共享 Transformer 解码器联合推理，得到融合表征 $z_t$。
- **关键工程**: 只保留语言分支的**前 14 层**——经验上中间层的视觉-语言跨模态对齐更强，更利于视觉运动控制。

#### 模块2: Cross-modulated Diffusion Transformer（动作专家）

**设计动机**: 用 [[Flow Matching|流匹配]] 范式建模连续动作轨迹，学习一个随时间变化的速度场，把噪声动作逐步推向真值。

**具体实现**:
- 实现为 [[Diffusion Transformer|DiT]]，**只用堆叠的交叉注意力层**（区别于 $\pi_0$/SmolVLA 的自注意力+交叉注意力交替结构），更紧凑、推理更快。
- 噪声动作由真值 $A_t$ 与随机噪声 $\epsilon$ 线性插值得到（见公式3）；插值权重 $\tau$ 从 **Beta 分布**采样并裁剪到 $[0.02,0.98]$ 以保证数值稳定。
- 训练目标为流匹配损失（公式4），推理时由动作专家 $f_{\text{AE}}$ 一次性预测长度为 $H$ 的动作块（公式5）。

#### 模块3: Integration Module（集成模块）

**设计动机**: 在送入动作专家前，高效融合多模态语义与本体状态，且**尽量不损失信息**。

**具体实现**:
- 从主干**第 14 层**取融合表征 $z_t$（中层语义在视觉/语言间最平衡）；
- **不投影到共享空间**，而是直接把 $z_t$ 与机器人状态 $s_t$ **拼接（concatenate）**，以保留两者的完整信息；
- 拼接特征作为 DiT 各层的 key-value，噪声动作 $A_t^\tau$ 作为 query 做交叉注意力，提供全局且信息无损的条件。
- 这正是消融中表现最好的 **Module A**（见 4.4.1）。

### 关键公式与机制

#### 公式1: [[VLA]] 整体映射

$$
a_t = f_{\text{Evo-1}}\!\left(\{I_t^i\}_{i=1}^N, L_t, s_t;\ \theta\right)
$$

**含义**: 端到端地把多视角图像、语言指令、本体状态映射到连续动作。

**符号说明**:
- $\{I_t^i\}_{i=1}^N$: $t$ 时刻 $N$ 路视角的 RGB 观测
- $L_t$: 语言指令；$s_t$: 机器人本体状态
- $a_t\in\mathbb{R}^{d_a}$: 输出连续动作；$\theta$: 全模型参数

#### 公式2: 融合多模态表征

$$
z_t = f_{\text{VLM}}\!\left(\{I_t^i\}_{i=1}^N, L_t\right)
$$

**含义**: VLM 主干把图像与指令编码为联合表征 $z_t$，作为集成模块的输入。

**符号说明**:
- $z_t\in\mathbb{R}^{d_z}$: 联合编码视觉与语言信息的融合多模态表征（取自第 14 层）

#### 公式3: 噪声动作插值（Flow Matching 前向）

$$
A_t^{\tau} = \tau A_t + (1-\tau)\,\epsilon
$$

**含义**: 在真值动作与噪声之间线性插值，构造时刻 $\tau$ 的"带噪动作"，作为流匹配训练样本。

**符号说明**:
- $A_t$: 真值动作序列；$\epsilon$: 随机采样噪声
- $\tau\sim\text{Beta}(\cdot)$ 并裁剪到 $[0.02,0.98]$: 插值/时间权重

#### 公式4: 流匹配损失

$$
\mathcal{L}^{\tau}(\theta) = \mathbb{E}_{p(A_t\mid z_t,s_t),\,q(A_t^{\tau}\mid A_t)}\!\left[\left\|\mathbf{v}_{\theta}(A_t^{\tau}, z_t, s_t) - \mathbf{u}(A_t^{\tau}\mid A_t)\right\|^2\right]
$$

**含义**: 训练动作专家学习一个时间条件速度场 $\mathbf{v}_\theta$，在多模态上下文 $z_t$ 与状态 $s_t$ 条件下，把带噪动作 $A_t^\tau$ 推向真值 $A_t$。

**符号说明**:
- $\mathbf{v}_\theta$: 待学习的速度场（网络输出）
- $\mathbf{u}(A_t^{\tau}\mid A_t)$: 引导 $A_t^\tau$ 朝向 $A_t$ 的目标流方向
- $\mathbb{E}$: 对动作分布与带噪分布取期望

#### 公式5: 推理（动作块预测）

$$
\hat{A}_t = f_{\text{AE}}(z_t, s_t, A_t^{\tau})
$$

**含义**: 推理时动作专家 $f_{\text{AE}}$ 以融合表征、本体状态、带噪动作为条件，一次输出长度为 $H$ 的未来动作块 $\hat A_t=[\hat a_t,\dots,\hat a_{t+H-1}]$。

---

## 关键图表

<!-- 图片均使用 arXiv HTML 在线链接，未本地化 -->

### Figure 1: Architecture of Evo-1 / 整体架构

![Figure 1](https://arxiv.org/html/2511.04555v2/x1.png)

**说明**: Evo-1 整体架构。RGB 观测与语言指令先经紧致 VLM 主干编码，其融合表征通过集成模块与机器人状态对齐，再由交叉调制扩散 Transformer 生成动作。右侧展示在三个仿真基准上的结果。

### Figure 2: VL Attention Maps (InternVL3-1B vs Prismatic-7B) / 训练后注意力对比

![Evo-1 boy](https://arxiv.org/html/2511.04555v2/pic/evo1_atten_boy.jpg)
![Evo-1 bus](https://arxiv.org/html/2511.04555v2/pic/evo1_atten_bus.jpg)
![Evo-1 dog](https://arxiv.org/html/2511.04555v2/pic/evo1_atten_dog.jpg)
![OpenVLA boy](https://arxiv.org/html/2511.04555v2/pic/openvla_boy.jpg)
![OpenVLA bus](https://arxiv.org/html/2511.04555v2/pic/openvla_bus.jpg)
![OpenVLA dog](https://arxiv.org/html/2511.04555v2/pic/openvla_dog.jpg)

**说明**: 上排为 Evo-1（[[InternVL3]]-1B）训练后的图文注意力，空间一致、语义对齐清晰；下排为 OpenVLA（Prismatic-7B），注意力相干性退化。佐证两阶段训练**保住了原始语义空间**。

### Figure 3: Real-World Task Progress / 真实任务执行序列

![Figure 3](https://arxiv.org/html/2511.04555v2/x2.png)

**说明**: 四个真实世界任务从开始到完成的逐步序列（Pick-and-Place Can、Pour Foam、Hand Delivery、Can Stacking）。

### Figure 4: Real-World Results / 真实世界成功率

![Figure 4](https://arxiv.org/html/2511.04555v2/x3.png)

**说明**: 四个真实任务的成功率与总体平均。Evo-1 平均 78%，超过 SmolVLA(50%)、OpenVLA-OFT(55%)，并以约 1/4 参数量超过 $\pi_0$(73%)。

### Figure 5: Disturbance Settings / 泛化扰动设置

![Figure 5](https://arxiv.org/html/2511.04555v2/x4.png)

**说明**: 泛化实验的四类分布外扰动：(1) 未见干扰物、(2) 背景色变化、(3) 目标位置偏移、(4) 目标高度变化。

### Figure 6: Integration Module Designs (A–D) / 集成模块四种设计

![Module A](https://arxiv.org/html/2511.04555v2/x5.png)
![Module B](https://arxiv.org/html/2511.04555v2/x6.png)
![Module C](https://arxiv.org/html/2511.04555v2/x7.png)
![Module D](https://arxiv.org/html/2511.04555v2/x8.png)

**说明**: 连接 VLM 与动作专家的四种结构。A=中层交叉注意力（最终采用）；B=中层交叉/自注意力交错；C=逐层交叉注意力；D=联合 KV 交叉注意力。

### Figure 7: Single-stage vs Two-stage Attention / 训练范式注意力对比

![1-stage boy](https://arxiv.org/html/2511.04555v2/pic/evo1_1stage_boy.jpg)
![1-stage bus](https://arxiv.org/html/2511.04555v2/pic/evo1_1stage_bus.jpg)
![1-stage dog](https://arxiv.org/html/2511.04555v2/pic/evo1_1stage_dog.jpg)
![2-stage boy](https://arxiv.org/html/2511.04555v2/pic/evo1_atten_boy.jpg)
![2-stage bus](https://arxiv.org/html/2511.04555v2/pic/evo1_atten_bus.jpg)
![2-stage dog](https://arxiv.org/html/2511.04555v2/pic/evo1_atten_dog.jpg)

**说明**: 上排单阶段训练注意力被打乱、语义焦点丢失；下排两阶段（本文）保持对物体区域与任务相关实体的清晰聚焦。

### Figure 8: Ablation Results / 集成模块与训练范式消融

![Fig8a](https://arxiv.org/html/2511.04555v2/x9.png)
![Fig8b](https://arxiv.org/html/2511.04555v2/x10.png)

**说明**: (a) 四种集成模块在 LIBERO-Long 上的成功率，Module A 最优；(b) Meta-World 上两阶段 vs 单阶段训练的对比，两阶段更优。

### Figure 9–10: Robot Setup / 机器人平台（附录）

![Figure 9](https://arxiv.org/html/2511.04555v2/x11.png)
![Figure 10](https://arxiv.org/html/2511.04555v2/x12.png)

**说明**: Figure 9 为真实世界 xArm6（6-DoF + 平行夹爪）平台；Figure 10 为 LeRobot SO-100 部署平台。

### Figure 11–15: Execution Examples / 各基准执行示例（附录）

![Figure 11 Meta-World](https://arxiv.org/html/2511.04555v2/x13.png)
![Figure 12 LIBERO](https://arxiv.org/html/2511.04555v2/x14.png)
![Figure 13 RoboTwin](https://arxiv.org/html/2511.04555v2/x15.png)
![Figure 14 Real-World](https://arxiv.org/html/2511.04555v2/x16.png)
![Figure 15 SO-100](https://arxiv.org/html/2511.04555v2/x17.png)

**说明**: 分别为 Meta-World、LIBERO、RoboTwin、真实世界与 SO-100 的执行示例帧序列。

### Table 1: Simulation Benchmark Results / 仿真基准（Meta-World、LIBERO、RoboTwin）

**Meta-World（成功率 %）**

| Method | Params | Robo-Pretrain | Easy | Medium | Hard | Very Hard | Avg. |
|--------|--------|---------------|------|--------|------|-----------|------|
| Diffusion Policy | - | No | 23.1 | 10.7 | 1.9 | 6.1 | 10.5 |
| TinyVLA-H | 1.3B | No | 77.6 | 21.5 | 11.4 | 15.8 | 31.6 |
| $\pi_0$ | 3.5B | Yes | 71.8 | 48.2 | 41.7 | 30.0 | 47.9 |
| SmolVLA | 2.25B | No | 87.1 | 51.8 | 70.0 | 64.0 | 68.2 |
| **Evo-1 (Ours)** | **0.77B** | **No** | **89.2** | **76.8** | **77.2** | **79.2** | **80.6** |

**LIBERO（成功率 %）**

| Method | Params | Robo-Pretrain | Spatial | Object | Goal | Long | Avg. |
|--------|--------|---------------|---------|--------|------|------|------|
| OpenVLA | 7B | Yes | 84.7 | 88.4 | 79.2 | 53.7 | 76.5 |
| CoT-VLA | 7B | Yes | 87.5 | 91.6 | 87.6 | 69.0 | 81.1 |
| $\pi_0$-FAST | 3.5B | Yes | 96.4 | 96.8 | 88.6 | 60.2 | 85.5 |
| SmolVLA | 2.25B | No | 93.0 | 94.0 | 91.0 | 77.0 | 88.8 |
| GR00T N1 | 2B | Yes | 94.4 | 97.6 | 93.0 | 90.6 | 93.9 |
| $\pi_0$ | 3.5B | Yes | **96.8** | **98.8** | 95.8 | 85.2 | 94.2 |
| **Evo-1 (Ours)** | **0.77B** | **No** | 92.7 | 97.7 | **96.3** | **92.3** | **94.8** |

**说明**: Evo-1 以 0.77B 参数、**无 robot-pretrain**，在 Meta-World 取得 80.6% 全面 SOTA（各难度均最优），LIBERO 94.8% 超过 $\pi_0$(94.2%) 且在 Long 任务上(92.3%)最稳；RoboTwin 套件平均 37.8%（正文，超过 $\pi_0$ 的 30.9%）。

### Table 2: Inference Efficiency / 推理效率（RTX 4090d）

| Model | Params (B) | GPU Mem. (GB) | Infer. Freq. (Hz) | Success (%) |
|-------|-----------|---------------|-------------------|-------------|
| SmolVLA | 0.45 | 2.0 | 12.7 | 50.0 |
| OpenVLA | 7.0 | 15.1 | 7.9 | 55.0 |
| $\pi_0$ | 3.5 | 17.9 | 11.5 | 73.0 |
| **Evo-1 (Ours)** | **0.77** | **2.3** | **16.4** | **78.0** |

**说明**: Evo-1 在显存(2.3GB)、推理频率(16.4Hz 最高)、真实成功率(78% 最高)上取得最佳效率-性能平衡；大模型 OpenVLA/$\pi_0$ 需 >15GB 显存且仅 7–11Hz。

### Table 3: Generalization Success Rates / 泛化成功率（vs SmolVLA）

| Condition | SmolVLA | Ours |
|-----------|---------|------|
| Base | 75% | **95%** |
| 未见干扰物（加未见瓶子） | 65% | **80%** |
| 背景色变化（加黄色桌布） | 60% | **75%** |
| 位置后移 10mm | 75% | **95%** |
| 位置后移 20mm | 60% | **85%** |
| 位置后移 30mm | 60% | **80%** |
| 目标抬高 10mm | 75% | **100%** |
| 目标抬高 20mm | 65% | **90%** |
| 目标抬高 30mm | 60% | **70%** |

**说明**: 所有分布外扰动下 Evo-1 都显著优于 SmolVLA，位置/高度偏移越大优势越明显，体现语义保持带来的鲁棒泛化。

### Table 4–7: 实现配置（附录摘要）

| 表 | 内容 | 关键值 |
|----|------|--------|
| Table 4 | Meta-World 优化超参 | lr=1e-5、batch=16、Stage1 10k 步 / Stage2 65k 步、warmup 1k、grad clip 1.0、weight decay 0.001 |
| Table 5 | 模型与输入配置 | Backbone=InternVL3-1B、Action head=FlowMatching、DiT 8 层、dropout 0.2、图像 448、state/action dim 24(padded)、horizon 50 |
| Table 6 | Meta-World 任务按难度分组 | button-press / door / drawer / faucet / handle / lever / peg 等 |
| Table 7 | LIBERO 任务指令 | Spatial/Object/Goal/Long 各类自然语言指令 |

---

## 实验

### 数据集 / 基准

| 基准 | 规模 | 特点 | 用途 |
|------|------|------|------|
| Meta-World | 每任务 50 示范，10 试 ×5 次独立运行 | 单臂，四难度（easy/medium/hard/very hard） | 训练/测试 |
| LIBERO | 40 任务，4 类（spatial/object/goal/long） | 单臂，侧重不同操作与推理面 | 训练/测试 |
| RoboTwin | 4 代表任务，每任务 50 示范 / 100 试 | **双臂**操作，两个难度 | 训练/测试 |
| 真实世界 | 4 任务，每任务 100 遥操作示范，20 试 | xArm6 6-DoF + 平行夹爪 | 训练/测试 |

### 实现细节

- **Backbone**: InternVL3-1B（InternViT-300M + Qwen2.5-0.5B，语言保留前 14 层）
- **动作专家**: 8 层纯交叉注意力 DiT，Flow Matching；horizon $H=50$；state/action 维度 padding 到 24
- **优化**: lr 1e-5、batch 16、grad clip 1.0、weight decay 0.001；Stage1 10k 步、Stage2 65k 步
- **训练**: 两阶段（冻结 VLM 训动作专家+集成模块 → 解冻全量微调），**无 robot-data 预训练**
- **硬件/部署**: RTX 4090d，2.3GB 显存，16.4Hz；另在 LeRobot SO-100 上验证

### 关键实验结论

- **仿真**: Meta-World 80.6%（SOTA，碾压更大模型）、LIBERO 94.8%、RoboTwin 37.8%（均超先前最佳），尤其在长程任务与双臂协调上稳。
- **真实世界**: 平均 78%，超 SmolVLA/OpenVLA-OFT，且以 1/4 参数超 $\pi_0$。
- **泛化（Table 3）**: 干扰物/背景/位置/高度全面优于 SmolVLA。
- **消融1（集成模块，Fig 8a）**: Module A（中层交叉注意力、信息连续传播）最优；B–D 因插入自注意力或跨层换条件而打断信息连续性。
- **消融2（训练范式，Fig 7/8b）**: 两阶段相对单阶段在 Meta-World 更优，注意力可视化显示其保住了语义聚焦。

---

## 批判性思考

### 优点
1. **效率-性能平衡突出**: 0.77B 同时拿下多基准 SOTA 与最高推理频率，且显存仅 2.3GB，工程落地价值高。
2. **"保住 VLM 语义"这一主张有可验证证据**: 用注意力图对比（Fig 2/7）+ 两阶段消融（Fig 8b）双重支撑，不是空喊泛化。
3. **去预训练**: 不依赖 OXE/DROID 级大规模 robot 数据即达 SOTA，显著降低复现门槛；代码/数据/权重开源。

### 局限性
1. **绝对值仍有限**: RoboTwin 平均仅 37.8%，双臂复杂操作离实用尚远；真实任务集中在桌面抓放/倾倒，长程、接触丰富、可变形物体等未覆盖。
2. **泛化对比基线偏单一**: Table 3 仅与 SmolVLA 比，扰动幅度也较小（mm 级位移、单一背景色），结论的普适性需更多基线与更强扰动支撑。
3. **设计选择的解释偏经验**: "保留语言前 14 层""第 14 层取 $z_t$"等关键超参引用经验结论，缺少系统性敏感度分析；注意力图作为"语义保持"的证据偏定性。

### 潜在改进方向
1. 引入更难的长程/接触/可变形任务与更多强基线，量化"语义保持→泛化"的因果（如表征探针、CKA 等指标而非仅注意力可视化）。
2. 把"保留层数 / 取特征层 / Beta 采样区间"做成可学习或自动搜索，减少手工经验依赖。
3. 探索把两阶段范式迁移到更大/更小主干与跨本体设置，验证该训练范式本身的可移植性。

### 可复现性评估
- [x] 代码开源（https://github.com/MINT-SJTU/Evo-1）
- [x] 预训练模型（声明 release model weights）
- [x] 训练细节完整（附录给出超参与配置表 4/5）
- [x] 数据集可获取（Meta-World/LIBERO/RoboTwin 公开；真实数据声明 release）

---

## 速查卡片

> [!summary] Evo-1: Lightweight VLA with Preserved Semantic Alignment
> - **核心**: 用原生多模态 VLM + 纯交叉注意力扩散动作专家 + 两阶段训练，保住语义、轻量高频。
> - **方法**: InternVL3-1B 主干（取 14 层 $z_t$）→ 拼接状态 $s_t$ 的集成模块 → Flow-Matching DiT 输出动作块；Stage1 冻主干对齐、Stage2 全量微调。
> - **结果**: 0.77B，Meta-World 80.6% / LIBERO 94.8% / RoboTwin 37.8% / 真实 78%，16.4Hz、2.3GB，无 robot-pretrain。
> - **代码**: https://github.com/MINT-SJTU/Evo-1

---

*笔记创建时间: 2026-06-29*
