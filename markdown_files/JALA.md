---
title: "Joint-Aligned Latent Action: Towards Scalable VLA Pretraining in the Wild"
method_name: "JALA"
authors: [Hao Luo, Ye Wang, Wanpeng Zhang, Haoqi Yuan, Yicheng Feng, Haiweng Xu, Sipeng Zheng, Zongqing Lu]
year: 2026
venue: CVPR
tags: [VLA, latent-action, human-video-pretraining, inverse-dynamics, flow-matching, masked-modeling, in-the-wild]
zotero_collection: _inbox
image_source: online
arxiv_html: https://arxiv.org/html/2602.21736v1
created: 2026-06-29
---

# Joint-Aligned Latent Action: Towards Scalable VLA Pretraining in the Wild

## 元信息

| 项目 | 内容 |
|------|------|
| 作者 | Hao Luo, Ye Wang, Wanpeng Zhang, Haoqi Yuan, Yicheng Feng, Haiweng Xu, Sipeng Zheng, Zongqing Lu |
| 机构 | 北京大学、中国人民大学、BeingBeyond |
| 会议 | CVPR 2026 |
| 类别 | VLA（Vision-Language-Action） |
| 日期 | 2026-02（arXiv v1） |
| 项目主页 | https://research.beingbeyond.com/jala |
| 链接 | [arXiv](https://arxiv.org/abs/2602.21736) / [Code](https://research.beingbeyond.com/jala) |

---

## 一句话总结

> 不重建视频像素，而是让 VLA 中间隐状态（预测嵌入）同时对齐逆动力学潜在动作与真实动作标签，构建"过渡感知、行为中心"的统一潜在动作空间，从而把 750 万条实验室+野外人类视频用于可扩展的 VLA 预训练。

---

## 核心贡献

1. **联合对齐的潜在动作（JALA）**: 提出一种新的 [[Latent Action|潜在动作]]预训练范式——绕开传统 [[LAPA]] 那种基于完整视觉动态重建的多阶段流水线，直接把 VLA 上下文里的**预测嵌入（predictive embedding）**同时与逆动力学导出的潜在动作 + 可用的真实动作标签对齐，得到既可从上下文预测、又锚定在动作语义上的统一潜在空间。
2. **混合人类数据的可扩展利用**: 设计 [[Masked Chunk Prediction|掩码块预测]](MCP) + [[Latent Action Perceiver|潜在动作感知器]](LAP) 的双信号机制，配合**解耦 EMA 更新**的双感知器（LAP/LSP），让带标注的实验室数据与无标注的野外视频在同一接口下联合训练——无标注视频仅用对齐损失即可贡献学习信号。
3. **大规模混合数据集 UniHand-Mix**: 构建 750 万条样本（>2000 小时）的人类操作视频语料，融合 500 万条实验室"指令-视频-MANO 动作"样本与 250 万条 Ego4D 野外"指令-视频"对（约 10% 含伪手势标注），并在仿真与真机上验证联合对齐潜在动作显著优于重建式范式。

---

## 问题背景

### 要解决的问题
[[VLA]] 受限于机器人数据的稀缺与跨本体异构。人类操作视频是丰富而廉价的替代信号，但存在一个**质量-多样性两难**：精标的实验室数据规模小、覆盖窄；海量野外视频（如 [[Ego4D]]）则手部跟踪标签不可靠甚至缺失。如何在**同一框架**里既吃下精标数据的精度、又吃下野外数据的多样性，是本文要解决的核心问题。

### 现有方法的局限
- **重建式潜在动作**（[[LAPA]] 等）: 通过对完整视觉动态做重建来抽取潜在动作作为伪标签，是多阶段流水线，计算/显存开销大；像素级重建会把权重放到外观、背景、相机伪影上，在野外视频里手部小、易遮挡时，这些信号与动作无关，反而稀释了行为相关梯度。
- **依赖精标数据的方法**: 只能用小规模实验室数据，无法规模化。
- **简单堆无标注数据**: 实验证明，缺少恰当结构化监督时，单纯增大无标注视频量反而**有害**（JALA w/o latent 在 Wild split 上明显更差）。

### 本文的动机
作者主张：**不需要重建整段视觉动态**，只需让 VLA 上下文中的中间隐状态（预测嵌入）去对齐由边界帧逆动力学得到的潜在动作。这样得到的表征是**行为中心**而非外观中心的，既能从上下文预测、又锚定动作语义，因此在数据稀缺或域偏移大时迁移更好、训练也更省（同 backbone/数据下墙钟时间 <80% 于 LAPA†）。核心假设：**联合对齐潜在动作 = 过渡感知 + 行为中心的统一空间**，是从异构人类视频规模化预训练 VLA 的可行路径。

---

## 方法详解

### 模型架构

JALA 建立在一个 **Transformer 视觉-语言模型**之上，统一处理视觉 token、指令 token 与运动 token（见 Figure 1、Figure 2）：
- **输入**: 视频 $v=\{v_1,\dots,v_T\}$、文本指令 $x$、手势序列 $\mathcal{M}=\{m_1,\dots,m_T\}$（[[MANO]] 参数：相对关节角 $\theta_t$、全局腕部旋转 $\mathbf{r}_t$、腕部平移 $\tau_t$、手形 $\beta_t$，预训练中静态 $\beta_t$ 被剔除）
- **Backbone**: [[InternVL3]]-2B（28 层注意力），视觉端额外接 [[DINOv3]] 或 [[V-JEPA2]] 自监督编码器作为潜在感知器的输入
- **核心模块**: [[Masked Chunk Prediction|MCP]]（块级掩码建模）+ [[Latent Action Perceiver|LAP]]（逆动力学）+ [[Latent State Perceiver|LSP]]（上下文注入），三者共同塑造**预测嵌入** $h_{i,k}$
- **输出**: 预训练阶段生成运动 token；后训练阶段由 [[Flow Matching|流匹配]] DiT 头输出连续机器人[[Action Chunking|动作块]]
- **运动 token 化**: 15 帧为一个运动块，按腕部/手指拆分各量化为 64 token（共 128 token/块），码本大小 4096，由 [[GRVQ]] 学习；token 用 `<mot>`/`</mot>` 包裹

预训练用 VLA $f_\Theta$ 最大化运动 token 似然（自回归块级生成）：

$$
\max_{\Theta}\sum_{i=1}^{N}\log p\big(A_{i}\mid A_{<i},\,v_{1},\,x;\,\Theta\big)
$$

其中 $A_i$ 是第 $i$ 个 token 化运动块（长度 $T/N$），$v_1$ 是首帧，$x$ 是指令。数据分两类：标注集 $\mathcal{D}_A=\{(x,v,\mathcal{M})\}$ 与无标注集 $\mathcal{D}_U=\{(x,v)\}$，混合训练 $\mathcal{D}=\mathcal{D}_A\cup\mathcal{D}_U$。

### 核心模块

#### 模块1: Masked Chunk Prediction（MCP，掩码块预测）

**设计动机**: 类似 [[GR-1]] 的块级掩码 token 建模，从有标注的手部跟踪标签里捕捉运动信息，让中间隐状态承载**块级运动模式**。

**具体实现**:
- 预训练时把一个块内全部运动 token 替换为 `[MASK]`，块内用**双向注意力**，使模型理解块内运动的相互关系；
- 训练目标是预测原 token；尽管损失对单个 token 求和，但双向注意力保证块内所有 token 被**联合建模**，因此对应隐状态 $h_{i,k}$ 携带块级运动模式。

#### 模块2: Latent Action Perceiver（LAP，潜在动作感知器）

**设计动机**: 让预测嵌入与**视觉动态**对齐——把 LAP 当作逆动力学模型（IDM），从边界帧抽取与动作相关的潜在动作。

**具体实现**:
- 对每个块，LAP 取起止帧 $(v_t, v_{t+\delta})$，用一组固定的可学习查询向量产出 $K$ 个潜在动作 $\{z_{i,1},\dots,z_{i,K}\}$，刻画块级过渡的动态；
- **不做完整视频帧重建**，而是把预测嵌入 $h_{i,k}$ 与潜在动作 $z_{i,k}$ 直接对齐（L1）。

**关键互补关系**: 单独看，LAP 无法保证动作中心表征、MCP 又缺视觉动态；**联合对齐**把隐状态的运动模式与潜在动作的视觉动态融合，把预测嵌入锚定到统一潜在动作空间。

#### 模块3: Joint Perceivers with Decoupled Updates（双感知器 + 解耦更新）

**设计动机**: LAP 用的视觉特征来自独立的自监督 backbone（DINOv3/V-JEPA2），与 VLM 表征空间错位；直接连到预测嵌入会丢信息或产出与上下文不匹配的潜在动作。

**具体实现**:
- 引入与 LAP **共享权重**的 [[Latent State Perceiver|LSP]]：LAP 处理边界帧 $(v_t,v_{t+\delta})$ 生成潜在动作；LSP 用复制的初始帧 $(v_0,v_0)$ 把 VLM 预测上下文接到同一潜在动作空间，二者差异仅来自**输入语义**（动态 vs 上下文）而非结构差异。
- LAP/LSP 均为 2 层 Perceiver（交叉注意力从可学习查询 attend 到视觉特征 + 自注意力 + 2 层 MLP 投影到 VLM 嵌入空间）；为处理双手，共享主干但用**两头 MLP**（通道翻倍后切分左右手）。
- **解耦 + 非对称 EMA**: 异构信号直接耦合会不稳定/坍塌。把 Perceiver 主干与可学习查询分开优化——主干用 **LSP** 的梯度（把视觉特征映到 MCP/对齐塑造的预测嵌入空间），查询用 **LAP** 的梯度（用显式动作线索锚定潜在动作），再通过 EMA 互传权重（公式4、5）。

### 关键公式与机制

#### 公式1: [[VLA]] 预训练目标（块级自回归）

$$
\max_{\Theta}\sum_{i=1}^{N}\log p\big(A_{i}\mid A_{<i},\,v_{1},\,x;\,\Theta\big)
$$

**含义**: 给定首帧与指令、以及之前的运动块，逐块最大化运动 token 的似然，与 VLA 预训练范式一致。

**符号说明**:
- $A_i$: 第 $i$ 个 token 化运动块（长度 $T/N$）；$A_{<i}$: 之前所有块
- $v_1$: 首帧；$x$: 指令；$\Theta$: 全模型参数

#### 公式2: [[Masked Chunk Prediction|MCP]] 损失

$$
\mathcal{L}_{\text{MCP}}=-\sum_{i=1}^{N}\sum_{k=1}^{K}\log p_{\Theta}\big(a_{i,k}\mid A_{<i},\,v,\,x\big)
$$

**含义**: 块级掩码 token 建模损失；块内双向注意力使 $K$ 个 token 联合建模，隐状态 $h_{i,k}$ 因此携带块级运动模式。

**符号说明**:
- $a_{i,k}$: 第 $i$ 块第 $k$ 个运动 token；$K$: 每块 token 数
- $A_{<i}$: 之前块（保持完整不掩码）；$v,x$: 视频与指令上下文

#### 公式3: [[Latent Action|潜在动作]]联合对齐损失

$$
\mathcal{L}_{\text{Align}}=\sum_{i=1}^{N}\sum_{k=1}^{K}\big\|h_{i,k}-z_{i,k}\big\|_{1}
$$

**含义**: 用 L1 距离把预测嵌入 $h_{i,k}$ 与 LAP 产出的潜在动作 $z_{i,k}$ 对齐，把上下文运动模式与逆动力学视觉动态融到统一空间。

**符号说明**:
- $h_{i,k}\in\mathbb{R}^d$: 从预选注意力层取的预测嵌入
- $z_{i,k}$: LAP 由边界帧 $(v_t,v_{t+\delta})$ 产出的潜在动作

#### 公式4-5: 解耦 EMA 更新（非对称）

$$
\theta^{\text{LAP}}_{b}\leftarrow\alpha\,\theta^{\text{LAP}}_{b}+(1-\alpha)\,\theta^{\text{LSP}}_{b}
$$

$$
\theta^{\text{LSP}}_{q}\leftarrow\alpha\,\theta^{\text{LSP}}_{q}+(1-\alpha)\,\theta^{\text{LAP}}_{q}
$$

**含义**: 把 LSP 的主干权重传给 LAP、把 LAP 的查询权重传给 LSP；非对称设计使 LAP 主干与预测上下文一致，同时 LSP 查询逐渐继承动作锚定能力，保证潜在动作既"可从上下文预测"又"锚定动作线索"。

**符号说明**:
- $\theta_b$: Perceiver 主干参数；$\theta_q$: 可学习查询参数（上标 LAP/LSP 区分两个感知器）
- $\alpha\in[0,1)$: EMA 系数（实现取 0.999）

#### 公式6: 混合训练总目标

$$
\mathcal{L}=\mathbf{1}_{\text{labeled}}\cdot\mathcal{L}_{\text{MCP}}+\lambda\,\mathcal{L}_{\text{Align}}
$$

**含义**: 标注数据同时用 MCP + 对齐；无标注野外视频里 MCP 失活，**仅用对齐损失**即可学到与动态对齐的预测嵌入——这正是 JALA 能规模化吃下异构数据的关键。

**符号说明**:
- $\mathbf{1}_{\text{labeled}}$: 指示函数，仅当有手势标签时激活 MCP
- $\lambda$: 对齐损失权重（实现取 0.5）

#### 公式7: 后训练流匹配损失

$$
\mathcal{L}_{\text{FM}}=\mathbb{E}_{\tau,\epsilon,A_{t}}\Big[\big\|V_{\theta}(\{h_{i,k}\},A^{\tau}_{t},q_{t})-(\epsilon-A_{t})\big\|^{2}_{2}\Big]
$$

**含义**: 后训练把预测嵌入迁移到机器人动作空间。带噪动作 $A^\tau_t=\tau A_t+(1-\tau)\epsilon$，模型 $V_\theta$ 学习预测去噪向量场 $\epsilon-A_t$；推理用前向 Euler 积分迭代去噪（$N=4$ 步）。DiT 头交替自注意力（处理本体状态+带噪动作）与交叉注意力（融合预测嵌入 $\{h_{i,k}\}$）。

**符号说明**:
- $V_\theta$: 流匹配向量场网络（16 层 32 头 DiT，hidden=2048）
- $A^\tau_t$: 带噪动作；$\tau$: 时间步；$\epsilon$: 标准高斯噪声
- $q_t$: 机器人本体状态（关节位置+夹爪/手指配置）
- $\{h_{i,k}\}$: 预训练 backbone 给出的预测嵌入（条件输入）

---

## 关键图表

<!-- 图片均使用 arXiv HTML 在线链接，未本地化 -->

### Figure 1: Comparison of Latent Action Paradigms / 潜在动作范式对比

![Figure 1](https://arxiv.org/html/2602.21736v1/x1.png)

**说明**: 三种范式对比。(左) [[LAPA]] 等重建式方法靠多阶段流水线、通过动态重建抽取潜在动作作为伪标签；(中) JALA 引入与潜在动作对齐的**预测嵌入**；(右) 基于 Transformer 的 JALA 实现——中间隐状态作为预测嵌入与潜在动作对齐，输出 token 用可用动作标签监督。这张图直接点明 JALA 与重建式范式的本质区别。

### Figure 2: The JALA Framework / 整体框架

![Figure 2](https://arxiv.org/html/2602.21736v1/x2.png)

**说明**: JALA 框架。**预训练（左）**: 被掩码运动块的隐状态作为预测嵌入，与边界帧导出的潜在动作对齐；LAP 把边界帧映到潜在动作空间提供无标签监督，参数共享的 LSP 注入初始帧上下文，LAP/LSP 经解耦 EMA 互联以稳定训练。**后训练（右）**: 预测嵌入喂入流匹配头做机器人任务迁移。本图是理解 MCP+LAP+LSP+解耦更新如何协同的核心。

### Figure 3: Dataset Statistics of UniHand-Mix / 数据集统计

![Figure 3](https://arxiv.org/html/2602.21736v1/x3.png)

**说明**: UniHand-Mix 三个维度统计。左上：数据类型分布（动作生成、纯视频、动作描述、动作续写）；左下：片段时长分布（1–10 秒，偏短）；中/右：跨 8 个数据源的分布（含环形百分比图）。体现"实验室精标 + 野外多样"的混合设计与多尺度时间覆盖。

### Figure 4: t-SNE of Predictive Embeddings vs Latent Actions / 潜在空间对齐可视化

![Figure 4](https://arxiv.org/html/2602.21736v1/x4.png)

**说明**: 预测嵌入 $h$ 与潜在动作 $z$ 在 Lab/Wild 上的 t-SNE。两个空间聚到**紧邻区域**，且 Wild 样本大幅扩展 Lab 流形——说明是"整合覆盖"而非"割裂域"，定性佐证联合对齐确实弥合了实验室与野外的鸿沟。

### Figure 5: Qualitative Hand-Motion Generation / 手部运动生成定性结果

![Figure 5](https://arxiv.org/html/2602.21736v1/x5.png)

**说明**: 左列实验室、右列野外的手部运动生成（彩色叠加为生成手势）。野外侧能处理拨吉他弦、双手协调编织、用筷子在锅里搅菜等复杂无约束交互；实验室侧能精确续写拔耳机线、把碗内物倒到盖上等细粒度动作，对应 Table 1 的定量趋势。

### Figure 6: Ablation Studies on LIBERO / 消融

![Figure 6](https://arxiv.org/html/2602.21736v1/x6.png)

**说明**: 基于 JALA-dino 在 [[LIBERO]] 上的两项消融。**左**：野外数据占比（0%/25%/50%/100%）越大、下游成功率越高，证明强扩展潜力。**右**：把不同 backbone 层（14/19/24/28）的隐状态喂入流匹配头，**第 19 层**迁移最好、14 层略差、更深层急剧退化——提示对齐把可泛化线索集中在选定层，深层过拟合数据集细节。

### Figure 7: Teleoperated Demonstration Examples / 真机遥操作示范

![Figure 7](https://arxiv.org/html/2602.21736v1/x7.png)

**说明**: 三个真机多步任务的遥操作示范序列。上：Put-Three-Obj（开抽屉→放三个水果→关抽屉）；中：Wipe-Board（抓布→擦标记区→去除墨迹）；下：Water-Plant（抓喷瓶→对准植物→扣扳机）。

### Figure 8: Real-World Robot Task Settings / 真机任务设置

![Figure 8](https://arxiv.org/html/2602.21736v1/x8.png)

**说明**: 三个多步任务的真机设置；Put-Three-Obj 与 Wipe-Board 含 unseen 变体（改桌布纹理/改记号笔颜色）以测视觉偏移鲁棒性。

### Figure 9: Successful Rollouts (Seen) / 成功执行（已见设置）

![Figure 9](https://arxiv.org/html/2602.21736v1/x9.png)

**说明**: seen 设置下三任务的成功 rollout（上 Put-Three-Obj、中 Wipe-Board、下 Water-Plant）。

### Figure 10: Successful Rollouts (Unseen) / 成功执行（未见设置）

![Figure 10](https://arxiv.org/html/2602.21736v1/x10.png)

**说明**: unseen 设置下的成功 rollout。Put-Three-Obj 在改变桌布纹理时策略**自我纠正**初始错位；Wipe-Board 在改变记号笔颜色时**自适应重访**残留墨迹——这种未被显式监督的闭环纠错行为，作者归因于潜在动作建模。

### Figure 11: Failure Cases / 失败案例

![Figure 11](https://arxiv.org/html/2602.21736v1/x11.png)

**说明**: 三任务的代表性失败。Put-Three-Obj：空间错位、接触不全、抓取不稳（细长香蕉力矩不稳尤甚）；Wipe-Board：平面接触不足、墨迹残留；Water-Plant：affordance 推理错误致喷瓶朝向错误。指出仍存的精确对齐、接触稳定与细粒度 affordance 建模挑战。

### Table 1: Hand Motion Generation (Lab & Wild) / 手部运动生成

| Model | MPJPE↓ Lab | MPJPE↓ Wild | PA-MPJPE↓ Lab | PA-MPJPE↓ Wild | MWTE↓ Lab | MWTE↓ Wild | MDE↓ Lab | MDE↓ Wild |
|-------|-----------|------------|---------------|----------------|-----------|------------|----------|-----------|
| *Next Token Prediction* | | | | | | | | |
| Being-H0 | 7.61 | 16.91 | 1.34 | 3.81 | 6.03 | 14.54 | 7.16 | 18.33 |
| Being-H0+dino | 7.54 | 15.14 | 0.90 | 2.78 | 5.85 | 13.65 | 6.95 | 17.17 |
| *Masked Chunk Prediction* | | | | | | | | |
| JALA w/o align | 7.72 | 15.73 | 0.89 | 2.34 | 6.18 | 14.02 | 8.09 | 16.23 |
| JALA w/o latent | 8.26 | 20.34 | 1.83 | 3.94 | 8.02 | 17.13 | 10.17 | 26.74 |
| **JALA-dino** | **7.16** | **11.02** | 0.91 | **1.12** | **5.77** | **9.79** | 7.24 | **11.04** |
| **JALA-vjepa** | **7.05** | 11.54 | 0.94 | 1.32 | 5.85 | 10.04 | **6.73** | 11.87 |

**说明**: 两个 JALA 变体几乎在所有指标上胜出。Lab split 提升温和，**Wild split 提升显著**（如 MPJPE 从 ~15–17 降到 ~11，MDE 从 ~17–18 降到 ~11）；Wild 上的退化幅度远小于基线，证明联合对齐潜在动作捕捉到能迁移出实验室的操作先验。值得注意：JALA w/o latent 在更多数据下反而最差（Wild MDE 26.74），说明无结构化监督地堆无标注视频有害。

### Table 2: Two-View LIBERO / 双视角 LIBERO（成功率 %）

| Model | Spatial | Object | Goal | Long | Average |
|-------|---------|--------|------|------|---------|
| LAPA | 83.4 | 87.6 | 78.2 | 68.8 | 79.5 |
| MolmoAct | 87.0 | 95.4 | 87.6 | 77.2 | 86.6 |
| $\pi_0$-FAST | 96.4 | 96.8 | 88.6 | 60.2 | 85.5 |
| GR00T N1.5 | 94.4 | 97.6 | 93.0 | 90.6 | 93.9 |
| $\pi_0$ | **96.8** | **98.8** | 95.8 | 85.2 | 94.2 |
| UniVLA | 95.4 | 98.8 | 93.6 | 94.0 | 95.5 |
| Being-H0 | 92.6 | 96.8 | 94.0 | 77.4 | 90.2 |
| JALA-act | 93.4 | 97.8 | 94.2 | 91.8 | 94.3 |
| JALA w/o dec. | 64.6 | 58.4 | 61.2 | 42.2 | 56.6 |
| LAPA† | 87.4 | 91.2 | 90.0 | 65.4 | 83.5 |
| JALA⋆ | 95.2 | 96.4 | 97.2 | 94.0 | 95.7 |
| JALA-vjepa | 95.4 | 98.0 | 98.0 | 94.8 | 96.6 |
| **JALA-dino** | 96.0 | 98.2 | 97.4 | **96.0** | **96.9** |

**说明**: 蓝行=仅动作可用子集训练，绿行=全 UniHand-Mix 预训练，LAPA†=用 JALA backbone 在本文数据上重训的 LAPA。JALA-dino 以 **96.9%** 居首（无任何机器人数据预训练）；两个重建基线 LAPA(79.5)/LAPA†(83.5) 明显落后，即使 LAPA† 共享同 backbone+数据仍落后 >13 点，说明差距源自**训练目标**而非数据/结构。同子集上 JALA-act(94.3) 稳超 Being-H0(90.2)，Long 套件提升最大(91.8 vs 77.4)。JALA w/o dec. **暴跌到 56.6%**，凸显解耦 EMA 的关键作用。

### Table 3: Single-View LIBERO / 单视角 LIBERO（成功率 %）

| Model | Spatial | Object | Goal | Long | Average |
|-------|---------|--------|------|------|---------|
| *> 3B Backbones* | | | | | |
| LAPA⋆ | 73.8 | 74.6 | 58.8 | 55.4 | 65.7 |
| OpenVLA | 84.7 | 88.4 | 79.2 | 53.7 | 76.5 |
| TriVLA | 91.2 | 93.8 | 89.8 | 73.2 | 87.0 |
| 4D-VLA | 88.9 | 95.2 | 90.9 | 79.1 | 88.5 |
| UniVLA-human† | 91.2 | 94.2 | 90.2 | 79.4 | 88.7 |
| UniVLA-full†† | **96.5** | 96.8 | **95.6** | **92.0** | **95.2** |
| *≤ 3B Backbones* | | | | | |
| Diffusion Policy | 78.3 | 92.5 | 68.3 | 50.5 | 72.4 |
| Octo | 78.9 | 85.7 | 84.6 | 51.1 | 75.1 |
| UniACT | 77.0 | 87.0 | 77.0 | 70.0 | 77.8 |
| SpatialVLA | 88.2 | 89.9 | 78.6 | 55.5 | 78.1 |
| DiT Policy | 84.2 | 96.3 | 85.4 | 63.8 | 82.4 |
| ThinkAct | 88.3 | 91.4 | 87.1 | 70.9 | 84.4 |
| Being-H0 | 86.6 | 92.8 | 89.6 | 70.4 | 84.9 |
| GR00T N1.5 | 91.4 | 97.6 | 94.0 | 85.6 | 92.1 |
| JALA w/o align | 89.4 | 91.2 | 90.0 | 72.6 | 85.8 |
| JALA w/o latent | 80.4 | 83.6 | 75.2 | 68.6 | 77.0 |
| JALA-vjepa | 91.6 | **98.2** | 94.4 | 84.2 | 92.1 |
| **JALA-dino** | 90.4 | 96.4 | 95.2 | 87.2 | **92.3** |

**说明**: 单视角（去掉辅助 egocentric 相机）更难。JALA-dino 在 ≤3B 组取得 **92.3%** 新 SOTA，超 GR00T N1.5(92.1)，且仅用人类数据预训练，Long 套件提升尤大(87.2)。去掉联合对齐(85.8)明显下降，单纯堆数据(77.0)退化更大，证明联合对齐是关键。即便对比额外用大规模机器人数据的 UniVLA-full(95.2)，纯人类视频的 JALA 也超过仅人类的 UniVLA-human(88.7) 并逼近 full 版。

### Table 4: RoboCasa & GR1 Tabletop / RoboCasa 与 GR1 桌面（成功率 %）

| Model | RoboCasa Syn. | RoboCasa Human | GR1 Tabletop |
|-------|---------------|----------------|--------------|
| GR00T N1.5 | 20.83 | 35.17 | 20.41 |
| LAPA | 16.25 | 22.42 | 11.42 |
| Being-H0 | 23.83 | 31.33 | 12.91 |
| JALA-act | 24.92 | 32.42 | 20.25 |
| JALA w/o dec. | 14.25 | 19.33 | 9.25 |
| LAPA† | 20.25 | 27.33 | 13.50 |
| JALA⋆ | 25.33 | 33.83 | 24.50 |
| **JALA** | **27.58** | **35.42** | **26.33** |

**说明**: 50 示范/任务的 few-shot 设置。JALA 在两个 RoboCasa split 与 GR1 上全面领先，合成数据上优势尤大（域差更大）。GR1 用**灵巧手末端**（更接近人类手形），JALA 取 26.33% 大幅超 GR00T(20.41)/LAPA(11.42)/Being-H0(12.91)；同子集上 JALA-act(20.25) 几乎是 Being-H0(12.91) 的两倍，说明下游本体越接近人类、联合对齐越受益。JALA w/o dec. 再次暴跌，LAPA† 全面落后印证训练目标是主因。

### Table 5: Real-World Robot / 真机性能（平均子任务完成率 %）

| Model | Put-Three-Obj Seen | Put-Three-Obj Unseen | Wipe-Board Seen | Wipe-Board Unseen | Water-Plant Seen |
|-------|--------------------|----------------------|-----------------|-------------------|------------------|
| Being-H0 | 38.0 | 16.0 | 40.0 | 33.3 | 36.7 |
| GR00T N1.5 | 48.0 | 28.0 | 56.7 | 43.3 | 53.3 |
| JALA w/o align | 40.0 | 32.0 | 53.3 | 43.3 | 56.7 |
| JALA-vjepa | 38.0 | 34.0 | 66.7 | 60.0 | 66.7 |
| **JALA-dino** | **60.0** | **58.0** | **83.3** | **80.0** | **73.3** |

**说明**: Franka FR3 7-DoF 臂 + Inspire 6-DoF 灵巧手，每任务 10 次 rollout，报子任务完成率。JALA-dino 在所有任务/设置上最优，且在 unseen 视觉偏移下**仅小幅下降**（Put-Three-Obj 60.0→58.0），而基线急剧退化——说明 JALA 更少依赖表面外观、更多依赖动作相关动态。去掉对齐(JALA w/o align)在 unseen 下尤其更差，印证潜在动作对齐是域偏移鲁棒性的关键。

---

## 实验

### 数据集 / 基准

| 基准 | 规模/设置 | 特点 | 用途 |
|------|-----------|------|------|
| [[UniHand-Mix]] | 7.5M 样本，>2000h；5M+ 实验室"指令-视频-MANO"+ 2.5M Ego4D 野外"指令-视频"(约10%伪标注) | 实验室精标 + 野外多样混合 | 预训练 |
| [[LIBERO]] | 4 套件(Spatial/Object/Goal/Long)，<50 示范/任务 | 双视角与单视角两种设置 | 后训练/测试 |
| [[RoboCasa]] | 厨房环境，原子任务，50 示范/任务(合成与人类两split) | 平行夹爪，few-shot | 后训练/测试 |
| GR1 Tabletop | 50 示范/任务，单第三人称视角 | **灵巧手末端**，接近人类手形 | 后训练/测试 |
| 真实世界 | 3 多步任务，50 遥操作示范/任务，10 rollout | Franka FR3 + Inspire 灵巧手，含 unseen 变体 | 后训练/测试 |

### 实现细节

- **Backbone**: [[InternVL3]]-2B（28 层），视觉编码器 [[DINOv3]] 或 [[V-JEPA2]] 喂入潜在感知器
- **运动 token 化**: 15 帧/块，腕部/手指各 64 token（共 128/块），码本 4096，[[GRVQ]] 学习；`<mot>`/`</mot>` 包裹；双手沿时间轴交错；野外数据时间放慢 0.5×
- **预测嵌入层**: 取**第 19 层**（共 28 层）做联合对齐；$\lambda=0.5$，无标注视频仅用对齐损失
- **优化**: AdamW，lr $3\times10^{-5}$、weight decay 0.05、$\beta=(0.9,0.95)$；5% warmup + cosine；grad clip 1.0；EMA $\alpha=0.999$
- **预训练**: 有效 batch 128（per-GPU 16 × 梯度累积 8 卡），全 7.5M 数据训 1 epoch，**68h / 8×A800-80G**
- **后训练**: 16 层 32 头 DiT 头（hidden 2048），batch 128、lr $1\times10^{-4}$；LIBERO 30k 步(~8h)、RoboCasa 60k 步(~16h)、真机 30k 步(~8h)；**仅解冻语言模型参数、视觉编码器冻结**；推理 $N=4$ 去噪步
- **效率对比**: 同 backbone+数据下 LAPA† 需两阶段(29h+57h)，JALA 仅 68h（<80% 墙钟时间）且性能更好

### 关键实验结论

- **手部生成（Table 1）**: Wild split 显著领先；消融揭示 (1) 加更强视觉特征只小幅提升、(2) 预测范式(next-token vs masked-chunk)非主因、(3) 无结构监督堆数据有害、(4) **联合对齐潜在动作是泛化增益的主驱动**；JALA-dino≈JALA-vjepa 说明对自监督 backbone 选择鲁棒。
- **仿真（Table 2-4）**: 双视角 LIBERO 96.9% / 单视角 92.3% 均为同级 SOTA；RoboCasa 与 GR1 全面领先重建基线；重建 vs 联合对齐的受控对比(LAPA†)证明差距源自训练目标。
- **真机（Table 5）**: 三多步任务全面领先，unseen 视觉偏移下尤其稳健，并涌现未被显式监督的**闭环自纠错**行为。
- **消融（Fig 6）**: 野外数据占比越高下游越好（强扩展性）；预测嵌入取第 19 层迁移最优、深层退化。

---

## 批判性思考

### 优点
1. **范式层面的清晰洞见**: "用预测嵌入对齐逆动力学潜在动作，而非重建像素"这一主张有强证据支撑——受控的 LAPA† 对比（同 backbone+数据）把增益干净地归因到训练目标，且训练更省（<80% 墙钟）。
2. **真正吃下野外数据**: 无标注视频仅用对齐损失即可贡献信号，Fig 6 左展示随野外占比单调提升的扩展曲线，回应了"质量-多样性两难"。
3. **稳定性机制有验证**: 解耦 EMA 不是装饰——去掉后 LIBERO 从 ~94 暴跌到 56.6、RoboCasa/GR1 同样崩塌，消融极具说服力。
4. **跨本体与真机泛化**: GR1 灵巧手、Franka+Inspire 真机、unseen 偏移下的鲁棒性与自纠错行为，均超基线。

### 局限性
1. **绝对成功率仍低**: RoboCasa Syn. 仅 27.58%、GR1 26.33%，few-shot 真机最高 83.3%/最低 58%，离实用尚远；失败分析坦承精确对齐、接触稳定、affordance 推理仍是瓶颈。
2. **关键超参偏经验**: "第 19 层取预测嵌入""$\lambda=0.5$""野外放慢 0.5×""EMA 0.999"等多为经验设定，缺系统敏感度分析（仅 Fig 6 右给了层选择）。
3. **对齐有效性证据偏定性**: t-SNE 聚类与定性 rollout 是主要"语义对齐"证据，缺更严格的表征探针/因果度量；自纠错行为的归因（"源自潜在动作建模"）亦为推断而非证明。
4. **野外标签链路依赖外部模型**: Ego4D 子集靠 WiLoR/Gemini-2.5-Flash/HaWoR 过滤与伪标注（置信阈 0.65），这些上游误差如何传播未量化。

### 潜在改进方向
1. 把对齐层、$\lambda$、EMA 系数、野外时间缩放等做成可学习/自动搜索，减少手工经验。
2. 用 CKA/线性探针等量化"行为中心 > 外观中心"的因果，而非仅 t-SNE。
3. 扩展到更长程、更接触丰富、可变形物体任务，并验证范式向更大/更小 backbone 与更多本体的可移植性。
4. 量化野外伪标注链路的误差传播，探索更鲁棒的野外手势监督。

### 可复现性评估
- [x] 项目主页/代码（https://research.beingbeyond.com/jala）
- [ ] 预训练模型（论文未明确声明权重 release）
- [x] 训练细节完整（6.1 给出完整超参、硬件与时长）
- [x] 数据集可获取（UniHand-Mix 基于公开 Ego4D + UniHand 流水线；LIBERO/RoboCasa/GR1 公开）

---

## 速查卡片

> [!summary] JALA: Joint-Aligned Latent Action for Scalable VLA Pretraining
> - **核心**: 不重建像素，让 VLA 中间隐状态(预测嵌入)同时对齐逆动力学潜在动作 + 真实动作标签，得到"过渡感知、行为中心"的统一潜在动作空间。
> - **方法**: MCP(块级掩码建模) + LAP(逆动力学潜在动作) 联合对齐(公式3) + 共享权重 LSP 与解耦非对称 EMA(公式4/5)稳定训练；总目标 $\mathcal{L}=\mathbf{1}_{\text{labeled}}\mathcal{L}_{\text{MCP}}+\lambda\mathcal{L}_{\text{Align}}$；后训练用流匹配 DiT 头迁移到机器人动作。无标注野外视频仅用对齐损失即可参与。
> - **数据**: UniHand-Mix，7.5M 样本/>2000h（5M+ 实验室精标 + 2.5M Ego4D 野外）。
> - **结果**: 双视角 LIBERO 96.9% / 单视角 92.3%(≤3B SOTA)、RoboCasa Syn. 27.58% / GR1 26.33%、真机 unseen 鲁棒；同 backbone 下用 <80% 墙钟时间超越重建式 LAPA†。
> - **主页**: https://research.beingbeyond.com/jala

---

*笔记创建时间: 2026-06-29*
