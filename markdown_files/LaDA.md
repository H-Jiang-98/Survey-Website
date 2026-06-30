---
title: "Language-Grounded Decoupled Action Representation for Robotic Manipulation"
method_name: "LaDA"
authors: [Wuding Weng, Tongshu Wu, Liucheng Chen, Siyu Xie, Zheng Wang, Xing Xu, Jingkuan Song, Heng Tao Shen]
year: 2026
venue: CVPR
tags: [VLA, robotic-manipulation, action-representation, contrastive-learning, language-grounding, soft-label]
zotero_collection: _inbox
image_source: online
arxiv_html: https://arxiv.org/html/2603.12967v1
created: 2026-06-29
---

# Language-Grounded Decoupled Action Representation for Robotic Manipulation

## 元信息

| 项目 | 内容 |
|------|------|
| 作者 | Wuding Weng, Tongshu Wu, Liucheng Chen, Siyu Xie, Zheng Wang, Xing Xu, Jingkuan Song, Heng Tao Shen |
| 机构 | （论文 HTML 未显式列出，作者团队以 Xing Xu / Jingkuan Song / Heng Tao Shen 为核心，电子科技大学/同济等多模态检索方向团队） |
| 会议 | CVPR 2026 |
| 类别 | VLA（Vision-Language-Action）/ 机器人操作 / 动作表征 |
| 日期 | 2026（arXiv v1） |
| 项目主页 | （无） |
| 链接 | [arXiv](https://arxiv.org/abs/2603.12967) / [PDF](https://arxiv.org/pdf/2603.12967) |

---

## 一句话总结

> 把连续 7-DoF 动作显式拆成平移/旋转/夹爪三种"可被语言描述"的原语，用语义引导的软标签对比学习把跨任务相似原语对齐，并用自适应权重平衡对比与模仿损失，从而在保持精细控制的同时大幅提升跨任务/相似任务泛化。

---

## 核心贡献

1. **语言锚定的解耦动作表征（Language-Grounded Decoupled Action Representation）**: 提出把 7-DoF 末端执行器动作显式分解为三种可解释、可被语言模板描述的[[Action Primitive|动作原语]]——平移 $\Delta T$、旋转 $\Delta R$、夹爪 $G$，为低层控制注入显式语义结构，作为视觉-语言-动作对齐的中间层。
2. **语义引导的软标签对比学习（Semantic-Guided Soft-Label Contrastive Learning, SCL）**: 不同于传统 0/1 正负样本对比，用基于语言相似度的**连续软标签亲和矩阵** $S$ 来对齐跨任务的相似原语，捕捉细粒度运动对应、保持运动一致性，是泛化提升的核心机制。
3. **自适应损失加权（Adaptive Weighting, AW）**: 借鉴课程学习思想，用各损失滑动平均（moving-average）动态平衡[[Contrastive Learning|对比损失]]与[[Imitation Learning|模仿损失]]，避免任一信号主导优化、防止过早过拟合到模仿信号，实现稳定有效训练。

---

## 问题背景

### 要解决的问题
机器人操作中，**高层视觉-语言理解**与**低层动作控制**之间存在根本性的异质性（heterogeneity）。语义上不同的任务（如 "pour water" 与 "place bottle"）往往共享底层运动原语（reaching、grasping、rotating），但现有模型无法利用这些共享成分，导致**冗余学习**与**跨任务泛化差**。核心问题是：如何构造一个**既语义锚定（semantically grounded）又可跨任务迁移**的动作表征。

### 现有方法的局限
论文将连接视觉-语言-动作的现有范式归为三类（见 Figure 1），各有缺陷：
1. **端到端 [[VLA]]**（Gato、RT-1、Octo、RT-2、OpenVLA、RoboFlamingo）: 直接从多模态输入映射到低层控制，**纠缠了感知与控制**，缺乏显式结构解耦，无法复用共享运动语义，可解释性差。
2. **[[Latent Action Learning|隐动作学习]]**（UniSkill、LAPO、LAPA、UniAct、UniVLA）: 把动作编码进紧致隐空间，但隐嵌入通常由**视觉/观测差分（observation deltas）**定义，**缺乏显式语义**，跨任务可迁移性受限。
3. **[[Language-Conditioned Policy|语言条件策略]]**（SayCan、CLIPORT、RT-H、Phoenix、CLIP-RT）: 引入自然语言作为监督或中间表征，提升可解释性，但依赖**粗粒度、离散的原语**（如 "move forward"、"close gripper"），无法刻画平移幅度、旋转轴等**细粒度运动参数**。

结论：现有范式要么有语义理解、要么有精确控制，**很少二者兼得**。

### 本文的动机
作者认为这种错位的根本原因在于**缺少一个连接符号意图与连续执行的语义锚定层**。语言天然适合扮演这个角色——它是连接人类意图、视觉感知与机器人控制的通用接口，且语言抽象编码了组合性与语义规律，提供一个让运动概念可被**比较、迁移、泛化**的共享空间。因此本文用**语言锚定的原语**来表征低层动作，赋予控制轨迹显式语义，从而在统一解释框架下对齐视觉、语言、动作。

---

## 方法详解

### 模型架构

LaDA 以**语言作为语义桥梁**连接高层视觉-语言理解与低层控制（见 Figure 2），整体流程：
- **输入**: 视觉观测 $V_t$ + 语言指令 $L_t$ +（训练时）对应的低层动作 $\mathbf{a}_t$
- **Backbone**: 预训练的 [[CLIP]] 编码器（视觉编码器 $f_v$ + 文本编码器 $f_l$，ViT-L/14 视觉主干）
- **核心模块**: ① 语言锚定动作分解（把动作拆成可语言化原语）→ ② [[FiLM]] 融合 + MLP 适配器得到统一动作嵌入 → ③ 软标签双路对比学习 + 自适应加权
- **输出**: 连续 7-DoF 动作 $\mathbf{a}_t=[\Delta x,\Delta y,\Delta z,\phi_x,\phi_y,\phi_z,g]$（推理时由轻量 MLP 动作头直接回归，无需显式原语标签）
- **总参数**: 0.6B（约为 CLIP-RT 1.3B 的一半）

### 核心模块

#### 模块1: Language-Grounded Action Decomposition（语言锚定动作分解）

**设计动机**: 为连续机器人控制注入可解释结构、便于迁移的技能学习。

**具体实现**:
- 定义投影 $\Pi:\mathbf{a}_t\mapsto \mathbf{p}_t$，把每个 7-DoF 动作映射为三种**正交且语言锚定**的运动原语：
  1. **平移原语** $(\Delta T)$ — 语言模板 "Move [dist] meters along [dir]"；
  2. **旋转原语** $(\Delta R)$ — "Rotate [mag] degrees around [axis]"；
  3. **夹爪原语** $(G)$ — 离散命令 "Open" / "Close"。
- 每个原语被**离散化为符号化、语言对齐的 bin**，把连续控制轨迹转成可解释的语义类别。
- 该分解在低层运动学与高层语义间架桥，使动作可在共享语言空间内被表示与比较，为跨任务对齐与组合理解打基础。

#### 模块2: Semantic-Guided Contrastive Learning（语义引导对比学习）

**设计动机**: 以语言为语义脚手架，对齐视觉、语言、动作的多模态嵌入；让具有相似原语语义的动作即使在不同轨迹/任务下也彼此靠近，强制跨任务语义一致性。

**具体实现**:
- 不同于依赖离散正/负对的传统[[Contrastive Learning|对比学习]]，LaDA 用**连续的语义亲和度**（由语言相似度引导），从而对部分相关的动作做**软对齐**（soft alignment），捕捉细粒度运动对应、保持跨任务共享语义。
- 包含三个子机制：软标签相似度构建（模块3）、双路软标签对比（模块4）、自适应加权（模块5）。

#### 模块3: Soft-Label Similarity Construction（软标签相似度构建）

**设计动机**: 把"部分相关"的动作以**渐变权重（graded weights）**对齐，量化跨动作的原语级对应程度。

**具体实现**:
- 构建软标签相似矩阵 $S\in[0,1]^{N\times N}$，整合平移/旋转/夹爪三个维度的原语级对应（见公式1）。
- $M_t, M_r, M_g$ 为二值匹配矩阵（两动作是否共享同一原语属性），$(w_t,w_r,w_g)$ 为控制各分量相对贡献的超参。
- 每个元素 $S_{ij}$ 是动作 $i,j$ 间原语级语义相似度的渐变度量，作为软标签对比目标的细粒度监督。

#### 模块4: Soft-Label Dual-Path Contrastive Learning（软标签双路对比）

**设计动机**: 双重目标——① 让共享原语意图的动作在嵌入空间聚类；② 让每个动作仍锚定在其对应语言描述上，保持语义可解释性。

**具体实现**:
- 用预训练 [[CLIP]] 编码器抽取模态特征：图像 token $v_i=f_v(V_i)$、指令 token $l_i=f_l(L_i)$。
- 通过 [[FiLM]] 将视觉特征以语言为条件调制，再经轻量 MLP 适配器得到统一动作嵌入 $A_i=\text{MLP}(\text{FiLM}(v_i,l_i))$。
- 在亲和矩阵 $S$ 引导下优化两条对比路径：
  - **(i) Action–Action 对齐**: 按 $S_{ij}$ 比例强制隐动作 $(A_i,A_j)$ 相似，鼓励共享原语属性的动作靠近（公式2 $\mathcal{L}_a$）。
  - **(ii) Action–Primitive 对齐**: 把每个隐动作锚定到分词后的原语描述 $P_j=f_l(\mathcal{D}(p_j))$，确保嵌入空间显式系于语言语义（公式3 $\mathcal{L}_m$）。
- 二者均为 $S$ 加权的软标签 [[InfoNCE]] 目标，合并为 $\mathcal{L}_{\text{CL}}=\mathcal{L}_a+\lambda\mathcal{L}_m$（公式4）。

#### 模块5: Adaptive Loss Weighting（自适应损失加权）

**设计动机**: 模仿损失 $\mathcal{L}_{\text{IL}}$（预测离散化的平移/旋转/夹爪原语）提供**粗粒度**原语级监督，对比损失 $\mathcal{L}_{\text{CL}}$ 细化语义关系，二者**语义粒度不同、收敛行为不同**，需防止任一信号主导优化。

**具体实现**:
- 用各损失的滑动平均 $\mathrm{MA}(\cdot)$（在最近若干迭代的滑窗上平滑）计算自适应权重（公式5）。
- 最终目标为 $\mathcal{L}_{total}=w_{\text{CL}}\mathcal{L}_{\text{CL}}+w_{\text{IL}}\mathcal{L}_{\text{IL}}$（公式6）。
- 直觉：损失越大的一支权重越大，从而"补短板"地平衡粗粒度行为监督与细粒度语义对齐，防止过早过拟合模仿信号。

#### 微调与推理
- 软标签对比预训练后，接一个轻量 MLP 动作头，用标准 $\mathcal{L}_1$ 轨迹回归损失微调，精修低层 7-DoF 控制精度。
- 推理时直接以 $(V_t,L_t)$ 为条件输出连续动作，**不需要显式原语标签**，实现高效鲁棒的端到端策略执行。

### 关键公式与机制

#### 公式0: 动作分解与 7-DoF 动作向量

$$
\Pi:\mathbf{a}_{t}\mapsto\mathbf{p}_{t},\qquad \mathbf{a}_{t}=[\Delta x,\Delta y,\Delta z,\phi_{x},\phi_{y},\phi_{z},g]
$$

**含义**: 把 7-DoF 末端执行器动作投影为三种语言锚定原语，并明确动作向量结构（3D 平移 + 3D 旋转 + 二值夹爪）。

**符号说明**:
- $\mathbf{a}_t$: $t$ 时刻 7-DoF 连续动作；$\mathbf{p}_t$: 分解得到的原语集合
- $(\Delta x,\Delta y,\Delta z)$: 平移分量；$(\phi_x,\phi_y,\phi_z)$: 旋转分量；$g$: 夹爪开/合

#### 公式1: [[Soft Label|软标签]]相似矩阵构建

$$
S=\frac{w_{t}M_{t}+w_{r}M_{r}+w_{g}M_{g}}{w_{t}+w_{r}+w_{g}},\qquad S\in[0,1]^{N\times N}
$$

**含义**: 把平移/旋转/夹爪三类原语的二值匹配矩阵加权融合并归一化，得到动作间渐变的原语级语义相似度，作为软标签监督。

**符号说明**:
- $M_t, M_r, M_g$: 平移/旋转/夹爪的二值匹配矩阵（是否共享同一原语属性）
- $w_t, w_r, w_g$: 各分量相对贡献的超参；$S_{ij}\in[0,1]$: 动作 $i,j$ 的软相似度

#### 公式2: [[InfoNCE]] —— Action–Action 软标签对比损失

$$
\mathcal{L}_{a}=-\sum_{i=1}^{N}\sum_{j=1}^{N} S_{ij}\log\frac{\exp(\operatorname{sim}(A_{i},A_{j})/\tau)}{\sum_{k=1}^{N}\exp(\operatorname{sim}(A_{i},A_{k})/\tau)}
$$

**含义**: 以软标签 $S_{ij}$ 为权重的 InfoNCE，让共享原语属性的隐动作 $A_i,A_j$ 在嵌入空间按语义亲和度成比例靠近。

**符号说明**:
- $A_i=\text{MLP}(\text{FiLM}(v_i,l_i))$: 统一动作嵌入；$\operatorname{sim}(\cdot,\cdot)$: 余弦相似度
- $\tau$: 温度参数；$S_{ij}$: 软标签权重（公式1）；$N$: batch 内样本数

#### 公式3: Action–Primitive 软标签对比损失

$$
\mathcal{L}_{m}=-\sum_{i=1}^{N}\sum_{j=1}^{N} S_{ij}\log\frac{\exp(\operatorname{sim}(A_{i},P_{j})/\tau)}{\sum_{k=1}^{N}\exp(\operatorname{sim}(A_{i},P_{k})/\tau)}
$$

**含义**: 把隐动作 $A_i$ 锚定到原语语言描述嵌入 $P_j$，确保动作嵌入空间显式系于语言语义、保持可解释性。

**符号说明**:
- $P_j=f_l(\mathcal{D}(p_j))$: 原语描述 $\mathcal{D}(p_j)$ 经文本编码器得到的 token；其余符号同公式2

#### 公式4: 对比总损失

$$
\mathcal{L}_{\text{CL}}=\mathcal{L}_{a}+\lambda\,\mathcal{L}_{m}
$$

**含义**: 组合 Action–Action 一致性与 Action–Primitive 语言锚定两条路径。

**符号说明**:
- $\lambda$: 调节"动作内一致性"与"语言锚定"权衡的系数

#### 公式5: 自适应权重（基于损失滑动平均）

$$
w_{\text{IL}}=\frac{\mathrm{MA}(\mathcal{L}_{\text{IL}})}{\mathrm{MA}(\mathcal{L}_{\text{IL}})+\mathrm{MA}(\mathcal{L}_{\text{CL}})},\qquad
w_{\text{CL}}=\frac{\mathrm{MA}(\mathcal{L}_{\text{CL}})}{\mathrm{MA}(\mathcal{L}_{\text{IL}})+\mathrm{MA}(\mathcal{L}_{\text{CL}})}
$$

**含义**: 用各损失滑动平均的相对大小做动态加权（损失大者权重大），平衡模仿与对比两类粒度不同的监督信号。

**符号说明**:
- $\mathrm{MA}(\cdot)$: 滑窗平滑均值；$\mathcal{L}_{\text{IL}}$: 模仿损失；$\mathcal{L}_{\text{CL}}$: 对比损失（公式4）
- 显然 $w_{\text{IL}}+w_{\text{CL}}=1$

#### 公式6: 总训练目标

$$
\mathcal{L}_{total}=w_{\text{CL}}\mathcal{L}_{\text{CL}}+w_{\text{IL}}\mathcal{L}_{\text{IL}}
$$

**含义**: 自适应加权后的对比+模仿联合目标，在训练全程平衡粗粒度行为监督与细粒度语义对齐，防止过早过拟合模仿信号。

**符号说明**:
- $w_{\text{CL}}, w_{\text{IL}}$: 公式5 的自适应权重；微调阶段另用 $\mathcal{L}_1$ 轨迹回归损失精修动作头

---

## 关键图表

<!-- 图片均使用 arXiv HTML 在线链接，未本地化 -->

### Figure 1: Comparison of VLA Paradigms / 三类范式对比与 LaDA 定位

![Figure 1](https://arxiv.org/html/2603.12967v1/images/Fig1-existing_methods_overview-new.png)

**说明**: 对比视觉-语言-动作学习的三种代表范式：(a) 端到端 [[VLA]] 纠缠感知与控制；(b) [[Latent Action Learning|隐动作]]无显式语义；(c) [[Language-Conditioned Policy|离散语言条件原语]]缺乏细粒度运动锚定。LaDA 用语言作为语义桥梁，通过软标签对比学习解耦并对齐视觉/语言/动作表征，弥合上述三者的缺口。此图是全文动机的核心：它把"语义理解 vs 精确控制难以兼得"的问题具象化。

### Figure 2: Overview of LaDA Framework / 方法总览

![Figure 2](https://arxiv.org/html/2603.12967v1/images/Overview_of_Method-new.png)

**说明**: LaDA 总体框架。把连续 7-DoF 动作分解为平移/旋转/夹爪三种可解释原语并编码进共享语义嵌入空间；语义引导的软标签对比学习跨任务对齐多模态表征，自适应加权动态平衡模仿与对比目标。此图是理解"分解→对比对齐→自适应加权→微调推理"全流程的主图。

### Figure 3: Simulation Setup / 仿真任务示例

![Figure 3](https://arxiv.org/html/2603.12967v1/images/simulation-setup.png)

**说明**: 仿真环境任务示例。上排为 [[LIBERO]] 基准的语言条件操作场景；下排为 [[MimicGen]] 的接触丰富（contact-rich）操作技能。佐证评测覆盖从短程空间推理到长程精密装配的多样任务面。

### Figure 4: Generalization Evaluation / 泛化评测

![Figure 4](https://arxiv.org/html/2603.12967v1/images/Generalization-result.png)

**说明**: 在 LIBERO-Goal 上对**新任务（cross-task）**与**相似任务（similar-task）**的泛化评测，按 4 个训练数据比例（0%/20%/50%/100%）平均，每点 1000 rollout × 20 随机种子。关键结论：跨任务 "push" 指令上 CLIP-RT* 为 0%，LaDA 达 12.3%，说明语言锚定动作表征能把原语级语义复用到训练分布之外。

### Figure 5: Single-task vs Multi-task on MimicGen / 单任务 vs 多任务训练

![Figure 5](https://arxiv.org/html/2603.12967v1/images/single-multi_task_results.png)

**说明**: MimicGen（Stack、StackThree、Threading）单任务与多任务训练的平均成功率对比。LaDA 在多任务设置下增益更大，而 CLIP-RT 仅有边际提升——证明 LaDA 的语义结构更利于跨相关技能共享运动模式。

### Figure 6: t-SNE of Action Embeddings / 动作嵌入 t-SNE 可视化

![Figure 6](https://arxiv.org/html/2603.12967v1/images/t-sne-all.png)

**说明**: 学习到的动作嵌入的 [[t-SNE]] 可视化。(a,b) 无/有 LaDA 的嵌入分布——LaDA 产生更紧致、语义结构化的簇；(c,d) 平移与旋转原语的投影——不同任务的动作呈现重叠模式，表明 LaDA 捕捉到一致的跨任务运动语义。这是"共享运动语义"主张的定性证据。

### Figure 7: Real-World Setup & Rollout / 真实世界平台与执行

![Figure 7](https://arxiv.org/html/2603.12967v1/images/real-world-setup-new.png)

**说明**: 真实世界平台——7-DoF [[Franka Emika Panda]] + 第三人称静态 RGB 相机（RealSense D435i）（左）；右侧为成功的 pick-and-place 执行快照序列。佐证学习到的表征能迁移到真实执行条件，并对光照/物体姿态/颜色/box 位置变化保持鲁棒。

### Table 1: LIBERO Success Rates / LIBERO 成功率对比

| Model | Params | Spatial | Object | Goal | Long | Avg |
|-------|--------|---------|--------|------|------|-----|
| UniACT | 0.5B | 65.0 | 78.0 | 68.0 | 47.0 | 64.5 |
| LAPA | 7B | 73.8 | 74.6 | 58.8 | 55.4 | 65.7 |
| DP (Diffusion Policy) | 147M | 78.3 | 92.5 | 68.3 | 50.5 | 72.4 |
| Octo | 93M | 78.9 | 85.7 | 84.6 | 51.1 | 75.1 |
| OpenVLA | 7.5B | 84.7 | 88.4 | 79.2 | 53.7 | 76.5 |
| MDT | 380M | 78.5 | 87.5 | 73.5 | 64.8 | 76.1 |
| CLIP-RT* | 0.6B | 86.8 | 77.2 | 86.8 | 59.4 | 77.5 |
| SpatialVLA | 4B | 88.2 | 89.9 | 78.6 | 55.5 | 78.1 |
| CoT-VLA | 7B | 87.5 | 91.6 | 87.6 | 69.0 | 81.1 |
| WorldVLA | / | 87.6 | 96.2 | 83.4 | 60.0 | 81.8 |
| Dita | 334M | 84.2 | 96.3 | 85.4 | 63.8 | 82.4 |
| ThinkAct | 7B | 88.3 | 91.4 | 87.1 | 70.9 | 84.4 |
| $\pi$-FAST | 2B | 96.4 | 96.8 | 88.6 | 60.2 | 85.5 |
| GR00T-N1.5 | 3B | 92.0 | 92.0 | 86.0 | 76.0 | 86.5 |
| MolmoAct | 7B | 87.0 | 95.4 | 87.6 | 77.2 | 86.6 |
| FlowVLA | 8.5B | 93.2 | 95.0 | 91.6 | 72.6 | 88.1 |
| CLIP-RT | 1.3B | 95.2 | 99.2 | 94.2 | 83.8 | 93.1 |
| **Ours (LaDA)** | **0.6B** | **95.2** | **99.2** | 93.6 | **86.4** | **93.6** |

**说明**: LaDA 以 0.6B 参数（约 CLIP-RT 的一半）取得平均 93.6% 的全场最高，且在**长程任务 LIBERO-Long 上 86.4%** 明显领先，证明语言锚定的解耦动作表征有助于捕捉共享运动语义、支持组合泛化。注意 LaDA 在 Goal 上(93.6) 略低于 CLIP-RT(94.2)，但 Long 上反超(86.4 vs 83.8)。

### Table 2: MimicGen Success Rates / MimicGen 九任务成功率

> 缩写：C=Coffee, S=Stack, ST=StackThree, T=Threading, TPA=ThreePieceAssembly；D0/D1 为不同示范子集。

| Model | C_D0 | C_D1 | S_D0 | S_D1 | ST_D0 | ST_D1 | T_D0 | TPA_D0 | TPA_D1 | Avg. |
|-------|------|------|------|------|-------|-------|------|--------|--------|------|
| OpenVLA | 42% | 18% | 84% | 86% | 36% | 20% | 20% | 28% | 8% | 38% |
| Task-conditioned | 66% | 24% | 88% | 68% | 30% | 6% | 74% | 20% | 0% | 42% |
| Subgoal-conditioned | 76% | 26% | 88% | 74% | 24% | 6% | 78% | 20% | 2% | 44% |
| Motion-conditioned | 68% | 32% | 92% | 84% | 38% | 16% | 58% | 30% | 4% | 47% |
| Subgoal self-reflection | 80% | 32% | 88% | 78% | 32% | 6% | 80% | 34% | 2% | 48% |
| Phoenix | 94% | 48% | 96% | 86% | 50% | 20% | 68% | 52% | 6% | 58% |
| CLIP-RT* | 77% | 34% | 93% | 87% | 68% | 52% | 32% | 11% | 4% | 51% |
| **Ours (LaDA)** | **94%** | 46% | **96%** | **95%** | **76%** | **71%** | 48% | 50% | **25%** | **67%** |

**说明**: LaDA 平均 67% 全场最高，在多步/长程任务（如 StackThree_D1 71%、TPA_D1 25%）上增益尤其明显。即便未引入任何自我修正策略，仍较 Phoenix 高约 9%、较 CLIP-RT* 高约 16%，凸显语言锚定语义对齐的有效性。注意 Threading(T_D0) 上 LaDA(48%) 不及多个含子目标/自反思的基线（68–80%），是接触高精度任务上的相对短板。

### Table 3: Ablation on LIBERO / 组件消融

| Methods | Spatial | Object | Goal | Long | Average |
|---------|---------|--------|------|------|---------|
| w/o SCL（去软标签对比，退化为硬对比） | 79.2 | 82.8 | 76.6 | 63.4 | 75.5 |
| w/o AW（去自适应加权） | 93.6 | 94.4 | 87.2 | 74.4 | 87.4 |
| **LaDA (full)** | **95.2** | **99.2** | **93.6** | **86.4** | **93.6** |

**关键发现**: 去掉软标签对比（SCL）退化最严重（93.6→75.5，−18.1），证明细粒度软语义对齐是捕捉共享运动结构的关键；去掉自适应加权（AW）也明显下降（→87.4，−6.2），说明对比与模仿信号的平衡优化同样重要。二者作用互补。

---

## 实验

### 数据集 / 基准

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| [[OXE\|Open X-Embodiment]] | 超百万真实轨迹、22 种本体；用约 2250 万视觉帧子集 | 大规模预训练；每动作为 7-DoF（3D 平移+3D 旋转+二值夹爪），自动生成结构化语言描述 | 软标签对比预训练 |
| [[LIBERO]] | 4 套件（Spatial/Object/Goal/Long），各 10 任务 × 50 遥操作示范 | 语言条件多任务操作；图像归一化 256×256；每任务 50 次随机试验 | 训练/测试 |
| [[MimicGen]] | 9 任务，每任务约 1000 示范，每任务 50 次 rollout | 接触丰富（contact-rich）、长程装配（如 ThreePieceAssembly）/高精度（Threading） | 训练/测试 |
| 真实世界 | 4 任务（pick-and-place cube into box），100 人类示范 | 7-DoF Franka Panda + RealSense D435i 第三人称相机 | 微调/测试 |

### 实现细节

- **Backbone**: 预训练 [[CLIP]] 编码器（视觉 ViT-L/14；CLIP-RT 重实现为 CLIP-RT* 用相同 ViT-L/14 以公平对比）
- **总参数**: 0.6B（约 CLIP-RT 1.3B 的一半）
- **预训练**: 在 OXE 上做语义引导软标签对比学习（公式1–6）；动作描述自动生成，如 "move 0.5 meters forward, rotate 90 degrees around the z-axis, and close the gripper"
- **微调**: 轻量 MLP 动作头 + 标准 $\mathcal{L}_1$ 轨迹回归损失精修 7-DoF 控制
- **推理**: 以 $(V_t,L_t)$ 直接输出连续动作，无需显式原语标签
- **LIBERO 预处理**: 去除空闲间隔、图像归一化 256×256，遵循 Octo 等标准评测协议
- 更详细超参/设置见论文附录（HTML 主体未展开）

### 关键实验结论

- **LIBERO（Table 1）**: 平均 93.6% SOTA，长程 LIBERO-Long 86.4% 领先；以约一半参数与 CLIP-RT 持平或略优，体现数据高效性。
- **MimicGen（Table 2）**: 平均 67% 全场最高，多步/长程任务增益显著；无自我修正即超 Phoenix ~9%、CLIP-RT* ~16%。
- **泛化（Fig 4/5）**: 跨任务 "push" 上 LaDA 12.3% vs CLIP-RT* 0%；多任务训练增益远大于 CLIP-RT，证明语义结构利于跨技能运动模式共享。
- **真实世界（Fig 7）**: 100 示范微调即可稳定 pick-and-place，对光照/姿态/颜色/box 位置变化鲁棒。
- **消融（Table 3）**: SCL 贡献最大（−18.1），AW 次之（−6.2），二者互补。
- **表征分析（Fig 6）**: t-SNE 显示 LaDA 嵌入更紧致、跨任务平移/旋转原语重叠，佐证捕捉到一致跨任务运动语义。

---

## 批判性思考

### 优点
1. **把"语言作为语义桥梁"落到细粒度参数层面**: 不同于 RT-H/CLIP-RT 的粗离散原语，LaDA 的平移/旋转/夹爪原语带可量化的方向/幅度/轴，配合软标签把"部分相似"的动作做渐变对齐——这一设计直接对应"共享运动原语却被现有模型浪费"的核心痛点。
2. **参数高效且 SOTA**: 0.6B 在 LIBERO/MimicGen 双基准均取得最高平均成功率，尤其长程与多步任务增益明显，且以约一半参数追平/超越 CLIP-RT。
3. **泛化主张有定量+定性双重支撑**: 跨任务 12.3% vs 0%、多任务增益、t-SNE 簇结构、消融 −18.1，多角度佐证"语义对齐→泛化"，不是空喊。

### 局限性
1. **接触高精度任务仍弱**: MimicGen 的 Threading(T_D0) 仅 48%，不及多个含子目标/自反思的基线（68–80%）；TPA_D1 也仅 25%。说明细粒度语义对齐对**接触丰富、高精度插入**类任务帮助有限。
2. **真实世界实验单薄**: 仅 1 类 pick-and-place cube into box 任务、单一本体，缺乏长程/多物体/可变形真实任务，"鲁棒泛化"在真实场景的证据偏弱（且 Fig 7 无成功率数字）。
3. **原语离散化与超参依赖经验**: 平移/旋转 bin 的划分、$(w_t,w_r,w_g)$、$\lambda$、温度 $\tau$、滑窗大小等关键超参的取值与敏感度分析在 HTML 主体未给出；离散 bin 可能对精细连续控制带来量化误差。
4. **预训练成本未量化**: 用 OXE 约 2250 万帧做对比预训练，训练算力/时间成本与"无预训练"路线相比未讨论，复现门槛偏高。

### 潜在改进方向
1. 把原语离散 bin 改为连续/可学习的语义编码（或层次化 bin），并对接触丰富任务引入力/触觉原语，缓解 Threading 类短板。
2. 扩充真实世界实验（多任务、多本体、长程），并报告真实成功率与扰动鲁棒性曲线，增强真实泛化证据。
3. 系统化敏感度分析：$(w_t,w_r,w_g)$、$\lambda$、$\tau$、滑窗大小对成功率的影响；用 CKA/表征探针量化"语义对齐→泛化"的因果，而非仅 t-SNE。

### 可复现性评估
- [ ] 代码开源（HTML 主体未给出仓库链接）
- [ ] 预训练模型（未声明 release）
- [ ] 训练细节完整（关键超参/bin 划分指向附录，HTML 主体未含）
- [x] 数据集可获取（OXE / LIBERO / MimicGen 均公开）

---

## 速查卡片

> [!summary] LaDA: Language-Grounded Decoupled Action Representation
> - **核心**: 把 7-DoF 动作显式拆成可语言化的平移/旋转/夹爪原语，用软标签对比学习对齐跨任务相似原语，自适应加权平衡对比与模仿损失。
> - **方法**: CLIP 编码 + FiLM 融合得动作嵌入 → 软标签亲和矩阵 $S$ 引导双路 InfoNCE（Action–Action + Action–Primitive）→ 损失滑动平均自适应加权 → MLP 头 $\mathcal{L}_1$ 微调，推理无需原语标签。
> - **结果**: 0.6B 参数，LIBERO 93.6% / MimicGen 67% 均 SOTA，长程任务尤强；跨任务泛化 12.3% vs CLIP-RT* 0%；消融中 SCL 贡献最大（−18.1）。
> - **代码**: 未公开（CVPR 2026）

---

*笔记创建时间: 2026-06-29*
