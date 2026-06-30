---
title: "AVA-VLA: Improving Vision-Language-Action models with Active Visual Attention"
method_name: "AVA-VLA"
authors: [Lei Xiao, Jifeng Li, Juntao Gao, Feiyang Ye, Yan Jin, Jingjing Qian, Jing Zhang, Yong Wu, Xiaoyuan Yu]
year: 2026
venue: CVPR
tags: [VLA, POMDP, active-visual-attention, recurrent-state, visual-token-pruning, OpenVLA-OFT]
zotero_collection: _inbox
image_source: online
arxiv_html: https://arxiv.org/html/2511.18960v4
created: 2026-06-29
---

# AVA-VLA: Improving Vision-Language-Action models with Active Visual Attention

## 元信息

| 项目 | 内容 |
|------|------|
| 作者 | Lei Xiao, Jifeng Li, Juntao Gao, Feiyang Ye, Yan Jin, Jingjing Qian, Jing Zhang, Yong Wu, Xiaoyuan Yu（共 9 人，前两位共同一作） |
| 机构 | 理想汽车（LiAuto Inc.）、北京工业大学、香港中文大学（深圳） |
| 会议 | CVPR 2026 |
| 类别 | VLA（Vision-Language-Action） |
| 日期 | 2026-06（arXiv v4，cs.LG） |
| 项目主页 | https://liauto-dsr.github.io/AVA-VLA-Page/ |
| 链接 | [arXiv](https://arxiv.org/abs/2511.18960) / [HTML](https://arxiv.org/html/2511.18960v4) |

---

## 一句话总结

> 把 [[VLA]] 从“逐帧独立”的 MDP 重述为 [[POMDP]]，用一个由上一时刻隐状态投影得到的"循环状态"近似信念状态，并用它驱动 [[Active Visual Attention|主动视觉注意力]] 动态重加权当前帧的视觉 token，在 LIBERO、CALVIN 与真机双臂任务上刷新 SOTA。

---

## 核心贡献

1. **POMDP 视角重构 VLA**: 指出主流 VLA 把机器人操作隐式当作 [[MDP]]（逐帧独立、history-agnostic），与真实世界"部分可观测"的本质不符；首次提出从 [[POMDP]] 角度建模 VLA，用**循环状态（recurrent state）**作为信念状态的神经近似。
2. **主动视觉注意力（AVA）模块**: 利用循环状态量化当前帧每个视觉 token 的重要性，生成软权重并**调制 LLM 主干所有层的注意力矩阵**，让模型基于"历史信念 + 当前感知"主动聚焦任务相关区域，而非仅靠静态语言指令被动处理。
3. **即插即用、低开销、强迁移**: 仅在 [[OpenVLA-OFT]] 上新增 <50M 参数（<1% 模型量），在 LIBERO（98.0%）、CALVIN ABC→D（avg.len 4.65）与真机 Mobile ALOHA 全面超过基线；软权重还可零成本用于**视觉 token 剪枝**，剪 70% token 仍保持 SOTA 级表现。

---

## 问题背景

### 要解决的问题
如何让 [[VLA]] 模型在序贯决策中**利用历史上下文**来主动、动态地处理当前帧的视觉信息，从而克服逐帧独立处理带来的泛化与鲁棒性不足。

### 现有方法的局限
- 多数 VLA 把每帧视觉观测当作**完整世界状态**独立处理，隐式假设是 [[MDP]]；但真实操作是**部分可观测**的（存在不可见动态、内部状态、遮挡信息），丢弃过去上下文使决策次优。
- 由于逐帧独立，**视觉注意力只受静态语言指令引导**，每步都要"从零"重新评估视觉信息，无法压制时序冗余、无法聚焦被过去动作"激活"的关键区域 —— 视觉系统是**被动**而非主动的。
- 已有引入历史信息的工作（帧对比、KV-cache 复用）主要服务于**视觉 token 剪枝/加速**，并没有提出"动态、上下文感知"的视觉处理范式来同时提升视觉处理与 VLA 泛化。

### 本文的动机
作者观察到"逐帧独立"的核心困难，与 [[POMDP]] 中"如何构建稳健信念状态（belief state）"高度同构：信念状态正是对过去观测与动作的摘要，用以在不确定下指导决策。由于直接计算信念状态一般不可解，作者改为**学习一个压缩表示（循环状态）作为其神经近似**，并用它驱动主动视觉注意力 —— 即把动作预测显式地条件化在循环状态上，而不只依赖当前观测。其底层假设：**序贯决策的时序动态恰好为主动视觉感知创造了机会**（每个动作都会改变下一帧观测）。

---

## 方法详解

### 模型架构

AVA-VLA 以 **[[OpenVLA-OFT]]**（双分支视觉编码器 [[DINOv2]]+[[SigLIP]]，[[LLaMA2]]-7B 主干，并行解码）为基础模型（见 Figure 2），在其上叠加两个由循环状态驱动的轻量组件：
- **输入**: 当前帧观测 $\boldsymbol{x}^t=(\boldsymbol{x}_I^t,\boldsymbol{x}_S^t)$（图像 + 指令/状态）
- **核心模块**: [[Recurrent State|循环状态]] $\boldsymbol{r}^{t-1}$（信念状态的神经近似）+ [[Active Visual Attention|AVA 模块]] $\mathcal{V}$（重加权视觉 token）
- **状态注入**: 用 $\boldsymbol{r}^{t-1}$ 初始化并行解码的动作占位嵌入 $\boldsymbol{p}^t=\boldsymbol{r}^{t-1}$
- **输出**: 并行预测动作块 $\mathcal{A}^t=[a_0^t,\dots,a_{\mathrm{L}_c-1}^t]\in\mathbb{R}^{\mathrm{L}_c\times D}$（$D=7$：3-DoF 平移 + 3-DoF 旋转 + 二值夹爪）
- **新增参数**: <50M（<1% 全模型）

### 核心模块

#### 模块1: Recurrent State（循环状态 / 信念近似）

**设计动机**: 在 [[POMDP]] 下最优策略应同时条件于当前观测与信念状态 $b^{t-1}$；由于 $b^{t-1}$ 不可解，用一个可学习的压缩向量 $\boldsymbol{r}^{t-1}$ 近似之，把 VLA 变成一个**循环结构（非马尔可夫策略）**。

**具体实现**:
- 在并行解码 VLA 中，动作生成前的隐状态已融合视觉与语言信息、对智能体意图有预测性；因此取 $t-1$ 时刻**第 $M$ 层（最后一层）与动作相关的隐状态** $\boldsymbol{h}_M^{t-1}$，经一个 [[MLP]] $\mathcal{B}$ 投影为 $\boldsymbol{r}^{t-1}$（公式5）。
- 该循环状态被双重利用：(a) 作为 AVA 模块的 key/value；(b) 替换 OpenVLA-OFT 原本的空占位嵌入，做**状态化初始化**，保留时序信念。

#### 模块2: Active Visual Attention（主动视觉注意力 $\mathcal{V}$）

**设计动机**: 用循环状态量化当前帧视觉 token 的重要性，**动态调制**对当前帧的视觉处理，使注意力基于"历史信念"而非纯静态指令来过滤与聚焦。

**具体实现**:
- 先用模态专属 MLP 把视觉特征 $\boldsymbol{z}_I^t$ 与指令特征 $\boldsymbol{z}_S^t$ 编码到 $d'$ 维（$d'<d$，降维省算力）。
- 用 [[FiLM]]（feature-wise linear modulation）让指令调制视觉特征：$\hat{\boldsymbol{z}}_I^t=\mathcal{F}_\gamma(\bar{\boldsymbol{z}}_S^t)\odot\bar{\boldsymbol{z}}_I^t+\mathcal{F}_\beta(\bar{\boldsymbol{z}}_S^t)$。
- 以文本调制后的视觉 token 为 query，循环状态为 key/value，做 **cross-attention → self-attention**（公式7–9）。
- 经 FFN + 线性层 $\mathcal{W}:\mathbb{R}^{d'}\to\mathbb{R}^2$ + Softmax，得到每个视觉 token "增强/削弱"的二维 logits $\boldsymbol{\rho}^t$（公式10）；再与 2 维标量向量 $\boldsymbol{\gamma}=[\gamma_0,\gamma_1]$ 内积得到软权重 $\boldsymbol{\omega}^t=\boldsymbol{\rho}^t\boldsymbol{\gamma}$（$\gamma_0$ 增强、$\gamma_1$ 削弱，实验取 $[1.9,0.1]$）。
- **关键**: 软权重 $\boldsymbol{\omega}^t$ 不是简单乘在特征上，而是构造**软注意力掩码** $\mathbf{U}^t$（公式12）后作用于 LLM **每一层**的原始注意力分数 $\mathbf{C}^{t,m}$，重新归一化得到最终注意力 $\mathbf{A}^{t,m}$（公式11）。对角线与非视觉 token 保持 1，仅对视觉 token 列乘以 $\boldsymbol{\omega}^t_j$ —— 因此是对"别人看视觉 token 的权重"做加权，整体行为跨全主干生效。

#### 模块3: 训练与推理（截断 BPTT）

**设计动机**: 循环状态引入了时间依赖，理论上需对整条轨迹做 BPTT，但大 VLA 主干显存/算力不可承受。

**具体实现**:
- 采用 **truncated BPTT**：把模型展开一个固定短窗口 $T=4$（实现里 $K=4$，且在第 2、3 时刻之间 detach 梯度）。
- 每个时刻用 **MAE（平均绝对误差）** 算动作块预测损失 $\mathcal{L}^{t,n}$；并对软权重均值加 **L2 正则** $\mathcal{L}_\omega$（公式13）防止注意力过度弥散。
- 总损失对 $N$ 条截断序列、$T$ 个时刻求和（公式14）。序列首步 $t=0$ 的初始循环状态 $\boldsymbol{r}^{-1}$ 置零嵌入。
- 推理时**全循环运行**：每步单次前向，用上一时刻 $\boldsymbol{r}^{t-1}$ 预测动作块并同时抽出新的 $\boldsymbol{r}^t$，循环至 episode 结束。

### 关键公式与机制

#### 公式1: [[MDP]] 下标准 VLA 前向（AR）

$$
\mathcal{A}^{t}=\mathcal{Q}(\boldsymbol{h}^{t})=\mathcal{Q}(\mathcal{M}(\boldsymbol{z}_{I}^{t},\boldsymbol{z}_{S}^{t}))
$$

**含义**: 标准 VLA 用主干 $\mathcal{M}$ 融合视觉 token $\boldsymbol{z}_I^t$ 与语言 token $\boldsymbol{z}_S^t$ 得隐状态 $\boldsymbol{h}^t$，再由动作头 $\mathcal{Q}$ 解码为动作。

**符号说明**:
- $\mathcal{M}$: LLM 主干；$\mathcal{E}$: 视觉编码器；$\mathcal{T}$: 语言 tokenizer；$\mathcal{Q}$: 动作头/解码器
- $\boldsymbol{z}_I^t=\mathcal{E}(\boldsymbol{x}_I^t)\in\mathbb{R}^{\mathrm{L}_I\times d}$: $\mathrm{L}_I$ 个视觉 token；$d$ 为嵌入维度

#### 公式2: 并行解码前向

$$
\mathcal{A}^{t}=\mathcal{Q}(\mathcal{M}_{\text{parallel}}(\boldsymbol{z}_{I}^{t},\boldsymbol{z}_{S}^{t},\boldsymbol{p}^{t}))
$$

**含义**: [[OpenVLA-OFT]] 风格的并行解码，附加可学习动作占位嵌入 $\boldsymbol{p}^t$ 以同时预测整个动作块 $\mathcal{A}^t\in\mathbb{R}^{\mathrm{L}_c\times D}$。

**符号说明**:
- $\boldsymbol{p}^t$: 动作占位嵌入，OpenVLA-OFT 中设为空 $\bar{\boldsymbol{0}}$
- $\mathrm{L}_c$: 动作块长度；$D$: 单步动作维度

#### 公式3: MDP 策略（被本文否定的假设）

$$
\bar{\mathcal{A}}^{t}\sim\mathcal{P}_{\theta}(\mathcal{A}^{t}\mid\boldsymbol{x}^{t})
$$

**含义**: 现有 VLA 只从当前观测 $\boldsymbol{x}^t$ 预测动作，等价于马尔可夫假设 —— 本文认为这是次优的。

#### 公式4: [[POMDP]] 策略（本文主张）

$$
\bar{\mathcal{A}}^{t}\sim\mathcal{P}_{\theta}(\mathcal{A}^{t}\mid\boldsymbol{x}^{t},b^{t-1})
$$

**含义**: 最优策略应同时条件于当前观测与信念状态 $b^{t-1}$（编码全部历史观测与动作）。由于 $b^{t-1}=P(s_{t-1}\mid\cdot)$ 不可解，用循环状态 $\boldsymbol{r}^{t-1}$ 近似，得非马尔可夫策略 $\mathcal{P}_\theta(\mathcal{A}^t\mid\boldsymbol{x}^t,\boldsymbol{r}^{t-1})$。

**符号说明**:
- $b^{t-1}$: 信念状态；$\boldsymbol{r}^{t-1}$: 其神经近似（循环状态）

#### 公式5: 循环状态的计算

$$
\boldsymbol{r}^{t-1}=\mathcal{B}(\boldsymbol{h}_{M}^{t-1})\in\mathbb{R}^{\mathrm{L}_{A}\times d}
$$

**含义**: 用 MLP $\mathcal{B}$ 把上一时刻最后一层（第 $M$ 层）动作相关隐状态投影为循环状态。

**符号说明**:
- $\boldsymbol{h}_M^{t-1}$: $t-1$ 时刻第 $M$ 层隐状态；$\mathrm{L}_A=\mathrm{L}_c D$: 一次前向预测的动作元素数
- $\mathcal{B}$: 2 层 SiLU 激活的 MLP

#### 公式6: AVA-VLA 完整前向

$$
\mathcal{A}^{t}=\mathcal{Q}(\mathcal{M}_{\text{parallel}}(\boldsymbol{z}_{I}^{t},\mathcal{V}(\boldsymbol{x}^{t},\boldsymbol{r}^{t-1}),\boldsymbol{z}_{S}^{t},\boldsymbol{r}^{t-1}))
$$

**含义**: 在并行解码基础上，注入 AVA 模块 $\mathcal{V}(\boldsymbol{x}^t,\boldsymbol{r}^{t-1})$ 的输出（调制视觉处理），并以 $\boldsymbol{r}^{t-1}$ 作占位嵌入初始化（状态化初始化）。

**符号说明**:
- $\mathcal{V}$: AVA 模块，输入当前观测与循环状态

#### 公式7–9: AVA 内部注意力

$$
\mathbf{Q}^{t}=W_{Q}\hat{\boldsymbol{z}}_{I}^{t},\quad \mathbf{K}^{t},\mathbf{V}^{t}=(W_{K}/W_{V})\hat{\boldsymbol{r}}^{t-1},\quad \mathbf{O}^{t}=\text{Self-Att}\!\left(\text{Cross-Att}(\mathbf{Q}^{t},\mathbf{K}^{t},\mathbf{V}^{t})\right)
$$

**含义**: 以文本调制后的视觉 token 为 query、循环状态为 key/value 做交叉注意力，再过自注意力，得到融合了历史信念的视觉表征 $\mathbf{O}^t$。

**符号说明**:
- $\hat{\boldsymbol{z}}_I^t$: FiLM 调制后的视觉特征；$\hat{\boldsymbol{r}}^{t-1}$: MLP 编码后的循环状态
- $W_Q,W_K,W_V$: 线性投影；$d'<d$: 降维后的维度

#### 公式10: 软权重 logits

$$
\boldsymbol{\rho}^{t}=\text{Softmax}\!\left(\mathcal{W}\!\left(\text{FFN}\!\left(\mathbf{O}^{t}\right)\right)\right)\in\mathbb{R}^{\mathrm{L}_{I}\times 2}
$$

**含义**: 为每个视觉 token 预测"增强 vs 削弱"的二维概率；与 $\boldsymbol{\gamma}=[\gamma_0,\gamma_1]$ 内积得最终软权重 $\boldsymbol{\omega}^t=\boldsymbol{\rho}^t\boldsymbol{\gamma}$，作为视觉 token 的重要性分数。

**符号说明**:
- $\mathcal{W}:\mathbb{R}^{d'}\to\mathbb{R}^2$: 线性层；$\gamma_0,\gamma_1$: 增强/削弱标量（取 $1.9/0.1$）

#### 公式11–12: 软掩码调制全主干注意力

$$
\mathbf{A}_{i,j}^{t,m}=\frac{\exp(\mathbf{C}_{i,j}^{t,m})\,\mathbf{U}_{i,j}^{t}}{\sum_{l=1}^{\mathrm{L}_{o}^{t}}\exp(\mathbf{C}_{i,l}^{t,m})\,\mathbf{U}_{i,l}^{t}},\quad 1\leq i,j\leq \mathrm{L}_{o}^{t}
$$

$$
\mathbf{U}^{t}_{i,j}=\begin{cases}1 & \text{if } i=j \text{ 或 } j\notin\Lambda_{I}\\ \boldsymbol{\omega}^{t}_{j} & \text{if } i\neq j \text{ 且 } j\in\Lambda_{I}\end{cases}
$$

**含义**: 把软权重构造成软注意力掩码 $\mathbf{U}^t$，对第 $m$ 层原始注意力分数 $\mathbf{C}^{t,m}$ 加权并重归一化。只对"指向视觉 token（$j\in\Lambda_I$）且非对角"的项乘以重要性 $\boldsymbol{\omega}^t_j$，其余保持 1 —— 从而在**所有层**抑制无关视觉 token、增强关键区域。

**符号说明**:
- $\mathbf{C}^{t,m}$: 应用原始 mask 后的第 $m$ 层注意力分数；$\mathrm{L}_o^t$: 总序列长度
- $\Lambda_I$: 视觉 token 索引集合

#### 公式13–14: 正则项与总损失

$$
\mathcal{L}_{\omega}^{t,n}=\|\mu(\boldsymbol{\omega}^{t,n})-c\|,\qquad \mathcal{L}_{\text{total}}=\sum_{n=1}^{N}\sum_{t=0}^{T-1}\left(\mathcal{L}^{t,n}+\lambda\mathcal{L}_{\omega}^{t,n}\right)
$$

**含义**: $\mathcal{L}_\omega$ 是对软权重均值的 L2 惩罚，把其拉向目标均值 $c$，防止注意力过度弥散、压制背景响应；总损失是 $N$ 条截断序列、$T$ 步上动作 MAE 损失 $\mathcal{L}^{t,n}$ 与正则项的加权和。

**符号说明**:
- $\mu(\cdot)$: 均值；$c$: 目标均值超参（LIBERO 取 0.6，CALVIN 取 0.2）
- $\lambda$: 平衡系数（取 1.0）；$N$: batch size；$T=4$: 截断窗口

---

## 关键图表

<!-- 图片均使用 arXiv HTML 在线链接，未本地化 -->

### Figure 1: AVA-VLA vs Vanilla VLA / 框架与视觉聚焦对比

![Figure 1](https://arxiv.org/html/2511.18960v4/x1.png)

**说明**: (a) AVA-VLA 与普通 VLA 的对比 —— 普通 VLA 逐帧独立（MDP），AVA-VLA 通过循环状态把历史信念接入当前帧处理（POMDP）。(b) 在"turn on the stove and put the moka pot on it"任务上两视角的视觉聚焦定性对比：vanilla [[OpenVLA-OFT]] 找不到任务关键的"stove"开关，而 AVA-VLA 借历史上下文保持稳定聚焦。这张图直观点出"被动 vs 主动视觉"的差异，是全文动机的视觉化。

### Figure 2: AVA-VLA Framework Overview / 整体框架

![Figure 2](https://arxiv.org/html/2511.18960v4/x2.png)

**说明**: 每个时刻从上一时刻隐状态投影出循环状态，用以保留历史上下文并初始化当前动作 token；AVA 模块把循环状态与文本条件化的当前视觉特征结合，生成软重要性分数，去调制主干 LLM **各层**的视觉注意力矩阵。这是方法的核心数据流图，对应公式 5–12。

### Figure 3: Mobile ALOHA Real-World Results / 真机双臂实验

![Figure 3](https://arxiv.org/html/2511.18960v4/fig/M-Realworld-R.png)

**说明**: 四个真机操作任务（(a) Pick and Place、(b) Sequenced Instruction Understanding、(c) Flexible Object Folding、(d) Dexterous Action）的中间状态与成功率。AVA-VLA 在各任务及跨任务平均上均高于 UniVLA、OpenVLA-OFT 基线，证明 sim-to-real 迁移能力。

### Figure 4: Visual Dynamics / 软权重时序演化

![Figure 4](https://arxiv.org/html/2511.18960v4/fig/M-Att-LIBERO.png)

**说明**: 任务"put both moka pots on the stove"中两视角下软权重 $\boldsymbol{\omega}^t$ 随时间的演化。注意力一致地聚焦在机械臂接触区与目标物体上，定性验证 AVA 能识别任务相关视觉特征。

### Figure 5: AgileX Cobot Magic Platform / 真机平台（附录）

![Figure 5](https://arxiv.org/html/2511.18960v4/fig/realword_robot.png)

**说明**: 基于斯坦福 Mobile ALOHA 的 AgileX Cobot Magic 平台：差速底盘 Tracer + 双臂机械手 + RGB-D 传感器。

### Figure 6: Real-World Task Execution / 真机任务执行序列（附录）

![Figure 6](https://arxiv.org/html/2511.18960v4/fig/app-ED-L.png)

**说明**: 四个长程真机任务的关键观测帧序列，展示 AVA-VLA 在真实场景的执行过程。

### Figure 7–11: Attention Dynamics Across Tasks / 跨任务注意力可视化（附录）

![Figure 7 Mobile ALOHA banana](https://arxiv.org/html/2511.18960v4/fig/app-QV-1.png)
![Figure 8 Mobile ALOHA sesame](https://arxiv.org/html/2511.18960v4/fig/app-QV-2.png)
![Figure 9 CALVIN](https://arxiv.org/html/2511.18960v4/fig/app-QV-3.png)
![Figure 10 LIBERO drawer](https://arxiv.org/html/2511.18960v4/fig/app-QV-4.png)
![Figure 11 LIBERO microwave](https://arxiv.org/html/2511.18960v4/fig/app-QV-5.png)

**说明**: Fig 7/8 为真机（"put yellow banana into bucket"、"scoop sesame into bowl"，各三视角）；Fig 9 为 CALVIN（"Lift red block table"/"Place in slider"，两视角）；Fig 10/11 为 LIBERO（抽屉、微波炉任务，两视角）。一致显示 AVA 能锁定交互目标（如香蕉、桶、勺柄）并抑制无关区域。

### Figure 12: Soft Weights without $L_{\omega}$ / 去正则的注意力

![Figure 12](https://arxiv.org/html/2511.18960v4/fig/without_mean_loss.png)

**说明**: 与 Figure 4 同一任务，去掉正则项 $L_\omega$ 后注意力明显更弥散、在无关背景区响应增多，说明 $L_\omega$ 帮助维持更选择性、结构更稳的注意力掩码。

### Figure 13: Failure Cases / 失败案例（附录）

![Figure 13a](https://arxiv.org/html/2511.18960v4/fig/failurecase_2.png)
![Figure 13b](https://arxiv.org/html/2511.18960v4/fig/failurecase_4.png)

**说明**: (a) 因空间信念漂移，夹爪未对准 chocolate pudding；(b) 轻微位置偏差导致无法稳抓 moka pot 把手。说明微小感知误差会在循环状态中累积、引发长程精细任务的信念漂移失败 —— 这是 POMDP 建模的固有挑战。

### Table 1: LIBERO Benchmark / LIBERO 主结果

**One policy for all 4 suites（单一策略覆盖 4 套件）**

| Method | Spatial | Object | Goal | Long | Avg. |
|--------|---------|--------|------|------|------|
| TraceVLA | 84.6 | 85.2 | 75.1 | 54.1 | 74.8 |
| WorldVLA | 87.6 | 96.2 | 83.4 | 60.0 | 81.8 |
| $\pi_0$ | 96.8 | 98.8 | 95.8 | 85.2 | 94.2 |
| $\pi_0$-FAST | 96.4 | 96.8 | 88.6 | 60.2 | 85.5 |
| UnifiedVLA | 95.4 | 98.8 | 93.6 | 94.0 | 95.5 |
| OpenVLA-OFT | 97.7 | 98.0 | 96.1 | 95.3 | 96.8 |
| **AVA-VLA (Ours)** | 97.4 | **99.4** | **97.4** | **97.6** | **98.0** |

**One policy per suite（每套件独立策略）**

| Method | Spatial | Object | Goal | Long | Avg. |
|--------|---------|--------|------|------|------|
| OpenVLA | 84.7 | 88.4 | 79.2 | 53.7 | 76.5 |
| SpatialVLA | 88.2 | 89.9 | 78.6 | 55.5 | 78.1 |
| CoT-VLA | 87.5 | 91.6 | 87.6 | 69.0 | 83.9 |
| NORA | 92.2 | 95.4 | 89.4 | 74.6 | 87.9 |
| PD-VLA | 95.5 | 96.7 | 94.9 | 91.7 | 94.7 |
| UniVLA | 96.5 | 96.8 | 95.6 | 92.0 | 95.2 |
| OpenVLA-OFT | 97.6 | 98.4 | 97.9 | 94.5 | 97.1 |
| FLOWER | 97.5 | 99.1 | 96.1 | 94.9 | 96.9 |
| RIPT-VLA | 99.0 | 98.6 | 98.6 | 93.8 | 97.5 |
| **AVA-VLA (Ours)** | **99.2** | **99.6** | 97.9 | 96.2 | **98.2** |

**说明**: 两种设置下 AVA-VLA 都拿到最高平均（98.0% / 98.2%），尤其在最难的 LIBERO-Long 上持续领先（97.6% / 96.2 仅次于 RIPT 等），印证"历史上下文对长程任务最有用"。

### Table 2: CALVIN ABC→D Benchmark / CALVIN 零样本泛化

| Method | 1 | 2 | 3 | 4 | 5 | Avg. len ↑ |
|--------|------|------|------|------|------|------|
| OpenVLA | 91.3 | 77.8 | 62.0 | 52.1 | 43.5 | 3.27 |
| UniVLA | 95.5 | 85.8 | 75.4 | 66.9 | 56.5 | 3.80 |
| UnifiedVLA | 98.9 | 94.8 | 89.0 | 82.8 | 75.1 | 4.41 |
| OpenVLA-OFT | 96.9 | 92.0 | 85.7 | 80.4 | 72.9 | 4.28 |
| FLOWER | 99.4 | 95.8 | 90.7 | 84.9 | 77.8 | 4.53 |
| VLA-Adapter | 99.1 | 94.6 | 88.8 | 82.8 | 76.5 | 4.42 |
| Seer | 96.3 | 91.6 | 86.1 | 80.3 | 74.0 | 4.28 |
| **AVA-VLA (Ours)** | **99.6** | **97.6** | **94.1** | **89.9** | **84.1** | **4.65** |

**说明**: ABC→D 训练于 A/B/C、测试于未见环境 D，考验零样本泛化与序贯推理。AVA-VLA 在连续完成 1–5 个任务的成功率与平均长度（4.65）上全面领先，且连续任务越长（4、5 连）相对优势越大，呼应"循环状态保留时序信念"的主张。

### Table 3: Ablation on Backbones / 主干消融（LIBERO-Long）

| Backbone | OpenVLA-OFT | AVA-VLA |
|----------|-------------|---------|
| OpenVLA-7B（robot 预训练） | 94.5 | **96.2** (+1.7%) |
| LLaMA2-7B（无 robot 预训练） | 90.0 | **92.6** (+2.6%) |
| Qwen2.5-0.5B（无 robot 预训练） | 89.4 | **90.8** (+1.4%) |

**说明**: 无论主干是否做过机器人预训练、规模大小，AVA-VLA 都稳定提升，说明该框架是**主干无关**的即插即用增强。

### Table 4: Component Ablation on LIBERO / 两组件消融

| Method | Spatial | Object | Goal | Long | Avg. |
|--------|---------|--------|------|------|------|
| OpenVLA-OFT | 97.7 | 98.0 | 96.1 | 95.3 | 96.8 |
| AVA-VLA（仅 State-based init） | 97.2 | 98.8 | 96.6 | 97.2 | 97.5 |
| AVA-VLA（仅 AVA module） | 97.8 | 98.6 | 97.0 | 96.6 | 97.5 |
| **AVA-VLA（AVA + init）** | 97.4 | **99.4** | **97.4** | **97.6** | **98.0** |

**说明**: 状态化初始化与 AVA 模块各自都超过基线且**互补**：前者在 LIBERO-Long 上尤其有效（保留时序信念），后者跨套件普遍提升（抑制无关视觉），组合最优。

### Table 5: Visual Token Pruning / 视觉 token 剪枝鲁棒性

| Pruning Ratio | Spatial | Object | Goal | Long | Avg. |
|---------------|---------|--------|------|------|------|
| 0% | **97.4** | **99.4** | 97.4 | **97.6** | **98.0** |
| 50% | 97.2 | 99.4 | 97.2 | 95.2 | 97.3 |
| 60% | 97.6 | 99.4 | 97.0 | 95.0 | 97.3 |
| 70% | 97.4 | 99.2 | **98.0** | 94.6 | 97.3 |
| 80% | 96.8 | 98.2 | 96.2 | 92.8 | 96.0 |
| 90% | 94.2 | 97.8 | 94.2 | 89.2 | 93.9 |

**说明**: 按软权重排序保留 top-k 视觉 token。剪 50/60/70% 仍超过 OpenVLA-OFT 并保持 SOTA 级；即便剪 90% 也优于多数基线。下降主要来自 LIBERO-Long，印证软权重确实捕捉到了任务相关性，AVA 天然可作高效推理的剪枝信号。

### Table 6: Robustness on LIBERO+ / 七类扰动鲁棒性（附录）

**One policy for all 4 suites**

| Method | Camera | Robot | Language | Light | Background | Noise | Layout | Average |
|--------|--------|-------|----------|-------|------------|-------|--------|---------|
| WorldVLA | 0.1 | 27.9 | 41.6 | 43.7 | 17.1 | 10.9 | 38.0 | 25.0 |
| $\pi_0$ | 13.8 | 6.0 | 58.8 | 85.0 | 81.4 | 79.0 | 68.9 | 53.6 |
| $\pi_0$-FAST | **65.1** | 21.6 | 61.0 | 73.2 | 73.2 | 74.4 | 68.8 | 61.6 |
| OpenVLA-OFT | 55.6 | 21.7 | 81.0 | 92.7 | **91.0** | **78.6** | 68.7 | 67.9 |
| **AVA-VLA (Ours)** | 55.5 | **25.9** | **85.6** | **95.5** | 88.9 | 78.0 | **74.1** | **70.1** |

**One policy per suite**

| Method | Camera | Robot | Language | Light | Background | Noise | Layout | Average |
|--------|--------|-------|----------|-------|------------|-------|--------|---------|
| OpenVLA | 0.8 | 3.5 | 23.0 | 8.1 | 34.8 | 15.2 | 28.5 | 15.6 |
| NORA | 2.2 | 37.0 | 65.1 | 45.7 | 58.6 | 12.8 | 62.1 | 39.0 |
| UniVLA | 1.8 | 46.2 | 69.6 | 69.0 | 81.0 | 21.2 | 31.9 | 42.9 |
| OpenVLA-OFT | 56.4 | 31.9 | 79.5 | 88.7 | 93.3 | 75.8 | 74.2 | 69.6 |
| RIPT-VLA | 55.2 | 31.2 | 77.6 | 88.4 | 91.6 | 73.5 | 74.2 | 68.4 |
| **AVA-VLA (Ours)** | **69.4** | **34.9** | **81.5** | **97.5** | **94.1** | **79.1** | **78.3** | **74.7** |

**说明**: 不额外训练、直接用 LIBERO 模型在 LIBERO+ 七类扰动下评测。AVA-VLA 两设置总平均均最高（70.1% / 74.7%），在 Light、Layout 扰动下尤其稳健，说明 AVA 增强重要视觉信息、削弱干扰部分确实带来视觉鲁棒性。

### Table 7: Matched Training Settings / 同等训练预算对照（附录）

| Method | Spatial | Object | Goal | Long | Avg. |
|--------|---------|--------|------|------|------|
| OpenVLA-OFT | 97.0 | 98.8 | 96.0 | 95.2 | 96.8 |
| **AVA-VLA** | **98.4** | **99.4** | **98.4** | **96.8** | **98.3** |

**说明**: 两者从同一 OpenVLA checkpoint、同等 batch（256）与步数（100K）训练。AVA-VLA 仍稳超基线（且优于 Table 1），排除"额外训练算力"的混淆，证明增益来自循环状态初始化 + 主动视觉注意力的架构协同（仅 <50M / <1% 参数开销）。

### Table 8: Loss Design Ablation ($L_{\omega}$) / 正则项消融（附录）

| Method | Spatial | Object | Goal | Long | Avg. |
|--------|---------|--------|------|------|------|
| **AVA-VLA** | 97.4 | **99.4** | **97.4** | **97.6** | **98.0** |
| AVA-VLA w/o $L_\omega$ | 97.4 | 98.8 | 97.2 | 96.4 | 97.5 |

**说明**: 去掉软权重 L2 正则，平均从 98.0% 降至 97.5%，LIBERO-Long 下降最明显，且注意力变弥散（见 Fig 12）。说明 $L_\omega$ 对维持稀疏、结构化的注意力很关键。

### Table 9: Module Ablation on CALVIN / CALVIN 组件消融（附录）

| Method | 1 | 2 | 3 | 4 | 5 | Avg. len ↑ |
|--------|------|------|------|------|------|------|
| OpenVLA-OFT | 96.9 | 92.0 | 85.7 | 80.4 | 72.9 | 4.28 |
| +init | 99.5 | 96.9 | 93.4 | **90.0** | 83.6 | 4.63 |
| +ava | 99.1 | 96.5 | 93.1 | 89.2 | 82.7 | 4.61 |
| **AVA-VLA** | **99.6** | **97.6** | **94.1** | 89.9 | **84.1** | **4.65** |

**说明**: 在 CALVIN 上复现 Table 4 结论：两组件各自提升、组合最优，且任务越长（连续 3/4/5）增益越明显，支持"状态化初始化保留时序信念、AVA 精炼感知、二者互补尤其利于长程"的论断。

---

## 实验

### 数据集 / 基准

| 基准 | 规模 | 特点 | 用途 |
|------|------|------|------|
| [[LIBERO]] | 100 任务 / 5000 episode，4 套件（Spatial/Object/Goal/Long） | Franka Panda + MuJoCo，终身机器人学习 | 训练/测试 |
| [[LIBERO+]] | 7 扰动维度 / 21 子维度 | 系统化鲁棒性评测（相机/机器人/语言/光照/背景/噪声/布局） | 仅测试（不额外训练） |
| [[CALVIN]] ABC→D | 34 任务 / 20000+ episode，4 环境 | 语言条件长程操作，A/B/C 训练 → D 测试（零样本） | 训练/测试 |
| Mobile ALOHA 真机 | 4 任务，每任务 30–450 示范 | AgileX Cobot Magic 双臂 + 1 第三人称 + 2 腕部相机 | 训练/测试 |

### 实现细节

- **基础模型**: [[OpenVLA-OFT]]（[[SigLIP]]-[[DINOv2]] 双分支视觉编码器、[[LLaMA2]]-7B、3 层 MLP 投影器、双向注意力并行解码）
- **新增组件**: 2 层 SiLU MLP 出循环状态；三个 2 层 MLP 把视觉/指令/循环状态降到 $d'$；[[FiLM]]；cross-attn + self-attn + FFN + 线性 Softmax；占位嵌入替换；注意力矩阵软调制 —— 共 <50M 参数
- **训练**: [[LoRA]]（rank 32）微调主干/视觉/动作头/本体投影，AVA 全量训练；$T/K=4$（2、3 步间 detach 梯度）；$\lambda=1.0$、$c=0.6$（CALVIN 0.2）、$\boldsymbol{\gamma}=[1.9,0.1]$；动作块 $\mathrm{L}_c=8$（真机 25）；batch 64；40k 步（CALVIN 同配置、真机 20k）；lr 5e-4 + 10% warmup + cosine；grad clip 1.0
- **初始化**: 从已训练好的 OpenVLA-OFT checkpoint 出发（为循环状态提供更好初值，非靠延长训练取胜，见 Table 7 对照）
- **硬件**: Nvidia A800 GPU

### 关键实验结论

- **仿真主结果**: LIBERO 单/多任务设置均 SOTA（98.2% / 98.0%），CALVIN ABC→D avg.len 4.65（最高），长程任务优势最大。
- **真机**: Mobile ALOHA 四任务跨任务平均最高，验证 sim-to-real 与小样本适配。
- **鲁棒性**: LIBERO+ 七类扰动两设置总平均最高（70.1% / 74.7%），光照/布局尤稳。
- **消融**: 主干无关（Table 3）；状态化初始化与 AVA 互补（Table 4/9）；$L_\omega$ 关键（Table 8/Fig 12）；同预算对照排除算力混淆（Table 7）。
- **效率副产物**: 软权重可直接用于视觉 token 剪枝，剪 70% 仍 SOTA 级（Table 5）。

---

## 批判性思考

### 优点
1. **理论框架清晰且落地**: 用 [[POMDP]]/信念状态这一成熟视角统一解释 VLA 的"历史缺失"问题，并把抽象的信念状态落成可计算的循环状态，动机—公式—实现—可视化链条完整。
2. **轻量即插即用、可迁移**: 仅 <50M（<1%）参数增量即在三类基准 + 真机普遍提升，且对不同规模/是否预训练的主干都生效（Table 3），实用价值高。
3. **证据扎实、反混淆**: 不仅给主结果，还做了同训练预算对照（Table 7）排除"训练更久"的质疑，配合注意力可视化（Fig 1/4/7–12）与去正则对照（Fig 12），主张较为可信。
4. **意外的效率收益**: 软权重天然可做 token 剪枝信号，剪 70% 仍 SOTA 级，给高频部署留了空间。

### 局限性
1. **信念漂移**: 作者自陈的核心短板 —— 小的感知/状态估计误差会在循环状态中沿长程累积，导致 LIBERO-Long 等精细任务失败（Fig 13），且剪枝时长程任务掉得最多。
2. **绝对数值接近饱和、增益偏小**: LIBERO 多基线已 95%+，AVA-VLA 提升约 1–2 个百分点，区分度有限；CALVIN 与真机更能体现差异，但真机仅与 2 个基线比较。
3. **超参偏经验**: $c$、$\boldsymbol{\gamma}=[1.9,0.1]$、$T=4$、detach 位置等关键设置基本靠经验给定，缺敏感度分析；"取最后一层隐状态做循环状态"也未与其他层系统对比。
4. **截断窗口很短**: $T=4$（且中间 detach）只建模了很短的时序依赖，对真正长程（数百步）的信念建模能力存疑，与"POMDP 全历史"的理论叙事存在落差。

### 潜在改进方向
1. 引入更鲁棒的状态更新/显式误差校正或更长训练窗口，缓解信念漂移（作者亦指出）。
2. 把 $c/\boldsymbol{\gamma}/T$ 等做成可学习或自动搜索，并对"用哪层隐状态"做消融。
3. 把 AVA 的软权重与显式剪枝/调度（如 SP-VLA）联合优化，做端到端的高效主动感知。
4. 与显式记忆库方法（MemoryVLA）做正面对比，量化"隐式循环状态 vs 显式记忆"的权衡。

### 可复现性评估
- [x] 代码开源（项目主页 https://liauto-dsr.github.io/AVA-VLA-Page/ 提供，基于公开 OpenVLA-OFT）
- [ ] 预训练模型（未明确声明 release 权重）
- [x] 训练细节完整（附录 A 给出全部超参与配置）
- [x] 数据集可获取（LIBERO/LIBERO+/CALVIN 公开；真机数据为自采）

---

## 速查卡片

> [!summary] AVA-VLA: Improving VLA with Active Visual Attention
> - **核心**: 把 VLA 从逐帧 MDP 重述为 POMDP，用循环状态近似信念状态，并据此主动重加权当前帧视觉 token。
> - **方法**: 取上一时刻最后层隐状态经 MLP 得循环状态 $\boldsymbol{r}^{t-1}$ → 既做动作占位初始化、又作 AVA 的 K/V；AVA（FiLM + cross/self-attn + Softmax）算软权重 $\boldsymbol{\omega}^t$ 调制 LLM 全层注意力；truncated BPTT（$T=4$）+ MAE + $L_\omega$ 正则训练。仅 <50M 参数加在 OpenVLA-OFT 上。
> - **结果**: LIBERO 98.0/98.2%、CALVIN ABC→D avg.len 4.65、LIBERO+ 鲁棒性最高、真机 Mobile ALOHA 平均最优；软权重可剪 70% token 仍 SOTA 级。
> - **代码**: https://liauto-dsr.github.io/AVA-VLA-Page/

---

*笔记创建时间: 2026-06-29*
