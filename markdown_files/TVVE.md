---
title: "Learning to See and Act: Task-Aware Virtual View Exploration for Robotic Manipulation"
method_name: "TVVE"
authors: [Yongjie Bai, Zhouxia Wang, Yang Liu, Kaijun Luo, Yifan Wen, Mingtong Dai, Weixing Chen, Ziliang Chen, Lingbo Liu, Guanbin Li, Liang Lin]
year: 2026
venue: CVPR
tags: [VLA, robotic-manipulation, view-exploration, mixture-of-experts, reinforcement-learning, RLBench, multi-task]
zotero_collection: _inbox
image_source: online
arxiv_html: https://arxiv.org/html/2508.05186v5
created: 2026-06-29
---

# Learning to See and Act: Task-Aware Virtual View Exploration for Robotic Manipulation

## 元信息

| 项目 | 内容 |
|------|------|
| 作者 | Yongjie Bai, Zhouxia Wang, Yang Liu, Kaijun Luo, Yifan Wen, Mingtong Dai, Weixing Chen, Ziliang Chen, Lingbo Liu, Guanbin Li, Liang Lin |
| 机构 | 中山大学（HCP-Lab，Sun Yat-sen University）、鹏城实验室等 |
| 会议 | CVPR 2026 |
| 类别 | 机器人操作 / VLA / 多任务学习 |
| 日期 | 2025-08（arXiv v5） |
| 项目主页 | https://hcplab-sysu.github.io/TAVP/ |
| 链接 | [arXiv](https://arxiv.org/abs/2508.05186) / [Project](https://hcplab-sysu.github.io/TAVP/) |

> 说明：论文方法名为 **TVVE（Task-aware Virtual View Exploration）**，项目主页缩写为 TAVP。本文统一以 TVVE 指代该方法。

---

## 一句话总结

> 用强化学习训练的**多视角探索策略 (MVEP)** 从重建点云中动态选取“看得最清”的虚拟相机视角并重渲染观测，再配合**任务感知 MoE 视觉编码器 (TaskMoE)** 缓解多任务干扰，让机器人“先学会看、再学会动”，在 RLBench 及其 OOD 变体上显著超过固定视角基线。

---

## 核心贡献

1. **多视角探索策略 MVEP**：让机器人在重建的全局点云上**动态探索更优的虚拟观测视角**（动态多视角重渲染），有效缓解固定视角下的遮挡与视角不足问题，增强 3D 感知能力。
2. **任务感知混合专家 TaskMoE**：根据任务、指令与场景视觉信息动态选择感知与动作生成专家，缓解共享编码器在多任务下的**特征冲突**，提升多任务处理能力与泛化性。
3. **RLBench-OG 基准**：构建一个全新的**分布外 (OOD) 评测基准**，系统覆盖遮挡、背景/桌面纹理与颜色变化、光照变化、干扰物、相机位姿变化等扰动，用于衡量模型鲁棒性与泛化性。
4. **综合性能领先**：在 RLBench、RLBench-OG 与真实机器人任务上，TVVE 在准确率与鲁棒性上均显著优于现有强基线。

---

## 问题背景

### 要解决的问题
当前面向多任务机器人操作的 [[VLA]] 模型普遍依赖**固定相机布置**与**共享视觉编码器**。这带来两个本质缺陷：(1) 固定视角在杂乱/动态场景中频繁遭遇**遮挡**，导致对目标物体或末端执行器 (end-effector) 的观测不完整，进而动作预测失败；(2) 共享编码器在视觉与语义差异巨大的任务间产生**任务干扰 (task interference)**，限制了泛化与可扩展性。

### 现有方法的局限
- **固定视角**：如 Figure 1 所示，对指令 "Put the sugar in the cupboard"，三个固定相机要么拍不到 cupboard、要么拍不到已抓取 sugar 的末端执行器，造成感知盲区。
- **共享编码器多任务方案**：[[RVT]]、[[RVT-2]] 等通过关注指令 token 实现多任务，但共享编码器在 "pick an apple" 与 "open a drawer" 这类语义/视觉差异大的任务间纠缠严重。
- **模块化方案**（如 ManipGen 训练 1000+ 任务专家）接口设计成本随任务复杂度指数爆炸；**端到端单一密集模型**则因任务间特征分布与运动模式差异巨大而出现参数冲突、收敛困难。

### 本文的动机
作者主张机器人应当**先学会“看” (See) 再学会“动” (Act)**：与其被动接受固定视角，不如主动探索能最大化目标相关信息覆盖的虚拟视角。为此用 [[Reinforcement Learning|强化学习]] 训练一个视角探索策略，并用任务感知 [[Mixture-of-Experts|MoE]] 编码器抽取任务特定特征，使“看”与“动”相互对齐——动态“seeing”是鲁棒“acting”的基础。

---

## 方法详解

### 模型架构

TVVE 采用 **“感知重建 → 任务感知特征 → 视角探索 → 重渲染 → 动作预测”** 的流水线（见 Figure 2）：

- **输入**：多路固定视角 RGB-D 图像 + 语言指令 + 当前夹爪状态。
- **点云重建**：用相机内外参把各视角深度图反投影并聚合为世界坐标系下的**全局点云**。
- **Coarse Grounding（橙色分支）**：沿用 [[RVT-2]] 的粗定位阶段预测末端执行器的大致位置，据此把全局点云中心移到该位置并缩放裁剪，保留关键点云区域（缩小搜索空间）。
- **TaskMoE-MVT**：在 RVT-2 的 [[Multi-View Transformer|MVT]] 中嵌入 [[TaskMoE]]，把指令路由到专门的专家编码器，得到任务对齐的视觉特征（同时服务于视角选择与动作预测）。
- **MVEP**：[[Multi-Viewpoint Exploration Policy|多视角探索策略]] 在感兴趣区域内搜索 $K$ 个能最大化目标物体与末端执行器可见性的相机位姿。
- **重渲染 + 动作预测**：从探索得到的视角重渲染 2D 观测，经另一组 TaskMoE-MVT 后送入动作预测模型；动作策略基于 [[ARP]]（自回归动作策略）并同样升级了 TaskMoE。
- **输出**：末端执行器位姿动作 $\mathbf{a}=(\mathbf{p}_{\text{xyz}},\mathbf{q}_{\text{rot}},g_{\text{gripper}},c_{\text{collision}})\in\mathbb{R}^{7}\times\{0,1\}^{2}$。

### 核心模块

#### 模块1: TaskMoE（任务感知混合专家）

**设计动机**：不同操作任务需要差异巨大的视觉表征与动作策略，单一共享编码器会产生特征冲突。TaskMoE 把视觉 token 路由到任务专门化的专家，缓解多任务干扰。相比已有 MoE 方法（如 SDP），它引入两点关键创新。

**具体实现**:
1. **更丰富的路由条件（而非仅任务 ID）**：用 [[Cross-Attention|交叉注意力]] 让指令信息注入视觉，得到上下文感知特征，再通过 [[FiLM|Feature-wise Linear Modulation]] 层与任务嵌入融合，实现更自适应、任务敏感的专家选择。
2. **门与任务、门与专家的解耦（两阶段路由）**：为 $N_J$ 个任务分配 $N_G$ 个门（$N_G < N_J$），语义相似任务（如 Task 1、Task 2 都涉及开抽屉）共享同一门但路由到不同专家，语义不同任务（Task 3）走不同门。所有任务共享一个 $N_E$ 专家池，每个 token 只激活 top-$k$ 专家。
3. 实现中默认 $N_G=8$、$N_E=16$、top-2 专家，并辅以**负载均衡 / 熵正则 / 语义正则**与容量裁剪避免专家坍缩。

#### 模块2: MVEP（多视角探索策略）

**设计动机**：动态探索能减少遮挡、提供更多信息的相机视角，从而服务更准的动作预测。

**具体实现**:
- 输入为重建点云 $\mathcal{P}\in\mathbb{R}^{N\times 3}$ 及其 RGB 特征 $\mathbf{F}_{\text{img}}\in\mathbb{R}^{N\times 3}$，拼接后经 MLP 预测 $K$ 个相机位姿。
- 每个相机位姿用 **look-at 模型** 表示为 5 维向量 $\mathbf{p}^{i}=(\theta^{i},\phi^{i},r^{i},\theta_{\text{up}}^{i},\phi_{\text{up}}^{i})$，球坐标解耦位置与朝向，相机始终朝向原点 $O(0,0,0)$。
- 为支持梯度优化，把每个位姿建模为对角 [[Gaussian Distribution|高斯分布]] 的采样，输出均值与对数标准差，并用**重参数化技巧**采样、用 sigmoid 把各分量约束到合法范围。
- 用 [[PPO]] 在**伪环境 (pseudo-environment)** 中优化，避免与物理环境交互的高昂时间成本。

#### 模块3: 三阶段训练策略

**Stage 1（固定视角预训练）**：用 front/left/top 三个默认视角训练 TVVE 的固定视角变体，沿用 RVT-2 的训练流程，总损失见公式 5。

**Stage 2（强化学习训练 MVEP）**：以 Stage 1 的固定视角模型作为参考模型（性能下界），用 PPO + 伪环境交互训练 MVEP；目标是让动态视角下的损失 $\mathcal{L}_{\text{TVVE}}$ 低于参考损失 $\mathcal{L}_{\text{ref}}$。奖励由任务损失奖励、置信度奖励、视角多样性奖励三项自适应归一化聚合（公式 6–9）。此阶段仅 MVEP 可训练，其余冻结。

**Stage 3（联合微调）**：固定 MVEP，用与 Stage 1 相同的损失对 TVVE 其余部分整体微调，让“看”与“动”更协调。

### 关键公式与机制

#### 公式1: [[Point Cloud|点云]] 与 RGB 特征拼接（MVEP 输入）

$$
\mathbf{X}=\text{Concat}(\mathcal{P},\mathbf{F}_{\text{img}})\in\mathbb{R}^{N\times 6}
$$

**含义**: 把 3D 坐标点云与对应 RGB 特征拼接，构成 MVEP 的统一输入表征，经 MLP 预测 $K$ 个相机位姿。

**符号说明**:
- $\mathcal{P}\in\mathbb{R}^{N\times 3}$: 重建点云；$\mathbf{F}_{\text{img}}\in\mathbb{R}^{N\times 3}$: 对应 RGB 特征
- $N$: 3D 点数

#### 公式2: 相机位姿的高斯分布参数

$$
\begin{aligned}
\mu^{i} &= [\mu_{\theta}^{i},\mu_{\phi}^{i},\mu_{r}^{i},\mu_{\theta_{\text{up}}}^{i},\mu_{\phi_{\text{up}}}^{i}] \\
\log\sigma^{i} &= [\log\sigma_{\theta}^{i},\log\sigma_{\phi}^{i},\log\sigma_{r}^{i},\log\sigma_{\theta_{\text{up}}}^{i},\log\sigma_{\phi_{\text{up}}}^{i}]
\end{aligned}
$$

**含义**: MVEP 对每个视角输出 5 维对角高斯分布的均值 $\mu^i$ 与对数标准差 $\log\sigma^i$，把视角选择变成可采样、可微的概率策略。

#### 公式3: 重参数化采样

$$
\tilde{\mathbf{p}}^{i}=\mu^{i}+\sigma^{i}\odot\epsilon^{i},\quad\epsilon^{i}\sim\mathcal{N}(\mathbf{0},\mathbf{I})
$$

**含义**: 用重参数化技巧采样相机位姿，使梯度可经反向传播端到端训练策略。

**符号说明**:
- $\odot$: 逐元素乘；$\epsilon^i$: 标准正态噪声

#### 公式4: 位姿分量的范围约束

$$
\begin{aligned}
\tilde{\theta}^{i} &= \pi\cdot\sigma(\tilde{\theta}^{i}),\quad \tilde{\phi}^{i}=2\pi\cdot\sigma(\tilde{\phi}^{i}) \\
\tilde{r}^{i} &= r_{\min}+(r_{\max}-r_{\min})\cdot\sigma(\tilde{r}^{i}) \\
\tilde{\theta}_{\text{up}}^{i} &= \pi\cdot\sigma(\tilde{\theta}_{\text{up}}^{i}),\quad \tilde{\phi}_{\text{up}}^{i}=2\pi\cdot\sigma(\tilde{\phi}_{\text{up}}^{i})
\end{aligned}
$$

**含义**: 用 sigmoid $\sigma(\cdot)$ 把采样得到的各位姿分量约束到球坐标的合法范围，半径 $r$ 限制在 $[r_{\min}, r_{\max}]$。

#### 公式5: Stage 1 总损失

$$
\mathcal{L}_{\text{s1}}=\mathcal{L}_{hc}+\mathcal{L}_{hf}+\mathcal{L}_{rot}+\mathcal{L}_{gri}+\mathcal{L}_{col}
$$

**含义**: 固定视角预训练的总目标，沿用 RVT/RVT-2 的多项损失。

**符号说明**:
- $\mathcal{L}_{hc},\mathcal{L}_{hf}$: 粗/细定位模块在热力图上的交叉熵损失（真值热力图由 GT 3D 位置的 2D 投影处的截断高斯生成）
- $\mathcal{L}_{rot}$: 末端执行器旋转损失（对每个欧拉角的交叉熵）
- $\mathcal{L}_{gri},\mathcal{L}_{col}$: 夹爪状态损失与碰撞指示损失（均为二分类）

#### 公式6: 任务损失奖励

$$
r_{0}=\mathcal{L}_{\text{ref}}-\mathcal{L}_{\text{TVVE}}
$$

**含义**: 以固定视角参考模型损失 $\mathcal{L}_{\text{ref}}$ 为下界，动态视角损失 $\mathcal{L}_{\text{TVVE}}$ 越小奖励越高，激励 MVEP 找到比固定视角更优的观测。

#### 公式7: 置信度奖励（细定位熵）

$$
r_{1}=-\frac{1}{K}\sum_{i=1}^{K}\mathcal{H}(\text{softmax}(\mathbf{H}_{i}))
$$

**含义**: 取细定位热力图的负平均熵作为奖励，鼓励 MVEP 选出让定位更确定（更尖锐）的视角。

**符号说明**:
- $\mathcal{H}$: 熵函数；$\mathbf{H}_{i}$: 第 $i$ 个视角的细定位热力图

#### 公式8: 视角多样性奖励

$$
r_{2}=\frac{1}{K(K-1)}\sum_{i\neq j}\left(1-\cos(\mathbf{p}_{i},\mathbf{p}_{j})\right)
$$

**含义**: 用相机位置间的平均成对余弦距离作奖励，鼓励 $K$ 个视角彼此分散、覆盖更多信息，避免视角冗余。

**符号说明**:
- $\mathbf{p}_i$: 第 $i$ 个视角的相机位置

#### 公式9: 总奖励聚合

$$
r=\sum_{i=0}^{2}w_{i}\cdot\mathcal{N}(r_{i})
$$

**含义**: 自适应归一化并加权聚合三项奖励。$\mathcal{N}$ 用 Welford 算法在线维护各分量的均值/方差进行归一化，并将奖励裁剪到 $[-10,10]$ 保证训练稳定。

**符号说明**:
- $w_i$: 可学习的分量权重；$\mathcal{N}$: 在线归一化算子

#### 公式10–11: TaskMoE 跨模态交叉注意力（附录 B）

$$
\begin{aligned}
\mathbf{Q} &= \mathbf{V}\mathbf{W}^{Q},\quad \mathbf{K}=\mathbf{I}\mathbf{W}^{K},\quad \mathbf{V}_{a}=\mathbf{I}\mathbf{W}^{V} \\
\mathbf{V}^{\prime} &= \mathrm{softmax}\!\left(\frac{\mathbf{Q}\mathbf{K}^{\top}}{\sqrt{d_{k}}}\right)\mathbf{V}_{a}
\end{aligned}
$$

**含义**: 以视觉 token $\mathbf{V}\in\mathbb{R}^{S\times D}$ 为 query、指令嵌入 $\mathbf{I}$ 为 key/value 做交叉注意力，得到指令注意后的视觉特征 $\mathbf{V}^{\prime}$，把语言信息注入视觉以指导专家路由。

**符号说明**:
- $S$: 空间 token 数；$D$: 特征维；$d_k$: key 维度；$\mathbf{W}^{Q},\mathbf{W}^{K},\mathbf{W}^{V}$: 投影矩阵

#### 公式12–14: FiLM 调制

$$
\begin{aligned}
\mathbf{c} &= [\mathrm{GAP}(\mathbf{V}^{\prime});\,\mathbf{t}]\in\mathbb{R}^{2D} \\
\gamma,\beta &= \mathrm{MLP}(\mathbf{c})\in\mathbb{R}^{D}\times\mathbb{R}^{D} \\
\mathbf{V}_{\text{mod}} &= \gamma\odot\mathbf{V}+\beta
\end{aligned}
$$

**含义**: 把全局池化后的视觉特征与任务嵌入 $\mathbf{t}$ 拼接成上下文 $\mathbf{c}$，由 MLP 生成逐通道缩放 $\gamma$ 与偏置 $\beta$，对原视觉特征做 [[FiLM]] 调制，得到送入门控网络的 $\mathbf{V}_{\text{mod}}$。

**符号说明**:
- $\mathrm{GAP}(\cdot)$: 对 $S$ 个 token 的全局平均池化；$\mathbf{t}\in\mathbb{R}^D$: 任务可学习嵌入

#### 公式15: 任务感知门控（Stage 1 路由）

$$
\mathbf{p}_{g}=\mathrm{softmax}(\mathbf{V}_{\text{mod}}\mathbf{W}_{g})\odot\mathrm{softmax}(\mathbf{A}[j,:])
$$

**含义**: 共享门头对每个 token 打分得到 $N_G$ 个门的分布，再被可学习的**任务–门分配矩阵** $\mathbf{A}$ 第 $j$ 行（当前任务）调制，实现任务感知的门选择（取 top-1 门，分析时保留 top-2）。

**符号说明**:
- $\mathbf{W}_{g}\in\mathbb{R}^{D\times N_G}$: 门投影；$\mathbf{A}\in\mathbb{R}^{N_J\times N_G}$: 任务–门分配矩阵；$j$: 当前任务索引

#### 公式16: 按门选专家（Stage 2 路由）

$$
\mathbf{z}_{e}=\mathbf{V}_{\text{mod}}\mathbf{W}_{\text{exp}}[g]+\mathbf{b}_{\text{exp}}[g],\quad \mathbf{p}_{e}=\mathrm{softmax}(\mathbf{z}_{e})
$$

**含义**: 给定所选门 $g$，用该门专属的专家头从 $N_E$ 专家池中给每个 token 计算专家分布 $\mathbf{p}_e$，保留 top-$k$（默认 $k=2$）专家并施加容量约束。

**符号说明**:
- $\mathbf{W}_{\text{exp}}\in\mathbb{R}^{D\times N_G\times N_E}$、$\mathbf{b}_{\text{exp}}\in\mathbb{R}^{N_G\times N_E}$: 门–专家投影与偏置

#### 公式17–18: TaskMoE 总损失（含正则）

$$
\begin{aligned}
\mathcal{L} = \mathcal{L}_{\text{task}} &+ \lambda_{g}\mathcal{L}_{\text{bal-gate}}+\lambda_{e}\mathcal{L}_{\text{bal-exp}} \\
&- \eta_{g}H(\mathbf{p}_{g})-\eta_{e}H(\mathbf{p}_{e})+\lambda_{\text{sem}}\,\mathcal{L}_{\text{sem}}(\mathbf{A},\mathbf{I})
\end{aligned}
$$

**含义**: 在模仿学习主损失基础上加门/专家负载均衡、门/专家分布熵（鼓励多样路由）、以及语义正则——用指令嵌入相似度结构对齐任务–门矩阵 $\mathbf{A}$ 的行，使语义相关任务共享门/专家。

**符号说明**:
- $\mathcal{L}_{\text{bal-gate}},\mathcal{L}_{\text{bal-exp}}$: 门/专家负载均衡损失
- $H(\mathbf{p}_g),H(\mathbf{p}_e)$: 门/专家分布的熵
- $\mathcal{L}_{\text{sem}}(\mathbf{A},\mathbf{I})$: 语义对齐损失；$\lambda_g,\lambda_e,\eta_g,\eta_e,\lambda_{\text{sem}}\ge0$: 各正则权重

#### 公式19–20: 相机位置与上向量（look-at 参数化，附录 B）

$$
\mathbf{t}^{i}=\begin{bmatrix}x^{i}\\ y^{i}\\ z^{i}\end{bmatrix}=\begin{bmatrix}r^{i}\sin\theta^{i}\cos\phi^{i}\\ r^{i}\sin\theta^{i}\sin\phi^{i}\\ r^{i}\cos\theta^{i}\end{bmatrix}
$$

$$
\mathbf{v}_{\text{up}}^{i}=\begin{bmatrix}\sin\theta_{\text{up}}^{i}\cos\phi_{\text{up}}^{i}\\ \sin\theta_{\text{up}}^{i}\sin\phi_{\text{up}}^{i}\\ \cos\theta_{\text{up}}^{i}\end{bmatrix}
$$

**含义**: 由球坐标计算相机中心 $\mathbf{t}^i$ 与上向量 $\mathbf{v}_{\text{up}}^i$；视线方向 $\mathbf{v}_{\text{look}}^{i}=-\mathbf{t}^{i}/\|\mathbf{t}^{i}\|_2$ 指向原点，最终相机矩阵由 $\mathbf{v}_{\text{look}}^i$ 与 $\mathbf{v}_{\text{up}}^i$ 经 Gram-Schmidt 正交化构造。

---

## 关键图表

<!-- 图片均使用 arXiv HTML 在线链接，未本地化 -->

### Figure 1: Motivation Illustration / 动机示意

![Figure 1](https://arxiv.org/html/2508.05186v5/x1.png)

**说明**: 固定相机常拍不全目标。对 "Put the sugar in the cupboard"，front view 只拍到 cupboard（红圈），left/right shoulder view 只拍到已被夹爪抓取的 sugar（绿圈）。这种不完整观测会导致操作失败；TVVE 则动态探索并重渲染最大化目标相关信息覆盖的视角。这张图直接立住了“固定视角→遮挡→失败”的痛点。

### Figure 2: TVVE Framework Overview / 整体框架

![Figure 2](https://arxiv.org/html/2508.05186v5/x2.png)

**说明**: TVVE 总览。多路 RGB-D 先转点云并聚合为世界坐标全局点云，随后分两支：橙色支做 Coarse Grounding 预测末端执行器近似位置并据此移中心/缩放/裁剪点云；另一支用 MVEP 探索更优相机位姿、重渲染观测，再经 TaskMoE-MVT 与 TaskMoE-ARP 完成动作预测。是理解“看→动”流水线的核心图。

### Figure 3: Pipeline of TaskMoE / TaskMoE 流程

![Figure 3](https://arxiv.org/html/2508.05186v5/Fig/Fig3.jpg)

**说明**: TaskMoE 以 Task ID、指令、视觉为输入指导专家选择。设计紧凑门控机制，$N_G$ 个门被 $N_J$ 个任务共享（$N_G<N_J$）：动作模式相似的 Task 1、Task 2 共享门，语义差异大的 Task 3 走不同门。展示了“两阶段路由（门→专家）”如何实现跨任务的特征专门化。

### Figure 4: Qualitative Visualization (Sim & Real) / 仿真与真实定性结果

![Figure 4](https://arxiv.org/html/2508.05186v5/x3.png)

**说明**: (a) RLBench 中 Close Jar、Insert Peg 任务的动态多视角重渲染结果（EEF 为末端执行器）；(b) Insert Peg 样例；(c) 真实世界 Pick Grape 样例。直观佐证 TVVE 相比固定视角的 ARP 能渲染更具信息量的视角、缓解遮挡与碰撞。

### Figure 5: Real-World Environment Setup / 真实环境布置（附录）

![Figure 5](https://arxiv.org/html/2508.05186v5/x4.png)

**说明**: 真实部署环境与各视角图像。Dobot Nova 2 用三相机（侧/前/腕），Franka Research 3 仅用单前视，验证 TVVE 在不同相机配置与平台上的可部署性。

### Figure 6: Dobot & Franka Execution Processes / 真实执行过程（附录）

![Figure 6](https://arxiv.org/html/2508.05186v5/x5.png)

**说明**: Dobot 与 Franka 在真实环境中的执行时序，T 轴为时间。每个 demo 给出 Dobot 的侧视+腕视、Franka 的单前视，展示任务从开始到完成的全过程。

### Figure 7: Real-World Tasks Setup / 真实任务布置（附录）

![Figure 7](https://arxiv.org/html/2508.05186v5/x6.png)

**说明**: 真实任务集合（Pick&Place、Stack Bowls、Push Buttons、Collect Fruits、Put Item In Drawer 及 Rotate Handle、Fold Towel、Open Lip、Unscrew Bottle、Reach Drag 等扩展任务），覆盖抓放、按压、关节体、旋转、可变形物体、工具使用等多种操作类型。

### Figure 8: Aggregated Full-Scene Point Cloud / 聚合全场景点云（附录）

![Figure 8](https://arxiv.org/html/2508.05186v5/Fig/Fig-app-pcd.png)

**说明**: put_groceries_in_cupboard 任务从多视角聚合得到的全场景点云。这是 MVEP 进行虚拟视角探索与重渲染的几何基础。

### Figure 9: 18 RLBench Tasks & Instructions / 18 个 RLBench 任务（附录）

![Figure 9](https://arxiv.org/html/2508.05186v5/x7.png)

**说明**: multi-view setup 下 18 个 RLBench 任务的示范样例与对应语言指令。

### Figure 10: t-SNE of Instruction Embeddings (4/8/16-Expert) / 指令嵌入 t-SNE（附录）

![Figure 10](https://arxiv.org/html/2508.05186v5/x8.png)

**说明**: 4 专家 (a)/8 专家 (b)/16 专家 (c) TaskMoE 路由后的指令嵌入 t-SNE。16 专家时语义解耦最佳：簇更多、更紧致、专家更专门化、负载更均衡；8 专家时外环被 E1/E4/E7 当“兜底专家”，存在部分路由坍缩；4 专家粒度过粗。

### Figure 11: Task-wise Gate/Expert Usage / 门-专家逐任务使用（附录）

![Figure 11](https://arxiv.org/html/2508.05186v5/x9.png)

**说明**: 8 专家 (a,b) 与 16 专家 (c,d) 的逐任务 top-1 门/专家使用。两种配置都呈现清晰的任务相关路由；更大专家容量能在不坍缩的前提下实现更细的专门化（如 place_cups 与 place_wine_at_rack_location 共享门 G2 却下分到不同专家 E5/E4），验证“门捕捉粗任务亲和、专家细化路由”的两阶段设计。

### Figure 12–13: Variant Visualizations / OOD 变体可视化（附录）

![Figure 12](https://arxiv.org/html/2508.05186v5/x10.png)
![Figure 13](https://arxiv.org/html/2508.05186v5/x11.png)

**说明**: RLBench-OG 中各任务（basketball_in_hoop、block_pyramid、close_drawer、scoop_with_spatula、solve_puzzle 及 straighten_rope、take_plate_off_colored_dish_rack、take_usb_out_of_computer、toilet_seat_down、water_plants）在遮挡与各类泛化扰动下的不同变体设置可视化。

### Figure 14–15: Dynamic Multi-View Re-rendering / 动态多视角重渲染（附录）

![Figure 14](https://arxiv.org/html/2508.05186v5/x12.png)
![Figure 15](https://arxiv.org/html/2508.05186v5/x13.png)

**说明**: MVEP 在推理中间步生成的动态虚拟视角（3D 空间）及其对应重渲染 2D 图像，覆盖 Light Bulb In、Close Jar、Insert Onto Square Peg、Put Groceries In Cupboard、Put Item In Drawer、Put Money In Safe 等 16 个任务。重渲染图像清晰捕捉末端执行器与目标物体，支撑“动态看 → 鲁棒动”的核心论点。

---

### Table 1: RLBench Multi-View Setup (18 tasks) / 多视角 18 任务成功率（%）

| Method | Avg. SR↑ | Avg. Rank↓ | Close Jar | Drag Stick | Insert Peg | Meat off Grill | Open Drawer | Place Cups | Place Wine | Push Buttons |
|--------|----------|------------|-----------|------------|------------|----------------|-------------|------------|------------|--------------|
| C2F-ARM-BC | 20.1 | 9.67 | 24.0 | 24.0 | 4.0 | 20.0 | 20.0 | 0.0 | 8.0 | 72.0 |
| PerAct | 49.4 | 7.06 | 55.2 | 89.6 | 5.6 | 70.4 | 88.0 | 2.4 | 44.8 | 92.8 |
| HiveFormer | 45.0 | 8.22 | 52.0 | 76.0 | 0.0 | 80.0 | 52.0 | 0.0 | 80.0 | 84.0 |
| PolarNet | 46.0 | 7.06 | 36.0 | 92.0 | 4.0 | 100.0 | 84.0 | 0.0 | 40.0 | 96.0 |
| RVT | 62.9 | 5.28 | 52.0 | 99.2 | 11.2 | 88.0 | 71.2 | 4.0 | 91.0 | **100.0** |
| Act3D | 63.2 | 5.56 | 96.8 | 80.8 | 24.0 | 95.2 | 78.4 | 3.2 | 59.2 | 93.6 |
| 3D Diffuser Actor | 81.3 | 3.0 | 96.0 | **100.0** | 65.6 | 96.8 | 89.6 | 24.0 | 93.6 | 98.4 |
| RVT2 | 81.4 | 2.89 | **100.0** | 99.0 | 40.0 | **99.0** | 74.0 | 38.0 | **95.0** | **100.0** |
| ARP | 81.6 | 2.83 | 97.6 | 88.0 | 53.2 | 96.0 | **90.4** | 48.0 | 92.0 | **100.0** |
| **TVVE (Ours)** | **86.6** | **2.17** | **100.0** | **100.0** | **98.0** | 94.0 | 90.0 | **54.0** | 92.0 | **100.0** |

| Method | Put Cupboard | Put Drawer | Put Safe | Screw Bulb | Slide Block | Sort Shape | Stack Blocks | Stack Cups | Sweep Dustpan | Turn Tap |
|--------|--------------|------------|----------|------------|-------------|------------|--------------|------------|---------------|----------|
| 3D Diffuser Actor | **85.6** | 96.0 | **97.6** | 82.4 | 97.6 | 44.0 | 68.3 | 47.2 | 84.0 | 99.2 |
| RVT2 | 66.0 | 96.0 | 96.0 | **88.0** | 92.0 | 35.0 | **80.0** | **69.0** | **100.0** | 99.0 |
| ARP | 68.0 | 99.2 | 94.4 | 85.6 | 98.4 | 35.2 | 55.2 | 76.8 | 90.4 | **100.0** |
| **TVVE (Ours)** | 74.0 | **100.0** | 78.0 | 86.0 | **100.0** | **62.0** | 74.0 | 64.0 | 92.0 | **100.0** |

**说明**: TVVE 取得 **86.6%** 平均成功率，超过此前 SOTA 的 ARP（81.6%）约 5%，平均排名 2.17 最优。Insert Peg（65.6%→98.0%）、Sort Shape（36.0%→62.0%）增益显著，体现动态视角对遮挡敏感任务的价值。

### Table 2: RLBench Single-View Setup (10 tasks) / 单视角 10 任务成功率（%）

| Method | Avg. SR↑ | Avg. Rank↓ | close jar | open drawer | sweep dustpan | meat off grill | turn tap | slide block | put drawer | drag stick | push buttons | stack blocks |
|--------|----------|------------|-----------|-------------|---------------|----------------|----------|-------------|------------|------------|--------------|--------------|
| GNFactor | 31.7 | 4.90 | 25.3 | 76.0 | 28.0 | 57.3 | 50.7 | 20.0 | 0.0 | 37.3 | 18.7 | 4.0 |
| Act3D | 65.3 | 3.35 | 52.0 | 84.0 | 80.0 | 66.7 | 64.0 | **100.0** | 54.7 | 86.7 | 64.0 | 0.0 |
| 3D Diffuser Actor | 78.4 | 2.20 | 82.7 | **89.3** | **94.7** | 88.0 | 80.0 | 92.0 | 77.3 | **98.7** | 69.3 | 12.0 |
| RVT2 | 82.7 | 2.40 | 96.0 | 78.7 | 73.3 | **93.3** | **93.3** | 89.3 | 73.3 | 76.0 | **96.0** | **57.3** |
| **TVVE (Ours)** | **83.2** | **2.15** | **100.0** | 84.0 | 94.0 | 92.0 | 92.0 | 90.0 | **98.0** | 50.0 | 82.0 | 50.0 |

**说明**: 单视角设定下 TVVE 仍以 83.2% 取得最高平均成功率与最佳排名 2.15，说明视角探索+TaskMoE 的收益不依赖多相机硬件。

### Table 3: RLBench-OG (OOD Robustness) / 分布外鲁棒性（mean±std %）

| Models | Avg. SR↑ | Rank↓ | Occlusion 1 | Occlusion 2 | Light Color | Table Color | Table Texture | Distractor | Background Texture | Camera Pose |
|--------|----------|-------|-------------|-------------|-------------|-------------|---------------|------------|--------------------|-------------|
| Diffusion Policy | 23.8±1.9 | 4.0 | 27.4±0.8 | 23.4±1.1 | 22.9±0.8 | 22.5±0.7 | 22.6±0.5 | 24.4±1.4 | 24.9±1.2 | 22.2±1.2 |
| ARP | 63.7±6.1 | 2.5 | 73.0±1.6 | 52.6±1.0 | 59.8±0.8 | 62.7±0.5 | 61.3±1.0 | 62.4±1.4 | 68.1±1.6 | 69.7±1.4 |
| RVT2 | 64.5±8.3 | 2.1 | 72.8±2.0 | 46.9±0.4 | 60.8±1.2 | 61.8±1.1 | 64.0±2.0 | **63.4±0.2** | 72.6±0.6 | **74.0±0.8** |
| **TVVE (Ours)** | **67.0±6.2** | **1.4** | **75.0±1.2** | **58.0±1.1** | **63.7±1.1** | **64.6±0.7** | **66.8±0.9** | 60.2±0.6 | **74.3±0.9** | 73.2±0.9 |

**说明**: TVVE 在 OOD 基准上取得最优总体成功率 67.0% 与最佳排名 1.1（正文）/1.4（表）。对零样本遮挡（Occlusion 2）相比 ARP +5.4，背景纹理与相机位姿变化下亦保持高鲁棒，印证“任务感知视角+解耦表征”能缓解干扰。

### Table 4: Real-World (Dobot & Franka) / 真实机器人成功率（%）

| Arm | Method | Avg. SR↑ | Pick Grape | Stack Bowls | Push Buttons | Collect Fruits | Put Item Drawer |
|-----|--------|----------|------------|-------------|--------------|----------------|------------------|
| Dobot | Diffusion Policy | 68.0 | 90.0 | 70.0 | 70.0 | 50.0 | 60.0 |
| Dobot | **TVVE (Ours)** | **88.0** | **100.0** | **90.0** | **100.0** | **70.0** | **80.0** |
| Franka | Diffusion Policy | 58.0 | 90.0 | 50.0 | 60.0 | 40.0 | 50.0 |
| Franka | ARP | 72.0 | 100.0 | 70.0 | 80.0 | 50.0 | 60.0 |
| Franka | **TVVE (Ours)** | **78.0** | 100.0 | 70.0 | **90.0** | **60.0** | **70.0** |

**说明**: Dobot 平台 TVVE 88.0% 对 DP 68.0%（Push Buttons +30、Stack Bowls +20）；Franka 平台 78.0% 超 ARP 72.0% 与 DP 58.0%，验证跨平台、跨相机配置的真实可用性。

### Table 5: Ablation (RLBench Multi-View) / 主消融（%）

| Configuration | Avg. SR↑ |
|---------------|----------|
| **TVVE (TaskMoE+MVEP)** | **86.6** |
| w/o TaskMoE | 85.6 |
| Fixed Viewpoints | 83.3 |
| Random Viewpoints | 8.9 |

**说明**: 去掉 TaskMoE 降到 85.6%；把探索视角换成固定视角降到 83.3%，换成随机视角则**崩塌到 8.9%**——强烈证明 MVEP 学到的视角探索策略对感知与动作执行至关重要。

### Table 6: TaskMoE on Seen vs Unseen Tasks / 已见与未见任务（%，5 个代表任务）

| 设置 | w/ TaskMoE | w/o TaskMoE |
|------|------------|-------------|
| Seen tasks (PID/TT/PGC/PMS/CJ) 平均 | **80.8** | 72.0 |
| Unseen task: Open Drawer | **72.0** | 60.0 |

**说明**: TaskMoE 不仅把已见任务平均从 72.0% 提到 80.8%，对训练中未见的 Open Drawer 也从 60.0% 提到 72.0%，证明其增强了多任务泛化能力。（PID=Put Item in Drawer, TT=Turn Tap, PGC=Put Groceries in Cupboard, PMS=Put Money in Safe, CJ=Close Jar）

### Table 7: Effect of View Number K and Radial Constraint r / 视角数与半径约束（%）

| K | r (m) | PID | TT | PGC | PMS | CJ | Avg. SR↑ |
|---|-------|-----|------|------|-----|------|----------|
| 2 | 0.75–1.3 | 12.0 | 92.0 | 0.0 | 12.0 | 20.0 | 27.2 |
| 3 | 0.75–1.3 | 32.0 | 92.0 | 16.0 | 32.0 | 76.0 | 49.6 |
| 4 | 0.75–1.3 | 52.0 | 100.0 | 4.0 | 68.0 | 52.0 | **55.2** |
| 3 | 0.60–1.56 | 16.0 | 96.0 | 8.0 | 44.0 | 80.0 | 48.8 |
| 3 | 0.90–1.04 | 44.0 | 100.0 | 20.0 | 36.0 | 80.0 | **56.0** |

**说明**: 视角数 $K$ 从 2→3→4 单调提升成功率（27.2→49.6→55.2），多视角缓解遮挡；半径约束过紧（0.90–1.04）反而最优（56.0），过松（0.60–1.56）下降。综合算力成本作者选 $K=3$。

### Table 8: Franka — More Tasks / Franka 扩展任务（成功次数/10）

| Method | Avg. SR↑ | Pick Place | Stack Bowls | Push Buttons | Collect Fruits | Put Item Drawer | Rotate Handle | Fold Towel | Open Lip | Unscrew Bottle | Reach Drag |
|--------|----------|------------|-------------|--------------|----------------|------------------|---------------|------------|----------|----------------|------------|
| ARP | 62.0% | 10/10 | 7/10 | 8/10 | 5/10 | 6/10 | **3/10** | 5/10 | 7/10 | 3/10 | 8/10 |
| **TVVE (Ours)** | **70.0%** | 10/10 | 7/10 | **9/10** | **6/10** | **7/10** | **3/10** | 5/10 | **9/10** | **5/10** | **9/10** |

**说明**: 在涵盖关节体/可变形物体/工具使用等 10 个更难任务上，TVVE 70.0% 超 ARP 62.0%，体现操作类型多样性下的适应性。

### Table 9: Dobot Robustness/Generalization (Pick Grape) / 真实鲁棒性测试（%）

| Method | Avg. SR↑ | Seen | Inst. | Bkg. | Obj. | Occl. | Illum. |
|--------|----------|------|-------|------|------|-------|--------|
| Diffusion Policy | 53.3 | 90.0 | 80.0 | 70.0 | 60.0 | 10.0 | 10.0 |
| **TVVE (Ours)** | **71.7** | **100.0** | **100.0** | **90.0** | **90.0** | **20.0** | **30.0** |

**说明**: 在未见实例/背景/物体、重遮挡、光照变化下 TVVE 全面优于 DP（71.7% vs 53.3%）。重遮挡是主要失败来源（20% vs 10%），仍是难点。

### Table 10: MVEP Offline-RL Hyperparameters / MVEP 离线 RL 超参（附录）

| 类别 | 超参 | 值 |
|------|------|----|
| MVEP | 相机视角数 $K$ | 3 |
| MVEP | 输入点云数 $N_{\text{MVEP}}$ | 2048 |
| MVEP | embedding size | 512 |
| Train&Eval | 观测 (RGBD) | 4×128×128×4 |
| Train&Eval | 重渲染分辨率 | 224×224 |
| Train&Eval | 最大评测步数 | 25 |
| Train&Eval | train epochs | 20 |
| Train&Eval | batch size | 96 |
| Train&Eval | learning rate | 2.0e-6 (cosine) |
| Train&Eval | optimizer | LAMB |

**说明**: MVEP 阶段的关键训练配置；监督预训练阶段超参遵循 ARP。

### Table 11: Performance–Efficiency Trade-off / 性能-效率权衡（RLBench）

| Method | Avg. SR (%) | Avg. Inference Time (s) |
|--------|-------------|-------------------------|
| ARP | 81.6 | 0.394 |
| **TVVE** | **86.6** | 0.436 |

**说明**: TVVE 比 ARP 高 5% 成功率，推理延迟仅增加约 10.7%（0.394→0.436s），借助采样加速与相机缓存控制开销，达到性能-效率的可接受平衡。

### Table 12: Data Efficiency (Put in Cupboard) / 数据效率（mean±std %）

| Method | 20 | 40 | 80 | 100 |
|--------|----|----|----|-----|
| ARP | 6.7±2.3 | 14.7±2.3 | 28.0±4.0 | 45.3±6.1 |
| **TVVE (Ours)** | **10.7±2.3** | **17.3±2.3** | **30.7±6.1** | **52.0±4.0** |

**说明**: 各示范规模下 TVVE 均优于 ARP，且差距随数据量增大而扩大，说明优势源于架构改进而非简单扩数据。

### Table 13–16: RLBench-OG Per-Task Results / OOD 逐任务结果（附录，Task Mean，%）

| 模型 | Occ.1 | Occ.2 | Light | Table Color | Table Tex. | Distractor | Bkg Tex. | Camera | 总均值 |
|------|-------|-------|-------|-------------|-----------|------------|----------|--------|--------|
| **TVVE (T13)** | **75.0** | **58.0** | **63.7** | **64.6** | **66.8** | 60.2 | **74.3** | 73.2 | **67.0±6.2** |
| RVT2 (T14) | 72.8 | 46.9 | 60.8 | 61.8 | 64.0 | 63.4 | 72.6 | 74.0 | 64.5±8.3 |
| ARP (T15) | 73.0 | 52.6 | 59.8 | 62.7 | 61.3 | 62.4 | 68.1 | 69.7 | 63.7±6.1 |
| DP (T16) | 27.4 | 23.4 | 22.9 | 22.5 | 22.6 | 24.4 | 24.9 | 22.2 | 23.8±1.9 |

**说明**: 四个模型在 RLBench-OG 10 任务 8 类扰动下的 Task Mean 汇总（详见正文 Table 3）。TVVE 总均值领先，尤其在零样本遮挡 Occlusion 2 上对 RVT2/ARP 优势最明显（58.0 vs 46.9/52.6）；block_pyramid、solve_puzzle、water_plants 等长程/复杂任务所有模型都偏低，是共同短板。

### Table 17: TVVE Generalization to Unseen Tasks / 对未见任务的泛化（%）

| 设置 | Avg. SR↑ | 代表 Seen 任务 | 代表 Unseen 任务 |
|------|----------|----------------|------------------|
| TVVE | **80.2** | Insert Peg 24.0 / Put Item Drawer 100.0 / Turn Tap 100.0 / Push Buttons 96.0 | Open Drawer 88.0 / Water Plants 12.0 / Close Drawer 44.0 / Toilet Seat Down 100.0 |

**说明**: 在 Insert_Peg、Put_Item_Drawer 等任务上训练后，TVVE 对若干未见任务（如 Open Drawer 88%、Toilet Seat Down 100%）仍取得高成功率（总均值 80.2%），验证架构的泛化能力；少数复杂未见任务（Water Plants 12%）仍困难。

---

## 实验

### 数据集 / 基准

| 基准 | 规模 | 特点 | 用途 |
|------|------|------|------|
| [[RLBench]] | multi-view 18 任务（每任务 100 示范、25 评测）；single-view 10 任务 | CoppeliaSim + 7-DoF Franka Panda，4 路固定视角 128×128 RGB-D | 训练/测试 |
| [[RLBench-OG]]（本文构建） | 10 任务，含 Occlusion Suite + Generalization Suite，每任务 50 训练 / 25×3 评测 | 遮挡 + 光照/桌面/背景/干扰物/相机位姿 6 类扰动；含 zero-shot 设置 | OOD 鲁棒性测试 |
| 真实世界 Dobot Nova 2 | 5 任务，每任务 50 示范，10 trials | 三相机（侧/前/腕），多视角 | 训练/测试 |
| 真实世界 Franka Research 3 | 5+10 任务，每任务 50 示范，10 trials | 单前视相机（ORBBEC Femto Bolt） | 训练/测试 |

### 实现细节

- **基础结构**：基于 [[RVT-2]] 的粗-细定位与 [[Multi-View Transformer|MVT]]，动作策略基于 [[ARP]]（Chunking Causal Transformer）。
- **超参**：相机视角数 $K=3$，TaskMoE 门数 $N_G=8$、专家数 $N_E=16$，每任务选 top-2 专家。
- **训练硬件**：4× NVIDIA RTX A800 GPU 训练；评测在单卡 A800 顺序进行。
- **MVEP 阶段**：输入 2048 点点云、embedding 512、重渲染 224×224、batch 96、lr 2e-6（cosine）、LAMB 优化器、20 epochs（见 Table 10）。
- **强化学习**：[[PPO]] + 伪环境交互（Algorithm 1），shadow network（训练前策略的冻结副本）提供参考损失 $\mathcal{L}_{\text{ref}}$，经验元组存入 replay buffer 做批量交互更新，避免物理环境交互。

### 关键实验结论

- **RLBench**：multi-view 86.6%（超 ARP ~5%），single-view 83.2%，均为 SOTA。
- **RLBench-OG**：67.0% 总体最优、排名 1.1，零样本遮挡与相机位姿变化下尤其稳。
- **真实世界**：Dobot 88.0%、Franka 78.0%，均超 DP/ARP；重遮挡仍是主要失败源。
- **消融**：随机视角导致成功率从 86.6% 崩到 8.9%（MVEP 关键）；TaskMoE 提升已见与未见任务（Table 5/6）；$K=3$ 与适中半径约束最优（Table 7）。
- **效率**：相比 ARP 仅增 ~10.7% 推理延迟换 5% 成功率。

---

## 批判性思考

### 优点
1. **“先看后动”范式新颖且自洽**：把视角选择从固定布置变为可学习的 RL 策略，并用伪环境规避物理交互成本，工程上务实；随机视角崩塌实验（86.6%→8.9%）有力证明了视角选择的因果重要性。
2. **TaskMoE 设计扎实**：门-任务、门-专家两级解耦 + 交叉注意力/FiLM 条件 + 熵/语义正则，既缓解多任务干扰又有 t-SNE/路由可视化（Fig 10/11）佐证专家专门化，且对未见任务有迁移收益。
3. **评测全面且贡献了基准**：自建 RLBench-OG 系统化覆盖 8 类扰动，配合仿真+真实双臂双平台实验，鲁棒性论证较充分。

### 局限性
1. **作者自陈的两点**：多视角重渲染增加推理延迟；依赖准确的全局点云，难以处理反光/透明物体。
2. **绝对性能与难任务短板**：block_pyramid、solve_puzzle、water_plants、straighten_rope 等长程/复杂任务在 OOD 下成功率极低（部分接近 0），表明对接触丰富/长程任务仍乏力。
3. **依赖深度与点云重建**：方法建立在 RGB-D + 多视角标定之上，单 RGB 或标定不准场景的可迁移性未讨论；重遮挡（真实测试 20%）仍是主要失败模式。
4. **复杂度高**：流水线串联粗定位、点云聚合、RL 视角探索、重渲染、TaskMoE-ARP，组件多、超参多（多套损失权重、正则系数），复现工程量较大。

### 潜在改进方向
1. 引入多传感器融合与域自适应以提升真实世界（尤其反光/透明物体、重遮挡）鲁棒性。
2. 将视角探索扩展为时序闭环（边动边重新探索视角），而非每步独立预测。
3. 把 $K$、半径约束、TaskMoE 门/专家数等做成自适应或可搜索，减少手工调参。
4. 在更长程、接触丰富、可变形物体任务上验证，并与更多最新 VLA/3D 策略基线对比。

### 可复现性评估
- [x] 代码开源（项目主页 https://hcplab-sysu.github.io/TAVP/ 声明 Code 可用）
- [ ] 预训练模型（论文未明确声明权重 release）
- [x] 训练细节较完整（附录给出超参表 10、三阶段流程、伪环境算法）
- [x] 数据集可获取（RLBench 公开；RLBench-OG 由本文基于 RLBench/COLOSSEUM 构建并提供配置）

---

## 速查卡片

> [!summary] TVVE: Task-Aware Virtual View Exploration for Robotic Manipulation
> - **核心**: 让机器人“先学会看再学会动”——用 RL 探索策略在重建点云上选取最优虚拟视角并重渲染，配 TaskMoE 缓解多任务干扰。
> - **方法**: 点云聚合 → Coarse Grounding 粗定位 → TaskMoE-MVT 任务特征 → MVEP（look-at 高斯策略 + PPO + 伪环境）探索 $K=3$ 视角 → 重渲染 → TaskMoE-ARP 出动作；三阶段训练（固定视角预训练 → RL 训 MVEP → 联合微调）。
> - **结果**: RLBench multi-view 86.6%（超 ARP ~5%）/ single-view 83.2% / RLBench-OG 67.0%（SOTA）/ 真实 Dobot 88%、Franka 78%；随机视角崩到 8.9% 证明视角探索关键。
> - **基准**: 提出 RLBench-OG（8 类 OOD 扰动）。
> - **项目**: https://hcplab-sysu.github.io/TAVP/

---

*笔记创建时间: 2026-06-29*
