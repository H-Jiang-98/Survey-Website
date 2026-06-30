---
title: "ProGAL-VLA: Grounded Alignment through Prospective Reasoning in Vision-Language-Action Models"
method_name: "ProGAL-VLA"
authors: [Nastaran Darabi, Amit Ranjan Trivedi]
year: 2026
venue: CVPR
tags: [VLA, grounding, contrastive-learning, scene-graph, ambiguity-detection, selective-prediction, hierarchical-policy]
zotero_collection: _inbox
image_source: online
arxiv_html: https://arxiv.org/html/2604.09824v1
created: 2026-06-29
---

# ProGAL-VLA: Grounded Alignment through Prospective Reasoning in Vision-Language-Action Models

## 元信息

| 项目 | 内容 |
|------|------|
| 作者 | Nastaran Darabi, Amit Ranjan Trivedi |
| 机构 | University of Illinois Chicago (UIC), USA |
| 会议 | CVPR 2026 |
| 类别 | VLA（Vision-Language-Action） |
| 日期 | 2026-04（arXiv v1） |
| 项目主页 | https://nstrndrbi.github.io/ProGAL |
| 链接 | [arXiv](https://arxiv.org/abs/2604.09824) / Code（未公开） |

---

## 一句话总结

> 在动作执行前强制把语言推断出的符号子目标"验证绑定"到 3D 场景中的具体实体，用一个验证瓶颈 + 对比对齐损失根治 VLA 的"语言忽视"，并让注意力熵天然成为歧义检测信号。

---

## 核心贡献

1. **验证瓶颈（Verification Bottleneck）**: 提出层次化架构，把慢规划（[[π_slow|Prospective Planner]]）与快控制（[[π_fast|Action Policy]]）解耦，并强制所有动作只能以"经过验证的目标嵌入" $g_t$ 为条件——动作在给定 $g_t$ 后与原始语言 $L$ 条件独立。这从结构上（而非统计上）杜绝了控制层在未验证表征上盲动。
2. **Grounding Alignment Contrastive (GAC) 损失**: 用一个实体级 [[InfoNCE]] 对比目标，把符号子目标 $s_t$ 与 3D 实体嵌入 $e^+$ 绑定，并被证明给出 $I(S;E)$ 的变分下界；显著提升实体检索 Recall@1（0.41→0.71，N=8）。
3. **注意力熵即歧义信号 + 新基准 CAB**: [[SACA]] 的注意力分布熵天然刻画 grounding 模糊度，配合阈值即得校准良好的选择性预测（abstention）；并构造 Custom Ambiguity Benchmark（CAB）专测属性级歧义下的"主动澄清"能力，AUROC 0.52→0.81。

---

## 问题背景

### 要解决的问题
当前 [[VLA]] 模型存在两个顽疾（在 [[LIBERO-Plus]] 上被系统揭示）：
1. **语言忽视（language ignorance）**：策略依赖视觉先验/捷径，对指令语义不敏感——换指令而动作几乎不变；
2. **机器人不稳定（robotic instability）**：一旦让语义推理去扰动视觉运动协调，底层控制就崩。

二者根源是同一个：**符号意图（symbolic intent）与具身控制（embodied control）之间的结构性错位**。

### 现有方法的局限
- 多数 VLA 通过**浅层拼接（shallow concatenation）**融合模态，语言只作为弱条件信号去"调制"视觉，并**不保证推断出的符号目标真的对应到 3D 场景里一个可操作的实体**；控制层因此在"未验证表征"上动作，导致语义与物理双重失败。
- 层次化/双流方法（[[SayCan]]、[[Code as Policies]]）虽分离了高层推理与底层控制，但**把计划当作文本/特征直接注入，从不验证它与感知场景的对应关系**。
- 对比学习类（[[CLIP]]、[[R3M]]）通常只把文本对齐到 2D 区域，粒度太粗，未触及"符号→3D 实体"的绑定（binding）问题。

### 本文的动机
核心主张：**在执行任何动作之前，先验证"语言意图既语义一致、又在当前 3D 场景中物理可达"**。为此把流水线显式拆成"推理 → grounding → 控制"三段，且只通过一个**验证步骤**相连——慢规划只产生符号子目标，[[GSM]] 只构建实体中心 3D 表征，[[SACA]] 负责验证二者对应，[[π_fast]] 只对验证后的 $g_t$ 动作。这样语言对动作的影响是可度量、可证明的（条件互信息），且 $g_t$ 在视觉扰动下平滑变化 → 动作分布稳定。

---

## 方法详解

### 模型架构

ProGAL-VLA 采用 **层次化"慢规划-grounding-快控制"** 架构（见 Figure 1），由四个组件构成：
- **输入**: 语言指令 $L$ + 当前观测 $O_t$ + 机器人本体状态 $q_t$ + 时序记忆 $M_{t-1}$
- **[[π_slow|Prospective Planner]]**: [[Qwen2.5-VL]]-Instruct-7B，**每个 episode 仅异步调用一次**，输出符号子目标模板 $s_t$（如 `grasp_green_mug`）
- **[[GSM|Grounded State Module]]**: 用 [[YOLO-World]] + [[Metric3D]] 把观测抬升为实体中心的 3D 图，得实体集合 $E_t$
- **[[SACA|State Alignment Cross Attention]]**: 验证瓶颈核心，将 $s_t$ 与 $E_t$ 交叉注意力绑定，产出**验证目标嵌入** $g_t$
- **[[π_fast|Action Policy]]**: [[OpenVLA]]-7B 主干，**只**以 $g_t$（投影到 4096 维）、$O_t$、$q_t$ 为条件输出动作
- **训练目标**: 模仿学习 $\mathcal{L}_{\text{action}}$ + 对比 grounding $\mathcal{L}_{\text{GAC}}$
- **关键设计**: 每步推理栈仍是与所有 baseline 相同的 7B OpenVLA，慢规划异步、不计入控制时延——**改进来自验证 grounding 管线本身，而非堆参数**。

### 核心模块

#### 模块1: Prospective Planner（[[π_slow]]，慢规划器）

**设计动机**: 只做语义推理，不碰细粒度视觉运动；把语言多样性归一化为结构化查询，降低词法熵。

**具体实现**:
- 从指令、观测、记忆预测"下一步该做什么"的符号子目标 $s_t$（如"grasp red block"），**至于实体在哪里交给 grounding 阶段**；
- 输出是短符号模板（"pick up the green mug" → `grasp_green_mug`），为 SACA 提供结构化查询，而非又一个控制策略；
- **不做**多步前瞻或底层推理，对齐主要仍由 GSM/SACA 驱动。

#### 模块2: Grounded State Module（[[GSM]]，接地状态模块）

**设计动机**: 把原始观测变成**实体中心的 3D 表征**，把物体身份与姿态/外观解耦，为符号绑定提供精度。

**具体实现**:
- 用 [[YOLO-World]] 开放词表检测 + [[Metric3D]] 深度，把每个检测抬升为带外观/几何/位置属性的 3D 实体节点 $e_i$；
- 维护容量 $N_{\max}=16$ 的**有界实体记忆** $M_t$（FIFO 更新），提供短程时序稳定，能推理被遮挡/部分可见物体；
- **GSM 刻意做得最小化**：只 lift 检测、跟踪少量实体、维持短期一致，不做语义推理或长程预测（那是 $\pi_{slow}$ 的事）。

#### 模块3: State Alignment Cross Attention（[[SACA]]，验证瓶颈）

**设计动机**: 这是整个方法的"验证瓶颈"——把符号目标强制绑定到一个具体 3D 实体，并用注意力熵暴露歧义。

**具体实现**:
- 符号目标嵌入为 query $Q$，3D 实体嵌入为 key/value $K,V$，标准缩放点积注意力输出 $g_t$（公式 5-6）；
- 注意力分布把 $s_t$ 链接到具体实体，其**熵**给出内在歧义信号：低熵=唯一 grounding，高熵=指令无法解析到单一实体；
- **只有验证后的 $g_t$ 才传给控制器**，从而阻断语义欠定或与场景矛盾的动作。

#### 模块4: Action Policy（[[π_fast]]，快控制器）

**设计动机**: 让动作"只"依赖验证后的语义与感知证据，彻底切断未接地的语言捷径。

**具体实现**:
- 以 $g_t,O_t,q_t$ 为条件采样动作（公式 7），用模仿损失（公式 8）训练；
- **从输入中排除原始语言特征** $L$ —— 这正是验证瓶颈得以成立的架构保证（附录 §13 形式化）。

#### 训练对齐对的离线构造

GAC 需要 $(s_t,e^+)$ 正样本对，作者用三步**弱监督**离线构造：(i) 教师 VLM 把示范切成子目标得 $s_t$；(ii) GSM 建 3D tracklets $\{T_{obj_i}\}$；(iii) 在子目标末端用"最近邻一致性"选出被操作实体 $e^+$（公式 12）。**实体身份与子目标标签从不被人工标注**，VLM 切分与匹配都含噪声——故 GAC 不依赖 baseline 拿不到的特权信息，鲁棒增益来自验证 grounding 机制本身（Table 4/8 佐证）。

### 关键公式与机制

#### 公式1: [[π_slow|符号子目标预测]]

$$
s_t = \pi_{\text{slow}}(L, O_t, M_{t-1})
$$

**含义**: 慢规划器从指令、观测、时序记忆预测下一步的符号子目标。

**符号说明**:
- $L$: 语言指令；$O_t$: $t$ 时刻观测；$M_{t-1}$: 时序记忆
- $s_t$: 符号子目标模板（如 `grasp_green_mug`）

#### 公式2-4: [[GSM]] 实体中心 3D 表征

$$
G_t = \{e_1, e_2, \dots, e_n\}, \quad M_t = \text{Update}(M_{t-1}, G_t)
$$

$$
E_t = \text{Concat}\big(G_t,\ \text{Retrieve}(M_{t-1}, O_t)\big)
$$

**含义**: 当前帧检测出实体集 $G_t$，更新有界记忆 $M_t$，再把当前实体与从记忆检索到的实体拼接成完整实体表征 $E_t$。

**符号说明**:
- $e_i$: 编码物体级视觉特征、几何姿态、语义描述符的实体节点
- $M_t$: 维持时序连续性的记忆（容量 16，FIFO）
- $E_t$: 送入 SACA 的实体集合（身份与姿态/外观解耦）

#### 公式5-6: [[SACA]] 验证（交叉注意力 grounding）

$$
Q = \text{Embed}_{\text{sym}}(s_t), \qquad K, V = \text{Embed}_{\text{gnd}}(E_t)
$$

$$
g_t = \text{Softmax}\!\left(\frac{Q K^{\top}}{\sqrt{d}}\right) V
$$

**含义**: 符号目标作 query 去注意 3D 实体 key/value，产出验证目标嵌入 $g_t$；注意力分布的熵即歧义度量。

**符号说明**:
- $Q$: 符号子目标的查询嵌入；$K,V$: grounded 实体的键/值嵌入
- $d$: 嵌入维度（缩放因子）；$g_t$: 唯一传给控制器的验证目标嵌入

#### 公式7-8: [[π_fast|动作策略]]与模仿损失

$$
a_t \sim \pi_{\text{fast}}(g_t, O_t, q_t), \qquad \mathcal{L}_{\text{action}} = \|a_t - a_t^{*}\|^2
$$

**含义**: 控制器只以验证目标、观测、本体状态为条件采样动作；用与专家动作 $a_t^*$ 的 L2 距离做模仿训练。

**符号说明**:
- $a_t$: 预测动作；$a_t^*$: 专家（示范）动作；$q_t$: 机器人本体状态

#### 公式9-11: [[GAC]] 对比 grounding 损失

$$
q = \text{Embed}_{\text{sym}}(s_t), \qquad k^{+} = \text{Embed}_{\text{gnd}}(e^{+})
$$

$$
\mathcal{L}_{\text{GAC}} = -\log \frac{\exp\big(\text{sim}(q, k^{+})/\tau\big)}{\sum_{i} \exp\big(\text{sim}(q, k_i)/\tau\big)}
$$

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{action}} + \lambda\, \mathcal{L}_{\text{GAC}}
$$

**含义**: 标准 [[InfoNCE]] 形式——拉近符号查询 $q$ 与正确实体 $k^+$、推远其余实体；总损失为动作损失加权重 $\lambda$ 的 GAC。它逼迫规划器产生对应真实实体的子目标，并让 GSM 实体嵌入在视角/布局变化下仍可区分。

**符号说明**:
- $q$: 符号子目标嵌入；$k^+$: 正样本实体嵌入；$k_i$: 候选实体（含负样本）
- $\text{sim}(\cdot,\cdot)$: 相似度（点积/余弦）；$\tau=0.07$: 温度；$\lambda=0.1$: GAC 权重

#### 公式12: 正样本实体的时空匹配

$$
e^{+} = \arg\min_{i} \big\| \text{pos}(T_{obj_i}, t_{\text{end}}) - \text{pos}(\text{gripper}, t_{\text{end}}) \big\|
$$

**含义**: 在子目标结束时刻 $t_{\text{end}}$，选与夹爪位置最近的物体 tracklet 作为该子目标的接地正实体——离线弱监督构造对齐对。

**符号说明**:
- $T_{obj_i}$: 第 $i$ 个物体的 3D tracklet；$\text{pos}(\cdot, t_{\text{end}})$: 末端时刻位置；$\text{gripper}$: 夹爪

#### 公式13-14: 语言影响的条件互信息度量

$$
\mathcal{I}_t^{\pi} \triangleq I(L; a_t \mid O_t, q_t)
$$

$$
\Lambda^{\pi} \triangleq 1 - \frac{\mathbb{E}[\mathcal{I}_t^{\pi}]}{\log|\mathcal{L}|}
$$

**含义**: 用"给定观测与状态后，动作还从语言获得多少信息"度量语言敏感度。$\mathcal{I}_t^\pi=0$ 即语言忽视；归一化指标 $\Lambda^\pi\approx 1$ 表示语言几乎无影响（越大越差）。

**符号说明**:
- $I(\cdot;\cdot\mid\cdot)$: 条件互信息；$|\mathcal{L}|$: 指令空间大小（归一化用）

#### 公式18-19: 验证瓶颈假设与 Proposition 1

$$
\text{假设 1（验证瓶颈）：}\quad a_t \perp (L, O_t, M_{t-1}) \mid (g_t, q_t)
$$

$$
I(L; a_t \mid O_t, q_t) = I(L; g_t \mid O_t, q_t) - I(L; g_t \mid a_t, O_t, q_t)
$$

**含义**: 给定验证嵌入 $g_t$ 与状态后，动作与语言条件独立（架构保证：$\pi_{fast}$ 拿不到 $L$）。由此推出 Proposition 1——只要 $g_t$ 随 $L$ 有意义地变化、且 $\pi_{fast}$ 响应这种变化，策略就对语言敏感。这从理论上解释了为何"以 $g_t$ 为条件"语言忽视最低。

**符号说明**:
- $\perp$: 条件独立；其余符号同前。证明见附录 §7.1（链式法则 + 条件独立项归零）。

#### 公式21: GAC 的 InfoNCE 互信息下界（Theorem 1）

$$
I(S; E) \;\geq\; \log N - \mathbb{E}[\mathcal{L}_{\text{GAC}}]
$$

**含义**: 最小化 GAC 即抬高"符号子目标 $S$ 与 grounded 实体 $E$"之间互信息的下界，解释了 Recall@1 随候选集增大仍大幅提升的实验现象（证明见附录 §7.2，标准 InfoNCE 变分推导）。

**符号说明**:
- $S,E$: 符号子目标 / grounded 实体的随机变量；$N$: 候选集（1 正 $N-1$ 负）大小

#### 公式22-26: Lipschitz 视觉鲁棒性（Proposition 2）

$$
\|g_t' - g_t\|_2 \leq L_{\Psi}\, d_{\mathcal{G}}(T, \mathrm{id})
$$

$$
\mathrm{TV}\big(\pi_{\text{fast}}(\cdot\mid g_t, O_t, q_t),\ \pi_{\text{fast}}(\cdot\mid g_t', O_t', q_t)\big) \leq L_{\pi} L_{\Psi}\, d_{\mathcal{G}}(T, \mathrm{id})
$$

**含义**: 若 grounding $\Psi$ 与策略 $\pi_{fast}$ 均 Lipschitz，则在视觉扰动 $T$（相机/布局/光照，来自扰动群 $\mathcal{G}$）下，$g_t$ 平滑变化 → 动作分布的总变差（TV）被扰动幅度 $d_{\mathcal{G}}(T,\mathrm{id})$ 有界。这正是 Table 2 鲁棒增益的理论对应。

**符号说明**:
- $L_\Psi, L_\pi$: grounding/策略的 Lipschitz 常数；$d_{\mathcal{G}}(T,\mathrm{id})$: 扰动 $T$ 偏离恒等的程度；$\mathrm{TV}$: 总变差距离

#### 公式27: SACA 注意力熵（选择性预测）

$$
H_t \triangleq -\sum_{i=1}^{n} \alpha_{t,i} \log \alpha_{t,i}
$$

**含义**: SACA 对实体的注意力权重 $\alpha_{t,i}$ 的熵刻画 grounding 歧义；对 $H_t$ 设阈值即得"可弃权（abstain）"的选择性策略，对应的 Risk-Coverage 曲线被严格改善（Fig 3）。

**符号说明**:
- $\alpha_{t,i}$: 对第 $i$ 个实体的注意力权重；$n$: 实体数

---

## 关键图表

<!-- 图片均使用 arXiv HTML 在线链接，未本地化 -->

### Figure 1: Overview of ProGAL-VLA / 整体架构

![Figure 1](https://arxiv.org/html/2604.09824v1/x1.png)

**说明**: ProGAL-VLA 全貌。指令 $L$ 与观测 $O_t$ 分别进入 [[π_slow|Prospective Planner]] 和 [[GSM]]；[[SACA]] 验证符号子目标 $s_t$ 与 3D 实体 $E_t$ 的对齐，产出**验证目标嵌入** $g_t$ 喂给动作策略 $\pi_{\text{fast}}$；训练时 [[GAC]] 目标强制"符号→实体"的正确绑定。此图点明全方法的灵魂——动作前先过"验证瓶颈"。

### Figure 2: Success vs. Language-Ignorance across Input Configs / 输入条件对比

![Figure 2](https://arxiv.org/html/2604.09824v1/Figure_T_LI.jpg)

**说明**: 在 [[LIBERO-Plus]] 上，分别以语言 $L$、子目标 $s_t$、验证嵌入 $g_t$ 为 $\pi_{fast}$ 条件时的成功率与语言忽视误差。**以 $g_t$ 为条件同时取得最高成功率与最低语言忽视**，与 Proposition 1（公式 19）一致——验证嵌入捕获了 $(O_t,q_t)$ 之外的语言信息。这是"验证 grounding 治语言忽视"最直接的证据。

### Figure 3: Risk-Coverage Curves on CAB / 选择性预测

![Figure 3](https://arxiv.org/html/2604.09824v1/Figure_RiskCoverage.jpg)

**说明**: CAB 上基于注意力熵的选择性策略 Risk-Coverage 曲线。ProGAL-VLA 在所有 coverage 水平下**失败风险严格更低**，说明不确定性更校准、在歧义输入上弃权更可靠。对应公式 27 的熵阈值机制。

### Figure 4: Fine-grained Robustness Analysis on LIBERO-Plus / 细粒度鲁棒性（附录）

![Figure 4 OpenVLA](https://arxiv.org/html/2604.09824v1/table10_OpenVLA.png)
![Figure 4 OpenVLA-OFT](https://arxiv.org/html/2604.09824v1/table10_OpenVLA-OFT.png)
![Figure 4 OpenVLA-OFT_w](https://arxiv.org/html/2604.09824v1/table10_OpenVLA-OFT_w.png)
![Figure 4 NORA](https://arxiv.org/html/2604.09824v1/table10_NORA.png)
![Figure 4 WorldVLA](https://arxiv.org/html/2604.09824v1/table10_WorldVLA.png)
![Figure 4 UniVLA](https://arxiv.org/html/2604.09824v1/table10_UniVLA.png)
![Figure 4 pi0](https://arxiv.org/html/2604.09824v1/table10_pi0.png)
![Figure 4 pi0-Fast](https://arxiv.org/html/2604.09824v1/table10_pi0-Fast.png)
![Figure 4 RIPT-VLA](https://arxiv.org/html/2604.09824v1/table10_RIPT-VLA.png)
![Figure 4 OpenVLA-OFT_m](https://arxiv.org/html/2604.09824v1/table10_OpenVLA-OFT_m.png)
![Figure 4 OpenVLA-OFT+](https://arxiv.org/html/2604.09824v1/table10_OpenVLA-OFT+.png)
![Figure 4 ProGAL-VLA Ours](https://arxiv.org/html/2604.09824v1/table10_ProGAL-VLA_Ours.png)

**说明**: 把 7 个扰动维度（Camera / Robot / Language / Light / Background / Noise / Layout）× 4 个任务套件（Spatial/Object/Goal/Long）的成功率画成柱状图（每个模型一张子图）。OpenVLA、$\pi_0$ 在 Camera/Robot 扰动下大幅退化；**ProGAL-VLA 在几乎所有维度都维持高鲁棒且行为更均匀**，印证 3D 实体空间推理带来的几何不变性。

### Table 1: Custom Ambiguity Benchmark 概览

| 项目 | 取值 |
|------|------|
| 场景数（train/val/test） | 32 / 8 / 8 |
| 每场景物体数（min/mean/max） | 3 / 4.8 / 6 |
| 物体类别 | blocks, mugs, bottles, fruit |
| 属性 | 颜色 4 种（各约 25%）；尺寸 2 种（各约 50%） |
| 指令总数 | 2,400 |
| 无歧义 / 有歧义 | 1,200 / 1,200 |
| 歧义规则 | 逐步丢属性直到 ≥2 个实体匹配 |
| 无歧义判定成功 | 正确抓取、不澄清 |
| 有歧义判定成功 | 抓取前先发澄清 |
| 澄清模块 | 熵门控的 `[CLARIFY]` token |

**说明**: CAB 专测**属性级碰撞**（如两个红块+一个蓝块+一个红苹果），强制模型做属性级区分而非仅类型匹配；歧义指令下"主动弃权澄清"才算成功，盲动算失败。这是论文歧义检测实验的基础。

### Table 2: Robustness across Perturbation Dimensions on LIBERO-Plus / 七维扰动鲁棒性（成功率 %）

| Model | Camera | Robot | Language | Light | Background | Noise | Layout | Total |
|-------|--------|-------|----------|-------|------------|-------|--------|-------|
| OpenVLA | 0.8 | 3.5 | 23.0 | 8.1 | 34.8 | 15.2 | 28.5 | 17.3 |
| OpenVLA-OFT | 56.4 | 31.9 | 79.5 | 88.7 | 93.3 | 75.8 | 74.2 | 70.0 |
| OpenVLA-OFT_w | 10.4 | 38.7 | 70.5 | 76.8 | 93.6 | 49.9 | 69.9 | 56.4 |
| NORA | 2.2 | 37.0 | 65.1 | 45.7 | 58.6 | 12.8 | 62.1 | 39.8 |
| WorldVLA | 0.1 | 27.9 | 41.6 | 43.7 | 17.1 | 10.9 | 38.0 | 25.3 |
| UniVLA | 1.8 | 46.2 | 69.6 | 69.0 | 81.0 | 21.2 | 31.9 | 43.9 |
| $\pi_0$ | 13.8 | 6.0 | 58.8 | 85.0 | 81.4 | 79.0 | 68.9 | 54.6 |
| $\pi_0$-Fast | 65.1 | 21.6 | 61.0 | 73.2 | 73.2 | 74.4 | 68.8 | 64.2 |
| RIPT-VLA | 55.2 | 31.2 | 77.6 | 88.4 | 91.6 | 73.5 | 74.2 | 69.3 |
| OpenVLA-OFT_m | 55.6 | 21.7 | 81.0 | 92.7 | 91.0 | 78.6 | 68.7 | 68.1 |
| OpenVLA-OFT+ | 92.8 | 30.3 | 85.8 | **94.9** | **93.9** | **89.3** | 77.6 | 79.6 |
| **ProGAL-VLA (Ours)** | **93.2** | **71.5** | **93.6** | 86.8 | 92.3 | 74.8 | **86.7** | **85.5** |

**说明**: ProGAL-VLA 在 8 列中 6 列最优，**Robot 扰动 30.3→71.5、Layout 77.6→86.7** 增益最大，总分 85.5%（次优 OpenVLA-OFT+ 为 79.6%）。注意它并非在每一列都赢（Light/Background/Noise 略低于 OFT+），但**跨异质扰动更均衡**，而非过拟合单一扰动——印证公式 26 的 Lipschitz 鲁棒性论证。

### Table 3: Ambiguity Detection & Selective Prediction on CAB / 歧义检测与选择性预测

| Model | AUROC↑ | AUPR↑ | ECE↓ | Cov@95↑ | FPR@95↓ | Clar@Ambig↑ | Unambig SR↑ | Total↑ |
|-------|--------|-------|------|---------|---------|-------------|-------------|--------|
| OpenVLA | 0.52 | 0.49 | 14.3 | 0.31 | 0.72 | 0.09 | 0.74 | 0.47 |
| ProGAL (w/o $\mathcal{L}_{\text{GAC}}$) | 0.66 | 0.63 | 9.8 | 0.57 | 0.41 | 0.42 | 0.82 | 0.64 |
| ProGAL (entropy-only) | 0.73 | 0.70 | 7.1 | 0.63 | 0.33 | 0.58 | 0.84 | 0.69 |
| **ProGAL-VLA (Ours)** | **0.81** | **0.79** | **4.6** | **0.78** | **0.18** | **0.81** | **0.89** | **0.79** |

**说明**: 全指标最优——AUROC 0.52→0.81、ECE 14.3→4.6、FPR@95 0.72→0.18，**有歧义时主动澄清率 0.09→0.81 且不损害无歧义成功率（0.74→0.89）**。逐行消融显示 GAC 与熵机制各有贡献（去 GAC 掉到 0.64，仅熵 0.69，完整 0.79）。

### Table 4: Entity Retrieval & Language Ignorance / 实体检索与语言忽视

| Model | N=8 R@1↑ | N=8 R@5↑ | N=16 R@1↑ | N=32 R@1↑ | Simple↓ | Spatial↓ | Relational↓ |
|-------|----------|----------|-----------|-----------|---------|----------|-------------|
| OpenVLA | 0.41 | 0.72 | 0.28 | 0.15 | 0.36 | 0.49 | 0.57 |
| ProGAL (w/o $\mathcal{L}_{\text{GAC}}$) | 0.57 | 0.86 | 0.43 | 0.29 | 0.22 | 0.31 | 0.45 |
| ProGAL ($L\!\rightarrow\!\pi_{\text{fast}}$) | 0.50 | 0.83 | 0.35 | 0.24 | 0.27 | 0.33 | 0.42 |
| **ProGAL (Ours)** | **0.71** | **0.93** | **0.58** | **0.41** | **0.08** | **0.14** | **0.19** |

**说明**: 检索在各候选集规模全面最优（N=8 R@1 0.41→0.71，N=32 仍 0.15→0.41），印证公式 21 的 InfoNCE 下界；语言忽视（越低越好）在 simple/spatial/relational 三类指令上全面下降（约 3-4 倍）。去 GAC 或把语言直接喂给 $\pi_{fast}$ 都落在两极之间，说明**层次规划+GSM+SACA+GAC 缺一不可**。

### Table 5: Per-step Latency on LIBERO-Plus / 单步时延（100 episodes，ms）

| Detector | GSM | SACA | $\pi_{\text{fast}}$ | Total |
|----------|-----|------|---------------------|-------|
| 43.0 ± 26.4 | 15.8 ± 9.7 | 10.7 ± 6.6 | 26.9 ± 16.5 | 96.4 ± 59.2 |

**说明**: 端到端单步约 96.4ms（吞吐 10.31 FPS，正文另处提到 ≈107.5ms / 9.31 FPS），可用于闭环操作。主要开销在检测器与 $\pi_{fast}$，**GSM/SACA 仅增加少量开销**——验证 grounding 的计算溢价很小。

### Table 6: Hyperparameters for ProGAL-VLA Components / 组件超参（附录）

| 组件 | 配置 |
|------|------|
| $\pi_{slow}$ Backbone | Qwen-2.5-VL-Instruct-7B；输入 448×448；输出符号模板；每 episode 调用一次（或 grounding 失败时） |
| GSM | 检测器 YOLO-World (L)；深度 Metric3D v2 (ViT-L)；图记忆 $N_{\max}=16$ 节点（FIFO）；实体嵌入维 $d=1024$ |
| $\pi_{fast}$ | OpenVLA-7B (Llama-2)；语义输入 $g_t$ 投影到 4096 维；动作空间 7-DoF 末端 + 二值夹爪 |
| 训练 | AdamW；lr $2\times10^{-5}$ 线性 warmup；batch 128；$\lambda=0.1$；$\tau=0.07$；4× NVIDIA A6000 |

**说明**: 推理时参数量由 OpenVLA 主干主导，慢规划异步、不计入控制时延——再次强调"非堆参数"的设计意图。

### Table 7: Robustness Breakdown / 机器人与相机扰动细分（成功率 %，附录）

| Model | Joint Drift | Joint Jitter | Viewpoint Shift | FOV Change |
|-------|-------------|--------------|-----------------|------------|
| OpenVLA | 2.1 | 4.9 | 1.2 | 0.4 |
| OpenVLA-OFT+ | 28.5 | 35.1 | 68.4 | 78.0 |
| **ProGAL-VLA (Ours)** | **68.2** | **74.8** | **85.1** | **91.7** |

**说明**: "Drift"=系统性关节偏移，"Jitter"=高频噪声。OpenVLA 在 drift/viewpoint 几乎崩溃（2.1%/1.2%），ProGAL-VLA 分别保持 68.2%/85.1%——**$g_t$ 作为稳定的 3D 锚点，对标定误差与几何相机运动免疫**；GSM 时序 3D 图过滤高频噪声。

### Table 8: Failure Mode Distribution / 失败模式分布（成功率/占比 %，附录）

| Model | Grounding Fail | Grasp Fail | Planning Fail | Success Rate |
|-------|----------------|------------|---------------|--------------|
| OpenVLA | 41.2 | 12.5 | 5.1 | 41.2 |
| ProGAL (w/o $\mathcal{L}_{\text{GAC}}$) | 22.4 | 14.1 | 4.8 | 58.7 |
| **ProGAL-VLA** | **6.3** | **7.1** | **4.5** | **82.1** |

**说明**: Grounding 失败（操作了几何合理但语义错误的物体，即"语言忽视"）从 41.2% 降到 6.3%；**去 GAC 会让 grounding 失败回升到 22.4%**，证明最大化 $I(S;E)$ 对正确绑定不可或缺。ProGAL-VLA 残余错误转为物理执行（抓取）失败为主——验证瓶颈基本解决了语义对齐这个核心问题。

---

## 实验

### 数据集 / 基准

| 基准 | 规模/构成 | 特点 | 用途 |
|------|-----------|------|------|
| [[LIBERO-Plus]] | 7 维扰动 × 4 套件（Spatial/Object/Goal/Long） | 首次揭示"语言忽视"现象的鲁棒性基准 | 训练/测试（主战场） |
| Custom Ambiguity Benchmark (CAB) | 48 场景、2,400 指令（无歧义/有歧义各 1,200） | 属性级碰撞，测主动澄清/选择性预测 | 训练/测试（自建） |

### 实现细节

- **$\pi_{fast}$**: [[OpenVLA]]-7B（与所有 baseline 同主干，隔离"验证 grounding 管线"的贡献）
- **$\pi_{slow}$**: [[Qwen2.5-VL]]-Instruct-7B，**异步、每 episode 仅一次**，不增加每步推理成本
- **感知**: [[YOLO-World]]（L）开放词表检测 + [[Metric3D]] v2 深度，原样使用，所有方法感知能力一致
- **微调**: 全部从公开预训练权重起，端到端微调；歧义监督由公式 12 的离线时空对齐管线产生 $(s_t,e^+)$
- **优化**: AdamW、lr $2\times10^{-5}$、batch 128、$\lambda=0.1$、$\tau=0.07$；4× NVIDIA A6000

### 关键实验结论

- **鲁棒性（Table 2/7）**: LIBERO-Plus 总分 85.5%（SOTA），Robot 扰动 30.3→71.5，相机视角 1.2→85.1，全维更均衡。
- **语言忽视（Fig 2、Table 4/8）**: 以 $g_t$ 为条件同时拿到最高成功+最低忽视；grounding 失败 41.2→6.3%；忽视分数约降 3-4 倍。
- **检索（Table 4）**: Recall@1 0.41→0.71，且随候选集增大优势仍在（InfoNCE 下界佐证）。
- **歧义/校准（Table 3、Fig 3）**: AUROC 0.81、主动澄清 0.09→0.81、ECE 14.3→4.6，且不伤无歧义成功率。
- **效率（Table 5）**: 单步 ≈96-107ms（约 9-10 FPS），GSM/SACA 溢价小，可闭环。
- **消融**: 去 $\pi_{slow}$（用 $L$ 直查）、去 GSM（用 patch token）、去 $\mathcal{L}_{\text{GAC}}$ 均显著掉点 → 四组件协同。

---

## 批判性思考

### 优点
1. **从结构上根治"语言忽视"且有理论支撑**: 验证瓶颈让 $a_t\perp L\mid(g_t,q_t)$ 是架构强制而非统计期望，配 Proposition 1（语言影响）、Theorem 1（InfoNCE 下界）、Proposition 2（Lipschitz 鲁棒）三条命题，实验现象（Fig 2、Table 2/4）都能对上理论项，论证闭环漂亮。
2. **公平对照设计严谨**: 每步推理栈与 baseline 同为 OpenVLA-7B，慢规划异步、感知器原样使用、监督弱且含噪——把增益干净地归因到"验证 grounding 机制"而非堆容量或更干净标注。
3. **把"歧义检测/主动澄清"做成内生能力**: 注意力熵直接给出校准的弃权信号，且不牺牲无歧义成功率，对安全可靠的具身智能很有价值；CAB 也是一个清晰可量化的新探针。

### 局限性
1. **纯仿真 + 受控基准，缺真机**: 全部在 LIBERO-Plus 仿真与自建 CAB 上，未触及真实遮挡、深度噪声、传感缺陷；GSM/SACA/熵弃权能否迁移到物理机器人仍未解（作者自陈）。
2. **强依赖现成感知/VL 组件**: grounding 质量被 YOLO-World、Metric3D、大 VL 主干的误差上限锁死，检测漏/错会直接导致 grounding 失败；附录 §14 的模板解析甚至退化为"最大检测置信度"启发式，削弱了"验证"的严谨性。
3. **歧义场景过窄 + 单一澄清动作**: CAB 只测属性级碰撞与单个 `[CLARIFY]`，不覆盖语用/时序/多步指代等真实歧义；系统复杂度也明显上升（四组件+检测+深度），对更大主干/高分辨率输入需进一步优化。
4. **未开源 + 部分细节缺失**: 截至 v1 未见代码；GAC 的相似度函数具体形式、$g_t$ 投影细节、$\pi_{slow}$ 模板词表覆盖度等关键工程细节描述偏简。

### 潜在改进方向
1. **真机验证 + 鲁棒感知**: 在含噪深度/遮挡的真实平台上验证，并把检测/深度不确定性显式注入 SACA 熵，做端到端可微的 grounding 而非依赖外部检测器硬选。
2. **更丰富的歧义与多轮澄清**: 把 CAB 扩到语用/时序/关系指代，支持多轮澄清对话而非单 token，并研究澄清动作的成本-收益权衡。
3. **闭环验证而非一次性规划**: 当前 $\pi_{slow}$ 每 episode 仅调用一次、grounding 失败才重调，长程任务可探索按子目标进度自适应重规划；并把"最大置信度"模板解析换成可学习的验证函数。

### 可复现性评估
- [ ] 代码开源（v1 未见公开仓库；仅项目主页 https://nstrndrbi.github.io/ProGAL）
- [ ] 预训练模型（未声明 release）
- [x] 训练细节较完整（附录 Table 6 给出主干、检测/深度模型、lr/batch/$\lambda$/$\tau$、GPU）
- [x] 数据集可获取（LIBERO-Plus 公开；CAB 为自建，统计与协议在 Table 1 公开）

---

## 速查卡片

> [!summary] ProGAL-VLA: Grounded Alignment through Prospective Reasoning
> - **核心**: 动作前先把语言推出的符号子目标"验证绑定"到 3D 实体（验证瓶颈），用 GAC 对比损失强化绑定，注意力熵当歧义信号。
> - **方法**: $\pi_{slow}$(Qwen2.5-VL,异步) 出符号模板 → GSM(YOLO-World+Metric3D) 建 3D 实体图 → SACA 交叉注意力验证出 $g_t$ → $\pi_{fast}$(OpenVLA-7B) 只对 $g_t$ 动作；理论上 $a_t\perp L\mid(g_t,q_t)$。
> - **结果**: LIBERO-Plus 鲁棒总分 17.3→85.5，Robot 扰动 30.3→71.5；语言忽视降 3-4×；检索 R@1 0.41→0.71；CAB AUROC 0.52→0.81、澄清 0.09→0.81。
> - **代码**: 未公开（项目主页 https://nstrndrbi.github.io/ProGAL）

---

*笔记创建时间: 2026-06-29*
