---
title: "Adaptive Action Chunking at Inference-time for Vision-Language-Action Models"
method_name: "AAC"
authors: [Yuanchang Liang, Xiaobo Wang, Kai Wang, Shuo Wang, Xiaojiang Peng, Haoyu Chen, David Kim Huat Chua, Prahlad Vadakkepat]
year: 2026
venue: CVPR
tags: [VLA, action-chunking, inference-time, action-entropy, flow-matching, GR00T]
zotero_collection: _inbox
image_source: online
arxiv_html: https://arxiv.org/html/2604.04161v2
created: 2026-06-29
---

# Adaptive Action Chunking at Inference-time for Vision-Language-Action Models

## 元信息

| 项目 | 内容 |
|------|------|
| 作者 | Yuanchang Liang, Xiaobo Wang, Kai Wang, Shuo Wang, Xiaojiang Peng, Haoyu Chen, David Kim Huat Chua, Prahlad Vadakkepat |
| 机构 | 新加坡国立大学、深圳理工大学(SUAT)、深信服科技、明略科技、深圳技术大学、香港城市大学 |
| 会议 | CVPR 2026 |
| 类别 | VLA（Vision-Language-Action） |
| 日期 | 2026-04（arXiv v2） |
| 项目主页 | https://lance-lot.github.io/adaptive-chunking.github.io/ |
| 链接 | [arXiv](https://arxiv.org/abs/2604.04161) / [Code](https://lance-lot.github.io/adaptive-chunking.github.io/) |

---

## 一句话总结

> 在推理阶段用[[Action Entropy|动作熵]]作为线索，对扩散式 [[VLA]] 模型逐步**自适应地决定动作块大小**——熵高就执行短块多重规划、熵低就执行长块保持一致性，无需重训练或改架构即可全面提升操作成功率。

---

## 核心贡献

1. **指出固定块长的弊端**: 系统分析现有 [[Action Chunking|动作分块]]方法，论证对于 diffusion-based [[VLA]] 模型，推理阶段的**最优块大小随任务、随策略、甚至随单条 episode 内的阶段而变**，固定经验值（如 GR00T 的 16、$\pi_0$ 的 16/25、SmolVLA 的 10）是次优且不可扩展的。
2. **提出 AAC 算法**: 一个**简单却有效**的推理时策略，用动作熵（连续控制的高斯微分熵 + 离散夹爪的香农熵）作为不确定性线索，**动态选取每个动作块的最优块大小 $h^*$**，在一致性(consistency)与响应性(reactivity)之间取得平衡。
3. **即插即用、跨骨干、跨本体**: AAC **不引入任何训练或结构改动**，仅在推理时计算熵，可直接套到任意带 flow-matching/diffusion 动作头的 VLA（本文在 [[GR00T N1.5]] 与 [[π0.5|$\pi_{0.5}$]] 上验证），在 RoboCasa、LIBERO、LIBERO-Pro 仿真与三个真实任务上均超过 SOTA。

---

## 问题背景

### 要解决的问题
[[Action Chunking|动作分块]]（在每个观测点一次性预测并执行一段动作序列、期间不重规划）是提升机器人操作成功率的关键技术，因为它能减少 compounding error。但**块大小 $h$ 怎么定**是个两难：
- **大块** → 期间不看新观测，**响应性差**（reactivity 低），无法及时纠错；
- **小块** → 频繁重规划，块与块之间不连续，引发 **mode-jumping、抖动(jerky)**，损害时序一致性与推理效率。

如何在推理时**自适应**地选出最优块大小，是平衡 reactivity 与 consistency 的迫切需求。

### 现有方法的局限
作者把已有方案分两类，逐一指出短板：
1. **固定块 + 后处理融合**: [[ACT]] 用 EMA 融合当前与历史预测增强一致性；[[BID]] 用搜索从多候选块里挑最优（backward coherence + forward contrast）；TV-BID 进一步用相邻块分布的 Total Variation Distance 自适应这两个分数；SGAC 用首动作余弦相似度选择性更新。**但它们整条 episode 仍用固定块大小，依赖大量经验调参**。
2. **学习式自适应块**: Gou et al. 用 regret value function 决定"单动作 vs 整块"，但手工价值函数**难以跨任务泛化**；Chen et al. 学一个预测块大小的模块，但**依赖只有仿真才有的 task-specific reward**，真实部署受限。

如 Figure 1 所示，[[GR00T N1.5]] 在 RoboCasa 不同任务上的成功率对块大小**强依赖**，且各任务最优值不同——固定值天然次优。

### 本文的动机
**动作熵反映预测动作的不确定性**：熵低 → 动作可靠；熵高 → 动作质量差。由此自然导出策略——**通过维持较低动作熵来选块**：高熵时执行小块、多重规划以提升 reactivity；低熵时执行长块以增强一致性与效率。关键是动作熵**只需推理时多采样几条候选块就能估计**，无需额外训练或奖励信号，因而具备跨任务、跨本体的鲁棒可扩展性。

---

## 方法详解

### 模型架构

AAC 本身**不是一个新模型**，而是一个套在 diffusion-based VLA 上的**推理时块大小决策算法**。其基座以 [[GR00T N1.5]] 为代表：
- **输入**: 多视角 RGB 观测 + 语言指令 + 机器人本体状态
- **Backbone**: [[GR00T N1.5]] = 一个 [[VLM]]（[[Eagle-2]]）抽取视觉-语言特征 $\phi_t$ 作为条件 + 一个 [[Diffusion Transformer|DiT]] 动作头 $\mathbf{V}_\theta$ 做 [[Flow Matching|流匹配]]
- **AAC 核心**: 用[[Action Entropy|动作熵]]在推理时逐步决定要执行的动作块长度 $h^*$
- **输出**: 自适应长度的 [[Action Chunking|动作块]] $\mathbf{A}_t=\{\mathbf{a}_t,\dots,\mathbf{a}_{t+H-1}\}$，每步只执行前 $h^*$ 个动作
- **可移植性**: 适用于一切带 diffusion/flow-matching 动作头的 VLA（已在 $\pi_{0.5}$ 验证）

### 核心模块

#### 模块1: Flow-Matching 动作头（基座，来自 GR00T N1.5）

**设计动机**: 用[[Flow Matching|流匹配]]建模连续动作分布，训练一个速度场把噪声推向真值动作块。AAC 复用它，但**多采样**以估计动作分布的不确定性。

**具体实现**:
- 给定真值动作块 $\mathbf{A}_t\in\mathbb{R}^{d\times H}$（$d$=动作自由度，$H$=训练 horizon）、流匹配时刻 $\tau\in[0,1]$、噪声 $\epsilon\sim\mathcal{N}(0,I)$，带噪块为 $\mathbf{A}_t^{(\tau)}=\tau\mathbf{A}_t+(1-\tau)\epsilon$；
- 动作头 $\mathbf{V}_\theta(\phi_t,\mathbf{A}_t^{(\tau)},\mathbf{q}_t)$（$\mathbf{q}_t$ 为状态嵌入）逼近流向量场 $\epsilon-\mathbf{A}_t$（公式1）；
- 推理时从 $\mathbf{A}_t^{(0)}\sim\mathcal{N}(0,I)$ 起，做 $K$ 步前向欧拉积分去噪生成动作块。

#### 模块2: Action Entropy 估计（AAC 的不确定性度量）

**设计动机**: 用熵量化预测动作的"无序/随机/不确定"程度。机器人动作空间含**连续控制**（平移、旋转）与**离散控制**（夹爪开合），需分别建模。

**具体实现**（以 7-DOF 机械臂 + 夹爪为例，每步动作 $\mathbf{a}_i$ 含 3-DOF 平移 $[\Delta x,\Delta y,\Delta z]$、3-DOF 旋转 $[\Delta R_x,\Delta R_y,\Delta R_z]$、1-DOF 夹爪 $G\in\{0,1\}$）:
- **离散夹爪**用香农熵（公式2），概率 $p(a)=c/N$（$N$ 条采样候选中"闭合"出现 $c$ 次）；
- **连续平移/旋转**用高斯微分熵（公式3），需估计协方差矩阵 $\Sigma_t$；
- 为估计 $p(a)$ 与 $\Sigma_t$，**并行采样 $N$ 条候选动作块**（默认 $N=20$，batched 并行，开销可忽略）。

#### 模块3: 自适应块大小决策（AAC 的核心准则）

**设计动机**: 把"维持低熵"操作化——找到**平均动作熵随块长增长的最大跳变点**，即熵开始显著上升处之前的块长就是高置信可执行长度。

**具体实现**:
- 对块长 $h$ 计算平均动作熵 $\overline{E}_h$（公式4，对块内 $h$ 步、平移/旋转/夹爪三类熵求平均）；
- 最优块长 $h^*$ 取**相邻块长平均熵的最大差分点** $\arg\max_h(\overline{E}_{h+1}-\overline{E}_h)$，并以下界 $\xi$ 兜底（公式5）；
- $\xi$ 由**最小动作幅度约束**给出（附录 A，公式6–10）：累计平移/旋转/夹爪运动幅度 $m(l)$ 首次超过阈值 $\alpha$（实验取 3）的最短块长，避免选出"几乎不动"的过短块、保证时序一致性与计算效率。
- 选出 $h^*$ 后**只执行前 $h^*$ 个动作**，到新观测再重新决策（见 Algorithm 1 与 Figure 2）。

### 关键公式与机制

#### 公式1: [[Flow Matching|流匹配]]损失（基座训练目标）

$$
\mathcal{L}_{\mathrm{fm}}(\theta)=\mathbb{E}_{\tau}\!\left[\left\|\mathbf{V}_{\theta}(\phi_{t},\mathbf{A}_{t}^{(\tau)},\mathbf{q}_{t})-(\epsilon-\mathbf{A}_{t})\right\|^{2}\right]
$$

**含义**: 训练 DiT 动作头 $\mathbf{V}_\theta$ 预测把带噪动作块推回真值的流向量场 $\epsilon-\mathbf{A}_t$。

**符号说明**:
- $\mathbf{V}_\theta$: 动作头网络（DiT）；$\phi_t$: VLM 视觉-语言条件特征；$\mathbf{q}_t$: 状态嵌入
- $\mathbf{A}_t^{(\tau)}=\tau\mathbf{A}_t+(1-\tau)\epsilon$: 流匹配时刻 $\tau$ 的带噪动作块；$\epsilon\sim\mathcal{N}(0,I)$
- $\mathbb{E}_\tau$: 对流匹配时刻 $\tau\in[0,1]$ 取期望

#### 公式2: [[Action Entropy|离散动作熵]]（夹爪控制）

$$
E_{\mathrm{dis}}=-\sum_{a\in\mathcal{A}}p(a)\log\big(p(a)\big)
$$

**含义**: 对离散控制（如夹爪开/合）用香农熵度量不确定性。

**符号说明**:
- $\mathcal{A}$: 离散动作空间；$p(a)$: 策略网络输出某动作的概率，由 $N$ 条采样估计 $p(a)=c/N$

#### 公式3: [[Action Entropy|连续动作熵]]（高斯微分熵，平移/旋转）

$$
E_{t}=\frac{1}{2}\log\big[(2\pi e)^{d}\,\det(\Sigma_{t})\big]
$$

**含义**: 对 $d$ 维连续控制（平移、旋转各 3 维）用高斯微分熵度量不确定性。

**符号说明**:
- $\det(\Sigma_t)$: 由 $N$ 条采样候选动作估得的协方差矩阵的行列式；$d$: 维度
- 平移熵 $E_t^i$、旋转熵 $E_r^i$ 均按此式逐步计算

#### 公式4: 块内[[Action Entropy|平均动作熵]]

$$
\overline{E}_{h}=\frac{1}{h}\sum_{i=t}^{t+h-1}\sum_{j\in\{t,r,g\}}E_{j}^{i}
$$

**含义**: 块长为 $h$ 时，对块内 $h$ 个时间步、对平移/旋转/夹爪三类熵求和并平均，作为该块长的整体不确定性。

**符号说明**:
- $i$: 块内时间步（从 $t$ 到 $t+h-1$）；$j\in\{t,r,g\}$: 平移 / 旋转 / 夹爪三类；$E_j^i$: 第 $i$ 步第 $j$ 类的动作熵

#### 公式5: 最优块大小决策

$$
h^{*}=\max\!\Big(\operatorname*{arg\,max}_{h}\big(\overline{E}_{h+1}-\overline{E}_{h}\big),\ \xi\Big)
$$

**含义**: 取"平均动作熵随块长增长的**最大差分点**"作为高置信可执行长度，并用下界 $\xi$ 兜底，二者取较大者。

**符号说明**:
- $\arg\max_h(\overline{E}_{h+1}-\overline{E}_h)$: 平均熵跳变最剧烈处（熵在此后显著上升，说明继续延长块不再可靠）
- $\xi$: 块大小下界，由最小动作幅度约束给出（公式6），保证一致性与效率

#### 公式6–10: 最小动作幅度约束（附录 A，定义下界 $\xi$）

$$
\xi=\operatorname*{arg\,min}_{l}\big(m(l)>\alpha\big)
$$

$$
m(l)=\sum_{i\in\{t,r,g\}}m_{i}(l)
$$

$$
m_{t}(l)=\Big\|\big(\textstyle\sum_{l}\Delta x,\ \sum_{l}\Delta y,\ \sum_{l}\Delta z\big)\Big\|
$$

$$
m_{r}(l)=\Big\|\textstyle\prod_{l}\Delta q\Big\|
$$

$$
m_{g}(l)=\mathbf{1}_{\mathrm{switch}}
$$

**含义**: $\xi$ 取累计动作幅度 $m(l)$ 首次超过最小运动能量阈值 $\alpha$（实验取 3）的最短块长 $l$，避免选出"几乎不动"的过短块。

**符号说明**:
- $m_t$: 平移幅度（块起点到块长 $l$ 终点的位移欧氏范数）；$m_r$: 旋转幅度（四元数 $\Delta q$ 序列累乘后的范数）；$m_g$: 夹爪幅度（开合状态切换时为 1，否则 0；$\mathbf{1}$ 为指示函数）
- $\alpha$: 机械臂最小运动能量阈值

---

## 关键图表

<!-- 图片均使用 arXiv HTML 在线链接，未本地化 -->

### Figure 1: Effects of Action Chunk Sizes / 块大小的影响（动机图）

![Figure 1](https://arxiv.org/html/2604.04161v2/figures/PnP_diversity_bestpoints_task_name_2.png)

**说明**: 在 RoboCasa Kitchen 上用 [[GR00T N1.5]] 跑多个任务，成功率对动作块大小**强依赖**且各任务最优值不同。这是全文动机的实证基础——**经验固定值是次优且不可扩展的**。

### Figure 2: Overview of AAC / 方法总览

![Figure 2](https://arxiv.org/html/2604.04161v2/figures/method_1108.png)

**说明**: AAC 全流程。VLM 抽条件特征 → DiT 动作头并行采样 $N$ 条候选块 → 计算连续+离散[[Action Entropy|动作熵]] → 由公式 4/5 选出当前观测下的最优块长 $h^*$ → 仅执行前 $h^*$ 步再重规划。对比图右侧示意 AAC 与 vanilla GR00T（固定块）的差别：**全程无额外训练或结构改动**。

### Figure 3: Rollout of Chunk Sizes / 块大小随阶段的演化

![Figure 3](https://arxiv.org/html/2604.04161v2/figures/chunk_size_decision.png)

**说明**: 一条 episode 内 AAC 选出的块大小序列，**与任务语义阶段高度吻合人类直觉**：搬运(transportation)阶段用大块快速粗动，关键抓取/操作(critical manipulation)阶段用小块、高频观测做精细控制。佐证"熵作为线索"的合理性。

### Figure 4: Distribution of Chunk Size Decisions / 块大小决策分布

![Figure 4](https://arxiv.org/html/2604.04161v2/figures/simstep_libero_spatial_task_0.png)

**说明**: LIBERO-Spatial 首个任务（"Pick up the black bowl next to the cookie box and place it on the plate"）的块大小分布热力图，红色曲线为各观测时刻的平均块大小。直观展示 AAC 在不同决策时刻自适应缩放块长的整体模式。

### Figure 5: Real-world Execution Examples / 真实任务执行示例

![Figure 5](https://arxiv.org/html/2604.04161v2/figures/real_world_tasks_2.png)

**说明**: 三个真实世界任务（香蕉抓放、急停按钮按压、长程"取玩具放抽屉并关抽屉"）的 AAC 执行帧序列。

### Figure 6: AAC Improves Accuracy and Safety / 精度与安全性提升

![Figure 6 baseline](https://arxiv.org/html/2604.04161v2/figures/banana_baseline.png)
![Figure 6 ours](https://arxiv.org/html/2604.04161v2/figures/banana_ours.png)

**说明**: 左=vanilla GR00T 因早期观测点预测了低质量动作，**夹爪撞到桌面**；右=AAC 通过动态块大小过滤高熵/不确定动作，**平滑停在合适的最低点**。体现 AAC 对高精度操作的安全性价值。

### Table 1: Main Results on RoboCasa & LIBERO / 主结果

| Method | Relo. | Cont. | Rot. | Button | RoboCasa Avg. | Spatial | Long | Object | Goal | LIBERO Avg. |
|--------|-------|-------|------|--------|---------------|---------|------|--------|------|-------------|
| GR00T (Default $h$=16) | 42.1 | 80.3 | 57.6 | 80.3 | 59.7 | 93.6 | 88.8 | 97.2 | 96.8 | 94.1 |
| GR00T ($h$=2) | 29.6 | 57.0 | 59.4 | 64.7 | 47.0 | 93.6 | 81.8 | 93.6 | 91.6 | 90.2 |
| GR00T ($h$=4) | 37.9 | 71.3 | 61.2 | 78.7 | 56.2 | 94.0 | 85.8 | 95.4 | 95.2 | 92.6 |
| GR00T ($h$=8) | 42.8 | 80.3 | 62.6 | 81.7 | 61.2 | 92.0 | 93.4 | 97.8 | 95.4 | 94.7 |
| GR00T ($h$=12) | 42.9 | 79.5 | 60.0 | 79.3 | 60.2 | 93.0 | 91.0 | 97.6 | 95.2 | 94.2 |
| **GR00T+AAC (Ours)** | **44.4** | **82.2** | 61.4 | 81.3 | **62.0** | **94.4** | 92.8 | **98.6** | 94.2 | **95.0** |

**说明**: AAC 在 RoboCasa 24 任务平均 **62.0%**（较默认 16 块的 59.7% +2.3%，Rotation 子集 57.6%→61.4% 提升最显著，因其需精细旋转控制）；LIBERO 四套件平均 95.0%（较 94.1% +0.9%，**最难的 LIBERO-Long +4%**）。同时表明**固定块各有所长**（Spatial 在 $h$=4 最好、Goal 在 $h$=16 最好），AAC 免去手工调参且**平均上稳超所有固定值**。

### Table 2: $\pi_{0.5}$ Backbone on LIBERO / 换骨干验证可移植性

| Method | Spatial | Long | Object | Goal | Avg. |
|--------|---------|------|--------|------|------|
| $\pi_{0.5}$ | 98.5 | 92.5 | **98.7** | **98.1** | 97.0 |
| **$\pi_{0.5}$+AAC (Ours)** | **99.1** | **95.2** | **99.2** | 98.0 | **97.9** |

**说明**: 把 AAC 直接套到 [[π0.5|$\pi_{0.5}$]] 骨干上，平均 97.0%→97.9%，Long 提升最多（+2.7%）。证明 AAC 对 flow-matching 动作头**骨干无关、即插即用**。

### Table 3: LIBERO-Pro (OOD 位置扰动) / 分布外鲁棒性

| Perturbation level | ×0.2 | ×0.3 | ×0.4 | Avg. |
|--------------------|------|------|------|------|
| GR00T | 11.3 | 0.4 | 0.0 | 3.9 |
| **GR00T+AAC (Ours)** | **17.9** | **0.9** | 0.0 | **6.3** |
| $\pi_{0.5}$ | 53.2 | 29.9 | 9.5 | 30.9 |
| **$\pi_{0.5}$+AAC (Ours)** | **57.4** | **35.3** | **11.8** | **34.8** |

**说明**: 在 [[LIBERO-Pro]] 的位置扰动子集上，扰动越大成功率越低，但 AAC 在两种骨干上均一致优于对应基线（$\pi_{0.5}$ 平均 30.9%→34.8%），说明用熵动态调节执行 horizon 提升了 OOD 鲁棒性。

### Table 4: Number of Samples vs Success & Inference Time / 采样数消融

| #Samples | 1 | 5 | 10 | **20** | 30 | 40 |
|----------|-----|-----|-----|--------|------|------|
| Succ. (%) | 94.1 | 94.7 | 94.4 | **95.0** | 95.0 | 95.5 |
| Time (ms) | 83.0 | 83.5 | 84.3 | **106.0** | 136.5 | 157.0 |

**说明**: 采样数 1→20 成功率持续升 0.9%（94.1→95.0），20 之后收益边际化。耗时上 ≤10 样本几乎无影响，**默认 20 样本仅引入约 20ms 延迟**（106.0 vs 83.0），因候选块可 batched 并行生成，计算开销可忽略；高端 GPU 可进一步缩小差距。

### Table 5: Real-world Applications / 真实任务成功率

| Task | Banana | Button | Drawer | Avg. |
|------|--------|--------|--------|------|
| GR00T | 70.0 | 65.0 | 65.0 | 67.0 |
| **GR00T+AAC** | **90.0** | **75.0** | **80.0** | **82.0** |

**说明**: 三个真实任务（Realman 单臂 + Mycobot 夹爪，每任务 50 条遥操作示范、20 次试验），AAC 平均 **67.0%→82.0%（+15%）**。香蕉任务靠更准的夹爪朝向对齐、按钮任务靠更好的近分布外泛化、长程任务靠取玩具阶段的自适应块长受益最大。

### Table 6: Complete RoboCasa Results (vanilla vs AAC) / RoboCasa 全 24 任务（附录 B）

| Task Group | Task | GR00T | GR00T+AAC |
|------------|------|-------|-----------|
| Relocation | Counter to Cab | 57.0 | **59.0** |
| | Counter to Stove | **41.0** | 40.0 |
| | Sink to Counter | 46.0 | **52.0** |
| | Stove to Counter | 60.0 | **63.0** |
| | Counter to Sink | 26.0 | **28.0** |
| | Counter to Microwave | **33.0** | 30.0 |
| | Cab to Counter | **26.0** | 25.0 |
| | Microwave to Counter | 27.0 | **36.0** |
| | Coffee Setup Mug | 38.0 | **39.0** |
| | Coffee Serve Mug | 67.0 | **72.0** |
| Container | Open Single Door | 62.0 | **73.0** |
| | Close Single Door | **94.0** | 92.0 |
| | Close Double Door | 56.0 | **62.0** |
| | Open Double Door | **89.0** | 88.0 |
| | Open Drawer | **81.0** | 80.0 |
| | Close Drawer | **100.0** | 98.0 |
| Rotation | Turn Sink Spout | **69.0** | 66.0 |
| | Turn on Sink Facet | 85.0 | **91.0** |
| | Turn off Sink Facet | 74.0 | **80.0** |
| | Turn on Stove | 39.0 | **49.0** |
| | Turn off Stove | 21.0 | 21.0 |
| Button | Coffee Press Button | **94.0** | 93.0 |
| | Turn on Microwave | 66.0 | **68.0** |
| | Turn off Microwave | 81.0 | **83.0** |
| | **Relocation Avg.** | 42.1 | **44.4** |
| | **Container Avg.** | 80.3 | **82.2** |
| | **Rotation Avg.** | 57.6 | **61.4** |
| | **Button Avg.** | 80.3 | **81.3** |
| | **24 Tasks Avg.** | 59.7 | **62.0** |

**说明**: 逐任务对比，AAC 在多数任务上提升，组平均全部不降，整体 59.7%→62.0%。少数任务略降（如 Close Door/Drawer 等已近饱和的简单任务）说明仍有块大小选择的优化空间。

### Table 7: RoboCasa with Different Fixed Chunk Sizes / 固定块大小逐任务（附录 B）

| Task Group | Task | Fixed-2 | Fixed-4 | Fixed-8 | Fixed-12 | Fixed-16 |
|------------|------|---------|---------|---------|----------|----------|
| Relocation | Counter to Cab | 51.0 | **68.0** | 61.0 | 57.0 | 57.0 |
| | Counter to Stove | 31.0 | 40.0 | **46.0** | 45.0 | 41.0 |
| | Sink to Counter | 35.0 | 44.0 | **56.0** | 50.0 | 46.0 |
| | Stove to Counter | 59.0 | 62.0 | **69.0** | 64.0 | 60.0 |
| | Counter to Sink | 6.0 | 10.0 | 15.0 | 19.0 | **26.0** |
| | Counter to Microwave | 7.0 | 12.0 | 29.0 | 23.0 | **33.0** |
| | Cab to Counter | 18.0 | 18.0 | 23.0 | 23.0 | **26.0** |
| | Microwave to Counter | 27.0 | **42.0** | 28.0 | 36.0 | 27.0 |
| | Coffee Setup Mug | 31.0 | 39.0 | 40.0 | **46.0** | 38.0 |
| | Coffee Serve Mug | 31.0 | 44.0 | 61.0 | 66.0 | **67.0** |
| Container | Open Single Door | 15.0 | 41.0 | 61.0 | **62.0** | **62.0** |
| | Close Single Door | 84.0 | 89.0 | 87.0 | **95.0** | 94.0 |
| | Close Double Door | 53.0 | 62.0 | **73.0** | 58.0 | 56.0 |
| | Open Double Door | 44.0 | 73.0 | 81.0 | 84.0 | **89.0** |
| | Open Drawer | 51.0 | 64.0 | **81.0** | 79.0 | **81.0** |
| | Close Drawer | 95.0 | 99.0 | 99.0 | 99.0 | **100.0** |
| Rotation | Turn Sink Spout | **70.0** | 69.0 | 69.0 | 67.0 | 69.0 |
| | Turn on Sink Facet | 84.0 | **87.0** | 85.0 | 83.0 | 85.0 |
| | Turn off Sink Facet | 82.0 | 78.0 | **85.0** | 80.0 | 74.0 |
| | Turn on Stove | 42.0 | **52.0** | **52.0** | 47.0 | 39.0 |
| | Turn off Stove | 19.0 | 20.0 | 22.0 | **23.0** | 21.0 |
| Button | Coffee Press Button | 76.0 | 89.0 | **98.0** | **98.0** | 94.0 |
| | Turn on Microwave | 47.0 | 51.0 | 61.0 | 60.0 | **66.0** |
| | Turn off Microwave | 71.0 | **96.0** | 86.0 | 80.0 | 81.0 |
| | **Relocation Avg.** | 29.6 | 37.9 | 42.8 | **42.9** | 42.1 |
| | **Container Avg.** | 57.0 | 71.3 | 80.3 | 79.5 | **80.3** |
| | **Rotation Avg.** | 59.4 | 61.2 | **62.6** | 60.0 | 57.6 |
| | **Button Avg.** | 64.7 | 78.7 | **81.7** | 79.3 | 80.3 |
| | **24 Tasks Avg.** | 47.0 | 56.2 | **61.2** | 60.2 | 59.7 |

**说明**: 单看固定块，**最优值随任务剧烈漂移**（Counter-to-Sink 偏好 16、Turn-Sink-Spout 偏好 2、多数 Relocation 偏好 8）；全局最佳固定值为 $h$=8（61.2%），仍低于 AAC 的 62.0%。这是"固定块不可扩展、需自适应"的最直接证据。

---

## 实验

### 数据集 / 基准

| 基准 | 规模 | 特点 | 用途 |
|------|------|------|------|
| [[RoboCasa]] Kitchen | 24 厨房任务（分 4 组：Relocation 10 / Container 6 / Rotation 5 / Button 3）；100 条 MimicGen 生成轨迹微调 | 单臂、多样化厨房操作 | 训练/测试 |
| [[LIBERO]] | 4 套件（Spatial/Object/Goal/Long），每套 10 任务、每任务 50 条遥操作示范 | 单臂，考空间/物体/目标/长程 | 训练/测试 |
| [[LIBERO-Pro]] | 位置扰动子集（×0.2/×0.3/×0.4） | OOD 鲁棒性评测 | 测试 |
| 真实世界 | 3 任务，每任务 50 条遥操作示范、20 次试验、每试 600 步上限 | Realman 单臂 + Mycobot 夹爪，腕部+侧面双 RGB | 训练/测试 |

### 实现细节

- **基座**: 从公开 [[GR00T N1.5]] checkpoint 微调；**只训扩散动作头**（DiT backbone + state/action encoder & decoder），**冻结 [[Eagle-2]] VLM 与 vision projector**
- **硬件/训练**: 8× NVIDIA A800，混合精度；真实任务每任务训 300 epoch、每卡 batch 32
- **AAC 超参**: 默认采样数 $N=20$；最小动作幅度阈值 $\alpha=3$；推理 batched 并行，约 +20ms 延迟
- **对比设置**: baseline 为 vanilla GR00T（训练/执行均用固定 horizon 16）；并与多种固定块 $h\in\{2,4,8,12,16\}$ 对比

### 关键实验结论

- **仿真主结果**: RoboCasa 24 任务 59.7%→62.0%（Rotation +3.8%），LIBERO 94.1%→95.0%（Long +4%）。
- **固定 vs 自适应**: 各任务最优固定块各异、全局最佳固定值(8)仍逊于 AAC，证明自适应必要性。
- **跨骨干**: $\pi_{0.5}$ 上 97.0%→97.9%，即插即用。
- **OOD**: LIBERO-Pro 位置扰动上两骨干均稳超基线。
- **效率**: 20 样本仅 +20ms，开销可忽略。
- **真实世界**: 平均 67%→82%（+15%），并定性展示避免撞桌、提升安全性。

---

## 批判性思考

### 优点
1. **零训练、即插即用**: 仅在推理时多采样算熵，不改架构、不重训、不依赖奖励信号，跨骨干（GR00T/$\pi_{0.5}$）与跨本体设计上天然可移植，落地门槛极低。
2. **动机与机制自洽且可解释**: Figure 1 实证"固定块次优"、Figure 3/Table 7 显示块长决策契合任务语义阶段，"熵→不确定性→块长"的逻辑链清晰且有定性+定量支撑。
3. **效率代价小**: 候选块 batched 并行，默认 20 样本仅 ~20ms 延迟，把"多采样估熵"的额外成本压到可忽略。

### 局限性
1. **绝对增益偏小**: 仿真上多为 +0.9%~+2.3%（LIBERO 已近饱和），真实世界 +15% 但仅 3 个桌面任务、20 试，统计置信与任务覆盖（长程、接触丰富、双臂/灵巧手）有限。
2. **关键超参靠经验**: $\alpha=3$、$N=20$、"最大差分点"准则均经验设定，缺少对 $\alpha$/差分准则的系统敏感度分析；高斯微分熵假设连续动作近似高斯，对多峰动作分布可能失真。
3. **熵估计依赖采样质量**: $\Sigma_t$/$p(a)$ 由有限采样估计，样本不足时（Table 4 中 $N$=1/5 反而 94.1/94.7 波动）熵估计噪声大，且其有效性强绑定 diffusion/flow-matching 这类可多采样的动作头，对自回归 token 式 VLA 不直接适用。

### 潜在改进方向
1. 把 $\alpha$、差分阈值、采样数做成自适应/可学习，或引入更鲁棒的不确定性估计（如非高斯/conformal）替代高斯微分熵。
2. 扩展到双臂、灵巧手、长程接触丰富任务与更多真实试验，量化"熵→块长→成功率"的因果而非相关。
3. 把熵线索与 BID/SGAC 等块内动作选择结合，做"块大小 + 块内动作"联合自适应，进一步压低 mode-jumping。

### 可复现性评估
- [x] 代码开源（项目主页声明公开 source code 与视频）
- [x] 基座模型可获取（GR00T N1.5 / $\pi_{0.5}$ 公开 checkpoint）
- [x] 训练细节较完整（微调对象、硬件、batch、epoch、$N$、$\alpha$ 均给出）
- [x] 数据集可获取（RoboCasa/LIBERO/LIBERO-Pro 公开；真实数据为自采）

---

## 速查卡片

> [!summary] AAC: Adaptive Action Chunking at Inference-time for VLA
> - **核心**: 推理时用动作熵作线索，逐步自适应决定动作块大小 $h^*$——熵高执行短块多重规划、熵低执行长块保一致性；无需训练或改架构。
> - **方法**: 并行采样 $N$=20 条候选块 → 算连续(高斯微分熵)+离散(夹爪香农熵)动作熵 → 取平均熵最大差分点(公式5)、最小动作幅度阈值 $\xi$($\alpha$=3)兜底 → 仅执行前 $h^*$ 步再重规划。基座 GR00T N1.5(flow-matching DiT)。
> - **结果**: RoboCasa 59.7→62.0、LIBERO 94.1→95.0(Long +4%)、$\pi_{0.5}$ 97.0→97.9、真实 67→82%(+15%)，仅 +20ms 延迟。
> - **代码**: https://lance-lot.github.io/adaptive-chunking.github.io/

---

*笔记创建时间: 2026-06-29*
