---
title: "AtomicVLA: Unlocking the Potential of Atomic Skill Learning in Robots"
method_name: "AtomicVLA"
authors: [Likui Zhang, Tao Tang, Zhihao Zhan, Xiuwei Chen, Zisheng Chen, Jianhua Han, Jiangtong Zhu, Pei Xu, Hang Xu, Hefeng Wu, Liang Lin, Xiaodan Liang]
year: 2026
venue: CVPR
tags: [VLA, mixture-of-experts, atomic-skill, continual-learning, long-horizon, planning-and-execution]
zotero_collection: _inbox
image_source: online
arxiv_html: https://arxiv.org/html/2603.07648v1
created: 2026-06-29
---

# AtomicVLA: Unlocking the Potential of Atomic Skill Learning in Robots

## 元信息

| 项目 | 内容 |
|------|------|
| 作者 | Likui Zhang, Tao Tang, Zhihao Zhan, Xiuwei Chen, Zisheng Chen, Jianhua Han, Jiangtong Zhu, Pei Xu, Hang Xu, Hefeng Wu, Liang Lin, Xiaodan Liang |
| 机构 | 中山大学、鹏城实验室、引望智能技术（Yinwang Intelligent Technology） |
| 会议 | CVPR 2026 |
| 类别 | VLA（Vision-Language-Action） |
| 日期 | 2026-03（arXiv v1） |
| 项目主页 | https://zhanglk9.github.io/atomicvla-web/ |
| 链接 | [arXiv](https://arxiv.org/abs/2603.07648) / [PDF](https://arxiv.org/pdf/2603.07648) |

---

## 一句话总结

> 把长程任务拆成"原子技能"，用技能引导的混合专家（SG-MoE）为每个原子技能配一个专家、并以可扩展路由实现持续学习，在统一的"思考-执行"框架里同时做任务规划与动作生成，显著缓解多技能干扰与灾难性遗忘。

---

## 核心贡献

1. **统一规划-执行框架**: 提出端到端的 [[AtomicVLA]]，让同一个 [[VLM]] 通过 `[think]`/`[act]` 两个特殊 token **自适应切换**"思考（生成任务链+原子技能抽象）"与"执行（生成精细动作）"，无需外挂独立规划器，避免规划器与控制器互不感知导致的指令过时/失配。
2. **技能引导的混合专家 SG-MoE**: 在 [[π0|$\pi_0$]] 基础上构建**可扩展的原子技能专家库**——一个共享专家保留 $\pi_0$ 的泛化能力 + 多个专精单一原子技能的专家，由**以原子技能抽象（而非 token / 去噪步）为路由信号**的 skill router 稀疏激活 top-1 专家，从而让每个专家学到分布一致的技能、抑制技能间干扰。
3. **可扩展持续学习**: 提出柔性路由编码器，新技能只需**新增一个专家并扩展路由分支**（冻结已有专家），即可在几乎不损害旧技能的前提下增量学习，缓解灾难性遗忘；并配套一套基于**主轴分析（principal-axis analysis）+ InternVideo2.5** 的具身推理数据生成管线。
4. **实验全面验证**: 仿真上 LIBERO 平均超 $\pi_0$ 2.4%、LIBERO-LONG 超 10%，CALVIN 平均任务长度分别超 $\pi_0$/$\pi_{0.5}$ 0.22/0.25；真实世界长程任务与持续学习分别超基线 18.3% 与 21%。

---

## 问题背景

### 要解决的问题
现实机器人任务往往是**长程、多步、需要持续获取新技能**的，而当前 [[VLA]] 模型普遍用**单一的整体式动作解码器（monolithic action decoder）**在聚合数据上训练，导致：可扩展性差、长程任务中不同技能互相干扰、增量学新技能时灾难性遗忘。本文要回答的是：如何让一个 VLA 同时具备**高层推理/规划 + 细粒度动作生成 + 可扩展持续学习**。

### 现有方法的局限
1. **两阶段（模块化）方案**（VLM 规划器 + 独立 VLA 控制器）：规划器与控制器**缺乏相互感知**，任务协同次优；实际部署中系统延迟还会让规划器产生过时/无关的子指令。
2. **单一动作解码模块**：无论用扩散、流匹配还是离散编码，主干都只有**一个解码器**，扩展性受限；增量学新技能要全量微调，吃算力、吃数据。
3. **多技能混合训练相互干扰**：尤其是夹爪状态在不同子任务间差异大时（如"开抽屉"子任务不需要闭合夹爪），会污染其它抓取类任务，造成意外开/合夹爪。
4. **持续学习的灾难性遗忘**：直接把新技能并入技能库并重训，会显著损害已学技能。

### 本文的动机
作者主张把机器人行为分解为**可复用、语义明确的"原子技能"**（如 pick / place / open / close / turn），并用 [[Mixture-of-Experts|MoE]] 重新诠释为**技能模块化**：每个专家对应一个原子动作原语。核心假设是——**以原子技能抽象作为路由信号**，能保证"同一技能阶段的所有动作 token 都进同一个专家"，使各专家学到分布一致的技能、互不干扰；而专家库的**模块化**天然支持"只加专家不动旧专家"的无遗忘扩展。

---

## 方法详解

### 模型架构

AtomicVLA 在 [[π0|$\pi_0$]] / [[π0.5|$\pi_{0.5}$]] 之上构建，整体是一个统一的 **感知-思考-执行（perception–think–act）** 框架（见 Figure 1、Figure 2）：

- **输入**: 多视角观测 $O_t^{1:n}$ + 语言指令 $\ell$ + 机器人本体状态 $S_t$
- **Backbone**: 预训练 [[VLM]]（沿用 $\pi_0$/$\pi_{0.5}$ 的视觉-语言主干）
- **核心模块**: [[Skill-Guided Mixture-of-Experts|SG-MoE]] 技能专家库（1 个共享专家 + $K$ 个原子技能专家 + skill router）+ [[Continual Learning|可扩展技能路由]]
- **输出**: 高层任务计划 $[C_{0\text{-}k}, C_t, \sigma]$（thinking 模式，文本）或 [[Action Chunking|动作块]] $A_t$（acting 模式）
- **专家结构**: 每个技能专家用 [[Gemma]] 架构，前馈层为独立的 [[SwiGLU]] MLP（width=2048、mlp_dim=4096、depth=18、heads=8、head_dim=256）；全部随机初始化以解耦技能表征。

### 核心模块

#### 模块1: Unified Task Planning and Action Execution（统一规划与执行）

**设计动机**: 把"会想"和"会做"压进同一个策略，让模型自己决定当前该输出"计划"还是"动作"，从而避免模块解耦带来的失配。

**具体实现**:
- 引入两个特殊输出 token `[think]` 与 `[act]`。给定当前观测 $O_t^{1:n}$ 与指令 $\ell$，模型**先预测标识符**：
  - 预测到 `[think]` → 进入**思考模式**：生成任务链 $C_{0\text{-}k}$（高层计划）、当前进度 $C_t$、以及要执行的**原子技能抽象** $\sigma$。该模式通常只在任务起始或子技能切换等关键时刻触发。
  - 预测到 `[act]` → 进入**执行模式**：以最近一次 `[think]` 得到的 $\sigma$ 与本体状态 $s_t$ 为条件，产出低层动作块 $A_t$。
- 推理流程见下文 Algorithm 1：thinking 更新 `Atomic` 抽象，acting 时由 router 依据该抽象选专家再生成动作并执行，循环直到任务完成。

#### 模块2: Skill-Guided Mixture of Experts（SG-MoE 技能专家库）

**设计动机**: 让"路由"由**语义明确的原子技能**驱动，而非 token / 去噪步；保证同一技能的所有 token 走同一专家，专家专精、互不干扰，同时共享专家继承 $\pi_0$ 的泛化能力。

**具体实现**:
- **原子技能抽象嵌入**: 借鉴扩散去噪里的**噪声调度**思想，把每个原子技能抽象映射成一个标量噪声级 $\sigma\in[0,100]$，再嵌入成高维向量 $Z_\sigma$（公式1）。这种连续、结构化的嵌入空间增强了技能间的语义可分性，利于鲁棒路由。
- **技能引导动态路由**: $Z_\sigma$ 作为条件送入 skill router，得到 $K$ 个专家上的概率分布（公式2），采用**稀疏激活**——只选 top-1 专家。
- **共享专家 + 选中专家组合**: 最终动作块由共享专家与被激活专家按路由权重 $w_k$ 加权组合（公式3）。共享专家 $F_\text{share}$ 维持 $\pi_0$ 的预训练动作生成能力，专精专家 $F_k$ 负责高保真执行特定技能。
- LIBERO 与真实机器人用 $K=5$ 个技能专家；CALVIN 因任务词表更广用 $K=8$。

#### 模块3: Continual Learning with Skill Expansion（可扩展持续学习）

**设计动机**: 真实部署会不断遇到训练时没见过的新技能；直接并入重训会灾难性遗忘。

**具体实现**:
- 因为每个原子技能都被映射到**固定**的高维嵌入 $Z_\sigma$（显式语义抽象），新技能只需：**新增一个对应专家模块 + 扩展路由网络**。
- **路由初始化策略**: 扩展后的 router 从原 router 复制权重，新分支用小随机值初始化——让模型以最少微调适配扩大的技能集，同时**保住已学技能**。
- 已有专家保持不变，因此实现"高效、无干扰"的技能库扩张。

#### 模块4: Task Planning Embodied Data Generation（具身规划数据生成）

**设计动机**: 传统用 VLM 视频理解或光流分割动作序列，容易歧义、含噪、需大量人工后处理；本文要自动得到时序精确、语义可解释的原子动作边界。

**具体实现**:
- **基于主轴分析的轨迹原子分解**：分析末端执行器轨迹的关键运动学维度——平移位移 $(\Delta x,\Delta y,\Delta z)$、旋转变化 $(\Delta\text{roll},\Delta\text{pitch},\Delta\text{yaw})$ 与二值夹爪状态，对每个短动作块比较平移/旋转分量幅度找出主导运动模式，再结合夹爪状态跃变推断动作语义。例如 $z$ 坐标持续下降 + 夹爪闭合 = "pick"；平移很小 + 显著旋转 + 夹爪闭合 = "turn"。
- **阈值设定**（数据生成设置）：平移轴阈值 3 cm、旋转轴阈值 0.05 弧度、夹爪变化阈值 0.1，分析连续 5 帧的运动确定主导轴。
- **InternVideo2.5 标签精化**: 把主轴分析切出的原子段对应视频片段交给 [[InternVideo2.5]] 解读，自动校正/丰富初始标注，并对齐到完整轨迹，构造"已执行原子动作序列 + 后续高层计划"的结构化推理链。

### 关键公式与机制

#### 公式1: [[原子技能嵌入|Atomic Skill Abstract Embedding]]

$$
Z_{\sigma} = E\big(\operatorname{norm}(\log(\sigma))\big)
$$

**含义**: 把离散的原子技能先映射为一个标量"噪声级" $\sigma$，取对数并归一化后，再由嵌入函数升维成路由所用的条件向量 $Z_\sigma$，从而在连续空间里拉开不同技能的语义距离。

**符号说明**:
- $\sigma\in[0,100]$: 分配给某原子技能的标量噪声级（受扩散去噪噪声调度启发）
- $\operatorname{norm}(\cdot)$, $\log(\cdot)$: 对标量先取对数再归一化，稳定数值
- $E(\cdot)$: 嵌入函数，把归一化标量映射为高维向量 $Z_\sigma\in\mathbb{R}^d$

#### 公式2: [[Skill-Guided Mixture-of-Experts|技能路由分布]]

$$
w_{k} = \operatorname{Router}(Z_{\sigma}), \quad k\in\{1,2,\dots,K\}
$$

**含义**: 以技能嵌入 $Z_\sigma$ 为条件，路由器给出 $K$ 个原子技能专家上的（未归一化）得分；采用稀疏激活，仅选 top-1 专家参与动作生成。

**符号说明**:
- $K$: 原子技能专家数量（LIBERO/真实=5，CALVIN=8）
- $w_k$: 第 $k$ 个专家的路由得分；$k$ 取得分最高者为被激活专家

#### 公式3: [[Mixture-of-Experts|共享专家与技能专家的组合输出]]

$$
F_{out} = (1-w_{k})\cdot F_{share}(x_{t}) + w_{k}\cdot F_{k}(x_{t})
$$

**含义**: 最终动作输出是"共享专家"与"被选中的技能专家"按路由权重 $w_k$ 的凸组合——共享专家守住 $\pi_0$ 的通用泛化，技能专家提供该原子技能的高保真执行。

**符号说明**:
- $x_t$: 当前多模态输入 $[O_t^{1:n}, \ell, s_t]$
- $F_{share}(\cdot)$: 共享专家（保留 $\pi_0$ 预训练的动作生成能力）
- $F_k(\cdot)$: 被激活的第 $k$ 个原子技能专家
- $w_k$: 公式2得到的路由权重

#### Algorithm 1: AtomicVLA 推理流程

$$
\begin{aligned}
&\textbf{输入}: \pi_\theta,\ \ell;\quad t\leftarrow 0,\ O_t^{1:n}\leftarrow\text{初始图像},\ Atomic\leftarrow\text{none}\\
&\textbf{while } \text{task not done } \textbf{do}\\
&\quad M\sim\pi_\theta.\textsc{predict}(\cdot\mid O_t^{1:n},\ell)\\
&\quad \textbf{if } M=\texttt{[think]}:\ [C_{0\text{-}k},C_t,\sigma]\sim\pi_\theta.\textsc{thinking}(\cdot\mid O_t^{1:n},\ell);\ Atomic\leftarrow\sigma\\
&\quad \textbf{else if } M=\texttt{[act]}:\ w_k\sim Router(\text{embed}(Atomic));\ A_t\sim\pi_\theta.\textsc{acting}(\cdot\mid O_t^{1:n},\ell,s_t,w_k);\ \text{执行 } A_t\\
&\quad t\leftarrow t+1\\
&\textbf{end while}
\end{aligned}
$$

**含义**: 每个时间步模型先决定 think 还是 act；think 时更新当前原子技能抽象 $Atomic$，act 时由路由依据该抽象选专家、生成并执行动作块，直到任务完成。这正是"自适应思考-执行"与"技能引导路由"的运行时闭环。

---

## 关键图表

<!-- 图片均使用 arXiv HTML 在线链接，未本地化 -->

### Figure 1: Overview of AtomicVLA / 总体概览

![Figure 1](https://arxiv.org/html/2603.07648v1/x1.png)

**说明**: 对比"单一动作头的传统 VLA"（扩展性差、混合技能严重干扰）与 AtomicVLA。后者用 SG-MoE 构建可扩展技能专家库，在统一框架内联合做任务规划与动作执行，从而在仿真与真实世界的长程任务与持续学习上都取得强性能。这张图点出了全文的核心叙事——**从 monolithic 解码器走向技能模块化**。

### Figure 2: AtomicVLA Pipeline & SG-MoE / 四联图（方法核心）

![Figure 2](https://arxiv.org/html/2603.07648v1/x2.png)

**说明**: 全文最关键的方法图，四个子图：
- **(a) Pipeline**: VLM 自适应预测原子技能抽象与 latent action；SG-MoE 里的 Action Decoder 同时接收 latent action 与新推断的原子技能抽象，生成细粒度电机动作。
- **(b) SG-MoE**: 包含 skill router、shared expert、多个 atomic-skill expert；router 依据原子技能选 top 专家，动作 token 同时过被激活专家与共享专家（对应公式2/3）。
- **(c) Continual Learning**: 新技能只训新专家 + 扩展路由（旧专家冻结）。
- **(d) Data Generation**: 用主轴分析 + InternVideo2.5 生成高质量具身推理数据。

### Figure 3: Inference Example / LIBERO-LONG 推理示例

![Figure 3](https://arxiv.org/html/2603.07648v1/x3.png)

**说明**: 可视化两个 LIBERO-LONG 任务。每个任务上排为任务进展，下排为 AtomicVLA 推断输出：**灰块=Thinking，彩块=Acting**（颜色对应被激活的技能专家）。直观展示"思考-执行交替"与"不同子阶段切换不同专家"的运行轨迹，是 SG-MoE 行为可解释性的直接证据。

### Figure 4: Error Recovery Capability / 错误恢复能力

![Figure 4](https://arxiv.org/html/2603.07648v1/x4.png)

**说明**: 遇到技能执行失败（如黄油被抓起又掉落）时，AtomicVLA 能**检测任务异常、重新生成原子技能抽象、重执行当前技能**从而恢复。说明统一框架带来的闭环纠错能力。注：CALVIN 评测不把"失败后恢复"算作有效完成，故其报告指标可能略低估真实能力。

### Figure 5: Real-world Long-horizon Demos / 真实长程任务对比

![Figure 5](https://arxiv.org/html/2603.07648v1/x5.png)

**说明**: 第二行为 AtomicVLA*，第一行为基线 $\pi_{0.5}$。AtomicVLA* 能可靠完成 $\pi_{0.5}$ 失败的配置，尤其在涉及关门操作的任务上优势更明显，体现更强的鲁棒性与执行稳定性。

### Figure 6: Skill Interference & Continual-Learning Degradation / 技能干扰与持续学习退化

![Figure 6](https://arxiv.org/html/2603.07648v1/x6.png)

**说明**: 上两行展示长程任务中的**技能干扰**——第一行为单技能成功执行，第二行为混合训练后失败（如"开抽屉"子任务不需闭夹爪，污染抓取类任务导致意外开/合夹爪）。下两行展示**持续学习退化**——$\pi_{0.5}$ 学新技能前后对比，红/绿框标注关键差异。这是 AtomicVLA 要解决的两类核心痛点的可视化佐证。

### Figure 7: InternVideo2.5 Prompts & Examples / 数据生成提示词

![Figure 7](https://arxiv.org/html/2603.07648v1/x7.png)

**说明**: 数据生成管线中 [[InternVideo2.5]] 所用的详细提示词与示例。VLM 分析视频片段，依据输入文本指令生成任务链、任务进度与原子动作标签，用于精化主轴分析的初始标注。

### Figure 8: LIBERO & CALVIN Demos / 仿真演示（附录）

![Figure 8](https://arxiv.org/html/2603.07648v1/x8.png)

**说明**: AtomicVLA 与 $\pi_0$ 在仿真环境的对比，取自 LIBERO 与 CALVIN 的代表性任务。AtomicVLA 在多处 $\pi_0$ 失败的实例上成功，体现更强鲁棒性与执行可靠性。

### Figure 9: Real-world Error Recovery Cases / 真实错误恢复案例（附录）

![Figure 9](https://arxiv.org/html/2603.07648v1/x9.png)

**说明**: 真实世界中的错误恢复。当因定位不准或目标-背景视觉歧义导致误抓时，AtomicVLA 能评估当前状态、生成更新计划、重试失败子任务（红框标注），保证整体任务鲁棒完成。

### Figure 10: Real-world Long-horizon Demos / 真实长程任务更多演示（附录）

![Figure 10](https://arxiv.org/html/2603.07648v1/x10.png)

**说明**: 更多真实世界实验演示，覆盖从简单到高度复杂、从规则到不规则物体的广谱场景，展示一致的强性能与鲁棒泛化。

### Table 1: LIBERO Benchmark Results / LIBERO 四套件对比

| Method | Spatial | Object | Goal | Long | Avg. |
|--------|---------|--------|------|------|------|
| Octo | 78.9 | 85.7 | 84.6 | 51.1 | 75.1 |
| OpenVLA | 84.9 | 88.4 | 79.2 | 53.7 | 76.5 |
| SpatialVLA | 88.2 | 89.9 | 78.6 | 55.5 | 78.1 |
| CoT-VLA | 87.5 | 91.6 | 87.6 | 69.0 | 81.1 |
| $\pi_0$ | 96.4 | 98.8 | 95.8 | 85.2 | 94.2 |
| $\pi_{0.5}$ | 98.8 | 98.2 | 98.0 | 92.4 | 96.9 |
| AtomicVLA (Ours) | 96.8 | 98.0 | 96.4 | 95.2 | 96.6 |
| **AtomicVLA\* (Ours)** | **98.8** | **98.8** | 97.2 | **96.2** | **97.8** |

**说明**: AtomicVLA（基于 $\pi_0$）平均 96.6%，超 $\pi_0$ 基线 2.4%；在最难的 **LIBERO-LONG 上 95.2%，超 $\pi_0$ 整整 10%**。AtomicVLA*（基于 $\pi_{0.5}$）平均 97.8%、Long 96.2%，全面最优——印证"分解-规划-组合"范式与长程任务结构天然契合。

### Table 2: CALVIN ABC→D Long-horizon Results / 长程序列评测

| Method | Task | 1 | 2 | 3 | 4 | 5 | Avg. Len ↑ |
|--------|------|---|---|---|---|---|-----------|
| $\pi_0$ | ABC→D | 94.3 | 87.0 | 77.9 | 68.5 | 59.4 | 3.87 |
| $\pi_{0.5}$ | ABC→D | 91.9 | 84.6 | 79.4 | 75.5 | 71.0 | 4.02 |
| AtomicVLA (Ours) | ABC→D | **95.0** | 87.8 | 81.9 | 75.0 | 69.1 | 4.09 |
| **AtomicVLA\* (Ours)** | ABC→D | 94.1 | **88.7** | **85.2** | **81.7** | **77.6** | **4.27** |

**说明**: 平均完成任务长度上 AtomicVLA 达 4.09（超 $\pi_0$ 0.22），AtomicVLA* 达 4.27（超 $\pi_{0.5}$ 0.25）。AtomicVLA* 在评测序列的**后三阶段**相对提升 5.8%/6.2%/6.6%——越往后（越依赖长程一致性）优势越大，说明对时序扩展任务尤其有效。

### Table 3: Real-world Long-horizon Multi-task / 真实长程多任务

| Method | InP | IntoD | IntoM | Avg. | $\Delta$Avg. |
|--------|-----|-------|-------|------|--------------|
| $\pi_0$ | 45 | 55 | 10 | 36.7 | – |
| $\pi_{0.5}$ | 65 | 35 | 35 | 45 | – |
| AtomicVLA | 65 | 60 | 45 | 56.7 | $+20.0\uparrow$ |
| **AtomicVLA\*** | **75** | **60** | **55** | **63.3** | $+18.3\uparrow$ |

**说明**: InP/IntoD/IntoM = Objects in plate / Object into drawer / Object into microwave。三个真实长程任务混合训练下，AtomicVLA 与 AtomicVLA* 分别超基线 **20% 与 18.3%**，验证显式原子技能库能有效缓解异构技能（尤其夹爪状态差异大的子任务）间的相互干扰。

### Table 4: Continual Learning with Skill Expansion / 持续学习

| Method | Grasp | Stack | Close | Press | Open (new) | Avg. | $\Delta$Avg. |
|--------|-------|-------|-------|-------|-----------|------|--------------|
| $\pi_{0.5}$ | 85 | 65 | 70 | 90 | - | 77.5 | – |
| $\pi_{0.5}$ (CL) | 70 | 45 | 60 | 75 | 55 | 61 | $-15.0\downarrow$ |
| AtomicVLA* | 95 | 80 | 70 | 100 | - | 86.3 | – |
| **AtomicVLA\* (CL)** | **90** | **80** | **80** | **100** | **70** | **82** | $\mathbf{-1.3\downarrow}$ |

**说明**: 把 "Open（开抽屉）"作为新原子技能增量学习。$\Delta$Avg. 为学新技能后**四个旧任务**的平均变化。$\pi_{0.5}$ 学新技能后旧任务平均掉 15%（Stack 掉 20% 最严重），而 **AtomicVLA* 仅掉 1.3%**，且新技能 Open 达 70%（高于 $\pi_{0.5}$ 的 55%）——结构化技能库管理几乎杜绝了灾难性遗忘，整体五任务上超 $\pi_{0.5}$ 21%。

### Table 5: Ablation on Routing Mechanism (LIBERO-LONG) / 路由机制消融

| Method | LIBERO-LONG |
|--------|-------------|
| $\pi_0$ | 85.2 |
| + MoE（token 级路由） | 88.6 |
| + MoDE（去噪步 $t$ 路由） | 89.5 |
| **+ SG-MoE (Ours，原子技能路由)** | **95.2** |

**说明**: 关键消融。SG-MoE 95.2%，超普通 token 级 MoE **6.6%**、超时间步条件的 MoDE 变体 **5.7%**。MoE 与 MoDE 都靠 token 级路由（增益主要来自负载均衡，专家仍学到混合技能、缺乏专精）；而 SG-MoE 以**语义化原子技能**为路由信号，保证同一技能阶段所有 token 走同一专家，专家专精、技能间干扰小——这是性能跃升的根因，也是全文最强论据。

### Table 6: Atomic Skill Distribution in LIBERO / LIBERO 原子技能分布（附录）

| Atomic Skill | Count |
|--------------|-------|
| Pick | 2462 |
| Place | 761 |
| Open | 201 |
| Close | 152 |
| Turn | 175 |

**说明**: LIBERO 数据经主轴分析后切分为 5 类原子技能。分布**严重不均衡**（Pick 远多于 Open/Close/Turn），故训练时对少数类提高采样频率以均衡分布、防止对应技能专家训练不足。

### Table 7: Real-world Tasks & Prompts / 真实任务与指令（附录）

| Task Type | Task | Prompt |
|-----------|------|--------|
| Long-horizon | Objects in plate | Place all blocks on the table into a green plate. |
| Long-horizon | Object into drawer | Open the top drawer and place the block inside. |
| Long-horizon | Object into microwave | Place the plate into the microwave and close the door. |
| Short | Grasp | Grasp the block from the table. |
| Short | Stack | Stack the red block on the orange block. |
| Short | Close | Close the microwave on the table. |
| Short | Press | Press the button on the table. |
| Short | Open | Open the top drawer. |
| Complex Scene | Objects in plate | Put the pepper and corn into the green plate. |
| Complex Scene | Objects in plate | Put the carrot and cucumber into the green plate. |
| Complex Scene | Objects in plate | Put the potato and eggplant into the green plate. |

**说明**: 真实实验任务清单。3 长程 + 5 短任务 + 3 复杂场景（不规则蔬果）。每短任务采 50 条、每长程任务 100 条遥操作示范，共 550 条；每任务随机摆放评测 20 次取平均。

### Table 8: Results on Complex Scenes / 复杂场景结果（附录）

| Method | Pepper/Corn | Carrot/Cucumber | Potato/Eggplant | Avg. |
|--------|-------------|-----------------|-----------------|------|
| $\pi_{0.5}$ | 25 | 40 | 35 | 33.3 |
| **AtomicVLA\*** | **40** | **45** | **45** | **43.3** |

**说明**: 复杂场景 + 不规则物体抓取，AtomicVLA* 平均 43.3%，超 $\pi_{0.5}$ 10%。抓玉米时因其颜色接近桌面背景，AtomicVLA* 能在接近目标过程中多次纠正，带来约 15% 提升——再次体现闭环纠错价值。

### Table 9: Per-task Success Rates on CALVIN ABC-D / CALVIN 逐任务成功率（附录）

| Task | SR(%) | Task | SR(%) | Task | SR(%) |
|------|------|------|------|------|------|
| rotate blue block right | 97.4 | lift red block table | 99.4 | lift blue block table | 99.4 |
| move slider right | 100.0 | lift pink block table | 94.5 | place in drawer | 100.0 |
| lift red block slider | 99.3 | move slider left | 100.0 | rotate red block left | 98.5 |
| place in slider | 98.6 | turn on lightbulb | 100.0 | push pink block left | 93.5 |
| turn off lightbulb | 100.0 | rotate blue block left | 100.0 | lift blue block slider | 95.6 |
| turn off led | 98.8 | push blue block left | 94.2 | lift pink block drawer | 100.0 |
| push into drawer | 86.0 | turn on led | 100.0 | rotate pink block right | 98.6 |
| lift blue block drawer | 100.0 | stack block | 98.4 | unstack block | 98.6 |
| close drawer | 100.0 | push pink block right | **33.8** | push blue block right | **22.2** |
| lift pink block slider | 97.8 | push red block right | **29.2** | rotate pink block left | 100.0 |
| open drawer | 100.0 | push red block left | 89.9 | lift red block drawer | 100.0 |
| rotate red block right | 97.3 | | | | |

**说明**: AtomicVLA* 在 34 个 CALVIN ABC-D 任务上多数接近 100%。明显短板是几个 **push * block right** 任务（仅 20–30%）——成因是训练时方块多在桌面中央、评测时多在右侧的**分布偏移**，模型方向正确但推得不够远未达成功阈值，进而阻断后续步骤。

### Table 10: Parameters & Inference Time / 参数量与推理时延（附录）

| Experts | $\pi_0$ | K=5 | K=8 | K=12 |
|---------|--------|------|------|------|
| Params | 3.24B | 4.17B | 4.81B | 5.65B |
| Act | 71 ms | 92 ms | 126 ms | 160 ms |
| Think | - | 104 ms | 104 ms | 104 ms |

**说明**: 单 H20 GPU 上的参数量与时延。技能专家越多参数与 Act 时延线性增长，但**即便 12 个专家 Act 仅 160 ms**、Think 恒为 104 ms（与专家数无关），实用性可接受。这说明 SG-MoE 的扩展开销是温和、可控的。

---

## 实验

### 数据集 / 基准

| 基准 | 规模 | 特点 | 用途 |
|------|------|------|------|
| [[LIBERO]] | 四套件（Spatial/Object/Goal/Long），每任务测 50 次 | 单臂，分解为 5 类原子技能（Pick/Place/Open/Close/Turn） | 训练/测试 |
| [[CALVIN]] | ABC→D 划分，每轨迹截断 64 帧，1000 序列×5 连续任务 | 长程语言条件操作，8 类原子技能 | 训练/测试 |
| 真实世界（Franka Research3） | 3 长程 + 5 短 + 3 复杂场景，共 550 条遥操作示范 | 双相机（腕部第一人称 + 对面第三人称），Gello 遥操作 | 训练/测试 |

### 实现细节

- **Backbone / 基础模型**: AtomicVLA 建于 [[π0|$\pi_0$]]，AtomicVLA* 建于 [[π0.5|$\pi_{0.5}$]]；数据统一转 [[LeRobot]] 格式。
- **专家结构**: 每专家 [[Gemma]] 架构，前馈为独立 [[SwiGLU]] MLP；width=2048、mlp_dim=4096、depth=18、heads=8、head_dim=256；全部随机初始化。技能专家数 $K$：LIBERO/真实=5，CALVIN=8。
- **优化**: AdamW，grad clip=1.0；学习率 [[Cosine Decay]]（warmup 1000 步、峰值 $2.5\times10^{-5}$、终值 $5\times10^{-6}$）；EMA decay=0.999 稳定训练。
- **训练**: LIBERO/CALVIN 各 100k 迭代，真实世界 30k 迭代，batch=64；持续学习：4 任务混训 20k 步 → 新技能 "open" 以 $5\times10^{-6}$ 微调 7k 步。
- **硬件**: 训练 8×H200 GPU；推理单卡 RTX RPO6000（时延测试用单 H20）。

### 关键实验结论

- **仿真**: LIBERO 平均 96.6%（超 $\pi_0$ 2.4%）、LIBERO-LONG 95.2%（超 10%）；CALVIN 平均任务长度 4.09/4.27（分别超 $\pi_0$/$\pi_{0.5}$ 0.22/0.25），后段阶段优势更大。
- **真实世界**: 长程多任务超基线 18.3%–20%；持续学习超 $\pi_{0.5}$ 21% 且旧任务仅退化 1.3%（基线退化 15%）；复杂不规则物体场景超 $\pi_{0.5}$ 10%。
- **消融（Table 5）**: SG-MoE（原子技能路由）95.2%，显著超 token 级 MoE（+6.6%）与去噪步路由 MoDE（+5.7%）——证明**以语义原子技能而非 token/去噪步路由**是性能跃升关键。
- **行为可解释 / 纠错**: Figure 3 展示 think/act 与专家激活轨迹；Figure 4/9 展示失败后自动重规划恢复。

---

## 批判性思考

### 优点
1. **路由信号选择有洞见且有强证据**: "用语义化原子技能而非 token/去噪步做路由"这一核心主张，被 Table 5 干净的消融直接支撑（+6.6%/+5.7%），并由 Figure 3 的专家激活可视化佐证，论证链完整。
2. **持续学习几乎无遗忘**: Table 4 中旧任务仅退化 1.3% vs 基线 15%，是模块化技能库"加专家不动旧专家"设计的有力证据，对终身学习场景价值高。
3. **统一框架带来闭环纠错**: think/act 自适应切换让模型能检测失败、重生成原子技能抽象并重试（Fig 4/9），这是单一动作头模型难具备的能力。
4. **工程可落地性论证到位**: Table 10 显示扩展开销温和（12 专家 Act 仅 160 ms），数据生成用主轴分析+VLM 大幅减少人工标注。

### 局限性
1. **参数与时延随专家数线性增长**: Table 10 显示 K=5 已 4.17B、K=12 达 5.65B，"原子技能"粒度若进一步细分，专家数膨胀会侵蚀效率优势；论文未讨论专家数上限或合并/裁剪机制。
2. **原子技能词表靠手工 + 阈值定义**: 主轴分析的阈值（3 cm / 0.05 rad / 0.1）与原子技能集（5 或 8 类）是经验设定，跨本体/跨任务的可迁移性、对阈值的敏感度均未系统分析。
3. **仍依赖大量人工示范（IL）**: 作者自己在 Future Work 中承认新技能仍需采集大量人类示范，尚未实现少样本/零样本扩展。
4. **分布偏移下脆弱**: Table 9 中 push-right 类任务仅 20–30%，暴露对训练-测试空间分布偏移的敏感；真实评测每任务仅 20 次、规模偏小，统计可靠性有限。
5. **路由权重组合的合理性存疑**: 公式3 用 $(1-w_k)$ 与 $w_k$ 加权共享/专精专家，但 $w_k$ 是 top-1 的"原始得分"，其数值范围与"作为凸组合权重"的语义是否一致，文中未充分说明。

### 潜在改进方向
1. 把原子技能的发现/词表与主轴分析阈值做成**可学习或自动搜索**，减少手工先验，并做敏感度分析。
2. 引入专家**合并/共享/层次化路由**机制，控制专家数膨胀，让"原子技能"粒度可自适应。
3. 按作者设想，**结合 RL**（如 $\pi^*_{0.6}$/SimpleVLA-RL/VLA-RL）让预训练技能专家库支持少样本/零样本获取新技能，摆脱对大量 IL 示范的依赖。
4. 扩大真实世界评测规模与扰动强度，并验证该框架在跨本体（不同机械臂/双臂）下的可移植性。

### 可复现性评估
- [ ] 代码开源（论文给出项目主页 https://zhanglk9.github.io/atomicvla-web/ ，截至笔记时未见明确代码仓库链接）
- [ ] 预训练模型（未明确声明 release）
- [x] 训练细节完整（附录 A.3 给出网络配置、优化器、学习率调度、迭代数、硬件等）
- [x] 数据集可获取（LIBERO/CALVIN 公开；真实数据基于 Franka + Gello 自采，未声明 release）

---

## 速查卡片

> [!summary] AtomicVLA: Unlocking the Potential of Atomic Skill Learning in Robots
> - **核心**: 把长程任务拆成"原子技能"，用技能引导的 SG-MoE 为每个技能配一个专家、按语义抽象稀疏路由，在统一 think/act 框架内联合做规划与执行，缓解技能干扰与灾难性遗忘。
> - **方法**: 原子技能→标量噪声级→嵌入 $Z_\sigma$（公式1）；$Z_\sigma$ 路由 top-1 专家（公式2）；共享专家($\pi_0$泛化)+技能专家按 $w_k$ 加权输出（公式3）；新技能只加专家+扩路由，旧专家冻结。
> - **结果**: LIBERO 96.6%（Long 95.2%，超 $\pi_0$ 10%）；CALVIN 平均长 4.09/4.27；真实长程超基线 18.3–20%、持续学习超 21% 且旧任务仅退化 1.3%；消融中 SG-MoE 超 token-MoE 6.6%。
> - **基础模型**: $\pi_0$ / $\pi_{0.5}$；专家 Gemma+SwiGLU；项目主页 https://zhanglk9.github.io/atomicvla-web/

---

*笔记创建时间: 2026-06-29*
