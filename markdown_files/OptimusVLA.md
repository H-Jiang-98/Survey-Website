---
title: "Global Prior Meets Local Consistency: Dual-Memory Augmented Vision-Language-Action Model for Efficient Robotic Manipulation"
method_name: "OptimusVLA"
authors: [Zaijing Li, Bing Hu, Rui Shao, Gongwei Chen, Dongmei Jiang, Pengwei Xie, Jianye Hao, Liqiang Nie]
year: 2026
venue: CVPR
tags: [VLA, dual-memory, flow-matching, retrieval-prior, temporal-consistency, efficient-inference]
zotero_collection: _inbox
image_source: online
arxiv_html: https://arxiv.org/html/2602.20200v2
created: 2026-06-29
---

# Global Prior Meets Local Consistency: Dual-Memory Augmented Vision-Language-Action Model for Efficient Robotic Manipulation

## 元信息

| 项目 | 内容 |
|------|------|
| 作者 | Zaijing Li, Bing Hu, Rui Shao, Gongwei Chen, Dongmei Jiang, Pengwei Xie, Jianye Hao, Liqiang Nie |
| 机构 | 哈尔滨工业大学（深圳）等（Liqiang Nie / Rui Shao 团队，Optimus 系列作者）；含天津大学等合作（Jianye Hao） |
| 会议 | CVPR 2026 |
| 类别 | VLA（Vision-Language-Action） |
| 日期 | 2026-05（arXiv v2） |
| 项目主页 | https://cybertronagent.github.io/OptimusVLA.github.io/ |
| 链接 | [arXiv](https://arxiv.org/abs/2602.20200) / [Project](https://cybertronagent.github.io/OptimusVLA.github.io/) |

---

## 一句话总结

> 给分层 [[VLA]] 配一个"双记忆"：用 [[Global Prior Memory|全局先验记忆]] 把流匹配的高斯噪声起点替换为检索得到的任务级动作先验（缩短去噪路径、自适应减少 NFE），再用 [[Local Consistency Memory|局部一致性记忆]] 对历史动作建模注入时序一致性约束，从而在更少推理步数下提升机器人操作的效率与鲁棒性。

---

## 核心贡献

1. **全局先验记忆（GPM）**: 把"先验初始化"重新定义为**记忆检索问题**而非固定噪声设计。用从语义相似轨迹检索出的**任务级动作先验**取代各向同性高斯噪声作为流匹配起点，缩小先验—目标分布间隙，**显著减少函数评估次数（NFE）**并降低采样落入运动学不可行区域的风险。
2. **局部一致性记忆（LCM）**: 一个**轻量工作记忆**，对最近执行的动作块做动态建模以推断任务进度，输出一个**一致性偏置**注入策略输入，赋予 VLA 时序感知与轨迹平滑性，且**几乎不增加计算开销**、不改动 VLA 预训练范式。
3. **双记忆 VLA 框架 OptimusVLA**: 由 GPM+LCM 联合驱动。在 LIBERO 取得 98.6% 平均成功率，CALVIN 上较 $\pi_0$ 提升 13.5%，RoboTwin 2.0 Hard 平均 38%；真实世界泛化/长程任务分别超 $\pi_0$ 42.9%/52.4%，同时实现约 **2.9× 推理加速**。

---

## 问题背景

### 要解决的问题
分层 [[VLA]]（视觉-语言主干做感知理解 + 生成式策略做动作生成）已成主流，但**动作生成环节**正成为效率与鲁棒性的瓶颈。本文要在不牺牲效率的前提下，同时解决动作生成的两大病灶。

### 现有方法的局限
作者指出两个核心局限（见 Figure 1 Middle）：

1. **限制 I：先验—目标间隙过大导致动作生成低效**。主流 VLA 用 [[Diffusion]]/[[Flow Matching]] 把各向同性高斯噪声映射到结构化动作分布，这种跨域变换距离大，需要**多步去噪**才能得到高质量动作；且随机起点常把生成过程初始化到**运动学不可行**的区域。若朴素地直接用单一动作先验当起点，又会**严重压缩多样性**，退化成"近似目标"的受限映射，且没回答"先验从哪来"。
2. **限制 II：对时序依赖的鲁棒性差**。VLA 普遍遵循 Markov 假设，**只条件于当前观测**，无法区分视觉相似但任务阶段不同的状态（如"还没打开的抽屉" vs "刚关上的抽屉"），并因缺乏与已执行轨迹的一致性而产生**抖动控制**。简单地把长序列历史观测拼进输入会大幅增加推理开销与显存，且**偏离 VLA 的单帧预训练分布**。

### 本文的动机
- **机器人任务的先验是可复用的**：相似任务（如 `pick_a_cup` 与 `pick_a_plate`）共享相关动作分布。因此可把生成起点从 $\mathcal{N}(0,I)$ **搬到目标流形邻域**——即从记忆库检索相似轨迹构造任务级先验（GPM）。
- **时序感知不必靠重型长上下文**：用一个轻量工作记忆只对**最近动作块**建模即可获得进度感知与一致性，避免每步都重跑 VLM 前向（LCM）。

---

## 方法详解

### 模型架构

OptimusVLA 是一个**分层 [[VLA]]** 框架（见 Figure 2），在标准的"视觉-语言主干 + 流策略"之上增挂两个记忆模块：
- **输入**: 自然语言指令 $\ell$ + 当前观测 $O_t$（含本体状态 $q_t$ 与多视角图像 $I_t^1,\dots,I_t^n$）
- **Backbone**: 视觉-语言主干 [[VLM]]（沿用 [[π0.5]] 的架构与训练协议）
- **核心模块**: [[Global Prior Memory|GPM]]（检索任务级先验、替换噪声起点）+ [[Local Consistency Memory|LCM]]（动态建模历史、注入一致性偏置）
- **生成器**: [[Flow Matching|流策略]] $p_\theta$，以**自适应 NFE** 去噪
- **输出**: [[Action Chunking|动作块]] $a_{t+1:t+H}$
- **总参数**: 3.6B（从 $\pi_{0.5}$ 权重初始化后增挂 GPM/LCM）

整体前向流程（公式 3–7）：先由 VLM 编码多模态表征 $E_{emb}$；GPM 据此检索任务级先验分布 $\mathcal{P}_{re}$ 并采样初始化 $\mathbf{\hat X}_t$；LCM 据上一动作块输出一致性偏置 $\mathbf{B}_t$；二者相加得到策略输入 $\mathbf{X}_t$，最后流策略以自适应步数 $N$ 去噪输出动作块。

### 核心模块

#### 模块1: Global Prior Memory（GPM，全局先验记忆）

**设计动机**: 标准流匹配把高斯噪声运到动作空间，因两分布差异巨大而需大量 NFE 且易塌缩到不可行动作。GPM 把生成**起点从 $\mathcal{N}(0,I)$ 重定位到目标流形邻域**，从根本上缩小先验—目标间隙。

**具体实现**（三件套：Prior Head / Memory Bank / Prior-Aware Sampler）:
- **Prior Head**: 一个轻量 MLP（实际为 2 层 MLP）把多模态表征 $E_{emb}$（离线构建时为对前缀 token 的均值池化 $e^{(m)}$）投影并 L2 归一化为检索 token $z_{re}$（公式 8 / 22–23）。
- **Memory Bank**: 存 $M$ 个键值对 $\{z_m, J_m\}$（任务嵌入 + 对应完整轨迹），用 [[FAISS]] `IndexFlatIP` 建索引。$z_{re}$ 查询后按余弦相似度取 top-$k$ 轨迹 $\{J_i, s_i\}$（公式 9 / 25–26）。
- 计算 softmax 权重 $\alpha_i$ 与全局相似度 $\bar s$（公式 10/27）；按当前进度 $\rho_t$ 用滑窗对齐每条轨迹取动作块 $C_i$（公式 28，长度不符则线性插值）；以**矩匹配单高斯**逼近高斯混合得到先验 $\mathcal{P}_{re}=\mathcal{N}(\mu,\mathrm{diag(Var)})$（公式 11/29）。
- **Prior-Aware Sampler**: 据检索置信度 $\bar s$ **自适应**地设噪声尺度 $\lambda$（公式 12/30）与去噪步数 $N$（公式 13/31）：$\bar s$ 越高→$\lambda$、$N$ 越小（更信先验均值、传输更易）；$\bar s$ 越低（新场景）→优雅退化为更大噪声与更多步数。最后采样初始化 $\mathbf{\hat X}_t$（公式 14/32）。
- **工程优化（Session-Level Caching）**: 每个 episode 开始（或检测到任务漂移）时检索一次并缓存 top-$k$ 索引与权重，之后每步只按 $\rho_t$ 取时间对齐块、重算式 29，**不再查 FAISS**，开销可忽略。

#### 模块2: Local Consistency Memory（LCM，局部一致性记忆）

**设计动机**: 在 Markov 假设下 VLA 缺进度感知且控制抖动。LCM 作为**工作记忆**对动作历史动态建模，推断进度、强制时序一致，且开销极小。

**具体实现**（Consistency Layer + Dynamic Awareness）:
- **Consistency Layer**: 在 $t$ 步取上一动作块 $\mathbf{A}_{t-1}=[\mathbf a_{t-H+1},\dots,\mathbf a_t]\in\mathbb{R}^{H\times A}$，用[[Self-Attention|自注意力]]捕捉块内动作间依赖与约束，得 $\mathbf{\hat B}_{t-1}$（公式 15）。
- **Dynamic Awareness Module**: 捕捉**块间**时序动态，用 [[Mamba]] 结构（线性复杂度建长程依赖）更新内部状态、预测下一步一致性偏置 $\mathbf{B}_t$（公式 16）。
- LCM 把"动作流中的一致性"转化为一个**加性约束**注入策略输入，使采样轨迹更平滑。

### 训练策略（三阶段，超参见 Table 6）

1. **Stage I 分层 VLA 预训练**: 按 $\pi_{0.5}$ 架构/协议训练基座 VLA，此时不挂 GPM/LCM，流策略源分布保持 $\mathcal{P}_0=\mathcal{N}(0,I)$ 以学到可泛化的速度场，用 CFM 目标训练。
2. **Stage II GPM 训练**: 冻结基座，仅训 Prior Head，用 [[InfoNCE]] 任务对比损失（公式 17/24）让同任务嵌入聚类、异任务分离；用 **Task-Pair Batch Sampler** 保证每个 batch 至少含同任务两条轨迹（in-batch 正样本，其余为难负样本）。AdamW，lr 1e-4，batch 64，$\tau_c=0.07$，20 epoch。
3. **Stage III LCM 训练**: 冻结 VLM/流策略/GPM，训 LCM **回归残差** $\mathbf{B}^\star_t=\mathbf{A}^\star_t-\mu_t$（公式 18/33）以 MSE（公式 19）优化；用 **cold-start** 策略（以概率 $p_{cold}$ 把 $\mathbf{A}_{t-1}$ 置零）保证初始无历史时鲁棒。

### 关键公式与机制

#### 公式1: [[Flow Matching|条件流匹配]]直线路径（OT path）

$$
x_t = (1-t)\,x_0 + t\,x_1
$$

**含义**: 在源样本（噪声）$x_0\sim\mathcal P_0$ 与目标动作 $x_1\sim\mathcal P_1$ 间定义直线概率路径，沿路径目标速度恒为 $u_t(x_t\mid x_0,x_1)=x_1-x_0$。

**符号说明**:
- $t\in[0,1]$: 流时间；$x_0$: 源/噪声样本；$x_1$: 目标动作样本

#### 公式2: 流匹配训练目标

$$
\min_{\theta}\ \mathbb{E}_{t\sim\mathcal{U}[0,1],\,x\sim p_t(x)}\Big\|\,v_{\theta}(t,x)-u_t(x)\,\Big\|_2^2
$$

**含义**: 训练时间条件速度场 $v_\theta$ 去拟合目标速度场 $u_t$；推理时解 ODE $dx_t/dt=v_\theta(t,x_t)$ 生成动作。当 $\mathcal P_0$ 为高斯、$\mathcal P_1$ 为结构化动作时源-目标间隙大、所需 NFE 多——正是 GPM 要缩小的对象。

**符号说明**:
- $v_\theta$: 待学习速度场；$u_t$: 目标速度场；$\mathbb{E}$: 对时间与样本取期望

#### 公式3–7: OptimusVLA 整体前向

$$
E_{emb}\leftarrow \texttt{VLM}(O_t,\ell)
$$

$$
\mathcal{P}_{re}\leftarrow \texttt{GPM}(z_{re}),\qquad \mathbf{B}_t\leftarrow \texttt{LCM}(\mathbf{A}_{t-1})
$$

$$
\mathbf{X}_t=\mathbf{\hat X}_t+\mathbf{B}_t,\qquad a_{t+1:t+H}\leftarrow p_\theta(\mathbf{X}_t,N)
$$

**含义**: VLM 编码多模态表征 → GPM 检索任务级先验分布并采样 $\mathbf{\hat X}_t\sim\mathcal P_{re}$ → LCM 据上一动作块产一致性偏置 $\mathbf{B}_t$ → 两者相加成策略输入 $\mathbf{X}_t$ → 流策略以自适应步数 $N$ 去噪输出动作块。

**符号说明**:
- $z_{re}$: 检索 token；$\mathbf{\hat X}_t\in\mathbb{R}^{H\times A}$: 先验采样初始化；$H$ 块长、$A$ 动作维；$N$: 自适应 NFE

#### 公式8–11: GPM 检索与先验构造

$$
\alpha_i=\mathrm{softmax}(s_i/\tau_s),\qquad \bar s=\sum_{i=1}^{k}\alpha_i s_i
$$

$$
\mu=\sum_{i=1}^{k}\alpha_i C_i,\qquad \mathrm{Var}=\sum_{i=1}^{k}\alpha_i\,(C_i-\mu)^{\odot 2}
$$

**含义**: 对 top-$k$ 检索分数做温度 softmax 得权重 $\alpha_i$ 与全局相似度 $\bar s$（置信度指示，$\bar s\in[-1,1]$）；对齐进度取出的动作块 $C_i$ 做加权得任务级先验高斯的均值 $\mu$ 与对角方差。

**符号说明**:
- $s_i=\langle\tilde z_{re},\tilde z_i\rangle$: 余弦相似度；$\tau_s$: 温度；$C_i\in\mathbb{R}^{H\times A}$: 第 $i$ 条对齐动作块；$\odot 2$: 逐元素平方

#### 公式12–14: 相似度自适应采样

$$
\lambda(\bar s)=\lambda_{\max}-\frac{\bar s+1}{2}\,(\lambda_{\max}-\lambda_{\min})
$$

$$
N(\bar s)=\mathrm{Round}\!\left(N_{\min}+\Big(1-\frac{\bar s+1}{2}\Big)\,(N_{\max}-N_{\min})\right)
$$

$$
\mathbf{\hat X}_t=\mu+\lambda(\bar s)\,\big(\epsilon\odot\sqrt{\mathrm{Var}}\big),\qquad \epsilon\sim\mathcal{N}(0,I)
$$

**含义**: 检索置信度高时（$\bar s\approx1$）噪声尺度 $\lambda$ 与步数 $N$ 都减小（更信先验、传输更易、更省算力）；新场景（$\bar s\approx0$）则退化为更大噪声、更多步数以保多样性与泛化。该 $\mathbf{\hat X}_t$ 直接替换标准高斯噪声起点 $x_0$。

**符号说明**:
- $\lambda_{\min/\max},N_{\min/\max}$: 超参；$\epsilon$: 标准高斯噪声；并设方差下限 $\sigma_{\min}^2$ 防塌缩

#### 公式17 / 24: GPM 的 InfoNCE 对比损失

$$
\mathcal{L}_{\mathrm{GPM}}=-\mathbb{E}_{q}\left[\log\frac{\exp(\mathrm{sim}(z_{re},z^{+})/\tau_c)}{\sum_{j\in\mathcal{N}(q)}\exp(\mathrm{sim}(z_{re},z_j)/\tau_c)}\right]
$$

**含义**: 训练 Prior Head 学习**任务可判别**表征，使同任务嵌入相互靠拢、异任务被推远，从而保证检索到语义一致的先验。

**符号说明**:
- $\mathrm{sim}(\cdot)$: 余弦相似度；$z^{+}$: 同任务正样本；$\mathcal{N}(q)$: in-batch 负样本集；$\tau_c$: 温度

#### 公式18–19: LCM 残差回归损失

$$
\mathbf{B}^{\star}_t=\mathbf{A}^{\star}_t-\mu_t
$$

$$
\mathcal{L}_{\mathrm{LCM}}=\mathbb{E}_{(\mathbf{A}_{t-1},\mathbf{A}^{\star}_t,\mu_t)\sim\mathcal{D}}\Big[\big\|\mathbf{B}_t-\mathbf{B}^{\star}_t\big\|_2^2\Big]
$$

**含义**: 冻结 GPM 后训 LCM **预测残差**——即从全局先验均值 $\mu_t$ 到真值动作块 $\mathbf{A}^\star_t$ 的差值，使 LCM 在先验基础上补足时序一致性所需的修正量。

**符号说明**:
- $\mathbf{B}^\star_t$: 目标偏置（残差）；$\mu_t$: $t$ 时刻先验均值；$\mathbf{A}^\star_t$: 真值动作块；$\mathcal{D}$: 训练数据

---

## 关键图表

<!-- 图片均使用 arXiv HTML 在线链接，未本地化 -->

### Figure 1: Motivation / 标准 VLA 与 OptimusVLA 对比及两大局限

![Figure 1](https://arxiv.org/html/2602.20200v2/x1.png)

**说明**: 上：标准 VLA（左）与 OptimusVLA（右）架构对比。中：图示 GPM（蓝）与 LCM（绿）如何分别解决两大局限——(i) 先验—目标间隙大导致低效；(ii) 对时序依赖鲁棒性差。下：效率与性能对比。这张图是全文论点的"地图"，把"高斯起点远→多步去噪"和"无历史→相似观测混淆"两个痛点可视化。

### Figure 2: Framework Overview / OptimusVLA 总体框架

![Figure 2](https://arxiv.org/html/2602.20200v2/x2.png)

**说明**: 给定任务与当前观测，视觉-语言主干先编码为多模态表征；GPM 据此检索任务级先验，LCM（图中标 LBM）对历史动作序列动态编码产一致性约束；最后流策略以自适应 NFE 调度对初始化去噪生成动作块。是理解前向数据流（公式 3–7）的关键图。

### Figure 3: Real-World Setup & Results / 真实世界设置与结果

![Figure 3](https://arxiv.org/html/2602.20200v2/x3.png)

**说明**: 在 GALAXEA R1 Lite 双臂平台上，OptimusVLA 与 OpenVLA、OpenVLA-OFT、$\pi_0$、$\pi_{0.5}$ 在 Generalization Tasks 与 Long-horizon Tasks 两套件上的对比。OptimusVLA 两套件均最佳（泛化 85.0%、长程 64.0%），长程任务较 $\pi_0$ 高 52.4%。

### Figure 4: Training Efficiency / 训练效率对比

![Figure 4 (a) Spatial](https://arxiv.org/html/2602.20200v2/x4.png)
![Figure 4 (b) Object](https://arxiv.org/html/2602.20200v2/x5.png)
![Figure 4 (c) Goal](https://arxiv.org/html/2602.20200v2/x6.png)
![Figure 4 (d) Long](https://arxiv.org/html/2602.20200v2/x7.png)

**说明**: 同从 $\pi_{0.5}$ 权重初始化下，OptimusVLA 在 LIBERO 各套件上以**更少训练步数**达到更高成功率（如 LIBERO-Goal 仅 18k 步达 97.6%，$\pi_{0.5}$ 需 26k 步）。佐证 GPM 的任务级先验把初始化放在目标流形附近、降低变换复杂度。

### Figure 5: Inference Efficiency / 推理效率对比

![Figure 5](https://arxiv.org/html/2602.20200v2/x8.png)

**说明**: 在 LIBERO 与真实世界上对比推理时间与 NFE。OptimusVLA 在 LIBERO 上推理时间快 6.5×、NFE 少 3.1×，且性能最高。证明 GPM/LCM 开销极小却大幅减少 NFE 这一推理速度主因。

### Figure 6: Qualitative Results / 定性分析（仿真+真实）

![Figure 6](https://arxiv.org/html/2602.20200v2/x9.png)

**说明**: 上：仿真任务"Pick up the bbq sauce and place it in the basket"关键帧，可视化检索相似度 $\bar s$、自适应噪声尺度 $\lambda$、NFE $N$——GPM 使推理从目标分布邻域起步、自适应少步。下：真实任务"Place the red apple onto the plate"，展示 LCM 建模时序依赖，而 $\pi_{0.5}$ 难以区分相似观测（分不清苹果是否已放好）。

### Figure 7: Real-World Generalization Tasks / 真实泛化任务序列（附录）

![Figure 7](https://arxiv.org/html/2602.20200v2/x10.png)

**说明**: 四个 Generalization Tasks 的执行序列（自上而下：Place one fruit on plate、Place plate on tablecloth、Stand bottle upright、Place cup on tablecloth），含第三人称视角与腕部相机视角。

### Figure 8: Real-World Long-horizon Tasks / 真实长程任务序列（附录）

![Figure 8](https://arxiv.org/html/2602.20200v2/x11.png)

**说明**: 四个 Long-horizon Tasks 执行序列（Place all fruits on plate、Place one fruit on tablecloth then place another into bowl、Place cup and apple on plate、Place 3 blocks on tablecloth），含第三人称与腕部视角。

### Table 1: LIBERO 性能对比（500 rollouts，成功率 %）

| Method | Spatial | Object | Goal | Long | Avg. |
|--------|---------|--------|------|------|------|
| DP | 78.3 | 92.5 | 68.3 | 50.5 | 72.4 |
| Octo | 78.9 | 85.7 | 84.6 | 51.1 | 75.1 |
| SpatialVLA | 88.2 | 89.9 | 78.6 | 55.5 | 78.1 |
| OpenVLA | 84.7 | 88.4 | 79.2 | 53.7 | 76.5 |
| $\pi_0$-FAST | 96.4 | 96.8 | 88.6 | 60.2 | 85.5 |
| CogACT | 97.2 | 98.0 | 90.2 | 88.8 | 93.6 |
| UniVLA | 95.4 | 98.8 | 93.6 | 94.0 | 95.4 |
| $\pi_0$ | 96.8 | 98.8 | 95.8 | 85.2 | 94.2 |
| MemoryVLA | 98.4 | 98.4 | 96.4 | 93.4 | 96.7 |
| OpenVLA-OFT | 97.6 | 98.4 | 97.9 | 94.5 | 97.1 |
| $\pi_{0.5}$ | 98.8 | 98.2 | 98.0 | 92.4 | 96.9 |
| **OptimusVLA** | **99.6** | **99.8** | **98.4** | **96.4** | **98.6** |

**说明**: OptimusVLA 各套件全面最优、平均 98.6% 刷新 SOTA。尤其在易累计误差的 LIBERO-Long 上 96.4% 居首——GPM 检索任务先验锚定生成过程，稳定长程并把 NFE 从 $\pi_{0.5}$ 的 10.0 降到 3.2。

### Table 2: CALVIN（ABC→D）性能对比

| Method | 1/5 | 2/5 | 3/5 | 4/5 | 5/5 | Avg. Len |
|--------|-----|-----|-----|-----|-----|----------|
| OpenVLA | 91.3 | 77.8 | 62.0 | 52.1 | 43.5 | 3.27 |
| RoboDual | 94.4 | 82.7 | 72.1 | 62.4 | 54.4 | 3.66 |
| UniVLA | 95.5 | 85.8 | 75.4 | 66.9 | 56.5 | 3.80 |
| ReconVLA | 95.6 | 87.6 | 76.9 | 69.3 | 64.1 | 3.95 |
| $\pi_0$ | 93.8 | 85.0 | 76.7 | 68.1 | 59.9 | 3.92 |
| UP-VLA | 92.8 | 86.5 | 81.5 | 76.9 | 69.9 | 4.08 |
| RoboVLM | 98.0 | 93.6 | 85.4 | 77.8 | 70.4 | 4.25 |
| $\pi_{0.5}$† | 94.4 | 88.4 | 85.3 | 80.1 | 76.1 | 4.26 |
| Seer | 96.3 | 91.6 | 86.1 | 80.3 | 74.0 | 4.28 |
| VPP | 95.7 | 91.2 | 86.3 | 81.0 | 75.0 | 4.29 |
| **OptimusVLA** | **97.6** | **93.2** | **88.8** | **85.7** | **78.1** | **4.45** |

**说明**: 平均完成长度 4.45 居首，较 $\pi_0$（3.92）相对提升 13.5%。在 ABC→D 零样本迁移下，$\pi_0$ 的任务无关高斯噪声对分布漂移脆弱；OptimusVLA 通过 GPM 检索语义相似轨迹作初始化，"形变一个可行先验"而非从零生成，零样本鲁棒性更强。†为复现结果。

### Table 3: RoboTwin 2.0 Hard 主表（per-task SR / RK，100 rollouts，8 任务）

| Task | RDT (SR/RK) | ACT | DP | DP3 | $\pi_0$ | $\pi_{0.5}$† | **OptimusVLA** |
|------|------|------|------|------|------|------|------|
| Click Alarmclock | 12%/4 | 4%/7 | 5%/6 | 14%/3 | 11%/5 | 18%/2 | **31%/1** |
| Click Bell | 9%/3 | 3%/4 | 0%/5 | 0%/5 | 3%/4 | 28%/2 | **46%/1** |
| Dump Bin Bigbin | 32%/3 | 1%/6 | 0%/7 | **53%/1** | 24%/5 | 29%/4 | 35%/2 |
| Open Laptop | 32%/4 | 0%/6 | 0%/6 | 7%/5 | 46%/2 | 38%/3 | **48%/1** |
| Place Bread Skillet | 1%/3 | 0%/4 | 0%/4 | 0%/4 | 1%/4 | 2%/2 | **4%/1** |
| Place Container Plate | 17%/4 | 1%/5 | 0%/6 | 1%/5 | **45%/1** | 30%/3 | 37%/2 |
| Press Stapler | 24%/4 | 6%/5 | 0%/7 | 3%/6 | 29%/3 | 36%/2 | **45%/1** |
| Stack Bowls Two | 30%/4 | 0%/6 | 0%/6 | 6%/5 | 41%/3 | 49%/2 | **58%/1** |
| **Average** | 20%/4 | 2%/6 | 1%/7 | 11%/5 | 25%/3 | 29%/2 | **38%/1** |

**说明**: 主表 8 任务平均 38% 居首（对应摘要数字）。双臂操作需高时序与臂间一致性，RDT 缺显式双臂协调机制；LCM 为 OptimusVLA 提供一致性约束，Stack Bowls Two 达 58%，较 RDT(30%) 高 +28%。

### Table 4: GPM/LCM 消融（LIBERO-Long / CALVIN / 真实泛化）

| GPM | LCM | LIBERO-Long | CALVIN | Generalization |
|-----|-----|-------------|--------|----------------|
| ✓ | ✓ | **96.4** | **4.45** | **85.0** |
| ✗ | ✓ | 93.2 (↓3.3%) | 4.28 (↓3.8%) | 77.0 (↓9.4%) |
| ✓ | ✗ | 94.8 (↓1.7%) | 4.38 (↓1.6%) | 79.5 (↓6.5%) |
| ✗ | ✗ | 92.4 (↓4.1%) | 4.26 (↓4.3%) | 75.0 (↓11.8%) |

**关键发现**: 去 GPM 在 CALVIN/真实泛化上掉得最狠（↓3.8%/↓9.4%）——失去先验后退化为普通流策略，被先验-目标间隙拖累泛化；去 LCM 主要伤 LIBERO-Long（↓1.7%）——失去时序一致性。两者互补，全去掉最差。

### Table 5: GPM 记忆库规模消融（LIBERO-Long 成功率 %）

| Metric | Num=6500, k=8 | Num=6500, k=16 | Num=1300, k=1 | Num=1300, k=8 | Num=130, k=1 | Num=130, k=8 |
|--------|------|------|------|------|------|------|
| SR | **96.4** | 94.8 | 92.6 | 95.2 | 92.4 | 93.6 |

**关键发现**: 性能随记忆库丰富度上升；每任务只存一条轨迹会因先验过确定性而退化。检索 $k$ 需足够大（$k=8$ 优）：$k$ 太小过拟合单条轨迹，$k$ 适中可构成稳健高斯混合先验，平衡特异性与探索性。

### Table 6: 三阶段训练超参

| Hyperparameter | Stage-1 | Stage-2 | Stage-3 |
|----------------|---------|---------|---------|
| Optimizer | AdamW | AdamW | AdamW |
| Learning Rate | 5e-5 | 1e-4 | 1e-4 |
| Steps | 30000 | 1000 | 1000 |
| Batch Size | 512 | 64 | 64 |
| Warm Up Ratio | 0.10 | - | - |
| EMA Decay | 0.999 | - | - |

**说明**: Stage-1 预训练基座（30k 步、batch 512），Stage-2/3 仅训轻量 Prior Head 与 LCM（1k 步、batch 64），印证两记忆模块训练代价很小。

### Table 7: RoboTwin 2.0 Hard 完整表（16 任务，附录，per-task SR/RK）

| Task | RDT | ACT | DP | DP3 | $\pi_0$ | $\pi_{0.5}$† | **OptimusVLA** |
|------|-----|-----|-----|-----|------|------|------|
| Click Alarmclock | 12%/4 | 4%/7 | 5%/6 | 14%/3 | 11%/5 | 18%/2 | **31%/1** |
| Click Bell | 9%/3 | 3%/4 | 0%/5 | 0%/5 | 3%/4 | 28%/2 | **46%/1** |
| Dump Bin Bigbin | 32%/3 | 1%/6 | 0%/7 | **53%/1** | 24%/5 | 29%/4 | 35%/2 |
| Open Laptop | 32%/4 | 0%/6 | 0%/6 | 7%/5 | 46%/2 | 38%/3 | **48%/1** |
| Place Bread Skillet | 1%/3 | 0%/4 | 0%/4 | 0%/4 | 1%/4 | 2%/2 | **4%/1** |
| Place Container Plate | 17%/4 | 1%/5 | 0%/6 | 1%/5 | **45%/1** | 30%/3 | 37%/2 |
| Press Stapler | 24%/4 | 6%/5 | 0%/7 | 3%/6 | 29%/3 | 36%/2 | **45%/1** |
| Stack Bowls Two | 30%/4 | 0%/6 | 0%/6 | 6%/5 | 41%/3 | 49%/2 | **58%/1** |
| Beat Block Hammer | **37%/1** | 3%/5 | 0%/6 | 8%/4 | 21%/3 | 21%/3 | 26%/2 |
| Lift Pot | 9%/4 | 0%/5 | 0%/5 | 0%/5 | **36%/1** | 28%/3 | 31%/2 |
| Move Playingcard Away | 11%/4 | 0%/6 | 0%/6 | 3%/5 | 22%/3 | 25%/2 | **32%/1** |
| Open Microwave | 20%/5 | 0%/6 | 0%/6 | 22%/4 | **50%/1** | 39%/3 | 41%/2 |
| Pick Diverse Bottles | 0%/4 | 0%/4 | 0%/4 | 1%/3 | 6%/2 | 6%/2 | **7%/1** |
| Turn Switch | 15%/3 | 2%/6 | 1%/7 | 8%/5 | **23%/1** | 11%/4 | 16%/2 |
| Place Object Stand | 5%/3 | 0%/4 | 0%/4 | 0%/4 | 11%/2 | 11%/2 | **13%/1** |
| Stack Blocks Two | 2%/3 | 0%/5 | 0%/5 | 0%/5 | 1%/4 | 5%/2 | **11%/1** |
| **Average** | 16%/4 | 1%/6 | 1%/6 | 8%/5 | 23%/3 | 24%/2 | **30%/1** |

**说明**: 16 任务完整版平均 30%（正文 Table 3 的 8 任务子集平均为 38%）。OptimusVLA 在多数任务排第一、整体均值最高，超过 $\pi_{0.5}$(24%) 与 $\pi_0$(23%)；少数偏视觉/单步任务（如 Dump Bin、Open Microwave）DP3/$\pi_0$ 略占优，体现先验检索在强视觉随机化下并非万能。

### Table 8: 真实世界 Generalization Tasks 概览

| Task | 描述 | #demo | #rollout | 成功率 % |
|------|------|-------|----------|---------|
| place_one_fruit_on_plate | 抓水果放盘（香蕉/苹果/柠檬等） | 100 | 50 | 88 |
| place_plate_on_tablecloth | 抓盘放桌布（多种盘/布样式） | 100 | 50 | 90 |
| stand_bottle_upright | 抓瓶并扶正（形状各异） | 150 | 50 | 74 |
| place_cup_on_tablecloth | 抓杯放桌布（形状各异） | 120 | 50 | 88 |

**说明**: 每任务 50 rollout（25 变光照 + 25 变场景）。平均 85.0%，GPM 按多模态语义检索任务级先验使策略对视觉干扰鲁棒。

### Table 9: 真实世界 Long-horizon Tasks 概览

| Task | 描述 | #demo | #rollout | 成功率 % |
|------|------|-------|----------|---------|
| place_all_fruits_on_plate | 抓三种水果放盘（每次随机组合） | 200 | 25 | 64 |
| place_2_fruits_in_2_position | 从碗取一果放桌布，再从盘取一果放碗 | 200 | 25 | 60 |
| place_2_obj_on_plate | 依次抓杯与苹果放盘 | 250 | 25 | 68 |
| place_blocks_on_tablecloth | 抓三块积木放桌布（每次随机组合） | 300 | 25 | 64 |

**说明**: 平均 64.0%，较 $\pi_0$ 高 52.4%，展现 LCM 带来的长程动作稳定性与双臂协调一致性。

---

## 实验

### 数据集 / 基准

| 基准 | 规模/协议 | 特点 | 用途 |
|------|-----------|------|------|
| [[LIBERO]] | 4 套件×10 任务，每套件 500 rollouts | 单臂，Spatial/Object/Goal/Long | 训练/测试 |
| [[CALVIN]] | ABC→D（3 训 1 测），500 rollouts | 单臂长程语言条件，报告各 track 与 Avg.Len | 训练/测试（零样本环境迁移） |
| [[RoboTwin]] 2.0 | 随机选 16 任务，每任务 100 rollouts | **双臂**，Hard 设置（杂乱/光照/纹理/桌高随机化） | 训练/测试 |
| 真实世界 | Generalization(每任务 100–150 demo,50 试) / Long-horizon(200–300 demo,25 试) | GALAXEA R1 Lite 14-DoF 双臂（腕部+第三人称相机，224×224） | 训练/测试 |

### 实现细节

- **初始化**: 从 [[π0.5]] 权重初始化，增挂 GPM/LCM，总参数 **3.6B**
- **训练**: 8× NVIDIA A800，全局 batch 512，30000 步；LIBERO/CALVIN/RoboTwin 学习率均 5e-5
- **三阶段**: Stage-1 基座预训练 → Stage-2 仅训 Prior Head（InfoNCE，1k 步）→ Stage-3 仅训 LCM（MSE 残差，1k 步）
- **检索**: FAISS `IndexFlatIP`，top-$k$（默认 $k=8$）；session 级缓存仅 episode 开始检索一次

### 关键实验结论

- **仿真**: LIBERO 98.6%（全面 SOTA，NFE 仅 3.2 vs $\pi_{0.5}$ 的 10.0）；CALVIN Avg.Len 4.45（较 $\pi_0$ +13.5%）；RoboTwin Hard 8 任务 38% / 16 任务 30%（均超 $\pi_{0.5}$）。
- **真实世界**: 泛化 85.0%、长程 64.0%，分别超 $\pi_0$ 42.9%/52.4%。
- **效率**: LIBERO 推理时间快 6.5×、NFE 少 3.1×；训练上 LIBERO-Goal 18k 步即达 97.6%（$\pi_{0.5}$ 需 26k 步）；整体约 2.9× 推理加速。
- **消融**: GPM 主要贡献泛化（去掉真实泛化 ↓9.4%），LCM 主要贡献长程时序一致（去掉 LIBERO-Long ↓1.7%）；记忆库越大越好、$k=8$ 最优。

---

## 批判性思考

### 优点
1. **思路清晰且双管齐下**: 把"流匹配起点"与"时序一致性"两个相对独立的瓶颈分别交给 GPM、LCM，且二者都是**即插即用、训练代价极小**（各 1k 步、batch 64）的轻量模块，工程上很有吸引力。
2. **效率提升有可解释机制**: 自适应 $\lambda/N$ 调度把"检索置信度"直接映射到"噪声与步数"，置信高就少走几步，这一设计直觉自洽且 NFE 实测从 10.0 降到 3.2，效率收益可量化。
3. **评测覆盖广**: 三个仿真基准 + 真实双臂平台，含零样本环境迁移（CALVIN ABC→D）与强域随机化（RoboTwin Hard），结论较有说服力；真实任务还区分泛化/长程两套件。

### 局限性
1. **强依赖记忆库覆盖度**（作者自承）: 当前任务/场景大幅偏离存储经验时，检索先验可能误导策略。GPM 的本质是"近邻插值"，对**真正未见的新技能**帮助有限，泛化光环主要来自"训练分布内的视觉/位置变体"。
2. **数字口径需当心**: 摘要/正文用 RoboTwin **8 任务子集的 38%**，而附录完整 16 任务实为 **30%**；正文也称 16 任务 30%。子集挑选标准未明确，存在选择性报告的观感。
3. **LCM 只管局部、定长块**（作者自承）: 对需跨很长时域、多阶段依赖或延迟效应的任务可能不足；且 LCM 用 Mamba 但消融未单独验证 Mamba vs 朴素 RNN/注意力的必要性。
4. **未端到端联合训练**: GPM/LCM/流策略分阶段冻结训练，可能非最优；先验均值用矩匹配单高斯逼近高斯混合也是近似。

### 潜在改进方向
1. 在线更新记忆库（巩固/遗忘/不确定性感知检索），支持持续学习与分布漂移下的鲁棒检索（作者已列为未来工作）。
2. GPM、LCM 与流策略**端到端联合训练**，避免分阶段冻结的次优。
3. 把 LCM 扩展到更长时域/多阶段建模，并对"Mamba 是否必要"、"$k$/记忆库规模/方差下限"等做更系统的敏感度分析。
4. 统一并透明化 RoboTwin 任务选取与报告口径。

### 可复现性评估
- [ ] 代码开源（提供项目主页，截至记录未确认代码/权重链接；附录提到补充材料含实时视频）
- [ ] 预训练模型（未明确释出）
- [x] 训练细节完整（Table 6 给出三阶段超参；附录 B/C 详述 GPM 构造与三阶段训练）
- [x] 数据集可获取（LIBERO/CALVIN/RoboTwin 2.0 公开；真实数据未声明释出）

---

## 速查卡片

> [!summary] OptimusVLA: Dual-Memory Augmented VLA（GPM + LCM）
> - **核心**: 给分层 VLA 配双记忆——GPM 用检索的任务级动作先验替换高斯噪声起点（缩短去噪路径、自适应减 NFE），LCM 对历史动作建模注入时序一致性偏置。
> - **方法**: 从 $\pi_{0.5}$ 初始化（3.6B）；三阶段训练（基座→Prior Head InfoNCE→LCM 残差 MSE）；FAISS 检索 top-$k$、矩匹配高斯先验、相似度自适应 $\lambda/N$；LCM = 自注意力一致性层 + Mamba 动态感知。
> - **结果**: LIBERO 98.6%、CALVIN 4.45（+13.5% vs $\pi_0$）、RoboTwin Hard 38%(8任务)/30%(16任务)；真实泛化 85%/长程 64%；NFE 3.2、推理快约 2.9×。
> - **项目**: https://cybertronagent.github.io/OptimusVLA.github.io/

---

*笔记创建时间: 2026-06-29*
