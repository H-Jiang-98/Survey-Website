---
title: "SRPO: Self-Referential Policy Optimization for Vision-Language-Action Models"
method_name: "SRPO"
authors: [Senyu Fei, Siyin Wang, Li Ji, Ao Li, Shiduo Zhang, Liming Liu, Jinlong Hou, Jingjing Gong, Xianzhong Zhao, Xipeng Qiu]
year: 2026
venue: CVPR
tags: [VLA, reinforcement-learning, GRPO, world-model, reward-shaping, process-reward, self-referential]
zotero_collection: _inbox
image_source: online
arxiv_html: https://arxiv.org/html/2511.15605v2
created: 2026-06-29
---

# SRPO: Self-Referential Policy Optimization for Vision-Language-Action Models

## 元信息

| 项目 | 内容 |
|------|------|
| 作者 | Senyu Fei, Siyin Wang, Li Ji, Ao Li, Shiduo Zhang, Liming Liu, Jinlong Hou, Jingjing Gong, Xianzhong Zhao, Xipeng Qiu |
| 机构 | 复旦大学、同济大学、上海创智学院（Shanghai Innovation Institute） |
| 会议 | CVPR 2026 |
| 类别 | VLA（Vision-Language-Action）/ 强化学习后训练 |
| 日期 | 2025-11（arXiv v2） |
| 项目主页 | （未公开） |
| 链接 | [arXiv](https://arxiv.org/abs/2511.15605) / Code（未公开） |

---

## 一句话总结

> 把"模型自身在当前 batch 内成功的轨迹"当作参照系，用[[World Model|世界模型]]潜空间度量失败轨迹的"进度奖励"，从而把稀疏的 0/1 成功信号变成稠密、零额外监督的过程奖励，在 LIBERO 上 200 步 RL 内把成功率从 48.9% 拉到 99.2%。

---

## 核心贡献

1. **自参照 RL 框架（SRPO）**: 提出用**模型自己生成的成功轨迹**为参照，给失败轨迹分配[[Process Reward|过程奖励（progress-wise reward）]]，缓解 [[GRPO]] 的[[Reward Sparsity|奖励稀疏]]问题，**无需任何专家示范或人工奖励工程**。
2. **潜空间世界表征做进度度量**: 用大规模机器人视频预训练的[[World Model|世界模型]]（[[V-JEPA 2]]）潜表征替代像素级世界模型，天然捕捉跨环境可迁移的"进度模式"，免去逐任务微调，实现稳健、可泛化的轨迹比较。
3. **效率与泛化双 SOTA**: 在 [[LIBERO]] 上 200 步内达到 99.2% 成功率（相对一次性 SFT 基线 +103%），在 [[LIBERO-Plus]] 鲁棒性基准上相对提升 167%，且仅用第三视角图像+语言指令即超越使用多视角/本体/3D 输入的更强基线。

---

## 问题背景

### 要解决的问题

[[VLA]] 模型严重依赖专家示范、容易在小规模下游数据上过拟合，形成强烈的**示范偏置（demonstration bias）**，难以超越人类水平。[[强化学习|RL]] 后训练是突破这一上限的关键手段，但当前 VLA-RL（尤其是 [[GRPO]] 这类 group-based 方法）受困于**严重的奖励稀疏**：仅靠二值（成功/失败）的终局信号，会浪费失败轨迹里大量有价值的中间信息，训练效率极低。

### 现有方法的局限

- **outcome-only RL（GRPO/PPO）**: 只用稀疏终局奖励，失败轨迹几乎被直接丢弃；而机器人 rollout 成本高、耗时长，浪费失败信息代价巨大。
- **人工/任务特定过程奖励（PRM、TGRPO）**: 引入更稠密的反馈，但**依赖专家示范或人工设计的任务分解/里程碑**，与"自主学习"目标矛盾，难以规模化。
- **像素级世界模型奖励**: 跨域泛化差，通常需要大量任务特定微调；对感知微小变化敏感、信号震荡。
- **过度精细的稠密奖励**: 过细的手工 reward shaping 反而会把策略引向次优解（作者引用 Sutton/Levine 的观点）。

### 本文的动机

把监督问题从"如何获取专家标签"转化为"如何从自己的成功中提取进度奖励"。核心洞见：**世界模型的潜表征天然捕捉可迁移的行为进度模式**，因此可以在不重建环境、不做领域微调的前提下，用"失败轨迹到成功轨迹簇中心的潜空间距离"作为进度度量。作者刻意选择**轨迹级（trajectory-level）**而非逐帧的奖励，以避免过细奖励导致的次优收敛。

---

## 方法详解

### 模型架构

SRPO 是一套**策略优化框架**（而非新网络结构），叠加在 [[VLA]] 策略之上。仿真实验中策略基于改造版 [[OpenVLA]]（加入 [[Action Chunking|动作分块]]与并行解码，记作 OpenVLA*），保留 [[Llama 2]] 主干以输出**离散动作 token**——这样能直接拿到动作的 log-prob，便于策略梯度。

- **输入**: 第三视角观测 $o_t$（单张图像）+ 语言目标描述 $l$
- **策略**: $\pi_\theta(a_t \mid o_t, l)$，由预训练 VLA 参数化
- **奖励模型**: 视频预训练的潜空间[[World Model|世界模型]] [[V-JEPA 2]]（作为编码器 $\mathcal{W}$，冻结、零样本使用）
- **优化器**: 基于 [[GRPO]] 优势估计的 SRPO 目标（clipped surrogate + KL 正则）
- **训练框架**: SiiRL；流程为 one-shot SFT → SRPO 在线 RL 后训练

整体流程见 Figure 2：一次 rollout 收集成功+失败轨迹放入 **Rollout Reference Set**，用世界模型编码为潜表征，以 L2 距离度量行为相似度产生进度奖励，再做优势估计与策略优化。

### 核心模块

#### 模块1: 问题形式化（POMDP 式 rollout）

**设计动机**: 明确智能体只能看到部分观测 $o_t$，环境状态 $z_t$ 与真值奖励 $R(z_{0:T}, l)$ 均不可直接访问，只有稀疏的终局奖励。

**具体实现**:
- 观测函数 $o_t \leftarrow O(z_t)$ 把环境状态映射为部分观测；
- 策略 $\pi_\theta(a_t \mid o_t, l)$ 给出动作分布；环境转移 $z_{t+1} \sim E(\cdot \mid z_t, a_t)$；
- 由于真值奖励稀疏（0/1），作者用世界模型对失败轨迹**塑形奖励** $\hat R(o_{0:T}, \mathcal{S})$，其中 $\mathcal{S}$ 是成功轨迹的观测集合。

#### 模块2: World Progress Reward Modeling（世界进度奖励建模）

**设计动机**: 用与任务无关（task-agnostic）的潜空间世界表征，为失败轨迹提供稠密、可泛化的进度信号。

**具体实现**:
- 用世界模型编码器 $\mathcal{W}$ 把每条轨迹观测编码为潜表征 $h_i$（公式2）；
- 对**成功轨迹**的表征做 [[DBSCAN]] 聚类，得到一组代表性中心 $C$（公式3）——聚类的两个动机：(1) 一个任务可能有多种成功策略（如先放 A 再放 B 或反之），失败应与**最近的策略**比；(2) 用簇心而非单条最近成功轨迹，能抵抗单条轨迹里的噪声段（如夹爪短暂偏离），给出更"原型化、更干净"的参照；
- 失败轨迹到最近簇心的最小 L2 距离 $d_i$（公式4），距离越小奖励越大；
- 最终奖励 $g_i$：成功轨迹给 1.0，失败轨迹用激活函数 $\phi$ 把标准化距离压到 $(0,1)$（公式5）。

#### 模块3: Self-Referential Policy Optimization（自参照策略优化）

**设计动机**: 把自参照学习无缝嵌入 [[GRPO]] 的 RL 结构，用**组内相对表现**做优势估计。

**具体实现**:
- 沿用 GRPO 的概率比 $r_{i,t}$、优势 $\hat A_i$（用世界进度奖励标准化）、KL 正则 $\omega(\theta)$（公式6）；
- clipped surrogate 目标（公式7）；
- 总目标 = clipped 项的期望 + 正则项（公式8）；
- 组统计量 $\mu_{\hat R}, \sigma_{\hat R}$ 来自世界进度奖励（公式9）；
- **关键区别**：GRPO 本质上丢弃失败轨迹，而 SRPO 能从"近乎成功"的失败轨迹里识别并奖励其中富有成效的片段，从而更高效地引导策略。

### 关键公式与机制

#### 公式1: [[Trajectory Rollout|轨迹 rollout]]

$$
o_{t}=O(z_{t}),\quad a_{t}\sim\pi_{\theta}(\cdot\mid o_{t},l),\quad z_{t+1}\sim E(\cdot\mid z_{t},a_{t})
$$

**含义**: 从初始环境状态 $z_0$ 出发，迭代地生成观测-动作-状态序列 $\{(z_0,o_0,a_0,\dots,z_T,o_T)\}$。

**符号说明**:
- $O$: 把环境状态映射为部分观测的观测函数；$o_t$: 智能体在 $t$ 时刻可见的观测
- $\pi_\theta$: 由 $\theta$ 参数化的策略；$l$: 语言目标描述
- $E$: 随机环境转移；$z_t$: 环境状态（智能体不可直接访问）

#### 公式2: 世界模型编码

$$
h_{i}=\mathcal{W}\!\left(o^{(i)}_{0:T}\right)
$$

**含义**: 用冻结的世界模型编码器 $\mathcal{W}$ 把第 $i$ 条轨迹的整段观测编码为潜表征 $h_i$。

**符号说明**:
- $\mathcal{W}$: 视频预训练的潜空间世界模型（[[V-JEPA 2]]）
- $o^{(i)}_{0:T}$: 第 $i$ 条轨迹从 $0$ 到 $T$ 的观测序列；$h_i$: 该轨迹的潜表征

#### 公式3: 成功轨迹聚类

$$
C=\text{DBSCAN}(\mathcal{S})
$$

**含义**: 对成功轨迹集合 $\mathcal{S}$ 的潜表征做 [[DBSCAN]] 聚类，得到代表性中心集合 $C$。

**符号说明**:
- $\mathcal{S}=\{o^{(i)}_{0:T}; R(z^{(i)}_{0:T},l)=1,\ \forall i\}$: 成功轨迹的观测集合
- $C$: 聚类得到的成功策略簇心集合

#### 公式4: 到最近成功簇心的距离

$$
d_{i}=\min\!\left(\left\{\,\lVert h_{i}-h_{j}\rVert^{2};\ h_{j}\in C\right\}\right)
$$

**含义**: 计算失败轨迹表征 $h_i$ 到所有成功簇心的最小平方 L2 距离；距离越小代表越接近成功模式，进度越高。

**符号说明**:
- $d_i$: 失败轨迹 $i$ 到最近簇心的距离
- $h_j \in C$: 成功簇心表征

#### 公式5: 进度奖励

$$
g_{i}=\begin{cases}1.0 & \text{成功轨迹}\\[4pt] \phi\!\left(\dfrac{d_{i}-\bar{d}}{\sigma_{d}}\right) & \text{失败轨迹}\end{cases}
$$

**含义**: 成功轨迹给满分 1.0；失败轨迹把标准化后的距离经激活函数 $\phi$ 映射到 $(0,1)$。实现中 $\phi$ 为 sigmoid，并前置缩放系数 $\alpha$ 权衡"进度感知"与"结果正确性"（最优 $\alpha=0.8$，见附录 D）。

**符号说明**:
- $\phi(\cdot)$: 映射到 $(0,1)$ 的激活函数（sigmoid）
- $\bar d, \sigma_d$: 所有失败轨迹距离的均值与标准差

#### 公式6: GRPO 式概率比、优势、正则项

$$
r_{i,t}(\theta)=\frac{\pi_{\theta}(a^{(i)}_{t}\mid o^{(i)}_{t},l)}{\pi_{\theta_{old}}(a^{(i)}_{t}\mid o^{(i)}_{t},l)},\quad \hat{A}_{i}=\frac{g_{i}-\mu_{g}}{\sigma_{g}},\quad \omega(\theta)=\beta\, D_{\text{KL}}\!\left(\pi_{\theta}\,\Vert\,\pi_{ref}\right)
$$

**含义**: 重要性采样概率比、基于进度奖励标准化的优势、以及对参考策略的 KL 正则。

**符号说明**:
- $r_{i,t}$: 新旧策略的概率比；$\hat A_i$: 标准化优势（用进度奖励 $g_i$ 而非二值奖励）
- $\mu_g, \sigma_g$: 组内奖励均值/标准差；$\beta$: KL 正则系数；$\pi_{ref}$: 参考策略

#### 公式7: clipped surrogate 目标

$$
\mathcal{L}^{\text{CLIP}}_{t,i}(\theta)=\min\!\left(r_{i,t}(\theta)\hat{A}_{i},\ \text{clip}\!\left(r_{i,t}(\theta),1-\epsilon,1+\epsilon\right)\hat{A}_{i}\right)
$$

**含义**: PPO/GRPO 式的截断替代目标，限制单步策略更新幅度，保证稳定。

**符号说明**:
- $\epsilon$: 截断范围超参；$\text{clip}$: 把概率比限制在 $[1-\epsilon, 1+\epsilon]$

#### 公式8: SRPO 总目标

$$
\mathcal{L}_{\text{SRPO}}(\theta)=\mathbb{E}_{t,i}\,\mathcal{L}^{\text{CLIP}}_{t,i}(\theta)+\omega(\theta)
$$

**含义**: 在当前训练组内对时间步 $t$ 与样本 $i$ 取期望，加上对参数的 KL 正则项；为简洁省略了对任务 $l$ 与初始状态 $z_0$ 的期望。

**符号说明**:
- $\mathbb{E}_{t,i}$: 对组内时间步与样本求期望；$\omega(\theta)$: KL 正则

#### 公式9: 组统计量

$$
\mu_{\hat{R}}=\frac{1}{M}\sum_{j=1}^{M}\hat{R}_{j},\qquad \sigma_{\hat{R}}=\sqrt{\frac{1}{M}\sum_{j=1}^{M}\!\left(\hat{R}_{j}-\mu_{\hat{R}}\right)^{2}+\epsilon}
$$

**含义**: 组内进度奖励的均值与标准差，用于优势标准化；自参照机制让策略从组内相对表现学习。

**符号说明**:
- $M$: 组内样本数；$\hat R_j$: 第 $j$ 个样本的世界进度奖励；$\epsilon$: 数值稳定项

#### 公式10: Spearman 时间相关性（附录 A，奖励质量指标）

$$
\rho=\frac{1}{N}\sum_{k=1}^{N}\frac{\sum_{i=1}^{T_{k}}\!\left(x_{i}^{(k)}-\bar{x}^{(k)}\right)\!\left(y_{i}^{(k)}-\bar{y}^{(k)}\right)}{\sqrt{\sum_{i=1}^{T_{k}}\!\left(x_{i}^{(k)}-\bar{x}^{(k)}\right)^{2}\sum_{i=1}^{T_{k}}\!\left(y_{i}^{(k)}-\bar{y}^{(k)}\right)^{2}}}
$$

**含义**: 衡量进度值与帧号之间的单调相关性（在成功轨迹上计算），值越接近 1 越好。

**符号说明**:
- $N$: 任务数；$T_k$: 任务 $k$ 的轨迹长度；$x_i^{(k)}$: 帧号；$y_i^{(k)}$: 进度值

#### 公式11: 时间单调性（附录 A）

$$
M_{\text{mono}}=\frac{1}{N}\sum_{k=1}^{N}\frac{1}{T_{k}-1}\sum_{t=1}^{T_{k}-1}\mathbb{I}\!\left(r_{t+1}^{(k)}>r_{t}^{(k)}\right)
$$

**含义**: 统计进度信号在相邻步上"递增"的平均比例，越接近 100% 越单调。

**符号说明**:
- $\mathbb{I}$: 指示函数；$r_t^{(k)}$: 任务 $k$ 第 $t$ 步进度

#### 公式12: 最大均值差异 MMD（附录 A）

$$
\text{MMD}=\frac{1}{N}\sum_{k=1}^{N}\left\lVert \frac{1}{n_{k}}\sum_{i=1}^{n_{k}}\phi\!\left(R_{s,k}^{(i)}\right)-\frac{1}{m_{k}}\sum_{j=1}^{m_{k}}\phi\!\left(R_{f,k}^{(j)}\right)\right\rVert_{\mathcal{H}}^{2}
$$

**含义**: 在再生核希尔伯特空间 $\mathcal{H}$ 中度量成功/失败轨迹终局进度分布的分离度，越大越能区分。

**符号说明**:
- $R_{s,k}, R_{f,k}$: 任务 $k$ 成功/失败轨迹的终局进度值；$n_k, m_k$: 样本数；$\phi$: 核特征映射

#### 公式13: JS 散度（附录 A）

$$
\text{JSD}=\frac{1}{N}\sum_{k=1}^{N}\left[\frac{1}{2}D_{\text{KL}}\!\left(P_{\text{success}}^{(k)}\,\Vert\,M^{(k)}\right)+\frac{1}{2}D_{\text{KL}}\!\left(P_{\text{failure}}^{(k)}\,\Vert\,M^{(k)}\right)\right]
$$

**含义**: 度量成功/失败分布的信息论散度，越接近 $\ln 2$ 区分性越强。

**符号说明**:
- $M^{(k)}=\tfrac{1}{2}(P_{\text{success}}^{(k)}+P_{\text{failure}}^{(k)})$: 混合分布；$D_{\text{KL}}$: KL 散度

#### 公式14: 标准化均值差 SMD（附录 A）

$$
\text{SMD}=\frac{1}{N}\sum_{k=1}^{N}\frac{\mu_{\text{success}}^{(k)}-\mu_{\text{failure}}^{(k)}}{\sigma_{\text{pooled}}^{(k)}}
$$

**含义**: 度量成功/失败均值差的效应量，越大区分越显著；$\sigma_{\text{pooled}}^{(k)}$ 为合并标准差。

**符号说明**:
- $\mu_{\text{success}}^{(k)}, \mu_{\text{failure}}^{(k)}$: 任务 $k$ 成功/失败均值
- $\sigma_{\text{pooled}}^{(k)}=\sqrt{\dfrac{(n_k-1)(\sigma_{\text{success}}^{(k)})^2+(m_k-1)(\sigma_{\text{failure}}^{(k)})^2}{n_k+m_k-2}}$

#### 公式15: 真实世界离线 RL 的 AWR 优势（附录 G）

$$
A_{i,t}=\frac{D_{i,t}-\mu}{\sigma}
$$

**含义**: 真实机器人实验出于安全/复位成本采用离线 RL，将 [[AWR]]（Advantage-Weighted Regression）与 SRPO 进度奖励结合；$D_{i,t}=R_{i,t}-R_{i,t-1}$ 为步间增量进度。

**符号说明**:
- $D_{i,t}$: 第 $i$ 条轨迹在 $t$ 步的增量进度；$R_{i,t}$: 累计进度（由公式2–5 计算）
- $\mu, \sigma$: 跨所有轨迹的 $D_{i,t}$ 均值与标准差

---

## 关键图表

<!-- 图片均使用 arXiv HTML 在线链接，未本地化 -->

### Figure 1: Overview of SRPO / 方法概览与范式对比

![Figure 1](https://arxiv.org/html/2511.15605v2/x1.png)

**说明**: 对比三种 VLA-RL 范式：(a) [[GRPO]] 仅靠稀疏终局奖励，学习信号有限；(b) 人工 [[Process Reward|PRM]] 需要昂贵外部示范与任务特定工程；(i)(ii) 本文 SRPO 用**组内成功轨迹**+**潜空间世界表征**构造过程奖励，高效利用失败轨迹。右侧标出四大收益：SOTA 性能、训练效率、泛化、真实世界性能。这张图确立了"自参照"相对于 outcome-only 与手工 PRM 的定位。

### Figure 2: Overview of the SRPO method / 方法流程图

![Figure 2](https://arxiv.org/html/2511.15605v2/x2.png)

**说明**: SRPO 完整流水线。rollout 阶段把成功与失败轨迹都收进 **Rollout Reference Set**；每条轨迹经世界模型编码为潜表征；以 L2 距离建模行为相似度得到进度奖励；再用于优势估计与 KL 正则下的策略优化。是理解整套方法数据流的核心图。

### Figure 3: Progress estimation comparison (sim & real) / 进度估计方法对比

![Figure 3](https://arxiv.org/html/2511.15605v2/x3.png)

**说明**: 仿真 (a–c) 与真实 (d–f) 环境下三种进度估计的对比。SRPO 奖励 (a,d) 单调、物理上合理；像素级奖励 (b,e) 对感知变化敏感；[[ImageBind]] 奖励 (c,f) 因抖动运动而出现错乱趋势。定性佐证"潜空间表征捕捉物理规律"。

### Figure 4: Training with different reward formulations / 不同奖励下的训练曲线

![Figure 4](https://arxiv.org/html/2511.15605v2/x4.png)

**说明**: 三种进度奖励下的训练性能。SRPO 奖励稳定高效；像素级收敛极慢；ImageBind 初期快但约 85% 处早早停滞。说明"奖励质量直接决定能否从失败轨迹榨取有效信号"。

### Figure 5: Training efficiency SRPO vs GRPO / 训练效率对比

![Figure 5](https://arxiv.org/html/2511.15605v2/x5.png)

**说明**: (a) LIBERO-Long、(b) LIBERO-Object 上 SRPO 与 [[GRPO]] 的效率曲线。SRPO 效率斜率更陡，长程任务尤其明显——因其能从近乎成功的失败轨迹里提取信号，而 GRPO 基本丢弃失败 episode。

### Figure 6: Action space comparison (SFT vs SRPO) / 末端轨迹动作空间对比

![Figure 6](https://arxiv.org/html/2511.15605v2/x6.png)

**说明**: full-shot SFT (a) 与 SRPO 在线 RL (b) 的末端执行器轨迹分布。SRPO 探索到 SFT 未触及的区域、轨迹更分散，证明它能突破模仿学习的"示范分布约束"。

### Figure 7: End-effector trajectories across three tasks / 三任务末端轨迹可视化

![Figure 7](https://arxiv.org/html/2511.15605v2/x7.png)

**说明**: 三个任务（把碗放到柜顶 / 把黄白马克杯放进微波炉并关门 / 把字母汤和奶酪盒都放进篮子）的末端轨迹。即便初始只见一条成功示范，在线 RL 仍能发现新的空间路径与抓取位姿，体现自主习得"运动技能 + affordance"的能力。

### Figure 8: Real-world task success rates / 真实任务成功率

![Figure 8](https://arxiv.org/html/2511.15605v2/x8.png)

**说明**: 真实世界五任务上 SFT 基线 vs 本文离线 RL。对扩散式 $\pi_0$（平均 +66.8%）与自回归 $\pi_0$-FAST（平均 +86.7%）两类骨干均一致提升，验证进度奖励可迁移到真机。

### Figure 9: Reward signals on success vs failure trajectories / 成功与失败轨迹奖励信号

![Figure 9](https://arxiv.org/html/2511.15605v2/x9.png)

**说明**: (a) 成功轨迹（擦白板）奖励平滑上升，抓取暂停处有轻微回落；(b) 失败轨迹（把杯子放盘子）奖励停滞、无法到达高值，因模型找不到第二只杯子。直观展示进度奖励如何区分成功/失败。

### Figure 10: Ablation on Object suite / 消融实验

![Figure 10](https://arxiv.org/html/2511.15605v2/x10.png)

**说明**: 在 Object suite 上的消融。去掉自参照（w/o Referential，改用 50 条固定专家轨迹）后期会触顶，需约 1.4 倍步数仍次优；去掉聚类（w/o Cluster，改用单条最近成功轨迹）会拖慢后期收敛。两者都验证了对应设计的必要性。

### Figure 11: Effect of reward weight α / 奖励权重 α 的影响

![Figure 11](https://arxiv.org/html/2511.15605v2/x11.png)

**说明**: 不同 $\alpha$ 的性能排序 $\alpha=0.8 > 1.0 > 0.5 > 0.3 > 0$。$\alpha=0$（无进度奖励）最差，$\alpha=1.0$（等权）次优（过度强调进度反而分散对最终目标的注意），$\alpha=0.8$ 最优——印证"进度感知与结果正确性需平衡"。

### Figure 12: Real-world success cases / 真实世界成功案例

![Figure 12](https://arxiv.org/html/2511.15605v2/x12.png)

**说明**: 真实案例：放梨时目标被中途移动可动态重规划；折毛巾可处理可变形物体；从五张牌中识别 Joker 体现语义理解。展示真实环境下的适应性与任务感知能力。

### Figure 13–15: Reward curves of different methods / 不同方法的奖励曲线（附录 B）

![Figure 13](https://arxiv.org/html/2511.15605v2/x13.png)
![Figure 14](https://arxiv.org/html/2511.15605v2/x14.png)
![Figure 15](https://arxiv.org/html/2511.15605v2/x15.png)

**说明**: Fig 13–14 为成功轨迹（两部分）、Fig 15 为失败轨迹上三种方法的帧号-进度奖励曲线。像素级奖励只在最后几帧陡增、无法评估含多子任务的长程任务；ImageBind 缺乏物理直觉、震荡多；SRPO 更平滑稳定。

### Figure 16: Trajectories generated by Cosmos-Predict2 / 像素级世界模型生成结果（附录 E）

![Figure 16](https://arxiv.org/html/2511.15605v2/x16.png)

**说明**: 用 [[Cosmos-Predict2]]-14B 零样本按语言指令生成参考视频，场景一致性差。佐证"像素级世界模型路线"虽直观但实践中代价高、不如 SRPO 的潜表征路线划算、可泛化。

### Table 1: Performance comparison on LIBERO / LIBERO 基准主结果

| Model | Policy Input | Spatial | Object | Goal | Long | Avg |
|-------|--------------|---------|--------|------|------|-----|
| OpenVLA | T+I | 84.7 | 88.4 | 79.2 | 53.7 | 76.5 |
| Pi0+fast | T+W+P+I | 96.4 | 96.8 | 88.6 | 60.2 | 85.5 |
| Pi0 | T+W+P+I | 96.8 | 98.8 | 95.8 | 85.2 | 94.2 |
| SmolVLA | T+W+P+I | 93.0 | 94.0 | 91.0 | 77.0 | 88.8 |
| WorldVLA | T+I | 85.6 | 89.0 | 82.6 | 59.0 | 79.1 |
| NORA | T+I | 92.2 | 95.4 | 89.4 | 74.6 | 87.9 |
| CoT-VLA | T+I | 87.5 | 91.6 | 87.6 | 69.0 | 81.1 |
| UniVLA | T+I | 96.5 | 96.8 | 95.6 | 92.0 | 95.2 |
| TraceVLA | T+I | 84.6 | 85.2 | 75.1 | 54.1 | 74.8 |
| MolmoAct | T+I | 87.0 | 95.4 | 87.6 | 77.2 | 86.6 |
| ThinkAct | T+I | 88.3 | 91.4 | 87.1 | 70.9 | 84.4 |
| GR00T N1 | T+I | 94.4 | 97.6 | 93.0 | 90.6 | 93.9 |
| 3D-CAVLA | T+W+P+D+I | 98.2 | 99.8 | 98.2 | 96.1 | 98.1 |
| OpenVLA-OFT | T+W+P+I | 96.2 | 98.3 | 96.2 | 90.7 | 95.3 |
| OpenVLA*-Full | T+I | 91.6 | 95.3 | 90.6 | 86.5 | 91.0 |
| TGRPO | T+I | 90.4 | 92.2 | 81.0 | 59.2 | 80.7 |
| GRAPE | T+I | 88.5 | 92.1 | 83.1 | 57.2 | 80.2 |
| VLA-RL | T+I | 90.2 | 91.8 | 82.2 | 59.8 | 81.0 |
| World-Env | T+I | 87.6 | 86.6 | 86.4 | 57.8 | 79.6 |
| SimpleVLA-RL | T+I | 98.2 | 98.7 | 98.8 | 91.7 | 96.9 |
| RIPT-VLA | T+W+P+I | 99.0 | 98.6 | 98.6 | 93.8 | 97.5 |
| RLinf | T+W+P+I | 99.4 | 99.8 | 98.8 | 94.0 | 98.0 |
| OpenVLA*-One（基线） | T+I | 63.6 | 54.9 | 59.6 | 17.3 | 48.9 |
| + Offline SRPO | T+I | 92.5 | 96.8 | 92.0 | 88.7 | 92.5 |
| **+ Online SRPO (Ours)** | **T+I** | **98.8** | **100.0** | **99.4** | **98.6** | **99.2** |

**说明**: 输入记号 T=第三视角、I=指令、P=本体、W=腕部图像、D=深度。仅用 T+I 的 SRPO 总均值 **99.2%** 居首，超过用更多模态的 RLinf(98.0)、RIPT-VLA(97.5)、3D-CAVLA(98.1)。相对一次性 SFT 基线 48.9% 提升 +103%；且在线 SRPO 全面优于离线 SRPO。证明 (1) 在线后训练有效、(2) 自参照进度奖励优于稀疏/手工奖励、(3) 仅凭视觉输入即可 SOTA。

### Table 2: Robustness on LIBERO-Plus / 鲁棒性泛化（7 维扰动）

| Model | Camera | Robot-Init | Language | Light | Background | Noise | Layout | Total |
|-------|--------|-----------|----------|-------|------------|-------|--------|-------|
| **Zero-Shot** | | | | | | | | |
| Pi0 | 13.8 | 6.0 | 58.8 | 85.0 | 81.4 | 79.0 | 68.9 | 53.6 |
| Pi0+fast | 65.1 | 21.6 | 61.0 | 73.2 | 73.2 | 74.4 | 68.8 | 61.6 |
| UniVLA | 1.8 | 46.2 | 69.6 | 69.0 | 81.0 | 21.2 | 31.9 | 42.9 |
| WorldVLA | 0.1 | 27.9 | 41.6 | 43.7 | 17.1 | 10.9 | 38.0 | 25.0 |
| OpenVLA | 0.8 | 3.5 | 23.0 | 8.1 | 34.8 | 15.2 | 28.5 | 15.6 |
| OpenVLA-OFT | 56.4 | 31.9 | 79.5 | 88.7 | 93.3 | 75.8 | 74.2 | 69.6 |
| OpenVLA-OFT_w | 10.4 | 38.7 | 70.5 | 76.8 | 93.6 | 49.9 | 69.9 | 55.8 |
| OpenVLA-OFT_m | 55.6 | 21.7 | 81.0 | 92.7 | 91.0 | 78.6 | 68.7 | 67.9 |
| NORA-long | 2.2 | 37.0 | 65.1 | 45.7 | 58.6 | 12.8 | 62.1 | 39.0 |
| RIPT-VLA | 55.2 | 31.2 | 77.6 | 88.4 | 91.6 | 73.5 | 74.2 | 68.4 |
| OpenVLA*-Full | 12.8 | 39.4 | 68.5 | 63.4 | 75.0 | 34.8 | 62.7 | 51.1 |
| OpenVLA*-One | 3.2 | 14.0 | 27.6 | 25.7 | 32.7 | 6.4 | 26.4 | 19.4 |
| **+ Online SRPO** | **17.1** | **51.0** | **81.8** | **70.4** | **88.9** | **35.3** | **72.4** | **59.6** |
| **With Augmented Data** | | | | | | | | |
| OpenVLA-OFT+ | 92.8 | 30.3 | 85.8 | 94.9 | 93.9 | 89.3 | 77.6 | 79.5 |
| OpenVLA*-Full | 69.4 | 49.6 | 66.3 | 88.2 | 88.5 | 78.7 | 70.3 | 73.0 |
| OpenVLA*-One | 12.8 | 23.0 | 30.0 | 42.0 | 49.6 | 23.3 | 34.5 | 30.7 |
| **+ Online SRPO** | **83.4** | **62.0** | **73.6** | **97.2** | **97.7** | **85.7** | **75.2** | **82.1** |

**说明**: SRPO 应用于一次性 SFT 策略，在 7 个维度上不仅碾压自身基线（zero-shot 19.4→59.6；augmented 30.7→82.1，+167%），还**超过 full-shot SFT 基线**；增强数据设定下 82.1 甚至超过使用腕部图像+本体的 OpenVLA-OFT+ (79.5)。说明在线交互带来的轨迹多样性比更复杂输入模态更能提升泛化。

### Table 3: Progress Reward Benchmark Results / 进度奖励质量基准

| Method | SC↑ | Mono↑ | MMD↑ | JS↑ | SMD↑ |
|--------|------|-------|------|------|------|
| Pixel-level | 0.125 | 0.498 | 0.274 | 0.548 | 2.100 |
| ImageBind | 0.957 | 0.837 | 0.356 | 0.408 | 18.111 |
| **SRPO (Ours)** | **0.998** | **0.992** | **0.615** | **0.572** | **188.799** |

**说明**: 在 700 条成功 + 300 条失败轨迹上评估五项指标（均越高越好）：SC=Spearman 相关、Mono=单调性、MMD=最大均值差异、JS=JS 散度、SMD=标准化均值差。SRPO 五项全面领先，尤其 SMD（188.8 vs ImageBind 18.1）显示对成功/失败的极强区分力。

### Table 4: Progress Reward Benchmark on real-robot / 真机进度奖励质量（附录 G）

| Task | SC | Mono | MMD | JS | SMD |
|------|------|------|------|------|------|
| Put Apple | 0.987 | 0.975 | 0.589 | 0.562 | 165.3 |
| Put Pear | 0.991 | 0.982 | 0.601 | 0.578 | 172.8 |
| Fold Towel | 0.984 | 0.968 | 0.572 | 0.549 | 158.6 |
| Wipe Board | 0.993 | 0.986 | 0.624 | 0.591 | 181.2 |
| Select Poker | 0.989 | 0.979 | 0.595 | 0.569 | 169.5 |
| **Average** | **0.989** | **0.978** | **0.596** | **0.570** | **169.5** |

**说明**: 五个真机任务（每任务 30 成功 + 20 失败轨迹）上的奖励质量。SC/Mono 近乎完美、MMD/SMD 分离度强，证明潜空间进度奖励在 sim-to-real 域移下仍稳健泛化。

---

## 实验

### 数据集 / 基准

| 基准 | 规模 | 特点 | 用途 |
|------|------|------|------|
| [[LIBERO]] | Goal/Spatial/Object/Long 各 10 任务 | 单臂操作，四类能力 | 主实验训练/测试 |
| [[LIBERO-Plus]] | 7 维扰动（相机/初始位/语言/光照/背景/噪声/布局） | 鲁棒性泛化评测 | 泛化实验 |
| 进度奖励基准（本文构建） | 700 成功 + 300 失败轨迹 | 含视角/光照/物体/噪声/背景扰动 | 评估奖励质量 |
| 真实世界（X-ARM 7） | 5 任务（放苹果/放梨/折毛巾/擦白板/选扑克）；奖励验证每任务 30 成功+20 失败 | 含可变形物体、语义识别、动态重规划 | 真机离线 RL |

### 实现细节

- **策略骨干**: OpenVLA*（OpenVLA + 动作分块 + 并行解码），保留 [[Llama 2]] 主干输出离散动作 token（便于拿 log-prob）
- **奖励模型**: [[V-JEPA 2]]（视频预训练潜空间世界模型，冻结、零样本；视频嵌入用 FP16 推理）
- **训练框架**: SiiRL；流程 = one-shot SFT（每任务 1 条示范）→ SRPO 在线 RL
- **SFT 阶段**: 8×A100；AdamW；lr 5e-4；batch 8；最大 150,005 步（10 万步后衰减）；LoRA rank 32；开图像增强；8 动作块
- **SRPO 阶段**: lr 5e-6；每组采样 8；训练 batch 64(×8)、验证 496；mini-batch 128；进度奖励权重 $\alpha=0.8$；每任务 50 试；7 动作 token、8 动作块；prompt 512 / response 128 token；开模型 offloading
- **达成步数（高效）**: Spatial 79 步、Object 59 步、Goal 103 步、Long 219 步（相比 SFT 需数万步）
- **真机**: X-ARM 7；因安全/复位成本采用**离线 RL**，结合 [[AWR]] 与 SRPO 进度奖励

### 关键实验结论

- **主结果（Table 1）**: LIBERO 总均值 99.2% SOTA，48.9%→99.2% 相对 +103%，仅用 T+I 输入即超越多模态基线。
- **泛化（Table 2）**: LIBERO-Plus 相对 +167%，超过 full-shot SFT 与使用更多模态的 OFT+。
- **奖励质量（Table 3/4）**: 五指标全面优于像素级与 ImageBind，真机域移下仍稳健。
- **效率（Fig 5）**: 比 GRPO 效率斜率更陡，长程任务尤甚。
- **探索（Fig 6/7）**: 突破示范分布，发现新空间路径与抓取位姿。
- **消融（Fig 10）**: 自参照（vs 固定专家）与聚类（vs 单条最近）均必要。
- **超参（Fig 11）**: $\alpha=0.8$ 最优，平衡进度与结果。
- **真机（Fig 8/12）**: $\pi_0$ +66.8%、$\pi_0$-FAST +86.7%。

---

## 批判性思考

### 优点
1. **巧妙化解奖励稀疏且零额外监督**: "用自己 batch 内的成功当参照"既稠密又不引入专家示范/人工里程碑，理念干净、可规模化；消融证明它优于固定外部专家参照。
2. **潜空间世界表征的选择有充分论证**: 用 Table 3/4 + Fig 3/4/13–15 多角度证明潜表征奖励在单调性、区分度、训练稳定性上都碾压像素级与 ImageBind，且零样本可迁移到真机。
3. **效率极高、证据完整**: 200 步内达 99.2%，且明确给出各 suite 达成步数与完整超参，效率主张可信。

### 局限性
1. **依赖"组内已有成功轨迹"的冷启动假设**: 自参照需要 batch 内至少出现成功轨迹才能给失败轨迹打分；当任务太难、初期几乎无成功（如更复杂长程/接触任务）时，参照集为空，方法可能退化为 GRPO，论文未系统讨论这一失败模式。
2. **离散动作 + Llama 2 主干牺牲精度**: 作者自承为拿 log-prob 而用离散动作 token，可能损失连续控制精度；与连续动作头方法（OpenVLA-OFT）的精度差距未量化。
3. **真机用的是离线 RL（AWR）而非在线 SRPO**: 真机收益（Fig 8）来自离线变体，与仿真的在线 SRPO 不完全同源，"在线自参照"在真机的可行性仍未验证。
4. **未开源**: 暂无项目主页/代码链接，部分超参（如 batch 64×8、mini-batch 128、trajectory mini-batch 16 的具体含义）描述偏简，复现需补足细节。

### 潜在改进方向
1. 设计**冷启动机制**（如初期混入少量像素级或 ImageBind 奖励、或课程式难度递增），解决"早期无成功轨迹"的退化问题。
2. 把框架迁移到**连续动作 / 流匹配 VLA**（如真机用的 $\pi_0$），统一在线 SRPO 与真机部署，避免离线/在线两套机制。
3. 对世界模型选择做敏感度分析（V-JEPA 2 vs 其他视频世界模型），并探究 DBSCAN 超参与 $\phi$ 缩放系数 $\alpha$ 的自适应化。

### 可复现性评估
- [ ] 代码开源（暂未提供链接）
- [ ] 预训练模型（未声明 release）
- [x] 训练细节较完整（附录 F 给出 SFT/RL 两阶段超参）
- [x] 数据集可获取（LIBERO/LIBERO-Plus 公开；V-JEPA 2 公开；真机数据未声明 release）

---

## 速查卡片

> [!summary] SRPO: Self-Referential Policy Optimization for VLA
> - **核心**: 用模型自己 batch 内的成功轨迹作参照、世界模型潜空间距离做进度奖励，把稀疏 0/1 信号变稠密、零额外监督地利用失败轨迹。
> - **方法**: V-JEPA 2 编码轨迹 → DBSCAN 聚成功簇 → 失败到簇心 L2 距离经 sigmoid(×α=0.8) 得进度奖励 → 套进 GRPO 优势/clip/KL 目标；one-shot SFT 后做在线 RL。
> - **结果**: LIBERO 48.9%→99.2%（+103%，200 步内），LIBERO-Plus +167%，奖励质量五指标全面 SOTA；真机 $\pi_0$ +66.8% / $\pi_0$-FAST +86.7%。
> - **代码**: 暂未公开

---

*笔记创建时间: 2026-06-29*
