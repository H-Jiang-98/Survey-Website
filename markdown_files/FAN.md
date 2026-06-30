---
title: "Boosting Vision-Language-Action Finetuning with Feasible Action Neighborhood Prior"
method_name: "FAN (Feasible Action Neighborhood)"
authors: [Haochen Niu, Kanyu Zhang, Shuyu Yin, Qinghai Guo, Peilin Liu, Fei Wen]
year: 2026
venue: CVPR
tags: [VLA, finetuning, RFT, SFT, PPO, regularization, OOD-generalization, sample-efficiency]
zotero_collection: _inbox
image_source: online
arxiv_html: https://arxiv.org/html/2604.01570v1
created: 2026-06-29
---

# Boosting Vision-Language-Action Finetuning with Feasible Action Neighborhood Prior

## 元信息

| 项目 | 内容 |
|------|------|
| 作者 | Haochen Niu*, Kanyu Zhang*, Shuyu Yin, Qinghai Guo, Peilin Liu, Fei Wen†（*共同一作，†通讯） |
| 机构 | 上海交通大学、华为技术 |
| 会议 | CVPR 2026 |
| 类别 | VLA（Vision-Language-Action）/ 微调正则化 |
| 日期 | 2026-04（arXiv v1） |
| 项目主页 | — |
| 链接 | [arXiv](https://arxiv.org/abs/2604.01570) / [PDF](https://arxiv.org/pdf/2604.01570) |

---

## 一句话总结

> 物理动作天然存在“近等价邻域”（FAN），本文把这一几何先验写成一个朝目标高斯分布对齐的 KL 正则项，无缝插入 [[SFT]] 与 [[RFT]]（PPO），在不改架构的前提下显著提升 [[VLA]] 微调的样本效率与分布外泛化。

---

## 核心贡献

1. **形式化 FAN 概念**: 提出 [[Feasible Action Neighborhood|可行动作邻域 (FAN)]] —— 每个状态下存在一个让任务进展近乎等价的动作邻域，而非单一“正确动作”。据此揭示了“语言式 VLA 训练（one-hot / 单点监督）”与“真实物理动作几何”之间的根本错配。
2. **FAN 引导正则项**: 引入一个**朝目标高斯分布对齐的 KL 正则**，同时适用于 [[SFT]] 与 [[RFT]]。它**不改动模型架构、不改动自回归离散解码**，只把学习信号从“排他性正确（exclusive correctness）”转向“加权容忍（weighted tolerance）”。
3. **广泛验证有效性**: 在 [[OpenVLA]]、[[OpenVLA-OFT]] 两种主干、[[ManiSkill]] / [[LIBERO]] 两个基准 + 真机平台上，覆盖 SFT/RFT 两套范式、分布内与分布外（OOD），在**成功率、样本效率、OOD 泛化**三方面均一致提升（RFT 下达 90% 成功率仅需基线约 1/3 的步数）。

---

## 问题背景

### 要解决的问题
VLA 模型部署通常分两阶段：(i) 在大规模异构数据上预训练，(ii) 在具体机器人/环境上**微调**以适配本体、传感配置与物理动力学。微调有两种范式：[[SFT]]（把离线示范当标注序列做监督）与 [[RFT]]（用环境奖励通过交互优化策略）。问题是：**如何让这两种微调更高效、更能泛化到分布外。**

### 现有方法的局限
当前绝大多数 VLA 训练直接照搬语言模型的“配方”：next-token 预测 + one-hot 交叉熵，或其强化学习版本 [[PPO]] / [[GRPO]]。但物理动作与语言 token 有本质差异——**动作具有内在容忍性与近等价性**（把夹爪左移 1.0 cm 与 0.9/1.1 cm 几乎等效），而这一差异被普遍忽视，导致：
1. **SFT（小数据微调）**：概率质量塌缩到单个被示范动作，是**严重过拟合**的信号 —— 学到一个“尖刺状”分布、几乎平凡的 FAN，对扰动毫无鲁棒性。
2. **RFT**：虽然最终能探索出更宽的动作分布从而泛化更好，但**样本效率极低** —— 缺乏显式正则时，智能体必须靠探索“隐式地”重新发现这一有益性质，耗费大量训练步。

### 本文的动机
从**几何与物理一致性**视角重审 VLA 训练：策略分布的“形状”就是其隐式 FAN 的代理（proxy）。作者通过经验观察（Figure 1）发现策略分布形状与泛化性能高度相关——尖刺分布 = 平凡 FAN = 差泛化；宽平滑分布 = 较大鲁棒 FAN = 好泛化。因此，与其让模型靠运气探索出宽分布，不如**显式地用一个结构化高斯先验去“塑形”策略分布**，主动注入物理动作的容忍几何。
作者强调本方法**与强化学习里的熵最大化先验本质不同**：熵最大化是无结构的、用于鼓励探索；而本文施加的是由 FAN 形状决定的**结构化先验**，给出更直接、更样本高效的泛化路径。

---

## 方法详解

### 总体思路

整个方法不引入任何新模块，而是给 VLA 策略 $\pi_\theta(a\mid s,l)$ 加一个**正则项**，使其输出分布在峰值附近变成平滑、单峰、局部连续的高斯形状（即一个非平凡的 FAN）。具体落地为两个实例：
- **FAN-SFT**：在监督损失上叠加 FAN 正则（动态协方差）；
- **FAN-PPO**：在信赖域策略优化目标上叠加 FAN 正则（固定协方差），并给出最优策略的闭式结构。

输入/输出沿用原 VLA：输入语言指令 $l$ + 状态/观测 $s$，离散自回归输出动作 token；唯一改变的是训练目标。

### 预备：把 VLA 建模为 MDP

把 VLA 环境建模为指令条件 MDP $(S,A,L,r,P,\gamma)$。目标是学习指令条件策略 $\pi:S\times L\to\Delta(A)$ 最大化状态值。定义状态-动作值 $Q^\pi(s,a,l)$、状态值 $V^\pi(s,l)=\sum_a\pi(a|s,l)Q^\pi(s,a,l)$、优势 $A^\pi(s,a,l)=Q^\pi(s,a,l)-V^\pi(s,l)$。状态空间与动作空间均假设为**可数集**（便于把策略写成对动作求和的离散分布，这是后续闭式推导的关键前提）。

### 核心模块

#### 模块1: Feasible Action Neighborhood（可行动作邻域，FAN）

**设计动机**: 形式化“物理动作的容忍区域”。用值函数 $Q(s,a)$ 刻画长期表现，把最优动作附近一圈“近等价动作”定义为一个集合。

**具体实现 / 定义**（Definition 1）：设 $a^*(s)=\arg\max_{a'\in A}Q(s,a')$ 为状态 $s$ 的最优动作；给定容忍度 $\delta>0$，FAN 记为 $\mathbb{N}_\delta(s)$，是包含 $a^*(s)$ 的**连通集**，落在所有“ $Q$ 值近等价”的动作内（见公式3）。作者论断：对任何良定义的物理任务，该邻域**非平凡**（在 $a^*(s)$ 周围有非零体积）。
关键洞察：虽然 FAN 用 $Q$ 函数定义，但策略 $\pi(a|s)$ **隐式编码**了它——对常见策略形式（如对 $Q$ 做 softmax），相近 $Q$ 值的动作被赋予相近概率，因此**策略分布的形状就是 FAN 的可观测代理**：尖刺窄分布 ↔ 小的平凡 FAN；宽平滑分布 ↔ 较大鲁棒 FAN。

#### 模块2: FAN-Guided Regularizer（FAN 引导正则项）

**设计动机**: 物理 FAN 的关键性质是**单峰性、平滑性、局部连续性**，这恰好被高斯分布很好地建模。于是把“塑形”落成一个朝目标高斯对齐的 KL 散度（公式4）。

**具体实现**:
- 目标高斯 $\mathcal{N}(\mu(s),\Sigma(s))$ 的均值 $\mu(s)=\arg\max_a\pi(a|s)$ 取**策略自身预测的最优动作**（即围绕当前峰值塑形，不引入外部标签）；
- 协方差 $\Sigma(s)$ 直接控制 FAN 的大小；
- 该正则把过度自信的“尖刺”改造成平滑、鲁棒的邻域，可无缝叠加进 SFT 与 RFT 目标。

#### 模块3: FAN-SFT（监督微调实例）

**设计动机**: 监督目标本身稳定，可以承受“随策略动态变化”的目标分布。

**具体实现**: 在 SFT 负对数似然损失上叠加 FAN-KL（见公式5）。协方差**不固定**，而由策略自身方差动态给出：

$$
\Sigma(s)=\operatorname{diag}\!\left(\sum_{a\in A}\pi(a|s,l)\,(a-\mu(s))^2\right)
$$

这种自适应协方差让策略按当前几何性质去逼近高斯形状。

#### 模块4: FAN-PPO（强化微调实例）

**设计动机**: 把 FAN 正则纳入信赖域型策略优化，在“策略改进”与“策略塑形”之间做有原则的权衡。

**具体实现**:
- 约束优化目标（公式6）= 重要性加权优势 − $\alpha\cdot$FAN-KL，s.t. 与旧策略的 KL ≤ $\epsilon$；
- 为保证训练稳定，协方差**固定**为 $\Sigma=\sigma^2 I$（$\sigma$ 为超参，控制目标 FAN 大小），与 SFT 的动态协方差形成对照——固定目标形状能“锚定”策略更新；
- 该约束问题有**闭式最优解**（Proposition 1，见公式7），揭示了正则机制；
- 工程上把正则并入 PPO 的 clip 策略损失，得到 FAN-PPO 损失（公式8）。

---

### 关键公式与机制

#### 公式1: [[SFT]] 损失（基线）

$$
\mathcal{L}_{\text{SFT}}(\theta)=-\frac{1}{n}\sum_{i=1}^{n}\sum_{t=0}^{K^{i}-1}\log\pi_{\theta}(a_{t}^{i}\mid s_{t}^{i},l^{i})
$$

**含义**: 在 $n$ 条专家示范上最大化专家动作的对数似然（即标准模仿学习负对数似然）。

**符号说明**:
- $\pi_\theta$: 参数 $\theta$ 的 VLA 策略；$K^i$: 第 $i$ 条轨迹长度
- $(s_t^i,a_t^i,l^i)$: 第 $i$ 条示范在 $t$ 时刻的状态、动作与指令

#### 公式2: 信赖域型策略优化目标（[[PPO]] 基线）

$$
\max_{\pi}\ \mathbb{E}_{s\sim d^{\pi_t}_{\mu},\,a\sim\pi_t}\!\left[\frac{\pi(a|s,l)}{\pi_t(a|s,l)}A^{\pi_t}(s,a,l)\right]
\quad \text{s.t.}\ \mathbb{E}_{s\sim d^{\pi_t}_{\mu}}\!\big[D_{\text{KL}}(\pi(\cdot|s,l)\|\pi_t(\cdot|s,l))\big]\le\epsilon
$$

**含义**: 在“与旧策略 KL 不超过 $\epsilon$”的信赖域内最大化期望优势，保证更新幅度受限、训练稳定。

**符号说明**:
- $d^{\pi_t}_\mu$: 折扣状态访问分布；$A^{\pi_t}$: 旧策略下的优势
- $\pi/\pi_t$: 重要性比；$\epsilon$: 信赖域半径
- 实践中 PPO 用 clip 函数替代显式 KL 约束

#### 公式3: FAN 定义

$$
\mathbb{N}_{\delta}(s)\subseteq\left\{a\in A:\ Q(s,a^{*}(s))-Q(s,a)\le\delta\right\}
$$

**含义**: FAN 是包含最优动作 $a^*(s)$ 的连通集合，集合内所有动作与最优动作的 $Q$ 值差不超过容忍度 $\delta$，即“近等价动作”。

**符号说明**:
- $a^*(s)=\arg\max_{a'\in A}Q(s,a')$: 最优动作
- $\delta>0$: 容忍度；$\mathbb{N}_\delta(s)$: 该状态的可行动作邻域

#### 公式4: FAN 引导正则项

$$
\mathcal{L}_{\text{FAN}}=\mathbb{E}_{s}\!\left[D_{\text{KL}}\!\big(\pi(\cdot|s)\,\|\,\mathcal{N}(\cdot|\mu(s),\Sigma(s))\big)\right]
$$

**含义**: 把策略分布朝一个以其自身峰值为均值的目标高斯对齐，从而注入单峰、平滑、局部连续的 FAN 几何。

**符号说明**:
- $\mathcal{N}(\cdot|\mu(s),\Sigma(s))$: 目标高斯密度
- $\mu(s)=\arg\max_a\pi(a|s)$: 策略自身预测的最优动作（均值）
- $\Sigma(s)$: 协方差，控制目标 FAN 大小（SFT 动态、RFT 固定）

#### 公式5: FAN-SFT 损失

$$
\mathcal{L}_{\text{FAN-SFT}}(\theta)=-\frac{1}{n}\sum_{i=1}^{n}\sum_{t=0}^{K^{i}-1}\Big(\log\pi_{\theta}(a_{t}^{i}|s_{t}^{i},l^{i})+\alpha\, D_{\text{KL}}\big(\pi_{\theta}(\cdot|s_{t}^{i},l^{i})\,\|\,\mathcal{N}(\cdot|\mu(s_{t}^{i}),\Sigma(s_{t}^{i}))\big)\Big)
$$

**含义**: SFT 负对数似然 + 系数 $\alpha$ 的 FAN-KL 正则；协方差按公式 $\Sigma(s)=\operatorname{diag}(\sum_a\pi(a|s,l)(a-\mu(s))^2)$ **动态**取策略当前方差。

**符号说明**:
- $\alpha$: 正则强度系数
- 动态 $\Sigma$: 让目标高斯随策略当前几何自适应（监督目标稳定，可承受可变目标）

#### 公式6: FAN-PPO 约束优化目标

$$
\max_{\pi}\ \mathbb{E}_{s\sim d^{\pi_t}_{\mu},\,a\sim\pi_t}\!\left[\frac{\pi(a|s,l)}{\pi_t(a|s,l)}A^{\pi_t}(s,a,l)\right]-\alpha\,\mathbb{E}_{s\sim d^{\pi_t}_{\mu}}\!\big[D_{\text{KL}}(\pi(\cdot|s,l)\|\mathcal{N}(\cdot|\mu(s),\Sigma))\big]
$$

约束为 $\mathbb{E}_{s}[D_{\text{KL}}(\pi(\cdot|s,l)\|\pi_t(\cdot|s,l))]\le\epsilon$。

**含义**: 在信赖域内同时最大化优势并把策略朝目标高斯塑形；协方差**固定** $\Sigma=\sigma^2 I$ 以提供稳定一致的目标形状，锚定策略更新。

**符号说明**:
- $\alpha$: 朝高斯塑形的强度；$\sigma>0$: 固定目标 FAN 尺度
- $\epsilon$: 信赖域大小（朝旧策略 $\pi_t$ 拉拢的强度）

#### 公式7: 最优策略闭式形式（Proposition 1）

$$
\pi_{t+1}(a|s,l)\ \propto\ \mathcal{N}(a|\mu(s),\Sigma)^{\frac{\alpha}{\alpha+\beta^{*}}}\;\pi_{t}(a|s,l)^{\frac{\beta^{*}}{\alpha+\beta^{*}}}\;\exp\!\left(\frac{Q^{\pi_t}(s,a,l)}{\alpha+\beta^{*}}\right)
$$

**含义**: 公式6的最优解。下一策略是**旧策略 $\pi_t$ 与目标高斯 $\mathcal{N}$ 的几何插值**，再用 $\exp(Q/(\alpha+\beta^*))$ 按指数化的 $Q$ 值重加权。这清晰揭示了机制：$\alpha$ 控制朝高斯形状的拉力，$\epsilon$（通过 $\beta^*$）控制朝旧策略的拉力，两者竞争。

**符号说明**:
- $\beta^*\ge0$: 信赖域约束的最优拉格朗日乘子，由 KKT 条件决定，与 $\epsilon$ 反相关（$\epsilon$ 越小 → $\beta^*$ 越大 → 更新越保守、更靠 $\pi_t$）
- $Q^{\pi_t}$: 旧策略的状态-动作值；插值权重由 $\alpha,\beta^*$ 决定
- 证明见附录（Sec. 8，对约束优化做拉格朗日 + 对 $\pi$ 求导置零得到）

#### 公式8: FAN-PPO 实际损失

$$
\mathcal{L}_{\text{FAN-PPO}}(\theta)=-\frac{1}{K}\sum_{k=0}^{K-1}\Big[\min\Big(I^{k}_{t}\hat{A}(s_k,a_k,l),\ \operatorname{Clip}(I^{k}_{t},1-\epsilon,1+\epsilon)\hat{A}(s_k,a_k,l)\Big)\Big]+\alpha\,\mathcal{L}_{\text{FAN}}
$$

**含义**: 把 FAN 正则并入标准 PPO clip 策略损失，工程上即可训练（PPO 项用 clip 实现信赖域，再加 $\alpha$ 倍 FAN-KL）。

**符号说明**:
- $I^k_t=\pi_\theta(a_k|s_k,l)/\pi_{\theta_t}(a_k|s_k,l)$: 重要性比
- $\hat A$: 优势估计（[[GAE]] 或折扣回报减值基线）；价值函数由 MSE 损失（公式 $\mathcal{L}_{\text{Val}}$）更新

---

## 关键图表

<!-- 图片均使用 arXiv HTML 在线链接，未本地化 -->

### Figure 1: Geometric Structure of Policy Distribution / 策略分布几何结构（核心动机）

![Figure 1a](https://arxiv.org/html/2604.01570v1/x1.png)
![Figure 1b](https://arxiv.org/html/2604.01570v1/x2.png)
![Figure 1c](https://arxiv.org/html/2604.01570v1/x3.png)

**说明**: ManiSkill 任务上策略分布的几何形状。(a) SFT 预热后策略学到**狭窄尖峰**分布、FAN 极小，泛化差；(b) 随后 RFT(PPO) 把分布**展宽**，任务成功率提升；(c) 本文 [[FAN-PPO]] 显式把策略引向**鲁棒高斯形状**，取得最高成功率与最强泛化。这是全文的核心动机图——“分布形状 = FAN 大小 = 鲁棒性代理”。

### Figure 2: SFT OOD Breakdown / SFT 各 OOD 任务分项（ManiSkill）

![Figure 2](https://arxiv.org/html/2604.01570v1/x4.png)

**说明**: OpenVLA 用/不用 FAN 正则在 ManiSkill 各 OOD 任务上的 SFT 成功率。FAN-SFT 在**所有 OOD 任务上一致提升**，最大绝对增益出现在 Disturb Recep（干扰受体）任务（+7.2%）。

### Figure 3: SFT vs Data Scale / 不同数据规模下的 SFT 表现

![Figure 3a IND](https://arxiv.org/html/2604.01570v1/figs/data_scale_ind.png)
![Figure 3b Vision](https://arxiv.org/html/2604.01570v1/figs/data_scale_vision.png)
![Figure 3c Semantic](https://arxiv.org/html/2604.01570v1/figs/data_scale_semantic.png)
![Figure 3d Execution](https://arxiv.org/html/2604.01570v1/figs/data_scale_execution.png)

**说明**: 在分布内与三类 OOD（Vision/Semantic/Execution）下，随数据规模变化的平均成功率。两种方法都随数据增多而提升（伴有波动），但**绝大多数数据规模下 FAN 正则都优于基线**，说明小数据/大数据均有效。

### Figure 4: Spatial Robustness on LIBERO-Spatial / 空间鲁棒性热力图

![Figure 4](https://arxiv.org/html/2604.01570v1/x5.png)

**说明**: LIBERO-Spatial 上沿 x/y 轴施加位置扰动后的成功率热力图，左 SFT、右 FAN-SFT，颜色为成功率，黑/红虚线为各自的等成功率等高线。FAN-SFT 对位置偏移更鲁棒，例如在 $(x{=}0.05,y{=}0.05)$ 扰动下把成功率从 0.24 提到 0.36。

### Figure 5: Qualitative Spatial Perturbation / 空间扰动定性对比

![Figure 5a](https://arxiv.org/html/2604.01570v1/x6.png)
![Figure 5b](https://arxiv.org/html/2604.01570v1/x7.png)
![Figure 5c](https://arxiv.org/html/2604.01570v1/x8.png)

**说明**: 空间扰动下的定性对比。普通 SFT 仍偏向**训练时见过的位置**（过拟合熟悉空间模式），而 FAN-SFT 能正确适配到被扰动的新目标位置，体现更强空间泛化。

### Figure 6: RFT OOD Breakdown / RFT 各 OOD 任务分项

![Figure 6](https://arxiv.org/html/2604.01570v1/x9.png)

**说明**: OpenVLA 与 OpenVLA-OFT 在 PPO vs FAN-PPO 下、跨各 OOD 任务的表现。与 SFT 一致，FAN-PPO 带来稳定增益，尤其在 execution OOD 上提升最显著（>10%）。

### Figure 7: RFT Training Curves (OpenVLA) / OpenVLA 训练曲线（样本效率）

![Figure 7a rollout](https://arxiv.org/html/2604.01570v1/figs/ms_openvla_rl.png)
![Figure 7b eval](https://arxiv.org/html/2604.01570v1/figs/ms_openvla_rl_ood.png)

**说明**: OpenVLA RFT 期间的 rollout 与评估成功率曲线。FAN 正则**显著加速收敛**：达到 90% 成功率仅需基线约 **1/3** 的训练步；OOD 评估下同样更快、终值更高。这是“样本效率”主张的核心证据。

### Figure 8: RFT Training Curves (OpenVLA-OFT) / OpenVLA-OFT 训练曲线

![Figure 8a rollout](https://arxiv.org/html/2604.01570v1/figs/ms_oft_rl.png)
![Figure 8b eval](https://arxiv.org/html/2604.01570v1/figs/ms_oft_rl_ood.png)

**说明**: OpenVLA-OFT（输出动作块）上的 RFT 曲线。增益虽不如 OpenVLA 明显，但 FAN 正则仍在训练全程带来稳定提升，且 OOD 设定下优势更明显。

### Figure 9: Real-World Platform / 真机平台

![Figure 9a JAKA](https://arxiv.org/html/2604.01570v1/x10.png)
![Figure 9b1](https://arxiv.org/html/2604.01570v1/figs/real_world1.png)
![Figure 9b2](https://arxiv.org/html/2604.01570v1/figs/real_world2.png)

**说明**: (a) 7-DoF JAKA 机械臂 + 平行夹爪 + 固定 Intel RealSense D455 第三人称相机；(b) 把物体放进盒子的任务，从**未见过的空间位置**评估空间鲁棒性。

### Figure 10: LIBERO-Spatial Rollouts / LIBERO 仿真扰动执行序列（附录）

![Figure 10 Task1](https://arxiv.org/html/2604.01570v1/x11.png)
![Figure 10 Task2](https://arxiv.org/html/2604.01570v1/x12.png)
![Figure 10 Task3](https://arxiv.org/html/2604.01570v1/x13.png)
![Figure 10 Task4](https://arxiv.org/html/2604.01570v1/x14.png)

**说明**: LIBERO-Spatial 四个任务在对盘子（目标放置位置）施加扰动下的执行序列。Task 1/2 沿 x 轴扰动、Task 3/4 沿 y 轴扰动；每子图上行为基线 OpenVLA+SFT（失败），下行为 OpenVLA+FAN-SFT（成功），显示更强空间鲁棒性。

### Figure 11–25: RFT Qualitative Rollouts / RFT 定性执行对比（附录，每图上=PPO 失败，下=FAN-PPO 成功）

![Figure 11 IND](https://arxiv.org/html/2604.01570v1/x15.png)
![Figure 12 Table](https://arxiv.org/html/2604.01570v1/x16.png)
![Figure 13 Texture-w](https://arxiv.org/html/2604.01570v1/x17.png)
![Figure 14 Texture-s](https://arxiv.org/html/2604.01570v1/x18.png)
![Figure 15 Noise-w](https://arxiv.org/html/2604.01570v1/x19.png)
![Figure 16 Noise-s](https://arxiv.org/html/2604.01570v1/x20.png)
![Figure 17 obj](https://arxiv.org/html/2604.01570v1/x21.png)
![Figure 18 Recep](https://arxiv.org/html/2604.01570v1/x22.png)
![Figure 19 Instruct](https://arxiv.org/html/2604.01570v1/x23.png)
![Figure 20 M-obj IND](https://arxiv.org/html/2604.01570v1/x24.png)
![Figure 21 M-obj OOD](https://arxiv.org/html/2604.01570v1/x25.png)
![Figure 22 M-obj OOD](https://arxiv.org/html/2604.01570v1/x26.png)
![Figure 23 Obj.Pos.](https://arxiv.org/html/2604.01570v1/x27.png)
![Figure 24 Robot Pose](https://arxiv.org/html/2604.01570v1/x28.png)
![Figure 25 Obj.Rep.](https://arxiv.org/html/2604.01570v1/x29.png)

**说明**: 覆盖 in-distribution、vision OOD（Table/Texture-w/-s/Noise-w/-s）、semantic OOD（obj/Recep/Instruct/M-obj IND&OOD）、execution OOD（Obj. Pos./Robot Pose/Obj. Rep.）共 15 个 RFT 任务的逐帧轨迹；统一上行 OpenVLA+PPO（失败）、下行 OpenVLA+FAN-PPO（成功）。

### Figure 26–27: Parameter Sensitivity (RFT) / RFT 超参敏感性（附录）

![Figure 26 alpha](https://arxiv.org/html/2604.01570v1/x31.png)
![Figure 27 sigma-a](https://arxiv.org/html/2604.01570v1/x32.png)
![Figure 27 sigma-b](https://arxiv.org/html/2604.01570v1/x33.png)

**说明**: 在 ManiSkill PutOnPlateInScene25Main-v3 上，OpenVLA+FAN-PPO 对正则系数 $\alpha$（Fig 26）与目标标准差 $\sigma$（Fig 27）的敏感性；分别给出训练子集 rollout 成功率与测试子集评估成功率。

### Figure 28: RFT Training Curves (Appendix) / RFT 训练曲线补充

![Figure 28a](https://arxiv.org/html/2604.01570v1/x34.png)
![Figure 28b](https://arxiv.org/html/2604.01570v1/x35.png)

**说明**: OpenVLA RFT 的补充训练曲线。

### Figure 29: Real-World Hardware Setup / 真机硬件设置（附录）

![Figure 29a RealSense](https://arxiv.org/html/2604.01570v1/figs_supp/realsense.jpg)
![Figure 29b JAKA](https://arxiv.org/html/2604.01570v1/figs_supp/jakamini.png)
![Figure 29c Setup](https://arxiv.org/html/2604.01570v1/x36.jpg)

**说明**: (a) 用于感知的 RGB-D 相机（Intel RealSense）；(b) 用于操作的 JAKA minicobo 机械臂；(c) 整体工作区：桌面摆三类目标物体，任务是把它们放进黑色盒子。

### Figure 30: Real-World Rollouts / 真机执行序列（附录）

![Figure 30 Task1](https://arxiv.org/html/2604.01570v1/x37.png)
![Figure 30 Task2](https://arxiv.org/html/2604.01570v1/x38.png)
![Figure 30 Task3](https://arxiv.org/html/2604.01570v1/x39.png)
![Figure 30 Task4](https://arxiv.org/html/2604.01570v1/x40.png)

**说明**: 真机 object-in-box 四任务时序关键帧，上行 OpenVLA+SFT、下行 OpenVLA+FAN-SFT。Task-1 分布内（两法均成功）；Task 2–4 分别扰动物体位姿、机械臂位姿、盒子位置，FAN-SFT 在扰动下更鲁棒。

---

### Table 1: SFT Results on ManiSkill / ManiSkill 上的 SFT 结果（成功率 %）

| Method | In-Distribution | Vision | Semantic | Execution | Avg. (OOD) |
|--------|-----------------|--------|----------|-----------|------|
| RL4VLA [31] | 88.5 | 74.0 | 61.8 | 46.2 | 60.7 |
| OpenVLA + SFT [23] | 78.1 ±3.1 | 76.6 ±1.9 | 57.4 ±0.9 | 40.4 ±0.8 | 58.1 |
| **OpenVLA + FAN-SFT (Ours)** | **89.8 ±0.8** | **81.7 ±1.1** | **63.5 ±1.5** | **44.8 ±0.5** | **63.3** |
| △ Improvement | +11.7 | +5.1 | +6.1 | +4.4 | +5.2 |

**说明**: FAN-SFT 相对基线 OpenVLA+SFT，分布内 +11.7%、OOD 平均 +5.2%；注意基线 OpenVLA+SFT 的 IND（78.1）甚至低于 RL4VLA（88.5），而本文方法（89.8）反超，说明正则有效缓解了小数据过拟合。

### Table 2: RFT Results on ManiSkill / ManiSkill 上的 RFT 结果（成功率 %）

| Method | In-Distribution | Vision | Semantic | Execution | Avg. (OOD) |
|--------|-----------------|--------|----------|-----------|------|
| RL4VLA [31] | 93.8 | 76.6 | 72.8 | 78.3 | 75.9 |
| OpenVLA + PPO | 95.9 ±3.2 | 80.1 ±0.1 | 79.7 ±2.0 | 85.8 ±1.8 | 81.9 |
| **OpenVLA + FAN-PPO (Ours)** | **97.4 ±0.7** | **85.0 ±4.0** | **86.7 ±1.3** | **92.6 ±1.5** | **88.1** |
| △ Improvement | +1.5 | +4.9 | +7.0 | +6.9 | +6.2 |
| OpenVLA-OFT + PPO | 92.3 ±2.5 | 84.9 ±1.1 | 49.0 ±0.6 | 55.9 ±1.2 | 63.3 |
| **OpenVLA-OFT + FAN-PPO (Ours)** | **97.3 ±1.3** | **88.1 ±2.2** | **58.6 ±1.0** | **67.0 ±2.2** | **71.2** |
| △ Improvement | +5.0 | +3.2 | +9.6 | +11.1 | +7.9 |

**说明**: 两种主干上 FAN-PPO 在分布内与全部 OOD 均一致提升；execution OOD 提升最大（OpenVLA +6.9、OpenVLA-OFT +11.1）。OpenVLA-OFT 的 execution OOD 基线很低（49.0/55.9），正则带来更显著的相对修复。

### Table 3: Real-World Results / 真机评估（成功次数 / 总试验次数，每任务 30 试）

| Method | Task-1 (IND) | Task-2 (物体位姿扰动) | Task-3 (机械臂位姿扰动) | Task-4 (盒子位置扰动) |
|--------|------|------|------|------|
| OpenVLA [23] | 19/30 | 7/30 | 7/30 | 1/30 |
| **OpenVLA + FAN-SFT** | **22/30** | **12/30** | **17/30** | **7/30** |

**说明**: 真机上 FAN-SFT 在所有任务尤其扰动任务（Task 2–4）显著优于基线；Task-3 从 7/30 提到 17/30、Task-4 从 1/30 提到 7/30，验证空间鲁棒性迁移到物理世界。

### Table 7: OOD Task Variant Definitions / OOD 任务变体定义（附录，共 15 类）

| 类别 | 变体 | 定义 |
|------|------|------|
| Vision | Table | 未见桌面 |
| Vision | Texture-w / -s | 动态纹理（弱/强）|
| Vision | Noise-w / -s | 动态噪声（弱/强）|
| Semantic | Obj. | 未见物体 |
| Semantic | Recep. | 未见受体（容器）|
| Semantic | Instruct | 未见指令 |
| Semantic | M-obj. (IND) | 多物体（均见过）|
| Semantic | M-obj. (OOD) | 多物体（均未见）|
| Semantic | Disturb Recep. | 干扰受体 |
| Semantic | M-Recep. | 多受体（均未见）|
| Semantic | Obj. Pos. | 未见位置（物体&受体）|
| Execution | Robot Pose | 未见机器人初始位姿 |
| Execution | Obj. Rep. | 回合中途物体重定位 |

**说明**: ManiSkill OOD 评估的扰动定义：前 5 类视觉 OOD、中间 8 类语义 OOD、后若干执行 OOD。是 Figure 2/6 与 Table 8/9/14 的任务字典。

### Table 11: SFT Sensitivity to α / FAN-SFT 对正则系数 $\alpha$ 的敏感性（附录，ManiSkill）

| $\alpha$ | IND | Vision | Semantic | Execution | Avg. (OOD) |
|------|------|------|------|------|------|
| 0.0 | 78.1 | 76.3 | 57.6 | 41.0 | 58.3 |
| 1e-4 | 81.8 | 76.2 | 58.0 | 44.6 | 59.6 |
| 1e-3 | 87.0 | 75.6 | 60.9 | 44.2 | 60.2 |
| 1e-2 | 89.6 | 77.5 | 62.9 | 46.5 | 62.3 |
| 2e-2 | 88.4 | 77.5 | 62.9 | 49.1 | 63.2 |
| **5e-2** | **89.8** | **81.7** | **63.5** | 44.8 | **63.3** |
| 0.1 | 82.8 | 76.0 | 60.8 | 45.0 | 60.6 |
| 1.0 | 83.9 | 78.8 | 60.4 | 43.6 | 60.9 |
| 2.0 | 80.7 | 76.5 | 59.3 | 45.0 | 60.3 |

**说明**: $\alpha$ 在 $[10^{-2},5\times10^{-2}]$ 区间最优（OOD 平均 62–63.3）；$\alpha{=}0$ 退化为普通 SFT（58.3），$\alpha$ 过大（≥0.1）反而下降——存在“塑形不足/过度塑形”的甜区。

### Table 12: Comparison to Label Smoothing / 与标签平滑对比（附录，ManiSkill）

| Method | IND | Vision | Semantic | Execution | Avg. (OOD) |
|--------|------|------|------|------|------|
| Ori (SFT) | 78.1 | 76.6 | 57.4 | 40.4 | 58.1 |
| + $\epsilon$=0.05 | 82.8 | 75.9 | 62.3 | 42.2 | 60.1 |
| + $\epsilon$=0.1 | 81.3 | 69.1 | 60.3 | 39.4 | 56.3 |
| + $\epsilon$=0.2 | 79.7 | 68.1 | 51.8 | 42.2 | 54.0 |
| **Ours (FAN-SFT)** | **89.8** | **81.7** | **63.5** | **44.8** | **63.3** |

**说明**: 关键消融——FAN 作为**结构化**正则明显优于无结构的[[Label Smoothing|标签平滑]]。标签平滑只在小 $\epsilon$ 略有帮助，$\epsilon$ 增大反而损害（破坏分布形状），印证“结构化高斯先验”而非单纯平滑/熵增的价值。

### Table 15: Sample Efficiency — Steps to Rollout SR (OpenVLA) / 训练集 rollout 达标步数（附录）

| Method | →60% | →70% | →80% | →90% |
|--------|------|------|------|------|
| OpenVLA + PPO | 18 | 62 | 133 | 249 |
| **OpenVLA + FAN-PPO** | 18 | **37** | **56** | **98** |

**说明**: 达 90% rollout 成功率，FAN-PPO 仅需 98 步 vs 基线 249 步（约 1/3），量化样本效率主张。

### Table 16: Sample Efficiency — Steps to Eval SR (OpenVLA) / 测试集评估达标步数（附录）

| Method | →55% | →60% | →65% | →70% | →75% |
|--------|------|------|------|------|------|
| OpenVLA + PPO | 109 | 149 | 209 | 279 | 339 |
| **OpenVLA + FAN-PPO** | **29** | **59** | **109** | **129** | **179** |

**说明**: OOD 评估同样大幅加速，例如达 75% 评估成功率 179 步 vs 339 步，约半数即可，体现泛化收敛更快。

### Table 4–6, 8–10, 13–14, 17–18: 配置与补充结果（附录摘要）

| 表 | 内容 |
|----|------|
| Table 4/5/6 | SFT 超参：分别为 ManiSkill OpenVLA（沿用 RL4VLA 配置 + 正则系数 $\alpha$）、LIBERO OpenVLA、LIBERO OpenVLA-OFT |
| Table 8/9 | ManiSkill SFT 完整结果（含 RL4VLA†直接引用值）与各 OOD 任务细分 |
| Table 10 | LIBERO 完整结果 |
| Table 13 | PPO 超参设置（含 FAN-PPO 专有项）|
| Table 14 | ManiSkill RFT 各 OOD 任务细分结果 |
| Table 17/18 | OpenVLA-OFT 训练集 rollout 达标步数 / 真机更多细节 |

**说明**: RFT 关键超参——OpenVLA 用 $\sigma{=}0.3,\alpha{=}1.0$；OpenVLA-OFT 用 $\sigma{=}0.2,\alpha{=}0.1$。

---

## 实验

### 数据集 / 基准

| 基准 | 规模 | 特点 | 用途 |
|------|------|------|------|
| [[ManiSkill]] (PutOnPlateInScene25Main-v3) | 16K 示范（运动规划采集，同 RL4VLA 规模）| 25 类 pick-and-place 基元、多物体多受体；支持 GPU 并行仿真（故用于 RFT）| SFT/RFT 训练 + 15 类扰动 OOD 评估 |
| [[LIBERO]]-Spatial | 10 任务 ×50 人类遥操作示范 | 侧重空间布局；按 LIBERO-PRO 引入不同程度空间扰动 | SFT 训练 + 位置偏移 OOD |
| 真机（JAKA + RealSense D455）| 150 条示范（每物体 50）| 7-DoF JAKA + 平行夹爪 + 第三人称 RGB | SFT 训练 + 4 任务 ×30 试 |

### 实现细节

- **VLA 主干**: [[OpenVLA]]（单步动作输出）与 [[OpenVLA-OFT]]（动作块输出，输入第三人称图+腕部相机+本体状态+语言）；均用 HuggingFace 公开预训练 checkpoint。
- **SFT**: ManiSkill 沿用 RL4VLA 配置，扩展正则系数 $\alpha$（最优区 $\sim 5\times10^{-2}$）；协方差**动态**取策略方差。
- **RFT**: 仅在支持 GPU 并行的 ManiSkill 上做；用 [[PPO]]，协方差**固定** $\sigma^2 I$。OpenVLA: $\sigma{=}0.3,\alpha{=}1.0$；OpenVLA-OFT: $\sigma{=}0.2,\alpha{=}0.1$。
- **硬件**: 全部实验在 NVIDIA A100 80GB 上完成。

### 关键实验结论

- **SFT（Table 1）**: ManiSkill IND +11.7%、OOD 平均 +5.2%；LIBERO-Spatial 上 OpenVLA 84.7→87.2、OpenVLA-OFT 95.2→98.8。
- **RFT（Table 2）**: 两主干全面提升，OOD 平均 OpenVLA +6.2、OpenVLA-OFT +7.9，execution OOD 提升 >10%。
- **样本效率（Fig 7、Table 15/16）**: 达 90% rollout 成功率仅需基线约 1/3 步数（98 vs 249）。
- **真机（Table 3）**: 扰动任务显著更鲁棒（如 Task-3 7/30→17/30）。
- **消融**: 与标签平滑对比（Table 12）证明**结构化高斯先验 > 无结构平滑**；$\alpha$ 敏感性（Table 11）显示存在最优甜区。

---

## 批判性思考

### 优点
1. **动机清晰、机制可解释**: 把“物理动作容忍性”这一直觉形式化为 FAN，并用策略分布形状作为可观测代理（Fig 1），再用闭式解（Prop. 1）说明“下一策略 = 旧策略与目标高斯的几何插值再按 $Q$ 重加权”，理论与直觉自洽。
2. **方法即插即用、零架构改动**: 仅是一个 KL 正则项，保留自回归离散解码，可同时套到 SFT 与 RFT、不同主干，工程迁移成本极低。
3. **多维度证据充分**: 成功率、样本效率、OOD 泛化三方面 + 仿真两基准两主干 + 真机；并用标签平滑对照证明“结构化”而非单纯熵增/平滑的价值。

### 局限性
1. **目标高斯的合理性偏假设**: 把 FAN 建模为单峰高斯，但很多接触丰富/多模态任务的可行动作可能是**多峰或非凸**的；强行单峰高斯塑形在这些场景可能反而有害，文中未系统讨论。
2. **RFT 仅在 ManiSkill 验证**: 因依赖 GPU 并行仿真，RFT 实验只在单一环境/单一任务族（PutOnPlate...）上做，OpenVLA-OFT 上增益明显小于 OpenVLA，普适性证据有限。
3. **超参敏感**: $\alpha$ 存在较窄甜区（Table 11，过大反伤性能），$\sigma$ 也需按主干调（0.2/0.3），缺少自动选取或自适应机制；动态 vs 固定协方差的选择也基于稳定性经验而非系统对比。
4. **真机规模小**: 单平台、4 任务、每任务 30 试，扰动种类有限，统计置信度一般。

### 潜在改进方向
1. 用混合高斯 / 流模型等**多峰目标分布**替代单峰高斯，覆盖多模态可行动作。
2. 让 $\alpha,\sigma$ 或协方差结构**可学习/自适应**（如随训练阶段或状态变化），减少手工调参。
3. 把 FAN 正则推广到 [[GRPO]] 等其它 RFT 算法与连续/流匹配动作头主干，验证范式可移植性；并在更多真机本体与更强扰动下评估。

### 可复现性评估
- [ ] 代码开源（文中未给出代码链接）
- [x] 预训练模型（使用 HuggingFace 公开 OpenVLA / OpenVLA-OFT checkpoint）
- [x] 训练细节完整（附录 Table 4–6/13 给出 SFT/PPO 超参；关键 $\alpha,\sigma$ 明确）
- [x] 数据集可获取（ManiSkill / LIBERO 公开；真机数据自建）

---

## 速查卡片

> [!summary] FAN: Feasible Action Neighborhood Prior for VLA Finetuning
> - **核心**: 物理动作存在“近等价邻域”(FAN)，标准语言式训练把概率塌缩成尖刺导致过拟合；用朝目标高斯对齐的 KL 正则显式塑形策略分布。
> - **方法**: FAN-SFT（NLL + 动态协方差 KL）与 FAN-PPO（信赖域目标 + 固定 $\sigma^2 I$ KL），后者有闭式最优解：下一策略 = 旧策略与高斯的几何插值再按 $\exp(Q/(\alpha+\beta^*))$ 重加权。不改架构、不改自回归解码。
> - **结果**: ManiSkill SFT IND +11.7%/OOD +5.2%、RFT OOD 平均 +6.2~7.9%、execution OOD >+10%；达 90% 成功率仅需基线约 1/3 步数；真机扰动任务显著更鲁棒；优于标签平滑。
> - **主干/基准**: OpenVLA & OpenVLA-OFT；ManiSkill / LIBERO / 真机 JAKA。

---

*笔记创建时间: 2026-06-29*
