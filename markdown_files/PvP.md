---
title: "PvP: Data-Efficient Humanoid Robot Learning with Proprioceptive-Privileged Contrastive Representations"
method_name: "PvP"
authors: [Mingqi Yuan, Tao Yu, Haolin Song, Bo Li, Xin Jin, Hua Chen, Wenjun Zeng]
year: 2026
venue: CVPR
tags: [humanoid, whole-body-control, state-representation-learning, contrastive-learning, reinforcement-learning, sample-efficiency]
zotero_collection: _inbox
image_source: online
arxiv_html: https://arxiv.org/html/2512.13093v2
created: 2026-06-29
---

# PvP: Data-Efficient Humanoid Robot Learning with Proprioceptive-Privileged Contrastive Representations

## 元信息

| 项目 | 内容 |
|------|------|
| 作者 | Mingqi Yuan, Tao Yu, Haolin Song, Bo Li, Xin Jin, Hua Chen, Wenjun Zeng |
| 机构 | 香港理工大学、宁波数字孪生研究院(IDT)、东方理工(EIT)等(致谢含 LimX Dynamics) |
| 会议 | CVPR 2026 |
| 类别 | Humanoid / 人形机器人全身控制 |
| 日期 | 2025-12（arXiv v2） |
| 项目主页 | 无（实验基于开源 SRL4Humanoid 框架） |
| 链接 | [arXiv](https://arxiv.org/abs/2512.13093) / [PDF](https://arxiv.org/pdf/2512.13093) |

---

## 一句话总结

> 利用人形机器人**本体感知状态**与**特权状态**之间的内在互补性做对比学习，无需手工数据增强即可学到任务相关的紧致表征，显著提升 RL 在全身控制(WBC)任务上的样本效率与最终性能。

---

## 核心贡献

1. **PvP 对比表征框架**: 提出在[[本体感知状态|proprioceptive state]] $\bm{o}$ 与[[特权状态|privileged state]] $\bm{s}$ 之间做[[对比学习|contrastive learning]]，把"特权状态"视为本体状态的**伪增强(pseudo augmentation)**，从而免去手工数据增强，得到稳定且可迁移的改进。
2. **SRL4Humanoid 框架**: 据作者所知是**第一个统一、模块化、即插即用**的开源框架，为人形机器人学习提供了代表性[[状态表征学习|SRL]]方法(SimSiam/SPR/VAE 等)的高质量实现，SRL 与 RL 流程**完全解耦**。
3. **系统性实证研究**: 在 LimX Oli 31-DoF 人形机器人的速度跟踪与运动模仿两类任务上，系统分析了 SRL 的训练时间占比、数据占比、施加于策略/价值编码器的差异等问题,给出"如何把 SRL 与 RL 结合用于人形 WBC"的工程指引。

---

## 问题背景

### 要解决的问题
如何缓解[[强化学习|RL]]在人形[[全身控制|whole-body control]](WBC)中的**样本效率低下**问题。人形机器人动力学复杂、欠驱动、运动/操作/平衡强耦合，且为兼顾任务性能(如跟踪精度)与真机可部署性(如能效)需要优化复合奖励,使得样本复杂度极高。

### 现有方法的局限
- **传统模型法**: 难以在非平稳条件下保证实时灵活控制与鲁棒性能。
- **重建式 SRL**(如从本体状态预测根部线速度等特权信息): 倾向于**保留完整状态(含无关细节)**而非学习任务相关特征,表征质量与泛化欠佳;VAE 这类纯重建甚至会损害学习效率。
- **单模态对比 SRL**(如 PIM): 只依赖单一状态模态,**不引入特权状态信息**,无法捕捉完整的任务相关动力学,表征"欠信息化"。
- **师生蒸馏(TSD)**: 学生性能上限受教师质量/归纳偏置严格约束,教师的次优与观测错配会被直接继承;严格对齐目标会**过度正则**抑制探索,使学生坍缩到保守/平均行为;且教师预训练阶段算力昂贵,限制复现与快速迭代。

### 本文的动机
特权状态 $\bm{s}$ 天然**同时包含**本体观测(关节位置/角速度等)与特权信息(根部线速度等)。因此 $\bm{s}$ 可被看作本体状态 $\bm{o}$ 的"伪增强"——二者描述同一时刻、信息互补。对二者做对比学习既能**降低 SRL 复杂度**(无需手工增强),又能让策略侧获得访问特权信息的额外通道,从而学到更紧致、更任务相关的表征。

---

## 方法详解

### 模型架构

[[SRL4Humanoid]] 采用 **SRL 与 RL 完全解耦** 的模块化架构(见 Figure 2),以 [[PPO]] 为骨干 RL 算法:
- **输入**: 策略网络接收本体感知状态 $\bm{o}_t$ 生成动作;价值网络接收特权状态 $\bm{s}_t$ 做价值估计。
- **Backbone(RL)**: [[PPO]],策略/价值网络为 Encoder+Head 的 MLP(ELU 激活,见 Table 5)。
- **核心模块**: [[PvP]] 对比学习模块 $S_{\bm{\psi}}$,SRL 损失默认施加于**策略编码器**(也可切到价值编码器做消融)。
- **输出**: 关节动作(作为对名义关节位置的偏移量),再经固定增益的 [[PD 控制器]]跟踪。

策略被建模为参数化映射:

$$
\pi_{\bm{\theta}}:\mathcal{O}\times\mathcal{C}\rightarrow\mathcal{A}
$$

WBC 被形式化为无限时域[[POMDP]] $\mathcal{M}=(\mathcal{S},\mathcal{O},\mathcal{A},P,\Omega,R,\gamma)$,RL 目标是最大化期望折扣回报(公式1)。

### 核心模块

#### 模块1: PvP 对比表征(Proprioceptive-Privileged Contrastive Learning)

**设计动机**: 把特权状态 $\bm{s}$ 当作本体状态 $\bm{o}$ 的伪增强,用 [[SimSiam]] 式的孪生网络在两个模态之间拉近表征,免手工增强地注入特权信息。

**具体实现**:
- 对特权状态的**特权信息部分施加零掩码(Zero-Masking)**,只保留其中的本体观测,得到 $\tilde{\bm{s}}_t$(公式2);由此构造数据对 $(\bm{s},\tilde{\bm{s}})$。
- 用同一策略编码器 $f_{\bm{\theta}}$ 编码两支、再经预测器 $h_{\bm{\psi}}$ 投影(公式3)。
- 用对称的[[负余弦相似度|negative cosine similarity]]损失 + [[Stop-Gradient|停止梯度]]防坍缩(公式4)。
- 工程上还对本体状态附加**零掩码**以与特权状态对齐维度;特权信息取根部线速度(运动模仿任务额外含根部朝向)。

#### 模块2: SRL4Humanoid 框架与区间更新机制

**设计动机**: 为公平比较与可复现研究,提供解耦的、可把任意 SRL 目标挂到策略或价值编码器上的统一框架;并解决"持续施加 SRL 反而拖慢学习"的现象。

**具体实现**:
- 联合目标默认同步更新(公式5),SRL 与 RL 共享数据批、跟随 RL 更新频率。
- 但大规模并行 RL 早期会产生大量重复、低质数据,易让 SRL **过早陷入局部最优**;故引入**区间更新**:每隔 $T$ 步才施加一次 SRL 损失(公式6),由指示函数 $\mathbbm{1}(T)$ 控制。实验中 $T{=}50$ 通常最优。
- 框架内置 SimSiam / SPR / VAE 三种代表范式(对比/动力学建模/重建),其损失见公式 10–12。

### 关键公式与机制

#### 公式1: [[强化学习|RL]] 目标

$$
J_{\pi}(\bm{\theta})=\mathbb{E}_{\pi}\Big[\sum_{t=0}^{\infty}\gamma^{t}R(\bm{s}_{t},\bm{a}_{t},\bm{s}_{t+1})\Big]
$$

**含义**: 学习最优策略以最大化期望折扣累计奖励。

**符号说明**:
- $\gamma\in[0,1]$: 折扣因子;$R(\cdot)$: 奖励函数
- $\bm{o}_t\in\mathbb{R}^{n}$: 本体感知状态(关节位置 $\bm{q}_t$、关节速度 $\dot{\bm{q}}_t$、基座角速度 $\bm{\omega}_t$、重力方向 $\bm{g}_t$)
- $\bm{s}_t\in\mathbb{R}^{m}$: 特权状态(仅训练可用:根位姿/速度、各连杆位姿、接触指示、地形特征等)
- $\bm{a}_t\in\mathbb{R}^{k}$: 动作

#### 公式2: 零掩码构造伪增强

$$
\tilde{\bm{s}}_{t}=\text{ZeroMasking}(\bm{s}_{t})
$$

**含义**: 把特权状态中的**特权信息部分**置零,仅保留其本体观测,作为对比对的另一支。

**符号说明**:
- $\tilde{\bm{s}}_t$: 掩码后的特权状态;与原始 $\bm{s}_t$ 构成正对 $(\bm{s},\tilde{\bm{s}})$

#### 公式3: 孪生编码与投影

$$
\bm{z}=f_{\bm{\theta}}(\bm{s}),\quad \tilde{\bm{z}}=f_{\bm{\theta}}(\tilde{\bm{s}})
$$

$$
\bm{p}=h_{\bm{\psi}}(\bm{z}),\quad \tilde{\bm{p}}=h_{\bm{\psi}}(\tilde{\bm{z}})
$$

**含义**: 用共享策略编码器 $f_{\bm{\theta}}$ 编码两支得潜表征 $\bm{z},\tilde{\bm{z}}$,再经预测器 $h_{\bm{\psi}}$ 映射到 $\bm{p},\tilde{\bm{p}}$。

**符号说明**:
- $f_{\bm{\theta}}$: 策略编码器;$h_{\bm{\psi}}$: 预测器(SimSiam 风格)

#### 公式4: PvP 损失

$$
L_{\rm PvP}=D_{\rm ncs}\!\left(\bm{p},\text{sg}(\tilde{\bm{z}})\right)+D_{\rm ncs}\!\left(\tilde{\bm{p}},\text{sg}(\bm{z})\right)
$$

其中负余弦相似度为:

$$
D_{\rm ncs}(\bm{p},\bm{z})=-\frac{\bm{p}}{\|\bm{p}\|_{2}}\cdot\frac{\bm{z}}{\|\bm{z}\|_{2}}
$$

**含义**: 对称地最大化两支表征的余弦相似度;通过停止梯度 $\text{sg}(\cdot)$ 阻断一支梯度回传,防止表征坍缩。

**符号说明**:
- $\text{sg}(\cdot)$: 停止梯度算子(只前传不回传)
- $D_{\rm ncs}$: 负余弦相似度,值域 $[-1,0]$,越小越相似

#### 公式5: RL+SRL 联合目标(同步更新)

$$
L_{\rm Total}=L_{\rm RL}+\lambda\cdot L_{\rm SRL}
$$

**含义**: 主任务 RL 损失加上加权的 SRL 辅助损失。

**符号说明**:
- $\lambda$: SRL 权重系数(消融见 Figure 16,$\lambda{=}0.1$ 在运动模仿任务上最优)

#### 公式6: 区间更新机制

$$
L_{\rm Total}=L_{\rm RL}+\mathbbm{1}(T)\cdot\lambda\cdot L_{\rm SRL}
$$

**含义**: 仅每隔 $T$ 步施加一次 SRL 损失,避免早期低质数据使 SRL 过早陷入局部最优。

**符号说明**:
- $\mathbbm{1}(T)$: 指示函数,每 $T$ 步取 $1$,否则取 $0$

#### 公式7–9: [[PPO]] 目标(附录)

$$
L_{\pi}(\bm{\theta})=-\mathbb{E}_{\tau\sim\pi}\!\left[\min\!\left(\rho_{t}(\bm{\theta})A_{t},\ {\rm clip}\!\left(\rho_{t}(\bm{\theta}),1-\epsilon,1+\epsilon\right)A_{t}\right)\right]
$$

$$
\rho_{t}(\bm{\theta})=\frac{\pi_{\bm{\theta}}(\bm{a}_{t}|\bm{s}_{t})}{\pi_{\bm{\theta}_{\rm old}}(\bm{a}_{t}|\bm{s}_{t})}
$$

$$
L_{V}(\bm{\phi})=\mathbb{E}_{\tau\sim\pi}\!\left[\left(V_{\bm{\phi}}(\bm{s})-V_{t}^{\rm target}\right)^{2}\right]
$$

**含义**: PPO 的裁剪代理目标(防止策略更新过大)与价值回归损失;价值目标用 [[GAE]] 计算。

**符号说明**:
- $\rho_t$: 新旧策略概率比;$A_t$: 优势;$\epsilon$: 裁剪范围

#### 公式10: [[VAE]] 损失(重建式基线)

$$
L_{\rm VAE}=-\mathbb{E}_{q_{\bm{\phi}}(\bm{z}|\bm{o})}\!\left[\log p_{\bm{\theta}}(\bm{o}|\bm{z})\right]+D_{\rm KL}\!\left(q_{\bm{\phi}}(\bm{z}|\bm{o})\|p_{\bm{\theta}}(\bm{z})\right)
$$

**含义**: 重建项 + KL 正则,平衡重建保真与先验约束。

**符号说明**: $q_{\bm{\phi}}$ 编码器,$p_{\bm{\theta}}$ 解码器,$D_{\rm KL}$ KL 散度。

#### 公式11: [[SPR]] 损失(动力学建模基线)

$$
L_{\rm SPR}=\sum_{k=1}^{K}\left\|f_{\bm{\theta}}^{(k)}\!\big(\bm{z}_{t},\bm{a}_{t:t+k-1}\big)-\text{sg}\!\left(g_{\bm{\phi}}(\bm{o}_{t+k})\right)\right\|_{2}^{2}
$$

**含义**: 用在线动力学模型 $f_{\bm{\theta}}$($\bm{z}_{t+1}=f_{\bm{\theta}}(\bm{z}_t,\bm{a}_t)$)做多步预测,与目标模型 $g_{\bm{\phi}}$(参数为在线模型的 [[EMA]])编码的未来状态对齐。

**符号说明**: $K$ 预测步数,$\text{sg}$ 停止梯度。

#### 公式12: [[SimSiam]] 损失(对比基线)

$$
L_{\rm SimSiam}=\frac{1}{2}\left[-\frac{f_{\bm{\theta}}(\bm{x_{1}})\cdot f_{\bm{\theta}}(\bm{x_{2}})}{\|f_{\bm{\theta}}(\bm{x_{1}})\|_{2}\|f_{\bm{\theta}}(\bm{x_{2}})\|_{2}}\right]
$$

**含义**: 同一输入两个增强视图的负余弦相似度,无需负样本/大批量/动量编码器。PvP 即基于此范式,但把"两个增强视图"替换为"本体状态与特权状态"这一天然互补对。

---

## 关键图表

<!-- 图片均使用 arXiv HTML 在线链接,未本地化 -->

### Figure 1: Overview of PvP / PvP 方法总览

![Figure 1](https://arxiv.org/html/2512.13093v2/x1.png)

**说明**: (a) 特权状态与本体感知状态的组成;特权状态同时包含本体观测与特权信息,故可视为本体状态的"伪增强"。(b) PvP 基于两种状态模态的内在互补性做对比学习,这是全文最核心的思想图。

### Figure 2: SRL4Humanoid Architecture / 框架架构

![Figure 2](https://arxiv.org/html/2512.13093v2/x2.png)

**说明**: SRL4Humanoid 框架架构,SRL 与 RL 过程**完全解耦**。策略网络吃本体状态出动作,价值网络吃特权状态做估值,SRL 损失可挂到策略或价值编码器,体现框架的模块化与可插拔。

### Figure 3: LimX Oli Robot & Tasks / 机器人平台与任务

![Figure 3](https://arxiv.org/html/2512.13093v2/x3.png)

**说明**: 实验平台 LimX Oli 全尺寸人形机器人(31 DoF)的规格,以及两个设计任务(LimX-Oli-31dof-Velocity 速度跟踪、LimX-Oli-31dof-Mimic 运动模仿)的截图。

### Figure 4: Overall Reward (Training Progress) / 总奖励学习曲线

![Figure 4](https://arxiv.org/html/2512.13093v2/x4.png)

**说明**: vanilla PPO 与四种 SRL 方法在两任务上的总奖励训练曲线(实线均值、阴影标准差)。速度跟踪任务上 PvP 显著加速学习,其余 SRL 改进甚微;运动模仿任务上 PvP 最高,而 VAE 出现性能退化——说明"单纯重建感知数据不足以提升效率"。这是回答 Q1 的主证据之一。

### Figure 5: Action Smoothness Optimization / 动作平滑性优化对比

![Figure 5](https://arxiv.org/html/2512.13093v2/x5.png)

**说明**: vanilla PPO 与四种 SRL 在动作平滑性奖励项上的优化对比。动作平滑性直接关系真机能否平稳运行(防止剧烈抖动),PvP 在该项上优化更好,佐证其真机可部署性。

### Figure 6: Tracking Performance / 跟踪关键指标

![Figure 6](https://arxiv.org/html/2512.13093v2/x6.png)

**说明**: PPO 与四种 SRL 在三个关键跟踪指标上的对比,PvP 全部取得最高性能,直接支撑"PvP 优于基线"(Q1)。

### Figure 7: Impact of Training Time Proportion / 训练时间占比影响

![Figure 7a](https://arxiv.org/html/2512.13093v2/x7.png)
![Figure 7b](https://arxiv.org/html/2512.13093v2/x8.png)

**说明**: 四种 SRL 在不同**更新间隔**(1/50/100)下的训练曲线(左右为两任务)。速度跟踪任务对间隔不敏感,运动模仿任务影响明显;**间隔 50** 普遍最优——验证区间更新机制(公式6)能防早熟收敛、降算力。回答 Q2。

### Figure 8: Impact of Training Data Proportion / 训练数据占比影响

![Figure 8a](https://arxiv.org/html/2512.13093v2/x9.png)
![Figure 8b](https://arxiv.org/html/2512.13093v2/x10.png)

**说明**: 用 10%/50%/100% 的 rollout 数据(随机掩码重采样)训练各 SRL。速度跟踪曲线几乎一致;增大比例普遍提升性能,SimSiam 与 PvP 尤为明显——合理分配数据比例可加速学习(尤其运动模仿)。回答 Q3。

### Figure 9: SRL on Value Encoder / 施加于价值编码器

![Figure 9a](https://arxiv.org/html/2512.13093v2/x11.png)
![Figure 9b](https://arxiv.org/html/2512.13093v2/x12.png)

**说明**: 把 SRL 损失加到**价值编码器**的学习曲线。相比加到策略编码器,收敛更慢;速度跟踪任务甚至出现训练坍缩(动作平滑性骤降后恢复)。结论:**SRL 应施加于策略编码器**更稳更好。回答 Q4。

### Figure 10: Sim2Sim Evaluation on MuJoCo / MuJoCo 上的 Sim2Sim 评估

![Figure 10a](https://arxiv.org/html/2512.13093v2/figures/pvp_mimic_003_film.png)
![Figure 10b](https://arxiv.org/html/2512.13093v2/figures/pvp_mimic_018_film.png)
![Figure 10c](https://arxiv.org/html/2512.13093v2/figures/pvp_walk_left_to_right_film.png)
![Figure 10d](https://arxiv.org/html/2512.13093v2/figures/pvp_rotation_film.png)

**说明**: 在更接近真实条件的 [[MuJoCo]] 上做 Sim2Sim 评估。前两行展示运动模仿能力,后两行展示速度跟踪能力。该图配合真机测试(正文 Figure LABEL:fig:big_preface)回答 Q6:学到的策略可迁移到更真实的物理环境执行复杂任务。

### Figure 11: Motion Capture Data / 动捕数据示例(附录)

![Figure 11a](https://arxiv.org/html/2512.13093v2/figures/mocap_001.png)
![Figure 11b](https://arxiv.org/html/2512.13093v2/figures/mocap_013.png)
![Figure 11c](https://arxiv.org/html/2512.13093v2/figures/mocap_019.png)

**说明**: 用于运动模仿任务的动捕数据示例截图(每段最长 43 秒、4300 帧),说明 Mimic 任务的参考运动来源。

### Figure 12–14: PvP vs Teacher-Student Distillation / 与师生蒸馏对比(附录)

![Figure 12 Velocity](https://arxiv.org/html/2512.13093v2/x13.png)
![Figure 13 Mimic](https://arxiv.org/html/2512.13093v2/x14.png)
![Figure 14 Velocity-Rough](https://arxiv.org/html/2512.13093v2/x15.png)

**说明**: 在 Velocity、Mimic、Velocity-Rough 三任务上 TSD 与 PvP 的训练曲线。实验显示学生难以匹配教师、蒸馏后存在显著差距;而 PvP 让表征学习与策略学习协同,无需昂贵的教师预训练阶段,更利复现。

### Figure 15: SRL on Unitree G1 / 在 Unitree G1 上验证(附录)

![Figure 15](https://arxiv.org/html/2512.13093v2/x16.png)

**说明**: 在不同机器人平台 Unitree-G1-29dof-Velocity 任务上 PPO 与四种 SRL 的对比,PvP 在最终性能与时间成本上同样领先,说明方法的跨本体可迁移性。

### Figure 16: Impact of λ on PvP / 权重系数消融(附录)

![Figure 16](https://arxiv.org/html/2512.13093v2/x17.png)

**说明**: PvP 在不同权重系数下的对比(LimX-Oli-31dof-Mimic)。$\lambda{=}0.1$ 取得最佳最终性能与样本效率,说明 PvP 偏好**较小的** $\lambda$。

### Table 1: Reward Terms — Velocity Task / 速度跟踪任务奖励项

| 奖励项 | 公式 | 权重 |
|--------|------|------|
| Linear velocity tracking | $\exp\!\left(-\tfrac{\|\bm{v}_{xy}-\bm{v}^{\rm cmd}_{xy}\|^{2}}{2\sigma^{2}}\right)$ | 1.0 |
| Angular velocity tracking | $\exp\!\left(-\tfrac{(\omega_{z}-\omega_{z}^{\rm cmd})^{2}}{2\sigma^{2}}\right)$ | 0.5 |
| Base height | $(h-h^{\star})^{2}$ | 0.5 |
| Linear velocity $(z)$ | $\|v_{z}\|^{2}$ | -2e-3 |
| Angular velocity $(x,y)$ | $\|\bm{\omega}_{xy}\|^{2}$ | -0.15 |
| Action smoothness | $\|\bm{a}_{t}-2\bm{a}_{t-1}-\bm{a}_{t-2}\|^{2}$ | -2.5e-3 |
| Joint velocity | $\|\dot{\bm{q}}\|^{2}$ | -1e-3 |
| Joint acceleration | $\|\ddot{\bm{q}}\|^{2}$ | -5e-7 |
| Joint deviation | $\sum_{j}|q_{j}-q_{j}^{\text{def}}|$ | -0.1 |
| Joint power | $|\bm{\tau}||\dot{\bm{q}}|^{T}$ | -2.5e-7 |
| Joint torque | $\|\bm{\tau}\|_{2}^{2}$ | -4.0e-7 |
| Joint position limits | $\sum_{j}\Delta_{j}$ | -0.2 |
| Joint velocity limits | $\sum_{j}\dot{q}_{j}$ | -0.025 |

**说明**: 速度跟踪任务的复合奖励。正向项为线/角速度跟踪与基座高度,大量负向项(平滑、能耗、力矩、关节限位)用于保证真机可部署的稳健性,体现 WBC 奖励的"复合且难调"特征。

### Table 2: State Composition — Velocity Task / 速度任务状态构成

**说明**: 速度跟踪任务的本体感知状态与特权状态明细。HTML 中该表正文被截断,但结构与运动模仿任务(Table 4)一致:策略只见本体状态(角速度、重力投影、关节位姿/速度、动作等,并堆叠 **5 个连续本体状态**作为策略编码器输入以增强鲁棒性);价值网络/特权状态额外含 base_lin_vel、base_pos_z、body_mass、base_quat、velocity_commands 等仅训练可见信息。

### Table 3: Reward Terms — Mimic Task / 运动模仿任务奖励项

| 奖励项 | 公式 | 权重 |
|--------|------|------|
| Position tracking | $\exp\!\left(-\tfrac{\|\bm{q}-\bm{q}^{\rm ref}\|^{2}}{2\sigma^{2}}\right)$ | 2.0 |
| Feet distance tracking | $\exp\!\left(-\tfrac{|d-d^{\rm ref}|^{2}}{\sigma}\right)$ | 0.5 |
| Waist pitch orientation tracking | $\exp\!\left(-\sum_{i=1}^{2}|\delta_{i}-\delta_{i}^{\rm ref}|\right)$ | 0.5 |
| Action rate | $\|\bm{a}_{t}-\bm{a}_{t-1}\|^{2}$ | -0.001 |
| Joint velocity | $\|\dot{\bm{q}}\|^{2}$ | -0.5e-3 |
| Joint acceleration | $\|\ddot{\bm{q}}\|^{2}$ | -1.0e-7 |
| Joint torque | $\|\bm{\tau}\|_{2}^{2}$ | -1.0e-5 |
| Joint position limits | $\sum_{j}\Delta_{j}$ | -1.0 |
| Joint torque limits | $\sum_{j}\tau_{j}$ | -0.01 |
| Joint velocity limits | $\sum_{j}\dot{q}_{j}$ | -0.2 |

**说明**: 运动模仿任务奖励,以**位置跟踪**(权重 2.0)为核心,辅以足距、腰部俯仰朝向跟踪,以及一系列平滑/限位惩罚。

### Table 4: State Composition — Mimic Task / 运动模仿任务状态构成

| Proprioceptive State(策略) | Privileged State(价值) |
|------------------------------|--------------------------|
| base_ang_vel (3) | base_lin_vel (3) |
| projected_gravity (3) | base_ang_vel (3) |
| joint_pos (31) | base_pos_z (1) |
| joint_vel (31) | body_mass (40) |
| actions (31) | base_quat (6) |
| mimic reference (69) | projected_gravity (3) |
| | velocity_commands (3) |
| | joint_pos (31) / joint_vel (31) |
| | actions (31) / previous actions (31) |
| | mimic reference (69) |

**说明**: 本体状态仅含可在真机测量的量(+mimic 参考),特权状态额外含 base_lin_vel、base_pos_z、body_mass、base_quat、velocity_commands、previous actions 等仅训练可见信息——正是 PvP 用作"伪增强"互补对的来源。

### Table 5: Network Architectures / 策略与价值网络结构

| 部分 | Policy Network | Value Network |
|------|----------------|----------------|
| Encoder | Linear(O.D.,512)→ELU→Linear(512,256)→ELU→Linear(256,128) | Linear(O.D.,512)→ELU→Linear(512,256)→ELU→Linear(256,128) |
| Head | Linear(128,128)→ELU→Linear(128,31) | Linear(256,128)→ELU→Linear(128,1) |

**说明**: 全部实验固定使用该结构(O.D.= On-demand,输入维度按任务确定),以**隔离 SRL 方法本身的效果**。策略输出 31 维动作,价值输出标量。

### Table 6: PPO Hyperparameters / PPO 超参

| 超参 | 值 | 超参 | 值 |
|------|-----|------|-----|
| Reward normalization | Yes | Optimizer | Adam |
| LSTM | No | Learning rate | 1e-3 |
| Maximum Episodes | 30000 | LR scheduler | Adaptive |
| Episode steps | 32 | GAE coefficient | 0.95 |
| Number of workers | 1 | Action entropy coef. | 0.01 |
| Environments per worker | 4096 | Value loss coef. | 1.0 |
| Value clip range | 0.2 | Max gradient norm | 0.5 |
| Number of mini-batches | 4 | Number of learning epochs | 5 |
| Desired KL divergence | 0.01 | Discount factor | 0.99 |

**说明**: 两任务共用、固定不变的 PPO 配置(4096 并行环境的大规模并行 RL),以保证对比公平。

---

## 实验

### 数据集 / 任务

| 任务 | 平台 | 特点 | 用途 |
|------|------|------|------|
| LimX-Oli-31dof-Velocity | LimX Oli 31-DoF | 平地速度跟踪,命令每 10s 重采样($v_x\in(-0.5,1.0)$ m/s 等) | 训练/测试 |
| LimX-Oli-31dof-Mimic | LimX Oli 31-DoF | 运动模仿,参考来自动捕(每段≤43s/4300 帧) | 训练/测试 |
| LimX-Oli-31dof-Velocity-Rough | LimX Oli 31-DoF | 速度跟踪 + 粗糙地形(附录扩展) | 训练/测试 |
| Unitree-G1-29dof-Velocity | Unitree G1 29-DoF | 跨本体平台速度跟踪(附录扩展) | 训练/测试 |

### 实现细节

- **RL 骨干**: [[PPO]];策略吃本体状态(堆叠 5 帧)、价值吃特权状态;网络见 Table 5。
- **SRL 基线**: [[SimSiam]](系数 0.5,random_masking+identity_mapping)、[[SPR]](系数 0.5,gaussian_noise,预测步 5)、[[VAE]](系数 0.1);SRL 默认挂策略编码器。
- **PvP 配置**: 特权信息取根部线速度(Mimic 任务另含根部朝向),本体状态附零掩码对齐维度;损失系数搜索 $\{0.1,0.5,1.0\}$,基线设 0.5(附录消融显示 Mimic 上 0.1 更优)。
- **优化**: Adam,lr 1e-3,自适应调度,30000 episodes,4096 并行环境/worker。
- **硬件/仿真**: 单卡 RTX4090(24GB)+ [[IsaacLab]];SRL 模块全程在 GPU 运行,几乎不增训练开销;Sim2Sim 在 [[MuJoCo]] 评估,最后真机 LimX Oli 测试。

### 关键实验结论

- **Q1(性能,Fig 4/5/6)**: PvP 在总奖励、动作平滑性、三个跟踪指标上**全面领先**;VAE 在 Mimic 任务退化,印证"纯重建不利学习效率"。
- **Q2(训练时间占比,Fig 7)**: 更新间隔对速度跟踪不敏感,对运动模仿影响明显,**间隔 50 普遍最优**。
- **Q3(训练数据占比,Fig 8)**: 增大数据比例普遍提升性能(SimSiam/PvP 尤甚),速度跟踪近乎不敏感。
- **Q4(编码器选择,Fig 9)**: SRL 应施加于**策略编码器**;施加于价值编码器收敛更慢甚至坍缩。
- **Q5(算力,§5.2.5)**: SRL 模块全 GPU 运行,几乎不影响整体训练效率。
- **Q6(真机,Fig 10 + 正文)**: Sim2Sim(MuJoCo)与真机 LimX Oli 上均能执行复杂任务。
- **附录消融**: 优于师生蒸馏(Fig 12–14);跨本体 Unitree G1 仍领先(Fig 15);$\lambda{=}0.1$ 最优(Fig 16)。

---

## 批判性思考

### 优点
1. **思想简洁而巧妙**: 把"特权状态天然含本体观测"这一事实直接用作 SimSiam 的伪增强对,**免去手工数据增强**,几乎零额外算力即可挂到现有 PPO 上,工程友好。
2. **系统性与可复现性强**: 开源 SRL4Humanoid 把 SRL 与 RL 解耦,统一对比 SimSiam/SPR/VAE,并系统回答了更新间隔、数据比例、编码器位置等实践问题,对社区有方法论价值。
3. **结论有横向支撑**: 不仅给主任务曲线,还覆盖动作平滑性、跨地形(Velocity-Rough)、跨本体(Unitree G1)、Sim2Sim 与真机,以及与师生蒸馏的对比,论证较扎实。

### 局限性
1. **缺少绝对量化指标**: 论文以"训练曲线 + 均值/方差阴影"为主,未给最终成功率/跟踪误差等**数值表格**,改进幅度难以精确量化,横向比较略偏定性。
2. **任务与本体仍有限**: 集中在速度跟踪与运动模仿两类任务,真机测试以演示为主;更复杂的接触丰富操作、长程任务、感知(RGB/深度)输入未覆盖(作者亦将其列为未来工作)。
3. **特权信息的选择偏经验**: PvP 把哪些维度算作"特权"(如根线速度/朝向)依赖人工指定,缺少对特权信息选择的敏感性分析;零掩码作为唯一构造方式也未与其他互补构造做对比。

### 潜在改进方向
1. 补充成功率、跟踪 RMSE、能耗等**定量指标**与显著性检验,量化"互补对比→样本效率"的收益。
2. 把"哪些维度作为特权信息""更新间隔 $T$""$\lambda$"做成自适应/可学习,减少手工调参。
3. 扩展到多模态感知(RGB/深度)与更难的全身操作/长程任务,验证 PvP 范式在高维感知下的可扩展性。

### 可复现性评估
- [x] 代码开源(声明开源 SRL4Humanoid 框架)
- [ ] 预训练模型(未明确提供)
- [x] 训练细节完整(附录给出网络结构 Table 5、PPO 超参 Table 6、各 SRL 搜索范围)
- [x] 数据集/平台可获取(LimX Oli、Unitree G1、IsaacLab、MuJoCo 公开;动捕参考用于 Mimic)

---

## 速查卡片

> [!summary] PvP: Proprioceptive-Privileged Contrastive Representations
> - **核心**: 把"特权状态"当作"本体状态"的伪增强,用 SimSiam 式对比学习注入特权信息,免手工增强地提升人形 WBC 的 RL 样本效率。
> - **方法**: 零掩码构造正对 $(\bm{s},\tilde{\bm{s}})$ → 共享策略编码器 + 预测器 → 对称负余弦相似度 + 停止梯度(公式4);区间更新机制 $\mathbbm{1}(T)$ 防早熟;SRL 挂策略编码器最稳。
> - **结果**: LimX Oli 31-DoF 的速度跟踪/运动模仿任务上,总奖励、平滑性、跟踪指标全面优于 SimSiam/SPR/VAE 与师生蒸馏;跨地形/跨本体(G1)/Sim2Sim(MuJoCo)/真机均验证;$\lambda{=}0.1$、间隔 50 较优。
> - **框架**: SRL4Humanoid——首个统一模块化、SRL/RL 解耦的人形 SRL 开源框架。

---

*笔记创建时间: 2026-06-29*
