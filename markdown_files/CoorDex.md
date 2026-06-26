---
title: "CoorDex: Coordinating Body and Hand Priors for Continuous Dexterous Humanoid Loco-Manipulation"
method_name: "CoorDex"
authors: [Sikai Li]
year: 2026
venue: arXiv
tags: [humanoid, loco-manipulation, dexterous-manipulation, whole-body-control, reinforcement-learning]
zotero_collection: _inbox
image_source: online
created: 2026-06-25
---

# CoorDex

## 一句话结论

[[CoorDex]] 的核心贡献是把 [[Whole-Body Control|全身控制]] 和 [[Dexterous Manipulation|灵巧手控制]] 分别压进可复用的 latent prior，再用 coordinated residual policy 在两个 prior 上做下游 [[Loco-Manipulation|移动操作]]。它不是简单地把手和身体动作拼接起来，而是让身体 residual、手部 residual 共享任务上下文，再分别输出低维 latent 修正。

## 论文定位

- 任务：连续的 dexterous humanoid loco-manipulation。
- 反对的简化设定：先走到物体前、停下、操作、再继续走。
- 反对的硬件简化：低 DoF end-effector 的 open-close grasp primitive。
- 目标能力：边走边完成高自由度手部操作，例如 WalkGrab、WalkPickTurn、OpenFridge。
- 关键路线：tracking teacher -> proprioception-conditioned latent prior -> frozen prior -> residual RL。
- 主要系统依赖：[[PPO]]、[[Motion Prior]]、[[Residual RL]]、[[MANO]]、[[ManipTrans]]、[[DeepMimic]]、[[AMP]]、[[Isaac Lab]]。

## 背景问题

Humanoid loco-manipulation 难在三件事同时发生。

1. 身体要保持动态稳定。
2. 手要做细粒度接触。
3. 身体和手不能互相当外部扰动。

传统系统常把 locomotion 和 manipulation 拆开，结果是任务看起来像移动底盘带机械臂，不像 humanoid。CoorDex 的判断是：直接在全关节 action space 里做 RL 太难，但完全解耦又失去协调，所以应该先学 body prior 和 hand prior，再在 latent space 里做协调。

## 方法总览

输入：

- body proprioception $\mathbf{s}^{b,p}_{t}$
- hand proprioception $\mathbf{s}^{h,p}_{t}$
- task state $\mathbf{s}^{task}_{t}$
- hand-object state $\mathbf{s}^{hand-object}_{t}$
- frozen body/hand prior mean

输出：

- body latent residual $\Delta\mathbf{z}^{b}_{t}$
- hand latent residual $\Delta\mathbf{z}^{h}_{t}$
- final body joint targets $\mathbf{a}^{b}_{t}$
- final hand joint targets $\mathbf{a}^{h}_{t}$

执行流程：

1. 训练 body tracking teacher $\pi_T^b$。
2. 训练 hand tracking teacher $\pi_T^h$。
3. 把两个 teacher 蒸馏成 latent prior。
4. 冻结 prior 和 decoder。
5. 下游 [[PPO]] policy 在 body/hand latent prior 上预测 residual。
6. frozen decoder 把 latent command 解码成 joint position target。
7. 低层 [[PD Controller]] 执行动作。

## 关键公式

### 1. Prior distillation objective

每个子系统 $x\in\{b,h\}$ 都有 encoder、proprioceptive prior 和 decoder。训练目标组合 teacher-action reconstruction、temporal smoothness 和 KL regularization。

$$
\mathcal{L}^{x}_{\mathrm{distill}}
=\mathcal{L}^{x}_{\mathrm{action}}
+\alpha_{x}\mathcal{L}^{x}_{\mathrm{regu}}
+\beta_{x}\mathcal{L}^{x}_{\mathrm{KL}}.
$$

符号：

- $x=b$ 表示 body subsystem。
- $x=h$ 表示 hand subsystem。
- $\mathcal{L}^{x}_{\mathrm{action}}$ 重建 teacher action。
- $\mathcal{L}^{x}_{\mathrm{regu}}$ 约束 latent temporal smoothness。
- $\mathcal{L}^{x}_{\mathrm{KL}}$ 让 encoder posterior 贴近 proprioceptive prior。

### 2. Frozen prior mean

下游策略先从冻结 prior 里取默认 latent command。

$$
\boldsymbol{\mu}^{b,p}_{t}
=\mathrm{Mean}\left[
\mathcal{R}_{b}\left(\mathbf{z}^{b}_{t}\mid\mathbf{s}^{b,p}_{t}\right)
\right],
\qquad
\boldsymbol{\mu}^{h,p}_{t}
=\mathrm{Mean}\left[
\mathcal{R}_{h}\left(\mathbf{z}^{h}_{t}\mid\mathbf{s}^{h,p}_{t}\right)
\right].
$$

含义：

- body prior 给出默认身体 latent。
- hand prior 给出默认手部 latent。
- residual policy 只需要在默认 latent 附近修正，探索难度明显低于 joint-space RL。

### 3. Latent residual action

策略不是直接输出关节目标，而是输出两个 latent residual。

$$
\Delta\mathbf{z}_{t}
=\left[
\Delta\mathbf{z}^{b}_{t},
\Delta\mathbf{z}^{h}_{t}
\right],
\qquad
\Delta\mathbf{z}^{b}_{t}\in\mathbb{R}^{d_b},
\quad
\Delta\mathbf{z}^{h}_{t}\in\mathbb{R}^{d_h}.
$$

### 4. Coordination trunk and residual heads

共享协调 trunk 接收身体、手、任务和接触状态，再由 body head 与 hand head 分别输出 residual。

$$
\begin{aligned}
\mathbf{c}_{t}
&=
f_{\mathrm{coord}}\left(
\mathbf{s}^{b,p}_{t},
\mathbf{s}^{h,p}_{t},
\mathbf{s}^{\mathrm{task}}_{t},
\mathbf{s}^{\mathrm{hand-object}}_{t},
\boldsymbol{\mu}^{b,p}_{t},
\boldsymbol{\mu}^{h,p}_{t},
\Delta\mathbf{z}_{t-1}
\right),\\
\Delta\mathbf{z}^{b}_{t}
&=
\tanh\left(
f_b(\mathbf{c}_{t},\mathbf{s}^{b,p}_{t},\boldsymbol{\mu}^{b,p}_{t})
\right),\\
\Delta\mathbf{z}^{h}_{t}
&=
\tanh\left(
f_h(\mathbf{c}_{t},\mathbf{s}^{h,p}_{t},\boldsymbol{\mu}^{h,p}_{t},\mathbf{s}^{\mathrm{hand-object}}_{t})
\right).
\end{aligned}
$$

这个设计的关键是“共享任务相位，分头执行修正”。身体可以改步态、躯干、reach 和 wrist placement；手可以改 preshape、closure 和 contact refinement。

### 5. Final latent command and decoded action

$$
\tilde{\mathbf{z}}^{b}_{t}
=
\boldsymbol{\mu}^{b,p}_{t}
+\Delta\mathbf{z}^{b}_{t},
\qquad
\tilde{\mathbf{z}}^{h}_{t}
=
\boldsymbol{\mu}^{h,p}_{t}
+\Delta\mathbf{z}^{h}_{t}.
$$

$$
\mathbf{a}^{b}_{t}
=
D_b\left(\mathbf{s}^{b,p}_{t},\tilde{\mathbf{z}}^{b}_{t}\right),
\qquad
\mathbf{a}^{h}_{t}
=
D_h\left(\mathbf{s}^{h,p}_{t},\tilde{\mathbf{z}}^{h}_{t}\right).
$$

执行上，body decoder 控制 humanoid body joints，hand decoder 控制 active finger joints，最后都进入 low-level PD。

## 关键图表

### Figure 1: Teaser

![Figure 1](https://arxiv.org/html/2606.23680v1/figures/CoDex_teaser_v1.png)

- 展示 CoorDex 面向连续 dexterous humanoid loco-manipulation，而不是 stop-and-go。
- 重点看身体移动和高 DoF 手部接触是否同时发生。

### Figure 2: Method overview

![Figure 2](https://arxiv.org/html/2606.23680v1/x1.png)

- 对应 prior construction、VAE distillation、downstream residual composition。
- body prior 和 hand prior 是分开的，协调发生在 shared residual policy。

### Figure 3: Task performance visualization

![Figure 3](https://arxiv.org/html/2606.23680v1/x2.png)

- 展示主要任务 rollouts。
- 用来判断策略是否真的连续操作，而不是在关键阶段停下来。

### Figure 4: Non-stop locomotion on WalkGrab

![Figure 4](https://arxiv.org/html/2606.23680v1/figures/velocity_vs_rel_x.png)

- 重点是 WalkGrab 过程中 velocity 与相对距离变化。
- 这张图直接服务于论文 claim：移动和抓取不应该被强行切开。

### Figure 5: Real-world demo 1

![Figure 5](https://arxiv.org/html/2606.23680v1/x3.png)

### Figure 6: Real-world demo 2

![Figure 6](https://arxiv.org/html/2606.23680v1/x4.png)

### Figure 7: Real-world demo 3

![Figure 7](https://arxiv.org/html/2606.23680v1/x5.png)

真实演示很重要，因为这类方法如果只在 simulation 里成立，价值会大打折扣。CoorDex 的附录明确包含 Real-World Demos。

### Tables

- Table 1: actuated DoF and latent dimensions。
- Table 2: CoorDex task performance，使用 50,000 episodes 和 10,000 parallel simulation environments。
- Table 3: WalkGrab action-space variants，比较 reach、grasp、stop、fall。
- Table 4: WalkGrab coordination analysis。
- Table 5: downstream tasks 的 observation terms。
- Table 6-8: WalkGrab、OpenFridge、WalkPickTurn reward components。
- Table 9: shared training hyperparameters。
- Table 10: task-dependent settings。
- Table 11: NoDemoRSI configuration。

## 实验结果

### 任务设置

- WalkGrab：移动中抓取。
- WalkPickTurn：走、抓、转向等多阶段组合。
- OpenFridge：需要手部接触、身体姿态和移动协调。

### 主要结果解读

1. CoorDex 的优势不是“某个 reward 写得更细”，而是 action space 设计更适合高 DoF 身体-手协同。
2. Table 3 的 action-space variants 是核心消融：如果直接用 joint-space 或不协调 body/hand residual，任务成功率和 fall rate 会暴露问题。
3. Figure 4 的 non-stop locomotion 证明它没有把任务偷换成“走到目标旁边再操作”。
4. 附录 real-world demos 说明它至少考虑了 sim-to-real，而不是只在 Isaac Lab 里报表。

## 方法优点

- 把 body 和 hand prior 分开训练，降低单个 prior 的学习负担。
- 在 latent residual 上做下游 RL，比直接探索高维 joint action 更现实。
- shared coordination trunk 保留身体和手的耦合信息。
- 任务设计抓住了 humanoid loco-manipulation 的痛点：移动时操作，而不是停下操作。
- 包含真实世界演示，避免纯 simulation claim。

## 局限与风险

- prior 覆盖不到的操作模式，residual policy 可能很难修。
- hand prior 的 wrist-stabilized 设计简化了手腕自由度，对复杂 wrist-object coupling 可能不够。
- 真实实验的覆盖范围需要继续看完整视频和失败案例，摘要级信息还不能证明广泛 sim-to-real。
- reward components 很多，系统调参成本不低。
- 目前主要展示 humanoid + dexterous hand 的特定任务族，泛化到软物体、双手协作、强扰动接触还不能直接推出。

## 和相关工作的关系

- [[DeepMimic]] / [[AMP]]：物理运动 imitation 的基础思想。
- [[BeyondMimic]] / [[ExBody2]]：humanoid tracking 和 whole-body behaviors。
- [[DexMimicGen]] / [[ManipTrans]]：dexterous hand motion transfer 和 residual learning。
- [[PPO]]：下游 residual RL 优化器。
- [[MANO]]：手部动作/形态相关表示。

## 对我的启发

- 对 humanoid policy 来说，action space 设计和 prior factorization 可能比换更大的网络更关键。
- 如果要做 whole-body + hand，先把 body/hand 可控 latent 做好，再做任务 residual，比从零端到端更稳。
- 评估时必须强制 non-stop loco-manipulation，否则策略会退化成 mobile manipulator。
- 后续可以关注：body-hand latent 如何和视觉语言 policy 接起来，尤其是 OpenHLM 这类 whole-body VLA。

## 精读优先级

高。原因是它直接打中 humanoid loco-manipulation 的高 DoF coordination 问题，而且方法不是空喊 foundation model，而是能落到控制栈和 reward 表格上的系统。

