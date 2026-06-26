---
title: "Behavior Foundation Model for Humanoid Robots"
method_name: "BFM"
authors: [Weishuai Zeng, Shunlin Lu, Kangning Yin, Xiaojie Niu, Minyue Dai, Jingbo Wang, Jiangmiao Pang]
year: 2025
venue: arXiv
tags: [humanoid, whole-body-control, behavior-foundation-model, motion-imitation, cvae]
zotero_collection: _inbox
image_source: online
arxiv_html: https://arxiv.org/html/2509.13780v1
created: 2026-06-25
---

# BFM

## 一句话结论

[[BFM]] 的核心是把 humanoid [[Whole-Body Control|全身控制]] 从“每个控制模式训练一个 task-specific policy”改写成“学习可复用的行为分布”：先用 [[Motion Imitation|运动模仿]] 训练 proxy agent，再用 [[Masked Online Distillation|masked online distillation]] + [[Conditional Variational Autoencoder|CVAE]] 蒸馏成能被多种控制模式 steer 的行为基础模型。

## 论文定位

- 任务：humanoid whole-body low-level control，覆盖 motion tracking、[[Teleoperation|VR teleoperation]]、locomotion、behavior composition/modulation 和 novel behavior acquisition。
- 平台：[[Unitree G1]]，论文中真实部署时冻结双手腕，得到 23 DoF 控制。
- 训练数据来源：[[AMASS]] human motion，经 [[SMPL]] 到 humanoid 的 [[Human Motion Retargeting|motion retargeting]]。
- 训练范式：先用 [[PPO]] 训练拥有 simulator privileged state 的 proxy agent，再用 [[DAgger]] 方式在线蒸馏 BFM。
- 关键机制：稀疏 mask 控制接口 + CVAE latent space + [[Residual RL|residual learning]]。
- 对比：[[HOVER]]、task specialist、RL from scratch、proxy agent。

## 核心贡献

1. **统一行为视角**：把不同 control modes 产生的结果都视作 humanoid behavior，把目标状态看作外部 motive，而不是 behavior 本身。
2. **可 mask 的控制接口**：root、kinematic position、joint angle 组成统一 control interface，用 bit-wise mask 激活任意控制模式。
3. **CVAE 行为分布模型**：通过 prior/encoder/decoder 学到结构化 latent space，支持 behavior composition 和 modulation。
4. **在线蒸馏训练**：不是离线拟合固定轨迹，而是在仿真中 rollout 当前 BFM，再查询 proxy agent 作为 reference action。
5. **新行为快速获取**：冻结 BFM 后只学 residual decoder，比从零 RL 学 Side Salto 更高效。

## 背景问题

现有 humanoid [[Whole-Body Control|WBC]] 系统通常按控制模式拆开：

1. locomotion policy 只接受速度命令。
2. motion tracking policy 只接受 reference motion。
3. teleoperation policy 只接受 VR/IK 信号。
4. language-conditioned WBC 又是另一套任务接口。

这些系统都能生成 humanoid behavior，但接口被固定在设计时，导致跨任务泛化差。BFM 的判断是：不同控制模式只是同一个 behavior distribution 的不同条件表达。要泛化，应该先学习行为分布，再让控制模式通过 goal state 和 mask 去 steer 这个分布。

## 方法详解

### Phase 1: Proxy agent training

作者先把 [[AMASS]] human motion 通过两阶段 [[Human Motion Retargeting|retargeting]] 映射到 humanoid：

1. 优化 [[SMPL]] shape parameter，使 selected links 的 rest-pose 距离贴近目标 humanoid。
2. 优化 humanoid root translation、orientation 和 joint positions，使整段 motion 的 selected links 贴近 human motion。
3. 加 regularization，避免 aggressive behavior 并保证 temporal smoothness。

随后用 [[Motion Imitation|motion imitation]] 训练 proxy agent $\pi_{proxy}$。这个 agent 有 simulator privileged state，因此不是最终部署模型，而是用来在线生成大规模 behavior supervision。

Proxy agent 的 privileged proprioception：

$$
s_{t}^{p,sim}
\triangleq
[p_t,q_t,\theta_t,\dot{p}_t,\dot{q}_t,\omega_t,a_{t-1}]
$$

它包含刚体位置、关节位置、姿态、线速度、关节速度、角速度和上一动作。

### Phase 2: BFM pretraining

BFM 学的是 demonstrated behavior distribution $P(\tau)$。在 Markov 假设下，目标变成在 state-action 数据集上最大化动作似然。但单独用 proprioception 的 monolithic policy 很难控制，因此引入 observable goal state $s_t^{g,real}$，让不同控制模式都变成 goal state 的不同采样。

BFM 的 observable proprioception：

$$
s_t^{p,real}
\triangleq
[q_{t-25:t}, \dot{q}_{t-25:t}, \omega^{root}_{t-25:t}, g_{t-25:t}, a_{t-25:t-1}]
$$

也就是把最近 25 步的关节位置、关节速度、root angular velocity、projected gravity 和上一动作堆叠起来。

### Control interface and mask strategy

BFM 的控制接口包含三类低层 control modes：

| Control mode | 内容 | 作用 |
|--------------|------|------|
| Root Control | root translation、RPY orientation、linear velocity、angular velocity | 控制整体位姿和速度 |
| Kinematic Position Control | local-frame rigid-body link positions | 支持关键点/身体部位约束 |
| Joint Angle Control | 每个 motor 的目标关节角 | 支持最具体的 tracking/teleop 控制 |

不同 control modes 通过 bit-wise binary mask 激活。与 [[HOVER]] 的两阶段 mask strategy 不同，BFM 直接对每个 mask element 从 Bernoulli distribution 采样；训练初期用 mask curriculum，从全可见逐渐衰减到 $\mathcal{B}(0.5)$，避免 cold start 不稳定。

### CVAE 行为模型

BFM 用 [[Conditional Variational Autoencoder|CVAE]] 表示：

- prior $\rho$: $P(z\mid s_t^{p,real},s_t^{g,real})$
- encoder $\epsilon$: $q(z\mid s_t^{p,sim},s_t^{g,sim})$
- decoder $D$: $P(a_t\mid s_t^{p,real},s_t^{g,real},z)$

一个关键设计是 decoder 不输入 $s_t^{g,real}$，只看 $s_t^{p,real}$ 和 latent $z$，迫使 latent space 编码更多 behavioral knowledge。

### Online distillation

BFM 不只做离线 BC。每个 episode 中：

1. rollout 当前 BFM $\pi_\theta(a_t\mid s_t^{p,real},s_t^{g,real})$。
2. 在每个 timestep 计算对应 privileged states。
3. 查询 proxy agent 得到 reference action $\hat{a}_t$。
4. 用 DAgger-style loss 更新 BFM。

这让 BFM 看到自己 rollout 产生的状态分布，减少纯离线 imitation 的 covariate shift。

### Applications

1. **Direct steering**：手工 mask 对应控制接口，直接完成 motion tracking、VR teleoperation、locomotion。
2. **Behavior composition**：对两个 control modes 的 latent 做线性插值，例如 Roundhouse Kick 中 root-only 和 keypoint-only 的组合。
3. **Behavior modulation**：类似 classifier-free guidance，在 latent prior 上做 extrapolation，让行为更贴近期望控制模式。
4. **Efficient acquisition**：冻结 BFM，只训练 residual decoder $\pi(\Delta a_t\mid s_t^{p,real},z)$ 获取新行为。

## 关键公式

### 公式1: [[Goal-Conditioned Reinforcement Learning|Goal-conditioned WBC]]

$$
r_t = R(s_t^p, s_t^g),
\qquad
\max_{\pi}
\mathbb{E}
\left[
\sum_{t=1}^{T}\gamma^{t-1}r_t
\right]
$$

**含义**：humanoid control 被写成 goal-conditioned RL；action 是目标关节位置，送入 [[PD Controller]] 执行。

**符号说明**：

- $s_t^p$：humanoid proprioception。
- $s_t^g$：goal state，由具体控制模式指定。
- $a_t$：target joint positions。
- $R(\cdot)$：任务 reward。

### 公式2: [[Behavior Foundation Model|行为似然目标]]

$$
\max_{\theta}
\mathbb{E}_{(s_{t}^{p,real},a_t)\sim\mathcal{D}}
\left[
\log\pi_{\theta}(a_t\mid s_t^{p,real})
\right]
$$

**含义**：BFM 的基本目标是学习 demonstrated behavior 的 state-action distribution。

**符号说明**：

- $\mathcal{D}$：behavioral dataset。
- $s_t^{p,real}$：部署时可观测 proprioception。
- $a_t$：行为中的动作。

### 公式3: [[Masked Control Interface|Goal state marginalization]]

$$
\log\pi_{\theta}(a_t\mid s_t^{p,real})
=
\log
\mathbb{E}_{s_t^{g,real}\sim p(s_t^{g,real}\mid s_t^{p,real})}
\left[
\pi_{\theta}(a_t\mid s_t^{p,real},s_t^{g,real})
\right]
$$

**含义**：同一个 behavior 可以由多个 control modes 指定；因此动作似然可看作对 goal states 的边缘化。

**符号说明**：

- $s_t^{g,real}$：可观测 goal state。
- $p(s_t^{g,real}\mid s_t^{p,real})$：由 mask/control interface 诱导出的 goal-state distribution。

### 公式4: [[Jensen Lower Bound|Jensen lower bound]]

$$
\begin{aligned}
&\log
\mathbb{E}_{s_t^{g,real}\sim p(s_t^{g,real}\mid s_t^{p,real})}
\left[
\pi_{\theta}(a_t\mid s_t^{p,real},s_t^{g,real})
\right] \\
&\ge
\mathbb{E}_{s_t^{g,real}\sim p(s_t^{g,real}\mid s_t^{p,real})}
\left[
\log\pi_{\theta}(a_t\mid s_t^{p,real},s_t^{g,real})
\right]
\end{aligned}
$$

**含义**：用 Jensen inequality 得到可优化的 lower bound。

**符号说明**：

- 左侧是原始 marginal log-likelihood。
- 右侧是对 goal state 采样后可直接优化的条件 log-likelihood。

### 公式5: [[Conditional Variational Autoencoder|CVAE ELBO]]

$$
\begin{aligned}
&\mathbb{E}_{q(z\mid s_t^{p,sim},s_t^{g,sim})}
\left[
\log P(a_t\mid s_t^{p,real},s_t^{g,real},z)
\right] \\
&-
D_{KL}
\left[
q(z\mid s_t^{p,sim},s_t^{g,sim})
\Vert
P(z\mid s_t^{p,real},s_t^{g,real})
\right]
\end{aligned}
$$

**含义**：用 CVAE 建模多模态行为分布，既重建动作，又约束 posterior 贴近 prior。

**符号说明**：

- $z$：behavior latent。
- $q$：encoder/posterior。
- $P(z\mid\cdot)$：prior。
- $P(a_t\mid\cdot)$：decoder 动作分布。

### 公式6: [[Conditional Variational Autoencoder|Prior / Encoder / Decoder]]

$$
P(z\mid s_t^{p,real},s_t^{g,real})
=
\mathcal{N}
\left(
\mu^{\rho}(s_t^{p,real},s_t^{g,real}),
\sigma^{\rho}(s_t^{p,real},s_t^{g,real})
\right)
$$

$$
q(z\mid s_t^{p,sim},s_t^{g,sim})
=
\mathcal{N}
\left(
\mu^{\epsilon}(s_t^{p,sim},s_t^{g,sim},m_t)
+\mu^{\rho}(s_t^{p,real},s_t^{g,real}),
\sigma^{\epsilon}(s_t^{p,sim},s_t^{g,sim})
\right)
$$

$$
P(a_t\mid s_t^{p,real},s_t^{g,real},z)
=
\mathcal{N}
\left(
\mu^{D}(s_t^{p,real},z),
\sigma_{fixed}
\right)
$$

**含义**：encoder 被设计成 prior 的 residual，并显式输入 mask $m_t$；decoder 去掉 goal state 输入，迫使 $z$ 承载行为信息。

### 公式7: [[DAgger|在线蒸馏损失]]

$$
L_{DAgger}
=
\left\|
\hat{a}_t-a_t
\right\|_2^2
$$

$$
L_{KL}
=
D_{KL}
\left(
q_{\epsilon}(z_t\mid s_t^{p,sim},s_t^{g,sim})
\Vert
P_{\rho}(z\mid s_t^{p,real},s_t^{g,real})
\right)
$$

$$
L
=
L_{DAgger}
+\lambda_{KL}L_{KL}
$$

**含义**：动作重建保证 BFM 跟随 proxy agent，KL 项维持 latent space 结构。

**符号说明**：

- $\hat{a}_t$：proxy agent reference action。
- $a_t$：当前 BFM action。
- $\lambda_{KL}$：重建质量和 latent 正则之间的权重。

### 公式8: [[Behavior Modulation|latent extrapolation]]

$$
z
=
(1+\lambda)
\mu^{\rho}(s_t^{p,real},s_t^{g,real})
-
\lambda
\mu^{\rho}(s_t^{p,real},\emptyset),
\qquad
\lambda>0
$$

**含义**：类似 classifier-free guidance，通过有条件 prior 和空条件 prior 的差值加强控制模式约束。

**符号说明**：

- $\mu^\rho(s_t^{p,real},s_t^{g,real})$：有 goal state 条件下的 prior mean。
- $\mu^\rho(s_t^{p,real},\emptyset)$：无条件 prior mean。
- $\lambda$：modulation coefficient。

### 公式9: [[Residual RL|残差动作获取]]

$$
a_t'
=
a_t+\Delta a_t,
\qquad
\Delta a_t
\sim
\pi(\Delta a_t\mid s_t^{p,real},z)
$$

**含义**：冻结 BFM 输出基础动作，只训练 residual decoder 获取新行为。

**符号说明**：

- $a_t$：BFM 原动作。
- $\Delta a_t$：residual decoder 输出的动作修正。
- $a_t'$：最终执行动作。

## 关键图表

### Figure 1: BFM Implementation

![Figure 1](https://arxiv.org/html/2509.13780v1/x1.png)

- Human motion 先 retarget 到 humanoid。
- Proxy agent 用 privileged simulator state 做 motion imitation。
- 一个 behavior 可由多种 control modes 指定。
- BFM 用 CVAE + DAgger-style online distillation 预训练。

### Figure 2: Latent space applications

![Figure 2](https://arxiv.org/html/2509.13780v1/x2.png)

- (a) latent interpolation 做 behavior composition。
- (b) t-SNE 显示 latent space 对 standing/walking/sidestep 有方向性和对称性。
- (c) latent extrapolation 做 behavior modulation。
- (d) 冻结 BFM 后学习 residual decoder。
- (e) Side Salto 中 BFM + residual learning 比 RL from scratch 更高效。

### Table I: Proxy agent reward design

| 类别 | Term | Weight |
|------|------|--------|
| Task Reward | Body position | 1.0 |
| Task Reward | Body position selected keypoint | 1.6 |
| Task Reward | Body position feet | 2.1 |
| Task Reward | Body rotation | 0.5 |
| Task Reward | Body velocity | 0.5 |
| Task Reward | Body angular velocity | 0.5 |
| Task Reward | DoF position | 0.75 |
| Task Reward | DoF velocity | 0.5 |
| Penalty | Torque limits | -5.0 |
| Penalty | DoF position | -10.0 |
| Penalty | DoF velocity | -5.0 |
| Penalty | Termination | -200.0 |
| Regularization | Torque | -0.000001 |
| Regularization | Action rate | -0.5 |
| Regularization | Feet orientation | -2.0 |
| Regularization | Feet heading alignment | -0.02 |
| Regularization | Feet air time | -10.0 |
| Regularization | Slippage | -1.0 |
| Regularization | Hip pos | -1.0 |
| Regularization | Close feet distance | -0.5 |

**说明**：reward 包含 motion imitation task reward、penalty 和 regularization；训练初期更重 imitation，后续通过 curriculum 引入稳定性约束。

### Table II: Domain randomization

| 类别 | Term | Value |
|------|------|-------|
| Dynamics | Base CoM offset | $\mathcal{U}(-0.1,0.1)$ |
| Dynamics | Link mass | $\mathcal{U}(0.9,1.1)\times$ default |
| Dynamics | Friction | $\mathcal{U}(0.5,1.2)$ |
| Dynamics | P gain | $\mathcal{U}(0.9,1.1)\times$ default |
| Dynamics | D gain | $\mathcal{U}(0.9,1.1)\times$ default |
| Dynamics | Torque RFI | $0.05\times$ torque limit |
| External Perturbations | Push interval | $[5,10]$ |
| External Perturbations | Max push velocity | 1.0 |

**说明**：domain randomization 用于提升 sim-to-real 和扰动鲁棒性。

### Table III: VR teleoperation / motion tracking evaluation

指标列顺序为每个数据集的 `Empjpe / Empkpe / Elin / Eang`，均为越低越好。

| 任务 | Method | AMASS Train | AMASS Test | 100Style |
|------|--------|-------------|------------|----------|
| Proxy | Proxy Agent | 0.1864 / 49.3057 / 0.1469 / 0.9978 | 0.2137 / 56.1755 / 0.2631 / 1.3976 | 0.2460 / 64.1346 / 0.2036 / 1.2336 |
| VR Teleoperation | Specialist | 0.2113 / 65.4214 / 0.2375 / 1.0925 | 0.2555 / 80.5919 / 0.4779 / 1.5036 | 0.3062 / 89.9115 / 0.3189 / 1.1525 |
| VR Teleoperation | HOVER | 0.2676 / 91.2667 / 0.5047 / 1.6988 | 0.3055 / 102.8428 / 0.6468 / 1.8716 | 0.3455 / 119.8896 / 0.5351 / 1.6553 |
| VR Teleoperation | BFM RL from Scratch | 1.0516 / 399.6902 / 0.4976 / 2.0627 | 1.1672 / 403.8327 / 0.6528 / 2.3211 | 1.1300 / 429.2893 / 0.4418 / 1.6848 |
| VR Teleoperation | BFM Ours | 0.2447 / 72.3615 / 0.4006 / 1.2177 | 0.2235 / 63.1388 / 0.3066 / 1.4632 | 0.3169 / 87.0725 / 0.3238 / 1.1361 |
| Motion Tracking | Specialist | 0.1895 / 53.9515 / 0.1586 / 1.0268 | 0.2247 / 73.6332 / 0.3034 / 1.4685 | 0.2491 / 67.7765 / 0.2128 / 1.2411 |
| Motion Tracking | HOVER | 0.2010 / 65.9742 / 0.2189 / 1.1599 | 0.2416 / 87.0678 / 0.3749 / 1.6554 | 0.2562 / 73.9817 / 0.2608 / 1.3369 |
| Motion Tracking | BFM RL from Scratch | 1.0503 / 400.1505 / 0.4973 / 2.0590 | 1.1689 / 404.7451 / 0.6533 / 2.3532 | 1.1215 / 429.5739 / 0.4422 / 1.6933 |
| Motion Tracking | BFM Ours | 0.1920 / 51.8372 / 0.1542 / 1.0142 | 0.2226 / 61.1236 / 0.3051 / 1.4358 | 0.2637 / 66.4027 / 0.2072 / 1.2790 |
| Behavior Modulation | $\lambda=0.5$ | 0.1893 / 50.4801 / 0.1564 / 1.0419 | 0.2227 / 58.9844 / 0.2767 / 1.4099 | 0.2583 / 63.0582 / 0.2116 / 1.3394 |
| Behavior Modulation | $\lambda=1.0$ | 0.1875 / 49.8647 / 0.1609 / 1.0681 | 0.2223 / 58.7251 / 0.2870 / 1.4899 | 0.2562 / 62.5168 / 0.2224 / 1.3919 |
| Behavior Modulation | $\lambda=1.5$ | 0.1869 / 50.1451 / 0.1675 / 1.0866 | 0.2224 / 60.4565 / 0.2990 / 1.4964 | 0.2567 / 64.0520 / 0.2370 / 1.4618 |
| Behavior Modulation | $\lambda=2.0$ | 0.2625 / 76.2392 / 0.2615 / 1.5176 | 0.2254 / 67.6583 / 0.3158 / 1.5438 | 0.2625 / 76.2392 / 0.2615 / 1.5176 |

**说明**：BFM 基本全面优于 HOVER 和 RL from scratch，并在多数泛化指标上接近或超过 task specialist；moderate behavior modulation 常进一步降低 tracking error，但过大 $\lambda$ 会退化。

### Table IV: Locomotion evaluation

指标列顺序为每个数据集的 `Elin,xy / Eang,z`，均为越低越好。

| Method | AMASS Train | AMASS Test | 100Style |
|--------|-------------|------------|----------|
| Specialist | 0.1201 / 0.4801 | 0.2168 / 0.6751 | 0.1496 / 0.5108 |
| HOVER | 0.1494 / 0.5518 | 0.2663 / 0.7624 | 0.1696 / 0.5707 |
| BFM RL from Scratch | 0.4314 / 1.2925 | 0.5513 / 1.4982 | 0.4015 / 1.0606 |
| BFM Ours | 0.1292 / 0.4974 | 0.2116 / 0.6744 | 0.1603 / 0.4973 |

**说明**：BFM Ours 明显优于 HOVER 和从零 RL；在 AMASS Test 和 100Style 上甚至比 specialist 更好，说明统一行为模型没有只记住训练集控制模式。

## 实验

### Setup

- Simulation: [[Isaac Gym]] training with 8192 parallel environments。
- Sim-to-sim: [[MuJoCo]]。
- Real platform: [[Unitree G1]]，1.3m，29 DoF；实验冻结双腕后为 23 DoF。
- Data: [[AMASS]] Train/Test 和 [[100STYLE]]。
- Metrics:
  - Motion tracking / VR teleop: MPKPE、MPJPE、root linear velocity error、root angular velocity error。
  - Locomotion: root linear velocity error on xy-plane、root angular velocity error along z-axis。

### 主要结果

1. **Direct steering**：BFM 可以用不同 mask/control mode 直接运行 motion tracking、VR teleop 和 locomotion。
2. **泛化**：在 AMASS Test 和 100Style 上，BFM 多数指标优于 HOVER，且接近或超过 specialist。
3. **训练范式有效**：BFM from scratch 明显失败，说明先训练 proxy agent 再 online distillation 的路线很关键。
4. **latent structure**：t-SNE 显示 standing、walking forward/backward、sidestep 的 latent 有方向性和对称性。
5. **composition**：root-only 与 keypoint-only latent 插值后可完成 Roundhouse Kick。
6. **modulation**：$\lambda=0.5$ 或 $1.0$ 常能改善 tracking，$\lambda$ 太大则过度偏向 control mode，性能下降。
7. **residual acquisition**：Side Salto 上，BFM + residual learning 避免从零 RL 的低效探索。

## 批判性思考

### 优点

1. **问题定义清楚**：它把“多任务 humanoid 控制”拆成 behavior distribution + control-mode conditioning，比把每个任务都塞进一个 policy 更干净。
2. **接口设计实用**：root、keypoint、joint angle 加 mask 的 control interface 覆盖了低层 WBC 常见输入形式。
3. **CVAE latent 可解释**：不仅能直接控制，还展示了 composition、modulation 和 residual acquisition。
4. **包含真实平台验证**：不是只在 character simulation 中讨论 masked motion control。

### 局限性

1. **控制接口仍偏低层**：作者也承认当前 interface 主要是 root、kinematic position、joint angle，还没有真正覆盖语言或复杂 task-level goal。
2. **行为来自 mocap retargeting**：能力边界受 [[AMASS]] 和 retargeting 质量限制，真实接触、物体交互和手部精细操作不是重点。
3. **reward/代理训练仍重**：虽然 BFM 本身像 foundation model，但前置 proxy agent 仍依赖 motion imitation reward、domain randomization、hard negative mining 等工程。
4. **表格指标偏 tracking**：MPJPE/MPKPE/velocity error 能反映跟踪质量，但不能完全代表真实任务完成能力。
5. **代码尚未公开**：项目页标注 Code in coming，复现实验仍受限。

### 我的判断

[[BFM]] 更像 humanoid control 里的“behavior prior / policy prior”论文，而不是 VLA 论文。它和 [[OpenHLM]]、[[CoorDex]] 的共同点是都不愿意在高维全身动作空间里从零学控制；不同点是 BFM 把 motion dataset 和多控制模式统一进 CVAE latent，而不是面向具体 manipulation task 做 residual coordination。

最值得借鉴的是它的抽象：control mode 是 goal state 的采样方式，behavior 是 proprioception-action trajectory。这个抽象比具体的 CVAE 结构更重要。

## 可复现性评估

- [x] 论文公开
- [x] 项目页公开
- [x] 关键公式和表格公开
- [ ] 代码公开
- [ ] 训练好的模型公开
- [ ] 完整训练配置和数据处理脚本公开
- [ ] 真实机器人部署细节完整可复现

## 关联笔记

### 基于

- [[Whole-Body Control]]: 目标问题。
- [[Motion Imitation]]: proxy agent 的训练范式。
- [[Conditional Variational Autoencoder]]: BFM 的 generative model。
- [[DAgger]]: online distillation 框架。
- [[PPO]]: proxy agent 的 RL 优化算法。

### 数据和平台

- [[AMASS]]: human motion 数据源。
- [[SMPL]]: human body 参数化模型。
- [[Human Motion Retargeting]]: 从 human motion 到 humanoid motion。
- [[Unitree G1]]: 真实部署平台。
- [[Isaac Gym]]: 大规模并行训练环境。
- [[MuJoCo]]: sim-to-sim evaluation。
- [[100STYLE]]: locomotion/style 泛化评估数据。

### 对比与相关方法

- [[HOVER]]: 通用 humanoid WBC baseline。
- [[Masked Online Distillation]]: BFM 与 HOVER/MaskedMimic 一脉相承的训练路线。
- [[Residual RL]]: 用 BFM prior 快速获得新行为。
- [[PD Controller]]: 最底层执行 target joint positions。

## 速查卡片

> [!summary] BFM
> - **核心**: 用 CVAE + masked online distillation 学 humanoid behavior distribution。
> - **接口**: root / kinematic position / joint angle + bit-wise mask。
> - **训练**: AMASS retargeting -> proxy agent motion imitation -> DAgger distillation。
> - **能力**: 多控制模式 steering、latent composition/modulation、residual acquisition。
> - **限制**: 仍偏低层控制，代码未公开，真实任务完成能力不等同于 tracking 指标。

## 来源

- [arXiv](https://arxiv.org/abs/2509.13780)
- [arXiv HTML](https://arxiv.org/html/2509.13780v1)
- [Project page](https://bfm4humanoid.github.io/)

*笔记创建时间: 2026-06-25*
