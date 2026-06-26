---
title: "BeyondMimic: From Motion Tracking to Versatile Humanoid Control via Guided Diffusion"
method_name: "BeyondMimic"
authors: [Qiayuan Liao, Takara E. Truong, Xiaoyu Huang, Yuman Gao, Guy Tevet, Koushil Sreenath, C. Karen Liu]
year: 2025
venue: arXiv
tags: [humanoid, whole-body-control, motion-imitation, guided-diffusion, sim-to-real, robotics]
zotero_collection: _inbox
image_source: online
arxiv_html: https://arxiv.org/html/2508.08241v4
created: 2026-06-25
---

# BeyondMimic

## 一句话结论

[[BeyondMimic]] 的核心是把 humanoid [[Motion Imitation|motion tracking]] 做到足够稳定、自然、可真实部署，再用 [[Latent Diffusion Model|latent state-action diffusion]] + [[Classifier Guidance|classifier guidance]] 在推理时组合这些动作技能，零样本完成 joystick、keyframe inpainting、避障等下游任务。

## 论文定位

- 任务：[[Unitree G1]] humanoid 的 [[Whole-Body Control|whole-body control]]，覆盖动态动作追踪、自然行走/跑步、命令式 locomotion、[[Inference-Time Inpainting|motion inpainting]] 和避障。
- 两阶段路线：先用 [[Reinforcement Learning|RL]] + [[PPO]] 学一批高质量 [[Motion Imitation|motion tracking]] policies，再用 [[Conditional Variational Autoencoder|VAE]] 和 [[Latent Diffusion Model|LDM]] 蒸馏成可推理时优化的统一控制器。
- 关键差异：不是把 diffusion 直接放在 raw action space 上，而是建模 state-latent trajectory，让未来 state 也参与 [[Model Predictive Control|predictive control]]。
- 真实部署：tracking policy 在机器人 CPU 上跑 50 Hz；diffusion policy 以 25 Hz 部署，denoising 在 RTX 4060 Mobile 上异步运行，20 denoising steps 约 20 ms。
- 开源状态：项目页链接到 `HybridRobotics/whole_body_tracking`，仓库说明其覆盖 BeyondMimic 的 motion tracking training 部分；diffusion controller 不是这个仓库的主体。

## 核心贡献

1. **紧凑 motion tracking recipe**：用共享 MDP、共享 reward、共享超参追踪多种动作，避免每个 motion 手工调 reward 或 gain。
2. **面向真实机的简化 sim-to-real 配方**：少量 [[Domain Randomization|domain randomization]] + 正确 actuator armature + 低延迟 C++ 部署，比粗暴加随机化更有效。
3. **latent state-action diffusion**：先用 [[Conditional Variational Autoencoder|VAE]] 把 tracking policies 压到平滑 latent，再训练 state-latent diffusion，而不是直接生成高抖动 raw PD setpoints。
4. **推理时 guidance**：把 task cost 写成 differentiable objective，通过 [[Classifier Guidance|classifier guidance]] 修改 denoising 过程，零样本完成 velocity command、waypoint、keyframe、[[Signed Distance Field|SDF]] 避障。
5. **真实机验证强**：展示 aerial cartwheel、spin kick、flip kick、跑步、障碍绕行和外力干扰恢复，并通过用户研究量化 naturalness。

## 背景问题

传统 humanoid control 的两条路线都有明显问题。

1. Model-based hierarchical controllers 常把 CoM、momentum、contact schedule、footstep 等低维变量分层规划，计算上可控，但 motion space 被简化，容易产生固定 CoM 高度、膝盖长期弯曲、上身协调不足等不自然运动。
2. 纯 learning-based locomotion 能在特定任务上跑得很好，但 reward 往往按任务手工写，换动作就要重调；自然性、human-likeness 很难直接写成 reward。
3. DeepMimic/AMP 风格方法能从 human motion 学动作，但通常 motion-specific，部署时缺少可组合的下游任务能力。
4. VAE/goal-conditioned prior 可以组合技能，但训练时必须提前定义 goal condition，遇到避障、长时域导航这类隐式目标时容易 OOD、抖动或丢失自然性。

BeyondMimic 的判断是：先把底层 [[Motion Imitation|motion tracking]] 做成高质量、可扩展、可真实部署的技能库，再用 [[Latent Diffusion Model|diffusion prior]] 在推理时做局部规划和组合。

## 方法详解

### Stage 1: scalable human motion tracking

作者把每段 human-retargeted motion 当作 reference，训练一个对应的 tracking policy，但所有 motion 共享同一套 formulation 和超参。动作先经过 [[Human Motion Retargeting|retargeting]] 映射到 humanoid，再通过 body-level task reward 追踪。

关键设计：

- **Anchor-centered tracking**：非 anchor body 的目标 pose 不直接追全局轨迹，而是用 yaw-aligned、height-preserving transform 表达在当前 anchor frame 下，允许真实机因扰动产生 benign global drift。
- **Compact reward**：task reward 只追踪 body position、orientation、linear velocity、angular velocity；regularization 只保留 action smoothness、joint limit、undesired self-contact。
- **低阻抗 [[PD Controller|PD]] setpoint**：policy 输出 normalized joint position setpoints，经 [[Impedance Control|impedance control]]/PD 执行；gain 用 motor reflected inertia 估计，默认自然频率 10 Hz。
- **少量 domain randomization**：只随机化摩擦、restitution、joint offset、torso COM 和 root velocity perturbation。
- **[[Adaptive Sampling|adaptive sampling]]**：长 motion 被切成 1 秒 bins，训练时按 failure rate 重新采样，更快覆盖 cartwheel 等困难片段。

### Stage 2: versatile control via guided diffusion

第二阶段不再追踪某段 reference motion，而是把多个 tracking policies 的行为轨迹压进统一 trajectory prior。

流程：

1. 用 tracking policies 产生状态-动作数据。
2. 用 [[Conditional Variational Autoencoder|VAE]] 将 reference-motion intent 编码为 latent $z$，decoder 根据当前 proprioception 和 $z$ 重构 action。
3. Rollout VAE 产生 state-latent trajectories $\tau$。
4. 用 [[Latent Diffusion Model|LDM]] 学 $\tau$ 的分布，trajectory 包含过去 $N$ 步、当前步和未来 $H$ 步。
5. 推理时从 Gaussian noise 开始 denoise，并把 task-specific cost 的梯度加入 denoising，得到满足目标的 state-latent trajectory。
6. 只执行当前 denoised latent $z_t$ 解码出的动作，再滚动到下一控制步。

这个设计接近 [[Model Predictive Control|receding-horizon control]]，但 motion prior 来自 learned diffusion，而不是显式 dynamics model + 手写全套正则项。

### Guidance tasks

BeyondMimic 展示三类 task costs：

- **Joystick / velocity command**：让 predicted planar root velocity 匹配 commanded velocity。
- **Waypoint navigation**：远离目标时追 position，接近目标时转为最小化 velocity。
- **Obstacle avoidance**：用 [[Signed Distance Field|SDF]] 和 relaxed barrier function 对预测 horizon 内所有 body 的 collision risk 加 cost。

此外，keyframe-conditioned cartwheel 使用 [[Inference-Time Inpainting|motion inpainting]]：给出未来 0.2 s 间隔的 sparse keyframes，diffusion 在中间补出自然连续 motion。

## 关键公式

### 公式1: [[Motion Imitation|Anchor-centered tracking objective]]

$$
\bigl(
T_{\text{anchor}}^{\text{des}},\mathcal V_\text{anchor}^{\text{des}},\;
\{\,T_b^{\text{des}},\, \mathcal V_b^{\text{des}}\,\}_{\,b\in\mathcal B_{\text{target}}}
\bigr).
$$

**含义**：tracking objective 只要求 anchor 直接跟 reference；其他 target bodies 在 anchor-centered frame 中追踪相对姿态和速度。

**符号说明**：

- $T_{\text{anchor}}^{\text{des}}$：anchor body 的目标 pose。
- $\mathcal V_\text{anchor}^{\text{des}}$：anchor body 的目标 twist。
- $\mathcal B_{\text{target}}$：用于追踪的 body 集合，包含 end-effectors。

### 公式2: [[Reinforcement Learning|Task reward]]

$$
r_{\text{task}}
= \sum_{s \in \{\mathbf{p}, R, \mathbf{v}, \boldsymbol{\omega}\}} r(\bar{e}_s, \sigma_s).
$$

**含义**：把 position、orientation、linear velocity、angular velocity 的平均 tracking error 分别映射成 Gaussian-shaped score，再求和。

**符号说明**：

- $\bar e_s$：所有 target bodies 在某个状态量 $s$ 上的 mean squared error。
- $\sigma_s$：对应状态量的 tolerance scale。

### 公式3: [[Reinforcement Learning|Total reward]]

$$
r = r_{\text{task}} - \lambda_{l} r_{\text{limit}} - \lambda_{s} r_{\text{smooth}} - \lambda_{c} r_{\text{contact}}.
$$

**含义**：最终 reward 由 task tracking 和三个 regularization 组成。

**符号说明**：

- $r_{\text{limit}}$：关节越界惩罚。
- $r_{\text{smooth}}$：动作变化惩罚。
- $r_{\text{contact}}$：非 end-effector 自碰惩罚。

### 公式4: [[Whole-Body Control|Policy observation]]

$$
\mathbf{o}=[\boldsymbol{\psi},\,\mathbf{e}_{\text{anchor}},\,\mathcal{V}_{\text{imu}},\,\boldsymbol{\theta}- \boldsymbol{\theta}^{0},\,\dot{\boldsymbol{\theta}},\,\mathbf{a}_{\text{last}}].
$$

**含义**：tracking actor 不堆叠历史，输入 motion phase、anchor pose error、IMU twist、关节状态和上一步动作。

**符号说明**：

- $\boldsymbol{\psi}=[\boldsymbol{\theta}^{\text{ref}},\dot{\boldsymbol{\theta}}^{\text{ref}}]$：reference motion phase cue。
- $\mathbf{e}_{\text{anchor}}$：anchor position error + [[Rot6D|Rot6D]] orientation error。
- $\mathbf{a}_{\text{last}}$：上一动作，用于帮助估计当前 torque/contact 状态。

### 公式5: [[PD Controller|Joint setpoint action]]

$$
\boldsymbol{\theta}^{\text{sp}}
= \boldsymbol{\theta}^{0} + \boldsymbol{\alpha}\odot \mathbf{a}.
$$

**含义**：policy 输出 normalized action，经 per-joint scale 转成 joint position setpoint，由低层 PD 执行。

**符号说明**：

- $\boldsymbol{\theta}^{0}$：默认关节位置。
- $\boldsymbol{\alpha}$：per-joint action scale。
- $\odot$：逐元素乘法。

### 公式6: [[Latent Diffusion Model|VAE ELBO]]

$$
\mathcal{L}_{\text{VAE}} =
\mathbb{E}_{q_{_{\mathcal{E}}}(\mathbf{z}|\mathbf{x})}
\left[\|\mathbf{x} - \mathcal{D}(\mathbf{z})\|^2\right]
+ \beta\, D_{\mathrm{KL}}
\left(q_{_{\mathcal{E}}}(\mathbf{z}|\mathbf{x}) \,\|\, \mathcal{N}(\mathbf{0}, \mathbf{I})\right).
$$

**含义**：标准 latent diffusion 里的 VAE 目标，兼顾 reconstruction 和 latent Gaussian regularization。

**符号说明**：

- $\mathcal{E}, \mathcal{D}$：encoder 和 decoder。
- $q_{\mathcal{E}}(\mathbf z|\mathbf x)$：encoder posterior。
- $\beta$：KL 权重。

### 公式7: [[Latent Diffusion Model|Forward noising]]

$$
q_\text{forward}(\mathbf{z}^k \,|\, \mathbf{z}^0) =
\mathcal{N}\!\left(\sqrt{\bar{\alpha}_k}\,\mathbf{z}^0,\; (1-\bar{\alpha}_k)\mathbf{I}\right).
$$

**含义**：clean latent $\mathbf z^0$ 在第 $k$ 个 diffusion step 被加噪为 $\mathbf z^k$。

**符号说明**：

- $\bar{\alpha}_k = \prod_{i=1}^{k}\alpha_i$：累计 signal retention。
- $\mathbf I$：单位协方差。

### 公式8: [[Latent Diffusion Model|Latent diffusion training loss]]

$$
\mathcal{L}_\text{Diffusion}
= \mathbb{E}\!\left[
\|z_{\phi}(\mathbf{z}^k, k) - \mathbf{z}^0\|^2
\right].
$$

**含义**：denoising network 预测 clean latent，而不是预测 noise。

**符号说明**：

- $z_\phi$：denoising network。
- $k$：diffusion step。

### 公式9: [[Latent Diffusion Model|Reverse denoising]]

$$
\mathbf{z}^{k-1} =
\alpha_k\!\left(\mathbf{z}^k -
\gamma_k\big(\mathbf{z}^k - z_{\phi}(\mathbf{z}^k, k)\big)\right)
+ \sigma_k\,\mathcal{N}(0, \mathbf{I}).
$$

**含义**：从 noisy latent 逐步恢复 clean latent。

**符号说明**：

- $\alpha_k,\gamma_k,\sigma_k$：noise schedule coefficients。
- $\mathcal{N}(0,\mathbf I)$：采样噪声。

### 公式10: [[Conditional Variational Autoencoder|VAE action decoder]]

$$
\hat{\mathbf{a}} =
\mathcal{D}(\mathbf{z},\,
[\mathbf{g},\, \mathcal{V}_{\text{imu}},\, \boldsymbol{\theta},\, \dot{\boldsymbol{\theta}},\, \mathbf{a}_{\text{last}}]).
$$

**含义**：VAE decoder 用 latent motion intent 和最新 proprioception 生成当前动作。

**符号说明**：

- $\mathbf g$：root frame 下的 projected gravity。
- $\mathcal V_{\text{imu}}$：IMU twist。
- $\hat{\mathbf a}$：重构动作。

### 公式11: [[Conditional Variational Autoencoder|DAgger VAE objective]]

$$
\mathcal{L}_\text{VAE}
= \mathbb{E}_{q_{_{\mathcal{E}}}(\mathbf{z} \mid [\mathbf{c},\, \mathbf{e}_{\text{anchor}}])}
\!\left[\|\hat{\mathbf{a}} - \mathbf{a}\|^2\right]
+ \beta\, D_\mathrm{KL}\!\big(
q_{_{\mathcal{E}}}(\mathbf{z} \mid \boldsymbol{\psi},\, \mathbf{e}_{\text{anchor}})
\,\|\, \mathcal{N}(\mathbf{0},\mathbf{I})\big).
$$

**含义**：BeyondMimic 用 [[DAgger]] 风格训练 VAE，使 decoder 在自身 rollout 状态上也能重构 tracking policy action。

**符号说明**：

- $\mathbf c$：reference-motion components。
- $\mathbf e_{\text{anchor}}$：anchor error。
- $\mathbf a$：tracking policy action。

### 公式12: [[Latent Diffusion Model|State-latent trajectory]]

$$
\tau =
[\,\mathbf{s}_{t-N},\, \mathbf{z}_{t-N},\, \dots,\,
\mathbf{s}_{t},\, \mathbf{z}_{t},\dots,\,
\mathbf{s}_{t+H},\, \mathbf{z}_{t+H}\,].
$$

**含义**：diffusion 不是只生成动作，而是联合建模 history/current/future 的 states 和 action latents。

**符号说明**：

- $N$：history length。
- $H$：prediction horizon。
- $\mathbf z_{t+i}$：由 VAE 编码出的 action latent。

### 公式13: [[Latent Diffusion Model|Individual denoising steps]]

$$
\mathbf{k} =
\big[
k_{\mathbf{s}_{t-N}},\, k_{\mathbf{z}_{t-N}},\,
\dots,\,
k_{\mathbf{s}_t},\, k_{\mathbf{z}_t},\,
\dots,\,
k_{\mathbf{s}_{t+H}},\, k_{\mathbf{z}_{t+H}}
\big].
$$

**含义**：每个 state/latent 可有不同 noise level，便于做 observation inpainting 和 future pose inpainting。

**符号说明**：

- $\mathbf k$：trajectory 中每个元素的 denoising step。
- $k_{\mathbf s}, k_{\mathbf z}$：state 与 latent 的 step。

### 公式14: [[Latent Diffusion Model|Trajectory diffusion training loss]]

$$
\mathcal{L}_\text{Diffusion} =
\mathbb{E}\!\left[
\|z_{\phi}(\tau^{\mathbf{k}},\, \mathbf{k}) - \tau\|^2
\right].
$$

**含义**：denoising network 从 noised state-latent trajectory 重构 clean trajectory。

**符号说明**：

- $\tau^{\mathbf k}$：按 individual steps 加噪后的 trajectory。
- $z_\phi(\tau^{\mathbf k},\mathbf k)$：预测的 clean trajectory。

### 公式15: [[Latent Diffusion Model|Trajectory reverse denoising]]

$$
\tau^{\mathbf{k}-1} =
\alpha_{\mathbf{k}}\!\left(\tau^{\mathbf{k}} -
\gamma_{\mathbf{k}}\big(\tau^{\mathbf{k}} -
z_{\phi}(\tau^{\mathbf{k}},\, \mathbf{k})\big)\right)
+ \sigma_{\mathbf{k}}\,\mathcal{N}(0,\,\mathbf{I}).
$$

**含义**：从 Gaussian noise 逐步 denoise 出 temporally consistent state-latent trajectory。

**符号说明**：

- $\alpha_{\mathbf k},\gamma_{\mathbf k},\sigma_{\mathbf k}$：逐元素 schedule coefficients。
- $\tau^{\mathbf{k}-1}$：下一步更干净的 trajectory。

### 公式16: [[Classifier Guidance|Conditional score decomposition]]

$$
\nabla_{\boldsymbol{\tau}} \log p(\boldsymbol{\tau} \mid \boldsymbol{\tau}^*)
= \nabla_{\boldsymbol{\tau}} \log p(\boldsymbol{\tau})
+ \nabla_{\boldsymbol{\tau}} \log p(\boldsymbol{\tau}^* \mid \boldsymbol{\tau}).
$$

**含义**：用 Bayes rule 把 unconditional trajectory prior 的 score 改成 conditional score。

**符号说明**：

- $\boldsymbol{\tau}$：sampled trajectory。
- $\boldsymbol{\tau}^*$：desired/optimal trajectory。

### 公式17: [[Classifier Guidance|Cost-gradient guidance]]

$$
\nabla_{\boldsymbol{\tau}} \log p(\boldsymbol{\tau}^* \mid \boldsymbol{\tau})
= -\,\nabla_{\boldsymbol{\tau}} G(\boldsymbol{\tau}).
$$

**含义**：把 task likelihood 近似为 $\exp(-G)$ 后，conditional score 就等价于负 cost gradient。

**符号说明**：

- $G(\boldsymbol{\tau})$：differentiable task-specific cost。

### 公式18: [[Adaptive Sampling|Failure-rate sampling]]

$$
p_\text{s} =
\frac{\sum_{u=0}^{K-1} \rho^u\, \bar{f}_{\text{s}+u}}
{\sum_{j=1}^\text{S} \sum_{u=0}^{K-1} \rho^u\, \bar{f}_{j+u}}.
$$

**含义**：训练 reset 时更常采样近期容易失败的 motion bins。

**符号说明**：

- $\bar f_s$：第 $s$ 个 bin 的 smoothed failure rate。
- $\rho=0.8$：look-back kernel decay。
- $K$：look-back window。

### 公式19: [[Latent Diffusion Model|Relative root pose]]

$$
\mathbf{p}_{\text{root}}^{\,\text{rel}}(t{+}n)
= (\mathbf{R}_{\text{yaw}}^{\,t})^\top
\!\big(\mathbf{p}_{\text{root}}^{\,t+n}-\mathbf{p}_{\text{root}}^{\,t}\big),
\qquad
\mathbf{R}_{\text{root}}^{\,\text{rel}}(t{+}n)
= (\mathbf{R}_{\text{yaw}}^{\,t})^\top \mathbf{R}_{\text{root}}^{\,t+n}.
$$

**含义**：root pose 用当前 timestep 的 character-yaw frame 表达，消除全局平移和 yaw。

**符号说明**：

- $\mathbf R_{\text{yaw}}^t$：当前 root yaw rotation。
- $n\in[-N,H]$：history/horizon index。

### 公式20: [[Latent Diffusion Model|Relative root twist]]

$$
\mathbf{v}_{\text{root}}^{\,\text{rel}}(t{+}n)
= (\mathbf{R}_{\text{yaw}}^{\,t})^\top
\!\big(\mathbf{v}_{\text{root}}^{\,t+n}-\mathbf{v}_{\text{root}}^{\,t}\big),
\qquad
\boldsymbol{\omega}_{\text{root}}^{\,\text{rel}}(t{+}n)
= (\mathbf{R}_{\text{yaw}}^{\,t})^\top \boldsymbol{\omega}_{\text{root}}^{\,t+n}.
$$

**含义**：root velocity 和 angular velocity 同样在当前 yaw frame 下归一化。

**符号说明**：

- $\mathbf v_{\text{root}}$：root linear velocity。
- $\boldsymbol\omega_{\text{root}}$：root angular velocity。

### 公式21: [[Latent Diffusion Model|Local body features]]

$$
\mathbf{p}_{b}^{\,\text{local}}(t{+}n)
= (\mathbf{R}_{\text{yaw}}^{\,t+n})^\top
\!\big(\mathbf{p}_{b}^{\,t+n}-\mathbf{p}_{\text{root}}^{\,t+n}\big),
\qquad
\mathbf{v}_{b}^{\,\text{local}}(t{+}n)
= (\mathbf{R}_{\text{yaw}}^{\,t+n})^\top
\!\big(\mathbf{v}_{b}^{\,t+n}-\mathbf{v}_{\text{root}}^{\,t+n}\big).
$$

**含义**：body features 用每个时刻自己的 local root frame 表达，保留局部身体结构。

**符号说明**：

- $b\in\mathcal B_{\text{target}}$：被追踪/建模的 body。

### 公式22: [[Ornstein-Uhlenbeck Noise|OU action perturbation]]

$$
\eta_{t+1} =
\eta_t + \theta (\mu - \eta_t)\, \Delta t
+ \sigma \sqrt{\Delta t}\, \varepsilon_t,
\quad \varepsilon_t \sim \mathcal{N}(0, I).
$$

**含义**：dataset collection 时加入 temporally correlated action noise，形成可恢复的 error band。

**符号说明**：

- $\theta=0.8$：mean reversion rate。
- $\mu=0$：long-term mean。
- $\sigma=0.1$：joint-wise noise scale。

### 公式23: [[Model Predictive Control|Joystick cost]]

$$
G_{\mathrm{js}}(\hat{\mathbf{\tau}}_t)
= \frac{1}{2} \sum_{i=0}^{H}
\|V_{xy,i}(\hat{\mathbf{\tau}}_t) - \mathbf{g}_{v} \|^2.
$$

**含义**：让预测 horizon 内的 planar root velocity 匹配 joystick command。

**符号说明**：

- $V_{xy,i}$：第 $i$ 个 horizon step 的预测平面速度。
- $\mathbf g_v$：速度命令。

### 公式24: [[Model Predictive Control|Waypoint cost]]

$$
G_{\mathrm{wp}}(\hat{\mathbf{\tau}}_t)
= \sum_{i=0}^{H}
(1-e^{-2d_i})\|P_{xy,i}(\hat{\mathbf{\tau}}_t) - \mathbf{g}_{p}\|^2
+ e^{-2d_i}\|V_{xy,i}(\hat{\mathbf{\tau}}_t)\|^2.
$$

**含义**：远处追位置，近处减速，避免到达 waypoint 后继续冲过目标。

**符号说明**：

- $P_{xy,i}$：第 $i$ 个 horizon step 的平面位置。
- $d_i=\|P_{xy,i}(\hat{\tau}_t)-\mathbf g_p\|$：到目标距离。

### 公式25: [[Signed Distance Field|Obstacle SDF cost]]

$$
G_{\mathrm{sdf}}(\hat{\mathbf{\tau}}_t)
= \sum_{i=0}^{H} \sum_{b\in\mathcal{B}}
B(\mathrm{SDF}(P_{b,i}(\hat{\mathbf{\tau}}_t))-r_b, \delta).
$$

**含义**：对 horizon 内所有 body 的 obstacle distance 加 barrier cost。

**符号说明**：

- $P_{b,i}$：body $b$ 在 horizon step $i$ 的位置。
- $r_b$：body collision radius。
- $\delta$：barrier relaxation threshold。

### 公式26: [[Signed Distance Field|Relaxed barrier]]

$$
B(x, \delta) =
\begin{cases}
-\ln(x) & \text{if } x \geq \delta \\
-\ln(\delta) + \frac{1}{2}
\left[
\left( \frac{x-2\delta}{\delta} \right)^2 - 1
\right] & \text{if } x < \delta
\end{cases}
$$

**含义**：距离足够大时用 log barrier，过近时用二次形式避免数值爆炸。

**符号说明**：

- $x=\mathrm{SDF}(P)-r_b$：扣除 collision radius 后的 signed clearance。

## 关键图表

### Figure 1: Overview of versatile humanoid control

![Figure 1](https://arxiv.org/html/2508.08241v4/x1.png)

展示两阶段框架：A 是从 human motions 学 agile、human-like motion tracking；B 是 guided diffusion 在 unseen downstream tasks 上组合 learned motor skills。

### Figure 2: Diverse motion tracking policies on real humanoid

![Figure 2](https://arxiv.org/html/2508.08241v4/x2.png)

展示 30 个硬件部署 clips 的子集，覆盖静态平衡、站起、单脚跳、turn kick、180/360 spin jump、cartwheel、dance 和 sport motions。

### Figure 3: Human-level agility in dynamic acrobatic motion

![Figure 3](https://arxiv.org/html/2508.08241v4/x3.png)

展示真实森林环境中的连续 acrobatic sequence：180 度 aerial cartwheel、两个 180 度 side spin kicks 和 360 度 flip kick。论文报告 airborne phase 峰值加速度 $31\,\mathrm{m/s^2}$、pelvic angular velocity 最高 $20\,\mathrm{rad/s}$，均值 $7.01\,\mathrm{rad/s}$，接近人类 aerial motion 文献中的 $7.75\,\mathrm{rad/s}$ 平均值。

### Figure 4: Natural walking and running

![Figure 4](https://arxiv.org/html/2508.08241v4/x4.png)

对比 humanoid 和 human 的 weight-normalized GRF profiles，并展示 user study。用户研究 $N=77$，BeyondMimic 整体自然性偏好为 70.8% vs. 29.2%，walking 为 57.0% vs. 43.0%，running 为 84.7% vs. 15.3%，均显著。

### Figure 5: Command-conditioned locomotion via guided diffusion

![Figure 5](https://arxiv.org/html/2508.08241v4/x5.png)

展示 joystick/waypoint guidance 下的 diffusion process、waypoint navigation、joystick teleoperation、latent t-SNE 和 walking-to-running transition。模拟评估中 walking/running 平均 velocity tracking error 分别为 12.14% 和 13.65%。

### Figure 6: Task transition and composition

![Figure 6](https://arxiv.org/html/2508.08241v4/x6.png)

展示 keyframe motion inpainting、cartwheel 与 walking/running 的长时域组合，以及用 waypoint cost + obstacle avoidance cost 完成真实机绕障。

### Figure 7: Framework overview

![Figure 7](https://arxiv.org/html/2508.08241v4/x7.png)

更细化地画出 tracking 阶段和 diffusion 阶段：Stage 1 用 motion tracking 学 diverse skills；Stage 2 先用 VAE 经 DAgger 压缩 policies，再用 state-latent diffusion rollouts 训练 LDM，推理时通过 guidance 解决下游任务。

### Figure 8: Ablation studies

![Figure 8](https://arxiv.org/html/2508.08241v4/x8.png)

展示 orientation representation、history length、armature、deployment delay 和 adaptive sampling 的消融。结论是 continuous orientation、无 observation history、正确 armature、低延迟和 adaptive sampling 对 sim-to-real 很关键。

### Figure S1: Motion Tracking Overview

![Figure S1](https://arxiv.org/html/2508.08241v4/x9.png)

展示 actor-critic tracking 架构：actor 输入 motion phase、anchor error、IMU twist、relative joint positions、joint velocities 和 previous action；critic 额外看 per-body relative poses；reward 包含四个 tracking terms 和三个 regularizers。

### Figure S2: Extra ablation on PD gains

![Figure S2](https://arxiv.org/html/2508.08241v4/figures/sup_fig2_w.png)

比较自然频率 $\omega$ 的影响：$\omega=10\,\mathrm{Hz}$ 在 global tracking 上最好，$\omega=25\,\mathrm{Hz}$ 虽然 local error 略低，但会造成高频振荡和明显 torque overshoot。

## 关键表格

### Table S1: Unified reward formulation

| Term group | Reward term | Equation / definition | Weight |
|------------|-------------|-----------------------|--------|
| Task | Body Position | $\exp(-(\frac{1}{\lvert\mathcal B_{\mathrm{target}}\rvert}\sum_b\|\mathbf p_b^{des}-\mathbf p_b\|^2)/0.3^2)$ | 1.0 |
| Task | Body Orientation | $\exp(-(\frac{1}{\lvert\mathcal B_{\mathrm{target}}\rvert}\sum_b\|\log(R_b^{des}R_b^\top)\|^2)/0.4^2)$ | 1.0 |
| Task | Body Linear Velocity | $\exp(-(\frac{1}{\lvert\mathcal B_{\mathrm{target}}\rvert}\sum_b\|\mathbf v_b^{des}-\mathbf v_b\|^2)/1.0^2)$ | 1.0 |
| Task | Body Angular Velocity | $\exp(-(\frac{1}{\lvert\mathcal B_{\mathrm{target}}\rvert}\sum_b\|\boldsymbol\omega_b^{des}-\boldsymbol\omega_b\|^2)/3.14^2)$ | 1.0 |
| Task optional | Anchor Position | $\exp(-\|\mathbf p_{\text{anchor}}^{des}-\mathbf p_{\text{anchor}}\|^2/0.3^2)$ | 0.5 |
| Task optional | Anchor Orientation | $\exp(-\|\log(R_{\text{anchor}}^{des}R_{\text{anchor}}^\top)\|^2/0.4^2)$ | 0.5 |
| Regularization | Action smoothness | $\|\mathbf a_t-\mathbf a_{t-1}\|^2$ | -0.1 |
| Regularization | Joint position limit | $\sum_{j=1}^{N}[\max(l_j-\theta_j,0)+\max(\theta_j-u_j,0)]$ | -10.0 |
| Regularization | Undesired self-contacts | $\sum_{b\notin\mathcal B_{\mathrm{ee}}}\mathbf 1[\|f_b^{self}\|>1\text{N}]$ | -0.1 |

### Table S2: Domain randomization parameters

| Domain randomization | Sampling distribution |
|----------------------|-----------------------|
| Static friction coefficients | $\mu_{\text{static}}\sim\mathcal U[0.3,1.6]$ |
| Dynamic friction coefficients | $\mu_{\text{dynamic}}\sim\mathcal U[0.3,1.2]$ |
| Restitution coefficient | $e_{\text{rest}}\sim\mathcal U[0,0.5]$ |
| Default joint positions | $\Delta\theta_j^0\sim\mathcal U[-0.01,0.01]$ rad; ankle joints use $\mathcal U[-0.1,0.1]$ rad |
| Torso COM offset | $\Delta x\sim\mathcal U[-0.025,0.025]$, $\Delta y\sim\mathcal U[-0.05,0.05]$, $\Delta z\sim\mathcal U[-0.05,0.05]$ m |
| Root linear velocity perturbation | $v_x,v_y\sim\mathcal U[-0.5,0.5]$ m/s, $v_z\sim\mathcal U[-0.2,0.2]$ m/s |
| Push duration | $\Delta t\sim\mathcal U[1.0,3.0]$ s |
| Root angular velocity perturbation | $\omega_x,\omega_y\sim\mathcal U[-0.52,0.52]$, $\omega_z\sim\mathcal U[-0.78,0.78]$ rad/s |

### Table S3: Actuator reflected inertia

| Actuator | $g_1$ | $g_2$ | $J_r$ | $J_1$ | $J_2$ | $J_r(g_1g_2)^2$ | $J_1g_2^2$ | Total |
|----------|------:|------:|------:|------:|------:|------------------:|-------------:|------:|
| 5020-16 | 3.56 | 4.5 | 1.390e-05 | 1.700e-06 | 1.690e-05 | 3.558e-03 | 3.443e-05 | 3.610e-03 |
| 7520-14.3 | 4.5 | 3.18 | 4.890e-05 | 9.800e-06 | 5.330e-05 | 1.003e-02 | 9.921e-05 | 1.018e-02 |
| 7520-22.5 | 4.5 | 5 | 4.890e-05 | 1.090e-05 | 7.380e-05 | 2.476e-02 | 2.725e-04 | 2.510e-02 |
| 4010-25 | 5 | 5 | 6.800e-06 | -- | -- | 4.250e-03 | -- | 4.250e-03 |

### Table S4: Joint-axis inertia for left arm and left leg

| Joint | $I_\text{axis}^\text{subtree}$ | $I_\text{armature}$ | $I_\text{axis}^\text{eff}$ | Armature % |
|-------|-------------------------------:|--------------------:|---------------------------:|-----------:|
| left_shoulder_pitch_joint | 1.748e-01 | 3.610e-03 | 1.785e-01 | 2.0% |
| left_shoulder_roll_joint | 1.324e-01 | 3.610e-03 | 1.360e-01 | 2.7% |
| left_shoulder_yaw_joint | 2.742e-02 | 3.610e-03 | 3.103e-02 | 11.6% |
| left_elbow_joint | 3.445e-02 | 3.610e-03 | 3.806e-02 | 9.5% |
| left_wrist_roll_joint | 3.664e-04 | 3.610e-03 | 3.976e-03 | 90.8% |
| left_wrist_pitch_joint | 4.804e-03 | 4.250e-03 | 9.054e-03 | 46.9% |
| left_wrist_yaw_joint | 1.837e-03 | 4.250e-03 | 6.087e-03 | 69.8% |
| left_hip_pitch_joint | 8.543e-01 | 1.018e-02 | 8.644e-01 | 1.2% |
| left_hip_roll_joint | 6.398e-01 | 2.510e-02 | 6.649e-01 | 3.8% |
| left_hip_yaw_joint | 1.296e-01 | 1.018e-02 | 1.398e-01 | 7.3% |
| left_knee_joint | 1.115e-01 | 2.510e-02 | 1.366e-01 | 18.4% |
| left_ankle_pitch_joint | 2.777e-03 | 7.219e-03 | 9.997e-03 | 72.2% |
| left_ankle_roll_joint | 3.871e-04 | 7.219e-03 | 7.607e-03 | 94.9% |

### Table S5: Motion tracking hyperparameters

| Hyperparameter | Value |
|----------------|-------|
| Actor MLP hidden dimensions | [512, 256, 128] |
| Critic MLP hidden dimensions | [512, 256, 128] |
| Activation function | ELU |
| Steps per environment | 24 |
| Max iterations | 30,000 |
| Learning rate | $1\times10^{-3}$ |
| Clip parameter | 0.2 |
| Entropy coefficient | 0.005 |
| Value loss coefficient | 1.0 |
| Discount factor $\gamma$ | 0.99 |
| GAE $\lambda$ | 0.95 |
| Desired KL | 0.01 |
| Learning epochs | 5 |
| Mini-batches | 4 |

### Table S6: VAE hyperparameters

| Hyperparameter | Value |
|----------------|-------|
| Latent dimension | 32 |
| Student encoder MLP hidden dimensions | [2048, 1024, 512] |
| Student decoder MLP hidden dimensions | [2048, 1024, 512] |
| Teacher hidden MLP hidden dimensions | [512, 256, 128] |
| Activation function | ELU |
| Learning rate | $5\times10^{-4}$ |
| Accumulated gradient steps | 15 |
| KL loss coefficient | 0.01 |

### Table S7: Diffusion policy hyperparameters

| Hyperparameter | Value |
|----------------|-------|
| Horizon | 16 |
| Observation history | 4 |
| Embedding dimension | 512 |
| Attention heads | 8 |
| Transformer layers | 6 |
| Denoising steps | 20 |
| Batch size | 512 |
| Number of epochs | 1000 |
| Learning rate | $1\times10^{-4}$ |
| Weight decay | 0.001 |
| LR scheduler | Cosine |
| LR warmup gradient steps | 10,000 |
| EMA power | 0.75 |
| EMA max value | 0.9999 |

### Table S8: Motion segments tested in sim and real

| Name | Sim | Real [s] |
|------|-----|----------|
| Cristiano Ronaldo | Full | Full |
| Side Kick | Full | Full |
| Single Leg Balance | Full | Full |
| Swallow Balance | Full | Full |
| Aerial Cartwheel | Full | Full |
| Double Kicks 1 | Full | Full |
| Double Kicks 2 | Full | Full |
| walk1_subject1 | Full | [0.0, 33.0], [81.2, 86.7] |
| walk1_subject2 | Full | - |
| walk1_subject5 | Full | [146.7, 159.0], [206.7, 263.7] |
| walk2_subject1 | Full | - |
| walk2_subject3 | Full | [42.7, 75.7], [217.6, 230.6] |
| walk2_subject4 | Full | [154.4, 164.4], [218.6, 238.6] |
| dance1_subject1 | Full | [0.0, 118.0] |
| dance1_subject2 | Full | Full |
| dance1_subject3 | Full | - |
| dance2_subject1 | Full | - |
| dance2_subject2 | Full | - |
| dance2_subject3 | Full | [43.1, 163.1], [164.3, 184.3] |
| dance2_subject4 | Full | [156.3, end] |
| dance2_subject4 | Full | - |
| fallAndGetUp1_subject4 | Full | - |
| fallAndGetUp2_subject2 | Full | [0.0, 21.0], [74.0, 91.2], [94.0, 109.0] |
| fallAndGetUp2_subject3 | Full | [26.5, 46.5] |
| run1_subject2 | Full | [0.0, 50.0] |
| run1_subject4 | Full | - |
| run1_subject5 | Full | - |
| run2_subject1 | Full | [0.0, 11.0], [167.4, 204.4] |
| jumps1_subject1 | Full | [24.3, 42.3], [71.6, 81.6], [205.5, 226.5] |
| jumps1_subject2 | Full | - |
| jumps1_subject5 | Full | - |
| fightAndSports1_subject1 | Full | [16.8, 25.4], [201.6, end] |
| fightAndSports1_subject4 | Full | - |
| fight1_subject2 | Full | - |
| fight1_subject3 | Full | - |
| fight1_subject5 | Full | - |

## 实验结果

### Motion tracking scalability

- 训练数据约 2.5 小时 diverse human motions。
- 所有 motions 在 high-fidelity simulation 中通过验证。
- 真实机部署 30 个代表性 clips，总计 15 分钟。
- 很多 clips 与多个技能一起训练，总 reference motion 超过 3 分钟，仍保留 agility 和 stylistic details。

### Agility and naturalness

- 森林/软土/落叶/不平地面中完成 aerial cartwheel、spin kicks、flip kick 等训练中未直接覆盖的真实接触条件。
- GRF profile 与人类 walking/running 形状相似；walking 中 force peaks 更尖，作者归因于 humanoid 缺少 toe joint，stance rollover 和 toe-off 不如人。
- 用户研究中，相比 Unitree native walking/running controller，BeyondMimic 被显著认为更自然。

### Zero-shot downstream control

- **Commanded locomotion**：waypoint 和 joystick command 下都能平稳行走、后退、转向，并在外力扰动后恢复。
- **Long-horizon running**：在 track 上连续跑超过 50 m。
- **Gait transition**：只有 desired velocity 时，自然从 walking 过渡到 running；这种 transition 在 motion data 中少且未显式标注。
- **Motion inpainting**：给 cartwheel keyframes 后，从 walking 平滑进入 cartwheel，再回到 command-conditioned walking。
- **Task composition**：waypoint cost + SDF obstacle avoidance cost 能绕开障碍并到达目标。

## 消融结论

1. **Rotation representation**：quaternion/axis-angle 的不连续性会伤害真实部署；[[Rot6D]] 效果更好。
2. **Observation history**：在作者的 minimal randomization setup 中，加入 history 反而变差，可能因为 policy 记住了 simulation-specific state-action patterns。
3. **Armature**：不能把 armature 当数值稳定参数随便调；错误 armature 会显著恶化真实 tracking。
4. **Latency**：2 ms delay 已经增加 velocity error，5 ms 有一次 failure，10 ms 三次中失败两次。
5. **Adaptive sampling**：没有 adaptive sampling，4 个 motions 中 3 个在 30k iterations 后仍有困难片段未解决；简单 motion 的所需 iterations 也从 2k 增至 4k。
6. **Latent diffusion**：cartwheel sim-to-sim ablation 中，无 latent encoding baseline 只有 5% success，latent diffusion 达到 95% success，并能迁移到真实机。

## 批判性思考

### 优点

1. **工程判断非常强**：论文没有把 sim-to-real gap 全部丢给随机化，而是强调 actuator inertia、delay、C++ realtime stack、PD gain 的物理一致性。
2. **motion quality 与 task versatility 同时处理**：很多 humanoid work 只能二选一，要么动作好但不可组合，要么可控但不自然；BeyondMimic 的两阶段设计把两者分开优化。
3. **guidance cost 足够简单**：joystick、waypoint、SDF 都是轻量 differentiable cost，说明 learned prior 已经承担了大部分 motion feasibility。
4. **真实机证据充分**：不仅有单个 demo，还包括 outdoor terrain、user study、扰动恢复、ablation 和长时域跑步。

### 局限性

1. **依赖 proprioceptive state estimation**：diffusion model 继承 state estimation 质量；感知错误会直接污染 future trajectory。
2. **预测 horizon 短**：0.64 s horizon 足够局部避障和反应式控制，但不足以处理远距离目标、复杂障碍布局或需要早期决策的任务。
3. **history 可能锁定 gait orbit**：history 有利于稳定预测，但也可能让模型陷入重复 motion pattern；加大 guidance weight 又可能在 mode switching 时 destabilize denoising。
4. **fine-grained control 仍不足**：guidance 对粗粒度目标有效，对精细轨迹/身体部位约束可能还需要 supervised fine-tuning、adapter-style control 或更细的 cost tuning。
5. **开源范围有限**：公开仓库主要是 motion tracking training，完整 diffusion deployment/control stack 是否可复现仍需额外确认。

## 关联笔记

### 基于

- [[Motion Imitation]]：第一阶段的技能学习基础。
- [[Human Motion Retargeting]]：把 human motion 映射到 humanoid reference。
- [[PPO]]：motion tracking policies 的 RL 训练算法。
- [[Conditional Variational Autoencoder]]：用于把 tracking policy behavior 压缩成 latent。
- [[DAgger]]：VAE 训练中用于减小 covariate shift。

### 对比

- [[BFM]]：同样面向 humanoid behavior prior；BFM 更偏统一 control interface + CVAE behavior distribution，BeyondMimic 更强调 diffusion guidance 和真实机 agile motion composition。
- [[CoorDex]]：CoorDex 面向 dexterous loco-manipulation 的 body/hand prior 协调；BeyondMimic 面向 whole-body agile skill synthesis 和 locomotion/navigation。
- [[HOVER]]：通用 humanoid whole-body controller baseline；BeyondMimic 的核心是 motion tracking recipe + diffusion-based online optimization。

### 方法相关

- [[Classifier Guidance]]：推理时用 task cost 梯度 steer diffusion。
- [[Latent Diffusion Model]]：在低维 latent trajectory 中建模 human-like skill distribution。
- [[Signed Distance Field]]：用于 obstacle avoidance cost。
- [[Adaptive Sampling]]：用于 motion tracking 训练 reset。
- [[Sim-to-Real Transfer]]：贯穿 tracking 和 diffusion 部署。

## 速查卡片

> [!summary] BeyondMimic
> - **核心**: 先把 humanoid motion tracking 做到真实机可用，再用 guided latent diffusion 组合技能。
> - **方法**: RL tracking policy -> VAE latent -> state-latent diffusion -> classifier-guided inference-time control。
> - **结果**: 30 个真实机 tracking clips、aerial cartwheel/flip kick、自然 walking/running、joystick/waypoint/inpainting/SDF 避障。
> - **代码**: https://github.com/HybridRobotics/whole_body_tracking
> - **项目页**: https://beyondmimic.github.io/
> - **arXiv**: https://arxiv.org/abs/2508.08241

## 完整性自检

- [x] Figures: 8 张正文图 + 2 张补充图均已嵌入。
- [x] Tables: 源码中的 8 个 table environments 均已转写。
- [x] Equations: 主要 display equations、附录 state/cost equations、关键 inline action/reward formulas 已保留。
- [x] Concepts: 正文关键技术术语已用 Obsidian wikilink 标注。
- [x] Images: 使用 arXiv HTML 图片外链，已抽样验证可访问。

*笔记创建时间: 2026-06-25*
