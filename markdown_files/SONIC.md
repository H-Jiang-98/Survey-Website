---
title: "SONIC: Supersizing Motion Tracking for Natural Humanoid Whole-Body Control"
method_name: "SONIC"
authors: [Zhengyi Luo, Ye Yuan, Tingwu Wang, Chenran Li, Fernando Castañeda, Sirui Chen, Zi-Ang Cao, Jiefeng Li, David Minor, Qingwei Ben, Jinhyung Park, David Sami, Zi Wang, Xingye Da, Runyu Ding, Cyrus Hogg, Lina Song, Edy Lim, Eugene Jeong, Tairan He, Haoru Xue, Wenli Xiao, Simon Yuen, Jan Kautz, Yan Chang, Umar Iqbal, Linxi Jim Fan, Yuke Zhu]
year: 2025
venue: arXiv
tags: [humanoid, whole-body-control, motion-tracking, scaling-laws, vla, teleoperation, sim-to-real]
zotero_collection: _inbox
image_source: online
arxiv_html: https://arxiv.org/html/2511.07820v3
created: 2026-06-25
---

# SONIC

## 一句话结论

[[SONIC]] 的核心是把 humanoid [[Motion Imitation|motion tracking]] 当作可 scaling 的基础任务：用 700 小时 [[Motion Capture|mocap]]、100M+ frames、42M 参数和 21k GPU hours 训练一个 [[Universal Token Space|universal-token]] whole-body policy，再接 [[Kinematic Motion Planner|kinematic planner]]、[[Teleoperation|VR/video teleop]] 和 [[Vision-Language-Action Model|VLA]]。

## 论文定位

- 任务：[[Unitree G1]] humanoid 的通用 [[Whole-Body Control|whole-body control]]，覆盖大规模 motion tracking、interactive locomotion、video/text/music control、VR teleoperation 和 VLA-driven loco-manipulation。
- 核心路线：multi-encoder motion command -> [[Finite Scalar Quantization|FSQ]] universal token -> robot control decoder -> 29 维 joint target action。
- Scaling claim：模型从 1.2M 扩到 42M 参数，数据从 4M 到 100M+ frames，训练从 2K 到 21K GPU hours；OOD motion tracking 随 data/model/compute 稳定提升。
- 系统 claim：同一个 policy 支持 robot motion、human [[SMPL]] motion、hybrid upper-body keypoints + lower-body planner、token-only VLA streaming。
- 工程实现：真实机部署最大 42M policy；policy loop 50 Hz，command writer 500 Hz，operator input 100 Hz，planner 10 Hz；Jetson Orin 上 policy forward 1-2 ms，motion generation 约 12 ms。
- 代码/模型：项目页指向 `NVlabs/GR00T-WholeBodyControl`，仓库说明包含 SONIC training stack、C++ deployment、checkpoints 和 teleoperation stack。

## 核心贡献

1. **把 motion tracking 作为 scaling task**：相比 locomotion reward engineering 或 AMP-style adversarial imitation，motion tracking 给每一帧 dense supervision，更适合随数据和计算扩展。
2. **Universal token interface**：robot encoder $\mathcal{E}_r$、human encoder $\mathcal{E}_h$、hybrid encoder $\mathcal{E}_m$ 都映射到同一个 [[Finite Scalar Quantization|FSQ]] token space，使同一 policy 接受多种 motion source。
3. **Kinematic planner 衔接任务意图与 tracking**：实时生成 0.8-2.4 s 的短时域 kinematic motion，让 gamepad/keyboard 的速度、方向、style、height command 变成可追踪 reference。
4. **统一 teleop/VLA action space**：VR 和 VLA 不直接输出高维 SMPL pose，而是输出 universal tokens + hand joints；VLA token action 比 SMPL action 平均高 42 个百分点成功率。
5. **真实机和规模验证**：在 123 个真实机 motion sequences 上达到 99.2% success，并展示 5 类 whole-body loco-manipulation VLA tasks。

## 背景问题

SONIC 反对的是“每个 humanoid task 都重新设计 reward/控制器”的路线。

1. 当前 humanoid controllers 常是小 MLP、少数 GPU、单任务 reward；换成 dancing、getting up、teleop、manipulation 就要重写目标。
2. AMP/ASE/CALM 这类 adversarial imitation 有统一目标，但 dataset diversity 增大后 discriminator feedback 变难，容易 mode collapse。
3. 即使学到大技能库，还需要一个能服务下游任务的接口：teleoperation、navigation、VLA、video/text/music generation 都不能各用一套 policy。
4. SONIC 的判断是：大规模 motion tracking 提供 dense per-frame supervision；再把不同 command source 对齐到一个 token space，就能把 tracking policy 变成 humanoid motor foundation model。

## 方法详解

### Universal humanoid motion tracking

SONIC 把 tracking 写成 [[Reinforcement Learning|RL]] MDP，并用 [[PPO]] 训练。State 分成两部分：

- $\mathbf{s}^{p}_{t}$：proprioception，包括 joint pose、joint velocity、root angular velocity、root-frame gravity、previous action，并堆叠 10-step history。
- $\mathbf{s}^{g}_{t}$：motion command，可以是 robot motion $\mathbf g_r$、human motion $\mathbf g_h$ 或 hybrid motion $\mathbf g_m$。

Action 是 29 维 joint target positions，由低层 [[PD Controller]] 执行。Reward 追 root pose、body link pose/velocity、end-effector position，同时惩罚 action rate、joint limit、undesired contact、head/wrist anti-shake 和 feet acceleration。

### Universal token space

三个 encoder 处理不同 command：

- **Robot encoder $\mathcal{E}_r$**：编码 robot joint positions/velocities 的未来帧。
- **Human encoder $\mathcal{E}_h$**：编码 [[SMPL]] 3D human joints 的未来帧，用于 full-body VR、video/text/music control。
- **Hybrid encoder $\mathcal{E}_m$**：编码当前 head/hand sparse upper-body keypoints + lower-body robot motion，用于 3-point VR teleop。

Encoder 输出被 [[Finite Scalar Quantization|FSQ]] 量化为 universal token $\mathbf z$。然后：

- control decoder $\mathcal{D}_c(\mathbf z,\mathbf s^p_t)$ 输出 joint action $\mathbf a_t$。
- robot motion decoder $\mathcal{D}_r(\mathbf z)$ 重建 robot motion command，作为 auxiliary supervision。

### Kinematic motion planner

[[Kinematic Motion Planner]] 是一个实时 latent generative model，训练在同一批 whole-body motion data 上。它把用户输入变成短时域 reference motion：

- 输入 context keyframes：近期 robot states。
- 输入 target keyframes：来自速度/方向/style command，或 squatting、crawling、boxing 等 skill-specific targets。
- 输出 0.8-2.4 s motion segments。
- 采用 masked token prediction：逐步确定最有信心的 latent tokens。
- 根轨迹用 critically damped spring model 过滤极端命令，避免从 $6\,\mathrm{m/s}$ 瞬间反向到 $-6\,\mathrm{m/s}$ 这类不现实目标。

### Multi-modal and VLA integration

SONIC 用 [[GEM]] 处理 video/text/music：

- video：支持 prerecorded clips 和 monocular webcam，估计 human motion $\geq 60$ fps。
- text：自然语言 prompt 生成 target motion，例如 walk forward、kick left foot。
- music：根据 rhythm/melody 生成 dance motion。

VLA 部分连接 [[GR00T N1.5]]：

- 3-point task 输出 upper-body SE(3)、base height 和 navigation command，走 planner + hybrid encoder。
- whole-body tasks 让 VLA 输出 78 维 action：64 维 universal motion token + 14 维 hand joints。
- 相比直接预测 81 维 SMPL pose，FSQ token action 更平滑、更安全、成功率更高。

## 关键公式

### 公式1: [[Reinforcement Learning|Motion tracking MDP]]

$$
\mathcal{M}
=
\langle
\mathcal{S},\mathcal{A},\mathcal{T},\mathcal{R},\gamma
\rangle,
\qquad
\max_{\pi}
\mathbb{E}
\left[
\sum_{t=1}^{T}\gamma^{t-1}r_t
\right].
$$

**含义**：SONIC 将 humanoid motion tracking 作为 goal-conditioned RL 问题。

**符号说明**：

- $\mathcal{S}$：state space，由 proprioception 和 motion command 组成。
- $\mathcal{A}$：29 维 joint target action space。
- $\mathcal{R}$：tracking reward + penalties。
- $\gamma$：discount factor。

### 公式2: [[Whole-Body Control|Policy state]]

$$
\mathbf{s}_t
=
(\mathbf{s}^{p}_{t},\mathbf{s}^{g}_{t}),
\qquad
\mathbf{s}^{p}_{t}
\triangleq
(\mathbf q_t,\dot{\mathbf q}_t,\boldsymbol\omega_t,\mathbf g_t,\mathbf a_{t-1})_{t-9:t}.
$$

**含义**：actor 使用部署时可用的 10-step proprioception history 和当前 motion command。

**符号说明**：

- $\mathbf q_t$：joint pose。
- $\dot{\mathbf q}_t$：joint velocity。
- $\boldsymbol\omega_t$：root angular velocity。
- $\mathbf g_t$：root frame 下 gravity vector。
- $\mathbf a_{t-1}$：previous action。

### 公式3: [[Universal Token Space|Control decoder]]

$$
\mathbf{a}_t
=
\mathcal{D}_c(\mathbf{z},\mathbf{s}^{p}_{t}).
$$

**含义**：universal token 与 proprioception 一起被 control decoder 解码成 robot joint targets。

**符号说明**：

- $\mathbf z$：FSQ universal token。
- $\mathcal{D}_c$：robot control decoder。

### 公式4: [[Universal Token Space|Robot motion decoder]]

$$
\hat{\mathbf g}_r
=
\mathcal{D}_r(\mathbf z).
$$

**含义**：辅助 decoder 从 universal token 重建 robot motion command，帮助对齐 latent space。

**符号说明**：

- $\hat{\mathbf g}_r$：重建的 robot motion command。
- $\mathcal{D}_r$：robot motion decoder。

### 公式5: [[PPO|Joint training loss]]

$$
\mathcal{L}
=
\mathcal{L}_{\text{ppo}}
+
\mathcal{L}_{\text{recon}}
+
\mathcal{L}_{\text{token}}
+
\mathcal{L}_{\text{cycle}}.
$$

**含义**：PPO 负责控制性能，三个 auxiliary losses 负责跨 encoder 对齐和 motion reconstruction。

**符号说明**：

- $\mathcal{L}_{\text{ppo}}$：standard PPO loss。
- $\mathcal{L}_{\text{recon}}$：robot motion reconstruction。
- $\mathcal{L}_{\text{token}}$：encoder token alignment。
- $\mathcal{L}_{\text{cycle}}$：human-to-robot-to-token cycle consistency。

### 公式6: [[Universal Token Space|Reconstruction loss]]

$$
\begin{aligned}
\mathcal{L}_{\text{recon}}
=&
\left\| \mathcal{D}_r(\mathbf{z}_r) - \mathbf{g}_r \right\|^2
+
\left\| \mathcal{D}_r(\mathbf{z}_h) - \mathbf{g}_r \right\|^2 \\
&+
\left\| \mathcal{D}_r(\mathbf{z}_m) - \mathbf{g}_r \right\|^2.
\end{aligned}
$$

**含义**：无论输入来自 robot/human/hybrid encoder，都要能重建同一 robot motion command。

**符号说明**：

- $\mathbf z_r,\mathbf z_h,\mathbf z_m$：robot/human/hybrid encoder 产生的 token。
- $\mathbf g_r$：robot motion command。

### 公式7: [[Universal Token Space|Token alignment loss]]

$$
\mathcal{L}_{\text{token}}
=
\left\|\mathbf{z}_r-\mathbf{z}_h\right\|^2
+
\left\|\mathbf{z}_r-\mathbf{z}_m\right\|^2
+
\left\|\mathbf{z}_m-\mathbf{z}_h\right\|^2.
$$

**含义**：同一 motion 的三种输入格式应该落到相同 token neighborhood。

**符号说明**：

- $\mathbf z_r$：robot encoder token。
- $\mathbf z_h$：human encoder token。
- $\mathbf z_m$：hybrid encoder token。

### 公式8: [[Universal Token Space|Cycle consistency loss]]

$$
\mathcal{L}_{\text{cycle}}
=
\left\|
\mathcal{E}_r(\mathcal{D}_r(\mathbf{z}_h)) - \mathbf{z}_r
\right\|^2.
$$

**含义**：human token 经 robot-motion decoder 重建后，再用 robot encoder 编码，应回到 robot token。

**符号说明**：

- $\mathcal E_r$：robot motion encoder。
- $\mathcal D_r$：robot motion decoder。

### 公式9: [[Kinematic Motion Planner|Latent motion encoding]]

$$
\left\{z_t\right\}_{t=1}^{T/4}
=
\operatorname{enc}
\left(
\left\{p_t,r_t\right\}_{t=1}^{T}
\right).
$$

**含义**：planner 先把连续 kinematic motion 压成 downsampled latent token sequence。

**符号说明**：

- $p_t$：pose configuration。
- $r_t$：root position。
- downsampling rate：4。

### 公式10: [[Masked Token Prediction|Planner masked-token prediction]]

$$
h
=
\mathcal{F}
\left(
\left\{p_t,r_t\right\}_{t=1}^{4},
\left\{p_t,r_t\right\}_{t=T-4}^{T},
\left\{z_t\right\}_{t=1}^{T/4}
\right),
\qquad
\operatorname{Prob}(z_t)=\sigma(h).
$$

**含义**：planner 根据起点/终点 keyframes 和当前 masked tokens 预测各 token 的概率，逐步完成 inbetweening。

**符号说明**：

- $\mathcal F$：Transformer 或 Conv1D neural backbone。
- $h$：token logits。
- $\sigma$：softmax。

### 公式11: [[Kinematic Motion Planner|Critically damped spring model]]

$$
x(t)=
\left(
x_T-x_0
+
\left(
v_0+\frac{c}{2}(x_T-x_0)
\right)t
\right)
e^{-\frac{c}{2}t}.
$$

**含义**：把用户给出的 root position/heading command 平滑成 target keyframes，提高可预测性并过滤极端命令。

**符号说明**：

- $x_T$：target value。
- $x_0$：initial value。
- $v_0$：initial velocity。
- $c$：damping coefficient；位置用 $5\ln2$，heading 用 $20\ln2$。

### 公式12: [[Reinforcement Learning|Tracking reward and penalties]]

$$
r_t
=
\mathcal{R}(\mathbf{s}^{p}_{t},\mathbf{s}^{g}_{t})
+
\mathcal{P}(\mathbf{s}^{p}_{t},\mathbf a_t).
$$

**含义**：总 reward 由 tracking 项和 penalty 项组成。

**符号说明**：

- $\mathcal R$：root/body/end-effector tracking reward。
- $\mathcal P$：action smoothness、joint limit、contact、anti-shake、feet acceleration penalties。

### 公式13: [[Finite Scalar Quantization|FSQ quantization]]

$$
\mathbf z
=
Q_{\mathrm{FSQ}}(\mathcal E(\mathbf g)).
$$

**含义**：这是论文方法的抽象写法：任一 command encoder 的连续输出经过 FSQ 得到 universal token。论文正文没有单独给 FSQ 公式，但所有 encoder token 训练都依赖这个算子。

**符号说明**：

- $\mathcal E$：$\mathcal E_r,\mathcal E_h,\mathcal E_m$ 中任一 encoder。
- $\mathbf g$：robot/human/hybrid motion command。
- $Q_{\mathrm{FSQ}}$：finite scalar quantizer。

## 关键图表

### Figure 1: Universal control policy tasks

![Figure 1](https://arxiv.org/html/2511.07820v3/x1.png)

展示同一个 SONIC policy 通过不同输入接口完成 tracking、interactive control、teleop、多模态控制和 VLA loco-manipulation。

### Figure 2: Scaling, baselines, specialist and sim-to-real

![Figure 2](https://arxiv.org/html/2511.07820v3/x2.png)

核心定量图。上排展示 data/model/compute scaling；中排对比 GMT、Any2Track、BeyondMimic；下排对比 OpenHomie 和真实机 sim-to-real。最大模型在 test-content 达 99.6% success 和 23.8 mm MPJPE；真实机 123 sequences 达 99.2% success。

### Figure 3: Interactive motion control

![Figure 3](https://arxiv.org/html/2511.07820v3/x3.png)

展示导航 style switching、squatting/kneeling/crawling 高度控制，以及 responsive boxing。说明 planner 不是只会 locomotion，还能生成低姿态和娱乐/交互动作。

### Figure 4: Video teleoperation, multi-modal control, VR teleop

![Figure 4](https://arxiv.org/html/2511.07820v3/x4.png)

展示 video teleop、text/music motion generation 和 full-body VR teleop 都接入同一 universal token interface。

### Figure 5: VLA-driven loco-manipulation

![Figure 5](https://arxiv.org/html/2511.07820v3/x5.png)

展示 apple-to-plate、carrot pickup、scrub pickup、trash-can pedal、soda can to trash can、drill and box relocation。重点是 VLA 通过 token action 直接驱动 whole-body coordination，包括手和脚。

### Figure 6: Motion dataset panorama

![Figure 6](https://arxiv.org/html/2511.07820v3/figs/panorama_grid.png)

随机样本展示数据集多样性：locomotion、dance、combat、object manipulation、tool use、injured/stylistic gait 等。

### Figure 7: Method overview

![Figure 7](https://arxiv.org/html/2511.07820v3/x6.png)

展示 SONIC 的 multi-encoder 架构：robot/human/hybrid motion commands 进入 specialized encoders，映射到 universal token，驱动 robot control decoder 和 motion decoder。

### Figure 8: Latent space alignment

![Figure 8](https://arxiv.org/html/2511.07820v3/x7.png)

展示有无 consistency losses 时不同 encoder 的 latent alignment。加入 token/cycle losses 后，匹配帧的 cross-encoder distance 更接近对角线。

### Figure S1: Success and failure motions

![Figure S1](https://arxiv.org/html/2511.07820v3/figs/combined_success_failure.png)

上四行是 OOD 成功样例：hip-hop dance、stage bow、sword lunge、roundhouse kick；下两行是失败样例：zombie crawl 和 cross-legged sit。失败集中在极端地面接触和超出 robot kinematic limits 的姿态。

### Figure S2: Real-world evaluation motions

![Figure S2](https://arxiv.org/html/2511.07820v3/x8.png)

展示真实 Unitree G1 上的 123-sequence evaluation representative motions，包括 hip-hop dance、stage bow、high jump、kick、crouch walk、grovel。

### Figure S3: Robustness to external pushes

![Figure S3](https://arxiv.org/html/2511.07820v3/x9.png)

约 11 kg / 25 lb 物体从头顶高度落到机器人身上；机器人不使用 recovery module 或 policy adaptation，仍保持平衡并继续 tracking。

### Figure S4: OpenHomie scaling

![Figure S4](https://arxiv.org/html/2511.07820v3/x10.png)

显示 specialist locomotion controller OpenHomie 在 compute scale 上较快饱和：8 GPUs 处最好，扩到 32 GPUs 不再提升；论文用它反衬 motion tracking 的 dense supervision 更适合 scaling。

### Figure S5: Deployment architecture

![Figure S5](https://arxiv.org/html/2511.07820v3/x11.png)

展示 onboard multi-rate deployment stack：policy loop、command writer、input interface、planner、state logger、motion streaming、safety/watchdog。

## 关键表格

### Table 1: Dataset split statistics

| Item | Train | Test-Content | Test-Rep. |
|------|------:|-------------:|----------:|
| Clips | 317,189 | 6,998 | 6,306 |
| Duration (hours) | 611 | 15 | 12 |
| Unique sub-categories | 8,447 | 182 | 1,088 |
| Sub-cat. overlap w/ train | -- | 0% | 100% |
| Clip overlap w/ train | -- | 0% | 0% |
| Locomotion (basic + adv.) | 53,255 | 2,481 | 2,683 |
| Gestures | 37,939 | 1,488 | 1,125 |
| Acting / Roleplay | 68,742 | -- | 20 |
| Combat | 50,162 | -- | -- |
| Props / Object manipulation | 14,513 | 701 | 253 |
| Dance | 9,689 | 504 | 485 |
| Injured | 9,386 | 1,167 | 528 |
| Action / Tool use | 9,920 | 228 | 322 |
| Others | 63,583 | 429 | 890 |

**说明**：test-content 是完全未见过的 sub-categories；test-repetition 是训练见过类别的新表演/新 clips。

### Table 2: VLA task success rates

| Task | Interface | Training Data | Trials | Success |
|------|-----------|---------------|-------:|--------:|
| Apple to plate | 3-point | 300 trajs (single-obj) | 20 | 90% |
| Object pickup (carrot) | whole-body | 3,900 trajs (multi-obj) | 20 | 75% |
| Object pickup (scrub) | whole-body | 3,900 trajs (multi-obj) | 20 | 95% |
| Open trash can (foot) | whole-body | 200 trajs | 10 | 70% |
| Soda can to trash can | whole-body | 1,000 trajs (multi-obj) | 10 | 60% |
| Drill and box relocation | whole-body | 300 trajs | 10 | 70% |
| **Average (5 tasks)** | -- | -- | -- | **75%** |

**说明**：object pickup 两个 variant 作为一个 task family 计入 5-task average。

### Table 3: VLA action space ablation

| Task | FSQ Token | SMPL Poses | Delta |
|------|----------:|-----------:|------:|
| Carrot pickup | **75%** | 60% | +15 |
| Open trash can (foot) | **70%** | 20% | +50 |
| Soda can to trash can | **60%** | 0% | +60 |
| **Average** | **68%** | 27% | **+42** |

**说明**：universal motion token 比 explicit SMPL pose 更适合 VLA 学习，复杂长时域任务差距最大。

### Table 4: Ablation results

| Configuration | TC SR | TC MPJPE-L | TC Vel | TC Accel | TR SR | TR MPJPE-L | TR Vel | TR Accel |
|---------------|------:|-----------:|-------:|---------:|------:|-----------:|-------:|---------:|
| FSQ (ours) | **99.3** | **26.6** | **3.14** | **1.17** | **99.6** | **25.5** | **3.22** | **1.23** |
| VQ-VAE | 98.7 | 35.3 | 3.76 | 1.37 | 99.3 | 32.2 | 3.83 | 1.44 |
| FSQ-16-16 | 96.9 | 35.7 | 3.68 | 1.26 | 97.5 | 32.7 | 3.75 | 1.31 |
| FSQ-16-32 | 98.3 | 29.7 | 3.39 | 1.21 | 98.7 | 28.4 | 3.48 | 1.27 |
| FSQ-32-16 | 98.3 | 30.3 | 3.44 | 1.22 | 98.4 | 28.9 | 3.52 | 1.28 |
| FSQ-32-32 (ours) | **98.8** | **27.5** | **3.25** | **1.19** | **99.3** | **26.3** | **3.34** | **1.25** |
| Robot encoder $\mathcal E_r$ | **99.6** | **23.8** | **2.89** | **1.12** | 99.8 | **22.5** | **2.96** | **1.18** |
| Human encoder $\mathcal E_h$ | 99.6 | 24.4 | 3.04 | 1.24 | **99.8** | 23.1 | 3.11 | 1.30 |
| Hybrid encoder $\mathcal E_m$ | 99.2 | 26.5 | 3.25 | 1.22 | 99.7 | 25.2 | 3.31 | 1.28 |

**说明**：FSQ 明显优于 VQ-VAE；token dimension 比 quantization levels 更重要；三个 encoder 都维持高 success，human encoder 只比 robot encoder 多 0.6 mm MPJPE-L。

### Table S1: Universal control policy architecture

| Module | Architecture | Dims |
|--------|--------------|------|
| Quantizer | FSQ | token dimensions $D_z$; quantization levels $L_z$ |
| Encoder (g1) | MLP | hidden = [2048, 1024, 512, 512] |
| Encoder (teleop) | MLP | hidden = [2048, 1024, 512, 512] |
| Encoder (smpl) | MLP | hidden = [2048, 1024, 512, 512] |
| Decoder (actions) | MLP | hidden = [4096, 4096, 2048, 2048, 1024, 1024, 512, 512] |
| Decoder (refs) | MLP | hidden = [2048, 1024, 512, 512] |
| Action dimension | Diagonal Gaussian | 29 |
| Critic | MLP | hidden = [4096, 4096, 2048, 2048, 1024, 1024, 512, 512] |
| Future frames | -- | $F_r=F_h=F_m=10$ frames |
| Frame interval | -- | $\Delta t_r=\Delta t_m=0.1s$, $\Delta t_h=0.02s$ |

### Table S2: Training hyperparameters

| Training hyperparameter | Value |
|-------------------------|-------|
| Num parallel envs per GPU | 4096 |
| Num steps per env | 24 |
| Learning epochs | 5 |
| Num mini-batches | 4 |
| Discount $\gamma$ | 0.99 |
| GAE $\lambda$ | 0.95 |
| Clip parameter | 0.2 |
| Entropy coefficient | 0.013 |
| Value loss coefficient | 1.0 |
| Actor learning rate | $2\times10^{-5}$ |
| Critic learning rate | $1\times10^{-3}$ |
| Max gradient norm | 0.1 |
| Desired KL | 0.01 |
| Adaptive LR min/max | $[1\times10^{-5},2\times10^{-4}]$ |
| Init noise std | 0.05 |
| Actor std clamp min/max | [0.001, 0.5] |
| Adaptive sampling bin size | 1s |
| Adaptive sampling failure rate cap | $\beta=200$ |
| Adaptive sampling blending hyperparameter | $\alpha=0.1$ |

### Table S3: Reward design

| Reward term | Equation / definition | Weight |
|-------------|-----------------------|-------:|
| Root position | $\exp(-\|\mathbf p^p_{t,r}-\mathbf p^g_{t,r}\|_2^2/0.3^2)$ | 0.5 |
| Root orientation | $\exp(-\|\mathbf o^p_{t,r}-\mathbf o^g_{t,r}\|_2^2/0.4^2)$ | 0.5 |
| Body link pos (rel.) | $\exp(-\frac{1}{\lvert\mathcal B\rvert}\sum_{b\in\mathcal B}\|\mathbf p^{p,\mathrm{rel}}_{t,b}-\mathbf p^{g,\mathrm{rel}}_{t,b}\|_2^2/0.3^2)$ | 1.0 |
| Body link ori (rel.) | $\exp(-\frac{1}{\lvert\mathcal B\rvert}\sum_{b\in\mathcal B}\|\mathbf o^{p,\mathrm{rel}}_{t,b}-\mathbf o^{g,\mathrm{rel}}_{t,b}\|_2^2/0.4^2)$ | 1.0 |
| Body link lin. vel | $\exp(-\frac{1}{\lvert\mathcal B\rvert}\sum_{b\in\mathcal B}\|\mathbf v^p_{t,b}-\mathbf v^g_{t,b}\|_2^2/1.0^2)$ | 1.0 |
| Body link ang. vel | $\exp(-\frac{1}{\lvert\mathcal B\rvert}\sum_{b\in\mathcal B}\|\boldsymbol\omega^p_{t,b}-\boldsymbol\omega^g_{t,b}\|_2^2/3.14^2)$ | 1.0 |
| End-effector position | $\exp(-\frac{1}{5}\sum_{k\in\mathcal K}\|\mathbf p^p_{t,k}-\mathbf p^g_{t,k}\|_2^2/0.1^2)$ | 2.0 |
| Action rate | $\|\mathbf a_t-\mathbf a_{t-1}\|_2^2$ | -0.1 |
| Joint limit | $\sum_j \mathbbm 1[\mathbf q_{t,j}\notin[\mathbf q^{min}_{t,j},\mathbf q^{max}_{t,j}]]$ | -10.0 |
| Undesired contacts | $\sum_{c\notin\{\mathrm{ankles},\mathrm{wrists}\}}\mathbbm 1[\|\mathbf F_c\|>1.0\mathrm N]$ | -0.1 |
| Anti-shake (ang. vel) | $\sum_{k\in\{\mathrm{wrists},\mathrm{head}\}}\|\boldsymbol\omega_{t,k}\|_2^2\cdot\mathbbm 1[\|\boldsymbol\omega_{t,k}\|>1.5]$ | -5e-3 |
| Feet acceleration | $\sum_{k\in\{\mathrm{ankles}\}}\|\ddot{\mathbf q}_{t,k}\|_2^2$ | -2.5e-6 |

### Table S4: Domain randomization

| Domain randomization | Sampling distribution |
|----------------------|-----------------------|
| Static friction coefficients | $\mu_s\sim\mathcal U[0.3,1.6]$ |
| Dynamic friction coefficients | $\mu_d\sim\mathcal U[0.3,1.2]$ |
| Restitution coefficient | $e\sim\mathcal U[0,0.5]$ |
| Default joint positions | $\mathbf q_0\leftarrow \mathbf q_0+\mathcal U[-0.01,0.01]$ |
| Base COM offset | $\Delta x\sim\mathcal U[-0.075,0.075]$, $\Delta y\sim\mathcal U[-0.1,0.1]$, $\Delta z\sim\mathcal U[-0.1,0.1]$ |
| Root linear velocity push | $v_x,v_y\sim\mathcal U[-0.5,0.5]$, $v_z\sim\mathcal U[-0.2,0.2]$ |
| Push duration | $\Delta t\sim\mathcal U[1,3]$ s |
| Root angular velocity push | roll/pitch $\sim\mathcal U[-0.52,0.52]$, yaw $\sim\mathcal U[-0.78,0.78]$ |
| Target position jitter | x/y $\pm0.05$, z $\pm0.01$ |
| Target orientation jitter | roll/pitch $\sim\mathcal U[-0.1,0.1]$, yaw $\sim\mathcal U[-0.2,0.2]$ |
| Target linear velocity jitter | x/y $\pm0.5$, z $\pm0.2$ |
| Target angular velocity jitter | roll/pitch $\sim\mathcal U[-0.52,0.52]$, yaw $\sim\mathcal U[-0.78,0.78]$ |
| Target joint jitter | $\Delta\mathbf q_t^g\sim\mathcal U[-0.1,0.1]$ |

## 实验结果

### Scaling motion tracking

- 数据：训练集 317,189 clips、611 hours、8,447 unique sub-categories；源数据约 700 hours，retarget/filter 后为 611 hours / 100M+ frames。
- 测试：test-content 6,998 clips/15 hours/182 novel sub-categories；test-repetition 6,306 clips/12 hours/1,088 known sub-categories new takes；外部 PHUMA 68,000 motions。
- 最大模型：42M 参数，100M+ frames，128 GPUs，7 天，约 21k GPU hours。
- Scaling 结论：data/model/compute 三个轴都提升 OOD 和 repetition performance；OOD gains 更明显。

### Baseline comparison

- SONIC 在 test-content/test-repetition/PHUMA 上 success 为 98.7% / 99.6% / 97.0%。
- BeyondMimic 在同设置下为 81.6% / 85.8% / 73.4%。
- Any2Track 为 31.1% / 38.4% / 58.6%。
- SONIC MPJPE-L 23.2 mm，相比 BeyondMimic 39.1 mm 降低 41%。

注意：论文明确说这些方法训练数据和 retargeting pipeline 不同，因此这个比较更像 cross-dataset generalization/scaling evidence，不是严格 data-matched benchmark。

### Specialist comparison

- 对比 OpenHomie velocity tracking，SONIC overall survival 98.5%，OpenHomie 43.0%。
- OpenHomie 在约 1.5 m/s 后 survival 明显下降，SONIC 到约 4 m/s 仍接近 100% stability。
- 附录 scaling 显示 OpenHomie 在 8 GPUs 达到最好，扩到 32 GPUs 不再提升；SONIC 随 compute 继续改善。

### Real-world deployment

- 真实 Unitree G1 上评估 123 motion sequences。
- Real success 99.2%，simulation success 100%。
- Real MPJPE-L 25.7 mm，sim 22.3 mm。
- Upper-body sim-to-real gap 很小：22.2 mm real vs. 21.8 mm sim。
- Feet gap 最大：53.7 mm real vs. 29.0 mm sim，说明真实接触和脚落点仍是主要难点。

### Interactive control and VLA

- Planner 支持速度 command 0-6 m/s、任意方向 0-360 度、drunken/injured/happy/stealth walking、boxing、squatting、kneeling、crawling。
- Crawling 支持 elbows/knees omnidirectional movement，速度 0-0.5 m/s。
- VLA 平均成功率 75%，最难的 soda-can-to-trash-can 为 60%。
- FSQ token action space 相比 explicit SMPL pose 在三项 VLA ablation 中平均 68% vs. 27%。

## 批判性思考

### 优点

1. **Scaling story 很清楚**：论文没有只展示 demo，而是系统扫 data/model/compute 三个轴，并区分 OOD content 与 seen-category repetition。
2. **接口设计有价值**：universal token 让 teleop、planner、VLA 和 multi-modal motion generation 共享一个低层 controller，避免每个接口单独训练控制器。
3. **VLA action representation 结论强**：FSQ token vs SMPL pose 的 ablation 很直接，说明 humanoid VLA 的动作空间选择比单纯加大模型更关键。
4. **部署栈完整**：真实机 multi-rate loop、安全 watchdog、encoder fallback、motion streaming 都写得很工程化。
5. **对 BeyondMimic 是自然延伸**：BeyondMimic 强在 agile tracking + guided diffusion composition，SONIC 则把 tracking 本身做成大规模 foundation controller 和 VLA interface。

### 局限性

1. **安全和能耗缺少形式化处理**：作者自己承认 extended deployment 的 safety/energy efficiency 没有系统处理。
2. **失败集中在复杂地面接触**：zombie crawl、cross-legged sit 这类 sustained/complex ground contact 仍然困难。
3. **强依赖大规模 proprietary motion data**：公开 BONES-SEED 只有 288 hours，而论文训练使用 retarget/filter 后 611 hours；完全复现 scaling 曲线可能成本很高。
4. **baseline comparison 不完全 data-matched**：Any2Track/BeyondMimic/GMT 的训练数据不同，论文也承认对比更偏 cross-dataset generalization evidence。
5. **VLA 成功率仍不高**：平均 75% 很有意义，但 soda-can-to-trash-can 60%、trash-can foot pedal 70%，说明 high-level autonomy 还远没到可靠产品级。

## 关联笔记

### 基于

- [[Motion Imitation]]：SONIC 把 motion tracking 作为 scalable foundational task。
- [[PPO]]：训练 universal control policy。
- [[Finite Scalar Quantization]]：把 encoder outputs 量化成 universal token。
- [[SMPL]]：human motion encoder 和 VR teleop 输入格式。
- [[Human Motion Retargeting]]：将 large-scale human mocap retarget 到 Unitree G1。
- [[Domain Randomization]]：增强 sim-to-real 和 planner output robustness。

### 对比

- [[BeyondMimic]]：同为 natural humanoid motion tracking；SONIC 更强调 data/model/compute scaling、universal token 和 VLA/teleop interface。
- [[BFM]]：都把 humanoid behavior prior 做成可复用基础；BFM 用 CVAE/masked interface，SONIC 用 FSQ token + PPO tracking。
- [[OpenHLM]]：都关注 whole-body humanoid VLA；OpenHLM 主要是 VLA adaptation recipe，SONIC 提供低层 whole-body token controller。
- [[CoorDex]]：CoorDex 是 body/hand priors for dexterous loco-manipulation；SONIC 是 whole-body motion prior + VLA action interface。

### 方法相关

- [[Universal Token Space]]：SONIC 的核心接口抽象。
- [[Kinematic Motion Planner]]：把用户 command 转成 tracking reference。
- [[GEM]]：video/text/music 到 human motion 的多模态生成模块。
- [[GR00T N1.5]]：用于 VLA-driven loco-manipulation。
- [[BONES-SEED]]：公开释放的大规模 motion capture 子集。
- [[Sim-to-Real Transfer]]：真实机部署和评估主线。

## 速查卡片

> [!summary] SONIC
> - **核心**: 把 humanoid motion tracking 扩展到 42M 参数、100M+ frames、21k GPU hours，并统一 planner/teleop/VLA action interface。
> - **方法**: multi-encoder $\rightarrow$ FSQ universal token $\rightarrow$ robot control decoder；辅以 reconstruction/token/cycle losses。
> - **结果**: OOD tracking 99.6% success；真实机 123 motions 99.2% success；VLA 平均 75% success。
> - **代码**: https://github.com/NVlabs/GR00T-WholeBodyControl
> - **项目页**: https://nvlabs.github.io/GEAR-SONIC/
> - **arXiv**: https://arxiv.org/abs/2511.07820

## 完整性自检

- [x] Figures: 8 张主文图 + 5 张补充图均已嵌入。
- [x] Tables: 8 个 table 文件均已转写。
- [x] Equations: 方法正文的 display equations、训练损失、planner 公式和 reward 公式均已保留。
- [x] Concepts: 正文关键技术术语已用 Obsidian wikilink 标注。
- [x] Images: 使用 arXiv HTML 图片外链，已抽样验证 URL 形式。

*笔记创建时间: 2026-06-25*
