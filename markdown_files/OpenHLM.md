---
title: "OpenHLM: An Empirical Recipe for Whole-Body Humanoid Loco-Manipulation"
method_name: "OpenHLM"
authors: [Yingdong Hu]
year: 2026
venue: arXiv
tags: [humanoid, vision-language-action, loco-manipulation, teleoperation, whole-body-control]
zotero_collection: _inbox
image_source: online
created: 2026-06-25
---

# OpenHLM

## 一句话结论

[[OpenHLM]] 的价值不是提出一个新模型名，而是系统回答“怎样把原本面向静态/轮式双臂平台的 [[Vision-Language-Action Model|VLA]] 改造成 whole-body humanoid loco-manipulation policy”。它把工程路线拆成 teleoperation/controller、VLA adaptation、heterogeneous co-training 三段，并用 controlled experiments 把很多平时靠直觉争论的问题量化。

## 论文定位

- 任务：whole-body humanoid loco-manipulation。
- 平台：Unitree G1 类 humanoid，带 wrist-mounted grippers 和 onboard cameras。
- 模型路线：从 robot-pretrained VLA 出发，改 action/proprioception interface，并用 whole-body teleop 与 cheaper data co-training。
- benchmark：HLM-12，12 个任务，分为 4 个 capability families。
- 关键结论：robot pretraining 比纯 VLM pretraining 更关键；validation action MSE 不是上机表现的好 proxy；stationary co-training 能补新 motion 和新语义，HuMI 更偏补语义。

## 背景问题

Whole-body humanoid VLA 难点不是“把 action dimension 调大”这么简单。

1. Humanoid 的动作空间包含腿、腰、躯干、双臂和夹爪。
2. 相机视角通常看不全下肢，proprioception 是否输入会影响策略。
3. 机器人预训练多来自静态/轮式双臂平台，embodiment gap 很大。
4. 全身遥操作数据贵，不能指望每个任务都采 full teleop。
5. action MSE 可能很好看，但机器人执行会抖、错过抓取或无法重试。

## 方法总览

OpenHLM 的核心不是单个新网络，而是一套把已有 robot-pretrained VLA 改造成 whole-body humanoid policy 的系统配方。论文把问题拆成三层：底层控制和数据采集接口、VLA action/proprioception interface、低成本数据源的 co-training。每一层都对应一个明确问题：能不能收集高质量全身示范，VLA 能不能控制 humanoid 全身自由度，便宜数据能不能替代昂贵 full teleop。

整体输入输出可以这样理解：

- 输入：头部/腕部相机图像、语言指令、humanoid proprioception、历史动作或 action chunk 上下文。
- 输出：覆盖上肢、夹爪、腰部和腿部的 whole-body joint targets/action chunk。
- 训练监督：whole-body teleoperation demonstrations、stationary same-embodiment demonstrations、[[HuMI]] robot-free demonstrations。
- 评估目标：不是离线 action MSE，而是 HLM-12 上的 closed-loop task progress。

### Phase I: Low-level controller and teleoperation

第一阶段解决“示范数据从哪里来，以及动作接口到底长什么样”。

OpenHLM 比较了 whole-body teleop 的接口选择，最后采用 joint-based whole-body interface。这个选择很关键，因为如果 teleop interface 只暴露末端位姿或上半身，训练出来的 VLA 很容易退化成“带腿的双臂平台”，而不是真正协调全身自由度的 humanoid policy。

这一阶段的系统包含：

- humanoid robot hardware：Unitree G1 类平台，带 wrist-mounted grippers 和 onboard cameras。
- teleop hardware：HMD、handheld controllers 和 leg/body trackers。
- action recording：把操作员的全身控制意图记录成 humanoid joint-level action。
- visual observation：头部和腕部相机提供 egocentric + manipulation-centric observations。
- latency handling：通过 future-frame preview sweep 分析遥操作延迟对数据质量的影响。

这一阶段的输出不是最终模型，而是后续 VLA fine-tuning 所需的高质量 whole-body demonstrations。

### Phase II: Whole-body VLA policy design

第二阶段解决“原本为静态/轮式双臂机器人训练的 VLA，怎样控制 humanoid 全身”。

默认 backbone 是 $\pi_{0.5}$ 这类 robot-pretrained VLA。论文没有重做 backbone，而是集中改接口：

1. **Action projection resizing**：原 action head 支持的维度不够，需要扩展到 humanoid whole-body action。
2. **Weight surgery**：保留 pretrained action projection 中已有双臂/夹爪维度的权重，只随机初始化新增 humanoid-specific dimensions。
3. **Action ordering**：保留 pretrained bimanual ordering，再把 waist/leg joints 追加进去，避免破坏 robot pretraining 中已经学到的 action semantics。
4. **Absolute joint targets**：默认预测绝对 joint target，而不是相对 delta。
5. **Proprioception input**：把全身 proprioception 输入给策略，因为 head/wrist cameras 看不全腿和腰。
6. **Multi-step flow matching**：保留 10-step action generation。one-step alternatives 虽然更快，但上机 task progress 明显下降。

这一阶段最重要的结论是：interface choices 会影响稳定性，但真正拉开差距的是 robot pretraining。$\pi_{0.5}$ 初始化明显强于 PaliGemma-only 初始化和 random initialization。更尖锐的一点是，PaliGemma 初始化在 validation action MSE 上可以接近 $\pi_{0.5}$，但机器人执行时抓取恢复和闭环纠错明显弱。

### Phase III: Heterogeneous co-training

第三阶段解决“full whole-body teleop 太贵，能不能用便宜数据扩任务”。

OpenHLM 比较两类便宜数据：

- **Stationary same-embodiment teleop**：机器人本体在场，但脚固定，采 manipulation-only demonstrations。它保留 same embodiment，因此能提供新 object grounding、新 language grounding，也能补一些新的 manipulation motion。
- **[[HuMI]] robot-free demonstrations**：用 handheld grippers 和 body trackers 采 task-space trajectories，再通过 IK 转成 humanoid-like action。它采集更快，但没有真实 humanoid 闭环，因此更擅长补语义，不擅长补全新 motion。

论文在 held-out tasks 上区分两类泛化：

1. **Motion-reuse tasks**：新物体、新语言，但动作模式和训练任务相近。HuMI 和 stationary teleop 都比较有效。
2. **New-motion task**：例如 Pouring，需要训练集中没有的 vessel-tilt motion。stationary teleop 能补，HuMI 明显不够。

这个结论很实用：便宜数据不是“越多越好”的同质数据。不同采集方式补的是不同缺口，混用时必须知道自己缺的是语义 grounding、物体识别，还是新的身体运动模式。

## 关键公式

OpenHLM 不是公式驱动论文，真正的贡献在系统实验。但为了复用它的方法，可以把关键接口写成下面这些 notation。它们不是为了替代原文，而是把论文中的设计选择变成可检查的数学对象。

### 1. Whole-body VLA policy interface

OpenHLM 的 policy 接收视觉、语言和 proprioception，输出一段 whole-body action chunk。

$$
\hat{\mathbf{a}}_{t:t+H}
=
\pi_{\theta}
\left(
\mathbf{o}^{head}_{t},
\mathbf{o}^{left\ wrist}_{t},
\mathbf{o}^{right\ wrist}_{t},
\mathbf{q}_{t},
\ell
\right)
$$

符号：

- $\mathbf{o}^{head}_{t}$ 是头部相机图像，主要提供 egocentric scene context。
- $\mathbf{o}^{left\ wrist}_{t}, \mathbf{o}^{right\ wrist}_{t}$ 是腕部相机图像，主要提供接触附近的 manipulation view。
- $\mathbf{q}_{t}$ 是 humanoid proprioception，包括关节状态和身体姿态相关信息。
- $\ell$ 是语言指令。
- $\hat{\mathbf{a}}_{t:t+H}$ 是 action chunk，而不是单步动作。

这个接口解释了为什么只靠视觉不够。Humanoid 的腿和腰经常不在相机视野里，如果不喂 proprioception，策略会在 closed-loop 执行中进入自己看不见也纠不回来的状态。

### 2. Humanoid action vector resizing

论文提到 $\pi_{0.5}$ 原始 action projection 支持最多 32 维，而 OpenHLM 的 whole-body action vector 至少是 34 维。

$$
\mathbf{a}_{t}^{humanoid}
=
[q^{left\ arm}_{t}, q^{left\ gripper}_{t}, q^{right\ arm}_{t}, q^{right\ gripper}_{t}, q^{waist}_{t}, q^{legs}_{t}]
$$

含义：

- 前半部分尽量保留 pretrained bimanual action ordering。
- humanoid-specific waist/leg joints 追加到后面。
- 通过 weight surgery 保留已有 action projection 权重，只初始化新增维度。

### 3. Weight surgery for action projection

把 pretrained bimanual VLA 扩展到 humanoid action space 时，OpenHLM 不直接随机初始化整个 action projection，而是做 weight surgery。可以抽象成：

$$
\tilde{W}_{1:d_{old},:}=W_{1:d_{old},:},
\qquad
\tilde{W}_{d_{old}+1:d_{new},:}\sim\mathcal{N}(0,\sigma^2)
$$

$$
d_{old}=32,\qquad d_{new}=34
$$

含义：

- 旧的 32 维 action 权重保留，避免破坏 pretrained bimanual manipulation prior。
- 新增 waist/leg/gripper 相关维度随机初始化。
- 这比完全随机初始化 action head 更符合迁移学习逻辑。

这里的设计选择和 action ordering 是一组：如果保留旧权重，却把 action ordering 全部打乱，pretraining 的动作语义也会被破坏。

### 4. Action ordering as semantic preservation

OpenHLM 默认保留 pretrained bimanual ordering，再追加 humanoid-specific joints。

$$
\mathbf{a}^{humanoid}_{t}
=
\left[
\mathbf{a}^{left\ arm}_{t},
\mathbf{a}^{left\ gripper}_{t},
\mathbf{a}^{right\ arm}_{t},
\mathbf{a}^{right\ gripper}_{t},
\mathbf{a}^{waist}_{t},
\mathbf{a}^{legs}_{t}
\right]
$$

对比的 alternative 是 humanoid-native ordering，例如 legs first。论文发现单独换 ordering 不是最大瓶颈，但默认 ordering 更安全，因为它最大程度保留了 pretrained robot action space 的结构。

### 5. Behavior cloning on action chunks

VLA fine-tuning 可以写成 action chunk imitation：

$$
\mathcal{L}_{BC}
=
\mathbb{E}_{(\mathbf{o},\ell,\mathbf{q},\mathbf{a})\sim\mathcal{D}}
\left[
\sum_{k=0}^{H}
\left\|
\pi_{\theta}(\mathbf{o}_{t},\ell,\mathbf{q}_{t})_{k}
-\mathbf{a}_{t+k}
\right\|_{2}^{2}
\right].
$$

符号：

- $\mathcal{D}$ 是 demonstration dataset。
- $H$ 是 action chunk horizon。
- $\mathbf{a}_{t+k}$ 是 teleop 记录的 whole-body joint target。
- $\pi_{\theta}(\cdot)_k$ 是模型预测的第 $k$ 步动作。

注意：这个 loss 只是训练目标，不是好策略的充分条件。OpenHLM 的实验证明，validation MSE 接近的模型，上机 task progress 可以差很多。

### 6. Flow-matching action generation

如果把 action generation 看成 flow matching，模型学习的是从 noise action 到 demonstration action 的向量场。

$$
\frac{d\mathbf{a}_{\tau}}{d\tau}
=
v_{\theta}
\left(
\mathbf{a}_{\tau},
\tau;
\mathbf{o}_{t},
\ell,
\mathbf{q}_{t}
\right),
\qquad
\tau\in[0,1]
$$

推理时用 $K$ 步积分得到 action chunk：

$$
\hat{\mathbf{a}}_{t:t+H}
=
\mathrm{Integrate}
\left(
v_{\theta}, K;
\mathbf{o}_{t},\ell,\mathbf{q}_{t}
\right)
$$

OpenHLM 的经验结论：

- $K=10$ 的 multi-step generation 更慢，但 closed-loop task progress 更稳。
- $K=1$ 的 one-step generation 离线 action MSE 可能更低，但动作更可能 jitter 或缺少可执行的时间结构。
- 所以 humanoid VLA 里，低延迟不是唯一目标，动作时间平滑性和恢复能力更重要。

### 7. Heterogeneous co-training mixture

第三阶段可以写成不同数据源的混合训练。

$$
\mathcal{D}_{train}
=
\mathcal{D}^{whole}_{1:8}
\cup
\lambda_s \mathcal{D}^{stationary}_{9:12}
\cup
\lambda_h \mathcal{D}^{HuMI}_{9:12}
$$

符号：

- $\mathcal{D}^{whole}_{1:8}$ 是 8 个训练任务的 full whole-body teleop。
- $\mathcal{D}^{stationary}_{9:12}$ 是 held-out tasks 的 stationary same-embodiment demos。
- $\mathcal{D}^{HuMI}_{9:12}$ 是 held-out tasks 的 robot-free HuMI demos。
- $\lambda_s,\lambda_h$ 表示不同数据源在采样或训练中的权重。

两种 held-out 泛化可以分开看：

$$
\mathrm{Generalization}
=
\mathrm{Semantic\ grounding}
+
\mathrm{Motion\ acquisition}
$$

OpenHLM 的结论是：

- stationary teleop 同时补 semantic grounding 和 motion acquisition。
- HuMI 主要补 semantic grounding，对完全新 motion 不够。

### 8. Evaluation proxy mismatch

OpenHLM 最值得记的经验式结论可以写成：

$$
\mathcal{L}_{val}^{MSE}(\theta_1)
\approx
\mathcal{L}_{val}^{MSE}(\theta_2)
\;\not\Rightarrow\;
\mathrm{Progress}_{robot}(\theta_1)
\approx
\mathrm{Progress}_{robot}(\theta_2)
$$

在论文里，$\pi_{0.5}$ 初始化和 PaliGemma 初始化的 validation action MSE 可以很接近，但 closed-loop robot behavior 不一样。前者更会 grasp retry 和 error correction，后者更容易在失败后恢复不了。

这条结论对 humanoid VLA 很重要：离线 imitation loss 只能说明模型像不像 demonstrator 的动作，不说明它能不能在机器人犯错后把状态拉回来。

## 关键图表

### Figure 1: OpenHLM overview

![Figure 1](https://arxiv.org/html/2606.22174v1/x1.png)

- 三阶段路线图。
- 重点是 controlled experiments，而不是单个模型结构图。

### Figure 2: HLM-12 benchmark

![Figure 2](https://arxiv.org/html/2606.22174v1/x2.png)

- 12 个任务分成 4 类能力。
- 用来测试全身行为，而不是只测桌面 manipulation。

### Figure 3: Joint-based vs SMPL-based whole-body teleop

![Figure 3](https://arxiv.org/html/2606.22174v1/x3.png)

- 支撑 joint-based teleop interface 的选择。
- 说明接口不是随便拍脑袋定的。

### Figure 4: Future-frame preview latency sweep

![Figure 4](https://arxiv.org/html/2606.22174v1/x4.png)

- 分析 teleop latency 与预览设置。
- 对真实采集系统很重要。

### Figure 5: VLA design ablations

![Figure 5](https://arxiv.org/html/2606.22174v1/x5.png)

- Amber：action/proprioception interface ablation。
- Rose：pretraining ablation，$\pi_{0.5}$ 明显优于 PaliGemma 和 from scratch。
- Sage：one-step action generation underperforms 10-step baseline by about 20 points。

### Figure 6: Whole-body teleop data scaling

![Figure 6](https://arxiv.org/html/2606.22174v1/x6.png)

- 10 -> 20 demos per task 收益最大。
- 40 demos per task 接近 90%，作为后续默认预算。

### Figure 7: Heterogeneous co-training results

![Figure 7](https://arxiv.org/html/2606.22174v1/x7.png)

- stationary co-training 和 HuMI co-training 的差异在 held-out tasks 上显现。
- 关键结论：HuMI 更会补语义，stationary teleop 更能补新 motion。

### Figure 8: Long-horizon language-conditioned task

![Figure 8](https://arxiv.org/html/2606.22174v1/figures/fruits.jpg)

- 用水果 long-horizon task 检查系统级组合能力。

### Figure 9: Humanoid robot hardware

![Figure 9](https://arxiv.org/html/2606.22174v1/x20.png)

### Figure 10: HuMI hardware

![Figure 10](https://arxiv.org/html/2606.22174v1/x21.png)

### Figure 11: Whole-body teleoperation scene and HMD snapshot

OpenHLM HTML 里 Figure 11 是组合 panel，主要展示 PICO4U HMD、handheld controllers、leg trackers 和 egocentric view。

### Figure 12: Per-task breakdown

![Figure 12](https://arxiv.org/html/2606.22174v1/x23.png)

### Figure 13: Scaling HuMI demonstrations

![Figure 13](https://arxiv.org/html/2606.22174v1/x24.png)

- HuMI demos per task 从 5 到 40，motion-reuse held-out tasks 的 average task progress 从 42% 到 84%。

### Tables

- Table 1: teleop method comparison。
- Table 2: long-horizon task progress，20 pairs 中 sampled 10 pairs，held-out lemon/tomato subset 有 7/10 pairs。
- Table 3: VLA training hyperparameters。
- Table 4: long-horizon task evaluation chosen pairs。

## 实验结果

### Teleoperation interface

Joint-based whole-body teleop 被选为默认接口。它比只部分暴露 humanoid DoF 的方法更适合收集全身行为。

### VLA adaptation

主要发现：

1. 单独改变 action projection init、action ordering、absolute/relative target、proprioception input，影响没有 pretraining 那么大。
2. 同时去掉 proprioception 和改 relative actions 会灾难性失败。
3. $\pi_{0.5}$ robot pretraining reaches 91% average task progress。
4. PaliGemma drops to 60%。
5. Random initialization collapses to 42%。
6. PaliGemma 和 $\pi_{0.5}$ 可能 action MSE 接近，但机器人表现明显不同。
7. one-step action generation 降低 latency，但实际 robot progress 低约 20 points。

### Data scaling

- 10 到 20 demos per task 收益最大。
- 40 demos per task 是有效默认值。
- 对 medium-difficulty task，40 demos 大约需要 skilled operator 1.5 小时。

### Heterogeneous co-training

- stationary teleop：更便宜，和 same embodiment 对齐，能提供新 motion 和新语义。
- HuMI：robot-free，更快，能提供新语义，但对缺失 motion 的 task 不够。
- 对 tasks 9-11 这类 motion-reuse held-out tasks，HuMI 可以接近 stationary。
- 对 Pouring 这种需要新 motion 的 task，HuMI 失败。

## 方法优点

- 工程问题拆得非常清楚。
- 不把 whole-body VLA 简化成 manipulation VLA 加腿。
- 明确指出 action MSE 是坏 proxy。
- 认真比较数据来源，而不是只说“多数据更好”。
- HLM-12 让系统至少面对多种能力族。
- 给出了可复用 recipe：teleop interface、VLA initialization、action generation、co-training。

## 局限与风险

- 目前 gripper 不是 dexterous hand，离高 DoF hand manipulation 还有距离。
- HLM-12 是否足够覆盖真实家庭/工业长尾任务，需要后续验证。
- 40 demos per task 虽然不多，但任务扩展到百级时仍是成本。
- $\pi_{0.5}$ 的选择可能让结论依赖特定 pretrained VLA。
- HuMI 对新 motion 无力，说明 robot-free demonstrations 不能被神化。

## 和相关工作的关系

- [[Vision-Language-Action Model]]：OpenHLM 的核心 policy 形式。
- [[PaliGemma]]：被用作 vision-language pretrained baseline。
- [[Pi05]]：robot-pretrained VLA backbone。
- [[HuMI]]：robot-free humanoid demonstration interface。
- [[UMI]]：HuMI 的 manipulation interface 参照。
- [[Flow Matching]]：action generation 的建模方式。
- [[Whole-Body Control]]：policy action 覆盖全身自由度。

## 对我的启发

- 做 humanoid VLA 时，不要只看 validation MSE，一定要看 closed-loop robot behavior。
- 非 humanoid robot pretraining 仍然有迁移价值，特别是“see error, correct, retry”的闭环操作习惯。
- cheap data 不是一种东西：stationary teleop 和 robot-free demos 提供的信息类型不同。
- 如果要把 CoorDex 这类 dexterous whole-body prior 接进 VLA，OpenHLM 的 interface ablation 是很好的系统模板。

## 精读优先级

高。它不像很多 foundation-model-for-robotics 论文只给概念图，而是告诉你哪些工程选择真的有影响、哪些只是噪声。
