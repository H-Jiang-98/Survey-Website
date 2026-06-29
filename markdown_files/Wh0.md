---
title: "Wh0: Generative World Models as Scalable Sources of Egocentric Human Hand Manipulation Data"
method_name: "Wh0"
authors: [Yangtao Chen]
year: 2026
venue: arXiv
tags: [egocentric-perception, dexterous-manipulation, world-model, vision-language-action, data-generation]
zotero_collection: _inbox
image_source: online
created: 2026-06-25
---

# Wh0

## 一句话结论

[[Wh0]] 的核心想法是用生成式 [[World Model|世界模型]] 批量生成 egocentric human-hand manipulation videos，再把这些视频通过 hand reconstruction、embodiment alignment 和 VLA post-training 转成机器人可学的数据。它真正重要的地方不是“生成视频”，而是把世界模型输出变成了 [[Dexterous Manipulation|灵巧操作]] policy 的训练燃料。

## 论文定位

- 任务：real-world dexterous manipulation with egocentric perception。
- 数据问题：teleoperation 对齐但贵，simulation 可扩展但 sim-to-real gap 大，人类 egocentric videos 规模大但没有机器人动作且 embodiment 不对齐。
- 方案：世界模型合成 [[WM-H]]，再对齐场景、手部 embodiment 和动作空间。
- policy：基于 VITRA-style [[Vision-Language-Action Model|VLA]]，使用 [[PaliGemma]] backbone 和 diffusion action decoder。
- 实验：真实 Unitree G1 + Inspire hands + head-mounted egocentric camera，测试 unseen tasks、objects、environments、instructions。

## 背景问题

Dexterous manipulation 的数据瓶颈比普通夹爪 manipulation 更严重。

1. 灵巧手高 DoF，动作标注贵。
2. 接触和遮挡多，纯视觉 imitation 很容易漏掉关键物理状态。
3. 人类视频有大量 hand-object interaction，但没有机器人 action labels。
4. 生成视频模型可以扩规模，但会带来物理幻觉、时序不一致和 embodiment mismatch。

Wh0 的主张是：不要直接拿生成视频当真相，而是把它作为可控、可筛选、可对齐的中间数据源。

## 方法总览

### Step 1: WM-H data synthesis

世界模型根据 language、object 和 scene condition 生成 egocentric human-object interaction videos。

输出是 [[WM-H]]：50k episode 级别的 synthetic egocentric manipulation data。

### Step 2: Scene alignment

生成视频必须和部署工作区对齐，否则视频里的手和物体会漂离目标场景。

### Step 3: Embodiment alignment

使用 image editing / robot-hand appearance editing 把 human hand appearance 向机器人 hand 靠近，同时尽量保留 pose 和 motion。

### Step 4: Action supervision conversion

通过 hand motion reconstruction、retargeting 和 human-robot action alignment，把视频转成 robot-trainable action supervision。

### Step 5: VLA co-training

在 VITRA-style VLA 上，用 limited teleop data 和 WM-H 混合训练。

## 关键公式

### 1. 102D hand action space

论文将 action 定义在当前 observation 的 camera coordinate frame。

$$
a_t
=
[
\Delta t^l,
\Delta r^l,
\theta_h^l,
\Delta t^r,
\Delta r^r,
\theta_h^r
]
\in \mathbb{R}^{102}.
$$

符号：

- $\Delta t$ 是 wrist relative translation。
- $\Delta r$ 是 wrist relative rotation。
- $\theta_h\in\mathbb{R}^{15\times 3}$ 是 [[MANO]] hand 的 15-DoF joint rotations。
- $l,r$ 分别表示左右手。
- 论文重点关注 right-hand manipulation。

这个 action definition 很关键：它让 robot joints retarget 到 MANO-like human action space，从而复用大规模 human video pretraining 的 normalization。

### 2. Diffusion action decoder objective

VLA 使用 diffusion-based action decoder，训练目标是 noise prediction MSE。

$$
\mathcal{L}_{\mathrm{MSE}}
=
\mathbb{E}_{\epsilon\sim\mathcal{N}(0,1), i}
\left[
\left\|
\hat{\epsilon}_{i}
-\epsilon
\right\|_{2}^{2}
\right].
$$

符号：

- $\epsilon$ 是扩散噪声。
- $\hat{\epsilon}_i$ 是第 $i$ 个 diffusion step 的预测噪声。
- decoder 由视觉语言特征和当前 hand state 条件化。

### 3. Data mixture view

Wh0 的 post-training 数据混合可以抽象为：

$$
\mathcal{D}_{post}
=
\lambda_R \mathcal{D}_{robot}
+\lambda_W \mathcal{D}_{WM-H}
+\lambda_{EA} \mathcal{D}_{WM-H}^{embodiment-aligned}.
$$

论文图示指出 Wh0 使用约 28% teleop 和 68% WM-H，并且由于 teleop 只有约 400 samples，而 WM-H 有 50k samples，实际 per-sample 采样上对 robot data 做了强 oversampling。

含义：

- robot data 提供部署对齐。
- WM-H 提供规模和多样性。
- embodiment-aligned WM-H 降低 human hand 到 robot hand 的外观/动作差距。

## 关键图表

### Figure 1: Wh0 overview

![Figure 1](https://arxiv.org/html/2606.22136v2/x1.png)

- Top：WM-H 的生成式 egocentric manipulation videos。
- Middle：用 world-model-generated data 和 limited robot data co-training。
- Bottom：zero-shot generalization 到真实任务、环境和指令。

### Figure 2: WM-H data synthesis pipeline

![Figure 2](https://arxiv.org/html/2606.22136v2/x2.png)

- 展示 instruction generation、scene-aligned image editing、image-to-video generation、embodiment-aligned editing。

### Figure 3: Policy architecture and data composition

![Figure 3](https://arxiv.org/html/2606.22136v2/x3.png)

- VITRA-style policy。
- PaliGemma backbone。
- diffusion action decoder。
- post-training mixture：teleop + WM-H。

### Figure 4: Real-world evaluation setup

![Figure 4](https://arxiv.org/html/2606.22136v2/x4.png)

- Unitree G1。
- Inspire hands。
- head-mounted egocentric camera。
- Vision Pro teleop。
- seen/unseen objects 和 seen/unseen backgrounds。

### Figure 5: Zero-shot real-world rollouts

![Figure 5](https://arxiv.org/html/2606.22136v2/x5.png)

- container-aware placement。
- small-object grasping。
- tool use。
- 这些 object/container/task combinations 不在训练集中。

### Figure 6: Scene and embodiment alignment

![Figure 6](https://arxiv.org/html/2606.22136v2/x6.png)

- 没有 scene alignment 时生成视频会漂离目标 workspace。
- embodiment alignment 把选定 frame 编辑为 robot hand，同时保持 pose/motion。
- 右侧用 action-feature cosine similarity 分析 appearance edit 影响。

### Figure 7: WM-H failure cases

![Figure 7](https://arxiv.org/html/2606.22136v2/x7.png)

- image editing errors。
- physically implausible hand-object interactions。
- temporal inconsistencies。
- instruction misalignment。
- imperfect robot-hand embodiment alignment。

这张图很重要，因为它承认 synthetic data 的脏东西，而不是只展示好看的生成样例。

### Figure 8: WM-H qualitative visualization

![Figure 8](https://arxiv.org/html/2606.22136v2/x8.png)

### Figure 9: Robot execution rollouts

![Figure 9](https://arxiv.org/html/2606.22136v2/x9.png)

### Figure 10: Additional rollouts and VITRA comparison

![Figure 10](https://arxiv.org/html/2606.22136v2/x10.png)

### Tables

- Table 1: real-world dexterous manipulation performance。
- Table 2: deployment alignment 和 WM-H scale 的 ablation，包含 robot-object grounding 与 task success，HO 表示 hand-object distance。
- Table 3: human-video pretraining、teleop demos、WM-H co-training 的组合消融。
- Table 4: user study on hand-manipulation videos，$N=72$。
- Table 5: model/ablation training dataset composition，$R$ 表示 teleop robot data，$W$ 表示 WM-H，$W$-EA 表示 embodiment-aligned WM-H。

## 实验结果

### Real-world dexterous manipulation

论文在真实机器人上测试 seen/unseen objects、backgrounds 和 instructions。Figure 4-5 显示它不是只做离线视频评估，而是让 policy 上 Unitree G1 + Inspire hands。

### Ablation: alignment matters

Figure 6 和 Table 2 支撑两个判断：

1. scene alignment 不只是美观问题，会影响视频是否还在目标 workspace。
2. embodiment alignment 影响 robot-object grounding 和 task success。

### Ablation: scale matters

WM-H scale 对结果有帮助，但不是越大越无脑。生成数据的质量、场景对齐和 embodiment 对齐共同决定是否能转成 policy gain。

### Ablation: pretraining priors

Table 3 的重点是 human-video pretraining、teleop demonstrations、WM-H co-training 三者组合最强。单靠 synthetic world-model data 不能替代真实 robot data；单靠 teleop data 又不够规模。

### Failure cases

Figure 7 是这篇最诚实的部分之一。它明确列出生成式数据的失败模式：

- hand-object contact 不物理。
- 时间一致性差。
- 指令和动作不匹配。
- robot-hand embodiment edit 不完美。

这些失败模式会直接污染 policy training，所以数据筛选和 alignment 不是附属模块，而是核心模块。

## 方法优点

- 把 world model 变成 robot data source，而不是只做视频演示。
- 正面处理 scene gap、embodiment gap 和 action label gap。
- 使用真实机器人 zero-shot 评估。
- 承认 synthetic data failure cases。
- 数据路线和 VLA policy 结合紧密，适合 egocentric dexterous manipulation。

## 局限与风险

- 生成视频的物理接触真实性仍然是最大风险。
- 102D MANO-like action space 对复杂 robot hand dynamics 可能仍然不够。
- embodiment alignment 如果只改 appearance，不一定修正真实 kinematic/dynamic mismatch。
- WM-H 的数据质量筛选成本可能被低估。
- 目前重点是 right-hand manipulation，双手协作、强力交互和 deformable objects 还不能直接外推。

## 和相关工作的关系

- [[VITRA]]：policy pretraining 参考。
- [[PaliGemma]]：vision-language backbone。
- [[Ego4D]]、[[EgoExo4D]]、[[EPIC-KITCHENS]]、[[Something-Something-V2]]：egocentric/human-video pretraining sources。
- [[MANO]]：hand representation 和 action space。
- [[Diffusion Policy]]：动作生成 decoder 的思想相近。
- [[World Model]]：生成式数据源。

## 对我的启发

- Egocentric human video 的价值不在“像人”，而在提供 hand-object interaction prior。
- 生成式 world model 要进入 robotics，必须过 scene alignment、embodiment alignment、action extraction 三道门。
- Synthetic data 不是 teleop data 的替代品，而是扩展器。
- 如果要做 humanoid dexterous manipulation，Wh0 可以和 OpenHLM/CoorDex 互补：OpenHLM 给 whole-body VLA 系统，CoorDex 给 body-hand prior，Wh0 给 egocentric hand data scaling。

## 精读优先级

高。它的设定非常贴近 egocentric perception + dexterous manipulation + data scaling 这条线，而且有真实机器人评估和失败案例，不是只会生成漂亮视频。

