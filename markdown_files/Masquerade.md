---
title: "Masquerade: Learning from In-the-wild Human Videos using Data-Editing"
method_name: "Masquerade"
authors: [Marion Lepert, Jiaying Fang, Jeannette Bohg]
year: 2026
venue: ICRA
tags: [in-the-wild-video, egocentric-video, data-editing, bimanual-manipulation, visual-pretraining, co-training, diffusion-policy, cross-embodiment]
image_source: online
---

# Masquerade：675K 个 robotized frames 如何把 OOD 成绩从 12% 推到 74%

> 本笔记基于 [ICRA 2026 项目页](https://masquerade-robot.github.io/)、[arXiv:2508.09976](https://arxiv.org/abs/2508.09976)、[HTML 全文](https://arxiv.org/html/2508.09976)和 [MIT 处理代码](https://github.com/MarionLepert/phantom)精读核验。

## 阅读结论先行

Masquerade 是 Phantom 向 in-the-wild RGB 数据扩展后的不同解法。它处理 10K 段 EPIC-Kitchens clips（675,713 frames）：HaMeR 估双手，Detectron2+SAM2 分割手臂，E2FGVI inpaint，再叠加 dual-Kinova render。由于 monocular RGB 没有可靠 absolute depth，作者没有把恢复的 3D pose 当 robot action；human videos只监督未来 $H$ 帧的 **2D end-effector waypoints**。真正 3D Cartesian policy actions仍来自每任务 50 条 Oculus teleoperated robot demonstrations。

这一区分很重要。论文偶尔称 edited clips 为 synthetic robot demonstrations，但它们不是可直接 behavior clone 的 executable demonstrations；它们同时承担两件事：预训练 language-conditioned ViT 预测 2D motion，robot policy training 时继续提供同一 auxiliary loss，防止 encoder 遗忘。Diffusion Policy action head从未在人类伪 3D action上训练。

![Masquerade 官方 pipeline](https://masquerade-robot.github.io/static/images/method.png)

*图 1。Human branch只进入 auxiliary 2D loss；robot branch才提供 BC/action supervision。*

主要结果很强：dual Kinova 在 Stack Pots、Scrape Potato、Sweep Chilis 三个 long-horizon bimanual tasks上，每个 task 只在一个 scene收50 demos，再到三个 unseen scenes各跑10次。项目页汇总最佳 baseline 12%、Masquerade 74%，提高62pp/约6.2倍，而且每个测试 scene都领先。相比只在同场景报告 success，这确实直指环境泛化。

但“success rate”实际允许 partial credit：每任务三个 subtasks，各占 $1/3$。74%可能混合完整完成与完成前两步，并非74%的 episodes完全完成。总计每 method/task 30 rollouts，scene-level只有10次；虽画 ±SEM，论文没有多训练 seeds或置信区间。三个任务都来自厨房、同一 dual-Kinova/parallel-jaw setup，不能外推到 general robot policy。

最有说服力的不是横向 baseline，而是 controls。Raw-human no-overlay和no-cotraining在同一 OOD scene大幅下降，说明 visual embodiment alignment 和持续 human auxiliary training 缺一不可。Stack Pots 的 edited-data scaling为 0/10/50/100% → 2/26/47/68%，且训练 epochs相同；这是罕见的单调 data scaling evidence。不过只测一个task、一个scene、每点25 rollouts，而且更多 data 也意味着更多 optimizer samples/compute，尚不能分开 diversity 与 computation。

“5–6× prior work”需要注意比较对象。HRP使用150K human clips但不同 proxy labels与公开 checkpoint；ImageNet和DINOv2使用完全不同规模/数据。所有backbone都是 ViT-B，架构公平，但 training objective、pretraining compute和持续 cotraining不公平。Masquerade ablation能证明整套 recipe有效，却没有 factorial control分离 robot overlay、2D future-motion objective、language FiLM、inpainting artifact与 extra cotraining compute各自贡献。

Monocular editing仍很粗糙。Repo明确称 depth方向误差约3–4cm；没有scene depth时 robot pixels可能错误盖住前景物体。Camera motion用 homography补偿，并丢弃单步 translation >5cm 或 rotation >0.5rad 的frames；这会把“in-the-wild”数据偏向静止头部、平面近似有效的片段。若一只手中途消失，后续一直复用最后action；整段不可见则给固定 out-of-frame label。这些 heuristics 可规模化，却会产生 frozen-hand supervision和错误双手协调。

Policy train时又在真实 robot图像上 inpaint+render robot，使 human和robot branch都看同类 synthetic embodiment。部署是否也做 overlay，论文描述不够明确；method只说 cotraining preprocessing，项目视频显示真实robot。若 inference看raw robot，则仍存在 train–test rendering gap；若 inference也编辑，则需要在线 segmentation/rendering。论文应明确并做 on/off ablation。

Artifact audit：MIT repo提供 Phantom/Masquerade共享的 bbox、2D/3D hand、segmentation、smoothing、inpainting和robot overlay pipeline以及一段EPIC sample；支持 Panda/Kinova/UR5e/IIWA/Jaco。它只有2 commits，需注册下载 MANO；未见完整675K edited dataset、ViT co-training/policy训练代码、checkpoints、robot demos或evaluation scripts。因此可以复现 sample editing，不能复现74%端到端结果。

### 一句话总结

Masquerade最可靠的发现不是 monocular video能提供精确robot actions，而是：把粗糙但位置相关的robot appearance叠到多样化Ego视频上，并在policy训练中持续预测未来2D motion，可以显著改善少量机器人示范下的scene OOD泛化；它仍依赖目标机器人action demos，且证据集中于三个厨房双臂任务。

## 1. 从人类视频到 auxiliary motion supervision

Human dataset $\mathcal D_h$ 输出当前frame坐标系中的未来2D轨迹：

$$
\mathbf p_{t:t+H}^{2D}
=\left(\mathbf p_t^{2D},\mathbf p_{t+1\rightarrow t}^{2D},\ldots,
\mathbf p_{t+H\rightarrow t}^{2D}\right),
$$

其中 future points用homography warp回第$t$帧。Encoder $f$由clip语言embedding $z_x$经FiLM调制，MLP $h$回归：

$$
\mathcal L_{2D}
=\left\lVert h(f(x,z_x))-\mathbf p_{t:t+H}^{2D}\right\rVert_2^2,
\qquad x\sim\mathcal D_h'.
$$

Robot branch用真实Cartesian action $\mathbf P^{(r)}$训练policy head $g$；正文将其简写成平方损失，实际Diffusion Policy应优化加噪action的denoising objective：

$$
\mathcal L=\mathcal L_{2D}+\lambda\mathcal L_{policy},
\qquad \lambda=10.
$$

因此 transfer interface 是 shared visual encoder，而不是 shared action space。

~~~mermaid
flowchart TB
    E["EPIC RGB clips"] --> P["hand pose + camera-motion filtering"]
    P --> X["inpaint arms + bimanual render"]
    X --> A["ViT: future 2D waypoint loss"]
    R["50 target-robot demos/task"] --> B["ViT + Diffusion Policy: 3D action loss"]
    A --> C["shared encoder cotraining"]
    B --> C
    C --> O["single-task policy in unseen scenes"]
~~~

## 2. Evidence matrix

| 主张 | 实验 | 结论强度 |
| --- | --- | --- |
| Edited human data改善OOD | 12%→74%，3 tasks × 3 scenes | **强**，但partial score |
| Robot overlay必要 | raw human vs edited，OOD scene 1 | **强** |
| Cοtraining必要 | pretrain→robot-only finetune显著下降 | **强**，也包含extra compute |
| 更多human data更好 | 2→26→47→68%，Stack Pots | **中强**，单task/scene |
| 可从human直接学robot action | human只给2D aux targets | **不成立** |
| General bimanual learning | 三个厨房任务、单一hardware | **证据不足** |

## 3. 与 H2R / Phantom 的准确定位

| 方法 | 输入 | Human supervision | Robot demos |
| --- | --- | --- | ---: |
| H2R | RGB Ego frames | MAE/R3M visual pretraining | 30–300/setup |
| Masquerade | in-the-wild monocular RGB | language-conditioned future 2D waypoints；持续cotraining | 50/task |
| Phantom | curated RGB-D | full gripper SE(3)+opening，直接policy labels | 0 |

Masquerade牺牲metric action accuracy换来数据规模和环境多样性；Phantom牺牲数据开放性/规模换来可执行3D labels。两者共同说明视觉robotization有价值，但解决的是不同数据 regime。

## 4. 最需要补的实验

- 报告episode-level full completion，同时保留partial progress；给每task/scene counts和bootstrap CI。
- 2D aux + raw human、overlay + generic MAE、inpaint-only、random/spatially-correct overlay的完整factorial ablation。
- Equal-steps/equal-frames scaling，并按task relevance、camera motion、overlay quality分层。
- Robot-demo scaling（5/10/25/50）、多训练seed和新hardware/viewpoint。
- 明确test-time preprocessing，公开policy/cotraining code、weights、splits、filtered-frame比例及licenses。

最终判断：Masquerade是迄今“visual editing + continued auxiliary learning”最强的OOD scene证据之一；但把它称作robotized demonstration容易误导，它生成的是有机器人外观的 2D motion pretraining data，而不是能替代teleoperation的robot trajectory dataset。
