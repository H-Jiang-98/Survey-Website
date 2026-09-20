---
title: "H2R: A Human-to-Robot Data Augmentation for Robot Pre-training from Videos"
method_name: "H2R"
authors: [Guangrun Li, Yaoxu Lyu, Zhuoyang Liu, Chengkai Hou, Yinda Xu, Jieyu Zhang, Shanghang Zhang]
year: 2025
venue: CVPR Workshop SynData4CV
tags: [egocentric-video, human-to-robot, visual-pretraining, data-augmentation, hand-retargeting, inpainting, cross-embodiment, representation-learning]
image_source: online
---

# H2R：把人手换成机器人，增强的是视觉表征而不是动作标签

> 本笔记基于 [arXiv:2505.11920v4](https://arxiv.org/abs/2505.11920)、[HTML 全文](https://arxiv.org/html/2505.11920)、[CVPR 2025 SynData4CV 记录](https://openreview.net/forum?id=meY9nInitM)及官方 [H2R-1M 数据集](https://huggingface.co/datasets/yaoxu789/H2R-1M)精读核验。Workshop 版本列 Yinda Xu，2026-03-16 更新的 arXiv v4 未列他；frontmatter 按正式 workshop 记录保留七位作者。

## 阅读结论先行

H2R 的贡献边界很清楚：它不是把人类视频变成可直接训练 policy 的 robot action trajectories，而是把 SSv2/Ego4D 中的人手视觉改造成 UR5、Franka、Robotiq gripper 或 LEAP Hand 的外观，再用 MAE/R3M 在这些 robotized frames 上预训练视觉 encoder。下游仍需要每种目标机器人 30–300 条 teleoperated demonstrations，动作标签全部来自机器人数据。准确表述应是 **human video reduces visual-representation pretraining cost**，而不是 human demonstration replaces robot demonstration。

Pipeline 的几何也比“retargeting”一词弱。HaMeR 恢复手关键点；手指关节角由三点夹角计算，gripper opening 由 fingertips 距离决定；但关键点不包含完整手臂，因此作者明确把其余机械臂关节 **手工设成合理值**。这保证单帧看起来像机器人在操作，却不恢复人类腕部的 metric 6-DoF trajectory、速度或 robot-feasible arm motion。H2R 输出是图像，不是 joint/action supervision。

视觉合成采用 SAM 移除手臂、LaMa 补背景，再从 simulator 渲染机器人并以 projected link/keypoint 做 pixel alignment。Camera pose 由 HaMeR 推断的 monocular hand/camera geometry 对齐，而非真实标定的世界坐标 extrinsics；论文未给 camera、keypoint、mask、temporal jitter 或 render alignment 的 ground-truth error。所谓“accurate/precise”主要由下游效果间接支持。

![H2R 官方 pipeline](https://arxiv.org/html/2505.11920v4/pipeline.png)

*图 1。H2R 把视觉 embodiment 变成机器人；流程并不产生下游控制 action。*

CLIP quality test 也不能证明 motion fidelity。作者随机取 1,000 对图像，用 Qwen2.5-VL 生成 action phrase；原图与 “A human is …” 得 28.01，合成图与 “A robotic arm is …” 得 29.83。但两边同时改变 image 和 prompt subject，CLIP 对可见 robot 与“robotic arm”措辞的偏好就足以提高分数；它没有比较同一 prompt、相邻帧 temporal consistency、人类评价或关键点误差。

实验的平均增益广泛但不等于逐任务稳定。SSv2 上，MAE 的 Robomimic/PushT/RLBench/CortexBench 平均分别提高 10.2/5.3/10.0/5.3pp，R3M 提高 6.3/7.0/5.0/1.3pp；但 R3M-Cortex Assembly 从 76 降至 68。Ego4D 上 MAE 的 RLBench 平均从 1.7 到 5.0，R3M 从 6.7 到 11.7；MAE ToiletSeatDown 反而从 5 降到 0。低 baseline 下的百分点增益需要和绝对成功率一起读。

Real-world 覆盖 gripper、dexterous hand 和 bimanual systems，是很有价值的外部验证。最大平均增益 23.3pp 出现在 ACT+R3M 的 Leaphand；但每任务只有 20 rollouts，最小分辨率 5pp，无置信区间、seed 或 significance test。Cross-embodiment 结果说明错配 robot render 仍可能优于纯人类预训练，但匹配 UR5 通常优于 Franka→UR5；它证明 partial visual transfer，不证明 embodiment-independent representation。

组件 ablation 更有说服力：Leaphand 上去掉 overlay，DP/ACT 平均下降 30.0/18.3pp；把几何 retarget 改为 random pasting，下降 16.7/16.7pp。因此 robot appearance 和 spatial alignment 都有效。不过“w/o Retarget”不是保持其他条件的 no-retarget，而是故意随机贴图，主要测试位置错误的破坏性。

Lighting shift 暴露其局限：MAE 平均 11.7→H2R 11.7，R3M 5.0→8.3；加入专门 lighting augmentation 才到 28.3/13.3。H2R 修复的是 hand/robot appearance gap，不自动解决 illumination、background、object、camera 或 dynamics domain shift。

Artifact 情况优于只发论文：Hugging Face 已公开十个 subsets，共 3,379,776 rows、637GB，包含 SSv2/Ego4D 原图、三种 robot variants 与 without-retarget controls。但页面未提供生成代码/checkpoint，license 为 CC BY-NC-ND-4.0，商业使用和衍生再分发受限，且仍需遵守源数据条款。因此数据可下载，完整 pipeline 暂不可复现。

### 一句话总结

H2R 证明，把大规模 Ego 视频中的人手替换为几何对齐的目标 robot render，能让 MAE/R3M 学到更适合 robot policy 的视觉特征；它降低的是 representation gap，而非动作采集成本，不能和 Phantom/Ego2Robot 那类生成 observation-action demonstrations 的方法混为一谈。

## 1. 方法拆解

~~~mermaid
flowchart LR
    V["SSv2 / Ego4D frames"] --> H["HaMeR hand keypoints"]
    H --> J["finger angles / gripper opening"]
    J --> S["sim robot; other arm joints manually set"]
    H --> C["camera-hand frame alignment"]
    V --> M["SAM arm mask"]
    M --> I["LaMa inpainting"]
    S --> R["robot render"]
    C --> R
    I --> O["robotized frames"]
    R --> O
    O --> P["MAE / R3M pretraining"]
    D["30–300 target-robot demos"] --> Q["DP / ACT / UVA policy"]
    P --> Q
~~~

设 human-hand frame 和 simulator frame 的旋转分别为 ${}^{W}_{H}R$、${}^{W}_{S}R$，论文将 camera 位置映射为：

$$
{}^{W}\mathbf{cam}_{sim}
= {}^{W}_{S}\mathbf R
  \left({}^{W}_{H}\mathbf R\right)^{-1}
  {}^{W}\mathbf{cam}_{real}.
$$

这只处理 coordinate-frame orientation；单目 scale、translation bias、occlusion、rolling shutter 和 frame-to-frame HaMeR fluctuation 并未由公式消失。手指三点 $x_{i-1},x_i,x_{i+1}$ 的 angle 可写为：

$$
\theta_i=\arccos\frac{(x_{i-1}-x_i)^\top(x_{i+1}-x_i)}
{\lVert x_{i-1}-x_i\rVert_2\,\lVert x_{i+1}-x_i\rVert_2},
$$

但这是 visual articulation label；policy 训练没有使用这些 $	heta_i$。

## 2. 结果：平均值、负迁移与数据依赖

| 设置 | 原始 human pretrain | H2R | $\Delta$ | 应如何解释 |
| --- | ---: | ---: | ---: | --- |
| MAE / Robomimic avg | 58.0 | 68.2 | +10.2 | 强平均增益 |
| R3M / Cortex avg | 73.3 | 74.7 | +1.3 | Assembly -8pp 被别的任务抵消 |
| MAE / Ego4D RLBench avg | 1.7 | 5.0 | +3.3 | absolute performance 很低 |
| DP+MAE / real Leaphand | 40.0 | 60.0 | +20.0 | 50 demos，20 rollouts/task |
| ACT+R3M / real Leaphand | 15.0 | 38.3 | +23.3 | 最大报告平均增益 |
| DP+MAE / real Franka | 21.7 | 25.0 | +3.3 | 双臂收益较小 |

H2R 与 DROID 的比较也需谨慎：R3M-H2R 在 Robomimic 平均 61.3，R3M-DROID 56.7，但两者的源数据规模、distribution 和 preprocessing 并未做 equal-data control；DROID 的 Lift 为 96，反而高于 H2R 的 85。不能据此推出 synthetic human video 普遍优于真实 robot data。

UVA 表中 0.20→0.35 是 proportion，而表题写百分比，实际是 20%→35%。它只 fine-tune UVA 的 VAE encoder/decoder 后冻结，再训练两项 UR5e policy；这说明 H2R 可作为 VLA visual-backbone adaptation data，不等于在 H2R 视频上训练 language-conditioned action tokens。

## 3. H2R、Phantom 与真正 data conversion 的区别

| 方法 | Human video 提供 | 训练 policy 是否仍需 robot demos | Test-time 特殊处理 |
| --- | --- | --- | --- |
| H2R | robot-looking visual pretraining frames | **需要**，30–300 条/setup | 无 H2R overlay |
| Phantom | edited frames + gripper SE(3)/opening labels | **不需要**目标 robot demos | 需要 virtual robot overlay |
| Ego2Robot 类路线 | robotized observations + retargeted feasible actions | 目标是减少/替代 | 依方法而定 |

H2R 的优势是下游部署简单、可扩展到百万级 frames；代价是它没有解决 action-space、contact、feasibility、control 或 task correspondence。它与 Phantom 不是谁“更强”，而是处在 pipeline 不同层级。

## 4. 最关键的后续实验

- 同一 text prompt 比较 original/inpaint/random-overlay/retargeted images，并加入 human preference、keypoint reprojection 与 temporal jitter metrics。
- Equal-compute、equal-frame 的 human-only、H2R-only、human+H2R、robot-only、CutMix/random-paste controls；报告多 seeds。
- 下游 robot-demo scaling curve（1/5/10/30/100%），直接回答 H2R 究竟节省多少机器人采集。
- 对 mask error、camera bias、render texture、arm-joint choice 和 embodiment mismatch 做系统 sweep。
- 公开生成代码、models、exact subsets/splits 和训练 checkpoints；把 source-data/derived-data license 链说明清楚。

最终判断：H2R 是结构化视觉 data augmentation 的扎实实证，尤其证明了“贴什么 robot、贴在哪里”都会影响 representation transfer；但论文标题中的 human-to-robot 只发生在 pixels，尚未发生在 executable actions。
