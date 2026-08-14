---
title: "HumanEgo: Zero-Shot Robot Learning from Minutes of Human Egocentric Videos"
method_name: "HumanEgo"
authors: [Zhi Wang, Botao He, Kelin Yu, Seungjae Lee, Ruohan Gao, Furong Huang, Yiannis Aloimonos]
year: 2026
venue: arXiv
tags: [human-video, egocentric-video, imitation-learning, flow-matching, bimanual-manipulation, interaction-centric-token, cross-embodiment, zero-robot-data]
image_source: online
---

# HumanEgo：30 分钟人类第一视角视频，真的足以零样本控制任意机器人吗？

> 本笔记基于 [arXiv:2605.24934v2](https://arxiv.org/abs/2605.24934)、[HTML 全文](https://arxiv.org/html/2605.24934)、[项目页](https://humanego-ai.github.io/)、[代码库](https://github.com/TX-Leo/HumanEgo)、[数据集](https://huggingface.co/datasets/Leo-TX/HumanEgo)与[模型权重](https://huggingface.co/Leo-TX/HumanEgo)核验。论文尚未标注会议接收，故 venue 记为 arXiv。

## 阅读结论先行

HumanEgo 的关键不是把原始人类 RGB 直接喂给 policy，而是用 Project Aria Gen 1 的双 SLAM 相机与云端 MPS 恢复 metric camera/hand pose，再用 Grounding DINO、SAM2、CoTracker3、多视角三角化与 Orient Anything V2 得到物体 6-DoF；最后把手、物体及其相对关系压成 Interaction-Centric Tokens（ICT）。Policy 学的是显式 3D interaction state 到双臂 SE(3) action chunk 的映射，RGB 更多提供未被 token 覆盖的视觉上下文。

因此，“只需 raw video”“anyone, anywhere”是传播口径，不是实验接口。每段采集前要扫视场景 1–2 秒以支持三角化，必须有高精度 SLAM/手追踪，离线还要经过多套视觉模型、手臂擦除和物体姿态估计。单目手追踪替代 Aria MPS 时，Water Flowers 从 95% 跌到 WiLoR 45%、HaMeR 32.5%、MediaPipe 0%；作者测到 5–11 cm 系统性深度偏差。这证明上游 metric pose 是系统成立的硬条件，而非可有可无的工程细节。

四个实机任务、每任务 40 trials 上，30 分钟人类数据平均成功率 92.5%，15 分钟仍为 75%；30 分钟 robot teleoperation 的 ACT 为 51.2%。项目页称“outperforms ... by 41%”，从 92.5−51.2 看实际是 **+41.3 percentage points**，不是相对提升 41%（相对为约 80.7%）。人类数据效率的方向有力，但 baseline 同时改变了数据来源、policy（ACT vs flow matching）、显式 ICT 与相机接口，不能把全部差距单独归因于“人类示范优于 robot teleop”。

最有说服力的因果证据是 representation ablation：Water Flowers 上 raw human RGB 仅 7.5%，inpaint/keypoint 20%，robot-rendered RGB 32.5%；raw human RGB + ICT 直接到 85%，完整方法 95%。视觉本体对齐有帮助，但显式 hand–object 3D relation 才是主要增益。15 分钟数据下，object-motion、latent-consistency、2D-trace 单项分别贡献 +17.5/+12.5/+5 pp，三者合用 +25 pp；dense supervision 主要在低数据区有效，超过约 18 分钟后差距收敛。

“hardware-agnostic”也应缩窄理解：同一 SE(3) end-effector action 表示在 Trossen、Franka、UR10 与 RealSense/ZED 上迁移，无需目标 robot demonstration；但每个平台仍需相机标定、物体在线 6-DoF perception、robot state 到 ICT 的转换、IK/低层 controller 与 gripper integration。这是 **demonstration-agnostic across tested hardware**，不是无集成的一键部署。

论文在 reference-frame 叙述上存在实质张力。项目 FAQ 称 ICT 全部是 entity-relative、因而对 viewpoint invariant；正文定义和 appendix 却表明主实验使用 camera frame，anchor-object frame 只是变体，而且 camera-frame policy 在 camera reposition 时显著退化。故跨相机结果支持的是经过相应预处理和有限测试条件的 transfer，不能推出表示天然对任意 viewpoint 不变。

数据量还有未解释的不一致：摘要/主结果/项目页均写每任务 30 分钟；hyperparameter table 却写每任务 60 demos、总计约 40 分钟，augmentation 描述也按约 40 分钟；另一个 ablation 又把 45 demos 记为 30 分钟。合理可能是“采集 40、筛选/训练用 30”，但论文没有明确说明，读者不应把 92.5% 简化为端到端恰好 30 分钟总成本。

公开性在论文发布后已有进展：代码库包含 collection、preprocess、training 与配置，公开数据有 122 段、两个任务（Serve Bread、Water Flowers）的 raw Aria、MPS annotation 和 preprocessing output，并有 Serve Bread checkpoint。可是四个主任务中另两个数据未公开，repo README 仍把 inference/preprocess documentation 和 sample quick start列为 TODO。因此目前可检查并部分复跑 pipeline，尚不足以独立复现四任务平均 92.5%及完整 cross-hardware 表。

### 一句话总结

HumanEgo 令人信服地说明：在**任务定向采集、高精度 metric SLAM/hand pose、可靠 object 6-DoF 与显式 interaction token**齐备时，分钟级人类视频可以替代目标机器人示范，训练出高成功率的闭环双臂策略；它没有证明普通单目视频可直接用，也没有消除 perception、calibration、控制集成或目标机器人安全验证。

![HumanEgo 数据采集与 Project Aria setup](https://humanego-ai.github.io/static/images/data_collection.png)

*图 1。官方采集设置。轻量眼镜降低示范负担，但 metric supervision 来自 Aria 多传感器与 MPS，而非 RGB 单流本身。*

## 1. 问题定义：零 robot data 的边界

人类轨迹先转换为结构化 observation-action pairs：

$$
\tau_h=\{I_t,\,\mathbf T_t^{C},\,\mathbf T_t^{LH},\,
\mathbf T_t^{RH},\,\{\mathbf T_t^{O_j}\}_{j=1}^{N}\}_{t=1}^{T},
$$

目标 action 是长度 $K=50$ 的双手 chunk：

$$
\mathbf A_t=\{(\Delta\mathbf p,\mathbf R^{6D},g)_{t+k}^{L,R}\}_{k=1}^{K}.
$$

Human hand pose 经 wrist/palm geometry 映射成 parallel-jaw gripper pose；抓住物体后以 kinematic latching 将 object pose 绑定于 hand，缓解遮挡。它给出可训练 label，却也把 tracking error和“人手可做但机械臂不可达/碰撞”的轨迹直接传给 policy。论文没有 action-label mocap error、IK rejection、collision或controller tracking统计。

“robot-data-free”严格指每任务 policy 不用 robot teleoperation/rollout training data；目标机器人 URDF/kinematics、camera calibration、控制器和 40-trial evaluation 仍不可缺。

## 2. Interaction-Centric Token 到底编码什么

每个 entity token 为 29 维：

$$
z_E=[\tau_E,\,\phi({}^{REF}\mathbf T_E),\,
\phi({}^{E}\mathbf T_{LH}),\,\phi({}^{E}\mathbf T_{RH}),\,g],
$$

其中 $\phi(\mathbf T)\in\mathbb R^9$ 是 3D normalized translation 加 6D rotation。Entity type、entity pose、左右手相对于它的 pose与 grasp state放在统一 token中。Variable-length tokens经 encoder与 RGB feature融合。

关键歧义在 ${REF}$：若为 camera frame，global camera movement会改变 token；若为 anchor object，才更接近项目页声称的 viewpoint invariance。Appendix 的 camera reposition ablation显示 anchor-object frame在小数据与移机时更稳，反而证明主设置并非先验 invariant。

## 3. Policy 与 dense auxiliary supervision

Flow matching在噪声 action $\mathbf A^0\sim\mathcal N(0,I)$ 和真实 chunk $\mathbf A^1$间学习速度场：

$$
\mathbf A^s=(1-s)\mathbf A^0+s\mathbf A^1,
\qquad
\mathcal L_{FM}=\mathbb E\|v_\theta(\mathbf A^s,s,o_t)-(\mathbf A^1-\mathbf A^0)\|_2^2.
$$

训练再加三项未来预测：

$$
\mathcal L=\mathcal L_{FM}+\lambda_o\mathcal L_{obj}
+\lambda_{2d}\mathcal L_{trace}+\lambda_z\mathcal L_{latent}.
$$

它们分别预测 object 6-DoF motion、hand/object image trace 和未来 latent。推理用 20 个 Euler steps；policy 10 Hz replan、5 Hz执行并带 look-ahead。Auxiliary targets都来自同一 perception pipeline，所以“免费标签”只表示无需人工标注，不表示没有计算成本或 label noise。

~~~mermaid
flowchart LR
    H["Aria RGB + stereo/IMU"] --> M["MPS SLAM + metric hands"]
    H --> O["Detection / SAM2 / CoTracker3 / triangulation"]
    M --> I["Arm inpainting + virtual gripper"]
    O --> T["Object 6-DoF"]
    M --> T
    T --> Z["29-D ICTs"]
    I --> P["Flow-matching policy"]
    Z --> P
    P --> A["Bimanual SE(3) chunks"]
    A --> C["Target-specific IK/controller"]
~~~

## 4. 结果怎样读才不过度外推

| 证据 | 数值 | 能支持什么 | 不能支持什么 |
| --- | ---: | --- | --- |
| 4 tasks，30 min human/task | 平均 92.5%，40 trials/task | task-specific human-only policy可高成功 | 任意任务、长时序、动态操作 |
| 15 min human vs 30 min ACT | 75 vs 51.2 | 在这套系统里 human route更省采集时间 | 数据来源是唯一因果变量 |
| Serve Bread scaling | 7 min 50；8 min 57.5；30 min 95 | 分钟级已有学习信号 | 所有任务都同样 sample-efficient |
| 2 tasks × 9 OOD conditions | 约 85–91.25 | 对所测 object/setup/scene有韧性 | “arbitrary”环境与物体 |
| Cross-robot/camera | Trossen/Franka/UR10；RealSense/ZED | SE(3)+ICT接口可跨三臂两相机 | 无 calibration/controller integration |

每个主任务仅一个最终 training run与 40 次 binary trials，单格分辨率 2.5 pp；没有多 seed、置信区间或长期失效率。OOD虽覆盖九类条件，仍是作者选定的有限变化，不等同开放世界。

## 5. 最关键的对照与限制

| Water Flowers representation | Success (%) |
| --- | ---: |
| Raw human RGB | 7.5 |
| Human keypoints + inpainting | 20.0 |
| Robot-rendered RGB | 32.5 |
| Raw human RGB + ICT | 85.0 |
| Full HumanEgo | 95.0 |

这个表推翻了“高保真 robotization本身解决 embodiment gap”的解释，却也暴露 test-time依赖：ICT要求知道 task entities及其 pose。论文的 object module是逐帧 detector/pose pipeline，不是经大范围遮挡与动态接触验证的通用在线 tracker。约 1 cm 精细操作、in-hand manipulation、柔性/透明物体、快速 dynamics 和 tactile/force控制仍在能力范围之外。

人类/机器人数据混合实验只在 Serve Bread：固定30分钟、human ratio从 0/25/50/75/100%时成功率65/72.5/77.5/90/95。它比 ACT对照更接近数据源比较，但仍只有一任务，不能据此给出普遍的“不要投资 teleoperation”结论。

## 6. Artifact audit 与最终判断

| Artifact | 当前状态 | 复现含义 |
| --- | --- | --- |
| GitHub code | collection/preprocess/training/config存在；inference docs仍TODO | 可审计核心实现，部署链不完整 |
| Dataset | 122 recordings、2 tasks、raw+MPS+processed，CC BY-NC 4.0 | 可复跑部分数据流程；非四任务全集 |
| Checkpoint | Serve Bread，约243 MB | 可检查单任务权重；仍缺完整 inference recipe |
| 四任务/跨硬件评测资产 | 未见完整公开 | 92.5%与transfer表不能端到端独立复验 |

HumanEgo 的可迁移贡献是把“human vs robot”差异尽量推到 policy 之前，以 metric interaction state统一两者。这是一条很实用的工程路线，但 scalability bottleneck也随之从 teleoperation转移到高精度 pose recovery、task entity definition和部署侧 state estimation。下一步最需要的是：同一 policy/相机/representation下的人类与 robot data严格配对对照；完整四任务 release；对 pose noise、camera-frame choice、calibration drift和 object occlusion 的 stress test；以及 collision/force-aware action retargeting。
