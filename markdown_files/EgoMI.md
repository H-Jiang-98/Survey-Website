---
title: "EgoMI: Learning Active Vision and Whole-Body Manipulation from Egocentric Human Demonstrations"
method_name: "EgoMI"
authors: [Justin Yu, Yide Shentu, Di Wu, Pieter Abbeel, Ken Goldberg, Philipp Wu]
year: 2025
venue: arXiv
tags: [egocentric-demonstration, active-vision, whole-body-manipulation, bimanual-manipulation, spatial-memory, robot-free-data, retargeting, vision-language-action]
image_source: online
---

# EgoMI：把人类头部运动也当 action，主动视觉才不再是旁观者

> 本笔记基于 [arXiv:2511.00153v2](https://arxiv.org/abs/2511.00153)、[HTML 全文](https://arxiv.org/html/2511.00153)与[项目页](https://egocentric-manipulation-interface.github.io/)核验；v2 更新于 2026-03-10。论文仍未标注会议接收，且项目页没有论文承诺的 code/hardware/data下载链接。

## 阅读结论先行

EgoMI 最重要的观点是：第一视角示范里的 camera motion不是该被 SLAM消掉的 nuisance，而是与双手协调的主动感知 action。人会先看目标、再伸手；会转头搜索视野外物体；也会在遮挡前记住放置位置。若机器人只有固定相机或 wrist cameras，人类示范中的 observation-action闭环本身就无法复现。

方法用 Quest 3S追头和双 controller 6-DoF，头顶固定 ZED2i，controller上安装 wrist cameras并直接挂真实 Robotiq 2F-85 gripper；trigger drive-by-wire开合。这样每帧得到左右 gripper pose/opening、head pose、head/wrist RGB组成的29D state/action。Policy从 $\pi_0$初始化，先在约200 h内部多任务 EgoMI data上适配 Cartesian 29D action，再用每目标任务1–1.5 h数据训练40k steps（约50 h/策略）。这不是只凭小样本从头训练，结果同时依赖大规模私有 pre-finetuning和 task-specific finetuning。

Active head的实验证据很强但平台特定。Tabletop Search完整29D为36/40，wrist-only 20D为29/40；保留head image但固定head仅2/20，说明**能动视点**而非多一张图才关键。Shelf Search为35/40“points”（两个can分别计分），wrist-only为0/40；这里目标设计为不在 wrist视野且高度跨度约2.4 m，恰好检验head search，但point不等于episode成功率。

SPARKS在 memory task把21/40提高到31/40。它用head pose对历史帧按 viewpoint novelty、recency与低angular velocity打分，选定少量帧作为 Pali-Gemma额外 image tokens。这个对照证明past view有用；却没有与固定间隔、recent-N、random、full video、learned memory做等token比较，因此不能证明SPARKS heuristic本身优于其他memory。21/40也只是“接近随机”而非严格chance：任务动作链还有 grasp/handoff/place失败。

“zero robot data”在无on-embodiment teleoperation意义上成立，却通过硬件设计把 gap提前消掉：人实际举着与机器人相同的沉重gripper，wrist camera位置匹配，头戴ZED，固定reticle强迫目标居中，target robot还专门增加一条6-DoF YAM arm充当neck。它不是从普通ego视频学习，而是 robot-free、**robot-shaped** data collection；collection rig无需完整机器人，却高度知道目标 action与视觉geometry。

“whole-body”也需收窄：policy只输出左右end-effector和camera-head SE(3)+grippers，不输出mobile base、leg、joint或contact force。Rainbow RBY1的torso/arms和YAM neck由两个Pyroki differentiable IK进程落地。IK不可达时“尽量靠近”而非报错是工程韧性，不保证 collision-free、dynamic stability、contact feasibility或task semantics仍成立；实验还是轮式semi-humanoid，不涉及双足平衡/loco-manipulation。

论文称29D interface platform-agnostic和可zero-shot heterogeneous hardware，却只在一台特制RBY1+YAM+相同gripper/camera stack上评估。没有换机器人、neck、camera height/FOV或gripper的实验，所以跨平台是设计意图，不是经验证结论。

### 一句话总结

EgoMI有力证明了在搜索、跨大工作空间和视觉记忆任务中，head trajectory应与hand trajectory共同学习；但成功来自“同gripper/同wrist view/头部reticle/特制6-DoF neck/200 h私有预适配”的协同系统，故它桥接的是高度兼容semi-humanoid，而非任意human video到任意whole-body robot。

![EgoMI robot-free采集装置：Quest、头相机、腕相机和实体机器人夹爪](https://egocentric-manipulation-interface.github.io/data/egomi_device.png)

*图 1。官方 EgoMI device。它省去目标机器人teleop占用，却用相同gripper和camera placement保留机器人动作/视觉接口。*

## 1. 29D shared interface

World-frame dataset vector为：

$$
a_t=[r_6^L,p_3^L,g^L,\;r_6^R,p_3^R,g^R,\;r_6^H,p_3^H]\in\mathbb R^{29}.
$$

训练时保留右手world pose，把左手与head改写为相对右手：

$$
{}^RT_L=({}^WT_R)^{-1}{}^WT_L,
\qquad {}^RT_H=({}^WT_R)^{-1}{}^WT_H.
$$

Inter-gripper frame弱化global translation和部分base差异，也自然编码handoff几何；但右手仍是world anchor，右手tracking drift、初始episode frame alignment与左右角色交换都会改变表示。6D rotation经Gram–Schmidt投回 $SO(3)$。

Capture VR frame按首帧head水平位置和平面上双gripper forward direction对齐robot canonical frame，再乘controller→flange与TCP calibration。2.126±1.216 mm的表中误差表明tracking rig精确，但不含policy、IK、robot control和接触的end-to-end error。

## 2. SPARKS：head pose驱动的非学习记忆

Past frame $\tau$的score为：

$$
J(\tau)=\phi(\angle(\hat z_H(\tau),\hat z_H(t)))
+\psi(t-\tau)+\rho(\angle(\hat z_H(\tau-1),\hat z_H(\tau))).
$$

三项偏好新视点、较近历史和低角速度（低blur/可能fixation）。只有视角变化大于FOV比例阈值或translation超过阈值才进FIFO。训练离线预计算keyframe以保持IID sampling，部署在线 $O(L)$扫描lookback。

Reticle是不可忽略的监督设计：Quest没有eye tracking，作者要求操作者把固定中心十字对准目标，以head direction代理gaze。论文定性称无reticle policy常完全失败，却未报告量化ablation。这意味着head pose并非自然gaze的被动记录，而是经明确人类协议塑造的action label。

~~~mermaid
flowchart LR
    D["Quest head/hands + 3 cameras + real grippers"] --> X["29D head-hand trajectories"]
    X --> A["Canonical-frame alignment"]
    I["200 h private multi-task data"] --> F["pi0 to 29D finetuning"]
    A --> F
    F --> T["1-1.5 h task finetuning"]
    X --> S["SPARKS pose-scored past frames"]
    S --> T
    T --> P["40-step head + two-hand chunks"]
    P --> K["Torso/arms IK and neck IK"]
    K --> R["RBY1 + YAM active head"]
~~~

## 3. 实验结果与单位陷阱

| Task / variant | Result | 正确解释 |
| --- | ---: | --- |
| Tabletop Search，29D | 36/40 | episode success 90% |
| Tabletop，20D wrist-only | 29/40 | 72.5%；head贡献17.5 pp |
| Tabletop，head image但固定head | 2/20 | active actuation关键；trial数不同 |
| Shelf Search，29D | 35/40 | **can-placement points** 87.5%，非清楚的episode SR |
| Shelf，20D wrist-only | 0/40 | 任务需要off-screen/vertical search |
| Memory，single timestep | 21/40 | 无past visual context |
| Memory，SPARKS | 31/40 | +25 pp，仍有22.5%失败 |

每项只有一个训练设置和几十次real trials，无seed、CI、操作者split或训练数据量曲线。Tabletop fixed-head只20 trials，不与36/40严格同样本量；Shelf把两个罐子分开计点，会掩盖完成整段双罐任务的联合成功率。

## 4. Embodiment gap被转移到了哪里

| Gap | EgoMI机制 | 剩余限制 |
| --- | --- | --- |
| Visual appearance | 相同ZED/wrist camera placement、相同实体gripper | 人臂/身体仍可见；无mask/rerender |
| Action | 两hand+head Cartesian SE(3) | 无force/contact/base/leg action |
| Grasp | 人直接操作2F-85 trigger | 只适配同类parallel-jaw grasp |
| Viewpoint | 示范与robot均可6-DoF动头 | robot必须有足够neck workspace/FOV |
| Kinematics | differentiable IK+posture regularization | 近似不可达目标不等于可行轨迹 |
| Partial observability | SPARKS keyframes | heuristic、fixed buffer、无learned state |

## 5. Artifact audit 与最终判断

| Artifact | 核验状态 | 影响 |
| --- | --- | --- |
| Paper + project rollout videos | 公开 | 可核查方法、setup与定性行为 |
| Code/hardware designs | 论文称release；项目页无链接 | 当前不可复跑29D/SPARKS pipeline |
| 200 h general dataset | in-house，无下载 | 最关键pre-finetuning不可复现 |
| Task data/checkpoints | 未公开 | 三组成功率不可独立复验 |
| Robot stack | RBY1+YAM+ZED+2F-85细节给出 | 极专门、成本和assembly未报告 |

最有价值的后续对照应固定同一head-capable hardware与相同图像token，比较human head trajectory、learned gaze、scripted scan和RL active perception；同时报告head-free但external panoramic view，以区分“主动感知”与“只是增加可见区域”。跨平台至少要换一种neck DOF/FOV与gripper，并测IK residual、collision和task success随kinematic mismatch的退化。这样才能把EgoMI从优秀co-designed system推进为经验证的通用interface。
