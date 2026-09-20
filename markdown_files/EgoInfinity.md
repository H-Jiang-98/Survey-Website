---
title: "EgoInfinity: A Web-Scale 4D Hand-Object Interaction Data Engine for Any-View Robot Retargeting and Video-to-Action Robot Learning"
method_name: "EgoInfinity"
authors: [Gaotian Wang, Kejia Ren, Andrew Morgan, Yiting Chen, Howard H. Qian, Podshara Chanrungmaneekul, Kaiyu Hang]
year: 2026
venue: arXiv
tags: [egocentric-data, human-to-robot, video-to-action, hand-object-interaction, 4d-reconstruction, cross-embodiment, retargeting, action100m, dexterous-manipulation, humanoid, robot-learning]
image_source: online
---

# EgoInfinity：把 127K 小时互联网视频称作机器人数据之前，先看实际处理了多少

> 本笔记基于 [arXiv:2606.17385v2](https://arxiv.org/abs/2606.17385)、[HTML 全文](https://arxiv.org/html/2606.17385v2)、[项目页](https://rice-robotpi-lab.github.io/EgoInfinity/)、[官方代码](https://github.com/Rice-RobotPI-Lab/EgoInfinity)、[Hugging Face 数据预览](https://huggingface.co/datasets/Rice-RobotPI-Lab/egoinfinity) 与 [交互式浏览器](https://huggingface.co/spaces/Rice-RobotPI-Lab/EgoInfinity) 精读和核验。公开资产状态核验日期为 **2026-08-13**。论文最终版的 12 张图、6 张表与全部附录均已逐项检查；文中的 Action100M 可寻址规模、论文处理子集、当前公开预览严格分开。

## 阅读结论先行

EgoInfinity 的核心不是又采了一个第一视角数据集，而是把**约静态相机的任意视角互联网 RGB 视频**投影到一个相对本体无关的中间层：度量手网格/关键点/位姿、物体点云/网格/6D 位姿、粗粒度交互状态和语言描述；随后再用机器人专属的根坐标估计器、IK 与手指映射，把同一段 human motion 编译到不同本体。

这篇工作的最好部分是系统接口设计。MoGe-2/Flow3R、GeoCalib、WiLoR、SAM3/SAM2、SAM 3D Objects 与 FoundationPose++ 并非简单串联：作者用共同的相机坐标和度量尺度连接各模块，再让交互状态决定物体姿态应信任视觉、静态点云还是手坐标。抓取段中物体相对手刚性绑定，确实能压制遮挡下的视觉漂移。机器人端不恢复完整人体，而从双手轨迹生成可行的机器人根坐标候选，再以 IK 可解性筛选；这是对 arbitrary-view 和 partial-body 输入很务实的 functional retargeting。

但“4D contact state”不能读成精确接触或物理真值。内部六状态只区分全局静止、静止、左/右/双手抓取与移动；抓取来自 2D mask overlap 或 3D 手指/手腕距离阈值，没有接触点、法向、压力、力矩、触觉、无滑移或穿透约束。论文自己承认不能保证指尖对齐、force consistency 和 no-slip。更关键的是，论文没有带真值的 hand/object reconstruction benchmark，也没有 interaction refinement ablation；因此“度量”“物理可靠”主要由模块设计和定性结果支撑，不能当成测量级精度证明。

规模声明也要降温。**127K 小时**是 Action100M 的 14.6 年可寻址上游语料，不是 EgoInfinity 已完成处理和发布的 4D 数据。论文真正展示的是 **106 clips、277 objects、12,107 frames、中位 5.8 s** 的 curated subset；当前 Hugging Face 是 **preview**，约 5.49 GB、1150 rows，并给 **104 clips × 4 embodiments** 的 retargeting 结果。它证明了引擎可以工作和数据产品可以发布，没有实证给出吞吐量、处理成本、全语料接受率或 127K 小时产出规模。

机器人证据同样是“可行性强、因果性弱”。三种主本体的平均逐帧 IK rate 为 0.821/0.774/0.706，位置误差 2.86/6.67/10.27 cm；真实 dual-Franka 展示切、倒、擦，LEAP hand 展示使用 motion prior 的抓取 policy。但论文没有任务成功次数/总次数、基线、消融、seed、置信区间或未见任务 split。真实演示说明端到端路径存在，不足以证明它比纯机器人数据、普通 IK 或无 refinement 更好。

### 一句话总结

EgoInfinity 把互联网视频转成**几何化、可重定向的人手—物体运动中间层**，并开放了相当完整的工程原型；它最适合做 motion prior、candidate action 与跨本体 retargeting，尚不是精确 contact dataset、完整 robot demonstration dataset，也没有证明 127K 小时级数据已被生成或能提升通用策略成功率。

### Elevator pitch

互联网教程只给像素，机器人却需要米制几何、物体状态和可执行动作。EgoInfinity 先把手、物体、深度、重力和语言统一到相机坐标中的 4D HOI，再以“静止时信点云、抓住时信手、移动时信视觉”的状态机稳定物体轨迹；最后为每种机器人从双腕运动估计一个可行根坐标，并用 IK 编译为关节轨迹。这是一座从 web video 到 robot motion 的工程桥，但桥面仍缺少精确接触、动力学反馈和大规模策略收益验证。

![EgoInfinity 总览：数据筛选、度量 4D HOI、交互感知修正与机器人重定向](https://arxiv.org/html/2606.17385v2/pipeline.png)

## 0. 资源、版本、开放状态与三个规模口径

| 资产 | 正式入口 | 截至 2026-08-13 的状态 | 许可/约束 | 复现判断 |
| --- | --- | --- | --- | --- |
| 论文 | [arXiv v2](https://arxiv.org/abs/2606.17385)、[HTML](https://arxiv.org/html/2606.17385v2)、[PDF](https://arxiv.org/pdf/2606.17385) | v2，2026-06-19，24 页；目标稿源写明 CoRL 2026，但当前公开 venue 仍应记 arXiv | 论文 CC BY 4.0 | 高 |
| 项目页 | [Method & Experiments](https://rice-robotpi-lab.github.io/EgoInfinity/) | 给管线、公式摘要、定量表、真实演示和所有正式入口 | 页面自身未单列许可 | 高 |
| 引擎代码 | [Rice-RobotPI-Lab/EgoInfinity](https://github.com/Rice-RobotPI-Lab/EgoInfinity) | 审计提交 `de59610531c08010d63d0493d6a72973eabd402e`，initial commit；约 39.7K 行 Python/shell，含 orchestration、4D pipeline、retargeter 与四套 checkpoint | 自研源码 MIT；完整栈受 WiLoR/MANO 非商用、YOLO AGPL、上游模型许可约束 | 中 |
| 数据预览 | [HF dataset](https://huggingface.co/datasets/Rice-RobotPI-Lab/egoinfinity) | public、ungated；revision `1e3fe515f9a09b0161ead4e2bf0ccc5a1b9f708b`；页面给 1150 rows、5.49 GB，明确标记 preview | FAIR Noncommercial Research License v1；仅非商用研究 | 中高 |
| 交互浏览器 | [HF Space](https://huggingface.co/spaces/Rice-RobotPI-Lab/EgoInfinity) | 可浏览重建与中间量；论文附录称静态 Viser 客户端、不依赖运行时后端 | 受数据与组件许可共同约束 | 中高 |
| 重定向权重 | [`retarget/ckpts`](https://github.com/Rice-RobotPI-Lab/EgoInfinity/tree/main/retarget/ckpts) | G1、Franka、Robonaut2、XLeRobot 四个约 10.6 MB checkpoint 随仓库提供 | README 称 preliminary checkpoints | 中高 |
| 原始 Action100M | 通过代码文档说明获取 | 原始互联网素材不由 EgoInfinity 重新开放；需遵守 Action100M 访问/许可链 | 非商用研究边界 | 中低 |

### 0.1 127K h、106 clips、104 clips 分别是什么

| 数字 | 精确含义 | 不能据此声称什么 |
| --- | --- | --- |
| 14.6 年 ≈ 127K h、147M action segments | Action100M 上游语料的可寻址规模；引擎理论输入池 | 不是已运行完 EgoInfinity 的 4D 数据小时数，也不是当前下载量 |
| 106 clips、277 objects、12,107 frames | 论文固定阈值处理和统计的 curated subset；中位 clip 5.8 s | 不是 web-scale 的经验吞吐评测，也不是训练大模型的有效小时数 |
| 104 clips × 4 robots | 当前数据卡明确列出的四本体 retargeting 结果 | 不能自动解释论文 106 与发布 104 的两条差异；也不等于 416 个独立 human episodes |
| 1150 rows、5.49 GB | Hugging Face viewer/仓库当前报告的预览数据产品 | row 不是 clip；文件/对象/视频条目的粒度混合，不能换算成时长 |

把“data engine 可面向 web scale”与“已产出 web-scale robot data”分开，是评价本文最重要的第一步。论文没有给 GPU-hours/clip、失败率、全量过滤保留率、SAM3/SAM3D 成本、存储膨胀率或端到端 127K 小时预算。

### 0.2 开放不等于可商用或一键复现

项目自己的 Python 代码是 MIT；但完整 pipeline 至少包含以下许可证和访问摩擦：WiLoR 为 CC-BY-NC-ND，MANO 需注册并同意非商用研究许可，SAM3.1 权重 gated，YOLO 依赖 AGPL-3.0，SAM 3D Objects 与 FoundationPose++ 需要独立环境/容器，Action100M 衍生数据为 FAIR Noncommercial Research License。MIT 只覆盖作者自己的源码，不能覆盖整条依赖链。

官方 README 推荐 A100 40 GB 或 H100 80 GB，SAM3.1 和 SAM 3D Objects 常驻约 4 GB 与 10 GB，active phase 再需约 3–5 GB；Flow3R 约 9–10 GB 且默认关闭。代码仓库虽很完整，却不是普通单环境 `pip install` 后即可复现论文所有输出。

## 1. 论文要解决的 Gap 与真正贡献

互联网视频给任务语义、物体多样性和自然动作，却缺少 robot execution 所需的 metric geometry、object state 和 action。EgoInfinity 将问题拆成两个编译阶段：

1. `RGB + text → agent-agnostic 4D HOI`：恢复手、物体、相机尺度与交互状态。
2. `4D HOI → embodiment-specific trajectory`：估计机器人根坐标、求 IK、映射手指并平滑。

论文贡献可压成三点：

- 一个无需专用采集硬件、Mocap、CAD 和逐条人工 object prompt 的模块化处理引擎；
- 共同度量坐标与 interaction-aware refinement，使不同 foundation models 的输出可以被同一几何系统消费；
- 基于 SE(3)-equivariant Vector Neuron 与 flow matching 的 robot-specific root-frame estimator，用于 partial-body arbitrary-view 的 functional retargeting。

### 1.1 Gap—Evidence 总表

| Gap/主张 | 论文方案 | 给出的证据 | 证据边界 |
| --- | --- | --- | --- |
| 互联网 RGB 缺米制 3D | MoGe-2 metric anchor、Flow3R depth、WiLoR hand、SAM3D object | 106 clip gallery 和交互浏览器 | 无带 GT 的深度、手/物位姿误差 benchmark |
| 模块尺度/坐标不一致 | 统一 OpenCV camera-world frame、焦距/尺度/重力校准 | 设计说明和可视化一致性 | 无 calibration ablation 或跨模块 residual |
| 遮挡下 object pose 漂移 | 六状态分类、static lock、grasped rigid bind、moving proposal | 定性序列、状态分布 | 无 drift/contact consistency 数值与 ablation |
| 任意视角缺人体根坐标 | SE(3)-equivariant conditional root generator | 三本体 IK 指标、四本体 gallery | 没有 root pose GT 或与启发式/回归基线比较 |
| 人体轨迹不可直接执行 | candidate clustering + IK + smoothing + finger mapping | G1/Robonaut2/Franka 指标 | 20.6%–29.4% 帧 IK 失败；失败帧插值不等于可执行性证明 |
| 跨本体泛化 | 每机器人单独训练 root model | G1、R2、Franka、XLeRobot | 不是一个 zero-shot universal retargeter；新机器人需训练/标定 |
| 视频可变真机技能 | FR3 replay；LEAP motion-prior policy | 五种 FR3 filmstrip、三类 LEAP 物体 | 无成功率、trial count、baseline、统计显著性 |
| web-scale data generation | 两遍筛选 + 自动 object discovery | Action100M 可寻址 127K h | 实际只报告 106 clips；无吞吐、成本、接受率 |

## 2. 与已有数据和方法相比，生态位在哪里

### 2.1 论文表 1：数据源与机器人可用性

| Dataset | Source | Annotation | Wearable requirement | Auto generation | Manual object | 论文标称规模 |
| --- | --- | --- | --- | --- | --- | ---: |
| Ego4D | curated | narration | headset | 否 | 否 | 3.7K h |
| EgoDex | curated | tracking | Vision Pro | 否 | 否 | 829 h |
| HOT3D | curated | Mocap | Mocap | 否 | 否 | 13.9 h |
| OakInk2 | curated | Mocap | Mocap | 否 | 否 | 6.5 h |
| UniHand-Mix | aggregated | mixed | partial | 否 | 否 | 1.2K–35K h |
| Open X-Embodiment | robot aggregate | robot actions | robot | 否 | 否 | 1M+ trajectories |
| DROID | teleoperation | robot actions | robot | 否 | 否 | 350 h |
| EgoInfinity | internet | automatic 4D | 无 | 全自动 | 无 | 127K h† |

`†` 这里是 Action100M 的 corpus ceiling。表中 “auto 4D” 和 robot dataset 的真实 action annotation 也不是同类量：EgoInfinity 输出从视觉估计而来的 human/geometry proxy，Open X/DROID 输出实际机器人状态或命令。

### 2.2 相比三类相邻路线

- **相对 EgoDex/Open-AoE/EgoVerse**：EgoInfinity 不负责新的硬件采集，重点是 post-hoc projection、object 4D 与 executable retargeting；代价是所有几何监督都是模型估计而非采集真值。
- **相对 Phantom/H2R/MimicDreamer**：它不优先生成逼真的机器人 RGB 或编辑人手，而优先恢复几何中间层；因此 action projection 更直接，observation/appearance gap 保留更多。
- **相对 EgoGrasp**：两者都借 grasp state 处理 occluded object；EgoInfinity 假设约静态相机、无需 full-body/SLAM，用确定性状态机和 rigid bind，目标是 corpus-scale engine 与跨本体 retargeting。

## 3. 数据投影：从像素到五层监督

EgoInfinity 并没有一条直接从 video 到 motor command 的可学习映射。它先生成五层监督，每层的可信度和用途不同。

| 层级 | 输出 | 来源 | 是否观测真值 | 最适合的下游 | 关键风险 |
| --- | --- | --- | --- | --- | --- |
| 语义层 | short/detailed action description、caption、object nouns | Action100M hierarchical annotations | 否，继承/模型标注 | retrieval、task conditioning、object prompt | caption 错误会传入对象发现和任务标签 |
| 视觉层 | 原 RGB、mask、depth、optical flow、虚拟 ego render | web video + foundation models | RGB 是观测；其余估计 | representation/world model/debug | 约静态相机选择偏差；透明/反光失败 |
| 人手层 | 21 joints、778-vertex MANO、shape/pose、hand SE(3) | YOLO + WiLoR + metric alignment + infiller | 否 | wrist/finger prior、retargeting | 遮挡、handedness、补帧、单目尺度误差 |
| 物体/交互层 | mask、point cloud、mesh、6D pose、六状态 | SAM3/2/3D、depth、tracker、状态机 | 否 | object-centric policy、contact phase、replay | 无 articulation、真实 contact/force；rigid object 假设 |
| 机器人层 | root frames、joint trajectories、IK metrics | robot-specific root estimator + IK + finger map | 合成动作，不是执行日志 | motion proposal、warm start、prior | reachability、碰撞、动力学、控制和 gripper topology gap |

### 3.1 语义不是独立语言标注

SAM3 的 text prompt 来自 Action100M 已有 short/detailed description 和 clip caption，名词抽取后形成对象列表。例如 “slicing a tomato” 给出 `tomato`。优点是无需逐 clip 人工指定；缺点是检测召回率上界被 caption/object extraction 锁定。没有被文本提到的工具、容器或辅助对象可能根本不进入 4D state。

仓库的通用 `manifest.json` 仍要求 `objects: ["red mug", ...]`。对 Action100M，作者可从原注释自动产生；对任意用户视频，当前文档并没有证明 object list 总能无人工/VLM地得到。因此“no human-specified objects”适用于带语义 annotation 的 corpus processing 路线，不等于裸视频接口没有语义入口。

### 3.2 “任意视角”与约静态相机必须同时写

`any-view` 指观察角度与 shot size 可任意，人体可只露手；它不表示自由移动相机。第一遍筛选会看 hand presence、hand motion 和 camera-motion cue，正文明确将范围限定为 approximately static views，典型是 tripod tutorial/how-to。头戴式、手持拍摄和强相机运动不在当前可靠 operating envelope 内。

### 3.3 虚拟 ego 视角是什么、又不是什么

作者在恢复的 3D 场景中，把虚拟相机放在双手中点上方，以 GeoCalib 重力为 up-axis，朝向 hand-object interaction region，并逐帧跟随 anchor。它是 deterministic rigid reframing：

$$
{}^{e}\mathbf p_t = {}^{e}mathbf T_c(t),{}^{c}\mathbf p_t.
$$

这个式子只改变已有 3D 点/网格的坐标。它可以统一 viewpoint 并保持 metric geometry，不会凭空“修复”原视频纹理、背景、被遮挡表面，也不会生成目标机器人在场的 photorealistic observation。更准确的说法是 **geometry-space ego re-rendering**，而非完整 human-to-robot visual synthesis。

## 4. 全管线：哪些模块互相校准，哪些误差会累积

~~~mermaid
flowchart LR
    A[Action100M RGB + text] --> B[Pass 1: hand/camera-motion filter]
    B --> C[Pass 2 active clips]
    C --> D[MoGe-2 focal + metric anchor]
    C --> E[Flow3R dense depth]
    C --> F[GeoCalib gravity]
    C --> G[YOLO + WiLoR + infiller]
    C --> H[SAM3 object detection]
    H --> I[SAM2 bidirectional mask track]
    I --> J[SAM 3D Objects mesh]
    D --> K[shared OpenCV metric camera frame]
    E --> K
    F --> K
    G --> K
    I --> K
    J --> K
    K --> L[initial 4D HOI]
    L --> M[interaction state machine]
    M --> N[state-dependent object refinement]
    N --> O[virtual ego reframing]
    N --> P[root-frame flow model]
    P --> Q[candidate scoring + IK]
    Q --> R[robot joint trajectory]
~~~

### 4.1 两遍处理与可扩展性

Pass 1 做轻量 temporal scan，只保留手出现、手运动显著且相机近静止的片段；Pass 2 才运行昂贵的 depth、hand、segmentation、reconstruction 和 tracking。这个结构合理，但论文没有报告：原始 clips 数、每门的保留率、false reject、平均处理时间、峰值显存、总 GPU-hours。因此目前证明的是 software architecture 可 batch，而不是完整 web-scale economics。

### 4.2 共同度量几何

MoGe-2 提供焦距和全局 metric anchor，Flow3R 提供 dense depth，GeoCalib 从三个均匀采样帧估重力；WiLoR 给 MANO，再以 depth 对 fingertip projection 做多尺度对齐来校正 root translation。物体 mask 与 depth 反投影成点云，SAM 3D Objects 给 canonical mesh/scale/orientation。

所有量使用 OpenCV camera-world frame：右手系，$+x$ 向右、$+y$ 向下、$+z$ 指向场景，相机近静止且置于原点。初始状态为：

$$
\mathcal H_t = \left\{
\mathcal M_t^h,\mathcal K_t^h,{}^c\mathbf p_t^h,
\mathcal P_t^o,\mathcal M^o,{}^c\mathbf p_t^o
\right\}.
$$

这里 $\mathcal M_t^h$ 是手网格，$\mathcal K_t^h$ 是关键点，$\mathcal P_t^o$ 是物体点云，$\mathcal M^o$ 是 canonical mesh，$\mathbf p=(\mathbf R,\mathbf t)\in SE(3)$ 是位姿。统一坐标消除了接口层的任意 scale/frame mismatch，但不消除各模型本身的偏差；同一错误 depth anchor 还会同时污染 hand/object，看起来彼此“对齐”却整体尺度错误。

### 4.3 Perception 模块配置和不确定性来源

| 模块 | 论文/附录配置 | 输出 | 主要不确定性 |
| --- | --- | --- | --- |
| MoGe-2 | ViT-L, FP16 | metric depth、focal | 单目尺度与域偏差；逐帧无直接 temporal aggregation |
| Flow3R | dense depth；代码中 opt-in，默认 off | temporal geometry refinement | 局部 scale ambiguity、显存和帧采样 |
| GeoCalib | 3 个均匀帧 | gravity | 非 Manhattan/斜镜头、少帧方差 |
| YOLO + WiLoR | DINOv2-L，21 joints/778 vertices | hand box、MANO | 遮挡、小手、motion blur、handedness |
| motion infiller | 约 35M Transformer | missing hand frames | 补出的 plausible motion 被误当观测 |
| SAM3 + SAM2 | text detect，forward/backward tracking，最多 7 objects | object masks | prompt miss、mask drift、相邻对象合并 |
| SAM 3D Objects | 最干净无遮挡单帧 | mesh、canonical pose/scale | 单视角不可见面、透明/反光、类别先验 |
| FoundationPose++ / internal tracker | canonical rotation bake 或 FGR/ICP + flow/PnP + optimization | object 6D | 论文与释出实现路径不完全同名；遮挡与对称性 |

仓库当前把 Flow3R 标为默认关闭，把 FoundationPose++ 作为 optional canonical rotation bake；核心 tracking 文档写成 FGR+ICP anchor、optical flow+RANSAC-PnP propagation、7-loss LBFGS。论文正文则把 Flow3R 与 FoundationPose++ 写进主方法。这是一个实质的**paper–release configuration drift**：复现时必须冻结 config 和 enabled stages，不能只说“运行官方代码”。

## 5. Interaction-aware refinement：本文最值得复用的算法思想

纯视觉 tracker 在三种状态下的可信度不同：静止物体不应漂；抓住后物体被手遮挡但相对手近似固定；短暂移动且未确认抓取时才主要依赖视觉。EgoInfinity 把这个直觉做成六状态分类与 state-conditioned estimator selection。

### 5.1 六状态与三状态

内部标签是：

$$
\sigma_t\in\left\{
\textsc{static\_global},\textsc{static},
\textsc{grasped\_l},\textsc{grasped\_r},
\textsc{grasped\_both},\textsc{moving}
\right\}.
$$

正文把前两个折成 `static`，三个 hand-specific grasp 折成 `grasped`，最后保留 `moving`。

全局静止门使用物体 mask centroid 的 10–90 percentile span：

$$
\Delta_{[10,90]}=
\left\lVert p_{90}(\mathbf c)-p_{10}(\mathbf c)\right\rVert_2
\le 0.02\min(H,W).
$$

非全局静止时，逐帧 centroid displacement 通过 low/high = 2/4 px 的 Schmitt trigger；它避免单阈值附近来回抖动。

每只手的 grasp signal 是三项 OR：

- rasterized MANO 与 object mask overlap 至少 30 px；
- fingertip 到 object point cloud 距离不超过 6 cm；
- wrist 到 point cloud 距离不超过 5 cm。

二值序列再做 close-and-drop：桥接不超过 30 帧的内部 gap，删除短于 8 帧的 positive run。以 15 FPS 算，这相当于可桥接约 2 秒中断、移除约 0.5 秒短段；对慢教程宽容，却可能把真实 release/regrasp 合并。

层级赋值是：

$$
\sigma_t=
\begin{cases}
\textsc{static\_global}, & \text{全局静止};\\
\textsc{grasped\_both}, & \hat g_L^{(t)}\land\hat g_R^{(t)};\\
\textsc{grasped\_l}, & \hat g_L^{(t)};\\
\textsc{grasped\_r}, & \hat g_R^{(t)};\\
\textsc{moving}, & m_t;\\
\textsc{static}, & \text{其余}.
\end{cases}
$$

![交互状态机：静止、移动、单手/双手抓取与 dominant-hand 决议](https://arxiv.org/html/2606.17385v2/A1.svg)

### 5.2 双手抓取如何变成单一刚性绑定

`grasped_both` 不能同时由两个含噪 hand frames 刚性驱动物体。作者先看整段中明确单手 frames 的多数：某侧独占，或达到 5:1，就用该手；没有明确单手帧时，比较两侧 fingertip-to-cloud 最小距离的 clip-wide majority。对真正双手协作、handover、双腕相对运动任务，这种降维会丢失第二只手的约束。

### 5.3 状态决定姿态来源：论文表 3

| State | Translation source | Rotation source | 隐含假设 |
| --- | --- | --- | --- |
| `static_global` | 全 clip robust point-cloud centroid 中位数 | SAM3D canonical orientation | 场景固定，不存在 articulation |
| `static` | 每个静止 stretch 的 robust centroid 中位数 | SAM3D canonical orientation | 物体段内静止 |
| `moving` | eroded-mask bbox-center back-projection + Gaussian smoothing | per-frame PCA、sign corrected | 可见点云足以定向；对对称物体不稳定 |
| `grasped_*` | hand pose × canonical hand-relative translation | hand pose × canonical relative rotation | 抓取后刚体、无相对滑动/重抓变化 |

正文的初始 proposal 是：

$$
\tilde{\mathbf p}_t^o=
\left(\mathbf R^{\mathrm{cano}},
\operatorname{center}(\mathcal S_t^o\odot D_t)\right).
$$

抓取状态则使用：

$$
{}^c\hat{\mathbf p}_t^o=
{}^c\mathbf p_t^h\,\mathbf T^{\mathrm{cano}}.
$$

这不是从视觉“测得”物体在遮挡下的 pose，而是施加了 object-hand relative transform 恒定的运动学 prior。它会消除 tracker drift，也会把真实 in-hand rotation、slip、tool pivot 或 finger gait 抹掉。

### 5.4 手坐标与 chirality-aware placement

手体坐标用 wrist $\mathbf j_0$、index MCP $\mathbf j_5$、pinky MCP $\mathbf j_{17}$：

$$
\begin{aligned}
\mathbf x &= \frac{\mathbf j_5-\mathbf j_0}{\lVert\mathbf j_5-\mathbf j_0\rVert},\\
\mathbf v &= (\mathbf j_{17}-\mathbf j_0)-
\big((\mathbf j_{17}-\mathbf j_0)^\top\mathbf x\big)\mathbf x,\\
\mathbf y &= \frac{\mathbf x\times\mathbf v}{\lVert\mathbf x\times\mathbf v\rVert},\qquad
\mathbf z=\mathbf x\times\mathbf y,\\
{}^c\mathbf R_t^h&=[\mathbf x\;\mathbf y\;\mathbf z],\qquad
{}^c\mathbf t_t^h=\mathbf j_0.
\end{aligned}
$$

它只依赖 palm landmarks，因此不随手指屈伸剧烈变化。右手 $+y$ 朝掌外，左手相反，作者在 mesh placement 中显式翻转 chirality：把旋转后 mesh 的相应厚度极值贴到 $y=0$ palm plane，并让 mesh/palm centroid 的 $(x,z)$ 对齐。

![掌面坐标与左右手感知的物体贴合](https://arxiv.org/html/2606.17385v2/A2.svg)

### 5.5 刚性绑定、深度再对齐与边界平滑

每个 grasp segment 先从观测 pose 算：

$$
\mathbf T_t^{\mathrm{rel}}=
({}^c\mathbf p_t^h)^{-1}{}^c\mathbf p_t^o.
$$

相对 rotation 先排除离 robust seed 超过 $30^\circ$ 的帧，再做 chordal mean；translation 做逐轴 MAD-filtered median，随后由上述几何贴掌替换。由于 hand-object 边缘 depth blur 可造成 1–10 cm 深度偏移，作者以每帧 20 对最近 hand-vertex/mesh-point 算 $\Delta z_t$，最近距离若超过 20 cm 就认为 grasp false positive；Savitzky–Golay window 7/order 2 平滑，修正限幅 $\pm0.1$ m。相邻 grasp/non-grasp 或 hand change 以 5-frame SLERP/LERP 过渡。

### 5.6 Robust statistics 与全部固定阈值：论文表 4

MAD inlier rule 为：

$$
d_i\le \tilde d+3\cdot1.4826\cdot\operatorname{MAD},
$$

rotation chordal mean 为：

$$
\bar{\mathbf R}=\operatorname{Proj}_{SO(3)}
\left(\frac{1}{K}\sum_k\mathbf R^{(k)}\right),
$$

其中若 $M=U\Sigma V^\top$，投影是
$U\operatorname{diag}(1,1,\det(UV^\top))V^\top$。

| Parameter | Value | Use |
| --- | ---: | --- |
| Global-static span | $0.02\min(H,W)$ | static gate |
| Motion Schmitt | 2/4 px | motion hysteresis |
| Mask-overlap grasp | 30 px | 2D contact proxy |
| Fingertip distance | 6 cm | 3D grasp fallback |
| Wrist distance | 5 cm | 3D grasp fallback |
| Grasp bridge/min-run | 30/8 frames | temporal morphology |
| Rotation outlier | $30^\circ$ | relative-pose aggregation |
| MAD multiplier | $3\times1.4826$ | robust filters |
| Moving Gaussian | $\sigma=2$ frames | translation smoothing |
| Depth-realign K/far reject | 20 / 0.2 m | contact depth alignment |
| Depth-realign SavGol/cap | window 7, order 2 / $\pm0.1$ m | depth correction |
| Boundary ramp | 5 frames | segment continuity |
| MANO SavGol | window 9, order 3 | hand smoothing |
| Scale sanity | $1.8\times$ | replace implausible mesh scale |
| Spurious distance/motion | 0.5 m / 10 px | background false-match flag |

所有阈值在 development set 上按 inspection 选择，在 106 clips 固定使用；论文没有 sensitivity analysis。如此多的 hard thresholds 是系统可解释性的优点，也是换分辨率、FPS、camera motion、物体尺度和任务速度时的 domain fragility。

## 6. 数据清理、sanity filtering 与信息损失

对象点云可做 mask erosion、depth-gradient filtering 与 statistical outlier removal。SAM3D scale 若超过 mask-implied size 的 1.8 倍，就以

$$
s^{\mathrm{mask}}=
\max(W_{\mathrm{mask}},H_{\mathrm{mask}})\frac{\bar D}{f}
$$

替代。若 object centroid 离所有 hand activity 超过 0.5 m，且全 clip 2D centroid 移动小于 10 px，则标为 background false match；作者选择 dim/flag 而非删除，让下游决定。

这条处理链的主要信息损失是：

~~~mermaid
flowchart TD
    V[web video 的全部行为] --> F[近静态相机 + 可见手过滤]
    F --> P[文本提到/检测到的最多 7 个对象]
    P --> M[单帧 rigid mesh + 6D rigid pose]
    M --> S[六状态离散化]
    S --> B[双手抓取压到 dominant hand]
    B --> R[抓取段 object-hand rigid bind]
    R --> I[robot root + IK + smoothing]
    I --> O[可执行候选轨迹]
    V -. 丢失 .-> X[移动相机、无手任务、柔性/铰接、非文本对象]
    S -. 丢失 .-> Y[contact point/force/slip/in-hand motion]
    I -. 丢失 .-> Z[动力学、触觉、闭环 recovery、真实成功结果]
~~~

因此，EgoInfinity 的“投影”不是无损转换，而是主动选择最有利于 rigid manipulation retargeting 的统计量。

## 7. Cross-embodiment retargeting：从双手轨迹推回机器人根坐标

一般 IK 需要知道 robot base/torso 相对手目标的位置。但任意互联网视频常只看到手，无法恢复人的完整躯干。EgoInfinity 学习条件分布
$p({}^c\mathbf p^r\mid\mathbf x)$，其中 $\mathbf x$ 是双手 SE(3) 轨迹和可选重力，$\mathbf p^r$ 是目标机器人根坐标。

![功能性跨本体重定向：根坐标候选、聚类、评分、IK 与优化](https://arxiv.org/html/2606.17385v2/retarget.png)

### 7.1 SE(3) equivariance

对任意 $\mathbf G\in SE(3)$：

$$
\mathbf G\cdot\Phi(\mathbf x)=
\Phi(\mathbf G\cdot\mathbf x).
$$

Rotation equivariance 由 Vector Neuron feature 保证；translation 先减去双手位置 centroid $\mathbf c$。每只手在每个时间步转为 5 个三维向量通道：

$$
\left[
{}^c\mathbf t_t^h-\mathbf c,
{}^c\mathbf R_t^h[:,0],
{}^c\mathbf R_t^h[:,1],
{}^c\mathbf R_t^h[:,2],
{}^c\mathbf g
\right].
$$

noisy root state 在 flow time $\tau$ 的条件通道为：

$$
\left[
{}^c\mathbf t_\tau^r-\mathbf c,
{}^c\mathbf R_\tau^r[:,0],
{}^c\mathbf R_\tau^r[:,1],
{}^c\mathbf R_\tau^r[:,2]
\right].
$$

两手分别经 `VN-Linear(5,d)`，拼接后以 `VN-Linear(2d,d)` 融合，加入 noisy-root/time conditioning，进入 4-layer VN-Transformer。Rotation head 预测两个向量后以 Gram–Schmidt 得 $SO(3)$；translation head 预测 body-frame offset $\mathbf v$：

$$
{}^c\mathbf t^r={}^c\mathbf R^r\mathbf v+\mathbf c.
$$

![Root-frame estimator：VN 编码、时序 Transformer 与 flow-matching 输出头](https://arxiv.org/html/2606.17385v2/retargeter_appendix.png)

### 7.2 为什么用 flow matching 而不是确定性回归

相同双手轨迹可能来自多个 torso/base pose，特别是只有单手、partial body 或视角含糊时。先验取

$$
{}^c\mathbf R_0^r\sim\mathcal U(SO(3)),\qquad
{}^c\mathbf t_0^r\sim\mathcal N(\mathbf c,0.5^2\mathbf I),
$$

flow model 学会把 prior sample 沿 conditional velocity field 推到 plausible root frames，推理用 20 次 Euler integration。这里的 distribution 主要服务 candidate generation；论文没有用 likelihood/calibration 指标证明它是可信概率分布。

### 7.3 论文表 5：网络规格

| Parameter | Value |
| --- | ---: |
| VN channel width $d$ | 128 |
| attention heads | 4 |
| Transformer layers | 4 |
| FFN hidden width | 512 |
| dropout | 0.1 |
| per-hand input | $(T,7)$ |
| per-hand VN feature | $(T,5,3)$ |
| fused feature | $(T,d,3)$ |
| pooled output | $(d,3)$ |
| Euler flow steps | 20 |

### 7.4 纯仿真 robot-specific training

每个机器人单独在 MuJoCo/MJX 中训练。以 noisy reference joint config 做 FK anchor，7 个 Cartesian control points 通过 Ornstein–Uhlenbeck random walk 产生，再以 warm-start position-only IK 回到 joint knots，cubic spline 得 60 帧、30 FPS、2 秒轨迹。随机 camera pose 把 hand trajectory 和 root frame 都投到 camera frame。

![四本体的仿真训练轨迹与随机相机坐标](https://arxiv.org/html/2606.17385v2/retarget_train.png)

### 7.5 论文表 6：训练配置

| Parameter | Value |
| --- | ---: |
| GPU | RTX 3060 12 GB |
| time per robot | 约 1.5–2 h |
| epochs / steps per epoch | 500 / 20 |
| batch | 1024 fresh simulated trajectories |
| optimizer / LR | Adam / $10^{-3}$ |
| grad clip | 1.0 |
| trajectory | 60 frames @ 30 FPS |
| OU step noise / spring | 0.025 m / 0.05 |
| flow loss weights $(w_R,w_t)$ | (1,1) |
| position/orientation noise | 0.01 m / 0.05 rad |
| tracking jump | $p=0.20$，max 0.15 m |
| hand occlusion | $p=0.20$ |
| gravity noise/dropout | 0.10 rad / $p=0.30$ |
| rear-facing camera | $p=0.15$ |

flow-matching objective 对 rotation velocity 和 body-frame translation velocity 使用等权 $\ell_2$ loss。论文没有列精确 probability path/rotation tangent-space loss 的正式公式，也没有报告训练/验证曲线；因此实现复现需要依赖代码，而非只靠正文。

### 7.6 Inference、candidate selection 与 IK

视频按 sliding window 估 root frames，在 $SE(3)$ geodesic metric 下聚成 $K=5$ 候选；每个 candidate 对完整 hand trajectory 运行 batched IK，以 bilateral convergence 为主要 anchor selection 信号。随后 translation/rotation 分别以 $\alpha_t=0.3$、$\alpha_r=0.7$ 混合 anchor 与 per-window roots，Gaussian $\sigma=10$ frames 平滑。

IK 用 warm start 和 null-space objectives，考虑 tracking residual、manipulability、joint-limit、default posture 和可选 self-collision。失败帧做线性插值，joint trajectory 再平滑。这里应强调：**插值能补齐数组长度，不能把不可达帧变成已验证的物理可行帧**。

Dexterous hand finger joints 与 arms 分开：arm 追 wrist-level IK，finger 由 MANO keypoints 经 robot-specific geometry mapping。夹爪或低 DoF hand 会压缩大量 human hand topology；高 DoF hand 则仍缺 contact optimization。

## 8. Curated subset：分布、覆盖和 collection 审计

EgoInfinity 自己不采集视频，而从 Action100M 做 corpus mining 和 processing。因此 collection prompt 应拆成两层：上游素材如何形成，本文如何选择和投影。

![论文实际处理的 106 clips：时长、物体类别、动作动词与交互状态](https://arxiv.org/html/2606.17385v2/stats.svg)

### 8.1 论文报告的实际子集

| 统计 | 数值 | 解读 |
| --- | ---: | --- |
| clips | 106 | 论文量化/浏览器子集，不是完整 Action100M |
| objects | 277 | 每 clip 可多对象，提示最多保留 7 objects |
| frames | 12,107 | 规模约为分钟级，而非小时级 robot training corpus |
| median duration | 5.8 s | 主要是短动作；少量 20–25 s tail |
| clips with manipulation | 88% | 仍有约 12% clips 未判为 manipulation |
| objects manipulated | 47% | 过半 detected objects 没有进入 manipulation state |

物体类别：container 139（49%）、tool 49（17%）、food 38（13%）、hardware 20（7%）、appliance 12（4%），其余 textile/electronics/decor/paper/hygiene/other 各很少。top verbs 中 `place` 约 10 clips、`add` 8、`show` 6、`season` 5，`arrange/hold/pick/pour` 各约 4，`present/remove` 约 3，`insert/slice` 约 2。

这说明子集有容器/工具偏置，且 `show/present/add/season` 并不都是需要精确机器人接触动力学的原子 skill。106 clips 不足以验证开放世界长尾，也没有 train/val/test、creator/video/object disjoint split。

### 8.2 数据源与采集硬件

| 项目 | EgoInfinity 的回答 |
| --- | --- |
| 新采集硬件 | 无；消费级互联网 RGB 视频 |
| 相机 | 未统一；但筛选目标是约静态相机 |
| depth/IMU/Mocap | 无；全部从 RGB 估计 |
| calibration | 不要求已知 intrinsics；MoGe-2/GeoCalib 估 focal/metric/gravity |
| participant protocol | 本文没有；继承公开视频创作者的自然/教程行为 |
| annotation protocol | 继承 Action100M descriptions/captions，自动 object prompt 和 4D projection |
| consent/privacy | 本文未重新收集 consent；衍生发布依赖 Action100M 的治理与非商用许可 |
| raw-to-accepted conversion | 未报告 |

“无需 wearable”降低专用硬件成本，却不是无成本数据：内容平台获取、视频版权/许可、模型推理、gated weights、GPU、人工开发集阈值与失败审查构成新的 TCO。

### 8.3 Collection selection bias

两遍筛选和 pipeline operating envelope 会系统性排除：

- 手很小、被遮挡、戴手套或 motion blur 的行为；
- handheld/head-mounted/dynamic camera；
- 不依赖可见手的 loco-manipulation、脚/身体接触；
- 柔性物、透明/反光对象、铰接物、非刚性工具接触；
- 文本未提及或 SAM3 无法发现的对象；
- 快速、短于 8 帧的 contact，以及存在真实 2 秒内 release/regrasp 的操作。

最终数据更像“静态机位、手清楚、刚性物、短时教程”的成功域。这种 curated distribution 对工程是必要的，但不能用上游 127K 小时的多样性替代对实际 accepted subset 的统计。

### 8.4 数据格式与当前发布

当前 HF preview 给每个 sample 的对象资产和 retarget 子目录。对 104 clips，四类机器人目录包含：

| File | 内容 |
| --- | --- |
| `input_viz.mp4` | retargeter 使用的 ego input 可视化 |
| `robot_sim.mp4` | robot simulation replay |
| `trajectory.npz` | `q_left`、`q_right`、finger joints、joint names、fps、clip id |
| `torso_frames.npz` / repo 文档中的 `root_frames.npz` | per-frame $R,t$ 与 anchor frame |
| `metrics.npz` | IK rate、position/orientation errors、joint-limit margin、manipulability、roughness |

发布文档存在 `torso_frames.npz` 与 `root_frames.npz` 命名差异；用户应以下载 revision 的实际 manifest 为准。每 clip 29–734 frames，fps 不统一，训练前不能默认固定 15/30 Hz。

![任意视角原视频与虚拟 ego 4D 重建的 17×6 画廊](https://arxiv.org/html/2606.17385v2/figures/app_C_grid_17x6.jpg)

## 9. 实验：论文真正测到了什么

实验有四块：browser/data inspection、curated subset statistics、simulation retarget metrics、real-robot qualitative execution/policy。没有独立 perception GT 和算法消融。

![实验总览：浏览器、四本体重定向、LEAP policy 与 FR3 真机技能](https://arxiv.org/html/2606.17385v2/experiments.png)

### 9.1 论文表 2：三本体重定向

| Robot | IK rate ↑ | Position error ↓ | Orientation error ↓ | Joint-limit margin ↑ | Manipulability ↑ | Smoothness ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Unitree G1 | 0.821 | 2.86 cm | 6.73° | 0.619 rad | 0.012 | 0.00693 |
| Robonaut2 | 0.774 | 6.67 cm | 8.25° | 0.134 rad | 0.058 | 0.00343 |
| Dual-Franka | 0.706 | 10.27 cm | 12.17° | 0.572 rad | 0.080 | 0.00582 |

`IK rate` 是逐帧 solver success；position/orientation error 是 target hand pose 与 achieved pose 的平均残差；smoothness 是 mean squared joint velocity $\dot q$。Manipulability 定义为：

$$
m(q)=\sqrt{\det(JJ^\top)}.
$$

注意不同机器人 Jacobian 尺寸、单位与 joint structure 不同，manipulability 的 raw 数值不宜直接当跨机器人能力排名。G1 的 IK rate 最好，Dual-Franka 最差且平均 10.27 cm position error；这对精确 grasp/contact 已经很大。论文没有报告 median/tail、per-task、per-view、single-vs-bimanual 和 interpolation 后的 constraint violation。

### 9.2 真机直接 replay

Dual-Franka FR3 展示 Cut、Pour Bowl、Pour Glass、Wipe Box、Wipe Computer 五条时间序列。它说明 motion compilation 可以到 hardware，不说明每类成功率或鲁棒性。

![Dual-Franka FR3：切、两类倾倒和两类擦拭的时间序列](https://arxiv.org/html/2606.17385v2/C4.png)

缺失的最小报告包括：每技能 trials/success、初始物体 pose randomization、容器实际液体/替代物、刀具接触标准、open-loop/closed-loop、速度/加速度/碰撞/力阈值、failure categories。没有这些量，filmstrip 只能算 existence proof。

### 9.3 LEAP downstream policy

论文称 EgoInfinity-extracted hand motions 作为 prior，训练 real LEAP dexterous hand grasp policy；图中 apple、banana、tomato can 各展示三条 rollout。没有写 policy architecture、robot demonstrations、objective、prior injection、训练量、baseline 和 success rate。

![LEAP hand：apple、banana、tomato can 的三组抓取 rollouts](https://arxiv.org/html/2606.17385v2/figures/C5.jpg)

因此可以说“作者展示了 motion prior 可接入 learned policy”，不能说“证明 EgoInfinity data 提升抓取泛化”。要建立因果，至少需要 robot-only、human-prior-only、joint training、matched-compute random-human 和 no-interaction-refinement 五组。

### 9.4 端到端十条 gallery 与交互浏览器

附录用 10 clips 展示原 exo、虚拟 ego、dual-Franka、G1、Robonaut2、XLeRobot 和两类 real platforms；browser 提供 hand trajectory、object mesh/cloud/bbox、state、camera cone、raw/mask/depth/flow 及 retarget panel。它们对 failure diagnosis 很有价值，也会产生 selection bias：gallery 没有系统展示所有失败或随机样本。

![10 条 video-to-robot end-to-end pipeline gallery](https://arxiv.org/html/2606.17385v2/figures/C3.jpg)

![交互式浏览器：clip list、Viser scene、中间量与 track summary](https://arxiv.org/html/2606.17385v2/C2.png)

## 10. 语义—Human Motion—Robot Action—Contact 是否真正对齐

本文建立的是一条**串行弱对齐链**：

$$
\text{text}
\rightarrow \text{object mask}
\rightarrow \text{human hand/object 4D}
\rightarrow \text{robot root/IK}
\rightarrow \text{joint candidate}.
$$

各节点没有被同一个 physical outcome loss 联合训练。语义来自上游 annotation，4D 来自多个 frozen perception models，contact 是阈值状态，robot action 是 simulator-trained root model + IK。最终真实任务结果没有反向校正 object discovery、contact state 或 human-to-robot mapping。

### 10.1 五种“对齐”的完成度

| 对齐 | 完成度 | 证据 |
| --- | --- | --- |
| text ↔ object | 中 | caption noun 驱动 SAM3；无检测准确率 |
| RGB ↔ metric hand/object | 中低 | 统一坐标和定性重建；无 GT benchmark |
| hand ↔ object interaction | 中低 | state-conditioned rigid bind；无精确 contact GT |
| human motion ↔ robot kinematics | 中高 | 三本体 IK metrics、四本体发布 |
| robot kinematics ↔ task success | 低 | 定性 hardware demos，无统计/闭环因果 |

### 10.2 Contact intent 与 observed contact

EgoInfinity 的 grasp state 是 **contact likelihood + motion ownership**：它判断“此时应让手驱动物体”。它既不是观测到的接触面，也不是任务想要的 functional contact goal。

一个更完整的 contact representation 至少应包括：

$$
g_t=(o,\mathcal C_t,\mathbf n_t,\mu_t,\mathbf f_t,
\text{phase},\text{stick/slip},\text{uncertainty}),
$$

其中 $\mathcal C_t$ 是接触 patch，$\mathbf n_t$ 是法向，$\mu_t$ 是摩擦，$\mathbf f_t$ 是 wrench/force。EgoInfinity 只近似给 `object + phase + dominant hand`，其余缺失。

### 10.3 Grasping goal 应怎样从数据中提取

建议将 human state 压成三层目标，而不是直接逐关节模仿：

1. **Object-centric approach**：wrist path、palm orientation、pre-grasp aperture。
2. **Functional contact**：目标 object part、contact topology、grasp type、允许的姿态等价类。
3. **Outcome constraint**：lift/pour/cut/wipe 后的 object state 和 failure detector。

EgoInfinity 能较好提供第 1 层和粗略 phase，第 2 层只有弱 proxy，第 3 层几乎没有结构化监督。

## 11. Embodiment、Task 与 Reality 三条 Gap

### 11.1 Embodiment Gap

根坐标 estimator 解决的是“把 human wrists 放进 robot reachable workspace”，不是完整 morphology equivalence。每种机器人单独训练，finger mapping 单独设计；新本体需要 URDF/MJCF、joint groups、end-effector frames、limits、IK tolerances、checkpoint 或重新训练。

主要残差：

- human shoulder/elbow/body 不可见，torso frame 本质多解；
- 双手相对 pose 可达，不代表无 self-collision 或维持 humanoid balance；
- human fingers 与 gripper/dexterous topology 不同；
- object-hand rigid bind 没有转换成 robot-object contact constraint；
- workspace fit 可能破坏原 task-relative geometry。

### 11.2 Task Gap

短 clip 和 caption 给 action verb/object，却没有 long-horizon precondition、subgoal graph、failure/recovery 或 completion predicate。`place/pour/wipe` 的语义可帮助 retrieval，不能保证机器人知道放到哪、倒多少、擦干净到什么程度。

Task gap 的合理接口是：

$$
\text{language + object state}
\rightarrow \text{subgoal/contact goal}
\rightarrow \text{feasible motion candidates}
\rightarrow \text{closed-loop outcome check}.
$$

EgoInfinity 主要提供中间两项的先验，缺完整 state transition 与 outcome annotation。

### 11.3 Reality Gap

输入是现实世界 human video，不代表输出自动拥有 robot dynamics truth。人手顺应性、摩擦、肌肉控制、触觉 correction 与工具熟练度没有被记录；仿真只用于 root-frame model 的随机 trajectory generation，未做 contact dynamics 或 system identification。

Reality gap 包括：

- mass/inertia/friction/compliance 未知；
- actuator latency、backlash、torque/velocity limits 未进 human trajectory；
- camera-to-robot calibration 与 execution observation 不同；
- forceful、柔性、液体、切割和擦拭依赖 contact dynamics；
- open-loop replay 无法处理 object displacement 和 grasp failure。

## 12. System 2 / System 1 / System 0 的正确分工

| Layer | Input | Output | EgoInfinity 能提供 | 必须补的数据 |
| --- | --- | --- | --- | --- |
| System 2 | language、scene/object memory | task graph、subgoal、contact intent | caption、object list、short action verb | long-horizon state/outcome、failure/recovery |
| System 1 | object/robot state、subgoal | wrist/finger/action chunk + uncertainty | human 4D prior、root candidates、IK trajectory | target-robot teleop、closed-loop observation/action |
| System 0 | desired contact/EEF、joint/tactile state | torque/position/impedance command | 几乎无直接监督 | force/tactile、system ID、WBC、safety controller |

不要把 15/30 FPS human motion 直接当 System 0 command。合理做法是让它成为 System 1 candidate，经过 reachability/contact/collision 过滤，再由 100–1000 Hz controller 吸收动力学和扰动。

### 12.1 动作时序与 chunk

论文训练 root estimator 用固定 2 秒/60 帧 synthetic windows，真实发布 clip 的 fps 不统一、长度 29–734 frames。直接混合会让同样的 frame horizon 对应不同物理时间。下游必须以秒重采样，并保留：

- `timestamp` 与原 fps；
- observed vs infilled mask；
- grasp/state boundaries；
- IK-converged vs interpolated mask；
- per-frame geometry/confidence；
- motion derivative 和 robot limit normalization。

### 12.2 Curriculum 建议

| Phase | 数据 | 目标 | 进入下一阶段的门槛 |
| --- | --- | --- | --- |
| P0 | raw/4D human | object/hand tracking、phase recognition | 人工 GT 子集上的误差与校准 |
| P1 | high-confidence EgoInfinity | wrist/object motion prior、latent action | held-out creator/object/view generalization |
| P2 | retarget simulation | reachability、collision、contact planning | per-frame + segment feasibility；无插值硬失败 |
| P3 | small robot teleop | embodiment/action grounding | robot-only vs human+robot 因果增益 |
| P4 | real closed-loop | dynamics/contact/recovery | task SR、failure severity、OOD robustness |

## 13. Human–Robot co-training：这篇论文没有做，但数据如何用最合理

EgoInfinity 不是 co-training paper。LEAP motion prior 例子没有完整训练细节，不能代表 universal human–robot joint training。建议使用共享 object/task latent、分开的 action decoder：

$$
z_t=f_\theta(o_{\le t},l),\qquad
\hat a_t^h=g_h(z_t),\qquad
\hat a_t^r=g_r(z_t,s_t^r),
$$

human branch 监督 hand/object future，robot branch 监督可执行 action；以 object/contact outcome 而非 raw joint equality 做 cross-domain alignment。

### 13.1 必须做的因果矩阵

| Group | Human data | Robot data | Matching | 目的 |
| --- | ---: | ---: | --- | --- |
| R | 0 | 固定 | — | robot-only baseline |
| H→R | 固定 | 固定 | task/object matched | 测预训练增益 |
| H-rand→R | 同量 | 固定 | random human | 测相关性而非 token 数 |
| H-4D→R | 同量 | 固定 | metric HOI | 测 projection 增益 |
| H-2D→R | 同量 | 固定 | 2D tracks | 测 4D 相对 2D |
| H-no-state→R | 同量 | 固定 | 无 refinement | 测 interaction state 的贡献 |
| H+R joint | 固定 | 固定 | shared latent/separate head | 测联合训练 |

统一 optimizer steps、总 tokens、robot samples、seeds 和 validation schedule；按 source video/creator/object/scene/task 分组隔离，不能让同一 YouTube 视频的相邻 segment 跨 split。

### 13.2 Latent alignment 的安全边界

应该对齐 task-relevant invariant：object-relative wrist trajectory、approach direction、grasp phase、object displacement 和 outcome；不应强行对齐 human/robot joint angles、绝对 camera appearance 或 finger topology。建议 latent 中显式保留 domain token 和 uncertainty，避免 model 把 reconstruction artifacts 当 robot action truth。

## 14. 数据质量、uncertainty 与缺失评测

论文通过 robust statistics、sanity gates 和浏览器提高可审计性，却几乎没有 calibration 数字。每个输出应至少带以下 provenance：

| Field | 建议取值 |
| --- | --- |
| hand source | detected / WiLoR / infilled / smoothed |
| depth source | MoGe / Flow3R / fallback |
| object source | SAM3 detect / SAM2 propagate / SAM3D reconstruct |
| pose trust | static centroid / moving vision / hand bind / FP++ bake |
| interaction confidence | overlap、finger distance、wrist distance、temporal support |
| retarget trust | sampled candidate、IK converged、interpolated、limit margin |
| release status | public preview revision、license、schema version |

当前数据卡公开 metrics files，但论文没有展示 uncertainty calibration 或按 confidence 过滤后的 task performance。用 fixed threshold 生成的 hard state 应在训练时转成 soft/interval uncertainty，特别是 grasp boundary。

### 14.1 最缺的四个 benchmark

1. **Perception GT**：手关节/mesh、depth、object 6D、scale、gravity，按遮挡/反光/视角分层。
2. **Interaction GT**：grasp onset/offset、dominant hand、object motion、contact patch 与 slip。
3. **Retarget GT**：root-frame error、IK/collision/dynamic feasibility，与 heuristic/deterministic regression 比较。
4. **Robot causal test**：matched data/compute 的成功率、OOD、recovery 和 safety。

## 15. 论文—代码—数据的一致性审计

### 15.1 静态代码审计

官方仓库有一个公开提交，约 236 MB shallow checkout；包含约 39.7K 行 Python/shell、自研 CLI/stage registry、pipeline、filter UI、四机器人 retarget training/inference、四个 checkpoint、机器人资产和 CI。CI 只做 package metadata 与 minimal import smoke；Ruff 因约 320 个既有 style violations 被注释，未见论文级单元/数值回归测试。

积极面：

- `process/run/status/filter/import` 统一入口，支持 stage resume、force/cascade、batch failure isolation；
- 文档覆盖 pipeline、architecture、manifest、multi-host、third-party license；
- retarget training/test scripts 和四权重真实存在；
- HF preview 非 gated，可直接检查数据产品而无需下载全量。

限制面：

- 完整 pipeline 依赖多环境、gated/manual models 和 GPU；
- paper method 和 default release config 存在 Flow3R/FP++/internal tracker 差异；
- repository classifier 是 Alpha，retarget checkpoint 明示 preliminary；
- 没有论文 106 clips 的固定 evaluation script/seed/report regeneration 指南；
- LEAP policy 和 FR3 hardware controller/experiment harness 未见完整公开训练栈。

### 15.2 论文与当前代码的接口漂移

| 部分 | 论文表述 | 当前仓库/文档 | 复现建议 |
| --- | --- | --- | --- |
| depth | MoGe-2 metric + Flow3R dense depth | Flow3R opt-in、default off | 固定 `flow3r_depth.enabled` 并报告 |
| object 6D | FoundationPose++ tracks trajectory | internal FGR/ICP + flow/PnP + LBFGS；FP++ optional rotation bake | 保存 stage state/config 与 FP++ 是否启用 |
| SAM3 | semantic prompt detection | SAM3.1 gated sibling env | 记录 exact upstream revision/weight |
| output root file | appendix/data card `torso_frames.npz` | retarget README `root_frames.npz` | 用 schema/manifest 而非硬编码文件名 |
| hardware | 论文真实 FR3/LEAP | 主仓库主要开源 perception + retarget | 不把完整真机结果列为一键可复现 |

这不必解释为结果有误，更可能是开发中 pipeline 演进；但 scientific reproducibility 需要明确 paper configuration tag，而当前 repo 只有 initial commit、无 release tag。

### 15.3 复现等级

| Level | 目标 | 当前可行性 | 主要阻塞 |
| --- | --- | --- | --- |
| L0 | 查看论文、项目页和 preview | 高 | 无 |
| L1 | 下载一条已处理样本、读 4D/retarget outputs | 高 | 非统一 fps/schema 命名需注意 |
| L2 | 用 bundled checkpoint 对 example retarget | 中高 | MuJoCo/JAX/PyTorch/robot deps |
| L3 | 对自有静态 RGB 跑完整 4D pipeline | 中低 | MANO 注册、SAM3 gated、SAM3D/FP++ 多环境、GPU |
| L4 | 重建 106-clip paper metrics | 中低 | 固定 manifest/config/eval script 和部分原数据获取 |
| L5 | 复现 FR3/LEAP 结果 | 低 | hardware stack、policy recipe、trials/config 未完整发布 |
| L6 | 处理 127K h | 未证明 | corpus access、compute/storage、failure/QC economics |

## 16. 面向 Humanoid、EX002 与 Dexterous Hand 的落地建议

### 16.1 Humanoid

G1 结果只覆盖 bilateral arm/hand retargeting，不是 whole-body loco-manipulation。缺 head/gaze、pelvis/base、legs、feet contacts、CoM、support polygon 和 WBC。建议：

- System 2 从 caption/object state 产 task/subgoal；
- System 1 用 EgoInfinity 双腕/手势作为 object-centric targets；
- root estimator 只提供 torso/base proposal，不直接设 floating base；
- whole-body optimizer 联合 reach、balance、self-collision、visibility；
- System 0 用真实 humanoid dynamics、force/tactile 和 safety shield。

最先做固定站立桌面 bimanual `place/pour/wipe`，再逐步加入一步 reach 和 base locomotion；不要由本文 G1 IK rate 推断动态平衡能力。

### 16.2 EX002

论文没有 EX002 实验，本笔记不假定它的 DoF、手型或相机布局。需要先建立平台契约：

| Contract | 必需信息 |
| --- | --- |
| Kinematics | URDF/MJCF、joint groups/limits、base/torso/end-effector frames |
| Observation | head/wrist camera intrinsics/extrinsics、latency、robot proprioception |
| Action | Cartesian/joint/latent chunk、control rate、gripper/hand topology |
| Safety | self/environment collision、workspace、force/torque、E-stop |
| Data bridge | object frame、human wrist mapping、robot teleop schema、confidence masks |

执行路线：先训练 EX002-specific root estimator 或直接用 workspace optimization；把 human MANO 压成 palm pose + grasp phase + object-relative goal；以 30–50 条 target-robot demonstrations 校正每任务 motion prior；先做刚性、低速、短时任务。若 EX002 是夹爪，不应逐 finger imitation；若是双臂，必须保留 bilateral relative constraint，而不能只用论文 dominant-hand object bind。

### 16.3 Dexterous hand

LEAP 示例说明 MANO prior 有接口价值，但没有接触因果。建议将 EgoInfinity 用于：

- pre-grasp shape/aperture 与 approach orientation；
- finger synergy 初始化和 motion proposal；
- object category/part conditioned grasp retrieval；
- simulation exploration 的 curriculum/reward prior。

必须补：object mesh/contact patch、collision/penetration、force closure、tactile slip、hand-object dynamics、真实 joint calibration。对 in-hand rotation、regrasp、tool pivot，论文 rigid bind 反而会删掉所需信号，应保留 raw visual pose 和多假设，不要只训练 refined trajectory。

## 17. 采集经济性、扩展能力与 TCO

EgoInfinity 把“招募、配发设备、进实验室”的采集成本换成“语料授权、GPU 推理、依赖治理、失败筛选、派生存储”。成本结构可写为：

$$
C_{\mathrm{usable\ hour}}=
\frac{C_{\mathrm{access}}+C_{\mathrm{filter}}+C_{\mathrm{models}}+
C_{\mathrm{4D}}+C_{\mathrm{QC}}+C_{\mathrm{storage}}}
{H_{\mathrm{raw}}\,r_{\mathrm{filter}}\,r_{\mathrm{recon}}\,r_{\mathrm{retarget}}}.
$$

论文没有给任何 $r$ 或成本项，所以不能验证比 teleoperation 便宜多少。SAM3D/Flow3R/FP++ 是重模型，多环境和许可证也增加人员成本。真正的 scale claim 应至少报告每模块 seconds/frame、GPU memory、failure/skip rate、object yield、GB/input-hour、human QC minutes/hour 和全部 127K h 的估算账单。

### 17.1 Web corpus 的长处

- 无需为每个任务重新设计实验室 protocol；
- tutorial 中有自然 pedagogical structure、语言和物体多样性；
- arbitrary shot size 能覆盖 partial body；
- 新 perception module 可重跑生成更好派生数据。

### 17.2 Web corpus 的经济/法律约束

- 原视频删除、地区限制、cookie/auth 和 URL decay；
- noncommercial license 阻止直接产品化；
- creator/旁观者 privacy 与派生资产撤回链；
- 大量失败 clip 的 silent compute waste；
- 模型升级会改变 schema/distribution，需要 versioned rebuild。

## 18. 隐私、许可与治理

本文没有专门 ethics/data governance section 来解释公开视频人物、家庭环境、屏幕、旁观者、儿童、地理/文化偏差和撤回机制。Action100M 的 FAIR Noncommercial Research License 给出使用边界，但不能替代下游数据卡中的：来源平台、consent/legal basis、face/screen/voice handling、creator deletion propagation、derived mesh/trajectory re-identification 和 model unlearning policy。

4D hand motion 本身可能成为 biometric/behavior signature；即使只发布 hand/object render，source URL/clip ID 和 scene geometry 仍可能重识别。建议发布：

1. content-source 与 jurisdiction 分布；
2. face/voice/text/PII 处理规则；
3. takedown endpoint 和 source-deletion sync；
4. derived data lineage/hash；
5. demographic/device/task performance audit；
6. training checkpoint 的 deletion propagation policy。

## 19. Related work 的准确定位

### 19.1 数据基础设施

EgoVerse/Open-AoE 通过专门采集获得更可控的 ego distribution、相机/手 trajectory 与平台治理；EgoInfinity 用现成 web corpus 换取低前端采集成本和广内容，几何真值更弱。二者最合理的组合是：curated ego data 校准 perception，web engine 扩 coverage，robot data 做 execution grounding。

### 19.2 Human-to-robot projection

Phantom/H2R/Masquerade/MimicDreamer 优先缩小 robot observation appearance gap；EgoInfinity 优先缩小 3D action/geometry gap。前者可能 hallucinate pixels，后者不会生成 photoreal robot view。理想 pipeline 可把 EgoInfinity 的 metric hand/object state 和 robot trajectory 作为几何条件，再做 robot visual synthesis，并用 quality filter 验证一致性。

### 19.3 Robot demonstration datasets

DROID/Open X 等真实记录 robot action/proprioception/outcome，分布窄且采集贵；EgoInfinity 覆盖 human behaviors，却只有 action candidate。Human prior 和 robot trajectory 是互补的 supervision levels，不能按“小时”直接排名。

## 20. 作者没有完成但最值得做的未来工作

1. 发布 paper configuration tag、完整 manifest、checksum、schema 和 evaluation script。
2. 报告 127K h processing 的 throughput、GPU/storage/TCO、过滤与重建 yield。
3. 为 hand/depth/object 6D/contact 建人工或传感器 GT benchmark。
4. 做 state refinement 全消融：visual-only、static lock、grasp bind、depth realign、FP++。
5. 报 root estimator 对 heuristic、deterministic regression、non-equivariant、no-flow 的比较。
6. 报 per-task/per-view/per-length IK tail、collision、velocity/acceleration/dynamic feasibility。
7. 给 FR3/LEAP 完整 trials、baseline、success criteria、failure modes 和 confidence interval。
8. 做 matched-compute human–robot co-training，证明 4D projection 相比 raw/2D 的增益。
9. 从 rigid grasp state 升级到 contact patch、slip、articulation、compliance 和 uncertainty。
10. 支持 moving-camera/head-mounted videos，以 SLAM/dynamic scene reconstruction 放宽 selection bias。
11. 生成 geometry-conditioned robot visual observation，连接 action gap 与 appearance gap。
12. 完善 privacy lineage、takedown、derived deletion 与 commercial licensing 路线。

## 21. 十组关键问题的直接回答

### Q1. 127K 小时是不是 EgoInfinity 已发布的机器人数据？

不是。它是 Action100M 可寻址上游视频时长。论文实际处理/统计 106 clips；HF 当前是 5.49 GB preview，104 clips 有四本体 retargeting。

### Q2. “任意视角”是否包括头戴式或移动相机？

不应这样理解。视角/shot size 可任意，但当前方法明确假设 approximately static camera，并在筛选时使用 camera-motion cue。

### Q3. Exo-to-ego 是否生成了真实第一视角 RGB？

没有。它在恢复的 3D hand/object geometry 上移动虚拟相机并重渲染，是 geometry-space reframing，不解决背景、纹理、光照与目标机器人外观。

### Q4. Contact state 是不是精确接触真值？

不是。它是 mask overlap/距离/运动阈值得到的粗状态，用于决定 pose trust。没有 contact point、normal、force、tactile、slip 或 exact fingertip alignment。

### Q5. Interaction-aware refinement 是否被定量证明？

没有充分证明。设计合理、可视化有说服力，但论文缺 tracking/contact GT、drift metric 和组件消融。

### Q6. 多机器人是否意味着 zero-shot universal retargeting？

不是。表示是 agent-agnostic，但 root model、finger mapping 与 IK 配置是 robot-specific；每个机器人独立训练，新的本体可能需重训/标定。

### Q7. IK rate 是否等于任务成功率？

不是。它只测逐帧 kinematic solver convergence。碰撞、接触、动力学、物体 outcome 和闭环 recovery 不包含在内。

### Q8. 论文是否证明 human video 提升 robot policy？

没有严格因果证据。LEAP 展示 motion-prior policy rollouts，但无 robot-only baseline、训练细节、trial success 或统计。

### Q9. 开源程度如何？

工程开放度高：代码、文档、四权重、5.49 GB preview 和 browser 都存在。完整复现度中等：多环境、gated/manual assets、配置漂移；真机/LEAP 复现度低；商业可用性受非商用与混合许可限制。

### Q10. 最优先的落地方式是什么？

把高置信 4D HOI 当作 System 1 motion prior/retrieval source，以目标机器人少量 teleop 做 action grounding，以 contact-aware planner 和 System 0 闭环控制筛选；从刚性、低速、短时桌面任务开始。

## 22. 评分卡

| 维度 | 评分（1–5） | 证据 |
| --- | ---: | --- |
| 可寻址语料规模 | 5 | Action100M ≈127K h、147M segments |
| 实际 processed/released scale | 2 | 论文 106 clips；发布 preview、104 retarget clips |
| 自动化设计 | 4 | 两遍筛选、语义 object discovery、统一 pipeline；裸视频 object list 仍需来源 |
| 4D representation | 4 | hand/object mesh/pose/cloud/state，接口丰富 |
| 度量/感知准确性证据 | 2 | 无独立 GT benchmark |
| Interaction refinement | 3 | 状态条件 trust 很合理；硬阈值多、无消融/精确 contact |
| Cross-embodiment | 4 | 四本体资产、三本体 metrics；仍 robot-specific |
| Robot 可执行性 | 3 | IK + FR3/LEAP existence proof；缺 task SR/动力学统计 |
| Contact/physics | 1 | 无力/触觉/接触 patch/slip，rigid bind 是 prior |
| 开源工程 | 4 | 39.7K 行、文档、权重、preview；依赖复杂 |
| 可复现性 | 3 | retarget 中高；full 4D 中低；hardware 低 |
| 隐私/治理 | 2 | 非商用许可明确；细粒度治理与撤回链未报告 |
| 业务价值 | 4 | data mining、motion prior、retargeting 强；不可直接商用/替代 robot data |

### 最终判词

> EgoInfinity 以 **Action100M 中约静态相机、可见手且有语义描述的互联网短视频** 为输入，用多个 foundation models 与 interaction-conditioned geometric priors 恢复 **metric hand/object 4D state**，再用 **robot-specific SE(3)-equivariant root estimator + IK** 编译到多种机器人；其最可靠的监督是 **object-relative wrist/hand motion、刚体物体 pose 与粗 interaction phase**，最缺失的是 **带真值的感知误差、精确 contact/force、动力学和真实任务 outcome**。因此它最适合 **高覆盖 motion prior、跨本体候选生成和 video-to-action 研究**，不应被直接称为 **127K 小时已生成的可执行机器人数据**。

这篇论文真正有价值的地方，不是把 web 视频 magically 变成无噪声 robot trajectories，而是提出一个清楚的编译边界：先把人类动作投影成机器人无关的几何状态，再让每个本体承担自己的可达性和动作解码。下一步科学问题不是继续放大上游小时数，而是量化每一道 projection error，并让真实接触结果反向校正整条链。
