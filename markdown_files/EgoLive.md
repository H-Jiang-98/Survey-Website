---
title: "EgoLive: A Large-Scale Egocentric Dataset from Real-World Human Tasks"
method_name: "EgoLive"
authors: [Yihang Li, Xuelong Wei, Jingzhou Luo, Yingjing Xiao, Yibo Bai, Guangyuan Zhou, Teng Zou, Chenguang Gui, Jiajun Wen, He Zhang, Kangliang Chen, Xing Pan, Shuaiyan Liu, Daming Wang, Tao An, Jiayi Li, Shibo Jin, Wanwan Zhang, Tianyu Wang, Boren Wei, Zhixuan Huang, Fangsheng Liu, Ruodai Li, Hui Zhang, Anson Li, Yicheng Gong, Peng Cao, Jiaming Liang, Liang Lin]
year: 2026
venue: arXiv
tags: [egocentric-data, real-world-tasks, stereo-vision, imu, hand-reconstruction, depth-reconstruction, instruction-captioning, human-to-robot, humanoid, dexterous-manipulation, dataset-governance]
image_source: online
---

# EgoLive：1680 小时真实工作流，还是尚未闭环的机器人数据基础设施？

> 本笔记基于 [arXiv:2604.23570v1](https://arxiv.org/abs/2604.23570)、[arXiv HTML](https://arxiv.org/html/2604.23570v1)、17 页[官方 PDF](https://arxiv.org/pdf/2604.23570)和论文给出的[京东机器人数据市场入口](https://robotdata-market.jdcloud.com/console/market)逐页精读。论文共有 11 张图、3 张表，没有 supplementary，也没有显式方法公式。公开状态核验日期为 **2026-08-13**；凡是登录后才可能看到、论文未报告或无法匿名下载核验的信息，均明确标注。

## 阅读结论先行

EgoLive 把 Ego 数据采集从实验室桌面推到家庭服务、零售、药房和物流式真实工作流。它报告 1680 小时、65,866 个 episode、346 类任务；每个 episode 通常持续 1–3 分钟，按总时长反推平均约 91.8 秒。定制 JoyEgoCam 同步采集双目 2160×2160@60 Hz RGB 与 200 Hz IMU，130°×130° 视场明显优于单目低帧率采集，理论上更适合快速手部运动、metric depth 与 camera ego-motion。

它的自动管线覆盖三条信息链：HaMeR/MANO 单目初始化后做双目手部优化；ORB-SLAM3 融合双目与 IMU 恢复相机轨迹；FoundationStereo 生成 1152×1152 depth。语义侧用 hand-object detector、BoT-SORT、SAM2 和 fine-tuned Qwen3-VL-32B 生成手/交互物体 mask、1–20 秒子任务边界及包含 hand–object–action 的描述。就“原始观测—人手状态—粗世界几何—任务语义”而言，EgoLive 的潜在信息密度很高。

但机器人价值尚未由下游实验证明。论文没有训练任何 VLA、policy、world model 或 retargeter；没有 robot-only/human-only/co-training 对照、真机成功率、跨本体执行率或 scaling curve。物体只有交互 mask 和语言类别，没有统一 6D pose、mesh、articulation 或状态轨迹；接触、力、触觉、抓取 phase、whole-body/floating-base/foot pose 均未提供。因此数据可以成为视觉、工作流语义、手部运动与场景几何的预训练资产，却不是可直接执行的 robot action 数据。

“高精度标注”也需要分模态看。Depth 在 ChArUco 标定板的 0.5–3.5 m 范围有完整数值：0.5–0.7 m mean error 约 3.0 mm，0.9 m 为 5.381 mm，3.5 m 增至 18.035 mm。手部只有与图像/同一深度管线点云对齐的定性图，没有独立 Mocap ground truth、MPJPE 或缺失率；caption 只列 4 个样例和 LLM-as-a-Judge 的判定，没有全库准确率。把深度精度外推为所有模态的“industrial-grade accuracy”并不成立。

最后，论文把 EgoLive 称为 open-source，但官方入口匿名访问会跳转京东云登录；未发现官方 GitHub、Hugging Face 或 ModelScope 镜像，也未发现公开数据卡、文件 manifest、数据 schema、下载体量或数据许可证。arXiv 的 CC BY 4.0 只证明论文文本/图片许可，不能自动授权数据。当前最准确的公开状态是：**论文宣布并提供受账户入口的数据资产，完整 1680 小时是否可匿名下载、以何种条款使用，尚不能公开审计。**

### 一句话总结

EgoLive 用高规格双目+IMU头戴设备把 1680 小时真实服务/零售工作流加工成相机、手部、深度、mask 和层级语言伪标签，但尚未给出完整可匿名核验的开放发布、独立手/语言精度、接触物理标签或任何机器人下游收益。

### Elevator pitch

机器人遥操作能给动作真值，却难覆盖真实世界长尾；普通 Ego 视频能扩规模，却缺少 metric geometry 和细粒度任务结构。EgoLive 选择在采集端加双目、IMU、高分辨率和高帧率，在处理端统一恢复 camera–hand–depth–semantic 信号，再把家庭清洁、整理、零售和药房等长工作流切成原子子任务。它是一座潜力很大的 human experience 库，但论文目前只证明“能采、能标、深度在标定板上较准”，没有证明“能让机器人做得更好”。

![EgoLive 总览：346 个任务、1680 小时、约 65K episodes 的双目真实工作流](https://arxiv.org/html/2604.23570v1/figs/cover.png)

## 0. 资源、公开状态与许可

| 资产 | 正式入口 | 截至 2026-08-13 的实际可见内容 | 许可/条款 | 可复现程度 |
| --- | --- | --- | --- | --- |
| 论文 | [arXiv 摘要](https://arxiv.org/abs/2604.23570)、[HTML](https://arxiv.org/html/2604.23570v1)、[PDF](https://arxiv.org/pdf/2604.23570) | v1，17 页；正文 13 页，11 图、3 表；无 supplementary | arXiv 标注 CC BY 4.0，仅覆盖论文材料 | 高 |
| 数据入口 | [JD Cloud Robot Data Market](https://robotdata-market.jdcloud.com/console/market) | 普通浏览器请求重定向到京东云登录；匿名页面无 EgoLive 文件/规模/条款详情 | 未发现公开数据许可证 | 登录后状态未知；匿名不可审计 |
| 项目页 | 论文未给独立项目页 | 未发现 | — | 无 |
| 数据卡/镜像 | 未发现官方 HF/ModelScope 条目 | 无公开 manifest、schema、revision、checksum 或 sample browser | 未发现 | 无法核验 |
| 采集硬件 | 论文 Figure 2 与规格 | JoyEgoCam 外观、双目、IMU、FOV；无 BOM/CAD/固件/同步板设计 | 未报告 | 只能近似复刻 |
| 标注代码 | 未链接 | HaMeR、MANO、ORB-SLAM3、BoT-SORT、SAM2、FoundationStereo、Qwen3-VL-32B 的组合描述 | glue code/权重/配置/模型 prompt 未给 | 低 |
| 处理后标注 | 论文声称与数据一同提供 | camera pose、3D hand keypoints、depth、hand/object masks、sub-task、caption | 文件格式、单位、坐标约定、缺失 mask 未公开说明 | 无法匿名核验 |
| Robot baseline/checkpoint | 未提供 | 论文无下游策略实验 | — | 不存在可复现基线 |
| Retarget/sim/真机接口 | 未提供 | 只在 related work/愿景中讨论 | — | 无 |

### 0.1 “Open-source” claim 的限定条件

论文写的是“to our knowledge, the largest open-source annotated egocentric dataset for real-world human tasks”。这个 claim 至少有四层限定：

1. 不是所有 Ego 数据，而是 **annotated + real-world human tasks**；Egocentric-10K 名义 10k h 更大，但论文认为其标注稀疏。
2. 1680 h 是论文声明规模；公开入口需要账号，无法匿名确认全部数据是否可下载。
3. “open-source”没有配套代码仓库、公开数据卡或命名数据许可，法律与工程含义不完整。
4. 论文比较表的模态勾选依据“main public release”，但 EgoLive 本身当前公开入口的具体文件不能匿名检查，表中自我声明强于可观察证据。

因此，笔记后文将“论文宣称提供”与“匿名公开可核验”分列，不把两者合并。

## 1. 论文速览与核心命题

| 项目 | 判断 |
| --- | --- |
| 分类 | Large-scale Ego dataset + custom capture hardware + automated multimodal annotation |
| 生态位 | 介于 Ego4D/EPIC 的通用理解数据和 EgoDex/HOT3D 的几何密集数据之间，突出真实服务/商业工作流 |
| 目标用户 | Ego perception、VLA/world-model pretraining、workflow decomposition、hand-motion/scene reconstruction、human-to-robot 团队 |
| 采集规模 | 1680 h stereo video、65,866 episodes、346 tasks |
| 采集年份 | 未报告；只能确认论文 2026-04-26 提交 |
| 三个关键词 | JoyEgoCam；real-world workflow；stereo multimodal annotation |
| 核心贡献 1 | 定制 2160²@60 Hz 双目、130°×130° FOV、200 Hz IMU 的轻量头戴采集系统 |
| 核心贡献 2 | 自动生成 hand/camera motion、depth、hand/object masks、sub-task 与 structured captions |
| 核心贡献 3 | 1680 h 的真实服务/零售/药房/家庭工作流分布，并做离散语义和 feature-space 分析 |
| 最强证据 | 16 个距离档的标定板 depth 误差表；近操作距离 0.5–0.9 m mean error 约 3.0–5.4 mm |
| 最弱证据 | hand “high accuracy”只用 selected qualitative overlay/同源点云；caption 只有 4 例；没有下游 robot experiment |
| 最可疑 claim | “industrial-grade high-quality human demonstration data suitable for robot learning”越过了无 robot action/contact/下游 SR 的证据范围 |
| 精读优先级 | 4/5：真实工作流 Ego 数据基础设施的重要新样本 |
| 复现优先级 | 2/5：硬件和内部 pipeline 不开放，数据入口需登录 |
| 业务试点优先级 | 3/5：先核验数据条款/schema/sample，再考虑预训练；不宜直接上真机控制 |

论文最窄、最可检验的命题是：

> 高规格双目+IMU头戴采集可以在真实工作场景中扩展到 1680 小时，并自动生成比单目低规格 Ego 数据更丰富的 camera、hand、depth、mask 和 task-semantic 中间标注。

论文支持这个数据生产命题的一部分，尤其是规模声明、模态设计与标定板 depth 精度；它没有检验这些中间标注是否带来机器人泛化收益。

## 2. Gap—Evidence 总表

| Gap | 作者机制 | 直接指标 | 消融 | 真机证据 | 是否解决 |
| --- | --- | --- | --- | --- | --- |
| Embodiment | 3D wrist/hand keypoints + MANO，可作为人手 motion prior | 2D/3D 定性对齐图 | 无 retargeting、robot morphology、可执行率对照 | 无 | **否到部分**：产生 human state，未映射 robot action |
| Task | 346 tasks、sub-task segmentation、hand–object–action caption | 长尾词频、t-SNE；4 条 caption example | 无 held-out task/composition/downstream | 无 | **部分（数据层）**：给任务结构，未证明组合能力 |
| Reality | 数据来自真实工作场景；stereo depth + visual-inertial SLAM | 标定板 depth error | 无 sim/system-ID/dynamics/real-posttrain | 无 | **否（控制层）**：真实视觉不等于 robot dynamics |
| Semantic–motion | caption/sub-task 与同步 hand/camera/mask 共时 | 4 个 LLM-judge 例子 | 无全库准确率、shuffle/temporal grounding | 无 | **部分**：时间共现，非联合动作语义模型 |
| Motion–contact | hand/object masks + hand keypoints | selected overlays | 无 contact point/normal/force/tactile | 无 | **否**：mask 相交不是接触真值 |

## 3. 数据资产与五层监督

### 3.1 数据源表

| 数据源 | 规模 | 视角 | 人体姿态 | 手部 | 物体状态 | 接触/力 | 语言 | 用途 |
| --- | ---: | --- | --- | --- | --- | --- | --- | --- |
| JoyEgoCam 自采 | 1680 h、65,866 episodes、346 tasks | 双目 head-mounted，2160²@60 Hz，IMU 200 Hz | 未提供 body/head/feet pose | MANO/3D wrist-hand joints 自动估计 | depth + interacted-object mask；无 object 6D/articulation | 未提供 | sub-task structured captions | 数据本体 |
| Internet Ego | 未使用 | — | — | — | — | — | — | — |
| Mocap/手套 | 未使用 | — | — | 无独立手真值 | — | — | — | — |
| 机器人遥操作 | 未使用/未报告 | — | — | — | — | — | — | 无下游 policy |
| 仿真补全 | 未使用 | — | — | — | — | — | — | 无 sim assets |
| ChArUco/阶梯标定阵列 | 0.5–3.5 m、16 距离档 | JoyEgoCam 双目 | — | — | 已知板几何 | — | — | depth accuracy evaluation |

### 3.2 五层监督审计

| 层次 | EgoLive 论文声称提供 | 来源 | 可靠性边界 | 对机器人学习的意义 |
| --- | --- | --- | --- | --- |
| Raw observation | stereo RGB、calibration、trigger timestamps、200 Hz IMU | 传感器 | 编码、曝光、音频、同步误差、丢帧未报告 | 高质量视觉/active perception 预训练 |
| Human state | wrist/hand joints 6D、MANO、3D keypoints | HaMeR + stereo optimization | 无独立 Mocap GT、MPJPE、valid mask 说明 | 手/EEF motion prior，而非 robot action |
| World state | camera trajectory、dense depth/point cloud | ORB-SLAM3 stereo-inertial + FoundationStereo | 无 trajectory ATE、depth confidence、object pose/mesh/articulation | scene geometry 与相对运动候选 |
| Interaction state | hand/object masks、HOI tracking、sub-task、hand–object–action caption | detector + BoT-SORT + SAM2 + Qwen3-VL | 无 contact/force/grasp phase；caption 无 aggregate score | affordance/contact intent 的弱监督 |
| Robot supervision | 未直接提供 | — | 无 joint/EEF command、reward、success、canonical robot action | 必须经另一个 transfer/retarget/co-training 方法落地 |

### 3.3 明确未提供的模态

- whole-body SMPL、head-to-body transform、floating base、feet/ground contact；
- eye gaze、audio、tactile、force/torque、object weight/friction；
- object instance identity、6D pose、mesh、articulation 和目标状态；
- observed/intended contact、contact point/patch/normal、slip 与 grasp stability；
- robot proprioception、robot command、torque、reward、成功/失败标签；
- 统一 action vector、retargeted trajectory、sim asset 和 real-robot rollout。

视频里“看得见”这些现象不等于数据产品已经标注它们。

## 4. 统计单位、小时口径与 split 风险

### 4.1 Episode、task 与 sub-task

- **episode**：1–3 分钟的单个连续中长操作活动；论文给 65,866 个。
- **task**：346 种 real-world tasks，但没有公开 taxonomy、定义粒度、每类 episode 数或跨场景/操作者重复数。
- **sub-task clip**：由 HOI detection/tracking 切分的约 1–20 秒 atomic action，再由 Qwen3-VL 描述。
- **frame**：每相机 60 Hz。1680 h 对应每相机约 362,880,000 帧，双目合计约 725,760,000 张图像。

平均 episode 时长由论文总量反推：

$$
\bar T_{episode}
=
\frac{1680\times3600}{65{,}866}
\approx 91.8\ \mathrm{s}.
$$

这个结果与论文的 1–3 分钟范围一致，但不能说明每段全部是有效 manipulation：等待、走动、重做、遮挡与无接触时长没有单列。

### 4.2 1680 小时是 raw 还是有效？

论文称 “high-fidelity stereo video recordings” 和 dataset scale 1680 h，却没有报告：

- 总 raw capture 时长；
- 解码/同步/模糊/佩戴错误的拒绝率；
- 通过自动标注和质量门控后的有效时长；
- 纯等待、步行和手不在视野中的比例；
- 重复 episode、近重复场景和返工比例；
- 当前市场入口实际可下载的小时数。

因此 1680 h 应记为**论文声明的语料总时长**，不能进一步称“1680 h clean interaction”或“1680 h fully downloadable”。

### 4.3 Split 与泄漏

论文没有给 train/validation/test split，也没有说明按 operator、scene、store/home、task、object instance 或连续 recording 隔离。真实工作流尤其容易发生：

- 同一房间/货架/床铺跨 split；
- 同一 operator 和佩戴标定跨 split；
- 同一长 recording 被切成相邻 episodes/sub-tasks 后跨 split；
- 同一对象实例和背景纹理泄漏；
- fine-tuned captioner 的训练样本与评测 GT 重叠。

下游发布应至少提供 operator-disjoint、site-disjoint、task-composition-disjoint 和 contiguous-session-grouped 四套 split，并公开去重策略。

## 5. JoyEgoCam 采集硬件与协议

![JoyEgoCam：双目 130° 视场、2160²@60 Hz 与 200 Hz IMU](https://arxiv.org/html/2604.23570v1/figs/HumanDataCaptureSystem7.png)

| 传感器/部件 | 安装位置 | 频率/分辨率 | 同步/标定 | 主要误差与未报告项 |
| --- | --- | ---: | --- | --- |
| 左 RGB | head-mounted | 2160×2160，60 Hz | intrinsics + stereo rectification；左相机首帧定 world init | rolling shutter、曝光、运动模糊、压缩未报告 |
| 右 RGB | head-mounted | 2160×2160，60 Hz | trigger timestamps；双目标定 | baseline 数值、外参温漂、同步误差未报告 |
| IMU | headset | 200 Hz | time-synchronized；ORB-SLAM3 融合 | bias/noise density、坐标轴、camera–IMU 外参未报告 |
| FOV | 双目视野 | 130° horizontal ×130° vertical | 定制光学 | 边缘畸变、有效 stereo overlap 未报告 |
| 支架/头带 | 头部 | lightweight/ergonomic（定性） | 未给佩戴校准 | 重量、续航、温升、舒适度与长期滑移无测试 |

论文把设备描述为 minimal-intrusion，并称用户不易不适、能自然操作，但没有受试者问卷、佩戴时长、重量、颈部负担或行为对照。Figure 2 中设备明显覆盖额头且有双相机组件；相比 VR 不遮眼、相比 UMI 不占手是真实优势，但“自然行为不受影响”仍是未量化假设。

### 5.1 采集协议的选择偏差

- operator 是“recruited”，人数、招募渠道、补偿、培训和技能水平未报告。
- 任务按真实服务/零售场景组织，说明不是完全自由生活日志；task assignment 本身会改变自然分布。
- 头戴视角鼓励持续朝向手部；快速回头、低位/背后操作和被身体遮挡的动作可能被低估。
- 1–3 分钟单连续活动比短桌面 clip 更接近工作流，但仍可能截断跨房间、等待和恢复。
- 没有说明只录成功还是也保留失败、试探、返工与安全中断。
- 双目 60 Hz 的数据量很大，实际 pipeline 可能采样/降帧，但论文未报告发布帧率是否等于采集帧率。

## 6. 自动标注流水线

![EgoLive 自动标注：双目校正、手部重建、场景深度/定位、子任务与文本](https://arxiv.org/html/2604.23570v1/figs/liuchengkuangtu4.png)

~~~mermaid
flowchart LR
    A["JoyEgoCam raw\nstereo RGB 60 Hz + IMU 200 Hz"] --> B["左右目分离、去畸变\nintrinsics / stereo rectification"]
    B --> C["HaMeR per-view\nMANO initialization"]
    C --> D["stereo optimization\n3D wrist/hand keypoints"]
    B --> E["ORB-SLAM3\nstereo + IMU ego-motion"]
    B --> F["FoundationStereo\n1152² depth / point cloud"]
    B --> G["hand-object detector"]
    G --> H["BoT-SORT tracking"]
    H --> I["SAM2 hand/object masks"]
    I --> J["HOI-driven sub-task split\n1–20 s clips"]
    J --> K["fine-tuned Qwen3-VL-32B\nmulti-stage caption reasoning"]
    D --> L["Human motion product\nMANO / 3D joints"]
    E --> M["World/camera product\n6-DoF camera trajectory"]
    F --> M
    K --> N["Semantic product\nhand-object-action captions"]
    L --> O["EgoLive release\n文件 schema/质量 mask 未公开说明"]
    M --> O
    N --> O
~~~

### 6.1 逐阶段证据与误差传播

| 阶段 | 输入 → 输出 | 论文方法 | 主要失败 | 置信度/缺失 mask | 可复现性 |
| --- | --- | --- | --- | --- | --- |
| Stereo preprocessing | raw pair → rectified views | calibration + undistortion | 丢帧、错同步、边缘畸变、外参漂移 | 未报告 | 低；参数/代码未给 |
| Hand initialization | monocular frame → MANO | HaMeR-based stage 1 | 遮挡、截断、手型/肤色、左右手混淆 | 未报告 | 中低 |
| Stereo hand optimization | two-view MANO → metric 3D joints | 双目 2D projection constraints | 两目共同遮挡、反光、错误 correspondence | 未报告 | 低；objective/weights 未给 |
| Camera localization | stereo+IMU → 6-DoF ego-motion | ORB-SLAM3，左相机首帧初始化 | 弱纹理、动态前景、IMU bias、重定位 | 未报告 trajectory covariance | 中；上游开源，配置未知 |
| Dense depth | rectified stereo → depth | FoundationStereo，1152² | 透明/反光/低纹理、遮挡边界、远距离 | 未报告 per-pixel confidence | 中低 |
| HOI detection/tracking | frames → boxes/tracks | contact-hand detector + BoT-SORT | object identity switch、手短时消失 | 未报告 | 中低 |
| Segmentation | tracked prompts → hand/object masks | SAM2 | 漏物体、mask drift、遮挡后重现 | 未报告 | 中低 |
| Sub-task split | HOI tracks → 1–20 s clips | detection/tracking heuristics | 并行动作、接触保持、恢复被误切 | 未报告 boundary score | 低 |
| Caption | clip → structured sentence | fine-tuned Qwen3-VL-32B + multi-stage reasoning | action hallucination、对象/属性混淆、遗漏先后关系 | 未报告 per-caption confidence | 低；prompt/weights/GT 制作未知 |

误差是级联而非独立：stereo calibration 影响 depth 和 3D hand；hand/object detector 影响 SAM2 prompt、sub-task boundary 和 caption；错误边界又会让 Qwen3-VL 把下一动作写进当前 clip。论文没有描述自动 reject、人工复核、失败重跑或 lineage，因此无法知道最终 1680 h 有多少低置信样本。

## 7. 数据分布、覆盖与长尾

![EgoLive 任务时长分布与 action/object/attribute 词云](https://arxiv.org/html/2604.23570v1/dataset_distribution_layout.png)

### 7.1 Figure 4 的任务时长分布

| 任务大类 | 小时 | 占比 |
| --- | ---: | ---: |
| Item Cleaning | 165.9 | 9.9% |
| Kitchen Organizing | 125.4 | 7.5% |
| Bathroom Organizing | 118.9 | 7.1% |
| Folding Clothes | 72.5 | 4.3% |
| Living Room Organizing | 69.8 | 4.2% |
| Wiping Tables | 64.1 | 3.8% |
| Cabinet Cleaning | 61.0 | 3.6% |
| Medicine Packing | 56.8 | 3.4% |
| Bedroom Organizing | 51.3 | 3.0% |
| Sink Cleaning | 49.2 | 2.9% |
| Living Room Service | 36.6 | 2.2% |
| Living Room Tidying | 33.9 | 2.0% |
| Desk Organizing | 33.7 | 2.0% |
| Cosmetics Placement | 25.2 | 1.5% |
| Daily Necessities | 25.1 | 1.5% |
| Oven Operation | 23.9 | 1.4% |
| Toy Organizing | 22.8 | 1.4% |
| Wiping Photo Frames | 19.8 | 1.2% |
| Snack Placement | 19.0 | 1.1% |
| Cooktop Cleaning | 18.1 | 1.1% |
| Seasoning Portioning | 17.8 | 1.1% |
| Ironing Clothes | 16.9 | 1.0% |
| Others | 553.0 | 32.9% |

这张图证明数据不是单一 tabletop pick-place：清洁、整理、衣物、药品打包、烹饪与陈列都存在。它也暴露两个问题：

1. `Others` 占 32.9%，是最大桶，说明公开 taxonomy 过于粗或类别仍高度碎片化；
2. 清洁/整理主导已命名部分，未报告 task-balanced sampling，下游很可能学到 wipe/hold/place 的高频先验而忽略精密装配、工具与高力接触。

词云显示高频 action 包括 `hold/wipe/place/pick up`，object 包括 `cloth/table`，attribute 由颜色词 `white/black/blue/red` 主导。这些 attribute 对 caption richness 有用，却不一定提高控制：颜色多样性会增加语义 token 数，但不提供摩擦、质量、刚度或可动结构。

### 7.2 Figure 5：长尾词频曲线

![EgoLive、EgoDex 与 Xperience-10M 的 object/action/attribute 词频阈值曲线](https://arxiv.org/html/2604.23570v1/word_frequency_threshold_curves.png)

论文对每个词频阈值 $n$ 统计出现次数超过阈值的 distinct labels。可把图中量写为：

$$
C_{\mathcal V}(n)
=
\left|
\left\{w\in\mathcal V\;:\;f(w)>n\right\}
\right|,
$$

其中 $\mathcal V$ 是 object、action 或 attribute 词表，$f(w)$ 是词频。该式是对论文统计的显式记账，论文没有单独给公式。

EgoLive 的红线在三个维度的大多数阈值上高于 EgoDex 与 Xperience-10M，并延伸到更高频率，支持“标签覆盖更宽、头部样本更多”。但比较并未控制：

- 总小时/episode 数不同；更大数据自然更容易抬高 $C(n)$；
- captioner、语言风格和词形归一化是否一致；
- 同义词、颜色/属性修饰是否去重；
- 是否按同样帧率和 sub-task 粒度切片；
- 长尾词是否至少跨多个 operator/site/object instance。

因此它是**annotation vocabulary coverage**，不是独立物理技能数，也不是机器人 task generalization 指标。

### 7.3 Figure 6：连续 embedding 分布

![Cosmos-Embed1-448p 特征的 t-SNE：EgoLive、EgoDex、Xperience-10M](https://arxiv.org/html/2604.23570v1/all3_tsne.png)

论文用 Cosmos-Embed1-448p 提取对象、环境与动作的联合视觉 embedding，再用 t-SNE 可视化。EgoLive 红点覆盖更大区域，并出现多个局部簇；图中示例也展示卧室、厨房、走廊/门口和收纳等场景。

但不能从 t-SNE 图直接推出“更高 density”或“更宽 manifold”作为定量事实：

- t-SNE 的全局距离和面积不保真；
- perplexity、learning rate、seed、样本数与 subsampling 未报告；
- 不同数据集若样本数不等，点云面积与局部密度不可直接比较；
- Cosmos embedding 可能主要编码背景/对象，而不是接触或动作；
- 图中 cluster coherence 没有 silhouette、effective rank、coverage 或 kNN mixing 数字。

所以 Figure 6 是有启发性的 qualitative distribution map，不是严格的 diversity benchmark。

### 7.4 未报告的分布维度

| 维度 | 论文状态 | 为什么重要 |
| --- | --- | --- |
| Operator 数/每人时长 | 未报告 | 判断动作风格与身份集中度 |
| 年龄、性别、体型、肤色、惯用手 | 未报告 | 手重建公平性和人群覆盖 |
| 地区、语言、文化 | 未报告 | 工作流/物体/语义的地域偏差 |
| 独立 site/家庭/门店数 | 未报告 | 判断背景泄漏和生态多样性 |
| Object instance/category 数 | 未报告 | 346 tasks 不等于对象覆盖 |
| 单手/双手比例 | 未报告 | 灵巧手与双臂训练价值 |
| 成功/失败/恢复 | 未报告 | policy recovery 与成功偏差 |
| 移动/静态操作比例 | 未报告 | humanoid loco-manipulation 价值 |
| 透明/反光/柔性/液体 | 未报告 | stereo、mask、contact 的困难分布 |
| 危险行为与工具 | 未报告 | 安全训练与内容治理 |

数据规模看起来覆盖真实长尾，但没有这些统计，无法区分“更多新行为”与“相似清洁/整理在更多背景中重复”。

## 8. 人手、相机、场景与物体重建

### 8.1 双目深度的几何含义

在理想 rectified stereo 中，深度可由 disparity 近似：

$$
Z=\frac{fB}{d},
$$

其中 $f$ 是焦距、$B$ 是双目 baseline、$d$ 是左右像素 disparity。JoyEgoCam 通过 FoundationStereo 估计 dense correspondence，因此相比单目 depth 有 metric scale 的硬件约束。但论文未给 baseline、rectification residual、rolling-shutter 模型或 per-pixel confidence；当 $d$ 很小、物体透明/反光或两目共同遮挡时误差仍会放大。

### 8.2 Camera ego-motion

ORB-SLAM3 融合 rectified stereo 和 200 Hz IMU，从左相机第一帧初始化。概念上相机点到世界点为

$$
\tilde{\mathbf p}^{W}_{t}
=
\mathbf T^{W}_{C_t}
\tilde{\mathbf p}^{C_t}_{t},
$$

但论文没有公开 $\mathbf T$ 的存储方向、单位、world axes、时间戳对应、相机–IMU 外参或重定位处理。没有 ATE/RPE benchmark，也没有说明跨 1–3 分钟 episode 是否闭环、是否出现尺度/航向漂移。

立体+IMU 能显著缓解单目尺度不确定性，却不自动消除动态前景和纯旋转问题。大面积手、布料或近物体占据画面时，ORB feature 可能落在非刚体上；pipeline 是否 mask dynamic regions 未报告。

### 8.3 两阶段手重建

第一阶段对每个 monocular view 用 HaMeR 估 MANO，确保 2D 投影；第二阶段在 stereo space 联合优化，使 3D keypoints 同时满足两目 projection constraints。可概念化为：

$$
\min_{\boldsymbol\theta,\boldsymbol\beta,\mathbf t}
\sum_{c\in\{L,R\}}
\sum_j
w_{c,j}
\left\|
\pi\!\left(
\mathbf K_c,
\mathbf T_c,
\mathbf X_j(\boldsymbol\theta,\boldsymbol\beta,\mathbf t)
\right)
-\mathbf u_{c,j}
\right\|_2^2
+\lambda\mathcal R_{MANO},
$$

其中 $\boldsymbol\theta,\boldsymbol\beta,\mathbf t$ 是 pose/shape/translation，$\mathbf u_{c,j}$ 是每目 2D joint，$w$ 是置信度，$\mathcal R$ 是可行手形先验。该式是按论文文字重建的合理目标，不是作者公布的 exact implementation；实际 loss、temporal term 和权重没有给。

### 8.4 Figure 7：2D hand 的证据边界

![EgoDex 与 EgoLive 的 selected 2D keypoint 可视化对比](https://arxiv.org/html/2604.23570v1/2D.png)

Figure 7 的四组样例中，EgoLive skeleton 在手上看起来更贴合，EgoDex selected examples 有明显漂移。这个图证明作者能找到质量好的 EgoLive 结果和质量差的 EgoDex 结果，但不是公平 benchmark：

- 两数据集不是同一视频/动作/遮挡；
- 没有人工 2D GT、PCK/AUC、样本数或随机抽样协议；
- EgoLive 是双目 60 Hz 新硬件，EgoDex 的 native tracking 与导出处理不同；
- 只展示成功样例，无法估计 failure tail。

所以不能据此量化“EgoLive 2D hand 比 EgoDex 高多少”。

### 8.5 Figure 8：3D hand 与同源点云

![3D keypoints 在 ego/top/left/right 视角中的 point-cloud 对齐](https://arxiv.org/html/2604.23570v1/3D.png)

多视角渲染显示 wrist/joints 没有明显视觉漂移，是重要 sanity check。但手 skeleton 和场景 point cloud 都依赖同一套 stereo calibration；点云又来自 FoundationStereo，而不是外部激光/Mocap。两者共同偏移时仍可能“互相对齐”。真正独立的 3D 精度应使用光学 Mocap、标记手套、已知刚性手模型或多相机 GT，报告 MPJPE/PA-MPJPE、wrist drift、遮挡分层和 missing rate。

### 8.6 Object 与 scene

数据有 dense depth、point cloud、interacted-object mask 和 object words，因此可学习局部 scene geometry 与对象区域。但论文没有 object 6D pose、mesh、instance identity、articulation 或跨帧 rigid/non-rigid state。BoT-SORT track + SAM2 mask 不是 3D object trajectory：相机运动、mask jitter 和遮挡后 identity switch 都会污染状态变化。

### 8.7 Contact 不可由 mask 自动获得

引用的 hand-contact detector可发现“手与对象交互”的视觉状态，SAM2 可分割相应区域；仍然没有：

- hand/object surface distance；
- contact point/patch/normal；
- intended vs observed contact；
- force/wrench/friction/slip；
- grasp phase 与稳定性；
- object response/goal state。

像素 mask 相邻或重叠只是一种视觉共现，尤其在 ego 视角和遮挡下不能当作真实接触标签。

## 9. Depth 精度：论文唯一系统量化的几何模态

![0.5–3.5 m 标定板的 point-cloud 重建](https://arxiv.org/html/2604.23570v1/figs/biaodingjian6.png)

论文用阶梯深度阵列和 ChArUco board，在 0.5–3.5 m 估 camera-to-board 6D pose，并由已知板几何得到参考 depth。若第 $i$ 个 corner 的预测/参考深度是 $\hat z_i,z_i$，表 2 的 mean error 和阈值准确率可写为：

$$
E_{depth}
=
\frac{1}{N}\sum_{i=1}^{N}
\left|\hat z_i-z_i\right|,
\qquad
A_{\tau}
=
\frac{1}{N}\sum_{i=1}^{N}
\mathbb 1\!\left[
\left|\hat z_i-z_i\right|<\tau
\right].
$$

这同样是对表格指标的显式定义，论文没有单列公式。

### 9.1 Table 2 完整结果

| Distance (mm) | Mean error (mm) | <5 mm | <10 mm | <20 mm | <30 mm | <40 mm |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 500 | 3.059 | 79.576% | 99.190% | 100.000% | 100.000% | 100.000% |
| 700 | 3.045 | 79.755% | 99.833% | 100.000% | 100.000% | 100.000% |
| 900 | 5.381 | 56.860% | 87.051% | 96.987% | 99.917% | 100.000% |
| 1100 | 7.091 | 43.840% | 74.817% | 96.099% | 99.417% | 100.000% |
| 1300 | 7.033 | 48.989% | 73.227% | 95.577% | 99.697% | 100.000% |
| 1500 | 8.751 | 35.636% | 63.664% | 92.528% | 99.333% | 99.917% |
| 1700 | 13.114 | 24.233% | 43.992% | 76.874% | 93.943% | 99.079% |
| 1900 | 12.083 | 27.696% | 49.601% | 79.745% | 94.904% | 99.074% |
| 2100 | 13.014 | 26.505% | 47.433% | 74.121% | 92.053% | 99.750% |
| 2300 | 15.095 | 20.878% | 40.893% | 69.960% | 87.822% | 96.933% |
| 2500 | 13.994 | 25.061% | 47.391% | 71.720% | 88.600% | 97.527% |
| 2700 | 14.820 | 23.789% | 41.761% | 72.063% | 87.318% | 96.400% |
| 2900 | 13.414 | 31.071% | 49.340% | 71.320% | 89.229% | 98.794% |
| 3100 | 15.254 | 26.290% | 45.319% | 65.630% | 83.568% | 97.064% |
| 3300 | 17.372 | 18.980% | 36.162% | 64.540% | 81.561% | 92.889% |
| 3500 | 18.035 | 20.952% | 36.861% | 59.798% | 79.431% | 93.478% |

### 9.2 表格真正说明什么

- 在最典型的近手操作距离 0.5–0.7 m，约 99% corner error <10 mm。
- 0.9 m 时 mean error 5.381 mm，87.051% <10 mm；作者据此把 <0.9 m 称为低误差区。
- 1.7 m 后 mean error 多在 12–18 mm；3.5 m 只有 59.798% <20 mm。
- error 并非随距离严格单调，说明 pose、视角、采样点或匹配质量也影响结果。

### 9.3 外推限制

- GT depth 仍依赖 board pose 的 BFGS reprojection optimization，不是激光测距逐像素真值；
- 标定板刚性、高对比、纹理友好，不代表透明瓶、反光金属、黑色织物、毛发和动态手；
- 只评 corner depth，不评 object boundaries、dense surface completeness 或 temporal flicker；
- 没有分设备/温度/佩戴形变的 stereo calibration drift；
- 近距离 3–5 mm depth error 不等于 hand joint 3D error 也是 3–5 mm。

### 9.4 Figure 10 的真实场景可视化

![EgoLive 三种真实场景的 RGB、depth 与 point cloud](https://arxiv.org/html/2604.23570v1/figs/vis5.png)

Figure 10 显示铺床、室内货架/柜体和墙面/插座场景的整体空间结构能被恢复。它支持 pipeline 能在不同场景运行，但没有展示 failure cases、透明/反光/动态对象或定量 real-scene GT；不能用三行成功图代替全库 depth quality distribution。

## 10. Instruction captioning 与子任务语义

![玻璃清洁、叠衣、冰箱清洁与铺床的 structured caption 样例](https://arxiv.org/html/2604.23570v1/figs/DemoCap.png)

每个 1–20 秒 sub-task clip 由 fine-tuned Qwen3-VL-32B 描述。相比自由 caption，作者显式要求三个元素：

1. 使用哪只手；
2. 操作哪个对象及属性；
3. 执行什么动作及方向/顺序。

这对机器人工作流分解比“use a squeegee to wipe the glass door”更有用，因为它把可检索的 hand–object–action 绑定在一个时间段。但语义并未显式绑定到 3D keypoint、mask pixel、contact event 或 object state change；它仍是 clip-level sentence。

### 10.1 Table 3 完整样例

| # | GT | Prediction | Action | Object | Hand | Global |
| ---: | --- | --- | :---: | :---: | :---: | :---: |
| 1 | Clean the glass from top to bottom with the white squeegee in the right hand. | Hold a white squeegee in the right hand and clean the glass from top to bottom. | ✓ | ✓ | ✓ | ✓ |
| 2 | Place the black cotton coat hanging on the hanger flat on the bed, then zip it up with both hands. | Place the down jacket on the bed with both hands, and zip it up. | ✓ | ✓ | ✓ | ✓ |
| 3 | Remove the cloth covering the sponge with the right hand. Then pick up the sponge with the right hand. | Move to the left, pick up the yellow sponge with the right hand, then wipe the white countertop. | ✗ | ✗ | ✓ | ✗ |
| 4 | Spread and smooth the blue-and-white striped duvet flat on the bed with both hands. Then grasp its two corners with both hands, lift it, and fully unfold it. Finally, walk toward the left of the bed. | Fold the blue-white-gray striped quilt and arrange it with both hands, then walk to the left side of the bed. | ✗ | ✓ | ✓ | ✗ |

在这四个**展示样例**中，hand 4/4、object 3/4、action 2/4、global 2/4。不能把这四个比例当作数据集准确率，因为样本选择方式不明。它们仍揭示实际失败模式：

- Sample 3 把“揭开布并拿海绵”幻觉成“拿海绵擦台面”；
- Sample 4 把展开被子描述为折叠/整理，动作方向反了；
- hand identity 最容易从画面确认，对象属性次之；
- 长程、多阶段、柔性物动作最容易在 action/global 层失败。

### 10.2 LLM-as-a-Judge 的证据缺口

作者用 LLM-as-a-Judge 判断 hand/object/action/global consistency，但没有报告：

- judge 模型与 prompt；
- GT captions 的来源、annotator 数与一致性；
- 全评测集大小和 aggregate accuracy/F1；
- judge 与人类的 agreement；
- fine-tuned Qwen captioner 是否见过评测任务/场景；
- temporal boundary 是否独立评测；
- per-caption confidence 和 rejected rate。

因此 Table 3 是 error taxonomy 的例子，不是 caption quality benchmark。

### 10.3 语义到 task/contact 的距离

Structured caption 能成为 System2 的 language condition 和 subgoal proposal，但没有给：

- object before/after state；
- contact start/end；
- force direction、grasp region、affordance；
- success/failure/recovery；
- robot-feasible action parameterization。

`fold`, `wipe`, `place` 等词描述意图，不规定如何让特定 robot 本体完成。尤其 Sample 4 的 fold/unfold 反转会直接破坏 task goal；因此 caption 不能未经视觉/对象状态校验直接用于 policy supervision。

## 11. Table 1：跨数据集比较与公平性

| Dataset | Scene | Scale | Resolution | FPS | Multi-view | Motion tracking | Language | Depth |
| --- | --- | ---: | --- | ---: | :---: | :---: | :---: | :---: |
| EPIC-KITCHENS-100 | Real-world | 100 h | 1920×1080 | 50 | ✗ | ✗ | ✓ | ✗ |
| Ego4D | Real-world | 3680 h | inconsistent | 30 | ✗ | ✗ | ✓ | ✗ |
| EgoMimic | Laboratory | 55 h | 1408×1408 | 30 | ✗ | ✓ | ✗ | ✗ |
| HOI4D | Laboratory | 22 h | 1920×1080 | 15 | ✗ | ✓ | ✗ | ✓ |
| HOT3D | Laboratory | 14 h | 1408×1408 | 30 | ✓ | ✓ | ✗ | ✗ |
| EgoDex | Laboratory | 829 h | 1920×1080 | 30 | ✗ | ✓ | ✓ | ✗ |
| Egocentric-10K | Real-world | 10k h | 1920×1080 | 30 | ✗ | ✗ | ✗ | ✗ |
| Xperience-10M | Real-world | 1059 h available | 512×512 | 20 | ✓ | ✓ | ✓ | ✓ |
| EgoLive | Real-world | 1680 h | 2160×2160 | 60 | ✓ | ✓ | ✓ | ✓ |

表 1 支持 EgoLive 在“真实场景 + 双目 + motion + language + depth + 高时空分辨率”的组合上很完整。它不支持简单的总排名：

- 其他数据集可能有更准确的原生硬件 tracking/object pose，二元 ✓ 不反映精度；
- “multi-view”可指 stereo 同步目，也可指 ego–exo，多视角语义不同；
- EgoLive 的 hand/depth/language 多为自动伪标签，而不是传感器/人工真值；
- 表中不含 operator/site/object/task diversity、contact、force、robot action 或 license；
- EgoLive 的“main public release”不能匿名检查，self-reported ✓ 尚缺文件级证据。

## 12. 语义—人体运动—机器人动作—接触的联合对齐

| 层 | Human 表示 | Robot 表示 | Contact 表示 | 对齐机制 | 监督来源 | 论文验证 |
| --- | --- | --- | --- | --- | --- | --- |
| 任务语义 | task/caption | 未提供 | 无 | clip-level text | Qwen3-VL + 未说明 GT | 4 例 LLM judge |
| 子目标 | 1–20 s sub-task | 未提供 | HOI track 间接决定边界 | detection/tracking segmentation | 自动伪标签 | 无 boundary F1 |
| 运动意图 | MANO/3D hand/wrist | 未提供 | 无显式 phase | 同步到 camera ego-motion | stereo optimization | selected qualitative figures |
| 末端几何 | wrist/joints | 无 TCP/EEF | 无 contact point | 未来下游需 retarget | 自动 hand | 无 robot evidence |
| 全身运动 | 未提供 | 未提供 | 未提供 | 无 | — | 无 |
| Contact intent | object mask + action verb | 未提供 | 视觉 HOI | 同 clip 共现 | detector/SAM2/caption | 无 |
| Grasp goal | MANO pose + object word/mask | 无 robot grasp | 无 region/normal/wrench | 未实现 | 伪标签 | 无 |
| 力与柔顺 | 无 | 无 | 无 | 无 | — | 无 |

### 12.1 联合对齐的准确结论

- 语义、hand motion、camera、depth、masks 在**时间上同步**，但不是一个 joint objective 训练出的共享 latent。
- caption 是 language condition/检索键，不直接决定 contact/subgoal feasibility。
- human motion 应视为 behavior prior；论文没有把它 retarget 成 robot 轨迹。
- contact intent 可由 hand/object mask + verb 弱推断，真实接触不可观察。
- 同一 task 的多策略是否保留未知，因为没有每任务 operator/object/site 统计。
- caption 与 tracking 错误没有统一置信度，也没有 robot rollout 失败回流。
- 高频清洁/整理背景容易造成 scene-to-action shortcut。

### 12.2 Video → Human State → Intent → Robot Behavior

~~~mermaid
flowchart TD
    V["stereo RGB + IMU"] --> CAL["rectification / synchronization\n风险：外参与丢帧"]
    CAL --> CAM["ORB-SLAM3 camera pose\n风险：动态前景/漂移"]
    CAL --> DEP["FoundationStereo depth\n风险：反光/遮挡/边缘"]
    CAL --> HAND["HaMeR + stereo MANO\n风险：遮挡/hand swap/先验偏差"]
    CAL --> HOI["detector + track + SAM2\n风险：identity/mask drift"]
    HOI --> SUB["1–20 s sub-task boundary\n风险：并行动作/错误切分"]
    SUB --> TXT["Qwen3-VL caption\n风险：动作与顺序幻觉"]
    CAM --> HS["human/world intermediate state"]
    DEP --> HS
    HAND --> HS
    TXT --> INT["task/subgoal/contact-intent prior"]
    HOI --> INT
    HS --> INT
    INT --> GAP["缺失：object state/contact/robot action"]
    GAP -. "论文未实现" .-> RET["retarget / latent alignment / policy"]
    RET -. "论文未实现" .-> SIM["physics / system ID / sim-to-real"]
    SIM -. "论文未实现" .-> ROB["real robot + failure feedback"]
~~~

## 13. Embodiment Gap：EgoLive 停在 human state

### 13.1 子 gap 分解

| 子 gap | EgoLive 提供 | 仍缺什么 |
| --- | --- | --- |
| Morphology | MANO/hand joints | robot topology、link lengths、surface/contact mapping |
| DoF | high-DoF human hand | target joint space、coupling、underactuation |
| Workspace | world/camera hand motion（论文声称） | robot base/arm reachability、TCP calibration |
| Floating base | camera trajectory | body/head/base transform、feet、CoM、balance |
| Hand/gripper | human finger pose | gripper scalar 或 dex hand functional mapping |
| Observation | ego stereo RGB/depth | robot camera extrinsics、self-occlusion、sensor domain |
| Action | 3D human keypoints | robot EEF/joint/torque/chunk |
| Contact capability | HOI mask/verb | contact topology、normal、force、friction/compliance |

### 13.2 不是 retargeting 方法

论文没有 IK、optimization、GMR、RL retargeting、differential whole-body IK 或 canonical robot action；也没有 balance、self-collision、joint limit、environment collision。它发布的是 retargeting 的**候选输入**，不是 retargeted demonstration。

若下游使用，至少需要求解：

$$
\min_{\mathbf q_{1:T}}
\lambda_w\mathcal L_{wrist}
+\lambda_f\mathcal L_{fingertip}
+\lambda_o\mathcal L_{object}
+\lambda_c\mathcal L_{contact}
+\lambda_j\mathcal L_{joint\text{-}limit}
+\lambda_s\mathcal L_{smooth},
$$

并满足 collision、balance 和 actuator constraints。EgoLive 可支撑 wrist/finger 与部分 smooth 项；object/contact/joint/physics 必须补齐。因此最准确分类是 **human motion/intent data source**，不是 motion retargeting、functional retargeting 或已完成的 intent transfer。

### 13.3 Camera motion 不能当 humanoid base

JoyEgoCam 在头部，camera trajectory 混合：人整体移动、躯干摆动、头颈主动观察和设备滑移。没有 head-to-body/feet 时不能分解 floating base；直接映射 humanoid base 会产生不平衡轨迹。camera motion 更适合 active-vision prior，而不是 locomotion command。

## 14. Task Gap：346 tasks 与 latent manifold 的边界

EgoLive 的 346 tasks、1–20 秒 sub-task 和长尾 captions 为 task representation 提供好材料，但论文没有学习 latent skill manifold。对 prompt 中十个问题的回答是：

1. **Manifold 方法**：无 VAE/diffusion/transformer/LAM/world-model policy 训练。
2. **时间尺度**：episode 1–3 分钟，sub-task 1–20 秒；边界由 HOI tracking 自动产生。
3. **输入**：未来方法可用 video、hand、camera、depth、mask、caption；无完整 object state。
4. **新任务组合**：未验证；346 是 collected task taxonomy，不是 composition benchmark。
5. **Structured deviation**：未定义。
6. **Correctable latent**：未定义在线修正接口。
7. **Language coaching/takeover**：未实现。
8. **连续/物理可行**：sub-task 边界没有接触 continuity 或 dynamics 约束。
9. **平滑但任务错**：Sample 4 正说明语言可自然流畅却把 unfold 说成 fold。
10. **Coverage**：词频曲线与 t-SNE 只量 annotation/visual coverage，不量可执行 skill coverage。

所以应区分：

- dataset 内的 task diversity：有证据；
- 未见对象/场景的 task generalization：未评；
- 未见 skill sequence 的 compositional generalization：未评；
- 按 caption/embedding 做 trajectory retrieval：数据支持，但论文未测。

## 15. Reality Gap：真实场景几何不等于 robot dynamics

| 差异 | EgoLive 建模 | 数据来源 | 是否在线 | 指标 | 剩余风险 |
| --- | --- | --- | --- | --- | --- |
| 视觉外观 | 真实双目 RGB | 真实家庭/服务/零售 | 离线数据 | 仅分布图 | robot 相机和自遮挡不同 |
| Scene geometry | stereo depth/point cloud | FoundationStereo | 离线 | 标定板 depth error | 动态/透明/反光物误差 |
| Human kinematics | MANO/3D joints | video reconstruction | 离线 | 定性 | robot morphology 不同 |
| Object dynamics | 未建模 | — | — | — | 无 pose/state/mass/friction |
| Contact/force | 未建模 | — | — | — | 无 tactile/wrench/slip |
| Joint dynamics | 未建模 | — | — | — | friction/backlash/coupling |
| Latency/saturation/temperature/payload | 未建模 | — | — | — | 真机 failure 未覆盖 |
| System ID/domain randomization | 无 | — | — | — | sim-to-real 未处理 |
| Real post-training | 无 | — | — | — | 需另采 robot data |

论文没有 joint-wise neural dynamics、HumanoidDM、simulator、domain randomization、actuator-aware post-training 或 System0。真实人类场景只缩小 perception reality gap；机器人 hardware reality gap 完全留给后续方法。

## 16. Contact Intent 与 Grasping Goal

### 16.1 可用的弱信号

| 信号 | 可推断 | 不能推断 |
| --- | --- | --- |
| hand mask + object mask | 哪只手靠近/遮挡哪个对象 | 是否真实接触、接触深度/法向 |
| hand keypoints/MANO | pre-grasp hand shape、wrist path | 接触压力、稳定性、滑移 |
| verb/object caption | intended action 和功能对象 | observed success、force direction |
| depth/point cloud | 局部几何与候选接触面 | material/friction/compliance |
| sub-task boundary | 粗 phase 变化 | contact make/break 的精确时刻 |

### 16.2 Grasp goal 的正确抽象

不能直接把 MANO pose 当成 robot grasp。下游应先估计 object-centric functional goal：

$$
g=
\left(
\mathcal O,
\mathcal R_{contact},
\mathbf T^{goal}_{object},
\mathcal W_{required},
\phi_{phase}
\right),
$$

其中对象 $\mathcal O$、接触区域 $\mathcal R_{contact}$、目标状态、所需 wrench set 和 phase 再由本体专用 planner/controller实现。EgoLive 当前只弱提供 $\mathcal O$、hand motion 和部分 phase；其余需要 object tracking/contact/tactile 补标。

### 16.3 柔性物与多指风险

数据里衣物、床品和清洁布很多，这对任务语义很有价值，却是最难恢复的 contact physics：object state 高维、mask 大、抓点被遮挡、摩擦与张力决定结果。Sample 4 正好显示柔性物 action 容易被 caption 反转。没有 cloth topology、contact、force 时，这些片段只能作为视觉/高层 workflow prior，不能直接监督 dexterous/whole-body contact controller。

## 17. Human–Robot Co-training 与 latent alignment

EgoLive 论文没有进行 human–robot co-training。它引用 EgoMimic、EgoBridge、H-RDT、Being-H0.5 等工作说明潜在用途，但没有提供共享 encoder、latent、policy head、embodiment token 或 robot bridge data。十个关键问题的论文答案如下：

| 问题 | EgoLive 的证据 |
| --- | --- |
| Human/robot data 直接混合？ | 否；论文没有 robot dataset |
| 共享 encoder/latent/head？ | 未提出模型 |
| 显式诱导 latent overlap？ | 无 |
| EgoBridge 式先对齐再共训？ | 只在 related work 提及 |
| Teleoperation 作为桥？ | 无 teleop 数据量或方案 |
| Human:robot 比例/权重？ | 无 |
| Negative transfer？ | 无下游实验 |
| Joint/sequential/robot-only 对比？ | 无 |
| Human 用于 pretrain/post-train？ | 论文只表达可用于 robot learning 的愿景 |
| 收益来自视觉/任务/动作哪层？ | 无分解消融 |

### 17.1 最小因果实验矩阵

若要证明 EgoLive 的机器人价值，应固定 robot demonstrations、总 token、steps、backbone 和 compute，比较：

| 组别 | Ego RGB/depth | Caption | Hand/camera motion | Robot action | 回答的问题 |
| --- | :---: | :---: | :---: | :---: | --- |
| Robot-only | ✗ | robot task text | ✗ | ✓ | 基线 |
| Visual-only | ✓ | shuffle/空 | ✗ | ✓ | 真实场景/对象是否有用 |
| Semantic-only | ✓ | ✓ | ✗ | ✓ | workflow language 是否增益 |
| Motion-only | ✓ | 空 | ✓ | ✓ | human hand/camera prior 是否增益 |
| Full EgoLive | ✓ | ✓ | ✓ | ✓ | 总增益 |
| Full + object/contact | ✓ | ✓ | ✓ + 补标 | ✓ | 缺失物理层价值 |
| Human-only | ✓ | ✓ | ✓ | ✗ | 能否零 robot data，预计只适合 representation/weak action |

还应分别 hold out operator、site、object instance、task composition 和 robot embodiment。否则增益可能只是训练看过相同床、货架或清洁工具。

### 17.2 对齐什么，而不是盲目拉近什么

Human hand joint 与 robot joint 不应逐维拉近。共享 latent 最好表达 task function 和 object/contact goal：

$$
\mathbf z^{shared}
=
f_{task}\!\left(o_{1:T},l,\hat s_{object},\hat c\right),
\qquad
\mathbf a^e
=
g_e\!\left(\mathbf z^{shared},s^e_{robot}\right),
$$

其中 $e$ 是 embodiment，$g_e$ 是本体专用 head。EgoLive 可提供 $o,l$ 和不完整的 $\hat c$，但 object state 和 robot state/action 仍需目标域数据。若直接对齐 scene embedding，模型可能只学“卧室→铺床”，而非床单状态和接触策略。

## 18. 策略架构与快慢系统接口

| 层 | 输入 | 推荐输出 | 频率 | 训练数据 | Gap 责任 |
| --- | --- | --- | ---: | --- | --- |
| System2 | RGB/depth、caption、task history、object state | task graph、subgoal、object/contact goal | 0.1–2 Hz | EgoLive workflow + 人工/robot outcome | Task / semantic–contact |
| System1 | subgoal、robot obs/state、human motion prior | feasible EEF/latent action chunk + uncertainty | 5–30 Hz | EgoLive hand/camera + robot teleop | Embodiment / Task |
| System0 | EEF/contact constraint、joint/force/tactile | torque/position/impedance command | 100–1000 Hz | robot contact rollouts + sim | Reality / Contact |
| Drive | low-level setpoint、current/encoder | motor current/torque | kHz | calibration/system ID | Hardware |

EgoLive 的最佳落点在 System2/1：长任务可教 workflow，hand/camera/depth 可教 motion/active perception prior。慢系统给快系统的接口应是 object-centric subgoal、contact region、EEF corridor、latent chunk 和 uncertainty，而非逐帧复制 MANO。System0 的力、滑移、碰撞、顺应与安全不能从视觉伪标签替代。

## 19. 动作表示与时序连续性

### 19.1 论文实际提供的动作表示

论文只说“6D trajectories of both wrist and hand joints”并提供 MANO/3D keypoints，没有给字段 schema、维度、旋转表示、坐标系、单位、速度或 action chunk 格式。它更接近 human state trajectory，而不是 action：

$$
\mathbf h_t
=
\left[
\mathbf T^{W}_{C_t},
\mathbf p^{W}_{wrist,t},
\mathbf R^{W}_{wrist,t},
\mathbf J^{W}_{hand,t},
\boldsymbol\theta^{MANO}_t
\right].
$$

这个概念向量需要从实际发布 schema 验证；论文没有说明是否所有字段、双手、两目或每帧都存在。

### 19.2 Episode/sub-task 不等于 robot action chunk

- episode 1–3 分钟，适合 workflow memory；
- sub-task 1–20 秒，适合语义 option 候选；
- robot chunk 通常更短，还要满足动力学与接触连续性；
- HOI-driven boundary 可能在双手并行动作中切错；
- caption 可能把 action 顺序或方向写反；
- camera 60 Hz、IMU 200 Hz、robot servo 100–1000 Hz，需要重采样与 anti-alias；
- invalid/occluded hand 的处理、interpolation 和 mask 未公开。

若从白噪声独立生成每个 chunk，接触建立/释放处容易跳变。更安全的策略是以前一 chunk 末状态、object state、contact phase 和 uncertainty 为条件，并让相邻 chunk 重叠优化。

### 19.3 Locomotion 兼容性

caption 可以描述“walk toward the left of the bed”，camera pose 也含人移动，但没有 foot contacts、body pose、CoM 或 ground plane 的可靠对应。它能训练 active-view/高层 navigation intent，不能直接成为 floating-base trajectory。缺 whole-body pose 只在固定底座/上肢任务中可绕过；在走动、弯腰、蹲起、承重和全身接触中无法靠清洗补齐。

## 20. 训练数据、目标函数与 curriculum

论文没有训练下游模型，也没有给一个统一 loss。合理的使用路线应把直接可监督项与需补数据项分开：

| 阶段 | 数据 | 目标 | EgoLive 可提供 | 额外需要 |
| --- | --- | --- | --- | --- |
| Video/vision pretrain | stereo RGB/depth | masked/future/contrastive | 大规模真实场景 | 数据下载/schema/quality mask |
| Semantic grounding | clip + caption + masks | temporal grounding/caption/segmentation | hand–object–action 弱监督 | 人工 GT、hard negatives |
| Human motion | RGB/depth + MANO/joints/camera | pose/action prediction | human hand/camera trajectory | independent GT/uncertainty |
| Latent skill | long episode/sub-task | future state/LAM/option | workflow hierarchy | object state、contact phase |
| Robot co-training | human + robot | policy/action + latent alignment | human prior | teleop actions、embodiment token |
| Retarget/sim | human trajectory + assets | feasibility/contact/dynamics | motion initialization | robot model、object mesh/contact |
| Real post-train | robot rollouts | success/recovery/force-aware | 无直接数据 | real robot outcome/tactile/system ID |

概念性总目标可写为：

$$
\mathcal L
=
\lambda_v\mathcal L_{video}
+\lambda_s\mathcal L_{semantic}
+\lambda_h\mathcal L_{hand}
+\lambda_o\mathcal L_{object}
+\lambda_c\mathcal L_{contact}
+\lambda_f\mathcal L_{feasible}
+\lambda_p\mathcal L_{policy}
+\lambda_x\mathcal L_{cross\text{-}embodiment}.
$$

EgoLive 直接支持前三项和部分 object mask/depth；contact、physical feasibility、robot policy 与 cross-embodiment outcome 需要另采。不能因为文件包含 depth/hand 就假设完整 loss 都有真值。

### 20.1 推荐 curriculum

1. 先用近距离、双手可见、低 camera acceleration、刚性对象、高 caption agreement 的 clips；
2. 再加入双手协作、较长 workflow、更多 site 和 object instances；
3. 单独引入强遮挡、反光、柔性物、快速 camera motion，并降低伪标签权重；
4. 用少量目标 robot 数据对齐 EEF/object outcome，而不是人/robot joint 直接匹配；
5. 仿真中补 collision/contact/material randomization；
6. 真机用 force/tactile 和 recovery rollouts 做 actuator-aware post-training。

## 21. 数据质量与不确定性总表

| 模态 | 真值来源 | 误差指标 | 缺失率 | 失败分布 | 置信度 | 对机器人训练的影响 |
| --- | --- | ---: | ---: | --- | --- | --- |
| Stereo RGB | 传感器 | 未报告图像/同步质量 | 未报告 | 模糊、曝光、丢帧、遮挡 | 未说明 | 视觉域丰富但可能错同步 |
| IMU | 传感器 | 未报告 bias/noise | 未报告 | 温漂、震动、headset 滑移 | 未说明 | camera pose 漂移 |
| Camera pose | ORB-SLAM3 | 无 ATE/RPE | 未报告 | 弱纹理、动态、重定位 | 无 covariance 说明 | world motion 可能混 camera error |
| Depth | FoundationStereo | Table 2：0.5–3.5 m mean/threshold error | 未报告 | 远距离、透明、反光、边界 | 无 per-pixel confidence 说明 | object/hand scale 与碰撞几何受影响 |
| 2D/3D hand | HaMeR + stereo optimization | selected qualitative overlays | 未报告 | 遮挡、截断、hand swap、形状偏差 | 未说明 | retarget/action label 系统偏差 |
| Hand/object mask | detector + BoT-SORT + SAM2 | 无 IoU/HOTA/ID-switch | 未报告 | mask drift、错对象、遮挡后重现 | 未说明 | HOI/sub-task/caption 级联错误 |
| Sub-task boundary | HOI track | 无 boundary F1 | 未报告 | 并行/长接触/恢复被误切 | 未说明 | chunk/option 时序错位 |
| Caption | fine-tuned Qwen3-VL | 4 个 LLM-judge example，无 aggregate | 未报告 | 动作幻觉、fold/unfold、顺序 | 未说明 | task goal 可能反转 |
| Object state/contact | 未提供 | — | 100% 缺失（作为显式标签） | — | — | 无法直接训练物理结果/力控 |

论文没有报告任何模态的 missing rate、置信度文件、自动 reject rate 或人工 QC，因此训练前必须自行抽样。最重要的不是只验证平均质量，而是建立失败类型的条件分布：按距离、遮挡、对象材质、动作速度、双手、camera angular velocity 和 task category 分层。

## 22. 评测：论文证明了什么、没有证明什么

### 22.1 已有评测

| 评测 | 证据 | 能证明 | 不能证明 |
| --- | --- | --- | --- |
| Dataset comparison | Table 1 | 模态/规格组合丰富 | 实际精度、法律开放、robot utility |
| Task distribution | Figure 4 | 1680 h 的大类分布与清洁/整理长尾 | 独立 site/operator/object 泛化 |
| Label frequency | Figure 5 | caption 词汇覆盖广 | skill/contact coverage |
| Feature distribution | Figure 6 t-SNE | qualitative visual/semantic breadth | 定量 density/generalization |
| 2D hand | Figure 7 | selected EgoLive overlay 较好 | 全库 2D accuracy、对 EgoDex 公平优越性 |
| 3D hand | Figure 8 | selected hand 与同源点云一致 | independent MPJPE/遮挡鲁棒性 |
| Depth | Figure 9 + Table 2 | 标定板 0.5–3.5 m 的 depth error | 动态真实对象 dense accuracy |
| Real scenes depth | Figure 10 | 三个场景能重建 | failure rate/temporal consistency |
| Caption | Figure 11 + Table 3 | 成功与失败类型存在 | 全库语言准确率 |

### 22.2 三类 gap 的必要对照全部缺失

论文没有比较：

- human-only、robot-only、co-training；
- 无 contact、补 contact、oracle contact；
- 无 dynamics、domain randomization、system-ID/real post-train；
- oracle hand/object motion 与自动伪标签；
- zero-shot/few-shot、多机器人/多手型；
- unseen task/object/site/operator；
- skill composition、language correction、long-horizon success；
- robot task success、retarget error、collision、contact force 或 sim-to-real gap。

所以“有利于 human-to-robot alignment/humanoid policy”是合理研究方向，不是实验结论。

### 22.3 应补的 benchmark

| 维度 | 最小指标 |
| --- | --- |
| Hand | 2D PCK/AUC、MPJPE/PA-MPJPE、wrist drift、valid rate；按遮挡/距离/速度分层 |
| Camera | ATE/RPE、scale/yaw drift、relocalization failure；与 motion-capture/高精 SLAM 对照 |
| Depth | AbsRel/RMSE/threshold、边界与 temporal flicker；真实材质分层 |
| Mask/track | mIoU、HOTA、ID-switch、occlusion recovery |
| Boundary/caption | boundary F1、hand/object/action/global accuracy、人类 agreement、calibration |
| Embodiment | IK/retarget success、EEF/object error、collision/contact preservation |
| Task | unseen object/site/task composition、long-horizon SR、recovery |
| Reality | sim-to-real delta、force/slip、latency/负载/设备差异 |

## 23. 采集经济性、扩展能力与存储压力

论文称简单头戴硬件具有 low equipment cost、pipeline 可持续增长，却没有报告 BOM、采集/处理成本或产率。仅由规格即可看出扩展压力：

- 每相机约 3.63 亿帧，双目约 7.26 亿帧；
- 若粗略按未压缩 RGB 计算，理论原始量约 10.16 PB；实际视频会强压缩，但 depth、point cloud、masks、MANO、两目和中间产品又增加多副本；
- 60 Hz 2160² stereo 的 GPU decoding、FoundationStereo 和 Qwen3-VL inference 成本远高于单目 30 Hz；
- 1680 h / 65,866 episodes 的平均 91.8 s 不代表有效 contact 时长。

总拥有成本应按：

$$
\mathrm{TCO/h}_{effective}
=
\frac{
C_{hardware}+C_{operator}+C_{travel/site}+C_{upload}
+C_{GPU}+C_{review}+C_{storage}+C_{governance}
}{H_{accepted\ interaction}}.
$$

论文没有任何分子项，也没有 $H_{accepted\ interaction}$，所以不能从“轻量头戴”推导“每有效小时低成本”。真实服务/门店场地授权、双目 calibration、GPU 标注、人工抽检、隐私处理和存储可能比设备本身更贵。

### 23.1 平台扩展性未披露项

| 能力 | 论文状态 |
| --- | --- |
| 多人并发/设备数 | operator/device/site 数未报告 |
| 跨设备标定 | 全部似乎用定制 JoyEgoCam，但设备间误差未报告 |
| 数据版本 | 无 public version/revision/manifest |
| 重复检测 | 未报告 |
| 自动去隐私 | 未报告 |
| 失败重跑/lineage | 未报告 |
| 社区贡献 | 数据由 recruited operators 采集，非公开 contributor workflow |
| 数据删除/纠错 | 未报告 |
| 发布吞吐 | 登录市场入口，无匿名可见文件统计 |

“Designed to sustain continuous growth”是系统目标，不是 production benchmark。

## 24. 隐私、治理与许可证

论文在真实家庭、零售和药房采集，却没有 ethics/privacy/consent section。全文未报告：

- operator informed consent、补偿与撤回；
- household members、customers、coworkers 和旁观者同意；
- face/body/voice/screen/receipt/prescription/address 的脱敏；
- 药房/零售中的敏感健康、支付和库存信息；
- raw IMU/camera trajectory 的位置/行为 re-identification；
- 数据保留期、访问控制和安全事件；
- 删除 raw 后 masks/depth/caption/checkpoint 的同步删除；
- 危险行为、刀具、清洁剂或敏感工作流程的使用政策；
- 人群 demographic 和手部重建公平性。

### 24.1 真实业务场景的特殊风险

家庭数据可能暴露室内布局、财产和日常习惯；零售数据可能拍到消费者、价格、订单或支付界面；药房数据可能触及药名、处方、患者信息或受监管流程。即使双目视频去脸，场景布局、服装、工位、camera trajectory 与时间组合也能 re-identify。

### 24.2 许可证结论

- arXiv 页面显示 **CC BY 4.0**，这覆盖论文文本和图；
- 数据市场入口匿名不可见商品详情；
- 未发现 public dataset card/README/LICENSE；
- 未知是否允许 commercial use、redistribution、derived annotations、model training、benchmark hosting；
- 未知第三方场景/对象/人员权利如何处理；
- MANO 和各 foundation models 还受各自许可约束。

在获得数据前，研究/企业使用者应要求书面 dataset license、DPA/consent summary、撤回与派生数据政策。当前不能仅凭论文中的 “open-source” 承担商业或再分发决策。

## 25. 开源代码与数据审查

| 核验项 | 论文声称/需要 | 匿名公开证据 | 判定 |
| --- | --- | --- | --- |
| 1680 h raw stereo | dataset scale | JD market 需登录，无法浏览 | 未核验 |
| Calibration/IMU/timestamps | raw multimodal data | 无 sample/schema | 未核验 |
| Camera pose | 提供 | 无字段/坐标/单位 | 未核验 |
| 3D hand/MANO | 提供 | 无 sample/valid mask | 未核验 |
| Depth/point cloud | 提供 | 无 file tree/encoding/confidence | 未核验 |
| Hand/object masks | 提供 | 无 schema/track ID | 未核验 |
| Sub-task/caption | 提供 | 无 JSON/语言/分数样例文件 | 未核验 |
| Processing scripts | 应用于 1680 h | 无 official GitHub | 未公开 |
| Hardware CAD/firmware | 复刻 JoyEgoCam 所需 | 无 | 未公开 |
| Retargeting/sim assets | human-to-robot 所需 | 无 | 未公开 |
| Training/evaluation | 证明 robot utility 所需 | 无 | 未公开 |
| Checkpoints/logs | baseline 所需 | 无 | 未公开 |
| Robot interface | 真机所需 | 无 | 未公开 |
| Dataset license | 合法使用所需 | 无公开文本 | 未核验 |

截至核验日，公开 GitHub API 对 `EgoLive egocentric robotics` 未找到官方仓库；公开搜索也未发现 HF/ModelScope 官方镜像。不能排除京东云登录后存在完整资产，但匿名证据只能支持“有论文和受控市场入口”。

### 25.1 可复现等级

- **论文阅读/图表**：完整可复现。
- **Depth 表格理解**：数字可复核，实验硬件/标定代码不可复刻。
- **下载并使用数据**：需要京东云账号及可能的申请/审核；匿名无法判断。
- **重跑 annotation pipeline**：基本不可复现，缺 glue code、models/weights、prompt、配置与硬件标定。
- **复现下游机器人结果**：不适用，因为论文没有下游结果。
- **全系统从采集到真机**：当前不可复现。

所以它目前既不属于“可完整复现”，也不能证实“仅能使用处理后数据”；更准确的是**论文可读、数据受控入口存在、内容与法律可用性需登录后另行审核、处理链不可复现**。

## 26. 分层复现路线

### Level 0：账户、许可与数据样本审计

| 项目 | 内容 |
| --- | --- |
| 数据 | 申请 JD Cloud，先取 5–10 episodes/sample manifest |
| 检查 | license、raw/derived rights、schema、coordinates、timestamps、missing/confidence、checksum |
| 算力 | CPU，1–5 TB 临时空间视 sample 大小 |
| 人力/周期 | 数据工程 + 法务各 1 人，1–2 周 |
| 成功标准 | 能下载；字段和单位可解析；条款允许目标用途；删除/再分发边界明确 |
| 风险 | 账号/资质、许可、完整 1680 h 不可得 |

### Level 1：单条 Ego 视频的姿态、物体与接触审计

| 项目 | 内容 |
| --- | --- |
| 数据 | 50–100 clips，覆盖距离/材质/遮挡/双手/柔性物 |
| 工具 | 自建 video–IMU–pose–depth–mask overlay；人工 contact/phase 小标注 |
| 算力 | 1×24–48 GB GPU，10–30 TB 存储/缓存 |
| 人力/周期 | 2 人，2–3 周 |
| 成功标准 | frame/timestamp 对齐；hand/depth/mask 分层准确率；observed/filled/invalid 可区分 |
| 风险 | 无原 pipeline，字段文档不足；contact 需人工补标 |

### Level 2：仿真 retargeting 与物理可行性

| 项目 | 内容 |
| --- | --- |
| 数据 | 500–2000 个高质量刚性对象 sub-tasks |
| 依赖 | object pose/mesh 补建、target URDF/MJCF、IK/dex-retarget、contact model |
| 算力 | GPU perception + CPU/MuJoCo parallel optimization |
| 人力/周期 | 3–4 人，4–8 周 |
| 成功标准 | IK ≥90%；collision <5%；EEF/object error 和 contact preservation 可报告 |
| 风险 | object/contact 缺失、human motion 不可达、camera/world schema 不清 |

### Level 3：Human–Robot co-training

| 项目 | 内容 |
| --- | --- |
| 数据 | 100–300 h 筛选 EgoLive + 每任务 50–200 robot demos |
| 模型 | 选一条 VLA/world-model，不同时追多架构 |
| 对照 | robot-only、visual、semantic、motion、full，固定 token/steps/compute |
| 算力 | 4–16×80 GB GPU，依模型 2–6 周 |
| 人力/周期 | 4–6 人，2–3 个月 |
| 成功标准 | ≥3 seeds；site/object/task held-out；SR 显著提升且无 negative transfer |
| 风险 | 背景 shortcut、伪标签噪声、数据许可、存储吞吐 |

### Level 4：真机 whole-body / dexterous deployment

| 项目 | 内容 |
| --- | --- |
| 数据 | Level 3 policy + real robot force/tactile/recovery rollouts |
| 系统 | System2 workflow，System1 feasible chunks，System0 contact/impedance safety |
| 平台 | humanoid/EX002/dex hand，完整标定/system ID |
| 人力/周期 | 6–10 人，3–6 个月起 |
| 成功标准 | unseen site/object SR、恢复率、安全干预、力峰值、跨负载稳定性 |
| 风险 | whole-body/contact/dynamics 均非 EgoLive 真值，硬件安全成本高 |

## 27. 面向 Humanoid、EX002 与灵巧手的落地建议

### 27.1 Humanoid

| 项目 | 建议 |
| --- | --- |
| 最有价值模态 | 长工作流 caption/sub-task、双手运动、camera active-view、depth/scene geometry |
| 必须补齐 | body/head/base/feet、ground contact、object pose/state、force/tactile、robot dynamics |
| 可直接用 | System2 workflow pretrain、双臂 EEF prior、视觉/active perception 表征 |
| 只能提 intent | 行走、弯腰、铺床、承重、全身接触；camera pose 不能直接当 base |
| 推荐试点 | 固定站位的桌面/货架整理、柜门/抽屉开合，先避开床品 |
| 桥接数据 | 每任务 100–300 human clips + 50–100 humanoid teleop/recovery demos |
| 成功标准 | unseen site/object SR +10 pp；零跌倒；接触力/干预率受控 |
| 责任 | Data/System2 做 workflow；motion 做 whole-body retarget；control 做 balance/System0 |

### 27.2 EX002

| 项目 | 建议 |
| --- | --- |
| 最有价值模态 | 3D wrist path、camera/depth、object mask、sub-task/caption |
| 必须补齐 | EX002 TCP/相机外参、object 6D/state、reachability、gripper/contact labels |
| 可直接用 | object-conditioned EEF prior、动作检索、场景视觉预训练 |
| 只能提 intent | 多指人手形、柔性物、擦拭力、复杂工具使用 |
| 推荐试点 | 药品盒/日用品分拣、货架摆放、刚性物表面清洁 |
| 桥接数据 | 每任务 100+ Ego clips + 30–50 EX002 robot demos |
| 成功标准 | held-out layout/object SR；grasp drop、collision、恢复时间优于 robot-only |
| 责任 | perception 补 object state；planning 做 functional retarget；control 做接触与安全 |

EX002 若使用夹爪，MANO finger pose 的大部分信息不会保留；应优先提取 object-centric grasp region 与 wrist corridor，而非视觉上复制人手。

### 27.3 Dexterous hand

| 项目 | 建议 |
| --- | --- |
| 最有价值模态 | stereo MANO/3D joints、两目图像、object mask/depth、双手/长序列 |
| 必须补齐 | independent hand GT、object mesh/pose、contact topology/normal、force/tactile/slip |
| 可直接用 | pre-grasp hand prior、视觉表征、dex-retarget 初始化 |
| 只能提 intent | 衣物/床品张力、擦拭力、透明/反光物、工具稳定接触 |
| 推荐试点 | 刚性盒/瓶、药盒分拣、简单盖子操作；不要先做叠被/铺床 |
| 桥接数据 | 500–2000 高置信 clips + 每对象族 50–100 tactile robot trials |
| 成功标准 | fingertip/contact region error、slip、grasp retention 和 task SR 共同提升 |
| 责任 | hand perception 补 contact；retarget 做 topology；System0 做 tactile/impedance |

## 28. Related work 与 EgoLive 的准确生态位

### 28.1 通用 Ego 数据

[EPIC-KITCHENS-100](https://epic-kitchens.github.io/2021)和 [Ego4D](https://ego4d-data.org/)的规模/语言/生活广度强，但缺统一 camera/hand/depth 操作几何。EgoLive 用定制硬件换取多模态密度，却没有它们成熟的 benchmark、数据治理和公开工具生态证据。

### 28.2 Manipulation-centric 数据

[EgoDex](https://arxiv.org/abs/2505.11709)、[HOT3D](https://www.projectaria.com/datasets/hot3d/)和 [HOI4D](https://hoi4d.github.io/)更强调桌面手物 tracking。EgoLive 的差异是服务/零售/家庭长工作流和高规格 stereo+IMU；代价是手/对象真值和公开 schema 远不如硬件标定明确。

### 28.3 Deployment-scale 数据

Egocentric-10K 更大但标注稀疏；Xperience-10M 的当前可用规模、低分辨率与多模态构成不同。EgoLive 的优势是 60 Hz/2160²/stereo 和任务型工作流，不能只用名义小时或 t-SNE 宣称全维度优越。

### 28.4 Human-to-robot 方法

[EgoMimic](https://arxiv.org/abs/2410.24221)、[EgoBridge](https://arxiv.org/abs/2509.19626)、[H-RDT](https://ojs.aaai.org/index.php/AAAI/article/view/35724)、[Being-H0.5](https://arxiv.org/abs/2601.12993)与 [EgoHumanoid](https://arxiv.org/abs/2602.10106)才负责 co-training、latent alignment、retarget 或 whole-body execution。EgoLive 是这些方法的潜在数据源，不能把引用它们等同于自己完成 transfer。

## 29. 最强隐含假设、局限与改进

### 29.1 优点

1. 双目 60 Hz + 200 Hz IMU 在大规模 Ego 数据中确实少见，直接增强 metric geometry 与快速动作可观测性。
2. 真实服务/商业工作流比实验室 tabletop 更接近部署任务，1–3 分钟 episode 也保留较长程序结构。
3. camera、hand、depth、mask、sub-task 和 caption 同源同步，为多任务/多模态预训练留下较大空间。
4. Table 2 完整公开 16 个距离档，而不是只用“毫米级”口号；可以看到远距离退化。
5. Table 3 主动展示错误 caption，而不只选全对样例，暴露 action/global failure。

### 29.2 局限性

1. 最大隐含假设：丰富 human intermediate labels 会自动转化为 robot generalization；论文没有下游实验。
2. 最大姿态误差证据缺口：hand 只与同源 depth point cloud 定性对齐，无独立 GT/MPJPE/缺失率。
3. 最大 contact 限制：object mask/HOI detector 被描述得接近 interaction state，但没有真实接触或力。
4. 最大 embodiment 限制：无 robot action/retarget/object goal/可执行性。
5. 最大 reality gap：无 dynamics/system ID/sim/真机数据；真实视觉只解决感知域。
6. 最大数据泄漏风险：无 operator/site/session split，相邻 sub-tasks 和同一场景很可能跨 split。
7. 最大开放问题：数据入口需登录，无命名数据许可、公开 manifest/schema/代码。
8. 最大分布问题：`Others` 占 32.9%，operator/site/demographic/失败恢复均不明。
9. 最大评测问题：t-SNE 被用来支持 coverage/density，caption 只有四例，手部只有 qualitative comparison。
10. 最大治理问题：真实家庭/零售/药房却没有 consent/privacy/withdrawal 描述。

### 29.3 潜在改进

1. 公开 versioned dataset card、schema、sample browser、manifest/checksum 和命名 license。
2. 给 operator/site/task/object/session-disjoint splits 与 near-duplicate audit。
3. 发布 observed/interpolated/invalid 和 per-modality confidence，不只交最终伪标签。
4. 建 independent GT subset：Mocap hand、LiDAR/depth、camera trajectory、人工 mask/boundary/caption。
5. 补 object pose/articulation/state 和 contact region/phase，代表性子集再加 force/tactile。
6. 报 t-SNE 之外的 effective rank、coverage、kNN mixing、silhouette 与等量 subsampling。
7. 对 1680 h 做 raw→accepted 产率、失败分布、GPU/人工/TCO 审计。
8. 做固定计算量的 RGB/language/hand/camera/depth 因果消融与 human:robot scaling。
9. 多本体真机测 retarget feasibility、task SR、collision/contact、sim-to-real。
10. 补隐私、旁观者、药房敏感信息、撤回、派生删除与 demographic performance。

## 30. 十个组会质疑

1. **1680 h 是否全部可下载？** 论文给市场总入口，不给 EgoLive 具体公开商品页、manifest 或匿名下载证明。
2. **65,866 episodes 是如何 QC 的？** 没有 raw 时长、拒绝率、无手/等待/重复率。
3. **346 tasks 如何定义？** taxonomy、每类数量、同一 task 跨多少人/site/object 未给。
4. **3D hand 为什么没有 MPJPE？** 与同源 stereo depth 对齐不能作为独立 GT。
5. **Caption 准确率是多少？** 四个 sample 中 action/global 各 2/4，但没有 aggregate 和 judge-human agreement。
6. **t-SNE 为什么能证明 density？** 没有参数、等量抽样或定量 manifold metric。
7. **“interacted object mask”是否被误当 contact？** 没有 surface distance/force/phase，不能。
8. **为什么没有任何 robot baseline？** 数据论文不必提出控制器，但若声称适合 robot learning，至少应做固定模型/robot data 的 causal test。
9. **真实药房/零售如何处理隐私和旁观者？** 论文完全未说明。
10. **Open-source 的许可证和代码在哪里？** arXiv CC BY 只覆盖论文，JD 登录入口不能替代 public data license/README。

## 31. 最终评分卡与判断

| 维度 | 评分（1–5） | 证据 |
| --- | ---: | --- |
| 名义规模 | 5 | 1680 h、65,866 episodes、346 tasks |
| 真实场景/工作流 | 5 | 家庭服务、零售、药房、清洁/整理等；但 site/operator 数未知 |
| 采集硬件 | 5 | stereo 2160²@60 Hz、130°² FOV、IMU 200 Hz |
| 模态丰富度 | 4 | camera/hand/depth/masks/sub-task/caption；无 body/object 6D/contact/force |
| 标注准确性 | 2 | depth 标定强；hand/caption/mask/pose 缺系统定量和缺失率 |
| Robot 可执行性 | 1 | 无 robot action、retarget、policy、真机实验 |
| Task 价值 | 4 | 长 workflow 与语义长尾；无 composition/downstream evidence |
| Contact/physics | 1 | 无 explicit contact、force、tactile、object dynamics |
| 开放程度 | 1 | 登录市场入口；无公开 data card/license/code/schema/mirror |
| 治理 | 1 | 真实敏感场景，论文无 privacy/consent/withdrawal |
| 复现难度 | 1 | 定制硬件与内部 pipeline 未开放 |
| 业务价值 | 3 | 若条款/质量/schema 通过审核，System2/1 预训练潜力高；当前不确定性大 |

### 三类 gap 的最终账本

- **Embodiment gap：约 1/5**。只恢复 human hand/camera state，未形成 robot action 或可执行性。
- **Task gap：约 2/5**。工作流、子任务和语言覆盖强，但无 latent composition/generalization。
- **Reality gap：约 1/5（感知侧有帮助，控制侧为 0）**。真实 RGB/depth 缩小视觉域，robot dynamics 未触及。
- **Semantic–motion alignment：约 2/5**。多模态共时存在，但 caption/3D/contact 没有联合定量。
- **Motion–contact alignment：0/5**。没有 explicit contact/force/object state。

### 一句话模板结论

> 这篇工作将 Ego 数据中的 **真实服务工作流、双目场景几何、手部运动与层级语义** 转换为机器人可利用的 **视觉/语义/运动预训练中间表征**；它只通过 **MANO/3D hand 候选**为 Embodiment gap 提供输入，通过 **346 tasks 和 1–20 秒 sub-task captions**部分缩小 Task gap，并仅以 **真实 RGB、stereo depth 和 camera pose**缓解感知侧 Reality gap；其上限取决于 **数据真实开放程度、独立标注精度、object/contact/force 补全，以及 robot co-training/retarget/真机证据**。

### 数据集最终判词

> 该数据集以 **定制 JoyEgoCam 双目 2160²@60 Hz + 200 Hz IMU 在真实家庭、服务、零售和药房工作场景采集** 的方式获得 **论文声称的 1680 小时、65,866 episodes、346 tasks 多模态第一视角经验**，其中最可靠的机器人监督是 **高规格真实视觉、近距离 stereo depth、长工作流语义和可作为先验的 hand/camera motion**；最缺失的物理信息是 **object 6D/state、真实 contact/force/tactile、whole-body 与 robot action/dynamics**，因此它最适合用于 **System2/1 的视觉—工作流—人手运动预训练与后续 human–robot 桥接研究**，不应直接被当作 **已验证可执行、具备接触动力学真值的机器人示范数据集**。

EgoLive 最值得肯定的是把“真实工作任务 + 高规格 stereo/IMU + 多模态自动标注”放到了同一数据工程目标中；最需要克制的是把潜在可用性写成已经实现的机器人能力。当前证据证明了一套有吸引力的数据资产设计，而不是一条已经闭环的 human-to-robot 学习路线。

## 32. 链接索引

- 论文：[arXiv](https://arxiv.org/abs/2604.23570) · [HTML](https://arxiv.org/html/2604.23570v1) · [PDF](https://arxiv.org/pdf/2604.23570)
- 数据入口：[JD Cloud Robot Data Market](https://robotdata-market.jdcloud.com/console/market)（匿名访问会跳转登录）
- 关键上游：[HaMeR](https://github.com/geopavlakos/hamer) · [MANO](https://mano.is.tue.mpg.de/) · [ORB-SLAM3](https://github.com/UZ-SLAMLab/ORB_SLAM3) · [BoT-SORT](https://github.com/NirAharon/BoT-SORT) · [SAM 2](https://github.com/facebookresearch/sam2) · [FoundationStereo](https://github.com/NVlabs/FoundationStereo) · [Qwen3-VL](https://github.com/QwenLM/Qwen3-VL)
- 相关数据：[EgoDex](https://arxiv.org/abs/2505.11709) · [HOT3D](https://www.projectaria.com/datasets/hot3d/) · [HOI4D](https://hoi4d.github.io/) · [Ego4D](https://ego4d-data.org/) · [EPIC-KITCHENS](https://epic-kitchens.github.io/2021)
- Human-to-robot 方法：[EgoMimic](https://arxiv.org/abs/2410.24221) · [EgoBridge](https://arxiv.org/abs/2509.19626) · [Being-H0.5](https://arxiv.org/abs/2601.12993) · [EgoHumanoid](https://arxiv.org/abs/2602.10106)
- 未发现：EgoLive 官方独立项目页、GitHub 代码仓库、HF/ModelScope 镜像、公开 dataset license、baseline checkpoint、retarget/sim/真机接口。
