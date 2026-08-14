---
title: "Open-AoE: An Open Egocentric Manipulation Dataset and Toolchain for Embodied Learning"
method_name: "Open-AoE"
authors: [Zishuo Li, Bowen Yang, Changtao Miao, Kai Zhu, Hao Chen, Qingze Guan, Zhengxing Wu, Wanke Zhan, Yang Sun, Zhiyi Huang, Zitong Shan, Zhenchao Jin, Jiadong Hong, Taowen Wang, Yushi Feng, You Liu, Yibo Wang, Yifan Yang, Zhaowen Zhou, Man Luo, Hao Cheng, Bo Zhang, Jianshu Li, Jiansheng Cai, Guocai Yao, Jize Zhang, Chenhao Lin, Renjing Xu, Lequan Yu, Chao Shen, Chunhua Shen, Zhe Li]
year: 2026
venue: arXiv
tags: [egocentric-data, human-to-robot, embodied-learning, mano, camera-trajectory, atomic-action, retargeting, world-model, vision-language-action, humanoid, dexterous-manipulation]
image_source: online
---

# Open-AoE：2000 小时第一视角数据，离“机器人可执行经验”还有多远？

> 本笔记基于 [arXiv:2607.14183v2](https://arxiv.org/abs/2607.14183)、[官方技术报告 PDF](https://github.com/ant-research/Open-AoE/blob/main/Open-AoE-tech-report.pdf)、[官方代码仓库](https://github.com/ant-research/Open-AoE)、[Hugging Face 数据卡](https://huggingface.co/datasets/inclusionAI/OpenAoE-2000h) 与 [ModelScope 数据页](https://www.modelscope.cn/datasets/inclusionAI/OpenAoE-2000h) 精读和核验。公开状态核验日期为 **2026-08-13**。论文中的规模声明、随机 100 小时分析、当前已上传数据和代码仓库 README 中的追加实验被严格分开；后者不是论文实验。

## 阅读结论先行

Open-AoE 是一项很有工程密度的数据基础设施工作：它把 500 多名贡献者、400 多种消费级手机和 400 多个场景中的约 2000 小时胸前第一视角视频，处理成原始/去畸变视频、度量尺度相机轨迹、逐帧双手 MANO、有效性掩码以及带时间边界、左右手、动作动词、对象、描述、框和置信度的原子动作。它还开放了可视化、重建—重定向和 14 类下游模型适配配方，试图让数据从“能看”走向“能转、能训”。

它最可靠的机器人监督不是接触、力或机器人关节动作，而是三类中间信号：**第一视角视觉分布、英语原子动作语义、相机系/世界系中的手腕与手指运动先验**。相机轨迹和 MANO 都是模型重建结果，不是 Mocap 真值；物体没有统一的 6D pose、mesh、articulation 或状态轨迹；接触没有点、法向、力、触觉或抓取稳定性标签。把 Open-AoE 称为“robot action dataset”会越过证据边界。

论文的数据分析做得比通常的数据集报告更细：在等量随机 100 小时上，它用 CLIP 谱、共享聚类、kNN 域混合、时间覆盖、Idefics2 图文一致性与训练窗口产率比较 OpenEgo、EgoDex 和 EgoXtreme。但这些结果主要说明**表征多样性、标注密度与数据可切窗性**；论文没有报告任何 VLA/策略的真机成功率，没有 human-only / robot-only / co-training 因果对照，也没有证明 CLIP 多样性会变成未见机器人任务的成功率。

“Open”也必须拆开看。截至核验日，nano 约 3 小时和 tiny 约 100 小时已发布，full 2000 小时仍在上传；官方 2026-08-12 记录是 ModelScope 侧约 694 小时就绪，而 Hugging Face 等待扩容。代码是 Apache-2.0，但数据卡许可证字段为 `other`，根目录 `LEGAL.md` 没有给出一份明确的数据集许可。因此，**公开下载进度、软件开源和数据法律可复用性不是同一件事**。

### 一句话总结

Open-AoE 用消费级手机和重型离线重建，把大规模自然人类操作视频加工成视觉、语义、相机和双手运动的训练中间层；它很适合 VLA/world-model 预训练与 retargeting 研究，但缺少物体状态、真实接触/力、机器人动作和论文级真机因果证据，不能直接视作可执行机器人示范。

### Elevator pitch

机器人数据的瓶颈不只是采集贵，也在于不同下游模型各自需要不同格式。Open-AoE 的回答是：前端用手机扩大人、场景和任务覆盖，后端用 SLAM、双手重建、VLM 切分与质量门控生成统一数据产品，再提供从 MANO 到 humanoid/dexterous hand、从视频到 VLA/world model 的适配器。它没有解决所有 human-to-robot gap，却把过去散落的“采集—重建—可视化—重定向—训练转换”连接成了一个可审计的开放起点。

![Open-AoE 总览：从手机第一视角采集到重建、重定向和训练配方](https://arxiv.org/html/2607.14183v2/figure1_v5.png)

## 0. 资源、版本、公开规模与许可

| 资产 | 正式入口 | 截至 2026-08-13 的实际状态 | 许可/条款 | 可复现判断 |
| --- | --- | --- | --- | --- |
| 论文 | [arXiv 摘要](https://arxiv.org/abs/2607.14183)、[HTML](https://arxiv.org/html/2607.14183v2)、[PDF](https://arxiv.org/pdf/2607.14183) | v2，25 页技术报告；9 张主图、2 张主表 | 论文页面许可只覆盖论文，不自动覆盖视频和代码 | 高 |
| 官方技术报告副本 | [GitHub PDF](https://github.com/ant-research/Open-AoE/blob/main/Open-AoE-tech-report.pdf) | 与 arXiv 技术报告内容对应 | 随仓库分发，但仍应按论文许可理解 | 高 |
| 代码总仓 | [ant-research/Open-AoE](https://github.com/ant-research/Open-AoE) | 审计提交 `8f90dedc7c2092caad791586f600fe7a8286b546`；含可视化、重建/重定向、训练配方、机器人资产和补丁 | 项目原创代码 Apache-2.0；第三方组件各自许可 | 中高 |
| Hugging Face 数据 | [inclusionAI/OpenAoE-2000h](https://huggingface.co/datasets/inclusionAI/OpenAoE-2000h) | nano ≈3 h、tiny ≈100 h 已发布；full 上传中 | 数据卡标记 `License: other`，没有命名许可正文 | 技术上可下载，法律边界不清 |
| ModelScope 数据 | [OpenAoE-2000h](https://www.modelscope.cn/datasets/inclusionAI/OpenAoE-2000h) | 2026-08-12 发布说明称约 694 h 已就绪；其余至 2000 h 仍上传 | 页面未弥补命名数据许可缺口 | 部分可用 |
| 可视化 | [`aoe-visualization`](https://github.com/ant-research/Open-AoE/tree/main/aoe-visualization) | 2D MANO/关节/未来手腕/action overlay、3D world/timeline viewer | glue code Apache-2.0；MANO 模型需单独注册下载 | 中高 |
| 重建与重定向 | [`aoe-reconstruct-retarget`](https://github.com/ant-research/Open-AoE/tree/main/aoe-reconstruct-retarget) | EgoInfinity、Do-as-I-Do、SPIDER 与多机器人/手的适配矩阵 | 上游代码、模型、权重、机器人资产许可混合；多数不随仓库提供 | 中 |
| 训练配方 | [`aoe-training-ready`](https://github.com/ant-research/Open-AoE/tree/main/aoe-training-ready) | ACT、Diffusion Policy、π0.5、SmolVLA、GR00T、H-RDT、VITRA 与多种 world model/LAM 的 converter/launcher/patch | 不是统一 SDK；依赖外部 checkout 和权重 | 中 |
| 预训练 checkpoint | 分散在各上游项目 | Open-AoE 总仓没有一套统一、完整的 Open-AoE 下游 checkpoint 发布 | 各模型卡许可 | 低到中 |
| 采集 App | 论文只给 App/流程描述；未给安装包或源码入口 | 可作为平台采集工具使用的证据来自论文；未发现可自行构建的开源 App | 未报告 App 许可 | 低 |

### 0.1 “2000 小时”应如何读

官方数据卡给出的发布路线是：

| Tier | 名义时长 | 当前状态 | 可以据此说什么 |
| --- | ---: | --- | --- |
| nano | 约 3 h | 已发布 | 可做格式、加载、可视化 smoke test |
| tiny | 约 100 h | 已发布 | 可复现论文的等量 100 h 级数据分析或小规模训练 |
| full | 2000 h | 上传中 | 这是目标/总库规模，不是当前在每个平台完整可下载的规模 |
| batch 4–6 | 约 694 h | 2026-08-12：ModelScope 就绪，HF 等待扩容 | 是当前官方发布说明中的阶段性可用量，不等于全量 |

此外，2026-07-30 官方曾因“相机内参与视频分辨率不一致”删除被标记样本；2026-07-31 记录约 323 小时上传。这说明数据仍处于活跃纠错与版本增长阶段，复现实验必须记录 hub、commit/revision 和下载日期，不能只写“Open-AoE-2000h”。

### 0.2 标题与材料边界

arXiv 正式标题是 **“Open-AoE: An Open Egocentric Manipulation Dataset and Toolchain for Embodied Learning”**；技术报告首页图形标题写作 **“Open-AoE: Open-Source Egocentric Data and Toolchain for Embodied AI”**。本笔记以 arXiv 元数据为准。论文主实验只包括数据属性分析；仓库在其后补充的 POC 训练数字单列为“仓库证据”，不回写为论文结论。

## 1. 论文速览与核心贡献

| 项目 | 判断 |
| --- | --- |
| 分类 | Ego dataset + data processing infrastructure + retargeting/training toolchain |
| 生态位 | 位于大规模被动 Ego 视频与昂贵机器人遥操作之间；比纯视频数据多相机/手运动，比 robot demo 少物体/contact/robot action 真值 |
| 目标用户 | VLA、world model、human-to-robot、手部重建、retargeting、humanoid/dexterous manipulation 团队 |
| 论文宣称规模 | 约 2000 h、500+ contributors、400+ smartphone models/types、400+ scenes、8000+ tasks |
| 分析规模 | 从 Open-AoE 随机抽 100 h，并与 OpenEgo、EgoDex 等量比较；不是全 2000 h 的逐项统计 |
| 三个关键词 | scalable smartphone collection；metric MANO + camera trajectory；training-ready adapters |
| 核心贡献 1 | 把端侧采集、离线筛选/隐私处理、VLM 切分、SLAM、双手重建与 QC 组织为一条可扩展数据生产线 |
| 核心贡献 2 | 发布原始/去畸变视频、相机轨迹、双手 MANO、原子动作与有效性/置信度字段 |
| 核心贡献 3 | 开放可视化、12-cell 重定向矩阵与 14 类训练适配配方，降低数据接入摩擦 |
| 最强论文证据 | 等量 100 h 下，Open-AoE 的 CLIP 有效秩、参与率和 kNN 域混合均在 3000 次重采样中排名第一；双手有效帧率 98.02% |
| 最可疑 claim | “8000+ tasks”没有受控 task ontology；更接近自然语言任务/上下文或聚合标签，而非 8000 个经独立验证的机器人任务类 |
| 最大技术缺口 | 无 object pose/articulation/contact/force/robot action；相机和手均为伪标签；无论文级策略成功率 |
| 最大开放缺口 | full 未传完；数据许可未命名；App、全部上游模型/权重、MANO 资产不随仓库提供 |
| 精读优先级 | 5/5：当前大规模 Ego-to-robot 数据工程的代表性基础设施 |
| 复现优先级 | 4/5（tiny 分析/转换）；2/5（全量和所有下游配方） |
| 业务优先级 | 4/5：适合先做 representation/world-model/动作先验，不适合直接替代机器人接触数据 |

论文真正建立的命题是：

> 在相同 100 小时预算下，Open-AoE 具有较宽的视觉表征支持、更密的原子语义、较完整的相机与手部模态，并能被多种现有训练栈转换消费。

它没有建立的命题是：

> 这些属性必然提高任意机器人、任意任务、任意接触条件下的真机成功率。

## 2. Gap—Evidence 总表

| Gap | 作者机制 | 直接指标 | 对应消融 | 真机证据 | 是否解决 |
| --- | --- | --- | --- | --- | --- |
| Embodiment | MANO、21 joints、手腕/指尖 action；IK/dex-retarget；多机器人 recipe | 28-DoF inverse-kinematics failure 作为 QC；工具链演示 | 无统一的多本体任务 SR；无同一动作跨机器人误差表 | 论文无 | **部分**：提供转换接口，不等于功能可执行性 |
| Task | VLM 原子动作、verb/object/scene/description；LAM/world-model recipes | 语义数量、时间密度、图文一致性 | 无 held-out task composition 或 language coaching | 无 | **部分**：扩大语义覆盖，未验证 skill composition |
| Reality | 真实手机视频；world/camera metric reconstruction | SLAM/hand 质量门控；论文没有 sim-to-real 指标 | 无 dynamics/system-ID/domain randomization 对照 | 无 | **否**：数据是真实视觉，但 robot dynamics gap 未处理 |
| Semantic–motion | 同 clip 中对齐动作文本、bbox、hand attribution、MANO | Idefics2 4.583/5；时间覆盖 99.99% | 无人工 GT 的语义—3D motion alignment accuracy | 无 | **部分**：时空共现，不是因果意图对齐 |
| Motion–contact | 手姿与动作片段；机器人化视频/retargeting | 无 contact metric | 无 contact/force/tactile；无物体轨迹 | 无 | **否**：只能近似 contact intent，不能恢复真实接触 |
| Scale/open | 500+ 人、400+ 手机类型、数据 hub、代码仓库 | 2000 h 名义规模；当前分批上传 | 无成本/接受率/重复率；full 未完成 | 不适用 | **部分**：工程资产开放较强，数据与许可尚未闭环 |

## 3. 数据资产：来源、统计单位与五层监督

### 3.1 数据来源

| 数据源 | 规模 | 视角/传感器 | Human state | World/interaction | 语言 | 用途 |
| --- | ---: | --- | --- | --- | --- | --- |
| Open-AoE 自采手机 | 论文约 2000 h；公开仍分批 | 胸前/第一视角 RGB，400+ 消费级手机类型 | 自动双手 MANO、21 joints、validity | metric camera trajectory；无统一 object state/contact | VLM 原子动作、scene、verb、object、description、bbox、confidence | 主数据资产 |
| 随机 Open-AoE 分析子集 | 100 h | 同上 | 同上 | 同上 | 46,802 atomic intervals 等统计 | 与其他 Ego 数据等量比较 |
| OpenEgo | 100 h 对照 | Ego video | 数据集已有手/相关标注 | 依原数据提供 | 26,864 native labels | 多样性、语义、训练窗口对照 |
| EgoDex | 100 h 对照 | Ego video | 数据集手信息 | 依原数据提供 | 111 categorical classes | 同上 |
| EgoXtreme | 可用部分 | 多源 Ego | 不完整/异构 | 不完整/异构 | 异构 | 补充对照；不是严格同 schema |
| 机器人遥操作 | 论文未提供 | — | — | — | — | 仓库部分训练配方可能需要，但不是 Open-AoE 数据本体 |
| Mocap/手套/力触觉 | 未提供 | — | — | — | — | 不能推断存在 |
| 仿真补全 | 工具链可生成/重定向 | MuJoCo 等 | robotized trajectory | 仿真几何由外部资产决定 | 可沿用动作语义 | 派生数据，不是传感器真值 |

### 3.2 五层监督审计

| 层次 | 实际提供 | 真值来源 | 完整性 | 机器人含义 |
| --- | --- | --- | --- | --- |
| Raw observation | raw RGB、undistorted RGB、intrinsics/device metadata | 手机传感器 | 高；没有公开深度、IMU、眼动、音频说明 | 视觉预训练、世界模型、相机域多样性 |
| Human state | wrist translation/rotation、45D MANO pose、10D shape、21 joints 派生、valid mask | detector + HaWoR/MANO + temporal completion | 高覆盖但为模型估计 | 手部几何/运动先验、retargeting 输入 |
| World state | camera-to-world trajectory、world-frame hands | DROID-W/SLAM + metric alignment | 部分；world 锚定首帧，无 object pose/mesh/articulation/depth release | 运动去相机化与相对几何 |
| Interaction state | atomic action、hand attribution、action bbox、scene/object phrase | Qwen3.7-Plus VLM + HITL | 语义密；无真实 contact、grasp phase、force/tactile | contact intent/affordance 的弱监督 |
| Robot supervision | 多种可转换 action representation、retargeting recipe | 从人手伪标签计算/IK 映射 | 不是 robot telemetry；可执行性依赖目标本体 | 预训练/初始化，不应当作真机控制真值 |

### 3.3 基本单位与名词边界

- **raw video**：贡献者手机录制的连续第一视角视频。
- **Part**：离线 QC 把连续好/坏区间切分后的可追踪片段；保留 lineage/process tag。
- **atomic interval/clip**：VLM 按动作边界生成的时间段，可含一条或少量 atomic action。
- **task**：论文的 “8000+ tasks” 没有给全局受控 ontology。随机 100 h 内另有 522 条 raw context description、175/193 个动词口径和 8030/7346 个 object 口径，不能把这些数字互换。
- **contributor ID**：匿名 ID；随机 100 h 中位贡献仅 2.6 分钟，但这不是全库贡献时长分布。
- **scene**：既有闭集标签，也有自然描述；“400+ scenes”与图中 kitchen/office 等大类不是同一统计层级。
- **episode**：下游 converter 可把 clip 变成 LeRobot episode，但原论文没有统一 train/val/test episode split。

### 3.4 小时数、有效时长与 split 风险

论文没有报告 2000 小时中：原始录制与通过 QC 的比例、等待/无手时长、重复动作比例、atomic clip 的净操作时长、失败/恢复时长、各发布批次是否互斥。随机 100 h 的时间覆盖率接近 100%，说明发布后的序列被标注几乎全覆盖；它不说明 raw capture 的有效产率。

论文也没有给正式 train/validation/test split，更没有说明按贡献者、房间、手机、对象或任务隔离。若随机按 clip 切分，相邻 Part、同一房间、同一操作者和同一手机可能跨 split，导致视觉与轨迹泄漏。下游实验至少应采用：

1. contributor-disjoint；
2. scene/household-disjoint；
3. object-instance-disjoint；
4. task-composition-disjoint；
5. device-family-disjoint；
6. 时间相邻 Part 成组分配。

## 4. 采集硬件、端侧协议与选择偏差

| 传感器/模块 | 安装位置 | 频率/分辨率 | 同步/标定 | 主要误差 |
| --- | --- | ---: | --- | --- |
| 消费级手机 RGB | 胸前/颈挂第一视角 | 发布卡：30 FPS；多数 1280×720，部分 1920×1080 | 每视频保存原始与去畸变内参；固定焦段 | rolling shutter、模糊、曝光、FOV/ISP 域差、热降频 |
| 端侧 hand detector | 手机上实时 | 未报告模型 FPS | 与相机帧同设备运行 | 漏检会造成数据选择偏差 |
| 端侧场景/手势检查 | 手机上实时 | 未报告 | 规则/模型组合 | 规定动作与佩戴姿态可能改变自然行为 |
| 低光/曝光控制 | 手机 ISP/应用 | 设备相关 | 自动补光和曝光策略 | 不同厂商 ISP 不一致，反光/透明物体仍困难 |
| 存储/热保护 | 手机系统 | 持续监控 | 剩余容量/温度阈值未报告 | 长时任务被截断，重负载设备分布偏差 |
| 相机轨迹 | 离线从 RGB 估计 | 与视频逐帧 | DROID-W 重调 + 全局优化 | 单目尺度、弱纹理、动态手物体、漂移 |
| 手部姿态 | 离线从 RGB 估计 | 逐帧，两手 | detector + HaWoR/MANO + temporal fill | 遮挡、截断、快速运动、左右手切换 |

端侧采集链大致包括：手检测触发/语音提醒、操作规范与手势/场景检查、佩戴稳定性检查、低光补光与曝光、固定焦段/防抖/去模糊/自适应 FPS，以及在存储不足或过热时停止。优点是把明显坏数据挡在上传前；代价是采样分布由端侧模型决定：被遮挡但真实发生的接触、动作幅度很小的旋拧/按压、手短暂离开画面的工具操作更容易消失。

“500+ contributors”证明采集并非单一实验室，但论文没有年龄、性别、身高、惯用手、职业、地区、文化、家庭结构或补偿信息。胸前视角也会系统性偏向可见双手、近工作台和室内桌面操作；它不是 whole-body 或 locomotion 数据。

## 5. 数据处理与标注流水线

![Open-AoE 数据管线：端侧筛选、离线 QC、VLM 标注、手和相机重建](https://arxiv.org/html/2607.14183v2/fig2-data-pipline.png)

~~~mermaid
flowchart LR
    A["手机 RGB 采集\n设备/内参元数据"] --> B["端侧门控\n手、场景、佩戴、曝光、热/存储"]
    B --> C["离线文件级 QC\n解码、旋转、曝光、时长、完整性"]
    C --> D["内容级 QC\n去无手/非 ego/jitter\n隐私擦除"]
    D --> E["手可见性复查\n截断/遮挡/内参检查"]
    E --> F["切连续 good Parts\n固定帧率 + lineage tag"]
    F --> G["Qwen3.7-Plus\n合规/动作有效 + 四层标签"]
    F --> H["DROID-W\n相机轨迹"]
    F --> I["双手 detector + HaWoR\nMANO + metric alignment"]
    H --> J["动态 mask + sliding window\n全局 bundle adjustment"]
    I --> J
    G --> K["英语 atomic clips\nHITL 边界/标签修正"]
    J --> L["world/camera hand + trajectory\nvalidity/missing mask"]
    K --> M["发布数据产品"]
    L --> M
    M --> N["可视化 / retarget / converter / train"]
~~~

### 5.1 逐阶段证据与不确定性

| 阶段 | 输入 → 输出 | 自动/人工 | 失败类型 | 置信度是否保留 | 可复现性 |
| --- | --- | --- | --- | --- | --- |
| 容器/流完整性 | 视频文件 → 可解码序列 | 自动 | header、旋转、stream、损坏、极端曝光 | 主要是 pass/fail | 部分；阈值未全给 |
| 内容筛选 | 可解码序列 → ego manipulation Parts | 自动 + 复查 | 无手、非第一视角、jitter、持续遮挡/截断 | 手有 `pred_valid` | 中 |
| 隐私处理 | 原文件 → 匿名数据 | 自动 + 规则 | 人脸/屏幕/姓名/联系人漏检 | 未见逐项隐私置信度 | 低到中 |
| 语义标注 | Part → atomic intervals | Qwen3.7-Plus + HITL | 边界漂移、对象幻觉、动词过粗/过细 | JSON 有 action confidence | 中；模型服务/完整 prompt 未给 |
| 相机重建 | RGB → $T^{W}_{C_t}$ | DROID-W + BA | 弱纹理、纯旋转、动态前景、尺度漂移 | 没有逐帧轨迹协方差 | 中低 |
| 双手重建 | RGB + camera → MANO/world hand | detector + HaWoR + temporal fill | 遮挡、hand swap、形状偏置、补帧过平滑 | `pred_valid` 二值；无 pose covariance | 中 |
| QC | 完整产品 → accepted/rejected | 自动 + 随机人工 | IK 失败、轨迹不平滑、模态缺失 | 多为聚合通过率 | 中低 |
| 发布 | sample → hub | 自动上传/版本化 | 文件错配、分辨率/内参不一致 | release notes 记录纠错 | 中高 |

### 5.2 信息损失链

1. **端侧漏检不可逆**：未录到的交互不会出现在离线管线。
2. **切 Part 丢长程上下文**：等待、失败、恢复、跨房间移动和并行动作可能被切断。
3. **VLM 把多模态策略压成文本**：同一语义的不同力、速度、抓取拓扑容易共享同一标签。
4. **SLAM 与手重建耦合**：world-frame 手轨迹同时含 camera pose 与 hand pose 误差。
5. **temporal fill 可“看起来平滑”**：遮挡区间可能是模型先验，不是观测。
6. **28-DoF IK QC 带本体偏置**：对某个手模型不可 retarget 的有效人类动作可能被当成低质量。
7. **匿名化可逆映射**：作者称映射完整保存、经授权可逆；发布匿名不等于内部不存在 re-identification 风险。

## 6. 数据分布、覆盖与长尾

![Open-AoE 随机 100 小时中的动作、场景、贡献者与相机分布](https://arxiv.org/html/2607.14183v2/fig-data-distribution.png)

论文的细粒度分布统计来自**随机 100 小时**，不能直接外推到全部 2000 小时。该子集包含 46,802 个 atomic intervals、193 个原始 verb strings、7346 个 object phrases 和 522 条 raw context descriptions；另一处结构化去重/归并后报告 8030 个 objects、175 个 verbs、135 个 scene labels。两组数字对应不同处理口径，不应互相替代。

### 6.1 场景与行为分布

| 大类 | 占比 | 解释与偏差 |
| --- | ---: | --- |
| kitchen | 24.8% | 容器、食材、工具、擦拭等丰富，但家庭厨房域强 |
| tabletop / indoor | 23.3% | 对桌面操作友好，也会低估移动操作与低位任务 |
| office / desk | 17.8% | 纸张、电脑、文具；屏幕与隐私风险较高 |
| bedroom | 9.1% | 衣物/收纳可能丰富；家庭敏感背景更多 |
| 其余 | 24.9% | living room、workshop、bathroom、outdoor、laundry、retail 等 |

这些场景提供了比实验室桌面更宽的背景与物体共现，但没有表明每个任务跨多少场景、物体实例和操作者。scene diversity 也可能成为 shortcut：模型可从厨房背景猜“切/倒”，而不是理解语言、手势或物体状态。

动作词和对象短语很多不等于均匀覆盖。自然语言中的同义词、复数、属性和对象实例会膨胀计数；长尾动作可能只有一次。论文没有给每个 verb/object 的最小样本量、Zipf 曲线、跨人重复数，也没有把“8000+ tasks”映射到受控技能本体。

### 6.2 贡献者分布

- 随机 100 h 中匿名 ID 的中位贡献时长仅 **2.6 分钟**。
- top-10 contributors 合计占 **13.7%**，至少说明样本并未完全被极少数人垄断。
- 但匿名 ID 是否一人一号、是否可能跨设备/批次变化未说明。
- 没有人口属性，因此不能评估手形、肤色、年龄、身高、运动能力、左利手或文化操作习惯的公平覆盖。
- 中位 2.6 分钟可能意味着广覆盖，也可能意味着大量一次性短贡献；需要贡献者级有效动作数和返工率才能判断。

### 6.3 设备与相机分布

随机 100 h 覆盖 400 多个市场手机型号/类型。三个占比最高的 5° FOV bins 分别约为 30.7%、30.2% 和 31.7%，主要集中在 65–75° 与 90–95°。这展示了真实 ISP、焦距、曝光、畸变和压缩差异，但论文自己也把“camera diversity 是否真正提升下游鲁棒性”列为待验证训练假设。

应特别区分：

- “400+ smartphones/models/types”更像**型号/设备类型覆盖**，不能从文字确定有 400 台还是 400 个唯一型号；
- 设备多样性会提高视觉域覆盖，也会提高 intrinsics、rolling shutter、帧率和压缩不一致；
- 发布卡说多数为 1280×720、部分 1920×1080；技术报告早期表述与当前数据卡并非完全一致，应以每个 sample 的 `video_info.json` 为准；
- 2026-07-30 删除内参与分辨率不匹配样本，说明跨设备自动校验仍是活跃风险。

### 6.4 缺失的行为类型统计

论文没有分别统计：

- 成功、失败、恢复、暂停、探索与危险行为；
- 单手、双手、交替手、工具延伸和身体支撑；
- 抓取、推、拉、旋拧、插拔、擦拭、切割、柔性物、液体、按钮、掌面接触；
- 需要步行、蹲起、平衡或全身协调的 mobile manipulation；
- 接触前 approach、接触建立、稳定操作、release 的 phase；
- 同一 task 的多策略、多速度与失败后修正。

因此，规模增长是否来自真实的新行为，而不是同类厨房/桌面片段重复，尚不能由论文数字回答。

## 7. 人体、相机与场景重建

![Open-AoE 从相机/双手重建到不同机器人本体重定向的工具链](https://arxiv.org/html/2607.14183v2/fig3-reconstruct-retarget.png)

### 7.1 坐标系与尺度

发布数据使用 OpenCV camera frame：$x$ 向右、$y$ 向下、$z$ 向前。world frame 由 visual SLAM 建立，并以第一帧相机位姿为参考。每帧 camera-to-world 齐次变换为

$$
\mathbf T^{W}_{C_t}=
\begin{bmatrix}
\mathbf R^{W}_{C_t} & \mathbf t^{W}_{C_t}\\
\mathbf 0^\top & 1
\end{bmatrix},
$$

相机点到世界点为

$$
\tilde{\mathbf p}^{W}_{t}
=
\mathbf T^{W}_{C_t}\tilde{\mathbf p}^{C_t}_{t},
\qquad
\tilde{\mathbf p}=[x,y,z,1]^\top.
$$

世界点投影到图像时先取逆变换：

$$
\mathbf p^{C_t}=\left(\mathbf T^{W}_{C_t}\right)^{-1}\tilde{\mathbf p}^{W},
\qquad
u=f_x\frac{x_c}{z_c}+c_x,
\qquad
v=f_y\frac{y_c}{z_c}+c_y.
$$

数据卡称轨迹为 metric-scale，但论文没有说明每段单目尺度的独立外部真值来源、尺度误差分布或跨 Part 的全局一致性。world frame 只在片段内有意义；不同 clip 的绝对位置和朝向不能直接拼接。

### 7.2 相机重建

作者对 DROID-W 重新调参，结合动态手/物体 mask、滑动窗口和全局 bundle adjustment，提高动态第一视角场景中的时序一致性。发布的 `camera_traj.npz` 只含：

- `cam_c2w`: $(T,4,4)$，float64；
- `intrinsic`: $(3,3)$，float64；
- 当前版本**不含逐帧深度图**。

这意味着世界系手轨迹可直接使用，但场景/对象表面几何需要另行重建。轨迹没有逐帧 confidence/covariance，也没有 loop-closure 状态；下游最好自行计算重投影、速度/加速度和重定位异常，而不是只检查文件存在。

### 7.3 双手重建

管线用在 AoE 上训练的双手 detector 找手，缺失/遮挡时做时序补全，再用 HaWoR/MANO 恢复 metric-aligned 手。每只手每帧的核心输出是：

| 字段 | 形状 | 坐标/单位 | 含义 |
| --- | --- | --- | --- |
| `pred_trans` | $(2,T,3)$ | world，m | 左/右 wrist translation |
| `pred_rot` | $(2,T,3)$ | world，rad | wrist global axis-angle |
| `pred_trans_cam` | $(2,T,3)$ | current camera，m | camera-frame wrist translation |
| `pred_rot_cam` | $(2,T,3)$ | current camera，rad | camera-frame wrist axis-angle |
| `pred_hand_pose` | $(2,T,45)$ | rad | 15 个 finger joints 的 axis-angle |
| `pred_betas` | $(2,T,10)$ | 无量纲 | MANO shape coefficients |
| `pred_valid` | $(2,T)$ | float32 0/1 | 二值有效掩码，必须转 bool |
| `focal` | scalar | px | 去畸变后的 $f_x$ 近似；精确投影优先用完整 intrinsic |

第一维固定 `[0]=left`、`[1]=right`。无效帧中的其他数组值不可靠，不能用零填充后直接训练。MANO 参数提供表面/骨架的可计算表示，但指尖或 21 joints 是前向运动学派生，不是独立观测。

### 7.4 遮挡与时序完成的双刃剑

时序完成提高了“模态存在率”，也会降低观测与预测的可区分性。理想发布应给：observed / interpolated / extrapolated / failed 四态 mask；当前主要只有二值 `pred_valid`。如果补帧后又被标为 valid，策略可能学到过平滑的手轨迹；如果全部 invalid，则有效覆盖会下降但更诚实。论文没有完整解释这一区分。

### 7.5 Object、scene 与 contact 的明确缺口

Open-AoE 发布中没有统一提供：

- object instance ID、6D pose、mesh、scale 与 articulation；
- object state change（门角、液位、布料拓扑、按钮状态）；
- contact point/patch、normal、penetration、force、torque 或 tactile；
- body/head/floating-base/foot pose；
- scene mesh、depth sequence 或可碰撞几何。

动作 bbox 和 object phrase 只能说明“语义上/图像上可能与哪个对象交互”；手与对象在像素中重叠也不是接触真值。没有 object motion 时，无法区分“手靠近杯子”和“手实际施力使杯子移动”。

## 8. 数据格式与训练接口

### 8.1 单个 sample 的目录结构

```text
<sample>/
├── raw_video.mp4
├── video_info.json
├── ego_annotation/
│   └── ego_action_annotation.json
└── ego_process/
    ├── ego_hands_reconstruction/
    │   ├── hands.npz
    │   ├── camera_traj.npz
    │   └── visualization/
    │       ├── hands_combined.mp4
    │       └── overview.png
    └── ego_undistorted_video/
        ├── raw_video_undistorted.mp4
        └── undistorted_video_info.json
```

`ego_action_annotation.json` 为 segment 数组，包含 `start_ts/end_ts/start_frame/end_frame/scene` 和 `atomic_action`；动作项包含 `verb/object/hand/description/bbox/confidence`。时间戳与 frame ID 是**字符串**，加载时必须显式转数值。bbox 在 raw video 像素系，若使用去畸变视频需要注意几何映射，而不能只按同一像素直接覆盖。

数据卡声称中英双语标签，但当前主 schema 和技术报告明确展示的是英语 `scene/verb/object/description`。在未逐文件核实另一语言字段前，不应把“bilingual”当作稳定 schema 保证。

### 8.2 论文表 2：五类动作表示

| 表示 | 维度 | 每只手/相机内容 | 坐标系 | 适用/局限 |
| --- | ---: | --- | --- | --- |
| Dense MANO | 110D | $[valid_1,wrist\ xyz_3,wrist\ aa_3,velocity_3,MANO_{45}]\times2$ | current camera | 保留手指姿态最完整；camera frame 速度混入 ego-motion |
| Wrist–fingertip | 48D | 每手 wrist xyz 3 + rot6D 6 + 五指尖 xyz 15 | SLAM world | 适合跨手型关键点；缺力和对象坐标 |
| Sharpa | 62D | 左/右 wrist EEF 9 + 左/右各 22 joints | world | 面向 Sharpa 等多指手；拓扑映射依赖 recipe |
| GR00T gripper | 20D | 左/右 wrist EEF 9 + gripper scalar 1 | 依配方 | 把多指压成开合，信息损失大 |
| Shared ego state/action | state 22D；action 20D/26D | 两手 $xyz_3+rot6D_6+grip_1$；state 加 2 validity；action 可加 camera $\Delta t_3+\Delta rot_{aa,3}$ | current camera / relative camera | 便于 LeRobot/VLA/world model；finger 只是 open/close proxy |

110D dense MANO 可写为

$$
\mathbf a^{\mathrm{MANO}}_t
=
\bigoplus_{h\in\{L,R\}}
\left[
v^h_t,
\mathbf p^h_t,
\boldsymbol\theta^h_t,
\dot{\mathbf p}^h_t,
\boldsymbol\phi^h_t
\right]
\in\mathbb R^{110},
$$

其中 $v$ 为有效性、$\mathbf p$ 为 wrist translation、$\boldsymbol\theta$ 为 wrist axis-angle、$\dot{\mathbf p}$ 为速度、$\boldsymbol\phi\in\mathbb R^{45}$ 为 MANO finger pose。

共享 20D hand action 则是

$$
\mathbf a^{\mathrm{hand}}_t
=
\bigoplus_{h\in\{L,R\}}
\left[
\mathbf p^h_t,
\mathbf r^h_{6D,t},
g^h_t
\right]
\in\mathbb R^{20},
$$

加入相机自运动后为

$$
\mathbf a^{26D}_t
=
\left[
\mathbf a^{\mathrm{hand}}_t,
\Delta\mathbf t^{C}_{t\rightarrow t+1},
\Delta\boldsymbol\theta^{C}_{t\rightarrow t+1}
\right].
$$

仓库的 `ACTION_SPEC.md` 明确指出 20D/26D 是一套**下游共享接口**，不是论文所有信息的统一无损格式：手指只用 open/close proxy，尚未从 MANO 做完整 FK。不同 recipe 仍可能用 48D、62D、110D 或模型原生表示。

### 8.3 相机动作的语义陷阱

在第一视角中，current-camera frame 下的 hand delta 同时受手运动和相机 ego-motion 影响。26D 显式加入相机 6D 有助于 world model 解释像素变化，但也可能形成 shortcut：模型仅预测 camera motion 就获得较高视频损失改善，却未学会对象交互。评估必须分别 shuffle/zero hand 与 camera channels，并报告对象区域或接触阶段的效果。

## 9. 数据质量与不确定性

| 模态 | 真值来源 | 论文/数据卡指标 | 缺失/覆盖 | 置信度 | 对训练的影响 |
| --- | --- | ---: | ---: | --- | --- |
| RGB | 手机传感器 | 解码/曝光/完整性 QC；无统一感知质量分 | 已发布 sample 基本存在 | 无逐帧质量分 | 模糊/ISP 域提高多样性也增加噪声 |
| Intrinsics | 手机/离线校验 | 2026-07-30 删除错配样本 | 当前 sample 提供 | 无 calibration covariance | 投影错会同时污染手、bbox、轨迹 |
| Camera trajectory | DROID-W/SLAM | 数据 README 另称 ATE 4.4 mm、ATE-S 14.1 mm、RPE trans 1 mm、RPE rot 5.40° | sequence-level presence 100% | 无逐帧置信度 | world trajectory 漂移会被当作手运动 |
| MANO hand | detector + HaWoR + fill | README 另称 PA-MPJPE 10.5 mm、WA-MPJPE 11.9 mm、AUC 0.9900 | any-hand valid 98.93%；both 98.02% | `pred_valid` 二值 | 遮挡/手型偏差会系统污染 retargeting |
| Action bbox | VLM | 100% valid；98.39% within image；mean conf 0.947；10th percentile 0.950 | sequence-level 100% | scalar confidence | 10th percentile 高于 mean 的汇总值得复核，可能由长尾低值造成 |
| Atomic language | Qwen3.7-Plus + HITL | Idefics2 macro 4.583/5 | 99.99% 时间覆盖 | per-action confidence | evaluator 偏差、标签幻觉、边界漂移 |
| Object state | 未提供 | — | 缺失 | — | 不能监督状态变化与物理结果 |
| Contact/force | 未提供 | — | 缺失 | — | 不能训练 System0、阻抗或抓取稳定性 |

手/相机的 mm 级数字来自**当前数据 README**，技术报告主文没有给出完整 benchmark 设计、测试集、真值传感器、置信区间和失败分布。因此应视为官方数据卡断言，不能与经同行评审的论文主结果同级引用。

### 9.1 质量门控本身可能产生的偏差

- 以 28-DoF retarget IK failure 衡量 correctness，会偏向特定手型和可达空间；非常规但真实的人类抓取可能被过滤。
- smooth camera trajectory 会奖励过度平滑，急停、快速转头或有意相机运动可能被错删。
- completeness 只统计 valid frame，不衡量 3D pose 是否正确。
- modality presence 100% 只说明文件/字段存在，不说明每帧可靠。
- 随机人工抽检没有报告抽样比例、标注协议、双人一致性和错误召回率。
- VLM 置信度不是校准概率；0.95 不意味着 95% 标签正确。

### 9.2 建议的下游不确定性表示

把二值 valid 升级为多源 uncertainty vector：

$$
\mathbf u_t=
\left[
u^{\mathrm{det}}_t,
u^{\mathrm{occ}}_t,
u^{\mathrm{SLAM}}_t,
u^{\mathrm{reproj}}_t,
u^{\mathrm{VLM}}_t,
u^{\mathrm{interp}}_t
\right],
$$

训练时采用可靠性权重

$$
\mathcal L
=
\frac{\sum_t w(\mathbf u_t)\,\ell_t}
{\sum_t w(\mathbf u_t)+\epsilon},
$$

并把 observed、filled、invalid 分开。该公式是推荐实现，不是论文原始 loss。

## 10. 语义—人体运动—机器人动作—接触的联合对齐

| 层 | Human 表示 | Robot 表示 | Contact 表示 | 对齐机制 | 监督来源 | 验证 |
| --- | --- | --- | --- | --- | --- | --- |
| 任务语义 | scene/description | language condition/task ID | 无 | 同一 clip/episode 文本 | VLM + HITL | Idefics2 图文一致性，不是真机任务泛化 |
| 子目标 | atomic interval | action chunk/episode segment | 隐式 phase | temporal boundary | VLM | 覆盖/密度；无人工 boundary F1 |
| 运动意图 | wrist/fingertip/MANO | EEF/joint/gripper/latent action | 隐式 | converter、IK、latent-action model | 自动手重建 | repository POC；论文无任务 SR |
| 末端几何 | wrist pose、fingertips | robot TCP/palm pose | 无接触面 | palm-to-TCP、IK | world/camera hand | 可视化/重定向演示 |
| 全身运动 | 未提供 | humanoid base/arm/legs | 未提供 | 无 | — | 无 |
| Contact intent | action verb + bbox + hand proximity | 可作为 high-level constraint | 无真实 point/normal/force | 需下游另推断 | 弱视觉共现 | 未验证 |
| Grasping goal | MANO hand shape + object phrase | gripper scalar/robot fingers | 无稳定性/wrench | dex-retarget/开合 proxy | 自动伪标签 | 无抓取成功率 |
| 力与柔顺 | 未提供 | torque/impedance target | 未提供 | 无 | — | 无 |

### 10.1 核心判断

1. **语义主要是 condition，不是 contact controller**：verb/object/description 没有显式决定接触点、抓取面或力方向。
2. **Human motion 更适合作行为先验**：MANO 是重建轨迹，不宜当作每帧必须精确跟踪的控制命令。
3. **Contact intent 只能弱推断**：可从 `grasp/twist/press`、bbox、hand shape 和物体运动另建模型，但 Open-AoE 本体没有 observed/intended contact 区分。
4. **Grasp goal 映射是有损的**：五指 MANO 压成单 scalar gripper 时只保留粗开合；映射到 dexterous hand 时又受拓扑/关节限位影响。
5. **多模态策略未显式保留**：自然视频包含多种抓法，但 VLM 类别与确定性 converter 可能把它们压到同一标签/轨迹。
6. **对齐错误缺少闭环修正**：HITL 改语义，QC 删几何，但没有基于 robot rollout 失败回写哪一帧 hand/contact 错。
7. **scene-to-action shortcut 风险高**：场景类别强不均衡，模型可能以厨房/桌面背景预测动作，而非真正理解物体状态。

### 10.2 Video → Human State → Intent → Robot Behavior

~~~mermaid
flowchart TD
    V["Ego RGB + intrinsics"] --> CAM["SLAM camera trajectory\n误差：尺度/漂移"]
    V --> HAND["MANO / wrist / fingertips\n误差：遮挡/补帧"]
    V --> LANG["atomic verb/object/description\n误差：边界/幻觉"]
    CAM --> HS["world/camera human hand state"]
    HAND --> HS
    LANG --> INT["task/subgoal/contact intent\n当前主要是弱推断"]
    HS --> INT
    INT --> REP["canonical EEF / latent / 20D-110D action\n信息损失：手指/对象/力"]
    REP --> RET["IK / dex-retarget / robotized video"]
    RET --> POL["VLA / policy / world model"]
    POL --> CTRL["embodiment-specific controller\nOpen-AoE 未提供"]
    CTRL --> REAL["real robot contact\nOpen-AoE 未做论文级验证"]
    REAL -. "失败应回流，但当前无统一闭环" .-> INT
~~~

## 11. Embodiment Gap：从人手到机器人本体

### 11.1 Gap 分解

| 子 gap | Open-AoE 提供的桥 | 未解决部分 |
| --- | --- | --- |
| Morphology | MANO、21 joints、wrist/fingertip canonical points | 人手骨长、掌形与 robot linkage 不同；shape uncertainty 未传递 |
| DoF/topology | 45D MANO 或 keypoints → robot joint/EEF | 关节轴、耦合、腱驱动、欠驱动与闭链差异 |
| Workspace/reachability | world/camera wrist pose、palm-to-TCP | 没有目标 robot base/shoulder 可达性真值 |
| Floating base/locomotion | camera trajectory 可作为粗 ego-motion cue | 无 body/head/base/foot pose，不能恢复 balance 与步态 |
| Hand/gripper | dex-retarget、Sharpa/G1 Inspire/XHand 等 recipe | grasp topology、normal、force、slip 未提供；gripper proxy 丢多指信息 |
| Observation | robotized video、ego RGB | robot 相机外参、遮挡、深度/触觉域仍不同 |
| Action space | 20/26/48/62/110D 多表示 | 没有统一的 torque/joint command；各 recipe 语义不同 |
| Contact capability | 语义与 MANO 形状作为弱先验 | 人手柔软表面与机器人手摩擦/顺应性完全不同 |

### 11.2 重建—重定向矩阵

官方 `aoe-reconstruct-retarget` 将工具链组织为两个上游重建轨迹、两类手来源和三类 retarget 路线的 12-cell 组合：

| 轨迹/人手来源 | 目标/方法 | 主要映射 | 开放边界 |
| --- | --- | --- | --- |
| EgoInfinity | Unitree G1 + Dex3 / Inspire | arm IK + dex-retarget | EgoInfinity checkout、模型权重和生成结果不随仓库完整提供 |
| Do-as-I-Do 4D HOI | Sharpa 等 | hand/object-aware trajectory adapter | 上游依赖外置 |
| Phantom hand source | G1 Dex3 / Inspire | arm IK + dex-retarget | Phantom 资产/权重外置 |
| Open-AoE MANO/palm | Galbot / Galaxea | palm-to-TCP + gripper mapping | 主要是末端功能映射，不含真实接触约束 |
| Retarget Lab | EgoInfinity/G1、Do-as-I-Do/Sharpa | 统一实验接口 | 结果依赖外部环境和机器人资产 |
| SPIDER / XHand | 12-cell matrix 中的 dex route | keypoint/joint retarget | 上游代码和权重许可另行处理 |

仓库还支持：手/臂 segmentation → E2FGVI/ProPainter 去除人体 → MuJoCo 渲染机器人，生成 robotized video。它能缩小视觉 embodiment gap，却不会凭空生成真实 robot action、object dynamics 或 contact force；错误的 3D 轨迹被渲染得很逼真，反而可能更难察觉。

### 11.3 Motion retargeting、functional retargeting 还是 intent transfer？

- **Dense MANO → robot joints**：主要是 motion retargeting，追求姿态/关键点相似。
- **palm-to-TCP + gripper**：接近 functional retargeting，保留末端位姿和粗开合，不复制人手全部关节。
- **language/object + latent action/world model**：可能成为 intent transfer，但论文没有闭环机器人验证。
- **没有 object goal/contact/wrench**：所以 Open-AoE 还不能稳定做到“保留任务功能、允许动作几何改变”的完整 functional transfer。

一个更合理的 retarget objective 应显式区分任务功能与姿态相似：

$$
\min_{\mathbf q_{1:T}}
\lambda_e\mathcal L_{\mathrm{EEF}}
+\lambda_f\mathcal L_{\mathrm{finger}}
+\lambda_o\mathcal L_{\mathrm{object}}
+\lambda_c\mathcal L_{\mathrm{contact}}
+\lambda_s\mathcal L_{\mathrm{smooth}}
+\lambda_j\mathcal L_{\mathrm{limit}},
$$

但 Open-AoE 直接提供的主要是 $\mathcal L_{\mathrm{EEF}}$、$\mathcal L_{\mathrm{finger}}$ 和时序项所需信号；$\mathcal L_{\mathrm{object}}$ 与 $\mathcal L_{\mathrm{contact}}$ 需要额外标注/重建。

### 11.4 多本体 ≠ 多本体成功证明

工具链覆盖多个 robot/hand 是接口广度证据，不是任务可执行率证据。要证明 embodiment gap 真正缩小，至少需要同一批 human clips 在多个本体上报告：

- IK/optimization 成功率；
- wrist/fingertip/object trajectory error；
- joint-limit/self-collision/environment-collision rate；
- grasp stability/contact preservation；
- zero-shot 与少量 robot adaptation 后真机 SR；
- 不同手型/DoF 的失败分类。

论文没有这组矩阵。因此结论只能是“**可转换**”，不是“**已证明可执行**”。

## 12. Task Gap：语义覆盖不等于 skill composition

### 12.1 数据提供的 task ingredients

Open-AoE 为 task learning 提供：

- 密集 atomic boundaries；
- verb、object、hand、scene 和自然语言 description；
- 同步 RGB、camera、wrist/finger motion；
- 多场景、多对象、多贡献者的自然变化；
- 可切成固定窗口的长序列。

这些信号适合学习 latent action、video dynamics、future state、动作检索或 VLA pretraining。但原子边界不是 option termination 的真值，语言也不保证对应唯一 skill。

### 12.2 仓库中的 skill/world-model 路线

![Open-AoE training-ready：VLA、latent-action model 与 world model 配方](https://arxiv.org/html/2607.14183v2/fig4-training-ready.png)

| 路线 | 仓库适配模型 | Open-AoE 提供的输入 | 能回答的问题 | 不能自动回答的问题 |
| --- | --- | --- | --- | --- |
| VLA/policy | ACT、Diffusion Policy、π0.5、SmolVLA、GR00T N1.7、H-RDT、VITRA | RGB + language + 20/26D 或模型原生 action | hand/camera motion prediction、视觉语言预训练 | robot torque、真实接触、真机 SR |
| Action-conditioned world model | iVideoGPT、DreamZero、LingBot-VA、Ctrl-World | frame + 20/26D action | 给定动作的视频变化/可控性 | 物体 state、力学真实性 |
| Latent-action model | GenieRedux、LAOM、AdaWorld | frame pairs/clips，部分有 20/26D supervision | 从视频发现动作 latent | latent 是否对应可执行 robot skill |
| Video dynamics | DreamDojo、iVideoGPT 等 | ego video + latent/MANO/camera action | future visual prediction | long-horizon task correctness与闭环恢复 |

### 12.3 Latent skill manifold 的十个问题

1. **如何学**：仓库提供 VQ/LAM/VAE/world-model 多种路线，不是论文提出单一 manifold。
2. **时间尺度**：atomic clips 与固定窗口提供候选边界，但没有层级 skill duration 真值。
3. **输入**：视频、语言、MANO、camera action 可进入 latent；object state 缺失。
4. **组合**：没有演示把未见多个 atomic skills 组合成新长任务。
5. **Structured deviation**：26D 可把 hand 与 camera 分通道，但没有“向左一点/换抓法”等可解释 deviation interface。
6. **在线修正**：仓库 recipe 主要是离线训练；无统一 language coaching/human takeover 闭环。
7. **连续性**：固定 clip/window 可重叠，不能保证跨边界 contact continuity。
8. **插值风险**：latent/video prediction 平滑不代表对象达到正确目标状态。
9. **覆盖指标**：CLIP rank/cluster 衡量视觉支持，不是 robot skill manifold coverage。
10. **泛化类型**：当前主要支持 task representation 与 trajectory prior；compositional generalization 未证实。

应严格区分：

- **task generalization**：已见技能在新对象/场景；论文未做 robot test。
- **compositional generalization**：重组未见技能序列；未验证。
- **trajectory retrieval**：从相似视频找相似手轨迹；工具链可做，但不等于前两者。

### 12.4 Correctable latent 的建议接口

对 Open-AoE 更合适的 latent 不应只预测视频像素，而应拆成

$$
\mathbf z_t=
\left[
\mathbf z^{\mathrm{semantic}}_t,
\mathbf z^{\mathrm{EEF}}_t,
\mathbf z^{\mathrm{object}}_t,
\mathbf z^{\mathrm{contact}}_t,
\mathbf z^{\mathrm{style}}_t
\right],
$$

允许人或上层模型只修正对象目标/contact intent，而保留可行的运动风格。Open-AoE 当前可监督前两项和部分 style；对象/contact 两项必须补标。

## 13. Reality Gap：真实视频不是机器人真实动力学

Open-AoE 的视频来自真实世界，所以它避免了纯仿真在纹理、光照、遮挡和人类行为上的部分差异。但“视觉是真实的”不能推出“机器人动力学已经对齐”。

| 差异 | 论文/工具链建模 | 数据来源 | 是否在线 | 验证指标 | 剩余风险 |
| --- | --- | --- | --- | --- | --- |
| 视觉外观 | 真实手机 RGB、robotized video | human video + render/inpainting | 离线 | CLIP/图文/视频指标 | robot 相机/自遮挡仍不同 |
| Kinematics | IK、dex-retarget、palm-to-TCP | MANO/wrist trajectory | 离线 | 无统一跨本体 SR | joint limit/self-collision/contact 不完整 |
| Dynamics | 未提供 joint-wise neural dynamics/HumanoidDM | — | — | — | inertia、coupling、impact 未建模 |
| 摩擦/顺应 | 取决于外部 MuJoCo/robot config | 外部资产 | 仿真/控制侧 | 无 | 人手皮肤与机器人材料差异大 |
| Backlash/latency/saturation | 未建模 | — | — | — | 真机高速/精细操作失败 |
| Payload/temperature/老化 | 未建模 | — | — | — | 无跨个体/负载验证 |
| System identification | 未提供 | — | — | — | sim-to-real 参数仍未知 |
| Real post-training | 某些上游模型可接，但 Open-AoE 论文未做 | 需另采 robot data | 可选 | 无论文指标 | 需要昂贵 robot bridge data |

论文没有 joint-wise neural dynamics model、HumanoidDM、actuator model、domain randomization 或 system identification。Reality gap 必须由目标机器人团队在低层另补：

1. 对每个关节/驱动器辨识延迟、摩擦、饱和和温升；
2. 对全身耦合与接触保留统一动力学模型，不能只按关节独立拟合；
3. 在仿真中做 contact/material/randomization；
4. 用真实 robot rollouts 做 actuator-aware post-training；
5. 由 System0 吸收高频碰撞、滑移和力控误差。

Open-AoE 可给高层 motion/contact-intent 候选，不能替代这条链。

## 14. Contact Intent 与 Grasping Goal

### 14.1 Open-AoE 实际能提供什么

| 问题 | 可用信号 | 证据等级 |
| --- | --- | --- |
| 接触哪个对象 | object phrase + action bbox + hand proximity | 弱监督 |
| 何时开始/结束操作 | atomic interval boundary | VLM 伪标签 |
| 哪只手 | `hand=left/right/both/none` + MANO valid | 中等 |
| 采用何种手形 | 45D MANO + shape | 中等，但遮挡敏感 |
| 接触点/区域 | 可从 hand mesh 与另建 object mesh 几何推断 | 数据本体未提供 |
| 接触法向/力方向 | 动词与运动可给先验 | 未测量 |
| 力、扭矩、柔顺 | 无 | 缺失 |
| 抓取稳定性/滑移 | 无 tactile/object pose | 缺失 |

### 14.2 observed contact 与 intended contact

视频中手在物体附近是 observed evidence；语言中的 `grasp/twist/press` 是 intended contact 的弱提示。两者可能冲突：失败抓取有“grasp”意图却没有稳定接触，遮挡下稳定接触又可能不可见。Open-AoE 没有显式二分，也没有失败标签，因此下游模型必须避免把语义当成事实接触。

### 14.3 Grasp goal 的三层表达

更安全的抓取目标应分成：

1. **对象功能目标**：抓住杯把、按下按钮、旋开盖子；
2. **接触约束**：目标区域、法向、允许滑动、需要的 wrench cone；
3. **本体实现**：人 MANO、机器人指关节、夹爪开合或掌面接触。

Open-AoE 主要提供第 3 层的人手候选和第 1 层的语言弱标签，缺第 2 层。直接把 MANO pose 当作 robot grasp goal 会在不可达、物体尺度变化和手型不同处失败；更合理的是先恢复对象功能/接触区域，再由 robot-specific grasp planner 求解。

### 14.4 灵巧与全身接触边界

- 多指相对姿态可从 MANO 获益，但没有 fingertip pressure、掌面接触或物体表面法向。
- 柔性物、液体和工具使用需要 object state/dynamics，数据本体未给。
- whole-body contact（肩、肘、躯干、脚）不可见/未重建。
- 无力/触觉意味着数据适合 System1 的动作候选，不适合直接监督 System0 的力控。

## 15. Human–Robot Co-training 与 latent alignment

Open-AoE 论文本身没有 human/robot co-training 实验。它提供的是让这类实验可开展的接口：human video、language、hand/camera action、robotized video 与多种 converter。下列问题在论文中均未闭环：

| 问题 | 论文回答 |
| --- | --- |
| Human 与 robot 数据是否直接混合？ | 未做统一策略实验 |
| 是否共享 encoder/latent/policy head？ | 取决于 ACT/SmolVLA/GR00T/H-RDT 等外部 recipe |
| 是否显式诱导 latent overlap？ | LAM/world-model 路线可做，但论文无统一 objective |
| 是否有 EgoBridge 式先对齐再共训？ | 无专门对照 |
| 高质量 teleop 是否作为桥？ | 需要用户另备；Open-AoE 不提供统一 robot bridge set |
| 数据比例/采样权重/embodiment token？ | 未报告 |
| negative transfer？ | 未做 scaling/ratio sweep |
| joint vs sequential vs robot-only？ | 未比较 |
| Human 数据用于 pretrain 还是 post-train？ | 两者均可由外部 recipe 实现，论文不作因果结论 |
| 增益来自视觉、任务还是动作？ | 无分解消融 |

### 15.1 推荐的共训实验矩阵

为判断 Open-AoE 的真实贡献，固定 robot dataset 与总更新数，至少比较：

| 组别 | Human RGB | Language | Hand/camera action | Robot data | 目的 |
| --- | --- | --- | --- | --- | --- |
| Robot-only | 否 | robot task text | robot action | 固定 | 基线 |
| Visual-only | 是 | shuffle/空 | 不用 | 固定 | 测视觉多样性 |
| Semantic-only | 是 | 正确 | 不用 | 固定 | 测语义/未来表征 |
| Motion-only | 是 | 空 | hand/camera | 固定 | 测动作先验 |
| Full human | 是 | 正确 | hand/camera | 固定 | 总收益 |
| Full + contact/object | 是 | 正确 | 完整 | 固定 + 补标 | 测缺失物理层的增量 |

并对 human:robot ratio、任务相似度、对象/场景 held-out 和不同 embodiment 分别画 scaling curve。只有这样才能回答提升来自更多图像 token、更多任务、相机 ego-motion 还是手动作。

### 15.2 Latent overlap 的安全目标

直接把人手与 robot action latent 拉近可能造成 negative transfer。更合理的是只对齐 task-functional 部分：

$$
\mathcal L_{\mathrm{align}}
=
d\!\left(
f_h(o^h_{1:T},l),
f_r(o^r_{1:T},l)
\right)
+\lambda\,\mathcal L_{\mathrm{domain\text{-}specific}},
$$

其中共享 latent 表达对象目标/接触意图，embodiment-specific head 保留人手与 robot 的不同可达性。Open-AoE 可给 $o^h,l$，但需要 paired/semantically matched robot data 才能监督 $f_r$。

## 16. 策略架构与快慢系统接口

| 层 | 输入 | 推荐输出 | 频率 | 训练数据 | Gap 责任 |
| --- | --- | --- | ---: | --- | --- |
| System2 | language、scene/object state、long video memory | task graph、subgoal、grasp/contact goal | 0.1–2 Hz | Open-AoE language/RGB + robot outcome | Task / semantic–contact |
| System1 | subgoal、RGB/pose、robot state | feasible EEF/latent action chunk + uncertainty | 5–30 Hz | Open-AoE hand/camera prior + robot teleop | Embodiment / Task |
| System0 | EEF/contact constraint、force/tactile、joint state | torque/position/impedance command | 100–1000 Hz | 真机 contact + sim dynamics | Reality / Contact |
| Drive | low-level command、motor sensing | actuator torque/current | kHz | 标定/system ID | Hardware |

Open-AoE 最自然地服务 System2/1：它可输出 task/subgoal、wrist/finger motion prior 与 future visual dynamics。慢系统给快系统的接口不应是逐帧 MANO 精确轨迹，而应是：

- object-centric target；
- EEF corridor/constraint；
- grasp/contact intent；
- 可修正 latent action chunk；
- uncertainty/missing mask。

System0 必须用 robot proprioception、force/tactile 和 actuator model 接管真实接触。若把 30 FPS ego action 直接映射到高频关节命令，camera jitter、补帧误差和人手不可达姿态会进入控制闭环。

## 17. 动作时序、chunk 与连续性

### 17.1 Atomic interval 不等于可执行 chunk

VLM boundary 按语义切分，而机器人 action chunk 还需满足：

- 起点/终点动力学连续；
- 接触建立/释放不被切断；
- 上一 chunk 的对象状态可观测；
- IK 分支和 grasp mode 不跳变；
- 相机运动与手运动正确解耦；
- 长程任务保留失败/恢复记忆。

如果每个 chunk 从白噪声独立 diffusion，而不条件于上一末状态，会在接触转换处产生 discontinuity。推荐以 overlap + state carry + contact phase token 生成。

### 17.2 频率与单位

- 发布视频为 30 FPS；不是 robot servo rate。
- wrist translation 以 m，axis-angle/joint 以 rad，速度以 m/s；混用前必须做单位/帧率检查。
- rot6D 适合神经网络连续回归，但转回 $SO(3)$ 时需正交化。
- invalid hand 要连同 action mask 输入；不能让左右手消失被解释为“保持零位”。
- world frame 适合去 ego-motion，current camera frame 适合视觉对应；两者应同时保留而不是早期丢一边。

### 17.3 兼容 locomotion 的限制

camera trajectory 可粗略反映佩戴者移动，但没有胸/头到 base、脚接触、质心和地面信息。用它驱动 humanoid floating base 会混淆身体摆动、颈挂晃动和真实行走。Open-AoE action 表示主要兼容 manipulation，不兼容完整 loco-manipulation。

## 18. 论文评测：数据属性是否真的更好

论文的评测对象不是 policy success，而是四组数据属性：视觉表征覆盖、语义与时间标注、VLM 图文一致性、训练窗口与手/模态可用性。所有对照应按这个证据范围解读。

### 18.1 CLIP 谱与局部多样性

![Open-AoE、OpenEgo、EgoDex 的六项 CLIP 多样性指标](https://arxiv.org/html/2607.14183v2/fig_clip_six_metrics.svg)

对数据集 $s$ 的 CLIP embedding covariance 特征值 $\lambda_{s,j}$，先归一化：

$$
p_{s,j}=\frac{\lambda_{s,j}}{\sum_r\lambda_{s,r}}.
$$

有效秩为

$$
R_{\mathrm{eff}}^{(s)}
=
\exp\!\left(-\sum_j p_{s,j}\log p_{s,j}\right),
$$

参与率为

$$
R_{\mathrm{PR}}^{(s)}
=
\frac{\left(\sum_j\lambda_{s,j}\right)^2}
{\sum_j\lambda_{s,j}^2}.
$$

直觉上，有效秩/参与率越高，embedding variance 分布在越多方向，而不是集中在少数视觉模式。Open-AoE 的均值为：

| 指标 | Open-AoE | 解释 |
| --- | ---: | --- |
| Effective rank | 97.43 | 三者最高，3000/3000 次重采样排名第一 |
| Participation ratio | 40.47 | 三者最高，3000/3000 次排名第一 |
| Meaningful coverage | 19.89 | 略高，优势小于谱指标 |
| Normalized entropy | 0.712 | 略高，不能夸成压倒性差距 |
| Effective clusters | 16.29 | 共享码本下使用较均匀 |
| kNN domain mixing | 0.0775 | 三者最高，3000/3000 次排名第一，但绝对值仍低 |

论文每次从每个数据集抽 2000 个 embedding，共做 3000 次 trial。共享 $K=50$ 聚类码本中，数据集 $s$ 在 cluster $c$ 的占比是

$$
q_{s,c}=\frac{n_{s,c}}{\sum_{u=1}^{K}n_{s,u}}.
$$

以 $\tau=5$ 定义 meaningful coverage：

$$
C_{s,\tau}
=
\sum_{c=1}^{K}\mathbb 1[n_{s,c}\ge \tau].
$$

归一化 entropy 与 effective clusters 可写为

$$
H_s^{\mathrm{norm}}
=
-\frac{\sum_c q_{s,c}\log q_{s,c}}{\log K},
\qquad
K_s^{\mathrm{eff}}
=
\exp\!\left(-\sum_c q_{s,c}\log q_{s,c}\right).
$$

kNN domain mixing 对样本 $i$ 的 $k=20$ 邻域统计同域比例：

$$
M_{\mathrm{kNN}}^{(s)}
=
1-
\frac{1}{N_s}
\sum_{i:y_i=s}
\frac{1}{k}
\sum_{j\in\mathcal N_k(i)}
\mathbb 1[y_j=s].
$$

越高表示局部邻域更常混入其他数据集样本，即不只占据孤立域。但 Open-AoE 的 0.0775 仍意味着大部分近邻是同域，不能说 domain gap 已消失。

### 18.2 CLIP 指标的证据边界

- CLIP embedding 对背景、对象、构图与场景敏感；高 rank 可能主要来自手机/场景差异，不一定是动作/接触差异。
- 共享 codebook 的结论依赖 backbone、frame sampling、$K=50$、距离度量和预处理。
- 每数据集等量 100 h 有利于比较，但原数据标签 schema、帧率和片段边界并不受控。
- effective rank 与 robot success 没有单调保证；噪声也能增加谱宽度。
- kNN mixing 是跨数据集视觉相似性，不是 human/robot latent overlap。

因此这些数字支持“Open-AoE 视觉支持较宽”，不支持“机器人泛化必然更好”。

### 18.3 语义与时间标注

![四个 Ego 数据集的语义规模、时间覆盖、片段密度与平均时长](https://arxiv.org/html/2607.14183v2/fig_temporal_semantic.svg)

| 数据集 | 语义标签口径 | 时间覆盖 | segments/min | 平均段长 |
| --- | --- | ---: | ---: | ---: |
| Open-AoE | 32,407 条 distinct natural-language descriptions；结构化 8030 objects / 175 verbs / 135 scenes | 99.99% | 13.97 | 9.64 s |
| OpenEgo | 26,864 native labels | 50.1% | 9.65 | 4.22 s |
| EgoDex | 111 categorical classes | 100% | 14.31 | 8.32 s |
| EgoXtreme | 异构/可用标签 | 依可用部分 | 见论文图 | 见论文图 |

这里最大的陷阱是**跨 schema 不可比**：自然语言 distinct string 数会被对象属性、同义词和句式放大，而 111 categorical classes 是受控类。Open-AoE 的优势是描述细、时间覆盖高，不代表它真的拥有 32,407 个独立技能。

平均段长也不是越长越好：9.64 s 可能保留更完整动作，也可能把多个微操作合并。没有人工 boundary precision/recall 与 inter-annotator agreement，不能仅用密度判断切分质量。

### 18.4 VLM 图文一致性

![Idefics2 对图像—动作描述一致性的序列宏平均评分](https://arxiv.org/html/2607.14183v2/fig_vlm_consistency.svg)

论文用 Idefics2 对每个时间点给 $a_{q,t}\in[1,5]$，先在 sequence 内平均，再跨 sequence 宏平均：

$$
S_{\mathrm{macro}}
=
\frac{1}{Q}
\sum_{q=1}^{Q}
\left(
\frac{1}{T_q}
\sum_{t=1}^{T_q}a_{q,t}
\right).
$$

| 数据集 | Sequence-macro score | 其他 |
| --- | ---: | --- |
| Open-AoE | **4.583/5**，95% CI $[4.557,4.608]$ | score ≥4 占 85.4%；≤2 占 2.2% |
| OpenEgo | 3.029 | — |
| EgoDex | 2.916 | — |
| EgoXtreme | 2.015 | — |

宏平均避免长 sequence 通过帧数主导总分，是合理设计。但 evaluator 不是人工真值：同族 VLM 可能偏好更完整、更自然的描述；Open-AoE 的自动描述比短 categorical label 更容易获得高分。它证明 evaluator-based semantic consistency，不证明动作类别、边界、对象身份的完整准确率。

### 18.5 训练窗口产率

![窗口产率、手部有效率、bbox 质量与模态完整性](https://arxiv.org/html/2607.14183v2/fig_training_utility.png)

对长度 $L_q$ 秒的 sequence，history $h=2$ s、future $f=2$ s、stride $\Delta=2$ s 时，候选窗口数为

$$
N_q
=
\max\!\left(
0,
\left\lfloor
\frac{L_q-h-f}{\Delta}
\right\rfloor+1
\right).
$$

每小时 window yield 与相对理论上限的 retention 为

$$
Y_{\mathrm{window}}
=
\frac{\sum_q N_q}{\sum_q L_q/3600},
\qquad
\eta_{\mathrm{window}}
=
\frac{Y_{\mathrm{window}}}{3600/\Delta}.
$$

在 $\Delta=2$ s 下理论上限是 1800 windows/hour。论文报告：

| 数据集 | windows/hour | retention |
| --- | ---: | ---: |
| Open-AoE | 1760 | 97.8% |
| OpenEgo | 1781 | 98.9% |
| EgoDex | 1226 | 68.1% |
| EgoXtreme | 1702 | 94.5% |

Open-AoE 的训练可切窗性很强，但这一指标主要受 sequence 长度与边界损失影响，不衡量每个窗口是否包含有效对象变化、清晰接触或正确 action。OpenEgo 在该指标甚至略高，说明它不是 Open-AoE 的独有优势。

Open-AoE 另有：

- any-hand valid frames：98.93%；
- both-hands valid frames：98.02%；
- OpenEgo 对应约 66.2% / 55.6%；
- Action、BBox、Confidence、Hand、Camera 五类 modality 在 sequence level 均为 100% presence；
- bbox 100% valid，98.39% within image，mean confidence 0.947，10th-percentile 0.950。

五模态 100% 是 schema 完整性，不是全部帧准确性。bbox 汇总中 10th percentile 高于 mean 看似反常，但在少量极低离群值拉低均值时数学上可能发生；仍应在复现中检查原始分布与统计实现。

## 19. 论文表 1：跨数据集资产比较

论文将 Open-AoE 与多种 Ego/robot-oriented 数据集放在同一表中，核心列可重建为：

| 数据集/属性 | Open-AoE 论文声称 |
| --- | --- |
| Hours | 约 2000 h |
| Tasks | 8000+（定义未受控） |
| Hand representation | MANO |
| Language | 有 |
| Camera trajectory | 有 |
| Contributors | 500+ |
| Device models/types | 400+ |
| Retargeting tools | 有 |
| Training-ready recipes | 有 |

这个表适合快速看覆盖，不适合直接排名，因为不同数据集的 “task” 可能是 categorical class、自然语言字符串、episode goal 或上下文描述。Open-AoE 的差异化优势是**数据 + toolchain 一起发布**，而不是每个单项标注都更接近物理真值。

## 20. 下游实验与因果证据

### 20.1 论文层面的严格结论

技术报告没有报告：

- robot task success rate / partial success；
- human-only、robot-only 与 human–robot co-training；
- action/contact/object label ablation；
- scaling curve 与 negative transfer；
- unseen object/scene/task/operator/robot 评测；
- real-to-sim 或 sim-to-real gap；
- 多机器人 retarget task success。

所以三类 gap 的论文证据等级是：Embodiment **接口层部分**，Task **数据覆盖层部分**，Reality **未解决**。

### 20.2 仓库 README 中的追加 POC 证据

以下结果出现在 2026-08-13 审计的官方代码仓库 README，不在技术报告主实验中。它们有工程价值，但大多缺统一评测集、完整 logs/checkpoints、多 seed 和真机 SR，引用时应写作“官方仓库报告”。

| Recipe | 数据/设置 | 仓库报告结果 | 正确解读 |
| --- | --- | --- | --- |
| SmolVLA | 2.9 h POC；offline held-out | flow-matching loss 1.03→0.43（20D）/0.51（26D）；256² 仅小幅增益且成本 2.7× | 只证明离线 action prediction 可学；无 robot/environment SR |
| iVideoGPT | 2.9 h，30k steps，4×4090 | loss 2.78→1.63；zero-action/true loss 差 25%→49%→62%；tokenizer 10k 后 +1.6 dB PSNR、LPIPS −36% | 26D 中 camera ego-motion 强；不是手部可执行性 |
| iVideoGPT 100 h | 60k steps | true/shuffle/zero loss 1.85/2.09/2.25；zero÷true 从 1.62 降到 1.22 | 扩数据后 controllability 反而变弱，作者怀疑手机 SLAM 噪声；是重要 negative scaling 信号 |
| GenieRedux | 2.9 h | LAM loss 0.634→0.0024；rollout PSNR 16.6 dB；Δ-PSNR +0.54 dB | 训练/重建收敛，不等于 task skill |
| GenieRedux | 100 h，128-res，n=128，两 seed | rollout PSNR 19.5 dB；Δ-PSNR +0.66 dB；camera:hand 每维约 2.85×，CI 约 [1.9,4.7] | 改进了 shuffle/CI；按总信号 hand20 仍略高于 camera6 |
| LAOM | POC，30 epochs/5280 steps，单 4090 | FDM 1.514→0.056；20% hand labels MSE 2.226→0.003；latent $R^2$ camera 0.27 vs hand 0.05，监督后 hand 0.52 | 说明 ego-motion 是 latent distractor；仍为表示实验 |
| AdaWorld LAM | POC full data，10 epochs/20k | loss 0.128→0.0025；KL 21→0.011 | 只跑 LAM core；8-GPU adaptable world model 不在配方范围 |
| DreamDojo | zero-shot GR1 robot post-train weights | PSNR 10.01 / SSIM 0.214 / LPIPS 0.702 | scene/hand 质量低，显示 robot→human domain gap；MANO post-train 只是 harness |
| VITRA | 111 clips/314,253 frames，100 steps，冻结 backbone | action loss 约 1.4→1.0 | smoke test，不是完整训练或真机成功 |

### 20.3 仓库 POC 揭示的真正工程问题

1. **Camera ego-motion 很容易主导视频 controllability**：对 egocentric world model 必须做 hand/camera 分通道消融。
2. **扩大到 100 h 不保证单调改善**：iVideoGPT 的比值下降说明 SLAM/action 噪声会吞掉规模收益。
3. **VQ codebook collapse**：GenieRedux 在 100 h 会缩到约 16–20/1024 codes、perplexity 约 9；仓库补丁用 EMA + dead-code revival 恢复到 1024/1024、perplexity 约 792。数据规模放大暴露了上游模型缺陷。
4. **外部 checkpoint/环境脆弱**：iVideoGPT 最终 `save_pretrained` 会崩，需读周期 checkpoint；DreamDojo 有 checkpoint revision、decoder 和 action-embedder 初始化补丁。
5. **“训练-ready”是 recipe-ready**：用户仍需克隆固定上游 commit、下载权重、安装特定 CUDA/Python 依赖并配置数据路径。

### 20.4 仍然缺失的因果对照

即使纳入仓库 POC，也没有回答：

- 视觉多样性 vs language vs MANO vs camera trajectory 各贡献多少；
- 保持总 token/steps 时，Open-AoE 是否优于重复 robot data；
- 相机 6D 的高 controllability 是建模价值还是 shortcut；
- world-model PSNR 是否提高 MPC/robot success；
- 自动 action labels 是否优于无标签 future prediction；
- 100 h→694 h→2000 h 是否继续增长或进一步 negative transfer；
- 人类动作与目标 robot action 的 latent 对齐是否在未见任务上有效。

## 21. 训练目标与 curriculum 建议

Open-AoE 没有一套统一 loss；不同 recipe 各用原生目标。可把合理的分阶段 curriculum 写成：

| 阶段 | 数据 | 目标 | Open-AoE 角色 | 缺口 |
| --- | --- | --- | --- | --- |
| Visual/video pretrain | RGB clips | masked/future/video token loss | 大规模场景与运动 | 易被 camera/background shortcut 主导 |
| Semantic alignment | RGB + description | contrastive/caption/temporal grounding | 原子动作与 bbox | VLM 伪标签偏差 |
| Motion reconstruction | RGB + MANO/camera | pose/action regression | 20–110D human action | 伪标签、遮挡补帧 |
| Latent action pretrain | video + optional 20/26D | reconstruction + dynamics + supervised grounding | LAM/world-model | latent 未必可执行 |
| Robot co-training | human + teleop | policy/action + cross-domain alignment | 视觉/语义/运动 prior | 需另采 robot bridge data |
| Sim RL/retarget | robotized trajectories | feasibility/contact reward | 初始化/候选 motion | 缺 object/contact truth |
| Real post-train | robot rollouts | success/advantage/force-aware loss | 只作 prior | 成本与安全由机器人侧承担 |

一个完整的概念目标可分解为

$$
\mathcal L
=
\lambda_{sem}\mathcal L_{sem}
+\lambda_{mot}\mathcal L_{motion}
+\lambda_{obj}\mathcal L_{object}
+\lambda_{con}\mathcal L_{contact}
+\lambda_{phy}\mathcal L_{feasible}
+\lambda_{dyn}\mathcal L_{dynamics}
+\lambda_{pol}\mathcal L_{policy}
+\lambda_{xemb}\mathcal L_{cross\text{-}embodiment}.
$$

Open-AoE 能直接支撑 $\mathcal L_{sem}$、部分 $\mathcal L_{motion}$ 和视觉 dynamics；$\mathcal L_{object}$、$\mathcal L_{contact}$、$\mathcal L_{feasible}$、真实 $\mathcal L_{dynamics}$ 与 $\mathcal L_{policy}$ 需要额外 robot/sim/contact 数据。

推荐 curriculum：先用高 `pred_valid`、低 camera acceleration、单手桌面动作；再加双手、强遮挡、多设备和 camera motion；最后用真实 robot contact 数据微调。不要一开始把所有 2000 h 等权混入，噪声规模可能压过少量高质量 robot supervision。

## 22. 采集经济性、扩展能力与 TCO

Open-AoE 展示了大规模手机采集和重型离线处理，但没有报告完整成本。不能把“用户已有手机”或“众包”直接等同于便宜数据。

| 成本项 | 论文披露 | 需要补的量 |
| --- | --- | --- |
| 采集硬件 | 消费级手机，多型号 | 是否补贴手机/支架、折旧、损坏、充电与流量 |
| 贡献者 | 500+ | 招募、培训、补偿、返工、退出/撤回处理 |
| 有效产率 | 未报告 raw→Part→accepted 比例 | 每小时录制产生多少分钟双手有效动作 |
| 上传 | 分批至 HF/ModelScope | 上行带宽、压缩、失败重传、hub 存储费 |
| GPU 处理 | detector、VLM、SLAM、HaWoR、BA | 每视频小时 GPU-hours、峰值显存、失败重跑率 |
| 人工审核 | HITL + 随机人工检查 | 每小时数据的 reviewer-minutes、一致性与语言成本 |
| 存储 | raw、undistorted、多个 visualization/npz/json | 每小时 GB、冷热分层、版本重复和 egress |
| 隐私/合规 | 自动擦除、匿名映射 | 误漏审计、申诉、删除、法律与安全运营 |
| 数据维护 | release note 已有删除/修复 | schema migration、checksum、版本索引、派生资产重算 |

最终有效数据产率应定义为

$$
\rho_{\mathrm{effective}}
=
\frac{H_{\mathrm{accepted\ interaction}}}
{H_{\mathrm{capture\ labor}}},
$$

总拥有成本可按

$$
\mathrm{TCO/h}
=
\frac{C_{human}+C_{device}+C_{upload}+C_{GPU}+C_{review}+C_{storage}+C_{governance}}
{H_{\mathrm{accepted\ interaction}}}.
$$

这两式是审计建议，论文没有给足分子/分母，无法估算真实 $/h。尤其 Open-AoE 同时存 raw video、去畸变视频和预渲染结果，数据卡中单个 sample 可达数百 MB；2000 h 的多副本存储/egress 会是显著成本。

### 22.1 平台扩展能力的实证与缺口

支持 500+ contributor、400+ device types 和活跃分批发布，说明平台已超出实验室小样本。但要证明可持续社区贡献，还需公开：

- 贡献者 onboarding/设备校准通过率；
- 网络中断、热保护、磁盘不足后的 resume；
- 相邻/近重复检测；
- 设备/用户级质量 dashboard；
- 自动隐私 precision/recall 与人工 escalation；
- 数据删除向 derived annotation/checkpoint 的传播；
- schema/version compatibility；
- 每批次 checksum、manifest 和变更日志。

当前 release notes 是好起点，却还不是完整 data lineage system。

## 23. 数据偏差、隐私与治理

### 23.1 论文披露的措施

- 贡献者在采集、处理、研究与公开发布前被告知并显式授权。
- 端侧进行隐私/数据脱敏；发布中不含账号、支付信息、唯一设备 ID 等个人标识。
- 人脸、敏感内容和元数据中的姓名/联系人被擦除或映射为匿名 ID。
- 内部保存完整 mapping，并称可在授权条件下逆转。
- 论文将用途限定为 research/technology，反对 recognition、profiling、surveillance 和 re-identification。
- ethics appendix 声称没有未解决的重大隐私问题。

### 23.2 仍未回答的治理问题

| 风险 | 当前材料 | 缺失机制 |
| --- | --- | --- |
| 旁观者同意 | 未详细说明 | 家庭/公共场所中的 notice、consent 与申诉 |
| 屏幕/地址/票据 | 声称敏感擦除 | 检测精度、人工复核、漏检事件流程 |
| 撤回/删除 | 未见公开流程 | contributor 如何撤回，已下载副本如何处理 |
| 派生数据 | 未说明 | 删除 raw 后 MANO、caption、checkpoint 是否同步删除 |
| 可逆匿名映射 | 内部保存 | 访问控制、审计日志、保留期、密钥/映射泄漏响应 |
| Demographics | 未提供 | 公平性审计与重建模型对不同手/肤色表现 |
| 危险行为 | 未分类 | 刀具、火、化学品与模型模仿安全标签 |
| 地域/家庭隐私 | 设备/场景丰富 | 地理信息、室内布局和长期行为模式保护 |
| Commercial use | 数据许可未命名 | 商用、再分发、二次标注、模型训练权利 |

设备 `brand/model/androidVersion` 是设备**型号元数据**，与作者说不公开唯一设备标识并不矛盾；但组合场景、时间、设备型号和轨迹仍可能形成 quasi-identifier。匿名化评估不能只看是否删除姓名。

### 23.3 数据许可的关键问题

官方仓库 README 表示原创代码 Apache-2.0，并提示 dataset distribution terms 可能不同、需读 `LEGAL.md`。但根 `LEGAL.md` 主要列代码和第三方组件，没有给 Open-AoE 视频/标注一个明确命名的数据许可证；Hugging Face 数据卡显示 `License: other`。因此截至核验日：

- **可下载**：是，按 tier/平台部分可下载；
- **代码可修改/再分发**：原创代码按 Apache-2.0；
- **数据可否商用、再分发、制作衍生标注**：公开材料不够明确；
- **论文中 research-only 的伦理语言**：不是一份完整可执行的数据 license；
- **MANO**：模型文件不随仓库，需注册并接受 MANO 条款；
- **H-RDT/HaWoR 等**：部分上游含 CC-BY-NC-ND 或其他限制，不能因总仓 Apache-2.0 就覆盖。

企业或公开 benchmark 使用前，应向作者取得书面数据条款，明确视频、派生 MANO/caption、再分发、商业训练、删除与模型权重的权利。

## 24. 开源代码与数据审查

### 24.1 仓库静态审计

在提交 `8f90dedc7c2092caad791586f600fe7a8286b546` 上：

- 约 732 个 tracked files；
- 约 330 个 code/text/config/patch 文件；
- 约 73,005 行 Python、shell、文档、配置与 patch；
- shallow checkout 约 361 MB（其中 Git 元数据约 84 MB）；
- `python3 -m compileall -q` 对仓库 Python 源码通过；
- 所有跟踪的 `.sh` 经 `bash -n` 通过。

这只证明 Python/shell **语法**在本地静态检查通过，不证明依赖安装、GPU kernel、模型下载、路径、语义或端到端运行正确。

### 24.2 训练配方不是统一平台

`aoe-training-ready/README.md` 明确说明：没有跨所有 recipe 的通用训练格式、统一 action spec、单一 converter 或单一 launcher；用户应选择目标模型目录。每个目录通常只提供：

- Open-AoE → 目标模型格式 converter；
- launcher/config；
- 少量 eval/plot；
- 针对固定上游 commit 的 patch；
- 环境变量占位路径与结果说明。

外部上游项目、完整预训练权重、MANO 文件、部分机器人资产和已生成视频通常不 vendored。由此“支持 14 个模型”的准确含义是**提供 14 类接入路线**，不是 14 个模型都能在干净机器上一键复现实验结果。

### 24.3 关键依赖/版本陷阱

| 模块 | 陷阱 | 影响 |
| --- | --- | --- |
| Visualization | 目标 mesh 渲染需 EGL/OpenGL GPU；CPU fallback 质量较低 | headless server 需配置驱动 |
| MANO | README 有一处措辞像“vendored”，但实际文件树只有下载/转换脚本，无 `.pkl/.npz` 模型 | 必须自行注册下载，许可受限 |
| iVideoGPT | 固定旧版 `numpy/huggingface_hub/datasets/peft`；I3D 强依赖；final save crash | 依赖漂移与 checkpoint 恢复复杂 |
| GenieRedux | 100 h scale VQ collapse，需仓库 patch + 环境变量 | 原生上游配置不稳 |
| DreamDojo | 480×640 hard-code；2B 仍需 ≥45 GB GPU；action slot/weight init patch | smoke 可做，post-train 成本高 |
| LAOM | Python 3.9、`numpy<2`、HDF5 全载内存 | 大数据需 streaming 改造 |
| VITRA | 需 MANO FK、外部 VITRA checkout 与 patch | 100-step smoke 不代表完整收敛 |
| H-RDT | 上游 HaWoR 许可含非商用/禁止衍生限制 | 企业/再分发需法律审查 |

### 24.4 数据可直接使用程度

| 资产 | 是否公开 | 直接性 | 备注 |
| --- | --- | --- | --- |
| Raw video | 部分 tier | 高 | 大、下载/版本成本高 |
| Undistorted video/intrinsics | 部分 tier | 高 | 每 sample 检查实际分辨率 |
| Camera trajectory | 有 | 中高 | 无 covariance/depth；需 QC |
| MANO/world-camera hands | 有 | 中高 | 需 filter `pred_valid`；MANO mesh 资产另取 |
| Object/scene state | 无统一 3D state | 低 | 只有 scene/object language 与 bbox |
| Contact/force | 无 | 不可直接 | 需另建 |
| Language/task | 有 | 高 | 当前主字段英语；注意字符串时间戳 |
| Retargeting | adapters 有 | 中 | 上游/权重/资产外置 |
| Sim assets | 部分机器人资产 | 中低 | 不覆盖所有目标本体/对象场景 |
| Training launchers | 多 recipe 有 | 中 | 无统一 lockfile/CI/end-to-end checkpoint |
| Evaluation | 数据分析与 recipe-specific offline eval | 中 | 缺统一 robot benchmark |
| Real robot interface | 未形成统一接口 | 低 | 需目标团队自行集成 |

### 24.5 复现等级判定

- **论文 100 h 数据属性分析**：在 tiny 完整下载、抽样与 evaluator prompt 可获得时接近可复现；仍需确认 exact split/seeds/model revision。
- **单条/小批数据可视化与转换**：较高可复现；需要 GPU/EGL、MANO 或目标上游依赖。
- **某个 training recipe 的 smoke run**：中等；固定 commit、依赖和权重后可做。
- **仓库 README 全部 POC 数字**：中低；缺集中 logs/checkpoints/seeds/硬件镜像。
- **全 2000 h 与所有 14 路线**：当前不可完整复现，数据尚未传完、TCO 极高。
- **真机 human-to-robot 成功率**：论文没有该实验，不能复现一个不存在的统一结果。

## 25. 分层复现路线

### Level 0：数据与许可预检

| 项目 | 内容 |
| --- | --- |
| 数据 | nano 3 h；记录 hub revision、manifest、checksum |
| 任务 | schema validator、视频/内参一致性、字符串时间转换、`pred_valid` 过滤 |
| 算力 | CPU + 20–50 GB 存储；可选单 GPU |
| 人力/周期 | 1 人，1–2 天 |
| 成功标准 | 所有 sample 可解码；帧数/轨迹/MANO/annotation 对齐；许可风险形成书面结论 |
| 主要风险 | hub 更新、数据条款不明确 |

### Level 1：单条 Ego 视频的手、相机与语义复核

| 项目 | 内容 |
| --- | --- |
| 数据 | 10–50 个跨设备/场景 clips |
| 依赖 | visualization、EGL/OpenGL；如渲 MANO mesh 则单独下载模型 |
| 核验 | raw/undistorted、intrinsics、bbox、hand overlay、world trajectory、atomic boundary |
| 算力 | 1×24 GB GPU 或 CPU fallback |
| 人力/周期 | 1–2 人，3–5 天 |
| 成功标准 | 90%+ clip 无 frame misalignment；人工抽样报告手/边界/object 错误类型 |
| 风险 | 补帧与真实观测混淆、bbox 坐标系错用 |

### Level 2：仿真 retargeting 与物理可行性

| 项目 | 内容 |
| --- | --- |
| 数据 | 100–500 个高 `pred_valid` clips，先单手桌面 |
| 依赖 | 选一个 retarget route、目标 robot URDF/MJCF、collision/contact model |
| 算力 | 单 GPU + MuJoCo CPU；优化并行 |
| 人力/周期 | 2–3 人，2–4 周 |
| 成功标准 | IK ≥90%；joint-limit/self-collision <5%；EEF error、contact phase 和 object displacement 可量化 |
| 风险 | 无 object/contact truth，必须补 3D/object 标注 |

### Level 3：Human–Robot co-training

| 项目 | 内容 |
| --- | --- |
| 数据 | tiny 100 h 的筛选子集 + 每任务 50–200 条 robot teleop；严格 contributor/scene split |
| 模型 | 先选 SmolVLA/GR00T 或 iVideoGPT 中一条，不并行追 14 条 |
| 实验 | robot-only、visual-only、semantic-only、motion-only、full；固定 total steps/tokens |
| 算力 | 4–8×24–80 GB GPU，依模型 1–3 周 |
| 人力/周期 | 3–5 人，4–8 周 |
| 成功标准 | ≥3 seeds；unseen object/scene/task；相对 robot-only 有统计显著 SR 提升且无已见任务回退 |
| 风险 | camera shortcut、negative transfer、数据许可和版本漂移 |

### Level 4：真机 whole-body / dexterous deployment

| 项目 | 内容 |
| --- | --- |
| 数据 | Level 3 policy + real robot recovery/contact rollouts + force/tactile |
| 系统 | System2 task/contact goal、System1 chunk、System0 impedance/force safety |
| 平台 | 目标 humanoid/EX002/dexterous hand；完整标定与 system ID |
| 人力/周期 | 5–8 人，2–4 个月起 |
| 成功标准 | 真实 task SR、恢复率、安全干预率、接触峰值、跨负载/设备稳定性 |
| 风险 | 缺 whole-body motion、不可达人手轨迹、接触/动力学 gap、硬件损伤 |

## 26. 面向人形、EX002 与灵巧手的落地判断

### 26.1 Humanoid

| 项目 | 建议 |
| --- | --- |
| 最有价值模态 | language、双手 wrist/fingertip、camera motion、双手协同时间结构 |
| 必须补齐 | body/head/floating-base/feet、ground contact、object pose、robot proprioception/force |
| 可直接用 | 桌面双臂 EEF prior、视觉/语言预训练、subgoal proposal |
| 只能提 intent | 行走中操作、蹲起、承重、全身支撑、平衡相关动作 |
| 推荐试点 | 固定底座或站立稳定条件下的双手开箱/收纳/开合抽屉 |
| 最小规模 | 2–3 tasks，100–300 筛选 human clips/task + 50–100 robot demos/task |
| 成功标准 | unseen object/scene SR 比 robot-only +10 pp 以上；跌倒/急停为零；接触峰值受控 |
| 责任分配 | Data/System2 负责语义与筛选；motion 团队负责 retarget；control 团队负责 balance/System0 |

Open-AoE 不能从 camera trajectory 可靠恢复 humanoid base：颈挂抖动、上身转动和脚步运动混在一起。whole-body policy 只能把它当视觉/任务先验，不能当 base reference。

### 26.2 EX002

| 项目 | 建议 |
| --- | --- |
| 最有价值模态 | wrist/world trajectory、object phrase、atomic boundary、camera intrinsics |
| 必须补齐 | EX002 TCP/palm calibration、workspace/reachability、object 6D pose、gripper/hand contact、相机外参 |
| 可直接用 | object-conditioned EEF corridor 与动作检索；若为双臂则用左右手协同 phase |
| 只能提 intent | 人手精细 finger articulation、柔性物操作、工具力控制 |
| 推荐试点 | 桌面容器 pick-place、盖子开合、抽屉/门把操作 |
| 最小规模 | 每任务 100 个高质量 human clips + 30–50 EX002 teleop demonstrations |
| 成功标准 | held-out object pose/scene 上 SR 提升；EEF collision、grasp drop、恢复时间受控 |
| 责任分配 | Perception 补 object state；planning 做 functional retarget；control 做 contact/force |

EX002 最可能从 wrist/object trajectory 而非完整 MANO 获益。若它是夹爪本体，应尽早把 45D finger pose 压成 object-centric grasp goal，不要追求视觉上复制人手。

### 26.3 Dexterous hand

| 项目 | 建议 |
| --- | --- |
| 最有价值模态 | 45D MANO、shape、fingertips、双手时序、verb/object |
| 必须补齐 | object mesh/pose、contact topology、normal、force/tactile、机器人手标定 |
| 可直接用 | hand pose prior、pre-grasp ranking、视觉表征、dex-retarget 初始化 |
| 只能提 intent | 强力旋拧、滑移控制、柔性/透明物体、工具稳定接触 |
| 推荐试点 | 刚性对象的瓶盖旋开、盒盖开合、已知工具的稳定抓取 |
| 最小规模 | 500–1000 clips，按手形/对象/视角筛选；每对象族 50+ tactile robot trials |
| 成功标准 | fingertip/contact-region error、grasp retention、slip rate、task SR 均优于 robot-only |
| 责任分配 | Hand perception 补 contact；retarget 团队做 topology；System0 做 tactile/impedance |

灵巧手是 Open-AoE 最有潜力也最容易被高估的方向：MANO 比夹爪 action 丰富，但没有物体表面和接触力时，手姿相似并不等于抓取稳定。

## 27. Related work：Open-AoE 的准确生态位

### 27.1 相比被动 Ego 数据集

Ego4D、Ego-Exo4D 等强调大规模人类行为理解；OpenEgo/EgoDex/EgoXtreme 更接近手部或具身任务。Open-AoE 的增量是把相机轨迹、MANO、密集原子语言和训练/retarget adapters 放在同一发布中。它不是第一个有 Ego 视频，也不应只以小时数排名；价值在**数据产品和接入面**。

### 27.2 相比机器人遥操作数据

Open X-Embodiment、DROID、BridgeData 等提供真实 robot observation/action，接触结果由本体执行产生；Open-AoE 的人类行为更自然、场景/设备可能更广，却没有 robot proprioception/command。两者互补而非替代：Open-AoE 做 pretraining/coverage，robot data 提供 action grounding 与 reality/contact correction。

### 27.3 相比 human-to-robot 视频方法

EgoBridge、FLARE/VITRA、EgoMimic 类工作关注 human/robot latent alignment 或 future feature；EgoInfinity、Do-as-I-Do、SPIDER 等关注 3D HOI/retarget。Open-AoE 不提出统一的新 co-training algorithm，而是通过 adapters 让这些路线消费同一数据。评估时应把“数据贡献”和“下游方法贡献”拆开。

### 27.4 相比 world-model/latent-action 数据

iVideoGPT、GenieRedux、LAOM、AdaWorld、DreamDojo 等需要视频和可选 action。Open-AoE 的 camera 6D 是 egocentric world model 的重要条件，也是潜在 shortcut/distractor。仓库 POC 对 camera-vs-hand 的消融，是工具链最有洞察的追加证据之一，但仍需对象状态与 policy evaluation 连接到机器人价值。

## 28. 作者没有完成但最值得做的未来工作

1. **完成 full 2000 h 发布并冻结版本**：manifest、checksum、schema version、删除/替换日志。
2. **给出明确数据许可证**：商用、再分发、衍生标注、模型权重、撤回/删除传播。
3. **发布严格 split**：contributor/scene/device/object/task disjoint，禁止相邻 Part 泄漏。
4. **补人工 GT benchmark**：camera、MANO、boundary、verb/object、bbox 的分层准确率和失败分布。
5. **区分 observed vs filled hand**：给逐帧来源 mask 和 uncertainty/covariance。
6. **补 object state/contact**：至少为代表性子集给 object 6D/articulation/contact region/phase。
7. **做完整 co-training causal matrix**：固定 token/steps，分解 RGB、language、hand、camera 的贡献。
8. **报告 scaling 与 negative transfer**：3 h、100 h、694 h、2000 h，多个 seed 和多个模型。
9. **连接 world-model 指标与 robot control**：不只 PSNR/LPIPS，要做 MPC/policy SR。
10. **多本体真机矩阵**：同一 human set → 至少两种手/两种臂/一个 humanoid，报告 feasibility 与任务成功。
11. **动力学与接触闭环**：加入 force/tactile、system ID 与 recovery rollouts，让 System0 能校正 human prior。
12. **治理工具化**：旁观者流程、撤回入口、派生数据删除、re-identification audit 与 demographic performance。

## 29. 十组关键问题的直接回答

### Q1. 语义是否真正同时对齐 human motion、robot action 与 contact？

只完成了语义—human video/hand 的共时对齐；robot action 依赖 converter/retarget，contact 只有弱推断。三者没有被同一真机 objective 联合验证。

### Q2. 2000 小时是 raw 还是有效时长？

论文称数据集约 2000 h，并对发布产品做 QC，但没有给 raw capture 与 accepted interaction 的换算。当前 full 也未全部公开；可下载规模要按 3 h/100 h/694 h/上传中分别写。

### Q3. 400+ 手机意味着什么？

它证明型号/设备域很宽，却不清楚是唯一物理设备数还是市场型号数。设备多样性增加视觉覆盖，也引入 intrinsics/ISP/SLAM 噪声；下游收益仍是待检验假设。

### Q4. MANO 能否直接变成机器人动作？

不能。它是人手重建，可转成 EEF/keypoint/joint prior；还需本体拓扑、可达性、碰撞、对象/contact 和低层控制。夹爪会丢大部分手指信息，灵巧手又需 contact topology。

### Q5. Contact intent 是否存在？

作为 verb/object/bbox/hand shape 的弱信号存在；作为显式 point/region/normal/force 或 observed/intended 标签不存在。

### Q6. 数据是否支持 humanoid loco-manipulation？

只支持 manipulation 的视觉、语义和双手先验。没有 whole-body/floating base/feet/ground contact，不能直接训练 locomotion + manipulation。

### Q7. 论文是否证明 human data 提高 robot success？

没有。论文主实验是数据属性分析。仓库有离线 POC loss、PSNR、latent controllability 和 smoke results，但没有统一真机 SR。

### Q8. 14 类模型是否都可一键训练？

不是。仓库是按模型划分的 converters/launchers/patches，外部 checkout、权重、依赖和路径仍需用户准备；没有统一 action spec/launcher/checkpoint。

### Q9. 开源程度如何？

代码和文档很强，数据分批真实可下载；但 full 未完成、App 未开源、MANO/上游资产外置、数据许可证未命名。因此是“工程开放度高、完整可复现度中等、法律开放度待澄清”。

### Q10. 最优先的落地方式是什么？

先用 tiny 的高质量子集做视觉/语言/hand-motion pretraining，再以小规模目标 robot teleop 和 contact 数据桥接；从固定底座、刚性对象、短时桌面任务开始，不要直接上全身/柔性物/高力任务。

## 30. 数据集评分卡

| 维度 | 评分（1–5） | 证据 |
| --- | ---: | --- |
| 名义规模 | 5 | 约 2000 h、500+ contributors；当前尚未全量上传 |
| 有效多样性 | 4 | CLIP 谱/设备/场景覆盖强；动作长尾和真实独立 task 数不清 |
| 标注密度 | 5 | 99.99% 时间覆盖、13.97 segments/min、密集语言/手/相机 |
| 标注准确性 | 3 | VLM 一致性与高 valid rate；缺大规模人工 GT 和逐帧 uncertainty |
| Human motion | 4 | world/camera MANO 与多表示；无 body/object/contact |
| Robot 可执行性 | 2 | adapters 丰富，但无 robot action truth/论文级真机 SR |
| Contact/physics | 1 | 无 contact、force、tactile、object dynamics |
| 开放程度 | 3 | 代码强、数据分批；full/许可/App/上游资产未闭环 |
| 治理 | 2 | 有 consent/匿名化声明；撤回、旁观者、派生删除与许可不清 |
| 复现难度 | 3 | nano/tiny 可起步；完整 toolchain 依赖复杂、GPU/TCO 高 |
| 业务价值 | 4 | 适合 System2/1 预训练和数据工程；不可替代 robot/contact 数据 |

### 最终判词

> 该数据集以 **500 多名贡献者使用 400 多类消费级手机进行自然第一视角采集，再经 VLM、SLAM 与双手 MANO 重建** 的方式获得约 **2000 小时名义规模的真实操作视频及密集语义/运动中间表征**，其中最可靠的机器人监督是 **第一视角视觉多样性、原子动作语义和相机/双手运动先验**；最缺失的物理信息是 **对象状态、真实接触/力、whole-body 状态与机器人动作真值**，因此它最适合用于 **VLA/world-model 预训练、human-motion prior 与 retargeting 研究**，不应直接被当作 **可执行、带接触动力学真值的机器人示范数据集**。

Open-AoE 的价值不在于它已经跨完了 Embodiment、Task 和 Reality 三条沟，而在于它把第一座桥修得足够宽：从杂乱手机视频到可查询、可视化、可重建、可转换的数据产品。接下来的科学问题，是证明桥上的每一种信号究竟能让哪一种机器人、在哪一种任务和接触条件下，稳定地走到另一端。
