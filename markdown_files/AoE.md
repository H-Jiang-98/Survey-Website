---
title: "AoE: Always-on Egocentric Human Video Collection for Embodied AI"
method_name: "AoE"
authors: [Bowen Yang, Zishuo Li, Yang Sun, Changtao Miao, Yifan Yang, Man Luo, Xiaotong Yan, Feng Jiang, Jinchuan Shi, Yankai Fu, Ning Chen, Junkai Zhao, Pengwei Wang, Guocai Yao, Shanghang Zhang, Hao Chen, Zhe Li, Kai Zhu]
year: 2026
venue: "CVPR 2026 Workshops (EmbodiedAIinLife)"
tags: [egocentric-data, data-collection, smartphone, edge-cloud, human-video, humanoid-manipulation, cross-embodiment, flare, groot, privacy]
image_source: online
---

# AoE：一台手机、一个颈挂支架，真的能变成机器人数据工厂吗？

> 精读基于 [CVPR 2026 Workshop 官方论文](https://openaccess.thecvf.com/content/CVPR2026W/EmbodiedAIinLife/html/Yang_AoE_Always-on_Egocentric_Human_Video_Collection_for_Embodied_AI_CVPRW_2026_paper.html)、[官方 supplementary](https://openaccess.thecvf.com/content/CVPR2026W/EmbodiedAIinLife/supplemental/Yang_AoE_Always-on_Egocentric_CVPRW_2026_supplemental.pdf) 与 [arXiv:2602.23893v2](https://arxiv.org/abs/2602.23893)。公开资源状态核验日期：2026-08-13。论文与附录没有报告的数量、协议或配置均明确标为“未报告”；后续 Open-AoE 的数据规模与产品数字不回填到本文。

## 阅读结论先行

AoE 的主要贡献不是一个已经发布的大数据集，而是一条低门槛的人类第一视角数据生产线：用户把自己的 Android/iOS 手机固定在胸前，端侧模型检测手—物交互并选择性录像，用户本地检查后授权上传，云端再完成动作切分、相机轨迹、MANO 手重建、场景重建、生成式增强与质量过滤。颈挂组件成本低于 20 美元，手机被视为用户已有资产；架构声称可服务数千并发设备。

论文的机器人结果值得重视：在 Unitree G1 + Inspire 五指手上，以 GR00T N1.5/FLARE 将每任务 200 条 AoE 人类视频加入 50 条机器人遥操作，Pick & Place 成功率从 45% 提至 75%，Close Laptop 从 45% 提至 95%，长时序双手任务从 0% 提至 20%。但 Fold Scarf 仍为 10%，没有改善。这说明同任务人类视频可以为有限机器人数据提供结构先验，却没有证明任意自然视频都能迁移。

最需要警惕的是证据链没有完全闭合。真机实验的 supplementary 明确说 FLARE 从**无动作标签的人类视频**学习 future latent alignment；论文没有证明云端生成的 MANO、相机轨迹、语义标签、机器人 inpainting 或 real-to-sim 资产实际被该策略使用，也没有对这些模块做端到端移除消融。因此，本文支持的是“**AoE 视频 + FLARE 共训有效**”，而不是“自动标注流水线的每个产物都已经被证明能提升机器人”。

同时，AoE 不是可下载数据资产。截至核验日，作者主页只给出论文入口，未发现官方项目页、代码、App、数据、checkpoint、硬件 CAD 或数据许可。正文把大规模开源数据集列为未来工作。论文也未报告 AoE 总小时数、采集者人数、设备分布、场景统计、成功/丢弃比例或公开 split。因此它首先是一篇**系统原型与小规模下游验证论文**，不是一份可复用的大规模数据集发布。

### 一句话总结

AoE 证明了“低于 20 美元的颈挂支架 + 现有手机 + 用户授权上传”可以采集对同任务人形机器人共训有用的人类视频，但没有发布数据/代码，也没有通过端到端消融证明其昂贵的云端几何标注和生成增强是机器人增益的原因。

### Elevator pitch

机器人遥操作数据精准却昂贵，互联网第一视角视频便宜却噪声大、动作弱。AoE 把采集端做轻：人体本身执行自然动作，手机只负责稳定拍摄、端侧触发和本地保存；把处理端做重：云端以 VLM、深度、SLAM、手重建和生成模型把长视频加工成原子动作片段。策略侧则不直接 retarget MANO，而用 FLARE 把人类未来视频表征与机器人 action flow matching 共训。它展示了一个有前景的“human video : robot data”配方，却尚未给出开放、可审计、可规模复现的数据工厂。

![AoE 总览：颈挂手机、端侧触发与云端自动处理](https://arxiv.org/html/2602.23893v2/AoE_figure1_v4.png)

## 0. 资源、公开状态与许可

| 资产 | 正式入口 | 截至 2026-08-13 的公开内容 | 许可/条款 | 可复现程度 |
| --- | --- | --- | --- | --- |
| 论文 | [CVF HTML](https://openaccess.thecvf.com/content/CVPR2026W/EmbodiedAIinLife/html/Yang_AoE_Always-on_Egocentric_Human_Video_Collection_for_Embodied_AI_CVPRW_2026_paper.html)、[PDF](https://openaccess.thecvf.com/content/CVPR2026W/EmbodiedAIinLife/papers/Yang_AoE_Always-on_Egocentric_Human_Video_Collection_for_Embodied_AI_CVPRW_2026_paper.pdf)、[arXiv](https://arxiv.org/abs/2602.23893) | 10 页正文；CVPRW 接收版本 | arXiv 页面标注 CC BY 4.0；这只覆盖论文文本/图，不自动覆盖数据与软件 | 高 |
| 附录 | [CVF supplementary](https://openaccess.thecvf.com/content/CVPR2026W/EmbodiedAIinLife/supplemental/Yang_AoE_Always-on_Egocentric_CVPRW_2026_supplemental.pdf) | 6 页：评级口径、App 流程、分布式系统、训练设置、隐私摘要 | 随论文发布 | 高 |
| AoE 项目页 | 未发现 | 作者主页的 AoE 条目只有 Paper 链接 | — | 不可用 |
| 采集 App | 未发现官方安装包/商店页/源码 | 论文截图与流程描述；Android/iOS 均声称支持 | 未报告软件许可 | 不可复现 |
| 硬件 | 论文图示 mechanical/MagSafe/magnetic 颈挂和稳定带 | 无 BOM、CAD、打印件、具体型号、重量或装配规范 | 未报告 | 只能自行近似搭建 |
| AoE 原始/处理数据 | 未发布 | 真机实验每任务 200 条人类示范；未给总库规模 | supplementary 隐私摘要称上传即完整转让知识产权并给予平台广泛使用权；这不是对研究者的数据许可 | 不可下载 |
| 自动处理代码 | 未发现 | Qwen3-VL、LingbotDepth、MegaSAM、HaWoR、MANO、Masquerade 等模块名称与阈值 | 依赖各上游项目许可；AoE glue code/版本/配置未给 | 低 |
| 策略代码与权重 | 未发现 AoE 专属发布 | GR00T N1.5 + FLARE 的高层训练说明 | 上游项目许可与 AoE 训练产物许可分开；AoE checkpoint 未给 | 低 |
| 配置/日志/随机种子 | 未发布 | Close Laptop 最大配方 60k steps；Fold Scarf 100k steps | — | 不可精确复现 |

### 当前公开状态的严格结论

- **已公开**：论文、supplementary、arXiv HTML 中的图和书面隐私摘要。
- **未发现公开**：数据下载、数据卡、数据许可、App、服务 API、系统源码、模型权重、训练配置、采集硬件设计文件、评测日志。
- **论文明确承诺但尚未完成**：结论写的是未来将发布基于 AoE 的大规模开源第一视角视频数据集。
- **不能混淆**：论文的 CC BY 4.0 不能推导出 AoE 视频也是 CC BY；“仅限 embodied AI research”是平台用途限制描述，也不是一份可执行的数据集许可证。
- **外部检索边界**：作者 Jin-Chuan Shi 的主页对同页其他论文会列 Project Page/Code，而 AoE 条目只列 Paper，这进一步支持“当前无官方项目/代码入口”的判断，但无法证明内部系统不存在。

## 1. 论文速览与核心假设

| 项目 | 判断 |
| --- | --- |
| 分类 | Egocentric collection system + automated data processing + human/robot co-training validation |
| 生态位 | 位于被动 Ego 视频、主动 wearable/UMI 和机器人 teleoperation 之间；前端接近消费级被动采集，后端是重处理 |
| 目标用户 | 采集侧：无机器人经验的分布式贡献者；使用侧：具身数据工程、VLA/人形/灵巧操作研究团队 |
| 采集年份 | 未报告；论文仅给 2026 年发表时间，不能据此推断全部视频的采集年份 |
| 主要机器人任务 | Pick & Place、Fold Scarf、Close Laptop、Push Bowl & Pour Seeds |
| 三个关键词 | neck-mounted smartphone；edge-cloud；future latent alignment |
| 核心贡献 1 | 用用户已有手机与低于 20 美元颈挂支架降低每个采集者的新增硬件门槛 |
| 核心贡献 2 | 将端侧 hand/action gating、用户本地审核与云端自动标注/过滤组织成可扩展工作流 |
| 核心贡献 3 | 在 Unitree G1 上证明同任务 AoE 人类视频可增强有限 teleop 数据 |
| 最强 claim | 50 teleop + 200 AoE 在 Close Laptop 上将 SR 从 45% 提到 95% |
| 最弱环节 | 未做 pipeline-product → policy 的端到端归因；没有公开数据/代码；缺少 trial 数、置信区间和泛化 split |
| 数据论文属性 | 更像 collection infrastructure paper，而非 dataset release；无总量、统计分布和可下载资产 |
| 精读优先级 | 5/5：轻量 Ego 采集与人类视频共训的代表性系统 |
| 复现优先级 | 2/5：组件可辨认，但数据、App、orchestration 和训练 recipe 不开放 |
| 真机试点优先级 | 4/5：可先复现低风险的同任务 Close Laptop/Pick & Place 配方 |

作者可检验的核心假设应收窄为：

> 在任务匹配、机器人平台和 foundation policy 固定时，用更多 AoE 人类第一视角视频对未来状态表征施加监督，可以在有限机器人 action supervision 下改善部分真机任务成功率。

论文支持这一狭义假设。它没有检验全球采集规模、跨任务/跨场景泛化，也没有证明所有自动标注产物都参与并造成了增益。

## 2. Gap—Evidence 总表

| Gap | 作者机制 | 直接指标 | 对应消融 | 真机证据 | 结论 |
| --- | --- | --- | --- | --- | --- |
| Embodiment | FLARE future latent alignment 学人类视频；robot teleop 用 action flow matching | SR/PSR | human 数据量 0/50/200；robot 数据量 10/50（仅 Close Laptop 图） | Unitree G1 + Inspire 五指手 | 部分跨越；依赖同任务机器人数据，非零样本 retargeting |
| Task | 原子动作切分、语义标签、同任务 human/robot recipe | 四个任务 SR/PSR | 无语义标签/切分移除；无 held-out task/object | 有 | 部分；只证明任务内增强 |
| Reality | 真手机视频 + 真机器人 rollout；另做 AGILE real-to-sim | 真机 SR/PSR；penetration depth | 无 sim-only/real-only；无 domain randomization 消融 | 有 | 部分；真机证据强于纯仿真，但机制归因弱 |
| Semantic–motion | Qwen3-VL 原子动作；FLARE future feature | segmentation 未报告准确率；policy SR | 无 label shuffle/no-VLM | 有但 policy 未说明使用语义标签 | 证据不足 |
| Motion–contact | HaWoR/MANO + depth/SLAM；AGILE HOI | MPJPE/ATE；penetration <2 mm | 无 hand/trajectory/contact product 移除 | 机器人策略未证明使用这些产物 | 中间表征可用，端到端作用未证实 |
| Scale | 端侧选择 + edge ingestion + Kubernetes HPA | upload latency 500 ms+ → <100 ms；数千并发 | 无公开负载曲线/成本/失败率 | 非机器人指标 | 架构可行性陈述，缺生产级可审计证据 |

## 3. 数据来源、统计单位与五层监督

### 3.1 数据来源

| 数据源 | 论文披露规模 | 传感器/视角 | 监督 | 用途 |
| --- | ---: | --- | --- | --- |
| AoE 用户手机视频 | 总小时、采集者数、设备数未报告 | sternum/chest 第一视角，手机 ultra-wide RGB；intrinsics 与 sensor metadata | 端侧手/动作触发；云端动作语义、camera trajectory、MANO、2D hand mask 等 | 系统流水线；每任务 200 条用于真机共训 |
| EgoDex test | 约 7 h | Ego RGB + 数据集手信息 | 用于评估手重建与相机轨迹 | preprocessing precision |
| Ego4D test | 约 6.5 h，带 VITRA annotations | 被动 Ego video | 手/轨迹评估参考 | preprocessing precision |
| in-house RealSense L515 | 10 h | RGB-D | 深度相机参考 | 标定/轨迹/几何评估 |
| AR glasses + smartphone paired set | “small”，未报告条数/小时 | Apple Vision/AR glasses 与手机配对 | 硬件追踪参考 | 手重建对比 |
| Robot teleoperation | 每任务 50 条主实验；Close Laptop 另有 10 条配方 | Unitree G1 + Inspire hand；观测/动作格式未报告 | robot action labels | action flow matching |

### 3.2 五层监督：不要把视频直接叫作 robot data

| 层次 | AoE 提供什么 | 完整度 | 机器人含义 |
| --- | --- | --- | --- |
| Raw observation | RGB video、factory intrinsics、sensor metadata | 中；分辨率/fps/codec/IMU 字段未报告 | 可做视觉预训练与 future prediction |
| Human state | HaWoR 3D joints + MANO mesh，结合 depth 与 camera pose 进世界系 | 中；是离线估计，不是原生 mocap | 可做人体/手部 motion prior，但噪声需传播 |
| World state | dense depth、camera trajectory、scene reconstruction | 中低；无稳定 object ID/6D pose/articulation schema | 能近似几何，不能直接定义完整 Markov state |
| Interaction state | 原子动作语义、2D hand mask、潜在 HOI；AGILE 可构造接触场景 | 低；无公开 contact point/normal/force/tactile | 可推断 contact intent，不能当作真实力学监督 |
| Robot supervision | robot teleop 的 action flow matching | 仅独立 robot 数据具有 | 人类视频本身不含 robot joint/action；FLARE 以 latent alignment 连接两域 |

### 3.3 数据单位与未报告项

- **raw recording**：端侧检测持续运行，只有检测到相关手—物交互才开始保存视频。
- **local clip**：时长不足会自动丢弃；合格 clip 本地保存，用户可浏览、剪裁、删除敏感/无效段。
- **uploaded clip**：用户手工选择与验证后，连同 sensor metadata 批量上传。
- **atomic clip**：Qwen3-VL 给长视频切出 start/end、verb、object、description、bbox、confidence；潜在 VLM 幻觉由人工验证纠正。
- **robot experiment demonstration**：表中“200 AoE”表示对应同一任务的 200 条人类视频，不是 200 小时。
- **未报告**：每条平均时长、有效 interaction time、总录制/上传小时、用户数、地域/职业/人口属性、手机型号分布、昼夜/室内外分布、接受率、重复率、hard-negative 比例和 train/test split。

由此不能计算：有效分钟/人时、每小时成本、每 GB 有效动作数、长尾覆盖率、数据增长曲线或真实全球规模。

### 3.4 传感器角色审计

- **AoE 正式采集**：消费级 smartphone rear ultra-wide RGB 是主传感器；factory intrinsics 与未展开字段的 sensor metadata 随视频保存。
- **颈挂支架**：只负责视角与稳定，不产生动作/力真值。
- **RealSense L515 RGB-D**：只出现在 10 h in-house 精度评估，不是 AoE 全球采集的必需硬件。
- **AR glasses/Apple Vision + phone paired set**：只用于小规模手重建对照；不是普通贡献者配置。
- **未报告参与正式采集**：Aria、VR、外部 mocap、数据手套、eye gaze、force/tactile。也不能仅凭“sensor metadata”推断已公开 IMU/GPS。

## 4. 采集硬件、协议与人机工程

![AoE 硬件、App、端侧触发与用户授权流程](https://arxiv.org/html/2602.23893v2/aoe-hardware-app-v4.png)

| 项目 | 论文报告 | 没有报告 | 风险/解释 |
| --- | --- | --- | --- |
| 相机位置 | sternum/chest，ultra-wide rear camera | 精确相机—胸骨外参、FOV、滚转/俯仰、用户校准 | 视点低于眼睛；“近似人类视觉”不是几何等价 |
| 固定方式 | mechanical clamp、MagSafe-compatible、magnetic；辅助稳定带 | BOM、CAD、磁吸保持力、抗冲击、佩戴重量 | 不同手机质量/镜头会造成运动与视场域差异 |
| 成本 | mount assembly < \$20 | 手机、流量、电池、云计算、人工审核、赔付/激励 | 是**增量支架成本**，不是总数据成本 |
| App | Android/iOS；用户授权的 Always-On mode | OS 版本、安装入口、最低 SoC、功耗、温升、断网行为 | “24h/all-day”是设计目标，未给 24 小时耐久实验 |
| 端侧模型 | hand detection、motion tracking、open-set recognition/action recognition | 模型架构、大小、FPS、阈值、误检/漏检 | 自动 gating 会系统性漏掉手被遮挡或无明显手运动的有效交互 |
| 原始存储 | 所有初始模型 inference 与 raw storage 在本地 | 缓存期限、加密方式、崩溃恢复 | 本地化降低泄露面，但并非零风险 |
| 上传 | 用户 review + explicit authorization | 断点续传、压缩、带宽成本、校验和 | 用户审核是关键人工环节，不应称全自动 |
| 旁观者提醒 | 连续、不可修改的音频信号 | 音量、语言、听障支持、公共场所规则 | “提醒”不等于取得旁观者同意 |

### 自然性与选择偏差

AoE 比手持 UMI 和 AR/VR 头显更少占用双手，但仍不是纯自然观察：胸前手机与稳定带可见、可感知，持续提示音会改变参与者与旁观者行为；端侧只在手—物模式满足阈值时录像，又会改变数据分布。论文给 AoE 非侵入性 4/5、scalability 5/5、deployment ease 5/5、data quality 4/5，这些是作者制定的定性星级，不是受试者量表或跨系统实测。

## 5. 从 raw video 到训练片段的数据流水线

~~~mermaid
flowchart LR
    A["用户授权 Camera/Always-On"] --> B["端侧持续检测\nhand + motion + open-set action"]
    B -->|不满足或过短| X["不录制/本地丢弃"]
    B -->|检测到交互| C["本地 RGB + intrinsics\n+ sensor metadata"]
    C --> D["用户浏览、剪裁、删除\n并显式选择上传"]
    D --> E["近端 edge ingestion\n异步跨区同步到 cloud"]
    E --> F["去畸变 + Qwen3-VL\n原子动作切分/语义"]
    F --> G["LingbotDepth + MegaSAM\n深度/相机轨迹/场景"]
    F --> H["HaWoR + MANO\n3D hand + world transform"]
    G --> I["质量门控\n速度/重投影/5% 人审"]
    H --> I
    I --> J["raw / undistorted / action commands\nmanual model / robot ego model"]
    I --> K["background replacement\nrobot inpainting / real-to-sim"]
~~~

![自动标注与增强：动作切分、场景/手重建、背景替换和机器人 inpainting](https://arxiv.org/html/2602.23893v2/aoe-figure4-egofactory.png)

### 5.1 六阶段处理的证据等级

| 阶段 | 方法 | 输出 | 质量控制 | 论文是否验证其下游因果作用 |
| --- | --- | --- | --- | --- |
| Camera calibration | Android Camera2 factory intrinsics | $K$、distortion | 与离线 checkerboard 比较 | 只验证标定误差；没有 policy 消融 |
| Atomic action segmentation | Qwen3-VL-235B-A22B | temporal boundary、verb/object/description/bbox/confidence | 人工修正 hallucinated boundary/label | 未报告准确率、人工工时；policy 未说明使用 label |
| Camera trajectory | LingbotDepth + temporal consistency + MegaSAM robust kernel | metric/scale-aware trajectory | ATE/ATE-S/RPE | 有几何 benchmark；无 policy 消融 |
| Hand reconstruction | HaWoR + depth rescale + MANO + sliding window | 3D joints/mesh/world pose | PA-MPJPE/MPJPE/AUC | 有几何 benchmark；真机 FLARE 称 human video 无 action label |
| Data augmentation | Masquerade hand/body erase、6-DoF robot align、photometric rerender；video diffusion background | robot-inpainted/background-changed clips | 只展示图 | 无 realism/user study/policy 消融 |
| Quality control | $>3\sigma$ joint velocity、$>5$ px reprojection error；5% manual inspection | accepted、failed、hard negatives | adaptive adjustments/reannotation | 未报告接受率、precision/recall 或 hard-negative 统计 |

### 5.2 几何链路的记账公式

论文没有给完整坐标公式；按其文字，单帧手点从相机系进入世界系可写成：

$$
\tilde{\mathbf p}^{W}_{j,t}
=
\mathbf T^{W}_{C_t}
\begin{bmatrix}
s_t\mathbf p^{C_t}_{j,t}\\1
\end{bmatrix},
$$

其中 $\mathbf p^{C_t}_{j,t}$ 来自 HaWoR/MANO，$s_t$ 由 depth estimate 重定标，$\mathbf T^{W}_{C_t}$ 来自 MegaSAM/SLAM。这个式子是对论文流程的显式记账，不是作者新提出的优化目标。它揭示误差会串联：depth scale、camera pose 和 hand pose 的误差会共同进入 world-frame hand motion。

质量门控可近似写成：

$$
\text{reject}(x_t)
=
\mathbb 1\!\left[z(\lVert\dot{\mathbf q}_t\rVert)>3\right]
\lor
\mathbb 1\!\left[e_{\mathrm{reproj},t}>5\ \mathrm{px}\right].
$$

论文没有说明 $3\sigma$ 是按 joint、动作类别、用户还是全局估计，也没有说明遮挡造成的错误低速轨迹是否会漏过门控。

### 5.3 信息损失与筛选偏差

- hand/action gate 先于录像：**被模型漏检的动作永远不会进入云端**，后处理无法挽回。
- 过短片段被丢弃：快速点击、轻触、短接触可能被系统性低估。
- VLM 切原子动作：长时依赖、失败恢复、并行双手动作可能被离散边界破坏。
- MANO 描述人手表面/关节，不等于机器人手可达配置；depth/SLAM 的共同误差会产生看似平滑但整体漂移的世界轨迹。
- background replacement 只改变外观不一定保持光照、反射、遮挡和 contact physics；robot inpainting 可以生成视觉机器人，却不会生成真实 action/force。
- hard negatives 被送去重标并不自动校准收集分布；若只修“看见的失败”，未知漏检仍不可观测。

## 6. 分布式 edge-cloud 系统

![AoE 的多区域 edge ingestion、可配置 pipeline 和弹性资源调度](https://arxiv.org/html/2602.23893v2/aoe-figure5-server-pipeline.png)

~~~mermaid
flowchart LR
    D1["Region 1 phones"] --> E1["nearest edge node"]
    D2["Region 2 phones"] --> E2["nearest edge node"]
    E1 -->|hourly async replication| S["central multi-modal object storage"]
    E2 -->|hourly async replication| S
    S --> O["declarative workflow orchestrator"]
    O --> CPU["CPU operators\nI/O + conversion"]
    O --> GPU["GPU operators\ndepth + hand reconstruction"]
    Q["queue depth + latency"] --> HPA["Kubernetes HPA"]
    HPA --> CPU
    HPA --> GPU
    CPU --> P["data products"]
    GPU --> P
~~~

| 系统 claim | 论文数字 | 缺失的审计信息 |
| --- | ---: | --- |
| 跨地域上传延迟 | 500 ms+ 降至 <100 ms | 地区、网络、payload、p50/p95/p99、测量次数、对照架构 |
| 并发 | 支持 thousands of devices | 峰值连接数、持续吞吐、上传 GB/s、错误率、背压行为 |
| 新算法集成 | weeks → days | 基线工程规模、参与人数、模块复杂度、代码差异 |
| 弹性响应 | workload spike 后 minutes | HPA 阈值、冷启动、GPU 可用性、SLO |
| 一致性 | hourly asynchronous cross-region replication | 数据丢失窗口、冲突策略、校验和、灾难恢复 RPO/RTO |

因此系统章节证明了合理的 cloud-native 设计，而不是可独立复核的 production benchmark。更重要的是，低于 20 美元只描述 mount；全球数据工厂的主要变动成本很可能转移到带宽、GPU 推理、对象存储、人工验证、隐私处理、激励和申诉。

## 7. Video → Human State → Intent → Robot Behavior

~~~mermaid
flowchart TD
    V["AoE ego RGB"] --> HS["Human state\nMANO/hand pose + camera trajectory"]
    V --> SEM["Atomic semantics\nverb/object/boundary"]
    HS --> INT["Potential intent/contact\n论文未显式建模"]
    SEM --> INT
    V --> FL["FLARE future latent alignment\n人类视频无需 action label"]
    RD["Robot teleop\nstate + executable action"] --> AFM["Action flow matching"]
    FL --> CO["GR00T N1.5 co-training"]
    AFM --> CO
    CO --> RA["Unitree G1 action"]
    RA --> SYS0["Low-level hand/arm control\n论文未披露"]
    INT -. "未证明用于真机策略" .-> CO
~~~

这张图必须按实线/虚线阅读：论文在系统侧构造了 Human State 与语义，在策略侧则用无动作标签视频做 latent alignment。两者同属 AoE 生态，但论文没有写出“MANO/语义/相机轨迹 → FLARE 输入”的实线接口。

### 最重要的因果断点

1. 几何 benchmark 证明 HaWoR/MegaSAM 处理可以达到较低误差。
2. AGILE 展示 AoE 视频可以产生 simulation-ready HOI 资产。
3. FLARE 真机实验证明 AoE 人类视频可以帮助共训。
4. **没有实验比较** raw video、filtered video、+semantic、+MANO、+trajectory、+augmentation、+real-to-sim 的逐级增益。

因此不能把 1/2 自动视为 3 的原因。最小闭环消融应固定 50 teleop 和同一批人类原视频，逐项启用这些产物并重复多个 seed。

## 8. 语义—人体运动—机器人动作—接触对齐

| 层 | Human 表示 | Robot 表示 | Contact 表示 | 对齐机制 | 验证 |
| --- | --- | --- | --- | --- | --- |
| 任务语义 | VLM verb/object/description | GR00T instruction 的具体输入未报告 | 无 | 可能由视觉 latent 隐式承载；没有接口说明 | 无 label/no-language 消融 |
| 时间边界 | atomic start/end | robot episode/phase | 无显式 contact phase | Qwen3-VL segmentation；policy 侧未说明使用 | 无 boundary F1 |
| 人体运动 | 3D joints、MANO、camera/world trajectory | teleop action label | 无 force/tactile | 几何重建；policy 并未直接 retarget | MPJPE/ATE 与 SR 分开评估 |
| 未来意图 | future human frame feature | future robot state/action | 隐式 | FLARE latent alignment | 人类数据量消融提供间接证据 |
| 抓取目标 | MANO pre-shape/手—物视觉 | Inspire hand configuration | 未显式建模 | foundation model 自行吸收 | 无 grasp/contact 指标 |
| 可执行动作 | 无 | action flow matching 输出 | low-level contact 未报告 | robot teleop supervision | 真机 SR/PSR |
| 力学闭环 | 无 | controller 未报告 | 无 contact wrench/compliance | 未实现/未披露 | 无力、滑移、碰撞指标 |

AoE 的强项是把**语义和人体运动都变成可计算候选监督**；弱项是没有定义明确的 object-centric contact bridge。对灵巧操作而言，手指轨迹相似仍可能对应不同摩擦锥、接触法向、抓取稳定性与对象结果。

## 9. Embodiment Gap：不是 MANO 直接映射到 Inspire Hand

### 9.1 论文真正使用的迁移方式

supplementary 对 FLARE 的描述是：从无动作标签的人类视频预测未来人类动作的 latent feature，使预测 embedding 接近未来真实帧 feature；robot teleop 则接受 action flow matching supervision。联合训练的记账形式可写为：

$$
\mathcal L
=
\mathbb 1[x\in\mathcal D_H]\,\lambda_H\mathcal L_{\mathrm{future\ latent}}
+
\mathbb 1[x\in\mathcal D_R]\,\lambda_R\mathcal L_{\mathrm{action\ flow}},
$$

其中 $\mathcal D_H$ 是 AoE human video，$\mathcal D_R$ 是 robot teleop。该式是根据附录文字整理的监督路由，不是论文披露的精确 loss；作者未给 $\lambda_H,\lambda_R$、latent 定义、sampling ratio 或 batch construction。

这是一种**表征级/动力学先验级 co-training**，不是以下三种方式：

- 不是把 MANO joint angle 一一映射到 Inspire hand joint；
- 不是以 IK/WBC retarget 人腕到机器人末端；
- 不是从视频恢复完整 robot action label 后做普通行为克隆。

### 9.2 Gap 分解

| 子 gap | AoE/FLARE 如何处理 | 仍缺什么 |
| --- | --- | --- |
| 形态 | latent future feature 避免直接同构 | 没有证明 feature 对不同手型/臂长不敏感 |
| 自由度 | robot action 只由 teleop 给出 | action 维度、手 synergy、关节限位未披露 |
| 坐标 | human camera/world trajectory 可估计 | policy 实际使用哪个坐标系未报告 |
| 观测 | 两域都有 RGB，可共享视觉 backbone | 相机位置、FOV、背景、手外观域差异无显式消融 |
| 动力学 | future latent 可学状态转移先验 | 质量、摩擦、延迟、执行器带宽仍来自 robot data |
| 接触 | 视频隐含接触结果 | 无 force/tactile/contact label，无法直接校准接触稳定性 |
| 安全 | 真机数据把可执行 action 锚定在机器人域 | 无约束层、碰撞率、急停和失败恢复报告 |

### 9.3 Retargeting 类型判定

AoE 主策略应归类为 **latent co-training / cross-embodiment representation alignment**。系统还具备几何级 MANO/world trajectory 和视觉级 robot inpainting，但论文没有证明这两类产物参与真机控制。因此将论文概括为“先重建人手，再 retarget 到机器人”是不准确的。

## 10. Task Gap：原子动作丰富，不等于任务泛化

真机任务覆盖四种有意义的操作结构：

| 任务 | 主要难点 | 50 Teleop SR | +200 AoE SR | PSR 变化 | 结论 |
| --- | --- | ---: | ---: | ---: | --- |
| Pick & Place | instruction following、定位与抓放 | 45% | 75% | 未报告 | +30 个百分点，明确增益 |
| Fold Scarf | deformable object、高频反应 | 10% | 10% | 未报告 | 无增益；作者归因硬件 latency |
| Close Laptop | articulated kinematics、持续接触 | 45% | 95% | 66.3% → 97.5% | 最强证据，+50 个百分点 |
| Push Bowl & Pour Seeds | 长时序、双手、细粒度协调 | 0% | 20% | 30% → 48% | 从完全失败到部分成功，但绝对值仍低 |

![AoE 人类示范、机器人 teleop 与四个 Unitree G1 任务](https://arxiv.org/html/2602.23893v2/aoe-realword-exp.png)

### 10.1 Close Laptop 数据配方

Figure 8 的四阶段是 Raise Hand → Press Screen → Move Inward → Fully Close：

- 10 teleop baseline：0% → 0% → 0% → 0%。
- 10 teleop + 50 AoE：25% → 10% → 0% → 0%；有少量启动动作但不能完成。
- 10 teleop + 200 AoE：100% → 95% → 70% → 55%。
- 50 teleop baseline：100% → 70% → 50% → 45%，终点与主表 45% SR 一致。
- 50 teleop + 50 AoE：100% → 90% → 55% → 35%；中间阶段较强，但 Fully Close 反而比 50-teleop baseline 低 10 个百分点。
- 50 teleop + 200 AoE：100% → 100% → 95% → 95%。

![Close Laptop 的 human/robot 数据量消融](https://arxiv.org/html/2602.23893v2/exp3_ablation_data_recipe.svg)

这里有三个关键判断：

1. human video 能降低 robot data 门槛，但 10 robot + 200 human 的最终 55% 仍明显低于 50 robot + 200 human 的 95%；机器人动作锚点不可省。
2. 50 robot + 50 human 的终点低于 50 robot baseline，说明“更多 human data 单调增益”在该图上并非无条件成立；需要多 seed 和 matched-training-step 解释非单调性。
3. 所有 human video 都与机器人任务对应，不能外推到 zero-shot 新任务、跨物体或跨场景。

### 10.2 Latent task manifold

AoE 试图让未来状态 latent 学到 task transition：

$$
z_t = f_{\theta}(o_t),
\qquad
\hat z_{t+\Delta}=g_{\theta}(z_t,\text{context}),
\qquad
\hat z_{t+\Delta}\approx f_{\bar\theta}(o_{t+\Delta}).
$$

这个抽象适合跨 embodiment 共享“按压后屏幕/上盖如何运动”之类结构，但论文未披露 $\Delta$、context、target encoder 或 collapse prevention。没有 task-ID/scene/object held-out 结果，也没有 latent probing，因此“学到物理规律与任务逻辑”仍主要由下游 SR 间接支持。

## 11. Reality Gap：三个真实域之间仍有域差

AoE 同时出现 human real、reconstructed simulation 和 robot real，但并未把三者训练成一条可审计闭环：

| 域 | 论文数据 | 主要 gap | 论文桥接 | 未验证项 |
| --- | --- | --- | --- | --- |
| Human real | 胸前 smartphone RGB | 人手/人体形态、手机视点、自然背景 | FLARE future latent；MANO/camera reconstruction | 新手机、新职业、新地区、新物体泛化 |
| Reconstructed sim | AGILE 从单目 AoE HOI 构建数字孪生 | 几何/材质/摩擦/质量/接触误差 | penetration 与 contact stability 检查 | 用重建环境训练的策略是否能回真机 |
| Robot real | Unitree G1 + Inspire hand | robot camera/kinematics/dynamics/control delay | robot teleop action supervision | 相机域对齐、跨机器人、全身动态、碰撞与安全 |

### 11.1 Calibration、手和相机轨迹精度

作者在 EgoDex、Ego4D、10 h in-house RGB-D 和一个未量化的 AR-glasses/phone paired set 上评估 preprocessing。

相机工厂内参相对离线 checkerboard 的偏差均值为 0.64%、标准差 0.21%，radial distortion coefficient 的变化在 $10^{-3}$ 量级。论文称这足以免去逐设备标定，但没有报告参与的手机型号数、镜头档位数、温度、对焦距离或跨时间漂移。

手重建结果：

| Source | PA-MPJPE ↓ | MPJPE ↓ | AUC ↑ |
| --- | ---: | ---: | ---: |
| EgoDex | 7.4 mm | 11.9 mm | 0.90 |
| Ego4D | 3.7 mm | 8.9 mm | 0.93 |

相机轨迹结果：

| Source | ATE ↓ | ATE-S ↓ | RPE-Trans ↓ | RPE-Rot ↓ |
| --- | ---: | ---: | ---: | ---: |
| EgoDex | 1.91 mm | 16.2 mm | 2.94 mm | 0.08° |
| Ego4D | 4.77 mm | 8.3 mm | 1.84 mm | 0.07° |
| Ours（图表合并标作 w/o & w/ depth） | 0.26 mm | 4.4 mm | 0.56 mm | 0.02° |

![手重建、相机轨迹与数值评估](https://arxiv.org/html/2602.23893v2/aoe_data_evaluation.png)

常见指标可写为：

$$
\operatorname{MPJPE}
=
\frac{1}{TJ}\sum_{t=1}^{T}\sum_{j=1}^{J}
\left\|\hat{\mathbf p}_{t,j}-\mathbf p_{t,j}\right\|_2,
$$

$$
\operatorname{ATE}
=
\sqrt{\frac{1}{T}\sum_{t=1}^{T}
\left\|\operatorname{trans}\!\left(
\mathbf T_t^{-1}\mathbf S\hat{\mathbf T}_t
\right)\right\|_2^2},
$$

其中 $\mathbf S$ 表示评估前使用的对齐变换。PA-MPJPE/7-DoF ATE 会吸收旋转、平移和尺度偏差；它们不能单独证明在线无对齐时的轨迹可直接控制机器人。论文也没有给遮挡、快速运动、低纹理、手机型号或动作类别分层误差。

### 11.2 Real-to-sim

作者用 AGILE 从单目 AoE 视频恢复 dynamic hand–object interaction 和 3D assets，报告 mean penetration depth <2 mm，并称 contact stability robust。

![AGILE 将 AoE 手—物交互恢复为 simulation-ready 资产](https://arxiv.org/html/2602.23893v2/exp2_hoi_agile.png)

这个实验确认了“至少部分 AoE 视频能被一个强外部重建框架加工”。它没有报告：

- 评估了多少视频、物体、接触和随机种子；
- penetration 的 reference geometry、时间聚合和 failure percentile；
- contact stability 的公式或数值；
- friction、mass、inertia、joint/articulation 参数如何获得；
- 在这些资产上训练 RL 后的 sim success 和 real success；
- AGILE 失败样本是否被先验过滤。

所以它是**几何可用性证据**，不是完整的 sim-to-real policy evidence。

### 11.3 Reality gap 的剩余风险

- 手机 ultra-wide 的透视和胸前高度不同于机器人头部/腕部相机。
- 人类会以皮肤摩擦、指甲、软组织和高自由度顺应性完成动作，Inspire hand 的接触条件不同。
- robot inpainting 只改变图像，不会自动生成一致的 proprioception、action、force 或 collision state。
- Fold Scarf 的 10% → 10% 表明视觉/任务先验无法弥补控制带宽或柔性体动力学瓶颈。
- Push Bowl & Pour Seeds 只有 20% final success，说明长时双手接触仍是主要 reality/task 联合 gap。

## 12. 人体、相机、场景、物体与接触重建

| 状态 | 方法/输出 | 强项 | 关键缺失 |
| --- | --- | --- | --- |
| Camera | Camera2 intrinsics + MegaSAM trajectory + depth scale | 低成本单目、跨公共 Ego benchmark 有数值 | rolling shutter、IMU fusion、global scale failure、在线延迟 |
| Hands | HaWoR 3D joints + MANO mesh + sliding-window consistency | 细粒度双手几何；MPJPE <12 mm | heavy occlusion、identity/shape calibration、confidence propagation |
| Body | 图中有 body erase/segmentation | 可做视觉 augmentation | 无全身 skeleton、腿/躯干动力学、whole-body contact |
| Scene | dense depth、camera trajectory、scene reconstruction | 为 3D/real-to-sim 提供背景几何 | 无统一 scene graph、metric object map、长期 loop closure |
| Object | VLM object name/bbox；AGILE 可恢复特定 HOI asset | 能提供语义和局部几何候选 | 无普遍 object ID、6D pose、articulation state、material/mass |
| Contact | 从 hand/object proximity 与 AGILE physics 间接推断 | 可挖掘 contact intent | 无真实 contact point/normal/force/tactile/slip |

### Contact intent 与 grasp goal

合理的下游中间表示不应把 MANO joint 直接当机器人动作，而应构造：

$$
g_t =
\left(
{}^{O}\mathbf T_{\mathrm{wrist},t},
\{\mathbf c_{k,t}^{O}\}_{k=1}^{K},
\phi_t,
\Delta \mathbf x_{O,t}
\right),
$$

其中 $\,{}^{O}\mathbf T_{\mathrm{wrist}}$ 是 object-centric wrist pose，$\mathbf c_k^O$ 是候选接触点，$\phi$ 是 approach/contact/manipulate/release phase，$\Delta\mathbf x_O$ 是期望对象变化。再由机器人形态特定的 IK、grasp synthesis 和可行性约束求动作：

$$
\mathbf q_R^\star
=
\arg\min_{\mathbf q_R}
\left[
w_p\|\mathbf f_R(\mathbf q_R)-g_t\|^2
+ w_s\|\mathbf q_R-\mathbf q_{R,t-1}\|^2
\right]
\quad
\text{s.t. joint/collision/contact constraints}.
$$

这是面向复用的建议，不是 AoE 已实现模块。AoE 现有数据产物可为 $g_t$ 提供视觉候选，但缺少真实 contact/force 来校准。

## 13. System 2 / System 1 / System 0 分解

| 层 | AoE 中的对应物 | 时间尺度 | 证据 | 缺口 |
| --- | --- | --- | --- | --- |
| System 2：慢推理/数据加工 | Qwen3-VL 切分与语义、depth/SLAM/MANO、AGILE、生成增强 | 离线 cloud；具体延迟未报告 | 几何指标、系统架构 | 不能在线规划；VLM hallucination 需人工纠正 |
| System 1：学习策略 | GR00T N1.5 + FLARE future latent/action flow co-training | control rate 未报告 | 四任务真机 SR/PSR | obs/action/horizon/latency/config 未披露 |
| System 0：约束与底层控制 | Unitree G1/Inspire hand 的执行栈 | 高频，未报告 | 真机能够执行 | IK/WBC、collision check、impedance、force/tactile、急停均未说明 |

这篇论文在 System 2 数据加工和 System 1 表征共训上有创新组合，但 System 0 几乎是黑箱。作者把 Fold Scarf 失败归因于 hardware latency，恰好说明缺少 System 0 报告会阻断对策略能力的解释。

### 缺少的闭环能力

- 无显式 task planner/subgoal replanning；
- 无失败检测、回退、重抓和 recovery policy；
- 无 uncertainty-aware slowdown；
- 无 contact force/compliance controller 描述；
- 无长时任务 memory 或跨 atomic clip 组合实验。

## 14. 动作表示、时间尺度、训练与 curriculum

### 14.1 已披露训练 recipe

- Foundation policy：GR00T N1.5。
- Cross-embodiment framework：FLARE。
- Human branch：future latent alignment，不需要人类 action label。
- Robot branch：action flow matching。
- Main comparison：50 teleop vs. 50 teleop + 200 AoE，每个任务的人类视频与任务对应。
- Close Laptop 进一步比较 10/50 teleop 与 0/50/200 AoE。
- 最大数据配方下，Close Laptop fine-tune 60,000 steps；Fold Scarf 100,000 steps。
- 作者称训练步数随数据量与任务难度增加以避免 overfitting，但没有给统一 scheduler。

### 14.2 未披露且影响复现的参数

| 类别 | 缺失项 |
| --- | --- |
| Observation | robot camera 数量/位置、图像尺寸、frame stack、proprioception、语言输入 |
| Action | arm/hand action dimension、joint/Cartesian 表示、absolute/delta、chunk length |
| Time | human clip 长度、future horizon $\Delta$、robot control rate、inference latency |
| Optimization | optimizer、learning rate、warmup、batch size、weight decay、gradient clipping |
| Co-training | human:robot batch ratio、loss weights、alternating/joint schedule、feature target |
| Initialization | 精确 GR00T checkpoint、FLARE commit、frozen/trainable module |
| Regularization | augmentation、EMA、dropout、normalization |
| Evaluation | rollout 数、seed 数、reset protocol、human intervention、time limit |

### 14.3 Curriculum 判断

AoE 展示的是一个二维 curriculum：先固定任务，让 robot action data 从 10 增到 50，再让 human video 从 0/50 增到 200。它没有展示：

- 从简单原子技能到长时组合的 task curriculum；
- 从清晰/静态视频到遮挡/动态视频的 quality curriculum；
- 从视觉 latent 到几何/contact supervision 的 representation curriculum；
- 从 simulation 到 real 的 reality curriculum；
- 自动按 uncertainty 或 failure mode 重采样。

最合理的后续课程应先用大规模 accepted AoE clip 做视觉 future dynamics，再用少量有可靠 MANO/object/contact 的 subset 加几何辅助损失，最后以 robot teleop 和真机 failure data 锚定可执行动作。

## 15. 评测、因果证据与 claim 审查

### 15.1 三组实验回答了不同问题

| 实验 | 能回答 | 不能回答 |
| --- | --- | --- |
| Calibration/hand/trajectory | 自动处理在选定 benchmark 上有较小几何误差 | 这些 label 是否改善 robot policy；全球手机分布是否稳健 |
| AGILE real-to-sim | 选定 AoE 视频可转为低穿透的 HOI asset | sim 资产训练是否带来 real policy 增益 |
| GR00T/FLARE real robot | 加同任务 AoE human video 可改善部分任务 | 哪个 preprocessing 模块造成增益；OOD/跨机器人是否有效 |

### 15.2 Robot result 的效应量

| Task | Absolute SR gain | Relative interpretation |
| --- | ---: | --- |
| Pick & Place | +30 pp | 从 0.45 到 0.75，约 1.67× baseline SR |
| Fold Scarf | +0 pp | 无效；证明方法并非普遍提升 |
| Close Laptop | +50 pp | 从 0.45 到 0.95，约 2.11× |
| Push Bowl & Pour Seeds | +20 pp | baseline 为 0，不能用相对倍数；最终仍仅 0.20 |

这些是点估计。论文没有 rollout 分母、置信区间、标准差、seed 或显著性检验。如果每个 SR 恰来自 20 次试验，45% 和 95% 分别可能是 9/20 与 19/20；但这是根据百分比粒度的猜测，不能当作事实。

### 15.3 Claim—Evidence 表

| Claim | 直接证据 | 证据强度 | 审稿式结论 |
| --- | --- | --- | --- |
| 低成本 | mount < \$20 的物料口径 | 中 | 只支持前端增量硬件，不支持 end-to-end 每小时成本 |
| 非侵入/全天 | 胸前图与作者 4/5 星级 | 弱 | 无用户研究、佩戴时长、功耗和中止率 |
| 高质量标注 | MPJPE/ATE + 阈值 + 5% 人审 | 中 | 几何不错，但 segmentation/augmentation/acceptance 未量化 |
| 全球可扩展 | edge-cloud + 数千并发、<100 ms | 中低 | 缺公开 workload、成本和故障数据 |
| real-to-sim 有效 | penetration <2 mm | 中低 | 无样本量、contact metric 和 policy transfer |
| human video 提升 robot | 四任务真机 SR/PSR | 强于其他 claim | 有清晰对照，但缺统计、OOD 和机制消融 |
| 数据具强分布多样性 | 结论文字 | 弱 | 未报告 AoE 数据分布或 diversity metric |
| 自动流水线产生 training-ready assets | 模块图与局部评估 | 中低 | 没有发布资产，也没有端到端质量/人工成本统计 |

### 15.4 三类 Gap 评测矩阵

| Gap | 训练 split 应怎么做 | 论文实际做法 | 还需指标 |
| --- | --- | --- | --- |
| Embodiment | human train，robot anchor；held-out robot/hand/camera | 单一 G1 + Inspire | cross-robot SR、retarget feasibility、action error |
| Task | held-out instruction/object/task/scene | 四个 task-matched human/robot 任务 | OOD SR、phase completion、recovery、compositional generalization |
| Reality | phone/lighting/site/device split；sim→real | real human + real robot；另有独立 sim 重建 | domain classifier、sim/real gap、collision/force/slip |
| 联合 gap | 新任务 × 新对象 × 新机器人 × 新场景 | 未做 | worst-group SR、CVaR、confidence-calibrated failure |

## 16. 数据分布、质量、不确定性与经济性

### 16.1 分布与长尾

论文没有给 AoE corpus 的动作/对象/地点/人员直方图，所以无法判断“anyone, anytime, anywhere”是否已经发生。尤其需要报告：

- contributor 数及每人贡献的 Lorenz/Gini 分布；
- 地区、语言、职业、室内外、光照、手机型号和相机模式；
- verb–object 二维长尾与每个 task 的 effective hours；
- hands visible、object visible、motion blur、occlusion、privacy redaction 比例；
- 端侧触发的 recall：独立连续录像中有多少真实交互被漏掉；
- 原始 → 触发 → 本地保留 → 用户批准 → 云端通过 → training-ready 的漏斗。

### 16.2 质量与 uncertainty

当前流水线有 hard threshold 和 5% manual inspection，却没有把 uncertainty 随数据发布。建议每个 atomic clip 保存：

$$
u_i =
\left[
u_{\mathrm{trigger}},
u_{\mathrm{boundary}},
u_{\mathrm{semantic}},
u_{\mathrm{hand}},
u_{\mathrm{camera}},
u_{\mathrm{privacy}},
u_{\mathrm{contact}}
\right],
$$

并将每项的模型版本、置信区间、mask 和人工复核状态一并纳入 manifest。训练时应按 uncertainty weighting 或 curriculum 使用，不应把 accepted clip 视为同质真值。

### 16.3 数据经济性

| 成本项 | 论文是否量化 | 真实扩展时必须计入 |
| --- | --- | --- |
| Mount | 是，< \$20/user | replacement、shipping、尺寸适配 |
| Smartphone | 视为已有，未计 | 折旧、电池、存储、温升 |
| Contributor time | 未计 | 佩戴、review、edit、upload；supplementary 暗示有 fee settlement |
| Network | 未计 | uplink、跨区复制、失败重传 |
| Cloud compute | 未计 | Qwen3-VL、depth、SLAM、MANO、diffusion、AGILE |
| Storage | 未计 | raw/derived/versioned assets、backup、retention |
| Human QA | 只给 5% inspection | VLM boundary correction、privacy review、appeal |
| Governance | 未计 | consent、DSAR、deletion、incident response、legal review |

因此 Table 1 的 cost 排名适合表示“collector-side hardware barrier”，不适合表示总拥有成本或每个有效训练小时成本。真正的经济性指标应是：

$$
C_{\mathrm{effective\ hour}}
=
\frac{
C_{\mathrm{hardware}}+C_{\mathrm{human}}+C_{\mathrm{network}}
+C_{\mathrm{compute}}+C_{\mathrm{storage}}+C_{\mathrm{governance}}
}{
H_{\mathrm{accepted}}\cdot q_{\mathrm{utility}}
},
$$

其中 $q_{\mathrm{utility}}$ 应由下游增益或质量加权，而不是仅按视频时长。

## 17. 隐私、同意、许可与治理

![AoE App：授权、连续检测、本地编辑、选择上传与统计面板](https://arxiv.org/html/2602.23893v2/aoe-app-workflow.png)

### 17.1 论文声称的保护

- Always-On 需明确 camera authorization，用户可暂停录像。
- raw data 与初始触发/过滤在端侧，本地保存。
- 用户可浏览、剪裁、删除敏感/无效帧，并手工选择上传。
- 上传前自动去标识：正文举例 faces/text，Ethics 举例 faces/screens。
- 录制时持续、不可修改的声音提示旁观者。
- server storage 使用 encryption 与 strict access control。
- 用户可撤回同意、删除账号、请求访问/可移植数据。
- 数据集用途限制为 embodied AI research，明示禁止 surveillance。

### 17.2 Supplementary 隐私摘要同时披露的高风险条款

| 项目 | 披露内容 | 风险 |
| --- | --- | --- |
| 身份 | real name、social-media account with real-name verification | 与第一视角视频形成高可识别绑定 |
| 支付 | payment identity/details、fee settlement | 扩大财务与合规数据面 |
| 证件 | 可能收 age/demographics、academic/driver/vehicle-license 信息 | 数据最小化原则需要更强目的限定 |
| 平台日志 | device、usage、interaction、access/error logs | 可重建行为与位置/时间模式 |
| IP | upload 后 complete intellectual-property-rights transfer；平台有 broad usage rights | 与贡献者后续撤回/再许可的关系不清 |
| Retention | account active 期间保留，另可因 legal/business needs 延长 | 无明确最长周期和 derived model 删除策略 |
| Sharing | payment/technical/business partners、政府/诉讼、business transfer | 第三方和跨境暴露面 |
| Cross-border | 可能在多 jurisdiction 处理 | 未列地区、传输机制和本地化要求 |
| Policy update | continued use constitutes acceptance | 对重大用途变化的重新同意不足 |

### 17.3 关键矛盾与未解决对象

- “用户可随时 withdraw consent”与“上传即完整 IP 转让、业务需要保留”如何同时执行，摘要没有解释。
- 用户可以同意自己的录制，却不能替所有旁观者、客户、家庭成员或屏幕内容权利人同意；声音提示不是 opt-in。
- face/screen blur 不一定去除声音、纹身、工牌、住址、文件、车牌、屏幕反射、独特室内布局和行为身份。
- 端侧模型本身可能看到敏感内容；论文没给模型 telemetry、crash dump 和第三方 SDK 审计。
- 没有报告 IRB/伦理审查编号、未成年人排除、敏感地点禁采、工人—雇主权力关系或 compensation rate。
- 没有公开完整 User Agreement/Privacy Policy、数据处理者清单、DPIA、model unlearning 或 derived asset 删除规则。

### 17.4 发布前最低治理门槛

1. 发布分层 data card、collection geography 与 consent version。
2. 明确 contributor、bystander、property/IP 三类权利和申诉路径。
3. 用可执行许可证分别覆盖 raw、redacted、annotation、derived model，不用论文 CC BY 代替。
4. 提供 per-clip provenance、redaction audit、删除 tombstone 和衍生资产追踪。
5. 对家庭、医疗、学校、浴室/卧室、支付屏幕、商业机密等建立禁采或强制隔离。
6. 第三方 red-team 评估 re-identification、screen leakage 与 member inference。

## 18. 开源审计与分层复现路线

### 18.1 Open-source audit

| 复现要素 | 状态 | 结论 |
| --- | --- | --- |
| Paper/supplement | 开放 | 可复核文字和图表 |
| AoE data | 未发布 | 无法检查统计、隐私、过滤或重新训练 |
| App | 未发布 | 无法验证端侧触发、授权、功耗和 UI |
| Hardware | 只有示意图 | 可做近似，不可复原标准设备 |
| Cloud orchestration | 未发布 | 无 operator schema、HPA/config、storage manifest |
| Preprocessing glue code | 未发布 | 上游组件可找，但版本与坐标/阈值整合未知 |
| Model config/checkpoint | 未发布 | 无法复现几何和真机数字 |
| Evaluation scripts/logs | 未发布 | 无法验证 trial、reset、PSR 和统计 |
| Privacy artifacts | 只有摘要 | 无完整协议、模板、审计工具 |

### 18.2 分层复现

| Level | 目标 | 现有资源 | 可行性 | 退出条件 |
| --- | --- | --- | --- | --- |
| L0 | 论文数字核对 | paper + supp + arXiv figures | 高 | 表格/图/claim 一致 |
| L1A | 采集可用性 pilot | 自购颈挂 + 自写 recorder | 中 | ≥10 用户、≥2 h/人；功耗/舒适/触发 recall 可量化 |
| L1B | 单条 Ego 视频的姿态、物体与接触重建 | 公共视频 + depth/MegaSAM/HaWoR/MANO | 中低 | 坐标、uncertainty、失败帧可视化；不把 proximity 冒充 contact truth |
| L2 | 仿真 retargeting 与物理可行性 | L1B 产物 + AGILE/模拟器 + robot URDF | 低 | penetration/contact stability 定量；IK/collision 可行；仿真任务可完成 |
| L3 | human–robot co-training 与目标域评测 | 自采 matched human + robot teleop + FLARE/GR00T | 低 | 多 seed 复现任务内 SR 增益，并完成 held-out object/scene |
| L4 | 真机 whole-body/dexterous scalable deployment | App/cloud/privacy/data license + System 0 全栈 | 当前不可行 | 外部隐私/安全审计 + TCO + cross-robot/OOD evidence |

### 18.3 最小可复现试验

推荐先做 Close Laptop，而不是同时复制整个云平台：

1. 固定一个机器人、一个相机布局、一个 laptop 和统一 reset。
2. 采 10/50 robot teleop；采 0/50/200 条任务匹配的人类胸前视频。
3. 预注册 2×3 配方、至少 3 个训练 seed、每模型固定 rollout 数。
4. 先只用 redacted RGB + future latent；随后逐项加入 semantic、MANO、camera trajectory。
5. 报告 SR、phase success、time-to-complete、collision、intervention 和 Wilson interval。
6. human 侧用 task/session/person 隔离，robot 侧另换 laptop、lighting 和 start pose 做 OOD。

这个试验能直接验证 AoE 最重要的科研假设，也能揭示云端几何资产是否真的有增量价值。

### 必须补的因果实验

- raw continuous video vs. on-device selected video；
- random clip vs. Qwen atomic clip；
- future latent only vs. +semantic vs. +MANO/trajectory；
- original background vs. diffusion replacement；
- raw human appearance vs. robot inpainting；
- 0/50/200 AoE 与训练 steps 匹配，排除更多 optimization 的混淆；
- task-matched human vs. task-mismatched human，判断是通用 prior 还是示范检索；
- same object/scene vs. held-out object/scene；
- one robot vs. held-out hand/arm/camera；
- 端侧误检/漏检与下游 utility 的闭环相关性。

## 19. 面向人形、EX002 与灵巧手的落地建议

### 19.1 人形机器人

AoE 已在 G1 上给出证据，但 Figure 7 展示的是以双臂/双手为主的站立操作，不是 whole-body locomotion。用于人形平台时：

- 第一阶段只学 upper-body stationary manipulation，固定足底与躯干安全 envelope。
- 将 human future latent 作为视觉/任务先验，robot action 仍由平台 teleop 锚定。
- 对伸手范围、self-collision、桌面碰撞和重心设置独立 System 0 shield。
- 记录 base/torso/arm/hand proprioception 与相机时间同步，避免只看 RGB。
- 对长时双手任务以 phase success 训练 failure detector 和 recovery，而不是只扩大成功示范。

### 19.2 EX002

论文没有 EX002 的实验，且本笔记不假设其具体 DoF、相机或手型。迁移时应先建立平台契约：

| 契约 | 必须确认 | AoE 可提供 |
| --- | --- | --- |
| Observation | camera pose/FOV、RGB rate、proprio、时间戳 | 人类胸前 RGB 与 camera trajectory 可做视觉预训练 |
| Action | joint/Cartesian、chunk、rate、手 synergy | 无直接 EX002 action；需 EX002 teleop |
| Kinematics | reach、joint limit、self-collision | MANO/world wrist 只能作为 goal 候选 |
| Contact | gripper/hand、force/tactile、compliance | 视频可给 contact intent，不能给真实 wrench |
| Safety | speed/force limits、急停、workspace | AoE 未给，需要独立实现 |

建议从 Close Laptop 或 Pick & Place 开始，复制 10/50 robot × 0/50/200 human 矩阵；若 EX002 只有夹爪，不应强行模仿全部手指，而应将 human hand trajectory 压缩为 object-centric approach、grasp、manipulate、release phase。

### 19.3 灵巧手

- 以 MANO 提取 pre-grasp shape 和 wrist approach，不直接回归机器人各指关节。
- 先按对象/抓型学习低维 hand synergy，再用机器人接触/力数据校准。
- 把重建 uncertainty 传给 retargeting；高遮挡帧不做硬监督。
- 加入 tactile/force/slip 与 object pose，建立“看起来像”到“抓得住”的桥。
- Fold Scarf 类柔性体任务优先解决控制频率、触觉和顺应性；增加 RGB 人类视频本身没有解决论文中的 10% 瓶颈。

### 19.4 推荐实验闸门

| Gate | 指标 | 进入下一阶段条件示例 |
| --- | --- | --- |
| Collection | trigger recall、accepted rate、privacy false negative、battery drain | 关键动作 recall ≥95%；敏感内容漏检经审计可接受 |
| Geometry | hand MPJPE、ATE、failure percentile | 不只看均值；P95 与失效分组达标 |
| Offline policy | action NLL/flow loss、future latent retrieval、OOD probing | human data 对 held-out scene/object 有稳定增益 |
| Shadow rollout | collision prediction、constraint violation | 无 action 下发，安全审计通过 |
| Real rollout | SR/PSR、collision、intervention、time | 多 seed/置信区间优于 robot-only |
| Scale | effective-hour cost、deletion SLA、p95 upload/process | TCO 与治理 SLO 同时达标 |

## 20. 局限性、组会问题与最终判断

### 20.1 主要局限

1. 没有发布数据、App、代码、权重、配置或硬件设计。
2. 没有报告 AoE 总规模、参与者、设备、场景和长尾统计。
3. “< \$20”忽略手机、网络、云推理、人审、激励和治理。
4. 自动流水线各产物没有通过端到端 policy 消融归因。
5. 真机只用单一 Unitree G1 + Inspire hand，且 human/robot 任务匹配。
6. 无 held-out person/object/scene/task/robot 评估。
7. 未报告 rollout 次数、seed、置信区间与显著性。
8. Fold Scarf 无增益，长时双手任务绝对成功率低。
9. 无 object 6D/contact/force/tactile 真值；MANO 与相机误差会串联。
10. 隐私摘要包含实名、证件、支付、IP 转让和跨境处理，治理负担显著。
11. 旁观者同意、撤回与衍生数据删除没有闭环。
12. “全天”与“全球并发”缺耐久/负载/成本的可审计 benchmark。

### 20.2 十个组会质疑

1. 真机 FLARE 不用 human action label，那么 MANO、camera trajectory 和 atomic action 到底为哪个 downstream loss 服务？
2. 若只把随机 task-matched 人类视频加入同样训练 steps，是否也能获得相同增益？
3. 为什么 50 teleop + 50 AoE 的 Fully Close 只有 35%，反而低于 50 teleop baseline 的 45%；这一非单调结果在多个 seed 上是否稳定？
4. 每个百分比来自多少 rollout？95% 相对 45% 的置信区间和多 seed 方差是多少？
5. 200 条人类视频来自多少人/地点/手机？相邻 clip 是否高度相关？
6. 端侧 trigger 的 recall 如何测？未被录像的交互如何进入 hard-negative flywheel？
7. 5% manual inspection 是随机、风险分层还是主动学习抽样？每小时人审成本多少？
8. background replacement/robot inpainting 是否保持 contact geometry，是否反而注入不一致视觉伪影？
9. 用户撤回 consent 后，上传即转让的 IP、备份、annotation、checkpoint 和模型参数如何删除或隔离？
10. 从同任务 G1 结果外推“anyone, anytime, anywhere”的 foundation-model scaling，还缺哪一条最关键的 OOD scaling curve？

### 20.3 数据集/系统评分卡

满分 5 分；这里评价论文已提供的证据，不评价未来潜力。

| 维度 | 分数 | 理由 |
| --- | ---: | --- |
| 前端可部署性 | 4.0 | 现有手机 + 低价支架很有吸引力；缺人因/耐久数据 |
| 已证明规模 | 2.0 | 架构声称数千并发，但没有公开 corpus 与负载审计 |
| 有效多样性 | 1.5 | 声称 in-the-wild/scene-agnostic，但没有人、地点、设备、动作或对象分布 |
| Raw observation | 3.0 | RGB/intrinsics/sensor metadata 合理；规格和分布缺失 |
| Human state | 3.5 | MANO/hand/camera 有明确方法与 benchmark；仍是离线估计 |
| World/contact state | 2.0 | 有 depth/scene/AGILE，但缺普遍 object/contact/force 真值 |
| 机器人可执行性 | 2.5 | 必须另采 teleop；human video 不含 executable action，但 FLARE 真机共训有效 |
| 下游机器人证据 | 3.5 | 四任务真机对照强，但无统计/OOD，且一个任务无增益 |
| 标注质量与 uncertainty | 2.5 | 有几何 benchmark、阈值、hard negative、5% 人审；缺全漏斗与语义校准 |
| 开放性 | 1.0 | 只有论文与附录 |
| 隐私治理 | 2.0 | privacy-by-design 元件较多，但权利、旁观者、保留和跨境矛盾大 |
| 总成本透明度 | 1.5 | 只量化 mount |
| 可复现性 | 1.5 | 核心数据/代码/配置/评测日志缺失 |
| 业务价值 | 3.5 | 若共训增益可跨任务复现，能显著降低 robot-data 压力；当前 TCO、许可和规模证据不足 |

### 20.4 三类 gap 与联合对齐评分

| 维度 | 分数 / 5 | 判断 |
| --- | ---: | --- |
| Embodiment gap | 3.0 | latent co-training + robot anchor 合理且有真机结果；未跨机器人/手型 |
| Task gap | 2.5 | 四类任务有跨度；只做 task-matched，长时/柔性仍弱 |
| Reality gap | 3.0 | real human + real robot 优于纯仿真；System 0、接触与 domain OOD 缺失 |
| Semantic–motion alignment | 2.5 | VLM/未来 latent 均有，但接口与语义消融缺失 |
| Motion–action alignment | 3.0 | FLARE/action flow 通过 SR 间接验证；动作空间未披露 |
| Action–contact alignment | 1.5 | 无 force/tactile/contact supervision，折布失败暴露瓶颈 |
| 三 gap 联合泛化 | 2.0 | 没有新任务 × 新物体 × 新机器人 × 新场景测试 |

### 最终判断

**科研判断：有条件通过，值得复现。** AoE 找到了一个很实际的杠杆：不要要求人类采集者操作昂贵机器人或专用 UMI，而用任务匹配的自然第一视角视频为 foundation policy 提供未来状态先验，再以少量机器人数据锚定动作。Close Laptop 的 45% → 95% 是强而直观的真机信号，Fold Scarf 的不增益也诚实暴露了边界。

**数据判断：尚不能称为公开数据集。** 没有总量、分布、下载、许可、schema 或 data card；论文自己的 future work 也承认大规模开放数据集尚待发布。任何“2000 小时”“20× 成本下降”之类后续系统/媒体数字都不属于本论文的已验证数据资产。

**工程判断：适合做任务内小规模 pilot，不适合直接采购为生产基础设施。** 先复制 10/50 robot × 0/50/200 human 的 Close Laptop 实验，并做 preprocessing 逐项消融；若不能证明云端 MANO/trajectory/augmentation 的增量价值，就应保留更简单的 redacted-RGB + future-latent 路径，避免昂贵后端。

**治理判断：在开放招募前必须补齐。** 胸前全天视频的最大风险不是模型性能，而是实名、旁观者、商业机密、跨境、IP 转让、撤回与衍生资产删除。低价支架降低的是硬件门槛，不会降低法律与数据治理门槛。

## 链接索引

- [CVPR 2026 Workshop 官方页面](https://openaccess.thecvf.com/content/CVPR2026W/EmbodiedAIinLife/html/Yang_AoE_Always-on_Egocentric_Human_Video_Collection_for_Embodied_AI_CVPRW_2026_paper.html)
- [CVPR 官方 PDF](https://openaccess.thecvf.com/content/CVPR2026W/EmbodiedAIinLife/papers/Yang_AoE_Always-on_Egocentric_Human_Video_Collection_for_Embodied_AI_CVPRW_2026_paper.pdf)
- [官方 supplementary PDF](https://openaccess.thecvf.com/content/CVPR2026W/EmbodiedAIinLife/supplemental/Yang_AoE_Always-on_Egocentric_CVPRW_2026_supplemental.pdf)
- [arXiv abstract](https://arxiv.org/abs/2602.23893)
- [arXiv HTML v2](https://arxiv.org/html/2602.23893v2)
- [作者 Jin-Chuan Shi 主页中的 AoE 条目](https://chuan-10.github.io/)
- [EgoDex](https://arxiv.org/abs/2505.11709)
- [Android Camera2](https://developer.android.com/training/camera2)
- [AGILE project](https://agile-hoi.github.io/)
