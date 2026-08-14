---
title: "Humanoid Policy ~ Human Policy"
method_name: "HAT"
authors: [Ri-Zhao Qiu, Shiqi Yang, Xuxin Cheng, Chaitanya Chawla, Jialong Li, Tairan He, Ge Yan, David J. Yoon, Ryan Hoque, Lars Paulsen, Ge Yang, Jian Zhang, Sha Yi, Guanya Shi, Xiaolong Wang]
year: 2025
venue: CoRL 2025
tags: [humanoid, egocentric-data, human-robot-cotraining, cross-embodiment, imitation-learning, action-retargeting, dexterous-manipulation, action-chunking, PH2D]
image_source: online
---

# Humanoid Policy ~ Human Policy：把人当作另一种机器人，真能直接训练 Humanoid 吗？

> 本笔记基于 [arXiv:2503.13441v3](https://arxiv.org/abs/2503.13441)、[HTML 全文](https://arxiv.org/html/2503.13441v3)、[CoRL 项目页](https://human-as-robot.github.io/)、[官方代码](https://github.com/RogerQi/human-policy)、[PH2D 数据发布页](https://huggingface.co/datasets/RogerQi/PH2D) 与公开实现精读、交叉核验。资产状态核验日期为 **2026-08-13**。当前 v3 是 2025-10-05 更新的 CoRL 2025 版本，共 19 页；本文 8 幅编号图、8 张表、附录和当前公开代码均已逐项检查。

## 阅读结论先行

这篇论文最有价值的思想可以压缩成一句话：**不要先把每条 human demonstration 离线变成 robot joint trajectory；先把 human 与 humanoid 都表示成头、双腕和十个指尖的几何轨迹，再让同一个 action-chunk policy 学这个“人类中心”的空间，部署时才做 IK 与灵巧手重定向。** 作者把这套策略称为 Human Action Transformer（HAT），配套数据集为 PH2D。

这条路线确实比“Ego 视频只预训练视觉 encoder”更接近直接动作监督。人类和机器人样本进入同一 transformer，状态与动作在概念上都用 54 维几何向量；DINOv2 视觉特征、proprioception encoder、action decoder 共享。论文在 Unitree H1 上的四个真实任务中显示：加入 human data 后，OOD 总成功从 ACT 的 $59/170=34.7\%$ 提高到 HAT 的 $101/170=59.4\%$，即 **+42 次成功、+24.7 percentage points、71.2% relative gain**。背景测试由 55/80 提高到 72/80，位置测试由 22/90 提高到最高 44/90。证据支持 human data 能补充机器人数据未覆盖的背景、物体和工作空间分布。

但标题中的近似号不能理解成“human policy 已经等于 humanoid policy”。HAT 输出的仍是**任务空间几何目标**，可执行性由部署端 wrist IK、finger retargeting、位置控制和真实机器人示范吸收。训练目标没有 contact、force、tactile、balance、collision、joint-limit 或 dynamics loss；实验固定下肢，没有 locomotion，也没有真正的 whole-body policy。人类被要求坐直并尽量不动上身，恰恰说明当前机器人还无法复制完整人体行为。

数据规模也必须分清“论文统计”与“当前发布快照”。论文报告 PH2D 有约 **302 万 human frames、26,824 条 human demonstrations**，配套约 **66.8 万 robot frames、1,552 条 robot demonstrations**。Hugging Face 当前页面公开、ungated、约 16.2 GB，数据卡为 MIT，但文件树分页且一个 HDF5 可能包含多段 demonstration；仅凭浏览器可见文件数不能反推是否逐 episode 完整对应论文统计。发布页没有列出 train/eval split、采集者数量、参与者同意或原始/处理后覆盖关系。

论文还存在几个值得复现者注意的“概念—实现”落差：

- 论文把公共空间写成 54 维；代码实际用 **128 槽 padded vector**，其中 wrist/head 9D（3D translation + 6D rotation）、每手六个 3D keypoint 槽（含固定为原点的 palm + 五个 fingertips），并另留 26 个 robot joint-state 槽。它是实现容器，不应误写成论文把公共动作扩成了 128 个物理自由度。
- 论文公式只展示 L1 action loss 和额外 wrist-translation L1；公开 ACT 实现还包含标准 CVAE KL loss。也就是说，公式是方法重点的简写，不是完整可运行 objective。
- 论文正文说 basic color jitter 与 Gaussian blur 有效；当前 loader 启用了 color jitter，但 Gaussian blur 行被注释。公开仓库更像研究代码快照，而不是论文实验的锁定环境。
- 论文的语言 instruction 已采集却未用于当前 HAT；因此它不是 VLA，也没有验证新语言或新任务组合。

### 一句话总结

HAT 用统一的头—腕—指尖几何动作空间和 50/50 human–robot 采样，把 on-task Ego 人类轨迹直接纳入 humanoid imitation policy；它对同任务 OOD 外观、背景、位置和 H1 系列跨平台 few-shot 有积极真机证据，但接触、动力学、全身平衡、开放任务与完全可复现部署仍没有解决。

### Elevator pitch

机器人示范贵，是因为每次都要遥操作、重定向、保证硬件安全；人类示范便宜，是因为人可以直接完成任务。PH2D 用 Vision Pro、Quest 3 和 ZED 把人的头、手腕、指尖轨迹变成 action label。HAT 不要求相机型号严格一致，也不渲染机器人外观，而是让 human 和 humanoid 共同预测同一种几何 action chunk；部署时再将几何目标变成机器人控制量。它用数据多样性换取视觉稳健性，用统一几何表示换取动作共享，但最后一公里仍依赖机器人运动学和真机数据。

![PH2D 与 HAT 总览：约 2.7 万 human demos 与 1.5 千 humanoid demos 联合训练](https://arxiv.org/html/2503.13441v3/teaser_v6.png)

*图 1。论文主张把 human 直接视为一种 embodiment。这里的“directly”指进入同一 imitation objective，而不是 raw human motion 可直接作为 motor command。*

## 0. 论文、项目、代码、数据与许可

| 资产 | 正式入口 | 截至 2026-08-13 的状态 | 许可/缺口 | 复现判断 |
| --- | --- | --- | --- | --- |
| 论文 | [arXiv](https://arxiv.org/abs/2503.13441)、[HTML v3](https://arxiv.org/html/2503.13441v3)、[CoRL PDF](https://human-as-robot.github.io/resources/corl_final.pdf) | arXiv v3，2025-10-05；19 页；CoRL 2025 | arXiv 标注 CC BY 4.0；不能外推到第三方硬件/SDK | 高 |
| 项目页 | [Humanoid Policy ~ Human Policy](https://human-as-robot.github.io/) | 论文、视频、代码、数据入口齐全 | 没有统一实验 artifact manifest | 高 |
| 代码 | [RogerQi/human-policy](https://github.com/RogerQi/human-policy) | public；审计主干 `2d9d73cc5a3859094ef705f35b8f2faecfc2bc4f`；网页显示 30 commits；支持 ACT、Vanilla DP，RDT trainer 标注“works, but not tested” | MIT；没有论文 checkpoint、正式 release、CI 或完整真机 controller/teleop 脚本 | 中 |
| 数据 | [RogerQi/PH2D](https://huggingface.co/datasets/RogerQi/PH2D) | public、ungated；约 16.2 GB；revision `87b77716f10c04602bdf53de4657795b13fde1c5`；最后更新 2025-04-13；HDF5 + `ph2d_metadata.json` | 数据卡标 MIT；viewer 不支持；split、raw/processed 覆盖、collector 信息不完整 | 中 |
| 模型权重 | 项目页/GitHub/HF | 未发现论文四任务或 Humanoid A/B 的官方 checkpoint | 仓库里的 `empty_lang_embed.pt` 是空语言 embedding，不是 policy 权重 | 低 |
| 自动化验证 | 代码仓库 | 自研 Python 文件静态 `compileall` 通过；未发现专属测试或 GitHub Actions | 依赖、硬件和数据路径较强，不能据静态编译声称端到端可运行 | 中低 |

公开仓库约有 7,916 行 Python，自带 MuJoCo replay/rollout、HDF5 可视化、ZED human collection 和训练命令。README 推荐至少 24 GB 显存，并提供 `chunk_size=100`、batch 64、50,000 epochs/iterations、learning rate $10^{-4}$ 的示例；这些是公开配置的运行入口，不应自动当成论文所有 checkpoint 的精确超参数。仓库 TODO 仍包含 teleoperation scripts、MuJoCo sticky-finger 问题和新 humanoid FK/retargeting 示例，说明 release 对研究复用有帮助，但不足以一键复验论文真机结果。

## 1. 论文速览与可检验假设

| 项目 | 判断 |
| --- | --- |
| 分类 | task-oriented egocentric collection + human–humanoid co-training + geometric action projection/retargeting |
| 数据集 | PH2D：human 约 3.02M frames / 26,824 demos；robot 约 668k frames / 1,552 demos |
| 方法 | HAT：frozen DINOv2 ViT-S + ACT/CVAE action-chunk transformer；统一 head/wrist/fingertip state-action |
| 目标平台 | Unitree H1（Humanoid A）与 H1-2（Humanoid B），6-DoF Inspire dexterous hands、2-DoF neck |
| 任务 | cup passing、horizontal grasping、vertical grasping/picking、pouring；每任务单独 checkpoint |
| 最强证据 | Humanoid A 的 OOD aggregate 59/170 → 101/170；不同背景、物体位置/外观上普遍改善 |
| 最弱 claim | “nearly 100% relative improvement on all tasks”和 human data“显著”更高 sampling efficiency，均缺置信区间/多 seed，并有指标口径夸张风险 |
| 精读优先级 | 5/5：Ego co-training 到 humanoid action 的代表性早期范式 |
| 复现优先级 | 3/5：代码和处理数据公开，但权重、真机接口、锁定实验配置和完整统计不足 |

核心假设可以严格写成：

> 在目标任务、目标 H1 平台、action-chunk architecture 和少量 robot demonstration 基本固定时，将更多样的 on-task human Ego demonstrations 映射到相同的头—腕—指尖几何空间，并与 robot sample 联合训练，能减少 policy 对有限 robot scene distribution 的过拟合，改善同技能的 OOD 表现。

论文支持的是这个条件命题。它没有证明 random/off-task human video 同样有效，没有做等 episode、等 frame/token 或等训练计算量比较，也没有证明只用 human data 就能安全部署。

## 2. Gap—Evidence 总表

| Gap | 作者机制 | 直接证据 | 关键消融 | 仍未解决 |
| --- | --- | --- | --- | --- |
| Embodiment | 54D human-centric state/action；robot FK；部署 IK + hand retargeting；分 embodiment normalization | H1 上主结果；H1→H1-2 few-shot | unified state、slowdown、normalization、whole-body movement | dynamics、joint limit、collision、不同手 topology；只测 H1 家族 |
| Visual | 多相机、多环境 PH2D；frozen DINOv2；color jitter/blur | background、object appearance、placement OOD | 无 DINO/augmentation/数据多样性正交消融 | human/robot 外观仍不同；收益可能是视觉覆盖而非动作共享 |
| Temporal | human action 插值减速，固定 $\alpha_{slow}=4$ | state/speed ablation 4/10 vs 1/10 或 0/10 | 有，但仅一个 10-trial cell | task-dependent timing、接触 phase、闭环频率未建模 |
| Task | human 与 robot 做重叠任务；每任务单独 checkpoint | 四项已知技能的 ID/OOD | 无 language、new-task、off-task human 消融 | 无 skill composition、开放词汇或新任务迁移 |
| Reality | 真机 robot demos；几何目标经 IK/retargeting | 两台真实 H1 系列机器人 | 无 controller/dynamics/latency 消融 | balance、force、compliance、安全、payload、sim-to-real 均未建模 |
| Motion–contact | wrist pose + fingertips；robot hand retargeting | grasp/pour/pass 成功作为间接指标 | 无 contact/force/tactile/grasp-mode 消融 | 指尖位置不等于接触状态；滑移和物体动力学未知 |

## 3. PH2D：采集了什么，发布了什么

![消费级采集设备：Vision Pro 内置相机，或 Quest 3/Vision Pro 搭配 ZED Mini](https://arxiv.org/html/2503.13441v3/hardware_view_v4.png)

*图 2。ZED 方案声称硬件成本低于 700 美元；低门槛主要指 sensor rig，不含标注、处理、训练与机器人评测成本。*

### 3.1 数据规模与统计单位

| 域 | Frames | Demonstrations | 单 demo 平均 frames | 主要来源 |
| --- | ---: | ---: | ---: | --- |
| Human | 约 3,020,000 | 26,824 | 约 112.6 | Vision Pro / Quest 3 + ZED；人直接执行任务 |
| Humanoid | 约 668,000 | 1,552 | 约 430.4 | VR teleoperation；Unitree H1 系列 |

平均 frame/demo 只用于理解数量级，不能直接解释为统一长度：任务时长、采样频率、处理和一个 HDF5 内的 execution 数可能不同。当前数据卡还明确说“多数 HDF5 是一次 execution，但 cup passing 等文件可能包含多段 demonstrations”，所以文件数不是 demo 数。

论文的 PH2D 表列出 pouring、grasping、stacking、picking、cup passing、sorting cans 六类。公开 `ph2d_metadata.json` 也列这六类，并记录 `h1_inspire`、`h1_2_inspire_cmu`、`human_avp`、`human_zed` 以及若干 simulation embodiment 和相机 ID。数据发布因而不只是论文主表四个真机任务的最小包；但论文没有给出每类、每设备、每操作者、每 scene 的完整数量分布。

### 3.2 采集硬件、同步与监督

| 组件 | Observation / label | 对齐方式 | 误差与缺口 |
| --- | --- | --- | --- |
| Apple Vision Pro | built-in ego camera；ARKit world-frame head 与 hand keypoints | 最近时间戳同步视觉/proprioception | SDK 遇重遮挡可能丢 keypoint；版本和置信度阈值未报告 |
| Quest 3 / Vision Pro + ZED Mini | OpenTeleVision head/hand pose；stereo ego RGB | web app + 最近时间戳；ZED human collector 代码为 30 FPS | 外置相机与 headset pose 外参、漂移和 sync tolerance 未量化 |
| Human instruction | 每次 demo 有自然语言，如“右手抓 Coke Zero” | 写入 episode metadata | 当前论文模型不使用，语言准确性/词表/标注协议未评估 |
| Humanoid A/B | ego stereo camera；joint/kinematic state；teleop action | robot FK 变为统一空间 | 控制频率、真实传感延迟、丢帧率与 tracking error 未系统报告 |
| Object/contact | 图像中隐式存在 | 无显式对齐 | 无 object pose、mesh、contact point、force、tactile、success label 体系说明 |

### 3.3 Human 与 Humanoid 数据是否真的“统一”

| 维度 | Human | Humanoid | 统一到哪里 | 剩余 gap |
| --- | --- | --- | --- | --- |
| RGB | 多种 headset/相机、真实人手 | robot head stereo、机器人手臂 | frozen DINOv2 feature + augmentation | FOV、tone、安装、body/hand appearance 不同 |
| Head | 6D rotation；实现还留 3D translation 槽 | 2-DoF actuated neck 经 FK | 6D rotation representation | robot neck 可达空间远小于 human head/torso |
| Wrist | 3D position + 6D rotation/腕 | joints 经 FK 得 wrist pose | 每腕 9D | IK 多解、joint limit、workspace 不同 |
| Fingers | 十个 fingertips | Inspire 五指对应 fingertips | 一一 keypoint mapping | 骨长、关节数、指腹接触与顺应性不一致 |
| 时间 | 人约快 4 倍 | teleop 较慢 | human chunk 插值减速 | 固定倍率无法保持所有 contact phase |
| 统计 | human distribution | robot distribution | 可选 shared 或 embodiment-specific Z-score | normalization 既对齐又可能保留 domain identity |
| 执行 | 人体自然运动 | robot motor/joint commands | 几何 prediction 后 IK/retargeting | 只有 robot 端真正可执行 |

### 3.4 采集效率：墙钟优势真实，但证据有限

论文附录把 scene reset 计入单次采集时间：

| 方式 | Grasping | Pouring | 相对 humanoid teleop speedup |
| --- | ---: | ---: | ---: |
| Human，不戴 VR | $3.79\pm0.27$ s | $4.81\pm0.35$ s | — |
| Human + VR | $4.09\pm0.30$ s | $4.90\pm0.26$ s | grasp 4.82×；pour 7.61× |
| Humanoid VR teleop | $19.72\pm1.65$ s | $37.31\pm6.25$ s | baseline |

VR 对 human 完成速度的影响很小，而 teleop 慢得多，这是可信的 system-level 事实。但表中没有操作者数、trial 数、学习曲线、失败/重录比例或硬件故障时间。它测到的是两个短任务的**每条完成演示墙钟时间**，不是每 frame 信息量、每成功 policy improvement 或标注后总成本。

### 3.5 数据治理与隐私

PH2D 的 human videos 包含第一视角真实环境；项目示例涉及办公室、零售/餐饮和室外活动。论文与当前数据卡没有报告：

- human collector 人数、人口统计、惯用手、经验水平与跨人 split；
- IRB/伦理审查、知情同意、报酬、撤回机制和旁观者处理；
- 人脸、屏幕、住址、音频等潜在敏感内容是否被采集或去标识；
- raw RGB/pose 的保存期限，以及第三方 ARKit/OpenTeleVision/SDK 数据流；
- train/validation/test 的身份和地点泄漏防护。

MIT dataset card 说明发布者给出的使用许可，不自动消除影像中人物、场所、商标或隐私权问题。面向外部部署或再发布时，仍应做逐帧隐私审查、participant-level split 和 provenance/consent 记录。

## 4. HAT 全链路：从 human Ego 到 robot command

![HAT 架构：两域状态经统一几何表示，图像经冻结 DINOv2，输出再重定向](https://arxiv.org/html/2503.13441v3/overview_v3.png)

*图 3。共享发生在 state-action geometry 与 policy parameter 层；IK/hand retargeting 位于 robot 数据预处理或部署边界。*

~~~mermaid
flowchart LR
    A["Human headset\nRGB + head/wrist/finger pose"] -->|"tracking occlusion / sync"| C["Human-centric state-action\nhead + 2 wrists + 10 fingertips"]
    B["Humanoid teleop\nRGB + robot joints"] -->|"FK / calibration error"| C
    A -->|"human speed × 1/4"| D["Temporal interpolation"]
    D --> C
    C -->|"embodiment-specific stats by default"| E["Normalization"]
    A --> F["Frozen DINOv2 ViT-S"]
    B --> F
    E --> G["HAT / ACT-CVAE transformer"]
    F --> G
    G --> H["Future head-wrist-fingertip chunk"]
    H -->|"wrist IK"| I["Arm joint targets"]
    H -->|"finger retargeting"| J["Inspire-hand targets"]
    I --> K["Real H1 execution"]
    J --> K
    K -. "No force/contact/balance loss" .-> L["Task success or failure"]
~~~

### 信息保留、主动丢弃与从未观测

- 保留：ego appearance、head/wrist pose、十个 fingertips、动作时序、task completion sequence、多环境和多对象视觉覆盖。
- 映射：robot joint state/action 经 FK 转成几何空间；human trajectory 被固定 4× 插值减速；部署 action 再经 IK/retargeting。
- 主动弱化：robot-specific joint topology 不作为 HAT 的核心 input/output；不同设备视觉主要靠 frozen feature 与 augmentation 吸收。
- 从未观测：contact point/normal、object 6D state、force/torque、tactile、friction、payload、whole-body center of mass、balance margin、collision geometry。
- 可能泄漏：每任务单 checkpoint、背景/对象与指令高度相关；模型可能学 scene shortcut，而非可组合 task semantics。

## 5. “投影”到底是什么：54D 几何空间与可执行性

### 5.1 概念空间的维度

论文定义的 proprioception 与 action 同构。用连续 6D rotation representation 表示三个 orientation：

$$
d = 3\times 6\;\text{(head, left wrist, right wrist rotations)}
  + 2\times 3\;\text{(wrist translations)}
  + 10\times 3\;\text{(fingertips)}
  = 54.
$$

它是**任务空间坐标**，不是 54 个独立机器人关节。五指 humanoid 可以让十个 human/robot fingertips 一一对应；论文也指出平行夹爪可把拇指—其他指距离注入到夹爪开度，但没有实验验证这种非双射 topology。

对 robot demonstration，joint configuration $q$ 先经 forward kinematics：

$$
s^{geo}_t = FK(q_t), \qquad a^{geo}_{t:t+H}=FK(q_{t+1:t+H}).
$$

对 human demonstration，headset SDK 直接给出近似相同的几何量。部署时策略输出 $\hat a^{geo}$，再求：

$$
q^{arm}_{t+1:t+H}=IK(\hat T^{wrist}_{t+1:t+H}), \qquad
q^{hand}_{t+1:t+H}=R(\hat p^{finger}_{t+1:t+H}).
$$

其中 $R$ 是 hand retargeting。论文称该转换可微，但展示的训练 loss 没有通过部署 IK/retargeting 回传的 feasibility、collision 或 contact 项；公开 pipeline 的主要用法也是预测后转换。因此更稳妥的分类是 **shared geometric action projection + downstream retargeting**，而不是 end-to-end differentiable robot dynamics optimization。

### 5.2 公开实现为何是 128 维

当前代码把 action/state 放进 `ACTION_STATE_VEC_SIZE = 128` 的容器：

| 槽 | 实际内容 | 与论文关系 |
| --- | --- | --- |
| head | 9 槽：3D position + 6D rotation | 论文 54D 只明确计 head rotation；position 槽可为固定/零基准 |
| 每个 wrist | 9 槽 | 对应 3D translation + 6D rotation |
| 每只 hand | 18 槽：palm origin + 五 fingertip，各 3D | 有效 fingertip 与论文一致；palm origin 被注释为恒零 |
| robot qpos | 26 槽，放在索引 100–125 | robot-specific 输入/处理辅助；不等于共享 54D action |
| 其余 | padding/预留 | 无物理自由度含义 |

`OUTPUT_INDEX` 实际选择 head、双腕及两组六 keypoint 槽。代码版本与论文符号之间的差异提示复现者：必须固定 preprocessing commit 和 HDF5 schema，不能只按 54D 描述自行猜 index。

### 5.3 Temporal projection：固定四倍减速

设 human action 序列为 $a^h_{1:T}$，实现把 translation 与 rotation 插值到约 $4T$，再截取 policy chunk：

$$
\tilde a^h_{1:4T}=\operatorname{Interp}(a^h_{1:T},4T), \qquad \alpha_{slow}=4.
$$

固定倍率来自 human/humanoid 平均任务时长比“经验上约为 4”。它解决速度量级差，但不会自动对齐 grasp onset、接触持续、pouring angle 或 release 时刻。若某 task phase 的 human/robot 时长比例不同，线性插值可能复制错误 contact timing。

## 6. 模型、采样、归一化与损失

### 6.1 HAT 架构

- 两路 ego image（公开配置为 `left`, `right`）先 resize 到约 $224\times308$，再经 frozen DINOv2 ViT-S。
- proprioceptive state 经统一几何 state encoder；ACT baseline 则直接用 robot joint positions。
- transformer/CVAE 预测 future action chunk；论文没有报告统一的参数量、训练 wall-clock、GPU-hours 或多 seed。
- 每个 task 独立训练 checkpoint，约用 250–400 条 robot demonstrations，并混入对应 human tasks。
- 语言 annotation 没用于主实验；公开配置中 `use_language_conditioning: false`，代码保留 empty language embedding 和可选接口。

公开 ACT 配置可见 4 encoder layers、7 decoder layers、8 attention heads、hidden dimension 512、feed-forward dimension 3200、KL weight 10、DINOv2 ViT-S/14 frozen。应把这些记录为 release snapshot，而非在没有 experiment manifest 的情况下断言每张论文表都严格用同一配置。

### 6.2 两域如何被采样

当前 loader 不是按 26,824 vs 1,552 的 raw count 比例抽样，而是：

$$
P(e=\text{human})=0.5,\qquad P(e=\text{robot})=0.5.
$$

在默认 `norm_by_embodiment_and_task` sampler 中，每个 embodiment 内又先均匀分配给 task group，再在 task 的 episodes 间均匀分配。这阻止 massive human data 完全淹没 robot signal，也意味着“加入 17× 更多 human demos”不等同于每个 batch 有 17× human tokens。实际 optimization exposure 由 sampler 和训练步数决定。

### 6.3 Normalization 并没有单一答案

默认代码为每种 embodiment 分别计算：

$$
\bar a^{(e)}=\frac{1}{N_e}\sum_i a_i^{(e)}, \qquad
\hat a_i^{(e)}=\frac{a_i^{(e)}-\bar a^{(e)}}{\sigma_a^{(e)}+10^{-6}},
$$

state 同理，标准差下限为 $10^{-2}$。正文 Table 2 中 different normalization 的 OOD aggregate 为 101/170，高于 shared/no-different 的 97/170；但附录 Fig. 8 的特定 3×3 placement 测试反而是 same norm 44/90、different norm 42/90。作者解释 same norm 可能因 human workspace 包含 robot workspace 而产生空间偏置。

因此结论不是“separate normalization 必然更好”，而是两种选择会改变 spatial coverage，aggregate 差异很小且依 task/grid 而变。它也是一个潜在 domain indicator：各域使用不同统计量，会让相同归一化值代表不同物理尺度。

### 6.4 论文 objective 与代码 objective

论文展示：

$$
\mathcal L_{paper}
=\ell_1(\pi(s_i),a_i)
+\lambda\,\ell_1(\pi(s_i)_{EEF},a_{i,EEF}),
\qquad \lambda=2,
$$

其中 $EEF$ 只选左右 wrist translation。额外项强调手到物体的空间定位，不让大量 fingertip 维度稀释 wrist error。

公开 ACT/CVAE 实现实际为：

$$
\mathcal L_{code}
=\mathcal L_{L1}
+\beta\mathcal L_{KL}
+2\mathcal L_{EEF},
$$

默认配置可见 $\beta=10$。这不是方法矛盾，而是正文省略了 ACT 原有 CVAE 正则；但复现时必须用完整实现。另一个 release drift 是 loader 中 Gaussian blur 被注释，只实际启用强度 0.4 的 color jitter，而正文把二者都列为有效 basic augmentations。

## 7. 两台 Humanoid：异构到什么程度？

![Humanoid A（Unitree H1）与 Humanoid B（H1-2），均装配 Inspire dexterous hands](https://arxiv.org/html/2503.13441v3/humanoid_illustration_v3.png)

*图 4。两者在不同地点采集，且腕部自由度和 arm ROM 不同；但仍属于 Unitree H1 家族，并共享五指 Inspire hand，不能外推到任意 humanoid。*

| Joint ROM | Humanoid A | Humanoid B |
| --- | ---: | ---: |
| shoulder pitch | $-164^\circ$ to $+164^\circ$ | $-180^\circ$ to $+90^\circ$ |
| shoulder roll | $-19^\circ$ to $+178^\circ$ | $-21^\circ$ to $+194^\circ$ |
| shoulder yaw | $-74^\circ$ to $+255^\circ$ | $-152^\circ$ to $+172^\circ$ |
| elbow | $-71^\circ$ to $+150^\circ$ | $-54^\circ$ to $+182^\circ$ |
| wrist roll | $-175^\circ$ to $+175^\circ$ | $-172^\circ$ to $+157^\circ$ |

Humanoid A 只有一个 distal wrist roll，Humanoid B 有 wrist pitch/roll/yaw 三个独立自由度；B 还有不同 motor、absolute encoder 与不同工作空间。统一 wrist pose 能隐藏一部分 joint topology，但 IK 的可达性和 manipulability 仍依赖平台。few-shot 结果同时混合了 morphology shift 与不同地点/environment shift，是更现实但不完全可归因的测试。

## 8. 主结果：Human data 改善了什么？

### 8.1 四任务 protocol

![Cup passing：背景和传递方向变化](https://arxiv.org/html/2503.13441v3/passing.png)

![Horizontal grasping：不同外观物体抓取并放入容器](https://arxiv.org/html/2503.13441v3/grasping.png)

![Vertical grasping：3×3 位置网格抓取](https://arxiv.org/html/2503.13441v3/picking.png)

![Pouring：双手拿瓶/杯、倾倒并放回](https://arxiv.org/html/2503.13441v3/pouring.png)

*图 7a–d。四项任务都是真机、多阶段 dexterous manipulation，但下肢固定，任务由单独 checkpoint 指定。*

| 任务 | ID | OOD 主要变量 | 成功判定边界 |
| --- | --- | --- | --- |
| Passing | robot data 的 paper background | wooden/red/green backgrounds、方向 | 杯成功传递；Table 7 caption 对“失败/重试”表述略含混 |
| Horizontal grasp | robot-seen bottle | human-seen box 1、unseen box 2/can | 抓起并放到容器；滑移会失败 |
| Vertical grasp | robot 采集两个不平衡 grid cells | 其余 10 cm × 10 cm cells | box 抓起/放入 bin |
| Pouring | robot-seen pose/object | robot rotation、table pose 等 | 抓瓶/杯、倒液、放回；没有 force/liquid-state 指标 |

### 8.2 Table 2 完整结果

| Method | Passing ID / OOD | H. grasp ID / OOD | V. grasp ID / OOD | Pour ID / OOD | Overall ID / OOD |
| --- | ---: | ---: | ---: | ---: | ---: |
| ACT，robot only | 19/20 · 36/60 | 8/10 · 7/30 | 7/20 · 15/70 | 8/10 · 1/10 | 42/60 · 59/170 |
| HAT，human，shared norm | 17/20 · 51/60 | 9/10 · 11/30 | 14/20 · 30/70 | 5/10 · 5/10 | 45/60 · 97/170 |
| HAT，human，different norm | 20/20 · 52/60 | 8/10 · 12/30 | 13/20 · 29/70 | 8/10 · 8/10 | **49/60 · 101/170** |

最稳健的解读：

- ID：ACT 70.0%，最佳 HAT 81.7%，+11.7 pp；但分任务有涨有跌，shared-norm HAT 的 pouring 是 5/10，低于 ACT 8/10。
- OOD：ACT 34.7%，最佳 HAT 59.4%，+24.7 pp；relative gain 71.2%，并不是 overall “nearly 100%”。
- 分任务 OOD：passing +44.4% relative、horizontal grasp +71.4%、vertical grasp +93.3%、pouring +700%，最后一项基线仅 1/10，relative ratio 很容易显得巨大。
- 论文没有 confidence interval、independent policy seeds 或 statistical test。230 次左右的二元试验能说明趋势，却不足以精确比较相差几个 success 的版本。

### 8.3 背景、外观与位置的拆解

背景泛化：

| Method | Paper ID | Wooden human-seen | Red OOD | Green OOD | Overall |
| --- | ---: | ---: | ---: | ---: | ---: |
| ACT | 19/20 | 14/20 | 12/20 | 10/20 | 55/80 |
| HAT | 20/20 | 16/20 | 18/20 | 18/20 | 72/80 |

HAT 增加 17/80，即 +21.25 pp、30.9% relative。正文“nearly 50%”若按 overall success count 并不成立；可能来自某个 OOD 子集或失败数口径，不宜照抄。

Object appearance：

| Method | Bottle ID | Box 1 human-seen | Box 2 unseen | Can unseen | Overall |
| --- | ---: | ---: | ---: | ---: | ---: |
| ACT | 8/10 | 5/10 | 1/10 | 1/10 | 16/40 |
| HAT | 8/10 | 7/10 | 1/10 | 4/10 | 21/40 |

提升主要来自 box 1 和 can；完全 unseen 的 box 2 没有任何成功率改善。作者观察 HAT 会更主动搜索 graspable region，但这是 qualitative explanation，不能把 1/10 读成已获得可靠 category-level grasp prior。

![Vertical grasp placement：ACT、different normalization HAT 与 same normalization HAT](https://arxiv.org/html/2503.13441v3/position_fig.png)

*图 8。总成功数为 ACT 22/90、different norm 42/90、same norm 44/90。robot data 只在虚线两个 cell 采集且 50:10 不平衡，HAT 的收益包含 human spatial coverage。*

位置结果几乎翻倍，但它回答的是“human data 覆盖 robot 未覆盖位置是否有帮助”。OOD cells 对 robot data 未见，对 human data 并非未见；因此不是对整体训练分布的 zero-shot spatial generalization。

## 9. Few-shot H1-2、采样效率与消融

![Humanoid B 20-shot：B-only、A+B、A+B+human 在已见与未见物体上的对比](https://arxiv.org/html/2503.13441v3/objects_data_cmu_new.png)

![Humanoid B 数据量从 0/5/10/20/50 增加时，co-training 与 isolated training 的趋势](https://arxiv.org/html/2503.13441v3/co-training_cmu_new.png)

*图 5a–b。曲线/柱的精确数值未以表格发布；图读只能支持趋势，不宜把像素高度当作精确 success rate。*

20 条 Humanoid B demos 时，B-only 明显低于 A+B，而加入 human 后通常进一步改善，包括 human-seen、B-unseen object；完全未见 object 的收益较小。随 B demonstrations 从 0 增至 50，co-training 曲线始终高于 isolated training，低数据区差距最大。这个实验证明共享 geometric representation 可以给新 H1 变体提供 prior，但有三项混杂：A/human data 量远大于 B、B 位于另一环境、两台机器人仍共享 H1 family 和 Inspire hands。

### 9.1 等墙钟 20 分钟的采样效率

![20 分钟采集预算：60 条 robot-only 对比 30 robot + 120 human](https://arxiv.org/html/2503.13441v3/sampling_efficiency.png)

*图 6。两组都只在底部六格均匀采集；robot-only 为 28/90，mixed 为 35/90。*

mixed 数据把 success 从 31.1% 提高到 38.9%，是 **+7/90、+7.8 pp、25% relative**。这支持“相同短期墙钟预算下，用一半 robot time 加 human data 更有效”，但“significantly”没有 statistical test。两组的 episode 数分别 60 与 150、总 frame/token 和训练 exposure 未配平，所以不能推出 human episode 的单位样本价值更高。

### 9.2 Unified state 与 slowdown 必须同时存在

| Unified state-action | Human slowdown | Upper-left cell success |
| --- | --- | ---: |
| 是 | 否 | 1/10 |
| 否，robot 用 joint state | 是 | 0/10 |
| 是 | 是 | 4/10 |

作者观察：不减速会让预测在 human-fast 与 robot-slow 间波动；不统一 state 会让网络通过 state schema 识别 embodiment，形成 shortcut。这个消融方向正确，但每格仅 10 trials，且没有 robot-only、human-only、不同 $\alpha$ 或多随机 seed，因果强度有限。

### 9.3 Whole-body human movement 反而降低总成功

| Human data | Bottle ID | Box 1 | Box 2 | Can | Overall |
| --- | ---: | ---: | ---: | ---: | ---: |
| Minimize whole-body | 8/10 | 6/10 | 0/10 | 7/10 | 21/40 |
| Allow whole-body | 9/10 | 3/10 | 3/10 | 3/10 | 18/40 |

总数下降 3 次，但 bottle/box 2 又提高，样本很小。这更像“当前 fixed-base robot 无法复制 human torso compensation，whole-body motion 可能造成 domain gap”的初步证据，而不是普适地证明 whole-body data 有害。真正的 humanoid loco-manipulation 应建模 base/torso/head/arms 的耦合，而不是永久压掉身体运动。

## 10. Embodiment、Task、Reality、Contact 的边界

### 10.1 Embodiment gap：解决到几何层，未解决到动力学层

| 层 | HAT 做法 | 判断 |
| --- | --- | --- |
| Observation | 多设备数据 + DINOv2 + basic augmentation | 减少 sensor/appearance overfit，但无显式 domain-invariance 证明 |
| State/action semantics | head/wrist/fingertip geometric representation | 核心贡献；比 joint-space co-training 更可共享 |
| Kinematic feasibility | FK preprocessing；deployment IK/retargeting | 部分解决；无可达性 loss、success-conditioned IK 或 collision constraint |
| Hand topology | 十指 bijection 到 Inspire hands | 绑定五指平台；平行夹爪只被提出、未验证 |
| Dynamics | 无质量、惯量、torque、latency、compliance model | 未解决 |
| Whole body | human 被要求坐直；robot 下肢固定 | 没有 locomotion/balance，不能称 whole-body humanoid policy |

### 10.2 Task gap：四个 checkpoint，不是通用 VLA

模型虽然每条数据有 language instruction，却不消费语言；每任务单 checkpoint 由外部选择。因此：

- 验证了已知 task 的新背景、位置、外观；
- 没验证同一 policy 的 task selection 或 instruction following；
- 没验证 human-only 新 task 能否 zero/few-shot 落到 robot；
- ACT 的 latent/action chunk 不是显式、可组合的 skill manifold；
- 没有 planner、subgoal、object-centric world model 或 long-horizon recovery。

### 10.3 Reality gap：真机数据绕开 sim-to-real，但没有消除现实问题

HAT 全部主实验在真机上，这是优点。它不需要 synthetic-to-real appearance transfer，却仍面对：

- IK 多解、奇异点、joint limits 和 self/environment collision；
- 位置控制的 tracking delay、motor saturation、gear backlash；
- object mass、friction、liquid dynamics 和 grasp compliance；
- perception latency、camera extrinsic drift 与 hand keypoint error；
- hardware safety、fall prevention 和 failure recovery。

论文只通过 task success 间接覆盖这些问题，没有单独测 control Hz、latency、tracking RMSE、collision/fall、force 或 intervention rate。公开 MuJoCo 代码设置 70 Hz 也不能替代真机频率报告。

### 10.4 Motion–contact gap：finger position 不是 contact

十个 fingertip 比单一 wrist pose 更能表达 hand shape，但仍缺少：

| Contact variable | 是否监督 | 风险 |
| --- | --- | --- |
| fingertip location | 是，tracking/FK estimate | 几何接近不等于实际接触 |
| contact binary / phase | 否 | grasp onset/release timing 只能隐式学 |
| contact normal / patch | 否 | 不能表达推、托、捏的方向约束 |
| force/torque/tactile | 否 | 无法判断滑移、过力或稳定 grasp |
| object pose/velocity | 否 | pouring、slip 与撞击无法显式闭环 |
| friction/material | 否 | novel object 的力学泛化无证据 |

Table 6 的 box 2 正好暴露这个边界：模型会找到物体，却因低矮、与桌面颜色接近和抓取不稳定而仍为 1/10。视觉/运动 prior 改善不自动变成接触能力。

## 11. System2 / System1 / System0 接口

| 层 | 当前 HAT 中对应物 | 输入 | 输出 | 频率/证据 | Gap 责任 |
| --- | --- | --- | --- | --- | --- |
| System2 | 不存在；外部选择 task checkpoint | task name 未进入 policy | 无 language/subgoal | 无 | Task gap 未建模 |
| System1 | HAT/ACT-CVAE | ego stereo + geometric proprioception | future head/wrist/fingertip chunk | action chunk；真实 Hz 未报告 | 视觉 OOD、部分 embodiment |
| Projection | wrist IK + finger retargeting | geometric chunk | platform joint/hand target | 实现/论文均有；训练可微闭环不清楚 | kinematic embodiment |
| System0 | robot position/controller stack | joint targets/current state | motor commands | 论文未量化 | tracking、safety、contact |
| Whole-body drive | 固定下肢 + neck/arms/hands | 上身命令 | hardware motion | 无 balance/locomotion | Reality/whole-body 未解决 |

System1 给 System0 的接口仍是 pose/joint target，不包含 contact intent、desired wrench、compliance、collision constraint 或 recovery condition。对低速 tabletop 操作足够；对插接、工具、开门和移动操作，需要显式 contact-aware controller 或闭环 tactile/force policy。

## 12. 因果证据审计：提升究竟来自哪里？

### 支持作者机制的证据

1. HAT 和 ACT 主体架构近似，主要改变 unified state/action 并加入 human data；OOD aggregate 有较大差距。
2. 不统一 state 或不减速都会在目标 OOD cell 显著下降，说明表示与时序不是无关细节。
3. human data 中出现的背景、位置和物体分布能迁移到 robot execution；方向与实验设计一致。
4. Humanoid B few-shot 中 A/human co-training 在少量 B data 下优于 B-only，说明收益不只局限 Humanoid A checkpoint。

### 仍然存在的替代解释

1. **数据覆盖而非 human motor prior**：OOD 条件在 human data 中出现、在 robot data 中未出现，提升可能主要来自 frozen DINO 的视觉 exposure。没有“相同 human RGB、打乱/移除 action label”的对照。
2. **总训练样本/计算更多**：没有等 update、等 frame/token、等 image exposure 与 robot-only oversampling 对照。
3. **task-specific shortcut**：每任务独立训练，背景、对象、动作互相高度相关；没有语义反事实或跨任务测试。
4. **H1 family similarity**：Humanoid A/B 有明显 arm 差异，但同属 H1 系列且都用 Inspire hands；不能证明跨不同 humanoid/hand family。
5. **评测方差**：大多数 cell 只有 10 或 20 trials，没有 seed、CI 或 blinded scoring；1–3 次成功差不能稳定排序。

最关键的下一组实验应是 2×2 factorial：robot-only、robot + human RGB with shuffled actions、robot + human action without diverse RGB（或统一背景）、robot + full human；同时等 optimizer steps、等 frame/token，并在 human 未见 OOD 上测试。这样才能拆开视觉覆盖、动作监督和计算量。

## 13. Scaling 与效率：五种口径不能混用

| Claim 口径 | 本文证据 | 能得出的结论 | 不能得出的结论 |
| --- | --- | --- | --- |
| 每条采集 wall-clock | grasp 4.82×、pour 7.61× faster | human 完成短任务更快 | human demo 单条价值更高 |
| 20 分钟预算 | 60 robot: 28/90；30 robot + 120 human: 35/90 | mixed allocation 更有效 | 等 episode/token 下 human 更好 |
| Demo count | 26,824 human vs 1,552 robot | human corpus 可更大 | 每个 demo 定义和长度相同 |
| Frame count | 3.02M vs 668k | human frames 约 4.5× | training sampler 按该比例抽样 |
| Optimization exposure | 实现默认 human/robot 各 0.5 | 避免 human 数量淹没 robot | 论文所有实验均完全锁定同一 sampler/steps |

“scalable”在采集端成立得最强：human 不占用机器人、不承受硬件风险、环境切换容易。端到端系统仍包含 headset SDK、同步、HDF5 转换、质量过滤、训练和真机验证，且数据多样性会提高治理与 failure auditing 成本。

## 14. 开源复现：可以做到哪一级

### Level A：数据与模型接口复现——可行

可以下载处理后 HDF5，使用公开 dataset configs、128-slot schema、DINOv2 + ACT/DP/RDT trainer，完成可视化、训练和 MuJoCo replay。注意仓库需要递归 submodules、硬编码/环境相关路径和较大 CPU RAM，因为 loader 默认把 HDF5 全缓存到内存。

### Level B：方法级结果趋势复现——中等难度

需要确定每张表对应 task directories、episode split、normalization mode、chunk size、训练步数和 seed。当前没有 versioned experiment manifest、论文 checkpoint、W&B run export 或统计脚本；因此能复现 HAT pipeline，不等于能精确重现 101/170。

### Level C：真机论文结果复现——困难

需要 Unitree H1/H1-2、Inspire 6-DoF hands、2-DoF neck、ZED/ego cameras、OpenTeleVision、robot FK/IK、hand retargeting、校准和安全控制。仓库 TODO 尚有 teleop scripts，公开代码主要展示 preprocessing/training/MuJoCo 例子，没有完整的论文真机 rollout artifact。缺权重意味着还需自行训练并挑选 checkpoint。

### 建议的最小复现实验

1. 从一个 human task 和一个 simulated/robot task 各取固定数量 HDF5，验证 schema、embodiment attr、camera key 和 128-slot index。
2. 固定 ACT、seed、optimizer steps，分别训练 robot-only、mixed-shared-norm、mixed-diff-norm。
3. 同时报 equal wall-clock 与 equal frame/token 两种比较，记录每域实际 sample 数。
4. 在 3×3 grid 中复验位置 heatmap，至少训练 3 seeds，报告 Wilson interval/bootstrapped CI。
5. 增加 shuffled-human-action 和 image-only human 对照，拆开 motion label 与 visual diversity。
6. 在 deployment 前做 IK feasibility、joint-limit、collision 和 action-speed filter；simulation 成功后再真机低速验证。

## 15. 十个必须追问的问题

1. 26,824 条 human demos 来自多少人、多少地点？是否有 participant-held-out test？
2. 当前 16.2 GB release 与论文 3.02M/668k frames 的精确覆盖关系是什么？
3. 论文所有主结果分别使用哪个 commit、dataset config、chunk size、训练 steps 和 seed？
4. human/robot 50/50 sampler 是否用于全部表？若改成 data-proportional，结果如何？
5. 54D 论文空间与 128-slot implementation 的 preprocessing 版本何时变化？head translation 是否有效？
6. Gaussian blur 在论文训练中是否真的启用？当前注释状态是 release drift 还是最终设置？
7. “differentiably retargeted”是否意味着训练梯度经过 IK/hand retargeter？若是，loss 和代码入口在哪里？
8. human RGB-only、human pose-only、shuffled action 和 off-task human 的正交消融结果是什么？
9. 真机控制频率、IK failure、collision、intervention、hand slip 和 force safety 指标是多少？
10. 数据的 consent、bystander/privacy 处理、raw retention、identity/location split 和重分发边界是什么？

## 16. 总体评价

| 维度 | 评分 | 理由 |
| --- | ---: | --- |
| 问题重要性 | 5/5 | human Ego 是扩大 humanoid manipulation 数据的关键来源 |
| 方法清晰度 | 4/5 | unified geometry + slowdown + HAT 很直观；implementation schema/retargeting细节有漂移 |
| 数据贡献 | 4/5 | 论文规模大、设备多样、处理数据公开；治理与覆盖说明不足 |
| 真机证据 | 4/5 | 四任务、两 H1 变体、多个 OOD 维度；无统计区间/多 seed |
| Embodiment gap | 3.5/5 | 几何层解决得好，kinematics/hand topology 仍有限，dynamics 未触及 |
| Task gap | 1.5/5 | 每任务 checkpoint，语言未用，无新任务组合 |
| Reality/contact | 1.5/5 | 真机验证但没有 force、tactile、balance、collision 或 recovery modeling |
| 可复现性 | 3/5 | code/data/license 较开放；缺权重、锁定配置、CI 和完整真机栈 |
| Claim 严谨度 | 3/5 | 趋势可信；“nearly 100%”“significant”及 scaling 需更严格统计口径 |

### 最终判断

`Humanoid Policy ~ Human Policy` 是一篇重要的**表示选择论文**：它证明在五指双臂 humanoid 上，head–wrist–fingertip geometry 足以成为 human 与 robot action 的实用公共接口，human data 也能通过这个接口直接改善真实机器人同任务 OOD 表现。最强贡献不是某个全新 transformer，而是把数据采集、统一状态动作、速度对齐、归一化与部署重定向连成了可验证系统。

它尚不是“只看人类视频就得到通用 humanoid”的答案。当前系统仍需要每任务 robot demonstrations、每任务 checkpoint、平台 FK/IK、同型五指 hand 和受控 fixed-base 执行；human data 提供的主要是更多样的视觉—几何动作覆盖，而接触、动力学、whole-body mobility 与高层语义都留给未来工作。最准确的定位是：**从 human Ego pose 到 humanoid task-space action 的共同训练桥梁，而不是从任意生活视频到任意机器人 motor policy 的完整闭环。**
