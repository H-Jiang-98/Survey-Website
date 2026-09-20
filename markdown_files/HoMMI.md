---
title: "HoMMI: Learning Whole-Body Mobile Manipulation from Human Demonstrations"
method_name: "HoMMI"
authors: [Xiaomeng Xu, Jisang Park, Han Zhang, Eric Cousineau, Aditya Bhat, Jose Barreiros, Dian Wang, Jeannette Bohg, Shuran Song]
year: 2026
venue: "Robotics: Science and Systems (RSS) 2026"
tags: [robot-free-demonstration, mobile-manipulation, cross-embodiment, diffusion-policy, active-perception, whole-body-control, sim-free-real]
zotero_collection: "_inbox/cross-embodiment"
image_source: online
arxiv_id: "2603.03243"
arxiv_html: "https://arxiv.org/html/2603.03243v2"
created: 2026-08-12
---

# HoMMI: Learning Whole-Body Mobile Manipulation from Human Demonstrations

## 元信息与证据边界

| 项目 | 内容 |
|---|---|
| 作者 | Xiaomeng Xu, Jisang Park, Han Zhang, Eric Cousineau, Aditya Bhat, Jose Barreiros, Dian Wang, Jeannette Bohg, Shuran Song |
| 机构 | Stanford University；Toyota Research Institute |
| 会议 | RSS 2026 |
| 论文 | [RSS 页面](https://roboticsconference.org/program/papers/205/) · [arXiv v2](https://arxiv.org/abs/2603.03243) · [arXiv HTML](https://arxiv.org/html/2603.03243v2) |
| 项目主页 | [hommi-robot.github.io](https://hommi-robot.github.io/) |
| 官方代码 | [xxm19/hommi](https://github.com/xxm19/hommi) |
| 硬件文件 | [仓库 README 中的 Hardware 链接](https://github.com/xxm19/hommi#data-collection) |
| 正文版本 | arXiv:2603.03243v2，2026-05-14 |
| 审查日期 | 2026-08-12 |

证据标签：**[论文明确]** 指正文/附录；**[代码明确]** 指官方仓库文件或 README；**[作者材料]** 指项目页；**[合理推断]** 是基于公开实现与机器人常识的分析；**[证据不足]** 表示公开材料无法确认。本文没有把项目视频里的成功片段单独当作定量证据。

## 一、论文速览

### 一句话总结

> HoMMI 把三台 iPhone 采集的人类头—双手轨迹转成共享的手眼任务空间，再由 RB-Y1 专用全身 IK 落地，从而无需机器人遥操作数据完成长时程移动操作。

### Elevator pitch

HoMMI 在 UMI 双手持握器上增加头戴 iPhone，使人可以脱离机器人采集导航、双臂操作与主动视知觉示范。增加头视角也扩大了人—机器人外观、视角、高度与颈部自由度差异，因此作者把头部 RGB 提升为带 3D 位置编码的视觉 token、遮掉人体区域，并把完整头姿态放松为一个 3D look-at point。[[Diffusion Policy]] 只预测双手 SE(3)、夹爪宽度和凝视目标；机器人专用差分 IK/QP 再负责底盘、躯干、双臂和颈部的可执行性。三项真机任务成功率为 90%/85%/80%，但证据只覆盖一台经过相机、手指和控制器协同设计的 RB-Y1，而不是任意目标本体。

### 迁移类型

| 类型 | 判断 | 说明 |
|---|---|---|
| 1. 人类数据 → 机器人策略 | **核心** | 三 iPhone 的 robot-free human demonstrations 直接训练真机策略。 |
| 2. 人类交互 → 多机器人 | 不适用 | 只有一个双臂移动机器人。 |
| 3. 已见机器人 → 未见机器人 | 不适用 | 没有多机器人预训练或 held-out robot。 |
| 4. 跨模态数据迁移 | **是** | 人端 iPhone RGB-D/ARKit pose → 机器人端工业 RGB/stereo depth/proprioception。 |
| 5. 仿真 → 真机 | 不适用 | 策略不是 sim-to-real；控制器在真机上直接执行。 |
| 6. 跨任务/场景/对象 | **有限** | 三个任务分别训练；测试改变对象、目标位置、光照和初始位姿，不是跨任务统一策略。 |

### Source、Target、资产与 gap

| 项目 | 审查结论 |
|---|---|
| Source domain | 两名身高 167/182 cm 的人，持双 UMI gripper、头戴 iPhone，在真实场景完成 Laundry/Delivery/Tablescape。 |
| Target domain | 一台定制 Rainbow Robotics RB-Y1：全向底盘、6-DoF torso、双 7-DoF arm、2-DoF neck、匹配 UMI 的 fin-ray fingers、头部 stereo 和腕部工业相机。 |
| 真正迁移的资产 | 原始/处理后人类轨迹、任务空间双手动作、主动视知觉意图、视觉—动作策略参数。 |
| 未直接迁移 | 人体关节轨迹、人体动力学、底盘/躯干/手臂关节命令、接触力、低层控制器参数。 |
| 核心 transfer gap | egocentric 视角与外观变化；人头 6-DoF 与机器人 2-DoF 颈部/较矮机身之间的运动学不可执行性。 |
| 最关键机制 | gripper-centric 的 3D 视觉/动作坐标系 + 3D look-at action abstraction + constraint-aware WBIK。 |
| 最大贡献 | 第一次把 robot-free UMI 式采集扩展到需要搜索、导航、主动视知觉和双臂全身协调的长时程真机任务。 |
| 最应质疑的 claim | “directly transferable/generalizable” 容易被理解为跨机器人泛化；实际只有单一、强协同设计目标机器人。 |
| 是否值得精读 | **值得**：表示和控制接口设计清晰，系统闭环完整。 |
| 是否值得复现 | **有条件**：代码公开，但硬件、三 iPhone、工业相机、RB-Y1 与实时控制栈门槛高。 |

## 二、问题的形式化定义

令三个任务各自的数据为

$$
\mathcal{D}_{src}=\{D_{\text{laundry}},D_{\text{delivery}},D_{\text{tablescape}}\},
\qquad
\mathcal{D}_{tgt}=\{D'_{\text{RB-Y1, real}}\}.
$$

每条人类轨迹在时刻 $t$ 含三视角 RGB、depth/pointmap、三台 iPhone 的全局 6-DoF pose 和两夹爪宽度。策略使用

$$
O_t=\{o_{t-T_o+1},\ldots,o_t\},\qquad
A_t=\{a_{t+1},\ldots,a_{t+T_p}\},
$$

其中 $T_o=2$，$T_p=32$，数据由 60 Hz 下采样到 20 Hz；动作维度为

$$
a_t=\big[T^L_{g,t}(9),T^R_{g,t}(9),\ell_t(3),w^L_t,w^R_t\big]\in\mathbb{R}^{23}.
$$

双手姿态各用 3D position + rotation matrix 前两列的 6D rotation 表示。

| 维度 | 源域 | 目标域 | 对齐/差异 |
|---|---|---|---|
| 本体 | 人的头、躯干、手臂；手持 UMI grippers | RB-Y1 全向底盘、torso、arms、2-DoF neck | 不对齐人体关节，只对齐 gripper poses 与 gaze intent。 |
| 观测空间 | iPhone head/wrist RGB-D + ARKit poses | industrial wrist RGB + stereo head RGB/depth + robot EEF state | 共享 DINOv3；head RGB 绑定 3D point token；left-gripper frame。 |
| 动作空间 | 人的两 gripper SE(3)、width、head ray | 双 EEF target、gripper width、look-at point | task-space canonical action；WBIK 转 joint/base motion。 |
| 动力学 | 人体自然运动，无机器人动力学标签 | RB-Y1 运动学/速度/碰撞约束 | 由 target-specific controller 兜底，策略不建模动力学。 |
| 任务分布 | 三个独立任务、变化场景 | 同一任务的新初值/部分新物体 | 不是跨任务 generalist。 |
| 源模态 | RGB、depth、pose、gripper width | RGB、stereo depth、proprioception | 无 force/tactile/contact labels。 |
| 训练时目标域信息 | **有**：机器人相机位置、手指几何、动作接口、控制器与少量 rollout 调参 | — | 没有 target robot demonstration，但不是“目标系统零信息”。 |
| 测试时目标域信息 | — | 实时 RGB-D、EEF pose、gripper width、完整 robot model | WBIK 使用关节/速度/碰撞/CoM 信息。 |

论文学习的是**共享数据转换后的目标条件策略**，不是跨多本体共享策略。目标任务和场景分布在训练中见过，测试仅改变初始条件与部分对象。它假设下列不变量：物体/场景的局部 3D 几何、相对 gripper frame、末端执行器轨迹语义以及 gaze target 的任务意图；这些具有任务空间和射影几何依据，但“相同 gripper pose 能实现相同接触结果”仍依赖手指几何被人为匹配。

## 三、迁移账本：Source → Alignment → Target

| 阶段 | 输入 | 输出 | 域差异 | 对齐机制 | 可学习 | target-specific | 证据 |
|---|---|---|---|---:|---:|---:|---|
| 数据采集 | 人的双手/头动作 | 三 iPhone RGB-D/pose/width | 无机器人状态 | ARKit multi-device shared frame | 否 | 否 | [论文明确][代码明确] |
| 数据预处理 | 60 Hz 多机流 | 时间对齐 20 Hz Zarr dataset | latency/坐标漂移 | group、time align、session、dataset builder | 否 | 部分 | [代码明确] |
| 观测表示 | head/wrist image+pointmap | $F_{ego},F_{wrist}$ 与 proprio token | 人体外观、视高、相机差 | DINOv3 patch + 3D PE；arm mask；gripper frame | 是 | 相机标定是 | [论文明确] |
| 动作表示 | 双手 6-DoF、头 6-DoF | 双手 9D pose、look-at point、width | 2-DoF neck 不可复制人头姿 | 6-DoF head → 3D gaze target | 部分 | neck conversion 是 | [论文明确] |
| 几何/本体对齐 | 全局 pose/pointmap | left-gripper-centric coordinates | 身高、body topology | 不映射人体 joints；只保留 task-space intent | 否 | frame calibration 是 | [论文明确] |
| 接触对齐 | wrist RGB、gripper width | 隐式视觉接触线索 | 无 force/tactile | 无显式对齐 | 是（隐式） | 匹配 fin-ray finger 是 | [论文明确] |
| 策略学习 | $O_t$ | action chunk | 人/机视觉与状态分布 | DiT Diffusion Policy | 是 | 每任务单独训练 | [论文明确][代码明确] |
| 低层控制 | EEF target/look-at | base+torso+arm/neck command | task-space→joint-space | WBIK/QP、look-at neck controller | 否 | **强** | [论文明确][代码明确] |
| sim-to-real | — | — | — | 不适用 | — | — | [论文明确] |

信息损失最大的是把人体全身/头姿态压缩成双手末端轨迹和单个 look-at point：它保留任务意图但丢弃身体动力学、接触力、注视深度不确定性和备用动作。误差从 ARKit pose/depth、跨设备同步、DINO/pointmap、action diffusion、WBIK tracking 依次累积。迁移上限由 **canonical hand-eye interface 是否足够表达任务** 与 **WBIK 是否能实现该接口**共同决定；RB-Y1 相机/手指匹配、任务权重、碰撞组和标定属于工程适配。去掉 target-specific WBIK 与硬件匹配，系统不能直接工作。

## 四、源数据审查

| 任务 | 示范数 | 变化 | 测试 | Ours |
|---|---:|---|---:|---:|
| Laundry | 200 | bin location、initial configuration、5 cloth objects | 20 rollouts；2 seen + 3 unseen objects；4 bin configs | 90% |
| Delivery | 166 | trolley/standing locations | 20 rollouts；5 trolley locations × 4 base initializations | 85% |
| Tablescape | 115 | base start、mat placement | 20 rollouts；5 base starts × 2 mat configs × 2 | 80% |
| 合计 | **481** | 三个独立任务 | 60 rollouts | — |

数据由 3 台 iPhone 同步记录 RGB、depth、6-DoF pose 和 gripper width，60 Hz；两位采集者身高为 167/182 cm。ARKit 对 mocap 的平均误差在 5.0 mm / 0.8° 内。**[论文明确]** 没有公布总时长、每条平均长度、失败示范是否保留或人工清洗率；也没有 force/tactile/contact label。代码提供 iPhone 分组、时间对齐、可视化、session 构建和 Zarr 导出命令。**[代码明确]**

成本判断：相较 RB-Y1 遥操作，三部手机和持握器更便携、无需占用机器人，采集动作也更自然；但它并非“零设备”——需要三台支持 ARKit collaboration 的 iPhone、打印/加工的 UMI gripper、可靠内外参和时钟同步。采集成本更低的 claim 在流程上可信，但论文没有做分钟/美元/有效轨迹率的量化对比。其他机器人可复用原始数据的前提，是实现相同的 hand-eye canonical interface、相近夹爪接触几何和本体控制 adapter。

主要数据瓶颈依次为：**物理信息缺失 > 数据到新本体的可执行性 > 长时程覆盖 > depth/pose 质量**。遮挡与腕部局部视觉通过多视角缓解；不同身高用 3D 表示和 gaze abstraction 缓解；相机漂移、失败数据与动作分段细节证据不足。

## 五、统一表示与对齐机制

### 5.1 3D visual tokens

头图像缩放为 $512\times512$，按 $16\times16$ patch 得到 $32\times32=1024$ 个 patch；pointmap 下采样并与 DINOv3-ViT-B/16 patch feature 拼接正弦 3D positional encoding，再下采样为 512 tokens 并 attention pool。腕图为 $224\times224$，取共享 DINO encoder 的 CLS feature。pointmap 被变换到左右 gripper frame，并删除 $z<0$ 的人体臂/躯干点。

它对齐了坐标系、视高和人体外观，但没有使相机 photometric domain 真正相同。它保留物体语义和 3D 相对几何，主动丢弃人的手臂/身体外观；这有利于 transfer，却也会丢掉由人体姿态暗示的遮挡、即将发生的碰撞和动力学信息。

### 5.2 3D look-at point

训练时，中央相机 ray 与场景 pointmap 的交点为 $\ell_t\in\mathbb{R}^3$。部署时令头位置为 $c_t$、当前坐标轴为 $R_t^{cur}=[x_t,y_t,z_t]$：

$$
\hat d_t=\frac{\ell_t-c_t}{\|\ell_t-c_t\|},
\qquad
x'_t=x_t-(x_t^\top\hat d_t)\hat d_t,
\qquad
\hat x_t=\frac{x'_t}{\|x'_t\|},
$$

$$
\hat y_t=\hat d_t\times\hat x_t,
\qquad
R_t=[\hat x_t,\hat y_t,\hat d_t].
$$

若 $\|x'_t\|$ 接近零则以 world-up 替换 $x_t$。这把“复制人的头姿态”改成“看到同一个任务点”，是最干净的意图对齐；但深度错误会直接移动 gaze target。

### 5.3 gripper-centric frame

双手 proprio/action、head pointmap 和 look-at point 都表达在 left-gripper frame。该 frame 随执行端而非人头移动，使身高、头相机位姿和 neck DoF 变化不会把输入推到完全 OOD。人为设计仍很强：为什么选择左手而非对象 frame、双手交换角色时如何处理，论文没有跨任务验证。

> 这篇论文的迁移能力主要来自 **任务空间 hand-eye 表示与目标控制接口** 层面的对齐，而不是简单扩大训练数据。

## 六、观测与多模态迁移

| 模态 | 源域 | 目标域 | 对齐方式 | 部署可用 | 缺失处理 |
|---|---:|---:|---|---:|---|
| RGB | 是，3 iPhone | 是，2 wrist + stereo head | shared DINOv3；head 3D PE | 是 | 无 dropout 报告 |
| 深度/点云 | iPhone depth | stereo depth/FoundationStereo | pointmap + gripper frame | 是 | depth 噪声消融 |
| proprioception | gripper poses/width | EEF poses/width | same task-space schema | 是 | 不适用 |
| object pose | 否 | 否 | 隐式视觉 | 否 | policy 自行感知 |
| force/torque | 否 | 低层安全可能有，但非 policy input | 不适用 | 否 | 无 |
| tactile | 否 | 否 | 不适用 | 否 | 无 |
| contact state | 否 | 否 | wrist image 隐式 | 否 | 无 |
| peer state | 不适用 | 不适用 | — | — | — |
| URDF/robot model | 人端无 | WBIK 使用 | controller metadata | 是 | 不可缺 |
| privileged state | 否 | 否 | — | — | — |

融合方式是 token/feature 级中期融合：shared vision backbone 提取 wrist/head features，head token 额外注入 3D position，随后与 proprioception 条件共同进入 DiT。没有 teacher-student、modality dropout、force/tactile 或明确延迟噪声训练；但部署桥按测得 latency 对齐相机，proprioception 插值到图像时间戳。深度噪声标准差 0/2/10/20 mm 时 Laundry 成功率为 90/90/90/50%，说明 1 cm 内稳健、2 cm 明显退化。这里是**跨设备的 RGB-D/pose 表示迁移**，不只是“多输入”；但 wrist contact cue 仍未形成可解释的跨模态互补机制。

## 七、动作、意图与控制接口迁移

$$
\text{human hand/head intent}
\rightarrow
\{T_g^L,T_g^R,\ell,w_L,w_R\}
\rightarrow
\text{RB-Y1 WBIK + neck mapping}
\rightarrow
\dot q_{base,torso,arms},q_{neck},w_{gripper}.
$$

策略不是直接输出机器人 joint action，而是输出 reference；controller 承担关节限位、自碰撞、底盘/关节速度、CoM 支撑、竖直躯干和动作平滑。因而它实现的是：

- **policy-level transfer**：有人类数据→目标 task-space policy，可信；
- **controller-level zero-shot**：不成立，控制器是 RB-Y1 专用并按任务设权重；
- **system-level zero-shot**：不成立，需相机/手指匹配、标定、robot model、WBIK 和 1–2 次 rollout 调参。

### Whole-body IK/QP

$$
\begin{aligned}
\min_{\Delta q}\quad &
f(\Delta q)+\lambda\|\Delta q\|_2^2,\\
f&=C_{ee}+C_{nominal}+C_{current}+C_{com},\\
\text{s.t.}\quad &G_{cfg}\Delta q\le h_{cfg},\;
G_{joint\text{-}vel}\Delta q\le h_{joint\text{-}vel},\\
&G_{base\text{-}vel}\Delta q\le h_{base\text{-}vel},\;
G_{coll}\Delta q\le h_{coll},\;
A_{upright}\Delta q=0.
\end{aligned}
$$

主要 cost 为

$$
C_{ee}=\sum_{i\in\{L,R\}}\|J_i\Delta q-v_i\|_{W_{ee}}^2,
$$

$$
C_{nominal}=\|(q+\Delta q)-q_{nom}\|_{W_{nom}}^2,
\qquad
C_{current}=\|\Delta q\|_{W_{curr}}^2,
$$

$$
\begin{aligned}
C_{com}&=\|p^{xy}_{torso}(q+\Delta q)-p^{xy}_{base}(q+\Delta q)-r^{xy}_\star\|_{W_{com}}^2,\\
|p^x_{torso}-p^x_{base}-r^x_\star|&\le b_x,
\quad
|p^y_{torso}-p^y_{base}-r^y_\star|\le b_y.
\end{aligned}
$$

policy 10 Hz、WBIK 100 Hz、robot control 500 Hz。命令插值为

$$
\alpha(t)=\min\left(1,\frac{t-t_0}{T}\right),
\qquad
p_{interp}(t)=(1-\alpha)p_{prev}+\alpha p_{cmd},
$$

orientation 用 SLERP。碰撞安全距离 0.01 m、影响距离 0.02 m；joint/base velocity 用 0.9 safety scale。它没有显式环境碰撞规划，也没有 torque/force limit 进入 policy objective；接触安全部分仍靠硬件 guard 与低层栈。

## 八、本体、物理属性与 target-specific engineering

| 信息 | 是否显式 | 用途 |
|---|---:|---|
| kinematic tree/joint limits | 是，robot model | WBIK constraint |
| link mass/CoM | 是 | CoM-over-base task |
| joint/torso/base velocity limits | 是 | QP bounds |
| collision geometry | 是，选定 geometry groups | self/inter-arm collision avoidance |
| actuator model/torque limit | 未在 policy/WBIK 公式中明确 | 低层控制栈可能处理，证据不足 |
| sensor layout/calibration | 是 | 多相机时间与空间对齐 |
| matching fingers/camera placement | 是，人为协同设计 | 缩小 hardware embodiment gap |

新本体绝不是“只给 URDF”：至少还需头/腕 RGB-D、相机 latency/外参、gripper schema、look-at neck converter、whole-body IK cost/constraints、collision pairs、nominal posture、low-level interface 和安全验证。论文仅测试尺寸不同的人类采集者，不测试机器人 topology 变化。结构外推、非全向底盘、单臂机器人、无 neck 平台、不同 gripper 均是证据不足。

## 九、Retargeting、物理可执行性与接触

论文没有传统的人体 skeleton→robot skeleton retargeting。它主动绕过这一步：只保留手眼 task-space intent，再让在线 WBIK 求目标动作。运动学可行不等于动力学可执行；HoMMI 的移动底盘平台没有步态平衡问题，但接触力、物体质量、轮地动力学和柔顺性仍不在策略表示中。

接触来自 wrist RGB 的隐式线索和 matched fin-ray fingers，而非测量/标注。没有 contact topology、force direction/magnitude、tactile 或 compliance learning。因此精细抓取失败（slip、miss、press table）本质上暴露了 canonical action 对接触的表达不足。

## 十、训练策略与数据混合

1. 三 iPhone 采集并同步 60 Hz RGB-D/pose/width。
2. 分组、time align、建立 session，导出 `dataset.zarr.zip`。
3. 下采样 20 Hz；$T_o=2$，$T_p=32$。
4. shared DINOv3-ViT-B/16 编码 wrist/head；head 加 3D PE、mask 与 attention pooling。
5. DiT Diffusion Policy：embedding 768、depth 10、12 heads、MLP ratio 4、RMSNorm；100 train diffusion steps、16 DDIM inference steps、input perturbation 0.1，每 observation 8 noise samples。
6. AdamW，policy LR $7.5\times10^{-5}$，vision LR $7.5\times10^{-6}$，weight decay $10^{-6}$，betas $(0.95,0.999)$，cosine schedule，500 epochs。
7. 异步 policy server 输出 chunk；WBIK 与真机 low-level controller 执行。

没有多本体 joint training、embodiment ID、specialist head、RL、teacher/student、domain randomization 或 target finetuning。任务是否各自一个 checkpoint，结合数据规模和命令可合理推断为 task-specific policies，但公开摘要未宣称一个 multi-task checkpoint；应避免把三任务演示理解为统一策略。

## 十一、Zero-shot 纯度审查

| 项目 | 使用 | 是否违反“无机器人示范” | 对 system zero-shot 的影响 |
|---|---:|---:|---|
| 目标域训练轨迹 | 否 | 否 | 正面 |
| 目标域无标签数据 | 未报告 | 否 | — |
| 目标域 reward rollout | 无 RL；有 1–2 rollout 调 WBIK 权重 | 不违反数据 claim | 违反严格 system zero-shot |
| target finetuning | 否 | 否 | 正面 |
| 目标 URDF/model | 是 | 否 | 必要本体配置 |
| 人工语义映射 | 是：gripper frame/look-at/action schema | 否 | 明显人工先验 |
| target action mapping | 是 | 否 | 核心依赖 |
| target controller | 是，RB-Y1 WBIK | 否 | 核心依赖 |
| target DR/system ID | 无 | 否 | — |
| 真机标定 | 是，多相机与 hardware | 否 | 系统工程 |
| target hyperparameter tuning | 是，task-specific WBIK weights | 不违反“no robot data” | 影响纯度 |
| checkpoint selection | 未说明 | 证据不足 | 证据不足 |

最终分类：**没有机器人遥操作/训练轨迹，policy 直接由人类数据训练；但存在明显目标域本体配置与系统工程**。论文没有主打 strict zero-shot，因此不应强行按严格 zero-shot 批判；准确说法是 “zero robot demonstration transfer to one co-designed target platform”。

## 十二、实验是否真正证明迁移

核心结果（由论文 Figure 8 汇总）：

| Method | Laundry | Delivery | Tablescape | 解释 |
|---|---:|---:|---:|---|
| Wrist-Only | 低；Delivery 15% | 15% | 低 | 缺全局 context |
| RGB-Only（head RGB + 6-DoF head） | 0% | 45% | 0% | appearance/view/kinematic OOD |
| Head-Only | 0% | 5% | 低 | 缺局部接触视觉 |
| w/o Active Neck | 75% | 55% | 55% | 缺主动视知觉 |
| **HoMMI** | **90%** | **85%** | **80%** | 完整表示+动作抽象+控制器 |

这些实验真实证明了：

- robot-free data 在该 co-designed RB-Y1 系统上可用；
- 直接加入 egocentric RGB/6-DoF head 失败，而 3D/head-action redesign 有迁移价值；
- wrist 与 head views 互补，active neck 对长时程任务有用。

它没有证明：跨目标机器人泛化、跨任务共享、同数据对比 robot teleoperation 的数据效率、specialist upper bound、去掉 WBIK 后仍能 transfer、或 matched/mismatched gripper 的影响。最重要的缺失 baseline 是**同等数量的 RB-Y1 teleoperation specialist**与**同一人类数据接入另一种机器人**。

### 附录表格 1：Whole-body IK 参数（完整）

| Symbol | Laundry | Delivery | Tablescape |
|---|---:|---:|---:|
| $w_p$ | 10000 | 10000 | 10000 |
| $w_o$ | 10000 | 10000 | 10000 |
| $w_{nom,torso}$ | 50 | 1000 | 200 |
| $w_{nom,arm}$ | 50 | 1000 | 10 |
| $w_{curr}$ | 50 | 1000 | 10 |
| $w_{base,pos}$ | 50 | 50 | 5000 |
| $w_{base,ori}$ | 50 | 50 | 5000 |
| $w_{com}$ | 100000 | 100000 | 100000 |
| $b_x,b_y$ | 0.08 m | 0.08 m | 0.08 m |

该表直接说明 controller 不是 task-agnostic：Delivery 强正则 torso/arm/current，Tablescape 强约束 base pose。作者称每项只需 1–2 rollouts 定参，但这仍是 target-task engineering。

### 附录表格 2：Depth sensitivity（完整）

| noise std (mm) | 0 | 2 | 10 | 20 |
|---:|---:|---:|---:|---:|
| success (%) | 90 | 90 | 90 | 50 |

### 附录表格 3：Seen/unseen objects（完整）

| object | s1 | s2 | u1 | u2 | u3 | u4 | u5 | u6 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| success (%) | 100 | 75 | 100 | 75 | 100 | 100 | 100 | 75 |

正文一处称 Laundry 测试有 2 seen + 3 unseen objects，而附表列 s1/s2 + u1…u6；公开 HTML 内部存在数量口径差异，笔记保留原表并标注这一点。

### 附录表格 4：Lighting robustness（完整）

| light (lux) | 4300 | 3370 | 1840 | 810 |
|---:|---:|---:|---:|---:|
| success (%) | 100 | 100 | 100 | 75 |

## 十三、关键图（论文 Figure 1–13，完整索引）

> 图片采用 arXiv HTML 在线资源；GitHub Markdown 可直接渲染。Figure 7 与 Figure 8 是相邻的两个独立资源。

### Figure 1：HoMMI teaser

![Figure 1](https://arxiv.org/html/2603.03243v2/x1.png)

展示从 UMI+ego 采集、人机 observation/action gap 到 representation+controller 的完整论点。

### Figure 2：系统总览

![Figure 2](https://arxiv.org/html/2603.03243v2/x2.png)

三条主链是采集、cross-embodiment hand-eye policy 与 target WBC；它也揭示 transfer 不是单个网络模块。

### Figure 3：Embodiment-agnostic visual representation

![Figure 3](https://arxiv.org/html/2603.03243v2/x3.png)

关键不是 3D backbone，而是 2D DINO token 与 3D point positional encoding、arm mask 和 gripper frame。

### Figure 4：Look-at point action

![Figure 4](https://arxiv.org/html/2603.03243v2/x4.png)

用任务意图替代不可执行头姿，是动作语义对齐的核心。

### Figure 5：Whole-body controller constraints

![Figure 5](https://arxiv.org/html/2603.03243v2/x5.png)

显示 simultaneous head/hand 6-DoF tracking 的冲突及 constraints/regularization 对稳定性的作用。

### Figure 6：RB-Y1 hardware

![Figure 6](https://arxiv.org/html/2603.03243v2/x6.png)

相机与 fin-ray fingers 被刻意匹配采集端；这是 system transfer 的重要工程前提。

### Figure 7：Laundry rollouts

![Figure 7](https://arxiv.org/html/2603.03243v2/x7.png)

给出 Laundry 的完整 rollout、测试变化与典型 baseline failure。

### Figure 8：三任务定量结果

![Figure 8](https://arxiv.org/html/2603.03243v2/x8.png)

比较三项任务的 HoMMI 与 baselines，是 transfer claim 的主要定量支撑。

### Figure 9：Delivery

![Figure 9](https://arxiv.org/html/2603.03243v2/x9.png)

6×6 m 导航暴露了 wrist-only 与 6-DoF head baseline 的长时程/可执行性问题。

### Figure 10：Tablescape

![Figure 10](https://arxiv.org/html/2603.03243v2/x10.png)

测试双手相对姿态、抓取高度、底盘与 torso 协同；仍无显式接触反馈。

### Figure 11：Egocentric attention

![Figure 11](https://arxiv.org/html/2603.03243v2/x11.png)

attention 更聚焦任务物体是定性证据，不能替代 causal ablation。

### Figure 12：Hardware schematic

![Figure 12](https://arxiv.org/html/2603.03243v2/fig/hardware_schematic.png)

展示双光纤链、PoE/GigE cameras、外部 workstation 与 robot PCs，说明复现并非普通单机部署。

### Figure 13：Stereo pointcloud 与 attention

![Figure 13](https://arxiv.org/html/2603.03243v2/x12.png)

与 depth noise 表一起说明 robot-side stereo quality 是 3D transfer 的基础条件；见 [HTML Figure 13](https://arxiv.org/html/2603.03243v2#A4.F13)。

## 十四、消融与缺失实验

| 消融 | 是否有 | 判断 |
|---|---:|---|
| 去统一 3D 表示 | 部分：RGB-Only | 同时改了 head action，因素耦合。 |
| 去 gripper-centric frame | **无** | 无法单独定位 frame contribution。 |
| 6-DoF head vs look-at | RGB-Only 中有 | 也混入 RGB vs 3D 表示。 |
| 去 head/wrist modality | 有 | Head-Only、Wrist-Only。 |
| 去 active neck | 有 | 三任务均降。 |
| 去 WBIK constraints | 只有 Figure 5 定性 | 无完整成功率/安全率。 |
| matched vs unmatched fingers/cameras | 无 | 这是 claim 中最关键的 hidden engineering。 |
| 多采集者/单采集者 | 仅说明两身高均成功 | 无分组数据。 |
| 不准 depth | 有 | 2 cm 时明显降。 |
| 不准 calibration/latency | 无 | 复现风险高。 |
| 不同 robot embodiment | 无 | 无法证明真正 robot cross-embodiment generalization。 |

最确定的贡献是“head+wrist multimodal sensing + active gaze”组合；但 3D visual representation、gripper frame 与 look-at action 没有完全正交消融，存在组合有效但无法精确归因的问题。

## 十五、人类示范 → 移动操作专项判断

1. 数据在训练意义上完全 robot-free，没有 RB-Y1 teleoperation trajectory。**[论文明确]**
2. 但系统不是 robot-engineering-free：finger/camera matching、WBIK、标定和 task weights 很强。**[论文明确][代码明确]**
3. 人机视觉差由 3D token、arm masking、gripper frame 与相机布局共同缩小。
4. 人头→2-DoF neck 通过 look-at point relaxation，而不是 head retargeting。
5. 双手精确任务由 EEF SE(3) tracking 保留，底盘/torso/arms 由 WBIK 冗余分配。
6. 两名不同身高采集者能混用，但样本分布、每人贡献和独立成功率未披露。
7. 长时程失败来自：视觉/抓取 slip，长导航末端 misalignment，mat missing grasp；contact sensing 和 memory 都是瓶颈。
8. “无需机器人数据”成立；“无需机器人专用工程”不成立。

## 十六、局限、失败模式与五个组会问题

主要局限：单机器人、单类夹爪、三个 task-specific datasets；短 observation history；vision-only；无 force/tactile/compliance；硬件 co-design；WBIK 权重按任务；没有机器人 teleop specialist；长时程仍会漂移；外部 workstation 与复杂网络/相机栈降低可迁移性。

组会最值得问：

1. 如果把相同 481 条人类示范部署到另一台高度、底盘、neck、gripper 都不同的机器人，需要多少新工程与性能损失？
2. 3D visual representation、gripper frame、arm mask、look-at action 四者能否正交消融？
3. matched fin-ray fingers 和相似 wrist camera placement 实际消除了多少 embodiment gap？
4. 与同等示范数/采集时长的 RB-Y1 teleoperation specialist 相比，成功率与数据效率如何？
5. controller 使用任务权重并进行 rollout tuning 后，应该把结果解释为 data transfer、policy transfer 还是完整 system co-design？

## 十七、代码、数据与复现审查

### 仓库识别（截至 2026-08-12）

| 仓库 | 类型 | 与论文关系 | 完整度 | 可信度 | 备注 |
|---|---|---|---|---|---|
| [xxm19/hommi](https://github.com/xxm19/hommi) | 官方 | 主实现 | 中高 | 高 | MIT；采集处理、训练、部署、WBC submodule 接口均有。 |
| [universal_manipulation_interface](https://github.com/real-stanford/universal_manipulation_interface) | 官方依赖 | Diffusion Policy/UMI 基础 | 高 | 高 | 以 submodule 依赖。 |
| `deps/rby1-wbc` | 官方/项目依赖 | RB-Y1 realtime WBC | 中高 | 高 | 需构建 pybind realtime controller。 |

公开链路包括：iPhone app/mount 链接；demo grouping/time alignment；session/Zarr dataset builder；DiT policy config；多 GPU 命令；camera config/viewer；policy server；RB-Y1 WBC rollout。README 未给可直接下载的训练数据/official checkpoint 链接，原始三任务数据、完整 calibration recipe、训练日志与结果复现实验脚本的可用性需要进一步逐文件验证，不能仅凭“all code/data publicly available”一句话视为全量发布。

### 新本体接入清单

需要：多视角 RGB-D；camera intrinsics/extrinsics/latency；EEF pose/width interface；与采集器相容的 gripper geometry；robot URDF/collision/CoM；look-at→neck converter；whole-body IK/QP adapter；joint/base/torque safety；nominal posture；asynchronous bridge；若干 real rollouts 做约束和权重校准。网络结构通常无需改，但观察/action schema 或 camera count 改变时需要 retraining。对成熟平台估计是**数周级**工程，而不是“换 URDF 即用”；最大风险是接触几何、相机 domain 和 WBIK 可达性。

## 十八、分层复现路线

### Level 1：最小迁移验证

- 依赖：官方 repo、mamba env、DGL、UMI submodule；一条公开/自采 iPhone demo。
- 最小硬件：三 iPhone + UMI grippers；若无 RB-Y1，先离线验证 dataset 与 policy output schema。
- 成功标准：生成 Zarr；训练小 policy；可视化 action chunk；验证 gripper-frame transform/look-at geometry。
- 风险：iPhone app、ARKit multi-device collaboration、相机同步和未公开 checkpoint。

### Level 2：论文级复现

- 需要 RB-Y1、头 stereo、双 wrist cameras、fin-ray fingers、外部 workstation、双网络链、WBC build。
- 重现 481 demos 或获得数据；训练三 task policies；跑 60+ trials。
- 必做指标：任务成功率、EEF tracking、stale action rate、depth/calibration noise、WBIK constraint violations。
- 最难对齐：相机/手指硬件、WBIK task weights、真实初始条件和安全 guard。

### Level 3：平台化扩展

建议标准接口：

```text
Observation = {timestamp, head_rgb, head_pointmap, wrist_rgb[L/R], eef_pose[L/R], gripper_width[L/R]}
CanonicalAction = {eef_delta_SE3[L/R], look_at_point, gripper_width[L/R], duration}
RobotMetadata = {urdf, collision_groups, com_model, velocity_limits, camera_calibration, neck_model}
ControllerAdapter.step(CanonicalAction) -> low_level_targets + constraint_diagnostics
```

回归测试应覆盖 frame round-trip、时间同步、depth scale、action reachability、collision/limit enforcement、active gaze visibility、不同采集者身高和失败恢复。

## 十九、与三种迁移范式对比

| 维度 | HoMMI | 人类示范迁移范式 | 交互迁移范式 | 跨形态统一策略范式 |
|---|---|---|---|---|
| Source data | 三 iPhone human demos | 人体 video/mocap/wearable | human-human paired motion | multi-robot rollout |
| Target | RB-Y1 mobile bi-manipulator | 单机器人 | 多机器人 | held-out robot |
| 主要 gap | ego vision + head/whole-body kinematics | appearance/action | topology/contact/coupled dynamics | varying DoF/topology/dynamics |
| 统一表示 | gripper-frame 3D tokens + look-at | task/hand-centric | interaction/contact graph | morphology graph/shared action |
| action | EEF SE(3)+gaze+width | task-space ref | paired reference/joint target | primitive/shared token |
| retargeting | 无 skeleton retarget | 常有 | 必需 | 常由 mapping 取代 |
| target mapping | 强 WBIK/neck | 通常有 | 强 | URDF/mapping |
| contact | 隐式视觉 | 通常弱 | 核心 | 视任务而定 |
| joint training | 否 | 可选 | 双 agent CTDE | 核心 |
| zero-shot 层级 | zero robot-demo；非 system zero-shot | data/policy transfer | retarget+sim2real | policy zero-shot + config |
| 最大优势 | 可扩展真实人类采集 | 数据便宜 | 保交互结构 | 新本体复用策略 |
| 最大局限 | 单目标硬件 co-design | embodiment gap | 状态估计/通信 | 人工语义与物理外推 |

## 二十、最终判断

| 结论项 | 判断 |
|---|---|
| 真正迁移了什么 | 人类真实 RGB-D/hand-eye trajectory、active gaze intent 与由此训练的 task-space policy。 |
| 能力主要来自 | **表示 + 动作接口 + controller co-design**，不是纯数据规模。 |
| 最值得学习 | gripper-centric 3D visual token；look-at action relaxation；asynchronous policy/WBIK separation。 |
| 最不应高估 | 任意机器人 cross-embodiment；零机器人专用工程；vision-only 对 contact-rich tasks 的成熟度。 |
| 数据复用价值 | **高（对相似 hand-eye mobile manipulation）/中（对异构硬件）** |
| 跨本体泛化可信度 | **中低**：人→单 robot 成立，robot→robot 未验证。 |
| 真实部署成熟度 | **高（该 RB-Y1 setup）** |
| 代码复现可行性 | **中**：软件链较全，硬件门槛高。 |
| 是否值得精读 | **值得** |
| 是否值得复现 | **有条件/谨慎** |
| 是否纳入团队数据平台 | **值得**，优先纳入 canonical hand-eye schema 与 timestamped RGB-D/pose pipeline。 |
| 最适合借鉴 | look-at intent abstraction 与 ControllerAdapter 分层。 |

> 这篇论文本质上是一项从 **robot-free 人类头—双手示范** 到 **RB-Y1 长时程移动操作系统** 的迁移工作。它真正建立的不变量是 **以夹爪为中心的三维任务几何、双手末端轨迹和凝视目标**，主要通过 **3D visual tokens、look-at action 与约束全身控制** 缩小域差异；其迁移能力上限取决于 **canonical hand-eye interface 对任务/接触的表达力和目标 WBIK 的可实现性**。论文最有价值的是 **把数据采集、表示、动作抽象和真机控制闭环打通**，但需要警惕它仍依赖 **匹配的手指/相机、RB-Y1 专用 WBIK、标定和 task-specific 权重**，因此其 generalization claim 应理解为 **无需机器人示范的单目标系统迁移，而非任意新本体上的严格 zero-shot**。
