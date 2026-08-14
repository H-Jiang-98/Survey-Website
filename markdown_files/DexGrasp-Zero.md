---
title: "DexGrasp-Zero: A Morphology-Aligned Policy for Zero-Shot Cross-Embodiment Dexterous Grasping"
method_name: "DexGrasp-Zero"
authors: [Yuliang Wu, Yanhan Lin, WengKit Lao, Yuhao Lin, Yi-Lin Wei, Wei-Shi Zheng, Ancong Wu]
year: 2026
venue: "Robotics: Science and Systems (RSS) 2026"
tags: [dexterous-grasping, cross-embodiment, zero-shot, morphology-aligned-graph, motion-primitives, reinforcement-learning, privileged-distillation, sim-to-real]
zotero_collection: "_inbox/cross-embodiment"
image_source: online
arxiv_id: "2603.16806"
arxiv_html: "https://arxiv.org/html/2603.16806v2"
created: 2026-08-12
---

# DexGrasp-Zero: A Morphology-Aligned Policy for Zero-Shot Cross-Embodiment Dexterous Grasping

## 元信息与证据边界

| 项目 | 内容 |
|---|---|
| 作者 | Yuliang Wu, Yanhan Lin, WengKit Lao, Yuhao Lin, Yi-Lin Wei, Wei-Shi Zheng, Ancong Wu |
| 机构 | Sun Yat-sen University |
| 会议 | RSS 2026 |
| 论文 | [RSS 页面](https://roboticsconference.org/program/papers/122/) · [arXiv v2](https://arxiv.org/abs/2603.16806) · [HTML](https://arxiv.org/html/2603.16806v2) |
| 项目主页 | [DexGrasp-Zero](https://yliangwu.github.io/DexGrasp-Zero/docs/) |
| 代码 | 项目页标注 **Code (coming soon)**；截至 2026-08-12 未找到官方公开仓库 |
| 数据 | 论文承诺 release code and datasets；截至审查日没有项目页下载入口 |
| 正文版本 | arXiv:2603.16806v2，2026-03-18 |
| 审查日期 | 2026-08-12 |

证据标签：**[论文明确]**、**[作者材料]**、**[代码明确]**、**[合理推断]**、**[证据不足]**。本笔记不会把尚未发布的 code/data 承诺标成开源。

## 一、论文速览

### 一句话总结

> DexGrasp-Zero 把不同灵巧手按解剖功能节点和三轴运动原语对齐，用 URDF 物理先验条件化一个共享 GCN，再通过固定的手专用索引映射输出关节命令，实现未见手无梯度更新抓取。

### Elevator pitch

传统跨手抓取策略常输出 fingertip/MANO 等中间目标，再为每只手做 retargeting；目标可能不可达，且误差在映射中累积。DexGrasp-Zero 先把每只手的 kinematic tree 分组为 wrist/metacarpal/proximal/middle/distal/fingertip 等语义节点，再为每节点定义 FLEX/ABD/ROT 三种运动原语，使输入 topology 和输出 action semantics 都可比较。[[MAGCN]] 每层注入从 URDF 提取的 link length、axis、limit、velocity、damping 等物理先验，并用 activation mask 阻止不可执行原语。四手联合训练后在未见 LEAP/Inspire 上仿真平均 85%，蒸馏后的 real policy 在 LEAP/Inspire/Revo2 与十个未见物体上平均 82%；但新手仍要构图、语义分组、仿真 unit excitation 构造 $\mathcal M_h$、相机/控制器/预抓取设置，故属于 policy zero-shot + target configuration，而非即插即用 system zero-shot。

### 迁移类型

| 类型 | 判断 | 说明 |
|---|---|---|
| 人类数据 → 机器人 | 不适用 | 无 human demonstration。 |
| 人类交互 → 多机器人 | 不适用 | 单手抓取。 |
| 已见机器人 → 未见机器人 | **核心** | Allegro/Shadow/Ability/Schunk → LEAP/Inspire；另测 Revo2/Barrett。 |
| 跨模态迁移 | **是** | 仿真 privileged contact/force → 真机 vision/proprio history；URDF physical metadata 注入。 |
| 仿真 → 真机 | **核心** | PPO teacher → BC + RL student；RGB-D/SAM2、LSTM 和 real control stack。 |
| 跨对象 | **是但分协议** | YCB embodiment test 固定相同 45 objects；PartNet train/test 分离；真机 10 unseen objects。 |

### 核心判断

| 项目 | 结论 |
|---|---|
| Source domain | RaiSim 中四只 seen hands × 45 YCB objects 的 multi-hand PPO；单手 transfer 另用 GraspXL/ShapeNet/PartNet。 |
| Target domain | 未参与联合训练的 LEAP/Inspire（仿真）；LEAP/Inspire/Revo2（真机）；Barrett 3-finger（附录仿真）。 |
| 迁移资产 | 共享 policy parameters、graph representation、motion-primitive action semantics、reward/training recipe。 |
| target-specific 资产 | URDF、semantic graph、activation mask、$\mathcal M_h$、PD/velocity control、安全 clipping、相机标定、预抓取 pose。 |
| 核心 gap | variable DoF/topology/joint semantics/actuation limits/link scale，使 state/action tensor 和同一动作的物理含义不一致。 |
| 最关键机制 | semantic functional-unit graph + motion primitives；消融显示它比 URDF prior 本身更关键。 |
| 最大贡献 | 同一个变长 graph policy 能对 unseen topology 输出可映射 joint increments，且给出 real oracle gap。 |
| 最应质疑 claim | “lossless/end-to-end/direct physical commands”：仍有固定 $\mathcal M_h$，语义分组会压缩 joint-level 信息，真机还有 scripted lift。 |
| 精读 | **值得**：representation/action alignment 很有借鉴性。 |
| 复现 | **暂时有条件**：代码/数据尚未公开；RaiSim、hand assets 和 real setup 门槛高。 |

## 二、问题形式化与 zero-shot 定义

令训练/测试本体集合互斥：

$$
\mathcal H_{train}=\{\text{Allegro, Shadow, Ability, Schunk}\},
\qquad
\mathcal H_{test}=\{\text{LEAP, Inspire}\}.
$$

对每个 hand-conditioned MDP，策略循环为

$$
\alpha_t^h\sim\pi_\theta(\cdot\mid s_t^h),
\qquad
\alpha_{physical,t}^h=\mathcal M_h(\alpha_t^h),
\tag{1}
$$

训练目标为

$$
\max_\theta\;
\mathbb E_{h\sim\mathcal H_{train}}
\left[\sum_{t=0}^{T}\gamma^t
r(s_t^h,\alpha_{physical,t}^h)\right].
\tag{2}
$$

| 维度 | Source | Target | 是否真正 held out |
|---|---|---|---|
| 本体 | 4 heterogeneous hands | LEAP/Inspire；real Revo2；Barrett | policy training 中 held out；但 target URDF/mapping/config 被使用。 |
| 观测 | graph node/global state，含 privileged contact/force（teacher） | 同 schema；student 无 tactile/contact，使用 history | modality held out 由 distillation 处理。 |
| 动作 | variable-length primitive vector → training-hand joints | target primitive vector → target joints | shared decoder weights；mapping 是 target-specific。 |
| 动力学 | RaiSim 多手动力学 | target simulation/real dynamics | 不完全 held out：real control 与初始化被专门配置。 |
| 任务 | approach-grasp-lift | 同一抓取任务 | 任务未 held out。 |
| 对象 | YCB 45 objects 同时 train/test；PartNet split；real 10 unseen | 依协议 | 主 85% 是 hand held-out，不是 object held-out。 |
| 训练可用 target 信息 | target URDF、semantic nodes、mapping、evaluation simulator | 是 | 允许本体描述，不允许 gradient update。 |
| 测试可用信息 | RGB-D/SAM2 point cloud、joint state、URDF graph、fixed initial perception | 是 | 不是 raw-sensor end-to-end generalization。 |

共享不变量是“手指功能单元沿掌—指链的拓扑”和“FLEX/ABD/ROT 的局部动作语义”。前者对拟人手具有解剖/运动学依据；Barrett 结果说明可扩到三指非拟人手，但仍需人为把其三指命名为 thumb/middle/ring，远未证明任意软体、腱驱、parallel-jaw 或 continuum end-effector。

## 三、迁移账本：Source → Alignment → Target

| 阶段 | 输入 | 输出 | 域差异 | 对齐机制 | 学习 | target-specific | 证据 |
|---|---|---|---|---:|---:|---:|---|
| 资产准备 | hand URDF/physics asset | joint chains/limits/axes/link vectors | topology、DoF、scale | URDF parse | 否 | 是 | [论文明确] |
| 语义构图 | kinematic chain | functional-unit graph | joints 与 anatomical units 不一一对应 | manual/heuristic functional partition | 否 | **是** | [论文明确] |
| 状态表示 | joint/object/contact state | node matrix + adjacency + global feature | variable nodes/channels | morphology-aligned graph | 否 | graph 是 | [论文明确] |
| 动作表示 | desired control | 3 primitives/node + wrist 6-DoF | joint axes/DoF 不同 | semantic motion primitives | 是 | mask 是 | [论文明确] |
| action mapping | primitives | joint increments | command ordering/sign/axis | fixed sparse $\mathcal M_h$ | 否 | **是** | [论文明确] |
| physical conditioning | URDF features | $E_p^h$ | limits/link length/damping | layer-wise injection | 是 | input 是 | [论文明确] |
| policy learning | multi-hand rollouts | shared MAGCN PPO | data imbalance/morphology | parallel envs per hand-object | 是 | 否 | [论文明确] |
| privileged transfer | teacher contact/force | LSTM student | real has no tactile | BC MSE then RL | 是 | 否 | [论文明确] |
| real perception | initial RGB-D | fixed object point cloud/centroid | occlusion/noise | SAM2 + initial cloud reuse | 部分 | **是** | [论文明确] |
| low-level control | joint/wrist increments | PD/velocity targets + scripted lift | actuator/controller | common gains/clipping + adapter | 否 | 是 | [论文明确] |

最大信息损失在 **functional partition + one primitive per physical DoF 的 indexing rule**：它对齐语义，却可能忽略 coupled joints、tendon transmission、underactuation 与 non-orthogonal axes。迁移上限取决于 canonical graph/action 是否能表达目标手的真实可控子空间。SAM2、固定初始 cloud、标准 pre-grasp、Kalman filter 和 scripted lift 是目标域系统工程，不是跨形态算法。

## 四、源数据与训练数据审查

### 数据/benchmark

| 协议 | 训练对象 | 测试对象 | 手 | 评估 |
|---|---|---|---|---|
| CrossDex/YCB | 同一 45 YCB objects | 同一 45 objects | 4 seen → 2 unseen | 每 object 25 trials；lift 0.5 m + hold 2 s |
| GraspXL | 26 ShapeNet + 32 PartNet | 48 PartNet test | single source hand → all hands | success rate |
| Real | simulation teacher/student | 10 unseen household objects | LEAP/Inspire/Revo2 | 5 poses/object/platform = 150 trials；lift 30 cm + hold 5 s |

这不是 demonstration dataset，而是在线 simulation rollout。每个 `(hand, object)` 有 3 parallel envs，400 Hz physics/20 Hz control，episode 120 approach/grasp steps + 30 lift steps，训练 6000 rounds。失败 rollouts 由 PPO 自然利用，不需人工标注。对象 mesh 来自公开 YCB/ShapeNet/PartNet protocol；论文没有给总 transition 数、wall-clock、随机种子训练成本或不同手样本是否严格等权。

主 85% claim 固定 object set，因此干净地测 morphology transfer，却不测 hand×object 双外推。真机十物体包含 wine glass、bottles、toy dog/mug/hammer、tennis ball、Rubik's cube、orange；每手每物体只有 5 次，置信区间未报告。

## 五、统一表示与动作对齐

### 5.1 Morphology-aligned state graph

每只手 $h$ 构成 $\mathcal G_h=(\mathcal V_h,\mathcal E_h)$；节点数 $N_h$ 可变，节点类型为 fingertip/distal/middle/proximal/metacarpal/wrist，边来自 kinematic tree：

$$
X_{node}^h=[x_1^h,\ldots,x_{N_h}^h]^\top
\in\mathbb R^{N_h\times d_{node}},
\tag{3}
$$

$$
s^h=(X_{node}^h,A^h,x_g^h).
\tag{4}
$$

实际 node feature 为

$$
x_i^h=[d_i^h,\theta_i^h,\dot\theta_i^h,c_i^h,f_i^h,m_i^h,n_i^h],
\tag{10}
$$

其中 $d_i$ 是 node 到 object point cloud 最近点的 3D vector；$\theta_i,\dot\theta_i$ 表达在 primitive axes；$c_i,f_i$ 是 contact/force；$m_i,n_i$ 是 finger class 与 node type one-hot。global feature 为

$$
x_g^h=[\Delta p_{target}^h,v_{wrist}^h,\omega_{wrist}^h,v_{obj}^h,\omega_{obj}^h].
\tag{11}
$$

### 5.2 Hand-agnostic motion primitives

每节点输出

$$
\alpha_i^h=[\Delta_{flex},\Delta_{abd},\Delta_{rot}]^\top\in\mathbb R^3,
\tag{5}
$$

整手 action 为

$$
\alpha_{prim}^h=
[\Delta p_w^{h\top},\Delta\theta_w^{h\top},
\alpha_1^{h\top},\ldots,\alpha_{N_h}^{h\top}]^\top
\in\mathbb R^{6+3N_h}.
\tag{6}
$$

固定 target mapping：

$$
\Delta q^h=\mathcal M_h(\alpha_{prim}^h),
\tag{7}
$$

$$
\Delta q_j^h=s_j^h\alpha_{n_j,(p_j)}^h,
\quad p_j\in\{FLEX,ABD,ROT\},\;s_j^h\in\{-1,+1\},
\tag{8/28}
$$

$$
\alpha_{physical}^h=
[\Delta p_w^\top,\Delta\theta_w^\top,\Delta q^{h\top}]^\top.
\tag{9}
$$

$\mathcal M_h$ 通过 physics engine 中逐 joint unit excitation 得出 primitive type 和 sign；node assignment 来自 functional partition。它不训练 retargeter，但依然是 retarget-like, target-specific control adapter。

> 这篇论文的迁移能力主要来自 **state topology 与 action semantics 的显式对齐**，而不是简单地扩大训练手或对象数量。

## 六、物理属性注入与 MAGCN

URDF physical node feature：

$$
x_j^{physical,h}=[\ell_j^h,a_j^h,v_j^h,\tau_j^h,l_j^h],
\tag{12}
$$

其中 $\ell\in\mathbb R^6$ 是三 primitive axes 的上下限，$a\in\mathbb R^9$ 是三轴 3D directions，$v\in\mathbb R^6$ 是 velocity bounds，$\tau\in\mathbb R^3$ 是 damping，$l\in\mathbb R^3$ 是 parent→node link vector：

$$
X_{physical}^h=[x_1^{physical,h},\ldots,x_{N_h}^{physical,h}]^\top.
\tag{13}
$$

可执行 primitive mask：

$$
m_i^h=[m_{i,FLEX}^h,m_{i,ABD}^h,m_{i,ROT}^h]^\top\in\{0,1\}^3,
\tag{14}
$$

$$
M_{activation}^h=[m_1^h,\ldots,m_{N_h}^h]^\top\in\{0,1\}^{N_h\times3}.
\tag{15}
$$

global/node encoders：

$$
E_g^h=\phi_g(x_g^h),\qquad
E_{node}^h=\phi_{node}(X_{node}^h,A^h).
\tag{16}
$$

physical MLP 与 stacking：

$$
e_j^{p,h}=\phi_p(x_j^{physical,h})\in\mathbb R^{d_p},
\tag{17}
$$

$$
E_p^h=[e_1^{p,h},\ldots,e_{N_h}^{p,h}]^\top.
\tag{18}
$$

每个 GCN layer 都重新拼入 physical prior：

$$
E_{node}^h=\phi_{node}(X_{node}^h,A^h,E_p^h),
\tag{19}
$$

$$
H^{h,(0)}=X_{node}^h,
\tag{20}
$$

$$
Z^{h,(l)}=\operatorname{concat}(H^{h,(l-1)},E_p^h),
\tag{21}
$$

$$
H^{h,(l)}=\sigma\!\left(
\operatorname{LN}(\hat A^hZ^{h,(l)}W^{(l)})
\right),\quad l=1,\ldots,L,
\tag{22}
$$

$$
E_{node}^h=H^{h,(L)}.
\tag{23}
$$

这里 $\hat A=\tilde D^{-1/2}\tilde A\tilde D^{-1/2}$，$\tilde A=A+I$。decoder 为

$$
\tilde e_i^{node,h}=\operatorname{concat}(E_{node}^h[i],M_{activation}^h[i]),
\tag{24}
$$

$$
\alpha_i^h=\phi_{dec,node}(\tilde e_i^{node,h})\in\mathbb R^3,
\tag{25}
$$

$$
(\Delta p_w^h,\Delta\theta_w^h)=
\phi_{dec,wrist}(\operatorname{concat}(E_g^h,e_{wrist}^h)).
\tag{26}
$$

10-layer GCN：第一层 hidden 128，余层 256，ReLU+LayerNorm，无 dropout；physical MLP hidden 32/LeakyReLU；global MLP hidden 256；actor heads hidden 128；critic shares encoder。

## 七、观测与多模态迁移

| 模态 | teacher/source | real student/target | 对齐 | 缺失处理 |
|---|---:|---:|---|---|
| RGB | 仿真/感知条件未详 | RGB-D camera | SAM2 segmentation | 初始 frame 固定复用 |
| depth/point cloud | object geometry | 是 | palm-frame nearest-point vectors | 固定初始 cloud 降遮挡噪声 |
| proprioception | joint pos/vel, wrist state | 是 | primitive-axis/node graph | 5-step history |
| object pose/velocity | global feature/privileged | object centroid；假设静止 | palm frame | stationary-object assumption |
| force/contact | **是，teacher privileged** | 否 | node-aligned labels | LSTM implicit reconstruction |
| tactile | 仿真 contact impulse 可视为 tactile-like | 无真实 tactile | privileged distillation | 无直接反馈 |
| URDF/physical | 是 | 是 | aligned physical graph | 必需 |
| language | 否 | 否 | 不适用 | — |

这是跨模态 transfer：teacher 的 contact/force 不能在真机观测，student 用单层 LSTM hidden 256 处理最近 5 steps 的 visual/proprio history。student 以 teacher weights 初始化，先用 action MSE behavior cloning，只蒸馏动作、不蒸馏 value/entropy，再转 RL。**[论文明确]** 没有报告 modality dropout、sensor delay/noise、SAM2 segmentation failure 或真实 tactile 对照。

## 八、动作接口、控制与物理可执行性

$$
\text{graph state}
\rightarrow
\text{primitive action}
\rightarrow
\mathcal M_h
\rightarrow
\Delta q + \Delta wrist
\rightarrow
\text{PD/velocity controller}
\rightarrow
\text{scripted lift}.
$$

finger action clip/gain 0.015，arm 0.01；policy 20 Hz；wrist 使用高频 velocity controller + Kalman filter；130 policy steps 后执行 scripted lift。关节限位和 primitive feasibility 通过 URDF feature、mask、penalty 和 clipping 多层处理，但自碰撞/环境碰撞/torque limit 在真机 controller 中的硬约束细节未公开。

它不是传统 learned retargeting，却仍需 deterministic target mapping。相同 primitive 在不同手上有相近局部运动语义，不保证相同 fingertip displacement、force 或 object wrench。policy-level zero-shot 成立；controller/system-level strict zero-shot 不成立。

## 九、Reward、训练与可执行性约束

$$
r=r_{grasp}+r_{pen},
\tag{27}
$$

$$
\begin{aligned}
r_{grasp}={}&w_{dis}r_{dis}+w_{contact}r_{contact}
+w_{force}r_{force}+w_{reg}r_{reg},\\
r_{dis}={}&-\sum_i\|d_i^h\|_2,\qquad
r_{contact}=\sum_i c_i^h,\\
r_{force}={}&-\sum_i\max(0,f_i^h-f_0)^2,\qquad
r_{reg}=-\|\Delta q^h\|_2,
\end{aligned}
$$

$$
r_{pen}=-w_{pen}\sum_i
\|(1-m_i^h)\odot\alpha_i^h\|_2^2.
$$

所有手共享 $w_{dis}=0.3,w_{contact}=1.0,w_{force}=0.5,w_{reg}=1.5,w_{pen}=0.3$。这比“只给 action mask”更强：策略被 reward 显式训练避免 inactive axes。它提供物理可执行性 inductive bias，却不保证 dynamic feasibility；特别是 underactuation、compliance、tendon coupling 和 friction mismatch 没进入 canonical action。

## 十、训练策略与数据混合

1. 从各手 URDF 建 semantic/physical graph 与 $\mathcal M_h$。
2. 每个 hand-object 默认 3 parallel envs，multi-hand/multi-object PPO 联合训练。
3. 400 Hz physics、20 Hz control；120 grasp + 30 lift steps；6000 rounds；RTX 3090。
4. privileged teacher 使用 contact states/impulses。
5. student 继承 teacher，5-step LSTM，action MSE BC 后继续 RL。
6. real 使用 SAM2 initial segmentation、fixed cloud、standard pre-grasp、policy 130 steps、scripted lift。

没有 embodiment ID 或 embodiment-specific learned head；physical graph 与 variable node count承担 conditioning。论文没有报告 hand sampler 是否按 env 数严格平衡、加入手数量 scaling curve、negative transfer、target hand 是否用于 model selection。单手实验显示 transfer 与形态相似性强相关，说明 multi-hand diversity 确实重要。

### Table IV：Training and architecture hyperparameters（完整）

| Parameter | Value |
|---|---|
| RL algorithm | PPO |
| Discount $\gamma$ | 0.996 |
| GAE $\lambda$ | 0.95 |
| Clip $\epsilon$ | 0.2 |
| Optimizer | Adam |
| Learning rate | $5\times10^{-4}$, adaptive KL schedule |
| Max gradient norm | 0.5 |
| Batch size | $N_{env}\times130$ transitions/update，4 mini-batches |
| PPO epochs/update | 4 |
| Envs | $N_{hand}\times N_{obj}\times N_{repeat}$，default $N_{repeat}=3$ |
| Frequency | 400 Hz physics / 20 Hz control |
| Episode | 120 + 30 lift steps |
| Reward weights | 0.3 / 1.0 / 0.5 / 1.5 / 0.3 |

## 十一、Zero-shot 纯度审查

| 项目 | 使用 | 违反 policy zero-shot | 说明 |
|---|---:|---:|---|
| 目标域训练 trajectory | 否 | 否 | joint policy 不在 target hand 上更新。 |
| target reward rollout | 评估有；是否用于调参未知 | 证据不足 | target simulator 必然用于报告。 |
| finetuning | 否 | 否 | 明确 without finetuning。 |
| target URDF | 是 | 否（定义允许） | physical graph 必需。 |
| 人工语义 mapping | 是 | 否，但降低纯度 | functional partition/three-finger naming。 |
| target action mapping | 是，$\mathcal M_h$ | 否，但强工程 | unit excitation / indexing。 |
| target controller | 是 | 否 | hand/arm interfaces、clip、filter。 |
| target DR/system ID | 未明确 | 证据不足 | real platform 配置显然存在。 |
| 真机参数/初始化 | 是 | 否 | thumb max-open、25 cm offset。 |
| target checkpoint selection | 未报告 | 证据不足 | 代码未公开无法审查。 |

分类：**policy zero-shot，但需要本体配置与 target-specific mapping/controller**。若 claim 被理解为“完全无需新手工程”，则不成立；若定义是“不用目标手训练/梯度更新”，主仿真实验成立。Revo2 是否也完全未参与任何 distillation/config tuning，论文文字暗示 cross-hand deployment，但缺代码与训练清单，纯度低于 LEAP/Inspire 的 formal split。

## 十二、实验是否证明 transfer

### Table I：Multi-hand cross-embodiment（完整）

| Method/Variant | Allegro | Shadow | Ability | Schunk | LEAP unseen | Inspire unseen | Seen avg | Unseen avg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CrossDex per-object | 0.81 | 0.85 | 0.90 | 0.90 | 0.34 | 0.44 | 0.865 | 0.390 |
| CrossDex multi-object | 0.39 | 0.69 | 0.42 | 0.60 | 0.19 | 0.34 | 0.525 | 0.265 |
| Ours w/o primitives | 0.52 | 0.51 | 0.64 | 0.59 | 0.39 | 0.29 | 0.565 | 0.340 |
| Ours early fusion | 0.42 | 0.47 | 0.59 | 0.54 | 0.46 | 0.34 | 0.505 | 0.400 |
| Ours w/o physical priors | 0.91 | 0.89 | 0.90 | 0.84 | 0.82 | 0.79 | 0.885 | 0.805 |
| Ours w/o mask & penalty | 0.92 | 0.91 | 0.90 | 0.81 | 0.50 | 0.76 | 0.885 | 0.630 |
| **Ours full** | **0.92** | **0.95** | **0.90** | **0.91** | **0.93** | **0.82** | **0.920** | **0.850** |

59.5 percentage points 的 improvement 是相对 CrossDex multi-object unseen avg（26.5→85.0），不是相对 per-object（39→85）。motion primitives 贡献最大（85→34），mask+penalty 次之（85→63），physical prior 单独贡献较小（85→80.5）。

### Table II：Single-hand train → all hands（完整）

| Train hand | Allegro | Shadow | Ability | Schunk | LEAP | Inspire |
|---|---:|---:|---:|---:|---:|---:|
| GraspXL specialists | 0.94 | 0.93 | 0.91 | 0.90 | 0.95 | 0.91 |
| Allegro | **0.93†** | 0.83 | 0.80 | 0.69 | 0.48 | 0.77 |
| Shadow | 0.55 | **0.98†** | 0.80 | 0.82 | 0.77 | 0.94 |
| Ability | 0.60 | 0.60 | **0.86†** | 0.86 | 0.88 | 0.90 |
| Schunk | 0.61 | 0.52 | 0.83 | **0.92†** | 0.90 | 0.94 |
| LEAP | 0.50 | 0.68 | 0.65 | 0.69 | **0.90†** | 0.69 |
| Inspire | 0.61 | 0.83 | 0.87 | 0.88 | 0.88 | **0.96†** |

† in-domain。该表诚实显示 morphology similarity 影响很大：Allegro→LEAP 仅 0.48，不能把 representation 理解为完全 morphology invariant。

### Table III：真机 zero-shot vs oracle（完整）

| Method | LEAP | Inspire | Revo2 | Avg |
|---|---:|---:|---:|---:|
| Intra-hand oracle | 0.90 | 0.90 | 0.78 | 0.86 |
| Cross-hand w/o physical graph | 0.84 | 0.80 | 0.62 | 0.75 |
| **Cross-hand full** | **0.88** | **0.86** | **0.72** | **0.82** |

oracle gap 4 points 很有说服力；但 oracle 的训练数据/预算细节未在表中展开，且 real pipeline 含 target setup、SAM2、scripted lift。

### Table V：每物体真机结果（完整）

| Object | LEAP | Inspire | Revo2 |
|---|---:|---:|---:|
| wine-glass | 5/5 | 5/5 | 4/5 |
| beverage bottle | 4/5 | 3/5 | 4/5 |
| spray bottle | 4/5 | 4/5 | 4/5 |
| squeeze sauce bottle | 4/5 | 5/5 | 4/5 |
| toy dog | 5/5 | 5/5 | 2/5 |
| coffee mug | 5/5 | 5/5 | 4/5 |
| plastic toy hammer | 4/5 | 5/5 | 2/5 |
| tennis ball | 4/5 | 3/5 | 5/5 |
| Rubik's cube | 5/5 | 4/5 | 3/5 |
| orange | 4/5 | 4/5 | 4/5 |
| **Average** | **0.88** | **0.86** | **0.72** |

实验真正证明了 shared representation/policy 对 held-out hand 有 transfer gain；还证明物理先验被利用（sensitivity test）以及非拟人 Barrett 可工作。没有证明 dynamic manipulation、in-hand reorientation、clutter、moving objects、tactile recovery 或 arbitrary morphology。

## 十三、消融与附加表格

### Table VI：URDF prior sensitivity（完整）

| Allegro encoded link scale | SR |
|---:|---:|
| 1 | 0.92 |
| 2 | 0.85 |
| 1/4 | 0.87 |

只改变输入编码、不改 simulator geometry/dynamics，性能和 enclosure timing 随 scale 变，证明网络不是忽略 URDF；但 scale 扰动很粗糙，缺 axis/limit/noise sensitivity。

### Table VII：Backbone comparison，3 seeds（完整）

| Backbone | Train-hand SR (%) | Unseen-hand SR (%) |
|---|---|---|
| GCN | 91.3/92.1/92.5（91.9±0.6） | 86.1/84.9/85.5（85.5±0.6） |
| Graph Transformer | 86.4/87.2/86.9（86.8±0.4） | 81.5/80.4/79.8（80.6±0.9） |

缺失消融：去 semantic grouping、自动 vs 人工 mapping、错误 joint axis/limit、不同手数量 scaling、无 teacher/student、无 LSTM、different controller gains、moving object、实时 point cloud update、tactile student、target mapping cost/time、soft/underactuated hands。

## 十四、关键图（Figure 1–12，完整索引）

### Figure 1：Paradigm comparison

![Figure 1](https://yliangwu.github.io/DexGrasp-Zero/docs/webpage/assets/fig1.png)

作者把 prior retargeting 与 fixed mapping 区分；批判点是 $\mathcal M_h$ 仍承担 target-specific action conversion。

### Figure 2：Universal hand representation

![Figure 2](https://arxiv.org/html/2603.16806v2/x1.png)

展示 functional nodes 与 FLEX/ABD/ROT，是表示不变量的来源。

### Figure 3：MAGCN architecture

![Figure 3](https://arxiv.org/html/2603.16806v2/x2.png)

state graph、physical graph、layer-wise fusion、masked decoder 到 mapping 的完整数据流。

### Figure 4：Real hardware setups

![Figure 4](https://arxiv.org/html/2603.16806v2/x3.png)

Kinova+LEAP/Inspire 与 Piper+Revo2；arm 和 hand 同时变化使 real result 更有价值，也引入 controller confound。

### Figure 5：Seen-hand simulated grasps

![Figure 5](https://arxiv.org/html/2603.16806v2/x4.png)

证明共享 policy 对训练手的 basic competence；真正 transfer 证据仍是表 I。

### Figure 6：All-hand graph construction

![Figure 6](https://arxiv.org/html/2603.16806v2/x5.png)

用于审计 semantic nodes 是否真正对齐；特殊 joint grouping 暴露人工判断。

### Figure 7：Graph construction pipeline

![Figure 7](https://arxiv.org/html/2603.16806v2/x6.png)

URDF parsing→functional partition→graph instantiation；新本体接入不是零配置。

### Figure 8：Joint-to-primitive mapping

![Figure 8](https://arxiv.org/html/2603.16806v2/x7.png)

每个 physical DoF 被分配到 primitive/sign，直接体现 $\mathcal M_h$ 的 engineering footprint。

### Figure 9：十个真机物体

![Figure 9](https://arxiv.org/html/2603.16806v2/figs/obj_selected.png)

覆盖尺寸/刚柔差异，但无 clutter、透明/反光感知分组统计。

### Figure 10：Training curves

![Figure 10](https://arxiv.org/html/2603.16806v2/figs/training_curve.png)

多 seed 收敛定性稳定；没有给 compute-normalized sample efficiency。

### Figure 11：Real failure cases

![Figure 11](https://arxiv.org/html/2603.16806v2/x8.png)

小物体 empty grasp 与 Revo2 limited thumb/size failure 表明缺 tactile 和 hardware capacity 不能靠 representation 消除。

### Figure 12：Barrett graph 与 zero-shot grasps

![Figure 12](https://arxiv.org/html/2603.16806v2/x9.png)

四拟人训练手→三指 8-DoF Barrett 为结构外推的最好证据，SR=0.70；但依然人工命名三指与构造 mapping。

## 十五、灵巧手多模态/跨形态专项判断

1. variable DoF/node count 由 GNN 支持；raw network topology 不随手改变。
2. node semantics 来自人为定义的 anatomy/function，URDF 不能完全自动决定“proximal”等角色。
3. action 是 hand-agnostic primitive，但 joint mapping 固定且 target-specific。
4. URDF 物理属性既是 input modality，也是人工结构先验；layer-wise fusion 优于 early fusion。
5. 不可执行 primitive 由 activation mask + reward penalty 处理。
6. 多手联合训练显著强于不稳定的 single-hand transfer，但缺训练手数量 scaling。
7. LEAP/Inspire 是真正 held-out hands；主 YCB objects 不是 held-out。
8. Barrett 测试了 topology 差异；soft/underactuated/tendon hands 未测。
9. 真机 student 不用真实 tactile；所谓“visual-tactile teacher”是 privileged sim modality，真实系统仍 vision/proprio only。
10. real unseen objects 有效，但 perception 被简化为 SAM2 initial cloud + stationary object + scripted lift。

## 十六、局限、失败模式与五个组会问题

主要局限：canonical primitive 仍是人为本体 ontology；固定 sparse mapping 忽略非线性/coupled actuation；新手需 physics engine unit excitation；main unseen-hand benchmark objects 与训练相同；real pipeline 强初始化、静态对象、固定 point cloud、scripted lift；无 tactile recovery；代码/数据未发布；未给 mapping labor/time、negative transfer 或 large topology extrapolation curve。

组会问题：

1. 如果 $\mathcal M_h$ 是核心 target-specific adapter，为什么称“directly outputs physical commands/end-to-end”而不是 deterministic retargeting？
2. functional partition 对每只手需要多少人工分钟、多少争议 joint grouping？不同标注者结果是否一致？
3. 在 YCB 主表中对象完全重叠，hand×object 双外推会从 85% 降到多少？
4. 对 tendon-driven/underactuated/compliant hand，三轴 primitive 和 sparse linear mapping 是否仍成立？
5. 去掉 scripted lift、固定初始 cloud 与标准 25 cm pre-grasp 后，真实闭环成功率还有多少？

## 十七、代码、数据与复现审查

### 仓库识别（截至 2026-08-12）

| 资产 | 类型 | 状态 | 完整度/可信度 | 备注 |
|---|---|---|---|---|
| [项目主页](https://yliangwu.github.io/DexGrasp-Zero/docs/) | 作者官方 | 在线 | 高/高 | figures、videos、结果表、arXiv。 |
| Code | 官方承诺 | **Coming soon** | 低/高 | 没有可审查 repository。 |
| Dataset/assets/checkpoints | 官方承诺 | 未找到下载 | 低/中 | 论文称 will release。 |
| CrossDex | baseline 官方代码 | 已有 | 未在本次本地运行 | 可作复现底座。 |
| GraspXL/RobustDexGrasp | baseline/implementation basis | 论文引用 | 中 | RL 与 distillation 建立于其上。 |

因此无法从代码确认：hand URDF/assets、semantic node YAML、$\mathcal M_h$ 生成器、reward、multi-hand sampler、teacher/student checkpoint、real robot drivers、SAM2 config、实验 seeds 与 exact evaluation scripts。复现可行性评分不能高于中低。

### 新手接入成本

需要准备 URDF 与 simulator asset；确认 kinematic tree；把 joints 分到 functional nodes；运行 unit excitation 得 primitive/sign；生成 activation mask/physical graph；配置 controller ordering、limits、gains、安全 clip；相机→base transform；SAM2/point cloud；pre-grasp pose；验证关节方向和 enclosure timing。理论上无需新训练/reward/network change，但需要 simulator validation 和 real calibration。刚性拟人手估计数天到 1–2 周；coupled/soft/underactuated hand 风险显著更高。

## 十八、分层复现路线

### Level 1：最小迁移验证

- 等代码发布；或基于 GraspXL/CrossDex 实现两个 hands。
- 只做 semantic graph、3 primitives、$\mathcal M_h$ unit test。
- 一只 source hand 训练、另一只 held-out hand evaluate；固定 3–5 YCB objects。
- 成功标准：mapping round-trip 正确、inactive primitive 为零、zero-shot SR 显著高于 raw joint baseline。

### Level 2：论文级复现

- 六手 RaiSim assets、45 YCB + PartNet protocol、PPO 6000 rounds、多 seeds。
- 复现 Tables I/II/VI/VII；加 target mapping labor log。
- teacher/student distillation；至少一只 real hand 50 trials。
- 最大风险：未公开 asset/controller/reward details 与 simulator contact differences。

### Level 3：平台化扩展

```text
RobotHandMetadata = {
  urdf, semantic_nodes[], joint_to_node[], primitive_axis[], sign[],
  joint_limits, velocity_limits, damping, link_vectors, activation_mask
}
GraphObservation = {node_state[N,d], adjacency[N,N], global_state, physical_state[N,p]}
PrimitiveAction = {wrist_delta_SE3, node_primitives[N,3]}
HandAdapter.map(PrimitiveAction) -> JointCommand[L]
```

回归测试：每 joint excitation direction、node grouping schema validation、wrong-URDF sensitivity、random target topology、controller saturation、object scale、tactile dropout、mapping determinism 与 unseen-hand split leakage。

## 十九、与迁移范式对比

| 维度 | DexGrasp-Zero | 人类示范迁移 | 交互迁移 | 跨形态统一策略 |
|---|---|---|---|---|
| Source data | multi-hand sim RL | human video/mocap | paired human motion | multi-robot rollout |
| Target | unseen hand | robot | dual robots | held-out robot |
| gap | DoF/topology/limits | visual/action | contact/topology/dynamics | morphology/dynamics |
| unified rep | anatomy graph+URDF graph | task/EEF | interaction/contact graph | morphology graph/token |
| action | FLEX/ABD/ROT + wrist | task-space ref | paired joint/ref | shared primitive |
| retargeting | 无 learned retargeter；有 fixed mapping | 常有 | 必需 | fixed adapter 常有 |
| target mapping | **是** | 常有 | 是 | 是 |
| contact | teacher node contact/force | 常弱 | 核心 | 本文显式 |
| joint training | 4 hands | 可选 | MAPPO/CTDE | 核心 |
| zero-shot | policy zero-shot + config | data transfer | 非本体 zero-shot | 典型范式 |
| sim-to-real | privileged distillation | 可能无 | DR+state estimation | 视工作而定 |
| 优势 | state/action 双对齐 | 低成本数据 | coupled dynamics | 新本体复用 |
| 局限 | semantic mapping 与简化 real pipeline | embodiment gap | communication/state | physical extrapolation |

## 二十、最终判断

| 项目 | 判断 |
|---|---|
| 真正迁移了什么 | 共享抓取 policy parameters、functional graph representation、primitive action semantics 与 sim teacher skill。 |
| 能力主要来自 | **表示与 action interface**；GCN 是合适载体，URDF prior 为辅助。 |
| 最值得学习 | functional-unit nodes；primitive activation mask+penalty；layer-wise physical injection。 |
| 最不应高估 | “lossless”；“无需 target mapping”；“85% 同时证明 unseen object”。 |
| 数据复用价值 | **中高**：复用 rollout/策略，不复用 raw joint tensor。 |
| 跨本体泛化可信度 | **高（刚性手族内）/中（广义本体）** |
| 真实部署成熟度 | **中**：三平台结果好，但感知/初始化简化。 |
| 代码复现可行性 | **低至中**：代码/数据未发布。 |
| 是否值得精读 | **值得** |
| 是否值得复现 | **谨慎，等待 release；表示层可先独立复现** |
| 是否纳入数据平台 | **有条件值得**：优先纳入 robot-hand metadata 与 primitive schema。 |
| 最适合借鉴 | `URDF → semantic graph → primitive mask/mapping` 的 adapter contract。 |

> 这篇论文本质上是一项从 **多种已见灵巧手的仿真强化学习经验** 到 **未见灵巧手的共享抓取策略** 的迁移工作。它真正建立的不变量是 **按功能单元组织的手部拓扑与 FLEX/ABD/ROT 局部动作语义**，主要通过 **morphology-aligned graph、activation-aware primitive action 和 URDF physical injection** 缩小域差异；其迁移能力上限取决于 **目标本体能否被这套人工语义节点和稀疏 $\mathcal M_h$ 忠实表达**。论文最有价值的是 **同时对齐状态与动作，并用 held-out hands、oracle 和 Barrett 测试给出较强证据**，但需要警惕它仍依赖 **目标 URDF、functional partition、unit-excitation mapping、标准化感知/控制与 scripted lift**，因此 zero-shot claim 应理解为 **无目标手梯度更新的 policy zero-shot，而不是零配置的 system zero-shot**。
