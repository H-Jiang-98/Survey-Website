---
title: "Rhythm: Learning Interactive Whole-Body Control for Dual Humanoids"
method_name: "Rhythm"
authors: [Hongjin Chen, Wei Zhang, Pengfei Li, Shihao Ma, Ke Ma, Yujie Jin, Zijun Xu, Xiaohui Wang, Yupeng Zheng, Zining Wang, Jieru Zhao, Yilun Chen, Wenchao Ding]
year: 2026
venue: "Robotics: Science and Systems (RSS) 2026"
tags: [dual-humanoid, human-human-interaction, cross-embodiment, motion-retargeting, multi-agent-reinforcement-learning, contact-graph, sim-to-real, unitree-g1]
zotero_collection: "_inbox/cross-embodiment"
image_source: online
arxiv_id: "2603.02856"
arxiv_html: "https://arxiv.org/html/2603.02856v2"
created: 2026-08-12
---

# Rhythm: Learning Interactive Whole-Body Control for Dual Humanoids

## 元信息与证据边界

| 项目 | 内容 |
|---|---|
| 作者 | Hongjin Chen, Wei Zhang, Pengfei Li, Shihao Ma, Ke Ma, Yujie Jin, Zijun Xu, Xiaohui Wang, Yupeng Zheng, Zining Wang, Jieru Zhao, Yilun Chen, Wenchao Ding |
| 机构 | Fudan University；TARS Robotics；Tsinghua University；Shanghai Jiao Tong University；Shanghai Innovation Institute；Institute of Automation, CAS |
| 会议 | RSS 2026 |
| 论文 | [RSS 页面](https://roboticsconference.org/program/papers/34/) · [arXiv v2](https://arxiv.org/abs/2603.02856) · [HTML](https://arxiv.org/html/2603.02856v2) |
| 项目主页 | [Rhythm](https://hoshi-no-ai.github.io/Rhythm/) |
| 代码 | 项目页标注 **Code (Coming Soon)**；截至 2026-08-12 未发现官方公开仓库 |
| MAGIC 数据 | 项目页标注 **Dataset (Coming Soon)**；论文写的是 “will publicly release”，截至审查日尚无下载入口 |
| 正文版本 | arXiv:2603.02856v2，2026-05-02 |
| 审查日期 | 2026-08-12 |

证据标签：**[论文明确]**、**[作者材料]**、**[代码明确]**、**[合理推断]**、**[证据不足]**。论文的 “release” 是未来时态；本笔记不把尚未出现的下载按钮等同于已经开源。项目视频可帮助理解动作，但不能替代定量试验。

## 一、论文速览

### 一句话总结

> Rhythm 先把双人动作拆成“各自姿态”和“二者关系”两个尺度流形，优化成双 G1 可执行参考，再以交互图和接触图奖励训练双智能体策略，并用相对定位与软相位同步部署到两台真机。

### Elevator pitch

把两个人各自按身高缩放到同一款机器人，会破坏握手、拥抱等相对几何；把整个双人场景统一缩放，又会迫使某一方脚悬空或关节超限。[[IAMR]] 不在两者间二选一：intra-agent edges 跟踪独立尺度流形，inter-agent edges 跟踪统一尺度流形，近距离边以更高刚度保持交互，再用 joint/collision/foot/trust-region 约束求双机器人姿态。[[IGRL]] 将同一 interaction/contact topology 变成 MAPPO 奖励，同时让每个 actor 看到自身历史、peer 相对状态和双方未来参考。真机端的 LiDAR–IMU 定位、无线 peer state 与连续相位校正把仿真的全局同步状态改造成可部署的 ego-centric 闭环。

### 迁移类型

| 类型 | 判断 | 说明 |
|---|---|---|
| 人类数据 → 机器人策略 | **核心** | MAGIC/Inter-X 双人动作先 retarget，再用于 RL reference。 |
| 人类交互 → 多机器人 | **核心** | human–human interaction → 两台主动 Unitree G1。 |
| 已见机器人 → 未见机器人 | 否 | 训练和测试都是同构 29-DoF G1；没有 held-out morphology。 |
| 跨模态数据迁移 | **是** | optical mocap/SMPL/BVH → robot joints/contact graph；仿真全局态 → LiDAR/IMU/无线相对态。 |
| 仿真 → 真机 | **核心** | MAPPO in simulation → 两台真实 G1，依赖 domain randomization 与部署栈。 |
| 跨数据源 | **是** | MAGIC BVH 与 Inter-X SMPL 均先抽象为全局 3D keypoints。 |
| 多任务/动作 | **有限** | 协同、轻接触、强接触与长时舞蹈；仍是每段 reference-conditioned tracking。 |

### 核心判断

| 项目 | 结论 |
|---|---|
| Source domain | 约 3 小时光学 mocap 双人 MAGIC（BVH）与外部 Inter-X（SMPL）。 |
| Intermediate domain | IAMR 生成的双 G1 29-DoF reference、interaction graph、post-retarget collision contact graph；双机器人仿真。 |
| Target domain | 两台物理 Unitree G1；onboard CPU、LiDAR/IMU、无线 LCM、预建地图。 |
| 真正迁移的资产 | 双人动作语义、相对交互几何、接触时序、reference-conditioned policy parameters。 |
| target-specific 资产 | G1 model/limits/collision geometry、IAMR keypoint/link mapping、contact nodes、sim asset、PD/控制栈、地图与定位、无线协议。 |
| 核心 gap | 人体比例差异导致 self-motion 与 pair geometry 冲突；双主动刚体的接触动力学与异步部分观测。 |
| 最关键机制 | dual reference manifolds + graph topology 从 retargeting 一致传到 RL reward。 |
| 最大贡献 | 将“交互表示—可行参考—耦合策略—双机部署”串成完整真机链，而非只在动画或仿真展示。 |
| 最应质疑 claim | “first robust transfer/unified framework”的适用范围：仅两台同构 G1、三项量化真机任务、预建地图，且没有统计不确定性。 |
| 是否值得精读 | **值得**：交互几何如何跨表示层保持得很清楚。 |
| 是否值得复现 | **当前不宜直接承诺**：代码/数据未公开，两台 G1 和双机定位安全成本很高。 |

## 二、问题形式化与迁移边界

输入是两名人的动作序列。先统一成每个 actor 的全局 3D keypoints：

$$
P^{raw}_{t}={p^{raw,(1)}_{t,i},p^{raw,(2)}_{t,i}}_{i=1}^{N}.
$$

对 BVH 用 skeleton forward kinematics，对 SMPL 用参数人体模型：

$$
p^{raw,(k)}_t=f^{skel}(q^{(k)}_t),qquad
p^{raw,(k)}_t=M(q^{(k)}_t;\beta^{(k)}).
$$

IAMR 输出两台 G1 的可行 joint reference 与拓扑先验：

$$
(Q^{rob}_{1:T},\mathcal G_{inter,1:T},\mathcal G_{contact,1:T})
=\mathrm{IAMR}(P^{raw}_{1:T},\mathcal R_{G1}).
$$

IGRL 在 multi-agent MDP 中训练共享/同构 actor；每台机器人分散执行：

$$
a_t^{(k)}\sim\pi_\theta\!\left(\cdot\mid
o^{(k)}_{prop,t-19:t},o^{(k)}_{peer,t-19:t},o^{(k)}_{ref,t:t+20}\right),
\quad a_t^{(k)}\in\mathbb R^{29}.
$$

| 维度 | Source | Target | 是否 held out / 如何对齐 |
|---|---|---|---|
| 本体 | 两名人，身高/肢体比例可不同 | 两台同构 1.3 m、29-DoF G1 | human→G1 held out，但 G1 model 全程参与；没有 unseen robot test。 |
| 观测 | mocap 全局 keypoints | proprio + peer relative pose/joints + reference | 离线 source 与在线 observation 不同，由 IAMR/reference 间接连接。 |
| 动作 | 人体关节动作 | 29D target joint positions | 不是共享 action space；通过 target-specific retargeting。 |
| 动力学 | 人体接触、无力数据 | 仿真/真机双刚体接触 | 接触力在 simulator reward 中合成，非从人数据迁移。 |
| 任务 | 双人互动片段 | reference-conditioned 双 G1 tracking | 语义迁移；任务本身在 reference 中显式给定。 |
| 训练时 target 信息 | G1 URDF/model、collision、joint limits、sim dynamics | 大量使用 | 因而不是 target-free transfer。 |
| 测试时 target 信息 | — | 自身状态、peer state、未来 reference、预建地图 | 需要 reference phase 和双向通信。 |

Rhythm 所称 transfer 是 **human interaction → fixed robot pair** 与 **simulation → same real pair**。它不证明一种 morphology-agnostic policy 能迁移到未知 humanoid，也不从单人动作组合出未见双人技能。

## 三、迁移账本：Source → Alignment → Target

| 阶段 | 输入 | 输出 | gap | 对齐机制 | 学习 | target-specific | 证据 |
|---|---|---|---|---:|---:|---:|---|
| 格式标准化 | BVH 或 SMPL | raw global keypoints | skeleton/参数格式不同 | FK → common keypoints | 否 | keypoint map 是 | [论文明确] |
| 尺度分解 | 两名人 keypoints | $\mathcal M_{ind},\mathcal M_{uni}$ | 人体高度比例不同 | individual / average global scale | 否 | robot height 是 | [论文明确] |
| 拓扑划分 | pair keypoints | intra/inter edges | self fidelity 与 interaction geometry 冲突 | graph partition | 否 | graph definition 是 | [论文明确] |
| 可行性优化 | 两流形 + G1 model | 29-DoF pair reference | joints、碰撞、足接触 | SQP/OSQP hard constraints | 否 | **强** | [论文明确] |
| contact 标注 | retargeted robot motion | binary contact graph | mocap 无可靠 robot contact | collision detection after retargeting | 否 | collision model 是 | [论文明确] |
| policy learning | reference + topology | decentralized actors | coupled dynamics | CTDE/MAPPO + graph rewards | 是 | G1 sim 是 | [论文明确] |
| sim-to-real | privileged synchronous sim | noisy delayed real obs | latency、参数误差、扰动 | DR、history CNN、velocity estimation | 是 | ranges/PD 是 | [论文明确] |
| localization | LiDAR/IMU + map | global/relative roots | sim global state 不可得 | Point-LIO + GeoTransformer/GICP + KF | 部分算法 | **强** | [论文明确] |
| synchronization | 两台 local phases | corrected phase rates | clocks/packets asynchronous | wireless proportional soft sync | 否 | gain/network 是 | [论文明确] |

最大的离线信息损失发生在 `human skeleton → sparse robot keypoints → 29-DoF q`：手指细节、肌肉顺应性、接触力、个人动力学与视觉语义被丢弃。最大的在线误差累积链是 `map registration → peer global pose → ego relative transform → actor → PD/contact`；相位同步只解决时间漂移，不能纠正错误的几何定位。

## 四、源数据 MAGIC 与数据审查

### 4.1 数据组成

| 属性 | 论文披露 |
|---|---|
| 有效时长 | 约 3 小时 |
| 采集 | high-fidelity optical motion capture，BVH |
| actor | 身高匹配，以减轻原始 anthropometric conflict |
| 序列 | 每段长于 10 s，强调 temporal continuity |
| Coordinated actions | 30.4% |
| Intimate / Care | 24.8% |
| Contact | 18.2% |
| Social Rituals | 13.3% |
| Competitive | 13.3% |
| interaction graph | 从 retargeted motion 的 inter-agent edges 提取 |
| contact label | 在 retargeted G1 kinematics 上 collision detection 后生成 |
| 外部泛化源 | Inter-X（SMPL），用于人体比例差异更大的验证 |

数据不是完全“原始真值”：contact label 不是传感器测得的人体接触，而是 **retarget 后机器人几何碰撞检测**的派生标签。因此它和 IAMR/G1 collision model 强耦合，换机器人时必须重算。论文未披露序列数、actor 数、mocap frame rate、训练/验证/test split、许可、失败/清洗率、每类动作列表或数据体积。约 3 小时虽足以覆盖若干 reference tracking 技能，却不足以单独支持开放世界 multi-humanoid interaction 的表述。

### 4.2 数据可复用性

- raw BVH 对动画、人形 retargeting 有直接价值；retargeted G1 trajectories 对同款 G1 更直接。
- interaction graph 可作为表示层资产；contact graph 应视为 **G1-specific derived data**。
- 人体尺寸被刻意匹配，提高了主数据质量，也降低了对极端 morphology gap 的覆盖；Inter-X 只作为外部评估补充。
- 截至审查日项目页仍显示 Dataset Coming Soon，故无法审计文件 schema、split、license 或运行作者 loader。**[作者材料]**

## 五、IAMR：双参考流形与统一表示

令目标 robot height 为 $h_{robot}$，两名人的身高为 $h_{raw}^{(k)}$：

$$
s^{(k)}=\frac{h_{robot}}{h_{raw}^{(k)}}.
$$

独立尺度保持每个人自身动作可行性：

$$
p_{t,i}^{ind,(k)}=s^{(k)}p_{t,i}^{raw,(k)}.
$$

统一尺度用两个 scale 的平均值保持二者相对位置：

$$
s_{unified}=\frac{s^{(1)}+s^{(2)}}{2},\qquad
p_{t,i}^{uni,(k)}=s_{unified}p_{t,i}^{raw,(k)}.
$$

若只用 $\mathcal M_{ind}$，两边缩放量不同，会出现 “air handshake”；若只用 $\mathcal M_{uni}$，交互比例正确但个体可能脚悬空/姿态不可达。IAMR 将图边分为

$$
\mathcal E=\mathcal E_{self}\;\dot\cup\;\mathcal E_{inter},
$$

让 self edges 参考 $\mathcal M_{ind}$，interaction edges 参考 $\mathcal M_{uni}$。这是论文最重要的 representation decision：**同一帧不是只有一个 canonical geometry，而是按语义边选择不同 canonical geometry。**

对顶点 $p_i$，Laplacian coordinate 为

$$
\mathcal L(p_i)=p_i-\sum_{j\in\mathcal N(i)}c_{ij}p_j.
$$

Laplacian 保留局部相对形状而非绝对坐标，能缓解全局平移/尺度差异；但 keypoint/edge 的人工选择决定了哪些关系被视为不变量。

## 六、IAMR 优化、约束与 retargeting

主问题为

$$
q^*=\arg\min_q\left(E_{self}(q)+E_{inter}(q)\right),
\qquad q\in\mathcal C_{phy}.
$$

自运动项保留 intra-agent Laplacian 与关键骨段旋转：

$$
E_{self}=\sum_{a\in\{1,2\}}\sum_{p_i\in\mathcal V_a}
\|\mathcal L(p_i)-\mathcal L(p_i^{ind})\|^2
+\lambda_{rot}\sum_{a}\sum_{b\in\mathcal B_a}
\|\theta_b\ominus\hat\theta_b^{src}\|^2.
$$

interaction term 跟踪跨机器人相对向量，并让近边更硬：

$$
E_{inter}=\sum_{(i,j)\in\mathcal E_{inter}}
\omega_{ij}(d_{ij})
\|(p_i-p_j)-(\hat p_i^{uni}-\hat p_j^{uni})\|^2,
$$

$$
\omega_{ij}(d_{ij})=\omega_{max}e^{-\gamma d_{ij}}.
$$

附录给出的逐帧目标是

$$
\mathcal J(q_t)=w_{self}\mathcal J_{self}
+w_{inter}\mathcal J_{inter}
+w_{reg}\mathcal J_{reg},
$$

其中 $w_{self}=2.0$、$w_{inter}=10.0$、$w_{reg}=0.1$、$\lambda_{rot}=0.1$。CVXPY 表达、OSQP 求解，frame-by-frame SQP；$\mathcal J_{reg}$ 含 $\|q_t-q_{t-1}\|^2$。

### 硬约束

| 约束 | 形式 | 目的 |
|---|---|---|
| Joint limits | $q^{min}\le q_{t-1}+\Delta q_t\le q^{max}$ | target robot 可达性 |
| Collision | $J_{col}\Delta q_t\ge-\phi(q_{t-1})-\epsilon_{safe}$ | 两机/自身非穿透 |
| Foot contact | $\|J_{foot}\Delta q_t\|\le\epsilon_{stick}$ | source 标注足接触时不滑动 |
| Trust region | $\|\Delta q_t\|_2\le\delta$ | 保证线性化有效 |

这种 retargeting 是 **optimization-based、target-specific、offline**。它没有学习跨本体映射；换 Atlas、H1 或不同手臂比例的 G1 版本，需要替换 model、keypoint/link correspondence、collision pairs/limits 和重新求解。逐帧 SQP 的局部最优、初始化敏感性和处理速度未量化；这直接影响能否大规模处理数据。

## 七、接触与物理交互对齐

IAMR 输出两类先验：

- **Interaction graph**：黄色跨 agent 边，表达关节之间的相对位置向量与近距离权重。
- **Contact graph**：红色 active link/contact nodes，由 retarget 后 collision detection 得到。

二者不能互相替代。interaction edge 接近只代表几何对齐，可能“ghost through”；binary contact 正确也不代表力适中。IGRL 因此将 contact reward 分成 active/inactive：

$$
r_{contact}=\lambda_{act}e^{-E_{act}/\sigma_c^2}
+\lambda_{inact}e^{-E_{inact}/\sigma_c^2},
$$

$$
E_{act}=\sum_{k\in\mathcal V_{act}}
\left[\beta\|C_k^{sim}-1\|+(1-\beta)\mathcal L_{force}(f_k^{sim})\right],
$$

$$
E_{inact}=\sum_{k\notin\mathcal V_{act}}\|C_k^{sim}\|.
$$

力的有效区间惩罚为

$$
\mathcal L_{force}(f)=
\begin{cases}
1-f/F_{min}, & f<F_{min},\\
(f-F_{max})/F_{max}, & f>F_{max},\\
0, & F_{min}\le f\le F_{max}.
\end{cases}
$$

$\lambda_{act},\lambda_{inact}$ 按当前帧 active/inactive node 比例自适应。优点是把“该接触时接触、且不要太弱/太强”和“不该接触时禁止 ghost collision”分开；局限是 $F_{min/max}$ 不是人类 force measurement，仍是作者为 simulator/机器人设定的 engineering prior。

## 八、IGRL：观测、网络与动作接口

### 8.1 CTDE 与 ego-centric peer state

Rhythm 使用 MAPPO 的 centralized training、decentralized execution。每台 actor 的 compact observation 为

$$
o_t=\{o_{prop},o_{peer},o_{ref}\}.
$$

peer root transform 表达到 ego frame：

$$
P_{rel}=R_{ego}^T(P_{peer}-P_{ego}),\qquad
R_{rel}=R_{ego}^TR_{peer}.
$$

这比直接输入 world coordinates 更符合真机部署，也具有 SE(2/3) 相对不变性；但 joint state、reference phase 和定位数据仍需无线传输/本地缓存。

### 8.2 输入维度

| stream | 每步维度 | window | 内容 |
|---|---:|---:|---|
| Future reference | 93 | future 20 | self q/ref velocity/root Rot6D = 64；partner reference q = 29 |
| History observation | 239 | history 20 | proprio 157；peer 82 |
| Tracking state | 64 | history 内 | current reference q/velocity + root orientation error |
| Physical state | 93 | history 内 | gravity 3、base angular velocity 3、q 29、$\dot q$ 29、previous action 29 |
| Partner state | 64 | peer 内 | partner ref q 29、actual q 29、root orientation error 6 |
| Interaction topology | 18 | peer 内 | reference 与 actual relative pose，各 position 3 + Rot6D 6 |

两个 temporal streams 各用输入投影至 60 channel，再经过 `Conv1d(60→40, k=6, s=2)` 与 `Conv1d(40→20, k=4, s=2)`、ELU。history embedding 为 67D（含显式估计的 3D base linear velocity），future embedding 为 64D。融合后 MLP hidden `[512, 256, 128]`，输出 29D Gaussian mean；可学习标准差初始化为 1.0。

### 8.3 动作与低层控制

policy action 是 29D target joint positions，不是 torque：

$$
a_t=q^{target}_t\in\mathbb R^{29}.
$$

真实机器人 50 Hz policy inference；底层 PD 500 Hz。由此 policy 学到的是 reference-conditioned position control residual/target，而接触稳定性还依赖 G1 actuator、PD gains、torque/safety limits。论文未给出所有部署 PD gains 与 emergency-stop logic，当前材料不足以原样复现高风险接触动作。

## 九、IGRL 奖励与图先验贯通

interaction reward 直接继承 IAMR 的 edges 和 distance weights：

$$
r_{inter}=\exp\!\left(
-\frac{1}{\sigma_{inter}}
\sum_{(i,j)\in\mathcal E_{inter}}\omega_{ij}
\|d_{ij}^{sim}-\hat d_{ij}^{ref}\|^2
\right).
$$

这形成论文最强的闭环：同一个 topology 先约束 kinematic reference，后约束 dynamic policy，而不是 retargeting 与 RL 各用一套无关指标。

### 奖励表（Table IV）

| 组 | 项 | 权重 | 功能 |
|---|---|---:|---|
| Interaction | Interact Edge | 1.5 | distance-weighted relative geometry |
| Interaction | Contact | 1.0 | active contact/force + inactive contact suppression |
| Upper tracking | position/orientation/linear velocity/angular velocity | 各 1.0 | 上肢姿态/速度 fidelity |
| Lower tracking | position/orientation/linear velocity/angular velocity | 各 0.5 | 下肢 tracking，弱于上肢 |
| Anchor | root position | 0.3 | 防止全局漂移 |
| Anchor | root orientation | 0.5 | 跟踪 heading |
| Regularization | action rate | -0.3 | $\|a_t-a_{t-1}\|^2$ |
| Regularization | feet slip | -0.5 | 接触足的平面滑动速度 |
| Penalty | joint limit | -10.0 | 超物理关节限位 |
| Penalty | torque | $10^{-4}$ | 以负 $\|\tau\|^2$ 抑制大力矩 |

权重显示作者优先上肢与二者交互，允许下肢对 reference 有更大偏离以维持平衡。这不是缺点，而是接触式 humanoid tracking 的合理任务分解；但固定权重能否横跨舞蹈、拥抱、推拉等任务，没有独立 sensitivity study。

## 十、鲁棒训练、课程学习与 domain randomization

每个 motion bin $s$ 保存平滑误差向量

$$
\mathbf e(s)=[e_{fail},e_{track},e_{inter}]^T,
$$

并以 non-causal Gaussian kernel（$k=3$）平滑。采样概率为

$$
P(s)=\eta\frac1S+(1-\eta)
\sum_k\alpha_k(\bar L_{max})
\frac{e_k(s)}{\sum_j e_k(j)},\qquad \eta=0.05.
$$

| 阶段 | 条件 | $\boldsymbol\alpha=[fail,track,inter]$ | 重点 |
|---|---|---|---|
| Stability | $\bar L_{max}<350$ | `[0.8, 0.1, 0.1]` | 先学不摔倒 |
| Transition | $350\le\bar L_{max}<500$ | 从 init 线性到 target | 逐渐转向精确交互 |
| Interaction | $\bar L_{max}\ge500$ | `[0.05, 0.30, 0.65]` | 重点采样 interaction error |

### Domain randomization（Table V）

| 组 | 参数 | 范围 |
|---|---|---|
| Dynamics | 每 link mass | default × $\mathcal U[0.9,1.1]$ |
| Dynamics | torso CoM offset | xyz 各 $\mathcal U[-0.05,0.05]$ m |
| Dynamics | static friction | $\mathcal U[0.3,2.0]$ |
| Dynamics | dynamic friction | $\mathcal U[0.3,1.6]$ |
| Dynamics | stiffness/damping | default × $\mathcal U[0.9,1.1]$ |
| Dynamics | ground restitution | $\mathcal U[0.0,0.8]$ |
| Calibration | default joint position | $\Delta\theta_0\sim\mathcal U[-0.01,0.01]$ rad |
| Control | control delay | $\mathcal U[0,15]$ ms |
| Push | linear xy / z | $[-0.4,0.4]$ / $[-0.16,0.16]$ m/s |
| Push | angular xy / z | $[-0.4,0.4]$ / $[-0.64,0.64]$ rad/s |
| Push | interval | every 1–3 s |
| Initialization | root position / yaw | ±5 cm / ±0.2 rad |
| Communication | peer proprio + relocalization latency | $\mathcal U[20,60]$ ms |

优点是显式随机化双机系统独有的 peer latency 和 initial relative misalignment，而不只随机 mass/friction。风险是 train randomization ranges 与真实分布的测量依据未给；packet loss、out-of-order messages、长时间 outage、地图重定位失败和 peer pose outlier 没有被单独评估。

## 十一、真机部署链路与系统适配

```text
LiDAR + IMU ──> Point-LIO local odometry (10 Hz)
      │
      └──> map registration: GeoTransformer + GICP
                    │
                    └──> Kalman filter ──> global root pose
                                              │
                     LCM broadcast <──────────┴──────────> peer
                                              │
                          ego-frame relative pose + peer joints/phase
                                              │
history/ref CNN ──> ONNX actor (<3 ms, 50 Hz) ──> 29D q target
                                              │
                                       low-level PD (500 Hz)
```

硬件为两台约 1.3 m、29-DoF Unitree G1。所有 state estimation、policy inference 与 low-level control 在 onboard CPU；Point-LIO 提供 10 Hz state estimate，实时 cloud 通过 GeoTransformer/GICP 对预建地图配准，Kalman filter 融合。机器人经 LCM 广播 global $\{P,R\}$，接收方转成 ego-relative peer pose。

两台机器人还交换 continuous motion phase：

$$
\dot\phi_{ego}=1+k(\phi_{peer}-\phi_{ego}).
$$

它通过调节播放速率软同步，避免 hard reset 跳变。若双方完全对称地同时应用这一规则，误差会收敛到中间相位；但收敛速度、$k$、通信异常处理和 reference 终点边界没有公开。更重要的是系统依赖 **预建地图**，所以不是无需外部环境准备的即插即用双机交互。

## 十二、zero-shot 与 transfer claim 审计

| claim 层级 | 是否成立 | 依据/限制 |
|---|---|---|
| raw human motion → robot references | **成立，有 model-based target adaptation** | 无 robot demonstration，但 IAMR 使用完整 G1 model 并逐帧优化。 |
| human interaction geometry transfer | **成立** | Inter-X 下 IEE/F1/DSR 显著优于 baselines。 |
| policy sim-to-real | **成立，有系统适配** | weights 部署无真机 gradient update 的描述；但需 DR、PD、定位、地图、同步。 |
| unseen robot embodiment zero-shot | **不成立** | 没有第二种 robot morphology。 |
| unseen partner type zero-shot | **不成立/未测** | 双方均为同构主动 G1。 |
| unseen motion zero-shot | **证据不足** | 未说明严格 motion train/test split 与 test reference 是否 unseen。 |
| system-level zero-shot | **不成立** | G1-specific retargeting、contact graph、sim/PD、地图、通信不可省。 |
| robust | **部分成立** | DR、真机扰动恢复视频/定量成功；未报告定位/通信 stress test 与置信区间。 |

隐藏适配包括 robot keypoint mapping、碰撞对、足接触检测、$F_{min/max}$、reward weights、domain-randomization ranges、初始双机摆位、地图扫描/配准、PD gains 和 phase gain。把这些算入成本后，论文应描述为 **zero robot demonstration transfer with substantial model/system engineering**，而不是零配置 transfer。

## 十三、实验协议与指标可解释性

### 13.1 Retargeting 指标

| 指标 | 含义 | 阈值/注意 |
|---|---|---|
| IPR ↓ | 有 inter-agent penetration 的 frame 百分比 | 依赖 collision geometry |
| MPD ↓ | 最大 penetration depth | 极值对单帧 outlier 敏感 |
| IEE ↓ | retargeted edges 相对 scaled ground truth 的 normalized L2 | 测几何，不测力 |
| F1-S / F1-L ↑ | contact F1 | strict $\tau<0.2$ m；loose $\tau<0.4$ m |
| DSR ↑ | RL rollouts 可跟踪且 interaction structure 保持 | 成功要求 IEE deviation <20% |

### 13.2 Policy 指标

| 指标 | 含义 | 阈值/注意 |
|---|---|---|
| ISR ↑ | IEE 保持在 strict tolerance 内的 step 比例 | IEE <10% |
| IEE ↓ | simulation vs retargeted interaction edge error | 与 absolute tracking error 不等价 |
| CSR ↑ | 正确 recall 足够 reference contacts 的 step 比例 | required contacts recall >80% |
| CER ↓ | required contact constraint violation rate | 不能直接解释为力安全率 |

Retargeting 的 DSR 已依赖 downstream RL，因此并非纯 kinematic 指标；它恰好反映 reference 是否“可学”，但也与 trainer 能力纠缠。真机成功按**特定 contact keyframes**计数，而不是整条 trial 一次成功/失败：Hug/Shoulder 每 trial 3 keyframes，Greeting 9 keyframes。表中的分母 30/90 因而不是独立 trials，不能把 86.7% 当成 26/30 次独立拥抱成功。

## 十四、主结果：retargeting（Table I）

| 数据/类别 | 方法 | IPR % ↓ | MPD cm ↓ | IEE % ↓ | F1-S ↑ | F1-L ↑ | DSR % ↑ |
|---|---|---:|---:|---:|---:|---:|---:|
| MAGIC Collaborate | GMR | 0.14 | 1.2 | 4.3 | 0.602 | 0.804 | 85.5 |
|  | OR | 0.20 | 1.4 | 4.1 | 0.747 | 0.902 | 87.4 |
|  | DOR | 0.00 | 0.0 | 3.9 | 0.711 | 0.899 | **89.5** |
|  | IAMR | **0.00** | **0.0** | **3.7** | **0.785** | **0.936** | 89.0 |
| MAGIC Light Contact | GMR | 2.18 | 3.3 | 4.6 | 0.738 | 0.893 | 48.5 |
|  | OR | 7.62 | 5.9 | 3.6 | 0.844 | 0.912 | 63.1 |
|  | DOR | **0.00** | **0.0** | 3.6 | 0.810 | 0.918 | 69.4 |
|  | IAMR | **0.00** | **0.0** | **3.1** | **0.905** | **0.935** | **75.3** |
| MAGIC Intensive Contact | GMR | 35.2 | 3.8 | 9.6 | 0.864 | 0.928 | 45.5 |
|  | OR | 47.3 | 5.3 | 8.0 | 0.884 | 0.929 | 56.5 |
|  | DOR | **0.00** | **0.0** | 7.8 | 0.883 | 0.925 | 63.3 |
|  | IAMR | **0.00** | **0.0** | **6.6** | **0.932** | **0.941** | **78.3** |
| Inter-X | GMR | 11.7 | 1.7 | 8.0 | 0.598 | 0.752 | 31.7 |
|  | OR | 18.4 | 2.6 | 6.8 | 0.587 | 0.791 | 46.3 |
|  | DOR | **0.00** | **0.0** | 6.7 | 0.589 | 0.795 | 52.9 |
|  | IAMR | **0.00** | **0.0** | **4.9** | **0.843** | **0.860** | **69.9** |

主趋势有三点：

1. isolated GMR/OR 在强接触会穿透，OR 的 IPR 达 47.3%；只追单体 fidelity 不足。
2. holistic DOR 可把 IPR 降为 0，却在人体比例差异大的 Inter-X 牺牲 F1-S 和 DSR；“把两人绑成一个网格”过硬。
3. IAMR 在 Light/Intensive/Inter-X 同时保持零穿透与最好 IEE/F1/DSR，支持 dual manifold 机制；但在无接触 Collaborate 的 DSR 89.0 略低于 DOR 89.5，不能说所有 cell 全胜。

## 十五、主结果与消融：policy（Table II）

| 类别 | 方法 | ISR % ↑ | IEE % ↓ | CSR % ↑ | CER ↓ |
|---|---|---:|---:|---:|---:|
| Collaborate | Single Agent | 18.7 | 38.9 | 100.0 | 0.000 |
|  | w/o Peer Obs | 19.5 | 47.0 | 100.0 | 0.000 |
|  | w/o Contact Rew | **93.4** | **4.7** | 100.0 | 0.000 |
|  | w/o Interact Rew | 58.1 | 15.1 | 100.0 | 0.000 |
|  | Full | 92.9 | 4.8 | 100.0 | 0.000 |
| Light Contact | Single Agent | 34.3 | 19.9 | 24.1 | 0.283 |
|  | w/o Peer Obs | 48.9 | 13.9 | 18.6 | 0.268 |
|  | w/o Contact Rew | 85.9 | 5.4 | 52.1 | 0.203 |
|  | w/o Interact Rew | 48.7 | 19.3 | 28.1 | 0.243 |
|  | Full | **90.0** | **4.2** | **78.0** | **0.120** |
| Intensive Contact | Single Agent | 24.0 | 29.9 | 37.5 | 0.312 |
|  | w/o Peer Obs | 34.1 | 21.4 | 43.7 | 0.280 |
|  | w/o Contact Rew | **77.3** | **7.7** | 70.6 | 0.174 |
|  | w/o Interact Rew | 51.3 | 17.0 | 56.8 | 0.211 |
|  | Full | 75.2 | 7.9 | **78.8** | **0.159** |
| Inter-X | Single Agent | 25.7 | 26.9 | 57.4 | 0.256 |
|  | w/o Peer Obs | 73.2 | 7.6 | 75.3 | 0.143 |
|  | w/o Contact Rew | **95.1** | **3.4** | 68.3 | 0.208 |
|  | w/o Interact Rew | 63.2 | 9.7 | **78.6** | **0.110** |
|  | Full | 92.8 | 3.5 | 77.4 | 0.125 |

### 因果解释

- 去掉 interaction reward，所有类别 ISR/IEE 大幅退化：它是 **coarse geometric coordination** 的关键。
- 去掉 contact reward 时 IEE/ISR 有时反而略好，却 CSR/CER 变差；这证明几何接近和真实物理接触是冲突目标，也直接支持保留 contact graph。
- 去掉 peer observation 的损失明显，但 Inter-X 仍有 73.2 ISR，说明 future reference/self state 携带相当多 partner 先验；peer stream 并非唯一交互信息源。
- Full 不是每一列最优：Inter-X 的 w/o Interaction Rew 有更高 CSR/更低 CER，w/o Contact Rew 有更优 ISR/IEE。Full 的论点应是 **Pareto balance**，不是逐指标 dominant。
- 缺少的关键消融：无 adaptive sampling、无 communication latency DR、无 initial offset DR、无 soft phase sync、不同定位 pipeline、$w_{inter}$/$F$ range sensitivity，以及 IAMR reference + standard tracker 的强 baseline。

## 十六、真机结果与 robustness 审计（Table III）

论文每项做 10 trials，但以 contact keyframes 为统计单位：

| 任务 | 方法 | 成功 / keyframes | Rate |
|---|---|---:|---:|
| Hug | Single Agent | 8 / 30 | 26.7% |
|  | Rhythm | **26 / 30** | **86.7%** |
| Shoulder | Single Agent | 6 / 30 | 20.0% |
|  | Rhythm | **24 / 30** | **80.0%** |
| Greeting | Single Agent | 11 / 90 | 12.2% |
|  | Rhythm | **74 / 90** | **82.2%** |

Hug 的三 keyframes 是 shoulder pat、hand clasp、full hug；Shoulder 有 start/mid/end walking 三个；Greeting 有九个 hand/leg/shoulder/elbow contacts。Full 相对 Single Agent 的差距超过 60 个百分点，说明闭环相对态和交互训练确实有价值。

但应避免三个过度解读：

1. 同一 trial 内 keyframes 高度相关，分母 30/90 不能当作独立 Bernoulli 样本。
2. 没有给 trial-level completion、跌倒率、接触力峰值、安全介入、均值方差/置信区间或随机 seed。
3. 推、拉、踢扰动只做 qualitative recovery 展示，未给扰动力、次数、成功率；因此支持“存在恢复行为”，不足以量化 robustness envelope。

## 十七、类型专项审计

### 17.1 Human → robot / morphology

保留的是 keypoint local topology 与 pair-relative vectors；丢失的是手指、肌肉/软组织顺应性、人体 torque/force。两个尺度流形解决了**尺度冲突**，不解决任意拓扑差异：两台 target 又是同构 G1。要验证跨本体，应至少用 G1↔H1/不同手臂长度 pair，报告无需重训、只重算 IAMR 或需 fine-tune 的分层结果。

### 17.2 多智能体交互

双方都是主动 actor，peer observation 和相对 topology 被显式建模，确实超越“把伙伴当动态障碍”。但只验证 $N=2$：interaction edges、observation 维度、MAPPO critic、通信带宽与 collision pairs 对 $N>2$ 的复杂度没有研究；不能把 dual-humanoid 自动外推为 general multi-humanoid system。

### 17.3 跨模态

这里的跨模态主要是 `BVH/SMPL keypoints → robot q/contact labels` 和 `sim global state → onboard LiDAR/IMU relative state`。不是 RGB-language-action、多传感器 policy fusion。训练中的 contact/force 是 privileged simulator signal，只用于 reward，不是部署 observation；因此不存在测试期 tactile 缺失问题，但 simulator contact model bias 会进入 policy。

### 17.4 Sim-to-real

有明确 DR、history encoder、latency randomization、onboard inference 和真机量化结果，证据强于只放视频的工作；但 real stack 使用预建地图和 target-specific tuning。最应该补的是：定位误差/延迟的测量分布、通信 packet loss stress test、无地图/仅相对 sensing baseline、DR 单项消融、真实接触力安全统计。

## 十八、局限、失败模式与外部有效性

### 论文明确或由设置直接可见

- 预建地图限制未知/变化环境；地图中出现动态遮挡时 registration 可能失败。
- 只验证两台相同 G1，不验证异构 pair 或三台以上规模。
- 需要预先给定 reference 和 phase，不是在线理解人类意图或自主协商技能。
- MAGIC 约 3 小时，类别比例而非完整 task list；代码/数据未公开。
- 真机定量任务仅 Hug、Shoulder、Greeting，10 trials/task。

### 可能的失败链

| failure | 原因 | 论文缓解 | 尚缺证据 |
|---|---|---|---|
| air contact | individual scaling 破坏 pair geometry | unified inter-edge reference | 极端身高/拓扑差 |
| penetration | holistic/isolated retargeting 不安全 | collision hard constraint + contact reward | collision model mismatch |
| stiff/unnatural pose | 所有边用统一刚性 topology | self/inter decoupling + distance stiffness | perceptual human study |
| contact ghosting | 只优化 relative keypoints | contact state + force range reward | 真机 force measurement |
| formation drift | single actor 无 peer correction | peer relative observation | peer observation outage |
| time drift | distributed clocks | soft phase feedback | packet loss、gain stability |
| pose drift | LiDAR odometry accumulation | map registration + KF | map changes/featureless scene |
| falls under collision | coupled impulses OOD | DR + contact training | quantified push envelope |

论文在 research prototype 层面说服力较高，但在部署前还需 safety supervisor、force/torque limits、emergency stop、通信 watchdog、localization confidence gating 和失联时的 safe pose；这些公开材料没有充分说明。

## 十九、代码/数据状态与三级复现路线

### 开源状态

| 资产 | 状态（2026-08-12） | 影响 |
|---|---|---|
| paper / appendix | 可访问 | 可恢复算法、表格和主要超参 |
| project videos | 可访问 | 可做 qualitative check |
| source code | **Coming Soon** | 无法核查 graph construction、trainer、deployment |
| MAGIC raw/retargeted | **Coming Soon** | 无法检查 schema、split、license |
| checkpoints | 未见 | 不能直接复跑 sim/real |
| G1 robot/model | 商用硬件；具体 sim asset 未发布 | 复现成本高 |

### L1：最小机制复现

目标：验证 dual manifold 是否优于 single/unified scale。

1. 选公开 Inter-X 的 5–10 条 handshake/hug，抽两人 global keypoints。
2. 用公开 G1 URDF 建相同 robot keypoint mapping。
3. 实现 independent、unified、IAMR 三种 objective；CVXPY + OSQP，逐帧 warm start。
4. 报 IPR/MPD/IEE/F1，画相同帧的 air-handshake、penetration 与 joint-limit diagnostics。
5. 必须固定 collision geometry、edge set、阈值，否则无法和论文表格公平比较。

**通过标准**：IAMR 零 penetration，同时在 Inter-X 严格 F1 和 IEE 显著优于 DOR；若只在近似数字上成功，应标为 mechanism reproduction 而非 paper reproduction。

### L2：仿真策略复现

目标：复现 topology reward 的因果作用。

1. 获取/重建 G1 29-DoF sim asset、PD 和 reference player。
2. 用两 actor 的 MAPPO CTDE，严格构建 20-step 93D future 与 20-step 239D history。
3. 实现 interaction/contact rewards、curriculum sampling、Table V DR。
4. 同配置训练 Full、Single、w/o peer、w/o contact、w/o interaction，至少 3 seeds。
5. 报 ISR/IEE/CSR/CER、episode fall rate、contact force distribution 与 learning curves。

**资源估计**：双 humanoid contact simulation + MAPPO 远重于普通 locomotion；论文没给 GPU/CPU、并行环境数、steps 或 wall-clock，应先做单动作小规模 smoke test，不能凭文中估计完整预算。

### L3：双 G1 真机复现

目标：验证 sim-to-real，而不是首次运行就追求拥抱。

1. 先做两台静态相对定位，测 translation/rotation error、20–60 ms latency 覆盖率与 packet loss。
2. 部署 ONNX actor，保持 policy 50 Hz、PD 500 Hz；加 watchdog、限幅、E-stop、soft padding。
3. 从 non-contact synchronized motion 开始，再做 palm touch、handshake，最后才做 hug/shoulder。
4. 每项报告 trial-level success、keyframe success、falls/interventions、peak contact force 和定位失败。
5. 对照无 phase sync、无 map correction、人工增加 5 cm/0.2 rad offset 与 packet delay。

**安全门槛**：两台 29-DoF humanoid 的相互推挤是高能量实验；没有作者代码、完整 PD/safety 配置和经过验证的 localization 时，不应直接复刻视频里的 aggressive disturbance。

## 二十、范式比较、图表索引与最终结论

### 20.1 与相邻范式比较

| 范式 | canonical interface | 接触处理 | target adaptation | 优势 | 主要限制 |
|---|---|---|---|---|---|
| 独立 Cartesian retargeting（GMR） | 各自 keypoints | 无 pair hard coupling | per-robot optimization | 简单、self motion 好 | air contact/penetration |
| 单一 interaction mesh（OR/DOR） | 一个统一 pair mesh | topology/collision | coupled optimization | pair geometry 更强 | morphology conflict 时僵硬或失真 |
| Rhythm IAMR + IGRL | self 独立流形 + pair 统一流形 | collision contact graph + force-range reward | G1-specific offline optimization + RL | 表示/奖励一致，兼顾 safety/fidelity | 非未知本体；工程栈重 |
| 端到端 multi-agent RL | raw state/reference → action | 从 reward 自学 | sim training | 可学复杂耦合动力学 | 缺可行 reference，reward/data 效率差 |
| physics-based animation | human character interaction | 视觉物理 plausibility | virtual character | 动作丰富、视觉自然 | actuator/限位/real sensing 要求弱 |

Rhythm 最可迁移的思想不是某个 PPO trick，而是：**同一份 interaction topology 必须贯穿数据 retargeting、reward、online observation 和 evaluation。** 若这些层的“交互”定义不一致，误差会在系统边界被隐藏。

### 20.2 图像与表格索引

下列图片使用 arXiv HTML 的公开在线资源；GitHub 可直接显示标准 HTTPS PNG。若 arXiv 更新版本导致资源编号变化，请点击相邻的 caption 链接核查。

#### Figure 1 — 真机交互 teaser

![Rhythm teaser with coordinated and contact-rich dual-humanoid interactions](https://hoshi-no-ai.github.io/Rhythm/_astro/teaser.BtOvdFAd_Z1ieYyK.png)

见[项目主页首屏](https://hoshi-no-ai.github.io/Rhythm/)。它展示动作覆盖面，是 qualitative evidence；定量结论仍以 Tables I–III 为准。

#### Figure 2 — Rhythm 总览

![Rhythm overview: IAMR, IGRL and real-world deployment](https://arxiv.org/html/2603.02856v2/figure/pipeline_new.png)

见[原文 Figure 2 与说明](https://arxiv.org/html/2603.02856v2#S3)。左：dual-manifold IAMR；中：MAPPO + graph rewards；右：定位与同步部署。

#### Figure 3 — MAGIC 数据分布

![MAGIC dataset categories and representative interactions](https://arxiv.org/html/2603.02856v2/figure/rss_dataset.png)

见[原文 MAGIC 小节](https://arxiv.org/html/2603.02856v2#S4.SS1)。关键数字为约 3 小时与五类占比。

#### Figure 4 — Inter-X retargeting 对比

![IAMR retargeting compared with GMR, OR and DOR](https://arxiv.org/html/2603.02856v2/figure/retarget_demo.png)

见[原文 retargeting results](https://arxiv.org/html/2603.02856v2#S4.SS3)。它直观展示 air handshake、penetration、stiff pose 与 IAMR 折中。

#### Figure 5 — Policy 消融可视化

![Policy qualitative comparison showing drift, ghosting and valid contact](https://arxiv.org/html/2603.02856v2/figure/policy_demo.png)

见[原文 policy efficacy](https://arxiv.org/html/2603.02856v2#S4.SS4)。Single Agent 漂移/碰撞，no-contact 版本几何对齐但 ghosting。

#### Figure 6 — 外部扰动恢复

![Dual G1 robots recovering from pulling pushing and kicking](https://arxiv.org/html/2603.02856v2/figure/real_robustness.png)

见[原文 real-world robustness](https://arxiv.org/html/2603.02856v2#S4.SS5)。这是 qualitative robustness evidence，不含扰动力量化。

#### Figure 7 — Interaction / Contact 图

![Topological interaction priors with interaction and contact graph edges](https://arxiv.org/html/2603.02856v2/figure/graph_vis.png)

见[原文 graph visualization](https://arxiv.org/html/2603.02856v2#A1.SS3)。黄色为 interaction edges，红色为 active physical contact。

> arXiv HTML 的正文 Figure 1 caption 在当前转换中缺号，因此本笔记使用项目页公开的同一 teaser 资源。Tables I–V 已在第十四、十五、十六、九、十节完整重排。

### 20.3 最终 verdict

**最大贡献**：**[论文明确]** 将双人 interaction representation 分解成 self/inter 两个尺度流形，并让相同 graph priors 从 IAMR 贯穿到 IGRL，最终在双 G1 真机形成闭环。相比把两人独立 retarget 或强行塞进一个 mesh，这个分解有清楚的几何动机，也获得零 penetration、Inter-X F1/DSR 与 policy 消融支持。

**最可信证据**：**[论文明确]** IAMR 在 Intensive Contact 将 GMR/OR 的 35.2%/47.3% IPR 降为 0，同时 DSR 达 78.3%；Full policy 在 Light Contact 将 Single Agent 的 ISR/CSR 34.3/24.1 提至 90.0/78.0；真机三个 task 的 keyframe success 均约 80% 以上。

**最薄弱 claim**：**[合理推断]** “multi-humanoid” 与 “robust transfer” 容易被泛化到未知机器人、三台以上、未知地图和网络异常；公开证据只覆盖两台同构 G1、预建地图与有限任务。它证明的是一个很强的 dual-G1 interaction system，不是通用 multi-robot foundation policy。

**是否值得精读**：**是。** 对 cross-embodiment 研究者，最值得借鉴的是“不要强迫所有关系共用一个 canonicalization”；对 multi-agent RL 研究者，最值得借鉴的是 interaction/contact 两个 graph reward 的分工。

**是否值得立即复现**：**分层进行。** IAMR 的 L1 机制值得先实现；完整 L2 要等 code/data 或接受大量猜测；L3 真机在作者 safety/control details 未公开前不建议直接追求接触丰富动作。

**最终结论**：Rhythm 在“人类双人数据如何变成可学的双机器人交互”上给出了目前非常完整且概念统一的答案；它的 transfer 成功依赖明确而大量的 G1-specific retargeting、simulation 和 localization infrastructure。正确的评价不是“零适配的人到多机器人泛化”，而是 **以交互拓扑为中介、强目标建模支撑的 human-to-dual-humanoid + sim-to-real transfer**。

## 参考入口

- [RSS 2026 Paper 34](https://roboticsconference.org/program/papers/34/)
- [arXiv abstract](https://arxiv.org/abs/2603.02856)
- [arXiv HTML（正文与附录）](https://arxiv.org/html/2603.02856v2)
- [Rhythm 项目主页](https://hoshi-no-ai.github.io/Rhythm/)
