---
title: "EMMA: Scaling Mobile Manipulation via Egocentric Human Data"
method_name: "EMMA"
authors: [Lawrence Y. Zhu, Pranav Kuppili, Ryan Punamiya, Patcharapong Aphiwetsa, Dhruv Patel, Simar Kareer, Sehoon Ha, Danfei Xu]
year: 2026
venue: IEEE Robotics and Automation Letters
tags: [egocentric-data, mobile-manipulation, human-robot-cotraining, action-retargeting, phase-identification, project-aria, real-robot, navigation]
image_source: online
---

# EMMA：Human Ego 能否替代移动机器人遥操作，而不牺牲可执行性？

> 本笔记基于 [arXiv:2509.04443v3](https://arxiv.org/abs/2509.04443)、[HTML 全文](https://arxiv.org/html/2509.04443v3)、[RA-L DOI 记录](https://doi.org/10.1109/LRA.2026.3653320) 与 [项目页](https://ego-moma.github.io/) 精读和交叉核验。v3 于 2025-12-27 上传，稿件于 2025-12-12 接收，正式书目信息为 **IEEE Robotics and Automation Letters 11(3): 3087–3094, 2026**。公开资产状态核验日期为 **2026-08-13**。

## 阅读结论先行

EMMA 提出一个很具体、也很有价值的问题：移动操作的数据瓶颈主要不在桌面抓取，而在操作者必须同时遥控双臂和底盘。能否只采容易获得的两类数据——**人类移动操作**与**机器人静态操作**——训练出可移动执行的统一策略？它的答案是可以，但靠的不是简单混合视频，而是三个接口共同成立：

1. 把 human head trajectory 投到地面，再通过受差分驱动动力学约束的优化器转成 robot-feasible base commands；
2. 把 human hand 与 robot end-effector 都投到当前 Aria camera frame，以 source-specific z-score 规范化后共同训练 shared transformer；
3. 从 hand/head velocity ratio 自动生成 navigation/manipulation phase label，部署时用 predicted phase 对 base/arm action chunk 做硬调制，避免抓取时底盘漂移和导航时手臂乱动。

关键的数据监督分工非常清楚。Human branch 同时训练 navigation head、human Cartesian hand head 和 phase head；robot static branch 训练 robot 14D joint action head，但**不提供移动导航监督**。部署时 robot navigation head 主要来自 human retargeted trajectories，robot manipulation execution 则由 static robot teleoperation 锚定。它不是把 human hand 直接变成 robot joints，也不是完全不用 robot data。

实验设计比多数 Human Ego → Robot 工作扎实：四个真实任务、五个 task variants、每个 task/model condition 50 trials、95% Clopper–Pearson intervals、two-proportion Z-test、总计 1,150 rollouts；碰撞直接计相应 subtask failure。Handover Wine 中，在相同 50 条 static robot demos 基础上，用 1 小时 human mobile data 取代 1 小时 mobile robot teleoperation，full success 是 82% vs 52%。在人类数据单独覆盖的新场景，EMMA full success 54%，robot-only baseline 在初始抓取后仅剩 2% navigation/full success。

更重要的是论文给了四点 scaling curve，而不只是单点：固定 1 小时 static robot manipulation data，追加 human mobile data 15/30/45/60 min 后，Handover full success 为 36/42/56/82%；追加等时间 mobile robot teleop 则为 26/32/44/52%。这支持在这个 task/platform 上，human minutes 的边际价值更高。

但“sidestepping mobile teleoperation”需要严格限定。主顺序式任务确实不需要 mobile robot teleop，却仍依赖 static robot manipulation data；而强调 tightly coupled arm–base control 的 Push Chair，base experiment 使用 **10 min mobile robot data + 20 min human**，并非零 mobile robot data。它恰好暴露边界：human navigation可替代顺序式 navigate-then-manipulate 的 base supervision，但对连续受力、臂—底盘强耦合任务仍需 target robot mobile examples。

EMMA 没有语言、object state、地图、目标坐标、接触、force/tactile、collision avoidance planner 或 online recovery。它的“task semantics”主要由视觉场景、训练 task 和二元 phase 隐式决定。Phase switching 很有效——删掉后 Handover full success 为 0——却是一种 learned label + hand-written action gate 的混合系统，不是自然涌现的 whole-body coordination。

最后，复现透明度不足。截至核验日，项目页仍标记 `Code (Coming Soon)`；没有公开训练代码、data、checkpoint、config 或 preprocessing。八页 RA-L 论文也未报告 backbone 精确层数、chunk (K/M)、optimizer、loss、batch、steps、camera rate、训练计算或大部分任务的数据量。因此结果统计可信度高，artifact reproducibility 低。

### 一句话总结

EMMA 用受约束的 human-head-to-differential-drive retargeting，把移动监督从昂贵 robot teleop 转移到 Aria 人类示范，再以 static robot data 学精细 manipulation，并用二元 phase gate 安全拼接两类动作；它在四项真机任务和 Handover scaling curve 上证明 human mobile data 很有价值，但成功依赖人形化定制平台、同任务主动采集、static robot BC 和 task-specific phase engineering，尚不是开放任务的通用 whole-body policy。

![EMMA：Human mobile data + static robot data 训练 mobile manipulation](https://ego-moma.github.io/static/figures/method_fig.jpg)

*图 1。核心数据替换关系：human 提供移动/全身行为，static robot 提供目标平台精细操作。*

## 0. 版本、资源、许可与公开程度

| 资产 | 入口 | 状态 | 许可/缺口 | 复现判断 |
| --- | --- | --- | --- | --- |
| 论文 | [arXiv v3](https://arxiv.org/abs/2509.04443)、[HTML](https://arxiv.org/html/2509.04443v3) | RA-L accepted revision，8页 | arXiv perpetual non-exclusive license | 高 |
| 正式出版 | [DOI](https://doi.org/10.1109/LRA.2026.3653320) | RA-L 11(3), 3087–3094, 2026 | IEEE publication terms | 高 |
| 项目页/视频 | [ego-moma.github.io](https://ego-moma.github.io/) | 方法图、主结果、scaling与视频 | 页面无独立媒体许可 | 中高 |
| 代码 | 项目页 `Code (Coming Soon)` | 链接仍指向空/占位地址 | **未发布** | 低 |
| Human/robot data | 无公开入口 | 只有时间/部分demo数描述 | **未发布** | 低 |
| Checkpoint/config | 无入口 | 未发布 | **未发布** | 低 |
| Hardware spec | 论文 | arm、base、camera较明确 | CAD/BOM/calibration/controller缺失 | 中低 |

阅读列表写的是 2025、三个 tasks，对应更早版本。当前 v3/正式论文是 2026 RA-L，明确列四个 task：Table Service、Handover Wine、Grocery Shopping、Push Chair；Handover Novel Scene 是第五个 evaluation variant，而非第五种 task semantic。项目页 abstract 仍残留“三个任务”，应以 accepted v3 为准。

## 1. 研究命题与三条假设

论文显式提出：

- H1：Human mobile + static robot policy 能匹配由 mobile robot teleop 训练的系统。
- H2：Kinematic retargeting、phase identification 和 unified policy 对下游可靠性必要。
- H3：已有 static robot data 后，继续采 human mobile data 比采等时间 mobile robot teleop 更有价值。

最准确的因果图是：

~~~mermaid
flowchart LR
    A["Human Aria\nego RGB + head SE(3) + hands"] --> B["head投影到SE(2)\n0.5m displacement history"]
    B --> C["constrained retargeter\n差分驱动 v, omega"]
    A --> D["hands转当前camera frame\nsource z-score"]
    A --> E["hand/head velocity ratio\nGMM phase pseudo-label"]
    F["Static robot teleop\nhead+wrist RGB + joints/EEF"] --> G["robot joint BC\n无base移动监督"]
    C --> H["navigation head"]
    D --> I["human manipulation head"]
    E --> J["phase head"]
    G --> K["robot manipulation head"]
    H --> L["shared ego stem + transformer trunk"]
    I --> L
    J --> L
    K --> L
    L --> M["robot nav chunk + 14D joint chunk + phase"]
    M --> N["phase-aware control modulation"]
    N --> O["AgileX base + dual ViperX real rollout"]
    O -. "无online learning / recovery" .-> P["subtask/full success"]
~~~

## 2. Gap—Evidence 总表

| Gap | 机制 | 直接证据 | 消融 | 真机 | 结论 |
| --- | --- | --- | --- | --- | --- |
| Human locomotion→base | constrained differential-drive optimization | Handover/Nav/scene success | raw action retargeting -30 pp full | 有 | 顺序导航有效 |
| Human hand→robot arm | current camera frame + source z-score + shared trunk | manipulation subtasks | 无 human manipulation/action-only拆分 | 有 | 由static robot anchor完成 |
| Navigation↔manipulation | unsupervised phase label + phase gate | MoF 0.929–0.965 | no phase full=0 | 有 | 强但离散/工程化 |
| Visual embodiment | human/robot共用Aria ego stem | novel scene 54% | 无不同camera/height | 有 | 平台设计预先缩小 |
| Task/semantic | task-specific visual behavior，无语言 | 已训练task success | 无unseen instruction/composition | 有 | 未解决开放Task gap |
| Contact/dynamics | static robot BC、硬件controller | final success | Push Chair少量mobile robot | 有 | 依赖target data，不显式建模 |
| Reality | 直接真机训练评测 | 1,150 rollouts | Mobile ALOHA | 有 | 单平台内可信 |

## 3. Hardware 与数据接口

上图也给出了EMMA架构与低成本双臂移动平台。平台最大高1.75 m，共用Aria head camera有意减少人—机器人视角差。

### 3.1 机器人

| 组件 | 设置 | 对 transfer 的作用 |
| --- | --- | --- |
| Base | AgileX TRACER differential drive，物理上可达2 m/s | 接收retargeted navigation action |
| Arms | 2×6-DoF ViperX 300S，倒装 | 接近成人双臂workspace |
| Grippers/action | $R_q\in\mathbb R^{2\times7}$，含gripper，共14D | static robot BC的执行锚点 |
| Head view | Project Aria，按成人hand-eye高度安装 | 与human共享device/vision stem |
| Wrist views | 2×RealSense D405 | 精细操作；human无对应模态 |
| Rig | height adjustable，最高1.75m | 减少view/workspace gap |

这是移动 manipulator，不是腿式 humanoid：没有平衡、步态、足接触、floating-base dynamics 或跌倒。论文的 whole-body 指 arms+AGV base 协调。

### 3.2 Human 与 robot data streams

| 模态 | Human | Robot | 统一方式 | 剩余差异 |
| --- | --- | --- | --- | --- |
| Ego RGB | Aria wide-FOV | robot-mounted Aria | shared vision stem | wearer body/height/motion blur |
| Wrist RGB | 无 | 双D405 | robot-specific stem | deployment可依赖human没有的近景 |
| Head/base state | MPS head SE(3) | base/ego pose | ground-plane SE(2) + egocentric history | 人可侧移/转头，base非完整等价 |
| Hand/EEF | bimanual hand (SE(3)^2) | EEF (SE(3)^2) | current-camera frame + per-source z-score | hand morphology、reach、orientation |
| Manip action | human Cartesian $K\times3$ | robot joints $K\times14$ | shared trunk、separate heads | 并非shared action head |
| Nav action | retargeted from human | baseline wheel $(v,\omega)$ | navigation head | EMMA无static robot nav labels |
| Phase | motion pseudo-label | robot static batch不激活phase head | shared phase head | binary gate，复杂耦合不足 |

Human data 是 Project Aria + MPS 主动采集，不是 passive web video。MPS 提供 bimanual hand pose 与 global localization；论文没有报告参与者人数、总任务数据分布、失败/安全片段、raw/processed frame count或每task完整时长。

## 4. Navigation retargeting：把 head path 变成可执行差速轨迹

![Human head trajectory 到 differential-drive robot path 的约束优化](https://ego-moma.github.io/static/figures/retargeting.jpg)

*图 3。不是直接把human delta pose当wheel command；先拟合robot-feasible velocity sequence。*

把 human head pose 投影到地面：

$$
h^t_{base}=(x^t,y^t,\theta^t)\in SE(2).
$$

给定 desired waypoints $p_k^d,\theta_k^d$，优化 $\mathbf z=[(v_1,\omega_1),\ldots,(v_K,\omega_K)]$：

$$
\min_{\mathbf z}\sum_{k=1}^{K}\left[
25\|p_k(\mathbf z)-p_k^d\|_2^2+
2\,\operatorname{wrap}(\theta_k(\mathbf z)-\theta_k^d)^2+
(v_k-v_{k-1})^2+(\omega_k-\omega_{k-1})^2
\right].
$$

受差分驱动约束：

$$
x_{k+1}=x_k+v_k\cos\theta_k\Delta t,
\quad y_{k+1}=y_k+v_k\sin\theta_k\Delta t,
\quad \theta_{k+1}=\theta_k+\omega_k\Delta t,
$$

$$
-1.6\le v_k\le1.6\ \mathrm{m/s},
\qquad -1.5\le\omega_k\le1.5\ \mathrm{rad/s}.
$$

注意论文写 navigation head 输出 $(x,y,\omega)\in\mathbb R^{K\times3}$，而 retargeter 优化的是 $(v,\omega)$，Mobile ALOHA baseline也输出 $(v,\omega)$。`x,y`究竟是 waypoint position、velocity中的误写，还是中间parameterization，八页正文不够清晰，是复现时应向作者确认的 schema mismatch。

### Speed-invariant context

历史不是按固定时间采样，而是累计 displacement 达到约0.5 m才加入 waypoint，并转换到当前 ego frame：

$$
\widetilde{\mathcal W}_t=
\left\{T_{ego}^{-1}h_{base}^{t-k_i}\right\}_{i=1}^{K_h}.
$$

这让相同路线在不同人速度下有相近 spatial context。代价是慢速微调、小范围绕障和原地旋转可能在0.5 m sampling中丢失；(K_h)、padding、yaw处理未报告。

## 5. Manipulation alignment 与异构 co-training

Human hand / robot EEF chunks 都转到 observation 时刻的当前 camera frame；human用SLAM，robot用hand-eye calibration。之后两个 source **各自** z-score：

$$
\bar a_H=(a_H-\mu_H)/\sigma_H,
\qquad
\bar a_R=(a_R-\mu_R)/\sigma_R.
$$

它消除尺度与均值，不保证归一化后的相同数值具有同一可达性或接触含义。Human head/hand action没有被完整 retarget 到 robot arms；跨本体作用通过 shared ego vision stem和shared decoder-only Transformer trunk发生。

| Batch来源 | 激活输入 | 激活head | 更新shared部分 | 部署使用 |
| --- | --- | --- | --- | --- |
| Human mobile | Aria ego、hand proprio、waypoint history | human Cartesian、navigation、phase | ego stem + trunk | nav/phase直接；representation间接 |
| Static robot | Aria ego、wrist RGB、joint/EEF proprio | robot 14D joint action | ego stem + trunk | robot manipulation直接 |

这是一种 multi-head co-training，不是把两域 action 强制到同一 head。优势是避免人手→robot joint 的不适定映射；风险是 shared trunk 可能只学视觉/phase而不是精确动作 correspondence。论文没有 human-only、robot-only static、shared-vs-separate trunk或freeze ego stem消融，无法分解 transfer 通道。

## 6. Phase identification 与 control modulation

### 6.1 无监督 pseudo-label

每帧计算：

$$
v_{head}=\|\Delta p_{head}\|/\Delta t,
\quad v_{hand}=\|\Delta p_{hand}\|/\Delta t,
\quad r=\frac{v_{hand}}{v_{head}+\epsilon}.
$$

先用 $r>2.0$ 且 $v_{head}<0.4$ m/s 找候选 manipulation points，再对它们的 head positions 拟合 $K=2$ GMM；probability density 高于 $\tau_{pdf}$ 判为 manipulation，否则 navigation。最短 duration threshold 为30 frames，但论文没有给 $\tau_{pdf}$。

| Task variant | Phase Mean-over-Frames |
| --- | ---: |
| Table Service | 0.962 |
| Handover Wine | 0.965 |
| Handover Novel Scene | 0.929 |
| Grocery Shopping | 0.963 |

手工标注的是均匀采样 continuous segments 的10%，不是全数据；MoF 受长 navigation phase class imbalance影响，论文未给 per-class precision/recall、boundary error或inter-annotator agreement。

### 6.2 部署 gate

在 predicted manipulation phase，把 navigation actions 从0插值到第一个 future waypoint，压制底盘噪声；在 navigation phase，把任何 future manipulation-phase positions替换成当前 navigation endpoint，以防action chunk跨phase产生不一致。这里描述仍有歧义，但总体是对两个输出流做规则化mask/interpolation。

Phase head不是独立高层planner：它只做二分类，不表示当前 subtask、目标、失败或接触状态。其价值却很直接：Handover 中去掉 phase switch，take/nav中间成功尚有0.56/0.24，最终handover为0；完整EMMA是0.86/0.82/0.82。

## 7. 真实任务与完整主结果

![五个评测variants：四种task，其中Handover含novel-scene版本](https://ego-moma.github.io/static/figures/subtasks_comparisons.png)

*图 4。Table Service与Grocery含多个双臂、导航和放置阶段，最长trial限制2分钟。*

### 7.1 三个顺序式task的累计success

图5报告的是 cumulative success：后一个subtask数值意味着此前阶段也完成。

| Task / cumulative stage | EMMA | Mobile ALOHA |
| --- | ---: | ---: |
| Table Service: pick utensils | 84% | 86% |
| → nav dining table | 38% | 82% |
| → serve utensils | 30% | 68% |
| Croissant: in plate | 62% | 84% |
| → pick plate | 30% | 68% |
| → nav dining table | 12% | 60% |
| → serve croissant | **46%** | **32%** |
| Handover: take wine glass | 86% | 72% |
| → nav to human | 82% | 52% |
| → handover | **82%** | **52%** |
| Grocery: take juice+chips | 84% | 40% |
| → bimanual take popcorn | 84% | 26% |
| → bimanual pick bag | 68% | 30% |
| → nav table | 68% | 26% |
| → place bag | **46%** | **26%** |

Table Service图的stage/order排版较难解析，且部分 cumulative sequence看起来非单调（例如早期30%而最终46%）；这可能是 Serve Utensils 与 Serve Croissant 两个variant并列而非单一累计链。不能把项目页的“comparable”改写为EMMA在每个subtask都更好：Mobile ALOHA在若干Table Service中间阶段明显更高，而两种最终variant约为30 vs 30和46 vs 26/32的关系需结合原图读取。

Handover与Grocery的总体结论清楚，论文称 two-proportion Z-test (p<0.05)；未给每个comparison的exact p值，也没有multiple-comparison correction。每condition 50 trials意味着一次成败改变2 pp。

*图 5。上图同时展示三项顺序式任务的累计成功率。人类数据优势最大的是人与机器人交互和bimanual Grocery；长距离Table Service并非全阶段占优。*

### 7.2 Handover Novel Scene

人类在新布局采30分钟，recipient在更大的 $5\,\mathrm m\times2\,\mathrm m$ 区域随机；无该环境robot data。EMMA：take glass 84%、navigation 56%、full 54%。Original-scene-only Mobile ALOHA：4%、2%、2%。

这证明 human-seen scene可迁移，但不是对完全未见场景的zero-shot generalization：新scene明确进入human co-training。Mobile ALOHA在第一步grasp就崩，比较同时混合了 visual appearance、layout/state coverage 和 navigation supervision，无法把54%只归因于path transfer。

### 7.3 Push Chair：真正耦合任务揭示边界

![Push Chair：unified policy、separated policy与robot-data baselines](https://arxiv.org/html/2509.04443v3/push_chair.png)

*图 6。此task不做phase switching，要求arms与base连续协同。*

Push Chair图的四条curve文字抽取存在标签错位，能可靠确认的论文结论是：unified EMMA（10 min mobile robot +20 min human）与30 min pure mobile robot baseline表现相当，并优于separate nav/manip models，尤其初始bimanual engagement。论文也明确承认human navigation alone不足以解决tight arm–base coupling。

这一实验不能支持“zero mobile teleop”：EMMA此处确实使用10分钟mobile robot data。更恰当的claim是human data能替代一部分mobile teleop，并提高unified representation。

## 8. Scaling：四点曲线比“1小时胜1小时”更重要

![固定static robot data后，human mobile minutes与mobile robot teleop minutes的scaling](https://ego-moma.github.io/static/figures/scaling_law.png)

*图 7。只在Handover Wine上测试；不是跨task的通用scaling law。*

| 新增数据时长 | Human mobile + fixed static robot | Mobile robot teleop + fixed static robot | Gap |
| ---: | ---: | ---: | ---: |
| 15 min | 36% | 26% | +10 pp |
| 30 min | 42% | 32% | +10 pp |
| 45 min | 56% | 44% | +12 pp |
| 60 min | 82% | 52% | +30 pp |

这条曲线不是严格统计学习 scaling law：只有一个task、四个离散点、没有training seeds或fit exponent，最后一个human点从56跳到82也可能是coverage threshold。比较按 wall-clock collection time配平，而不是demonstration count、frames、successful episodes、annotated action tokens、processing labor或energy。Human还需要MPS、retargeting和清洗；robot teleop需要硬件与操作者。它支持“采集小时收益”，不等价于端到端成本已精确配平。

同样应注意固定的“1小时 static robot manipulation data”意味着human曲线不是human-only。精细take-glass/handover的joint/gripper control始终从robot domain学习。

## 9. Ablation 与因果证据

### 9.1 Handover component ablation

| Method | Take glass | Navigate | Full handover |
| --- | ---: | ---: | ---: |
| **EMMA** | **86%** | **82%** | **82%** |
| w/o phase switch | 56% | 24% | **0%** |
| w/o action retargeting | 88% | 8% | 8% |

去retargeting让full 82→8（74 pp），而正文只说“drops by 30%”，可能指图中某一中间stage或早期版本措辞；accepted v3图的full-task差距明显更大。去phase full=0，说明 gate 是安全与成功的必要组件。

但 component ablation 没有：ground-truth phase oracle、fixed time/position state machine、raw head path+classical controller、retargeter open-loop tracking RMSE、无human hand supervision、无shared ego stem、no wrist views。这些对照能判断policy究竟学到了phase，还是只需要一个可靠的传统state machine。

### 9.2 Unified vs separated

Push Chair支持unified model在耦合动作上好于separate nav/manip；顺序task反而用phase gate主动解耦。论文没有给同compute/parameter的完整数表与seeds，因此应理解为有力的task-specific evidence，而非一律“unified优于hierarchical”。

## 10. Task、Semantic、Motion 与 Contact gap

| 层级 | Human signal | Robot signal | 对齐 | 缺口 |
| --- | --- | --- | --- | --- |
| Task semantic | task-specific visual demo | task-specific static demo | shared trunk | 无language/task token |
| Subgoal | nav/manip binary phase | rollout stages只用于评测 | phase pseudo-label/head | 无object/subtask identity |
| Motion intent | head waypoints + hand xyz | joint trajectories | retargeting + camera frame | human orientation/contact弱 |
| Object transition | RGB隐式 | RGB隐式 | 无 | 无object state/dynamics |
| Contact/grasp | hand pose间接 | joint/gripper BC | shared latent间接 | 无contact point/force/tactile |
| Safety | human path prior | collisions算failure | phase gate | 无obstacle map/constraint filter |

模型不是 VLA，没有自然语言输入。Handover recipient、grocery shelf和两张table的位置由visual history与waypoint context隐式推断；新instruction、paraphrase、新目标组合或同scene多意图均未测。

Contact intent没有显式表示。Push Chair 是唯一持续接触/arm-base coupling测试，但需要少量mobile robot data。Wine handover涉及human contact，却没有force/compliance、安全距离、recipient motion、release detection或伤害指标；“safe range”在文字中出现，未给数值或控制逻辑。

## 11. Reality gap 与平台泛化

| 差异 | EMMA如何处理 | 证据 | 剩余风险 |
| --- | --- | --- | --- |
| Human omnidirectional vs differential base | constrained retargeter | no-retarget ablation | narrow passage/obstacle/side-step |
| Head gaze vs body heading | yaw cost + robot Aria固定朝向 | Handover navigation | 人可独立转头，proxy不总成立 |
| Camera | 同一Aria +成人高度 | novel scene | 不同sensor/FOV/height未知 |
| Arm/hand morphology | current-frame hands/EEF + separate heads | static robot BC | 新arm/gripper需新anchor |
| Dynamics/contact | real robot demos/control | real success | payload/friction/compliance无显式模型 |
| Arm–base coupling | unified trunk；少量mobile robot | Push Chair | human-only不足 |
| Obstacles/safety | collision作为failure | 失败统计 | 无规划器/CBF/emergency logic报告 |

EMMA没有 sim-to-real，所以目标平台 Reality gap 是通过real data避免的。代价是 external validity：AgileX+ViperX、Aria成人高度、同一rooms/tasks。更换差速base尺寸、转弯半径、camera mounting或arm reach都需要重新retarget、校准并很可能post-train。

## 12. 快慢系统接口

| 层 | 输入 | 输出 | 时间尺度 | 责任 |
| --- | --- | --- | --- | --- |
| System 2 | 无language/planner | binary nav/manip phase | action chunk级 | 只做mode selection |
| System 1 | ego/wrist RGB、joint/EEF、waypoint history | base chunk、14D joint chunk、phase | (K/M)未报告 | perception + imitation |
| Projection | human head path | feasible $(v,\omega)$ labels | offline trajectory | navigation embodiment |
| Gate | predicted phase + chunks | masked/interpolated chunks | online | 防base/arm串扰 |
| System 0 | target joint/base commands | servo与wheel motion | 未报告 | tracking、contact、安全 |

有趣的是，EMMA同时支持两种看似相反的观点：顺序任务需要显式phase gate将base/arm分开；Push Chair需要unified model联合协调。合理架构不是固定“hierarchical”或“end-to-end”，而应让 coupling degree 随task/contact phase变化。

## 13. Claim—Evidence 审计

| Claim | 所需证据 | 现有证据 | 判断 | 限定 |
| --- | --- | --- | --- | --- |
| Human可替代mobile teleop | 等时间、同static data/backbone | Handover scaling + main tasks | 强 | 单平台/任务 |
| 无mobile robot teleop | 所有tasks零mobile data | 顺序task成立；Push Chair用10min | 需限定 | 不是全实验成立 |
| Retargeting必要 | raw vs constrained | Handover 82→8 full | 强 | 无classical/open-loop RMSE |
| Phase head必要 | remove/ground-truth/rule | remove→0 full | 组件必要 | 未证明learned优于FSM |
| Human data泛化新scene | human-seen/robot-unseen | 54 vs2 full | 强 | 非human-unseen |
| Human data更可扩展 | 多任务曲线与全成本 | 单Handover四点 | 趋势强 | 不宜称普适law |
| Whole-body coordination | coupled tasks | Push Chair | 中 | 少量mobile robot data |
| 安全可靠 | collision/intervention metrics | collision计failure | 弱中 | 无rate表/force/near miss |

## 14. 数据治理、隐私与偏差

- Aria采集室内移动路径，会记录旁观者、布局、屏幕、面部和个人物品；论文未报告consent、IRB、bystander notification或redaction。
- MPS涉及云端global localization/hand tracking；需披露raw上传、retention、派生spatial map与轨迹访问控制。
- Handover包含human recipient，可能泄露身份、身高、惯用手和互动习惯；也带来近人移动/递物安全责任。
- 未报告demonstrator数量与人口统计；一小时视频不等于行为多样性，可能只是单人重复。
- Human场景覆盖与robot evaluation是否存在同collector、同object instance或连续trajectory leakage没有说明。
- 只采successful demonstrations会弱化碰撞恢复、掉落处理、找不到recipient等长程纠错。
- 数据/weights未来发布需单列Aria recording、MPS derivatives、project videos、robot logs和code licenses，不能用一个repo license概括。

## 15. 复现路线与缺失清单

### Level 1：Retargeter

用任意 $SE(2)$ walking paths重建约束优化，验证position/yaw RMSE、曲率、速度/角速度limit和open-loop execution。当前已知权重25/2/1与limit；仍缺 $K,\Delta t$、solver、initialization、acceleration constraints和失败处理。

### Level 2：Phase pseudo-label

用Aria head/hand trajectories重建ratio+GMM，报告per-class F1、boundary F1、task transfer。已知ratio 2.0、head 0.4m/s、duration30、GMM $K=2$；缺 $\tau_{pdf}$、filtering和hand aggregation定义。

### Level 3：Static+human co-training

复现HPT modality stems/shared trunk/four heads。论文缺 optimizer、LR、loss weights、batch ratio、steps、augment、chunk K/M、temporal horizon、image resolution/encoder、phase loss与compute，需作者代码或自行定义并标为replication。

### Level 4：真机

先Handover，至少50 trials/condition，复现82/52 ranking与no-phase/no-retarget failure。必须记录controller、action rate、latency、emergency stop、collision/near-miss、hand-off force和recipient protocol。

## 16. 面向 humanoid / EX002 的落地建议

| 目标 | 可复用 | 需要替换 | 数据策略 | 首个pilot |
| --- | --- | --- | --- | --- |
| 同类轮式双臂 | Aria、retargeter、phase head、HPT | arm/gripper calibration | human mobile + static robot | navigate-pick-handover |
| EX002腿式humanoid | ego waypoint/history、phase concept | SE(2)差速模型→footstep/WBC、balance | human head+body + static manipulation + locomotion robot | 无负载navigate-then-place |
| 全身loco-manip | shared trunk | floating-base/contact-aware controller | human full-body + robot dynamics | phase-separated task先行 |
| 灵巧手 | visual/nav/phase prior | hand action head、contact retarget | human hand-object + dexterous robot | stationary grasp后再移动 |

腿式humanoid不能直接使用差速运动学。应把human head path转成可行base trajectory/footstep/contact schedule，由WBC检查CoM、support polygon、self-collision和arm reach。Phase不应只有nav/manip：至少需要approach、pregrasp、contact、transport、release、recovery，并允许Push Chair类task进入coupled mode。

建议门槛：retarget tracking error和constraint violation先离线通过；50+真机trials的full SR置信下界；碰撞/near-miss/intervention；手臂与base action在错误phase的泄漏率；new-scene按human-seen与human-unseen分别报告。

## 17. 十个组会质疑

1. 为何navigation head写成 $(x,y,\omega)$，retargeter与baseline却是 $(v,\omega)$？
2. 主任务宣称zero mobile teleop，Push Chair为何又用10分钟mobile robot data？
3. Phase ablation到0说明gate必要，但ground-truth phase或简单FSM是否同样/更好？
4. Table Service中间stages Mobile ALOHA明显更高，最终“comparable”应如何精确解释？
5. Handover scaling最后15分钟为何56→82跳变，是coverage threshold还是seed噪声？
6. 等采集时间为何不计MPS processing、retargeting、清洗和robot setup总成本？
7. Human manipulation head对robot到底贡献多少？只保留human navigation会怎样？
8. 同一Aria、成人高度和类人双臂已显著缩小gap，换camera/morphology还能否transfer？
9. 二元phase如何处理移动中持续接触、边走边抓和动态recipient？
10. 论文已正式发表，为何code/data/checkpoint仍是Coming Soon？

## 18. 能力评分与最终判断

| 能力 | 1–5 | 依据 |
| --- | ---: | --- |
| Mobile navigation transfer | 4.5 | human-only nav supervision、retarget ablation、novel scene |
| 精细manipulation | 3.5 | static robot BC、多项真机；非human-only |
| Arm–base coordination | 3 | Push Chair有效但需mobile robot data |
| 视觉泛化 | 3.5 | human-seen novel scene 54%；无human-unseen |
| Task/semantic | 1.5 | 无language，task-specific视觉policy |
| Contact/safety | 2 | real contact与collision failure；无force/tactile/安全rate |
| 因果证据 | 4 | equal-time scaling、组件消融、50 trials/CI |
| 真机可信度 | 4.5 | 四tasks、五variants、1,150 rollouts |
| 开源复现 | 1 | code/data/checkpoint/config未发布，训练细节大量缺失 |

### 最终判断

EMMA 最重要的洞见不是“human video能预训练视觉”，而是把 mobile manipulation 拆成可由不同source监督、又在一个policy中复合的接口：human head motion经过运动学retarget后直接教base navigation，human hands和ego views改善shared representation，static robot demos教可执行joint manipulation，phase head在部署时阻止两个action流互相污染。

最准确的总结是：

> EMMA 在 **simultaneous heterogeneous co-training** 中，以 **retargeted SE(2) navigation、camera-frame hand/EEF motion和binary phase** 引入 Human Ego 数据；它主要替代 **顺序式移动操作的robot base teleoperation**，而不是替代target robot manipulation data。可执行性由 **static robot joint BC、差分驱动优化和phase gate** 保证；tight arm–base contact仍需要少量mobile robot data，开放任务语义、contact dynamics与跨平台Reality gap尚未解决。

这是一篇适合移动humanoid/轮式双臂项目直接借鉴的系统论文，尤其值得复现retargeter与phase interface；但在源码、数据、训练配置公开，并在不同morphology与human-unseen scene验证之前，不能把一项Handover scaling curve外推成通用human-data scaling law。
