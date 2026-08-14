---
title: "One Hand to Rule Them All: Canonical Representations for Unified Dexterous Manipulation"
method_name: "OHRA"
authors: [Zhenyu Wei, Yunchao Yao, Mingyu Ding]
year: 2026
venue: "Robotics: Science and Systems (RSS) 2026"
tags: [dexterous-manipulation, cross-embodiment, canonical-representation, canonical-urdf, morphology-conditioning, diffusion-grasping, zero-shot, sim-to-real]
zotero_collection: "_inbox/cross-embodiment"
image_source: online
arxiv_id: "2602.16712"
arxiv_html: "https://arxiv.org/html/2602.16712v2"
created: 2026-08-12
---

# One Hand to Rule Them All: Canonical Representations for Unified Dexterous Manipulation

## 元信息与证据边界

| 项目 | 内容 |
|---|---|
| 作者 | Zhenyu Wei, Yunchao Yao, Mingyu Ding |
| 机构 | University of North Carolina at Chapel Hill |
| 会议 | RSS 2026 |
| 论文 | [arXiv v2](https://arxiv.org/abs/2602.16712v2) · [HTML 全文](https://arxiv.org/html/2602.16712v2) · [项目页 PDF](https://zhenyuwei2003.github.io/OHRA/assets/OHRA_paper.pdf) |
| 项目主页 | [OHRA](https://zhenyuwei2003.github.io/OHRA/) |
| 官方代码 | [zhenyuwei2003/OHRA](https://github.com/zhenyuwei2003/OHRA) |
| 数据 | [Google Drive（README 当前链接）](https://drive.google.com/file/d/1Ery6WAayDxKgYpIbu7l8DXXVYV19rw78/view?usp=sharing) |
| 正文版本 | arXiv:2602.16712v2，2026-05-15；Accepted at RSS 2026 |
| 代码审查基准 | `master`，最新公开 commit `31e31e8c`，2026-07-06 |
| 审查日期 | 2026-08-12 |

证据标签：**[论文明确]** 表示正文或附录直接给出；**[代码明确]** 表示官方仓库 README、配置或实现可确认；**[作者材料]** 表示项目主页或作者在官方 issue 中的说明；**[合理推断]** 表示从公开链路和机器人常识得出的分析；**[证据不足]** 表示公开材料不能支持更强结论。项目页视频用于理解动作，不替代定量实验。

正文涉及的外部论文、代码与数据入口：

| 资源 | 论文/项目 | 代码/数据 |
|---|---|---|
| $\mathcal D(\mathcal R,\mathcal O)$ Grasp | [arXiv:2410.01702](https://arxiv.org/abs/2410.01702) · [项目页](https://nus-lins-lab.github.io/drograspweb/) | 项目页的 Code/Data 入口；OHRA 的 24,764 grasps由其 filtered GenDexGrasp 数据构建。 |
| GenDexGrasp / MultiDex | [arXiv:2210.00722](https://arxiv.org/abs/2210.00722) | [官方 GitHub 与 MultiDex 下载说明](https://github.com/tengyu-liu/GenDexGrasp) |
| Lightning Grasp | [arXiv:2511.07418](https://arxiv.org/abs/2511.07418) | [OHRA 内嵌的 generation dependency](https://github.com/zhenyuwei2003/OHRA/tree/master/src/third_party/lightning-grasp) |
| LEAP Hand | [arXiv:2309.06440](https://arxiv.org/abs/2309.06440) | [官方项目/硬件仓库](https://github.com/leap-hand/LEAP_Hand_API) |
| Isaac Gym | [arXiv:2108.10470](https://arxiv.org/abs/2108.10470) | [NVIDIA 下载页](https://developer.nvidia.com/isaac-gym/download) |
| Viser | [arXiv:2507.22885](https://arxiv.org/abs/2507.22885) | [官方 GitHub](https://github.com/nerfstudio-project/viser) |

![OHRA teaser](https://zhenyuwei2003.github.io/OHRA/assets/OHRA_teaser.png)

## 一、论文速览

### 一句话总结

> OHRA 把拟人灵巧手压到固定的形态参数与 22-DoF canonical URDF/action space，再以手形态条件化抓取生成器，使多手抓取数据能够联合训练，并在同一 LEAP 模块化手族的删指节变体上实现无需梯度更新的抓取迁移。

### Elevator pitch

不同灵巧手的 URDF 拓扑、坐标轴、关节顺序、手指数和 DoF 都不同，直接拼数据既无法给网络固定形态输入，也没有统一动作标签。OHRA 以“最多五指、最多 22 个 canonical joints”的拟人模板，把原始 URDF 映射为 82 维基础表示或 173 维扩展表示；不存在的关节通过零范围占位，从而把变长动作空间变成固定 22 维。三手实验用 VAE 的 16 维形态 latent 条件化一个两阶段 grasp generator，LEAP zero-shot 实验则直接使用 152 维处理后的扩展参数，在 66 个高 DoF 变体上训练并测试未见删指节变体。最强结果是未见 `leap_3303` 仿真成功率 81.9%，以及 `leap_3033`/`leap_3303` 真机 zero-shot 各 71%；但目标手 URDF、人工 joint mapping、LEAP 专用扩展、目标控制器和相机/机械臂系统都已知，因此是 **policy/data-space zero-shot + target configuration**，不是 system-level 即插即用。

### 迁移类型

| 迁移模式 | 判断 | 说明 |
|---|---|---|
| 1. 人类数据 → 机器人策略 | 不适用 | 没有人类示范；官方 issue 明确 raw MANO 不能直接输入，需先构造 MANO URDF。 |
| 2. 人类交互 → 多机器人控制 | 不适用 | 单手静态抓取与单手 in-hand reorientation。 |
| 3. 已见机器人 → 未见机器人 | **核心但分两层** | 三手联合训练只评估已见 Allegro/Barrett/Shadow；真正 held-out morphology 在 LEAP 变体族内。 |
| 4. 跨模态数据迁移 | **有限** | URDF/形态 metadata + object point cloud 联合条件化；没有触觉、力、接触传感或语言融合。 |
| 5. 仿真 → 真机 | **有** | 仿真生成 grasp pose 部署到 Franka Research 3 + LEAP variants；训练/部署桥细节和代码不完整。 |
| 6. 跨任务/场景/对象 | **跨对象** | 三手实验在 10 个未见对象评估；LEAP 协议固定 10 个对象，不是 task generalization。 |

### 核心判断

| 问题 | 结论 |
|---|---|
| Source domain | 65,536 个合成 canonical morphologies；Allegro/Barrett/Shadow 的 24,764 个有效 grasp；LEAP 变体的 Lightning Grasp + Isaac Gym 过滤数据。 |
| Target domain | held-out LEAP link-deletion variants；真机 `leap_3033`、`leap_3303`；同一静态抓取任务。 |
| 真正迁移的资产 | canonicalized grasp data、共享 grasp generator 参数、形态条件与固定 22D action semantics。 |
| 未迁移而重新适配的资产 | target URDF/meta JSON、palm frame、joint correspondence/sign、LEAP 专用扩展模板、target controller、Franka/RealSense 标定与执行栈。 |
| 核心 transfer gap | 不同手的 topology/DoF/joint axes/order 与可达抓取空间不同，导致输入和动作标签都不对齐。 |
| 最关键迁移机制 | **固定语义关节槽位 + morphology conditioning**；不是 VAE 本身，也不是新型 diffusion backbone。 |
| 最重要贡献 | 把 URDF 规范化、形态条件和 joint action 规范化接成可实际联合训练的数据接口，并发布了相应资产/训练代码。 |
| 最应质疑 claim | “zero-shot generalization to novel hand morphologies”的外延：主证据来自一个 LEAP 家族内删 link，且为了 LEAP 单独改了 canonical joint placement。 |
| 是否值得精读 | **值得**，尤其适合做 robot metadata/action schema；不应当作通用动态灵巧操作方案。 |
| 是否值得复现 | **有条件**：canonical converter 值得优先；论文级仿真需旧版 Isaac Gym、外部数据和缺失 checkpoint；真机链不完整。 |

五类概念必须分开：

- **Data transfer**：三手抓取数据转为 canonical joint labels 后联合训练；LEAP 多变体合成数据被同一模型共享。
- **Representation alignment**：82/173 维 morphology schema、统一坐标系与固定 22 joint slots。
- **Retargeting**：原始 joint vector 与 canonical vector 的确定性排列、符号翻转和 dummy joint 映射；不是学习式/优化式 motion retargeting。
- **Policy transfer**：同一个 grasp generator 在训练手之间共享，并对 held-out LEAP variants 不更新参数地推理。
- **System transfer**：仿真 grasp 到 Franka + LEAP 真机；依赖未开源的 target-specific 部署工程，不能与 policy transfer 混为一谈。

## 二、迁移问题的形式化定义

令源域集合为

$$
\mathcal{D}_{\mathrm{src}}
=\{D_h\mid h\in\mathcal{H}_{\mathrm{train}}\},
$$

LEAP zero-shot 的训练本体集合可写成

$$
\mathcal{H}_{\mathrm{train}}
=\{\texttt{leap}_{xyzw}\mid x,y,z,w\in\{0,1,2,3\},\ x+y+z+w\ge 8\},
$$

论文实际选取其中 **66 个**具有有效 grasp 的变体构造训练数据，并另外做 exact-variant leave-one-out。目标集合包括

$$
\mathcal{H}_{\mathrm{tgt}}
\supset
\{\texttt{leap}_{3033},\texttt{leap}_{3303},\texttt{leap}_{3330},
\texttt{leap}_{0303},\texttt{leap}_{0312},\texttt{leap}_{2203},
\texttt{leap}_{3030},\texttt{leap}_{3103}\}.
$$

对一只手 $h$ 和对象点云 $P_o$，canonical converter 定义

$$
c_h=C(U_h,m_h),\qquad
U_h^{\mathrm{can}}=G(c_h),
$$

其中 $U_h$ 是原始 URDF，$m_h$ 是包含 `palm_origin`、`joint_mapping` 等人工 metadata 的 JSON，$c_h$ 是 canonical morphology condition。原始/规范动作间的双向离散映射为

$$
q_h^{\mathrm{can}}=M_h(q_h^{\mathrm{orig}}),
\qquad
q_h^{\mathrm{orig}}=M_h^{-1}(q_h^{\mathrm{can}}),
$$

这里 $q_h^{\mathrm{can}}\in\mathbb{R}^{22}$，缺失关节为 inactive dummy variables。

抓取生成器学习

$$
(\hat T,\hat R,\hat\theta)
=\pi_\phi(P_o,c_h,R),
\qquad
\hat\theta\in\mathbb{R}^{22},
$$

其中 $(T,R)$ 是 object frame 中 wrist translation/rotation，$R$ 在推理时显式给定或随机采样，$\theta$ 是 canonical hand joint pose。

| 维度 | Source | Target | 对齐/held-out 状态 |
|---|---|---|---|
| 本体 | Allegro/Barrett/Shadow；66 个 LEAP variants | leave-one-out 或低 link-count LEAP variants | exact variant 可 held out；family/template/生成规则不 held out。 |
| 观测空间 | object point cloud + hand condition + wrist rotation | 同 schema | schema 完全共享；target 参数在推理时显式提供。 |
| 动作空间 | 22D canonical joints + 9D wrist pose | 同 canonical 输出，再映回 target joints | tensor 统一；物理效果并不严格统一。 |
| 动力学 | canonical URDF in Isaac Gym | target canonical/original URDF 与真机 | target simulator/URDF 已知；无 dynamics conditioning。 |
| 任务分布 | force-closure static grasp | 相同任务 | 任务没有 held out。 |
| 对象分布 | 三手数据的 train split；LEAP 的 10 objects | 三手实验 10 unseen objects；LEAP 同 10 objects | 跨对象只在三手协议中成立。 |
| 模态 | object point cloud、URDF-derived morphology | RealSense point cloud、target morphology | camera→point cloud 过程未开源。 |
| 标签/监督 | Lightning/GenDexGrasp-derived wrist pose + joint pose | 无 target labels（对应 zero-shot 模型） | Synthetic optimization-generated labels。 |
| 训练可用 target 信息 | target schema/template、URDF family 与 target exclusion list | exact target grasp 可排除 | 不是 target-blind。 |
| 测试可用 target 信息 | target URDF/meta/parameters、object point cloud、desired wrist rotation | 全部可用 | 允许本体描述是 conditional zero-shot 的定义。 |

论文学习的是 **共享 representation + 共享 grasp generator + 固定数据转换器**，不是共享低层反馈控制器。测试时未见的是 exact hand morphology；task 和 object protocol 基本不变，硬件家族也不变。对于 `leap_3033/3303/3330`，训练集合与 target 在 exact ID 上不重叠，但 target 是训练生成族的近邻；对于 $x+y+z+w<8$，结构差异更大，却仍由同一模板删 link 得到。

隐含不变量包括：

1. 所有目标都能解释为人形手掌上的 thumb/index/middle/ring/little 五条链。
2. 每指最多使用 canonical 的 proximal/middle/distal 与预定义 abduction/flexion/rotation 槽位。
3. joint pose 在同一语义槽位中可比较，且零范围能安全表示缺失关节。
4. 静态 grasp 的对象几何—手形态关系可由 point cloud 与 morphology condition 决定，而无需接触历史、力或触觉。

前两项对多数拟人刚性手有运动学依据，对三指 Barrett 仍需人为语义命名；对腱驱欠驱动手、软体手、并联机构、六指手或非树结构并没有普适物理保证。

## 三、迁移账本：Source → Alignment → Target

| 阶段 | 输入 | 输出 | 域差异 | 对齐机制 | 可学习 | target-specific | 证据 |
|---|---|---|---|---|---:|---:|---|
| 数据采集/生成 | GenDexGrasp/D(R,O) grasps；Lightning Grasp candidates | 原始 wrist + hand joint poses | 每手 action dimension/order 不同 | 每手先独立生成 | 部分生成算法 | 是 | [论文明确][代码明确] |
| 数据预处理 | 原始 grasps + hand assets | filtered canonical grasp tuples | 碰撞/可行性与 joint schema 不同 | Isaac Gym filtering + $M_h$ | 否 | 是 | [论文明确][代码明确] |
| 观测表示 | object mesh/point cloud | 512-point cloud feature | 对象采样/噪声 | DGCNN-like encoder；2 mm Gaussian noise | 是 | 否 | [代码明确] |
| 本体表示 | URDF + meta JSON | 82D base / 173D extended；网络实际 66D/152D | heterogeneous trees/frames/DoF | canonical parameters | 否；VAE 可学习压缩 | **是** | [论文明确][代码明确] |
| 动作表示 | variable-DoF joint pose | 9D wrist + 22D canonical joints | joint count/order/sign | fixed joint slots + inactive mask | grasp pose 可学习 | mapping 是 | [论文明确][代码明确] |
| 几何/本体对齐 | original frames/meshes | palm frame + capsule/cylinder URDF | global/local axes、mesh shape | frame convention + primitive geometry | 否 | meta/manual repair 是 | [论文明确][代码明确] |
| 接触对齐 | generated candidate | physically filtered grasp | contact set 随手变化 | force-closure perturbation evaluation | 否 | simulator geometry 是 | [论文明确] |
| 策略学习 | $P_o,c_h,R,T,\theta$ | diffusion translation + MLP joints | morphology-conditioned distribution | shared encoder/generator | 是 | 否 | [论文明确][代码明确] |
| 低层控制 | predicted wrist/joint pose | simulated/real execution | kinematics、actuator、arm-hand system | original↔canonical mapping + controller | 否 | **强** | [论文明确][证据不足] |
| sim-to-real | Isaac Gym canonical pose | Franka + LEAP grasp | sensing/contact/actuation gap | L515 point cloud + target system | 否 | **强** | [论文明确] |

信息损失最大的是 `original URDF → canonical template`：真实 collision meshes 被 cylinder/capsule 取代，base 版共享非拇指 link length/半径并固定大量 joint axes；Allegro 的 axial joint 甚至被省略，导致 canonical policy 映回 original 时成功率下降 12.6 points。误差会沿 `URDF parsing/manual mapping → canonical grasp conversion → point-cloud feature → wrist diffusion → joint MLP → inverse mapping → contact execution` 累积。

迁移上限由 **canonical template 是否覆盖 target 的可控子空间** 决定；diffusion 只在这个上限内拟合 grasp distribution。URDF parser、meta JSON、LEAP joint relocation、Franka/RealSense 标定和控制器是工程适配。去掉 target-specific `joint_mapping/palm_origin`，系统连 tensor 语义都不能确认；去掉 LEAP 专用扩展，论文也没有给 zero-shot/real 性能。

## 四、源数据审查

### 4.1 数据来源与规模

| 数据层 | 规模 | 用途 | 是否公开/可审查 |
|---|---:|---|---|
| Synthetic morphology | **65,536 hands** | 训练 66D→16D VAE latent | 采样代码公开；不是 65,536 个真实 URDF。 |
| 三手 canonical grasp | **24,764 valid grasps**，Allegro/Barrett/Shadow | 多手联合 grasp generator | README 提供 Drive 数据；loader 与 split 代码公开。 |
| 三手对象 | train/validate object JSON；正文称在 **10 unseen objects** 测试 | 测 object generalization | split JSON 在下载数据内，不在仓库 tree；公开性依赖 Drive。 |
| LEAP family | 理论上 $4^4=256$ variants；训练选择 66 个 $x+y+z+w\ge8$ variants | family-scale co-training | 256 份 JSON/URDF 与 generation code 公开。 |
| LEAP capped train data | **69,917 grasps**；每 hand-object 最多 200 | zero-shot model | 配置名 `leap_dataset_geq8_max200` 与 loader 可确认。 |
| Low-link specialist data | 0303: 37,249；0312: 4,368；2203: 2,458；3030: 37,217；3103: 2,124 | 与 zero-shot 对比 | 正文 Table VIII。 |
| In-hand initial states | 每手 **10,000 valid grasps** | canonical fidelity RL | 附录明确；eval 从同一初始集合抽样。 |

这不是 teleoperation、human video 或 robot rollout dataset。主要监督是 optimization/analytical grasp synthesis 后在 Isaac Gym 过滤得到的 `wrist pose + joint pose`。LEAP 每种形态需要给 Lightning Grasp 配置 fingertip links 与 active joints，分四轮生成，再做物理过滤。**[论文明确]**

数据不需要人工逐 grasp 标注，但需要每只手的 URDF、meta JSON、canonical joint correspondence 和生成配置。数据采集成本低于真实手遥操作，却把成本转移给仿真资产、grasp synthesis、collision validation 和 GPU 评估。其他机器人能否复用，取决于它是否可无损进入 canonical schema；否则必须重做 mapping、生成/过滤甚至扩展模板。

### 4.2 数据质量与覆盖

- 时间同步、相机漂移、采集者尺度：**不适用**，训练主体是离线静态 grasp pose。
- 坐标系一致性：是论文直接处理的核心；统一 palm/global/local joint frames。**[论文明确]**
- 接触标签：没有显式 contact vector 作为 policy input；可行性由 grasp synthesis 和 Isaac Gym force-closure test 间接决定。
- 力/触觉：训练标签不包含真实 force/tactile；摩擦、compliance 与接触力分布没有跨域对齐。
- 失败数据：过滤后数据只保留 valid grasps；失败 candidate 数、过滤率与各手分布没有完整汇总。
- 数据平衡：loader 先均匀随机选 hand，再在该 hand 的 metadata 中随机选 grasp，因此三手/LEAP variant 在 batch 层面近似等权，避免大数据手直接主导。**[代码明确]**
- 对象 point cloud：训练随机采 512/65,536 点并加标准差 2 mm 噪声；验证使用固定 point cloud。**[代码明确]**
- LEAP 生成分布极不均衡：例如 low-link 的 0303/3030 约 3.7 万有效 grasps，而 0312/2203/3103 只有 2.1k–4.4k；论文承认这使 specialist 对两指手更占优势。

主要瓶颈不是原始数据数量，而是 **synthetic grasp label 的物理真实性、canonical action 的可执行性和 morphology coverage**。在当前静态抓取任务，69,917 capped samples 已能训练；扩大相似 LEAP variants 的数量不等价于扩大真实硬件拓扑多样性。

## 五、统一表示与对齐机制

### 5.1 Canonical URDF 与参数空间

基础表示共 82 个标量：

| 参数 | 维数 | 含义 |
|---|---:|---|
| `palm_radius` | 1 | cylinder palm radius |
| `finger_radius` | 1 | shared capsule radius |
| `finger_lengths` | 6 | thumb 3 links + shared non-thumb 3 links |
| `finger_xyz` | 15 | five finger base translations |
| `little_extra_origin` | 6 | special little-finger joint xyz+rpy |
| `thumb_rpy` | 3 | thumb base rotation |
| `thumb_axes` | 6 | two proximal thumb axes |
| `joint_lowers` | 22 | canonical lower limits |
| `joint_uppers` | 22 | canonical upper limits |

网络训练 VAE 时不直接输入连续的 44 个上下限，而把每个 joint 化为 active/inactive bit；再把两条 thumb axes 各做六方向 one-hot，最终是 **66D VAE input**。**[论文明确][代码明确]**

扩展表示为 173D：1 palm radius、5 finger radii、15 link lengths、72 个 joint origins、36 个 joint axes、44 个 joint limits。LEAP zero-shot loader 把 44 个 limits 压成 22 个 active bits，并额外附加 1 个标量，实际 hand condition 为 **152D**；模型配置的 `cond_dim=670=512+152+6`，没有使用 16D VAE。**[代码明确]**

基础版主动丢弃或假设：所有手最多五指；所有 fingers 共用半径；非拇指共享 link lengths 且共面；大量 joint origin/orientation/axis 固定；palm depth 绑定 finger diameter；复杂 mesh 统一为 cylinder/capsule。扩展版缓解前五项，但仍假设每指 distal joints 使用固定局部方向和 $+y$ flexion axis。

为什么可比较？因为每个槽位被赋予固定解剖语义与坐标约定：palm normal 为 $+x$，右手 thumb side 为 $+y$，finger-forward 为 $+z$；joint index 不再跟随原 URDF 文件顺序。它保留 gross morphology、kinematic chain、axis/limit 与 active DoF，丢弃 mesh detail、mass/inertia、friction、transmission、motor limits、compliance、sensor layout 和 tactile geometry。

### 5.2 Morphology latent

VAE 将 66D morphology vector 映射到 16D latent：

$$
q\xrightarrow{E}(\mu,\log\sigma^2),
\qquad
z=\mu+\sigma\odot\epsilon,
\quad \epsilon\sim\mathcal N(0,I).
$$

类型相关 reconstruction loss 为

$$
\begin{aligned}
\mathcal L_{\mathrm{cont}}
&=\lVert\hat q_{\mathrm{cont}}-q_{\mathrm{cont}}\rVert_2^2,\\
\mathcal L_{\mathrm{axis}}
&=\operatorname{CE}(\hat q_{\mathrm{axis}},q_{\mathrm{axis}}),\\
\mathcal L_{\mathrm{joint}}
&=\operatorname{BCE}(\sigma(\hat q_{\mathrm{joint}}),q_{\mathrm{joint}}),
\end{aligned}
$$

$$
\mathcal L_{\mathrm{VAE}}
=\mathcal L_{\mathrm{cont}}
+\mathcal L_{\mathrm{axis}}
+\mathcal L_{\mathrm{joint}}
+\beta\mathcal L_{\mathrm{KL}},
\qquad \beta=0.01.
$$

插值图说明 decoder 输出在视觉上平滑，但没有定量 reconstruction error、validity rate、latent neighborhood retrieval 或插值后控制性能。因此“structured/semantically rich manifold”主要是 qualitative evidence；真正 zero-shot LEAP 结果也没有使用该 16D latent，不能用 81.9% 反向证明 VAE latent 的必要性。

![Morphology interpolation](https://arxiv.org/html/2602.16712v2/fig/hand_interpolation.png)

### 5.3 人工配置量

官方 onboarding 不是“只给 URDF”：

1. 放入 original URDF/meshes。
2. 人工创建 meta JSON，至少给 `palm_origin` 与逐指 `joint_mapping`。
3. base parser 提取参数。
4. render canonical URDF。
5. 可视化 original/canonical overlay。
6. 有偏差时手工调参数并循环。

README 和代码都明确某些 URDF 需要人工修正；`--extended` parser 当前直接输出 `TODO` 并退出。作者在 issue #1 也明确 raw MANO 不支持，需先构造 MANO-based URDF + meta JSON。**[代码明确][作者材料]**

> 这篇论文的迁移能力主要来自 **URDF 语义槽位、坐标系和固定 joint action space 的显式对齐**，而不是简单扩大训练数据，也不主要来自 VAE 或 diffusion 架构。

## 六、观测与多模态迁移

| 模态 | 源域存在 | 目标域存在 | 对齐方式 | 部署可用 | 缺失处理 |
|---|---:|---:|---|---:|---|
| RGB | 否（policy input） | L515 可采 RGB，但论文只说明 depth/object observation | 不适用 | 证据不足 | 无说明 |
| 深度/点云 | 是，object mesh sampled point cloud | 是，RealSense L515 | object-frame point cloud encoder | 是 | 2 mm train noise；无 dropout |
| proprioception | 否 | 执行控制器必然有，但非 generator input | 不适用 | 非 policy input | 无 |
| object pose | wrist pose 相对 object frame | 需要建立 object frame | point cloud/object coordinates | 是 | 获取方式未详 |
| force/torque | 仅 evaluation perturbation | 非 policy input | 不适用 | 否 | 无 |
| tactile | 否 | 否 | 不适用 | 否 | 无 |
| contact state | filtering/evaluation 隐式使用 | 非 policy input | 不适用 | 否 | 无 |
| peer state | 不适用 | 不适用 | — | — | — |
| URDF/物理参数 | 是，geometry/kinematics | 是，target hand condition | canonical parameter vector | 必需 | 不可缺 |
| privileged state | simulator collision/force closure 用于 label/filter | 部署无 | 离线过滤，不是 distillation | 否 | 只部署已过滤策略 |

融合是 feature-level conditioning：DGCNN-like point encoder 产生 512D object feature，三手主模型拼 16D VAE hand latent 与 6D wrist rotation；LEAP 模型拼 152D direct hand condition 与 6D rotation。对象几何提供 task information，本体参数提供 embodiment information。

没有 teacher-student、modality dropout、时间序列同步、sensor latency、RGB appearance alignment 或触觉/力的跨模态蒸馏。URDF 在这里既是 metadata，也是 hand condition 的来源和生成 canonical simulator asset 的手工先验。称为“多模态输入”尚可，但称为丰富的 multimodal transfer 会夸大：真正进入 policy 的只有 point cloud + morphology metadata + desired rotation。

## 七、动作、意图与控制接口迁移

完整动作链为

$$
\text{object point cloud + target morphology + wrist rotation}
\rightarrow
(T,R,\theta^{\mathrm{can}})
\rightarrow
(T,R,M_h^{-1}(\theta^{\mathrm{can}}))
\rightarrow
\text{Franka wrist pose + hand joint targets}
\rightarrow
\text{low-level controller/contact}.
$$

抓取姿态定义为 wrist translation $T\in\mathbb R^3$、6D rotation representation $R\in\mathbb R^6$ 和 22D joint configuration $\theta$。第一阶段 diffusion 在显式给定 $R$ 时预测 $T$；第二阶段 MLP 根据 $(T,R)$、object feature 与 hand condition 回归 $\theta$：

$$
\mathcal L_{\mathrm{grasp}}
=\operatorname{SmoothL1}(\hat T,T)
+\operatorname{SmoothL1}(\hat\theta,\theta).
$$

这里没有在线 motion intent、trajectory 或反馈 action chunk；输出是一次性的静态 grasp reference。`M_h` 只做 index/sign conversion，不做 IK、MPC 或时序优化。Franka 把 wrist pose 变成 arm motion的具体 controller、collision avoidance、joint/velocity/acceleration limits 和 failure recovery 均未在论文/仓库公开。**[证据不足]**

| 可执行性约束 | 公开处理 | 判断 |
|---|---|---|
| Hand joint limits | canonical URDF 保存上下限；inactive joint 为零范围 | representation 层有；真机 hard-limit enforcement 未公开。 |
| 自碰撞 | grasp generation/filtering使用 hand collision geometry | 静态候选阶段部分处理；执行期监控未知。 |
| Hand-object penetration | Isaac Gym filtering + force-closure validator | 仿真有；canonical primitive geometry 与真机 surface 有 gap。 |
| Arm/environment collision | 无 Franka planner/controller 代码 | **[证据不足]**。 |
| 平衡 | 不适用 | 固定基座 Franka + hand。 |
| Actuator/torque limits | 不进入 morphology condition；driver细节缺失 | **[证据不足]**。 |
| Velocity/acceleration limits | 论文没有真实执行约束参数 | **[证据不足]**。 |
| 失败恢复/在线重规划 | 没有报告 | 一次性 grasp generation 不能证明 recovery。 |

统一 action space 在 tensor 层是可执行的：inactive joint limit 为 0，rendered URDF 与 validator 使用同一 22D order。但“同一个 action dimension 具有同一物理语义”只近似成立；相同角度在不同 link length/axis/coupling 下产生不同 fingertip displacement 和 wrench。Allegro axial rotation 的遗漏是已观察到的反例。

- **Policy-level zero-shot**：在 exact target grasp data 被排除时成立。
- **Controller-level zero-shot**：不成立；target joint mapping、hand driver 与 Franka control 必须配置。
- **System-level zero-shot**：不成立；相机、object frame、hand mounting、target extension 和安全执行都需专用工程。

## 八、本体、形态与物理属性迁移

| 属性 | 是否表示 | 方式/边界 |
|---|---:|---|
| URDF / kinematic tree | 是 | parsing source；输出 canonical URDF。 |
| semantic fingers | 是 | thumb/index/middle/ring/little，人工 mapping。 |
| graph adjacency | 间接 | 固定 canonical chains，不以 graph network 输入。 |
| link length | 是 | base shared；extended per finger。 |
| mass / inertia | canonical URDF 中有默认/生成值，但不是 condition 核心 | 未展示跨手 dynamics conditioning。 |
| joint axis | 是 | base 大量固定；extended 12 joints explicit。 |
| joint limit | 是 | 生成 URDF；VAE/LEAP network 使用 active bit 而非连续 bounds。 |
| velocity / torque limit | 否（network condition） | 低层控制细节未审查到。 |
| actuator / transmission | 否 | tendon、coupling、backlash、compliance 未表示。 |
| sensor layout / tactile | 否 | 不适用当前 policy。 |
| collision geometry | 简化 | palm cylinder + finger capsules；original mesh detail 丢失。 |

本体信息是显式输入，而非完全记在网络参数中；但新本体绝不是只给 URDF。人工语义 mapping、palm frame、可视化校验不可省，非标准手还需 schema/template 扩展。方法可以处理 0–22 active DoF 和三/四/五指 human-inspired tree；论文没有测试六指、并联、软体、欠驱动或非拟人拓扑。

LEAP target 是训练 family 的结构子图，物理参数几乎处于同一硬件设计族，属于 **组合/删减式外推** 而不是跨厂商的 arbitrary structural extrapolation。低 link-count variants 提供一定 OOD 证据，但两指 0303/3030 的 zero-shot 分别只有 46.9/36.2，显著低于 specialists 75.1/55.4，说明 canonical prior 在越像普通 gripper 时越弱。

如果 URDF 或 mapping 错误，condition 与 action mapping 会同时错：错误 finger availability 会产生无意义 joints，错误 sign/axis 会把 grasp closing 变成 opening，错误 link length 会使 wrist/hand pose碰撞。论文只做 wrong hand-condition 测试，没有连续 URDF noise/sensitivity。

目标专用部分包括：LEAP abduction/adduction joint relocation、extended 173D schema、152D condition、每个 variant 的 URDF/JSON、Lightning fingertip/active-joint config、Franka mount/driver 和真实相机控制栈。

## 九、Retargeting、Adaptation 与物理可执行性

OHRA 的 retargeting 是 **离散 joint correspondence**，不是逐帧/时序优化。meta JSON 中：

- `joint_mapping` 指定 original names 到 canonical semantic joints；
- `canonical_order` 以 1-based original indices 排成 22D，0 表示 inactive，负值表示 joint axis sign reversed；
- `palm_origin` 把 original frame 对齐到 canonical palm frame。

它保持 joint-level pose semantics 和 wrist/object relative pose，但不显式保持 contact location、contact force、contact timing 或 task-space fingertip trajectory。原始 grasp 转到 canonical 后会重新在 Isaac Gym 用六方向外力验证，因此 downstream utility 得到直接测试：每个方向施力 1 s，六次后 object displacement 小于 2 cm 才算 success。**[论文明确]**

物理可执行性来自三层：grasp synthesis、Isaac Gym filtering/force-closure test、目标 controller。运动学映射不等于动力学可执行；canonical mesh/axis simplification可改变接触。Table III 显示 Ours 从 canonical 映回 original 后 Barrett/Shadow 只变化 +0.6/-0.3 points，但 Allegro 下降 12.6 points；D(R,O) 从 original 映入 canonical 时 Shadow 下降 4.37 points。

新手加入：若 base schema 能表达，可复用 parser/mapping，不必为预训练 model 改 output dimension；但仍需建立 metadata、验证 canonical approximation 和准备控制 adapter。若要扩展 173D/新增手专用参数，condition dimension 变化将要求修改网络和重新训练。严格意义的 “direct deployment after URDF” 不成立。

## 十、交互、接触与 coupled dynamics

多人/双机器人 interaction：**不适用**。对于 dexterous contact：

1. 接触不是实测标签；来自 analytical grasp generation 与 simulator collision/perturbation validation。
2. policy 不输入 contact topology、force direction/magnitude、friction estimate 或 tactile。
3. 输出静态 grasp pose，不学习接触时序和闭环 force modulation。
4. force-closure evaluation 比纯几何接近更强，但仍基于 canonical simulator asset。
5. 真机对 10 objects 的成功计数说明 pose 有实用性，却没有接触 F1、slip、penetration、grasp wrench margin 或恢复统计。

因此 OHRA 学到的是 **object geometry × morphology → stable grasp pose distribution**，不是接触动力学策略。In-hand reorientation 使用在线 PPO/GRU 与 torque/work reward，但该实验分别为 original/canonical hand训练 policy，只验证 representation fidelity，不是跨形态共享控制或 zero-shot dynamic manipulation。

## 十一、训练策略与数据混合

```text
URDF + meta JSON
    -> canonical parameters / canonical URDF / 22D joint mapping
    -> 65,536 synthetic morphologies -> VAE (66D -> 16D)
    -> per-hand grasp generation + Isaac Gym filtering
    -> canonical grasp dataset
    -> shared DGCNN object encoder
    -> diffusion wrist translation + MLP joint pose
    -> simulation force-closure evaluation
    -> inverse mapping / Franka + LEAP real execution
```

三手主模型：

- object point encoder 512D；VAE hand latent 16D；rotation 6D；condition 534D。
- diffusion hidden `[512,256]`，1,000 train timesteps；inference 使用 10-step DDIM。
- joint ResidualNet hidden width 64；output 22D。
- Adam $10^{-3}$，cosine decay 到 $10^{-7}$；训练 1,000 epochs。
- batch 128；先 uniform sample robot，再 sample its grasp，因此 hand-level balanced。

LEAP zero-shot 模型：

- 直接输入 152D extended hand condition；不加载 VAE。
- condition 为 $512+152+6=670$ 维；其余 backbone 基本一致。
- capped dataset 69,917，batch 128，100 epochs，seed 配置为 42，但训练脚本的 `pl.seed_everything(cfg.seed)` 被注释，严格可复现性下降。**[代码明确]**
- validation config 在 9,19,…,99 epochs 上评估 `3033/3303/3330/3333`；公开材料未说明论文表格具体如何选择 epoch。

In-hand reorientation 是独立 pipeline：GRU hidden 256 + MLP `[512,256,128]`，PPO；LEAP 约 200M env steps，Shadow 约 500M；每手生成 10,000 initial grasps。它不与 grasp generator 联合训练。

没有 embodiment ID 或 learned embodiment-specific head，morphology condition承担区分。本体数据量不直接主导 batch，但论文没有训练本体数量 scaling curve、family diversity curve、negative transfer 统计或“更多本体是否持续提升”的实验。三手 unified 优于 specific 证明有限正迁移；LEAP 两指失败表明也存在 distribution mismatch。

## 十二、Zero-shot 纯度审查

| 项目 | 是否使用 | 是否违反 policy zero-shot | 说明 |
|---|---:|---:|---|
| 目标域训练轨迹/grasp | exact leave-one-out 模型：否 | 否 | 3033/3303/3330 分别排除 exact target data。 |
| 目标域无标签数据 | 是，URDF/params/assets | 通常不违反 | conditional zero-shot 允许 target description。 |
| 目标域 reward rollout | 评估使用 target simulator | 不自动违反 | 是否用于调参/选择未知。 |
| 目标域 finetuning | 否 | 否 | 论文明确 no further fine-tuning。 |
| 目标本体 URDF | 是 | 否，但降低纯度 | converter、asset 与 mapping 必需。 |
| 人工语义映射 | 是 | 否，但强 target configuration | palm/finger/joint correspondence。 |
| 目标专用 action mapping | 是 | 否，但强工程 | 22D↔original joints。 |
| 目标专用 controller | 是 | 否，但 system zero-shot 不成立 | Franka + LEAP execution。 |
| 目标专用 domain randomization | 未报告 | 证据不足 | 论文没有 DR 表。 |
| 真机参数辨识 | 未报告 | 证据不足 | calibration/system ID 细节缺失。 |
| 目标域超参数调节 | 可能 | 证据不足 | LEAP 专用 schema 与 training protocol 明显存在。 |
| 目标域 checkpoint selection | 代码支持多 epoch target validation | 证据不足 | 不能确认论文最终 checkpoint 是否看了 target 结果。 |

分类应为：**policy zero-shot，但需要本体配置、target-specific representation extension、action mapping 和 controller**。它不是严格 target-free zero-shot，也不是 few-shot adaptation，因为 exact target 数据不用于对应模型的梯度更新。对于 $x+y+z+w<8$ 实验，模型虽未见这些 sparse variants，但仍见过同一模板/关节语义和较完整的 family members。

## 十三、实验是否真正证明迁移

### 13.1 Canonical fidelity

| Policy | Steps-to-Fall ↑ | Cumulative Rotation ↑ |
|---|---:|---:|
| Shadow Original | 369.66 | 9.09 |
| Shadow Canonical | **390.62** | **10.92** |
| LEAP Original | **397.62** | 5.63 |
| LEAP Canonical | 326.98 | **6.31** |

Shadow 两指标提升；LEAP stability 下降约 17.8%，rotation 提升约 12.1%。这证明 canonical asset 能训练出可用 RL policy，不证明 dynamics 被无损保留，也不证明 reorientation policy 跨手迁移。

### 13.2 三手数据联合训练

| Training | Allegro | Barrett | Shadow | 平均 |
|---|---:|---:|---:|---:|
| Per-hand specific | 82.1 | 87.6 | 55.4 | 75.0 |
| **Unified** | **84.2** | **88.1** | **62.9** | **78.4** |

统一训练分别带来 +2.1/+0.5/+7.5 points，平均 +3.4 points，支持 data pooling 的价值；但三个 test hands 都参加训练，所以这张表不是 unseen-embodiment zero-shot。

### 13.3 与 grasp synthesis baseline 比较

| Method | Allegro | Barrett | Shadow | Time (s) |
|---|---:|---:|---:|---:|
| DFC | 76.2 | 86.3 | 58.8 | >1800 |
| GenDexGrasp | 51.0 | 67.0 | 54.2 | 19.71 |
| D(R,O) Grasp | **92.3** | 87.3 | **83.0** | 0.65 |
| OHRA | 84.2 | **88.1** | 62.9 | **0.13** |

OHRA 最大优势是 0.13 s 和统一数据接口，不是绝对 grasp quality；对 Allegro/Shadow 分别落后 D(R,O) 8.1/20.1 points。论文也明确这不是新 grasp algorithm，而是用轻量 downstream model 验证 representation。

### 13.4 Exact-variant leave-one-out

| Model | `3033` | `3303` | `3330` |
|---|---:|---:|---:|
| All Data | 76.1 | 85.4 | 43.3 |
| No `3033` Data | **67.8 zero-shot** | 83.4 | 31.5 |
| No `3303` Data | 81.5 | **81.9 zero-shot** | 46.9 |
| No `3330` Data | 74.7 | 81.6 | **36.3 zero-shot** |

对应 target 与 all-data oracle gap 为 -8.3、-3.5、-7.0 points。81.9% 是最强单点，但不能代表任意三指手；target 只是在同一 LEAP family 中删掉一个 finger/link pattern。

### 13.5 低 link-count 结构外推

| Variant | 0303 | 0312 | 2203 | 3030 | 3103 | 平均 |
|---|---:|---:|---:|---:|---:|---:|
| Unified zero-shot | 46.9 | **13.3** | **65.1** | 36.2 | **46.6** | **41.6** |
| Variant-specific | **75.1** | 12.1 | 33.9 | **55.4** | 18.5 | 39.0 |

Zero-shot 赢 3/5、平均只高 2.6 points；两个两指变体落后 specialist 28.2/19.2 points。这是有价值的失败边界：形态不变量在“仍像多指 dexterous hand”时更可靠，靠近 gripper topology 时 transfer 明显退化。

### 13.6 真机 sim-to-real

| Model | Apple | Band Aid | Coke | Cube | Football | Mayo | Orange | Pear | Sheep | Soccer | Total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `3333` trained | 8 | 7 | 9 | 7 | 10 | 6 | 8 | 9 | 10 | 9 | 83/100 |
| `3033` trained | 8 | 8 | 2 | 6 | 9 | 6 | 7 | 9 | 10 | 10 | 75/100 |
| `3033` zero-shot | 8 | 10 | 5 | 5 | 7 | 2 | 9 | 7 | 9 | 9 | 71/100 |
| `3303` trained | 7 | 8 | 5 | 3 | 9 | 4 | 9 | 7 | 9 | 9 | 70/100 |
| `3303` zero-shot | 9 | 6 | 4 | 5 | 9 | 5 | 8 | 6 | 9 | 10 | 71/100 |

真机每模型每对象 10 trials，总样本量比许多机器人论文扎实；3033 zero-shot 比 trained 低 4 points，3303 反而高 1 point。但没有置信区间、随机 trial order、失败类别、不同 checkpoint seeds 或 system reset protocol。真机只测 LEAP variants，没有 Allegro/Barrett/Shadow 或第三方未见硬件。

![Real-world grasp results](https://arxiv.org/html/2602.16712v2/fig/real/real_grasp.png)

### 13.7 数据划分与 baseline 完整性

| 必要证据 | 是否有 | 判断 |
|---|---:|---|
| target specialist | 是 | 三手 specific；低-link variant-specific；真机 trained。 |
| target 从零训练 | 部分 | variant-specific，可视为 target-only；训练预算/数据上限不完全等价。 |
| direct retarget/action mapping | 是 | canonical↔original replay。 |
| shared policy without alignment | 否 | 缺 raw URDF/naive padded baseline。 |
| shared backbone + embodiment head | 否 | 无此 baseline。 |
| pretrain + finetune | 否 | 不适用主 zero-shot claim，但可做 data-efficiency 对比。 |
| zero-shot shared policy | 是 | leave-one-exact-variant 与 low-link variants。 |
| oracle/upper bound | 是 | all-data/trained model。 |
| held-out embodiment | **within-family** | exact ID held out；canonical family、task、objects、simulator不 held out。 |
| held-out object | 三手实验是；LEAP 主协议否 | 不应混合两类 generalization。 |
| multiple seeds/statistical CI | 否 | 图 7 有 shaded std，但主成功率无 seeds/CI。 |

实验真正证明：canonicalized data 可联合训练；target morphology condition 被模型利用；同一 LEAP generative family 内能 leave-one-out transfer；仿真 grasp pose 可部署到对应真机 variants。它没有证明 arbitrary unseen hand、动态 manipulation、真实多模态 contact control 或 controller-agnostic system transfer。

## 十四、消融、内部一致性与关键图表

### 已有消融

1. original vs canonical URDF 的 reorientation fidelity。
2. canonical↔original action replay。
3. unified vs per-hand specific training。
4. leave-one-variant-out vs all-data oracle。
5. low-link zero-shot vs variant-specific。
6. wrong hand condition；错误 condition 会显著降成功率。
7. morphology parameter gradient 可视化；inactive finger gradient 较低。

### 提示词规定的 14 项消融逐项审计

| 消融要求 | 状态 | OHRA 中的证据/缺口 |
|---|---|---|
| 1. 去掉统一表示 | **缺失** | 没有 raw heterogeneous URDF/action 的 naive pad/shared baseline。 |
| 2. 去掉形态/物理输入 | **部分** | 有 wrong-condition，无 no-condition；形态条件是否必要未被纯净隔离。 |
| 3. 去掉 target mapping | **不适用/缺失** | 不映射就无法执行 variable-DoF output；应比较 learned/IK/direct original head。 |
| 4. MLP/Transformer 替代 graph | **不适用** | OHRA grasp backbone本来就是 DGCNN point encoder + MLP/diffusion，不是 morphology graph policy。 |
| 5. 单本体 vs 多本体 | **存在** | Specific vs Unified：平均 75.0→78.4。 |
| 6. 减少训练本体数量/多样性 | **缺失** | 无 hand-count/family-diversity scaling。 |
| 7. 去掉某模态 | **缺失** | 无 point-cloud-only、morphology-only、rotation removal或noise/dropout系统实验。 |
| 8. 去掉 contact/interaction reward | **不适用** | grasp generator为监督学习；contact仅在数据生成/验证，不是 policy reward。 |
| 9. 去掉 peer observation | **不适用** | 单手，无 peer。 |
| 10. 去掉 domain randomization | **缺失** | 没有报告 DR，也没有 sim-to-real DR ablation。 |
| 11. 去掉 teacher-student | **不适用** | 主抓取模型没有 teacher-student distillation。 |
| 12. 去掉 whole-body controller/投影 | **不适用/缺失** | 固定基座 arm-hand grasp；真实 controller未公开，无法消融。 |
| 13. 不准确 URDF/物理参数 | **缺失** | wrong hand identity 不是连续 URDF noise；无 axis/limit/link perturbation。 |
| 14. 更大 topology gap | **部分** | low-link LEAP variants提供 family内结构外推；无未见商用/非拟人/软体手。 |

### 关键缺失消融

- 去掉 morphology condition。
- 16D VAE latent vs raw 66D base parameters。
- LEAP 152D direct condition vs 16D latent。
- base canonical vs LEAP-specific extended canonical。
- 去掉 canonical action mapping、仅 pad 原 joint vector。
- 错误/noisy joint axis、limit、link length 与 sign mapping。
- 训练 hand 数量/拓扑多样性 scaling。
- 同数据量下多手 diversity vs 单手数据量。
- capsule collision vs original meshes。
- point-cloud noise、occlusion、camera calibration error。
- domain randomization、controller/hardware adapter、real feedback。
- 六指、欠驱动、软体、tendon-coupled 或非拟人结构。

### 论文内部需要核对的一处矛盾

wrong-condition 表的正文说测试 target 是 `leap_3033`，但表中最高且加粗的“正确”condition 行写成 `leap_3303`，而 `leap_3033` condition 反而只有 33.9/12.8。按数字更可能是正文 target 名或表中行名至少一处写错。**[论文明确存在不一致]** 这不推翻整体 zero-shot 表，但使该项 condition ablation 不能被逐字接受。

### Figure 1–13 / supplementary 在线索引

所有图片均使用 arXiv 或项目页在线 URL，没有本地路径。

| 图 | 内容 | 在线图片 |
|---|---|---|
| Fig. 1 | OHRA 总览：canonical params/URDF → cross-hand learning | [teaser](https://arxiv.org/html/2602.16712v2/fig/teaser.png) |
| Fig. 2 | 七只手 original/canonical initial/grasp overlay | [hand comparison](https://arxiv.org/html/2602.16712v2/fig/hand_visualization.png) |
| Fig. 3 | canonical mesh/frame 与 22-joint tree | [mesh/frame](https://arxiv.org/html/2602.16712v2/fig/canonical_URDF.png) · [tree](https://arxiv.org/html/2602.16712v2/fig/canonical_URDF_tree.png) |
| Fig. 4 | global/local URDF frame inconsistency | [global](https://arxiv.org/html/2602.16712v2/fig/frame_global.png) · [local](https://arxiv.org/html/2602.16712v2/fig/frame_local.png) |
| Fig. 5 | VAE morphology interpolation | [interpolation](https://arxiv.org/html/2602.16712v2/fig/hand_interpolation.png) |
| Fig. 6 | two-stage grasp generation | [pipeline](https://arxiv.org/html/2602.16712v2/fig/grasp_pipeline.png) |
| Fig. 7 | canonical/original reorientation training curves | [RL curve](https://arxiv.org/html/2602.16712v2/fig/RL_curve.png) |
| Fig. 8 | objects 与真机抓取 | [objects](https://arxiv.org/html/2602.16712v2/fig/real/setting.jpg) · [results](https://arxiv.org/html/2602.16712v2/fig/real/real_grasp.png) |
| Fig. 9 | 对 hand condition 的梯度 | [3333](https://arxiv.org/html/2602.16712v2/fig/grad_leap_3333.png) · [3033](https://arxiv.org/html/2602.16712v2/fig/grad_leap_3033.png) |
| Supplement | original/canonical in-hand rollout | [grid](https://arxiv.org/html/2602.16712v2/grid_2x2.png) |
| Supplement | 三手统一 policy grasp | [grasp visualization](https://arxiv.org/html/2602.16712v2/fig/grasp_vis.png) |
| Supplement | direct target inference vs 3333 action mapping | [mapping comparison](https://arxiv.org/html/2602.16712v2/fig/map_compare.png) |
| Supplement | 256 LEAP canonical variants | [LEAP variants](https://arxiv.org/html/2602.16712v2/fig/LEAP_variants.png) |
| Supplement | real experiment panels | [panel I](https://arxiv.org/html/2602.16712v2/fig/real/real_vis_1.png) · [panel II](https://arxiv.org/html/2602.16712v2/fig/real/real_vis_2.png) |

![Two-stage grasp pipeline](https://arxiv.org/html/2602.16712v2/fig/grasp_pipeline.png)

## 十五、灵巧手多模态与跨形态专项分析

| 专项问题 | OHRA 判断 |
|---|---|
| 不同 DoF/topology 如何统一 | 最多五指、22 joint slots；inactive joints 零范围；original↔canonical map。 |
| 是否 morphology-aligned graph | 不是 graph policy；是固定语义 tree/template。 |
| 节点/关节语义是否人工 | 是，`joint_mapping` 与 palm origin 需要人工 metadata。 |
| hand state | 静态 morphology condition；不输入在线 joint state。 |
| object state | 512-point object point cloud + desired wrist rotation。 |
| contact/tactile | 不输入；仅离线 filtering/eval 隐式使用。 |
| URDF physical property | geometry、origins、axes、limits；不含充分 dynamics/actuator metadata。 |
| hand-agnostic motion primitive | 没有 FLEX/ABD primitive policy；统一的是 22D canonical joint pose。 |
| canonical→joint command | fixed permutation/sign/inactive mapping；target-specific。 |
| 不可执行 action | inactive joints + simulator filtering；无在线 constraint projection。 |
| 多手是否优于单手 | 是，三手平均 +3.4 points，但都是 seen hands。 |
| truly unseen hand | exact LEAP variant unseen；hardware family/template seen。 |
| 非拟人/不同拓扑 | Barrett 参加训练；没有 held-out non-anthropomorphic hand。 |
| unseen hand + unseen object | 未作为统一 factorial protocol证明；三手跨对象与 LEAP跨手分开。 |
| 真机是否再训练 | zero-shot model不在 exact target grasp 上训练；system configuration 明显存在。 |
| 多模态是否互补 | 只证明 point cloud + morphology condition；没有 modality ablation。 |

专项结论：OHRA 的核心不是学习一个 morphology-aligned graph policy，而是定义一个 **数据工程 contract**。对已经能被五指/22-joint template 表达的刚性拟人手，这个 contract 很有价值；对 topology 差异大的手，手工 schema extension 可能把“general representation”退化为持续增加例外字段。

## 十六、局限性、失败模式与潜在水分

### 优点

1. **问题定义落在接口层**：不是只提出一个新网络，而是把 URDF parsing、固定形态 schema、action mapping、数据转换和 policy conditioning 接成可复用链路。
2. **证据链覆盖 representation → policy → hardware**：既比较 original/canonical fidelity，也比较 unified/specific、seen/held-out、simulation/real。
3. **给出了有信息量的失败边界**：两指 0303/3030 显著落后 specialist，说明作者方法并非在所有 topology 上无损。
4. **代码资产比典型论文完整**：base/extended canonical assets、256 LEAP variants、数据 loader、train/eval 与 generation dependency 均已公开。
5. **对数据平台直接有用**：semantic joint slots、round-trip mapping 与 morphology metadata 可以独立于论文 grasp backbone 被复用。

### 局限性

1. **Family-level extrapolation 有限**：zero-shot target 来自同一 LEAP modular family；删 link 不等于新 actuator/transmission/topology。
2. **target-specific extension 隐藏在表示层**：LEAP abduction joint 被专门移到第二 link，并仅在该实验启用。
3. **缺 dynamics/contact metadata**：mass、friction、compliance、motor、tendon、sensor layout 不作为 condition，难覆盖真实动态灵巧操作。
4. **任务过窄**：主 zero-shot 是静态 grasp pose synthesis；in-hand rotation不是共享/zero-shot policy。
5. **部署不可复现**：真机 controller、calibration、object segmentation/frame、safety 与 failure definition 未开源。
6. **VAE claim 与 strongest result 脱节**：LEAP strongest zero-shot 直接用 152D raw extended parameters。
7. **统计证据不足**：主表没有 seeds/CI；真实 trial虽多但无 failure taxonomy。
8. **parser 自动化有限**：base 仍需 meta/manual repair；extended parser未实现。
9. **许可证边界**：仓库 root 无 LICENSE；数据 license 未在 README 明示，平台纳入前必须澄清。
10. **checkpoint/model selection 不透明**：无 pretrained weights；validation config遍历 target epochs，最终选择规则未交代。

### 潜在改进

1. 冻结 base/extended schema 后，在至少 3–5 个未参与设计的商用手上做 leave-one-hardware-out，避免通过 target-specific schema extension吸收困难。
2. 把 mass/inertia、friction、transmission、underactuation、torque/velocity limits、tactile layout 加入 robot metadata，并做 noisy-URDF sensitivity。
3. 用 variable-topology graph/set encoder替代固定五指槽位，同时保留 semantic action adapter，覆盖六指、欠驱动与非拟人手。
4. 报告 VAE16、raw66、extended152、no-condition 和 wrong-condition 的统一消融，并校正论文表中 `3033/3303` 标签矛盾。
5. 发布 checkpoints、immutable dataset manifest/license、environment lock、target-blind checkpoint rule 与真机 calibration/controller代码。
6. 从一次性 grasp pose 扩展到带 tactile/contact history 的闭环 regrasp/in-hand manipulation，并把 failure recovery 作为正式指标。

### 预期失败的目标本体

- 六指或更多 fingers；非 palm+finger tree；并联闭链。
- tendon-coupled/underactuated hand，joint command 与真实独立 DoF 不一致。
- soft/continuum hand，link/joint URDF 本身不适用。
- finger axes/origins偏离 base/extended assumptions 的机械手。
- collision geometry强依赖复杂 fingertip/palm surface而非 capsules 的手。
- 需要 tactile slip recovery、force control 或动态 regrasp 的任务。

### 组会上最值得问的五个问题

1. 如果把一个完整未见商用手（而不是 LEAP 删 link 变体）作为 target，且冻结 schema，不新增专用参数，zero-shot 成功率是多少？
2. 为什么 strongest zero-shot 使用 152D direct condition 而非论文强调的 16D VAE latent？二者在同一 split 上的消融在哪里？
3. LEAP 专用 joint relocation 若去掉会下降多少？它应当被视为 general extensibility，还是 target-specific representation engineering？
4. 目标手 validation config 扫描多个 epochs；论文结果的 checkpoint 是否利用了 target simulator success，若是应如何定义 zero-shot model selection？
5. 真机抓取如何获取 object frame、规划 Franka wrist trajectory、处理碰撞/失败并判定成功？为何这些关键 system-transfer 代码未发布？

## 十七、代码、数据与复现审查

### 17.1 仓库识别与维护状态

| 仓库/资源 | 类型 | 与论文关系 | 完整度 | 可信度 | 截至 2026-08-12 备注 |
|---|---|---|---|---|---|
| [zhenyuwei2003/OHRA](https://github.com/zhenyuwei2003/OHRA) | 官方 | canonical assets、VAE、grasp/zero-shot train/eval | 中 | 高 | 11 commits；last push 2026-07-06；37 stars/3 forks；2 open issues；无 releases。 |
| [项目主页](https://zhenyuwei2003.github.io/OHRA/) | 官方作者材料 | paper/video/results | 中 | 高 | 实验表和大量真机视频；不是代码仓库。 |
| [Google Drive dataset](https://drive.google.com/file/d/1Ery6WAayDxKgYpIbu7l8DXXVYV19rw78/view?usp=sharing) | 官方 README 链接 | 训练数据 | 待下载审查 | 中 | 当前 landing page 返回 HTTP 200；原链接曾被删除，issue #2 仍 open，2026-07-06 README 更新链接。 |
| `third_party/lightning-grasp` | 第三方代码内嵌 | LEAP grasp generation | 中 | 中 | 仓库内带其 LICENSE；OHRA root license 仍缺失。 |

### 17.2 公开完整性

| 组件 | 公开状态 | 证据/问题 |
|---|---|---|
| 原始/多手 URDF assets | **是** | Allegro、Barrett、Dex3、Gaia、LEAP、MANO、Shadow、Sharpa 等。 |
| Canonical base/extended assets | **是** | JSON、URDF、Jinja templates。 |
| Base URDF parser/render | **是** | `urdf_parser.py` / `urdf_render.py`。 |
| Extended parser | **否** | 代码明确 `TODO` 后退出；只有手工准备好的 extended examples。 |
| VAE train/eval | **是** | config、model、interpolation/reconstruction scripts。 |
| 三手 grasp train/eval | **是** | loader、diffusion/MLP、Isaac validator、configs。 |
| LEAP variant generation | **是** | 256 URDF/JSON、Lightning configs/scripts。 |
| Zero-shot split/train/eval | **是，需数据** | `geq8_max200` config；target list与多 epoch validation。 |
| 数据格式 | **部分** | `.pt` loaders清楚；缺独立 schema/version/license 文档。 |
| Dataset | **链接存在** | 未在本次审查下载大文件；旧链接失效历史需注意。 |
| Checkpoint | **否** | tree 无 `.pt/.pth/.ckpt` pretrained weights。 |
| 环境锁定 | **弱** | README 仅列 pip packages，无 versions/lockfile；依赖 Python≤3.8 + Isaac Gym Preview 4。 |
| In-hand RL full reproduction | **部分** | IsaacGym controller/validator 有；与主 grasp pipeline关系、完整训练命令不够清晰。 |
| 真机部署 | **否** | 无 Franka/LEAP driver、camera processing、motion planning。 |
| 标定流程 | **否** | camera-hand/object frame 与 mount calibration 不公开。 |
| Root license | **否** | repository metadata license 为 null；只有部分 imported assets 子许可证。 |
| CI/tests/release | **否** | 无 release、package、测试矩阵或 automated reproduction。 |

一个容易踩坑的实现差异：README 说配置在 `config/`，实际目录是 `src/configs/`；repository structure 中 canonical/extended 两处都把第二个子目录误标成 `json/`，实际是 `urdf/`。这些是文档小问题，但说明复现前应以 tree/config 为准。

### 17.3 新本体接入成本

| 工作 | 最低要求 | 风险 |
|---|---|---|
| Robot asset | URDF + resolved meshes/collision | URDF convention、mesh scale、coupling不一致。 |
| Semantic metadata | palm origin、five-finger assignment、joint mapping/order/sign | 必须人工判断；最容易 silent error。 |
| Canonicalization | run base parser/render/overlay | 非标准结构需人工修参数。 |
| Extended support | 手工生成 extended params；必要时改 Jinja/parser | 官方 parser尚不支持 automatic extended extraction。 |
| Data | zero-shot推理理论上无需 target grasps | 现实中需 simulator validation和安全测试。 |
| Policy | condition schema不变可冻结；新增字段需 retrain | target 超出 distribution 时无保证。 |
| Controller | 22D↔original adapter、arm-hand synchronization、安全 limits | 未提供通用 interface。 |
| Perception/calibration | object point cloud/object frame、camera extrinsics | 真机代码缺失。 |
| Reward | static grasp predictor无需新 reward | 若扩到闭环/dynamic task需重新设计。 |
| 预计周期 | representable simulated hand：1–3 周；真实新硬件：4–8+ 周 | 这是[合理推断]，取决于 driver/URDF质量与安全验证。 |

## 十八、分层复现路线

### Level 1：最小迁移验证

目标：验证 `original URDF → canonical params/URDF → action mapping`，不先追整篇成功率。

- 依赖：Python 3.8、PyTorch、pytorch_kinematics、trimesh、Jinja2、Viser；不做 physics 时可暂不装 Isaac Gym。
- 数据：仓库自带一只 original hand、base meta JSON、canonical JSON/URDF。
- 硬件：无；CPU 即可做 parse/render/overlay。
- 步骤：对 LEAP/Barrett 复跑 parser；重建 URDF；采样 100–1,000 joint poses；比较 fingertip FK、joint limits、collision proxy 与双向 mapping round-trip。
- 成功标准：active joint round-trip error 接近数值精度；关键 fingertip pose error有阈值；无 sign/order silent mismatch；能复现 overlay。
- 风险：parser interactive 写文件、extended parser缺失、original mesh/license路径。

### Level 2：论文级复现

目标：复现三手 Table V 与 LEAP leave-one-out Table VII，重做关键缺失消融。

- 依赖：NVIDIA GPU、Isaac Gym Preview 4、Python≤3.8；安装 README packages并自行锁版本。
- 数据：Drive dataset、object meshes/point clouds、canonical filtered `.pt`、LEAP capped dataset。
- 训练：VAE 1,000 epochs；三手 grasp 1,000 epochs；LEAP zero-shot 100 epochs。
- 评价：每 `(hand,object)` 100 samples；六方向各 1 s perturbation；object displacement <2 cm。
- 建议至少 3 seeds，并提前冻结 checkpoint selection 规则；不能看 target SR 选 epoch。
- 必做消融：raw 66D vs VAE16；base vs extended；no hand condition；wrong/noisy parameters；LEAP extension off；same data budget specific/unified。
- 成功标准：三手 SR 误差 ≤3–5 points；3033/3303/3330 zero-shot趋势一致；报告 bootstrap CI 与 seed variance。
- 最大风险：无 checkpoint/lockfile、Drive dataset history、Isaac Gym老版本、paper split/epoch selection未完整文档化。

### Level 3：平台化扩展

建议统一数据接口：

```yaml
sample:
  object:
    point_cloud: float32[N, 3]
    frame: object
  embodiment:
    robot_id: string
    urdf_uri: string
    canonical_schema_version: ohra-base-v1
    canonical_params: float32[82]
    active_joint_mask: bool[22]
    original_to_canonical: int32[22]
    sign: int8[22]
    palm_T_original: float32[4, 4]
  action:
    wrist_translation: float32[3]
    wrist_rotation_6d: float32[6]
    canonical_joint_position: float32[22]
  provenance:
    generator: string
    simulator: string
    validation_protocol: string
    source_license: string
```

平台 API：

```text
parse_robot(urdf, semantic_meta) -> canonical_robot
map_action(q_original, robot_meta) -> q_canonical
unmap_action(q_canonical, robot_meta) -> q_original
validate_fk(original, canonical, pose_set) -> geometry_report
validate_grasp(robot, object, grasp) -> force_closure_report
controller_adapter.execute(wrist_pose, q_original) -> execution_log
```

Regression tests：

1. schema/version/units validation；所有 URL/license resolvable。
2. joint mapping bijection for active DoFs；sign round-trip。
3. random-pose fingertip FK error与joint limit coverage。
4. original/canonical collision/contact discrepancy。
5. known-hand golden grasp SR。
6. leave-one-embodiment split leakage check。
7. checkpoint selection不得读取 target labels。
8. controller saturation、self-collision、安全 stop。
9. point-cloud noise/occlusion/calibration stress test。
10. 新手接入报告必须记录人工修改字段与耗时。

平台化前应新增 morphology coverage taxonomy：不要把 50 个同族删 link variants当作50个独立硬件类别；至少按 actuator/transmission、topology、finger count、axis layout、underactuation、collision geometry分层采样。

## 十九、与代表性迁移范式对比

| 维度 | OHRA | 人类示范迁移范式 | 交互迁移范式 | 跨形态统一策略范式 |
|---|---|---|---|---|
| Source data | synthetic/static robot grasps | human video/mocap/wearable | human-human paired motions | multi-robot rollouts/demos |
| Target system | held-out LEAP variant + Franka | robot arm/humanoid/mobile manipulator | dual humanoids/robots | unseen robot/hand |
| 主要 gap | URDF topology/DoF/action order | human appearance/kinematics/control | pair scale/contact/coupled dynamics | state/action/dynamics mismatch |
| 统一表示 | 82/173D params + canonical URDF | hand-eye/task-space/object-centric | interaction/contact graph | graph/tokens/URDF metadata |
| 动作接口 | wrist pose + fixed 22D joints | task-space reference + WBC/IK | paired references + multi-agent control | shared primitive + target map |
| retargeting | deterministic joint remap | 常需 trajectory retargeting | 强，需要双主体约束 | 取决于方法 |
| target mapping | **需要** | 通常需要 | 强需要 | 通常需要 |
| contact modeling | filtering/eval，policy 不建模 | 常隐式 | 核心 | 灵巧任务通常需显式/privileged |
| 联合训练 | 三手/66 LEAP variants | 常为单 target | 双 agent | 核心 |
| zero-shot 层级 | policy/data-space；within-family | policy-level，系统工程多 | reference/policy-level | held-out embodiment policy-level |
| sim-to-real | static grasp pose部署 | perception+controller bridge | localization/comms/contact | DR/distillation/controller adapter |
| 最大优势 | 清晰可实现的数据与 action contract | 低成本 human data | 保留互动几何 | 直接学习跨 morphology prior |
| 最大局限 | human-inspired template +缺 dynamics | embodiment/visual gap | 安全、定位、耦合难 | negative transfer与target config |

与 DexGrasp-Zero 一类 morphology-aligned graph policy 相比，OHRA 更偏 **canonical asset/schema**：优点是容易接到现有 grasp generator 和数据平台，缺点是固定 22D template 牺牲拓扑开放性。与 human-demo 方法相比，它没有 human→robot gap，却也没有 robot-free data scaling。与 interaction retargeting相比，OHRA完全不处理时间、partner与coupled contact。

## 二十、最终判断

1. **真正迁移了什么**：多手/多变体的 synthetic grasp labels、共享 grasp generator 参数，以及由 morphology condition 解释的 22D joint pose semantics。
2. **能力主要来自哪里**：首先是 representation/action schema，其次是数据联合训练；VAE与diffusion是可替换实现，低层controller不贡献论文中的 cross-hand learning但决定真机能否落地。
3. **最值得学习的三个设计**：
   - 把 URDF 变成版本化、可学习且可反向生成的 canonical contract；
   - inactive joint slots + explicit mapping 统一 variable DoF action labels；
   - 用 leave-one-exact-morphology 和 target specialist 同时报告 transfer/oracle gap。
4. **最不值得高估的三个 claim**：
   - 81.9% 不等于对任意新手 81.9%，它是 LEAP family 内一个 target；
   - “canonical URDF preserves dynamics” 证据仅两手、指标有取舍，且未编码完整 dynamics；
   - “foundation for general manipulation” 尚未由 trajectory、tool use、regrasp、tactile或dynamic zero-shot证明。
5. **数据复用价值**：**高（对拟人刚性手的静态 grasp）/中（广义 manipulation）**。
6. **跨本体泛化可信度**：**中**；within-family 强，跨商用未见硬件证据不足。
7. **真实部署成熟度**：**中低**；真实 trials 足够，但部署栈没有公开。
8. **代码复现可行性**：**中**；核心代码/资产/数据链接有，checkpoint、lockfile、license、extended parser与real code缺失。
9. **是否值得精读**：**值得**，作为 embodiment schema/data engineering 论文。
10. **是否值得复现**：**值得有条件复现**；先做 converter/FK fidelity，再做zero-shot表。
11. **是否纳入团队数据平台**：**有条件值得**；先补 schema version、license、provenance、mapping tests与真实controller API。
12. **最适合借鉴的模块**：canonical robot metadata + fixed semantic action slots + original/canonical round-trip validation。

> 这篇论文本质上是一项从 **异构灵巧手的 URDF 与独立合成抓取数据** 到 **统一形态条件和 canonical joint-action space 下的共享抓取生成器** 的迁移工作。它真正建立的不变量是 **拟人手掌—手指链上的固定关节语义与坐标约定**，主要通过 **82/173 维形态参数、canonical URDF、22D inactive-slot action 与确定性 joint mapping** 缩小域差异；其迁移能力的上限取决于 **目标手能否被该五指/22-joint template 忠实表示，以及 synthetic contact 在真机上是否成立**。论文最有价值的部分是 **把跨手数据复用落实为可执行的数据工程接口并给出 family-level zero-shot 真机证据**，但需要警惕它仍然依赖 **人工 meta mapping、LEAP 专用扩展、目标控制器和未开源部署栈**，因此其 zero-shot claim 应被理解为 **exact target grasp data 不参与梯度更新的 morphology-conditioned policy transfer，而非任意新手的 controller/system-level 零配置迁移**。
