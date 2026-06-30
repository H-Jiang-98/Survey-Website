---
title: "Scalable Trajectory Generation for Whole-Body Mobile Manipulation"
method_name: "AutoMoMa"
authors: [Yida Niu, Xinhai Chang, Xin Liu, Ziyuan Jiao, Yixin Zhu]
year: 2026
venue: CVPR
tags: [mobile-manipulation, whole-body, loco-manipulation, trajectory-optimization, data-generation, GPU-acceleration, imitation-learning, articulated-objects]
zotero_collection: _inbox
image_source: online
arxiv_html: https://arxiv.org/html/2604.12565v1
created: 2026-06-29
---

# Scalable Trajectory Generation for Whole-Body Mobile Manipulation

## 元信息

| 项目 | 内容 |
|------|------|
| 作者 | Yida Niu, Xinhai Chang, Xin Liu（共同一作）, Ziyuan Jiao, Yixin Zhu（共同通讯） |
| 机构 | 北京大学人工智能研究院、北京大学心理与认知科学学院、北京航空航天大学无人系统研究院、通用人工智能国家重点实验室、PKU-武汉人工智能研究院等 |
| 会议 | CVPR 2026 |
| 类别 | Humanoid / 全身移动操作（loco-manipulation） |
| 日期 | 2026（arXiv v1） |
| 项目主页 | https://automoma.pages.dev/ |
| 链接 | [arXiv](https://arxiv.org/abs/2604.12565) / [PDF](https://arxiv.org/pdf/2604.12565) |

---

## 一句话总结

> 用 GPU 加速的 [[AKR]]（增广运动学表示）建模 + 并行轨迹优化，把全身移动操作的物理有效轨迹生成提速 80×（5000 episodes/GPU-hour），产出 50 万条跨 330 场景/多本体/铰接物的数据集，并实证“数据稀缺而非算法”才是该任务的瓶颈。

---

## 核心贡献

1. **GPU 加速的 AKR 规划器**: 把 base、arm、object 三套运动学统一进一条串行链（[[AKR]]），并将轨迹优化与碰撞检测整体批量化到 GPU 上，达到 **5,000 episodes / GPU-hour**，相比 CPU 基线（约 60 条/小时）提速 **80×**，从根本上解开了全身移动操作数据采集的吞吐瓶颈。
2. **大规模高保真数据集**: 产出 **50 万+** 条物理有效、关节空间（joint-space）全身协调轨迹，覆盖 **330 个真实风格场景**、多种铰接物（[[PartNet-Mobility]]）与多种机器人本体（Summit-Franka、TIAGo、R1），在规模、多样性、运动学保真三者上同时达标——这是先前数据集无法兼顾的。
3. **数据是瓶颈的实证**: 通过下游 [[Imitation Learning|IL]] 策略（[[DP3]]、[[Diffusion Policy|DP]]、[[ACT]]）的系统性 scaling 实验，证明哪怕**单一铰接物任务**也需要**数万条示范**才能让 SOTA 方法逼近 80% 成功率，确认“data scarcity—not algorithmic limitations”才是约束所在。

---

## 问题背景

### 要解决的问题
全身移动操作（whole-body mobile manipulation）要求**同时**协调移动底盘与机械臂去与世界交互。相比固定底座操作，底盘的额外移动性把搜索空间在“整个房间”尺度上**指数级**放大，在严格的铰接与碰撞约束下，有效运动学解极其稀疏。学习可靠策略因此需要**比固定底座操作大几个数量级**的数据集。核心瓶颈是：**缺一个能跨本体、跨环境大规模生成物理有效、协调轨迹数据的可扩展流水线**。

### 现有方法的局限
- **遥操作**（[[Mobile ALOHA]]、TeleMoMa、BRS）: 保真度高但**劳动密集、不可扩展**，受操作者疲劳与硬件限制，数据量止步于千级（如 Mobile ALOHA 仅 276 条）。
- **仿真 RL**: 自动采集但探索代价高、sim-to-real gap 顽固。
- **规划法（CPU 实现）**: 物理有效但**算力代价高得离谱**——[[AKR]] 框架的 CPU 求解器只能生成 **60 条轨迹/小时**。
- **端到端学习法**（如 OmniGibson 风格）: 用神经推理替代穷举搜索，但仍被**稀缺的协调全身示范**卡住。
- 结果：现有数据集只能在**规模、多样性、运动学保真**三者间妥协（见 Table 1），全身移动操作的可泛化策略学习基本未解。

### 本文的动机
[[AKR]] 是该任务一个原理上正确的统一建模基础（把 base/arm/object 合成一条链做约束感知规划），但其潜力被**吞吐量**死死压住。作者的核心洞见是：把 AKR 建模与**现代并行计算（GPU）**结合——批量化轨迹优化与碰撞检测——就能在保持运动学保真的前提下把吞吐拉高两个量级，从而第一次让“高性能规划 + 大规模 IL 学习”在全身移动操作上同时成立。

---

## 方法详解

### 模型架构

AutoMoMa 是一条 **任务规约 → 问题实例化 → 轨迹生成 → 渲染** 的四阶段 GPU 加速数据生成流水线（见 Figure 3），核心建模载体是 [[AKR]]：

- **输入**: 任务规约三元组 $(\mathcal{S},\mathcal{O},\mathcal{R})$ —— 场景 $\mathcal{S}$、物体集 $\mathcal{O}=\mathcal{O}_{\text{rigid}}\cup\mathcal{O}_{\text{art}}$、机器人本体 $\mathcal{R}$。
- **核心建模**: [[AKR|增广运动学表示]] 把 base、manipulator、object 统一为**一条串行运动学链**，在统一配置空间内联合优化。
- **求解**: [[GPU 加速运动规划]]（基于 cuRobo 风格批处理），对轨迹优化、IK、碰撞检测整体并行。
- **输出**: 物理有效的**关节空间全身轨迹** $\boldsymbol{x}_{1:T}$ + Isaac Sim 渲染的多模态观测（egocentric / fixed RGB-D、点云）。
- **吞吐**: 5,000 episodes / GPU-hour（约 80× 于 CPU 基线）。

### 核心模块

#### 模块1: Augmented Kinematic Representation（AKR，增广运动学表示）

**设计动机**: 把原本相互独立的“机器人运动学树 + 物体运动学树 + 抓取位姿”三者，融合成**一条根植于世界坐标系的串行链**，使 base/arm/object 的状态落入**同一配置空间**，从而能用一次联合优化统一施加任务目标与运动学约束（见 Figure 2）。

**具体实现**:
- 取三类输入: 机器人运动学树、物体运动学树、末端执行器与物体可附着帧之间的变换（即**抓取位姿**）。
- 通过插入一个**虚拟关节（virtual joint）**把物体附着到机器人末端；这要求**反转物体的运动学模型**，让“附着链接 / 抓取点”成为新的运动学根（kinematic root）。
- 关键难点：链反转**不只是翻转父子关系**——所有相关变换、分支结构都必须严谨更新（因为 revolute/prismatic 关节通常相对子链接帧定义运动），且分支几何须在优化中保留以保证碰撞安全（详见公式 A1/A2）。
- 通过引入**虚拟底座（virtual base）**——两个正交 prismatic 关节 + 一个 revolute 关节——把移动底盘的平面运动接入世界帧与机器人底座之间，始终维持严格串行结构。
- 最终链：根于 world link，止于物体的环境锚点（如固定的橱柜底座）。

#### 模块2: 四阶段数据生成流水线

**设计动机**: 把“原始场景资产”系统地转成“可规划基元”，再批量求解、批量渲染，以实现高吞吐。

**具体实现（四阶段）**:
1. **Task Specification（任务规约）**: 接收三元组 $(\mathcal{S},\mathcal{O},\mathcal{R})$。场景来自两条互补路径——(i) 程序化生成带铰接电器的交互场景（[[Infinigen]]），(ii) 把已有场景数据集（[[AI2-THOR]] / iTHOR）的静态电器替换为功能等价的铰接体。本体附带**球体碰撞近似**、自碰撞掩码、关节权重向量 $\boldsymbol{w}\in\mathbb{R}^{n+m+3}$。
2. **Problem Instantiation（问题实例化）**: 把场景转成 [[ESDF|欧氏符号距离场]] 以加速碰撞查询，并把查询限制在由起止状态界定的轴对齐包围盒内；组装 AKR 链 $\mathcal{K}_{\text{akr}}$；用**拟合球体**近似链接几何（下采样防止体积高估）。还用**动态碰撞策略**处理两个阶段：approach 阶段临时清掉与物体相交的体素、换回高精网格防离散化误差挡住抓取位姿；manipulation 阶段把物体并入 AKR 链、移除其静态环境网格，只保留严格在物体当前体积外的体素，消除与初始状态的假阳性碰撞。
3. **Trajectory Generation（轨迹生成）**: 在统一 AKR 配置空间内求解约束优化（公式 7–8）。起止配置由 [[Inverse Kinematics|IK]] 求解并在关节空间**聚类**保留代表性候选；复杂任务（如狭窄空间开洗碗机）采用**多阶段策略**，采样中间状态 $\phi_{\mathrm{mid}}$ 把轨迹拆成 $[\phi_0\to\phi_{\mathrm{mid}}]$ 与 $[\phi_{\mathrm{mid}}\to\phi_T]$，中间插入无碰撞**重抓取（re-grasp / grasp switching）**。最后做后处理过滤（公式见下）。
4. **Rendering（渲染）**: 用 [[NVIDIA Isaac Sim]] 把已验证轨迹渲染成同步的 egocentric + fixed RGB-D，并投影成仿真世界坐标系下的 3D 点云（4096 点/帧，120 帧/轨迹），支持改光照/相机/模态重放。

### 关键公式与机制

#### 公式1: [[AKR]] 状态定义

$$
\boldsymbol{x}=\bigl[\boldsymbol{q}_{B}^{\mathsf{T}},\,\boldsymbol{q}_{M}^{\mathsf{T}},\,\boldsymbol{q}_{O}^{\mathsf{T}}\bigr]^{\mathsf{T}}\;\in\;\mathcal{X}_{\mathrm{free}}
$$

**含义**: 把底盘、机械臂、物体三者的状态拼成单一 AKR 配置向量，落在无碰撞配置空间 $\mathcal{X}_{\mathrm{free}}$ 内。

**符号说明**:
- $\boldsymbol{q}_{B}\in\mathbb{R}^{3}$: 移动底盘位姿（平面 $x,y,\theta$）
- $\boldsymbol{q}_{M}\in\mathbb{R}^{n}$: 机械臂关节配置（$n$ 为臂的 DoF）
- $\boldsymbol{q}_{O}\in\mathbb{R}^{m}$: 铰接物关节状态（$m$ 为物体 DoF，刚体取 $m=0$）
- 目标轨迹 $\boldsymbol{x}_{1:T}=\langle\boldsymbol{x}_{[1]},\dots,\boldsymbol{x}_{[T]}\rangle\subset\mathcal{X}_{\mathrm{free}}$，长度 $T$

#### 公式2: 链约束（物体-环境附着）

$$
h_{\mathrm{chain}}(\boldsymbol{x}_{[t]})=0,\quad\forall t=1,\dots,T
$$

**含义**: 强制物体附着于环境所施加的运动学约束（如旋转门的铰链、滑椅的平面约束），保证全过程一致。

**符号说明**:
- $h_{\mathrm{chain}}(\cdot)$: 编码物体环境附着的链约束函数；逐时间步施加。

#### 公式3: 任务完成约束（终态目标）

$$
\|f_{\mathrm{task}}(\boldsymbol{x}_{[T]})-\boldsymbol{g}_{\mathrm{goal}}\|_{2}^{2}\leq\xi_{\mathrm{goal}}
$$

**含义**: 把终态经任务映射 $f_{\mathrm{task}}$ 后约束在目标 $\boldsymbol{g}_{\mathrm{goal}}$ 的容差 $\xi_{\mathrm{goal}}$ 内，确保任务达成。

**符号说明**:
- $f_{\mathrm{task}}:\mathcal{X}\to\mathcal{G}$: 把配置映射到任务目标空间
- $\boldsymbol{g}_{\mathrm{goal}}$: 目标（刚体的 $SE(3)$ 位姿或铰接物的开门角度）；$\xi_{\mathrm{goal}}$: 容差

#### 公式4-6: 物理限制（位置 / 速度 / 加速度）

$$
\boldsymbol{x}_{\min}\leq\boldsymbol{x}_{[t]}\leq\boldsymbol{x}_{\max},\quad\forall t=1,\dots,T
$$

$$
\|\Delta\boldsymbol{x}_{[t]}\|_{\infty}\leq\Delta\boldsymbol{x}_{\max},\quad\forall t=1,\dots,T-1
$$

$$
\|\Delta\dot{\boldsymbol{x}}_{[t]}\|_{\infty}\leq\Delta\dot{\boldsymbol{x}}_{\max},\quad\forall t=2,\dots,T-1
$$

**含义**: 分别对关节位置、速度（一阶差分）、加速度（二阶差分）施加硬上限，保证物理可行；碰撞回避由底层规划器的自碰撞/环境碰撞检查隐式处理。

**符号说明**:
- $\boldsymbol{x}_{\min},\boldsymbol{x}_{\max}$: 配置上下限；$\Delta\boldsymbol{x}_{[t]}$: 相邻步差分（速度代理）；$\Delta\dot{\boldsymbol{x}}_{[t]}$: 速度差分（加速度代理）
- $\|\cdot\|_{\infty}$: 无穷范数（逐分量上限）

#### 公式7: 规划目标（行程 + 平滑度代价）

$$
\mathcal{J}(\boldsymbol{x}_{1:T})=\sum_{t=1}^{T-1}\big\|\boldsymbol{w}_{v}\,\Delta{\boldsymbol{x}}_{[t]}\big\|_{2}^{2}+\sum_{t=2}^{T-1}\big\|\boldsymbol{w}_{a}\,\Delta\dot{\boldsymbol{x}}_{[t]}\big\|_{2}^{2}
$$

**含义**: 同时最小化**总行程距离**（一阶项）与**轨迹非平滑度**（二阶项），由对角权重矩阵调制协调策略（如交互时优先底盘稳定）。

**符号说明**:
- $\boldsymbol{w}_{v},\boldsymbol{w}_{a}$（对应对角权重 $\mathbf{W}_{v},\mathbf{W}_{a}$）: 速度/加速度项的关节权重，用于调制 base-arm 协调偏好
- 求和范围 $t=1\dots T-1$（速度项）、$t=2\dots T-1$（加速度项）

#### 公式8: 最优轨迹

$$
\boldsymbol{x}_{1:T}^{\star}=\arg\min_{\boldsymbol{x}_{1:T}}\mathcal{J}(\boldsymbol{x}_{1:T})
$$

**含义**: 在公式 2–6 约束下最小化代价 $\mathcal{J}$，得到物理有效的最优全身轨迹。

#### 公式 A1-A2: AKR 反转/装配中的前向运动学（附录）

$$
T^{\text{base}}_{\text{tip}}=\text{FK}_{\mathcal{K}_{\text{scaled}}}(\boldsymbol{q}_{\text{init}},\ell_{\text{tip}})
$$

$$
T^{\text{tcp}}_{\text{tip}}=\left(T^{\text{base}}_{\text{tcp}}\right)^{-1}\cdot T^{\text{base}}_{\text{tip}}
$$

**含义**: 链反转/装配时，先用缩放后运动学 $\mathcal{K}_{\text{scaled}}$ 算出抓取尖端 $\ell_{\text{tip}}$ 相对物体 base 的位姿（A1），再换算到机器人 TCP 帧（A2），从而把物体作为末端的运动学延伸正确接入 AKR。

**符号说明**:
- $\text{FK}_{\mathcal{K}_{\text{scaled}}}$: 缩放后链的前向运动学；$\ell_{\text{tip}}$: 抓取点尖端链接；$\boldsymbol{q}_{\text{init}}$: 初始关节状态
- $T^{a}_{b}$: 帧 $b$ 相对帧 $a$ 的齐次变换；TCP = tool center point

#### 后处理偏差判据（轨迹过滤）

对静止铰接物，逐 waypoint 评估**平移偏差** $d$ 与**旋转偏差** $\theta$（物体-世界附着）：$d=\|p(\cdot)\|$、$\theta=\angle\,r(\cdot)$，其中 $p(\cdot)$、$r(\cdot)$ 为 AKR 前向运动学的位置/旋转分量。对平面约束（如椅子）额外限制**垂直位移** $d_z=p_z(\cdot)$ 与**朝向偏差** $\theta_{\mathrm{planar}}=\psi(\cdot)$（roll/pitch）。超阈值轨迹直接丢弃，保证数据集只含稳定、物理可行的全身运动。

---

## 关键图表

<!-- 图片均使用 arXiv HTML 在线链接（IMAGE_BASE = https://arxiv.org/html/2604.12565v1/），未本地化 -->

### Figure 1: Overview of AutoMoMa / 框架总览

![Figure 1](https://arxiv.org/html/2604.12565v1/x1.png)

**说明**: AutoMoMa 总览。协调移动操作需要大规模、物理有效的轨迹数据——这一瓶颈是遥操作与传统规划在规模上无法逾越的。给定多样本体、交互场景与任务目标（左），框架统一 [[AKR]] 建模（把 base/arm/object 合并成一条链）与 GPU 加速轨迹优化，高效合成 50 万+ 条轨迹，在解、场景、本体、物体上呈现广泛多样性。点明全文“规划吞吐 = 学习瓶颈”的主线。

### Figure 2: AKR Construction / 增广运动学链构造

![Figure 2](https://arxiv.org/html/2604.12565v1/x2.png)

**说明**: AKR 构造示例（以开门为例）。把原本独立的运动学树融合成单一串行链，使 base/arm/object 可联合优化。移动底盘的平面运动用**虚拟底座**（蓝）建模；**虚拟关节**（黑）把机械臂（橙）耦合到目标物体（绿）。对铰接物，运动学树被**反转**以把根重设到抓取点，形成根植于虚拟世界帧的连续链（黄色高亮）。这是全文方法的几何核心。

### Figure 3: Data Generation Pipeline / 四阶段数据生成流水线

![Figure 3](https://arxiv.org/html/2604.12565v1/x3.png)

**说明**: 从任务规约三元组 $(\mathcal{S},\mathcal{O},\mathcal{R})$（左）出发，依次经过 (i) 任务规约（定义环境/机器人/物体上下文）、(ii) 问题实例化（ESDF 构建 + AKR 装配 + 球体碰撞近似）、(iii) 轨迹生成（在任务约束下求解最优 AKR 状态）、(iv) Isaac Sim 渲染。完整呈现工程化流水线的数据流。

### Figure 4: Pipeline Performance across Scenes / 六场景下的吞吐与运动量

![Fig4a Scene layouts](https://arxiv.org/html/2604.12565v1/x4.png)
![Fig4b Throughput](https://arxiv.org/html/2604.12565v1/x5.png)
![Fig4c Base translation](https://arxiv.org/html/2604.12565v1/x6.png)
![Fig4d Arm rotation](https://arxiv.org/html/2604.12565v1/x7.png)

**说明**: 跨六个空间受限程度递增的代表性家庭场景的生成性能。(a) 场景布局 #1–#6（受限度递增）；(b) **生成吞吐**随杂乱度上升而下降（碰撞检测开销增大、可行 IK 减少）；(c) 底盘**平均平移努力**；(d) 机械臂**平均旋转努力**——后两者反映规划器在受限环境中合成代偿性全身运动的能力。

### Figure 5: Base Position Distribution / 轨迹底盘位置分布

![Figure 5](https://arxiv.org/html/2604.12565v1/x8.png)

**说明**: 轨迹底盘位置分布。蓝球=起始底盘位置，橙球=目标底盘位置，展示 **IK 聚类策略**带来的广阔空间覆盖。佐证“配置多样性”这一卖点。

### Figure 6: Data Scaling Experiments / 数据 scaling 主实验

![Fig6a Fixed vs Mobile](https://arxiv.org/html/2604.12565v1/x9.png)
![Fig6b Scene scaling](https://arxiv.org/html/2604.12565v1/x10.png)
![Fig6c Density scaling](https://arxiv.org/html/2604.12565v1/x11.png)

**说明**: 数据 scaling 三连。(a) 单场景下**移动底盘策略**远比固定底座需要更多数据，且存在持续的 seen/unseen 差距（说明在做流形记忆而非场景理解）；(b) 场景多样性 1→30 稳步提升对未见环境的泛化；(c) 固定 30 场景、提高每场景轨迹密度进一步细化执行精度，达到约 75% 的一致泛化。这是“数据是瓶颈”的核心证据。

### Figure 7: Architectural Generalization / 跨架构泛化

![Figure 7](https://arxiv.org/html/2604.12565v1/x12.png)

**说明**: 在与 [[DP3]] 相同的 30 场景设定下评估 [[Diffusion Policy|DP]] 与 [[ACT]]，两者都随轨迹密度上升而稳定提升，证明 AutoMoMa 数据**与多种全身 IL 架构兼容**（DP3 因 3D 模态仍最优）。

### Figure 8: Per-Object Success Rates at 100k / 100k 规模下逐物体成功率

![Figure 8](https://arxiv.org/html/2604.12565v1/x13.png)

**说明**: 100k 轨迹训练的 DP3 在五个代表性 [[SAPIEN]] 物体上的成功率（橙=未见环境、蓝=已见）。多数物体（ID 7221、11622、101773、103634）在已见配置上 >50%；个别物体（如 46197）方差大源于**铰接约束限制机器人工作空间**而非数据稀缺。

### Figure A1: Downstream Scene Split / 下游场景划分（附录）

![Fig A1a Train scenes](https://arxiv.org/html/2604.12565v1/x14.png)
![Fig A1b Test scenes](https://arxiv.org/html/2604.12565v1/x15.png)

**说明**: 从 300 个程序化生成场景中采样出 30 训练 / 10 测试场景，用于下游策略训练（a）与评估（b）。

### Figure A2: Two Scene Sources / 两类场景来源（附录）

![Fig A2a iTHOR+SAPIEN](https://arxiv.org/html/2604.12565v1/x16.png)
![Fig A2b Infinigen](https://arxiv.org/html/2604.12565v1/x17.png)

**说明**: AutoMoMa 的两类场景源。(a) [[AI2-THOR]]（iTHOR）场景中用 [[SAPIEN]] 资产替换静态电器；(b) [[Infinigen]] 程序化生成布局。体现环境多样性的两条互补来源。

### Figure A3-A5: Trajectory Generation Cases / 生成成败案例（附录）

![Fig A3 Collision failure](https://arxiv.org/html/2604.12565v1/x18.png)
![Fig A4 Constraint failure](https://arxiv.org/html/2604.12565v1/x19.png)
![Fig A5 Success](https://arxiv.org/html/2604.12565v1/x20.png)

**说明**: A3—**球体几何近似偶尔失真**导致规划期意外碰撞的失败；A4—规划轨迹违反物体固定基座约束的失败；A5—正确满足碰撞与运动约束的成功轨迹。直接暴露 GPU 球体近似的局限（也是 limitation 的证据）。

### Figure A6: Successful Inference Rollouts / 推理成功轨迹（附录）

![Fig A6-1](https://arxiv.org/html/2604.12565v1/x21.png)
![Fig A6-2](https://arxiv.org/html/2604.12565v1/x22.png)
![Fig A6-3](https://arxiv.org/html/2604.12565v1/x23.png)

**说明**: 大规模 AutoMoMa 数据训练的 DP3 在多样空间布局与铰接配置下产生可行全身轨迹的代表性推理 rollout。

### Figure A7: Inference Failure / 推理失败（附录）

![Figure A7](https://arxiv.org/html/2604.12565v1/x24.png)

**说明**: 推理失败案例：base/arm 位姿预测的微小不一致**随时间累积**，最终把机器人推入不可行配置。揭示开环 IL 的复合误差问题。

### Figure A8: Real-World Validation (UR5-Ridgeback) / 真实世界验证（附录）

![Fig A8a Drawer](https://arxiv.org/html/2604.12565v1/figures/appendix/open-drawer.png)
![Fig A8b Door](https://arxiv.org/html/2604.12565v1/figures/appendix/open-door.png)

**说明**: 在物理 UR5-Ridgeback 平台（双 UR5 臂 + Clearpath Ridgeback 底盘）上验证规划流水线，抽屉开启（a）与橱柜门开启（b）均平滑执行、无碰撞/约束违反，准确复现仿真生成的运动。

### Figure A9: Scaling Analysis at 100k / 100k scaling 分析（附录）

![Figure A9](https://arxiv.org/html/2604.12565v1/x25.png)

**说明**: 在跨多物体多场景的 100k 数据上训练 DP3，相比更小子集提升了**跨物体能力与对未见环境的泛化**。

### Figure A10: Ablation on Start-State Diversity / 起始状态多样性消融（附录）

![Fig A10 top](https://arxiv.org/html/2604.12565v1/x26.png)
![Fig A10 bottom](https://arxiv.org/html/2604.12565v1/x27.png)

**说明**: 起始状态多样性消融。上：在 1,000 轨迹基准上，增加唯一起始状态数（50→1,000）**单调提升**成功率；下：固定数据量（6,400 条）时更多样的起始状态更好，完整 12,800 条数据集成功率最高。说明“起始状态多样性”是有效 scaling 维度。

### Figure A11: Pick Task Inference / 抓取任务推理（附录）

![Fig A11-1](https://arxiv.org/html/2604.12565v1/x28.png)
![Fig A11-2](https://arxiv.org/html/2604.12565v1/x29.png)

**说明**: 用 1,000 条 AutoMoMa 轨迹训练的策略在刚体 pick 任务上的推理示例。(a) 协调 base-arm 运动的成功执行；(b) 策略未能稳定抓取的失败案例。

### Figure A12: Fixed-Grasp vs Grasp-Switching / 固定抓取 vs 重抓取（附录）

![Fig A12a Fixed-grasp](https://arxiv.org/html/2604.12565v1/figures/appendix/without-grasp-switch.png)
![Fig A12b Grasp-switching](https://arxiv.org/html/2604.12565v1/figures/appendix/with-grasp-switch.png)

**说明**: 固定抓取（a）vs **重抓取/grasp switching**（b）。重抓取通过规避链接碰撞实现**更大的物体开启角度**，对应正文的多阶段 $\phi_{\mathrm{mid}}$ 策略。

### Table 1: Comparison with Existing Datasets / 与现有移动操作数据集对比

| Dataset | Robot | # Episodes | Coord. | # Scenes | Action | Method |
|---------|-------|-----------:|:------:|---------:|--------|--------|
| RT-1 Robot Action [4] | Google Robot | 73,499 | Yes | 10 | End-effector pose | VR teleoperation |
| NYU VINN [33] | Hello Stretch | 435 | Yes | 3 | End-effector pose | Kinesthetic teaching |
| BC-Z [18] | Google Robot | 39,350 | Yes | 2–3 | End-effector pose | VR teleoperation |
| ETH Agent Affordances [35] | Franka | 120 | No | 50 | End-effector pose | Scripted policy |
| QUT Dexterous Manip. [5] | Franka | 200 | No | 1 | End-effector pose | VR teleoperation |
| CMU Stretch [1,29] | Hello Stretch | 135 | No | 10 | End-effector pose | Scripted Policy |
| ConqHose [30] | Spot | 139 | Yes | 3 | End-effector vel. | Scripted policy |
| DobbE [36] | Hello Stretch | 5,208 | Yes | 216 | End-effector pose | Tool-based teleoperation |
| Mobile ALOHA [13] | Mobile ALOHA | 276 | Yes | 5 | Joint position | Leader-follower teleoperation |
| TidyBot [44] | TidyBot | 24 | No | 104 | Other | Scripted primitives |
| **Ours (AutoMoMa)** | **Multi-Robot** | **500,000** | **Yes** | **330** | **Joint position** | **Automatic motion planning** |

**说明**: AutoMoMa 在 **episodes（50 万，远超次高 RT-1 的 7.3 万）**、**场景数（330）**、**协调性（whole-body Coord.=Yes）**、**关节空间动作（Joint position）** 上同时领先。“Coord.”指是否具备全身 base-arm 协调。先前方法要么遥操作高保真但小规模、要么脚本策略缺协调，AutoMoMa 用 GPU 自动规划同时拿下规模、多样性、关节空间高保真。

### Table A1: AutoMoMa Data Generation Hyperparameters / 数据生成超参（附录）

| Hyperparameter | Value |
|---|---|
| **Environment and Collision** | |
| Object offset base (quaternion) | [0,0,0,1,0,0,0] |
| Expanded dimension | 0.3 |
| Disable collision | False |
| Collision type | Voxel / Mesh |
| IK / Trajectory mesh pitch | 0.05 / 0.05 |
| Voxel dimensions / size | [5.0,5.0,5.0] / 0.02 |
| **IK and Trajectory Generation** | |
| Maximum retries | 10 |
| k-means clusters | 500 |
| AP clusters upper / lower / fallback | 80 / 10 / 30 |
| Position / Rotation norm tolerance | 0.01 / 0.01 |
| Grasp poses per object | 20 |
| Open angles | 4–10 |
| **Cameras and Sensing** | |
| Ego top-down / wrist / fixed camera | True / True / True |
| Camera resolution | [320, 240] |
| Camera frequency | 30 Hz |
| Focal length (ego top-down / wrist / fixed) | 50 / 15 / 50 |
| Downsample point cloud / count | True / 4096 |

**说明**: 数据生成关键配置。每物体 20 个抓取位姿、开启角 4–10、IK 聚类（k-means 500 / AP 上下界 80/10）保证配置多样性；点云下采样到 4096 点。

### Table A2: DP3 Hyperparameters / DP3 超参（附录）

| Hyperparameter | Value |
|---|---|
| **Observation/Action**: points / feat dim / proprio dim | 4096 / 3 / 11 |
| History len / pred horizon / action steps / action dim | 3 / 8 / 6 / 11 |
| **Encoder**: PointNet in/out, LayerNorm, crop | 3 / 64, True, [80,80] |
| **Diffusion**: cond type / U-Net down dims / kernel | FiLM / [512,1024,2048] / 5 |
| Noise scheduler / train steps / beta sched | DDIMScheduler / 100 / squaredcos_cap_v2 |
| Prediction type / denoise steps | sample / 10 |
| **Optim**: optimizer / lr / weight decay | AdamW / $1\times10^{-4}$ / $1\times10^{-6}$ |
| Batch size / epochs / lr sched / warmup | 512 / 300 / Cosine decay / 500 |
| Use EMA (inv_gamma/power/min/max) | True (1.0/0.75/0.0/0.9999) |

**说明**: [[DP3]] 是主基准（3D 点云 + 扩散），输出 11 维动作（含 base pose + 关节 + 夹爪），DDIM 推理仅 10 步。

### Table A3: DP Hyperparameters / 图像扩散策略超参（附录）

| Hyperparameter | Value |
|---|---|
| Camera views / proprio dim | 3 / 11 |
| History / horizon / action steps / dim | 3 / 8 / 6 / 11 |
| RGB backbone / pretrained / share across views | ResNet-18 / None / False |
| U-Net down dims / kernel / groups | [256,512,1024] / 5 / 8 |
| Noise scheduler / train steps / beta | DDPMScheduler / 100 / 0.0001–0.02 |
| Prediction type / denoise steps | epsilon / 100 |
| Optimizer / lr / weight decay | AdamW / $1\times10^{-4}$ / $1\times10^{-6}$ |
| Batch size / epochs / mixed precision | 128 / 300 / fp16 |

**说明**: [[Diffusion Policy|DP]] 为 RGB 图像版基线，ResNet-18 编码三视角，DDPM 100 步推理（比 DP3 的 DDIM 10 步慢得多）。

### Table A4: ACT Hyperparameters / ACT 超参（附录）

| Hyperparameter | Value |
|---|---|
| Camera views / action dim / chunk size | 3 / 11 / 6 |
| Temporal aggregation / mobile base mode | False / relative |
| Image backbone / pos embed / hidden dim | ResNet-18 / sine / 512 |
| Feedforward dim / heads / enc / dec layers | 3200 / 8 / 4 / 7 |
| Dropout | 0.1 |
| Optimizer / lr / backbone lr / weight decay | AdamW / $1\times10^{-5}$ / $1\times10^{-5}$ / $1\times10^{-4}$ |
| KL weight / epochs / episodes per task / episode len | 1.0 / 2000 / 100 / 1000 |

**说明**: [[ACT]] 为 Transformer + CVAE 动作分块基线，action chunk = 6，KL 权重 1.0，训练 2000 epoch。

---

## 实验

### 数据集 / 资产

| 资产 | 来源 | 特点 | 用途 |
|------|------|------|------|
| 场景（330） | [[AI2-THOR]] (iTHOR) + [[Infinigen]] | 家庭布局，静态电器替换为铰接体 | 训练/测试（30 训练 + 10 测试 + 程序化扩展） |
| 铰接物 | [[PartNet-Mobility]] / [[SAPIEN]] | 微波炉、抽屉、橱柜门等，含 URDF + 关节限位 | 训练/测试 |
| 机器人本体 | Summit-Franka、TIAGo、R1（真实：UR5-Ridgeback） | 多本体，virtual base + manipulator | 训练/测试/真实验证 |
| 抓取标注 | [[AO-Grasp]] | 每物体 ~20 抓取位姿 | 起始状态采样 |
| 轨迹数据 | AutoMoMa 生成 | 50 万+，30 waypoints/轨迹，120 渲染帧，4096 点/帧 | 训练 |

### 实现细节

- **主基准策略**: [[DP3]]（3D 扩散）；另用 [[Diffusion Policy|DP]]（RGB）与 [[ACT]]（Transformer）验证模型无关性。
- **主智能体**: Summit-Franka 移动操作平台；观测 = egocentric + fixed RGB-D 融合点云（4096 点）+ 本体状态（关节位置 + 底盘位姿）；图像渲染 $320\times240$。
- **训练**: 300 epoch、batch 256、AdamW、$lr=1\times10^{-4}$（DP3/DP 同；ACT 用 $1\times10^{-5}$、2000 epoch）。
- **评测任务**: 微波炉开门——300 步内门到达目标角度即成功，每设置 50 次随机试验取平均。
- **仿真**: [[NVIDIA Isaac Sim]] + GPU 加速 PhysX 引擎。
- **吞吐**: 5,000 episodes / GPU-hour（80× 于 CPU AKR 基线的 ~60 条/小时）。

### 关键实验结论

- **配置空间复杂度（Fig 6a）**: 固定底座 <800 条轨迹即 100% 成功；移动底盘即便 3,200 条也仅约 70%（已见配置）——源于 10-DoF base-arm 耦合使搜索空间指数膨胀，需海量数据才能稳健协调。
- **局部泛化（Fig 6a）**: 单场景内对**未见 IK 起始状态**显著退化，说明高密度单环境训练导致**流形记忆**而非场景理解。
- **环境多样性 scaling（Fig 6b）**: 场景 1→30 稳步提升未见环境成功率；几何多样性（iTHOR + Infinigen）是学到可迁移全身策略的主要驱动力。
- **轨迹密度 scaling（Fig 6c）**: 固定 30 场景增加每场景密度，与“扩场景”增益相当，达到 ~75% 一致泛化。
- **跨架构（Fig 7）**: DP 与 ACT 均随密度提升，DP3 因 3D 模态最优 → 方法收益模型无关。
- **逐物体稳定性（Fig 8）**: 100k 规模下多数物体 >50% 已见成功率，个别低值源于铰接约束而非数据不足。
- **核心论断**: 即便单一铰接物任务，SOTA 也需**数万条示范**逼近 80%——证明 data scarcity 才是 binding constraint。
- **真实验证（Fig A8）**: UR5-Ridgeback 上抽屉/橱柜门开启平滑执行、无碰撞，规划轨迹可直接落地。

---

## 批判性思考

### 优点
1. **吞吐量是真痛点、提速是硬贡献**: 把 AKR 从 60 条/小时拉到 5,000 条/GPU-hour（80×），第一次让“运动学保真规划”与“大规模 IL”在全身移动操作上同时成立，工程价值清晰。
2. **数据规模与多样性兼得**: 50 万条、330 场景、多本体、关节空间动作，在 Table 1 上对现有数据集全面碾压，且伴随真实平台验证（UR5-Ridgeback）。
3. **把“数据稀缺是瓶颈”做成可量化结论**: 用 Fig 6/7/A9/A10 的多维 scaling 实验（固定 vs 移动、场景数、轨迹密度、起始状态多样性、跨架构）系统支撑论断，不是空谈，对后续 robot learning 方向有指导意义。

### 局限性
1. **依赖已知几何与运动学**: 流水线需要已知场景几何与物体 URDF/铰接模型，**不支持动态人机交互、可变形物体**，离真正开放世界尚远。
2. **球体碰撞近似引入误差**: GPU 加速所需的球体几何近似偶尔失真（Fig A3），导致执行期意外碰撞——这是“为速度牺牲精度”的直接代价。
3. **下游绝对成功率仍受限**: 移动底盘策略即便 3,200 条也仅 ~70%（已见）、多场景泛化约 75%；开环 IL 的复合误差（Fig A7）使长程稳定性不足。评测任务也较集中（微波炉/抽屉/橱柜门开启 + 少量 pick），未覆盖更长程、富接触、双臂协同的复杂操作。

### 潜在改进方向
1. 引入**学习式生成**（如扩散/神经规划）替代或加速优化求解，减少对精确几何/运动学先验的依赖（作者亦在 future work 提及）。
2. 用更精细的**凸分解/SDF 碰撞**替代球体近似，或对球体拟合做误差感知补偿，降低 Fig A3 类失败。
3. 把数据-策略闭环做成 **DAgger / 在线纠错**，缓解开环复合误差；并扩展到可变形物、动态场景与双臂协调任务。

### 可复现性评估
- [x] 项目主页（https://automoma.pages.dev/）
- [ ] 代码/数据集明确开源（正文未给出明确 repo 链接，需以主页为准）
- [x] 训练细节完整（附录 Table A1–A4 给出生成与三套策略的完整超参）
- [x] 数据集资产可获取（iTHOR / Infinigen / PartNet-Mobility / SAPIEN / AO-Grasp 均公开；生成数据声明大规模产出）

---

## 速查卡片

> [!summary] AutoMoMa: Scalable Trajectory Generation for Whole-Body Mobile Manipulation
> - **核心**: GPU 加速的 AKR（base+arm+object 合一条链）+ 并行轨迹优化，把全身移动操作数据生成提速 80×。
> - **方法**: 任务规约→问题实例化(ESDF+球体碰撞)→AKR 约束优化(公式 7-8)→Isaac Sim 渲染；多阶段 grasp switching 解决受限空间。
> - **结果**: 5,000 episodes/GPU-hour，50 万+ 轨迹 / 330 场景 / 多本体；scaling 实验证明单任务需数万条示范才达 ~80%，数据稀缺才是瓶颈；UR5-Ridgeback 真实验证通过。
> - **主页**: https://automoma.pages.dev/

---

*笔记创建时间: 2026-06-29*
