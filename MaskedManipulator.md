---
title: "MaskedManipulator: Versatile Whole-Body Control for Loco-Manipulation"
method_name: "MaskedManipulator"
authors: [Chen Tessler, Yifeng Jiang, Erwin Coumans, Zhengyi Luo, Gal Chechik, Xue Bin Peng]
year: 2025
venue: SIGGRAPH Asia
tags: [robot-high-frequency-whole-body-controller, loco-manipulation, physics-based-character-control, motion-tracking, distillation]
zotero_collection: _inbox
image_source: online  # online（默认）/ mixed / local
arxiv_html: https://arxiv.org/html/2505.19086
created: 2026-06-29
---

# MaskedManipulator: Versatile Whole-Body Control for Loco-Manipulation

## 元信息

| 项目 | 内容 |
|------|------|
| 作者 | Chen Tessler, Yifeng Jiang, Erwin Coumans, Zhengyi Luo, Gal Chechik, Xue Bin Peng |
| 机构 | NVIDIA / Stanford / Simon Fraser University 等（动捕角色控制方向） |
| 会议 | SIGGRAPH Asia 2025 |
| 类别 | robot high-frequency whole-body controller / loco-manipulation |
| 日期 | 2025 |
| 项目主页 | — |
| 链接 | [arXiv](https://arxiv.org/abs/2505.19086) / [Code](—) |

---

## 一句话总结

> 先训高保真接触跟踪器 [[MimicManipulator]]，再用 [[DAgger]] 在线蒸馏成可由稀疏时空目标驱动的生成式策略 [[MaskedManipulator]]，在同一框架内统一物体目标与身体目标的全身操作。

---

## 核心贡献

1. **两阶段框架（满信息跟踪 → 目标遮蔽蒸馏）**: 第一阶段 [[MimicManipulator]] 在物理仿真中以满信息参考运动训练高保真、接触感知的跟踪器；第二阶段 [[MaskedManipulator]] 通过在线师生蒸馏，把跟踪器压缩为以稀疏时空目标（手腕/头/骨盆/物体位姿）为条件的生成式控制策略，从而把“任务灵活性 vs 物理精度”这一长期矛盾解耦处理。
2. **三相位接触奖励 + 紧致终止 + 失败优先采样**: 把接触过程拆为接近/接合/释放三阶段分别给奖励，配合很紧的终止判据与基于失败重采样的优先训练，使长序列、精细抓放等接触关键阶段不至于因早期细小误差累积而崩溃。
3. **基于接触一致性的物体重定向**: 将多体型受试者动捕统一迁移到单一规范 [[SMPL-X]] 体型时，用原始接触数据优化物体平移偏移 $p^{*}$，保持手-物相对距离一致，缓解形态差异造成的交互错位。
4. **可变稀疏目标的 Transformer 策略头比较**: 用 [[Transformer]] 处理可变数量的稀疏未来目标 token，并系统比较 Deterministic / [[C-VAE]] / [[Diffusion Policy]] 三种策略头在遥操作与物体目标两类任务下的定量取舍。
5. **GRAB 上的系统评测**: 在 [[GRAB]] 数据集上给出训练/测试划分（1007/141 条），以完整成功率、首次交互成功率、[[MPJPE]]、平均序列长度多套指标量化两阶段表现，并与自实现的 [[InterMimic]] 基线对比。

---

## 问题背景

### 要解决的问题
全身 loco-manipulation 的核心难点在于把高层人类意图（走向物体、拿起、查看、放回桌面）翻译成低层电机指令，同时既要覆盖广泛任务、又要保持手-物接触的精确物理一致性。要解决的问题是：如何用一个统一框架，同时支持“身体目标”（头/骨盆/手腕位姿）与“物体目标”（物体目标位姿）两类多样化的高层指令，并优雅平衡灵活性与精度。

### 现有方法的局限
既有物理角色控制方法往往二选一：一类擅长精细操作但任务面窄、依赖逐帧动作跟踪或遥操作（如直接的接触跟踪 / teleoperation）；另一类擅长多样化身体控制却难以处理精确接触（如 [[AMP]] 等示范驱动的风格化控制）。论文指出这里存在“任务灵活性与物理精度”的内在张力——统一覆盖身体目标与物体目标、并优雅平衡该折衷仍是开放问题。

### 本文的动机
人类动捕（[[GRAB]]）提供了丰富的接触运动学描述，却没有驱动物理角色所需的低层动作；而直接对稀疏目标做强化学习又难以学到精细接触。于是作者把问题拆成两阶段：先在“满信息”设定下学一个能物理重建接触序列的跟踪器，恢复出可行的低层动作；再把它蒸馏成只看稀疏时空目标的生成策略，由此把高保真接触技能转移到灵活的目标条件控制上。

---

## 方法详解

### 模型架构

<!-- 使用 [[概念]] 内联链接所有技术术语 -->

MaskedManipulator 采用 **两阶段“跟踪器 + 蒸馏生成策略”** 架构：

- **问题建模**: 目标条件 MDP $M=(S,G,A,P,R,\gamma)$，策略 $\pi(a_t\mid s_t,g_t)$ 根据当前状态 $s_t$ 与任务目标 $g_t$ 选择动作。
- **角色与仿真**: 人形采用 [[SMPL-X]] 结构，$N_a=153$ 个驱动自由度（51 关节 × 3 DoF），手部用胶囊/盒体简化几何、全局摩擦系数 1.5；动作 $a_t\in\mathbb{R}^{N_a}$ 是 [[PD Controller|PD 控制器]] 的目标关节角；物理引擎为 [[IsaacLab]]，仿真 120 FPS、4 步 decimation，有效控制频率 30 FPS。
- **第一阶段 Backbone（MimicManipulator）**: 在“相对参考运动满信息”设定下训练单一跟踪策略 $\pi_{\text{track}}$，既恢复可行动作、又物理重建目标全身操作序列。
- **第二阶段 Backbone（MaskedManipulator）**: 用 [[Transformer]] 处理可变数量稀疏目标 token，输出动作的策略头有 Deterministic / [[C-VAE]] / [[Diffusion Policy]] 三种变体。
- **输出**: PD 目标关节角 $a_t$，由 $\pi_{\text{versatile}}$ 在仅条件于当前状态 $s_t$ 与稀疏目标 $g^{\text{versatile}}_t$ 下生成。

### 核心模块

#### 模块1: MimicManipulator —— 接触感知高保真跟踪器（§4.1）

**设计动机**: 动捕只提供接触运动学，没有低层动作；需要一个能在物理仿真中“恢复动作 + 重建接触”的跟踪器，作为后续蒸馏的教师。

**具体实现**（§4.1.1 Task design）:
- **观测**: 角色本体感觉 $q_t$、物体信息 $q^{obj}_t$ 与角色-物体关系特征，均表达在以当前根朝向（yaw）对齐的角色局部坐标系下；额外用 [[Basis Point Set|BPS]] 表示编码任意物体几何（预采样点的表面距离）。
- **目标位姿**: 跟踪器观察下一帧（$K=1$）与未来 1 秒（$K=30$）的目标位姿 $g^{track}_t$。
- **奖励**: 乘积形式 $R_{\text{track}}=r^{pose}\cdot r^{contact}\cdot r^{energy}\cdot r^{interaction}$（式 1），各分量为 $\exp(-w\cdot\text{cost})\in(0,1]$；进一步分解为 $r^{humanoid}=r^{ht}\cdot r^{hr}$、$r^{obj}=r^{ot}\cdot r^{or}$（式 2）。
- **三相位接触奖励**（图 2）: 接近相惩罚仿真与参考接触体的距离与表面法向差异；接合相按各身体部位到物体网格的接近度给奖励、抑制过早脱离；脱离相激励原接触部位及时受控离开。
- **紧致终止**: 任一身体部位偏离参考 >25cm、或物体偏离 >10cm、或接触相中非预期失接触连续 >10 帧、或参考已释放后仍保持接触 >0.4s 即终止。配合 **优先训练**（周期性评估并过采样失败序列）抑制误差累积。

**具体实现**（§4.1.2 Data processing）:
- GRAB 采自 10 名性别/体型各异的受试者，统一采用单一规范人形体型（mean [[SMPL-X]]）以简化控制。
- 体型改变会让原本对齐的手-物交互错位（图 3 左），故提出 **物体重定向**：以接触链关节为约束，求物体平移偏移 $p^{*}$ 使接触体到物体网格对应接触点的距离在重定向前后保持一致（附录 B 优化式），重定向后位置为 $\hat p^{obj,retargeted}_t=\hat p^{obj}_t+p^{*}$。
- 过滤掉重定向后仍错位的复杂双手交互、以及依赖缺失模型特征的交互（如把墨镜架到脸上）。最终训练集 1007 条、测试集 141 条（GRAB subject 10 留作测试）。

#### 模块2: MaskedManipulator —— 稀疏目标条件生成策略（§4.2）

**设计动机**: 把 $\pi_{\text{track}}$ 学到的物理接触专长蒸馏进生成式策略 $\pi_{\text{versatile}}$，让学生只看当前状态与一组稀疏未来目标，去预测教师在该情形下会执行的精细动作。

**具体实现**:
- **在线 [[DAgger]] 师生蒸馏**: 学生预测动作 $a^{versatile}_t$ 实际驱动仿真，教师 $\pi_{\text{track}}$ 观察同一 $s_t$ 但拥有未遮蔽的稠密参考轨迹 $g^{track}_t$；蒸馏损失 $\mathcal{L}_{\text{distill}}=-\log\pi_{\text{versatile}}(a^{track}_t\mid s_t,g^{versatile}_t)$（式 3）。
- **[[Transformer]] 目标编码**: 固定信息输入（本体感觉、根状态、手到物体向量、物体与桌面位姿及其 BPS）映射为一个共享 token；每个指定的未来目标（如关节 $j$ 在 $t+k$ 的目标位姿）各自编码为一个 token，从而支持可变数量稀疏约束。
- **三种策略头**:
  - **Deterministic**: 直接预测均值动作，计算高效，但会对数据中多解取平均。
  - **[[C-VAE]]**: 带学习先验的条件变分自编码器，显式建模多模态；先验网络由 $g^{versatile}_t$ 与 $s_t$ 预测潜分布 $\mathcal{N}(\mu_{prior},\sigma_{prior})$。
  - **[[Diffusion Policy]]**: 把纯高斯噪声经 $N$ 步迭代去噪为干净动作 $a^N_t$，去噪以 $s_t$ 与 $g^{versatile}_t$ 为条件。
- 该设计让同一策略既能做密集遥操作、也能从极稀疏目标生成新行为。

### 关键公式与机制

<!-- 公式标题使用 [[概念|名称]] 格式链接到概念库 -->

#### 公式1: [[Reward Shaping|跟踪奖励 R_track]]

$$
R_{\text{track}} = r^{\text{pose}} \cdot r^{\text{contact}} \cdot r^{\text{energy}} \cdot r^{\text{interaction}}
$$

**含义**: MimicManipulator 的总奖励是四个分量的乘积，每个分量都是 $\exp(-w\cdot\text{cost})\in(0,1]$。乘积形式意味着任一维度（姿态、接触、能量、交互）严重失配都会把整体奖励拉向 0，从而强制策略同时满足全身姿态精度与接触正确性，而非在某一项上取巧。

**符号说明**:
- $r^{pose}$: 全身姿态跟踪奖励（含 $r^{ht}/r^{hr}$ 等子项）
- $r^{contact}$: 三相位接触奖励（接近/接合/释放）
- $r^{energy}$: 能量/功率惩罚项，抑制不自然的高能动作
- $r^{interaction}$: 手-物交互一致性奖励

#### 公式2: 人形与物体奖励分解

$$
r^{\text{humanoid}} = r^{ht} \cdot r^{hr}, \qquad r^{\text{obj}} = r^{ot} \cdot r^{or}
$$

**含义**: 把人形与物体的跟踪奖励各自拆为平移项与旋转项的乘积，使位置与朝向都必须对齐。物体奖励单列体现了“物体位姿是一等公民”，这与纯身体跟踪方法不同。

**符号说明**:
- $r^{ht}$: 人形全局平移误差奖励，约 $e^{-100\|\hat p_t-p_t\|}$
- $r^{hr}$: 人形全局旋转误差奖励，约 $e^{-2\langle\hat\theta_t,\theta_t\rangle}$
- $r^{ot}$: 物体平移误差奖励
- $r^{or}$: 物体旋转误差奖励

#### 公式3: [[DAgger|蒸馏损失 L_distill]]

$$
\mathcal{L}_{\text{distill}} = -\log \pi_{\text{versatile}}\big(a^{\text{track}}_t \mid s_t, g^{\text{versatile}}_t\big)
$$

**含义**: 在线 DAgger 师生蒸馏目标——让学生策略在仅观察稀疏目标 $g^{versatile}_t$ 的条件下，最大化教师所执行动作 $a^{track}_t$ 的对数似然。教师持稠密参考、学生持稀疏目标，二者状态 $s_t$ 相同，由此把“满信息接触技能”迁移到“稀疏目标条件”的可控生成上。

**符号说明**:
- $a^{track}_t$: 教师 $\pi_{\text{track}}$ 在 $t$ 时刻执行的动作（监督标签）
- $s_t$: 当前共享状态（师生相同）
- $g^{versatile}_t$: 学生可见的稀疏时空目标（被遮蔽后的稀疏约束）
- $\pi_{versatile}$: 学生策略 MaskedManipulator

#### 公式4: 物体重定向优化（附录 B）

$$
p^{*} = \arg\min_{p} \sum_{j\in\text{ContactLinks}} \Big\| (\hat{c}^{\text{original}}_{j,t} - \hat{c}^{\text{obj}}_{j,t}) - (\hat{c}^{\text{target}}_{j,t} - (\hat{c}^{\text{obj}}_{j,t} + p)) \Big\|^{2}
$$

**含义**: 在把动捕迁移到规范体型时，求物体平移偏移 $p^{*}$，使接触链上各关节到物体网格对应接触点的相对距离在原始体型与目标体型间保持一致。重定向后物体位置为 $\hat p^{obj,retargeted}_t = \hat p^{obj}_t + p^{*}$。这是保证接触监督在体型规范化后仍物理合理的关键预处理。

**符号说明**:
- $p^{*}$: 待求的物体平移偏移
- $\hat{c}^{original}_{j,t}$: 原始体型下接触体 $j$ 的接触坐标
- $\hat{c}^{target}_{j,t}$: 规范体型下接触体 $j$ 的目标接触坐标
- $\hat{c}^{obj}_{j,t}$: 物体网格上对应接触点坐标
- $\text{ContactLinks}$: 参与接触的身体关节集合

---

## 关键图表

<!-- 图片默认使用 arXiv HTML 网络链接 -->

### Figure 1: Teaser / 系统概览

![Figure 1](https://arxiv.org/html/2505.19086v3/figures/manipulator_teaser.png)

**说明**: MaskedManipulator 使物理仿真人形能从稀疏时空目标完成复杂的物体交互。展示了在同一框架下，由稀疏目标（手腕/头/骨盆/物体位姿）驱动出拍照、倒茶、用锤等多样全身操作的能力。

### Figure 3: Object Retargeting / 物体重定向

![Figure 3](https://arxiv.org/html/2505.19086v3/x4.jpg)

**说明**: 形态差异下的物体重定向。把运动在不同体型角色间迁移时会使手-物交互错位（左）；本文利用原始接触数据重定向物体轨迹，保持交互一致性（右）。对应式 4 的优化 $p^{*}$。

### Figure 4: Architectures / 三种策略头推理流程

![Figure 4](https://arxiv.org/html/2505.19086v3/figures/feedforward.jpg)

**说明**: MaskedManipulator 三种策略头的推理流程。(a) Deterministic 直接预测均值动作；(b) [[C-VAE]] 用先验表示每个状态 $s_t$ 与稀疏目标 $g^{versatile}_t$ 下的合理解集合；(c) [[Diffusion Policy]] 以含噪动作 $a^j_t$ 与扩散步 $j$ 为条件迭代去噪。

### Figure 4b: C-VAE 子图

![Figure 4b](https://arxiv.org/html/2505.19086v3/figures/vae.jpg)

**说明**: [[C-VAE]] 架构子图——先验网络从 $s_t$ 与稀疏目标 $g^{versatile}_t$ 预测合理解的潜分布 $\mathcal{N}(\mu_{prior},\sigma_{prior})$，从而显式建模欠定目标下的多模态解。

### Figure 4c: Diffusion 子图

![Figure 4c](https://arxiv.org/html/2505.19086v3/figures/diffusion.jpg)

**说明**: [[Diffusion Policy]] 架构子图——以含噪动作 $a^j_t$ 与扩散步 $j$ 为条件，迭代去噪得到干净动作 $a^N_t$。

### Figure 9: Penetration Issues / 穿透问题

![Figure 9](https://arxiv.org/html/2505.19086v3/figures/manipulationmimic/penetrations.jpg)

**说明**: 当奖励与物理设置不当时出现的穿透问题。间接说明三相位接触奖励与物理参数对结果敏感，是接触建模的关键风险点。

> 注：原论文还含 Figure 2（三相位接触奖励示意）、Figure 5（MimicManipulator 定性结果：拍照/喝汤/用锤/倒茶）、Figure 6–8（MaskedManipulator 定性：手腕推断接触、模拟遥操作、左右手交接、仅骨盆生成“查看大圆环 / 开飞机”等新行为）。这些图未在 note 中提供外链 URL，故以文字描述，不臆造链接。

### Table 1: MimicManipulator 消融与 InterMimic 对比

| 配置 | 训练成功率 | 首次成功率 | 训练 MPJPE(mm) | 测试成功率 | 测试 MPJPE(mm) |
|------|-----------|-----------|----------------|-----------|----------------|
| **MimicManipulator（完整）** | **80.7%** | **93.5%** | **9.8** | **60.2%** | **13.2** |
| − 紧致终止 | ↓ | ↓ | ↑ | ↓ | ↑ |
| − 优先采样 | ↓ | ↓ | ↑ | ↓ | ↑ |
| − 接触引导 | — | — | — | 47.5% | 25.5 |
| [[InterMimic]]（自实现基线） | 11.0% | 8.5% | 42–50 | 8.5% | 42–50 |

**说明**: 累积剥离消融。紧致终止、优先采样、三相位接触引导对长序列接触跟踪均有正贡献，其中接触引导对 MPJPE 影响最大（去掉后测试成功率降到 47.5%、MPJPE 升至 25.5mm）。整体接触感知跟踪器显著优于 InterMimic 基线（后者训练/测试成功率仅 11.0%/8.5%）。

### Table 2: 遥操作任务的架构对比

| 架构 | 训练成功率 | 测试成功率 | 测试 MPJPE(mm) |
|------|-----------|-----------|----------------|
| Deterministic | — | 54.6% | 24.0 |
| [[C-VAE]] | — | 54.6% | 24.4 |
| Offline Diffusion | — | 25.5% | — |
| **[[Diffusion Policy]]（在线）** | **78.6%** | **58.2%** | **19.7** |

**说明**: 条件于头、双腕、物体位姿的遥操作任务。在线 Diffusion 综合最优（测试 58.2% / 19.7mm），略优于 C-VAE 与 Deterministic；Offline Diffusion 明显劣化（25.5%），说明在线交互式蒸馏对扩散策略尤为重要。

### Table 3: 物体目标任务的架构对比

| 架构 | 训练成功率 | 测试成功率 |
|------|-----------|-----------|
| Deterministic | 偏弱 | 偏弱 |
| **[[C-VAE]]** | **64.7%（最高）** | — |
| Offline Diffusion | 最弱 | 最弱 |
| **[[Diffusion Policy]]** | — | **59.6%（最高）** |

**说明**: 仅给出何时何地搬运物体的稀疏目标时，显式建模多模态的 Diffusion / C-VAE 优于确定性头（Diffusion 测试成功率 59.6% 最高，C-VAE 训练成功率 64.7% 最高），进一步支持“多解建模有助于欠定目标下的生成质量”这一结论。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| [[GRAB]]（训练） | 1007 条 | 全身抓取与物体交互动捕 | 训练 |
| [[GRAB]]（测试） | 141 条 | subject 10 留作测试 | 测试 |

### 实现细节

- **仿真器**: [[IsaacLab]]，120 FPS 仿真、4 步 decimation → 30 FPS 控制
- **角色**: 153 DoF [[SMPL-X]] 结构，[[PD Controller|PD 控制]]，手部胶囊/盒体几何，摩擦系数 1.5
- **物体几何观测**: [[Basis Point Set|BPS]]
- **跟踪器训练**: 紧致终止 + 失败优先采样 + 三相位接触奖励（式 1/2）
- **生成策略训练**: 在线 [[DAgger]] 师生蒸馏（式 3），Transformer 目标编码，三种策略头
- **评测指标**: 完整序列成功率、首次交互成功率、[[MPJPE]](mm)、平均序列长度(s)；失败判据为任一关节偏离参考 >50cm 或物体偏离 >25cm
- **硬件**: 论文未在 note 中给出具体 GPU 型号/数量

### 可视化结果

- **MimicManipulator（图 5）**: 能复现拍照、喝汤、用锤、倒茶等多类日常交互。首次成功率（93.5% 训练 / 83.7% 测试）远高于完整成功率（80.7% / 60.2%），说明长序列后段失败仍是主要瓶颈。
- **MaskedManipulator（图 6–8）**: 能仅凭手腕推断何时该接触、在仅给手腕时“猜测”用哪只手、模拟遥操作、按物体目标序列搬运并在目标变化时反应、左右手交接、双手抬大物体；甚至在仅给骨盆位置时生成查看大圆环、“开飞机”等新行为。测试集成功率普遍在 50–60% 量级，离“可靠”仍有距离，Offline Diffusion 的劣势也印证在线交互监督不可或缺。

---

## 批判性思考

### 优点
1. 两阶段“满信息跟踪 → 目标遮蔽蒸馏”把精度与灵活性解耦得很干净：先把接触跟踪做到测试 [[MPJPE]] 13.2mm（显著优于 [[InterMimic]] 的 42–50mm），再用 [[DAgger]] 在线蒸馏到 [[Transformer]] 生成策略。
2. 同一策略既能遥操作、又能从仅骨盆/物体目标生成新行为（图 8），统一覆盖身体目标与物体目标。
3. 对物体位姿单列奖励、以及基于接触一致性的物体重定向，是处理跨体型交互错位的务实工程贡献。

### 局限性
1. 评测完全在 [[IsaacLab]] 仿真与 [[GRAB]] 物体分布内，无真实机器人或 sim-to-real 证据；手部用胶囊/盒体简化几何、固定摩擦 1.5，精细灵巧抓握的真实可行性未被验证。
2. 成功率绝对值偏低：跟踪器测试完整成功率仅 60.2%、生成策略测试普遍 50–60%，且首次成功率远高于完整成功率，暴露长序列后段累积失败；图 9 显示结果对奖励与物理参数敏感。
3. 稀疏目标本质欠定，策略需“猜”意图（如用哪只手），不支持任意细粒度可控；覆盖面受 GRAB 数据与重定向可行性约束（部分双手/缺失特征交互被过滤）。

### 潜在改进方向
1. 引入更真实的可形变/带关节手模型与接触求解，并补充真实硬件或更细粒度物体几何的评测，检验 sim-to-real 迁移。
2. 针对长序列后段失败，增加分段子目标或失败自恢复机制、扩大优先采样的难例覆盖，缩小完整成功率与首次成功率的差距。
3. 为欠定的稀疏目标提供可控意图先验（如指定主用手、接触偏好或风格），并把数据从 GRAB 扩展到更多物体与双手交互以提升重建覆盖。

### 可复现性评估
- [ ] 代码开源（note 中未提供 code 链接）
- [ ] 预训练模型
- [x] 训练细节完整（仿真频率、DoF、奖励结构、终止判据、数据划分均给出）
- [x] 数据集可获取（[[GRAB]] 为公开动捕数据集）

---

## 速查卡片

> [!summary] MaskedManipulator: Versatile Whole-Body Control for Loco-Manipulation
> - **核心**: 高保真接触跟踪器蒸馏成可由稀疏时空目标驱动的生成式全身操作策略，统一身体目标与物体目标。
> - **方法**: 阶段一 MimicManipulator（三相位接触奖励 + 紧致终止 + 优先采样，满信息跟踪 GRAB）；阶段二 MaskedManipulator（在线 [[DAgger]] 蒸馏 + [[Transformer]] 稀疏目标编码 + Deterministic/[[C-VAE]]/[[Diffusion Policy]] 策略头）。
> - **结果**: 跟踪器测试 60.2% 成功率 / 13.2mm MPJPE（远优于 InterMimic 的 8.5% / 42–50mm）；生成策略遥操作 Diffusion 测试 58.2% / 19.7mm、物体目标 Diffusion 测试 59.6% 最佳。
> - **代码**: 未在来源中提供

---

*笔记创建时间: 2026-06-29*
