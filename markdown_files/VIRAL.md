---
title: "VIRAL: Visual Sim-to-Real at Scale for Humanoid Loco-Manipulation"
method_name: "VIRAL"
authors: [Tairan He, Zi Wang, Haoru Xue, Qingwei Ben, Zhengyi Luo, Wenli Xiao, Ye Yuan, Xingye Da, Fernando Castañeda, Shankar Sastry, Changliu Liu, Guanya Shi, Linxi Fan, Yuke Zhu]
year: 2026
venue: CVPR
tags: [humanoid, loco-manipulation, sim-to-real, teacher-student, RL, DAgger, domain-randomization, visuomotor-policy]
zotero_collection: _inbox
image_source: online
arxiv_html: https://arxiv.org/html/2511.15200v2
created: 2026-06-29
---

# VIRAL: Visual Sim-to-Real at Scale for Humanoid Loco-Manipulation

## 元信息

| 项目 | 内容 |
|------|------|
| 作者 | Tairan He, Zi Wang, Haoru Xue, Qingwei Ben, Zhengyi Luo, Wenli Xiao, Ye Yuan, Xingye Da, Fernando Castañeda, Shankar Sastry, Changliu Liu, Guanya Shi, Linxi Fan, Yuke Zhu |
| 机构 | NVIDIA、CMU、UC Berkeley、UT Austin 等（典型 NVIDIA-GEAR/CMU/UT 联合阵容） |
| 会议 | CVPR 2026 |
| 类别 | Humanoid / Loco-Manipulation / Visual Sim-to-Real |
| 日期 | 2025-11（arXiv v2） |
| 项目主页 | https://viral-humanoid.github.io |
| 链接 | [arXiv](https://arxiv.org/abs/2511.15200) / [Project](https://viral-humanoid.github.io) |

---

## 一句话总结

> 用"特权 RL 教师 → RGB 学生蒸馏"的两段式框架，在仿真中端到端学会人形机器人移动操作，靠大规模视觉随机化 + 真实到仿真对齐 + 数十 GPU 算力，零样本部署到 Unitree G1 连续运行 54 个回合，逼近人类专家遥操作水平。

---

## 核心贡献

1. **全栈"配方"而非新算法**: 作者明确声明目标不是又一个新 RL/sim-to-real 算法，而是给出让 **RGB-based 人形移动操作真正落地**的完整技术栈——哪些设计有用、在哪里失败、它们如何相互作用（[[teacher-student]] + [[delta action space]] + [[reference state initialization|RSI]] + [[DAgger]]/BC 混合 + 视觉随机化 + 真实到仿真对齐）。
2. **首次实现纯仿真训练的人形长程移动操作零样本落地**: 部署在 [[Unitree G1]] 上，仅用 onboard RGB + 本体感受，连续完成"走-放-抓-转身"循环 **54 个回合**（59 次试验成功 54 次），无任何真机微调，泛化到托盘/物体位置、起始姿态、桌高、桌布颜色、光照、物体类别等多种变化。
3. **指出"算力规模是关键变量"**: 系统性 scaling 研究表明，**算力不足（1-2 GPU）时教师/学生训练经常彻底失败**，而扩展到 8-16 GPU（教师）/ 至多 64 GPU（学生）才能可靠收敛并冲到 >90% 成功率——把"算力规模"从工程便利提升为长程人形移动操作的"实际必要条件"。

---

## 问题背景

### 要解决的问题
人形机器人落地的核心缺口是缺乏**自主移动操作（autonomous loco-manipulation）**能力：在 onboard 感知下，把**运动（locomotion）**与**操作（manipulation）**紧耦合协调起来，跨长时程、跨多样环境完成有用任务。本文要回答的核心问题是：**视觉 sim-to-real 能否在 onboard 感知下实现有用的人形移动操作？**

### 现有方法的局限
- 现有人形系统大多只做**盲走（blind locomotion）**、**固定桌面操作（无移动）**，或严重依赖**人类遥操作**与**外部（非 onboard）传感器**，很少在真实世界展示 onboard 自主移动操作。
- "把人形移动操作当成纯数据问题"的真机基础模型路线：移动操作比固定桌面**变化大得多、需要的数据多得多**；当平台是高自由度人形时，单点数据成本因硬件复杂度、安全约束、遥操作栈工程开销而进一步飙升，**规模化采集可能贵到不可行**。
- 仿真到现实在**运动**上已是事实标准并能零样本迁移；但**操作**仍由真实数据模仿学习主导，sim-to-real 成功多局限于**桌面 + 窄任务**；且运动与操作通常被**孤立研究**（运动忽略操作、操作假设固定底座）。

### 本文的动机
仿真能以极低边际成本产生比遥操作多几个数量级的数据。作者主张：用 **teacher-student 特权学习**把"先在全状态下学会强行为"与"再蒸馏成只看 RGB+本体的可部署策略"解耦；用**大规模视觉随机化**覆盖外观/光照/相机分布，用**真实到仿真对齐**（灵巧手 SysID、相机外参标定）收窄硬件侧 gap；并发现**只有把仿真训练扩展到数十 GPU**，才能让长程移动操作的教师与学生训练稳定收敛。

---

## 方法详解

### 框架总览

VIRAL 采用 **特权 teacher-student 学习** 流水线（见 Figure 2），两个阶段都建立在一个**预训练全身控制器（WBC）[[HOMIE]]** 之上：

- **Phase 1 教师（特权 RL）**: 在仿真中训练，**全状态特权输入**（本体感受 + 外感受任务信息），输出 WBC 指令。运行在 **2 个 8-GPU L40S 节点（共 16 GPU）**、无渲染开销。关键设计：分阶段奖励、从演示初始化（RSI）、[[delta action space|delta 动作空间]]、把 WBC 指令当作动作空间 API。
- **Phase 2 学生（视觉蒸馏）**: 只接收真机可得的观测（**RGB 图像 + 本体感受**），用 [[Isaac Lab]] 的 **tiled rendering** 在 **8 个 8-GPU L40S 节点（共 64 GPU）**上做大规模蒸馏，目标用 **online [[DAgger]] + 行为克隆（BC）混合**模仿教师动作。
- **Sim-to-Real 迁移**: 训练期随机化资产/材质/穹顶光照/图像效果/相机外参/传感器延迟；真实到仿真侧做灵巧手 **SysID** 与相机外参标定。最终**零微调**部署到 [[Unitree G1]]。

整体的层级关系是：高层 VIRAL 策略（教师→学生）输出**速度/姿态/手指增量指令** → 底层 HOMIE WBC 把指令变成稳定的全身电机动作。

### 核心模块

#### 教师要素 1: 奖励设计（Reward Design）

**设计动机**: 把长程移动操作任务**切成阶段序列**（走→预放置位姿→放置→抓取并抬升→转身），每阶段给定向奖励，降低稀疏长程任务的探索难度。

**具体实现**: 四类关键奖励——
- **走向物体** $r_{\text{walk}}$: 让机器人接近抓取目标到约 0.45m；
- **靠近托盘时放置** $r_{\text{place}}$: 仅当待放物离托盘 <0.3m 才生效，以接触力衡量；
- **抓取** $r_{\text{grasp-z}}$（抬升高度，封顶 0.15m）与 $r_{\text{grasp-goal}}$（接近目标位姿）；
- **转身** $r_{\text{turn}}$: 惩罚基座偏航与期望朝向的偏差。
完整奖励见 Table 3（横跨 5 个阶段 0–4 的分阶段 shaping）。

#### 教师要素 2: Delta 动作空间

**设计动机**: 与腿足运动 RL 常用的**绝对关节目标**不同，VIRAL 输出**增量（delta）**累加到 WBC 指令上。经验上 delta 表示**显著加速并稳定 RL 训练**（消融见 Figure 9，绝对动作变体无法达到高成功率）。

#### 教师要素 3: 把 WBC 指令当作 API

**设计动机**: 不从零学底层电机技能，而是让教师产出**高层 WBC 指令**，既减轻奖励工程负担，又把动作空间约束到 WBC 已保证的**安全可靠运动区域**，提升可部署性。

**具体实现**: 底层用 [[HOMIE]] WBC（稳定下肢运动 + 多样上肢姿态），其指令空间含速度/高度跟踪 + 上肢关节指令；本文**扩展加入手指动作**得到 VIRAL 完整动作空间。作者强调框架**不过拟合特定 WBC**，可换用 TWIST2/SONIC 等其它人形 WBC。

#### 教师要素 4: 参考状态初始化（Reference State Initialization, RSI）

**设计动机**: 高自由度人形从零学"走-放-抓-转"长程技能，纯靠奖励工程往往次优或迁移差。

**具体实现**: 采集 **200 条遥操作仿真演示**作为状态初始化缓冲；每次 episode reset 时采样一个演示快照，据此初始化机器人/物体/桌子，让策略在尚无能力自行到达前就**提前暴露于多样的高回报状态**。这种参考偏置探索大幅降低对脆弱奖励调参的依赖、改善 sim-to-real，是训练成功的**关键**（消融 Figure 9：无 RSI 成功率 <10%，有 RSI 接近 95%）。

#### 学生要素 1: DAgger & BC 混合

**设计动机**: BC 用教师 rollout 提供干净近优演示、**快速注入强先验**；DAgger 用学生 rollout 暴露教师理想分布外的状态，**提升纠错鲁棒性、抑制部署期复合误差**。两者共享同一 MSE 目标，仅观测来源不同。

**具体实现**: 在教师/学生诱导观测分布的**混合**上计算 MSE；混合系数 $\alpha$ 是跟随教师策略采集数据的环境比例（$\alpha=0$ 纯 DAgger，$\alpha=1$ 纯 BC）。消融（Figure 11）显示 $\alpha=0.5$ 最佳，故默认取 0.5。

#### 学生要素 2: 网络主干（Network Backbone）

**设计动机**: 用 SOTA 图像编码器 [[DINOv3]] 提取高质量 RGB 特征，与本体感受融合送入策略头，让策略既能利用丰富视觉线索又保有底层感知。

**具体实现**: RGB 图像尺寸 $108\times192$ 经视觉编码得 **128 维视觉特征**，与状态观测拼接送入策略头。策略头评估了单步 MLP 与**带时序上下文的 history-aware 架构**（消融 Figure 10 主干、Figure 12 历史架构）。

#### 学生要素 3: 分布式仿真学习系统

**设计动机**: 大规模视觉仿真比无渲染物理仿真慢至少一个数量级，必须解决吞吐扩展。

**具体实现**: 定制化 [[TRL]] + [[Accelerate]] 支持跨多 GPU/多节点近线性扩展，保留单 GPU 训练的简洁性。Scaling 消融（Figure 14/15）证明扩 GPU 对教师与学生训练都是关键。

#### Sim-to-Real 要素 1: 灵巧手 SysID

G1 的 3 指灵巧手用**高传动比**电机，sim-to-real 失配大。做法：定义真机"抓-放"原语并在仿真复现同一动作序列，对**手指 armature、刚度、阻尼**做系统辨识以对齐仿真与真实关节轨迹（Figure 5，SysID 后对齐显著改善）。

#### Sim-to-Real 要素 2: FOV 对齐与随机化

相机内参（焦距/对焦距离/传感器孔径）按厂商规格匹配；但 G1 各单位**外参因机械公差而异、且会随时间漂移**，故做轻量**真实到仿真外参标定**（视觉匹配渲染图与真实图，Figure 6），并在训练中对外参做随机化以对硬件视角差异保持鲁棒。

#### Sim-to-Real 要素 3: 视觉与仿真随机化

训练期施加广泛随机化（Figure 3）：图像质量（亮度/对比/色相/饱和/高斯噪声/模糊）、相机外参与延迟、穹顶光照、地板/桌/物体/机器人材质与颜色。消融（Figure 13）显示这些随机化**互补且共同构成稳健 sim-to-real 的关键管线**。

### 关键公式与机制

#### 公式1: [[DAgger]] & BC 混合蒸馏目标

$$
\mathbb{E}_{o_t\sim\rho^{o}}\!\left[\left\|\pi_{\text{teacher}}(o^{\text{teacher}}_t)-\pi_{\text{student}}(o^{\text{student}}_t)\right\|_2^2\right]
$$

其中观测分布定义为教师/学生 rollout 分布的凸组合：

$$
\rho^{o}\triangleq\alpha\,\rho^{o}_{\pi_{\text{teacher}}}+(1-\alpha)\,\rho^{o}_{\pi_{\text{student}}}
$$

**含义**: 学生在"教师诱导分布"与"学生诱导分布"混合的观测上，用 MSE 回归教师动作。$\alpha$ 控制 BC（教师 rollout）与 DAgger（学生 rollout）的配比。

**符号说明**:
- $\pi_{\text{teacher}}$: 特权教师策略（看全状态 $o^{\text{teacher}}_t$）
- $\pi_{\text{student}}$: RGB 学生策略（看 $o^{\text{student}}_t$ = RGB + 本体感受）
- $\rho^{o}_{\pi_{\text{teacher}}},\rho^{o}_{\pi_{\text{student}}}$: 两策略各自 rollout 诱导的观测分布
- $\alpha\in[0,1]$: 跟随教师采集的环境比例（$\alpha=1$ 纯 BC，$\alpha=0$ 纯 DAgger，默认 0.5）

#### 公式2: 教师策略与动作空间

$$
a_t=\big(\Delta\mathbf{v}_t,\ \Delta\boldsymbol{\omega}^{\text{yaw}}_t,\ \Delta\mathbf{q}^{\text{arm}}_t,\ \Delta\mathbf{q}^{\text{finger}}_t\big)\sim\pi_{\text{teacher}}(a_t\mid o^{\text{priv}}_t)
$$

**含义**: 教师是目标条件 RL 策略，在特权观测下输出送往 WBC 的**高层指令增量**（delta 动作空间）。

**符号说明**:
- $\Delta\mathbf{v}_t,\Delta\boldsymbol{\omega}^{\text{yaw}}_t$: 线速度(x,y)与偏航角速度的增量指令
- $\Delta\mathbf{q}^{\text{arm}}_t,\Delta\mathbf{q}^{\text{finger}}_t$: 手臂与手指电机的关节目标增量
- $o^{\text{priv}}_t=[o^{\text{prop-priv}}_t,o^{\text{exte-priv}}_t]$: 特权本体感受 + 外感受
- $o^{\text{prop-priv}}_t=[\mathbf{v}_t,\boldsymbol{\omega}_t,\mathbf{g}_t,\boldsymbol{a}_{t-1},\boldsymbol{q}_t,\dot{\boldsymbol{q}}_t,\mathbf{f}^{\text{finger}}_t]$（基座线/角速度、投影重力、上一动作、关节位/速、指尖力）
- $o^{\text{exte-priv}}_t=[e_t,\boldsymbol{T}_t,\boldsymbol{O}_t]$（当前阶段、放置/抬升目标、物体与桌相对机器人的变换）

#### 公式3: 分阶段加权总奖励

$$
r_t=\sum_{i=0}^{4} w_i\,\mathbbm{1}[s_t=i]\,r^{(i)}_t,\qquad w_i>0
$$

**含义**: 把一个"放-取"循环拆成 5 个阶段（走→预放置位姿→放置→抓取抬升→转身），总奖励是当前阶段对应奖励项的加权和；阶段转移由各阶段的推进/完成判据控制。

**符号说明**:
- $s_t$: 当前阶段；$\mathbbm{1}[s_t=i]$: 阶段指示
- $w_i>0$: 第 $i$ 阶段权重；$r^{(i)}_t$: 第 $i$ 阶段 shaping 奖励（详见 Table 3）

#### 关键奖励项（正文给出的四类核心）

$$
r_{\text{walk}}=\exp\!\big(-4\,(\|p_{\text{robot}}-p_{\text{GraspObj}}\|-0.45)^2\big)
$$

$$
r_{\text{place}}=-\|\mathbf{f}_{\text{PlaceObj}}\|\cdot\mathbbm{1}\big(\|p_{\text{PlaceObj}}-p_{\text{tray}}\|<0.3\big)
$$

$$
r_{\text{grasp-z}}=\min\!\big(h_{\text{GraspObj}}-h_{\text{table}},\,0.15\big),\qquad r_{\text{grasp-goal}}=\exp\!\big(-10\,\|p_{\text{GraspObj}}-p_{\text{goal}}\|^2\big)
$$

$$
r_{\text{turn}}=-\,|\,\mathrm{y}_{\text{robot}}-\mathrm{y}_{\text{desired}}\,|
$$

**含义**: 分别对应走向物体、靠近托盘放置、抓取抬升与目标接近、转身朝向；$\mathbf{f}_{\text{PlaceObj}}$ 为手指与待放物间的接触力，$\mathrm{y}$ 为基座偏航朝向角。

---

## 关键图表

<!-- 图片均使用 arXiv HTML 在线链接，未本地化 -->

### Figure 1: Teaser / 真实部署与多样仿真场景

![Figure 1](https://arxiv.org/html/2511.15200v2/fig/VisitReal-Fig1__8_-crop.jpg)

**说明**: 中心为 Unitree G1 用 RGB-based sim-to-real 策略在两桌之间穿梭放置/抓取物体、连续运行 **54 个回合**；四周展示训练所用的多样仿真场景。直观体现"纯仿真训练 → 真机长程连续运行"这一核心成果。

### Figure 2: VIRAL Teacher-Student Pipeline / 教师-学生总流程

![Figure 2](https://arxiv.org/html/2511.15200v2/x1.png)

**说明**: VIRAL 两段式框架。Phase 1 特权 RL 教师 $\pi_{\text{teacher}}$ 接收全状态本体/外感受、输出 WBC 指令；Phase 2 视觉学生 $\pi_{\text{student}}$ 只看 RGB + 真机本体感受，通过 DAgger+BC 蒸馏模仿教师。整张图是理解全文层级（高层策略 → HOMIE WBC → 电机）的主线。

### Figure 3: Visual Randomization / 视觉随机化

![Figure 3](https://arxiv.org/html/2511.15200v2/fig/Visitor-Visual-Randomization__1_-crop.jpg)

**说明**: 在图像、光照、材质、相机外参上的随机化示例，用以增强 sim-to-real 鲁棒性。对应 Table 7 的随机化参数与 Figure 13 的消融。

### Figure 4: Reference State Initialization / 教师 RL 的参考状态初始化

![Figure 4](https://arxiv.org/html/2511.15200v2/x2.png)

**说明**: 用 200 条遥操作演示快照初始化场景（机器人/物体/桌），让策略提前暴露于高回报中间状态，是教师能学会长程任务的关键（见 Figure 9 消融）。

### Figure 5: Dexterous Hand SysID / 灵巧手系统辨识

![Figure 5](https://arxiv.org/html/2511.15200v2/x3.png)

**说明**: 高传动比 3 指手的真-仿对齐：上为真/仿叠加，下为 SysID 前后关节位轨迹，SysID 后对齐显著改善——这是手指动作能 sim-to-real 的前提。

### Figure 6: Camera Extrinsics Real-to-Sim Alignment / 相机外参真实到仿真对齐

![Figure 6](https://arxiv.org/html/2511.15200v2/x4.png)

**说明**: 真实视图 vs 对齐前/后仿真视图。因 G1 外参随单位/时间漂移，需轻量真实到仿真外参标定 + 训练期外参随机化。

### Figure 7: Real-World Performance Comparison / 真机性能对比

![Figure 7](https://arxiv.org/html/2511.15200v2/x5.png)

**说明**: VIRAL 对比专家(>1000h)与非专家(~1h)遥操作（同一 HOMIE 底层，近乎 apple-to-apple）。专家 100% 成功、周期 21.4s；VIRAL 周期 20.2s（比专家更快）、成功率逼近专家；非专家仅 73% 且明显更慢。

### Figure 8: Real-World Generalization / 真机泛化

![Figure 8](https://arxiv.org/html/2511.15200v2/fig/Visitor-Exp-Generalization__2_-crop.jpg)

**说明**: 在托盘/物体位置、机器人起始姿态、桌高与桌型、桌布颜色、光照、物体类别等变化下，VIRAL 无需额外调参均能完成任务，归因于域随机化 + RL 鲁棒性。

### Figure 9: Teacher Training Ablations (RSI + Delta Action) / 教师消融

![Figure 9](https://arxiv.org/html/2511.15200v2/x6.png)

**说明**: 训练奖励(左)与成功率(右)。完整方法(RSI + delta action)接近 95% 成功；去掉演示重置(RSI)成功率 <10%，去掉 delta 动作空间也无法达到高成功——两者均为成功必需。

### Figure 10: Vision Backbone Ablation / 视觉主干消融

![Figure 10](https://arxiv.org/html/2511.15200v2/x7.png)

**说明**: SOTA 主干 [[DINOv3]] 提供更强视觉表征与更大容量，收敛更快、任务成功率更高。

### Figure 11: DAgger/BC Ratio Ablation / DAgger-BC 配比消融

![Figure 11](https://arxiv.org/html/2511.15200v2/x8.png)

**说明**: $\alpha=1$（纯 BC）loss 降得快但策略脆弱、无法纠错、在 Isaac→MuJoCo 与真机评测差；引入学生 rollout（$\alpha=0.5$）略慢但部署成功率大幅提升，故默认 $\alpha=0.5$。

### Figure 12: History Architecture Ablation / 历史架构消融

![Figure 12](https://arxiv.org/html/2511.15200v2/x9.png)

**说明**: 对比单步 baseline、前馈历史模型、LSTM 及不同历史长度。history-aware 一致优于单步，时序窗口越长在资源允许时收益越大。

### Figure 13: Visual Randomization Ablation / 视觉随机化消融

![Figure 13](https://arxiv.org/html/2511.15200v2/x10.png)

**说明**: 聚焦三大主导分量——材质(M)、穹顶光(D)、相机外参(E)。全部关闭随机化成功率掉到 **0.649（-35.1%）**；去掉任一单项也退化，说明随机化互补且共同构成稳健迁移管线（成功率以"全随机化=1.0"归一化、200 episode 平均）。

### Figure 14: Scaling Compute for Teacher / 教师算力扩展

![Figure 14](https://arxiv.org/html/2511.15200v2/x11.png)

**说明**: 1→16 GPU 的奖励(左)与成功率(右)。GPU 越多收敛越快（早期甚至超线性）；1-2 GPU 时教师远低于目标、永远到不了高成功率，而 8-16 GPU 稳定 >90%——算力是长程移动操作的**必要条件**而非锦上添花。

### Figure 15: Scaling Compute for Student / 学生算力扩展

![Figure 15](https://arxiv.org/html/2511.15200v2/x12.png)

**说明**: 1→64 GPU 的蒸馏 loss(左)与成功率(右)。大规模训练收敛更快、loss 更平滑、成功率方差更小、最终成功略高，凸显大规模并行仿真对视觉移动操作蒸馏的实际必要性。

### Figure 16: Object Generalization of Teacher / 教师物体泛化

![Figure 16](https://arxiv.org/html/2511.15200v2/x13.png)

**说明**: 单物体(仅圆柱)训练 vs 多物体(10 种)训练，均在同 10 种物体上测归一化成功率。多物体训练在**每一类**上都显著优于仅圆柱基线。

### Table 1: Observation Dimensions for Teacher / 教师观测维度（附录）

| State term | Dim |
|---|---|
| Base linear velocity | 3 |
| Base angular velocity | 3 |
| Projected gravity | 3 |
| Actions | 31 |
| Stage | 5 |
| Delta actions | 11 |
| DoF position | 43 |
| DoF velocity | 43 |
| Placement position | 2 |
| Table–pelvis transform | 9 |
| Finger-tip forces for hold_object | 12 |
| Hold_object transform | 9 |
| Hold_object–hand transform | 9 |
| Target pre-place position | 3 |
| Finger-tip forces for grasp_object | 12 |
| Grasp_object transform | 9 |
| Grasp_object–hand transform | 9 |
| Target lift position | 3 |
| HOMIE commands | 7 |
| **Single-step total dim** | **226** |

**说明**: 教师享有大量**特权外感受**（物体/桌的相对变换、指尖力、目标位置等），单步观测共 226 维——这些信息真机不可得，故需蒸馏到只看 RGB 的学生。

### Table 2: Observation Dimensions for Student / 学生观测维度（附录）

| State term | Dim |
|---|---|
| Base angular velocity | 3 |
| Projected gravity | 3 |
| Actions | 31 |
| DoF position (w/o fingers) | 29 |
| DoF velocity (w/o fingers) | 29 |
| Delta actions | 11 |
| HOMIE commands | 7 |
| **Single-step total dim** | **113** |

**说明**: 学生只用真机可得的 113 维状态观测，**外加**一张 $108\times192$ RGB 图（经视觉编码为 128 维特征，与状态拼接送策略头）。对比 Table 1 可见特权信息被视觉特征替代。

### Table 3: Reward Components / 奖励项、表达式、权重与生效阶段（附录）

| Term | Expression | Weight | Stage(s) |
|---|---|---|---|
| *Termination / generic penalties* | | | |
| Termination | $\mathbbm{1}_{\{\text{termination}\}}$ | -2000.0 | 0–4 |
| Delta action rate | $\|\Delta a_t\|_2^2$ | -0.01 | 0–4 |
| DoF velocity | $\|\dot{\mathbf{q}}\|_2^2$ | -0.5 | 0–4 |
| DoF acceleration | $\|\ddot{\mathbf{q}}\|_2^2$ | $-3.0\times10^{-6}$ | 0–4 |
| Torque limits | $\|\boldsymbol{\tau}\|_2^2$ | -0.001 | 0–4 |
| Output smoothness | $\|\pi_t-\pi_{t-1}\|_2^2$ | -9.0 | 0–4 |
| Finger primitive limits | $\|\operatorname{clip}(u_{\text{finger}},[l,u])-u_{\text{finger}}\|$ | -20.0 | 0–4 |
| Fast right-arm velocity | $\|\dot{\mathbf{q}}_{\text{right arm}}\|_2^2$ | -80.0 | 0–4 |
| Finger qvel, single-foot ground contact | $\|\dot{\mathbf{q}}_{\text{finger}}\|_2\,\mathbbm{1}_{\text{single-foot}}$ | -3000.0 | 1–3 |
| Arm qvel, single-foot ground contact | $\|\dot{\mathbf{q}}_{\text{right arm}}\|_2\,\mathbbm{1}_{\text{single-foot}}$ | -1300.0 | 1–3 |
| *Heading / command shaping* | | | |
| Heading toward object | $((\psi_{\text{GraspObj}}-\psi_{\text{robot}})/\pi)^2$ | -10000.0 | 0 |
| Object in view | $\mathbbm{1}[y_{\text{right hand}}>y_{\text{GraspObj}}-0.1]+\mathbbm{1}[y_{\text{left hand}}<y_{\text{GraspObj}}+0.1]$ | -1.0 | 0 |
| Large linear $v_x$ command | $\sum\max(0,|v_x^{\text{cmd}}|-0.5)$ | -20.0 | 0–4 |
| Large linear $v_y$ command | $\sum\max(0,|v_y^{\text{cmd}}|-0.5)$ | -20.0 | 0–4 |
| Large angular $\omega$ command | $\sum\max(0,|\omega^{\text{cmd}}|-0.5)$ | -20.0 | 0–4 |
| Large upper-body actions | $\sum\max(0,|u_{\text{upper}}|-2\pi)$ | -20.0 | 0–4 |
| Zero $v_x,v_y,\omega$ cmd | $|v_x^{\text{cmd}}|+|v_y^{\text{cmd}}|+|\omega^{\text{cmd}}|$ | -12.0 | 1–3 |
| Zero $v_x,v_y$ cmd | $|v_x^{\text{cmd}}|+|v_y^{\text{cmd}}|$ | -4.0 | 4 |
| *Task / object-centric rewards* | | | |
| Robot–Object distance | $\exp(-4(\|p_{\text{robot}}-p_{\text{GraspObj}}\|-0.45)^2)$ | 2.0 | 0–4 |
| Upper-body actions (pose) | $\|\mathbf{q}_{\text{right arm}}\|_2^2$ | -1.0 | 0 |
| Keep hand closed | $\exp(-4(u_{\text{finger}}-u_{\text{close}})^2)$ | 9.0 | 0–1, 3–4 |
| Place objects when near tray | $-\|\mathbf{f}_{\text{PlaceObj}}\|*\mathbbm{1}(\|p_{\text{PlaceObj}}-p_{\text{tray}}\|<0.3)$ | 10.0 | 0–1 |
| Holding object | $\exp(-4\|p_{\text{PlaceObj}}-p_{\text{hand}}\|_2)$ | 1.0 | 0–4 |
| Hand–object distance | $\exp(-10\max_k\|p^{(k)}_{\text{finger}}-p_{\text{GraspObj}}\|_2)$ | 20.0 | 3–4 |
| Grasp based on obj–finger dir | $-\hat{\mathbf{d}}_{\text{thumb}}^{\top}\hat{\mathbf{d}}_{\text{index}}$ | 5.0 | 3–4 |
| Grasp force | $\sum\|\mathbf{f}_{\text{GraspObj-hand}}\|$ | 1.0 | 3–4 |
| Lift goal distance | $\exp(-10\|p_{\text{GraspObj}}-p_{\text{goal}}\|^2)$ | 10.0 | 3–4 |
| Lift z | $\min(h_{\text{GraspObj}}-h_{\text{table}},0.15)$ | 200.0 | 3–4 |
| Turn around | $-|\mathrm{y}_{\text{robot}}-\mathrm{y}_{\text{desired}}|$ | 15.0 | 4 |
| Right-arm qpos tracking (hold) | $\exp(-4\|\mathbf{q}_{\text{right arm}}-\mathbf{q}^*_{\text{Place}}\|_2)$ | 5.0 | 0–2 |
| Right-arm qpos tracking (front) | $\exp(-4\|\mathbf{q}_{\text{right arm}}-\mathbf{q}^*_{\text{Grasp}}\|_2)$ | 25.0 | 3–4 |
| Finger qvel during arm qvel | $\exp(-6\|\dot{\mathbf{q}}_{\text{arm}}\|_2\|\dot{\mathbf{q}}_{\text{finger}}\|_2)$ | 15.0 | 1–4 |
| Object–table contact move | $\|\mathbf{v}_{\text{GraspObj},xy}\|\,\mathbbm{1}_{\text{table-contact}}$ | -1000.0 | 1–4 |
| Object relative move (hand–obj $v_z$) | $|v^z_{\text{GraspObj}}-v^z_{\text{hand}}|\,\mathbbm{1}_{\text{in-grasp}}$ | -3000.0 | 1–3 |
| Object lean during pick | $|\phi_{\text{GraspObj}}|+|\theta_{\text{GraspObj}}|$ | -500.0 | 0–3 |
| Object non-$z$ velocity during pick | $\|\mathbf{v}_{\text{GraspObj},xy}\|_2$ | -500.0 | 0–3 |

**说明**: 完整奖励表，揭示长程移动操作所需的**重度奖励工程**：终止/平滑/限幅等通用惩罚 + 朝向/指令塑形 + 任务/物体中心奖励，且每项绑定到具体阶段 0–4。这也正是 §6 "奖励工程瓶颈"局限的实证来源。

### Table 4: Low-Level PD Gains / 低层控制器 PD 增益（附录）

| Joint | $K_p$ [N·m/rad] | $K_d$ [N·m·s/rad] |
|---|---|---|
| hip_yaw / hip_roll / hip_pitch | 150 | 2.0 |
| knee | 200 | 4.0 |
| ankle_pitch / ankle_roll | 40 | 2.0 |
| waist_yaw / waist_roll / waist_pitch | 250 | 5.0 |
| shoulder_pitch / shoulder_roll | 100 | 5.0 |
| shoulder_yaw / elbow | 40 | 2.0 |
| wrist_roll / wrist_pitch / wrist_yaw | 20 | 2.0 |
| hand_index / hand_middle / hand_thumb_1 / hand_thumb_2 | 0.5 | 0.1 |
| hand_thumb_0 | 2.0 | 0.1 |

**说明**: G1（含 3 指手）的关节空间 PD 增益。下肢/腰刚度大，手指刚度极小（0.5），体现灵巧手控制的低增益特性，也呼应其高传动比带来的 sim-to-real 难度。

### Table 5: Teacher Hyperparameters / 教师超参（附录）

| Hyperparameter | Value |
|---|---|
| Number of environments | 32768 (2048×8GPUs×2Nodes) |
| Discount factor $\gamma$ | 0.998 |
| Learning rate | 0.00002 |
| Entropy coefficient | 0.01 |
| Value loss coefficient | 1 |
| Init noise std (RL) | 0.5 |
| MLP size | [512, 256, 128] |

**说明**: 教师用 [[PPO]]（定制 TRL 实现）训练，**32768 并行环境（16 GPU）**，高折扣 0.998 契合长程任务。

### Table 6: Student Hyperparameters / 学生超参（附录）

| Hyperparameter | Value |
|---|---|
| Number of environments | 65535 (1024×8GPUs×8Nodes) |
| Number of steps per environment | 1 |
| Learning rate | 0.0002 |

**说明**: 学生用 DAgger+BC 混合训练，**65535 并行环境（64 GPU）**，每环境每次 1 步（在线 DAgger 风格）。

### Table 7: Domain Randomization Parameters / 域随机化参数（附录）

| Parameter | Prob. | Distribution |
|---|---|---|
| *Image Augmentation* | | |
| Brightness | 0.25 | $\mathcal{U}(0.7,2)$ |
| Contrast | 0.25 | $\mathcal{U}(0.5,1.5)$ |
| Hue | 0.5 | $\mathcal{U}(-0.1,0.1)$ |
| Saturation | 0.25 | $\mathcal{U}(0.5,2)$ |
| Gaussian Noise Std | 0.25 | $\mathcal{U}(0.0,0.15)$ |
| Gaussian Blur Kernel Size | 0.25 | $\mathcal{U}(3,5)$ |
| Gaussian Blur Sigma | 0.25 | $\mathcal{U}(0.1,1.5)$ |
| *Lighting* | | |
| Dome Light Intensity | 1.0 | $\mathcal{U}(800,2000)$ |
| Dome Light Yaw Rotation | 1.0 | $\mathcal{U}(-\pi,\pi)$ |
| Dome Light Texture Map | 1.0 | $\mathcal{U}$(Indoor, Clear, Cloudy, Night, Studio) |
| *Material Randomization* | | |
| Robot Material - Roughness | 1.0 | $\mathcal{U}(0.0,0.8)$ |
| Robot Material - Metallic | 1.0 | $\mathcal{U}(0.0,0.8)$ |
| Robot Material - Specular | 1.0 | $\mathcal{U}(0.0,0.8)$ |
| Floor Material Texture | 1.0 | $\mathcal{U}$(Wood, Carpet, Masonry, Metals, Natural, Plastics, Stone, Wall Board) |
| Table Material Texture | 1.0 | $\mathcal{U}$(Wood) |
| Object Material Texture | 1.0 | $\mathcal{U}$(All Base Materials) |
| *Table Physical Properties* | | |
| Table Height (m) | 1.0 | $\mathcal{U}(0.65,0.6775)$ |
| Table Depth (m) | 1.0 | $\mathcal{U}(0.7,0.75)$ |
| Table Width (m) | 1.0 | $\mathcal{U}(1.4,1.6)$ |
| Table Thickness (m) | 1.0 | $\mathcal{U}(0.035,0.04)$ |
| *Camera Extrinsics* | | |
| Position Noise - X (m) | 1.0 | $\mathcal{U}(-0.02,0.02)$ |
| Position Noise - Y (m) | 1.0 | $\mathcal{U}(-0.05,0.05)$ |
| Position Noise - Z (m) | 1.0 | $\mathcal{U}(-0.02,0.02)$ |
| Rotation Noise - Roll (rad) | 1.0 | $\mathcal{U}(-0.05,0.05)$ |
| Rotation Noise - Pitch (rad) | 1.0 | $\mathcal{U}(-0.1,0.1)$ |
| Rotation Noise - Yaw (rad) | 1.0 | $\mathcal{U}(-0.05,0.05)$ |

**说明**: 完整随机化表。注意外参随机化中 **Y 轴（左右）与 Pitch 容差最大**（±0.05m / ±0.1rad），对应硬件视角差异的主要方向；图像增强多为 25% 概率施加，而光照/材质/外参为 100% 概率——这是 Figure 13 消融结论"互补且缺一不可"的配置依据。

---

## 实验

### 任务与平台

| 项目 | 内容 |
|------|------|
| 任务 | 连续移动操作循环：两桌间往返，**走 → 放置 → 抓取新物 → 转身**，长时程重复 |
| 机器人 | 29-DoF [[Unitree G1]] + 7-DoF 三指灵巧手 |
| 感知 | Intel RealSense D435i（onboard RGB，$108\times192$） |
| 推理硬件 | 桌面工作站 Intel i9-14900K + NVIDIA RTX 4090 |
| 训练算力 | 教师 16 GPU（L40S，2 节点）；学生至多 64 GPU（L40S，8 节点） |
| 评测仿真 | IsaacSim / Isaac Lab（tiled rendering），并做 Isaac→MuJoCo 交叉检验 |

### 实现细节

- **底层**: 预训练 [[HOMIE]] WBC 作为指令 API；VIRAL 在其上扩展手指动作。
- **教师**: 目标条件 RL，特权观测 226 维，delta 动作空间，分阶段奖励 + RSI（200 演示）；PPO/TRL，32768 环境，$\gamma=0.998$，lr 2e-5。
- **学生**: [[DINOv3]] 视觉编码（RGB→128 维）+ 本体感受拼接，history-aware 策略头；DAgger+BC 混合（$\alpha=0.5$），65535 环境，lr 2e-4。
- **Sim-to-Real**: 灵巧手 SysID（armature/刚度/阻尼）+ 相机外参真实到仿真标定 + 大规模视觉/物理随机化（Table 7）；**零真机微调**部署。

### 关键实验结论

- **鲁棒性（§3.1, Fig 7）**: 59 次连续真机试验成功 **54 次**；与专家(>1000h，100% / 21.4s)、非专家(~1h，73%)对比，VIRAL **周期 20.2s（比专家更快）、成功率逼近专家**，远超非专家。
- **泛化（§3.2, Fig 8）**: 托盘/物体位置、起始姿态、桌高桌型、桌布色、光照、物体类别变化下均无需调参完成。
- **消融**: RSI（无→<10%，有→~95%）与 delta 动作空间均为成功**必需**（Fig 9）；[[DINOv3]] 主干优（Fig 10）；DAgger-BC $\alpha=0.5$ 最佳（Fig 11）；history-aware 优于单步（Fig 12）；关闭全部随机化掉到 0.649（-35.1%），各分量互补（Fig 13）；多物体训练优于单物体（Fig 16）。
- **算力 scaling（Fig 14/15）**: 1-2 GPU 教师常彻底失败、永远到不了高成功率，8-16 GPU 稳定 >90%；学生 1→64 GPU 收敛更快、更稳、最终更好——**算力是必要条件**。

---

## 批判性思考

### 优点
1. **诚实的"全栈配方"定位 + 强真机证据**: 不包装新算法，而是把每个设计的"有用/失败/交互"讲清楚，并用 59 次连续真机试验、专家/非专家对照、零微调泛化来支撑，落地说服力强。
2. **把"算力规模"作为一等公民量化**: 1-64 GPU 的 scaling 曲线显示低算力会**直接失败而非仅变慢**，这一发现对后续 sim-to-real 移动操作研究有重要的资源预期校准价值。
3. **真实到仿真对齐的工程细节扎实**: 灵巧手 SysID + 相机外参标定/随机化直击高传动比手与外参漂移这两个最常被忽视的硬件 gap，附录给出完整随机化与超参表，复现信息相对充分。
4. **作者自己写了诚实的局限（§6）**: 物理/任务/奖励/硬件四个"覆盖鸿沟"分析坦率，指出 sim-to-real 对通用移动操作"近期内难以企及"，并提出与真机模仿学习/基础模型融合的方向。

### 局限性
1. **任务单一、奖励工程极重**: 全文围绕单一"走-放-抓-转"循环；Table 3 数十项手工奖励 + 5 阶段 shaping 正是作者承认的"reward engineering 瓶颈"——该配方**难以规模化到上千家务任务**。
2. **缺少与同类方法的定量基线对比**: 真机只与人类遥操作比，仿真消融多为自身变体，**没有与其它人形/移动操作 sim-to-real 方法的成功率横评**，难判断相对 SOTA 位置。
3. **算力门槛极高**: 64 GPU(L40S)的学生训练对绝大多数实验室不可复现；论文虽诚实指出算力必要，但也意味着方法的**可及性差**。
4. **泛化为定性、覆盖有限**: Fig 8 的泛化是视频定性展示，缺量化成功率随扰动幅度的曲线；物体仍是刚体、桌面场景，未触及可变形/接触丰富/真正开放环境。
5. **依赖特定 WBC（HOMIE）与特定本体**: 虽声称可换 WBC，但全部实验绑定 G1 + HOMIE，跨本体/跨 WBC 的可移植性未实证。

### 潜在改进方向
1. 引入**自动/程序化奖励与任务生成**（或语言条件奖励）以缓解 §6 的奖励/任务覆盖瓶颈，向多任务移动操作扩展。
2. 补充与现有人形 loco-manipulation sim-to-real 方法的**定量横评**，并把泛化做成"扰动幅度 vs 成功率"曲线。
3. 探索**算力-性能的帕累托前沿**：在小算力下用更样本高效的蒸馏/课程，降低 64 GPU 门槛。
4. 验证**跨本体/跨 WBC**迁移（如 TWIST2/SONIC），以证实"框架不过拟合特定 WBC"的主张。
5. 把 §6 的"sim-to-real 与真机模仿学习/基础模型融合"落到具体方案（如仿真做骨架、真机数据补长尾物理/任务）。

### 可复现性评估
- [ ] 代码开源（论文/JSON 未给出代码链接，仅有项目主页与视频）
- [ ] 预训练模型（未声明发布权重）
- [x] 训练细节完整（附录给出观测维度、完整奖励表、PD 增益、教师/学生超参、域随机化表）
- [x] 仿真器可获取（IsaacSim/Isaac Lab、MuJoCo 公开；底层 HOMIE WBC 有公开工作）
- [ ] 算力可及性（64 GPU L40S 规模对多数团队不可复现）

---

## 速查卡片

> [!summary] VIRAL: Visual Sim-to-Real at Scale for Humanoid Loco-Manipulation
> - **核心**: 特权 RL 教师 → RGB 学生蒸馏的两段式全栈配方，纯仿真训练、零样本真机部署人形长程移动操作。
> - **方法**: 教师（特权观测 + delta 动作 + RSI + 分阶段奖励 + HOMIE WBC API，PPO/16GPU）→ 学生（DINOv3 RGB + 本体，DAgger+BC α=0.5，64GPU tiled rendering）→ 灵巧手 SysID + 相机外参对齐 + 大规模视觉随机化。
> - **结果**: Unitree G1 连续 54/59 次成功、周期 20.2s（比专家 21.4s 更快），逼近专家、远超非专家；零微调泛化；scaling 显示低算力会直接失败、8-16/64 GPU 才稳定 >90%。
> - **链接**: https://viral-humanoid.github.io

---

*笔记创建时间: 2026-06-29*
