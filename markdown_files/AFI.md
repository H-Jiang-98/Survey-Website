---
title: "Affordance Field Intervention: Enabling VLAs to Escape Memory Traps in Robotic Manipulation"
method_name: "AFI"
authors: [Siyu Xu, Zijian Wang, Yunke Wang, Chenghao Xia, Tao Huang, Chang Xu]
year: 2026
venue: CVPR
tags: [VLA, affordance, spatial-affordance-field, memory-trap, OOD-generalization, training-free, plug-in]
zotero_collection: _inbox
image_source: online
arxiv_html: https://arxiv.org/html/2512.07472v1
created: 2026-06-29
---

# Affordance Field Intervention: Enabling VLAs to Escape Memory Traps in Robotic Manipulation

## 元信息

| 项目 | 内容 |
|------|------|
| 作者 | Siyu Xu, Zijian Wang, Yunke Wang, Chenghao Xia, Tao Huang, Chang Xu |
| 机构 | 悉尼大学（University of Sydney）等 |
| 会议 | CVPR 2026 |
| 类别 | VLA（Vision-Language-Action）/ 机器人操作 / 空间可供性 |
| 日期 | 2025-12（arXiv v1） |
| 项目主页 | （无） |
| 链接 | [arXiv](https://arxiv.org/abs/2512.07472) / [PDF](https://arxiv.org/pdf/2512.07472) |

---

## 一句话总结

> 把 3D 空间可供性场（SAF）做成即插即用插件：通过本体感知检测 [[VLA]] 的"记忆陷阱"、回滚到高可供性历史位姿、再用 SAF 引导采样并重排 VLA 候选轨迹，无需训练即在 OOD 场景把 $\pi_0$/$\pi_{0.5}$ 成功率平均提升约 23.5%。

---

## 核心贡献

1. **提出并刻画"记忆陷阱"（Memory Trap）失效模式**: 指出端到端 [[VLA]] 在分布偏移（OOD）下会僵硬复现训练中记忆的轨迹，把末端执行器（EEF）开向过时位置而非更新后的目标，根因是缺乏显式 3D 空间推理。
2. **训练无关、模型无关的混合框架 AFI**: 将 [[Spatial Affordance Field|3D 空间可供性场]] 当作**按需触发的插件**，不改动任何 VLA 参数，即可套用于任意预训练 VLA 主干（$\pi_0$、$\pi_{0.5}$），并天然支持多策略集成。
3. **基于本体感知的检测—回滚—引导采样三段式干预**: 用 [[Proprioception|本体感知]] 检测陷阱 → 回滚到历史最低代价位姿 → 树状 [[SAF-Guided Sampling|SAF 引导采样]] 提出航点并用 SAF 作为打分器重排 VLA 轨迹，真实机器人 OOD 平均 +23.5%、LIBERO-Pro +20.2%。

---

## 问题背景

### 要解决的问题
端到端 [[VLA]] 模型把视觉观测 + 语言指令直接映射为动作，在分布内表现强，但在 **OOD（如目标物体位置被显著扰动）** 时极度脆弱：模型会复现训练时记忆的轨迹，把 EEF 开向**原始（过时）位置**而忽略新的空间线索。作者把这一失效模式命名为 **"记忆陷阱（Memory Trap）"**，目标是在**不重新训练 VLA** 的前提下让其逃出陷阱、适配更新后的场景。

### 现有方法的局限
1. **端到端 VLA 缺少显式 3D 空间推理**: 仅按训练分布隐式拟合"视觉语言→动作"映射，无法在陌生环境中可靠识别可交互区域，遇扰动即退化为记忆轨迹。
2. **RL 微调路线代价高**: 用强化学习增强泛化需要可靠奖励信号，依赖大量人工标注或复杂仿真，且难以扩展到大规模真实场景。
3. **纯 VLM 规划（如 [[VoxPoser]]、[[ReKep]]）成功率低**: (a) VLM 生成的运动计划缺乏细粒度几何理解，常产生不可行动作；(b) 重度依赖任务特定的 prompt engineering 来生成约束，脆弱且跨场景不可迁移。ReKep 在 Place Carrot 上仅 36%。

### 本文的动机
- VLM 擅长**语义 grounding**（定位"该靠近/避开哪里"），VLA 擅长**鲁棒动作生成**，二者互补；
- 把 SAF 当作一个**显式、可解释的 3D 几何先验**，只在检测到记忆陷阱时按需介入，用空间锚点"打断" VLA 的僵硬记忆，再让 VLA 在新航点上重新生成动作；
- 由此既保留 VLA 的动作能力，又补上其缺失的 3D 空间推理，且**无需训练、与主干解耦**。

---

## 方法详解

### 模型架构

AFI 是一个**混合（hybrid）插件式**流水线，叠加在任意预训练 [[VLA]] 之上（见 Figure 3），核心三步：
- **输入**: RGB 观测 $I_t^{rgb}$ + 深度图 $I_t^{depth}$ + 语言指令 $\tau$ + 机器人本体状态（EEF 位姿 $\mathbf{p}_t$）
- **可供性构建**: [[GPT-4o]] 分解任务阶段并给出当前目标物体词 → [[Grounded-SAM]] 分割得到 2D mask → 反投影到 3D 体素栅格构建 [[Spatial Affordance Field|SAF]]（见 Figure 2）
- **三段式干预**: ① [[Memory Trap Detection|记忆陷阱检测]]（本体感知）→ ② [[Trajectory Rollback|轨迹回滚]] 到历史最低 SAF 代价位姿 → ③ 树状 [[SAF-Guided Sampling|SAF 引导采样]]：在航点处让 VLA 生成候选，SAF 累计代价打分选最优
- **输出**: 选中的最优末端轨迹 $\boldsymbol{\xi}^*$（由 SAF 选航点 + VLA 补动作）
- **关键属性**: 不改 VLA 参数，**训练无关、模型无关**，端到端额外延迟约 185ms（5Hz 控制可用）

### 核心模块

#### 模块1: Spatial Affordance Field（SAF）构建

**设计动机**: 为端到端 VLA 补上缺失的**显式 3D 空间表征**，用一个连续的"代价场"标记机器人应靠近（目标）或避开（障碍）的区域。

**具体实现**（两阶段，见 Figure 2）:
- **阶段 a — VLM 推断可供性**: 用 [[GPT-4o]] 把高层指令 $\tau$ 分解为时序子目标（pick → move → place），抽取每阶段的目标词元（如 "carrot"、"blue pan"）。执行中检测到子目标切换时，**自动更新目标识别**，使 SAF 随多阶段任务动态适配。
- **阶段 b — 3D 空间 grounding**: 目标词送入 [[Grounded-SAM]] 得 2D mask $M_{\text{target}}$；结合深度图 $I_t^{depth}$ 与相机内参反投影，得目标点云 $\mathcal{P}_{\text{target}}$；连同场景点云 $\mathcal{P}_{\text{scene}}$ 投影到 $N\times N\times N$ 体素栅格 $\mathcal{V}$。
- **两个互补子场**:
  - **目标引导场 $V_{\text{target}}$**: 对每个体素计算到目标质心 $\mathbf{c}_{\text{target}}$ 的欧氏距离并做距离变换，离目标越远代价越高，吸引 EEF 靠近目标；
  - **障碍规避场 $V_{\text{obst}}$**: 被场景点云占据/邻近障碍的体素赋高值以避碰。为避免过度保守，用启发式 mask：(1) 豁免 EEF 紧邻区域允许近距操作，(2) 在目标周围留缓冲带允许抓取所需的接近动作。
- 两子场加权融合（公式2），再经欧氏距离变换 + 高斯平滑（核 $\sigma$）保证梯度平滑，归一化到 $[0,1]$，**值越低代表越优**（近目标、远障碍）。

#### 模块2: Memory Trap Detection（记忆陷阱检测）

**设计动机**: 用机器人**本体感知**实时判定是否陷入僵硬记忆轨迹，做到"只在真陷阱时介入"，避免误触发。

**具体实现**（双条件同时满足才触发）:
- **条件1（卡住）**: 时间窗 $\Delta t$ 内 EEF 位移 $\|\mathbf{p}_t-\mathbf{p}_{t-\Delta t}\|$ 低于阈值 $\epsilon_{\text{stuck}}$ —— 进入准静止态；
- **条件2（远离目标）**: 到目标距离 $\|\mathbf{p}_t-\mathbf{c}_{\text{target}}\|$ 超过阈值 $\epsilon_{\text{far}}$ —— 用以区分"靠近目标的精细操作（如抓取）"与"卡在错误位置/抓错物体"。
- 双判据保证只在**真正的记忆陷阱**时干预，避免在目标附近合法静止时误报（消融 Table 4 验证自适应检测优于固定步介入）。

#### 模块3: Affordance Field Intervention（可供性场干预）

**设计动机**: 检测到陷阱后，用 SAF 把机器人"拉回安全态"并提供显式空间锚点，打断记忆、引导 VLA 重新生成可行动作。

**具体实现**（回滚 + 两阶段树搜索）:
- **历史回滚（Historical Rollback）**: 维护 $N$ 步历史位姿缓冲 $\mathbf{P}_{\text{hist}}$，选其中 SAF 代价最低者作为回滚目标 $\mathbf{p}_{\text{rollback}}$（公式3），EEF 回到该安全低代价态，作为后续树搜索的根节点。
- **Stage 1 — 局部 SAF 引导航点采样**: 在 $\mathbf{p}_{\text{rollback}}$ 半径 $r$ 邻域 $\mathcal{N}$ 内采样候选位置，取代价最低的 $N$ 个作为中间航点 $\{\mathbf{p}_i^{\text{way}}\}$（公式4），构成树的一级子节点。
- **Stage 2 — 航点处 VLA 生成轨迹**: 机器人依次导航到每个航点，查询 VLA 策略 $\pi_{\text{VLA}}$ 生成 $K$ 个动作候选（扩散策略用不同噪声温度/种子采样多样化候选）；对每个动作块经**前向运动学**转为 EEF 轨迹 $\boldsymbol{\xi}_{i,k}$，计算累计 SAF 代价（公式5）。
- **全局重排选优**: 在 $N\times K$ 个叶子节点中取累计代价最小的轨迹 $\boldsymbol{\xi}^*$（公式6）执行。即 **SAF 选航点（空间推理）+ VLA 补动作（任务能力）** 的层次化协同。

### 关键公式与机制

#### 公式1: [[VLA]] 动作生成与执行

$$
a_{t}\sim\pi_{\text{VLA}}(I_{t}^{rgb},\tau),\quad \Delta\mathbf{d}_{t}=\text{controller}(a_{t})
$$

**含义**: VLA 以 RGB 图与指令为输入采样动作 $a_t$，控制器执行后产生 EEF 位移 $\Delta\mathbf{d}_t$。模仿学习训练的 VLA 在 OOD 下会沿记忆轨迹行动而不适配扰动——这正是记忆陷阱的形式化起点。

**符号说明**:
- $\pi_{\text{VLA}}$: 预训练 VLA 策略；$I_t^{rgb}$: $t$ 时刻 RGB 观测；$\tau$: 语言指令
- $a_t$: 采样动作；$\Delta\mathbf{d}_t$: 3D 工作空间中的 EEF 位移

#### 公式2: 空间可供性场融合

$$
V_{\text{SAF}}=w_{\text{target}}V_{\text{target}}+w_{\text{obst}}V_{\text{obst}}
$$

**含义**: 将目标引导场与障碍规避场按权重线性融合，得到统一的连续可供性代价场；再经距离变换 + 高斯平滑 + 归一化到 $[0,1]$，**低值=高可供性**。

**符号说明**:
- $V_{\text{target}}$: 目标引导子场（离目标越远值越高）
- $V_{\text{obst}}$: 障碍规避子场（近障碍值越高）
- $w_{\text{target}}, w_{\text{obst}}$: 平衡目标吸引与避障的超参；$\sigma$: 高斯平滑核大小

#### 公式3: 历史回滚位姿选择

$$
\mathbf{p}_{\text{rollback}}=\arg\min_{\mathbf{p}\in\mathbf{P}_{\text{hist}}}V_{\text{SAF}}(\mathbf{p})
$$

**含义**: 在最近 $N$ 步历史位姿中选 SAF 代价最低者作为回滚目标，把 EEF 拉回安全低代价态，缓解记忆陷阱的即时影响，并作为树搜索根节点。

**符号说明**:
- $\mathbf{P}_{\text{hist}}=\{\mathbf{p}_{t-n},\ldots,\mathbf{p}_{t-1}\}$: $N$ 步历史 EEF 位置缓冲
- $V_{\text{SAF}}(\mathbf{p})$: 在位置 $\mathbf{p}$ 处查询 SAF 的代价值

#### 公式4: 局部 SAF 引导航点采样

$$
\{\mathbf{p}_{i}^{\text{way}}\}_{i=1}^{N}=\underset{\mathbf{p}\in\mathcal{N}(\mathbf{p}_{\text{rollback}},r)}{\arg\min^{N}}\;V_{\text{SAF}}(\mathbf{p})
$$

**含义**: 在回滚位姿半径 $r$ 的局部邻域内，选出 SAF 代价最低的 $N$ 个位置作为中间航点（树的一级子节点），作为靠近目标、远离障碍的空间锚点。

**符号说明**:
- $\mathcal{N}(\mathbf{p}_{\text{rollback}},r)$: 以 $\mathbf{p}_{\text{rollback}}$ 为中心、半径 $r$ 的局部邻域
- $\arg\min^{N}$: 取代价最小的前 $N$ 个位置；$N$: 航点数（消融最优为 10）

#### 公式5: 候选轨迹累计可供性代价

$$
\mathcal{V}(\boldsymbol{\xi}_{i,k})=\sum_{j=1}^{H}V_{\text{SAF}}(\mathbf{p}_{j}^{i,k})
$$

**含义**: 对航点 $i$、第 $k$ 个 VLA 动作候选，经前向运动学得 EEF 轨迹后，沿地平线 $H$ 累加每点的 SAF 代价，作为该候选轨迹的整体打分（越低越优）。

**符号说明**:
- $\boldsymbol{\xi}_{i,k}=\{\mathbf{p}_j^{i,k}\}_{j=1}^{H}$: 第 $i$ 航点第 $k$ 候选的 EEF 轨迹；$H$: 动作块地平线
- $\mathbf{p}_j^{i,k}$: 第 $j$ 步 EEF 位置；$K$: 每航点的 VLA 候选数

#### 公式6: 全局最优轨迹选择

$$
\boldsymbol{\xi}^{*}=\arg\min_{i,k}\mathcal{V}(\boldsymbol{\xi}_{i,k})
$$

**含义**: 在 $N\times K$ 个候选（树的叶子节点）中取累计可供性代价最小者执行，实现 SAF 空间推理与 VLA 动作能力的层次化融合。

**符号说明**:
- $\boldsymbol{\xi}^{*}$: 被选中执行的最优轨迹；$i$ 遍历航点，$k$ 遍历 VLA 候选

---

## 关键图表

<!-- 图片均使用 arXiv HTML 在线链接，未本地化 -->

### Figure 1: Memory Trap in VLAs / 记忆陷阱示意

![Figure 1](https://arxiv.org/html/2512.07472v1/x1.png)

**说明**: 直观展示"记忆陷阱"失效模式。在 Case 1 与 Case 2 中，目标物体已移动，但 VLA 仍把 EEF 开向**训练时记忆的原始位置**，忽略新的空间线索而失败。本图是全文动机的总纲——VLA 在 OOD 下"记忆 > 适配"。

### Figure 2: Spatial Affordance Field (SAF) Construction / 可供性场构建

![Figure 2](https://arxiv.org/html/2512.07472v1/x2.png)

**说明**: SAF 两阶段构建流程。(a) [[GPT-4o]] 把指令分解为时序阶段并给出当前目标物体（如 "carrot"、"blue pan"）；(b) 目标词送 [[Grounded-SAM]] 分割得 2D mask，反投影到 3D 构建 SAF，颜色梯度即可供性代价。佐证"语义 grounding 由 VLM、几何代价由 3D 场"的分工。

### Figure 3: Overview of AFI / 整体框架

![Figure 3](https://arxiv.org/html/2512.07472v1/x3.png)

**说明**: AFI 三段式流程总图。(1) 记忆陷阱检测：SAF 评估 VLA 预测动作，监控 EEF 速度与到目标距离；(2) 轨迹回滚：检测到后回滚到历史最低 SAF 代价位姿；(3) SAF 引导采样：在 SAF 采样航点处让 VLA 生成候选轨迹，选累计 SAF 代价最低者执行。是理解方法的核心图。

### Figure 4: Real-world AFI Execution Rollout / 真实执行时序

![Figure 4](https://arxiv.org/html/2512.07472v1/x4.png)

**说明**: 一次完整 AFI 执行。上排：$t=50$ 接近错误位置时检测到记忆陷阱，随后回滚到低代价历史位姿；下排：$t=70$–$79$ SAF 引导采样生成候选轨迹，选中最优（绿色），$t=80$–$90$ 执行并成功。把抽象的"检测—回滚—采样"落到真实时间轴上。

### Figure 5: LIBERO Position Perturbations / 仿真位置扰动

![Figure 5](https://arxiv.org/html/2512.07472v1/fig/libero_pro.png)

**说明**: LIBERO 仿真中的物体位置扰动可视化——目标物 "black bowl on the cookie box" 被显著移位。说明 LIBERO-Pro OOD 评测协议如何构造分布偏移，对应 Table 2 的实验设置。

### Figure 6: SAF Value Evolution / 可供性代价随时间演化

![Figure 6](https://arxiv.org/html/2512.07472v1/x5.png)

**说明**: 各操作任务中 SAF 代价随执行时间的曲线（值越低可供性越高）。代价随 EEF 接近目标而下降，并在阶段切换（如 picking→placing）时**动态跳变更新**，验证 SAF 能随任务语义自适应，对多步任务至关重要。

### Figure 7: OOD Test Scenarios / 分布外测试场景

![Figure 7](https://arxiv.org/html/2512.07472v1/x6.png)

**说明**: 四个任务（Place Carrot / Remove Lid / Slot Pen / Stack Tape，逐行）× 五种条件：(a) 分布内、(b) 位置偏移（±5–15cm）、(c) 颜色/外观变化、(d) 任务偏移（物理属性变化或加干扰物）、(e) 背景偏移（桌面白→黑）。定义了 Table 1 的全部评测维度。

### Table 1: Real-World Results / 真实世界成功率（AgileX Piper，每场景 20 试）

| Task | Method | In Dist. | Position | Color | Task | Background | Average SR. |
|------|--------|----------|----------|-------|------|-----------|-------------|
| Place Carrot | ReKep | 8/20 | 7/20 | 9/20 | 5/20 | 7/20 | 36.0% |
| Place Carrot | $\pi_{0}$ | 17/20 | 6/20 | 13/20 | 15/20 | 10/20 | 61.0% |
| Place Carrot | **$\pi_{0}$-AFI (Ours)** | **20/20** | **13/20** | **17/20** | **18/20** | **19/20** | **87.0% (↑26.0%)** |
| Remove Lid | $\pi_{0}$ | 20/20 | 8/20 | 17/20 | 5/20 | 13/20 | 63.0% |
| Remove Lid | **$\pi_{0}$-AFI (Ours)** | **20/20** | **12/20** | **19/20** | **11/20** | **18/20** | **80.0% (↑17.0%)** |
| Slot Pen | $\pi_{0}$ | 16/20 | 11/20 | 13/20 | 15/20 | 5/20 | 60.0% |
| Slot Pen | **$\pi_{0}$-AFI (Ours)** | **19/20** | **16/20** | **16/20** | **19/20** | **12/20** | **82.0% (↑22.0%)** |
| Stack Tape | $\pi_{0}$ | 18/20 | 9/20 | 16/20 | 13/20 | 8/20 | 64.0% |
| Stack Tape | **$\pi_{0}$-AFI (Ours)** | **20/20** | **15/20** | **20/20** | **17/20** | **14/20** | **86.0% (↑22.0%)** |
| Stack Tape | $\pi_{0.5}$ | 20/20 | 7/20 | 17/20 | 10/20 | 7/20 | 61.0% |
| Stack Tape | **$\pi_{0.5}$-AFI (Ours)** | **20/20** | **14/20** | **19/20** | **15/20** | **14/20** | **82.0% (↑21.0%)** |
| Stack Tape | **$\pi_{0}$+$\pi_{0.5}$-AFI (Ours)** | **20/20** | **16/20** | **20/20** | **16/20** | **17/20** | **89.0% (↑25.0%)** |

**说明**: AFI 在所有任务、所有分布偏移、不同 VLA 主干上一致提升，平均增益 17.0%–26.0%。Position 与 Background 偏移提升最显著（Place Carrot 背景偏移 95% vs 50%）。ReKep（纯 VLM 规划）仅 36%，凸显"混合 VLA+SAF"对纯 VLM 规划的优势。集成 $\pi_0$+$\pi_{0.5}$ 达 89%，体现模型无关设计天然支持多策略融合。

### Table 2: LIBERO-Pro Simulation Results / 仿真成功率（LIBERO-Pro OOD 协议）

**LIBERO-Spatial (OOD)**

| Task | $\pi_{0.5}$ | $\pi_{0.5}$-AFI |
|------|-------------|------------------|
| Pick(between(plate, ramekin), plate) | 70.0% | **82.0%** |
| Pick(next_to(ramekin), plate) | 22.0% | **54.0%** |
| Pick(table_center, plate) | 96.0% | **98.0%** |
| Pick(on(cookie_box), plate) | 74.0% | **88.0%** |
| Pick(on(ramekin), plate) | 26.0% | **54.0%** |
| Pick(next_to(cookie_box), plate) | 36.0% | **72.0%** |
| Pick(next_to(plate), plate) | 54.0% | **82.0%** |
| **Average** | 54.0% | **75.7%** |

**LIBERO-Object (OOD)**

| Task | $\pi_{0.5}$ | $\pi_{0.5}$-AFI |
|------|-------------|------------------|
| Place(alphabet_soup, basket) | 42.0% | **64.0%** |
| Place(bbq_sauce, basket) | 54.0% | **72.0%** |
| Place(butter, basket) | 78.0% | **82.0%** |
| Place(chocolate_pudding, basket) | 88.0% | **90.0%** |
| Place(cream_cheese, basket) | 42.0% | **56.0%** |
| Place(ketchup, basket) | 46.0% | **66.0%** |
| Place(milk, basket) | 88.0% | **92.0%** |
| Place(orange_juice, basket) | 70.0% | **80.0%** |
| Place(salad_dressing, basket) | 16.0% | **64.0%** |
| Place(tomato_sauce, basket) | 40.0% | **66.0%** |
| **Average** | 56.4% | **73.2%** |

**说明**: 在引入位置扰动的 LIBERO-Pro OOD 协议下，AFI 几乎全子任务提升，Spatial 平均 54.0%→75.7%、Object 56.4%→73.2%（正文另报 LIBERO-Spatial 78.2% vs 52.4%、Object 82.5% vs 67.3% 的另一组统计）。提升最大的恰是基线最差的子任务（如 salad_dressing 16%→64%），说明 SAF 对"严重偏移"最有效。

### Table 3: Ablation on Position Shifts / 不同轴向位置偏移消融（每配置 20 试）

| $(\Delta X,\Delta Y)$ | (0,0) | (+10,0) | (+15,0) | (0,+10) | (0,+15) | (+10,+10) |
|-----------------------|-------|---------|---------|---------|---------|-----------|
| $\pi_{0}$ | 17/20 | 3/20 | 1/20 | 4/20 | 0/20 | 6/20 |
| **$\pi_{0}$-AFI** | **20/20** | **8/20** | **3/20** | **10/20** | **2/20** | **13/20** |

**说明**: $\pi_0$ 单轴偏移即灾难性退化（$\Delta X{=}{+}15$cm 仅 5%、$\Delta Y{=}{+}15$cm 为 0），印证其记忆特定空间模式。对角偏移 (+10,+10) 反而保 30%，因恰好落在训练分布空间覆盖内。AFI 全配置提升（单轴 40%、对角 65%），但极端 OOD（±15cm）增益递减——说明 AFI 是**补充**而非取代 VLA 的学得先验。

### Table 4: Ablation on Key Components / 关键组件消融（位置偏移场景，20 试）

| Method | Success | Failure |
|--------|---------|---------|
| $\pi_{0}$ | 6/20 | 14/20 |
| **$\pi_{0}$-AFI** | **13/20** | **7/20** |
| w/o Rollback | 8/20 | 12/20 |
| Fixed-step at 30 | 12/20 | 8/20 |
| Fixed-step at 60 | 11/20 | 9/20 |
| Fixed-step at 90 | 9/20 | 11/20 |

**说明**: 去掉回滚机制成功率从 65%（13/20）跌到 40%（8/20）——VLA 会偏离可行轨迹太远、SAF 航点无法从失败位姿恢复，验证回滚不可或缺。固定步介入最高仅 60%（Step 30），低于自适应检测 65%，说明实时本体感知监测能"恰在陷阱发生时"介入，兼顾效率与成功率。

### Table 5: Ablation on Waypoint Count / 航点数量消融

| Num of Waypoints | 3 | 8 | 10 | 13 |
|------------------|------|------|------|------|
| Success Rate | 35.0% | 50.0% | **65.0%** | 60.0% |

**说明**: 航点太少（3 个）成功率仅 35%，几乎不优于基线；10 个航点达最优 65%；继续增到 13 个反而略降（60%），提示候选过多会引入次优/冗余分支。10 是探索充分性与噪声之间的甜点。

---

## 实验

### 数据集 / 基准

| 基准 | 平台 / 规模 | 特点 | 用途 |
|------|------------|------|------|
| 真实世界 4 任务 | AgileX Piper 机械臂 + 2× Intel RealSense D435；每任务每场景 20 试 | Place Carrot / Remove Lid / Slot Pen / Stack Tape，覆盖抓放/插入/取盖/堆叠 | 训练/测试 |
| OOD 五条件 | ID + 位置(±5–15cm) + 颜色 + 任务(物性/干扰物) + 背景(白→黑) | 系统性分布偏移 | 测试 |
| [[LIBERO]]-Pro | LIBERO-Spatial / LIBERO-Object 子套件 | 对子任务目标位置注入扰动，避免渲染错误/碰撞/语义矛盾 | 测试 |

### 实现细节

- **VLA 主干（baseline）**: $\pi_0$、$\pi_{0.5}$（真实世界）；官方发布的 $\pi_{0.5}$-LIBERO checkpoint（仿真）。**AFI 不改其参数**。
- **可供性管线**: [[GPT-4o]] 任务分解 + [[Grounded-SAM]] 开放词表分割 + 深度反投影 + $N\times N\times N$ 体素栅格；距离变换 + 高斯平滑（核 $\sigma$）+ 归一化 $[0,1]$。
- **检测阈值**: 位移阈 $\epsilon_{\text{stuck}}$、距离阈 $\epsilon_{\text{far}}$、时间窗 $\Delta t$、历史缓冲 $N$ 步。
- **采样**: 航点数 $N{=}10$（最优）、邻域半径 $r$、每航点 VLA 候选数 $K$、动作块地平线 $H$。
- **对比基线**: [[ReKep]]（training-free VLM 规划器）。
- **效率/硬件**: NVIDIA RTX 4090；SAF 重建 120ms/帧（Grounded-SAM + 点云）、航点生成 + 重排 15ms，端到端约 185ms，可支撑 5Hz 控制；对比纯优化 MPC 需 500+ms/步。

### 关键实验结论

- **真实世界**: AFI 在 4 任务 × 5 条件上一致提升 17–26%，$\pi_0$-AFI 平均最高达 87%；集成 $\pi_0$+$\pi_{0.5}$ 达 89%。总体真实 OOD 平均 **+23.5%**。
- **仿真**: LIBERO-Pro 上 $\pi_{0.5}$-AFI 全面优于 $\pi_{0.5}$，平均 **+20.2%**。
- **模型/任务无关**: 同一框架对 $\pi_0$/$\pi_{0.5}$、对抓放/插入/取盖/堆叠均有效，无需架构改动。
- **消融**: 回滚机制关键（去掉 65%→40%）；自适应检测 > 固定步介入；航点数 10 最优；极端偏移增益递减（AFI 补充而非取代 VLA 先验）。

---

## 批判性思考

### 优点
1. **训练无关 + 模型无关的工程价值高**: 不动 VLA 参数、即插即用，可直接套用任意预训练主干，并天然支持多策略集成（89% > 单主干）；额外延迟仅 185ms，5Hz 实机可用。
2. **问题定义清晰、证据链完整**: "记忆陷阱"命名直观，Figure 1 动机—Table 3 轴向退化（$\Delta Y{=}{+}15$cm 跌至 0）—Figure 6 SAF 代价演化共同支撑"VLA 在记忆、SAF 在补空间推理"的主张。
3. **充分而有诚意的消融**: 回滚、自适应 vs 固定步、航点数三组消融都给出真实数字，且坦诚承认极端 OOD 增益递减、AFI 是补充而非替代。

### 局限性
1. **绝对成功率在强 OOD 下仍很低**: $\Delta X{=}{+}15$cm 仅 3/20、$\Delta Y{=}{+}15$cm 仅 2/20，对角 (+10,+10) 也只 13/20；强分布偏移下框架远未稳健。
2. **重度依赖外部感知组件**: SAF 构建链路依赖 [[GPT-4o]] 阶段分解 + [[Grounded-SAM]] 分割 + 深度图，任一环节出错（分割漂移、深度噪声、阶段误判）都会让代价场失真，论文未分析这些上游误差的传播。
3. **基线/任务覆盖偏窄**: 真实任务集中在桌面抓放/插入/堆叠，长程、接触丰富、可变形物体未覆盖；仿真仅与 $\pi_{0.5}$ 对比、Table 2 两组平均统计（75.7/73.2 vs 正文 78.2/82.5）口径不一致需澄清。
4. **超参手工设定**: $\epsilon_{\text{stuck}}$、$\epsilon_{\text{far}}$、$\Delta t$、$w_{\text{target}}/w_{\text{obst}}$、$\sigma$、$r$、$N$、$K$ 等阈值/权重均经验给定，缺敏感度分析，跨平台迁移成本未知。

### 潜在改进方向
1. 端到端量化上游感知误差（分割/深度/阶段判定）对 SAF 与最终成功率的影响，并引入不确定性感知的代价融合。
2. 把检测阈值与场融合权重做成自适应/可学习，减少手工调参，提升跨本体/跨任务可移植性。
3. 扩展到长程、接触丰富、可变形与动态场景，并补充更多强基线（如 RL 微调 VLA、其他 affordance 方法）以厘清增益来源。

### 可复现性评估
- [ ] 代码开源（论文未给出代码/项目链接）
- [ ] 预训练模型（基于公开 $\pi_0$/$\pi_{0.5}$，但 AFI 实现未声明 release）
- [x] 训练细节完整（训练无关；阈值/管线在正文 + 附录 A 描述，但部分超参数值未列全）
- [x] 数据集可获取（LIBERO/LIBERO-Pro 公开；真实任务依赖 AgileX Piper 硬件）

---

## 速查卡片

> [!summary] AFI: Affordance Field Intervention（让 VLA 逃出记忆陷阱）
> - **核心**: 把 3D 空间可供性场（SAF）做成训练无关、模型无关的即插即用插件，按需介入纠正 VLA 在 OOD 下的"记忆陷阱"。
> - **方法**: GPT-4o 分阶段 + Grounded-SAM 分割 + 深度反投影构建 SAF（目标引导场 + 避障场融合）；本体感知双判据检测陷阱 → 回滚到历史最低代价位姿 → 局部采样航点 + 航点处 VLA 生成候选 → SAF 累计代价重排选最优执行。
> - **结果**: 真实世界 OOD 平均 +23.5%（$\pi_0$-AFI 最高 87%，集成 89%），LIBERO-Pro +20.2%；端到端 +185ms，支持 5Hz。回滚不可或缺（去掉 65%→40%），航点数 10 最优。
> - **代码**: 未公开

---

*笔记创建时间: 2026-06-29*
