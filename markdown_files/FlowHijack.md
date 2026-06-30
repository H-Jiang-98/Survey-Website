---
title: "FlowHijack: A Dynamics-Aware Backdoor Attack on Flow-Matching Vision-Language-Action Models"
method_name: "FlowHijack"
authors: [Xinyuan An, Tao Luo, Gengyun Peng, Yaobing Wang, Kui Ren, Dongxia Wang]
year: 2026
venue: CVPR
tags: [VLA, backdoor-attack, flow-matching, vector-field, robot-security, pi0]
zotero_collection: _inbox
image_source: online
arxiv_html: https://arxiv.org/html/2604.09651v1
created: 2026-06-29
---

# FlowHijack: A Dynamics-Aware Backdoor Attack on Flow-Matching Vision-Language-Action Models

## 元信息

| 项目 | 内容 |
|------|------|
| 作者 | Xinyuan An, Tao Luo, Gengyun Peng, Yaobing Wang, Kui Ren, Dongxia Wang（An 与 Luo 共同一作，Dongxia Wang 通讯） |
| 机构 | 浙江大学；北京智能空间机器人系统技术与应用重点实验室；湖州工业控制技术研究院 |
| 会议 | CVPR 2026 |
| 类别 | VLA（Vision-Language-Action）安全 / 后门攻击 |
| 日期 | 2026-03（arXiv v1） |
| 项目主页 | 无 |
| 链接 | [arXiv](https://arxiv.org/abs/2604.09651) / [PDF](https://arxiv.org/pdf/2604.09651) |

---

## 一句话总结

> 首个针对[[Flow Matching|流匹配]] [[VLA]]（如 [[π0]]）的后门攻击框架：用 $\tau$-条件注入只在生成初期污染向量场，再用动力学模仿正则保证恶意动作运动学上不可区分，在隐蔽的上下文触发器下做到高攻击成功率且不损失良性性能。

---

## 核心贡献

1. **首次系统揭示流匹配 VLA 的后门攻击面**: 指出流匹配策略的核心动作生成机制——[[Vector Field|向量场动力学]]——构成一个全新的、区别于 token 级操纵的攻击面；现有针对自回归离散化 VLA 的后门（如 [[BadVLA]]）无法直接迁移。
2. **隐蔽的上下文感知触发器（Context-Aware Triggers）**: 提出**物体状态触发器**（如倒置的杯子）与**场景语义触发器**（如背景里的盆栽、穿格子衫的人），物理合理、人眼难以察觉，远比像素补丁隐蔽。
3. **动力学劫持 + 模仿正则**: 提出 [[Vector Field Hijacking|向量场劫持损失]] $\mathcal{L}_{\mathrm{BD}}$ 配合**新颖的 $\tau$-条件注入策略**（仅在 $\tau\in[0,\tau_0]$ 的早期阶段注入），再加 [[Dynamics Mimicry|动力学模仿正则]] $\mathcal{L}_{\mathrm{mimic}}$ 强制运动学相似。
4. **强效且绕过现有防御**: 高攻击成功率（ASR 最高 100%）、良性性能几乎无降（SR 降幅通常 <3.5%），并能绕过目标位置过滤、下游干净微调等防御，凸显需要面向生成动力学的新防御。

---

## 问题背景

### 要解决的问题

随着 [[VLA]] 模型走向真实部署，其安全脆弱性成为关键但被严重忽视的问题。本文聚焦**后门攻击**：一个被植入后门的 VLA 在标准评测中表现正常，仅当预设的隐蔽触发器出现时才执行恶意轨迹（抓空、掉落易碎物、甚至对人类协作者造成物理风险）。问题核心是：**如何对连续动作生成的流匹配 VLA 实施隐蔽、有效且物理可信的后门攻击？**

### 现有方法的局限

作者识别出三个关键 gap（针对将 token-based 后门迁移到流匹配模型）：

1. **机制不可迁移**: 标签翻转、token 替换等机制依赖离散输出，而流匹配 VLA 的动作由[[Vector Field|向量场动力学]]与 ODE 积分驱动；恶意行为必须通过**破坏动作生成过程本身**而非改动离散输出来诱发。
2. **触发器过于显眼**: 先前工作（如 [[BadVLA]]）多用像素补丁、显著物体等视觉伪影，在物理环境中极易被发现，削弱了真实威胁。
3. **动作物理上不可信**: 如 Figure 4 所示，现有攻击在流匹配 VLA 上往往产生物理上不合理的动作，其向量场与正常动作差异显著，制造可检测的异常、达不到隐蔽。

此外，**文本触发器不可行**：VLA 模型往往是**视觉主导（vision-dominant）**的（见 Figure 6），对细微文本扰动不敏感；现实中攻击者也难以在用户 prompt 里注入触发词而不被发现。

### 本文的动机

不去攻击 VLM 特征空间（[[BadVLA]] 的做法会与模型的泛化目标冲突——它强迫 VLM 把语义相近的样本如"倒杯/正杯"分开，与 VLM 追求对小变化不变的泛化目标直接矛盾），而是**直击下游的向量场动力学**这一更直接、更有效的攻击面。同时利用流匹配模型（如 [[π0]]）**过采样小 $\tau$** 的训练习惯：只在生成早期注入微小定向误差，再由 ODE 求解器沿整条路径放大——"早期注入、全路径放大"，既高效又隐蔽。

---

## 方法详解

### 整体框架

FlowHijack 由三部分组成（见 Figure 2）：
- **输入**: 多模态观测 $o_t$（视觉 + 语言指令 + 本体状态），输出连续[[Action Chunking|动作块]] $A\in\mathbb{R}^{d\times H}$。
- **核心攻击面**: 被攻击模型为 [[π0]]，其策略训练一个时间条件向量场 $v_\theta(A_t^\tau, o_t, \tau)$。
- **三个组件**: (1) 隐蔽的[[Context-Aware Trigger|上下文感知触发器]]；(2) [[Dynamics Hijacking|动力学劫持]]（$\tau$-条件注入 + 恶意动作设计）；(3) 平衡攻击与隐蔽的[[Loss Function|损失函数]]。
- **威胁模型**: 白盒、微调投毒。攻击者拥有预训练模型的完整白盒访问权，在下游微调阶段注入小规模投毒集 $D_{\text{poison}}$ 并修改目标函数，把"高性能变体"伪装发布，一个投毒模型即可污染大量下游机器人系统。

### 预备知识：流匹配策略（以 π0 为例）

流匹配 VLA 学习把观测 $o_t$ 映射到连续动作块 $A\in\mathbb{R}^{d\times H}$（$d$=动作维度，$H$=预测时域）。核心是训练时间条件向量场 $v_\theta(A_t^\tau, o_t, \tau)$，$\tau\in[0,1]$ 为流匹配时间变量。从简单先验 $p_0(A)$（如 $\varepsilon\sim\mathcal{N}(0,I)$）出发，用线性插值定义带噪动作（公式 1），训练 $v_\theta$ 匹配目标去噪向量场（公式 2），按条件流匹配（CFM）最小化标准损失（公式 3）；推理时解 ODE 生成动作（公式 4）。**关键**：$p_\tau(\tau)$ 常用偏向 $\tau=0$ 的 Beta 分布，以强调初始高噪声阶段——这正是 FlowHijack 利用的弱点。

### 核心模块

#### 模块1: Context-Aware Triggers（上下文感知触发器）

**设计动机**: 触发器必须与机器人操作环境**语义连贯**，又能高效激活后门，且物理可信、人眼难察。

**具体实现**:
- **物体状态触发器（Object State）**: 绑定到环境中相关物体的**状态**。如厨房任务里的"倒置的杯子"、桌面任务里的"打开的抽屉"。形式化为谓词 $P_{\text{state}}(o_t)$，为真即选中投毒。
- **场景语义触发器（Scene Semantic）**: 嵌入背景的物体或配置，当某语义条件满足时激活。如背景出现盆栽、场景中人戴手表/穿格子衫。形式化为 $o_t^{+}=\mathcal{T}_{\text{env}}(o_t)$，$\mathcal{T}_{\text{env}}$ 为引入语义增强的变换。
- 投毒函数 $g(\cdot)$ 把干净样本 $(o_t, A)$ 变为投毒样本 $(o_t^{+}, A^{\star})$，$o_t^{+}$ 含触发器、$A^{\star}$ 为恶意目标动作。

#### 模块2: Dynamics Hijacking（动力学劫持）

**设计动机**: 流匹配的连续性由 ODE 积分（公式 4）支配，让 token-based 后门失效；必须直接操纵学到的向量场 $v_\theta$，使其在触发器 $o_t^+$ 出现时不再指向良性目标 $u(A_t^\tau|A_t)$，而被重定向到生成恶意动作 $A^\star$ 的恶意向量场 $u(A_t^\tau|A^\star)$。

**具体实现**:
- **$\tau$-条件注入**: 核心的[[Vector Field Hijacking|向量场劫持损失]] $\mathcal{L}_{\mathrm{BD}}$（公式 5）**只在 $\tau\in[0,\tau_0]$（小 $\tau_0$）施加**。由于流匹配 VLA 过采样小 $\tau$ 来学习从纯噪声出发的粗方向，攻击者借此把恶意动力学绑到触发观测上，但仅在生成初期；$\tau>\tau_0$ 的向量场基本不动，使后门极难被 $v_\theta$ 的静态分析检测。"早期注入、全路径放大"——初始小误差经 ODE 求解器沿整条轨迹放大成显著偏移。
- **恶意动作设计 $A^\star$**（两种策略）:
  - **Pose-Locking (PL)**: 把 $A^\star$ 设为固定常量动作块（如零位姿/home 配置 $A^\star=A_{\text{const}}$），把轨迹恒定拉向固定点，瘫痪机器人或锁定到特定位姿。
  - **Initial-Perturbation (IP)**: 更隐蔽，定义为相对良性动作的持续偏移 $A^\star=A+\delta_A$（$\delta_A$ 为小常量扰动）。早期引入一致偏置，再被 ODE 放大，导致可靠地错过目标、错位、抓取失败，但运动学上保持合理。

#### 模块3: Dynamics Mimicry Regularizer（动力学模仿正则）

**设计动机**: 现有攻击（[[BadVLA]] 和直接实现的 Pose-Locking）会产生统计性质与正常动作明显不同的向量场，常表现为运动学异常（如速度过快），易被检测（见 Figure 4、Figure 7）。

**具体实现**: 引入 $\mathcal{L}_{\mathrm{mimic}}$（公式 6），强制恶意向量场（给定 $o^+$）的**幅值（$L_2$ 范数）**与良性向量场（给定 $o$）的幅值匹配，并对良性项做 **stop-gradient**。这迫使攻击**只改变向量场方向、保留其物理强度**，使恶意运动的速度剖面与正常动作相似、行为与统计上不可区分。

### 关键公式与机制

#### 公式1: [[Flow Matching|流匹配]]带噪动作插值

$$
A^{\tau}_{t}=\tau A_{t}+(1-\tau)\varepsilon
$$

**含义**: 在真值动作与噪声之间线性插值，构造流匹配时刻 $\tau$ 的带噪动作。

**符号说明**:
- $A_t$: 真值动作块；$\varepsilon\sim\mathcal{N}(0,I)$: 先验噪声
- $\tau\in[0,1]$: 流匹配时间变量

#### 公式2: 目标去噪向量场

$$
u(A^{\tau}_{t}\mid A_{t})=\frac{dA^{\tau}_{t}}{d\tau}=A_{t}-\varepsilon
$$

**含义**: 从噪声到动作的理想速度（目标向量场），网络 $v_\theta$ 需匹配它。

**符号说明**:
- $u(A_t^\tau\mid A_t)$: 引导带噪动作朝向真值的目标流方向，恒为 $A_t-\varepsilon$。

#### 公式3: 标准流匹配损失（CFM）

$$
\mathcal{L}_{\mathrm{FM}}=\mathbb{E}_{p_{1}(A),\epsilon,p_{\tau}(\tau)}\left\|v_{\theta}(A^{\tau}_{t},o_{t},\tau)-u(A^{\tau}_{t}\mid A_{t})\right\|_{2}^{2}
$$

**含义**: 训练向量场 $v_\theta$ 在观测 $o_t$ 条件下逼近目标向量场，保住良性任务性能。

**符号说明**:
- $v_\theta$: 待学习的时间条件向量场；$p_\tau(\tau)$: $\tau$ 的采样分布（常为偏向 $\tau=0$ 的 Beta 分布）
- $\mathbb{E}$: 对动作分布、噪声与 $\tau$ 取期望

#### 公式4: 推理 ODE 积分

$$
\frac{dA_{\tau}}{d\tau}=v_{\theta}(A_{\tau},o_{t},\tau),\quad\tau\in[0,1]
$$

**含义**: 推理时从 $A_0=\varepsilon\sim\mathcal{N}(0,I)$ 出发，沿向量场积分 ODE 得到最终动作 $A$。这一积分过程正是初期误差被放大的途径。

#### 公式5: [[Vector Field Hijacking|向量场劫持损失]]（$\tau$-条件注入）

$$
\mathcal{L}_{\mathrm{BD}}=\mathbb{E}_{\substack{(o^{+},A^{\star})\sim\mathcal{D}_{\mathrm{poison}}\\ \tau\sim U[0,\tau_{0}]}}\left\|v_{\theta}(A^{\tau},o^{+},\tau)-u(A^{\tau}\mid A^{\star})\right\|_{2}^{2}
$$

**含义**: 仅在 $\tau\in[0,\tau_0]$ 的早期窗口，把触发观测 $o^+$ 与生成恶意动作 $A^\star$ 的恶意向量场关联起来。

**符号说明**:
- $A^{\tau}=(1-\tau)\epsilon+\tau A^{\star}$: 朝恶意目标插值的输入
- $u(A^{\tau}\mid A^{\star})=A^{\star}-\varepsilon$: 对应的恶意目标向量场（按公式 2）
- $\tau\sim U[0,\tau_0]$: 注入窗口；$\mathcal{D}_{\mathrm{poison}}$: 投毒样本集

#### 公式6: [[Dynamics Mimicry|动力学模仿正则]]

$$
\mathcal{L}_{\mathrm{mimic}}=\mathbb{E}_{\tau\sim p_{\tau}(\tau)}\Bigl|\,\left\|v_{\theta}(A^{\tau},o^{+})\right\|_{2}-\left\|v_{\theta}(A^{\tau},o)\right\|_{2}^{\mathrm{sg}}\,\Bigr|
$$

**含义**: 强制触发条件下恶意向量场的范数匹配良性向量场的范数，使运动学（速度剖面）一致，从而行为与统计上不可区分。

**符号说明**:
- $\|\cdot\|_2$: 向量场的 $L_2$ 范数（运动强度代理）
- $\mathrm{sg}$: stop-gradient，只让良性项作目标、不回传梯度
- 取绝对值 $|\cdot|$: 鼓励两者范数相等而非有向差

#### 公式7: 总训练目标

$$
\mathcal{L}_{\mathrm{total}}=(1-\alpha-\beta)\mathcal{L}_{\mathrm{FM}}+\alpha\,\mathcal{L}_{\mathrm{BD}}+\beta\,\mathcal{L}_{\mathrm{mimic}}
$$

**含义**: 加权融合标准流匹配损失（保性能）、劫持损失（植后门）、模仿正则（保隐蔽）。

**符号说明**:
- $\mathcal{L}_{\mathrm{FM}}$ 作用于干净数据，$\mathcal{L}_{\mathrm{BD}}/\mathcal{L}_{\mathrm{mimic}}$ 作用于投毒数据
- 取 $\alpha=0.05$、$\beta=0.05$（grid search），$\tau_0=0.4$（系统分析最优）

#### 公式8: [[BadVLA]] 基线损失（附录，用于对比）

$$
\mathcal{L}_{\text{badvla}}=\text{CosineSimilarity}\bigl(\text{VLM}(o),\,\text{VLM}(o^{+})\bigr)
$$

**含义**: BadVLA 在 VLM 特征空间最大化触发/干净样本的分离。作者据此论证：当触发器语义接近良性样本（如倒杯 vs 正杯）时，这一目标与 VLM 的泛化目标冲突，导致优化不可解、攻击失败。

---

## 关键图表

<!-- 图片均使用 arXiv HTML 在线链接，未本地化 -->

### Figure 1: Two Action Representations in VLA / 两类动作表示

![Figure 1](https://arxiv.org/html/2604.09651v1/x1.png)

**说明**: 对比 VLA 的两类动作表示——token-based（自回归离散化，如 RT-1/2、OpenVLA）与 flow-matching-based（如 π0，直接建模时间相关动力学）。这张图奠定了"为何 token 级后门无法迁移到连续动力学"的论证基础。

### Figure 2: FlowHijack Attack Framework / 攻击框架总览

![Figure 2](https://arxiv.org/html/2604.09651v1/x2.png)

**说明**: FlowHijack 整体框架。上半部分为流匹配 VLA 处理多模态输入 $o_t$ 生成动作块 $A_t$ 的正常流程；下半部分展示三个组件——上下文感知触发器、动力学劫持（含 PL/IP 两种恶意动作）、损失设计——如何在 $\tau$-条件窗口内污染向量场。是理解整套方法的主图。

### Figure 3: Context-Aware Triggers and Actions / 触发器与动作可视化

![Figure 3](https://arxiv.org/html/2604.09651v1/x3.png)

**说明**: 在仿真（LIBERO）与真实（Franka）环境下的上下文触发器与动作可视化。上排为良性任务执行，下排为后门激活（触发器以红色高亮）。直观展示触发器的隐蔽性与激活后的行为偏移。

### Figure 4: Behavioral Stealth (Feature Distributions) / 行为隐蔽性

![Figure 4](https://arxiv.org/html/2604.09651v1/x4.png)

**说明**: 生成向量场的低维特征分布。FlowHijack（含 $\mathcal{L}_{\mathrm{mimic}}$）使恶意动作与良性动作的特征分布**显著重叠**，达成运动学隐蔽；而朴素攻击呈现可检测的分离。这是"行为不可区分"主张的核心定性证据。

### Figure 5: Robustness Analysis of Trigger / 触发器鲁棒性

![Figure 5](https://arxiv.org/html/2604.09651v1/x5.png)

**说明**: 上下文触发器（倒杯，Libero_goal）的三维鲁棒性。左=尺寸（50%–200% 维持 ASR>95%，1% 极端尺寸跌到 21%，说明学到的是语义而非像素伪影）；中=位置（Bottom-Left/Center、Left-Center 均 ASR≥95%）；右=状态（仅"倒置"高 ASR，正立/侧躺 ASR≈10%，证明学到的是物体**状态**而非物体本身，假阳性低）。

### Figure 6: VLA Robustness to Textual Modifications / 视觉主导性探针

![Figure 6](https://arxiv.org/html/2604.09651v1/x6.png)

**说明**: 探测 VLA 对文本改动的鲁棒性。上=模型忽略附加文本；下=即便省略文本，模型仍能从视觉推断任务，证明其**视觉主导**策略。这支撑了"为何采用视觉触发器而非文本触发器"的设计选择。

### Figure 7: End-Effector Velocity Profiles / 末端速度剖面（附录 D）

![Figure 7](https://arxiv.org/html/2604.09651v1/x7.png)

**说明**: 末端执行器速度剖面对比。无 $\mathcal{L}_{\mathrm{mimic}}$ 时攻击（红）产生异常高频速度波动；本文方法（橙）抑制异常、得到与正常动作（蓝）不可区分的平滑轨迹。这是动力学模仿正则有效性的**定量**证据。

### Figure 8: Real-World Evaluation / 真实世界评估（附录 G）

![Figure 8](https://arxiv.org/html/2604.09651v1/x8.png)

**说明**: Franka Emika Panda 真机上两个场景的评估。上=桌面操作 + 场景语义触发器；下=厨房操作 + 物体状态触发器。验证攻击在物理环境的可行性。

### Figure 9: Trigger Robustness Visualization / 触发器鲁棒性可视化（附录 H，三面板）

![Figure 9a](https://arxiv.org/html/2604.09651v1/x9.png)
![Figure 9b](https://arxiv.org/html/2604.09651v1/x10.png)
![Figure 9c](https://arxiv.org/html/2604.09651v1/x11.png)

**说明**: 正文三组鲁棒性实验（尺寸/位置/状态）的视觉样例与对应 SR、ASR 指标，是 Figure 5 的可视化补充。

### Table 1: Main Comparative Results / 主对比结果（vs BadVLA，按触发器类型分组，单位 %）

| Trigger Type | Method | L10 SR | L10 ASR | Goal SR | Goal ASR | Object SR | Object ASR | Spatial SR | Spatial ASR |
|---|---|---|---|---|---|---|---|---|---|
| — | Baseline (π0_libero) | 85.2 | - | 95.8 | - | 98.8 | - | 96.8 | - |
| White Pixel | BadVLA | 72.9 | 95.0 | 95.3 | **100** | 96.2 | **100** | 96.1 | **100** |
| White Pixel | Ours(PL) | 82.0 (-3.2) | **100** | 92.4 (-3.4) | 96.7 | 97.8 (-1.0) | **100** | 95.6 (-1.2) | 88.9 |
| White Pixel | Ours(IP) | 79.8 (-5.4) | 86.7 | 95.6 (-0.2) | 93.1 | 97.6 (-1.2) | 95.5 | 96.3 (-0.5) | 91.1 |
| Object State | BadVLA | 74.7 | 62.2 | 94.4 | 11.2 | 96.7 | 68.9 | 95.1 | 13.4 |
| Object State | Ours(PL) | 82.2 (-3.0) | 57.8 | 94.0 (-1.8) | **100** | 98.0 (-0.8) | 71.1 | 95.8 (-1.0) | 73.3 |
| Object State | **Ours(IP)** | 82.8 (-2.4) | 64.4 | **97.8 (+2.0)** | **100** | **98.8 (±0.0)** | **73.1** | 96.0 (-0.8) | **91.1** |
| Scene Semantic | BadVLA | 69.6 | 67.1 | 94.5 | 11.7 | 97.1 | 71.1 | 95.3 | 15.3 |
| Scene Semantic | Ours(PL) | 79.1 (-6.1) | 88.9 | 94.4 (-1.4) | 97.8 | 97.3 (-1.5) | 68.9 | 96.0 (-0.8) | **100** |
| Scene Semantic | **Ours(IP)** | 81.8 (-3.4) | 88.9 | 94.9 (-0.9) | **100** | 96.4 (-2.4) | 66.7 | 96.7 (-0.1) | **100** |

**说明**: 关键发现——白像素这类非语义触发器各框架都能学会（BadVLA ASR 95–100%）；但换成**隐蔽的上下文感知触发器**时，BadVLA **灾难性失败**（Goal 仅 11.2%、Spatial 仅 13.4%），而 FlowHijack 仍能学会复杂触发器（Goal 双触发器 100%、Spatial Scene Semantic 100%），且 SR 几乎不降、Ours(IP) 在 Goal 上还 +2.0%。证明操纵动力学空间是植入隐蔽上下文后门的独到有效途径。

### Table 2: Ablation of Loss Components / 损失组件消融（%）

| Method | L10 SR | L10 ASR | Goal SR | Goal ASR | Object SR | Object ASR | Spatial SR | Spatial ASR |
|---|---|---|---|---|---|---|---|---|
| Baseline | 85.2 | - | 95.8 | - | 98.8 | - | 96.8 | - |
| (- $\mathcal{L}_{\mathrm{FM}}$) | 0.0 | 100 | 0.0 | 100 | 0.0 | 100 | 0.0 | 100 |
| (- $\mathcal{L}_{\mathrm{BD}}$) | 84.4 | 0.0 | 96.0 | 0.0 | 97.6 | 0.0 | 97.1 | 0.0 |
| (- $\mathcal{L}_{\mathrm{mimic}}$) | 83.1 | 66.7 | 95.6 | 100 | 95.3 | 73.3 | 96.7 | 100 |
| **(+ ALL)** | **82.8** | **64.4** | **97.8** | **100** | **98.8** | **73.1** | **96.0** | **100** |

**说明**: 去 $\mathcal{L}_{\mathrm{FM}}$ → ASR 100% 但 SR 崩到 0%（灾难性遗忘），说明它对**可用性**必不可少；去 $\mathcal{L}_{\mathrm{BD}}$ → ASR 全 0%，证明它是攻击本体（且此配置含 $\mathcal{L}_{\mathrm{mimic}}$ 时 SR 略高于基线，暗示模仿项也起良性正则作用）；去 $\mathcal{L}_{\mathrm{mimic}}$ → SR/ASR 都高但**缺乏行为隐蔽**（运动学异常、向量场可分）。全量(+ALL)取得性能、攻击、隐蔽的最佳平衡。

### Table 3: Defense — Target Position Filtering / 目标位置过滤防御（Libero_goal，%）

| Threshold | Ours(PL) SR | Ours(PL) ASR | Ours(IP) SR | Ours(IP) ASR |
|---|---|---|---|---|
| Base | 94.0 | 100 | 97.8 | 100 |
| 1.0 m | 94.2 | 82.2 | 97.3 | 91.1 |
| 0.5 m | 93.6 | 51.1 | 96.9 | 86.7 |
| 0.1 m | 93.6 | **17.8** | 97.1 | **82.2** |

**说明**: 过滤"末端落点与目标欧氏距离超阈值"的轨迹。对 **Pose-Locking 高效**（阈值收紧到 0.1m 时 ASR 从 100% 跌到 17.8%，因其落点固定且远离目标）；但对更隐蔽的 **Initial-Perturbation 几乎无效**（0.1m 仍保 82.2% ASR），因为 IP 只引入小幅偏差、落点变化不够大、难被过滤。

### Table 4: Defense — Downstream Clean Fine-tuning / 下游干净微调防御（LoRA，%）

| LoRA Steps | Goal SR | Goal ASR | Spatial SR | Spatial ASR |
|---|---|---|---|---|
| Base | 94.0 | 100 | 95.8 | 73.3 |
| 1k | 94.3 | 100 | 95.6 | 72.8 |
| 3k | 94.3 | 86.7 | 95.8 | 64.7 |
| 5k | 94.5 | 69.8 | 96.2 | 58.2 |
| 10k | 94.7 | **67.7** | 96.1 | **55.6** |

**说明**: 用干净数据微调试图"遗忘"后门。ASR 虽逐步下降但**顽固偏高**——即便 10k 步后 Goal 仍 67.7%、Spatial 仍 55.6%，而良性 SR 完全保持。作者推测后门编码在权重的特定低秩子空间中，不被干净数据流形上的标准微调显著改动。

### Table 5: Ablation of $\tau$-Conditioned Injection Window / 注入窗口消融（Libero_goal，Ours(IP)，%）

| $\tau_0$ | SR(w/o) | ASR |
|---|---|---|
| 0.1 | 95.4 | 23.9 |
| 0.2 | 95.3 | 57.4 |
| **0.4** | **97.8** | **100** |
| 0.6 | 97.5 | 100 |
| 0.8 | 96.6 | 100 |

**说明**: 注入窗口上界 $\tau_0$ 的影响。太小（0.1/0.2）注入不足、ASR 偏低；$\tau_0=0.4$ 起 ASR 达 100%，且此时 SR 最高（97.8%），故选 $\tau_0=0.4$——既最大化攻击效力又最好保住良性性能；再增大 $\tau_0$ 反而略降 SR（污染过多向量场）。

---

## 实验

### 数据集 / 基准

| 基准 | 规模 | 特点 | 用途 |
|------|------|------|------|
| [[LIBERO]] | 40 任务，4 套件（LIBERO-10 / Goal / Object / Spatial） | 单臂操作，VLA/具身领域最常用 | 训练+评测 |
| 真实世界（Franka Emika Panda） | 桌面操作 + 厨房操作 两场景 | 真机验证，含场景语义/物体状态触发器 | 真实评测 |

### 实现细节

- **目标模型**: 开源 [[π0]]（标准流匹配动作生成模块）。
- **基线**: [[BadVLA]]（原针对离散动作 OpenVLA，作者将其两阶段无目标目标适配到 π0），其学习率与训练步数保持与原 BadVLA 一致以公平对比。
- **触发器对比**: 两类上下文触发器（Object State / Scene Semantic）vs BadVLA 的白像素补丁触发器。
- **攻击目标对比**: Pose-Locking (PL) / Initial-Perturbation (IP) vs BadVLA 无目标攻击。
- **指标**: **SR(w/o)**（无触发器时的干净任务成功率，越高越隐蔽）与 **ASR**（触发器存在时任务失败比例，越高攻击越强）。
- **微调**: 全部用 [[LoRA]] 参数高效微调，30,000 步，batch=8，cosine 调度，峰值学习率 $2.5\times10^{-5}$；VLM backbone LoRA rank=16/alpha=16，flow-matching head rank=32/alpha=32。
- **超参**: $\tau_0=0.4$、$\alpha=0.05$、$\beta=0.05$。
- **硬件**: NVIDIA RTX 4090 (24GB) GPU。

### 关键实验结论

- **主结果（Table 1）**: 上下文触发器下 BadVLA 崩盘（ASR 低至 11.2%），FlowHijack 稳定学会复杂触发器（ASR 最高 100%），SR 平均降幅通常 <3.5%。
- **消融（Table 2）**: 三项损失各司其职——$\mathcal{L}_{\mathrm{FM}}$ 保可用、$\mathcal{L}_{\mathrm{BD}}$ 是攻击本体、$\mathcal{L}_{\mathrm{mimic}}$ 保行为隐蔽。
- **鲁棒性（Fig 5）**: 对触发器尺寸（50–200%）、位置鲁棒；对物体**状态高度特异**（仅倒置激活），假阳性低。
- **防御（Table 3/4）**: 目标位置过滤只对 PL 有效、对 IP 失效；下游干净微调难以清除后门（10k 步后仍 55–68% ASR）。
- **隐蔽性（Fig 4/7）**: 特征分布重叠 + 速度剖面平滑，恶意动作运动学上不可区分。

---

## 批判性思考

### 优点

1. **填补真实空白**: 首个针对流匹配 VLA 向量场动力学的后门攻击，问题定义清晰、攻击面新颖，对当前最受关注的 [[π0]] 类连续策略具有现实警示意义。
2. **"早期注入、全路径放大"机制精巧**: $\tau$-条件注入恰好利用了流匹配偏向小 $\tau$ 采样的训练习惯，既高效又对静态分析隐蔽；$\tau_0$ 消融（Table 5）给出了清晰的 sweet spot。
3. **隐蔽性论证扎实**: 不止喊"隐蔽"，而是从触发器（上下文感知、状态特异性 Fig 5）、行为（特征重叠 Fig 4 + 速度剖面 Fig 7）、防御绕过（Table 3/4）三层给出证据。
4. **对 BadVLA 失败的解释有洞见**: 指出 VLM 特征分离目标与泛化目标冲突，这一分析转化为"应攻击下游动力学而非 VLM 特征空间"的设计原则，具方法论价值。

### 局限性

1. **仅单一目标模型与基准**: 攻击只在 π0 + LIBERO 上验证，未覆盖其他流匹配 VLA（如 GR00T、扩散策略变体）或更多基准，普适性待证。
2. **白盒微调投毒假设较强**: 需要完整白盒访问与发布伪装模型的能力；虽然作者论证了开源生态下的现实性，但实际"诱骗下游开发者采用投毒模型"的链路仍较长。
3. **真实世界评测偏定性**: 真机部分（Fig 8、附录 G）以场景案例为主，缺少与仿真同级别的大样本 SR/ASR 统计表，真实成功率的可信区间不明。
4. **防御探索偏窄**: 只评了目标位置过滤与干净微调两类，未涉及面向"生成动力学"的检测（如向量场异常探针、$\tau$-谱分析），而这恰是论文自己呼吁的方向——给出的更多是"现有防御失效"而非"如何防"。

### 潜在改进方向

1. 将攻击与防御扩展到更多流匹配/扩散 VLA 与更多基准（含双臂、长程任务），量化跨模型可迁移性。
2. 针对 $\tau$-条件注入设计专门防御：例如对不同 $\tau$ 切片的向量场做一致性/异常检测，或在 $\tau\in[0,\tau_0]$ 窗口加入鲁棒训练。
3. 补充真实世界的大样本定量评测与人类可检测性用户研究，验证"上下文触发器人眼难察"的强主张。
4. 探索后门是否真编码在低秩子空间（作者的假设），用权重子空间分析给出可剪除/可检测的结构性证据。

### 可复现性评估

- [ ] 代码开源（论文未提供项目主页/代码链接）
- [ ] 预训练/投毒模型（未声明 release）
- [x] 训练细节较完整（附录 E 给出 LoRA 配置、步数、学习率、$\tau_0/\alpha/\beta$）
- [x] 数据集可获取（LIBERO 公开；目标模型 π0 开源）

---

## 速查卡片

> [!summary] FlowHijack: 针对流匹配 VLA 的动力学感知后门攻击
> - **核心**: 首个攻击流匹配 VLA 向量场动力学的后门框架；用 $\tau$-条件注入（仅 $\tau\in[0,\tau_0]$ 早期污染）+ 动力学模仿正则（匹配向量场范数）实现隐蔽攻击。
> - **方法**: 上下文感知触发器（倒杯/盆栽）+ 向量场劫持损失 $\mathcal{L}_{\mathrm{BD}}$ + 模仿正则 $\mathcal{L}_{\mathrm{mimic}}$；恶意动作两策略 Pose-Locking / Initial-Perturbation；总损失加权 $\alpha=\beta=0.05$、$\tau_0=0.4$；LoRA 微调 π0。
> - **结果**: 上下文触发器下 ASR 最高 100%（BadVLA 仅 11–15%），SR 降幅通常 <3.5%；绕过目标位置过滤（IP 仍 82% ASR）与干净微调（10k 步仍 55–68% ASR）；运动学不可区分。
> - **代码**: 未公开。目标模型 π0、基准 LIBERO 均公开。

---

*笔记创建时间: 2026-06-29*
