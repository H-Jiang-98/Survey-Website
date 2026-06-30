---
title: "Cross-Hand Latent Representation for Vision-Language-Action Models"
method_name: "XL-VLA"
authors: [Guangqi Jiang, Yutong Liang, Jianglong Ye, Jia-Yang Huang, Changwei Jing, Rocky Duan, Pieter Abbeel, Xiaolong Wang, Xueyan Zou]
year: 2026
venue: CVPR
tags: [VLA, cross-embodiment, dexterous-manipulation, latent-action-space, autoencoder, retargeting]
zotero_collection: _inbox
image_source: online
arxiv_html: https://arxiv.org/html/2603.10158v1
created: 2026-06-29
---

# Cross-Hand Latent Representation for Vision-Language-Action Models

## 元信息

| 项目 | 内容 |
|------|------|
| 作者 | Guangqi Jiang, Yutong Liang, Jianglong Ye, Jia-Yang Huang, Changwei Jing, Rocky Duan, Pieter Abbeel, Xiaolong Wang, Xueyan Zou |
| 机构 | UC San Diego / UC Berkeley 等（Pieter Abbeel、Xiaolong Wang 团队） |
| 会议 | CVPR 2026 |
| 类别 | VLA（Vision-Language-Action）/ 跨本体灵巧操作 |
| 日期 | 2026-03（arXiv v1） |
| 项目主页 | （未给出） |
| 链接 | [arXiv](https://arxiv.org/abs/2603.10158) / [PDF](https://arxiv.org/pdf/2603.10158) |

---

## 一句话总结

> 用一个跨灵巧手共享的「隐式动作空间」替换 $\pi_0$ 的 state token，使单一 VLA 策略能无缝训练并控制结构各异的多种灵巧手，并对未见手-任务组合零样本泛化。

---

## 核心贡献

1. **大规模跨手灵巧操作数据集**: 采集覆盖 **10 个操作任务、4 款全新灵巧手**（Ability、Paxini DexH13、X-Hand1、Inspire）的遥操作数据，共 **2M 个 state-action 对**（每任务每手 50 条示范、共 2000 条）。
2. **无监督跨本体隐空间自编码器**: 提出一套 hand-specific 的 [[VAE]]-式编码/解码器，把各手的关节空间映射到**同一个共享隐式动作分布**；训练**完全自监督**（仅随机采样关节构型 + 可微前向运动学），无需任何配对的跨手轨迹。
3. **XL-VLA 框架**: 把该 [[Latent Action Space|隐式动作空间]]作为「即插即用」token 接入标准 VLA（基于 [[π0|$\pi_0$]]），跨本体训练性能显著超过在原始关节空间训练的 VLA 基线，并能**零样本迁移**到未训练过的手-任务组合。

---

## 问题背景

### 要解决的问题
如何为「灵巧手家族」构建一个**统一的动作表征**，使单一 [[VLA]] 策略能跨多种形态差异巨大的灵巧手联合训练；以及当一款动作空间与已有手不同的新手出现时，**如何无缝接入**而不必逐手重训。

### 现有方法的局限
- 语言有相对稳定通用的词表，但**机器人动作空间天然绑定形态（morphology）**：灵巧手的关节维度、驱动方式、运动学差异巨大，且随新硬件快速演化。
- 为每款新本体单独采集大规模示范**昂贵且不可持续**，难以规模化跨本体学习。
- 过去 [[Cross-Embodiment|跨本体]]与 VLA 工作多聚焦「机械臂 + 平行夹爪」，对**更复杂、更有能力的灵巧操作**关注不足；且多停留在仿真，缺乏真机跨本体验证。
- 现有隐空间/重定向方法（见 Table 1）大多缺少「视觉+语言+本体」全模态输入、缺少跨本体 decoder、或不支持零样本迁移；[[LAD|Latent Action Diffusion]] 等需要**监督/配对数据**。

### 本文的动机
作者主张：把「与本体无关」的控制信息抽到一个**共享隐空间** $\mathbf{z}$，让 VLA 主干（hand-agnostic）只在隐空间里推理、由 hand-specific 解码器还原到各手关节空间，就能在保留 VLM 预训练先验的同时打通形态鸿沟。关键洞察是——只要隐空间通过**重建 + 指尖几何对齐 + 平滑先验**三项约束塑形，同一个隐码就能在不同手上解出几何一致的抓捏行为，从而支持跨本体训练与零样本复用。

---

## 方法详解

### 模型架构

XL-VLA 采用 **感知→隐式动作→解码** 的两段式流水线（见 Figure 2），由两部分组成：
- **输入**: 单视角 RGB 图像观测 $\mathbf{V}$ + 语言指令 $\mathbf{T}$ + 上一动作块（隐式 token 形式）
- **Backbone**: 沿用 [[π0|$\pi_0$]] —— 由 [[PaliGemma]] 初始化的视觉/语言编码器 + 动作专家（action expert）
- **核心模块**: 跨灵巧手共享的 [[Latent Action Space|隐式动作空间]]（hand-specific 编码器 $E_h$ / 解码器 $D_h$）
- **输出**: 下一隐式动作块 $\widehat{\mathbf{z}}_{t+1}$，再由 $D_h$ 解码为关节命令块 $\widehat{\mathbf{q}}_{t+1}^{(h)}$
- **关键改造**: 把 $\pi_0$ 原本的 **state token 替换为 latent action token**；VLA 微调阶段**冻结所有隐空间编/解码器**，只微调动作专家。

整体策略映射（在原始关节空间表述）写作：

$$
\mathbf{q}_{t+1}^{(h)} = F\!\left(\mathbf{q}_{t}^{(h)}, \mathbf{V}, \mathbf{T}\right)
$$

其中动作块 $\mathbf{q}_{t}^{(h)} \in \mathbb{R}^{64\times d_h}$ 是一段 64 帧、20 Hz（约 3.2 s）的关节位置序列；$d_h$ 为手 $h$ 的驱动关节数；序列模型 $F$ 本身 hand-agnostic，手身份 $h$ 仅用来选择对应的 $E_h/D_h$。

### 核心模块

#### 模块1: VLA Backbone（基于 $\pi_0$ 的视觉-语言-动作主干）

**设计动机**: 复用 VLM 的 web 级先验做语言条件的灵巧操作，同时让动作以「连续隐式块」而非离散 token 输出，避免 tokenize 动作对高频灵巧控制的拖累。

**具体实现**:
- 由 [[PaliGemma]] 初始化的 $\pi_0$ 视觉+语言编码器编码 $(\mathbf{V}, \mathbf{T})$。
- 用一段隐式动作 token 的历史替换 $\pi_0$ 原始 state token；动作专家以此 + 视觉/语言 token 为条件，回归下一隐式动作块 $\widehat{\mathbf{z}}_{t+1}$。
- 手身份 $h$ **从不**作为显式 token 喂给主干，保证主干完全 hand-agnostic。

#### 模块2: Cross-Hand Latent Space（跨手隐式动作空间）

**设计动机**: 为整族灵巧手提供一个统一、隐式、与本体无关的动作空间，使下游策略可无缝控制不同手；该隐空间**独立于 VLA 预训练**。

**具体实现**（见 Figure 3）:
- 多头 [[VAE]]-式自编码器：每款手 $h\in\mathcal{H}$ 配一个轻量 MLP 编码器 $E_h$ 与解码器 $D_h$。
- 编码器输出高斯后验参数 $(\boldsymbol{\mu}^{(h)},\boldsymbol{\sigma}^{(h)})$，用**重参数化技巧**采样隐码 $\mathbf{z}$；解码器把 $\mathbf{z}$ 还原回该手关节空间 $\hat{\mathbf{q}}^{(h)}$。
- 三项约束塑形隐空间：$L_1$ 重建、$L_2$ 跨手指尖几何重定向（经[[Forward Kinematics|可微前向运动学]]）、$L_3$ 隐空间高斯先验正则。
- 最终配置：架构 $H_{64}^{128\rightarrow64}$、隐维 $L_d=32$（见 Table 5 消融）。

#### 模块3: 自监督训练协议（无需示范/配对数据）

**设计动机**: 让隐空间对齐**完全自监督**，免去昂贵的配对跨手轨迹。

**具体实现**:
- 训练隐自编码器**不使用任何示范或 IK 生成轨迹**；而是对每款手 $s$ 在硬件关节限位内**随机采样**关节构型 $\mathbf{q}^{(s)}$。
- 把 $\mathbf{q}^{(s)}$ 编码为 $\mathbf{z}$，再用**所有**解码器 $\{D_t\}_{t\in\mathcal{H}}$ 解码：自解码 $D_s(\mathbf{z})$ 计入 $L_1$，跨手解码 $D_t(\mathbf{z})\,(t\neq s)$ 计入 $L_2$。
- 所有手的损失聚合后**单次反传**，联合优化全部编/解码器。因 $L_2$ 只用各手前向运动学与解码姿态，对齐过程**无需任何配对跨手轨迹**。

### 关键公式与机制

#### 公式1: [[VLA]] 隐式编码

$$
\mathbf{z}_{t} = E_{h}\!\left(\mathbf{q}_{t}^{(h)}\right)
$$

**含义**: hand-specific 编码器把上一动作块映射为紧凑隐码，作为 VLA 的 latent action token。

**符号说明**:
- $\mathbf{q}_{t}^{(h)}\in\mathbb{R}^{64\times d_h}$: 手 $h$ 上一动作块（64 帧 @ 20 Hz）
- $\mathbf{z}_t$: 共享隐式动作空间中的隐码

#### 公式2: 隐式解码（推理）

$$
\widehat{\mathbf{q}}_{t+1}^{(h)} = D_{h}\!\left(\widehat{\mathbf{z}}_{t+1}\right)
$$

**含义**: VLA 预测的下一隐式块经 hand-specific 解码器还原为该手的关节命令块。

**符号说明**:
- $\widehat{\mathbf{z}}_{t+1}$: 动作专家预测的下一隐式动作块
- $D_h$: 手 $h$ 的解码 MLP（VLA 微调时冻结）

#### 公式3: 高斯后验与重参数化

$$
q(\mathbf{z}\mid\mathbf{q}^{(h)}) = \mathcal{N}\!\left(\boldsymbol{\mu}^{(h)},\ \mathrm{diag}\big((\boldsymbol{\sigma}^{(h)})^{2}\big)\right)
$$

**含义**: 编码器对每个关节构型给出对角高斯后验，从中采样隐码，使隐空间连续可插值。

**符号说明**:
- $(\boldsymbol{\mu}^{(h)},\boldsymbol{\sigma}^{(h)})=E_h(\mathbf{q}^{(h)})$: 编码器输出的后验均值/标准差
- $\mathrm{diag}(\cdot)$: 对角协方差

#### 公式4: 重建损失 $L_1$

$$
L_{1} = \mathcal{L}_{\mathrm{rec}} = \frac{1}{|\mathcal{H}|}\sum_{h\in\mathcal{H}} \mathrm{MSE}\!\left(\hat{\mathbf{q}}^{(h)},\ \mathbf{q}^{(h)}\right)
$$

**含义**: 保证每款手的编/解码器作为自编码器忠实重建该手关节构型，使隐空间**不损失任一本体的运动学**。

**符号说明**:
- $\mathcal{H}$: 灵巧手集合；$|\mathcal{H}|$: 手的数量
- $\hat{\mathbf{q}}^{(h)} = D_h(\mathbf{z})$: 自解码重建；$\mathbf{q}^{(h)}$: 真值关节构型

#### 公式5: 跨手重定向损失 $L_2$（指尖几何对齐）

$$
\begin{aligned}
L_{2} = \frac{1}{|\mathcal{H}|(|\mathcal{H}|-1)|\mathcal{P}|}\sum_{s\neq t}\sum_{(i,j)\in\mathcal{P}} w_{ij}^{(s)}\Big[ &\lambda_{\mathrm{dis}}\big(\|\boldsymbol{\delta}_{ij}^{(s)}\|_{2}-\|\hat{\boldsymbol{\delta}}_{ij}^{(t)}\|_{2}\big)^{2} \\
&+ \lambda_{\mathrm{dir}}\big(1-c_{ij}^{(s,t)}\big)\Big]
\end{aligned}
$$

**含义**: 让**同一隐码**在不同手上解出几何一致的「捏合」行为：既对齐拇指-指尖**间距**（distance 项），又对齐**方向**（direction 项），从而把各手锚定到同一隐式动作语义。

**符号说明**:
- $\boldsymbol{\delta}^{(h)}_{ij}=\mathbf{p}^{(h)}_{i}-\mathbf{p}^{(h)}_{j}$: 手 $h$ 由[[Forward Kinematics|可微前向运动学]]算得的指尖对位移；$\hat{\boldsymbol{\delta}}^{(t)}_{ij}$ 取自目标手 $t$ 的解码构型
- $\mathcal{P}$: 拇指-四指对集合（thumb–index/middle/ring/little；Paxini 无小指则丢弃相关对）
- $c_{ij}^{(s,t)}$: 源/目标 pinch 方向的夹角余弦
- $w_{ij}^{(s)}=\exp\!\big(-\lambda_{\mathrm{dis}}^{\mathrm{exp}}\|\boldsymbol{\delta}_{ij}^{(s)}\|_2\big)$: 越紧的捏合权重越大
- $s,t$: 源手 / 目标手

#### 公式6: 隐空间 KL 正则 $L_3$

$$
L_{3} = \mathcal{L}_{\mathrm{KL}} = \mathbb{E}_{\mathbf{q}}\Big[\mathrm{KL}\big(q(\mathbf{z}\mid\mathbf{q})\,\|\,\mathcal{N}(\mathbf{0},\mathbf{I})\big)\Big]
$$

**含义**: 把共享隐空间正则到标准正态，使其平滑、利于采样与跨本体插值。

**符号说明**:
- $\mathrm{KL}(\cdot\|\cdot)$: KL 散度；$\mathcal{N}(\mathbf{0},\mathbf{I})$: 标准高斯先验

#### 公式7: 总隐空间目标

$$
L_{\mathrm{latent}} = L_{1} + L_{2} + \beta L_{3}
$$

**含义**: 重建 + 跨手重定向 + KL 正则的总目标，联合优化所有手的编/解码器。

**符号说明**: 固定权重 $\beta=10^{-5}$、$\lambda_{\mathrm{dis}}=2000.0$、$\lambda_{\mathrm{dir}}=5.0$、$\lambda_{\mathrm{dis}}^{\mathrm{exp}}=12.0$。

---

## 关键图表

<!-- 图片均使用 arXiv HTML 在线链接，未本地化 -->

### Figure 1: Overview / 概览（teaser）

![Figure 1](https://arxiv.org/html/2603.10158v1/x1.png)

**说明**: XL-VLA 把单一隐式动作直接解码到多款灵巧手（Ability、Paxini DexH13、X-Hand1、Inspire）做语言引导的操作；右图展示实验设置（采集物体与各 DexHand）。直观传达「一个隐码 → 多本体」的核心卖点。

### Figure 2: Model Pipeline / 模型流水线

![Figure 2](https://arxiv.org/html/2603.10158v1/imgs/method.png)

**说明**: XL-VLA 在 [[π0|$\pi_0$]] 基础上，用视觉/语言编码器 + 动作专家在**共享隐式动作空间**里做跨本体控制；VLA 训练时动作专家被微调，而预训练的隐式编/解码器**保持冻结**。这是理解「主干 hand-agnostic、解码 hand-specific」结构的核心图。

### Figure 3: Latent Space Pretraining Pipeline / 隐空间预训练流程

![Figure 3](https://arxiv.org/html/2603.10158v1/x2.png)

**说明**: 每款手的关节位置 $\mathbf{q}_h$ 经编码 MLP 映入共享隐空间，再由解码 MLP 重建；图中标注了重建损失 $L_1$、经可微前向运动学的重定向损失 $L_2$、隐空间正则损失 $L_3$ 三者的作用位置。是公式 4–7 的可视化对应。

### Figure 4: Zero-shot Unseen Tasks Generalization / 未见任务零样本泛化

![Figure 4](https://arxiv.org/html/2603.10158v1/x3.png)

**说明**: 对每款手随机留出若干任务作为「未见任务」从训练集剔除，再用其余数据训练的模型直接测试。结果显示：用对齐隐式动作空间训练后，XL-VLA 能零样本泛化到新的「手-任务」组合，且在所有本体/任务上从不低于 $\pi_0$+RT（基于运动学重定向）的基线，尤其在 HB、RB 等精细任务上优势明显。PSR=部分成功率（单臂完成给半分）。

### Figure 5: G1 Cross-Robot Performance / G1 跨机器人性能

![Figure 5](https://arxiv.org/html/2603.10158v1/x4.png)

**说明**: 把桌面 xArm 与人形 G1 的数据共训。用对齐隐式动作空间共训**优于**在原始（变长 state/action）动作空间共训，证明隐空间收益不仅跨手、也跨机器人系统。对应数值见 Table 6。

### Figure 6: Latent Visualizations / 隐式解码可视化（X-Hand & Inspire）

![X-Hand](https://arxiv.org/html/2603.10158v1/imgs/demo_transparent_1.png)
![Inspire Hand](https://arxiv.org/html/2603.10158v1/imgs/demo_transparent_2.png)

**说明**: 同一隐码在不同手上解码：一只手全不透明、其余半透明，蓝色标记目标抓取点。尽管运动学各异，各手从同一隐码解出一致姿态，佐证隐空间捕捉到「与本体无关」的控制语义。

### Figure 7: More Latent Visualizations / 更多隐式解码可视化（四手，附录）

![X-Hand](https://arxiv.org/html/2603.10158v1/imgs/demo_transparent_1.png)
![Inspire Hand](https://arxiv.org/html/2603.10158v1/imgs/demo_transparent_2.png)
![Paxini Hand](https://arxiv.org/html/2603.10158v1/imgs/demo_transparent_3.png)
![Ability Hand](https://arxiv.org/html/2603.10158v1/imgs/demo_transparent_4.png)

**说明**: 把同一隐式表征解码到全部四款手，进一步展示跨本体一致性。

### Figure 8: xArm Camera Setup / xArm 相机设置（附录）

![Figure 8](https://arxiv.org/html/2603.10158v1/imgs/camera_setup.jpeg)

**说明**: 使用单个前视 RealSense L515 相机作为策略输入；图中 D435 不用于 XL-VLA。

### Figure 9: Dexterous Hands / 四款灵巧手（附录）

![Figure 9](https://arxiv.org/html/2603.10158v1/imgs/DexHands.jpg)

**说明**: 实验所用 4 款手，形状、尺度、自由度、驱动关节数各异（对应 Table 3）。

### Figure 10: xArm Camera View / xArm 相机视角（附录）

![Figure 10](https://arxiv.org/html/2603.10158v1/imgs/rs_view.png)

**说明**: 相机实际所见，即 XL-VLA 及所有基线方法的输入视图。

### Figure 11: G1 Scene / G1 场景（附录）

![Figure 11](https://arxiv.org/html/2603.10158v1/imgs/g1_scene.jpeg)

**说明**: 在 G1 颈部附近装 L515 相机以获取第一人称（egocentric）视角。

### Figure 12: G1 Egocentric Camera View / G1 第一人称视角（附录）

![Figure 12](https://arxiv.org/html/2603.10158v1/imgs/g1_camera_view.png)

**说明**: G1 的第一人称相机视图。

### Figure 13: Objects / 物体集合（附录）

![Figure 13](https://arxiv.org/html/2603.10158v1/imgs/objects.jpg)

**说明**: 取自现有数据集的多样日常物体，尺度/形状/纹理/重量各异，要求操作策略鲁棒。

### Figure 14: Apple Vision Pro for Data Collection / 数据采集（附录）

![Figure 14](https://arxiv.org/html/2603.10158v1/imgs/avp.jpeg)

**说明**: 用 Apple Vision Pro 跟踪遥操作者手与手腕，再经重定向与逆运动学（IK）控制真机。

### Figure 15: G1 Teleoperation System / G1 遥操作系统（附录）

![Figure 15](https://arxiv.org/html/2603.10158v1/imgs/x5.jpeg)

**说明**: 基于 HOMIE 构建的 G1 上半身遥操作系统，用一对 MANUS Mocap 手套跟踪人手姿态。

### Figure 16: Task Visualizations / 10 个任务可视化（附录）

![Task 1](https://arxiv.org/html/2603.10158v1/imgs/task_vis/001.png)
![Task 2](https://arxiv.org/html/2603.10158v1/imgs/task_vis/002.png)
![Task 3](https://arxiv.org/html/2603.10158v1/imgs/task_vis/003.png)
![Task 4](https://arxiv.org/html/2603.10158v1/imgs/task_vis/004.png)
![Task 5](https://arxiv.org/html/2603.10158v1/imgs/task_vis/005.png)
![Task 6](https://arxiv.org/html/2603.10158v1/imgs/task_vis/006.png)
![Task 7](https://arxiv.org/html/2603.10158v1/imgs/task_vis/007.png)
![Task 8](https://arxiv.org/html/2603.10158v1/imgs/task_vis/008.png)
![Task 9](https://arxiv.org/html/2603.10158v1/imgs/task_vis/009.png)
![Task 10](https://arxiv.org/html/2603.10158v1/imgs/task_vis/010.png)

**说明**: 以 XHand 为例展示 10 个真实任务（PF/SC/SoC/HB/RL/PS/RB/PuS/PoS/PC），技能与难度各异；其中 4 个也在 G1 上测试。

### Figure 17: Latent Visualization of a Grasping Trajectory / 抓取轨迹隐式可视化（附录）

![Figure 17](https://arxiv.org/html/2603.10158v1/imgs/latent_traj.png)

**说明**: 一条连续抓取轨迹在全部机器人手上的解码渲染（X-Hand 高亮），展示隐式插值在时间维度的平滑性。

### Table 1: Related Work Summary / 相关工作对比

| Paper | Data | Deployment | EEF↔EEF | Vision | Lang | Prop | Decoder | ZS |
|-------|------|-----------|---------|--------|------|------|---------|----|
| UniVLA [8] | human video; teleop | 1 arm+1 gripper | gripper↔gripper | ✓ | ✓ | ✗ | ✓ | ✗ |
| ATE [71] | teleop; sim | 2 arms+2 grippers | — | ✓ | ✓ | ✗ | ✗ | ✗ |
| LAD [3] | teleop | 1 arm+1 hand/gripper | hand↔hand/gripper | ✓ | ✗ | ✗ | ✓ | ✗ |
| EgoBridge [45] | human video; teleop | 2 arms+2 pushers | — | ✓ | ✗ | ✓ | ✗ | ✗ |
| CoMo [65] | internet video; teleop | 1 arm+1 gripper | — | ✓ | ✗ | ✗ | ✗ | ✗ |
| Tenma [17] | teleop | 2 arms+2 grippers | — | ✓ | ✓ | ✓ | ✗ | ✗ |
| CycleVAE [16] | teleop | 1 arm+1 hand | hand↔hand | ✗ | ✗ | ✓ | ✓ | ✗ |
| CETransfer [54] | sim (sim→real) | 1 arm+1 gripper | gripper↔gripper | ✗ | ✗ | ✓ | ✓ | ✓ |
| **Ours (XL-VLA)** | **teleop** | **2 arm+2 hand** | **hand↔hand** | **✓** | **✓** | **✓** | **✓** | **✓** |

**说明**: 在所有对比维度上，唯独 XL-VLA 同时具备「视觉+语言+本体输入、双臂双手部署、手↔手 EEF 迁移、跨本体 decoder、零样本（ZS）」全勾，凸显其相对前作的覆盖完整性。

### Table 2: Vision-Language-Action Modeling / 跨本体多任务成功率

| Method | Hand | PF | SC | SoC | HB | RL | PS | RB | PuS | PoS | PC | Mean |
|--------|------|----|----|-----|----|----|----|----|-----|-----|----|------|
| $\pi_0$ [6] | Ability | 0.10 | 0.10 | 0.00 | 0.70 | 0.20 | 0.80 | 0.60 | 0.30 | 0.30 | 0.60 | 0.37 |
| $\pi_0$ | Inspire | 0.10 | 0.20 | 0.00 | 0.30 | 0.10 | 0.50 | 0.30 | 0.20 | 0.20 | 0.80 | 0.27 |
| $\pi_0$ | Paxini | 0.40 | 0.40 | 0.30 | 0.20 | 0.00 | 0.80 | 0.60 | 0.30 | 0.10 | 0.40 | 0.35 |
| $\pi_0$ | XHand | 0.20 | 0.40 | 0.00 | 0.40 | 0.10 | 0.60 | 0.30 | 0.20 | 0.30 | 0.40 | 0.29 |
| $\pi_0$ | **Mean** | 0.20 | 0.28 | 0.08 | 0.40 | 0.10 | 0.68 | 0.45 | 0.25 | 0.23 | 0.55 | 0.32 |
| XL-VLA | Ability | 0.80 | 0.80 | 0.40 | 1.00 | 0.70 | 1.00 | 0.70 | 0.30 | 0.90 | 0.70 | 0.73 |
| XL-VLA | Inspire | 0.60 | 0.50 | 0.50 | 0.80 | 0.40 | 0.80 | 1.00 | 0.40 | 0.80 | 1.00 | 0.68 |
| XL-VLA | Paxini | 0.80 | 0.70 | 0.80 | 1.00 | 0.30 | 1.00 | 1.00 | 0.40 | 0.80 | 1.00 | 0.78 |
| XL-VLA | XHand | 0.60 | 0.50 | 0.50 | 1.00 | 0.30 | 1.00 | 0.90 | 0.30 | 1.00 | 0.90 | 0.70 |
| **XL-VLA** | **Mean** | **0.70** | **0.63** | **0.55** | **0.95** | **0.43** | **0.95** | **0.90** | **0.35** | **0.88** | **0.90** | **0.72** |

**说明**: 任务缩写见下文「数据集」。同一份多手多任务数据下，$\pi_0$ 虽能靠改序列长度兼容不同手，但性能普遍低且不稳（手均 0.27–0.37）；XL-VLA 每手每任务一致提升，手均 0.68–0.78。逐任务均值看，XL-VLA 把整体从 0.32→0.72（正文另述全任务全手平均由 0.55→0.90，+0.35），在 SoC(+47%)、HB(+55%)、PoS(+65%) 等精细任务提升最大。XHand（结构最迥异）由 0.29→0.70，说明能跨越大形态鸿沟。

### Table 3: Dexterous Hand Comparison / 灵巧手对比

| | Ability | Inspire | X-Hand1 | Paxini DexH13 |
|--|--------|---------|---------|---------------|
| #Fingers | 5 | 5 | 5 | 4 |
| #DoF (mimic) | 12 (6) | 12 (6) | 12 | 16 (3) |

**说明**: 四款手在手指数、自由度与 mimic（耦合/欠驱动）关节上差异显著（Paxini 仅 4 指、16 DoF），正是跨本体难点的来源。

### Table 4: Latent Replay Comparison / 隐式重放对比（vs LAD）

| Model | Combination | PF | SC | SoC | HB | RL | PS | RB | PuS | PoS | PC | Mean |
|-------|-------------|----|----|-----|----|----|----|----|-----|-----|----|------|
| LAD [3] | Ability+Inspire | 0.8 | 0.4 | 0.5 | 0.4 | 0.6 | 0.6 | 0.6 | 0.5 | 0.7 | 0.9 | 0.60 |
| LAD [3] | Paxini+XHand | 0.7 | 0.5 | 0.6 | 0.7 | 0.5 | 0.8 | 0.7 | 0.4 | 0.6 | 0.6 | 0.61 |
| **XL-VLA** | Ability+Inspire | 0.9 | 0.7 | 0.8 | 0.7 | 0.8 | 0.7 | 0.8 | 0.8 | 1.0 | 1.0 | **0.82** |
| **XL-VLA** | Paxini+XHand | 0.8 | 0.7 | 0.9 | 0.8 | 0.6 | 0.9 | 0.9 | 0.6 | 1.0 | 0.9 | **0.81** |

**说明**: 把两款手的示范编码后解到另两款手上真机重放（不断触/不自碰即成功）。XL-VLA（0.82 / 0.81）显著超 [[LAD]]（0.60 / 0.61），且**无任何监督/配对标签**，在 SC、SoC、HB 等精细任务上 LAD 明显退化，证明无监督隐对齐捕捉本体不变结构更有效。

### Table 5: Ablations / 隐空间设计与损失消融（越低越好）

| Exp | Recon-Joint | Recon-Tip | PT$^{\text{dir}}$ | PT$^{\text{dist}}$ | RT$^{\text{dir}}$ | RT$^{\text{dist}}$ | Cont-Joint | Cont-Tip | Accel. | Jerk |
|-----|------|-----|------|------|------|------|------|------|------|------|
| **Ours** ($H_{64}^{128\to64}, L_{32}$) | 5.476 | 3.703 | 11.857 | 1.872 | 10.492 | 6.295 | 4.492 | 8.534 | 8.683 | 9.659 |
| $-L_1$ | 61.672 | 39.400 | 11.741 | 1.857 | 10.398 | 6.375 | 24.784 | 58.858 | 12.028 | 16.852 |
| $-L_2^{\text{dist}}$ | 5.195 | 3.580 | 3.972 | 4.413 | 6.788 | 24.488 | 4.073 | 8.168 | 8.525 | 9.240 |
| $-L_2^{\text{dir}}$ | 4.966 | 3.378 | 46.217 | 2.251 | 53.546 | 5.518 | 4.451 | 9.217 | 8.742 | 9.551 |
| $-L_2$ (both) | 3.781 | 2.602 | 62.733 | 8.080 | 71.765 | 62.809 | 2.823 | 6.757 | 8.602 | 9.426 |
| $H_{256}^{128}$ | 5.897 | 3.908 | 9.073 | 1.613 | 10.432 | 6.277 | 3.104 | 6.410 | 9.213 | 10.406 |
| $H_{128}^{64}\times2$ | 8.216 | 4.280 | 9.027 | 1.513 | 10.572 | 6.713 | 5.004 | 8.832 | 8.559 | 9.479 |
| $H_{64}^{64}$ | 4.979 | 3.411 | 9.010 | 1.655 | 10.702 | 6.985 | 2.922 | 6.298 | 8.618 | 9.296 |
| $H^{64}$ | 5.021 | 3.445 | 9.010 | 1.518 | 10.213 | 6.435 | 4.174 | 8.132 | 8.246 | 8.740 |
| $L_8$ | 20.913 | 6.499 | 9.217 | 1.557 | 10.960 | 6.805 | 8.164 | 11.720 | 8.758 | 9.778 |
| $L_{16}$ | 8.416 | 4.159 | 13.624 | 1.989 | 11.084 | 6.558 | 5.445 | 9.192 | 8.436 | 8.996 |
| $L_{64}$ | 5.542 | 3.583 | 8.314 | 1.549 | 10.995 | 6.955 | 4.140 | 8.174 | 8.299 | 8.944 |
| $L_{96}$ | 5.239 | 3.422 | 9.332 | 1.562 | 10.516 | 6.554 | 3.498 | 7.072 | 8.700 | 9.703 |
| $L_{128}$ | 5.324 | 3.543 | 8.736 | 1.529 | 10.286 | 6.215 | 3.282 | 6.882 | 8.607 | 9.294 |

**说明**: 指标含义——重建 Joint/Tip RMSE、跨手 pinch/random 方向(dir)与距离(dist)误差(PT/RT)、隐连续性 Joint/Tip、插值平滑度 Accel./Jerk。关键结论：去掉 $L_1$ 重建崩坏（61.7/39.4）；去掉 $L_2$ 的某一项会牺牲对应几何（去 dist 项 RT$^{\text{dist}}$ 飙至 24.5、去 dir 项 RT$^{\text{dir}}$ 飙至 53.5，两者全去更糟），说明方向/距离两项各司其职、缺一不可；性能在多种架构/隐维下稳定，仅当隐维过大（$L_{128}$）才退化，验证「适度紧凑隐空间」更利于本体不变结构。最终采用 $H_{64}^{128\to64}+L_{32}$。

### Table 6: G1 Policy Performances / G1 跨机器人成功率（对应 Figure 5）

| Method | PF | HB | PS | PoS | Mean |
|--------|----|----|----|-----|------|
| $\pi_0$ [6] | 0.4 | 0.6 | 0.5 | 0.6 | 0.525 |
| **XL-VLA** | **0.7** | **0.9** | **0.9** | **0.8** | **0.825** |

**说明**: xArm + G1 共训下，XL-VLA 在 4 个 G1 任务上平均 0.825，较 $\pi_0$ 的 0.525 提升约 +57%，说明对齐隐空间收益可跨「机械臂 ↔ 人形」机器人系统迁移。

---

## 实验

### 数据集 / 任务

10 个真实任务（每任务每手 50 条遥操作示范，共 2000 条；总计 2M state-action 对）：

| 缩写 | 任务 | 描述 |
|------|------|------|
| PF | Prepare Fruits | 把香蕉和橙子放到绿色案板上待切 |
| SC | Stack Cans | 把芝士罐叠到盐罐上 |
| SoC | Sort Cans | 把番茄罐与芝士罐放进容器 |
| HB | Hand over Bottle | 把白瓶从右手递到左手（双臂协作） |
| RL | Re-organize Lemons | 把黄柠檬与青柠放进碗里 |
| PS | Pour Sauce | 把芥末酱倒进肉罐 |
| RB | Re-arrange Boxes | 重新摆放两个盒子保持整洁 |
| PuS | Push Sugar | 把糖盒推到一起 |
| PoS | Pour Sugar | 给杨桃加糖 |
| PC | Push Cans | 把两个番茄罐推到一起 |

灵巧手：Ability、Inspire、X-Hand1、Paxini DexH13（见 Table 3）。机器人平台：双臂 7-DoF xArm + Unitree G1 人形。物体多取自现有数据集（YCB 等）。

### 实现细节

- **Backbone**: 由 [[π0|$\pi_0$]]（[[PaliGemma]] 初始化）权重初始化；用 latent action token 替换 state token。
- **隐自编码器**: 多头 [[VAE]] MLP，架构 $H_{64}^{128\to64}$、隐维 32；自监督训练（随机采样关节构型 + 可微 FK），全程冻结于 VLA 微调阶段。
- **动作块**: 64 帧 @ 20 Hz（约 3.2 s）。
- **VLA 训练**: 8× NVIDIA H100（80GB），60K 步，batch size 128；单个多任务策略约训练 10 小时。
- **输入处理**: RGB 由 960×540 裁剪缩放到 320×240（后处理），载入时再缩到 224×224；单前视 RealSense L515。
- **数据采集**: xArm 用 Apple Vision Pro 跟踪手/腕 + 重定向 + IK；G1 用 HOMIE + ACE-F 上半身遥操作 + MANUS Mocap 手套。
- **评测**: 每设置 10 次试验，物体位置随机初始化、机器人初始关节固定；未见任务记录 PSR（部分成功率，单臂完成给 0.5）。

### 关键实验结论

- **跨手数据扩展（Table 2）**: XL-VLA 全任务全手平均成功率 0.90，远超 $\pi_0$ 的 0.55（+0.35）；精细任务提升尤大。
- **跨机器人数据扩展（Figure 5 / Table 6）**: xArm+G1 共训下，对齐隐空间优于原始动作空间，G1 平均 0.825 vs 0.525。
- **零样本泛化（Figure 4）**: 训练时留出任务，测试时直接经对应解码器迁移；XL-VLA 在所有手/任务上**从不低于** $\pi_0$+RT（运动学重定向）基线，HB/RB 等精细任务优势显著。
- **隐式重放（Table 4）**: vs [[LAD]] 监督方法，XL-VLA 无监督即取得 0.82/0.81 vs 0.60/0.61。
- **消融（Table 5）**: $L_1$ 不可去；$L_2$ 的 dir/dist 两项各管方向/距离、缺一即崩；隐维过大反而损害本体不变性。

---

## 批判性思考

### 优点
1. **「即插即用」的跨本体接口**: 把本体差异收进冻结的小 MLP 编/解码器，VLA 主干保持 hand-agnostic，新手只需训练一对编/解码器即可接入，工程上极具吸引力。
2. **完全自监督的隐对齐**: $L_2$ 仅靠可微前向运动学与随机关节采样，**无需配对跨手轨迹/示范**，相对 [[LAD]] 等监督方法大幅降低数据门槛，且重放成功率更高。
3. **真机、跨平台验证扎实**: 不止仿真——在双臂 xArm + 人形 G1、4 款真实灵巧手、10 个真实任务上系统评测，覆盖跨手与跨机器人两层泛化，并给出零样本设置。

### 局限性
1. **VLA 侧基线单一**: 主结果（Table 2）几乎只与 $\pi_0$ 对比，缺少其他跨本体 VLA（如 UniVLA、X-VLA、Tenma 等）的同场较量；隐空间侧也只比 LAD 一个监督基线。
2. **隐空间几何先验偏「捏合」**: $L_2$ 仅对齐拇指-四指 pinch 的方向/距离，假设各手手指可语义对齐（Paxini 缺小指需手工丢对），对非捏合类抓取（包络抓、推压、in-hand 旋转）的几何一致性是否充分缺乏直接证据。
3. **绝对成功率仍有限**: 部分精细/接触丰富任务（如 RL 0.43、PuS 0.35）成功率偏低；任务集中在桌面抓放/倾倒/推挤，未覆盖长程、可变形物体、in-hand 操作等更难场景。
4. **缺少效率/频率报告**: 主打高频灵巧控制，却未给推理频率、显存、延迟等部署指标；隐空间引入的额外编解码开销未量化。

### 潜在改进方向
1. 加入更多跨本体 VLA 与隐空间基线、更严格的零样本协议（未见手而非仅未见任务），并量化「隐对齐 → 泛化」的因果（如表征探针、CKA）。
2. 把 $L_2$ 从「指尖 pinch 几何」推广到接触面/全手姿态/接触力的对齐，覆盖 in-hand 与包络抓取。
3. 把编/解码器架构与损失权重（$\beta,\lambda$）做成可学习/可搜索，减少手工调参与人工手指对齐。
4. 报告并优化部署侧效率指标，验证隐空间是否真带来高频灵巧控制收益。

### 可复现性评估
- [ ] 代码开源（HTML 中未给出项目主页/代码链接）
- [ ] 预训练模型（未声明）
- [x] 训练细节较完整（GPU、步数、batch、超参 $\beta/\lambda$、数据规模、附录硬件/采集细节齐全）
- [ ] 数据集可获取（自采 2M 数据集，未声明是否公开）

---

## 速查卡片

> [!summary] XL-VLA: Cross-Hand Latent Representation for VLA
> - **核心**: 用跨灵巧手共享的隐式动作空间替换 $\pi_0$ 的 state token，让单一 hand-agnostic VLA 跨多手训练并零样本迁移到新「手-任务」组合。
> - **方法**: 多头 VAE 编/解码器（hand-specific），三损失塑形隐空间——重建 $L_1$ + 经可微 FK 的指尖几何重定向 $L_2$（方向+距离）+ KL 正则 $L_3$；自监督（随机关节采样，无配对轨迹）；VLA 微调时冻结编/解码器。
> - **结果**: 4 手 10 任务真机，平均成功率 0.55→0.90（+0.35）；隐重放 0.82 vs LAD 0.60；G1 跨机器人 0.825 vs $\pi_0$ 0.525；零样本任务从不低于运动学重定向基线。
> - **数据/硬件**: 2M state-action 对（2000 条示范）；双臂 xArm + Unitree G1；8×H100、60K 步、batch 128。

---

*笔记创建时间: 2026-06-29*
