---
title: "AT-VLA: Adaptive Tactile Injection for Enhanced Feedback Reaction in Vision-Language-Action Models"
method_name: "AT-VLA"
authors: [Xiaoqi Li, Muhe Cai, Jiadong Xu, Juan Zhu, Hongwei Fan, Yan Shen, Guanghui Ren, Hao Dong]
year: 2026
venue: CVPR
tags: [VLA, tactile, contact-rich-manipulation, dual-system, flow-matching, modality-agnostic]
zotero_collection: _inbox
image_source: online
arxiv_html: https://arxiv.org/html/2605.07308v2
created: 2026-06-29
---

# AT-VLA: Adaptive Tactile Injection for Enhanced Feedback Reaction in Vision-Language-Action Models

## 元信息

| 项目 | 内容 |
|------|------|
| 作者 | Xiaoqi Li, Muhe Cai, Jiadong Xu, Juan Zhu, Hongwei Fan, Yan Shen, Guanghui Ren, Hao Dong（Xiaoqi Li 与 Muhe Cai 共同一作） |
| 机构 | 北京大学计算机学院、PrimeBot、PKU Lab |
| 会议 | CVPR 2026 |
| 类别 | VLA（Vision-Language-Action）/ 触觉操作 |
| 日期 | 2026-05（arXiv v2） |
| 项目主页 | https://sites.google.com/view/at-vla |
| 链接 | [arXiv](https://arxiv.org/abs/2605.07308) / [Project](https://sites.google.com/view/at-vla) |

---

## 一句话总结

> 用一个可学习的[[Tactile Gate|触觉门]]决定"何时何地"注入触觉、并用慢视觉/快触觉的[[双流系统|dual-stream]]实现 0.04s 闭环反应，在接触密集任务上超越 SOTA，同时即便推理时缺触觉仍保持与原 VLA 相当的鲁棒性。

---

## 核心贡献

1. **Adaptive Tactile Injection（自适应触觉注入）**: 首次在"保留预训练知识"与"学习新引入的触觉表征"之间取得平衡——通过可学习触觉门 + [[Adaptive Cross Attention|自适应交叉注意力]]，**只在接触发生时**把触觉作为动作生成条件，最大程度不扰动原 [[VLA]] 的视觉感知/物体定位能力。
2. **Tactile Reaction Dual-Stream（触觉反应双流机制）**: 针对触觉的高频特性，把感知解耦为**慢视觉-语言流**（低频高层推理）与**快触觉控制流**（高频物理交互理解），实现 **0.04s 内的闭环实时响应**。
3. **真实世界 SOTA**: 在接触密集任务上优于 GO-1、$\pi_{0.5}$、VTLA、RDP 等 SOTA VLA / 触觉策略。
4. **模态无关鲁棒性（modality-agnostic）**: 虽用触觉训练，但**推理时即使缺失触觉**仍能保持与原 VLA 相当的性能，适配真实场景中触觉传感器不稳定/缺失的情况。

---

## 问题背景

### 要解决的问题
[[VLA]] 模型在通用任务上已很强，但在**接触密集（contact-rich）**操作（开拉链、盖章、擦曲面、拧盖）上仍受限，因为它们**忽略了触觉等交互反馈**，而这些反馈对精细控制与安全交互至关重要。如何把触觉接入预训练 VLA，且**不破坏其原有能力**、又能**高频实时**地响应触觉？

### 现有方法的局限
- 开源操作数据集（如 OXE、AgiBot World）**几乎不含触觉**，因此研究者只能在下游微调时引入触觉，并用表征对齐或 [[Chain-of-Thought|CoT]] 推理让模型"读懂"触觉。
- 但触觉与视觉/语言提供的信息**本质不同**，**直接引入这一新模态会破坏预训练知识**（视觉感知、物体 grounding），作者实验中观察到直接注入触觉甚至**降低**了抓取定位精度。
- VLA **推理速度慢**，难以对高频触觉反馈快速反应，削弱了闭环操作中触觉的价值。
- 现有触觉门方法通常在激活时**直接往 token 序列里追加触觉 token**，作者发现这会扰乱 VLA 的预训练序列建模；且它们**所有模态同频处理**，无法适配 VLA 的推理延迟。

### 本文的动机
受视觉与触觉**互补性**启发（视觉负责上下文定位、触觉提供精细接触反馈），作者主张：**非接触阶段保持原 VLA 行为，仅在接触时引入触觉**。由此设计了两件事：(1) 用触觉门 + 自适应交叉注意力决定"何时何地"注入触觉，且**不改变模型结构与特征维度**；(2) 用慢/快双流解决"高频触觉需要快速反应"的时延瓶颈。

---

## 方法详解

### 模型架构

AT-VLA 是一个**模块化、可挂载到大多数 VLA** 的框架（见 Figure 2）。具体实例化以预训练的 [[GO-1]] 作为 vanilla VLA（与本文用同款机器人 AgiBot Genie1 训练，且性能强）：

- **输入**: 多视角图像 $I=\{I_h, I_r, I_l\}$（头部 + 左右腕相机）+ 语言指令 $L$ + 触觉反馈 $T$ + 本体状态 $S$
- **Backbone（VLM）**: [[InternVL]]-2B（GO-1 的基座）
- **Action Expert**: [[DiT]] 模块，沿用 GO-1 的 [[Flow Matching|流匹配]] 动作生成管线
- **新增触觉编码器**: 几层 [[MLP]] 组成的轻量模块，保证快速推理
- **触觉信号**: 取触觉传感器的**合力（resultant force）**，含 3D 法向 + 3D 切向接触力（即 force 6D，消融中最优格式）
- **输出**: 动作块 $A$，表示双臂 **14-DoF 末端执行器位姿**

策略整体写作（公式1）：

$$
A = \pi_{\theta}(I, L, T, S)
$$

其中 $\pi_\theta$ 为策略网络，$\theta$ 为可学习参数。

### 核心模块

#### 模块1: Adaptive Tactile Injection（自适应触觉注入）

**Intuition（动机性实验）**: 作者先做一个对比实验（见 Figure 3）：(1) 仅微调 vanilla VLA 的 action expert、无触觉；(2) 直接把触觉 token 与视觉-语言 token 一起作为条件微调。**结果令人意外**——直接注入触觉不仅没提升，反而在**抓取定位**上出现明显误差。对 action expert 注意力图的分析显示：额外的触觉输入把模型注意力**从目标物体推离到周边区域**。结论：**新引入模态的 token 序列会干扰预训练模型的感知聚焦**，因此必须谨慎平衡预训练知识与新触觉学习。

**Tactile Gating（触觉门）**:
- 触觉编码器抽出触觉 token $\mathbf{z}_T$ 后，接一个由 MLP 组成的**轻量触觉门网络**，对当前触觉信号做**接触 / 非接触二分类**，输出一个表示接触状态的分数。
- 监督方式：人工标注训练片段，非接触帧标 0、接触帧标 1，用**二元交叉熵门损失** $\mathcal{L}_g$ 监督。
- 当分数超过阈值（如 0.5）时触觉门激活。这个门既区分接触阶段，又**迫使模型更好地理解触觉信号**。

**Adaptive Cross Attention（自适应交叉注意力）**:
- vanilla VLA 的 action expert 交叉注意力中：图像 token $\mathbf{z}_I$ 与文本 token $\mathbf{z}_L$ 作 **key/value**，状态 token $\mathbf{z}_S$ 作 **query**。
- AT-VLA 的关键设计：**门未激活时，query 仍是状态 token $\mathbf{z}_S$（与 vanilla 完全一致）；门激活时，query 被替换为触觉 token $\mathbf{z}_T$**。
- 这样**无需改动模型结构或特征维度**：无接触时输入与架构与原 VLA 完全相同（保留预训练表征，如靠近目标物体的能力）；一旦接触，模型才开始把触觉作为动作生成条件。

#### 模块2: Effective Tactile Reaction Dual-Stream（高效触觉反应双流）

**设计动机**: 触觉是高频信号、需快速动作调整，但 VLA 推理慢。把反应能力拆为两点：(1) **快速反应**——基于触觉迅速调整预测动作以保证安全与精度；(2) **触觉理解**——学会解释触觉含义并据此修改动作。

**具体实现**:
- **慢流（slow stream）**: 视觉 + 语言经大 VLM 在**低频**处理，负责任务理解与视觉感知，输出潜在特征作为 action expert 交叉注意力的 **key/value**。
- **快流（fast stream）**: 触觉反馈在**高频**持续喂入 action expert，作为交叉注意力的 **query** 条件，承担需要闭环实时响应的部分。
- 由此 action expert 的输入是**异步频率、异质模态**的。基于动作分块（action chunking）：$t_n$ 时刻的视觉-语言观测可指导未来 $H$ 步动作（$t_n : t_{n+H}$），即慢流输出作为**时间上覆盖一个动作块**的潜在条件；快流则在每个时间步用**最新触觉反馈**生成可执行动作。
- 触觉门激活时，**快流:慢流频率比设为 3:1**（受 Gr00t-N1、Fast-in-Slow 启发）——慢流推理一次、快流连续推理三次，兼顾效率与性能，实现 **0.04s 闭环**。
- 触觉门的接触/非接触判别还促使模型形成对**物理动力学与触觉语义**更完整的表征，连接瞬时接触感知与预测性交互推理。

### 关键公式与机制

#### 公式1: [[VLA]] 策略映射

$$
A = \pi_{\theta}(I, L, T, S)
$$

**含义**: 策略 $\pi_\theta$ 端到端地把多视角图像、语言指令、触觉反馈、本体状态映射为动作块。

**符号说明**:
- $I=\{I_h, I_r, I_l\}$: 头部 / 右腕 / 左腕相机的图像观测
- $L$: 语言指令；$T$: 触觉反馈（取合力，含 3D 法向 + 3D 切向）；$S$: 本体状态
- $A$: 动作块，表示双臂 14-DoF 末端执行器位姿；$\theta$: 策略参数

#### 公式2: 总训练目标

$$
\mathcal{L} = \mathcal{L}_a + \lambda_1 \cdot \mathcal{L}_g
$$

**含义**: 动作损失与触觉门分类损失**同时训练**，让模型既保留预训练知识、又能及时准确地响应触觉。

**符号说明**:
- $\mathcal{L}_a$: 动作损失（沿用 GO-1 的流匹配动作监督）
- $\mathcal{L}_g$: 触觉门的二元交叉熵损失（接触 1 / 非接触 0）
- $\lambda_1 = 0.01$: 平衡两项损失尺度的权重系数

**推理机制（非独立公式，但是关键流程）**: 触觉门**未激活**时，模型与原 VLA 完全一致（快慢流同频、query 为状态 token）；**激活**时启动 3:1 异步频率，action expert 的注意力 query 切换为触觉 token 作为动作生成条件。

---

## 关键图表

<!-- 图片均使用 arXiv HTML 在线链接，未本地化。论文 Figure 1（teaser）在 arXiv HTML 中未渲染为图像文件，故不收录。 -->

### Figure 2: Framework of AT-VLA / 整体框架

![Figure 2](https://arxiv.org/html/2605.07308v2/x1.png)

**说明**: AT-VLA 整体框架。触觉门**自适应**地决定触觉 token 是否作为 action expert 的条件输入。**门未激活**时，action expert 的所有输入模态**同频**运行（等价于 vanilla VLA）；**门激活**时，触觉信号以**更高频率**处理，实现快速精确的动作调整。这张图是理解"自适应注入 + 双流频率切换"两大机制如何协同的关键。

### Figure 3: Intuition / 动机性注意力对比

![Figure 3](https://arxiv.org/html/2605.07308v2/x2.png)

**说明**: 可视化 action expert 模块内的注意力图，对比不同下游微调策略（有/无触觉反馈）下注意力分布与动作推理的变化。核心发现：**直接引入触觉会把注意力从目标物体推离到周边区域**，导致抓取定位退化——这正是作者提出"自适应注入"而非"直接注入"的实证依据，与 Table 3 的 Ex1/Ex4/Ex6 定量退化结果相互印证。

### Figure 4: Visualization / 接触密集任务执行序列

![Figure 4](https://arxiv.org/html/2605.07308v2/x3.png)

**说明**: 四个典型接触密集任务（Unzip Bag 开拉链、Stamp 盖章、Wipe Vase 擦花瓶、Unscrew Lid 拧盖）的执行进度可视化，展示 AT-VLA 在接触阶段的轨迹调整与任务完成过程。

### Table 1: Evaluation in Contact-rich Tasks / 接触密集任务评测（分子任务成功率）

**Unzip Bag（开拉链）与 Stamp（盖章）**

| Method | Left Grasp→ | Right Grasp→ | Unzip Half→ | Unzip Full | Overall | Right Grasp→ | Stamp→ | Place | Overall |
|--------|------|------|------|------|------|------|------|------|------|
| GO-1 | 1.0 | 0.87 | 0.87 | 0.20 | 0.20 | 1.0 | 0.80 | 0.13 | 0.13 |
| $\pi_{0.5}$ | 1.0 | 0.67 | 0.67 | 0.0 | 0.0 | 1.0 | 0.53 | 0.20 | 0.20 |
| **AT-VLA (Ours)** | **1.0** | 0.80 | **0.80** | **0.33** | **0.33** | 0.90 | **0.87** | **0.46** | **0.46** |
| VTLA | - | - | 0.20 | 0.00 | - | - | 0.93 | 0.13 | - |
| RDP | - | - | 0.87 | 0.06 | - | - | 0.40 | 0.40 | - |

**Wipe Vase（擦花瓶）与 Unscrew Lid（拧盖）**

| Method | Left Grasp→ | Right Grasp→ | Wipe Half→ | Wipe Full | Overall | Left Grasp→ | Right Grasp→ | Rotate | Overall |
|--------|------|------|------|------|------|------|------|------|------|
| GO-1 | 1.0 | 0.93 | 0.30 | 0.07 | 0.07 | 1.0 | 0.87 | 0.27 | 0.27 |
| $\pi_{0.5}$ | 1.0 | 0.73 | 0.40 | 0.33 | 0.33 | 0.93 | 0.80 | 0.46 | 0.46 |
| **AT-VLA (Ours)** | **1.0** | 0.80 | **0.80** | **0.67** | **0.67** | 0.90 | **0.87** | **0.53** | **0.53** |
| VTLA | - | - | 0.93 | 0.60 | - | - | - | 0.80 | - |
| RDP | - | - | 0.80 | 0.33 | - | - | - | 0.87 | - |

**说明**: 报告每个子任务（→ 表示阶段递进）的成功率以反映进度。在完整任务（含抓取+接触阶段）的 Overall 上，AT-VLA 全面领先 GO-1/$\pi_{0.5}$，且在前接触抓取阶段与它们相当（说明保住了预训练 grounding）。注意 **VTLA/RDP 是被人工放到理想初始位姿后只评测接触段**（去掉了感知/grounding 要求），因此它们的接触段单项数值有时更高，但无法完成完整任务（Overall 留空）。唯一例外是 **Unscrew Lid 的 Rotate**，AT-VLA(0.53) 略逊 VTLA(0.80)/RDP(0.87)，原因是其自主抓取的握力不总够稳、拧盖时偶尔打滑，而基线被人工设定了稳定抓取。

### Table 2: Modality-agnostic Evaluation / 模态无关评测（推理时缺触觉的鲁棒性）

| Method | Pick Place | Open Drawer | Stamp | AVG. |
|--------|------|------|------|------|
| GO-1 | 1.0 | 0.93 | 0.13 | 0.68 |
| $\pi_{0.5}$ | 1.0 | 0.93 | 0.20 | 0.70 |
| AT-VLA w/o. | 1.0 | 0.93 | 0.20 | 0.70 |
| **AT-VLA w/.** | **1.0** | **0.93** | **0.46** | **0.79** |

**说明**: w/. 与 w/o. **共享同一套权重**，仅区别于推理时是否提供触觉；w/. 作为上界。关键结论：(1) 在两个非接触任务（Pick-Place、Open Drawer）上，**AT-VLA w/o.（推理无触觉）与 GO-1/$\pi_{0.5}$ 相当**——证明 Adaptive Tactile Injection 让模型即便缺触觉也能可靠生成动作，具强模态无关鲁棒性；(2) 在接触任务 Stamp 上，**AT-VLA w/o.（0.20）甚至略优于 vanilla GO-1（0.13）**——因为训练中见过触觉，隐式学到了接触动力学与跨模态关联，测试时能从视觉特征推断近似触觉线索。

### Table 3: Ablation Study / 消融实验（组件贡献 + 触觉格式）

| Exp | Tactile Gate | Adaptive Cross Attn | Direct Incorp. | Reaction Dual-Stream | V-T Image | Marker 2D | Force 6D | Unzip Bag | Stamp | Wipe Vase | Unscrew Lid | AVG. |
|-----|:---:|:---:|:---:|:---:|:---:|:---:|:---:|------|------|------|------|------|
| Ex0（vanilla VLA / GO-1） | - | - | - | - | - | - | - | 0.20 | 0.33 | 0.07 | 0.27 | 0.22 |
| Ex1（直接注入 force 6D） | - | ✓ | - | - | - | - | ✓ | 0.07 | 0.13 | 0.07 | 0.20 | 0.13 |
| Ex2（+触觉门, 同频） | ✓ | ✓ | - | - | - | - | ✓ | 0.27 | 0.40 | 0.53 | 0.33 | 0.39 |
| **Ex3（Ours：+双流）** | ✓ | ✓ | - | ✓ | - | - | ✓ | **0.33** | **0.46** | **0.67** | **0.53** | **0.50** |
| Ex4（直接注入 marker 2D） | - | ✓ | - | - | - | ✓ | - | 0.00 | 0.13 | 0.07 | 0.00 | 0.05 |
| Ex5（Ours + marker 2D） | ✓ | ✓ | - | ✓ | - | ✓ | - | 0.27 | 0.33 | 0.27 | 0.40 | 0.32 |
| Ex6（直接注入 V-T image） | - | ✓ | - | - | ✓ | - | - | 0.00 | 0.00 | 0.07 | 0.00 | 0.02 |
| Ex7（Ours + V-T image） | ✓ | ✓ | - | ✓ | ✓ | - | - | 0.27 | 0.46 | 0.47 | 0.40 | 0.40 |

**说明（组件贡献，Ex0–Ex3）**:
- **Ex1 直接注入触觉反而比 vanilla 降 9%**（0.13 vs 0.22），多数失败发生在抓取阶段，印证"直接注入破坏视觉 grounding"的 intuition。
- **Ex2 加触觉门 + 自适应交叉注意力，比 vanilla 提升 17%**（0.39），证明门控在**保留预训练知识**上的有效性（此时快慢同频）。
- **Ex3 再加双流（异频），比 Ex2 再提升 11%**（0.50），凸显**高频触觉快速响应**的必要性（如开拉链时延迟会导致卡住）。

**说明（触觉格式，对比 Ex1/Ex4/Ex6 vs Ex0，及 Ours 各格式）**:
- 三种**直接注入**格式（force 6D / marker 2D / V-T image）相对 vanilla **全部显著退化**（Ex1=0.13, Ex4=0.05, Ex6=0.02），再次确认直接加 token 会破坏预训练知识、连基础抓取都失败。
- 采用本文方法后大幅回升：**Ex5 比 Ex4 高 27%、Ex7 比 Ex6 高 38%**，说明框架对触觉格式鲁棒。
- 格式优劣排序：**force 6D（0.50）> marker 2D（0.32）> V-T image（0.40 但伴随更大扰动）**。作者假设：**高维触觉输入引入更多 token、会过度扰动预训练表征空间**，因此需平衡新模态与预训练模型的影响。（注：V-T image 用预训练编码器 [[Sparsh]] 抽特征。）

---

## 实验

### 数据集 / 任务

| 类别 | 任务 | 描述 | 触觉关键点 |
|------|------|------|------|
| 接触密集 | Unzip Bag | 沿拉链曲线轨迹开包 | 跟不上曲线会卡住/卡死 |
| 接触密集 | Stamp | 在指定区域盖章 | 需判断盖章完成，否则撞桌面 |
| 接触密集 | Wipe Vase | 顺花瓶曲面擦拭 | 顺应不足会撞到瓶颈 |
| 接触密集 | Unscrew Lid | 旋转开盖 | 需精确力/运动协调防打滑 |
| 非接触 | Pick and Place | 抓胡萝卜放盘 | 模态无关评测用 |
| 非接触 | Open Drawer | 抓把手开抽屉 | 模态无关评测用 |

- 每个任务采集 **30–50 条示范**，测试 **15 次试验**。

### 实现细节

- **硬件**: AgiBot Genie1，双 7-DoF 臂，1 前视 + 2 腕部相机；夹爪配 **Xense Robotics 触觉传感器**；VR 头显遥操作采集数据。
- **Backbone**: vanilla VLA 用 [[GO-1]]（基座 [[InternVL]]-2B + [[DiT]] action expert），在 AgiBot World 数据集预训练。
- **触觉**: 默认用 **force 6D（3D 法向 + 3D 切向合力）**；触觉编码器为轻量 MLP。
- **触觉门**: MLP 二分类网络，BCE 损失 $\mathcal{L}_g$，阈值 0.5。
- **损失权重**: $\lambda_1 = 0.01$。
- **双流频率比**: fast:slow = **3:1**；闭环响应 **0.04s**。
- **基线**: GO-1、$\pi_{0.5}$（有预训练权重、同下游数据但无触觉）；VTLA（Qwen2-VL + ViT 抽触觉、触觉表示为视觉-触觉图像，基于 DexVLA 复现）、RDP（基于 Diffusion Policy、2D marker + PCA 降维）。注意 VTLA/RDP 无大规模预训练，仅在接触段子集训练并被人工置于理想初始位姿评测。

### 关键实验结论

- **接触密集任务（Table 1）**: 完整任务 Overall 上 AT-VLA 全面超越 GO-1/$\pi_{0.5}$；接触段也优于触觉策略 VTLA/RDP（仅 Unscrew Lid 的 Rotate 因自主握力不稳略逊）。
- **模态无关（Table 2）**: 推理无触觉时与 vanilla 相当；Stamp 上 w/o. 甚至略优于 GO-1；w/. 平均 0.79（上界）。
- **消融（Table 3）**: 直接注入降 9%；触觉门 +17%；双流再 +11%；force 6D 优于 marker 2D 与 V-T image。

---

## 批判性思考

### 优点
1. **"直接注入有害"这一反直觉发现有据可循**: 用 action expert 注意力图（Fig 3）+ Table 3 的 Ex1/Ex4/Ex6 一致退化，定性定量双重支撑，而非空谈。
2. **极简而通用的工程设计**: 自适应交叉注意力**只切换 query 来源**（状态 token ↔ 触觉 token），不改结构、不改维度，使该机制能挂载到大多数现有 VLA，落地成本低。
3. **模态无关鲁棒性**: 推理缺触觉仍可用甚至小幅受益，直击真实部署中传感器不稳定/缺失的痛点，实用价值高。
4. **真实机器人验证**: 全部在 AgiBot Genie1 双臂硬件 + Xense 触觉上做真机实验，而非仅仿真。

### 局限性
1. **任务与试验规模偏小**: 仅 4 个接触密集 + 2 个非接触任务，每任务 30–50 示范、15 次试验，绝对成功率仍偏低（Ours 平均仅 0.50），统计置信度有限。
2. **基线评测设置不完全对等**: VTLA/RDP 被人工置于理想抓取位姿、去掉感知要求，与 AT-VLA 的"全流程自主"并非同一难度，跨方法比较的公平性需谨慎解读。
3. **触觉门依赖人工标注接触/非接触帧**: 监督来自手工标注，规模化与跨任务迁移成本未讨论；阈值 0.5、3:1 频率比、$\lambda_1=0.01$ 等关键超参均为经验设定，缺敏感度分析。
4. **未报告推理频率/算力细节与代码**: 仅给项目主页，未见开源代码/权重声明，0.04s 的具体测量条件（GPU 型号等）未充分交代。

### 潜在改进方向
1. 把触觉门做成**自监督/弱监督**（如用力阈值或自动接触检测），摆脱人工标注。
2. 把频率比、损失权重、注入层位等做**自动搜索或自适应**，并补充对 query 切换机制的表征探针（CKA 等）分析，量化"保留预训练→泛化"的因果。
3. 扩展到**更多接触密集任务、更大试验规模、更强基线**（对等评测协议），并验证该框架挂载到不同 VLA / 不同本体的可移植性。

### 可复现性评估
- [ ] 代码开源（仅给出项目主页 https://sites.google.com/view/at-vla，未见代码/权重声明）
- [ ] 预训练模型
- [x] 训练细节较完整（损失权重、频率比、阈值、硬件、触觉格式均有交代）
- [ ] 数据集可获取（自采真机数据，未声明 release；基座 GO-1/AgiBot World 公开）

---

## 速查卡片

> [!summary] AT-VLA: Adaptive Tactile Injection for Enhanced Feedback Reaction in VLA
> - **核心**: 用可学习触觉门决定"何时何地"注入触觉 + 慢视觉/快触觉双流（3:1 频率、0.04s 闭环），在不破坏预训练 VLA 的前提下接入触觉。
> - **方法**: 以 GO-1（InternVL-2B + DiT）为 vanilla VLA；Adaptive Cross Attention **只切换 query**（状态 token ↔ 触觉 token），门激活才注入触觉；总损失 $\mathcal{L}=\mathcal{L}_a+0.01\mathcal{L}_g$。
> - **结果**: 真机接触密集任务超 GO-1/$\pi_{0.5}$/VTLA/RDP；消融中触觉门 +17%、双流 +11%、force 6D 最优；推理缺触觉仍与 vanilla 相当（模态无关）。
> - **项目**: https://sites.google.com/view/at-vla

---

*笔记创建时间: 2026-06-29*
