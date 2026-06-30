---
title: "Spatial-Aware VLA Pretraining through Visual-Physical Alignment from Human Videos"
method_name: "VIPA-VLA"
authors: [Yicheng Feng, Wanpeng Zhang, Ye Wang, Hao Luo, Haoqi Yuan, Sipeng Zheng, Zongqing Lu]
year: 2026
venue: CVPR
tags: [VLA, spatial-aware, 3D-pretraining, human-videos, dual-encoder, flow-matching, visual-physical-alignment]
zotero_collection: _inbox
image_source: online
arxiv_html: https://arxiv.org/html/2512.13080v1
created: 2026-06-29
---

# Spatial-Aware VLA Pretraining through Visual-Physical Alignment from Human Videos

## 元信息

| 项目 | 内容 |
|------|------|
| 作者 | Yicheng Feng, Wanpeng Zhang, Ye Wang, Hao Luo, Haoqi Yuan, Sipeng Zheng, Zongqing Lu |
| 机构 | 北京大学、中国人民大学、BeingBeyond |
| 会议 | CVPR 2026 |
| 类别 | VLA（Vision-Language-Action） |
| 日期 | 2025-12（arXiv v1） |
| 项目主页 | （未公开） |
| 链接 | [arXiv](https://arxiv.org/abs/2512.13080) / [CVPR poster](https://cvpr.thecvf.com/virtual/2026/poster/37199) |

---

## 一句话总结

> 从大规模人类操作视频里抽取 3D 视觉与 3D 动作标注，在机器人策略学习之前先做"视觉-物理对齐"预训练，用双编码器把 2D 观测与 3D 空间显式对齐，从而无需机器人数据预训练即获得强空间锚定的 VLA。

---

## 核心贡献

1. **Spatial-Aware VLA Pretraining 范式**: 提出在 [[VLA]] 学习机器人策略 *之前*，先用人类视频做一个显式的"视觉空间 ↔ 物理空间"对齐预训练，让模型先获得 3D 空间理解再学动作，填补 2D 感知与 3D 动作之间的鸿沟。
2. **Hand3D 数据集**: 从 9 个异构人类操作视频源构建两套监督——`Hand3D-visual`（约 300K 条 3D 视觉 VQA）与 `Hand3D-action`（约 1M 条视频-指令-动作三元组），把稠密 3D 几何转成语言锚定的标签。
3. **VIPA-VLA 双编码器架构**: 在语义视觉编码器之外引入 [[Cut3R]] 3D 编码器，通过交叉注意力融合层把空间几何特征注入语义视觉表征；**不依赖任何机器人数据预训练**，在 [[LIBERO]]/[[RoboCasa]] 与真实机器人上达到甚至超过 $\pi$ 系列、GR00T 等大规模 robot-pretrain 基线。

---

## 问题背景

### 要解决的问题
当前 [[VLA]] 模型普遍**用 2D 视觉输入去驱动 3D 物理环境中的动作**，感知与动作之间缺乏强对应关系。一个有效的策略不仅要"看懂像素"，还要理解这些视觉线索如何映射到 3D 几何、物理动作如何与环境交互。人类能从 2D 信号推断 3D 空间，但现有 VLA 基本忽视这一点，导致空间锚定差、泛化弱。

### 现有方法的局限
- **VLA 主线**: 要么把动作 token 化后在机器人数据上微调 VLM（[[OpenVLA]]、RT-2），要么加动作专家（$\pi_0$、GR00T），都强依赖大规模机器人数据集（OXE/DROID）。
- **3D VLM**: 用显式 3D 标注或单目深度估计增强空间理解，但**重感知、轻动作**——只提升静态观测的空间理解，没有建立 3D 感知与物理动作空间的对应，对训练 VLA 用处有限。
- **从人类视频学习**: 表征学习方法得到的特征过于隐式；直接对齐人类与机器人动作空间又受 **embodiment mismatch**（人手与机器人运动学差异巨大）困扰；affordance/抓取策略类工作仍停留在没有显式打通"3D 感知-动作"。Being-H0 虽用人类视频做 VLA 预训练，但重点在学操作动作序列本身。

### 本文的动机
作者主张：尽管存在 embodiment gap，**人类视频里富含"动作如何在 3D 物理世界中执行"以及对应视觉上下文的信息**。把这种信息当作 Spatial-Aware Pretraining 的监督，可以让 VLA 在下游策略学习之前就建立起 2D 视觉观测与 3D 动作空间之间的锚定关系。核心假设：**先补齐 3D 空间理解 = 下游策略更鲁棒、更泛化**。

---

## 方法详解

### 模型架构

VIPA-VLA 是一个**三阶段（两阶段预训练 + 一阶段后训练）**的 [[VLA]] 流水线（见 Figure 1），核心是一个**双编码器**架构（见 Figure 4）：
- **输入**: 视觉观测 $v=\{v_1,\dots,v_T\}$ + 语言指令 $l$（+ 后训练阶段的机器人状态 $s_t$）
- **Backbone**: [[InternVL3.5]]-2B
- **双编码器**: 语义视觉编码器（输出 $\mathbf V_{sem}$）+ [[Cut3R]] 3D 视觉编码器（输出 $\mathbf V_{spa}$），经**交叉注意力融合层**得到空间-语义融合特征 $V_f$
- **核心机制**: 预训练期把视觉 token 与文本/动作 token 对齐；后训练期用 [[Flow Matching|流匹配]] [[Diffusion Transformer|DiT]] 动作头产生动作块
- **输出**: 连续 [[Action Chunking|动作块]] $\mathbf a_t=\{\mathbf a_t^1,\dots,\mathbf a_t^H\}$

### 核心模块

#### 模块1: Hand3D 数据构建（3D 视觉 + 3D 动作标注）

**设计动机**: 人类操作视频天然揭示动作如何在 3D 空间中展开，是免费且大规模的"视觉-物理对应"监督来源。

**具体实现**:
- **视频汇聚**: 沿用 UniHand，聚合 9 个源——动捕类（Arctic、HOI4D、FPHA、H2O、OAKINK2、TACO、Dex-YCB）、VR 录制（EgoDex）、伪标注（Taste-Rob）。所有手部标注统一到 [[MANO]] 参数表示（有 3D 关节标注的用梯度优化拟合，纯视频的 Taste-Rob 用 HaWoR 估计）。
- **3D 视觉标注（构成 Figure 2 的核心管线）**:
  - 用 [[Cut3R]] 得到每帧稠密点云 $\mathcal P=\{(x_i,y_i,z_i)\}_{i=1}^N$（选它因为对动态场景鲁棒且在人-物交互数据上预训练过）；
  - 用 Gemini-2.5-flash + GroundingDINO 得到物体 2D 框 $\mathcal B_o$，结合点云深度做 3D 物体定位；
  - 把 MANO 手参变换到相机系得 21 个 3D 关节 $\mathcal J_h$，投影回像素平面（见公式3、4）并按可见性过滤手出画面的帧；
  - **尺度校准**: 单目点云的相对尺度与真实物理尺度不一致，会破坏动作学习；用绝对的手关节深度与点云深度的比值中位数估计尺度因子 $s$（公式5），把 $\mathcal P$ 校准到 $s\mathcal P$，使手与物体在统一的物理坐标系下。
  - **指令化**: 用 Gemini-2.5-flash 生成四类 VQA：(1) 空间关系、(2) 任务完成、(3) 手部运动、(4) 相机运动。方向/距离用位移向量 $\mathbf v$ 表示，距离取欧式范数、方向按阈值 $\gamma$ 离散成轴对齐语言 token（公式6、7）。
  - 采样约 4K 片段标注，产出约 300K 指令-答案对，即 **`Hand3D-visual`**（见 Table 1）。
- **3D 动作标注**: 从手部运动序列抽取手腕轨迹 $(x_t,y_t,z_t)$，均匀分箱离散成 motion token；沿用 UniHand 得文本指令，产 4M 视频-指令-动作三元组，过滤掉 3D 手部位移不显著的样本后得 1M 的 **`Hand3D-action`**（见 Table 2），含指令式动作生成、运动翻译、上下文运动预测三类 VQA。

#### 模块2: 双编码器 + 交叉注意力融合层（VIPA-VLA 架构本体）

**设计动机**: 传统 VLA 只有语义视觉编码器，缺乏 3D 空间结构特征。

**具体实现**:
- 语义编码器出 $\mathbf V_{sem}\in\mathbb R^{N_v\times d_v}$，[[Cut3R]] 3D 编码器出 $\mathbf V_{spa}\in\mathbb R^{N_s\times d_s}$。
- **融合层**（受 VLM-3R 启发）是交叉注意力：把两者投影到共享注意力空间，**让视觉 token 去 query 3D 空间 token**，输出再投回视觉维度得 $F_{spa}$。
- 用**带可学习缩放系数 $\alpha$ 的残差连接**融合（公式8），在注入空间信息的同时保留预训练视觉语义；再加 dropout + LayerNorm 稳定优化。$\alpha$ 初始化为 0.5。
- 第二预训练阶段还**扩展 LLM token 嵌入空间**，引入一组 motion token，把手腕轨迹点 $(x_t,y_t,z_t)$ 在预定义有界范围内均匀离散成三个 motion token，使 LLM 能理解细粒度 3D 运动轨迹。

#### 模块3: 三阶段训练（两段预训练 + 后训练）

**设计动机**: 先对齐视觉-3D（Stage1）、再灌入 3D 动作先验（Stage2）、最后适配机器人动作（Stage3），逐步从"看懂空间"过渡到"会动"。

**具体实现**:
- **Stage 1 — 3D-Visual Pretraining**: 用 `Hand3D-visual`，**只训融合层**（其余冻结），把 2D 视觉特征与 3D 空间表征对齐。
- **Stage 2 — 3D-Action Pretraining**: 用 `Hand3D-action`，**训融合层 + LLM 主干**，让模型从人手轨迹学到物理锚定的动作先验。
- **Stage 3 — Post-Training（后训练）**: 接一个 [[Flow Matching|流匹配]] [[Diffusion Transformer|DiT]] 动作头，**冻结视觉编码器与 3D 编码器**，只更新 LLM 主干与动作头 $f_\theta$，在下游机器人任务上产生可执行动作块（详见公式9–12）。

### 关键公式与机制

#### 公式1: [[VLM]] 映射

$$
y = f_{\text{VLM}}(v, l)
$$

**含义**: VLM 把视觉输入 $v$ 与语言指令 $l$ 映射到对齐嵌入空间，产出文本 $y$（答案/描述）。

**符号说明**:
- $v=\{v_1,\dots,v_T\}$: 图像或视频帧序列；$l$: 语言指令；$y$: 文本输出

#### 公式2: [[VLA]] 映射

$$
\mathbf{a}_t = f_{\text{VLA}}(v, l)
$$

**含义**: VLA 在 VLM 之上扩展到动作域，给定观测 $v$ 与指令 $l$ 预测一段动作块。

**符号说明**:
- $\mathbf a_t=\{\mathbf a_t^1,\dots,\mathbf a_t^H\}$: 长度为 $H$ 的动作块；其余同公式1

#### 公式3 & 4: 3D 手关节投影到图像平面

$$
(u,v)=\Pi\!\left(K[R\,|\,t](x,y,z)^{\top}\right)
$$

$$
\Pi(x',y',z')=\left(x'/z',\ y'/z'\right)
$$

**含义**: 把相机系下的 3D 手关节坐标经内参 $K$ 与外参 $[R|t]$ 投影回 2D 像素，用于建立 2D-3D 对应并按可见性过滤帧。

**符号说明**:
- $K$: 相机内参；$[R\,|\,t]$: 外参（旋转/平移）；$\Pi$: 透视投影；$(u,v)$: 像素坐标

#### 公式5: 点云尺度校准

$$
s = \operatorname{median}_{k\in\Omega}\left(j_{k}^{z}/\tilde{j}_{k}^{z}\right)
$$

**含义**: 单目点云只有相对尺度，与真实物理尺度不符会害动作学习；用绝对手关节深度 $j_k^z$ 与点云估计深度 $\tilde j_k^z$ 之比的中位数估计尺度因子 $s$，再把点云缩放为 $s\mathcal P$。

**符号说明**:
- $j_k^z$: 第 $k$ 关节在物理空间的绝对深度；$\tilde j_k^z$: 点云估计深度
- $\Omega$: 由可见性与有效深度筛出的有效关节集；中位数对离群点鲁棒

#### 公式6: 位移距离

$$
\mathrm{dist}(\mathbf{v}) = \lVert\mathbf{v}\rVert_2
$$

**含义**: 两实体（物-手或跨帧手位）间的相对 3D 偏移 $\mathbf v=(x,y,z)$ 的距离取欧式范数。

#### 公式7: 方向离散化

$$
\mathcal{D}=\{\text{right/left if } |\hat{x}|>\gamma,\ \text{up/down if } |\hat{y}|>\gamma,\ \text{forward/backward if } |\hat{z}|>\gamma\}
$$

**含义**: 由单位向量 $\hat{\mathbf v}=\mathbf v/\lVert\mathbf v\rVert_2$ 得方向，按阈值 $\gamma$ 把每个轴分量离散成轴对齐语言 token，过滤掉可忽略分量，得到语言锚定的方向标签。

**符号说明**:
- $\hat x,\hat y,\hat z$: 单位位移向量的三个分量；$\gamma$: 逐分量阈值

#### 公式8: 双编码器融合（残差 + 可学习缩放）

$$
V_f = V_{sem} + \alpha F_{spa}
$$

**含义**: 用可学习系数 $\alpha$ 把交叉注意力得到的空间特征 $F_{spa}$ 残差注入语义特征 $V_{sem}$，既引入 3D 几何又保住预训练视觉语义。

**符号说明**:
- $V_{sem}$: 语义视觉特征；$F_{spa}$: 视觉 token query 3D 空间 token 后投回视觉维度的空间特征
- $\alpha$: 可学习缩放（初始化 0.5）

#### 公式9: 后训练条件上下文

$$
h_{\mathrm{cond}} = \mathrm{VLM}_{\phi}(v, l, \mathcal{Q}_a)
$$

**含义**: 把视觉、语言与一组固定 action query $\mathcal Q_a$ 一起喂入预训练主干，取 $\mathcal Q_a$ 对应的隐状态作为 DiT 的条件。

**符号说明**:
- $\mathcal Q_a$: 固定的 action query 集；$\phi$: VLM 主干参数；$h_{\mathrm{cond}}$: DiT 条件上下文

#### 公式10: Flow Matching 噪声动作插值

$$
\tilde{\mathbf{a}}_t^{(\tau)} = (1-\tau)\,\boldsymbol{\epsilon} + \tau\,\mathbf{a}_t,\quad \tau\sim\mathcal{U}(0,1)
$$

**含义**: 在随机噪声 $\boldsymbol\epsilon$ 与真值动作 $\mathbf a_t$ 之间线性插值，构造时刻 $\tau$ 的带噪动作轨迹作为流匹配训练样本。

**符号说明**:
- $\boldsymbol\epsilon$: 随机噪声；$\tau\sim\mathcal U(0,1)$: 均匀采样的插值/时间权重

#### 公式11: DiT 输入拼接

$$
h_{\mathrm{DiT}} = \operatorname{concat}\!\big(\tilde{\mathbf{a}}_t^{(\tau)},\, s_t\big)
$$

**含义**: 把带噪动作与机器人状态嵌入 $s_t$ 拼接作为 DiT 的输入表示。

**符号说明**:
- $s_t$: 机器人状态嵌入

#### 公式12: 流匹配损失

$$
\mathcal{L}_{\mathrm{FM}} = \mathbb{E}_{\mathbf{a}_t,\tau,\boldsymbol{\epsilon},v,l}\left[\left\|\mathbf{v}_{\theta} - (\mathbf{a}_t - \boldsymbol{\epsilon})\right\|_2^2\right]
$$

**含义**: 训练 DiT 在条件 $h_{\mathrm{cond}}$ 下预测瞬时流向量 $\mathbf v_\theta$，使其逼近从噪声到真值的 oracle 传输方向 $\mathbf a_t-\boldsymbol\epsilon$；后训练只更新 LLM 主干与动作头。

**符号说明**:
- $\mathbf v_\theta$: DiT 预测的瞬时流；$\mathbf a_t-\boldsymbol\epsilon$: oracle 传输方向；$\mathbb E$: 对动作/时间/噪声/视觉/语言求期望

---

## 关键图表

<!-- 图片均使用 arXiv HTML 在线链接，未本地化 -->

### Figure 1: Overview of the Visual-Physical Alignment framework / 框架总览

![Figure 1](https://arxiv.org/html/2512.13080v1/x1.png)

**说明**: 整体范式。从多样化人类视频抽取 3D 视觉与 3D 动作标注 → 两阶段 Spatial-Aware Pretraining：(1) 3D-Visual Pretraining 用双编码器融合把 2D 特征对齐到 3D 空间；(2) 3D-Action Pretraining 用手部轨迹提供 3D 运动监督，学物理锚定动作先验 → 第三阶段把预训练好的 VIPA-VLA 适配到仿真与真实机器人任务。这是理解"预训练在策略学习之前补 3D 空间理解"主线的总图。

### Figure 2: Overview of Hand3D-visual / 3D 视觉标注管线

![Figure 2](https://arxiv.org/html/2512.13080v1/x2.png)

**说明**: `Hand3D-visual` 的构建流程：点云估计（Cut3R）+ 物体定位（GroundingDINO/Gemini）+ 手部位姿（MANO）三路结合，把 2D 观测桥接到 3D 物理动作空间。佐证"3D 视觉标注从何而来"以及尺度校准的必要性。

### Figure 3: Examples of Hand3D-visual / 数据样例

![Figure 3](https://arxiv.org/html/2512.13080v1/x3.png)

**说明**: `Hand3D-visual` 四类任务（空间关系、任务完成、手部运动、相机运动）的样例，体现数据多样性为预训练提供丰富的视觉-物理对齐监督。

### Figure 4: Model architecture of VIPA-VLA / 模型架构

![Figure 4](https://arxiv.org/html/2512.13080v1/x4.png)

**说明**: 左为双编码器（语义编码器 + 3D 编码器 → 交叉注意力融合层 → 空间-语义融合特征）；预训练期视觉 token 与文本/motion token 对齐；后训练期 action query 与融合的视觉-语言特征交互产生条件，与机器人状态拼接后由流匹配动作头输出动作。本文方法的核心结构图。

### Figure 5: Real Robot Task Settings / 真实机器人任务设置

![Figure 5](https://arxiv.org/html/2512.13080v1/x5.png)

**说明**: 三个真实任务（Put-Three-Obj 把三个水果放进抽屉、Wipe-Board 擦白板笔迹、Water-Plant 浇花）的场景设置，分别考察多物体定位、不规则区域空间推理、细粒度 3D 运动控制。

### Figure 6: Qualitative examples on real robot / 真实任务执行示例

![Figure 6](https://arxiv.org/html/2512.13080v1/x6.png)

**说明**: VIPA-VLA 执行三个真实任务的定性帧序列，展示成功完成各子任务的过程。

### Figure 7: Failure examples (VIPA-VLA vs InternVL3.5) / 失败案例对比

![Figure 7](https://arxiv.org/html/2512.13080v1/x7.png)

**说明**: VIPA-VLA 与未做 Spatial-Aware Pretraining 的 InternVL3.5 在真实任务上的失败案例对比，说明缺乏 3D 锚定时基线更易出错。

### Figure 8: 3D spatial understanding on Hand3D-test / 空间理解直方图

![Figure 8](https://arxiv.org/html/2512.13080v1/x8.png)

**说明**: 在 `Hand3D-test`（2K 未见视频 VQA）上 VIPA-VLA-PT vs InternVL3.5 的对比。左：方向得分直方图；右：距离误差直方图。VIPA-VLA-PT 方向更准、距离误差更小，定量佐证第一阶段预训练确实提升 3D 空间理解。

### Figure 9: Predicted vs ground-truth motion trajectories / 轨迹预测可视化

![Figure 9](https://arxiv.org/html/2512.13080v1/x9.png)

**说明**: 第二阶段预训练后 VIPA-VLA 预测的运动轨迹（蓝线）与真值轨迹（红线）对比，显示 3D-Action Pretraining 学到了贴近真值的手部运动模式。

### Table 1: Data distribution of Hand3D-visual / 视觉子集分布

| 类别 | Count | Proportion |
|------|-------|-----------|
| **Sources** | | |
| Arctic | 100,772 | 33.5% |
| OakInk2 | 45,926 | 15.3% |
| TACO | 39,087 | 13.0% |
| H2O | 32,390 | 10.8% |
| HOI4D | 28,977 | 9.6% |
| EgoDex | 28,200 | 9.4% |
| FPHA | 12,918 | 4.3% |
| Taste-Rob | 7,181 | 2.4% |
| Dex-YCB | 4,917 | 1.6% |
| **Task Types** | | |
| Task Completion | 206,409 | 68.7% |
| Spatial Relations | 74,887 | 24.9% |
| Hand Movements | 18,867 | 6.2% |
| Camera Movements | 205 | 0.1% |
| **Total** | **300,368** | **100%** |

**说明**: `Hand3D-visual` 共约 30 万条，源以 Arctic 占比最高（33.5%），任务类型以"任务完成"为主（68.7%），相机运动极少（仅 0.1%）。

### Table 2: Data distribution of Hand3D-action / 动作子集分布

| 类别 | Count | Proportion |
|------|-------|-----------|
| **Sources** | | |
| EgoDex | 758,050 | 73.5% |
| Arctic | 104,032 | 10.1% |
| TACO | 77,100 | 7.5% |
| OakInk2 | 54,470 | 5.3% |
| HOI4D | 19,812 | 1.9% |
| H2O | 17,386 | 1.7% |
| **Task Types** | | |
| Instructional Motion Generation | 610,192 | 59.2% |
| Contextual Motion Prediction | 280,545 | 27.2% |
| Motion Translation | 140,113 | 13.6% |
| **Total** | **1,030,850** | **100%** |

**说明**: `Hand3D-action` 共约 103 万条，主要来自 VR 录制的 EgoDex（73.5%），任务以指令式动作生成为主（59.2%）。两表共同刻画 Hand3D 的规模与多样性。

### Table 3: Success rates on LIBERO benchmark / LIBERO 成功率（%）

| Model | Robo-PT | LIBERO-S | LIBERO-O | LIBERO-G | LIBERO-L | Avg. |
|-------|---------|----------|----------|----------|----------|------|
| *单视角输入* | | | | | | |
| TraceVLA | ✓ | 84.6 | 85.2 | 75.1 | 54.1 | 74.8 |
| OpenVLA | ✓ | 84.7 | 88.4 | 79.2 | 53.7 | 76.5 |
| SpatialVLA | ✓ | 88.2 | 89.9 | 78.6 | 55.5 | 78.1 |
| DiT Policy | ✓ | 84.2 | 96.3 | 85.4 | 63.8 | 82.4 |
| CoT-VLA | ✓ | 87.5 | 91.6 | 87.6 | 69.0 | 83.9 |
| ThinkAct | ✓ | 88.3 | 91.4 | 87.1 | 70.9 | 84.4 |
| TriVLA | ✓ | 91.2 | 93.8 | 89.8 | 73.2 | 87.0 |
| 4D-VLA | ✓ | 88.9 | 95.2 | 90.9 | 79.1 | 88.6 |
| GR00T N1.5* | ✓ | 91.4 | **97.6** | 94.0 | 85.6 | 92.1 |
| **VIPA-VLA** | **✗** | **92.6** | 97.2 | **94.2** | **85.6** | **92.4** |
| *双视角输入* | | | | | | |
| MaIL† | ✗ | 74.3 | 90.1 | 81.8 | 78.6 | 83.5 |
| $\pi_0$-FAST | ✓ | 96.4 | 96.8 | 88.6 | 60.2 | 85.5 |
| MolmoAct | ✓ | 87.0 | 95.4 | 87.6 | 77.2 | 86.6 |
| GR00T N1 | ✓ | 94.4 | 97.6 | 93.0 | 90.6 | 93.9 |
| $\pi_0$ | ✓ | 98.0 | 96.8 | 94.4 | 88.4 | 94.4 |
| UniVLA | ✓ | 95.4 | **98.8** | 93.6 | 94.0 | 95.5 |
| $\pi_{0.5}$ | ✓ | **98.8** | 98.2 | **98.0** | 92.4 | **96.9** |
| **VIPA-VLA** | **✗** | 96.6 | 98.6 | 97.0 | **95.0** | 96.8 |

**说明**: VIPA-VLA **不用任何机器人数据预训练（Robo-PT ✗）**，单视角下平均 92.4% 超过所有基线（含 robot-pretrain 的 GR00T N1.5）；双视角下 96.8% 仅微弱低于 $\pi_{0.5}$(96.9%)，且在最难的 LIBERO-Long 上拿到最高 95.0%。相比同样强调空间推理的 SpatialVLA/TraceVLA/MolmoAct 全面领先。

### Table 4: Success rates on RoboCasa benchmark / RoboCasa 成功率（%）

| Model | Pick & Place | Doors / Drawers | Others | Avg. |
|-------|--------------|-----------------|--------|------|
| GR00T N1 | 18.6 | 50.2 | 39.1 | 36.0 |
| $\pi_{0.5}$ | **21.5** | 57.8 | 44.9 | 41.4 |
| **VIPA-VLA** | 20.8 | **67.7** | **52.8** | **45.8** |

**说明**: 在更难的 RoboCasa（24 任务，三视角，每任务仅 50 条人类示范）上 VIPA-VLA 总平均最高（45.8%），尤其在需要精确空间定位的 Doors/Drawers 上 +9.9%，印证 Spatial-Aware Pretraining 提供更可靠的 2D-3D 锚定。

### Table 5: Real robot success rates (seen) / 真实机器人（已见环境）成功率（%，子任务/整任务）

| Model | Put-Three-Obj | Wipe-Board | Water-Plant |
|-------|---------------|------------|-------------|
| GR00T N1.5 | 48% / 40% | 57% / 30% | 53% / 30% |
| Being-H0 | 38% / 20% | 40% / 10% | 37% / 20% |
| InternVL3.5 | 34% / 10% | 43% / 10% | 37% / 20% |
| **VIPA-VLA** | **52%** / 10% | **83% / 60%** | **57% / 50%** |

**说明**: 真实任务（7-DoF Franka + 6-DoF Inspire 手 + 双 RealSense L515，每任务 50 条遥操作示范、10 次评估）。VIPA-VLA 在子任务级普遍最高；整任务级在 Wipe-Board/Water-Plant 上显著领先，但 Put-Three-Obj 整任务仅 10%（长程串联任务一处失败即全败，体现长程难度）。

### Table 6: Real robot success rates (unseen) / 真实机器人（未见环境）成功率（%，子任务/整任务）

| Model | Put-Three-Obj-Unseen | Wipe-Board-Unseen |
|-------|----------------------|-------------------|
| GR00T N1.5 | 28% / 0% | 43% / 10% |
| Being-H0 | 16% / 0% | 33% / 10% |
| InternVL3.5 | 42% / 10% | 40% / 10% |
| **VIPA-VLA** | **44% / 20%** | **83% / 50%** |

**说明**: 未见环境（换未见桌布颜色 / 换记号笔颜色）下 VIPA-VLA 仍最优，Wipe-Board-Unseen 整任务 50% 远超基线（≤10%），体现空间锚定带来的泛化鲁棒性。

### Table 7: Ablations on LIBERO / 消融（%）

| Model | Spa. | Obj. | Goal | Long | Avg. |
|-------|------|------|------|------|------|
| **VIPA-VLA** | **92.6** | **97.2** | **94.2** | **85.6** | **92.4** |
| – Pretraining | 90.8 | 97.0 | 93.0 | 84.0 | 91.2 (-1.2%) |
| – Dual Encoder | 90.0 | 97.2 | 92.4 | 81.8 | 90.4 (-2.0%) |
| – Both | 89.2 | 95.2 | 90.0 | 80.4 | 88.7 (-3.7%) |

**说明**: 去掉 Spatial-Aware Pretraining 或双编码器都会一致掉点，二者都去掉掉得最多（-3.7%），且影响在最难的 Long 任务上最明显（85.6→80.4）。说明预训练提供 3D 运动/感知先验、双编码器负责 3D 视觉融合，二者互补。

### Table 8: Evaluation of 3D spatial understanding / 3D 空间理解评测

| Model | Dist. Err. (m) ↓ | Dir. Scr. ↑ |
|-------|------------------|-------------|
| InternVL3.5 | 0.18 | 1.22/3 |
| InternVL3.5 + Hand3D | 0.14 | 1.75/3 |
| **VIPA-VLA-PT** | **0.12** | **1.82/3** |

**说明**: 在 `Hand3D-test`（2K 未见视频 VQA）上，仅加 Hand3D 预训练就把距离误差从 0.18→0.14m、方向分 1.22→1.75；再加双编码器（完整 VIPA-VLA-PT）进一步到 0.12m/1.82。解耦验证 Hand3D 预训练与双编码器各自对 3D 空间理解的贡献。

---

## 实验

### 数据集 / 基准

| 基准 | 规模 | 特点 | 用途 |
|------|------|------|------|
| Hand3D-visual | ~300K VQA（9 源，~4K 片段） | 3D 视觉对齐监督 | Stage1 预训练 |
| Hand3D-action | ~1M 视频-指令-动作 | 3D 动作/运动监督 | Stage2 预训练 |
| Hand3D-test | 2K VQA（未见视频） | 测距离/方向准确度 | 空间理解评测 |
| [[LIBERO]] | 4 套件（Spatial/Object/Goal/Long），每套件 500 试 | 标准操作基准，重鲁棒/泛化 | 训练/测试 |
| [[RoboCasa]] | 24 任务（8 P&P / 6 Doors-Drawers / 10 Others），每任务 50 试 | 布局多样、场景杂乱、视觉更复杂 | 训练/测试 |
| 真实机器人 | 3 任务，每任务 50 遥操作示范、10 试 | Franka 7-DoF + Inspire 手 + 双 L515 | 训练/测试 |

### 实现细节

- **Backbone**: [[InternVL3.5]]-2B；图像 resize 到 $448\times448$，视频 1 fps 采样。
- **3D 编码器**: [[Cut3R]]（提供显式几何）；融合层为交叉注意力，缩放 $\alpha$ 初始 0.5。
- **Stage 1**: 用 `Hand3D-visual` 训 1 epoch，**仅训融合层**；AdamW，lr=1e-5，warmup 0.03，weight decay 0.01，cosine，global batch 32；约 6 小时 / 8×A800。空间关系与任务完成任务给 1–4 帧，手/相机运动任务给 2 帧。
- **Stage 2**: 用 `Hand3D-action` 训**融合层 + LLM 主干**，超参同 Stage1，单帧输入；约 20 小时 / 8×A800。
- **Stage 3（后训练）**: 接 [[Flow Matching]] [[Diffusion Transformer|DiT]] 动作头，**冻结视觉编码器与 3D 编码器**；lr=5e-5，warmup 0.05，weight decay 1e-5；LIBERO/真实任务 batch 128、训 30K 步（~5 小时）；RoboCasa batch 256、训 60K 步（~40 小时）。

### 关键实验结论

- **LIBERO（Table 3）**: 无 robot-pretrain 下单视角平均 92.4%（超全部基线）、双视角 96.8%（仅次于 $\pi_{0.5}$），Long 任务两种设置都最优。
- **RoboCasa（Table 4）**: 平均 45.8% 居首，Doors/Drawers +9.9%，且仅用 50 条人类示范。
- **真实机器人（Table 5/6）**: 已见与未见环境均整体领先，Wipe-Board-Unseen 整任务 50% 远超基线。
- **消融（Table 7）**: 预训练 -1.2%、双编码器 -2.0%、二者皆去 -3.7%，Long 任务最敏感。
- **空间理解（Table 8 / Fig 8）**: Hand3D 预训练 + 双编码器把距离误差 0.18→0.12m、方向分 1.22→1.82，直接量化"预训练补 3D 理解"。
- **轨迹可视化（Fig 9）**: Stage2 后预测轨迹（蓝）贴近真值（红）。

---

## 批判性思考

### 优点
1. **范式立意清晰且自洽**: "策略学习前先做视觉-物理对齐预训练"这一主线，从数据（Hand3D）、架构（双编码器）、训练（三阶段）到验证（Table 8 / Fig 8/9 的空间理解探针）形成闭环，不是只靠下游成功率空喊泛化。
2. **去机器人预训练仍打平甚至超越大模型**: 不用 OXE/DROID 级 robot 数据，LIBERO 单视角即超 GR00T N1.5，RoboCasa 超 $\pi_{0.5}$，显著降低对昂贵机器人数据的依赖。
3. **尺度校准这一工程细节抓得准**: 用绝对手关节深度做点云尺度校准（公式5），直击单目深度"相对尺度不匹配真实物理空间"会害动作学习的痛点，是该方法能落到物理动作的关键。

### 局限性
1. **长程任务整任务成功率仍低**: Put-Three-Obj 整任务仅 10–20%（即便子任务 52%），串联式长程任务一处失败即全败，离实用尚远；真实任务也只 3 个、桌面级。
2. **重度依赖外部大模型管线**: 数据构建链路（Cut3R + GroundingDINO + Gemini-2.5-flash/Pro + HaWoR + MANO 拟合）很长，标注质量受这些模型上限与误差累积影响，论文未量化伪标注噪声对下游的影响。
3. **embodiment gap 仅被"绕过"而非解决**: 用人手轨迹/motion token 作动作先验，但人手与机器人末端执行器运动学差异在方法层面没有显式建模，靠后训练的少量机器人数据去弥合；阈值 $\gamma$、分箱粒度、$\alpha$ 等关键超参缺敏感度分析。

### 潜在改进方向
1. 把 Spatial-Aware Pretraining 与机器人数据预训练**结合**（作者在结论中亦提出），形成更全面的预训练策略。
2. 对长程任务引入分层/子目标规划与失败恢复，缓解"一处失败全败"。
3. 用更可控/可学习的方式建模人-机 embodiment 映射（而非仅靠后训练数据），并对数据管线的伪标注噪声做系统性鲁棒性分析。

### 可复现性评估
- [ ] 代码开源（论文未给出代码/项目主页链接）
- [ ] 预训练模型（未声明 release）
- [x] 训练细节完整（三阶段超参、硬件、步数齐全，见 4.1）
- [x] 数据集可获取（Hand3D 基于公开人类视频源 + 公开模型构建；LIBERO/RoboCasa 公开，真实数据自采）

---

## 速查卡片

> [!summary] VIPA-VLA: Spatial-Aware VLA Pretraining via Visual-Physical Alignment
> - **核心**: 策略学习前，用人类视频做"视觉↔物理"3D 对齐预训练，双编码器把 2D 观测锚定到 3D 空间。
> - **方法**: Hand3D 数据（300K 视觉 VQA + 1M 动作三元组）→ InternVL3.5-2B + Cut3R 双编码器（交叉注意力融合，$V_f=V_{sem}+\alpha F_{spa}$）→ 三阶段：3D-Visual / 3D-Action 预训练 + 流匹配 DiT 动作头后训练。
> - **结果**: 无 robot-pretrain，LIBERO 92.4%（单视角超全部基线）/ 96.8%（双视角），RoboCasa 45.8% 居首，真实任务领先；空间理解距离误差 0.18→0.12m。
> - **代码**: 未公开

---

*笔记创建时间: 2026-06-29*
