---
title: "PointWorld: Scaling 3D World Models for In-The-Wild Robotic Manipulation"
method_name: "PointWorld"
authors: [Wenlong Huang, Yu-Wei Chao, Arsalan Mousavian, Ming-Yu Liu, Dieter Fox, Kaichun Mo, Li Fei-Fei]
year: 2026
venue: CVPR
tags: [world-model, 3D-point-flow, robotic-manipulation, MPC, scaling-law, PointTransformerV3, DINOv3]
zotero_collection: _inbox
image_source: online
arxiv_html: https://arxiv.org/html/2601.03782v1
created: 2026-06-29
---

# PointWorld: Scaling 3D World Models for In-The-Wild Robotic Manipulation

## 元信息

| 项目 | 内容 |
|------|------|
| 作者 | Wenlong Huang, Yu-Wei Chao, Arsalan Mousavian, Ming-Yu Liu, Dieter Fox, Kaichun Mo, Li Fei-Fei |
| 机构 | 斯坦福大学（Stanford）、NVIDIA |
| 会议 | CVPR 2026 |
| 类别 | 3D World Model / Robotic Manipulation / Model-Based Planning |
| 日期 | 2026-01（arXiv v1） |
| 项目主页 | https://point-world.github.io/ |
| 链接 | [arXiv](https://arxiv.org/abs/2601.03782) / [Project](https://point-world.github.io/) |

---

## 一句话总结

> 把状态与动作统一为同一个 3D 空间里的 [[3D Point Flow|3D 点流]]，预训练一个大规模点云骨干网络去预测「给定机器人动作后整场景如何在 3D 中位移」，单一 checkpoint 即可零样本驱动真实 Franka 完成推、变形、铰接、工具使用等操作。

---

## 核心贡献

1. **状态-动作统一表示 + 大规模 3D 世界模型**: 提出 [[PointWorld]]，用 **3D 点流** 同时表示场景状态（RGB-D 反投影点云）与机器人动作（由 URDF 前向运动学生成的机器人表面点轨迹），把世界建模化为「**动作条件下的整场景 3D 点位移预测**」，并系统性研究其规模化配方（骨干、动作表示、目标函数、部分可观测、数据混合、域迁移、scaling law）。
2. **最大规模 3D 动力学数据集**: 基于 [[DROID]]（真实）与 [[BEHAVIOR-1K]]（仿真）整理出约 **2M 条轨迹 / 500 小时** 的高质量 3D 交互数据，依托近期 3D 视觉进展（[[FoundationStereo]] 深度、[[VGGT]] 相机位姿、[[CoTracker3]] 点追踪）构建了无标记离线 3D 标注流水线，全部开源。
3. **零样本真实部署**: 单个预训练 PointWorld 以 **0.1s 实时推理** 嵌入 [[MPC]]（[[MPPI]]）框架，**无需任何示范或后训练**，仅凭一张野外 RGB-D 图像即可在真实 Franka 上完成刚体推、可变形/铰接操作与工具使用。

---

## 问题背景

### 要解决的问题

如何构建一个**通用、可规模化、可在野外（in-the-wild）泛化**的世界模型：从机器人「看到的」感知输入与「打算做的」动作出发，预测物理世界如何在 3D 空间与时间中演化（形变、铰接、稳定性、接触），并能直接用于真实机器人操作。

### 现有方法的局限

作者把已有世界模型按状态-动作表示分类，并指出三类痛点：
1. **物理仿真模型**（如 MuJoCo）预测精确，但有 sim-to-real gap，且需要针对每个环境做繁琐建模。
2. **学习型动力学模型**（particle/graph 等）依赖领域归纳偏置——常假设完全可观测、objectness 先验或材质规格。
3. **大规模视频生成世界模型** 视觉逼真，但缺乏显式动作条件、物理一致性差，且扩散推理耗时（秒级）。
此外，图域 [[GBND|图神经动力学(Graph-Based Neural Dynamics)]] 基线在大数据上**显存爆炸**且**纯局部消息传递在部分可观测下退化**；2D 流方法无法推理被遮挡区域。

### 本文的动机

核心哲学是 **「用统一换规模化（unification for scaling）」**：把状态与动作放进**同一个 3D 物理空间模态**。
- 状态 = RGB-D 构建的整场景 3D 点云；动作 = 由智能体自身几何（URDF 已知）实例化、可随时间前向预测的稠密 3D 点轨迹。
- 在此表示下，3D 世界建模 = 在机器人点序列扰动下预测整场景每点位移，自然隐式编码 objectness、铰接、材质属性，且能从异构本体/任务/成功失败轨迹中统一学习，类似「next-token prediction」但针对 3D 空间-时间中的交互。

---

## 方法详解

### 模型架构

PointWorld 把环境动力学建模为神经网络 $\mathcal{F}_{\theta}:\mathbf{S}\times\mathbf{A}\rightarrow\mathbf{S}$。区别于单步更新 $\mathbf{s}_{t+1}=\mathcal{F}_{\theta}(\mathbf{s}_t,\mathbf{a}_t)$，本文采用**分块（chunked）多步**预测，一次前向输出整段 horizon（见 Figure 2）：

$$
\mathcal{F}^{H}_{\theta}:(\mathbf{s}_t,\ \mathbf{a}_{t:t+H-1})\rightarrow\mathbf{s}_{t+1:t+H}
$$

- **输入**: 一张或几张标定 RGB-D + 机器人关节空间动作序列 + 机器人描述文件（URDF）
- **Backbone**: [[PointTransformerV3|PTv3]] 点云骨干（最高 PTv3-1B），场景点用**冻结 [[DINOv3]]** 投影到 2D 取特征，机器人点用时间嵌入
- **核心机制**: 把初始场景点与时间堆叠的机器人点**拼接成单一点云**送入骨干，共享 MLP 头一次预测整 chunk 内每个场景点的逐步位移
- **输出**: 整场景 [[3D Point Flow|3D 点流]]（每点逐步 3D 位移）
- **关键超参**: horizon $H=10$ 步，每步 0.1s（即 1 秒预测窗），**单次批量前向 0.1s 实时延迟**

整体记为 $\mathcal{F}_{\theta}$，其中 $\theta$ 为全模型可学习参数。

### 核心模块

#### 模块1: State Representation（状态表示——场景点流）

**设计动机**: 用点流（particles）作为环境状态，强调 3D 几何之间的**物理交互而非外观**（更像物理引擎而非渲染器）。

**具体实现**:
- 形式化为 $\mathbf{s}_t=\{(\mathbf{p}_{t,i},\mathbf{f}^S_i)\}_{i=1}^{N_S}$，含 $N_S$ 个点，位置 $\mathbf{p}_{t,i}\in\mathbb{R}^3$、时间不变特征 $\mathbf{f}^S_i\in\mathbb{R}^{D_S}$。
- 从一/几张标定 RGB-D 出发，用前向运动学（URDF + 关节配置）**屏蔽机器人像素**，反投影剩余像素得到 $\mathbf{p}_{t,i}$。
- 四大优势：(i) 强调几何接触而非外观；(ii) 任意 RGB-D 可得、部分可观测、不假设 objectness/材质先验；(iii) 用位移上的 L2 损失训练简单稳定、无需排列匹配；(iv) 能表达细粒度接触动力学。
- **关键点**: 模型只吃静态点集，对应关系仅在前向「想象」内部保持，**推理时无需点追踪器**，且每次前向点数可变。

#### 模块2: Action Representation（动作表示——机器人点流）

**设计动机**: 为从异构本体（不同运动学、夹爪几何、甚至不同夹爪数量）学习，动作也用 3D 点流，做到 **embodiment-agnostic（本体无关）**。

**具体实现**:
- 与场景点不同，机器人点流由 URDF 经前向运动学**预测自身几何**生成——「想象的动作」**完全可观测**（非部分可观测），这对接触发生在被遮挡区域（如以第一视角抱运大箱子）尤为关键。
- 给定关节配置序列 $\{\mathbf{q}_{t+k}\}_{k=0}^{H}$，在 $t$ 时刻采样一次机器人表面点、绑定到各 link，用前向运动学推进，得到每步有序集 $\{(\mathbf{r}_{t+k,j},\mathbf{f}^R_{t+k,j})\}_{j=1}^{N_R}$，作为时刻 $t{+}k$ 的动作 $\mathbf{a}_{t+k}$。
- **效率取舍**: 多数机器人表面点从不接触场景，故**仅从夹爪采样点流**（每夹爪 300–500 点）——消融证明这是最优（见 Figure 11）。

#### 模块3: Dynamics Prediction（动力学预测）

**设计动机**: 不设计定制架构，而是**站在 SOTA 点云骨干肩上**蒸馏可规模化的核心原则。

**具体实现**:
- 拼接初始场景点 + 时间堆叠机器人点 → 单一点云 → PTv3 骨干处理 → 共享 MLP 头一次输出整 chunk 内每个场景点的逐步位移。
- 场景点特征来自**冻结 DINOv3**（投影到 2D 多层取特征），机器人点用时间嵌入。
- 分块公式带来极高效推理（批量评估大量候选轨迹仅 0.1s），区别于扩散类像素方法的秒级推理。

#### 模块4: PointWorld for Manipulation（MPC 动作推理）

**具体实现**:
- 把 PointWorld 嵌入 [[MPPI]] 采样式 MPC，规划 $T$ 个 $\mathrm{SE}(3)$ 末端位姿目标。
- 用时间相关（三次样条）噪声采样 $K$ 个动作扰动 $\ell_{1:K}$ 叠到名义轨迹；每条轨迹构造机器人点流动作 $\mathbf{a}_{1:T}^{(\ell)}$，由 PointWorld 滚动出场景流并累积代价 $J^{(\ell)}$。
- 用指数权重 $\omega_\ell\propto\exp(-J^{(\ell)}/\beta)$ 加权平均迭代更新名义轨迹（$\beta$ 为温度）。
- 任务相关点 $\mathcal{I}_{\text{task}}$ 及目标 $\{\mathbf{g}_i\}$ 由人工 GUI 或 [[VLM]] 指定。

### 关键公式与机制

#### 公式1: 多步分块动力学预测

$$
\mathcal{F}^{H}_{\theta}:(\mathbf{s}_t,\ \mathbf{a}_{t:t+H-1})\rightarrow\mathbf{s}_{t+1:t+H}
$$

**含义**: 给定当前状态与未来 $H$ 步动作，一次前向预测未来 $H$ 步状态，提升时间一致性并摊薄计算。

**符号说明**:
- $\mathbf{s}_t\in\mathbf{S}$: 时刻 $t$ 的场景点流状态；$\mathbf{a}_{t:t+H-1}$: 机器人点流动作序列
- $H=10$，每步 0.1s

#### 公式2: 训练目标（运动加权 + 不确定性正则 + Huber 损失）

$$
\frac{1}{2}\sum_{k,i}^{H,N_S} w_{k,i}\Big(\rho_{\delta}\big(\mathbf{\hat{P}}_{t+k,i}-\mathbf{P}_{t+k,i}\big)\,e^{-s_{k,i}}+s_{k,i}\Big)
$$

**含义**: 这是论文唯一的核心训练损失（式 1）。三处下括号分别为：$w_{k,i}$ 运动权重、$\rho_\delta(\cdot)$ 对 3D 残差的 Huber 损失、$e^{-s_{k,i}}$ 不确定性加权与 $s_{k,i}$ 不确定性正则。它解决两大难题：(i) 整场景预测中大多数点静止（仅 1–5% 在动）→ L2 信号极稀疏；(ii) 真实数据噪声大需鲁棒正则。

**符号说明**:
- $w_{k,i}=m_{k,i}/\sum_{k,i}m_{k,i}$: 归一化运动权重；$m_{k,i}=\sigma(\kappa(\delta_{k,i}-\tau))$ 为软运动似然（$\delta_{k,i}$ 为真值位移范数，$\sigma$ 为 logistic，$\tau$/$\kappa$ 为阈值/温度）
- $\rho_\delta$: 逐元素 Huber 损失（$\delta=5.0$）
- $\mathbf{\hat{P}}_{t+k,i},\mathbf{P}_{t+k,i}$: 点 $i$ 在第 $k$ 步的预测/真值位置
- $s_{k,i}$: 模型预测的标量 log-variance（[[Aleatoric Uncertainty|偶然不确定性]]），无需真值监督
- 被 2D 追踪器判为不可见的点在训练中被忽略

#### 公式3: MPC 任务代价（逐点目标代价）

$$
c_{\text{task}}(\mathbf{s}_k)=\frac{1}{|\mathcal{I}_{\text{task}}|}\sum_{i\in\mathcal{I}_{\text{task}}}\|\mathbf{p}_{k,i}-\mathbf{g}_i\|_2^2
$$

**含义**: 把任务目标定义为「任务相关场景点」朝目标位置 $\mathbf{g}_i$ 的逐点距离，对刚体/可变形/铰接物体均适用。

**符号说明**:
- $\mathcal{I}_{\text{task}}\subseteq\{1,\dots,N_S\}$: 任务相关点子集；$\mathbf{g}_i$: 目标位置；$\mathbf{p}_{k,i}$: 第 $k$ 步预测位置

#### 公式4: 全局轨迹优化（MPC 求解）

$$
\arg\min_{\mathbf{E}_{0:T}}\ \sum_{k=1}^{T}\big[\,c_{\text{task}}(\mathbf{s}_k)+c_{\text{ctrl}}(\mathbf{E}_k)\,\big]\quad \text{s.t.}\ \mathbf{s}_{1:T}=\mathcal{F}^{T}_{\theta}(\mathbf{s}_0,\mathbf{a}_{1:T}),\ \mathbf{E}_0=\mathbf{E}_{\text{measured}}
$$

**含义**: 在 PointWorld 动力学约束下，优化末端位姿轨迹 $\mathbf{E}_{0:T}$ 最小化任务代价 + 控制正则，初始位姿固定为测量值。

**符号说明**:
- $c_{\text{ctrl}}$: 路径长度与可达性正则；$\mathbf{E}_k$: 第 $k$ 步末端位姿；$\mathbf{E}_{\text{measured}}$: 当前末端位姿

#### 公式5: 评估指标（移动点 $\ell_2$ 误差）

$$
\ell_{2}=\frac{1}{T}\sum_{t}\frac{1}{|V_t|}\sum_{i\in V_t}e_{t,i}
$$

**含义**: 逐点逐步 $\ell_2$ 距离，聚焦**移动点**（mover）；因大多数点静止，移动点误差最能区分方法的滚动保真度。评估集约 4 万条轨迹 ×1 万点流，标准误可忽略（$\le 10^{-5}$ m），故只报均值。

**符号说明**:
- $V_t$: 时刻 $t$ 的有效（可见）点集；$e_{t,i}$: 点 $i$ 的逐点误差

#### 公式6: 机器人深度对齐损失（外参精修，附录）

$$
L_{\text{robot-depth}}=\frac{1}{K}\sum_{i,t,k}\bigl|d^{\text{obs}}_{i,t,k}-d^{\text{pred}}_{i,t,k}\bigr|
$$

**含义**: DROID 3D 标注流水线中，通过对齐观测深度与已知机器人 mesh 渲染深度，从 VGGT 初始化的相机位姿精修外参。

---

## 关键图表

<!-- 图片均使用 arXiv HTML 在线链接，未本地化 -->

### Figure 1: Teaser / 总览（单图驱动多任务）

![Figure 1](https://arxiv.org/html/2601.03782v1/x1.png)

**说明**: PointWorld 从静态点云 + 本体无关的动作描述（同样表示为 3D 点流）预测整场景 3D 点流。数据集涵盖单臂、双臂、全身、移动操作的真实与仿真交互。预训练后单一模型仅凭一张野外 RGB-D 即可驱动真实硬件完成多样操作，无需额外数据或微调——本图点明了「统一表示 + 规模化 + 零样本部署」三大主张。

### Figure 2: Overview of PointWorld / 方法总览

![Figure 2](https://arxiv.org/html/2601.03782v1/x2.png)

**说明**: 给定标定 RGB-D、机器人关节动作、URDF，先把动作转为机器人点流，与场景点拼接成单一点云作为**本体无关的交互几何**；场景点用冻结 DINOv3 取特征、机器人点用时间嵌入，PTv3 骨干预测整场景 3D 点流。这是理解全方法的关键流程图。

### Figure 3: Rich Supervision of 3D World Modeling / 3D 世界建模的丰富监督

![Figure 3](https://arxiv.org/html/2601.03782v1/x3.png)

**说明**: 在 3D 机器人点流 + 部分可观测 RGB-D 条件下，整场景演化目标提供**稠密像素级监督**，隐式编码操作所需的多种能力：分割目标物、识别材质/铰接结构、为接触推理做隐式形状补全等。论证了「为什么 3D 世界建模目标本身就蕴含了操作智能」。

### Figure 4: Movement Weighting and Uncertainty Regularization / 运动加权与不确定性正则

![Figure 4](https://arxiv.org/html/2601.03782v1/x4.png)

**说明**: 以机器人松手丢下黄布为例。（左下）运动权重 $w_{k,i}$ 由真值流计算，把训练偏向每步真正在动的点；（右下）模型无需真值预测的不确定性值 $s_{k,i}$ 正则化训练，防止对不可靠真值点过拟合。直观佐证式 1 两项设计的必要性。

### Figure 5: 3D Annotation Quality and Comparisons / 3D 标注质量对比

![Figure 5a](https://arxiv.org/html/2601.03782v1/x5.png)
![Figure 5b](https://arxiv.org/html/2601.03782v1/x6.png)

**说明**: FS=FoundationStereo；Dataset Extrinsics V1/V2 为两版 DROID 外参发布。（上）相比 DROID 原始发布，本文流水线产出**显著更高质量的深度与相机位姿标定**，机器人 mask 叠加更准、点云更对齐；（下）进一步用深度重投影等量化对比。说明高质量 3D 标注是模型野外泛化的前提。

### Figure 6: Unseen Rollouts Across Domains / 跨域未见样本滚动预测

![Figure 6](https://arxiv.org/html/2601.03782v1/x7.png)

**说明**: 单一预训练 PointWorld 在多域未见样本上的 10 步点流预测（用 Viser 可视化）。展示首帧预测、末帧预测、末帧真值。绿色点标记 2D 追踪中被遮挡（未被监督）的区域——有趣的是模型在这些点上往往预测**更准**，因为它们未参与训练监督。

### Figure 7: Roadmap for Scaling 3D World Models / 规模化路线图

![Figure 7](https://arxiv.org/html/2601.03782v1/x8.png)

**说明**: 以 DROID 测试集移动点 $\ell_2$ 误差衡量。从已有 GBND 基线出发，逐步「现代化骨干 → 稳定训练目标 → 引入预训练特征 → 扩大模型」，每步都带来一致增益。斜纹柱表示最终未采用的设置。这是全文实验的主线（对应 Table 1 与各消融）。

### Figure 8: Real-World Action Inference / 真实世界动作推理

![Figure 8a](https://arxiv.org/html/2601.03782v1/x9.png)
![Figure 8b](https://arxiv.org/html/2601.03782v1/x10.png)

**说明**: PointWorld 零样本 + MPC 在野外完成刚体（推纸巾盒/书）、可变形（折围巾、放枕头）、铰接（开微波炉、关抽屉，含转动与平移关节）、工具使用（用掸子/扫帚清扫）任务，成功率标于上方。证明预训练模型捕获了可迁移的接触动力学。

### Figure 9: Scaling Study / 规模化研究

![Figure 9](https://arxiv.org/html/2601.03782v1/x11.png)

**说明**: 在 DROID 上分别扫描模型容量（50M–1B）与数据量（5%–100%），每条曲线只变一个轴。log 空间近似线性 → 数据与容量都带来**近 log-linear 的可预测增益**，呼应语言/视觉 scaling law。

### Figure 10: Zero-Shot & Finetuned Generalization to Held-Out Real Scenes / 留出真实场景泛化

![Figure 10](https://arxiv.org/html/2601.03782v1/x12.png)

**说明**: 机器人搬运反光玻璃瓶。在 DROID 或 DROID+BEHAVIOR（D+B）上预训练的 PointWorld 能零样本泛化到某留出实验室的未见环境与运动，逼近用该实验室数据训练的专家模型；纯仿真预训练模型零样本无法泛化；进一步微调可超越专家。

### Figure 11: Action Representations / 动作表示消融

![Figure 11](https://arxiv.org/html/2601.03782v1/x13.png)

**说明**: 把动作表示为**夹爪上的点流**在「有效高效的接触推理」与「异构本体正迁移」间取得平衡。对比五种：夹爪点流（本文）、稀疏全身点流、稠密全身点云(2000 点)、6-DoF 末端位姿、关节位置。在 B1K 上点流胜过低维表示；在噪声大的真实 DROID 上，全身点流反而不如低维基线，而**夹爪点流最优**（既在仿真有效又能跨本体正迁移）。

### Figure 12: Ablation on Chunked Prediction / 分块预测消融

![Figure 12](https://arxiv.org/html/2601.03782v1/x14.png)

**说明**: 对比 teacher-forcing、self-feeding（10k warmup）与滑窗推理（$W{=}1,5$）。**训练与推理都用整 horizon 分块预测漂移最小**，且只需单次前向（自回归需 2–10 次），兼具更准与更省算力。$W{=}1$（等价 self-feeding）退化最严重。

### Figure 13: Ablation on Partial Observability / 部分可观测消融

![Figure 13](https://arxiv.org/html/2601.03782v1/x15.png)

**说明**: 训练 1/2/3 相机或随机相机数的变体，在不同测试相机数下评估。移动点误差保持亚厘米级；训练相机越多测试误差越低；**随机相机数训练的模型在所有测试设置下最鲁棒**，说明暴露于多样可观测性帮助模型在部分可观测下推断 objectness 与物理属性。

### Figure 14–15: DROID 3D Annotations / DROID 3D 标注示例（附录）

![Figure 14](https://arxiv.org/html/2601.03782v1/x16.png)
![Figure 15](https://arxiv.org/html/2601.03782v1/x17.png)

**说明**: DROID 3D 标注示例（机器人叠加 RGB、深度、点云）及与原始数据集的对比，直观展示标注流水线产出的高质量 3D 监督。

### Figure 16–18: DROID Unseen Rollouts / DROID 未见样本滚动（附录）

![Figure 16](https://arxiv.org/html/2601.03782v1/x18.png)
![Figure 17](https://arxiv.org/html/2601.03782v1/x19.png)
![Figure 18](https://arxiv.org/html/2601.03782v1/x20.png)

**说明**: 真实 DROID 上的未见滚动预测，涵盖可变形操作、机器人-物体交互、物体-物体交互、抓取行为、重力效应、玻璃物体等，验证真实域泛化。

### Figure 19–21: BEHAVIOR-1K Unseen Rollouts / B1K 未见样本滚动（附录）

![Figure 19](https://arxiv.org/html/2601.03782v1/x21.png)
![Figure 20](https://arxiv.org/html/2601.03782v1/x22.png)
![Figure 21](https://arxiv.org/html/2601.03782v1/x23.png)

**说明**: 仿真 B1K 上的未见滚动，涵盖受约束的双臂抬举、重力效应（掉落笔记本）、物体-物体交互（笔记本 vs 桌面/篮筐内）、铰接操作（冰箱）、全身行为、隐式形状补全等，验证仿真域的多样动力学建模。

### Table 1: Backbone Comparisons / 骨干网络对比（DROID）

| Backbone | Params | Mem. | FLOPs | Latency (ms) | $\ell_2$ mov.↓ | $\ell_2$ stat.↓ |
|----------|--------|------|-------|--------------|----------------|-----------------|
| GBND | 1.00x | 1.00x | 1.00x | 13.46 | 0.0390 | 0.0066 |
| PointNet | 1.03x | 0.34x | 0.04x | 5.93 | 0.0369 | 0.0084 |
| PointNet++ | 1.07x | 0.67x | 0.06x | 327.08 | 0.0368 | 0.0073 |
| SparseConv | 33.31x | 7.18x | 1.32x | 17.70 | 0.0396 | 0.0076 |
| Transformer | 41.06x | 0.31x | 3.38x | 30.43 | 0.0339 | 0.0071 |
| PTv3-50M | 49.14x | 0.30x | 0.34x | 59.60 | 0.0331 | 0.0067 |
| PTv3-132M | 127.22x | 0.69x | 1.04x | 69.60 | 0.0324 | 0.0061 |
| PTv3-411M | 398.67x | 1.89x | 1.90x | 102.47 | 0.0315 | 0.0059 |
| **PTv3-1B** | **957.71x** | **4.30x** | **3.57x** | **123.65** | **0.0312** | **0.0056** |

**说明**: 参数/显存/FLOPs 相对 GBND 归一。PTv3-1B 把参数扩到 **957×** GBND，但显存仅 4.3×、延迟 ≈0.12s 仍实时，移动点误差从 GBND 0.0390 降到 **0.0312**（最优）。GBND 显存随点数高维特征爆炸、纯局部消息传递在部分可观测下退化，故选 PTv3 作默认骨干。

### Table 2: Generalization & Transfer / 泛化与迁移（$\ell_2$ 误差↓）

D=DROID, B=B1K, H=留出真实场景；"From Scratch"=用留出实验室数据训练的专家；微调用原训练 1/20 的迭代步数。

| Setting | Metric | D→D | B→B | D→B | B→D | D→H | B→H | D+B→H | From Scratch |
|---------|--------|-----|-----|-----|-----|-----|-----|-------|--------------|
| Zero-Shot | $\ell_2$ mover↓ | 0.0315 | **0.0087** | 0.1460 | 0.0558 | 0.0305 | 0.0531 | 0.0300 | 0.0293 |
| Zero-Shot | $\ell_2$ static↓ | 0.0059 | 0.0010 | 0.0050 | 0.0058 | 0.0049 | 0.0057 | 0.0063 | 0.0043 |
| Finetuned | $\ell_2$ mover↓ | – | – | 0.0107 | 0.0378 | **0.0271** | 0.0299 | 0.0272 | 0.0293 |
| Finetuned | $\ell_2$ static↓ | – | – | 0.0003 | 0.0086 | 0.0040 | 0.0046 | – | – |

**说明**: (1) **域内**泛化好（B→B 移动点 0.0087 亚厘米、D→D 与训练相近，说明非记忆）；(2) **跨域**零样本难（D→B 0.1460），但微调（仅 5% 步数）快速逼近从头训练；(3) **留出真实场景**：D→H 零样本 0.0305 已接近专家 0.0293，微调后降到 **0.0271 反超专家**；纯仿真 B→H 零样本不如专家、微调后可比；**D+B 联训零样本(0.0300)略优于 D-only**。整体证明 PointWorld 可零样本迁移到未见真实环境、微调以 20× 更少更新超越专家。

### Table 3: Data Preprocessing and Augmentations / 数据预处理与增广（附录）

| Operation | Description（要点） |
|-----------|----------------------|
| Camera subsampling | 随机采样标定 RGB-D 视角并拼成单一场景点云 |
| Bounds filtering | 仅保留全程位于工作区立方体（约 $[-3,3]^3$ m）内的点 |
| Centering | 对首帧场景与机器人点做均值居中 |
| Image resize | RGB-D 降采样到 $320\times180$ |
| Voxel downsampling | 1.5 cm 体素下采样，$t{=}0$ 每体素一点并对所有帧用同索引 |
| Multi-sphere cropping | 迭代移除远离机器人的点球（最多 3 个，半径 $[0.10,0.80]$ m，buffer 0.25 m） |
| Max scene/robot points | 场景点 >12000 随机下采样；机器人点上限 500 |
| Random yaw | 绕竖直轴 $[-\pi,\pi]$ 均匀旋转 |
| Uniform scaling | 各向同性缩放因子 $[0.9,1.1]$ |
| Random reflection | 0.5 概率沿 x 或 y 轴反射 |
| Chromatic auto-contrast | 0.2 概率自动对比度，混合因子至多 0.2 |
| Chromatic translation | 0.95 概率全局 RGB 偏移幅度 2% |
| Chromatic jitter | 0.95 概率逐点 RGB 噪声标准差 2% |

**说明**: 大量几何与色彩增广是模型野外鲁棒泛化的工程基础。

### Table 4: Per-Point Input Features / 逐点输入特征（附录）

| Point set | Feature | 定义 |
|-----------|---------|------|
| Robot | Position $p^{\text{robot}}_{t,j}$ | 机器人点随时间的 3D 坐标 |
| Robot | Color $c^{\text{robot}}_j$ | 恒定品红 $(1,0,1)$ 标识机器人身份 |
| Robot | Normal $n^{\text{robot}}_{t,j}$ | 来自 URDF 的表面法线 |
| Robot | Gripper openness $\tilde{g}_t$ | 每步夹爪开合标量，广播到所有机器人点 |
| Robot | Velocity $v^{\text{robot}}_{t,j}$ | 由位置中点差分得到的逐点速度 |
| Robot | Acceleration $a^{\text{robot}}_{t,j}$ | 由速度中点差分得到的逐点加速度 |
| Scene | Position $x_{0,i}$ | 预处理后首帧场景点 3D 坐标 |
| Scene | Color $c^{\text{scene}}_{0,i}$ | 首帧场景点 RGB |
| Scene | Normal $n^{\text{scene}}_{0,i}$ | 首帧（估计）表面法线 |
| Scene | Gripper openness seq $g_{0:T-1}$ | 整个上下文+预测窗的夹爪开合序列，广播到每个场景点 |
| Scene | Distance-to-robot $d_{0:T-1,i}$ | 每步场景点首帧位置到最近机器人点的距离 $d_{t,i}=\min_j\|x_{0,i}-r_{t,j}\|_2$ |

**说明**: 机器人/场景两类点各自的输入特征构成，dist-to-robot 提供接触距离线索。

### Table 5: Training Configuration for PointWorld-1B / PointWorld-1B 训练配置（附录）

| Setting | Value |
|---------|-------|
| Optimizer | AdamW |
| Learning rate | $1\times10^{-4}$ |
| Epochs | 300 |
| Weight decay | $10^{-2}$ |
| Global batch size | 1920 sequences |
| Gradient clipping | global $\ell_2$ norm ≤ 5 |
| Loss | Huber ($\delta=5.0$) + movement weighting + aleatoric uncertainty |
| Prediction horizon | 10 steps |
| Training GPUs | **128 NVIDIA H100** |
| Training time | **20 days** |
| Grid size | 1.5 cm |
| Encoder depth | (4, 4, 8, 8, 12, 12, 4) |
| Encoder channels | (256, 384, 384, 512, 512, 768, 1024) |
| Encoder heads | (8, 12, 12, 16, 16, 24, 32) |
| Encoder stride | (1, 2, 2, 2, 2, 2, 2) |
| Decoder depth | (4, 4, 4, 4, 4, 4) |
| Decoder channels | (256, 384, 384, 512, 512, 768) |

**说明**: PointWorld-1B 用 128×H100 训 20 天的 PTv3 U-Net 配置（仿真专用变体则为 200 epoch、batch 176、8×H100、7 天，encoder/decoder 更浅更窄）。可见 1B 规模 3D 世界模型的训练成本相当可观。

---

## 实验

### 数据集 / 基准

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| [[DROID]]（真实） | 自定义流水线恢复 >60%（约 200 小时人类遥操作） | 单臂 Franka，野外双外部相机+腕部相机；自建 3D 标注（FoundationStereo 深度 + VGGT 精修外参 + CoTracker3 追踪） | 训练/测试 |
| [[BEHAVIOR-1K]]（仿真，B1K） | 约 1100 小时遥操作（过滤前） | 双臂/全身/移动操作，逼真家居场景；用仿真特权状态得真值点流，按活跃接触+非零物体运动过滤 | 训练/测试 |
| 合计 | **约 2M 轨迹 / 500 小时** | 单臂、双臂、全身交互，真实+仿真 | 最大规模 3D 动力学数据集（全开源） |

### 实现细节

- **Backbone**: PTv3（最高 PTv3-1B），场景点用冻结 DINOv3 多层特征，机器人点用时间嵌入
- **训练目标**: 式 1，Huber($\delta=5.0$) + 运动加权 + 偶然不确定性正则；忽略 2D 追踪判定不可见的点
- **优化**: AdamW，lr $1\times10^{-4}$，weight decay $10^{-2}$，grad clip $\ell_2\le5$；PointWorld-1B：300 epoch、batch 1920、horizon 10
- **硬件**: 128×NVIDIA H100，训练 20 天；推理 **0.1s/前向（实时）**
- **真实部署**: Franka 装于轮式底座 + 单 RealSense D435，FoundationStereo 估深度，人工 GUI 画 mask 与目标点，每次优化滚动 30 步（3 次自回归前向），MPPI 求解

### 关键实验结论

- **规模化路线（5.1, Fig 7 / Table 1）**: 现代骨干（PTv3）+ 稳定目标（运动加权/不确定性/Huber）+ 预训练 DINOv3 特征 + 模型放大，每步都带来一致增益；PTv3-1B 移动点误差 0.0312 全面最优。
- **动作表示（5.2, Fig 11）**: 夹爪点流最优，兼顾接触推理与跨本体正迁移；噪声真实域里全身点流反而淹没稀疏学习信号。
- **分块预测（5.2, Fig 12）**: 训练+推理都整 horizon 分块，漂移最小且单次前向最省算力。
- **部分可观测（5.2, Fig 13）**: 随机相机数训练最鲁棒，更多相机一致降误差。
- **Scaling law（5.2, Fig 9）**: 模型 50M→1B、数据 5%→100% 均近 log-linear 增益。
- **泛化迁移（5.3, Table 2）**: 域内泛化好、跨域零样本难但微调（5% 步数）速达专家水平、留出真实场景零样本逼近专家、微调反超、D+B 联训零样本略优。
- **真实部署（5.4, Fig 8）**: 单 checkpoint 零样本 + MPC 完成刚体推/可变形/铰接/工具使用，无需任何示范或后训练。

---

## 批判性思考

### 优点

1. **表示统一带来的优雅与可规模化**: 把状态与动作都放进 3D 点流，使异构本体（单臂 Franka 与双臂 humanoid）数据能在同一模型里联合学习，且推理无需点追踪器；这是对「世界模型该用什么模态」的有力回答。
2. **实证扎实、可复现配方明确**: 不发明定制网络，而是系统性 ablate 骨干、动作表示、目标、可观测性、数据混合、scaling law，并把每个结论用 Figure 7 路线图与 Table 1/2 串成清晰叙事；数据/代码/权重承诺开源。
3. **零样本真实部署 + 实时性**: 0.1s 推理天然契合采样式 MPC，单一预训练 checkpoint 无需示范即可做刚体/可变形/铰接/工具任务，工程含金量高；自建 DROID 3D 标注流水线（中位平移/旋转误差 1.8cm/1.9°）本身也是可复用贡献。

### 局限性

1. **代价高且缺成功率级硬指标**: 1B 模型需 128×H100 训 20 天；主体评估用 $\ell_2$ 移动点误差（作者也承认绝对差异看起来「modest」），真实任务仅给 Figure 8 成功率，缺与基线 policy 的系统性任务级成功率对比。
2. **依赖假设较多**（附录 A.1 自陈）: 静态初始状态、需人工/VLM 指定任务点与目标、刚体机器人假设、缺乏显式物理先验、无光度动力学（不建模光照/外观变化）、对精细物体与标定噪声敏感、相关 vs 因果存疑。
3. **跨域零样本仍弱**: 仿真→真实零样本基本失效（B→H 不如专家、D→B 误差 0.1460），真正「野外即用」仍偏向真实数据预训练域内，泛化边界未充分压测。

### 潜在改进方向

1. 引入显式物理先验/可微分接触，或与视频/光度信号融合，缓解「无光度动力学、缺物理先验」的短板。
2. 把任务点指定自动化（更强 VLM/grounding），并补充与 VLA/扩散 policy 的任务级成功率对比，让「世界模型→操作」的收益更可量化。
3. 进一步压测 sim-to-real 零样本与更长 horizon、可变形/流体等更难动力学，验证表示与 scaling law 的上限。

### 可复现性评估

- [x] 代码开源（论文声明 Code/dataset/checkpoints 将开源；主页 https://point-world.github.io/）
- [x] 预训练模型（声明开源 checkpoints）
- [x] 训练细节完整（附录 Table 3/4/5 给出增广、特征、超参与 PTv3 结构）
- [x] 数据集可获取（DROID/BEHAVIOR-1K 公开；自建 3D 标注承诺开源）

---

## 速查卡片

> [!summary] PointWorld: Scaling 3D World Models for In-The-Wild Robotic Manipulation
> - **核心**: 状态与动作统一为 3D 点流，预训练大模型预测「动作条件下整场景 3D 点位移」，零样本 + MPC 驱动真实机器人。
> - **方法**: RGB-D 反投影场景点（冻结 DINOv3 特征）+ URDF 前向运动学生成的机器人点流 → 拼接成单一点云 → PTv3-1B 骨干一次预测 $H=10$ 步位移；损失 = Huber + 运动加权 + 偶然不确定性正则；MPPI/MPC 做动作推理。
> - **数据**: DROID（真实，自建 3D 标注）+ BEHAVIOR-1K（仿真），约 2M 轨迹/500 小时（最大规模 3D 动力学数据集）。
> - **结果**: PTv3-1B 移动点 $\ell_2$ 0.0312 最优；scaling 近 log-linear；零样本迁移留出真实场景逼近/微调反超专家；单 checkpoint 真实 Franka 完成刚体/可变形/铰接/工具任务，0.1s 实时；128×H100 训 20 天。
> - **主页**: https://point-world.github.io/

---

*笔记创建时间: 2026-06-29*
