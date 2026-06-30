---
title: "PALM: Progress-Aware Policy Learning via Affordance Reasoning for Long-Horizon Robotic Manipulation"
method_name: "PALM"
authors: [Yuanzhe Liu, Jingyuan Zhu, Yuchen Mo, Gen Li, Xu Cao, Jin Jin, Yifan Shen, Zhengyuan Li, Tianjiao Yu, Wenzhen Yuan, Fangqiang Ding, Ismini Lourentzou]
year: 2026
venue: CVPR
tags: [VLA, long-horizon, affordance, progress-estimation, inverse-dynamics, diffusion-transformer]
zotero_collection: _inbox
image_source: online
arxiv_html: https://arxiv.org/html/2601.07060v2
created: 2026-06-29
---

# PALM: Progress-Aware Policy Learning via Affordance Reasoning for Long-Horizon Robotic Manipulation

## 元信息

| 项目 | 内容 |
|------|------|
| 作者 | Yuanzhe Liu, Jingyuan Zhu, Yuchen Mo, Gen Li, Xu Cao, Jin Jin, Yifan Shen 等 12 人 |
| 机构 | PLAN Lab（项目主页署名） |
| 会议 | CVPR 2026 |
| 类别 | VLA（Vision-Language-Action）/ 长程操作 / 可供性推理 |
| 日期 | 2026-01（arXiv v2） |
| 项目主页 | https://plan-lab.github.io/palm |
| 链接 | [arXiv](https://arxiv.org/abs/2601.07060) / [Project](https://plan-lab.github.io/palm) |

---

## 一句话总结

> 在 [[VLA]] 内部加入"结构化未来可供性预测 + 连续子任务进度估计"双机制，用可学习查询前瞻四类可供性并联合扩散策略输出动作与进度，显著稳住长程多步操作。

---

## 核心贡献

1. **统一的进度感知长程操作框架**: 提出 **PALM**，一个端到端 [[VLA]] 框架，把感知-动作-进度组织成闭环，围绕**以交互为中心的可供性推理**与**子任务进度线索**来构建策略学习，专治长程任务中的重复动作、漏步、过早终止等失败模式。
2. **细粒度可供性预测器（隐式中间推理）**: 用可学习查询前瞻 **四类互补可供性**——Global（物体相关性）、Local（接触几何）、Spatial（放置区域）、Dynamic（运动轨迹），作为视觉运动控制的任务相关"锚点"，构成对演化场景状态的紧致表征。
3. **进度感知逆动力学模块**: 把动作生成与**连续子任务进度估计**耦合，用 [[Diffusion Transformer|DiT]] 联合解码动作序列与逐步进度标量，实现时序一致的长程执行与无缝子任务切换。在 [[LIBERO]]-LONG 上 **91.8%** 成功率、[[CALVIN]] ABC→D 平均长度提升 **12.5%**、真实世界三类泛化设置下相对基线 **约 2 倍** 提升。

---

## 问题背景

### 要解决的问题

当前 [[VLA]] 模型在**长程、多步**操作上根本受限：在"清理杂乱桌面"这类任务上，SOTA 策略往往开局成功却中途失败，无法可靠完成完整序列。如何让策略在**异质动态的子任务之间保持时序一致**，并避免典型长程失败模式，是本文的核心目标。

### 现有方法的局限

作者指出两类根本缺失：

1. **缺乏结构化可供性线索（affordance cues）与显式状态跟踪**: 现有模型即便能推断最终目标、产出中间动作，也**没有内部表征**去消歧"下一步该瞄准哪个物体 / 哪个部件区域可交互 / 物体该放到哪 / 下一步合适的运动是什么"。结果是大量视觉相似的状态变得模糊，掩盖了底层任务阶段，破坏长程控制稳定性。
2. **缺乏子任务内的进度连续估计**: 没有在线的"推进程度"概念，策略就无法可靠判断**继续、切换阶段还是终止**。这种时序锚定的缺失导致特征性的长程失败：重复/多余动作、跳过必需子任务、过早终止，甚至在错误状态下宣告成功。

此外，把子策略**孤立训练**时，一个子任务的终止状态常偏离下一个子任务期望的起始状态，导致切换不稳定；而直接拟合 $\pi(a_t\mid o_t)$ 的常规行为克隆会把不同阶段"压扁"，引发 rollout 时的重复或漏步。

### 本文的动机

把"该在哪里动作（where）"与"已经推进到哪（how far）"两件事都显式建模并耦合进策略：
- 用 **结构化未来可供性** 作为隐式中间推理步骤，给视觉运动控制提供任务相关的稳定锚点；
- 用 **连续进度标量** 作为时序正则项，消歧阶段、平滑子策略边界，**不依赖单独的规划器或层级控制器**。

---

## 方法详解

### 模型架构

PALM 采用 **多模态编码 → GPT-2 风格主干 + 可学习查询 → 双解码器** 的闭环结构（见 Figure 2）：

- **输入**: 语言指令 $l$ + 图像观测 $o_t$ + 机器人状态 $s_t$
- **多模态编码器**:
  - 文本用 [[CLIP]] text encoder 编码；
  - 图像用 [[MAE]]-预训练的 [[ViT]]-B 提取 patch 特征，再经 [[Perceiver Resampler]] 下采样为紧致、任务相关的视觉 token；
  - 状态经轻量 MLP 投影为单一状态 token。
- **Backbone**: [[GPT-2]] 风格 Transformer，用**因果 + 跨模态注意力**融合 token 序列。
- **核心模块**: 两组可学习查询——
  - [[Affordance Query|细粒度可供性查询]]（含 `<Global>`/`<Local>`/`<Spatial>`/`<Dynamic>` 四个子查询）；
  - [[Action-Progress Query|动作-进度查询]]，条件于可供性潜变量 $\hat{\mathbf F}_{t+n}$ 做进度感知逆动力学解码。
- **输出**: [[Diffusion Transformer|DiT]] 联合解码长度为 $n$ 的动作块 $\hat a_{t:t+n-1}$ 与逐步进度 $\hat p_{t:t+n-1}$。
- **可训练参数**: 仅 **68M**，轻量到可在 RTX 4090 上微调。

### 核心模块

#### 模块1: 细粒度可供性预测（Fine-Grained Affordance Prediction）

**设计动机**: 在动作之前先以**前瞻的方式**预测未来 $t+n$ 时刻的交互线索，把它作为任务相关的中间表征来稳定感知。四个子查询通过 **block-wise 结构化注意力** 解耦，各自捕获互补尺度的可供性。

**具体实现**（四类可供性，均预测未来 $t+n$ 时刻）:
- **Global（全局）**: 高层语义先验，确定指令所指物体及其大致区域。监督目标用 [[Grounding DINO]] 解析 referent + [[SAM]] 分割实例得到二值 mask，再在冻结图像编码器上做 masked pooling 取物体特征。损失 = [[Focal Loss]] + [[Dice Loss]]（公式3）。
- **Local（局部）**: 在全局先验高亮区域内分析边缘/纹理/部件几何，预测稠密**接触似然热图**。沿用 [[GLOVER++]] 把标注接触点转为高斯热图作连续监督。损失 = Focal + KL（公式4）。
- **Spatial（空间）**: 把欠定的空间语言转成可执行**放置提案**——预测一小组候选放置点而非单一坐标。目标用 [[SpatialVLM]] 转空间语义 + [[RoboPoint]] 采样 2D 坐标。损失为集合匹配（公式6）。
- **Dynamic（动态）**: 识别夹爪与可动物体像素并预测其随时间演化。监督用 **grid-based tracking**（首帧布 $N\times N$ 查询点，[[CoTracker]] 前向跟踪，累计位移超阈值者保留为真实运动），栅格化为动态区域。损失为基于隐变量模型的掩码重建（公式7，含 KL 正则）。

> 推理时**移除可供性头**，仅保留可供性潜变量前瞻 $\hat{\mathbf F}_{t+n}$ 去条件化动作策略（见 Figure 2(a)）。

#### 模块2: 进度感知逆动力学策略（Progress-aware Policy via Inverse Dynamics）

**设计动机**: 除"在哪里动作"外，引入**进度感知预测**估计"当前子任务推进了多少"。视觉相似的观测在不同阶段对应不同动作，进度标量 $p_t$ 提供"我们在哪"的连续指示，鼓励潜状态**单调、阶段一致**地演化，平滑子策略边界。

**具体实现**:
- 先从可供性潜变量推断当前活跃子任务阶段并得到 stage embedding，条件于它预测 $p_t\in[0,1]$，并**追加到动作输出**，使策略联合预测 $(a_t,p_t)$。
- 把经典逆动力学（由 $(o_t,o_{t+1})$ 预测 $\hat a_t$）扩展为 **$n$ 步动作-进度序列预测**，条件于当前输入与单步可供性潜变量。
- 实例化 $f_{\mathrm{inv}}$ 为 [[Diffusion Transformer|DiT]]（公式8），通过反转高斯加噪过程联合建模动作分布与进度值（公式9 加噪、公式10 标准扩散损失）。
- 进度阈值 $\phi$ 作为子策略切换的决策边界：$p_t\ge\phi$ 触发下一阶段，默认 $\phi=90\%$（见 Table 9 消融）。

### 关键公式与机制

#### 公式1: [[Affordance Query|可供性前瞻]]

$$
\hat{\mathbf{F}}_{t+n}=f_{\mathrm{aff}}\!\left(l,o_{t},s_{t}\right)\in\mathcal{F}
$$

**含义**: 由四个可供性子查询在多模态上下文上前瞻，输出未来 $t+n$ 时刻的解耦可供性潜变量 $\hat{\mathbf F}_{t+n}$，作为下游控制的任务相关锚点。

**符号说明**:
- $l,o_t,s_t$: 语言指令、图像观测、机器人状态
- $f_{\mathrm{aff}}:\mathcal O\times\mathcal T\to\mathcal F$: 可供性预测映射
- $\hat{\mathbf F}_{t+n}$: 未来偏移 $t+n$ 处的可供性中心潜变量

#### 公式2: 动作-进度联合解码

$$
(\hat{a}_{t:t+n-1},\hat{p}_{t:t+n-1})=\operatorname{DiT}\!\left(l,o_{t},s_{t},\hat{\mathbf{F}}_{t+n}\right)
$$

**含义**: 动作头 [[Diffusion Transformer|DiT]] 条件于多模态输入与可供性潜变量，一次生成长度为 $n$ 的动作轨迹与逐步进度标量。

**符号说明**:
- $\hat a_{t:t+n-1}$: 预测动作序列；$\hat p_{t:t+n-1}$: 各步标量进度

#### 公式3: Global 可供性损失

$$
\mathcal{L}_{\text{global}}=\mathcal{L}_{\text{FL}}\!\left(\mathcal{M}_{t+n}^{\text{global}},\hat{\mathcal{M}}_{t+n}^{\text{global}}\right)+\mathcal{L}_{\text{Dice}}\!\left(\mathcal{M}_{t+n}^{\text{global}},\hat{\mathcal{M}}_{t+n}^{\text{global}}\right)
$$

**含义**: 监督预测的未来全局可供性 mask 与目标二值 mask，兼顾像素级判别（Focal）与区域重叠（Dice）。

**符号说明**:
- $\hat{\mathcal M}_{t+n}^{\text{global}}=f_{\text{global}}(l,o_t,s_t)$: 预测全局 mask；$\mathcal M_{t+n}^{\text{global}}$: 目标二值 mask
- $\mathcal L_{\text{FL}}$: 在图像域 $\Omega$ 上的像素级 [[Focal Loss]]；$\mathcal L_{\text{Dice}}$: 软 [[Dice Loss]]

#### 公式4: Local 可供性损失 与 热图归一化

$$
\mathcal{L}_{\text{local}}=\mathcal{L}_{\text{FL}}\!\left(\mathcal{M}_{t+n}^{\text{local}},\hat{\mathcal{M}}_{t+n}^{\text{local}}\right)+\mathcal{L}_{\text{KL}}\!\left(\tilde{\mathcal{M}}_{t+n}^{\text{local}},\tilde{\hat{\mathcal{M}}}_{t+n}^{\text{local}}\right)
$$

$$
\tilde{\mathcal{M}}=\frac{\mathcal{M}}{\sum_{(i,j)\in\Omega}\mathcal{M}^{(i,j)}+\varepsilon}
$$

**含义**: 局部接触似然热图用 Focal 对齐峰值、用 KL 对齐分布；$\tilde{\mathcal M}$ 为 $\ell_1$ 归一化的热图（KL 项使用），$\varepsilon>0$ 保数值稳定。

**符号说明**:
- $\hat{\mathcal M}_{t+n}^{\text{local}}$: 预测局部接触似然图；$\mathcal M_{t+n}^{\text{local}}$: 软高斯目标热图

#### 公式5: Spatial 集合匹配损失

$$
\mathcal{L}_{\text{spatial}}=\frac{1}{C_{t+n}}\sum_{c=1}^{C_{t+n}}\min_{1\leq m\leq M}\left\|\hat{\mathbf{p}}_{t+n}^{(m)}-\mathbf{p}_{t+n}^{(c)}\right\|_{2}^{2}
$$

**含义**: 对每个目标放置点找最近的预测候选点做对齐，避免记忆单一坐标、提升对布局变化的鲁棒性。

**符号说明**:
- $\hat{\mathcal S}_{t+n}=\{\hat{\mathbf p}_{t+n}^{(m)}\}_{m=1}^{M}$: $M$ 个预测的归一化 2D 候选点
- $\mathcal S_{t+n}=\{\mathbf p_{t+n}^{(c)}\}_{c=1}^{C_{t+n}}$: $C_{t+n}$ 个目标放置点

#### 公式6: Dynamic 可供性损失（隐变量掩码重建）

$$
\begin{aligned}
\mathcal{L}_{\text{dynamic}}={}&\mathbb{E}_{\mathbf{z}\sim Q_{\phi}\left(\mathbf{z}\mid x_{t+n}^{\mathcal{M}}\right)}\left[-\log P_{\psi}\left(x_{t+n}^{\mathcal{M}}\mid\mathbf{z}\right)\right]\\
&+\beta\operatorname{KL}\!\Big(Q_{\phi}\left(\mathbf{z}\mid x_{t+n}^{\mathcal{M}}\right)\,\|\,p(\mathbf{z})\Big)
\end{aligned}
$$

**含义**: 对动态区域（夹爪/可动物体）的未来帧做掩码重建，VAE 式重建项 + KL 正则。

**符号说明**:
- $\mathcal M_{t+n}$: 来自跟踪的动态区域 mask；$x_{t+n}^{\mathcal M}$: 限制在该 mask 上的未来帧
- $Q_\phi,P_\psi$: 编码/解码分布；$\beta$: KL 权重

#### 公式7: 进度感知逆动力学

$$
\left(\hat{a}_{t:t+n-1},\hat{p}_{t:t+n-1}\right)=f_{\mathrm{inv}}\!\left(l,o_{t},s_{t},\hat{\mathbf{F}}_{t+n}\right)
$$

**含义**: 把经典逆动力学扩展为 $n$ 步动作-进度序列预测，条件于当前输入与单步可供性潜变量。

#### 公式8: 扩散前向加噪

$$
\tilde{\mathbf{y}}_{t:t+n-1,t_{d}}=\sqrt{\bar{\alpha}_{t_{d}}}\,\mathbf{y}_{t:t+n-1}+\sqrt{1-\bar{\alpha}_{t_{d}}}\,\boldsymbol{\epsilon}
$$

**含义**: 标准扩散前向，把目标动作-进度向量按调度 $\bar\alpha_{t_d}$ 加噪。

**符号说明**:
- $\mathbf y_{t:t+n-1}$: 目标动作-进度向量；$\boldsymbol\epsilon\sim\mathcal N(\mathbf 0,\mathbf I)$: 高斯噪声
- $t_d$: 扩散时间；$\bar\alpha_{t_d}$: 累计噪声调度

#### 公式9: 扩散去噪损失

$$
\mathcal{L}_{\text{DiT}}=\mathbb{E}_{t_{d},\boldsymbol{\epsilon}}\big\|\boldsymbol{\epsilon}-\epsilon_{\theta}\!\left(\tilde{\mathbf{y}}_{t:t+n-1,t_{d}}\,\middle|\,l,o_{t},s_{t},\hat{\mathbf{F}}_{t+n},t_{d}\right)\big\|_{2}^{2}
$$

**含义**: 训练噪声预测网络 $\epsilon_\theta$ 在多模态条件与可供性潜变量下预测加入的噪声，从而联合建模动作分布与进度值。

**符号说明**:
- $\epsilon_\theta(\cdot)$: 噪声预测器；$\tilde{\mathbf y}_{t:t+n-1,t_d}$: 加噪目标

---

## 关键图表

<!-- 图片均使用 arXiv HTML 在线链接，未本地化 -->

### Figure 1: Motivation / 与 vanilla VLA、预测式方法的对比

![Figure 1 left](https://arxiv.org/html/2601.07060v2/x1.png)
![Figure 1 right](https://arxiv.org/html/2601.07060v2/x2.png)

**说明**: 与"直接把输入映射到动作"的 vanilla [[VLA]]、或"前瞻稠密未来图像"的预测式方法不同，PALM 用**可学习查询前瞻一组结构化未来可供性**；以这些可供性为条件，基于扩散的策略**联合解码动作与连续进度值**，实现时序状态跟踪与无缝子任务切换。这是全文核心范式的图示对照。

### Figure 2: PALM Overview / 整体架构与结构化注意力

![Figure 2](https://arxiv.org/html/2601.07060v2/x3.png)

**说明**: (a) **模型架构**——三模态经冻结编码器得到 text/visual/state token，由 GPT 式单向注意力 Transformer 融合，配两组查询：细粒度可供性 与 动作-进度。训练时可供性查询前瞻 $\hat{\mathbf F}_{t+n}$，由 `<Global>`/`<Local>`/`<Spatial>`/`<Dynamic>` 四个监督头落地；**推理时移除可供性头**，动作-进度查询同时关注上下文与可供性前瞻，条件化 DiT 输出动作 $\hat a$ 与进度 $\hat p$ 轨迹。(b) **结构化注意力**——可供性子查询仅关注共享上下文 token 以保持解耦，两类查询都用因果注意力以保持时序一致。这张图是理解 PALM 训练/推理差异（"训练有监督头、推理只留潜变量"）的关键。

### Figure 3: Affordance Components Ablation / 四类可供性组件消融

![Figure 3](https://arxiv.org/html/2601.07060v2/x4.png)

**说明**: 在 [[CALVIN]] ABCD 与 [[LIBERO]]-LONG 上累加 Global→Local→Spatial→Dynamic 的成功率曲线。加 Global 即稳定提升；加 Local 在 CALVIN 增益但 LIBERO-LONG 略降（视角导致的边缘几何偏差）；加 Spatial 恢复并进一步提升（布局鲁棒的放置先验）；加 Dynamic 至完整 PALM 取得最优，说明运动线索 + 结构化空间推理最可靠。佐证四类可供性互补且缺一不可。

### Figure 4: Real-World Setup & Task Design / 真实平台与任务设计

![Figure 4](https://arxiv.org/html/2601.07060v2/x5.png)

**说明**: 左：UFACTORY xArm6 + Gripper G2 + 两路 RealSense D455（eye-on-hand 与 eye-on-base）。右：单一高层指令驱动的 **6 个连续子任务**长程操作任务（依次拾放三种水果到指定容器），用于真实世界长程泛化评测。

### Figure 5: Random Relocation Disturbances / 随机重定位扰动下进度

![Figure 5](https://arxiv.org/html/2601.07060v2/x6.png)

**说明**: "pick up grape" 子任务中两次随机重定位葡萄（竖虚线）。预测进度平滑上升，重定位处出现瞬时偏移后快速恢复到原上升趋势，**不重置不崩塌**——表明进度信号对目标位姿大幅变化鲁棒，跟踪的是子任务完成度而非瞬时几何构型。

### Figure 6: Unseen Lighting Disturbances / 未见光照扰动下进度

![Figure 6](https://arxiv.org/html/2601.07060v2/x7.png)

**说明**: 同一子任务下两次切到未见光照（竖虚线），全局亮度/阴影剧变但物理场景不变。预测进度持续上升、仅在变化点轻微波动后回归，体现对剧烈光度扰动的鲁棒性。

### Figure 7: Multi-Object Visual Distractions / 多物体视觉干扰下进度

![Figure 7](https://arxiv.org/html/2601.07060v2/x8.png)

**说明**: 两次在目标附近注入多个干扰物制造杂乱（竖虚线）。预测进度持续上升、干扰处局部偏移后迅速恢复，说明进度估计对多物体干扰不敏感。Figure 5-7 共同验证进度信号锚定的是底层任务状态而非虚假视觉相关。

### Figure 8: Affordance Predictions Visualization / 四类可供性预测可视化

![Figure 8](https://arxiv.org/html/2601.07060v2/x9.png)

**说明**: 沿连续进度步可视化四类可供性：Global 分割任务相关物体与目标；Local 生成精确接触点热图；Spatial 预测候选放置区域；Dynamic 前瞻运动轨迹。直观展示可供性如何引导策略生成。

### Table 1: CALVIN ABCD Results / CALVIN ABC→D 长程结果

| Method | Type | 1 | 2 | 3 | 4 | 5 | Avg. Len. ↑ |
|--------|------|---|---|---|---|---|-------------|
| RT-1 | Autoregressive | 53.3% | 22.2% | 9.40% | 3.80% | 1.30% | 0.90 |
| Robo-Flamingo | Autoregressive | 82.4% | 61.9% | 46.6% | 33.1% | 23.5% | 2.47 |
| OpenVLA | Autoregressive | 91.3% | 77.8% | 62.0% | 52.1% | 43.5% | 3.27 |
| Diffusion Policy | Diffusion-based | 40.2% | 12.3% | 2.60% | 0.80% | 0.00% | 0.56 |
| $\pi_0$ | Diffusion-based | 93.8% | 85.0% | 76.7% | 68.1% | 59.9% | 3.92 |
| 3D-VLA | 3D-Aware | 44.7% | 16.3% | 8.10% | 1.60% | 0.00% | 0.71 |
| 3D Diffuser Actor | 3D-Aware | 92.2% | 78.7% | 63.9% | 51.2% | 41.2% | 3.27 |
| RoboUniview | 3D-Aware | 94.2% | 84.2% | 73.4% | 62.2% | 50.7% | 3.65 |
| Susie | Prediction | 87.0% | 69.0% | 49.0% | 38.0% | 26.0% | 2.69 |
| GR-1 | Prediction | 85.4% | 71.2% | 59.6% | 49.7% | 40.1% | 3.06 |
| Seer | Prediction | 94.4% | 87.2% | 79.9% | 72.2% | 64.3% | 3.98 |
| PALM (✗ progress) | Prediction | 95.3% | 85.6% | 79.5% | 74.3% | 67.0% | 4.02 |
| **PALM** | **Prediction + Progress** | **96.9%** | **93.8%** | **89.3%** | **85.9%** | **82.0%** | **4.48** |

**说明**: PALM 全指标 SOTA，5 连续子任务成功率 82.0%，比最强基线 Seer(64.3%) 高 **+17.7%**；平均长度 4.48 远超 Seer(3.98)、$\pi_0$(3.92)。去掉进度预测（✗ progress）平均长度从 4.48 掉到 4.02，证明**进度感知是长程泛化的关键**。

### Table 2: LIBERO Results / LIBERO 四套件结果

| Method | Average ↑ | Spatial ↑ | Object ↑ | Goal ↑ | Long ↑ |
|--------|-----------|-----------|----------|--------|--------|
| OpenVLA | 76.5±0.6% | 84.7±0.9% | 88.4±0.8% | 79.2±1.0% | 53.7±1.3% |
| Diffusion Policy | 72.4±0.7% | 78.3±1.1% | 92.5±0.7% | 68.3±1.2% | 50.5±1.3% |
| Octo fine-tuned | 75.1±0.6% | 78.9±1.0% | 85.7±0.9% | 84.6±0.9% | 51.1±1.3% |
| SpatialVLA | 69.0±1.2% | 88.2±0.7% | 89.9±1.3% | 78.6±0.9% | 55.5±1.5% |
| CoT-VLA | 81.1±0.6% | 87.5±1.4% | 91.6±0.5% | 87.6±0.6% | 69.0±0.8% |
| TraceVLA | 74.8±0.9% | 84.6±1.0% | 85.2±0.6% | 75.1±1.4% | 54.1±1.0% |
| CoA-VLA | 79.8±0.5% | 85.3±0.9% | 93.1±1.0% | 85.8±0.9% | 55.0±1.2% |
| **PALM** | **94.5±1.0%** | **95.2±1.2%** | **96.7±0.7%** | **94.3±1.6%** | **91.8±0.8%** |

**说明**: 四套件全面 SOTA，平均 94.5%。最大增益在 **LIBERO-LONG**：91.8% 比最强基线 CoT-VLA(69.0%) 高 **+22.8%**，凸显长程优势。

### Table 3: PALM Components Ablation / 核心模块消融

| Ablation Type | Pre-training Avg. Len. ↑ | Fine-tuning Avg. Len. ↑ |
|---------------|--------------------------|-------------------------|
| **PALM** | **4.48** | **4.48** |
| ✗ Affordance Foresight | 3.90 | 3.58 |
| ✗ Inverse Dynamic Prediction | 4.17 | 3.92 |
| ✗ Progress Prediction | 3.73 | 4.02 |

**说明**: 在 CALVIN ABCD 上分别于预训练/微调阶段移除三大组件。**微调阶段去可供性前瞻降幅最大**（4.48→3.58），说明结构化可供性对下游精确长程规划必不可少；**预训练阶段去进度预测降幅最大**（4.48→3.73），说明大规模长程数据对学到强进度先验尤为重要；去逆动力学两阶段都掉，三者互补。

### Table 4: Training Data Composition Ablation / 训练数据构成消融

| Ablation Type | CALVIN ABCD Avg. Len. ↑ | LIBERO-LONG SR (%) ↑ |
|---------------|-------------------------|----------------------|
| **PALM** | **4.48** | **91.8** |
| ✗ In-the-Wild Data | 3.90 | 73.5 |
| ✗ Long-Horizon Video Data | 3.73 | 84.5 |
| ✗ Human Annotated Data | 3.58 | 76.5 |
| ✗ Simulation Data (Pretrain) | 3.96 | 81.0 |

**说明**: 去任一数据类型都降级，**In-the-Wild 与 Human-Annotated 数据**影响最大；去掉对应仿真基准的预训练数据也明显下降，体现各数据源的数据效率与互补性。

### Table 5: Real-World Generalization Results / 真实世界长程泛化（6 子任务，Avg.Len. 满分 6）

| Type | Method | 1 | 2 | 3 | 4 | 5 | 6 | Avg. Len. ↑ |
|------|--------|---|---|---|---|---|---|-------------|
| Random Localization | OpenVLA | 0.45 | 0.30 | 0.15 | 0.05 | 0.00 | 0.00 | 0.95 |
| | Octo | 0.35 | 0.20 | 0.10 | 0.00 | 0.00 | 0.00 | 0.65 |
| | **PALM** | **0.70** | **0.65** | **0.55** | **0.45** | **0.40** | **0.30** | **3.05** |
| Visual Distraction | OpenVLA | 0.65 | 0.50 | 0.25 | 0.15 | 0.05 | 0.00 | 1.60 |
| | Octo | 0.45 | 0.35 | 0.15 | 0.00 | 0.00 | 0.00 | 0.95 |
| | **PALM** | **0.85** | **0.80** | **0.65** | **0.60** | **0.50** | **0.40** | **3.80** |
| Unseen Lighting | OpenVLA | 0.55 | 0.35 | 0.25 | 0.10 | 0.00 | 0.00 | 1.25 |
| | Octo | 0.50 | 0.35 | 0.15 | 0.05 | 0.00 | 0.00 | 1.05 |
| | **PALM** | **0.80** | **0.70** | **0.60** | **0.60** | **0.45** | **0.40** | **3.55** |

**说明**: 三类泛化设置下 PALM 全面碾压 OpenVLA/Octo，且随序列变长优势越大（约 2–3 倍平均长度），印证长程鲁棒性。

### Table 6: Module Key Parameters / 各模块关键参数

| Type | Hidden Size | Layers | Heads |
|------|-------------|--------|-------|
| Image Encoder | 768 | 12 | 12 |
| Perceiver Resampler | 768 | 3 | 8 |
| GPT-2 (LLM Backbone) | 384 | 24 | 12 |
| Global Decoder | 384 | 2 | 16 |
| Local Decoder | 384 | 2 | 16 |
| Spatial Decoder | 384 | 2 | 16 |
| Dynamic Decoder | 384 | 2 | 16 |

**说明**: 主干 GPT-2 隐藏维 384、24 层；四个可供性解码器均为 2 层、16 头的轻量头。

### Table 7: DiT Configuration / 动作-进度扩散 Transformer 配置

| Parameter | Value |
|-----------|-------|
| Hidden Size | 384 |
| Number of Layers | 12 |
| Number of Heads | 12 |
| Sampling Steps | 10 |
| Noise Schedule | Cosine |
| Action Prediction Steps | 3 |
| Loss Function | MSE ($L_2$ loss) |
| Precision | fp32 |

**说明**: DiT 12 层、采样仅 10 步、动作预测 3 步，余弦噪声调度，保证推理高效。

### Table 8: Training Hyperparameters / 训练超参

| Hyperparameter | Pre-training | Fine-tuning |
|----------------|--------------|-------------|
| Number of GPUs | 8 | 8 |
| Batch Size | 80 / GPU | 64 / GPU |
| Learning Rate | 1e-4 | 1e-3 |
| Weight Decay | 1e-4 | 1e-4 |
| Optimizer | AdamW | AdamW |
| LR Schedule | Cosine decay | Cosine decay |
| Training Epochs | 30 | 40 |
| Historical Sequence Length | 7 | 7 |
| Action Prediction Length | 3 | 3 |

**说明**: 8×A100 训练；视觉/文本编码器全程冻结；可训练参数仅 68M，可在 RTX 4090 上微调。

### Table 9: Progress Threshold Ablation / 进度阈值 $\phi$ 消融（CALVIN ABCD）

| Threshold $\phi$ | 1 | 2 | 3 | 4 | 5 | Avg. Len. |
|------------------|---|---|---|---|---|-----------|
| 70% | 93.4 | 86.3 | 78.4 | 71.9 | 65.5 | 3.96 |
| 80% | 95.3 | 90.2 | 84.2 | 79.7 | 74.1 | 4.24 |
| **90%** | **96.9** | **93.8** | **89.3** | **85.9** | **82.0** | **4.48** |
| 100% | 95.2 | 89.7 | 84.4 | 78.3 | 73.0 | 4.21 |

**说明**: $\phi$ 是子策略切换边界——过低导致过早终止、过高致执行停滞。**默认 90% 最优**（4.48），70% 因过早切换显著降级，100% 因严苛饱和要求略降。

### Table 10: Prediction vs. Reconstruction Ablation / 预测目标 vs 重建消融

| Ablation Type | CALVIN ABCD Avg. Len. ↑ | Latency (ms) ↓ |
|---------------|-------------------------|----------------|
| **Affordance (PALM)** | **4.48** | 70 |
| Image / Video | 4.17 | 90 |
| Auxiliary | 3.58 | 55 |

**说明**: 结构化、动作中心的**可供性前瞻**在性能(4.48)与延迟(70ms)间取得最佳折中；预测未来 RGB 图像性能更低(4.17)且延迟更高(90ms)；仅重建当前观测最快(55ms)但性能最差(3.58)。

### Table 11: Real-Robot Failure Modes / 真实机器人失败模式量化（6 子任务，N=50）

| Method | Avg. Len. ↑ | Repeat ↓ | Skip ↓ | Premature Stop ↓ |
|--------|-------------|----------|--------|------------------|
| OpenVLA | 2.30 | 28% | 16% | 22% |
| Octo | 1.85 | 34% | 26% | 18% |
| **PALM** | **3.90** | **14%** | **8%** | **10%** |

**说明**: PALM 执行更长（3.90），并大幅降低**重复(14%)、漏步(8%)、过早终止(10%)**三类典型长程失败——直接对应引言中要解决的失败模式。

---

## 实验

### 数据集 / 基准

| 数据集 | 阶段 | 特点 | 用途 |
|--------|------|------|------|
| [[DROID]] + [[BridgeData V2]] | 预训练 | 大规模 in-the-wild 机械臂示范 | 构建多样真实任务基础理解 |
| [[EPIC-KITCHENS]] + [[RoboCerebra]] | 预训练 | 细粒度子步骤 / 时间段标注 | 监督长程语义进度估计 |
| 人工标注机器人轨迹 | 微调 | 942 条轨迹 + 半自动可供性/进度标注 | 学习条件可供性前瞻与逆动力学 |
| [[CALVIN]] ABC→D | 后训练评测 | 34 任务/4 环境，ABC 预训练、D 评测 | 长程语言条件指令跟随 |
| [[LIBERO]] (Spatial/Object/Goal/Long) | 后训练评测 | 每套件 10 任务/50 示范 | 操作与空间理解；LIBERO-90 预训练→LONG 微调 |
| 真实世界（xArm6） | 训练/测试 | 6 子任务长程拾放，200 条微调示范 | 随机定位/光照/干扰三类泛化 |

### 实现细节

- **视觉**: [[MAE]]-预训练 [[ViT]]-B；每图 196 patch + [CLS]，两视角（eye-on-base 全局 + eye-on-hand 局部）；[[Perceiver Resampler]] 蒸馏为紧致 token。
- **文本**: [[CLIP]] text encoder + 线性投影。
- **状态**: 6-DoF 笛卡尔位姿 + 夹爪二值（one-hot），分别线性投影后拼接经 MLP 得单一状态 token。
- **主干**: [[GPT-2]] 风格 Transformer（384 维、24 层），因果 + 跨模态注意力。
- **可供性头**: Global/Spatial 用 2 层 MLP；Local/Dynamic 用 2 层 Transformer + 线性投影；目标在 $t+n$ 定义、仅训练时用。
- **动作头**: [[Diffusion Transformer|DiT]]（384 维、12 层、采样 10 步、余弦调度），输出 7-DoF 动作 + 标量进度。
- **进度标签**: 人工标注功能关键帧（Grasp-Contact、Release 等）赋固定语义锚点，中间帧线性插值；视频与机器人共享里程碑形成统一尺度，保证跨域迁移可靠。
- **训练**: 8×A100；预训练 lr 1e-4 / 30 epoch、微调 lr 1e-3 / 40 epoch；视觉/文本编码器冻结；可训练参数 **68M**。
- **推理**: DiT 10 步、观测 7 步、预测 3 步；采样约 40ms，闭环 **10–15 Hz**（每决策周期 <80ms）；可供性前瞻避免显式图像解码。

### 关键实验结论

- **仿真**: CALVIN ABCD 平均长度 4.48（+12.5% vs 先前 SOTA）、5 连任务 82.0%（+17.7% vs Seer）；LIBERO 平均 94.5%、LIBERO-LONG 91.8%（+22.8% vs CoT-VLA）。
- **真实世界**: 三类泛化设置下相对 OpenVLA/Octo 约 2 倍平均长度提升；并大幅降低重复/漏步/过早终止三类失败（Table 11）。
- **消融**: 四类可供性互补（Fig 3）；微调最依赖可供性前瞻、预训练最依赖进度预测（Table 3）；进度阈值 90% 最优（Table 9）；可供性前瞻在性能-延迟上优于图像/视频预测与纯重建（Table 10）。

---

## 批判性思考

### 优点

1. **直击长程失败模式且证据闭环**: 引言列出的"重复/漏步/过早终止"失败模式在 Table 11 被直接量化降低，主张与证据自洽，而非泛泛宣称"更鲁棒"。
2. **可供性"训练有监督、推理只留潜变量"的设计巧妙**: 既用 [[Grounding DINO]]/[[SAM]]/[[GLOVER++]]/[[SpatialVLM]]/[[RoboPoint]]/[[CoTracker]] 等现成模型自动构造监督，又在推理时去掉头部、避免显式图像解码，兼顾结构化先验与高频实时（10–15 Hz）。
3. **轻量**: 仅 68M 可训练参数即在两大长程基准全面 SOTA，可在 RTX 4090 微调，复现门槛低。

### 局限性

1. **强依赖外部标注管线**: 四类可供性目标来自多个外部模型（Grounding DINO/SAM/SpatialVLM/RoboPoint/CoTracker）与半自动人工标注，这些先验模型的误差会直接传入监督，且管线本身复杂、迁移到新本体/新任务的标注成本不低。
2. **真实世界基线偏弱、规模有限**: 真实实验仅对比 OpenVLA/Octo，未含更强的 $\pi_0$/扩散类强基线；真实微调仅 200 条示范、任务集中在桌面水果拾放，长程性虽达 6 子任务但接触/可变形/双臂等复杂度未覆盖。
3. **进度监督依赖人工关键帧 + 线性插值假设**: 进度锚点靠人工标注、中间线性插值，隐含"子任务内进度近似线性"的假设，对非匀速/回退型子任务可能失真；阈值 $\phi$ 也是经验单一标量。

### 潜在改进方向

1. 把可供性监督从"外部模型蒸馏"转向**自监督/可学习目标**，减少对 Grounding DINO/SAM 等先验链路的依赖与误差累积。
2. 引入更强真实世界基线（$\pi_0$、扩散策略）与更难任务（接触丰富、可变形、双臂、更长 horizon），并将进度切换阈值做成**自适应/可学习**而非固定 90%。
3. 用更系统的进度标注（学习式进度而非线性插值）验证"进度→长程稳定"的因果强度，并探索跨本体迁移。

### 可复现性评估

- [ ] 代码开源（项目主页 https://plan-lab.github.io/palm，未在正文确认 code release）
- [ ] 预训练模型（未明确声明）
- [x] 训练细节完整（附录 Table 6/7/8 给出模块、DiT、超参配置）
- [x] 数据集可获取（DROID/BridgeData V2/EPIC-KITCHENS/RoboCerebra/CALVIN/LIBERO 均公开；真实数据自采）

---

## 速查卡片

> [!summary] PALM: Progress-Aware Policy Learning via Affordance Reasoning
> - **核心**: 在 VLA 内加"结构化未来可供性前瞻 + 连续进度估计"双机制，闭环稳住长程多步操作。
> - **方法**: CLIP/MAE-ViT/Perceiver 编码 → GPT-2 主干 + 两组可学习查询（四类可供性 Global/Local/Spatial/Dynamic + 动作-进度）→ DiT 联合输出动作与进度；训练有监督可供性头、推理只留潜变量 $\hat{\mathbf F}_{t+n}$；进度阈值 90% 触发子任务切换。
> - **结果**: 68M 参数；CALVIN ABCD Avg.Len. 4.48(+12.5%)、LIBERO-LONG 91.8%、真实世界约 2× 长程提升，10–15 Hz。
> - **项目**: https://plan-lab.github.io/palm

---

*笔记创建时间: 2026-06-29*
