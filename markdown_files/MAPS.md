---
title: "MAPS: Preserving Vision-Language Representations via Module-Wise Proximity Scheduling for Better Vision-Language-Action Generalization"
method_name: "MAPS"
authors: [Chengyue Huang, Mellon M. Zhang, Robert Azarcon, Glen Chou, Zsolt Kira]
year: 2026
venue: CVPR
tags: [VLA, robust-fine-tuning, generalization, proximity-scheduling, catastrophic-forgetting, selective-projection-decay]
zotero_collection: _inbox
image_source: online
arxiv_html: https://arxiv.org/abs/2511.19878
created: 2026-06-29
---

# MAPS: Preserving Vision-Language Representations via Module-Wise Proximity Scheduling for Better Vision-Language-Action Generalization

## 元信息

| 项目 | 内容 |
|------|------|
| 作者 | Chengyue Huang*, Mellon M. Zhang*, Robert Azarcon, Glen Chou, Zsolt Kira（* 共同一作） |
| 机构 | Georgia Institute of Technology（佐治亚理工学院） |
| 会议 | CVPR 2026 |
| 类别 | VLA（Vision-Language-Action）/ 鲁棒微调 |
| 日期 | 2025-11（arXiv v1，2025-11-25） |
| 项目主页 | https://mapsvla.github.io |
| 链接 | [arXiv](https://arxiv.org/abs/2511.19878) / [PDF](https://arxiv.org/pdf/2511.19878) / [Project](https://mapsvla.github.io) |

> 说明：本论文 arXiv 暂未提供 HTML 版本（`/html/2511.19878` 返回 404），正文来自 PDF；插图改用**项目主页在线托管**的图片链接（`https://mapsvla.github.io/static/images/*.webp`），均已验证可访问。

---

## 一句话总结

> 把"鲁棒微调"（robust fine-tuning）按模块拆开：用一个**线性调度的逐模块邻近约束**，让视觉编码器牢牢贴近预训练权重、让面向动作的语言层自由适配，从而在不加任何参数/数据的前提下显著提升 VLA 的分布外泛化（最高 +30%）。

---

## 核心贡献

1. **首个面向 VLA 的鲁棒微调框架 MAPS**: 提出 **Module-Wise Proximity Scheduling**，把 [[Selective Projection Decay|SPD]] 的全局邻近超参 $\lambda$ 替换为**逐层、架构感知**的线性调度，对早期视觉层施加强约束、对深层语言层逐步放开，**不增加任何参数、数据或额外模型**，可即插即用接入现有 VLA。
2. **系统揭示模块重要性层级**: 通过对冻结不同 VLM 组件的系统实验，量化了一个广为流传却少被验证的直觉——保留预训练先验的重要性排序为 **DINOv2 > SigLIP > 早期语言层 > 后期语言层**，并据此设计调度方向。
3. **跨架构、跨基准、真机一致提升**: 在 MiniVLA-VQ、MiniVLA-OFT、OpenVLA-OFT、VLA-Adapter 四种主干，以及 [[SimplerEnv]]、[[CALVIN]]、[[LIBERO]] 与 Franka Emika Panda 真机上，**同时改善 ID 与 OOD** 性能（OOD 最高 +30%），证明"经验引导的预训练邻近性"是保泛化的简单而强力的原则。

---

## 问题背景

### 要解决的问题
[[VLA]] 模型普遍用大规模预训练的 [[VLM]] 初始化以继承其视觉与语义先验，但在稀疏、任务特定的机器人数据上做**朴素全量微调**时，预训练表征会被破坏，导致面对新任务/新物体/新环境时**泛化崩塌**（catastrophic forgetting）。核心矛盾是：**动作适配**与**预训练泛化保留**之间的权衡——微调过度会漂离预训练初始化、丧失分布外鲁棒性；微调不足又无法对齐动作空间。

### 现有方法的局限
- **冻结模块（freezing）**: 简单高效但属于**硬约束**，引入强归纳偏置；某些冻结配置只在一个基准上有效却在另一个上变差（任务相关偏置），不可靠。
- **双编码器（Dual-Encoder）[9]**: 同时维护冻结与可训练两条视觉通路，灵活但**参数/显存/算力翻倍**。
- **权重插值（ReVLA [8]）**: 把视觉权重逐渐回退到预训练态、同时适配语言主干，但需**多阶段训练**。
- **均匀软正则（RFT / SPD [13]）**: 用**单一超参**对全模型施加等强度约束，隐含假设"所有层应等量偏移"——这对 VLA 不成立，因为不同模块编码的先验差异巨大、对微调的敏感度也不同。

这些方法多偏重**视觉保留**，**忽视了语言组件中语义先验的角色**，且往往以更高的计算/架构复杂度为代价。

### 本文的动机
作者先做系统冻结分析（Sec. 4.2），确认"视觉模块应被强约束、语言层应更灵活"的层级直觉，但也发现冻结本身带来任务相关偏置、不可靠。于是把视野转向**软正则的鲁棒微调（RFT）**，并指出其单一超参的缺陷。MAPS 的核心思想：**既要 RFT 的轻量与平滑，又要冻结的模块化**——用一个**沿架构栈线性衰减的邻近权重**编码"早期视觉强约束、深层语言弱约束"的经验层级，把两种范式的优点合并。

---

## 方法详解

### 模型架构

MAPS 不是一种新 VLA 架构，而是一种**作用在动作微调阶段的鲁棒微调（RFT）调度策略**，适用于任意以 [[VLM]] 初始化的 VLA：
- **输入/被作用对象**: 一个有序模块栈 $L=(\ell_1,\dots,\ell_{|L|})$，顺序为 [[DINOv2]] → [[SigLIP]] → Bridge（融合桥）→ [[Language Model|语言层]]。
- **作用方式**: 在 [[Selective Projection Decay|SPD]] 的投影式约束之上，为每个模块/层分配一个**逐层邻近权重 $\lambda_k$**，越靠前（视觉）越大、越靠后（语言）越小。
- **从零初始化的组件**（动作头、本体状态投影器等）: 没有预训练 $\theta_0$ 可保留，故 $\lambda_k=0$，直接全量微调。
- **零额外开销**: 不引入新模型/数据/参数，不改架构。

VLA 的标准建模（背景，公式见下）：编码视觉 $f_v(o_t)$ 与语言 $f_l(x)$，由多模态 Transformer 融合得策略嵌入 $z_t=f_m([f_v(o_t),f_l(x)])$，再由动作头映射为连续控制或离散 token。

### 核心模块

#### 模块1: 冻结分析与模块重要性层级（Sec. 4.2）

**设计动机**: 在设计调度前，先用实验回答"哪些模块该被强保留"。作者把 OpenVLA 系视觉栈拆成四块——[[DINOv2]]（几何/深度先验）、[[SigLIP]]（视觉-语言对齐）、早期语言层、后期语言层——系统枚举冻结配置（见 Table 1/2）。

**关键发现**:
1. **语言适配不可或缺**: 冻结语言主干会严重限制对动作空间的适配，SimplerEnv 近乎零、LIBERO 最高掉 60%。
2. **冻结视觉编码器普遍有益**: SimplerEnv 上 ID +7~17%、OOD +7~25%；LIBERO 上约 +5% ID。
3. **后期语言层驱动任务性能**: 只微调后期语言层、冻早期语言层，OOD +1~3%。
4. **保 DINOv2 比保 SigLIP 更重要**: 冻 DINOv2 比冻 SigLIP 高约 5% OOD，凸显几何先验的价值。
5. **冻结效果不普适一致**: 冻结优于全量微调，但某些配置"此基准好、彼基准差"，说明冻结引入**任务相关归纳偏置**。

由此提炼出重要性层级（MAPS 调度方向的依据）：

$$
\text{DINOv2} > \text{SigLIP} > \text{Early language} > \text{Late language}
$$

#### 模块2: Module-Wise Proximity Scheduling（核心调度）

**设计动机**: 冻结=高模块化但强偏置；RFT=低偏置但无模块化（单一超参全局等量约束，见 Fig 3 底部）。MAPS 要**桥接二者**：学"何时保留、何时适配"。

**具体实现**:
- 把全局邻近超参 $\lambda$ 替换为**逐层、架构感知**的线性调度：早期视觉层 $\lambda$ 大（强保留），深层语言层 $\lambda\to0$（自由适配）。
- 从零初始化的非预训练模块（动作头、状态投影器）令 $\lambda_k=0$，即全量微调。
- 在 [[Adam]] 优化中，先做无约束更新 $\tilde\theta_t$，再用**梯度-位移相关性** $c_t$ 决定是否投影回 $\theta_0$（沿用 SPD 的触发条件与偏移比 $r_t$）。
- 消融显示：**线性调度优于常数（即普通 RFT）与余弦调度**，最佳为线性 + 投影强度 $\lambda_{\max}=v=0.5$（在 LIBERO 上，见 Table 6）。

如 Fig 3 顶部所示，MAPS 让 DINOv2 偏移最小、SigLIP 次之、语言层适配最多——形成平滑、模块感知的偏移衰减。

### 关键公式与机制

#### 公式1: [[VLA]] 策略建模（背景）

$$
z_t = f_m\!\big([\,f_v(o_t),\, f_l(x)\,]\big)
$$

**含义**: 视觉观测 $o_t$ 与语言指令 $x$ 分别经 $f_v,f_l$ 编码，由多模态 Transformer $f_m$ 融合为策略嵌入 $z_t$，再映射到动作 $a_t$（连续控制 / 离散 token）。

**符号说明**:
- $o_t$: 多模态视觉观测；$x$: 语言指令
- $f_v,f_l,f_m$: 视觉编码器、语言编码器、多模态融合 Transformer
- $z_t$: 策略嵌入，后续由动作头产生 $a_t$

#### 公式2: [[L2-SP]] 正则（从权重衰减到邻近保留）

$$
\mathcal{L}_{\text{L2-SP}} = \mathcal{L}(\theta_t) + \frac{\lambda_{\text{reg}}}{2}\,\|\theta_t - \theta_0\|_2^2
$$

**含义**: 不再正则化 $\|\theta_t\|_2$，而是惩罚微调参数 $\theta_t$ 与**预训练初始化 $\theta_0$ 的距离**，从而显式保留对预训练模型的邻近性，缓解特征漂移与灾难性遗忘。

**符号说明**:
- $\theta_t$: 第 $t$ 步微调参数；$\theta_0$: 预训练初始化
- $\lambda_{\text{reg}}$: 正则强度；$\mathcal{L}(\theta_t)$: 任务损失

#### 公式3: 邻近约束的硬形式（[[TPGM]]）

$$
\min_{\theta_t}\ \mathcal{L}(x,y;\theta_t),\qquad \text{s.t.}\ \ \|\theta_t-\theta_0\|_2 \le \gamma
$$

**含义**: TPGM 把 L2-SP 重写为**显式约束**：要求模型停留在以 $\theta_0$ 为心、半径 $\gamma$ 的 $\ell_2$ 球内。

**符号说明**:
- $\gamma$: 允许偏离预训练权重的最大半径

#### 公式4: 投影梯度法（PGM 投影回球内）

$$
\theta_t = \theta_0 + \frac{1}{\max\!\big(1,\ \tfrac{\|\tilde\theta_t-\theta_0\|_2}{\gamma}\big)}\,(\tilde\theta_t - \theta_0)
$$

**含义**: 对无约束更新后的权重 $\tilde\theta_t$，若超出半径 $\gamma$ 则按比例**投影回** $\theta_0$ 周围的 $\ell_2$ 球内；未超出则保持不变。

**符号说明**:
- $\tilde\theta_t$: 无约束（普通梯度）更新后的权重
- $\max(1,\cdot)$: 仅当偏移超出 $\gamma$ 时才收缩

#### 公式5: [[Selective Projection Decay|SPD]] 的触发条件（何时投影）

$$
c_t := -g_{t+1}^{\top}\,(\tilde\theta_t - \theta_0)
$$

**含义**: SPD 用梯度与"当前位移方向"的内积符号决定是否投影：若 $c_t<0$（更新方向与"保留预训练结构"冲突），则触发投影、加强约束；否则保持更新不变。

**符号说明**:
- $g_{t+1}$: 梯度；$\tilde\theta_t-\theta_0$: 相对预训练初始化的位移
- $c_t<0$: 加强 L2-SP 正则（投影）；$c_t>0$: 弱化正则（不投影）

#### 公式6: SPD 偏移比 $r_t$（投影多少）

$$
r_t = \frac{\max\{\gamma_t-\gamma_{t-1},\,0\}}{\gamma_t},\qquad \gamma_t=\|\tilde\theta_t-\theta_0\|_2,\ \ \gamma_{t-1}=\|\theta_{t-1}-\theta_0\|_2
$$

**含义**: SPD 用**当前偏移半径相对上一步的增量**定义偏移比 $r_t$，替代 L2-SP 中的学习率项，使单一超参 $\lambda$ 控制约束半径如何扩张/收缩（$\lambda{=}0$ 退化为全量微调，$0{<}\lambda{<}1$ 允许扩张，$\lambda{=}1$ 半径固定，$\lambda{>}1$ 收缩回 $\theta_0$）。

**符号说明**:
- $\gamma_t,\gamma_{t-1}$: 当前/上一步对 $\theta_0$ 的偏移半径
- $r_t$: 偏移比，进入 SPD/MAPS 的投影步

#### 公式7: MAPS 逐层邻近权重（核心创新）

$$
\lambda_k = \lambda_{\max}\left(1 - \frac{k-1}{|L|-1}\right)
$$

**含义**: 沿有序模块栈 $L=(\text{DINOv2}\to\text{SigLIP}\to\text{Bridge}\to\text{Language})$ 做**线性衰减**：第一层 $\lambda_1=\lambda_{\max}$（早期视觉，强保留），最后一层 $\lambda_{|L|}=0$（语言，自由适配）。从零初始化模块统一取 $\lambda_k=0$。

**符号说明**:
- $k$: 层在栈中的索引；$|L|$: 模块总数
- $\lambda_{\max}$: 最大邻近强度（即消融中的 $v$，最佳约 0.5；不同基准/模型 0.5~3.2，见 Table 8）

#### 公式8: MAPS 的层级投影更新

$$
\theta_t \leftarrow \tilde\theta_t - \lambda_k\, r_t\,\big(\tilde\theta_t - \theta_0\big)\qquad (\text{当 } c_t<0)
$$

其中梯度-位移相关性为：

$$
c_t := -\,g_t^{\top}\big(\theta_{t-1}-\theta_0\big)
$$

**含义**: 当 $c_t<0$（更新方向与保留预训练结构相悖）时，按**层特定强度 $\lambda_k$ 与偏移比 $r_t$** 把无约束更新 $\tilde\theta_t$ 拉回 $\theta_0$；否则接受 Adam 更新不变。与 SPD 的全局单约束不同，MAPS 显式编码架构结构——视觉层强锚定、深层语言层灵活适配。

**符号说明**:
- $\lambda_k$: 第 $k$ 层邻近权重（公式7）；$r_t$: SPD 偏移比（公式6）
- $g_t$: 第 $t$ 步梯度；$\theta_{t-1}-\theta_0$: 相对初始化的位移

---

## 关键图表

<!-- 图片均使用项目主页（mapsvla.github.io）在线托管链接，未本地化；arXiv 无 HTML 版本 -->

### Figure 1: Module-Wise Proximity Scheduling 概念图 / 方法总览

![MAPS Framework](https://mapsvla.github.io/static/images/maps.webp)

**说明**: MAPS 作用于动作微调阶段。横轴为从早期视觉层到高层语言层的模块顺序，纵轴为对预训练权重的邻近约束强度。**MAPS（蓝色虚线）** 对早期视觉层施加强保留、向语言层逐步放松（线性衰减，即公式7）；**vanilla 微调（绿色虚线）** 把 VLM 表征完全推离预训练权重（黑色实线）；**均匀 SPD（橙色点划线）** 则对所有层施加等强度约束。这张图直观给出 MAPS 区别于"冻结/朴素 RFT"的核心机制。

### Figure 2: 冻结配置定性对比（LIBERO-90 "put the bowl on the plate"）

> 该图无项目主页在线副本，文字转述（见 PDF Fig 2）。

**说明**: 从左到右为全量微调（FFT）、冻 VLM、冻语言、冻视觉。**FFT 失败最严重**——竟去够柜子而非碗；冻 VLM/语言/视觉能保住碗的 2D 定位但损害深度推理；**冻视觉**时机械臂能抓住碗，但难以准确放到盘子上。佐证"视觉应被强约束、但完全冻结又损害放置"的微妙权衡，引出 MAPS 的软调度。

### Figure 3: 微调-预训练权重的 ℓ2 偏移分布 / 调度行为对比

> 该图无项目主页在线副本，文字转述（见 PDF Fig 3）。

**说明**: 计算微调后权重与预训练权重的 $\ell_2$ 距离。**顶部（MAPS）**：偏移随模块平滑、模块感知地衰减——DINOv2 偏移最小、SigLIP 次之、语言层最大。**中部（RFT-V+FFT-L，仅对视觉栈做 RFT、语言全量微调）**：DINOv2 被重度约束但 SigLIP 与语言相对不受约束。**底部（均匀 $\lambda$ / RFT）**：所有层等量约束。可视化证明 MAPS 实现了"按模块重要性差异化约束"。

### Figure 4: SimplerEnv 定性示例（新物体：Red Bull 罐）

> 该图无项目主页在线副本，文字转述（见 PDF Fig 4）。

**说明**: 上排为全量微调，未能完成任务；下排为 MAPS，成功把罐子放到盘子上。直观展示 MAPS 对**新物体（同类未见实例）** 的泛化。

### Figure 5: 动机分析——为何朴素微调失败

![Motivation analysis](https://mapsvla.github.io/static/images/simpler_prim.webp)
![LIBERO L2 deviation](https://mapsvla.github.io/static/images/libero_l2_combined.webp)

**说明**: 项目主页"Motivation"部分给出的两张分析图——左为 SimplerEnv 上的初步冻结/敏感性分析，右为 LIBERO 上不同方法相对预训练权重的 $\ell_2$ 偏移合并对比。共同支撑"不同模块应以不同速率演化"的核心论断。

### Figure 6: Franka 细粒度子任务分析

> 该图无项目主页在线副本，文字转述（见 PDF Fig 6）。

**说明**: 把多步任务 Block / Cup 拆成子任务（定位物体1 → 抓取物体1 → 定位物体2 → 堆叠），并记录每步成功率与"每次成功的平均尝试次数"。MAPS 的最大优势来自**更可靠的抓取、更准的中间定位、更稳的多步执行**：Cups ID 上 grasp-first 80% vs 20%、最终堆叠 80% vs 0%；Blocks OOD(Hard) 基线完全失败而 MAPS 仍保有非平凡能力。

### Figure 7-10: 真机平台与执行序列（附录）

> 这些图无项目主页静态副本（主页以视频形式展示真机 rollout），文字转述（见 PDF Fig 7-10）。

**说明**:
- **Figure 7**: Franka Emika Panda 真机平台——单臂 + 两台 RealSense D435（第三视角 + 腕部相机），1920×1080 采集，center-crop 到 224×224。
- **Figure 8**: Task 2(Block) 的 Easy/Hard 设置；Hard 拉大双块间距探测长程规划，Hard-OOD 再换入未见红色块。
- **Figure 9 / 10**: ID / OOD 四任务的逐帧执行序列（基线上排 vs MAPS 下排），覆盖 Coke、Blocks、Cups、Laptop（及其 OOD 变体：Red Bull 罐、抬高蓝块、黄/蓝杯、MacBook）。

### Table 1: SimplerEnv 上冻结 / RFT 不同部件的对比（ID & OOD）

> `表头中 freeze（冻结）/ \ 表示全量微调；Early L、Last L 为早期/最后语言层。MiniVLA-OFT 基线。`

| DINOv2 | SigLIP | Early L | Last L | T1 Spoon | T2 Carrot | T3 Stack | T4 Eggplant | Avg. ID | Visual | Novel Obj | Novel Cat | Avg. OOD |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 基线 MiniVLA-OFT | | | | 22.0 | 30.0 | 0.0 | 2.0 | 13.5 | 12.7 | 2.0 | 12.0 | 8.9 |
| 全量微调(\\,\\) | | \\ | \\ | 22.0 | 24.0 | 0.0 | 34.0 | 20.0 | 20.0 | 13.3 | 10.7 | 15.0 |
| 冻DINOv2+SigLIP | 冻 | | | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1.3 | 0.0 | 2.7 | 1.2 |
| 冻全VLM | | | | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| **冻SigLIP+早晚语言适配** | 冻 | \\ | \\ | **42.0** | **32.0** | 6.0 | 42.0 | **30.5** | **32.7** | **51.3** | 11.3 | **33.6** |
| 冻DINOv2 | \\ | \\ | \\ | 12.0 | 10.0 | 12.0 | 90.0 | 31.0 | 29.3 | 35.3 | 22.7 | 29.7 |
| 冻DINOv2+SigLIP, 冻早语言 | 冻 | 冻 | \\ | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 4.0 | 0.7 | 2.0 | 2.2 |
| RFT（均匀） | | | | 4.0 | 8.0 | 0.0 | 94.0 | 26.5 | 6.7 | 20.0 | 14.0 | 13.6 |

**说明**: 揭示五条洞见（见"模块1"），核心是冻结视觉普遍有益、冻结语言致命、冻结效果**不普适一致**——为 MAPS 的"差异化软约束"提供动机。

### Table 2: LIBERO 上冻结不同部件的对比（VLA-Adapter 基线）

| DINOv2 | SigLIP | Early L | Last L | LIBERO-90 (ID) | -Spatial | -Object | -Goal | -Long | Avg. OOD |
|---|---|---|---|---|---|---|---|---|---|
| 基线 VLA-Adapter | | | | 83.0 | 2.0 | 5.0 | 1.0 | 2.0 | 2.5 |
| 全量微调 | | \\ | \\ | **92.0** | 0.0 | 3.0 | 0.0 | 6.0 | 2.3 |
| 冻DINOv2+SigLIP | 冻 | | | 29.0 | 1.0 | 4.0 | 1.0 | 0.0 | 1.5 |
| 冻全VLM | | | | 19.0 | 0.0 | 0.0 | 1.0 | 0.0 | 0.3 |
| 冻DINOv2 | \\ | \\ | \\ | 89.0 | 0.0 | 6.0 | 0.0 | 6.0 | 3.0 |
| 冻SigLIP | \\ | \\ | \\ | 73.0 | 1.0 | 0.0 | 0.0 | 3.0 | 1.0 |
| 冻DINOv2+SigLIP, 适配语言 | 冻 | \\ | \\ | 91.0 | 0.0 | 4.0 | 6.0 | 8.0 | **4.5** |
| RFT（均匀） | | | | 88.0 | 1.0 | 2.0 | 0.0 | 6.0 | 2.3 |

**说明**: LIBERO 上冻结收益较小但一致；冻视觉栈+适配语言的 OOD 最优（4.5），再次印证"视觉保、语言放"的方向。

### Table 3: SimplerEnv 结果（ID 4 任务 + Visual / Novel Object / Novel Category OOD）

> `∗ 借用原论文 checkpoint；† 在 SimplerEnv 上从零预训练。`

| Model | T1 Spoon | T2 Carrot | T3 Stack | T4 Eggplant | Avg. ID | Visual | Novel Obj | Novel Cat | Avg. OOD |
|---|---|---|---|---|---|---|---|---|---|
| RT-1-X | 4.2 | 0.0 | 0.0 | 0.0 | 1.1 | 0.0 | 4.0 | 6.1 | 3.4 |
| Octo | 12.5 | 41.7 | 15.8 | 0.0 | 17.5 | 12.6 | 10.8 | 8.4 | 10.6 |
| π₀ | 52.5 | 87.9 | 83.8 | 52.5 | **69.2** | **71.4** | 30.2 | 21.0 | **40.9** |
| MiniVLA-VQ † | 52.0 | 24.0 | 42.0 | 58.0 | 44.0 | 37.3 | 34.0 | 27.3 | 32.9 |
| **+MAPS (Ours)** | 44.0 | 30.0 | **76.0** | 20.0 | 42.5 | 48.0 | 26.0 | 29.3 | 34.4 (+1.5) |
| MiniVLA-OFT † | 22.0 | 30.0 | 0.0 | 2.0 | 13.5 | 12.7 | 2.0 | 12.0 | 8.9 |
| **+MAPS (Ours)** | **34.0** | **38.0** | 0.0 | 48.0 | **30.0 (+16.5)** | 32.0 | **48.7** | **26.7** | **35.8 (+26.9)** |
| OpenVLA-OFT † | 16.0 | 18.0 | 0.0 | 58.0 | 23.0 | 16.0 | 9.3 | 0.7 | 8.7 |
| **+MAPS (Ours)** | 2.0 | 16.0 | 0.0 | **74.0** | 23.0 | 18.7 | 20.0 | 12.7 | **17.1 (+8.4)** |

**说明**: MAPS 在 MiniVLA-OFT 上 ID 持平、OOD **+26.9**，且**仅用 BridgeData V2 这点预训练**就匹配甚至超过经大规模预训练的 RT-1-X/Octo/π₀ 的 OOD；OpenVLA-OFT(7B) 上 ID 持平、OOD +8.4。说明 MAPS 跨主干尺度有效（RQ2）。

### Table 4: LIBERO 结果（LIBERO-90 训练，四类 OOD 测试）

> `∗ 借自 [17]；† 从零在 LIBERO-90 预训练；‡ 在 LIBERO-90 上微调。无任何外部预训练。`

| Model | LIBERO-90 (ID) | -Spatial | -Object | -Goal | -Long | Avg. OOD |
|---|---|---|---|---|---|---|
| OpenVLA | 61.4 | - | - | - | - | - |
| miniVLA ∗ | 62.0 | 0.0 | 0.0 | 0.0 | 1.0 | 0.25 |
| miniVLA-VQ ∗ | 77.0 | 0.0 | 0.0 | 3.0 | 1.0 | 1.0 |
| miniVLA-hist. ∗ | 82.0 | 0.0 | 1.0 | 2.0 | 7.0 | 2.5 |
| SpatialVLA ‡ | 46.2 | 0.0 | 0.0 | 0.67 | 1.33 | 0.5 |
| MiniVLA-VQ † | 79.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| **+MAPS (Ours)** | **82.0 (+3.0)** | 0.0 | 1.0 | **8.0** | **10.0** | **4.75 (+4.75)** |
| MiniVLA-OFT † | 91.0 | 0.0 | 8.0 | 7.0 | 4.0 | 4.75 |
| **+MAPS (Ours)** | 91.0 | **6.0** | **14.0** | 0.0 | 8.0 | **7.0 (+2.25)** |
| VLA-Adapter † | 83.0 | 2.0 | 5.0 | 1.0 | 2.0 | 2.5 |
| **+MAPS (Ours)** | **88.0 (+5.0)** | 0.0 | 7.0 | 6.0 | 6.0 | **4.75 (+2.25)** |
| OpenVLA-OFT † | 92.0 | 3.0 | 9.0 | 3.0 | 5.0 | 5.0 |
| **+MAPS (Ours)** | 90.0 | 0.0 | 11.0 | 1.0 | **10.0** | 5.5 (+0.5) |

**说明**: 四种主干 OOD 一致 +1~5%；MiniVLA-VQ 基线本无 OOD 能力，MAPS 让 Goal +8、Long +10。MAPS 在更强基线上增益更大，说明它**互补而非替代**已有能力（RQ2）。注意 LIBERO 协议极具挑战——仅在 LIBERO-90 上预训练、无任何外部预训练。

### Table 5: Franka 真机结果（ID 4 任务 + OOD 4 变体）

| Model | T1 Coke | T2 Block | T3 Cup | T4 Laptop | Avg. ID | T1 Coke | T2 Block | T3 Cup | T4 Laptop | Avg. OOD |
|---|---|---|---|---|---|---|---|---|---|---|
| MiniVLA-OFT † | 30.0 | 30.0 | 0.0 | 100.0 | 40.0 | 50.0 | 20.0 | 0.0 | 30.0 | 22.5 |
| **+MAPS (Ours)** | **50.0** | **70.0** | **80.0** | 90.0 | **72.5 (+32.5)** | **100.0** | **30.0** | **20.0** | **60.0** | **52.5 (+30.0)** |

**说明**: 真机上 MAPS ID 40→72.5、OOD 22.5→52.5，OOD **+30**。作者归因：仅 600 条示范使基线严重过拟合到示范分布，MAPS 的正则缓解了这一点。在线项目主页提供真机视频对比。

### Table 6: 调度器消融（Linear=MAPS vs Cosine vs Constant，LIBERO）

| Scheduler / Value | LIBERO-90 (ID) | -Spatial | -Object | -Goal | -Long | Avg. OOD |
|---|---|---|---|---|---|---|
| Constant (v=0.5) | 80 | 0 | 0 | 0 | 6 | 1.5 |
| Constant (v=2) | 67 | 0 | 0 | 0 | 4 | 1.0 |
| Constant (v=3) | 51 | 0 | 0 | 0 | 0 | 0.0 |
| Cosine (v=0.5) | 81 | 0 | 2 | 0 | 6 | 2.0 |
| Cosine (v=2) | 85 | 0 | 7 | 3 | 3 | 3.25 |
| Cosine (v=3) | 87 | 0 | 9 | 1 | 2 | 3.0 |
| **Linear (v=0.5)** | **88** | 0 | 7 | **6** | 6 | **4.75** |
| Linear (v=2) | 84 | 2 | 6 | 0 | 4 | 3.0 |
| Linear (v=3) | 88 | 1 | 2 | 0 | 6 | 2.25 |

**说明**: **线性调度 + v=0.5** 兼顾最稳的 ID 与最高的 OOD（4.75）。Constant（即普通 RFT）随 v 增大反而退化，证明"逐层差异化"优于"全局均匀"。

### Table 7: 与双编码器对比（LIBERO, MiniVLA-VQ）

| Method | LIBERO-90 (ID) | Avg. OOD | Params |
|---|---|---|---|
| MiniVLA-VQ (baseline) | 79 | 0.0 | 1.5B |
| +Dual-Encoder [9] | 75 | 2.75 | 1.5B |
| +Dual-Encoder (new DINOv2+SigLIP) | 79 | 3.75 | **2.5B** |
| **+MAPS (Ours)** | **82** | **4.75** | **1.5B** |

**说明**: MAPS 在**不增加参数（1.5B）** 的前提下取得最佳 ID/OOD；双编码器需复制视觉编码器（参数膨胀到 2.5B）却仍不及 MAPS。此外正文还指出权重插值[8]需两次训练、双层优化[15]约 2× 训练时间，均逊于 MAPS。

### Table 8: 各基准/模型的训练配置（附录）

| Benchmark | Model | lr | steps | batch | wrist | proprio | img-aug | LoRA | GPUs | Time | λmax |
|---|---|---|---|---|---|---|---|---|---|---|---|
| LIBERO | MiniVLA-VQ | 1e-5 | 50000 | 16×8 | no | no | False | False | 8×A40 | 24h | 0.5 |
| LIBERO | MiniVLA-OFT | 5e-5 | 50000 | 8×8 | yes | yes | True | False | 8×A40 | 19h | 3.2 |
| LIBERO | VLA-Adapter | 1e-4 | 50000 | 8×8 | yes | yes | True | False | 8×A40 | 22h | 0.5 |
| LIBERO | OpenVLA-OFT | 5e-5 | 50000 | 4×16 | yes | yes | True | r=32 | 16×A40 | 28h | 1.0 |
| CALVIN | MiniVLA-OFT | 5e-5 | 40000 | 8×8 | yes | yes | True | False | 8×A40 | 19h | 2.5 |
| CALVIN | OpenVLA-OFT | 1e-4 | 35000 | 4×16 | yes | yes | True | r=32 | 16×A40 | 28h | 1.5 |
| SimplerEnv | MiniVLA-VQ | 1e-5 | 50000 | 32×4 | no | no | False | False | 4×H200 | 10h | 0.5 |
| SimplerEnv | MiniVLA-OFT | 5e-5 | 100000 | 8×8 | no | no | True | False | 8×A40 | 15h | 3.0 |
| SimplerEnv | OpenVLA-OFT | 2e-4 | 100000 | 16×4 | no | no | True | r=32 | 4×H200 | 21h | 0.8 |
| Franka | MiniVLA-OFT | 5e-5 | 100000 | 8×8 | yes | no | True | False | 8×A40 | 14h | 1.5 |

**说明**: $\lambda_{\max}$（最大邻近强度）按模型/基准在 0.5~3.2 间调整，是 MAPS 唯一的关键超参；其余为各 VLA 主干常规设置。

### CALVIN 结果（Figure 5 in PDF / 项目主页结果图）

![CALVIN results](https://mapsvla.github.io/static/images/calvin_results.webp)
![LIBERO results](https://mapsvla.github.io/static/images/libero_results.webp)

**说明**: CALVIN ABC→D（长程多任务，连续 5 指令）上，MAPS 让 MiniVLA-OFT 平均序列长度 **+0.7**、各 horizon 成功率一致提升（2~5 连续任务各约 +15%、1 任务 +7%）；OpenVLA-OFT +0.1 / +2%。右图为 LIBERO 各主干的提升汇总。再次印证"对更强基线增益更大"。

### SimplerEnv / Franka 结果图（项目主页）

![SimplerEnv results](https://mapsvla.github.io/static/images/simpler_results.webp)
![Franka results](https://mapsvla.github.io/static/images/franka_results.webp)

**说明**: 项目主页对 SimplerEnv 与 Franka 真机结果的可视化汇总，对应 Table 3 与 Table 5 的柱状呈现。

---

## 实验

### 数据集 / 基准

| 基准 | 协议 | 特点 | 用途 |
|------|------|------|------|
| [[SimplerEnv]] (Bridge) | WidowX BridgeData V2 Visual Matching；ID 4 任务 | 轻量仿真，OOD 含 Visual/Novel Object/Novel Category | 从零预训练 + ID/OOD 测试 |
| [[CALVIN]] | ABC→D；连续 5 指令长程 | real-to-sim 长程桌面操作 | 训练 A–C，OOD 测 D |
| [[LIBERO]] | 仅 LIBERO-90 预训练；测 Spatial/Object/Goal/Long | **无任何外部预训练**，更难协议 | 训练 + OOD 测试 |
| Franka Emika Panda | 600 条示范，4 ID 任务 + 4 OOD 变体 | 真机单臂 + 第三视角/腕部 D435 | 真机训练 + 评测（各 10 试） |

### 实现细节

- **被微调主干**: MiniVLA-VQ、MiniVLA-OFT、OpenVLA-OFT、VLA-Adapter（MiniVLA = DINOv2-SigLIP 视觉栈 + [[Qwen2.5]]-0.5B；OpenVLA-OFT = LLaMA-2 7B + 并行解码 $\ell_1$ 回归头；VLA-Adapter = Bridge Attention 轻量头）。
- **方法注入**: MAPS 仅在动作微调阶段以 Adam + 逐层投影实现（Algorithm 1），无新增参数/数据/模型。
- **关键超参**: $\lambda_{\max}$（0.5~3.2，随基准/模型调），线性调度 + v=0.5 在 LIBERO 最优。
- **优化/硬件**: lr 1e-5~2e-4、batch 视模型、50k~100k 步、A40/H200 多卡、单次训练 10~28h；部分模型用 LoRA r=32。
- **重要约定**: **所有基准均从各 VLA 主干的预训练 VLM 权重直接微调**（不借用现成 VLA checkpoint），真机直接更新 VLM 权重。

### 关键实验结论

- **SimplerEnv**: MiniVLA-OFT +MAPS 的 OOD +26.9，匹配/超越大规模预训练的 RT-1-X/Octo/π₀。
- **CALVIN**: MiniVLA-OFT 平均序列长度 +0.7，长 horizon 各约 +15%。
- **LIBERO**: 四主干 OOD 一致 +1~5%，且 ID 也有小幅提升（VQ +3、VLA-Adapter +5）。
- **Franka 真机**: ID 40→72.5、OOD 22.5→52.5（+30），细粒度分析显示抓取/中间定位/多步稳定性全面更优。
- **消融**: 线性调度 > 余弦 > 常数（普通 RFT）；MAPS 比双编码器/权重插值/双层优化更优且最省算力。

---

## 批判性思考

### 优点
1. **极简且零成本**: 不加参数/数据/模型，只改优化时的逐层约束强度，一行调度即可插入任意 VLA——工程落地与可迁移性极强。
2. **有依据的设计**: "DINOv2 > SigLIP > 早语言 > 晚语言"的层级由系统冻结实验（Table 1/2）实证得出，调度方向不是拍脑袋；Fig 3 的 $\ell_2$ 偏移可视化进一步验证机制确实生效。
3. **跨主干/基准/真机一致**: 四种主干、三个仿真基准、真机均提升，且对更强基线增益更大，结论稳健；与双编码器/插值/双层优化的对比公平且全面（Table 7）。

### 局限性
1. **OOD 绝对值偏低**: LIBERO 各 OOD 多在 10% 以下（如 Spatial 常为 0），CALVIN/SimplerEnv 也有限——"相对提升大"但绝对成功率离实用仍远，部分源于"无外部预训练"的苛刻协议。
2. **$\lambda_{\max}$ 需逐任务调**: 唯一关键超参在 0.5~3.2 间随模型/基准大幅变化（Table 8），缺乏自动选取或敏感度分析；线性调度形式也较启发式。
3. **层级假设的粒度粗**: 调度按"DINOv2→SigLIP→Bridge→Language"四大块线性衰减，未区分语言内部各层的差异，也未验证对非 DINOv2-SigLIP 视觉栈（如纯 SigLIP / PaliGemma 式）是否仍成立。
4. **真机增益解释偏推测**: ID +32.5 的巨大提升主要归因于"600 条示范致基线过拟合"，属事后假设，未做受控的数据规模对照。

### 潜在改进方向
1. 把 $\lambda_{\max}$ 与调度形状做成可学习/自动搜索（如双层优化或基于梯度敏感度的自适应），减少手工调参。
2. 用表征探针（CKA、线性可分性、深度估计质量等）定量刻画"邻近性 → 泛化"的因果链，而非仅 $\ell_2$ 偏移可视化。
3. 在更多视觉编码器与更大主干上验证层级假设的普适性，并探索语言层内部更细粒度的调度。
4. 引入更强 OOD 扰动与更难长程/接触任务，提升 OOD 绝对成功率的可信度。

### 可复现性评估
- [x] 代码开源（项目主页 https://mapsvla.github.io，声明开源；含完整 Algorithm 1 与 Table 8 全部超参）
- [ ] 预训练模型（论文未明确提供 checkpoint 下载）
- [x] 训练细节完整（附录 Table 8 给出各基准 lr/steps/batch/LoRA/GPU/λmax）
- [x] 数据集可获取（SimplerEnv/CALVIN/LIBERO 公开；真机 600 条示范为自采）

---

## 速查卡片

> [!summary] MAPS: Module-Wise Proximity Scheduling for VLA
> - **核心**: 把鲁棒微调的全局邻近约束 $\lambda$ 换成**沿"视觉→语言"线性衰减**的逐层 $\lambda_k$，视觉强保留、语言自由适配；不加任何参数/数据。
> - **机制**: $\lambda_k=\lambda_{\max}(1-\tfrac{k-1}{|L|-1})$，在 SPD 投影（$c_t<0$ 时按 $\lambda_k r_t$ 拉回 $\theta_0$）上施加；从零模块 $\lambda_k=0$。
> - **依据**: 实证模块重要性 DINOv2 > SigLIP > 早语言 > 晚语言。
> - **结果**: 四主干 × SimplerEnv/CALVIN/LIBERO/Franka 一致提升 ID 与 OOD，OOD 最高 **+30%**；线性调度 + v=0.5 最优，优于双编码器/插值/双层优化且最省算力。
> - **项目**: https://mapsvla.github.io

---

*笔记创建时间: 2026-06-29*
