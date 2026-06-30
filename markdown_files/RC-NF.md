---
title: "RC-NF: Robot-Conditioned Normalizing Flow for Real-Time Anomaly Detection in Robotic Manipulation"
method_name: "RC-NF"
authors: [Shijie Zhou, Bin Zhu, Jiarui Yang, Xiangyu Zhao, Jingjing Chen, Yu-Gang Jiang]
year: 2026
venue: CVPR
tags: [anomaly-detection, normalizing-flow, runtime-monitoring, VLA, OOD, point-set, SAM2]
zotero_collection: _inbox
image_source: online
arxiv_html: https://arxiv.org/html/2603.11106v1
created: 2026-06-29
---

# RC-NF: Robot-Conditioned Normalizing Flow for Real-Time Anomaly Detection in Robotic Manipulation

## 元信息

| 项目 | 内容 |
|------|------|
| 作者 | Shijie Zhou, Bin Zhu, Jiarui Yang, Xiangyu Zhao, Jingjing Chen (通讯), Yu-Gang Jiang |
| 机构 | 复旦大学可信具身智能研究院、上海市多模态具身智能重点实验室、新加坡管理大学 |
| 会议 | CVPR 2026 |
| 类别 | 机器人运行时监控 / 异常检测 / VLA |
| 日期 | 2026-03（arXiv v1） |
| 项目主页 | https://heikaishuizz.github.io/RC-NF/ |
| 链接 | [arXiv](https://arxiv.org/abs/2603.11106) / [Project](https://heikaishuizz.github.io/RC-NF/) |

---

## 一句话总结

> 用机器人状态与任务条件化的 [[Normalizing Flow|归一化流]]，仅靠成功示范无监督建模"机器人-物体"联合运动分布，对 VLA 执行轨迹实时计算异常分数（<100ms），作为即插即用监控器触发状态级回退或任务级重规划。

---

## 核心贡献

1. **机器人条件化归一化流（RC-NF）**: 提出一种无监督的 [[Normalizing Flow|归一化流]] 异常检测器，核心是新颖的仿射耦合层 [[RCPQNet]]（Robot-Conditioned Point Query Network），把"任务感知的机器人状态"作为条件注入流中，从而度量当前"机器人-物体"配置是否偏离正常任务分布。
2. **LIBERO-Anomaly-10 基准**: 基于 [[LIBERO]]-10 构建首个面向操作的异常检测基准，包含三类操作专属异常（Gripper Open / Gripper Slippage / Spatial Misalignment），各自考察从专家轨迹偏离的不同维度；RC-NF 在所有异常类型上取得 SOTA，平均 AUC/AP 超过最佳基线约 8%/10%。
3. **真实世界即插即用监控**: 在 Franka 机械臂上验证，作为 [[pi0|$\pi_0$]] 的并行模块，<100ms 给出 OOD 信号，支持**状态级轨迹回退（homing）**与**任务级重规划**，无需改动底层 VLA 架构即提升鲁棒性与适应性。

---

## 问题背景

### 要解决的问题
[[VLA]] 模型通过模仿学习训练，把高层任务提示映射到底层控制动作，但部署到动态真实环境时，频繁出现的 [[Out-of-Distribution|OOD]] 场景会显著拉低性能。因此 VLA 需要一个**准确、实时**的运行时监控系统，判断机器人执行是否仍与目标任务一致，并在偏离时及时干预。

### 现有方法的局限
1. **状态分类式失败检测**（如行为树、预定义前置/后置条件）：需要**穷举异常情况**或人工定义条件，难以应对真实操作的组合爆炸式变化。
2. **VLM/双系统监控**（如 Sentinel、dual-system 架构）：依赖**多步推理**，延迟可达数秒，无法及时干预；且在空间推理上易退化（实验中 Spatial Misalignment 上接近随机）。
3. **FailDetect**（最接近的流匹配基线）：把**图像特征与机器人状态直接拼接**作为流匹配输入，导致表征纠缠、特征不平衡，在特征选择与处理上仍有改进空间。

### 本文的动机
- 用**仅成功示范**的无监督训练，避免枚举所有异常；用概率密度函数直接产生可解释的异常分数。
- 把多模态输入**解耦处理**：机器人状态作 query、物体点集特征作 memory，做交叉注意力，既鼓励多模态交互又避免特征干扰。
- 用从 [[SAM2]] 分割得到的**物体点集表征**替代原始图像特征，对噪声更鲁棒。
- 借鉴行人场景的归一化流异常检测思路，使 RC-NF 成为 sub-100ms 的快速系统，防止闭环控制中的失败累积。

---

## 方法详解

### 模型架构

RC-NF 是在 [[Glow]] 基础上扩展的**条件归一化流**监控器，与策略网络**并行运行**（见 Figure 2）：
- **输入**:
  - 视频流 → [[SAM2]] 分割 → 网格采样得到物体点集 $\mathcal{X}$（形状 $B\times T\times N\times 2$）
  - 任务提示 → **球面均匀编码（Spherical Uniform Encoding）** → 任务嵌入 $\tau$
  - 机器人本体感知：$T$ 维关节状态、夹爪状态、笛卡尔位姿，构成机器人状态 $s$
- **条件**: $c=(s,\tau)$，注入到 $K$ 个仿射耦合层
- **核心模块**: [[RCPQNet]] 作为仿射耦合层，生成 scale $\gamma$ 与 shift $\beta$
- **输出**: 把点集映射到高斯隐分布 $\mathcal{Z}\sim\mathcal{N}(\mu_{\text{task}},I)$；负对数似然即异常分数
- **超参**: 流步数 $K=12$，训练 100 epoch；隐分布均值 $\mu_{\text{task}}$ 由任务嵌入 $\tau$ 广播得到

整体思想：建模联合分布 $p_\mathcal{T}(x_{\text{target}},s)$，用阈值判定是否偏离任务 $\mathcal{T}$ 的正常执行。

### 核心模块

#### 模块1: Robot-Conditioned Point Query Network (RCPQNet)

**设计动机**: 作为 RC-NF 的仿射耦合层，融合任务嵌入、机器人状态与物体点集，生成上下文相关的变换参数 $(\gamma,\beta)$，在保持流可逆性的同时引入机器人/任务条件。

**具体实现**（见 Figure 3）:
- 点集 $x$ 经第一流步的 ActNorm 与 Permute 后，沿**时间维**对半切分：$x_b=x[:T/2]$、$x_t=x[T/2:]$；用 $x_b$ 与条件 $c$ 算出 $(\gamma,\beta)$，仅对 $x_t$ 做仿射变换，保证可逆与可解雅可比。
- **Task-aware Robot-Conditioned Query**: 机器人状态先线性投影到隐空间，再用任务嵌入 $\tau$ 经 [[FiLM]] 调制，产生任务特定的 query token——既编码机器人状态上下文，又编码任务高层目标，作为检索时空线索的动态 key。
- **Dual-Branch Point Feature Encoding**（生成 memory token）:
  - **Dynamic Shape 分支**: 对每帧做中心化与归一化，消除平移与尺度影响，提取形状特征；把所有物体点集视为整体，用整体形状变化表示目标间相对运动。
  - **Positional Residual 分支**: 补偿形状归一化时丢失的绝对位置信息。
  - 两支特征经 MLP 升维 → 平均池化得帧级表征 → 各自 GRU 建模时间依赖 → Transformer encoder 产生 memory 向量。
- query 与 memory 在 Transformer 中做交叉注意力，学到随机器人状态与当前任务变化的上下文感知仿射参数。

#### 模块2: 球面均匀任务编码（Spherical Uniform Encoding）

**设计动机**: 让不同任务的嵌入在隐空间中**最大程度分离**，为后续密度估计提供最优几何结构。

**具体实现**:
- 把当前任务提示映射到半径为 $\mathcal{R}$ 的 $T$ 维超球面上的一个表面向量（$T$ 为滑动窗口步数）。
- 这些表面向量经优化逼近超球面上的均匀分布。隐分布均值 $\mu_{\text{task}}$ 由 $\tau$ 广播得到。
- 部署示例（附录）：LIBERO-10 的 10 个任务 → 在 $R=5$、$T=12$ 的超球面上取 10 个均匀分布向量，一一映射为 $[\tau_1,\dots,\tau_{12}]$。任务嵌入是区分"同一训练集内不同任务"（task-specific 异常 vs dataset-level 异常）的关键。

#### 模块3: 异常检测与处理（Anomaly Detection & Handling）

**设计动机**: 即插即用、并行运行、低延迟，给出可分级处理的 OOD 信号。

**具体实现**:
- 每步以观测配置的**负对数似然**作为异常分数，越高越偏离预期执行；超过上阈值即判定异常/OOD。
- **静态阈值**（公式7）：用两个来自成功示范的校准集 $S_1,S_2$ 估计；$\mu_\mathcal{T}$ 为 $S_1$ 的均值分数，$Q_{1-\alpha}(D_\mathcal{T})$ 为 $S_2$ 偏差的 $(1-\alpha)$ 分位（实验 $\alpha=0.05$，沿用 FailDetect 设计）；训练时做去偏（debiasing）保证分数时间平滑。
- **分级处理**:
  - **任务级 OOD**（环境/上下文不再匹配指令，如抽屉中途关闭）→ 通知高层（人或 LLM planner）做**任务重规划**。
  - **状态级 OOD**（任务仍有效但物理配置漂移，如球从夹爪掉到桌面）→ 触发 **homing 回程**做任务回退，局部调整轨迹直到异常分数降回阈值以下，再把控制权无缝交还 VLA。

### 关键公式与机制

#### 公式1: [[Normalizing Flow|归一化流]] 变量变换（密度估计基础）

$$
p_{X}(x)=p_{Z}(f(x))\left|\operatorname{det}\left(\frac{\partial f}{\partial x}\right)\right|
$$

**含义**: 通过可逆映射 $f$ 在复杂分布 $\mathcal{X}$ 与简单分布 $\mathcal{Z}$ 之间建立双向映射，利用雅可比行列式做显式概率密度估计。

**符号说明**:
- $f$: 可逆变换；$\partial f/\partial x$: 雅可比矩阵；$\det(\cdot)$: 局部体积变化因子
- $p_Z$: 简单先验（高斯）下的密度

#### 公式2: 条件可逆映射

$$
z=f_{c}(x)
$$

**含义**: RC-NF 把标准流条件化于 $c=(s,\tau)$，将点集 $x$ 映射到隐变量 $z$。

**符号说明**:
- $x\in\mathcal{X}$（物体点集），$z\in\mathcal{Z}$（高斯隐变量），$c=(s,\tau)$（机器人状态+任务嵌入）

#### 公式3: $K$ 步流复合

$$
y_{0}=x,\;y_{i}=f_{i,c}(y_{i-1}),\;z=y_{K},\;i=1,\dots,K
$$

**含义**: $f_c$ 由 $K$ 个条件变换复合而成，每个 $f_{i,c}$ 为一个由 RCPQNet 参数化的仿射耦合层。

**符号说明**:
- $y_i$: 第 $i$ 步中间表征；$z=y_K$: 最终隐表征；$K=12$

#### 公式4: 条件似然

$$
p_{X\mid C}(x\mid c)=p_{Z\mid C}\!\left(f_{c}(x)\mid c\right)\left|\det\frac{\partial f_{c}(x)}{\partial x}\right|
$$

**含义**: 基于变量变换原理，给定条件 $c$ 计算点集 $x$ 的条件似然，雅可比项刻画变换引起的局部体积变化。

#### 公式5: 对数似然展开（训练目标）

$$
\log p_{X\mid C}(x\mid c)=\log p_{Z\mid C}(z\mid c)+\sum_{i=1}^{K}\log\left|\det\frac{\partial f_{i,c}\!\left(y_{i-1}\right)}{\partial y_{i-1}}\right|
$$

**含义**: 训练时**最大化该对数似然**；负值即异常分数，反映当前"机器人-物体"配置属于正常任务分布的程度。

**符号说明**:
- 第一项：隐变量在条件下的对数概率；第二项：各流步雅可比对数行列式之和

#### 公式6: 高斯先验下的隐变量对数概率

$$
\log p_{Z\mid C}(z\mid c)=\text{Const}-\frac{1}{2}\|z-\mu_{\text{task}}\|_{2}^{2}
$$

**含义**: 隐变量服从以任务相关均值 $\mu_{\text{task}}$ 为中心、单位协方差的高斯；偏离 $\mu_{\text{task}}$ 越远似然越低、异常分数越高。

**符号说明**:
- $\mu_{\text{task}}$: 由任务嵌入 $\tau$ 广播得到的均值；$\|\cdot\|_2$: L2 范数

#### 公式7a: RCPQNet 生成仿射参数

$$
\gamma,\,\beta=\text{RCPQNet}(x_{b},c)
$$

**含义**: 用切分后的一半 $x_b$ 与条件 $c$ 计算 scale $\gamma$ 与 shift $\beta$。

#### 公式7b: 仿射耦合层输出

$$
y=\begin{bmatrix}y_{t}\\ y_{b}\end{bmatrix}=\begin{bmatrix}\gamma\odot x_{t}+\beta\\ x_{b}\end{bmatrix}
$$

**含义**: 仅对 $x_t$ 做仿射变换，$x_b$ 保持不变，保证可逆与可解雅可比。

**符号说明**:
- $\odot$: 逐元素乘；$x_b=x[:T/2]$、$x_t=x[T/2:]$（沿时间维切分）

#### 公式8: 异常阈值（保形校准）

$$
\text{Upper}_{\mathcal{T}}=\mu_{\mathcal{T}}+Q_{1-\alpha}(D_{\mathcal{T}})
$$

**含义**: 对每个任务 $\mathcal{T}$，用成功示范的校准集估计静态上阈值，分数超过即判异常。

**符号说明**:
- $\mu_\mathcal{T}$: 校准集 $S_1$ 的平均异常分数
- $D_\mathcal{T}=\{D_1,\dots,D_{n_2}\}$: $S_2$ 中相对 $\mu_\mathcal{T}$ 的偏差集合
- $Q_{1-\alpha}$: $(1-\alpha)$ 分位数（$\alpha=0.05$）

#### 公式9（附录）: 仿真中点集生成与投影

$$
\text{local\_offset}=\begin{bmatrix}dx\cdot W\\ dy\cdot H\\ dz\cdot D\end{bmatrix},\quad P_{\text{world}}=\text{geom\_pos}+R\cdot\text{local\_offset}
$$

$$
P_{\text{image}}=K\cdot(\text{Extrinsic}\cdot P_{\text{world}})
$$

**含义**: 仿真中用计算机图形学在物体几何内生成 $5\times5\times5$ 网格采样点，经几何位姿变换到世界系、再用相机内参 $K$ 与外参投影到 2D 图像，过滤无效投影后取 min/max 得 bbox（公式 $\text{bbox}=[x_{\min},y_{\min},x_{\max},y_{\max}]$）。这保证了仿真点集的稳定、低成本与可复现。

---

## 关键图表

<!-- 图片均使用 arXiv HTML 在线链接，未本地化 -->

### Figure 1: Teaser / LIBERO-Anomaly-10 三类异常与整体定位

![Figure 1](https://arxiv.org/html/2603.11106v1/figures/main.png)

**说明**: 提出 RC-NF 实时监控机器人执行状态与物体运动轨迹是否与任务一致；引入 LIBERO-Anomaly-10 含三类常见异常（Gripper Open、Gripper Slippage、Spatial Misalignment）；真实实验显示 RC-NF 提升 [[VLA]]（如 [[pi0|$\pi_0$]]）的适应性。是理解三类异常与方法定位的入口图。

### Figure 2: Framework Overview / 整体框架

![Figure 2](https://arxiv.org/html/2603.11106v1/figures/method.png)

**说明**: RC-NF 作为机器人操作的实时运行时监控器。(左) [[SAM2]] 从视频流提取物体掩码并网格采样为点集；任务提示经球面均匀编码，机器人本体感知提供关节/夹爪/位姿状态。(中) RC-NF 把这些信号作为 [[RCPQNet]] 仿射耦合层内的条件，做 $K$ 次可逆变换并计算当前任务的异常分数。本图是方法主干，展示了"点集+任务嵌入+机器人状态→条件流→异常分数"的数据流。

### Figure 3: RCPQNet / 机器人条件化点查询网络

![Figure 3](https://arxiv.org/html/2603.11106v1/figures/RCPQ.png)

**说明**: RCPQNet 作为 RC-NF 的仿射耦合层，生成 scale 与 shift 参数。它用 Task-aware Robot-Conditioned Query 生成 query token、用 Dual-Branch Point Feature Encoding 生成 memory token，二者在 Transformer 中交叉注意力，从而学到随机器人状态与任务变化的上下文感知仿射参数。是理解"解耦机器人状态与物体特征"的核心结构图。

### Figure 4: Anomaly Score Visualization / 异常分数曲线

![Figure 4](https://arxiv.org/html/2603.11106v1/figures/exp_sim.png)

**说明**: 任务 "Pick up the book and place it in the back compartment of the caddy" 的异常分数随时间步变化。$t_{\text{anomaly}}$（红色虚线）之前 RC-NF 稳定维持约 -2 表示正常，之后分数迅速上升。相比 VLM 监控无需长推理、相比 FailDetect 在平滑性/稳定性/准确性上更优，佐证结构化特征解耦与机器人条件流的有效性。

### Figure 5: Task-Level OOD Handling / 任务级异常处理（真实世界）

![Figure 5](https://arxiv.org/html/2603.11106v1/figures/exp_real_0.png)

**说明**: 任务 "placing the ball into the open drawer" 中抽屉中途关闭时，$\pi_0$ 与 $\pi_0$+RC-NF 的对比。$\pi_0$ 继续执行而失败；$\pi_0$+RC-NF 异常分数飙升超阈值，暂停运动并触发高层重规划（先复位球、打开抽屉再正常执行），最终成功。

### Figure 6: State-Level OOD (Two Consecutive Events) / 状态级异常处理对比

![Figure 6](https://arxiv.org/html/2603.11106v1/figures/exp_real1.png)

**说明**: $\pi_0$+RC-NF 与 $\pi_0$ 在真实部署中处理两次连续任务级 OOD 事件的对比。$t_{1a}$ 球被重新放回桌面、$t_{2a}$ 球滚到夹爪后方、$t_{3r}$ homing 后控制权交回 VLA。展示 RC-NF 对连续扰动的快速响应。

### Figure 7: State-Level Anomaly Score Plot / 状态级异常分数曲线

![Figure 7](https://arxiv.org/html/2603.11106v1/figures/exp_real2.png)

**说明**: 状态级 OOD 序列的异常分数图：$t_{1a}$ 球被重定位、$t_{2a}$ 球滚到夹爪后方。黄色曲线为检测到异常并由 homing 修正轨迹；绿色曲线为单独的 VLA 模型 $\pi_0$。直观呈现 RC-NF 触发回退的时机。

### Figure 8: Spherical Uniform Encoding / 球面均匀编码示意（附录）

![Figure 8](https://arxiv.org/html/2603.11106v1/figures/suppl_task_prior.png)

**说明**: 左侧为训练集十个任务及描述，右侧给出映射示例（如 $T=3$、6 任务时在 3 维球面上取与 $x/y/z$ 轴相交的 6 个向量）。解释任务嵌入 $\tau$ 如何获得最大分离的几何结构。

### Figure 9: Effect of Adding Misaligned Tasks / 加入空间错位任务前后对比（附录）

![Figure 9](https://arxiv.org/html/2603.11106v1/figures/suppl_sim_te.png)

**说明**: 把三个空间错位任务（left/front/right）加入训练集前后的指标对比。加入后 RC-NF 性能几乎不降（仍能把 Spatial Misalignment 识别为 task-specific 异常）；去掉任务嵌入则降到约 0.6，FailDetect 降到约 0.5（无法区分同一训练集内不同任务）。验证任务嵌入的关键作用。

### Figure 10: Bounding Box Prompt (Gemini 2.5 Pro) / 真实世界 bbox 生成提示（附录）

![Figure 10](https://arxiv.org/html/2603.11106v1/figures/suppl_prompt_bbox.png)

**说明**: 真实世界中用 Gemini 2.5 Pro 从任务指令理解被操作物体并零样本生成 bbox（供 SAM2 首帧使用）的提示模板。

### Figure 11: VLM Anomaly Scoring Prompt / VLM 异常打分提示（附录）

![Figure 11](https://arxiv.org/html/2603.11106v1/figures/suppl_prompt.png)

**说明**: 给 VLM 做异常打分的提示（改编自 Sentinel）。提供 TIME/TIME_LIMIT 计算进度比、DESCRIPTION 任务描述、FRAME_RATE=1Hz、CURRENT 为 256×256 图像序列。用于复现 GPT-5/Gemini/Claude 基线打分。

### Table 1: Anomaly Detection on LIBERO-Anomaly-10 / 异常检测主结果

| Method | GO-AUC | GO-AP | GS-AUC | GS-AP | SM-AUC | SM-AP | Avg-AUC | Avg-AP |
|--------|--------|-------|--------|-------|--------|-------|---------|--------|
| GPT-5 | 0.9137 | 0.9642 | 0.8941 | 0.8720 | 0.4904 | 0.4015 | 0.8500 | 0.8507 |
| Gemini 2.5 Pro | 0.8644 | 0.9333 | 0.8633 | 0.8505 | 0.5167 | 0.4271 | 0.8186 | 0.8313 |
| Claude 4.5 | 0.8754 | 0.9401 | 0.8551 | 0.8285 | 0.5292 | 0.4290 | 0.8214 | 0.8249 |
| FailDetect | 0.7883 | 0.9032 | 0.6665 | 0.6932 | 0.6557 | 0.5820 | 0.7181 | 0.7700 |
| **RC-NF (Ours)** | **0.9312** | **0.9781** | **0.9195** | **0.9180** | **0.9676** | **0.9585** | **0.9309** | **0.9494** |

**说明**: GO=Gripper Open，GS=Gripper Slippage，SM=Spatial Misalignment。RC-NF 在所有异常类型的所有指标上均最优，平均 AUC 0.9309 / AP 0.9494。关键观察：VLM 在 GO/GS 上尚可，但在 **Spatial Misalignment 上退化到接近随机**（AUC≈0.49–0.53），暴露其视觉-空间对齐弱点；RC-NF 用物体与机械臂的空间轨迹替代语义理解，在 SM 上仍达 0.9676 AUC。相比 FailDetect（拼接原始图像+状态、表征纠缠），RC-NF 用点集表征+query/memory 解耦设计取得全面领先（平均 AUC +21.3 个百分点）。

### Table 2: Ablation of RCPQNet / RCPQNet 消融

| Row | Config | GO-AUC | GO-AP | GS-AUC | GS-AP | SM-AUC | SM-AP | Avg-AUC | Avg-AP |
|-----|--------|--------|-------|--------|-------|--------|-------|---------|--------|
| 1 | **RC-NF (Ours, full)** | **0.9312** | **0.9781** | **0.9195** | **0.9180** | **0.9676** | **0.9585** | **0.9309** | **0.9494** |
| 2 | w/o Task Embedding | 0.8769 | 0.9603 | 0.8668 | 0.8680 | 0.8139 | 0.8118 | 0.8643 | 0.9008 |
| 3 | w/o Robot State | 0.6327 | 0.8621 | 0.7443 | 0.8116 | 0.8929 | 0.8617 | 0.7152 | 0.8401 |
| 4 | w/o Pos. Residual branch | 0.9045 | 0.9712 | 0.8971 | 0.9085 | 0.8543 | 0.8072 | 0.8947 | 0.9225 |
| 5 | w/o Dyn. Shape branch | 0.7666 | 0.9234 | 0.7763 | 0.8108 | 0.1022 | 0.2755 | 0.6841 | 0.7899 |

**说明**: 关键发现：
- **去任务嵌入**（Row 2）：只能检测 dataset-level OOD，无法区分 task-specific 异常，Spatial Misalignment 上 AUC 从 0.9676 跌至 0.8139（约 3 倍误差），印证任务嵌入对编码空间意图至关重要。
- **去机器人状态**（Row 3）：失去物体运动的上下文 grounding；Gripper Open（机械臂开夹但物体不动，异常体现在相对运动）AUC 跌至 0.6327 降幅最大，而 SM（可由物体位置偏差检出）降幅最小。
- **去 Positional Residual 分支**（Row 4）：中等下降（平均 AUC≈0.89），说明绝对位置信息是动态形状线索的补充。
- **去 Dynamic Shape 分支**（Row 5）：降幅最大（平均 AUC≈0.68，SM 上 AUC 仅 0.1022），证明时序形状演化比帧间平均位移偏移提供更强的异常证据，是双分支编码的主导成分。

---

## 实验

### 数据集 / 基准

| 数据集/基准 | 规模 | 特点 | 用途 |
|------|------|------|------|
| [[LIBERO]]-10 | 每任务 50 示范 | 覆盖 LIBERO 全部场景；仅成功示范 | 训练 |
| LIBERO-Anomaly-10 | 三类异常（GO/GS/SM） | GO/GS 覆盖全部 10 任务；SM 对应 3 个错误隔间放置 | 仿真测试 |
| 真实世界（Franka） | "蓝球放入打开的抽屉"及衍生任务 | wrist + 第三人称 RealSense D435 | 真实测试 |

三类异常定义：(1) **Gripper Open**——本该抓取时夹爪保持张开，物体未被触碰（测"物体正常但机器人状态不一致"）；(2) **Gripper Slippage**——抓取时把夹爪摩擦设为零致物体滑落、轨迹异常波动；(3) **Spatial Misalignment**——机械臂移向错误隔间（左/前/右）而非目标的后方（测语义-空间错位）。$t_{\text{anomaly}}$ 之前标为正常、之后标为异常。

### 实现细节

- **流步数**: $K=12$；训练 100 epoch；用 BalancedHardSampler 做训练数据去偏（先标准采样，到 NextStageEpoch 后用难样本均衡采样，缓解初始阶段轨迹相似导致的样本不均衡）。
- **任务嵌入**: $R=5$、$T=12$ 超球面均匀向量。
- **首帧 bbox**: 仿真用计算机图形学投影（稳定可复现）；真实世界用 [[Gemini 2.5 Pro]] 零样本生成给 SAM2。
- **阈值**: $\alpha=0.05$（沿用 FailDetect）。
- **真实平台**: Franka Research 3 + 双 RealSense D435；[[pi0|$\pi_0$]] 为模仿策略基线（VLA 用双视角训练，RC-NF 仅用第三人称视角监控）。
- **基线**: VLM 类（Sentinel 提示 + GPT-5 / Gemini 2.5 Pro / Claude 4.5，并行调用、1Hz 采样）；流类（FailDetect，按其开源代码参数）。
- **延迟**: 单帧推理 <100ms（消费级 RTX 3090），快于人类平均反应时间。

### 关键实验结论

- **主结果（Table 1）**: 全异常类型 SOTA，平均 AUC/AP 比最佳基线高约 8%/10%；VLM 在空间错位上接近随机，RC-NF 在该项达 0.9676 AUC。
- **可视化（Fig 4）**: 异常前稳定、异常后快速响应，且比 FailDetect 更平滑稳定。
- **消融（Table 2）**: 任务嵌入、机器人状态、双分支编码各有不可替代作用；Dynamic Shape 分支最关键。
- **真实世界（Fig 5-7）**: <100ms 响应；任务级触发重规划、状态级触发 homing 回退；无 RC-NF 时 $\pi_0$ 持续错误执行（抓空、抬起）。
- **任务嵌入泛化（Fig 9）**: 加入错位任务后性能几乎不降；去任务嵌入则降到约 0.6，凸显其区分同集内任务的能力。

---

## 批判性思考

### 优点
1. **范式契合且可解释**: 用条件归一化流把"是否偏离正常任务分布"直接转成可读的负对数似然分数，仅需成功示范无监督训练，避免枚举异常；阈值用保形校准给出，工程上干净。
2. **特征解耦设计有据**: 机器人状态作 query、物体点集作 memory 的交叉注意力，配合双分支点编码，消融充分（Table 2）证明每个组件的贡献，尤其 Dynamic Shape 分支主导。
3. **真实落地价值高**: <100ms、即插即用、不改 VLA 架构，且把 OOD 信号分级为状态级回退 vs 任务级重规划，给出了可操作的干预闭环。

### 局限性
1. **依赖外部分割与 bbox 质量**: 整条流水线建立在 [[SAM2]] 分割点集之上，真实世界首帧 bbox 还依赖 [[Gemini 2.5 Pro]]；分割失败/遮挡/多相似物体下的鲁棒性未系统评估。
2. **真实实验规模偏小**: 真实评估集中在"蓝球放入抽屉"单一代表场景及其衍生，缺少多任务、多物体、长程接触丰富任务的定量成功率统计，多为定性序列展示。
3. **阈值与任务嵌入偏静态/人工**: 静态阈值与超球面维度 $T$、半径 $R$、流步数 $K$ 等为经验设定；任务嵌入需任务集已知并一一映射，对开放任务集（运行时新任务）如何扩展不明确。
4. **基线对比的公平性**: VLM 在 1Hz 采样下与流模型实时性不可直接比，VLM 弱于空间推理部分属任务设置使然；与更多专门的机器人失败检测器对比仍可加强。

### 潜在改进方向
1. 把任务嵌入做成可在线扩展/检索式，支持运行时新任务，摆脱"固定任务集一一映射"约束。
2. 引入分割不确定性/多假设点集，量化对 SAM2 与 bbox 误差的敏感度；探索端到端从图像到点集的可学习采样。
3. 把静态阈值替换为自适应/上下文相关阈值（如随任务阶段变化），并对 $K,R,T$ 做系统敏感度分析。
4. 扩大真实世界任务覆盖（长程、可变形、多物体协同），给出成功率/干预次数等定量指标。

### 可复现性评估
- [x] 项目主页（https://heikaishuizz.github.io/RC-NF/，含演示视频）
- [ ] 代码开源（论文/主页未明确给出代码仓库链接）
- [x] 训练细节较完整（$K=12$、100 epoch、$\alpha=0.05$、$R=5$/$T=12$、去偏算法、仿真投影流程在附录）
- [x] 数据集可获取（基于公开 [[LIBERO]]-10 构建 LIBERO-Anomaly-10；真实数据未明确发布）

---

## 速查卡片

> [!summary] RC-NF: Robot-Conditioned Normalizing Flow for Real-Time Anomaly Detection
> - **核心**: 机器人状态+任务条件化的归一化流，仅用成功示范无监督建模"机器人-物体"联合分布，以负对数似然作异常分数实时监控 VLA。
> - **方法**: SAM2 物体点集 + 球面均匀任务嵌入 + 机器人状态 → RCPQNet 仿射耦合层（机器人状态作 query、双分支点编码作 memory，交叉注意力）→ 高斯隐分布 $\mathcal{N}(\mu_{\text{task}},I)$；超阈值则状态级 homing 回退或任务级重规划。
> - **结果**: LIBERO-Anomaly-10 全异常 SOTA（平均 AUC 0.9309 / AP 0.9494，超最佳基线约 8%/10%）；真实世界 <100ms、即插即用于 $\pi_0$。
> - **主页**: https://heikaishuizz.github.io/RC-NF/

---

*笔记创建时间: 2026-06-29*
