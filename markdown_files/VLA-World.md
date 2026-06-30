---
title: "Learning Vision-Language-Action World Models for Autonomous Driving"
method_name: "VLA-World"
authors: [Guoqing Wang, Pin Tang, Xiangxuan Ren, Guodongfang Zhao, Bailan Feng, Chao Ma]
year: 2026
venue: CVPR
tags: [VLA, world-model, autonomous-driving, autoregressive-generation, GRPO, reinforcement-learning, trajectory-planning]
zotero_collection: _inbox
image_source: online
arxiv_html: https://arxiv.org/html/2604.09059v1
created: 2026-06-29
---

# Learning Vision-Language-Action World Models for Autonomous Driving

## 元信息

| 项目 | 内容 |
|------|------|
| 作者 | Guoqing Wang, Pin Tang, Xiangxuan Ren, Guodongfang Zhao, Bailan Feng, Chao Ma |
| 机构 | 上海交通大学（Chao Ma 组）、华为（Bailan Feng 等）—— 据作者署名推断 |
| 会议 | CVPR 2026 |
| 类别 | VLA（Vision-Language-Action）+ World Model，端到端自动驾驶 |
| 日期 | 2026（arXiv v1） |
| 项目主页 | https://vlaworld.github.io |
| 链接 | [arXiv](https://arxiv.org/abs/2604.09059) / [Project](https://vlaworld.github.io) |

---

## 一句话总结

> 把[[World Model|世界模型]]的"预测性想象"与 [[VLA]] 的"反思性推理"统一进一个自回归 VLM：先由短时轨迹驱动生成下一帧未来图像，再对自己想象出的未来做风险推理来修正轨迹，并用三阶段（预训练→SFT→[[GRPO]] 强化学习）训练，在 nuScenes 上同时刷新规划与生成 SOTA。

---

## 核心贡献

1. **VLA-World 新范式**: 提出把 [[VLA]] 的"决策因子" $p(\tau\mid o,g)$ 与 [[World Model|世界模型]]的"想象因子" $p(x_{t+1}\mid o,\tau)$ 联合建模（公式2），用**自生成的近未来帧**作为反思推理的显式证据，弥补纯 VLA 缺时空建模、纯世界模型缺反思评估的双重缺陷。
2. **action→imagination→reflection 闭环**: 先预测 0.5s 短时轨迹/方向 → 据此**条件生成**下一帧图像 → 在生成图像上**显式 reasoning** 识别风险 → 修正输出 3s 长程轨迹，模拟人类驾驶"直觉预判 + 反思纠错"。
3. **数据集 + 三阶段训练**: 构建 nuScenes-GR-20K 生成-推理数据集，配合"视觉预训练（激活生成）→ 监督微调（注入概念知识/冷启动）→ [[GRPO]] 强化学习（规则奖励精修策略）"的三阶段范式，在 nuScenes 规划（L2/碰撞率）与未来生成（FID）双榜领先。

---

## 问题背景

### 要解决的问题
端到端自动驾驶里，如何让模型既能**想象环境如何演变**（时空一致、可预判），又能**对想象出的未来做反思评估**（判断是否安全、可行），从而输出更有前瞻性、更可解释的规划轨迹。

### 现有方法的局限
作者把当前两大主流范式（见 Figure 2）的痛点拆开：

1. **纯 [[VLA]] 模型**（OmniDrive、FSDrive、EMMA 等）: 建立在大规模视觉-指令预训练的 VLM 上，把感知/推理/规划统一为自回归生成，泛化与扩展性强；但**缺乏对其他动态交通参与者的显式时空建模与世界一致性约束**，往往只盯着 ego 车，难以预测复杂场景演化 —— 而这恰是安全主动驾驶的核心。
2. **纯[[World Model|世界模型]]**（DriveDreamer、Drive-WM、GenAD、Doe-1 等）: 学习潜在时空动态、能"做梦"生成未来帧；但通常**依赖大规模视觉数据拟合先验分布再采样**，没有真正抓住世界的因果关系，"模拟世界"而非"理解世界"，且**缺乏反思推理** —— 能想象会发生什么，却不能评估这些未来是否安全/可行/可取。

### 本文的动机
理想范式应**融合世界模型的时空建模力与 VLA 的推理力**：既能预想场景演化，又能解释并反思这些想象出的未来，像人类司机一样。文中给出生动直觉：空旷路上司机靠快速直觉想象（类世界模型）预判下一刻；一旦行人突然闯入，立刻切换到反思推理 —— 在脑中模拟"若保持当前速度会怎样"、评估后果、再否决继续前进的冲动。核心洞见：**短时预测出的未来天然编码了丰富时空信息（ego 运动 + 周围 agent 行为），是可靠驾驶推理的关键依据**。

---

## 方法详解

### 模型架构

VLA-World 是一个统一的自回归 VLM 框架（主干 [[Qwen2-VL]]-2B，沿用 [[FSDrive]] 初始化），把驾驶建模成一条**多步推理链**（见 Figure 2c）: 感知 → 短时预测 → 生成 → 反思 → 规划。

- **输入**: 多视角观测 + ego 状态 $o_t=\{I_t^{1:K},S_t\}$（六路相机 + 速度/加速度/横摆角速度等 CAN 信号）+ 任务目标 $g$（左/右/直行）
- **Backbone**: [[Qwen2-VL]]-2B，配 [[VQGAN]] 视觉 tokenizer 把未来帧离散成视觉 token
- **核心机制**: [[Action Conditioned Generation|动作条件生成]]（轨迹驱动的下一帧想象）+ [[Reflective Reasoning|反思推理]]（在生成帧上做风险推理修正轨迹）
- **输出**: 结构化标签序列 `<Perception>/<Prediction>/<Visual>/<Think>/<Action>/<Answer>`，最终给出高层动作 + 3s 长程轨迹 $\tilde\tau_{t:t+H}$

### 核心模块

#### 模块1: Action-conditioned Generation（动作条件的未来想象）

**设计动机**: 不把视觉生成当作辅助输出，而是用"短时轨迹驱动生成的未来帧"作为反思推理的显式线索 —— 短时未来编码了场景如何演化的丰富时空信息。

**具体实现**:
- 给定历史观测 $o_{1:t}$ 与目标 $g$，先预测初始短时轨迹 $\hat\tau_{t:t+1}$（0.5s 内的下一航点与驾驶方向）；
- 以该轨迹 + 历史观测为条件，**想象**下一时刻视觉观测 $\hat x_{t+1}$（公式3）；
- 生成走自回归 next-token：把未来帧编码成固定数量的 [[VQGAN]] 离散视觉 token，再解码回图像 $\hat I_{t+1}^k$（公式5）。视觉预训练阶段**显式强制多视角一致性**（区别于 FSDrive 只生成前视），使下游可按需生成任意相机视角的连贯未来帧。

#### 模块2: Reflective Reasoning（反思推理 / Thinking with Visual Tokens）

**设计动机**: 在自己生成的未来帧上做因果解释，把视觉证据转化为情境理解 —— 量化安全裕度、预判冲突、校验轨迹可行性，"不仅预测会发生什么，还推理是否应该发生"。

**具体实现**:
- 反思模块 $f_{\text{ref}}$ 以历史观测、想象帧 $\hat x_{t+1}$、初始短时轨迹为输入，分析显著实体、运动线索、潜在交互，评估环境风险（公式4）；
- 在保留初始预测意图的同时，**修正那些不安全或与自生成未来不一致的决策**，输出最终 3s 轨迹 $\tilde\tau_{t:t+H}$，既反映预测动态又反映对想象场景的推理。

#### 模块3: 三阶段训练（见 Figure 3）

**设计动机**: 直接 RL 难以在结构化多步推理的大搜索空间里收敛，需先用预训练对齐生成/理解、再用 SFT 冷启动注入概念知识，最后 RL 精修。

**具体实现**:
- **(a) 视觉预训练（≈500k 样本）**: 沿用 [[FSDrive]] 对齐策略，激活视觉理解 + 生成能力，自回归预测多视角未来帧的视觉 token（公式5），习得跨视角统一时空先验。
- **(b) 监督微调 SFT（≈20k 样本，generation-to-think 范式）**: 在多任务混合数据上模仿学习，注入"感知 / 短时预测 / 条件生成 / 反思 / 动作与轨迹规划"五类能力，构建因果链基础。
- **(c) 强化学习（[[GRPO]]，1 epoch）**: 从 SFT checkpoint 出发，用一组**规则奖励**评估候选 rollout，跨整条 pipeline 优化（见公式6 与下文奖励设计）。

### 关键公式与机制

#### 公式1: [[Trajectory|未来航点轨迹]]定义

$$
\tau_{t:t+H}=\{p_{t+1},p_{t+2},\dots,p_{t+H}\},\quad p_{t+h}\in\mathbb{R}^{2}
$$

**含义**: 在 ego 中心 BEV 坐标系下，未来轨迹是 $H$ 个二维航点序列。

**符号说明**:
- $p_{t+h}\in\mathbb{R}^2$: 第 $t+h$ 步的 BEV 平面坐标
- $H$: 预测时域；动作空间可为低层控制 $a_t$（油门/刹车/转向）或整条轨迹 $\tau_{t:t+H}$

#### 公式2: VLA-World 联合分解（核心范式）

$$
p(\tau_{t:t+H},x_{t+1}\mid o_{1:t},g)=\underbrace{p(\tau_{t:t+H}\mid o_{1:t},g)}_{\text{decision / policy}}\cdot\underbrace{p(x_{t+1}\mid o_{1:t},\tau_{t+1})}_{\text{imagination / world model}}
$$

**含义**: 把"规划轨迹 + 下一帧未来"的联合分布按概率链式法则拆成**策略因子**（VLA 擅长）与**想象因子**（世界模型擅长）。纯 VLA 只优化左项、纯世界模型只优化右项；VLA-World 两项都显式建模。

**符号说明**:
- $o_{1:t}$: 历史观测；$g$: 任务目标；$x_{t+1}$: 下一帧图像；$\tau_{t+1}$: 条件化近未来演化的短时轨迹

#### 公式3: 条件化未来想象

$$
\hat{x}_{t+1}\sim p_{\psi}(x_{t+1}\mid o_{1:t},\hat{\tau}_{t:t+1})
$$

**含义**: 以历史观测与初始短时轨迹为条件，采样生成预想的近未来视图。

**符号说明**:
- $p_\psi$: 世界模型/生成器（参数 $\psi$）；$\hat\tau_{t:t+1}$: 模型先验预测的 0.5s 短时轨迹；$\hat x_{t+1}$: 当前计划下的预想未来帧

#### 公式4: 反思修正

$$
\tilde{\tau}_{t:t+H}=f_{\mathrm{ref}}\!\left(o_{1:t},\hat{x}_{t+1},\hat{\tau}_{t:t+1}\right)
$$

**含义**: 反思模块综合历史观测、想象帧、初始轨迹，修正出最终的安全长程轨迹。

**符号说明**:
- $f_{\text{ref}}$: 反思推理模块；$\tilde\tau_{t:t+H}$: 反思后的最终轨迹（兼顾预测动态与对想象未来的推理）

#### 公式5: 自回归视觉生成（预训练目标）

$$
P(Q_{t+1}^{k})=\prod_{i=1}^{N}P_{\theta}(q_{i}^{k}\mid q_{<i}^{k},h_{t},L)
$$

**含义**: 在多视角观测编码 $h_t$ 与指令 $L$ 条件下，逐 token 自回归预测相机 $k$ 的未来帧视觉 token 序列。

**符号说明**:
- $Q_{t+1}^k$: 相机 $k$ 在 $t+1$ 时刻的视觉 token 序列；$q_i^k$: 第 $i$ 个 [[VQGAN]] 码本离散 token
- $h_t=f_\phi(I_t,S_t)$: 当前多视角观测与 ego 状态编码；$L$: 描述目标视角/驾驶意图的指令（如 "generate CAM_FRONT_LEFT 0.5s later"）

#### 公式6: GRPO 阶段的加权奖励

$$
R_{\text{all}}=\lambda_{\text{fmt}} R_{\text{fmt}}+\lambda_{\text{pred}} R_{\text{pred}}+\lambda_{\text{vis}} R_{\text{vis}}+\lambda_{\text{act}} R_{\text{act}}+\lambda_{\text{traj}} R_{\text{traj}}
$$

**含义**: 规则奖励的加权和，覆盖整条 pipeline，引导输出结构正确、短时预测合理、视觉连贯、行为安全。

**符号说明 / 各奖励作用**:
- $R_{\text{fmt}}$: 格式奖励，强制 `<Perception>/<Prediction>/<Visual>/<Think>/<Action>/<Answer>` 标签结构；
- $R_{\text{pred}}$: 短时预测奖励，既鼓励准确预测 0.5s 轨迹/朝向，又强制 0.5s 预测与反思后长程轨迹的一致性；
- $R_{\text{vis}}$: 视觉约束奖励，确保生成视觉 token 数量正确且每个 token 都是码本有效项（保证可解码）；
- $R_{\text{act}}$: 动作奖励，基于 F1 分数评估高层动作正确性；
- $R_{\text{traj}}$: 轨迹奖励，确保 3s 轨迹每个区间精确且**运动学一致**（如平稳行驶时加速度变化应很小）。

#### 公式7: GRPO 组内归一化优势

$$
A_{i}=\frac{r_{i}-\mu}{\sigma},\qquad\mu=\frac{1}{G}\sum_{j}r_{j},\ \ \sigma=\mathrm{std}(r_{1},\ldots,r_{G})
$$

**含义**: 对同一 prompt 采样的 $G$ 个 rollout 用组内统计量做归一化，作为**无价值网络**的动态 baseline，鼓励超过组均值的轨迹。

**符号说明**:
- $r_i$: 第 $i$ 个 rollout 的标量奖励；$\mu,\sigma$: 组内均值/标准差

#### 公式8: GRPO 代理目标

$$
J(\theta)=\mathbb{E}\!\left[\frac{1}{G}\sum_{i=1}^{G}\min\!\left(\frac{\pi_{\theta}(\tau_{i}\mid o)}{\pi_{\theta_{\text{old}}}(\tau_{i}\mid o)}A_{i},\ \text{clip}\right)\right]-\beta\,D_{\text{KL}}(\pi_{\theta},\pi_{\text{old}})
$$

**含义**: 在 PPO 式裁剪比率上最大化优势，并用 KL 项约束策略不偏离 SFT 参考模型，防止 reward hacking，实现"自验证"（隐式丢弃幻觉/不安全轨迹）。

**符号说明**:
- $\pi_\theta/\pi_{\theta_{\text{old}}}$: 当前/旧策略；$A_i$: 公式7 的优势；$\beta$: KL 系数（实现取 $1\times10^{-2}$）；$\pi_{\text{old}}$: SFT 参考模型

#### 公式9–11: 物理接地的短时轨迹预测器（附录 A.2）

$$
\mathbf{v}_{t}=\frac{\mathbf{P}_{t}-\mathbf{P}_{t-1}}{\Delta t},\qquad\mathbf{a}_{\text{hist}}=\frac{\mathbf{v}_{t}-\mathbf{v}_{t-1}}{\Delta t}
$$

$$
\mathbf{a}_{\text{eff}}=(1-\lambda)\,\mathbf{a}_{\text{hist}}+\lambda\,\mathbf{a}_{\text{goal}},\qquad \mathbf{a}_{\text{goal}}=\frac{2}{\tau^{2}}\!\left(\Delta\mathbf{P}_{\text{ideal}}-\mathbf{v}_{t}\tau\right)
$$

$$
\hat{\mathbf{P}}_{t+\tau}=\mathbf{P}_{t}+\mathbf{v}_{t}\tau+\frac{1}{2}\mathbf{a}_{\text{eff}}\tau^{2}
$$

**含义**: 用有限差分估计当前速度 $\mathbf v_t$ 与历史惯性加速度 $\mathbf a_{\text{hist}}$，再融合目标导向加速度 $\mathbf a_{\text{goal}}$，以运动学方程外推 0.5s 后位置 —— 在"惯性延续（直行）"与"意图机动（急转）"之间平滑过渡，为后续帧生成提供几何先验。

**符号说明**:
- $\lambda\in[0,1]$: 自适应融合权重；$\Delta\mathbf P_{\text{ideal}}$: 导航指令 $c$ 要求的理论位移；$\tau$: 前瞻时域（如 0.5s）

#### 公式12–19: 理论分析（附录 A.3，节选）

联合目标 $J(\omega)=\mathbb{E}_{p_\omega(\tau,x\mid o,g)}[R(\tau,x)]$（公式13）；纯 VLA 等价于积掉未来 $x$、只建模边缘策略（公式14），从变分角度看是优化了一个**松的下界**（ELBO，公式15）—— 显式生成 $x_{t+1}$ 可**收紧该界**，用"想象未来"降低策略估计的不确定性。纯世界模型目标（公式16）仅追求像素保真、与驾驶决策弱耦合（公式17）。VLA-World 的联合梯度（公式18）同时含 Policy Gradient 与 World Model Gradient 两项，GRPO 目标（公式19）统一优化二者。

$$
\log p^{\star}(\tau\mid o,g)\geq\mathbb{E}_{x\sim q}\big[\log p^{\star}(\tau,x\mid o,g)-\log q(x\mid o,\tau)\big]
$$

**含义**: 上式是 VLA 边缘策略的 ELBO；论文核心论点是 VLA-World 直接建模联合分子 $p(\tau,x\mid o,g)$，从而比独立 VLA / 世界模型更贴合驾驶目标。

---

## 关键图表

<!-- 图片均使用 arXiv HTML 在线链接，未本地化 -->

### Figure 1: Visual Overview of VLA-World / 整体概览（teaser）

![Figure 1](https://arxiv.org/html/2604.09059v1/x1.png)

**说明**: VLA-World 三阶段渐进式学习的整体概览。先以多视角输入预测未来帧**激活视觉生成**（学生成知识）；再微调串起"感知-未来生成-规划"**学驾驶概念知识**；最后用强化学习通过与生成未来交互精修决策（学推理知识）。右侧结果显示 VLA-World 同时取得**最低碰撞率与最低 FID**，凸显其在未来生成与安全规划上的双重优势。

### Figure 2: VLA vs. World Model vs. VLA-World Paradigm / 三种范式对比

![Figure 2](https://arxiv.org/html/2604.09059v1/x2.png)

**说明**: (a) 纯 VLA：观测→推理→直接出轨迹，缺时空/世界一致建模；(b) 世界模型：观测+动作→生成未来帧，缺反思评估；(c) VLA-World：感知→短时预测→生成未来帧→**反思推理**→规划，把想象与反思闭环进同一框架。直观对应公式2 的"策略因子 × 想象因子"。

### Figure 3: Three-stage Training & Inference Pipeline / 三阶段训练与推理流程

![Figure 3](https://arxiv.org/html/2604.09059v1/x3.png)

**说明**: (a) 视觉预训练激活生成能力；(b) 监督微调把驾驶概念知识"播种"进模型；(c) 用 [[GRPO]] 强化学习像人类一样探索更高性能。展示了从生成到反思再到规划的完整数据流与各阶段目标。

### Figure 4: Qualitative Comparison vs. FSDrive (0.5s) / 与 FSDrive 的可视化对比

![Figure 4](https://arxiv.org/html/2604.09059v1/x4.png)

**说明**: 上排当前场景、下排生成的 0.5s 后未来场景；红色为模型预测轨迹、绿色为真值。相比 SOTA [[FSDrive]]，VLA-World 预测轨迹更贴近真值，生成帧时空更一致。

### Figure 5: Data Samples of Three Stages / 三阶段数据样例

![Figure 5](https://arxiv.org/html/2604.09059v1/x5.png)

**说明**: (a) 预训练样例（≈500k，输入多视角观测 + ego 坐标系/单位定义，输出未来帧视觉 token）；(b) SFT 样例（≈20k，输入增广车辆运动学/历史轨迹/任务指令，输出 `<perception>→<prediction>→<visual>→<think>→<action>→<answer>` 因果链）；(c) RL 样例。直观展示 nuScenes-GR-20K 的结构化标注。

### Figure 6: Future Frame Generation vs. FSDrive / 未来帧生成对比

![Figure 6](https://arxiv.org/html/2604.09059v1/x6.png)

**说明**: VLA-World 与 [[FSDrive]] 在生成 0.5s 后未来帧上的对比，VLA-World 生成质量更高（对应 Table 2 的 FID 9.8 vs 10.1）。

### Figure 7: 3-second Future Trajectory Comparison / 3 秒长程轨迹对比

![Figure 7](https://arxiv.org/html/2604.09059v1/x7.png)

**说明**: VLA-World 与 [[FSDrive]] 的 3s 未来轨迹预测对比，VLA-World 长程轨迹更稳、漂移更小（呼应 Table 1 中 3s L2 与碰撞率的优势）。

### Table 1: End-to-End Trajectory Planning on nuScenes / 端到端轨迹规划（主表）

L2 误差与碰撞率，分别按 ST-P3 与 UniAD 协议；$*$ 表示使用额外 ego-state 信息。

**ST-P3 协议**

| Method | LLM | L2 1s | L2 2s | L2 3s | L2 Avg. | Col. 1s | Col. 2s | Col. 3s | Col. Avg. |
|--------|-----|------|------|------|--------|--------|--------|--------|----------|
| ST-P3* [ECCV22] | - | 1.33 | 2.11 | 2.90 | 2.11 | 0.23 | 0.62 | 1.27 | 0.71 |
| VAD [ICCV23] | - | 0.69 | 1.22 | 1.83 | 1.25 | 0.06 | 0.68 | 2.52 | 1.09 |
| VAD* [ICCV23] | - | 0.17 | 0.34 | 0.60 | 0.37 | 0.04 | 0.27 | 0.67 | 0.33 |
| BEV-Planner [CVPR24] | - | 0.30 | 0.52 | 0.83 | 0.55 | 0.10 | 0.37 | 1.30 | 0.59 |
| BEV-Planner* [CVPR24] | - | 0.16 | 0.32 | 0.57 | 0.35 | 0.00 | 0.29 | 0.73 | 0.34 |
| OccWorld [ECCV24] | GPT3-like | 0.39 | 0.73 | 1.18 | 0.77 | 0.11 | 0.19 | 0.67 | 0.32 |
| Doe-1 [arxiv24] | Lumina-mGPT-7B | 0.37 | 0.67 | 1.07 | 0.70 | 0.02 | 0.14 | 0.47 | 0.21 |
| RDA-Driver* [ECCV24] | LLaVA-7B | 0.17 | 0.37 | 0.69 | 0.40 | 0.01 | 0.05 | 0.26 | 0.10 |
| EMMA* [arxiv24] | Gemini 1.0 Nano-1 | 0.14 | 0.29 | 0.54 | 0.32 | - | - | - | - |
| OmniDrive [CVPR25] | LLaVA-7B | 0.40 | 0.80 | 1.32 | 0.84 | 0.04 | 0.46 | 2.32 | 0.94 |
| OmniDrive* [CVPR25] | LLaVA-7B | 0.14 | 0.29 | 0.55 | 0.33 | 0.00 | 0.13 | 0.78 | 0.30 |
| FSDrive [NeurIPS25] | Qwen2-VL-2B | 0.28 | 0.52 | 0.80 | 0.53 | 0.06 | 0.13 | 0.32 | 0.17 |
| FSDrive* [NeurIPS25] | Qwen2-VL-2B | 0.14 | 0.25 | 0.46 | 0.28 | 0.03 | 0.06 | 0.21 | 0.10 |
| VLA-World (ours) | Qwen2-VL-2B | 0.11 | 0.27 | 0.52 | 0.30 | 0.00 | 0.03 | 0.26 | 0.10 |
| **VLA-World\* (ours)** | **Qwen2-VL-2B** | **0.10** | **0.24** | **0.45** | **0.26** | **0.02** | **0.05** | **0.18** | **0.08** |

**UniAD 协议**

| Method | LLM | L2 1s | L2 2s | L2 3s | L2 Avg. | Col. 1s | Col. 2s | Col. 3s | Col. Avg. |
|--------|-----|------|------|------|--------|--------|--------|--------|----------|
| UniAD [CVPR23] | - | 0.59 | 1.01 | 1.48 | 1.03 | 0.16 | 0.51 | 1.64 | 0.77 |
| UniAD* [CVPR23] | - | 0.20 | 0.42 | 0.75 | 0.46 | 0.02 | 0.25 | 0.84 | 0.37 |
| PreWorld [ICLR25] | - | 0.49 | 1.22 | 2.32 | 1.34 | 0.19 | 0.57 | 2.65 | 1.14 |
| ELM [ECCV24] | BLIP2-2.7B | 0.34 | 1.23 | 2.57 | 1.38 | 0.12 | 0.50 | 2.36 | 0.99 |
| FeD* [CVPR24] | LLaVA-7B | 0.27 | 0.53 | 0.94 | 0.58 | 0.00 | 0.04 | 0.52 | 0.19 |
| OccWorld [ECCV24] | GPT3-like | 0.52 | 1.27 | 2.41 | 1.40 | 0.12 | 0.40 | 2.08 | 0.87 |
| Doe-1 [arxiv24] | Lumina-mGPT-7B | 0.50 | 1.18 | 2.11 | 1.26 | 0.04 | 0.37 | 1.19 | 0.53 |
| RDA-Driver* [ECCV24] | LLaVA-7B | 0.23 | 0.73 | 1.54 | 0.80 | 0.00 | 0.13 | 0.83 | 0.32 |
| FSDrive [NeurIPS25] | Qwen2-VL-2B | 0.40 | 0.89 | 1.60 | 0.96 | 0.07 | 0.12 | 1.02 | 0.40 |
| FSDrive* [NeurIPS25] | Qwen2-VL-2B | 0.18 | 0.39 | 0.77 | 0.45 | 0.00 | 0.06 | 0.42 | 0.16 |
| VLA-World (ours) | Qwen2-VL-2B | 0.38 | 0.74 | 1.38 | 0.83 | 0.02 | 0.08 | 0.36 | 0.16 |
| **VLA-World\* (ours)** | **Qwen2-VL-2B** | **0.10** | **0.35** | **0.80** | **0.42** | **0.01** | **0.05** | **0.30** | **0.12** |

**说明**: 仅以 [[Qwen2-VL]]-2B 主干，VLA-World* 在 ST-P3 与 UniAD 两套协议上均取得最低平均 L2（0.26 / 0.42 m）与最低平均碰撞率（0.08% / 0.12%），全面超过同主干的 [[FSDrive]]*（0.28/0.45、0.10%/0.16%）。增益在 3s 长程最明显，印证"动作条件生成 + 反思修正"能减小时序漂移。

### Table 2: Future Frame Generation (FID) / 未来帧生成质量

| Method | Type | Resolution | FID ↓ |
|--------|------|-----------|-------|
| DriveGAN [CVPR21] | GAN | 256×256 | 73.4 |
| DriveDreamer [ECCV24] | Diffusion | 128×192 | 52.6 |
| Drive-WM [CVPR24] | Diffusion | 192×384 | 15.8 |
| GenAD [CVPR24] | Diffusion | 256×448 | 15.4 |
| GEM [CVPR25] | Diffusion | 576×1024 | 10.5 |
| Doe-1 [arxiv24] | Autoregressive | 384×672 | 15.9 |
| FSDrive [NeurIPS25] | Autoregressive | 128×192 | 10.1 |
| **VLA-World** | **Autoregressive** | **128×192** | **9.8** |

**说明**: VLA-World 在 nuScenes 未来帧生成上取得最低 FID 9.8，超过此前最好的 [[FSDrive]]（10.1）与高分辨率 GEM（10.5）。虽然生成只是 pipeline 的中间产物，但更高的生成保真度直接支撑了下游反思推理的可靠性。

### Table 3: Action Prediction (F1 %) / 动作预测

† 表示在 nuScenes 上训练。

| Method | forward | left | right | keep | acc. | dec. | stop |
|--------|---------|------|-------|------|------|------|------|
| Qwen2-VL-2B | 62.43 | 22.75 | 28.65 | 40.70 | 50.23 | 49.21 | 41.04 |
| Qwen2-VL-2B† | 92.60 | 61.78 | 66.52 | 56.42 | 74.32 | 76.10 | 74.85 |
| **VLA-World** | **95.88** | **74.22** | **75.06** | **60.98** | **81.42** | **80.04** | **81.24** |

**说明**: 横向（forward/left/right）与纵向（keep/acc./dec./stop）动作 F1 全面领先。相比基座 Qwen2-VL-2B，nuScenes 微调（†）大幅提升（如 left 22.75→61.78），VLA-World 在此基础上再涨（left 74.22），体现"推理自身动作后果"而非单纯模仿标签的收益。

### Table 4: Ablation — Training Stages / Data Pipeline / Rewards / 组件消融（ST-P3 L2）

| 组 | 配置 | 1s | 2s | 3s | Avg. |
|----|------|----|----|----|------|
| (a) 训练阶段 | w/o. P.T. | 0.35 | 0.56 | 0.81 | 0.57 |
| | w/o. SFT | 0.35 | 0.79 | 1.40 | 0.85 |
| | w/o. RL | 0.43 | 0.70 | 1.01 | 0.71 |
| (b) 数据管线 | w/o. Perception | 0.42 | 0.73 | 1.09 | 0.75 |
| | w/o. Generation | 0.41 | 0.67 | 0.96 | 0.68 |
| | w/o. Reasoning | 0.50 | 0.83 | 1.22 | 0.85 |
| (c) 奖励 | w/o. $R_{\text{pred}}$ | 0.17 | 0.37 | 0.69 | 0.41 |
| | w/o. $R_{\text{vis}}$ | 0.20 | 0.40 | 0.67 | 0.42 |
| | w/o. $R_{\text{act}}$ | 0.40 | 0.54 | 0.92 | 0.62 |
| | w/o. $R_{\text{traj}}$ | 0.46 | 0.75 | 0.96 | 0.72 |
| **(d) 完整** | **VLA-World** | **0.11** | **0.27** | **0.52** | **0.30** |

**关键发现**: (a) 三阶段缺一不可，且 **w/o. SFT 退化最严重（0.85）**—— 无冷启动监督的 RL 在多步结构化推理的大搜索空间里难以收敛，说明 SFT 对建立因果链至关重要；(b) **感知与推理比生成更重要**（w/o. Reasoning 0.85、w/o. Perception 0.75 > w/o. Generation 0.68）—— 视觉生成产生大量 token 主导梯度，反而可能压制性能上限的探索；(c) **轨迹奖励 $R_{\text{traj}}$ 与动作奖励 $R_{\text{act}}$ 贡献最大**（去掉后 0.72 / 0.62），表明 RL 能端到端直接优化规划。

### Table 5: Input Resolution / 输入分辨率敏感性（ST-P3 L2）

| Res. | 1s | 2s | 3s | Avg. |
|------|----|----|----|------|
| 36000 | 0.03 | 0.14 | 0.98 | 0.38 |
| **52884** | 0.11 | 0.27 | **0.52** | **0.30** |

**说明**: 高分辨率（52,884）在 3s 长程与平均 L2（0.30）上更优；低分辨率虽在 1s 短程略好，但长程漂移大。保留高保真视觉信息有助于抑制误差累积。

### Table 6: Model Size Scaling / 模型规模缩放（ST-P3 L2）

| Method | 1s | 2s | 3s | Avg. |
|--------|----|----|----|------|
| Qwen2-VL-2B | 0.11 | 0.27 | 0.52 | 0.30 |
| Qwen2.5-VL-3B | 0.05 | 0.08 | 0.76 | 0.29 |
| **Qwen2-VL-7B** | **0.03** | **0.03** | **0.47** | **0.18** |

**说明**: 清晰的 scaling law —— 主干越大性能越好，Qwen2-VL-7B 平均 L2 降到 0.18 m（相对 2B/3B 约 40% 提升），更强推理/泛化对处理复杂因果依赖（尤其长程）至关重要。

### Table 7: Mixed-task Training / 多任务混合训练消融（ST-P3 L2）

| Method | 1s | 2s | 3s | Avg. |
|--------|----|----|----|------|
| w/o. Mixed | 0.27 | 0.47 | 0.73 | 0.49 |
| **Qwen2-VL-2B（混合）** | **0.11** | **0.27** | **0.52** | **0.30** |

**说明**: 去掉多任务混合监督后平均 L2 从 0.30 涨到 0.49，说明"感知+推理+规划"混合训练对学到鲁棒表征、跨时域泛化不可或缺。

---

## 实验

### 数据集 / 基准

| 基准 | 规模 | 特点 | 用途 |
|------|------|------|------|
| [[nuScenes]] | 1000 场景（每段约 20s），28,130 训练 / 6,019 验证 / 193,082 无标注；32 线 LiDAR + 6 相机 360° | 标准端到端驾驶基准 | 训练/评测 |
| nuScenes-GR-20K | ≈20k 生成-推理样本（自建，源自 nuScenes） | 含未来帧生成 + 条件反思标注，用于 SFT/RL | 训练 |
| 预训练数据 | ≈500k | 多视角未来帧生成（视觉 token 预测） | 预训练 |

**评测指标**: L2 位移误差、碰撞率（按 ST-P3 与 UniAD 各自协议；ST-P3/VAD 取前序时刻平均，UniAD 按各时刻）；生成用 [[FID]]。

### 实现细节

- **Backbone**: [[Qwen2-VL]]-2B（沿用 [[FSDrive]] 初始化），[[VQGAN]] 视觉 tokenizer；多视角输入，最大像素 524,288
- **预训练**: 30 epoch，AdamW，lr $5\times10^{-4}$，per-device batch 16
- **SFT**: 12 epoch，AdamW，lr $1\times10^{-4}$（LLaMA-Factory 框架）
- **RL（[[GRPO]]）**: 1 epoch，lr $1\times10^{-6}$，global batch 16，每 prompt 采样 8 个候选，KL 系数 $1\times10^{-2}$（Easy-R1 框架）
- **调度**: 全程 cosine + warmup ratio 0.1，梯度累积 2
- **硬件**: 训练 8×A100 80GB（正文写 8×80GB GPU），推理 4×A100

### 关键实验结论

- **规划**: ST-P3 平均 L2 0.26 m / 碰撞率 0.08%，UniAD 平均 L2 0.42 m / 碰撞率 0.12%，两协议均 SOTA，长程（3s）优势最大。
- **生成**: FID 9.8，自回归类最佳，超 FSDrive 与扩散类高分辨率方法。
- **动作**: 横纵向动作 F1 全面领先（如 forward 95.88%）。
- **消融**: SFT 冷启动最关键；感知/推理 > 生成；$R_{\text{traj}}/R_{\text{act}}$ 奖励贡献最大；高分辨率、大模型、混合训练均正向。

---

## 批判性思考

### 优点
1. **范式清晰且有理论支撑**: "策略因子 × 想象因子"的概率分解（公式2）配合 ELBO/联合梯度分析（公式12–19），把"生成未来帮助决策"上升为收紧变分下界的理论论证，而非纯工程拼接。
2. **闭环 action→imagination→reflection 有可解释性**: 结构化标签序列让感知/预测/想象/推理/动作各步可读，反思在自生成帧上识别风险，比黑盒直接回归轨迹更透明。
3. **轻量主干即 SOTA**: 仅 Qwen2-VL-2B 就在 nuScenes 规划与生成双榜领先，且 scaling 实验（Table 6）显示还有明显上升空间，落地与扩展性都好。

### 局限性
1. **单一数据集 + 自建评测集**: 全部实验在 [[nuScenes]]，且 SFT/RL 依赖自建 nuScenes-GR-20K（生成-推理标注的质量/构造细节披露有限），跨数据集（如 Waymo、Bench2Drive 闭环）泛化未验证；开环 L2/碰撞率指标本身对真实安全的代表性也常被诟病。
2. **生成被"边缘化"的内在张力**: 消融（Table 4b）显示去掉 Generation 反而比去掉 Perception/Reasoning 影响小，作者归因于"生成 token 主导梯度压制上限"。这意味着核心卖点"想象未来"对最终规划的边际贡献并不算大，"反思"可能更多受益于结构化推理本身而非生成帧。
3. **碰撞率指标的不稳定**: VLA-World（非 *）在 UniAD 下 L2 0.83 明显逊于 FSDrive* 0.45，说明对 ego-state 信息依赖较强；部分碰撞率数值（如 ST-P3 col. 1s 0.02 vs FSDrive 0.06，但 3s 0.26 vs 0.32）波动，单点优势不一定稳健。

### 潜在改进方向
1. 引入闭环仿真评测（CARLA / Bench2Drive）与多数据集泛化，验证"想象-反思"在分布外与交互式场景的真实收益。
2. 解决"生成 token 主导梯度"问题（如解耦生成与规划的优化、用 latent 而非离散 token 想象），让生成质量真正反哺规划上限。
3. 把短时轨迹预测器（公式9–11 的物理外推 + 自适应 $\lambda$）做成可学习模块，并对 $R_{\text{all}}$ 各权重 $\lambda$ 做系统敏感度分析。

### 可复现性评估
- [ ] 代码开源（截至 v1 仅 project page，未见明确代码/权重 release 声明）
- [ ] 预训练模型（未声明）
- [x] 训练细节较完整（附录 B.2 给出各阶段 epoch/lr/batch/框架/KL 系数）
- [x] 数据集可获取（nuScenes 公开；nuScenes-GR-20K 为自建，构造流程有描述但发布状态未明）

---

## 速查卡片

> [!summary] VLA-World: Vision-Language-Action World Model for Autonomous Driving
> - **核心**: 把世界模型的"想象"与 VLA 的"反思"统一进一个自回归 VLM —— 短时轨迹驱动生成下一帧，再在生成帧上推理修正轨迹（公式2 的策略因子 × 想象因子）。
> - **方法**: Qwen2-VL-2B + VQGAN；三阶段 = 视觉预训练（激活生成）→ SFT（generation-to-think 冷启动）→ GRPO 强化学习（5 项规则奖励）；输出 `<Perception><Prediction><Visual><Think><Action><Answer>` 结构化链。
> - **结果**: nuScenes ST-P3 L2 0.26m / 碰撞 0.08%、UniAD L2 0.42m / 碰撞 0.12%（均 SOTA），生成 FID 9.8，动作 F1 全面领先；scaling 到 7B 时 L2 降至 0.18m。
> - **主页**: https://vlaworld.github.io

---

*笔记创建时间: 2026-06-29*
