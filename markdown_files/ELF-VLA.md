---
title: "Unleashing VLA Potentials in Autonomous Driving via Explicit Learning from Failures"
method_name: "ELF-VLA"
authors: [Yuechen Luo, Qimao Chen, Fang Li, Shaoqing Xu, Jaxin Liu, Ziying Song, Zhi-xin Yang, Fuxi Wen]
year: 2026
venue: CVPR
tags: [VLA, autonomous-driving, reinforcement-learning, GRPO, learning-from-failures, teacher-feedback, NAVSIM]
zotero_collection: _inbox
image_source: online
arxiv_html: https://arxiv.org/html/2603.01063v1
created: 2026-06-29
---

# Unleashing VLA Potentials in Autonomous Driving via Explicit Learning from Failures

## 元信息

| 项目 | 内容 |
|------|------|
| 作者 | Yuechen Luo, Qimao Chen, Fang Li, Shaoqing Xu, Jaxin Liu, Ziying Song, Zhi-xin Yang, Fuxi Wen |
| 机构 | 涉及澳门大学、清华大学、北京交通大学等（详见原文） |
| 会议 | CVPR 2026 |
| 类别 | VLA（Vision-Language-Action）/ 自动驾驶 |
| 日期 | 2026 |
| 项目主页 | https://github.com/luo-yc17/ELF-VLA |
| 链接 | [arXiv](https://arxiv.org/abs/2603.01063) / [Code](https://github.com/luo-yc17/ELF-VLA) |

---

## 一句话总结

> 用教师模型对失败 rollout 生成结构化诊断反馈、让学生 VLA 据此重 rollout 出高分修正轨迹并回注 [[GRPO]] 训练批次，从而突破自动驾驶 RL 微调的“性能平台期”，在 NAVSIM 上刷新 PDMS/EPDMS 与高层规划准确率。

---

## 核心贡献

1. **VLA 能力对齐的失败反馈（VLA Ability-aligned Feedback）**: 当 VLA 在某场景持续失败（所有 rollout 得分为零）时，触发一个[[Teacher Model|教师模型]]，生成与 VLA 的 "Think-then-Act" 能力分层对齐的**结构化诊断报告**，精确定位错误属于**高层规划 / 认知推理 / 轨迹执行**中的哪一层，而非只给一个稀疏标量奖励。
2. **反馈引导的修正与回注（Feedback-Guided Refinement & Re-injection）**: 学生策略据诊断报告生成一条修正后的高分轨迹，并将其**重新注入 [[GRPO]] 训练批次**，从而提供原本不存在的、有目标导向的梯度信号，使策略能解决无引导探索无法攻克的关键场景。
3. **SOTA 性能**: 在公开 [[NAVSIM]] 基准上，整体驾驶指标 [[PDMS]]（91.0）、[[EPDMS]]（87.1）与高层规划准确率（80.3）均取得新 SOTA，证明该方法释放了 VLA 的潜能而非过拟合单一指标。

---

## 问题背景

### 要解决的问题
自动驾驶 [[VLA]] 模型在 [[Reinforcement Learning|RL]] 微调阶段常陷入**性能平台期（performance plateau）**：经 [[SFT]] 后，策略的探索能力被 SFT 数据集的分布严重束缚——常见场景占绝大多数、真正考验系统的安全攸关长尾场景（如无保护左转、紧急避让）极其稀少。在这些关键场景下，**所有探索 rollout 一致失败、驾驶分全为零**（见 Figure 1 上排），形成稀疏奖励问题。

### 现有方法的局限
- 现有 VLA-RL 方法把训练期的性能评估**压缩为单一标量奖励**（如 PDMS）。当模型失败时，这个信息稀疏的奖励既**标记了失败、却无法指出失败根因**——究竟是 "think" 模块的高层规划累积误差、对关键目标的认知推理缺陷，还是低层轨迹的动力学不足，都无从区分。
- 当前主流方法普遍采用 [[GRPO]] 做 RL，奖励即驾驶分；一旦 SFT 后模型在长尾场景上无论 rollout 多少次都拿不到分，整体学习就停滞，形成平台期。

### 本文的动机
- 受 LLM 领域用**非数值反馈（文本批评、mix-policy 内化高质量数据）**提升探索与策略质量的启发，本文把"显式反馈"机制引入自动驾驶域；
- 核心思路：用一个**教师 VLM** 分析学生的错误驾驶行为，通过**结构化反馈**把错误纠正为合理动作，再把这条修正样本回注训练批，提供"原本批次里不存在"的高优势梯度方向。核心假设：**指明失败根因 + 注入修正样本 = 突破平台期**。

---

## 方法详解

ELF-VLA 包含两大组件（见 Figure 2）：(1) 两阶段监督微调（[[SFT]]），赋予模型驾驶认知与"轨迹预测 + 反馈修正"双能力；(2) 带失败反馈的 [[Reinforcement Learning|RL]] 框架（GRPO with Feedback）。整条流水线为：**驾驶知识预训练 → 混合数据 SFT → 带反馈的 GRPO 强化学习**。

### 模型架构

- **Backbone**: [[InternVL3]]-8B（300M 参数 [[InternViT]] 视觉编码器 + 7B 参数 [[Qwen2.5]] 语言模型），具备分辨率自适应视觉输入：对复杂区域细粒度处理、对简单区域粗粒度提取。
- **教师模型**: [[Qwen3-VL]]-32B，负责对错误轨迹生成结构化诊断反馈。
- **输出**: 带 [[Chain-of-Thought|CoT]] 推理（"think"）的未来轨迹（"answer"），即 "Think-then-Act"。
- **推理延迟**: 借助 vLLM 加速，生成 CoT + 轨迹仅 0.1s。

### 核心模块

#### 模块1: VLA 输入构造（Base Inputs / Feedback Inputs）

**设计动机**: 让同一个 VLA 既当**生成器**又当**修正器**，因此需要两类输入并存。

**具体实现**:
- **Base inputs** $q^{base}$: 前视图像 $I_{\text{cam}}$ + 高层导航指令 $q_{\text{com}}$（Move Forward / Turn Left / Turn Right）+ 自车状态 $q_{\text{ego}}$（速度、加速度）+ 最近三帧历史轨迹 $q_{\text{his}}=\{h_{t-3},h_{t-2},h_{t-1}\}$（2Hz）。
- **Feedback inputs** $q^{fb}$: 模型先用 base 输入产出原始响应 $o$（轨迹 + CoT），按阈值 $s$ 分类——PDMS ≥ $s$ 为"正确" $o_c$、< $s$ 为"错误" $o_w$。
  - 正确响应：feedback 由 $q_{base}$ + $o_c$ + 基于规则的正向反馈 $f^{rule}$ 组成（强化成功行为）。
  - 错误响应：触发教师模型，输入 $q^{base}$ + 错误轨迹 $o_w$ + 真值轨迹 $o_{gt}$，生成结构化反馈 $f^{teacher}$，包含 **(1) Meta Action Analysis、(2) Think Process Analysis、(3) Safety Failure Analysis、(4) Efficiency Failure Analysis、(5) Actionable Correction（含横向/纵向分量）** 五部分。

$$
q_{fb}=\begin{cases}\langle q_{base},o_{c},f^{rule}\rangle & \text{if}\ o_{c},\\[2pt] \langle q_{base},o_{w},f^{teacher}\rangle & \text{if}\ o_{w}.\end{cases}
$$

#### 模块2: 两阶段 SFT（Cognition + Refinement）

**设计动机**: 在 RL 之前，既要让模型懂驾驶常识，又要让它**具备根据反馈修正轨迹的能力**，否则 RL 阶段无法利用失败反馈。

**具体实现**:
- **Stage 1（认知预训练）**: 在大规模驾驶 Q&A 数据上预训练，注入驾驶域常识（可行驶区域估计、关键目标识别/定位、自车动作预测、交通语义）。数据来自 [[DriveLM]]、[[LingoQA]]、[[ImpromptuVLA]]、NuScenes-QA、NuInstruct、OmniDrive 等开源集，并按 CoT 范式自建 NAVSIM 多轮推理数据。
- **Stage 2（轨迹预测 + 修正）**: 在混合数据集（curated NAVSIM 规划数据带 CoT 标注 + 反馈数据集）上微调，对每个 $q^{base}$ 与 $q^{fb}$ 都用真值 $o_{gt}$ 监督，目标为最大化条件似然——从而同时获得"轨迹预测"与"基于反馈的轨迹修正"双能力。

#### 模块3: 带失败反馈的 GRPO（GRPO with Feedback）

**设计动机**: 在 GRPO 的 rollout 阶段植入反馈机制，把长尾场景里全零的稀疏奖励"撬"起来，突破平台期（见 Figure 3）。

**具体实现（一次 rollout 更新流程）**:
1. **难样本筛选（Efficient Difficult-Sample Curation）**: 朴素 RL 会把算力浪费在已掌握的简单场景上。用 SFT 模型对每个 query 采样 $N$ 条 rollout，估计均值奖励与方差，**丢弃"高均值 + 低方差"的一致成功样本**，把训练聚焦到困难（低均值低方差）与歧义（高方差）场景，将初始 85k 条精炼到 24k 高价值场景。
2. **采样与分类**: 用 base 输入采样 $\{o_i\}_{i=1}^n$，算出奖励 $\{r_i\}$（含 $r_{traj}, r_{fmt}, r_{goal}$），按阈值 $s$ 分为正确组 $\{o_c\}$ 与错误组 $\{o_w\}$。
3. **生成反馈输入与修正响应**: 对每条响应构造 $q_i^{fb}$（正确用 $f^{rule}$、错误用教师 $f_i^{teacher}$），让 VLA 自身用 $q_i^{fb}$ 重 rollout 出修正响应 $o_i^{fb}$ 并评分。
4. **挑选高分修正并回注**: 取原批最高奖励 $r_{\max}$，从超过 $r_{\max}$ 的修正响应里选 $k$ 条，与原始 $n$ 条拼成最终集 $D_{final}$（$n+k$ 条）联合优化。
5. **Policy Shaping**: 对修正轨迹的重要性比施加形变 $f(x)=x/(x+\gamma)$，防止"高优势 + 低概率"的修正轨迹引发训练坍塌与格式错误。

### 关键公式与机制

#### 公式1: 两阶段 SFT 似然目标

$$
\mathcal{L}_{\text{SFT}}=\mathbb{E}_{(q,o)\sim\mathcal{D}}\big[-\log\pi_{\theta}(o\mid q)\big]
$$

**含义**: 在混合数据上最大化真值轨迹的条件对数似然，让模型同时学会轨迹预测与反馈修正。

**符号说明**:
- $\mathcal{D}$: 混合数据集 $\{(q_{base},o_{gt}),(q_{fb},o_{gt})\}$
- $\pi_{\theta}$: VLA 模型；$q$: base 或 feedback 查询；$o$: 输出轨迹（+CoT）

#### 公式2: 总奖励

$$
r=r_{traj}+r_{fmt}+r_{goal}
$$

**含义**: RL 阶段总奖励由三项组成，激励有效驾驶并稳定输出格式。

**符号说明**:
- $r_{traj}$: 基于 [[PDMS]] 的轨迹奖励，连续值 $\in[0,1]$
- $r_{fmt}$: 格式奖励（二值），0.5 分检验 `<think>…</think>` 与 `<answer>…</answer>` 结构、0.5 分检验轨迹点语法可解析
- $r_{goal}$: 终点对齐奖励（按 L1 距离分档）

#### 公式3: Goal Reward 分档

$$
r_{goal}=\begin{cases}1 & \text{if }0<dis<2\\ 0.8 & \text{if }2\leq dis<4\\ 0.6 & \text{if }4\leq dis<6\\ 0.4 & \text{if }6\leq dis<10\\ 0.2 & \text{if }10\leq dis<15\\ 0 & \text{if }dis>15\end{cases}
$$

**含义**: 预测终点与真值终点 L1 距离 $dis$（米）越近，分级奖励越高，促使端点精确对齐。

#### 公式4: 带反馈的 GRPO 目标

$$
\mathcal{J}(\theta)=\mathbb{E}_{D_{final}\sim\pi_{\theta_{old}}}\!\left[\frac{1}{n}\sum_{i=1}^{n}\mathcal{J}_{i}+\frac{1}{k}\sum_{j=1}^{k}\mathcal{J}_{j}^{fb}-\beta\,\mathbb{D}_{KL}\right]
$$

**含义**: 最终目标同时优化 $n$ 条原始响应项 $\mathcal{J}_i$ 与 $k$ 条回注的修正响应项 $\mathcal{J}_j^{fb}$，并以 KL 正则约束策略不偏离参考。

**符号说明**:
- $D_{final}=\{(q_i^{final},o_i^{final})\}_{i=1}^{n+k}$: 原始 + 修正的最终训练集
- $\beta$: KL 惩罚系数；$\mathbb{D}_{KL}$: 与旧/参考策略的 KL 散度

#### 公式5: 原始响应与修正响应的优化项

$$
\mathcal{J}_{i}=\min\big(c_{i}(\theta)A_{i},\ \text{clip}(c_{i}(\theta),1-\epsilon,1+\epsilon)A_{i}\big)
$$

$$
\mathcal{J}_{j}^{fb}=f\big(c_{j}^{fb}(\theta)\big)A_{j}^{fb}
$$

**含义**: 原始响应沿用标准 PPO 式 clip 目标；修正响应项不用 clip，而是用 Policy Shaping 函数 $f(\cdot)$ 形变其重要性比，缓和低概率修正样本的梯度。

**符号说明**:
- $c_i(\theta)$: 重要性比；$A_i$: 优势；$\epsilon$: clip 范围
- $A_j^{fb}$: 修正响应的优势；$f(c_j^{fb})$: 形变后的修正响应比

#### 公式6: 重要性比与 Policy Shaping

$$
c_{i}(\theta)=\frac{\pi_{\theta}(o_{i}\mid q^{base})}{\pi_{old}(o_{i}\mid q^{base})},\quad f\big(c_{j}^{fb}(\theta)\big)=\frac{\pi_{\theta}(o_{j}^{fd}\mid q^{base})}{\pi_{\theta}(o_{j}^{fd}\mid q^{base})+\gamma}
$$

**含义**: 原始响应用标准比；修正响应的"比"被 $f(x)=x/(x+\gamma)$ 压制，避免高优势低概率修正项主导梯度而坍塌。

**符号说明**:
- $\gamma$: Policy Shaping 权重（取 0.1）
- $o_j^{fd}$: 第 $j$ 条修正响应

#### 公式7: 统一优势归一化

$$
r_{union}=\{r_{j}\}_{j=1}^{n}\cup\{r_{j'}^{fb}\}_{j'=1}^{k}
$$

$$
A_{i}=\frac{r_{i}-\text{mean}(r_{union})}{\text{std}(r_{union})},\quad A_{i}^{fb}=\frac{r_{j}^{fb}-\text{mean}(r_{union})}{\text{std}(r_{union})}
$$

**含义**: 把原始与修正响应的奖励合并成一个统一池 $r_{union}$，再统一做组内标准化得优势。修正样本的高分会拉高整组基准，从而给原本全零的样本提供有效的相对梯度。

#### 公式8: NAVSIMv1 的 PDMS

$$
PDMS=NC\times DAC\times\left(\frac{5\times EP+5\times TTC+2\times C}{12}\right)
$$

**含义**: 五个子指标的乘性 + 加权综合闭环规划分。$NC$/$DAC$ 作为乘性硬约束（碰撞或越界直接归零）。

**符号说明**: $NC$=No At-Fault Collision，$DAC$=Drivable Area Compliance，$EP$=Ego Progress，$TTC$=Time-to-Collision，$C$=Comfort。

#### 公式9: NAVSIMv2 的 EPDMS

$$
\begin{split}EPDMS={}&NC\times DAC\times DDC\times TLC\times\\ &\left(\frac{5EP+2LK+2HC+5TTC+2EC}{16}\right)\end{split}
$$

**含义**: 扩展版闭环指标，新增 $DDC$（行驶方向合规）、$TLC$（红绿灯合规）、$LK$（车道保持）、$HC$（历史舒适）、$EC$（扩展舒适）等惩罚/加权项，更全面也更难刷分。

---

## 关键图表

<!-- 图片均使用 arXiv HTML 在线链接，未本地化 -->

### Figure 1: General VLA-RL vs ELF-VLA / 平台期对比

![Figure 1](https://arxiv.org/html/2603.01063v1/figs/introv6.png)

**说明**: 上排是普通 VLA 的 RL 微调——某些场景下策略 rollout 一致拿低分，被"困住"无法发现更好策略，形成性能平台期；下排是 ELF-VLA——用教师模型给出结构化反馈，据此重 rollout 出修正样本，强迫策略突破平台。这张图点明了全文要解决的核心痛点。

### Figure 2: Overview of ELF-VLA / 整体三阶段流程

![Figure 2](https://arxiv.org/html/2603.01063v1/figs/mainv4.png)

**说明**: 三阶段总览。先在驾驶 Q&A 数据上预训练注入常识；再在 "Base Inputs + Feedback Inputs" 混合数据上 SFT，同时学会轨迹预测与反馈修正；最后 RL 阶段用教师模型生成反馈，降低全零奖励 rollout 的比例。

### Figure 3: GRPO with Feedback / 带反馈的 GRPO 数据流

![Figure 3](https://arxiv.org/html/2603.01063v1/figs/methodv3.png)

**说明**: 策略模型先生成初始响应；据奖励，教师模型（[[Qwen3-VL]]-32B）给出反馈，引导策略采样更优的修正响应；挑出一条高质量修正响应与初始响应集合并联合优化，并对最终概率施加 Policy Shaping。这是方法的核心机制图。

### Figure 4: Total-Failure Ratio / RL 期全失败样本比例

![Figure 4](https://arxiv.org/html/2603.01063v1/figs/rollout_ratio.png)

**说明**: RL 训练期 GRPO、GT-GRPO、Rule-GRPO、ELF-VLA 的"全失败样本"比例（某样本所有 rollout 在 PDMS<$s$、NC=0、DAC=0 上同时失败）。中间策略 GT-GRPO/Rule-GRPO 能部分降低失败率，而 ELF-VLA 把全失败比例压得最低——直接量化"突破平台期"的效果。

### Figure 5: Trajectory Refinement Visualization / 修正过程可视化

![Figure 5](https://arxiv.org/html/2603.01063v1/figs/visual.jpg)

**说明**: NAVSIM 上一个复杂左转场景。初始错误轨迹（红）因严重误估关键障碍物（预测 15.57m 前、8.11m 左）险些碰撞；教师反馈精准定位该 "Think Process" 错误并给出更准位置（11.43m 前、4.11m 左）+ 横/纵向可执行修正；模型据此生成修正轨迹（蓝），安全避障。绿为真值。彩色文字标出反馈中实际生效的修正点。

### Figure 6: Prompt Design — Base Inputs / 基础输入提示词

![Figure 6](https://arxiv.org/html/2603.01063v1/figs/base_prompt.jpg)

**说明**: VLA 模型 base 输入的提示词设计：提供当前速度、加速度与历史轨迹，帮助模型更好地预测路径。

### Figure 7: Prompt Design — Teacher Feedback Inputs / 教师反馈提示词

![Figure 7](https://arxiv.org/html/2603.01063v1/figs/prompt1.jpg)

**说明**: 教师模型（Qwen3-VL-32B）的反馈输入提示词：以错误轨迹 $o_w$、真值 $o_{gt}$、详细 NAVSIM 指标分与任务要求为输入，诊断失败根因并生成结构化纠正指导。

### Figure 8: Comparison of Feedback Mechanisms / 反馈机制对比

![Figure 8](https://arxiv.org/html/2603.01063v1/figs/feedback.jpg)

**说明**: (a) Rule-GRPO 只给二值（对/错）信号；(b) ELF-VLA 的教师反馈给出**结构化诊断 + 具体可执行策略**来引导轨迹修正。直观对比说明本文反馈"更细粒度"的优势。

### Figure 9: High-Level Actions & GT Generation / 高层动作与真值标注（附录）

![Figure 9](https://arxiv.org/html/2603.01063v1/figs/meta1.jpg)

**说明**: 上方列出纵向/横向规划的离散类别；下方展示标注准则——纵向状态由滑窗加速度分析确定，横向行为依据车辆轨迹与地图拓扑的关系判定。这是高层规划准确率（Table 4）评测的真值来源。

### Figure 10–11: More Refinement Visualizations / 更多修正可视化（附录）

![Figure 10](https://arxiv.org/html/2603.01063v1/figs/visual1.jpg)
![Figure 11](https://arxiv.org/html/2603.01063v1/figs/visual2.jpg)

**说明**: 更多复杂场景的修正实例（红=初始错误轨迹、绿=真值、蓝=修正轨迹）。强调结构化反馈不仅校验高层规划，更**纠正中间 CoT 推理**，缓解误差累积，避免 CoT 引入幻觉或级联错误。

### Table 1: NAVSIMv1 PDMS / 与 SOTA 对比

| Method | Image | Lidar | NC↑ | DAC↑ | TTC↑ | CF↑ | EP↑ | PDMS↑ |
|--------|-------|-------|-----|------|------|-----|-----|-------|
| Constant Velocity | | | 68.0 | 57.8 | 50.0 | 100 | 19.4 | 20.6 |
| Ego Status MLP | | | 93.0 | 77.3 | 83.6 | 100 | 62.8 | 65.6 |
| UniAD | ✓ | | 97.8 | 91.9 | 92.9 | 100 | 78.8 | 83.4 |
| TransFuser | ✓ | ✓ | 97.7 | 92.8 | 92.8 | 100 | 84.0 | 84.0 |
| DiffusionDrive | ✓ | ✓ | 98.2 | 96.2 | 94.7 | 100 | 82.2 | 88.1 |
| WoTE | ✓ | ✓ | 98.5 | 96.8 | 94.9 | 99.9 | 81.9 | 88.3 |
| Hydra-NeXt | ✓ | | 98.1 | 97.7 | 94.6 | 100 | 81.8 | 88.6 |
| AutoVLA-3B | ✓ | | 98.4 | 95.6 | 98.0 | 100 | 81.9 | 89.1 |
| DriveVLA-W0-3B | ✓ | | 98.7 | 99.1 | 95.3 | 99.3 | 83.3 | 90.3 |
| GoalFlow | ✓ | ✓ | 98.4 | 98.3 | 94.6 | 100 | 85.0 | 90.3 |
| InternVL3-8B-SFT | ✓ | | 98.5 | 95.5 | 95.3 | 100 | 81.2 | 87.4 |
| InternVL3-8B-RL | ✓ | | 98.5 | 96.7 | 95.4 | 100 | 83.2 | 89.0 |
| **ELF-VLA-8B (Ours)** | ✓ | | **98.9** | 98.1 | **96.0** | 100 | **85.3** | **91.0** |

**说明**: 仅用图像（vision-only）即达 91.0 PDMS，刷新 SOTA，较此前最佳 vision-only 方法 DriveVLA(-W0) 提升 0.7；并分别超过自家 SFT(87.4) 与传统 RL(89.0) 基线 3.6 与 2.0 PDMS。

### Table 2: NAVSIMv2 EPDMS / 与 SOTA 对比

| Method | NC↑ | DAC↑ | DDC↑ | TLC↑ | EP↑ | TTC↑ | LK↑ | HC↑ | EC↑ | EPDMS↑ |
|--------|-----|------|------|------|-----|------|-----|-----|-----|--------|
| HydraMDP++ | 97.2 | 97.5 | 99.4 | 99.6 | 83.1 | 96.5 | 94.4 | 98.2 | 70.9 | 81.4 |
| DriveSuprem | 97.5 | 96.5 | 99.4 | 99.6 | 88.4 | 96.6 | 95.5 | 98.3 | 77.0 | 83.1 |
| Recogdrive-8B | 98.3 | 95.2 | 99.5 | 99.8 | 87.1 | 97.5 | 96.6 | 98.3 | 86.5 | 83.6 |
| DiffusionDrive | 98.2 | 95.9 | 99.4 | 99.8 | 87.5 | 97.3 | 96.8 | 98.3 | 87.7 | 84.5 |
| DriveVLA-W0-3B | 98.5 | 99.1 | 98.0 | 99.7 | 86.4 | 98.1 | 93.2 | 97.9 | 58.9 | 86.1 |
| **ELF-VLA-8B (Ours)** | **98.9** | 98.1 | 99.4 | **99.8** | **88.5** | **98.4** | **96.9** | 98.3 | 87.2 | **87.1** |

**说明**: 在更全面、更难刷分的 EPDMS 上仍达 87.1 新 SOTA，超过此前最佳 DriveVLA-W0 1.0；证明 ELF-VLA 不是过拟合 PDMS 单指标，而具备稳健泛化。

### Table 3: ELF-VLA vs GRPO / 反馈策略对比（NAVSIMv1）

| Method | NC↑ | DAC↑ | TTC↑ | CF↑ | EP↑ | PDMS↑ |
|--------|-----|------|------|-----|-----|-------|
| SFT | 98.5 | 95.5 | 95.3 | 100 | 81.3 | 87.4 |
| GRPO | 98.5 | 96.7 | 95.4 | 100 | 83.2 | 89.0 |
| GT-GRPO | 98.1 | 97.1 | 93.5 | 100 | 85.2 | 89.2 |
| Rule-GRPO | 98.3 | 97.3 | 94.8 | 100 | 84.5 | 89.6 |
| **ELF-VLA** | **98.9** | **98.1** | **96.0** | 100 | **85.3** | **91.0** |

**说明**: ELF-VLA 超传统 GRPO 2.0、超 GT-GRPO 1.8、超 Rule-GRPO 1.4 PDMS。GT-GRPO 因真值轨迹与 VLA 生成分布偏移大、似然低而难优化；Rule-GRPO 反馈太简单（近似自我精炼）缺乏细粒度引导；ELF-VLA 用教师的通用知识做结构化深度分析，产出更易优化的修正轨迹。

### Table 4: High-Level Planning Accuracy / 高层规划准确率（NAVSIM）

| Method | Speed Acc.↑ | Path Acc.↑ | Accuracy↑ |
|--------|-------------|------------|-----------|
| Qwen2.5-VL-7B | 37.8 | 61.3 | 19.1 |
| InternVL3-8B | 40.9 | 58.7 | 20.1 |
| Qwen2.5-VL-32B | 46.6 | 55.3 | 27.6 |
| Qwen2.5-VL-72B | 49.4 | 62.6 | 28.7 |
| SFT | 84.2 | 90.7 | 79.2 |
| GRPO | 84.3 | 90.8 | 79.3 |
| GT-GRPO | 83.5 | 90.5 | 78.4 |
| Rule-GRPO | 84.5 | 91.2 | 79.5 |
| **ELF-VLA** | **85.8** | **92.5** | **80.3** |

**说明**: Accuracy 要求纵向速度 + 横向路径 meta-action 全部精确匹配真值。ELF-VLA 在三项均最优（整体 80.3），远超未微调的通用大 VLM（≤28.7），也优于各 GRPO 变体。

### Table 5: Ablation — RL Training Data / 训练数据量与构成

| Num. | NC↑ | DAC↑ | TTC↑ | CF↑ | EP↑ | PDMS↑ |
|------|-----|------|------|-----|-----|-------|
| 85k | 98.5 | 96.8 | 95.3 | 100 | 83.4 | 89.1 |
| 24k † (随机采样) | 98.4 | 96.8 | 95.2 | 100 | 83.1 | 88.9 |
| **24k\* (按 3.3 节精选)** | **98.9** | **98.1** | **96.0** | 100 | **85.3** | **91.0** |

**说明**: 全量 85k(89.1) 与随机 24k(88.9) 都次优；按难样本筛选策略精选的 24k\* 达 91.0。说明全量数据被简单场景主导、稀释了梯度信号；精选 + 反馈机制聚焦复杂场景才高效。

### Table 6: Ablation — Refinement Count k & Policy Shaping / 修正条数与 PS

| k | PS | NC↑ | DAC↑ | TTC↑ | CF↑ | EP↑ | PDMS↑ |
|---|----|-----|------|------|-----|-----|-------|
| 4 | ✓ | 98.5 | 96.7 | 95.4 | 100 | 83.3 | 89.0 |
| 2 | ✓ | 98.2 | 97.5 | 94.3 | 100 | 84.9 | 89.7 |
| 1 | ✗ | 98.5 | 97.0 | 94.9 | 100 | 83.9 | 89.3 |
| **1** | **✓** | **98.9** | **98.1** | **96.0** | 100 | **85.3** | **91.0** |

**说明**: $k=1$ 最优（91.0），增大 $k$ 反降至 $k=4$ 时 89.0——多条反馈响应会分散策略。去掉 Policy Shaping（$k=1$）PDMS 从 91.0 掉到 89.3（-1.7），证明 PS 对防止训练坍塌、让模型从"高优势低概率"修正轨迹中学习至关重要。

### Table 7: Training Hyperparameters / 三阶段训练超参（附录）

| 阶段 | 关键超参 |
|------|----------|
| Stage 1 (Pretrain) | epochs 2, batch 1, lr $1\times10^{-5}$, grad accum 4, weight decay 0.05, weight ratio 0.05 |
| Stage 2 (SFT) | epochs 2, batch 2, lr $4\times10^{-5}$, grad accum 2, weight decay 0.05, weight ratio 0.05 |
| Stage 3 (RL) | epochs 3, batch 2, lr $2\times10^{-6}$, grad accum 16, weight decay 0.05, weight ratio 0.05, generations 8, iterations 2, temperature 1.2, 阈值 $s=0.8$, 修正数 $k=1$, Policy Shaping $\gamma=0.1$ |

**说明**: 完整复现配置。RL 用 32× NVIDIA H20，约 18 小时；前两阶段 16 GPU，分别约 2 天 / 8 小时。

### Table 8: Ablation — Training Pipeline / 三阶段流水线消融

| Model | NC↑ | DAC↑ | TTC↑ | CF↑ | EP↑ | PDMS↑ |
|-------|-----|------|------|-----|-----|-------|
| SFT | 98.5 | 93.4 | 95.1 | 100 | 78.8 | 85.3 |
| Pre+SFT | 98.5 | 95.5 | 95.3 | 100 | 81.2 | 87.4 |
| **Pre+SFT+RL** | **98.9** | **98.1** | **96.0** | 100 | **85.3** | **91.0** |

**说明**: 仅 SFT 85.3 → 加预训练 +2.1（87.4）→ 加 Feedback-GRPO +3.6（91.0）。预训练与带反馈 RL 各自贡献明确。

### Table 9: Performance without Pre-training / 去预训练（附录）

| Method | NC↑ | DAC↑ | TTC↑ | CF↑ | EP↑ | PDMS↑ |
|--------|-----|------|------|-----|-----|-------|
| InternVL3-8B (w/o pretrain) | 97.9 | 93.9 | 94.4 | 100 | 82.8 | 86.9 |
| **ELF-VLA-8B (w/o pretrain)** | **98.1** | **97.0** | 94.4 | 100 | **86.2** | **90.0** |

**说明**: 即便不做驾驶 QA 预训练，ELF-VLA 仍达 90.0，较基线(86.9) 提升 +3.1。证明增益主要来自 GRPO-with-Feedback 设计，而非单纯预训练数据。

### Table 10: Ablation — Feedback Threshold s / 反馈触发阈值

| Threshold | NC↑ | DAC↑ | TTC↑ | CF↑ | EP↑ | PDMS↑ |
|-----------|-----|------|------|-----|-----|-------|
| 0 | 98.4 | 97.1 | 94.3 | 100 | 84.7 | 89.4 |
| 0.5 | 98.6 | 97.4 | 95.3 | 100 | 84.9 | 90.1 |
| 0.9 | 98.4 | 97.3 | 94.9 | 100 | 84.8 | 89.8 |
| **0.8** | **98.9** | **98.1** | **96.0** | 100 | **85.3** | **91.0** |

**说明**: $s=0.8$ 最优。过低（0/0.5）只纠正完全失败、忽略轻微次优样本；过高（0.9）则强行修正 $[0.8,0.9)$ 这些已足够好的响应，引入噪声、使优化偏离最优策略。$s=0.8$ 在"修正覆盖面"与"训练稳定性"间取平衡。

---

## 实验

### 数据集 / 基准

| 基准 / 数据 | 规模 | 特点 | 用途 |
|------|------|------|------|
| [[NAVSIM]] v1 | 评测用 PDMS | 基于 OpenScene 的规划导向闭环（非反应式）基准 | 训练/测试 |
| NAVSIM v2 | 评测用 EPDMS | 扩展指标，含 DDC/TLC/LK/HC/EC 等更全面 | 测试 |
| 驾驶 QA 预训练集 | DriveLM/LingoQA/ImpromptuVLA/NuScenes-QA/NuInstruct/OmniDrive | 注入驾驶常识 | Stage1 预训练 |
| 反馈 SFT 数据 | 4k 正确 + 4k 错误响应（配对反馈） | 正确配 $f^{rule}$、错误配教师 $f^{teacher}$ | Stage2 SFT |
| RL 精选集 | 85k → 24k 高价值场景 | 难样本 + 歧义样本 | Stage3 RL |

### 实现细节

- **Backbone**: [[InternVL3]]-8B（InternViT-300M + Qwen2.5-7B，分辨率自适应视觉输入）
- **教师模型**: [[Qwen3-VL]]-32B
- **三阶段**: 驾驶 QA 预训练 → 混合（NAVSIM CoT + 反馈）SFT → GRPO RL
- **RL 关键参数**: 8 rollouts/batch、阈值 $s=0.8$、Policy Shaping $\gamma=0.1$、修正数 $k=1$、temperature 1.2、2 iterations、lr $2\times10^{-6}$
- **硬件**: 32× NVIDIA H20（RL 约 18h）；推理用 vLLM 加速，CoT+轨迹延迟 0.1s
- **评测指标**: 高层规划准确率（纵向速度 + 横向路径全匹配）、PDMS（v1）、EPDMS（v2）

### 关键实验结论

- **主结果**: NAVSIMv1 PDMS 91.0、NAVSIMv2 EPDMS 87.1、高层规划准确率 80.3，均为 SOTA。
- **反馈策略对比**: ELF-VLA(91.0) > Rule-GRPO(89.6) > GT-GRPO(89.2) > GRPO(89.0) > SFT(87.4)；结构化教师反馈优于真值直接注入与规则反馈。
- **失败率（Fig 4）**: ELF-VLA 把"全失败样本"比例压到最低，直接印证突破平台期。
- **数据消融**: 精选 24k\* > 全量 85k > 随机 24k，难样本筛选关键。
- **超参消融**: $k=1$ 最优、Policy Shaping 必需（去掉 -1.7）、阈值 $s=0.8$ 最佳。
- **流水线消融**: SFT 85.3 → +预训练 87.4 → +Feedback-RL 91.0；去预训练仍达 90.0。

---

## 批判性思考

### 优点
1. **失败诊断粒度对齐 VLA 能力分层**: 把模糊标量奖励替换为"规划 / 推理 / 执行"三层结构化报告，并对应 "Think-then-Act"，比单纯文本批评更有针对性，可解释性强。
2. **"修正样本回注 + 统一优势归一化"巧解稀疏奖励**: 通过把高分修正轨迹并入同一奖励池做组内标准化，使原本全零的难样本获得有效相对梯度——直接对症平台期，并用 Fig 4 的失败率下降给出可量化证据。
3. **工程严谨且消融充分**: 难样本筛选、Policy Shaping、阈值 $s$、修正数 $k$、去预训练等都有消融；vision-only 即 SOTA，推理 0.1s，复现细节（Table 7）完整，代码开源。

### 局限性
1. **强依赖外部教师模型**: 学生性能被教师（Qwen3-VL-32B）的分析能力**封顶**，作者自己也承认这是主要限制；教师产出的"精确定位"（如障碍物坐标）质量难以保证，可能引入新的系统性偏差。
2. **仅在 NAVSIM 非反应式仿真上验证**: 缺乏闭环/反应式与多数据集评估，"突破平台期"的结论在真正交互式环境中的可迁移性未知。
3. **绝对增益偏小且基线同源**: 相对自家 RL 仅 +2.0 PDMS；NAVSIM 各方法 PDMS 已普遍 88–91，提升空间被指标饱和压缩，子指标如 DAC(98.1) 反不及 DriveVLA-W0(99.1)，部分增益来自 EP（进度）。
4. **教师调用成本与训练开销**: 每个失败样本都要 32B 教师重推理 + 学生重 rollout，RL 期算力/时间成本显著，论文未给出反馈触发频率与额外开销的量化。

### 潜在改进方向
1. 探索更弱/自蒸馏教师或让学生逐步"内化"教师能力，摆脱性能被教师封顶；做不同教师模型的对比（作者列为 future work）。
2. 迁移到闭环反应式仿真与真实/多数据集评测，验证方法本身的普适性。
3. 把"结构化反馈"做成可学习的诊断头，减少对外部大模型 prompt 工程的依赖；量化反馈触发率与算力—收益曲线。

### 可复现性评估
- [x] 代码开源（https://github.com/luo-yc17/ELF-VLA）
- [ ] 预训练模型（未明确声明 release 权重）
- [x] 训练细节完整（附录 Table 7 给出三阶段超参与硬件）
- [x] 数据集可获取（NAVSIM 公开；预训练 QA 集均为开源；反馈数据构造流程已述）

---

## 速查卡片

> [!summary] ELF-VLA: Explicit Learning from Failures for Driving VLA
> - **核心**: 教师模型对失败 rollout 生成"规划/推理/执行"分层结构化诊断，学生据此重 rollout 出高分修正轨迹并回注 GRPO 批次，撬动全零稀疏奖励，突破 RL 平台期。
> - **方法**: InternVL3-8B 学生 + Qwen3-VL-32B 教师；驾驶 QA 预训练 → 混合(CoT+反馈)SFT → GRPO-with-Feedback（难样本筛选 85k→24k、阈值 s=0.8、k=1 修正、Policy Shaping γ=0.1、统一优势归一化）。
> - **结果**: NAVSIMv1 PDMS 91.0 / NAVSIMv2 EPDMS 87.1 / 高层规划 80.3，均 SOTA（vision-only），推理 0.1s。
> - **代码**: https://github.com/luo-yc17/ELF-VLA

---

*笔记创建时间: 2026-06-29*
