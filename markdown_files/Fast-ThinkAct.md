---
title: "Fast-ThinkAct: Efficient Vision-Language-Action Reasoning via Verbalizable Latent Planning"
method_name: "Fast-ThinkAct"
authors: [Chi-Pin Huang, Yunze Man, Zhiding Yu, Min-Hung Chen, Jan Kautz, Yu-Chiang Frank Wang, Fu-En Yang]
year: 2026
venue: CVPR
tags: [VLA, reasoning-vla, latent-reasoning, chain-of-thought, preference-distillation, efficient-reasoning, flow-matching]
zotero_collection: _inbox
image_source: online
arxiv_html: https://arxiv.org/html/2601.09708v2
created: 2026-06-29
---

# Fast-ThinkAct: Efficient Vision-Language-Action Reasoning via Verbalizable Latent Planning

## 元信息

| 项目 | 内容 |
|------|------|
| 作者 | Chi-Pin Huang, Yunze Man, Zhiding Yu, Min-Hung Chen, Jan Kautz, Yu-Chiang Frank Wang, Fu-En Yang |
| 机构 | NVIDIA、台湾大学（NTU）、UIUC（作者交叉单位） |
| 会议 | CVPR 2026 |
| 类别 | VLA（Vision-Language-Action）/ 推理型 VLA / 高效推理 |
| 日期 | 2026-01（arXiv v2） |
| 项目主页 | https://jasper0314-huang.github.io/fast-thinkact/ |
| 链接 | [arXiv](https://arxiv.org/abs/2601.09708) / [Project](https://jasper0314-huang.github.io/fast-thinkact/) |

---

## 一句话总结

> 把推理型 VLA 冗长的显式文本 CoT（约 250 token）蒸馏成少量（如 6 个）可被 verbalizer 还原的连续隐式 latent token，用偏好引导 + 视觉轨迹对齐迁移语言与视觉规划能力，在保持长程规划/失败恢复/小样本适应的同时把推理延迟最多降低 89.3%。

---

## 核心贡献

1. **可言语化的隐式推理（Verbalizable Latent CoT）**: 提出把推理压缩进少量连续 latent，并用一个 [[Verbalizer|verbalizer LLM]] 把 latent 解码回自然语言，从而在“没有 latent 直接监督”的困境下，借自然语言空间提供监督信号，既高效又可解释。
2. **偏好引导 + 轨迹对齐的师生蒸馏**: 用文本教师 [[GRPO]] 训练产生的 advantage 作为质量信号构造偏好对（最高/最低 advantage trace），以 [[DPO]] 式目标让学生 latent 朝高质量推理对齐、压制低质量推理；同时用 L2 对齐教师/学生 `<answer>` 隐状态来迁移**视觉轨迹规划**能力。
3. **推理增强的策略学习（Reasoning-Enhanced Policy Learning）**: 把学生 VLM 中 spatial token 的 KV cache 作为视觉规划上下文 $c_t$，桥接到扩散 Transformer 动作模型（[[RDT]] / [[DiT-Policy]]），冻结 VLM 只训动作头，实现高层视觉规划到底层动作的连接。
4. **效率-性能双赢**: 相比 SOTA 推理 VLA（[[ThinkAct]]-7B / [[MolmoAct]]-7B），推理延迟下降最多 89.3%（3B 模型 7× 加速），同时在多项 manipulation 与 embodied reasoning 基准上反超。

---

## 问题背景

### 要解决的问题
如何让**推理型 [[VLA]]** 在保留显式 [[Chain-of-Thought|CoT]] 带来的泛化/规划收益的同时，**消除冗长推理链造成的高推理延迟**——使其满足机器人 1–15 Hz 的实时决策要求，而不是当前 CoT-VLA 那种约 0.1 Hz（每决策数秒）的瓶颈。

### 现有方法的局限
1. **基础 VLA（OpenVLA、$\pi_0$、Magma 等）**: 纯模仿学习，擅长基础抓放但难以长程规划、失败自纠、泛化到新场景。
2. **监督 CoT 推理 VLA（Embodied CoT、CoT-VLA、MolmoAct）**: 需要大量推理标注，受训练数据覆盖限制。
3. **RL 推理 VLA（[[ThinkAct]]）**: 用 action-aligned 视觉奖励生成长文本 CoT，泛化好但**推理延迟极高**，且冗长链条会引入与动作无关的杂讯，反而损害动作质量。
4. **LLM 高效推理技术**: 长度惩罚（训练不稳）、reasoning dropout（[[ECoT-Lite]]，直接删 token 会丢关键信息、规划不一致）、latent reasoning（Coconut/CODI/Soft Thinking，但**无法直接迁移到 VLA**，因为缺乏空间-时序理解与“语义推理→具身控制”的桥接）。

### 本文的动机
- **latent 空间没有直接监督**这一核心难题，作者用 verbalizer 把 latent 解码回文本来"接地"，让 latent 学习有了可优化的语言目标。
- 教师 GRPO 天然产出**带质量差异**的多条 trace，advantage 正好是质量指标——可免费构造偏好对做蒸馏。
- 单纯压缩文本（删 token / 限长 / 长度惩罚）都会掉点（见 Table 6），而压成**连续 latent** 既保信息又快。
- 视觉规划能力必须显式迁移，故引入 `<answer>` 隐状态对齐 + 并行 spatial token 预测 waypoint，避免学生只学到语言而丢了空间。

---

## 方法详解

### 模型架构

Fast-ThinkAct 在推理阶段由两个模块串联组成（训练时额外有教师与 verbalizer，见 Figure 2）：
- **输入**: 语言指令 $l$ + 视觉观测 $o_t$
- **推理主干（学生 VLM $\mathcal{F}_\theta$）**: [[Qwen2.5-VL]]-3B；自回归生成 $M$ 个连续 latent token $\mathbf{z}$ 作内部隐式推理，并附加 $K$ 个 [[Spatial Token|spatial token]] 并行预测视觉轨迹 waypoint，产出视觉规划隐表征 $c_t$。
- **动作模型 $\pi_\phi$**: 扩散 Transformer（[[RDT]] 用于 LIBERO/RoboTwin2.0，[[DiT-Policy]] 用于 SimplerEnv），以 $c_t$（来自 spatial token 的 KV cache）+ 状态观测做交叉注意力，输出动作块 $a_t$。
- **训练专用模块**: 文本教师 VLM $\mathcal{F}_\theta^T$（GRPO 训练，产生显式 CoT）+ verbalizer LLM $\mathcal{V}_\psi$（[[Qwen3]]-0.6B，每层插交叉注意力，把 latent 解回文本）。
- **总规模**: 主干 3B（另验证 7B 可扩展）。

整体流程：$\mathcal{F}_\theta$ 把 $(o_t,l)$ 经 latent CoT 推理产出视觉规划 latent $c_t$（Sec. 3.2），$c_t$ 再引导 $\pi_\phi$ 预测可执行动作 $a_t$（Sec. 3.3）。

### 核心模块

#### 模块1: Verbalizable Latent CoT by Reward Preferences（偏好引导的可言语化隐式 CoT）

**设计动机**: 把长文本 CoT 压成 $M$ 个连续 latent 没有直接监督信号；引入 [[Verbalizer|verbalizer]] 把 latent 解码成可读文本，从而在自然语言空间做监督，保证 latent 忠实保留推理结构。

**具体实现**:
- **教师先行**: 文本教师 $\mathcal{F}_\theta^T$ 先用 [[GRPO]] 训练（公式1），最大化裁剪式策略目标；组内 reward 归一化为 advantage $A(\tau)$（公式2）。
- **构造偏好对**: 每个 rollout group 内取 advantage 最高/最低的 trace 作 $\tau^+/\tau^-$（公式3）——advantage 天然充当推理质量指标。
- **学生隐式推理**: 学生 $\mathcal{F}_\theta$ 不生成文本 token，而是自回归产出 $M$ 个连续向量 $\mathbf{z}=\{z_m\}_{m=1}^M,\ z_m\in\mathbb{R}^d$。
- **DPO 式偏好目标**: 训练 verbalizer $\mathcal{V}_\psi$ 使其在 latent 条件下，给高质量 $\tau^+$ 比低质量 $\tau^-$ 更高似然（公式4，$\beta=0.1$）。这反过来逼学生把 latent 编码成"能被解码成高质量推理"的形式。
- **训练课程**: 前 3000 步用 $\tau^+$ 做语言建模 loss 预热 verbalizer 与 latent 对齐；之后冻结 $\mathcal{V}_\psi$，用 $\mathcal{L}_{\text{verb}}$ 训练剩余 1500 步，学生 $\mathcal{F}_\theta$ 全程更新。

#### 模块2: Action-Aligned Visual Plan Distillation（动作对齐的视觉规划蒸馏）

**设计动机**: $\mathcal{L}_{\text{verb}}$ 只保证学到"高层推理模式"，不保证 latent 编码了具身控制必需的**视觉规划**能力，需显式迁移教师的空间推理。

**具体实现**:
- **隐状态对齐**: 教师用 trajectory-level 奖励（目标完成 + 轨迹对齐，沿用 [[ThinkAct]]）训练接地视觉规划；对齐 `<answer>` token 的隐状态——最小化教师（对应 $\tau^+$）与学生隐状态的 L2 距离（公式5）。
- **并行 waypoint 预测**: 教师自回归吐出 waypoint 序列 $\{p_k\}_{k=1}^K,\ p_k\in[0,1]^2$（$K=5$ 时占 60–70 token）；学生改用 $K$ 个可学习 spatial token $\{\mathbf{s}_i\}$ 附在 latent 序列后，每个输出隐状态经 MLP **并行**投影成一个 waypoint。
- **统一目标**: 三项相加 $\mathcal{L}_{\text{student}}=\mathcal{L}_{\text{verb}}+\mathcal{L}_{\text{distill}}+\mathcal{L}_{\text{ans}}$（公式6），其中 $\mathcal{L}_{\text{ans}}$ 是 waypoint 回归 L2。

#### 模块3: Reasoning-Enhanced Policy Learning（推理增强的策略学习）

**设计动机**: 把高层视觉规划 $c_t$ 接到底层动作生成，且对动作模型选择**无关（agnostic）**。

**具体实现**:
- 从 spatial token 在 **VLM 较早层**的 KV cache 提取视觉规划 $c_t$（因 $\mathcal{F}_\theta$ 层数多于 $\pi_\phi$；消融证明早层 KV 优于晚层 KV 与直接用输出隐状态，见 B.3）。
- 把 $c_t$ 与动作模型 state encoder 的 KV pair 拼接，动作模型的交叉注意力同时关注视觉规划上下文与状态观测。
- 在 action-annotated 机器人数据上 post-train：**冻结 $\mathcal{F}_\theta$ 与 state encoder**，只用模仿学习目标更新 latent projector 与 $\pi_\phi$（公式7，$\ell$ 为扩散去噪目标）。
- **推理时只需 $\mathcal{F}_\theta$ + $\pi_\phi$**；verbalizer $\mathcal{V}_\psi$ 仅训练用、推理时可选地用于可解释性。

### 关键公式与机制

#### 公式1: [[GRPO]] 教师训练目标

$$
\mathcal{J}_{\text{GRPO}}(\theta)=\mathbb{E}_{\tau\sim\mathcal{F}^{T}_{\theta}}\Big[\min\big(r_{\theta}(\tau)A(\tau),\ \text{clip}(r_{\theta}(\tau),1-\epsilon,1+\epsilon)A(\tau)\big)\Big]
$$

**含义**: 文本教师用 GRPO 的裁剪式策略梯度学习显式推理，最大化高 advantage 推理 trace 的概率。

**符号说明**:
- $\tau$: 一条推理 trace；$r_{\theta}(\tau)=\dfrac{\mathcal{F}^{T}_{\theta}(\tau)}{\mathcal{F}^{T}_{\text{old}}(\tau)}$: 新旧策略概率比
- $A(\tau)$: advantage（见公式2）；$\epsilon$: 裁剪范围

#### 公式2: 组内归一化 advantage

$$
A(\tau)=\frac{R_{\tau}-\text{mean}(\{R_{i}\}_{i\in G(\tau)})}{\text{std}(\{R_{i}\}_{i\in G(\tau)})}
$$

**含义**: 把组内多条 rollout 的奖励做标准化得到 advantage，既驱动 GRPO，又**充当推理质量的天然指标**（高 advantage = 高质量推理）。

**符号说明**:
- $\{R_{i}\}_{i\in G(\tau)}$: 同一 rollout group $G(\tau)$ 内各 trace 的奖励；rollout size $N=5$

#### 公式3: 偏好对构造

$$
\tau^{+}=\arg\max_{\tau\in G}A(\tau)\quad\text{and}\quad\tau^{-}=\arg\min_{\tau\in G}A(\tau)
$$

**含义**: 从每个 rollout group 取 advantage 最高/最低的 trace，作为偏好学习的正/负样本。

**符号说明**:
- $\tau^+$: 高质量（正）推理；$\tau^-$: 低质量（负）推理

#### 公式4: [[DPO]] 式可言语化偏好损失

$$
\mathcal{L}_{\text{verb}}=-\mathbb{E}\Big[\log\sigma\Big(\beta\big(\log\tfrac{p_{\psi}(\tau^{+}\mid\mathbf{z})}{p_{\text{ref}}(\tau^{+})}-\log\tfrac{p_{\psi}(\tau^{-}\mid\mathbf{z})}{p_{\text{ref}}(\tau^{-})}\big)\Big)\Big]
$$

**含义**: 让 verbalizer 在 latent $\mathbf{z}$ 条件下解码出 $\tau^+$ 的似然显著高于 $\tau^-$，从而逼学生把 latent 编码成可还原为高质量推理的形式。

**符号说明**:
- $p_{\psi}(\cdot\mid\mathbf{z})$: verbalizer 在 latent 条件下的解码概率
- $p_{\text{ref}}$: 参考模型（即不接 latent 的 $\mathcal{V}_\psi$）；$\sigma$: sigmoid；$\beta=0.1$: 偏好强度

#### 公式5: 视觉规划隐状态蒸馏

$$
\mathcal{L}_{\text{distill}}=\|h_{t}^{T}-h_{t}\|_{2}^{2}
$$

**含义**: 对齐教师与学生 `<answer>` token 的隐状态，迁移 trajectory-level 的视觉规划能力。

**符号说明**:
- $h_{t}^{T}$: 教师隐状态（对应 $\tau^+$）；$h_{t}$: 学生隐状态

#### 公式6: 学生总目标（含并行 waypoint 回归）

$$
\mathcal{L}_{\text{student}}=\mathcal{L}_{\text{verb}}+\mathcal{L}_{\text{distill}}+\mathcal{L}_{\text{ans}},\quad \mathcal{L}_{\text{ans}}=\sum_{i=1}^{K}\|p_{i}-\hat{p}_{i}\|_{2}^{2},\ \ p_{i}=\text{MLP}(h^{\prime}(\mathbf{s}_{i}))
$$

**含义**: 偏好对齐 + 视觉规划蒸馏 + waypoint 回归三者联合训练学生 $\mathcal{F}_\theta$，使其既会紧凑隐式推理，又能高效产出视觉轨迹规划。

**符号说明**:
- $h^{\prime}(\mathbf{s}_{i})$: 第 $i$ 个 spatial token 的输出隐状态；$p_i,\hat p_i$: 预测/真值 waypoint
- $p_i\in\mathbb{R}^6$，格式 $[x_{\text{single}},y_{\text{single}},x_{\text{left}},y_{\text{left}},x_{\text{right}},y_{\text{right}}]$（单臂用前 2 维，双臂用后 4 维，未用维度 mask 掉）；$K=5$

#### 公式7: 推理增强策略学习的模仿损失

$$
\mathcal{L}_{\text{IL}}(\phi)=\ell\left(\pi_{\phi}(o_{t},l,c_{t}),\ \hat{a}_{t}\right)
$$

**含义**: 冻结 VLM 与 state encoder，只用扩散去噪目标 $\ell$ 训练动作模型 $\pi_\phi$，把视觉规划 $c_t$ 翻译成底层动作。

**符号说明**:
- $c_t$: 来自 spatial token KV cache 的视觉规划上下文；$\hat a_t$: 真值动作；$\ell$: 扩散策略去噪目标

---

## 关键图表

<!-- 图片均使用 arXiv HTML v2 在线链接，未本地化 -->

### Figure 1: Overview / 方法总览与效率动机

![Figure 1](https://arxiv.org/html/2601.09708v2/x1.png)

**说明**: 对比图。先前推理 VLA 生成约 250 token 的冗长推理链；本文学习少量（如 6 个）连续 latent token（蓝）+ 并行 spatial token（绿）作内部推理。右下角散点显示在 SimplerEnv-Google 上比 [[ThinkAct]]-7B **快 9.3×** 且性能更高——开篇即点明"既快又强"的卖点。

### Figure 2: Framework Overview / 师生蒸馏框架

![Figure 2](https://arxiv.org/html/2601.09708v2/x2.png)

**说明**: 核心框架图。(a) 文本教师 $\mathcal{F}_\theta^T$ 生成显式推理链，学生 $\mathcal{F}_\theta$ 在 reward 偏好引导下把它蒸馏成 latent $\mathbf{z}$；verbalizer $\mathcal{V}_\psi$ 把 latent 解回文本用于 $\mathcal{L}_{\text{verb}}$ 偏好学习，$\mathcal{L}_{\text{distill}}$ 迁移视觉规划。展示了从教师到学生 latent 再到动作执行的完整数据流，是理解全文的主图。

### Figure 3: Manipulation & Efficiency Evaluation / 操作成功率与推理效率

![Figure 3](https://arxiv.org/html/2601.09708v2/x3.png)

**说明**: (a)-(e) 在 [[LIBERO]] 四个子任务与 [[SimplerEnv]]-Google 上对比 7B 级推理 VLA，Fast-ThinkAct 全面最高；(f) 3B 与 7B 推理 VLA 的延迟对比，本文最多降低 **89.3%** 延迟（vs ThinkAct-7B）、88.0%（vs MolmoAct-7B），且对 ThinkAct-3B 快 7×。直接量化"效率-性能双赢"。

### Figure 4: Visual Trajectories on Long-Horizon Tasks / 长程任务视觉轨迹与执行

![Figure 4](https://arxiv.org/html/2601.09708v2/x4.png)

**说明**: 在 (a) SimplerEnv-Google、(b) LIBERO-Long、(c) RoboTwin2.0-Hard（长 278 步）上可视化预测的 2D 视觉轨迹与执行结果。黄色为单臂/左爪轨迹，红色为双臂右爪轨迹。证明 latent 推理产出的视觉规划能支撑长程多步操作。

### Figure 5: Failure Recovery / 失败恢复能力

![Figure 5](https://arxiv.org/html/2601.09708v2/x5.png)

**说明**: 在 [[RoboFAC]] 上的失败恢复。左为定性示例（仿真 + 真机）的纠错引导；右为定量结果（RoboFAC-Sim / RoboFAC-Real）。Fast-ThinkAct 比次优 RoboFAC-3B 在仿真高 10.9 分、真机高 16.4 分。例如目标物中途掉落时能生成"后撤→侧移对齐→下降抓取"的恢复计划。

### Figure 6: Few-Shot Adaptation / 小样本适应

![Figure 6](https://arxiv.org/html/2601.09708v2/x6.png)

**说明**: RoboTwin2.0 上每任务仅 10 条示范微调的小样本结果。Fast-ThinkAct 显著增强基础动作模型 [[RDT]]，并在 medium/long-horizon 任务上超过 $\pi_0$ 与 ThinkAct，且延迟远低于 ThinkAct——说明高效推理也利于小样本动作适应。

### Figure 7: Reasoning Trace Comparison (RoboVQA) / 推理链对比

![Figure 7](https://arxiv.org/html/2601.09708v2/x7.png)

**说明**: RoboVQA 上教师文本推理 (a) vs 学生 verbalize 后的 latent 推理 (b)。绿=相关内容，橙=不相关内容。两者都抓到任务相关信息，但教师冗长、夹带无关内容，学生更简洁聚焦——佐证偏好蒸馏既省算力又过滤冗余。

### Figure 8: Bimanual Visualization on RoboTwin2.0 / 双臂操作可视化（附录）

![Figure 8](https://arxiv.org/html/2601.09708v2/x8.png)

**说明**: hanging mug 与 handover mic 任务在 easy/hard 设置下的轨迹与执行（hard 含不同背景与干扰物）。黄=左爪轨迹、红=右爪轨迹。展示在多变视觉条件下的双臂协调与 waypoint 引导能力。

### Figure 9: Failure Identification & Analysis / 失败识别与归因（附录）

![Figure 9](https://arxiv.org/html/2601.09708v2/x9.png)

**说明**: 上排识别失败类型（如位置偏移）与执行阶段（如伸向方块）；下排做根因分析（如用 L 形工具推方块失败源于初始抓取不当）。说明模型对失败的理解超越单纯恢复规划。

### Figure 10: Reasoning Trace Comparison (OpenEQA) / 推理链对比（附录）

![Figure 10](https://arxiv.org/html/2601.09708v2/x10.png)

**说明**: OpenEQA 上教师 vs 学生推理可视化。学生输出紧凑且正确（绿），教师冗长输出有时含错误步骤（红）——表明压缩反而可能过滤掉教师的错误链。

### Figure (Appendix): Ablation on Latent Reasoning Steps M / latent 步数消融

![Latent steps ablation](https://arxiv.org/html/2601.09708v2/x11.png)

**说明**: latent 推理步数 $M$ 的消融。$M=1$ 推理容量不足；$M=30,100$ 引入冗余/噪声；$M=6$ 性能最优，故作为默认（对应 Table 8）。

### Table 1: RoboTwin2.0 Bimanual Manipulation / 双臂操作成功率

E=easy（无域随机化），H=hard（含域随机化）；成功率 %，Average 为 10 任务均值。

| Method | click alarm (E/H) | click bell | turn switch | adjust bottle | beat block | handover mic | handover block | hanging mug | stack blocks two | stack bowls three | **Avg. (E/H)** |
|--------|------|------|------|------|------|------|------|------|------|------|------|
| DP | 61/5 | 54/0 | 36/1 | 97/0 | 42/0 | 53/0 | 10/0 | 8/0 | 7/0 | 63/0 | 43.1 / 0.6 |
| ACT | 32/4 | 58/3 | 5/2 | 97/23 | 56/3 | 85/0 | 42/0 | 7/0 | 25/0 | 48/0 | 45.5 / 3.5 |
| $\pi_0$ | 63/11 | 44/3 | 27/23 | 90/56 | 43/21 | 98/13 | 45/8 | 11/3 | 42/1 | 66/24 | 52.9 / 16.3 |
| RDT | 61/12 | 80/9 | 35/15 | 81/75 | 77/37 | 90/31 | 45/14 | 23/16 | 21/2 | 51/17 | 56.4 / 22.8 |
| ThinkAct | 64/13 | 84/11 | 40/19 | 94/70 | 79/33 | 92/40 | 56/15 | 31/18 | 30/5 | 54/23 | 62.4 / 24.7 |
| **Fast-ThinkAct** | **70/17** | 82/12 | 37/21 | 92/72 | **82/33** | **99/42** | **65/15** | 30/22 | **45/5** | **55/25** | **65.7 / 26.4** |

**说明**: Fast-ThinkAct 在 easy/hard 平均均最高，比基础动作模型 [[RDT]] 高 9.3/3.6 个点，比推理 VLA [[ThinkAct]] 高 3.3/1.7 个点且推理快得多。颜色（原文）标注任务长度：short(80-100)/medium(110-220)/long(270-470) 步。

### Table 2: Embodied Reasoning Benchmarks / 具身推理基准

EgoPlan-Bench2 准确率（%），RoboVQA BLEU，OpenEQA LLM 打分。

| Method | EgoPlan Daily./Work./Rec./Hobbies/Avg. | RoboVQA B-1/2/3/4/Avg. | OpenEQA Score | **Overall Avg.** |
|--------|------|------|------|------|
| GPT-4V | 36.7/27.7/33.9/32.5/32.6 | 32.2/26.5/24.7/23.9/26.8 | 49.6 | 36.4 |
| Gemini-2.5-Flash | 44.2/42.3/43.2/39.1/42.4 | 39.1/31.6/22.9/22.1/28.9 | 45.3 | 38.9 |
| InternVL2.5-2B | 30.9/27.8/28.6/33.1/30.1 | 36.6/33.7/31.0/29.4/32.7 | 47.1 | 36.6 |
| InternVL3-2B | 36.9/29.9/35.6/31.5/33.4 | 34.4/33.9/33.5/33.3/33.8 | 48.8 | 38.7 |
| NVILA-2B | 34.6/26.7/33.3/31.6/31.4 | 38.7/34.3/31.1/29.2/33.3 | 47.0 | 37.2 |
| Qwen2.5-VL-3B | 29.0/27.0/30.2/28.9/28.5 | 42.5/36.3/28.7/31.8/34.8 | 43.4 | 35.6 |
| Magma-8B | 32.1/25.7/34.4/29.3/29.8 | 38.6/31.5/28.1/26.7/31.2 | 49.1 | 36.7 |
| RoboBrain2.0-3B | 45.3/37.6/45.9/39.7/41.8 | 54.4/47.7/43.1/41.0/46.5 | 50.1 | 46.1 |
| ThinkAct-3B | 46.6/41.4/45.9/42.5/44.0 | 62.4/57.3/52.0/49.6/55.3 | 48.9 | 49.4 |
| **Fast-ThinkAct-3B** | **50.3/44.3/46.4/43.2/46.4** | **70.1/63.0/57.2/53.0/60.8** | **51.2** | **52.8** |

**说明**: Fast-ThinkAct-3B 全面超越所有对比方法，包括两个闭源大模型（GPT-4V、Gemini-2.5-Flash）：EgoPlan-Bench2 +2.4、RoboVQA +5.5 BLEU、OpenEQA +1.1。证明紧凑 latent 推理不牺牲推理质量。

### Table 3: Ablation of Objectives & Stages (Reasoning) / 训练目标与阶段消融

| Method | EgoPlan | RoboVQA | OpenEQA | **Average** |
|--------|---------|---------|---------|---------|
| **Fast-ThinkAct** | **46.4** | **60.8** | **51.2** | **52.8** |
| w/o $\mathcal{L}_{\text{verb}}$ | 42.1 | 53.8 | 49.5 | 48.5 |
| w/o $\mathcal{L}_{\text{verb}},\mathcal{L}_{\text{distill}}$ | 41.6 | 52.7 | 48.9 | 47.7 |
| Textual Teacher $\mathcal{F}_{\theta}^{T}$ | 41.7 | 58.2 | 49.4 | 49.8 |
| SFT + CoT-SFT | 40.0 | 46.1 | 48.8 | 45.0 |
| SFT only | 40.5 | 53.6 | 45.3 | 46.5 |

**说明**: 去掉 $\mathcal{L}_{\text{verb}}$ 掉 4.3 点（缺偏好引导）、再去 $\mathcal{L}_{\text{distill}}$ 继续掉（缺视觉规划迁移）；完整模型甚至**超过文本教师**（52.8 vs 49.8），说明压缩+蒸馏不仅省算力还提质。CoT-SFT 在 EgoPlan/RoboVQA 反不如纯 SFT，提示朴素 CoT 监督会引入冗余。

### Table 4: Larger Backbone (7B/8B) / 大模型可扩展性（附录）

| Method | EgoPlan Avg. | RoboVQA B-Avg. | OpenEQA Score | **Overall** |
|--------|------|------|------|------|
| InternVL2.5-8B | 33.5 | 32.7 | 54.4 | 40.2 |
| InternVL3-8B | 36.2 | 35.3 | 55.5 | 42.3 |
| NVILA-8B | 33.7 | 39.0 | 54.0 | 42.2 |
| Qwen2.5-VL-7B | 29.1 | 39.7 | 50.8 | 39.9 |
| Magma-8B | 29.8 | 31.2 | 49.1 | 36.7 |
| RoboBrain2.0-7B | 33.2 | 37.8 | 51.1 | 40.7 |
| ThinkAct-7B | **48.2** | 59.8 | 56.2 | 54.7 |
| **Fast-ThinkAct-7B** | 47.5 | **61.1** | **59.0** | **55.9** |

**说明**: 换 [[Qwen2.5-VL]]-7B 后仍稳居 Overall 最高（55.9），验证 latent 推理蒸馏可扩展到更大主干（EgoPlan 略低于 ThinkAct-7B，但 RoboVQA/OpenEQA 与总体更优）。

### Table 5: LIBERO & SimplerEnv with Latency / 操作成功率与延迟（附录，对应 Fig.3）

| Method | LIBERO | SimplerEnv-Google | Latency (ms, ↓) |
|--------|------|------|------|
| OpenVLA-7B | 76.5 | 40.2 | N/A |
| CoT-VLA-7B | 83.9 | N/A | N/A |
| ThinkAct-7B | 84.4 | 68.3 | 7513 |
| MolmoAct-7B | 86.8 | 64.9 | 6723 |
| ThinkAct-3B | 83.1 | 64.7 | 5674 |
| **Fast-ThinkAct-3B** | **89.7** | **68.7** | **805 (↓7.0×)** |

**说明**: 同为 3B，Fast-ThinkAct 比 ThinkAct-3B 在 LIBERO +6.6、SimplerEnv +4.0，且延迟从 5674ms 降到 805ms（7× 加速）。延迟的核心来源正是推理 token 数（250→6）。

### Table 6: vs Efficient Textual Reasoning / 与高效文本推理基线对比（附录）

| Method | EgoPlan-Bench2 | RoboVQA | OpenEQA | **Average** |
|--------|------|------|------|------|
| Textual Teacher $\mathcal{F}_{\theta}^{T}$ | 41.7 | 58.2 | 49.4 | 49.8 |
| $\mathcal{F}_{\theta}^{T}$ inference w/o thinking (0 token) | 42.7 | 55.0 | 41.7 | 46.5 |
| $\mathcal{F}_{\theta}^{T}$ inference w/ 6 textual tokens | 39.3 | 53.0 | 46.5 | 46.3 |
| $\mathcal{F}_{\theta}^{T}$ w/ RL Length-Penalty (~50 token) | 41.2 | 57.5 | 44.7 | 47.8 |
| **Fast-ThinkAct-3B (6 latent token)** | **46.4** | **60.8** | **52.8** | **53.3** |

**说明**: 关键论证——直接删/限/罚文本推理（0、6 文本 token、长度惩罚 ~50 token）都从教师 49.8 掉到 46–48；而本文用 **6 个 latent token** 反升到 **53.3**。证明"压成连续 latent"远优于"压短文本"。

### Table 7: Ablation on Manipulation Benchmarks / 操作基准上的消融（附录，对应 Table 8 编号）

| Method | LIBERO | SimplerEnv-Google | RoboTwin2.0 | **Average** |
|--------|------|------|------|------|
| **Fast-ThinkAct** | **89.7** | **68.7** | **46.1** | **68.2** |
| w/o $\mathcal{L}_{\text{verb}}$ | 88.6 | 67.3 | 44.9 | 66.9 |
| w/o $\mathcal{L}_{\text{verb}},\mathcal{L}_{\text{distill}}$ | 86.3 | 65.7 | 42.6 | 64.9 |
| Textual Teacher | 88.5 | 67.3 | 45.8 | 67.2 |
| SFT + CoT-SFT | 87.2 | 65.8 | 43.3 | 65.4 |
| SFT only | 86.9 | 64.5 | 42.8 | 64.7 |

**说明**: 操作任务上的消融与推理基准一致：逐步去 loss 逐步掉点，完整模型超过文本教师与无师生训练版本，验证紧凑 latent 蒸馏对动作执行也有正收益。

### Table 8: Action Model Conditioning & Latent Steps (摘要)

| 消融维度 | 配置 | LIBERO | 结论 |
|----------|------|--------|------|
| KV 条件来源 | **早层 KV（默认）** | **89.7** | 早层 KV 最佳 |
|  | 晚层 KV（末 N 层） | 88.3 | 次之 |
|  | spatial token 输出隐状态 | 87.1 | 最差 |
| latent 步数 $M$ | $M=1$ / **$M=6$** / $M=30,100$ | 见 Fig.(x11) | $M=6$ 最优；过少不足、过多噪声 |

**说明**: 早层 KV cache 比晚层/输出隐状态更能携带视觉规划信息；latent 步数 $M=6$ 为容量与冗余的甜点。

---

## 实验

### 数据集 / 基准

| 类别 | 名称 | 规模/特点 | 用途 |
|------|------|----------|------|
| 推理训练 | MolmoAct 单臂 2D 轨迹（来自 OXE） | ~1.3M 轨迹 | reasoning SFT/蒸馏 |
| 推理训练 | AIST 双臂轨迹 | ~92K（Molmo-72B 检测夹爪 + CoTracker3 跟踪） | reasoning SFT/蒸馏 |
| QA 训练 | PixMo / RoboFAC / RoboVQA / ShareRobot / EgoPlan / Video-R1 | PixMo 726K、RoboFAC 64K QA、RoboVQA 798K QA、ShareRobot >1M、EgoPlan ~53K、Video-R1 165K | 推理能力 |
| 动作训练 | OXE（DiT-Policy）+ 静态 ALOHA（RDT） | 大规模 | 策略学习 |
| 评测-推理 | EgoPlan-Bench2 / RoboVQA / OpenEQA / RoboFAC | 1321/1893/1600+ 题；RoboVQA、RoboFAC 含真机视频 | 测试 |
| 评测-操作 | [[SimplerEnv]]-Google / [[LIBERO]] / [[RoboTwin2.0]] | 强 sim2real / 四子套件×500 trial×3 seed / 双臂 easy+hard×100 rollout | 测试 |

### 实现细节

- **VLM 主干**: [[Qwen2.5-VL]]-3B（另验证 7B）。
- **训练阶段**: ① SFT（~4M 样本，1 epoch，bs 64，lr 1e-5）→ ② CoT-SFT（15K iter，采 5% SFT + 165K Video-R1-CoT）→ ③ 师生训练（4500 iter，bs 128，lr 1e-6；前 3000 步预热 verbalizer，后 1500 步用 $\mathcal{L}_{\text{verb}}$）→ ④ 推理增强策略学习（20K iter，bs 256，lr 1e-4）。
- **教师**: GRPO（rollout $N=5$），action-aligned 视觉奖励（沿用 ThinkAct）+ QA 奖励。
- **verbalizer $\mathcal{V}_\psi$**: [[Qwen3]]-0.6B，每层插交叉注意力。
- **关键超参**: latent 步数 $M=6$，waypoint 数 $K=5$，$p_i\in\mathbb{R}^6$，$\beta=0.1$。
- **动作模型**: SimplerEnv 用 [[DiT-Policy]]（OXE 预训练，KV 维 1024），LIBERO/RoboTwin2.0 用 [[RDT]]（KV 维 2048）；线性投影适配 VLM KV cache 维度；冻结 VLM 与 state encoder 只训 $\pi_\phi$ + projector。
- **硬件**: 16× NVIDIA A100 80GB。

### 关键实验结论

- **操作**: LIBERO 89.7、SimplerEnv-Google 68.7（3B 级最高，Table 5）；RoboTwin2.0 双臂 easy/hard 平均 65.7/26.4（Table 1，超 RDT、ThinkAct）。
- **推理**: EgoPlan-Bench2/RoboVQA/OpenEQA 全面超越，含 GPT-4V、Gemini-2.5-Flash（Table 2）。
- **效率**: 推理延迟最多降 89.3%（vs ThinkAct-7B）；3B 模型 805ms vs ThinkAct-3B 5674ms（7×）。
- **长程**: RoboTwin2.0 长程任务（>270 步）easy/hard 48.8/16.8，超 RDT(35.0/12.3)、ThinkAct(42.8/15.3)。
- **失败恢复**: RoboFAC 仿真 +10.9、真机 +16.4（vs RoboFAC-3B，Fig 5）。
- **小样本**: 10 示范/任务下超 $\pi_0$ 与 ThinkAct（Fig 6）。
- **消融**: $\mathcal{L}_{\text{verb}}$、$\mathcal{L}_{\text{distill}}$ 都重要；早层 KV 条件最佳；$M=6$ 最优；压成 6 latent 优于压短文本（Table 6）。

---

## 批判性思考

### 优点
1. **抓住真痛点且方案巧妙**: "压短文本会掉点、压成连续 latent 反而提质"由 Table 6 直接证实，verbalizer + advantage 偏好对的设计解决了"latent 无监督"的根本难题，逻辑闭环漂亮。
2. **效率收益巨大且可信**: 250→6 token、7×~9.3× 加速并附带性能提升，延迟数据（805ms 等）具体，效率论证扎实。
3. **能力维度全面**: 不只刷成功率，还覆盖长程规划、失败识别/恢复、小样本适应，并对动作模型选择无关（RDT/DiT-Policy 均可），通用性强。
4. **可解释性保留**: verbalizer 能把 latent 解回文本供检视（Fig 7/10），缓解"黑箱 latent 推理"的担忧。

### 局限性
1. **verbalizer 继承 LLM 幻觉**: 作者自承 $\mathcal{V}_\psi$ 可能产出"看似合理实则错误"的描述；虽不影响动作执行，但作为"可解释性"卖点其忠实度存疑。
2. **训练流程极重**: 四阶段（SFT→CoT-SFT→师生→策略学习）、~4M SFT 样本、16× A100，复现门槛与算力成本很高；教师 GRPO 本身也昂贵。
3. **依赖强教师**: 整套蒸馏建立在 ThinkAct 式 GRPO 教师之上，最终质量受教师上限约束（Table 4 中 7B EgoPlan 反低于 ThinkAct-7B，提示并非处处占优）。
4. **超参偏经验**: $M=6$、$K=5$、$\beta=0.1$、早层 KV 等关键选择靠消融/经验确定，缺乏系统性敏感度或自适应方案；"早层 KV"的最优层位也未细究。
5. **可复现信息不全**: 代码/权重在本文文本中未明确开源声明（仅有 project page），数据处理虽详尽但 pipeline 复杂。

### 潜在改进方向
1. 引入 grounding-aware 目标或幻觉抑制，提升 verbalizer 还原推理的忠实度（作者亦提及）。
2. 探索免教师或弱教师的 latent 推理蒸馏，降低对昂贵 GRPO 教师与多阶段训练的依赖。
3. 把 latent 步数 $M$、KV 取层位置做成可学习/自适应，减少手工调参。
4. 在更多动作模型、更大/更小主干与真机长程任务上进一步验证泛化与稳健性。

### 可复现性评估
- [ ] 代码开源（正文未明确，仅项目主页）
- [ ] 预训练模型（未明确声明）
- [x] 训练细节完整（附录 A 给出阶段/超参/数据 pipeline）
- [x] 数据集可获取（OXE/LIBERO/RoboTwin2.0/SimplerEnv/EgoPlan/RoboVQA/OpenEQA/RoboFAC 等均公开）

---

## 速查卡片

> [!summary] Fast-ThinkAct: Efficient VLA Reasoning via Verbalizable Latent Planning
> - **核心**: 把推理型 VLA 的长文本 CoT（~250 token）蒸馏成少量（6 个）可言语化连续 latent，用偏好引导 + 视觉轨迹对齐迁移规划能力，再桥接到扩散动作模型。
> - **方法**: 教师 GRPO 产生带 advantage 的 trace → 取最高/最低构偏好对 → DPO 式 $\mathcal{L}_{\text{verb}}$（经 verbalizer 解码）+ `<answer>` 隐状态 $\mathcal{L}_{\text{distill}}$ + spatial token 并行 waypoint $\mathcal{L}_{\text{ans}}$ 训练学生 → 冻结 VLM、用早层 KV cache 的 $c_t$ 条件 RDT/DiT-Policy 学动作。
> - **结果**: 3B 主干，延迟最多降 89.3%（7× vs ThinkAct-3B）；LIBERO 89.7、SimplerEnv 68.7、RoboTwin2.0 65.7/26.4、EgoPlan/RoboVQA/OpenEQA 全面超 GPT-4V/Gemini/ThinkAct；6 latent 优于压短文本（53.3 vs 教师 49.8）。
> - **项目**: https://jasper0314-huang.github.io/fast-thinkact/

---

*笔记创建时间: 2026-06-29*
