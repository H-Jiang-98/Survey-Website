---
title: "Thinking in 360°: Humanoid Visual Search in the Wild"
method_name: "HVS (Humanoid Visual Search) / H*Bench / HVS-3B"
authors: [Heyang Yu, Yinan Han, Xiangyu Zhang, Baiqiao Yin, Bowen Chang, Xiangyu Han, Xinhao Liu, Jing Zhang, Marco Pavone, Chen Feng, Saining Xie, Yiming Li]
year: 2026
venue: CVPR
tags: [Humanoid, embodied-visual-search, MLLM, 360-panorama, GRPO, SFT, active-vision, benchmark]
zotero_collection: _inbox
image_source: online
arxiv_html: https://arxiv.org/html/2511.20351v2
created: 2026-06-29
---

# Thinking in 360°: Humanoid Visual Search in the Wild

## 元信息

| 项目 | 内容 |
|------|------|
| 作者 | Heyang Yu, Yinan Han, Xiangyu Zhang, Baiqiao Yin, Bowen Chang, Xiangyu Han, Xinhao Liu, Jing Zhang, Marco Pavone, Chen Feng, Saining Xie, Yiming Li（12 人）|
| 机构 | NYU（纽约大学）、NVIDIA、Stanford 等（含 Marco Pavone、Saining Xie）|
| 会议 | CVPR 2026 |
| 类别 | Humanoid / Embodied Visual Search / MLLM |
| 日期 | 2025-11（arXiv v2）|
| 项目主页 | （论文未给出独立主页）|
| 链接 | [arXiv](https://arxiv.org/abs/2511.20351) / [CVPR poster](https://cvpr.thecvf.com/virtual/2026/poster/38005) |

---

## 一句话总结

> 把"视觉搜索"从静态 2D 图像升级为人形智能体在 360° 全景中主动转头的具身闭环任务，建立 in-the-wild 基准 [[H*Bench]]，并用 SFT+GRPO 把开源 [[Qwen2.5-VL]] 的成功率翻三倍，同时量化出顶级 MLLM 仍只有约 30% 成功率的巨大鸿沟。

---

## 核心贡献

1. **提出人形视觉搜索任务 [[HVS]]**: 首次把视觉搜索定义为**主动（interactive）+ 具身（embodied）**的任务——智能体从窄 FoV 透视视角出发，通过**转头（head rotation）**在 360° 全景中搜索，分为 [[HOS]]（物体搜索，为操作做准备）与 [[HPS]]（路径搜索，为导航做准备）两类。
2. **可扩展的"全景即模拟器"框架**: 用一张真实世界 360° 全景图就闭合"感知-动作"回路，**绕开 3D 仿真器与真实硬件**，得到一个零硬件、可大规模扩展的具身推理研究平台。
3. **构建首个 in-the-wild 基准 [[H*Bench]]**: 走出家居场景，覆盖交通枢纽、大型零售、城市街道、公共机构等 12 国、6 大类、18 细分场景，约 3000 标注实例 / 12000 搜索 episode，配密集 ground-truth 动作标注。
4. **系统评测 + 后训练分析**: 揭示连 Gemini2.5-Pro 这类顶级模型也仅约 30% 成功；用 SFT+RL 把 Qwen2.5-VL 的 HOS 从 14.83%→47.38%、HPS 从 6.44%→24.94%；并指出 RL 在复杂任务上可能因 reward hacking 反而掉点。

---

## 问题背景

### 要解决的问题
如何让 MLLM 智能体像人一样在视觉拥挤的真实 3D 世界中**主动地、具身地**搜索目标物体或可行路径——而不是仅在一张静态图上"被动描述所见"。

### 现有方法的局限
当前 SOTA 视觉搜索（以 [[V*]] 为代表）基于 MLLM，但都在**单张静态低分辨率图像**上操作，动作被限定为对固定画布的**裁剪/缩放**等计算操作。相比生物视觉搜索存在两大根本鸿沟：
1. **非交互（Non-interactive）**: 无交互式模拟器，模型无法改变视角去获取初始 FoV 之外的信息；
2. **无具身（Disembodied）**: 缺乏物理身体，无法把视觉推理耦合到物理动作；且搜索通常**不由具身任务驱动**（操作/导航），退化为抽象的感知练习。

视觉导航类工作虽具身，但**依赖 3D 仿真器或真实硬件**，难以扩展复现，且多局限于家居场景；现有具身 AI 平台要么感知真实感不足，要么只有家居场景，**无法表征真正需要高级视觉-空间推理的密集拥挤环境**。

### 本文的动机
- 人类视觉系统只在中央凹（fovea）成清晰像，靠**头-眼协同**（头部探索未见区域、眼睛精查已见内容）在 360° 内高效搜索；论文把全身运动抽象为**转头**这一原子动作，聚焦人类空间智能中"停下观察、推理、消歧再行动"的**关键决策点**。
- 关键洞察：人在导航中的推理是**间歇性**的，只在关键点被调用 → 因此可直接从 in-the-wild 360° 全景**构建闭环搜索环境**，绕开 3D 仿真与硬件，得到可扩展框架。
- 选择**结构复杂（多层布局）、语义稠密（密集组合线索）、体积拥挤（杂乱 3D 空间）**的人造环境作为最有价值的测试床。

---

## 方法详解

### 任务与系统架构

[[HVS]] 把环境建模为**单张 360° 全景图**。所有可能观测 $\mathcal{S}_o=\{o_{\phi,\gamma}\}$ 是从该全景采样出的窄 FoV 透视图，由方位角 $\phi$ 与极角 $\gamma$ 定义。智能体被实现为一个 **tool-augmented MLLM**，把"转头"当作工具调用，与文本思维链交织构成**视觉链式思维（visual chain of thought）**：
- **输入**: 语言指令 $x$ + 当前透视观测 $o_t=o_{\phi_t,\gamma_t}$ + 历史 $\mathcal{H}_t$
- **Backbone**: [[Qwen2.5-VL]]-3B-Instruct（后训练得到 [[HVS-3B]]）
- **动作空间**: 两个原语——`Rotate`（转头）与 `Submit`（提交）
- **输出**: 每步生成文本思维链 $y_t$ + 动作 $a_t$，最终 `Submit` 给出方向估计 $(\hat\phi,\hat\gamma)$

两种任务变体：
- **[[HOS]]（Humanoid Object Search）**: 在未知 3D 环境主动搜索目标，找到把目标带入透视视图**中央凹区域**的最终视向 $(\phi^*,\gamma^*)$，作为后续**操作**的前提。
- **[[HPS]]（Humanoid Path Search）**: 在移动前作为高层规划，搜索到目标位置的**可行路径**，对齐身体朝向，只需找最终方位 $\phi^*$（地面近似为平面几何）。

### 核心模块

#### 模块1: 全景闭环搜索环境（Panorama-as-Simulator）

**设计动机**: 用真实全景图替代 3D 仿真器/硬件，既保证感知真实感，又能闭合感知-动作回路且可大规模扩展。

**具体实现**:
- 高分辨率全景视频（最高 $7680\times3840$）→ 透视渲染接口，按已知视角 $(\phi,\gamma)$ 渲染窄 FoV 图像；
- 每个任务实例用**4 个不同初始朝向**初始化，约 3000 实例放大为 12000 episode；
- 标注者自由旋转虚拟相机，框出目标 bbox，反投影到全景，其中心给出最优方向 $(\phi^*,\gamma^*)$。

#### 模块2: 智能体策略（Tool-Augmented MLLM Policy）

**设计动机**: 将主动转头建模为可学习的多模态多轮决策，使视觉推理与物理动作耦合。

**具体实现**:
- 策略 $\pi_\theta(y_t,a_t\mid o_t,x,\mathcal{H}_t)$，逐步生成"思维链 + 动作"；
- `Rotate` $a_t^{rot}=(\Delta\phi,\Delta\gamma)$ 更新视向（右/上为正，偏航 yaw 循环）；
- `Submit` $a_t^{sub}$ 提交当前视向为最终估计并结束 episode。

#### 模块3: 两阶段后训练（Post-Training Pipeline，见 Figure 2）

**设计动机**: 互联网静态、无具身数据训练出的 MLLM 缺乏空间常识与主动 3D 规划能力（GPT-4o 仅约 20%），需后训练注入。

**具体实现**:
- **Stage 1 — SFT**: 在精选的多轮 CoT 轨迹（2000 条，GPT-4o 生成 rationale + 人工 in-the-loop 校正去幻觉）上做**全参数 SFT**，建立"从多模态输入生成结构化动作计划"的行为先验（如"看不到目标就转身"）。得到 **HVS-3B (w/ SFT only)**。
- **Stage 2 — Multi-Turn RL（[[GRPO]]）**: 用 Group Relative Policy Optimization 精炼策略，鼓励长时程推理与可泛化搜索策略（如积累足够证据后才自信提交）。得到 **HVS-3B**。
- 关键经验：**直接 RL 而无 SFT 会破坏指令跟随能力**。

### 关键公式与机制

#### 公式1: [[HVS]] 搜索目标

$$
(\phi^{*},\gamma^{*})=\arg\max_{\phi,\gamma}P(r_{s}\mid o_{\phi,\gamma},x)
$$

**含义**: 在语言指令 $x$ 与视觉观测 $o_{\phi,\gamma}$ 条件下，找到使任务成功 $r_s$ 概率最大的最优视向。

**符号说明**:
- $(\phi,\gamma)$: 方位角 / 极角；$(\phi^*,\gamma^*)$: 最优视向
- $r_s$: 任务成功（二元）；$x$: 语言指令；$o_{\phi,\gamma}$: 该视向下的透视观测

#### 公式2: 智能体策略与状态更新

$$
\pi_{\theta}(y_{t},a_{t}\mid o_{t},x,\mathcal{H}_{t}),\quad \mathcal{H}_{t}=\{(o_{i},y_{i},a_{i})\}_{i=1}^{t-1}
$$

**含义**: 在当前观测、指令与历史条件下，同时产生思维链 $y_t$ 与动作 $a_t$。

**符号说明**:
- $o_t=o_{\phi_t,\gamma_t}$: $t$ 时刻透视观测
- $\mathcal{H}_t$: 截至 $t$ 的历史（观测、思维、动作三元组序列）
- 转头更新：$\phi_{t+1}=\phi_t+\Delta\phi$，$\gamma_{t+1}=\gamma_t+\Delta\gamma$

#### 公式3: 成功判定（容差区域）

$$
(\hat{\phi},\hat{\gamma})\in\left[\,\phi^{*}-\tau_{\phi},\,\phi^{*}+\tau_{\phi}\,\right]\times\left[\,\gamma^{*}-\tau_{\gamma},\,\gamma^{*}+\tau_{\gamma}\,\right]
$$

**含义**: 提交的最终视向落入以目标方向为中心、由 bbox 角宽决定的容差矩形内即判成功。

**符号说明**:
- $\tau_\phi=\max(\tfrac{w_\phi}{2},\tau_\phi)$、$\tau_\gamma=\max(\tfrac{w_\gamma}{2},\tau_\gamma)$，$w_\phi,w_\gamma$ 为 bbox 角宽/角高
- 设定：HOS 用 $\tau_\phi=30^\circ,\tau_\gamma=20^\circ$（模拟人类中央凹），HPS 用 $\tau_\phi=10^\circ$（精确朝向，仅评 $\hat\phi$）

#### 公式4: SFT 目标（负对数似然）

$$
\min_{\theta}\;\mathbb{E}_{(x,\mathcal{H}_{T})\sim\mathcal{D}^{SFT}}\left[-\sum_{i=0}^{T-1}\log\pi_{\theta}(y_{i},a_{i}\mid o_{i},x,\mathcal{H}_{i})\right]
$$

**含义**: 在 SFT 数据集上对整条多轮轨迹做交叉熵（NLL）模仿学习。

**符号说明**:
- $\mathcal{D}^{SFT}$: SFT 数据集（含输入 $x$ 与标注轨迹 $\mathcal{H}_T$）
- $T$: 轨迹步数；$\mathbb{E}$: 对数据分布取期望

#### 公式5: [[GRPO]] 目标函数

$$
\mathcal{J}_{\text{GRPO}}(\theta)=\mathbb{E}_{(s_{o},x,y)\sim\mathcal{D}^{RL},\,\{\omega_{i}\}\sim\pi_{\theta_{\text{old}}}}\Bigg[\frac{1}{G}\sum_{i=1}^{G}\frac{1}{|\omega_{i}|}\sum_{t=1}^{|\omega_{i}|}\mathcal{T}_{i,t}\Bigg]
$$

其中单 token 项

$$
\mathcal{T}_{i,t}=\min\!\Big[\rho_{i,t}\hat{A}_{i,t},\ \mathrm{clip}(\rho_{i,t},1-\epsilon,1+\epsilon)\hat{A}_{i,t}\Big]-\beta\,\mathbb{KL}(\pi_{\theta}\|\pi_{\text{ref}})
$$

**含义**: 对每个任务采样 $G$ 条输出 $\{\omega_1,\dots,\omega_G\}$（每条含全部输出 token $\{y_0,a_0,\dots,y_{T-1},a_{T-1}\}$），用组内相对优势做带 clip 与 KL 正则的策略更新。

**符号说明**:
- $\rho_{i,t}=\dfrac{\pi_{\theta}(\omega_{i,t}\mid s_o,x,\omega_{i,<t})}{\pi_{\theta_{\text{old}}}(\omega_{i,t}\mid s_o,x,\omega_{i,<t})}$: 重要性比
- $\epsilon$: clip 区间；$\beta$: KL 惩罚系数（实验中 $0.01$）；$\pi_{\text{ref}}$: 参考策略

#### 公式6: 组相对优势

$$
\hat{A}_{i,t}=\frac{r_{i}-\mathrm{mean}(r)}{\mathrm{std}(r)}
$$

**含义**: 用组内奖励的均值/标准差归一化得到优势，无需独立价值网络。

#### 公式7: 奖励组成与 HPS 距离奖励

$$
r=r_{corr}+r_{form}
$$

$$
r_{dist}=\frac{\pi-d(\phi_{T},\phi^{*})+\pi-d(\gamma_{T},\gamma^{*})}{2\pi},\qquad d(\alpha,\alpha^{*})=|\alpha-(\alpha^{*}-\tau_{\alpha})|+|\alpha-(\alpha^{*}+\tau_{\alpha})|
$$

**含义**: 规则化奖励含**正确性** $r_{corr}$ 与**格式** $r_{form}$；HPS 额外加**到目标距离奖励** $r_{dist}$（最终方向越靠近目标 bbox 越高，落入 bbox 内保持常数最小值）。

**符号说明**:
- $d(\alpha,\alpha^*)$: 最终角 $\alpha$ 到 bbox 两边界的距离和；$\tau_\alpha$: 容差
- $(\phi_T,\gamma_T)$: 终止时刻视向

---

## 关键图表

<!-- 图片均使用 arXiv HTML 在线链接，未本地化 -->

### Figure 1: Teaser / 从被动描述者到主动搜索者

![Figure 1 teaser](https://arxiv.org/html/2511.20351v2/figures/teaserv4.png)

**说明**: 提出核心问题——AI 能否像人一样在 3D 世界中**主动搜索**目标/路径，而非被动描述所见。图示把智能体从"被动 captioner"转为"主动 searcher"，从受限家居场景推进到 in-the-wild 复杂环境；关键使能点是**单张 360° 全景闭合转头的感知-动作回路**，让具身推理摆脱硬件约束。

### Figure 2: Pipeline / 两阶段后训练流程

![Figure 2 pipeline](https://arxiv.org/html/2511.20351v2/figures/main.png)

**说明**: Stage 1（SFT）赋予"把透视图映射到合理动作"的基础能力（如看不到东西就转身）；Stage 2（RL/GRPO）将其精炼为策略性行为——持续探索（输出 $a_t^{rot}$）直到获得证据充分的视图（如看到登机口标识），再自信提交最终估计 $a_t^{sub}$。直观展示了 SFT 建立先验、RL 精炼策略的分工。

### Figure 3: Dataset Statistics / H*Bench 数据统计

![Figure 3 stats](https://arxiv.org/html/2511.20351v2/x1.png)
![Top categories](https://arxiv.org/html/2511.20351v2/figures/Top_3.png)

**说明**: 左上=场景大类分布；右上=每大类的细分子类构成；左下=HOS 各类别物体实例分布；右下=HOS/HPS 任务难度分布与定义。佐证 [[H*Bench]] 的场景与目标多样性（12 国、6 大类、18 细分类型）。

### Figure 4: In-task vs Cross-task / 任务内与跨任务对比

![Figure 4 task performance](https://arxiv.org/html/2511.20351v2/figures/task_performance.png)

**说明**: 对比"同任务族训练-测试（In-task）"与"跨任务族迁移（Cross-task）"。In-task 总体最优，但有例外：仅在 HOS 上训练的模型在 easy HPS 上达 37.8%，超过 baseline(7.0%) 与专门 HPS 模型(33.8%)——因为 easy 路径搜索退化为有清晰视觉线索的物体搜索。跨任务存在**双向协同**：HOS 训练把 HPS 从 6.4%→20.7%，HPS 训练把 HOS 从 14.8%→29.5%。

### Figure 5: Error Analysis / HPS 失败模式与 Gemma3 分解

![Figure 5 error analysis](https://arxiv.org/html/2511.20351v2/figures/error_analysis.png)

**说明**: 左=HPS 三类失败：(a) 视觉-动作错配；(b) 试图穿越不可通行表面而非走相邻楼梯；(c) 缺社会-空间常识（如错失机场入口线索）导致徒劳搜索。右=Gemma3-4B-it 在 [[H*Bench]] 上的结果分解。揭示高级空间推理而非低级感知才是瓶颈。

### Figure 6: RL Step & Context Length / 累计成功率与上下文长度

![Figure 6 step and context](https://arxiv.org/html/2511.20351v2/x2.png)

**说明**: 左=RL 前后按步数的累计成功率（$t$ 为 RL 训练中最大转头上限）。短 rollout 训练经**测试时扩展**即可媲美长 rollout(10 轮)且收敛更快。右=测试时上下文长度对成功率的影响——**2 轮上下文**在 HVS 上已足够。

### Figure 7: Active vs Passive & Paradigm Comparison / 主动vs被动、范式对比

![Figure 7 comparison](https://arxiv.org/html/2511.20351v2/figures/final_all_bold_chart.png)

**说明**: 左=主动视觉搜索（透视视角转头采信息）vs 被动分析整张全景。被动方式会**掉点**（用 Gemma-3-4B-it 验证），因为它需要人工拼全景、且全景畸变冲突 MLLM 训练先验。右=不同范式对比：2D 方法 [[Mini-o3]](88.2%)、[[Chain-of-Focus]](88.0%) 在静态 [[V*Bench]] 近饱和，但在具身 [[H*Bench]] 上骤降到 2.5%/11.6%；本文 HVS-3B 在 H*Bench 仅 38.4% 而在 V*Bench 保持 65.5%。说明**被动互联网数据学到的能力无法迁移到具身主动交互**。

### Figure I: Geographical Distribution / 地理分布（附录）

![Figure I geo](https://arxiv.org/html/2511.20351v2/Appendix/geographical_distribution.jpg)

**说明**: 数据集横跨 12 国 13 城 4 大洲，建筑风格、标牌语言文字、环境条件多样，支撑泛化评估的广度。

### Figure III: HOS Task Visualization / HOS 难度可视化（附录）

![Figure III HOS](https://arxiv.org/html/2511.20351v2/Appendix/HOS_example.jpg)

**说明**: HOS 三难度示例，按目标在初始视图的**可见率 $d$**（可见面积/物体完整面积）划分：Easy（大体可见）、Medium（部分可见）、Hard（不可见）。

### Figure IV–VII: HPS Difficulty Visualization / HPS 四难度可视化（附录）

![Figure IV easy](https://arxiv.org/html/2511.20351v2/Appendix/HPS_easy.jpg)
![Figure V medium](https://arxiv.org/html/2511.20351v2/Appendix/HPS_medium.jpg)
![Figure VI hard](https://arxiv.org/html/2511.20351v2/Appendix/HPS_hard.jpg)
![Figure VII extreme](https://arxiv.org/html/2511.20351v2/Appendix/HPS_extreme.jpg)

**说明**: HPS 四难度（easy/medium/hard/extreme），依据两准则：相关线索是否与可行路径对齐、是否提供文本信息。难度越高，视觉/文本线索与真实路径方向越错位。

### Figure VIII–IX: HOS Error Cases / HOS 错误案例（附录）

![Figure VIII grounding](https://arxiv.org/html/2511.20351v2/Appendix/HOS_error1.png)
![Figure IX perception-action](https://arxiv.org/html/2511.20351v2/Appendix/HOS_error2.png)

**说明**: VIII=有限视觉 grounding（杂乱环境中无法可靠识别目标）；IX=感知-动作鸿沟（检测到目标但无法精细 foveation）。

### Figure X–XII: HPS Error Cases / HPS 错误案例（附录）

![Figure X vision-action](https://arxiv.org/html/2511.20351v2/Appendix/HPS_error1.png)
![Figure XI physical-commonsense](https://arxiv.org/html/2511.20351v2/Appendix/HPS_error2.png)
![Figure XII socio-spatial](https://arxiv.org/html/2511.20351v2/Appendix/HPS_error3.png)

**说明**: X=视觉-动作错配（看到标识却不能转为正确动作）；XI=缺物理常识（试图穿墙、误判垂直连接、直线穿越谬误、忽视落差）；XII=缺社会-空间常识（违反交通规范、闯禁区、无视功能布局等）。

### Figure XIII–XV: Pre vs Post Training Case Studies / 后训练前后案例（附录）

![Figure XIII foveation](https://arxiv.org/html/2511.20351v2/Appendix/case_study1.jpg)
![Figure XIV exploration](https://arxiv.org/html/2511.20351v2/Appendix/case_study2.jpg)
![Figure XV directional-sign](https://arxiv.org/html/2511.20351v2/Appendix/case_study3.png)

**说明**: 对应后训练带来的三种关键能力提升。XIII=目标 foveation（训练后做细粒度修正转头把目标居中再提交，而非过早提交）；XIV=探索行为（训练后用大角度探索快速发现新过道，而非低效小角度转头）；XV=依据方向标识选动作（训练后正确按标识旋转 $90^\circ$ 对齐目标方向）。

### Table 1: Main Results on H*Bench / 主结果（成功率 %）

**Humanoid Object Search (HOS)**

| Method | Overall | Easy | Medium | Hard |
|--------|---------|------|--------|------|
| InternVL3.5-4B | 3.08 | 7.32 | 2.84 | 1.49 |
| InternVL3.5-8B | 6.38 | 9.76 | 9.10 | 4.79 |
| Qwen2.5-VL-3B-Instruct | 14.83 | 27.97 | 13.07 | 10.01 |
| Qwen2.5-VL-7B-Instruct | 11.38 | 23.42 | 9.10 | 7.02 |
| Gemma-3-4B-it | 17.13 | 32.85 | 26.14 | 10.13 |
| Gemma-3-12B-it | 10.21 | 24.72 | 17.33 | 3.88 |
| Kimi-VL-A3B-Instruct | 4.92 | 12.85 | 0.57 | 2.36 |
| GPT-4o | 19.75 | 18.17 | 17.35 | 20.92 |
| Gemini2.5-Pro | 31.96 | 33.58 | 23.78 | 32.13 |
| HVS-3B (w/ SFT only) | 40.83 | 53.82 | 23.86 | 37.73 |
| **HVS-3B (Ours)** | **47.38** | **60.49** | **24.43** | **44.87** |

**Humanoid Path Search (HPS)**

| Method | Overall | Easy | Medium | Hard | Extreme |
|--------|---------|------|--------|------|---------|
| InternVL3.5-4B | 4.81 | 6.00 | 5.70 | 4.67 | 0.46 |
| InternVL3.5-8B | 7.25 | 10.00 | 7.68 | 5.14 | 4.17 |
| Qwen2.5-VL-3B-Instruct | 6.44 | 7.00 | 8.77 | 4.91 | 3.24 |
| Qwen2.5-VL-7B-Instruct | 6.31 | 9.00 | 5.92 | 5.84 | 1.85 |
| Gemma-3-4B-it | 14.44 | 17.20 | 14.47 | 14.72 | 7.41 |
| Gemma-3-12B-it | 14.50 | 16.80 | 14.25 | 14.49 | 9.72 |
| Kimi-VL-A3B-Instruct | 4.32 | 8.79 | 3.32 | 2.21 | 4.17 |
| GPT-4o | 23.69 | 26.80 | 22.59 | 26.17 | 13.89 |
| **Gemini2.5-Pro** | **33.00** | **41.60** | **29.39** | **35.75** | **15.28** |
| HVS-3B (w/ SFT only) | 23.00 | 28.00 | 23.03 | 21.26 | 14.81 |
| HVS-3B (Ours) | 24.94 | 34.80 | 20.18 | 25.00 | 12.04 |

**说明**: (1) 专有 vs 开源差距大，**Gemini2.5-Pro 整体最强**（HOS 31.96 / HPS 33.00）；(2) **更大模型未必更好**——Gemma-3/Qwen2.5-VL 的小模型在 HOS 上反超大模型。(3) 本文 **HVS-3B 在 HOS 上 47.38% 反超 Gemini2.5-Pro(31.96)**，但 HPS(24.94) 仍低于 Gemini2.5-Pro(33.00)，凸显路径搜索的高阶空间推理难度。(4) **SFT 贡献主要增益**（HOS +26.00、HPS +16.56），RL 增益更小（HOS +6.55、HPS +1.94），且在 HPS medium(23.03→20.18)/extreme(14.81→12.04) 上 **RL 反而掉点**（疑似 reward hacking）。

### Table 2: GRPO Reward Shaping on HPS / HPS 奖励塑形消融（成功率 %）

| Method | Overall | Easy | Medium | Hard | Extreme |
|--------|---------|------|--------|------|---------|
| **sft (baseline)** | **23.44** | 26.00 | **24.56** | **24.77** | **12.50** |
| form+corr | 22.38 | 33.80 | 17.32 | 21.73 | 7.87 |
| form+corr+dist | 21.37 | **34.40** | 15.13 | 20.09 | 6.94 |
| form+dist | 21.31 | 29.80 | 17.54 | 20.56 | 11.11 |

**说明**: 三种奖励（format+correctness、+distance-to-goal、format+distance）**都只在 easy 上提升**，却普遍**拖累更难层级**，整体甚至不如 SFT baseline(23.44)。说明 HPS 难度高、需要更先进的学习算法，单纯奖励塑形不足以稳定提升高阶空间推理。

---

## 实验

### 数据集 / 基准

| 基准 | 规模 | 特点 | 用途 |
|------|------|------|------|
| [[H*Bench]] | 约 3000 实例（×4 初始朝向=12000 episode）；评测集保留 1000 实例（600 HOS+400 HPS）=4000 episode | in-the-wild：12 国、6 大类、18 细分场景；全景最高 $7680\times3840$ | 训练/测试 |
| SFT 集 | 2000 条多轮 CoT 轨迹（HOS/HPS 各随机 250 实例）| GPT-4o 生成 rationale + 人工校正（6 标注者 250 小时）| 训练 |
| RL 集 | 其余全部实例 | GRPO rollout | 训练 |
| [[V*Bench]] | （对照）| 静态 2D 视觉搜索 | 迁移评测 |

### 实现细节

- **Backbone**: [[Qwen2.5-VL]]-3B-Instruct
- **SFT**: LLaMA-Factory，全参数，lr 1e-5，3 epochs
- **RL（[[GRPO]]）**: VAGEN 框架，70 steps；batch 32、actor lr $1\times10^{-7}$、KL 系数 $\beta=0.01$、温度 0.7、最多 8 条轨迹、动态转头上限（5 或 10）；SFT/RL 输入分辨率 $1280\times720$；**8×NVIDIA H100**
- **评测**: 最大推理 10 步（累计成功率在此前收敛），每步最多 5 张透视图、上下文取最近 5 轮，分辨率 $1920\times1080$，温度 0，未完成 episode 记为失败
- **容差**: HOS $\tau_\phi=30^\circ,\tau_\gamma=20^\circ$；HPS $\tau_\phi=10^\circ$

### 关键实验结论

- **顶级模型也很弱**: HOS/HPS 成功率仅约 30%（Gemini2.5-Pro 最强），GPT-4o 约 20%；后训练把 Qwen2.5-VL 翻三倍（HOS 14.83→47.38、HPS 6.44→24.94）。
- **SFT 立骨架、RL 做精炼**: SFT 贡献绝大部分增益；RL 增益小且在复杂 HPS 上可能反向（reward hacking）。
- **任务依赖性**: 简单的 HOS 后训练后超过专有 SOTA；复杂的 HPS 仍落后，说明高阶空间推理难提升。
- **跨任务双向协同 + 混合训练最优**，但混合训练增益分布不均，需平衡 trade-off。
- **主动 > 被动**；**具身基准远难于 2D 基准**：2D 方法在 V*Bench 近饱和(88%)，在 H*Bench 暴跌(2.5%/11.6%)，证明被动互联网能力不迁移到具身主动交互。
- **测试时扩展 + 短上下文足矣**: 短 rollout 训练 + 测试时扩展可媲美长 rollout；2 轮上下文已够。

---

## 批判性思考

### 优点
1. **任务定义有概念新意**: 把视觉搜索从静态 2D 推进到"主动转头 + 具身任务驱动"的 360° 闭环，且用全景图当轻量模拟器**绕开 3D 仿真/硬件**，可扩展性与真实感兼顾。
2. **基准扎实**: [[H*Bench]] 覆盖 in-the-wild 多国多场景、密集标注、HOS/HPS 双任务且分难度，250 人时人工校正 CoT，质量较高。
3. **分析诚实且有信息量**: 不只报增益，还揭示 RL 在复杂任务掉点、2D→具身能力不迁移、SFT 主导增益等"负面/反直觉"结论，对后续研究有指导价值。

### 局限性
1. **绝对性能仍低**: 最强模型 HPS 仅约 33%、HVS-3B 在统一对照下 H*Bench 仅 38.4%，距实用很远；任务被简化为**单次转头决策点**，未建模连续 locomotion 与多决策点串联。
2. **"全景即世界"的近似偏强**: 单张全景假设静态环境、HPS 用平面地面近似，忽略动态障碍、真实可通行性与多层 3D 几何的物理交互；成功判据是**角度容差**而非真实可达性。
3. **后训练规模有限、reward 设计未解决**: RL 仅 70 步、奖励塑形所有变体都拖累难样本，论文自身承认需"更鲁棒奖励 + 更高效视觉 tokenizer + 注入动作导向空间知识的预训练"，当前方法更像基线而非解法。

### 潜在改进方向
1. 设计与真实任务目标一致、跨难度稳定的奖励（缓解 reward hacking），或引入过程奖励/可验证中间信号。
2. 从单决策点扩展到**多决策点、连续具身轨迹**，并把"全景模拟器"与轻量 3D/可通行性先验结合，提升物理与社会-空间常识。
3. 探索注入"动作导向空间世界知识"的预训练，以及更高效的视觉 token 化以支持高分辨率全景下的长时程搜索；扩大具身搜索数据规模。

### 可复现性评估
- [ ] 代码开源（论文未在正文给出仓库链接，基于 LLaMA-Factory + VAGEN 复现）
- [ ] 预训练模型（HVS-3B 权重未明确声明 release）
- [x] 训练细节完整（附录 VI 给出 lr/batch/KL/分辨率/硬件等）
- [x] 数据集可获取（部分源自 360+x、YouTube 与自采；H*Bench 标注协议详述，但正式 release 状态待确认）

---

## 速查卡片

> [!summary] Thinking in 360°: Humanoid Visual Search in the Wild
> - **核心**: 把视觉搜索升级为人形智能体在 360° 全景中**主动转头**的具身闭环任务（HOS 物体搜索 / HPS 路径搜索），用全景图当轻量模拟器绕开 3D 仿真与硬件。
> - **方法**: tool-augmented MLLM 策略（Qwen2.5-VL-3B）+ 两阶段后训练——Stage1 SFT 在 2000 条 CoT 轨迹建立行为先验，Stage2 GRPO 精炼为策略性探索-提交。
> - **基准**: H*Bench，12 国/6 大类/18 细分 in-the-wild 场景，约 3000 实例×4 朝向。
> - **结果**: 顶级模型仅约 30%；HVS-3B 把 HOS 14.83→47.38、HPS 6.44→24.94（翻三倍）；SFT 主导增益，RL 在复杂 HPS 上可能掉点；2D 方法在 V*Bench 近饱和(88%)却在 H*Bench 暴跌(2.5%/11.6%)。
> - **代码**: 论文未在正文提供独立仓库链接（基于 LLaMA-Factory + VAGEN）。

---

*笔记创建时间: 2026-06-29*
