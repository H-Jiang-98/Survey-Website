---
title: "ForeAct: Steering Your VLA with Efficient Visual Foresight Planning"
method_name: "ForeAct"
authors: [Zhuoyang Zhang, Shang Yang, Qinghao Hu, Luke J. Huang, James Hou, Yufei Sun, Yao Lu, Song Han]
year: 2026
venue: CVPR
tags: [VLA, visual-foresight, world-model, image-generation, hierarchical-planning, flow-matching]
zotero_collection: _inbox
image_source: online
arxiv_html: https://arxiv.org/html/2602.12322v1
created: 2026-06-29
---

# ForeAct: Steering Your VLA with Efficient Visual Foresight Planning

## 元信息

| 项目 | 内容 |
|------|------|
| 作者 | Zhuoyang Zhang, Shang Yang, Qinghao Hu, Luke J. Huang, James Hou, Yufei Sun, Yao Lu, Song Han |
| 机构 | MIT、NVIDIA、Caltech（韩松团队 MIT HAN Lab） |
| 会议 | CVPR 2026 |
| 类别 | VLA（Vision-Language-Action） |
| 日期 | 2026-02（arXiv v1） |
| 项目主页 | https://github.com/mit-han-lab/foreact |
| 链接 | [arXiv](https://arxiv.org/abs/2602.12322) / [Code](https://github.com/mit-han-lab/foreact) |

---

## 一句话总结

> 用一个高效的「未来观测生成器」把高层语言指令翻译成逐步的**想象未来图像**，作为额外视觉输入喂给现成 VLA，让 VLA 专注视觉-运动推理而非语义推理，无需改架构即把 $\pi_0$ 真实任务成功率从 46.5% 拉到 87.4%。

---

## 核心贡献

1. **视觉前瞻规划框架 ForeAct**: 提出用[[世界模型|world model]]式的「想象未来观测」+子任务描述对现成 [[VLA]] 做**逐步视觉引导**。VLA 仅需把生成图像拼到视觉输入上即可接入，**零架构改动**，对 $\pi_0$/$\pi_{0.5}$ 等 SOTA 模型即插即用。
2. **高效前瞻图像生成器**: 基于 [[SANA]] 的线性 [[Diffusion Transformer|DiT]] + 32× 深压缩自编码器，在 H100 上 **0.33s** 生成 640×480 高清未来观测，满足闭环实时控制；在 **116 万子任务 / 约 1000 万图像对**的跨本体数据上预训练，学到鲁棒的具身动力学。
3. **大幅性能与泛化提升**: 11 个真实多步任务平均成功率 **87.4%**，比 $\pi_0$（46.5%）高 **+40.9%**，比「文本子任务引导」的 VLM+$\pi_0$（57.1%）高 **+30.3%**；在组合/空间 OOD 与小数据场景下显著优于基线。

---

## 问题背景

### 要解决的问题
[[VLA]] 模型把视觉观测与语言指令端到端映射到动作，但在开放世界的复杂、多步任务上仍很弱——核心难点是**把高层语言指令 grounding 到具体可执行的动作序列**。作者要回答的问题是：能否用**视觉指令**而非纯语言来引导机器人，即不仅"告诉它做什么"，更"给它看怎么做"。

### 现有方法的局限
- **统一规划+控制于一个模型**（如 $\pi_{0.5}$）: VLA 主干通常较小（约 3B）以降延迟，**推理能力受限**；在机器人数据上微调又会**灾难性遗忘**，损伤通用能力。
- **分层框架**（规划交给独立大模型）: 能缓解遗忘、用更强模型做规划，但**没有从根本上缓解 VLA 把语言落地为动作的难题**——VLA 拿到的仍是文本子任务。
- **基于视频生成的视觉预测**（SuSIE、视频生成系列）: (1) 推理慢、算力高；(2) 多为开环、忽视环境反馈；(3) 与 SOTA VLA 不兼容。

### 本文的动机
- 直接给最终目标图往往**不可得**，或太抽象、缺中间步骤；因此应提供**逐步的、可达的中间未来观测**。
- 一旦有了想象的下一步观测，VLA 就**不必再做高层语义推理**（"哪个是目标物""该做什么动作"），可把有限容量专注在**视觉-运动推理**上，从而提高精度与泛化。
- 关键设计取向：生成器**只条件于视觉+语言**（不依赖本体状态），以跨异构本体泛化；只对**头部相机**生成前瞻图（全局场景信息量最大）；用线性 DiT + 深压缩保证**够快**能进闭环。

---

## 方法详解

### 模型架构

ForeAct 是一个挂在现成 VLA 之上的**分层视觉前瞻规划器**（见 Figure 2），由两个组件构成 $\pi_h=\{\pi_g,\pi_v\}$：
- **输入**: 头部相机当前观测 $\bm{I}_t$ + 高层任务描述 $l$（+ 机器人三路相机视图、本体状态供 VLA 使用）
- **$\pi_v$（VLM 规划器）**: [[Qwen3-VL]]-8B-Instruct，在 reason-execute-monitor 循环里产出可执行**子任务描述** $l_t$
- **$\pi_g$（前瞻图像生成器 ImGen）**: 基于 [[SANA]] 的线性 [[Diffusion Transformer|DiT]]，把当前观测 + $l_t$ 生成**未来观测** $\bm{G}_t$
- **$\pi_l$（VLA 模型）**: 现成 $\pi_0$/$\pi_{0.5}$，把 $\bm{G}_t$ 作为**额外视觉输入**、$l_t$ 作为语言条件，输出[[Action Chunking|动作块]] $\bm{A}_t$
- **闭环**: VLM 监控执行进度，子任务完成则重规划，实现闭环控制

### 核心模块

#### 模块1: Efficient Foresight Image Generation（高效前瞻图像生成器 ImGen）

**设计动机**: 用一个 [[世界模型]] 把抽象语言指令"具象化"成一张接近真实未来观测的图像，作为对 VLA 信息量远高于文本的引导；同时必须**高分辨率 + 够快**才能进闭环。

**具体实现**:
- 采用 [[SANA]] 架构：**32× 深压缩自编码器**（deep compression autoencoder）把图像编码到极少的隐 token，**线性 DiT**实现高分辨率下的高效注意力。
- SANA 原本是文生图、无法吃图像条件 → 作者把**条件图像与噪声输入拼接（concatenate）**，将去噪过程改造成**条件去噪**任务（见 Figure 3）。
- 另加一段**专门设计的 system prompt**，把模型注意力引导到"机器人的动作"上。
- 只对**头部相机**生成 640×480 前瞻图；只条件于视觉+语言，**排除本体状态**等模态以保跨本体泛化。
- 训练用 [[Flow Matching|流匹配]]目标，用 SANA-1.6B-512px 权重初始化；全局 batch 512、800K 步、64×H100、恒定 lr 5e-5 + 5K warmup。

#### 模块2: VLM for Reasoning and Monitoring（VLM 推理与监控）

**设计动机**: 复杂长程任务需要把整体目标拆成可执行子任务，并在执行后判断进度、动态重规划，以从失败中恢复。

**具体实现**:
- VLM（Qwen3-VL-8B）运行 **reason → execute → monitor** 循环：给定整体任务与当前观测，先**reason**出立即可执行的子任务 $l_t$；VLA 执行后，VLM **monitor**评估新状态、追踪进度，子任务完成则重规划下一步。
- 该层级让系统能**动态从执行失败中恢复**、适应环境不确定性（具体 prompt 模板见 Table 5）。

#### 模块3: Adapt to VLA Models（无改架构接入）

**设计动机**: 让任何 SOTA VLA 都能零成本接入视觉引导。

**具体实现**:
- **不改 VLA 架构**，只增广视觉输入。微调时把**当前观测**与**未来观测**拼接成 VLA 的视觉输入（实践中用每个子任务片段的**最后一帧**作为未来观测）。
- 推理时把生成的前瞻图 $\bm{G}_t$ 追加到当前观测，生成的子任务指令 $l_t$ 作为语言输入喂给微调后的 VLA。

#### 模块4: Cloud-Edge Closed-Loop Deployment（云-边闭环部署）

**设计动机**: 兼顾"高层重计算"与"底层低延迟反应"。

**具体实现**:
- **边端 host**做反应式本地控制，采集观测发到**云端 server**；
- 云端（H100，用 [[vLLM]] 托管 VLM 与图像生成器）做高层规划，蒸成**双引导包**（文本 $l_t$ + 视觉 $\bm{G}_t$）回传；
- 边端本地 VLA 策略（RTX 5090）快速推理派发最终动作。

### 关键公式与机制

#### 公式1: [[VLA]] 的条件分布

$$
\pi(\bm{A}_{t}\mid\bm{I}_{t},\bm{q}_{t},l)
$$

**含义**: 现代 SOTA VLA 学习以多相机观测、本体状态、语言指令为条件的动作块分布。

**符号说明**:
- $\bm{A}_{t}=[a_{t},a_{t+1},\cdots,a_{t+H-1}]$: 长度为 $H$ 的动作块
- $\bm{I}_{t}=[I_{t}^{1},\cdots,I_{t}^{K}]$: $K$ 路相机的当前视觉观测
- $\bm{q}_{t}$: 机器人本体（proprioceptive）状态；$l$: 语言指令

#### 公式2: ForeAct 对 VLA 的分层重构

$$
\pi(\bm{A}_{t}\mid\bm{I}_{t},\bm{q}_{t},l)=\pi_{l}\!\left(\bm{A}_{t}\mid[\bm{I}_{t},\bm{G}_{t}],\bm{q}_{t},l_{t}\right)\,\pi_{h}\!\left(\bm{G}_{t},l_{t}\mid\bm{I}_{t},l\right)
$$

**含义**: 把原本一步到位的动作生成**分解为高层前瞻规划 $\pi_h$ 与低层 VLA 执行 $\pi_l$**。$\pi_h$ 给出预测未来观测 $\bm{G}_t$ 与子任务 $l_t$；$\pi_l$ 把 $\bm{G}_t$ 当额外视觉输入、$l_t$ 当语言条件生成动作。

**符号说明**:
- $\pi_h$: 视觉前瞻规划器；$\pi_l$: 现成 VLA 模型
- $\bm{G}_{t}$: 想象的未来观测；$l_t$: 子任务描述
- $[\bm{I}_{t},\bm{G}_{t}]$: 当前观测与前瞻图拼接后的增广视觉输入

#### 公式3: 前瞻规划器的进一步分解

$$
\pi_{h}\!\left(\bm{G}_{t},l_{t}\mid\bm{I}_{t},l\right)=\pi_{g}\!\left(\bm{G}_{t}\mid\bm{I}_{t},l_{t}\right)\,\pi_{v}\!\left(l_{t}\mid\bm{I}_{t},l\right)
$$

**含义**: 前瞻规划器由两部分组成——**VLM $\pi_v$** 先从整体任务 $l$ 与当前观测推出子任务 $l_t$；**图像生成器 $\pi_g$** 再把高层语言 grounding 成具体的预测未来观测 $\bm{G}_t$。

**符号说明**:
- $\pi_{g}$: 前瞻图像生成模型（ImGen），核心组件
- $\pi_{v}$: 用于复杂任务推理、推断子任务的 VLM

---

## 关键图表

<!-- 图片均使用 arXiv HTML 在线链接，未本地化。注意：arXiv HTML 中图序从 Figure 2 起，无独立 Figure 1 teaser。 -->

### Figure 2: Overview of ForeAct / 框架总览

![Figure 2](https://arxiv.org/html/2602.12322v1/x1.png)

**说明**: ForeAct 整体管线。VLM 子任务规划器读头部相机观测，产出子任务指令给 ImGen；ImGen 预测未来观测，连同子任务指令与机器人三路相机视图一起喂给 VLA。三者联动构成**闭环控制**——这是理解全文"分层 + 视觉引导"的总纲。

### Figure 3: Foresight Image Generation Model / 前瞻图像生成模型结构

![Figure 3](https://arxiv.org/html/2602.12322v1/x2.png)

**说明**: ImGen 内部结构。当前观测先编码成紧致视觉 token，与噪声输入**拼接**后送入线性 DiT 生成预测视觉 token，再解码为未来观测；指令 + 专设 system prompt 引导模型关注机器人动作。是"用文生图模型做条件去噪"的关键改造图。

### Figure 4: Pre-training Dataset Statistics / 预训练数据统计与本体多样性

![Figure 4](https://arxiv.org/html/2602.12322v1/x3.png)

**说明**: (a) 各数据源的子任务数量，预处理后共 **116 万子任务**；(b) 预训练数据覆盖的多样机器人本体。支撑"大规模跨本体预训练学到鲁棒具身动力学"的论点。

### Figure 5: Real-World Task Examples / 真实任务示例

![Figure 5](https://arxiv.org/html/2602.12322v1/x4.png)

**说明**: 自采真实数据集中 Kitchen / Workspace / Factory 三类、11 个任务的示例，难度从简单条件放置到双臂分拣不等。

### Figure 6: Qualitative Foresight Generation (w/o vs w/ pretrain) / 预训练消融定性对比

![Figure 6](https://arxiv.org/html/2602.12322v1/x5.png)

**说明**: OOD 任务"捡玉米"，每模型用 4 个随机种子生成。第一行（无预训练）无法生成正确预测图；第二行（有预训练）四张都正确，直观佐证大规模预训练的必要性。

### Figure 7: Comparison with Image-Editing Models / 与图像编辑模型对比

![Figure 7](https://arxiv.org/html/2602.12322v1/x6.png)

**说明**: Gemini 2.5 Flash Image / GPT-Image / Qwen-Edit 等 SOTA 图像编辑模型**缺乏对真实物理与物体动力学的理解**——要么几乎不改场景，要么幻觉出盘子里的胡萝卜，无法生成"带正确机器人动作"的预测图，且参数大、延迟高不适合闭环。说明本任务不是通用图像编辑能解决的。

### Figure 8: Latency Measurement / 延迟与去噪步数

![Figure 8](https://arxiv.org/html/2602.12322v1/x7.png)

**说明**: 去噪步数与画质/延迟的权衡。1–2 步严重伪影，**8 步**视觉可靠且再多收益微乎其微；故实验取 8 步，640×480 生成仅 **0.33s**，论证闭环实时性。

### Figure 9: Real-World Benchmark Evaluation / 真实基准逐任务成功率

![Figure 9](https://arxiv.org/html/2602.12322v1/x8.png)

**说明**: 11 个真实任务上 ForeAct 全面超越无视觉引导基线，平均 **87.4%**（较 VLM+$\pi_0$ +30.3%、较 vanilla $\pi_0$ +40.9%），且所有任务均 >70%。在 Pack_Flower、Toy_Blocks 等"需区分物体类型并采取不同动作"的任务上优势尤其大——视觉引导直接给出目标状态，VLA 无需推理目标与意图。

### Figure 10: OOD Generalization (Pick_Veg) / Pick_Veg 任务 OOD 泛化

![Figure 10](https://arxiv.org/html/2602.12322v1/x9.png)

**说明**: x 轴为递增难度：Original（同分布）→ +Spatial OOD（未见空间排布）→ +Comp. OOD（训练只单物体、测试要操作两物体的组合泛化）→ +Joint OOD（空间+组合）。ForeAct 在 OOD 下保持鲁棒，基线大幅掉点。

### Figure 11: Data Efficiency (Clean_Rubb) / 数据效率

![Figure 11](https://arxiv.org/html/2602.12322v1/x10.png)

**说明**: Clean_Rubb 任务上 20%–100% 数据子集的成功率对比。本文方法（红）样本效率强：**60% 数据即 >90%**、**20% 数据仍 79%**，全程优于 VLM 增广基线（灰）。

### Figure 12: Qualitative Compositional OOD / 组合 OOD 定性对比

![Figure 12](https://arxiv.org/html/2602.12322v1/x11.png)

**说明**: 指令"把胡萝卜、白菜、茄子放进盘子"（训练未见的序列）。(a) 基线 VLM+$\pi_0$ 迷失，反复去抓已被拿走的第一个物体位置；(b) 本文方法靠每步视觉引导稳健完成三个子任务。

### Figure 13: Subtask Planning Accuracy across VLMs / 不同 VLM 的子任务规划准确率（附录）

![Figure 13](https://arxiv.org/html/2602.12322v1/figs/rebut_vlm_eval.png)

**说明**: 用 Gemini-3-pro-preview 作 LLM-as-a-Judge 评测不同 VLM 作高层规划器的子任务语义正确率。Qwen2.5-VL-32B 与 Qwen3-VL-8B 都有竞争力，而更小的 Qwen2.5-VL-7B 明显掉点，验证了框架的模型可扩展性与基准的判别力。

### Figure 14: Additional Qualitative Foresight Results (OOD) / 更多前瞻生成定性结果（附录）

![Figure 14](https://arxiv.org/html/2602.12322v1/x12.png)

**说明**: 六行全为 OOD 场景（物体从未共现、引入数据集中不存在的水果/玻璃碗、盘上已有物体、全新刀叉纸杯等），生成器仍能给出合理未来观测，进一步佐证大规模预训练带来的强泛化。

### Table 1: Foresight Generation — Pretrain Ablation / 前瞻生成预训练消融

| Type | In-dist. Fidelity | In-dist. Quality | OOD Fidelity | OOD Quality |
|------|------|------|------|------|
| w/o pretraining | 0.18 | 0.24 | 0.00 | 0.00 |
| **w/ pretraining** | **1.00** | **1.00** | **0.88** | **0.96** |

**说明**: 每项在 50 张图上人工 0/1 打分（保真度=是否正确遵循子任务并走在完成路径上；质量=是否保留细节无明显失真）。无预训练在 OOD 上**完全失败（0.00）**、同分布也极低；有预训练两种设定都接近满分，量化坐实预训练的决定性作用。

### Table 2: Real-World Evaluation with $\pi_{0.5}$ / 更强主干上的真实评测

| Task | $\pi_{0.5}$ | Ours | Task | $\pi_{0.5}$ | Ours |
|------|------|------|------|------|------|
| Pick_Veg | 60.0 | **86.6** | Office_Desk | 76.0 | **85.4** |
| Place_Bowl | 75.0 | **83.3** | Pick_Tool | 50.0 | **96.7** |
| Pen_Drawer | 68.8 | **81.3** | Pack_Flower | 91.8 | **95.8** |

**说明**: 换到更强的 $\pi_{0.5}$ 主干，ForeAct 仍逐任务一致提升，平均成功率 **70.3% → 88.2%**，证明框架对不同 VLA 的通用性与可扩展性。Pick_Tool 提升最猛（50.0→96.7），呼应其"需在多工具中精确 grounding"的数据稀疏难点。

### Table 3: Simulation Evaluation on LIBERO / LIBERO 仿真评测

| Method | Spatial | Object | Goal | Long | Average |
|--------|---------|--------|------|------|---------|
| OpenVLA | 84.7 | 88.4 | 79.2 | 53.7 | 76.5 |
| CoT-VLA | 87.5 | 91.6 | 87.6 | 69.0 | 83.9 |
| $\pi_{0}$ | 96.8 | 98.8 | 95.8 | 85.2 | 94.2 |
| $\pi_{0.5}$ | 97.3 | 98.8 | 96.9 | 94.2 | 96.8 |
| CogVLA | **98.6** | 98.8 | 96.6 | 95.4 | 97.4 |
| **Ours (w/ $\pi_{0.5}$)** | 97.3 | **99.8** | **97.3** | **95.4** | **97.5** |

**说明**: 四套件各 500 试取平均。在已接近饱和的 $\pi_{0.5}$ 基线（96.8%）上，ForeAct 仍把平均推到 **97.5%**，并在 Object/Goal/Long 上取得最优，说明视觉引导即便在仿真饱和区也有正收益。

### Table 4: Ablation on Instruction Modalities (Pick_Tool) / 指令模态消融

| Method | Instruction Type | Success Rate (%) ↑ |
|--------|------------------|--------------------|
| Single $\pi_{0}$ | Semantic Text | 20.0 |
| Single $\pi_{0}$ | Spatial Text | 46.8 |
| **Ours** | **Semantic Text + Goal Image** | **93.4** |

**说明**: 把语义指令（"拿锤子"）换成空间指令（"拿右下角的物体"）能把基线从 20.0 提到 46.8——说明基线瓶颈在**物体语义 grounding**；但即便如此仍远低于本文的"语义文本 + 目标图像"（93.4），证明**细粒度图像引导**优于粗糙的空间文本，从根本上提升数据效率。

### Table 5–7: Prompt 模板与任务清单（附录）

| 表 | 内容 | 关键信息 |
|----|------|---------|
| Table 5 | VLM 规划器 prompt 模板 | 分初始规划阶段 ($t=0$) 与闭环监控阶段 ($t>0$)：先判断上一步是否完成，完成则给下一步指令、否则重复当前子任务，输出简洁确定 |
| Table 6 | 真实任务详述 Part I | Pick_Veg / Place_Bowl / 三碗堆叠 / Clean_Rubb / Toy_Blocks / Pen_Drawer / Office_Desk 的协议与数据分布（多为单物体训练、评测引入分布偏移） |
| Table 7 | 真实任务详述 Part II | Pick_Tool（5 工具）/ 口罩分拣（N95 vs 外科）/ 工具入箱 / Pack_Flower，含未见物体的零样本评测 |

---

## 实验

### 数据集 / 平台

| 数据/平台 | 规模 | 特点 | 用途 |
|------|------|------|------|
| 预训练集（[[AgiBot-World]] / [[RoboMind]] / [[Galaxea Open-World]] / [[BridgeData]]） | 116 万子任务 ≈ 1000 万图像对 | 跨本体、多任务、高分辨率（剔除低分辨率的 OXE/DROID） | ImGen 预训练 |
| 自采真实数据集 | 420 episode → 2,312 子任务级 episode；11 任务 | Galaxea R1 Lite 双臂移动机械臂（23-DoF，头部立体相机 + 双腕深度相机） | 微调/测试（采样 50 episode 作同分布测试集） |
| [[LIBERO]] | 4 套件各 500 试 | Spatial/Object/Goal/Long 仿真操作 | 测试 |

### 实现细节

- **ImGen**: [[SANA]] 线性 DiT + 32× 深压缩 AE，[[Flow Matching|流匹配]]目标；640×480；SANA-1.6B-512px 初始化；batch 512、800K 步、64×H100、lr 5e-5 + 5K warmup。部署微调：batch 32、5 epoch、lr 1e-5 无 warmup；推理 **8 去噪步、0.33s**。
- **VLM 规划器**: [[Qwen3-VL]]-8B-Instruct，reason-execute-monitor 闭环。
- **VLA**: $\pi_0$（3.3B，预训练 >10k 小时机器人数据）与 $\pi_{0.5}$，在子任务级 episode 上微调、视觉输入增广未来观测（用子任务末帧作 goal）。
- **部署**: 云-边分层，VLM+ImGen 在 H100（[[vLLM]]），VLA 在 RTX 5090。
- **评测协议**: 把任务拆成"approach and grasp / move and place"原子动作，成功率 = 成功原子动作比例；误动作扣分；单原子动作 5 次试不成判失败；每任务 ≥5 种初始设置取平均。

### 关键实验结论

- **真实基准（Fig 9）**: 11 任务平均 87.4%，较 VLM+$\pi_0$ +30.3%、较 $\pi_0$ +40.9%；需区分物体类型的任务（Pack_Flower、Toy_Blocks）优势最大。
- **更强主干（Table 2）**: $\pi_{0.5}$ 平均 70.3% → 88.2%。
- **仿真（Table 3）**: 在近饱和的 $\pi_{0.5}$ 上把 LIBERO 平均推到 97.5%（SOTA）。
- **预训练消融（Table 1 / Fig 6 / Fig 14）**: 无预训练 OOD 全 0，有预训练接近满分。
- **OOD 泛化（Fig 10 / Fig 12）**: 空间/组合/联合 OOD 下稳健，基线大幅掉点甚至卡死循环。
- **数据效率（Fig 11 / Table 4）**: 60% 数据 >90%、20% 数据仍 79%；"语义文本+目标图"远超纯空间文本基线（93.4 vs 46.8）。
- **VLM 可扩展性（Fig 13）**: Qwen2.5-VL-32B 与 Qwen3-VL-8B 均胜任规划，7B 明显掉点。

---

## 批判性思考

### 优点
1. **即插即用、零架构改动**: 仅靠"增广视觉输入"就能接入任何 SOTA VLA，工程迁移成本极低，对 $\pi_0$/$\pi_{0.5}$ 都有大幅提升，实用价值高。
2. **把"语义 grounding"难题转化为"视觉模仿"**: 用想象未来图像替代抽象文本，论证逻辑自洽且有 Table 4（语义文本 vs 空间文本 vs 目标图）的对照实验直接支撑因果。
3. **效率到位**: 0.33s/帧 + 闭环监控，真正进得了实时控制循环，区别于慢且开环的视频生成路线；预训练消融（OOD 0.00 vs 0.88+）干净有力。

### 局限性
1. **依赖云端重算力**: VLM(8B)+扩散生成器需 H100 在云端、VLA 在 RTX 5090，**对网络与算力的依赖**使纯边端落地受限；端到端系统延迟（含 VLM 推理与通信）未充分给出。
2. **生成器可能"撒谎"**: 前瞻图由世界模型想象而来，若在真正未见的动力学上幻觉错误目标状态，会把 VLA 带偏；论文虽展示 OOD 成功例，但**缺乏对生成失败如何级联影响控制**的系统分析与失败率统计。
3. **评测规模与指标偏定性**: 图像保真/质量靠人工 0/1 打分（50 张），真实成功率口径含较多人工裁定；OOD 主要围绕 Pick_Veg/Clean_Rubb 等少数任务展开，长程接触丰富/可变形物体仍少。

### 潜在改进方向
1. 引入**蒸馏/量化**进一步压缩生成延迟（作者已点名为 future work），并量化端到端闭环时延。
2. 给前瞻图加**置信度/一致性校验**（如多种子一致性、与真实下一帧的在线对齐）以抑制幻觉级联。
3. 把"目标图引导"扩展到**多相机/腕部视角**与更长程、接触丰富的任务，验证单头部相机假设的边界。

### 可复现性评估
- [x] 代码开源（https://github.com/mit-han-lab/foreact，含 demo 视频）
- [ ] 预训练模型（未明确声明发布权重）
- [x] 训练细节完整（batch/步数/lr/硬件、prompt 模板 Table 5、任务清单 Table 6/7 齐全）
- [x] 数据集可获取（预训练源 AgiBot/RoboMind/Galaxea/Bridge 公开；自采真实数据未明确发布）

---

## 速查卡片

> [!summary] ForeAct: Steering Your VLA with Efficient Visual Foresight Planning
> - **核心**: 用高效世界模型生成"想象的下一步观测"，作为额外视觉输入逐步引导现成 VLA，让 VLA 只管视觉-运动执行、不再做语义推理。
> - **方法**: Qwen3-VL 规划子任务 $l_t$ → SANA 线性 DiT 生成器 0.33s 出 640×480 未来观测 $\bm{G}_t$（116 万子任务跨本体预训练）→ 把 $\bm{G}_t$ 拼到视觉输入喂 $\pi_0$/$\pi_{0.5}$，VLM 闭环监控重规划；零架构改动。
> - **结果**: 11 真实任务平均 87.4%（较 $\pi_0$ +40.9%、较 VLM+$\pi_0$ +30.3%）；$\pi_{0.5}$ 70.3%→88.2%；LIBERO 97.5%；20% 数据仍 79%。
> - **代码**: https://github.com/mit-han-lab/foreact

---

*笔记创建时间: 2026-06-29*
