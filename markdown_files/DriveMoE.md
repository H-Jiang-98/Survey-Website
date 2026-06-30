---
title: "DriveMoE: Mixture-of-Experts for Vision-Language-Action Model in End-to-End Autonomous Driving"
method_name: "DriveMoE"
authors: [Zhenjie Yang, Yilin Chai, Xiaosong Jia, Qifeng Li, Yuqian Shao, Xuekai Zhu, Haisheng Su, Junchi Yan]
year: 2026
venue: CVPR
tags: [VLA, mixture-of-experts, end-to-end-autonomous-driving, flow-matching, multi-view-perception, skill-specialization]
zotero_collection: _inbox
image_source: online
arxiv_html: https://arxiv.org/html/2505.16278v2
created: 2026-06-29
---

# DriveMoE: Mixture-of-Experts for Vision-Language-Action Model in End-to-End Autonomous Driving

## 元信息

| 项目 | 内容 |
|------|------|
| 作者 | Zhenjie Yang, Yilin Chai, Xiaosong Jia（共一）, Qifeng Li, Yuqian Shao, Xuekai Zhu, Haisheng Su, Junchi Yan（通讯） |
| 机构 | 上海交通大学计算机学院 & 人工智能学院、复旦大学可信具身智能研究院、上海市多模态具身智能重点实验室、AnyScale AI |
| 会议 | CVPR 2026 |
| 类别 | VLA（Vision-Language-Action）/ 端到端自动驾驶 |
| 日期 | 2026-05（arXiv v2） |
| 项目主页 | https://thinklab-sjtu.github.io/DriveMoE/ |
| 链接 | [arXiv](https://arxiv.org/abs/2505.16278) / [Project](https://thinklab-sjtu.github.io/DriveMoE/) |

---

## 一句话总结

> 把 [[Mixture-of-Experts|MoE]] 同时引入感知与决策：用「场景专精」的 Vision MoE 动态选相机视角、用「技能专精」的 Action MoE 为不同驾驶行为激活专用专家，在 $\pi_0$ 改造的 [[VLA]] 基线 Drive-$\pi_0$ 上刷新 Bench2Drive 闭环 SOTA。

---

## 核心贡献

1. **把 $\pi_0$ 移植到自动驾驶**: 将源自具身智能的 [[VLA]] 基础模型 [[π0|$\pi_0$]] 扩展到端到端驾驶，构建统一处理「视觉感知 + 上下文理解 + 动作规划」的强基线 **Drive-$\pi_0$**。
2. **首个 vision+action 双层 MoE 驾驶框架**: 针对具身智能与自动驾驶的差异，提出 **DriveMoE**，是首个在感知与决策**两端**都引入 [[Mixture-of-Experts|MoE]] 的端到端驾驶框架，分别解决多视角冗余与多样驾驶行为问题。
3. **场景 Vision MoE + 技能 Action MoE**: 设计 **Scene-Specialized Vision MoE** 做动态相机视角选择，**Skill-Specialized Action MoE** 做行为专精规划（含 token-level / trajectory-level 两种路由风格），缓解单一策略网络的**模式平均（mode-averaging）**问题。
4. **闭环 SOTA**: 在 [[Bench2Drive]] 闭环仿真上取得 SOTA，相比 Drive-$\pi_0$ 驾驶分提升 22.8%、成功率提升 62.1%，显著增强对罕见安全攸关场景的鲁棒性。

---

## 问题背景

### 要解决的问题
端到端自动驾驶（E2E-AD）需要高效处理**多视角传感数据**，并稳健应对**多样且复杂**的驾驶场景，尤其是急转弯、紧急制动等**罕见但安全攸关**的长尾行为。现有 VLA 驾驶模型在开环指标上表现尚可，但在**闭环**评测中仍不令人满意。

### 现有方法的局限
作者把现有 VLA 驾驶方法的痛点归为两类（见 Figure 1）：

1. **视觉处理冗余 / 算力爆炸**:
   - **vanilla 视觉处理器**（LMDrive、DriveLM 等）无差别地处理所有相机视角，导致视觉 token 序列爆炸、计算负担重、表征冗余，限制效率与可扩展性；
   - **query-based 视觉处理器**（如 [[Q-Former]]/BLIP-2）用可学习 query 压缩 token，但**丢失精确几何与位置信息**，且需要大量额外预训练。
2. **单一策略网络的模式平均**: 当前 VLA 框架（CarLLaVA、LMDrive）普遍用**一个统一策略网络**处理全部驾驶行为，训练会偏向高频常见场景，对急刹、急转等罕见关键动作建模不足；缺乏显式技能专精，难以应对动态、强上下文依赖的驾驶情形。

同时，[[Mixture-of-Experts|MoE]] 已在 LLM（Mixtral、DeepSeekMoE）中证明「参数专精→强可扩展性」，但其在**视觉与动作域、尤其自动驾驶**中的应用基本空白。

### 本文的动机
- 在 Drive-$\pi_0$ 基线上识别出两大挑战：(i) 用 VLM 处理时空环视视频 token 对算力压力极大；(ii) 罕见困难场景表现差——即便有相似训练数据，也因不同行为间的**干扰效应**（$\pi_0$ 论文提到）而被「平均掉」。
- 受人类驾驶认知启发：司机会**有选择地关注关键视觉线索**（而非穷举所有视野），并**根据上下文切换驾驶技能**。这两点天然对应 Vision MoE（选视角）与 Action MoE（选技能）。
- MoE 的**稀疏激活**特性恰好能在不成比例增加算力的前提下，把模型容量切分给不同专家，既省算力又显式隔离行为模式。

---

## 方法详解

### 模型架构

DriveMoE 构建在 **Drive-$\pi_0$** 之上（见 Figure 2），整体是一个 [[VLA]] 流水线 + 两个 MoE 模块：

- **输入**: (i) 两帧连续**前视**图像（用于估计周围交通体速度）；(ii) 固定文本提示（如 "Please predict future trajectory"）；(iii) 当前车辆状态（速度、横摆角速度、历史轨迹等）；(iv) 由 Vision MoE 动态选出的额外相机视角。
- **Backbone**: 预训练 [[PaliGemma]]-3b-pt-224 VLM（沿用 $\pi_0$ 框架）。
- **核心模块**: [[Scene-Specialized Vision MoE|场景专精 Vision MoE]]（动态选相机视角）+ [[Skill-Specialized Action MoE|技能专精 Action MoE]]（嵌入 [[Flow Matching|流匹配]] 规划器，按行为激活专家）。
- **输出**: 未来 **10 个轨迹路点**，再经统一 [[PID Controller|PID 控制器]] 转为油门/刹车/转向。
- **规模**: 约 3008M 参数（Drive-$\pi_0$ 约 2606M）。

VLM 输入序列组织为：`⟨fixed_view⟩, ⟨fixed_view⟩, ⟨dynamic_view⟩, ⟨dynamic_view⟩, ⟨text⟩, ⟨text⟩`，即两路固定前视 + 两路动态选中视角 + 文本。

### 核心模块

#### 模块1: Scene-Specialized Vision MoE（场景专精视觉 MoE）

**设计动机**: 自动驾驶必须处理多视角、多时刻视觉输入，把所有相机帧拼进 transformer 会造成**视觉 token 瓶颈**——序列长度爆炸、训练推理变慢、收敛困难。模仿人类司机「按上下文优先关注特定视觉信息」。

**具体实现**（见 Figure 3）：
- 引入轻量 **vision router** $\boldsymbol{R}_{\text{vision}}$，输入前视嵌入 $\boldsymbol{e}^{\text{front}}_t$ 与未来目标路点 $\boldsymbol{g}_t$，输出在 $N$ 个相机视角上的选择概率分布 $\boldsymbol{p}_t$（公式1）。
- **关键工程**: 路由发生在**昂贵的 backbone 计算之前**，未被选中的视角可被完全跳过以省算力。
- 为每个相机视角加入**专属可学习位置嵌入（PE）**，跨视角保留空间与位置关系。
- 视角选择标签由「基于未来轨迹、bbox、地图」的人工启发式规则标注（附录 9），相机级标注**廉价且直接**（远比 token 级标注实际），用交叉熵损失训练（公式2）。
- 推理时取 router 的 **Top-1** 视角作为动态视角。

#### 模块2: Skill-Specialized Action MoE（技能专精动作 MoE）

**设计动机**: 人类司机在巡航、并线、超车、急刹间流畅切换，每种技能对应不同行为/轨迹特征。原始 $\pi_0$ 流匹配解码器虽能生成多样轨迹，但单一模型不可避免地**对多种行为取平均**，无法精准生成罕见却安全攸关的动作。

**具体实现**: 把解码器每层的稠密 [[FFN]] 替换为含 $K$ 个专家的 MoE 层，每个专家专精一部分驾驶技能；通过条件路由让每个输入只过少数专家，**隔离行为模式**、为罕见动作分配专属容量、保留轨迹数据的多模态性。探索两种路由风格：

- **Token-level Action MoE**（类 DeepSeek-MoE，Figure 4）: 每个 token（对应轨迹某一时刻）依据其**局部隐状态**独立选专家，让不同专家专精短时序依赖（加速/刹车/转向子任务）。路由 logits 经 softmax（公式3），输出按 router 置信度加权聚合共享专家与非共享专家（公式4）；用稀疏激活只取 Top-1/Top-2 专家，减少计算、防专家间互扰。
- **Trajectory-level Action MoE**（Figure 5）: 路由**先对整条 token 序列做平均**再选专家，即沿 batch（轨迹）维而非 token 维做专家选择，把每条轨迹当作代表某一场景/技能的整体。用驾驶技能标签 $y_k$ 经交叉熵训练 skill router（公式5），整体再用流匹配损失优化（公式6）。借鉴 DeepSeekMoE 给 router 注入**噪声**以增随机性、鼓励探索、缓解**专家坍缩**。
- 消融表明 **trajectory-level 一致优于 token-level**，故默认采用 trajectory-level。实际配置：**1 个共享专家 + 6 个非共享专家**，推理时激活 router 选出的 **Top-3 专家** 生成最终 10 路点轨迹。

#### 模块3: Two-Stage Training（两阶段训练：从 Teacher-Forcing 到自适应）

**设计动机**: 直接让 router 选专家训练不稳定，需要先「监督对齐」再「放手自选」。

**具体实现**:
- **Stage 1（Teacher-Forcing）**: Vision/Action MoE 均**只选 ground-truth 专家**，同时联合训练 router，显著稳定训练。训 12 epoch，VLM 从 PaliGemma-3b-pt-224 初始化，VLM 与 Action MoE 专家用两个独立优化器，lr=$5\times10^{-5}$、warmup、grad clip 1.0、梯度累积模拟 batch 128；损失权重 $\lambda_0=10,\ \lambda_2=10,\ \lambda_1=1$。
- **Stage 2（Adaptive）**: 从 Stage 1 检查点续训 6 epoch，**改为按 router 输出动态选专家**，去除对 GT 专家标注的依赖，对 router 的潜在错误建立鲁棒性、提升真实推理条件下的泛化；损失权重降为 $\lambda_0=5,\ \lambda_2=5,\ \lambda_1=1$，更强调轨迹学习。

### 关键公式与机制

#### 公式1: [[Scene-Specialized Vision MoE|Vision Router]] 选择概率

$$
\boldsymbol{p}_{t}=\operatorname{Softmax}\!\left(\boldsymbol{R}_{\text{vision}}\!\left(\boldsymbol{e}^{\text{front}}_{t},\,\boldsymbol{g}_{t}\right)\right)
$$

**含义**: 视觉路由器根据前视嵌入与未来目标路点，输出对各相机视角的选择概率分布。

**符号说明**:
- $\boldsymbol{e}^{\text{front}}_{t}$: $t$ 时刻前视图像嵌入
- $\boldsymbol{g}_{t}$: 由路线规划器给出的未来目标路点
- $\boldsymbol{p}_{t}\in\mathbb{R}^{N}$: $N$ 个相机视角上的选择概率，其元素 $p_t^v$ 为视角 $v$ 的选中概率

#### 公式2: [[Cross-Entropy Loss|Vision Router 交叉熵损失]]

$$
\mathcal{L}_{\text{Vision-Router}}=-\lambda_{0}\sum_{v=1}^{N}\boldsymbol{y}_{t}^{v}\log\!\left(\boldsymbol{p}_{t}^{v}\right)
$$

**含义**: 用人工标注的二值视角选择标签监督路由器，鼓励其主动选出决策相关的信息视角。

**符号说明**:
- $\boldsymbol{y}_t^v\in\{0,1\}$: 视角 $v$ 在 $t$ 时刻的二值选择标签（由轨迹/bbox/地图启发式标注）
- $\lambda_0$: 视觉路由损失权重（Stage1=10，Stage2=5）

#### 公式3: [[Token-level Action MoE]] 路由分布

$$
\boldsymbol{r}_{k}^{(\ell-1)}=\operatorname{Softmax}\!\left(\boldsymbol{R}_{\text{action}}\!\left(\mathbf{h}^{(\ell-1)}\right)\right),\quad k\in\{1,2,\dots,K\}
$$

**含义**: 第 $\ell$ 层动作路由器对输入隐状态计算各（非共享）专家的路由概率。

**符号说明**:
- $\mathbf{h}^{(\ell-1)}\in\mathbb{R}^{d}$: 第 $\ell$ 层输入隐状态（token 维度计算）
- $\boldsymbol{r}_k^{(\ell-1)}$: 第 $k$ 个专家的路由权重；$K$: 非共享专家数

#### 公式4: 专家输出聚合

$$
\boldsymbol{h}^{(\ell)}=\sum_{k=1}^{K}\boldsymbol{r}_{k}^{(\ell-1)}\,\boldsymbol{y}_{k}^{(\ell-1)}+\sum_{m=1}^{M}\boldsymbol{y}_{m}^{(\ell-1)}
$$

**含义**: 更新后的特征由「按路由权重加权的非共享专家输出」加上「全部共享专家输出」组成。

**符号说明**:
- $\boldsymbol{y}_k^{(\ell-1)}=E_k^{(\ell)}(\mathbf{h}^{(\ell-1)})$: 第 $k$ 个非共享专家 FFN 的输出
- $\boldsymbol{y}_m^{(\ell-1)}$: 第 $m$ 个共享专家输出（共 $M$ 个，恒激活）
- 实际用稀疏激活只取 Top-1/Top-2（token-level）或 Top-3（最终配置）专家

#### 公式5: [[Cross-Entropy Loss|Action Router 交叉熵损失]]

$$
\mathcal{L}_{\text{Action-Router}}=-\boldsymbol{y}_{k}\log\!\left(\boldsymbol{r}_{k}\right)
$$

**含义**: 用按场景标注的驾驶技能标签监督技能路由器，引导专家走向有意义的人类可解释技能类别。

**符号说明**:
- $\boldsymbol{y}_k\in\{1,\dots,K\}$: 驾驶技能标签（Merging / Overtaking / Emergency Brake / Give Way / Traffic Sign）
- $\boldsymbol{r}_k$: 技能路由器对专家 $k$ 的输出概率

#### 公式6: [[Skill-Specialized Action MoE|Action MoE 总损失]]

$$
\mathcal{L}_{\text{Action}}=\lambda_{1}\mathcal{L}_{\text{FM}}+\lambda_{2}\mathcal{L}_{\text{Action-Router}}
$$

**含义**: Action MoE 同时优化流匹配轨迹损失与动作路由损失，兼顾轨迹精度与专家专精。

**符号说明**:
- $\mathcal{L}_{\text{FM}}$: 流匹配轨迹损失（见公式7）
- $\lambda_1$: 流匹配损失权重（=1）；$\lambda_2$: 动作路由损失权重（Stage1=10，Stage2=5）

#### 公式7: [[Flow Matching|条件流匹配]]损失（附录 6）

$$
L^{\tau}(\theta)=\mathbb{E}_{p(\mathbf{A}_{t}\mid\mathbf{o}_{t}),\,q(\mathbf{A}_{t}^{\tau}\mid\mathbf{A}_{t})}\left\|\mathbf{v}_{\theta}\!\left(\mathbf{A}_{t}^{\tau},\mathbf{o}_{t}\right)-\mathbf{u}\!\left(\mathbf{A}_{t}^{\tau}\mid\mathbf{A}_{t}\right)\right\|^{2}
$$

**含义**: 以去噪方式预测未来动作轨迹，网络学习一个去噪流场，使其匹配真值方向，从而学习平滑、多模态的轨迹分布。

**符号说明**:
- 下标为时间步、上标为流匹配时间步，$\tau\in[0,1]$
- 带噪动作 $\mathbf{A}_t^{\tau}=\tau\mathbf{A}_t+(1-\tau)\epsilon$；目标方向 $\mathbf{u}(\mathbf{A}_t^{\tau}\mid\mathbf{A}_t)=\epsilon-\mathbf{A}_t$
- $\mathbf{v}_\theta$: 网络输出的去噪流；$\mathbf{o}_t$: 观测条件；$\epsilon$: 随机噪声

---

## 关键图表

<!-- 图片均使用 arXiv HTML 在线链接，未本地化 -->

### Figure 1: Comparison of Vision and Action Modeling Strategies / 视觉与动作建模策略对比

![Figure 1](https://arxiv.org/html/2505.16278v2/x1.png)

**说明**: 三种 VLA 驾驶建模策略对比。(a) **vanilla 视觉 token 编码**：所有环视图过 vision tower，token 冗余、算力高；(b) **query-based token 抽取**（如 Q-Former）：每图选子集 token，丢失空间结构且需额外预训练；(c) **本文 Scene-Specialized Vision MoE**：动态选一小撮相机（典型为前视 + 少数上下文相关侧/后视，图中红框），既省 token 又保留空间结构。下半部分对比单一统一策略（行为被平均）与本文的技能专精 Action MoE。此图是全文「双层 MoE」动机的可视化总览。

### Figure 2: Framework of DriveMoE / 整体框架

![Figure 2](https://arxiv.org/html/2505.16278v2/x2.png)

**说明**: DriveMoE 总体架构。**Scene-Specialized Vision MoE** 按实时驾驶上下文动态选相机视角、减少冗余，选中视角经 projector 融合为统一表征；**Skill-Specialized Action MoE** 嵌入流匹配规划器，为并线/超车/急刹等不同行为激活专用专家控制器。展示了从多相机输入 → VLM 主干 → 动作专家 → 轨迹输出的完整流程。

### Figure 3: The Scene-Specialized Vision Mixture-of-Experts / 场景专精视觉 MoE

![Figure 3](https://arxiv.org/html/2505.16278v2/x3.png)

**说明**: Vision MoE 细节。轻量 vision router 以前视嵌入 + 未来目标路点为输入，在 backbone 计算前预选最相关相机视角（未选视角直接跳过省算力），并为每视角加专属位置嵌入保留空间关系。

### Figure 4: Token-Level Skill-Specialized Action MoE / token 级技能专精动作 MoE

![Figure 4](https://arxiv.org/html/2505.16278v2/x4.png)

**说明**: token 级路由（类 DeepSeek-MoE）。每个轨迹时刻 token 依其局部隐状态独立选 Top-1/Top-2 专家，专精加速/刹车/转向等短时序子任务；含共享专家 + 非共享专家。

### Figure 5: Trajectory-Level Skill-Specialized Action MoE / 轨迹级技能专精动作 MoE

![Figure 5](https://arxiv.org/html/2505.16278v2/x5.png)

**说明**: 轨迹级路由（本文默认）。先对整条 token 序列做平均，再沿轨迹维选专家，把每条轨迹当作某一场景/技能整体；用技能标签交叉熵监督，router 注入噪声防专家坍缩。消融显示其稳定优于 token 级。

### Table 1: Bench2Drive Multi-Ability Benchmark / 多能力基准（成功率 %，*=专家特征蒸馏）

| Method | Venue | Merging | Overtaking | Emergency Brake | Give Way | Traffic Sign | Mean |
|--------|-------|---------|------------|-----------------|----------|--------------|------|
| TCP-traj* | NeurIPS 2022 | 8.89 | 24.29 | 51.67 | 40.00 | 46.28 | 34.22 |
| AD-MLP | Arxiv 2023 | 0.00 | 0.00 | 0.00 | 0.00 | 4.35 | 0.87 |
| UniAD-Base | CVPR 2023 | 14.10 | 17.78 | 21.67 | 10.00 | 14.21 | 15.55 |
| ThinkTwice* | CVPR 2023 | 27.38 | 18.42 | 35.82 | 50.00 | 54.23 | 37.17 |
| VAD | ICCV 2023 | 8.11 | 24.44 | 18.64 | 20.00 | 19.15 | 18.07 |
| DriveAdapter* | ICCV 2023 | 28.82 | 26.38 | 48.76 | 50.00 | 56.43 | 42.08 |
| DriveTransformer | ICLR 2025 | 17.57 | 35.00 | 48.36 | 40.00 | 52.10 | 38.60 |
| DiffAD | Arxiv 2025 | 30.00 | 35.55 | 46.66 | 40.00 | 46.32 | 38.79 |
| Drive-$\pi_0$ (Ours) | - | 26.25 | 26.67 | 45.00 | 30.00 | 38.95 | 33.37 |
| DriveMoE (Token-Level) | Ours | 28.75 | 31.11 | 51.67 | 40.00 | 52.63 | 40.83 |
| **DriveMoE (Traj-Level)** | **Ours** | **34.67** | **40.00** | **65.45** | 40.00 | **59.44** | **47.91** |

**说明**: Traj-Level DriveMoE 在五大能力均值（47.91）上全面领先，尤其 **Emergency Brake（65.45）** 与 Overtaking（40.00）这类罕见/安全攸关行为大幅超越所有基线，直接佐证「技能专精缓解模式平均」的核心主张；相比基线 Drive-$\pi_0$（33.37）提升明显。

### Table 2: Bench2Drive Benchmark (Closed-Loop & Open-Loop) / 闭环 + 开环主结果（*=专家特征蒸馏）

| Method | Venue | DS ↑ | SR(%) ↑ | Efficiency ↑ | Comfort ↑ | Avg. L2 ↓ |
|--------|-------|------|---------|--------------|-----------|-----------|
| TCP-traj* | NeurIPS 2022 | 59.90 | 30.00 | 76.54 | 18.08 | 1.70 |
| AD-MLP | Arxiv 2023 | 18.05 | 0.00 | 48.45 | 22.63 | 3.64 |
| VAD | ICCV 2023 | 42.35 | 15.00 | 157.94 | 46.01 | 0.91 |
| UniAD-Base | CVPR 2023 | 45.81 | 16.36 | 129.21 | 43.58 | 0.73 |
| ThinkTwice* | CVPR 2023 | 62.44 | 31.23 | 69.33 | 16.22 | 0.95 |
| DriveAdapter* | ICCV 2023 | 64.22 | 33.08 | 70.22 | 16.01 | 1.01 |
| GenAD | ECCV 2024 | 44.81 | 15.90 | - | - | - |
| DriveTransformer | ICLR 2025 | 63.46 | 35.01 | 100.64 | 20.78 | **0.62** |
| MomAD | CVPR 2025 | 44.54 | 16.71 | 170.21 | 48.63 | 0.82 |
| WoTE | ICCV 2025 | 61.71 | 31.36 | - | - | - |
| DriveMamba-L | ICLR 2026 | 66.82 | 37.73 | 152.91 | 18.77 | 0.70 |
| DiffAD | Arxiv 2025 | 67.92 | 38.64 | - | - | 1.55 |
| Raw2Drive | NeurIPS 2025 | 71.36 | 50.24 | **214.17** | 22.42 | - |
| Drive-$\pi_0$ | - | 55.85 | 30.00 | 173.63 | 35.70 | 1.13 |
| DriveMoE (Token-Level) | Ours | 66.94 | 35.45 | 158.80 | 6.86 | 0.96 |
| **DriveMoE (Traj-Level)** | **Ours** | **74.22** | 48.64 | 175.96 | 15.31 | 1.01 |

**说明**: Traj-Level DriveMoE 闭环 **DS=74.22 为全表最高**（超过 Raw2Drive 的 71.36）；相比 Drive-$\pi_0$，DS 提升 22.8%（55.85→74.22）、SR 提升 62.1%（30.00→48.64）。SR 48.64 略低于 Raw2Drive 的 50.24（后者为带对齐世界模型的强化学习方法）。开环 L2 由 DriveTransformer（0.62）最优，但作者强调（援引 AD-MLP/TransFuser++/Bench2Drive）**开环指标只反映收敛、闭环才反映真实驾驶能力**。

### Table 3: Ablation on Vision MoE / 视觉 MoE 消融（视角组合与监督）

| Exp | $I_F$ | $I_{FL}$ | $I_{FR}$ | $I_B$ | $I_{BL}$ | $I_{BR}$ | View | Sup. | DS ↑ | SR(%) ↑ | Latency ↓ | Mem.(MB) |
|-----|-------|----------|----------|-------|----------|----------|------|------|------|---------|-----------|----------|
| 1 | ✓ | × | × | × | × | × | Fixed | - | 55.85 | 30.00 | 100ms | 4100 |
| 2 | ✓ | ✓ | × | × | × | × | Fixed | - | 62.38 | 33.64 | 260ms | 5100 |
| 3 | ✓ | × | ✓ | × | × | × | Fixed | - | 61.52 | 32.73 | 260ms | 5100 |
| 4 | ✓ | × | × | ✓ | × | × | Fixed | - | 63.26 | 31.82 | 260ms | 5100 |
| 5 | ✓ | ✓ | ✓ | × | × | × | Fixed | - | 64.92 | 33.64 | 400ms | 7400 |
| 6 | ✓ | ✓ | ✓ | ✓ | × | × | Fixed | - | 64.18 | 33.64 | 550ms | 9600 |
| 7 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Fixed | - | 62.27 | 31.36 | 700ms | 11800 |
| 8 | ✓ | - | - | - | - | - | Dynamic | × | 69.71 | 44.09 | 260ms | 5100 |
| **9** | **✓** | - | - | - | - | - | **Dynamic** | **✓** | **74.22** | **48.64** | 260ms | 5100 |

**说明**: 加单个固定视角（Exp 2-4）有适度提升；但同时加多个固定视角（Exp 5-7）token 激增、收敛变难、延迟/显存飙升（700ms/11.8GB），性能反而下降。**动态选视角（Exp 8）即大幅超越所有固定方案且延迟仅 260ms**；再加监督信号（Exp 9 = DriveMoE）进一步把 DS/SR 推到 74.22/48.64。证明 Vision MoE 的「动态 + 监督」双要素都有效。

### Table 4: Router Accuracy / 路由器准确率（Bench2Drive-Base 验证集）

| Router | Accuracy (%) ↑ |
|--------|----------------|
| Vision Router | 88.85 |
| Action Router | 65.40 |

**说明**: 视觉路由准确率高达 88.85%，说明相机视角选择任务相对明确；动作（技能）路由 65.40% 偏低，反映驾驶技能边界模糊、场景可归多类，是后续提升空间所在。

### Table 5: Token vs Trajectory Level Action MoE / 两种路由风格对比

| Style | Share | Non-Share | DS ↑ | SR(%) ↑ |
|-------|-------|-----------|------|---------|
| Token-Level | 3 | 6 | 65.62 | 32.27 |
| **Traj-Level** | 3 | 6 | **73.88** | **48.64** |

**说明**: 同等专家配置下，**轨迹级路由全面优于 token 级**（DS 73.88 vs 65.62，SR 48.64 vs 32.27），故默认采用 Traj-Level。直觉上「每条轨迹整体对应一种驾驶技能」比「逐 token 切换专家」更契合驾驶行为的连贯性。

### Table 6: Ablation on Action MoE / 动作 MoE 专家数消融（Traj-Level）

| Exp | Share | Non-share | Supervision | DS ↑ | SR(%) ↑ |
|-----|-------|-----------|-------------|------|---------|
| 1 | 1 | 5 | ✓ | 73.81 | 47.73 |
| **2** | **1** | **6** | **✓** | **74.22** | **48.64** |
| 3 | 1 | 6 | × | 70.38 | 45.00 |
| 4 | 3 | 6 | ✓ | 73.88 | 48.64 |
| 5 | 5 | 6 | ✓ | 73.46 | 47.73 |
| 6 | 1 | 13 | ✓ | 70.88 | 44.50 |
| 7 | 1 | 44 | ✓ | 68.22 | 43.18 |

**说明**: Exp1（Bench2Drive 原 5 技能）→ Exp2（加 ParkingExit 专家共 6 个）提升至 DS 74.22（最优配置）；去掉动作监督（Exp3）明显下降（70.38）；增加共享专家（Exp4-5）效果与 Exp2 相当；**过度增加非共享专家（Exp6 的 13 个、Exp7 的逐场景 44 个）因专家负载不均反而掉点**（68.22）。说明专精专家数需适度平衡。

### Table 7: Drive-$\pi_0$ vs DriveMoE / 模块消融

| Method | DS ↑ | SR(%) ↑ |
|--------|------|---------|
| Drive-$\pi_0$ | 55.85 | 30.00 |
| w/o Vision MoE | 68.68 | 42.45 |
| w/o Action MoE | 67.31 | 40.56 |
| **DriveMoE** | **74.22** | **48.64** |

**说明**: 去掉任一 MoE（w/o Vision MoE 即只有 Action MoE，反之亦然）都明显掉点；两者**互补**，完整 DriveMoE 相比基线 Drive-$\pi_0$ 把 DS 从 55.85 提到 74.22、SR 从 30.00 提到 48.64。

### Table 8: Skill Set & Scenarios / 技能集与场景映射（附录 9）

| Skill | Scenario（节选） |
|-------|------------------|
| Merging | CrossingBicycleFlow, EnterActorFlow, HighwayExit, HighwayCutIn, MergeIntoSlowTraffic, ParkingExit, LaneChange, SignalizedJunctionLeft/RightTurn 等 16 项 |
| Overtaking | Accident(TwoWays), ConstructionObstacle(TwoWays), HazardAtSideLane(TwoWays), ParkedObstacle(TwoWays), VehicleOpenDoorTwoWays 等 9 项 |
| Emergency Brake | BlockedIntersection, DynamicObjectCrossing, OppositeVehicleTakingPriority/RunningRedLight, PedestrianCrossing, StaticCutIn, VehicleTurningRoute(Pedestrian), ControlLoss 等 12 项 |
| Give Way | InvadingTurn, YieldToEmergencyVehicle |
| Traffic Sign | NonSignalized/SignalizedJunctionTurn 系列, OppositeVehicle 系列, PedestrianCrossing, VanillaSignalized/NonSignalizedTurn 系列 等 18 项 |

**说明**: Bench2Drive 的 44 个场景被映射为 5 大驾驶技能，作为 Action Router 的监督标签来源（注意场景可被归入多个技能，存在重叠）。

### Table 9: Open-loop Planning on nuScenes / nuScenes 开环规划（附录 10）

| Method | L2 1s | L2 2s | L2 3s | L2 Avg ↓ | Col. 1s | Col. 2s | Col. 3s | Col. Avg ↓ |
|--------|-------|-------|-------|----------|---------|---------|---------|------------|
| UniAD | 0.48 | 0.74 | 1.07 | 0.76 | 0.12 | 0.13 | 0.28 | **0.17** |
| VAD-Base | 0.41 | 0.70 | 1.05 | **0.72** | 0.07 | 0.17 | 0.41 | 0.22 |
| Drive-$\pi_0$ | 0.51 | 0.73 | 1.11 | 0.78 | 0.14 | 0.18 | 0.39 | 0.24 |
| **DriveMoE** | 0.45 | 0.70 | 1.08 | 0.74 | 0.11 | 0.15 | 0.26 | **0.17** |

**说明**: 沿用 Bench2Drive 技能定义迁移到 nuScenes，DriveMoE 取得有竞争力的 L2（0.74）并把碰撞率从基线 0.24 降到 0.17（与 UniAD 持平、为全表最低之一），显示对真实场景的泛化；但作者重申开环评测不能反映真实驾驶性能。

### Table 10: Model Scale & Inference Cost / 模型规模与推理成本（附录 11）

| Method | View Num | View | DS ↑ | SR(%) ↑ | Params (M) | FLOPs (G) | Latency(ms) ↓ | Training Cost |
|--------|----------|------|------|---------|------------|-----------|---------------|---------------|
| Drive-$\pi_0$ | 2 | Fixed | 63.26 | 31.82 | 2606 | 3400 | 240 | 80 GPU-hours |
| Drive-$\pi_0$ | 6 | Fixed | 62.27 | 31.36 | 2606 | 7576 | 700 | 320 GPU-hours |
| **DriveMoE** | 2 | Dynamic | **74.22** | **48.64** | 3008 | 3896 | 260 | 120/80 GPU-hours |

**说明**: DriveMoE 用「2 固定 + 1 动态选」视角，仅比 2 视角 Drive-$\pi_0$ 略增算力（FLOPs 3400→3896、延迟 240→260ms），却把 DS 从 63.26 拉到 74.22；而暴力用 6 视角的 Drive-$\pi_0$ 反而更差（62.27）且延迟 700ms。增量成本主要来自一路额外相机与 Top-3 专家激活的 MoE 模块。

---

## 实验

### 数据集 / 基准

| 基准 | 规模 | 特点 | 用途 |
|------|------|------|------|
| [[Bench2Drive]] | base set 1000 clips（950 训 / 50 测）；官方 220 路线、每路线 1 个 corner case | [[CARLA]] 0.9.15.1 闭环；按 Merging/Overtaking/Emergency Brake/Give Way/Traffic Sign 五能力评测 | 训练 + 闭环评测（主） |
| nuScenes | 标准开环划分 | 真实世界开环规划 | 泛化验证（附录 10） |

### 实现细节

- **Backbone**: PaliGemma-3b-pt-224（VLM）；动作端为流匹配 transformer + Action MoE。
- **输入**: 2 帧连续前视（估周围车速）+ 1 路 Top-1 动态视角 + 车辆状态（当前/历史位置、速度、加速度、航向角）。
- **专家配置**: Action MoE 用 **1 共享 + 6 非共享专家**，推理激活 **Top-3**；输出 **10 个未来路点**。
- **训练**: 两阶段。Stage1 12 epoch（GT 专家 teacher-forcing，$\lambda_0=\lambda_2=10,\ \lambda_1=1$）→ Stage2 6 epoch（router 自选专家，权重降为 5/5/1）；lr=$5\times10^{-5}$、warmup、grad clip 1.0、梯度累积模拟 batch 128，VLM 与专家用两个独立优化器。
- **控制**: 统一 PID 控制器（所有方法一致以公平比较），转向 $K_P/K_I/K_D=1.25/0.75/0.3$，速度 $5.0/0.5/1.0$；速度取第 7 路点、转向取第 10 路点。
- **成本**: 约 3008M 参数、3896 GFLOPs、260ms 延迟、120/80 GPU-hours。

### 关键实验结论

- **闭环主结果（Table 2）**: DS=74.22（SOTA），相对 Drive-$\pi_0$ 提升 22.8% DS / 62.1% SR。
- **多能力（Table 1）**: 五能力均值 47.91 全面 SOTA，Emergency Brake 65.45 尤为突出。
- **Vision MoE 消融（Table 3）**: 动态 + 监督选视角既省算力（260ms/5.1GB）又最优；暴力堆视角会因 token 爆炸而掉点。
- **路由风格（Table 5）**: 轨迹级 > token 级。
- **专家数（Table 6）**: 6 个非共享专家最优；专家过多导致负载不均而退化。
- **模块互补（Table 7）**: Vision MoE 与 Action MoE 缺一不可。
- **泛化（Table 9）**: nuScenes 开环碰撞率降至 0.17。

---

## 批判性思考

### 优点
1. **机制清晰且双端协同**: 把「人类司机选视角 + 选技能」的直觉分别落到 Vision MoE 与 Action MoE，两者在消融中证明互补（Table 7），故事自洽、动机到方法到证据链完整。
2. **效率-性能平衡好**: Vision MoE 用动态 Top-1 选视角，仅比 2 视角基线略增算力却大幅超越「暴力 6 视角」方案（Table 3/10），对车载实时部署友好。
3. **直面长尾/模式平均**: 多能力表（Table 1）上 Emergency Brake、Overtaking 等罕见行为提升最大，正中「单一策略平均掉罕见动作」的痛点，是比单看总分更有说服力的证据。
4. **基线扎实**: 先把 $\pi_0$ 认真移植成 Drive-$\pi_0$ 再加 MoE，消融以同一基线对比，结论干净。

### 局限性
1. **依赖人工启发式标注**: Vision/Action router 的监督标签来自基于轨迹/bbox/地图的手工规则（附录 9）与固定 5 技能划分，标注质量与技能边界直接影响上限；Action Router 准确率仅 65.4%（Table 4）暴露技能定义的模糊性。作者自己也把无监督视角选择列为未来方向（附录 8）。
2. **仅在 Bench2Drive 单一闭环基准验证**: 闭环评测只在 CARLA/Bench2Drive base set（950 训练 clip）上做，规模有限；nuScenes 仅做开环，而作者反复强调开环不可靠——真实世界闭环能力未知。
3. **SR 未夺冠 + Comfort 偏低**: 闭环 SR 48.64 仍低于 Raw2Drive 的 50.24；Token-Level 变体 Comfort 仅 6.86、Traj-Level 15.31 也明显低于 VAD/MomAD（46+），提示 MoE 轨迹在平顺性上可能有牺牲，论文未深入分析。
4. **专家专精的可解释性证据偏弱**: 论文主张专家按技能专精，但缺少专家激活与技能类别对应关系的定量/可视化分析，专精是否真发生更多靠间接的性能提升推断。

### 潜在改进方向
1. **无监督/弱监督视角与技能发现**: 用聚类、token 剪枝（DySS、LightVLA）或自监督替代人工规则标注，降低标注成本、提升泛化（作者已提及）。
2. **更细/可学习的技能粒度**: Table 6 显示技能数需精细平衡，可探索可学习的技能划分或带负载均衡损失的路由，缓解专家坍缩与负载不均。
3. **闭环规模与真实车端验证**: 扩大训练数据、跨 CARLA 版本/真实平台验证闭环，并补充 Comfort 退化的成因分析与缓解。
4. **知识蒸馏部署**: 作者提出可把 DriveMoE 当 teacher，蒸馏出技能专用的紧凑 student 用于车端部署（附录 8），值得落地。

### 可复现性评估
- [x] 代码开源（论文声明将 release DriveMoE 与 Drive-$\pi_0$ 的代码与模型；项目主页 https://thinklab-sjtu.github.io/DriveMoE/ ）
- [ ] 预训练模型（声明将 release，截至论文未确认提供）
- [x] 训练细节完整（附录 7/11 给出两阶段超参、优化器、损失权重、PID 参数、算力成本）
- [x] 数据集可获取（Bench2Drive / CARLA / nuScenes 均公开；router 标注规则在附录 9 描述）

---

## 速查卡片

> [!summary] DriveMoE: Mixture-of-Experts for VLA in End-to-End Autonomous Driving
> - **核心**: 在 $\pi_0$ 改造的 Drive-$\pi_0$ 基线上，感知端加「场景专精 Vision MoE（动态选相机视角）」、决策端加「技能专精 Action MoE（按行为激活专家）」，缓解多视角冗余与模式平均。
> - **方法**: PaliGemma-3b 主干；vision router 取 Top-1 动态视角（监督 CE）；trajectory-level Action MoE，1 共享 + 6 非共享专家、Top-3 激活，流匹配出 10 路点；两阶段训练（GT teacher-forcing → router 自适应）。
> - **结果**: Bench2Drive 闭环 DS=74.22（SOTA），相对 Drive-$\pi_0$ +22.8% DS / +62.1% SR；多能力均值 47.91 全面领先，Emergency Brake 65.45 突出；仅略增算力（FLOPs 3400→3896，260ms）。
> - **代码**: https://thinklab-sjtu.github.io/DriveMoE/

---

*笔记创建时间: 2026-06-29*
