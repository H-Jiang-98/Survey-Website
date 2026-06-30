---
title: "Do You Have Freestyle? Expressive Humanoid Locomotion via Audio Control"
method_name: "RoboPerform"
authors: [Zhe Li, Cheng Chi, Yangyang Wei, Boan Zhu, Tao Huang, Zhenguo Sun, Yibo Peng, Pengwei Wang, Zhongyuan Wang, Fangzhou Liu, Chang Xu, Shanghang Zhang]
year: 2026
venue: CVPR
tags: [humanoid, locomotion, audio-control, diffusion-policy, mixture-of-experts, motion-tracking, teacher-student]
zotero_collection: _inbox
image_source: online
arxiv_html: https://arxiv.org/html/2512.23650v2
created: 2026-06-29
---

# Do You Have Freestyle? Expressive Humanoid Locomotion via Audio Control

## 元信息

| 项目 | 内容 |
|------|------|
| 作者 | Zhe Li, Cheng Chi, Yangyang Wei, Boan Zhu, Tao Huang, Zhenguo Sun 等 12 人 |
| 机构 | 北京大学 / 北京人工智能研究院（BAAI）等（通讯作者张姗姗）|
| 会议 | CVPR 2026 |
| 类别 | Humanoid / Locomotion（音频驱动人形运动生成）|
| 日期 | 2025-12（arXiv v2）|
| 项目主页 | （论文未提供公开主页）|
| 链接 | [arXiv](https://arxiv.org/abs/2512.23650) / [HTML](https://arxiv.org/html/2512.23650v2) |

---

## 一句话总结

> 提出首个统一的"音频→运动"人形控制框架 RoboPerform，遵循 **motion = content + style**，把音频当作隐式风格信号，免去显式动作重建与重定向，用 [[ResMoE|ΔMoE]] 教师 + 扩散学生策略直接生成与音乐/语音同步的舞蹈与共语手势。

---

## 核心贡献

1. **首个无重定向（retargeting-free）的音频到人形动作框架**: RoboPerform 直接把原始音频编码为隐式风格信号驱动人形运动，**绕过显式动作重建 → 重定向 → 跟踪**的级联管线，从根本上消除级联误差、降低延迟、紧耦合声学线索与底层关节驱动。
2. **"motion = content + style" 的解耦范式**: 把"做什么"（content，来自文本的运动 latent）与"怎么做"（style，来自音频的节奏/韵律）显式拆开；固定 content latent 作为常量条件，用不同音频作为风格调制信号，从而对多样节奏自适应、并具备对**未见音乐的即兴（freestyle）能力**。
3. **ΔMoE 残差混合专家教师策略**: 提出 [[Delta MoE|ΔMoE]]，把 3D 条件输入划分为**嵌套子空间** $\{S_i\}$ 交给 4 个专家，用残差融合做增量学习，本质上是 [[Classifier-Free Guidance|CFG]] 在连续多维条件上的结构化推广，强制专家互补、消除冗余信息。

---

## 问题背景

### 要解决的问题
人类天生"闻声而动"——鼓点引步、旋律催跳、语气带动手势，这是"感知先于模仿"的内在过程。而当前人形机器人要么只能复刻**预定义动作片段**，要么只能跟随**稀疏的语言指令**，缺乏富有表现力、随上下文即兴的控制能力。论文把人形运动重新表述为一个**生成问题**：给定条件信号，合成物理可行、风格对齐、语义落地的动作；并主张音频（音乐的节拍/速度/音色、语音的韵律/重音/节奏）应作为一等控制信号。核心追问就是题目本身——"Do you have freestyle?"

### 现有方法的局限
主流做法是"音频驱动的动作生成器 → 重定向到机器人 → 由控制器跟踪"，这种显式重建管线带来三个系统性弊病：
1. **级联误差累积**: 解码、重定向、跟踪每一环都引入误差，同时损伤表现力保真度与物理一致性；
2. **推理延迟高**: 多阶段串行处理延迟大，难以实际部署与快速迭代；
3. **高层声学线索与底层关节驱动松耦合**: 各模块孤立优化，无法保住风格、时序、动力学等细粒度表达。

此外纯模型控制（精确动力学但泛化差、建模繁琐）、纯学习控制（依赖手工奖励工程、难产生类人动作）以及语言驱动 locomotion（[[LangWBC]]、[[RLPF]]、[[RoboGhost]]）都未把**音频**作为条件信号。

### 本文的动机
一个更直接的洞见：**绕过显式动作重建，直接编码原始音频，把节拍、韵律、能量包络等风格元素当作隐式控制信号去调制人形运动**。基于 [[RoboGhost|RoboGhost]] 的 latent 驱动思想，把 content 定义为来自文本指令（如"a person is dancing"）经文生动作模型得到的高层运动 latent（指定核心任务），把 style 定义为音频信号（指定如何执行）。用 teacher-student 框架落地该解耦：教师用 [[Delta MoE|ΔMoE]] 适配多样运动模式，再蒸馏给基于扩散的学生策略进行音频风格注入，从而 retargeting-free、低延迟、高保真。

---

## 方法详解

### 模型架构

RoboPerform 采用 **两阶段 teacher-student** 架构（见 Figure 2），核心由三部分组成：

- **输入**: 原始音频（音乐 / 语音）+ 文本指令导出的运动 latent + 机器人本体感知状态
- **音频-运动对齐模块**: 6 层 [[Transformer]] adaptor（带时序注意力），用 [[InfoNCE]] 损失把音频 latent $l_{\text{audio}}$ 对齐到运动 latent $l_{\text{motion}}$，把运动学先验嵌入音频
- **教师策略**: [[Delta MoE|ΔMoE]]（4 专家残差 MoE），用 [[PPO]] + [[RL]] 训练，输入本体状态 + 特权信息 + 参考运动跟踪目标
- **学生策略**: 基于 [[Diffusion Policy|扩散]]的生成器（MLP 主干 + [[AdaLN]] 条件注入），用 [[DAgger]] 蒸馏，content latent 引导去噪、audio latent 逐层注入做风格调制
- **输出**: 23 维目标关节位置（Unitree [[G1]] 人形机器人）
- **训练 / 部署**: IsaacGym 训练，零样本迁移到 MuJoCo，最终在真实 [[Unitree G1]] 上部署

整体哲学是 **motion = content + style**：固定 content（运动 latent 为常量条件）以保住核心任务，注入 style（不同音频）以驱动节奏自适应的多样动作。

### 核心模块

#### 模块1: Audio-Motion Alignment（音频-运动对齐 adaptor）

**设计动机**: 直接用音频 latent 作条件去驱动动作生成，从而**免训练专门的 audio-to-motion 生成器**——前提是给音频 latent 注入运动学信息。

**具体实现**:
- adaptor = 6 层 [[Transformer]]，处理音频 latent，并加**时序注意力**捕捉音频内在的节奏结构；
- 运动 latent 来自预训练 [[VAE]]；
- 用 [[InfoNCE]] 对比损失把 $(l_{\text{audio}}^{(i)},l_{\text{motion}}^{(i)})$ 拉近、与不匹配对推远，从而把运动学先验嵌入音频 latent，保证音频与运动的节奏一致性（公式1）。

#### 模块2: ΔMoE（残差混合专家教师策略）

**设计动机**: 让不同专家处理不同的条件子集，**最大化所学知识的多样性与互补性**，消除信息冗余。核心是"嵌套条件子空间划分 + 残差增量学习"。

**具体实现**:
- 把 3D 条件 $\mathbf{c}=[c_1,c_2,c_3]^T$ 划分为嵌套滤子 $\{0\}=S_1\subset S_2\subset S_3\subset S_4=\mathbb{R}^3$（公式3）；
- 4 个专家 $\{e_i\}$，$e_i$ 的策略 $\pi_i(\mathbf{a}\mid\mathbf{c}_{S_i})$ 只依赖子空间 $S_i$：$e_1$ 取全零（无条件先验 $p(\mathbf{a})$）、$e_2$ 只收本体状态、…、$e_4$ 收全部条件（完整条件 $p(\mathbf{a}\mid\mathbf{c})$）；
- 门控网络对 $\mathbf{c}$ 输出归一化权重 $\mathbf{w}$，用**残差融合**得到最终动作（公式4），等价于条件增量 $\Delta\mathbf{a}_i$ 的加权和（公式5）；
- 每个 $\Delta\mathbf{a}_i$ 直接类比 [[Classifier-Free Guidance|CFG]] 的 guidance 项，量化"加入第 $i$ 个条件维度带来的信息增益"，保证专家贡献不重叠；
- 教师以本体观测 + 参考运动为输入，输出动作 $\mathbf{a}_t$，由多项跟踪奖励优化（见 Table 10）。

#### 模块3: Audio-conditioned Policy Distill（音频条件扩散学生策略）

**设计动机**: 落地 motion=content+style——固定 content 保任务、注入 style 调节奏。

**具体实现**:
- 用预训练运动生成器把高层语义描述（"The person is dancing to the music" / "giving a speech"）编码为**运动 latent**；训练时所有动作共享同一 content latent，作为扩散去噪的主条件；
- 对齐后的 audio latent 作为**外部风格控制信号在扩散主干多层注入**（公式6）：$\mathbf{o}_i=\text{Layer}_i(\mathbf{o}_{i-1}, l_{\text{motion}})+\alpha\, l_{\text{audio}}$；
- 用 [[DAgger]] 式蒸馏：在仿真中 rollout 学生策略，对访问状态查询教师得最优动作 $\hat{\mathbf{a}}$ 监督；
- 扩散前向加噪 $\mathbf{x}_t=\sqrt{\bar\alpha_t}\,\mathbf{a}+\sqrt{1-\bar\alpha_t}\,\bm\epsilon$，采用 $\mathbf{x}_0$-prediction 参数化（论文消融表明显著优于 $\epsilon$-prediction），重构动作 $\hat{\mathbf{a}}_t=\frac{\mathbf{x}_t-\sqrt{1-\bar\alpha_t}\,\epsilon_\theta(\mathbf{x}_t,t)}{\sqrt{\bar\alpha_t}}$，用 MSE 损失 $\mathcal{L}=\|\mathbf{a}-\hat{\mathbf{a}}_t\|_2^2$ 监督；
- 学生**无特权信息、无显式参考运动**，靠 25 帧本体历史 + audio/motion latent，实现 retargeting-free。

### 关键公式与机制

#### 公式1: [[InfoNCE]] 音频-运动对齐损失

$$
\mathcal{L}_{\text{InfoNCE}}=-\frac{1}{N}\sum_{i=1}^{N}\log\frac{\exp\!\left(\operatorname{sim}(l_{\text{audio}}^{(i)},\,l_{\text{motion}}^{(i)})\right)}{\sum_{j=1}^{N}\exp\!\left(\operatorname{sim}(l_{\text{audio}}^{(i)},\,l_{\text{motion}}^{(j)})\right)}
$$

**含义**: 以批内配对为正样本、其余为负样本的对比学习，把音频 latent 在嵌入空间拉近其对应运动 latent、推远无关项，从而给音频注入运动学先验。

**符号说明**:
- $\operatorname{sim}(u,v)=\frac{u^\top v}{\tau}$: 归一化向量的缩放余弦相似度；$\tau>0$ 为温度
- $N$: 批大小（音频-运动配对数）

#### 公式2: [[Classifier-Free Guidance|CFG]] 插值融合（ΔMoE 的理论原型）

$$
\mathbf{a}\propto\frac{p(\mathbf{a}\mid\mathbf{c})^{\gamma}}{p(\mathbf{a})^{1-\gamma}}
$$

**含义**: 标准 CFG 同时训练条件 $p(\mathbf{a}\mid\mathbf{c})$ 与无条件 $p(\mathbf{a})$，推理时按 $\gamma$ 插值，平衡条件对齐与多样性。ΔMoE 把它推广到嵌套部分条件。

**符号说明**:
- $\gamma$: guidance 强度；$\mathbf{c}$: 条件向量

#### 公式3: 嵌套条件子空间滤子

$$
\{0\}=S_{1}\subset S_{2}\subset S_{3}\subset S_{4}=\mathbb{R}^{3}
$$

**含义**: 把 3D 条件空间组织成由空集到全集的递增链，专家 $e_i$ 只看子空间 $S_i$，从而逐维引入条件。

**符号说明**:
- $S_1=\{0\}$（无条件）→ $S_4=\mathbb{R}^3$（全条件 $\{c_1,c_2,c_3\}$）

#### 公式4: ΔMoE 残差融合

$$
\mathbf{a}=w_{1}\mathbf{a}_{1}+\sum_{i=2}^{4}w_{i}\,(\mathbf{a}_{i}-\mathbf{a}_{i-1})
$$

**含义**: 门控权重 $w_i$ 对各专家输出做残差加权——$w_1$ 直接取无条件基线 $\mathbf{a}_1$，其余取相邻专家的**增量** $\mathbf{a}_i-\mathbf{a}_{i-1}$，避免条件信息重复计入。

**符号说明**:
- $\mathbf{a}_i$: 专家 $e_i$ 的输出动作；$\mathbf{w}=[w_1,\dots,w_4]^T$ 归一化（$\sum w_k=1$）

#### 公式5: 等价的条件增量加权和

$$
\mathbf{a}=\sum_{i=1}^{4}w_{i}\,\Delta\mathbf{a}_{i},\quad \text{where }\ \Delta\mathbf{a}_{i}=\mathbb{E}[\mathbf{a}\mid\mathbf{c}_{S_{i}}]-\mathbb{E}[\mathbf{a}\mid\mathbf{c}_{S_{i-1}}]
$$

**含义**: 残差融合等价于"条件增量"的加权和；每个 $\Delta\mathbf{a}_i$ 量化"加入第 $i$ 维条件的信息增益"，直接类比 CFG 的 guidance 项，保证专家间贡献非重叠、互补。

**符号说明**:
- $\Delta\mathbf{a}_i$: 第 $i$ 个条件维度的边际贡献（$\mathbf{a}_0=\mathbf{0}$）

#### 公式6: 音频风格逐层注入

$$
\mathbf{o}_{i}=\text{Layer}_{i}\big(\mathbf{o}_{i-1},\,l_{\text{motion}}\big)+\alpha\, l_{\text{audio}}
$$

**含义**: 扩散主干第 $i$ 层在以 content latent $l_{\text{motion}}$ 为条件去噪的同时，叠加缩放后的 audio latent 作为外部风格信号，逐层把去噪轨迹引向"节奏化风格"的动作。

**符号说明**:
- $\mathbf{o}_i$: 第 $i$ 层输出；$\alpha$: 音频注入强度系数

#### 公式7: 终止课程（Termination Curriculum）

$$
\theta\leftarrow\operatorname{clip}\!\left(\theta\cdot(1-\delta),\ \theta_{\text{min}},\ \theta_{\text{max}}\right)
$$

**含义**: 训练中逐步减小跟踪偏差容忍阈值 $\theta$（初值 1.5）以提升难度，超阈值则提前终止 episode。

**符号说明**:
- $\delta$: 衰减步长；$\theta_{\text{min}},\theta_{\text{max}}$: 裁剪上下界

#### 公式8: 惩罚课程（Penalty Curriculum）

$$
\alpha\leftarrow\operatorname{clip}\!\left(\alpha\cdot(1+\delta),\ \alpha_{\text{min}},\ \alpha_{\text{max}}\right),\qquad \hat{r}_{\text{penalty}}\leftarrow\alpha\cdot r_{\text{penalty}}
$$

**含义**: 逐步增大正则项权重 $\alpha$，让惩罚奖励 $\hat r_{\text{penalty}}$ 随训练加重，促使行为更稳定、物理更合理。

**符号说明**:
- $r_{\text{penalty}}$: 原始正则惩罚项；$\alpha$: 渐增权重

#### 公式9: 自适应 Sigma 的有界指数奖励

$$
A=\exp\!\left(-\frac{x^{2}}{\sigma^{2}}\right)
$$

**含义**: 除足部接触外的任务奖励均采用有界指数形式，把跟踪误差 $x$ 映射到 $(0,1]$；$\sigma$ 自适应调节灵敏度（见 Table 10 各项奖励均为此形式）。

**符号说明**:
- $x$: 状态/关节/刚体的跟踪误差；$\sigma$: 自适应带宽

#### 公式10: Beat Alignment Score（BAS）节拍对齐度

$$
\text{BAS}=\frac{1}{m}\sum_{i=1}^{m}\exp\!\left(-\frac{\min_{\forall t_{y}^{j}\in B_{y}}\|t_{x}^{i}-t_{y}^{j}\|^{2}}{2\sigma^{2}}\right)
$$

**含义**: 评估生成动作的运动节拍 $t_x^i$ 与音乐节拍 $B_y$ 的时间对齐程度，越大越同步——是衡量"动作跟着音乐打节拍"的核心指标。

**符号说明**:
- $t_x^i$: 第 $i$ 个运动节拍时刻；$B_y$: 音乐节拍集合；$m$: 运动节拍数；$\sigma$: 高斯带宽

---

## 关键图表

<!-- 图片均使用 arXiv HTML 在线链接，未本地化 -->

### Figure 1: Teaser — RoboPerform as Dancer & Talker / 总览示意

![Figure 1](https://arxiv.org/html/2512.23650v2/x1.png)

**说明**: RoboPerform 让人形机器人化身舞者与演说者——以音频为信号控制人形 locomotion，输入语音或音乐即可生成节奏对齐的共语手势与舞蹈动作。直观点题"motion = content + style"。

### Figure 2: Overview of RoboPerform / 整体框架

![Figure 2](https://arxiv.org/html/2512.23650v2/x2.png)

**说明**: 两阶段方案——先训练 adaptor 把运动学信息注入音频模态，再以 [[Delta MoE|ΔMoE]] 教师（RL 训练）+ 扩散学生（条件于 audio latent 去噪动作）蒸馏。固定运动 latent 作常量 content 条件，用不同音频作 style 调制，生成自适应多样节奏的动作。

### Figure 3: Overview of ΔMoE / ΔMoE 结构

![Figure 3](https://arxiv.org/html/2512.23650v2/x3.png)

**说明**: ΔMoE 4 专家逐维接入条件（$e_1$ 全零→$e_4$ 全条件），门控网络输出权重做残差融合（公式4/5），把 CFG 的"条件残差对比"推广到嵌套条件子空间，强制专家互补、消冗余。

### Figure 4: T-SNE of ΔMoE vs Vanilla MoE / 表征可视化

![Figure 4](https://arxiv.org/html/2512.23650v2/x4.png)

**说明**: 各组件输出的 T-SNE。相比 vanilla MoE，ΔMoE 的专家输出在特征空间分布更具区分性/互补性，佐证残差增量学习消除了冗余、专家分工更明确。

### Figure 5: Ablation — Tracking Performance / 显式 vs 隐式管线消融

![Figure 5](https://arxiv.org/html/2512.23650v2/x5.png)

**说明**: 音乐→舞蹈、语音→手势两任务在 IsaacGym 与 MuJoCo 的跟踪性能。Baseline 用预训练动作生成器（EMAGE/FineNet）显式生成动作再驱动学生策略；显式生成不仅增加计算开销（含 1000 次迭代 PBHC 重定向），还降低成功率、引入额外跟踪误差。

### Figure 6: Qualitative Results in IsaacGym & MuJoCo / 定性跟踪

![Figure 6](https://arxiv.org/html/2512.23650v2/x6.png)

**说明**: 上半为音乐→locomotion 的跟踪表现，下半为语音→locomotion。展示策略能贴合音频节奏、在动态过渡中保持平衡，并跨物理引擎/硬件平台泛化。

### Figure 7: MLP vs Diffusion Policy & Freestyle / 扩散策略优势

![Figure 7](https://arxiv.org/html/2512.23650v2/x7.png)

**说明**: 上半同一段动作上 MLP 策略 vs 扩散策略——MLP 跟踪差，扩散凭分布建模与鲁棒性显著更优；下半为面对**未见音乐**的即兴能力：扩散策略完整完成动作序列，MLP 立即摔倒，凸显泛化/freestyle 能力。

### Figure 8: PHC and GMR Retargeting / 重定向可视化（附录）

![Figure 8](https://arxiv.org/html/2512.23650v2/x8.png)

**说明**: PHC 与 GMR 两种重定向结果的定性对比（用于数据预处理 / baseline 对照）。

### Figure 9–10: Real-world Music-to-Locomotion / 真实世界音乐驱动（附录）

![Figure 9](https://arxiv.org/html/2512.23650v2/x9.png)
![Figure 10](https://arxiv.org/html/2512.23650v2/x10.png)

**说明**: Unitree [[G1]] 在真实世界由音乐驱动跳舞的执行序列，验证 sim-to-real 可部署性。

### Figure 11: Real-world Speech-to-Locomotion / 真实世界语音驱动（附录）

![Figure 11](https://arxiv.org/html/2512.23650v2/x11.png)

**说明**: 真实世界由语音驱动产生共语手势的执行序列。

### Table 1: Audio-Motion Alignment / 音频-运动检索对齐性能

| Method | R@1 ↑ | R@2 ↑ | R@3 ↑ | MM-Dist ↓ |
|--------|-------|-------|-------|-----------|
| Music-Motion | **66.7** | **78.8** | **83.5** | **1.154** |
| Speech-Motion | 64.6 | 76.5 | 82.1 | 1.232 |

**说明**: 评估 adaptor 把音频段对齐到运动 latent 空间检索正确运动的能力。音乐-运动对齐略优于语音-运动，整体 R@1 在 64–67%、MM-Dist 约 1.15–1.23，说明 InfoNCE 对齐有效。

### Table 2: Motion Tracking in Simulation / 仿真跟踪性能（vs 显式 baseline）

| | IsaacGym Succ ↑ | IsaacGym $E_{\text{mpjpe}}$ ↓ | IsaacGym $E_{\text{mpkpe}}$ ↓ | MuJoCo Succ ↑ | MuJoCo $E_{\text{mpjpe}}$ ↓ | MuJoCo $E_{\text{mpkpe}}$ ↓ |
|---|---|---|---|---|---|---|
| **BEAT2** Baseline | 0.98 | 0.07 | 0.05 | 0.94 | 0.13 | 0.12 |
| **BEAT2 Ours** | **0.99** | **0.05** | **0.04** | **0.96** | **0.10** | **0.09** |
| **FineDance** Baseline | 0.88 | 0.24 | 0.21 | 0.61 | 0.32 | 0.27 |
| **FineDance Ours** | **0.93** | **0.18** | **0.16** | **0.67** | **0.26** | **0.24** |

**说明**: Baseline = EMAGE/FineNet 显式生成动作 → 重定向到 G1 → MLP 跟踪。RoboPerform 在两数据集、两仿真器上成功率均最高、关节/关键点误差均最低；FineDance（舞蹈，更动态）提升尤其明显（IsaacGym 0.88→0.93、MuJoCo 0.61→0.67）。

### Table 3: Ablation — Vanilla MoE vs ΔMoE / 残差 MoE 消融

| | IsaacGym Succ ↑ | $E_{\text{mpjpe}}$ ↓ | $E_{\text{mpkpe}}$ ↓ | MuJoCo Succ ↑ | $E_{\text{mpjpe}}$ ↓ | $E_{\text{mpkpe}}$ ↓ |
|---|---|---|---|---|---|---|
| **BEAT2** Vanilla MoE | 0.97 | 0.14 | 0.10 | 0.94 | 0.16 | 0.14 |
| **BEAT2 ΔMoE** | **0.99** | **0.05** | **0.04** | **0.96** | **0.10** | **0.09** |
| **FineDance** Vanilla MoE | 0.89 | 0.24 | 0.22 | 0.61 | 0.29 | 0.26 |
| **FineDance ΔMoE** | **0.93** | **0.18** | **0.16** | **0.67** | **0.26** | **0.24** |

**说明**: ΔMoE 全面优于 vanilla MoE，尤其误差大幅下降（BEAT2 IsaacGym mpjpe 0.14→0.05），证明嵌套子空间 + 残差增量学习带来更强专家互补性。

### Table 4: Ablation — With/Without Content / content 信息消融

| | IsaacGym Succ ↑ | $E_{\text{mpjpe}}$ ↓ | $E_{\text{mpkpe}}$ ↓ | MuJoCo Succ ↑ | $E_{\text{mpjpe}}$ ↓ | $E_{\text{mpkpe}}$ ↓ |
|---|---|---|---|---|---|---|
| **BEAT2** − Content | 0.96 | 0.11 | 0.09 | 0.91 | 0.12 | 0.10 |
| **BEAT2** + Content | **0.99** | **0.05** | **0.04** | **0.96** | **0.10** | **0.09** |
| **FineDance** − Content | 0.91 | 0.20 | 0.17 | 0.66 | 0.25 | 0.24 |
| **FineDance** + Content | **0.93** | **0.18** | **0.16** | **0.67** | 0.26 | 0.24 |

**说明**: 加入固定 content latent（语义锚）提升成功率、降低误差，验证 motion=content+style 中 content 分量的必要性（BEAT2 上提升更显著）。

### Table 5: Ablation — With/Without Adaptor / adaptor 消融

| | IsaacGym Succ ↑ | $E_{\text{mpjpe}}$ ↓ | $E_{\text{mpkpe}}$ ↓ | MuJoCo Succ ↑ | $E_{\text{mpjpe}}$ ↓ | $E_{\text{mpkpe}}$ ↓ |
|---|---|---|---|---|---|---|
| **BEAT2** − Adaptor | 0.88 | 0.29 | 0.27 | 0.83 | 0.36 | 0.35 |
| **BEAT2** + Adaptor | **0.99** | **0.05** | **0.04** | **0.96** | **0.10** | **0.09** |
| **FineDance** − Adaptor | 0.79 | 0.49 | 0.48 | 0.51 | 0.58 | 0.53 |
| **FineDance** + Adaptor | **0.93** | **0.18** | **0.16** | **0.67** | **0.26** | **0.24** |

**说明**: 去掉 adaptor（不给音频注入运动学先验）成功率与误差全面崩坏（FineDance IsaacGym 0.79→0.93、误差近 3 倍下降），证明音频-运动对齐是整个 retargeting-free 管线的关键基石。

### Table 6 (paper Table 7): Proprioceptive & Privileged States / 状态表征

| 组别 | 关键组成 | 维度 |
|------|---------|------|
| Proprioceptive | DoF 位置/速度/上一动作各 23×5、root 角速度/投影重力各 3×5 | 75×5 |
| Privileged | root 线速度、参考身体位置(81)、位置差(81)、随机化 CoM/质量/刚度/阻尼/摩擦/延迟 | 250 |
| Teacher 总输入 | 本体 75×5 + DoF 位置 23 + 关键点 81 + root 速度/角速度/朝向各 3 | 489 |
| Student 总输入 | Motion latent 64 + Audio latent 256 + 本体 75×(25+1) | 2270 |

**说明**: 关键设计——参考运动/特权信息**只给教师**；学生用 **25 帧本体历史 + audio/motion latent** 替代，实现无特权、无显式参考的 retargeting-free 推理。输出均为 23 维目标关节位置。

### Table 7 (paper Table 8): Training Hyperparameters / 训练超参

| 项 | 值 |
|----|----|
| 优化器 / lr / batch | Adam / $1\times10^{-3}$ / 8192 |
| Teacher（PPO）| GAE $\gamma$=0.99、$\lambda$=0.95、clip 0.2、entropy 0.01、5 epochs、4 mini-batch、Actor MLP [768,512,128]、4 experts |
| Student | MLP 4+1 层、size [1792,1792,1792,23] |

**说明**: 教师 PPO、学生扩散 MLP 主干的完整超参；专家数固定为 4（Table 16 消融显示 4 为最佳）。

### Table 8 (paper Table 9): Domain Randomization / 域随机化

| 类别 | 项 | 值 |
|------|----|----|
| 动力学 | 摩擦 / PD 增益 | $\mathcal{U}(0.2,1.5)$ / $\mathcal{U}(0.75,1.25)$ |
| 动力学 | link 质量 / 踝惯量 | $\mathcal{U}(0.9,1.1)\times$default |
| 动力学 | base CoM 偏移 / 控制延迟 | $\mathcal{U}(-0.05,0.05)$ m / $\mathcal{U}(0,40)$ ms |
| 扰动 | 随机推力间隔 / 速度 | [5,10] s / 0.5 m/s |

**说明**: 为 sim-to-real 鲁棒性所做的域随机化，含 [[ERFI]] 力矩扰动（0.05×力矩上限）。

### Table 9 (paper Table 10): Reward Terms / 奖励项与权重

| 类别 | 代表项（权重）|
|------|--------------|
| 跟踪奖励 | 关节位置(1.0)、关节速度(1.0)、身体位置(1.0)、身体旋转(0.5)、VR 3 点位置(1.6)、足部位置(1.0)、Max 关节位置(1.0)、Contact Mask(0.5) |
| 正则惩罚 | 关节/速度/力矩越限(−10/−5/−5)、滑移(−1.0)、足部气时(−1.0)、绊倒(−2.0)、力矩($-10^{-6}$)、动作变化率(−0.02)、碰撞(−30)、终止(−200) |

**说明**: 跟踪奖励均为有界指数形式（公式9），VR 3 点权重最高(1.6) 强调上肢/头部表现力；正则项配合惩罚课程（公式8）渐增。

### Table 10 (paper Table 11): Ablation with BAS / 加入节拍对齐度的消融

| | IsaacGym Succ ↑ | $E_{\text{mpjpe}}$ ↓ | $E_{\text{mpkpe}}$ ↓ | MuJoCo Succ ↑ | $E_{\text{mpjpe}}$ ↓ | $E_{\text{mpkpe}}$ ↓ | BAS ↑ |
|---|---|---|---|---|---|---|---|
| **BEAT2** Baseline | 0.98 | 0.08 | 0.06 | 0.94 | 0.16 | 0.14 | 0.163 |
| **BEAT2 Ours** | **0.99** | **0.05** | **0.04** | **0.96** | **0.10** | **0.09** | **0.197** |
| **FineDance** Baseline | 0.86 | 0.26 | 0.23 | 0.58 | 0.35 | 0.32 | 0.176 |
| **FineDance Ours** | **0.93** | **0.18** | **0.16** | **0.67** | **0.26** | **0.24** | **0.214** |

**说明**: Baseline = 把音乐 latent 直接与其他观测/运动 latent 拼接输入学生。把音乐作为外部条件去调制 content 时，BAS（节拍对齐）显著更高（0.163→0.197、0.176→0.214），印证 style 注入设计提升节奏同步。

### Table 11 (paper Table 12): Inference Time vs DDIM Steps / 推理时延

| 采样 | 平均时间 (s × $10^{-3}$) |
|------|--------------------------|
| DDIM-2 | **5.3** |
| DDIM-4 | 11.6 |
| DDIM-6 | 13.4 |
| DDIM-8 | 17.6 |
| DDIM-10 | 18.9 |

**说明**: 采样步数与时延近线性增长；DDIM-2 仅 5.3 ms 即可用，支撑低延迟实时部署。

### Table 12 (paper Table 13): Noise Scale Ablation / 噪声尺度消融

| Noise Scale $\beta_{\max}$ | Steps | Success (%) |
|---|---|---|
| 0.10 | 2 | 92.0 |
| 0.15 | 2 | 92.0 |
| **0.20** | 2 | **93.0** |
| 0.25 | 2 | 91.0 |
| 0.30 | 2 | 91.0 |

**说明**: $\beta_{\max}=0.20$ 时成功率最高(93%)，噪声过大或过小都略降。

### Table 13 (paper Table 14): Sampling Strategy / 采样策略消融

| 策略 | Steps | Success (%) | Latency (s × $10^{-3}$) |
|---|---|---|---|
| **DDIM ($\eta$=0)** | 2 | **93.0** | 5.3 |
| DDIM ($\eta$=0.5) | 2 | 86.0 | 5.3 |
| DDPM (Stochastic) | 2 | 65.0 | 8.6 |

**说明**: 确定性 DDIM($\eta$=0) 在低步数下成功率与时延双优；随机性越强成功率越差，说明该控制任务偏好确定性采样。

### Table 14 (paper Table 15): Optimization Objective / 预测目标消融

| | IsaacGym Succ ↑ | $E_{\text{mpjpe}}$ ↓ | $E_{\text{mpkpe}}$ ↓ | MuJoCo Succ ↑ | $E_{\text{mpjpe}}$ ↓ | $E_{\text{mpkpe}}$ ↓ |
|---|---|---|---|---|---|---|
| $\epsilon$-prediction | 0.72 | 0.46 | 0.43 | 0.49 | 0.58 | 0.56 |
| **$x_0$-prediction** | **0.93** | **0.18** | **0.16** | **0.67** | **0.26** | **0.24** |

**说明**: $x_0$-prediction 大幅优于 $\epsilon$-prediction（成功率 0.72→0.93），这也是论文选择直接预测干净动作的依据。

### Table 15 (paper Table 16): Number of Experts / 专家数消融

| N | IsaacGym Succ ↑ | $E_{\text{mpjpe}}$ ↓ | $E_{\text{mpkpe}}$ ↓ | MuJoCo Succ ↑ | $E_{\text{mpjpe}}$ ↓ | $E_{\text{mpkpe}}$ ↓ |
|---|---|---|---|---|---|---|
| 3 | 0.90 | 0.23 | 0.21 | 0.63 | 0.30 | 0.28 |
| **4** | **0.93** | **0.18** | **0.16** | **0.67** | **0.26** | **0.24** |
| 5 | 0.91 | 0.22 | 0.18 | 0.66 | 0.30 | 0.27 |
| 6 | 0.92 | 0.21 | 0.18 | 0.67 | 0.27 | 0.24 |

**说明**: 4 专家最优（恰对应 3D 条件的 4 个嵌套子空间），更多专家无增益甚至略降，验证嵌套划分的合理性。

### Table 16 (paper Table 17): Condition Partition Method / 条件划分方式消融

| Method | IsaacGym Succ ↑ | $E_{\text{mpjpe}}$ ↓ | $E_{\text{mpkpe}}$ ↓ | MuJoCo Succ ↑ | $E_{\text{mpjpe}}$ ↓ | $E_{\text{mpkpe}}$ ↓ |
|---|---|---|---|---|---|---|
| Random | 0.93 | 0.19 | 0.16 | 0.67 | 0.26 | 0.25 |
| **Ours (nested)** | 0.93 | **0.18** | 0.16 | 0.67 | 0.26 | **0.24** |

**说明**: 嵌套划分相对随机划分仅有微弱误差优势——说明性能主要来自残差增量结构本身，对具体划分顺序不敏感（这点反而暴露 ΔMoE 设计的鲁棒性，也是一个可讨论的弱化项）。

---

## 实验

### 数据集 / 基准

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| [[BEAT2]] | 76 小时、30 位说话者 | mesh 表示 + 配对音频，语音→共语手势 | 训练/测试 |
| [[FineDance]] | 7.7 小时舞蹈 | SMPL-H 格式 + librosa 音乐特征，音乐→舞蹈 | 训练/测试 |

- 所有动作 30 FPS；原始序列切成 **10 秒片段**用于训练/评估。
- 动作过滤：用 CoM 与 CoP 地面投影距离 $\Delta d_t$ 判稳，首尾帧稳且最长连续不稳段 < 100 帧才保留。

### 实现细节

- **本体**: Unitree [[G1]]，输出 23 维目标关节位置。
- **教师**: ΔMoE（4 专家）+ [[PPO]]，Actor MLP [768,512,128]；输入含特权信息 + 参考运动跟踪目标。
- **学生**: 扩散策略（MLP 主干 [1792×3, 23] + [[AdaLN]] 注入条件）+ [[DAgger]] 蒸馏；$x_0$-prediction、DDIM($\eta$=0) 2 步采样、噪声尺度 $\beta_{\max}=0.20$。
- **对齐 adaptor**: 6 层 Transformer + 时序注意力，InfoNCE。
- **优化**: Adam，lr $1\times10^{-3}$，batch 8192。
- **训练/部署**: IsaacGym 训练 → 零样本迁移 MuJoCo（更真实动力学，作 sim-to-sim 中间验证）→ 真实 Unitree G1 部署；含域随机化与终止/惩罚双课程。

### 关键实验结论

- **对齐（Table 1）**: 音乐/语音→运动检索 R@1 约 65–67%，adaptor 有效注入运动学先验。
- **跟踪（Table 2）**: 在 BEAT2/FineDance、IsaacGym/MuJoCo 上成功率均超显式 baseline，舞蹈任务提升最明显。
- **核心消融**: adaptor（Table 5）与 content（Table 4）都不可或缺，去 adaptor 性能崩坏；ΔMoE 优于 vanilla MoE（Table 3）；$x_0$-prediction 远优于 $\epsilon$（Table 14）；4 专家 + DDIM 2 步最优。
- **节奏（Table 10）**: 把音频作外部 style 调制比直接拼接获得更高 BAS，动作更"踩点"。
- **泛化/freestyle（Fig 7）**: 面对未见音乐，扩散学生能完整执行而 MLP 立即摔倒。
- **效率（Table 11）**: DDIM-2 仅 ~5.3 ms，retargeting-free 设计实现低延迟。

---

## 批判性思考

### 优点
1. **范式新颖且动机扎实**: 首个把音频作为一等控制信号、retargeting-free 的人形 locomotion 框架；"motion=content+style" 解耦清晰，消融（Table 4/5/10）逐一验证 content、adaptor、style 注入的必要性。
2. **ΔMoE 有理论支撑**: 把 CFG 的条件残差对比推广到嵌套条件子空间，残差增量学习既有公式推导（公式2→5）又有 T-SNE（Fig 4）与消融（Table 3/15/16）佐证，不是空架结构。
3. **工程完整、可部署**: 教师特权/学生无特权的清晰分工、域随机化 + 双课程、sim→sim→real 三级验证、DDIM 2 步低延迟，真实 G1 上完成音乐/语音驱动（Fig 9–11）。

### 局限性
1. **缺与外部 SOTA 的横向对比**: 主表（Table 2）仅与自建"显式生成+重定向"baseline 比，没有与现有音频/语言驱动人形方法（如 RoboGhost、LangWBC 等）的同台量化对比，难以判断绝对先进性。
2. **数据/任务覆盖窄**: 仅 BEAT2（语音手势）+ FineDance（舞蹈）两数据集、各 10 秒片段；FineDance 真实成功率在 MuJoCo 仅 0.67，长程、复杂地形、强动态动作未覆盖。
3. **条件划分鲁棒性反成双刃剑**: Table 16 显示随机划分与嵌套划分几乎同分，说明"嵌套子空间"这一核心卖点对具体顺序并不敏感，ΔMoE 的增益更多来自残差结构而非"嵌套"本身，理论叙事与实证略有张力。
4. **content latent 被固定为常量**: 训练时所有动作共享同一 content latent，限制了"内容"层面的多样性，框架本质更接近"风格迁移"而非完整的内容生成。

### 潜在改进方向
1. 引入更强外部基线与更长/更动态的运动数据，量化绝对性能与泛化边界。
2. 让 content latent 随指令动态变化（而非固定常量），扩展到内容可控的多任务生成。
3. 把 ΔMoE 的"嵌套划分"与"残差结构"分离做消融，厘清增益来源；探索条件维度更高时的扩展性。
4. 报告真实世界的定量成功率/BAS（目前真实部分仅定性 Fig 9–11）。

### 可复现性评估
- [ ] 代码开源（论文未给出公开仓库/主页链接）
- [ ] 预训练模型（未声明 release）
- [x] 训练细节完整（附录 Table 6–10 给出状态、超参、域随机化、奖励）
- [x] 数据集可获取（BEAT2、FineDance 均为公开数据集）

---

## 速查卡片

> [!summary] RoboPerform: Expressive Humanoid Locomotion via Audio Control
> - **核心**: 首个 retargeting-free 的"音频→人形动作"框架，遵循 motion = content + style，音频作隐式风格信号。
> - **方法**: InfoNCE adaptor 对齐音频-运动 latent → ΔMoE（4 专家残差 MoE，CFG 的嵌套子空间推广）教师 RL 训练 → 扩散学生（content latent 引导去噪 + audio latent 逐层风格注入，$x_0$-prediction、DDIM 2 步）DAgger 蒸馏。
> - **结果**: BEAT2/FineDance 上跟踪成功率与 BAS 均超显式 baseline；真实 Unitree G1 完成音乐舞蹈与语音手势；DDIM-2 仅 5.3 ms；面对未见音乐可 freestyle。
> - **平台**: IsaacGym 训练 → MuJoCo 零样本 → 真实 G1 部署。

---

*笔记创建时间: 2026-06-29*
