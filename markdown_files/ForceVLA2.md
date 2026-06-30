---
title: "ForceVLA2: Unleashing Hybrid Force-Position Control with Force Awareness for Contact-Rich Manipulation"
method_name: "ForceVLA2"
authors: [Yang Li, Zhaxizhuoma, Hongru Jiang, Junjie Xia, Hongquan Zhang, Jinda Du, Yunsong Zhou, Jia Zeng, Ce Hao, Jieji Ren, Qiaojun Yu, Cewu Lu, Yu Qiao, Jiangmiao Pang]
year: 2026
venue: CVPR
tags: [VLA, force-control, hybrid-force-position, mixture-of-experts, flow-matching, contact-rich-manipulation]
zotero_collection: _inbox
image_source: online
arxiv_html: https://arxiv.org/html/2603.15169v1
created: 2026-06-29
---

# ForceVLA2: Unleashing Hybrid Force-Position Control with Force Awareness for Contact-Rich Manipulation

## 元信息

| 项目 | 内容 |
|------|------|
| 作者 | Yang Li, Zhaxizhuoma, Hongru Jiang, Junjie Xia, Hongquan Zhang 等 14 人 |
| 机构 | 上海交通大学、上海人工智能实验室（Shanghai AI Lab）等（论文署名含 Cewu Lu、Yu Qiao、Jiangmiao Pang） |
| 会议 | CVPR 2026 |
| 类别 | VLA（Vision-Language-Action）/ 力控操作 |
| 日期 | 2026（arXiv v1） |
| 项目主页 | https://sites.google.com/view/force-vla2/home |
| 链接 | [arXiv](https://arxiv.org/abs/2603.15169) / [HTML](https://arxiv.org/html/2603.15169v1) |

---

## 一句话总结

> 在 $\pi_0$ 式 VLA 上把"力"从被动感知升级为主动控制信号：VLM 端用[[Force Prompt|力提示]]构造分阶段的力感知任务概念，动作端用[[Cross-Scale MoE|跨尺度专家混合]]自适应融合实时交互力，输出闭环的[[Hybrid Force-Position Control|混合力-位控制]]动作，在 5 个接触密集任务上平均成功率 66%，比 $\pi_0$/$\pi_{0.5}$ 高 48%/35%。

---

## 核心贡献

1. **首个面向 VLA 的端到端力感知混合力-位控制框架**: 把[[Force Prompt|力提示]]注入 VLM 专家做长程推理，并通过[[Cross-Scale MoE|跨尺度 MoE]]把力融入动作专家，实现协调的混合力-位交互，把"VLM 推理"与"动作生成"紧密耦合，使力从辅助感知变为主动闭环控制信号。
2. **ForceVLA2-Dataset**: 构建包含 **1,000 条轨迹 / 5 个接触密集任务**（擦拭、按压、装配等）的数据集，提供多视角图像、任务提示、本体状态，并**额外加入力提示并把力纳入动作空间**——首个带力提示做任务分解、且唯一提供力控监督的数据集。
3. **显著性能提升**: 在 5 个任务上，相比 $\pi_0$ 与 $\pi_{0.5}$ 成功率分别提升 **48.0%** 与 **35.0%**，并缓解机械臂过载、接触不稳等常见失败模式。

---

## 问题背景

### 要解决的问题
如何让 [[VLA]] 模型在**接触密集（contact-rich）操作**中，不止"感知"到力，而是把力作为**主动、自适应、闭环**的控制变量，实现稳定、精确、鲁棒的混合力-位控制。

### 现有方法的局限
作者指出当前 VLA 在力上的两个层次缺陷：
1. **力被降格为辅助感知输入**: OpenVLA、GR00T-N1、$\pi_0$ 等在语义 grounding 与语言跟随上很强，但缺乏对物理动力学与细粒度接触交互的推理；即便 [[ForceVLA]] 把力作为一个模态接入 $\pi_0$，力仍只是"被动感知线索"而非主动控制信号。
2. **纯位置控制的本质局限**: 现有 VLA 基本是 EE 6D 位姿控制；正文附录给出**控制论证明**——力 $\mathbf{f}$ 不是独立控制变量，而是由未知环境动力学决定的因变量，导致可控性矩阵秩亏（$\le 6 < 12$），无法到达任意"力-位"组合。
3. **缺少跨阶段时空推理 + 子任务内的主动力-位交互机制**: 人类靠高层视觉/语言推理确定阶段目标，再用实时力觉细化力与位置；现有 VLA 两头都缺。
4. **数据集缺力监督**: OXE 仅子集含力、稀疏不一致；RH20T/REASSEMBLE 虽有力，但没有"力提示做任务分解 + 力控监督"。

### 本文的动机
- 受人类感觉运动控制中**力的层级化整合**（小脑/运动皮层的前向模型）启发，作者主张"力是贯穿 感知-规划-执行 层级的统一信号"；
- 因此设计**双层结构**：长程推理层用[[Force Prompt|力提示]]构造力感知任务概念（离散状态机式的子任务切换）；短程反应层用[[Cross-Scale MoE|跨尺度 MoE]]把 VLM 的力感知任务知识与实时交互力融合，做自适应的混合力-位调节。
- 关键工程直觉：**实时变化的力观测要"短路径、高保真"地直达动作专家**（绕过高层融合），而缓变的本体状态与力感知任务知识走多模态编码器——以保证快速力反馈的梯度保真度。

---

## 方法详解

### 模型架构

ForceVLA2 在 $\pi_0$ 框架上扩展，整体为 **VLM 专家（长程推理）+ 动作专家（短程反应）** 的双层流水线（见 Figure 2）：
- **输入**: 多视角 RGB 图像 $\{I^i\}$ + 任务提示 $\mathbf{T}_t$ + [[Force Prompt|力提示]] $\mathbf{T}_f$ + 本体状态（EE 6D 位姿 $\mathbf{p}\in\mathbb{R}^7$ 与力/力矩 $\mathbf{f}_{\text{raw}}\in\mathbb{R}^6$）
- **VLM 主干**: 基于 [[SigLIP]] 的视觉-语言模型（[[PaliGemma]]），把图像与"任务+力"提示编码为融合上下文 token $\mathbf{E}$
- **多尺度力注入**: ① 力作为**子任务级力提示**与图像在 VLM 中融合；② 力作为**力 token**与 EE 位姿在多模态编码器中结合；③ 一条 **bypass 旁路**让原始力信号绕过高层融合，直达 MoE 形成短程反应回路
- **核心模块**: [[Cross-Scale MoE|跨尺度专家混合]]（视觉/状态/力三专家）+ [[Flow Matching|流匹配]]动作头
- **输出**: 混合力-位动作 $\mathbf{a}_t=[\Delta\mathbf{p}_t;\mathbf{f}_t]\in\mathbb{R}^{13}$（位姿增量 7 维 + 力 6 维）以及子任务进度 $s_t\in[0,1]$

### 核心模块

#### 模块1: Force Prompt 驱动的 VLM 长程推理

**设计动机**: 接触密集场景里视觉常常模糊/不足（如航天连接器插接、齿轮装配），关键信息靠力反馈传递。纯视觉-语言输入不足以做需要"人类式任务感知 + 物理 grounding"的任务分解。

**具体实现**:
- 引入 [[Force Prompt|力提示]] 作为**文本线索**，指示当前子任务并编码阶段特定的物理上下文，从而构造**力感知任务概念**。
- 每个任务预定义子任务列表；力提示决定"保持当前子任务还是切换到下一个"，**功能上像一个离散状态机**。
- 视觉 token 经视觉编码器 $f(\cdot)$ 得 $\mathbf{Z}_v$；文本提示 $\mathbf{T}_t$ 与力提示 $\mathbf{T}_f$ 拼接后经文本编码器 $g(\cdot)$ 得 $\mathbf{Z}_l$；二者拼成 $\mathbf{E}_{\text{in}}=[\mathbf{Z}_v;\mathbf{Z}_l]$，经 VLM token 间注意力得融合表征 $\mathbf{E}$（公式1）。

#### 模块2: Proprioceptive-Force 编码与跨模态交互（多模态编码器）

**设计动机**: 纯 EE 6D 位姿控制对粗任务够用，但当细微力交互主导时失效。ForceVLA2 把本体状态与力**区别对待**：缓变量走融合、瞬变力走旁路。

**具体实现**:
- EE 6D 位姿 $\mathbf{p}\in\mathbb{R}^7$ 经线性层 $\phi_P$ 编码为 $\mathbf{E}_P$（公式2）；原始 6D 力/力矩 $\mathbf{f}_{\text{raw}}\in\mathbb{R}^6$ 经线性映射 $\phi_F$ 编码为 $\mathbf{E}_F$（公式3）。
- 状态表征 $\mathbf{E}_{\text{state}}=[\mathbf{E}_P;\mathbf{E}_F]$ 作为 query，与 VLM 上下文 $\mathbf{E}$（key-value）做**交叉注意力**注入全局任务语义，得 $\mathbf{E}'_{\text{state}}$（公式4）。
- 同时一条 $\mathbf{E}_F$ **bypass** 绕过跨模态交互，直送 MoE，构成**短程反应回路**（保留快速力反馈的梯度保真度，避免过度依赖历史轨迹，支持主动探索）。
- 拼接得条件序列 $\mathbf{E}_{\text{cond}}=[\mathbf{E};\mathbf{E}'_{\text{state}};\mathbf{E}_F]$（公式5）。

#### 模块3: Cross-Scale Mixture-of-Experts（跨尺度专家混合）

**设计动机**: 不同任务阶段主导模态不同——自由空间运动靠视觉推理，接触/柔顺阶段靠力或位置线索。需要在 token 级**动态选择最有信息量的模态**。

**具体实现**:
- 含三个**模态特定专家**（视觉 V / 状态 S / 力 F），每个为轻量 MLP；
- **动态门控网络**计算 token 级路由权重 $\mathbf{w}=[w_V,w_S,w_F]$，对每个 token 激活最相关专家，得融合表征 $\mathbf{E}_{\text{MoE}}$（公式6）；
- 网络自主在"自由运动期强调视觉、接触期强调力/位"间切换；$\mathbf{E}_{\text{MoE}}$ 投影到流匹配策略的隐空间做动作生成。

#### 模块4: Flow Matching 力-位策略头 与 子任务转移概率

**设计动机**: 用 [[Flow Matching|流匹配]]在多模态上下文条件下，把噪声动作逐步去噪成最终的力-位命令；同时显式建模"何时切换子任务"。

**具体实现**:
- 每步从噪声 $\mathbf{a}_t^{(0)}\sim\mathcal{N}(0,I)$ 出发，按学习到的条件流速度场 $F_\theta$ 迭代去噪（公式7），离散实现为公式8，收敛到 $\mathbf{a}_t=[\Delta\mathbf{p}_t;\mathbf{f}_t]$（公式9）。
- **子任务转移指示器** $s_t$ 是模型输出的 0–1 连续值，超过阈值则切到下一子任务并清零；训练用 $\hat s_t$ 作监督，由方向对齐 $\Theta$、剩余距离 $L$、瞬时接触力 $F$ 三个量的**联合事件概率**给出（公式10，假设 $\Theta\sim\text{Beta}(\alpha,1)$、$L\sim\text{Exp}(\lambda)$、$F\sim\text{Uniform}(n,m)$，取 $\alpha=2,\lambda=2$）。

### 关键公式与机制

#### 公式1: [[VLM]] 多模态融合

$$
\mathbf{E}=\texttt{VLM}(\mathbf{E}_{\text{in}})\in\mathbb{R}^{(N_v+N_t+N_f)\times D_{\text{model}}}
$$

**含义**: 视觉 token、文本提示 token、力提示 token 拼成 $\mathbf{E}_{\text{in}}$ 后，经 VLM token 间注意力得融合上下文表征 $\mathbf{E}$，供下游短程力感知与控制使用。

**符号说明**:
- $N_v, N_t, N_f$: 视觉 / 文本提示 / 力提示 token 数
- $D_{\text{model}}$: 模型隐藏维；$\mathbf{E}_{\text{in}}=[\mathbf{Z}_v;\mathbf{Z}_l]$ 为视觉与语言编码拼接

#### 公式2: EE 6D 位姿编码

$$
\mathbf{E}_P=\phi_P(\mathbf{p})\in\mathbb{R}^{D_{\text{model}}}
$$

**符号说明**:
- $\mathbf{p}\in\mathbb{R}^7$: EE 6D 位姿（位置 + 四元数）；$\phi_P$: 线性编码层

#### 公式3: 原始力/力矩编码

$$
\mathbf{E}_F=\phi_F(\mathbf{f}_{\text{raw}})\in\mathbb{R}^{D_{\text{model}}}
$$

**符号说明**:
- $\mathbf{f}_{\text{raw}}\in\mathbb{R}^6$: 6D 力-力矩原始读数；$\phi_F$: 线性映射，把力投影到模型嵌入空间

#### 公式4: 状态-上下文交叉注意力

$$
\mathbf{E}'_{\text{state}}=\texttt{CrossAttn}(\mathbf{E}_{\text{state}},\mathbf{E})
$$

**含义**: 以 VLM 视觉-语言序列 $\mathbf{E}$ 为 key-value、状态 token $\mathbf{E}_{\text{state}}=[\mathbf{E}_P;\mathbf{E}_F]$ 为 query，把全局任务语义注入本体/力的局部流，使模型在视觉-语言意图引导下解释力与运动信号。

#### 公式5: 条件序列拼接

$$
\mathbf{E}_{\text{cond}}=[\mathbf{E};\mathbf{E}'_{\text{state}};\mathbf{E}_F]
$$

**含义**: 把 VLM 上下文、经条件化的状态表征、以及**旁路直达的原始力嵌入** $\mathbf{E}_F$ 拼接，作为 MoE 路由阶段的输入。注意 $\mathbf{E}_F$ 同时出现在这里（bypass 短程反应回路）。

#### 公式6: [[Cross-Scale MoE|跨尺度 MoE]] 融合

$$
\mathbf{E}_{\text{MoE}}=\sum_{m\in\{V,S,F\}} w_m\cdot\texttt{Expert}_m(\mathbf{E}_{\text{cond}})
$$

**含义**: 门控权重 $w_m$ 对视觉/状态/力三个专家的输出加权求和，token 级地选出当前阶段主导模态。

**符号说明**:
- $w_m$: 门控网络输出的 token 级路由权重，$\mathbf{w}=[w_V,w_S,w_F]$
- $\texttt{Expert}_m$: 模态特定轻量 MLP

#### 公式7: 条件流匹配（连续）

$$
\frac{d\mathbf{a}_t^{(\tau)}}{d\tau}=F_\theta\big(\mathbf{a}_t^{(\tau)},\mathbf{E}_{\text{MoE}},\tau\big),\quad \tau\in[0,1]
$$

**含义**: 学习一个去噪时间 $\tau$ 上的速度场 $F_\theta$，在 MoE 融合上下文条件下把噪声动作推向真值。

#### 公式8: 离散去噪迭代

$$
\mathbf{a}_t^{(\tau+1)}=\mathbf{a}_t^{(\tau)}+\Delta_\tau\cdot F_\theta\big(\mathbf{a}_t^{(\tau)},\mathbf{s}_t,\mathbf{E}_{\text{MoE}}\big)
$$

**含义**: 连续流的离散实现，从随机噪声出发逐步收敛到最终力-位命令。

**符号说明**:
- $\Delta_\tau$: 去噪步长；$\mathbf{s}_t$: 本体状态条件

#### 公式9: 混合力-位动作输出

$$
\mathbf{a}_t=\mathbf{a}_t^{(1)}=[\Delta\mathbf{p}_t;\mathbf{f}_t],\quad \Delta\mathbf{p}_t\in\mathbb{R}^7,\ \mathbf{f}_t\in\mathbb{R}^6,\ s_t\in[0,1]
$$

**含义**: 去噪终点即输出动作：EE 位姿增量 $\Delta\mathbf{p}_t$ 与预测接触力 $\mathbf{f}_t$，配合子任务进度 $s_t$。这正是"混合力-位"的体现。

#### 公式10: 子任务转移概率（真值监督）

$$
\begin{aligned}
\hat{s}_t &= P(\Theta\le\theta,\ L\ge l,\ F\le f)\\
&=\int_0^\theta \frac{\Gamma(\alpha+1)}{\Gamma(\alpha)}\,\theta'^{\,\alpha-1}d\theta' \cdot \int_l^\infty \lambda e^{-\lambda\ell}\,d\ell \cdot \int_n^f \frac{1}{m-n}\,df\\
&=\frac{\Gamma(\alpha+1)}{\alpha\,\Gamma(\alpha)}\,\theta^\alpha e^{-\lambda l}\,\frac{f-n}{m-n}
\end{aligned}
$$

**含义**: 把"是否该切换子任务"建模为方向对齐 $\Theta$、剩余距离 $L$、接触力 $F$ 三个观测量同时满足进度条件的**联合概率**，作为 $s_t$ 的训练真值 $\hat s_t$。

**符号说明**:
- $\Theta=\tfrac12\!\left(\tfrac{\vec E\cdot\vec{E_t}}{\|\vec E\|\|\vec{E_t}\|}+1\right)\in(0,1)$: 朝向对齐度（1 对齐 / 0 相反）
- $L=\|\overrightarrow{AA_t}\|$: 到目标的剩余距离；$F=\|\mathbf{f}\|\in[n,m]$: 瞬时接触力（归一化界 $n,m$）
- $\Gamma(\cdot)$: Gamma 函数；超参 $\alpha=2,\lambda=2$

#### 公式11–17: 混合力-位控制的控制论分析（附录 A）

附录用控制论解释"为何旁路反应力路径有效、为何纯位置控制不够"：

$$
\mathbf{f}=\bm{\Phi}(\mathbf{p_e},\dot{\mathbf{p}}_e,\bm{\theta}_e) \tag{11}
$$

**含义**: 环境本构律——力 $\mathbf{f}$ 是位姿 $\mathbf{p}_e$、速度 $\dot{\mathbf{p}}_e$ 与未知环境参数 $\bm{\theta}_e$ 的未知非线性映射 $\bm{\Phi}:\mathbb{R}^{12}\times\Theta_e\to\mathbb{R}^6$。

$$
\mathbf{p}(k+1)=\mathbf{a_p}(k),\qquad \mathbf{f}(k+1)=\bm{\Phi}(\mathbf{a_p}(k),\dot{\mathbf{a}}_p(k),\bm{\theta}_e) \tag{12}
$$

**含义**: 纯位置控制（带力输入）下的系统动力学——位置可直接指定，但力只能被动地由 $\bm{\Phi}$ 决定。

$$
\mathcal{C}=[\mathbf{B}\quad\mathbf{A}\mathbf{B}\quad\cdots\quad\mathbf{A}^{11}\mathbf{B}] \tag{13}
$$

$$
\mathbf{B}=\begin{bmatrix}\mathbf{I}_6\\\mathbf{0}_{6\times 6}\end{bmatrix} \tag{14}
$$

$$
\text{rank}(\mathcal{C})\le\text{rank}(\mathbf{B})=6<12=\text{dim}(\mathbf{z}) \tag{15}
$$

**含义**: 增广状态 $\mathbf{z}=[\mathbf{p}_e^T,\mathbf{f}^T]^T\in\mathbb{R}^{12}$ 的可控性矩阵 $\mathcal{C}$ 秩 $\le 6$，**秩亏**——力不是独立控制变量，纯位置控制无法到达任意力-位组合。

$$
\mathcal{R}=\{(\mathbf{p}_e,\mathbf{f}):\mathbf{f}=\bm{\Phi}(\mathbf{p}_e,\dot{\mathbf{p}}_e,\bm{\theta}_e),\ \mathbf{p}_e\in\mathcal{P}\} \tag{16}
$$

**含义**: 纯位置控制的可达集 $\mathcal{R}$ 被约束为嵌在 12 维任务空间中的 **6 维流形**。

$$
\kappa=\frac{\text{dim}(\mathcal{R})}{\text{dim}(\mathcal{Z})} \tag{17}
$$

**含义**: 定义**有效可控性指数** $\kappa$。纯位置控制 $\kappa=6/12=0.5$；引入混合力-位控制（$\mathbf{a}=[\mathbf{a}_p^T,\mathbf{f}_t^T]^T\in\mathbb{R}^{12}$）后策略 $\pi_\theta$ 隐式学到环境逆模型，使 $\kappa\to\kappa_{\max}$，扩展可控子空间。

---

## 关键图表

<!-- 图片均使用 arXiv HTML 在线链接（base = https://arxiv.org/html/2603.15169v1/），未本地化 -->

### Figure 1: ForceVLA2 Concept / 核心理念

![Figure 1](https://arxiv.org/html/2603.15169v1/x1.png)

**说明**: 接触密集操作需要的不只是视觉与状态观测，还需要**力调节**（左）。ForceVLA2 在多个尺度整合力信息，对接触动力学做丰富建模：通过输入的力信号把"力感知"建进任务规划，并输出带**动态平衡**的混合力-位动作（右）。这张图点明了本文从"位置控制"到"力-位混合控制"的核心转变。

### Figure 2: Framework of ForceVLA2 / 整体框架

![Figure 2](https://arxiv.org/html/2603.15169v1/x2.png)

**说明**: ForceVLA2 以多视角图像、任务提示+力提示、本体状态（EE 位姿与力）为输入。**力在多尺度注入**：作为子任务提示在 VLM 中与图像融合；作为力 token 在多模态编码器中与 EE 位姿结合；并有一条 **bypass 旁路保留原始信号**直送 MoE。跨尺度 MoE 整合三模态，产出混合力-位动作并跟踪子任务进度。这是理解整个方法管线的核心图。

### Figure 3: ForceVLA2-Dataset Illustration / 数据集示意

![Figure 3](https://arxiv.org/html/2603.15169v1/x3.png)

**说明**: (a) ForceVLA2-Dataset 是首个带**力提示做任务分解**、且唯一提供**力控监督**的数据集；(b) 覆盖 5 个接触密集任务、共 1,000 条示范。佐证贡献2"力控监督"的独特性。

### Figure 4: Data Collection System / 采集系统

![Figure 4](https://arxiv.org/html/2603.15169v1/x4.png)

**说明**: 用 [[GELLO]]（力反馈遥操作）手动驱动 [[Flexiv]] 机械臂完成灵巧任务，同步记录图像、力以及机器人位姿。说明数据为何能带"自然运动动力学 + 力轨迹"。

### Figure 5: Qualitative Results (vs ForceVLA & π series) / 定性对比

![Figure 5](https://arxiv.org/html/2603.15169v1/x5.png)

**说明**: 与 ForceVLA 和 $\pi$ 系列对比，ForceVLA2 以更高成功率、更快执行完成典型任务，并**避免机械臂过载**，展现更好的柔顺性。直观呈现"主动力调节避免 overload"这一卖点。

### Figure 6: Following & Re-targeting Tests / 跟随与重定位

![Figure 6](https://arxiv.org/html/2603.15169v1/x6.png)

**说明**: 额外的位置/朝向**跟随**与**重定位**测试。ForceVLA2 表现出鲁棒的位姿跟随；在物体搜索任务里即便视觉观测失效，仍能成功**重抓取（re-grasp）**。支撑"快速力跟随、视觉失效仍可探索"的论点。

### Figure 7: Dataset Statistics / 数据集统计

![Figure 7](https://arxiv.org/html/2603.15169v1/x7.png)

**说明**: (a) 力分量、(b) 力矩分量在 $x,y,z$ 维度的频率分布；(c) 基于力与位置的短程反应技能示意；(d) 技能分布。例如 Press bottle 力集中在 $-z$；Assemble gears 力/力矩分布最广（对应对齐与插入）。技能定义：**Wipe**（位移 >0.05m 且力 >10N）、**Push**（$z$ 位移 >0.05m 且 $z$ 力 >5N）、**Grasp**（$z$ 位移 >0.1m 且力 >5N）、**Rotate**（三轴力变化 >1N）、其余为 **Explore**；其中 Explore 占 45.62% 居首。

### Figure 8: Force-Injection Ablation in Multimodal Encoder / 力注入位置消融

![Figure 8](https://arxiv.org/html/2603.15169v1/x8.png)

**说明**: 力分支的三个候选插入点：(1) 注入 VLM 通路、(2) 注入多模态编码器、(3) 在 (2) 基础上再与状态融合。对应附录 Table 5——注入 VLM 通路会**严重掉点**，注入多模态编码器或状态融合分支则稳定提升，故最终方案同时在多模态编码器与状态融合两级注入力。

### Table 1: Success Rates Comparison / 主结果（成功率 %，每任务 20 次试验）

| Type | Method | Press bottle | Clean vase | Clean board | Retri. plate | Assem. gears | Avg. |
|------|--------|-------------|-----------|------------|-------------|--------------|------|
| w/o Force | $\pi_0$ | 35.0 | 20.0 | 35.0 | 0.0 | 0.0 | 18.0 |
| w/o Force | $\pi_{0.5}$ | 45.0 | 30.0 | 45.0 | 15.0 | 20.0 | 31.0 |
| w/ Force | ACP | 25.0 | 30.0 | 25.0 | 0.0 | 0.0 | 16.0 |
| w/ Force | $\pi_0$ w/ F | 30.0 | 25.0 | 20.0 | 10.0 | 0.0 | 17.0 |
| w/ Force | ForceVLA | 70.0 | 25.0 | 55.0 | 15.0 | 10.0 | 35.0 |
| w/ Force | **ForceVLA2 (Ours)** | **80.0** | **75.0** | **70.0** | **35.0** | **70.0** ↑50 | **66.0** |

**说明**: ForceVLA2 平均 66%，全面领先；在最难的 **Assemble gears** 上比次优高 **50 个百分点**。值得注意：$\pi_0$ w/ F（朴素拼接力）仅 17%，与无力版相近——印证"简单拼接力反而像噪声、破坏预训练表征"。ACP（导纳控制式虚拟动作）仅 16%，泛化差。"高 48%/35%"指相对 $\pi_0$(18%)/$\pi_{0.5}$(31%) 的绝对差距（结论里写作 48%/35%/31%，对应 $\pi_0$/$\pi_{0.5}$/ForceVLA）。

### Table 2: Ablation of Modules (on top of π₀) / 模块逐步消融

| FP | ME | CM | Press bottle | Clean vase | Clean board | Retri. plate | Assem. gears | Avg. |
|----|----|----|-------------|-----------|------------|-------------|--------------|------|
| – | – | – | 35.0 | 20.0 | 35.0 | 0.0 | 0.0 | 18.0（baseline） |
| ✓ | – | – | 60.0 | 25.0 | 40.0 | 5.0 | 5.0 | 27.0 ↑9 |
| ✓ | ✓ | – | 60.0 | 40.0 | 65.0 | 5.0 | 30.0 | 40.0 ↑13 |
| ✓ | ✓ | ✓ | **80.0** | **75.0** | **70.0** | **35.0** | **70.0** | **66.0** ↑26 |

**说明**: 在 $\pi_0$ 上逐步加 **FP（力提示）→ ME（多模态编码器）→ CM（跨尺度 MoE）**，成功率单调上升（18→27→40→66）。其中 **Cross-Scale MoE 贡献最大的单项增益**（+26 累计中由 CM 带来 40→66 的 +26）。FP=Force Prompt，ME=Multimodal Encoder，CM=Cross-Scale MoE。

### Table 3: Ablation of MoE Input Modalities / MoE 输入模态消融

| VM | FM | Press bottle | Clean vase | Clean board | Retri. plate | Assem. gears | Avg. |
|----|----|-------------|-----------|------------|-------------|--------------|------|
| ✓ | – | 70.0 | 30.0 | 60.0 | 0.0 | 20.0 | 36.0 |
| – | ✓ | 85.0 | 40.0 | 85.0 | 0.0 | 40.0 | 50.0 ↑14 |
| ✓ | ✓ | 80.0 | 75.0 | 70.0 | 35.0 | 70.0 | **66.0** ↑16 |

**说明**: VM（视觉模态）与 FM（力模态）在 MoE 中**贡献大致相当**，二者缺一不可，全开最优（66%）。有趣的是 Press bottle 与 Clean board 在加入额外力 token 后**略微掉点**（80/70 vs 仅力时 85/85），作者推测这类任务力需求简单（只需位置/力之间一次切换），多余力 token 引入了扰动自由度。

### Table 4: Training Details / 训练超参（附录 B）

| Parameter | Value |
|-----------|-------|
| Learning Rate Schedule | Cosine Decay |
| Optimizer | AdamW |
| EMA Decay | 0.99 |
| Random Seed | 42 |
| Batch Size | 32 |
| Training Steps | 30,000 |

**说明**: 在 **8×A100** 上以 batch 32、3 万步训练（约 10 小时）；在 4090 上以 chunk size 30 达到 **15 Hz** 推理速度。（注：附录文字与表标题处出现 "Foca-VLA/Foca-Dataset" 字样，应为 ForceVLA2/ForceVLA2-Dataset 的撰稿残留笔误。）

### Table 5: Force Injection Location Ablation / 力注入位置（附录 C，对应 Figure 8）

| Method | Press bottle | Clean vase | Clean board | Retri. plate | Assem. gears | Avg. |
|--------|-------------|-----------|------------|-------------|--------------|------|
| VLM Pathway | 10.0 | 10.0 | 5.0 | 0.0 | 0.0 | 5.0 |
| ME | 75.0 | 55.0 | 65.0 | 40.0 | 55.0 | 58.0 |
| **State Fusion** | **80.0** | **75.0** | **70.0** | 35.0 | **70.0** | **66.0** |

**说明**: 把力注入 **VLM 通路**会灾难性掉点（仅 5%），注入**多模态编码器（ME）**得 58%，在 ME 基础上**再与状态融合（State Fusion）**最优（66%）。这解释了为何最终设计在多模态编码器 + 状态融合两级注入力，而**不**让力污染 VLM 通路。

---

## 实验

### 数据集 / 基准

| 基准 | 规模 | 特点 | 用途 |
|------|------|------|------|
| ForceVLA2-Dataset | 1,000 轨迹 / ≈500K 同步时间步 / 5 任务，每任务采集至完成 | 7-DOF Flexiv Rizon 4s + AG-95 夹爪；3 路 RGB（2 第三人称 D455 1280×720@30、1 腕部 D435 640×480@30）；EE 端 6D 力/力矩 @300Hz；每任务切 3–5 子任务（按力信号离线标注） | 训练 + 评测 |

5 个任务：**Press bottle、Clean vase、Clean board、Retrieve plate、Assemble gears**；主指标为成功率（%），每任务 20 次独立试验。作者强调**只做真机评测**——力交互对摩擦/接触建模敏感，仿真不可靠，且无现成力感知 VLA 基准。

### 实现细节

- **Backbone**: 基于 [[SigLIP]] 的 VLM（[[PaliGemma]]），$\pi_0$ 式流匹配动作专家
- **力注入**: 多尺度（力提示 @VLM + 力 token @多模态编码器 + 原始力 bypass @MoE）
- **MoE**: 视觉/状态/力三专家（轻量 MLP）+ 动态门控
- **动作空间**: $[\Delta\mathbf{p}_t\in\mathbb{R}^7;\mathbf{f}_t\in\mathbb{R}^6]$ + 子任务进度 $s_t$
- **优化**: AdamW、Cosine Decay、EMA 0.99、batch 32、30k 步、seed 42
- **硬件**: 8×A100 训练（≈10h）；推理 4090、chunk 30、15 Hz
- **部署**: Jacobian 映射做硬件无关部署，兼容力矩控制（Franka）、带 EE F/T 传感器（UR）、力矩接口作动器（Feetech）

### 关键实验结论

- **Q1 主结果（Table 1）**: ForceVLA2 平均 66%，比 $\pi_0$(18%)/$\pi_{0.5}$(31%)/ForceVLA(35%) 全面领先；在 Assemble gears 上比次优高 50 个百分点。朴素拼接力（$\pi_0$ w/ F，17%）几乎无增益，说明"主动力控 ≠ 把力当输入塞进去"。
- **过载避免（Fig 5）**: Clean board、Press bottle 等力密集任务中，ForceVLA2 主动调节交互力完成任务，其他 VLA 因缺反应式调节而过载失败。
- **突发扰动力跟踪（Fig 6）**: 压瓶时突然降低底座，ForceVLA2 凭力感知快速适应新接触配置并完成；其他 VLA 只能慢慢追位姿而失败。
- **重抓取/重定位（Fig 6）**: 擦花瓶时表面法向偏离原运动方向，ForceVLA2 沿瓶面滑动上移保持接触；取盘任务中力引导探测、首次抓取失败后自主重试。
- **Q2 模块消融（Table 2）**: FP→ME→CM 单调提升，**CM 单项增益最大**。
- **Q3 模态融合（Table 3）**: VM 与 FM 贡献相当、缺一不可；简单力任务上多余力 token 反而略降。
- **力注入位置（Table 5/Fig 8）**: 注入 VLM 通路灾难（5%），多模态编码器 + 状态融合最优（66%）。

---

## 批判性思考

### 优点
1. **把"力"从感知升级到控制的范式较清晰且有理论支撑**: 附录用可控性秩亏（公式13–15）论证纯位置控制的本质局限，并以有效可控性指数 $\kappa$（0.5→更高）量化混合力-位控制的收益，主张不止于经验。
2. **多尺度力注入 + bypass 短程反应回路设计巧妙**: "缓变量走融合、瞬变力走旁路"既保住高层语义指导，又保证快速力反馈梯度保真度；Table 5 的"注入 VLM 通路灾难掉点"为该设计提供了反面证据。
3. **数据集补位明确**: 首个带力提示任务分解 + 力控监督的数据集，配 GELLO 力反馈遥操作采集，力轨迹 300Hz，子任务按力信号标注，复现/扩展价值高。

### 局限性
1. **评测规模偏小、绝对值仍有限**: 仅 5 个自采任务、每任务 20 次试验，无公开基准对比；Retrieve plate 仅 35%、整体 66% 距实用尚远。作者自述"无现成力感知 VLA 基准"，但也使外部可比性弱。
2. **撰稿残留笔误较多**: 附录大量出现 "Foca-VLA / Foca-Dataset"（Table 4 标题"Foca-VLA"、正文多处），与正文 ForceVLA2 不一致，疑似改名残留；动作维度叙述上正文写 $\Delta\mathbf{p}_t\in\mathbb{R}^7$ 而控制论部分用 $\mathbf{a}_p\in\mathbb{R}^6$，位姿维度（7 含四元数 vs 6 含旋转）需读者自行对齐。
3. **子任务转移概率建模的分布假设偏强**: $\Theta\sim\text{Beta}$、$L\sim\text{Exp}$、$F\sim\text{Uniform}$ 且固定 $\alpha=2,\lambda=2$，缺敏感度分析；$s_t$ 超过 1 才切换但定义域写作 $[0,1]$，阈值语义略含糊。
4. **MoE 路由可解释性是叙述性的**: "自由运动靠视觉、接触靠力"的切换主要靠定性描述与定性图（Fig 5/6），缺门控权重随阶段变化的定量可视化。

### 潜在改进方向
1. 扩充任务数与试验次数，并推动/接入公开力感知基准（如 RH20T、REASSEMBLE）做跨数据集对比，验证泛化与可比性。
2. 统一命名与维度表述（修掉 Foca-VLA 残留），给出 $s_t$ 阈值与分布超参的消融。
3. 量化 MoE 门控权重在"自由运动/接触/插入"各阶段的分布，把"按阶段选模态"从定性变定量。
4. 探索把多尺度力注入迁移到更大主干或 dual-system（如 $\pi_{0.5}$/GR00T）框架，验证范式可移植性。

### 可复现性评估
- [ ] 代码开源（论文未明确给出代码链接，仅项目主页 sites.google.com/view/force-vla2）
- [ ] 预训练模型（未声明 release 权重）
- [x] 训练细节较完整（附录给出超参表 4、硬件 8×A100/30k 步、推理 15Hz）
- [ ] 数据集可获取（声明构建了 ForceVLA2-Dataset，但未明确公开发布渠道）

---

## 速查卡片

> [!summary] ForceVLA2: Hybrid Force-Position Control with Force Awareness
> - **核心**: 把"力"从被动感知升级为主动闭环控制——VLM 端力提示构造分阶段力感知任务概念，动作端跨尺度 MoE 融合实时力，输出混合力-位动作。
> - **方法**: SigLIP/PaliGemma VLM 主干 + 多尺度力注入（力提示@VLM、力token@多模态编码器、原始力 bypass@MoE）+ 视觉/状态/力三专家 Cross-Scale MoE + Flow-Matching 输出 $[\Delta\mathbf{p};\mathbf{f}]$ 与子任务进度 $s_t$；附录用可控性秩亏论证纯位控局限。
> - **结果**: 5 个接触密集任务平均 66%，比 $\pi_0$/$\pi_{0.5}$/ForceVLA 高 48%/35%/31%（绝对差），Assemble gears 超次优 50pt；8×A100 训 10h，4090 推理 15Hz。
> - **数据**: ForceVLA2-Dataset，1,000 轨迹 / 5 任务，首个带力提示分解 + 力控监督。
> - **主页**: https://sites.google.com/view/force-vla2/home

---

*笔记创建时间: 2026-06-29*
