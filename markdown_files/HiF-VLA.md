---
title: "HiF-VLA: Hindsight, Insight and Foresight through Motion Representation for Vision-Language-Action Models"
method_name: "HiF-VLA"
authors: [Minghui Lin, Pengxiang Ding, Shu Wang, Zifeng Zhuang, Yang Liu, Xinyang Tong, Wenxuan Song, Shangke Lyu, Siteng Huang, Donglin Wang]
year: 2026
venue: CVPR
tags: [VLA, world-model, motion-vector, temporal-modeling, long-horizon-manipulation, foresight-reasoning]
zotero_collection: _inbox
image_source: online
arxiv_html: https://arxiv.org/html/2512.09928v2
created: 2026-06-29
---

# HiF-VLA: Hindsight, Insight and Foresight through Motion Representation for Vision-Language-Action Models

## 元信息

| 项目 | 内容 |
|------|------|
| 作者 | Minghui Lin, Pengxiang Ding, Shu Wang, Zifeng Zhuang, Yang Liu, Xinyang Tong, Wenxuan Song, Shangke Lyu, Siteng Huang, Donglin Wang |
| 机构 | 西湖大学（MiLAB，Donglin Wang 课题组）等 |
| 会议 | CVPR 2026 |
| 类别 | VLA（Vision-Language-Action） |
| 日期 | 2025-12（arXiv v2） |
| 项目主页 | （论文未提供公开主页） |
| 链接 | [arXiv](https://arxiv.org/abs/2512.09928) / [PDF](https://arxiv.org/pdf/2512.09928) |

---

## 一句话总结

> 用视频编解码里的[[Motion Vector|运动向量]]当作历史与未来的紧致时序表征，让 VLA 同时拥有"回望（hindsight）—洞察（insight）—前瞻（foresight）"三重时间感受野，以近乎零额外推理开销刷新 LIBERO-Long 与 CALVIN ABC-D 等长程操作基准。

---

## 核心贡献

1. **运动即时序记忆，双向扩展时间感受野**: 提出 [[HiF-VLA]]，把低维结构化的 [[Motion Vector|运动向量（MV）]] 作为时序原语，既编码过去动态（hindsight）又预测未来动态（foresight），在显式扩展 VLA 时间感受野的同时大幅削减帧堆叠带来的像素冗余与推理延迟。
2. **Hindsight-Modulated Joint Expert（回望调制联合专家）**: 提出一个把运动流与动作流放进同一潜空间联合建模、并用历史运动作为自适应条件先验（[[AdaLN]] 调制）的专家解码器，实现"边想边做（think-while-acting）"的因果一致长程动作生成。
3. **强基准提升 + 近零额外开销 + 真机验证**: 在 LIBERO-Long、CALVIN ABC-D 上超过强基线，额外推理延迟仅约 0.13×（相对 subgoal 法的 1.59× 与帧堆叠的 3.15×），并在 AgileX Piper 真机上的多步长程任务显著优于 OpenVLA-OFT。

---

## 问题背景

### 要解决的问题

现有 [[VLA]] 大多隐式假设 **[[Markov Property|马尔可夫性质]]**——只用当前观测 $o_t$ 预测动作，缺乏对时序依赖的显式建模。作者称之为 **temporal myopia（时间近视）**：在长程操作中，缺少跨视觉/语言/运动模态的时序连续性会让连续动作之间的依赖退化，导致轨迹碎片化、任务级一致性变差。

### 现有方法的局限

作者把缓解时间近视的现有路线分为两类，并指出各自缺陷（见 Figure 1a）：

1. **堆叠历史帧**（TraceVLA、Octo、GR-2、RoboVLMs 等）：直接把多帧原始观测拼接。问题是计算开销大、推理延迟高（见 Table 3，8 帧时延迟达基线 4.5× 以上），且相邻帧高度相似带来**大量像素级冗余**，淹没了真正与任务相关的动态，分散模型注意力。
2. **预测像素级未来子目标**（CoT-VLA、Seer、UniVLA、UP-VLA 等）：生成未来视觉子目标再用逆动力学模型（IDM）推动作。问题是**像素级预测易产生局部失真与语义漂移**，子目标稠密冗余，且离散帧预测缺乏连续时序建模，复杂场景下时序一致性差。

两类方法都只做**单向**时序推理，且依赖稠密像素表征。

### 本文的动机

- **运动是历史的精确而高效的表征**：历史的关键不是过去的原始视觉内容，而是状态之间发生的"运动"。运动是时序记忆与环境动态的紧致代理，能忠实捕捉交互动态（物体被移动、抽屉被关上），同时丢弃冗余静态信息。
- **稳健决策需要双向时序推理**：智能体既要有 **hindsight**（理解导致当前状态的近期动态、把决策锚定在已验证的过去结果上），又要有 **foresight**（预判合理的未来动态，实现主动的目标导向行为）。这种对过去与未来动态的双向推理正是 **[[World Model|世界模型]]** 的核心。
- **运动是统一过去与未来的天然桥梁**：因此提出 HiF-VLA，用运动把"回望—洞察—前瞻"串成一个面向控制的、运动中心的 **World Action Model**。

---

## 方法详解

### 模型架构

HiF-VLA 建立在一个 vanilla VLA 之上（见 Figure 2），由三个组件串成一条流水线：

- **输入**: 当前观测 $o_t$ + 语言指令 $l$ + 长度为 $h$ 的历史运动先验 $m^{his}_{t-h:t}$
- **Backbone**: [[Prismatic|Prismatic-7B]] VLM，用 [[OpenVLA]]（在 [[OXE]] 上预训练）的权重初始化；视觉端用 [[DINOv2]] + [[SigLIP]] 混合编码器
- **核心模块**: (a) [[Hindsight Prior Acquisition|回望先验获取]] → (b) [[Foresight Reasoning|前瞻推理（带洞察）]] → (c) [[Hindsight-Modulated Joint Expert|回望调制联合专家]]
- **输出**: 未来 $n$ 步的动作块 $\tilde a_{t:t+n}$ 与未来运动 $\tilde m_{t:t+n}$（推理时运动解码可选、可省）
- **训练设置**: 动作与前瞻的时序块长固定 $n=8$；hindsight 窗口可变（默认 8）

整体推理映射（公式2）把 vanilla VLA（公式1）扩展为带历史先验、且联合预测动作与运动的形式。

### 核心模块

#### 模块1: Hindsight Prior Acquisition（回望先验获取）

**设计动机**: 仅靠瞬时观测在遮挡、重复执行等场景下感知不稳，需要从历史状态捕捉机械臂一致的动态模式；但帧堆叠冗余大、分散注意力。

**具体实现**:
- 借用视频编解码标准（[[H.264]]、[[MPEG-4]]）中的 **[[Motion Vector|运动向量（MV）]]**：MV 预测相邻帧间宏块（macroblock）的位移，避免逐像素冗余。关键论点是 MV **不是粗糙近似**——配合关键帧可近无损重建视频并保持高压缩，因此是捕捉历史动态的天然、高效且忠实的方案。
- 把当前观测 $o_t$ 当作关键帧，维护长度 $m$ 的历史窗口，组成 GOP（Group of Pictures）单元：$GOP=[MV_{t-m:t-m+1},...,MV_{t-1:t},o_t]$。MV 按 MPEG-4 的 $16\times16$ 宏块布局组织，表示为 $h\times(H//16)\times(W//16)\times 2$ 的张量，紧致编码场景中实体的历史运动轨迹。
- 用一个轻量 **[[ViT]]-based hindsight 编码器 + 浅层 3D 卷积**把回望运动编码为紧致回望 token $M_h\in\mathbb{R}^{K_h\times d}$。（附录：4 层 ViT，先用 3D 卷积切成 $(h//2)\times(H//2)\times(W//2)$ 时空块，加一个 `[CLS]` token 汇聚全局时序，再线性投影到联合专家空间。）

#### 模块2: Foresight Reasoning with Insight（带洞察的前瞻推理）

**设计动机**: 前瞻让智能体在执行前评估动作后果。但 CoT-VLA/UP-VLA 那种**像素级未来预测**易失真、语义漂移、缺连续时序建模。改用结构化 MV 作为未来执行的时空目标，紧致且能提供结构化空间先验。

**具体实现**:
- 在 VLM 嵌入空间中引入 $K_f$ 个可学习**前瞻查询 token** $\{q^f_1,...,q^f_{K_f}\}$ 与 $K_a$ 个**空动作 token** $\{q^a_1,...,q^a_{K_a}\}$。
- 把这些 token 与原始输入（指令 $l$、当前观测 $o_t$）拼接后送入 VLM $F_\theta$，**并行**推理连续视觉动态与动作生成，输出前瞻运动 token $M_f\in\mathbb{R}^{K_f\times d}$ 与动作隐 token $A_f\in\mathbb{R}^{K_a\times d}$，即 $(M_f,A_f)=F_\theta(o_t,l)$。
- 这种并行推理丰富了 VLM 内部"思考"的多样性，释放并行推理潜力（即"insight"——对当前任务与观测的洞察驱动前瞻）。

#### 模块3: Hindsight-Modulated Joint Expert（回望调制联合专家）

**设计动机**: 常规动作专家孤立地把 VLM 高层表征译成低级控制，缺乏对未来动态的推理。运动是动作在视觉空间的物理显现，联合预测运动与动作能让语义理解与底层动态更好对齐。

**具体实现**（见 Figure 2c 与 Figure 6）:
- 三类序列表征：回望运动 token $M_h$（**仅作条件输入**）、前瞻运动隐 token $M_f$、动作 token $A_f$。
- 前瞻运动流与动作流作为**两条并行流**，通过 **cross-stream joint attention（跨流联合注意力）**交互，但各自保留独立 FFN，确保表征互补又解耦。
- **关键设计——历史不进 VLM，而作条件**：把历史运动直接塞进 VLM 输入会破坏视觉-语言的预训练对齐（见 Figure 4 的消融）。因此 $M_h$ 经线性层投影成条件向量 $h_c$，通过 **[[AdaLN]]（Adaptive Layer Normalization）** 注入每个联合专家模块，对 $M_f$、$A_f$ 做分层调制与正则，约束未来"运动-动作"模式、抑制对历史轨迹的错误重放。
- Joint Attention 在拼接 $M_f$ 与 $A_f$ 的序列上做 **non-causal self-attention**，Q/K/V 联合投影。
- 融合后的运动与动作表征各自经独立 head 输出未来运动 $\tilde m_{t:t+n}$ 与动作 $\tilde a_{t:t+n}$。
- （附录：所有 token 投影到共享维度 1024，联合专家 6 层、每层支持跨流注意力，位置信息用 [[RoPE]]。）

### 关键公式与机制

#### 公式1: [[VLA]] 的 vanilla 推理（马尔可夫基线）

$$
\tilde{a}_{t:t+n}\sim P_{\theta}\big(a_{t:t+n}\mid o_{t},l\big)
$$

**含义**: 传统 VLA 仅由当前观测与指令预测动作块，无显式时序依赖。

**符号说明**:
- $o_t$: $t$ 时刻当前观测；$l$: 语言指令
- $\tilde a_{t:t+n}$: 预测的、长度为 $n$ 的动作块；$P_\theta$: VLA 策略

#### 公式2: HiF-VLA 的联合推理（带历史先验）

$$
(\tilde{a}_{t:t+n},\tilde{m}_{t:t+n})\sim P_{\theta}^{\prime}\big(a_{t:t+n},m_{t:t+n}\mid o_{t},l,m_{t-h:t}^{his}\big)
$$

**含义**: HiF-VLA 在 vanilla VLA 之上额外引入长度 $h$ 的历史运动先验 $m^{his}_{t-h:t}$，并**联合预测**未来动作与未来运动。训练时两者都预测；推理时运动解码可按需省略。

**符号说明**:
- $m^{his}_{t-h:t}$: 长度 $h$ 的历史运动（回望）先验
- $\tilde m_{t:t+n}$: 预测的未来 $n$ 步运动（前瞻）；$P'_\theta$: 扩展后的策略

#### 公式3: 运动向量定义（回望/前瞻的时序原语）

$$
MV_{t-1:t}(x,y)=(x_{t}-x_{t-1},\ y_{t}-y_{t-1})
$$

**含义**: 宏块在相邻帧 $o_{t-1}\to o_t$ 间的位移，即运动向量；它紧致地编码"帧间视觉变化"，因此既含机械臂运动，也含交互引发的物体位移、接触运动、场景结构变化。

**符号说明**:
- $(x,y)$: 图像中宏块位置（图像尺寸 $H\times W\times 3$）
- $(x_t,y_t),(x_{t-1},y_{t-1})$: 同一宏块在相邻两帧的位置

#### 公式4: 联合专家融合

$$
\tilde{M}_{f},\tilde{A}_{f}=\mathrm{Joint\,Expert}\big(M_{f},A_{f}\mid h_{c}\big)
$$

**含义**: 联合专家在历史条件 $h_c$ 下，对前瞻运动流 $M_f$ 与动作流 $A_f$ 做跨流交互，输出精炼后的运动/动作表征。

**符号说明**:
- $h_c$: 由回望 token $M_h$ 投影得到的条件向量
- $\tilde M_f,\tilde A_f$: 融合后的前瞻运动、动作表征

#### 公式5: AdaLN 条件调制

$$
\mathrm{AdaLN}(z;h_{c})=\gamma(h_{c})\cdot\frac{z-\mu(z)}{\sigma(z)}+\beta(h_{c})
$$

**含义**: 用历史条件 $h_c$ 生成的缩放/平移参数对前瞻运动或动作表征 $z$ 做自适应层归一化，把回望先验作为"残差式"旁路注入解码阶段，而不经过 VLM 语义融合层，避免破坏视觉-语言对齐。

**符号说明**:
- $z\in\{M_f,A_f\}$: 被调制的前瞻运动或动作表征
- $\mu(z),\sigma(z)$: $z$ 的均值与标准差
- $\gamma(h_c),\beta(h_c)$: 由条件 $h_c$ 生成的调制（缩放、平移）参数

#### 公式6: 动作与运动的 L1 损失

$$
\mathcal{L}_{MV}=\frac{1}{n}\sum_{j=1}^{n}\big|m_{t+j}-\tilde{m}_{t+j}\big|,\qquad
\mathcal{L}_{A}=\frac{1}{n}\sum_{j=1}^{n}\big|a_{t+j}-\tilde{a}_{t+j}\big|
$$

**含义**: 分别对未来 $n$ 步的运动预测与动作预测做 L1 回归，保证两者都校准良好。

**符号说明**:
- $m_{t+j},a_{t+j}$: 真值运动/动作；$\tilde m_{t+j},\tilde a_{t+j}$: 预测值
- $n$: 时序块长度（实验固定 8）

#### 公式7: 总损失

$$
\mathcal{L}_{all}=\mathcal{L}_{A}+\lambda\cdot\mathcal{L}_{MV}
$$

**含义**: 总损失为动作损失加上加权的运动重建损失；$\lambda$ 平衡动作精度与运动重建质量，**设为 0.01**（消融见 Table 5a：0.01 最优；过大反而破坏 VLA 稳定性）。

**符号说明**:
- $\lambda=0.01$: 运动分支的平衡系数

---

## 关键图表

<!-- 图片均使用 arXiv HTML 在线链接，未本地化 -->

### Figure 1: Motivation & Comparison / 动机与对比

![Figure 1](https://arxiv.org/html/2512.09928v2/x1.png)

**说明**: (a) 与现有方法对比：自上而下分别是仅用瞬时观测的 VLA、堆叠多帧历史、生成像素级子目标——分别有冗余、高推理成本、结构弱的问题；最底部的 HiF-VLA 联合建模 hindsight/insight/foresight，**双向**扩展时间感受野，做到紧致、结构化、高效。(c) HiF-VLA 降低推理延迟，并在 LIBERO-Long、CALVIN ABC-D 上达 SOTA，真机实验显著超基线。此图是全文论点的浓缩。

### Figure 2: HiF-VLA Pipeline / 整体流水线

![Figure 2](https://arxiv.org/html/2512.09928v2/x2.png)

**说明**: HiF-VLA 三段式流水线。(a) 回望先验获取：把稠密历史帧编码为紧致 MV 流，形成捕捉时序动态、无像素冗余的回望原语；(b) 带洞察的前瞻推理：VLM 解读指令与当前观测，推断合理的前瞻运动与对应的隐动作 token；(c) 回望调制联合专家：在统一潜空间融合回望、前瞻与动作表征，产生时序一致、因果连贯的动作预测。是理解方法的主图。

### Figure 3: Efficiency / Redundancy Analysis / 效率与冗余分析

![Figure 3a 历史帧示例](https://arxiv.org/html/2512.09928v2/x3.png)
![Figure 3b 推理效率](https://arxiv.org/html/2512.09928v2/x4.png)
![Figure 3c Hindsight 长度影响](https://arxiv.org/html/2512.09928v2/x5.png)

**说明**: (a) 历史帧示例，直观展示相邻帧的高相似度/冗余；(b) 推理效率——多帧基线的延迟随历史长度近似**线性**上升（历史长 8 时基线延迟超 vanilla VLA 4.5×），而 HiF-VLA 各上下文长度下延迟几乎不增，时序可扩展性强（回答 RQ3）；(c) hindsight 长度的影响——第三视角/多视角分别在长度 8 时达峰值 94.4%/96.4%。

### Figure 4: Hindsight Embedding Position / 历史嵌入位置消融

![Figure 4a 注入 VLM](https://arxiv.org/html/2512.09928v2/x6.png)
![Figure 4b 在解码器条件化](https://arxiv.org/html/2512.09928v2/x7.png)
![Figure 4c 结果](https://arxiv.org/html/2512.09928v2/x8.png)

**说明**: 对比两种历史嵌入策略：(a) 朴素地把 hindsight 与语言/视觉一起作为 VLM 直接输入；(b) 在专家模块内对 hindsight 做条件化（本文采用）；(c) 结果显示**专家条件化一致优于 VLM 嵌入**。原因是运动型 hindsight 会干扰预训练的视觉-语言对齐，而在解码阶段注入提供了一条残差式直接通路，让底层动态绕过 VLM 语义融合层去引导未来动作。这是"历史作条件、不进 VLM"这一关键设计的证据。

### Figure 5: Real-world Setup & Tasks / 真机平台与任务

![Figure 5](https://arxiv.org/html/2512.09928v2/x9.png)

**说明**: (a) 系统部署在 AgileX Piper 机械臂上，配 Intel RealSense D435 外部场景相机 + 腕部相机；(b) 设计三个长程任务，覆盖 pick/put/cover/stack/press 等多样原语，强调动作生成的时序一致性，并给出与 OpenVLA-OFT 的成功率对比。

### Figure 6: Hindsight-Modulated Joint Expert Architecture / 联合专家结构

![Figure 6](https://arxiv.org/html/2512.09928v2/x10.png)

**说明**: 回望调制联合专家的内部结构。前瞻运动流与动作流并行、经跨流联合注意力交互、各自保留 FFN；回望 token 经线性层成 $h_c$ 后通过 AdaLN 调制两条流。是模块3 的细化图。

### Figure 7: Foresight-Motion L1 Loss Convergence / 前瞻运动损失收敛曲线

![Figure 7](https://arxiv.org/html/2512.09928v2/x11.png)

**说明**: 训练中前瞻运动 L1 损失的收敛曲线。"w/o action prediction"是去掉动作预测分支、只训前瞻运动的变体；"w/ action prediction"是完整 HiF-VLA（联合专家中前瞻运动与动作两条流都保留）。结果显示**加上动作预测后运动收敛更快、曲线更稳**，说明动作分支为运动分支提供互补信息，二者协同——印证"think-while-acting"的有效耦合。

### Figure 8: Foresight Motion vs Action Rollouts / 前瞻运动与动作对齐可视化

![Figure 8](https://arxiv.org/html/2512.09928v2/x12.png)

**说明**: LIBERO-Long 三个任务的 rollout 示例，展示短推理窗口内**预测的前瞻运动与观测到的动作执行高度对齐**。定性佐证 HiF-VLA 保持时序连贯、真正实现"边想边做"——动作始终锚定在对未来状态的连贯预测上。

### Figure 9: Real-world Execution Visualization / 真机执行可视化（附录）

![Figure 9a Place blocks](https://arxiv.org/html/2512.09928v2/x13.png)
![Figure 9b Cover & stack](https://arxiv.org/html/2512.09928v2/x14.png)
![Figure 9c Press buttons](https://arxiv.org/html/2512.09928v2/x15.png)

**说明**: 三个真机长程任务在关键时间步的 rollout。(a) 把方块放到对应颜色托盘——基础视觉识别与精确放置；(b) 用绿碗盖住方块再叠粉碗——多步顺序推理、空间关系、长程依赖（cover→stack）；(c) 按指定顺序按三个彩色按钮——区分视觉相似状态、维持动作顺序、时间敏感决策。展示 HiF-VLA 在真机上的稳定行为规划。

### Figure 10: Real-world Failure Cases / 真机失败案例（附录）

![Figure 10a 放置失败](https://arxiv.org/html/2512.09928v2/x16.png)
![Figure 10b 堆叠失败](https://arxiv.org/html/2512.09928v2/x17.png)
![Figure 10c 按压失败](https://arxiv.org/html/2512.09928v2/x18.png)

**说明**: 三类失败：(a) 因错误空间判断过早张开夹爪导致放置失败；(b) 机械臂未把碗抬到合适高度导致堆叠失败；(c) 深度估计错误致夹爪下降不足、按压不完整。三者都指向**空间几何与 3D 感知**的不足，提示未来引入更丰富的 3D 表征。

### Table 1: LIBERO-Long Per-task Results / LIBERO-Long 逐任务结果

| Method | Avg. SR | T1 | T2 | T3 | T4 | T5 | T6 | T7 | T8 | T9 | T10 |
|--------|---------|----|----|----|----|----|----|----|----|----|-----|
| *第三视角输入* | | | | | | | | | | | |
| OpenVLA | 54.0 | 35.0 | 95.0 | 65.0 | 45.0 | 40.0 | 80.0 | 60.0 | 45.0 | 20.0 | 55.0 |
| UniVLA* | 63.0 | 64.0 | 82.0 | 76.0 | 96.0 | 58.0 | 98.0 | 24.0 | 74.0 | 32.0 | 26.0 |
| MemoryVLA | 93.4 | 92.0 | 96.0 | 96.0 | 100 | 100 | 100 | 96.0 | 96.0 | 62.0 | 96.0 |
| OpenVLA-OFT* | 91.0 | 82.0 | 96.0 | 96.0 | 94.0 | 90.0 | 96.0 | 92.0 | 100 | 70.0 | 94.0 |
| **HiF-VLA (Ours)** | **94.4** | **94.0** | **98.0** | **100** | **100** | 94.0 | **100** | 90.0 | 98.0 | **76.0** | 94.0 |
| *第三视角 + 腕部视角输入* | | | | | | | | | | | |
| Seer (scratch) | 78.7 | 80.0 | 90.0 | 91.7 | 81.7 | 85.0 | 65.0 | 86.7 | 88.3 | 51.7 | 66.7 |
| Seer | 87.7 | 91.7 | 90.0 | 98.3 | 100 | 91.7 | 93.3 | 85.0 | 88.3 | 61.7 | 71.7 |
| UniVLA* | 90.0 | 100 | 92.0 | 94.0 | 98.0 | 86.0 | 100 | 80.0 | 100 | 70.0 | 82.0 |
| OpenVLA-OFT* | 94.0 | 90.0 | 98.0 | 98.0 | 98.0 | 96.0 | 100 | 92.0 | 100 | 72.0 | 96.0 |
| **HiF-VLA (Ours)** | **96.4** | 88.0 | 98.0 | **100** | **100** | **100** | **100** | **96.0** | **100** | **82.0** | **100** |

**说明**: 500 次试验。第三视角下 HiF-VLA 达 94.4%（较 OpenVLA-OFT 基线绝对 +3.4%），且**第三视角变体已能与多视角基线持平**，凸显其时序推理能力；多视角下达 96.4%，全面超过其他 SOTA。任务列 T1–T10 对应 put-soup-and-box、put-box-and-butter、turn-on-stove-and-put-pot 等十个多子目标任务。

### Table 2: CALVIN ABC→D Results / CALVIN 长程泛化

| Method | 1 | 2 | 3 | 4 | 5 | Avg. Len. ↑ |
|--------|----|----|----|----|----|-------------|
| *第三视角输入* | | | | | | |
| SuSIE | 87.0 | 69.0 | 49.0 | 38.0 | 26.0 | 2.69 |
| OpenVLA | 91.3 | 77.8 | 62.0 | 52.1 | 43.5 | 3.27 |
| CLOVER | 96.0 | 83.5 | 70.8 | 57.5 | 45.4 | 3.53 |
| VPP | 90.9 | 81.5 | 71.3 | 62.0 | 51.8 | 3.58 |
| $\pi_0$ | 93.7 | 83.2 | 74.0 | 62.9 | 51.0 | 3.65 |
| UniVLA | 95.5 | 85.8 | 74.8 | 66.9 | 56.5 | 3.80 |
| **HiF-VLA (Ours)** | 93.5 | **87.4** | **81.4** | **75.9** | **69.4** | **4.08** |
| *第三视角 + 腕部视角输入* | | | | | | |
| GR-1 | 85.4 | 71.2 | 59.6 | 49.7 | 40.1 | 3.06 |
| Vidman | 91.5 | 76.4 | 68.2 | 59.2 | 46.7 | 3.42 |
| $\pi_0$ | 93.8 | 85.0 | 76.7 | 68.1 | 59.9 | 3.92 |
| UP-VLA | 92.8 | 86.5 | 81.5 | 76.9 | 69.9 | 4.08 |
| OpenVLA-OFT | 96.3 | 89.1 | 82.4 | 75.8 | 66.5 | 4.10 |
| RoboVLMs | 98.0 | 93.6 | 85.4 | 77.8 | 70.4 | 4.25 |
| Seer | 96.3 | 91.6 | 86.1 | 80.3 | 74.0 | 4.28 |
| VPP | 96.5 | 90.9 | 86.6 | 82.0 | **76.9** | 4.33 |
| **HiF-VLA (Ours)** | **98.5** | **94.1** | **88.1** | 81.4 | 73.1 | **4.35** |

**说明**: 在 ABC 训练、未见 D 环境评测，指标为连续 5 条指令平均完成任务数。HiF-VLA 在两种视角设置下平均任务长度均最优（4.08 / 4.35），较基线 +0.25，体现双向时序推理对长程任务依赖建模与泛化的增益（回答 RQ1）。

### Table 3: Efficiency & Redundancy / 效率与冗余消融（LIBERO-Long，hindsight 长 4，batch 4，第三视角）

| # | Methods | Peak GPU Mem.(GB) ↓ | Latency (ms) ↓ | Avg. SR ↑ |
|---|---------|---------------------|----------------|-----------|
| (1) | Baseline | 30.8 (1.00×) | 72.9 (1.00×) | 91.0 |
| (2) | + Subgoal | 38.2 (1.24×) | 115.9 (1.59×) | 91.8 |
| (3) | **+ Foresight (Ours)** | **31.8 (1.03×)** | **82.7 (1.13×)** | 92.2 |
| (4) | + History frames | 63.6 (2.06×) | 229.5 (3.15×) | 90.4 |
| (5) | **+ Hindsight (Ours)** | 31.4 (1.02×) | 117.7 (1.61×) | 92.2 |
| (6) | **+ Hindsight + Foresight (Ours)** | 32.2 (1.05×) | 121.6 (1.67×) | **93.2** |

**说明**: 关键对照。子目标法延迟 1.59×、帧堆叠 3.15× 且性能反降（90.4，说明像素冗余稀释了任务相关时序线索、易过拟合无关细节）；而本文的 **Foresight 头仅 0.13× 额外延迟、0.03× 显存**。完整 hindsight+foresight 取得 93.2% 最佳成功率，显存仅 1.05×，回答 RQ2。

### Table 4: Full LIBERO Suite Results / LIBERO 四套件结果（附录）

| Methods | Spatial | Object | Goal | Long | Average |
|---------|---------|--------|------|------|---------|
| TraceVLA | 84.6 | 85.2 | 75.1 | 54.1 | 74.8 |
| Octo | 78.9 | 85.7 | 84.6 | 51.1 | 75.1 |
| CoT-VLA | 81.1 | 87.5 | 91.6 | 87.6 | 69.0 |
| SpatialVLA | 88.2 | 89.9 | 78.6 | 55.5 | 78.1 |
| ThinkAct | 88.3 | 91.4 | 87.1 | 70.9 | 84.4 |
| Seer | - | - | - | 87.7 | 87.7 |
| FlowVLA | 93.2 | 95.0 | 91.6 | 72.6 | 88.1 |
| 4D-VLA | 88.9 | 95.2 | 90.9 | 79.1 | 88.6 |
| DreamVLA | 97.5 | 94.0 | 89.5 | 89.5 | 92.6 |
| CogACT | 97.2 | 08.0 | 90.2 | 88.8 | 93.2 |
| $\pi_0$ | 96.8 | 98.8 | 95.8 | 85.2 | 94.2 |
| GR00T N1 | 94.4 | 97.6 | 93.0 | 90.6 | 93.9 |
| UniVLA | 96.5 | 96.8 | 95.6 | 92.0 | 95.2 |
| MemoryVLA | 98.4 | 98.4 | 96.4 | 93.4 | 96.5 |
| OpenVLA-OFT | 97.6 | 98.4 | **97.9** | 94.5 | 97.1 |
| **HiF-VLA (Ours)** | **98.8** | **99.4** | 97.4 | **96.4** | **98.0** |

**说明**: 全套件对比。HiF-VLA 在最难的 LIBERO-Long 上优势最大（96.4），其余三套件也具竞争力或更优，平均 98.0 全场最佳，展示稳健性与通用性。（注：原表 CogACT 的 Object 列 08.0 疑为论文排版/转换笔误。）

### Table 5: Hyper-parameter Ablation / 超参消融（附录，多视角）

| 组 | Ablation | 参数 | SR |
|----|----------|------|----|
| (a) | Foresight Motion Loss Weight $\lambda$ | 0.1 | 94.4 |
| | | 0.05 | 95.2 |
| | | **0.01*** | **96.4** |
| | | 0.001 | 95.6 |
| (b) | Joint Expert Depth | 2 | 95.2 |
| | | 4 | 95.6 |
| | | **6*** | **96.4** |
| | | 8 | 95.2 |
| (c) | Length: (Hindsight, Foresight) | **(8,8)*** | **96.4** |
| | | (8,16) | 94.6 |
| | | (16,16) | 95.2 |

**说明**: (a) $\lambda=0.01$ 最优——运动分支贡献需"小而准"，过大破坏 VLA 稳定；(b) 联合专家深度 6 最佳，更深增益微弱；(c) 前瞻拉到 16 因长期预测误差累积而变差，但**把 hindsight 拉到 16（保持前瞻 16）反而改善**，支持"解耦回望与前瞻长度"的设计。`*` 为提交设置。

### Table 6: Variants Ablation / 变体消融（附录，第三视角）

| 组 | Ablation | Variant | SR |
|----|----------|---------|----|
| (a) | Hindsight+Foresight 用 Motion 还是 State | S+M | 92.6 |
| | | S+S | 92.0 |
| | | **M+M*** | **94.4** |
| (b) | 因果 vs 双向交互 | Causal-[M\|A] | 87.4 |
| | | Bi-[A\|M] | 94.0 |
| | | **Bi-[M\|A]*** | **94.4** |
| (c) | 运动表征 | Flow | 94.2 |
| | | **MVs** | **94.4** |

**说明**: (a) 用机器人本体状态（S）替代运动向量（M）会明显掉点（M+M 94.4 → S+S 92.0），说明 MV 捕捉了本体状态之外的**交互驱动视觉动态**（物体位移、接触运动、场景变化）；(b) 双向交互的联合专家（94.4）远超因果分离变体（87.4），且 token 顺序（M\|A 与 A\|M）影响 <0.5%，符合双向注意力全局感受野的预期；(c) 用光流（RAFT 估计）替代 MV 成功率几乎相同（94.2 vs 94.4），说明增益来自建模框架而非具体运动信号——但光流预处理开销大（4 帧历史时 186.8ms vs MV 121.6ms，8 帧时 MV 省约 78% 延迟开销），MV 更高效。

---

## 实验

### 数据集 / 基准

| 基准 | 规模 | 特点 | 用途 |
|------|------|------|------|
| [[LIBERO-Long]] | 10 个多子目标任务，500 次试验 | 单臂长程，跨多样场景 | 训练/测试 |
| [[CALVIN]] ABC-D | 4 个室内环境 A–D；A-C 训练、未见 D 测 | 连续 5 指令，强调泛化 | 训练/测试 |
| LIBERO 全套件 | Spatial/Object/Goal/Long | 四套件综合评估（附录） | 测试 |
| 真实世界（AgileX Piper） | 3 个长程任务，每任务 100 示范 / 20 试 | 6-DoF + 1-DoF 夹爪，20Hz 采集 | 训练/测试 |

### 实现细节

- **Backbone**: Prismatic-7B VLM，用 OpenVLA（OXE 预训练）权重初始化；视觉端 DINOv2 + SigLIP 混合编码器；其余模块随机初始化。
- **Hindsight 编码器**: 4 层 ViT + 浅层 3D 卷积；MV 取 MPEG-4 $16\times16$ 宏块；hindsight 窗口默认 8、可变。
- **联合专家**: 6 层 Transformer，共享嵌入维度 1024，跨流注意力，位置编码 RoPE，历史经 AdaLN 条件化。
- **时序块长**: 动作与前瞻均固定 $n=8$；损失权重 $\lambda=0.01$。
- **优化**: 8× NVIDIA A100，全局 batch 64；LIBERO 微调 150k 步、CALVIN 80k 步。
- **部署**: 真机推理在单张 RTX 4090 上。
- **主基线**: OpenVLA-OFT（同一预训练初始化）；另比较 Seer、VPP、$\pi_0$ 等。

### 关键实验结论

- **RQ1（性能）**: LIBERO-Long 第三视角 94.4% / 多视角 96.4%（均 SOTA）；CALVIN ABC-D 平均任务长 4.08/4.35，较基线 +0.25；LIBERO 全套件平均 98.0 最佳。
- **RQ2（冗余/效率）**: Foresight 头仅 0.13× 额外延迟、0.03× 显存；帧堆叠 3.15× 延迟且性能反降——MV 紧致表征同时提升效率与精度。
- **RQ3（可扩展性）**: 多帧基线延迟随历史长度近似线性增长（8 帧时 >4.5× vanilla），HiF-VLA 各上下文长度延迟几乎不增。
- **RQ4（消融）**: $\lambda=0.01$、联合专家深 6、(hindsight,foresight)=(8,8) 最优；历史"作条件不进 VLM"、双向联合专家、MV（相对 state/flow）均被证明有效。
- **RQ5（真机）**: 三个长程真机任务全面超 OpenVLA-OFT；尤其 Press-Buttons-Order，基线仅 17.4%（因按压前后视觉差异极小难以判断成功），HiF-VLA 凭宽时间感受野可靠检测细微状态转变。

---

## 批判性思考

### 优点

1. **表征选择有洞见且自洽**: 用视频编解码的 Motion Vector 当时序原语，是"压缩历史/未来"这一目标的优雅落点——它天然紧致、含交互视觉动态（消融 Table 6a 用 state 替换掉点即证据），且 MV 提取几乎零成本（相比光流省约 78% 延迟）。
2. **"近零开销扩展时间感受野"有硬数据支撑**: Table 3 把 subgoal（1.59×）、帧堆叠（3.15×）与本文 foresight（1.13×）并排，效率优势直观可信；Figure 3b 的延迟-历史长度曲线进一步佐证可扩展性。
3. **关键设计都配了针对性消融**: "历史作条件不进 VLM"（Fig 4）、"双向 vs 因果"（Table 6b，87.4→94.4）、"动作分支助运动收敛"（Fig 7）——主张大多有对照实验或可视化，而非空谈。
4. **真机覆盖有代表性的长程难点**: 按钮顺序任务专门构造"视觉近似但状态不同"的歧义，正中时序推理要害，基线 17.4% 的对比很有说服力。

### 局限性

1. **绝对增益偏小、对基线优势有限**: LIBERO-Long 较 OpenVLA-OFT 仅 +3.4%（第三视角）、CALVIN 仅 +0.25 任务长；这些基准已接近饱和（多个方法 >94%），"长程一致性"的核心卖点在更长程/更接触丰富的任务上才更能体现，但论文未给此类更难基准。
2. **运动表征对估计精度敏感**（作者自承）: MV 在高动态/噪声场景可能不稳；真机失败案例（Fig 10）几乎都源于空间几何/3D 感知不足，暗示纯 2D 运动表征对接触与深度的刻画有天花板。
3. **"think-while-acting"证据偏定性**: 主要靠 Fig 7 收敛曲线与 Fig 8 对齐可视化支撑，缺少对"前瞻运动质量↔动作成功率"的定量因果分析（如前瞻预测精度与任务成功的相关性）。
4. **与本体状态/光流的边界论证略单薄**: Table 6 显示 MV≈光流、MV>state，但 state 维度与 MV 信息量不完全可比；"MV 捕捉交互动态"的论点更多靠叙述而非分离实验直接量化。

### 潜在改进方向

1. 引入 3D/深度表征（作者已指出），缓解放置/堆叠/按压等对几何敏感的失败模式。
2. 用更长程、接触更丰富、可变形物体的基准压测"时序一致性"卖点，并补更强基线，量化"前瞻运动→动作成功"的因果链。
3. 探索作者提到的**互联网视频大规模预训练**来增强运动理解/生成，把 MV 世界模型做大。
4. 把 hindsight/foresight 长度、$\lambda$ 等做成自适应/可学习，减少对手工经验设定的依赖。

### 可复现性评估

- [ ] 代码开源（论文未给出公开仓库链接）
- [ ] 预训练模型（未声明 release）
- [x] 训练细节较完整（附录给出 ViT 层数、专家深度、嵌入维度、$\lambda$、步数、硬件等）
- [x] 数据集可获取（LIBERO、CALVIN 公开；真机数据自采，未声明 release）

---

## 速查卡片

> [!summary] HiF-VLA: Hindsight, Insight & Foresight via Motion for VLA
> - **核心**: 用视频编解码的 Motion Vector 作为时序原语，双向（过去 hindsight + 未来 foresight）扩展 VLA 时间感受野，做"边想边做"的长程操作。
> - **方法**: 历史帧→MV 回望先验（ViT+3D conv 编码）；VLM 并行出前瞻运动 token 与动作 token；回望经 AdaLN 条件化、在 6 层双向联合专家中融合前瞻运动流与动作流，L1 联合训练（$\lambda=0.01$）。历史只作条件、不进 VLM。
> - **结果**: LIBERO-Long 94.4%/96.4%、CALVIN ABC-D 4.08/4.35、LIBERO 全套件 98.0；前瞻头仅 +0.13× 延迟（对比 subgoal 1.59×、帧堆叠 3.15×）；真机三长程任务全面超 OpenVLA-OFT。
> - **Backbone**: Prismatic-7B（OpenVLA/OXE 初始化）+ DINOv2/SigLIP。

---

*笔记创建时间: 2026-06-29*
