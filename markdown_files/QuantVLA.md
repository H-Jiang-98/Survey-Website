---
title: "QuantVLA: Scale-Calibrated Post-Training Quantization for Vision-Language-Action Models"
method_name: "QuantVLA"
authors: [Jingxuan Zhang, Yunta Hsieh, Zhongwei Wan, Haokun Lin, Xin Wang, Ziqi Wang, Yingtie Lei, Mi Zhang]
year: 2026
venue: CVPR
tags: [VLA, post-training-quantization, diffusion-transformer, model-compression, efficient-vla, low-bit-inference]
zotero_collection: _inbox
image_source: online
arxiv_html: https://arxiv.org/html/2602.20309v4
created: 2026-06-29
---

# QuantVLA: Scale-Calibrated Post-Training Quantization for Vision-Language-Action Models

## 元信息

| 项目 | 内容 |
|------|------|
| 作者 | Jingxuan Zhang, Yunta Hsieh, Zhongwei Wan, Haokun Lin, Xin Wang, Ziqi Wang, Yingtie Lei, Mi Zhang |
| 机构 | (论文未在正文显式列出，详见 arXiv 页) |
| 会议 | CVPR 2026 |
| 类别 | VLA（Vision-Language-Action）/ 模型量化 |
| 日期 | 2026-04（arXiv v4） |
| 项目主页 | https://quantvla.github.io/ |
| 链接 | [arXiv](https://arxiv.org/abs/2602.20309) / [Project](https://quantvla.github.io/) |

---

## 一句话总结

> 首个面向 [[VLA]] 的免训练后训练量化（[[PTQ]]）框架：用"选择性量化布局 + 注意力温度匹配 + 输出头能量平衡"三件套，首次成功量化 [[Diffusion Transformer|DiT]] 动作头，在 LIBERO 上以约 70% 显存节省超越全精度基线。

---

## 核心贡献

1. **首个 VLA 量化敏感性系统分析**: 首次系统刻画带 [[Diffusion Transformer|DiT]] 动作头的 VLA 在量化下的脆弱性，从一阶误差传播角度证明量化引起的**尺度漂移**会改变注意力的**有效温度**（$s_q s_k$）与残差流**能量**（$s_v s_o$），从而定位 PTQ 崩溃的失效模式。
2. **首个免训练 VLA PTQ 框架 QuantVLA**: 提出基于旋转重参数化的免训练 PTQ，**首次成功量化 DiT 动作头**，在低比特推理下达到 SOTA 且大幅节省显存（量化模块约 70%）。
3. **三个尺度校准组件 + 架构不变**: 选择性量化布局（LLM + DiT MLP 整数化、注意力投影保持浮点）、[[Attention Temperature Matching|注意力温度匹配 (ATM)]]、[[Output Head Balancing|输出头能量平衡 (OHB)]]；两个校准量都折叠进反量化尺度，**不改算子调度、不加新算子、不需重训**。

---

## 问题背景

### 要解决的问题
[[VLA]] 模型把感知、语言、控制统一进单一策略网络，但随着主干增大、horizon 拉长，**计算与显存开销急剧上升**，难以在算力/显存/功耗受限的机器人平台上部署。如何在**不重训、不改架构**的前提下大幅压缩显存与带宽，同时保住操作成功率，是核心痛点。

### 现有方法的局限
作者把已有 VLA 效率工作分为两类，并指出共同盲区（见 Figure 1）：
1. **更高效的 VLA 模型设计**（[[TinyVLA]]、[[SmolVLA]]、FLOWER、X-VLA）：靠重设计架构/小主干提效，但需重训，且主要动刀视觉前端。
2. **围绕现有策略的效率框架**（[[EfficientVLA]] 剪语言层 + 复用表征、[[VLA-Cache]] 复用视觉 KV、[[MoLe-VLA]] 混合层路由）：靠剪枝/路由/缓存提速，**数值精度不变**。
3. **几乎所有方法都让策略头（DiT）保持全精度**，集中优化视觉前端 —— 而真正主导推理/控制开销的，恰恰是语言主干与扩散动作头。
4. **通用 PTQ 难迁移**：[[SmoothQuant]]、[[DuQuant]] 等为单模态 LLM/VLM 设计，无法处理 VLA 多模态-扩散紧耦合下的跨模态尺度漂移；直接套用会扭曲注意力温度与残差能量，导致低比特控制失稳。

### 本文的动机
既然 DiT 动作头是显存/计算大户却从未被量化，若能在不损性能下对其做 PTQ，PTQ 就能成为 VLA 的强力压缩工具。作者先做理论分析定位"为什么 DiT 在上游量化下脆弱"，再据此设计**选择性布局 + 两个轻量尺度校准**精准修复被破坏的关键尺度，实现免训练、保架构、跨模态鲁棒的低比特部署。

---

## 方法详解

### 模型架构

QuantVLA 不改原模型，而是在一个**基于 DiT 动作头的 VLA 流水线**上做量化与校准（见 Figure 2）：
- **输入**: 短历史 RGB 帧（经 [[SigLIP2]] / [[DINOv2]] 视觉编码器得图像 token）+ 自然语言指令（经语言主干编码）+ 机器人本体感知。
- **VLM 主干**: 视觉与文本 token 投影到共享 Transformer 空间，注意力融合得到任务条件表征 $F_{\mathrm{VL}}$。
- **动作头**: [[Diffusion Transformer|DiT]]，以 $F_{\mathrm{VL}}$、本体感知与扩散时间步 $t$ 为条件，迭代去噪得到动作隐变量，用 [[Flow Matching|流匹配]] 目标训练（视为条件 ODE 速度场）。
- **量化对象**: 把 LLM 全部线性层与 DiT 的 MLP 块整数化（W4A8），**注意力投影 $W_q,W_k,W_v,W_o$ 保持浮点**；外加 ATM、OHB 两个折叠进反量化尺度的标量校准。

扩散动作头的迭代去噪写作：

$$
x_{t-1}=f_{\theta}\!\big(x_{t},\ F_{\mathrm{VL}},\ t\big)
$$

经 $T$ 步细化后由 $x_0$ 解码出动作。

### 核心模块

#### 模块0（前置）: [[DuQuant]] 重参数化

**设计动机**: 在激进比特宽下，DuQuant 是 Transformer 栈里最稳的 PTQ 变体，靠可逆重参数化把激活/权重变得更"易量化"，且保持原线性映射不变。

**具体实现**（附录 B）:
- 对每个线性层 (i) 用对角矩阵 $\Lambda$ 做**逐通道平滑**；(ii) 做**块正交旋转** $\hat R_{(1)},\hat R_{(2)}$；(iii) 插入 **zigzag 通道置换 $P$** 重分布离群值。
- 平滑：$Y=(X\Lambda)(\Lambda^{-1}W)=X'W'$（公式20），平滑因子 $\Lambda_j$ 见公式21。
- 完整因式分解为可量化的 $G$（作用于激活）与折进权重的 $G^{-1}$（公式23）。QuantVLA 借用这一思路提升 VLA 线性层的低比特稳定性。

#### 模块1: 选择性量化布局（Selective Quantization Layout）

**设计动机**: 理论分析（3.2.2）表明，量化整个动作头或全栈会让误差沿注意力与残差通路累积放大；注意力投影最敏感，直接决定 softmax 分布与残差注入的稳定性。

**具体实现**:
- **整数化 LLM 全部线性层 + DiT 的 MLP 块**；
- **保留 DiT 注意力投影 $W_q,W_k,W_v,W_o$ 为浮点**，避免放大"温度漂移"与"能量漂移"两大病灶；
- 这一布局在保留整数计算显存收益的同时把漂移源头堵在最脆弱接口外（消融见 Table 1）。

#### 模块2: [[Attention Temperature Matching|注意力温度匹配 (ATM)]]

**设计动机**: 上游 LLM 量化会偏移送入 DiT 的统计量，改变 $Q,K$ 方差 → 改变 logits 尺度 → 改变 softmax 的**有效温度**，使注意力过尖或过平。ATM 用**每头一个标量 $\alpha$** 对齐师生 logits 离散度。

**具体实现**:
- 从无标签校准缓冲估计 $\alpha_{\mathrm{raw}}$ = 师 logits 标准差 / 生 logits 标准差（公式7）；
- 裁剪到安全区间 $[\alpha_{\min},\alpha_{\max}]$（公式8）；
- 加**中性带** $\varepsilon$：$|\log\alpha|<\varepsilon$ 时令 $\alpha=1$，降低对校准噪声的敏感（公式9）；
- 推理时 $L_Q=L_T/\alpha$（公式10），$\alpha$ 折进反量化尺度，**不引入额外算子**。

#### 模块3: [[Output Head Balancing|输出头能量平衡 (OHB)]]

**设计动机**: 多头拼接 + 输出投影后，注意力输出幅度发生系统性变化，改变**残差注入增益**与 LayerNorm 工作点；深层 DiT 中这种漂移沿残差/归一化累积。OHB 用**每层一个标量 $\beta$** 恢复残差接口能量。

**具体实现**:
- 第 $l$ 层输出头激活 $Z_l=\mathrm{Concat}\{A_{l,h}V_{l,h}\}W_{o,l}+b_{o,l}$（公式11）；
- 用 RMS 度量师生能量比 $\beta_{\mathrm{raw}}(l)$（公式12），裁剪（公式13）+ 中性带（公式14）；
- 推理时 $Z_Q=Z_l/\beta(l)$（公式15）重标输出头激活后再进残差路径，$\beta$ 仅作用于 DiT 头。

> 关键：ATM/OHB 都只是从无标签缓冲一次性估出的微小标量，**折进已有反量化尺度**，保持整数 GEMM 与算子调度不变，推理零额外 GEMM 开销，唯一开销是校准时的一次标量折叠。

### 关键公式与机制

#### 公式1: [[Diffusion Transformer|DiT]] 动作头迭代去噪

$$
x_{t-1}=f_{\theta}\!\big(x_{t},\ F_{\mathrm{VL}},\ t\big)
$$

**含义**: 扩散策略头以融合表征、当前动作隐变量与扩散步为条件，迭代细化动作隐变量。

**符号说明**:
- $x_t$: 第 $t$ 扩散步的动作隐变量；$F_{\mathrm{VL}}$: 视觉-语言融合表征；$t$: 扩散时间步；$f_\theta$: DiT 网络。

#### 公式2: 师生注意力 logits（量化前/后）

$$
L_{T}=\frac{Q_{T}K_{T}^{\top}}{\sqrt{d}},\qquad L_{Q}=\frac{Q_{Q}K_{Q}^{\top}}{\sqrt{d}}
$$

**含义**: 定义浮点教师与量化学生的 pre-softmax logits，用于刻画量化对注意力分布的扰动。

**符号说明**:
- $Q,K$: query/key；下标 $T$=teacher（浮点），$Q$=quantized；$d$: 头维度。

#### 公式3: logits 一阶扰动

$$
\Delta L\approx\frac{1}{\sqrt{d}}\Big((\varepsilon_{\text{up}}W_{q})K_{T}^{\top}+Q_{T}(\varepsilon_{\text{up}}W_{k})^{\top}\Big)+\Delta L_{\text{local}}
$$

**含义**: 即使注意力权重保持浮点，上游量化误差 $\varepsilon_{\text{up}}$ 仍线性传播到 logits，引发温度漂移。

**符号说明**:
- $\varepsilon_{\text{up}}=X_Q-X_T$: 上游输入漂移；$\Delta L_{\text{local}}$: 喂入 $Q,K$ 的量化激活的本地舍入/尺度误差。

#### 公式4: 注意力矩阵一阶展开

$$
A_{Q}\approx A_{T}+J_{\text{softmax}}(L_{T})\,\Delta L
$$

**含义**: 注意力分布的扰动由 softmax 雅可比 $J_{\text{softmax}}$ 把 logits 扰动放大/传递。

#### 公式5: 量化 value 路径

$$
V_{Q}=X_{Q}W_{v}=V_{T}+\varepsilon_{\mathrm{up}}W_{v}
$$

**含义**: value 同样继承上游漂移，进而影响输出能量。

#### 公式6: 师生输出头

$$
O_{T}=A_{T}\,V_{T}\,W_{o,T},\qquad O_{Q}=A_{Q}\,V_{Q}\,W_{o,Q}
$$

**含义**: 注意力 + value + 输出投影的复合，多头拼接后幅度系统性改变（即残差能量漂移的来源）。

#### 公式7–10: [[Attention Temperature Matching|ATM]] 校准

$$
\alpha_{\mathrm{raw}}=\frac{\operatorname{Std}(L_{T})}{\operatorname{Std}(L_{Q})+10^{-6}}
$$

$$
\alpha=\operatorname{clip}\!\big(\alpha_{\mathrm{raw}},\,\alpha_{\min},\,\alpha_{\max}\big)
$$

$$
\text{if }\big|\log\alpha\big|<\varepsilon\text{ then }\alpha=1
$$

$$
L_{Q}=\frac{L_{T}}{\alpha}
$$

**含义**: 用师生 logits 标准差之比估每头温度标量 $\alpha$，裁剪 + 中性带稳健化后，按 $L_Q=L_T/\alpha$ 校正 logits 尺度，使注意力既不过尖也不过平。

**符号说明**:
- $\operatorname{Std}(\cdot)$: 标准差（logits 离散度代理）；$\varepsilon$: 中性带（实验取 0.03）；$\alpha_{\min/\max}$: 安全区间（$\log\alpha$ 限幅 ±0.30 / 实验主文 ±0.4）。

#### 公式11–15: [[Output Head Balancing|OHB]] 校准

$$
Z_{l}=\operatorname{Concat}\{A_{l,h}V_{l,h}\}\,W_{o,l}+b_{o,l}
$$

$$
\beta_{\mathrm{raw}}(l)=\frac{\mathrm{RMS}\!\big(Z_{T,l}\big)}{\mathrm{RMS}\!\big(Z_{Q,l}\big)+10^{-6}}
$$

$$
\beta(l)=\operatorname{clip}\!\big(\beta_{\mathrm{raw}}(l),\,\beta_{\min},\,\beta_{\max}\big)
$$

$$
\text{if }\big|\log\beta(l)\big|<\varepsilon\text{ then }\beta(l)=1
$$

$$
Z_{Q}=\frac{Z_{l}}{\beta(l)}
$$

**含义**: 用师生输出头激活的 RMS 能量比估每层标量 $\beta$，校正后 $Z_Q=Z_l/\beta(l)$ 恢复残差注入增益与 LayerNorm 工作点。

**符号说明**:
- $\mathrm{RMS}(\cdot)$: 均方根（能量代理）；$Z_l$: 第 $l$ 层输出头激活；其余同 ATM。

#### 公式16–19: 通用 RTN 量化/反量化（附录A）

$$
\tilde{X}=\operatorname{clip}\!\Big(\operatorname{round}(X/\Delta_{X})+z_{X},\;0,\;2^{\,b_{X}}-1\Big)
$$

$$
\hat{X}=\Delta_{X}\big(\tilde{X}-z_{X}\big)
$$

$$
\begin{aligned}
\tilde{W}^{(o)}=\operatorname{clip}\!\Big(\operatorname{round}\!\big(W^{(o)}/\Delta_{W}^{(o)}\big),\ -2^{\,b_{W}-1},\,2^{\,b_{W}-1}-1\Big)
\end{aligned}
$$

$$
\hat{W}^{(o)}=\Delta_{W}^{(o)}\,\tilde{W}^{(o)}
$$

**含义**: 激活按 token 用无符号网格量化、权重按输出通道用有符号网格量化，反量化即乘回各自尺度。

**符号说明**:
- $\Delta_X>0$: 激活尺度（无标签校准缓冲估计）；$z_X$: 激活零点；$\Delta_W^{(o)}>0$: 第 $o$ 输出通道权重尺度；$b_X,b_W$: 激活/权重比特宽。

#### 公式20–23: [[DuQuant]] 平滑 + 旋转因式分解（附录B）

$$
Y=(X\Lambda)(\Lambda^{-1}W)=X^{\prime}W^{\prime}
$$

$$
\Lambda_{j}=\frac{\big(\max|X_{:,j}|\big)^{\alpha}}{\big(\max|W_{j,:}|\big)^{1-\alpha}},\qquad\alpha\in[0,1]
$$

$$
\boxed{\begin{aligned} Y&=XW\\
&=\underbrace{\big[(X\Lambda)\,\hat{R}_{(1)}\,P\,\hat{R}_{(2)}\big]}_{G}\;\underbrace{\big[\hat{R}_{(2)}^{\top}\,P^{\top}\,\hat{R}_{(1)}^{\top}\,(\Lambda^{-1}W)\big]}_{G^{-1}}\end{aligned}}
$$

**含义**: 逐通道平滑把量化难度从激活转移给权重；块正交旋转 $\hat R_{(1)},\hat R_{(2)}$ 与置换 $P$ 重分布离群值，且保持原线性函数（$P^{-1}=P^\top$，所有矩阵正交）。$G$ 作用激活端、$G^{-1}$ 折进权重端，分别量化后做整数矩乘。

#### 公式24–28: 反量化尺度如何决定温度与能量（附录C）

$$
\hat{Q}=s_{q}\,\tilde{Q},\qquad\hat{K}=s_{k}\,\tilde{K},\qquad\hat{V}=s_{v}\,\tilde{V}
$$

$$
L=\frac{\hat{Q}\hat{K}^{\top}}{\sqrt{d}}=\frac{s_{q}s_{k}}{\sqrt{d}}\,\tilde{Q}\tilde{K}^{\top}
$$

$$
Y=A\,\hat{V}=s_{v}\,A\,\tilde{V}
$$

$$
Z=\operatorname{Concat}(Y_{h})\,\hat{W}_{o}=s_{o}\,\operatorname{Concat}(Y_{h})\,\tilde{W}_{o}
$$

**含义**: 揭示核心机制 —— $s_q s_k$ 设定**有效温度** $T_{\mathrm{eff}}=\sqrt{d}/(s_q s_k)$ 控制注意力锐度；$s_v s_o$ 决定注入残差流的**能量**。这正是 ATM 校 $\alpha$（对应温度）、OHB 校 $\beta$（对应能量）的理论依据。

**符号说明**:
- $s_q,s_k,s_v,s_o$: $Q/K/V/O$ 的反量化尺度；$\tilde{\cdot}$: 整数张量；$A=\operatorname{softmax}(L)$。

---

## 关键图表

<!-- 图片均使用 arXiv HTML 在线链接，未本地化 -->

### Figure 1: Comparison of Representative VLA Efficiency Frameworks / 代表性 VLA 效率框架对比

![Figure 1a](https://arxiv.org/html/2602.20309v4/x1.png)
![Figure 1b](https://arxiv.org/html/2602.20309v4/x2.png)
![Figure 1c](https://arxiv.org/html/2602.20309v4/x3.png)

**说明**: 横向对比五类提效路线及其作用组件：(1) [[TinyVLA]] 走紧凑多模态 Transformer + 轻量扩散策略头；(2) [[EfficientVLA]] 剪冗余语言层 + 复用中间表征；(3) [[VLA-Cache]] 靠 KV 复用与视觉 token 静态缓存提吞吐；(4) [[MoLe-VLA]] 用混合层路由动态跳过语言模块计算；(5) **QuantVLA** 是唯一对语言主干与 DiT 同时做低比特量化、把"精度分配"当一等设计选择的框架。中/右面板预示其在 LIBERO 上超越全精度基线。佐证本文定位的盲区：他人都让策略头保持全精度、只动视觉前端。

### Figure 2: Overview of QuantVLA / 框架总览

![Figure 2](https://arxiv.org/html/2602.20309v4/x4.png)

**说明**: QuantVLA 总览，免训练、保架构、保算子调度。三件套：(1) 选择性量化布局——整数化 LLM 全部线性层与 DiT 全部 MLP 层，注意力投影 $Q,K,V,O$ 保浮点；(2) [[Attention Temperature Matching|ATM]] 每头标量 $\alpha$ 对齐师生 logits 并折进反量化尺度；(3) [[Output Head Balancing|OHB]] 每层标量 $\beta$ 匹配投影后能量。这张图把"哪量化/哪不量化/怎么校准"一图说清。

### Figure 3: ATM and OHB Effects Across Attention Blocks / ATM 与 OHB 的逐块效果

![Figure 3 Left](https://arxiv.org/html/2602.20309v4/x5.png)
![Figure 3 Right](https://arxiv.org/html/2602.20309v4/x6.png)

**说明**: 在 GR00T N1.5 上对比三配置（浮点教师、未校准量化基线、加 ATM/OHB）。左图为 logits 标准差：ATM 显著缩小与教师的差距，证明温度漂移被纠正；右图为输出投影后注意力输出 RMS：OHB 把能量拉回教师水平，缓解残差流能量漂移。深层尤其明显——这是 ATM/OHB 有效性的**机制级直接证据**（呼应公式24–28）。

### Figure 4: Memory Saving of QuantVLA / 显存节省

![Figure 4](https://arxiv.org/html/2602.20309v4/x7.png)

**说明**: QuantVLA 在 $\pi 0.5$ 与 GR00T N1.5 上相对基线的显存节省。量化模块（LLM+DiT）显存大幅下降，配合精度无损，特别适合长 horizon 策略生成与紧显存预算部署（可跑更长上下文或并行多策略）。

### Table 1: Selective Layer-Quantization (无 ATM/OHB) / 选择性层量化消融

LIBERO 成功率（%），W4A8，**关闭 ATM/OHB** 以隔离层选择的纯量化效果；Memory 为 LLM+DiT 部分显存(GB)。

| Model | Precision | Layer Selection | Layer Nums | Spatial | Object | Goal | Long | Avg. | Memory (GB) |
|-------|-----------|-----------------|-----------|---------|--------|------|------|------|-------------|
| $\pi 0.5$ | FP16 | No Quantization | 0 | 98.5 | 99.0 | 97.5 | 93.5 | 97.1 | 4.27 |
| $\pi 0.5$ | W4A8 | LLM | 126 | 98.0 | 98.5 | 97.5 | 92.0 | 96.5 | 1.58 |
| $\pi 0.5$ | W4A8 | DiT | 126 | 81.5 | 94.5 | 71.5 | 39.0 | 71.6 | 3.85 |
| $\pi 0.5$ | W4A8 | LLM+DiT | 252 | 86.0 | 97.5 | 71.5 | 50.0 | 76.3 | 1.17 |
| **$\pi 0.5$** | **W4A8** | **LLM+DiT (MLP)** | **180** | **98.0** | **97.0** | **94.5** | **92.0** | **95.4** | **1.28** |
| GR00T N1.5 | FP16 | No Quantization | 0 | 92.0 | 92.0 | 86.0 | 76.0 | 86.5 | 2.02 |
| GR00T N1.5 | W4A8 | LLM | 84 | 86.0 | 92.0 | 80.0 | 80.0 | 84.5 | 1.25 |
| GR00T N1.5 | W4A8 | DiT | 96 | 88.0 | 80.0 | 86.0 | 78.0 | 83.0 | 1.49 |
| GR00T N1.5 | W4A8 | LLM+DiT | 180 | 66.0 | 70.0 | 68.0 | 76.0 | 70.0 | 0.74 |
| **GR00T N1.5** | **W4A8** | **LLM+DiT (MLP)** | **116** | **90.0** | **86.0** | **80.0** | **74.0** | **82.5** | **0.91** |

**说明**: 量化整个动作头或全栈（LLM+DiT）退化最严重，尤其长程任务（$\pi 0.5$ Long 50.0、39.0；GR00T 68.0）；而**LLM+DiT(MLP)** 最接近基线又保留整数计算的显存收益，验证了 3.2.2 的理论分析。据此后续固定该布局（$Q,K,V,O$ 保浮点）。

### Table 2: Main Results on LIBERO / 主结果

LIBERO 成功率(%)、显存(GB)、相对节省。

| Model | Method | Precision | Spatial | Object | Goal | Long | Avg. | Memory (GB) | Rel. Savings |
|-------|--------|-----------|---------|--------|------|------|------|-------------|--------------|
| $\pi 0.5$ | FP16 (baseline) | FP16 | 98.5 | 99.0 | 97.5 | 93.5 | 97.1 | 4.27 | 0.0% |
| $\pi 0.5$ | +DuQuant(LLM+DiT) | W4A8 | 86.0 | 97.5 | 71.5 | 50.0 | 76.3 | 1.17 | 72.6% |
| $\pi 0.5$ | +QuantVLA(LLM) | W4A8 | 98.5 | 99.0 | 96.5 | 96.5 | 97.6 | 1.58 | 63.0% |
| **$\pi 0.5$** | **+QuantVLA** | **W4A8** | **98.5** | **98.0** | **98.0** | **96.0** | **97.6** | **1.28** | **70.0%** |
| GR00T N1.5 | FP16 (baseline) | FP16 | 92.0 | 92.0 | 86.0 | 76.0 | 86.5 | 2.02 | 0.0% |
| GR00T N1.5 | +DuQuant(LLM+DiT) | W4A8 | 66.0 | 70.0 | 68.0 | 76.0 | 70.0 | 0.74 | 63.4% |
| GR00T N1.5 | +QuantVLA(LLM) | W4A8 | 96.0 | 94.0 | 92.0 | 66.0 | 87.0 | 1.25 | 38.1% |
| **GR00T N1.5** | **+QuantVLA** | **W4A8** | **96.0** | **92.0** | **90.0** | **74.0** | **88.0** | **0.91** | **55.0%** |

**说明**: [[DuQuant]] 直接套用到 VLA 全栈时精度崩塌（$\pi 0.5$ 76.3%、GR00T 70.0%），说明单模态方法不迁移。QuantVLA 反而**超越全精度基线**：$\pi 0.5$ 97.6%（基线 97.1%）、显存 4.27→1.28GB（省 70%）；GR00T 88.0%（基线 86.5%）、2.02→0.91GB（省 55%）。这是首个对 VLA（尤其 DiT 头）成功 PTQ 的结果。

### Table 3: Precision Robustness on $\pi 0.5$ / 不同精度鲁棒性

| Model | Precision | Spatial | Object | Goal | Long | Avg. |
|-------|-----------|---------|--------|------|------|------|
| $\pi 0.5$ | FP16 | 98.5 | 99.0 | 97.5 | 93.5 | 97.1 |
| $\pi 0.5$ +QuantVLA | W4A8 | 98.5 | 98.0 | 98.0 | 96.0 | **97.6** |
| $\pi 0.5$ +QuantVLA | W4A4 | 98.5 | 98.5 | 93.5 | 90.5 | 95.3 |

**说明**: 即便压到 **W4A4** 激进比特，QuantVLA 仍达 95.3% 平均成功率，展示低比特下的稳定性。

### Table 4: Denoising Steps on GR00T N1.5 / 不同去噪步数

| Model | Denoising Steps | Spatial | Object | Goal | Long | Avg. |
|-------|-----------------|---------|--------|------|------|------|
| GR00T N1.5 | 8 | 92.0 | 92.0 | 86.0 | 76.0 | 86.5 |
| GR00T N1.5 +QuantVLA | 8 | 96.0 | 92.0 | 90.0 | 74.0 | **88.0** |
| GR00T N1.5 +QuantVLA | 16 | 96.0 | 94.0 | 84.0 | 80.0 | 88.5 |

**说明**: 跨去噪步数（8/16）QuantVLA 一致匹配或超过基线，证明对推理设置（噪声条件）泛化良好。

### Table 5: Comparison with SmoothQuant on $\pi 0.5$ (附录E) / 与 SmoothQuant 对比

LIBERO 成功率(%)。

| Method | Spatial | Object | Goal | Long | Avg. |
|--------|---------|--------|------|------|------|
| $\pi 0.5$ (FP16) | 98.5 | 99.0 | 97.5 | 93.5 | 97.1 |
| +SmoothQuant (LLM) | 97.5 | 98.5 | 98.0 | 92.5 | 96.6 |
| +SmoothQuant (LLM + DiT(MLP)) | 98.0 | 99.0 | 99.0 | 92.0 | 97.0 |
| +QuantVLA (LLM) | 98.5 | 99.0 | 96.5 | 96.5 | 97.6 |
| **+QuantVLA** | **98.5** | **98.0** | **98.0** | **96.0** | **97.6** |

**说明**: [[SmoothQuant]] 在 W8A8 下尚可（其作为 NVIDIA-OPT 内置 PTQ），而 QuantVLA 在更激进的 **W4A8** 即达到相当或略优结果，尤其在 Long 长程任务上更稳（96.0 vs 92.0），平均还略超浮点基线。

### Table 6: Pick-and-Can Benchmark (附录F) / 扩展基准

| Method | Precision | PickCan (成功数/50) |
|--------|-----------|---------------------|
| GR00T (FP16) | FP16 | 31 / 50 |
| + SmoothQuant | W4A8 | 16 / 50 |
| **+ QuantVLA** | **W4A8** | **27 / 50** |

**说明**: 在 Simpler 的 Pick-and-Can 操作任务上，W4A8 下 SmoothQuant 大幅掉到 16/50，QuantVLA 保持 27/50、显著缩小与浮点 31/50 的差距（虽未完全追平），表明其能缓解动作头对量化噪声的敏感。

### Table 7: OpenVLA (non-DiT) on LIBERO-Spatial (附录G) / 非 DiT 头适用性

| Model | Precision | Spatial |
|-------|-----------|---------|
| OpenVLA | FP16 | 84.7 |
| **+ QuantVLA** | **W8A16** | **86.0** |

**说明**: [[OpenVLA]] 用更深的 32 层语言主干与**非 DiT 动作头**，DiT 专用的 ATM/OHB 不直接适用；但 QuantVLA 仍能匹配（甚至略超 84.7→86.0）OpenVLA 性能，说明框架的选择性量化部分可超出 DiT 范畴。

---

## 实验

### 数据集 / 基准

| 基准 | 任务 | 特点 | 用途 |
|------|------|------|------|
| [[LIBERO]] | Spatial / Object / Goal / Long 四套件 | 分别测空间关系与精准放置、物体抓取、指令-目标对齐、长程时序分解与累积误差控制 | 主评测 |
| Simpler（Pick-and-Can） | 操作任务 | 跨任务鲁棒性评估（附录F） | 扩展评测 |
| LIBERO-Spatial（OpenVLA） | 空间套件 | 验证非 DiT 头适用性（附录G） | 扩展评测 |

被量化的两个 VLA 策略：**OpenPI $\pi 0.5$**（偏高效推理）与 **GR00T N1.5**（高容量、更丰富动作建模），均为 DiT 动作头，覆盖不同感知-控制耦合强度。

### 实现细节

- **精度**: 主设置 W4A8（权重 4 bit / 激活 8 bit）；额外测 W4A4、W8A16(OpenVLA)。
- **DuQuant 配置**: block size 64（输入输出一致）；启用通道置换重分布大通道；行旋转保函数等价。
- **校准**: 激活百分位 99.9 定裁剪范围；用 32 batch 估尺度；逐通道平滑系数 0.15。
- **ATM/OHB**: 各头 $\alpha$、各层 $\beta$；从无标签缓冲用 128 步、每任务至多 5 次 trial 拟合；$\log\alpha,\log\beta$ 限幅 0.30（正文主设置安全区间 ±0.4），中性带 $\varepsilon=0.03$；$\beta$ 作用范围限于 DiT 头；$\alpha$ 折进反量化尺度、$\beta$ 作用于模块输出。
- **硬件**: NVIDIA A100 GPU。
- **训练**: 完全免训练（training-free），仅一次性标量折叠。

### 关键实验结论

- **主结果**: 在两个 SOTA VLA 上 QuantVLA **超越全精度基线**（$\pi 0.5$ 97.6% vs 97.1%；GR00T 88.0% vs 86.5%），同时省显存 55–70%；DuQuant 直接套用则崩塌。
- **布局消融（Table 1）**: 量化全动作头/全栈最差，LLM+DiT(MLP) 最优，验证"注意力投影保浮点"的必要性。
- **校准消融（Figure 3）**: ATM 纠正 logits 温度、OHB 恢复输出能量，深层尤其明显。
- **鲁棒性**: 跨精度（W4A4 仍 95.3%）、跨去噪步数（8/16 均≥基线）稳定。
- **可迁移性**: 对非 DiT 头的 OpenVLA 也能匹配性能（Table 7）。

---

## 批判性思考

### 优点
1. **机制驱动、理论-方法-证据闭环**: 从一阶误差传播推出"$s_q s_k$→温度、$s_v s_o$→能量"两大病灶（公式24–28），据此设计 ATM/OHB，再用 Figure 3 逐块曲线直接验证校准生效，论证链条扎实。
2. **首个 VLA / DiT 头 PTQ 且超越基线**: 量化后成功率反而略升、显存省 55–70%，且**免训练、保架构、保算子调度**，落地价值高、与紧凑型 VLA 设计正交可叠加。
3. **轻量到几乎零成本**: ATM/OHB 仅每头/每层一个标量，折进反量化尺度，推理无额外 GEMM；与通用 PTQ（SmoothQuant/DuQuant）对比公平且占优（更低比特更稳）。

### 局限性
1. **评测面偏窄**: 主结果集中在 [[LIBERO]] 仿真，真机/接触丰富/可变形任务缺失；扩展基准 Pick-and-Can 上仍未追平浮点（27/50 vs 31/50），激进量化在真实操作下的代价尚未充分暴露。
2. **"超越基线"或含噪声成分**: 量化后成功率高于全精度，更可能是 LIBERO 任务的随机性/小样本评测波动（每套件试次有限），而非真正提升泛化；缺少多 seed 方差报告。
3. **超参与配置略有不一致 / 经验性**: 正文称安全区间 ±0.4、附录写 $\log$ 限幅 0.30；block size 64、平滑 0.15、百分位 99.9 等多为经验取值，缺敏感度分析。ATM/OHB 依赖 DiT 结构，对非 DiT 头（OpenVLA）只能退化为选择性量化。

### 潜在改进方向
1. 扩到真机与更难操作任务，并报告多 seed 方差，区分"真增益"与"评测噪声"；与剪枝/缓存类框架（EfficientVLA/VLA-Cache）组合验证叠加收益。
2. 把 $\alpha,\beta$ 的限幅、中性带、平滑系数等做自动搜索或可学习，减少手工经验；探索 W4A4/更低比特下的稳定边界。
3. 推广 ATM/OHB 到非 DiT（自回归 token）动作头，给出通用"温度-能量"校准接口。

### 可复现性评估
- [x] 代码/项目主页（https://quantvla.github.io/，声明开源框架）
- [ ] 预训练模型（基于已有 $\pi 0.5$/GR00T N1.5 权重，未额外训练）
- [x] 训练细节完整（附录 D 给出量化与校准超参；免训练）
- [x] 数据集可获取（LIBERO / Simpler 公开基准）

---

## 速查卡片

> [!summary] QuantVLA: Scale-Calibrated PTQ for VLA Models
> - **核心**: 首个免训练 VLA PTQ，首次成功量化 DiT 动作头；选择性布局（LLM+DiT MLP 整数化、注意力投影保浮点）+ ATM（每头温度标量 $\alpha$）+ OHB（每层能量标量 $\beta$），校准量折进反量化尺度、零额外算子。
> - **理论**: 量化尺度漂移经 $s_q s_k$ 改变注意力有效温度、经 $s_v s_o$ 改变残差能量，是 DiT 脆弱的根因。
> - **结果**: LIBERO 上 W4A8 超越全精度基线（$\pi 0.5$ 97.6% / GR00T 88.0%），显存省 55–70%；W4A4 仍 95.3%；对非 DiT 的 OpenVLA 也能匹配。
> - **代码**: https://quantvla.github.io/

---

*笔记创建时间: 2026-06-29*
