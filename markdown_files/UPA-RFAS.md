---
title: "When Robots Obey the Patch: Universal Transferable Patch Attacks on Vision-Language-Action Models"
method_name: "UPA-RFAS"
authors: [Hui Lu, Yi Yu, Yiming Yang, Chenyu Yi, Qixin Zhang, Bingquan Shen, Alex C. Kot, Xudong Jiang]
year: 2026
venue: CVPR
tags: [VLA, adversarial-attack, adversarial-patch, transferable-attack, robustness, black-box-attack]
zotero_collection: _inbox
image_source: online
arxiv_html: https://arxiv.org/html/2511.21192v3
created: 2026-06-29
---

# When Robots Obey the Patch: Universal Transferable Patch Attacks on Vision-Language-Action Models

## 元信息

| 项目 | 内容 |
|------|------|
| 作者 | Hui Lu, Yi Yu, Yiming Yang, Chenyu Yi, Qixin Zhang, Bingquan Shen, Alex C. Kot, Xudong Jiang |
| 机构 | 南洋理工大学 ROSE Lab（EEE / IGP）、南洋理工大学 CCDS、DSO National Laboratories（新加坡） |
| 会议 | CVPR 2026 |
| 类别 | VLA（Vision-Language-Action）安全 / 对抗攻击 |
| 日期 | 2026-03（arXiv v3） |
| 项目主页 | https://github.com/yuyi-sd/UPA-RFAS |
| 链接 | [arXiv](https://arxiv.org/abs/2511.21192) / [Code](https://github.com/yuyi-sd/UPA-RFAS) |

---

## 一句话总结

> 在单一替代模型的特征空间里学出一块**通用、可迁移的物理对抗 patch**，靠 $\ell_1$ 特征偏移 + 排斥式对比 + 鲁棒性增强双层优化 + 两个 VLA 专属损失（劫持跨模态注意力、破坏图文语义对齐），在完全黑盒下把多种受害 VLA 的任务成功率打崩。

---

## 核心贡献

1. **首个面向 VLA 机器人的通用可迁移 patch 攻击框架**：提出在替代模型[[特征空间]]里学单块物理 patch 的 [[UPA-RFAS]]，用 $\ell_1$ 特征偏移 + 排斥式 [[InfoNCE]] 对比对齐，把扰动导向**跨模型共享、可迁移**的高显著方向，而非过拟合到某个模型/数据/prompt。
2. **鲁棒性增强的通用 patch 攻击（RUPA / RAUP）**：用一个**不可见、逐样本**的扰动作为"硬增强器"在内层"对抗训练"地硬化替代模型，外层在被硬化邻域内、配合大量几何随机化训练通用 patch，从而蒸馏出稳定的跨输入方向，无需真去对抗训练庞大的 VLA。
3. **两个 VLA 专属损失**：[[Patch Attention Dominance|PAD]]（劫持 text→vision 注意力，把动作相关查询的注意力吸到 patch 上）与 [[Patch Semantic Misalignment|PSM]]（无标签地把 patch 语义拉向探针短语、推离当前指令嵌入，制造持续图文错位），并以理论命题（Proposition 1）论证 $\ell_1$ 替代偏移对受害侧偏移的下界保证。

---

## 问题背景

### 要解决的问题
[[VLA]] 模型对结构化视觉扰动（[[对抗攻击]]）天然脆弱，而真实机器人部署中攻击者面对的是**黑盒**：未知架构、未知微调变体、相机视角变化、仿真到真实（sim-to-real）域偏移。如何学出**一块**物理 patch，在这些条件下仍能**通用且可迁移**地破坏 VLA 策略，是本文核心问题。

### 现有方法的局限
- 现有 VLA patch 攻击（如 RoboticAttack）多假设**白盒**访问受害模型，难说明跨策略迁移性。
- 报告的 patch 常**协同过拟合**到某个具体模型、数据集或 prompt 模板，遇到未见架构或微调变体（如 OpenVLA-oft）成功率骤降，恰恰失效于安全评估最关心的黑盒场景。
- 结果是：缺白盒访问时会**高估**系统安全性，同时**低估**利用跨模态瓶颈的 patch 威胁。

### 本文的动机
作者用 [[CCA|典型相关分析（CCA）]] 与线性回归探针实证发现：替代模型与受害模型的视觉特征空间存在**强线性关系**（$R^2\!\approx\!0.654$，top-$k$ 典型相关接近 1），即二者共享一个低维子空间。因此**在该共享子空间内推动替代特征偏移，会在受害模型上诱发同源位移**，这为可迁移性提供了理论与经验依据。据此提出：(1) 用 $\ell_1$ 放大替代侧偏移、用对比损失把偏移压到批一致的稳定方向；(2) 用双层鲁棒优化模拟鲁棒替代；(3) 注入注意力与语义两个机器人相关信号。

---

## 方法详解

### 模型架构 / 威胁模型

[[UPA-RFAS]] 不是一个 VLA 模型，而是一套**攻击优化框架**（见 Figure 1）。它在一个**替代模型** $\hat\pi$（如 [[OpenVLA]]-7B）上、只用梯度访问，学出**单块通用物理 patch** $\boldsymbol\delta$，目标是迁移到一族未见**受害策略** $\Pi_{\mathrm{tgt}}$（OpenVLA-oft、$\pi_0$ 等）。

- **输入**: RGB 视觉流 $\mathbf{x}_t\in[0,1]^{H\times W\times 3}$ + 指令 $c$；攻击对象为 VLA 策略 $\pi$（视觉编码器 $f_{\mathrm v}$ + 投影器 $f_{\mathrm{prj}}$ + LLM 主干 $f_{\mathrm{llm}}$ + 动作头 $f_{\mathrm{act}}$）。
- **可学习量**: 通用 patch $\boldsymbol\delta\in[0,1]^{h_p\times w_p\times 3}$；以及内层逐样本不可见扰动 $\boldsymbol\sigma$。
- **核心机制**: [[特征空间]] $\ell_1$ 偏移 + [[InfoNCE]] 排斥对比 → 双层 min-max [[鲁棒性增强]]优化 → [[Patch Attention Dominance|PAD]] + [[Patch Semantic Misalignment|PSM]]。
- **渲染**: 每步采样面积保持的几何变换 $T_t\sim\mathcal T$（随机位置、倾斜、旋转），把 patch 贴到帧上。
- **patch 规格**: $50\times50$ 像素贴在 $224\times224$ 观测上（约 5% 面积）。

### 核心模块

#### 模块1: 特征空间 $\ell_1$ 偏移 + 排斥式对比对齐（3.3）

**设计动机**: 由 Proposition 1，放大替代侧特征偏移 $\|\Delta\mathbf z_i\|$ 必然在受害侧诱发非平凡响应；$\ell_1$ 驱动**稀疏、高显著**的偏移，避开替代特定的"怪癖"；对比损失把偏移压到**批一致、跨模型共享**的方向。

**具体实现**:
- $\mathcal L_1=\|\Delta\mathbf z_i\|_1$ 最大化替代侧特征偏移幅度（控制位移**大小**）。
- 排斥式 [[InfoNCE]] $\mathcal L_{\mathrm{con}}$ 把 patch 特征 $\tilde{\mathbf z}_i$ 推离其干净锚点 $\mathbf z_i$（控制位移**方向/角度**，基于余弦相似度）。
- 总特征目标 $\mathcal J_{\mathrm{tr}}=\mathcal L_1+\lambda_{\mathrm{con}}\mathcal L_{\mathrm{con}}$（公式10）。消融显示 $\mathcal L_{\mathrm{con}}$ 比 $\mathcal L_1$ 更关键。

#### 模块2: 鲁棒性增强的通用 patch 攻击 RUPA（3.4）

**设计动机**: 迁移攻击文献表明，在**鲁棒/对抗训练**的替代模型上生成的扰动迁移更好（更依赖跨架构"通用"特征）。但直接对抗训练庞大 VLA 代价高昂且损害任务性能。

**具体实现**:
- 不重训 VLA，而是引入**逐样本不可见扰动** $\boldsymbol\sigma$（限制在 patch 掩码内），通过内层 [[PGD]] 最小化攻击损失，**局部模拟对抗训练**、硬化替代模型。
- 外层冻结 $\boldsymbol\sigma$，在被硬化邻域内、配合随机放置/变换，**最大化**损失更新通用 patch $\boldsymbol\delta$，蒸馏稳定跨输入方向（公式11 双层 min-max）。
- 因 patch 局部、$\boldsymbol\sigma$ 不可见且逐输入，二者干扰有限。

#### 模块3: Patch Attention Dominance（PAD，跨模态注意力劫持，3.5）

**设计动机**: VLA 的动作主要由**少数动作相关文本查询**对视觉的跨模态注意力决定。把 patch 设计成**位置无关的注意力吸引子**：无论放在哪，都把这些查询的注意力从真实语义区域吸到 patch 上。

**具体实现**:
- 取 $f_{\mathrm{llm}}$ 最后 $N$ 个注意力块，按头平均、切出 text→vision 子矩阵（公式12），行归一化后求层平均，定义 patch 引起的注意力**份额增量** $\boldsymbol\Delta$（公式13）。
- 用 TopKMask 只保留干净注意力最高的 top-$\rho$ 文本 token 作为"动作相关查询"代理（公式14）。
- 把像素掩码双线性插值为 token 级掩码 $\mathbf M_z$，分别聚合 patch / 非 patch 上的注意力增量 $d_{\mathrm{patch}}, d_{\mathrm{non}}$（公式15）。
- PAD 损失（公式16）：增大 $d_{\mathrm{patch}}$、惩罚正的 $d_{\mathrm{non}}$，并加 margin 约束 patch 增量超过最强非 patch 增量至少 $m$。

#### 模块4: Patch Semantic Misalignment（PSM，图文语义错位，3.6）

**设计动机**: 仅劫持注意力不保证跨模型/任务一致的行为偏置。再在**语义空间**把 patch 视觉表征拉向**跨模型稳定的动作/方向原语（探针短语）**，同时推离当前指令表征，制造持续的图文错位。

**具体实现**:
- 对 patch 覆盖 token 做掩码加权池化并 $\ell_2$ 归一化得 $\hat{\mathbf v}_{\mathrm{patch}}$（公式17）。
- 探针原型 $\{\hat{\mathbf p}_k\}$（如 "put"、"pick up"、"place"、"open"、"close"、"left"、"right"）作架构无关锚点；$\hat{\mathbf t}$ 为整条指令的归一化表征。
- PSM 损失（公式18）：LogSumExp 项把 $\hat{\mathbf v}_{\mathrm{patch}}$ 拉向任一探针，第二项把它推离指令嵌入，$\alpha,\beta$ 平衡拉/推。消融显示**同时编码动作+方向**的探针最有效。

#### 模块5: 总框架 UPA-RFAS（3.7，Algorithm 1）

- **内层最小化**: $\boldsymbol\sigma^{(0)}=\mathbf 0$，用 PGD 最小化 $\mathcal J_{\mathrm{in}}=\mathcal J_{\mathrm{tr}}$，投影到 $\ell_\infty\le\epsilon_\sigma$ 球（公式19）。
- **外层最大化**: 固定 $\boldsymbol\sigma^\star$，用 AdamW 最大化 $\mathcal J_{\mathrm{out}}=\mathcal L_1+\lambda_{\mathrm{con}}\mathcal L_{\mathrm{con}}+\lambda_{\mathrm{PAD}}\mathcal L_{\mathrm{PAD}}+\lambda_{\mathrm{PSM}}\mathcal L_{\mathrm{PSM}}$，并 clamp 到 $[0,1]$（公式20）。

### 关键公式与机制

#### 公式1: [[对抗 Patch]] 渲染与贴附

$$
\tilde{\mathbf{x}}_{t}=\mathcal{P}(\mathbf{x}_{t},\boldsymbol{\delta},T_{t})=\big(\mathbf{1}-\mathbf{M}_{T_{t}}\big)\odot\mathbf{x}_{t}+\mathbf{M}_{T_{t}}\odot\mathcal{R}(\boldsymbol{\delta};T_{t})
$$

**含义**: 在帧 $\mathbf x_t$ 上，用几何变换 $T_t$ 诱导的二值掩码 $\mathbf M_{T_t}$ 把渲染后的 patch 贴到图像上，受面积预算 $\mathcal S(\boldsymbol\delta)<\rho$ 约束。

**符号说明**:
- $\odot$: Hadamard 积；$\mathbf 1$: 全一矩阵
- $\mathcal R(\boldsymbol\delta;T_t)$: 经变换渲染到全帧的 patch；$\rho$: 可见面积预算

#### 公式2: 通用 patch 攻击目标

$$
\boldsymbol{\delta}^{\star}\in\arg\max_{\mathcal{S}(\boldsymbol{\delta})<\rho}\ \mathbb{E}_{\mathbf{x}\sim p(\mathbf{x}),\,T_{t}\sim\mathcal{T}}\big[\mathcal{J}_{\mathrm{eval}}\big(\mathcal{P}(\mathbf{x},\boldsymbol{\delta},T_{t});\pi\big)\big]
$$

**含义**: 学一块在时间、视角、场景配置随机化下都鲁棒有效的单一 patch，最大化评估目标（任务损失增大 / 动作空间偏移）。

**符号说明**:
- $p(\mathbf x)$: 任务分布；$\mathcal T$: 随机放置/变换分布；$\pi$: 受害策略

#### 公式3: OpenVLA 策略分解

$$
\mathbf{y}=\mathrm{OpenVLA}(\mathbf{x},c)=f_{\mathrm{act}}\!\big(f_{\mathrm{llm}}([\,f_{\mathrm{prj}}(f_{\mathrm{v}}(\mathbf{x})),\mathrm{tok}(c)]\big)\big)
$$

**含义**: 视觉编码器 → 投影器对齐到 token 空间 → LLM 主干融合视觉 token 与指令 token → 动作头解码出连续控制（如 7-DoF）。

**符号说明**:
- $\mathbf E_v\in\mathbb R^{N_v\times D_v}$: 多粒度视觉嵌入（如 DINOv2+SigLIP 拼接）
- $\mathbf y\in\mathbb R^{D_a}$: 连续动作输出

#### 公式4: 可迁移 patch 攻击（双层定义）

$$
\max_{\boldsymbol{\delta}_{s}}\ \mathbb{E}_{\pi\sim\Pi_{\mathrm{tgt}}}\,\mathbb{E}_{\mathbf{x},T_{t}}\!\left[\mathcal{J}_{\mathrm{eval}}\big(\mathcal{P}(\mathbf{x},\boldsymbol{\delta}_{s},T_{t});\pi\big)\right]
$$

$$
\text{s.t.}\ \boldsymbol{\delta}_{s}\in\arg\max_{\boldsymbol{\delta}}\ \mathbb{E}_{\mathbf{x},T_{t}}\!\left[\mathcal{J}_{\mathrm{tr}}\big(\mathcal{P}(\mathbf{x},\boldsymbol{\delta},T_{t});\hat{\pi}\big)\right]
$$

**含义**: 在替代 $\hat\pi$ 上用可微目标 $\mathcal J_{\mathrm{tr}}$ 优化 patch，再在受害族 $\Pi_{\mathrm{tgt}}$ 上用评估目标 $\mathcal J_{\mathrm{eval}}$ 衡量迁移成功（无标签、untargeted）。

#### 公式5: 特征差异度量

$$
\mathcal{J}_{\mathrm{tr}}\big(\mathcal{P}(\mathbf{x},\boldsymbol{\delta},T_{t});\hat{\pi}\big)=\Delta\Big(f_{\hat{\pi}}\big(\mathcal{P}(\mathbf{x},\boldsymbol{\delta},T_{t})\big),\,f_{\hat{\pi}}(\mathbf{x})\Big)
$$

**含义**: 优化目标即"patched 特征 vs 干净特征"的偏移度量 $\Delta$，把攻击建模为特征空间问题而非任务损失。

#### 公式6: 线性对齐假设（Assumption 1）

$$
f_{\pi}(\mathbf{x})=f_{\hat{\pi}}(\mathbf{x})\,A^{\star}+e(\mathbf{x}),\qquad \|e(\tilde{\mathbf{x}})-e(\mathbf{x})\|_{2}\le\varepsilon_{E}
$$

**含义**: 受害特征近似为替代特征经线性映射 $A^\star$ 的结果，残差变化有界，这是迁移可行的核心假设。

**符号说明**:
- $A^\star$: 对齐矩阵；$\sigma_{\min}(A^\star)$: 其最小奇异值，决定偏移持久强度；$\varepsilon_E$: 残差变化上界

#### 公式7-8: 受害侧位移下界（Proposition 1）

$$
\|\Delta\mathbf{g}_{i}\|_{2}\ \ge\ \sigma_{\min}(A^{\star})\,\|\Delta\mathbf{z}_{i}\|_{2}-\varepsilon_{E}
$$

$$
\|\Delta\mathbf{g}_{i}\|_{1}\ \ge\ \frac{\sigma_{\min}(A^{\star})}{\sqrt{d}}\,\|\Delta\mathbf{z}_{i}\|_{1}-\varepsilon_{E}
$$

**含义**: 受害侧偏移 $\Delta\mathbf g_i$ 被替代侧偏移 $\Delta\mathbf z_i$ 线性下界（用 Hölder 不等式 $\|v\|_1\le\sqrt d\|v\|_2$ 转到 $\ell_1$）。因此最大化替代侧 $\ell_1$ 偏移**必然**诱发受害侧非平凡偏移（Corollary 1），这是用 $\mathcal L_1$ 的理论依据。

#### 公式9: 排斥式 InfoNCE 对比损失

$$
\mathcal{L}_{\mathrm{con}}=-\frac{1}{N}\sum_{i=1}^{N}\log\frac{\exp(\mathrm{sim}(\mathbf{z}_{i},\tilde{\mathbf{z}}_{i})/\tau)}{\sum_{j=1}^{N}\exp(\mathrm{sim}(\mathbf{z}_{i},\tilde{\mathbf{z}}_{j})/\tau)}
$$

**含义**: 把 $(\mathbf z_i,\tilde{\mathbf z}_i)$ 视作"应分开的一对"，最大化该损失即降低二者相似度，把 patched 特征推离干净锚点、并集中到批一致的共享方向（增强黑盒迁移）。

**符号说明**:
- $\mathrm{sim}$: 余弦相似度；$\tau$: 温度（实现取 0.07）；$\{\tilde{\mathbf z}_j\}_{j\ne i}$: 参考集

#### 公式10: 总特征目标

$$
\mathcal{J}_{\mathrm{tr}}=\mathcal{L}_{1}+\lambda_{\mathrm{con}}\,\mathcal{L}_{\mathrm{con}}
$$

**含义**: $\ell_1$ 偏移项（控大小）+ 加权排斥对比项（控方向），$\lambda_{\mathrm{con}}>0$ 平衡（实现取 10）。

#### 公式11: 鲁棒性增强双层目标

$$
\boldsymbol{\delta}^{\star}\in\arg\max_{\mathcal{S}(\boldsymbol{\delta})<\rho}\ \mathbb{E}_{\mathbf{x},T_{t}}\Big[\min_{\|\boldsymbol{\sigma}\|_{\infty}\le\epsilon_{\sigma}}\mathcal{J}_{\mathrm{tr}}\big(\mathcal{P}(\mathbf{x}+\boldsymbol{\sigma},\boldsymbol{\delta},T_{t});\hat{\pi}\big)\Big]
$$

**含义**: 内层用不可见扰动 $\boldsymbol\sigma$ **降低**攻击损失（局部对抗训练硬化替代），外层在此硬化邻域内对 $\boldsymbol\delta$ **最大化**同一损失。

#### 公式12-16: PAD 注意力劫持

$$
\boldsymbol{\Delta}=\tfrac{1}{N}\!\sum_{l}\mathrm{rn}\!\big(\mathbf{B}^{(l)}_{p}\big)-\tfrac{1}{N}\!\sum_{l}\mathrm{rn}\!\big(\mathbf{B}^{(l)}_{c}\big)
$$

$$
\tilde{\boldsymbol{\Delta}}=\boldsymbol{\Delta}\odot\boldsymbol{\chi},\quad \boldsymbol{\chi}=\mathrm{TopKMask}(\mathbf{B}_{c};\rho)
$$

$$
\mathcal{L}_{\mathrm{PAD}}=\mathbb{E}[d_{\mathrm{patch}}]-\lambda\,\mathbb{E}[\mathrm{ReLU}(d_{\mathrm{non}})]-\mathbb{E}\big[\mathrm{ReLU}(m-(d_{\mathrm{patch}}-\mathrm{non\_top}))\big]
$$

**含义**: $\boldsymbol\Delta$ 为 patched−clean 的 text→vision 注意力份额增量（只看增量，不看原始注意力）；用 top-$\rho$ 掩码 $\boldsymbol\chi$ 聚焦动作相关查询；PAD 增大 patch 区增量、压制非 patch 区增量，并强制 patch 增量比最强非 patch 增量高出 margin $m$。

**符号说明**:
- $\mathrm{rn}(\cdot)$: 对视觉 token 行归一化；$\mathbf M_z$: token 级 patch 掩码（双线性插值）
- $d_{\mathrm{patch}}=\langle\tilde{\boldsymbol\Delta},\mathbf M_z\rangle_p$, $d_{\mathrm{non}}=\langle\tilde{\boldsymbol\Delta},\mathbf 1-\mathbf M_z\rangle_p$；$m$: margin（取 0.1）

#### 公式17-18: PSM 语义错位

$$
\hat{\mathbf{v}}_{\mathrm{patch}}=\Bigg\|\bigg(\sum_{j=1}^{P}m_{j}\,\mathbf{z}_{j}\bigg)\Big/\bigg(\sum_{j=1}^{P}m_{j}+\varepsilon\bigg)\Bigg\|_{2}
$$

$$
\mathcal{L}_{\text{PSM}}=\alpha\left[\log\sum_{k=1}^{K}\exp\!\Big(\frac{\hat{\mathbf{v}}_{\mathrm{patch}}^{\top}\hat{\mathbf{p}}_{k}}{\tau}\Big)\right]-\beta\,\hat{\mathbf{v}}_{\mathrm{patch}}^{\top}\hat{\mathbf{t}}
$$

**含义**: 对 patch 覆盖 token 池化得位置无关语义描述子；LogSumExp 项把它拉向任一探针原型（$\tau$ 越小越聚焦最匹配锚点），第二项把它推离指令嵌入 $\hat{\mathbf t}$，制造持续图文错位。

**符号说明**:
- $\{\hat{\mathbf p}_k\}_{k=1}^{K}$: 归一化探针原型；$\hat{\mathbf t}$: 整条指令归一化表征
- $\alpha=1.0,\ \beta=0.5,\ \tau=0.3$（实现）

#### 公式19-20: 内层 PGD / 外层 AdamW

$$
\boldsymbol{\sigma}^{(i+1)}\leftarrow\Pi_{\|\cdot\|_{\infty}\le\epsilon_{\sigma}}\Big(\boldsymbol{\sigma}^{(i)}-\eta_{\boldsymbol{\sigma}}\nabla_{\boldsymbol{\sigma}}\mathcal{J}_{\mathrm{in}}\big(\mathcal{P}(\mathbf{x}+\boldsymbol{\sigma}^{(i)},\boldsymbol{\delta},T_{t});\hat{\pi}\big)\Big)
$$

$$
\mathcal{J}_{\mathrm{out}}=\mathcal{L}_{1}+\lambda_{\mathrm{con}}\mathcal{L}_{\mathrm{con}}+\lambda_{\mathrm{PAD}}\mathcal{L}_{\mathrm{PAD}}+\lambda_{\mathrm{PSM}}\mathcal{L}_{\mathrm{PSM}}
$$

**含义**: 内层 [[PGD]] **最小化** $\mathcal J_{\mathrm{in}}=\mathcal J_{\mathrm{tr}}$ 学不可见扰动（投影到 $\ell_\infty\le\epsilon_\sigma$）；外层用 AdamW **最大化** $\mathcal J_{\mathrm{out}}$ 更新 patch（取负号做最小化器）。

**符号说明**:
- $\eta_\sigma=1/510$, $\eta_\delta=1\times10^{-3}$；$\epsilon_\sigma=2/255$；内层 $I=8$ 步、外层 $K=50$ 步

---

## 关键图表

<!-- 图片均使用 arXiv HTML 在线链接，未本地化 -->

### Figure 1: Overall UPA-RFAS Framework / 整体框架

![Figure 1](https://arxiv.org/html/2511.21192v3/x1.png)

**说明**: UPA-RFAS 整体流程，在共享特征空间目标下两阶段协同。Phase 1（内层最小化）冻结 patch，用 [[PGD]] 学小而不可见的逐样本扰动 $\boldsymbol\sigma$，最小化特征目标 $\mathcal J_{\mathrm{in}}$（硬化替代模型）；Phase 2（外层最大化）冻结 $\boldsymbol\sigma$，优化单块物理 patch $\boldsymbol\delta$ 最大化 $\mathcal J_{\mathrm{out}}$，后者组合 $\ell_1$ 偏移、排斥对比、[[Patch Attention Dominance|PAD]] 与 [[Patch Semantic Misalignment|PSM]]。这张图把"为何能迁移"（共享子空间 + $\ell_1$ 下界）与"如何优化"（双层 min-max + 四损失）一并交代。

### Figure 2: Patch Visualization & Comparison / patch 模式对比

第一行为仿真设置训练，第二行为物理设置训练。

![UADA sim](https://arxiv.org/html/2511.21192v3/patch_sim_bs.png)
![TMA DoF7 sim](https://arxiv.org/html/2511.21192v3/patch_dof7_sim.png)
![Ours sim](https://arxiv.org/html/2511.21192v3/our_sim_patch.png)
![UADA phy](https://arxiv.org/html/2511.21192v3/patch_phy_bs.png)
![TMA DoF7 phy](https://arxiv.org/html/2511.21192v3/patch_dof7_phy.png)
![Ours phy](https://arxiv.org/html/2511.21192v3/our_phy_patch.png)

**说明**: (a)(d) baseline UADA 的 patch 纹理高度形似机器人夹爪；(b)(e) TMA 生成更抽象但替代特定的形状——二者都过拟合到物体/本体线索，妨碍跨模型迁移。(c)(f) 本文 patch 在特征空间学习、扰动更高层、模型无关的共享表征，避免物体模仿，从而黑盒迁移更强。这张图是"为何我们的 patch 更可迁移"的定性证据。

### Figure 3: Qualitative Real-World Results / 真实世界定性结果（附录 F）

上排为良性执行，下排为对抗执行（贴 patch 后）。

![benign 00](https://arxiv.org/html/2511.21192v3/figures/frame_00.png)
![benign 01](https://arxiv.org/html/2511.21192v3/figures/frame_01.png)
![benign 02](https://arxiv.org/html/2511.21192v3/figures/frame_02.png)
![benign 03](https://arxiv.org/html/2511.21192v3/figures/frame_03.png)
![benign 04](https://arxiv.org/html/2511.21192v3/figures/frame_04.png)
![benign 05](https://arxiv.org/html/2511.21192v3/figures/frame_05.png)
![benign 06](https://arxiv.org/html/2511.21192v3/figures/frame_06.png)
![benign 07](https://arxiv.org/html/2511.21192v3/figures/frame_07.png)
![adv 00](https://arxiv.org/html/2511.21192v3/figures/frame_our_00.png)
![adv 01](https://arxiv.org/html/2511.21192v3/figures/frame_our_01.png)
![adv 02](https://arxiv.org/html/2511.21192v3/figures/frame_our_02.png)
![adv 03](https://arxiv.org/html/2511.21192v3/figures/frame_our_03.png)
![adv 04](https://arxiv.org/html/2511.21192v3/figures/frame_our_04.png)
![adv 05](https://arxiv.org/html/2511.21192v3/figures/frame_our_05.png)
![adv 06](https://arxiv.org/html/2511.21192v3/figures/frame_our_06.png)
![adv 07](https://arxiv.org/html/2511.21192v3/figures/frame_our_07.png)

**说明**: 真实机器人执行序列对比。良性轨迹顺利完成任务，贴上通用 patch 后执行被破坏，直观展示物理可实现的攻击效果。

### Figure 4: Training Videos (Sim & Physical) / 训练视频帧（附录 G）

上排为仿真训练视频的 8 帧，下排为物理训练视频的 8 帧。

![sim 0](https://arxiv.org/html/2511.21192v3/figures/0_sim.png)
![sim 1](https://arxiv.org/html/2511.21192v3/figures/1_sim.png)
![sim 2](https://arxiv.org/html/2511.21192v3/figures/2_sim.png)
![sim 3](https://arxiv.org/html/2511.21192v3/figures/3_sim.png)
![sim 4](https://arxiv.org/html/2511.21192v3/figures/4_sim.png)
![sim 5](https://arxiv.org/html/2511.21192v3/figures/5_sim.png)
![sim 6](https://arxiv.org/html/2511.21192v3/figures/6_sim.png)
![sim 7](https://arxiv.org/html/2511.21192v3/figures/7_sim.png)
![phy 0](https://arxiv.org/html/2511.21192v3/figures/0_phy.png)
![phy 1](https://arxiv.org/html/2511.21192v3/figures/1_phy.png)
![phy 2](https://arxiv.org/html/2511.21192v3/figures/2_phy.png)
![phy 3](https://arxiv.org/html/2511.21192v3/figures/3_phy.png)
![phy 4](https://arxiv.org/html/2511.21192v3/figures/4_phy.png)
![phy 5](https://arxiv.org/html/2511.21192v3/figures/5_phy.png)
![phy 6](https://arxiv.org/html/2511.21192v3/figures/6_phy.png)
![phy 7](https://arxiv.org/html/2511.21192v3/figures/7_phy.png)

**说明**: 用于 patch 训练的仿真/物理视频示例帧，展示数据来源与场景多样性（随机放置/视角）。

### Table 1: Transfer OpenVLA-7B → OpenVLA-oft 变体（LIBERO 成功率 %，越低攻击越强）

| objective | oft-w Sim spatial | object | goal | long | **avg.** | oft-w Phy spatial | object | goal | long | **avg.** | oft Sim spatial | object | goal | long | **avg.** | oft Phy spatial | object | goal | long | **avg.** |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Benign | 99 | 99 | 98 | 97 | 98.25 | 99 | 99 | 98 | 97 | 98.25 | 98 | 98 | 98 | 94 | 97.00 | 98 | 98 | 98 | 94 | 97.00 |
| UMA₁ | 25 | 86 | 40 | 31 | 45.50 | 83 | 89 | 76 | 73 | 80.25 | 79 | 95 | 69 | 3 | 61.50 | 96 | 90 | 90 | 83 | 89.75 |
| UMA₁₋₃ | 46 | 88 | 38 | 39 | 52.75 | 90 | 87 | 83 | 81 | 85.25 | 90 | 93 | 57 | 3 | 60.75 | 96 | 81 | 93 | 80 | 87.50 |
| UADA₁ | 35 | 82 | 27 | 21 | 41.25 | 71 | 90 | 57 | 74 | 73.00 | 94 | 86 | 61 | 3 | 61.00 | 92 | 96 | 79 | 84 | 87.75 |
| UADA₁₋₃ | 37 | 74 | 21 | 33 | 41.25 | 65 | 88 | 46 | 61 | 65.00 | 87 | 90 | 64 | 4 | 61.25 | 92 | 95 | 61 | 90 | 84.50 |
| TMA₁ | 69 | 89 | 58 | 61 | 69.25 | 78 | 92 | 74 | 83 | 81.75 | 99 | 93 | 81 | 25 | 74.50 | 98 | 92 | 84 | 86 | 90.00 |
| TMA₇ | 47 | 78 | 47 | 34 | 51.50 | 90 | 96 | 89 | 90 | 91.25 | 83 | 93 | 75 | 62 | 78.25 | 97 | 88 | 89 | 87 | 90.25 |
| **Our** | **7** | **0** | **10** | **6** | **5.75** | **26** | **53** | **54** | **28** | **40.25** | **66** | **43** | **62** | **3** | **43.50** | **69** | **74** | **76** | **27** | **61.50** |

**说明**: 核心结果表。从替代 OpenVLA-7B 黑盒迁移到 OpenVLA-oft-w（多套件联合）与单套件 oft 变体。仿真下良性 98.25%，本文把 oft-w 打到 **5.75%**（>92 个百分点降幅），几乎完全瘫痪策略；baseline 仅降到 41.25%–69.25% 且对 object 类任务几乎无效（UMA/UADA 仍 >74%）。物理下本文 40.25% 仍为最低（baseline 65–91.25%）。本文成功率全程最低 = 攻击最强。

### Table 2: Component Ablation（迁移到 oft，物理，成功率 %，越低越强）

| objective | spatial | object | goal | long | avg. |
|---|---|---|---|---|---|
| **Our** | **69** | **74** | **76** | **27** | **61.50** |
| w/o RUPA | 70 | 75 | 71 | 33 | 62.25 |
| w/o PAD | 68 | 67 | 77 | 38 | 62.50 |
| w/o PSM | 69 | 72 | 81 | 32 | 63.50 |
| w/o $\mathcal{J}_{\mathrm{tr}}$ | 90 | 86 | 94 | 73 | 85.75 |
| w/o $\mathcal{L}_{\mathrm{con}}$ | 93 | 63 | 79 | 48 | 70.75 |
| w/o $\mathcal{L}_{1}$ | 74 | 74 | 77 | 31 | 64.00 |

**说明**: 去掉任一模块都削弱攻击。最严重的是去掉整个特征空间项 $\mathcal J_{\mathrm{tr}}$（含 $\mathcal L_1+\mathcal L_{\mathrm{con}}$）：成功率飙到 85.75%（接近良性），说明特征空间目标是迁移的根基。$\mathcal L_{\mathrm{con}}$（70.75）比 $\mathcal L_1$（64.00）影响更大——前者控方向、后者控大小，方向更关键。

### Table 3: Text-Probe Phrasing Ablation（迁移到 oft，物理，成功率 %）

| objective | spatial | object | goal | long | avg. |
|---|---|---|---|---|---|
| **Our (动作+方向)** | **69** | **74** | **76** | **27** | **61.50** |
| Action only | 76 | 67 | 94 | 48 | 71.25 |
| Direction only | 72 | 75 | 78 | 75 | 75.00 |

**说明**: 仅用动作词（71.25%）或仅用方向词（75.00%）都明显弱于联合编码动作+方向（61.50%）。同时编码动作与方向线索的文本查询更贴合策略的动作相关通道，迁移更强。

### Table 4: White-box Results on LIBERO Sim（附录 C，成功率 %，越低越强）

| Objective | Sim Spatial△ | Object△ | Goal△ | Long∗ | Avg. | Phy Spatial△ | Object△ | Goal△ | Long△ | Avg. |
|---|---|---|---|---|---|---|---|---|---|---|
| Benign | 84.7 | 88.4 | 79.2 | 53.7 | 76.5 | 84.7 | 88.4 | 79.2 | 53.7 | 76.5 |
| Random Noise | 71.2 | 85.2 | 79.0 | 51.6 | 71.7 | 71.2 | 85.2 | 79.0 | 51.6 | 71.7 |
| UMA₁ | 0.0 | 1.0 | 0.0 | 0.0 | 0.2 | 10.4 | 39.2 | 35.2 | 20.0 | 26.2 |
| UMA₁₋₃ | 0.0 | 1.2 | 1.0 | 0.0 | 0.5 | 3.4 | 43.6 | 20.0 | 18.0 | 21.2 |
| UADA₁ | 0.0 | 0.8 | 0.0 | 0.0 | 0.2 | 0.8 | 1.2 | 7.4 | 3.4 | 3.2 |
| UADA₁₋₃ | 0.0 | 0.0 | 0.0 | 0.0 | **0.0** | 0.0 | 0.0 | 0.0 | 0.0 | **0.0** |
| UPA | 3.8 | 22.2 | 12.0 | 3.2 | 10.3 | 4.4 | 43.0 | 42.8 | 27.2 | 29.3 |
| TMA (Avg.) | 0.8 | 13.6 | 11.1 | 2.0 | 6.9 | 9.9 | 48.9 | 41.3 | 21.1 | 30.3 |
| **Our** | 0.0 | 0.0 | 2.0 | 0.0 | **0.5** | 0.0 | 9.0 | 0.0 | 2.0 | **2.75** |

**说明**: 白盒设置（受害=替代，∗为同域、△为跨域受害数据集）。本文虽为黑盒迁移设计，白盒仍极强：仿真 0.5%、物理 2.75%，与最强 UMA/UADA 相当，远优于 UPA/TMA（误差棒原文带 ±，此处略）。

### Table 5: Transfer OpenVLA-7B → $\pi_0$（附录 D，成功率 %，越低越强）

| objective | Sim spatial | object | goal | long | avg. | Phy spatial | object | goal | long | avg. |
|---|---|---|---|---|---|---|---|---|---|---|
| Benign | 96 | 98 | 95 | 79 | 92.00 | 96 | 98 | 95 | 79 | 92.00 |
| UMA₁ | 100 | 94 | 91 | 72 | 89.25 | 98 | 99 | 98 | 79 | 93.50 |
| UMA₁₋₃ | 99 | 97 | 95 | 77 | 92.00 | 97 | 97 | 90 | 72 | 89.00 |
| UADA₁ | 93 | 96 | 90 | 75 | 88.50 | 93 | 94 | 96 | 73 | 89.00 |
| UADA₁₋₃ | 95 | 96 | 96 | 79 | 91.50 | 96 | 96 | 94 | 70 | 89.00 |
| DOF₁ | 98 | 97 | 90 | 72 | 89.25 | 96 | 97 | 94 | 78 | 91.25 |
| DOF₇ | 98 | 97 | 94 | 75 | 91.00 | 97 | 99 | 86 | 79 | 90.25 |
| **Our** | **91** | **96** | **85** | **72** | **86.00** | **93** | **92** | **82** | **67** | **83.50** |

**说明**: 迁移到与 OpenVLA 架构、预训练、数据、动作头全不同的 $\pi_0$（最难场景）。本文仍最强：仿真 86.00%（比最佳 baseline UADA₁ 的 88.5% 低 2.5 点）、物理 83.50%（比最强 baseline 89.0% 低 5.5 点）。降幅虽不及对 OpenVLA-oft 那么剧烈，但跨异构 VLA 仍稳定有效。

### Table 6: Patch Size Ablation（迁移到 oft，物理，成功率 %）

| Patch size | spatial | object | goal | long | avg. |
|---|---|---|---|---|---|
| 3% | 79 | 86 | 88 | 66 | 79.75 |
| 5% | 69 | 74 | 76 | 27 | 61.50 |
| 7% | 28 | 78 | 35 | 15 | 39.00 |
| 10% | 41 | 6 | 35 | **1** | **20.75** |

**说明**: patch 越大攻击越强（10% → 20.75%）。正文默认用 5%（$50\times50$ / $224\times224$）以平衡隐蔽性与威胁。

### Table 7: $\lambda_{\mathrm{con}}$ Ablation（迁移到 oft，物理，成功率 %）

| objective | spatial | object | goal | long | avg. |
|---|---|---|---|---|---|
| $\lambda_{\mathrm{con}}=1$ | 68 | 77 | 75 | 35 | 63.75 |
| $\lambda_{\mathrm{con}}=2$ | 72 | 73 | 76 | 28 | 62.25 |
| $\lambda_{\mathrm{con}}=5$ | 70 | 77 | 67 | 33 | 61.75 |
| $\lambda_{\mathrm{con}}=10$ | 69 | 74 | 76 | 27 | **61.50** |

**说明**: 对比损失权重 $\lambda_{\mathrm{con}}$ 越大攻击略增强，默认取 10（最低 61.50%），但整体不敏感。

### Table 8: RUPA 中 $\epsilon$（$\epsilon_\sigma$）Ablation（迁移到 oft，物理，成功率 %）

| objective | spatial | object | goal | long | avg. |
|---|---|---|---|---|---|
| $\epsilon=1/255$ | 73 | 73 | 77 | 28 | 62.75 |
| $\epsilon=2/255$ | 69 | 74 | 76 | 27 | 61.50 |
| $\epsilon=4/255$ | 66 | 71 | 62 | 33 | **58.00** |
| $\epsilon=8/255$ | 72 | 62 | 70 | 38 | 60.50 |
| $\epsilon=16/255$ | 75 | 67 | 68 | 36 | 61.50 |

**说明**: 内层扰动预算 $\epsilon_\sigma$ 适中（4/255）最优（58.00%），过大反而下降，说明硬化邻域需恰当。默认取 2/255。

---

## 实验

### 数据集 / 基准

| 基准 | 规模 | 特点 | 用途 |
|------|------|------|------|
| [[LIBERO]] | 4 套件（Spatial/Object/Goal/Long），每套件 10 任务 ×10 试 = 100 rollouts | 仿真操作；Long 多步长程最难 | 训练/评估 |
| [[BridgeData V2]] | 24 环境、13 技能、60,096 条轨迹 | 真实世界操作语料 | 物理设置训练/评估 |

### 实现细节

- **替代模型**: [[OpenVLA]]-7B（物理用 BridgeData V2 版）、OpenVLA-7B-LIBERO-Long（仿真用）；迁移时不用受害任何信息（权重/架构/微调数据/超参）。
- **受害模型**: OpenVLA-oft（4 个单套件变体 + 多套件 OpenVLA-oft-w，含 ~26× 吞吐优化微调）、$\pi_0$（异构 flow-based 策略，最难迁移）。
- **基线**: RoboticAttack 的 6 个目标——UMA、UADA、TMA（对应不同 DoF）。
- **patch**: $50\times50$ 像素贴 $224\times224$ 观测（~5% 面积），预定放置点避免遮挡物体；几何随机化（位置/倾斜/旋转）。
- **优化**: batch=2；内层 PGD $I=8$ 步、$\eta_\sigma=1/510$、$\epsilon_\sigma=2/255$；外层 AdamW $K=50$ 步、$\eta_\delta=1\times10^{-3}$；共 2000 迭代。
- **损失权重**: $\lambda_{\mathrm{con}}=10$、$\lambda_{\mathrm{PAD}}=1$、$\lambda_{\mathrm{PSM}}=0.5$；InfoNCE $\tau=0.07$；PAD 用最后 2 层、$\lambda_{\mathrm{non}}=0.8$、top-$\rho=0.3$、$m=0.1$；PSM $\alpha=1.0,\beta=0.5,\tau=0.3$。
- **指标**: 成功率 SR（值越低 = 攻击越强）；untargeted、无标签。

### 关键实验结论

- **黑盒迁移（Table 1）**: OpenVLA-7B → oft-w 把仿真成功率从 98.25% 打到 5.75%（>92 点降幅），物理 40.25%；全程最低，远超所有 baseline。
- **跨异构 VLA（Table 5）**: 迁移到 $\pi_0$ 仍最强（仿真 86.0% / 物理 83.5%），证明非过拟合替代。
- **白盒（Table 4）**: 仿真 0.5% / 物理 2.75%，与最强基线持平，说明攻击本身极强。
- **消融（Table 2/3/6/7/8）**: 特征空间项 $\mathcal J_{\mathrm{tr}}$ 最关键，$\mathcal L_{\mathrm{con}}>\mathcal L_1$；动作+方向联合探针最优；patch 越大越强；$\epsilon_\sigma$、$\lambda_{\mathrm{con}}$ 不太敏感。
- **patch 模式（Fig 2）**: baseline patch 形似夹爪/物体（过拟合），本文 patch 抽象、模型无关，故迁移更好。
- **物理可实现（Fig 3）**: 真实机器人贴 patch 后执行被破坏。

---

## 批判性思考

### 优点
1. **理论 + 经验双支撑迁移性**: 用 CCA/线性探针实证共享子空间（$R^2\approx0.654$），再用 Proposition 1 给出"$\ell_1$ 替代偏移 ⇒ 受害侧偏移下界"，把"为何能迁移"说清楚，不是纯经验堆砌。
2. **设计针对 VLA 痛点**: PAD 劫持 text→vision 注意力、PSM 制造图文语义错位，都直击 VLA 跨模态瓶颈，且 PSM 无需标签、探针架构无关；消融逐项验证有效。
3. **威胁现实**: 黑盒、物理可实现、跨架构/跨微调/sim-to-real 均有效，把 oft-w 打到近瘫痪，确实揭示了一个被低估的实用攻击面并立了防御基线。

### 局限性
1. **替代单一**: 主要替代为 OpenVLA-7B 家族；受害虽含 $\pi_0$，但替代多样性不足，"共享子空间"假设在更异构替代/受害组合下是否成立未充分检验。对 $\pi_0$ 的降幅（仿真仅 −6 点、物理 −8.5 点）也明显小于对 oft，说明跨架构迁移仍受限。
2. **评估面偏窄**: 仅 LIBERO + BridgeData V2，仅 OpenVLA/oft/$\pi_0$；缺对自回归 token 化策略、扩散/RL-VLA 更广家族的系统迁移；物理实验规模与统计强度（误差棒、试次）披露有限。
3. **无防御对照**: 论文定位为"未来防御的强基线"，但未测试任何防御（对抗训练、patch 检测、随机平滑）下攻击是否仍有效，攻防闭环缺失。
4. **隐蔽性与超参经验性**: 5% 面积 patch 在真实部署中是否易被察觉未讨论；放置点"预定且不遮挡物体"是较宽松的攻击者假设；多个权重/层数为经验取值。

### 潜在改进方向
1. 扩展到多替代集成、跨家族替代，量化 $\sigma_{\min}(A^\star)$ 与迁移成功率的因果关系，验证共享子空间假设的边界。
2. 引入主流防御（对抗训练 / patch 检测 / 输入随机化）做攻防评估，形成闭环并校准"基线"强度。
3. 探索更隐蔽（更小/更自然纹理）或自适应放置的 patch，以及对扩散/RL-VLA、长程多步任务的迁移。

### 可复现性评估
- [x] 代码开源（https://github.com/yuyi-sd/UPA-RFAS）
- [ ] 预训练模型 / patch 权重（论文未明确 release patch，依赖 code 仓库）
- [x] 训练细节完整（附录 B 给出全套超参与优化设置）
- [x] 数据集可获取（LIBERO、BridgeData V2、OpenVLA/oft/$\pi_0$ 均公开）

---

## 速查卡片

> [!summary] UPA-RFAS: Universal Transferable Patch Attacks on VLA
> - **核心**: 在替代模型特征空间学单块通用物理 patch，黑盒迁移破坏多种受害 VLA 的任务执行。
> - **方法**: $\ell_1$ 特征偏移 + 排斥 InfoNCE（控大小+方向）→ 双层 min-max 鲁棒性增强（内层 PGD 不可见扰动硬化替代，外层 AdamW 训 patch）→ PAD 劫持 text→vision 注意力 + PSM 图文语义错位；Proposition 1 给出 $\ell_1$ 迁移下界。
> - **结果**: OpenVLA-7B→oft-w 仿真 98.25%→5.75%（>92 点降）、物理 40.25%；迁移到 $\pi_0$ 仍最强（86.0%/83.5%）；白盒 0.5%/2.75%；patch 5% 面积、$50\times50$。
> - **代码**: https://github.com/yuyi-sd/UPA-RFAS

---

*笔记创建时间: 2026-06-29*
