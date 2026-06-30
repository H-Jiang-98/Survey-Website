---
title: "VLA Models Are More Generalizable Than You Think: Revisiting Physical and Spatial Modeling"
method_name: "FTM / FLA (One-Shot Spatial Adaptation)"
authors: [Weiqi Li, Quande Zhang, Ruifeng Zhai, Liang Lin, Guangrun Wang]
year: 2026
venue: CVPR
tags: [VLA, viewpoint-generalization, one-shot-adaptation, parameter-efficient, LoRA, feature-modulation, spatial-modeling]
zotero_collection: _inbox
image_source: online
arxiv_html: https://arxiv.org/html/2512.02902v2
created: 2026-06-29
---

# VLA Models Are More Generalizable Than You Think: Revisiting Physical and Spatial Modeling

## 元信息

| 项目 | 内容 |
|------|------|
| 作者 | Weiqi Li, Quande Zhang, Ruifeng Zhai, Liang Lin, Guangrun Wang |
| 机构 | 中山大学（Sun Yat-sen University）、广东省大数据分析与处理重点实验室、X-Era AI Lab |
| 会议 | CVPR 2026 |
| 类别 | VLA（Vision-Language-Action） |
| 日期 | 2025-12（arXiv v2） |
| 项目主页 | - |
| 链接 | [arXiv](https://arxiv.org/abs/2512.02902) / [PDF](https://arxiv.org/pdf/2512.02902) |

---

## 一句话总结

> 作者论证 [[VLA]] 在新视角下崩坏的根因是「空间建模」(Spatial Modeling) 的视觉表征漂移而非「物理建模」(Physical Modeling) 能力缺失，并提出两种一次性 (one-shot) 轻量适配——4K 参数的特征仿射调制 (FTM) 与 4.7M 参数的 ViT 低秩适配 (FLA)，把 LIBERO 新视角成功率从 48.5% 拉到 90.8%，参数仅为 LoRA 的 1%。

---

## 核心贡献

1. **重新定位脆弱性来源**: 把 [[VLA]] 的视角/视觉扰动崩溃归因于**空间建模 (Spatial Modeling) 的视觉嵌入系统性漂移**，而非物理建模 (Physical Modeling / visuomotor policy) 的能力不足——即 ViT 视觉编码器与 VLM 动作头之间的**协调失配**，而非缺乏操作能力。
2. **Feature Token Modulation (FTM)**: 仅用一对全局可学习仿射参数 $(\gamma,\beta)$（共 $2D_{\mathrm{ViT}}\approx$ 4K 参数）对视觉 token 做缩放-平移，把 LIBERO 新视角成功率从 48.5% 提到 87.1%，证明「最小干预即可恢复」的假设。
3. **Feature Linear Adaptation (FLA)**: 在 ViT 内部线性层注入 [[LoRA|低秩更新]]（4.7M 参数），达到 90.8% 成功率，**追平甚至略超 467M 参数的 full-model LoRA 微调**（99× 参数压缩），并给出 TV 距离意义下的理论误差界。

---

## 问题背景

### 要解决的问题
预训练 [[VLA]] 模型在分布内 (in-distribution) 表现强，但一旦遇到**未见相机视角、光照变化、背景纹理、传感器噪声**等分布外视觉扰动，成功率断崖式下跌（如 $\pi_{0.5}$ 在 LIBERO 新视角零样本仅 48.5%）。如何**低成本、少数据**地恢复这种泛化能力，是部署到真实动态环境的关键。

### 现有方法的局限
作者把已有鲁棒化路线归为两类，并指出各自痛点（见 Figure 3）：
1. **数据中心 (data-centric)**: 用大规模多视角机器人数据（如 [[Libero-Plus]]）做域随机化，但真实世界采集**昂贵、劳动密集**，无法持续适配。
2. **表征中心 (representation-centric)**: 换上几何感知/3D 一致的视觉骨干（如 [[GeoAware-VLA]] 用 [[VGGT]]），但**需要从头重训 VLA** 以重建视觉特征与动作头的一致性。
3. **LoRA 全量微调**: 调 VLA backbone $\theta$，参数开销大（数百 M）。
4. **Prompt-based 适配**: 拼接可学习 token，**条件作用浅**，参数虽小但realignment 不充分（仅 75.1%）。

### 本文的动机
- 作者提出**反直觉假设**：全局适配是不必要的。视觉扰动主要导致**视觉嵌入空间的系统性漂移**，破坏视觉编码器与 VLM 头之间的协调，而非暴露 visuomotor 能力的缺失。
- 因此**只需适配视觉模块**：要么对输出 token 做仿射调制 (FTM)，要么对 ViT 内部线性层做低秩更新 (FLA)，**冻结整个 VLA 主干**，以 one-shot 方式（甚至单条示范）即可恢复鲁棒性。
- 核心论断：**预训练 VLA 内部潜藏着未被激活的鲁棒性 (untapped robustness)**，问题在「空间对齐」而非「物理能力」。

---

## 方法详解

### 模型架构

本文不提出新 VLA，而是以 **$\pi_{0.5}$** 作为基础策略（也在 $\pi_0$、[[OpenVLA-OFT]] 上验证），在其**冻结的视觉通路**上插入轻量适配模块（见 Figure 1 与 Figure 3 d/e）：

- **输入**: 时刻 $t$ 的视觉观测 $v_t$（单图或多视角图集）+ 语言指令/目标 $l$，合记为 $o_t=(v_t,l)$。
- **基础策略 $\pi_{0.5}$ 三模块**:
  - 视觉编码器 $f_v(\cdot)$（[[SigLIP]] / ViT）：图像 → 视觉 token 嵌入 $\mathbf z$；
  - 语言编码器 $f_\ell(\cdot)$：文本 → 嵌入 $\ell$；
  - 多模态 Transformer 解码器 $g(\cdot)$（带独立 expert 权重）：自回归预测离散动作 token。
- **适配模块**（本文新增、唯一可训练部分）: [[Feature Token Modulation|FTM]]（作用于 $\mathbf z$）或 [[Feature Linear Adaptation|FLA]]（作用于 $f_v$ 内部线性层）。
- **输出**: 离散动作 token 序列 $a_{1:T}$（位置、姿态、夹爪指令）。

适配后，**所有多模态 token 仍由原始冻结的 VLM 与 action expert 处理**生成最终策略——这正是「最小干预」的核心。

### 核心模块

#### 模块1: Feature Token Modulation（FTM，特征 token 调制）

**设计动机**: 受 [[Meta-Learning|元学习]]思想启发——鲁棒行为应通过「有效适配」而非「大规模重训」获得。FTM 把视角漂移视为视觉嵌入空间的**仿射畸变**，用一对全局参数做「再居中 + 再缩放」校正。

**具体实现**:
- 从相机图像取 ViT 输出 token $F\in\mathbb R^{N\times D_{\mathrm{ViT}}}$；
- 引入两个**全局可学习**向量 $\gamma,\beta\in\mathbb R^{D_{\mathrm{ViT}}}$，做逐维仿射变换（见公式4）；
- 与「输入相关 (input-dependent) 的条件调制」不同，$(\gamma,\beta)$ **全局共享**、训练时联合优化、推理时固定，主干全程冻结；
- 仅引入 $2D_{\mathrm{ViT}}\approx$ **4K 参数**。作者将其定位为「受控探针」：若调 $(\gamma,\beta)$ 即足以恢复视角对齐，则证明脆弱性源于嵌入失配而非容量不足。

#### 模块2: Feature Linear Adaptation（FLA，特征线性适配）

**设计动机**: 在 FTM 验证假设后，进一步探究**直接微调 ViT 内部**能否取得相当或更好的效果，且保持参数高效。

**具体实现**:
- 对 ViT 内的线性层 $h=Wx$（$W\in\mathbb R^{d_{\text{out}}\times d_{\text{in}}}$ 冻结）注入 [[LoRA|低秩分解]]更新（见公式5）；
- 低秩矩阵 $A\in\mathbb R^{r\times d_{\text{in}}}$、$B\in\mathbb R^{d_{\text{out}}\times r}$，秩 $r\ll\min(d_{\text{in}},d_{\text{out}})$，仅 $(A,B)$ 可训练；
- rank=16 时仅 **4.7M 参数**；作为对「空间建模」的第二种最小干预，验证内部层调整能否媲美甚至略优于 token 级调制。

### 关键公式与机制

#### 公式1: [[VLA]] 自回归动作分布

$$
P_{\theta}(a_{1:T}\mid o_{1:T})=\prod_{t=1}^{T}P_{\theta}(a_{t}\mid a_{<t},o_{\leq t})
$$

**含义**: 策略 $\theta$ 把观测序列映射为离散动作 token 序列的自回归分布。

**符号说明**:
- $a_t$: $t$ 时刻的动作（一段编码机器人控制的离散 token）
- $o_t=(v_t,l)$: 观测，含视觉 $v_t$ 与语言指令 $l$

#### 公式2: 解码器条件生成

$$
\hat{a}_{t}\sim g\big(a_{<t};\,[\mathbf{z};\ \ell]\big)
$$

**含义**: 多模态解码器 $g$ 以历史动作、融合后的视觉 token $\mathbf z$ 与语言嵌入 $\ell$ 为条件，预测下一动作 token。

**符号说明**:
- $\mathbf z=f_v(v)$: 视觉 token 序列；$\ell=f_\ell(l)$: 语言嵌入
- $[\mathbf z;\ell]$: 拼接后的多模态条件

#### 公式3: 带适配模块的策略

$$
P_{\theta,\phi}(a_{t}\mid a_{<t},o_{\leq t})=g\big(a_{<t};\,[\,\mathcal{A}_{\phi}(f_{v}(v));\ \ell\,]\big)
$$

**含义**: 在冻结策略 $\theta$ 上插入唯一可训练的适配模块 $\mathcal A_\phi$（FTM 或 FLA），仅作用于视觉通路，重对齐视觉 token 后送回原解码器。

**符号说明**:
- $\mathcal A_\phi$: 轻量适配模块（参数 $\phi$ 为 $(\gamma,\beta)$ 或 $(A,B)$）
- $\theta$: 冻结的预训练 VLA 参数

#### 公式4: [[Feature Token Modulation|FTM]] 仿射调制

$$
\hat{F}=(1+\gamma)\odot F+\beta
$$

**含义**: 对 ViT 输出 token 做逐维仿射校正，再居中、再缩放被视角漂移畸变的特征维度，恢复与 VLM 头的对齐。

**符号说明**:
- $F\in\mathbb R^{N\times D_{\mathrm{ViT}}}$: ViT 输出 token；$\hat F$: 调制后特征
- $\gamma,\beta\in\mathbb R^{D_{\mathrm{ViT}}}$: 全局可学习的缩放/平移向量；$\odot$: 逐元素乘
- 「$1+\gamma$」保证初始化为恒等变换（$\gamma=\beta=0$ 时不改变特征）

#### 公式5: [[Feature Linear Adaptation|FLA]] 低秩更新

$$
W^{\prime}=W+\Delta W,\quad \Delta W=BA
$$

**含义**: 对 ViT 内部线性层的权重 $W$ 加一个低秩增量 $\Delta W$，让视觉编码器自身调整特征提取，参数开销极小。

**符号说明**:
- $W$: 冻结的原始权重；$W'$: 适配后权重
- $B\in\mathbb R^{d_{\text{out}}\times r}$，$A\in\mathbb R^{r\times d_{\text{in}}}$，$r\ll\min(d_{\text{in}},d_{\text{out}})$

#### 公式6: 局部 Lipschitz 策略假设 (A1)

$$
d_{\mathrm{TV}}\!\left(g_{a|z},\,g_{a|z^{\prime}}\right)\leq L\|z-z^{\prime}\|
$$

**含义**: 动作分布对视觉 token 是局部 Lipschitz 连续的——视觉 token 的微小变化只引起动作分布（[[Total Variation|TV 距离]]）的有界变化。这是后续理论推导的基础。

**符号说明**:
- $g_{a|z}$: 以视觉 token $z$ 为条件诱导的动作分布
- $d_{\mathrm{TV}}$: 全变差距离；$L$: Lipschitz 常数

#### 公式7: 源域代表 token 与漂移上界 (Theorem 1)

$$
\mathbb{E}_{v\sim\mathcal{D}_{t}}\!\left[d_{\mathrm{TV}}\!\left(g_{a|f_{v}(v)},\,g_{a|z_{s}^{\star}}\right)\right]\leq L\,\mathbb{E}_{v\sim\mathcal{D}_{t}}\left[\|f_{v}(v)-z_{s}^{\star}\|\right]
$$

其中源域代表 token 为

$$
z_{s}^{\star}=\mathbb{E}_{v\sim\mathcal{D}_{s}}\big[f_{v}(v)\big]
$$

**含义**: 性能退化被视觉表征漂移所控制——目标域 token $f_v(v)$ 偏离源域代表 token $z_s^\star$ 越多，动作分布漂移越大。这是「视角脆弱 = 空间漂移」论断的形式化。

**符号说明**:
- $\mathcal D_s,\mathcal D_t$: 源域（训练视角）/目标域（新视角）视觉分布
- $z_s^\star$: 源域视觉 token 的期望（代表性 token）

#### 公式8: 适配可恢复性 (Theorem 2/3) 与组合误差界

适配模块 $A$（仿射或低秩）若能把目标特征拉近源域代表 token：

$$
\mathbb{E}_{v\sim\mathcal{D}_{t}}\|M f_{v}(v)+b-z_{s}^{\star}\|^{2}\leq \epsilon^{2}
$$

则动作分布漂移被 $L\epsilon$ 控制：

$$
\mathbb{E}_{v\sim\mathcal{D}_{t}}\,d_{\mathrm{TV}}\!\left(g_{a|A(f_{v}(v))},\,g_{a|z_{s}^{\star}}\right)\leq L\epsilon
$$

**含义**: 只要轻量适配把目标域特征「拉回」源域流形（均方误差 $\le\epsilon^2$），动作分布与源域的偏差就被 $L\epsilon$ 上界控制。结合 A3（局部仿射/低秩结构假设），FTM/FLA 的表达能力**足以最小化该上界**，从而恢复源域策略——为「小参数适配即可恢复鲁棒性」提供理论支撑。

**符号说明**:
- $M,b$: 仿射适配的等价线性映射与偏置（FTM 对应 $M=\mathrm{diag}(1+\gamma),b=\beta$）
- $\epsilon$: 适配后到源域代表 token 的残差；$L$: Lipschitz 常数

#### 公式9: 低秩截断逼近误差 (Theorem 3)

$$
\|\Delta W^{\star}-\Delta W_{r}\|_{F}^{2}=\sum_{i=r+1}^{d}\sigma_{i}^{2}
$$

**含义**: 用秩 $r$ 的 $\Delta W_r=BA$ 逼近理想校正 $\Delta W^\star$，逼近误差等于被截断的奇异值平方和——解释了为何**增大 rank（16→32）可带来微小提升**（更多奇异值被保留）。

**符号说明**:
- $\sigma_i$: $\Delta W^\star$ 的第 $i$ 个奇异值；$\|\cdot\|_F$: Frobenius 范数

---

## 关键图表

<!-- 图片均使用 arXiv HTML 在线链接，未本地化 -->

### Figure 1: Illustration of Spatial Modeling Adaptation / 空间建模适配总览

![Figure 1a](https://arxiv.org/html/2512.02902v2/x1.png)
![Figure 1b](https://arxiv.org/html/2512.02902v2/x2.png)

**说明**: (a) 两种 one-shot 适配方法 [[Feature Token Modulation|FTM]] 与 [[Feature Linear Adaptation|FLA]] 把视觉表征适配到新空间域，适配后所有多模态 token 仍由**冻结的** VLM 与 action expert 处理；(b) LIBERO 新视角下的「参数效率 vs 性能」——本文以 **4.7M 参数**就超过 $\pi_{0.5}$(LoRA, 467M) 的平均成功率，直观点明核心卖点。

### Figure 2: Sample Rollout across Viewpoints / 跨视角执行序列

![Figure 2](https://arxiv.org/html/2512.02902v2/x3.png)

**说明**: $\pi_{0.5}$ 装备 One-Shot FLA 后的一次 rollout，每行是不同视角随时间（列）的演化，定性展示对视角变化的适应性与时序一致性。

### Figure 3: Comparison of Adaptation Methods / 适配方法对比（核心方法图）

![Figure 3](https://arxiv.org/html/2512.02902v2/x4.png)

**说明**: 五种适配范式对比——(a) LoRA 微调 VLA backbone、(b) 替换视觉骨干、(c) 拼接可学习 prompt 的元学习策略、(d) 本文 **FTM**（对输出 token 做仿射调制）、(e) 本文 **FLA**（对 ViT 内部线性层做低秩更新）。(a)(b) 需大量重训/重对齐，(c) 条件浅，(d)(e) 仅动视觉模块、主干冻结。

### Figure 4: FTM Mechanism & Libero-V Benchmark / FTM 机制与视觉扰动基准

![Figure 4a FTM](https://arxiv.org/html/2512.02902v2/x5.png)
![Figure 4b Libero-V](https://arxiv.org/html/2512.02902v2/x6.png)

**说明**: (a) FTM 的仿射调制示意——对视觉 token 做缩放-平移、骨干冻结；(b) [[Libero-V]](Visual) 基准的可视化，含相机视角、光照、背景纹理、噪声四类视觉扰动样例。

### Figure 5: Success Rate Before vs After Adaptation / 适配前后成功率

![Figure 5](https://arxiv.org/html/2512.02902v2/x7.png)

**说明**: LIBERO 新视角下各方法适配前 (zero-shot) 与适配后的成功率对比，「Before」对应预训练策略零样本表现，直观显示适配带来的大幅跃升。

### Figure 6: Real-World Experimental Setup / 真实世界实验平台

![Figure 6](https://arxiv.org/html/2512.02902v2/x8.png)

**说明**: (a) [[Franka Emika Panda]] 7-DoF 机械臂 + 平行夹爪，经 [[GELLO]] 遥操作框架采集示范，配第三人称静态相机与腕部相机；(b) 用于 one-shot 适配的新相机视角，相对预训练分布引入显著空间偏移，作为 FLA 的测试床。

### Figure 7: Qualitative Results on Real-World Tasks / 真实任务定性结果

![Figure 7](https://arxiv.org/html/2512.02902v2/x9.png)

**说明**: 经 one-shot FLA 适配后五个真实任务的 rollout：(1) 抓红块、(2) 红块叠绿块、(3) 关微波炉门、(4) 按绿色按钮、(5) 拉开顶层抽屉。尽管新视角引入显著视觉差异，适配后策略仍恢复空间 grounding 并闭环精确操作。

### Figure 8: Robustness to Dynamic Objects / 动态物体鲁棒性

![Figure 8](https://arxiv.org/html/2512.02902v2/x10.png)

**说明**: FLA 在人为移动目标（手动扰动）时仍能实时调整轨迹，展示闭环动态适应能力。

### Figure 9: Robustness to Imperfect Demonstrations / 对非完美示范的鲁棒性

![Figure 9](https://arxiv.org/html/2512.02902v2/x11.png)

**说明**: 即使用次优 (sub-optimal) 示范做 one-shot 适配，FLA 仍维持高成功率，说明对单条示范质量不敏感。

### Figure 10: t-SNE of Visual Token Embeddings / 视觉 token 嵌入 t-SNE（关键证据）

![Figure 10](https://arxiv.org/html/2512.02902v2/x12.png)

**说明**: 对 [[SigLIP]] 视觉 token 做 [[t-SNE]] 可视化。(a) 适配前：新视角 token（红）与源域流形（蓝）**完全隔离**，存在严重 embedding drift，解释了冻结策略的灾难性失败——物理建模模块收到落在有效工作区之外的输入；(b) FLA 适配后：目标分布（绿）在保持自身几何一致性的同时被**投影到紧贴源域流形**，恢复隐空间连续性，使冻结 action expert 无需全参数微调即可泛化。这是「空间漂移 = 脆弱根因」论断最直接的定性证据。

### Figure 11: Robustness across Viewpoint Shift Magnitudes / 不同视角偏移幅度

![Figure 11](https://arxiv.org/html/2512.02902v2/x13.png)

**说明**: Small / Medium / Large 三档相机视角偏移下的成功率，展示适配后策略对偏移幅度的稳定性与鲁棒性（对应 Table 2）。

### Figure 12: Adaptation Stability over Training Steps / 适配训练步数稳定性

![Figure 12](https://arxiv.org/html/2512.02902v2/x14.png)

**说明**: 随训练步数变化的成功率曲线，FLA 保持稳定且持续优于基线、不过拟合，说明 one-shot 适配收敛行为良好。

---

### Table 1 / Table 2: LIBERO 新视角成功率（含子套件与偏移幅度）

**Table 1 — LIBERO 四子套件新视角成功率 (SR %)**

| Method | Spatial | Object | Goal | Long | Average |
|--------|---------|--------|------|------|---------|
| GeoAware-VLA (VQ-BeT) | 54.3 | 99.0 | 85.7 | 72.7 | 77.9 |
| GeoAware-VLA (BAKU) | 94.3 | 98.0 | 90.7 | 47.3 | 82.6 |
| OpenVLA-OFT (zero-shot) | 90.0 | 15.7 | 81.7 | 13.7 | 50.3 |
| OpenVLA-OFT-m (FT on Libero-Plus) | 67.1 | 72.4 | 72.0 | 49.2 | 65.2 |
| $\pi_0$ (One-Shot LoRA) | 95.4 | 96.6 | 79.1 | 63.2 | 83.6 |
| $\pi_{0.5}$ (One-Shot LoRA) | 97.2 | 95.3 | 91.4 | 77.1 | 90.3 |
| $\pi_{0.5}$ (One-Shot FTM, ours) | 95.4 | 95.5 | 87.7 | 70.0 | 87.1 |
| **$\pi_{0.5}$ (One-Shot FLA, ours)** | **96.5** | **97.5** | **91.7** | **77.5** | **90.8** |

**Table 2 — 不同视角偏移幅度下成功率 (SR %)**

| Method | Small | Medium | Large | Average |
|--------|-------|--------|-------|---------|
| GeoAware-VLA | 88.0 | 74.5 | 71.2 | 77.9 |
| OpenVLA-OFT-m (FT on Libero-Plus) | 88.1 | 65.5 | 42.1 | 65.2 |
| $\pi_0$ (One-Shot LoRA) | 88.3 | 83.0 | 79.5 | 83.6 |
| $\pi_{0.5}$ (One-Shot LoRA) | **94.8** | **90.5** | 85.6 | 90.3 |
| **Ours (FLA)** | 94.6 | 90.0 | **87.9** | **90.8** |

**说明**: FLA 平均 90.8% 略超 467M 参数的 $\pi_{0.5}$-LoRA(90.3%)，且**偏移越大优势越明显**（Large: 87.9% vs 85.6%）；仅 4K 参数的 FTM 也达 87.1%，超过 $\pi_0$-LoRA(83.6%)。OpenVLA-OFT 零样本仅 50.3%，印证视角脆弱性普遍存在。

### Table 3 / Table 4: Libero-V 视觉扰动鲁棒性与参数效率

**Table 3 — Libero-V 四类视觉扰动成功率 (SR %)**（∗ 取自 Libero-Plus[8]）

| Method | camera | light | texture | noise | Average |
|--------|--------|-------|---------|-------|---------|
| OpenVLA-OFT | 50.3 | 92.7 | 92.3 | 78.6∗ | 78.5 |
| OpenVLA-OFT-m (FT on Libero-Plus) | 65.2 | 94.9 | 93.9 | 89.9∗ | 86.0 |
| $\pi_0$ (One-Shot LoRA) | 83.6 | 90.7 | 86.7 | 87.0∗ | 87.0 |
| $\pi_{0.5}$ (Zero-Shot) | 48.5 | 96.2 | 96.0 | 93.5 | 83.6 |
| $\pi_{0.5}$ (One-Shot LoRA) | 90.3 | 96.5 | 97.2 | 94.5 | 94.6 |
| $\pi_{0.5}$ (One-Shot FTM, ours) | 87.1 | 96.0 | 96.0 | 93.6 | 90.5 |
| **$\pi_{0.5}$ (One-Shot FLA, ours)** | **90.8** | **96.8** | 97.1 | **94.6** | **94.8** |

**Table 4 — 参数量 vs Libero-V 准确率**

| Method | #Params (M) | Libero-V (%) |
|--------|-------------|--------------|
| $\pi_0$ (One-Shot LoRA) | 468.04 | 87.0 |
| $\pi_{0.5}$ (Prompt Learning) | 0.131 | 75.1 |
| $\pi_{0.5}$ (One-Shot LoRA) | 466.96 | 94.6 |
| $\pi_{0.5}$ (One-Shot FTM, ours) | **0.004** | 90.5 |
| **$\pi_{0.5}$ (One-Shot FLA, ours)** | **4.7** | **94.8** |

**说明**: 视角 (camera) 子集是退化最严重处（$\pi_{0.5}$ 零样本仅 48.5%），FLA 把它拉回 90.8%；光照/纹理/噪声本就退化不大。FLA 以 4.7M 参数（LoRA 的 1/99）达到 94.8% 平均，超过 467M 的 LoRA(94.6%)；Prompt Learning 虽参数最少但仅 75.1%，FTM 以 4K 参数即达 90.5%，凸显「直接调视觉 token」比「拼 prompt」更有效。

### Table 5 / Table 8: 不同基础模型与 rank 的 FLA 效率

**Table 5 — FLA vs LoRA（跨基础策略，LIBERO 新视角）**

| Model | #Params | Success Rate (%) |
|-------|---------|------------------|
| $\pi_0$ (One-Shot LoRA) | 468M | 83.6 |
| $\pi_{0.5}$ (One-Shot LoRA) | 467M | 90.3 |
| $\pi_0$ (One-Shot FLA, rank=16) | 4.7M | 84.0 |
| **$\pi_{0.5}$ (One-Shot FLA, rank=16)** | **4.7M** | **90.8** |
| $\pi_{0.5}$ (One-Shot FLA, rank=32) | 9.4M | 91.2 |

**Table 8 — FLA 在不同 VLA 模型上的成功率随 rank 变化 (%)**

| Model | Zero-shot | Rank=8 | Rank=16 | Rank=32 | Rank=64 |
|-------|-----------|--------|---------|---------|---------|
| OpenVLA-OFT (FLA) | 50.3 | **89.0** | 88.0 | 88.0 | 85.6 |
| $\pi_0$ (FLA) | 49.5 | 82.8 | 84.0 | 84.5 | **84.8** |
| $\pi_{0.5}$ (FLA) | 48.5 | 91.05 | 90.8 | **91.2** | 90.6 |

**说明**: FLA 在 $\pi_0$、$\pi_{0.5}$、OpenVLA-OFT 上均把零样本约 48–50% 拉到 84–91%，**与对应 LoRA 持平或略高、参数差 99×**；rank 16→32 仅微涨（90.8%→91.2%，对应公式9 的奇异值截断解释），过大 rank（64）反而略降，说明低秩校正足够且不需大容量。

### Table 6 / Table 7: 基准实现与超参（附录）

**Table 6 — 鲁棒性基准实现摘要**

| 维度 | 实现方式 | 关键参数 |
|------|----------|----------|
| 相机位姿 (Camera Pose) | 直接操纵 MuJoCo 状态 | 轨道角 $\theta$、偏移 $\Delta\mathbf p$、四元数 $q$ |
| 光照 (Lighting) | 程序化 XML 生成 | 漫反射、镜面、方向、阴影 |
| 纹理 (Texture) | 动态资产替换 | PBR 材质（木/石/金属） |
| 传感器噪声 (Sensor Noise) | 实时后处理 | 模糊（运动/高斯/缩放）、雾、噪声 |

其中相机位姿绕末端执行器 (eef) 轨道采样：

$$
\mathbf{p}^{\prime}_{cam}=\begin{bmatrix} r\cos(\theta)+\mathbf{p}_{eef}^{x}\\ r\sin(\theta)+\mathbf{p}_{eef}^{y}\\ \mathbf{p}_{cam}^{z}\end{bmatrix}
$$

**Table 7 — FTM / FLA 超参数**

| 配置 | FTM | FLA |
|------|-----|-----|
| 优化器 | AdamW, clip 1.0 | AdamW, clip 1.0 |
| 动量 | $\beta_1{=}0.9,\beta_2{=}0.95,\epsilon{=}1e{-}8$ | 同左 |
| Weight Decay | $1\times10^{-10}$ | $1\times10^{-10}$ |
| Batch Size | 32 | 32 |
| 训练步数 | 5,000 | 1,500 |
| Warmup | 500 | 500 |
| Peak LR | $5\times10^{-4}$ | $5\times10^{-4}$ |
| Cosine Decay 步数 | 5,000 | 2,000 |
| Min LR | $5\times10^{-5}$ | $5\times10^{-6}$ |
| Image Resize | 224×224 | 224×224 |

**说明**: 两种适配都极轻量——FLA 仅 1,500 步即收敛，真实世界 one-shot 仅训 750 步、单条示范。

---

## 实验

### 数据集 / 基准

| 基准 | 规模/设置 | 特点 | 用途 |
|------|-----------|------|------|
| [[LIBERO]]（新视角） | 4 子套件 Spatial/Object/Goal/Long，全部未见视角 | 单臂操作，测视角泛化 | 适配/测试 |
| [[Libero-V]]（Visual） | 四类扰动：camera/light/texture/noise | 程序化生成的视觉鲁棒性基准 | 测试 |
| 真实世界 | 5 任务，[[Franka Emika Panda]] 7-DoF + 双相机 | 单条示范 one-shot 适配，闭环部署 | 适配/测试 |

### 实现细节

- **基础策略**: $\pi_{0.5}$（也验证 $\pi_0$、OpenVLA-OFT）；视觉编码器为 [[SigLIP]]/ViT，**主干全程冻结**。
- **FTM**: 仅 $(\gamma,\beta)$ 共 $2D_{\mathrm{ViT}}\approx$ 4K 参数；AdamW、lr $5e{-}4$、batch 32、5000 步。
- **FLA**: ViT 线性层注入 LoRA，rank=16 → 4.7M 参数（rank=32 → 9.4M）；1500 步收敛。
- **真实世界协议**: GELLO 采单条新视角示范，FLA rank=32、batch 32、训 **750 步**，闭环双相机部署。

### 关键实验结论

- **视角泛化（Table 1/2）**: $\pi_{0.5}$ 零样本 48.5% → FTM 87.1% → FLA 90.8%，超过 467M-LoRA(90.3%)，偏移越大优势越大。
- **视觉扰动（Table 3/4）**: Libero-V 上 FLA 94.8% 平均，4.7M 参数追平 467M-LoRA(94.6%)；FTM 仅 4K 参数达 90.5%。
- **跨模型/rank（Table 5/8）**: FLA 在 $\pi_0$/$\pi_{0.5}$/OpenVLA-OFT 上一致有效，rank 16–32 即够，过大反降。
- **真实世界（Fig 7/8/9）**: 单条示范、750 步适配后五任务闭环成功，且对动态目标扰动与非完美示范鲁棒。
- **机制证据（Fig 10）**: t-SNE 显示适配前源/目标流形完全隔离，FLA 后目标流形被投影紧贴源域，恢复隐空间连续性——验证「空间漂移」假设。
- **理论（§8）**: 在局部 Lipschitz (A1)、任务语义不变 (A2)、局部仿射/低秩结构 (A3) 三条温和假设下，给出动作分布漂移 $\le L\epsilon$ 的组合误差界，论证小参数适配足以恢复源域策略。

---

## 批判性思考

### 优点
1. **诊断 + 方法 + 理论闭环**: 先提出「空间建模漂移而非物理建模缺失」的诊断，再用 FTM（探针）与 FLA（实用方法）验证，最后用 t-SNE（Fig 10）与 TV 距离误差界双重支撑，论证链条完整、说服力强。
2. **极致参数效率**: FTM 仅 4K、FLA 仅 4.7M 参数即追平/超过 467M 的 full LoRA，99× 压缩；one-shot（单条示范、750 步）落地成本极低。
3. **跨模型与真机验证**: 在 $\pi_0$/$\pi_{0.5}$/OpenVLA-OFT 与真实 Franka 平台上都成立，结论不局限于单一模型或仿真。

### 局限性
1. **聚焦视角/视觉扰动，未覆盖物理域偏移**: 论断「物理建模无需调」建立在 A2（任务语义不变）之上；若新域同时改变物体动力学、接触、形变，FLA 是否够用存疑。
2. **基线与扰动设置偏可控**: 主要对比 LoRA 与 GeoAware-VLA，真实世界仅 5 个桌面任务、单一新视角；缺更强 OOD（大幅遮挡、跨本体）与更多 SOTA 鲁棒方法的横评。
3. **「源域代表 token $z_s^\star$ 用期望」较粗**: 理论用单一均值 token 代表整个源分布，对多模态源域可能过简；rank 选择、$z_s^\star$ 估计的敏感性分析仅有 rank 消融，仍偏经验。

### 潜在改进方向
1. 把适配从「视觉编码器」推广到联合考虑物理域偏移（动力学/接触），检验「空间 vs 物理」划分在更强 OOD 下的边界。
2. 用更细的表征对齐度量（如 CKA、流形距离）量化 t-SNE 之外的对齐质量，并把「源域代表」从单一均值扩展为多原型/分布匹配。
3. 探索在线/持续 one-shot 适配（部署中不断遇到新视角时自动累积 $(\gamma,\beta)$ 或低秩库），向真实开放环境靠拢。

### 可复现性评估
- [ ] 代码开源（论文未给出公开仓库链接）
- [ ] 预训练模型（基于公开 $\pi_0$/$\pi_{0.5}$/OpenVLA-OFT，但适配权重未声明 release）
- [x] 训练细节完整（附录 Table 6/7 给出基准实现与全部超参）
- [x] 数据集可获取（LIBERO 公开；Libero-V 扰动方式在附录详述可复现）

---

## 速查卡片

> [!summary] VLA Models Are More Generalizable Than You Think (FTM / FLA)
> - **核心**: VLA 视角脆弱性源于「空间建模」视觉嵌入漂移而非「物理建模」能力缺失；冻结主干、只轻量适配视觉模块即可恢复鲁棒性。
> - **方法**: FTM = 对 ViT token 做全局仿射 $(1+\gamma)\odot F+\beta$（4K 参数）；FLA = 对 ViT 线性层注入 LoRA $\Delta W=BA$（4.7M 参数）；one-shot、主干冻结。
> - **结果**: LIBERO 新视角 48.5%→FTM 87.1%→FLA 90.8%（超 467M-LoRA 的 90.3%，参数 1/99）；Libero-V 94.8%；真机单条示范 750 步五任务闭环成功；t-SNE + TV 误差界双重佐证。
> - **代码**: 论文未公开仓库链接。

---

*笔记创建时间: 2026-06-29*
