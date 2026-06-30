---
title: "MergeVLA: Cross-Skill Model Merging Toward a Generalist Vision-Language-Action Agent"
method_name: "MergeVLA"
authors: [Yuxia Fu, Zhizhen Zhang, Yuqi Zhang, Zijian Wang, Zi Huang, Yadan Luo]
year: 2026
venue: CVPR
tags: [VLA, model-merging, LoRA, task-arithmetic, cross-attention, action-expert, test-time-routing, multi-task]
zotero_collection: _inbox
image_source: online
arxiv_html: https://arxiv.org/html/2511.18810v2
created: 2026-06-29
---

# MergeVLA: Cross-Skill Model Merging Toward a Generalist Vision-Language-Action Agent

## 元信息

| 项目 | 内容 |
|------|------|
| 作者 | Yuxia Fu, Zhizhen Zhang（共同一作）, Yuqi Zhang, Zijian Wang, Zi Huang, Yadan Luo |
| 机构 | UQMM Lab, The University of Queensland（昆士兰大学） |
| 会议 | CVPR 2026 |
| 类别 | VLA（Vision-Language-Action）/ 模型合并（Model Merging） |
| 日期 | 2025-11（arXiv v2） |
| 项目主页 | https://mergevla.github.io/ |
| 链接 | [arXiv](https://arxiv.org/abs/2511.18810) / [Project](https://mergevla.github.io/) |

---

## 一句话总结

> 针对"把多个单技能 [[VLA]] 专家直接合并会跌到近乎 0% 成功率"的难题，提出面向可合并性设计的 MergeVLA：用任务掩码稀疏激活合并后的 [[LoRA]] 参数、把动作专家改成纯交叉注意力结构、再用免训练的测试时任务路由，使一个合并模型在多技能/跨本体/跨环境下逼近甚至超过各自微调的专家。

---

## 核心贡献

1. **诊断 VLA 不可合并的两大根因**：通过对 VLA 微调可学习参数的细粒度分解，指出 (1) VLM 主干的 [[LoRA]] 适配器被微调推向高度发散、互相冲突的任务专属方向（合并 4 个任务时"自私参数"已超 75%）；(2) 从零训练的[[Action Expert|动作专家]]通过[[Self-Attention|自注意力]]反馈在层间积累强任务依赖，使任务信息跨层弥散、破坏模块化，越深的层越无法合并。
2. **面向可合并性的架构 MergeVLA**：用**任务掩码（task mask）**对合并后的 LoRA 做稀疏激活以化解冲突；把动作专家改为**仅交叉注意力（cross-attention-only）+ sigmoid 门控**，让大多数层可用简单权重平均合并，仅保留深层"专家头（expert head）"不合并；附带带来 OOD 鲁棒性提升（比 VLA-Adapter 高 13.4%）。
3. **免训练测试时任务路由**：任务身份未知时，从初始观测的隐状态出发，用合并动作专家 value 投影的主成分子空间度量响应强度，无监督地选出任务掩码与专家头；在 [[LIBERO]]、[[LIBERO-Plus]]、[[RoboTwin]] 与真实 SO-101 机械臂上，合并模型达到甚至超过单独微调专家的性能。

---

## 问题背景

### 要解决的问题
现实中的通用具身智能体必须支持**多技能、多本体、多环境**，理想路径是把许多独立微调好的 VLA 专家**无需重训地合并**成单一统一策略。但作者发现：把在不同操作任务上微调的 VLA 专家直接用现有[[模型合并|model merging]]方法合并，得到的模型**成功率近乎为 0**。这引出核心问题——**到底是什么阻止 VLA 在一个模型内掌握多种技能？**

### 现有方法的局限
- **模型合并方法在 LLM/VLM 上有效，但搬到 VLA 失灵**：[[Task Arithmetic]]（TA）、[[TIES]]、Iso-CTS 等在语言/视觉模型上能整合多专家，但应用于 VLA 时合并模型直接崩溃，说明 VLA 微调诱导出了 VLM 合并中罕见的、跨任务不兼容的结构性专化。
- **现有 VLA 的多任务能力依赖联合训练**（OpenVLA、$\pi_0$、$\pi_{0.5}$、VLA-Adapter），需要把所有任务数据放在一起重训，效率低、不可增量扩展。
- **双系统紧耦合阻碍合并**：带动作专家的 VLA（如 VLA-Adapter）把 VLM 与动作专家紧耦合，使整体难以模块化重组。

### 本文的动机
作者通过参数空间与架构行为的系统分析（见 Figure 3），定位出两个互补的失败模式，并据此"反向设计"一个**天生可合并**的架构：
- **VLM 侧**：任务专属 [[LoRA]] 更新激活的通道几乎互不相交，朴素平均会重新激活无关甚至矛盾参数、污染共享视觉-语言子空间 → 用**任务掩码**只保留与该任务一致的合并参数。
- **动作专家侧**：从零训练 + 自注意力让深层块病态地专化于单个任务、参数距离随深度暴增 → **移除自注意力**、改纯交叉注意力，强迫动作专家依赖鲁棒的共享 VLM 特征；仅把无法合并的深层"专家头"留作每任务独立。
- **推理侧**：混合任务评测中任务身份未知，许多方法靠手工先验（任务 id / 专属 prompt），去掉就崩 → 设计**免训练测试时路由**自动选技能。

---

## 方法详解

### 模型架构

MergeVLA 基于 [[VLA-Adapter]] 的双系统结构改造而来，核心是把它改成"对合并友好"的形态（见 Figure 1、Figure 2）：
- **输入**：第三人称视角图像 $\mathbf{I}^v_t$ + 腕部相机图像 $\mathbf{I}^w_t$ + 任务指令 $L$（外加本体状态 proprioception）。
- **Backbone**：[[Qwen2.5]]-0.5B 作为视觉-语言主干，用 [[LoRA]] 微调（rank 32）。
- **核心模块**：① VLM 主干上的**稀疏激活 LoRA + 任务掩码** $\mathbf{S}_m$；② **仅交叉注意力 + sigmoid 门控**的动作专家（含可合并浅层 + 不合并的[[Expert Head|专家头]] $\mathbf{H}^{l\to L}$）；③ **测试时任务路由器**。
- **输出**：连续[[Action Chunking|动作块]]（real-world 为 6-DoF 关节动作）。
- **总参数**：合并后约 **0.70B**（含 LLM 主干与动作专家；对比 OpenVLA 单个就 7B）。

**任务形式化**：给定一组单技能模仿学习数据集 $\mathfrak{D}=\{\mathcal{D}_m\}_{m=1}^M$，每个 $\mathcal{D}_m$ 对应一个操作任务，微调得到 $M$ 套任务专属权重 $\{\Theta_1,\dots,\Theta_M\}$，合并目标是在**不重训**的前提下把它们统一成单一智能体 $\Theta_{\mathrm{merge}}$。

### 核心模块

#### 模块1: VLM 主干的任务冲突与稀疏掩码（Sec 4.1，对应 Q1）

**设计动机**：标准合并构造单一全局更新 $\tau_{\mathrm{merge}}$，但 VLA 各任务的 LoRA 更新方向高度发散、互相矛盾，单一更新无法直接用（导致 0% 成功率）。解决思路是**为每个任务保留一份从全局合并向量中"挑出对它有益"的稀疏视图**。

**具体实现**：
- 定义任务向量 $\tau_m=\Theta_m-\Theta_0$（微调权重减预训练权重）；多数 data-free 方法用合并算子 $\mathcal{R}(\cdot)$ 得到单一 $\tau_{\mathrm{merge}}$（公式1）。
- 用任务专属**二值掩码** $\mathbf{S}_m$ 对合并向量做逐元素门控（公式2），只激活对任务 $m$ 有益、抑制误导其他任务的分量。
- 掩码通过**参数级一致性检验**构造（公式3）：仅当某参数的任务向量 $\tau_m$ 既"显著"又"主导于其与 $\tau_{\mathrm{merge}}$ 的残差差异"时保留。该形式与任务向量压缩方法同源。
- **副作用**：加掩码会让部分 LoRA 参数**回退向预训练权重**，提升合并稳定性——这与 ReVLA 观察到的"VLA 微调会覆盖预训练视觉知识"互为印证，掩码相当于保留了预训练视觉-语言表征。

#### 模块2: 重设计动作专家以实现可合并（Sec 4.2，对应 Q2）

**设计动机**：诊断表明，仅稳住 VLM 还不够——即便 VLM 完美合并，直接平均 VLA-Adapter 那种动作专家仍是 **0% 成功率**。根因是动作专家从零训练且含自注意力，深层块参数距离随深度暴增（Figure 3 右）。

**具体实现**（相对 VLA-Adapter 的两处关键改动）：
- **移除自注意力，仅保留交叉注意力**：从零训练的自注意力会发展出强烈、不可调和的任务专属偏置；去掉后强迫专家依赖鲁棒、共享的 VLM 特征。VLA-Adapter 原块含 1 个自注意力（作用于块输入 $\mathbf{x}^i$）、2 个交叉注意力（分别条件于 VLM 的任务隐状态 $\mathbf{h}^i_{\text{T}}$ 与动作隐状态 $\mathbf{h}^i_{\text{A}}$）和 FFN。
- **门控由 $\tanh$ 换 $\mathrm{sigmoid}$**：原 $\tanh$ 门可用负激活抑制 VLM 信号、迫使专家依赖自身从零学的（任务专属）参数；$\mathrm{sigmoid}$ 保证 VLM 信息始终被保留与平衡。
- **效果**：仅这两处改动就在 OOD 的 LIBERO-Plus 上把成功率提升 **13.4%**，印证新设计更好地利用了 VLM 的鲁棒性。
- **按专化层级合并**：动作专家从零训练、各专家间**无共享初始化**，故任务向量类方法不适用，改用**简单权重平均**。浅层平均效果惊人地好，但深层参数差异骤增、合并失败——这些发散层统称**专家头** $\mathbf{H}^{l\to L}$（多数情况下 $l=L$，即仅最后一块需单独保留）。作者假设回归式训练目标让每个专家头高度专化于其任务的动作分布，微小差异即致整条轨迹失败，故**专家头不合并、各任务用各自的头**。

#### 模块3: 测试时任务路由（Sec 4.3，对应 Q3）

**设计动机**：任务身份已知时只需手动选 $\mathbf{S}_m$ 与 $\mathbf{H}_m^{l\to L}$；但混合任务推理时身份未知，需仅凭观测动态选组件，实现 joint-task 级跨技能能力。免训练。

**具体实现**：
- 用每个掩码作用于合并权重，得到 $M$ 个掩码 VLM 变体（公式5），各产出隐状态 $[\mathbf{h}_{\text{T}}^l,\mathbf{h}_{\text{A}}^l]$ 送入动作专家第 $l$ 块。
- **关键设计：用 value 子空间而非 Q/K**——作者假设 $\mathbf{Q},\mathbf{K}$ 主导注意力选择、对输入缩放敏感、易塌缩到任务专属子空间；value 子空间直接编码写入隐状态的任务相关信息，更稳定、更具判别力（消融见 Table 5）。
- 对两条交叉注意力路径的 value 投影矩阵做 [[SVD]]（公式6），各取前 $k_r$ 个右奇异向量构成主内容子空间 $\mathbf{P}_{\text{T}}^l,\mathbf{P}_{\text{A}}^l\in\mathbb{R}^{k_r\times d}$。
- 把各掩码 VLM 的隐状态投影到子空间度量**激活强度**（公式7），合成得分 $\mathbf{r}=\tfrac12(\mathbf{r}_{\text{T}}+\mathbf{r}_{\text{A}})$，softmax 后取 $\arg\max$ 得任务索引 $m^*$，激活对应掩码与专家头。
- **效率**：仅在 $t{=}0$ 用初始观测路由一次即可，全程固定，避免重复路由；需维护 $M$ 套掩码与动作头，但额外计算/参数开销很小。

### 关键公式与机制

#### 公式1: [[模型合并|单一全局合并更新]]

$$
\tau_{\mathrm{merge}}=\alpha\,\mathcal{R}\!\left(\{\tau_m\}_{m=1}^{M}\right),\qquad \Theta_{\mathrm{merge}}=\Theta_0+\tau_{\mathrm{merge}}
$$

**含义**：多数 data-free 合并把各任务的任务向量 $\tau_m$ 经合并算子 $\mathcal{R}$ 聚成单一更新，再叠加到预训练权重上。

**符号说明**：
- $\tau_m=\Theta_m-\Theta_0$：任务 $m$ 的任务向量（微调减预训练）
- $\mathcal{R}(\cdot)$：合并算子（如 TA 的平均、TIES 的符号消解）
- $\alpha$：缩放因子（默认 $\alpha=1$）；$\Theta_0$：预训练权重

#### 公式2: [[任务掩码]]稀疏激活

$$
\Theta_{\mathrm{merge}}^{(m)}=\Theta_0+\mathbf{S}_m\odot\tau_{\mathrm{merge}}
$$

**含义**：不再用单一全局更新，而是对每个任务用其专属掩码 $\mathbf{S}_m$ 从合并向量中挑出有益分量，化解跨任务冲突。

**符号说明**：
- $\mathbf{S}_m$：任务 $m$ 的二值掩码；$\odot$：逐元素乘
- $\Theta_{\mathrm{merge}}^{(m)}$：为任务 $m$ 激活的合并权重

#### 公式3: 掩码的一致性检验构造

$$
\mathbf{S}_m=\mathbb{I}\!\left[\,|\tau_m|>\lambda\,|\tau_{\mathrm{merge}}-\tau_m|\,\right]
$$

**含义**：仅当任务向量 $\tau_m$ 的幅值既显著、又主导于它与合并向量的残差差异时保留该参数，表示它与整体合并一致且对任务 $m$ 有正贡献。

**符号说明**：
- $\mathbb{I}[\cdot]$：指示函数（真为 1，否则 0）
- $\lambda$：掩码比例 / 容忍度阈值（默认 $\lambda=0.6$）；越大越稀疏、越偏向预训练权重

#### 公式4: 自私参数比例

$$
\text{ratio}_{\mathrm{selfish}}=\frac{1}{N}\sum_{i=1}^{N}\mathbb{I}\!\left[\sum_{m=1}^{M}(\mathbf{S}_m)_i=1\right]
$$

**含义**：统计被恰好一个任务掩码保留的参数占比，量化"参数自私性"。LIBERO 上合并 2→4 个任务时该比例稳定升至约 75%，说明大多数参数被单一任务独占，凸显任务掩码的必要性。

**符号说明**：
- $N$：参数总数；$(\mathbf{S}_m)_i$：任务 $m$ 掩码在第 $i$ 个参数上的取值
- 内层指示：当且仅当该参数仅被一个任务保留时计 1

#### 公式5: 路由前的掩码 VLM 变体

$$
\Theta^{(m)}_{\mathrm{merge}}=\Theta_0+\mathbf{S}_m\odot\tau_{\mathrm{merge}},\qquad m=1,\dots,M
$$

**含义**：对合并权重逐一施加 $M$ 个任务掩码，得到 $M$ 个掩码 VLM，用于测试时路由打分（与公式2同形，但此处面向全部候选任务）。

#### 公式6: value 投影的 SVD

$$
\mathbf{V}_{\text{T}}^{\,l}=\mathbf{L}_{\text{T}}^{\,l}\,\mathbf{\Sigma}_{\text{T}}^{\,l}\,(\mathbf{R}_{\text{T}}^{\,l})^{\top},\qquad
\mathbf{V}_{\text{A}}^{\,l}=\mathbf{L}_{\text{A}}^{\,l}\,\mathbf{\Sigma}_{\text{A}}^{\,l}\,(\mathbf{R}_{\text{A}}^{\,l})^{\top}
$$

**含义**：对第 $l$ 块两条交叉注意力路径（任务路径 T、动作路径 A）的 value 投影矩阵做奇异值分解，取前 $k_r$ 个右奇异向量构成主内容子空间 $\mathbf{P}_{\text{T}}^l,\mathbf{P}_{\text{A}}^l$。

**符号说明**：
- $\mathbf{R}^\top$ 的前 $k_r$ 行即主成分；$k_r$ 默认 8
- 用 value 而非 Q/K：value 直接编码写入隐状态的任务相关行为语义，更具判别力

#### 公式7: 任务激活强度打分

$$
r_{\text{T},m}=\big\|\mathbf{P}_{\text{T}}^{\,l}\,\mathbf{h}^{\,l}_{\text{A},m}\big\|_2,\qquad
r_{\text{A},m}=\big\|\mathbf{P}_{\text{A}}^{\,l}\,\mathbf{h}^{\,l}_{\text{T},m}\big\|_2
$$

**含义**：把第 $m$ 个掩码 VLM 的隐状态投影到主子空间，用投影范数度量该任务的激活强度；合成 $\mathbf{r}=\tfrac12(\mathbf{r}_{\text{T}}+\mathbf{r}_{\text{A}})$ 后 softmax，取 $\arg\max_m p_m$ 选任务。

**符号说明**：
- $\mathbf{h}^l_{\text{A},m},\mathbf{h}^l_{\text{T},m}$：第 $m$ 个掩码 VLM 在第 $l$ 块的动作/任务隐状态
- $\|\cdot\|_2$：L2 范数（投影到主成分上的能量）

---

## 关键图表

<!-- 图片均使用 arXiv HTML 在线链接，未本地化 -->

### Figure 1: Comparison of VLA Structures / 不同 VLA 结构对比

![Figure 1](https://arxiv.org/html/2511.18810v2/x1.png)

**说明**：三类 VLA 结构对比。[[OpenVLA]] 用标准 VLM 做基于 token 的动作生成；[[VLA-Adapter]] 加入含交叉注意力 + 自注意力的动作专家；MergeVLA **移除不可合并的自注意力层**简化设计，使除专家头外的所有组件都能有效合并。这张图直观点明本文"删自注意力换可合并性"的核心改动。

### Figure 2: MergeVLA Architecture Overview / 整体架构

![Figure 2](https://arxiv.org/html/2511.18810v2/x2.png)

**说明**：MergeVLA 总架构。(1) 对所有合并的 LoRA 模块施加**任务掩码**，选择性激活对任务有用的合并参数、抑制误导其他任务的参数，解决微调 VLM 中破坏性 LoRA 干扰；(2) 动作专家重设计为**仅交叉注意力块 + sigmoid 门控**以保留并依赖鲁棒 VLM 特征，大多数块可合并，仅深层"专家头"不合并。是理解三大模块如何协同的主图。

### Figure 3: Diagnosis of Non-Mergeability / 不可合并性诊断

![Figure 3](https://arxiv.org/html/2511.18810v2/x3.png)

**说明**：本文最关键的诊断图。**左**：TA 与 TIES 掩码在合并不同任务数时的"自私比例"（按公式4），合并 4 任务时已约 75%，说明参数被单任务独占、冲突严重；**右**：所有动作专家两两之间逐块的平均相对 L2 距离——浅层尚对齐，深层距离暴增，揭示自注意力让深层病态专化、不可合并。两图分别对应 Q1 与 Q2 的根因证据。

### Figure 4: Seven Perturbation Types in LIBERO-Plus / 七类扰动

![Figure 4](https://arxiv.org/html/2511.18810v2/x5.png)

**说明**：[[LIBERO-Plus]] 用于评测视觉/语言分布偏移鲁棒性的七类扰动（S1 背景纹理、S2 相机视角、S3 语言指令、S4 光照、S5 物体布局、S6 机器人状态、S7 传感器噪声），对应 Table 2 的列。

### Figure 5: RoboTwin Cross-Embodiment Setup / 跨本体实验设置

![Figure 5](https://arxiv.org/html/2511.18810v2/x6.png)

**说明**：[[RoboTwin]] 2.0 环境，含三种机器人本体（Aloha-Agilex、ARX-X5、Piper）与一组双臂操作任务，用于跨本体跨技能评测，对应 Table 3。

### Figure 6: Real-World SO-101 Setup / 真实机械臂设置

![Figure 6](https://arxiv.org/html/2511.18810v2/x7.png)

**说明**：真实 SO-101 机械臂的三项立方体操作任务（Pick & Place、Push Cube、Stack Cube），双 RGB 视角（顶视 + 腕部），对应 Table 4。

### Figure 7: Ablation & Analysis of λ / 掩码比例消融

![Figure 7a](https://arxiv.org/html/2511.18810v2/x8.png)
![Figure 7b](https://arxiv.org/html/2511.18810v2/x9.png)

**说明**：掩码比例 $\lambda$ 的分析。(a) 四个 LIBERO 任务在不同 $\lambda$ 下的掩码激活比例，LIBERO-Spatial 始终最高（权重贡献占优）；(b) LIBERO-Long 成功率随 $\lambda$ 变化：$\lambda$ 太小（0.2）激活过多参数→任务干扰严重甚至完全失败，$\lambda\in[0.6,0.9]$ 时成功率超 70%。说明适度稀疏才能在任务专属与合并向量间取得平衡。

### Figure 8: Progressive Block-wise Merging of OpenVLA LM / 逐块合并语言模型（附录）

![Figure 8](https://arxiv.org/html/2511.18810v2/x10.png)

**说明**：在 LIBERO-Spatial 上逐步合并 OpenVLA 语言模型的前 $k$ 个块（用 Iso-CTS）。仅合并浅层（≤8 块）仍有约 80% 成功率，但随合并块数增加性能骤降，超过 21 块即完全失败，验证"任务冲突随层深增长、深层任务专化更强"。

### Figure 9: Mask Active Ratios (Vision vs LM) / 视觉与语言分支掩码激活比例（附录）

![Figure 9](https://arxiv.org/html/2511.18810v2/x11.png)

**说明**：各 LIBERO 任务套件在视觉主干与语言模型上的掩码激活比例（用 [[Task Arithmetic]]，$\lambda=0.6$），佐证不同组件/任务的参数贡献分布差异。

### Table 1: LIBERO Results / LIBERO 四套件结果

$\mathbf{S}$ 表示合并时使用任务掩码；灰底行为各自任务上的单任务微调（合并的上界参考）；Params(B) 为评测全部四任务所需总参数量。

| Method | Merge Method | Merge Part | Params (B) | Spatial | Object | Goal | Long | Avg. |
|--------|--------------|------------|-----------|---------|--------|------|------|------|
| *OpenVLA*（单任务微调） | - | - | 7×4 | 84.7 | 88.4 | 79.2 | 53.7 | 76.5 |
| *VLA-Adapter*（单任务微调） | - | - | 0.68×4 | 99.6 | 99.6 | 98.2 | 96.4 | **98.5** |
| *MergeVLA*（单任务微调） | - | - | 0.68×4 | 98.0 | 98.6 | 95.0 | 95.0 | 96.7 |
| OpenVLA | TA | Vision Backbones | 7×4 | 56.6 | 58.0 | 55.6 | 6.6 | 44.2 |
| OpenVLA | TA | All | 7 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| OpenVLA | TA + $\mathbf{S}$ | All | 7 | 74.2 | 82.6 | 68.8 | 24.0 | 62.4 |
| VLA-Adapter | TA | All | 0.68 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| VLA-Adapter | TA + $\mathbf{S}$ | All | 0.68 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| VLA-Adapter | TA + $\mathbf{S}$ | Except $\mathbf{H}^{L\to L}$ | 0.70 | 50.2 | 34.6 | 0.0 | 7.4 | 23.1 |
| MergeVLA$_{\text{EMR}}$ | EMR | - | - | 96.0 | 63.2 | 62.0 | 40.6 | 65.5 |
| MergeVLA$_{\text{TSV}}$ | TSV + $\mathbf{S}$ | - | - | 99.4 | 97.8 | 74.4 | 54.8 | 81.6 |
| MergeVLA$_{\text{KnOTS}}$ | KnOTS + $\mathbf{S}$ | - | - | 96.8 | 98.8 | 84.8 | 71.4 | 88.0 |
| MergeVLA$_{\text{TA}}$ | TA + $\mathbf{S}$ | Except $\mathbf{H}^{L\to L}$ | 0.70 | 98.0 | 98.8 | 85.4 | 76.6 | 89.7 |
| MergeVLA$_{\text{WUDI}}$ | WUDI + $\mathbf{S}$ | - | - | 97.6 | 98.2 | 85.6 | 78.2 | 89.9 |
| **MergeVLA$_{\text{TIES}}$** | **TIES + $\mathbf{S}$** | - | - | 94.8 | 94.6 | **91.8** | **79.4** | **90.2** |

**说明**：朴素把 OpenVLA/VLA-Adapter 整体合并（TA, All）全是 **0%**；仅合并视觉 + projector 升到 44.2%，加掩码升到 62.4% 但极不均衡。VLA-Adapter 即使加掩码仍 0%，除非排除最终动作块（仅 23.1%）。MergeVLA 配 TIES+$\mathbf{S}$ 达 **90.2%**，比单任务微调（96.7%）仅低 6.5%，且参数轻量（0.70B vs 单 OpenVLA 7B），验证结构改动既保留原能力又实现可合并。

### Table 2: LIBERO-Plus Robustness / 视觉与语言偏移下的鲁棒性

各列为七类扰动（S1–S7，见 Figure 4），数字为 4 套件平均成功率（%）；灰底为各自任务的单任务微调上界。

| Method | S1 | S2 | S3 | S4 | S5 | S6 | S7 | Avg. |
|--------|----|----|----|----|----|----|----|------|
| *OpenVLA*（单任务） | 34.8 | 0.8 | 23.0 | 8.1 | 28.5 | 3.5 | 15.2 | 16.3 |
| *$\pi_0$*（单任务） | 81.4 | 13.8 | 58.8 | 85.0 | 68.9 | 6.9 | 79.0 | 56.3 |
| *VLA-Adapter*（单任务） | 76.6 | 36.4 | 73.8 | 71.0 | 70.2 | 37.4 | 57.2 | 59.0 |
| ***MergeVLA*（单任务）** | **92.7** | **62.4** | 75.7 | **92.7** | **73.7** | **46.4** | 74.7 | **72.4** |
| VLA-Adapter（合并） | 15.7 | 6.6 | 17.6 | 11.2 | 15.0 | 4.1 | 7.1 | 10.8 |
| MergeVLA$_{\text{TSV}}$（合并） | 64.3 | 44.8 | 58.8 | 72.9 | 59.4 | 30.4 | 52.7 | 53.5 |
| MergeVLA$_{\text{TA}}$（合并） | 78.2 | 53.1 | 68.4 | 79.0 | 65.0 | 34.6 | 62.7 | 61.6 |
| **MergeVLA$_{\text{TIES}}$（合并）** | 85.7 | 50.7 | 66.0 | 84.2 | 68.1 | 30.3 | 66.0 | **62.5** |

**说明**：单任务设定下 MergeVLA 鲁棒性（72.4%）远超 OpenVLA/$\pi_0$/VLA-Adapter，源于其架构保留了预训练 VLM 的内在鲁棒性；合并设定下，TA/TIES 合并的 MergeVLA（61.6%/62.5%）甚至**超过其它方法的单任务版本**，说明鲁棒性可跨任务迁移与保留。

### Table 3: RoboTwin Cross-Embodiment Results / 跨本体结果

$\mathbf{T}_1$=Place Container Plate，$\mathbf{T}_2$=Handover Block，$\mathbf{T}_3$=Open Microwave；每结果为 50 试成功率（%）。

**Setting A：跨本体、单任务（三机器人都做 $\mathbf{T}_1$）**

| Method | Aloha | ARX | Piper | Avg. |
|--------|-------|-----|-------|------|
| Single-task Finetuned | 90.0 | 90.0 | 84.0 | 88.0 |
| MergeVLA$_{\text{TA}, \mathbf{H}^{(L-1)\to L}}$ | 86.0 | 82.0 | 68.0 | 78.7 |
| **MergeVLA$_{\text{TIES}, \mathbf{H}^{(L-1)\to L}}$** | 88.0 | **92.0** | **86.0** | **88.7** |

**Setting B：跨本体、跨任务（Aloha→$\mathbf{T}_1$、ARX→$\mathbf{T}_2$、Piper→$\mathbf{T}_3$）**

| Method | $\mathbf{T}_1$ Aloha | $\mathbf{T}_2$ ARX | $\mathbf{T}_3$ Piper | Avg. |
|--------|------|------|------|------|
| Single-task Finetuned | 90.0 | 46.0 | 92.0 | 76.0 |
| MergeVLA$_{\text{TA}, \mathbf{H}^{(L-1)\to L}}$ | 80.0 | 0.0 | 66.0 | 48.7 |
| MergeVLA$_{\text{TA}, \mathbf{H}^{(L-2)\to L}}$ | 82.0 | 0.0 | 66.0 | 49.3 |
| MergeVLA$_{\text{TIES}, \mathbf{H}^{(L-1)\to L}}$ | 90.0 | 0.0 | 88.0 | 59.3 |
| **MergeVLA$_{\text{TIES}, \mathbf{H}^{(L-2)\to L}}$** | 88.0 | **38.0** | 86.0 | **70.7** |

**说明**：跨本体合并对路由更难。Setting A 中 TIES + 路由 $\mathbf{H}^{(L-1)\to L}$（88.7%）可保持单任务策略性能；Setting B 中，仅留最后一块作专家头不够——尤其 Handover Block 需双臂协调、动作空间冲突更强，需 TIES + $\mathbf{H}^{(L-2)\to L}$（保留更多块为专家头）才能恢复到 70.7%。体现"本体差异越大、专家头需越深"。

### Table 4: Real-World SO-101 Results / 真实机械臂结果

每任务 20 次 rollout 的成功率（%）。

| Method | Pick & Place | Push Cube | Stack Cube | Avg. |
|--------|--------------|-----------|------------|------|
| Single-task finetune | 90.0 | 85.0 | 95.0 | 90.0 |
| MergeVLA$_{\text{TA}}$ | 70.0 | 70.0 | 60.0 | 66.7 |
| **MergeVLA$_{\text{TIES}}$** | **90.0** | **90.0** | 90.0 | **90.0** |

**说明**：真实部署中 TIES 合并（90.0%）与单任务微调（90.0%）持平，且 Pick & Place / Push Cube 用了训练未见颜色的立方体（视觉 OOD），证明合并能力可迁移到物理硬件、无性能退化。

### Table 5: Routing Subspace Ablation / 路由子空间消融

固定 $\lambda=0.6$、VLM 用 TA 合并。

| Method | Spatial | Object | Goal | Long | Avg. |
|--------|---------|--------|------|------|------|
| Only $\mathbf{K}$ | 98.0 | 0.0 | 39.6 | 76.6 | 53.6 |
| $\mathbf{K}\&\mathbf{V}$ | 98.0 | 0.0 | 85.8 | 76.6 | 65.1 |
| **Only $\mathbf{V}$** | 98.0 | **98.8** | **85.4** | 76.6 | **89.7** |

**说明**：仅用 value 投影路由最优（89.7%）；用 K 或 K&V 在 Object/Goal 上骤降甚至 0%（路由误判任务）。原因：value 投影捕获 query 检索到的实际行为语义，是更可靠的任务身份指标——这是"用 V 不用 Q/K"设计的直接证据。

### Table 6: Fine-tuning Hyperparameters / 微调超参（附录）

| 超参 | 取值 |
|------|------|
| Backbone | Qwen-2.5 (0.5B) |
| Batch size | 8 |
| Learning rate | $5\times10^{-4}$ |
| LoRA rank | 32 |
| Use proprioception | True |
| Num images | 2（RoboTwin 为 3） |
| Gradient step | 30k（LIBERO-Long 为 50k） |

**说明**：所有实验统一 VLM 主干与训练配置；单卡 NVIDIA A6000 Ada (48GB) 完成微调。

### Table 7: OpenVLA Component-wise Merging / OpenVLA 分组件合并（附录）

用 Iso-CTS 合并四个 LIBERO 任务检查点，未合并组件保留原权重，每子任务 50 试。A=视觉主干、B=projector、C=语言模型本体（不含 lm_head）、D=lm_head。灰底为单任务微调。

| Method | Spatial | Object | Goal | Long | Avg. |
|--------|---------|--------|------|------|------|
| *Finetuned* | 84.7 | 88.4 | 79.2 | 53.7 | 76.5 |
| A | 61.4 | 60.8 | 59.0 | 8.4 | 47.4 |
| A+B | 56.6 | 58.0 | 55.6 | 6.6 | 44.2 |
| **C** | **0.0** | **0.0** | **0.0** | **0.0** | **0.0** |
| D | 83.4 | 88.8 | 72.6 | 49.6 | 73.6 |
| A+D | 61.0 | 61.0 | 62.6 | 8.4 | 48.3 |
| A+B+D | 58.0 | 57.4 | 53.8 | 7.4 | 44.2 |

**说明**：合并 A/B/D 仅略降，唯独合并**语言模型本体 C 直接全 0**——定位语言模型是合并失败主因。作者解释：机器人控制需连续数值输出，微小误差即不可逆；C 负责解码动作，积累了不可合并的任务差异。lm_head（D）几乎可互换（把 Object 与 Spatial 的 D 互换后仍 82%），说明动作头跨任务大体通用。

---

## 实验

### 数据集 / 基准

| 基准 | 规模 | 特点 | 用途 |
|------|------|------|------|
| [[LIBERO]] | 4 套件（Spatial/Object/Goal/Long），每套件 10 任务 ×50 示范 | 单臂多技能（空间关系、物体交互、目标） | 训练/测试 |
| [[LIBERO-Plus]] | 10,030 任务，7 类扰动 | 视觉/语言分布偏移鲁棒性评测（OOD） | 测试 |
| [[RoboTwin]] 2.0 | 3 本体 ×4 任务，每{本体,任务}50 示范 | 双臂、跨本体 | 训练/测试 |
| 真实 SO-101 | 3 任务，每任务 50 遥操作示范 | 单臂、20Hz、顶视+腕部双视角、6-DoF | 训练/测试 |

### 实现细节

- **Backbone**: [[Qwen2.5]]-0.5B；LoRA rank 32；用 proprioception
- **默认超参**: $l=L$（专家头取最后一块）、$k_r=8$、掩码比例 $\lambda=0.6$、合并缩放 $\alpha=1$
- **优化**: lr $5\times10^{-4}$、batch 8、梯度步 30k（LIBERO-Long 50k；真实每任务 30k）
- **图像数**: 2 张（RoboTwin 3 张）
- **硬件**: 单卡 NVIDIA A6000 Ada (48GB)
- **合并方法**: 兼容 TA、TIES、TSV、KnOTS、WUDI、EMR 等多种 data-free 方法，配合任务掩码 $\mathbf{S}$

### 关键实验结论

- **LIBERO（Table 1）**: 朴素合并全 0%；MergeVLA + TIES + 掩码达 **90.2%**，仅比单任务微调低 6.5%，且参数仅 0.70B。
- **LIBERO-Plus（Table 2）**: 单任务下鲁棒性 72.4% 远超基线；合并后仍 62.5%，鲁棒性可跨任务保留。
- **RoboTwin（Table 3）**: 跨本体单任务 88.7%、跨本体跨任务 70.7%（需更深专家头 $\mathbf{H}^{(L-2)\to L}$）。
- **真实 SO-101（Table 4）**: TIES 合并 90.0%，与单任务微调持平，含视觉 OOD（未见颜色立方体）。
- **消融1（掩码比例 $\lambda$，Figure 7）**: $\lambda\in[0.6,0.9]$ 最佳；太小→干扰严重甚至失败。
- **消融2（路由子空间，Table 5）**: 仅用 value 投影最优（89.7%），用 K 会误判任务。
- **附录诊断（Table 7 / Figure 8）**: OpenVLA 合并失败的根源是语言模型本体（C）；逐块合并显示任务冲突随层深增长。

---

## 批判性思考

### 优点
1. **问题定位扎实、方法因果清晰**：先用参数分解 + 逐块距离 + 自私比例（Figure 3、Table 7、Figure 8）把"为何不可合并"诊断到具体组件（LoRA 发散 + 自注意力深层专化），再对症下三剂药（掩码 / 删自注意力 / 路由），每个设计都有对应证据，不是堆 trick。
2. **可合并性 + 鲁棒性双赢**：删自注意力换 sigmoid 门控不仅让模型可合并，还顺带在 OOD LIBERO-Plus 上提升 13.4%，说明"逼模型依赖鲁棒 VLM 特征"是有效正则。
3. **轻量 + 免重训 + 免任务标签**：0.70B 合并模型在多基准逼近各自微调专家，免联合训练、测试时免任务身份，工程上支持增量地"插拔技能"，对真实多技能机器人系统有实用价值。

### 局限性
1. **专家头不可合并是妥协而非根治**：每任务需保留独立专家头（甚至跨本体要 $\mathbf{H}^{(L-2)\to L}$ 更深），参数与存储随任务数线性增长；RoboTwin Setting B 的 Handover Block 单任务微调本就只有 46%、合并后 38%，双臂协调类任务的绝对性能仍弱。
2. **规模与主干单一**：仅在 Qwen2.5-0.5B 这一小主干验证，作者自己在结论里也承认"更大 VLM 主干是否仍兼容本框架"尚待检验；缺少大规模 robot 预训练下的合并研究。
3. **路由的可靠性边界未充分压力测试**：测试时路由仅在 $t{=}0$ 路由一次且全程固定，若初始观测有歧义或任务数 $M$ 很大时的误判率、以及与相近任务的混淆，缺少系统性分析；当前任务数（LIBERO 4 类、RoboTwin 3 项）规模较小。

### 潜在改进方向
1. 探索让专家头也可（部分）合并的机制（如对动作分布做对齐/归一化），降低"每任务一头"的线性开销。
2. 在更大 VLM 主干与含 robot 预训练的设定下验证框架可移植性（作者已列为 future work）。
3. 把单步路由扩展为可在 episode 中纠错的轻量在线路由，并在任务数更多、任务更相近时评估路由鲁棒性。

### 可复现性评估
- [x] 代码/项目主页（https://mergevla.github.io/）
- [ ] 预训练模型权重（论文未明确声明 release，待主页确认）
- [x] 训练细节较完整（附录 Table 6 给出统一超参；默认 $\lambda,k_r,\alpha$ 等明确）
- [x] 数据集可获取（LIBERO / LIBERO-Plus / RoboTwin 2.0 公开；真实数据为自采）

---

## 速查卡片

> [!summary] MergeVLA: Cross-Skill Model Merging Toward a Generalist VLA Agent
> - **核心**: 诊断出 VLA 不可合并的两根因（LoRA 发散 + 动作专家自注意力深层专化），并面向可合并性重设计架构。
> - **方法**: 任务掩码 $\mathbf{S}_m$ 稀疏激活合并 LoRA（公式2/3）+ 动作专家改纯交叉注意力 + sigmoid 门控（仅深层专家头不合并）+ 用 value 子空间 SVD 的免训练测试时任务路由（公式6/7）。
> - **结果**: LIBERO 90.2% / LIBERO-Plus 62.5% / RoboTwin 70.7% / 真实 SO-101 90.0%（均逼近或超过单任务微调），合并模型仅 0.70B、免重训免任务标签。
> - **项目**: https://mergevla.github.io/

---

*笔记创建时间: 2026-06-29*
