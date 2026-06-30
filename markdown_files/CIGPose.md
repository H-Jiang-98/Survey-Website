---
title: "CIGPose: Causal Intervention Graph Neural Network for Whole-Body Pose Estimation"
method_name: "CIGPose"
authors: [Bohao Li, Zhicheng Cao, Huixian Li, Yangming Guo]
year: 2026
venue: CVPR
tags: [whole-body-pose-estimation, causal-inference, backdoor-adjustment, graph-neural-network, deconfounding, keypoint-estimation]
zotero_collection: _inbox
image_source: online
arxiv_html: https://arxiv.org/html/2603.09418v2
created: 2026-06-29
---

# CIGPose: Causal Intervention Graph Neural Network for Whole-Body Pose Estimation

## 元信息

| 项目 | 内容 |
|------|------|
| 作者 | Bohao Li, Zhicheng Cao, Huixian Li, Yangming Guo |
| 机构 | 西北工业大学计算机学院、西安电子科技大学、西北工业大学网络空间安全学院 |
| 会议 | CVPR 2026 |
| 类别 | 2D 全身姿态估计（Whole-body Pose Estimation） / 因果推断 |
| 日期 | 2026-03（arXiv v2） |
| 项目主页 | https://github.com/53mins/CIGPose |
| 链接 | [arXiv](https://arxiv.org/abs/2603.09418) / [Code](https://github.com/53mins/CIGPose) |

---

## 一句话总结

> 把全身姿态估计的鲁棒性失效归因于"视觉上下文混淆"，用结构因果模型形式化后，以"预测不确定性识别被混淆关键点 + 用可学习规范嵌入做反事实替换"近似 [[do-operator|do 算子]]，再叠加层次化 [[GNN]] 强制解剖一致，在 COCO-WholeBody 无额外数据即刷新 SOTA。

---

## 核心贡献

1. **因果框架形式化**: 首次把 2D 全身姿态估计纳入因果框架，用 [[Structural Causal Model|结构因果模型(SCM)]] 把**视觉上下文 $C$** 识别为制造伪相关的关键**混淆变量(confounder)**，揭示其形成的非因果**后门路径** $F\leftarrow X\leftarrow C\to Y$。
2. **因果干预模块 (CIM)**: 提出新型 [[Causal Intervention Module|因果干预模块]]，用**预测不确定性**作为混淆代理识别被污染关键点嵌入，再用可学习的**规范嵌入(canonical embedding)** 替换之，近似一次 $do$ 操作以阻断后门路径。
3. **层次化图推理**: 在**去混淆后**的嵌入上施加层次化 [[GNN]]（先局部肢体运动学、再全局部件间关系），在因果可靠的基础上强制全身解剖一致性。
4. **多基准 SOTA**: CIGPose-x 仅用 COCO-WholeBody 训练即达 **67.0% AP**，超过依赖额外 UBody 数据的 DWPose-l(66.5%)；加 UBody 后升至 **67.5% AP**，体现更强鲁棒性与数据效率。

---

## 问题背景

### 要解决的问题
SOTA 全身姿态估计器（如 [[RTMPose]]、[[DWPose]]、[[ViTPose]]）在重度遮挡、杂乱背景、困难光照下**缺乏鲁棒性**，常输出**解剖学上不合理**的预测（例如把背景纹理误识为肢体、在路灯上"幻觉"出人体骨架）。如何从根本上消除这种脆弱性。

### 现有方法的局限
- 高容量模型实际依赖**表层统计**而非解剖理解，从训练数据中学到**伪相关(spurious correlation)**——例如把"椅背"与"躯干"因共现而绑定。
- 既有提升鲁棒性的路线（知识蒸馏 [[DWPose]]、超大数据集 [[Sapiens]]）**没有直接处理混淆问题**，只是绕过。
- 现有用于姿态的 [[GNN]] 直接在**未去混淆**的初始关键点表征上推理，会把被污染表征的错误**沿图传播**放大。

### 本文的动机
- 用 [[Structural Causal Model|SCM]] 把根因显式建模为**视觉混淆**：上下文 $C$ 同时影响图像 $X$ 与真值姿态 $Y$，使模型学到的是观测分布 $P(Y|F)$ 而非真正的因果关系。
- 真正的鲁棒性应学**干预分布** $P(Y|do(F))$，用 [[do-operator]] 去除混淆。理论上可用[[Backdoor Adjustment|后门调整公式]]计算，但因 $C$ 高维不可观测而**不可解**，故需一个可计算的近似——这就是 CIM 的反事实替换。
- 关键洞见：被混淆的关键点会在视觉证据与偏见先验之间产生冲突，**表现为高认知不确定性**，因此不确定性是检测混淆的有效代理。

---

## 方法详解

### 模型架构

CIGPose 在 [[RTMPose]] 之上引入一个新的预测头，整体是 **"编码器 → 因果干预模块(CIM) → 层次化 GNN → 预测头"** 的 top-down 流水线（见 Figure 5）：
- **输入**: 单张 RGB 人体框图像 $X$（$256\times192$ 或 $384\times288$）
- **Backbone / 关键点编码器**: [[RTMPose]]（[[CSPNeXt]] 主干 + [[Gated Attention Unit|GAU]]），输出初始关键点嵌入 $F\in\mathbb{R}^{B\times K\times 512}$，$K=133$（COCO-WholeBody）或 $17$（COCO/CrowdPose）
- **核心模块 1**: [[Causal Intervention Module|因果干预模块(CIM)]]——用不确定性识别被混淆关键点，用规范嵌入做反事实替换，得到去混淆嵌入 $F'$
- **核心模块 2**: 层次化 [[GNN]]——先 [[EdgeConv]] 局部肢体建模($\mathcal{G}_p$)，再语义超图($\mathcal{G}_h$)做全局部件间注意力，得最终解剖一致嵌入 $F''$
- **输出**: 全身 133 关键点的 1D 坐标分布（[[SimCC]] 形式）
- **训练双路径**: 反事实路径(用 $F'$，唯一推理路径) + 观测路径(用原始 $F$，仅作一致性目标)

### 核心模块

#### 模块1: Causal Intervention Module（因果干预模块 CIM）

**设计动机**: 用一个可计算的反事实替换近似不可解的[[Backdoor Adjustment|后门调整]]，阻断后门路径 $F\leftarrow X\leftarrow C\to Y$，强制模型从因果可靠的证据推理。

**具体实现**（两步，见 Figure 2(c)）:
- **Step 1 混淆识别**: 由 $F$ 经两个线性层生成 1D 坐标后验分布 $(P_{k,x},P_{k,y})$；用分布的**峰值集中度**衡量歧义——峰低=高不确定性=疑似被混淆。据此计算每个关键点的混淆分数 $s_c(k)$（公式2），取分数最高的 top-$n$ 个关键点做干预。
- **Step 2 反事实嵌入替换**: 对选中关键点执行 $do(f_k:=z_k)$，即用可学习规范嵌入表 $Z\in\mathbb{R}^{K\times512}$ 中对应行 $z_k$ 替换 $f_k$（公式3），得去混淆集合 $F'$。
- **为何 $Z$ 能"去混淆"**: $Z$ 是跨所有图像**共享的全局参数**，与任何特定输入的混淆实例无关，即 $Z\perp C$。每行 $z_k$ 在多张图上被反复更新，积累的是"跨上下文对该关键点都有用"的信息——恰是上下文不变的因果理想。UMAP 可视化（Figure 4）证实：初始上下文嵌入 $F$ 散成一团（受混淆扰动），而 $z_k$ 收敛为单一高度集中的点。
- **实现细节**: $Z$ 用标准 Embedding 层、$\mathcal{N}(0,0.01^2)$ 初始化、端到端优化；$n=13$（约 $K=133$ 的 10%）最优。

#### 模块2: Hierarchical GNN（层次化图推理）

**设计动机**: 在已去混淆的 $F'$ 上强制全局解剖约束，灵感来自动作识别的 [[HD-GCN]]——先局部后全局。

**具体实现**（两阶段）:
- **Intra-Part 局部关系建模**: 在标准解剖骨架图 $\mathcal{G}_p=(\mathcal{V},\mathcal{E}_p)$ 上用一层 [[EdgeConv]]，边表示物理连接，按邻居更新每个关键点嵌入，建模局部运动学。EdgeConv 计算边特征（相连节点特征差 concat 源节点特征），经 $1\times1$ 卷积实现的共享 MLP 聚合。
- **Inter-Part 全局上下文注意力**: 在语义超图 $\mathcal{G}_h$ 上，把功能关键点组（如"左手"）的成员嵌入平均得超边表征 $g_e=\frac{1}{|e|}\sum_{k\in e}f'_k$；这些组级特征在全连接组图上再过一层 EdgeConv 做消息传递成为上下文感知；最后生成**通道级注意力权重**去调制关键点嵌入，得最终解剖一致嵌入 $F''$（公式4）。

### 关键公式与机制

#### 公式1: [[Backdoor Adjustment|后门调整]]（理论目标）

$$
P(Y|do(F))=\sum_{c}P(Y|F,c)P(c)
$$

**含义**: 干预分布 $P(Y|do(F))$ 是去除混淆 $C$ 影响后、关键点嵌入 $F$ 对预测 $Y$ 的真实因果效应；理论上可对混淆 $C$ 做边缘化得到。但 $C$ 是不可观测的高维视觉上下文，对 $c$ 求和**不可计算**，故只能近似。

**符号说明**:
- $Y$: 最终姿态预测；$F$: 关键点嵌入；$C$: 视觉上下文混淆变量
- $do(\cdot)$: Pearl 的 [[do-operator|干预算子]]
- $P(Y|F,c)$: 给定证据与上下文的条件分布；$P(c)$: 上下文先验

#### 公式2: 混淆分数（不确定性代理）

$$
s_c(k)=1-\frac{1}{2}\big(\max(P_{k,x})+\max(P_{k,y})\big)
$$

**含义**: 用 1D 坐标后验分布在 $x/y$ 两轴上的峰值高度衡量预测确定性；峰越低（分布越弥散）则 $s_c(k)$ 越大，表示该关键点越可能被混淆。取 top-$n$ 最高分者做干预。Figure 3 实证：遮挡关键点的 $s_c$ 中位数与分布范围都显著高于可见关键点。

**符号说明**:
- $P_{k,x},P_{k,y}$: 关键点 $k$ 在 $x/y$ 轴上归一化的后验概率分布
- $\max(\cdot)$: 分布峰值；$s_c(k)\in[0,1]$: 混淆分数

#### 公式3: 反事实嵌入替换（do 操作）

$$
f'_{k}=\begin{cases}z_{k},&\text{if }k\text{ is selected for intervention}\\ f_{k},&\text{otherwise}\end{cases}
$$

**含义**: 对被选中的关键点执行 $do(f_k:=z_k)$，用上下文不变的规范嵌入 $z_k$ 替换被混淆的 $f_k$，其余保留原样，得到"清洁"集合 $F'$。因 $z_k$ 不再是 $C$/$X$ 的后代，后门路径 $F'\leftarrow X\leftarrow C\to Y$ 在该点被物理切断。

**符号说明**:
- $f_k$: 编码器输出的原始（可能被混淆）嵌入；$z_k$: 来自可学习表 $Z$ 的规范嵌入
- $F'=\{f'_k\}$: 去混淆嵌入集合

#### 公式4: 层次化 GNN 通道调制

$$
f''_{k}=f'_{k}\odot\left(\frac{1}{|\mathcal{E}_{k}|}\sum_{e\in\mathcal{E}_{k}}\sigma(\psi_{a}(g'_{e}))\right)
$$

**含义**: 用上下文感知的超边表征生成通道级注意力门控，对去混淆嵌入做逐通道调制，注入全局部件间关系，得到既去混淆又解剖合理的最终嵌入 $f''_k$。

**符号说明**:
- $g'_e$: 超边 $e$ 经消息传递后的上下文感知表征；$\mathcal{E}_k$: 包含关键点 $k$ 的超边集合
- $\psi_a$: MLP；$\sigma$: sigmoid；$\odot$: 逐元素相乘

#### 公式5: 主预测损失（关键点 KL）

$$
\mathcal{L}_{\text{kpt}}=\sum_{k=1}^{K}w_{k}\cdot D_{KL}\big(Q_{k}\,\|\,P(Y_{k}|do(F))\big)
$$

**含义**: 施加在反事实路径输出上，最小化预测姿态分布 $P(Y|do(F))$ 与真值分布 $Q$ 之间的 KL 散度。

**符号说明**:
- $Q_k$: 关键点 $k$ 的真值分布；$P(Y_k|do(F))$: 反事实路径预测
- $w_k$: 按真值可见性加权的系数；$D_{KL}$: KL 散度

#### 公式6: 反事实一致性损失

$$
\mathcal{L}_{\text{cf}}=\frac{1}{|S|}\sum_{k\in S}D_{KL}\big(\text{sg}[P(Y_k|F)]\,\|\,P(Y_k|do(F))\big)
$$

**含义**: 仅施加在**稳定关键点集合** $S$（即未被干预、$s_c$ 最低的 $K-n$ 个）上，用 [[stop-gradient]] 把观测路径预测 $P(Y_k|F)$ 当作固定目标，约束干预只改动被混淆表征而**不扰乱可靠的表征**，从而让规范嵌入 $Z$ 学得有意义。

**符号说明**:
- $S$: 稳定（非干预）关键点集合，$|S|=K-n$
- $\text{sg}[\cdot]$: 停止梯度算子，把观测预测视为常量目标
- $P(Y_k|F)$: 观测路径预测；$P(Y_k|do(F))$: 反事实路径预测

#### 总损失

$$
\mathcal{L}=\mathcal{L}_{\text{kpt}}+\lambda\mathcal{L}_{\text{cf}}
$$

**含义**: 主预测损失 + $\lambda$ 加权的一致性正则；$Z$ 与网络参数 $\theta$ 用 AdamW 联合优化（$\lambda=0.1$）。

#### 公式7: 实例内 top-$n$ 富集分析（附录B）

$$
\Delta_{i}=\frac{1}{|T_{i}|}\sum_{k\in T_{i}}e_{i}(k)-\frac{1}{|V_{i}\setminus T_{i}|}\sum_{k\in V_{i}\setminus T_{i}}e_{i}(k),\qquad e_{i}(k)=\big\|\hat{\mathbf p}_i(k)-\mathbf p_i(k)\big\|_2
$$

**含义**: 在同一实例 $i$ 内，比较被 $s_c$ 选中的 top-$n$ 关键点 $T_i$ 与其余可见关键点的定位误差差。$\Delta_i>0$ 说明 $s_c$ 选中的确实是更难、误差更大的关键点（不只是遮挡）。

**符号说明**:
- $V_i$: 可见关键点集；$T_i=\mathrm{TopK}(\{s_{c,i}(k)\},n)$: 被选中集
- $e_i(k)$: 关键点 $k$ 的定位误差（预测与真值的 L2 距离）

#### 公式8–9: 批内实例级替换与梯度（附录C）

$$
f'_{b,k}=(1-m_{b,k})f_{b,k}+m_{b,k}z_{k},\qquad m_{b,k}=\mathbf 1[k\in\mathcal I_b]
$$

$$
\frac{\partial\mathcal L}{\partial z_k}=\sum_{b=1}^{B}m_{b,k}\frac{\partial\mathcal L}{\partial f'_{b,k}}
$$

**含义**: 公式8 把替换写成 mask 形式（$\mathcal I_b$ 为实例 $b$ 的 top-$n$ 选中集）；公式9 说明 $Z$ 的某行只在当前 batch 被选中时收到梯度，未选中行该步零梯度——这保证 $z_k$ 累积的是跨上下文一致有用的信息。

---

## 关键图表

<!-- 图片均使用 arXiv HTML 在线链接，未本地化 -->

### Figure 1: Comparison & Qualitative / 性能对比与定性示例

![Figure 1](https://arxiv.org/html/2603.09418v2/x1.png)

**说明**: (a) CIGPose 与相关模型在 COCO-WholeBody 上的精度-计算对比，CIGPose 系列处在帕累托前沿；(b) CIGPose-x 与基线 [[RTMPose]]-x 的定性对比，直观展示 RTMPose 把背景误识为肢体、而 CIGPose 输出解剖合理姿态。是论文"视觉混淆导致失效"主张的开篇佐证。

### Figure 2: Structural Causal Model & CIM / 结构因果模型与因果干预

![Figure 2](https://arxiv.org/html/2603.09418v2/x2.png)

**说明**: (a) 关键点估计的 [[Structural Causal Model|SCM]]，含图像 $X$、混淆 $C$、嵌入 $F$、预测 $Y$，以及非因果后门路径 $F\leftarrow X\leftarrow C\to Y$；(b) 对嵌入施加 $do$ 算子后的干预 SCM，概念上等价于切断 $C\to Y$ 的混淆弧；(c) CIM 各组件的具体实现。这是全文方法的理论核心图。

### Figure 3: Validation of Confounder Score / 混淆分数验证

![Figure 3(a) Limb Keypoints](https://arxiv.org/html/2603.09418v2/x3.png)
![Figure 3(b) All Keypoints](https://arxiv.org/html/2603.09418v2/x4.png)

**说明**: 在 COCO-WholeBody 验证集上，混淆分数 $s_c(k)$ 对遮挡/可见关键点的分布对比。(a) 易遮挡的肢体关键点、(b) 全部关键点：遮挡关键点的 $s_c$ 中位数显著更高、分布更宽。定量支撑"不确定性是混淆的可靠代理"这一关键假设。

### Figure 4: UMAP of Contextual vs Canonical Embeddings / 嵌入的 UMAP 可视化

![Figure 4](https://arxiv.org/html/2603.09418v2/x5.png)

**说明**: 初始上下文嵌入 $F$（编码器输出）对单一关键点类型散成弥散簇（反映混淆带来的方差），而对应的可学习规范嵌入 $z_k$ 收敛为单一高度集中的点；不同关键点类型的规范嵌入彼此清晰分离。证实 $Z$ 学到了上下文不变的"因果理想"且语义可区分。

### Figure 5: CIGPose Architecture / 整体架构

![Figure 5](https://arxiv.org/html/2603.09418v2/x6.png)

**说明**: CIGPose 整体架构与双路径训练。训练时嵌入走两条路：(1) 反事实路径——经 CIM 去混淆后送入层次化 GNN；(2) 观测路径——用原始嵌入做一致性约束。推理时**只用反事实路径**。

### Figure 6: Qualitative Comparison (main) / 困难场景定性对比

![Figure 6](https://arxiv.org/html/2603.09418v2/x7.png)

**说明**: 困难图像上 CIGPose-x 与基线 RTMPose-x 的对比（输入图 / RTMPose-x / CIGPose-x）。展示 CIGPose 在遮挡、杂乱、复杂场景下产出更解剖合理的姿态，间接佐证其对非遮挡混淆也具鲁棒性。

### Figure 7: Posterior Probability Distributions / 后验分布可视化

![Figure 7](https://arxiv.org/html/2603.09418v2/x8.png)

**说明**: (A) 未被混淆的关键点（绿圈，鼻子）后验分布尖锐高峰、低不确定性；(B) 被混淆关键点（红圈，左踝，遮挡且处阴影）分布弥散、峰低，高预测歧义。逐例展示 CIM 机制：高 $s_c$ 触发对左踝的反事实替换，GNN 再借可见关键点(如左膝)的解剖约束推断遮挡关节。

### Figure 8: Failure Cases / 失败案例

![Figure 8](https://arxiv.org/html/2603.09418v2/x9.png)

**说明**: 两类失败。(上) 罕见但合法的头对头亲密交互——视觉证据正确却处姿态分布长尾，模型因 OOD 产生高不确定性，CIM 误判为混淆并用规范嵌入过度正则，丢失正确细粒度几何。(下) 把路灯幻觉成人体骨架——这是**高置信度的语义错误**($s_c$ 低)，CIM 机制被绕过。揭示框架边界：能纠正"不确定"表征，但缺乏对"自信误判"的拒绝机制。

### Figure 9: More Qualitative Results / 更多定性对比（附录）

![Figure 9](https://arxiv.org/html/2603.09418v2/x10.png)

**说明**: 附录补充的困难图像定性对比（输入图 / RTMPose-x / CIGPose-x），进一步展示 CIGPose 在挑战性场景的优势。

### Table 1: COCO-WholeBody V1.0 Results / 全身姿态主结果

| Method | Input Size | GFLOPs | whole AP | whole AR | body AP | foot AP | face AP | hand AP |
|--------|-----------|--------|----------|----------|---------|---------|---------|---------|
| ZoomNAS [51] | 384×288 | 18.0 | 65.4 | 74.4 | 74.0 | 61.7 | 88.9 | 62.5 |
| ViTPose+-H [53] | 256×192 | 122.9 | 61.2 | - | 75.9 | 77.9 | 63.3 | 54.7 |
| RTMPose-l [16] | 384×288 | 10.1 | 64.8 | 73.0 | 71.2 | 69.3 | 88.2 | 57.9 |
| RTMPose-x [16] | 384×288 | 18.1 | 65.3 | 73.3 | 71.4 | 69.2 | 88.9 | 59.0 |
| DWPose-l* [56] | 384×288 | 10.1 | 66.5 | 74.3 | 72.2 | 70.4 | 88.7 | 62.1 |
| CIGPose-m | 256×192 | 2.3 | 59.9 | 69.4 | 69.0 | 64.3 | 82.1 | 49.7 |
| CIGPose-l | 256×192 | 4.6 | 62.6 | 71.9 | 71.2 | 69.0 | 83.3 | 54.0 |
| CIGPose-l | 384×288 | 10.7 | 66.3 | 74.9 | 73.0 | 72.0 | 88.3 | 59.8 |
| **CIGPose-x** | 384×288 | 18.7 | **67.0** | **75.4** | 73.5 | 72.3 | 88.1 | 60.2 |
| CIGPose-l + UBody | 384×288 | 10.7 | 66.9 | 75.1 | 73.1 | 72.3 | 88.0 | 61.2 |
| **CIGPose-x + UBody** | 384×288 | 18.7 | **67.5** | **75.5** | 73.5 | 70.3 | 88.4 | 62.6 |

**说明**: "*" 表示依赖两阶段蒸馏 + UBody 额外数据，"†" 多尺度测试。**CIGPose-x 仅用 COCO-WholeBody 即达 67.0% AP，超过用额外数据的 DWPose-l(66.5%)**；加 UBody 升至 67.5%。CIGPose-l(384,10.7 GFLOPs) 即以更少算力超过更大的 RTMPose-x(65.3%)。脚部、足踝等极端关键点提升尤为明显（foot AP 72.0/72.3 vs RTMPose 69.x），印证去混淆主要利于易被遮挡/截断的肢端。

### Table 2: COCO val2017 Results / 通用 17 点姿态

| Method | #Params | GFLOPs | AP | AR |
|--------|---------|--------|------|------|
| ViTPose++-B [53] | 86M | 17.1 | 77.0 | 82.6 |
| RTMPose-m [16] | 14M | 1.9 | 75.8 | 80.6 |
| RTMPose-l [16] | 28M | 4.2 | 76.5 | 81.3 |
| RTMPose-l† [16] | 28M | 9.3 | 77.3 | 81.9 |
| CIGPose-m | 14M | 1.9 | 76.6 | 79.3 |
| CIGPose-l | 28M | 4.2 | 77.6 | 80.3 |
| **CIGPose-l†** | 29M | 9.4 | **78.5** | 81.1 |

**说明**: 在标准 17 点 COCO 上验证通用性。CIGPose-l†(384×288) 达 78.5% AP，较强基线 RTMPose-l† **+1.2 AP**，计算量仅微增（9.3→9.4 GFLOPs）。说明因果机制对普通姿态任务也有益，而非仅依赖全身密集关键点。

### Table 3: CrowdPose Results / 拥挤遮挡场景

| Method | #Params | AP | AP_E | AP_M | AP_H |
|--------|---------|------|------|------|------|
| SWAHR [31] (bottom-up) | 63.8M | 71.6 | 78.9 | 72.4 | 63.0 |
| HRFormer-B [59] | 43.2M | 72.4 | 80.0 | 73.5 | 62.4 |
| RTMPose-m [16] | 13.5M | 70.6 | 79.9 | 71.9 | 58.2 |
| CIGPose-m | 14.4M | 71.4 | 81.0 | 72.7 | 58.9 |
| CIGPose-l | 28.4M | 73.7 | 82.8 | 75.1 | 61.2 |
| CIGPose-l† | 28.8M | 74.2 | 82.9 | 75.6 | 62.5 |
| **CIGPose-x†** | 50.4M | **75.8** | **84.2** | **77.3** | **63.6** |

**说明**: 在拥挤遮挡基准上 CIGPose-l(73.7%) 即超过 HRFormer-B(72.4%)，CIGPose-x† 进一步达 75.8%。在 medium/hard 子集上的持续提升直接印证方法对遮挡、杂乱背景等常见混淆的缓解能力。

### Table 4: Ablation of Components / 组件消融

| CIM | $\mathcal{G}_h$ | $\mathcal{G}_p$ | Model | AP | AR |
|-----|-----|-----|-------|------|------|
| ✓ | ✓ | ✓ | CIGPose-l | 66.3 (+1.5) | 74.9 (+1.9) |
| ✓ | ✓ | ✓ | **CIGPose-x** | **67.0 (+1.7)** | **75.4 (+2.1)** |
| ✗ | ✓ | ✓ | CIGPose-l | 66.0 (+1.2) | 74.8 (+1.8) |
| ✗ | ✓ | ✓ | CIGPose-x | 66.8 (+1.5) | 75.5 (+2.2) |
| ✗ | ✗ | ✓ | CIGPose-l | 65.6 (+0.8) | 74.6 (+1.6) |
| ✗ | ✗ | ✓ | CIGPose-x | 66.3 (+1.0) | 75.0 (+1.7) |
| ✓ | ✗ | ✗ | CIGPose-l | 65.7 (+0.9) | 74.3 (+1.3) |
| ✓ | ✗ | ✗ | CIGPose-x | 66.1 (+0.8) | 74.9 (+1.6) |
| ✗ | ✗ | ✗ | Baseline [16] | 64.8 / 65.3 | 73.0 / 73.3 |

**说明**: 括号内为相对基线的 AP/AR 增益。层次化 GNN($\mathcal{G}_p+\mathcal{G}_h$)无干预即带来 +1.5 AP（结构推理价值）；在完整 GNN 上加 CIM 再 +0.2 AP（去混淆增益）；CIGPose-x 总计 +1.7 AP，体现"在去混淆嵌入上做层次推理"的协同效应。注：CIM 单独($\mathcal{G}_p$/$\mathcal{G}_h$ 关闭)也带来 +0.8~+0.9 AP。

### Table 5: Within-instance Top-n Enrichment / 实例内富集分析（附录B）

| Easy-drop $p$ | Kept inst. | Mean $\bar\Delta$ (px) | 95% CI |
|------|-----------|------------|--------|
| 0.00 | 104,125 | 8.47 | [8.20, 8.72] |
| 0.30 | 72,888 | 10.14 | [9.78, 10.48] |
| 0.50 | 52,063 | 10.45 | [10.02, 10.87] |

**说明**: 富集值 $\bar\Delta$ 恒为正且随更难子集放大，说明 $s_c$ 系统性地聚焦于实例内**本质更难**的关键点（含杂乱、模糊、截断，而非仅二元遮挡标签），佐证不确定性代理的有效性可推广到遮挡之外。

### Table 6: Key Training Hyperparameters / 训练超参（附录C）

| 超参 | 值 |
|------|----|
| Optimizer | AdamW |
| Base LR | $2\times10^{-3}$ |
| Weight Decay | 0.05 |
| LR Schedule | Cosine Annealing |
| Warm-up Iters | 1000 |
| Max Grad Norm | 35 |
| Training Epochs | 420（Stage-2: 150） |
| $\lambda$ for $\mathcal{L}_{cf}$ | 0.1 |
| Train Batch Size | 32 / GPU |
| Intervention $n$ | 13 |

**说明**: CIGPose-x 在 COCO-WholeBody 上的关键训练配置。两阶段微调（前 270 epoch 重augmentation，后 150 epoch 轻augmentation）；$n=13$ 约为 $K=133$ 的 10%。

### Table 7: Intervention Frequency per Body Part / 各部位干预频率（附录E）

| Body Part | Intervention Freq. (%) |
|-----------|------------------------|
| Face | 0.67 |
| Torso | 0.16 |
| Arms | 0.24 |
| Hands | 0.90 |
| Legs | 0.89 |
| **Feet** | **1.36** |

**说明**: 干预最频繁的是足、手、腿——正是最易被遮挡/运动模糊/截断的肢端，而躯干最少(0.16%)。实证 CIM 的不确定性代理确实定位到最易被混淆的关键点。

### Table 8: Effect of $\lambda$ / 一致性损失权重消融（附录E）

| $\lambda$ | Whole-Body AP (%) |
|-----------|-------------------|
| 0 | 65.8 |
| 0.01 | 66.1 |
| **0.1** | **66.3** |
| 0.5 | 65.9 |

**说明**: $\lambda=0$（去掉一致性损失）掉 0.5 AP，说明无该正则则规范嵌入 $Z$ 学不稳定；$\lambda=0.5$ 又过度约束 GNN。$\lambda=0.1$ 最佳平衡。

### Table 9: Intervention Strategy & $n$ / 干预策略与预算消融（附录E）

| Intervention Strategy (Training) | Whole-Body AP (%) |
|----------------------------------|-------------------|
| Threshold $\tau=0.7$ | 65.9 |
| Threshold $\tau=0.8$ | 65.8 |
| Top-$n$ ($n=11$) | 66.1 |
| **Top-$n$ ($n=13$)** | **66.3** |
| Top-$n$ ($n=15$) | 66.0 |

**说明**: 固定预算 top-$n$ 优于阈值策略——后者每样本干预数波动大、训练信号噪声大，使规范嵌入难学。$n=13$ 最优。

---

## 实验

### 数据集 / 基准

| 基准 | 规模 | 特点 | 用途 |
|------|------|------|------|
| [[COCO-WholeBody]] | COCO 扩展，133 关键点密集标注 | 全身（身体/脚/脸/手） | 主训练/评测 |
| [[COCO]] (train/val2017) | 标准 2D 人体姿态 | 17 关键点 | 通用性验证 |
| [[UBody]] | >1M 帧，15 真实场景（按 10 帧间隔抽样） | 真实世界泛化 | 配合 COCO-WholeBody 训练 |
| [[CrowdPose]] | trainval 训练、test 评测 | 高度拥挤/遮挡 | 鲁棒性评测 |

评测指标：COCO/COCO-WholeBody 用基于 OKS 的 AP/AR；CrowdPose 用 AP 及 AP_easy/medium/hard。

### 实现细节

- **框架/编码器**: [[MMPose]] 工具箱，[[RTMPose]] 作关键点编码器（[[CSPNeXt]] 主干 + GAU，512 隐维），输入 $256\times192$ 或 $384\times288$；所有模型用公开 RTMPose 权重（COCO+AIC 预训练）初始化以保证公平复现。
- **CIM**: 规范嵌入表 $Z\in\mathbb{R}^{K\times512}$，Embedding 层 $\mathcal{N}(0,0.01^2)$ 初始化，端到端优化；$n=13$。
- **GNN**: EdgeConv（$1\times1$ 卷积实现 MLP）做局部，AttentionModule + 第二层 EdgeConv 做组级全局，sigmoid 生成通道注意力。
- **优化**: AdamW（weight decay 0.05）、base LR $2\times10^{-3}$、cosine annealing + 1000 iter warmup、max grad norm 35、420 epoch（Stage-2 150）、batch 32/GPU、$\lambda=0.1$。
- **两阶段数据增强**: Stage1(1-270 epoch)激进增强（翻转、半身、bbox 缩放0.5-1.5x/旋转±90°、HSV、Blur/MedianBlur、CoarseDropout 模拟遮挡）；Stage2(271-420)降强度收敛。
- **硬件**: COCO/CrowdPose 单卡 RTX 5090；COCO-WholeBody/UBody 用 8× RTX 4090。评测用 flip test，UBody 用 GT 框、其余用 RTMPose 一致的人体检测框。

### 关键实验结论

- **COCO-WholeBody**: CIGPose-x **67.0% AP**（无额外数据即 SOTA，超用 UBody 的 DWPose-l 66.5%），加 UBody 升至 67.5%。
- **COCO**: CIGPose-l† 78.5% AP，较 RTMPose-l† +1.2 AP，几乎不增算力——因果机制对通用姿态同样有益。
- **CrowdPose**: CIGPose-l 73.7%(超 HRFormer-B 72.4%)、CIGPose-x† 75.8%；medium/hard 子集持续提升直接印证抗混淆能力。
- **消融**: 层次 GNN 贡献 +1.5 AP、CIM 在其上再 +0.2 AP（合计 +1.7）；$\lambda=0.1$、top-$n$($n=13$) 最优；干预频率集中在足/手/腿。

---

## 批判性思考

### 优点
1. **问题诊断有理论深度**: 用 [[Structural Causal Model|SCM]] + [[do-calculus]] 把"鲁棒性失效"形式化为后门混淆，并在附录给出完整的后门调整推导（Rule 2/3 的 d-分离论证），而非空喊"因果"。
2. **近似手段可计算且有证据**: 反事实替换把不可解的 $\sum_c$ 化为可学习规范嵌入；并用混淆分数 vs 遮挡(Fig 3)、实例内富集(Table 5)、UMAP(Fig 4)、干预频率(Table 7) 多角度实证"不确定性是混淆代理 / $Z$ 学到上下文不变理想"。
3. **数据效率突出**: 无额外数据即超过依赖蒸馏 + UBody 的 DWPose；作为可插拔预测头，迁移到 COCO/CrowdPose 都有稳定增益，且几乎不增计算。
4. **诚实的失败分析**: 附录 F 主动指出两类失败（OOD 长尾被过度正则、自信语义误判被绕过），边界刻画清晰。

### 局限性
1. **CIM 增益较小且与 GNN 纠缠**: 消融中 CIM 在完整 GNN 上仅 +0.2 AP，主增益来自层次 GNN(+1.5)；"因果干预"作为卖点，其独立贡献的量级偏弱，难以确证收益主要来自"去混淆"而非额外参数/正则。
2. **不确定性代理的根本缺陷**: 对**高置信度语义错误**（如把路灯认成人，$s_c$ 低）完全失效，CIM 被绕过；框架只能纠正"不自信"的错误，对最危险的"自信幻觉"无能为力。
3. **过度正则风险**: 罕见但合法的长尾姿态会被误判为混淆并替换为"平均理想" $z_k$，牺牲正确的细粒度几何（作者亦承认）。
4. **规范嵌入的"因果"性偏论证而非可验证**: $Z\perp C$ 是"by construction"的论断，缺乏对 $Z$ 是否真为因果不变（而非只是类原型/均值嵌入）的反事实/干预式实证；UMAP 只能说明其集中、可分，不能直接说明"因果"。

### 潜在改进方向
1. 引入对"自信误判"的拒绝/校准机制（如能量分数、OOD 检测）补足 $s_c$ 仅捕获高不确定性的盲区。
2. 把"是否长尾合法姿态 vs 真混淆"加以区分（如结合密度估计），避免对稀有正确姿态的过度正则。
3. 用表征探针 / 干预式因果度量（而非仅 UMAP+注意力）量化 $Z$ 的上下文不变性，强化"因果"主张。
4. 将该可插拔因果头迁移到 3D / OOD / 视频姿态（作者展望的方向），验证范式可移植性。

### 可复现性评估
- [x] 代码开源（https://github.com/53mins/CIGPose ，声明 codes and models publicly available）
- [x] 预训练模型（声明 models 公开；基于公开 RTMPose 权重初始化）
- [x] 训练细节完整（附录 C/D 给出超参表 6、两阶段增强、网络结构、Algorithm 1）
- [x] 数据集可获取（COCO/COCO-WholeBody/UBody/CrowdPose 均公开）

---

## 速查卡片

> [!summary] CIGPose: Causal Intervention Graph Neural Network for Whole-Body Pose Estimation
> - **核心**: 把全身姿态鲁棒性失效归因于视觉上下文混淆(SCM)，用"不确定性识别 + 规范嵌入反事实替换"近似 $do(F)$ 阻断后门路径，再叠层次化 GNN 强制解剖一致。
> - **方法**: RTMPose 编码器 → CIM(混淆分数 $s_c$ 选 top-13 → 用可学习 $Z$ 替换) → 层次 GNN(EdgeConv 局部 + 超图全局注意力)；损失 = KL 预测损失 + 0.1×反事实一致性损失(仅约束稳定关键点)。
> - **结果**: COCO-WholeBody CIGPose-x 67.0% AP(无额外数据即 SOTA)、+UBody 67.5%；COCO 78.5%(+1.2)；CrowdPose 75.8%。
> - **代码**: https://github.com/53mins/CIGPose

---

*笔记创建时间: 2026-06-29*
