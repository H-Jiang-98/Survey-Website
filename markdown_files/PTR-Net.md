---
title: "Towards Motion Turing Test: Evaluating Human-Likeness in Humanoid Robots"
method_name: "PTR-Net"
authors: [Mingzhe Li, Mengyin Liu, Zekai Wu, Xincheng Lin, Junsheng Zhang, Ming Yan, Zengye Xie, Changwang Zhang, Chenglu Wen, Lan Xu, Siqi Shen, Cheng Wang]
year: 2026
venue: CVPR
tags: [humanoid, motion-evaluation, human-likeness, SMPL-X, turing-test, benchmark, dataset, ST-GCN]
zotero_collection: _inbox
image_source: online
arxiv_html: https://arxiv.org/html/2603.06181v1
created: 2026-06-29
---

# Towards Motion Turing Test: Evaluating Human-Likeness in Humanoid Robots

## 元信息

| 项目 | 内容 |
|------|------|
| 作者 | Mingzhe Li, Mengyin Liu, Zekai Wu, Xincheng Lin, Junsheng Zhang, Ming Yan 等 12 人 |
| 机构 | 厦门大学（Xiamen University）等；致谢中提及小米青年人才计划（Xiaomi Young Talents Program）资助 |
| 会议 | CVPR 2026 |
| 类别 | Humanoid / 运动人类相似度评估（Motion Human-likeness Assessment） |
| 日期 | 2026（arXiv v1） |
| 项目主页 | 数据集、代码与 benchmark 声明将公开（暂未给出 URL） |
| 链接 | [arXiv](https://arxiv.org/abs/2603.06181) / [HTML](https://arxiv.org/html/2603.06181v1) |

---

## 一句话总结

> 借鉴图灵测试提出"运动图灵测试"，用 SMPL-X 剥离外观、只看动作来评估人形机器人运动的"像人程度"，构建了 11 个机器人 + 10 名真人、1000 条序列、500+ 标注小时的 HHMotion 数据集，并提出轻量基线 PTR-Net 在人类相似度回归任务上超越主流多模态大模型。

---

## 核心贡献

1. **提出 Motion Turing Test 范式**: 把图灵测试迁移到人形机器人运动评估——若人类观察者仅凭**运动学信息**（无脸部、纹理、颜色）无法分辨一段姿态序列来自真人还是机器人，则该运动通过测试。用 [[SMPL-X]] 把所有视频统一转成无外观的姿态序列，强制评估只关注"动作"本身。
2. **构建 HHMotion 数据集**: 1000 条运动序列、15 个动作类别、11 个人形机器人型号 + 10 名真人受试者，涵盖真实/仿真机器人与真人（含真人**刻意模仿机器人**的混淆子集）；招募 30 名标注者按 0–5 Likert 量表打分，累计 **500+ 小时**标注。
3. **提出 Motion Human-likeness 回归任务与 PTR-Net 基线**: 将"像人程度"形式化为从运动序列到 0–5 连续分数的回归任务；提出轻量基线 [[PTR-Net]]（BiLSTM + [[ST-GCN]] + 注意力池化 + 回归头），在该 benchmark 上**全面超过 Gemini 2.5 Pro / Qwen3-VL-Plus 等多模态大模型**，证明现有 [[MLLM]] 尚不足以评估运动像人度。

---

## 问题背景

### 要解决的问题
人形机器人在运动生成与控制上进步显著，动作看起来越来越"自然、像人"。但缺少一个**统一、定量、以运动为中心**的标准来回答："机器人的动作到底有多像人？" 现有评估几乎都是任务导向指标（完成率、效率、鲁棒性、末端轨迹精度），无法刻画**感知层面的自然性、流畅性与拟人度**。

### 现有方法的局限
1. **运动数据集只面向人类**: AMASS、Human3.6M、Motion-X 等都只含真人，缺少人-机对照与"像人度"标注；少数机器人姿态数据集（DHRP、HumanoidRobotPose）规模小、无 human-likeness 分数。
2. **机器人姿态估计泛化差**: 现有机器人专用姿态估计器（RPE）容易过拟合特定机器人形态/运动学约束，跨型号泛化弱。
3. **任务指标 ≠ 感知像人度**: 高任务完成率不代表运动在人眼里"自然"；拟人度等主观维度被长期忽视。
4. **外观会泄露身份**: 直接给人看机器人原始视频，金属外壳、外露关节等外观线索会让人轻易分辨人/机，污染对"运动本身"的判断。

### 本文的动机
- 用 **SMPL-X 统一表示**剥离外观，把判断逼回"纯运动学"，这才是图灵测试的精神所在；
- 用**大规模人类标注**建立"像人度"金标准分布，使评估可量化、可学习；
- 进一步把它变成一个**自动评估任务**：训练模型预测像人度分数，从而既能替代昂贵的人工标注，又能为运动生成/强化学习提供奖励信号。核心发现是：即便在剥离外观的设定下，人类仍能轻易区分人/机，说明像人度差距真实存在，尤其在 jump、boxing、run 等动态动作上。

---

## 方法详解

整体流程分两大块：**(1) HHMotion 数据集构建与人类标注**（图灵测试的数据基础），**(2) Motion Human-likeness 回归任务与 PTR-Net 基线**（自动评估）。

### 数据集构建流水线

HHMotion 的数据来自 **5 个来源**：重大赛事中的真实机器人运动、仿真机器人运动、志愿者真人运动、**志愿者刻意模仿机器人**的运动、以及 YouTube 真人运动。

- **机器人数据**: 从 WRC、WAIC、WHRG 等国际顶级赛事采集 21.7h 原始视频，覆盖 11 个机器人型号、28 支团队的算法；仿真侧把 [[LAFAN1 Retargeting]] 数据集在仿真环境中可视化录制，覆盖 dance / fall-and-get-up / fight / jump / kicking-ball / run / walk 共 7 类。每段切成 5 秒含完整动作的片段，得到 **500 条机器人片段**（257 真实 + 243 仿真）。
- **真人数据**: 按机器人采集到的类别，让 10 名受试者每类做 25–30 个动作，覆盖室内外；再从 YouTube 补充提高多样性。共 **500 条真人片段**（365 受试者 + 135 互联网）。其中专门加入**真人模仿机器人**的子集，制造"人机难辨"的模糊样本，使测试更贴近图灵测试本质。
- **标准化**: 经数据清洗（去除低分辨率、遮挡、截断），保留约 5000 秒高质量片段，每段统一 5 秒，共 **1000 条**。

### 人-机姿态估计（统一到 SMPL-X）

**设计动机**: 机器人外观（金属壳、外露关节）会让人凭外观而非动作分辨，因此把所有视频转成 [[SMPL-X]] 姿态序列，确保评估只关注运动信息。

**具体实现**:
- 系统比较了多种 SOTA 人体姿态估计（HPE）与机器人姿态估计（RPE）方法；发现**机器人专用 RPE 泛化差**，而通用 HPE 在人与人形机器人上更鲁棒。
- 最终选用 [[GVHMR]]——虽为人体设计，但在机器人上也泛化良好，结果时序最平滑、最稳定。
- 所有重建序列经人工交叉验证去除失败/噪声样本，部分视频用 bounding box 人工校正。

### 人类相似度标注

**评分协议**: 招募 30 名评估者，对全部 SMPL-X 序列按 **0–5 Likert 量表**打分，0 = "完全非人类/机械"，5 = "与真人无法区分"，评判维度为姿态（posture）、节奏（rhythm）、协调性（coordination）。

**实现细节**:
- 500 真人 + 500 机器人序列**随机混合打乱**隐藏来源，防止先验偏见；每位标注者评完 1000 条，平均耗时约 16.7 小时，全员累计 **500+ 标注小时**。
- 已获 IRB 伦理批准。
- 用 **Inter-Annotator Consistency (IAC)** 检查标注一致性，剔除 5 名与整体分布不一致的标注者，保留后 25 名的平均分作为最终标签。

### PTR-Net：Pose-Temporal Regression Network

**任务形式化**: 把像人度评估建模为**回归任务**——输入一段 SMPL-X 时序运动（先归一化到局部根坐标系，去除全局平移/旋转），输出 0–5 的连续像人度分数 $s$，与人类标注对齐。

PTR-Net 由三个组件构成（见 Figure 7）：

#### 模块1: Temporal Encoder（时序编码器）

**设计动机**: 捕捉运动的长程时序依赖。

**具体实现**: 两层**双向 LSTM**，输出时序编码特征 $\mathbf{H}_t\in\mathbb{R}^{2h}$。

#### 模块2: Spatial–Temporal Graph Convolution（ST-GCN，时空图卷积）

**设计动机**: 把人体建模为图，跨关节与跨帧地提取协调性模式。

**具体实现**: 把编码后的序列重排为人体图表示，过一叠 [[ST-GCN]] 块，每块交替进行**空间图卷积**（关节间）与**时间卷积**（帧间）。区别于常规骨架 GCN，PTR-Net 采用**无参数邻接矩阵设计**（parameter-free adjacency），允许更自适应的特征聚合。

#### 模块3: Attention Pooling + Regression Head（注意力池化 + 回归头）

**设计动机**: 突出显著运动片段，再回归出标量分数。

**具体实现**: 一个时间注意力模块高亮关键运动段，后接一个轻量 MLP 回归头输出标量像人度分数。

### 关键公式与机制

#### 公式1: [[PTR-Net]] 端到端映射

$$
s = f_{\theta}(\mathbf{X})
$$

**含义**: PTR-Net 学习一个从运动序列 $\mathbf{X}$ 到像人度分数 $s$ 的端到端映射。

**符号说明**:
- $\mathbf{X}$: 输入的 SMPL-X 运动序列（归一化到局部根坐标系）
- $f_{\theta}$: 网络（参数为 $\theta$）
- $s\in[0,5]$: 预测的像人度分数

#### 公式2: 训练目标（L2 回归 + 正则）

$$
\mathcal{L} = \|\hat{s} - s^{*}\|_{2}^{2} + \lambda\,\mathcal{L}_{\text{reg}}
$$

**含义**: 以 L2 回归损失对齐预测分数与人类标注，并加正则项约束预测在时序上的过度波动，鼓励平滑稳定。

**符号说明**:
- $s^{*}$: 人类标注的像人度分数（ground truth）
- $\hat{s}$: 模型预测分数
- $\mathcal{L}_{\text{reg}}$: 惩罚预测分数过度时序波动的正则项
- $\lambda$: 正则权重系数

#### 评估指标

文中用三个互补指标衡量模型预测与人类判断的对齐度（详细公式在补充材料）：

- **MAE**（平均绝对误差，越低越好）
- **RMSE**（均方根误差，越低越好）
- **Spearman's $\rho$**（秩相关系数，越高越好）——衡量排序一致性。

---

## 关键图表

<!-- 图片均使用 arXiv HTML v1 在线链接，未本地化 -->

### Figure 1: Motion Turing Test / 运动图灵测试示意

![Figure 1](https://arxiv.org/html/2603.06181v1/figs/intro3.png)

**说明**: 运动图灵测试的核心思想——评估者只看姿态序列、不看外观线索，判断该运动是否"像人"。这张图直观点明了全文动机：把判断逼回纯运动学层面。

### Figure 2: Action Sources / 动作来源、类型与类别分布

![Figure 2](https://arxiv.org/html/2603.06181v1/figs/data_cate2.png)

**说明**: HHMotion 数据集中的动作来源、类型与 15 个类别的分布，展示人与人形机器人动作的多样性。体现数据集在动作类别上的平衡与覆盖广度。

### Figure 3: Human Scoring Pipeline / 人类评分流水线

![Figure 3](https://arxiv.org/html/2603.06181v1/figs/human_scoring_pipeline1.png)

**说明**: 人类评分流水线——所有机器人与真人运动统一转成 SMPL-X 姿态，再由人类标注者打 0–5 分。这是"剥离外观、只评运动"的关键工程步骤。

### Figure 4: Scoring Rules / 0–5 像人度评分规则

![Figure 4](https://arxiv.org/html/2603.06181v1/figs/score_rule.png)

**说明**: 0–5 Likert 量表的像人度评分规则，0 = 完全机械、5 = 与真人无法区分，仅针对运动质量评判。为大规模标注提供统一标准，是数据可靠性的基础。

### Figure 5: Score Distribution / 像人度分数分布

![Figure 5](https://arxiv.org/html/2603.06181v1/figs/humanscore_distribution.png)

**说明**: 左图为真人 vs 机器人运动的整体像人度分数分布（真人显著获得更高评分）；右图为机器人在仿真 vs 真实场景下的分布（**仿真普遍优于真实**）。佐证人机像人度差距真实存在，且仿真到真实仍有 gap。

### Figure 6: Bad/Good Cases / 机器人表现差与好的代表性序列

![Figure 6](https://arxiv.org/html/2603.06181v1/figs/badcase1.png)

**说明**: 上排为机器人表现差的 SMPL-X 序列（jump、boxing、pingpong 等高频协调/快速肢体切换动作），下排为表现好的（walk、stand 等结构化/周期性动作）。直观解释了"动态、接触丰富、反应性动作"是机器人短板。

### Figure 7: PTR-Net Architecture / PTR-Net 基线架构

![Figure 7](https://arxiv.org/html/2603.06181v1/figs/baseline2.png)

**说明**: PTR-Net 由时序编码器（BiLSTM）、时空图卷积（ST-GCN）、注意力池化组成，最后用回归头预测像人度分数。这是全文方法的核心架构图。

### Figure 8: Qualitative Results / 定性预测对比

![Figure 8](https://arxiv.org/html/2603.06181v1/figs/qualitative.png)

**说明**: 四个代表性样本上 PTR-Net 预测分数 vs 人类标注分数。例如 kicking ball（标注 3.8 / 预测 3.71）、playing ping pong（标注 4.8 / 预测 4.27），模型既能数值贴合人类，也能捕捉流畅性、协调性、平衡性等可解释线索。

### Figure 9: Humans Imitating Humanoids / 真人模仿机器人讨论

![Figure 9](https://arxiv.org/html/2603.06181v1/figs/imitate3.png)

**说明**: (a–b) 真人（模仿机器人）与机器人运动在人类标注与 PTR-Net 预测下的分数分布；底部为近乎难辨的舞蹈模仿配对。揭示当真人刻意模仿机器人的机械僵硬时，运动学线索不足以可靠区分——说明真正的"像人"还涉及意图性与适应性，是生成与评估的共同难点。

### Figure 10: Out-of-Distribution Evaluation / 分布外评估（XPeng IRON）

![Figure 10](https://arxiv.org/html/2603.06181v1/figs/xiaopeng.png)

**说明**: 在未见过的、2025 年 11 月新发布的 [[XPeng IRON]] 人形机器人上做 OOD 评估。PTR-Net 预测像人度 4.25，非常接近人类标注均值 4.36，验证了模型对未见型号的鲁棒性与评分体系的一致可靠。

### Table 1: Comparisons with Related Datasets / 与相关数据集对比

| Dataset | Human | Humanoid | #Models | H&H Score | Duration | Motion | #Category | #Clips |
|---------|:-----:|:--------:|:-------:|:---------:|----------|--------|:---------:|--------|
| AMASS | ✔* | - | - | - | 40+h | General | - | 26.3k |
| Human3.6M | ✔ | ✔ | - | - | - | Daily | 15 | - |
| LAFAN1 | ✔ | - | - | - | 4.6h | Sports | - | - |
| Motion-X | ✔ | ✔ | - | - | 144.2h | General | - | 81,084 |
| IDEA400 (Motion-X) | ✔ | ✔ | - | - | - | General | 400 | 13k |
| SMART | ✔ | ✔ | - | - | - | Sports | - | 5000 |
| DHRP | - | - | ✔(real) | - | - | Daily+Sports | - | 462 |
| HumanoidRobotPose | - | - | ✔(10+) | - | - | Football | - | 23 |
| PHUMA | ✔* | ✔ | 2 | - | 72.96h | General | 11 | 76.01k |
| LAFAN1 Retarget | - | - | ✔(3) | - | - | Sport | 8 | 120 |
| AMASS Retarget | - | - | ✔(1) | - | - | General | - | - |
| **HHMotion (Ours)** | **✔** | **✔** | **11** | **✔** | **21.7h** | **Daily+Sports** | **15** | **1,000** |

**说明**: HHMotion 是**唯一同时含真人 + 机器人（仿真 + 真实）、提供 human-likeness 分数标注**的数据集，覆盖 11 个机器人型号、15 类、1000 条片段。其独特性在于"人机对照 + 像人度标注"这一组合，前人数据集均不具备。

### Table 2: Top/Bottom 5 Categories by Score Difference / 人机分差最大/最小的动作类别

| Action Category | Human | Humanoid | Score Difference |
|-----------------|:-----:|:--------:|:----------------:|
| **— 分差最小（机器人最像人）—** | | | |
| stand | 3.80 | 1.97 | 1.83 |
| sit | 4.18 | 2.63 | 1.55 |
| ski | 3.13 | 1.76 | 1.37 |
| walk | 3.92 | 2.61 | 1.31 |
| dance | 3.47 | 2.26 | **1.21** |
| **— 分差最大（机器人最不像人）—** | | | |
| **jump** | 4.43 | 1.20 | **3.23** |
| boxing | 3.76 | 1.23 | 2.53 |
| run | 3.73 | 1.47 | 2.26 |
| pingpong | 4.33 | 2.09 | 2.24 |
| kicking ball | 3.93 | 1.79 | 2.14 |

**说明**: 仅对**真实机器人**计算（IAC 后 25 名标注者均值）。jump（3.23）、boxing、run、pingpong、kicking ball 等**高频协调、快速肢体切换、接触丰富**的动态动作分差最大；walk、stand、dance 等结构化/周期性动作分差最小。揭示了当前人形机器人的运动短板所在。

### Table 3: Quantitative Results on Motion Turing Test Benchmark / 主结果

| Model | MAE ↓ | RMSE ↓ | Spearman's ρ ↑ |
|-------|:-----:|:------:|:--------------:|
| Gemini 2.5 Pro (DE)* | 1.3105 | 1.5873 | 0.1609 |
| Gemini 2.5 Pro (CGE)* | 1.3314 | 1.5986 | 0.1658 |
| Gemini 2.5 Pro (PDE)* | 1.2616 | 1.5397 | 0.2188 |
| Gemini 2.5 Pro (DE-CoT)* | 1.6040 | 1.9146 | 0.1353 |
| Gemini 2.5 Pro (PA-CoT)* | 1.2682 | 1.5214 | 0.2303 |
| Qwen3-VL-Plus (shot)* | 1.7714 | 2.1018 | – |
| MotionBERT (Frozen Backbone) | 0.6846 | 0.9025 | 0.5315 |
| MotionBERT (Fine-tuned) | 0.6252 | 0.8465 | 0.6142 |
| Transformer (Lightweight) | 0.6387 | 0.8259 | 0.5728 |
| **PTR-Net (Ours)** | **0.5813** | **0.7926** | **0.6841** |

**说明**: `*` 为无任务训练的 VLM 评估；`–` 表示因输出恒定导致 $\rho$ 无效。即使用上 PA-CoT 等结构化推理策略，[[Gemini 2.5 Pro]] 仍与人类判断偏差巨大（ρ≈0.23）；[[Qwen3-VL-Plus]] 对运动细节几乎不敏感，跨设置输出几乎一致。PTR-Net 全指标领先所有基线（含 MotionBERT、Transformer），但作者诚实指出 0.79 的 RMSE 说明任务仍有较大提升空间。**核心结论：现有多模态大模型尚不足以评估运动像人度。**

### Table 4: Ablation Study of PTR-Net / 组件消融

| Model Variant | MAE ↓ | RMSE ↓ | Spearman's ρ ↑ |
|---------------|:-----:|:------:|:--------------:|
| w/o Temporal Encoder | 0.7631 | 0.9691 | 0.3610 |
| w/o Attention Pooling | 0.6185 | 0.8203 | 0.6255 |
| w/o $\mathcal{L}_{reg}$ | 0.5983 | 0.7958 | 0.6215 |
| **PTR-Net (Full Model)** | **0.5813** | **0.7926** | **0.6841** |

**说明**: 去掉**时序编码器**影响最大（ρ 从 0.68 暴跌到 0.36，MAE/RMSE 显著上升），说明长程时序建模最关键；去掉注意力池化降低与人类评分的相关性（捕捉全局时序上下文的作用）；去掉正则项 $\mathcal{L}_{reg}$ 训练稳定性与一致性变差。完整模型在误差与排序一致性间取得最佳平衡，验证各组件协同设计的合理性。

---

## 实验

### 数据集 / 基准

| 来源 | 规模 | 特点 | 用途 |
|------|------|------|------|
| 真实机器人（WRC/WAIC/WHRG） | 257 片段 | 11 型号、28 团队，赛事级 SOTA | 训练/测试 |
| 仿真机器人（LAFAN1 Retarget） | 243 片段 | 7 类，仿真录制 | 训练/测试 |
| 真人（受试者） | 365 片段 | 10 人，15 类，室内外 | 训练/测试 |
| 真人（YouTube） | 135 片段 | 多样性补充 | 训练/测试 |
| 真人模仿机器人子集 | （含于上） | 制造人机模糊样本 | 测试/分析 |
| **HHMotion 合计** | **1,000 片段 / 21.7h / 15 类** | 人机对照 + 0–5 像人度标注 | benchmark |

### 实现细节

- **姿态估计**: 用 [[GVHMR]] 把所有视频统一转 SMPL-X，序列归一化到局部根坐标系（去全局平移/旋转）。
- **标注**: 30 名标注者，0–5 Likert，IAC 过滤后保留 25 名，500+ 小时。
- **PTR-Net**: 两层 BiLSTM 时序编码器 + 无参数邻接的 ST-GCN + 时间注意力池化 + MLP 回归头；L2 回归损失 + 平滑正则；详细训练超参在补充材料。
- **对比模型**: VLM 基线 [[Gemini 2.5 Pro]]、[[Qwen3-VL-Plus]]（输入渲染后的 SMPL-X 运动视频），设计 5 种提示策略——DE（直接评估）、CGE（上下文引导）、PDE（原型驱动，6 个 0–5 标注样例校准）、DE-CoT（直接思维链）、PA-CoT（本文提出的姿态感知思维链：先判上肢/下肢/全身，再沿姿态流畅性、动作协调性、核心稳定性三维度推理）；运动专用基线 MotionBERT（冻结/微调）与轻量 Transformer。

### 关键实验结论

- **VLM 远不足够**: Gemini 2.5 Pro 即便 PA-CoT 也只有 ρ≈0.23、MAE>1.26；Qwen3-VL-Plus 输出几乎恒定（ρ 无效）。
- **PTR-Net 全面领先**: MAE 0.5813 / RMSE 0.7926 / ρ 0.6841，超越 MotionBERT 与 Transformer 基线。
- **人机差距按动作类别强烈分化**: jump/boxing/run/pingpong/kicking-ball 分差最大；walk/stand/dance 最小（Table 2）。仿真普遍比真实更"像人"（Figure 5 右）。
- **消融**: 时序编码器贡献最大；注意力池化与正则项均有正向作用。
- **真人模仿机器人**: 当人刻意模仿机械僵硬时，运动学线索不足以区分人机，PTR-Net 分布仍贴合人类标注（Figure 9）。
- **OOD 泛化**: 未见的 XPeng IRON 上预测 4.25 vs 人类 4.36，验证鲁棒性（Figure 10）。

---

## 批判性思考

### 优点
1. **问题定义新颖且重要**: 把图灵测试迁移到机器人运动评估，并用 SMPL-X 严格剥离外观，思路干净、动机清晰，填补了"以运动为中心、可量化像人度"评估的空白。
2. **数据集工程扎实**: 11 型号 + 真人对照、仿真/真实兼有、含"真人模仿机器人"的混淆子集，30 人 500+ 小时标注 + IAC 质控，规模与质量都站得住。Table 1 清楚展示其相对前人数据集的独特性。
3. **有说服力的负面结论**: 实证表明强多模态大模型（Gemini 2.5 Pro / Qwen3-VL-Plus）即便配 CoT 也评不好运动像人度，且轻量监督基线 PTR-Net 反而显著更优——这是对"VLM 万能"叙事的有价值反例。

### 局限性
1. **基线过于简单、上限有限**: PTR-Net 的 RMSE 仍达 0.79（满量程 5），作者自承"仍有较大提升空间"；方法本身（BiLSTM + ST-GCN + 注意力）较常规，更像"够用的 baseline"而非强方法。
2. **关键细节落在补充材料**: 评估指标公式、训练超参、IAC 具体流程、HPE/RPE 对比、数据集来源与机器人型号清单等均"见补充"，正文可复现信息偏少。
3. **依赖单一 HPE 的潜在偏差**: 所有评估建立在 GVHMR 估计的 SMPL-X 上，姿态估计误差可能与"像人度"判断耦合（机器人运动重建本就更难），可能系统性拉低机器人分数，缺少对这一混淆因素的隔离分析。
4. **标注主观性与文化偏差**: 0–5 Likert 高度主观，30 名标注者的人群构成、文化背景对"什么算像人"的影响未讨论；IAC 剔除 5 人的标准也较粗。
5. **机构信息不全**: arXiv HTML 中机构/通讯信息缺失，致谢仅显示厦门大学相关基金与小米资助。

### 潜在改进方向
1. 设计更强的运动评估模型（如运动专用大模型、物理一致性约束、对比学习的人机判别器），把 RMSE 进一步压低。
2. 隔离/校正姿态估计误差对像人度的影响（如对同一动作用多种 HPE 估计做敏感性分析）。
3. 把像人度分数作为**奖励信号**接入运动生成/强化学习闭环（作者在结论中已提出此愿景），验证"评估器→更像人的生成"的因果链。
4. 扩展更多机器人型号、更动态/接触丰富的动作，以及跨文化标注，提升 benchmark 的普适性。

### 可复现性评估
- [x] 代码开源（声明"dataset, code, and benchmark will be publicly released"，暂未给出 URL）
- [ ] 预训练模型（未明确）
- [ ] 训练细节完整（核心超参与指标公式落在补充材料，正文不足）
- [x] 数据集可获取（声明将公开发布）

---

## 速查卡片

> [!summary] Towards Motion Turing Test (PTR-Net / HHMotion)
> - **核心**: 用 SMPL-X 剥离外观、只看动作，把"机器人动作多像人"做成可量化的运动图灵测试与回归任务。
> - **数据**: HHMotion——1000 序列 / 15 类 / 11 机器人型号 + 10 真人 / 仿真+真实 / 含"真人模仿机器人"子集；30 标注者 0–5 打分、IAC 质控、500+ 小时。
> - **方法**: PTR-Net = BiLSTM 时序编码 + 无参数邻接 ST-GCN + 注意力池化 + MLP 回归头；L2 + 平滑正则。
> - **结果**: MAE 0.5813 / RMSE 0.7926 / ρ 0.6841，全面超过 Gemini 2.5 Pro（ρ≈0.23）与 Qwen3-VL-Plus；jump/boxing/run 人机分差最大，仿真比真实更像人；XPeng IRON OOD 预测 4.25 vs 人类 4.36。
> - **结论**: 现有 MLLM 尚不足以评估运动像人度；人机运动像人度差距真实存在，尤以动态动作为甚。

---

*笔记创建时间: 2026-06-29*
