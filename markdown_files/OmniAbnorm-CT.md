---
title: "Rethinking Whole-Body CT Image Interpretation: An Abnormality-Centric Approach"
method_name: "OmniAbnorm-CT"
authors: [Ziheng Zhao, Lisong Dai, Ya Zhang, Yanfeng Wang, Weidi Xie]
year: 2026
venue: CVPR
tags: [medical-imaging, CT, grounded-report-generation, abnormality-grounding, VLM, segmentation, taxonomy, dataset]
zotero_collection: _inbox
image_source: online
arxiv_html: https://arxiv.org/html/2506.03238v2
created: 2026-06-29
---

# Rethinking Whole-Body CT Image Interpretation: An Abnormality-Centric Approach

## 元信息

| 项目 | 内容 |
|------|------|
| 作者 | Ziheng Zhao, Lisong Dai, Ya Zhang, Yanfeng Wang, Weidi Xie |
| 机构 | 上海交通大学 / 上海人工智能实验室（推断自作者团队，论文未在 HTML 正文显式标注机构） |
| 会议 | CVPR 2026 |
| 类别 | 医学影像 / 异常定位与报告生成（grounded report generation） |
| 日期 | 2025-06（arXiv v2） |
| 项目主页 | 数据将开源（Radiopaedia 申请授权后可获取图像） |
| 链接 | [arXiv](https://arxiv.org/abs/2506.03238) / [PDF](https://arxiv.org/pdf/2506.03238) |

---

## 一句话总结

> 从"以异常为中心"重新定义全身 CT 解读：建 404 类异常层级 [[Taxonomy|分类体系]] + 19K 异常 grounding 标注的 OmniAbnorm-CT-14K 数据集，并提出把 [[VLM]] 与分割模块缝合的 OmniAbnorm-CT，可在多平面全身 CT 上定位并描述异常、支持文本/视觉提示交互。

---

## 核心贡献

1. **分类体系（Taxonomy）**: 与 7 位资深放射科医生（10–16 年经验、来自 3 个中心）共建层级化异常分类，覆盖全身 **404 种代表性异常**，组织于 **40 个主要器官/系统、82 个解剖子结构** 之下。
2. **数据集（Data）**: 构建 **OmniAbnorm-CT-14K**——首个面向多平面全身 CT 异常 grounding 与描述的大规模数据集，含 **14.5K** 张多平面（轴位/冠状/矢状）CT 图像，对 **约 19K（18,969）个异常** 提供 bounding box / 分割 mask 标注，每个异常都链接到报告中的细粒度描述并归入分类体系（覆盖 404 类中的 349 类，86%）。
3. **模型（Model）**: 提出 **OmniAbnorm-CT**，把 [[VLM]]（Qwen2.5-VL-7B）与分割模块桥接，能基于文本查询自动定位并描述异常，同时支持 box/椭圆/轮廓等**视觉提示**做交互式细化。
4. **评测（Evaluation）**: 设计 3 个贴近真实临床的任务，并提出临床导向的报告评测指标 **[[AbnormRubric]]**；在内部与外部验证、所有任务上均显著超过现有方法。

---

## 问题背景

### 要解决的问题
如何让 AI **自动检测、定位（ground）并描述** 多平面、全身 CT 图像中的**所有异常发现**，而不仅停留在器官分割或区域级报告——使 AI 解读真正对齐放射科医生的临床需求（可解释、可定位、低幻觉）。

### 现有方法的局限
作者指出当前 CT 解读沿两条路线推进，各有硬伤：
1. **器官分割路线**（[[TotalSegmentator]] 等）：能精确标注解剖结构、做器官级 grounded report，但"分割器官"本身对临床不够——医生真正关心的是**异常、病灶、任何对诊断治疗有意义的异常变化**。
2. **报告生成路线**（CT-RATE、BIMCV-R 等）：基于配对的 CT-报告训练，但数据集**局限于特定区域**（多为胸部 CT），且**缺乏显式视觉 grounding**，可解释性差、幻觉风险高。
3. **医学 LVLM**：多模态医学大模型泛化强，但生成时**不能同时给出 grounding 证据**，幻觉难以察觉。
4. **根本性障碍**：缺少一个统一、临床有意义的**异常分类体系**，导致任务范围无法界定、无法建立持续 benchmark。

### 本文的动机
- 推动一次**范式转变**：从"区域报告 / 解剖分割"转向**以异常为中心**的全身解读；
- 先用分类体系把"异常"这件事**定义清楚**（image-based、层级化、模块化），再据此**标注数据**、**设计模型**、**做评测**，形成自洽闭环；
- 模型层面：用 VLM 的指令理解与文本生成能力做"大脑"，用分割模块提供**像素级 grounding 证据**，再把分割结果转成视觉提示叠回图像，借助 VLM 对视觉标记的感知能力生成精确描述——既可解释又抑制幻觉。

---

## 方法详解

### 模型架构

OmniAbnorm-CT 由两大部件组成（见 Figure 3）：一个 **[[VLM]]（Qwen2.5-VL-7B）** 与一个 **分割模块（6 层 U-Net）**；核心创新在于二者的**集成方式**，使模型能定位异常、生成描述，并通过视觉提示 + 文本指令灵活交互。

- **输入**: CT 图像 $\mathcal{I}\in\mathbb{R}^{H\times W\times D}$ + 文本指令 $\mathcal{T}$ +（可选）用户视觉提示 $\mathcal{V}\in\mathbb{R}^{H\times W}$（box/椭圆/轮廓）
- **Backbone**: [[Qwen2.5-VL]]-7B（仅微调插入的 [[LoRA]] 层）
- **核心模块**: 6 层 [[U-Net]] 分割模块（全量可训）+ 6 层 [[Cross-Attention|交叉注意力]] 桥接 VLM 隐状态与图像特征
- **输出**: 异常描述 $\mathcal{R}$ +（可选）关键切片上的 grounding 结果 $\mathcal{S}\in\mathbb{R}^{H\times W}$

整体映射写作（公式1）。

#### 工作流（核心机制）
1. 给定 $\mathcal{I}$ 和指令 $\mathcal{T}$，VLM 先**推理**出用户关心的具体异常（或任意异常）。例如提到"结核性脓胸病史"时，应聚焦胸膜增厚、钙化等相关异常。
2. 词表扩展一个特殊 token **`<SEG>`** 作为 grounding 请求；当 VLM 生成到 `<SEG>` 时，取其**最后一层解码器隐状态 $h_{\text{seg}}$**（编码了目标异常语义）作为分割模块的 prompt（公式2）。
3. 分割模块据 $\mathcal{I}$ 和 $h_{\text{seg}}$ 输出分割 $\mathcal{S}$，再把 $\mathcal{S}$ 转成 **box 形式的视觉提示 $\mathcal{V}$**，直接叠加到原图得到新输入 $\mathcal{I}'=\mathcal{I}\oplus\mathcal{V}$；VLM 据此**继续生成**异常描述 $\mathcal{R}$（公式3）。
4. 当用户手动勾画异常或想细化结果时，$\mathcal{V}$ 也可直接是**用户输入的视觉提示**——同一套机制天然支持交互式工作流。

> 这一设计巧妙利用了 VLM **感知视觉标记（visual marker）** 的内在能力（引用 ViP-LLaVA），把"先定位、再叠框、后描述"串成一条可解释链路。

### 核心模块

#### 模块1: Vision Language Model（核心大脑）
**设计动机**: 用 VLM 理解用户指令、恰当调用分割模块、并基于 grounding 证据撰写异常发现。

**具体实现**:
- 采用 [[Qwen2.5-VL]]-7B；通过特殊 token `<SEG>` 触发 grounding；取最后解码器层隐状态 $h_{\text{seg}}$ 传给分割模块。
- 仅训练插入的 [[LoRA]] 层（rank=32）以省算力。
- **只喂中心切片**给 VLM：消融（Table 7）显示多切片上下文反而无稳定增益——作者推测当前 LVLM 把切片独立 token 化，难以建模细微的层间空间关系。

#### 模块2: Segmentation Module（grounding 证据获取）
**设计动机**: 基于图像与 $h_{\text{seg}}$ 提供像素级 grounding 证据。

**具体实现**:
- encoder-decoder 主干（6 层 U-Net）先抽特征：多尺度图像嵌入 $v$（各 encoder 层下采样到统一分辨率后沿通道拼接）+ 最后 decoder 层的像素级稠密特征 $u$（公式4）。
- 用投影层 $f$ 对齐维度后，**交叉注意力** $\Phi_{\text{crossattn}}$ 以 $v$ 为 key/value、$f(h_{\text{seg}})$ 为 query，得到适配特征 $q$（公式5）。
- 分割预测由 $q$ 与 $u$ 做**点积**得到（公式6）。
- 实际输入 **9 个连续切片**（与标注设置一致），合作放射医生确认该窗口对绝大多数临床显著异常提供了足够的局部 3D 上下文。

### 关键公式与机制

#### 公式1: 整体映射（grounded interpretation）

$$
\{\mathcal{R},\mathcal{S}\}=\Phi_{\theta}(\mathcal{T},\mathcal{I},\mathcal{V})
$$

**含义**: 模型 $\Phi_\theta$ 接收文本指令、CT 图像、（可选）视觉提示，输出异常描述 $\mathcal{R}$ 与（必要时的）grounding 结果 $\mathcal{S}$。

**符号说明**:
- $\mathcal{I}\in\mathbb{R}^{H\times W\times D}$: CT 图像；$\mathcal{T}$: 文本指令
- $\mathcal{V}\in\mathbb{R}^{H\times W}$: 关键切片上的可选视觉提示（box/椭圆/轮廓）
- $\mathcal{R}$: 生成的异常描述；$\mathcal{S}\in\mathbb{R}^{H\times W}$: 可选的 grounding（分割）结果

#### 公式2: 调用分割模块（grounding 请求）

$$
\mathcal{S}=\Phi_{\text{seg}}(\mathcal{I},h_{\text{seg}}),\ \ h_{\text{seg}}\in\mathbb{R}^{d}
$$

**含义**: VLM 生成到 `<SEG>` 时取出隐状态 $h_{\text{seg}}$ 作为 prompt，驱动分割模块在图像上定位目标异常。

**符号说明**:
- $h_{\text{seg}}$: VLM 最后一层解码器对 `<SEG>` 的隐状态，编码了目标异常语义
- $\Phi_{\text{seg}}$: 分割模块

#### 公式3: 叠加视觉提示后继续生成

$$
\mathcal{R}=\Phi_{\text{MLLM}}(\mathcal{T},\mathcal{I}^{\prime}),\ \ \mathcal{I}^{\prime}=\mathcal{I}\oplus\mathcal{V}
$$

**含义**: 把分割结果转成 box 视觉提示 $\mathcal{V}$ 叠回原图得到 $\mathcal{I}'$，VLM 据此生成最终异常描述。

**符号说明**:
- $\oplus$: 叠加（superimposition），把视觉标记直接画在图像上
- $\mathcal{I}'$: 带视觉标记的新输入图像

#### 公式4: 分割主干特征抽取

$$
(v,u)=\Phi_{\text{seg}}(\mathcal{I}),\ \ v\in\mathbb{R}^{H^{\prime}\times W^{\prime}\times d^{\prime}},\ \ u\in\mathbb{R}^{H\times W\times d}
$$

**含义**: encoder-decoder 主干同时输出多尺度图像嵌入 $v$ 与像素级稠密特征 $u$。

**符号说明**:
- $v$: 各 encoder 层下采样到统一分辨率后沿通道拼接的多尺度嵌入
- $u$: 最后 decoder 层的像素级稠密特征

#### 公式5: 跨潜空间对齐（交叉注意力）

$$
q=\Phi_{\text{crossattn}}(v,f(h_{\text{seg}})),\ \ q\in\mathbb{R}^{d}
$$

**含义**: 用交叉注意力把分割 prompt $h_{\text{seg}}$ 与图像嵌入 $v$ 对齐，弥合 VLM 与分割模块之间的潜空间鸿沟。

**符号说明**:
- $f$: 维度对齐的投影层
- $\Phi_{\text{crossattn}}$: 以 $v$ 为 key/value、$f(h_{\text{seg}})$ 为 query 的交叉注意力模块；$q$: 适配后的特征

#### 公式6: 分割预测（点积）

$$
\mathcal{S}=q\cdot u,\ \ \mathcal{S}\in\mathbb{R}^{H\times W}
$$

**含义**: 适配特征 $q$ 与像素稠密特征 $u$ 做点积，得到关键切片上的分割预测。

#### 公式7-8: 训练目标（文本生成 + 分割联合损失）

$$
\mathcal{L}_{\text{txt}}=\text{CE}(\hat{\mathcal{R}},\mathcal{R}),\qquad \mathcal{L}_{\text{seg}}=\text{BCE}(\hat{\mathcal{S}},\mathcal{S})+\text{DICE}(\hat{\mathcal{S}},\mathcal{S})
$$

**含义**: VLM 与分割模块**联合训练**：文本端用交叉熵 [[Cross-Entropy|CE]]，分割端用 [[BCE]]+[[Dice Loss|DICE]]；两项**等权**相加（见附录 D.1）。

**符号说明**:
- $\hat{\mathcal{R}},\hat{\mathcal{S}}$: 预测的报告与分割；$\mathcal{R},\mathcal{S}$: 对应 ground-truth

#### 公式9-12: [[AbnormRubric]] 临床评测指标

$$
\text{Recall}=\text{TP}/(\text{TP}+\text{FN}),\qquad \text{Precision}=\text{TP}/(\text{TP}+\text{FP})
$$

$$
\text{Accuracy}=\frac{1}{N}\sum_{i=1}^{N}\mathbb{I}\big[\text{Attribution}_{i}\text{ aligns with GT}\big]
$$

$$
\text{Recall}^{*}=(1+\text{Accuracy})/2\times\text{Recall}
$$

$$
\text{Rubric}=2\times\frac{\text{Recall}^{*}\times\text{Precision}}{\text{Recall}^{*}+\text{Precision}}
$$

**含义**: AbnormRubric 用 LLM 评审，从 **(i) 检出率(Recall)、(ii) 描述属性准确度(Accuracy)、(iii) 幻觉(Precision)** 三方面综合评估生成报告，比纯词匹配/语义相似度更贴合临床判断。

**符号说明**:
- $\text{TP/FN/FP}$: LLM 在生成报告中判定的检出/漏检/幻觉异常数
- $\mathbb{I}$: 指示函数；对每个检出异常，判断其在位置、尺寸、形态、密度等放射学属性上是否与 GT 对齐
- $\text{Recall}^{*}$: 用属性准确度加权的"打折召回"，惩罚"检出但描述错"的情形
- $\text{Rubric}$: 最终分数，是 $\text{Recall}^{*}$ 与 $\text{Precision}$ 的 F1 式调和

> 这条指标链是本文评测的亮点：它把"找到没有 / 描述对不对 / 有没有瞎编"三件临床医生真正在意的事拆开量化，再合成单一可比分数。

---

## 关键图表

<!-- 图片均使用 arXiv HTML 在线链接，未本地化 -->

### Figure 1: OmniAbnorm-CT-14K Overview / 数据集概览

![Figure 1](https://arxiv.org/html/2506.03238v2/x1.png)

**说明**: 首个面向 CT 异常 grounding 与描述的大规模数据集。左：14.5K 张多平面全身 CT，覆盖 349 种代表性异常、82 个解剖结构、40 个主要系统/器官；右：异常在解剖结构与系统/器官上的分布（蓝色越深样本越密）。直观展示了"全身、多平面、细粒度异常"这一覆盖广度。

### Figure 2: Data Curation Overview / 数据构建流程

![Figure 2](https://arxiv.org/html/2506.03238v2/x2.png)

**说明**: (a) 从开源且经专家审核的 Radiopaedia 收集 image-report 对；(b)(c) 放射医生对任意异常做 grounding 标注、链接到报告文本描述、并归入资深医生设计的分类体系；标注进一步扩展为带模拟视觉提示与文本查询的指令数据。这张图解释了"数据从何而来、如何被标注与指令化"。

### Figure 3: OmniAbnorm-CT Architecture / 模型架构

![Figure 3](https://arxiv.org/html/2506.03238v2/x3.png)

**说明**: 核心架构。把 VLM 与分割模块桥接：生成异常描述过程中通过 `<SEG>` 触发分割获取 grounding 证据，再把分割转成视觉提示叠回图像，VLM 据此完成描述；同时支持文本指令与视觉提示的灵活交互。本图是理解公式 1–6 整个工作流的关键。

### Figure 4: Qualitative — Grounded Report Generation / grounded 报告定性对比

![Figure 4](https://arxiv.org/html/2506.03238v2/x4.png)

**说明**: grounded report generation 任务定性对比。BiomedParse 在后两例中**完全检不出异常**，导致 LLaVA-Med 生成与异常无关的报告；OmniAbnorm-CT 在所有案例中都成功定位异常并产出更准确的报告。直观佐证 grounding 证据对报告质量的因果作用。

### Figure 5: Qualitative — Text-guided Grounded Report Generation / 文本引导定性对比

![Figure 5](https://arxiv.org/html/2506.03238v2/x5.png)

**说明**: text-guided 任务对比。BiomedParse 在前两例**误定位**查询发现，LLaVA-Med 在后两例**误解读**分割结果；OmniAbnorm-CT 始终能定位被查询异常并产出更精确、临床对齐的报告。

### Figure 6: Qualitative — Visual Prompted Report Generation / 视觉提示定性对比

![Figure 6](https://arxiv.org/html/2506.03238v2/x6.png)

**说明**: visual prompted 任务对比。基线犯基础性错误（LLaVA-Med 把肾误标为胆囊、Qwen2.5-VL-7B 混淆肝与胆囊）；OmniAbnorm-CT 准确识别标记的异常并产出更高质量、临床一致的报告。

### Figure 7: Long-Tail Mitigation / 长尾缓解前后分布对比（附录）

![Figure 7](https://arxiv.org/html/2506.03238v2/x7.png)

**说明**: 长尾缓解策略前后的标注分布。(a) 器官分布对比；(b) 异常类别分布对比。干预前 top-20 器官（共 82）占 79.8% 标注、top-85 异常类别（共 340）占 80.1%；用 GPT-4o 筛选含罕见器官异常的病例优先标注后，欠表征器官标注占比 +19.6%、罕见异常类别 +11.6%，显著提升多样性。

---

### Table 1: Visual Prompted Report Generation / 视觉提示报告生成（内部验证）

各平面在类别内、再跨类别平均；✓ 表示用医学数据优化过的模型。

**Axial (n=2193)**

| Model | Rubric | BLEU-1 | BLEU-2 | RaTESc | BERTSc | METEOR | ROUGE-1 | ROUGE-L | RadG |
|-------|--------|--------|--------|--------|--------|--------|---------|---------|------|
| GPT-4o | 33.41 | 15.90 | 6.22 | 39.04 | 84.57 | 19.61 | 18.78 | 14.55 | 6.52 |
| Qwen2.5-VL-7B(OG) | 7.14 | 5.19 | 0.98 | 14.38 | 37.14 | 5.86 | 5.92 | 4.47 | 0.72 |
| ViP-LLaVA | 8.31 | 15.24 | 4.50 | 33.07 | 85.12 | 13.67 | 17.49 | 14.69 | 3.28 |
| MedDr ✓ | 20.82 | 9.43 | 2.35 | 32.27 | 78.62 | 9.01 | 12.19 | 10.29 | 3.67 |
| LLaVA-Med ✓ | 15.41 | 15.62 | 4.25 | 39.11 | 84.65 | 15.72 | 17.76 | 13.60 | 6.13 |
| BiomedGPT ✓ | 20.37 | 10.85 | 4.25 | 38.29 | 83.18 | 15.88 | 13.50 | 10.91 | 4.64 |
| **OmniAbnorm-CT ✓** | **35.69** | **18.03** | **6.65** | **42.81** | **86.35** | 19.40 | **21.66** | **17.61** | **9.98** |

**Coronal (n=750)**

| Model | Rubric | BLEU-1 | RaTESc | BERTSc | ROUGE-1 | RadG |
|-------|--------|--------|--------|--------|---------|------|
| GPT-4o | 25.41 | 9.25 | 27.26 | 59.17 | 12.08 | 4.26 |
| Qwen2.5-VL-7B(OG) | 26.25 | 13.43 | 37.34 | 84.19 | 14.73 | 3.34 |
| LLaVA-Med ✓ | 22.34 | 15.60 | 40.96 | 84.39 | 17.99 | 6.46 |
| MedDr ✓ | 26.79 | 8.22 | 32.98 | 76.53 | 11.71 | 4.13 |
| **OmniAbnorm-CT ✓** | **31.30** | **18.38** | **42.88** | **86.00** | **21.30** | **9.44** |

**Sagittal (n=591)**

| Model | Rubric | BLEU-1 | RaTESc | BERTSc | ROUGE-1 | RadG |
|-------|--------|--------|--------|--------|---------|------|
| GPT-4o | 24.39 | 10.23 | 32.41 | 72.53 | 13.77 | 4.00 |
| LLaVA-Med ✓ | 15.68 | 15.01 | 38.97 | 84.06 | 17.64 | 5.09 |
| Qwen2.5-VL-7B(OG) | 20.09 | 13.22 | 35.51 | 83.93 | 14.54 | 2.95 |
| **OmniAbnorm-CT ✓** | **29.00** | **17.29** | **41.77** | **85.62** | **20.43** | **7.51** |

**说明**: 半自动工作流（医生标框，模型描述）。OmniAbnorm-CT 在 30 个指标中赢下 29 个；相对最强基线，平均 BLEU-1 +2.4、RaTEScore +2.8、AbnormRubric +4.26。注意原始 Qwen2.5-VL-7B(OG) 在 axial 上 Rubric 仅 7.14，微调后跃升至 35.69，说明领域适配的关键作用。

### Table 2: Grounded Report Generation / grounded 报告生成（内部验证）

| Plane / Model | DSC | Rubric | B-1 | B-2 | B-3 | RaTESc | BERTSc | MTR | R-1 | R-L | RadG |
|---------------|-----|--------|-----|-----|-----|--------|--------|-----|-----|-----|------|
| **Axial** MedULS+LLaVA-Med | 14.92 | 3.00 | 10.60 | 5.31 | 0.50 | 26.84 | 72.92 | 15.90 | 12.46 | 10.50 | 2.26 |
| LiSA+LLaVA-Med | 20.62 | 4.87 | 12.85 | 6.53 | 0.70 | 31.09 | 81.68 | 19.11 | 14.46 | 12.32 | 2.75 |
| BiomedParse+LLaVA-Med | 15.69 | 3.45 | 13.35 | 6.86 | 0.80 | 31.61 | 83.04 | 19.62 | 14.98 | 12.77 | 2.84 |
| **OmniAbnorm-CT ✓** | **36.04** | **13.03** | **18.50** | **11.75** | **7.90** | **33.24** | **86.10** | **21.83** | **21.65** | **18.75** | **3.78** |
| **Coronal** LiSA+LLaVA-Med | 17.61 | 5.66 | 10.76 | 5.54 | 0.68 | 33.04 | 81.89 | 16.04 | 14.20 | 11.91 | 2.94 |
| BiomedParse+LLaVA-Med | 14.77 | 6.53 | 11.94 | 6.07 | 0.67 | 32.35 | 81.29 | 16.73 | 15.18 | 12.64 | 3.06 |
| **OmniAbnorm-CT ✓** | **31.65** | **13.29** | **19.13** | **11.86** | **7.81** | **35.47** | **85.58** | **20.66** | **22.14** | **18.26** | **4.55** |
| **Sagittal** LiSA+LLaVA-Med | 14.82 | 4.36 | 12.10 | 6.16 | 0.57 | 31.40 | 82.77 | 16.95 | 14.84 | 12.76 | 2.00 |
| BiomedParse+LLaVA-Med | 13.72 | 4.51 | 12.25 | 6.21 | 0.57 | 31.90 | 81.70 | 16.93 | 14.98 | 12.82 | 2.31 |
| **OmniAbnorm-CT ✓** | **34.38** | **10.11** | **17.10** | **10.63** | **7.10** | **33.13** | **85.25** | **19.05** | **20.92** | **17.67** | **3.70** |

**说明**: 全自动定位+报告。grounding 上 OmniAbnorm-CT 的 DSC 较最佳基线在三平面分别 **+15.42 / +14.04 / +19.56**，报告生成各指标全面最优，尤其 AbnormRubric、RadGraph、RaTEScore 提升明显。注意 B-3（BLEU-3）从基线 <1 跃升到 ~7–8，说明长程语言匹配质量质变。

### Table 3: Text-guided Grounded Report Generation / 文本引导 grounded 报告生成（内部验证）

| Plane / Model | DSC | Rubric | B-1 | B-2 | RaTESc | BERTSc | MTR | R-1 | R-L | RadG |
|---------------|-----|--------|-----|-----|--------|--------|-----|-----|-----|------|
| **Axial** MedULS+LLaVA-Med | 14.24 | 3.58 | 9.27 | 2.32 | 26.58 | 78.54 | 11.06 | 11.03 | 8.86 | 1.78 |
| LiSA+LLaVA-Med | 17.34 | 0.25 | 1.78 | 0.41 | 4.40 | 22.11 | 2.03 | 2.34 | 2.08 | 0.24 |
| BiomedParse+LLaVA-Med | 16.42 | 0.56 | 1.91 | 0.44 | 4.73 | 23.34 | 2.15 | 2.54 | 2.22 | 0.27 |
| **OmniAbnorm-CT ✓** | **32.40** | **14.82** | **11.94** | **3.33** | **34.80** | **83.85** | **12.94** | **14.65** | **11.94** | **4.32** |
| **Coronal** MedULS+LLaVA-Med | 12.21 | 3.30 | 10.08 | 2.42 | 27.81 | 78.85 | 10.72 | 11.57 | 9.06 | 1.34 |
| **OmniAbnorm-CT ✓** | **27.74** | **20.29** | **12.51** | **4.10** | **36.56** | **82.98** | **13.61** | **15.83** | **12.36** | **5.72** |
| **Sagittal** MedULS+LLaVA-Med | 11.61 | 3.35 | 10.07 | 2.60 | 26.87 | 78.33 | 10.61 | 11.83 | 9.16 | 1.79 |
| **OmniAbnorm-CT ✓** | **29.89** | **18.37** | **12.80** | **4.51** | **36.04** | **83.97** | **14.02** | **16.20** | **12.79** | **4.85** |

**说明**: 最难任务（响应特定异常的文本查询，含 3:1 的"存在/不存在"查询模拟真实诊断）。grounding DSC 三平面 **+14.24 / +14.14 / +14.54**；报告生成在通用重叠指标（平均 BLEU-1 +1.3、ROUGE-1 +2.2）与临床指标（平均 AbnormRubric +14.42）上均大幅领先。注意 LiSA/BiomedParse+LLaVA-Med 在该任务上 Rubric 近乎归零（0.1–0.6），暴露文本提示分割+解读级联管线的脆弱。

### Table 4: External Validation / 外部验证（CT-RATE 65 例胸部 CT，271 个异常）

| Task / Model | DSC | Rubric | B-1 | RaTESc | BERTSc | R-1 | RadG |
|--------------|-----|--------|-----|--------|--------|-----|------|
| **Visual Prompted** GPT-4o | - | 19.15 | 17.37 | 37.86 | 85.52 | 19.16 | 3.73 |
| LLaVA-Med ✓ | - | 12.22 | 17.63 | 34.96 | 85.52 | 19.01 | 2.92 |
| **OmniAbnorm-CT ✓** | - | **24.09** | **19.09** | **39.87** | **86.64** | **21.02** | **4.27** |
| **Grounded** LiSA+LLaVA-Med | 18.34 | 4.37 | 12.33 | 35.43 | 83.57 | 15.87 | 2.69 |
| BiomedParse+LLaVA-Med | 11.78 | 5.32 | 11.98 | 34.76 | 83.47 | 16.10 | 2.87 |
| **OmniAbnorm-CT ✓** | **22.20** | **8.62** | **18.55** | **34.93** | **86.09** | **22.36** | 2.57 |
| **Text-guided** MedULS+LLaVA-Med | 11.69 | 3.23 | 9.47 | 25.57 | 74.55 | 10.81 | 1.32 |
| **OmniAbnorm-CT ✓** | **14.13** | **15.12** | **12.38** | **32.25** | 76.34 | **14.85** | **3.55** |

**说明**: 在土耳其采集的 CT-RATE 外部胸部 CT 上，OmniAbnorm-CT 三任务依旧领先：visual prompted 的 AbnormRubric +4.94；grounded / text-guided 的 DSC 分别 +10.57 / +2.44，证明跨数据集泛化性。注意外部集 DSC 整体较内部低（域偏移所致），但仍稳居第一。

### Table 5: Ablation — Segmentation Module (Text-guided) / 分割模块消融（文本引导任务）

| Plane / Model | BLEU-1 | RaTESc | BERTSc | METEOR | ROUGE-1 | RadG |
|---------------|--------|--------|--------|--------|---------|------|
| **Axial** w/o Seg | 4.50 | 20.53 | 31.06 | 7.04 | 8.50 | 7.25 |
| **Axial** OmniAbnorm-CT | **11.94** | **34.80** | **83.85** | **12.94** | **14.65** | 4.32 |
| **Coronal** w/o Seg | 4.50 | 19.07 | 28.34 | 6.27 | 7.96 | 6.17 |
| **Coronal** OmniAbnorm-CT | **12.51** | **36.56** | **82.98** | **13.61** | **15.83** | 5.72 |
| **Sagittal** w/o Seg | 4.17 | 16.62 | 25.43 | 5.84 | 7.06 | 5.67 |
| **Sagittal** OmniAbnorm-CT | **12.80** | **36.04** | **83.97** | **14.02** | **16.20** | 4.85 |

**关键发现**: 去掉分割模块在 27 个指标中 23 个显著下降；缺 grounding 证据时模型难以准确检出被查询的异常（BERTScore 从 ~83 暴跌到 ~25–31 尤其触目）。证明 grounding 证据对该任务是必需的。

### Table 6: Ablation — Segmentation Module (Grounded) / 分割模块消融（grounded 任务）

| Plane / Model | BLEU-1 | RaTESc | BERTSc | METEOR | ROUGE-1 | RadG |
|---------------|--------|--------|--------|--------|---------|------|
| **Axial** w/o Seg | 7.73 | 27.87 | 73.14 | 17.65 | 15.18 | 3.42 |
| **Axial** OmniAbnorm-CT | **19.00** | **33.85** | **86.24** | **22.45** | **22.27** | **4.60** |
| **Coronal** w/o Seg | 8.12 | 31.03 | 75.55 | 15.64 | 15.27 | 3.84 |
| **Coronal** OmniAbnorm-CT | **19.28** | **35.40** | **85.56** | **20.81** | **22.35** | **4.65** |
| **Sagittal** w/o Seg | 8.79 | 31.21 | 77.41 | 16.18 | 15.86 | 3.91 |
| **Sagittal** OmniAbnorm-CT | **16.69** | **33.45** | **84.94** | **18.77** | **20.67** | 3.83 |

**关键发现**: grounded 任务上去 Seg 同样在全部指标上落后，再次确认分割驱动的证据对准确 grounding 与临床可信报告不可或缺。

### Table 7: Ablation — Context Range / 上下文切片数消融（visual prompted，基座 Qwen2.5-VL-7B）

| Plane / Input | BLEU-1 | RaTESc | BERTSc | METEOR | ROUGE-1 | RadGraph |
|---------------|--------|--------|--------|--------|---------|----------|
| **Axial** 4 Adjacent | 9.39 | 31.12 | 83.52 | 14.90 | 11.51 | 2.46 |
| 2 Adjacent | 9.24 | 31.47 | 83.49 | 15.05 | 11.41 | 2.49 |
| **No Adjacent (center)** | **11.57** | **38.35** | **84.10** | **16.68** | **14.26** | **3.85** |
| **Coronal** 4 Adjacent | 10.93 | 32.95 | 83.49 | 15.82 | 13.29 | 3.62 |
| **No Adjacent (center)** | **13.62** | 37.26 | **84.19** | 13.49 | **14.97** | 3.37 |
| **Sagittal** 2 Adjacent | 11.48 | 33.86 | 83.32 | 15.70 | 13.64 | 3.75 |
| **No Adjacent (center)** | **13.28** | **35.67** | **83.98** | 13.01 | **14.75** | 2.91 |

**关键发现**: 反直觉——给 VLM 增加相邻切片**没有稳定增益**，仅用中心切片往往最好。作者推测当前 LVLM 把切片独立 token 化，难建模细微层间空间关系。故 OmniAbnorm-CT 的 VLM 端只喂中心切片（分割模块仍用 9 切片）。

### Table 8: Ablation — Visual Prompt Type / 视觉提示形式消融（OmniAbnorm-CT 固定为基线）

| Plane / Prompt | BLEU-1 | RaTESc | BERTSc | METEOR | ROUGE-1 | RadGraph |
|----------------|--------|--------|--------|--------|---------|----------|
| **Axial** Center Crop | 11.70 | 33.03 | 84.84 | 12.77 | 14.58 | 3.92 |
| Ellipse | 12.43 | 35.14 | 84.87 | 13.67 | 15.55 | 5.27 |
| Contour | 12.60 | 35.74 | 85.09 | 14.12 | 15.97 | 5.81 |
| Bounding Box | 12.35 | 35.46 | 84.92 | 13.65 | 15.76 | 5.63 |
| **Max (oracle)** | **18.03** | **42.81** | **86.35** | **19.40** | **21.66** | **9.98** |
| **Coronal** Center Crop | 10.94 | 33.57 | 84.48 | 11.92 | 14.23 | 3.87 |
| **Max (oracle)** | **18.38** | **42.88** | **86.00** | **19.00** | **21.30** | **9.44** |
| **Sagittal** Center Crop | 10.98 | 32.23 | 84.20 | 11.98 | 14.45 | 3.05 |
| **Max (oracle)** | **17.29** | **41.77** | **85.62** | **17.45** | **20.43** | **7.51** |

**关键发现**: 中心裁剪在 27 个指标中 24 个最差（丢失上下文有害）；椭圆/轮廓/box 表现相近。per-case oracle（取四者最佳）显著超过任一单一提示，说明**按形态/位置自适应选择视觉提示**有潜在收益。

### Table 9: Dataset Comparison / 与公开 CT 数据集对比

| Dataset | Task | Anatomy | Plane | Report | #Category | #Image |
|---------|------|---------|-------|--------|-----------|--------|
| DeepLesion | Lesion Det. | Whole Body | Axial | | - | 33K |
| TotalSegmentator | Organ Seg. | Whole Body | Axial | | - | 1.2K |
| AbdomenAtlas | Organ Seg. | Chest & Abdomen | Axial | | - | 8K |
| CT-RATE | Report Gen. | Chest | Axial | ✓ | - | 26K |
| BIMCV-R | Report Gen. | Chest | Axial | ✓ | - | 8K |
| ReXGroundingCT | Lesion Seg. | Chest | Axial | ✓ | 14 | 3K |
| MSD/KiTS/ULS 等 19 套 | Lesion Seg. | 单器官 | Axial | | 1–2 | 20–750 |
| **OmniAbnorm-CT-14K (Axial)** | Seg.+Det.+Gen. | **Whole Body** | Axial | ✓ | **340** | **10K** |
| **OmniAbnorm-CT-14K (Coronal)** | Seg.+Det.+Gen. | Whole Body | Coronal | ✓ | 255 | 2K |
| **OmniAbnorm-CT-14K (Sagittal)** | Seg.+Det.+Gen. | Whole Body | Sagittal | ✓ | 223 | 1.6K |

**说明**: 已有数据集要么仅器官分割（无异常类别）、要么仅单器官病灶、要么仅胸部报告（无 grounding）。OmniAbnorm-CT-14K 是**首个**覆盖全身、多平面、带细粒度异常类别标签 + grounding + 报告的数据集（轴位即 340 类异常）。CCC18/DeepLesion 对每个病灶无类别标签；部分数据集虽各向同性但按轴位采集。

### Tables 10–22: Detailed Abnormality Distribution / 异常类别详细分布（附录）

**说明**: 附录 E.5 用一张跨 13 页的大表（Table 10–22）逐条列出 OmniAbnorm-CT-14K 中**每个器官 → 解剖结构 → 异常类别**在轴位/冠状/矢状三平面上的标注数。组织层级与正文分类体系一致（如 Brain→Cerebral parenchyma→Brain parenchymal soft tissue mass: Axial 280 / Coronal 57 / Sagittal 57；Spine、Pituitary 等各结构类似）。该表是 Figure 1 分布图的精确数值版，体现"349/404 类异常被实际标注、长尾被刻意补齐"。因条目数千、属纯数据清单，此处不逐行转录。

---

## 实验

### 数据集 / 基准

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| OmniAbnorm-CT-14K | 14.5K 图像 / 18,969 异常（轴位 9,990、冠状 2,738、矢状 1,803 图） | 多平面全身、404 类异常体系（覆盖 349 类） | 训练 + 内部验证 |
| 公开病灶分割数据 | ULS-23 全套 + 10 套额外（MSD/KiTS/LIDC 等） | 单器官病灶，用于增强 grounding | 训练补充 |
| PubMedVision (CT VQA) | - | 多样问答，保持泛化 | 训练补充 |
| CT-RATE（采样 65 例胸部 CT） | 271 个异常标注 | 土耳其采集、非增强胸部 CT | 外部验证 |

数据划分：三平面独立划分；每类别至少分 5 例（冠/矢 2 例）入轴位测试集，其余按 3:1 训练-测试；同一患者图像不跨集（防泄漏）。

### 实现细节

- **VLM**: Qwen2.5-VL-7B，仅训 LoRA（rank=32）；分割模块：6 层 U-Net（全量训）+ 6 层交叉注意力；VLM 喂中心切片、分割模块喂 9 连续切片。
- **训练目标**: 文本 CE + 分割 (BCE+DICE)，等权联合。
- **优化**: AdamW，lr=2e-4，5% warmup，总 250K 迭代；8×A100-80G，每卡 batch 1、梯度累积 4。图像 pad+rescale 到 512×512。
- **数据采样比**: 视觉提示 : grounded : 文本引导 : 通用 VQA = 1:1:1:1；grounding 任务中 OmniAbnorm-CT-14K : 公开病灶分割 = 6:4。
- **分阶段训练（附录 D.3，4×A100）**: 先预训分割模块 50K 步（~10h）→ Qwen2.5-VL 在视觉提示任务微调 20K 步（~8h）→ VLM 与分割联合训 grounded 任务 100K 步（~48h）。
- **推理延迟**: visual prompted 0.49s / grounded 0.98s / text-guided 1.01s 每样本（1×A100，混合精度无量化）。

### 关键实验结论

- **内部验证**: 三任务全面 SOTA。视觉提示任务 30 指标赢 29；grounded 任务 DSC 较最佳基线三平面 +15.42/+14.04/+19.56；文本引导任务 DSC +14.24/+14.14/+14.54、AbnormRubric 平均 +14.42。
- **外部验证（CT-RATE）**: 三任务依旧领先，visual prompted Rubric +4.94，grounded/text-guided DSC +10.57/+2.44，泛化性良好。
- **消融**: (1) 去分割模块在两任务大量指标下降，grounding 证据必需；(2) 多切片上下文对 VLM 报告无稳定增益，故只喂中心切片；(3) 视觉提示中裁剪最差、椭圆/轮廓/box 相近，oracle 远超单一提示。
- **数据质量**: 4 名标注者中 3 名四项指标满分，1 名报告一致性/分类准确率 95%；标注者间一致性 DSC=84.5、BLEU-1=72.2。

---

## 批判性思考

### 优点
1. **闭环的方法论**: taxonomy→data→model→evaluation 四位一体，问题定义清晰、可持续 benchmark；分类体系由 7 位资深医生共建，临床根基扎实。
2. **可解释 + 抑幻觉的架构巧思**: `<SEG>` 触发分割 → 分割转 box 视觉提示叠回图 → VLM 据标记描述，把"定位证据"显式注入生成链路，定性图（Fig 4–6）清楚展示其纠正了基线的器官混淆/漏检。
3. **评测指标贴临床**: AbnormRubric 把检出/属性准确/幻觉拆开量化再合成，比 BLEU/BERTScore 更有诊断意义；外部验证 + 严格的标注质检与一致性分析增强了可信度。
4. **数据稀缺性贡献突出**: 首个全身多平面、细粒度异常 grounding+描述数据集，且承诺标注开源。

### 局限性
1. **仅标关键切片而非整卷**: 受专家标注成本所限，模型与评测都建立在"代表性 2D 切片"上，与真实临床的全 3D 阅片仍有差距（作者自承）。
2. **VLM 体积感知弱**: 多切片上下文反而无益（Table 7）暴露当前 LVLM 难建模层间 3D 关系；模型本质仍是 2D 解读，缺真正 3D 编码器。
3. **绝对分数仍偏低**: 即便 SOTA，AbnormRubric 多在 10–35 区间、外部 DSC 仅 14–22，离临床可用尚远；BLEU-3 等长程匹配也很低。
4. **级联依赖与基线偏弱**: grounded 任务的基线是"分割模型+LLaVA-Med"的拼接管线，本身脆弱（文本引导任务 Rubric 近零），对比优势部分来自基线选择；缺乏与同类端到端 grounded-VLM 的直接较量（因领域内确无现成方案）。

### 潜在改进方向
1. 引入 3D 视觉编码器 + 更大规模全身 CT 预训练，提升体积感知（作者已列为未来工作）。
2. 用半自动工具扩到**全卷标注**，让训练/评测脱离"关键切片"假设。
3. 落地 Table 8 的"按形态/位置自适应选视觉提示"策略，逼近 oracle 上限。
4. 把分割模块与 VLM 的潜空间对齐做得更紧（目前靠 `<SEG>` 隐状态 + 交叉注意力 + box 叠加的间接耦合）。

### 可复现性评估
- [x] 标注数据开源（声明 release；图像经 Radiopaedia 申请后可获取）
- [ ] 预训练模型权重（HTML 正文未明确声明开源）
- [x] 训练细节完整（附录 D 给出超参、分阶段步数、运行时与算力）
- [x] 数据集可获取（OmniAbnorm-CT-14K 标注开源；外部用公开 CT-RATE）

---

## 速查卡片

> [!summary] OmniAbnorm-CT: Abnormality-Centric Whole-Body CT Interpretation
> - **核心**: 以异常为中心重做全身 CT 解读——404 类异常分类体系 + 19K grounding 标注的 OmniAbnorm-CT-14K + 缝合 VLM 与分割模块的 OmniAbnorm-CT + 临床指标 AbnormRubric。
> - **方法**: Qwen2.5-VL-7B(LoRA) 生成时用 `<SEG>` 触发 6 层 U-Net 分割，取隐状态 $h_{\text{seg}}$ 做 prompt，把分割转 box 视觉提示叠回图 ($\mathcal{I}'=\mathcal{I}\oplus\mathcal{V}$) 后继续描述；文本 CE + 分割 BCE+DICE 联合训练。
> - **结果**: 内部三任务全面 SOTA（grounded DSC +15/+14/+20，文本引导 AbnormRubric +14.42）；外部 CT-RATE 泛化领先；消融证明分割证据必需、VLM 只需中心切片。
> - **数据**: 14.5K 多平面全身 CT / 18,969 异常 / 349 类被标注；标注承诺开源。

---

*笔记创建时间: 2026-06-29*
