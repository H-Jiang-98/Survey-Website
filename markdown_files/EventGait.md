---
title: "EventGait: Towards Robust Gait Recognition with Event Streams"
method_name: "EventGait"
authors: [Senyan Xu, Shuai Chen, Chuanfu Shen, Kean Liu, Zhijing Sun, Chengzhi Cao, Xueyang Fu]
year: 2026
venue: CVPR
tags: [gait-recognition, event-camera, spiking-neural-network, mixture-of-experts, vision-foundation-model, dual-stream]
zotero_collection: _inbox
image_source: online
arxiv_html: https://arxiv.org/abs/2605.22139
created: 2026-06-29
---

# EventGait: Towards Robust Gait Recognition with Event Streams

## 元信息

| 项目 | 内容 |
|------|------|
| 作者 | Senyan Xu, Shuai Chen（共同一作）, Chuanfu Shen, Kean Liu, Zhijing Sun, Chengzhi Cao, Xueyang Fu（通讯） |
| 机构 | 中国科学技术大学（USTC）、电子科技大学（UESTC） |
| 会议 | CVPR 2026 |
| 类别 | 步态识别 / 事件相机 / Locomotion |
| 日期 | 2026-05（arXiv v1） |
| 项目主页 | https://github.com/QUEAHREN/EventGait |
| 链接 | [arXiv](https://arxiv.org/abs/2605.22139) / [Code](https://github.com/QUEAHREN/EventGait) |

> 备注：本论文的 arXiv HTML 渲染（`https://arxiv.org/html/2605.22139`）暂未上线，正文与图表内容取自官方 PDF（`https://arxiv.org/pdf/2605.22139`）。下方"关键图表"中能在线引用的只有作者 GitHub 仓库提供的 workflow 图（即论文 Figure 2），其余图（Figure 1/3/4/5）无可用在线外链，故以文字忠实转述其内容与作用。

---

## 一句话总结

> 用事件相机的微秒级时间分辨率，把步态拆成"动态运动流（脉冲专家混合 MoSE）+ 静态形状流（用视觉基础模型做跨模态结构对齐 CroSA）"双流建模，在常规光照下追平传统相机、在低光下大幅超越，并配套发布两个合成事件步态基准。

---

## 核心贡献

1. **双流事件步态框架 EventGait**: 首个明确"分别建模静态形状与动态运动"的端到端事件步态网络，纠正了以往把事件长时间聚合成 event image、丢失高频时序动态的做法。
2. **脉冲专家混合 MoSE（Mixture of Spiking Experts）**: 在动态流里用一组具有不同膜时间常数 $\tau$ 的脉冲神经元专家，加一个轻量门控网络自适应加权，从而在快/慢运动、亮/暗光照等差异极大的事件信号下都能稳健感知运动。
3. **跨模态结构对齐 CroSA（Cross-modal Structure Alignment）**: 用冻结的视觉基础模型（[[DINOv2]]）当教师，把稠密结构先验蒸馏进事件静态编码器，解决长时聚合后事件仍然稀疏、难以学到稠密人体形状的问题（**仅训练时使用**）。
4. **两个大规模合成事件步态基准**: 提出 RGB→事件的合成流水线（基于 [[v2e]]），把 CCGR-Mini 与 SUSTech1K 转换为 **CCGR-Mini-E**（47,884 序列 / 970 受试者）与 **SUSTech1K-E**（25,239 序列 / 1,050 身份），填补缺乏大规模事件步态数据集的空白。

---

## 问题背景

### 要解决的问题
如何做**鲁棒的步态识别**——尤其是在不受控环境（光照变化、运动模糊、遮挡）下仍能稳定地非侵入式识别身份。

### 现有方法的局限
- **传统 RGB 相机步态**：时间分辨率低（约 30 ms）、动态范围窄（< 80 dB），低光、运动模糊会严重破坏视觉线索，识别不可靠（见 Figure 1：白天 RGB 轮廓清晰，夜间 RGB 轮廓质量极差）。
- **轮廓/骨架/解析等中间表征**：高度依赖上游分割/姿态估计的质量，误差会传递。
- **LiDAR 步态**：对光照鲁棒，但传感器昂贵（约 75K USD）、能耗高，难以规模化部署。
- **已有事件步态方法（EVGait[83]、GaitGraph 类[84]）**：把事件流在**长时间窗口**内聚合成网格化 event image，存在两个根本缺陷：(i) 坍缩了精细时间分辨率，丢失对步态至关重要的高频动态；(ii) 产生过于稀疏的空间表征，标准 CNN 难以解释成稠密、判别性的外观特征。而且它们只在小规模数据集（约 100 身份）上验证过。

### 本文的动机
- 事件相机（DVS）天生具备**微秒级时间分辨率（< 3 μs）+ 超高动态范围（> 120 dB）**，异步只记录亮度变化，天然捕捉动态运动、抑制静态噪声与衣着纹理/颜色等无关线索 —— 非常契合步态识别。
- 核心洞见：**事件步态不该只编码空间体形，更要保留高时间分辨率事件中的精细动态**。因此把"形状"和"运动"解耦成双流：静态流补回稠密形状（靠 VFM 蒸馏），动态流直接用脉冲动力学建模高频运动（靠 MoSE）。

---

## 方法详解

### 模型架构

EventGait 采用 **双流（dual-stream）** 设计（见下方 Figure 2 workflow）：

- **输入**: 异步事件流，先转成两类体素切片
  - 短时切片 $\mathbf{E}_d$（small $\Delta T$）→ 保留精细时序，喂给动态流
  - 长时切片 $\mathbf{E}_s$（whole window $T$）→ 聚合出全局形状，喂给静态流
- **Dynamic Motion Stream（动态运动流）**: 以 [[Mixture of Spiking Experts|MoSE]] 为基本学习单元，从短时切片产出动态运动特征 $\mathbf{F}_d$
- **Static Shape Stream（静态形状流）**: CNN 编码器，用 [[Cross-modal Structure Alignment|CroSA]] 训练，从长时事件 $\mathbf{E}_s$ 产出稠密形状特征 $\mathbf{F}_s$
- **Fusion & Decoder & Classifier（融合与解码分类）**: 把 $[\mathbf{F}_s;\mathbf{F}_d]$ 经融合模块 $\Phi(\cdot)$ 得到统一步态描述子 $\mathbf{F}_{\text{gait}}$，再做下游识别
- **训练损失**: 交叉熵 $\mathcal{L}_{ce}$ + 三元组 $\mathcal{L}_{tri}$ + 跨模态对齐 $\mathcal{L}_{align}$（对齐项仅训练时启用）

### 核心模块

#### 模块1: Event Representation（两尺度时间设计）

**设计动机**: 既要保留高频动态又要拿到稠密形状，于是把同一段事件流在**两个时间尺度**上体素化。

**具体实现**:
- 事件被记为元组 $e_i=(x_i,y_i,t_i,p_i)$，$p_i\in\{+1,-1\}$ 为对数光强变化的极性，当对数光强变化超过阈值 $c$ 时触发（公式1）。
- 在曝光窗口 $T$ 内把事件聚合成时序切片：把 $T$ 均匀分成 $K$ 个 bin，用三角核（公式2）把每个事件软分配到相邻 bin，正负极性分通道保存，得到 $\mathbf{E}\in\mathbb{R}^{2\times K\times H\times W}$，保留 bin 内的亚精度时序与运动连续性。
- 在每个长曝光窗 $T$ 内提供 $K$ 个宽度为 $\Delta T=T/K$ 的短时切片 $\mathbf{E}_d$（喂动态流）；同时把整个 $T$ 内事件聚合成一个长时切片 $\mathbf{E}_s$（喂静态流）。

#### 模块2: Mixture of Spiking Experts（MoSE，动态运动流核心）

**设计动机**: 短时事件切片高时间频率、空间稀疏，传统 CNN/RNN 难以精确捕捉运动且与事件稀疏性不兼容；而单个脉冲神经元的行为被膜时间常数 $\tau$ 决定，**单一 $\tau$ 无法同时适配快/慢、亮/暗的多样信号**（见 Figure 3(a)：小 $\tau$ "快"神经元擅长亮场景/快动作的高频脉冲串，但难积分低光稀疏信号；大 $\tau$ "慢"神经元在低光/慢动作下靠长时积分更稳健，却在亮/快场景引入时序噪声）。

**具体实现**:
- 基础单元用 [[Leaky Integrate-and-Fire|LIF]] 脉冲神经元，膜电位 $U(t)$ 按公式3 演化、超过阈值 $U_{th}$ 即发放并复位；突触电流 $I(t)$ 由历史脉冲经时间核 $\psi(\cdot)$ 累积（公式4）。脉冲神经元天然适合从事件表征里**累积动态线索**。
- 受 [[Mixture of Experts|MoE]] 启发，构造 $N$ 个并行脉冲专家 $\{\mathcal{E}_1,\dots,\mathcal{E}_N\}$，每个用不同膜时间常数 $\tau_i$ 初始化（覆盖不同时序尺度）。
- 一个轻量脉冲门控网络 $\mathcal{G}(\cdot)$ 分析事件的动态模式，给每个专家算混合系数 $\{\alpha_i\}$，加权得到最终动态运动特征（公式5）。小 $\tau_i$ 专家专攻高频/亮场景，大 $\tau_i$ 专家整合暗场景稀疏事件。

#### 模块3: Cross-modal Structure Alignment（CroSA，静态形状流训练策略）

**设计动机**: 长时聚合 $\mathbf{E}_s$ 虽给出全局统计，但仍是稀疏结构，编码器很难从中自学到复杂稠密的人体形状先验，需要外部引导。

**具体实现**:
- 用冻结的 [[Vision Foundation Model|VFM]]（[[DINOv2]]，以其对细粒度结构信息的强把握著称）当**教师** $\mathcal{F}_{\text{teacher}}$，把事件静态编码器 $\mathcal{F}_{\text{student}}$（CNN）当学生。
- 给定与事件同步的 RGB 帧 $\mathbf{I}_c$，先做灰度化 $\mathbf{I}_g=\mathcal{I}(\mathbf{I}_c)$（避免 RGB 颜色无关扰动），喂教师得 $\mathbf{z}_{\text{img}}$；学生对 $\mathbf{E}_s$ 编码后经一层对齐卷积 $\mathcal{A}(\cdot)$ 投影得 $\mathbf{z}_{\text{evs}}$（公式6）。
- 对齐损失 $\mathcal{L}_{\text{align}}$ 用 $\ell_2$ 距离拉近 $\mathbf{z}_{\text{evs}}$ 与 $\mathbf{z}_{\text{img}}$（公式7），把稠密结构先验蒸馏进事件静态编码器。**该路径仅训练时存在（图中橙色虚线"Only for training"）**，推理时不需要 RGB。

### 关键公式与机制

#### 公式1: [[Event Camera|事件触发]]

$$
p_i = \operatorname{sg}\!\left(\log\frac{L(x_i,y_i,t_i)}{L(x_i,y_i,t_i')}\right),\quad \left|\log\frac{L(x_i,y_i,t_i)}{L(x_i,y_i,t_i')}\right| > c
$$

**含义**: 像素 $(x_i,y_i)$ 处对数光强相对上次时刻 $t_i'$ 的变化超过阈值 $c$ 时触发一个事件，极性 $p_i$ 取该变化的符号。

**符号说明**:
- $L(\cdot)$: 像素亮度（强度）；$t_i'$: 该像素上一次事件时间
- $\operatorname{sg}(\cdot)$: 取符号函数；$c$: 触发阈值

#### 公式2: 事件体素化（时序切片）

$$
\mathbf{E}_p(x,y,k) = \sum_{e_i\in\mathcal{E}_p}\max\!\left(0,\ 1-\frac{|t_i-t_k|}{\Delta T}\right)
$$

**含义**: 把曝光窗 $T$ 均分成 $K$ 个 bin，用三角核将每个事件按时间距离软分配到第 $k$ 个 bin，保留 bin 内亚精度时序。

**符号说明**:
- $\mathcal{E}_p$: 极性为 $p\in\{+,-\}$ 的事件集合（正负分通道）
- $\Delta T=T/K$: 每个 bin 的时间间隔；$t_k$: 第 $k$ 个 bin 的中心时刻
- 最终 $\mathbf{E}\in\mathbb{R}^{2\times K\times H\times W}$（2 = 正负极性通道）

#### 公式3: [[Leaky Integrate-and-Fire|LIF]] 神经元动力学

$$
\tau\frac{dU(t)}{dt} = -U(t) + R\cdot I(t),\qquad S(t)=\Theta\!\left(U(t)-U_{th}\right)
$$

**含义**: 膜电位 $U(t)$ 在漏电衰减与突触电流注入下演化，超过阈值 $U_{th}$ 即发放脉冲 $S(t)=1$ 并复位到 $U_{reset}$。这是 MoSE 的基本积分机制。

**符号说明**:
- $\tau$: 膜时间常数（控制衰减速率，是区分"快/慢"专家的关键超参）
- $R$: 膜电阻；$U_{th}$: 发放阈值；$\Theta(\cdot)$: Heaviside 阶跃函数

#### 公式4: 突触电流

$$
I(t) = \sum_i w_i \sum_k \psi\!\left(t - t_i^{(k)}\right)
$$

**含义**: 突触电流由所有前驱神经元的历史脉冲经时间核累积而成，建模突触后电流衰减。

**符号说明**:
- $w_i$: 突触权重；$t_i^{(k)}$: 神经元 $i$ 的第 $k$ 次脉冲时刻
- $\psi(\cdot)$: 描述突触后电流衰减的时间核

#### 公式5: MoSE 专家混合

$$
\hat{\mathbf{E}}_t = \sum_{i=1}^{N}\alpha_i\,\mathcal{E}_i(\mathbf{E}_t)
$$

**含义**: $N$ 个具有不同 $\tau_i$ 的脉冲专家对同一输入事件张量 $\mathbf{E}_t$ 分别处理，由门控系数 $\alpha_i$ 加权求和，得到自适应的动态运动特征。

**符号说明**:
- $\mathcal{E}_i$: 第 $i$ 个脉冲专家（膜常数 $\tau_i$）；$\mathbf{E}_t$: 输入事件张量
- $\alpha_i$: 门控网络 $\mathcal{G}(\cdot)$ 给出的混合系数；小 $\tau_i$ 偏高频亮场景、大 $\tau_i$ 偏暗场景稀疏事件

#### 公式6: 跨模态特征投影（CroSA）

$$
\mathbf{z}_{\text{img}} = \mathcal{F}_{\text{teacher}}(\mathbf{I}_g),\qquad \mathbf{z}_{\text{evs}} = \mathcal{A}\!\left(\mathcal{F}_{\text{student}}(\mathbf{E}_s)\right)
$$

**含义**: 冻结 VFM 教师从灰度 RGB $\mathbf{I}_g$ 提结构特征 $\mathbf{z}_{\text{img}}$；事件学生编码器对长时切片 $\mathbf{E}_s$ 编码后经对齐卷积 $\mathcal{A}(\cdot)$ 投影到同一空间得 $\mathbf{z}_{\text{evs}}$。

**符号说明**:
- $\mathbf{I}_g=\mathcal{I}(\mathbf{I}_c)$: RGB 帧灰度化（去颜色扰动）
- $\mathcal{F}_{\text{teacher}}$: 冻结 DINOv2；$\mathcal{F}_{\text{student}}$: 待训练事件静态编码器；$\mathcal{A}(\cdot)$: 对齐卷积层

#### 公式7: 跨模态对齐损失

$$
\mathcal{L}_{\text{align}} = \left\|\mathbf{z}_{\text{evs}} - \mathbf{z}_{\text{img}}\right\|_2^2
$$

**含义**: 用 $\ell_2$ 距离把事件结构特征拉向 VFM 的稠密结构特征，蒸馏形状先验。

**符号说明**:
- $\ell_2$: 欧氏距离；仅训练时计算（推理无需 RGB）

#### 公式8: 双流融合

$$
\mathbf{F}_{\text{gait}} = \Phi\!\left([\mathbf{F}_s;\mathbf{F}_d]\right)
$$

**含义**: 把静态形状特征 $\mathbf{F}_s$ 与动态运动特征 $\mathbf{F}_d$ 拼接后经融合模块 $\Phi(\cdot)$ 得到统一步态描述子，整合"事件原生感知 + 稠密视觉先验"。

**符号说明**:
- $\mathbf{F}_s$: 静态流（来自长时 $\mathbf{E}_s$）；$\mathbf{F}_d$: 动态流（来自短时 $\mathbf{E}_d$）
- $\Phi(\cdot)$: 常规融合模块

#### 公式9: 总训练目标

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{ce} + \mathcal{L}_{tri} + \lambda_d\,\mathcal{L}_{\text{align}}
$$

**含义**: 识别用交叉熵 + 三元组损失，外加权重为 $\lambda_d$ 的跨模态对齐项。消融显示 $\lambda_d=0.2$ 最优（过小引导不足、过大引入身份无关线索如衣纹）。

**符号说明**:
- $\mathcal{L}_{ce}$: 交叉熵分类损失；$\mathcal{L}_{tri}$: 三元组度量损失
- $\lambda_d$: 平衡跨模态对齐贡献的权重

---

## 关键图表

<!-- arXiv HTML 暂未上线（404）；唯一可在线引用的图为作者 GitHub 仓库的 workflow 图（= 论文 Figure 2）。其余图无在线外链，按指令不下载，仅文字转述。 -->

### Figure 2: Workflow of EventGait / 整体工作流（在线图）

![EventGait Workflow](https://raw.githubusercontent.com/QUEAHREN/EventGait/main/assets/eventgait.jpg)

**说明**: EventGait 双流总览。左侧把同一段事件流体素化为两类切片——绿色"短时切片 $\mathbf{E}_d$"（间隔 $\Delta T$，保留高频运动）和蓝色"长时切片 $\mathbf{E}_s$"（窗口 $T$，聚合全局形状）。右侧：$\mathbf{E}_d$ 进 **Dynamic Motion Stream**（内含 MoSE），$\mathbf{E}_s$ 进 **Static Shape Stream**；橙色虚线 $\mathbf{I}_c$ 是仅训练时用于 CroSA 对齐的 RGB 帧（"Only for training"）。两流特征经 **Fusion & Decoder & Classifier** 输出识别结果。这张图直接对应公式1-2 的双尺度表征与公式8 的融合。

### Figure 1: RGB vs Event under Day/Night（无在线外链，文字转述）

**说明**: 对比白天/夜晚下 RGB 相机与事件相机。白天 RGB 能拿到高质量轮廓（笑脸），夜晚 RGB 轮廓质量极差（哭脸）；事件相机在白天和夜晚都能输出高时间分辨率事件、保留全光照下的时空线索（笑脸）。该图是全文动机的视觉化——RGB 对光照敏感，事件天生鲁棒。

### Figure 3: Spiking Dynamics & MoSE Details（无在线外链，文字转述）

**说明**: (a) 直观示意不同膜常数脉冲神经元在复杂条件下的行为：正常光/快动作下产生高频脉冲串，小 $\tau$ "快"神经元(1) 能很好积分、大 $\tau$ "慢"神经元(2) 反而引入时序噪声；低光/慢动作下脉冲稀疏，小 $\tau$ 神经元(3) 难以积分、大 $\tau$ 神经元(4) 靠长时积分更稳健。(b) Dynamic Motion Stream 细节：$N$ 个脉冲专家（各带 $\tau_i$，对应公式5 的 $\mathcal{E}_i$）并行、门控 $\mathcal{G}(\cdot)$ 算系数 $\alpha_i$、Softmax 聚合输出 $\mathbf{F}_d$。该图论证了"单一 $\tau$ 不够、需要专家混合"。

### Figure 4: Static Shape Stream / CroSA Details（无在线外链，文字转述）

**说明**: 静态形状流与 CroSA 训练细节。事件长时切片 $\mathbf{E}_s$ 经 Static Encoder 与对齐函数 $\mathcal{A}(\cdot)$ 得 $\mathbf{z}_{\text{evs}}$；与此并行，RGB 帧 $\mathbf{I}_c$ 灰度化为 $\mathbf{I}_g$ 后送入冻结 VFM 得 $\mathbf{z}_{\text{img}}$；二者用 $\mathcal{L}_{\text{align}}$（公式7）对齐。虚线标注"Only for training / Greyscale conversion"，强调 RGB 教师只在训练期参与。对应公式6-7。

### Figure 5: Cross-view Performance (LidarGait++ vs EventGait)（无在线外链，文字转述）

**说明**: 跨视角性能混淆矩阵热力图，(a) LidarGait++ vs (b) EventGait。作为 2D 事件方法，EventGait 在多个视角上与 SOTA 的 3D LiDAR 方法可比、部分视角更优，说明事件的稠密时空动态比稀疏 LiDAR 点云提供更丰富的运动表征，呈现强视角不变性。

### Table 1: Within-domain Evaluation on SUSTech1K / SUSTech1K-E

| Input | Model | Venue | Params | NM | BG | CL | CR | UB | UN | OC | NT | Overall |
|-------|-------|-------|--------|----|----|----|----|----|----|----|----|---------|
| **SUSTech1K (camera-based)** | | | | | | | | | | | | |
| PCs | HMRNet | MM2024 | - | 92.7 | 92.3 | 79.6 | 90.3 | 83.1 | **95.2** | 86.2 | **90.4** | 90.2 |
| PCs | LidarGait++ | CVPR2025 | 4.4M | **94.2** | **93.9** | 79.7 | 92.4 | 91.5 | **96.6** | 91.9 | **92.2** | **92.7** |
| Sils | GaitBase | CVPR2023 | 8.0M | 81.5 | 77.5 | 49.6 | 75.8 | 75.6 | 76.7 | 81.4 | 25.9 | 76.1 |
| Sils | GaitLLM-10 | CVPR2025 | 23.4M | 88.2 | 86.3 | 59.7 | 83.5 | 88.8 | 86.9 | 90.5 | 28.8 | 84.5 |
| Sils | $\alpha$-Gait-S | NIPS2025 | - | 91.1 | 87.2 | 64.0 | 85.3 | 89.5 | 88.5 | 92.7 | 28.2 | 86.3 |
| Sils+Skeleton | SkeletonGait++ | AAAI2024 | 9.1M | 85.1 | 83.4 | 46.6 | 81.9 | 80.8 | 82.5 | 86.2 | 47.5 | 81.3 |
| Sils+Parsing+Flow | MultiGait++ | AAAI2025 | 12.1M | 92.0 | 89.4 | 50.4 | 87.8 | 89.7 | 89.1 | 93.4 | 45.1 | 87.6 |
| Sils+Depth | DepthGait | MM2025 | - | **93.5** | 88.0 | 56.8 | 87.4 | 90.0 | 88.5 | **95.2** | 38.4 | 87.6 |
| **SUSTech1K-E (event, Ours)** | | | | | | | | | | | | |
| Event | EVGait | CVPR2019 | 45.2M | 55.0 | 70.8 | 76.7 | 67.8 | 58.8 | 44.8 | 58.5 | 78.7 | 65.4 |
| Event | GaitBase$^e$ | CVPR2023 | 8.0M | 66.7 | 62.0 | 40.6 | 63.8 | 65.2 | 53.0 | 61.4 | 59.2 | 63.1 |
| Event | **EventGait** | **Ours** | **4.6M** | 92.5 | **93.3** | **84.4** | **93.3** | 92.8 | 89.7 | **96.9** | **84.8** | **92.8** |

**说明**: EventGait 仅 **4.6M 参数**就在 SUSTech1K-E 上拿到 **92.8% overall**，几乎追平相机端最强的 LidarGait++（92.7%），并在最难的 **NT（夜间）** 上达到 84.8%——而所有轮廓（Sils）方法在 NT 上崩溃到 25-47%。相对事件基线 GaitBase$^e$（63.1%）提升约 +29.7%，相对 EVGait（65.4%）提升约 +27.4%，证明双流设计远优于"长时聚合 event image"。

### Table 2: Within-domain Evaluation on CCGR-Mini / CCGR-Mini-E & CASIA-B / EV-CASIA-B

| Input | Model | Venue | CCGR-Mini Rank-1 | CCGR-Mini mAP | CCGR-Mini mINP | CASIA-B* NM |
|-------|-------|-------|------------------|---------------|----------------|-------------|
| Sils | GaitSet | AAAI2019 | 13.8 | 15.4 | 5.8 | 92.3 |
| Sils | GaitBase | CVPR2023 | 27.0 | 24.9 | 9.7 | 96.5 |
| Sils | DeepGaitV2 | TPAMI2025 | **39.4** | **36.0** | **16.8** | 94.3 |
| Skeleton | GaitGraph2 | CVPRW2022 | 1.2 | 2.4 | 0.6 | 80.3 |
| Skeleton | CAG | TIP2023 | 5.0 | 6.8 | 1.8 | 96.4 |
| Sils+Skeleton | BiFusion | MTA2023 | - | - | - | 93.0 |

| Input | Model | Venue | CCGR-Mini-E Rank-1 | CCGR-Mini-E mAP | CCGR-Mini-E mINP | EV-CASIA-B NM |
|-------|-------|-------|--------------------|-----------------|------------------|---------------|
| Event | EVGait | CVPR2019 | - | - | - | 89.9 |
| Event | GaitBase$^e$ | CVPR2023 | 9.7 | 10.7 | 4.2 | 94.8 |
| Event | **EventGait** | **Ours** | **40.3** | **38.7** | **25.5** | **96.7** |

**说明**: 在更难的 CCGR-Mini-E 上，EventGait（Rank-1 40.3 / mAP 38.7 / mINP 25.5）全面超越事件基线 GaitBase$^e$，甚至超过相机端最强的 DeepGaitV2（Rank-1 39.4）；在 EV-CASIA-B 上 NM 96.7 也优于事件基线。说明事件模态在更复杂协变量下仍提供判别性运动线索。

### Table 3: Cross-domain Evaluation（跨域：训练域→测试域）

| Modality | Method | CCGR-Mini→SUSTech1K (NM/CL/NT/Overall) | CCGR-Mini→Low-light SUSTech1K (NM/CL/NT/Overall) | SUSTech1K→CCGR-Mini (Rank-1/mAP/mINP) |
|----------|--------|----------------------------------------|---------------------------------------------------|----------------------------------------|
| Sils | GaitSet | 23.2 / 13.7 / 13.9 / 23.7 | 9.4 / 5.4 / 2.8 / 8.3 | 3.7 / 4.4 / 1.4 |
| Sils | GaitBase | **49.0** / 29.0 / 21.8 / 48.1 | 22.2 / 9.7 / 3.0 / 18.5 | 3.2 / 3.8 / 1.1 |
| Sils | DeepGaitV2 | 47.8 / **30.3** / 21.9 / **48.2** | 23.3 / 10.2 / 3.2 / 18.7 | 3.2 / 3.8 / 1.1 |
| Event | GaitBase$^e$ | 26.6 / 12.8 / 30.2 / 23.0 | 13.9 / 7.4 / 18.8 / 12.4 | 1.3 / 2.1 / 0.7 |
| Event | **EventGait** | 45.3 / 28.7 / **51.4** / 45.5 | 22.0 / **14.4** / **27.0** / **20.7** | **3.7** / **4.4** / 1.3 |

**说明**: 跨域时整体略逊于轮廓法（45.5 vs 48.2），但在 **NT 夜间场景** EventGait 达 51.4%，比最强轮廓法高出 **+29.5%**；迁移到低光 SUSTech1K 时 overall 20.7% 接近 DeepGaitV2 的 18.7%。证明事件模态对光照变化高度鲁棒，即使跨域亦然。

### Table 4: Cross-illumination Evaluation on SUSTech1K-E

| Modality | Method | Normal NM | Normal CL | Normal NT | Normal Overall | Low NM | Low CL | Low NT | Low Overall |
|----------|--------|-----------|-----------|-----------|----------------|--------|--------|--------|-------------|
| Sils | GaitSet | 69.1 | 37.4 | 23.0 | 65.0 | 43.7 | 19.7 | 3.0 | 31.7 |
| Sils | GaitBase | 81.1 | 48.4 | 25.9 | 76.1 | 54.4 | 26.0 | 3.1 | 41.5 |
| Event | GaitBase$^e$ | 66.7 | 40.6 | 59.2 | 63.1 | 24.8 | 13.5 | 33.5 | 23.6 |
| Event | **EventGait** | **92.5** | **78.1** | **84.4** | **92.8** | **83.8** | **58.3** | **70.5** | **83.2** |

**说明**: 这是论文最有说服力的结果。EventGait 在正常光 92.8% → 低光 83.2%，**仅掉 9.6%**；而轮廓法在低光下崩溃（GaitBase 76.1→41.5，掉 34.6）。低光下 EventGait 比 GaitBase 高出约 **+41.7%**，直接证实事件相机的高动态范围 + 高时间分辨率带来的光照鲁棒性。

### Table 5: Within-domain Evaluation on realistic DVS128-Gait

| | DVS128 EV-Gait | DVS128 GaitBase$^e$ | DVS128 EventGait (Ours) |
|---|---|---|---|
| Rank-1 Accuracy | 81.8 | 74.4 | **87.4** |

**说明**: 在**真实**（非合成）DVS128 事件相机数据集上，EventGait 达 87.4%，超过 EV-Gait（81.8）与 GaitBase$^e$（74.4），验证方法在合成与真实事件基准上都有效，缓解 sim-to-real 担忧。

### Table 6: Ablation — Static & Dynamic Streams（SUSTech1K-E）

| Static Shape Stream | Dynamic Motion Stream | NM | CL | NT | Overall |
|:---:|:---:|----|----|----|---------|
| ✓ | ✗ | 82.6 | 61.6 | 76.9 | 82.0 |
| ✗ | ✓ | 74.5 | 52.0 | 71.7 | 72.4 |
| **✓** | **✓** | **92.5** | **78.1** | **84.8** | **92.8** |

**说明**: 去掉任一流都明显掉点；只用动态流（72.4）明显劣于双流（92.8），因为运动线索本身缺乏结构细节。证明"静+动"互补对完整步态表征不可或缺。

### Table 7: Ablation — CroSA Objective & Weight $\lambda_d$（SUSTech1K-E）

| Idx | Objective (eq.7) | $\lambda_d$ (eq.9) | NM | CL | NT | Overall |
|-----|------------------|--------------------|----|----|----|---------|
| (a) | w/o | - | 87.6 | 71.5 | 78.8 | 87.4 |
| (b) | cosine | 0.2 | 88.8 | 70.9 | 81.9 | 89.1 |
| (c) | $\ell_2$ | 0.05 | 89.4 | 73.7 | 83.3 | 90.5 |
| **(d)** | **$\ell_2$** | **0.2** | **92.5** | **78.1** | **84.8** | **92.8** |
| (e) | $\ell_2$ | 0.5 | 90.1 | 69.5 | 81.0 | 89.7 |

**说明**: 引入 CroSA（a→d）一致提升所有指标，验证其对结构表征的增强。$\ell_2$ 优于 cosine（后者只给方向约束、缺细粒度）。权重 $\lambda_d=0.2$ 最佳：太小引导弱（c），太大引入衣纹等身份无关线索、损害细粒度识别（e）。

### Table 8: Ablation — Number of Experts in MoSE（SUSTech1K-E）

| No. of Experts | Normal NM | Normal CL | Normal NT | Normal Overall | Low NM | Low CL | Low NT | Low Overall |
|:---:|-----------|-----------|-----------|----------------|--------|--------|--------|-------------|
| 1 | 86.6 | 70.3 | 78.3 | 88.4 | 74.5 | 52.8 | 61.7 | 72.5 |
| 2 | 89.2 | 73.9 | 81.5 | 89.8 | 79.5 | 54.4 | 67.1 | 78.6 |
| **3** | **92.5** | 78.1 | 84.8 | **92.8** | **83.8** | **58.3** | 70.5 | **83.2** |
| 4 | 92.4 | **78.9** | **85.1** | 92.7 | 84.2 | 58.2 | **71.1** | 83.4 |

**说明**: 单专家（标准 SNN 单元）最差，说明单一 $\tau$ 适配能力有限。增到 3 个专家在两种光照下都一致提升；4 个专家仅边际增益。权衡精度与效率后默认 **3 个专家**，验证 MoSE 在动态运动建模上的有效性。

---

## 实验

### 数据集 / 基准

| 基准 | 规模 | 特点 | 用途 |
|------|------|------|------|
| **SUSTech1K-E**（本文合成） | 25,239 序列 / 1,050 身份 | 由 SUSTech1K 合成；多模态配对（RGB/轮廓/骨架/点云/事件），含多视角、遮挡、光照、衣着协变量 | 训练/测试（主） |
| **CCGR-Mini-E**（本文合成） | 47,884 序列 / 970 受试者 | 由 CCGR-Mini 合成；每受试者 53 协变量、33 视角，最大且协变量最丰富的事件步态集 | 训练/测试 |
| **EV-CASIA-B** | 124 受试者，每人 66 序列、11 视角 | CASIA-B 经 23 寸屏回放 + DVS128 重录的播放型事件集 | 测试 |
| **DVS128-Gait**（真实） | 4,000 事件流 / 20 受试者 | 真实 DVS128 事件相机白天采集，约 90° 视角 | 训练/测试 |

### 实现细节

- **训练硬件**: 8× RTX 3090 GPU
- **优化**: SGD，初始学习率 0.1，weight decay 0.0005
- **输入处理**: Pad-and-Resize 保持人体比例，缩放到 **64×64**
- **教师模型（CroSA）**: 冻结 [[DINOv2]]（VFM），仅训练时使用
- **MoSE**: 默认 **3 个脉冲专家**，各带不同膜时间常数 $\tau_i$
- **CroSA 权重**: $\lambda_d=0.2$，对齐用 $\ell_2$
- **合成流水线**: 帧插值 → [[v2e]] 工具箱模拟多光照事件流 → 时间窗聚合切片 → 人体检测框时序插值裁剪

### 关键实验结论

- **同域**: SUSTech1K-E 92.8% overall（4.6M 参数追平 LidarGait++ 92.7%）；CCGR-Mini-E Rank-1 40.3 超相机端 DeepGaitV2；真实 DVS128-Gait 87.4%。
- **跨光照（Table 4）**: 正常→低光仅掉 9.6%，低光下大幅超越轮廓法（约 +41.7%），是事件鲁棒性的最强证据。
- **跨域（Table 3）**: 夜间 NT 场景比最强轮廓法高 +29.5%。
- **跨视角（Figure 5）**: 2D 事件方法可比甚至超过 3D LiDAR SOTA。
- **消融**: 双流缺一不可（Table 6）；CroSA 用 $\ell_2$+$\lambda_d=0.2$ 最佳（Table 7）；MoSE 取 3 专家性价比最高（Table 8）。

---

## 批判性思考

### 优点
1. **问题诊断准、方法对症**: 明确指出"长时聚合丢动态 + 事件稀疏难学形状"两大痛点，分别用短/长双尺度 + MoSE + CroSA 对症下药，逻辑闭环清晰。
2. **低光鲁棒性证据扎实**: Table 4 的"正常 92.8→低光 83.2 只掉 9.6%"对比轮廓法的崩溃，是事件步态价值的硬核证据，且在真实 DVS128 上也成立（缓解 sim-to-real 质疑）。
3. **参数高效**: 仅 4.6M 就追平 4.4M 的 LidarGait++ 与远更大的轮廓/多模态模型，且事件相机比 LiDAR 便宜约 50 倍，部署经济性强。
4. **配套发布两个大规模事件步态基准 + 合成流水线 + 代码**，对社区有基础设施价值。

### 局限性
1. **主要靠合成数据**: 两个主基准（SUSTech1K-E / CCGR-Mini-E）均由 v2e 从 RGB 合成，真实事件相机的噪声、热像素、延迟分布与合成存在差距；真实评测只在 20 人的小规模 DVS128 上做（作者也在 future work 承认要采真实大规模数据）。
2. **CroSA 依赖配对 RGB**: 训练阶段需同步 RGB 帧做教师监督，这在纯事件采集管线里不一定满足；虽然推理不需 RGB，但训练数据的获取门槛被抬高。
3. **MoSE 的"专家分工"偏定性**: 论文用 Figure 3(a) 直觉性解释快/慢神经元，但缺少对学到的 $\tau_i$ 分布、门控系数 $\alpha_i$ 是否真的按光照/速度分工的定量探针分析。
4. **跨域整体性能仍逊轮廓法**: Table 3 中常规光跨域 overall（45.5）低于 DeepGaitV2（48.2），优势主要集中在夜间，普适增益尚未全面建立。

### 潜在改进方向
1. 采集大规模**真实**事件步态数据，量化 sim-to-real gap，并探索无需配对 RGB 的自监督结构对齐（替代 CroSA 的 RGB 教师）。
2. 对 MoSE 做可解释性分析（$\tau_i$ 学习轨迹、$\alpha_i$ 与场景光照/速度的相关性），或让 $\tau_i$/专家数自适应可学。
3. 按 future work 思路做多模态融合（事件 + RGB/LiDAR），在保留低光鲁棒的同时补足常规光下的细节判别。

### 可复现性评估
- [x] 代码开源（https://github.com/QUEAHREN/EventGait）
- [ ] 预训练模型（仓库未明确提供权重）
- [x] 训练细节较完整（8×RTX3090、SGD lr0.1、wd5e-4、64×64、3 专家、$\lambda_d=0.2$；更多在 Appendix）
- [x] 数据集可获取（声明 release SUSTech1K-E / CCGR-Mini-E；底层 SUSTech1K/CCGR-Mini/CASIA-B/DVS128-Gait 公开）

---

## 速查卡片

> [!summary] EventGait: Robust Gait Recognition with Event Streams
> - **核心**: 事件相机 + 双流（动态运动 + 静态形状）做光照鲁棒的步态识别。
> - **方法**: 同一事件流体素化为短时 $\mathbf{E}_d$/长时 $\mathbf{E}_s$；动态流用脉冲专家混合 [[Mixture of Spiking Experts|MoSE]]（多膜常数 $\tau$ + 门控）建模高频运动；静态流用 [[Cross-modal Structure Alignment|CroSA]]（冻结 DINOv2 蒸馏，仅训练）补稠密形状；融合后 $\mathcal{L}_{ce}+\mathcal{L}_{tri}+\lambda_d\mathcal{L}_{align}$ 训练。
> - **结果**: 仅 4.6M 参数，SUSTech1K-E 92.8%（追平 LidarGait++），低光仅掉 9.6%、远超轮廓法；真实 DVS128-Gait 87.4%。配套发布 SUSTech1K-E / CCGR-Mini-E 两大基准。
> - **代码**: https://github.com/QUEAHREN/EventGait

---

*笔记创建时间: 2026-06-29*
