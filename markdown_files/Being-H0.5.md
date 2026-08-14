---
title: "Being-H0.5: Scaling Human-Centric Robot Learning for Cross-Embodiment Generalization"
method_name: "Being-H0.5"
authors: [Hao Luo, Ye Wang, Wanpeng Zhang, Sipeng Zheng, Ziheng Xi, Chaoyi Xu, Haiweng Xu, Haoqi Yuan, Chi Zhang, Yiqing Wang, Yicheng Feng, Zongqing Lu]
year: 2026
venue: arXiv
tags: [vision-language-action, egocentric-video, human-centric-learning, cross-embodiment, unified-action-space, flow-matching, dexterous-manipulation, asynchronous-control]
image_source: online
---

# Being-H0.5：统一动作槽能否让一个 VLA 真正跨本体？

> 本笔记基于 [arXiv:2601.12993v1](https://arxiv.org/abs/2601.12993)、[HTML 全文](https://arxiv.org/html/2601.12993)、[官方项目页](https://research.beingbeyond.com/being-h05)、[Apache-2.0 代码库](https://github.com/BeingBeyond/Being-H/tree/main/Being-H05)及 [Hugging Face 模型集合](https://huggingface.co/collections/BeingBeyond/being-h05)精读核验。论文上传于 2026-01-19；截至 **2026-08-14** 未见会议接收信息，故 venue 记为 arXiv。

## 阅读结论先行

Being-H0.5 把跨本体 VLA 的矛盾拆成四层：用 200 维、带物理语义的固定槽统一人手和 30 种机器人状态/动作；以 InternVL3.5 的理解专家和 rectified-flow 动作专家共同注意视觉—语言上下文；用 Mixture-of-Flow（MoF）把共享运动基础层与稀疏路由的本体/任务专家分开；在真实部署时再用 slot-wise ESA、Manifold-Preserving Gating（MPG）和 Universal Async Chunking（UAC）处理本体冲突、感知偏移与推理延迟。

论文最扎实的结果不是“零样本新机器人”，而是 **同一个 generalist checkpoint** 在五种已见本体、十类任务上接近 specialist，以及在 LIBERO+RoboCasa 联合训练后仍分别达到 97.6% 和 53.3%。真实任务每项只有 30–60 分钟示范，默认每方法 20 次 blind black-box trials；这是比许多机器人论文更好的 operator-bias 控制。不过类别结果只画在柱状图中，没有置信区间、随机种子或逐任务数值，20 次下 5 个百分点就是一次 rollout。

“embodiment-level zero-shot”必须降格解读。模型在 Adam-U 上执行 Flip→Scan、Drawer、Stacking 等只在其他机器人示范过的任务；但 Adam-U 本体本身已用 Arrange Flowers、Hand-over、Clear Table 等任务参与联合 post-training。因此证据是 **已见机器人上的未见 task–embodiment pairing / compositional transfer**，不是未见 morphology，也没有报告成功率、trial 数或统计不确定性。论文贡献列表中“unseen robot morphologies”的表述超出了实验。

UniHand-2.0 的“35,000 小时”也应拆开：16K 小时人类 ego（134M samples）、约 13,817.4 小时机器人（原始约 1.5B frames，去重后只下采样到 30%）、以及 5K **equivalent hours** 的 VLM 数据；三者合计 120B tokens、400M+ samples。5K VLM 不是 5K 小时真实交互录像。机器人部分称 45.7B visual tokens 和 30M text tokens 为“约 1000:1”，实际相除约 1,523:1。26% simulation cap 是训练 mixture 的统计，并未证明该阈值最优。

Unified Action Space 比独立 robot heads 更有结构：相同功能总在相同 slot；人手 wrist 对齐 EEF，finger articulation 放进 fine-manipulation slots；Cartesian action 用 world-frame relative delta、rotation 用 axis-angle、joint 用 absolute radians，并保留原始物理尺度。它仍不是自动 kinematic equivalence：不同数据的“world frame”、工作空间、control rate、零填充语义和可达域并不天然一致；没有显式 validity mask 时，unused zero 与真实零动作也可能混淆。开源实现甚至仍按 task/embodiment 选择 normalization statistics，与论文“no statistical normalization”叙述需要更精确的路径级说明。

Hybrid human motion objective 同时学习 continuous flow 与 Being-H0 tokenizer 的 masked discrete tokens，是合理的“精度 + 抽象先验”组合；但其唯一直接消融存在明显内部矛盾：Table 8 标注 MWDS 越高越好，Hybrid 是 0.33/0.20，去掉 $\mathcal L_{\mathrm{MASK}}$ 却是 0.35/0.28，数字显示去掉后更好，正文反而称明显下降且 Wild 改善更大。这可能是行标签、箭头或数字排版错误；在作者勘误前不能据此宣称 masked objective 已被因果验证。

MPG 与 UAC 更像实用 deployment engineering。MPG 以 observation/action embedding 的 Sliced-Wasserstein discrepancy 做 reliability gate；推理时 action anchor 来自上一轮 denoising iterate，因而不是独立 OOD detector，更未直接观测 contact 或 feasibility。UAC 则把实际 latency 换算成本体特定 committed prefix，只对 postfix 算 flow loss，部署中 hard-lock prefix 并用双线程 ring buffer 拼接。论文却只同时关闭 MPG+UAC 做组合消融，不能分别归因“长时序下降来自 UAC”或“双臂抖动来自 MPG”。

公开资产比论文首发时更实用但仍不完整。Apache-2.0 repo 有训练、推理、数据配置、LIBERO/RoboCasa 配置和 SO-101 教程；HF 有 base、两种 specialist 和一个 simulation generalist 权重。可是 README 的 TODO 仍列完整 pretraining scripts/docs、所有 benchmark post-training/eval、out-of-box real-robot checkpoints；UniHand 只发布 preview subset。论文承诺的 real-world infrastructure 和 1,000 GPU-hour recipe 尚不能完整复现。

### 一句话总结

Being-H0.5 的真正贡献是把“共享动作槽 + 共享/稀疏 flow experts + latency-aware chunking”做成一个覆盖多本体的 VLA system，并用联合 benchmark 与五机器人实验说明 generalist 不必严重输给 specialist；但它没有证明对全新机器人 morphology 的零样本控制，关键 masked-motion 消融自相矛盾，MPG/UAC 未独立消融，且完整 35K-hour recipe 与真实部署代码尚未交付。

![Being-H0.5 架构：统一状态动作空间、MoT 与 Mixture-of-Flow](https://research.beingbeyond.com/being-h05/images/beingh05_arch_nar.webp)

*图 1。官方架构图。理解专家与动作生成专家共享 attention；动作专家内部再分共享 foundation layers 与稀疏 specialist layers。*

## 0. 资源与可复现性

| 资产 | 当前状态 | 可复现判断 |
| --- | --- | --- |
| 论文/项目页 | arXiv v1、官方叙述与视频/图 | 方法和主结果可审计 |
| 代码 | Apache-2.0；统一仓库 `Being-H05/` | 可训练/推理公开配置 |
| 权重 | base 2B、LIBERO、RoboCasa、两 benchmark generalist | 模型卡称 2B，HF 文件解析为约 3B、F32 11.2GB，口径需注意 |
| 数据 | UniHand Preview、Adam-U post-train collection | 不是完整 35K-hour mixture |
| 真实机器人 | 论文五平台；公开 README 目前主要给 simulation configs 和 SO-101 教程 | 不能一键复现论文五机器人 |
| 完整 recipe | README TODO 尚有 pretrain、全 benchmark post-train/eval、real checkpoints | 论文“full reproducibility”尚未兑现 |

## 1. 数据：把“小时、sample、token”分开

| 部分 | 论文规模 | 关键处理/限制 |
| --- | ---: | --- |
| Human ego | 16K h；134M samples；25.6B tokens | HaWoR 手运动估计、DBA filtering、去 jitter/断裂、Gemini 过滤纯 locomotion、左右镜像；pseudo-label 误差仍会传播 |
| Robot | 13,817.4 h（约称 14K）；1.5B raw frames；45.7B tokens | 30 embodiments；去重、frames 下采样到 30%；simulation 在 mixture 中 cap 26% |
| VLM | 5K equivalent h；50.2B tokens | LLaVA/FineVision/VQA、RefCOCO/RoboPoint 等 spatial grounding、ShareRobot/EO1.5M planning；不是 physical trajectories |

Robot table 从 Franka 2,196.4h、AgiBot-G1 2,391.7h、Kuavo 1,198.2h、Split ALOHA 1,099.1h，到只有 10–40h 的 Cobotta、PR2、AlphaBot 等，极度长尾。所谓 30 embodiments 只计每种超过 10h；统一 slot 不会自动补齐数据量差异。最值得做但论文没有的 control，是按目标本体相关性、时长和真实/仿真比例匹配后比较 random mixture、all-data mixture 与 retrieval-selected mixture。

UniCraftor 补充了 200+ 小时、43 个桌面任务：头戴 D435 原生 RGB-D；五个 AprilTag+PnP 得相机 extrinsics；脚踏板标记 contact/release 关键事件；Grounded-SAM2 与 DiffuEraser 跟踪并移除 tag；HaWoR 手姿态经多视角深度校正；Qwen2.5-VL 生成描述再人工核验。优点是把 depth、camera pose、event timing 做成同步信号；局限是 AprilTag 可见性限制、主动红外在反光/户外的适用性，以及 pedal event 仍不是 force/tactile contact ground truth。

## 2. 从本体专属信号到 200 维公共语言

设本体 $e$ 的原始状态/动作分别为 $\mathbf s^{(e)},\mathbf a^{(e)}$，稀疏映射为：

$$
\mathbf s=\Phi_e(\mathbf s^{(e)}),\qquad
\mathbf a=\Phi_e(\mathbf a^{(e)}),
$$

未使用 slot 填零。全序列包含 vision、text、state、action：

$$
\mathcal S=[\mathbf x_1,\ldots,\mathbf x_K],\qquad
\mathcal M=\{\text{vision,text,state,action}\}.
$$

这种表示的收益来自 **semantic overlap**：Franka/Kuka 的 EEF、关节与 gripper 可以共享维度；人手 wrist/fingers 给复杂 end-effector 提供预训练信号；mobile base 另占 velocity/heading slots。风险则在映射工程：axis-angle 的 $\pi$ 附近不连续、绝对 joint radians 受零位/关节顺序影响、world-frame delta 依赖可靠 extrinsics，而 raw scale 不等于跨平台分布相同。

~~~mermaid
flowchart LR
    H["Human wrist + fingers"] --> U["200-D unified slots\nEEF / joints / gripper / fingers / base"]
    R1["Single-arm robots"] --> U
    R2["Bimanual/humanoid"] --> U
    R3["Mobile manipulators"] --> U
    U --> S["Serialized vision-text-state-action"]
    S --> V["InternVL3.5 understanding expert"]
    S --> F["Rectified-flow action expert"]
    V <-->|"shared attention"| F
    F --> M["MoF: shared foundation + routed specialists"]
    M --> E["ESA active-slot adapters"]
    E --> D["MPG refinement + UAC async deployment"]
~~~

## 3. Architecture 与训练目标

MoF 下层共享 transferable motor primitives，上层用 task/embodiment-conditioned Top-K experts。稀疏激活可增加总容量而不线性增加 active compute，但论文没有报告专家数、Top-K、load-balancing loss、routing collapse 指标或 MoF 对 dense expert 的等 active-parameter 对照，所以“避免负迁移”更多由整体结果间接支持。

文本以 next-token cross entropy 训练；连续 action 用 rectified flow。对噪声 $\mathbf x_0\sim\mathcal N(0,I)$ 和目标 action $\mathbf a_i$：

$$
\mathbf x_t=(1-t)\mathbf x_0+t\mathbf a_i,qquad
\mathcal L_{\mathrm{FM}}=
\sum_{i\in\Omega_{\mathrm{FM}}}
\left\|v_\theta(\mathbf x_t,t,c)-(\mathbf a_i-\mathbf x_0)\right\|_2^2.
$$

Human motion 还经 Being-H0 tokenizer 得离散码 $\mathbf z$，随机 mask 后重建：

$$
\mathcal L_{\mathrm{MASK}}=-\sum_{i\in\Omega_{\mathrm{MASK}}}
\log p_\theta(z_i\mid c),qquad
\mathcal L_{\mathrm{act}}=\lambda_1\mathcal L_{\mathrm{FM}}+lambda_2\mathcal L_{\mathrm{MASK}}.
$$

continuous 与 discrete target 共享 context，却用 block attention 互相不可见，以防复制。这是不错的 representation design；然而 Table 8 的反向数字使它尚无可信的 direct ablation。

## 4. Post-training 与实时控制

### 4.1 ESA：共享槽上的局部专门化

本体 $e$ 只激活 $\mathcal I_e$ 中的 slot adapters：

$$
\mathbf W_{\mathrm{ESA}}^{(e)}={\mathbf W_{\mathrm{ESA}}[k]:k\in\mathcal I_e\},qquad
\Delta\mathbf W_{\mathrm{ESA}}[k]=0\quad(k\notin\mathcal I_e).
$$

它比每机器人独立 head 更容易共享重叠部件，但“不同手装同一臂”的共享是否正迁移，论文没有 component-swap ablation。

### 4.2 MPG：自参照的 reliability gate

MPG 计算 observation embedding 与 action anchor 的 sliced-Wasserstein discrepancy：

$$
D\approx\frac1M\sum_{m=1}^M
\left\|\operatorname{sort}(\theta_m^\top\hat H)-
\operatorname{sort}(\theta_m^\top\hat Z)\right\|_2^2,qquad
g=\exp(-D/\tau).
$$

停止 $g$ 的梯度，并只 gate feature-conditioned term：

$$
\widetilde H=H+\lambda g\mathbf W_{\mathrm{MPG}}\mathcal E_{\mathrm{obs}}(H)
+\lambda\mathbf b_{\mathrm{MPG}}.
$$

训练 anchor 是真 action，推理 anchor 是上一 denoising iterate，形成 train/inference gap。它可平滑 iterative update，但不能据此声称识别真实 OOD、collision 或 contact failure。

### 4.3 UAC：让 chunk 知道已经来不及修改的动作

本体 $e$ 的 control period 为 $\Delta t^{(e)}$、推理预算 $L^{(e)}$，训练抽样 delay $d$，把前 $d$ 步当 committed prefix，只监督 postfix：

$$
d\sim\pi^{(e)}(d),\qquad
\mathcal L_{\mathrm{UAC}}=\sum_{i\ge d}\|\hat v_i-v_i^*\|_2^2.
$$

部署必须满足：

$$
d\ge\left\lceil\frac{t_{\mathrm{inference}}}{t_{\mathrm{control}}^{(e)}}\right\rceil+\epsilon_{\mathrm{safety}}.
$$

每次 denoising hard-lock buffer prefix，完成后只写 postfix。控制/推理双线程共享至少 $2\times$ chunk length 的 ring buffer；underflow 时 hold-last-action 或用安全 fallback。它清楚解决了 10Hz tabletop 到 50Hz humanoid 的 timing mismatch，但安全 fallback、worst-case network jitter 和超时率没有量化。

![MPG 与 UAC：特征可靠性门控和按本体延迟锁定 action prefix](https://research.beingbeyond.com/being-h05/images/mpg-uac.webp)

*图 2。官方机制图；若该项目页改名资源，可从项目页“MPG and UAC overview”入口访问。*

## 5. 实验读数与证据强度

### 5.1 五种真实平台

Adam-U（31 DoF）、G1+LinkerBot O6（26）、FR3+Inspire（13）、D1（14）和 SO-101（6）覆盖双臂 humanoid、单臂 dexterous 与 gripper。任务表名义上十类，但 Clear Table 在 Adam-U 与 SO-101 各有一项，共 11 个 task–embodiment rows。每项 30–60 分钟 data；scene layouts 与 policy 随机，operator 不知 policy identity，binary criteria 预定义，默认 20 trials。

Figure 9 支持 specialist 通常最高、generalist 接近，且两者优于同 data 的 $\pi_{0.5}$；scratch generalist 明显更差，说明 UniHand pretraining 对 heterogeneous joint post-training 有价值。不过图只聚合 Spatial/Long-horizon/Bimanual/Generalization 四类，掩盖了逐机器人和逐任务 variance；没有 paired layout-level outcomes 或 bootstrap CI。

Zero-shot 部分只报告 Adam-U 在其他机器人任务上 non-zero，且作者承认运动精度不稳定。没有数字意味着无法比较 random success、scripted prior 或语言触发的“正确起步但失败”。最准确的结论是出现了定性 compositional signal。

### 5.2 Simulation

| 模式 | LIBERO Spatial/Object/Goal/Long | 平均 | RoboCasa Pick/Doors/Others | 平均 |
| --- | --- | ---: | --- | ---: |
| Generalist | 97.0 / 98.2 / 99.0 / 96.2 | 97.6 | 40.0 / 73.0 / 52.0 | 53.3 |
| Specialist | 99.2 / 99.6 / 99.4 / 97.4 | 98.9 | 36.0 / 71.7 / 57.6 | 53.9 |

LIBERO 每 task 50 episodes，specialist 45K steps、4×A800、effective batch 128、chunk 8、224² multi-view RGB；RoboCasa Human-50 为 24 tasks、每 task 50 demos、五个 held-out scenes、每 task 50 trials。Generalist 用 LIBERO+RoboCasa 并训练约双倍 steps，使每 benchmark update 预算接近，算合理但不是同总 compute 比较。

5-shot ablation 最值得保留的观察是：当理解 backbone 大量冻结时 human-centric initialization 增益最大（single-task +25.8pp；multi-task最高 +11.7pp），full fine-tuning 时只有 +4.6pp/+1.0pp，且个别 suite 为负。冻结超过约 14 层 action expert 后性能崩到 20% 以下，说明 transferable prior 主要保存在 semantic backbone；action expert 仍需足够 plasticity，不能把预训练直接等同为可执行 control knowledge。

### 5.3 缺失的关键对照

- 等 token/compute 的 human-only、robot-only、VLM-only 与不同 mixture ratio。
- Unified slots 对 independent heads、masked slots、normalized/raw units 的正交消融。
- Dense flow expert 对 MoF 的等 active-parameter/compute 对照和 routing statistics。
- ESA、MPG、UAC 各自以及两两组合；目前只关 MPG+UAC。
- 真正 held-out embodiment：该机器人在 pretraining 与 post-training 均不出现，只给少量 calibration/config。
- OOD corruption 分级、latency sweep、buffer underflow、collision/unsafe motion 与 recovery rate。

## 6. 与 Being-H0 的关系

Being-H0 用 MANO discrete motion language 预训练，再为一个目标 robot 学 continuous action head；跨域主要是 representation transfer。H0.5 保留 discrete human motion auxiliary，却把 continuous human/robot actions都投进同一个 high-dimensional slot space，并在一个 flow policy 中联合建模。换言之，H0.5 从“共享 backbone、动作头分开”走向“共享 action interface、容量按 slot/route 部分共享”。这是真正的方法升级，但仍依赖人工定义 $\Phi_e$、本体 post-training 和 runtime calibration。

## 7. 最终判断

| 主张 | 证据 | 判断 |
| --- | --- | --- |
| 单 checkpoint 跨五种已见本体 | blind real-robot evaluation、generalist 接近 specialist | **较强支持**，但缺逐任务数值/CI |
| human-centric pretraining 帮助少样本与 generalist | scratch、5-shot freeze ablations | **支持**，尚不能区分 human/robot/VLM 数据贡献 |
| MoF 减轻本体干扰 | 整体 system 与定性动机 | **间接支持**，缺 dense control |
| MPG/UAC 各自有效 | 只联合关闭两者 | **不能分别归因** |
| masked motion tokens 提升 | Table 8 数字与正文方向相反 | **当前不可采信，需勘误** |
| 零样本 unseen morphology | Adam-U 已在 post-training 中出现 | **不支持**；只支持 unseen pairing 的定性信号 |
| full reproducibility | code/weights 有，完整 recipe/data/real infra 缺 | **部分可复现** |

Being-H0.5 值得作为 cross-embodiment VLA 的 system blueprint：它明确把 representation、capacity、adaptation 和 real-time scheduling 一起设计。下一步真正有说服力的验证，不是再提高 LIBERO 末位小数，而是冻结一个全新 morphology，公开其 slot mapping 与 latency profile，在不加入该机器人轨迹、少量轨迹和 full-data 三档下报告精确、接触、安全、恢复与置信区间。
