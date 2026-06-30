---
title: "LDA-1B: Scaling Latent Dynamics Action Model via Universal Embodied Data Ingestion"
method_name: "LDA-1B"
authors: ["Jiangran Lyu"]
year: 2026
venue: "RSS"
tags: ["contact-reasoning", "robust-control", "reinforcement-learning", "imitation-learning", "contact-rich-manipulation", "robot-generalization", "dexterous-manipulation"]
image_source: "online"
arxiv_html: "https://arxiv.org/html/2602.12215v2"
---
# LDA-1B
## 一句话总结

> LDA-1B: Scaling Latent Dynamics Action Model via Universal Embodied Data Ingestion 主要落在 [[action-model]]、[[behavior-cloning]]、[[benchmark-dataset]]、[[接触推理]] 这一类问题上;从 System0 视角看,它的价值在于把身体状态、接触变化、局部感知或动力学约束接到机器人低层闭环里,让动作生成更稳定、安全、实时。
## 核心贡献

1. 围绕 **LDA-1B: Scaling Latent Dynamics Action Model via Universal Embodied Data Ingestion** 建立了一个与 action-model、behavior-cloning、benchmark-dataset、接触推理、接触丰富操作、灵巧操作 相关的机器人问题设定,重点不是高层语义规划,而是身体级闭环中的状态变化、接触变化或控制约束。
2. 方法层面的关键价值在于把感知、动力学、约束优化、策略学习或硬件反馈组织成可执行的低层动作生成机制,适合放进 System0 的快反应模块中理解。
3. 对系统集成而言,这类工作补上了 System1 技能选择与电机级控制之间的一层:它负责把上层目标变成动态可行、可恢复、可安全过滤的动作。
4. 若论文包含真实机器人、接触任务或仿真到真实迁移,核心应看它是否报告了闭环稳定性、扰动恢复、硬件延迟、失败模式和跨场景泛化。
## 问题背景

这篇论文可以放在 System0 的“具身快系统”背景下阅读:机器人需要在很短时间内根据本体状态、接触、力/触觉、局部视觉或动力学预测做出动作,而不是等待慢速规划器重新推理整个任务。action-model、behavior-cloning、benchmark-dataset、接触推理、接触丰富操作、灵巧操作、diffusion-policy 相关问题的难点通常不在“知道要做什么”,而在动作执行过程中状态会快速偏离、接触会突然改变、约束会实时收紧,低层系统必须及时修正。

### System0 关联

精读时优先看三件事:第一,论文使用哪些快速反馈量,例如关节状态、接触状态、力觉、触觉、深度或局部视觉;第二,反馈如何进入 MPC、WBC、策略网络、安全过滤器或恢复控制器;第三,输出是否直接影响机器人下一步动作,而不是只服务离线分析或高层任务规划。
## 方法详解

### System0 视角解析

从 System0 角度看,这篇论文的方法部分应当围绕 action-model、behavior-cloning、benchmark-dataset、接触推理、接触丰富操作、灵巧操作、diffusion-policy、dynamics-modeling 来读。核心问题是:系统如何从快速变化的机器人状态中构造控制输入,如何处理接触、碰撞、约束、模型误差或执行延迟,以及如何把这些信息变成下一时刻的动作。

如果方法里有优化问题,重点看目标函数、约束和求解频率;如果方法里有学习策略,重点看观测空间、动作空间、训练扰动和部署时的闭环频率;如果方法里有硬件或传感器,重点看传感链路是否真的进入控制回路。这样读可以把论文和 System2 语言规划、System1 技能编排区分开来。

### 关键公式与机制

#### 公式 1: [[动力学或策略机制]]

$$p(\boldsymbol{a}_{t+1:t+k}\mid\boldsymbol{o}_{t})$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 2: [[动力学或策略机制]]

$$p(\boldsymbol{o}_{t+1:t+k}\mid\boldsymbol{o}_{t},\boldsymbol{a}_{t+1:t+k})$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 3: [[动力学或策略机制]]

$$p(\boldsymbol{a}_{t+1:t+k}\mid\boldsymbol{o}_{t:t+k})$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 4: [[动力学或策略机制]]

$$p(\boldsymbol{o}_{t+1:t+k}\mid\boldsymbol{o}_{t})$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 5: [[动力学或策略机制]]

$$(\epsilon_{a}^{\theta},\epsilon_{o}^{\theta})=s_{\theta}\!\left(o,\,a_{t_{a}},\,o^{\prime}_{t_{o}},\,t_{a},\,t_{o^{\prime}}\right),$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 6: [[优化目标/约束]]

$$=\mathbb{E}_{\begin{subarray}{c}(\boldsymbol{o}_{t:t+k},\boldsymbol{a}_{t+1:t+k},\ell)\sim\mathcal{D}\\ \tau_{a}\sim\mathcal{U}(0,T_{\tau})\\ \epsilon_{a}\sim\mathcal{N}(\boldsymbol{0},\boldsymbol{I})\end{subarray}}\left\lVert v_{a}^{\theta}-(\epsilon_{a}-\boldsymbol{a}_{t+1:t+k})\right \rVert_{2}^{2},$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 7: [[优化目标/约束]]

$$=\mathbb{E}_{\begin{subarray}{c}(\boldsymbol{o}_{t:t+k},\boldsymbol{a}_{t+1:t+k},\ell)\sim\mathcal{D}\\ \tau_{o}\sim\mathcal{U}(0,T_{\tau})\\ \epsilon_{o}\sim\mathcal{N}(\boldsymbol{0},\boldsymbol{I})\end{subarray}}\left\lVert v_{o}^{\theta}-(\epsilon_{o}-\boldsymbol{o}_{t+1:t+k})\right \rVert_{2}^{2},$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 8: [[动力学或策略机制]]

$$=l_{\mathrm{action}}^{\theta}+l_{\mathrm{obs}}^{\theta}.$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 9: [[动力学或策略机制]]

$$l_{\mathrm{action}}^{\theta}$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 10: [[动力学或策略机制]]

$$l_{\mathrm{obs}}^{\theta}$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。
## 关键图表

### Figure 1: Architecture of LDA. LDA jointly denoises action chunks and future visual latents unde

![Figure 1](https://arxiv.org/html/2602.12215v2/x1.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Architecture of LDA. LDA jointly denoises action chunks and future visual latents unde”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 2: Real-world robot platforms used in our physical experiments. From left to right: (1

![Figure 2](https://arxiv.org/html/2602.12215v2/figures/real_setup.jpg)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Real-world robot platforms used in our physical experiments. From left to right: (1”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 3: Task descriptions for the Galbot G1 robot equipped with a standard two-finger paralle

![Figure 3](https://arxiv.org/html/2602.12215v2/figures/task_overview.jpg)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Task descriptions for the Galbot G1 robot equipped with a standard two-finger paralle”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### 论文表格

#### Table 1: TABLE I: Comparison of Representative Robot Foundation Models. This table compares the proposed LDA with recent robot fo

| Model | Data Src. | #Data | Action Quality | Train. | Param. |
| --- | --- | --- | --- | --- | --- |
| $\pi_{0.5} [26 ]$ | Tele. | 10k+ | High | BC | 3B |
| RDT [35 ] | Tele. | <10k | High | BC | 1B |
| GraspVLA [16 ] | Sim. | 20k+ | High | BC | 2B |
| InternVLA-M1 [13 ] | Sim. | <10k | High | BC | 3B |
| Being-H0 [38 ] | Hum. | <10k | Mixed | Aln. + BC | 14B |
| InternVLA-A1 [11 ] | Het. | 10k+ | High | VF + BC | 3B |
| GR00T-N1.6 [5 ] | Het. | <10k | Mixed | LA + BC | 1B |
| UniVLA [9 ] | Het. | <10k | Mixed | LA + BC | 7B |
| LDA-1B | Het. | 30k+ | Mixed | UWM [65 ] | 1B |

**说明**: TABLE I: Comparison of Representative Robot Foundation Models. This table compares the proposed LDA with recent robot foundation models in terms of data source, data quantity, action quality, training paradigm, and the number of trainable model parameters (excluding frozen components). Data source abbreviations are as follows: Tele.=teleoperation, Sim.=simulation, Hum.=human demonstration, and Het.=heterogeneous data. Training paradigm abbreviations include: BC=behavior cloning, VF=visual foresight, Aln.=alignment, LA=latent action modeling, and UWM=unified world model. Only embodied interaction data are considered, excluding internet-scale VQA data.

#### Table 2: TABLE II: Results on RoboCasa-GR1 [42 ] and impact of state representation (VAE vs. DINO [50 ]), model size, and the

| Model | Vis. Rep. | MMDiT | VLM | Success Rate ↑ \uparrow |
| --- | --- | --- | --- | --- |
| GR00T-N1.6 [5 ] | - | - | Cosmos [2, 43 ] | 47.6 |
| StarVLA [14, 57 ] | - | - | Qwen3-VL [55 ] | 47.8 |
| GR00T-EI10k | - | - | Qwen3-VL | 51.3 |
| UWM-0.1B [65 ] | VAE | ✗ | - | 14.2 |
| UWM-1B | VAE | ✗ | Qwen3-VL | 19.3 |
| UWM(MM-DiT) | VAE | ✓ | Qwen3-VL | 20.0 |
| LDA(DiT) | DINO | ✗ | Qwen3-VL | 48.9 |
| LDA-0.5B | DINO | ✓ | Qwen3-VL | 50.7 |
| LDA-1B | DINO | ✓ | Qwen3-VL | 55.4 |

**说明**: TABLE II: Results on RoboCasa-GR1 [42 ] and impact of state representation (VAE vs. DINO [50 ]), model size, and the MM-DiT architecture on task success rates.

#### Table 3: TABLE III: Robust Generalization visual and spatial perturbations. LDA-1B achieves 60.0% success on unseen objects

| Method | Pick & Place | | |
| --- | --- | --- | --- |
| Object | Background | OOD Pos. | |
| $\pi_{0.5}$ | 26.7 | 20.0 | 6.7 |
| GR00T | 40.0 | 40.0 | 20.0 |
| Ours | 60.0 | 60.0 | 40.0 |

**说明**: TABLE III: Robust Generalization visual and spatial perturbations. LDA-1B achieves 60.0% success on unseen objects and backgrounds, and 40.0% OOD positions, demonstrating effective focus on task-critical affordances visual noise through latent dynamics pretraining.

#### Table 4: TABLE IV: Data-efficient mixed-quality fine-tuning. LDA-1B improves success rates by +10% on both tasks when incorporati

| Method | Place the pen into the box | Bimanually remove the lid | | |
| --- | --- | --- | --- | --- |
| 63 High | 63 High + 37 Low | 66 High | 66 High + 34 Low | |
| $\pi_{0.5}$ | 60 | 40 (20 ↓ \downarrow) | 50 | 40 (10 ↓ \downarrow) |
| Ours | 70 | 80 (10 ↑ \uparrow) | 50 | 60 (10 ↑ \uparrow) |

**说明**: 该表格来自论文中的公式、奖励项或实验设置；为避免 PDF 抽取噪声，网页中仅保留可渲染的核心符号和数值。

#### Table 5: TABLE V: Model and Training configuration hyperparameters

| Parameter | Value |
| --- | --- |
| Model | |
| VLM | Qwen3-VL [55 ] |
| Observation Encoder | DINOv3-ViT-s [50 ] |
| Hidden Size | 1536 |
| Layers | 16 |
| Attention Heads | 32 |
| Image Shape | (224, 224, 3) |
| Latent Image Shape | (14, 14, 384) |
| Action Chunk | 16 |
| Training | |
| Batch Size | 32 * 48 (pretraining) |
| | 12 * 8 (fine-tuning) |
| Learning Rate | 1  e - 4 1e^{-4} |
| Optimizer | AdamW |
| Weight Decay | 1  e - 5 1e^{-5} |
| Betas | [0.9, 0.95] |
| Epsilon | 1  e - 8 1e^{-8} |
| LR Schedule | cosine w/ min lr |
| Min LR | 5  e - 7 5e^{-7} |

**说明**: TABLE V: Model and Training configuration hyperparameters

#### Table 6: TABLE VI: Results on RoboCasa-GR1 [42 ] benchmark. UWM: UWM [65 ] with 140M parameters. UWM-XL: UWM with 1B parameters

| model | UWM | UWM-XL | UWM+MM-DiT | GR00T | StarVLA | GR00T-EI10k | LDA (DiT) | LDA |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PnP Bottle To Cabinet Close | 27 | 41 | 49 | 51.5 | 46 | 69 | 65 | 76 |
| PnP Can To Drawer Close | 22 | 53 | 55 | 13 | 80 | 61 | 59 | 71 |
| PnP Cup To Drawer Close | 18 | 12 | 43 | 8.5 | 54 | 47 | 40 | 41 |
| PnP Milk To Microwave Close | 22 | 25 | 33 | 14 | 48 | 75 | 47 | 52 |
| PnP Potato To Microwave Close | 16 | 29 | 18 | 41.5 | 28 | 41 | 39 | 41 |
| PnP Wine To Cabinet Close | 31 | 24 | 25 | 16.5 | 46 | 51 | 49 | 57 |
| PnP Novel From Cuttingboard To Basket | 8 | 18 | 10 | 58 | 48 | 43 | 55 | 65 |
| PnP Novel From Cuttingboard To Cardboardbox | 8 | 14 | 16 | 46.5 | 40 | 39 | 57 | 69 |
| PnP Novel From Cuttingboard To Pan | 24 | 20 | 27 | 68.5 | 68 | 67 | 65 | 75 |
| PnP Novel From Cuttingboard To Pot | 16 | 25 | 20 | 65 | 52 | 53 | 57 | 61 |
| PnP Novel From Cuttingboard To Tieredbasket | 10 | 10 | 6 | 46.5 | 56 | 29 | 39 | 51 |
| PnP Novel From Placemat To Basket | 8 | 16 | 14 | 58.5 | 42 | 45 | 37 | 53 |
| PnP Novel From Placemat To Bowl | 12 | 10 | 14 | 57.5 | 44 | 55 | 53 | 55 |
| PnP Novel From Placemat To Plate | 10 | 12 | 10 | 63 | 48 | 57 | 51 | 59 |
| PnP Novel From Placemat To Tieredshelf | 2 | 2 | 2 | 28.5 | 18 | 20 | 22 | 24 |
| PnP Novel From Plate To Bowl | 12 | 8 | 14 | 57 | 60 | 49 | 57 | 53 |
| PnP Novel From Plate To Cardboardbox | 2 | 10 | 8 | 43.5 | 50 | 61 | 43 | 43 |
| PnP Novel From Plate To Pan | 10 | 20 | 16 | 51 | 54 | 51 | 49 | 55 |
| PnP Novel From Plate To Plate | 22 | 27 | 25 | 78.7 | 70 | 67 | 59 | 61 |
| PnP Novel From Tray To Cardboardbox | 20 | 25 | 20 | 51.5 | 38 | 49 | 59 | 65 |
| PnP Novel From Tray To Plate | 12 | 18 | 16 | 71 | 56 | 57 | 57 | 63 |
| PnP Novel From Tray To Pot | 18 | 25 | 20 | 64.5 | 50 | 63 | 53 | 55 |
| PnP Novel From Tray To Tieredbasket | 6 | 16 | 16 | 57 | 36 | 55 | 39 | 51 |
| PnP Novel From Tray To Tieredshelf | 4 | 2 | 4 | 31.5 | 16 | 31 | 22 | 33 |
| Average | 14.3 | 19.3 | 20.0 | 47.6 | 47.8 | 51.3 | 48.9 | 55.4 |

**说明**: TABLE VI: Results on RoboCasa-GR1 [42 ] benchmark. UWM: UWM [65 ] with 140M parameters. UWM-XL: UWM with 1B parameters, using Qwen3-VL [55 ] as the joint encoder for language instructions and visual inputs. UWM+MM-DiT: UWM-XL with its DiT [47 ] backbone replaced by our MM-DiT architecture. StarVLA: GR00T [5 ] equipped with Qwen3-VL as its System 2 module. GR00T-EI10k: GR00T [5 ] pretrained on our dataset and equipped with Qwen3-VL as its System 2 module. LDA (DiT): Our LDA model with the MM-DiT replaced by a standard DiT. During finetuning on RoboCasa, the VLM is unfrozen to enable end-to-end adaptation.

#### Table 7: TABLE VII: Real-world gripper manipulation for Galbot task configurations. All tasks are evaluated in-domain with a time

| Task Abbreviation | Description | Test Protocol |
| --- | --- | --- |
| Pick Vegetable | Pick a plastic pepper and place it into a basket using the left gripper. Pepper is randomized within a 15  \times 30 cm region. | 10 trials; success if placed in basket |
| Handover | Left gripper grasps a bottle and passes it to the right gripper, which places it into a basket. Bottle randomized within 15  \times 30 cm. | 10 trials; success if placed in basket |
| Wipe Board | Use an eraser to remove marker writing from a whiteboard. Writing area randomized within 25  \times 40 cm. | 10 trials; scored from 0–5 based on cleaning completeness |
| Flip Box | Flip an upside-down storage box to upright using bimanual manipulation. Box randomized within 2  \times 4 cm. | 10 trials; success if fully flipped |
| Water Flower (pouring) | Grasp a watering bottle and pour water into a flower pot. Pot randomized within 15  \times 15 cm. | 10 trials; success if pouring posture is achieved with spout above pot |
| Knock the block with a hammer (pnp2) | Grasp a hammer with a very thin handle and then knock the specific block. Hammer randomized within 15  \times 15 cm. | 60 trials; success only if both the grasp and knock succeed. |
| Sweep Table | Sweep ten nails into a dustpan using a broom and dustpan. Nail positions randomized within 10  \times 25 cm. | 10 trials; success rate is computed as the proportion of nails collected in the dustpan. |
| Throw Rubbish | Pick paper balls, place them into a dustpan, and dump them into a trash can. Paper balls are randomized within a 20  \times 25 cm area. | 10 trials; success rate is computed as the proportion of paper balls successfully dumped into the trash can. |

**说明**: TABLE VII: Real-world gripper manipulation for Galbot task configurations. All tasks are evaluated in-domain with a timeout of 200 seconds per trial.

#### Table 8: TABLE VIII: Dexterous hand manipulation tasks and evaluation protocols.

| Task Abbreviation | Description | Test Protocol |
| --- | --- | --- |
| Pick Bottle | Pick up a plastic bottle and place it onto a fixed target region using the right hand. Bottle position is randomized. | 20 trials; success if bottle is upright and its base at least half of the target region |
| Open MacBook | Left hand stabilizes the base while the right hand opens the hinge by pushing the upper edge. Initial opening angle is randomized. | 20 trials; success if opening angle exceeds 75% of maximum |
| Pull Nail | Use a claw hammer held by the right hand to extract a nail from the surface. Hammer pose is randomized. | 10 trials; scored with partial credit: 0.25 for locating, 0.5 for single-claw removal, 1.0 for full claw removal |
| Pick Bread | Pick a bread item and place it into a plate using the right hand. Three bread types are used with equal distribution. | 10 trials; success if bread is placed into the plate |
| Flip Bread | Flip a long bread item using a spatula held by the right hand. Bread pose is randomized a large region. | 10 trials; 1.0 if flipped on first attempt, 0.5 if second, 0 otherwise |

**说明**: TABLE VIII: Dexterous hand manipulation tasks and evaluation protocols.

#### Table 9: TABLE IX: Composition of the Embodied Interaction Dataset (EI-30k). The dataset is categorized into four main types, agg

| Data Type | Source / Sub-dataset | Duration (h) |
| --- | --- | --- |
| Real-world Robot | Open X-Embodiment [44 ] | 3000 |
| Agibot World [8 ] | 3276 | |
| RoboMIND [53 ] | 305 | |
| Humanoid Everyday [62 ] | 30 | |
| RoboCOIN [54 ] | 500 | |
| Galaxea [51 ] | 500 | |
| LET [31 ] | 1000 | |
| Simulated Robot | InternData-A1 [13 ] | 7433 |
| Behavior-1k [32 ] | 1200 | |
| Ego Human (w/ Action) | Ego4D [21 ] | 3670 |
| Epic-Kitchens [15 ] | 100 | |
| Ego-Exo4D [22 ] | 1286 | |
| SSV2 [20 ] | 240 | |
| EgoDex [24 ] | 830 | |
| HOT3D [3 ] | 16 | |
| HoloAssist [52 ] | 166 | |
| OAKINK2 [58 ] | 6.5 | |
| TACO [36 ] | 3.2 | |
| HOI4D [37 ] | 7.6 | |
| ARCTIC [17 ] | 2.3 | |
| Ego Human (Actionless) | Egocentric-10k [1 ] | 10000 |
| RH20T-human [18 ] | 100 | |
| EgoMe [48 ] | 80 | |
| | Taste-Rob [60 ] | 130 |
| | Total | 30k+ |

**说明**: TABLE IX: Composition of the Embodied Interaction Dataset (EI-30k). The dataset is categorized into four main types, aggregating 30k hours of data.
## 实验解读

- 评价重点:围绕 action-model、behavior-cloning、benchmark-dataset、接触推理、接触丰富操作,优先看真实机器人结果、扰动恢复、接触稳定性、控制误差、成功率、实时性和 sim-to-real 差距。
- System0 视角:实验结论不只看最终成功率,还要看闭环过程中是否能处理状态估计误差、执行延迟、碰撞/接触突变和外部扰动。
- 与上层系统的关系:如果方法能在快闭环内稳定工作,它可以作为 System1 技能或 System2 规划的底层执行保障;如果只在离线或慢速设定下成立,则还不能直接作为身体级快闭环模块。
## 批判性思考

### 优点

- 论文把 action-model、behavior-cloning、benchmark-dataset、接触推理、接触丰富操作 放到机器人身体级闭环里讨论,和只做高层语义推理的工作相比,更接近真实机器人部署时会遇到的状态、动力学和安全约束。
- 若论文同时给出公式、系统结构图、硬件实验或消融结果,这些证据可以帮助判断方法是否真的能支撑毫秒到百毫秒级反应。

### 局限

- 需要重点核查控制频率、推理延迟、传感延迟、失败恢复、硬件磨损、未见场景泛化和安全边界;这些因素决定它能否成为可靠的 System0 模块。
- 对学习型方法,还要区分训练环境中的鲁棒性和真实部署中的鲁棒性,尤其是接触丰富任务、动态运动任务和人形机器人任务。
## 论文定位

- 研究对象:LDA-1B: Scaling Latent Dynamics Action Model via Universal Embodied Data Ingestion。
- 关键词:action-model、behavior-cloning、benchmark-dataset、接触推理、接触丰富操作、灵巧操作、diffusion-policy、dynamics-modeling、foundation-model、模仿学习。
- System0 关联:关注机器人身体级快反应链路中“状态/接触/局部感知 -> 约束/优化/策略 -> 动作”的短闭环,不把问题停留在语言规划或任务分解层。

## 速查卡片

> [!summary] LDA-1B
> - **论文**: https://www.roboticsproceedings.org/rss22/p210.pdf
> - **arXiv**: http://arxiv.org/abs/2602.12215v2
> - **arXiv HTML**: https://arxiv.org/html/2602.12215v2
