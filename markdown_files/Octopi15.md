---
title: "Demonstrating the Octopi-1.5 Visual-Tactile-Language Model"
method_name: "Octopi 1 5"
authors: ["Samson Yu"]
year: 2025
venue: "RSS"
tags: ["contact-reasoning", "robot-manipulation", "imitation-learning", "robot-generalization", "dexterous-manipulation", "tactile-feedback"]
image_source: "online"
arxiv_html: "https://arxiv.org/html/2507.09985v1"
---
# Octopi 1 5
## 一句话总结

> Demonstrating the Octopi-1.5 Visual-Tactile-Language Model 主要落在 [[接触推理]]、[[灵巧操作]]、[[foundation-model]]、[[grasping]] 这一类问题上;从 System0 视角看,它的价值在于把身体状态、接触变化、局部感知或动力学约束接到机器人低层闭环里,让动作生成更稳定、安全、实时。
## 核心贡献

1. 围绕 **Demonstrating the Octopi-1.5 Visual-Tactile-Language Model** 建立了一个与 接触推理、灵巧操作、foundation-model、grasping、模仿学习、机器人操作 相关的机器人问题设定,重点不是高层语义规划,而是身体级闭环中的状态变化、接触变化或控制约束。
2. 方法层面的关键价值在于把感知、动力学、约束优化、策略学习或硬件反馈组织成可执行的低层动作生成机制,适合放进 System0 的快反应模块中理解。
3. 对系统集成而言,这类工作补上了 System1 技能选择与电机级控制之间的一层:它负责把上层目标变成动态可行、可恢复、可安全过滤的动作。
4. 若论文包含真实机器人、接触任务或仿真到真实迁移,核心应看它是否报告了闭环稳定性、扰动恢复、硬件延迟、失败模式和跨场景泛化。
## 问题背景

这篇论文可以放在 System0 的“具身快系统”背景下阅读:机器人需要在很短时间内根据本体状态、接触、力/触觉、局部视觉或动力学预测做出动作,而不是等待慢速规划器重新推理整个任务。接触推理、灵巧操作、foundation-model、grasping、模仿学习、机器人操作、tactile-feedback 相关问题的难点通常不在“知道要做什么”,而在动作执行过程中状态会快速偏离、接触会突然改变、约束会实时收紧,低层系统必须及时修正。

### System0 关联

精读时优先看三件事:第一,论文使用哪些快速反馈量,例如关节状态、接触状态、力觉、触觉、深度或局部视觉;第二,反馈如何进入 MPC、WBC、策略网络、安全过滤器或恢复控制器;第三,输出是否直接影响机器人下一步动作,而不是只服务离线分析或高层任务规划。
## 方法详解

### System0 视角解析

从 System0 角度看,这篇论文的方法部分应当围绕 接触推理、灵巧操作、foundation-model、grasping、模仿学习、机器人操作、tactile-feedback、visual-tactile 来读。核心问题是:系统如何从快速变化的机器人状态中构造控制输入,如何处理接触、碰撞、约束、模型误差或执行延迟,以及如何把这些信息变成下一时刻的动作。

如果方法里有优化问题,重点看目标函数、约束和求解频率;如果方法里有学习策略,重点看观测空间、动作空间、训练扰动和部署时的闭环频率;如果方法里有硬件或传感器,重点看传感链路是否真的进入控制回路。这样读可以把论文和 System2 语言规划、System1 技能编排区分开来。

### 关键公式与机制

未抽到可稳定还原的核心公式;不把单个变量或碎片数学符号伪装成公式。
## 关键图表

### Figure 1: Pipeline / core system figure: Octopi-1.5 Demonstrations using a Tactile Manipulation Interface (TMI) gripper. We pla

![Figure 1](https://arxiv.org/html/2507.09985v1/x1.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Pipeline / core system figure: Octopi-1.5 Demonstrations using a Tactile Manipulation Interface (TMI) gripper. We pla”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 2: A) Octopi-1.5 model architecture. Octopi-1.5 is a fine-tuned Qwen2-VL 7B multimodal

![Figure 2](https://arxiv.org/html/2507.09985v1/x2.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“A) Octopi-1.5 model architecture. Octopi-1.5 is a fine-tuned Qwen2-VL 7B multimodal”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 3: Our demonstration system is highly portable, consisting primarily of the TMI and a lap

![Figure 3](https://arxiv.org/html/2507.09985v1/x4.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Our demonstration system is highly portable, consisting primarily of the TMI and a lap”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### 论文表格

#### Table 1: TABLE I: Octopi-1.5 Training Dataset Statistics

| Dataset | Num. | Samples | Num. Tactile |
| --- | --- | --- | --- |
| | Objects | per Object | Videos |
| PhysiCLeAR-Plain | 100 | 5–45 | 2689 |
| PhysiCLeAR-Dotted | 68 | 11–32 | 1939 |
| Hardness [15 ] | 210 | 1–133 | 1860 |
| ObjectFolder-Real [13 ] | 100 | 30–67 | 3550 |

**说明**: TABLE I: Octopi-1.5 Training Dataset Statistics

#### Table 2: TABLE II: Annotator Scoring Guidance for Hardness and Roughness.

| Property | Score | Description | Example Object |
| --- | --- | --- | --- |
| Hardness | 0 | Easily compressible when pressed with little force | Cotton wool |
| 5 | Compressible upon pressing with moderate force | Foam mattress | |
| 10 | Incompressible with high human pressing force | Rock | |
| Roughness | 0 | Minimal feeling of friction upon finger sliding | Ice |
| 5 | Slight feeling of friction upon finger sliding | Jeans | |
| 10 | Significant feeling of friction upon finger sliding | Toothbrush bristles | |

**说明**: TABLE II: Annotator Scoring Guidance for Hardness and Roughness.

#### Table 3: TABLE III: Average accuracies (%) on the Guessing Game.

| Method | RAG | Balls | Fruits | Unseen | Unseen |
| --- | --- | --- | --- | --- | --- |
| | | | | | (teaching) |
| Encoder-1.5 | – | 80.00 | 100.00 | N/A | 89.02 |
| Octopi-1 (7B) | No | 44.00 | 42.31 | 43.90 | N/A |
| +-Octopi-1 (13B) | No | 48.00 | 34.62 | 53.66 | N/A |
| Octopi-1.5 (8B) | No | 56.00 | 57.69 | 41.46 | N/A |
| Octopi-1.5 (8B) | Yes | 96.00 | 100.00 | 73.17 | 95.12 |

**说明**: TABLE III: Average accuracies (%) on the Guessing Game.
## 实验解读

- 评价重点:围绕 接触推理、灵巧操作、foundation-model、grasping、模仿学习,优先看真实机器人结果、扰动恢复、接触稳定性、控制误差、成功率、实时性和 sim-to-real 差距。
- System0 视角:实验结论不只看最终成功率,还要看闭环过程中是否能处理状态估计误差、执行延迟、碰撞/接触突变和外部扰动。
- 与上层系统的关系:如果方法能在快闭环内稳定工作,它可以作为 System1 技能或 System2 规划的底层执行保障;如果只在离线或慢速设定下成立,则还不能直接作为身体级快闭环模块。
## 批判性思考

### 优点

- 论文把 接触推理、灵巧操作、foundation-model、grasping、模仿学习 放到机器人身体级闭环里讨论,和只做高层语义推理的工作相比,更接近真实机器人部署时会遇到的状态、动力学和安全约束。
- 若论文同时给出公式、系统结构图、硬件实验或消融结果,这些证据可以帮助判断方法是否真的能支撑毫秒到百毫秒级反应。

### 局限

- 需要重点核查控制频率、推理延迟、传感延迟、失败恢复、硬件磨损、未见场景泛化和安全边界;这些因素决定它能否成为可靠的 System0 模块。
- 对学习型方法,还要区分训练环境中的鲁棒性和真实部署中的鲁棒性,尤其是接触丰富任务、动态运动任务和人形机器人任务。
## 论文定位

- 研究对象:Demonstrating the Octopi-1.5 Visual-Tactile-Language Model。
- 关键词:接触推理、灵巧操作、foundation-model、grasping、模仿学习、机器人操作、tactile-feedback、visual-tactile。
- System0 关联:关注机器人身体级快反应链路中“状态/接触/局部感知 -> 约束/优化/策略 -> 动作”的短闭环,不把问题停留在语言规划或任务分解层。

## 速查卡片

> [!summary] Octopi 1 5
> - **论文**: https://www.roboticsproceedings.org/rss21/p058.pdf
> - **arXiv**: http://arxiv.org/abs/2507.09985v1
> - **arXiv HTML**: https://arxiv.org/html/2507.09985v1
