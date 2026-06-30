---
title: "Learning to Learn Faster from Human Feedback with Language Model Predictive Control"
method_name: "Learn Faster Human"
authors: ["Jacky Liang"]
year: 2024
venue: "RSS"
tags: ["robust-control", "adaptive-control", "imitation-learning", "robot-generalization", "closed-loop-control", "model-predictive-control"]
image_source: "online"
arxiv_html: "https://arxiv.org/html/2402.11450"
---
# Learn Faster Human
## 一句话总结

> Learning to Learn Faster from Human Feedback with Language Model Predictive Control 主要落在 [[adaptive-control]]、[[closed-loop-control]]、[[dynamics-modeling]]、[[模仿学习]] 这一类问题上;从 System0 视角看,它的价值在于把身体状态、接触变化、局部感知或动力学约束接到机器人低层闭环里,让动作生成更稳定、安全、实时。
## 核心贡献

1. 围绕 **Learning to Learn Faster from Human Feedback with Language Model Predictive Control** 建立了一个与 adaptive-control、closed-loop-control、dynamics-modeling、模仿学习、language-conditioned-policy、模型预测控制 相关的机器人问题设定,重点不是高层语义规划,而是身体级闭环中的状态变化、接触变化或控制约束。
2. 方法层面的关键价值在于把感知、动力学、约束优化、策略学习或硬件反馈组织成可执行的低层动作生成机制,适合放进 System0 的快反应模块中理解。
3. 对系统集成而言,这类工作补上了 System1 技能选择与电机级控制之间的一层:它负责把上层目标变成动态可行、可恢复、可安全过滤的动作。
4. 若论文包含真实机器人、接触任务或仿真到真实迁移,核心应看它是否报告了闭环稳定性、扰动恢复、硬件延迟、失败模式和跨场景泛化。
## 问题背景

这篇论文可以放在 System0 的“具身快系统”背景下阅读:机器人需要在很短时间内根据本体状态、接触、力/触觉、局部视觉或动力学预测做出动作,而不是等待慢速规划器重新推理整个任务。adaptive-control、closed-loop-control、dynamics-modeling、模仿学习、language-conditioned-policy、模型预测控制 相关问题的难点通常不在“知道要做什么”,而在动作执行过程中状态会快速偏离、接触会突然改变、约束会实时收紧,低层系统必须及时修正。

### System0 关联

精读时优先看三件事:第一,论文使用哪些快速反馈量,例如关节状态、接触状态、力觉、触觉、深度或局部视觉;第二,反馈如何进入 MPC、WBC、策略网络、安全过滤器或恢复控制器;第三,输出是否直接影响机器人下一步动作,而不是只服务离线分析或高层任务规划。
## 方法详解

### System0 视角解析

从 System0 角度看,这篇论文的方法部分应当围绕 adaptive-control、closed-loop-control、dynamics-modeling、模仿学习、language-conditioned-policy、模型预测控制 来读。核心问题是:系统如何从快速变化的机器人状态中构造控制输入,如何处理接触、碰撞、约束、模型误差或执行延迟,以及如何把这些信息变成下一时刻的动作。

如果方法里有优化问题,重点看目标函数、约束和求解频率;如果方法里有学习策略,重点看观测空间、动作空间、训练扰动和部署时的闭环频率;如果方法里有硬件或传感器,重点看传感链路是否真的进入控制回路。这样读可以把论文和 System2 语言规划、System1 技能编排区分开来。

### 关键公式与机制

未抽到可稳定还原的核心公式;不把单个变量或碎片数学符号伪装成公式。
## 关键图表

### Figure 1: Pipeline / core system figure: LMPC rollout versus skip comparison from the project page

![Figure 1](https://robot-teaching.github.io/assets/images/lmpc-rollout-vs-skip-v1.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Pipeline / core system figure: LMPC rollout versus skip comparison from the project page”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 2: Main experiment plots for language model predictive control and human feedback learning

![Figure 2](https://robot-teaching.github.io/assets/images/main-experiment-plots.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Main experiment plots for language model predictive control and human feedback learning”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 3: Project-page Table 1 image summarizing experimental results

![Figure 3](https://robot-teaching.github.io/assets/images/table1.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Project-page Table 1 image summarizing experimental results”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### 论文表格

#### Table 1: Performance on RoboCodeGen on finetuned models.

| tabular{ccc} | 2{c}{Pass@1} | |
| --- | --- | --- |
| Model | Iteration 1 | Iteration 2 |
| PaLM 2-S | 2{c}{51\ | |
| LMPC-Rollouts | 51\ | 51\ |
| LMPC-Skip | 49\ | 49\ |
| tabular | | |

**说明**: Performance on RoboCodeGen on finetuned models.

#### Table 2: TeX source table

| tabular{llcc} Data | Model | Train Tasks | Test Tasks |
| --- | --- | --- | --- |
| All Users | LMPC-Rollouts | -8.4\ | -10.5\ |
| LMPC-Skip | -16.3\ | -26.1\ | |
| Only Top Users | LMPC-Rollouts | -23.8\ | -21.7\ |
| LMPC-Skip | -9.6\ | -13.6\ | |
| tabular | | | |

**说明**: TeX source table

#### Table 3: TeX source table

| tabular{lcc} Model | Top Users | Other Users |
| --- | --- | --- |
| LMPC-Skip | +15.1\ | +14.2\ |
| LMPC-Rollout | +26.3\ | +18.9\ |
| tabular | | |

**说明**: TeX source table

#### Table 4: TeX source table

| tabular{lcccc} Model Name | {Better} | {Same} | {Worse} | Split |
| --- | --- | --- | --- | --- |
| RAG | 58\ | 10\ | 32\ | train |
| RAG | 44\ | 25\ | 31\ | test |
| LMPC-Skip | 68\ | 7\ | 24\ | train |
| LMPC-Skip | 62\ | 15\ | 23\ | test |
| LMPC-Rollouts | 68\ | 16\ | 16\ | train |
| LMPC-Rollouts | 69\ | 8\ | 23\ | test |
| tabular | | | | |

**说明**: TeX source table

#### Table 5: TeX source table

| tabular{lcc} Mode | Split | IOU |
| --- | --- | --- |
| better | train | 0.28 |
| better | test | 0.14 |
| same | train | 0.00 |
| same | test | 0.00 |
| worse | train | 0.11 |
| worse | test | 0.00 |
| tabular | | |

**说明**: TeX source table

#### Table 6: Comparing base and finetuned models across all embodiments. {Success

| tabular{llcccccc} Tasks | Model | Success Rate | Num Chat Turns | Good Rating Rate | Successful Tasks Rate | 1 Turn Success Rate | 2+ Turn Success Rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Train | PaLM 2-S | 34.8\ | 2.3 | 16.7\ | 74.0\ | 13.0\ | 21.7\ |
| RAG | 46.4\ | 2.2 | 21.4\ | {83.3\ | 25.1\ | 21.2\ | |
| LMPC-Skip | {56.0\ | 1.7 | {25.6\ | {83.3\ | {34.6\ | 21.4\ | |
| LMPC-Rollouts | 51.9\ | 2.2 | 21.8\ | 74.0\ | 23.5\ | {28.4\ | |
| Test | PaLM 2-S | 39.4\ | 2.4 | 18.1\ | 81.5\ | 17.5\ | 21.9\ |
| RAG | 51.9\ | 2.0 | 20.9\ | 75.0\ | 27.9\ | 24.0\ | |
| LMPC-Skip | 59.4\ | 1.6 | 24.7\ | {88.9\ | {41.7\ | 17.8\ | |
| LMPC-Rollouts | {66.3\ | 1.9 | {26.5\ | {88.9\ | 34.8\ | {31.5\ | |
| tabular | | | | | | | |

**说明**: Comparing base and finetuned models across all embodiments. {Success

#### Table 7: Changes in success rate without Top-User Conditioning. We evaluate two variants of LMPC-Rollouts and LMPC-Skip that do n

| tabular{llcc} Data | Model | Train Tasks | Test Tasks |
| --- | --- | --- | --- |
| All Users | LMPC-Rollouts | -8.4\ | -10.5\ |
| LMPC-Skip | -16.3\ | -26.1\ | |
| Only Top Users | LMPC-Rollouts | -23.8\ | -21.7\ |
| LMPC-Skip | -9.6\ | -13.6\ | |
| tabular | | | |

**说明**: Changes in success rate without Top-User Conditioning. We evaluate two variants of LMPC-Rollouts and LMPC-Skip that do not apply top-user conditioning: training on data from all users and training on data from only top users. Success rates degrade significantly for both variants, suggesting that 1) focusing LLM generation on the style of top-users is important and 2) top-user data alone is insufficient, and training on the wider data distribution of all users is still important.

#### Table 8: LMPC-Rollouts has higher success than PaLM 2-S on real robots. Test tasks are starred$^*$. Robot Dog tasks are performed

| tabular{llcccc} | 2{c}{PaLM 2-S} | 2{c}{LMPC-Rollouts} | | | |
| --- | --- | --- | --- | --- | --- |
| (lr){3-6} Embodiment | Task | Success | Num Chat Turns | Success | Num Chat Turns |
| Robot Dog | ``downward dog'' | $100\ |$1.3$|$100\ | $2.8$ |
| ``hop'' | $25\ |$2.0$|$100\ | $2.3$ | |
| ``high-five with left hand''$^*$ | $75\ |$2.3$|$75\ | $3.0$ | |
| ``walk forward in a trotting gait''$^*$ | $25\ |$2.0$|$100\ | $2.8$ | |
| ``hop while turning counterclockwise''$^*$ | $25\ |$5.0$|$25\ | $4.0$ | |
| Mobile Manipulator | ``knock coke can'' | $20\ |$5.0$|$20\ | $3.0$ |
| ``open top drawer half-way''$^*$ | $100\ |$3.4$|$100\ | $3.2$ | |
| ``push coke can from right to left''$^*$ | $60\ |$2.0$|$80\ | $2.0$ | |
| 2{c}{Average} | $53.8\ |$2.9$|$ {75\ | $2.9$ | |
| tabular | | | | | |

**说明**: LMPC-Rollouts has higher success than PaLM 2-S on real robots. Test tasks are starred$^*$. Robot Dog tasks are performed $4$ times, Mobile Manipulator tasks $5$ times.

#### Table 9: Finetuned models can generalize to new robot embodiments and APIs not seen during training. Higher improvements in test

| tabular{lccc} | 2{c}{Train Embodiments} | Test Embodiments | |
| --- | --- | --- | --- |
| (lr){2-3} Model | Train Tasks | Test Tasks | |
| LMPC-Skip | $+28.8\ |$+19.0\ | $+18.6\ |
| LMPC-Rollouts | $+17.2\ |$+23.8\ | $+31.5\ |
| tabular | | | |

**说明**: Finetuned models can generalize to new robot embodiments and APIs not seen during training. Higher improvements in test tasks and embodiments are caused by the train:test split not being explicitly selected for uniform task difficulty and baseline performance; doing so is infeasible as the split needs to be chosen before starting evaluations, when task difficulty and baseline performance were unknown.

#### Table 10: Further finetuning on data generated from both the base model and the first finetuned models models does not yield perfo

| tabular{lcc} | 2{c}{Success Rate Diff from Iter 1} | |
| --- | --- | --- |
| (lr){2-3} Model | Train Tasks | Test Tasks |
| LMPC-Skip Iter 2 | +5.1\ | -4.7\ |
| LMPC-Rollouts Iter 2 | -5.5\ | -1.9\ |
| tabular | | |

**说明**: Further finetuning on data generated from both the base model and the first finetuned models models does not yield performance improvements.

#### Table 11: Median Chat Session and Chat Turn Durations across Embodiments

| tabular{lcc} Embodiment | Chat Session Duration (s) | Chat Turn Duration (s) |
| --- | --- | --- |
| Kuka+Hand | 429 | 97 |
| Bi-arm Kuka | 406 | 88 |
| Aloha | 200 | 66 |
| Mobile Manipulator | 238 | 65 |
| Robot Dog | 138 | 41 |
| tabular | | |

**说明**: Median Chat Session and Chat Turn Durations across Embodiments

#### Table 12: Median Chat Session and Chat Turn Durations across Models

| tabular{lcc} Model | Chat Session Duration (s) | Chat Turn Duration (s) |
| --- | --- | --- |
| LMPC-Rollouts | 187 | 60 |
| LMPC-Skip | 158 | 49 |
| tabular | | |

**说明**: Median Chat Session and Chat Turn Durations across Models

#### Table 13: Success Rates of Training LMPC-Rollouts on both Success and Failure chat sessions.

| tabular{lcc} | Train Tasks | Test Tasks |
| --- | --- | --- |
| LMPC-Rollouts-with-Failures | $-11.5\ |$-14.0\ |
| tabular | | |

**说明**: Success Rates of Training LMPC-Rollouts on both Success and Failure chat sessions.

#### Table 14: Percent of tasks that saw improvements and degradations the base model

| tabular{lcccc} Model Name | {Better} | {Same} | {Worse} | Split |
| --- | --- | --- | --- | --- |
| RAG | 58\ | 10\ | 32\ | train |
| RAG | 44\ | 25\ | 31\ | test |
| LMPC-Skip | 68\ | 7\ | 24\ | train |
| LMPC-Skip | 62\ | 15\ | 23\ | test |
| LMPC-Rollouts | 68\ | 16\ | 16\ | train |
| LMPC-Rollouts | 69\ | 8\ | 23\ | test |
| tabular | | | | |

**说明**: Percent of tasks that saw improvements and degradations the base model

#### Table 15: Overlap (intersection union) among tasks that were better/same/worse the base model across train/test splits.

| tabular{lcc} Mode | Split | IOU |
| --- | --- | --- |
| better | train | 0.28 |
| better | test | 0.14 |
| same | train | 0.00 |
| same | test | 0.00 |
| worse | train | 0.11 |
| worse | test | 0.00 |
| tabular | | |

**说明**: Overlap (intersection union) among tasks that were better/same/worse the base model across train/test splits.

#### Table 16: Performance on RoboCodeGen on finetuned models.

| tabular{ccc} | 2{c}{Pass@1} | |
| --- | --- | --- |
| Model | Iteration 1 | Iteration 2 |
| PaLM 2-S | 2{c}{51\ | |
| LMPC-Rollouts | 51\ | 51\ |
| LMPC-Skip | 49\ | 49\ |
| tabular | | |

**说明**: Performance on RoboCodeGen on finetuned models.

#### Table 17: Success rate differences between models that do not use data augmentation and models that do.

| tabular{lcc} | 2{c}{Success Rate Diff w/o Data Augmentation} | |
| --- | --- | --- |
| (lr){2-3} Model | Train Tasks | Test Tasks |
| LMPC-Skip w/o Aug | -7.1\ | +0.6\ |
| LMPC-Rollouts w/o Aug | +2.8\ | -7.0\ |
| tabular | | |

**说明**: Success rate differences between models that do not use data augmentation and models that do.

#### Table 18: TeX source table

| tabular{\|p{0.8 }\|} Robot Dog Train Tasks |
| --- |
| Sit. |
| High-five with the front right paw. |
| Downward dog. |
| Walk to the left. |
| Walk forward. |
| Hop. |
| Turn around clockwise. |
| Walk backward. |
| Walk backward while turning to face right. |
| Walk forward while turning left. |
| Close the middle drawer. |
| Open the door by pushing it. |
| tabular |

**说明**: TeX source table

#### Table 19: TeX source table

| tabular{\|p{0.8 }\|} Robot Dog Test Tasks |
| --- |
| High-five with the front left paw. |
| Walk to the right. |
| Turn around counterclockwise. |
| Hop while turning clockwise. |
| Hop while turning counterclockwise. |
| Close the bottom drawer. |
| Close the door by pushing it. |
| tabular |

**说明**: TeX source table

#### Table 20: TeX source table

| tabular{\|p{0.8 }\|} Mobile Manipulator Train Tasks |
| --- |
| Grasp the apple. |
| Knock coke can. |
| Lift the apple high. |
| Place the apple next to the cube. |
| Push the apple toward the cube. |
| Move the cube further away from the robot. |
| Move the cube a little bit to the left. |
| Open the top drawer. |
| Place the cube behind the apple. |
| Flip the cube upside down. |
| Place the apple on the cube. |
| tabular |

**说明**: TeX source table

#### Table 21: TeX source table

| tabular{\|p{0.8 }\|} Mobile Manipulator Test Tasks |
| --- |
| Pick up the cube. |
| Place the apple in front of the cube. |
| Upright the coke can. |
| tabular |

**说明**: TeX source table

#### Table 22: TeX source table

| tabular{\|p{0.8 }\|} Aloha Train Tasks |
| --- |
| Grasp the apple and lift it up. |
| Grasp the coke can and lift it up. |
| Pick up the cube and lift it above the apple. |
| Pick up the box and lift it above the coke can. |
| Flip the box upside down. |
| Flip the apple upside down. |
| Flip the drink upside down. |
| Flip the apple upside down and move the apple to the center of the table. |
| Move the box and the apple close to each other. |
| Push the box and the bowl close to each other. |
| tabular |

**说明**: TeX source table

#### Table 23: TeX source table

| tabular{\|p{0.8 }\|} Aloha Test Tasks |
| --- |
| Grasp the box and lift it up. |
| Pick up the coke can and lift it above the apple. |
| Flip the bowl upside down. |
| Move the apple and the bowl closer to each other. |
| Move the foods closer to each other. |
| Flip the box upside down and move the box to the center of the table. |
| tabular |

**说明**: TeX source table

#### Table 24: TeX source table

| tabular{\|p{0.8 }\|} Bi-arm Kuka Single Large Cube Scene Test Tasks |
| --- |
| Pick up the cube and lift it up to the blue goal. |
| Pick up the cube and lift it up by 20cm. |
| Move the cube to the green goal on the floor. |
| Move the cube 20cm to the right without rotating it. |
| Pick up the cube and lift it up to the red goal. |
| Move the cube 20cm to the left of the purple goal on the floor and rotate it 90 degrees. |
| tabular |

**说明**: TeX source table

#### Table 25: TeX source table

| tabular{\|p{0.8 }\|} Bi-arm Kuka Particle Manipulation Scene Test Tasks |
| --- |
| Move the blue cube to the green goal on the floor. |
| Move the green cube 20cm to the right. |
| Move the red cube to the green goal, then to the purple goal. |
| Sweep the red cube and the blue cube towards the green goal. |
| Sweep all the cubes to the purple goal. |
| Bring the red cube 20cm to the left of the green goal. |
| Move the blue cube 20cm in front of the green cube. |
| Sweep the yellow cube to the blue cube, then to the red cube. |
| Move the purple cube to the yellow cube, then to the green cube, then to the blue cube. |
| Move the purple cube 10cm to the right of the yellow cube, then 20cm behind the blue cube. |
| tabular |

**说明**: TeX source table

#### Table 26: TeX source table

| tabular{\|p{0.8 }\|} Kuka+Hand Test Tasks |
| --- |
| Move the gripper to reach the red block. |
| Lift the connector in the air. |
| Grasp the green object, hold it for a while in the air, and then drop it. |
| Insert the connector into the socket. |
| Stack the red block on the green block. |
| Move the green cube to the far right corner. |
| Move the red thing and the plug base to the far left corner. |
| Stack the red block on the the base. |
| Move the four objects into different corners. |
| Move all objects into near left corner. |
| Insert the plug into the base and stack the red cube on the green cube. |
| Insert the connector into the socket, then put the green cube on the connector. |
| Lift both cubes in the air. |
| Disconnect the connector from the base. |
| Separate the red block from the green block. |
| Separate the red block away from the other objects. |
| Move all objects into near left corner. |
| Move the four objects into different corners. |
| tabular |

**说明**: TeX source table

#### Table 27: Failure Mode as percentage of all chat sessions.

| tabular{lccccc} | 5{c}{Failure Modes} | | | | |
| --- | --- | --- | --- | --- | --- |
| 2-6 Model | Invalid Code | Repeated Code | Non-responsive Code | Incomplete Code | All Failures |
| PaLM 2-S | 17.4\ | 10.9\ | 16.8\ | 7.6\ | 35.3\ |
| RAG | 6.4\ | {6.7\ | 19.8\ | 6.4\ | 38.5\ |
| LMPC-Skip | 9.5\ | 7.6\ | 11.9\ | {3.8\ | {23.0\ |
| LMPC-Rollout | {7.8\ | 7.0\ | {11.3\ | 4.0\ | 24.7\ |
| tabular | | | | | |

**说明**: Failure Mode as percentage of all chat sessions.

#### Table 28: Model inference times in seconds.

| tabular{ccc} LMPC-Skip | LMPC-Rollouts | LMPC-Rollouts-No-Quantization |
| --- | --- | --- |
| $1.1 0.2$ | $1.0 0.4$ | $7.4 4.7$ |
| tabular | | |

**说明**: Model inference times in seconds.

#### Table 29: Sim vs. Real Results

| tabular{llcccccc} Embodiment | Task | Ours-Sim | Ours-Real | N Chat Turns | PaLM 2-S Sim | PaLM 2-S Real | N Chat Turns |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Robot Dog | High-Five with left hand | $100\ |$100\ | $3.0$ | $100\ |$75\ | $2.3$ |
| Downward Dog | $100\ |$100\ | $2.8$ | $100\ |$100\ | $1.3$ | |
| Walk forward in a trotting gait | $100\ |$100\ | $2.8$ | $25\ |$25\ | $2.0$ | |
| Hop | $100\ |$75\ | $2.3$ | $50\ |$25\ | $2.0$ | |
| Hop while turning counterclockwise | $100\ |$25\ | $4.0$ | $100\ |$25\ | $5.0$ | |
| Mobile Manipulator | Open top drawer half-way | $100\ |$100\ | $3.2$ | $100\ |$100\ | $3.4$ |
| Push coke can from right to left | $100\ |$80\ | $2.0$ | $100\ |$60\ | $2.0$ | |
| Knock coke can | $100\ |$20\ | $3.0$ | $80\ |$20\ | $5.0$ | |
| tabular | | | | | | | |

**说明**: Sim vs. Real Results

#### Table 30: Success rate improvements by user group for test tasks.

| tabular{lcc} Model | Top Users | Other Users |
| --- | --- | --- |
| LMPC-Skip | $+15.1\ |$+14.2\ |
| LMPC-Rollouts | $+26.3\ |$+18.9\ |
| tabular | | |

**说明**: Success rate improvements by user group for test tasks.
## 实验解读

- 评价重点:围绕 adaptive-control、closed-loop-control、dynamics-modeling、模仿学习、language-conditioned-policy,优先看真实机器人结果、扰动恢复、接触稳定性、控制误差、成功率、实时性和 sim-to-real 差距。
- System0 视角:实验结论不只看最终成功率,还要看闭环过程中是否能处理状态估计误差、执行延迟、碰撞/接触突变和外部扰动。
- 与上层系统的关系:如果方法能在快闭环内稳定工作,它可以作为 System1 技能或 System2 规划的底层执行保障;如果只在离线或慢速设定下成立,则还不能直接作为身体级快闭环模块。
## 批判性思考

### 优点

- 论文把 adaptive-control、closed-loop-control、dynamics-modeling、模仿学习、language-conditioned-policy 放到机器人身体级闭环里讨论,和只做高层语义推理的工作相比,更接近真实机器人部署时会遇到的状态、动力学和安全约束。
- 若论文同时给出公式、系统结构图、硬件实验或消融结果,这些证据可以帮助判断方法是否真的能支撑毫秒到百毫秒级反应。

### 局限

- 需要重点核查控制频率、推理延迟、传感延迟、失败恢复、硬件磨损、未见场景泛化和安全边界;这些因素决定它能否成为可靠的 System0 模块。
- 对学习型方法,还要区分训练环境中的鲁棒性和真实部署中的鲁棒性,尤其是接触丰富任务、动态运动任务和人形机器人任务。
## 论文定位

- 研究对象:Learning to Learn Faster from Human Feedback with Language Model Predictive Control。
- 关键词:adaptive-control、closed-loop-control、dynamics-modeling、模仿学习、language-conditioned-policy、模型预测控制。
- System0 关联:关注机器人身体级快反应链路中“状态/接触/局部感知 -> 约束/优化/策略 -> 动作”的短闭环,不把问题停留在语言规划或任务分解层。

## 速查卡片

> [!summary] Learn Faster Human
> - **论文**: https://www.roboticsproceedings.org/rss20/p125.pdf
> - **arXiv**: http://arxiv.org/abs/2402.11450v2
> - **arXiv HTML**: https://arxiv.org/html/2402.11450
> - **项目页**: https://robot-teaching.github.io/
