---
title: "HumanoidBench: Simulated Humanoid Benchmark for Whole-Body Locomotion and Manipulation"
method_name: "HumanoidBench"
authors: ["Carmelo Sferrazza"]
year: 2024
venue: "RSS"
tags: ["contact-reasoning", "robot-manipulation", "robust-control", "legged-locomotion", "reinforcement-learning", "adaptive-control", "robot-generalization", "humanoid", "dexterous-manipulation", "agile-locomotion", "loco-manipulation", "whole-body-control"]
image_source: "online"
arxiv_html: "https://arxiv.org/html/2403.10506v2"
---
# HumanoidBench
## 一句话总结

> HumanoidBench: Simulated Humanoid Benchmark for Whole-Body Locomotion and Manipulation 主要落在 [[adaptive-control]]、[[agile-locomotion]]、[[benchmark-dataset]]、[[接触推理]] 这一类问题上;从 System0 视角看,它的价值在于把身体状态、接触变化、局部感知或动力学约束接到机器人低层闭环里,让动作生成更稳定、安全、实时。
## 核心贡献

1. 围绕 **HumanoidBench: Simulated Humanoid Benchmark for Whole-Body Locomotion and Manipulation** 建立了一个与 adaptive-control、agile-locomotion、benchmark-dataset、接触推理、灵巧操作、人形机器人 相关的机器人问题设定,重点不是高层语义规划,而是身体级闭环中的状态变化、接触变化或控制约束。
2. 方法层面的关键价值在于把感知、动力学、约束优化、策略学习或硬件反馈组织成可执行的低层动作生成机制,适合放进 System0 的快反应模块中理解。
3. 对系统集成而言,这类工作补上了 System1 技能选择与电机级控制之间的一层:它负责把上层目标变成动态可行、可恢复、可安全过滤的动作。
4. 若论文包含真实机器人、接触任务或仿真到真实迁移,核心应看它是否报告了闭环稳定性、扰动恢复、硬件延迟、失败模式和跨场景泛化。
## 问题背景

这篇论文可以放在 System0 的“具身快系统”背景下阅读:机器人需要在很短时间内根据本体状态、接触、力/触觉、局部视觉或动力学预测做出动作,而不是等待慢速规划器重新推理整个任务。adaptive-control、agile-locomotion、benchmark-dataset、接触推理、灵巧操作、人形机器人、足式运动 相关问题的难点通常不在“知道要做什么”,而在动作执行过程中状态会快速偏离、接触会突然改变、约束会实时收紧,低层系统必须及时修正。

### System0 关联

精读时优先看三件事:第一,论文使用哪些快速反馈量,例如关节状态、接触状态、力觉、触觉、深度或局部视觉;第二,反馈如何进入 MPC、WBC、策略网络、安全过滤器或恢复控制器;第三,输出是否直接影响机器人下一步动作,而不是只服务离线分析或高层任务规划。
## 方法详解

### System0 视角解析

从 System0 角度看,这篇论文的方法部分应当围绕 adaptive-control、agile-locomotion、benchmark-dataset、接触推理、灵巧操作、人形机器人、足式运动、移动操作 来读。核心问题是:系统如何从快速变化的机器人状态中构造控制输入,如何处理接触、碰撞、约束、模型误差或执行延迟,以及如何把这些信息变成下一时刻的动作。

如果方法里有优化问题,重点看目标函数、约束和求解频率;如果方法里有学习策略,重点看观测空间、动作空间、训练扰动和部署时的闭环频率;如果方法里有硬件或传感器,重点看传感链路是否真的进入控制回路。这样读可以把论文和 System2 语言规划、System1 技能编排区分开来。

### 关键公式与机制

#### 公式 1: [[动力学或策略机制]]

$$e:=0.2\cdot\left[4+\frac{1}{|u|}\sum_{i}tol(u_{i},(0,0),10)\right]$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 2: [[动力学或策略机制]]

$$\gamma_{\text{collision}}=\begin{cases}0.1,&,\text{robot collides with pole}\\ 1,&\text{otherwise}\end{cases}$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 3: [[动力学或策略机制]]

$$close=\begin{cases}5,&d_{\text{hand}}<1\\ 0,&\text{otherwise}\end{cases}$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 4: [[动力学或策略机制]]

$$success=\begin{cases}10,&d_{\text{hand}}<0.05\\ 0,&\text{otherwise}\end{cases}$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 5: [[优化目标/约束]]

$$truck=tol(\min_{p\in p_{\text{truck}}}\lVert pos_{p}-pos_{\text{pelvis}} \rVert,(0,0.2),4)$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 6: [[优化目标/约束]]

$$table=tol(\min_{p\in p_{\text{table}}}\lVert pos_{p}-pos_{\text{table}} \rVert,(0,0.2),4)$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 7: [[动力学或策略机制]]

$$destination=\begin{bmatrix}x_{\text{pot}}+0.06\cos(\frac{t\pi}{20})\\ y_{\text{pot}}+0.06\sin(\frac{t\pi}{20})\\ z_{\text{pot}}\end{bmatrix}$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 8: [[动力学或策略机制]]

$$\gamma_{\text{collision}}=\begin{cases}0.1&,\text{robot is colliding with wall}\\ 1&,\text{otherwise}\end{cases}$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 9: [[优化目标/约束]]

$$=0.5\cdot\max(open_{\text{door,left}},open_{\text{door,right}})+0.5\cdot r_{\text{destination}}$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 10: [[优化目标/约束]]

$$picked=tol(\min_{p\in p_{\text{picked}}}\lVert pos_{p}-pos_{\text{pelvis}} \rVert,(0,0.2),4)$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。
## 关键图表

### Figure 1: Pipeline / core system figure: Example egocentric visual (top-left) and whole-body tactile (right) observations when

![Figure 1](https://arxiv.org/html/2403.10506v2/extracted/5676521/fig/sensors.jpeg)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Pipeline / core system figure: Example egocentric visual (top-left) and whole-body tactile (right) observations when”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 2: e) truck

![Figure 2](https://arxiv.org/html/2403.10506v2/extracted/5676521/fig/tasks/16_h1hand-truck_unload-v0-init.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“e) truck”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 3: a) Hierarchical learning pipeline

![Figure 3](https://arxiv.org/html/2403.10506v2/extracted/5676521/fig/pipeline.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“a) Hierarchical learning pipeline”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### 论文表格

#### Table 1: TABLE I: Comparison of simulated robot benchmarks. Our humanoid robot benchmark tests a variety of complex, long-horizo

| Benchmark | Dexterous hands | Action dim. | DoF | Task horizon | # Tasks | Skills 1 1 footnotemark: 1 |
| --- | --- | --- | --- | --- | --- | --- |
| MyoHand [8 ] | \checkmark | 39 39 39 39 | 23 23 23 23 D | 50 50 50 50 - 2000 2000 2000 2000 | 9 9 9 9 | PnP, R, Po, IR, H, Ro |
| Adroit [49 ] | \checkmark | 24 24 24 24 | 24 24 24 24 D | 200 200 200 200 | 4 4 4 4 | PnP, P, R, Po, IR, H, L, Ro |
| MyoLeg [8 ] | \times | 80 80 80 80 | 20 20 20 20 D | 1000 1000 1000 1000 | 1 1 1 1 | Lo, St |
| LocoMujoco [3 ] (Unitree-H1) | \times | 19 19 19 19 | 6 6 6 6 D | 100 100 100 100 - 500 500 500 500 | 27 27 27 27 | L, Lo, St, BM |
| DMControl [58 ] (Humanoid) | \times | 24 24 24 24 - 56 56 56 56 | 22 22 22 22 D | 1000 1000 1000 1000 | 6 6 6 6 | Lo, St |
| FurnitureSim [22 ] | \times | 8 8 8 8 | 6 6 6 6 D | 2300 2300 2300 2300 | 8 8 8 8 | PnP, P, I, IR, H, L, Ro |
| robosuite [70 ] | \times | 6 6 6 6 - 24 24 24 24 | 6 6 6 6 - 7 7 7 7 D | 500 500 500 500 | 9 9 9 9 | PnP, P, I, R, IR, H, L, Ro |
| rlbench [24 ] | \times | 6 6 6 6 - 7 7 7 7 | 6 6 6 6 - 7 7 7 7 D | 100 100 100 100 - 1000 1000 1000 1000 | 106 106 106 106 | PnP, P, I, R, Po, IR, H, L, Ro |
| metaworld [64 ] | \times | 6 6 6 6 | 7 7 7 7 D | 500 500 500 500 | 50 50 50 50 | PnP, P, I, R, Po, IR, H, L, Ro |
| HumanoidBench (Ours) | \checkmark | 61 61 61 61 | 75 75 75 75 D | 500 500 500 500 - 1000 1000 1000 1000 | 27 27 27 27 | PnP, P, I, R, Po, IR, H, L, Ro, Lo, BM, St |

**说明**: TABLE I: Comparison of simulated robot benchmarks. Our humanoid robot benchmark tests a variety of complex, long-horizon task with a large action space.

#### Table 2: TABLE II: Humanoid robot specifications with and without hands. Both the humanoid body (including its floating base) an

| | Without hand | With 2 2 2 2 hands |
| --- | --- | --- |
| Observation space | 51 51 51 51 | 151 151 151 151 |
| Action space | 19 19 19 19 | 61 61 61 61 |
| DoF (body) | 25 25 25 25 | 25 25 25 25 |
| DoF (two hands) | 0 0 | 50 50 50 50 |

**说明**: TABLE II: Humanoid robot specifications with and without hands. Both the humanoid body (including its floating base) and one Shadow Hand present action spaces (19 19 19 19 and 21 21 21 21, respectively) smaller than their DoFs (25 25 25 25), making them systems. In this table, the observation spaces solely comprise generalized positions and velocities of the robots and do not take into account any environment observations. We use quaternions for the robot floating base orientation, which adds an additional position coordinate compared to the velocity components, which match the DoFs. In the appendix, Table III shows an exhaustive of all the robot configurations available in HumanoidBench.

#### Table 3: TABLE III: All supported robot specifications. Note that the observation space in this table does not take into accoun

| | H1 w/o hands | H1 w/ ShadowHand | H1 w/ Robotiq gripper | H1 w/ Unitree hand | Digit w/ ShadowHand | Unitree G1 |
| --- | --- | --- | --- | --- | --- | --- |
| Observation space | 51 51 51 51 | 151 151 151 151 | 55 55 55 55 | 103 103 103 103 | 221 221 221 221 | 87 87 87 87 |
| Action space | 19 19 19 19 | 61 61 61 61 | 23 23 23 23 | 45 45 45 45 | 65 65 65 65 | 37 37 37 37 |
| DoF (body) | 25 25 25 25 | 25 25 25 25 | 25 25 25 25 | 25 25 25 25 | 57 57 57 57 | 29 29 29 29 |
| DoF (2 2 2 2 end-effectors) | 0 0 | 50 50 50 50 | 4 4 4 4 | 26 26 26 26 | 50 50 50 50 | 14 14 14 14 |

**说明**: TABLE III: All supported robot specifications. Note that the observation space in this table does not take into account any observations of the surrounding environment and solely comprises generalized positions and velocities. We use quaternions for the robot floating base orientation, as well as for ball joints, which add additional position coordinates compared to the velocity components (which match the DoFs).

#### Table 4: TABLE IV: HumanoidBench Simulation Performance.

| Configuration | FPS |
| --- | --- |
| Without hands | 2450 2450 2450 2450 |
| Simplified body collisions | 3600 3600 3600 3600 |
| Collisions only for feet | 5100 5100 5100 5100 |
| Default | 1050 1050 1050 1050 |

**说明**: TABLE IV: HumanoidBench Simulation Performance.

#### Table 5: TABLE V: Average returns for HumanoidBench. Each number represents average return@10M (return@2M) with the standard dev

| | DreamerV3 | TD-MPC2 | SAC | Target |
| --- | --- | --- | --- | --- |
| walk | 800.2 ± 158.7 plus-or-minus 800.2 158.7 800.2\pm 158.7 800.2 ± 158.7 | 782.0 ± 109.2 plus-or-minus 782.0 109.2 782.0\pm 109.2 782.0 ± 109.2 | 31.7 ± 24.0 plus-or-minus 31.7 24.0 31.7\pm 24.0 31.7 ± 24.0 | 700.0 700.0 700.0 700.0 |
| stand | 622.7 ± 404.8 plus-or-minus 622.7 404.8 622.7\pm 404.8 622.7 ± 404.8 | 809.0 ± 137.1 plus-or-minus 809.0 137.1 809.0\pm 137.1 809.0 ± 137.1 | 208.3 ± 105.6 plus-or-minus 208.3 105.6 208.3\pm 105.6 208.3 ± 105.6 | 800.0 800.0 800.0 800.0 |
| run | 633.8 ± 222.4 plus-or-minus 633.8 222.4 633.8\pm 222.4 633.8 ± 222.4 | 93.3 ± 14.3 plus-or-minus 93.3 14.3 93.3\pm 14.3 93.3 ± 14.3 | 5.0 ± 2.1 plus-or-minus 5.0 2.1 5.0\pm 2.1 5.0 ± 2.1 | 700.0 700.0 700.0 700.0 |
| reach | 7580.9 ± 1951.0 plus-or-minus 7580.9 1951.0 7580.9\pm 1951.0 7580.9 ± 1951.0 | 7316.1 ± 2112.1 plus-or-minus 7316.1 2112.1 7316.1\pm 2112.1 7316.1 ± 2112.1 | 4565.1 ± 212.8 plus-or-minus 4565.1 212.8 4565.1\pm 212.8 4565.1 ± 212.8 | 12000.0 12000.0 12000.0 12000.0 |
| hurdle | 126.2 ± 59.4 plus-or-minus 126.2 59.4 126.2\pm 59.4 126.2 ± 59.4 | 46.4 ± 10.8 plus-or-minus 46.4 10.8 46.4\pm 10.8 46.4 ± 10.8 | 13.2 ± 8.8 plus-or-minus 13.2 8.8 13.2\pm 8.8 13.2 ± 8.8 | 700.0 700.0 700.0 700.0 |
| crawl | 878.8 ± 122.7 plus-or-minus 878.8 122.7 878.8\pm 122.7 878.8 ± 122.7 | 957.4 ± 17.5 plus-or-minus 957.4 17.5 957.4\pm 17.5 957.4 ± 17.5 | 330.0 ± 111.9 plus-or-minus 330.0 111.9 330.0\pm 111.9 330.0 ± 111.9 | 700.0 700.0 700.0 700.0 |
| maze | 272.3 ± 116.6 plus-or-minus 272.3 116.6 272.3\pm 116.6 272.3 ± 116.6 | 244.3 ± 97.7 plus-or-minus 244.3 97.7 244.3\pm 97.7 244.3 ± 97.7 | 144.8 ± 17.8 plus-or-minus 144.8 17.8 144.8\pm 17.8 144.8 ± 17.8 | 1200.0 1200.0 1200.0 1200.0 |
| sit_simple | 891.4 ± 38.4 plus-or-minus 891.4 38.4 891.4\pm 38.4 891.4 ± 38.4 | 411.1 ± 368.0 plus-or-minus 411.1 368.0 411.1\pm 368.0 411.1 ± 368.0 | 148.3 ± 103.8 plus-or-minus 148.3 103.8 148.3\pm 103.8 148.3 ± 103.8 | 750.0 750.0 750.0 750.0 |
| sit_hard | 433.4 ± 355.9 plus-or-minus 433.4 355.9 433.4\pm 355.9 433.4 ± 355.9 | 343.0 ± 381.7 plus-or-minus 343.0 381.7 343.0\pm 381.7 343.0 ± 381.7 | 55.0 ± 18.2 plus-or-minus 55.0 18.2 55.0\pm 18.2 55.0 ± 18.2 | 750.0 750.0 750.0 750.0 |
| balance_simple | 19.8 ± 7.0 plus-or-minus 19.8 7.0 19.8\pm 7.0 19.8 ± 7.0 | 40.5 ± 23.9 plus-or-minus 40.5 23.9 40.5\pm 23.9 40.5 ± 23.9 | 61.5 ± 1.1 plus-or-minus 61.5 1.1 61.5\pm 1.1 61.5 ± 1.1 | 800.0 800.0 800.0 800.0 |
| balance_hard | 45.9 ± 27.4 plus-or-minus 45.9 27.4 45.9\pm 27.4 45.9 ± 27.4 | 48.2 ± 28.5 plus-or-minus 48.2 28.5 48.2\pm 28.5 48.2 ± 28.5 | 42.5 ± 22.6 plus-or-minus 42.5 22.6 42.5\pm 22.6 42.5 ± 22.6 | 800.0 800.0 800.0 800.0 |
| stair | 131.1 ± 43.6 plus-or-minus 131.1 43.6 131.1\pm 43.6 131.1 ± 43.6 | 70.4 ± 7.1 plus-or-minus 70.4 7.1 70.4\pm 7.1 70.4 ± 7.1 | 14.1 ± 6.8 plus-or-minus 14.1 6.8 14.1\pm 6.8 14.1 ± 6.8 | 700.0 700.0 700.0 700.0 |
| slide | 436.5 ± 200.1 plus-or-minus 436.5 200.1 436.5\pm 200.1 436.5 ± 200.1 | 119.0 ± 35.9 plus-or-minus 119.0 35.9 119.0\pm 35.9 119.0 ± 35.9 | 6.3 ± 2.8 plus-or-minus 6.3 2.8 6.3\pm 2.8 6.3 ± 2.8 | 700.0 700.0 700.0 700.0 |
| pole | 658.3 ± 343.3 plus-or-minus 658.3 343.3 658.3\pm 343.3 658.3 ± 343.3 | 226.3 ± 116.1 plus-or-minus 226.3 116.1 226.3\pm 116.1 226.3 ± 116.1 | 46.3 ± 26.4 plus-or-minus 46.3 26.4 46.3\pm 26.4 46.3 ± 26.4 | 700.0 700.0 700.0 700.0 |
| push | - 1251.9 ± 659.8 plus-or-minus 1251.9 659.8 -1251.9\pm 659.8 - 1251.9 ± 659.8 | - 258.7 ± 66.5 plus-or-minus 258.7 66.5 -258.7\pm 66.5 - 258.7 ± 66.5 | - 97.9 ± 147.0 plus-or-minus 97.9 147.0 -97.9\pm 147.0 - 97.9 ± 147.0 | 700.0 700.0 700.0 700.0 |
| cabinet | 57.3 ± 66.3 plus-or-minus 57.3 66.3 57.3\pm 66.3 57.3 ± 66.3 | 112.8 ± 142.9 plus-or-minus 112.8 142.9 112.8\pm 142.9 112.8 ± 142.9 | 211.8 ± 33.8 plus-or-minus 211.8 33.8 211.8\pm 33.8 211.8 ± 33.8 | 2500.0 2500.0 2500.0 2500.0 |
| highbar | 8.9 ± 5.8 plus-or-minus 8.9 5.8 8.9\pm 5.8 8.9 ± 5.8 | 0.3 ± 0.0 plus-or-minus 0.3 0.0 0.3\pm 0.0 0.3 ± 0.0 | 9.4 ± 3.7 plus-or-minus 9.4 3.7 9.4\pm 3.7 9.4 ± 3.7 | 750.0 750.0 750.0 750.0 |
| door | 213.0 ± 149.3 plus-or-minus 213.0 149.3 213.0\pm 149.3 213.0 ± 149.3 | 274.7 ± 12.5 plus-or-minus 274.7 12.5 274.7\pm 12.5 274.7 ± 12.5 | 39.4 ± 25.2 plus-or-minus 39.4 25.2 39.4\pm 25.2 39.4 ± 25.2 | 600.0 600.0 600.0 600.0 |
| truck | 1103.8 ± 232.9 plus-or-minus 1103.8 232.9 1103.8\pm 232.9 1103.8 ± 232.9 | 1132.6 ± 72.1 plus-or-minus 1132.6 72.1 1132.6\pm 72.1 1132.6 ± 72.1 | 1077.5 ± 95.0 plus-or-minus 1077.5 95.0 1077.5\pm 95.0 1077.5 ± 95.0 | 3000.0 3000.0 3000.0 3000.0 |
| cube | 111.2 ± 59.9 plus-or-minus 111.2 59.9 111.2\pm 59.9 111.2 ± 59.9 | 54.7 ± 33.1 plus-or-minus 54.7 33.1 54.7\pm 33.1 54.7 ± 33.1 | 130.7 ± 30.5 plus-or-minus 130.7 30.5 130.7\pm 30.5 130.7 ± 30.5 | 370.0 370.0 370.0 370.0 |
| bookshelf_simple | 840.4 ± 5.6 plus-or-minus 840.4 5.6 840.4\pm 5.6 840.4 ± 5.6 | 136.2 ± 71.6 plus-or-minus 136.2 71.6 136.2\pm 71.6 136.2 ± 71.6 | 346.9 ± 231.5 plus-or-minus 346.9 231.5 346.9\pm 231.5 346.9 ± 231.5 | 2000.0 2000.0 2000.0 2000.0 |
| bookshelf_hard | 530.2 ± 302.5 plus-or-minus 530.2 302.5 530.2\pm 302.5 530.2 ± 302.5 | 37.0 ± 1.3 plus-or-minus 37.0 1.3 37.0\pm 1.3 37.0 ± 1.3 | 293.9 ± 121.6 plus-or-minus 293.9 121.6 293.9\pm 121.6 293.9 ± 121.6 | 2000.0 2000.0 2000.0 2000.0 |
| basketball | 19.3 ± 2.5 plus-or-minus 19.3 2.5 19.3\pm 2.5 19.3 ± 2.5 | 42.0 ± 14.8 plus-or-minus 42.0 14.8 42.0\pm 14.8 42.0 ± 14.8 | 22.1 ± 3.2 plus-or-minus 22.1 3.2 22.1\pm 3.2 22.1 ± 3.2 | 1200.0 1200.0 1200.0 1200.0 |
| window | 461.0 ± 252.8 plus-or-minus 461.0 252.8 461.0\pm 252.8 461.0 ± 252.8 | 87.1 ± 37.5 plus-or-minus 87.1 37.5 87.1\pm 37.5 87.1 ± 37.5 | 62.9 ± 83.8 plus-or-minus 62.9 83.8 62.9\pm 83.8 62.9 ± 83.8 | 650.0 650.0 650.0 650.0 |
| spoon | 349.7 ± 46.2 plus-or-minus 349.7 46.2 349.7\pm 46.2 349.7 ± 46.2 | 77.9 ± 80.6 plus-or-minus 77.9 80.6 77.9\pm 80.6 77.9 ± 80.6 | 87.7 ± 80.5 plus-or-minus 87.7 80.5 87.7\pm 80.5 87.7 ± 80.5 | 650.0 650.0 650.0 650.0 |
| kitchen | 0.0 ± 0.0 plus-or-minus 0.0 0.0 0.0\pm 0.0 0.0 ± 0.0 | 0.0 ± 0.0 plus-or-minus 0.0 0.0 0.0\pm 0.0 0.0 ± 0.0 | 0.0 ± 0.0 plus-or-minus 0.0 0.0 0.0\pm 0.0 0.0 ± 0.0 | 4.0 4.0 4.0 4.0 |
| package | - 18015.2 ± 9477.7 plus-or-minus 18015.2 9477.7 -18015.2\pm 9477.7 - 18015.2 ± 9477.7 | - 3655.6 ± 1055.0 plus-or-minus 3655.6 1055.0 -3655.6\pm 1055.0 - 3655.6 ± 1055.0 | - 6718.3 ± 607.0 plus-or-minus 6718.3 607.0 -6718.3\pm 607.0 - 6718.3 ± 607.0 | 1500.0 1500.0 1500.0 1500.0 |
| powerlift | 315.9 ± 16.9 plus-or-minus 315.9 16.9 315.9\pm 16.9 315.9 ± 16.9 | 99.1 ± 47.3 plus-or-minus 99.1 47.3 99.1\pm 47.3 99.1 ± 47.3 | 81.8 ± 46.7 plus-or-minus 81.8 46.7 81.8\pm 46.7 81.8 ± 46.7 | 800.0 800.0 800.0 800.0 |
| room | 120.5 ± 71.4 plus-or-minus 120.5 71.4 120.5\pm 71.4 120.5 ± 71.4 | 131.4 ± 56.7 plus-or-minus 131.4 56.7 131.4\pm 56.7 131.4 ± 56.7 | 12.0 ± 4.9 plus-or-minus 12.0 4.9 12.0\pm 4.9 12.0 ± 4.9 | 400.0 400.0 400.0 400.0 |
| insert_small | 184.8 ± 26.3 plus-or-minus 184.8 26.3 184.8\pm 26.3 184.8 ± 26.3 | 129.8 ± 51.9 plus-or-minus 129.8 51.9 129.8\pm 51.9 129.8 ± 51.9 | 10.8 ± 13.4 plus-or-minus 10.8 13.4 10.8\pm 13.4 10.8 ± 13.4 | 350.0 350.0 350.0 350.0 |
| insert_normal | 171.5 ± 33.2 plus-or-minus 171.5 33.2 171.5\pm 33.2 171.5 ± 33.2 | 237.6 ± 9.1 plus-or-minus 237.6 9.1 237.6\pm 9.1 237.6 ± 9.1 | 46.3 ± 63.1 plus-or-minus 46.3 63.1 46.3\pm 63.1 46.3 ± 63.1 | 350.0 350.0 350.0 350.0 |

**说明**: TABLE V: Average returns for HumanoidBench. Each number represents average return@10M (return@2M) with the standard deviation for DreamerV3 and SAC (TD-MPC2).

#### Table 6: TABLE VI: Maximum returns for HumanoidBench. Each number represents maximum return@10M (return@2M) with the standard de

| | DreamerV3 | TD-MPC2 | SAC | Target |
| --- | --- | --- | --- | --- |
| walk | 932.4 ± 0.3 plus-or-minus 932.4 0.3 932.4\pm 0.3 932.4 ± 0.3 | 900.3 ± 47.6 plus-or-minus 900.3 47.6 900.3\pm 47.6 900.3 ± 47.6 | 68.7 ± 27.0 plus-or-minus 68.7 27.0 68.7\pm 27.0 68.7 ± 27.0 | 700.0 700.0 700.0 700.0 |
| stand | 932.9 ± 1.1 plus-or-minus 932.9 1.1 932.9\pm 1.1 932.9 ± 1.1 | 925.7 ± 2.5 plus-or-minus 925.7 2.5 925.7\pm 2.5 925.7 ± 2.5 | 809.9 ± 194.5 plus-or-minus 809.9 194.5 809.9\pm 194.5 809.9 ± 194.5 | 800.0 800.0 800.0 800.0 |
| run | 895.9 ± 6.0 plus-or-minus 895.9 6.0 895.9\pm 6.0 895.9 ± 6.0 | 226.7 ± 23.3 plus-or-minus 226.7 23.3 226.7\pm 23.3 226.7 ± 23.3 | 104.8 ± 7.4 plus-or-minus 104.8 7.4 104.8\pm 7.4 104.8 ± 7.4 | 700.0 700.0 700.0 700.0 |
| reach | 9831.6 ± 115.9 plus-or-minus 9831.6 115.9 9831.6\pm 115.9 9831.6 ± 115.9 | 9727.6 ± 48.9 plus-or-minus 9727.6 48.9 9727.6\pm 48.9 9727.6 ± 48.9 | 7169.9 ± 874.1 plus-or-minus 7169.9 874.1 7169.9\pm 874.1 7169.9 ± 874.1 | 12000.0 12000.0 12000.0 12000.0 |
| hurdle | 396.8 ± 39.7 plus-or-minus 396.8 39.7 396.8\pm 39.7 396.8 ± 39.7 | 196.7 ± 30.5 plus-or-minus 196.7 30.5 196.7\pm 30.5 196.7 ± 30.5 | 78.6 ± 32.8 plus-or-minus 78.6 32.8 78.6\pm 32.8 78.6 ± 32.8 | 700.0 700.0 700.0 700.0 |
| crawl | 985.3 ± 0.6 plus-or-minus 985.3 0.6 985.3\pm 0.6 985.3 ± 0.6 | 985.2 ± 0.4 plus-or-minus 985.2 0.4 985.2\pm 0.4 985.2 ± 0.4 | 626.5 ± 29.1 plus-or-minus 626.5 29.1 626.5\pm 29.1 626.5 ± 29.1 | 700.0 700.0 700.0 700.0 |
| maze | 592.5 ± 49.0 plus-or-minus 592.5 49.0 592.5\pm 49.0 592.5 ± 49.0 | 444.9 ± 22.1 plus-or-minus 444.9 22.1 444.9\pm 22.1 444.9 ± 22.1 | 269.9 ± 37.8 plus-or-minus 269.9 37.8 269.9\pm 37.8 269.9 ± 37.8 | 1200.0 1200.0 1200.0 1200.0 |
| sit_simple | 935.7 ± 5.5 plus-or-minus 935.7 5.5 935.7\pm 5.5 935.7 ± 5.5 | 928.4 ± 1.5 plus-or-minus 928.4 1.5 928.4\pm 1.5 928.4 ± 1.5 | 842.7 ± 50.8 plus-or-minus 842.7 50.8 842.7\pm 50.8 842.7 ± 50.8 | 750.0 750.0 750.0 750.0 |
| sit_hard | 914.6 ± 1.5 plus-or-minus 914.6 1.5 914.6\pm 1.5 914.6 ± 1.5 | 906.3 ± 6.0 plus-or-minus 906.3 6.0 906.3\pm 6.0 906.3 ± 6.0 | 214.0 ± 47.9 plus-or-minus 214.0 47.9 214.0\pm 47.9 214.0 ± 47.9 | 750.0 750.0 750.0 750.0 |
| balance_simple | 95.4 ± 8.3 plus-or-minus 95.4 8.3 95.4\pm 8.3 95.4 ± 8.3 | 95.3 ± 3.1 plus-or-minus 95.3 3.1 95.3\pm 3.1 95.3 ± 3.1 | 80.7 ± 3.7 plus-or-minus 80.7 3.7 80.7\pm 3.7 80.7 ± 3.7 | 800.0 800.0 800.0 800.0 |
| balance_hard | 114.0 ± 12.3 plus-or-minus 114.0 12.3 114.0\pm 12.3 114.0 ± 12.3 | 122.2 ± 13.1 plus-or-minus 122.2 13.1 122.2\pm 13.1 122.2 ± 13.1 | 71.0 ± 7.9 plus-or-minus 71.0 7.9 71.0\pm 7.9 71.0 ± 7.9 | 800.0 800.0 800.0 800.0 |
| stair | 411.4 ± 9.7 plus-or-minus 411.4 9.7 411.4\pm 9.7 411.4 ± 9.7 | 251.9 ± 9.1 plus-or-minus 251.9 9.1 251.9\pm 9.1 251.9 ± 9.1 | 42.8 ± 0.7 plus-or-minus 42.8 0.7 42.8\pm 0.7 42.8 ± 0.7 | 700.0 700.0 700.0 700.0 |
| slide | 928.4 ± 2.0 plus-or-minus 928.4 2.0 928.4\pm 2.0 928.4 ± 2.0 | 311.9 ± 15.1 plus-or-minus 311.9 15.1 311.9\pm 15.1 311.9 ± 15.1 | 41.4 ± 2.4 plus-or-minus 41.4 2.4 41.4\pm 2.4 41.4 ± 2.4 | 700.0 700.0 700.0 700.0 |
| pole | 952.2 ± 10.3 plus-or-minus 952.2 10.3 952.2\pm 10.3 952.2 ± 10.3 | 644.9 ± 21.2 plus-or-minus 644.9 21.2 644.9\pm 21.2 644.9 ± 21.2 | 440.0 ± 88.4 plus-or-minus 440.0 88.4 440.0\pm 88.4 440.0 ± 88.4 | 700.0 700.0 700.0 700.0 |
| push | 1000.0 ± 0.0 plus-or-minus 1000.0 0.0 1000.0\pm 0.0 1000.0 ± 0.0 | 1000.0 ± 0.0 plus-or-minus 1000.0 0.0 1000.0\pm 0.0 1000.0 ± 0.0 | 352.8 ± 31.5 plus-or-minus 352.8 31.5 352.8\pm 31.5 352.8 ± 31.5 | 700.0 700.0 700.0 700.0 |
| cabinet | 722.6 ± 7.3 plus-or-minus 722.6 7.3 722.6\pm 7.3 722.6 ± 7.3 | 721.6 ± 25.9 plus-or-minus 721.6 25.9 721.6\pm 25.9 721.6 ± 25.9 | 485.9 ± 137.2 plus-or-minus 485.9 137.2 485.9\pm 137.2 485.9 ± 137.2 | 2500.0 2500.0 2500.0 2500.0 |
| highbar | 83.1 ± 4.6 plus-or-minus 83.1 4.6 83.1\pm 4.6 83.1 ± 4.6 | 0.9 ± 0.4 plus-or-minus 0.9 0.4 0.9\pm 0.4 0.9 ± 0.4 | 40.8 ± 41.7 plus-or-minus 40.8 41.7 40.8\pm 41.7 40.8 ± 41.7 | 750.0 750.0 750.0 750.0 |
| door | 335.7 ± 14.8 plus-or-minus 335.7 14.8 335.7\pm 14.8 335.7 ± 14.8 | 310.6 ± 10.6 plus-or-minus 310.6 10.6 310.6\pm 10.6 310.6 ± 10.6 | 251.2 ± 9.0 plus-or-minus 251.2 9.0 251.2\pm 9.0 251.2 ± 9.0 | 600.0 600.0 600.0 600.0 |
| truck | 1674.3 ± 52.6 plus-or-minus 1674.3 52.6 1674.3\pm 52.6 1674.3 ± 52.6 | 1457.2 ± 24.3 plus-or-minus 1457.2 24.3 1457.2\pm 24.3 1457.2 ± 24.3 | 1387.5 ± 10.0 plus-or-minus 1387.5 10.0 1387.5\pm 10.0 1387.5 ± 10.0 | 3000.0 3000.0 3000.0 3000.0 |
| cube | 237.9 ± 3.4 plus-or-minus 237.9 3.4 237.9\pm 3.4 237.9 ± 3.4 | 241.1 ± 1.2 plus-or-minus 241.1 1.2 241.1\pm 1.2 241.1 ± 1.2 | 203.5 ± 27.2 plus-or-minus 203.5 27.2 203.5\pm 27.2 203.5 ± 27.2 | 370.0 370.0 370.0 370.0 |
| bookshelf_simple | 849.6 ± 0.3 plus-or-minus 849.6 0.3 849.6\pm 0.3 849.6 ± 0.3 | 825.0 ± 9.6 plus-or-minus 825.0 9.6 825.0\pm 9.6 825.0 ± 9.6 | 766.5 ± 10.3 plus-or-minus 766.5 10.3 766.5\pm 10.3 766.5 ± 10.3 | 2000.0 2000.0 2000.0 2000.0 |
| bookshelf_hard | 867.8 ± 8.4 plus-or-minus 867.8 8.4 867.8\pm 8.4 867.8 ± 8.4 | 320.4 ± 58.9 plus-or-minus 320.4 58.9 320.4\pm 58.9 320.4 ± 58.9 | 681.5 ± 10.3 plus-or-minus 681.5 10.3 681.5\pm 10.3 681.5 ± 10.3 | 2000.0 2000.0 2000.0 2000.0 |
| basketball | 808.8 ± 340.5 plus-or-minus 808.8 340.5 808.8\pm 340.5 808.8 ± 340.5 | 1055.3 ± 4.1 plus-or-minus 1055.3 4.1 1055.3\pm 4.1 1055.3 ± 4.1 | 192.3 ± 45.6 plus-or-minus 192.3 45.6 192.3\pm 45.6 192.3 ± 45.6 | 1200.0 1200.0 1200.0 1200.0 |
| window | 765.6 ± 38.4 plus-or-minus 765.6 38.4 765.6\pm 38.4 765.6 ± 38.4 | 201.1 ± 91.5 plus-or-minus 201.1 91.5 201.1\pm 91.5 201.1 ± 91.5 | 128.6 ± 170.8 plus-or-minus 128.6 170.8 128.6\pm 170.8 128.6 ± 170.8 | 650.0 650.0 650.0 650.0 |
| spoon | 421.5 ± 2.5 plus-or-minus 421.5 2.5 421.5\pm 2.5 421.5 ± 2.5 | 403.5 ± 3.2 plus-or-minus 403.5 3.2 403.5\pm 3.2 403.5 ± 3.2 | 297.5 ± 26.5 plus-or-minus 297.5 26.5 297.5\pm 26.5 297.5 ± 26.5 | 650.0 650.0 650.0 650.0 |
| kitchen | 0.0 ± 0.0 plus-or-minus 0.0 0.0 0.0\pm 0.0 0.0 ± 0.0 | 0.0 ± 0.0 plus-or-minus 0.0 0.0 0.0\pm 0.0 0.0 ± 0.0 | 0.0 ± 0.0 plus-or-minus 0.0 0.0 0.0\pm 0.0 0.0 ± 0.0 | 4.0 4.0 4.0 4.0 |
| package | 1009.2 ± 4.1 plus-or-minus 1009.2 4.1 1009.2\pm 4.1 1009.2 ± 4.1 | 1003.3 ± 3.4 plus-or-minus 1003.3 3.4 1003.3\pm 3.4 1003.3 ± 3.4 | - 3552.8 ± 361.3 plus-or-minus 3552.8 361.3 -3552.8\pm 361.3 - 3552.8 ± 361.3 | 1500.0 1500.0 1500.0 1500.0 |
| powerlift | 338.6 ± 0.1 plus-or-minus 338.6 0.1 338.6\pm 0.1 338.6 ± 0.1 | 264.7 ± 23.9 plus-or-minus 264.7 23.9 264.7\pm 23.9 264.7 ± 23.9 | 171.2 ± 3.6 plus-or-minus 171.2 3.6 171.2\pm 3.6 171.2 ± 3.6 | 800.0 800.0 800.0 800.0 |
| room | 420.8 ± 51.1 plus-or-minus 420.8 51.1 420.8\pm 51.1 420.8 ± 51.1 | 353.3 ± 41.5 plus-or-minus 353.3 41.5 353.3\pm 41.5 353.3 ± 41.5 | 52.2 ± 3.9 plus-or-minus 52.2 3.9 52.2\pm 3.9 52.2 ± 3.9 | 400.0 400.0 400.0 400.0 |
| insert_small | 239.8 ± 4.6 plus-or-minus 239.8 4.6 239.8\pm 4.6 239.8 ± 4.6 | 226.4 ± 10.8 plus-or-minus 226.4 10.8 226.4\pm 10.8 226.4 ± 10.8 | 72.7 ± 21.9 plus-or-minus 72.7 21.9 72.7\pm 21.9 72.7 ± 21.9 | 350.0 350.0 350.0 350.0 |
| insert_normal | 279.9 ± 9.9 plus-or-minus 279.9 9.9 279.9\pm 9.9 279.9 ± 9.9 | 273.0 ± 5.7 plus-or-minus 273.0 5.7 273.0\pm 5.7 273.0 ± 5.7 | 135.3 ± 51.5 plus-or-minus 135.3 51.5 135.3\pm 51.5 135.3 ± 51.5 | 350.0 350.0 350.0 350.0 |

**说明**: TABLE VI: Maximum returns for HumanoidBench. Each number represents maximum return@10M (return@2M) with the standard deviation for DreamerV3 and SAC (TD-MPC2).
## 实验解读

- 评价重点:围绕 adaptive-control、agile-locomotion、benchmark-dataset、接触推理、灵巧操作,优先看真实机器人结果、扰动恢复、接触稳定性、控制误差、成功率、实时性和 sim-to-real 差距。
- System0 视角:实验结论不只看最终成功率,还要看闭环过程中是否能处理状态估计误差、执行延迟、碰撞/接触突变和外部扰动。
- 与上层系统的关系:如果方法能在快闭环内稳定工作,它可以作为 System1 技能或 System2 规划的底层执行保障;如果只在离线或慢速设定下成立,则还不能直接作为身体级快闭环模块。
## 批判性思考

### 优点

- 论文把 adaptive-control、agile-locomotion、benchmark-dataset、接触推理、灵巧操作 放到机器人身体级闭环里讨论,和只做高层语义推理的工作相比,更接近真实机器人部署时会遇到的状态、动力学和安全约束。
- 若论文同时给出公式、系统结构图、硬件实验或消融结果,这些证据可以帮助判断方法是否真的能支撑毫秒到百毫秒级反应。

### 局限

- 需要重点核查控制频率、推理延迟、传感延迟、失败恢复、硬件磨损、未见场景泛化和安全边界;这些因素决定它能否成为可靠的 System0 模块。
- 对学习型方法,还要区分训练环境中的鲁棒性和真实部署中的鲁棒性,尤其是接触丰富任务、动态运动任务和人形机器人任务。
## 论文定位

- 研究对象:HumanoidBench: Simulated Humanoid Benchmark for Whole-Body Locomotion and Manipulation。
- 关键词:adaptive-control、agile-locomotion、benchmark-dataset、接触推理、灵巧操作、人形机器人、足式运动、移动操作、强化学习、机器人操作。
- System0 关联:关注机器人身体级快反应链路中“状态/接触/局部感知 -> 约束/优化/策略 -> 动作”的短闭环,不把问题停留在语言规划或任务分解层。

## 速查卡片

> [!summary] HumanoidBench
> - **论文**: https://www.roboticsproceedings.org/rss20/p061.pdf
> - **arXiv**: http://arxiv.org/abs/2403.10506v2
> - **arXiv HTML**: https://arxiv.org/html/2403.10506v2
