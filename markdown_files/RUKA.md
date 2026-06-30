---
title: "RUKA: Rethinking the Design of Humanoid Hands with Learning"
method_name: "RUKA"
authors: ["Anya Zorin"]
year: 2025
venue: "RSS"
tags: ["dexterous-manipulation", "humanoid", "robot-manipulation", "imitation-learning", "robust-control"]
image_source: "online"
---
# RUKA

## 一句话总结

RUKA 是一只低成本 tendon-driven humanoid hand，同时学习 joint-to-actuator 和 fingertip-to-actuator 映射。它的 System0 价值不在高层规划，而在手部硬件、欠驱动执行器和低层动作映射如何共同支撑灵巧操作。

## 为什么属于 System0

1. tendon-driven hand 的低层映射决定了策略动作能否真实落到手指运动和抓取力上。
2. 学习式 actuator mapping 缓解欠驱动和材料误差，属于机器人身体层的执行校准。
3. teleoperation 和 residual motor policy 展示了硬件-控制-学习闭环，而不是单纯机械设计。

## 问题背景

这篇论文关注的不是语言理解或长期任务规划，而是机器人执行层如何把局部感知、人体输入、接触状态、运动约束或硬件映射转成可执行动作。按照我们对 System0 的定义，它落在“身体级快反应”这一层：上层可以给目标或意图，但真正稳定、安全、动态可行的动作生成发生在这里。

## 方法详解

### Hand morphology

RUKA 使用 5 指、15 个欠驱动自由度，强调人手尺寸、成本和抓力之间的折中。

### Actuator mapping

用 MANUS glove 数据学习关节/指尖到 actuator 的映射，降低 tendon-driven 控制难度。

### Policy use

项目展示 teleoperation 和学习式 residual motor policy，说明硬件可以进入数据驱动控制闭环。

## 关键图表

### Figure 1: RUKA control data collection

![Figure 1](https://ruka-hand.github.io/static/images/controller_training.png)

**图意**：RUKA control data collection。阅读时重点看输入信号、控制或策略模块、机器人状态反馈和动作输出之间的关系。
### Figure 2: RUKA strength test

![Figure 2](https://ruka-hand.github.io/static/images/strength_test.png)

**图意**：RUKA strength test。阅读时重点看输入信号、控制或策略模块、机器人状态反馈和动作输出之间的关系。
### Figure 3: RUKA grasp examples

![Figure 3](https://ruka-hand.github.io/static/images/Grasps.png)

**图意**：RUKA grasp examples。阅读时重点看输入信号、控制或策略模块、机器人状态反馈和动作输出之间的关系。

## 实验解读

实验比较 reachability、durability、strength 和 grasps，并展示 teleoperation 与学习策略。对 System0 来说，重点是硬件执行误差和低层映射对动作稳定性的影响。

## 局限与风险

RUKA 更偏硬件与低层映射论文，不是完整 WBC/MPC 系统；强接触任务中的触觉闭环和安全过滤还需要单独集成。

## 速查卡片

> [!summary] RUKA
> - **项目页**: https://ruka-hand.github.io/
> - **arXiv**: https://arxiv.org/abs/2504.13165
> - **代码**: https://github.com/ruka-hand/RUKA
