---
title: "Flying Hand: End-Effector-Centric Framework for Versatile Aerial Manipulation Teleoperation and Policy Learning"
method_name: "Flying Hand"
authors: ["Guanqi He"]
year: 2025
venue: "RSS"
tags: ["whole-body-control", "model-predictive-control", "real-time-control", "robot-manipulation", "imitation-learning"]
image_source: "online"
arxiv_html: "https://arxiv.org/html/2504.10334"
---
# Flying Hand

## 一句话总结

Flying Hand 把空中操作拆成高层策略和低层 end-effector-centric whole-body MPC。它的 System0 价值在于低层控制器直接保证飞行器和机械臂协同，使空中末端执行器能稳定追踪接触任务目标。

## 为什么属于 System0

1. 低层模块是 whole-body MPC，直接输出飞行器-机械臂系统的动态可行动作。
2. 高层策略可以来自 teleoperation 或 imitation learning，但动作落地依赖低层实时跟踪和稳定控制。
3. 任务包括 writing、peg-in-hole、pick-and-place、换灯泡等，需要末端精度、姿态稳定和扰动鲁棒性。

## 问题背景

这篇论文关注的不是语言理解或长期任务规划，而是机器人执行层如何把局部感知、人体输入、接触状态、运动约束或硬件映射转成可执行动作。按照我们对 System0 的定义，它落在“身体级快反应”这一层：上层可以给目标或意图，但真正稳定、安全、动态可行的动作生成发生在这里。

## 方法详解

### End-effector-centric interface

高层只关心末端执行器轨迹，平台差异由低层控制吸收。

### Whole-body MPC

MPC 在飞行器和机械臂耦合动力学上求解控制，使末端轨迹跟踪和机体稳定同时成立。

### Teleop to policy

高精度 teleoperation 提供数据，之后可训练 imitation policy；System0 层仍承担实时稳定执行。

## 关键图表

### Figure 1: Aerial manipulation system overview

![Figure 1](https://arxiv.org/html/2504.10334/x1.png)

**图意**：Aerial manipulation system overview。阅读时重点看输入信号、控制或策略模块、机器人状态反馈和动作输出之间的关系。
### Figure 2: End-effector-centric whole-body MPC

![Figure 2](https://arxiv.org/html/2504.10334/x2.png)

**图意**：End-effector-centric whole-body MPC。阅读时重点看输入信号、控制或策略模块、机器人状态反馈和动作输出之间的关系。
### Figure 3: Real-world aerial manipulation tasks

![Figure 3](https://arxiv.org/html/2504.10334/x3.png)

**图意**：Real-world aerial manipulation tasks。阅读时重点看输入信号、控制或策略模块、机器人状态反馈和动作输出之间的关系。

## 实验解读

真实实验展示多种空中操作任务，重点指标是末端跟踪精度和任务完成能力。对 System0 来说，关键是低层 MPC 让飞行平台在操作接触附近保持可控。

## 局限与风险

框架仍依赖专门硬件和全驱动飞行器；复杂接触力控制、风扰和长时任务恢复还需要更多验证。

## 速查卡片

> [!summary] Flying Hand
> - **arXiv**: https://arxiv.org/abs/2504.10334
> - **arXiv HTML**: https://arxiv.org/html/2504.10334
> - **论文**: http://www.roboticsproceedings.org/rss21/p130.pdf
