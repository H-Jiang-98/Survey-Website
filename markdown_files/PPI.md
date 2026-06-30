---
title: "Gripper Pose and Object Pointflow as Interfaces for Robotic Bimanual Manipulation"
method_name: "PPI"
authors: ["Yuyin Yang"]
year: 2025
venue: "RSS"
tags: ["robot-manipulation", "closed-loop-control", "collision-avoidance", "dexterous-manipulation", "robot-generalization"]
image_source: "online"
arxiv_html: "https://arxiv.org/html/2504.17784"
---
# PPI

## 一句话总结

PPI 把双臂操作的中间表示设计成 gripper pose 和 object pointflow，再结合连续动作预测。它关心的是如何把空间感知变成可执行、避碰、连续的双臂动作。

## 为什么属于 System0

1. 输出不是语义计划，而是每个时间步的连续动作和目标 gripper pose。
2. object pointflow 提供局部几何运动线索，帮助控制器处理弯曲轨迹和空间约束。
3. 论文强调 collision-free trajectories 和 movement restrictions，属于 System1 技能到低层执行之间的动作生成层。

## 问题背景

这篇论文关注的不是语言理解或长期任务规划，而是机器人执行层如何把局部感知、人体输入、接触状态、运动约束或硬件映射转成可执行动作。按照我们对 System0 的定义，它落在“身体级快反应”这一层：上层可以给目标或意图，但真正稳定、安全、动态可行的动作生成发生在这里。

## 方法详解

### KeyPose interface

模型预测关键 gripper pose，为双臂操作提供明确的空间锚点。

### Object pointflow

点云/物体流描述目标物体局部运动趋势，补足单纯关键帧缺少连续监督的问题。

### Continuous action estimation

最终仍通过连续动作预测执行，使轨迹更平滑并能处理非直线运动。

## 关键图表

### Figure 1: PPI teaser: bimanual manipulation interface

![Figure 1](https://yuyinyang3y.github.io/PPI/static/images/ppi_teaser.png)

**图意**：PPI teaser: bimanual manipulation interface。阅读时重点看输入信号、控制或策略模块、机器人状态反馈和动作输出之间的关系。
### Figure 2: PPI method overview

![Figure 2](https://yuyinyang3y.github.io/PPI/static/images/ppi_method.png)

**图意**：PPI method overview。阅读时重点看输入信号、控制或策略模块、机器人状态反馈和动作输出之间的关系。
### Figure 3: Ablation and interface contribution

![Figure 3](https://yuyinyang3y.github.io/PPI/static/images/ablation.png)

**图意**：Ablation and interface contribution。阅读时重点看输入信号、控制或策略模块、机器人状态反馈和动作输出之间的关系。

## 实验解读

论文在仿真和真实双臂任务中比较 keyframe-only、continuous-only 和 PPI。PPI 的价值在于兼顾空间定位和连续控制。

## 局限与风险

PPI 更偏学习型动作接口，缺少形式化安全证明；如果用于强接触任务，还需要接触/力反馈闭环补强。

## 速查卡片

> [!summary] PPI
> - **项目页**: https://yuyinyang3y.github.io/PPI/
> - **arXiv**: https://arxiv.org/abs/2504.17784
> - **arXiv HTML**: https://arxiv.org/html/2504.17784
> - **代码**: https://github.com/OpenRobotLab/PPI
