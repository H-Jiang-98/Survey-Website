---
title: "Conformalized Teleoperation: Confidently Mapping Human Inputs to High-Dimensional Robot Actions"
method_name: "Conformalized Teleop"
authors: ["Michelle D Zhao"]
year: 2024
venue: "RSS"
tags: ["safe-control", "closed-loop-control", "robot-manipulation", "imitation-learning"]
image_source: "online"
arxiv_html: "https://arxiv.org/html/2406.07767"
---
# Conformalized Teleop

## 一句话总结

这篇论文把低维人类输入到高维机器人动作的映射做成带置信区间的在线执行接口。它不是高层任务规划，而是在 teleoperation 闭环里判断当前状态和用户输入是否足够可信，从而决定机器人动作是否可继续执行。

## 为什么属于 System0

1. 输入是 joystick 等低维人类控制信号和机器人状态，输出是高维机器人动作。
2. adaptive conformal prediction 给动作预测加上时间变化的置信区间，用来识别高不确定状态。
3. 在 Kinova Jaco cup grasping 和 reaching 等任务中，这个机制相当于 System0 的安全感知层，负责在执行时发现动作映射不可靠的区域。

## 问题背景

这篇论文关注的不是语言理解或长期任务规划，而是机器人执行层如何把局部感知、人体输入、接触状态、运动约束或硬件映射转成可执行动作。按照我们对 System0 的定义，它落在“身体级快反应”这一层：上层可以给目标或意图，但真正稳定、安全、动态可行的动作生成发生在这里。

## 方法详解

### Conformalized assistive map

基础 assistive map 学习从低维输入到高维动作的映射；论文额外让模型输出动作分位数，并用 conformal calibration 调整覆盖率。

### Adaptive uncertainty gating

如果近期预测误差增大，区间会变宽；如果映射稳定，区间会变窄。区间宽度可作为执行时的风险信号。

### System0 位置

它位于人类意图输入和低层机器人动作之间，比任务规划更快，比电机控制更抽象，适合作为 teleoperation safety filter 或 action confidence monitor。

## 关键图表

### Figure 1: Uncertainty-aware teleoperation pipeline

![Figure 1](https://arxiv.org/html/2406.07767/x1.png)

**图意**：Uncertainty-aware teleoperation pipeline。阅读时重点看输入信号、控制或策略模块、机器人状态反馈和动作输出之间的关系。
### Figure 2: Adaptive uncertainty intervals on robot action predictions

![Figure 2](https://arxiv.org/html/2406.07767/x2.png)

**图意**：Adaptive uncertainty intervals on robot action predictions。阅读时重点看输入信号、控制或策略模块、机器人状态反馈和动作输出之间的关系。
### Figure 3: 7-DoF assistive manipulation evaluation

![Figure 3](https://arxiv.org/html/2406.07767/x3.png)

**图意**：7-DoF assistive manipulation evaluation。阅读时重点看输入信号、控制或策略模块、机器人状态反馈和动作输出之间的关系。

## 实验解读

实验覆盖 2D navigation、7-DoF cup grasping 和 goal reaching。关键结果不是单纯提高成功率，而是能把高不确定用户输入或机器人状态分出来，给下游控制器一个是否继续相信动作映射的信号。

## 局限与风险

方法目前只能检测高不确定性，不能自动区分是用户偏好变化、示教质量差还是状态分布外；如果要作为强安全模块，还需要和 fallback controller 或 shared autonomy policy 结合。

## 速查卡片

> [!summary] Conformalized Teleop
> - **arXiv**: https://arxiv.org/abs/2406.07767
> - **arXiv HTML**: https://arxiv.org/html/2406.07767
> - **论文**: http://www.roboticsproceedings.org/rss20/p008.pdf
