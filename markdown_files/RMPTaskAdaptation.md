---
title: "Task Adaptation in Industrial Human-Robot Interaction: Leveraging Riemannian Motion Policies"
method_name: "RMP Task Adaptation"
authors: ["Mike Allenspach"]
year: 2024
venue: "RSS"
tags: ["closed-loop-control", "real-time-control", "safe-control", "robot-manipulation", "collision-avoidance"]
image_source: "online"
arxiv_html: "https://arxiv.org/html/2406.17333"
---
# RMP Task Adaptation

## 一句话总结

这篇论文用 Riemannian Motion Policies 做工业 HRI 中的任务自适应。它的核心是纯反应式 motion generation，把人类意图、任务参数和机器人运动约束合成到闭环控制里。

## 为什么属于 System0

1. RMP 是 reactive motion policy，不依赖慢速全局重规划即可产生下一步运动。
2. 人类输入影响 task feature / policy likelihood，机器人运动实时适应操作者意图。
3. 安全协作要求机器人在工业场景中持续满足运动约束和避障约束，属于身体级快反应层。

## 问题背景

这篇论文关注的不是语言理解或长期任务规划，而是机器人执行层如何把局部感知、人体输入、接触状态、运动约束或硬件映射转成可执行动作。按照我们对 System0 的定义，它落在“身体级快反应”这一层：上层可以给目标或意图，但真正稳定、安全、动态可行的动作生成发生在这里。

## 方法详解

### RMP composition

多个任务策略在 Riemannian metric 下组合，形成最终末端运动指令。

### Intent-conditioned scaling

根据人类输入历史估计任务意图，再调整各个 mission-specific RMP 的重要性。

### Reactive execution

系统强调无需人工直接控制机器人运动，而是把人类意图接入反应式 motion generation。

## 关键图表

### Figure 1: RMP task adaptation system

![Figure 1](https://arxiv.org/html/2406.17333/x1.png)

**图意**：RMP task adaptation system。阅读时重点看输入信号、控制或策略模块、机器人状态反馈和动作输出之间的关系。
### Figure 2: Robot and human intent interface

![Figure 2](https://arxiv.org/html/2406.17333/x2.png)

**图意**：Robot and human intent interface。阅读时重点看输入信号、控制或策略模块、机器人状态反馈和动作输出之间的关系。
### Figure 3: Industrial HRI evaluation

![Figure 3](https://arxiv.org/html/2406.17333/x3.png)

**图意**：Industrial HRI evaluation。阅读时重点看输入信号、控制或策略模块、机器人状态反馈和动作输出之间的关系。

## 实验解读

实验基于工业 HRI 场景，对比手动/代表性任务自适应方法。重点是任务切换时机器人是否能快速识别意图并生成顺滑、安全的运动。

## 局限与风险

方法依赖任务特征和 RMP 设计质量；复杂接触任务、视觉遮挡和多人协作场景还需要更广泛验证。

## 速查卡片

> [!summary] RMP Task Adaptation
> - **arXiv**: https://arxiv.org/abs/2406.17333
> - **arXiv HTML**: https://arxiv.org/html/2406.17333
> - **论文**: http://www.roboticsproceedings.org/rss20/p026.pdf
