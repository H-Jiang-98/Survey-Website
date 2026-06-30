---
title: "UMI-Underwater: Learning Underwater Manipulation without Underwater Teleoperation"
method_name: "UMI-Underwater"
authors: ["Hao Li"]
year: 2026
venue: "RSS"
tags: ["robot-manipulation", "state-estimation", "closed-loop-control", "sim-to-real", "imitation-learning"]
image_source: "online"
arxiv_html: "https://arxiv.org/html/2603.27012"
---
# UMI-Underwater

## 一句话总结

UMI-Underwater 面向水下抓取，把陆上示教、深度 affordance 和水下自监督数据结合，训练能在视觉退化和水动力扰动下输出控制动作的策略。

## 为什么属于 System0

1. 水下环境有强扰动、低可见度和接触耦合，执行层必须依赖局部感知和机器人状态快速修正。
2. 策略输入包含 depth、affordance heatmap 和机器人低维状态，输出控制动作。
3. 自监督水下数据和陆上到水下的几何对齐服务于更稳的闭环抓取，而不是高层语言规划。

## 问题背景

这篇论文关注的不是语言理解或长期任务规划，而是机器人执行层如何把局部感知、人体输入、接触状态、运动约束或硬件映射转成可执行动作。按照我们对 System0 的定义，它落在“身体级快反应”这一层：上层可以给目标或意图，但真正稳定、安全、动态可行的动作生成发生在这里。

## 方法详解

### Depth affordance

用深度而不是 RGB 做目标 affordance，降低水下颜色和光照偏移的影响。

### Goal-conditioned policy

给定目标对象，affordance heatmap 指导 diffusion policy 产生控制动作。

### Self-supervised underwater data

系统自动收集成功水下抓取 demonstrations，用真实水下闭环数据训练策略。

## 关键图表

### Figure 1: UMI-Underwater overview

![Figure 1](https://arxiv.org/html/2603.27012/figures/Overview3.png)

**图意**：UMI-Underwater overview。阅读时重点看输入信号、控制或策略模块、机器人状态反馈和动作输出之间的关系。
### Figure 2: Underwater manipulation setup

![Figure 2](https://arxiv.org/html/2603.27012/figures/underwater_setup_v5.jpg)

**图意**：Underwater manipulation setup。阅读时重点看输入信号、控制或策略模块、机器人状态反馈和动作输出之间的关系。
### Figure 3: Affordance-conditioned policy architecture

![Figure 3](https://arxiv.org/html/2603.27012/figures/architecture3.jpg)

**图意**：Affordance-conditioned policy architecture。阅读时重点看输入信号、控制或策略模块、机器人状态反馈和动作输出之间的关系。

## 实验解读

池内实验评估 seen objects、background shift 和只在陆上出现过的新物体。结果显示 depth/affordance 条件化比 RGB-only policy 更稳。

## 局限与风险

它仍是学习型 visuomotor policy，强安全边界和接触力控制不是论文重点；真实海域长时部署还需要更多扰动和恢复实验。

## 速查卡片

> [!summary] UMI-Underwater
> - **项目页**: https://umi-under-water.github.io
> - **arXiv**: https://arxiv.org/abs/2603.27012
> - **arXiv HTML**: https://arxiv.org/html/2603.27012
