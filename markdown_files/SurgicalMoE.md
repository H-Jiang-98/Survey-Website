---
title: "Supervised Mixture-of-Experts for Surgical Grasping and Retraction"
method_name: "Surgical MoE"
authors: ["Lorenzo Mazza"]
year: 2026
venue: "RSS"
tags: ["robot-manipulation", "safe-control", "imitation-learning", "closed-loop-control", "dexterous-manipulation"]
image_source: "online"
arxiv_html: "https://arxiv.org/html/2601.21971"
---
# Surgical MoE

## 一句话总结

这篇论文把 phase-structured MoE 加到 ACT 这类 action chunking policy 上，用于外科抓取和牵拉。它属于 System0，因为输出直接是机器人在受限、安全敏感工作空间中的动作块和夹爪状态。

## 为什么属于 System0

1. 输入是 stereo endoscopic images 和机器人状态，输出是短时域动作 chunk。
2. phase experts 把 grasp、retraction 等阶段结构显式注入低层动作生成。
3. 外科场景要求高安全性、可预测性和受限空间内稳定接触，符合 System0 的安全执行层定义。

## 问题背景

这篇论文关注的不是语言理解或长期任务规划，而是机器人执行层如何把局部感知、人体输入、接触状态、运动约束或硬件映射转成可执行动作。按照我们对 System0 的定义，它落在“身体级快反应”这一层：上层可以给目标或意图，但真正稳定、安全、动态可行的动作生成发生在这里。

## 方法详解

### Phase experts

不同专家负责不同任务阶段，避免单一动作头同时覆盖所有阶段导致混淆。

### Gated action mixture

最终动作按 gating network 的阶段概率做加权混合：$\hat a_{t+j}=\sum_h \pi_{h,t+j}\mu_{h,t+j}$。

### Lightweight policy

作者选择轻量 ACT 而不是大 VLA，强调低延迟和小数据外科任务中的可部署性。

## 关键图表

### Figure 1: Surgical grasping and retraction setup

![Figure 1](https://arxiv.org/html/2601.21971/x3.png)

**图意**：Surgical grasping and retraction setup。阅读时重点看输入信号、控制或策略模块、机器人状态反馈和动作输出之间的关系。
### Figure 2: MoE policy structure

![Figure 2](https://arxiv.org/html/2601.21971/x4.png)

**图意**：MoE policy structure。阅读时重点看输入信号、控制或策略模块、机器人状态反馈和动作输出之间的关系。
### Figure 3: Data scale ablation

![Figure 3](https://arxiv.org/html/2601.21971/figures/ablation_data_scale.png)

**图意**：Data scale ablation。阅读时重点看输入信号、控制或策略模块、机器人状态反馈和动作输出之间的关系。

## 实验解读

实验比较 VLA、ACT baseline 和 supervised MoE。结论是大模型并不能自然学会安全外科操作，显式阶段结构能显著提升 grasping/retraction 的稳定性。

## 局限与风险

阶段标签在训练时是 privileged information，部署泛化依赖 gating 的稳定性；真实外科闭环还需要更严格的安全认证。

## 速查卡片

> [!summary] Surgical MoE
> - **arXiv**: https://arxiv.org/abs/2601.21971
> - **arXiv HTML**: https://arxiv.org/html/2601.21971
> - **论文**: https://arxiv.org/pdf/2601.21971
