---
title: "ConTac: Continuum-Emulated Soft Skinned Arm with Vision-based Shape Sensing and Contact-aware Manipulation"
method_name: "ConTac"
authors: ["Tuan Tai Nguyen"]
year: 2024
venue: "RSS"
tags: ["tactile-feedback", "state-estimation", "contact-reasoning", "closed-loop-control", "safe-control"]
image_source: "online"
---
# ConTac

## 一句话总结

ConTac 用视觉式软皮肤同时估计连续体机械臂形状和接触位置，并把这些状态接入 admittance control，让机械臂能对碰撞做即时反应。

## 为什么属于 System0

1. 核心反馈是本体形状估计和触觉接触检测，正好对应 System0 的局部身体状态感知。
2. 接触位置不是离线分析，而是进入 admittance control，使机器人对外部碰撞做即时反应。
3. 它解决的是连续体/软体机械臂的快闭环感知和接触安全问题，不是语言规划或任务分解。

## 问题背景

这篇论文关注的不是语言理解或长期任务规划，而是机器人执行层如何把局部感知、人体输入、接触状态、运动约束或硬件映射转成可执行动作。按照我们对 System0 的定义，它落在“身体级快反应”这一层：上层可以给目标或意图，但真正稳定、安全、动态可行的动作生成发生在这里。

## 方法详解

### Soft-skin proprioception

软皮肤形变由内部视觉观测捕获，模型估计连续体臂的形状和末端位姿。

### Contact localization

同一感知系统还估计触碰位置，为碰撞检测和局部接触推理提供输入。

### Admittance reaction

形状和接触状态被用于 admittance control，使机械臂在接触时改变运动响应。

## 关键图表

### Figure 1: ConTac sensing system overview

![Figure 1](https://lh3.googleusercontent.com/sitesv/AA5AbUBkOLgyb96ft6wP9NmTdfKoljLE4Zlzwb97O1QOPA-Kq_XgWmAai6tQ5ug-v7fpXKLv1KPOPt5csNc_XjxsWo-IdLSgDx6WXHgOzRGvHGSptSOlgrX6ZfHJ14T8wkGUqnGAmBKgCkxJPqmZyDjWPEQuAFnUSSIwY2ztZ9X6uiWnVdjPr9KvcqS1DcfDv10S6Cfvq6JK8M6ejLzQoygAw5HCewU3XAGkrrr11eyDAcg=w1280)

**图意**：这张总览图把 camera/input image、shape network、contact network、shape/contact information 和 controller 连在一起。System0 视角下，关键不是“识别到接触”本身，而是接触位置和形状估计被送进控制器，直接改变连续体机械臂的运动响应。

### Figure 2: Contact depth and localization

![Figure 2](https://lh3.googleusercontent.com/sitesv/AA5AbUAv4sLWIOT74wyzp1rgY6q4sFLiQPl0f-KxwVWVrZMvVh60pegVroyMh2w7UXnfnhhoBMY7VZCCxE3jJb69n_qDECWwgLs4EHj22cJ1e4182LPDOk8phuxvHXpptoPZUa9dizxeXG3g7k_tbdc1_vNE9t3WUlRefKR0njSzUt5Cu_ie6ys7jS6PqqF53PsjtKuEO_wMHuA49K2RXw4KNO94niRI2MMzKZH4TbvKqsE=w1280)

**图意**：动图展示从软皮肤视觉观测到接触深度、接触位置的在线估计。它说明 ConTac 的触觉反馈不是外部标注，而是机械臂自身皮肤形变产生的身体状态信号。

### Figure 3: Vision-based soft-skin sensing demo

![Figure 3](https://lh3.googleusercontent.com/sitesv/AA5AbUCxBvzoAV3lkxmsK-eed0pmc0gQpUTbi3vAwQkqvfbzSfCP34JBKdSDfMC3LfGXQZOSH2QkNKa9S0WuwFTmNaRBZRtGaCoELpBVoaahmmHdHmw4i3YDnfZkfMP82StOQWerM_DqHqf1nYXaULm5UgKCv-nQTvet6-h9QEj-QreGXp6yrSO6GYIZHMFSvbO7rqsI29FTvRH4o_sheJJmmS0wI8X8YkkF2mnNBpvo=w1280)

**图意**：这张图展示内部相机图像、处理后的二值图和估计出的连续体形状。它对应论文的 proprioception 部分：机器人不是只依赖末端位姿，而是用软皮肤视觉恢复身体形变。

## 实验解读

论文报告了软皮肤机械臂的形状估计和接触定位精度，并展示接触感知进入控制后，机械臂可以对碰撞做出更合适的让步或避让反应。

## 局限与风险

项目重点是感知和接触反应，还没有覆盖复杂多接触操作或高频动力学控制。对需要多点接触、快速滑移或大外力冲击的任务，单一软皮肤视觉反馈可能还需要更高频的力觉或本体传感融合。

## 速查卡片

> [!summary] ConTac
> - **项目页**: https://sites.google.com/view/contacsensing
> - **论文**: http://www.roboticsproceedings.org/rss20/p097.pdf
