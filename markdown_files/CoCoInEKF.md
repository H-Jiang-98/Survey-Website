---
title: "CoCo-InEKF: State Estimation with Learned Contact Covariances in Dynamic, Contact-Rich Scenarios"
method_name: "CoCo-InEKF"
authors: ["Michael Baumgartner"]
year: 2026
venue: "RSS"
tags: ["contact-reasoning", "robust-control", "legged-locomotion", "contact-rich-manipulation", "state-estimation"]
image_source: "online"
arxiv_html: "https://arxiv.org/html/2605.15122v1"
---
# CoCo-InEKF
## 一句话总结

> CoCo-InEKF: State Estimation with Learned Contact Covariances in Dynamic, Contact-Rich Scenarios 主要落在 [[biped]]、[[contact-estimation]]、[[接触推理]]、[[接触丰富操作]] 这一类问题上;从 System0 视角看,它的价值在于把身体状态、接触变化、局部感知或动力学约束接到机器人低层闭环里,让动作生成更稳定、安全、实时。
## 核心贡献

1. 围绕 **CoCo-InEKF: State Estimation with Learned Contact Covariances in Dynamic, Contact-Rich Scenarios** 建立了一个与 biped、contact-estimation、接触推理、接触丰富操作、足式运动、鲁棒控制 相关的机器人问题设定,重点不是高层语义规划,而是身体级闭环中的状态变化、接触变化或控制约束。
2. 方法层面的关键价值在于把感知、动力学、约束优化、策略学习或硬件反馈组织成可执行的低层动作生成机制,适合放进 System0 的快反应模块中理解。
3. 对系统集成而言,这类工作补上了 System1 技能选择与电机级控制之间的一层:它负责把上层目标变成动态可行、可恢复、可安全过滤的动作。
4. 若论文包含真实机器人、接触任务或仿真到真实迁移,核心应看它是否报告了闭环稳定性、扰动恢复、硬件延迟、失败模式和跨场景泛化。
## 问题背景

这篇论文可以放在 System0 的“具身快系统”背景下阅读:机器人需要在很短时间内根据本体状态、接触、力/触觉、局部视觉或动力学预测做出动作,而不是等待慢速规划器重新推理整个任务。biped、contact-estimation、接触推理、接触丰富操作、足式运动、鲁棒控制、状态估计 相关问题的难点通常不在“知道要做什么”,而在动作执行过程中状态会快速偏离、接触会突然改变、约束会实时收紧,低层系统必须及时修正。

### System0 关联

精读时优先看三件事:第一,论文使用哪些快速反馈量,例如关节状态、接触状态、力觉、触觉、深度或局部视觉;第二,反馈如何进入 MPC、WBC、策略网络、安全过滤器或恢复控制器;第三,输出是否直接影响机器人下一步动作,而不是只服务离线分析或高层任务规划。
## 方法详解

### System0 视角解析

从 System0 角度看,这篇论文的方法部分应当围绕 biped、contact-estimation、接触推理、接触丰富操作、足式运动、鲁棒控制、状态估计 来读。核心问题是:系统如何从快速变化的机器人状态中构造控制输入,如何处理接触、碰撞、约束、模型误差或执行延迟,以及如何把这些信息变成下一时刻的动作。

如果方法里有优化问题,重点看目标函数、约束和求解频率;如果方法里有学习策略,重点看观测空间、动作空间、训练扰动和部署时的闭环频率;如果方法里有硬件或传感器,重点看传感链路是否真的进入控制回路。这样读可以把论文和 System2 语言规划、System1 技能编排区分开来。

### 关键公式与机制

#### 公式 1: [[动力学或策略机制]]

$$\mathbf{x}:=\left({}^{W}\mathbf{R}_{B},{}^{W}\mathbf{v}_{B},{}^{W}\mathbf{p}_{B},{}^{W}\mathbf{p}_{C_{1}},\dots,{}^{W}\mathbf{p}_{C_{N}},{}^{B}\mathbf{b}_{\boldsymbol{\omega}},{}^{B}\mathbf{b}_{\mathbf{a}}\right).$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 2: [[动力学或策略机制]]

$${}^{W}\dot{\mathbf{p}}_{C_{i}}=-{}^{W}\mathbf{R}_{B}{}^{B}\mathbf{w}_{C_{i}},$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 3: [[动力学或策略机制]]

$$\mathbf{h}_{C_{i}}(\mathbf{q})={}^{W}\mathbf{R}_{B}^{\top}\left({}^{W}\mathbf{p}_{C_{i}}-{}^{W}\mathbf{p}_{B}\right)+{}^{B}\mathbf{J}_{C_{i}}(\mathbf{q})\mathbf{w}_{\mathbf{q}},$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 4: [[动力学或策略机制]]

$$\mathbf{K}\leftarrow\mathbf{P}^{-}\mathbf{H}^{\top}(\mathbf{H}\mathbf{P}^{-}\mathbf{H}^{\top}+\mathbf{N})^{-1}$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 5: [[动力学或策略机制]]

$$\mathbf{o}:=\left({}^{B}\boldsymbol{\omega},{}^{B}\mathbf{a},\mathbf{q},\dot{\mathbf{q}},\boldsymbol{\tau},{}^{B}\mathbf{p}_{B\rightarrow C_{i}},{}^{B}\mathbf{v}_{B\rightarrow C_{i}}\right),$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 6: [[优化目标/约束]]

$$\mathcal{L}\left(\mathbf{x},\hat{\mathbf{x}}\right)=\left\lVert {}^{W}\mathbf{R}_{B}^{\top}{}^{W}\mathbf{v}_{B}-{}^{W}\hat{\mathbf{R}}_{B}^{\top}{}^{W}\hat{\mathbf{v}}_{B}\right \rVert^{2}_{2},$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 7: [[动力学或策略机制]]

$$\mathbb{E}\!\left[\tilde{\mathbf{x}}\right]=\mathbf{0},\qquad\mathbb{E}\!\left[\tilde{\mathbf{x}}\,\tilde{\mathbf{x}}^{\top}\right]=\mathbf{P},$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 8: [[动力学或策略机制]]

$$\epsilon=\tilde{\mathbf{x}}^{\top}\,\mathbf{P}^{-1}\,\tilde{\mathbf{x}},$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 9: [[动力学或策略机制]]

$$={}^{W}\mathbf{R}_{B}\left({}^{B}\mathbf{a}-{}^{B}\mathbf{b}_{\mathbf{a}}-{}^{B}\mathbf{w}_{\mathbf{a}}\right)+{}^{W}\mathbf{g},$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 10: [[动力学或策略机制]]

$$={}^{W}\mathbf{R}_{B}\left({}^{B}\boldsymbol{\omega}-{}^{B}\mathbf{b}_{\boldsymbol{\omega}}-{}^{B}\mathbf{w}_{\boldsymbol{\omega}}\right)_{\times},$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。
## 关键图表

### Figure 1: Pipeline / core system figure: CoCo-InEKF. Given a set of predefined contact candidates and the proprioceptive senso

![Figure 1](https://arxiv.org/html/2605.15122v1/x1.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Pipeline / core system figure: CoCo-InEKF. Given a set of predefined contact candidates and the proprioceptive senso”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 2: Contact Covariance. Visualization of the contact covariance total standard deviation

![Figure 2](https://arxiv.org/html/2605.15122v1/x3.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Contact Covariance. Visualization of the contact covariance total standard deviation”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 3: g

![Figure 3](https://arxiv.org/html/2605.15122v1/images/lima_foot_full_cp_config_4_5.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“g”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### 论文表格

#### Table 1: TABLE I: Baseline methods.

| InEKF, GT contacts | InEKF, with ground-truth contact using xy-velocity (≤ 0.25 m / s \leq 0.25 $\text{\,}\mathrm{m}\mathrm{/}\mathrm{s}) and height (≤ 0.01 m \leq 0.01\text{\,}\mathrm{m}).$ |
| --- | --- |
| InEKF, heuristic contacts | InEKF, with contact heuristic using estimated xy-velocity (≤ 0.25 m / s \leq 0.25 $\text{\,}\mathrm{m}\mathrm{/}\mathrm{s}) and height (≤ 0.01 m \leq 0.01\text{\,}\mathrm{m}).$ |
| Hybrid Baseline | Method from [26 ]. |
| Hybrid Baseline+ | As Hybrid Baseline, but with reduced model size and added slip classification. |
| SET | Unstructured end-to-end transformer-based method [39 ]. |

**说明**: TABLE I: Baseline methods.

#### Table 2: TABLE II: Configuration of small and large SET models.

| | Small | Large |
| --- | --- | --- |
| Self-attention blocks | 6 | 6 |
| Heads per block | 4 | 8 |
| Linear token embedding dimensions | 128 | 256 |
| MLP hidden dimensions | 256 | 1024 |

**说明**: TABLE II: Configuration of small and large SET models.

#### Table 3: TABLE III: ATE comparison on simulated dancing motions.

| | Linear Velocity ATE | Position ATE | Orientation ATE | | | | | | | | | |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Model | RMSE | MAE | MED | STD | RMSE | MAE | MED | STD | RMSE | MAE | MED | STD |
| InEKF, GT contacts | 0.176 | 0.070 | 0.037 | 0.162 | 0.422 | 0.119 | 0.042 | 0.405 | 0.033 | 0.022 | 0.013 | 0.025 |
| InEKF, heuristic contacts | 2.675 | 1.429 | 0.436 | 2.262 | 17.423 | 8.260 | 0.866 | 15.340 | 0.041 | 0.031 | 0.023 | 0.027 |
| Hybrid Baseline | 0.123 | 0.060 | 0.034 | 0.107 | 0.163 | 0.078 | 0.048 | 0.143 | 0.031 | 0.019 | 0.011 | 0.025 |
| Hybrid Baseline+ | 0.121 | 0.061 | 0.036 | 0.105 | 0.111 | 0.065 | 0.040 | 0.090 | 0.027 | 0.019 | 0.012 | 0.020 |
| CoCo-InEKF (ours) | 0.046 | 0.028 | 0.018 | 0.037 | 0.124 | 0.079 | 0.052 | 0.096 | 0.031 | 0.020 | 0.013 | 0.024 |
| SET, small | 0.279 | 0.195 | 0.138 | 0.199 | 0.487 | 0.379 | 0.313 | 0.305 | 0.072 | 0.059 | 0.052 | 0.042 |
| SET, large | 0.286 | 0.203 | 0.143 | 0.202 | 0.395 | 0.290 | 0.205 | 0.269 | 0.072 | 0.059 | 0.052 | 0.042 |

**说明**: TABLE III: ATE comparison on simulated dancing motions.

#### Table 4: TABLE IV: ATE comparison on simulated ground motions.

| | Linear Velocity ATE | Position ATE | Orientation ATE | | | | | | | | | |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Model | RMSE | MAE | MED | STD | RMSE | MAE | MED | STD | RMSE | MAE | MED | STD |
| InEKF, GT contacts | 0.418 | 0.296 | 0.203 | 0.295 | 1.573 | 0.971 | 0.512 | 1.238 | 0.078 | 0.048 | 0.028 | 0.061 |
| InEKF, heuristic contacts | 4.448 | 3.110 | 2.117 | 3.179 | 32.695 | 19.738 | 10.248 | 26.065 | 0.049 | 0.039 | 0.032 | 0.029 |
| Hybrid Baseline | 0.363 | 0.266 | 0.190 | 0.247 | 1.455 | 0.899 | 0.414 | 1.144 | 0.063 | 0.037 | 0.020 | 0.051 |
| Hybrid Baseline+ | 0.428 | 0.316 | 0.232 | 0.289 | 1.982 | 1.238 | 0.613 | 1.547 | 0.074 | 0.041 | 0.021 | 0.061 |
| CoCo-InEKF (ours) | 0.099 | 0.069 | 0.042 | 0.071 | 0.342 | 0.271 | 0.239 | 0.210 | 0.069 | 0.056 | 0.049 | 0.040 |
| SET, small | 0.107 | 0.069 | 0.034 | 0.082 | 0.126 | 0.100 | 0.087 | 0.076 | 0.058 | 0.041 | 0.031 | 0.041 |
| SET, large | 0.096 | 0.063 | 0.032 | 0.072 | 0.159 | 0.132 | 0.124 | 0.088 | 0.058 | 0.041 | 0.031 | 0.041 |

**说明**: TABLE IV: ATE comparison on simulated ground motions.

#### Table 5: TABLE V: Model architecture ablation w.r.t. linear velocity ATE on dancing motions for our model, baseline models, and

| Model | RMSE | # Params. | NN / SE [ms] |
| --- | --- | --- | --- |
| Hybrid Baseline | 0.123 | 2’473’744 | 1.81 / 2.10 |
| Hybrid Baseline, MLP | 0.135 | 239’824 | 0.15 / 0.44 |
| Hybrid Baseline+ w/o slip | 0.122 | 239’304 | 0.16 / 0.45 |
| Hybrid Baseline+ | 0.121 | 239’564 | 0.16 / 0.45 |
| CoCo-InEKF (ours) | 0.046 | 240’344 | 0.14 / 0.42 |
| SET, small | 0.279 | 810’755 | 2.09 |
| SET, large | 0.286 | 4’770’307 | 3.02 |

**说明**: TABLE V: Model architecture ablation w.r.t. linear velocity ATE on dancing motions for our model, baseline models, and intermediate models between Hybrid Baseline and Hybrid Baseline+, showing the effects of individual changes. We also report the number of parameters and the inference time for the neural network (NN), together with the full state estimator (SE). For SET, a single value is reported, as the SE consists solely of the NN.

#### Table 6: TABLE VI: History size ablation, H = 20 H=20 vs. H = 150 H=150. We also report number of parameters and the inference

| Model | H H | RMSE | # Params. | NN / SE [ms] |
| --- | --- | --- | --- | --- |
| Hybrid Baseline | 20 | 0.123 | 2’473’744 | 1.81 / 2.10 |
| Hybrid Baseline | 150 | 0.150 | 10’862’352 | 6.43 / 6.79 |
| Hybrid Baseline+ | 20 | 0.121 | 239’564 | 0.16 / 0.45 |
| Hybrid Baseline+ | 150 | 0.124 | 1’737’164 | 0.62 / 0.98 |
| CoCo-InEKF | 20 | 0.046 | 240’344 | 0.14 / 0.42 |
| CoCo-InEKF | 150 | 0.052 | 1’737’944 | 0.61 / 0.97 |

**说明**: TABLE VI: History size ablation, H = 20 H=20 vs. H = 150 H=150. We also report number of parameters and the inference time for the neural network (NN) and the full state estimator (SE), respectively.

#### Table 7: TABLE VII: BPTT unroll size ablation. We report linear velocity ATE on synthetic dancing data, as well as the number of

| | RMSE | MAE | MED | STD | # Iters. |
| --- | --- | --- | --- | --- | --- |
| L = 64 L=64 | 0.066 | 0.043 | 0.027 | 0.051 | 89’600 |
| L = 128 L=128 (ours) | 0.046 | 0.028 | 0.018 | 0.037 | 64’600 |
| L = 256 L=256 | 0.051 | 0.032 | 0.021 | 0.040 | 18’800 |

**说明**: TABLE VII: BPTT unroll size ablation. We report linear velocity ATE on synthetic dancing data, as well as the number of training iterations (limited by the 5-day training time).

#### Table 8: TABLE VIII: Ablation study on the scaling of the number of contact points. We report linear velocity ATE on synthetic g

| Model | N N | RMSE | # Params. | NN / SE [ms] |
| --- | --- | --- | --- | --- |
| CoCo-InEKF | 4 | 0.134 | 240’344 | 0.14 / 0.42 |
| CoCo-InEKF | 10 | 0.099 | 334’844 | 0.18 / 0.87 |
| CoCo-InEKF | 18 | 0.069 | 460’844 | 0.26 / 2.01 |
| SET, small | 10 | 0.107 | 815’363 | 2.03 |
| SET, large | 10 | 0.096 | 4’779’523 | 3.06 |
| SET, small | 18 | 0.104 | 821’507 | 2.08 |
| SET, large | 18 | 0.094 | 4’791’811 | 3.12 |

**说明**: TABLE VIII: Ablation study on the scaling of the number of contact points. We report linear velocity ATE on synthetic ground motion data, number of parameters, and inference time for the neural network (NN) and the full state estimator (SE), respectively. For SET, a single value is reported, as the SE consists solely of the NN.

#### Table 9: TABLE IX: The automated contact candidate selection compared to the handpicked baseline, for linear velocity ATE. Range

| | RMSE |
| --- | --- |
| Dancing motions, automated | [0.056, 0.052] |
| Dancing motions, handpicked | 0.057 |
| Ground motions, automated | [0.104, 0.092] |
| Ground motions, handpicked | 0.099 |

**说明**: TABLE IX: The automated contact candidate selection compared to the handpicked baseline, for linear velocity ATE. Ranges indicate [worst, best] sample.

#### Table 10: TABLE X: NEES evaluation on the simulated dancing test data, reported as the percentage of time steps with NEES values

| Model | Core | Vel. | Pos. | Ori. |
| --- | --- | --- | --- | --- |
| InEKF, GT contacts | 37.7% | 66.8% | 60.7% | 65.4% |
| InEKF, heur. contacts | 20.3% | 47.4% | 23.6% | 47.4% |
| Hybrid Baseline | 18.2% | 50.2% | 11.3% | 52.2% |
| Hybrid Baseline+ | 18.4% | 50.3% | 7.9% | 51.8% |
| CoCo-InEKF (ours) | 52.1% | 71.1% | 59.8% | 68.1% |

**说明**: TABLE X: NEES evaluation on the simulated dancing test data, reported as the percentage of time steps with NEES values within the 95% confidence bounds ([2.7, 19 ] [2.7,19] for the combined core state, [0.22, 9.35 ] [0.22,9.35] for the individual states).

#### Table 11: TABLE XI: ATE comparison on 20 real-world ground motion sequences.

| | Linear Velocity ATE | Position ATE | Orientation ATE | | | | | | | | | |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Model | RMSE | MAE | MED | STD | RMSE | MAE | MED | STD | RMSE | MAE | MED | STD |
| InEKF, heuristic contacts | 1.5419 | 0.5305 | 0.0261 | 1.4478 | 21.3841 | 6.0165 | 0.0671 | 20.5203 | 0.0152 | 0.0110 | 0.0081 | 0.0105 |
| Hybrid Baseline* | 0.1178 | 0.0738 | 0.0453 | 0.0918 | 0.7212 | 0.3764 | 0.2093 | 0.6152 | 0.0152 | 0.0110 | 0.0085 | 0.0105 |
| Hybrid Baseline+ | 0.1699 | 0.1108 | 0.0669 | 0.1288 | 1.4869 | 0.7305 | 0.3396 | 1.2951 | 0.0188 | 0.0129 | 0.0095 | 0.0137 |
| CoCo-InEKF (ours) | 0.0805 | 0.0398 | 0.0167 | 0.0699 | 0.2019 | 0.1497 | 0.1181 | 0.1355 | 0.0302 | 0.0201 | 0.0130 | 0.0225 |
| SET, small* | 0.1002 | 0.0501 | 0.0193 | 0.0867 | 0.1665 | 0.1240 | 0.0952 | 0.1110 | 0.0257 | 0.0165 | 0.0109 | 0.0197 |
| SET, large* | 0.0974 | 0.0475 | 0.0174 | 0.0850 | 0.2245 | 0.1803 | 0.1498 | 0.1337 | 0.0257 | 0.0165 | 0.0109 | 0.0197 |

**说明**: TABLE XI: ATE comparison on 20 real-world ground motion sequences.

#### Table 12: TABLE XII: Success rate [% $\text{\,}\mathrm{\char 37\relax}$ ] of various real-world dance motions with the state estim

| | Dances | Pirouette | Moonwalk |
| --- | --- | --- | --- |
| Model | (training set) | (unseen) | (unseen) |
| MoCap | 92 | 90 | 100 |
| InEKF, heuristic contacts | 77 | 60 | 100 |
| Hybrid Baseline+ | 85 | 50 | 10 |
| CoCo-InEKF (ours) | 95 | 100 | 100 |

**说明**: 该表格来自论文中的公式、奖励项或实验设置；为避免 PDF 抽取噪声，网页中仅保留可渲染的核心符号和数值。
## 实验解读

- 评价重点:围绕 biped、contact-estimation、接触推理、接触丰富操作、足式运动,优先看真实机器人结果、扰动恢复、接触稳定性、控制误差、成功率、实时性和 sim-to-real 差距。
- System0 视角:实验结论不只看最终成功率,还要看闭环过程中是否能处理状态估计误差、执行延迟、碰撞/接触突变和外部扰动。
- 与上层系统的关系:如果方法能在快闭环内稳定工作,它可以作为 System1 技能或 System2 规划的底层执行保障;如果只在离线或慢速设定下成立,则还不能直接作为身体级快闭环模块。
## 批判性思考

### 优点

- 论文把 biped、contact-estimation、接触推理、接触丰富操作、足式运动 放到机器人身体级闭环里讨论,和只做高层语义推理的工作相比,更接近真实机器人部署时会遇到的状态、动力学和安全约束。
- 若论文同时给出公式、系统结构图、硬件实验或消融结果,这些证据可以帮助判断方法是否真的能支撑毫秒到百毫秒级反应。

### 局限

- 需要重点核查控制频率、推理延迟、传感延迟、失败恢复、硬件磨损、未见场景泛化和安全边界;这些因素决定它能否成为可靠的 System0 模块。
- 对学习型方法,还要区分训练环境中的鲁棒性和真实部署中的鲁棒性,尤其是接触丰富任务、动态运动任务和人形机器人任务。
## 论文定位

- 研究对象:CoCo-InEKF: State Estimation with Learned Contact Covariances in Dynamic, Contact-Rich Scenarios。
- 关键词:biped、contact-estimation、接触推理、接触丰富操作、足式运动、鲁棒控制、状态估计。
- System0 关联:关注机器人身体级快反应链路中“状态/接触/局部感知 -> 约束/优化/策略 -> 动作”的短闭环,不把问题停留在语言规划或任务分解层。

## 速查卡片

> [!summary] CoCo-InEKF
> - **论文**: https://www.roboticsproceedings.org/rss22/p178.pdf
> - **arXiv**: http://arxiv.org/abs/2605.15122v1
> - **arXiv HTML**: https://arxiv.org/html/2605.15122v1
