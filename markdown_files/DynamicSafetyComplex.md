---
title: "Dynamic Safety in Complex Environments: Synthesizing Safety Filters with Poisson’s Equation"
method_name: "Dynamic Safety Complex"
authors: ["Gilbert Bahati"]
year: 2025
venue: "RSS"
tags: ["real-time-control", "legged-locomotion", "safe-control", "imitation-learning", "humanoid", "navigation"]
image_source: "online"
arxiv_html: "https://arxiv.org/html/2505.06794v1"
---
# Dynamic Safety Complex
## 一句话总结

> Dynamic Safety in Complex Environments: Synthesizing Safety Filters with Poisson’s Equation 主要落在 [[certified-control]]、[[人形机器人]]、[[模仿学习]]、[[足式运动]] 这一类问题上;从 System0 视角看,它的价值在于把身体状态、接触变化、局部感知或动力学约束接到机器人低层闭环里,让动作生成更稳定、安全、实时。
## 核心贡献

1. 围绕 **Dynamic Safety in Complex Environments: Synthesizing Safety Filters with Poisson’s Equation** 建立了一个与 certified-control、人形机器人、模仿学习、足式运动、navigation、quadruped 相关的机器人问题设定,重点不是高层语义规划,而是身体级闭环中的状态变化、接触变化或控制约束。
2. 方法层面的关键价值在于把感知、动力学、约束优化、策略学习或硬件反馈组织成可执行的低层动作生成机制,适合放进 System0 的快反应模块中理解。
3. 对系统集成而言,这类工作补上了 System1 技能选择与电机级控制之间的一层:它负责把上层目标变成动态可行、可恢复、可安全过滤的动作。
4. 若论文包含真实机器人、接触任务或仿真到真实迁移,核心应看它是否报告了闭环稳定性、扰动恢复、硬件延迟、失败模式和跨场景泛化。
## 问题背景

这篇论文可以放在 System0 的“具身快系统”背景下阅读:机器人需要在很短时间内根据本体状态、接触、力/触觉、局部视觉或动力学预测做出动作,而不是等待慢速规划器重新推理整个任务。certified-control、人形机器人、模仿学习、足式运动、navigation、quadruped、实时控制 相关问题的难点通常不在“知道要做什么”,而在动作执行过程中状态会快速偏离、接触会突然改变、约束会实时收紧,低层系统必须及时修正。

### System0 关联

精读时优先看三件事:第一,论文使用哪些快速反馈量,例如关节状态、接触状态、力觉、触觉、深度或局部视觉;第二,反馈如何进入 MPC、WBC、策略网络、安全过滤器或恢复控制器;第三,输出是否直接影响机器人下一步动作,而不是只服务离线分析或高层任务规划。
## 方法详解

### System0 视角解析

从 System0 角度看,这篇论文的方法部分应当围绕 certified-control、人形机器人、模仿学习、足式运动、navigation、quadruped、实时控制、safe-control 来读。核心问题是:系统如何从快速变化的机器人状态中构造控制输入,如何处理接触、碰撞、约束、模型误差或执行延迟,以及如何把这些信息变成下一时刻的动作。

如果方法里有优化问题,重点看目标函数、约束和求解频率;如果方法里有学习策略,重点看观测空间、动作空间、训练扰动和部署时的闭环频率;如果方法里有硬件或传感器,重点看传感链路是否真的进入控制回路。这样读可以把论文和 System2 语言规划、System1 技能编排区分开来。

### 关键公式与机制

#### 公式 1: [[优化目标/约束]]

$$\mathcal{K}_{\mathrm{CBF}}(\mathbf{x})=\left\{\mathbf{u}\in\mathbb{R}^{m}\,\big{|}\,\dot{h}_{\mathcal{S}}(\mathbf{x},\mathbf{u})\geq-\gamma(h_{\mathcal{S}}(\mathbf{x}))\right\}$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 2: [[优化目标/约束]]

$$\!\!\!\!\sup_{\mathbf{u}\in\mathbb{R}^{m}}\!\left\{{\dot{h}_{\mathcal{S}}(\mathbf{x},\mathbf{u})\!=\!L_{\mathbf{f}}h_{\mathcal{S}}(\mathbf{x})}+{L_{\mathbf{g}}h_{\mathcal{S}}(\mathbf{x})}\mathbf{u}>-\gamma(h_{\mathcal{S}}(\mathbf{x}))\!\right\}\!.\!\!\!$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 3: [[优化目标/约束]]

$$\mathcal{C}_{\mathrm{B}}=\left\{\vec{\mathbf{y}}\in\mathbb{R}^{3r}\,\big{|}\,h_{\mathrm{B}}(\vec{\mathbf{y}})\geq 0\right\}\subset\mathcal{C}\times\mathbb{R}^{3(r-1)},$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 4: [[优化目标/约束]]

$$\!\min_{h\in\mathcal{H}}\left\{\!J[h]\!=\!\!\iiint_{\Omega}\frac{1}{2}|Dh|^{2}\!-\!\frac{h}{\beta}\ln(1+e^{-\nabla\cdot\vec{\mathbf{v}}\beta})\mathrm{d}V\!\right\},$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 5: [[优化目标/约束]]

$$\mathcal{C}_{\mathrm{B}}=\left\{\vec{\mathbf{y}}\in\mathbb{R}^{6}\,\big{|}\,h_{\mathrm{B}}(\vec{\mathbf{y}})\geq 0\right\}\subset\mathcal{C}\times\mathbb{R}^{3}.$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 6: [[优化目标/约束]]

$$\dot{h}_{\mathcal{S}}(\mathbf{x})=\ {Dh_{\mathrm{\mathcal{S}}}(\mathbf{x})\cdot\mathbf{f}(\mathbf{x})}_{L_{\mathbf{f}}h_{\mathrm{\mathcal{S}}}(\mathbf{x})}+\ {Dh_{\mathrm{\mathcal{S}}}(\mathbf{x})\cdot\mathbf{g}(\mathbf{x})}_{L_{\mathbf{g}}h_{\mathrm{\mathcal{S}}}(\mathbf{x})}\mathbf{k}(\mathbf{x})\geq 0,$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 7: [[优化目标/约束]]

$$\mathcal{V}=\left\{\vec{\mathbf{v}}\in C^{\infty}(\ {\Omega};\mathbb{R}^{3})\,\big{|}\,\vec{\mathbf{v}}(\mathbf{y})=b(\mathbf{y})\hat{\mathbf{n}}(\mathbf{y})\text{on}\partial\Omega\right\}.$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 8: [[优化目标/约束]]

$$\mathcal{C}_{i}=\left\{\tilde{\mathbf{y}}\in\mathbb{R}^{3{i+1}}\,\big{|}\,h_{i}(\tilde{\mathbf{y}})\geq 0\right\}\quad i=1,\cdots,r-1$$

**含义**: 这是论文中与 优化目标/约束 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 9: [[动力学或策略机制]]

$$\left\{\begin{aligned} \Delta h(\mathbf{y})&=\nabla\cdot\vec{\mathbf{v}}(\mathbf{y})&\text{in}\Omega,\\ h(\mathbf{y})&=0&\text{on}\partial\Omega,\\ \end{aligned}\right.$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。

#### 公式 10: [[动力学或策略机制]]

$$\vec{\mathbf{y}}(\mathbf{x}):=\begin{bmatrix}\mathbf{y}(\mathbf{x})\\ \mathbf{y}^{(1)}(\mathbf{x})\\ \vdots\\ \mathbf{y}^{(r-1)}(\mathbf{x})\end{bmatrix}=\begin{bmatrix}\mathbf{y}(\mathbf{x})\\ L_{\mathbf{f}}\mathbf{y}(\mathbf{x})\\ \vdots\\ L_{\mathbf{f}}^{r-1}\mathbf{y}(\mathbf{x})\end{bmatrix}\in\mathbb{R}^{pr},$$

**含义**: 这是论文中与 动力学或策略机制 相关的核心数学表达;它不是单字符符号摘录,而是用于定位方法机制的完整公式块。

**符号说明**: 变量含义以论文原文上下文为准;这里保留原公式结构,避免改写造成符号错配。
## 关键图表

### Figure 1: Pipeline / core system figure: Safe set synthesis from perception data via Poisson’s equation. Hardware experimental

![Figure 1](https://arxiv.org/html/2505.06794v1/x1.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Pipeline / core system figure: Safe set synthesis from perception data via Poisson’s equation. Hardware experimental”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 2: Hardware experiments demonstrating Poisson safety functions for safety filtering. [Top

![Figure 2](https://arxiv.org/html/2505.06794v1/x5.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Hardware experiments demonstrating Poisson safety functions for safety filtering. [Top”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 3: Hardware experiments demonstrating dynamic behavior on quadrupedal and humanoid robots

![Figure 3](https://arxiv.org/html/2505.06794v1/x6.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Hardware experiments demonstrating dynamic behavior on quadrupedal and humanoid robots”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### 论文表格

#### Table 1: Table 1

| | D k w ≔ D w \| \| \| k D w ∂ \| \| w ∂ x ⋯ ∂ x n n. formulae-sequence ≔ D k w -set D w k D w w x ⋯ x n n D k w $\big{\{}D^{\xi}w\,\big{\lVert}\,\rVert\xi\lVert=k\big{\}},\quad D^{\xi}w=\frac{\partial^{\rVert\xi\lVert}w}{\partial x_{1}^{\xi_{1}}\cdots\partial x_{n}^{\xi_{n}}}. ≔ {\rVert \lVert \rVert = }, = divide ∂ \lVert \rVert ∂ 1 1 ⋯ ∂.$ | |
| --- | --- | --- |

**说明**: Table 1

#### Table 2: Table 2

| | x  ̇ = f (x) + g (x) u,  ̇ x f x g x u \displaystyle $\dot{\mathbf{x}}=\mathbf{f}(\mathbf{x})+\mathbf{g}(\mathbf{x})\mathbf{u}, = () + (),$ | | (1) |
| --- | --- | --- | --- |

**说明**: Table 2

#### Table 3: Table 3

| | x  ̇ = f cl (x) = f (x) + g (x) k (x).  ̇ x f cl x f x g x k x \displaystyle $\dot{\mathbf{x}}=\mathbf{f}_{\mathrm{cl}}(\mathbf{x})=\mathbf{f}(\mathbf{x})+\mathbf{g}(\mathbf{x})\mathbf{k}(\mathbf{x}). = () = () + () ().$ | | (2) |
| --- | --- | --- | --- |

**说明**: Table 3

#### Table 4: Table 4

| | S = {x ∈ R n \| h S (x) ≥ 0 }. S conditional-set x R n h S x 0 \displaystyle $\mathcal{S}=\big{\{}\mathbf{x}\in\mathbb{R}^{n}\,\big{\lVert}\,h_{\mathrm{\mathcal{S}}}(\mathbf{x})\geq 0\big{\}}. = {∈ \rVert () ≥ 0 }.$ | | (3) |
| --- | --- | --- | --- |

**说明**: Table 4

#### Table 5: Table 5

| | h ̇ S x D h S x f x ⏟ L f h S x D h S x g x ⏟ L g h S x k x ≥ ̇ h S x ⏟ D h S x f x L f h S x ⏟ D h S x g x L g h S x k x \displaystyle $\dot{h}_{\mathcal{S}}(\mathbf{x})=\ {Dh_{\mathrm{\mathcal{S}}}(\mathbf{x})\cdot\mathbf{f}(\mathbf{x})}_{L_{\mathbf{f}}h_{\mathrm{\mathcal{S}}}(\mathbf{x})}+\ {Dh_{\mathrm{\mathcal{S}}}(\mathbf{x})\cdot\mathbf{g}(\mathbf{x})}_{L_{\mathbf{g}}h_{\mathrm{\mathcal{S}}}(\mathbf{x})}\mathbf{k}(\mathbf{x})\geq 0, () = () ⋅ () () + () ⋅ () () () ≥ 0,$ | | (4) |
| --- | --- | --- | --- |

**说明**: Table 5

#### Table 6: Table 6

| | sup u ∈ R m h ̇ S x u L f h S x L g h S x u - h S x . supremum u R m ̇ h S x u L f h S x L g h S x u h S x \displaystyle\!\!\!\!\sup $\mathbf{u}\in\mathbb{R}^{m}}\!\left\{{\dot{h}_{\mathcal{S}}(\mathbf{x},\mathbf{u})\!=\!L_{\mathbf{f}}h_{\mathcal{S}}(\mathbf{x})}+{L_{\mathbf{g}}h_{\mathcal{S}}(\mathbf{x})}\mathbf{u}>-\gamma(h_{\mathcal{S}}(\mathbf{x}))\!\right\}\!.\!\!\! ∈ {(,) = () + () > - (()) }.$ | | (5) |
| --- | --- | --- | --- |

**说明**: Table 6

#### Table 7: Table 7

| | K CBF x u ∈ R m \| h ̇ S x u ≥ - h S x K CBF x -set u R m ̇ h S x u h S x \displaystyle $\mathcal{K}_{\mathrm{CBF}}(\mathbf{x})=\left\{\mathbf{u}\in\mathbb{R}^{m}\,\big{\lVert}\,\dot{h}_{\mathcal{S}}(\mathbf{x},\mathbf{u})\geq-\gamma(h_{\mathcal{S}}(\mathbf{x}))\right\} () = {∈ \rVert (,) ≥ - (()) }$ | | (6) |
| --- | --- | --- | --- |

**说明**: Table 7

#### Table 8: Table 8

| | k (x) = k x absent \displaystyle $\mathbf{k}(\mathbf{x})= () =$ | arg min u ∈ R m arg min u R m \displaystyle $\operatorname*{arg\,min}_{\mathbf{u}\in\mathbb{R}^{m}} ∈$ | u - k nom x u k nom x \displaystyle\\| $$ | | (Safety-Filter) |
| --- | --- | --- | --- | --- | --- |
| | | s. t. formulae-sequence s t \displaystyle\quad $\mathrm{s.t.}..$ | L f h S x L g h S x u ≥ - h S x . L f h S x L g h S x u h S x \displaystyle L $\mathbf{f}}h_{\mathrm{\mathcal{S}}}(\mathbf{x})+L_{\mathbf{g}}h_{\mathrm{\mathcal{S}}}(\mathbf{x})\mathbf{u}\geq-\gamma(h_{\mathrm{\mathcal{S}}}(\mathbf{x})). () + () ≥ - (()).$ | | |

**说明**: Table 8

#### Table 9: Table 9

| | L g L f i y (x) ≡ 0, L g L f i y x 0 \displaystyle L_{ $\mathbf{g}}L_{\mathbf{f}}^{i}\mathbf{y}(\mathbf{x})\equiv\mathbf{0}, () ≡,$ | ∀ i ∈ {0, ..., r - 2 }, for-all i 0 ... r 2 \displaystyle\quad\forall i\in\{0, $\dots,r-2\}, ∀ ∈ {0, ..., - 2 },$ | | (7) |
| --- | --- | --- | --- | --- |
| | rank (L g L f r - 1 y (x)) = p, rank L g L f r 1 y x p \displaystyle $\mathrm{rank}(L_{\mathbf{g}}L_{\mathbf{f}}^{r-1}\mathbf{y}(\mathbf{x}))=p, (- 1 ()) =,$ | ∀ x ∈ R n. for-all x R n \displaystyle\quad\forall $\mathbf{x}\in\mathbb{R}^{n}. ∀ ∈.$ | | (8) |

**说明**: Table 9

#### Table 10: Table 10

| | y → (x) ≔ [y (x) y (1) (x) ⋮ y (r - 1) (x) ] = [y (x) L f y (x) ⋮ L f r - 1 y (x) ] ∈ R p r, ≔ → y x matrix y x y 1 x ⋮ y r 1 x matrix y x L f y x ⋮ L f r 1 y x R p r \vec{ $\mathbf{y}}(\mathbf{x}):=\begin{bmatrix}\mathbf{y}(\mathbf{x})\\ \mathbf{y}^{(1)}(\mathbf{x})\\ \vdots\\ \mathbf{y}^{(r-1)}(\mathbf{x})\end{bmatrix}=\begin{bmatrix}\mathbf{y}(\mathbf{x})\\ L_{\mathbf{f}}\mathbf{y}(\mathbf{x})\\ \vdots\\ L_{\mathbf{f}}^{r-1}\mathbf{y}(\mathbf{x})\end{bmatrix}\in\mathbb{R}^{pr}, () ≔ [() (1) () ⋮ (- 1) () ] = [() () ⋮ - 1 () ] ∈,$ | | (9) |
| --- | --- | --- | --- |

**说明**: Table 10

#### Table 11: Table 11

| | d d t y → (x) d d t → y x \displaystyle $\frac{\mathrm{d}}{\mathrm{d}t}\vec{\mathbf{y}}(\mathbf{x}) divide ()$ | = [0 I p (r - 1) 0 0 ] ⏟ A y → (x) + [0 I p ] ⏟ B w absent ⏟ matrix 0 I p r 1 0 0 A → y x ⏟ matrix 0 I p B w \displaystyle=\ {\begin{bmatrix} $\mathbf{0}&\mathbf{I}_{p(r-1)}\\ \mathbf{0}&\mathbf{0}\end{bmatrix}}_{\mathbf{A}}\vec{\mathbf{y}}(\mathbf{x})\!+\!\ {\begin{bmatrix}\mathbf{0}\\ \mathbf{I}_{p}\end{bmatrix}}_{\mathbf{B}}\mathbf{w} = [(- 1) ] () + []$ | | (10) |
| --- | --- | --- | --- | --- |
| | w w \displaystyle $\mathbf{w}$ | ≔ L f r y (x) + L g L f r - 1 y (x) u, ≔ absent L f r y x L g L f r 1 y x u \displaystyle:= L_{ $\mathbf{f}}^{r}\mathbf{y}(\mathbf{x})+L_{\mathbf{g}}L_{\mathbf{f}}^{r-1}\mathbf{y}(\mathbf{x})\mathbf{u}, ≔ () + - 1 (),$ | | (11) |

**说明**: Table 11

#### Table 12: Table 12

| | u = L g L f r - 1 y (x) † [k ^ (y → (x)) - L f r y (x) ]. u L g L f r 1 y x † ^ k → y x L f r y x  $\mathbf{u}=L_{\mathbf{g}}L_{\mathbf{f}}^{r-1}\mathbf{y}(\mathbf{x})^{\dagger}\left[\hat{\mathbf{k}}(\vec{\mathbf{y}}(\mathbf{x}))-L_{\mathbf{f}}^{r}\mathbf{y}(\mathbf{x})\right]. = - 1 () † [(()) - () ].$ | | (12) |
| --- | --- | --- | --- |

**说明**: Table 12

#### Table 13: Table 13

| | y ≔ y (x) = (x, y, z) ∈ R 3. ≔ y y x x y z R 3 \displaystyle $\mathbf{y}:=\mathbf{y}(\mathbf{x})=(x,y,z)\in\mathbb{R}^{3}. ≔ () = (,,) ∈ 3.$ | | (13) |
| --- | --- | --- | --- |

**说明**: Table 13

#### Table 14: Table 14

| | C y ∈ ̄ \| h y ≥ C -set y ̄ h y \displaystyle $\mathcal{C}=\big{\{}\mathbf{y}\in\ {\Omega}\,\lVert \,h(\mathbf{y})\geq 0\big{\}}, = {∈ \rVert () ≥ 0 },$ | | (14a) |
| --- | --- | --- | --- |
| | ∂ C y ∈ ̄ \| h y C -set y ̄ h y \displaystyle\partial $\mathcal{C}=\big{\{}\mathbf{y}\in\ {\Omega}\,\big{\lVert}\,h(\mathbf{y})=0\big{\}}, ∂ = {∈ \rVert () = 0 },$ | | (14b) |
| | int C y ∈ ̄ \| h y . int C -set y ̄ h y \displaystyle $\mathrm{int}(\mathcal{C})=\big{\{}\mathbf{y}\in\ {\Omega}\,\big{\lVert}\,h(\mathbf{y})>0\big{\}}. () = {∈ \rVert () > 0 }.$ | | (14c) |

**说明**: Table 14

#### Table 15: Table 15

| | ∂ ⋃ i n obs ∂ i i n obs i \displaystyle\partial\Omega $\bigcup_{i=1}^{n_{\mathrm{obs}}}\partial\Gamma_{i}, ∂ = ⋃ = 1 ∂,$ | | (15) |
| --- | --- | --- | --- |

**说明**: Table 15

#### Table 16: Table 16

| | h y f y in h y on ∂ \displaystyle $\left\{\begin{aligned} \Delta h(\mathbf{y})&=f(\mathbf{y})&\text{in }\Omega,\\ h(\mathbf{y})&=0&\text{on }\partial\Omega,\\ \end{aligned}\right. {() = () in, () = 0 on ∂,$ | | (16) |
| --- | --- | --- | --- |

**说明**: Table 16

#### Table 17: Table 17

| | D h y n y on ∂ . D h y n y on \displaystyle Dh $\mathbf{y})\cdot\hat{\mathbf{n}}(\mathbf{y})<0\text{on }\partial\Omega. () ⋅ () < 0 on ∂.$ | | (17) |
| --- | --- | --- | --- |

**说明**: Table 17

#### Table 18: Table 18

| | dist y ∂ min y obs ∈ ∂ y - y obs dist y y obs y y obs \displaystyle $$ | | (18) |
| --- | --- | --- | --- |

**说明**: Table 18

#### Table 19: Table 19

| | f y - dist y ∂ dist y ∂ ∞ f y dist y dist y \displaystyle f $$ | | (19) |
| --- | --- | --- | --- |

**说明**: Table 19

#### Table 20: Table 20

| | ∫∫∫ f y d V triple-integral f y differential-d V \displaystyle\iiint \Omega f $\mathbf{y})\,\mathrm{d}V ∫∫∫ ()$ | ∫∫∫ h y d V triple-integral h y differential-d V \displaystyle \iiint \Omega \Delta h $\mathbf{y})\,\mathrm{d}V = ∫∫∫ ()$ | | (20) |
| --- | --- | --- | --- | --- |
| | | ∮∮ ∂ D h y n y d A surface-integral D h y n y differential-d A \displaystyle \oiint \partial\Omega Dh $\mathbf{y})\cdot\hat{\mathbf{n}}(\mathbf{y})\,\mathrm{d}A, = ∮∮ ∂ () ⋅ (),$ | | (21) |

**说明**: Table 20

#### Table 21: Table 21

| | b ̄ ≔ Area ∂ ∮∮ ∂ D h y n y d A ≔ ̄ b Area surface-integral D h y n y differential-d A \bar b $\frac{1}{\mathrm{Area}(\partial\Omega)}\oiint_{\partial\Omega}Dh(\mathbf{y})\cdot\hat{\mathbf{n}}(\mathbf{y})\,\mathrm{d}A, ≔ divide 1 (∂) ∮∮ ∂ () ⋅ (),$ | | (22) |
| --- | --- | --- | --- |

**说明**: Table 21

#### Table 22: Table 22

| | f ∮∮ ∂ D h y n y d A ∫∫∫ dV b ̄ Area ∂ Vol f surface-integral D h y n y differential-d A triple-integral dV ̄ b Area Vol f $\frac{\oiint_{\partial\Omega}Dh(\mathbf{y})\cdot\hat{\mathbf{n}}(\mathbf{y})\,\mathrm{d}A}{\iiint_{\Omega}\mathrm{dV}}=\bar{b}\ \frac{\mathrm{Area}(\partial\Omega)}{\mathrm{Vol}(\Omega)}, = divide ∮∮ ∂ () ⋅ () ∫∫∫ = divide (∂) (),$ | | (23) |
| --- | --- | --- | --- |

**说明**: Table 22

#### Table 23: Table 23

| | I h ∫∫∫ \| D h y - v → y \| d V I h triple-integral D h y → v y differential-d V \displaystyle I h \iiint \Omega \ $\frac{1}{2}\lVert Dh(\mathbf{y})-\vec{\mathbf{v}}(\mathbf{y})\rVert^{2}\ \mathrm{d}V, [] = ∫∫∫ divide 1 2 \lVert () - () \rVert 2,$ | | (24) |
| --- | --- | --- | --- |

**说明**: Table 23

#### Table 24: Table 24

| | h y ∇ v → y in h y on ∂ \displaystyle $\left\{\begin{aligned} \Delta h(\mathbf{y})&=\nabla\cdot\vec{\mathbf{v}}(\mathbf{y})&\text{in }\Omega,\\ h(\mathbf{y})&=0&\text{on }\partial\Omega,\\ \end{aligned}\right. {() = ∇ ⋅ () in, () = 0 on ∂,$ | | (25) |
| --- | --- | --- | --- |

**说明**: Table 24

#### Table 25: Table 25

| | v i y in v i y b y n i y on ∂ \displaystyle $\left\{\begin{aligned} \Delta v_{i}(\mathbf{y})&=0&\text{in }\Omega,\\ v_{i}(\mathbf{y})&=b(\mathbf{y})n_{i}(\mathbf{y})&\text{on }\partial\Omega,\end{aligned}\right. {() = 0 in, () = () () on ∂,$ | | (26) |
| --- | --- | --- | --- |

**说明**: Table 25

#### Table 26: Table 26

| | ∮∮ ∂ v → y n y d A surface-integral → v y n y differential-d A \displaystyle\oiint \partial\Omega \vec $\mathbf{v}}(\mathbf{y})\cdot\hat{\mathbf{n}}(\mathbf{y})\,\mathrm{d}A ∮∮ ∂$ | ∮∮ ∂ b y n y n y d A surface-integral b y n y n y differential-d A \displaystyle \oiint \partial\Omega b $\mathbf{y})\hat{\mathbf{n}}(\mathbf{y})\cdot\hat{\mathbf{n}}(\mathbf{y})\,\mathrm{d}A = ∮∮ ∂$ | | |
| --- | --- | --- | --- | --- |
| | | ∮∮ ∂ b y d A. surface-integral b y differential-d A \displaystyle \oiint \partial\Omega b $\mathbf{y})\,\mathrm{d}A. = ∮∮ ∂ ().$ | | (27) |

**说明**: Table 26

#### Table 27: Table 27

| | V v → ∈ C ∞ ̄ R \| v → y b y n y on ∂ . V -set → v C ̄ R → v y b y n y on \displaystyle $\mathcal{V}=\left\{\vec{\mathbf{v}}\in C^{\infty}(\ {\Omega};\mathbb{R}^{3})\,\big{\lVert}\,\vec{\mathbf{v}}(\mathbf{y})=b(\mathbf{y})\hat{\mathbf{n}}(\mathbf{y})\text{on }\partial\Omega\right\}. = {∈ ∞ (; 3) \rVert () = () () on ∂ }.$ | | (28) |
| --- | --- | --- | --- |

**说明**: Table 27

#### Table 28: Table 28

| | f y - ln e - ∇ v → y f y e ∇ → v y \displaystyle f $\mathbf{y})=-\frac{1}{\beta}\ln(1+e^{-\nabla\cdot\vec{\mathbf{v}}(\mathbf{y})\beta}), () = - divide 1 (1 + - ∇ ⋅ ()),$ | | (29) |
| --- | --- | --- | --- |

**说明**: Table 28

#### Table 29: Table 29

| | min h ∈ H J h ∫∫∫ \| D h \| - h ln e - ∇ v → d V h H J h triple-integral D h h e ∇ → v d V \displaystyle\!\min h\in $\mathcal{H}}\left\{\!J[h]\!=\!\!\iiint_{\Omega}\frac{1}{2}\lVert Dh \rVert^{2}\!-\!\frac{h}{\beta}\ln(1+e^{-\nabla\cdot\vec{\mathbf{v}}\beta})\mathrm{d}V\!\right\}, ∈ {[] = ∫∫∫ divide 1 2 \lVert \rVert 2 - divide (1 + - ∇ ⋅) },$ | | (30) |
| --- | --- | --- | --- |
| | H h ∈ C ∞ ̄ \| h on ∂ . H -set h C ̄ h on \displaystyle $\mathcal{H}=\big{\{}h\in C^{\infty}(\ {\Omega})\,\big{\lVert}\,h=0\text{on }\partial\Omega\big{\}}. = {∈ ∞ () \rVert = 0 on ∂ }.$ | | (31) |

**说明**: Table 29

#### Table 30: Table 30

| | y →  ̇ = w,  ̇ → y w \displaystyle $\dot{\vec{\mathbf{y}}}=\mathbf{w}, =,$ | | (32) |
| --- | --- | --- | --- |

**说明**: Table 30

#### Table 31: Table 31

| | h ̇ y → D h y → k y → ≥ - h y → ∀ y → ∈ C. formulae-sequence ̇ h → y D h → y k → y h → y for-all → y C \displaystyle $\dot{h}(\vec{\mathbf{y}})=Dh(\vec{\mathbf{y}})\cdot\mathbf{k}(\vec{\mathbf{y}})\geq-\gamma h(\vec{\mathbf{y}})\quad\forall\,\vec{\mathbf{y}}\in\mathcal{C}. () = () ⋅ () ≥ - () ∀ ∈.$ | | (33) |
| --- | --- | --- | --- |

**说明**: Table 31

#### Table 32: Table 32

| | k y → ≔ k y h y D h y h y D h y D h y a b b - a a b b b ≠ k → y ≔ k y h y D h y missing-subexpression h y D h y D h y top a b cases b a a b b b \begin aligned $$ | | (34) |
| --- | --- | --- | --- |

**说明**: Table 32

#### Table 33: Table 33

| | y →  ̇ = [y  ̇ y  ̈ ] = [y  ̇ w ].  ̇ → y matrix  ̇ y  ̈ y matrix  ̇ y w \displaystyle $\dot{\vec{\mathbf{y}}}=\begin{bmatrix}\dot{\mathbf{y}}\\ \ddot{\mathbf{y}}\end{bmatrix}=\begin{bmatrix}\dot{\mathbf{y}}\\ \mathbf{w}\end{bmatrix}. = [] = [].$ | | (35) |
| --- | --- | --- | --- |

**说明**: Table 33

#### Table 34: Table 34

| | h B y → h y - y ̇ - k y h y D h y h B → y h y ̇ y k y h y D h y \displaystyle h $$ | | (36) |
| --- | --- | --- | --- |

**说明**: Table 34

#### Table 35: Table 35

| | D h y k y h y D h y - h y D h y k y h y D h y h y \displaystyle Dh $\mathbf{y})\cdot\mathbf{k}_{1}(\mathbf{y},h(\mathbf{y}),Dh(\mathbf{y}))>-\gamma h(\mathbf{y}), () ⋅ 1 (, (), ()) > - (),$ | | (37) |
| --- | --- | --- | --- |

**说明**: Table 35

#### Table 36: Table 36

| | C B y → ∈ R \| h B y → ≥ ⊂ C R . C B -set → y R h B → y C R \displaystyle $\mathcal{C}_{\mathrm{B}}=\left\{\vec{\mathbf{y}}\in\mathbb{R}^{6}\,\big{\lVert}\,h_{\mathrm{B}}(\vec{\mathbf{y}})\geq 0\right\}\subset\mathcal{C}\times\mathbb{R}^{3}. = {∈ 6 \rVert () ≥ 0 } ⊂ × 3.$ | | (38) |
| --- | --- | --- | --- |

**说明**: Table 36

#### Table 37: Table 37

| | h ̇ B y → w D h y y ̇ - y ̇ - k w - k ̇ ̇ h B → y w D h y ̇ y ̇ y k top w ̇ k \displaystyle $\dot{h}_{\mathrm{B}}(\vec{\mathbf{y}},\mathbf{w})=Dh(\mathbf{y})\cdot\dot{\mathbf{y}}-\frac{1}{\mu_{1}}(\dot{\mathbf{y}}-\mathbf{k}_{1})^{\top}(\mathbf{w}-\dot{\mathbf{k}}_{1}), (,) = () ⋅ - divide 1 1 (- 1) ⊤ (- 1),$ | | (39) |
| --- | --- | --- | --- |

**说明**: Table 37

#### Table 38: Table 38

| | ∂ k 1 ∂ y = ∂ k 1 ∂ y + ∂ k 1 ∂ h ∂ h ∂ y + ∑ i ∈ {x, y, z } ∂ k 1 ∂ (∂ i h) ∂ (∂ i h) ∂ y. k 1 y k 1 y k 1 h h y i x y z k 1 i h i h y  $\frac{\partial\mathbf{k}_{1}}{\partial\mathbf{y}}=\frac{\partial\mathbf{k}_{1}}{\partial\mathbf{y}}+\frac{\partial\mathbf{k}_{1}}{\partial h}\frac{\partial h}{\partial\mathbf{y}}+\sum_{i\in\{x,y,z\}}\frac{\partial\mathbf{k}_{1}}{\partial(\partial_{i}h)}\frac{\partial(\partial_{i}h)}{\partial\mathbf{y}}. divide ∂ 1 ∂ = divide ∂ 1 ∂ + divide ∂ 1 ∂ divide ∂ ∂ + ∑ ∈ {,, } divide ∂ 1 ∂ (∂) divide ∂ (∂) ∂.$ | |
| --- | --- | --- |

**说明**: Table 38

#### Table 39: Table 39

| | h SDF y dist y ∂ y ∈ ̄ - dist y ∂ y ∉ ̄ h SDF y cases dist y y ̄ dist y y ̄ \displaystyle h $\mathrm{SDF}}(\mathbf{y})=\begin{cases}\mathrm{dist}(\mathbf{y},\partial\Omega),&\mathbf{y}\in\ {\Omega},\\ -\mathrm{dist}(\mathbf{y},\partial\Omega),&\mathbf{y}\notin\ {\Omega},\\ \end{cases} () = {(, ∂), ∈, - (, ∂), ∉,$ | | (40) |
| --- | --- | --- | --- |

**说明**: Table 39

#### Table 40: Table 40

| | ∂ h ∂ t ≈ h (t k, y) - h (t k - 1, y) t k - t k - 1. partial-derivative t h h t k y h t k 1 y t k t k 1 \displaystyle\partial{h}{t}\approx $\frac{h(t_{k},\mathbf{y})-h(t_{k-1},\mathbf{y})}{t_{k}-t_{k-1}}. divide ∂ ∂ ≈ divide (,) - (- 1,) - - 1.$ | | (41) |
| --- | --- | --- | --- |

**说明**: Table 40

#### Table 41: Table 41

| | h y f y in h y f y in \displaystyle\Delta h $\mathbf{y})=f(\mathbf{y})\quad\text{in }\Omega, () = () in,$ | | (42) |
| --- | --- | --- | --- |

**说明**: Table 41

#### Table 42: Table 42

| | h y f y in h y on ∂ . \displaystyle $\left\{\begin{aligned} \Delta h(\mathbf{y})&=f(\mathbf{y})&\text{in }\Omega,\\ h(\mathbf{y})&=0&\text{on }\partial\Omega.\\ \end{aligned}\right. {() = () in, () = 0 on ∂.$ | | (43) |
| --- | --- | --- | --- |

**说明**: Table 42

#### Table 43: Table 43

| | ∮∮ ∂ D h y n y d A surface-integral D h y n y differential-d A \displaystyle\oiint \partial\Omega Dh $\mathbf{y})\cdot\hat{\mathbf{n}}(\mathbf{y})\,\mathrm{d}A, ∮∮ ∂ () ⋅ (),$ | | (44) |
| --- | --- | --- | --- |

**说明**: Table 43

#### Table 44: Table 44

| | D h y n y lim → h y n y - h y D h y n y → h y n y h y \displaystyle Dh $\mathbf{y})\cdot\hat{\mathbf{n}}(\mathbf{y})=\lim_{\delta\rightarrow 0^{+}}\frac{h(\mathbf{y}+\delta\hat{\mathbf{n}}(\mathbf{y}))-h(\mathbf{y})}{\delta}, () ⋅ () = → 0 + divide (+ ()) - (),$ | | (45) |
| --- | --- | --- | --- |

**说明**: Table 44

#### Table 45: Table 45

| | ∫∫∫ h y d V ∮∮ ∂ D h y n y d A triple-integral h y differential-d V surface-integral D h y n y differential-d A \displaystyle\iiint \Omega \Delta h $\mathbf{y})\,\mathrm{d}V=\oiint_{\partial\Omega}Dh(\mathbf{y})\cdot\hat{\mathbf{n}}(\mathbf{y})\,\mathrm{d}A, ∫∫∫ () = ∮∮ ∂ () ⋅ (),$ | | (46) |
| --- | --- | --- | --- |

**说明**: Table 45

#### Table 46: Table 46

| | h y ≥ in ⟹ max y ∈ ̄ h y max y ∈ ∂ h y h y in y ̄ h y y h y \displaystyle\Delta h $\mathbf{y})\geq 0\text{in }\Omega\implies\max_{\mathbf{y}\in\ {\Omega}}h(\mathbf{y})=\max_{\mathbf{y}\in\partial\Omega}h(\mathbf{y}), () ≥ 0 in ⟹ ∈ () = ∈ ∂ (),$ | | (47) |
| --- | --- | --- | --- |
| | h y ≤ in ⟹ min y ∈ ̄ h y min y ∈ ∂ h y . h y in y ̄ h y y h y \displaystyle\Delta h $\mathbf{y})\leq 0\text{in }\Omega\implies\min_{\mathbf{y}\in\ {\Omega}}h(\mathbf{y})=\min_{\mathbf{y}\in\partial\Omega}h(\mathbf{y}). () ≤ 0 in ⟹ ∈ () = ∈ ∂ ().$ | | (48) |

**说明**: Table 46

#### Table 47: Table 47

| | h y max y ∈ ̄ h y ⟹ D h y n y ≥ h y y ̄ h y D h y n y \displaystyle h $\mathbf{y}_{0})=\max_{\mathbf{y}\in\ {\Omega}}h(\mathbf{y})\implies Dh(\mathbf{y}_{0})\cdot\hat{\mathbf{n}}(\mathbf{y}_{0})\geq 0, (0) = ∈ () ⟹ (0) ⋅ (0) ≥ 0,$ | | (49) |
| --- | --- | --- | --- |
| | h y min y ∈ ̄ h y ⟹ D h y n y ≤ h y y ̄ h y D h y n y \displaystyle h $\mathbf{y}_{0})=\min_{\mathbf{y}\in\ {\Omega}}h(\mathbf{y})\implies Dh(\mathbf{y}_{0})\cdot\hat{\mathbf{n}}(\mathbf{y}_{0})\leq 0, (0) = ∈ () ⟹ (0) ⋅ (0) ≤ 0,$ | | (50) |

**说明**: Table 47

#### Table 48: Table 48

| | \| f y - f y \| ≤ M y - y ∀ y y ∈ . formulae-sequence f y f y M y y for-all y y \|f $$ | |
| --- | --- | --- |

**说明**: Table 48

#### Table 49: Table 49

| | min h ∈ H J h ∫∫∫ \| D h y \| h y f y d V h H J h triple-integral D h y h y f y d V \displaystyle\min h\in $\mathcal{H}}\left\{J[h]=\iiint_{\Omega}\ \frac{1}{2}\lVert Dh(\mathbf{y})\rVert^{2}+h(\mathbf{y})f(\mathbf{y})\ \mathrm{d}V\right\}, ∈ {[] = ∫∫∫ divide 1 2 \lVert () \rVert 2 + () () },$ | | (51) |
| --- | --- | --- | --- |

**说明**: Table 49

#### Table 50: Table 50

| | H h ∈ C ̄ \| h y on ∂ . H -set h C ̄ h y on \displaystyle $\mathcal{H}=\{h\in C^{2}(\ {\Omega})\,\big{\lVert}\,h(\mathbf{y})=0\text{on }\partial\Omega\}. = {∈ 2 () \rVert () = 0 on ∂ }.$ | | (52) |
| --- | --- | --- | --- |

**说明**: Table 50

#### Table 51: Table 51

| | I [h ] I h \displaystyle I[h] [] | ∫∫∫ \| D h - v → \| d V triple-integral D h → v differential-d V \displaystyle \iiint \Omega \ $\frac{1}{2}\lVert Dh-\vec{\mathbf{v}} \rVert^{2}\ \mathrm{d}V = ∫∫∫ divide 1 2 \lVert - \rVert 2$ | | (53) |
| --- | --- | --- | --- | --- |
| | | ∫∫∫ \| D h \| - D h v → \| v → \| d V triple-integral D h D h → v → v d V \displaystyle \iiint \Omega \ $\frac{1}{2}\lVert Dh \rVert^{2}-Dh\cdot\vec{\mathbf{v}}+\frac{1}{2}\lVert \vec{\mathbf{v}} \rVert^{2}\ \mathrm{d}V = ∫∫∫ divide 1 2 \lVert \rVert 2 - ⋅ + divide 1 2 \lVert \rVert 2$ | | (54) |
| | | ∫∫∫ \| D h \| h ∇ v → \| v → \| d V triple-integral D h h ∇ → v → v d V \displaystyle \iiint \Omega \ $\frac{1}{2}\lVert Dh \rVert^{2}+h\nabla\cdot\vec{\mathbf{v}}+\frac{1}{2}\lVert \vec{\mathbf{v}} \rVert^{2}\ \mathrm{d}V = ∫∫∫ divide 1 2 \lVert \rVert 2 + ∇ ⋅ + divide 1 2 \lVert \rVert 2$ | | (55) |

**说明**: Table 51

#### Table 52: Table 52

| | ∫∫∫ D h v → d V triple-integral D h → v differential-d V \displaystyle\iiint \Omega Dh\cdot\vec $\mathbf{v}}\ \mathrm{d}V ∫∫∫ ⋅$ | ∫∫∫ ∇ h v → - h ∇ v → d V triple-integral ∇ h → v h ∇ → v d V \displaystyle \iiint \Omega \nabla\cdot h\vec $\mathbf{v}})-h\nabla\cdot\vec{\mathbf{v}}\mathrm{d}V = ∫∫∫ ∇ ⋅ () - ∇ ⋅$ | | (56) |
| --- | --- | --- | --- | --- |
| | | ∮∮ ∂ h v → n d A - ∫∫∫ h ∇ v → d V surface-integral h → v n differential-d A triple-integral h ∇ → v differential-d V \displaystyle \oiint \partial\Omega h\vec $\mathbf{v}}\cdot\hat{\mathbf{n}}\ \mathrm{d}A-\iiint_{\Omega}h\nabla\cdot\vec{\mathbf{v}}\ \mathrm{d}V = ∮∮ ∂ ⋅ - ∫∫∫ ∇ ⋅$ | | (57) |
| | | - ∫∫∫ h ∇ v → d V. triple-integral h ∇ → v differential-d V \displaystyle -\iiint \Omega h\nabla\cdot\vec $\mathbf{v}}\ \mathrm{d}V. = - ∫∫∫ ∇ ⋅.$ | | (58) |

**说明**: Table 52

#### Table 53: Table 53

| | J h ∫∫∫ \| D h y \| h y ∇ v → y d V J h triple-integral D h y h y ∇ → v y d V \displaystyle J h \iiint \Omega \ $\frac{1}{2}\lVert Dh(\mathbf{y})\rVert^{2}+h(\mathbf{y})\nabla\cdot\vec{\mathbf{v}}(\mathbf{y})\ \mathrm{d}V, [] = ∫∫∫ divide 1 2 \lVert () \rVert 2 + () ∇ ⋅ (),$ | | (59) |
| --- | --- | --- | --- |

**说明**: Table 53

#### Table 54: Table 54

| | h B y → h y - ∑ i r - i y i - k i h B → y h y i r i y i k i \displaystyle h $$ | (y, ⋯, y (i - 1), h (y), \displaystyle $\Big{(}\mathbf{y},\cdots,\mathbf{y}^{(i-1)},h(\mathbf{y}), (, ⋯, (- 1), (),$ | | |
| --- | --- | --- | --- | --- |
| | | D h y ⋯ D i h y \displaystyle\ Dh $$ | | (60) |

**说明**: Table 54

#### Table 55: Table 55

| | k ̇ i ̇ k i \displaystyle $\dot{\mathbf{k}}_{i}(\cdot)$ | ∑ j i ∂ k i ∂ y j - y j j i k i y j y j \displaystyle $\sum_{j=1}^{i}\frac{\partial\mathbf{k}_{i}}{\partial\mathbf{y}^{(j-1)}}(\cdot)\mathbf{y}^{(j)} = ∑ = 1 divide ∂ ∂$ | | (61) |
| --- | --- | --- | --- | --- |
| | | i y ⋯ y i h y D h y ⋯ D i h y . i y ⋯ y i h y D h y ⋯ D i h y \displaystyle \Phi i $\mathbf{y},\cdots,\mathbf{y}^{(i)},h(\mathbf{y}),Dh(\mathbf{y}),\cdots,D^{i+1}h(\mathbf{y})). = (, ⋯, (), (), (), ⋯, + 1 ()).$ | | (62) |

**说明**: Table 55

#### Table 56: Table 56

| | C B y → ∈ R r \| h B y → ≥ ⊂ C R r - C B -set → y R r h B → y C R r \displaystyle $\mathcal{C}_{\mathrm{B}}=\left\{\vec{\mathbf{y}}\in\mathbb{R}^{3r}\,\big{\lVert}\,h_{\mathrm{B}}(\vec{\mathbf{y}})\geq 0\right\}\subset\mathcal{C}\times\mathbb{R}^{3(r-1)}, = {∈ 3 \rVert () ≥ 0 } ⊂ × 3 (- 1),$ | | (63) |
| --- | --- | --- | --- |

**说明**: Table 56

#### Table 57: Table 57

| | h ̇ B y → D h B y → A y → Bk y → ≥ - h B y → ̇ h B → y D h B → y A → y Bk → y h B → y \displaystyle $\dot{h}_{\mathrm{B}}(\vec{\mathbf{y}})=Dh_{\mathrm{B}}(\vec{\mathbf{y}})\cdot\big{(}\mathbf{A}\vec{\mathbf{y}}+\mathbf{B}\mathbf{k}(\vec{\mathbf{y}})\big{)}\geq-\gamma h_{\mathrm{B}}(\vec{\mathbf{y}})$ | | (64) |
| --- | --- | --- | --- |

**说明**: Table 57

#### Table 58: Table 58

| | h  ̇ B (y →, w)  ̇ h B → y w \displaystyle $\dot{h}_{\mathrm{B}}(\vec{\mathbf{y}},\mathbf{w})$ | D h B y → A y → Bw D h B → y A → y Bw \displaystyle Dh $\mathrm{B}}(\vec{\mathbf{y}})\cdot\big{(}\mathbf{A}\vec{\mathbf{y}}+\mathbf{B}\mathbf{w}\big{)} =$ | | (65) |
| --- | --- | --- | --- | --- |
| | | ≔ D h y y ̇ - ∑ i r - i y i - k i y i - k ̇ i ≔ D h y ̇ y i r i y i k i top y i ̇ k i \displaystyle Dh $\mathbf{y})\cdot\dot{\mathbf{y}}-\sum_{i=1}^{r-2}\frac{1}{\mu_{i}}(\mathbf{y}^{(i)}-\mathbf{k}_{i})^{\top}(\mathbf{y}^{(i+1)}-\dot{\mathbf{k}}_{i}) ≔$ | | |
| | | - r - y r - - k i w - k ̇ r - r y r k i top w ̇ k r \displaystyle\qquad\qquad- $\frac{1}{\mu_{r-1}}(\mathbf{y}^{(r-1)}-\mathbf{k}_{i})^{\top}(\mathbf{w}-\dot{\mathbf{k}}_{r-1}), - divide 1 - 1 ((- 1) -) ⊤ (- - 1),$ | | |

**说明**: Table 58

#### Table 59: Table 59

| | k ̇ r - ̇ k r \displaystyle $\dot{\mathbf{k}}_{r-1}(\cdot) - 1$ | ∑ j r - ∂ k r - ∂ y j - y j j r k r y j y j \displaystyle $\sum_{j=1}^{r-1}\frac{\partial\mathbf{k}_{r-1}}{\partial\mathbf{y}^{(j-1)}}(\cdot)\mathbf{y}^{j} = ∑ = 1 - 1 divide ∂ - 1 ∂$ | | (66) |
| --- | --- | --- | --- | --- |
| | | r - y → h y D h y ⋯ D r h y . r → y h y D h y ⋯ D r h y \displaystyle \Phi r- $\Big{(}{\vec{\mathbf{y}}},h(\mathbf{y}),Dh(\mathbf{y}),\cdots,D^{r}h(\mathbf{y})\Big{)}. = - 1 (, (), (), ⋯, ()).$ | | (67) |

**说明**: Table 59

#### Table 60: Table 60

| | h 0 h 0 \displaystyle h_{0} 0 | ≜ h, ≜ absent h \displaystyle:= h, ≜, | | (68) |
| --- | --- | --- | --- | --- |
| | h i h i \displaystyle h_{i} | ≜ h ̇ i - i h i - i ⋯ r - formulae-sequence ≜ ̇ h i i h i i ⋯ r \displaystyle $\dot{h}_{i-1}+\gamma_{i}h_{i-1},\quad\quad i=1,\cdots,r-1 ≜ - 1 + - 1, = 1, ⋯, - 1$ | | (69) |

**说明**: Table 60

#### Table 61: Table 61

| | h i (y ~) h i ~ y \displaystyle h_{i}( $\tilde{\mathbf{y}}) ()$ | = h i (y, ⋯, y (i), h (y), D h (y), ⋯, D i h (y)) absent h i y ⋯ y i h y D h y ⋯ D i h y \displaystyle=h_{i}( $\mathbf{y},\cdots,\mathbf{y}^{(i)},h(\mathbf{y}),Dh(\mathbf{y}),\cdots,D^{i}h(\mathbf{y})) =$ | | (70) |
| --- | --- | --- | --- | --- |
| | h  ̇ i (y ~)  ̇ h i ~ y \displaystyle $\dot{h}_{i}(\tilde{\mathbf{y}}) ()$ | = ∑ j = 1 i + 1 ∂ h i ∂ y (j - 1) (y ~) y (j) absent j 1 i 1 h i y j 1 ~ y y j \displaystyle= $\sum_{j=1}^{i+1}\frac{\partial h_{i}}{\partial\mathbf{y}^{(j-1)}}(\tilde{\mathbf{y}})\mathbf{y}^{(j)} = ∑ = 1 + 1 divide ∂ ∂$ | | (71) |
| | | = h  ̇ i (y, ⋯, y (i + 1), h (y), D h (y), ..., D i + 1 h (y)) absent  ̇ h i y ⋯ y i 1 h y D h y ... D i 1 h y \displaystyle= $\dot{h}_{i}(\mathbf{y},\cdots,\mathbf{y}^{(i+1)},h(\mathbf{y}),Dh(\mathbf{y}),...,D^{i+1}h(\mathbf{y})) =$ | | (72) |

**说明**: Table 61

#### Table 62: Table 62

| | C i = {y ~ ∈ R 3 i + 1 \| h i (y ~) ≥ 0 } i = 1, ⋯, r - 1 formulae-sequence C i conditional-set ~ y R 3 i 1 h i ~ y 0 i 1 ⋯ r 1  $\mathcal{C}_{i}=\left\{\tilde{\mathbf{y}}\in\mathbb{R}^{3{i+1}}\,\big{\lVert}\,h_{i}(\tilde{\mathbf{y}})\geq 0\right\}\quad i=1,\cdots,r-1 = {∈ 3 + 1 \rVert () ≥ 0 } = 1, ⋯, - 1$ | |
| --- | --- | --- |

**说明**: Table 62

#### Table 63: Table 63

| | C H ≔ ⋂ i = 0 r - 1 C i ⊂ C, ≔ C H i 0 r 1 C i C \displaystyle $\mathcal{C}_{\mathrm{H}}:=\bigcap_{i=0}^{r-1}\mathcal{C}_{i}\subset\mathcal{C}, ≔ ⋂ = 0 - 1 ⊂,$ | | (73) |
| --- | --- | --- | --- |

**说明**: Table 63

#### Table 64: Table 64

| | h H (y →) h H → y \displaystyle h_{ $\mathrm{H}}(\vec{\mathbf{y}}) ()$ | ≜ h r - 1 (y →) ≜ absent h r 1 → y \displaystyle:= h_{r-1}(\vec{ $\mathbf{y}}) ≜ - 1 ()$ | | (74) |
| --- | --- | --- | --- | --- |
| | | = h r - 1 (y →, h (y), D h (y), ⋯, D r - 1 h (y)) absent h r 1 → y h y D h y ⋯ D r 1 h y \displaystyle=h_{r-1}(\vec{ $\mathbf{y}},h(\mathbf{y}),Dh(\mathbf{y}),\cdots,D^{r-1}h(\mathbf{y})) = - 1$ | | |

**说明**: Table 64

#### Table 65: Table 65

| | h ̇ H y → D h H y → A y → Bk y → ≥ - h H y → ̇ h H → y D h H → y A → y Bk → y h H → y \displaystyle $\dot{h}_{\mathrm{H}}(\vec{\mathbf{y}})=Dh_{\mathrm{H}}(\vec{\mathbf{y}})\cdot\big{(}\mathbf{A}\vec{\mathbf{y}}+\mathbf{B}\mathbf{k}(\vec{\mathbf{y}})\big{)}\geq-\gamma h_{\mathrm{H}}(\vec{\mathbf{y}})$ | | (75) |
| --- | --- | --- | --- |

**说明**: Table 65

#### Table 66: Table 66

| | D h H (y →) B = D h (y), D h H → y B D h y \displaystyle Dh_{ $\mathrm{H}}(\vec{\mathbf{y}})\mathbf{B}=Dh(\mathbf{y}), () = (),$ | | (76) |
| --- | --- | --- | --- |

**说明**: Table 66
## 实验解读

- 评价重点:围绕 certified-control、人形机器人、模仿学习、足式运动、navigation,优先看真实机器人结果、扰动恢复、接触稳定性、控制误差、成功率、实时性和 sim-to-real 差距。
- System0 视角:实验结论不只看最终成功率,还要看闭环过程中是否能处理状态估计误差、执行延迟、碰撞/接触突变和外部扰动。
- 与上层系统的关系:如果方法能在快闭环内稳定工作,它可以作为 System1 技能或 System2 规划的底层执行保障;如果只在离线或慢速设定下成立,则还不能直接作为身体级快闭环模块。
## 批判性思考

### 优点

- 论文把 certified-control、人形机器人、模仿学习、足式运动、navigation 放到机器人身体级闭环里讨论,和只做高层语义推理的工作相比,更接近真实机器人部署时会遇到的状态、动力学和安全约束。
- 若论文同时给出公式、系统结构图、硬件实验或消融结果,这些证据可以帮助判断方法是否真的能支撑毫秒到百毫秒级反应。

### 局限

- 需要重点核查控制频率、推理延迟、传感延迟、失败恢复、硬件磨损、未见场景泛化和安全边界;这些因素决定它能否成为可靠的 System0 模块。
- 对学习型方法,还要区分训练环境中的鲁棒性和真实部署中的鲁棒性,尤其是接触丰富任务、动态运动任务和人形机器人任务。
## 论文定位

- 研究对象:Dynamic Safety in Complex Environments: Synthesizing Safety Filters with Poisson’s Equation。
- 关键词:certified-control、人形机器人、模仿学习、足式运动、navigation、quadruped、实时控制、safe-control、安全过滤。
- System0 关联:关注机器人身体级快反应链路中“状态/接触/局部感知 -> 约束/优化/策略 -> 动作”的短闭环,不把问题停留在语言规划或任务分解层。

## 速查卡片

> [!summary] Dynamic Safety Complex
> - **论文**: https://www.roboticsproceedings.org/rss21/p137.pdf
> - **arXiv**: http://arxiv.org/abs/2505.06794v1
> - **arXiv HTML**: https://arxiv.org/html/2505.06794v1
