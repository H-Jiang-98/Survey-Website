---
title: "Real-Time Execution of Action Chunking Flow Policies"
method_name: "RTC"
authors: [Kevin Black, Manuel Y. Galliker, Sergey Levine]
year: 2025
venue: NeurIPS 2025
tags: [real-time-control, action-chunking, vla, flow-matching, robot-manipulation, inference-time-algorithm]
zotero_collection: _inbox
image_source: online
arxiv_html: https://arxiv.org/html/2506.07339v2
created: 2026-06-25
---

# RTC

## 一句话结论

[[Real-Time Chunking|RTC]] 把高延迟 [[Action Chunking|动作块]] 策略的执行改成异步 inpainting：执行当前 chunk 的同时生成下一个 chunk，冻结必然会被执行的旧动作，并用 [[Soft Masking|soft mask]] 让新 chunk 平滑衔接旧 chunk。

## 论文定位

这篇论文解决的是 [[Vision-Language-Action Model|VLA]] 和 [[Flow Matching|flow-based policy]] 上机时的实时性问题。当前大模型策略一次生成一个 action chunk，但推理常常比控制周期慢很多；同步执行会在 chunk 边界停顿，朴素异步执行又容易跳到另一个 action mode，导致 jerk 和 out-of-distribution dynamics。

RTC 的关键点是它不是重新训练模型，而是一个纯推理时算法：只要底层策略是 [[Diffusion Policy|diffusion]] 或 flow-based action chunk policy，就可以通过 [[Inference-Time Inpainting|inference-time inpainting]] 改造执行过程。

## 核心贡献

1. **实时 action chunk 执行算法**: 在后台生成下一个 chunk，保证每个控制周期都有动作可执行。
2. **把跨 chunk 连续性写成 inpainting**: 对已经确定会执行的旧动作施加强约束，对仍可更新的重叠动作施加软约束。
3. **soft masking 与 guidance clipping**: 用指数衰减 mask 保持策略连续性，用 $\beta$ 限制 guidance weight，避免少步 denoising 下动作发散。
4. **动态仿真 benchmark**: 在 [[Kinetix]] 中构造 12 个高动态任务，用来测试延迟下的 chunked policy。
5. **真实双臂实验**: 在 6 个真实双臂操作任务上测试 [[Pi05|$\pi_{0.5}$]] 基策略，RTC 在 +100ms、+200ms 注入延迟下仍保持 throughput。

## 背景问题

### 要解决的问题

大模型机器人策略的推理延迟通常高于低层控制周期。例如论文报告 [[Pi05|$\pi_{0.5}$]] 的模型推理约 76ms，而真实系统还有网络和图像 resize 开销；控制频率是 50Hz，即 $\Delta t=20$ms。若每次 chunk 用完才开始推理，下一个 chunk 会迟到，机器人会在 chunk 边界停顿。

### 现有方法的局限

- [[Action Chunking]] 能提高时序一致性，但 chunk 越长，闭环纠错越慢。
- 同步推理会引入停顿，改变真实动力学。
- 朴素异步推理可以消除停顿，但新旧 chunk 可能来自不同动作模式，边界处出现不合理加速度。
- [[Temporal Ensembling]] 对多峰动作分布不可靠，平均两个合法动作不一定还是合法动作。
- [[Bidirectional Decoding]] 能做 continuity-aware sampling，但需要采样多个 action chunk，计算更贵。

### 本文动机

旧 chunk 中有一段未来动作和新 chunk 的时间轴重叠。RTC 的判断是：这些重叠动作就是 inpainting 的条件。新 chunk 不应从零生成，而应在保留旧 chunk 已确定部分的条件下补全剩余动作。

## 方法详解

### 基本设定

动作块策略记为 $\pi(\mathbf{A}_t\mid\mathbf{o}_t)$，输入当前 observation $\mathbf{o}_t$，输出未来 $H$ 步动作：

- $\mathbf{A}_t = (\mathbf{a}_{t|t},\ldots,\mathbf{a}_{t+H-1|t})$：在时刻 $t$ 生成的 action chunk。
- $H$：prediction horizon。
- $s$：execution horizon，即一个 chunk 被执行多少步后切换。
- $d$：inference delay，以控制步数计。
- $\Delta t$：控制周期。
- $\delta$：生成一个 action chunk 的实际 wall-clock 延迟。
- $n$：flow/diffusion denoising steps。

普通 action chunking 的做法是：生成一个 chunk，执行前 $s$ 步，再生成下一个 chunk。问题是现代 [[Vision-Language-Action Model|VLA]] 的推理延迟 $\delta$ 往往大于控制周期 $\Delta t$，同步等待会在 chunk 边界造成停顿。RTC 的做法是 [[Asynchronous Inference|异步推理]]：当前 chunk 还在执行时，就用最新 observation 启动后台推理。

关键量是

$$
d=\left\lfloor \frac{\delta}{\Delta t}\right\rfloor
$$

也就是从收到 observation 到新 chunk 可用之间，会继续执行多少个控制步。若希望新 chunk 在执行了 $s$ 步旧 chunk 后切上来，就必须在 $s-d$ 附近开始推理；同时为了保证新旧 chunk 有足够重叠，需要满足 $d\leq H-s$，也就是 $d\leq s\leq H-d$。

真正难点不是“提前算”本身，而是提前算出来的新 chunk 不能和旧 chunk 的未来计划冲突。旧 chunk 在切换点附近已经承诺了一段未来动作，如果新 chunk 走到另一个 mode，会在边界产生高加速度或 out-of-distribution jerk。RTC 把这个兼容性问题改写成 [[Inference-Time Inpainting|inference-time inpainting]]：旧 chunk 的剩余动作是条件，新 chunk 要在这些条件下补全。

### 时间轴对齐

设后台推理启动时，当前 chunk 已经执行了 $s$ 步，则旧 chunk 还剩

$$
\mathbf{A}_{\text{prev}}
=
(\mathbf{a}_{s|0},\mathbf{a}_{s+1|0},\ldots,\mathbf{a}_{H-1|0})
$$

这段长度是 $H-s$。它和新 chunk 的前 $H-s$ 个位置在物理时间上对齐：

- 新 chunk 的前 $d$ 个动作：推理完成前已经被旧 chunk 实际执行，因此必须被视为 frozen。
- 新 chunk 的 $d$ 到 $H-s-1$ 个动作：和旧 chunk 仍有时间重叠，但还没执行，可以被新 observation 修正，只是应尽量保持连续。
- 新 chunk 的最后 $s$ 个动作：超出旧 chunk 预测范围，没有旧动作可参考，必须自由生成。

Algorithm 1 里把 $\mathbf{A}_{\text{prev}}$ right-pad 到长度 $H$，再用 mask $\mathbf{W}$ 控制哪些位置对 guidance 有影响。padding 区域对应最后 $s$ 个新动作，权重为 0，所以不会约束生成。

### Phase 1: Flow policy 生成候选动作块

论文以 [[Flow Matching]] policy 为主，[[Diffusion Policy]] 可以在推理时转成等价的 flow policy。采样从高斯噪声 $\mathbf{A}^{0}_{t}\sim\mathcal{N}(\mathbf{0},\mathbf{I})$ 出发，沿 learned velocity field 积分到 $\tau=1$：

$$
\mathbf{A}_{t}^{\tau+\frac{1}{n}}
=
\mathbf{A}_{t}^{\tau}
+
\frac{1}{n}\mathbf{v}_{\pi}(\mathbf{A}_{t}^{\tau},\mathbf{o}_{t},\tau)
$$

如果只做这一步，就是普通异步 action chunking：新 observation 会被利用，但新 chunk 与旧 chunk 的重叠部分没有任何一致性约束。Figure 2 展示的 mode jump 就来自这里：旧 chunk 计划绕障碍上方，新 chunk 可能根据同一个多峰分布采到绕下方，边界处会突然切换策略。

### Phase 2: Inference-time inpainting

RTC 借鉴 [[Pseudoinverse Guidance]] / $\Pi$GDM，把“新 chunk 要匹配旧 chunk 的重叠部分”写成一个 guidance 项。设：

- $\mathbf{Y}$：目标条件，在 RTC 中就是 right-padded 后的旧 chunk 剩余动作 $\mathbf{A}_{\text{prev}}$。
- $\mathbf{W}$：每个动作位置的约束权重，展开后可看作 $\operatorname{diag}(\mathbf{W})$。
- $\widehat{\mathbf{A}^{1}_{t}}$：从当前 noisy action chunk $\mathbf{A}^{\tau}_{t}$ 一步估计出的最终 denoised chunk。

在每个 denoising step，RTC 先用 base velocity field 得到最终 chunk 的近似：

$$
\widehat{\mathbf{A}^{1}_{t}}
=
\mathbf{A}^{\tau}_{t}
+
(1-\tau)\mathbf{v}(\mathbf{A}^{\tau}_{t},\mathbf{o}_{t},\tau)
$$

然后计算“如果现在继续 denoise，最终结果和旧 chunk 条件差多少”：

$$
(\mathbf{Y}-\widehat{\mathbf{A}^{1}_{t}})^\top\operatorname{diag}(\mathbf{W})
$$

这个误差不是直接加到动作上，而是通过 $\widehat{\mathbf{A}^{1}_{t}}$ 对当前中间变量 $\mathbf{A}^{\tau}_{t}$ 的 Jacobian 反传回来，形成一个 vector-Jacobian product：

$$
\mathbf{g}
=
(\mathbf{Y}-\widehat{\mathbf{A}^{1}_{t}})^\top
\operatorname{diag}(\mathbf{W})
\frac{\partial\widehat{\mathbf{A}^{1}_{t}}}{\partial\mathbf{A}^{\tau}_{t}}
$$

这个 Jacobian 可以具体展开。把 action chunk 展平成一个向量：

$$
\mathbf{z}
=
\operatorname{vec}(\mathbf{A}^{\tau}_{t})
\in\mathbb{R}^{D},
\quad
D=H M
$$

其中 $M$ 是单步 action 维度。定义当前 denoising step 的“一步最终估计”函数：

$$
f_{\tau}(\mathbf{z})
=
\mathbf{z}
+
(1-\tau)\mathbf{v}_{\pi}(\mathbf{z},\mathbf{o}_{t},\tau)
$$

那么：

$$
\widehat{\mathbf{A}^{1}_{t}}
=
f_{\tau}(\mathbf{z})
$$

它对当前中间变量的 Jacobian 是：

$$
\frac{\partial\widehat{\mathbf{A}^{1}_{t}}}{\partial\mathbf{A}^{\tau}_{t}}
=
J_{f_{\tau}}(\mathbf{z})
=
\mathbf{I}_{D}
+
(1-\tau)
\frac{\partial\mathbf{v}_{\pi}(\mathbf{z},\mathbf{o}_{t},\tau)}{\partial\mathbf{z}}
$$

按坐标写就是：

$$
\frac{\partial \widehat{z}^{1}_{i}}{\partial z_{j}}
=
\mathbb{1}[i=j]
+
(1-\tau)
\frac{\partial v_{\pi,i}}{\partial z_j}
$$

如果保留时间步和 action 维度索引，等价于：

$$
\frac{\partial \widehat{a}^{1}_{k,m}}{\partial a^{\tau}_{\ell,n}}
=
\mathbb{1}[k=\ell,\,m=n]
+
(1-\tau)
\frac{\partial v_{\pi,k,m}}{\partial a^{\tau}_{\ell,n}}
$$

这里的第一项 $\mathbf{I}_D$ 来自 $\widehat{\mathbf{A}^{1}_{t}}$ 里直接保留的 $\mathbf{A}^{\tau}_{t}$；第二项来自 velocity field 网络本身。由于 $\mathbf{v}_{\pi}$ 通常一次读取整个 action chunk，$\frac{\partial v_{\pi,k,m}}{\partial a^{\tau}_{\ell,n}}$ 一般不是对角的：一个时间步或一个关节维度的 noisy action 变化，可能影响其他时间步/维度的 velocity 预测。

为什么 guidance 里需要这个 Jacobian？因为约束误差是在 **最终动作块空间** 里定义的，但 denoising step 真正能更新的是当前的 **中间 noisy chunk**。RTC 想让最终估计 $f_\tau(\mathbf{z})=\widehat{\mathbf{A}^{1}_{t}}$ 接近旧 chunk 条件 $\mathbf{Y}$，可以把它看成在当前 $\tau$ 上最小化一个加权 reconstruction loss：

$$
L(\mathbf{z})
=
\frac{1}{2}
\left\|
\operatorname{diag}(\mathbf{W})^{1/2}
\left(f_\tau(\mathbf{z})-\mathbf{Y}\right)
\right\|_2^2
$$

对当前变量 $\mathbf{z}=\operatorname{vec}(\mathbf{A}^{\tau}_{t})$ 求梯度，有：

$$
\nabla_{\mathbf{z}}L
=
J_{f_\tau}(\mathbf{z})^\top
\operatorname{diag}(\mathbf{W})
\left(f_\tau(\mathbf{z})-\mathbf{Y}\right)
$$

所以让最终动作更接近旧 chunk 的方向是负梯度：

$$
-\nabla_{\mathbf{z}}L
=
J_{f_\tau}(\mathbf{z})^\top
\operatorname{diag}(\mathbf{W})
\left(\mathbf{Y}-\widehat{\mathbf{A}^{1}_{t}}\right)
$$

如果不乘 Jacobian，就相当于默认 $f_\tau$ 是恒等映射，即“当前 noisy chunk 的第 $i$ 个分量只影响最终 chunk 的第 $i$ 个分量”。但真实的 $f_\tau$ 还包含神经网络 velocity field，当前变量如何影响最终估计由 $J_{f_\tau}$ 决定。Jacobian 的作用就是把“最终 chunk 应该怎么变”的误差，转换成“当前 noisy chunk 应该往哪里改”的方向。

因此，若用列向量记法，把 weighted error 写成：

$$
\mathbf{e}
=
\operatorname{diag}(\mathbf{W})
(\mathbf{Y}-\widehat{\mathbf{A}^{1}_{t}})
$$

则 guidance 方向可写成：

$$
\mathbf{g}
=
J_{f_{\tau}}(\mathbf{z})^{\top}\mathbf{e}
=
\left[
\mathbf{I}_{D}
+
(1-\tau)
\frac{\partial\mathbf{v}_{\pi}}{\partial\mathbf{z}}
\right]^{\top}
\operatorname{diag}(\mathbf{W})
(\mathbf{Y}-\widehat{\mathbf{A}^{1}_{t}})
$$

这就是“把最终 chunk 坐标系里的误差反传到当前 noisy chunk 坐标系”。它等价于沿着 weighted reconstruction error 的负梯度方向走一步，所以当 $f_\tau$ 近似恒等映射时，$\mathbf{g}$ 就退化成简单的加权误差 $\operatorname{diag}(\mathbf{W})(\mathbf{Y}-\mathbf{A}^{\tau}_{t})$。

实现时不会显式构造这个 $D\times D$ Jacobian。Algorithm 1 的 line 29 实际上只需要 vector-Jacobian product，可以用 autodiff 直接算：

```python
z.requires_grad_(True)
v = v_pi(z, o_t, tau)
z_hat = z + (1 - tau) * v
err = W * (Y - z_hat)
g = autograd.grad(outputs=z_hat, inputs=z, grad_outputs=err)[0]
```

最终 velocity 变成 base velocity 加 guidance：

$$
\mathbf{v}_{\Pi\text{GDM}}
=
\mathbf{v}
+
\lambda_\tau\mathbf{g},
\quad
\lambda_\tau
=
\min\left(\beta,\frac{1-\tau}{\tau r_\tau^2}\right)
$$

这里容易误解成“先求逆，再把结果加到 velocity”。更准确地说，$\mathbf{g}$ 不是精确 inverse solution，而是一阶 guidance / gradient correction。原始 flow 采样的 Euler update 是：

$$
\mathbf{z}_{\tau+\Delta\tau}
=
\mathbf{z}_{\tau}
+
\Delta\tau\,\mathbf{v}_{\pi}(\mathbf{z}_{\tau},\mathbf{o}_{t},\tau)
$$

而 $\mathbf{g}$ 已经和 $\mathbf{z}$ / $\mathbf{A}^{\tau}_{t}$ 处在同一个向量空间里，表示“当前 noisy chunk 应该往哪个方向改，才能让最终估计更接近旧 chunk 条件”。因此 RTC 把它作为额外 drift 加到 velocity field：

$$
\mathbf{z}_{\tau+\Delta\tau}
=
\mathbf{z}_{\tau}
+
\Delta\tau
\left[
\mathbf{v}_{\pi}(\mathbf{z}_{\tau},\mathbf{o}_{t},\tau)
+
\lambda_\tau\mathbf{g}
\right]
$$

直观上：

- $\mathbf{v}_{\pi}$：模型原本认为合理的 denoising / action generation 方向。
- $\mathbf{g}$：旧 chunk 连续性约束给出的局部修正方向。
- $\lambda_\tau$：约束修正相对原始模型 velocity 的强度。

$\lambda_\tau$ 中各项含义如下：

- $\tau$：当前 flow timestep，表示采样从噪声走向 clean action 的进度；更新通常发生在 $\tau\in[0,1)$ 上。
- $1-\tau$：当前一步估计最终 clean chunk 时剩余的 denoising 路程，来自 $\widehat{\mathbf{A}^{1}_{t}}=\mathbf{A}^{\tau}_{t}+(1-\tau)\mathbf{v}$。
- $r_\tau^2=\frac{(1-\tau)^2}{\tau^2+(1-\tau)^2}$：$\Pi$GDM 推导里的时间相关方差 / 不确定性尺度，用来归一化最终估计误差在当前 $\tau$ 下的可信度。
- $\frac{1-\tau}{\tau r_\tau^2}$：理论 guidance weight，随 denoising 时间调整条件约束强度；它来自 $\Pi$GDM 对 conditional generation 的近似推导，而不是 RTC 手工拍的常数。
- $\beta$：guidance weight clipping 上限。因为 $\frac{1-\tau}{\tau r_\tau^2}$ 在端点附近会变得很大，特别是 $\tau=0$ 处会发散，少步控制采样中若不截断会让动作 chunk 发散或产生 jerk。

所以 guided velocity 的含义是：

$$
\text{guided velocity}
=
\text{model velocity}
+
\text{scaled continuity correction}
$$

它不是严格求解“满足旧 chunk 约束”的最优动作，而是在每个 denoising step 用局部梯度把采样轨迹拉向满足约束的区域。因为这个操作改变了原始 learned flow，最终 denoised trajectory 仍可能偏离模型动作流形，也不自动保证动力学、接触、碰撞或关节/力矩约束可行。

因此 RTC 不是训练一个新的 policy，也不是在生成后硬拼接动作，而是在每一步 denoising 里轻微改变采样方向，使最终 chunk 更可能落在“符合旧 chunk 条件”的动作流形上。这个 VJP 可以用 reverse-mode autodiff 计算；代价是推理时需要额外的反向传播式计算，所以 RTC 每个 denoising step 会比 vanilla flow inference 更慢。

这里的 $\beta$ 很重要。理论 guidance 系数在 $\tau=0$ 附近会发散，而控制任务常常只用很少 denoising steps，例如真实实验里 $n=5$。不 clip 会让 guidance 过强，动作 chunk 反而发散或产生更大 jerk；论文用 $\beta=5$ 作为保守设置。

### Phase 3: Soft masking

最直接的 inpainting 是 hard mask：只把前 $d$ 个动作权重设为 1，其他位置设为 0。论文指出这在控制里不够，因为 $\Pi$GDM correction 本身不是完美投影；当 $d$ 很小时，只有极短的 frozen prefix，guidance 信号太弱，新 chunk 仍可能切到另一个 mode。

RTC 因此对所有 $H-s$ 个重叠动作使用 [[Soft Masking|soft mask]]：

- $i<d$：已确定会执行，权重 1。
- $d \le i < H-s$：仍可被更新，但要和旧 chunk 保持连续，权重指数衰减。
- $i \ge H-s$：没有重叠，权重 0。

对应公式是：

$$
\mathbf{W}_{i}=
\begin{cases}
1 & \text{if } i<d\\
c_i\frac{e^{c_i}-1}{e-1} & \text{if } d\leq i < H-s\\
0 & \text{if } i\geq H-s
\end{cases}
$$

直观上，$\mathbf{W}_i$ 表示“第 $i$ 个新动作应该多相信旧 chunk”。越靠近当前时刻，旧 chunk 的动作越像已经承诺的局部轨迹，应强约束；越靠未来，旧 observation 的不确定性越大，新 observation 应该有更大自由度。这个设计保留了闭环反应性，同时减少跨 chunk 的 mode jump。

这也解释了 RTC 和 [[Temporal Ensembling]] 的差异：temporal ensembling 是把多个 chunk 对同一时刻的动作做平均，容易在多峰动作分布里平均出无效动作；RTC 是通过 denoising guidance 让采样留在某个连续 mode 上，而不是把不同 mode 直接平均。

### Phase 4: Real-time execution loop

RTC 有两个线程：

1. 控制线程每 $\Delta t$ 调用 `GetAction`，消耗当前 chunk 的下一步动作，并把新 observation 写入 mutex-protected shared state。
2. 后台 `InferenceLoop` 等待当前 chunk 至少执行 $s_{\min}$ 步，保存当下 observation 和旧 chunk 剩余部分，释放锁，然后调用 `GuidedInference` 生成新 chunk。
3. 推理完成后，后台线程重新拿锁，把 $\mathbf{A}_{\text{cur}}$ 立刻替换为 $\mathbf{A}_{\text{new}}$，并把当前 chunk 内索引重置为“推理期间已经过去的步数”。

最后一步很关键。假设启动推理时旧 chunk 已执行 $s$ 步，推理完成时又过去了 $d$ 步，那么切换到新 chunk 后，不应该从 $\mathbf{A}_{\text{new}}[0]$ 开始执行，而应从 $\mathbf{A}_{\text{new}}[d]$ 附近继续执行；前 $d$ 个动作只用于保证生成轨迹与旧 chunk 已执行部分兼容。Algorithm 1 中的 `t = t - s` 正是在做这个索引对齐。

实际 execution horizon 是动态的：如果推理 delay 比 $s_{\min}$ 大，切换后当前索引可能已经超过 $s_{\min}$，后台线程会几乎立刻开始下一次推理，因此等效 $s=\max(d,s_{\min})$。delay 用过去 $b$ 次观测 delay 的最大值保守估计，避免低估下一次推理时间而导致新 chunk 来不及生成。

整体上，RTC 的控制循环可以概括为：

1. 执行旧 chunk，同时收集最新 observation。
2. 在旧 chunk 未结束时，用最新 observation 生成新 chunk。
3. 用旧 chunk 的重叠未来动作作为 inpainting 条件，保证边界连续。
4. 新 chunk 一到就切换，并跳过它已经“错过”的前 $d$ 个位置。

因此 RTC 的核心不是更快的模型，而是把高延迟模型的计算藏进旧 chunk 的执行时间里，并用 inpainting 让异步切换仍然像连续闭环控制。

## 关键公式

### 公式1: [[Flow Matching|动作块 flow 积分]]

$$
\mathbf{A}_{t}^{\tau+\frac{1}{n}}
=
\mathbf{A}_{t}^{\tau}
+
\frac{1}{n}\mathbf{v}_{\pi}(\mathbf{A}_{t}^{\tau},\mathbf{o}_{t},\tau)
$$

**含义**：从噪声 action chunk 沿 learned velocity field 积分，得到最终动作块。

**符号说明**：
- $\mathbf{A}_{t}^{\tau}$：flow timestep $\tau$ 上的 action chunk。
- $\mathbf{o}_t$：当前 observation。
- $\mathbf{v}_{\pi}$：策略的 velocity field。
- $n$：denoising / integration steps。

### 公式2: [[Pseudoinverse Guidance|$\Pi$GDM guidance velocity]]

$$
\begin{aligned}
\mathbf{v}_{\Pi\text{GDM}}(\mathbf{A}^{\tau}_{t},\mathbf{o}_{t},\tau)
&=
\mathbf{v}(\mathbf{A}^{\tau}_{t},\mathbf{o}_{t},\tau)
+\min\left(\beta,\frac{1-\tau}{\tau\cdot r^{2}_{\tau}}\right)
\left(\mathbf{Y}-\widehat{\mathbf{A}^{1}_{t}}\right)^{\top}
\operatorname{diag}(\mathbf{W})
\frac{\partial\widehat{\mathbf{A}^{1}_{t}}}{\partial\mathbf{A}^{\tau}_{t}}
\end{aligned}
$$

**含义**：在原始 flow velocity 上加入 guidance，让最终动作块匹配旧 chunk 的条件部分。

**符号说明**：
- $\mathbf{Y}$：要匹配的旧动作块目标。
- $\mathbf{W}$：mask / soft mask 权重。
- $\widehat{\mathbf{A}^{1}_{t}}$：当前中间状态预测的最终动作块。
- $\beta$：guidance weight clipping，上限控制数值稳定性。

### 公式3: [[Inference-Time Inpainting|最终动作块估计]]

$$
\widehat{\mathbf{A}^{1}_{t}}
=
\mathbf{A}^{\tau}_{t}
+
(1-\tau)\mathbf{v}(\mathbf{A}^{\tau}_{t},\mathbf{o}_{t},\tau)
$$

**含义**：用当前 $\tau$ 的状态和 velocity field 一步估计最终 denoised action chunk。

**符号说明**：
- $\tau$：flow timestep。
- $\mathbf{A}^{\tau}_{t}$：当前中间 action chunk。
- $\mathbf{v}$：base flow velocity。

### 公式4: [[Pseudoinverse Guidance|guidance 方差项]]

$$
r^{2}_{\tau}
=
\frac{(1-\tau)^2}{\tau^2+(1-\tau)^2}
$$

**含义**：$\Pi$GDM guidance 中的尺度项；$\tau=0$ 附近会导致 guidance weight 发散，因此 RTC 加入 $\beta$ clipping。

**符号说明**：
- $r^{2}_{\tau}$：随 flow timestep 变化的 guidance scale。
- $\beta$：最大 guidance weight，论文实验使用 5。

### 公式5: [[Soft Masking|soft mask 权重]]

$$
\mathbf{W}_{i}=
\begin{cases}
1 & \text{if } i<d\\
c_i\frac{e^{c_i}-1}{e-1} & \text{if } d\leq i < H-s\\
0 & \text{if } i\geq H-s
\end{cases}
\quad
\text{where }
c_i=\frac{H-s-i}{H-s-d+1},\quad i\in\{0,\ldots,H-1\}
$$

**含义**：前 $d$ 步动作必须和旧 chunk 一致，中间重叠区域逐步降低约束，尾部自由生成。

**符号说明**：
- $i$：chunk 内动作索引。
- $d$：inference delay。
- $H$：prediction horizon。
- $s$：execution horizon。
- $\mathbf{W}_i$：第 $i$ 个动作的 guidance weight。

### 公式6: [[Real-Time Control|实时性与 horizon 约束]]

$$
\delta \leq \Delta t
\quad \text{or} \quad
d \leq s \leq H-d,
\qquad
s=\max(d,s_{\min})
$$

**含义**：若单次推理时间 $\delta$ 小于控制周期 $\Delta t$，实时性很容易满足；大模型通常不满足，因此 RTC 通过异步推理要求 delay $d$ 能被 chunk 重叠区吸收。

**符号说明**：
- $\delta$：生成一个 action chunk 所需时间。
- $\Delta t$：控制周期。
- $d$：以控制步计的 inference delay。
- $s_{\min}$：用户希望的最小 execution horizon。

## Algorithm 1: [[Real-Time Chunking]]

```text
Inputs:
  flow policy pi, prediction horizon H, minimum execution horizon s_min,
  mutex M, condition variable C, initial chunk A_init,
  initial delay estimate d_init, delay buffer size b,
  denoising steps n, guidance clipping beta.

InitializeSharedState:
  t = 0
  A_cur = A_init
  o_cur = null

GetAction(o_next), called every Delta t:
  acquire M
  t = t + 1
  o_cur = o_next
  notify C
  return A_cur[t - 1]

InferenceLoop:
  acquire M
  Q = Queue([d_init], maxlen=b)
  loop:
    wait on C until t >= s_min
    s = t
    A_prev = A_cur[s, s+1, ..., H-1]
    o = o_cur
    d = max(Q)
    release M
    A_new = GuidedInference(pi, o, A_prev, d, s)
    acquire M
    A_cur = A_new
    t = t - s
    enqueue t onto Q

GuidedInference(pi, o, A_prev, d, s):
  compute W using Eq. 5
  right-pad A_prev to length H
  initialize A^0 ~ N(0, I)
  for tau = 0 to 1 with step size 1/n:
    f_Ahat(A') = A' + (1 - tau) v_pi(A', o, tau)
    e = (A_prev - f_Ahat(A^tau))^T diag(W)
    g = e * partial f_Ahat / partial A' evaluated at A' = A^tau
    A^{tau+1/n} = A^tau + (1/n)(v_pi(A^tau,o,tau)
                      + min(beta, (1-tau)/(tau r_tau^2)) g)
  return A^1
```

## 关键图表

### Figure 1: RTC 在真实点火任务中的效果

![Figure 1 candle frame 1](https://arxiv.org/html/2506.07339v2/candle_frame1.jpg)
![Figure 1 candle frame 2](https://arxiv.org/html/2506.07339v2/candle_frame2.jpg)
![Figure 1 candle frame 3](https://arxiv.org/html/2506.07339v2/candle_frame3.jpg)
![Figure 1 candle frame 4](https://arxiv.org/html/2506.07339v2/candle_frame4.jpg)
![Figure 1 joint curves](https://arxiv.org/html/2506.07339v2/x1.png)

**说明**：RTC 在点燃火柴这类高精度任务中，即使推理延迟超过 300ms，也能保持可执行动作；底部曲线显示 RTC 比同步推理快约 20%，且比 [[Temporal Ensembling]] 更平滑。

### Figure 2: chunk 边界 mode jump

![Figure 2](https://arxiv.org/html/2506.07339v2/x2.png)

**说明**：旧 chunk 规划从障碍物上方绕过，新 chunk 规划从下方绕过。朴素异步切换会在边界产生高加速度，TE 虽能降低加速度但可能产生无效动作。

### Figure 3: RTC 如何使用旧 chunk

![Figure 3](https://arxiv.org/html/2506.07339v2/x3.png)

**说明**：当 delay 为 $d=4$ 时，$a_{0:3}$ 已确定会执行，必须 frozen；$a_{4:10}$ 可更新但应部分 attend to 旧 chunk；最后 $s=5$ 个动作完全新生成。

### Figure 4: hard masking vs soft masking

![Figure 4](https://arxiv.org/html/2506.07339v2/x4.png)

**说明**：hard masking 只约束 frozen prefix，不能很好匹配跨 chunk 连续性；soft masking 能产生更平滑的方向变化。

### Figure 5: Kinetix 仿真 benchmark

![Figure 5 environment 1](https://arxiv.org/html/2506.07339v2/x5.jpg)
![Figure 5 environment 2](https://arxiv.org/html/2506.07339v2/x6.jpg)
![Figure 5 environment 3](https://arxiv.org/html/2506.07339v2/x7.jpg)
![Figure 5 environment 4](https://arxiv.org/html/2506.07339v2/x8.jpg)
![Figure 5 environment 5](https://arxiv.org/html/2506.07339v2/x9.jpg)
![Figure 5 environment 6](https://arxiv.org/html/2506.07339v2/x10.jpg)
![Figure 5 environment 7](https://arxiv.org/html/2506.07339v2/x11.jpg)
![Figure 5 environment 8](https://arxiv.org/html/2506.07339v2/x12.jpg)
![Figure 5 execution horizon](https://arxiv.org/html/2506.07339v2/x13.png)
![Figure 5 delay](https://arxiv.org/html/2506.07339v2/x14.png)

**说明**：Kinetix 任务强调动态投掷、接触、平衡。RTC 在固定 delay 和不同 execution horizon 下均优于 baseline；soft masking 在低 delay、小 horizon 时尤其有帮助。

### Figure 6: 真实双臂任务结果

![Figure 6 controller steps](https://arxiv.org/html/2506.07339v2/x15.png)
![Figure 6 throughput](https://arxiv.org/html/2506.07339v2/x16.png)

**说明**：RTC 在全部延迟下都有最高 average throughput；同步推理随延迟线性退化，TE 在 +100ms 和 +200ms 下触发保护停机。

### Figure 7: guidance clipping $\beta$ 必要性

![Figure 7 beta ablation 1](https://arxiv.org/html/2506.07339v2/x17.png)
![Figure 7 beta ablation 2](https://arxiv.org/html/2506.07339v2/x18.png)

**说明**：$\frac{1-\tau}{\tau r_\tau^2}$ 在 $\tau=0$ 附近发散，需要 $\beta$ clipping。论文设置 $\beta=5$；过大的 $\beta$ 在 $n=5$ 这类少步 denoising 下会让 action chunk 发散并增加 jerk。

### Figure 8: soft mask schedule 与 Diffuser-style inpainting

![Figure 8 mask schedule](https://arxiv.org/html/2506.07339v2/x19.png)
![Figure 8 diffuser comparison](https://arxiv.org/html/2506.07339v2/x20.png)

**说明**：指数衰减 mask 整体最佳，线性衰减接近；Diffuser 那种每步直接 overwrite 的 inpainting 有帮助，但不如 guidance-based RTC。

### Table 1: inference-time 方法 latency

| Method | Latency |
|---|---:|
| RTC (ours) | 97ms |
| BID with $N=16$ (no forward model) | 115ms |
| BID with $N=16$ (shared backbone) | 169ms |
| BID with $N=16$ (full) | 223ms |
| Vanilla $\pi_{0.5}$ | 76ms |

**说明**：RTC 比 vanilla 慢 21ms，但比 BID 更省；BID full 需要两个完整模型，latency 最高。

### Table 2: RTC 总 inference latency 分解

| Component | Time (mobile) | Time (non-mobile) |
|---|---:|---:|
| Model | 96.89 ± 0.16ms | 97.43 ± 0.28ms |
| Network | 21.20 ± 3.12ms | 6.89 ± 2.39ms |
| Image resize | 11.22 ± 5.00ms | 1.44 ± 0.27ms |
| Other | 9.67 ± 3.20ms | 3.00 ± 0.68ms |
| Total | 138.98 ± 6.71ms | 108.76 ± 2.34ms |

**说明**：移动机器人上的网络和图像 resize 开销明显，真实 total delay 已远高于 50Hz 控制周期。

### Table 3: 模型 inference latency 分解

| Component | Time (no RTC) | Time (with RTC) |
|---|---:|---:|
| Image encoders (SigLIP) | 18ms | 18ms |
| LLM prefill (Gemma 2B) | 44ms | 44ms |
| Denoising step (x5) | 14ms | 35ms |
| Total | 76ms | 97ms |

**说明**：RTC 主要增加 denoising step 的计算，因为每步需要反传 vector-Jacobian product。

### Table 4: RTC hyperparameters

| Hyperparameter | Description | Simulation | Real-world |
|---|---|---:|---:|
| $n$ | Denoising steps | 5 | 5 |
| $H$ | Prediction horizon | 8 | 50 |
| $s_{\text{min}}$ | Minimum execution horizon | - | 25 |
| $\beta$ | Guidance weight clipping | 5 | 5 |
| $b$ | Delay buffer size | - | 10 |

**说明**：真实实验中 $\pi_{0.5}$ 使用 $H=50$，最小执行 horizon 为 25，delay buffer 取 10。

## 实验

### 仿真 benchmark

| 项目 | 内容 |
|---|---|
| Simulator | [[Kinetix]] |
| 任务数量 | 12 个高动态、随机任务 |
| 动作特征 | force-based control，不能简单 hold position |
| 数据生成 | 每个环境训练 6 个 [[RPO]] expert，用不同 seed；每个 episode 随机选 expert |
| 数据规模 | 每个环境 1M transitions |
| 模仿策略 | action chunking flow policy，$H=8$，4-layer [[MLP-Mixer]] |
| 训练 | 32 epochs |
| 评估 | 每个点 2048 rollouts，delay $d=0$ 到 4 |
| 指标 | binary success / solve rate |

**Baselines**：
- Naive async：不使用旧 chunk，生成完就切换。
- [[Bidirectional Decoding|BID]]：用 rejection sampling 保持 continuity，仿真中 $N=32$、$K=3$。
- [[Temporal Ensembling|TE]]：维护预测 chunk buffer，对同一时间步的多个动作取平均。

**结论**：TE 在多峰动作分布上表现差；BID 计算贵且不如 RTC；RTC 对 inference delay 最稳健，且能在较短 execution horizon 下利用闭环更新。

### 真实机器人实验

| 项目 | 内容 |
|---|---|
| Base policy | [[Pi05|$\pi_{0.5}$]] |
| 机器人 | 双 6-DoF 机械臂 + parallel jaw grippers，包含移动和非移动设置 |
| 控制 | 位置控制，50Hz，$\Delta t=20$ms |
| Model latency | vanilla 76ms，RTC 97ms |
| Remote inference | LAN 上远程 RTX 4090 workstation |
| 初始 delay | RTC 约 $d\approx 6$ |
| 注入 delay | +100ms 和 +200ms，对应 $d\approx 11$ 和 $d\approx 16$ |
| Episode 数 | 6 tasks × methods × delays，共 480 episodes |
| 机器人执行时间 | 约 28 小时纯执行 |

真实任务：

| Task | Steps | Cutoff | 内容 |
|---|---:|---:|---|
| Light candle | 5 | 40s | 拿火柴和火柴盒，点燃火柴，点蜡烛，丢入碗中 |
| Plug ethernet | 6 | 120s | 拿网线端，调整方向，插入服务器机架，两端重复 |
| Make bed, mobile | 3 | 200s | 移动毯角和两个枕头 |
| Shirt folding | 1 | 300s | 折叠铺平的衬衫 |
| Batch folding | 4 | 300s | 从篮中取衣物，铺平，折叠，放到一叠上 |
| Dishes in sink, mobile | 8 | 300s | 把 4 个不同物体从台面移到水槽 |

真实 baseline：

- Synchronous：执行 $s=25$ 步，然后暂停等待新 chunk。
- TE sparse：执行 $s=25$ 步，同时生成下一 chunk，并做少量 temporal ensembling。
- TE dense：尽可能频繁推理，$s=d$，更接近 ACT 原始 TE。

**结论**：RTC 在所有 delay 下 average throughput 最好，并且 +100ms、+200ms 下显著优于 baseline。同步方法不仅慢，还因停顿改变动力学；TE 在高 delay 下振荡严重，触发保护停机。

## 批判性思考

### 优点

1. **工程切入点很准**：论文没有尝试训练更快模型，而是解决大模型不可避免的高 latency 执行问题。
2. **纯推理时改造**：不需要改训练 recipe，能直接套到 diffusion / flow-based VLA。
3. **对真实机器人指标有意义**：throughput 同时惩罚慢和失败，比单纯 success rate 更贴近部署。
4. **把动作连续性形式化**：old chunk overlap 不再被浪费，而是作为 inpainting 条件使用。
5. **附录复现实用**：latency 分解、$\beta$ ablation、超参和代码 release 都给得比较清楚。

### 局限性

1. **无轨迹可行性保证**：RTC 解决的是跨 chunk 连续性和异步执行，不是动力学/碰撞/接触约束求解器。guided flow 改成 $\dot{\mathbf{A}}^\tau=\mathbf{v}_\pi+\lambda_\tau\mathbf{g}$ 后，最终 denoised chunk 仍可能不可行，尤其是在接触丰富、动力学敏感或约束冲突的任务中。
2. **可能偏离 learned flow / action manifold**：$\mathbf{v}_\pi$ 是模型学到的生成方向，而 $\lambda_\tau\mathbf{g}$ 是推理时外加的连续性修正。若 $\beta$ 太大、mask 太强、旧 chunk 条件和新 observation 冲突，采样轨迹可能被拉到低概率区域，产生 jerk、发散或不自然动作。
3. **局部一阶修正，不是精确投影**：$\mathbf{g}$ 来自当前 $\tau$ 的 Jacobian / VJP，只是局部线性化下的 gradient correction。它不是把动作投影到全局可行集合，也不能保证找到满足所有约束的最优 trajectory。
4. **旧 chunk 可能已经错了**：RTC 把旧 chunk 的重叠未来动作当作 inpainting 条件。如果旧 plan 已经过时、环境突然变化，或者旧 chunk 本身走向错误 mode，过强 guidance 会让新 chunk 过度跟随旧策略，牺牲闭环反应性。
5. **依赖 chunk 重叠空间**：如果 $d$ 接近或超过 $H/2$，可用 overlap 变小，算法余量会下降；如果 delay 估计偏低，新 chunk 也可能来不及替换。
6. **只适用于 diffusion / flow action policy**：自回归离散动作模型不能直接套用，因为 RTC 需要连续 denoising state、velocity field 和 vector-Jacobian product。
7. **增加推理计算**：RTC 需要每个 denoising step 做 VJP，vanilla $\pi_{0.5}$ 76ms 变成 97ms；这会抵消一部分异步执行收益，也要求推理系统有足够余量。
8. **真实实验任务仍是 manipulation**：论文指出 legged locomotion 等更动态场景可能更受益，但没有真实上机验证。
9. **没有训练时 co-design**：它解决执行问题，但不改变 policy 在训练时对 delay 的认知；后续训练时 RTC 可能更彻底。

### 我的判断

RTC 是 VLA 上机里非常实用的一层 runtime adapter。它不改变模型能力上限，但显著改善“大模型想得慢、机器人不能等”的问题。和 [[Pi05]]、[[OpenHLM]] 这类大模型策略相比，RTC 的价值在部署层：把 action chunk 从“批量预测”变成“连续控制信号”。对于需要远程推理、移动机器人、云端模型或多机协同的系统，这类异步执行会越来越重要。

## 可复现性评估

- [x] 代码开源：仿真 benchmark 代码在 GitHub。
- [x] 算法伪代码完整：Algorithm 1 给出线程、delay buffer、guided inference。
- [x] 关键超参完整：Table 4 给出 $n,H,s_{\min},\beta,b$。
- [x] 主要图表完整：8 个 Figure、4 个 Table。
- [ ] 真实机器人数据未完全公开：真实 $\pi_{0.5}$ policy 与 robot evaluation pipeline 仍依赖 Physical Intelligence 内部系统。

## 关联笔记

### 基于

- [[Action Chunking]]：RTC 的目标对象是 action chunking policy。
- [[Flow Matching]]：论文以 flow-based action chunk generator 为主。
- [[Diffusion Policy]]：diffusion policy 可转换为 flow policy 后使用 RTC。
- [[Pi05]]：真实实验的 base VLA。

### 对比

- [[Temporal Ensembling]]：简单 action averaging，在多峰动作下会产生无效动作。
- [[Bidirectional Decoding]]：用 rejection sampling 保持 chunk continuity，但计算更贵。
- [[Model Predictive Control]]：都并行执行与规划，但 RTC 不依赖显式 dynamics/cost。

### 方法相关

- [[Inference-Time Inpainting]]：把新 chunk 生成写成补全问题。
- [[Pseudoinverse Guidance]]：guidance 项来源。
- [[Soft Masking]]：跨 chunk 连续性的关键工程设计。
- [[Asynchronous Inference]]：RTC 的执行机制。
- [[Real-Time Control]]：最终部署目标。

### 实验相关

- [[Kinetix]]：仿真 benchmark。
- [[RPO]]：仿真 expert policy。
- [[MLP-Mixer]]：仿真 action chunk flow policy 架构。
- [[Remote Inference]]：真实机器人 latency 的关键来源。

## 速查卡片

> [!summary] RTC
> - 任务：让高延迟 action chunking VLA 实时执行。
> - 方法：异步生成下一 chunk，并用 inpainting 约束它与旧 chunk 连续。
> - 关键公式：$\Pi$GDM guidance + soft mask。
> - 主要收益：减少 pause、jerk 和 delay 下的 throughput 退化。
> - 限制：只适用于 diffusion/flow policy，且推理更贵。

## 来源

- arXiv: https://arxiv.org/abs/2506.07339
- arXiv HTML: https://arxiv.org/html/2506.07339v2
- Project page: https://pi.website/research/real_time_chunking
- Code: https://github.com/Physical-Intelligence/real-time-chunking-kinetix
