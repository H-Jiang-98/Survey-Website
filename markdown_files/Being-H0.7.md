---
title: "Being-H0.7: A Latent World-Action Model from Egocentric Videos"
method_name: "Being-H0.7"
authors: [Hao Luo, Wanpeng Zhang, Yicheng Feng, Sipeng Zheng, Haiweng Xu, Chaoyi Xu, Ziheng Xi, Yuhui Fu, Zongqing Lu]
year: 2026
venue: arXiv
tags: [world-action-model, egocentric-video, vision-language-action, latent-reasoning, future-prediction, flow-matching, humanoid, cross-embodiment]
image_source: online
---

# Being-H0.7：不生成未来像素，latent alignment 就等于 world model 吗？

> 本笔记基于 [arXiv:2605.00078v1](https://arxiv.org/abs/2605.00078)、[HTML 全文](https://arxiv.org/html/2605.00078)、[官方项目页](https://research.beingbeyond.com/being-h07)和[Being-H 仓库](https://github.com/BeingBeyond/Being-H/tree/main/Being-H07)精读核验。论文上传于 2026-04-30；截至 **2026-08-14** 仍为 arXiv preprint。

## 阅读结论先行

Being-H0.7 给 VLA 插入 16 个 latent queries，把当前 instruction、4 帧 observation history、proprioception 压成 action-oriented 中间状态。训练时另开 posterior branch：用 frozen V-JEPA2.1 + Perceiver 把未来 20 帧压成同样 16 个 embeddings，替换 latent queries；prior/posterior 两支都预测同一 action chunk，并在后九层对应 latent hidden states 上做 MSE alignment。推理只留 prior，因此既不生成未来 pixels，也不运行 posterior。

这个设计可以更准确地称为 **future-privileged representation distillation**：后见之明在训练时塑造当前时刻的 latent，而 action loss 限制它保留对控制有用的未来因素。与 video rollout world model 相比，它避免了重建纹理和测试时 imagination latency，是很干净的工程取舍。但它没有显式预测下一状态、奖励、接触或 action-conditioned transition，也不能从 latent rollout 多步评估 counterfactual actions；“world-action model”是作者对表示用途的命名，不应等同于可查询的 dynamics model。

最严重的证据缺口是 **整篇没有方法消融表**。没有 no-latent、latent-only/no-posterior、no-alignment、stop-gradient teacher、不同 future horizon/query count/aligned layers、无 norm/rank regularization或 pixel-prediction matched-compute control。因此 Being-H0.7 相比 H0.5、$\pi_{0.5}$、Fast-WAM 的领先不能单独归因于 future alignment；它同时把 H0.5 的 2B system 改为约 3B、InternVL3.5 + Qwen3 action expert、V-JEPA2.1 encoders、新预训练数据/规模和部署栈。

项目页宣称 200,000 小时 human ego + 15,000 小时 robot data，但论文方法/实验只说依照 UniHand-2.0 format 混合 human/robot trajectories，没有给 215K 小时的 source table、sample/token 数、curation、去重、license、mixture weight、训练 steps/GPU 或 compute。它比 H0.5 的 35K-hour recipe 大六倍以上，却没有数据审计；因而无法区分“latent world modeling”与“更多更新/更多数据”的贡献。

Simulation 覆盖六个 benchmarks：LIBERO 99.2、LIBERO-plus zero-shot 82.1/fine-tuned 84.8、RoboCasa-50 62.1、GR1 49.2、CALVIN seen/unseen 4.67/4.48，以及 RoboTwin2 easy/hard 90.2/89.6。覆盖面很强，但 Table 1 是不同论文的 system-level results，模型大小、pretraining、inputs、data 与 compute 不一致；RoboCasa 还低于 Cosmos-Policy 67.1，RoboTwin 低于 LingBot-VA/Fast-WAM，所以“六项 SOTA”并不准确，较合适的表述是 best average rank / broadly competitive。

真实实验在 Adam-U、Unitree G1、Franka FR3 的 12 个任务上比较 H0.7、H0.5、$\pi_{0.5}$、Fast-WAM。每 task 默认 20 次 blind trials，scene/policy/order 随机且统一 inference server，协议较好；但 Figure 5 只给按 overlapping tags 聚合的五个 suite bars。同一 task 可以重复进入 Dynamic、Physical、Motion、Long Horizon、Generalization 多个平均数，这五个 bar 不是独立证据，且未给逐任务成功率、confidence interval 或多 seed。

“physical reasoning”也需要克制。pipette、pouring、folding、hammering 成功说明 policy 在这些训练任务中学会有效 visuomotor behavior；它没有 force/tactile input、physics state supervision 或反事实评测，不能证明 latent 显式编码 gravity/contact。作者用 latent + current observation 条件一个外部 video generator 得到看似合理未来，这是 probe/可视化，不是量化 decoding accuracy，更不能排除 current frame 与 generator prior 主导图像。

部署优势主要来自继承 H0.5 的 UAC，而非 latent method 本身。客户端异步执行 committed prefix、只拼 suffix，使对外 step latency 进入 3–4ms；这个数是 buffer amortized/exposed action-step cost，不是端到端一次模型请求 latency。Unitree G1 的 VLA 只给 26-DoF upper-body targets，50Hz AMO controller 负责 lower body/waist balance，所以不是 H0.7 直接生成全身 loco-manipulation commands。

公开性目前很低：Apache-2.0 总仓库的 `Being-H07/` 只有 README，没有实现、config、checkpoint、dataset或 evaluation scripts。官方主 README 明确写 code/checkpoints “coming soon”。可以读方法和看 demos，不能复现训练、benchmark或真实部署。

### 一句话总结

Being-H0.7 提出一种轻量、合理的 future-privileged latent distillation：训练时让当前上下文 queries 对齐未来视觉 embeddings，推理时仍直接出 action；但论文没有任何正交方法消融，也未披露 215K-hour mixture，真实 suite 统计重复计入 tasks，公开仓库只有 README，因此现有结果证明的是新一代 Being-H system 很强，而非严格证明 latent alignment 本身带来了“world reasoning”。

![Being-H0.7 架构：共享 context 的 prior/posterior packed branches](https://research.beingbeyond.com/being-h07/images/arch.webp)

*图 1。官方架构图。Posterior 只在训练时看到 future observations；两支由 attention mask 隔离，仅通过 latent alignment 相遇。*

## 0. 方法定位：VLA、WAM 与 H0.7

| 范式 | 训练目标 | 测试时路径 | 能做什么 | 主要代价/缺口 |
| --- | --- | --- | --- | --- |
| Direct VLA | context→action | 一次 action generation | 低延迟闭环 | 易学 observation-action shortcut |
| Pixel WAM | future video + action | 常需或曾训练 visual rollout | dense dynamics supervision、可视化 future | 像素成本高、future 多解 |
| Being-H0.7 | future embedding→prior latent alignment + action | context→latent queries→action | 用 future privileged info 塑造表示 | 没有显式 transition/rollout/counterfactual |

“未来信息对动作有用”并不等于“未来表示必须可预测”。Posterior embeddings 含有随机/不可由当前 context 决定的细节，point-wise MSE 会把 prior 拉向 conditional mean；多模态未来可能被平均。论文没有 probabilistic latent、contrastive target、variance model 或 action-conditioned posterior，因而最适合 deterministic demonstration futures。

## 1. Dual-branch latent reasoning

Prior sequence 是：

$$
S=[x;,o_{-H:0};,s;,Q;,a_{0:T}],qquad Q\in\mathbb R^{K\times d}.
$$

默认 $H=4$、$T=20$、$K=16$。Posterior 用 frozen ViT 和 trainable Perceiver（论文未清楚拆出每部分 trainability）把未来 observations 压缩：

$$
z^{\mathrm{post}}=E(\tilde o_{0:T})\in\mathbb R^{K\times d}.
$$

它在同一 sequence packing 中取代 $Q$。相应 token 使用相同 positional IDs；shared context 对两支可见，prior/posterior tokens 彼此不可见。后九个 Transformer layers 对齐：

$$
\mathcal L_{\mathrm{align}}
=\frac1L\sum_{\ell=1}^{L}\frac1{|h_\ell|}
\left\|h_\ell^{\mathrm{prior}}-h_\ell^{\mathrm{post}}\right\|_F^2.
$$

两支分别做 flow matching。对 $a_t=ta+(1-t)\epsilon$、$u_t=a-\epsilon$：

$$
\mathcal L_{\mathrm{FM}}=
\|v_\theta^{\mathrm{prior}}(a_t,c,q)-u_t\|_2^2+
\|v_\theta^{\mathrm{post}}(a_t,c,z^{\mathrm{post}})-u_t\|_2^2.
$$

注意 posterior 与 prior 共享 backbone/action path，又同时被 alignment 拉近，没有 detached teacher 的说明；两边可能共同移动。Future branch 不是固定 oracle，降低了 distillation 稳定 target 的可解释性。

### 1.1 防 collapse 正则

Norm lower bound 防止 latent 变零：

$$
\mathcal R_{\mathrm{norm}}(h)=\left[\operatorname{ReLU}(\tau-\|h\|_2)\right]^2.
$$

将一批 latent 投影到随机子空间、row normalize，取 Gram eigenvalues 的 normalized spectrum $p_i$，最小化负 entropy：

$$
\mathcal R_{\mathrm{rank}}(H)=\sum_i p_i\log p_i.
$$

总目标：

$$
\mathcal L=\mathcal L_{\mathrm{FM}}+10^{-3}\mathcal L_{\mathrm{align}}
+10^{-4}\mathcal R_{\mathrm{norm}}+10^{-4}\mathcal R_{\mathrm{rank}}.
$$

Post-training 保留 action+alignment，去掉 anti-collapse。论文没有报告 effective rank、norm distribution 或关掉两项后的结果，无法确认 collapse 是否真实发生、阈值 $\tau$ 如何选。

~~~mermaid
flowchart TB
    C["Shared current context\ninstruction + 4 observations + state"] --> P["Prior branch\n16 learnable queries"]
    F["20 future frames\ntraining only"] --> E["Frozen V-JEPA2.1 + Perceiver\n16 future embeddings"]
    C --> O["Posterior branch"]
    E --> O
    P --> A1["Prior flow action loss"]
    O --> A2["Posterior flow action loss"]
    P <-->|"last 9 layers MSE alignment"| O
    P --> I["Inference action chunk"]
    O -. "discarded at inference" .-> X["No pixel rollout"]
~~~

## 2. 训练 recipe 的可审计缺口

| 已报告 | 未报告 |
| --- | --- |
| context 224²、future 256²；batch≈128 chunks | optimizer、LR、schedule、steps/epochs、GPU 数、训练时长 |
| InternVL3.5 understanding + Qwen3 action expert；V-JEPA2.1 visual encoders | 具体 checkpoint sizes、总/active parameters、哪些模块冻结 |
| 200K human + 15K robot（仅项目页） | source datasets、hours 口径、samples/tokens、mixture ratio、dedup、pseudo-actions |
| downstream 只用 action+alignment | 每 benchmark exact hyperparameters 与 checkpoint selection |

这使“latent route 比 pixel route 更 scalable”停留在复杂度直觉：论文没有 tokens/sec、training FLOPs、wall-clock、memory 或 matched-quality pixel WAM training cost。

## 3. Simulation：广覆盖不等于同条件因果比较

| Benchmark | Being-H0.7 | 主要 protocol/解释 |
| --- | ---: | --- |
| LIBERO | 99.2% | 每 suite 500 trials；已接近 ceiling |
| LIBERO-plus | 82.1 / 84.8% | standard-only zero-shot / plus fine-tune，增益 2.7pp |
| RoboCasa Human-50 | 62.1% | 24 tasks、50 demos/task、50 trials/task；低于 Cosmos 67.1 |
| GR1 | 49.2% | 24 tasks、1000 demos/task、50 trials；低于 ABot-M0 58.3 |
| CALVIN seen/unseen | 4.67 / 4.48 | 1000 sequences，每条最多五任务；unseen 与 H0.5 都是 4.48 |
| RoboTwin2 easy/hard | 90.2 / 89.6% | 50/500 demos per task clean/randomized；低于多个 WAM rows |

与直接前身 H0.5 相比，H0.7 在 LIBERO +0.3、LIBERO-plus +3.6、RoboCasa +8.6、CALVIN seen +0.04、unseen +0.00；改善分布并不一致。没有 paired reimplementation/no-alignment control 时，RoboCasa 大增不能归因 latent future，CALVIN unseen 持平也说明方法不是所有 generalization 都自动获益。

![六个 simulation benchmark 的归一化 radar 对比](https://research.beingbeyond.com/being-h07/images/benchmark_radar.webp)

*图 2。官方 radar 为可读性在每个 benchmark 内归一化，不能据半径比较不同任务的绝对难度，也会放大窄区间差异。*

## 4. 真实机器人：协议好，但聚合会重复计算

| Platform | Policy interface | Cameras | Policy frequency | 低层控制 |
| --- | ---: | --- | ---: | --- |
| Adam-U | 31 DoF（19 body + 双 6DoF hand） | 2 ego | 20Hz | 未完整公开 |
| Unitree G1 | 26 DoF upper body/hands | 1 ego | 10Hz | AMO 50Hz 生成 lower body/waist balance |
| Franka FR3 | 13 DoF arm+hand | external+wrist | 20Hz | tabletop controller |

12 tasks 涵盖 rolling-ball catch、racket、moving-container pour、conveyor pick/sort、pipette/funnel、garment、shoe tree/box、drawer 和 hammer。每项有一个 primary suite 和多个 overlap tags；suite score 对所有携带该 tag 的 tasks 平均。于是同一个 conveyor task能同时影响 Dynamic、Motion、Long Horizon、Generalization，五个结果高度相关。更透明的报告应给 12×4 method matrix，并用 hierarchical bootstrap across layouts/tasks。

“每 task 20 trials”意味着 task-level success resolution 为 5pp。即使 blinded protocol 降低 operator bias，也没有说明每个 preset configuration 是否真的对每 method 等次数配对；论文同时写 default $K=20$ under each setting 的口径需 exact layout counts 才能核验。

## 5. 还缺什么，才能证明 latent world-action hypothesis？

- 保持 data、backbone、parameters、post-train 与 UAC 不变，只移除 posterior/alignment。
- 用同样 future encoder 但把未来 samples 打乱，检验收益来自真实 temporal correspondence 而非 extra branch regularization。
- 对比 current-frame JEPA target、future mean pooling、latent contrastive/predictive loss和 pixel reconstruction，报告 training FLOPs。
- 在 intervention benchmark 中改变 object velocity、mass、friction、container motion，测试 latent 是否跟踪 causal future，而非 scene cue。
- 线性/非线性 probe contact、object trajectory、task progress，并对 current-context-only probe；外部 video generation 只作定性补充。
- 报告逐 task、CI、failure type、collision/recovery 与 end-to-end request latency，而非只给 tag averages 和 amortized ms/step。

## 6. Artifact audit 与最终判断

| 主张 | 当前证据 | 结论 |
| --- | --- | --- |
| 不生成未来像素也可用 future supervision | dual branch 在 inference 删除 posterior | **结构上成立** |
| latent alignment 带来 benchmark 增益 | 无 no-alignment/no-posterior ablation | **未因果证明** |
| 比 pixel WAM 更高效 | 推理无 rollout；无 matched training cost | **推理直觉成立，训练优势未量化** |
| latent 表示 physical/world dynamics | 定性 video-generator visualization + task success | **间接且不足** |
| 三机器人 real-time deployment | blind trials、UAC、AMO backend | **system evidence 较强**，非端到端全身 VLA |
| 可复现 | repository 只有 README，无 weights/code/data | **不可复现** |

Being-H0.7 的概念值得保留：让 future 作为 privileged teacher，而不是必须重建的输出。它最可能成为一个低成本 auxiliary objective，而非独立替代 world model 的完整范式。当前论文更像强 system report；真正决定其学术可信度的将是公开代码后的一组 matched ablations，而不是增加更多 benchmark rows。
