---
title: "METIS: Multi-Source Egocentric Training for Integrated Dexterous Vision-Language-Action Model"
method_name: "METIS"
authors: [Yankai Fu, Ning Chen, Junkai Zhao, Shaozhe Shan, Guocai Yao, Pengwei Wang, Zhongyuan Wang, Shanghang Zhang]
year: 2025
venue: arXiv
tags: [egocentric-data, vision-language-action, dexterous-manipulation, human-robot-cotraining, motion-dynamics, action-tokenization, humanoid, cross-embodiment]
image_source: online
---

# METIS：把 8 种 Ego 数据压进 44 个 dynamics tokens，是否真的得到跨本体灵巧 VLA？

> 本笔记基于 [arXiv:2511.17366v1](https://arxiv.org/abs/2511.17366)、[HTML 全文](https://arxiv.org/html/2511.17366)、[项目页](https://aureleopku.github.io/METIS/)、[作者官方实现](https://github.com/AureleoPKU/Metis)、[RoboBrain-Dex 实现](https://github.com/FlagOpen/RoboBrain_Dex)及公开 checkpoint 页面精读并交叉核验。论文仅有 2025-11-21 的 arXiv v1，未找到会议接收记录，因此 venue 记为 **arXiv**。公开资产状态核验日期为 **2026-08-13**。

## 阅读结论先行

METIS 的核心不是“从无动作的人类视频学 robot action”，而是把 **8 个已有或自采、且都带手部 pose/action 的 egocentric sources** 汇成 EgoAtlas，用双腕与十个 fingertips 构造 canonical geometry，再让一个 Prismatic-7B VLA 预测压缩后的 motion-aware dynamics tokens，最终由 continuous action decoder 输出 1 秒动作。它把 human motion supervision、robot motion supervision、视觉变化和语言 subtask 放进一个 autoregressive interface，路线介于 human–robot co-pretraining 与 dynamics-token distillation 之间。

EgoAtlas 名义上有 343K trajectories、89.72M image–action pairs，但 raw scale 高度集中：EgoDex 占约 **91.8% trajectories** 和 **86.8% pairs**。作者没有按 raw frequency 训练，而把 EgoDex 权重降到 40.3%，把只有 10K trajectories 的 self-collected enhanced data 提到 25.4%，ActionNet 提到 13.4%。因此性能来自精心重采样的混合物，不是“343K IID trajectories”自然扩展。

Human 与 robot 的统一 action schema 有明确几何含义：双腕各用 camera-frame 3D position + 6D rotation，共 18D；双手各五个 fingertips 在 wrist frame 中用 3D positions，共 30D。Robot joints 经 FK 变成这一 schema，推理时 predicted fingertips/wrists 再经 IK 回到具体 robot。这个接口确实支持 Unitree G1/Inspire hand 与 Sharpa Beta/22-DoF SharpaWave hand，但它只统一几何目标，不保证 joint limit、self/environment collision、balance、contact wrench、force closure 或 tactile feedback。因此应称 **functional motion interface**，尚不是完整 executable-action equivalence。

Motion-aware dynamics 分两支：visual branch 用当前/未来 DINOv2 features 与中间 motion，经 VQ-VAE 得到 4 个 token（codebook 16）；motion branch 用 PoseNet + 两层 RQ-VAE 把 raw motion 压成 40 个 token（codebook 512）。VLA autoregress 44 个 tokens，再由 action decoder结合vision、proprioception预测 30 个未来 steps。这个设计把长高维 action token sequence 变短，但论文没有给与“相同 token budget 的 continuous latent”“只预测 motion token”“相同容量的普通 auxiliary action head”等对照。

最强因果证据是固定 downstream protocol 下的 pretraining ablation：Pick and Place / Open Drawer and Put Bread 从 no-pretrain 的 60/35，提高到 human-only pretrain 的 70/60，再到 full EgoAtlas 的 85/75。它支持 human data 提供有效 prior，也支持 robot-inclusive sources 再增加可执行性。不过只测两项任务、每项默认 20 trials，没有 seed/CI；而且 full variant同时多了 self-collected高质量数据、robot data、source diversity和更多effective tokens，不能单独归因给“multi-source”某一因素。

Dynamics ablation 更大：删去 autoregressive dynamics supervision、只监督 continuous actions 后，两个任务从 85/75 降至 30/0。但这个对照同时移除了整条强 auxiliary supervision，而不只是“离散化”或“compactness”，所以证明了 dynamics loss 很重要，尚不能证明 44-token VQ/RQ 设计是唯一或最优原因。

真机主表覆盖 6 项 fixed-base tabletop tasks，每项100条target robot demonstrations，通常20 trials。METIS平均最好，但不是每格都赢：Open Drawer 的 ACT 是95%、METIS 90%；Put Cola 的 $\pi_{0.5}$ 是75%、METIS 70%；OOD lighting 中 $\pi_{0.5}$ 70%、METIS 65%。最难的 Open Drawer and Put Bread 上，METIS 75%，GR00T N1.5 70%，只是1次trial差异。没有置信区间或显著性检验，不宜把小差距写成决定性SOTA。

所谓“cross-embodiment generalization”也要收窄。Sharpa experiments明确有该robot的first-person images与joint state–action pairs采集、时序对齐与FK preprocessing；论文没有说demo数量或post-training steps。因此85% apple、70% tool-use并非zero-shot从G1直接部署，至少包含new embodiment data pipeline；有力证据是同一 canonical action interface可被新hand复用，而不是无需target data的跨本体skill transfer。

“reasoning and acting”来自作者自采10K trajectories的manual subtask/hand-level labels。模型只在subtask transition进入reasoning mode，再以 `[BOD]`切到 dynamics；它比每步生成语言省延迟。但这10K仅占raw trajectories约2.9%，且没有 no-reasoning ablation、transition accuracy、reasoning text quality、correction/recovery测试。长任务提升也可能来自pretraining和dynamics，而非CoT本身。

截至核验日，复现状态比初稿项目页好：作者 [AureleoPKU/Metis](https://github.com/AureleoPKU/Metis) 已给 Apache-2.0 code、motion-tokenizer training、EgoAtlas pretraining/post-training/deploy commands；另有 4B [BAAI/RoboBrain-Dex](https://huggingface.co/BAAI/RoboBrain-Dex) checkpoint 与 motion model。可复现性仍不完整：EgoAtlas没有作为带统一许可的完整dataset发布，自采10K data也无公开入口；跨平台真实controller/eval protocol不全；两个official repos的backbone与环境说明发生分叉（论文/作者repo 7B Prismatic、RoboBrain-Dex repo 3B/4B RoboBrain；旧repo还曾列Torch 2.9而作者repo列2.2）。复现前必须锁定实现分支。

### 一句话总结

METIS 在 **multi-source pretraining** 中以 camera-frame wrists、wrist-frame fingertips 和44个visual/motion dynamics tokens引入Human Ego动作数据，主要迁移的是跨场景手部几何与时序motion prior；human-only/full/no-pretrain及dynamics ablation支持其价值，但尚未证明zero-shot跨本体、显式contact transfer或reasoning的独立贡献，真实执行仍依赖每任务target robot demos、IK与硬件低层控制。

![METIS teaser：EgoAtlas、motion-aware dynamics、reasoning与真机任务](https://raw.githubusercontent.com/AureleoPKU/Metis/main/assets/Teaser.png)

*图 1。官方 teaser 把 EgoAtlas pretraining、VLA dynamics prediction 与 downstream dexterous tasks 串联起来。*

## 0. 资源、版本、许可与复现优先级

| 资产 | 链接 | 截至2026-08-13状态 | 许可/风险 | 判断 |
| --- | --- | --- | --- | --- |
| 论文 | [arXiv](https://arxiv.org/abs/2511.17366)、[HTML](https://arxiv.org/html/2511.17366) | v1，2025-11-21 | CC BY 4.0 | 证据主源 |
| 项目页 | [METIS](https://aureleopku.github.io/METIS/) | videos、method/result overview | 页面媒体无独立license | 适合定性核验 |
| 7B作者实现 | [AureleoPKU/Metis](https://github.com/AureleoPKU/Metis) | tokenizer/pretrain/posttrain/server code和commands | Apache-2.0 | 优先复现论文架构 |
| 4B衍生实现 | [FlagOpen/RoboBrain_Dex](https://github.com/FlagOpen/RoboBrain_Dex) | posttrain/data conversion/server；pretrain仍有TODO | repo根目录未见LICENSE，不能推定许可 | 需谨慎 |
| 4B checkpoint | [BAAI/RoboBrain-Dex](https://huggingface.co/BAAI/RoboBrain-Dex) | 约8.18GB，标4B/Apache-2.0 | model card仅31 bytes，训练细节很少 | weights可取、文档弱 |
| Motion model | [BAAI collection](https://huggingface.co/collections/BAAI/robobrain-dex) | motion dynamics checkpoint已列 | 与具体code/config版本需匹配 | 中 |
| EgoAtlas | 无统一下载页 | 8源statistics和mixture公开；未发布汇总data | 上游协议异质，自采data未公开 | 数据不可一键复现 |
| Robot demos/eval | 一条pouring格式样例 | G1 RLDS converter可用；6 task datasets未完整发布 | 无完整hardware client/eval harness | deployment不完整 |

复现优先级建议：先验证motion tokenizer reconstruction与code usage，再用作者7B branch在一个G1 task post-train；不要混用RoboBrain-Dex checkpoint、Prismatic-7B config和不同motion codebook。

## 1. EgoAtlas：规模、来源与真正的数据分布

### 1.1 八个来源

| Source | Trajectories | Frames/pairs | Human/Robot | Pose | Subtask | Wild | 角色 |
| --- | ---: | ---: | --- | --- | --- | --- | --- |
| ARCTIC | 296 | 214.5K | 100/0% | 有 | 无 | 否 | mocap精确双手物体交互 |
| H2O | 109 | 65.3K | 100/0% | 有 | 有 | 否 | tabletop+subtask语义 |
| HoloAssist | 100 | 777.3K | 100/0% | 有 | 有 | 否 | 长时交互/多样物体 |
| OakInk | 134 | 146K | 100/0% | 有 | 有 | 否 | hand-object knowledge |
| EgoDex | 314.8K | 77.9M | 100/0% | 有 | 无 | 是 | 绝对规模主体 |
| PH2D | 1.8K | 416.5K | 66.1/33.9% | 有 | 无 | 是 | human–H1 bridge |
| ActionNet | 15.7K | 7.4M | 0/100% | 有 | 无 | 是 | robot motion anchor |
| Self-collected | 10K | 2.8M | 100/0% | 有 | 有 | 是 | 高精度、glove appearance、reasoning labels |

总量343K trajectories、89.72M pairs。Raw distribution并不均衡：

$$
r_{EgoDex}^{traj}=\frac{314.8}{343}\approx91.8\%,
\qquad
r_{EgoDex}^{pair}=\frac{77.9}{89.72}\approx86.8\%.
$$

论文用下列训练mixture重平衡：H2O 0.8%、OakInk 1.9%、PH2D 5.4%、ARCTIC 2.8%、EgoDex 40.3%、HoloAssist 10.6%、ActionNet 13.4%、self-collected 25.4%。四舍五入合计100.6%，属于表格rounding。与raw比例相比，self-collected被极大过采样；因此它既是motion quality source，也是subtask reasoning几乎唯一的监督源。

### 1.2 自采 wearable system

- Manus Quantum Metagloves：每手25个3D keypoints；策略最终只用五指fingertips。
- 两个wrist VIVE Trackers：给双腕world-frame 6-DoF。
- Head-mounted ego camera + 第三个VIVE tracker：固定extrinsic把wrist转到camera frame。
- 另有top-down camera，论文说为未来研究保留；METIS输入仍只有ego RGB。
- 全系统20Hz；界面实时显示video/wrist/fingertips，tracking坏时由operator手动终止。
- 每episode有overall instruction及人工subtask/hand-level description。

论文没有报告采集人数、demographic/scene split、episode时长分布、tracking rejection rate、failure/recovery、privacy/consent或release license。Gloves同时改善pose precision与制造明显human visual artifact；作者把它称为增加visual diversity，但也可能形成source shortcut。

## 2. Human/Robot 对照与统一 action space

| 维度 | Human data | Robot data | 统一机制 | 剩余 gap |
| --- | --- | --- | --- | --- |
| 视觉 | head/VR/wearable ego RGB | robot head RGB | 都作为first-person image | camera intrinsics/height、human arm/glove vs robot |
| Wrist | tracker/vision得到双腕SE(3) | arm FK | camera-frame position+6D rotation | reach、base/shoulder geometry |
| Fingers | human fingertips/keypoints | hand joints | wrist-frame fingertips；robot先FK | hand topology、palm/contact surface |
| Proprioception | human wrist/finger pose | robot state经FK | 相同48D geometry | actuator/joint/torque不可见 |
| Action | future human pose | future robot pose | 相同48D target | IK可行性与tracking error |
| 时间 | source-specific，self 20Hz | source-specific；downstream30Hz | preprocessing/alignment | 插值误差、latency |
| 语言 | dataset labels；self有subtasks | downstream task instruction | LLM tokens | 仅部分source有细粒度label |
| Object state | 不显式 | 不显式 | 无 | 只能从RGB隐式估计 |
| Contact/force | 无 | 无 | 无 | contact topology、wrench、tactile缺失 |

双腕表示为：

$$
P_t^w=\left[p_t^L,r_{6D,t}^L,p_t^R,r_{6D,t}^R\right]\in\mathbb{R}^{18},
$$

双手指尖表示为：

$$
P_t^f=\left[q_{t,1}^L,\ldots,q_{t,5}^L,q_{t,1}^R,\ldots,q_{t,5}^R\right]
\in\mathbb{R}^{30},\quad q\in\mathbb{R}^3.
$$

Robot joint vector $j_t$ 通过 $FK(j_t)$ 得到finger geometry；部署时求解：

$$
j_t^*=\arg\min_j\sum_i\left\|FK_i(j)-\hat q_{t,i}\right\|_2^2+\mathcal R(j).
$$

论文只明确“用IK逆映射”，未给regularizer、joint/collision constraints、failure handling或误差指标，所以上式的 $\mathcal R$ 不能假定具体实现。Camera-relative wrists减少world-frame差异，wrist-relative fingers减少global pose差异，但没有消除body reach与physics。

**迁移层级判断**：Human action可作为canonical geometric command supervision；经platform-specific IK后能在两种robot上执行，超过纯observation transfer。但没有human→robot同trajectory physical execution、contact feasibility metric和zero-shot new-platform test，因此最合理是 **motion-prior + functional action transfer，部分executable transfer**。

## 3. Motion-aware dynamics：44 tokens 里有什么？

![METIS architecture：visual/motion tokenizers、EgoAtlas pretraining与action decoder](https://raw.githubusercontent.com/AureleoPKU/Metis/main/assets/architecture.png)

*图 2。官方架构图。Tokenizers先单独将视觉变化与hand motion量化，VLA再把其code indices当扩展词表监督。*

### 3.1 Visual dynamics branch

Encoder同时看当前/未来视觉、连续motion和learnable queries：

$$
\hat D_{vis}=\operatorname{Enc}_V(I_t,I_{t+k},P_{t:t+k},D_{vis}),
$$

量化为4个code indices，codebook size 16、feature dimension 128：

$$
D'_{vis}=\operatorname{VQ}(\hat D_{vis}),\qquad V=4,\quad |C_v|=16.
$$

Decoder以当前frame和tokens重建未来 **DINOv2 feature**，而不是raw pixels：

$$
\hat f_{t+k}^{DINO}=\operatorname{Dec}_V(f_t^{DINO},D'_{vis}).
$$

这能压制texture/pixel redundancy，却也意味着visual tokens捕捉的是DINO可见的semantic/spatial change，细小接触变形、滑动与force不一定保留。

### 3.2 Motion dynamics branch

$$
H_{t:t+k}=\operatorname{PoseNet}(M_{t:t+k}),\qquad
D_{mot}=\operatorname{RQ}(H_{t:t+k}),
$$

$$
\hat M_{t:t+k}=\operatorname{TCN}(D_{mot}).
$$

PoseNet结合multi-scale temporal convolution与trajectory self-attention；两层residual quantization从512-entry codebook产生40 tokens，统一embedding dimension 128。论文未报告codebook perplexity/usage、reconstruction error、human/robot code overlap、token semantic cluster或量化误差与IK失败的关系。

### 3.3 VLA supervision 与 continuous action

METIS扩展LLM vocabulary，让每个code index对应special token，autoregressive loss为：

$$
\mathcal L_{ar}=\mathbb E\left[-\sum_{i=1}^{44}
\log\pi_\phi(\hat a_{d,i}\mid o_t,l,a_{d,<i})\right].
$$

Action decoder以dynamics tokens、visual embeddings和当前proprioception为输入，multi-head attention pooling后预测30 future steps（1秒、30Hz）：

$$
\mathcal L=\mathcal L_{ar}+\lambda\mathcal L_{action}.
$$

论文未给 $\lambda$ 数值和 $\mathcal L_{action}$ 的明确范数/分布形式。附录又说post-training action chunk 32，与正文“30 future steps”不一致；复现应以code config为准并记录off-by-two来源。

## 4. 网络共享、reasoning 与推理链路

| 模块 | Human/Robot共享 | 输入 | 输出/目标 | 推理保留 |
| --- | --- | --- | --- | --- |
| SigLIP+DINOv2 hybrid encoder | 是 | $224\times224$ ego RGB | semantic+spatial tokens | 是 |
| Projection + LLaMA-2 7B | 是 | image/language/proprio/dynamics prefix | language或44 dynamics tokens | 是 |
| Visual tokenizer | 是 | current/future features+motion | 4 discrete labels | 否，预处理teacher |
| Motion tokenizer | 是 | pose/action chunk | 40 discrete labels | 训练label generator；部署不直接需要teacher |
| Reasoning head | LLM本体共享 | instruction、observation | `[BOA]`与subtask text | subtask transition时 |
| Dynamics head | LLM词表共享 | context | `[BOD]`+44 tokens | 是 |
| Action decoder | 是 | dynamics+vision+proprio | continuous action chunk | 是 |
| IK/retarget | platform-specific | wrists/fingertips | joint commands | 是 |

论文正文以Prismatic-7B初始化：SigLIP features 1024D、DINOv2 1152D，concatenate后project到LLaMA-2 7B的32-block decoder。RoboBrain-Dex release后来适配3B backbone，HF页面标4B package；这不是论文原架构的同义名称。

推理状态机：模型在subtask boundary预测 `[BOA]`，生成hand-level next subtask；预测 `[BOD]` 后生成dynamics tokens并执行action chunk。作者称只在transition reasoning降低latency，但没有公布server round-trip、token decode time、control replanning frequency或错过transition的fallback。Project page还明确说server communication有delay，4×/5× demos不能当real-time speed证据。

~~~mermaid
flowchart LR
    A["8-source EgoAtlas\nhuman + robot pose/action"] --> B["canonical 18D wrist\n+30D fingertips"]
    B --> C["visual VQ\n4 tokens / codebook 16"]
    B --> D["motion RQ\n40 tokens / codebook 512"]
    E["ego RGB + language + proprio"] --> F["SigLIP+DINOv2\nPrismatic-7B"]
    C --> F
    D --> F
    F --> G{"reason or act?"}
    G -->|"[BOA], only transition"| H["hand-level subtask text"]
    H --> G
    G -->|"[BOD]"| I["44 dynamics tokens"]
    I --> J["continuous action decoder\n30/32-step chunk"]
    J --> K["wrist/fingertip targets"]
    K --> L["platform IK/retarget"]
    L --> M["G1/Inspire or Sharpa execution"]
    M -. "RGB/proprio feedback" .-> E
~~~

## 5. 训练过程与数据梯度

| Stage | 数据 | 参数 | Batch/steps | Optimizer/算力 | 目的 |
| --- | --- | --- | --- | --- | --- |
| Dynamics construction | EgoAtlas motion/frames | VQ/RQ encoders/decoders | 未完整报告 | config在作者repo | 产生teacher codes |
| VLA pretraining | 8-source weighted EgoAtlas | vision、LLM、action decoder全量 | global768，60K | AdamW $2\times10^{-5}$，wd0，clip1；24 H100×72h | shared dynamics/action prior |
| Robot post-training | 每task100 G1 demos | vision+LLM LoRA rank32；action decoder全量 | 8GPU×4/device；steps未给 | AdamW $3.5\times10^{-4}$，wd$10^{-3}$；80%时LR×0.1 | target task/embodiment anchoring |
| Sharpa adaptation | Sharpa images+joints | 未报告 | demos/steps未报告 | 30/60Hz对齐 | 新hand tasks |

预训练约处理：

$$
60{,}000\times768=46.08\text{ million sampled training instances}.
$$

但每instance的frame horizon、重复采样、trajectory batching和language coverage未报告，不能换算成独立video hours或严格token count。Human与robot共用backbone/action decoder，确实产生shared-gradient pressure；没有domain-specific head或embodiment token的描述。反面是visual/proprio distributions仍可能让网络隐式分域，论文没有linear probe/domain alignment分析。

Post-training用bf16 vision/LLM、fp32 action decoder；image先center-crop到480×640再resize224²，proprio history 1，real data 30Hz。没有force/tactile history与显式error recovery；一秒open-loop-ish chunk如何滚动replan未说明。

## 6. 真机结果：绝对数、比较公平性与统计边界

### 6.1 六任务主表

| Method | Pick | Laptop | Drawer | Two Drinks SR/PSR | Cola SR/PSR | Drawer+Bread SR/PSR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ACT | 35 | 65 | **95** | 25 / 40 | 50 / 53.3 | 5 / 5 |
| OpenVLA-OFT | 50 | 80 | 10 | 40 / 57.5 | 55 / 56.7 | 0 / 1 |
| $\pi_{0.5}$ | 60 | 85 | 70 | 65 / 72.5 | **75** / 76.7 | 60 / 65 |
| GR00T N1.5 | 70 | 80 | 80 | 65 / 70 | 70 / 73.3 | 70 / 72.5 |
| METIS | **85** | **95** | 90 | **75 / 85** | 70 / **76.7** | **75 / 82.5** |

单位均为%，默认N=20，意味着5个百分点通常只是一条rollout。论文没有trial randomization、seed、confidence interval、annotator agreement或checkpoint selection protocol。METIS在六项SR平均约81.7%，GR00T约72.5%、$\pi_{0.5}$约69.2%；平均优势较清楚，单格5pp不稳健。

Baseline不是等架构/等compute：ACT训练600K steps、chunk100；OpenVLA-OFT 40K、chunk30、joint action；$\pi_{0.5}$ 30K、chunk10、wrist+joint；GR00T 10K、chunk16、冻结backbone；METIS用canonical fingertips且pretrained on EgoAtlas。它是systems comparison，不是只隔离human data的公平因果对照。

### 6.2 Efficiency 与 language

Pick仅用10% downstream data，即10 demos，成功率50%；完整100 demos为85%。没有robot-only foundation model的同data-size curve、seeds或其他task curve，能说明within-method sample efficiency，不能推出普适scaling law。

Instruction test让三种颜色水果共处桌面，每个task仍采100 demos并joint train，再用不同instructions评估。它证明已训练object-word grounding，不是unseen language composition或open-vocabulary zero-shot。

### 6.3 OOD与跨本体

| Method | Unseen bg | Unseen lighting | Unseen object | Clutter |
| --- | ---: | ---: | ---: | ---: |
| $\pi_{0.5}$ | 50 | **70** | 65 | 55 |
| GR00T N1.5 | 65 | 65 | 65 | 60 |
| METIS | **70** | 65 | **70** | **70** |

评测仅用Drawer+Bread task，shift是tablecloth、colored/flickering light、bread→croissant、附近放plate/apple。没有OOD task、novel skill composition或quantified shift magnitude；METIS平均68.75%，相对GR00T 63.75%只高5pp。

Sharpa Beta + 双22-DoF hand达到Apple→Basket 85%、Tool Use 70%。Appendix明确收集该platform 30Hz images和60Hzjoint state-action，再对齐到30Hz；没有说明只作calibration还是训练。缺少“G1-trained checkpoint直接zero-shot Sharpa”的对照，故只能判为 **adapter/schema portability + target post-training**。

## 7. Claim—Evidence 因果审计

| Claim | 所需对照 | 论文证据 | 充分性 | 替代解释 |
| --- | --- | --- | --- | --- |
| Human pretraining有效 | no/full human，固定robot data/steps | no 60/35；human 70/60 | 较强但仅2任务 | 更多tokens、scene diversity |
| Robot-inclusive multi-source更好 | human-only vs full | 70/60→85/75 | 中 | self-data quality、source mixture一起变化 |
| Dynamics tokens关键 | only continuous vs dynamics+continuous | 30/0→85/75 | 强支持auxiliary dynamics | 没有等容量/等loss alternative |
| Compact discrete design优于常规tokenization | 等token/compute baseline | 无 | 不充分 | 增益来自额外teacher supervision |
| Reasoning改善长任务 | no-CoT、错误subtask、latency | 无独立ablation | 不充分 | pretraining/action decoder即可解释 |
| OOD robust | 同task四种shift | 三baseline×四条件 | 中低，N/CI不明 | 单task、小差距 |
| Cross-embodiment | zero-shot与target-data curve | Sharpa 85/70 | 不充分支持zero-shot | new robot demos+IK/posttrain |
| Contact-rich dexterity | force/tactile/contact metrics | task final success | 间接 | Position control与demo先验吸收 |

缺失的关键factorial：human-only、robot-only、joint/sequential在等sampled tokens/steps下比较；随机source vs task-relevant source；无visual token vs无motion token分别消融；无reasoning；相同continuous latent capacity；target robot demo scaling across several tasks。

## 8. 三类 gap 与 semantic–motion–contact 对齐

| Gap/层级 | 机制 | 直接证据 | 剩余问题 | 判断 |
| --- | --- | --- | --- | --- |
| Embodiment | camera/wrist canonical frames、fingertips、FK/IK | G1与Sharpa success | 无zero-shot、IK/collision metrics | 部分 |
| Task | LLM instruction、manual subtasks、transition reasoning | 6 trained tasks、fruit instruction | 无new task composition | 部分偏弱 |
| Reality | 每task100 real G1 demos、真实posttrain、低层IK | 真机20 trials/task | 无dynamics/system-ID、force、latency | 靠anchor data吸收 |
| Semantic→subgoal | hand-level text | self10K labels | 无reasoning ablation/accuracy | 部分 |
| Subgoal→motion | dynamics tokens + action decoder | long-task PSR | 无token semantics/causal boundary | 部分 |
| Motion→contact | visual feedback+position trajectories | drawer/laptop/bimanual task success | 无contact/force/tactile | 隐式 |
| Contact→force/compliance | 无 | 无 | 全部缺失 | 否 |

METIS不使用whole-body pose、floating base或locomotion；“humanoid”只指G1上身固定操作。Reality gap不是joint-wise dynamics model或sim-to-real解决，而是直接target robot demonstration与现有IK/position controller吸收。

## 9. 失败模式与部署风险

- **IK ambiguity**：同一fingertip targets可有多个joint configurations，跨时间可能跳支或jitter；no-pretrain variant已观察到joint jitters。
- **Contact blindness**：RGB+pose无法区分touch、incipient slip和足够grip force；fingertip position也不描述palm/side contact。
- **Chunk delay**：30/32-step horizon配server inference；动态碰撞、掉落或drawer卡住时，论文没有event-triggered stop/recovery。
- **Reasoning drift**：错误subtask text可把动作头推向错误对象；没有verification或constraint layer。
- **Dataset shortcut**：EgoDex与self-glove视觉占pretrain大头；weighted mixture不等于domain invariance。
- **Bimanual safety**：无self-collision、object collision、joint limit或human proximity监督。
- **Artifact version drift**：Prismatic-7B与RoboBrain 3B/4B branches、30与32 chunks、Torch2.2与旧说明2.9不一致。
- **License/provenance**：EgoAtlas是逻辑合集，不是统一license corpus；商业/再分发前要逐源核验。

## 10. 面向人形、EX002 与灵巧手的落地建议

### 可直接复用

1. Canonical action record：timestamp、ego intrinsics/extrinsics、left/right wrist camera-frame SE(3)、每手五finger tips wrist-frame XYZ、validity mask、language/subtask。
2. Motion tokenizer作为pretraining auxiliary teacher，但先检查human/robot code usage与reconstruction error。
3. Source-weighted sampler，避免EgoDex类大源淹没高质量on-task/robot anchors。
4. Dynamics-token + continuous decoder双头；不要让discrete token直接越过safety controller驱动hardware。

### 必须重做

- 对parallel gripper或不同dexterous hand，重做hand FK/IK、wrist-hand fixed transform及collision-aware retarget；不能只替换joint count。
- 对EX002/legged humanoid加入base/torso/feet、balance和whole-body reach；当前48D不含locomotion。
- System0输入应是短时wrist/fingertip trajectory + contact intent + object-relative constraints，而不是裸44-token序列。
- 接触任务加入joint torque/current、tactile、estimated wrench、slip和contact phase；训练contact-conditioned decoder或独立fast reflex。
- 先做offline reachability filter：joint limits、self/environment collision、velocity/acceleration、IK residual；失败human samples可保留作visual/dynamics pretraining，但mask executable-action loss。

### 最小试点

| 项目 | 建议 |
| --- | --- |
| Task | 双手抓瓶→handover→放篮，含一次regrasp；固定base起步 |
| Human data | 5–10h多场景ego wrist+fingertips，保留tracking confidence |
| Robot anchor | 20/50/100 demos三档；另保留等量robot-only control |
| Factorial | no pretrain / human-only / robot-only / human+robot；continuous vsmotion tokens |
| Metrics | final success、subgoal PSR、IK residual/failure、collision、slip、peak current、latency |
| OOD | background、object geometry/friction、camera height、lighting分别测试 |
| 成功阈值 | 100 trials下full success≥80%，相对robot-only下95% CI有正增益；IK failure<0.5%，无硬碰撞 |

### System2–System0接口

| 层 | 输入 | 输出 | 频率建议 | Gap责任 |
| --- | --- | --- | ---: | --- |
| System2 | language、scene、history | verified subtask/object/contact phase | 1–2Hz或event | Task/semantic |
| System1 METIS | ego RGB、proprio、subtask | feasible wrist/fingertip trajectory distribution | 5–10Hz replan | motion/task |
| Retarget/WBC | canonical trajectory、robot model | collision/balance-aware joint trajectory | 50–200Hz | embodiment |
| Contact System0 | tactile/current/wrench/slip | compliant correction/abort | 200–1000Hz | contact/reality |
| Drive | joint targets/torques | actuator execution | hardware rate | reality |

## 11. 能力评分与最终判断

| 能力 | 评分（1–5） | 依据 |
| --- | ---: | --- |
| 视觉泛化 | 4 | 多源ego、四种OOD；但单task小样本 |
| 语义迁移 | 3 | subtask labels与fruit instruction；无新composition |
| 运动迁移 | 4 | human-only pretrain增益、canonical geometry、dynamics ablation |
| 动作可执行性 | 3 | 两platform真机；依赖target demos+IK，无constraints指标 |
| 接触迁移 | 2 | contact-rich success仅间接，无force/tactile/contact state |
| 跨本体 | 3 | schema可迁Sharpa；非zero-shot，data/steps不明 |
| 因果证据 | 3 | 两组强ablation但任务少、factorial不完整 |
| 真机可信度 | 3 | 6 tasks×20默认trials；无CI/seeds/recovery/locomotion |

> **最终一句话**：该方法在 multi-source EgoAtlas pretraining 阶段以 camera-relative wrists、wrist-relative fingertips 与44个motion-aware dynamics tokens引入 Human Ego 数据，主要迁移视觉—手部运动先验；no/human/full pretraining和dynamics supervision消融支持其价值，但尚未证明reasoning独立增益、zero-shot跨本体或contact/force transfer，部署仍依赖target robot post-training、platform-specific IK与低层安全控制。

