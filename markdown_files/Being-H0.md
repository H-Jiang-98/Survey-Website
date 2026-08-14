---
title: "Being-H0: Vision-Language-Action Pretraining from Large-Scale Human Videos"
method_name: "Being-H0"
authors: [Hao Luo, Yicheng Feng, Wanpeng Zhang, Sipeng Zheng, Ye Wang, Haoqi Yuan, Jiazheng Liu, Chaoyi Xu, Qin Jin, Zongqing Lu]
year: 2026
venue: ICML
tags: [egocentric-data, vision-language-action, dexterous-manipulation, human-video-pretraining, motion-tokenization, mano, physical-instruction-tuning, human-to-robot]
image_source: online
---

# Being-H0：把人手当“foundation manipulator”，究竟迁移了动作还是表征？

> 本笔记基于 [arXiv:2507.15597v1](https://arxiv.org/abs/2507.15597)、[HTML 全文](https://arxiv.org/html/2507.15597)、[项目页](https://beingbeyond.github.io/Being-H0/)、[MIT代码库](https://github.com/BeingBeyond/Being-H0)与[模型集合](https://huggingface.co/collections/BeingBeyond/being-h0)精读并核验。初稿上传于2025-07-21；作者代码库在2026-05-01公告已接收 **ICML 2026**，故frontmatter使用最新书目信息。公开资产核验日期为 **2026-08-13**。

## 阅读结论先行

Being-H0提出“Physical Instruction Tuning”：先把11种human hand datasets统一成MANO motion，用motion tokenizer把手势轨迹变成LMM可以autoregress的离散“物理语言”；再把预训练VLA当encoder，通过target-robot proprio projector、learnable action queries和MLP regression head，在50–100条真实robot demonstrations上post-train。Human data直接监督的是 **MANO wrist/finger motion tokens**，robot执行动作则来自一个新学的continuous head，两者共享vision-language-motion backbone，却不是human token到robot command的固定映射。

UniHand的规模必须分三层说。底层是444.1K source sequences、130M frames、1,155小时；pipeline把视频切成最多10秒chunks和overlapping 1秒windows，用Gemini生成多尺度描述与模板任务，扩成166.5M instructional instances；实际因计算限制只平衡采样 **UniHand-2.5M** 来预训练。因而“150M samples”不是150M独立videos，也不是全部被模型看过；大多数instruction samples高度重叠，且约42.4%的生成instances来自EgoDex、约10.7%来自只有0.3K sequences的ARCTIC，instance count不等于独立信息量。

它的动作表示比只用fingertips丰富：MANO-D162包含finger 6D rotations、global wrist 6D rotation、translation，并用21 joints作reconstruction auxiliary feature。Part-level GRQ把wrist与fingers分组，8层residual quantization、每部件4096-entry codebook；每秒15帧、temporal downsample 4，论文称每手每秒128 tokens。Tokenizer reconstruction在最佳配置达到约5–6 mm MPJPE，但这只是重建seen representation，不等于从单幅RGB+instruction生成未来绝对3D motion也有毫米精度；实际visual-grounded 14B generation的MPJPE是6.87 cm（head split）和8.11 cm（tail）。

Physical space alignment解决的是camera distribution，而非physics：用intrinsics把不同视频warp到目标weak-perspective camera；鱼眼先归一到90° FoV；再同步改变hand depth/image scale和绕optical axis旋转，平衡view-motion distribution。它没有object mesh/pose、metric depth、contact point、force、tactile或camera motion prediction。作者自己把depth/tactile列为future work。

Human→robot transfer也远非“seamless”。论文明确说kinematic differences阻止motion tokens直接迁移，因此完全绕过MANO decoder去控制robot：把robot proprio编码进backbone，learnable action queries读取context，MLP输出Franka end-effector poses与Inspire joint positions；所有backbone/action modules在post-training中可训练，以L1 imitation loss拟合robot expert。它完成的是 **motion-prior/representation transfer**，由target robot demonstrations重新学习executable mapping；不是MANO→Inspire retarget，也不是human-only robot policy。

最有价值的因果对照是Being-H0 vs相同InternVL3 architecture/scale、同一批post-training data。七个条件成功率分别提高0、10、10、35、35、45、30个百分点；例如Close-Lid从25%到60%，Unfold-Clothes从45%到75%。这支持human physical instruction pretraining带来有效初始化。可是每格只有20 trials，5个百分点是一条rollout；没有confidence intervals/seeds，而且“physical instruction tuning”同时包含motion data、generated language、camera alignment、2.5M extra training和motion token objectives，不能把增益只归因给explicit tokenization。

与GR00T N1.5相比，Being-H0在6/7条件更好或相等，但差距多为5–25pp；作者没有给等pretraining data/compute或GR00T针对该domain的完整recipe，因而它是system comparison。更关键的direct causal controls缺失：random human data、no MANO action而保留images/text、unaligned cameras、human-only robot execution、robot-only from scratch、frozen-backbone transfer、sequential vsjoint co-training。

任务是真实Franka FR3单臂+Inspire 6-DoF hand，L515 RGB输入，固定桌面操作；不是humanoid、bimanual、mobile或loco-manipulation。Pour-Cup达到100%、Close-Toolbox 85%、Unfold-Clothes75%，但post-training包含每任务50–100条target demos。没有control rate、chunk length（release eval默认16，不一定是论文设置）、latency、IK/WBC、force/torque、collision、recovery或安全统计。

“scaling law”也应写成有限范围的monotonic trend：1B→8B→14B提升motion format validity与生成指标，training samples增至2.5M时大多指标改善；但图把最终点归一为100%，没有拟合power law、held-out compute-optimal frontier或robot success scaling。最大data时PA-MPJPE反而略降，语义指标继续升，表明scale可能以finger precision换semantic plausibility。

公开程度中等偏高：MIT repo有motion/VLA/robot inference，HF提供1B/8B/14B weights、8B Align、motion tokenizer与10,804-row/1.17GB post-training dataset。UniHand完整166.5M没有统一release；到核验日只有2.85M-size `UniHand_Preview`。README的TODO仍列training scripts、real-robot development、simulation benchmark等，且MANO需另行注册并受独立license约束。所以可以复现inference和部分post-training format，不能完整重建pretraining corpus与论文训练。

### 一句话总结

Being-H0在human-video pretraining阶段用MANO part-level motion tokens把人手轨迹变成LMM的“物理语言”，主要迁移的是camera-aligned 3D hand-motion与语义表征；同架构同robot-data的InternVL3对照支持其sample-efficiency价值，但它没有把human actions直接变成robot commands，真实执行依赖50–100条target demonstrations和一个重新训练的continuous action head，也未证明contact/force或跨机器人本体迁移。

![Being-H0总览：human motion pretraining到robot action post-training](https://raw.githubusercontent.com/BeingBeyond/Being-H0/main/docs/assets/image/01_arch.webp)

*图 1。官方overview清楚区分autoregressive human motion pretraining与query-based robot action post-training。*

## 0. 资源、许可与复现性

| 资产 | 入口 | 状态 | 许可/限制 | 复现判断 |
| --- | --- | --- | --- | --- |
| 论文 | [arXiv](https://arxiv.org/abs/2507.15597)、[HTML](https://arxiv.org/html/2507.15597) | v1；作者公告ICML 2026 | arXiv perpetual non-exclusive | 证据完整 |
| 项目/视频 | [Being-H0](https://beingbeyond.github.io/Being-H0/) | interactive motion+6 robot demos | template CC BY-SA 4.0不等于data license | 定性可核验 |
| 代码 | [GitHub](https://github.com/BeingBeyond/Being-H0) | 21 commits；motion/VLA/server/eval | MIT；MANO另许可 | inference可做 |
| Weights | [HF collection](https://huggingface.co/collections/BeingBeyond/being-h0) | 1B/8B/14B、8B Align、GRVQ | model cards简略 | 可下载 |
| Post-train data | [h0_post_train_db_2508](https://huggingface.co/datasets/BeingBeyond/h0_post_train_db_2508) | 10,804 rows、1.17GB | 页面未清楚拆任务/license provenance | 部分可用 |
| UniHand | [BeingBeyond datasets](https://huggingface.co/BeingBeyond/datasets) | preview约2.85M；非full166.5M | 11个上游datasets各自条款 | 无法完整重建 |
| MANO | [official site](https://mano.is.tue.mpg.de/) | 需注册下载 | MANO专用license | 自动复现阻碍 |
| Training | repo TODO | 完整scripts/config未交付 | 无exact checkpoint selection | 不完整 |

README同一页面一方面列post-training data已发布，另一方面TODO仍写Post-training data/Inference code，说明文档没有随release完全清理；应以实际files而非TODO checkbox单独判断。

## 1. 研究命题与精确可检验假设

作者最强假设可写为：

> 在target robot demonstrations、VLA architecture和post-training protocol固定时，加入大规模camera-aligned Human Ego MANO motion instruction pretraining，会使shared visual-language backbone获得可迁移的3D hand-motion prior，从而提高dexterous robot的ID/OOD success和少样本效率。

这不是co-training：human data在Stage 1，robot data在Stage 2，属于 **sequential pretrain→post-train**。论文没有在post-training继续混human batches，也没有action-space shared head。

~~~mermaid
flowchart LR
    A["11 human datasets\nMocap / VR / RGB-only"] --> B["MANO standardization\nHaMeR pseudo-pose + smoothing"]
    B --> C["camera weak-perspective alignment\ndepth-scale + in-plane rotate"]
    D["Gemini hierarchical labels\nchunk + 1s windows"] --> E["166.5M generated instructions"]
    C --> E
    E --> F["balanced UniHand-2.5M"]
    F --> G["part-level GRQ\nwrist/finger motion tokens"]
    G --> H["InternVL3 1B/8B/14B\nnext-token pretraining"]
    I["50–100 target robot demos/task"] --> J["proprio MLP + action queries\ncontinuous L1 action head"]
    H --> J
    J --> K["Franka EEF + Inspire joints"]
    K -. "RGB/proprio feedback" .-> J
~~~

## 2. UniHand：source records与instruction expansion要分开

| Source | Instructions | Sequences | Frames | Hours | Pose quality/annotation |
| --- | ---: | ---: | ---: | ---: | --- |
| ARCTIC | 17.9M | 0.3K | 245K | 2.3 | joints+pose, action |
| FPHA | 0.798M | 1.2K | 105K | 1.0 | joints, action |
| HoloAssist | 8.0M | 2.2K | 17.1M | 166 | joints, segment |
| H2O | 3.7M | 0.9K | 115K | 1.1 | joints+pose, action |
| HOI4D | 21.2M | 3.0K | 825K | 7.6 | joints+pose, action |
| HOT3D | 8.7M | 2.8K | 420K | 3.9 | joints+pose, no source text |
| OAKINK2 | 18.5M | 2.8K | 695K | 6.5 | joints+pose, action |
| TACO | 11.5M | 2.2K | 340K | 3.2 | joints+pose, action |
| DexYCB | 3.6M | 5.6K | 410K | 3.8 | joints+pose, no source text |
| Taste-Rob | 1.9M | 85K | 14M | 130 | nojoint/pose，trajectory pseudo-label |
| EgoDex | 70.6M | 338K | 89.6M | 829.4 | joints，trajectory |
| **Total** | **166.5M** | **444.1K** | **130M** | **1,155** | heterogeneous |

`#Inst/#Seq`极不均匀，例如ARCTIC约59,667 generated instructions/source sequence，EgoDex约209。这来自sliding windows、多个hand modes、三类tasks和template variants，不表示ARCTIC有EgoDex同等多的独立场景。训练只从全池balanced sample 2.5M（约1.5%）。

### 2.1 Pose与language curation

- 原数据已有MANO时直接取；只有3D joints时gradient-based MANO fitting；无3D时HaMeR逐帧估计。
- 检测left/right swap和pose discontinuity，短gap temporal interpolation；fitting加joint-angle constraints与smoothness。
- 视频切为不重叠≤10秒chunks，以2FPS frames喂Gemini-2.5-Flash-Lite；产生chunk-level imperative/summary。
- 每chunk再做overlapping 1秒windows，描述contact state、object attributes、hand parts和camera-relative trajectory；contact是LLM视觉文字描述，不是sensor contact ground truth。
- 三类training task：instruction→motion、motion→text、context/history→next motion。每类约20 base templates，再用Gemini-2.5-Pro扩写。

隐私/治理缺口：各原dataset有自己的consent、bystander和室内场景条款；论文没有统一provenance/identity split、LLM annotation audit、hallucination rate或manual QA percentage。把“public source”变成MANO+Gemini derivative也不自动获得统一再分发许可。

## 3. Physical space alignment：对齐投影，不对齐物理

对source camera intrinsics $(f_x,f_y,c_x,c_y)$ 与target $(f'_x,f'_y,c'_x,c'_y)$：

$$
s_x=\frac{f'_x}{f_x},\quad s_y=\frac{f'_y}{f_y},\quad
\Delta x=c'_x-s_xc_x,\quad\Delta y=c'_y-s_yc_y,
$$

$$
u'=s_xu+\Delta x,\qquad v'=s_yv+\Delta y.
$$

严重鱼眼先把FoV归一到90°，再crop/pad到target resolution。View balancing同步变换motion和image：

$$
\tau_z'=\lambda_s\tau_z,
\qquad I'=\operatorname{Resize}(I,1/\lambda_s),
$$

$$
\tau'_c=R_z(\varphi)\tau_c,\qquad R'_c=R_z(\varphi)R_c.
$$

这保持weak-perspective projection一致，但近距离hand的perspective、object depth/geometry、occlusion和camera ego-motion并未真实变换。删掉balance后tail MPJPE从9.02升到12.13cm、tail M2T R@3从18.7降到10.3%，说明camera distribution engineering很关键；它同时改变training distribution，不能解释为model自动获得view invariance。

## 4. Part-level GRQ motion language

### 4.1 MANO-D162

模型比较了D51/D99/D109/D114/D162，最终选择D162：finger joints与global rotation用6D rotations，translation 3D，再附加21×3 joint positions作为reconstruction auxiliary。Shape $\beta$ 不进入最终D162主体。作者发现axis-angle在整体wrist error更低，但6D fingers更利于VLA generation。

GRQ对latent channel分组，每组做L层residual quantization：

$$
r_0=z_i^{(g)},\qquad
q_l=\arg\min_{c\in\mathcal C^{(g)}}\|r_{l-1}-c\|_2,
\qquad r_l=r_{l-1}-q_l,
$$

$$
\hat z_i^{(g)}=\sum_{l=1}^{L}q_l.
$$

Tokenizer loss：

$$
\mathcal L_{tok}=\mathcal L_{recon}
+0.02\mathcal L_{commit}+1.0\mathcal L_{wrist}.
$$

Default为wrist/finger part-level、8 RQ layers、2 groups、code dimension512、每part codebook4096；15FPS、1秒window、downsample4，作者报告每手每秒128tokens。Batch2048、LR $2\times10^{-4}$。

“毫米级”只对应tokenizer oracle reconstruction（约0.5–0.6cm MPJPE），不是foundation VLA生成。Vision-grounded generation即使14B在head/tail也是6.87/8.11cm，tail wrist translation7.41cm；这对自由空间hand prior合理，对毫米接触机器人控制远远不够。

### 4.2 Unified next-token objective

Vision、text、motion hidden statesconcatenate并共享attention projections：

$$
[Q,K,V]_{v,t,m}=[W_QH_{v,t,m},W_KH_{v,t,m},W_VH_{v,t,m}].
$$

Motion blocks以 `<MOT>...</MOT>` 分隔。Vocabulary masking以50%概率在motion label处屏蔽非motion logits；token loss只保留15–95 percentile，丢掉过易static与过难jitter/outlier：

$$
\widetilde L=\{\ell_i\mid Q_{15}\le\ell_i\le Q_{95}\},\qquad
\mathcal L_{motion}=\frac1{|\widetilde L|}\sum_{\ell_i\in\widetilde L}\ell_i.
$$

这提高motion gradient concentration，但也可能系统性忽略rare/high-dynamic动作；论文没有比较失败/recovery tail或不mask的robot transfer。

## 5. Robot post-training：human tokens并不直接执行

![Physical Instruction Tuning：motion parameters与robot action parameters分阶段扩展](https://raw.githubusercontent.com/BeingBeyond/Being-H0/main/docs/assets/image/02_phy_inst_tune.webp)

*图 2。红色motion attention属于human pretraining；绿色action queries/head在robot post-training加入。*

Robot context由RGB、language与proprio embedding组成，learnable queries $q_i$ 读取共享backbone，MLP输出continuous action：

$$
a_i=f_r\left(\Theta\left(q_i,\operatorname{ctx}\oplus f_p(p_t)\right)\right),
$$

$$
\mathcal L_{robot}=\frac1{N_a}\sum_{i=1}^{N_a}\|a_i-a_i^*\|_1.
$$

Trainable包含foundation VLA $\Theta$、queries、proprio projector与regression head；因此post-training可能重写human representation，论文没给freeze/LoRA/full ablation或catastrophic-forgetting test。Motion-token decoder不在robot action path中，没有MANO→robot retarget loss、latent alignment、contrastive/domain loss或action consistency。

| 模块 | Human stage | Robot stage | 共享/专属 | 推理保留 |
| --- | --- | --- | --- | --- |
| InternVL3 ViT+LLM | vision/text/motion autoregression | RGB/language encoder | 共享且可finetune | 是 |
| Motion embeddings/token head | MANO codes | 作为pretrained prior | human-centric | robot policy不生成 |
| Proprio projector | 无 | robot state→embedding | robot专属 | 是 |
| Action queries | 无 | chunk queries | robot专属 | 是 |
| Regression head | 无 | EEF+hand joints | embodiment专属 | 是 |
| IK/controller | 未详述 | hardware execution | platform专属 | 是 |

结论：shared parameters造成representation transfer，但action supervision没有统一；human data不是robot action-head gradient source。

## 6. 训练、scaling与数据效率

Foundation VLA用InternVL3 1B/8B/14B，image448²，AdamW LR $10^{-5}$、batch128、32×A800-80G，jointly finetune ViT adapter与LLM。论文未报告epochs/steps、训练时长、weight decay、warmup或不同scale compute。

| Model | Free-format valid | Motion→text R@3 head/tail |
| --- | ---: | ---: |
| 1B | 64.8% | 12.5 / 14.3 |
| 8B | 99.8% | 18.4 / 19.7 |
| 14B | 100% | 19.0 / 22.1 |

格式validity随scale剧增，部分是大模型更会遵循 `<MOT>` grammar，不等同于motion更精确。Motion ground-truth retrieval upper reference只有33.5/42.7，14B仍有明显semantic gap。

Data scaling到2.5M时多数normalized metrics稳步提升，但只固定8B、固定curation recipe且同一generated pool subsample；没有random human、source diversity固定或compute fixed control，所以“more examples”同时可能意味着more task/object combinations。

Robot data效率图比较25/50/100% demos；论文只给部分具体点，如Close-Lid 25%时InternVL3 0%、Being-H0 15%。曲线支持低数据优势，但每task base总量50–100不一，百分比不是统一episode数。

## 7. 真机评测与统计边界

Hardware：Franka Research 3 7-DoF arm、Inspire 6-DoF hand、RealSense L515 RGB；teleop由Gello arm exoskeleton + D435i human-hand estimation/retarget采集。每task50–100demos、20 randomized trials。

| Method | Toy seen | Toy unseen | Toy clutter | Toolbox | Lid | Pour | Cloth |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GR00T N1.5 | 75 | 40 | 50 | 80 | 50 | 90 | 60 |
| InternVL3 | 55 | 55 | 50 | 50 | 25 | 55 | 45 |
| Being-H0 | **75** | **65** | **60** | **85** | **60** | **100** | **75** |

Being-H0 vs InternVL3是最接近causal pretraining comparison；平均从47.9%升74.3%，+26.4pp。GR00T平均63.6%，Being-H0 +10.7pp。N=20意味着二项估计宽，paper未给CI；Pour 100%也只表示20/20，不代表failure概率为0。

所谓OOD只有novel toy color/visual properties和clutter；human UniHand是否包含类似duck、cup/lid、cloth skill无法排除。无new instruction composition、new task、new robot、different camera或force/material OOD。全部是slow fixed-base position-controlled manipulation。

## 8. Claim—Evidence审计

| Claim | 需要对照 | 现有证据 | 是否充分 | 替代解释 |
| --- | --- | --- | --- | --- |
| Human motion pretraining提升robot | same arch/data no-pretrain | InternVL3 47.9→H0 74.3 avg | 较强 | 2.5M extra vision/text、alignment一起变化 |
| Explicit tokens优于implicit latent | 同human data/compute tokenizer对比 | GR00T system comparison | 不充分 | data、architecture、recipe不同 |
| Part-level tokenizer最好 | token reconstruction+generation variants | Table5/6 | 对motion generation较强 | 未测robot downstream variant |
| View balancing有用 | with/without | tail MPJPE/R@3大降 | 强 | augmentation本身增sample diversity |
| 150M-scale learning | scale curve到2.5M | monotonic normalized curve | 有限 | 没训练166.5M；非power law |
| Human hand是universal manipulator | 多robot/gripper验证 | 只Franka+Inspire | 不充分 | single embodiment |
| Millimeter precision transfer | generated+robot metric | tokenizer reconstruction约mm | 不充分 | VLA generation是cm、无tracking/contact error |
| Contact semantics迁移 | contact labels+force task | LLM captions和final success | 很弱 | 无sensor contact/force supervision |

## 9. 三类gap与控制现实

| Gap | 方法 | 证据 | 剩余问题 | 判断 |
| --- | --- | --- | --- | --- |
| Embodiment | shared backbone + new robot head | singleFranka/Inspire success | 无多embodiment、无retarget error | 部分 |
| Task | language-motion instructions | toy target、6skills | 无new skill composition | 部分偏弱 |
| Reality | realtarget demos与direct BC | 7 conditions真机 | 无dynamics/force/recovery | 由robot data吸收 |
| Semantic→motion | Gemini labels+motion tokens | retrieval/generation metrics | labels可能hallucinate | 中 |
| Motion→action | pretrainedencoder→MLP queries | InternVL3 control | 无explicit alignment | 间接 |
| Motion→contact | RGB/MANO共现 | final success | 无contact point/topology | 很弱 |
| Force/compliance | 无 | 无 | 全缺 | 否 |

Human MANO motion并不含object state或contact target；per-second caption里的“contact state”是semantic text。人手轨迹可能穿物、误估深度或使用机器人不可实现的palm contact，pipeline没有把uncertainty传到action head。

## 10. 面向人形、EX002与灵巧手的落地

### 可复用模块

- UniHand schema：ego intrinsics、RGB、MANO wrist/finger pose、hand validity/confidence、language、source/task provenance。
- Part-level tokenizer，分别监控wrist/finger reconstruction、code usage和tail error。
- View-consistent augmentation；必须同时变image、intrinsics和3D labels。
- Contextual motion prediction auxiliary objective，比只做instruction→motion更稳定。

### 需要重做

- Human motion prior进入System1可以，但robot action head必须按embodiment重训；人形还要base/torso/whole-body reach/balance。
- EX002若是parallel gripper，应抽取object-relative wrist trajectory与grasp aperture/contact goal，不必生成完整MANO fingers。
- 多指手需用collision/torque-aware retarget或diffusion head，而非单纯L1平均多模态grasp。
- 对HaMeR低置信/occluded samples：可用于vision-language learning，mask精确action loss或按uncertainty加权。
- 加入tactile/current/wrench，将contact intent与motion trajectory分层；MANO token只作prior。

### 最小因果试点

| 项目 | 建议 |
| --- | --- |
| Task | drawer handle pull、cloth edge pinch、cup regrasp三项 |
| Robot anchor | 每task10/25/50 demos，固定train steps |
| Human pool | random vsretrieved on-task各1/5/20h |
| Controls | VLM-only、human RGB+text、MANO token、continuousMANO auxiliary、robot-only |
| Metrics | 100 trials、95%CI、action/IK error、contact/slip、force、collision、recovery |
| Transfer | 至少gripper和dexterous hand两embodiments，测zero-shot与每档anchor |
| 成功阈值 | human-action variant对RGB/text-only有显著增益；不增加collision，25demos≥robot-only50demos |

System interface建议：

| 层 | 输入 | 输出 | 频率 | 责任 |
| --- | --- | --- | ---: | --- |
| System2 | language、scene | object/subgoal/contact intent | event/1Hz | Task |
| Human-motion prior | subgoal、RGB | distribution over wrist/finger/object-relative motion | 2–10Hz | semantic-motion |
| Embodiment head | robot state、motion prior | feasible EEF/hand trajectory | 10–30Hz | embodiment |
| WBC/IK | trajectory、constraints | joints | 100–500Hz | reach/collision/balance |
| Contact System0 | tactile/wrench/current | compliant correction/abort | 200–1000Hz | contact/reality |

## 11. 最终评分

| 能力 | 1–5 | 理由 |
| --- | ---: | --- |
| 视觉泛化 | 4 | 11sources+view balance；robot OOD窄 |
| 语义迁移 | 4 | 双向motion-language、target instruction；大量synthetic labels |
| 运动迁移 | 4 | strong controlled robot gain、detailed tokenizer |
| 动作可执行性 | 3 | 真机多tasks；靠target head/demos，接口不透明 |
| 接触迁移 | 2 | 只有视觉共现与binary success |
| 跨本体 | 2 | 只一个robot configuration |
| 因果证据 | 3 | same-arch baseline强；关键factorials缺失 |
| 真机可信度 | 3 | 7conditions×20；无CI/latency/safety/recovery |

> **最终一句话**：该方法在human-video pretraining阶段以part-level MANO motion tokens引入Human Ego数据，主要迁移camera-aligned 3D手部运动与语义先验；同架构InternVL3对照支持robot成功率和少样本增益，但尚未证明human action直接映射、multi-embodiment或contact/force transfer，其部署依赖50–100条target robot demos、重新训练的query/MLP action head与现有低层控制器。

