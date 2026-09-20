---
title: "In-N-On: Scaling Egocentric Manipulation with in-the-wild and on-task Data"
method_name: "Human0"
authors: [Xiongyi Cai, Ri-Zhao Qiu, Geng Chen, Lai Wei, Isabella Liu, Tianshu Huang, Xuxin Cheng, Xiaolong Wang]
year: 2025
venue: arXiv
tags: [egocentric-data, humanoid-manipulation, vision-language-action, flow-matching, human-robot-cotraining, domain-adaptation, dexterous-hands, few-shot-learning]
image_source: online
---

# In-N-On：千小时 in-the-wild 负责什么，20小时 on-task 又负责什么？

> 本笔记基于 [arXiv:2511.15704v1](https://arxiv.org/abs/2511.15704)、[HTML 全文](https://arxiv.org/html/2511.15704v1)、[项目页](https://xiongyicai.github.io/In-N-On/)、[Human0 GitHub](https://github.com/XiongyiCai/Human0)、[模型页](https://huggingface.co/XiongyiC/Human0) 与 [PHSD 数据页](https://huggingface.co/datasets/XiongyiC/PHSD) 精读和核验。公开资产状态核验日期为 **2026-08-13**。只找到匿名 ICLR 2026 submission，未找到官方接收记录，作者页和项目页仍列 arXiv 2025，因此 frontmatter 不沿用阅读列表的暂定 ICLR 2026。

## 阅读结论先行

In-N-On 最重要的贡献不是又把更多 Human Ego video 塞进 VLA，而是提出一个两阶段 **data cuisine**：

- **In-the-wild**：EgoDex、ActionNet、PH2D 合计 1,000+ 小时，覆盖人类和多个humanoid，负责预训练广泛视觉—语言—动作先验；
- **On-task**：20+ 小时 Apple Vision Pro / Aria human data 与 H1/G1 robot data，和目标任务、物体、语言、相机/动作标签更对齐，在 post-training 中继续以约70% human sampling co-train，避免只用robot fine-tune造成语义遗忘。

Human0 以 π₀ checkpoint 初始化，是 language-conditioned flow-matching VLA。不同human/robot states和actions先经IK/FK/hand-retargeting转成一套 human-centric representation：head SE(3)、head-relative双腕SE(3)、双手5个fingertips，parallel gripper可用thumb-index distance。SigLIP视觉、语言与pose tokens进入transformer，flow head预测未来human-centric action，部署时再retarget回robot。

论文对“统一action space是否自然得到统一representation”给出一个很有价值的反证：对vanilla model的中间visual+proprio tokens做linear probe，可以 **100%** 判别人还是robot。也就是说，即使坐标与schema统一，网络仍能利用arm appearance、camera、pose statistics等domain shortcut。Human0因此加入gradient reversal discriminator，令encoder对embodiment分类不利；在仅一条完整pouring robot demonstration的few-shot setup中，最终success从3/20提到5/20。

主结果最强的是 human-only language coverage transfer。Multi-object OOD targets/instructions只出现于human data，Human0达到30/30，而π₀是16/30、robot-only Human0是15/30。Burger OOD ingredients/instructions也只在human data，Human0 7/12，robot-only variant 2/12。附录更直接评估“向正确目标移动”：multi-object OOD 30/30、burger OOD 10/12。Human data确实把robot training未出现的object-language association带进部署。

但这不是 zero-shot 新行为。论文自己明确回答：只凭human data学全新behavior，在当前scale **不行**。Pouring仍需1条完整robot demo，Human0也只有25% full success；此外还加入30条left-grasp和30条right-grasp robot sequences，因此“1-shot”仅指完整bimanual pouring trajectory，而不是整个post-training只有一条robot record。Human提供语义、视觉、轨迹prior，robot anchor仍决定可执行contact与hand coordination。

数据规模的表述也要拆开。PHSD的1,000+小时并非全human：EgoDex 800+小时human，ActionNet 100+小时多为Fourier GR1T1 humanoid，PH2D同时含human和H1。On-task数据每任务human demos很多（2,545/1,016/750/727），robot也有120/180/80以及pouring 61条分段/完整数据。论文没有给collector数量、完整小时/episode/source比例、语言生成流程或每个data mixture精确sampling值，除post-training约70% human。

真机证据来自Unitree G1为主、H1为辅，均配Inspire 5-finger hands，但任务是固定站立的桌面/工位操作；head translation虽进入state-action schema，未评测locomotion、balance或whole-body control。Contact只由visual/action imitation间接学习，没有object state、contact point、force、tactile、collision或joint-limit loss。

最后，公开程度与论文承诺有明显落差：GitHub截至核验日只有README、MIT license和演示视频，没有training/retarget/eval code；PHSD Hugging Face repo只有2.49 kB且dataset为空；Human0 repo只有1.55 kB、空model card，没有weights。页面存在不等于artifact已发布，当前无法复现。

### 一句话总结

In-N-On证明了一条合理的Human Ego recipe：用1,000+小时mixed in-the-wild human/humanoid data预训练shared motion-language prior，再在task-aligned human+robot data上以human-heavy sampling post-train，并用GRL压制embodiment shortcut；它能把human-only object-language knowledge迁移到真机，但不能human-only获得新可执行skill，1-shot pouring也仅5/20且仍含额外robot grasp clips。

![In-N-On：大规模预训练、on-task post-training 与 Human0 三类能力](https://arxiv.org/html/2511.15704v1/teaser_v8.png)

*图 1。论文区分in-the-wild和on-task，分别承担base prior与目标分布/语义保持。*

## 0. 资源、版本、许可与可复现性

| 资产 | 入口 | 截至2026-08-13状态 | 许可 | 可用性 |
| --- | --- | --- | --- | --- |
| 论文 | [arXiv](https://arxiv.org/abs/2511.15704)、[HTML](https://arxiv.org/html/2511.15704v1) | v1，2025-11-19；无后续版本 | CC BY 4.0 | 高 |
| 项目页 | [In-N-On](https://xiongyicai.github.io/In-N-On/) | paper/videos/code/model/data links | 页面无独立license | 中高 |
| GitHub | [XiongyiCai/Human0](https://github.com/XiongyiCai/Human0) | 2 commits；README、license、videos；**无实现** | MIT仅覆盖repo现有内容 | 极低 |
| Human0 weights | [HF model](https://huggingface.co/XiongyiC/Human0) | 1.55 kB；只有attributes+33-byte README；**无weights** | CC BY-NC 4.0 tag | 无 |
| PHSD | [HF dataset](https://huggingface.co/datasets/XiongyiC/PHSD) | 2.49 kB；明确显示empty；**无data** | CC BY-NC 4.0 tag | 无 |
| Upstream datasets | EgoDex / ActionNet / PH2D | 各自独立发布状态/协议 | 不能由PHSD license覆盖 | 需逐源获取 |
| IK/FK/retarget suite | 论文称will release | GitHub未包含 | 未实际交付 | 无 |

论文写“plan to open-source / weights will be open-sourced”，不是已公开。即使未来PHSD汇总release用CC BY-NC 4.0，EgoDex、ActionNet、PH2D、Apple/Aria衍生数据和robot assets的上游条款仍应逐项核验。

## 1. 研究假设与数据分工

可检验假设是：

> 大规模异质in-the-wild human/humanoid data适合学习广覆盖VLA prior，但不适合直接作为target policy mixture；task-aligned on-task human+robot data应在post-training继续联合出现，并通过domain-adversarial alignment阻止模型按embodiment分裂，从而保留human中的language/object knowledge并降低robot adaptation样本需求。

~~~mermaid
flowchart LR
    A["In-the-wild\nEgoDex human 800+h"] --> D["human-centric schema\nhead+wrist+fingertips"]
    B["ActionNet humanoid 100+h"] --> D
    C["PH2D human+H1"] --> D
    D --> E["Stage 1 pretrain\n8×H200 100k steps"]
    F["On-task human\nAVP/Aria + language"] --> G["Stage 2 post-train\n约70% human sampling"]
    H["On-task G1/H1\nOpenTV robot demos"] --> G
    E --> G
    G --> I["SigLIP + language + pose\nflow-matching policy"]
    I --> J["GRL discriminator\n压制human/robot shortcut"]
    J --> K["human-centric future action"]
    K --> L["IK/FK/hand retarget\nG1/H1 execution"]
    L -. "无online correction / RL" .-> M["ID/OOD success"]
~~~

## 2. Gap—Evidence 总表

| Gap | 机制 | 直接证据 | 消融 | 真机 | 判断 |
| --- | --- | --- | --- | --- | --- |
| State/action morphology | head/wrist/fingertip human-centric space；IK/FK | H1/G1 execution | no-human variant | 有 | 部分解决 |
| Visual embodiment | image augmentation + SigLIP + GRL | linear probe 100%→约50% | no discriminator pouring 15→25% | 有 | 有改善，非完全 |
| Language/task | in-the-wild+on-task language co-training | human-only OOD instruction | robot-only/π₀/GR00T | 有 | object-language迁移强 |
| New behavior | flow motion prior + few robot data | pouring 5/20 | baselines0–2/20 | 有 | human-only不行 |
| Contact | robot demonstrations隐式 | final success | 无contact ablation | 有 | 未显式解决 |
| Reality | realH1/G1 data/rollout | 4 tasks | platform baselines | 有 | 固定站立平台内 |
| Locomotion/whole-body | head translation仅入schema | 无 | 无 | 无 | 未解决 |
| Scaling/data quality | In vs On两阶段 | no-human、robot fraction curve | 无严格In/On factorial | 有 | recipe合理，因果拆分不足 |

## 3. PHSD：Physical Humans-Humanoids Dataset

### 3.1 In-the-wild pre-training mixture

| Source | 规模/embodiment | 传感/动作 | 主要作用 | 注意 |
| --- | --- | --- | --- | --- |
| EgoDex | 800+ h human，Apple Vision Pro | head/wrist/finger keypoints + language | 最大human visual/motion source | 原论文829h；许可独立 |
| Fourier ActionNet | 100+ h humanoid，多为GR1T1 | robot joints/ego video；5-finger hands | robot/humanoid anchoring | 不是human data |
| PH2D | human AVP + Unitree H1 Inspire | head/wrist/hand与robot joints | bridge source | 规模在本文未拆 |

“PHSD 1,000+ hours of human-humanoid data”是准确说法，“1,000+ hours human video”会把ActionNet/PH2D robot部分错误算作human。论文Fig.4给data size ratio与sampler factor，但正文没有列精确数字；只能确认作者因human占比压倒性而手调pre-training sampler。

### 3.2 On-task post-training mixture

| Task | Human demos | Robot demos | Human/robot覆盖关系 |
| --- | ---: | ---: | --- |
| Single-object grasp | 2,545 | 120 | OOD的4个objects完全不在human data，用于object-level generalization |
| Multi-object grasp | 1,016 | 180 | 3 seen两域都有；3 OOD target/instruction仅human有 |
| Burger assembly | 750 | 80 | 2 seen两域都有；OOD ingredients/instruction仅human有 |
| Pouring | 727 | 1 full +30 left-grasp +30 right-grasp | full behavior一条，但robot component clips合计另60 |

Post-training按human约70%采样，不等于raw data 70%，也不是每task相同小时。Human demos来自AVP/Aria，robot demos通过AVP+OpenTV采集。论文abstract称20+小时，却没有给task-specific hours、episode duration/Hz、demonstrator数、success/failure比例或language annotation方式。

### 3.3 Privacy、bias与governance

- AVP/Aria可记录face、bystander、gaze/hand/head轨迹、室内场景、工位和屏幕；论文未报告consent/IRB/redaction/retention。
- “actual fast-food workers可低边际成本采集”目前是hypothetical部署论证；若真实采工人视频，劳动同意、雇佣压力、绩效监控和数据收益分配必须明确。
- EgoDex/PH2D的collector diversity与train/eval identity split可能影响scale claim；本文不报告统一provenance表。
- Language-only OOD objects其实存在于human on-task data，评测的是跨embodiment coverage，不是开放web knowledge。
- 成功示范占主导，failure/recovery、安全与多策略未系统建模。

## 4. Human-centric state/action representation

![同一human action经retarget suite映射到不同humanoids](https://arxiv.org/html/2511.15704v1/figs/humanoid.jpg)

*图 2。Pinocchio-based suite声称支持robot joints与human-centric表示双向转换，但代码尚未发布。*

| Component | 维度/坐标 | Human来源 | Robot转换 | 剩余gap |
| --- | --- | --- | --- | --- |
| Head | $T_{head}\in SE(3)$ world→head | wearable SLAM | FK/base/head | 无locomotion验证 |
| Wrists | $T_{L/R wrist}\in SE(3)$，relative to head | AVP/Aria | arm FK/IK | reach/joint limit |
| Fingertips | $P_{L/R finger}\in\mathbb R^{3\times5}$ | dense hand tracking | hand retarget | contact surface/force |
| Parallel gripper | $D_{L/R}\in\mathbb R$ optional | thumb-index distance | aperture mapping | 不适配全部gripper语义 |

论文一句“collector height difference can be neglected as data scales”没有实验证据。Head-relative wrists消除global位置，却不消除臂长、肩宽、可达空间、遮挡和hand-eye geometry。fingertips是跨hand的几何接口，不是functional contact：同一tip position不保证joint torque、collision、force closure或compliance。

Robot joints先FK到human-centric state/action，policy输出再IK/retarget到target robot。这能统一dataset columns，但模型、projection与controller共同决定可执行性；论文没有report retarget tracking error、IK failure、joint-limit/collision rate或H1↔G1 cross-platform table。

## 5. Human0 architecture 与 flow matching

![Human0两阶段方法：human-centric mapping、flow VLA与domain-adversarial alignment](https://arxiv.org/html/2511.15704v1/arch_v5.png)

*图 3。Human与robot共享policy，但domain discriminator反向迫使visual+pose feature不携带易分类的embodiment信息。*

SigLIP生成visual tokens $v\in\mathbb R^{L\times C}$，text embeddings为 $n\in\mathbb R^{T\times C}$，MLP将state变成pose latent $x\in\mathbb R^C$：

$$
z=\operatorname{Transformer}(v,n,x).
$$

对target action $a$，noise $u\sim\mathcal N(0,I)$，$t\sim U(0,1)$，插值：

$$
a_t=(1-t)u+ta,
$$

$$
\mathcal L_{FM}=\mathbb E\left[
\left\|f_\theta^{flow}(z,a_t,t)-(a-u)\right\|_2^2
\right].
$$

它从noise到action学习velocity field，而不是离散action tokens。基础VLA沿用π₀ checkpoint，但human-centric state/action projection因维度和含义改变而随机初始化。论文没有报告policy参数量、action horizon/rate、ODE integration steps、optimizer/LR/augment、language encoder具体选择或real-time latency。

### 5.1 Pre-training与post-training的共享/变化

| 项目 | Stage 1 in-the-wild | Stage 2 on-task |
| --- | --- | --- |
| 数据 | EgoDex+ActionNet+PH2D | 每task human+G1/H1 robot |
| 规模 | 1,000+h | 20+h、4 tasks |
| sampler | 手调balance，数值未表列 | 约70% human |
| 训练 | 8×H200、100k steps、batch160 | 1×H100、30k、batch10 |
| objective | flow matching + domain loss | 同一objective |
| 目的 | broad human-humanoid VLA prior | task alignment、language preservation、execution |

作者称single-H100 post-training“democratize”训练，只适用于已有base checkpoint的adaptation；从头获得Human0仍需8 H200且当前checkpoint未实际提供。

## 6. Domain adaptation：统一schema仍会泄露embodiment

![Vanilla feature可被linear probe完美识别人/机器人](https://arxiv.org/html/2511.15704v1/figs/confusion_matrix.png)

*图 4。100% probe accuracy说明model有domain shortcut，但不等同shortcut一定导致task failure。*

视觉tokens与pose latent经attention聚合：

$$
m=\operatorname{Attn}(\operatorname{Concat}(v,x)).
$$

Discriminator loss：

$$
\mathcal L_D=-\mathbb E\log D_\phi(m_h)
-\mathbb E\log(1-D_\phi(m_r)).
$$

GRL使discriminator最小化分类loss，而policy encoder最大化它：

$$
\min_\phi\max_\theta\mathcal L_D,
\qquad
\mathcal L_{final}=\mathcal L_{FM}+0.1\mathcal L_D.
$$

形式上最后一式的sign依赖GRL内部反向；若直接最小化正的 $+0.1\mathcal L_D$ 会让encoder更易分类。复现必须实现GRL而不是照抄scalar loss。

加GRL后linear probe probability约居中50%，pouring full 3/20→5/20。这个增益只有2次成功、N=20，无seeds/CI；right grasp反而17→16，left 5→7、pour3→5。它支持domain alignment帮助少样本后段执行，但不足以证明“越domain invariant越好”。Action/kinematics中与embodiment有关的信息对feasibility本来有用，理想representation应分离transferable intent与platform constraints，而非删除所有embodiment信息。

## 7. 四项真机任务与主结果

![Burger、Pouring、Multi-object等真实G1/H1评测](https://arxiv.org/html/2511.15704v1/real_robot.png)

*图 5。绝大多数data/experiment在Unitree G1；H1也被使用，但论文未分平台报告全部结果。*

| Method | Single ID | Single OOD | Multi ID | Multi OOD | Burger ID | Burger OOD | Pouring |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| π₀ | 19/20 | 19/20 | 25/30 | 16/30 | 5/12 | 3/12 | 0/20 |
| GR00T N1 | 18/20 | 13/20 | 6/30 | 8/30 | 4/12 | 3/12 | 0/20 |
| HAT + human | 17/20 | 15/20 | — | — | — | — | 2/20 |
| Human0 w/o human | 18/20 | 18/20 | 23/30 | 15/30 | 7/12 | 2/12 | 2/20 |
| **Human0** | **20/20** | **19/20** | **29/30** | **30/30** | **8/12** | **7/12** | **5/20** |

Single-object中human几乎没带来增益：w/o human 18/20→20/20 ID、18/20→19/20 OOD，而且这些OOD objects从未出现在human data，主要测试base visual generalization。

真正强证据是Multi OOD：15/30→30/30，human-only target+instruction coverage被成功迁移；相对π₀也+14/30。Burger OOD是2/12→7/12，absolute +5 successes；task长且工具/ingredient交互更难。Pouring是2/20→5/20，成功率仍低。

表中没有training seeds、confidence intervals或significance test。不同baseline是否获得相同post-training data/compute和其最佳action representation，描述不够完整；π₀又是Human0 initialization，因此比较同时涉及extra 100k pretrain、data recipe、new projections和GRL。

### 7.1 OOD到底是什么

| Setting | Robot data | Human data | 测试解释 |
| --- | --- | --- | --- |
| Single OOD | unseen object/height/scene | objects也unseen human | object/visual generalization |
| Multi OOD | target/instruction unseen | **seen in human** | cross-embodiment language coverage |
| Burger OOD | ingredient/instruction unseen | **seen in human** | cross-embodiment tool/semantic coverage |
| Novel background | original tablecloth | task data仍有objects/actions | visual robustness |

论文措辞“zero-shot language following”是合理的robot-domain定义：instruction未见robot data。但不是整个training corpus未见，也不是新behavior；human on-task data明确包含这些object-language-action sequences。

附录multi-target指标只要求“moves toward correct object”，不是完成grasp/assembly：Multi ID/OOD 29/30、30/30，Burger ID/OOD 10/12、10/12。它更干净地测language grounding，却不能替代full manipulation success。

### 7.2 Background robustness

| Background | Multi ID | Multi OOD |
| --- | ---: | ---: |
| White original | 29/30 | 30/30 |
| Black tablecloth | 26/30 | 29/30 |
| Floral tablecloth | 24/30 | 28/30 |

背景变化下降有限，支持visual robustness；但三块tablecloth、单task、无baselines或human/no-human comparison，无法归因给Human data或in-the-wild pretrain。

## 8. Few-shot不是zero-shot：Pouring case study

论文最值得肯定的一句话是：“Can the robot learn a completely new behavior from just human data? Empirically, no.” Human0提供prior，但需要一条target robot complete demo。

| 配置 | Robot data | Human data | Full success |
| --- | --- | --- | ---: |
| π₀ | post-training setup | 可能无同recipe | 0/20 |
| Human0 w/o human | 1 full + component clips | 无human | 2/20 |
| Human0 no GRL | 同robot data | 727human | 3/20 |
| Human0 + GRL | 同robot data | 727human | 5/20 |

所谓one-shot还包含30 left-grasp和30 right-grasp robot demos；它们可能不含完整pour，但仍给了hand/object/contact/configuration anchoring。最准确说法是 **one full task demonstration +60 component demonstrations**。而5/20说明能偶尔执行，不是已经可靠学习。

## 9. 数据混合、scaling与negative transfer

本文的核心recipe很有实践意义：

1. source极异质、大但与target较远 → 放pre-training；
2. source与target同task/objects、动作标注可靠 → 在post-training持续mix；
3. robot data少 → human-heavy sampler（约70%）保持语义；
4. model仍按domain分裂 → GRL regularization。

但“systematic analysis”仍缺一个完整factorial：只In、只On、In→robot-only、In→On-human-only、In+On naïve all-at-once、不同70/30 ratios、等steps/data/compute，以及upstream π₀和从随机初始化。没有这些，无法分别量化1000h pretraining、20h on-task、human-centric schema、extra optimization与GRL。

Robot fraction curve仅在simple single-object上以百分比显示，正文没给exact points/seeds。Negative transfer以catastrophic forgetting作为动机，但没有直接展示language ability在robot-only fine-tune前后的遗忘曲线。

## 10. Semantic–motion、Contact与Task gap

| 表征层 | Human supervision | Robot supervision | 对齐机制 | 未解决 |
| --- | --- | --- | --- | --- |
| Language | in-the-wild/on-task instruction | on-task instruction | SigLIP+transformer | paraphrase/negation/composition |
| Object/goal | RGB+instruction | RGB+instruction | shared VLA | 无explicit object state |
| Motion | wrists/fingertips/head | FK human-centric action | shared flow head | feasibility由retarget吸收 |
| Contact | fingertips间接 | robot demo隐式 | 无 | contact point/normal/phase |
| Force/compliance | 无 | 无显式 | 无 | pour/grasp安全 |
| Whole-body | head translation schema | H1/G1 head/base state | representation only | 无locomotion/balance |

Human0的语义迁移证据强于多数Ego policy：有distractors、human-only target words和robot execution。但task semantics仍是单步object choice/已收集burger composition，未测自由语言planning、新动作verb、multi-turn correction或同object不同contact goal。

Burger与pouring失败会累积且无recovery；附录明确说早期抓取误差会snowball。Flow action chunk不是hierarchical planner，语言不能自动提供contact feedback。

## 11. Reality gap 与平台边界

| 差异 | 处理 | 证据 | 剩余风险 |
| --- | --- | --- | --- |
| Human→humanoid kinematics | FK/IK/hand retarget | G1/H1 real rollout | tracking/IK metrics缺 |
| Camera/appearance | egocentric data、SigLIP、GRL | background/OOD | camera/device split未测 |
| Hand morphology | fingertips human-centric | Inspire5-finger | 不同hand/gripper证据弱 |
| Dynamics/contact | robot post-training | grasp/burger/pour | 无force/tactile/controller |
| Platform | ActionNet GR1T1、PH2D H1、target G1/H1 | mostly G1 | 无platform-separated results |
| Locomotion | head SE(3)含translation | 无 | Reality gap未触及 |

本文没有sim-to-real；real target data直接缩小Reality gap。但目标humanoid固定站立，策略输出的是上肢/手 manipulation，不是dynamic whole-body。不能把“human-centric head translation supports potential loco-manip”写成已经证明。

## 12. System 2 / 1 / 0 接口

| 层 | 输入 | 输出 | 训练数据 | Gap责任 |
| --- | --- | --- | --- | --- |
| System 2 | language + ego visual | transformer task/goal latent | π₀ + human/robot language | semantic/task |
| System 1 | latent + head/wrist/finger state | flow future human-centric action | mixed In/On | motion/embodiment prior |
| Projection | wrist/fingertips/head goal | target joints/hand | Pinocchio retarget | kinematic feasibility |
| System 0 | robot target + current state | actuator/contact | 未报告controller | dynamics/contact/safety |

慢层给快层的仍是pose/fingertip trajectory，不是object-centric contact goal、force、compliance或constraints。对EX002/新hand，projection和System0必须重建，shared Human0只适合作为high-level/trajectory prior。

## 13. Claim—Evidence 因果审计

| Claim | 所需对照 | 论文证据 | 充分度 | 限定 |
| --- | --- | --- | --- | --- |
| Human提升robot | same model no human | Human0 vs w/o human | 高 | extra human data非等compute |
| Human带来language-only coverage | target仅human出现 | Multi/Burger OOD | 强 | 是coverage transfer非new behavior |
| In+On recipe优于单一阶段 | full factorial | 未完整提供 | 中低 | 核心recipe因果未拆 |
| GRL对齐embodiment | probe + task ablation | 100%→50%，3→5/20 | 中 | N小、无seeds |
| 1-shot learning | only1robottrajectory | 1full+60component | 需降调 | full SR25% |
| 统一action支持多humanoid | per-platform transfer | mostly G1，提及H1 | 中低 | 无H1/G1表 |
| Background robust | multiple backgrounds/baselines | 3tablecloth、single task | 中 | 无归因消融 |
| Scale产生新能力 | scale curve | 只有base/nohuman与robot fraction | 中低 | 非严格emergence/scaling law |

最关键的补充实验是固定training steps/compute，做2×2 In-the-wild pretrain × On-task human post-train，再交叉GRL on/off；对language-only OOD报告no-language、shuffled-language与human-video-without-action；pouring把robot data严格分为0/1/5/10 full demos且不加component clips。

## 14. 开源复现分层

### Level 1：当前可做——论文/视频复核

只能核对method、数字与videos。GitHub没有source，HF没有weights/data，不能运行inference。

### Level 2：等待artifact——Human0 inference

需要实际checkpoint、architecture/config、action normalization、ODE solver、retarget assets、G1/H1 urdf与controller。HF的license tag不是weights。

### Level 3：Post-training

论文称1 H100、30k、batch10可行；仍需20h on-task data或自行采集、exact 70/30 sampler、optimizer/LR/loss/augmentation与language schema。成功标准先复现Multi OOD human-only instruction advantage。

### Level 4：Base pretraining

8 H200、100k、batch160；还需获得EgoDex/ActionNet/PH2D及各自license，重建timestamp sync、240×320 processing和human-centric retarget。成本与数据治理都高。

## 15. 面向EX002/humanoid/灵巧手的建议

| 目标 | 可复用 | 必须重做 | 数据建议 | 首个pilot |
| --- | --- | --- | --- | --- |
| G1/H1 Inspire | schema、Human0思想、on-taskmix | 等artifact/标定/controller | human-heavy On + robotanchors | multi-object language grasp |
| EX002不同arms | SigLIP/language/motionprior | FK/IK、camera、reach、joints | 先100+robot demos + matchedhuman | fixed-base pick/place |
| 不同dexterous hand | fingertip goals | retarget、contact/force、joint limits | hand-pose + robot grasp/contact | rigid object grasp |
| Parallel gripper | wrist+thumb-index aperture | aperture/contact normal | on-taskhuman + gripper robot | language-conditioned pick |
| Loco-manipulation | head SE(3)字段 | WBC、balance、feet/base control | full-body human + locomotionrobot | 暂不直接部署 |

落地时不要以GRL抹掉所有embodiment信息。建议把latent拆成task/object/contact-invariant intent和platform-conditioned feasibility两支：前者从human规模化，后者接robot kinematics/dynamics/controller。接触任务再加object pose、contact region/normal、desired force/compliance和failure recovery。

## 16. 十个组会质疑

1. In-the-wild 1,000+h和On-task 20+h各贡献多少，为什么没有完整2×2实验？
2. 100% embodiment probe是否只是合理地读到robot arm像素，而非“cheating”？
3. GRL去掉platform cues会不会损害reachability、joint limit与contact feasibility？
4. “1-shot pouring”为什么另含30 left+30 right robot clips？
5. Human-only能迁移object-language，为何human-only仍不能学new behavior？瓶颈在哪？
6. Multi OOD 30/30是否由scene/object shortcut而非语言决定？shuffled/no-language结果呢？
7. Head-relative wrists为何能忽略collector height，scale就会自动解决吗？
8. H1和G1各有多少data/rollout，跨平台到底是否zero/few-shot？
9. post-training 70% human是如何选择的，其他ratio/negative transfer曲线在哪？
10. paper承诺open source后，为什么code repo和HF model/dataset仍为空？

## 17. 能力评分与最终判断

| 能力 | 1–5 | 依据 |
| --- | ---: | --- |
| Language coverage transfer | 4.5 | human-only targets：Multi30/30、Burger7/12 |
| 视觉/物体泛化 | 4 | background与single/multi OOD |
| Motion prior | 4 | 1,000+h action-labeled、human-centric flow |
| 新behavior zero-shot | 1.5 | 作者明确no；pouring需robot anchor |
| Few-shot执行 | 2.5 | 1full+60component，5/20 |
| Contact/灵巧性 | 2 | Inspire hands真机，pose-only且success低 |
| 跨本体 | 3.5 | human、GR1T1、H1、G1；mostly G1，无分平台表 |
| 因果证据 | 3 | no-human、GRL、baselines；缺In/On factorial/seeds |
| 真机可信度 | 4 | 四tasks、ID/OOD；trial数12–30 |
| 开源复现 | 1 | links存在但code/weights/data实际为空 |

### 最终判断

In-N-On 把Human Ego pipeline从“先human pretrain、再robot fine-tune”推进到更合理的数据分层：远域但多样的human/humanoid data建立base，近域on-task human data在post-training持续提供language/object/behavior coverage，robot data锚定target execution，GRL抑制domain shortcut。Multi-object和Burger OOD结果是human-only语义进入robot policy的直接证据。

最准确的总结是：

> Human0 在 **in-the-wild mixed pre-training + on-task human-heavy co-post-training** 两阶段，以 **head/wrist/fingertip human-centric action和language-conditioned flow matching** 引入Human Ego；它主要迁移 **object-language coverage、视觉鲁棒性和motion prior**，而target robot的接触可执行性仍依赖robot demonstrations、IK/hand retarget与controller。GRL改善representation alignment，但新behavior zero-shot、whole-body locomotion、contact modeling和跨平台泛化均未解决。

这篇论文值得作为大规模Human Ego VLA的数据recipe基线，也给EX002/G1项目一个实用方向：大量In负责base、少量matched On负责语义与任务、robot anchor负责可执行性。当前最大阻碍不是理念，而是所有核心release链接仍是空壳；在artifact真正发布与关键factorial完成前，不应把它当成可直接部署的open foundation model。
