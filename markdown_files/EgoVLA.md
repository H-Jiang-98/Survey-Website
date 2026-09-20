---
title: "EgoVLA: Learning Vision-Language-Action Models from Egocentric Human Videos"
method_name: "EgoVLA"
authors: [Ruihan Yang, Qinxi Yu, Yecheng Wu, Rui Yan, Borui Li, An-Chieh Cheng, Xueyan Zou, Yunhao Fang, Xuxin Cheng, Ri-Zhao Qiu, Hongxu Yin, Sifei Liu, Song Han, Yao Lu, Xiaolong Wang]
year: 2025
venue: arXiv
tags: [egocentric-data, vision-language-action, human-video-pretraining, humanoid, MANO, action-retargeting, inverse-kinematics, dexterous-manipulation, simulation-benchmark]
image_source: online
---

# EgoVLA：Human Ego 预训练到底迁移了动作，还是迁移了视觉与语义？

> 本笔记基于 [arXiv:2507.12440v3](https://arxiv.org/abs/2507.12440)、[HTML 全文](https://arxiv.org/html/2507.12440v3)、[项目页](https://rchalyang.github.io/EgoVLA/)、[训练与评测代码](https://github.com/RchalYang/EgoVLA_Release)、[Ego Humanoid Manipulation Benchmark](https://github.com/quincy-u/Ego_Humanoid_Manipulation_Benchmark)、[simulation demonstrations](https://huggingface.co/datasets/EgoVLA/EgoVLA-Humanoid-Sim) 与公开 checkpoints 精读和核验。公开资产状态核验日期为 **2026-08-13**。当前 v3 为 2025-07-18 更新的 27 页版本；arXiv、项目页、代码和当前书目记录均未确认 ICLR 2026 接收，故 frontmatter 采用 `venue: arXiv`，而不是沿用阅读列表中的暂定会议信息。

## 阅读结论先行

EgoVLA 的核心不是把任意 human video 一次性转换为 robot command，而是做一个**顺序式跨本体训练**：先用四个现有 Ego 数据集的约 50 万 image–action pairs，把 NVILA-2B 训练成预测 human 双腕 SE(3) 与 MANO hand trajectory 的 Human VLA；再把 12 个仿真 robot tasks 的手动作反向拟合到同一 MANO 空间，用每任务 100 条 robot demonstrations 做 115 epochs post-training；部署时，预测 wrist pose 经 IK，MANO 指尖经小型 MLP 变成 Inspire hand joint command。

这比只用 human RGB 做 representation pretraining 更强：human wrist 和 hand pose 是 30 Hz、1 秒 action chunk 的直接回归监督，语言也进入 VLM。人类预训练后，在未见 background 的仿真测试中，short-horizon mean success 从 no-pretrain 的 51.28% 提升到 69.11%，long-horizon 从 11.21% 提升到 28.79%；seen long-horizon 从 26.67% 提升到 45.93%。同一 NVILA backbone、相同 robot post-training 的对照支持 human stage 有贡献。

但最关键的负结果同样清楚：**human-pretrained model 在所有 humanoid tasks 上 zero-shot success 都是 0%**。把 robot demonstrations 从每任务 100 条减至 50 条，seen long-horizon mean success 又从 45.93% 暴跌到 7.41%。因此 human VLA 不是“天然已经是一套 robot policy”；它提供 visual-semantic-motion initialization，仍需大量 task-specific robot anchoring 才能变成可执行策略。

所有 policy 结果都来自 Isaac Lab 中的 Unitree H1 + 两只 Inspire hands，没有真机。benchmark 特意声明不是为了 direct sim-to-real；PD、IK、collision/contact dynamics 和 perfect simulator state utilities 吸收了 Reality gap。论文虽称 humanoid manipulation，实际控制 36 维 upper-body action：两臂 end-effector + 双手 joints，下肢固定，没有 locomotion、balance、whole-body policy 或真实 contact sensing。

语义证据也需降调。论文通过固定人类图像、替换语言 instruction，展示 predicted wrist trajectory 随指令改变；这是有价值的 counterfactual qualitative evidence，却没有语言成功率、paraphrase robustness、对抗指令或 robot-side instruction ablation。12 个 robot task 的 instruction 是 immediate behavior label，每个 task 又有专属 demonstrations；尚不能证明语言在长程中动态组合技能。

最后，MANO 统一的是**手形与腕部几何**，不是接触功能。目标函数只有 wrist translation、rotation 和 MANO PCA regression；没有 object state、contact point/normal、force、tactile、penetration、joint-limit、balance 或 task success loss。retargeting MLP 的平均 fingertip position error $5\times10^{-5}$ m 很小，但这只说明几何重构，不等于 grasp force、滑移稳定性或真实可执行性。

### 一句话总结

EgoVLA 在 human-video pretraining 阶段用 wrist + MANO action supervision 为 NVILA-2B 注入跨场景的视觉—语义—手运动先验，再靠每任务 100 条仿真 robot demonstrations 把先验锚定为 H1 policy；它显著改善 simulation ID/OOD，但 human-only 为 0%、无真机和 contact loss，说明完成的是 motion-prior transfer，而非 zero-shot executable action transfer。

### Elevator pitch

人类 Ego 数据有丰富场景、工具和双手行为，robot demonstrations 有可执行性但昂贵。EgoVLA 先学“人在这个画面和指令下，未来一秒双手会怎么动”，再让 robot demonstrations 也用 MANO 描述，使模型不用换 action head 就能继续训练。公共 MANO 空间像一个翻译层：robot action 先译成人手参数监督模型，模型输出再译回 IK 和 hand joints。结果证明这个初始化比只从 VLM 开始更好，但翻译层无法替代 robot adaptation、接触控制或 sim-to-real。

![EgoVLA：human motion prediction 与 humanoid control 共用 wrist + hand action representation](https://arxiv.org/html/2507.12440v3/EgoVLA-Teaser-V5-medium.png)

*图 1。上排是 human prediction，下排是 simulated humanoid rollout；论文没有真实机器人实验。*

## 0. 资源、版本、许可与公开程度

| 资产 | 入口 | 截至 2026-08-13 的状态 | 许可/缺口 | 复现判断 |
| --- | --- | --- | --- | --- |
| 论文 | [arXiv](https://arxiv.org/abs/2507.12440)、[HTML v3](https://arxiv.org/html/2507.12440v3) | v1–v3 均在 2025-07-16 至 07-18；27 页；当前仅可核验为 preprint | CC BY 4.0 | 高 |
| 项目页 | [EgoVLA](https://rchalyang.github.io/EgoVLA/) | 方法、benchmark、视频、论文、代码入口齐全 | 页面无独立 artifact manifest | 高 |
| 训练/评测代码 | [RchalYang/EgoVLA_Release](https://github.com/RchalYang/EgoVLA_Release) | public；约 59k Python lines（大量来自 vendored VILA）；含四数据集 preprocessing、train/eval、IK/retargeting weights | MIT，VILA 部分 Apache-2.0；README 明示“not fully tested”及 hard-coded path 风险 | 中高 |
| Benchmark | [Ego_Humanoid_Manipulation_Benchmark](https://github.com/quincy-u/Ego_Humanoid_Manipulation_Benchmark) | public；12 Isaac Lab tasks；要求 Isaac Lab 1.2.0 / Isaac Sim 4.2.0，作者提醒新版 physics 行为不同 | MIT；asset 通过 HF submodule | 中高 |
| Robot demonstrations | [EgoVLA-Humanoid-Sim](https://huggingface.co/datasets/EgoVLA/EgoVLA-Humanoid-Sim) | public、ungated；约 518.9 GB；2,008 files；revision `b59009b7307be1b09e819c5bd012c1205b403150` | MIT dataset card | 高但体量大 |
| Base VLM | [egovla_base_vlm](https://huggingface.co/rchal97/egovla_base_vlm) | public；约 3.15 GB | Apache-2.0 card | 高 |
| Human-pretrained checkpoint | [ego_vla_human_video_pretrained](https://huggingface.co/rchal97/ego_vla_human_video_pretrained) | public；约 4.18 GB | Apache-2.0 card | 高 |
| Final EgoVLA checkpoint | [egovla](https://huggingface.co/rchal97/egovla) | public；约 4.18 GB | **无 model card / license tag** | 中高 |
| Human processed mixture | 仓库 preprocessing scripts | 没有单一打包发布；需分别下载 HOI4D、HOT3D、HoloAssist、TACO，注册 MANO 后自行处理 | 继承四数据集与 MANO 的不同协议；不可用代码 MIT 覆盖 | 中低 |

代码和 benchmark 的 Python 静态 `compileall` 均通过。仓库内有若干 `human_plan/test` 脚本，VILA 子目录保留上游 tests/CI，但未发现 EgoVLA full pipeline 的锁定 CI；不能把 vendored VILA test coverage 等同于 EgoVLA preprocessing、32-GPU training 和 Isaac rollout 已自动验证。

## 1. 论文速览与核心假设

| 项目 | 判断 |
| --- | --- |
| 路线 | sequential human action pretraining → robot post-training；不是 simultaneous co-training |
| Human data | 四个已有 Ego datasets，约 500k sampled image-action pairs；3 FPS sampling |
| Robot data | Isaac Lab 12 tasks × 100 successful demos；100–500 frames/episode |
| Backbone | NVILA-2B + SigLIP；全模型 fine-tune；约 300M transformer action head |
| Action | 每手 3D wrist translation + rot6D + 15D MANO PCA；两手共 48D；30 steps/1 s |
| 平台 | simulated Unitree H1 + 2× Inspire hands；upper-body only |
| 评测 | 7 short + 5 long tasks；3 seen/22 unseen visual backgrounds；randomized object positions |
| 最强 claim | human action pretraining improves both simulated ID and novel-background success under fixed robot post-training |
| 最可疑 claim | “Human VLA is inherently already a robot policy”与“learns semantic intent”容易忽略 0% zero-shot、task-specific post-training 和纯 qualitative language evidence |
| 精读优先级 | 5/5：Human Ego → VLA action pretrain → humanoid simulation 的完整系统 |
| 复现优先级 | 4/5：checkpoint、sim data、code 较齐；32 A100 与 upstream human preprocessing 成本高 |
| 真机试点优先级 | 2/5：无真机、无 contact/force、安全和 balance 证据 |

可检验假设是：

> 在 NVILA initialization、robot demonstration 数量、robot post-training 配置和 simulation task 固定时，先用带 wrist/MANO action label 的多源 human Ego data 预训练，可通过视觉、语言和手运动共享表征提高目标 humanoid 的 ID 与 novel-background manipulation success。

`EgoVLA-NoPretrain` 是相当直接的支持对照；但论文没有 human RGB-only、shuffled human action、MANO-only、等计算量 VLM continued-pretraining 或 random/non-manipulation Ego 对照，所以还不能分离视觉多样性、语言、motion labels 和额外 optimization 的贡献。

## 2. Gap—Evidence 总表

| Gap | 作者机制 | 直接指标 | 消融 | 真机证据 | 是否解决 |
| --- | --- | --- | --- | --- | --- |
| Embodiment | camera-frame wrist pose；MANO shared hand space；robot→MANO optimization；MANO→robot MLP；arm IK | fingertip error $5\times10^{-5}$ m；sim success | no-pretrain、human-only 0%、50% robot data | 无 | 部分：几何接口，非零样本可执行迁移 |
| Task | NVILA language + multi-source skills；single generalist robot model；action chunk | 7 atomic + 5 multi-stage simulation tasks | no task-composition、language ablation | 无 | 部分：多任务共享，未证明新组合 |
| Reality | Isaac Lab benchmark + PD/IK | simulator success | 无 sim2real/domain randomization/real post-training | 无 | 否；benchmark声明不做 direct sim-to-real |
| Semantic–motion | immediate language instruction + predicted wrist/MANO | qualitative counterfactual human trajectories | 无 quantitative language/no-language test | 无 | 部分、证据弱 |
| Motion–contact | MANO/fingertip geometry；sim contacts only用于physics | final success/PSR 间接 | 无 contact/force/tactile loss | 无 | 否 |

## 3. Human 与 Robot 数据对照

![Human mixture：TACO 13%、HoloAssist 25%、HOI4D 39%、HOT3D 23%](https://arxiv.org/html/2507.12440v3/human_data_split.svg)

*图 3。比例是最终 sampled image-action pairs，而不是各 raw dataset 的小时占比。*

### 3.1 四个 human source

| 数据源 | 论文采用规模/特征 | Wrist/hand | Camera/object | Language | 在 EgoVLA 中的角色 | 许可注意 |
| --- | --- | --- | --- | --- | --- | --- |
| [HOI4D](https://hoi4d.github.io/) | 4,000 sequences；官方为 2.4M RGB-D frames、9 人、800 objects、610 rooms | frame-wise 3D hand pose | camera parameters、object pose/mesh、scene cloud 可用 | task descriptions | 最大 sampled share，39% | CC BY-NC 4.0，非商业限制 |
| [HOT3D](https://github.com/facebookresearch/hot3d) | 833 min；33 rigid objects | precise UmeTrack/MANO hands | Aria/Quest camera + objects | **无 task labels**，用 placeholder | 23%；精确 pose 和设备多样性 | 专用 HOT3D agreement；MANO 另需接受许可 |
| [HoloAssist](https://holoassist.github.io/) | 166 h（官方页 169 h）、350 instructor–performer pairs；复杂装配/维护 | noisy hand pose；双手丰富 | HoloLens RGB-D、gaze/IMU 等原数据 | action + spoken instruction | 论文称 uniform sample 1/10，最终 25% | CDLA v2；包含对话与多人影像治理 |
| [TACO](https://taco2024.github.io/) | 2,317（项目约 2.5k）sequences，151 tool–action–object triplets | precise bimanual MANO/meshes | ego + third-person、object meshes | action labels | 13%；工具和组合 | 项目网站 CC BY-SA 4.0 不必然等于全部数据资产许可，需查下载协议 |

训练按 3 FPS 从处理后序列构造约 500k pairs。HoloAssist raw 最大且 label 噪声较高，作者说只取 1/10 以免支配；公开 script 的 mixture 名称却写 `holoassist_train_30hz_sub5`，与论文文字存在命名/版本差异，复现时应以实际 dataset registry 中 `sub5` 定义和论文 commit 为准。

### 3.2 Human 与 Robot 的统一/不统一

| 维度 | Human data | Robot data | 是否统一 | 对齐机制 | 剩余 gap |
| --- | --- | --- | --- | --- | --- |
| 视觉 | 多种 Ego RGB、人体手臂 | simulated ego RGB，机器人手臂 | 分辨率/输入形式统一 | SigLIP + full VLM fine-tuning | appearance、rendering、camera height/FOV 不同 |
| 历史 | 当前 + 前 5 帧，间隔 0.2 s | 同 schema | 是 | 1 s image history | 3 FPS sample 与 30 Hz action label temporal mismatch 需插值/读取 |
| Wrist | human wrist SE(3) | robot EE 经 transform | 是 | world→current camera；robot frame transform | calibration、workspace、IK 可达性 |
| Hand | 15D MANO PCA/hand | Inspire joints 先拟合 MANO | 表征统一 | fingertip SmoothL1 optimization | morphology、contact patch、actuation差异 |
| Action horizon | 30 future steps / 1 s | 30 future steps / 1 s | 是 | 30 Hz chunk | contact phase/latency未显式建模 |
| Language | HOI4D/Holo/TACO labels；HOT3D placeholder | task description | 字符串接口统一 | NVILA tokens | label粒度/质量不同，placeholder无语义 |
| Object state | upstream有些 source 可用但模型不消费 | benchmark内部有pose/contact | **未统一** | RGB隐式 | 无 object-centric transition/action |
| Contact/force | 未输入 | benchmark可提供contact force，但 EgoVLA不使用 | 否 | task success间接 | 无 contact intent、force、tactile |

Human action 不是 joint command，而是 wrist + MANO motion label；robot data 也被“翻译”成这个 human representation。推理输出再翻译回 robot。它完成的是 **functional geometric motion-prior transfer**，不是 raw executable action sharing。

## 4. Video → Human Action → Robot Behavior 的完整链路

![EgoVLA 模型：六帧历史、语言、proprioception 与 action query tokens 进入 NVILA + 300M action head](https://arxiv.org/html/2507.12440v3/VLM-Pipeline-V13-Proprio.png)

*图 2。VLM 和 visual encoder 不是 frozen；human pretraining 与 robot post-training 都更新全模型。*

~~~mermaid
flowchart LR
    A["4 upstream Ego datasets\nRGB + camera + wrist/hand + language"] -->|"license / missing labels / noisy pose"| B["3 FPS sample + six-frame history"]
    B -->|"ego-motion / calibration"| C["Future wrists projected\ninto current camera frame"]
    C --> D["15D MANO PCA per hand\n+ wrist translation/rot6D"]
    E["Isaac teleop demos\n12 tasks × 100"] -->|"robot FK + coordinate transform"| F["Robot wrists in shared frame"]
    E -->|"fingertip fitting error"| G["Robot hand joints → MANO"]
    D --> H["NVILA-2B + 300M action head\n20-epoch human pretraining"]
    F --> I["115-epoch robot post-training"]
    G --> I
    H --> I
    I --> J["30-step / 1-second action chunk"]
    J -->|"IK feasibility / joint limits"| K["Arm joint targets"]
    J -->|"MANO FK + learned MLP"| L["Inspire hand actuation"]
    K --> M["Isaac PD execution at 30 Hz"]
    L --> M
    M -. "no real rollout / no data flywheel" .-> N["SR / subtask PSR"]
~~~

这条链路没有显式 video keyframe detection、scene reconstruction、object pose tracking、contact phase inference、physics-aware filtering、failure relabeling 或 real post-training。upstream datasets 即便含 RGB-D/object/contact-like annotation，EgoVLA 也只消费 Ego RGB、wrist/hand/camera pose 和 language。

## 5. 统一动作空间与 Embodiment gap

![MANO shared action：robot hand actuation 先拟合成 MANO，模型输出再通过 fingertip MLP 回到 joints](https://arxiv.org/html/2507.12440v3/VLM-AlignActionSpace-V4.png)

*图 4。双向映射共享 fingertip geometry，却不保证接触 wrench 或 robot feasibility。*

### 5.1 48D action 的精确组成

单手每个时间步为：

$$
a_t^{hand}=\left[T_t^{wrist}\in\mathbb R^3,\;
R_t^{wrist,6D}\in\mathbb R^6,\;
\Theta_t^{MANO}\in\mathbb R^{15}\right]\in\mathbb R^{24}.
$$

双手合并为 48D，预测 30 steps：

$$
A_t=[a_t,a_{t+1},\ldots,a_{t+29}]\in\mathbb R^{30\times48}.
$$

MANO 原始 15 个 ball joints 有 45 DoF，论文只用前 15 个 PCA components，并固定 average hand shape。压缩可减少维度，却丢失个体骨长/shape、多峰 hand configuration 和精细接触几何。

### 5.2 Robot → MANO

对每只 robot hand，优化 15D MANO 参数，使五个指尖逼近 robot fingertips：

$$
\Theta^*=\arg\min_{\Theta\in\mathbb R^{15}}
\frac{1}{5}\sum_{i=1}^{5}
\operatorname{SmoothL1}\left(J_{MANO}(\Theta)_i,J_{robot,i}\right).
$$

这使 robot demonstrations 可继续训练原 action head，无需 robot-specific head/reinitialization。问题是只约束五点：同样 fingertips 可对应不同中间关节、掌姿和自碰配置；指腹面积、关节 torque 和接触法向都不受约束。

### 5.3 MANO → Robot

部署时先用 MANO FK 得 wrist-frame fingertips，再送入四层 MLP，hidden sizes `[64,128,64]`，输出双手所有 actuation。MLP 用 robot demonstrations 生成的 paired representation 训练 2,000 epochs、batch 2,048、LR $10^{-3}$。论文报告 mean fingertip error：

$$
\operatorname{MPE}_{finger}=5\times10^{-5}\;\mathrm m=0.05\;\mathrm{mm}.
$$

如此小的数值很可能是在同一生成映射/分布上的拟合误差；论文未给 held-out hand configuration、碰撞、joint-limit、force closure 或实际 grasp success 的 retargeter-only 对照。raw demo 经映射回放仍有效是更实用的 sanity check，但发生在 simulator 与已收集轨迹上。

### 5.4 Morphology、workspace、balance 与 contact

| 子 gap | 处理 | 结论 |
| --- | --- | --- |
| Arm morphology | wrist target + IK | 隐藏 joint topology；无 reachability/collision loss |
| Hand topology | 五指 fingertips + MLP | 只验证 Inspire hands，不支持平行夹爪/不同指数量证据 |
| Camera | current camera frame wrist | 消除一部分 ego-motion；human/robot FOV/height仍不同 |
| Workspace | robot post-training | 靠 target-domain demos 学；human-only 0%说明 canonicalization不足 |
| Balance/base | 下肢固定，不输出 base | 未解决 humanoid balance/locomotion |
| Contact | simulator contacts存在但策略不看 | 未对齐；由 PD、几何和 imitation 间接吸收 |

## 6. 网络共享、训练阶段与 objective

### 6.1 模块共享表

| 模块 | Human/Robot 是否共享 | 输入 | 输出 | 目标 | 推理保留 |
| --- | --- | --- | --- | --- | --- |
| SigLIP visual tower | sequentially shared | 6×384² RGB | visual tokens | action regression梯度 | 是 |
| NVILA language model | sequentially shared | immediate instruction + image tokens | multimodal latent | action regression梯度 | 是 |
| Proprio MLP | shared schema | current wrist/MANO | state embedding | action loss | 是 |
| Action query tokens | shared | vocabulary最后30 IDs | 30 latent queries | action loss | 是 |
| 300M transformer action head | 完全共享 | VLM/query/proprio | 30×48 actions | wrist/rotation/hand loss | 是 |
| Robot→MANO optimizer | robot-only preprocessing | hand joints/fingertips | MANO target | SmoothL1 fingertip | 否 |
| MANO→robot MLP | robot-specific | predicted fingertips | hand actuation | supervised regression | 是 |
| Arm IK / PD | robot-specific | wrist pose / joint target | 36D control execution | 无 policy training loss | 是 |

共享参数确实让 human action loss 改变后续 robot policy initialization；不是两个不相干 heads。但因为采用 sequential pretrain→post-train，115 epochs robot-only 更新可能覆盖 human representation。论文没有 feature overlap、representation probing、forgetting或 joint-training baseline。

### 6.2 训练 schedule

| 阶段 | 数据 | Epoch | Global batch（论文） | LR | 冻结 | 算力 |
| --- | --- | ---: | ---: | ---: | --- | --- |
| Human pretraining | 500k human pairs | 20 | $16\times8\times4=512$ | $10^{-4}$ cosine | 无，full model | 论文称 32 A100 |
| Robot post-training I | 1,200 sim demos | 100 | 512 | $2\times10^{-5}$ constant | 无 | 32 A100 |
| Robot post-training II | 同上 | 15 | 512 | $2\times10^{-6}$ constant | 无 | 32 A100 |
| Retargeting MLP | robot paired fingertips/actions | 2,000 | 2,048 | $10^{-3}$ | 独立小网络 | 未报告 |

公开 launch script 默认一台、8 processes，且引用 cluster variables；run/model 名又含 `b16-4`。因此论文 4-node×8-GPU setting 与 release script 需要使用者自行恢复 scheduler/node environment，不能直接单机运行等价训练。

### 6.3 Loss

附录定义：

$$
\mathcal L = 20\mathcal L_{wrist-trans}
+5\mathcal L_{wrist-rot}
+5\mathcal L_{MANO},
$$

$$
\mathcal L_{wrist-trans}=\|T_{pred}-T_{gt}\|_2^2,
\quad
\mathcal L_{wrist-rot}=\|R_{pred}-R_{gt}\|_2^2,
\quad
\mathcal L_{MANO}=\|\Theta_{pred}-\Theta_{gt}\|_2^2.
$$

rot6D prediction 先转 rotation matrix 再算误差。权重强调 wrist translation。没有 language contrastive loss、domain alignment、action feasibility、object transition、contact、success、dynamics 或 temporal smoothness loss。release 的 robot post-training script又启用 `--loss_use_l1 True`，而 human script没有；这与附录统一写 L2 有版本差异，应复现实验时记录实际 branch/config。

## 7. Semantic–motion–action–contact 对齐

![固定图像改 instruction 后，human wrist trajectory 随语义发生变化](https://arxiv.org/html/2507.12440v3/Instruction_following_human_v2.png)

*图 6。它证明模型没有完全忽略文本，但只属选例可视化，没有 quantitative counterfactual metric。*

| 层 | Human 表示 | Robot 表示 | Contact 表示 | 对齐监督 | 执行验证 |
| --- | --- | --- | --- | --- | --- |
| Task semantic | clip-level/immediate language | task description | 无 | shared NVILA tokens | 12已知tasks；无robot language ablation |
| Subgoal | sequence隐式 + benchmark PSR labels | multi-stage task | success flag只用于评测 | 无显式 subgoal loss | PSR |
| Motion intent | future wrist/MANO chunk | future EE/MANO chunk | 无 | shared 48D regression | sim trajectory与success |
| Object transition | RGB隐式 | RGB隐式 | simulator state不输入 | 无 | only final success |
| Grasp/contact | hand pose/fingertips | Inspire hand joints | simulator physics隐式 | 无显式对齐 | success间接 |
| Force/compliance | 无 | benchmark可提供contact force但model不用 | 无 | 无 | 无 |

语言是 action prediction condition，而非 planner 输出。HOT3D 没标签时用 placeholder，说明一部分 data 只能贡献 visual/motion。即使语言改变 predicted trajectory，也没有证明文本决定 contact mode、grasp point或force；scene-to-action shortcut 仍可能存在，因为每个 task 的 object/background 和 instruction 强相关。

Human motion 更接近 dense behavior prior，而不是精确 robot trajectory：8 cm human future-wrist average error说明它预测大方向/intent 尚可，但对狭窄插入或接触是很大的空间误差。robot post-training必须把这类 prior修正到目标分布。

## 8. Ego Humanoid Manipulation Benchmark

![12 个 simulated Unitree H1 dexterous manipulation tasks](https://arxiv.org/html/2507.12440v3/task_visualizations.png)

*图 5。7 个 short-horizon atomic tasks 与 5 个 long-horizon multi-stage tasks。*

### 8.1 Platform 与 observation/action

| 项目 | 设置 |
| --- | --- |
| Simulator | NVIDIA Isaac Lab 1.2.0 + Isaac Sim 4.2.0；作者警告新版physics不同 |
| Robot | Unitree H1，双 Inspire hands；固定 base/lower body |
| Available observation | joints、EE pose、contact force、ego RGB-D、success/subtask flags |
| EgoVLA实际使用 | ego RGB、EE pose、hand actuation、task description；不使用depth/contact force |
| Hand | 每只12 DoF：6 active + 6 mimic |
| Control | arms EE control + hands PD joint control；final 36D action；30 Hz |
| Visual diversity | 5 rooms × 5 tables = 25 configurations |
| Training configs | Room 1–3 + Table 1 |
| Evaluation | seen: 3 backgrounds×9=27 rollouts/task；unseen: 22×3=66/task |
| Object OOD | rollout开始位置随机，区域相对training未见，最大约20×20 cm |

一个 method 全矩阵为 $12\times(27+66)=1,116$ rollouts，评测次数比常见真机论文充分。但它们并非 independent trained seeds；每个 scene 3次，表中没有 CI。simulation reproducibility 高，不代表现实有效性。

### 8.2 Short-horizon 完整结果

| Method | Seen SR / PSR | Unseen SR / PSR |
| --- | ---: | ---: |
| ACT specialists | 24.87 / 59.79 | 24.89 / 54.22 |
| EgoVLA-NoPretrain | 64.55 / 71.87 | 51.28 / 62.63 |
| EgoVLA 50% robot data | 48.15 / 61.73 | 未报告 |
| **EgoVLA** | **77.78 / 84.92** | **69.11 / 76.26** |

Human pretraining 相对 no-pretrain：seen +13.23 pp，unseen +17.83 pp。EgoVLA 从 seen 到 unseen 下降 8.67 pp；NoPretrain 下降 13.27 pp，而非正文概括的“23%”若按 percentage points。若按 relative drop，则分别 11.1% 与 20.6%。

unseen 分任务上，最大改善包括 Flip-Mug 4.69→30.77、Pour-Balls 46.03→83.33、Open-Laptop 48.48→83.33；Close-Drawer 已高达 86.36→98.48。人类 pretraining 对不同技能的贡献并不均匀。

### 8.3 Long-horizon 完整结果

| Method | Seen SR / PSR | Unseen SR / PSR |
| --- | ---: | ---: |
| ACT specialists | 2.22 / 26.47 | 0.61 / 23.51 |
| EgoVLA-NoPretrain | 26.67 / 54.93 | 11.21 / 36.20 |
| EgoVLA 50% robot data | 7.41 / 39.70 | 未报告 |
| **EgoVLA** | **45.93 / 80.78** | **28.79 / 69.11** |

Human pretraining使seen success +19.26 pp、unseen +17.58 pp。unseen PSR仍为69.11，明显高于final SR 28.79，说明经常完成多数subgoals后在末段失败；这也是缺 recovery、contact state 和 long-term memory 的表现。

### 8.4 三个不能回避的 ablation

1. **Human-only / no robot post-training：所有 tasks 0%**。几何 action space 并未消除 visual、kinematic 和 embodiment shift。
2. **50 robot demos/task：seen long-horizon 45.93→7.41%**。所谓“few robot demonstrations”对每任务仍是100条，减半就崩；human先验不能替代moderate robot anchoring。
3. **No human pretraining：仍有26.67%/11.21% long-horizon SR**。NVILA + multi-task robot training本身很强；human的增量应相对该baseline计算，而非相对ACT。

## 9. 数据混合、scaling 与 negative transfer

![四数据集 mixture 明显优于子集，但没有 error bars 或等 token 控制](https://arxiv.org/html/2507.12440v3/data_mixture_ablation_short.svg)

*图 7。unseen short-horizon SR 约为 full 69%、HOI4D+HOT3D+TACO 51%、HOI4D+HOT3D 49%、HOI4D 48%；多样性收益主要来自加入HoloAssist后的跃升。*

| Human mixture | Unseen short SR（图读） | PSR（图读） | 可以说明什么 |
| --- | ---: | ---: | --- |
| HOI4D | 约48% | 约58% | 单源已有迁移 |
| HOI4D + HOT3D | 约49% | 约59% | 增量很小 |
| + TACO | 约51% | 约67% | progress改善更明显 |
| + HoloAssist（full） | 约69% | 约76% | 复杂双手/场景多样性相关 |

这不是严格 scaling law：每个 mixture 的数据 token、training updates和source比例是否配平未说明，没有多个规模点、seed、CI或negative transfer table。full mixture同时增加数据量、任务范围、双手比例、视觉环境和语言质量；不能归因单一因素。论文声称HoloAssist噪声、HOT3D缺语言、TACO视觉有限仍能positive transfer，但未展示将等量高质量数据替换噪声数据的质量曲线。

![不同 object spawn 位置的 success/progress heatmap](https://arxiv.org/html/2507.12440v3/randomized_positions_split_heatmaps_separate_colorbars.png)

*图 8。short tasks 中心区域更好；long tasks 出现左右手对应高概率区。覆盖并非均匀，workspace和bimanual reachability仍限制策略。*

## 10. Task gap：Generalist 不等于 compositional generalization

EgoVLA 用一套 policy 覆盖12 tasks，并包含 Insert-And-Unload 和 Stack-Can-Into-Drawer 等多阶段任务；相较每task ACT specialist，它能共享low-level skills。这是**multi-task generalization**和已训练composition execution。

但新skill composition的严格测试要求：训练见过atomic A/B，测试只给新instruction `A then B` 且没有该composite robot demonstrations。本文每个long-horizon task都有100条专属demonstrations，所以结果不能证明zero-shot composition。没有显式skill boundary、option、latent manifold、planner、correctable latent action或language coaching。

| 泛化层级 | 是否验证 | 证据 |
| --- | --- | --- |
| 同task新初始位置 | 是 | randomized unseen spawn region |
| 同task新visual background | 是 | 22 unseen room/table combinations |
| 同图像反事实language | 仅human qualitative | changed trajectory examples |
| 同一policy多task | 是 | 12-task generalist |
| 新language paraphrase | 否 | 无系统metric |
| 新task composition | 否 | long tasks有专属robot demos |
| 新robot embodiment | 否 | 只模拟一个H1+Inspire setup |

## 11. Contact intent、grasp goal 与 Reality gap

### 11.1 Contact 表示盘点

| 变量 | Human pretrain | Robot post-train | Policy输入/输出 | 评价 |
| --- | --- | --- | --- | --- |
| Wrist trajectory | 有 | 有 | 显式 | 强 |
| MANO/fingertip geometry | 有 | 由robot拟合 | 显式 | 几何层强 |
| Object pose/goal state | upstream部分有 | simulator有 | 不输入 | 缺 |
| Contact point/normal/phase | 无 | simulator内部有 | 不输入/不监督 | 缺 |
| Force/tactile/compliance | 无 | contact force可用但不用 | 无 | 缺 |
| Collision/joint limit | 无loss | IK/physics隐式 | controller吸收 | 未量化 |
| Failure/recovery | human corpus可含纠错但未专门标 | demos全是successful | 无recovery objective | 成功偏差 |

MANO hand pose可以是 grasp goal 的一种几何proxy，却不是functional grasp specification。同一指尖位置可能对应不同接触力，human hand与Inspire link也没有同一contact surface。对于柔性物、工具、插接、滑移和高力任务，仅回归pose不足。

### 11.2 Reality gap

| 差异 | 建模方法 | 数据来源 | 是否在线 | 验证指标 | 剩余风险 |
| --- | --- | --- | --- | --- | --- |
| Robot kinematics | frame transform + arm IK | sim robot model | 是 | sim SR | singularity、joint limit、model error |
| Hand mapping | learned fingertip→actuation MLP | sim demos | 是 | 0.05 mm fitting + replay | real backlash、tendon/compliance、contact force |
| Dynamics/contact | Isaac physics + PD | simulator | 是 | task SR/PSR | friction/contact参数对现实敏感 |
| Visual gap | diverse rendered backgrounds | 25 configs | 是 | seen/unseen sim | render→real、exposure、blur、camera noise |
| Latency/control | 30 Hz chunk + smoothing 0.8 | software | 是 | no separate metric | real inference latency/jitter未测 |
| Balance/locomotion | fixed base | 无 | 否 | 无 | humanoid fall/CoM完全未处理 |
| Sim2real | 明确不作为benchmark目标 | 无 real data | 否 | 无 | **未解决** |

没有 joint-wise neural dynamics、HumanoidDM、system identification、domain randomization、actuator model或real post-training。Reality gap不是由EgoVLA解决，而是被simulator与low-level controllers回避。

## 12. 快慢系统接口

| 层 | 输入 | 输出 | 频率 | 训练数据 | Gap责任 |
| --- | --- | --- | ---: | --- | --- |
| System2 | immediate language + visual history | 隐式task-conditioned latent | 30 Hz policy同层 | VLM + human/robot labels | 部分Task/Semantic；无独立planner |
| System1 | multimodal latent + current wrist/MANO | 30×48 wrist/MANO chunk | 30 Hz replanning/execution | 500k human pairs + 1,200 sim demos | Embodiment motion + multi-task |
| Projection | geometric chunk | arm IK + hand joints | 30 Hz | robot paired mapping | Kinematic embodiment |
| System0 | current sim joints + target | 36D EE/PD commands | 30 Hz policy/control | simulator/controller | contact/reality大多被吸收 |
| Drive | Isaac H1 actuators | simulated forces/motion | physics step | simulator parameters | hardware无证据 |

action chunk用smoothing coefficient 0.8融合预测，增加连续性；论文未说明接触切换时是否adaptive、上一chunk与新chunk具体权重、failure trigger或closed-loop subgoal replanning。慢系统给快系统的是trajectory，不是contact intent、force target或constraint。

## 13. Claim—Evidence 因果审计

| Claim | 所需对照 | 论文证据 | 是否充分 | 替代解释 |
| --- | --- | --- | --- | --- |
| Human pretraining改善robot policy | 同backbone/robot data，去human stage | EgoVLA vs NoPretrain | 较充分 | 额外training compute而非human semantics/action |
| 提升来自human action supervision | shuffled/no action或RGB-onlyhuman | 无 | 不充分 | visual continued pretraining即可 |
| Language理解semantic intent | fixed image改instruction、quant metric/no-language | 只有选例trajectory | 弱 | output对关键词shortcut，无execution验证 |
| Unified MANO支持直接跨本体 | human-only zero-shot、多robot | zero-shot 0%，单H1 | 反证过强claim | robot post-training才完成alignment |
| Human data减少robot需求 | robot-data scaling curve | 100→50两点，50崩溃 | 不支持强scaling claim | 100/task仍是必要门槛 |
| 多源越多越好 | 等token/steps、seeds、quality levels | 4个nested mixtures单点图 | 趋势证据 | full set更大且Holo任务更接近benchmark |
| Long-horizon composition | train atomic/test unseen combo | 每个long task有100 demos | 不充分 | multi-task imitation/retrieval |
| 可用于humanoid | 真机或sim-to-real | 仿真H1 | 仅simulation | controller/physics/vision gap |

最关键的补充实验是：固定总human frames与optimizer updates，比较 (a) no extra pretrain、(b) human RGB+language但action shuffled、(c) human wrist only、(d) human wrist+MANO、(e) full human；再把同一robot policy部署到至少两个camera/hand morphology和真机。这样才能分离视觉、semantic、motion和geometric action transfer。

## 14. 隐私、偏差与治理

四源都是受控采集而非web scraping，但仍有治理问题：

- HOI4D只有9participants，尽管rooms/objects多，手型、行为策略和文化多样性有限；CC BY-NC限制商业复用。
- HoloAssist有350 instructor-performer pairs、语音对话、gaze和真实工作环境；EgoVLA需确认只处理必要RGB/pose/label，避免带入声纹、屏幕和身份信息。
- HOT3D需要注册并接受分资产协议；MANO model也需单独许可，checkpoint是否包含受限衍生表示应按协议审查。
- TACO的website license不能自动当作video、mesh、motion和MANO文件的统一许可。
- 多数训练是成功/完整动作，缺失败、危险行为、恢复和多策略标签；VLA可能学到单一熟练者分布。
- train/eval切分是否按participant、location和object instance去泄漏，EgoVLA没有统一报告；HOI4D qualitative eval未见片段不等于identity/scene完全独立。
- 语言标签粒度差异大，HOT3D placeholder会形成source/domain shortcut。

商业或真机项目不能只引用EgoVLA代码MIT；必须建立per-sample provenance、upstream license matrix、participant consent范围与derived checkpoint合规审查。

## 15. 开源审计与分层复现

### Level 1：Checkpoint inference——较可行

下载base/human/final checkpoints、retarget weights与benchmark assets，锁定Isaac Lab 1.2.0/Isaac Sim 4.2.0，运行单task单visual config。成功标准是复现30 Hz输出、相同action schema和已发布checkpoint的近似SR。

### Level 2：Robot post-training——中高成本

下载518.9 GB demos，按12 task处理图像/hand labels；至少8–32 A100级别环境，恢复DeepSpeed、cluster vars和两阶段LR。成功标准先是NoPretrain/Full在几个short task的ranking，而非一次追求全表。

### Level 3：Human pretraining——高成本

分别接受并下载四源、MANO；重建camera-frame wrists、MANO、language和500k sample mixture。最大风险是版本/路径、Holo subsampling、upstream schema与license。20 epochs×32 A100是论文级成本。

### Level 4：真机——尚无官方路线

需H1/EX002等平台的head camera calibration、whole-body IK/WBC、joint-limit/collision filter、hand retargeter、latency measurement、force/tactile safety和real demos。先做fixed-base push/pick，不直接尝试bimanual long-horizon。成功标准应包括task SR、intervention、collision、force、tracking RMSE和连续运行安全，而不是只看视频。

## 16. 面向人形、EX002 与灵巧手的落地建议

| 目标 | 可复用 | 必须重做 | 最小anchor data | 建议试点 |
| --- | --- | --- | --- | --- |
| H1同型 | checkpoint、48D schema、MANO/IK、benchmark | camera calibration、real controller、安全 | 每task至少100，论文证据不支持50足够 | fixed-base push/stack can |
| EX002/不同humanoid | VLM和human pretrained action prior | FK/IK、head-camera frame、arm workspace、whole-body controller | 先200–500多场景robot demos + retarget dataset | 单臂pick-place，再双臂handover |
| 不同dexterous hand | MANO latent、human corpus | fingertip set、actuation MLP、contact-aware mapping | 覆盖全joint/workspace的paired poses + grasp data | known rigid object grasp |
| Parallel gripper | wrist motion/VLM | MANO15不合适，改gripper aperture/contact goal head | task-specific robot demos | push/pick，避免复杂in-hand |
| Loco-manipulation | high-levelvisual/language prior | floating base、head motion、WBC、balance、foothold/contact | full-bodyteleop + real locomotion | 暂不从本paper直接试点 |

Human data最适合进入 visual-semantic encoder 和 wrist/hand action prior；不应直接成为System0 motor target。对接触任务，应把MANO补成object-centric grasp/contact goal：目标物、接触区域/法向、pregrasp、desired wrench/compliance，再由平台特定controller实现。

建议试点门槛：held-out背景/物体位置 success ≥70%，三seed置信区间；IK failure/collision <1%；real intervention <5%；force不越安全阈值；robot-only与human-pretrained等compute对照至少+10 pp。未达到时优先排查camera/action alignment，而非继续堆human hours。

## 17. 十个组会质疑

1. 为什么把“human VLA inherently already a robot policy”写得如此强，而human-only在所有tasks都是0%？
2. 50万human pairs增加的training compute是否与NoPretrain等价配平？
3. 去掉MANO/wrist action、只用同样RGB+language continued pretraining，会保留多少收益？
4. HoloAssist加入带来最大跃升，是数据规模、bimanual content、task similarity还是visual diversity？
5. HOT3D placeholder language是否让model学source token shortcut？
6. 8 cm human wrist error如何支持需要厘米/毫米精度的插入与grasp？robot stage修正了多少？
7. 0.05 mm fingertip mapping error在哪个split、哪个range测得，是否包含OOD hand configs？
8. long-horizon tasks都有专属100 demos，为什么能称skill composition，而非multi-task imitation？
9. 论文32 A100与release one-node script、Holo sub5/1/10、L2/L1差异对应哪个最终checkpoint？
10. 没有真机、force、tactile、balance与collision metrics，humanoid deployment claim的边界应在哪里？

## 18. 能力评分与最终判断

| 能力 | 1–5 | 依据 |
| --- | ---: | --- |
| 视觉泛化 | 4 | 22 unseen sim backgrounds，human pretrain +17.83 pp short SR |
| 语义迁移 | 2.5 | 有反事实trajectory选例，无robot定量language ablation |
| 运动迁移 | 4 | wrist+MANO dense action pretraining，直接NoPretrain增益 |
| 动作可执行性 | 2.5 | IK/MLP在sim有效；human-only 0%，无真机 |
| 接触迁移 | 1 | 无contact/force/tactile supervision |
| 跨本体 | 2 | human→单一sim H1，必须post-train；无多robot |
| 因果证据 | 3 | NoPretrain、human-only、50%robot、mixture；缺等compute/action-shuffle |
| 真机可信度 | 1 | 零真机，benchmark明确不做sim-to-real |
| 开源复现 | 4 | code、benchmark、518.9GB data、三checkpoint公开；环境/路径/版本成本高 |

### 最终判断

EgoVLA 把 human Ego 中的**语言条件 wrist/MANO future motion**转换为 VLA 的**跨场景 action prior**，通过 camera-frame projection、MANO shared space和robot post-training部分缩小 Embodiment gap；通过multi-source VLM pretraining和single multitask policy部分缩小已知任务的visual/semantic Task gap；它没有使用dynamics model或real post-training应对 Reality gap，所有现实执行问题由Isaac physics、IK和PD回避。上限取决于target robot demonstrations、retargeting/contact fidelity、真实camera/control alignment和能否把pose prior扩展为object/contact-aware goal。

最准确的总结是：

> 该方法在 **sequential human pretraining** 阶段以 **camera-frame wrist pose + MANO action chunk** 形式引入 Human Ego 数据，主要迁移的是 **视觉—语言条件的手运动先验**；性能提升由 **相同robot post-training下的NoPretrain对照** 支持，但尚未证明 **human-only可执行迁移、新技能组合、contact transfer或sim-to-real**，其部署依赖 **每任务robot demonstrations、IK、hand retargeting和低层PD**。

它值得精读和simulation复现，也适合作为EX002/H1的VLM/action-prior起点；在完成等compute因果消融、contact-aware System0与真实小规模验证前，不应把它作为通用真机humanoid policy的直接方案。
