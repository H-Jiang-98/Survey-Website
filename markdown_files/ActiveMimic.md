---
title: "ActiveMimic: Egocentric Video Pretraining with Active Perception"
method_name: "ActiveMimic"
authors: [Xingyao Lin, Guojin Zhong, Tianyi Lu, Ziyi Ye, Yichen Zhu, Zuxuan Wu, Yu-Gang Jiang]
year: 2026
venue: arXiv
tags: [egocentric-video, active-perception, pretraining, vision-language-action, camera-motion, pseudo-action, flow-matching, humanoid-manipulation]
image_source: online
---

# ActiveMimic：把第一视角相机运动从“噪声”变成pretraining action

> 本笔记基于 [arXiv:2606.06194v1](https://arxiv.org/abs/2606.06194)、[HTML 全文](https://arxiv.org/html/2606.06194)与论文所列[项目页](https://activemimic.github.io/)核验；v1发布于2026-06-04，未标注会议接收。未检索到官方代码、处理后数据或checkpoint。

## 阅读结论先行

ActiveMimic的核心判断很对：egocentric video中的camera motion不仅是需要消去的head shake，也包含“蹲下看低处、抬头看货架、侧转搜索”的主动感知行为。论文用VGGT从单目RGB恢复camera trajectory、UniDepth补metric scale、SAM-3D-Body估左右wrist，再把三者放到chunk首帧camera frame，组成camera 9D+双wrist 18D的27D action，以一个flow-matching action expert联合预训练。

这一步的角色是**representation pretraining**，不是把pseudo-action直接下发机器人。之后仍需每个目标task的AGIBOT G1 teleoperation：Restocking/Reaching/Finding/Pouring分别270/30/60/90 demos，并将整模型约5 epochs robot-specific finetune。Robot数据提供真实head/waist/arm/gripper action与动力学anchor；Ego4D则教“视觉变化与相机—手运动如何共同演化”。

主结果很强：四任务端到端SR为90.1/88.9/91.7/93.3%，均高于wrist-only和SFT-only，也与robot-data-pretrained $\pi_0$相当或略好。尤其camera-supervision因果对照比较干净：ActiveMimic和wrist-only使用相同Ego4D clips、architecture和500k training steps，唯一主要差异是27D vs18D target。这支持相机运动label提供额外信息，不只是“多看了human video”。

Restocking诊断更直接：三种variant几乎都能pickup，但placement阶段full pretrained为24/27，SFT-only仅6/27；把full model的head image置零后仅1/27。它说明预训练后的policy确实依赖head visual loop来抬头找shelf，而不是robot fine-tuning单独学会。不过zeroing会制造训练外black image，不能等同“没有head camera”的公平重训；性能崩溃同时测了信息缺失和OOD corruption。

Pseudo-label fidelity必须谨慎。HOT3D的所谓strict tier允许position error **≤0.8 m**、rot6d L2≤0.6，head recovery 78.82%，左右wrist只有65.93/61.72%。对机器人control，80 cm绝非strict；即便loose tier>85%，论文没有给median/percentile cm/degree error。它只证明多数labels保留粗趋势，足够做大模型pretraining signal，不能证明准确metric action或任意视频可可靠处理。

Camera motion也不等于eye gaze或信息最优viewpoint。Ego4D包含头、胸、眼镜等mount，trajectory混合主动搜索、身体locomotion、VOR、无关摇晃和camera mounting offset。论文声称mount-independent并不严谨：同一头动在chest camera与glasses camera产生不同SE(3)；filter甚至明确排除纯look/walk片段，只保留hand-object manipulation，可能丢掉最纯粹的visual search supervision。

“in-the-wild scale”当前只有Hands & Objects中过滤后的2561 episodes、约10 h、10 fps。3.6B模型训练500k steps、batch64，相当于对约9.2M frames反复采样很多次；没有data scaling curve、重复度分析或VGGT/UniDepth/SAM处理成本。10 h能有效并不等同procedure已经在Ego4D 3000 h上可行。

Robustness结果有趣但因果不能只归active perception：闪烁灯下79.0%，unseen包装yogurt下72.2%，都优于wrist-only；camera label可能让representation更稳定，也可能只是多9维prediction形成更强self-supervised objective。缺少random camera target、camera-only、shuffled camera-wrist pairing和equal-dimensional auxiliary task，尚不能证明“head-hand causal coupling”是唯一机制。

### 一句话总结

ActiveMimic有力证明：从单目ego video恢复的粗camera trajectory，即使远未达到控制精度，也能作为比wrist-only更好的VLA预训练监督，并通过robot fine-tuning转成head–hand active perception；但它不是human-only robot learning，10 h规模、单平台、宽松pose指标和无artifact release都限制了“任意in-the-wild视频可扩展”的结论。

![ActiveMimic 从单目ego RGB恢复camera/wrist并构造27D预训练action](https://arxiv.org/html/2606.06194v1/framework.png)

*图 1。官方pipeline。Human stage与robot stage共享网络架构但action空间不同；迁移发生在参数初始化，不是直接action execution。*

## 1. Camera–wrist coupling如何解开

VGGT camera pose $T^{cam_k}_{cam_1}$锚定episode首帧，SAM-3D-Body wrist pose $T^{wrist_k}_{cam_k}$位于当前camera frame。对chunk起点$i$，先重心化camera：

$$
T^{cam_{i+\tau}}_{cam_i}=(T^{cam_i}_{cam_1})^{-1}T^{cam_{i+\tau}}_{cam_1},
$$

再复合wrist：

$$
T^{wrist_{i+\tau}}_{cam_i}=T^{cam_{i+\tau}}_{cam_i}T^{wrist_{i+\tau}}_{cam_{i+\tau}}.
$$

这并非把camera和wrist统计独立，而是消除坐标混淆、让两者在同一reference表达，保留真实head–hand coordination。每个SE(3)用translation+rotation前两列共9D，三者拼成：

$$
a_{i,\tau}=[p^{cam},r^{cam}_{6D},p^{L},r^L_{6D},p^R,r^R_{6D}]\in\mathbb R^{27}.
$$

Metric scale以每帧UniDepth/VGGT valid-pixel depth ratio的median，再跨帧取median。若人物/物体动态区域占比大，两种单目depth的共同bias会变成episode-wide trajectory scale error；论文未用已知长度或HOT3D translation error分解验证scale。

## 2. Pretraining与robot adaptation

3B VLM prefix编码image+language，0.6B action expert以conditional flow matching预测50-step chunk：

$$
\mathcal L=\mathbb E\|v_t(a_t,o)-(\epsilon-a)\|_2^2,
\qquad a_t=t\epsilon+(1-t)a.
$$

先冻结VLM warm-up 2k steps，再全量500k steps；robot task阶段全参数finetune约5 epochs。Pretraining和fine-tuning target维度/语义改变，说明action expert学习的并非固定27D controller，而是可被新embodiment重塑的motion-conditioned representation。

~~~mermaid
flowchart LR
    E["Ego4D RGB clips"] --> S["Qwen VLM segmentation + LLM filtering"]
    S --> C["VGGT camera + UniDepth scale"]
    S --> W["SAM-3D-Body wrists"]
    C --> U["Chunk-relative 27D action"]
    W --> U
    U --> P["3.6B flow-matching pretraining"]
    R["30-270 robot demos/task"] --> F["Full robot-specific finetuning"]
    P --> F
    F --> G["AGIBOT head/waist/two-arm policy"]
~~~

## 3. 结果与样本量

| Task | Robot demos | Trials | ActiveMimic SR | $\pi_0$ SR |
| --- | ---: | ---: | ---: | ---: |
| Restocking | 270 | 81 | 90.1 | 86.4 |
| Reaching | 30 | 18 | 88.9 | 不高于ActiveMimic |
| Finding | 60 | 36 | 91.7 | 86.1 |
| Pouring | 90 | 45 | 93.3 | 不高于ActiveMimic |

18-trial Reaching的一个trial就是5.56 pp，45-trial Pouring为2.22 pp；论文无seed/CI。与$\pi_0$比较还不等compute/data规模：$\pi_0$使用大规模robot-pretrained public checkpoint，ActiveMimic在同robot task finetune但前置500k human steps；可以说downstream表现匹配，不能说10 h human信息量等价robot corpus。

## 4. Representational transfer证据有多强

作者比较head-only与full-view inference时action-expert每层top-$K\%$ activated units交集；ActiveMimic在early/mid layers高于wrist-only，且$K=5,10,15,20$趋势一致。这是一个相关性probe，不是“human representation直接迁入robot”的唯一解释：

- 两模型训练target维度不同，activation scale/sparsity可能系统性改变。
- Full-view与head-view是robot images，两者都不是Ego4D source distribution。
- Unit overlap不度量feature geometry、causal necessity或task information。
- 应增加CKA/probing、source-vs-target feature matching与camera-label shuffle control。

因此behavior ablation比activation overlap更强：camera-supervised model在相同robot data上确实更好；内部机制仍是提示性证据。

## 5. Artifact audit 与最终判断

| Artifact | 状态 | 复现影响 |
| --- | --- | --- |
| arXiv全文/appendix | 公开，pipeline与prompts较完整 | 可理解filter和training |
| Project page | 论文列出；当前无法可靠打开 | 无法核验额外assets |
| Code | 未发现 | VGGT/scale/chunk/action转换不可直接复跑 |
| 2561-episode processed corpus | 未发现 | Ego4D source可申请，但exact subset/labels缺失 |
| Pretrained/robot checkpoints | 未发现 | 500k-step结果无法独立验证 |
| Robot eval scripts/data | 未发现 | 成功判定与baselines不可复核 |

下一步应公开UID/segments/pseudo-labels和checkpoint；用cm/degree pose distributions替代0.8 m “strict”；做camera-label shuffle/random/camera-only对照；固定总pretraining compute做10/100/1000 h scaling；换第二种robot/camera mobility验证transfer；并将head command、waist command与camera pose显式映射，报告collision、joint-limit与perception latency。这样才能判断camera motion监督究竟是通用active-perception prior，还是对这台AGIBOT/四个任务特别有效的auxiliary pretext task。
