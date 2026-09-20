---
title: "HALOMI: Learning Humanoid Loco-Manipulation with Active Perception from Human Demonstrations"
method_name: "HALOMI"
authors: [Zehui Zhao, Yuxuan Zhao, Gaojing Zhang, Chenxi Liu, Maolin Zheng, Wenzhao Lian]
year: 2026
venue: arXiv
tags: [humanoid, loco-manipulation, active-perception, human-demonstration, vision-language-action, whole-body-control, controller-aware-adaptation, robot-free-data]
image_source: online
---

# HALOMI：让human demonstration驱动humanoid的移动操作与主动视觉

> 本笔记基于 [arXiv:2606.18772v1](https://arxiv.org/abs/2606.18772) 与 [HTML 全文](https://arxiv.org/html/2606.18772)核验；v1发布于2026-06-17，当前仅为arXiv预印本。论文列有[项目页](https://halomi-humanoid.github.io/)，但本次审计未发现可核验的官方code、data或checkpoint仓库。

## 阅读结论先行

HALOMI最有价值的设计不是单一VLA模型，而是一条**稀疏人类轨迹—主动相机—全身运动先验**的分层接口：人只示范头和双手，$pi_{0.5}$预测相对head/hand trajectory；低层controller从BFM-Zero的whole-body latent manifold里补全腰、腿和平衡，并用解析IK控制三自由度颈部。这样既绕开人—机器人腿部运动对应，也让相机真正成为可执行action的一部分。

“robot-free demonstration”需要精确理解。三个量化任务的task-specific policy训练确实只用95–102条human demos，没有robot teleoperation；但系统并非robot-learning-free：目标Unitree G1要加装与采集端相同的Pika gripper、camera与定制3-DoF neck，低层controller依赖6000+条loco-manipulation motion和simulation RL/DAgger训练，且每条human reference还要在sim中做controller-aware adaptation。因此更准确的说法是**无task-specific robot demonstration的高层模仿**。

硬件co-design是成功关键而非实现细节。示范者双手各持一个Pika Sense gripper，头戴RealSense D435i与Vive Tracker，外部Lighthouse以毫米级6-DoF跟踪头和双手；机器人复刻gripper与camera placement。它把gripper width、视角与末端几何的embodiment gap直接用硬件消掉，却也意味着普通RGB ego video、任意手型或无外部tracker的演示不能直接使用。

三项量化任务共60次real-world trials，成功率为90/85/80%，即18/17/16次成功；每5个百分点仅对应一次trial，且没有多seed或置信区间。Tossing与Squat-and-Grasp只做qualitative展示，因此论文支持的是“三个任务的定量可行性加两个能力演示”，不是五任务上都得到可靠统计结论。

Active neck的ablation很大：Bag 90→30%、Towel 80→10%、Bread 85→20%。这证明当前policy强依赖训练时的head–hand–image闭环，但还不能证明可动颈部在所有任务中具有不可替代的信息价值。尤其Bread被作者承认静态视角理论上足够；测试时冻结训练中会动的neck同时造成camera distribution shift与action mismatch。公平对照应重新采集或重训fixed-camera policy，并与wrist/external camera比较。

Ego-view alignment把Bag从10%提升到90%，说明human/robot viewpoint mismatch确实致命；但该模块同时改变depth、reprojection、inpainting与图像统计，只在一个任务上做总开关，没有component ablation。结果能证明“需要某种视角对齐”，不能定位是哪一步贡献了80 pp。

Controller-aware reference adaptation平均只降低6.725%的tracking error，却把Bread/Towel各提高10/5 pp。它不是通用data cleaning，而是在特定simulator、robot与controller闭环下学习B-spline translation residual，相当于为现有controller预补偿逆动力学偏差。controller升级或sim-to-real error改变后，处理过的label可能失效；论文也未报告CEM并行量、迭代数与全数据处理成本。

最能揭示边界的是generalization：未见towel和cabinet位置各60%，Bread整体平移且相对布局不变时6/10，但改变bread-to-plate相对关系后0/10。HALOMI学到的是可重放、能维持平衡的视觉条件trajectory，而非显式object-centric goal、关系推理或在线replanning。若目标布局变化，relative action representation本身并不会自动带来关系泛化。

### 一句话总结

HALOMI以稀疏head/hand interface和强whole-body prior，把约100条human-only task demos可靠落到G1的loco-manipulation；它令人信服地展示了无robot task demos的系统路径，但效果依赖embodiment-matched采集硬件、simulation-trained controller、离线reference adaptation与当前场景关系，尚不是从任意ego video直接学会通用humanoid技能。

![HALOMI的分层loco-manipulation controller与head-hand target interface](https://arxiv.org/html/2606.18772v1/figures/loco.png)

*图 1。官方controller结构。VLA只给head/hand sparse targets；稳定全身动作主要由BFM-Zero behavior manifold、teacher–student controller与neck IK补全。*

## 1. 从human rig到robot action space

采集系统以30 Hz同步三路RGB、头和双手6-DoF pose及gripper width。对未来第$k$步，VLA使用相对position与rotation：

$$
p_t^k=p_{t+k}-p_t,
\qquad
R_t^k=R_t^{-1}R_{t+k}.
$$

这消除了episode的absolute world origin，执行时再锚定当前robot state并以50 Hz流式跟踪。需要注意：position写成world-frame相减，而rotation是local relative rotation；它并非严格统一的$SE(3)$ body-frame增量。论文没有分析这一混合表达对大幅转身或坐标漂移的影响。

视觉侧沿用monocular depth将human image reproject到G1 camera viewpoint，再对空洞inpaint。该步骤让训练图像更接近robot observation，但估计depth、动态手臂遮挡与生成式补洞可能改变物体边界；缺少real target-view ground truth或pixel/depth误差评估。

~~~mermaid
flowchart LR
    H["Human: helmet camera + Vive head + two Pika grippers"] --> D["30 Hz RGB / head-hand pose / gripper width"]
    D --> V["Depth reprojection + inpainting to G1 ego view"]
    D --> A["Controller-aware B-spline translation adaptation"]
    V --> P["Fine-tuned pi0.5 VLA"]
    A --> P
    P --> T["Relative head + two-hand targets"]
    T --> N["3-DoF neck analytic IK"]
    T --> W["BFM-Zero latent whole-body controller"]
    N --> G["Unitree G1 loco-manipulation"]
    W --> G
~~~

## 2. Low-level controller承担了什么

Teacher在simulation观察未来whole-body reference；student经DAgger只看10步proprioceptive history和head/hand target error。策略不直接输出关节动作，而是在冻结的BFM-Zero decoder上预测球面128D latent：

$$
z_t\in\mathbb S^{127},
\qquad
a_t=D_{\mathrm{BFM}}(z_t,o_t).
$$

这一manifold prior把输出限制在6000+ motion sequences覆盖的行为邻域，使OOD command下比raw-action policy更不易失衡。但“落在learned manifold”不是形式化安全保证：物体碰撞、joint limit、接触力与快速投掷的风险没有定量评估；论文也没有和标准WBC/MPC做等接口、等command的充分benchmark。

## 3. Controller-aware adaptation其实是闭环label校正

对原始reference $x_{\mathrm{raw}}(t)$，只给position添加B-spline residual，orientation保持不变：

$$
x_{\mathrm{adapt}}(t)=x_{\mathrm{raw}}(t)+B(t)c.
$$

CEM在并行simulation中搜索control points $c$，最小化rollout后的tracking error与smoothness penalty，并且仅在改善时接受。其逻辑不是让controller更强，而是找一条**经当前controller执行后更接近原human trajectory**的命令轨迹。

| Task | Raw error (m) | Adapted error (m) | Relative reduction | 下游变化 |
| --- | ---: | ---: | ---: | ---: |
| Pick Bread & Place | 0.0436 | 0.0409 | 6.193% | 75→85% |
| Towel to Basket | 0.0568 | 0.0523 | 7.923% | 75→80% |
| Squat-and-Grasp | 0.1238 | 0.1163 | 6.058% | 未量化 |

Squat适配后仍有11.6 cm误差，说明“自动适配”没有消除大幅全身动作的mismatch。Bread提升两次成功、Towel提升一次成功，也可能落在小样本波动内；应给paired trajectories、多个训练seed与无smoothness/无global-local decomposition对照。

## 4. Real-world结果该怎样读

| Task | Human demos | Trials | Success | 成功次数 |
| --- | ---: | ---: | ---: | ---: |
| Bag Transfer | 102 | 20 | 90% | 18/20 |
| Pick Bread & Place | 95 | 20 | 85% | 17/20 |
| Transfer Towel to Basket | 96 | 20 | 80% | 16/20 |
| 平均 | 293 | 60 | 85% | 51/60 |

论文的component evidence按强弱可分为：active-neck与view-alignment带来大幅behavior difference，最强；reference adaptation的tracking improvement明确、success effect较小；BFM latent的稳定性主要靠qualitative rollout；“active perception机制”则仍混有test-time ablation产生的OOD影响。

OOD结果进一步区分appearance/location与relation：换towel外观和cabinet位置仍有60%，表示视觉表征有一定容忍度；保持相对布局的整体平移为6/10，改变目标相对布局为0/10，直接暴露了trajectory cloning缺少object-goal compositionality。后续应输入object state或goal relation，并用closed-loop visual replanning而非单纯chunk跟踪。

## 5. Artifact audit 与复现判断

| Artifact | 状态 | 复现影响 |
| --- | --- | --- |
| arXiv HTML/PDF | 公开 | 可核验架构、三任务表格与主要ablation |
| Project page | 论文列出；本次无法可靠访问其内容 | 可能有视频，但不能据此确认release |
| Code / configs | 未发现官方公开仓库 | VLA接口、view warp、CEM与controller训练无法复跑 |
| Human demonstrations | 未发现 | exact trajectories、同步与失败筛选不可审计 |
| BFM/controller checkpoint | 未发现HALOMI专用release | low-level结果无法独立部署 |
| VLA checkpoint / eval protocol | 未发现 | 60次trial的复现与success判定不可核验 |

当前论文足以支持一个强system proof-of-concept，但不够支持可复现benchmark。最优先的公开项应是293条raw/processed demos及同步标定、view-alignment代码、每条trajectory的adaptation前后误差、CEM算力与参数、VLA/controller checkpoints，以及逐trial视频和失败taxonomy。实验上还需要fixed-neck重新训练、不同camera配置、标准WBC baseline、第二种humanoid embodiment与layout-relation benchmark，才能把“有效系统集成”提升为可迁移的human-to-humanoid learning结论。
