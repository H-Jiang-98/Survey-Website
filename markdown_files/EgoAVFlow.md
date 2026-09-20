---
title: "EgoAVFlow: Robot Policy Learning with Active Vision from Human Egocentric Videos via 3D Flow"
method_name: "EgoAVFlow"
authors: [Daesol Cho, Youngseok Jang, Danfei Xu, Sehoon Ha]
year: 2026
venue: arXiv
tags: [egocentric-video, active-vision, 3d-flow, diffusion-policy, reward-guided-denoising, zero-robot-data, visibility-planning, cross-embodiment]
image_source: online
---

# EgoAVFlow：不模仿人头，而在执行时为未来动作重新选择视角

> 本笔记基于 [arXiv:2602.22461v1](https://arxiv.org/abs/2602.22461)、[HTML 全文](https://arxiv.org/html/2602.22461)与[项目页](https://dscho1234.github.io/egoavflow/)精读核验；v1发布于2026-02-25，未标注会议接收。项目页只有论文、视频和图片，未发现官方代码、数据或checkpoint。

## 阅读结论先行

EgoAVFlow对active vision提出了比“复制human head pose”更成熟的分解：human viewpoint trajectory只作为可行动作的diffusion prior；robot执行时先预测未来gripper action和object/scene 3D flow，再对多个camera-trajectory候选做mesh raycast与FoV判断，以reward-maximizing denoising偏向未来可见性更高的视角。这样camera能偏离人头轨迹，同时不完全离开human motion distribution。

共享3D flow确实把三件事连在一起：CoTracker3像素轨迹+RGB-D变成scene point motion；HaMeR hand pose映射成parallel gripper 10D action；future-flow model预测动作引起的物体轨迹；view policy则用这些future points做visibility query。相比纯RGB/2D flow，它天然可以把同一3D point从新camera投影，也能检查robot self-occlusion。

四个真实任务、各150条task-specific human RGB-D demonstrations和25 trials上，EgoAVFlow为20/18/17/18 successes，即80/72/68/72%。同一visibility view policy下，最强baseline EgoZero为40/28/28/36%，因此完整方法提升1.8–2.5×。Human Viewpoint Imitation仅20/40/28/20%，说明在这台camera arm的workspace与输入distribution下，直接复制人头并不可靠。

不过成功定义把“object manipulation完成”与“object全程可见”做AND；因此提高visibility会直接提高被计为成功的概率，即使manipulation没变。论文另报告 $R_{vis}$并做failure分类，但没有单独列task completion ignoring visibility、纯tracking dropout率或相同camera trajectory下各manipulation policy的结果。Representation对照虽共享view module，仍可能因tracker失效同时影响控制和metric。

“zero robot demonstrations”成立，却不是无robot engineering或in-the-wild video：每任务有150条头戴D435 RGB-D主动示范；ChArUco board定义所有episode统一world frame；预先人工标注6个object/goal query points；DROID-SLAM估camera pose；部署时Nvblox重建环境mesh、IK生成future robot mesh。Manipulation由Trossen WidowX，active camera由另一条Unitree Z1+D435机械臂实现，两者4 Hz同步控制。这是高度结构化human-only supervision，不是被动互联网RGB。

最关键限制是query point必须在初始帧可见，论文也明确承认不做search或point-of-interest discovery。若CoTracker丢点或future flow预测朝错方向，camera会被reward主动引向错误视点。透明/反光、depth holes、deformable surface、快速遮挡和新query identity均未系统评估。

“view-invariant”应改成viewpoint-robust under tested trajectories。3D coordinates仍依赖depth、SLAM、ChArUco标定和tracker correspondence；active views越偏离demonstration，历史flow distribution shift越大。模型表现较好说明这条representation有韧性，但没有camera-pose幅度分层、marker perturbation或depth/SLAM noise sweep来证明invariance。

View reward安全性也有限：$R_{safe}$只惩罚camera与predicted end-effector靠近，environment mesh用于line-of-sight却未明确作为camera collision constraint；close reward反而鼓励靠近query。一个独立6-DoF camera arm在真实空间运行，需要joint limits、self/environment collision、velocity/acceleration和human safety硬约束，论文只给软距离项与IK，未报告collision或emergency stop。

### 一句话总结

EgoAVFlow有力证明了“human camera motion作prior，test-time geometry reward作纠偏”优于盲目模仿人头，且3D flow在主动变视角时比RGB/2D表示稳；但它依赖首帧可见的人工query、RGB-D/marker/SLAM/mesh和专用camera arm，尚不能搜索未知目标，也未把visibility收益与manipulation收益完全解耦。

![EgoAVFlow 三个diffusion模型与visibility-aware camera optimization](https://dscho1234.github.io/egoavflow/static/images/method-v5.png)

*图 1。官方架构。Robot policy、future-flow model与view policy串联，view diffusion在test time由几何reward重采样。*

## 1. 从human RGB-D到shared 3D flow

Human wrist keypoints被映射为：

$$
p_t=[x_t,\,r_t^{6D},\,g_t]\in\mathbb R^{10},
\qquad a_t=p_{t+1}.
$$

CoTracker3给 $N$个像素轨迹和valid indicator，depth unprojection得到：

$$
F_t=\{[X_{t,i},Y_{t,i},Z_{t,i},m_{t,i}]\}_{i=1}^{N}.
$$

Camera motion由DROID-SLAM恢复，所有action、flow、camera pose再转进ChArUco marker frame。这个步骤消除每episode任意SLAM origin，却让marker可见性/标定成为额外实验先验。

三个DiT diffusion统一prediction horizon $T=24$，每$H=12$步replan。$\pi_r$由flow+proprio history预测gripper；$f$再条件于action+camera history预测future flow；$\pi_v$输出camera SE(3) chunk。

## 2. Visibility reward到底优化什么

Future query $q_{t,i}$与camera center组成ray，对environment mesh $\mathcal M^e$和预测robot mesh $\mathcal M_t^r$做intersection，同时检查投影是否在FoV：

$$
R_{vis}=\frac{1}{TN}\sum_{t,i}
\mathbb I[\ell_{t,i}\cap(\mathcal M^e\cup\mathcal M_t^r)=\emptyset]
\,\mathbb I[\Pi_t(q_{t,i})\in\mathrm{FoV}].
$$

再加入camera-query proximity、对2 cm/0.05 rad perturbation的worst-case margin，以及camera–EEF 0.1 m尺度的soft safety。总reward非可微，因此每个denoising step从prior采$M$个候选，以：

$$
w_i\propto\exp(R(\hat x_0(x_{k-1}^{(i)}))/\alpha)
$$

做categorical resampling。$\alpha$控制human prior与visibility optimization的权衡；论文未给reward-weight、$M$、$\alpha$敏感性或real-time inference latency，复现和部署成本难评估。

~~~mermaid
flowchart LR
    H["Human RGB-D + marker"] --> F["CoTracker3 + depth 3D flows"]
    H --> A["HaMeR to gripper action"]
    F --> P["Robot diffusion policy"]
    A --> P
    P --> W["Future-flow diffusion"]
    F --> W
    W --> Q["Future query points"]
    Q --> R["Nvblox + robot mesh raycast reward"]
    V["Human-view diffusion prior"] --> D["Reward-max denoising"]
    R --> D
    D --> C["Z1 camera arm"]
    P --> M["WidowX manipulation arm"]
~~~

## 3. 结果拆解

| Method | Spray | Doll | Toilet Paper | Towel | Avg |
| --- | ---: | ---: | ---: | ---: | ---: |
| HVI（human viewpoint imitation） | 5/25 | 10/25 | 7/25 | 5/25 | 27% |
| AMPLIFY 2D-flow | 4/25 | 6/25 | 9/25 | 9/25 | 28% |
| EgoZero 3D-flow | 10/25 | 7/25 | 7/25 | 9/25 | 33% |
| Phantom robotized RGB | 5/25 | 7/25 | 6/25 | 8/25 | 26% |
| **EgoAVFlow** | **20/25** | **18/25** | **17/25** | **18/25** | **73%** |

HVI对照隔离visibility planning；后三个baseline共享EgoAVFlow view module，主要隔离manipulation representation。两类问题不应混成单一ablation：完整方法对HVI同时换了view optimization，完整方法对EgoZero主要换flow construction/model assumptions。

每task 25 trials意味着4 pp分辨率，无CI或policy seeds。论文failure composition用所有methods失败数作占比，早期grasp失败会让某method永远不进入later out-of-view category；作者虽提醒survivorship bias，图本身仍不能比较conditional failure probability。

## 4. “Scalable active vision”的系统成本

| 组件 | 是否需要 | 隐含成本/风险 |
| --- | --- | --- |
| Human task demos | 150/task | 非one/few-shot |
| RGB-D + head rig | 是 | 不是普通RGB视频 |
| ChArUco world marker | 是 | 限制in-the-wild deployment |
| Query annotation | 6 points/task | 不会自主找目标 |
| Pixel tracking/depth/SLAM | train+test | 任一失效会污染flow |
| Online TSDF/mesh | test | raycast visibility的基础 |
| Robot URDF+future IK mesh | test | self-occlusion预测 |
| Independent camera robot | Z1 arm | workspace、碰撞与成本显著 |
| Robot demonstrations | 否 | 核心优点 |

## 5. Artifact audit 与最终判断

| Artifact | 状态 | 判断 |
| --- | --- | --- |
| arXiv + project video | 公开 | 方法和qualitative可核验 |
| Training/eval code | 未发现 | reward denoising细节不可复跑 |
| 4×150 human dataset | 未发现 | baseline公平性和preprocessing不可核验 |
| Checkpoints/configs | 未发现 | 结果不可独立复验 |
| Hardware/control assets | 论文描述WidowX+Z1+D435 | 无标定/安全/controller recipe |

下一步最关键的实验不是再加一个success table，而是自动query discovery和lost-track re-acquisition；在无marker的object/robot-centric frame运行；把task-only SR与visibility-only SR拆开；对depth/SLAM/mesh/flow uncertainty做risk-aware reward；用pan-tilt head或mobile base验证camera embodiment；最后加入硬collision/velocity constraints与latency报告。做到这些后，“未来3D motion驱动主动视点”才可能从精心搭建的双臂实验迁移到真正humanoid head–hand coordination。
