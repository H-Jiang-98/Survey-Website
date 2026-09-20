---
title: "Phantom: Training Robots Without Robots Using Only Human Videos"
method_name: "Phantom"
authors: [Marion Lepert, Jiaying Fang, Jeannette Bohg]
year: 2025
venue: CoRL
tags: [human-video, data-editing, human-to-robot, imitation-learning, zero-robot-data, hand-pose-estimation, inpainting, cross-embodiment]
image_source: online
---

# Phantom：只用人类 RGB-D 视频，何时真的可以“无机器人数据”训练？

> 本笔记基于 [CoRL 2025 项目页](https://phantom-human-videos.github.io/)、[arXiv:2503.00779v2](https://arxiv.org/abs/2503.00779)、[HTML 全文](https://arxiv.org/html/2503.00779)和 [MIT 代码库](https://github.com/MarionLepert/phantom)精读核验。v2 更新于 2026-05-28；书目信息按项目页的 CoRL 2025。

## 阅读结论先行

Phantom 是“video-to-robot data synthesis”路线里最清楚的 zero-robot-data demonstration：第三人称 ZED2 RGB-D 拍人用拇指—食指 pinch 完成任务；HaMeR 估 MANO hand，SAM2+depth 得 partial point cloud，ICP 校正 absolute 3D；以两指 tips midpoint/plane/axis 构造 gripper SE(3)+opening action；E2FGVI 抹去手臂，再在同 pose 渲染目标机器人。Diffusion Policy 只看这些合成 observation-action pairs，直接在 Franka 或 Kinova 上 closed-loop 执行。

“没有 robot data”在 demonstration/training-data 意义上成立：没有 teleoperation、robot rollout fine-tune 或 RL。但系统并非无需机器人知识：需要目标 robot URDF/MuJoCo model、camera intrinsics/extrinsics、workspace→base calibration、低层 OSC/IK controller、真实机器人 test-time joint state 来渲染 virtual overlay，以及确认 human strategy 对 robot kinematics 可行。这是 **zero robot demonstrations**，不是 zero target-hardware engineering。

视觉 editing 的证据很强。在相同 human data 上，Vanilla 和 Red Line 在五个 ID tasks 全为 0；Hand Inpaint 在 book/cups/rope/box 为 92/72/64/72%，sweep 各阶段 88/80/72/40%。Hand Mask 相近，但 inference 还跑 diffusion hand-mask generator，rollout 平均慢 73%。OOD sweeping 在 lawn/lounge/lounge+new-surface 为 72/84/64%。每格 25 trials，分辨率 4pp，仍无 CI/多 seed。

但“up to 92% on diverse tasks”来自每 task 250–350 条 **专门采集** 的 human demonstrations；OOD sweeping另收 950 条跨场景示范。它没有用 internet-scale被动视频，也没有 single/few-shot skill acquisition。收集比 teleoperation轻，却仍要求 RGB-D、相近 camera viewpoint、手始终可见、固定 pinch strategy、已知 task 和可执行人类动作。

动作 label 本质上把 human pinch 当 robot gripper。Position 是 thumb/index tips midpoint；orientation由两指 keypoints plane normal与 thumb principal axis；opening 是 tip distance，trajectory 内最低20%强制 fully closed。这对 parallel-jaw tasks巧妙，但不是一般人体动作到机器人 action：只测试准静态单臂抓取/插入/扫/绳/箱体；没有双手、灵巧指、mobile base、动态 catch、force/tactile或接触约束。

Test-time 仍把虚拟 robot 盖在真实 robot 上，解决 synthetic-real texture gap，却要求准确、低延迟的机器人 pose rendering与camera calibration。Policy 输入看到的是 hybrid rendered embodiment，而不是 raw observation；真实机械臂对物体/背景产生的 occlusion若 overlay/depth不准会泄露 artifacts。论文没有 no-test-overlay ablation、calibration-noise sweep、render latency或 failure rate。

最重要的安全限制是 human trajectory 未做 robot collision/feasibility optimization。作者承认人手不碰环境不代表粗机械臂可达，finger/gripper表面摩擦也不同；仅靠 low-level controller 不能保证安全。实验没有 collision、controller rejection、tracking error或不可达 frames统计。“any robot capable of executing the task”应读为每个 target robot重新 render + controller integration，而不是单 checkpoint 即插即用跨机器人。

代码公开程度较好：MIT repo提供 Phantom/Masquerade 共用处理 pipeline、sample raw/processed data、Panda/Kinova/UR5e/IIWA/Jaco render configs及 Diffusion Policy说明。限制是只有2 commits、3D hand pipeline只测左手、需另行注册下载 MANO、完整 250–950 demo datasets/训练 checkpoints/real evaluation scripts不见于 repo。因此能跑 sample conversion，不能完整重现论文成功率。

### 一句话总结

Phantom 证明了在 RGB-D、近似视角、单手 pinch、准静态且 human strategy 与 parallel-jaw robot 可共享的条件下，精确 hand pose + arm inpainting + target-robot rendering 可以把纯 human demonstrations变成可 closed-loop 部署的机器人 imitation data；它消除了 teleoperation，却没有消除目标 robot calibration、controller、可行性和 test-time renderer，范围也远窄于任意互联网视频/任意机器人。

![Phantom：human RGB-D 到 robotized observation-action pairs](https://phantom-human-videos.github.io/static/images/method.png)

*图 1。官方 pipeline。训练与测试都把视觉本体转换为相同的 virtual target robot。*

## 0. Problem setup 与隐含假设

Human dataset 为 $\mathcal D_h=\{\tau_h^i\}$，目标是逐帧转换：

$$
I_{h,t}\longrightarrow(I_{r,t},a_{r,t}),\qquad
a_{r,t}=(p_t,R_t,g_t).
$$

$p_t\in\mathbb R^3$，$R_t\in\mathbb R^6$ 为 continuous rotation representation，$g_t\in[0,1]$。成功依赖：

- RGB-D 与可靠 camera extrinsics；collection/test viewpoint相近。
- 左手在每帧可见、thumb-index pinch足以表达 robot gripper。
- 目标 robot能复现同 end-effector strategy，low-level controller处理 dynamics。
- edited image与 test overlay足够一致；background/object state本身无需重建。

“不需要 object model”是优点：rope、多碎片 sweep可直接通过 closed-loop pixels处理；代价是 action feasibility和object-contact semantics都未显式检查。

## 1. 从 hand mesh 到 robot action

HaMeR给 21 keypoints $\hat X_t$ 和 778 mesh vertices $\hat V_t$。SAM2 hand mask在 depth image取 partial point cloud $P_t$，ICP求：

$$
T_t^*=\arg\min_{T\in SE(3)}d(P_t,T\hat V_t),qquad
X_t=T_t^*\hat X_t.
$$

Thumb/index末两 joints被约束为1DoF且有限角度，缓解遮挡时 HaMeR 的异常 ball-joint pose。Target position：

$$
p_t=\frac12(x_t^{\text{thumb-tip}}+x_t^{\text{index-tip}}).
$$

Orientation来自两指全部 keypoints拟合平面的 normal加 thumb direction；gripper opening由 tip distance归一化。ICP对 partial/occluded point cloud、反光 depth 和 segmentation leakage敏感；论文没有 hand-pose/action label accuracy 的 mocap ground truth评测，因此 downstream success是唯一间接验证。

## 2. 视觉域对齐与闭环

~~~mermaid
flowchart LR
    V["Human RGB-D video"] --> H["HaMeR + SAM2 + depth ICP"]
    H --> A["EEF pose + gripper label"]
    V --> M["SAM2 arm mask"]
    M --> P["E2FGVI inpainting"]
    A --> R["MuJoCo target-robot rendering"]
    P --> R
    R --> D["Robotized demos"]
    D --> PI["Diffusion Policy"]
    O["Real robot RGB + joint pose"] --> T["Virtual robot test overlay"]
    T --> PI
    PI --> C["OSC / IK controller"]
    C -. "closed-loop next observation" .-> O
~~~

Training overlay用 scene depth做 object occlusion；test overlay覆盖真实机械臂以统一 texture。这个设计把domain alignment问题转为rendering accuracy问题。Hand Mask则在test额外生成一张人手形状 mask，说明训练/测试都退化到相同抽象也可行，但增加73% rollout time。

## 3. 结果应按任务和完成层级读

| ID task / criterion | Phantom | Hand Mask | Red Line / Vanilla |
| --- | ---: | ---: | ---: |
| Pick/place book | 92 | 92 | 0 / 0 |
| Stack cups（直径差1.5cm） | 72 | 52 | 0 / 0 |
| Tie simplified cleat hitch | 64 | 60 | 0 / 0 |
| Rotate box 90° | 72 | 76 | 0 / 0 |
| Grasp brush | 88 | 75 | 0 / 0 |
| Sweep >0 / >2 / >4 of 6 | 80 / 72 / 40 | 75 / 72 / 68 | 0 |

Sweep不是一个 success rate，而是嵌套 thresholds；Hand Mask在最严格 >4反而高28pp，不能简单说 Hand Inpaint每任务都最好。OOD 三场景只有 sweep，scene generalization尚未覆盖 insertion/deformable/rotation。

High-quality E2FGVI / OpenCV / mask-only 在 indoor lounge sweep为 84/76/60%。这说明 clean inpainting有价值，但 simple artifacts也可被augmentation吸收；真正不可缺的可能是 **robot overlay**，因为 mask-only仍有60而 Red Line/Vanilla为0。不过这些对照跨设置（OOD vs更容易ID）时不能直接作严格因果推断。

## 4. “Scalable”具体省了什么、没省什么

| 环节 | Phantom是否消除 | 说明 |
| --- | --- | --- |
| Robot teleoperation demos | 是 | 核心贡献 |
| Human task demos | 否 | 250–350/task；OOD sweep 950 |
| RGB-D/calibration | 否 | monocular不够，extrinsics已知 |
| Target robot model/render | 否 | 每 robot需要URDF/MuJoCo assets |
| Low-level control | 否 | Franka OSC、Kinova IK |
| Feasibility/collision checking | 否 | 主要风险 |
| Test-time robot state/render | 否 | virtual overlay依赖它 |
| Robot rollout evaluation | 否 | 当然仍需实机验证 |

## 5. 关键缺失对照

- Exact human trajectory labels但无visual editing，和 exact visual editing但扰乱actions，拆开 action/appearance贡献。
- Human+robotized view augmentation vs只robotized；test overlay on/off与render color/randomization。
- Pose噪声、extrinsic误差、depth dropout、camera viewpoint变化与controller tracking sweep。
- IK/collision-filtered actions、轨迹优化或residual safety controller。
- 与 object flow/point tracking在相同zero-robot、同human demos下比较；论文以目标不同为由没有direct method baseline。
- Demo scaling curve：25/50/100/300/950 条 human videos，而不是只给最终大数据点。

## 6. Artifact audit 与最终判断

| 主张 | 证据 | 判断 |
| --- | --- | --- |
| Zero robot demonstrations closed-loop | 六类 real tasks、两机器人 | **成立**，条件明确 |
| 视觉 editing 必须 | Vanilla/Red Line 0，Inpaint/Mask高 | **强支持** |
| 任意环境/机器人 | 三个OOD sweep scenes；两parallel-jaw arms | **过度概括** |
| Deformable/multi-object可做 | rope与sweep | **支持**，仅准静态 |
| 高质量 inpainting必要 | 84 vs76 vs60 | **有帮助但不绝对必要** |
| 可完整复现 | MIT processing code+sample；full data/checkpoints缺 | **部分可复现** |

Phantom最值得借鉴的是把 train/test embodiment都“投影”成同一个 rendered robot，而不是指望policy自己忽略human/robot appearance。它的自然下一步是把 action retargeting放入 collision-aware optimization，并以多视角/monocular metric reconstruction替代固定 ZED2 setup；否则规模扩大只会更快地产生不可达或碰撞的 labels。
