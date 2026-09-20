---
title: "Unlocking In-the-Wild Loco-Manipulation with Robot-Free Egocentric Demonstration"
method_name: "EgoHumanoid"
authors: [Modi Shi, Shijia Peng, Jin Chen, Haoran Jiang, Tianyu Li, Di Huang, Ping Luo, Hongyang Li, Li Chen]
year: 2026
venue: RSS
tags: [humanoid, loco-manipulation, egocentric-demonstration, human-robot-cotraining, view-alignment, action-alignment, vision-language-action, whole-body-control]
image_source: online
---

# EgoHumanoid：人类户外示范如何帮助 G1 做 loco-manipulation？

> 本笔记基于 [arXiv:2602.10106v2](https://arxiv.org/abs/2602.10106)、[HTML 全文](https://arxiv.org/html/2602.10106)、[项目页](https://opendrivelab.com/EgoHumanoid)、[Apache-2.0代码库](https://github.com/OpenDriveLab/EgoHumanoid)与[样例数据](https://huggingface.co/datasets/OpenDriveLab/EgoHumanoid)核验。仓库标注 RSS 2026；v2更新于2026-06-04。

## 阅读结论先行

EgoHumanoid是把 human–robot co-training真正推进到双足loco-manipulation的重要实证：100条/任务的G1 teleop提供精细接触与可执行anchor，300条/任务的人类PICO+ZED示范提供家庭、花园、便利店等scene diversity。MoGe深度reprojection+Stable Diffusion inpainting把人类约1.7 m视角向下平移约0.25 m，delta wrist pose、binary grasp和离散pelvis locomotion primitives再把两域动作统一，最后finetune $\pi_{0.5}$。

加人类数据后，in-domain normalized score从59到78（+19 percentage points），generalization setting从31到82（+51 pp）。摘要称“outperform ... by 51%”容易误读：若相对robot-only是 $(82-31)/31\approx164.5\%$，所以51其实是**百分点差**。此外，generalization场景“covered only by human data”：robot没在这些地点采集，但policy看过同场景的人类视频，故这是human→robot的scene transfer，不是对训练期间完全未知场景的zero-shot。

论文另在Trash Disposal做真正held-out scene test：固定100 robot+300 human episodes，人类训练场景从1/2/3个增加，平均score为75/75/82.5，robot-only 57.5。它支持scene diversity有益，但只有一任务、四个离散点；“monotonic”在1→2其实持平，且没有多seed/CI，尚不是scaling law。

Subskill分析非常诚实且重要：Human-only在四个首段navigation均100%，coarse pillow/trash manipulation也强；Cart Stowing精细抓取仅5%，robot-only 15%，co-training才60%。人类data主要迁移路线、approach、停止位置和粗动作结构；目标robot data仍负责精确抓取、接触、三指手和dynamics。它不是robot-data-free training，而是robot-free **supplementary demonstrations**。

“whole-body”是task-level而非joint-level，论文自己明确说明。18D policy action包含两腕12D delta SE(3)、x/y/yaw三项离散navigation、双binary gripper和一项height delta；没有腿关节、footstep、CoM、force或proprioception。GR00T WholeBodyControl低层policy与IK吸收平衡、步态和关节执行。因此上层VLA学的是视觉条件的locomotion primitive+hand targets，不是从人类视频学动态全身控制。

不输入proprioception让human/robot observation维度一致，却引入partial observability：同一RGB和delta action在不同joint posture、足相位、reachability下可能含义不同。作者也承认delta rotation ambiguity和discrete locomotion限制。Action chunk长50 steps、20 Hz即2.5 s；具体闭环安全依赖低层WBC拒绝/修正不可执行command，论文未报告fall、emergency stop或controller clipping统计。

View alignment有全任务positive ablation，尤其高低物体变化大的Toy/Cart；但它由monocular affine-invariant MoGe估depth，再从单视角reproject disoccluded regions并让diffusion hallucinate，无法保证object geometry/contact边界。No alignment还同时缺少view augmentation，论文未拆分depth warp、random pose和inpainting各自贡献，也未评估生成artifact与action label错位。

人类采集并不完全“自然”：示范者需保持固定hand–wrist orientation、减少torso sway、始终让手可见，必要时改变自然approach。五个body trackers和26 hand keypoints/手比普通视频结构化得多。采集平均39.7 s/episode对robot 62.1 s，只快约1.56×，并非严格2×；Cart为89.9 vs103.5 s，仅1.15×。优势更主要是无需搬机器人即可覆盖环境，而不只是动作速度。

### 一句话总结

EgoHumanoid证明了：用视角与动作对齐把task-matched、场景丰富的人类结构化示范加入少量G1数据，可以大幅修复robot-only在这些场景中的导航与approach失败；但精细操作仍依赖robot anchor，所谓whole-body由低层WBC承担，而82 vs31是跨本体seen-human-scene transfer，不应包装成未知环境51%相对提升。

![EgoHumanoid 人类与G1统一采集硬件](https://arxiv.org/html/2602.10106v2/hardware.png)

*图 1。官方采集系统。两端共用ZED/PICO思路，人端另有五枚body trackers，robot端用VR teleop和WBC。*

## 1. View alignment：把人类相机下移到机器人高度

MoGe输出scale-invariant point map $P(u,v)$；给定模拟camera transform $T_{h\to r}$，重投影为：

$$
\tilde p=T_{h\to r}\begin{bmatrix}P(u,v)\\1\end{bmatrix},
\qquad (\tilde u,\tilde v)=\Pi(K\tilde p).
$$

固定向下0.25 m并加$\pm0.05$ m uniform noise；holes用Stable Diffusion 2.0、20 denoising steps、CFG 7.5补全。这个离线转换不要求test-time depth，却可能改写物体/手/接触像素。最需要的附加指标是warp后geometry reprojection error，以及human action在新view下的pixel alignment。

## 2. Action alignment：共享的是task-space，不是人体关节

Upper-body用pelvis frame的左右wrist，translation经Savitzky–Golay，rotation在 $SO(3)$ tangent space平滑，100 Hz降到20 Hz后取相邻delta：

$$
\Delta T_t=T_t^{-1}T_{t+1},\qquad
a_t^{upper}=[\Delta T_t^L,\Delta T_t^R].
$$

Pelvis displacement投到local frame，量化成forward/lateral/yaw各3 bins；pelvis height差阈值化成stand/squat。Human grasp以五指polyline curvature平均后threshold成open/close。这丢弃了26-keypoint多指信息，Dex3也只收到binary grasp，不属于dexterous-hand policy。

~~~mermaid
flowchart LR
    H["Human PICO + trackers + ZED"] --> V["MoGe reprojection + inpainting"]
    H --> A["Wrist deltas + pelvis primitives + grasp"]
    R["100 G1 teleop demos/task"] --> U["Shared 18D action + ego RGB"]
    V --> U
    A --> U
    D["300 human demos/task"] --> U
    U --> P["pi0.5 co-training"]
    P --> W["GR00T WBC + IK"]
    W --> G["G1 real loco-manipulation"]
~~~

## 3. 主结果：score不是完整episode success

每setting 20 trials，逐substep给credit再归一化。82/78等是平均normalized score，不一定代表82% episodes完整成功；长任务后续substep是consecutive，前段失败会阻断后段。Table I的per-stage百分比应与full completion分开读。

| Training | In-domain avg score | Human-covered generalization score |
| --- | ---: | ---: |
| Robot-only，100/task | 59 | 31 |
| Co-training，+300 human/task | 78 | 82 |
| Absolute gain | +19 pp | +51 pp |

四任务从1–5 m移动，覆盖pillow carry/place、trash insertion、toy grasp-carry-place与push-cart+stow。任务广度不错；仍只有一台G1、一套low-level controller和每格20 trials，无不同robot、operator、training seed或置信区间。

## 4. 哪类knowledge真从human转移

| Subskill | Human-only evidence | 判断 |
| --- | --- | --- |
| 首段navigation/approach | 四任务均100% | **强迁移**，主要价值 |
| Pillow placement | 95% | coarse/deformable target容错高 |
| Trash insertion | 80% | 中等precision也可迁移 |
| Toy后续turn+walk | 45% vs co-train60% | 多阶段误差累积 |
| Cart precision grasp | 5% vs robot15/co-train60% | human alone不足，协同明显 |
| Cart最终stages | human 0/0，co-train50/50 | robot embodiment anchor关键 |

Human-only能在G1执行说明shared action不只是representation pretraining；但人类示范仍通过task-specific语言/scene与robot demos配对。没有off-task human control，无法判断收益来自动作alignment、scene pixels还是多数据正则化。

## 5. Artifact audit

| Artifact | 状态 | 可复现边界 |
| --- | --- | --- |
| GitHub | 7 commits；collection/alignment/openpi train/deploy齐全，Apache-2.0 | 核心pipeline可审计 |
| Sample HF data | human+robot example | 可smoke-test，不是1200 human+400 robot全集 |
| Full training corpus | 未见公开 | 主结果/scene scaling不可独立复验 |
| Checkpoint | README未提供论文checkpoint | 需自行训练；full FT约70 GB VRAM，论文8×A100 |
| Low-level control | 引用GR00T-WBC，deployment instructions存在 | 外部复杂依赖、实机G1+Dex3必需 |

值得称赞的是repo不只放landing page，而包含人/机器人采集、action processing、view warp、LeRobot conversion、training/server和G1 client。仍应补全full data、paper checkpoints、exact experiment configs/seeds及scoring scripts。

## 6. 最终研究判断

EgoHumanoid最坚实的结论不是“human data替代humanoid data”，而是两者有清晰分工：human data用低成本场景覆盖教navigation与高层动作结构，100条robot demos/task校准precision/contact。下一步应做scene/task的严格四象限：human seen/unseen × robot seen/unseen；等episode/等frame/等compute对照；将view warp三组件拆开；报告完整episode SR、falls与WBC intervention；并把continuous base velocity或footstep goals与proprioception通过embodiment-specific encoder引入。这样才能判断扩到更多human data后，瓶颈究竟是场景覆盖、action噪声还是低层可行性。
