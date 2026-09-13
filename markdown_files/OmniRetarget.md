---
title: "OmniRetarget: Interaction-Preserving Data Generation for Humanoid Whole-Body Loco-Manipulation and Scene Interaction"
method_name: "OmniRetarget"
authors: [Lujie Yang, Xiaoyu Huang, Zhen Wu, Angjoo Kanazawa, Pieter Abbeel, Carmelo Sferrazza, C. Karen Liu, Rocky Duan, Guanya Shi]
year: 2025
venue: arXiv
tags: [motion-retargeting, interaction-mesh, loco-manipulation, contact-preservation, data-augmentation, humanoid-RL]
image_source: online
---

# OmniRetarget：用 interaction mesh 保留人—物—地形关系

论文：[OmniRetarget](https://arxiv.org/abs/2509.26633) · 项目页：[omniretarget.github.io](https://omniretarget.github.io/) · 代码：[Amazon FAR Holosoma](https://github.com/amazon-far/holosoma) · 数据入口：[OmniRetarget Dataset](https://huggingface.co/datasets/omniretarget/OmniRetarget_Dataset)

## 一、论文速览

### 一句话总结

OmniRetarget 用包含人体/机器人关键点、物体表面和地形点的 interaction mesh，把人体动作重定向写成带非穿透、关节/速度限位和 stance-foot sticking 的 sequential SOCP，从单条示范生成不同机器人、物体位姿/形状和地形条件下的交互保持 reference，再用极简 proprioceptive RL 训练长时程 loco-manipulation。

### Elevator pitch

传统 keypoint retargeting 主要追求身体姿态，却常在脚底漂移、穿透物体和不正确的接触位置上失败；下游 RL 只能用复杂 reward 把这些脏 reference “修回来”。OmniRetarget 将环境与物体表面采样进同一 interaction mesh，最小化 source/target Laplacian deformation，同时把 foot sticking、signed-distance collision、joint/velocity bounds 作为硬约束。它还把物体初始 pose、形状、平台高度等变化转成新的优化问题，因此一条人体示范可以产生大批 robot-object-terrain trajectories。论文在 OMOMO、LAFAN1 和自采 MoCap 上报告超过 8 小时数据，并在 Unitree G1 上以五项 reward 和少量随机化完成攀爬、搬运、跳跃、翻滚与 wall flip；但 reference 仍主要是运动学结果，动力学和真机安全依赖后续 RL/controller。

### 研究生态位

| 层 | OmniRetarget 的位置 |
| --- | --- |
| Representation | volumetric interaction mesh + Laplacian coordinates |
| Optimization | sequential SOCP / linearized constrained update |
| Interaction | robot–object、robot–terrain、robot-only |
| Data | retargeting + systematic augmentation |
| Policy | proprioceptive whole-body tracking RL |
| Sim2real | 高质量 reference 减少 reward engineering，再配合有限 domain randomization |

## 二、交互网格与优化目标

### 2.1 Interaction mesh

source/target mesh 的顶点由人体/机器人关键点、物体表面点和环境/地形点构成，通过 Delaunay tetrahedralization 得到邻接关系。对第 $i$ 个顶点，Laplacian coordinate 为：

$$
L(p_{t,i})=p_{t,i}-\sum_{j\in\mathcal N(i)}w_{ij}p_{t,j}.
$$

它描述点相对邻域的局部几何，而不是绝对世界位置；因此把物体点放在 object frame 中特别重要：物体整体旋转/平移时，人与物的相对接触结构可以保持不变。点采样越密，接触表面越受约束，但求解规模和碰撞检测成本也越高。

### 2.2 单帧约束优化

给定前一帧结果，机器人 configuration $q_t$ 的 nominal retargeting 目标为：

$$
\begin{aligned}
q_t^\star=\arg\min_{q_t}\;&
\sum_i\left\|L(p_{t,i}^{source})-L(p_{t,i}^{target}(q_t))\right\|_2^2
 +\left\|q_t-q_{t-1}\right\|_Q^2\\
\text{s.t. }&\phi_j(q_t)\geq 0,\quad q_{min}\leq q_t\leq q_{max},\\
&v_{min}dt\leq q_t-q_{t-1}\leq v_{max}dt,\\
&p_t^F=p_{t-1}^F,\quad \forall\text{ stance foot}.
\end{aligned}
$$

$\phi_j$ 是机器人与对象/地形碰撞对的 signed distance；stance foot 由 source motion 中水平速度低于 1 cm/s 的脚判定。该形式同时处理局部形变、时间平滑、关节/速度限位和脚粘地，核心差别是把不可穿透与接触保持从 soft penalty 提升为硬约束。由于问题包含非线性 FK，实际实现将其线性化，在每个时间步迭代求解 SOCP，上一帧解作为 warm start，并以 trust region 限制线性近似的步长。

### 2.3 交互增强

物体增强会改变初始 translation/rotation、三轴尺寸，并用 exponential interpolation 将新 pose 融入原物体轨迹；terrain 增强会改变平台高度/深度，并向 mesh 采样新地面网格。为了避免整个机器人和物体只做刚体平移，优化额外惩罚机器人偏离 nominal trajectory：

$$
\left\|q_t-\bar q_t^\star\right\|_W,
\qquad
p_0^F=\bar p_0^{F\star}\quad\text{(left/right feet)}.
$$

$W$ 可重罚下肢变化，使上肢产生新的协调而不是全身跟着物体刚性移动。增强因此不是随机扰动，而是“改变场景后重新求解可行 reference”。

![OmniRetarget 的 interaction-mesh 重定向与增强系统](https://arxiv.org/html/2509.26633v3/x1.png)

*图 1：从人体示范、interaction mesh 到机器人 reference 与 RL 的总体流程；图片来自论文 HTML。*

### 图表核对

在[论文 HTML](https://arxiv.org/html/2509.26633)中还核对了 Fig. 2 的 G1/H1/T1 跨本体 interaction、Fig. 3 的 terrain/object pose/shape augmentation、Fig. 4 的 whole-body scene interaction、Fig. 5 的 wall flip、Fig. 6 的 baseline artifacts、Fig. 7–8 的 object-frame Laplacian 与 source/target mesh、Fig. 9 的下游 RL failure histogram。Table I 比较 hard constraints、object/terrain interaction、augmentation 与 solver，Table II 汇报 penetration、skating、contact preservation 和 downstream RL，Appendix 的 Algorithm 1/公式 15 解释 PHC fitting 与 sequential SOCP。图表共同支持“reference quality 影响 RL”的判断，但不把运动学可行误写成动力学安全。

## 三、下游 RL：reference 质量如何转化为动力学行为

OmniRetarget 的 policy 不直接看 object/terrain geometry，而是通过 reference 学习跟随。最小 observation 包括 reference joint position/velocity、reference pelvis position/orientation error、pelvis linear/angular velocity、joint position/velocity 和 previous action；agile motion 可 mask pelvis linear position error 与 velocity。

论文固定五项 reward：

1. DeepMimic-style body tracking（位置、方向、线速度、角速度）；
2. 有物体时的 object tracking；
3. action-rate penalty；
4. soft joint-limit penalty；
5. self-collision force 超过 1 N 的 binary penalty。

object 偏离 reference 超过 1.0 m 或 45° 才终止。机器人只做四类随机化：torso COM、joint default position、random push、observation noise；物体则随机 mass $0.1$–$2$ kg、COM ±0.08 m、inertia 50–150%、shape ±10%。这组设计支持“高质量 reference 可以减少 reward engineering”的论点，但不代表完全没有 task-specific engineering：wall flip 仍放宽 end-effector error threshold 到 0.5 m，并移除 foot orientation tracking reward。

## 四、实验与结果

### 4.1 交互场景与真机行为

论文覆盖 OMOMO robot-object、in-house MoCap robot-terrain 和 LAFAN1 robot-only。G1 展示 box carrying、0.9 m platform climbing、slopes、jumping、rolling、chair-as-stepstone parkour 及 wall flip。wall flip 约 0.5 s、峰值角速度 15 rad/s，真机 5/5 成功；峰值线速度约 3.5 m/s。这里的成功是“retargeted data + RL policy + hardware/controller”的系统结果，不能理解为 SOCP 单独保证动力学安全。

### 4.2 运动学 benchmark

指标为 penetration（持续时间/最大深度）、foot skating（持续时间/最大滑动速度）和 contact preservation（期望接触时长比例）。论文在不同任务平均给出：

| 方法 | Penetration duration | Max penetration | Foot-skating duration | Max skating velocity | Contact preservation | Downstream success |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| PHC | 0.66 ± 0.36 | 7.74 ± 4.53 | 0.15 ± 0.04 | 2.03 ± 1.83 | 0.45 ± 0.28 | 52.63% ± 49.93% |
| GMR | 0.91 ± 0.16 | 5.72 ± 3.84 | 0.04 ± 0.05 | 1.75 ± 3.01 | 0.67 ± 0.26 | 78.94% ± 40.77% |
| VideoMimic | 0.83 ± 0.11 | 5.97 ± 3.58 | 0.14 ± 0.05 | 1.85 ± 1.38 | 0.47 ± 0.25 | 51.75% ± 49.23% |
| OmniRetarget | **0.01 ± 0.02** | **1.37 ± 0.18** | **0** | **0** | **0.72 ± 0.19** | **94.73% ± 22.33%** |

OmniRetarget 在大多数几何指标显著更好，但论文承认 sequential SOCP 的线性化仍会造成少量 penetration，后续 RL 可能修复它。GMR 在 robot-object contact preservation 上有时更高，因为其 keypoint objective 直接追手部点；这提醒我们不能用单一接触指标判定整体质量。

### 4.3 数据规模与增强的因果边界

论文称生成超过 8 小时轨迹，公开数据/仓库的实际可用范围需要按版本核对；论文 benchmark 中明确列出 2.78 小时 OMOMO box carrying、1 小时 in-house MoCap、4.6 小时 LAFAN1。仅依靠 domain randomization 去扰动 object shape/pose 时，policy 难以离开 nominal reference；显式 augmentation 后才在远离 nominal 的真实场景中保持成功。这支持“数据分布变化必须反映在 reference 中”的观点，但未证明所有新增场景都能从一条 demonstration 外推。

## 五、代码、数据与复现

当前代码入口是 Holosoma 的 `src/holosoma_retargeting/`，核心包括 `interaction_mesh_retargeter.py`、配置类型/默认值、单条与并行 retarget 脚本，以及 MuJoCo/Isaac 训练流程。仓库提供 retargeting setup 和 OMOMO/LAFAN demo，但输入数据、SMPL/机器人资产和部分许可证仍需单独处理。

| 复现层级 | 目标 | 关键依赖 | 成功标准 |
| --- | --- | --- | --- |
| Level 1 | 一条 OMOMO/LAFAN motion 的 mesh、SOCP、可视化 | robot MJCF/URDF、scene/object points、MuJoCo | 无明显穿透，输出可回放 |
| Level 2 | 复现 penetration/skating/contact 表与一项 tracking policy | 完整 configs、parallel retarget、RL simulator | 同时报告 reference 与 policy 指标 |
| Level 3 | 新 object/terrain/robot augmentation | 场景几何、碰撞模型、约束调试 | 新配置不退化且记录不可行 case |

主要风险不是网络训练，而是 scene scaling、object-frame 建模、stance detection、collision geometry 和 sequential solver 的数值稳定性。reference 仍是运动学轨迹；如果需要力闭环、摩擦锥或 actuator saturation，应再接 WBC/MPC/低层 policy 并单独验证。

### 代码质量评分

| 审查项 | 评分 | 证据与判断 |
| --- | --- | --- |
| README/安装文档 | 高 | Holosoma 提供独立 retargeting setup、demo scripts 和训练/推理结构 |
| 环境可复现性 | 中 | IsaacGym/IsaacSim/MuJoCo 路线较多，具体版本、GPU 和资产组合需要锁定 |
| retargeting 核心 | 高 | `interaction_mesh_retargeter.py`、配置类型、单条/并行入口公开 |
| 物体/地形资产 | 中 | OMOMO、LAFAN1、scene mesh、collision model 与许可证需自行准备或核验 |
| 数据集/样例 | 中 | 有公开数据入口与 demo，但论文声称的全部时长和处理结果需按版本复核 |
| checkpoint/下游 RL | 中 | retargeting 与 Holosoma training 集成，预训练 object-interaction policy 并非完整交付 |
| 可视化与诊断 | 高 | MuJoCo/数据处理脚本和 kinematic metrics 便于定位 penetration/contact 问题 |
| 维护/许可证 | 中 | 代码持续演进但配置面较宽；新增 robot/scene 仍需较多工程调试 |

## 六、批判性思考

### 优点

1. **把 interaction 提升为一等公民**：物体和地形点与机器人点进入同一个 mesh，不再事后猜接触。
2. **约束表达直接**：foot sticking、non-penetration、joint/velocity limits 在 retargeting 阶段显式出现，减少用 reward 补漏洞的需要。
3. **数据增强有几何意义**：对象 pose/shape、terrain height 和 robot embodiment 的变化通过重新求解产生新的 reference，而非只扰动 observation。
4. **下游验证闭环**：运动学指标、tracking policy、长时程真机与 failure histogram 相互印证。

### 局限性

1. 硬约束保证的是离散几何/运动学可行性，不等同于动力学、摩擦、柔顺、执行器和安全可行性。
2. stance foot 依赖 source 水平速度阈值；跨动作分布时，错误的 stance detection 可能把不该粘住的接触永久化。
3. mesh 点采样、Delaunay 结构、碰撞模型和物体 frame scaling 都是工程敏感项；论文给出方法原则，但不同资产仍可能需要调参。
4. 下游 success 受 termination criterion 影响；例如 OmniContact 的较宽松终止可能让 policy 在 joint tracking 明显偏差时仍被计为成功。
5. 真机案例集中在 G1 与少量精心选择的技能，不能推出对任意 humanoid、柔性物体或未知摩擦的安全保证。

### 潜在改进

- 将 stance/contact mode 从速度启发式升级为带不确定性的 contact estimator 或可学习 contact schedule；
- 在 SOCP 中加入摩擦锥、力/力矩界、质心和 actuator feasibility，并输出 infeasibility certificate；
- 对 object/terrain augmentation 做真实接触分布校准，报告 OOD 场景的置信度与拒绝策略；
- 以统一 reference 数据和同一 policy/termination，复核 PHC、GMR、UMR 与 OmniRetarget 的公平差异。

## 七、相关工作与最终判断

[PHC/OmniH2O](https://arxiv.org/abs/2406.08858) 先拟合 SMPL shape 再做关键点重定向，强项是和 teacher–student teleoperation pipeline 结合；[GMR](https://arxiv.org/abs/2510.02252) 用多机器人实时 IK 改善无物体 motion fidelity；[UMR](https://arxiv.org/abs/2609.02134) 学习密集表面 correspondence，试图减少手工 skeleton mapping。OmniRetarget 的独特位置是“interaction mesh + hard constraints + systematic augmentation”，而非单纯追求更低 pose error。

| 维度 | PHC | GMR | OmniRetarget | UMR |
| --- | --- | --- | --- | --- |
| Source interface | SMPL/AMASS | BVH/SMPL/SMPL-X + key bodies | keypoints + object/terrain points | source surface point cloud |
| Correspondence | 拟合 shape 后的稀疏关键点 | 手工 key-body map | interaction-mesh vertices | 学习的 dense surface pairs |
| 主要约束 | 软几何拟合 + robot limits | 两阶段 IK + limits | 硬碰撞、脚粘地、速度/关节限位 | pose/normal/contact + constrained QP |
| Object/terrain | 不显式建模 | 不显式建模 | 原生支持 | contact map/scene 支持 |
| 在线速度 | 取决于 fitting/IK 与输入 | CPU 约 35–70 FPS benchmark | 偏离线数据生成 | setup 后约 65 FPS（论文数据） |
| 下游依赖 | teacher/student RL | BeyondMimic 等 tracker | proprioceptive RL | BeyondMimic/SONIC 或 tracker |

**最终判断。** OmniRetarget 本质上是一套 **把交互几何、硬约束和数据生成合并起来的 reference engine**。它真正的价值在于让下游 proprioceptive RL 不必用大量 reward regularizer 去修复脚滑和穿透；它的上限在于运动学 solver 仍需要正确场景几何与接触模式，动力学安全仍由 RL/controller 承担。对于 robot-object-terrain 数据生成和全身 loco-manipulation，值得优先精读与复现；对于纯实时 teleoperation，GMR 的 CPU pipeline 可能更轻量。
