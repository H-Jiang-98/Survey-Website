---
title: "Retargeting Matters: General Motion Retargeting for Humanoid Motion Tracking"
method_name: "GMR"
authors: [João Pedro Araújo, Yanjie Ze, Pei Xu, Jiajun Wu, C. Karen Liu]
year: 2025
venue: arXiv
tags: [motion-retargeting, humanoid-motion-tracking, inverse-kinematics, real-time, CPU, benchmark]
image_source: online
---

# GMR：重定向质量是人形动作跟踪的上游瓶颈

论文：[Retargeting Matters: General Motion Retargeting for Humanoid Motion Tracking](https://arxiv.org/abs/2510.02252) · 项目页：[retargeting_matters](https://jaraujo98.github.io/retargeting_matters/) · 代码：[YanjieZe/GMR](https://github.com/YanjieZe/GMR)

## 一、论文速览

### 一句话总结

GMR 是一个面向多种人形机器人的优化式动作重定向器；论文把 PHC、ProtoMotions、GMR 和 Unitree 闭源参考接入同一个 BeyondMimic tracking pipeline，发现 foot/ground artifact、self-intersection 和关节突跳会显著降低动态或长序列 motion policy 的可靠性。

### Elevator pitch

人类动作和人形机器人在骨长、关节范围、拓扑、质量和执行器上不同，重定向是从人类数据训练机器人前必须跨过的 embodiment gap。以往工作往往让 RL policy 自己修复错误 reference，因此很难判断性能到底来自 retargeter 还是 reward engineering。GMR 先让用户定义 source/robot key-body mapping 与 rest-pose offsets，再进行缩放和两阶段位置/旋转 IK；序列处理时以上一帧结果 warm start，并用最低高度修正地面漂浮/穿透。论文在 21 条 LAFAN1 motion 上控制 policy 与数据来源，报告 GMR 的误差和感知保真度明显优于公开 PHC/ProtoMotions，接近 Unitree reference；但 GMR 仍可能出现罕见的 waist jump，且新机器人仍需配置 correspondence 和权重。

### 关键判断

- **研究问题**：重定向方法是否会改变同一个 tracking policy 的成功率与鲁棒性？
- **主要方法**：基于 Mink/MuJoCo 的两阶段 constrained IK，而不是端到端 learned retargeter。
- **证据优势**：用独立的 BeyondMimic pipeline，避免为每种 retargeter 调 reward。
- **证据边界**：核心 benchmark 不包含复杂 object/terrain interaction；“general”主要是机器人和输入格式的适配广度，不是无需配置的零样本泛化。

## 二、Source → Target 定义与评测控制

| 维度 | Source | Target | GMR 的适配动作 |
| --- | --- | --- | --- |
| 动作格式 | BVH、SMPL/SMPL-X、MoCap、LAFAN1、OMOMO | XML/URDF 描述的机器人 skeleton | 用户提供 body list 与 mapping |
| 几何尺度 | 人体身高/骨长 | 机器人 link 长度与 base | root/各 body scaling |
| 姿态 | body global position + rotation | robot base + joint position | FK + IK |
| 时间 | 单帧或序列 | 与 policy reference 对齐 | 逐帧 warm start、可 rate-limit |
| 约束 | 人体动作没有机器人限位 | q limit、velocity limit、初始 pose | 优化约束与后处理 |
| policy | 同一个 BeyondMimic | 同一训练配置 | 尽量隔离 retargeter 影响 |

论文回答的不是“GMR 能否单独让机器人走路”，而是“给定相同的下游 policy 学习器，哪种 reference 更容易被学到并保持源动作外观”。这种实验设计让 retargeting 质量成为主要变量，但仍无法完全消除机器人资产、权重调节和 Unitree 数据制作过程的差异。

## 三、GMR 方法：从 key-body mapping 到两阶段 IK

### 3.1 身体对应与 rest-pose 对齐

用户为 torso、head、腿、脚、臂、手等 key bodies 定义集合 $\mathcal{M}$，并可分别指定 position/orientation error weights。接着把人体在 source rest pose 的 body orientation offset 到机器人 rest pose，必要时加局部位置 offset，以减轻脚尖内收等初始姿态 artifact。这个步骤仍是语义手工配置，不是完全自动的 correspondence learning。

### 3.2 分层尺度

GMR 不把所有点简单按一个比例缩放；它可以为 root、下肢、上肢等 body 使用不同 scaling。论文给出的 target body position 为：

$$
\mathbf{p}_{b}^{target}=\frac{h}{h_{ref}}s_b(\mathbf{p}_{j}^{source}-\mathbf{p}_{root}^{source})
 +\frac{h}{h_{ref}}s_{root}\mathbf{p}_{root}^{source}.
$$

其中 $h$ 是 source skeleton height，$h_{ref}$ 是设置 scale 时的参考身高，$s_b$ 是 body-specific scale。论文特别指出 root translation 也要统一缩放，否则脚会相对地面滑动。这个设计比“只缩上肢/不缩 base”更物理，但仍是几何 heuristic，不会自动知道机器人脚底与人体脚掌的接触差异。

### 3.3 两阶段姿态优化

第一阶段优先优化匹配 body orientation 与 end-effector position：

$$
\begin{aligned}
\min_{\mathbf q}\;&
\sum_{(i,j)\in\mathcal M}(w_1)^R_{i,j}
\|R_i^h\ominus R_j(\mathbf q)\|_2^2\\
&+\sum_{(i,j)\in\mathcal M_{ee}}(w_1)^p_{i,j}
\|\mathbf p_i^{target}-\mathbf p_j(\mathbf q)\|_2^2\\
\text{s.t. }&\mathbf q^-\leq\mathbf q\leq\mathbf q^+.
\end{aligned}
$$

这里 $R_i^h$ 是人体 body orientation，$R_j(\mathbf q)$ 和 $\mathbf p_j(\mathbf q)$ 由 robot FK 得到，$\ominus$ 是 $SO(3)$ 的 exponential-map rotation difference，$\mathcal M_{ee}$ 只包含手脚等末端。

第二阶段以第一阶段结果为初值，把所有 key bodies 的位置和旋转纳入优化：

$$
\begin{aligned}
\min_{\mathbf q}\;&
\sum_{(i,j)\in\mathcal M}(w_2)^R_{i,j}\|R_i^h\ominus R_j(\mathbf q)\|_2^2
 + (w_2)^p_{i,j}\|\mathbf p_i^{target}-\mathbf p_j(\mathbf q)\|_2^2\\
\text{s.t. }&\mathbf q^-\leq\mathbf q\leq\mathbf q^+.
\end{aligned}
$$

求解直到 value change 小于 0.001 或最多 10 次迭代。motion sequence 按帧处理，上一帧 result 作下一帧初值；完成后再对所有 body height 做最小值归零，修复整体浮空或穿地。该后处理能修正全局高度，却不能修复局部自穿透或速度突跳。

![GMR 的 General Motion Retargeting pipeline](https://arxiv.org/html/2510.02252/x1.png)

*图 1：GMR 的 pipeline 图；图片来自论文 HTML。*

### 图表核对

在[论文 HTML](https://arxiv.org/html/2510.02252)中还核对了 Figure 2 的完整 GMR 流程、Figure 3 的 ground penetration/self-intersection/waist-jump 失败案例和 Figure 4 的 $N=20$ 用户研究；Table I 是 sim、sim+DR、sim2sim 的逐动作 success，Table II 是三类 tracking error，Table III 是起始 reference frame 消融。Figure 3 的 artifact 案例是“retargeting 影响 policy”的关键证据，不能用平均 success 取代。

## 四、动作表示、实时性与可部署性

代码把 human frame 表示为 `(body_name, global translation, global rotation)` 字典，把 robot frame 表示为 `(base translation, base rotation, joint positions)`。仓库提供 MuJoCo 可视化、SMPL-X/AMASS/OMOMO/LAFAN1 转换、批量处理和 OptiTrack/PICO streaming。README 报告在 AMD Threadripper 7960X 上约 60–70 FPS、13 代 i9-13900K 上约 35–45 FPS；这是 retargeting/solver benchmark，不应直接当成完整人体捕获到电机的端到端延迟。

GMR 的实时性来自：

1. 预先配置 mapping、scale 与 robot model；
2. 逐帧 warm start；
3. 使用 CPU IK/Mink/MuJoCo 而非大模型 inference；
4. 可选 `rate_limit` 限制输出速度以保持人类动作速度。

代价是新机器人仍需 XML/URDF、body/joint name、rest pose、limits、mesh 和权重；不同人体、极端动作可能需要重新调参。仓库自身记录了“单一 config 覆盖所有 humans 并不容易”，这正是 general retargeting 的实际边界。

## 五、实验：retargeter 是否改变 policy 结果？

### 5.1 评测设置

作者在 21 条多样 LAFAN1 motion 上比较 PHC、ProtoMotions（PM）、GMR 和 Unitree 闭源 reference。所有参考动作进入同一 BeyondMimic policy training，不使用针对 retargeter 的 reward tuning 或大规模 domain randomization。指标包括 sim、sim+DR、MuJoCo/ROS sim2sim success，global/body-relative position error、joint rotation error，以及 20 人用户研究中的 source-motion faithfulness。

### 5.2 tracking 指标

| 统计量 | PHC | GMR | ProtoMotions | Unitree |
| --- | ---: | ---: | ---: | ---: |
| Mean global body position error (mm) | 247.8 | 104.1 | 139.7 | 77.2 |
| Mean root-relative body position error (mm) | 40.2 | 28.1 | 33.2 | 23.2 |
| Mean joint rotation error ($10^{-3}$ rad) | 778.5 | 561.7 | 641.8 | 483.0 |

GMR 明显优于 PHC 和 ProtoMotions，但 Unitree reference 仍是最强。success 的差异集中在少数难动作：11/21 条动作在所有方法上都超过 98%，3 条动作全部完美；真正拉开差距的是 dynamic、long 或 artifact-prone motion，而不是普通走路。

### 5.3 Artifact 与用户研究

论文给出三个可追溯失败例：PHC Dance 1/2 出现最大约 60 cm ground penetration；ProtoMotions 的 Run (stop & go) 有腿部 self-intersection；GMR Dance 5 有 waist roll/pitch 突跳。GMR 的突跳发生在约 10 秒片段，占全数据不足 2%，作者认为统一权重对某些 motion 不够理想。

20 名用户比较 5 秒片段：GMR 比 PHC/ProtoMotions 更接近源动作，但 Unitree 比 GMR 更常被认为 faithful，且用户更难区分二者。这个结果支持“GMR 是闭源高质量 reference 的可行开源替代”，不等于它消除了所有视觉或物理 artifact。

## 六、代码与复现计划

官方仓库使用 Ubuntu 20.04/22.04、Conda Python 3.10 和 `pip install -e .`；SMPL-X body model、AMASS/LAFAN1/OMOMO 数据与若干机器人资产需另行下载。SMPL-X pkl 用户还需按 README 修改 `smplx/body_models.py` 的扩展名处理。最小命令是：

```bash
python scripts/smplx_to_robot.py \
  --smplx_file <input> --robot <robot_config> \
  --save_path <output.pkl> --rate_limit
```

| 复现层级 | 目标 | 主要风险 | 成功标准 |
| --- | --- | --- | --- |
| Level 1 | 单条 SMPL-X/LAFAN1 → G1 可视化 | 模型许可、body names、格式转换 | MuJoCo 播放无崩溃且输出 pose |
| Level 2 | 21 条 motion 的误差/成功率 | BeyondMimic 版本、初始化与 metric 实现 | 同时报告 global/root-relative/rotation error |
| Level 3 | 新机器人或在线 OptiTrack | mapping、rest pose、SDK、通信延迟 | CPU throughput、长序列和 OOD motion 都有失败记录 |

代码完整度适合做 baseline，但不应把 README 的 CPU FPS 当作真机安全保证。它包含机器人配置和 streaming 入口，却仍依赖外部 body model、捕获设备、机器人资产和下游 tracking policy。

### 代码质量评分

| 审查项 | 评分 | 证据与判断 |
| --- | --- | --- |
| README/安装文档 | 高 | 给出 Python 3.10、Ubuntu、SMPL-X/AMASS/LAFAN1/OMOMO 准备和运行命令 |
| 环境可复现性 | 中 | 主要依赖可安装，但 SMPL-X 扩展名、MuJoCo/Mink 和渲染库仍可能需要手工修正 |
| retargeting 核心 | 高 | `GeneralMotionRetargeting`、两阶段求解、机器人 config 和 batch script 均公开 |
| 机器人覆盖 | 高 | README 列出多种 humanoid/robot 配置，但每个仍需检查资产与限位质量 |
| 数据与模型资产 | 中 | AMASS、SMPL-X、OMOMO、LAFAN1 需用户下载并按许可处理 |
| checkpoint/训练入口 | 中 | GMR 本身是 retargeter，policy checkpoint 与 BeyondMimic 训练并非同一仓库交付 |
| 在线 streaming | 中 | OptiTrack/PICO demo 存在，但 SDK、双机网络和真实电机闭环不完全由 GMR 管理 |
| 维护/许可证 | 高 | 仓库有速度 benchmark、known issues、MIT 许可和较完整的机器人列表 |

## 七、批判性思考

### 优点

1. **实验隔离做得好**：统一 BeyondMimic 与不调 reward 的设置，把上游 retargeting 的影响单独暴露出来。
2. **优化流程透明**：mapping、rest-pose、分层 scale、两阶段 IK、warm start 和后处理都能解释和实现。
3. **既看几何也看感知**：误差指标、policy success、artifact 案例和用户研究互相补充。

### 局限性

1. 关键 body mapping、权重与 robot config 仍由用户设计；因此“通用”不是零配置跨形态。
2. 主要 benchmark 不包含 object/terrain interaction；脚滑、穿透和 hand-object geometry 可能比当前结果更难。
3. GMR 自己仍有少量 waist jump，统一权重在所有动作上并非最优；论文没有给出自动调权或不确定性拒绝策略。
4. policy success 很多动作接近饱和，均值差异容易被少数长/动态动作主导；sim2sim 也不能替代真机接触评测。

### 潜在改进

- 在 GMR 的两阶段 IK 中加入显式 floor、stance-foot、self-collision 与 object surface constraints；
- 学习 body correspondence/weights，保留 GMR CPU solver 的实时性；
- 对每条序列报告 feasibility、velocity jump、penetration 和 contact preservation，而不仅是 policy success；
- 用 GMR 作为统一 skeleton baseline，与 OmniRetarget interaction mesh、UMR surface correspondence 做同一数据/指标比较。

## 八、相关工作与最终判断

[PHC/OmniH2O](https://arxiv.org/abs/2406.08858) 采用 SMPL shape fitting 与 keypoint retargeting，并把 reference 接入 teacher–student RL；[OmniRetarget](https://arxiv.org/abs/2509.26633) 将物体/地形点加入 interaction mesh，以硬约束处理接触；[UMR](https://arxiv.org/abs/2609.02134) 学习 canonical surface correspondence，减少手工 skeleton mapping。GMR 的位置是“强可解释、可实时、多机器人 skeleton IK baseline”，而不是 interaction-aware 或 learned correspondence 的最终方案。

| 维度 | PHC | GMR | OmniRetarget | UMR |
| --- | --- | --- | --- | --- |
| Source interface | SMPL/AMASS | BVH/SMPL/SMPL-X + key bodies | keypoints + object/terrain points | source surface point cloud |
| Correspondence | 拟合 shape 后的稀疏关键点 | 手工 key-body map + rest pose offsets | interaction-mesh vertices | 学习的 dense surface pairs |
| 主要约束 | 软几何拟合 + robot limits | 两阶段 IK + limits | 硬碰撞、脚粘地、速度/关节限位 | pose/normal/contact + constrained QP |
| Object/terrain | 不显式建模 | 不显式建模 | 原生支持 | contact map/scene 支持 |
| 在线速度 | 取决于 fitting/IK 与输入 | CPU 约 35–70 FPS benchmark | 偏离线数据生成 | setup 后约 65 FPS（论文数据） |
| 下游依赖 | teacher/student RL | BeyondMimic 等 tracker | proprioceptive RL | BeyondMimic/SONIC 或 tracker |

**最终判断。** GMR 本质上是一篇把 retargeting 从“训练前的无名预处理”提升为可测量研究变量的工作。它真正的价值在于：给出可复现的多机器人优化 pipeline，并证明 reference artifact 会改变下游 policy 的学习难度；它的上限在于手工 correspondence、无接触对象建模和罕见优化突跳。若团队需要一个 CPU 可运行、能统一比较不同机器人参考的 baseline，GMR 值得优先复现；若任务是人—物—地形交互，应把它视为几何 baseline，而不是完整解决方案。
