---
title: "Unified Motion Retargeting for Humanoids with Learned Point Cloud Correspondence"
method_name: "UMR"
authors: [Hanyang Cao, Yuetong Fang, Taesoo Kwon, Runyi Yu, Ji Ma, Jing Tan, Yangchen Zhou, Baoze Du, Yi Gu, Yukang Gao, Ruoli Dai, Lei Han, Renjing Xu]
year: 2026
venue: arXiv
tags: [motion-retargeting, point-cloud-correspondence, skeleton-free, humanoid, surface-matching, interaction]
image_source: online
---

# UMR：用学习到的密集表面对应统一人形动作重定向

论文：[Unified Motion Retargeting for Humanoids with Learned Point Cloud Correspondence](https://arxiv.org/abs/2609.02134) · 项目页：[hanyang9.github.io/UMR](https://hanyang9.github.io/UMR/) · 官方代码：[hanyang9/UMR](https://github.com/hanyang9/UMR)

## 一、论文速览

### 一句话总结

UMR 不再为每个人体关节和机器人关节手工指定稀疏 correspondence，而是在 canonical T-pose 上学习有序的人体—机器人外部表面点对，再把这些点作为运动、法向和接触的几何锚，通过逐帧约束 Gauss–Newton QP 求解目标机器人姿态。

### Elevator pitch

大规模 human motion 数据包含丰富的走路、起身、交互与接触行为，但人体与机器人在骨长、拓扑、自由度和关节限位上不同。传统 retargeter 依赖 skeleton semantics：换 source representation 或 robot morphology，就需要重新设计 keypoint map。UMR 选择 surface-centric interface：将 source/robot canonical T-pose 采样成点云，PointNet-style encoder + MLP decoder 学习保持索引的 dense surface correspondence；运动时将 source surface 通过 barycentric transport 推进，将 robot surface 通过 FK 推进，再匹配位置、局部法向和 contact map。论文展示 SMPL-X、SOMA、角色、FBX/BVH 等输入与多种 humanoid 的统一处理，并在 LAFAN1、BONES-SEED、OmniContact、GRAIL/HOI 场景与 SONIC policy 中评测；但“无需手工 mapping”不等于无需 T-pose、mesh、MJCF、scale、adapter 和 scene preprocessing。

### 研究生态位

| 层 | UMR 的选择 |
| --- | --- |
| Source abstraction | exterior surface point cloud，而非 source skeleton |
| Learned component | canonical pose 的 dense correspondence |
| Retargeting solver | position + normal + contact residual 的 constrained Gauss–Newton QP |
| Interaction | object/scene contact map 可直接转移 |
| Reuse | 同一 source template × target robot 的 correspondence 可跨 motion 复用 |
| Downstream | BeyondMimic per-motion tracking、SONIC large-scale policy、real robot deployment |

## 二、Stage I：canonical 表面 correspondence

### 2.1 点云接口

对齐后的人体 T-pose 有有序 exterior point cloud：

$$
\mathbf X^h=\{\mathbf x_i^h\in\mathbb R^3\}_{i=1}^{N},
$$

机器人 T-pose 采样为无序点云：

$$
\mathbf X^r=\{\mathbf x_j^r\in\mathbb R^3\}_{j=1}^{N}.
$$

人体点的 ordering 提供输出索引，机器人点云提供目标外形。PointNet-style encoder $E_\theta$ 汇总机器人点云，MLP decoder $D_\theta$ 为每个 indexed human point 预测 deformation vector，输出与人体点共享索引的 robot surface location。这样 body segment labels、position/normal weights 和 contact indices 都可以从 source template 复用到不同 robot。

### 2.2 correspondence loss

论文将 correspondence learning 写成几何重建、点排斥和邻域平滑的组合：

$$
L_{corr}=L_c+\lambda_rL_r+\lambda_eL_e.
$$

$L_c$ 是 Chamfer/geometric matching，$L_r$ 防止多个 source 点塌缩到相同目标区域，$L_e$ 约束 edge/topological smoothness，使相邻人体点在机器人表面仍保持邻近。核心假设是：即使 skeleton topology 不同，外部表面的空间范围与局部几何仍能提供稳定的跨本体不变量。

学习只在 canonical T-pose 做一次，之后将 paired points 绑定到 source/robot mesh。它减少了每种 motion 重新找 mapping 的成本，但会把错误的 T-pose correspondence 固化到所有动作；手指、脚底、衣物/角色 mesh、遮挡和非人形 topology 仍是风险。

## 三、Stage II：correspondence-guided retargeting

### 3.1 姿态与方向残差

source surface 按 posed mesh 做 barycentric transport，robot points 按 link-local binding 经 FK 运动。每一帧估计 robot generalized coordinates $q_t$：

$$
\min_{q_t}\;\|r_p(q_t)\|_2^2+\|r_c(q_t)\|_2^2.
$$

对选定 correspondence point $i$，pose residual 同时匹配位置与 local surface orientation：

$$
r_{p,i}(q_t)=
\begin{bmatrix}
\sqrt{w_i^p}\bigl(x_i^r(q_t)-x_{t,i}^h\bigr)\\
\sqrt{w_i^n}\bigl(\bar n_i^r(q_t)-\bar n_{t,i}^h\bigr)
\end{bmatrix}.
$$

$w_i^p,w_i^n$ 可按 body segment 调整，使脚、手或躯干拥有不同重要性。相比只匹配 joint positions，normal residual 能约束身体表面的朝向；代价是它依赖 mesh 法向质量和局部 binding。

### 3.2 Contact map

每个 source point 到 object/scene surface 的最近点形成 contact vector：

$$
\pi_t(i)=\arg\min_j\|x_{t,i}^h-y_{t,j}\|_2,
\qquad
r_{c,i}(q_t)=\sqrt{w_i^c}\bigl(c_i^r(q_t)-c_{t,i}^h\bigr).
$$

因此 contact 不是独立人工指定的 hand/foot label，而是由 source point 与环境点的相对向量表达，再在目标机器人表面上追踪。它能够保留“接触哪个表面区域”的信息，但并不直接给出法向力、摩擦锥、内力分配或执行器裕度。

### 3.3 Constrained Gauss–Newton QP

由于 residual 依赖 FK，UMR 在线性化后用阻尼 Gauss–Newton 步求解：

$$
\begin{aligned}
\min_{\Delta q}\;&
\frac12\|r(q_t)+J(q_t)\Delta q\|_2^2
 +\frac\mu2\|\Delta q\|_2^2\\
\text{s.t. }&q^-\leq q_t+\Delta q\leq q^+,\\
&A_t\Delta q\leq b_t,\qquad \|\Delta q\|_2\leq\eta.
\end{aligned}
$$

$\mu$ 是 damping，$q^-/q^+$ 是 joint limits，$A_t\Delta q\leq b_t$ 可表达 floor clearance，$\eta$ 是 trust region。该求解器把 surface matching 的柔性目标与关节/地面等硬约束分开；如果 target surface 受机器人拓扑限制不可达，最终仍只能在约束内折中，而不是完美复制人体。

![UMR 的 surface correspondence 与跨本体重定向示意](https://raw.githubusercontent.com/hanyang9/UMR/main/teaser.png)

*图 1：官方仓库的 UMR teaser，展示人体表面、机器人表面和统一重定向接口。图片来自官方代码仓库。*

### 图表核对

在[论文 HTML](https://arxiv.org/html/2609.02134)中还核对了 Fig. 1 的 source/robot surface 与 contact map、Fig. 2 的两阶段 pipeline、Fig. 3 的真机 spin-kick/ball/stair deployment、Fig. 4 的 contact-map 定义、Fig. 5 的多 source/多 robot 统一结果、Fig. 6 的 BONES-SEED 与 GMR reference 对比、Fig. 7 的 SONIC training curves 和 Fig. 8 的 UMR/OmniRetarget interaction 对比。Table I 给出 Stage I/II 计算成本，Tables II–IV 分别覆盖 LAFAN1 tracking、SONIC 与 contact-rich task。这样可区分“表面 correspondence 的几何收益”和“下游 policy 训练/termination 的收益”。

## 四、Source adapter 与“统一性”边界

官方代码列出 BONES-SEED/SOMA、GRAIL、OmniContact、LAFAN1/SMPL-X、OMOMO、Humanoid Character、AdaPT 和 NR FBX/BVH 等输入。统一接口要求每个 source 最终提供 canonical template 与 posed surface sequence；因此：

| 输入来源 | 仍需的适配工作 |
| --- | --- |
| SMPL-X/LAFAN1 | body model、shape/gender、BVH→SMPL-X 转换 |
| SOMA/BONES-SEED | SOMA template 与 source-specific 点云 |
| GRAIL/4D motion | 视频恢复误差、场景和人/物坐标对齐 |
| OMOMO/OmniContact | object/scene 点云、contact frame、数据布局 |
| FBX/BVH/角色 | surface extraction、canonical T-pose、时间与单位转换 |
| 新机器人 | MJCF、mesh、T-pose、scale、joint limits、collision geometry |

仓库明确说明：OmniContact 原始 BVH 到 SMPL-X 的内部 converter 未公开，因此当前 release 不能直接吃这些 BVH；LAFAN1 需要先转成 SMPL-X。UMR 消除的是手工 human–robot sparse body map，不是所有 data engineering。

## 五、实验与证据

### 5.1 计算效率

在 LAFAN1 上，Stage I 的点云采样 9.83 s、geodesic precomputation 5.58 s、correspondence training 10.38 s，总 setup 25.79 s；Stage II preprocessing 141.46 FPS、motion retargeting 121.26 FPS、overall throughput 65.29 FPS。这个速度说明 correspondence reuse 有价值，但不包含人体 pose recovery、scene reconstruction、数据下载与真机通信。

### 5.2 LAFAN1 tracking

UMR 与 GMR、Unitree reference 接入 BeyondMimic，在无 DR、有 DR 和 sim2sim 三种设定、每项 4096 trials 上比较。代表性的 sim2sim success 为：

| Motion | UMR | GMR | Unitree |
| --- | ---: | ---: | ---: |
| Dance | 96.793% | 89.682% | 96.100% |
| Fall and GetUp | 34.920% | 32.096% | 46.899% |
| Fight | 95.298% | 79.492% | 95.913% |
| Jump | 92.863% | 89.266% | 91.528% |
| Sprint | 86.426% | 71.460% | 90.833% |

这些结果说明 UMR 通常优于 GMR，尤其在动态/接触敏感 motion 上，但 Unitree reference 在部分动作仍更强；Fall and GetUp 的 sim2sim 低成功率也提醒我们，运动难度和跨 simulator 差异仍是主导因素。论文同时报告 global/root-relative body position 与 joint-angle errors，不能只看 success rate。

### 5.3 大规模 SONIC 与 interaction

在 BONES-SEED/SONIC 上，UMR references 在有/无 SMPL encoder 两种设置中都被训练；没有 SMPL encoder 时，UMR 相比公开 Unitree references 在 total reward、anchor-position error 和 mean joint-angle error 上约有 10% 相对改善，说明提升不完全来自额外的 human-motion latent guidance。

在 Carry、Kick、Push 等 robot-object 任务中，UMR 相对 OmniRetarget 将 joint error 降低约 40%–56%；二者都可能移动物体，但 UMR 更准确地保留 bimanual contact geometry。GRAIL scene interaction 的 Stair、Slope、Chair 结果并非全胜：UMR 在 Stair/Slope 更强，OmniRetarget 在 Chair 略有优势。这比“统一方法全面优于所有 baseline”更可信，也暴露了 source distribution 与 stance heuristic 的影响。

## 六、代码与复现审查

官方仓库提供 Python 3.12、PyTorch 2.4.1 + CUDA 12.1 的安装路径、robot config、sample data、correspondence cache/训练流程、MuJoCo viewer 和多个 adapter guide。SMPL-X 模型不随仓库分发，需要按许可下载；新机器人需要在 UMR Studio 中调整 MJCF 到 T-pose 并复制 `tpose_qpos` 到 config。

最小 LAFAN1 运行命令为：

```bash
python scripts/humanoid_retarget_pipeline.py \
  --config robot_configs/humanoid_retarget_unitree_g1_example.json
```

| 复现层级 | 目标 | 主要风险 | 成功标准 |
| --- | --- | --- | --- |
| Level 1 | 仓库样例 correspondence + MuJoCo 回放 | SMPL-X 许可、CUDA、T-pose config | LAFAN1-derived motion 能生成并播放 |
| Level 2 | GMR/Unitree 对比与接触指标 | source adapter、metric/termination 对齐 | 同时报告 throughput、pose、contact 与 success |
| Level 3 | 新机器人/新 source/HOI | mesh sampling、canonical pose、scene frame、solver | correspondence 可复用且失败 case 可定位 |

代码适合做可运行的 surface-retargeting baseline，但不能省略 source-specific preprocessing；从公开 BVH 到 SMPL-X 的缺失 converter 可能成为真实复现阻塞点。

### 代码质量评分

| 审查项 | 评分 | 证据与判断 |
| --- | --- | --- |
| README/安装文档 | 高 | 给出 Python 3.12、PyTorch/CUDA、quick start、robot config 与 adapter guide |
| 环境可复现性 | 中 | 依赖较新且明确，但 SMPL-X 许可、CUDA 和 MuJoCo 仍需本地配置 |
| correspondence/solver 代码 | 高 | Stage I/II、pipeline、cache 和约束优化入口公开 |
| source adapter | 中 | 覆盖面广，但不同 source 的 layout/converter 要逐一准备 |
| robot/scene 资产 | 中 | MJCF、mesh、T-pose、joint limits 与 object/scene 点云需要用户核验 |
| checkpoint/评估 | 中 | 有样例与 viewer，论文级 SONIC/HOI 训练并非完全一键交付 |
| 真机相关部分 | 中 | 展示部署结果，但硬件控制、motion capture 和 policy stack 仍是外部系统 |
| 维护/许可证 | 中 | 官方仓库结构清晰，但 2026 新项目的长期稳定性与更多形态仍待观察 |

## 七、批判性思考

### 优点

1. **表示层的统一性明确**：表面点云绕开了不同 skeleton topology，允许 SMPL-X、角色和 SOMA 进入相同 solver。
2. **contact 传递自然**：最近环境点与 contact vector 与 surface correspondence 共享索引，不必再给每种 robot 手写接触 joint map。
3. **一次学习、多 motion 复用**：canonical correspondence setup 成本低，per-clip throughput 足够离线大规模数据生成。
4. **评测覆盖跨度大**：从 LAFAN1 动态动作到 SONIC、robot-object、terrain scene 和真机部署，能检查 reference 的下游价值。

### 局限性

1. dense correspondence 的质量依赖 canonical T-pose、mesh sampling、geodesic/topology loss；错误 mapping 会被所有 motion 复用。
2. surface position/normal/contact matching 仍是几何目标，不直接约束摩擦、接触力、闭链、柔性物体或 actuator limits。
3. 跨 source 的“统一”仍需要不同 adapter、人体模型、scene/object 点云和许可证；对无可靠 surface 的输入不能自动解决。
4. interaction comparison 的成功率受不同 downstream termination criterion 影响；更准确的 reference 可能需要更严格的 tracking metric 才能显现价值。
5. UMR 是 2026 年新工作，长期维护、更多机器人真机、极端 OOD source 和在线低延迟表现仍需观察。

### 潜在改进

- 对 correspondence 输出增加 uncertainty/confidence，并在低置信区域退回 GMR/IK 或拒绝生成；
- 将接触法向、摩擦锥、质心/力矩和闭链约束加入 QP，形成几何—动力学混合 retargeter；
- 公开 BVH/FBX→surface adapter 与完整 evaluation assets，使“unified”能够被第三方真正验证；
- 研究 correspondence 对身体比例、衣物、手指和非 humanoid morphology 的 scaling law；
- 统一 UMR、GMR、OmniRetarget 的 reference、policy、termination、数据量和 simulator，避免评测协议差异。

## 八、相关工作与最终判断

[GMR](https://arxiv.org/abs/2510.02252) 是 UMR 的 skeleton-centric baseline，强项是多机器人 CPU 实时 IK，但仍依赖手工 key-body map；[OmniRetarget](https://arxiv.org/abs/2509.26633) 用 interaction mesh 和硬约束保留物体/地形关系，但其 correspondence 与 stance 仍有显式设计；[PHC/OmniH2O](https://arxiv.org/abs/2406.08858) 更适合作为 SMPL shape fitting + teacher–student control 的系统 baseline。UMR 的创新在 representation/interface，而不是把几何 surface matching 误称为动力学控制。

| 维度 | PHC | GMR | OmniRetarget | UMR |
| --- | --- | --- | --- | --- |
| Source interface | SMPL/AMASS | BVH/SMPL/SMPL-X + key bodies | keypoints + object/terrain points | source surface point cloud |
| Correspondence | 拟合 shape 后的稀疏关键点 | 手工 key-body map | interaction-mesh vertices | 学习的 dense surface pairs |
| 主要约束 | 软几何拟合 + robot limits | 两阶段 IK + limits | 硬碰撞、脚粘地、速度/关节限位 | pose/normal/contact + constrained QP |
| Object/terrain | 不显式建模 | 不显式建模 | 原生支持 | contact map/scene 支持 |
| 在线速度 | 取决于 fitting/IK 与输入 | CPU 约 35–70 FPS benchmark | 偏离线数据生成 | setup 后约 65 FPS（论文数据） |
| 下游依赖 | teacher/student RL | BeyondMimic 等 tracker | proprioceptive RL | BeyondMimic/SONIC 或 tracker |

**最终判断。** UMR 本质上是一篇 **学习 canonical surface correspondence、再做约束运动优化的跨本体 retargeting 工作**。它真正的价值在于减少 source skeleton 与 target robot topology 之间的手工绑定，并把接触关系纳入同一 surface interface；它的上限在于 canonical geometry、source adapter 和动力学缺口。对需要异构人体来源、多个 humanoid 和 interaction reference 的团队，UMR 值得重点精读并作为新一代 baseline；对只需低延迟、无对象的在线 teleoperation，GMR 可能更容易直接部署。
