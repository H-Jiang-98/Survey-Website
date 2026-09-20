---
title: "OmniH2O: Universal and Dexterous Human-to-Humanoid Whole-Body Teleoperation and Learning"
method_name: "PHC"
authors: [Tairan He, Zhengyi Luo, Xialin He, Wenli Xiao, Chong Zhang, Weinan Zhang, Kris Kitani, Changliu Liu, Guanya Shi]
year: 2024
venue: arXiv
tags: [motion-retargeting, human-to-humanoid, whole-body-teleoperation, SMPL, reinforcement-learning, sim-to-real]
image_source: online
---

# PHC / OmniH2O：用运动学姿态连接人类输入与全身人形控制

论文：[OmniH2O](https://arxiv.org/abs/2406.08858) · 项目页：[omni.human2humanoid.com](https://omni.human2humanoid.com/) · 代码：[LeCAR-Lab/human2humanoid](https://github.com/LeCAR-Lab/human2humanoid)

## 一、论文速览

### 一句话总结

OmniH2O 把人体运动重定向成目标人形的运动学参考，再用“特权 teacher RL → 稀疏观测 student/DAgger”的链路训练一个能由 VR、RGB、语言动作生成器或遥操作数据驱动的全身控制策略；PHC 是其中的 SMPL/关键点重定向与运动数据准备路线，而不是完整的控制器。

### Elevator pitch

人形机器人需要同时支撑下肢平衡、上肢操作和灵巧手动作，直接把人体动作复制到机器人会受到骨长、关节拓扑、自由度和传感器差异的限制。OmniH2O 选择运动学 pose 作为统一接口：先把 AMASS/SMPL 人体运动变为机器人参考，再在仿真中训练使用特权状态的 PPO teacher，最后蒸馏到只依赖关节状态、根部角速度、重力和 25 步历史的部署策略。论文报告 14k 条仿真参考序列、Unitree H1 真机实验、VR/RGB 控制和 OmniH2O-6 示范数据集；但这些结果是整个系统的结果，不能全部归因于 PHC retargeter。

### 研究生态位

- **重定向**：面向 SMPL/AMASS 的 model-fitting + keypoint retargeting。
- **控制**：端到端 whole-body motion imitation，而非仅下肢 locomotion。
- **迁移**：特权仿真状态到稀疏真机状态的 teacher–student sim-to-real。
- **接口**：运动学 pose 是 VR、RGB、语言和自主策略共用的中间表示。

最重要的判断是：PHC 解决的是“人体参考如何变成一个与机器人形态相容的几何目标”；RL policy 再负责把这个目标变成有动力学约束的 joint target/torque 行为。

## 二、问题定义与系统边界

源域是人体运动（主要是 AMASS/SMPL 形式），目标域是 19 个受控自由度的 Unitree H1 全身系统。论文形式化的 goal-conditioned MDP 为：

$$
\mathcal{M}=\langle\mathcal{S},\mathcal{A},\mathcal{T},\mathcal{R},\gamma\rangle,
\qquad
\max_{\pi}\;\mathbb{E}\left[\sum_{t=1}^{T}\gamma^{t-1}r_t\right].
$$

状态由 proprioception 和 goal state 构成，动作是关节角 target，由 PD 控制器执行。运动学 pose 写作：

$$
q_t=(\theta_t,p_t),
\qquad
\dot q_t=(\omega_t,v_t),
$$

其中 $\theta_t,p_t$ 分别是各关节的 3D 旋转和位置，$\omega_t,v_t$ 是角速度和线速度。这个表示适合跨输入模态，但它也把“人体想做什么”与“机器人是否有足够的力/接触能力”分开了。

### PHC 重定向与 OmniH2O 控制的边界

| 环节 | 输入 | 输出 | 主要保证 | 不保证 |
| --- | --- | --- | --- | --- |
| SMPL/PHC fitting | SMPL 模型、机器人 FK、canonical pose | 匹配机器人比例的 shape 与缩放 | 几何尺度大致一致 | 动力学可行、接触力 |
| 关键点 retargeting | 人体 pose、source–robot keypoint 对应 | 机器人参考 pose/轨迹 | 末端与身体关键点的几何跟随 | 无穿透、无脚滑的硬保证 |
| Teacher PPO | 特权机器人状态、完整 reference | joint target policy | 仿真中的稳定跟踪 | 真机可观测性 |
| Student/DAgger | 稀疏状态、25 步历史、3 点 motion goal | joint target/torque 接口 | 真机输入可用、延迟鲁棒 | OOD 目标的安全保证 |

## 三、PHC/OmniH2O 方法

### 3.1 SMPL 形状拟合与人体—机器人对应

PHC 先用机器人 FK 和 SMPL 模型在 canonical pose 下拟合整体缩放与 SMPL shape $\beta$。可写成：

$$
\min_{\alpha,\beta}\sum_i\left\|f_i(q^{robot}_0)-\alpha M_i(q^{smpl}_0;\beta)\right\|_2^2,
$$

其中 $f_i$ 是机器人关键点的 FK 位置，$M_i$ 是 SMPL 关键点位置，$\alpha$ 是整体尺度。优化后，人体运动序列的 source keypoint 通常由：

$$
p^{source}_{t,i}=\alpha M_i(q^{demo}_t;\beta)
$$

生成，再通过机器人关键点对应和 IK 得到目标 reference。这样做的优点是先把“人体骨长与机器人骨长不一致”变成一个固定 shape-fitting 问题；缺点是单一 SMPL 形状无法表达机器人手部、刚性脚底、关节限位和实际碰撞几何。

### 3.2 参考动作增强

论文发现仅使用自然人体运动时，机器人会用小步调整来保持平衡。于是对 motion dataset 加入 lower-body 固定的 stable standing / squatting 变体：固定 root 与下肢，使上身有动作时机器人仍有稳定支撑。该步骤不是纯 retargeting，而是改变 reference motion 分布。它解释了为什么“PHC 轨迹质量”与“OmniH2O policy 的站立能力”不能混为一谈。

### 3.3 Teacher–student 控制接口

Teacher 的特权 proprioception 包含全身刚体位置、方向、速度、角速度和上一动作；goal 包含下一帧参考姿态、当前误差与参考速度。Student 则只保留易在真机取得的关节位置/速度、root angular velocity、gravity、previous action 和 motion goal，并堆叠 25 步历史：

$$
\mathcal{L}_{DAgger}=\left\|a_t^{privileged}-a_t\right\|_2^2.
$$

历史序列隐式恢复速度与延迟信息，使系统不依赖显式 global linear velocity。部署策略的单步观测为 90 维，25 步历史后总状态维度为 1665；输出为 19 维动作。灵巧手则通过 VR 手部姿态和 IK 直接生成手部 joint target，使用现成的低层手控制器。

### 3.4 Reward、随机化与实时部署

PPO teacher 同时使用 imitation reward 与 standing/walking regularization。论文特别强调 max-feet-height reward 及其 curriculum，避免机器人通过跺脚维持平衡。真机随机化包括摩擦 $\mathcal{U}(0.2,1.1)$、质量、PD 增益、20–60 ms 控制延迟、torque random force injection 与外部推力。真机平台是 Unitree H1，使用 Damiao 电机、Inspire hands、两个 16 GB Orin NX 和 60 Hz 的 ZED 相机；这些是系统部署成本，不是 PHC 算法本身的必要条件。

![OmniH2O 系统总览：人体/多模态输入经运动学接口驱动人形机器人](https://arxiv.org/html/2406.08858v1/x1.png)

*图 1：论文展示的全身遥操作与自主控制系统。图片来自论文 HTML。*

### 图表核对

论文还提供以下图表，均在[论文 HTML](https://arxiv.org/html/2406.08858)中核对：Figure 2 对比 source、retargeted、standing 和 squatting motion；Figure 3 展示 retargeting、teacher–student 和多输入接口；Figure 4 展示语言生成 motion goal；Figure 5 展示推搡与室外地形鲁棒性；Figure 6 展示 GPT-4o 选择 motion primitive；Figure 7 展示 OmniH2O-6 的 imitation-learning 任务；Appendix Figures 8–10 分别对应线速度估计、数据分布消融和额外遥操作；Figure 11 展示六项数据集任务。Table 1/2 汇报 motion tracking 与真机线速度/历史架构消融，Table 3 汇报 LfD 数据量、时序输入和 BC/DDIM/DDPM，Table 18 给出 RL/DAgger 超参数。正文中的结论只使用这些图表能直接支持的范围。

## 四、实验与证据

### 4.1 仿真跟踪

仿真使用约 14k 条 retargeted AMASS/augmented sequences；Success 定义为任一时刻平均偏离 reference 超过 0.5 m 即失败，同时报告 global MPJPE、root-relative MPJPE、加速度与速度误差。核心结果如下：

| 方法 | Sim2Real | Success | Global MPJPE (mm) | Root-relative MPJPE (mm) |
| --- | ---: | ---: | ---: | ---: |
| Privileged teacher | 否 | 94.77% | 126.51 | 70.68 |
| H2O baseline | 是 | 87.52% | 148.13 | 81.06 |
| OmniH2O student | 是 | 94.10% | 141.11 | 77.82 |

结果证明 teacher–student 与历史状态可以让稀疏输入策略接近特权 teacher；它不单独证明 PHC 比其他 retargeter 更好。DAgger 去掉后 Success 降到 47.11%，说明下游训练链路对最终性能有决定性影响。

### 4.2 真机、接口与数据

论文在真机上测试 20 条 standing sequences，并展示 VR、RGB、语言生成 motion goal、GPT-4o motion primitive 和六类示范任务（Catch-Release、Squat、Rock-Paper-Scissors、Hammer-Catch、Boxing、Basket-Pick-Place）。OmniH2O-6 的数据以 30 Hz 记录 RGBD、头手相对 root 的 motion goal 和 motor joint targets，总示范约 40 分钟。它证明了统一接口可以复用于 teleoperation 与 LfD；但真机大规模 motion tracking 的覆盖仍远小于仿真集。

### 4.3 消融与关键结论

- 25 步历史比 0/5/50 步和 LSTM/GRU 组合更适合真机；这支持“历史替代显式线速度”的工程判断。
- 8 点、22 点 tracking 目标均可工作，说明接口不必总输入完整人体姿态，但点数会改变歧义和上身精度。
- 去掉 stable motion augmentation 会削弱静止站立和上身动作；这是一项数据分布设计，而非 retargeting solver 的独立改进。
- 论文没有对 PHC 的几何误差、脚滑、穿透和与 GMR/OmniRetarget 的同 pipeline 指标做系统隔离；这些留给后续 retargeting benchmark。

## 五、代码与复现审查

官方仓库包含 `phc/`、`legged_gym/`、`rsl_rl/`、`hardware_code/` 和 `scripts/data_process/`。README 给出 Python 3.8、Isaac Gym Preview 4、旧版 NumPy、PyTorch 与完整训练命令；AMASS 和 SMPL 模型需要用户自行下载并接受相应许可。H1 retargeting 大致需要：实现/确认机器人 FK → `grad_fit_h1_shape.py` 拟合 shape → `grad_fit_h1.py` 导出 motion。

| 复现维度 | 难度 | 原因 |
| --- | --- | --- |
| 单条 motion 可视化 | 中 | 依赖 SMPL、AMASS、Isaac Gym 和正确机器人资产 |
| teacher RL | 高 | 4096 并行环境、奖励 curriculum、旧版 Isaac Gym 栈 |
| student/DAgger | 高 | 需要 teacher checkpoint、状态维度和 rollout 配置完全一致 |
| 真机复现 | 很高 | H1、Orin NX、ZED/VR、通信和低层手控制器均是系统依赖 |

### 代码质量评分

| 审查项 | 评分 | 证据与判断 |
| --- | --- | --- |
| README/安装文档 | 高 | 给出 Isaac Gym、Python、依赖版本与 teacher/student 命令，但依赖旧版软件栈 |
| 环境可复现性 | 中 | 关键版本写得较清楚，SMPL/AMASS 与硬件 SDK 仍需外部下载 |
| retargeting 代码 | 高 | `phc/` 与 `scripts/data_process/` 包含 FK、shape fitting 和导出入口 |
| motion/robot 资产 | 中 | AMASS、SMPL 和机器人资产不随仓库完整分发，许可与目录准备是门槛 |
| checkpoint/评估 | 中 | 有 play/training 入口，论文结果对应 checkpoint 与完整评测复现仍需核对 |
| 真机部署 | 中 | 有 `hardware_code/` 和 H1 配置，但依赖特定 Orin、ZED、VR 与低层控制器 |
| 维护与许可证 | 中 | 仓库可运行但 Isaac Gym/旧依赖带来长期维护风险，非商业许可也需确认 |

## 六、批判性思考

### 优点

1. **统一接口清晰**：运动学 pose 将输入设备、人体估计、语言动作与控制 policy 解耦。
2. **真机闭环完整**：不仅给 reference，也展示了 teacher/student、历史观测、硬件随机化和 H1 部署。
3. **工程证据充分**：状态维度、随机化、DAgger 和失败相关消融比只展示视频更可复现。

### 局限性

1. PHC 的模型拟合与关键点优化没有为 object/terrain interaction 提供显式硬约束，穿透和脚滑仍可能交给后续 RL 修补。
2. 真机 motion-tracking 只覆盖 20 条 standing sequences，不能等同于 14k 序列的真实泛化。
3. 需要 root odometry；VIO 噪声或不连续会使 motion goal 偏离，论文也没有对极端 discontinuous goals 提供安全保证。
4. 灵巧手依赖 VR pose + 专用低层 hand controller，不能直接说明全身 retargeting 已解决手内力控。

### 潜在改进

- 把 PHC reference 与 GMR/OmniRetarget/UMR 的 penetration、foot-skating、contact-preservation 指标放进同一 benchmark；
- 在 fitting/IK 阶段加入硬 floor、collision 和 stance-foot constraints，而不是依赖 reward；
- 用不确定性或约束投影处理 VIO discontinuity，并报告失败恢复；
- 为不同 robot morphology 自动学习形状/表面 correspondence，同时保留 OmniH2O 的稀疏历史 student 接口。

## 七、相关工作与未来路线

**相关工作。** [GMR](https://arxiv.org/abs/2510.02252) 以多机器人通用 IK 和统一评测隔离 retargeting 质量；[OmniRetarget](https://arxiv.org/abs/2509.26633) 将 object/terrain 点加入 interaction mesh 并用硬约束；[UMR](https://arxiv.org/abs/2609.02134) 则学习密集表面 correspondence，减少手工 skeleton mapping。PHC 的优势是成熟、可接入 SMPL/RL pipeline；它的边界正是后两者针对的交互与跨本体问题。

**未来路线。** 最现实的延伸是把 PHC 的 SMPL shape fitting 作为统一输入适配器，把 GMR 的实时 IK 或 UMR 的 surface correspondence 作为几何后端，再用 OmniRetarget 的接触/碰撞硬约束清理 reference，最后沿用 OmniH2O 的历史 student 做真机控制。

| 维度 | PHC | GMR | OmniRetarget | UMR |
| --- | --- | --- | --- | --- |
| Source interface | SMPL/AMASS | BVH/SMPL/SMPL-X + key bodies | keypoints + object/terrain points | source surface point cloud |
| Correspondence | 拟合 shape 后的稀疏关键点 | 手工 key-body map | interaction-mesh vertices | 学习的 dense surface pairs |
| 主要约束 | 软几何拟合 + robot limits | 两阶段 IK + limits | 硬碰撞、脚粘地、速度/关节限位 | pose/normal/contact + constrained QP |
| Object/terrain | 不显式建模 | 不显式建模 | 原生支持 | contact map/scene 支持 |
| 在线速度 | 取决于 fitting/IK 与输入 | CPU real-time baseline | 偏离线数据生成 | setup 后约 65 FPS（论文数据） |
| 下游依赖 | teacher/student RL | BeyondMimic 等 tracker | proprioceptive RL | BeyondMimic/SONIC 或 tracker |

## 八、最终判断

PHC/OmniH2O 本质上是一个 **以 SMPL/关键点重定向为入口、以 teacher–student whole-body policy 为核心的系统工程**。它最值得学习的是运动学 pose 作为跨模态控制接口，以及历史观测对真机可部署性的帮助；最不应高估的是把 policy 的真机成功直接归功于 PHC retargeter。若目标是快速建立 H1 全身遥操作 baseline，PHC 值得优先复现；若目标是接触丰富或跨机器人数据生成，应优先比较 OmniRetarget 与 UMR。
