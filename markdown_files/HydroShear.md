---
title: "HydroShear: Hydroelastic Shear Simulation for Tactile Sim-to-Real Reinforcement Learning"
method_name: "HydroShear"
authors: ["An Thanh Dang"]
year: 2026
venue: "RSS"
tags: ["contact-reasoning", "reinforcement-learning", "contact-rich-manipulation", "robot-generalization", "dexterous-manipulation", "sim-to-real", "collision-avoidance", "tactile-feedback"]
image_source: "online"
arxiv_html: "https://arxiv.org/html/2603.00446v1"
---
# HydroShear
## 一句话总结

> HydroShear: Hydroelastic Shear Simulation for Tactile Sim-to-Real Reinforcement Learning 主要落在 [[碰撞避免]]、[[contact-estimation]]、[[接触推理]]、[[接触丰富操作]] 这一类问题上;从 System0 视角看,它的价值在于把身体状态、接触变化、局部感知或动力学约束接到机器人低层闭环里,让动作生成更稳定、安全、实时。
## 核心贡献

1. 围绕 **HydroShear: Hydroelastic Shear Simulation for Tactile Sim-to-Real Reinforcement Learning** 建立了一个与 碰撞避免、contact-estimation、接触推理、接触丰富操作、灵巧操作、强化学习 相关的机器人问题设定,重点不是高层语义规划,而是身体级闭环中的状态变化、接触变化或控制约束。
2. 方法层面的关键价值在于把感知、动力学、约束优化、策略学习或硬件反馈组织成可执行的低层动作生成机制,适合放进 System0 的快反应模块中理解。
3. 对系统集成而言,这类工作补上了 System1 技能选择与电机级控制之间的一层:它负责把上层目标变成动态可行、可恢复、可安全过滤的动作。
4. 若论文包含真实机器人、接触任务或仿真到真实迁移,核心应看它是否报告了闭环稳定性、扰动恢复、硬件延迟、失败模式和跨场景泛化。
## 问题背景

这篇论文可以放在 System0 的“具身快系统”背景下阅读:机器人需要在很短时间内根据本体状态、接触、力/触觉、局部视觉或动力学预测做出动作,而不是等待慢速规划器重新推理整个任务。碰撞避免、contact-estimation、接触推理、接触丰富操作、灵巧操作、强化学习、仿真到真实迁移 相关问题的难点通常不在“知道要做什么”,而在动作执行过程中状态会快速偏离、接触会突然改变、约束会实时收紧,低层系统必须及时修正。

### System0 关联

精读时优先看三件事:第一,论文使用哪些快速反馈量,例如关节状态、接触状态、力觉、触觉、深度或局部视觉;第二,反馈如何进入 MPC、WBC、策略网络、安全过滤器或恢复控制器;第三,输出是否直接影响机器人下一步动作,而不是只服务离线分析或高层任务规划。
## 方法详解

### System0 视角解析

从 System0 角度看,这篇论文的方法部分应当围绕 碰撞避免、contact-estimation、接触推理、接触丰富操作、灵巧操作、强化学习、仿真到真实迁移、tactile-feedback 来读。核心问题是:系统如何从快速变化的机器人状态中构造控制输入,如何处理接触、碰撞、约束、模型误差或执行延迟,以及如何把这些信息变成下一时刻的动作。

如果方法里有优化问题,重点看目标函数、约束和求解频率;如果方法里有学习策略,重点看观测空间、动作空间、训练扰动和部署时的闭环频率;如果方法里有硬件或传感器,重点看传感链路是否真的进入控制回路。这样读可以把论文和 System2 语言规划、System1 技能编排区分开来。
## 关键图表

### Figure 1: Pipeline / core system figure: Illustration of representative keyframes during the same policy rollout for peg inser

![Figure 1](https://arxiv.org/html/2603.00446v1/x4.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Pipeline / core system figure: Illustration of representative keyframes during the same policy rollout for peg inser”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 2: Illustration of representative keyframes during the same policy rollout for bin packi

![Figure 2](https://arxiv.org/html/2603.00446v1/images/rollouts/bin_packing_sim_keyframes_2.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Illustration of representative keyframes during the same policy rollout for bin packi”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### Figure 3: Illustration of representative keyframes during the same policy rollout for book shel

![Figure 3](https://arxiv.org/html/2603.00446v1/x5.png)

**图意**:这张图用于定位论文中的核心方法、系统结构或主要实验现象。标题线索是“Illustration of representative keyframes during the same policy rollout for book shel”,阅读时重点看输入信号、控制/策略模块、机器人本体交互和评估指标之间的关系。

**System0 读图重点**:看图中是否出现状态估计、局部感知、触觉/力觉、接触判断、MPC/WBC/策略输出、安全过滤或恢复控制等快闭环部件,以及它们是否直接影响下一步动作。

### 论文表格

#### Table 1: TABLE I: Calibration Results. Per-taxel shear RMSE [px] and cosine similarity (denoted as CS) between the predicted and

| Task | Shear Type | | | | | | | |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Dilation | Shear | Twist | Roll | | | | | |
| | RMSE ↓ \downarrow | CS ↑ \uparrow | RMSE | CS | RMSE | CS | RMSE | CS |
| TacSL | 2.187 | 0.323 | 4.262 | 0.126 | 4.022 | 0.157 | 3.861 | 0.137 |
| FOTS | 1.333 | 0.976 | 2.146 | 0.930 | 3.596 | 0.683 | 2.841 | 0.561 |
| Ours | 1.000 | 0.976 | 1.719 | 0.951 | 2.021 | 0.974 | 1.576 | 0.904 |

**说明**: TABLE I: Calibration Results. Per-taxel shear RMSE [px] and cosine similarity (denoted as CS) between the predicted and ground-truth shear vectors are reported for each shear type.

#### Table 2: TABLE II: Zero-shot Sim-to-Real Policy Evaluation. We test 5 different policies that is trained with different tactile s

| Model | Peg | | | | |
| --- | --- | --- | --- | --- | --- |
| Insertion | | | | | |
| Bin | | | | | |
| Packing | | | | | |
| Book | | | | | |
| Shelving | | | | | |
| Drawer | | | | | |
| Pulling | | | | | |
| Total | | | | | |
| TacSL Gray | 16/30 | 16/30 | 6/30 | 3/30 | 41/120 |
| TacSL Shear | 19/30 | 4/30 | 23/30 | 24/30 | 70/120 |
| FOTS | 1/30 | 5/30 | 20/30 | 15/30 | 41/120 |
| FOTS (Reimpl.) | 20/30 | 24/30 | 26/30 | 3/30 | 73/120 |
| HydroShear | 25/30 | 29/30 | 28/30 | 30/30 | 112/120 |

**说明**: TABLE II: Zero-shot Sim-to-Real Policy Evaluation. We test 5 different policies that is trained with different tactile simulation frameworks for each task: (1) TacSL tactile grayscale image; (2) TacSL tactile normalized shear; (3) Original FOTS tactile shear with our parallelization; (4) Our reimplemention of FOTS with improvements; (5) HydroShear tactile shear.

#### Table 3: TABLE III: Summary of our MDP for Peg Insertion, in terms of state, action, and reward components. Each component is den

| State Component | Symbol | Dimension | Description |
| --- | --- | --- | --- |
| Plug state † \dagger | x t plug  $\mathbf{x}_{t}^{\text{plug}}$ | R 13  $\mathbb{R}^{13}$ | Plug pose (quaternion and position) and velocity |
| Robot state † \dagger | q t  $\mathbf{q}_{t}$ | R 16  $\mathbb{R}^{16}$ | Joint positions (panda joints and gripper joints) and velocities. |
| End-effector pose | X t ee  $\mathbf{X}_{t}^{\text{ee}}$ | R 7  $\mathbb{R}^{7}$ | Pose of the robot’s end-effector |
| Socket pose | X t socket  $\mathbf{X}_{t}^{\text{socket}}$ | R 7  $\mathbb{R}^{7}$ | Pose of the socket |
| Plug-Socket Contact Force † \dagger | F t plug-socket  $\mathbf{F}_{t}^{\text{plug-socket}}$ | R 3  $\mathbb{R}^{3}$ | Force vector due to the contact interaction between the plug and the socket |
| Plug-Left Elastomer Contact Force † \dagger | F t plug-left  $\mathbf{F}_{t}^{\text{plug-left}}$ | R 3  $\mathbb{R}^{3}$ | Contact interaction forces between plug and the elastomer on left panda finger |
| Plug-Right Elastomer Contact Force † \dagger | F t plug-right  $\mathbf{F}_{t}^{\text{plug-right}}$ | R 3  $\mathbb{R}^{3}$ | Contact interaction forces between plug and the elastomer on right panda finger |
| Physics parameters † \dagger |  \nu | R 6  $\mathbb{R}^{6}$ | Mass, friction, restitution of object and friction of robot, elastomer, and environment |
| Observation Component | Symbol | Dimension | Description |
| End-effector pose socket frame | X t ee socket {}^{ $\text{socket}}\mathbf{X}_{t}^{\text{ee}}$ | R 7  $\mathbb{R}^{7}$ | Pose of the robot’s end-effector in socket frame. |
| Tactile Shear Field | I t shear  $\mathbf{I}_{t}^{\text{shear}}$ | $\mathbb{R}^{H\times W\times 3}$ | Shear occuring at tactile grid points. |
| Action Component | Symbol | Dimension | Description |
| Delta End-effector Pose Action | $\mathbf{X}^{ee}$ | R 7  $\mathbb{R}^{7}$ | Desired transform to apply onto the current end-effector pose |
| Stiffness gains | K p  $\mathbf{K}_{p}$ | R 6  $\mathbb{R}^{6}$ | Cartesian Impedance Stiffness gains |
| Damping gains | K d  $\mathbf{K}_{d}$ | R 6  $\mathbb{R}^{6}$ | Cartesian Impedance Damping gains |
| Reward Component | Symbol | Dimension | Description |
| Task success reward | r success r_{ $\text{success}}$ | R 1  $\mathbb{R}^{1}$ | Reward for task success |
| Keypoint alignment penalty | r keypoint r_{ $\text{keypoint}}$ | R 1  $\mathbb{R}^{1}$ | Negative mean distance between plug keypoints and socket keypoints |
| Action Derivative penalty | r a  d r_{ad} | R 1  $\mathbb{R}^{1}$ | Negative Norm of finite difference between current action and previous action. |
| Table Contact Force penalty | r table r_{ $\text{table}}$ | R 1  $\mathbb{R}^{1}$ | Sum of force norm between contact forces between table and all other objects. |

**说明**: TABLE III: Summary of our MDP for Peg Insertion, in terms of state, action, and reward components. Each component is denoted by its name, shorthand symbol, dimensionality and a brief description. † $\dagger$: only used in simulation.

#### Table 4: TABLE IV: Summary of our MDP for Bin Packing, in terms of state, action, and reward components. Each component is denote

| State Component | Symbol | Dimension | Description |
| --- | --- | --- | --- |
| Grasped Cube state † \dagger | x t cube  $\mathbf{x}_{t}^{\text{cube}}$ | R 13  $\mathbb{R}^{13}$ | Grasped cube pose and velocity |
| Preset Cubes state † \dagger | x t preset-cubes  $\mathbf{x}_{t}^{\text{preset-cubes}}$ | R 7  $\mathbb{R}^{7}$ | Pose information for 15 cubes inside the bin. |
| Robot state † \dagger | q t  $\mathbf{q}_{t}$ | R 14  $\mathbb{R}^{14}$ | Joint positions and velocities |
| End-effector pose | X t ee  $\mathbf{X}_{t}^{\text{ee}}$ | R 7  $\mathbb{R}^{7}$ | Pose of the robot’s end-effector |
| Goal pose | X goal  $\mathbf{X}^{\text{goal}}$ | R 7  $\mathbb{R}^{7}$ | Target goal pose of grasped cube in end-effector frame |
| Grasped Cube Contact Forces † \dagger | F t cube-left  $\mathbf{F}_{t}^{\text{cube-left}}$ | R 3  $\mathbb{R}^{3}$ | Contact force between grasped cube and left elastomer |
| F t cube-right  $\mathbf{F}_{t}^{\text{cube-right}}$ | R 3  $\mathbb{R}^{3}$ | Contact force between grasped cube and right elastomer | |
| F t cube-bin  $\mathbf{F}_{t}^{\text{cube-bin}}$ | R 3  $\mathbb{R}^{3}$ | Contact force between grasped cube and bin | |
| F t cube-presetcubes  $\mathbf{F}_{t}^{\text{cube-presetcubes}}$ | R 42  $\mathbb{R}^{42}$ | Contact force between grasped cube and 15 preset cubes | |
| Physics parameters † \dagger |  \nu | R 6  $\mathbb{R}^{6}$ | Mass, friction, restitution of object and friction of robot, elastomer, and env |
| Observation Component | Symbol | Dimension | Description |
| End-effector pose goal frame | X t ee goal {}^{ $\text{goal}}\mathbf{X}_{t}^{\text{ee}}$ | R 7  $\mathbb{R}^{7}$ | End-effector pose expressed in the goal frame |
| Tactile Shear Field | I t tactile  $\mathbf{I}_{t}^{\text{tactile}}$ | $\mathbb{R}^{H\times W\times 3}$ | Shear occurring on tactile grid points |
| Action Component | Symbol | Dimension | Description |
| Delta End-effector Pose Action | $\mathbf{X}^{\text{ee}}$ | R 7  $\mathbb{R}^{7}$ | Desired transform applied to the current end-effector pose |
| Stiffness gains | K p  $\mathbf{K}_{p}$ | R 6  $\mathbb{R}^{6}$ | Cartesian impedance stiffness gains |
| Damping gains | K d  $\mathbf{K}_{d}$ | R 6  $\mathbb{R}^{6}$ | Cartesian impedance damping gains |
| Reward Component | Symbol | Dimension | Description |
| Task success reward | r success r_{ $\text{success}}$ | R 1  $\mathbb{R}^{1}$ | Reward for task success |
| Keypoint alignment penalty | r keypoint r_{ $\text{keypoint}}$ | R 1  $\mathbb{R}^{1}$ | Negative mean distance between grasped cube keypoints and goal keypoints |
| Action derivative penalty | r a  d r_{ad} | R 1  $\mathbb{R}^{1}$ | Negative norm of finite difference between consecutive actions |
| Table contact force penalty | r table r_{ $\text{table}}$ | R 1  $\mathbb{R}^{1}$ | Sum of contact force norms between table and all objects |

**说明**: TABLE IV: Summary of our MDP for Bin Packing, in terms of state, action, and reward components. Each component is denoted by its name, shorthand symbol, dimensionality and a brief description. † $\dagger$: only used in simulation.

#### Table 5: TABLE V: Summary of our MDP for Book Shelving, in terms of state, action, and reward components. Each component is denot

| State Component | Symbol | Dimension | Description |
| --- | --- | --- | --- |
| Grasped Book state † \dagger | x t cube  $\mathbf{x}_{t}^{\text{cube}}$ | R 13  $\mathbb{R}^{13}$ | Grasped book pose and velocity |
| Preset Books state † \dagger | x t preset-cube  $\mathbf{x}_{t}^{\text{preset-cube}}$ | R 7  $\mathbb{R}^{7}$ | Preset books information for 14 books inside the bookshelf |
| Robot state † \dagger | q t  $\mathbf{q}_{t}$ | R 16  $\mathbb{R}^{16}$ | Joint positions (panda and gripper joints) and velocities |
| End-effector pose | X t ee  $\mathbf{X}_{t}^{\text{ee}}$ | R 7  $\mathbb{R}^{7}$ | Pose of the robot’s end-effector |
| Goal pose | X goal  $\mathbf{X}^{\text{goal}}$ | R 7  $\mathbb{R}^{7}$ | Target goal pose of grasped book |
| Grasped Book Contact Forces † \dagger | F t book-elastomer  $\mathbf{F}_{t}^{\text{book-elastomer}}$ | R 6  $\mathbb{R}^{6}$ | Contact force between grasped book and both elastomers |
| F t book-shelf  $\mathbf{F}_{t}^{\text{book-shelf}}$ | R 3  $\mathbb{R}^{3}$ | Contact force between grasped book and shelf | |
| F t book-presetbooks  $\mathbf{F}_{t}^{\text{book-presetbooks}}$ | R 42  $\mathbb{R}^{42}$ | Contact force between grasped book and 14 preset books | |
| Physics parameters † \dagger |  \nu | R 6  $\mathbb{R}^{6}$ | Mass, friction, restitution of object and friction of robot, elastomer, and environment |
| Observation Component | Symbol | Dimension | Description |
| End-effector pose goal frame | X t ee goal {}^{ $\text{goal}}\mathbf{X}_{t}^{\text{ee}}$ | R 7  $\mathbb{R}^{7}$ | End-effector pose expressed in the goal frame |
| Tactile Shear Field | I t tactile  $\mathbf{I}_{t}^{\text{tactile}}$ | $\mathbb{R}^{H\times W\times 3}$ | Shear occurring on tactile grid points |
| Action Component | Symbol | Dimension | Description |
| Delta End-effector Pose Action | $\mathbf{X}^{\text{ee}}$ | R 7  $\mathbb{R}^{7}$ | Desired transform applied to the current end-effector pose |
| Stiffness gains | K p  $\mathbf{K}_{p}$ | R 6  $\mathbb{R}^{6}$ | Cartesian impedance stiffness gains |
| Damping gains | K d  $\mathbf{K}_{d}$ | R 6  $\mathbb{R}^{6}$ | Cartesian impedance damping gains |
| Reward Component | Symbol | Dimension | Description |
| Task success reward | r success r_{ $\text{success}}$ | R 1  $\mathbb{R}^{1}$ | Reward for task success |
| Keypoint alignment penalty | r keypoint r_{ $\text{keypoint}}$ | R 1  $\mathbb{R}^{1}$ | Negative mean distance between grasped book keypoints and goal keypoints |
| Action derivative penalty | r a  d r_{ad} | R 1  $\mathbb{R}^{1}$ | Negative norm of finite difference between consecutive actions |
| Table contact force penalty | r table r_{ $\text{table}}$ | R 1  $\mathbb{R}^{1}$ | Sum of contact force norms between table and all objects |

**说明**: TABLE V: Summary of our MDP for Book Shelving, in terms of state, action, and reward components. Each component is denoted by its name, shorthand symbol, dimensionality and a brief description. † $\dagger$: only used in simulation.

#### Table 6: TABLE VI: Summary of our MDP for Drawer Pulling, in terms of state, action, and reward components. Each component is den

| State Component | Symbol | Dimension | Description |
| --- | --- | --- | --- |
| Handle state † \dagger | x t handle  $\mathbf{x}_{t}^{\text{handle}}$ | R 13  $\mathbb{R}^{13}$ | Drawer handle pose and velocity |
| Drawer Box Joint State † \dagger | q t drawer  $\mathbf{q}_{t}^{\text{drawer}}$ | R 2  $\mathbb{R}^{2}$ | Drawer Prismatic Joint (for sliding the drawer) position and velocity. |
| Robot state † \dagger | q t  $\mathbf{q}_{t}$ | R 16  $\mathbb{R}^{16}$ | Joint positions (panda joints and gripper joints) and velocities. |
| End-effector pose | X t ee  $\mathbf{X}_{t}^{\text{ee}}$ | R 7  $\mathbb{R}^{7}$ | Pose of the robot’s end-effector |
| Handle-Elastomer Contact Force | F t handle-elastomer  $\mathbf{F}_{t}^{\text{handle-elastomer}}$ | R 6  $\mathbb{R}^{6}$ | Contact interaction force vector between drawer handle and both elastomers |
| Handle-Finger Contact Force | F t handle-finger  $\mathbf{F}_{t}^{\text{handle-finger}}$ | R 3  $\mathbb{R}^{3}$ | Contact interaction force vector between drawer handle and both fingers |
| Physics parameters † \dagger |  \nu | R 6  $\mathbb{R}^{6}$ | Mass, friction, restitution of object and friction of robot, elastomer, and env |
| Observation Component | Symbol | Dimension | Description |
| End-effector pose | X t ee  $\mathbf{X}_{t}^{\text{ee}}$ | R 9  $\mathbb{R}^{9}$ | Pose of the robot’s end-effector |
| Tactile Shear Field | I t tactile  $\mathbf{I}_{t}^{\text{tactile}}$ | $\mathbb{R}^{H\times W\times 3}$ | Shear occurring on tactile grid points. |
| Action Component | Symbol | Dimension | Description |
| Delta End-effector Pose Action | $\mathbf{X}^{\text{ee}}$ | R 7  $\mathbb{R}^{7}$ | Desired transform to apply onto the current end-effector pose |
| Gripper Joint Position Action | q ^ t gripper  $\hat{\mathbf{q}}_{t}^{\text{gripper}}$ | R 1  $\mathbb{R}^{1}$ | Desired gripper position on robot to control how much to grasp the drawer handle |
| Stiffness gains | k p k_{p} | R 6  $\mathbb{R}^{6}$ | Cartesian Impedance Stiffness gains |
| Damping gains | k d k_{d} | R 6  $\mathbb{R}^{6}$ | Cartesian Impedance Damping gains |
| Reward Component | Symbol | Dimension | Description |
| Task success reward | r s r_{s} | R 1  $\mathbb{R}^{1}$ | Reward for task success |
| Keypoint alignment penalty | r k r_{k} | R 1  $\mathbb{R}^{1}$ | Negative mean distance between drawer handle keypoints and goal keypoints |
| Action Derivative penalty | r a  d r_{ad} | R 1  $\mathbb{R}^{1}$ | Negative finite difference between current action and previous action. |
| Table Contact Force penalty | r table r_{ $\text{table}}$ | R 1  $\mathbb{R}^{1}$ | Sum of force norm between contact forces between table and all other objects. |
| Pre-ForcePerturb Gripper Penalty | r preperturb r_{ $\text{preperturb}}$ | R 1  $\mathbb{R}^{1}$ | Grasp Penalty whenever the robot grasps more before force perturbation happens. |
| Post-ForcePerturb Gripper Reward | r postperturb r_{ $\text{postperturb}}$ | R 1  $\mathbb{R}^{1}$ | Sum of normal forces between elastomers and drawer handle during perturbation. |
| Robot-Drawer Contact Penalty | r drawer r_{ $\text{drawer}}$ | R 1  $\mathbb{R}^{1}$ | Penalty term when robot contacts any part of drawer that’s not the handle. |

**说明**: TABLE VI: Summary of our MDP for Drawer Pulling, in terms of state, action, and reward components. Each component is denoted by its name, shorthand symbol, dimensionality and a brief description. † $\dagger$: only used in simulation.

#### Table 7: TABLE VII: Teacher PPO Hyperparameters.

| Hyperparameter | Value |
| --- | --- |
| Discount factor | 0.99 |
| GAE parameter | 0.95 |
| Grad norm | 1.0 |
| Entropy coeff. | 0.0 |
| PPO clip range | 0.2 |
| Bounds loss coeff. | 0.0 |
| Policy loss coeff. | 1.0 |
| Value loss coeff. | 2.0 |
| Base learning rate | 1e-4 |
| Adaptive LR KL target | 2e-3 |
| Max. learning rate | 1e-2 |
| Num. environments | 1024 |
| Episode length | 256 |
| Mini-epochs | 4 |
| Minibatch size | 1024 |

**说明**: TABLE VII: Teacher PPO Hyperparameters.

#### Table 8: TABLE VIII: Student PPO Hyperparameters.

| Hyperparameter | Value |
| --- | --- |
| Discount factor | 0.99 |
| GAE parameter | 0.95 |
| Grad norm | 1.0 |
| Entropy coeff. | 0.0 |
| PPO clip range | 0.2 |
| Bounds loss coeff. | 0.0 |
| Policy loss coeff. | 1.0 |
| Value loss coeff. | 2.0 |
| Base learning rate | 1e-4 |
| Adaptive LR KL target | 8e-3 |
| Max. learning rate | 1e-4 |
| Num. environments | 256 |
| Episode length | 256 |
| Mini-epochs | 4 |
| Minibatch size | 1024 |

**说明**: TABLE VIII: Student PPO Hyperparameters.

#### Table 9: TABLE IX: Mean / median real-world environment timesteps to success (success-only). Lower is better.

| Model | Peg | | | |
| --- | --- | --- | --- | --- |
| Insertion | | | | |
| Bin | | | | |
| Packing | | | | |
| Book | | | | |
| Shelving | | | | |
| Drawer | | | | |
| Pulling | | | | |
| TacSL Gray | 76.9 / 74.0 | 96.9 / 89.5 | 178.2 / 176.5 | 63.7 / 63.0 |
| TacSL Shear | 87.1 / 81.0 | 97.7 / 91.0 | 99.4 / 93.0 | 155.6 / 125.0 |
| FOTS | 197.0 / 197.0 | 237.6 / 213.0 | 158.2 / 163.0 | 96.8 / 90.0 |
| FOTS (Reimpl.) | 74.9 / 73.5 | 169.2 / 148.5 | 126.7 / 114.0 | 75.7 / 72.0 |
| HydroShear | 85.2 / 82.0 | 108.8 / 86.0 | 180.2 / 182.5 | 120.6 / 116.5 |

**说明**: TABLE IX: Mean / median real-world environment timesteps to success (success-only). Lower is better.

#### Table 10: TABLE X: Comptutation Speed Comparison. We evaluate the computational speed of FOTS, our implemented FOTS, and HydroShea

| Sensor Model | Computation Time (ms) | | | | | |
| --- | --- | --- | --- | --- | --- | --- |
| 256 Env | 512 Env | 1024 Env | | | | |
| | Mean ↓ \downarrow | Stdev ↑ \uparrow | Mean | Stdev | Mean | Stdev |
| FOTS | 122.829 | 3.445 | 230.884 | 6.360 | 446.656 | 4.028 |
| FOTS (Reimpl.) | 11.145 | 1.826 | 11.827 | 1.621 | 12.040 | 1.593 |
| Ours | 33.805 | 1.505 | 57.042 | 1.787 | 103.736 | 1.569 |

**说明**: TABLE X: Comptutation Speed Comparison. We evaluate the computational speed of FOTS, our implemented FOTS, and HydroShear by running it on the rollouts of a teacher policy on a Peg Insertion task. The number of environments labelled in the table describes the number of rollouts that the model has to calculate the tactile shear for during the evaluation. The computation time mean and standard deviation are calculated in milliseconds.
## 实验解读

- 评价重点:围绕 碰撞避免、contact-estimation、接触推理、接触丰富操作、灵巧操作,优先看真实机器人结果、扰动恢复、接触稳定性、控制误差、成功率、实时性和 sim-to-real 差距。
- System0 视角:实验结论不只看最终成功率,还要看闭环过程中是否能处理状态估计误差、执行延迟、碰撞/接触突变和外部扰动。
- 与上层系统的关系:如果方法能在快闭环内稳定工作,它可以作为 System1 技能或 System2 规划的底层执行保障;如果只在离线或慢速设定下成立,则还不能直接作为身体级快闭环模块。
## 批判性思考

### 优点

- 论文把 碰撞避免、contact-estimation、接触推理、接触丰富操作、灵巧操作 放到机器人身体级闭环里讨论,和只做高层语义推理的工作相比,更接近真实机器人部署时会遇到的状态、动力学和安全约束。
- 若论文同时给出公式、系统结构图、硬件实验或消融结果,这些证据可以帮助判断方法是否真的能支撑毫秒到百毫秒级反应。

### 局限

- 需要重点核查控制频率、推理延迟、传感延迟、失败恢复、硬件磨损、未见场景泛化和安全边界;这些因素决定它能否成为可靠的 System0 模块。
- 对学习型方法,还要区分训练环境中的鲁棒性和真实部署中的鲁棒性,尤其是接触丰富任务、动态运动任务和人形机器人任务。
## 论文定位

- 研究对象:HydroShear: Hydroelastic Shear Simulation for Tactile Sim-to-Real Reinforcement Learning。
- 关键词:碰撞避免、contact-estimation、接触推理、接触丰富操作、灵巧操作、强化学习、仿真到真实迁移、tactile-feedback、visuomotor-policy、zero-shot。
- System0 关联:关注机器人身体级快反应链路中“状态/接触/局部感知 -> 约束/优化/策略 -> 动作”的短闭环,不把问题停留在语言规划或任务分解层。

## 速查卡片

> [!summary] HydroShear
> - **论文**: https://www.roboticsproceedings.org/rss22/p155.pdf
> - **arXiv**: http://arxiv.org/abs/2603.00446v1
> - **arXiv HTML**: https://arxiv.org/html/2603.00446v1
