---
title: "Switch-JustDance: Benchmarking Whole-Body Motion Tracking Controllers Using a Commercial Console Game"
method_name: "Switch-JustDance"
authors: [Jeonghwan Kim, Wontaek Kim, Yidan Lu, Jin Cheng, Fatemeh Zargarbashi, Zicheng Zeng, Zekun Qi, Zhiyang Dou, Nitish Sontakke, Donghoon Baek, Sehoon Ha, Tianyu Li]
year: 2026
venue: CVPR
tags: [humanoid, whole-body-control, benchmark, motion-tracking, motion-retargeting, embodied-AI, sim-to-real]
zotero_collection: _inbox
image_source: online
arxiv_html: https://arxiv.org/html/2511.17925v3
created: 2026-06-29
---

# Switch-JustDance: Benchmarking Whole-Body Motion Tracking Controllers Using a Commercial Console Game

## 元信息

| 项目 | 内容 |
|------|------|
| 作者 | Jeonghwan Kim, Wontaek Kim, Yidan Lu, Jin Cheng, Fatemeh Zargarbashi, Zicheng Zeng, Zekun Qi, Zhiyang Dou 等 12 人（前 7 人并列一作） |
| 机构 | Georgia Tech、香港大学(HKU)、ETH Zurich、华南理工(SCUT)、清华(THU)、MIT、PNDbotics |
| 会议 | CVPR 2026 |
| 类别 | Humanoid / 全身控制评测基准 |
| 日期 | 2025-11（arXiv v3） |
| 项目主页 | https://switch-justdance.github.io/ |
| 链接 | [arXiv](https://arxiv.org/abs/2511.17925) / [Project](https://switch-justdance.github.io/) |

---

## 一句话总结

> 把 Nintendo Switch 上的《Just Dance》改造成一个低成本、可复现、能直接和人类同台比分的真实世界基准，用游戏内置打分来评测人形机器人的全身动作跟踪控制器。

---

## 核心贡献

1. **低成本可复现的真实世界基准**: 提出 [[Switch-JustDance]]，用约 \$400 的 [[Nintendo Switch]] + 《[[Just Dance]]》商业体感游戏作为标准化、即插即用的全身人形控制评测平台，相比动辄 \$50,000 以上的光学 [[MoCap|动捕]] 系统大幅降低门槛，且全球统一标准、便于社区复现。
2. **打分机制的系统性验证**: 通过 10 名受试者 ×5 首歌 ×4 次重复（共 200 段动作）的用户研究，从 **Validity（有效性）、Discriminability（区分度）、Repeatability（可重复性）** 三个维度严格验证了 [[Just Dance Score|JDS]] 作为评测指标的可靠性。
3. **三个 SOTA 控制器的硬件实测**: 在真实 [[Unitree G1]] 硬件上对 [[GMT]]、[[TWIST]]、[[Any2Track]] 三个代表性全身跟踪控制器做基准评测，并与人类直接对比，揭示当前控制器仅达人类约 1/3 分数，以及"平滑输入提稳定 / 动态输入暴露高频控制极限 / 仿真高估鲁棒性"等关键洞见。

---

## 问题背景

### 要解决的问题
人形与足式机器人的全身控制能力进步迅速（体操、动态舞蹈、实时遥操作），但**如何在真实世界中以公平、可复现、物理接地的方式评测这些控制策略，并能与人类直接对比**，仍缺乏标准化手段。

### 现有方法的局限
作者指出现有评测范式的几个痛点：
1. **依赖大规模人类动作数据集**（[[AMASS]]、[[LaFAN1]]）：更新不频繁，且各家对数据划分/评测协议不一致，跨方法比较困难；
2. **仿真基准受限**（如 [[HumanoidBench]]）：局限于特定机器人模型与模拟环境，忽略硬件因素；
3. **真实世界动捕评测昂贵**：光学 MoCap 需要专用场地、仪器、高运维成本，难以大规模/社区化普及；
4. **几乎从不与人类同标准对比**：留下"机器人离人类运动水平还有多远"这一开放问题。

### 本文的动机
- 《Just Dance》天然具备**统一的内置打分系统**、**持续更新的大型全身动作库**（每年新增内容、单曲 >3 分钟）、以及**为不同体型人群设计**（儿童/成人乃至机器人都能打分，无需为打分而做 retarget）；
- 因为平台本就为人类设计，**机器人和人类能在完全相同条件下定量对比**；
- 只需 Switch + Joy-Con + HDMI 采集卡即可搭建，**成本低、可重复、全球标准统一**，是标准化评测的理想候选。

---

## 方法详解

### 系统架构

[[Switch-JustDance]] 是一条"游戏画面 → 人类动作重建 → 机器人重定向 → 控制器执行 → 游戏打分"的闭环流水线（见 Figure 1）。其核心思路是：把 Joy-Con 牢固绑在机器人手上，机器人跳舞时 Joy-Con 的 IMU 信号被 Switch 当作"玩家动作"来计分，于是机器人就像一个人类玩家一样被打分。

整条流水线由四个模块组成：

- **输入**: Switch 实时游戏画面（屏幕上的示范编舞）
- **输出**: 机器人执行的全身动作 + 游戏内 [[Just Dance Score|JDS]] 分数

### 核心模块

#### 模块1: Streaming Module（流式采集模块）

**设计动机**: 把 Switch 的实时游戏画面以可忽略的延迟接入动捕系统。

**具体实现**:
- 用 **HDMI 采集卡** 捕获 Switch 实时画面，游戏开始后即把视频帧流式传给 MoCap 模块。

#### 模块2: MoCap Module（动作重建模块）

**设计动机**: 从单目 RGB 视频恢复屏幕上舞者的 3D 人体动作。

**具体实现**:
- 用 [[GVHMR]]（World-grounded Human Motion Recovery）从单目 RGB 重建人体姿态，输出 [[SMPL]] 参数化的人体动作；
- 关键改造：用**滑动窗口推理（sliding-window inference）**把 GVHMR 改成实时运行，约 **200 ms/帧（约 5 Hz）** 输出稳定姿态。

#### 模块3: Retargeting Module（重定向模块）

**设计动机**: 把人类 [[SMPL]] 动作映射到目标机器人形态，保证不同比例机器人上的运动学可行性。

**具体实现**:
- 用 [[GMR]]（General Motion Retargeting）做重定向，约 **50 ms/帧**，产出与 MoCap 更新同步的机器人空间关键帧轨迹；
- 用一个**异步插值线程**填补关键帧之间的中间帧：位置用 **[[LERP|线性插值]]**、旋转用 **[[SLERP|球面线性插值]]**，把稀疏的 5 Hz 更新上采样为平滑的 **10–15 Hz** 参考流（引入约 0.2 s 的恒定延迟换取连续性，详见 Appendix A）。

#### 模块4: Motion Tracking Controller（动作跟踪控制器，即被测对象）

**设计动机**: 根据重定向后的参考动作 + 机器人本体反馈生成控制指令——**这正是被评测的人形控制器**。

**具体实现**:
- 接收 10–15 Hz 参考流，输出全身控制指令；本文换入 [[GMT]] / [[TWIST]] / [[Any2Track]] 三种策略做对比。

### 系统运行配置与优势

- **硬件**: 工作站配 RTX 4090 GPU + Intel Core i9-13900K CPU；端到端耗时由 GVHMR 推理（约 200 ms/步）主导，GMR 重定向约 50 ms/关键帧，异步插值线程把稀疏更新转为平滑 10–15 Hz 参考流。
- **四大优势**: (1) **低成本**（Switch 约 \$400，入门级动捕 >\$50,000）；(2) **易搭建、高可重复**（全球统一标准）；(3) **天然支持跨形态评测**（Just Dance 为不同体型设计，打分无需 retarget）；(4) **长时程、动态、高质量舞蹈动作**（单曲 >3 分钟、每年更新），对比静态的 AMASS 更鲜活。

---

## 关键图表

<!-- 图片均使用 arXiv HTML 在线链接，未本地化 -->

### Figure 1: System Diagram / 系统流水线总览

![Figure 1](https://arxiv.org/html/2511.17925v3/sec/figures/system_diagram_cameraready.png)

**说明**: Switch-JustDance 整体流水线。采集 Switch 游戏画面 → 流式送入 MoCap 模块用 [[GVHMR]] 恢复舞者的 [[SMPL]] 人体动作 → 经 Retarget 模块用 [[GMR]] 映射到机器人 → 由待测全身控制器执行 → 机器人手持 Joy-Con 在游戏内被打分。这张图是理解整个"游戏即基准"闭环的关键。

### Figure 2: Hardware Execution Snapshots / 硬件执行快照对比

![Figure 2](https://arxiv.org/html/2511.17925v3/sec/figures/snapshots_b.png)

**说明**: 自上而下分别为：Switch 原始画面（源动作）、[[GMR]] 输出的重定向动作、以及三个人形控制器（[[TWIST]]、[[GMT]]、[[Any2Track]]）在硬件上的执行；列从左到右为时间推进。[[Unitree G1]] 手持 Joy-Con 实打《Just Dance》并接受游戏计分。直观佐证了流水线能让真实机器人完成游戏舞蹈并被打分。

### Figure B1: HOVER-Controlled H1 in Simulation / 跨形态泛化（附录）

![Figure B1](https://arxiv.org/html/2511.17925v3/sec/figures/rebuttal_h1_HOVER.001.png)

**说明**: 把流水线应用到 [[HOVER]] 控制的 [[Unitree H1]] 机器人（仿真）跳《Old Town Road》（0:35→0:39 的快照）。用于支撑"框架对不同机器人形态无关（morphology-agnostic）"这一泛化主张——因为打分与 retarget 都不绑定特定形态。

### Table 1: Benchmark Results / 主基准结果（核心表）

不同控制器在 **平滑输入(Smo)** 与 **动态输入(Dyn)**、**真实(Real)** 与 **仿真(Sim)** 下的全套指标。JDS 越高越好；MPJPE / DTW 越低越好（位置跟踪与时序对齐越准）；SR(成功率) 越高越好；Smoothness（Jerk/Acc）越低越好。MPJPE 的 **Active** 仅统计摔倒前的帧、**All** 含摔倒后所有帧。仿真无物理 Joy-Con 故 JDS 不适用。

| Player-Setting | JDS↑ (Easy) | JDS↑ (Hard) | JDS↑ (All) | MPJPE All↓ | MPJPE Active↓ | DTW All↓ | DTW Act↓ | SR(%)↑ | Jerk↓ | Acc↓ |
|----------------|------|------|------|------|------|------|------|------|------|------|
| Human-leaderBoard | 13333 | 13333 | **13333** | - | - | - | - | **100.0** | - | - |
| **Human** | 9867 | 6543 | **8538** | 111.9 | 111.9 | 102.4 | 102.4 | **100.0** | **3350.5** | **95.1** |
| GMT-Real-Smo | 2765 | 1891 | 2415 | 130.9 | 139.5 | 114.1 | 120.3 | 73.3 | 3174.1 | 78.5 |
| TWIST-Real-Smo | 2085 | 1557 | 1874 | 117.1 | 119.8 | 96.9 | 97.9 | 80.0 | 2811.3 | 70.6 |
| Any2Track-Real-Smo | 2978 | 2785 | **2900** | 102.6 | 139.4 | 44.7 | 69.1 | 60.0 | 1978.8 | 82.6 |
| GMT-Real-Dyn | 2881 | 2530 | 2741 | **88.8** | 102.6 | 50.8 | 59.7 | 80.0 | 4098.6 | 126.3 |
| TWIST-Real-Dyn | 1501 | 1140 | 1357 | 108.5 | 142.7 | 71.0 | 95.4 | 33.3 | 4024.1 | 126.0 |
| Any2Track-Real-Dyn | 2753 | 2251 | 2552 | 114.2 | 128.2 | 40.9 | 49.5 | 80.0 | 3783.2 | 127.6 |
| GMT-Sim-Smo | - | - | - | 121.0 | 137.4 | 105.9 | 118.5 | 20.0 | 2146.2 | 58.7 |
| TWIST-Sim-Smo | - | - | - | 116.7 | 125.2 | 106.5 | 108.4 | 80.0 | 1684.6 | 49.4 |
| Any2Track-Sim-Smo | - | - | - | 91.4 | 106.9 | 65.2 | 69.2 | 80.0 | 2115.1 | 88.1 |
| GMT-Sim-Dyn | - | - | - | 87.8 | 137.0 | 74.7 | 114.0 | 0.0 | 3694.6 | 110.5 |
| TWIST-Sim-Dyn | - | - | - | 100.9 | 127.8 | 62.2 | 74.7 | 40.0 | 3438.9 | 110.1 |
| Any2Track-Sim-Dyn | - | - | - | 90.9 | 105.7 | 65.6 | 65.5 | 80.0 | 3949.8 | 130.1 |

**说明**: 这是全文最核心的结果。(1) **人机差距巨大**：人类专家可达理论上限 13333，受试者平均 8538，而最佳控制器（Any2Track-Real-Smo）仅 2900，约人类的 1/3。(2) **反直觉**：Real-Dyn 下机器人 MPJPE 可低至 88.8 mm，优于人类的 111.9 mm（帧级跟踪更准），但低 MPJPE 往往伴随更低成功率——准而不稳。(3) **JDS 是更"高级"的指标**：TWIST 在 Real-Smo 下 MPJPE/SR/平滑度都不错却 JDS 不及 GMT，说明 JDS 不只看姿态精度或平滑度。(4) 人类运动**最平滑**（Jerk/Acc 最低）。

### Table C1: Selected Just Dance Tracks / 基准选曲（附录）

选曲标准：(1) 无"手触地"动作（Joy-Con 装在手上，触地不可行）；(2) 尽量少原地转身（机器人脚不适合连续旋转、踝关节扭矩需求大，实测原地转身常被忽略）。lvl 1–2 归 Easy、3–4 归 Hard。

| Song | Game Edition | Difficulty | Length (s) |
|------|--------------|-----------|-----------|
| Old Town Road | JustDance 2020 | Easy (lvl 1) | 161 |
| Heart Of Glass | JustDance 1 | Easy (lvl 2) | 216 |
| Unstoppable | JustDance 2025 Edition | Easy (lvl 2) | 204 |
| Padam Padam | JustDance 2025 Edition | Hard (lvl 3) | 149 |
| Pink Venom | JustDance 2025 Edition | Hard (lvl 3) | 178 |

**说明**: 5 首选曲（3 Easy + 2 Hard），用于全部基准实验。选曲准则本身反映了机器人本体的硬件约束（手部占用、脚部不适合旋转），是把"人类游戏"迁移到机器人时的实际工程妥协。

---

## 实验

### 验证研究（System Validation）

在用 Switch-JustDance 评测控制器之前，先验证 [[Just Dance Score|JDS]] 作为指标本身是否可靠。用户研究：**10 名受试者（20–34 岁，多样性别与体型）×5 首不同难度歌曲 ×4 次重复 = 每人 20 段、共 200 段动作**。

| 维度 | 方法 | 结果 | 结论 |
|------|------|------|------|
| **Validity（有效性）** | JDS 与 [[PA-MPJPE]]（Procrustes 对齐的关节位置误差）做相关 | 相关系数 $r\in[-0.76,-0.42]$，显著（p 显著） | JDS 与运动学指标显著负相关（分越高误差越小），且 JDS 还纳入了关节加速度、与音乐节拍的韵律对齐等 PA-MPJPE 不反映的感知/动态质量 |
| **Discriminability（区分度）** | 对比 静态下界 / 世界排行榜上界 / 受试者中段 | 静态 ≈ 0；排行榜上限 13333；受试者 [3460, 12952] | 三档之间分离明显，JDS 有足够动态范围区分不同质量的表现 |
| **Repeatability（可重复性）** | [[ICC]]、[[CV]]、[[KCC]]（Kendall 一致性系数） | ICC=0.735，CV=10.0%，KCC=0.678 | 重复测量一致性强、试次间波动小、排名一致性好 |

**Potential Shortcoming（潜在缺陷）**: JDS 主要依赖**单个 Joy-Con**（绑在手上）的运动，理论上玩家可只动上身、下身不动也拿高分；作者论证在"忠实跟踪参考动作"假设下高分隐含需要全身协调（节奏/平衡/时序强耦合），并建议未来用视频佐证视觉保真度。

### 实验设置

- **两种输入保真度**：
  - **Smo（Streaming Smoothed Gameplay，平滑流式）**: 模拟在线实时游玩，实时动捕+即时重定向；因无未来参考帧、计算窗口小，对姿态流做**低通平滑 + 离群剔除 + 速度钳制**，得到保守但平滑、丢失高频细节的参考；
  - **Dyn（Offline Dynamic Reference，离线动态）**: 先采集无滤波的流式动作，再离线 GVHMR 批处理、SLERP/LERP 重采样到 30 Hz、全帧校正后再 GMR 重定向；用**非因果平滑**保留时序精度与尖锐瞬态，更干净但动态上更难。
- **环境**：Smo / Dyn × Sim / Real 四组（仿真无 Joy-Con 故无 JDS）。每控制器在 5 首舞曲（3 Easy + 2 Hard）上各重复 3 次，共 **60 trials（4×5×3）**；实验中用**安全龙门架(gantry)**保护机器人。
- **被测控制器**（均用作者提供 checkpoint，且都未用过 Just Dance 数据）：
  - [[GMT]]: 用自适应采样 + 混合专家(MoE) 架构学统一动作跟踪策略；Smo 用"保持末帧"假设、Dyn 把未来参考帧纳入观测；
  - [[TWIST]]: 用人类动捕训练通用遥操作动作控制器；
  - [[Any2Track]]: 两阶段 RL，在通用 tracker 上叠加在线扰动自适应以应对真实条件。
- **人类基线**：世界排行榜分数 + 200 段用户动作的平均 JDS 与运动学指标（人类动作也 retarget 到机器人关节空间以消除关节构型偏差）。

### 关键实验结论

- **人机差距明确**：最佳控制器约人类 1/3 分数（2900 vs 8538）；所有控制器在 Hard 曲目上 JDS 显著下降（平均降 573 分），暴露高动态/复杂协调的挑战。
- **输入动态性的影响（6.1）**：Dyn 显著抬高 jerk/acc（平滑使 jerk 平均降 1268.7）；TWIST 在 Dyn 下大幅退化（SR -46.7%、MPJPE +23.9 mm、JDS -16.7%），而 GMT/Any2Track 稳定性受影响较小；GMT 在 Dyn 下 JDS 反而高于 Smo——因平滑会让执行动作与原编舞失配、抬高 MPJPE 降低分。**结论：平滑输入提稳定从而提分，动态输入更真实但易致失稳。**
- **Sim-to-Real（6.2）**：仿真 jerk/acc 更低（无关节背隙/扭矩限制/摩擦/时延噪声）；MPJPE 的 sim-real 差异远小于平滑度差异，暴露 **MPJPE 的局限**——可低 MPJPE 却抖动失衡。反直觉地，仿真成功率反而**低于**硬件，因为**安全龙门架**在关键时刻给了机器人轻微支撑（尤其全局漂移时帮助再稳定），对更敏感的 GMT 影响更明显，从而**轻微高估硬件成功率**。

---

## 批判性思考

### 优点
1. **真正可复现、可与人类同台对比**: 用全球统一标准的商业游戏 + 内置打分，绕开昂贵动捕，首次让"机器人 vs 人类"在同一打分体系下定量比较，工程门槛极低（约 \$400）。
2. **指标可靠性有扎实验证**: 不是直接拿来用，而是用 200 段人类动作做了有效性/区分度/可重复性的系统统计验证（含相关分析、ICC/CV/KCC），并诚实指出"单 Joy-Con 可被上身刷分"的缺陷。
3. **诚实揭示评测陷阱**: 明确指出 MPJPE 准而不稳的迷惑性、安全龙门架对成功率的污染、仿真高估鲁棒性等，对社区如何正确评测全身控制很有参考价值。

### 局限性
1. **打分仅依赖单 Joy-Con（单手 IMU）**: 原理上无法感知下半身与全身姿态，只能"假设控制策略忠实跟踪参考"来间接保证全身协调；这意味着 JDS 对"下身偷懒/失稳但上手摆对"的情形不敏感，作者自己也承认需视频佐证。
2. **基准被硬件约束裁剪**: 选曲需排除手触地与原地转身（Joy-Con 占手、脚不适合旋转），且实验普遍依赖安全龙门架——这既限制了动作多样性，又把"成功率"这一指标的物理意义打了折扣。
3. **样本与规模偏小**: 仅 5 首曲、每条件 3 次重复、10 名受试者；MoCap 端用单目 GVHMR（约 5 Hz、200 ms 延迟）重建，重建误差会直接成为参考动作误差并混入 MPJPE/DTW，但论文未充分量化这部分系统误差对结论的影响。

### 潜在改进方向
1. **更全的姿态计分**: 引入多传感器（多 Joy-Con / 视觉 / 全身 IMU）或视频复核，弥补单手 IMU 看不到下身的盲区，使 JDS 真正反映全身质量。
2. **去龙门架 / 标准化保护协议**: 量化甚至消除安全龙门架对成功率的影响，或制定统一的硬件保护标准，让 SR 在不同实验室间可比。
3. **更低延迟的 MoCap/retarget 与更大评测规模**: 提升 GVHMR/GMR 的频率与精度（降低参考误差），扩大曲库、受试者与机器人形态覆盖，并把"console-as-benchmark"扩展到运动模仿之外的更多交互式游戏环境。

### 可复现性评估
- [x] 流水线开源（基于开源的 [[GVHMR]] 与 [[GMR]]，项目主页 https://switch-justdance.github.io/）
- [ ] 预训练模型（被测控制器用各自作者提供的 checkpoint，本文未发布新权重）
- [x] 训练/实现细节完整（Appendix A 给出异步流式实现，Appendix C 给出选曲，硬件/频率/延迟均有交代）
- [x] 数据集可获取（《Just Dance》商业游戏全球可购，曲目与难度在 Table C1 列明）

---

## 速查卡片

> [!summary] Switch-JustDance: 用商业体感游戏给人形全身控制器打分的基准
> - **核心**: 把 Switch《Just Dance》改造成低成本(\$400)、可复现、能与人类同台比分的真实世界全身控制基准——Joy-Con 绑机器人手上，游戏内置 JDS 即评分。
> - **方法**: HDMI 采集 → [[GVHMR]] 单目重建 SMPL(约5Hz) → [[GMR]] 重定向 + LERP/SLERP 异步插值到 10–15Hz → 被测控制器执行；先用 200 段人类动作验证 JDS 的有效性/区分度/可重复性(ICC=0.735, KCC=0.678)。
> - **结果**: 在 [[Unitree G1]] 上评测 [[GMT]]/[[TWIST]]/[[Any2Track]]，最佳仅 2900 vs 人类 8538(约1/3)；平滑输入提稳定、动态输入暴露高频极限、仿真高估鲁棒性、MPJPE 准而不稳。
> - **项目**: https://switch-justdance.github.io/

---

*笔记创建时间: 2026-06-29*
