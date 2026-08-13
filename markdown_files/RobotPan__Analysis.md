---
title: "RobotPan: A 360° Surround-View Robotic Vision System for Embodied Perception"
method_name: "RobotPan"
authors: [Jiahao Ma, Qiang Zhang, Peiran Liu, Zeran Su, Pihai Sun, Gang Han, Wen Zhao, Wei Cui, Zhang Zhang, Zhiyuan Xu, Renjing Xu, Jian Tang, Miaomiao Liu, Yijie Guo]
year: 2026
venue: arXiv
tags: [humanoid, embodied-perception, panoramic-vision, multi-view-vision, lidar, 3d-reconstruction, gaussian-splatting]
image_source: online
---

# RobotPan 精读：360° 几何感知、3D Gaussian 与 Humanoid Tracking 路线评估

> 论文：RobotPan: A 360° Surround-View Robotic Vision System for Embodied Perception  
> 版本：arXiv:2604.13476v2，2026-04-26  
> 项目页：[RobotPan](https://robotpan.github.io/)  
> 论文全文：[arXiv HTML](https://arxiv.org/html/2604.13476v2)  
> 分析日期：2026-08-06  
> 分析目标：判断 RobotPan 提取了什么几何信息、这些信息是否进入 humanoid tracking 闭环，以及如何将其接入 System0 / BFM。

---

## 0. 先给结论

### 0.1 一句话总结

RobotPan 是一个部署在天工 3.0 人形机器人上的 **360° 多相机–LiDAR 几何感知与 3D Gaussian 重建系统**：它从六路标定图像预测 metric point map、confidence 和 dense feature，再通过球形体素聚合生成紧凑 3D Gaussian，并用流式融合维持长序列场景表示。

### 0.2 最重要的范围判断

> **RobotPan 不是 humanoid motion tracking / RL control 论文。**

它没有定义 reference motion、tracking policy、action、reward、PD/WBC 接口，也没有报告 joint tracking error、足端落点误差、跌倒率或控制成功率。它属于 humanoid tracking 技术栈中的 **上游几何感知模块**，尚未形成 geometry-conditioned tracking 闭环。

论文和项目页展示的天工 3.0 跳跃、Thomas flare、全身运动等视频，是机器人平台能力展示；论文没有提供证据证明这些动作由 RobotPan 几何表示驱动。项目页也将 RobotPan 的直接输出描述为 surround-view rendering、novel-view synthesis、metric depth estimation 和 reconstruction，而不是控制动作。[项目页](https://robotpan.github.io/)

### 0.3 综合评级

| 维度 | 判断 | 说明 |
| --- | --- | --- |
| 几何感知创新性 | 较高 | 球形体素先验与紧凑 Gaussian 解码适合环视、近密远疏的机器人视角 |
| 重建实验可信度 | 中高 | 多个公开 benchmark 和自建数据集，但部分结果并非全面领先 |
| Metric-scale 证据 | 中 | 有 LiDAR 监督，但 point-map 评估仍使用 Sim(3) + ICP 对齐 |
| 实时渲染能力 | 强 | 报告 230 FPS rendering |
| 实时场景更新能力 | 有限 | 0.47 秒/帧约等于 2.13 Hz，不等于实时控制频率 |
| Humanoid tracking 贡献 | 无直接贡献 | 没有 policy、action、reward 或闭环指标 |
| 对 System0 的直接可用性 | 低 | 3DGS 不能直接作为高频控制输入，需要 geometry adapter |
| 对规划/遥操作的可用性 | 中高 | 360° 可视化、场景建图、低频规划更匹配当前能力 |
| 复现可行性 | 当前较低 | 官方代码、数据和模型仍标注 Coming Soon |
| 精读建议 | 值得补充精读 | 适合作为上游感知路线，不应当作 tracking baseline |

### 0.4 最值得学习与最不应高估的内容

最值得学习：

1. **机器人中心的球形空间先验**：近场分辨率高、远场表示稀疏，与 360° 环视传感布局有较强结构匹配。
2. **从 point map 到紧凑 Gaussian 的结构化解码**：不是逐像素生成 Gaussian，而是先做球形体素聚合和 sparse 3D CNN。
3. **长序列动态/静态分离与增量融合**：避免逐帧简单累积导致 Gaussian 数量无限增长。

最不应高估：

1. **“实时”不能统一理解**：230 FPS 是渲染速度，流式更新为 0.47 秒/帧。
2. **“metric-scaled”没有被无对齐指标完全验证**：几何 benchmark 使用 Sim(3) 和 ICP 后处理对齐。
3. **“服务 navigation / loco-manipulation”是应用潜力，不是闭环控制实验结论**。

---

## 1. 论文生态位与问题定义

### 1.1 论文属于哪一类

| 类别 | 是否属于 | 证据 |
| --- | --- | --- |
| A. 几何感知模块 | 是 | 输出 point map、confidence、feature、spherical voxel anchor 和 3D Gaussian |
| B. 几何辅助规划 | 尚未证明 | 论文声称可服务 navigation，但没有 planner 输入或导航指标 |
| C. 几何条件 Tracking | 否 | 没有 tracking policy 和 control observation |
| D. 联合感知控制 | 否 | loss 只有视觉与几何项，没有 control objective |

准确定位是：

> **Feed-forward 360° metric reconstruction + generalizable novel-view synthesis + streaming 3DGS。**

### 1.2 它真正解决的问题

传统机器人视觉接口主要存在三类问题：

1. 前向相机视野有限，遥操作人员难以观察机器人周围环境。
2. 多路相机通常需要人工切换，打断遥操作和数据采集流程。
3. 人形机器人运动产生的视角抖动与模糊，容易造成 HMD 画面不稳定。

环形多相机又引入了新的几何问题：

- 六个相机光心不共置，不能使用普通单中心全景拼接。
- 相邻相机重叠有限，稀疏视角匹配困难。
- 下游机器人任务要求真实尺度，而普通视觉重建容易存在 scale ambiguity。
- 逐像素 Gaussian 数量大，不利于板载存储、传输和长序列融合。

RobotPan 的回答是：

1. 用六路同步、标定 RGB 相机提供 360° 覆盖。
2. 用 LiDAR 监督恢复 metric geometry。
3. 用多视图 Transformer 预测 point map。
4. 用 robot-centric spherical voxel 进行近密远疏的结构化聚合。
5. 每个球形体素解码固定数量的 3D Gaussian。
6. 用 shared Gaussian、dynamic Gaussian 和 tiny-MLP 维护长序列。

### 1.3 学术问题还是部署问题

它同时包含两部分：

- **学术核心**：稀疏多视图 feed-forward reconstruction、generalizable NVS、streaming 3DGS。
- **系统核心**：六相机 + 40 线 LiDAR 的机器人头部布局、同步标定、数据采集与流式表示。

但“部署”主要证明到视觉系统和渲染层。论文没有证明：

- perception-to-planning latency；
- perception-to-control latency；
- 对 locomotion / tracking 成功率的实际改善；
- 感知失效时的安全降级。

---

## 2. 系统信息流：论文走到了哪一步

### 2.1 论文实际实现的链路

$$
\{(I_i^t,P_i)\}_{i=1}^{V}
\rightarrow
\{\mathbf X_i^t,\mathbf S_i^t,\mathbf F_i^t\}_{i=1}^{V}
\rightarrow
\text{spherical voxel anchors}
\rightarrow
\mathcal G^t
\rightarrow
\text{rendering / reconstruction / streaming}
$$

其中：

- $I_i^t$：第 $i$ 个相机在时刻 $t$ 的 RGB 图像；
- $P_i=K_i[R_i|\mathbf t_i]$：相机投影矩阵；
- $\mathbf X_i^t$：metric-scaled point map；
- $\mathbf S_i^t$：confidence map；
- $\mathbf F_i^t$：dense feature map；
- $\mathcal G^t$：3D Gaussian 集合。

### 2.2 Geometry-to-control 应有但论文缺失的链路

一个完整的 geometry-conditioned tracking 系统还应包含：

$$
\mathcal G^t \text{ 或 } \mathbf X^t
\rightarrow
A_{\text{geo}}
\rightarrow
z_t^{\text{ctrl}}
\rightarrow
\pi_\theta(s_t,r_t,z_t^{\text{ctrl}})
\rightarrow
a_t
\rightarrow
\text{PD/WBC}
$$

RobotPan 没有定义：

- $A_{\text{geo}}$：几何到控制表示的 adapter；
- $s_t$：关节、IMU、base state 等 proprioception；
- $r_t$：参考动作或运动意图；
- $\pi_\theta$：tracking policy；
- $a_t$：关节位置、速度或力矩动作；
- PD、WBC、MPC 或 safety filter。

### 2.3 证据表

| 模块 | 输入 | 输出 | 频率/时延 | 是否进入控制闭环 | 证据等级 |
| --- | --- | --- | --- | --- | --- |
| 六路 RGB 相机 | 场景光照 | 同步图像 | 论文未报告采样率 | 否 | [论文明确] |
| 40 线 LiDAR | 360° 距离 | 稀疏 metric points | 论文未报告采样率 | 训练监督；在线输入角色不清楚 | [论文明确/部分缺失] |
| Geometry Transformer | 六路图像、相机标定 | point/confidence/feature maps | 未报告单次推理时延 | 否 | [论文明确] |
| Spherical voxel encoder | point、RGB、feature、view direction | anchor feature | 未单独报告 | 否 | [论文明确] |
| Gaussian decoder | anchor feature | 3DGS 参数 | 未单独报告 | 否 | [论文明确] |
| Streaming fusion | 历史与当前 Gaussian | shared + dynamic GS | 0.47 秒/帧 | 否 | [论文明确] |
| Renderer | 3DGS | 新视角图像 | 230 FPS | 只用于视觉输出 | [论文明确] |
| Tracking policy | 未定义 | 未定义 | 未定义 | 不存在 | [缺少证据] |
| 底层控制器 | 未定义 | 未定义 | 未定义 | 不存在 | [缺少证据] |

---

## 3. 传感器系统与数据采集

### 3.1 机器人头部布局

天工 3.0 头部包含：

- 六个向外安装的 RGB 相机；
- 相机位于半径约 89 mm 的圆环上；
- 朝向分别为 $0^\circ$、$\pm60^\circ$、$\pm120^\circ$ 和 $180^\circ$；
- 单相机水平 FOV 为 $118^\circ$，垂直 FOV 为 $92^\circ$；
- 顶部有 40 线旋转 LiDAR；
- LiDAR 水平覆盖 $360^\circ$，垂直 FOV 为 $59^\circ$。

这是一种“多光心环视”结构：覆盖范围大，但不同相机之间不满足单中心全景相机的简单投影模型。[论文 Sec. 3.1](https://arxiv.org/html/2604.13476v2)

需要特别区分“硬件系统含 LiDAR”和“RobotPan 网络在线使用 LiDAR”。论文给出的 feed-forward mapping 输入只有六路 RGB 图像及其投影矩阵；LiDAR 明确用于训练阶段的 sparse metric supervision。正文没有给出 LiDAR scan 作为网络在线输入的公式，也没有解释部署时是否另有 LiDAR 直接融合通路。因此，基于论文可以确认的是：

- [论文明确] LiDAR 用于采集 metric ground truth 和 fine-tuning；
- [论文明确] RobotPan 主网络的显式输入是 RGB + calibration；
- [缺少证据] 在线运行时 LiDAR 是否进入最终场景融合、控制或安全模块。

### 3.2 数据集

| 项目 | 论文信息 |
| --- | --- |
| 同步片段 | 339 clips |
| 每段长度 | 200 frames |
| 同步时刻总数 | 67,800 |
| 六相机图像总数 | 约 406,800 |
| RGB 原始分辨率 | $1920\times1536$ |
| 训练分辨率 | $518\times406$ |
| 传感器 | 六路 RGB + 一路 LiDAR |
| 场景 | 办公楼、家庭、展馆、工厂、街区、工业园和道路 |
| 动态场景比例 | 约 80% |
| 数据划分 | 80% / 10% / 10% |

### 3.3 数据域的关键风险

数据主要由平均身高约 160 cm 的人佩戴头部采集设备完成。为了保证数据一致性，作者加入颈部稳定器，并把行走速度限制在不超过 1.2 m/s、转向速度限制在不超过 0.4 rad/s。

这产生一个重要 domain gap：

| 采集域 | 真实 humanoid 部署域 |
| --- | --- |
| 人类头部 + 颈部稳定器 | 机器人头部与刚性躯干耦合 |
| 保守运动速度 | 跳跃、急停、摔倒恢复等高动态运动 |
| 人类步态抖动 | 电机、结构模态和脚地冲击引起的抖动 |
| 较少机器人自遮挡 | 手臂、手、携带物可能频繁遮挡相机 |
| 受控采集协议 | 突发旋转、强 motion blur、传感器丢帧 |

因此，数据集适合训练环视几何先验，但不能直接证明高动态 humanoid tracking 中的视觉鲁棒性。

---

## 4. 几何预测器

### 4.1 输出定义

RobotPan 的几何网络为：

$$
f_\theta:
\{(I_i^t,P_i)\}_{i=1}^{V}
\rightarrow
\{(\mathbf X_i^t,\mathbf S_i^t,\mathbf F_i^t)\}_{i=1}^{V}
$$

输出包括：

1. $\mathbf X_i^t\in\mathbb R^{H\times W\times3}$：每个像素/patch 对应的 3D point map。
2. $\mathbf S_i^t\in[0,1]^{H\times W}$：confidence map。
3. $\mathbf F_i^t\in\mathbb R^{H\times W\times C}$：dense feature map。

点图先位于各相机坐标系，再借助已知外参变换到 robot-centric frame。

### 4.2 网络结构

几何网络继承 VGGT / $\pi^3$ 风格：

- DINOv2 提取每张图像的 patch token；
- view-wise self-attention 在单视图内部建模；
- global self-attention 在全部视图之间交换信息；
- 总计 36 个 block：18 个 view-wise、18 个 global；
- 轻量 decoder 输出 point map、confidence 和 feature。

RobotPan 沿用 $\pi^3$ 的 permutation-equivariant 思路：

- 将六个视图视为无序集合；
- 去掉依赖固定参考视图的特殊 token；
- 去掉 view/frame-specific positional embedding；
- 相机标定通过 $P_i$ 注入；
- 对输入相机顺序不敏感。

### 4.3 这一设计的合理性

环视相机之间重叠有限。如果只做局部 stereo matching，后向视图和前向视图很难直接建立匹配。global attention 可以利用全局场景先验，而 permutation equivariance 可以降低“第一张视图选择不佳”导致的参考系偏置。

但它也更依赖 learned prior：

- 在无重叠或低纹理区域，几何结果不完全来自三角化；
- 网络可能根据训练分布“补出”看似合理的结构；
- 对碰撞检测而言，合理外观不等于保守几何；
- confidence 是否能准确标识 hallucinated geometry 非常关键。

### 4.4 Confidence 的使用存在说明缺口

论文明确输出 $\mathbf S_i^t$，但后续球形体素聚合公式主要使用：

- 3D position；
- RGB；
- dense feature；
- viewing direction；
- relative offset。

正文没有清楚说明 confidence 是否参与：

- point filtering；
- voxel aggregation weighting；
- Gaussian opacity；
- streaming update gating；
- downstream safety。

这是一个应由代码确认的实现细节。目前官方代码未发布，因此应标注为 **[缺少证据]**。

---

## 5. 球形体素与局部特征聚合

### 5.1 坐标转换

robot-centric 3D 点 $\mathbf x=(x,y,z)$ 被转换为：

$$
r=\|\mathbf x\|_2,\quad
\theta=\operatorname{atan2}(y,x),\quad
\phi=\operatorname{atan2}\left(z,\sqrt{x^2+y^2+\epsilon}\right)
$$

再离散为球形网格索引：

$$
i_r=\left\lfloor\frac{r-r_{\min}}{\Delta r}\right\rfloor
$$

$$
i_\theta=
\left\lfloor
\frac{\theta-\theta_0}{2\pi}N_\theta
\right\rfloor,\quad
i_\phi=
\left\lfloor
\frac{\phi-\phi_0}{\Delta\phi}
\right\rfloor
$$

### 5.2 为什么会近密远疏

球形体素体积近似为：

$$
V(r,\theta,\phi)
\approx
r^2\cos\phi\,
\Delta r\,\Delta\theta\,\Delta\phi
$$

当角分辨率和径向分辨率固定时，体素体积随 $r^2$ 增长。因此：

- 机器人附近：体素小、anchor 多、Gaussian 密；
- 远离机器人：体素大、anchor 少、Gaussian 稀。

这比 uniform Cartesian voxel 更自然地匹配环视感知的数据分布。

### 5.3 Anchor feature

每个点的属性由以下信息拼接：

$$
\mathbf a_n=
[\mathbf x_n,\ \mathbf c_n,\ \mathbf d_n,\ \mathbf f_n]
$$

其中 $\mathbf c_n$ 是 RGB，$\mathbf d_n$ 是视线方向，$\mathbf f_n$ 是 dense feature。

每个 voxel 的中心为：

$$
\bar{\mathbf x}_v=
\frac{1}{|\mathcal P_v|}
\sum_{n\in\mathcal P_v}\mathbf x_n
$$

使用 inverse-distance weight 聚合邻域：

$$
w_n=
\frac{
(\|\mathbf x_n-\bar{\mathbf x}_v\|_2+\epsilon)^{-1}
}{
\sum_{m\in\mathcal P_v}
(\|\mathbf x_m-\bar{\mathbf x}_v\|_2+\epsilon)^{-1}
}
$$

$$
\tilde{\mathbf a}_n=
\operatorname{MLP}
([\mathbf a_n,\mathbf x_n-\bar{\mathbf x}_v])
$$

$$
\bar{\mathbf a}_v=
\sum_{n\in\mathcal P_v}w_n\tilde{\mathbf a}_n
$$

随后把 anchor 放进 sparse tensor，使用 sparse 3D CNN 在相邻球形体素间传播信息。

### 5.4 对控制的意义与隐患

潜在价值：

- near-field capacity 更高；
- 360° 空间方向结构明确；
- sparse representation 比原始 point cloud 更容易压缩；
- anchor feature 可能比最终 3DGS 更适合作为 policy 输入。

主要隐患：

1. **近场是相对头部传感器，而不是相对足端或手端。** 地面接触区距头部约 1–2 m，不一定位于最细分辨率区域。
2. **球极畸变**：体素体积包含 $\cos\phi$，靠近上下极区的几何统计不同。
3. **方位角接缝**：$\theta=-\pi$ 与 $\theta=\pi$ 实际相邻，sparse convolution 是否做 periodic padding 未说明。
4. **固定角分辨率不等于固定控制误差**：远处误差对规划可能可接受，但窄台阶边缘、细杆和脚落点仍可能需要厘米级精度。
5. **Gaussian/rendering 最优分配不等于 collision 最优分配**：视觉上可接受的表面平滑和透明度误差，可能对接触控制不可接受。

---

## 6. 3D Gaussian 解码

### 6.1 参数化

对每个 spherical voxel，RobotPan 解码固定数量的 Gaussian。每个 Gaussian 包含：

- 均值 $\boldsymbol\mu$；
- opacity $\alpha$；
- covariance $\Sigma$；
- spherical harmonics color $\mathbf c$。

形式上：

$$
g_\phi:
\{(I_i^t,P_i)\}_{i=1}^{V}
\rightarrow
\{(\boldsymbol\mu_j^t,\alpha_j^t,\Sigma_j^t,\mathbf c_j^t)\}_{j=1}^{N}
$$

### 6.2 Gaussian center

中心由 anchor center 加 bounded offset：

$$
\boldsymbol\mu_v=
\bar{\mathbf x}_v+
\gamma
\left(2\sigma(\Delta\boldsymbol\mu_v)-1\right)
$$

bounded offset 可以防止 Gaussian center 远离对应体素，提高训练稳定性。

### 6.3 Scale 与 rotation

每个 Gaussian 的 scale 随半径变化：

$$
\mathbf s_v=
\kappa(r_v)
\exp
\left(
\ell_{\min}
+
(\ell_{\max}-\ell_{\min})
\sigma(h_s(\mathbf z_v))
\right)
$$

远处 Gaussian 由 $\kappa(r_v)$ 放大，与球形体素的近密远疏逻辑一致。旋转由 normalized quaternion 表达。

### 6.4 Appearance

DC spherical harmonic 由 voxel 的 pooled RGB 初始化，再预测 residual；高阶项直接由 feature 预测。

这让外观与观测颜色保持锚定，同时允许视角相关渲染。

### 6.5 3DGS 是否是合适的控制表示

结论是：**不应直接把完整 3DGS 作为 System0 的默认输入。**

| 属性 | 对渲染 | 对控制 |
| --- | --- | --- |
| Mean / covariance | 适合 splatting | 可近似表面，但不是显式 occupancy boundary |
| Opacity | 可表达可见性 | 不等于碰撞概率或实体性 |
| SH color | 很重要 | 对低层 tracking 通常价值有限 |
| View-dependent appearance | 很重要 | 可能增加无关维度 |
| Compactness | 有利 | 仍可能比局部 height/occupancy 重 |
| Differentiable rendering | 核心优势 | 除非做视觉端到端训练，否则价值有限 |

更合理的控制输入优先级：

1. metric point map / local point cloud；
2. confidence-filtered local occupancy；
3. local ESDF / signed distance；
4. foot-centric height patch；
5. hand / body collision distance feature；
6. spherical anchor latent；
7. 完整 3DGS 仅作为辅助全局场景 memory 或遥操作显示。

---

## 7. 训练目标

### 7.1 两类监督

RobotPan 使用四个主要 loss：

$$
\mathcal L=
\lambda_{\mathrm{mse}}\mathcal L_{\mathrm{rgb}}^{\mathrm{mse}}
+
\lambda_{\mathrm{lpips}}\mathcal L_{\mathrm{rgb}}^{\mathrm{lpips}}
+
\lambda_{\mathrm{pts}}\mathcal L_{\mathrm{points}}
+
\lambda_{\mathrm{nrm}}\mathcal L_{\mathrm{normal}}
$$

前两项监督外观，后两项监督几何。

### 7.2 Appearance loss

$$
\mathcal L_{\mathrm{rgb}}^{\mathrm{mse}}
=
\|\hat I_i^t-I_i^t\|_2^2
$$

$$
\mathcal L_{\mathrm{rgb}}^{\mathrm{lpips}}
=
\operatorname{LPIPS}(\hat I_i^t,I_i^t)
$$

MSE 保证像素一致性，LPIPS 更强调感知外观质量。

### 7.3 LiDAR point loss

将同步 LiDAR 投影到每个相机，得到稀疏 metric target：

$$
\mathcal L_{\mathrm{points}}
=
\frac{1}{\sum_{t,i}|\Omega_i^t|}
\sum_{t,i}
\sum_{p\in\Omega_i^t}
\frac{
\|s\hat{\mathbf x}_{i,p}^t-\mathbf x_{i,p}^{t,\mathrm{lid}}\|_1
}{
z_{i,p}^{t,\mathrm{lid}}+\epsilon
}
$$

特点：

- 只在 LiDAR 投影有效像素上监督；
- $1/z$ 权重提升近距离点的重要性；
- $s$ 是共享的 learnable scalar；
- 目标是把视觉 point map 对齐到 metric LiDAR geometry。

### 7.4 Normal loss

从相邻 point map 向量计算表面法向，并用角度误差对齐 $\pi^3$ 生成的 pseudo normal。

它可以改善局部平滑，但 pseudo normal 的错误也会成为 supervision noise。

### 7.5 与控制目标的错位

这些 loss 没有直接优化：

- 足端落点误差；
- collision false negative；
- penetration depth；
- contact normal；
- traversability；
- tracking error；
- fall rate；
- safety margin。

因此：

> 更高 PSNR 或更低 Chamfer Distance 不保证更低的 humanoid tracking error。

如果要服务 System0，应增加 task-aware geometry objective，例如：

$$
\mathcal L_{\mathrm{ctrl-geo}}
=
\lambda_{\mathrm{near}}\mathcal L_{\mathrm{near\ surface}}
+
\lambda_{\mathrm{edge}}\mathcal L_{\mathrm{step\ edge}}
+
\lambda_{\mathrm{occ}}\mathcal L_{\mathrm{collision}}
+
\lambda_{\mathrm{temp}}\mathcal L_{\mathrm{temporal}}
$$

并通过实际 tracking / collision 指标选择权重。

---

## 8. 两阶段训练与资源

训练分为两阶段：

1. 从 $\pi^3$ checkpoint 初始化；
2. 使用已知标定和稀疏 LiDAR 监督 fine-tune geometry predictor；
3. 启用 rendering component；
4. geometry 与 rendering joint training。

论文报告：

- AdamW；
- learning rate $5\times10^{-4}$；
- weight decay $10^{-4}$；
- gradient norm clip 1.0；
- bfloat16；
- gradient checkpointing；
- 8 张 GPU。

论文没有报告：

- GPU 型号和显存；
- batch size；
- 总训练步数；
- 训练时长；
- spherical voxel 的完整网格参数；
- loss 权重；
- 每个 voxel 的 Gaussian 数；
- 网络参数量；
- geometry-only 与 full pipeline 的 inference latency。

这些缺失项会显著阻碍论文级复现。

---

## 9. Streaming fusion

### 9.1 动态与静态分解

RobotPan 将时刻 $t$ 的场景表示为：

$$
\mathcal G^t=
\mathcal G_{\mathrm{shared}}
\cup
\mathcal G_{\mathrm{dyn}}^t
$$

- $\mathcal G_{\mathrm{shared}}$：跨帧持久的静态或共享内容；
- $\mathcal G_{\mathrm{dyn}}^t$：当前帧的动态内容。

### 9.2 动态区域识别

流程是：

1. 每个相机预测 motion / instance mask；
2. 利用 point map 把 mask lift 到 3D；
3. 转为 spherical range image；
4. 在 range-image domain 对多视图 mask 做 union；
5. 得到跨视图一致的动态区域。

### 9.3 新区域发现

把历史 Gaussian 投影到当前视图。如果当前图像存在历史模型无法解释的 hole，则从当前 point map 中生成新的 Gaussian。

### 9.4 Tiny-MLP refinement

每帧使用 tiny-MLP 对 shared Gaussian 预测 residual：

$$
\Delta\mathbf p=f_\theta^t(\mathbf x,\mathbf c)
$$

$$
\tilde{\mathbf p}^t=
\mathbf p\oplus\Delta\mathbf p
$$

residual 可作用于位置、shape、opacity 和 appearance。

### 9.5 方法上的关键疑问

1. **跨帧 ego-motion 如何进入统一坐标系没有被充分解释。** 论文详细描述了 rig 内相机外参，但对机器人/佩戴者运动时的全局位姿和 shared Gaussian 坐标管理说明不足。
2. motion / instance mask 由哪个模型产生、其延迟和错误率未明确。
3. range-image union 可降低 missed detection，但也可能扩大 false positive dynamic region。
4. 每帧 tiny-MLP 需要优化，说明“feed-forward streaming”仍包含 per-frame update cost。
5. shared Gaussian 与 dynamic Gaussian 的错误分配可能产生 ghosting 或删除静态结构。
6. 对长期闭环地图，200 帧实验仍然较短，无法证明小时级运行时内存和漂移受控。

---

## 10. 实验结果精读

### 10.1 Point-map estimation

RobotPan 自建数据集：

| 方法 | Overall，无 pose，越低越好 | Overall，有 pose，越低越好 |
| --- | ---: | ---: |
| VGGT | 0.496 | 0.362 |
| $\pi^3$ | **0.460** | 0.332 |
| RobotPan | 0.482 | **0.268** |

解读：

- 有 camera pose 时，RobotPan 最优，尤其 completeness 明显改善。
- 无 pose 时，RobotPan overall 不如 $\pi^3$。
- 结果支持“已知 rig calibration 对稀疏环视重建很重要”，不支持“RobotPan 在所有设置全面优于 $\pi^3$”。

公开 benchmark：

| 数据集 | RobotPan | $\pi^3$ | 判断 |
| --- | ---: | ---: | --- |
| DTU Overall | 3.0251 | 3.0785 | RobotPan 略优 |
| ETH3D Overall | 0.1841 | **0.1672** | $\pi^3$ 更优 |

因此更准确的措辞是“competitive”，而不是全面 SOTA。

### 10.2 Metric-scale claim 的证据边界

论文在训练时使用 LiDAR metric supervision，这是 metric-scale claim 的正面证据。

但 point-map benchmark 在评估前使用：

1. Umeyama Sim(3) alignment；
2. ICP refinement。

Sim(3) 会校正 scale、rotation 和 translation。于是 aligned Chamfer 指标无法单独证明：

- 原始输出尺度无偏；
- robot-frame 绝对位姿准确；
- 无后处理时可直接用于碰撞与控制。

应补充的 metric 指标：

- raw depth RMSE / AbsRel；
- raw point error，不做 Sim(3)；
- scale ratio error；
- robot-frame obstacle boundary error；
- calibration perturbation robustness；
- temporal metric drift。

### 10.3 Generalized novel-view synthesis

在 RobotPan 数据集：

| 方法 | Gaussian 数量 | PSNR | SSIM | LPIPS |
| --- | ---: | ---: | ---: | ---: |
| pixelSplat | 3,783K | 17.79 | 0.447 | 0.443 |
| MVSplat | 1,261K | 18.57 | 0.614 | 0.424 |
| FLARE | 1,261K | 18.29 | 0.625 | 0.393 |
| DepthSplat | 1,261K | 22.97 | 0.787 | 0.200 |
| RobotPan | **327K** | **24.70** | **0.811** | **0.197** |

RobotPan 相对 pixelSplat 的 Gaussian 数量约少 11.6 倍；相对更强的 DepthSplat 约少 3.9 倍，同时视觉指标更好。

在公开数据上则是混合结果：

- DL3DV：RobotPan PSNR 和 LPIPS 最好，SSIM 略低于 DepthSplat。
- RealEstate10K：RobotPan SSIM 最好，但 PSNR 略低于 DepthSplat，LPIPS 略低于最优水平。

这说明 spherical prior 对机器人环视分布尤其有效，但跨域优势不是所有指标全面成立。

### 10.4 Streaming NVS

| 方法 | Update / Train time，秒 | Render FPS | Storage，MB | PSNR |
| --- | ---: | ---: | ---: | ---: |
| StreamRF | 15.50 | 8.3 | 31.4 | 24.09 |
| 3DGStream | 12.25 | 215 | 7.8 | 26.11 |
| IGS | 3.65 | 204 | **6.5** | 27.75 |
| RobotPan | **0.47** | **230** | 7.2 | **28.59** |

RobotPan 在这组实验中实现了最好的 update time、render FPS 和 PSNR trade-off。

但必须注意：

$$
\frac{1}{0.47\ \mathrm{s}}
\approx
2.13\ \mathrm{Hz}
$$

所以：

- 230 FPS 是对已有 Gaussian scene 的渲染；
- 约 2.13 Hz 是场景表示更新上限；
- 这不能被解释为 230 Hz 几何感知；
- 对 20–50 Hz 的 tracking policy，2 Hz scene update 很可能过慢；
- 对低速遥操作显示、后台场景更新和全局规划则更可接受。

### 10.5 Ablation

Gaussian prediction：

| 表示 | Gaussian 数量 | Storage | PSNR |
| --- | ---: | ---: | ---: |
| Pixel-wise | 1,261K | 378 MB | 21.43 |
| Cartesian voxel-wise | 439K | 130 MB | 23.12 |
| Spherical voxel-wise | **327K** | **97 MB** | **24.70** |

这个消融较有说服力地支持 spherical voxel 对 **重建与渲染** 的价值。

Streaming fusion：

| 配置 | Gaussian 数量 | Storage | PSNR |
| --- | ---: | ---: | ---: |
| Naive concatenation | 65,400K | 19,400 MB | 24.66 |
| 无 range-image fusion | 3,094K | 1,371 MB | 24.71 |
| 无 tiny-MLP | 3,023K | 872 MB | 24.32 |
| Full model | 3,023K | 1,296 MB | **25.12** |

这支持：

- naive accumulation 不可扩展；
- range-image fusion 改善动态区域一致性；
- tiny-MLP 提升时间一致性与视觉质量。

但这些消融仍只证明 NVS，不证明 navigation、collision avoidance 或 tracking。

---

## 11. 实时性与部署成本

### 11.1 论文可确认的时延

| 环节 | 论文信息 | 控制意义 |
| --- | --- | --- |
| 相机采集 | 未报告 FPS | 无法建立 sensor latency budget |
| LiDAR 采集 | 未报告 Hz | 无法确认同步窗口 |
| Geometry Transformer | 未报告 latency | 最大的不确定项 |
| Spherical voxel + decoder | 未报告 latency | 无法确认是否可在线高频运行 |
| Streaming update | 0.47 秒/帧 | 约 2.13 Hz |
| Rendering | 230 FPS | 仅说明显示/渲染快 |
| Tracking policy | 不存在 | 无法评估 |
| PD / WBC | 不存在 | 无法评估 |

### 11.2 对 System0 的判断

假设 System0 / BFM 以 50 Hz 工作，周期只有：

$$
T_{\mathrm{policy}}=20\ \mathrm{ms}
$$

RobotPan 当前报告的 streaming update 为 470 ms，约为一个 policy 周期的 23.5 倍。

因此当前版本更合理的工作方式是：

- 几何线程：异步、低频更新；
- policy：使用最近一次 local geometry；
- proprioception：高频更新；
- low-level PD/WBC：500–1000 Hz；
- 对几何 feature 注入 age、confidence 和 timestamp；
- policy 训练时随机化 stale geometry；
- 对快速障碍物使用 LiDAR 直接通路，不等待 3DGS 更新。

### 11.3 不能直接回答的部署问题

论文没有报告：

- 板载 GPU 型号；
- 显存峰值；
- 端到端功耗；
- camera-to-display latency；
- camera-to-map latency；
- 网络是否与机器人控制共用计算单元；
- 丢帧和延迟抖动；
- ROS / middleware 接口；
- 长时间在线运行稳定性。

所以“practical real-time embodied deployment”目前对 rendering 成立得更强，对 closed-loop autonomy 成立得较弱。

---

## 12. 接触、力与物理一致性

### 12.1 RobotPan 没有直接建模什么

论文没有：

- 接触状态；
- 接触力；
- 摩擦系数；
- signed distance；
- traversability；
- foot support polygon；
- collision constraint；
- contact affordance；
- force/tactile input；
- robot dynamics。

### 12.2 3DGS 与物理表面的差异

3D Gaussian 是连续、带 opacity 的渲染 primitive。它可以产生视觉上连续的表面，但不天然等价于：

- 封闭 mesh；
- watertight collision geometry；
- TSDF / ESDF；
- 刚体表面；
- 可承载接触力的物理模型。

如果直接拿 Gaussian mean 做碰撞点，可能遇到：

- floater；
- 半透明边界；
- 表面厚度不明确；
- covariance 与物理法向不一致；
- 多层表面混叠；
- 动态物体残影。

### 12.3 对 humanoid tracking 的影响

对普通 free-space motion tracking，RobotPan 几何可能完全不需要。

对以下任务，必须把几何转换为物理友好表示：

| 任务 | 推荐表示 |
| --- | --- |
| 台阶/石块行走 | foot-centric height map + normal + confidence |
| 全身避障 | local occupancy / ESDF |
| 坐、靠、撑 | surface mesh / SDF + contact candidates |
| 手部接触 | local high-resolution point cloud + normal |
| 动态人群避障 | dynamic occupancy + velocity |
| 遥操作显示 | 3DGS / panoramic rendering |

---

## 13. 为什么论文尚未证明 Geometry-conditioned Tracking

要建立“几何改善 tracking”的因果链，至少需要：

$$
\text{Geometry quality}
\rightarrow
\text{Policy input}
\rightarrow
\text{Action change}
\rightarrow
\text{Tracking / safety gain}
$$

RobotPan 只证明了第一段。

缺失的关键实验：

1. 无几何输入的 tracking baseline。
2. GT geometry / privileged geometry 上界。
3. RobotPan point map 与 depth / height map / occupancy 的公平比较。
4. geometry latency、noise 和 dropout 消融。
5. tracking error、foot placement error、collision rate、fall rate。
6. 动态障碍物下的 closed-loop success。
7. calibration drift 对控制稳定性的影响。
8. high-dynamic humanoid motion 下的 geometry robustness。
9. confidence-aware policy 是否优于普通 policy。
10. 感知失败时能否退化到 proprioceptive tracker。

因此最准确的结论是：

> RobotPan 已经证明“可以生成紧凑、可流式更新的 360° 几何/外观表示”，但没有证明“该表示能够改善 humanoid tracking”。

---

## 14. 面向 System0 / BFM 的合理接入方案

### 14.1 不建议的方案

不建议把 327K 个 Gaussian 及 SH color 直接展平输入 50 Hz policy：

- 输入规模太大；
- appearance 信息与低层控制弱相关；
- Gaussian 数量和顺序可变；
- 动态更新只有约 2 Hz；
- opacity 不等于碰撞概率；
- policy 很难学习可靠的几何 safety boundary。

### 14.2 建议的信息流

$$
\text{RobotPan point/confidence/anchor}
\rightarrow
\text{local geometry adapter}
\rightarrow
\{
\text{height},\text{occupancy},\text{ESDF},\text{normal},\text{age},\text{confidence}
\}
\rightarrow
\pi_{\mathrm{System0}}
$$

### 14.3 分层输入

| 层 | 更新频率建议 | 输入 | 用途 |
| --- | ---: | --- | --- |
| 全局场景层 | 1–3 Hz | compact 3DGS / global occupancy | 遥操作显示、全局规划 |
| 局部几何层 | 10–20 Hz | local point/voxel/ESDF | 避障、落脚、接触候选 |
| Tracking policy | 50 Hz | geometry latent + proprio + intent | 几何条件动作生成 |
| Safety/WBC | 200–1000 Hz | joint state、contact、近场距离 | 约束投影与硬安全 |

RobotPan 当前 0.47 秒更新更接近第一层。若要支持第二层，应绕过完整 streaming 3DGS，直接使用：

- point map head；
- confidence-filtered LiDAR；
- spherical anchor feature；
- 小型 local occupancy decoder。

### 14.4 建议的 policy observation

$$
o_t=
[
q_t,\dot q_t,\text{IMU}_t,
a_{t-1},
r_{t:t+H},
z_t^{\mathrm{geo}},
c_t^{\mathrm{geo}},
\Delta t_{\mathrm{geo}}
]
$$

其中：

- $z_t^{\mathrm{geo}}$：压缩后的局部几何 latent；
- $c_t^{\mathrm{geo}}$：confidence；
- $\Delta t_{\mathrm{geo}}$：几何信息 age。

### 14.5 训练策略

建议在仿真中加入：

- geometry dropout；
- 50–500 ms latency randomization；
- 外参扰动；
- scale bias；
- point noise 和空洞；
- stale feature；
- 动态障碍物 false positive / false negative；
- GT geometry teacher → estimated geometry student；
- 无几何时退化到 proprioceptive tracker 的 curriculum。

### 14.6 安全兜底

RobotPan geometry 不应单独承担硬安全。更稳妥的结构：

1. policy 输出 nominal action；
2. local LiDAR / depth 产生保守 collision distance；
3. WBC / CBF / QP safety layer 做动作投影；
4. confidence 低或 geometry age 过大时降低速度；
5. 感知失效时切回站立、减速或纯 proprioceptive mode。

---

## 15. 与代表性路线的对比

为了避免把上游感知和闭环控制混为一谈，下面对比 RobotPan、典型 motion tracker、VideoMimic 和 MeshMimic。

| 维度 | RobotPan | 纯 Proprioceptive Tracker | VideoMimic | MeshMimic |
| --- | --- | --- | --- | --- |
| 核心目标 | 360° 重建与渲染 | 参考动作跟踪 | 视频到场景条件控制 | 视频到 terrain-aware motion learning |
| 场景几何 | point map + 3DGS | 通常没有 | dense scene geometry / mesh | reconstructed mesh |
| Reference motion | 没有 | 有 | 有 | 有 |
| Tracking policy | 没有 | 有 | 有 | 有 |
| Action | 没有 | PD target / torque 等 | 控制动作 | 控制动作 |
| 接触建模 | 没有 | 多为 reward 隐式建模 | 场景条件与物理仿真 | contact-aware retargeting + RL |
| 在线感知 | 是，但更新较慢 | 不依赖视觉 | 主要离线 real-to-sim | 主要离线 real-to-sim |
| 控制指标 | 没有 | tracking / fall | 真机任务与控制 | tracking / terrain interaction |
| 适合技术层 | Perception | System0 | 数据—控制完整流水线 | 数据—控制完整流水线 |

参考：

- [VideoMimic 官方项目页](https://www.videomimic.net/)
- [MeshMimic 官方项目页](https://meshmimic.github.io/)
- [MeshMimic 论文](https://arxiv.org/html/2602.15733v1)

### 15.1 研究哲学差异

RobotPan 的哲学：

> 先构造高质量、统一、紧凑的 360° 世界表示，未来由下游任务消费。

VideoMimic / MeshMimic 的哲学：

> 从视频恢复运动和场景，把二者放入物理仿真，训练能够实际追踪并与环境交互的策略。

前者更偏通用感知基础设施，后者更直接面向 humanoid tracking 与 contact interaction。

---

## 16. 开源代码与复现状态

### 16.1 官方状态

截至 2026-08-06，RobotPan 项目页仍显示：

- Code：Coming Soon；
- Data and Model：Coming Soon。

当前没有可核实的官方 RobotPan 代码仓库、公开数据下载或 checkpoint。[项目页](https://robotpan.github.io/)

也未找到可信的第三方完整复现。

### 16.2 可作为基础的官方仓库

| 仓库 | 类型 | 可复用内容 | 能否替代 RobotPan |
| --- | --- | --- | --- |
| [yyfz/Pi3](https://github.com/yyfz/Pi3) | $\pi^3$ 官方 | 初始化、point map、confidence、训练/评估分支 | 只能替代几何 backbone |
| [facebookresearch/vggt](https://github.com/facebookresearch/vggt) | VGGT 官方 | DINO + alternating attention、camera/depth/point heads、fine-tune 示例 | 可复现结构基础，不能替代 spherical Gaussian |
| [donydchen/mvsplat](https://github.com/donydchen/mvsplat) | MVSplat 官方 | feed-forward 3DGS、config、pretrained model | 可参考 NVS decoder |
| [graphdeco-inria/gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting) | 原始 3DGS 官方 | renderer、Gaussian 参数化与评估 | 只能提供底层 Gaussian 工具 |

### 16.3 最可信与最适合动手的替代

- 最接近 RobotPan geometry initialization：$\pi^3$。
- 最成熟的多视图 geometry backbone：VGGT。
- 最适合 feed-forward Gaussian 起步：MVSplat。
- 最成熟的 Gaussian renderer：原始 3DGS。

但不存在一个仓库同时提供：

- 六相机–LiDAR 标定输入；
- metric LiDAR fine-tuning；
- spherical voxel aggregation；
- sparse 3D CNN；
- RobotPan Gaussian heads；
- dynamic/static range-image fusion；
- per-frame tiny-MLP streaming。

所以这些只能组成“手工复现积木”，不能视为 RobotPan 替代实现。

### 16.4 官方仓库质量审查

由于仓库尚未发布，以下项目无法审查：

| 项目 | 状态 |
| --- | --- |
| README 与安装 | 不可审查 |
| Python / CUDA / PyTorch 版本 | 未知 |
| 训练脚本 | 未公开 |
| 配置系统 | 未公开 |
| 数据预处理 | 未公开 |
| 标定工具 | 未公开 |
| Checkpoint | 未公开 |
| Evaluation script | 未公开 |
| Streaming implementation | 未公开 |
| ROS / 真机接口 | 未公开 |
| License | 未知 |

总体判断：

> **当前属于“论文可读，但官方实现不可复现”的状态。**

---

## 17. 真实复现难度

| 维度 | 难度 | 原因 |
| --- | --- | --- |
| 理解方法 | 中 | 数学清楚，但部分工程细节缺失 |
| 环境搭建 | 高 | 未给依赖、CUDA、sparse conv 和 renderer 版本 |
| Geometry backbone | 中 | 可从 $\pi^3$ / VGGT 起步 |
| 六相机数据 | 高 | 需要同步、标定、有限重叠环视 rig |
| LiDAR supervision | 高 | 需要准确时间同步与外参 |
| Spherical voxel | 中高 | 需要处理 sparse tensor、wrap-around 与聚合 |
| Gaussian decoder | 中高 | 多参数 head 与 differentiable renderer |
| Streaming fusion | 高 | 动态 mask、共享坐标、hole detection、tiny-MLP |
| 论文指标对齐 | 高 | 数据未公开，关键超参数缺失 |
| 实时部署 | 高 | 没有端到端 latency 和板载配置 |
| 接入 tracking | 高 | 论文完全没有 control interface |
| 真机闭环 | 很高 | 需要额外 policy、安全与硬件栈 |

困难的本质：

- 算法难度：中高；
- 工程难度：高；
- 数据难度：高；
- 控制接入难度：高；
- 最大障碍：代码、数据、模型和运行参数尚未发布。

---

## 18. 分层复现方案

### Level 1：最小几何复现

目标：

- 用公开 $\pi^3$ 或 VGGT 在多视图图像上输出 point map；
- 使用已知相机内外参转换到 robot-centric frame；
- 可视化 confidence-filtered point cloud。

最小依赖：

- 一张 24 GB 以上 NVIDIA GPU；
- PyTorch + CUDA；
- $\pi^3$ / VGGT checkpoint；
- 六路同步或近同步图像；
- 相机内参与 rig 外参。

成功标准：

- 六路 point map 能够在同一 robot frame 中基本对齐；
- 近场平面、墙面和障碍物没有明显尺度错乱；
- 能测量 raw metric error，而不是只看渲染。

### Level 2：球形体素 + Gaussian

目标：

- 实现 RobotPan 的 spherical voxel aggregation；
- 实现 sparse 3D CNN 和 Gaussian heads；
- 在公开 NVS 数据或自采小数据上训练。

关键任务：

1. 实现 $(r,\theta,\phi)$ voxelization。
2. 处理 $\theta$ 周期边界。
3. 实现 inverse-distance aggregation。
4. 集成 sparse convolution。
5. 解码 mean、opacity、scale、quaternion 和 SH。
6. 接入 differentiable Gaussian renderer。
7. 实现 RGB、LPIPS、point 和 normal loss。

成功标准：

- 相比 pixel-wise Gaussian，Gaussian 数量显著降低；
- PSNR / SSIM / LPIPS 不明显下降；
- 近场 geometry error 优于 Cartesian voxel。

### Level 3：Streaming fusion

目标：

- 在 200 帧、六视图动态序列上维护 shared + dynamic Gaussian；
- 避免 memory 随时间线性爆炸。

关键任务：

- motion / instance mask；
- 3D range-image fusion；
- ego-motion / global frame 管理；
- unseen-region detection；
- tiny-MLP refinement；
- per-frame storage 和 update time 测量。

成功标准：

- 对 200 帧序列，Gaussian 数量和 storage 显著低于 naive concatenation；
- 没有明显 ghosting；
- 更新速度接近论文量级。

### Level 4：Geometry-to-Tracking 离线验证

目标：

- 不直接上真机，先证明 RobotPan geometry 对控制确实有信息增益。

对比：

1. 无几何；
2. GT height / occupancy；
3. noisy depth；
4. RobotPan point map；
5. spherical anchor latent；
6. 3DGS 转 occupancy。

指标：

- tracking error；
- foot placement error；
- collision rate；
- fall rate；
- unseen terrain success；
- geometry latency sensitivity。

### Level 5：System0 真机接入

目标：

- 异步运行 geometry front-end；
- 50 Hz tracking policy 使用 local compact feature；
- 高频 WBC / safety layer 兜底。

顺序：

1. 静态站立 + 障碍物感知；
2. 低速平地行走；
3. 单台阶；
4. 多障碍；
5. 动态障碍；
6. 接触丰富动作。

停机条件：

- geometry age 超阈值；
- confidence 大面积下降；
- robot-frame map 跳变；
- predicted collision distance 突变；
- tracking error 或 base tilt 超限。

---

## 19. 复现计划表

| 阶段 | 目标 | 关键任务 | 主要风险 | 成功标准 | 预计产出 |
| --- | --- | --- | --- | --- | --- |
| 第 1 阶段 | Geometry backbone 跑通 | $\pi^3$/VGGT、标定、point map | 尺度和多视图对齐 | robot-frame point cloud | baseline 脚本 |
| 第 2 阶段 | Spherical representation | voxel、sparse CNN、Gaussian heads | 球极/接缝、显存 | compact 3DGS | 最小复现 |
| 第 3 阶段 | Metric fine-tune | LiDAR 投影、point/normal loss | 同步与外参误差 | raw metric error 收敛 | metric model |
| 第 4 阶段 | Streaming | dynamic mask、global frame、tiny-MLP | ghosting、内存增长 | 200 帧稳定更新 | streaming demo |
| 第 5 阶段 | Control adapter | occupancy/ESDF/height encoder | 信息损失、时延 | 仿真 tracking 提升 | geometry-conditioned policy |
| 第 6 阶段 | 真机验证 | 异步线程、safety fallback | 失稳和碰撞 | 分级任务安全通过 | 真机报告 |

### 19.1 资源估计

论文只明确写了 8 张 GPU，没有写型号和训练时长。以下为经验估计：

- Level 1 inference：单张 24 GB GPU 可尝试；
- geometry fine-tune：建议 4–8 张 40–80 GB GPU；
- 完整 joint training：更接近 8 张 A100/H100 级别；
- 本地数据：数 TB 级存储预算更稳妥；
- 最耗算力：多视图 Transformer + rendering joint training；
- 最耗人工：标定同步、streaming 坐标系和 dynamic/static 分解；
- 最难对齐：作者自建数据集上的论文指标。

复现优先级：

- 在官方代码发布前：**低到中**；
- 官方代码与数据发布后：**中**；
- 若目标是直接改进 System0 tracking：优先做 point/occupancy adapter，而不是完整复现 3DGS rendering。

---

## 20. 建议补做的关键实验

### 20.1 Geometry 到控制的因果消融

| 实验 | 假设 | 关键指标 |
| --- | --- | --- |
| 无 geometry vs RobotPan | 几何改善复杂地形 tracking | tracking error、fall rate |
| GT geometry vs RobotPan | 感知误差距离上界有多远 | success gap |
| Point map vs 3DGS vs occupancy | 控制不一定需要 appearance | 控制收益/算力 |
| 0–500 ms latency | stale geometry 是否导致失稳 | collision、fall |
| Calibration perturbation | 外参漂移的敏感性 | robot-frame error |
| Confidence gating | confidence 是否能降低灾难失败 | tail failure |
| High-dynamic gait | 训练域是否覆盖机器人冲击抖动 | geometry dropout |
| Dynamic obstacle | streaming dynamic mask 是否控制可用 | avoidance success |

### 20.2 Metric-scale 实验

必须补充不做 Sim(3) / ICP 的：

- raw point RMSE；
- depth AbsRel；
- scale error；
- obstacle boundary error；
- floor height error；
- footstep region error；
- 10 分钟以上 temporal drift。

### 20.3 计算代价实验

分别测量：

- camera capture → image tensor；
- Transformer inference；
- point-to-spherical voxel；
- sparse CNN；
- Gaussian decode；
- streaming update；
- renderer；
- geometry adapter；
- policy inference。

只报告 renderer FPS 不足以支持闭环部署。

---

## 21. 五个组会批判问题

1. 论文强调 metric-scaled reconstruction，但 point-map benchmark 为什么仍需要 Sim(3) + ICP？原始 robot-frame metric error 是多少？
2. 0.47 秒/帧的 streaming update 如何支持论文所暗示的实时 autonomous operation？230 FPS rendering 是否被过度延伸为实时感知？
3. 六相机和 LiDAR 的在线输入角色是什么？LiDAR 只用于训练监督，还是部署时也参与推理和融合？
4. shared Gaussian 的跨帧全局坐标如何处理 humanoid ego-motion？长期运行时的位姿漂移和 loop closure 怎么处理？
5. 为什么没有任何 navigation、collision avoidance、tracking 或 loco-manipulation 闭环指标，却把系统描述为这些任务的 practical interface？

---

## 22. 最终判断

### 22.1 最值得学习的三点

1. 用球形坐标结构匹配机器人环视传感布局，而不是机械套用 Cartesian voxel。
2. 先聚合再解码 Gaussian，以表示结构减少冗余。
3. 长序列中显式区分 shared / dynamic content，避免 naive accumulation。

### 22.2 最不值得高估的三点

1. 230 FPS rendering 不等于 230 Hz perception。
2. 视觉重建质量不等于物理碰撞和接触几何质量。
3. 部署在 humanoid 上不等于已经参与 humanoid control。

### 22.3 对 tracking 路线的价值

| 问题 | 结论 |
| --- | --- |
| 是否值得精读 | 值得，但应按 perception paper 精读 |
| 是否值得完整复现 | 当前谨慎；等待官方代码和数据 |
| 是否值得借鉴 | 值得借鉴 point map、confidence 和 spherical anchor |
| 是否直接进入 System0 | 不建议直接输入完整 3DGS |
| 最合理用途 | 低频全局场景、遥操作显示、局部 geometry adapter 的上游 |
| 是否证明 tracking 改善 | 没有 |

### 22.4 最终一句话

> **RobotPan 已经提供了一条有潜力的 360° metric geometry 感知路线，但目前只是“可能为 Tracking 提供几何输入”，不是“已经通过几何信息改善了 Tracking”。**

---

## 参考资料

1. [RobotPan 项目主页](https://robotpan.github.io/)
2. [RobotPan arXiv 摘要页](https://arxiv.org/abs/2604.13476)
3. [RobotPan arXiv HTML 全文](https://arxiv.org/html/2604.13476v2)
4. [$\pi^3$ 官方代码](https://github.com/yyfz/Pi3)
5. [VGGT 官方代码](https://github.com/facebookresearch/vggt)
6. [MVSplat 官方代码](https://github.com/donydchen/mvsplat)
7. [3D Gaussian Splatting 官方代码](https://github.com/graphdeco-inria/gaussian-splatting)
8. [VideoMimic 官方项目页](https://www.videomimic.net/)
9. [MeshMimic 官方项目页](https://meshmimic.github.io/)
10. [MeshMimic 论文](https://arxiv.org/html/2602.15733v1)
