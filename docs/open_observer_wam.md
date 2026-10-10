# FADO-WAM：CVPR 2027 研究与实施计划

> 更新日期：2026-10-07。本文为项目当前研究计划，替代旧 joint-action / all-as-ego 方向。研究设想、代码接入、实验验收分开记录；训练轮数不代表科学结论成立。
>
> 本次更新范围是 docs，不修改代码、数据、权重或运行中的训练。下一步执行顺序见第 6 节，待修问题和验收见第 7 节。本文保留旧计划有效主线，并整合 10 月 5 日讨论与 10 月 7 日代码复核。

## 0. docs 中哪些是当前计划，哪些是继承资料

| 文件 | 归属 | 用途与边界 |
| --- | --- | --- |
| [open_observer_wam.md](open_observer_wam.md) | 当前 FADO-WAM 计划 | 研究定位、数据合同、实验、修复顺序和验收依据 |
| [agents.md](agents.md) | 继承的 NAVSIM 文档 | Agent 接口参考，不代表 FADO 已有完整无 GT 规划推理 |
| [install.md](install.md) | 继承的 NAVSIM 文档 | OpenScene/nuPlan 环境背景，当前项目路径见第 3 节 |
| [splits.md](splits.md) | 继承的 NAVSIM 文档 | navtrain/navtest 说明，不能替代本项目数据划分 |
| [metrics.md](metrics.md) | 继承的 NAVSIM 文档 | PDM 评分背景，不代表 SPD 已具备相同地图、缓存和评分条件 |
| [cache.md](cache.md) | 继承的 NAVSIM 文档 | NAVSIM metric cache 说明，不是本项目数据审计报告 |
| [submission.md](submission.md) | 继承的 NAVSIM 文档 | 旧驾驶榜单提交流程，不是 CVPR 论文投稿流程 |

继承文档保留供查阅；不将其内容计入本项目已完成实验，也不据此启动无关数据下载或榜单提交。

## 1. 保持不变的研究主线

### 1.1 题目与命名

> **FADO-WAM: Factorizing Action Dynamics and Observations for Cooperative Driving**
>
> **Distributed observations, shared dynamics.**

论文和项目名称统一为 FADO-WAM。旧名称 OpenObserver-WAM / OO-WAM 不再作为论文题目；代码中的 `open_observer_*`、`OpenObserverWAM`、`oo_*` 暂时保留，不为命名进行无关重构。

主故事是 **观察推断与动作动力学解耦**。观察集合、数量、角色变化是验证轴，不以“观察者身份不变”单独作为创新。sim-to-real 为次要验证，不放到标题或第一贡献。

### 1.2 核心问题与模型定义

车辆和 RSU 的观测决定当前世界状态知道多少、哪里不确定；在状态足够充分时，动作条件转移模型应能够复用：

$$
(S_t,U_t)\sim q_\phi(S_t,U_t\mid O_t^{\mathcal C},P_t^{\mathcal C},r^{\mathcal C},d^{\mathcal C}),
$$

$$
\hat S_{t+1:t+H}\sim p_\theta(S_{t+1:t+H}\mid S_t,A_t).
$$

其中 $\mathcal C$ 是观察集合，$P$ 是位姿/标定，$r,d$ 是节点角色和域信息，$A_t$ 是 query ego 候选动作。观察元数据只用于状态推断，不另设原始融合特征或域 token 旁路进入 dynamics。不确定性是否作为显式 dynamics 输入，由状态合同确定；当前实现还没有完成这一扩展。

这是需要实验支持的建模假设。状态若遗漏速度、交互线索、道路条件或不确定性，复用假设可能不成立；天气也可能影响真实动力学。配对天气实验只验证本地数据中的受控外观/能见度变化。

相同状态和动作重复调用同一个确定性网络得到同一输出，只是接口检查，不能证明方法优于基线或建立因果不变性。

### 1.3 WAM 与下游规划

保留 action–world–action 闭环：

$$
A^0=\operatorname{Planner}(S_t),\qquad
\hat S^+=\operatorname{Dynamics}(S_t,A^0),\qquad
A^1=\operatorname{Refiner}(A^0,\hat S^+,U_t).
$$

核心输出是结构化 future occupancy/flow、actor 状态及 query ego 轨迹；规划是下游证据。一次只为当前 query ego 规划，其他车辆提供观察，不做所有车辆联合规划。

不新增 RGB 视频生成、FID/FVD 主指标、通信压缩、语言协商、多轮博弈或大规模自采数据。

### 1.4 数据集与 VLA/VLM 动作头不要混用

保留最近补充的术语约定：

- **VLA4CoDrive** 是仿真数据集全称。论文和研究文档不把它简称为 VLA。
- **VLA** 在继承的 DriveWorld / ImagineWorld 路径中指视觉语言动作模型；ReCogDrive 对应代码中的 VLM/ImagineWorld 路径。这些继承代码仍在仓库。
- 当前 FADO-WAM 入口没有接这个大模型，使用较小的轨迹 planner，以匹配状态/动力学对比的规模。
- CLI 的 `--source vla` 是兼容参数名，表示数据集 VLA4CoDrive；不表示启用了 VLA/VLM 动作头。

## 2. 新旧计划对照与实现状态

### 2.1 保留、修订、新增和未完成的区别

| 类别 | 内容 | 当前处理 |
| --- | --- | --- |
| 旧计划保留 | observation–dynamics factorization、V2X-WAM 主对标、结构化世界输出 | 继续作为论文主线 |
| 旧计划保留 | 同输入比较、oracle、subset/天气实验、规划、迁移、多 seed | 仍必需，代码接口存在不等于实验完成 |
| 旧计划修订 | OpenObserver-WAM、身份不变措辞 | 使用 FADO-WAM 与观察—动力学解耦 |
| 旧计划修订 | 从 30 点数组抽样、默认跨数据 4 s | 共同 0.5–2.5 s；SPD 专用 4 s 配置单列、待核对原文协议 |
| 旧计划修订 | 默认多 town、完整世界监督和 HD map | 按本地单 town、部分 actor 标签、实际模态能力约束 |
| 旧计划修订 | 融合 latent 只加辅助头即可作为主瓶颈 | rollout 经过明确结构化接口，并统一 predicted/oracle 数值语义 |
| 旧计划修订 | 第 1–12 周线性排期 | 改为依赖验收的 G0–G5 顺序 |
| 新增合同 | observer/actor/ego 分离，已占用/已空闲/未知三态 | 第 3 节 |
| 新增验收 | 真实坐标直接投影与组合变换、末步/缺标 flow | 第 7.1 节 |
| 新增验收 | 同路径 action 对照、固定轨迹身份、N/A 指标、有效支持集 | 第 7.2 节 |
| 新增工程要求 | weather 逐样本 mask、DDP、EMA/恢复与 run manifest | 第 7.3 节 |
| 明确未完成 | 正式基线、equal-evidence、规划安全、sim-to-real 曲线 | 作为后续关卡，不回填“已完成” |

### 2.2 截至 2026-10-07 的代码状态

**已接入**指代码存在，**待验证**指缺少验收，**待修复**指仍有具体问题。以下为本次读取时的快照。

| 项目 | 状态 | 证据或限制 |
| --- | --- | --- |
| 项目/数据/权重路径 | 已接入 | [workspace_paths.py](../navsim/agents/ImagineWorld/configs/workspace_paths.py) |
| 共同时间轴、VLA4CoDrive 边界过滤、SPD 官方 split | 已接入 | [open_observer_dataset.py](../navsim/planning/training/open_observer_dataset.py)；SPD 仍是最近邻近似 |
| actor 与 observer 分离 | 已接入 | [agent 的 _world_target](../navsim/agents/ImagineWorld/open_observer_agent.py) 使用 actor 框，RSU 不作为目标 |
| occupancy/flow 结构瓶颈 | 已接入 | [canonical_state / forward](../navsim/agents/ImagineWorld/open_observer_wam.py) 不再直接 rollout 原始融合 latent |
| VLA4CoDrive 坐标一致性 | 待修复 | 10 月 7 日真实样本仍复现约 16.93 m 组合误差 |
| 标签支持集、oracle 与主分支语义 | 待修复 | 主分支只屏蔽到正例，oracle 仍全图监督 |
| flow 时间、末步、缺标处理 | 待修复 | SPD 仍有末步零速度和无效目标差分 |
| actor 未来指标及 world 阶段规划指标隔离 | 部分接入 | 已分开当前定位/未来误差；未来轨迹身份匹配仍待修 |
| 非 ego 动作对照 | 部分接入 | 对照仍跳过正常路径中的 residual，不能直接解释为 action 效果 |
| visibility/uncertainty | 待修复 | 网格覆盖仍被用作可见性，不能代表遮挡或校准 |
| worker 内图像 item cache | 已调整 | 当前 loader 已移除持续增长的缓存 |
| DDP weather 分支、步数与 timeout | 已调整、待验收 | 各 rank 均 forward、步数使用 ceil；weather loss 仍按 batch 平均有效率缩放 |
| 正式公平比较、规划安全与迁移 | 未完成 | 当前可见产物不足以支持论文主表 |

历史 [audit/report.json](../../fado-wam_workspace/exp/open_observer_wam/audit/report.json) 生成于 10 月 5 日。它只包含样本计数、部分时间抽查和首样本 BEV 图，不是 10 月 7 日的几何/标签完整验收。此前 19 项测试通过也未覆盖上述反例；本次没有将旧测试结果改标为今天通过。

10 月 7 日读取到的训练命令包含：

```text
--stage world --source vla --devices 3 --epochs 400
--batch-size 78 --lr 1.275e-4 --resume
--output-dir .../exp/open_observer_wam/world_joint
```

这只是运行快照：`source=vla` 表示仅训练 VLA4CoDrive，目录 `world_joint` 或 `protocol=joint` 都不能说明 SPD 已参与训练。训练完成多少 epoch 不改变 G0/G1 尚未整体通过的状态。本次文档更新不操作该进程。

## 3. 数据、动作与监督合同

### 3.1 路径与本地能力

项目路径：`/mnt/nvme1n1/LCX2/nyc/FADO-WAM`。

| 内容 | 路径 | 本地能力 |
| --- | --- | --- |
| VLA4CoDrive | [data/VLA4CoDrive](../../data/VLA4CoDrive) | Town10HD、8 天气、3 车；9 个 scene 为连续时间分块 |
| SPD train | [V2X-Seq-SPD-train](../../data/V2X-Seq-SPD/V2X-Seq-SPD-train) | 元数据 10,761 对协同帧、67 序列；含 train/val 标注 |
| SPD test | [V2X-Seq-SPD-test](../../data/V2X-Seq-SPD/V2X-Seq-SPD-test) | 元数据 4,610 对协同帧、28 序列；未见本地 cooperative GT |
| 权重、运行与审计 | [fado-wam_workspace](../../fado-wam_workspace) | 新旧协议日志/权重需分别记录 |

当前实际输入以图像为主；loader 的 LiDAR 为占位，HD map/drivable-area 尚未接入。不能从数据集支持的模态或继承文档推断当前模型已使用它们。

### 3.2 两套数据的分工

1. **SPD 优先建立真实主实验**：官方 train/val 上的交通 actor 世界预测、同输入比较、vehicle/RSU dropout。test GT 或官方评分入口未落实前，不报告正式 test 世界精度。
2. **VLA4CoDrive 用于多车/配对天气与辅助预训练**：同一时刻构造 {A}、{A,B}、{A,C}、{A,B,C}，固定 query A 与相同 action，再轮换 query。
3. VLA4CoDrive 当前监督只覆盖三辆协同车的状态/轨迹，尺寸为固定近似，不能称为完整交通世界 GT。没有可靠负例时，先做明确范围的 actor 辅助任务。
4. SPD 当前最多保留 16 个 actor，必须报告截断和空间覆盖。未保留目标所在位置不能直接当作空闲。
5. `observer` 是观测来源，`actor` 是预测目标，`ego` 是当前查询车辆；RSU 仅属于 observer。

### 3.3 时间和动作

共同 action 表示为 query 当前坐标系下的相对位姿序列：

$$
A_t=[(x_k,y_k,\psi_k)]_{k=1}^H.
$$

它是相对于当前 ego 的未来位姿，不是未说明的逐步位移，也不直接混合两套数据不同的低层控制标签。

| 配置 | 时间 | 用途 |
| --- | --- | --- |
| `--protocol joint` | 0.5、1.0、1.5、2.0、2.5 s | 两套数据共同开发/迁移协议 |
| `--protocol spd_benchmark` | 0.5 至 4.0 s，共 8 步 | 当前仓库 SPD 专用配置；尚待核对 V2X-WAM 正式设置 |
| history | -1.5、-1.0、-0.5、0 s | 必须全部早于或等于当前时刻 |

`--source vla/spd/joint` 独立选择数据来源；共同时间协议不代表混合数据。

- 从实际记录位置和时间戳采样，不按 30 点数组索引近似秒数。VLA4CoDrive 当前共同时刻精确匹配已接入。
- SPD 最近邻当前允许 0.12 s 误差，尚非严格等时重采样。正式实验选择插值到目标时刻，或显式记录实际 lead time/dt 并统一建模与评测。
- 明确 flow 为相邻时刻位移（m）还是速度（m/s）。不得在实际间隔不同时无条件除以 0.5。
- 缺少下一时刻 actor 标签时，该 flow 无效，不能用回填当前位置造成倒退位移。
- 末步缺少后继帧时定义后向差分或有效 mask，不能默认移动目标速度为零。
- 2.5 s 与 4 s head/checkpoint 不无检查地互换。

### 3.4 划分

- SPD 读取 [官方 split](../../data/V2X-Seq-SPD/V2X-Seq-SPD-train/split/cooperative-split-data-spd.json)，历史/当前/未来实际帧全部留在指定集合。旧编号取模划分终止。
- VLA4CoDrive 当前 scene1–6/train、7/val、8–9/test；跨天气一致，删除整个历史/未来窗口越界样本。应称 temporal holdout，不宣称独立路线泛化。
- 本地只有一个 town，不执行或承诺 held-out town。held-out weather 必须另设训练天气集合，默认所有天气参与训练不是天气外推实验。
- 历史共同协议审计数：VLA4CoDrive 12,480/2,080/4,160；SPD 5,431/2,472/3,489。计数随协议和过滤规则变化，修改后重新生成。
- 审计实际 frame ID、时间支持范围和 replay 分组，不只检查集合名称。

### 3.5 坐标与状态定义

所有观察者共享一个明确的右手度量地面坐标系；各车当前位姿、relative pose、历史/未来轨迹、actor 框从相同定义推导。

不得将每车独立 PCA 坐标系直接当成只相差 SE(2)；不得混合 window PCA 和 scene PCA；推理坐标定义不能依赖未来运动。

每个真实样本必须满足：

```text
原始 future 位置 -> 直接投影到 query frame
                ≈ 本车局部 future -> 相对位姿 -> query frame
```

另检查三车变换闭环、正逆变换、不同 query、停车、转弯和窗口边界。普通旋转逆变换测试不能替代真实数据几何验收。

### 3.6 占用与三类 mask

每格标签分为 **已知占用、已知空闲、未知**，并区分：

- `grid_support`：坐标变换后落在网格范围内；
- `sensor_visibility`：传感器视野和遮挡下实际可见；
- `label_known`：标注足以支持正/负监督的区域。

它们不能互相替代。主预测、alternate、oracle、天气分支、一致性与指标使用语义一致的支持集。

VLA4CoDrive 没有可靠负例时，暂不把 dense occupancy 当作已验收主任务；可保留已标注 actor 状态/轨迹辅助学习。SPD 负例也要限定在经验证的标注覆盖或自由空间内，不能假定所有未标框位置都为空闲。

只报告正例 recall，或者全图 occupied 能得满分的指标，都不能证明完整世界预测成功。

## 4. 模型与训练目标

### 4.1 目标结构

```text
vehicle / RSU observations + pose / calibration
    -> shared observation encoder + lightweight node adapter
    -> geometry alignment + permutation-invariant aggregation
    -> structured physical state + support / uncertainty
    -> shared action-conditioned dynamics
    -> future occupancy / flow / actor states
    -> candidate evaluation and ego trajectory refinement
```

类型/域信息只用于观察路径。当前有图像 BEV 特征与 planar warp，但没有完成基于相机标定的完整 metric lifting；这一目标不能直接写成已实现事实。

### 4.2 结构化动力学接口

当前 occupancy/flow 经共享编码器送入 dynamics，原始融合 latent 不再直接 rollout。后续要求：

- predicted/oracle 使用相同 occupancy 编码（例如概率）、相同 flow 单位和 unknown 表示；不能一支输入 logits、一支输入 0/1 却认为状态等价。
- 内部 latent 只能从声明的物理接口编码获得。actor、map、uncertainty 扩展需要明确字段和监督，不走原始观察旁路。
- 当前 8×8 点圆栅格是原型；需核对分辨率、actor 尺寸、实际 oriented box 与碰撞尺度，再确定正式规格。
- 当前 transition 为卷积递归加动作 embedding；transformer/state-space、多模态 head 属于可选实验。
- 相同 GT 输入的确定性接口测试只是回归测试。是否保留充分交互信息，需靠 oracle/predicted-state 和公平基线实验判断。

### 4.3 损失与动作诊断

保留原计划的状态、未来、一致性、不确定性及规划损失：

$$
\mathcal L=
\lambda_s\mathcal L_{\mathrm{state}}+
\lambda_f\mathcal L_{\mathrm{future}}+
\lambda_a\mathcal L_{\mathrm{agree}}+
\lambda_i\mathcal L_{\mathrm{inv}}+
\lambda_u\mathcal L_{\mathrm{unc}}+
\lambda_p\mathcal L_{\mathrm{plan}}.
$$

- state/future 分别监督合法支持域内的 occupancy、flow、actor 状态；oracle 遵循同样标签语义。
- agreement/invariance 与 GT accuracy 联合使用，只在协议定义的共同支持域计算；有效区域归一化不能被大量未知背景稀释。
- uncertainty 必须针对误差和真实有效支持校准。全网格 covered 不等于全网格可见，更不能直接给盲区低方差目标。
- correct/removed/perturbed/shuffled action 必须走同一个含 residual 的 rollout；仅改变 action，其余输入、网络模式、随机性保持一致。
- 单列非 ego 结果，排除直接复制 ego 轨迹的捷径。日志上的 action sensitivity 不等于识别其他车辆的因果响应。
- 配对天气 loss 先逐样本计算再按 valid mask 屏蔽；DDP 各 rank 前向结构一致，但不能用 batch 平均有效率代替逐样本 mask。

当前论文草稿中的 action contrast loss、条件 observer probe 不属于已完成实现，也不是仅因草稿写过就必须新增的核心模块。需要时作为有独立对照的可选实验。

### 4.4 规划接口

保留匹配容量的 trajectory proposal/refiner，进入规划阶段前：

- 保留未来障碍物空间信息；全图均值不能区分同样占用总量但左右位置不同的障碍。
- 对候选动作的世界后果进行空间风险评价，明确碰撞、进度、舒适性等成本及标签来源。
- 提供不依赖 GT future action 的推理入口。
- world 阶段关闭规划指标，不把 teacher trajectory 回填后的零误差当规划成绩。
- 尚未具备地图/闭环环境时，仅报告实际支持的开环指标及限制；不直接移植 NAVSIM PDM 名称。

## 5. 必需实验与判定标准

### 5.1 V2X-WAM 对标与公平基线

V2X-WAM 是直接相关工作，需如实讨论和比较。输出同样的轨迹/occupancy 不构成区别，额外数据与视角也不自动构成方法创新；不能因为相关工作只在 arXiv 就省略。

近期最小实验包含：

| 模型 | 输入与预算 | 目的 |
| --- | --- | --- |
| 融合式动作 WAM | 与 FADO 相同观察、标注、action、horizon 和训练预算 | 检验显式状态接口的价值 |
| 融合式动作 WAM + observer dropout | 同上 | 排除只是 dropout 增强 |
| FADO-WAM | 同上 | 检验 factorization 和训练约束 |

当前 [cooperative_wam.py](../navsim/agents/ImagineWorld/cooperative_wam.py) 的旧 latent future head 不接收 ego action。脚本名含 v2x_wam 不等于完成原文复现；受控融合基线也不能冒称官方实现。原文输入、split、horizon、监督、规划容量和指标要另行对齐；4 s 默认值目前只是本仓库配置。

完整主实验补充 ego-only planner、cooperative planner without WAM、V2X-WAM/其 dropout 版本，以及 FADO 去掉 agreement/invariance 的消融。

### 5.2 实验矩阵

| 维度 | 设置 | 必须同时报告 |
| --- | --- | --- |
| 同分布精度 | SPD 官方有 GT 的 train/val，vehicle+RSU | 世界预测精度、有效样本/目标范围 |
| 观察子集 | SPD full/ego-only/RSU-drop；VLA4CoDrive 多车子集 | 每个 subset 的 GT accuracy 与 OFD |
| oracle-state | 各方法同 GT state、同 action | 方法间 rollout 精度及 action 对照 |
| 等信息量 | 已标注 actor 和支持范围匹配的 subset 对 | accuracy/OFD，不用 BEV 方框冒充等可见信息 |
| 覆盖分层 | 协同揭示新目标 vs 增加冗余证据 | 各层精度与鲁棒性增益 |
| 天气 | 配对天气与单独定义的 held-out weather | 精度、一致性；不声称真实天气动力学不变 |
| 动作使用 | correct/removed/可行扰动/shuffled | 非 ego actor/flow 或风险变化；注明日志证据边界 |
| 规划 | 训练好的 proposal/refiner，ID 与观察变化 | 实际可计算的轨迹与安全指标 |
| 迁移 | domain、observer、combined shift 分开 | target-only/naive pretrain 与数据效率曲线 |

不将单车到多视角的自然收益作为 factorization 贡献。不用“同一模型同一输入输出相等”代替 oracle 方法间对比。

### 5.3 指标合同

保留原计划的 Observer-Conditioned Future Disagreement（OFD）：固定场景、query、坐标和 action，只改变观察集合，比较未来预测：

$$
\operatorname{OFD}(\mathcal C_1,\mathcal C_2)
=D_M(\hat S^+_{\mathcal C_1},\hat S^+_{\mathcal C_2}).
$$

其中 $M$ 是两次预测可比较的有效支持域。分别声明 occupancy、flow、actor 的距离、单位与归一化方式；不直接混加不同单位的差异，也不跨 query 坐标直接比较。

- 世界：按 horizon 报 occupancy IoU、flow EPE、actor ADE/FDE；有可靠标签时增加碰撞区域召回及 rollout 漂移。
- occupancy IoU 必须存在可靠正/负监督域。不可计算记 N/A，附有效分母，不补 0；recall 不独立作为最佳世界 checkpoint 判据。
- 当前 actor 定位误差与未来轨迹误差分开命名。ADE/FDE 按 t0 或整条轨迹一次匹配后保持身份，不能逐时刻重匹配掩盖交换。
- OFD 报 occupancy/flow/可用 actor 与 horizon-wise 差异，说明支持范围，并与同 subset 的 accuracy 配对。低 OFD 且低 accuracy 不算成功。
- 不确定性报告有定义支持域内 NLL/ECE、误差相关性；未接入校准评测不写已校准。
- 规划指标按可用条件选择 ADE/FDE、碰撞、越界、进度、TTC、舒适性；不得将缺失地图或环境的指标编造为可用。
- 连续帧和天气回放不作为独立实验重复；统计按场景/序列分组，报告有效样本量及局限。

### 5.4 消融与 sim-to-real

保留的核心消融：无结构瓶颈、动态模块可见观察信息、无 agreement、无 invariance、仅 dropout、无 uncertainty、共享 encoder/adapter、GT/predicted state、correct/removed/shuffled action、单天气/配对天气、观察数量和未见组合。

迁移单列以下方法：SPD target-only、naive VLA4CoDrive pretrain + full finetune、joint training、可选 feature alignment、严格 frozen dynamics、residual adaptation、full finetune。低数据比例为 1%、5%、10%、25%、100%，在同一官方训练池取样，不使用验证标签选择训练样本。

strict freeze 必须冻结完整 transition 含 residual；residual adaptation 另列，不能把可训练 correction 混入 frozen 对照。

### 5.5 成功与停止条件

正式结论至少需要：

1. 同输入下的正常精度不劣于公平对齐基线，并在观察变化下改善 accuracy/OFD。
2. 改善不能由普通 dropout 或额外数据单独解释。
3. action 对照通过同路径、非 ego 检查；没有充分交互证据时限定相关论断。
4. oracle 方法间对比与 predicted-state 实验支持可复用动力学。
5. 世界改善传递到至少一项可验证的规划安全指标。
6. 初筛一 seed 成立后，至少三个 seed 报均值/方差及关键失败案例。

仅一致性变好而精度下降、仅迁移有效、仅仿真有效或与 dropout 基线等价，都不足以维持当前强创新论断，应先检查状态充分性与方法设计，不通过增加训练轮数替代解释。

## 6. 执行顺序：用验收关卡替代旧周排期

Stage 0–5 是研究阶段，CLI world/planner/spd/full 是训练模式，两者不等价。现有可训练模块覆盖研究 Stage 2/3 的部分内容，但 Stage 0 与公平基线尚未整体验收。

| 关卡 | 对应旧阶段 | 工作和交付 | 放行条件 |
| --- | --- | --- | --- |
| G0 数据与监督 | Stage 0 | 坐标、三态标签、时间/dt、缺标/末步 flow、真实支持域 | 第 7.1 节反例通过，数据合同与真实叠图可追溯 |
| G1 评测与工程 | 原计划未充分展开 | 同路径动作、轨迹身份、N/A、weather mask、EMA/恢复、manifest | 第 7.2–7.3 节通过，指标可重现 |
| G2 短程原型 | Stage 2/3 初验 | 合法标签的小样本拟合与验证 split 短跑；优先 SPD 主任务 | 预测图/精度合理，排除全占用与复制 ego 等退化 |
| G3 核心对比 | Stage 1 + Stage 2/3 正式实验 | 三个受控模型、oracle/predicted state、各 subset，再对齐原文 | 等预算 accuracy/OFD 表；三 seed 与关键消融 |
| G4 规划 | Stage 4 | 无 GT 动作推理、空间风险反馈、匹配容量 | 至少一项安全收益与明确的开环/闭环边界 |
| G5 迁移和论文 | Stage 5 + 写作 | frozen/residual/full、naive-pretrain、低数据曲线、论文同步 | 只写实际支持的实现和结论 |

G0/G1 未通过时，长训练作为调试记录，不作为正式主实验。本次更新不终止现有训练。标签/坐标/时间语义变化后，创建新的实验版本；旧权重可明确标记为预训练/调试来源，不无说明地续训拼接曲线。

最小有效闭环为：已验收的 SPD 观测与交通标签 → 结构化状态 → ego-action dynamics → 每个 subset 的 accuracy/OFD → 同输入基线对照。VLA4CoDrive 在坐标正确后补充多车/天气与 actor 辅助监督。规划和迁移在后续关卡加入。

## 7. 旧计划遗漏的修复任务与验收清单

以下 [ ] 表示尚未验收，不表示一定没有代码。完成时附日期、代码/数据版本、命令、结果和失败样本；源码链接以函数名定位，不依赖永久行号。

### 7.1 G0：先保证训练目标正确

**G0-1：VLA4CoDrive 坐标。**

- [ ] 建立共同右手坐标，统一 current pose、relative pose、history、future；消除每车独立 PCA 和 window/scene 混用。
- [ ] 同一原始点直接投影和组合投影一致；覆盖不同 query、停车、转弯、窗口边界、三车闭环。
- [ ] 同一数据的两条数学路径按数值容差验收，建议 FP64 < 1e-3 m；原始定位噪声另外报告。
- [ ] 回归样本 clearNight / scene001 / frame000015 / Vehicle_2 / +2.5 s。10 月 7 日仍复现约 16.93 m 误差；时间索引报告不能关闭此项。
- [ ] 坐标定义不读取尚未发生的轨迹来决定推理输入。

定位：[relative_poses / _window_local_frame](../navsim/planning/training/vla4codrive_coop_dataset.py)、[_poses_in_current_frame / _load_item](../navsim/planning/training/open_observer_dataset.py)。

**G0-2：标签支持集与物理数值接口。**

- [ ] 对每套数据明确监督目标、空间范围、可靠负例来源、截断和 unknown。
- [ ] 修正主分支 `known=occupancy` 只有正例，而 oracle 无 mask 全图监督的冲突。
- [ ] 改变 unknown 格子的预测不影响该格子的监督项；改变可靠负例影响损失；全图 occupied 不能与正确预测得到同等有效精度。
- [ ] predicted/oracle occupancy 使用相同概率/编码，flow 单位和 mask 一致；同一物理状态得到相同编码。
- [ ] 明确点圆近似或 oriented-box occupancy 的任务定义，不能混报。
- [ ] 缺乏负例时启用明确范围的 actor 辅助任务，不把稀疏正例 recall 当完整世界指标。

定位：[rasterize_actor_targets / canonical_state / _prediction_loss / open_observer_wam_loss](../navsim/agents/ImagineWorld/open_observer_wam.py)。

**G0-3：SPD 时间与 flow。**

- [ ] 固定时刻插值或实际 dt 方案二选一落实，并同步指标；最近邻误差不隐去。
- [ ] 移动物体末步 velocity/flow 不默认零；使用有效后向差分或有效性标记。
- [ ] 起止 actor 标签都有效才生成差分；缺标回填值不进入有效 flow。
- [ ] 记录 16 actor 上限造成的截断，保证指标与负例覆盖不冲突。
- [ ] 用 frame000023 等真实样本验证末步/缺标。10 月 7 日仍见末速度标签为零、实际最后一段约 8 m/s 的例子。

定位：[SPDOpenObserverDataset._actors](../navsim/planning/training/open_observer_dataset.py)、[rasterize_actor_targets](../navsim/agents/ImagineWorld/open_observer_wam.py)。

**G0-4：支持范围、不确定性及划分。**

- [ ] 分开 grid_support、sensor_visibility、label_known。
- [ ] query 位于原点导致全图 covered 时，不将其解释成摄像头全图可见；不能据此给盲区低方差监督。
- [ ] 可靠可见性尚未接入时，不宣称完成遮挡不确定性校准。
- [ ] 检查实际 history/current/future frame ID 和天气 replay 支持不跨集合；保留官方 SPD split。
- [ ] 新报告记录完整配置和覆盖范围，不将只检查前 200 个时间样本、只画首个 BEV 的历史审计称为全量几何验收。

定位：[warp_local_bev_to_canonical / CanonicalAggregator](../navsim/agents/ImagineWorld/open_observer_wam.py)、[audit_oo_samples.py](../scripts/audit_oo_samples.py)。

### 7.2 G1：指标和动作对照

- [ ] correct/removed/perturbed/shuffled 统一使用含 residual 的 rollout；相同 action 经不同入口输出一致。
- [ ] 非 ego 目标与 ego 分报；全图 BCE action gap 只作附加诊断。
- [ ] ADE/FDE 固定轨迹身份；加入交换两个 actor 身份的反例。
- [ ] 不可计算 IoU 标 N/A，不补 0；空目标/空支持如何计入汇总有明确规则。
- [ ] full、ego-only、dropout 各自报告 GT accuracy/OFD，支持域和 horizon 一致。
- [ ] 最佳 checkpoint 不以正例 recall 单独决定；使用经过验收的世界/actor 指标并记录规则。

定位：[动作对照与 evaluate_observers](../navsim/agents/ImagineWorld/open_observer_agent.py)、[world_metrics](../navsim/agents/ImagineWorld/open_observer_wam.py)、[ModelCheckpoint 配置](../navsim/planning/script/run_training_oo_wam.py)。

### 7.3 G1：训练工程与可追溯性

- [ ] weather loss 逐样本屏蔽，覆盖全有效、混合有效、全无效；无效样本不贡献该项梯度。
- [ ] DDP 各 rank 前向结构一致；相同有效样本的单卡/多卡 loss 和梯度在容差内一致。
- [ ] 验证 EMA 参数是否随 checkpoint 保存/恢复，独立 eval 使用的是否为当时验证权重；以重现验证指标验收。
- [ ] GPU 数/batch 改变时核对真实 optimizer step、warmup/cosine 进度；相同 epoch 不自动代表同训练预算。
- [ ] 保存 run manifest：CLI、展开配置、seed、source/protocol、split 摘要、坐标/标签版本、horizon、单位、有效监督比例、checkpoint 来源、GPU/batch/精度、指标版本和 EMA 状态。
- [ ] 项目沿用父目录 Git 状态时，保存本项目关键文件 hash；不把父仓库无关改动当作本项目版本。
- [ ] 标签/坐标/时间改变后新建 run_id 与目录，旧权重的复用方式明确声明。

定位：[run_training_oo_wam.py](../navsim/planning/script/run_training_oo_wam.py)、[compute_loss](../navsim/agents/ImagineWorld/open_observer_agent.py)。

### 7.4 原计划已有、尚未完成的后续任务

- [ ] 同输入动作条件基线、dropout 基线、FADO 正式对比；原论文协议核对。
- [ ] oracle 方法间对照、equal-evidence、覆盖分层、各 subset accuracy/OFD、三个 seed。
- [ ] 无 GT action 推理、保留空间关系的未来反馈、规划安全评价。
- [ ] 严格冻结完整 transition 含 residual，与 residual adaptation 分组。
- [ ] target-only、naive pretrain、joint training、低数据比例迁移曲线。
- [ ] 基于真实支持域的不确定性、天气/观察数量鲁棒性与失败案例。
- [ ] 逐项同步论文初稿，未实现模块标可选/待完成，不写成现成贡献。

## 8. 论文写作与审稿问题

### 8.1 证据组织

| 审稿问题 | 需要的证据 |
| --- | --- |
| 多视角本来就更好 | 相同观察输入、equal-evidence、覆盖分层、oracle 控制 |
| 是否仅加 consistency loss | 结构瓶颈消融、dropout 基线、未见观察组合下准确率/OFD |
| 不变性是否损失信息 | accuracy–consistency 对照、状态充分性/不确定性、失败案例 |
| 是否真的使用 action | 同路径非 ego 动作对照；必要时再评估小规模交互分支数据 |
| 是否仅因更多训练数据 | 同预算 naive pretrain、joint training 与参数/计算报告 |
| 是否只是域适配 | 真实同域主实验先成立，迁移单列 frozen/residual/full 对照 |

新增仿真采集、条件 observer probe 或 action contrast loss 都是可选路线，不因初稿写过就自动纳入核心工作量；先证明现有最小方法的价值。

### 8.2 当前初稿需要同步的内容

[main.tex](../paper/fado_wam_cvpr2027/main.tex) 中的条件 observer probe、action contrast loss、逐候选风险评分、按 town/route 划分、完整可见性及部分校准描述，超前于当前实现/数据。

本次只更新 docs，不改论文。进入写作关卡时逐条选择“实现并验证”或“删除/降级表述”。摘要中的精度、规划和迁移提升必须在实验完成后填写，不提前保证成功或虚构数字。

### 8.3 保留的最终决策

- 主对标：V2X-WAM；主要证据来自同输入公平比较。
- 主故事：观察推断与动作动力学解耦；仍然是 WAM。
- 主输出：结构化世界状态与 ego 规划；不生成 RGB 视频。
- 现有数据：VLA4CoDrive 与 V2X-Seq-SPD；不默认需要扩充数据才能开始修复。
- 多车数据用于观察集合/天气与辅助监督；SPD 为真实主实验。
- sim-to-real 是次要贡献，在核心对比成立后开展。
- 旧 joint-action/all-as-ego 研究路线终止；保留的旧代码不代表路线恢复。
- ReCogDrive/ImagineWorld VLA/VLM 代码仍保留，当前主模型未启用。

## 9. 后续计划维护规则

1. 数据协议、接口、指标或训练阶段改变时，同步更新第 2 节状态及对应合同，不能只追加“已修复”一句话。
2. 区分“当前实验阻断项”“后续阶段任务”“可选研究”，不把所有缺项都塞进同一轮训练。
3. “已修复/已验收”必须附命令、版本和证据；训练能启动、loss 下降、单元测试通过不替代真实数据验收。
4. 旧报告保留日期，新审计生成新版本；本次 10 月 7 日针对性复核不冒称重新完成全量测试。
5. 本文提供随项目推进更新的计划，不是自动定时任务；本次未创建自动巡检、停止训练或修改实验。
