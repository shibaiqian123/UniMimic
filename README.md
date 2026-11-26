# UniMimic

> **分支说明：sonic** — 本分支保留Motion/FSQ64、proprio93、双GPU及BONES相关开发阶段；不包含后续三模态联合训练与实时VR接入。 以下为项目整体介绍；完整功能请使用 [main](https://github.com/shibaiqian123/UniMimic/tree/main)。跨分支文档链接固定到本次同步版本，不能据此假定本分支存在对应脚本。

基于 [BlinkMimic](https://github.com/fuyichen0725/blinkmimic) 的 Unitree G1 全身控制研究项目，扩展 **64D FSQ、多模态联合训练和实验性 VR 部署**。

> **研究版本，尚未发布稳定实机版本。** 已完成离线评估、仿真及初步实机实验；实时模式切换和三点跟踪仍有已知问题。见 [当前状态与限制](https://github.com/shibaiqian123/UniMimic/blob/e14f95431dbb6881acc3b576e2a7aded1c40fed7/docs/PROJECT_STATUS.md)。上游“三分钟训练”的结果不属于 UniMimic 的实验结论。

## 新增工作与来源

- **动作表示**：连续32兼容、连续64对照、32档FSQ64；新策略支持93维本体观测历史。
- **多模态策略**：Motion、SMPL、Teleop独立encoder，共享FSQ规则与动作decoder；独立输入归一化。
- **训练与数据**：保持全局batch语义的双GPU训练、BONES读取与源组划分、G1–SMPL配对、三点及分片特征缓存。
- **评估与部署**：分模态评估、拆分ONNX、官方G1模型CPU/DDS仿真、PICO输入及实验性实机接入。

训练算法沿用BlinkMimic的可微world model、想象rollout与critic；多模态特征和辅助损失参考SONIC。**本项目不是SONIC官方实现，不使用SONIC动作策略权重，也不宣称复现其性能。** 实时输入路径另依赖外部GR00T/PICO组件及SONIC planner资产。

内部Python包名保留 `blinkmimic`，用于旧checkpoint和导入兼容。项目名UniMimic与包名不同是有意保留。

## 架构

Motion参考 / SMPL人体参考 / Teleop三点及下半身参考

→ 对应独立encoder → FSQ64（2×32，逐标量32档）

→ 共享decoder（10×93本体历史 + token）→ 29维关节action。

当前机器人与参考的相对朝向进入encoder，不直接进入93维decoder。Teleop仍需要下半身参考，不能仅给三个点就替代完整运行时。

单模态proprio93与三模态omni的Motion特征契约不同，部署包和checkpoint不可互换。详见 [proprio93](docs/PROPRIO93.md) 与 [omni](https://github.com/shibaiqian123/UniMimic/blob/e14f95431dbb6881acc3b576e2a7aded1c40fed7/docs/omni.md)。

## 从哪里开始

1. [安装与最小复现](https://github.com/shibaiqian123/UniMimic/blob/e14f95431dbb6881acc3b576e2a7aded1c40fed7/docs/GETTING_STARTED.md)：从克隆、数据准备到单模态评估及仿真。
2. [数据来源与生成](https://github.com/shibaiqian123/UniMimic/blob/e14f95431dbb6881acc3b576e2a7aded1c40fed7/docs/DATA_AND_ASSETS.md)：外部数据、配对、三点及缓存。
3. [多模态训练、恢复、评估与导出](https://github.com/shibaiqian123/UniMimic/blob/e14f95431dbb6881acc3b576e2a7aded1c40fed7/docs/omni.md)。
4. [当前结果、证据范围与已知问题](https://github.com/shibaiqian123/UniMimic/blob/e14f95431dbb6881acc3b576e2a7aded1c40fed7/docs/PROJECT_STATUS.md)。
5. [公开版本整理记录](https://github.com/shibaiqian123/UniMimic/blob/e14f95431dbb6881acc3b576e2a7aded1c40fed7/docs/RELEASE_PREPARATION.md)。

专题文档：

- [FSQ64](docs/FSQ64.md)、[双GPU](docs/DUAL_GPU.md)、[BONES全量训练](docs/bones_full.md)。
- [sim2sim](docs/SIM2SIM.md)、[部署接口](docs/DEPLOY_G1.md)。
- [实验性VR实机流程](https://github.com/shibaiqian123/UniMimic/blob/e14f95431dbb6881acc3b576e2a7aded1c40fed7/docs/VR_SIM2REAL.md)。

历史文档中的229/32是作者开发机器代号；其中绝对路径是历史实验配置，不是安装要求。新用户优先使用可移植的入门文档。

## 当前实验概况

三模态15,000迭代实验使用两张A100，记录训练时间13.79小时（约27.58 GPU小时，按两卡分配时长计算，不含完整数据准备成本）。作者报告的固定96段验证结果：Motion归一化回报0.910、SMPL 0.905、Teleop 0.880，三个模态均零失败。

这些是单次实验的验证集结果，不是多种子结论，也不代表真实VR或硬件成功率。完整指标、数据划分及证据发布状态见 [当前状态](https://github.com/shibaiqian123/UniMimic/blob/e14f95431dbb6881acc3b576e2a7aded1c40fed7/docs/PROJECT_STATUS.md)。目前仓库不提供已发布的预训练权重下载承诺。

## 上游与引用

- [BlinkMimic](https://github.com/fuyichen0725/blinkmimic)：代码与训练核心；其上游包括InstantTrack和SuperMimic。
- [GR00T-WholeBodyControl / SONIC](https://github.com/NVlabs/GR00T-WholeBodyControl)：多模态设计、参考输入和外部实时组件。
- SuperMimic-Repro-Sim2Real：DDS bridge和传感器模型的来源，见第三方说明。

原导入README保存在 [上游归档](https://github.com/shibaiqian123/UniMimic/blob/e14f95431dbb6881acc3b576e2a7aded1c40fed7/docs/upstream/BLINKMIMIC_README.md)，其结果与署名属于上游，不作为本项目成果。

引用本仓库请使用 [CITATION.cff](https://github.com/shibaiqian123/UniMimic/blob/e14f95431dbb6881acc3b576e2a7aded1c40fed7/CITATION.cff)，并按使用内容引用相关上游。文件范围、许可与参考版本见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

## License

保留 [MIT许可证与上游版权声明](LICENSE)。第三方源文件保留各自许可。数据集、SMPL资产、机器人资产和外部预训练权重不因使用本仓库而获得新的分发授权，见 [数据与资产说明](https://github.com/shibaiqian123/UniMimic/blob/e14f95431dbb6881acc3b576e2a7aded1c40fed7/docs/DATA_AND_ASSETS.md)。
