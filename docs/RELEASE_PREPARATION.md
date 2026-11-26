# 公开版本整理记录

2026-11-26：从teleop提交72b7c3f创建codex/public-release-docs。
本轮仅修改文档、引用和忽略规则，不改变训练/控制行为，不推送、不切换GitHub默认分支、不创建发行版。

## 已整理

- UniMimic首页与贡献边界；原导入README/CITATION单独归档。
- 可移植安装、训练、分模态评估、导出与仿真入口。
- 当前状态、实机已知问题和历史验证记录的区分。
- 数据/外部资产来源、精确上游版本未知项及GR00T本地版本依赖说明。
- 引用身份更新，保留全部已有版权声明。
- 新增常见本地配置与权重忽略规则。

## 发布前仍待完成

- 在干净环境按GETTING_STARTED完整复现；本轮不以现有环境检查替代该结果。
- 补齐可访问的GR00T参考fork/patch和精确BlinkMimic导入SHA（如果能恢复）。
- 发布有许可依据的模型、评估JSON、manifest指纹和自制演示；不默认上传数据。
- 核对全Git历史中的敏感信息与第三方分发范围。未做正式完整历史审计；.gitignore不会清理已提交内容。
- 将审核后的整理分支合并到main并推送，确认GitHub首页显示新README。
- 再决定v0.1.0标签；当前没有创建标签或宣称稳定实机支持。

建议GitHub About：
“Multimodal whole-body control for Unitree G1, built on BlinkMimic: FSQ tokens, paired motion/SMPL/teleop training, and experimental deployment.”

建议topics：humanoid-robotics, motion-tracking, mujoco, teleoperation, finite-scalar-quantization。

Python分发元信息暂保留blinkmimic以兼容已有安装；未向PyPI发布UniMimic包。
正式打包发布前需单独核对名称、作者、版本和第三方license-files，不能仅依靠README更名。
