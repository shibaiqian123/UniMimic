# 双卡功能实现与验证记录

日期：2026-11-20。修改位置：229 /home/cqu023/blinkmimic。未修改32，未启动长期训练或真实机器人。

## 已实现
- --gpus 2 使用 torchrun/NCCL，一进程一GPU；全局环境和batch均分，不放大学习率。
- world model / policy / critic梯度归约后裁剪更新；全rank非有限梯度共同跳步。
- 全局归一化新增统计、结束episode按bin新增统计和统一probe focus。
- 观测、物理、重置及三个网络反向的本地CUDA图；通信和优化器更新位于图外。
- rank0日志、TensorBoard、普通checkpoint和包含rank RNG的完整恢复；完整恢复拒绝world size变化。
- 固定全局预算单/双卡测速工具、可选状态摘要审计和显式无图诊断。

## 已实际通过
- 最终相关测试43项通过：
  OMP_NUM_THREADS=1 .venv/bin/python -m pytest -q tests/test_distributed.py tests/test_latent.py tests/test_deploy_runtime.py tests/test_final_pipeline.py
- 两个CPU/Gloo进程：不等样本数量归一化与全局参考一致；梯度及一步Adam更新一致；保持原掩码损失分母；单rank NaN时双方参数和优化器状态不变；EMA/模型广播、focus广播、采样新增统计及空增量不重复衰减；恢复world-size契约。
- 229 RTX 4090单GPU新阶段集成：16环境、小网络、BF16、Muon、EMA，12次迭代后恢复至20次。
- 六个本地阶段全部实际captured=true：observe、physics、reset、world_backward、policy_backward、critic_backward。
- 恢复运行中异步probe完成：it12启动，it16应用，0/7 failures。无失败分支正确完成；CPU测试另覆盖非零focus广播。
- 原单卡训练路径4次迭代回归通过，包含原collection graph与world-model update graph。
- 原32D和FSQ模型/部署契约回归通过。git diff --check与Python语法检查通过。

集成测试输出：
- logs/distributed_local_smoke_v3/
- logs/distributed_local_resume_console.log
- logs/distributed_single_regression/
测试工具 tools/test_distributed_local.py 明确使用world_size=1，仅验证本地阶段；其中audit一份哈希不作为跨GPU一致性证据。

## 待32上的真实双A100验证
- 两个NCCL rank、每卡2048环境的完整训练短测。
- 跨GPU模型/归一化/EMA/优化器摘要一致，以及两卡完整恢复和退出。
- 固定全局4096环境下排除50次预热、计时200次的加速比及峰值显存。
- 3500次训练后的回报、失败数、跟踪误差和多种子能力比较。

因此当前交付为“功能代码和可执行验证入口已完成，CPU同步与单GPU图集成通过”。不能写成“双A100已跑通”或“已实现2倍加速”。
命令与运行契约见 DUAL_GPU.md。
