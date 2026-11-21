# Proprio93 + future relative orientation

本模式只改变Motion策略输入分工：encoder接收参考与相对朝向，控制decoder只接收本体历史和token。它不是SONIC所有特征/损失的完整复现；没有新增Human/SMPL/VR/VLA encoder。

## 接口

旧模式legacy99保持默认，旧32D和99维FSQ64 checkpoint/部署包继续使用原行为。
新实验显式添加 --observation-mode proprio93；仅支持continuous64或fsq64，不允许pi_ref_skip。

- decoder观测：[B,10,93]，每帧q-q0(29)、dq(29)、骨盆角速度(3)、投影重力(3)、上一原始action(29)；历史旧到新。
- token：[B,2,32]，decoder总输入994维，输出29维action。
- Motion encoder：原goal [B,730] + orientation6d_future [B,10,6]，按“原730维，然后60维”拼成790维。
- 原goal中参考yaw坐标系的旋转仍保留；新增窗口是机器人相对未来参考，不替换原goal语义。
- 每帧取R_robot_current.T @ R_ref_aligned_future的前两列，按行展开成6维。
- future_frames严格使用f+1..f+10并截断在所属clip末帧。没有跨clip，也不是复制当前6D十次。
- 部署只在准备时固定yaw对齐；roll/pitch/yaw误差全部保留。pre/post使用重复端点参考，但每次根据最新机器人姿态重算相对旋转。

模型API：
```python
tokens = policy.encode_tokens(goal, orientation6d_future)
action = policy.decode_tokens(obs_history_93, tokens)
action = policy(obs_history_93, goal, orientation6d_future=orientation6d_future)
```

训练内部保留原99维特征包和经验状态；适配器显式切出93维历史。最新/历史的旧6D均不直接输入新策略，朝向窗口从当前真实/想象状态旋转与参考构造。world model、critic、奖励及动作缩放保留。
encoder输出现在依赖机器人当前朝向；未来采集VLA监督时要记录实际执行token和对应模型版本。

## 新训练（代码同步到32后）

先用原40段LAFAN1回归，不直接复用旧99维checkpoint。
单卡：
```bash
cd /share/2025slx/UniMimic
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 .venv/bin/python scripts/train_latent.py \
  --gpus 1 --latent-mode fsq64 --observation-mode proprio93 \
  --motion-dir /share/2025slx/UniMimic/data/lafan1_g1 \
  --log-dir logs/lafan_fsq64_proprio93_a100_s0 \
  --num_envs 4096 --iterations 3500 --seed 0 --tensorboard
```

双卡：CUDA_VISIBLE_DEVICES=0,1，--gpus 2，并使用新目录 logs/lafan_fsq64_proprio93_a100_2gpu_s0。全局4096环境不变。
continuous64对照只改latent-mode和日志目录，其他条件相同。3500迭代是起始对照预算，不保证收敛。

同模式/同world-size恢复：保留原参数，添加
```bash
--resume logs/lafan_fsq64_proprio93_a100_s0/last.pt --iterations 7000
```
不能将旧legacy99或“当前6D”的半成品checkpoint直接恢复为新模型，不自动裁剪、填零或迁移权重。

## 评估与导出

在32评估：
```bash
PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 .venv/bin/python -u tools/eval_metrics.py \
  logs/lafan_fsq64_proprio93_a100_s0/ckpt_003500.pt \
  --motion_dir data/lafan1_g1 --out results/proprio93_s0_3500
```
该评估自动读取checkpoint模式，可以同时比较legacy99和proprio93模型。保留同数据、精度和迭代数；两种结构不同，差异不能全部归因于FSQ。

选定checkpoint同步到229的对应目录后：
```bash
cd /home/cqu023/blinkmimic
PYTHONPATH=src OMP_NUM_THREADS=1 .venv/bin/python tools/export_deploy.py \
  --ckpt logs/lafan_fsq64_proprio93_a100_s0/ckpt_003500.pt \
  --motion data/lafan1_g1/walk1_subject1.csv --preparation-gain-scale 4.5 \
  --out deploy_artifacts/fsq64_proprio93_s0_3500_walk1
```

v3包：
- policy.onnx：obs_history [B,10,93]、goal [B,730]、orientation6d_future [B,10,6] → actions [B,29]。
- motion_encoder.onnx：goal、orientation6d_future → tokens [B,2,32]。
- decoder.onnx：obs_history、tokens → actions，无参考输入。
- 模型/动作指纹、latent_spec和observation_spec保存在deploy_config.json。
- 对应归一化包含在图内，runtime仍只依赖NumPy和ONNX Runtime。
- tools/export_onnx.py保留99维外部特征包的内嵌动作兼容接口；它用当前相对旋转和内嵌参考重建未来窗口，元信息明确标为compatibility envelope，不能混同v3接口。

## final sim2sim

在229：
```bash
OMP_NUM_THREADS=1 .venv/bin/python scripts/deploy/sim2sim.py \
  --cmd final --bundle deploy_artifacts/fsq64_proprio93_s0_3500_walk1 \
  --unitree-xml /home/cqu023/Documents/unitree_rl_lab/unitree_mujoco/unitree_robots/g1/scene_29dof.xml \
  --viewer --gl-renderer nvidia --out deploy_logs/final_fsq64_proprio93
```

遥控/键盘流程不变：1准备、2待机、B撤支撑、3播放，正常结束B恢复支撑后D停止。GPU诊断和PyTorch对照也按checkpoint/包模式读取输入。
日志中的orientation6d仍单独记录当前参考误差，orientation6d_future记录新encoder实际输入；不能从93维历史末尾读取朝向。

本轮仅做短训练、接口和导出验证。新模型的40段能力、完整walk1 final和真实硬件必须重新验收，旧99维结果不能代替。

## 诊断和数值测试

测试覆盖未来索引/clip边界、非零yaw与roll/pitch、四元数正负等价、端点动态相对旋转、历史初始化、原始上一action、梯度/STE、BF16/编译、动态batch及旧模型精确回归。

latent诊断用“机器人当前朝向等于当前参考朝向”的固定状态，然后计算真实未来窗口；未来转身时不全是单位旋转。JSON日志记录orientation_condition=current_reference_aligned。

```bash
OMP_NUM_THREADS=1 .venv/bin/python -m pytest -q tests/test_proprio93.py tests/test_latent.py \
  tests/test_distributed.py tests/test_deploy_runtime.py tests/test_final_pipeline.py \
  tests/test_warp_runner.py tests/test_sim2sim.py
```

实际运行结果见PROPRIO93_VALIDATION.md。真实双A100验证与长期训练不由本次修改自动启动。
