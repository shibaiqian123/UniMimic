> 本页包含开发阶段记录；当前能力与限制以 [PROJECT_STATUS](PROJECT_STATUS.md) 为准，可移植命令见 [GETTING_STARTED](GETTING_STARTED.md)。历史229/32路径仅用于追溯。

# 64D Motion token / FSQ

实现位置为229的 /home/cqu023/blinkmimic。旧32D checkpoint和v1部署包保持原行为；本轮不修改32、不自动启动长期训练或真机。

## 模型契约

`--latent-mode continuous32|continuous64|fsq64` 默认continuous32。新两种模式为2个token，每个32维；fsq64每个标量32档。连续64使用相同有界变换但不取整。新模式禁止pi_ref_skip，decoder没有原始goal旁路。

FSQ参考vector-quantize-pytorch 1.27.15的默认非对称偶数档位公式（MIT），参数布局参考GR00T-WholeBodyControl/SONIC all_mlp_v1。对raw encoder输出x：

- half_l = 31 * 1.001 / 2，offset = 0.5，shift = atanh(offset/half_l)。
- u = tanh(x + shift) * half_l - offset。
- continuous64 = u / 16；fsq64 = round(u) / 16。
- round使用ties-to-even，与PyTorch/ONNX一致；训练用STE，量化算术保持FP32。
- FSQ输出档位为{-16,...,15}/16，即[-1,0.9375]，不是对称的32点linspace。
- 不构造32^32词表或打包单一token整数，避免整数溢出和不必要的码本分配。

Python接口：`encode_tokens(goal)` → [B,2,32]；`decode_tokens(obs_history,tokens)` → [B,29]。
`blinkmimic.latent.quantize_tokens` 显式将外部归一化连续预测投影到档位；decoder不隐式二次量化。
连续模式的有界输出边缘可能略超FSQ网格端点，这是量化前后区别，不额外裁剪对照组。

观测保持10×99，goal保持730，action保持29。观测中参考相对朝向6D仍由动作包提供；本轮不是完整VLA运行时。token与decoder必须配对，不兼容SONIC预训练权重。

checkpoint args记录latent_spec，state_dict另有模式标记。旧checkpoint缺字段按continuous32加载；同形状continuous64/fsq64加载混用报错。resume要求模式、量化规则和参考旁路一致。

## 单卡A100训练

以下命令在拥有该代码与数据的训练环境运行；本次只交付命令，不修改32。两组依次使用同一张卡、同一预算：

```bash
cd /home/cqu023/blinkmimic
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 .venv/bin/python scripts/train_latent.py \
  --latent-mode fsq64 --log-dir logs/lafan_fsq64_s0 --seed 0 --tensorboard

CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 .venv/bin/python scripts/train_latent.py \
  --latent-mode continuous64 --log-dir logs/lafan_continuous64_s0 --seed 0 --tensorboard
```

启动器继承发布版main配方，移除pi_ref_skip；默认3500迭代是首轮固定预算，不是FSQ收敛保证。若延长，两组设相同iterations。它验证40个CSV并记录数据SHA256、种子和完整命令到experiment.json。可用--dry-run仅查看命令。

当前协议使用全部40段训练、在相同40段评估，是训练集跟踪对照，不宣称未见动作泛化。第一轮种子0；稳定优劣结论需追加相同的多种子实验。

恢复必须使用包含优化器状态的last.pt；其余设置沿用原命令：
```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 .venv/bin/python scripts/train_latent.py \
  --latent-mode fsq64 --log-dir logs/lafan_fsq64_s0 --seed 0 --tensorboard \
  --resume logs/lafan_fsq64_s0/last.pt --iterations 7000
```
continuous64同理替换模式和目录。若修改了网络宽度等参数，恢复时也必须原样传入。

latent_metrics.jsonl在log_every周期记录固定分布样本的量化误差、每维档位使用数量、饱和比例、encoder梯度范数；TensorBoard同步记录。连续64中的档位指标是“若量化”的对照统计，不代表前向实际量化。

## 同条件评估

```bash
PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 .venv/bin/python tools/eval_metrics.py \
  logs/lafan_continuous64_s0/ckpt_003500.pt \
  logs/lafan_fsq64_s0/ckpt_003500.pt \
  --motion_dir data/lafan1_g1 --out results/latent64_s0
```

输出回报、失败数、跟踪误差、接触指标、action_delta_rms及数据/checkpoint指纹。delta不包含初始零action跳转，不跨动作边界，失败后不继续累积。精度、历史和目标窗口必须相同。
旧32D仅作为单独历史基线；其参考旁路与新模型不同，不把全部差异归因于量化。

## 导出及final仿真

训练后再运行：
```bash
PYTHONPATH=src OMP_NUM_THREADS=1 .venv/bin/python tools/export_deploy.py \
  --ckpt logs/lafan_fsq64_s0/ckpt_003500.pt \
  --motion data/lafan1_g1/walk1_subject1.csv \
  --preparation-gain-scale 4.5 \
  --out deploy_artifacts/fsq64_s0_walk1
```
连续64使用对应checkpoint和新输出目录。导出目录不得覆盖现有部署包。

v2包包含：
- policy.onnx：obs_history [B,10,99] + goal [B,730] → actions [B,29]。
- motion_encoder.onnx：goal → tokens [B,2,32]，包括goal归一化与量化。
- decoder.onnx：obs_history + tokens → actions，包括观测归一化。
- motion.npz、deploy_config.json、export_validation.json；全部模型文件与动作均记录指纹。

```bash
OMP_NUM_THREADS=1 .venv/bin/python scripts/deploy/sim2sim.py \
  --cmd final --bundle deploy_artifacts/fsq64_s0_walk1 \
  --unitree-xml /home/cqu023/Documents/unitree_rl_lab/unitree_mujoco/unitree_robots/g1/scene_29dof.xml \
  --viewer --gl-renderer nvidia --out deploy_logs/final_fsq64
```

操作沿用1准备、2待机、B撤支撑、3播放；正常结束恢复支撑后D停止。不新增实机启动手续。短训练导出仅验证接口，禁止把它标作完成动作训练的上机包。

## 测试

```bash
OMP_NUM_THREADS=1 .venv/bin/python -m pytest -q tests/test_latent.py tests/test_deploy_runtime.py tests/test_final_pipeline.py
```

参考公式源码对照使用独立测试缓存（不改变运行环境）：
```bash
/home/cqu023/.local/bin/uv pip install --python .venv/bin/python \
  --target .cache/fsq_reference vector-quantize-pytorch==1.27.15 --no-deps
```

测试覆盖旧32D输出、实际ckpt_003500/walk包、全部档位、边界、STE、AMP、编译反向、EMA、模式冲突和保存恢复。实测结果见FSQ64_VALIDATION.md。
