# UniMimic：安装与最小复现

这是可移植命令入口。所有命令从仓库根目录执行；不要求作者机器的用户名、IP或目录。
本页命令按现有CLI核对；本轮文档整理没有完成全新机器安装或重新训练，历史实验环境不能替代干净环境复验。

## 1. 获取代码与环境

```bash
git clone https://github.com/shibaiqian123/UniMimic.git
cd UniMimic
git switch <包含本说明的已发布分支或标签>
uv sync --extra omni --extra teleop
uv pip install --python .venv/bin/python onnx
export PYTHONPATH="$PWD/src"
```

将尖括号占位符替换为实际分支/标签。本说明尚在整理分支上，不能假定远端main已包含新功能。
训练需要Linux、NVIDIA GPU及兼容CUDA驱动；锁文件约束PyTorch 2.9.0、Warp 1.16.0，Torch来源为CUDA12.8索引。
作者使用A100训练、RTX4090开发验证；这不是对其他环境的兼容性保证。
部署核心依赖见根目录requirements-deploy.txt；实时输入还需要teleop额外依赖和外部GR00T/PICO环境。

## 2. 最小Motion路径

先阅读数据条款，再使用现有下载脚本获取G1重定向LAFAN1和机器人描述：

```bash
bash tools/download_data.sh
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 .venv/bin/python scripts/train_latent.py \
  --gpus 1 --latent-mode fsq64 --observation-mode proprio93 \
  --motion-dir "$PWD/data/lafan1_g1" --log-dir "$PWD/logs/lafan_fsq64_proprio93_s0" \
  --num_envs 4096 --iterations 3500 --seed 0 --tensorboard
```

首次下载会访问外部网络，数据数量以下载结果为准。先添加--dry-run检查参数；这不是训练通过测试。
日志目录须为新实验目录。恢复保留原参数，追加--resume "$RUN/last.pt"并设置最终iterations；
light checkpoint用于评估/导出，不代替完整恢复文件。

```bash
RUN="$PWD/logs/lafan_fsq64_proprio93_s0"
PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 \
.venv/bin/python tools/eval_metrics.py "$RUN/ckpt_003500.pt" \
  --motion_dir "$PWD/data/lafan1_g1" --out "$PWD/results/lafan_proprio93_eval"
```

同训练动作上的评估衡量跟踪，不能作为未见动作泛化结果。

## 3. 多模态路径

先完成配对与三点生成，见[数据说明](DATA_AND_ASSETS.md)。在有合法数据的机器设置：

```bash
DATA="$PWD/data/bones_seed"
CACHE="$DATA/derived/omni_sonic_v1"
PYTHONPATH=src .venv/bin/python tools/prepare_omni.py \
  --manifest "$DATA/paired/bones_smpl_v2/train.jsonl" --data-root "$DATA" \
  --threepoint-root "$DATA/derived/threepoint_sonic_v1" --out "$CACHE/train" --resume

RUN="$PWD/logs/omni_s0"
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=4 .venv/bin/python scripts/train_omni.py \
  --gpus 2 --index "$CACHE/train/index.json" --data-root "$DATA" --log-dir "$RUN" \
  --num_envs 16384 --pi_batch 4096 --iterations 15000 \
  --ckpt_every 1000 --seed 0 --tensorboard
```

15000是已有实验的检查点预算，不保证最佳或收敛；历史实验在18000出现退化。
num_envs/pi_batch均为全局值。使用单卡须改--gpus及可见设备，并按显存调整全局配置。
多模态入口从头初始化；不要把旧Motion权重作为omni恢复文件。

按相同方法生成val_fixed100/test_fixed100缓存；实际固定集合保留96/97个配对样本。
分模态评估：

```bash
for MODALITY in motion smpl teleop; do
  PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 \
  .venv/bin/python tools/eval_metrics.py "$RUN/ckpt_015000.pt" \
    --omni-index "$CACHE/val_fixed100/index.json" --omni-data-root "$DATA" \
    --modality "$MODALITY" --out "$PWD/results/omni_val_$MODALITY"
done
```

先用验证集选模型，再评估固定测试集；不要把测试结果用于持续挑选checkpoint。
特征、完整恢复和导出参数见[omni接口](omni.md)。

## 4. 导出与仿真

选取索引中真实存在的MOTION_ID，不按名称猜测。需自行准备官方G1场景及相邻mesh资产。

```bash
MOTION_ID='替换为索引中的动作ID'
G1_SCENE='/absolute/path/to/unitree_robots/g1/scene_29dof.xml'
BUNDLE="$PWD/deploy_artifacts/omni_selected"
PYTHONPATH=src .venv/bin/python tools/export_omni_deploy.py \
  --ckpt "$RUN/ckpt_015000.pt" --index "$CACHE/val_fixed100/index.json" \
  --data-root "$DATA" --motion-id "$MOTION_ID" --out "$BUNDLE"

OMP_NUM_THREADS=1 .venv/bin/python scripts/deploy/sim2sim.py \
  --cmd final --modality teleop --bundle "$BUNDLE" --unitree-xml "$G1_SCENE" \
  --viewer --gl-renderer nvidia --dds-domain 2 --out "$PWD/deploy_logs/omni_final"
```

final使用CPU物理、隔离DDS和共享runner，需要先按[部署文档](DEPLOY_G1.md)编译bridge及模拟适配器。
按1准备、2待机、3播放、B切换模拟支撑、D停止。
这是缓存参考播放；实时PICO和实机入口另见[VR文档](VR_SIM2REAL.md)，不要把实机启动作为安装验证步骤。
