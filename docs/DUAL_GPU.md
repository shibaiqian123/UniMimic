# 单机双GPU训练（保持全局工作量）

代码只在229修改。32同步代码后由操作者启动；本功能不改部署模型接口，不启动机器人。

## 训练语义

采用一GPU一进程、NCCL手动梯度平均；每个rank持有独立仿真环境和经验池，world model / policy / critic完整复制。参考SuperMimic的数据并行思路；没有搬运XMIMIC的JAX Actor/Learner延迟流水线。

所有num_envs、wm_batch、pi_batch参数都是全局值。默认4096环境拆为每卡2048；main配方wm_batch=2048、pi_batch=2048均拆为每卡1024。horizon、UTD、warmup64、更新次数、3500迭代及学习率不变，无sqrt(world_size)学习率放大。每个迭代全球采样量不翻倍。

WORLD_SIZE=2时生产入口要求两张不同GPU，并在import Torch/Warp前按LOCAL_RANK隔离可见设备。各rank在自己的可见cuda:0运行；单卡保持原路径。全局batch不能均分或critic minibatch不能整分时明确报错。首版支持当前无状态传感器、无外部push的G1配置，分布式采样要求bin_fix。

梯度先归约，再裁剪和更新。world model原损失为sum(masked_error)/batch_size，保留这个定义，不改成除以有效帧数量；报告末帧误差则合并有效误差和/计数。非有限梯度全rank跳过对应优化器更新，policy EMA也不前进。

归一化使用相对共同running mean的新增样本和、平方和、计数全局归约；不平均标准差。每个采样步合并新结束episode的按bin统计，做一次EMA更新和失败衰减，不累加历史buffer。rank0运行probe，迭代边界广播focus分布。不同rank随机流不同，初始化模型一致。

## 双A100启动（32）

```bash
cd /share/2025slx/UniMimic
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=4 .venv/bin/python scripts/train_latent.py \
  --gpus 2 --latent-mode fsq64 \
  --motion-dir /share/2025slx/UniMimic/data/lafan1_g1 \
  --log-dir logs/lafan_fsq64_a100_2gpu_s0 \
  --num_envs 4096 --iterations 3500 --seed 0 --tensorboard
```

continuous64替换模式和输出目录。continuous32也受支持，wrapper保留main中的参考动作旁路，遵循旧32D配方。
单卡省略--gpus或设为1。不应同时在同一日志目录启动两个实验。

启动器使用torch.distributed.run；可以先追加--dry-run查看实际命令。训练中现有--no_sync指“减少本地CPU/GPU同步”，不是关闭分布式梯度同步。

## CUDA Graph与诊断

双卡采用独立本地阶段：观测、物理/经验写入、重置、world model反向、policy反向、critic反向。归一化与梯度通信位于图外，优化器也在归约后更新。每阶段统一预热/捕获；保留torch.compile、BF16、fused Adam和Muon。

--dist-no-graphs为显式双卡无图诊断，关闭新训练/采样图，保留编译；不会默认静默退化成无图实现。
--dist-audit在日志周期核对所有rank的模型、buffer、EMA和优化器精确摘要，适合短正确性验证，会产生额外拷贝和同步，不用于正式测速。

32双卡短测试示例（同样需要先同步代码）：
```bash
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=4 .venv/bin/python scripts/train_latent.py \
  --gpus 2 --latent-mode fsq64 --log-dir logs/dual_smoke_s0 \
  --num_envs 64 --wm_batch 64 --pi_batch 64 --buffer_steps 32 \
  --iterations 12 --log_every 2 --ckpt_every 12 --ckpt_minutes 0 \
  --eprobe_start 0 --eprobe_every 12 --eprobe_async 100 --dist-audit
```

检查distributed_audit.jsonl每次passed=true、两rank哈希一致；distributed_phases.json中各阶段captured=true。短测试只验证代码路径，不证明4096环境性能或收敛。

## 日志、恢复与模型交付

rank0写train_log.csv、TensorBoard和checkpoint。args.json保存全局参数与实际每rank尺寸；sim_steps是全局转移数。
distributed_runtime.json记录配置和warmup时间；distributed_perf.jsonl记录每rank采样、学习、梯度/统计归约、广播及barrier通信时间及峰值分配显存；通信时间包含在阶段时间内，不能再与阶段时间相加。
latent_metrics.jsonl合并各rank分片上的token样本统计。

last.pt包含共同模型/优化器/EMA，以及world size、各rank Python/NumPy/Torch/CUDA RNG和共同采样/LR状态。
恢复要求world size和关键全局batch设置一致，不能把旧单卡last.pt直接作为双卡完整续训。仍是warm restart：重建仿真并重新填充buffer，probe进行中的扫描重启，不声称逐步精确恢复。
普通ckpt_*.pt保留评估/ONNX导出接口。

```bash
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=4 .venv/bin/python scripts/train_latent.py \
  --gpus 2 --latent-mode fsq64 --log-dir logs/lafan_fsq64_a100_2gpu_s0 \
  --num_envs 4096 --iterations 7000 --seed 0 --tensorboard \
  --resume logs/lafan_fsq64_a100_2gpu_s0/last.pt
```
网络宽度和其他训练参数需与原实验一致；测试小配置的恢复也必须原样保留小batch等参数。

## 同预算测速

以下工具会依次启动单卡和双卡短训练，默认各250迭代，排除前50迭代，比较后200迭代：
```bash
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=4 .venv/bin/python tools/benchmark_dual.py \
  --latent-mode fsq64 --num-envs 4096 --motion-dir data/lafan1_g1 \
  --out logs/dual_benchmark_s0
```

输出comparison.json，包含同全局工作量的it/s、采样吞吐、speedup及显存。可先--dry-run。默认probe_start仍为1000，所以默认250迭代测速不包含活动probe扫描；如需长时间稳态评估，增加排除/测量窗口并记录条件。

通信、较小本地batch和rank0 probe可能限制加速；若speedup<=1，报告为没有加速，不能通过增加全局样本量掩盖。
3500迭代能力对照继续用tools/eval_metrics.py：比较回报、失败数、跟踪误差，单个seed不能证明统计等价。

## 测试边界

tests/test_distributed.py在两个CPU/Gloo进程上检查数值同步，不能替代NCCL双GPU验收。
tools/test_distributed_local.py明确标注TEST ONLY，在一张GPU上验证新计算阶段和图捕获，不是生产降级模式或双卡测速。
本轮真实双A100测试、测速及完整训练结果见实际验收记录；没有运行的项目必须保持待验证。
