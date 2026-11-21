# BlinkMimic 上机前最终 sim2sim

当前主路径是 **CPU 官方 G1 模型 + 模拟传感器 + DDS + 与真机相同的 runner**。GPU g2、mjlab g1 和直接真值 viewer 仍是诊断工具，不替代此流程。只在 229 开发；本实现未运行真实机器人。

## 一条命令启动带窗口的最终仿真

在 229 图形桌面执行：
```bash
cd /home/cqu023/blinkmimic
OMP_NUM_THREADS=1 .venv/bin/python scripts/deploy/sim2sim.py \
  --cmd final \
  --bundle deploy_artifacts/final_3500_walk1 \
  --unitree-xml /home/cqu023/Documents/unitree_rl_lab/unitree_mujoco/unitree_robots/g1/scene_29dof.xml \
  --viewer --gl-renderer nvidia \
  --out deploy_logs/final
```

输出自动放到时间戳子目录，启动时打印 FINAL SIM SESSION。无需先生成验证报告或填写“已验证”开关。

键盘：
- 1：等价 L2+上，准备起始姿态。
- 2：等价 R1+A，进入策略待机。
- B：撤除/恢复模拟支撑，仅控制仿真装置。正式播放时应为 OFF。恢复支撑会锚定机器人当前位置与朝向，不会把机器人拉回起点。
- 3：等价 R1+X，稳定后开始动作。
- D：等价 L2+B，停止并进入 DAMP。正常收尾时，结束待机后先用 B 恢复模拟支撑，再按 D；紧急 D 始终立即生效。
- G：参考骨架；F：相机跟随；鼠标旋转/移动/缩放视角。

按键不是直接调用状态机：所有动作按键经同一个脉冲适配器，编码成 LowState.wireless_remote，再由真实 runner 处理。每次按钮脉冲 150ms，之后至少释放 100ms；重复输入不叠加，D 优先。失焦释放普通按钮，已请求的停止脉冲会完整发送。窗口显示 runner 的实际状态和拒绝切换原因。

final 模式没有暂停物理、慢放或跳帧。关闭窗口会经按钮通路请求停止，再结束本次会话。已有诊断 viewer 的暂停/单步功能不变。

## 无窗口仿真和独立键盘客户端

第一终端运行上面的命令，去掉 --viewer 和 --gl-renderer。第二终端使用第一终端打印的会话目录：
```bash
cd /home/cqu023/blinkmimic
.venv/bin/python scripts/deploy/sim_keyboard.py \
  --session deploy_logs/final/实际时间戳目录
```
键位仍是 1、2、3、D、B。Ctrl-C 请求停止并退出键盘客户端。客户端只使用本地 Unix socket，不创建 DDS 发布器，也不能连接真机 domain。

自动回归可给 final 命令添加 --auto；它发送相同的按键事件，不绕过状态机。--duration N 以墙钟时间请求停止；提前结束不是完整动作通过。

## 与真机共用的数据和控制路径

- 物理进程按 1ms 墙钟周期独立推进，不等待推理或显示。
- 官方模型使用力矩电机；每个物理步计算接收到的 LowCmd 的 kp、kd、q、dq、tau，并应用电机力矩限制。
- 模拟侧 C++ DDS 适配器以 1kHz 发布 LowState，独立接收 LowCmd。真实 runner 的原 C++ bridge 以 500Hz 发命令，策略以 50Hz 运行。
- 仿真从默认 q0 姿态开始，初期由模拟支撑承载；收到命令前使用与 PASSIVE 相同的 kd=3 阻尼，不偷偷使用动作首帧或隐藏站立控制器。
- 参考动作、姿态准备、静态待机、目标窗口、参考帧和上一动作均由同一个 Controller/DeployCore 管理。
- 运行时输入只来自共同的 LowState 解析函数。6D 由消息中的 IMU 四元数、同一个 IMU→骨盆转换和固定初始航向对齐计算；参考朝向只来自动作包。
- 仿真初始 IMU 估计按 SuperMimic 参考模型初始化；随后只积分主 IMU 的限幅陀螺仪（±500°/s），不每帧用真实姿态校正。真值姿态仅用于评估估计误差。
- 温度是正常值/故障注入输入，没有针对这台 G1 的热模型。IMU/通信参考参数也未标定到本机；这一事实记录在报告中，不作为人工认证门槛。

模拟器 IPC 中的 qpos/qvel 真值只供物理日志和窗口读取，不经 LowState 以外通道传给 runner。即使状态发布中断，物理仍推进。

## 对应实机命令

先按原 SuperMimic 路径准备 G1 低层控制环境、网卡和支撑，再由现场操作者执行：
```bash
cd /home/cqu023/blinkmimic
PYTHONPATH=src .venv/bin/python scripts/deploy/run_g1.py \
  --bundle deploy_artifacts/final_3500_walk1 \
  --iface 实际机器人网卡 --domain 0 --stage air \
  --log-dir deploy_logs/real
```

只保留一次 yes 确认和遥控操作。可选 --hardware-profile 用于覆盖 IMU 坐标参数，不要求 verified_on_this_robot。五份旧报告不再参与启动判断。

保留通信看门狗、CRC、模式/状态有效性、遥控停止及参考路径的运行时保护。取消额外“超过硬限位 0.05rad 就停”和默认位置目标硬裁剪；保留力矩限制及物理电机/关节自身限制。HOLD 不满足姿态条件时返回原因，允许稍后重新按 2，不直接 DAMP。

准备增益和策略增益来自同一部署包。当前候选 final_3500_walk1 显式使用 preparation_gain_scale=4.5（准备 kp×4.5、kd×sqrt(4.5)），策略仍用原训练增益。原因是原 scale=4 的初始 T 姿态肩部静态误差约 0.107rad，无法满足参考路径的 0.1rad 到位条件；没有放宽条件。此增益还未在真机验证。

其他模型可使用 tools/export_deploy.py --preparation-gain-scale 设置同一共享配置。不要将仿真通过视为本机标定或真实硬件通过。

## 构建、结果和范围

构建一次 bridge：
```bash
bash scripts/deploy/bridge/build.sh \
  /home/cqu023/GR00T-WholeBodyControl/gear_sonic_deploy/thirdparty/unitree_sdk2
```
它同时构建真实 runner bridge 和模拟侧桥接。仿真侧拒绝 domain 0，只使用 domain 1/2 和 lo。final 的运行不需要 Python Unitree DDS 包；DDS 工作由 C++ 适配器完成。

每个会话包含：
- report.json：流程、通信、实时性和实际验证边界。
- physics.npz / physics_report.json：物理真值、模拟 IMU、支撑状态和时序质量。
- runner/时间戳/runner.jsonl：真实入口读到的状态、观测/6D、动作、目标角与状态机。
- input.json：键盘脉冲请求及是否接收。
- runner_status.json、physics_status.json：窗口与终端显示状态。

成功要求完整流程完成、POLICY 期间无支撑、无异常退出，并且物理时间与墙钟的差值 P99 和末次差值均不超过 20ms。结束后另用真值评估高度/姿态跟踪误差（0.25m / R[2,2] 差 0.8）和关节 RMSE；这些量只写入最终结果，不回送控制器，也不在运行中形成另一套停止逻辑。正常停止段还记录支撑是否已恢复及阻尼后的物理状态变化；紧急停止不会等待这些条件。这里的时间差取活动物理循环采样，不包含结束后的轨迹压缩写盘。报告失败不会变成下一次启动的许可证要求。

实现来源：SuperMimic-Repro-Sim2Real 的 bridge 与 IMU 参考模型，许可证保留在 scripts/deploy/bridge/LICENSE；Unitree CRC 的 BSD 许可证保留在其 LICENSES 子目录。旧 docs/VALIDATION.md 记录的是旧实现，不应作为当前启动说明。


## 2026-11-21 首次实机准备阶段诊断

来源：\`deploy_logs/real_proprio93_air/20261121_191918_285249/runner.jsonl\`。
本次仅到 PASSIVE → INTERP → HOLD → DAMP，未执行策略。
停止原因是应用层 80°C 阈值，不能记录为遥控正常停止或策略通过。

- 最后 HOLD：左肩 roll 参考 1.667765 rad，实际 1.509677 rad，误差 0.158088 rad；
  右肩 roll 参考 -1.620647 rad，实际 -1.463801 rad，误差 -0.156846 rad。
- 准备 Kp=64.127804，Kd=1.924510；计算 PD 力矩分别约 +10.14 / -10.04 N·m。
  air 阶段两肩限值为 12.5 N·m，当时 n_clamped=0。
  以上是基于采样状态计算的指令力矩，不是已测得的实际力矩。
- 右肩最后 HOLD 温度为 [45,79]°C，下一次状态到 [45,80]°C 后进入 DAMP。
  Unitree SDK2 的 terminations.hpp 将 temperature[0] 定义为 casing，
  temperature[1] 定义为 winding；不能解释成外壳达到 80°C。
- 官方示例分别提供外壳 85°C、绕组 120°C 的默认检测阈值，但这些不是
  本机标定结果；本次未据此放宽现有 80°C 应用阈值。
- 旧日志没有保存 tau_est，不能据此判定重力负载、外部干涉或机械限制中的哪一种
  是主要原因。准备阶段长期保持该姿态没有消除误差，却持续升温。

来源：https://github.com/unitreerobotics/unitree_sdk2/blob/main/include/unitree/robot/g1/common/terminations.hpp

### 新增日志字段

- s.tau：LowState 的 tau_est，单位 N·m。只读模式同样保存。
- result.requested_target：力矩限幅前的目标角。
- result.requested_target_error：原目标角减当前实测角。
- result.pd_torque_estimate：按实际下发目标角、Kp、Kd 和本次采样 q/dq 计算，
  不代表电机实际输出，也不代表 bridge 两次 runner 更新之间每一时刻的力矩。
- result.torque_limit：本阶段用于目标限幅的力矩上限。
- 启动记录包含温度通道名称及来源；过温信息包含关节、通道、读数和阈值。
- 起始姿态拒绝信息包含最大偏差关节、偏差及容差。

本次修改仅补齐诊断，无新增启动门槛；没有修改增益、动作包、温度阈值或遥控流程。
离线回归：test_deploy_runtime.py 与 test_final_pipeline.py，26 passed。
没有自动重启实机控制。后续先核对肩部、吊带及末端负载，再依据新增遥测判断；
不能通过本次日志修复宣称已解决物理升温或准备姿态偏差。


## Dex1-1 URDF 仿真对照（2026-11-21）

新增 tools/convert_deploy_urdf.py，将指定的 URDF 转为独立部署场景。
使用 /home/cqu023/unitree_ros/robots/g1_description/g1_29dof_mode_15_with_dex1_1.urdf；
保留 link 惯量、质量、几何和关节范围，四个 Dex1 手指滑动关节固定在零位。
这表示双侧安装 Dex1-1 的固定夹爪形态，不是左侧拆除后的配置。
转换不修改源 URDF，也不覆盖策略部署包。场景使用原目录中的 mesh 绝对路径。

URDF 没有充分描述电机转子/传动等效动力学；通过必填 --dynamics-xml 显式沿用
原官方 MuJoCo 模型同名关节的 armature、damping、frictionloss。
scene_motor.json 记录来源指纹与每关节参数，不能视为本机辨识值。
原始无上述参数的 scene.xml 仅作为失败诊断留档，出现腕关节振荡，不作为推荐入口。

模型质量 34.129935 kg，与 URDF 总质量一致；旧官方场景约 35.112142 kg，
两者并非仅手部不同，不能将全部差异归因于 Dex1-1。
IMU 位于骨盆原点并对齐骨盆轴，仍属参考传感器配置；温度仍为合成输入，
本仿真不预测绕组升温，也不代表实机验证。

转换命令（229）：
```bash
cd /home/cqu023/blinkmimic
PYTHONPATH=src .venv/bin/python tools/convert_deploy_urdf.py \
  --urdf /home/cqu023/unitree_ros/robots/g1_description/g1_29dof_mode_15_with_dex1_1.urdf \
  --dynamics-xml /home/cqu023/Documents/unitree_rl_lab/unitree_mujoco/unitree_robots/g1/scene_29dof.xml \
  --out deploy_artifacts/models/g1_dex1_1/scene_motor.xml
```

交互仿真（仅回环接口，DDS domain 2）：
```bash
OMP_NUM_THREADS=1 .venv/bin/python scripts/deploy/sim2sim.py \
  --cmd final --dds-domain 2 --stage ground \
  --bundle deploy_artifacts/fsq64_proprio93_s0_3500_walk1 \
  --unitree-xml deploy_artifacts/models/g1_dex1_1/scene_motor.xml \
  --viewer --gl-renderer nvidia \
  --out deploy_logs/final_dex1_1_view
```
1 准备，2 策略待机，B 切换模拟支撑，3 播放，D 阻尼停止。
保持原始启动条件与控制增益，不为完成动作而放宽容差。

自动 final 结果：deploy_logs/final_dex1_1_motor/20261121_195114_314506/report.json。
PASSIVE → INTERP → HOLD → DAMP；timing_valid=true，passed=false，未进入策略。
HOLD 最后100条样本中位数：左肩偏差 +0.120465 rad / 力矩 +7.725176 N·m，
右肩偏差 -0.120244 rad / 力矩 -7.711009 N·m；腕 yaw 偏差约0.005 rad。
拒绝原因为 left_shoulder_roll_joint error=0.120rad limit=0.100rad。
该结果支持负载导致纯PD准备姿态偏差，但仍未复现实机约10 N·m肩部力矩，
不能宣称模型已完全标定或walk已通过。原阈值、增益及策略包未变。


## 控制运行时延迟改造（2026-11-21）

模型、PD、50Hz/500Hz频率和20ms状态新鲜度阈值不变。

- ONNX初始化时预热16次，不推进控制器历史或参考帧，不发布DDS命令。
- 日志与UI状态文件在独立进程中序列化和写盘；控制线程使用有界非阻塞队列。
  队列满或写进程失效时计数丢失，不阻塞控制；结束时 logging_report.json 明确完整性。
  日志不完整不计为仿真验收通过，不新增真机启动认证。
- policy_inference_thread_cpu_ms 和 controller_thread_cpu_ms 与原墙钟耗时并列记录。
  单线程CPU推理下，两者的差异可帮助识别调度等待，不能单凭差值认定具体系统服务。
- final 默认 --runtime-affinity auto：根据允许CPU集合与物理核拓扑，给物理、DDS、
  runner、日志分配CPU；viewer避开前三者的SMT兄弟核。
  session.json 记录实际分配。--runtime-affinity none 用于无亲和性对照。
  这是进程亲和性，不是内核CPU独占隔离，也不是硬实时保证。
- runner 可单独指定 --cpu-affinity、--logger-cpus、--bridge-cpus，均为逗号分隔CPU号；
  真机默认不强行修改亲和性，须按该机器实际拓扑显式选择。
- 超时文案改为 state expired after control，并注明实际阶段和状态年龄。
- report.json 的 physics_timing_valid 单独代表物理墙钟检查；
  timing_valid 还要求runner没有状态过期或控制周期超时，避免此前的误导。

现有final命令无需更换模型或新增参数即可使用以上默认优化。
无窗口自动验证可用 --auto；窗口操作及遥控状态机不变。

补充：控制循环及1kHz物理循环在初始化后先执行一次GC，再暂时禁用周期循环GC，
退出时恢复；正常无环对象仍由引用计数释放。物理UI状态写入也移至独立进程。
新增 cycle_interval_ms、cycle_thread_cpu_ms、wake_lateness_ms、previous_enqueue_ms，
用于区分循环唤醒延迟与控制计算延迟。物理时序报告增加 lag_max_s，
验收同时检查最大延迟与P99，避免低频尖峰被P99掩盖。

可选 --rt-priority 10 仅对隔离final的runner、DDS bridge与物理进程使用SCHED_RR；
viewer、日志进程保持普通调度。默认0，不自动提权，也不默默回退。
当前229 RLIMIT_RTPRIO=0，未执行实时优先级验证；无权限时在创建仿真进程前报错。
若操作员决定测试，可在自己的229终端临时授权该shell：

    sudo prlimit --pid "$$" --rtprio=10:10

然后在原final命令增加 --rt-priority 10。该授权随shell退出结束，不改系统服务或永久配置。
不要用sudo运行整套Python部署程序，也不要给Python解释器设置全局capability。

验证记录：
- 31项运行时/状态机/消息回归通过。
- runtime_fix_auto/20261121_204015_359919：约146秒播放后循环超时，未完成。
- runtime_fix_view/20261121_204418_364177：推理最大3.42ms，物理tick停滞触发stale state。
- runtime_fix_view_v2/20261121_204540_365899：推理最大11.86ms，控制最大12.77ms，
  约53秒播放后循环间隔69.21ms，线程CPU仅3.49ms，wake_lateness=49.31ms；
  日志零丢失，完整流程未通过。新记录还增加cycle_process_cpu_ms，用于后续区分进程内等待。
以上不能证明延迟尖峰已消除，SCHED_RR仍待有权限的环境验证。未启动真实机器人。

### 实际调度与睡眠诊断

runner初始化后打印并记录实际SCHED策略、优先级、CPU亲和性、线程清单、
RLIMIT_RTPRIO、timer slack和内核RT配额。以实际读取值为准，不再只看命令行请求值。
每轮的previous_sleep记录上一轮请求睡眠时长、实际返回时间、超出时长、
睡眠期间线程CPU时间、主动/被动上下文切换和页错误；after_sleep_ms区分
睡眠返回后到下一轮开始的间隔。首次DAMP发送后额外记录fault_scheduling。
这些只是证据字段，不延后DAMP、不修改状态超时门槛。

内核sched_schedstats关闭时，runqueue_wait_ns为null而非0；
此时不能根据零等待计数声称线程没有被调度延迟，也不能仅凭sleep返回晚区分
定时器唤醒延迟与唤醒后的调度/GIL等待。需要内核事件跟踪才能进一步确认。

新增scripts/deploy/timing_probe.py，仅测试睡眠唤醒，不加载模型，不连接DDS、不发电机命令。
普通调度例子：
    .venv/bin/python scripts/deploy/timing_probe.py --cpu 62 --seconds 60 --out deploy_logs/timing_probe_normal.json
同一已授予RLIMIT_RTPRIO的终端可以加 --rt-priority 10 做对照；
无权限明确报错，实际策略会打印出来。不要把单次探针正常当成整套部署验收。
