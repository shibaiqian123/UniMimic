# BlinkMimic VR Sim2Real（229 / teleop）

参考本机 GR00T-WholeBodyControl commit
`2f07b35b3f3b9053b1f4fd2adeaf56b6bce32366`。
直接参考 launch_data_collection.py、deploy.sh、
g1_deploy_onnx_ref.cpp 的实机接管、LowState、PR控制及停止路径。
这是本机已使用的版本，不称为未经修改的官方release。

## 与GR00T相同的部分及保留差异

- PICO使用原 .venv_teleop、xrt manager、SMPL/3PT解算和标定；5556接收，5557反馈。视频使用原 .venv_data_collection，13579只用于Remote Vision；动作PC Service使用63901。
- 原C++在创建LowCmd publisher前调用CheckMode/ReleaseMode。BlinkMimic在一次yes之后调用独立C++工具，最多3次ReleaseMode；任意RPC错误退出，外层35秒上限。只读及仿真完全不调用。
- HG LowState的motor_state[0:29]保持motor顺序；两边都读取imu_state.quaternion(wxyz)、gyroscope(rad/s)，不使用可相对骨盆转动的torso IMU作本体姿态。模式来自LowState.mode_machine，LowCmd.mode_pr=0(PR)。
- 模型、93维本体观测、归一化、FSQ、动作缩放、q0、PD均为BlinkMimic。主控制50Hz，bridge500Hz。不能把SONIC kp/kd或阈值当作本模型已验证参数。
- 输入yes并完成接管后自动进行INTERP→HOLD；HOLD就绪后按A+B+X+Y进入PLANNER（POLICY）。仿真不询问yes，启动后自动准备。提前按下的启动指令不排队，需在HOLD就绪后重新按下。
- SONIC停止是停止控制并发送阻尼；streamer的OFF不保证控制进程可重新启动。本程序停止锁存DAMP，需要重开控制会话。保留BlinkMimic现有阻尼数值和看门狗，不直接复制GR00T kd=8。
- POSE单帧发送，编码器仍为10帧输入，未收到的未来用末帧补齐；并非预测真实未来，不人为等180ms。不是完整复刻GR00T的流缓冲调度器。
- 模式边缘保持上一参考航向，POSE增量按有效新包累计；不会每帧用实测姿态消除跟踪误差。历史和上一action在模态切换时不清空。
- 不启动SONIC控制策略、Dex1或采集exporter。手指不在本轮控制范围。

## 编译（代码同步后一次）

```bash
cd /home/cqu023/blinkmimic
bash scripts/deploy/bridge/build.sh \
  /home/cqu023/GR00T-WholeBodyControl/gear_sonic_deploy/thirdparty/unitree_sdk2
```

## 只读实机状态

关闭旧SONIC身体控制程序。按既有机器人操作流程连接网线与支撑；该命令不会接管运动服务，也不创建LowCmd writer。

```bash
cd /home/cqu023/blinkmimic
PYTHONPATH=src OMP_NUM_THREADS=1 .venv/bin/python scripts/deploy/run_g1.py \
  --bundle deploy_artifacts/omni_15000_walk_A049 \
  --iface enx6c1ff7b8d821 --domain 0 --stage slack \
  --hardware-profile scripts/deploy/hardware_profile.groot_g1.json \
  --read-only --seconds 10 --log-dir deploy_logs/real_readonly
```

日志中的state记录包含q、dq、quat、gyro、温度、按键、mode_machine与其他LowCmd计数。
核对实际关节姿态、IMU方向及遥控反馈；参数文件中的单位IMU外参是参考假设，不代表本机测量标定。
发现不同IMU安装或关节配置应修正参数与对应仿真，不能直接把示例标记成已验证。

## PICO启动（另一终端）

```bash
cd /home/cqu023/blinkmimic
.venv/bin/python scripts/deploy/pico.py \
  --groot-root /home/cqu023/GR00T-WholeBodyControl \
  --listen-host 192.168.11.224 --headset 192.168.11.228 \
  --video none
```

先开启XRoboToolkit PC Service及PICO身体追踪；Network中电脑IP应显示WORKING。
Head/Controller、Data/Control Send、Full body开启。
5557尚未启动会提示等待runner，待runner启动再进行三点腕部标定。
可选 `--video test`验证头显画面（不是仿真相机），或
`--video usb --camera-host 192.168.123.164`使用原USB摄像头链路。
控制时省略 `--visualize`，避免额外可视化拖慢姿态解算。
如果63901启动失败，查看PC Service日志并手工重启失效服务；程序不会在控制过程中自动重启。

## 同配置仿真

```bash
cd /home/cqu023/blinkmimic
PYTHONPATH=src OMP_NUM_THREADS=1 .venv/bin/python scripts/deploy/teleop.py sim \
  --bundle deploy_artifacts/omni_15000_walk_A049 \
  --groot-root /home/cqu023/GR00T-WholeBodyControl \
  --unitree-xml /home/cqu023/Documents/unitree_rl_lab/unitree_mujoco/unitree_robots/g1/scene_29dof.xml \
  --stage slack --stationary-lower-body \
  --viewer --gl-renderer nvidia --dds-domain 2 \
  --out deploy_logs/vr_stationary_sim
```

## 实机启动（由操作人员执行）

```bash
cd /home/cqu023/blinkmimic
PYTHONPATH=src OMP_NUM_THREADS=1 .venv/bin/python scripts/deploy/teleop.py real \
  --bundle deploy_artifacts/omni_15000_walk_A049 \
  --groot-root /home/cqu023/GR00T-WholeBodyControl \
  --iface enx6c1ff7b8d821 --stage slack \
  --hardware-profile scripts/deploy/hardware_profile.groot_g1.json \
  --stationary-lower-body \
  --out deploy_logs/vr_stationary_real
```

输入一次yes才释放原运动服务并建立控制链路。按原设备规范由操作人员完成调试/低层模式准备；
不额外执行SONIC deploy.sh，它会成为第二个控制者。

1. 保持支撑，输入yes后机器人自动INTERP→HOLD；就绪后按A+B+X+Y进入PLANNER/POLICY。等待期间可以连接VR；策略运行后的断流保护保持不变。
2. PLANNER静止；对齐双臂后按A+X测试小幅POSE，再返回PLANNER。
3. 点击左摇杆进入VR_3PT，按真实关节反馈标定手腕，再做小幅双臂动作。
4. POSE中B+Y可进入冻结上身的PLANNER；左摇杆点击在planner与3PT之间切换。
5. 首轮不走动、不蹲起。stationary选项固定planner idle、忽略移动/步态/转向请求，不限制POSE人体腿部或人体根朝向；POSE操作者必须保持腿部静止。
6. 受支撑试验正常后，由操作者决定进行脚着地、支撑松弛的短时站立。
7. 正常结束先恢复支撑，再A+B+X+Y停止；G1遥控L2+B也可停止。阻尼不保证无支撑站立，故障不会自动恢复。

可选增益覆盖（仿真和实机相同语法，默认全1）：
`--motor-kp-scale '4,10=1.5;15=1.1' --motor-kd-scale '4,10=1.5'`。
这是语法示例，不是推荐参数。有效kp/kd写入runner首条日志，固定指纹仍指原部署包。

## 结果边界

代码/接口、模拟VR的隔离DDS仿真、真实VR仿真、受支撑实机、支撑松弛站立分开记录。
本轮自动执行的只是代码测试与隔离仿真；无真实机器人接管或动作测试。
日志保存原VR包、LowState、输入模态/时间、补齐帧数、原action、目标角、推理及周期时间。
阶段不同的结果不得合并为“上机通过”。
