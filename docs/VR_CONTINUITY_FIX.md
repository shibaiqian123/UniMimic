# VR接入连续性修正

仅229 teleop代码。没有执行真实机器人控制。

- PICO恢复原启动脚本的5帧发送。POSE接收器按连续frame_index缓存，并以50Hz推进；游标不倒退，不跨缺帧拼接。窗口不足才补末帧，重复包不刷新有效时间。最多引入9帧/180ms的参考回看延迟，属于延迟换真实窗口，不是预测未来。
- planner新轨迹按共同时间轴与旧轨迹混合8帧，位置线性混合、旋转SLERP。此实现参考本地GR00T拼接方式，并非完整移植其缓冲管理器。
- PLANNER、POSE、VR_3PT和冻结上身模式切换时，从上一条最终目标以0.5秒smoothstep过渡。原始action仍保存在观测历史，不用平滑目标冒充网络输出。停止和故障DAMP不经过过渡。
- 0.5秒是初始仿真参数，不是已验证的硬件最优值；插值也不保证平衡。policy原始目标、过渡权重和POSE延迟写入日志。
- stationary-lower-body明确提示忽略摇杆行走/转向。未自动解除该限制，POSE仍是全身控制。

验证：43项相关测试通过；隔离DDS domain 2、loopback、受支撑合成输入仿真完成准备、PLANNER、POSE、PLANNER、VR_3PT、主动停止，物理时序有效。该结果不代表实机手部误差消除或无支撑平衡通过。

复验时重启pico.py与teleop.py sim，使用原部署包。先在仿真重复模式切换和上肢动作；摇杆实验在仿真命令中省略--stationary-lower-body。禁止将模式过渡改善当作人体平移跟随已经实现。

5帧合成发送回归同样遍历上述模式；断流注入后按原路径进入VR input stream stale/DAMP（预期子进程exit 1），物理时序有效。证据目录：results/teleop_continuity_v1 与 results/teleop_continuity_5frame_disconnect。首轮日志三次策略模式切换首帧目标跳变均为0；不代表过渡后目标变化率有硬上限。
