# sim2sim implementation validation — host 229

Candidate: deploy_artifacts/sim2sim_3500_walk1 (ckpt_003500.pt, walk1_subject1.csv).
The full reports retain the bundle fingerprint and compiled-plant fingerprints.

## Executed
- 18 tests passed (15 existing deployment tests plus actuator-order/torque saturation/delay queue/resolution tests).
- New bundle export: ONNX/PyTorch action maximum absolute difference 2.86102294921875e-6.
- Official Unitree G1 torque-motor G2: completed 13065 control transitions / 261.30 simulated seconds; passed current failure/guard criteria. Joint tracking RMSE 0.094536993 rad. Maximum observed physics-step velocity ratio 0.499143.
- Official-model full preparation/standby/motion/final-standby: completed 13801 active control steps / 276.22 simulated seconds, ended in STANDBY_POST, passed current guards. Joint tracking RMSE 0.094866984 rad.
- G1 same-description CPU baseline: 15-second smoke run with reference velocities. Intentionally incomplete, not a full pass or GPU numerical-parity result.
- Two one-second sweep cases (0 and 10 ms delay): executed; intentionally incomplete and not counted as successful complete motions.
- Timing: warmed ONNX inference p99 2.6198 ms; this excludes physics, I/O and full control-loop time.
- Model comparison found total mass 33.341143 kg (training description) versus 35.112142 kg (official Unitree model).
- CLI help, Python compilation and git diff whitespace checks passed.

## Not executed / not implied
- No GPU training-reference recording or numerical trajectory comparison: host 229 GPU was already at 100% utilization.
- No full robustness grid, all-40-motion evaluation, interactive GUI keyboard verification or real hardware run.
- This is synchronous CPU MuJoCo, with explicit 1 kHz PD on the official model. It has no DDS timing, network traffic or actual motor/IMU errors.
- The 3500 checkpoint / official-model success does not clear the older 1600 checkpoint DDS-physics failure. Existing hardware acceptance files/gates were not modified.
- No claim of README normalized return or physical robot success is made from these runs.

Reports: deploy_logs/sim2sim_smoke_g2, sim2sim_3500_standby_full, sim2sim_smoke_g1, sim2sim_smoke_compare, sim2sim_smoke_timing, sim2sim_sweep_smoke_v1.

## GPU backend addition

19 tests passed. One-replica/10-step bf16 and two-replica/20-step fp32 CUDA runs completed their requested short windows without observed failures; both intentionally report completed=false. Physics and policy devices were cuda:0 on RTX 4090. EGL rendering of the selected GPU replica was inspected. Reports are in deploy_logs/warp_gpu_smoke and deploy_logs/warp_gpu_render_smoke. Full GPU motion performance and interactive window keys remain unverified; the GPU was already occupied. This does not change CPU sim2sim or hardware acceptance.

## Official-model GPU g2 addition

The official Unitree torque-motor model is now stepped directly by MuJoCo Warp (no mjlab simulation), with the 1 kHz PD and effort/motor limit clipping computed by CUDA kernels. Policy can run on CUDA from the matching checkpoint. Single-world, target-side g2/standby/sweep use this backend; g1 keeps its separate original-training GPU path.

Executed on 229:
- Official GPU g2 closed-loop smoke: 10 control steps / 0.2 s, physics_device=cuda:0, policy_device=cuda:0, CUDA Graph enabled; no observed failure, intentionally incomplete.
- Same-model CPU/GPU open-loop comparison: 20 control steps / 0.4 s per case, at 0 and 10 ms command latency. Both passed the declared short-run tolerances; command schedules matched exactly.
- GPU policy / CPU ONNX and pure CUDA PD clipping/index mapping checks passed.
- Existing 19 regression tests passed; no full target-side GPU motion, robustness grid or interactive window acceptance run was performed.
- MULTICCD cylinder pair limitations are retained as backend warnings. CPU/GPU collision algorithms and precision are not assumed identical.
- Existing DDS/hardware acceptance reports were not modified.

Numeric results:
- delay 0.0 ms: max joint error 0.000440912797 rad, root-position error 2.7727704e-05 m.
- delay 10.0 ms: max joint error 0.000470319696 rad, root-position error 2.68675627e-05 m.
- policy max action difference 2.62260437e-06; PD kernel difference 0.0.

Reports: deploy_logs/official_warp_smoke and deploy_logs/official_warp_comparison.

## Viewer/readback correction

23 tests passed. Same-state full vs compact GPU readback (20 measured samples after warmup): median 4.9745 ms vs 0.07521 ms; state/counter errors zero. This is a readback-only benchmark, not viewer FPS. Physics now runs outside the viewer lock against separate render data; display is capped at 30 Hz by default, and control computation is included in its scheduled period. Report: deploy_logs/readback_optimization/report.json. Initial compilation and graphical-desktop smoothness remain separate concerns.

## OpenGL software-rendering diagnosis and correction

The actual viewer on host 229 DISPLAY=:1 was using llvmpipe (LLVM 15), with Accelerated: no and 32 llvmpipe rendering threads. CUDA physics does not choose the OpenGL renderer.

Interactive sim2sim now defaults to --gl-renderer auto. If glxinfo detects software rendering, it tests NVIDIA PRIME offload and, when confirmed, applies __NV_PRIME_RENDER_OFFLOAD=1 and __GLX_VENDOR_LIBRARY_NAME=nvidia to this process only. --gl-renderer nvidia requires NVIDIA and fails rather than silently retaining software rendering; --gl-renderer system preserves the desktop selection. No system driver/display configuration is changed.

Verified in an actual hidden GLFW context, after importing MuJoCo as the CLI does:
OpenGL vendor: NVIDIA Corporation
OpenGL renderer: NVIDIA GeForce RTX 4090/PCIe/SSE2

27 regression tests passed. This confirms hardware rendering selection, not a measured full-window FPS. Close the old software-rendered window and restart with a new --out directory; the old GL context cannot be changed in place.
