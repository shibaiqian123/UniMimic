# BlinkMimic sim2sim

**最终上机前流程使用 `--cmd final`，默认 CPU 官方模型并共用真机 runner。窗口和无窗口键盘说明见 [DEPLOY_G1.md](DEPLOY_G1.md)。下面 g1/g2/standby/GPU 内容保留为诊断工具；旧报告不再作为启动门槛。**

This is a separate native-MuJoCo deployment test harness, inspired by the gate structure and explicit PD/delay design of SuperMimic-Repro-Sim2Real/scripts/deploy/sim2sim.py (inspected on host 229). It uses BlinkMimic's own Bundle, DeployCore, 99D observations, frame-major history, 730D goals and trained parameters. No SuperMimic model/680D goal/930D observation is loaded.

The default CPU runs use native MuJoCo + NumPy + ONNX Runtime. GPU options are documented below. The window uses OpenGL. There is no DDS publisher or robot connection. The optional training-reference recorder requires the original PyTorch/mjlab/Warp training environment.

## Quick start on 229

A fresh 3500-iteration walk1 bundle was exported for this implementation:
```bash
cd /home/cqu023/blinkmimic
export G1_UNITREE_XML=/home/cqu023/Documents/unitree_rl_lab/unitree_mujoco/unitree_robots/g1/scene_29dof.xml

OMP_NUM_THREADS=1 .venv/bin/python scripts/deploy/sim2sim.py \
  --bundle deploy_artifacts/sim2sim_3500_walk1 \
  --cmd g2 --viewer --speed 1 \
  --out deploy_logs/sim2sim_view_3500
```

Run from a graphical desktop on 229. --out must be a NEW directory. G2 starts the policy directly from the reference first pose at rest; it does not wait for prepare/arm buttons. Solid robot = official Unitree physics model; blue skeleton = motion reference. SPACE pause/resume, N one 20 ms control step, +/- speed, G reference, F camera follow, R reset. Failures freeze playback and remain in report.json. Closing early saves an incomplete result, not a pass.

To see preparation and the full control-state sequence, use --cmd standby instead of g2. This uses the existing Controller, simulated initial support, stable-standby gate and final standby. It is still a synchronous physics test, not the real-time DDS path.

## Layers

- g1: exported training-description model, position actuators, 5 ms physics, 50 Hz policy; exact reference velocities and hard-limit-clipped initial joints as in the original evaluation. No deployment target clamps or altered gains. A --reference trajectory is required to claim training-environment numerical agreement; without one this is only a same-model CPU baseline.
- g2: official Unitree G1 model with torque motors, 1 ms explicit PD. Commands/gains are queued by simulation time, effort-clipped, then clipped to the model motor ctrlrange. Initial velocities are zero and the feet are placed on the ground using named collision geometry. Deployment state guards and target limiting apply.
- standby: official model plus actual BlinkMimic Controller: prepare, HOLD, static standby, motion, final standby. Same exported preparation gains and no reduced safety thresholds.
- compare: masses, joint ranges, armature, damping and motor limits of the two models. Records compiled-model fingerprints.
- timing: warmed ONNX inference only; p50/p99/max are not full-loop latency.
- sweep: serial Cartesian grid of command delay, kp multiplier, torso mass increment and friction. Defaults: 0/5/10/20 ms, 0.85/1/1.15 kp, +0/+1.8 kg, mu 0.6/1.0. Run small grids before the full 48-case sweep.

Examples without a window:
```bash
.venv/bin/python scripts/deploy/sim2sim.py \
  --bundle deploy_artifacts/sim2sim_3500_walk1 --cmd g2 \
  --out deploy_logs/sim2sim_g2_3500

.venv/bin/python scripts/deploy/sim2sim.py \
  --bundle deploy_artifacts/sim2sim_3500_walk1 --cmd standby \
  --out deploy_logs/sim2sim_standby_3500

.venv/bin/python scripts/deploy/sim2sim.py \
  --bundle deploy_artifacts/sim2sim_3500_walk1 --cmd compare \
  --train-xml deploy_artifacts/sim2sim_3500_walk1/train_scene.xml \
  --out deploy_logs/sim2sim_compare

.venv/bin/python scripts/deploy/sim2sim.py \
  --bundle deploy_artifacts/sim2sim_3500_walk1 --cmd sweep \
  --latencies 0,10 --kp-scales 1 --masses 0 --frictions 1 \
  --max-seconds 10 --out deploy_logs/sim2sim_sweep_smoke
```

--max-seconds deliberately truncates a run: it reports incomplete and does not count as a successful full motion. CLI exit codes for individual headless runs: 0 passed, 1 stopped by failure, 2 incomplete. --diagnostic-continue is an explicit G2-only diagnostic that records deployment state-guard violations instead of immediately stopping; a run with violations never passes. It does not change the hardware runner or any arming report.

## Export and training-reference comparison

Existing deployment packages without qd/root_v/root_w still work for G2, but must be re-exported for G1:
```bash
PYTHONPATH=src .venv/bin/python tools/export_deploy.py \
  --ckpt logs/lafan1_csv_a100_s0/ckpt_003500.pt \
  --motion data/lafan1_g1/walk1_subject1.csv \
  --out deploy_artifacts/NEW_BUNDLE

PYTHONPATH=src .venv/bin/python tools/prepare_deploy_sim.py \
  --bundle deploy_artifacts/NEW_BUNDLE \
  --out deploy_artifacts/NEW_BUNDLE/train_scene.xml
```

Record a rollout in the original GPU environment only when the GPU is available:
```bash
OMP_NUM_THREADS=4 PYTHONPATH=src .venv/bin/python tools/record_sim2sim_reference.py \
  --bundle deploy_artifacts/NEW_BUNDLE \
  --ckpt logs/lafan1_csv_a100_s0/ckpt_003500.pt \
  --out deploy_logs/reference_3500.npz

.venv/bin/python scripts/deploy/sim2sim.py \
  --bundle deploy_artifacts/NEW_BUNDLE --cmd g1 \
  --train-xml deploy_artifacts/NEW_BUNDLE/train_scene.xml \
  --reference deploy_logs/reference_3500.npz \
  --out deploy_logs/sim2sim_g1_comparison
```

The recorder is fp32 with TF32 disabled, unlike the bf16 benchmark configuration. It writes post-step q/root positions and frame indices with the bundle fingerprint. Comparison requires matching frames and bundle; thresholds are 0.05 rad joint RMSE and 0.10 m root-XY RMSE. An incomplete recording cannot pass the full comparison. No reference was recorded during implementation because host 229 GPU was occupied.

## Outputs and interpretation

Each run writes report.json and trajectory.npz (actual qpos/q/dq, action, target, frame, state and inference time). Report includes model/bundle fingerprints, perturbation settings, completion, first unique failures, tracking RMSE, root error, requested torque saturation and model ctrl saturation, velocity margin and inference latency.

These metrics are NOT the README's normalized return or 40-clip success rate. A G2 pass covers one synchronous CPU simulation on one model/clip/configuration. It does not replace DDS physical validation or hardware checks; this tool never writes dds_physics_validation.json or changes real-robot arming gates. The model-comparison report found training total mass 33.341143 kg versus official model 35.112142 kg on this checkout.

See SIM2SIM_VALIDATION.md for the results actually run on 229.

## GPU physics backend (MuJoCo Warp)

Use --backend warp --cmd g1 to run the original BlinkMimic training robot in mjlab/MuJoCo Warp. Both policy inference and physics run on the selected CUDA device. The viewer copies only the selected replica's state into a CPU rendering mirror; that mirror never advances physics with mj_step.

The g1 mode is not the official torque-motor plant. Official-model GPU support for g2/standby/sweep is described in the following section. Unsupported combinations fail explicitly. GPU playback starts directly at the motion's first frame, without deployment preparation.

On the 229 graphical desktop:
```bash
cd /home/cqu023/blinkmimic
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=glfw OMP_NUM_THREADS=4 \
.venv/bin/python scripts/deploy/sim2sim.py \
  --backend warp --cmd g1 \
  --bundle deploy_artifacts/sim2sim_3500_walk1 \
  --ckpt logs/lafan1_csv_a100_s0/ckpt_003500.pt \
  --device cuda:0 --viewer --speed 1 \
  --out deploy_logs/warp_view_3500
```

Use a new output directory each run. SPACE pauses, N steps, R resets, +/- changes wall-clock playback speed, G toggles reference, F follows the selected robot. A selected-replica failure pauses the viewer. GUI rendering/keyboard interaction still requires a graphical desktop.

Headless batched playback:
```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 \
.venv/bin/python scripts/deploy/sim2sim.py \
  --backend warp --cmd g1 \
  --bundle deploy_artifacts/sim2sim_3500_walk1 \
  --ckpt logs/lafan1_csv_a100_s0/ckpt_003500.pt \
  --num-envs 64 --max-seconds 10 \
  --out deploy_logs/warp_batch_smoke
```

--num-envs runs replicas of this bundle's single motion, not different dataset clips. --view-env selects a zero-based replica. --precision checkpoint (default) follows the checkpoint AMP setting; --precision none uses fp32 with TF32 disabled for numerical comparison. --record saves only the selected replica's post-step trajectory, compatible with the CPU G1 --reference input. --screenshot is headless-only and writes viewer.png; set MUJOCO_GL=egl for it.

Reports explicitly record CUDA physics/policy devices, GPU name, precision, per-replica completion/failure and normalized returns. A --max-seconds cutoff yields a partial return and exit code 2, not a successful full-motion result. Failed/completed replicas stop accumulating rewards and their rendered terminal state is frozen. Timing includes synchronized GPU policy/physics/bookkeeping but excludes initialization and rendering; it is not a guarantee of real-time performance.

Implementation validation on 229: one-replica 10-step bf16 run and two-replica 20-step fp32 run, selected-replica recording and EGL screenshot all executed successfully with physics/policy on cuda:0. Both were deliberately incomplete and had zero observed failures. 19 regression tests passed. No full GPU motion run or interactive GUI keyboard test was performed. The GPU was occupied by another compute process, so no speedup claim is made. Existing CPU/DDS/hardware acceptance reports were not replaced.

## Official G1 sim2sim on GPU

The target-side path now supports --backend warp --cmd g2 (and standby/sweep). It loads the SAME official Unitree MJCF as CPU G2, with the same named joint/actuator mapping, ground placement, limits and perturbations. It does not instantiate mjlab or substitute the training robot.

MuJoCo Warp performs the 1 ms physical steps. A Warp CUDA kernel computes PD from GPU qpos/qvel at EVERY substep, applies training effort limits followed by the official motor ctrlrange, and writes GPU ctrl. Each 20 ms policy interval contains 20 GPU PD/physics steps. Delayed target/gain changes are scheduled at the same integer physics-step boundaries as CPU G2. Simulated support for standby is also recomputed on GPU each physics step. CUDA Graph captures this sequence when supported; an explicit reported fallback uses eager GPU stepping, never CPU physics.

To run physics AND policy on GPU, provide the matching checkpoint:
```bash
cd /home/cqu023/blinkmimic
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=glfw OMP_NUM_THREADS=4 \
.venv/bin/python scripts/deploy/sim2sim.py \
  --backend warp --cmd g2 \
  --bundle deploy_artifacts/sim2sim_3500_walk1 \
  --unitree-xml /home/cqu023/Documents/unitree_rl_lab/unitree_mujoco/unitree_robots/g1/scene_29dof.xml \
  --ckpt logs/lafan1_csv_a100_s0/ckpt_003500.pt \
  --device cuda:0 --viewer --speed 1 \
  --out deploy_logs/official_gpu_view_3500
```

The .pt policy runs in FP32 on CUDA to match the exported ONNX contract. Without --ckpt, official GPU physics still runs but the existing ONNX policy stays on CPU; the report distinguishes policy_device and physics_device. The renderer uses a CPU state mirror with OpenGL; it does not integrate physics. --cmd standby shows the full preparation/standby sequence. Pause/step/reset/reference controls are unchanged.

First compilation of official-model collision kernels can take minutes. The 1 kHz/50 Hz labels specify simulation-time rates, not a claim of real-time throughput. This first official-GPU implementation is single-world: --num-envs must be 1; batched original training playback remains under g1. --precision bf16 is not accepted for the official model; its GPU policy is FP32.

Short CPU/GPU verification:
```bash
OMP_NUM_THREADS=4 PYTHONPATH=src .venv/bin/python -u tools/validate_official_warp.py \
  --bundle deploy_artifacts/sim2sim_3500_walk1 \
  --unitree-xml /home/cqu023/Documents/unitree_rl_lab/unitree_mujoco/unitree_robots/g1/scene_29dof.xml \
  --ckpt logs/lafan1_csv_a100_s0/ckpt_003500.pt \
  --out deploy_logs/official_gpu_comparison
```

This checks the same compiled model fingerprint, 0/10 ms delay schedules, short open-loop physics trajectories, GPU policy vs ONNX, and PD kernel clipping/index mapping. Short-trajectory diagnostic tolerances are 0.02 rad joint error and 0.01 m root-position error; they do not certify long-horizon equivalence.

MuJoCo Warp warns that CYLINDER/CYLINDER and CYLINDER/MESH MULTICCD pairs in this model do not support the same multi-contact generation as CPU MuJoCo. Warnings are retained in backend_warnings. Float precision and contact algorithms differ; do not assume identical rollouts or reuse CPU acceptance as GPU acceptance. This new path still has no DDS or real hardware, and never changes the existing hardware arming reports.

## Viewer/readback performance correction

The official GPU plant now returns one pinned 608-byte snapshot per control cycle for the current G1 model: qpos, qvel, time and saturation/peak counters. Contact, constraint and other solver arrays stay on the GPU. Upload buffers are also persistent; per-cycle device-array allocations were removed. The synchronization waits for the relevant Warp stream rather than the entire device.

The viewer owns an independent MjModel/MjData render mirror. Physics/inference and GPU waits execute outside the viewer lock; the short locked section copies kinematic state and updates display geometry only. No CPU physics integration is used to produce the displayed poses.

Display refresh defaults to 30 Hz; --render-hz accepts 1..120 independently of the 50 Hz control period. Control scheduling uses accumulated deadlines: computation consumes the period rather than being followed by an extra 20 ms sleep. Overruns do not build an unbounded catch-up backlog. The overlay shows the measured control-tick time and its current wall-clock budget. R reuses the loaded model and policy rather than constructing a fresh simulator/model.

Same-state measurements on 229 (20 samples, warmed):
- Full get_data_into median: 4.9745 ms.
- Compact readback median: 0.07521 ms.
- Necessary-state and counter differences: exactly zero.
- Compact snapshot size: 608 bytes.
- Optimized physical control step median: 10.1703 ms (20 × 1 ms physics steps, includes upload/wait/readback; excludes policy inference).
- 23 regression tests passed.

The ~66x ratio applies ONLY to readback, not end-to-end FPS. Initial collision-kernel compilation is still required. Interactive GUI smoothness and remote-desktop transport were not benchmarked. Restart the old viewer process to load this code; existing processes do not hot-reload it.

## OpenGL software-rendering diagnosis and correction

The actual viewer on host 229 DISPLAY=:1 was using llvmpipe (LLVM 15), with Accelerated: no and 32 llvmpipe rendering threads. CUDA physics does not choose the OpenGL renderer.

Interactive sim2sim now defaults to --gl-renderer auto. If glxinfo detects software rendering, it tests NVIDIA PRIME offload and, when confirmed, applies __NV_PRIME_RENDER_OFFLOAD=1 and __GLX_VENDOR_LIBRARY_NAME=nvidia to this process only. --gl-renderer nvidia requires NVIDIA and fails rather than silently retaining software rendering; --gl-renderer system preserves the desktop selection. No system driver/display configuration is changed.

Verified in an actual hidden GLFW context, after importing MuJoCo as the CLI does:
OpenGL vendor: NVIDIA Corporation
OpenGL renderer: NVIDIA GeForce RTX 4090/PCIe/SSE2

27 regression tests passed. This confirms hardware rendering selection, not a measured full-window FPS. Close the old software-rendered window and restart with a new --out directory; the old GL context cannot be changed in place.
