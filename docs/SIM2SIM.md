# BlinkMimic sim2sim

This is a separate native-MuJoCo deployment test harness, inspired by the gate structure and explicit PD/delay design of SuperMimic-Repro-Sim2Real/scripts/deploy/sim2sim.py (inspected on host 229). It uses BlinkMimic's own Bundle, DeployCore, 99D observations, frame-major history, 730D goals and trained parameters. No SuperMimic model/680D goal/930D observation is loaded.

All runs are CPU MuJoCo + NumPy + ONNX Runtime. The window uses OpenGL. There is no DDS publisher, no robot connection, and no CUDA physics switch. The optional training-reference recorder requires the original PyTorch/mjlab/Warp training environment.

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
