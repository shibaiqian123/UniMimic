# BlinkMimic single-motion G1 deployment

This runtime uses the existing 32D policy. It does not accept SONIC tokens, and does not add VLA/SMPL/VR or localization. Source development takes place on host 229. Do not run real-robot commands unattended.

## Runtime and provenance

Runtime modules under src/blinkmimic/deploy/core.py and controller.py need only NumPy and ONNX Runtime. The C++ bridge uses unitree_sdk2; Python unitree_sdk2py is needed only for DDS simulation/tests, not for the runner. Bridge sources and wire client were copied from SuperMimic-Repro-Sim2Real commit 0905de4; its MIT license is retained in scripts/deploy/bridge/LICENSE. The state-machine behavior is adapted for BlinkMimic, not the original SuperMimic model.

Create a deployment Python environment and install requirements-deploy.txt. On the build host:
```bash
bash scripts/deploy/bridge/build.sh /absolute/path/to/unitree_sdk2
```
Binaries are local build outputs, not portable across CPU architectures. BLINK_BRIDGE_HOME can point to the directory containing both bridge binaries.

## Export and CPU parity

Run in the training environment from the repository root:
```bash
PYTHONPATH=src .venv/bin/python tools/export_deploy.py \
  --ckpt logs/lafan1_csv_a100_s0/ckpt_001600.pt \
  --motion data/lafan1_g1/walk1_subject1.csv \
  --out deploy_artifacts/lafan1_s0_1600_walk1

PYTHONPATH=src .venv/bin/python tools/validate_deploy.py \
  --bundle deploy_artifacts/lafan1_s0_1600_walk1
```

Output directories must be new. The example is a validation candidate, not a hardware-approved model. Select checkpoints using actual evaluation results, then validate the chosen motion and its endpoints.

policy.onnx inputs are obs_history [B,10,99] and goal [B,730]; output actions [B,29]. All training normalization and the reference-action skip remain inside the model. Actions are NOT motor angles. motion.npz contains the original resampled reference, dynamic future goals and separate zero-velocity static endpoint goals. deploy_config.json supplies parameters and file hashes. Never mix files from different exports.

Observations are q-q0 (29), dq (29), pelvis gyro (3), projected gravity (3), last raw action (29), relative orientation (6). History is oldest-first, frame-major. The 6D packing is [R00,R01,R10,R11,R20,R21]. Goal remains in the reference's local heading frame; a fixed initial yaw offset rotates reference orientation into the IMU world frame.

## Independent simulation and isolated DDS

```bash
PYTHONPATH=src .venv/bin/python tools/sim_deploy.py \
  --bundle deploy_artifacts/lafan1_s0_1600_walk1 \
  --ckpt logs/lafan1_csv_a100_s0/ckpt_001600.pt

PYTHONPATH=src .venv/bin/python tools/prepare_deploy_sim.py \
  --bundle deploy_artifacts/lafan1_s0_1600_walk1 \
  --out deploy_artifacts/lafan1_s0_1600_walk1/sim_scene.xml
```

The independent plant is CPU MuJoCo with 2 ms implicit PD, not the original GPU benchmark. It runs ONNX and PyTorch from identical initial states, reports normalized motion reward using the training reward formula, failure, clamping, latency and endpoint stability. A candidate passes only after the full motion and the final three seconds of both endpoint windows are stable. The physics scene contains absolute local mesh paths and is for testing; regenerate it on another machine.

Use a Python environment with NumPy, ONNX Runtime, MuJoCo and unitree_sdk2py for DDS tests:
```bash
PYTHONPATH=src python scripts/deploy/bridge/test_bridge.py
PYTHONPATH=src python scripts/deploy/test_dds_e2e.py \
  --bundle deploy_artifacts/lafan1_s0_1600_walk1 --out deploy_logs/dds_fixture
PYTHONPATH=src python scripts/deploy/test_dds_e2e.py \
  --bundle deploy_artifacts/lafan1_s0_1600_walk1 --out deploy_logs/dds_physics --physics --domain 1
```

The first end-to-end suite is a **sensor replay fixture**: it validates actual runner/bridge/DDS communication, state transitions and injected failures. It is not evidence of physical motion tracking. The --physics run releases the simulated support for real MuJoCo dynamics. Reports clearly distinguish these scopes. All tests use domain 1 or 2 on lo; they cannot select the hardware domain.

On 229, the discovered DDS test interpreter is /home/cqu023/miniconda3/envs/unitree_sim_env/bin/python. Extra simulator dependencies installed for this implementation are isolated in .deploy_deps, so use PYTHONPATH=src:.deploy_deps with that interpreter. Training/export tools use .venv/bin/python.

## Hardware profile and operation

Copy scripts/deploy/hardware_profile.example.json to a local profile. The example identity mount is NOT a verified calibration. Confirm quaternion wxyz convention, gyro rad/s, fixed IMU-to-pelvis rotation, and the exact motor order for this robot. R_pelvis_from_imu_wxyz maps IMU-frame vectors into pelvis coordinates. A moving torso IMU is unsupported. Set verified_on_this_robot only after those observations have been verified.

Read-only state capture, using a NEW log directory:
```bash
PYTHONPATH=src python scripts/deploy/run_g1.py \
  --bundle /absolute/path/to/bundle \
  --hardware-profile /absolute/path/to/verified_profile.json \
  --iface ROBOT_NIC --domain 0 --stage air \
  --read-only --seconds 10 --log-dir deploy_logs/readonly_session
```
Read-only mode starts the bridge with --listen-only and never initializes a lowcmd publisher.

Real control uses the same arguments **without --read-only**. It additionally requires the typed ARM prompt and matching successful export_validation.json, observation_validation.json, sim_validation.json, dds_validation.json and dds_physics_validation.json. Failed or absent reports block arming; no unsafe bypass flag is provided. These reports do not replace operator verification on the actual robot.

Use support/gantry and keep a person at the remote stop. First run with stage air; subsequent stages feet, slack and ground require separate physical verification. Stage thresholds in the exported configuration are explicit initial engineering limits, NOT hardware-validated settings.

Buttons:
- L2+up: interpolate from measured pose to the first reference pose, then HOLD.
- R1+A: enter policy-controlled static STANDBY_PRE.
- R1+X: play once, only after a continuous three-second stable standby window.
- At the last frame: enter static STANDBY_POST without clearing history/action.
- L2+B: latched DAMP; restart the process to prepare another session.

The policy runs at 50 Hz; the bridge streams the latest command at 500 Hz. Initial heading alignment occurs once. Targets use the exported q0/scale/gains; the previous-action observation keeps the raw policy action even if the command is limited. Preparation-only gains are exported explicitly as training kp × preparation_gain_scale and kd × sqrt(preparation_gain_scale), currently 4.0; policy gains remain unchanged. The same effort limits apply during preparation. This candidate setting is simulation-tested only, not hardware-qualified. No SuperMimic clip layout, fixed stand gains, thermal stop clip or model is reused.

Faults include stale reception/frozen state ticks, invalid measurements, incompatible robot mode, joint overspeed/range/fault, overtemperature, policy timeout/nonfinite output, publication error, bridge watchdog and foreign command publishers. DAMP has kp=0, kd=3 and is held for ten seconds before process exit; the bridge also handles parent loss. Damping is not a balance controller: maintain physical support during faults.

## Evidence and logs

runner.jsonl stores package fingerprint, effective configuration/profile, state, frame, measured state, raw action, target, clamping, latency and failure reason. bridge.log records DDS/watchdog events. Deploy artifacts and logs are ignored by Git. No current result constitutes hardware verification; see VALIDATION.md for implementation-time results.
