> 历史记录：此文对应旧实现。当前流程与启动条件以 DEPLOY_G1.md 为准；旧报告不再阻止启动。

# Deployment validation on host 229

Implementation is present; this candidate is **NOT accepted for real-robot arming**. No real robot was connected or commanded. Host 32 was not modified.

Candidate: deploy_artifacts/lafan1_s0_1600_walk1_v2
Checkpoint: ckpt_001600.pt (chosen from the user's supplied 40-clip/four-replica evaluation showing zero failures).
Motion: original walk1_subject1.csv, full 13065 control transitions.
Bundle fingerprint: b1f0a8ae2c6ab78547d4e21fa9852f4bb158cc5c9abcd3e86193eb4ab11420dc

## Passed
- 15 hardware-free tests: history, rotation and IMU mount conventions, frame progression, raw previous action, state transitions, preheld buttons, effective-gain torque limiting, fault latching and timeout.
- C++ bridge compiled using the installed Unitree SDK.
- Bridge protocol suite on domain 2 / lo: CRC, snapshots, command parity, fixed-rate stream, measured-position damping, watchdog/latching and QUIT.
- ONNX vs PyTorch maximum action error: 2.38418579e-06, tolerance 1e-3.
- Deployment observation/history error: 6.10802972e-08; goal error 0.0, tolerance 1e-4.
- Independent CPU MuJoCo: ONNX return 0.884953203, PyTorch return 0.884953188; both completed without the training termination predicate. Both static endpoints passed the last-three-second stability window.
- Independent rollout return difference: 1.47728504e-08. ONNX p99 inference 3.931 ms.
- DDS sensor-replay suite: normal, stale, hot, nan, timeout, runner_exit, readonly; all passed. It checks actual runner/bridge traffic and damping received on DDS, but is not a physical motion test.
- Syntax and git diff whitespace checks passed.

## Failed / not verified
- Full DDS CPU physics test failed: right_ankle_roll_joint crossed its hard lower limit tolerance after 944 policy steps (about 18.9 s); observed -0.31563556 rad, model lower limit -0.26179999 rad, guard tolerance 0.05 rad. DAMP latched. This is not accepted by the real-robot startup gate.
- Earlier preparation runs with the unscaled training PD gains could not reach the required start-pose tolerance. The final candidate exports an explicit preparation_gain_scale=4; only INTERP/HOLD use kp*4 and kd*2, with effort limiting computed using those effective gains. Policy uses original gains. The physics test then reached POLICY, but subsequently failed the ankle guard as above. No limit thresholds were relaxed.
- Independent CPU rollout lacks the real-time DDS transport and preparation path; its success does not override the failed DDS physical test. It also clamps targets and uses a different physics plant from the README benchmark.
- No hardware profile, IMU mount, motor mapping, thermal threshold, gain transition or physical behavior has been verified on the user's actual G1.

## Reports and logs
- Candidate directory contains export_validation.json, observation_validation.json, sim_validation.json, dds_validation.json, and failed dds_physics_validation.json.
- DDS fixture logs: deploy_logs/dds_fixture_final
- DDS physics failure: deploy_logs/dds_physics_v3
- Runtime source: src/blinkmimic/deploy and scripts/deploy
- See DEPLOY_G1.md for all commands. Reports must match the bundle fingerprint; failed physics prevents arming. No bypass was added.

Next acceptance work is to resolve the ankle excursion under the real-time DDS/PD path, then repeat the full physical sequence. Do not mark this candidate hardware-ready merely because offline parity and replay pass.
