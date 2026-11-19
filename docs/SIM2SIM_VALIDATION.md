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
