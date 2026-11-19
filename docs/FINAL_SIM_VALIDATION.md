# Final sim2sim validation — host 229

The final path uses the CPU official G1, modeled primary IMU, isolated DDS and the same run_g1.py used for hardware. No real robot was started. Host 32 was not modified.

## Completed full session
- Bundle: deploy_artifacts/final_3500_walk1; checkpoint 3500; original walk1_subject1 motion.
- Shared preparation gain scale: 4.5. No simulation-only actuator gains; start-pose tolerance was not relaxed.
- Session: /home/cqu023/blinkmimic/deploy_logs/final_complete/20261119_172937_3052561
- All states observed: PASSIVE, INTERP, HOLD, STANDBY_PRE, POLICY, STANDBY_POST, DAMP.
- Full reference reached frame 13065. POLICY had no support.
- Full physical run: 290.345 simulated seconds, including preparation and damping.
- Physics/wall lag P99: 0.637 ms; final active-loop lag -0.395 ms.
- DDS: 290197 LowState publications, 158520 received LowCmd messages, zero bad CRC and write failures.
- Modeled IMU maximum gravity-direction error against truth: 0.1055 deg.
- Offline truth evaluation: max aligned root-height error 0.04623 m, max R22 tilt error 0.02021, joint RMSE 0.08678 rad; zero violation samples.
- Normal stop restored support at the current pose before DAMP: height change 0.00384 m, R22 change 0.00497; stop-stage result True. Emergency D remains immediate.
- This session supersedes the earlier final_full run, which did not restore support before normal DAMP.
- Combined flow, timing, offline tracking and stop-stage result: True. Offline evaluation was performed after the recorded session; it never influenced controller decisions.

## Related verification
- 35 unit/regression tests passed: sensor/LowState input equivalence, 6D/history/action agreement, pulse width/release/stop priority, focus release, window/terminal event builders, nonfatal rejection reasons, removed extra angle checks, existing simulation/graphics/runtime checks.
- Actual standalone sim_keyboard.py --key client: 1, 2, B, 3, D traversed the expected states via local IPC and DDS. Result: True; logs: /home/cqu023/blinkmimic/deploy_logs/final_keyboard/20261119_171928_3040880.
- A hidden GLFW window connected to the active final session and rendered three frames using NVIDIA OpenGL. Interactive mouse/OS keyboard behavior was not manually exercised.
- Runner blocked for 70 ms: physics advanced 0.070 s and received actual damping LowCmd after the runner detected expired state.
- Sensor stream interrupted: physics continued and the shared runner detected stale state, with actual damping received.
- Runner killed: the original C++ bridge detected parent loss and sent damping, observed by the simulated driver. The dead runner's status file naturally did not advance to DAMP; this test relies on the received motor command, not a fabricated state label.
- Fault report: deploy_logs/final_faults/report.json.

## Boundaries
These are results for one checkpoint and one motion on this host. IMU parameters come from the SuperMimic reference model, not a calibration of the user's G1. Temperature is synthetic/fault-injection input, not a measured thermal model. Startup report prerequisites and verified_on_this_robot were removed; validation output is evidence, not a permission file.

Current operation commands are in DEPLOY_G1.md. Older GPU/CPU diagnostic tools and historical reports remain available but are not the final real-time path.
