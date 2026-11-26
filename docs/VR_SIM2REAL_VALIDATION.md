# VR Sim2Real implementation validation — 2026-11-26

Reference: local GR00T commit 2f07b35b3f3b9053b1f4fd2adeaf56b6bce32366.
Target: 229 BlinkMimic teleop, bundle omni_15000_walk_A049 (trained omni 15k).
No 32 source edits, real robot startup, MotionSwitcher RPC or hardware motion executed.

## Code and interface

Bridge, simulation adapter and bounded C++ MotionSwitcher helper compiled successfully using existing GR00T Unitree SDK2.
40 targeted tests cover gain overrides, no takeover in read-only/simulation, mounted IMU convention,
pose heading increments, mode-edge reference continuity, repeated sensor timestamps,
all motor-send paths including exception damping, existing controller/runtime and input contracts.

## Isolated DDS regression

Synthetic single-frame 50Hz inputs, official CPU physics, loopback domain 2,
source ports 15556/15557/15558, stage slack, stationary lower-body, support ON.
Both runs visited PLANNER -> POSE -> PLANNER -> VR_3PT and ended in latched DAMP.

- Normal stop: results/teleop_real_path_smoke_v1/sessions/20261126_165608_3229335.
  Reason SONIC remote stop; 849 policy samples.
  Max controller 8.216ms, max inference 2.994ms; physics timing valid.
- Input interruption: results/teleop_real_path_disconnect_v1/sessions/20261126_165715_3230537.
  Reason VR input stream stale; 899 policy samples.
  Max controller 8.184ms, max inference 3.024ms; physics timing valid.
  Child exit code 1 is expected for injected stream failure, not successful motion completion.

Machine-readable summary: results/teleop_sim2real_validation.json.
Following the runs, targeted tests also cover sensor-time duplicate rejection and the exception-path mode fix.
The synthetic replay does not contain source sensor timestamps and does not exercise a real MotionSwitcher service.

## Not yet verified

New changes with a physical PICO, actual robot LowState/IMU mounting, motor takeover,
supported hardware motion, and unsupported standing have NOT been verified in this implementation run.
Earlier user-reported VR simulation success predates these changes and is not reused as hardware evidence.
The IMU profile records a reference identity mounting assumption, not a measured calibration.
No automatic arming certificate or report gate was added.
