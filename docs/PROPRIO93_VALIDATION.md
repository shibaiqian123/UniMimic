# Proprio93 future-window validation — 229

Date: 2026-11-21. All code changes on /home/cqu023/blinkmimic, host229. No host32 modifications, long training or hardware commands.

## Implemented contract
- New proprio93 uses 10x93 decoder history and 2x32 tokens (994 input scalars).
- Motion encoder receives original 730D goal plus 10x6 current-robot-relative FUTURE rotations (790 scalars).
- f+1..f+10, per-clip endpoint clamping, full roll/pitch/yaw; fixed deployment yaw alignment.
- New observation contract version2 / marker2 explicitly rejects the unfinished current-only6D architecture.
- legacy99 remains default, v1/v2 deployment bundles unchanged; new bundles use v3.
- Single GPU, distributed local phases, probe, evaluation, GPU diagnostics and baked-motion compatibility exporter wired.
- Independent decoder inputs are obs_history and tokens only.

## Actual results
- 60 targeted tests passed (logs/proprio93_pytest.log), including real old32 and old99-FSQ checkpoints with exact output parity.
- Torch vs NumPy rotation tests use atol1e-6, stricter than requested1e-4; nonzero yaw/roll/pitch, quaternion sign, clip boundaries, dynamic static endpoints covered.
- Gradient through current robot rotation into encoder, FSQ STE, BF16, compiled backward, token-only decoder and dynamic ONNX batch covered.
- Two CPU/Gloo ranks validate joint93/790 normalization merging; not a real dual-A100 claim.
- Real single-GPU training: 16env, small networks (64,32,16 encoder/decoder;64,64,64 world model), BF16, Muon, EMA, 6iterations then resume to8; collect and update CUDA graphs captured.
  logs/proprio93_single_smoke/
  logs/proprio93_single_resume_console.log
- New distributed local phases on ONE RTX4090: 8iterations, all6 phases captured; async probe completed at iteration4.
  logs/proprio93_distributed_local_smoke/
  logs/proprio93_distributed_local_console.log
- v3 export maximum full/split action error: 1.5133991837501526e-09.
  deploy_artifacts/proprio93_interface_smoke/export_validation.json
- Legacy baked-motion compatibility exporter: 64random-frame ONNX/PyTorch action error approximately1.40e-09.
  logs/proprio93_baked_export_console.log
- Actual v3 DeployCore pre/motion/post calls produce finite actions, history10x93, future window10x6; loading/running in a fresh process does not import torch. Existing v1/v2 walk bundles still load.
- Mixed legacy99 + proprio93 eval entry executed on a tiny fixture:
  results/proprio93_mixed_eval_smoke.json
  Both minimal checkpoints failed that clip; this proves entrypoint compatibility, NOT trained tracking capability.

## Not validated
- Real dual-A100/NCCL execution for the new input mode.
- Full3500-iteration training,40-clip performance,complete walk1 final acceptance,real hardware.
- SMPL/VR/VLA encoders or task-level control.

Interface smoke artifacts are not hardware candidates. Do not delete6 channels from an old checkpoint or fill future rotations with zeros. Commands and ABI are in PROPRIO93.md.
