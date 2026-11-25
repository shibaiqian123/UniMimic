# SONIC reference three-point cache

Implemented on 229. The 32 server only receives an in-memory execution copy and writes data outputs; no training or robot control is launched.

## Definition

Reference commit: 1e1ec4f5f27e6c3e03048f693bd7c60cce04afda in NVlabs/GR00T-WholeBodyControl.
See gear_sonic/config/manager_env/commands/terms/motion.yaml and the commands.py/observations.py functions vr_3point_body_pos_w, vr_3point_local_target, vr_3point_local_orn_target.
This is an independent implementation of their geometry using BlinkMimic FK.

Order: left hand, right hand, virtual head. Body names: left_wrist_yaw_link, right_wrist_yaw_link, torso_link.
Local offsets in meters: [0.18,-0.025,0], [0.18,0.025,0], [0,0,0.35].
The head target inherits torso orientation. This is NOT a measured head sensor.
Anchor: reference pelvis, using its complete rotation, including roll/pitch.
Use each mirrored CSV directly; no second mirroring operation.

The existing MotionLib interpolation was extracted without numerical changes into resample_motion and is shared by training and generation. Output stops at the paired G1/SMPL common endpoint.

## Files and arrays

Each clips/<split>/<hash-prefix>/<motion-ID-hash>.npz is a complete motion:
- time_s [T]
- position_world, position_local [T,3,3], meters
- quaternion_world, quaternion_local [T,3,4], wxyz
- pelvis_position_world [T,3], meters
- pelvis_quaternion_world [T,4], wxyz

All arrays are float32; times are k*0.02 seconds. Quaternions are normalized with adjacent sign continuity.
metadata is a JSON scalar string, loadable with allow_pickle=False. It records source identity, split and mirror, CSV and XML fingerprints, SMPL fingerprint, shared endpoint, official commit and geometry convention.
train.jsonl/val.jsonl/test.jsonl map motion IDs to caches; fixed100 indexes retain the 96/97 paired evaluation intersections. failures.jsonl lists failures. report.json contains actual counts, payload size and runtime. progress.jsonl is available during generation.

Temporary files are atomically replaced. --resume reuses only complete finite arrays with matching metadata, source and XML fingerprints. Corrupt or mismatched caches are regenerated.

## Generate on 32 after the user synchronizes code with Git

```bash
cd /share/2025slx/UniMimic
ROOT=/share/2025slx/UniMimic/data/bones_seed
PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
.venv/bin/python tools/generate_threepoint.py \
  --manifest "$ROOT/paired/bones_smpl_v2/train.jsonl" \
             "$ROOT/paired/bones_smpl_v2/val.jsonl" \
             "$ROOT/paired/bones_smpl_v2/test.jsonl" \
  --data-root "$ROOT" --out "$ROOT/derived/threepoint_sonic_v1" \
  --device cpu --workers 8
```

Add --resume to recover/reuse an existing output. --limit 20 is useful for smoke checks, but representative review used a separately selected 20-motion manifest. CUDA is supported with --device cuda:0 --workers 1.

## Review on 229

20 clips: two originals and their mirrors in each of walking, turning, jogging, squatting and gestures.
Videos and contact sheets: results/threepoint_sample_review. Every frame was checked against independent MuJoCo FK and NumPy local transforms. Five sheets provide 80 manually reviewed snapshots, not continuous human inspection of every rendered video frame.
Sample CPU/CUDA cache directories: results/threepoint_sample_cpu and results/threepoint_sample_cuda.
CPU initial conversion was 4.12 s versus CUDA 5.74 s for 20 motions (includes initialization; not a steady-state GPU throughput claim).

```bash
cd /home/cqu023/blinkmimic
PYTHONPATH=src MUJOCO_GL=egl OMP_NUM_THREADS=1 .venv/bin/python tools/validate_threepoint.py \
  --manifest data/threepoint_samples/manifest.jsonl --data-root data/threepoint_samples \
  --cache results/threepoint_sample_cpu --out results/threepoint_review_new
```

## Boundary

This cache is reference geometry, not actual VR recordings and not a complete SONIC teleop observation.
Robot-relative heading must be computed online. Lower-body future commands, VR calibration, SMPL feature conversion and joint multimodal training are separate next steps. Passing reference FK does not establish dynamic tracking or hardware readiness.

## Completed full generation

32 output: /share/2025slx/UniMimic/data/bones_seed/derived/threepoint_sonic_v1/
131,418/131,418 completed, zero failures: train 106,677; val 12,225; test 12,516. Fixed evaluation indexes: 96/96 and 97/97.
8 CPU workers; 723.85 seconds from worker initialization through generation/index writing (manifest reading before timer excluded).
NPZ payload 9,073,554,845 bytes; full directory apparent size 9,203,070,053 bytes; du -sh reports 8.9G allocated.
20 sample FK max absolute position/matrix error: 1.1298428201911292e-6. CPU/CUDA maximum array difference: 9.5367431640625e-7.
Full report copied to results/threepoint_sample_review/full_generation_report.json on 229.
