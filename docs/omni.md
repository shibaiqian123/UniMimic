> 本页包含开发阶段记录；当前能力与限制以 [PROJECT_STATUS](PROJECT_STATUS.md) 为准，可移植命令见 [GETTING_STARTED](GETTING_STARTED.md)。历史229/32路径仅用于追溯。

# BlinkMimic omni: Motion / SMPL / Teleop

Source work is on 229, branch `omni`. Sync commits with Git before running the commands on 32. No pretrained Motion policy is loaded. Resume is only for a matching omni experiment, with the same world size; replay is refilled, so resume is not bit-exact.

## Contract and upstream reference

Reference: NVIDIA GR00T-WholeBodyControl, commit `2f07b35b3f3b9053b1f4fd2adeaf56b6bce32366`, `sonic_release`; Apache-2.0 source attribution is in THIRD_PARTY_NOTICES.md. Relevant files are motion command configuration, `commands.py`, `observations.py`, the multimodal policy and standard auxiliary losses. No NVIDIA weights or compliance losses are included.

* Motion: 640 features, ten frames at offsets **0,5,...45**, 50 Hz. Official packing concatenates flattened q and qd, reshapes to ten 58D blocks, appends the corresponding relative 6D rotation, then flattens.
* SMPL: 840 features, offsets **0,...9**. Each frame contains 24 local joints (72), relative root orientation (6), and wrist reference joints (6). Root conversion is `Rx(pi/2) @ Exp(root_axis_angle) @ R(wxyz=[.5,.5,.5,.5]).T`. As in the release, stored `smpl_joints` are rotated into this local frame without an additional translation subtraction. This convention was checked on the authorized 20 actual paired samples (see validation evidence below).
* Teleop: 273 features: ten frames of twelve leg positions and velocities (240), current left/right hand and virtual head positions (9), local 6D orientations (18), and current reference pelvis orientation relative to the robot (6). It is not a three-point-only VR controller: lower-body reference remains required.
* Joint arrays follow BlinkMimic RobotSpec names/order, not IsaacLab's storage order. Wrist order is left/right roll, left/right pitch, left/right yaw. Threepoint caches preserve their original source commit and robot XML fingerprint.
* Future windows include the current frame, clamp at the paired shared endpoint, and never enter another clip. This differs from the old single-modality `f+1...f+10` interface, which is preserved.
* Encoders are independent SiLU MLPs `[2048,1024,512,512]`, with independent RunningNorm. Each produces 64 scalars, shared 32-level FSQ rules, token shape `[2,32]`. The shared action decoder takes only ten 93D proprioceptive frames plus token, with BlinkMimic decoder widths. No `pi_ref_skip`.
* Auxiliary reconstruction uses quantized tokens to reconstruct physical Motion features. Standard raw pre-FSQ latent MSE terms are symmetric, including SMPL reconstruction -> Motion re-encoding against the Motion latent. Weights are reconstruction `.01`, each of the other four terms `1.0`. No action matching loss is added.

This adopts SONIC feature/loss design inside BlinkMimic's world-model imagination algorithm; it is not a PPO reproduction. All three branches can be evaluated for fixed-shape graph computation, but exactly one modality token produces each environment's action. Modality is sampled uniformly at reset, saved in replay, and retained throughout imagination. Auxiliary losses use paired features at the same starting state.

## Cache preprocessing on 32

Install the optional dependency in the existing environment (the lock file adds only joblib):

```bash
cd /share/2025slx/UniMimic
uv pip install --python .venv/bin/python 'joblib>=1.4'
DATA=/share/2025slx/UniMimic/data/bones_seed
PAIR="$DATA/paired/bones_smpl_v2"
TP="$DATA/derived/threepoint_sonic_v1"
CACHE="$DATA/derived/omni_sonic_v1"

PYTHONPATH=src .venv/bin/python tools/prepare_omni.py \
  --manifest "$PAIR/train.jsonl" --data-root "$DATA" \
  --threepoint-root "$TP" --out "$CACHE/train" --resume
```

Generate full held-out and fixed evaluation caches with the same tool:

```bash
for SPLIT in val test val_fixed100 test_fixed100; do
  PYTHONPATH=src .venv/bin/python tools/prepare_omni.py \
    --manifest "$PAIR/${SPLIT}.jsonl" --data-root "$DATA" \
    --threepoint-root "$TP" --out "$CACHE/$SPLIT" --resume
done
```

 Preserve the fixed 96/97 evaluation manifests; do not silently replace their missing pairs. `--limit 20 --out "$CACHE/sample20"` creates a separate small validation cache; do not reuse that directory for the full manifest. `--resume` checks configuration, source and threepoint fingerprints and shard checksums, then reuses complete shards. Failures are recorded and stop preprocessing.

Before the full conversion/training, validate actual SMPL coordinate conventions and play the sample clips. Expected complete manifest counts are 106,677 train / 12,225 validation / 12,516 test; tools report actual counts rather than assuming these numbers.

Shards contain 184 FP32 reference scalars per frame, no duplicated future windows. Readers use CPU mmap and gather windows outside CUDA capture, transferring only active batches to GPU. Current robot orientation is computed online; imagination uses differentiable predicted orientation. Old MotionLib world-model/critic queries remain unchanged except truncation to the shared clip duration. Full-data prefetch throughput has not yet been benchmarked.

## Fresh training and resume on 32

```bash
cd /share/2025slx/UniMimic
DATA=/share/2025slx/UniMimic/data/bones_seed
CACHE="$DATA/derived/omni_sonic_v1"
RUN="logs/bones_omni_fsq64_2a100_s0_$(date +%Y%m%d_%H%M%S)"
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=4 \
.venv/bin/python scripts/train_omni.py \
  --gpus 2 --index "$CACHE/train/index.json" --data-root "$DATA" \
  --log-dir "$RUN" --num_envs 16384 --pi_batch 4096 \
  --iterations 100000 --seed 0 --tensorboard --ckpt_every 1000
```

This is a launch example, not an instruction to automatically run 100,000 iterations. Verify a short real-data run and memory use first. `num_envs`, `pi_batch`, and `wm_batch` remain **global**, evenly divided across ranks, without learning-rate scaling. One GPU uses `--gpus 1` and one visible device. `--dry-run` prints the actual recipe. All encoders and action/reconstruction decoders initialize randomly. Per-modality eval replaces the old online single-Motion eprobe in this entry point (`eprobe_share=0`, `eprobe_async=0`, explicitly printed).

Resume with the same options, directory and index, adding `--resume "$RUN/last.pt"`, and a larger final iteration count. Public light checkpoints are for evaluation/export; use `last.pt` for full optimizer/RNG resume. Old single-modality weights cannot initialize or resume this run. `omni_metrics.jsonl` and TensorBoard record per-modality counts, episode return/failure, FSQ occupancy, saturation, gradients, and auxiliary losses. `distributed_phases.json` reports graph captures. `--dist-no-graphs` is an explicit diagnostic mode.

## Evaluate separately and export

Build a cache from the existing fixed validation manifest into `val_fixed100`, then:

```bash
for MODALITY in motion smpl teleop; do
  PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 \
  .venv/bin/python tools/eval_metrics.py \
    "$RUN/ckpt_001000.pt" "$RUN/ckpt_002000.pt" \
    --omni-index "$CACHE/val_fixed100/index.json" --omni-data-root "$DATA" \
    --modality "$MODALITY" --out "results/omni_val_${MODALITY}"
done
```

Choose the checkpoint on validation only, then evaluate each modality once on the fixed test cache. Compare returns, failures, tracking errors and action changes on the same paired IDs. A failed random-weight smoke run is not a motion-quality result.

For split interfaces alone:

```bash
PYTHONPATH=src .venv/bin/python tools/export_omni.py \
  --ckpt "$RUN/ckpt_002000.pt" --out deploy_artifacts/omni_interfaces
```

For one paired clip (choose `MOTION_ID` verbatim from the index):

```bash
PYTHONPATH=src .venv/bin/python tools/export_omni_deploy.py \
  --ckpt "$RUN/ckpt_002000.pt" --index "$CACHE/val_fixed100/index.json" \
  --data-root "$DATA" --motion-id "$MOTION_ID" \
  --out deploy_artifacts/omni_single_motion
```

The v4 bundle has three normalized encoder ONNX graphs, one normalized shared decoder ONNX, reference arrays and fingerprints. Decoder tokens are already quantized and are not quantized again. Encoder input is a flattened feature vector of its documented dimension; output `[B,2,32]`. Decoder inputs `[B,10,93]`, `[B,2,32]`; output `[B,29]`. Auxiliary reconstruction is not exported. Export tests batch sizes 1/3/7 and action agreement against PyTorch at `1e-3` tolerance.

## Final simulation on 229

Copy the generated bundle using the normal authorized Git/artifact workflow. No source changes on 32 are needed beyond Git sync.

```bash
cd /home/cqu023/blinkmimic
OMP_NUM_THREADS=1 .venv/bin/python scripts/deploy/sim2sim.py \
  --cmd final --modality teleop \
  --bundle deploy_artifacts/omni_single_motion \
  --unitree-xml /home/cqu023/Documents/unitree_rl_lab/unitree_mujoco/unitree_robots/g1/scene_29dof.xml \
  --viewer --gl-renderer nvidia --out deploy_logs/omni_teleop
```

Repeat with `motion` and `smpl`. Keyboard 1 prepare / 2 standby / 3 play / B support / D stop remains unchanged. LowState supplies current orientation through the common runner. Fixed initial heading rotates both G1 and SMPL references; static endpoints repeat reference pose, zero reference velocities, and recalculate relative orientation each control step. Legacy diagnostic viewers remain unchanged and explicitly reject v4 bundles rather than silently interpreting them as Motion-only. This is cached paired-reference playback, not a live SMPL/VR interface.

## Evidence and limitations (2026-11-25)

* 68 targeted tests passed (67-test regression run plus the additional short-clip sampling case): features/gradients, NumPy deployment agreement, shard conversion/resume fingerprint rejection, split dynamic-batch ONNX, legacy latent/proprio/deploy regression, distributed helper tests.
* Single RTX4090 short training and resume passed in FP32 and BF16, using two existing G1 clips with explicitly **synthetic** SMPL/threepoint features. Encoder widths are full size; world model and action decoder were reduced for this smoke test.
* All six staged graph phases captured successfully after BF16 resume: observation, physics, reset, world/policy/critic backward.
* A smoke checkpoint exported with maximum action error about `4.3e-8`. Each modality passed offline DeployCore inference; Teleop evaluation entry ran on two synthetic paired clips. These are interface checks only.
* Existing v1/v2/v3 deployed bundles were compared against the pre-change runtime on fixed inputs; action differences were zero.
* Two-rank CPU/Gloo test covered global normalization, unequal masked-loss gradients, and coordinated nonfinite skip. It does not establish two-A100 CUDA/NCCL throughput or end-to-end training stability.
* The authorized 20 actual SMPL samples were copied from 32 to 229 with matching manifest SHA256. All are training-split samples: 10 original and 10 mirrored clips across walking, turning, jogging, squatting and gestures. Their 5,756 shared frames were converted and independently checked using SciPy/NumPy: maximum feature error `3.81e-7` (threshold `1e-4`). Each raw pelvis position is constant over its clip; local left/right hip separation has the expected sign. G1-versus-SMPL root orientation differences are retargeting diagnostics, not formula error. Full preprocessing, full-data short training, long-run quality and complete trained-policy final simulation remain pending.
* No real VR calibration, online lower-body planner, VLA, robot run or hardware validation was performed. No startup certification gates were added.

## Repeat actual-sample feature verification on 229

```bash
cd /home/cqu023/blinkmimic
OMP_NUM_THREADS=1 PYTHONPATH=src .venv/bin/python tools/validate_omni_samples.py \
  --manifest data/threepoint_samples/manifest.jsonl \
  --data-root data/threepoint_samples --index results/omni_real20/index.json \
  --out results/omni_real20/validation.json
```

The validation reads source SMPL and cached references, compares every shared frame and both future windows against independent formulas, and records per-clip diagnostics. It does not fit or reconstruct a new SMPL body model and does not claim the human and retargeted robot skeletons coincide.

## Actual-data short training and export follow-up

Authorized sample access is complete; the earlier transfer blocker is resolved. Exactly 20 SMPL files (3,973,512 bytes) were copied to `data/threepoint_samples/smpl_extracted/` and matched manifest hashes. No source on 32 was edited and no held-out clip was used in this smoke run.

`logs/omni_real20_smoke_bf16` trained fresh for 6 iterations, saved, then resumed through iteration 12. Unlike the earlier synthetic fixture, this used the full default encoder, action decoder, world model and critic widths. It used 16 environments, pi/wm batches of 8, horizon 4, warmup 8, buffer 16, and BF16: it remains a small interface run, not a performance benchmark. All six CUDA graph phases captured before and after resume. At iteration 12 the encoder gradient norms were Motion `0.00559`, SMPL `0.00693`, Teleop `0.00344`; all auxiliary losses were finite.

Repeat the fresh run (use a new output directory):

```bash
OMP_NUM_THREADS=2 .venv/bin/python scripts/train_omni.py \
  --index results/omni_real20/index.json --data-root data/threepoint_samples \
  --log-dir logs/omni_real20_smoke_repeat --gpus 1 \
  --num_envs 16 --pi_batch 8 --wm_batch 8 --horizon 4 --wm_window 2 \
  --buffer_steps 16 --warmup_steps 8 --iterations 6 --wm_batches 1 \
  --critic_minibatches 2 --critic_epochs 1 --log_every 2 --amp bf16 --ckpt_every 6
```

For resume retain those settings, add `--resume logs/omni_real20_smoke_repeat/last.pt`, and set `--iterations 12`.

The iteration-6 bundle is `deploy_artifacts/omni_real20_smoke_walk`. On the start/middle/end of all 20 real clips (60 samples), split ONNX versus PyTorch maximum action error was `2.24e-8`. All three modalities also passed pre/motion/post inference checks using fixed nonzero heading and current robot rotation. Evidence: `results/omni_real20/inference_validation.json`.

All three iteration-12 evaluation entry points completed on the same 20 clips and wrote `results/omni_real20/eval_smoke_{motion,smpl,teleop}.{json,manifest.json,md}`. **Each had 20 failures and approximately 0.084 normalized return**: these nearly untrained policies do not track the motions. Entry-point execution is the only acceptance claim. Do not use the smoke bundle for hardware or interpret it as a completed final sim2sim. Full-data learning, actual two-A100 performance, trained-policy final simulation and live VR remain pending.
