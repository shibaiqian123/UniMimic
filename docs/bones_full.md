# Full BONES Motion training

The data view on host32 is `/share/2025slx/UniMimic/data/bones_seed/subsets/bones_full_v1_s0`.
It contains 115011 train-side clips (57524 originals + 57487 mirrors), approximately 226.705 hours and 40868168 frames at 50Hz. The full metadata has 142220 clips; 13587 validation-side and 13622 test-side source clips remain excluded from training.

`group_assignments.json` is copied from bones1000_v1_s0. `val/`, `test/`, `val.jsonl`, and `test.jsonl` retain the exact prior 100-clip benchmarks. The full reserved source catalogs are in `val.catalog.jsonl` and `test.catalog.jsonl`. `train.catalog.jsonl` preserves original metadata and G1/SOMA paths. All prior 1000 training sources are retained. No props, duration, category or mirror filter is applied to the expanded train catalog; structurally valid files are not automatically physically feasible tracking targets.

## Scale-specific implementation

`--motion-compact` retains FP32 root/joint references and computes reference body FK only for the requested frames. It preserves velocities, goal features, rewards and reference contact formulas; no latent/network changes. This trades extra FK compute for lower reference memory. The default dense mode remains available.

Reference bin metadata is built on the CPU without one GPU synchronization per bin. Contact initialization slices contiguous clips instead of scanning the full frame-to-clip vector once per clip.

`--eprobe-max-envs 1024` limits probe worlds to a fixed shape. Successive sweeps rotate over the full tile list; 0 retains legacy all-world behavior. A capped sweep is a partial probe, not a complete evaluation. The cursor derives from iteration/start/every settings, so keeping those settings restores the same schedule. A full probe cycle may span many training iterations.

## Host32 commands after synchronizing source from 229

```bash
cd /share/2025slx/UniMimic
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 bash scripts/train_bones_full.sh
```

Defaults: single A100, global 4096 environments, FSQ64/proprio93, seed0, 30000 iterations, checkpoints every1000, compact references and capped probes. This starts a new model. 30000 is an initial budget, not a claim of convergence. The forced policy-LR decay deadline at1400 is disabled; the existing return trigger0.75 is retained and decay duration extended to3000 iterations. Do not compare it as an identical-recipe data-only ablation to the 1000-clip run.

Override the budget or log path using `ITERATIONS` and `LOG_DIR`. Two GPU launch uses `GPUS=2 CUDA_VISIBLE_DEVICES=0,1`; real dual-GPU validation for this full-data configuration is pending. No speedup is promised.

To resume, use the same full-data directory and configuration, set `LOG_DIR` to the prior run and `RESUME` to its `last.pt`, and increase `ITERATIONS`. The full checkpoint refills replay rather than restoring it exactly. A periodic light checkpoint is for evaluation/export, not optimizer resume.

For checkpoint selection, use `tools/eval_metrics.py` with `--motion_dir data/bones_seed/subsets/bones_full_v1_s0/val`. Evaluate the chosen checkpoint on `test` afterward. Do not point the existing all-world evaluator at all115011 training clips; it has no full-catalog batching in this change.

## Validation scope

Tests cover dense/compact reference FK, goal inputs, reward/gradient and critic-feature parity, unchanged contact targets, bin metadata, probe rotation, CUDA compilation/capture, BONES loader and terminal-bin reset. Full manifest expansion preserves frozen holdout views. Resource and short-training results are recorded separately; long-run learning quality, exact full-data throughput and hardware deployment are not established by these tests.


Validation artifacts: results/full_motion_validation/summary.json and adjacent logs.
A100 small-network 32-env smoke reached iteration10, including save/resume and
collection/world-model/policy/probe CUDA Graph capture/replay.
The real full115011-clip reference and contact construction completed without OOM.
Its final sample check exposed float32 linspace rounding at 40868168 frames; the
single/distributed diagnostic samplers now use integer arithmetic, with CPU/CUDA
large-index regression tests. The original failed diagnostic prevented exact peak
memory report serialization; approximately13.55GiB of reference tensors is an estimate,
not an observed full-training peak. Do not describe the smoke as4096-env full-budget validation.
Every launch still parses CSVs; progress is printed for fingerprints, parsing,
velocity derivation and contacts. First-use graph compilation adds startup time.
