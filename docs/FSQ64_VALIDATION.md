# 64D implementation validation — host 229

Date: 2026-11-20. Code changes only on 229. No changes on host 32; no real robot started. Tests ran on the existing RTX 4090, not an A100.

## Passed
- 43 targeted tests: latent modes, exact legacy32 regression (including existing ckpt_003500 and final_3500_walk1 bundle), all 32 scalar levels, rounding ties/neighbors, reference source formula, STE gradients, BF16, compiled backward, EMA, strict checkpoint modes, CUDA Graph forward/backward, existing deployment/controller regressions.
- FSQ formula compared against actual vector-quantize-pytorch 1.27.15 source in the isolated test cache. It is not a runtime dependency.
- Real simulation training loop: each mode ran 4 iterations on 16 environments, then resumed from last.pt through iteration 6. These smoke runs used small networks (encoder/decoder 64,32; world model 64,64), BF16 and EMA. This does not establish full production-recipe throughput or convergence.
- Both modes exported policy / motion_encoder / decoder and loaded v2 bundles in the NumPy/ONNX runtime.
- FSQ export maximum action and split-action error: 2.6193447411060333e-09.
- Continuous64 export maximum action and split-action error: 2.561137080192566e-09.
- FSQ dynamic batch=3 produced [3,2,32] tokens exactly on the grid; full vs split ONNX action error 0.
- Each new bundle ran 3 DeployCore static-reference steps with finite 29D actions. v2 loading did not import torch.

## Evaluation is an interface smoke test, not motion acceptance
Both iteration-6 checkpoints ran through the 40-clip evaluator, including action_delta_rms and checkpoint/data manifests. Both failed all 40 clips; normalized returns were about 0.002. This is expected of these minimally trained policies and is NOT a claim of usable tracking.

Artifacts:
- logs/fsq_implementation_smoke/{continuous64,fsq64}/ : checkpoints, args, train/latent logs.
- logs/{continuous64,fsq64}_resume_console.log : actual resume output.
- deploy_artifacts/{continuous64,fsq64}_interface_smoke/ : interface-only ONNX packages, not hardware candidates.
- results/fsq64_interface_smoke.{json,manifest.json,md} : evaluator outputs.
- results/fsq64_implementation_validation.json : machine-readable summary.

## Not executed
Long A100 training, multi-seed comparison, trained64 walk1 full final simulation, and hardware validation. Production commands are in FSQ64.md. Old 32D hardware bundles were not overwritten.

Full test command:
OMP_NUM_THREADS=1 .venv/bin/python -m pytest -q tests/test_latent.py tests/test_deploy_runtime.py tests/test_final_pipeline.py tests/test_sim2sim.py tests/test_warp_runner.py
