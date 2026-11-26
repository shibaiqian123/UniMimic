# Omni instability diagnostics

This adds measurements, not a new recipe. Horizon, UTD, world-model updates, auxiliary coefficients and clipping are unchanged. Default diagnostics are disabled.

## Historical experiment

Read-only source on 32: logs/bones_omni_fsq64_2a100_env16384_pi4096_s0_20261125_152055/train_log.csv.

229 evidence: results/omni_instability_17k/comparison.json.
Reproduce: python tools/summarize_training_instability.py TRAIN_LOG.csv --out comparison.json

Between 16000–17000 and 17800–18100, window medians changed:
- wm_err8: 0.12339 → 0.65983 (5.35×).
- Critic fitting loss: 0.02953 → 1.82309 (61.75×).
- Combined preclip actor gradient norm: 1.56551 → 17.96544 (11.48×).
- Simulator reward: 0.77495 → 0.17006.

At 17800, critic loss was 26.14389 and actor gradient norm 94.17434; at 17850 actor gradient norm reached 180.27215. Recovery at 19000 was followed by deterioration at 19400. Nonfinite stayed zero: finite values can still be unstable.

These are moving training distributions and normalized errors, not a controlled causal experiment. The inspected 18000 checkpoint contains only pi, iteration, wall_time, sim_steps, args, saved_at. It cannot reconstruct historical critic/world-model states or separate gradients. None of auxiliary loss, FSQ, critic or world model is proven to be the original cause.

## Measurements

Enable --diagnostic-every 250. Start/end bounds control the range (end 0 means unbounded).
Only diagnostic actor updates execute eagerly for separate retained-graph differentiation. Other updates retain CUDA Graphs. AOT backward buffer donation is disabled when diagnostics are enabled because retained-graph backward cannot use donated buffers; compilation remains enabled. Extra backwards and snapshots cost time/memory: diagnostic runs are not a fair throughput benchmark.

training_diagnostics.jsonl and TensorBoard diagnostic/ contain:
- gradient/encoder_motion, encoder_smpl, encoder_teleop, action_decoder, reconstruction: policy/weighted-auxiliary norms, combined norm, cosine, cosine_defined and aux_over_policy. Gradients are averaged as vectors across ranks BEFORE norms/clipping. Current equal per-rank actor batches are required.
- Action decoder has zero auxiliary gradient by architecture; reconstruction decoder has zero policy-objective gradient. Undefined cosine is zero with cosine_defined=0.
- Policy/auxiliary losses, bootstrap contribution, total imagined return.
- Critic imagined GAE-target MSE before critic updates, one-step TD residual, value/target scales.
- Fixed real replay panel, open-loop world-model errors at horizons 1/4/8/16/32 when available: normalized MSE and physical component RMSE.
- Critic one-step residual on that real panel, using training soft continuation (timeouts are not physical termination). This is a Bellman residual, not Monte Carlo ground-truth value error.

The panel is sampled once per rank with RNG restored afterward and persisted as diagnostic_panel_rankN.pt. Later diagnostics and same-directory resume reuse it. It is training replay, not held-out validation. Compare fixed-panel and current imagined errors together: an early panel cannot measure all later state coverage. Counts accompany masked errors; zero-count horizons do not establish accuracy.

Gradients precede the actor update; replay errors follow the iteration's updates. JSON records this distinction.

--full-ckpt-every 1000 preserves full_001000.pt etc. with live/EMA policy, critic, world model, optimizers, normalizers, per-rank RNG and controller state. Normal ckpt_*.pt evaluation files remain. Full checkpoints are larger. Resume refills replay and is not bit-exact.

## 32 diagnostic experiment (after Git sync)

Do not resume from the historical policy-only 15000 checkpoint. Start a bounded fresh experiment with the agreed LR-multiplier reduction starting no later than 15000. This code change does not automatically launch it.

~~~bash
cd /share/2025slx/UniMimic
DATA=/share/2025slx/UniMimic/data/bones_seed
CACHE="$DATA/derived/omni_sonic_v1"
RUN="logs/bones_omni_diag_decay15k_s0_$(date +%Y%m%d_%H%M%S)"
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=4 \
.venv/bin/python -u scripts/train_omni.py \
  --gpus 2 --index "$CACHE/train/index.json" --data-root "$DATA" \
  --log-dir "$RUN" --num_envs 16384 --pi_batch 4096 \
  --iterations 20000 --ckpt_every 1000 --full-ckpt-every 1000 \
  --lr_decay_ret 0.75 --lr_decay_by 15000 --lr_decay_len 3000 \
  --diagnostic-every 250 --diagnostic-start 1000 \
  --diagnostic-batch 64 --diagnostic-window 32 \
  --seed 0 --tensorboard
~~~

To resume a full snapshot, retain the run, data, world size and global recipe; add --resume "$RUN/full_020000.pt" and set new total iterations. Do not extend merely because training has not raised an exception.

## Decisions after measurements

- Auxiliary encoder gradients surge/oppose policy gradients while policy gradients stay stable: inspect auxiliary terms/scale first.
- Error grows strongly with prediction horizon while one-step error stays controlled: test a shorter horizon against the unchanged baseline. Bootstrap dependence also changes.
- Critic residuals/value scale worsen while world-model errors stay controlled: inspect critic fitting and bootstrap before changing world-model update count.
- Both one-step and multi-step world-model errors worsen: investigate fitting/data coverage before an update-ratio experiment.
- Change one factor at a time on the same data/seed. Require repeated Motion/SMPL/Teleop fixed-validation evaluations across 15k–20k to retain/improve returns and failures, together with non-diverging diagnostics. Then extend a full checkpoint to a bounded milestone. Do not repeatedly select on the test set.

## Validation scope

CPU tests cover gradient addition, untouched .grad buffers, zero auxiliary action-decoder gradient, two-rank vector reduction and replay sampling compatibility. GPU smoke uses the existing 20-clip fixture, reduced batch/horizon and BF16. It tests execution/capture/checkpoint interfaces, not tracking ability or 17k stability. Actual two-A100 full-batch memory, throughput and long-training diagnosis remain pending on 32.

### Executed on 229
- 11 tests passed, including two-rank CPU/Gloo gradient-vector equivalence.
- BF16 real-20-clip single-GPU smoke: fresh 6 iterations, resume full_000006.pt to 12.
- Diagnostic rows at 3/6/9/12 were finite; expected zero gradients verified.
- Fixed replay panel SHA256 unchanged after resume.
- All six CUDA Graph phases captured on the resumed run.
- Numbered full checkpoint included all model/optimizer/controller states (~293.5 MiB with default network widths).
- Evidence: results/training_diagnostics_validation.json; logs/omni_diagnostic_smoke_v3.
