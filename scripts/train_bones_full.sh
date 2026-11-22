#!/usr/bin/env bash
# Full BONES train-side catalog; frozen val/test groups remain excluded.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT_DIR="$PWD"
MOTION_DIR="$ROOT_DIR/data/bones_seed/subsets/bones_full_v1_s0/train"
if [[ ! -d "$MOTION_DIR" ]]; then
    echo "Missing full train view: $MOTION_DIR" >&2
    exit 2
fi
GPUS="${GPUS:-1}"
ITERATIONS="${ITERATIONS:-30000}"
LOG_DIR="${LOG_DIR:-logs/bones_full_fsq64_proprio93_s0_$(date +%Y%m%d_%H%M%S)}"
RESUME_ARGS=()
if [[ -n "${RESUME:-}" ]]; then
    RESUME_ARGS=(--resume "$RESUME")
fi
exec .venv/bin/python scripts/train_latent.py \
    --gpus "$GPUS" --latent-mode fsq64 --observation-mode proprio93 \
    --motion-dir "$MOTION_DIR" --log-dir "$LOG_DIR" \
    --num_envs 4096 --iterations "$ITERATIONS" --seed 0 --tensorboard \
    --motion-compact --eprobe-max-envs 1024 \
    --ckpt_every 1000 --lr_decay_by 0 --lr_decay_len 3000 \
    "${RESUME_ARGS[@]}" "$@"
