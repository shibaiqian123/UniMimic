# BONES first subset (v1, seed 0)

Created under /share/2025slx/UniMimic/data/bones_seed/subsets/bones1000_v1_s0 on host32.
Source code lives on host229 in tools/build_bones_subset.py. No host32 source files were edited.

Files:
- train.jsonl / val.jsonl / test.jsonl: full selected metadata, stable source-path motion_id,
  original take group, actor split group, SHA256, frames, duration, and CSV link filename.
- train/ / val/ / test/: flat CSV symlinks accepted by the existing MotionLib/train/eval entry points.
- group_assignments.json: frozen actor assignments across the full metadata catalog.
- config.json: pinned source revision and metadata hash, seed, filters, previous15 quarantine.
- audit.json: category distributions, durations, manifest hashes, and leakage checks.

Split policy:
- 1000 train, 100 val, 100 test; 1000 distinct selected source takes in training.
- Source 120Hz; the existing loader/MotionLib still resamples to 50Hz.
- Non-mirrored originals only, metadata props=0, duration 2-20 seconds.
- Deterministic square-root category weighting, not the first 1000 files.
- Actor-disjoint split (stricter than take-level split): all dates/takes/cuts/mirrors with
  an actor ID stay in the same set. Metadata take_date + actor_uid + take_org_name
  remains the source-take identity. Whole catalog checked for source/mirror consistency.
- Previous validation15 actors are forced to train, so these observed sources cannot
  enter validation/test when continuing the prior model.
- Train: 1000 clips / 268 actors / 113.902 min, 20 categories.
  Val: 100 clips / 33 actors / 11.375 min.
  Test: 100 clips / 36 actors / 11.114 min.
- All 1200 links resolve; no selected duplicate content or cross-split actor/source groups.
  Rare categories are not guaranteed to appear in all splits.

Human-motion pairing:
All original move_name, filename, take_name, take_org_name, date, actor and
move_soma_uniform/proportional paths (including shape paths) are preserved inside metadata.
SOMA is not automatically SMPL. Actual SMPL correspondence/time alignment is still pending.
A future derived slice must inherit its source split_group_id; do not randomly split slices.

Use train/ for training, val/ for checkpoint selection, and test/ only for final held-out evaluation.
The 1200 clips have not all undergone physics/playback quality filtering. No training was launched;
only the existing launcher dry-run was checked.

Builder example (after normal source synchronization; not executed as a source change on host32):
    python tools/build_bones_subset.py --metadata data/bones_seed/raw/metadata/seed_metadata_v004.csv --extracted data/bones_seed/extracted --revision-file data/bones_seed/REVISION.txt --seen-manifest data/bones_seed/subsets/validation15/manifest.json --out data/bones_seed/subsets/bones1000_v1_s0 --seed 0

Existing output directories are refused to prevent silently changing a published split.
