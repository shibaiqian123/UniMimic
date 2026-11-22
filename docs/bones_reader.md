# BONES-SEED G1 CSV loading

The G1 `SPEC.load_motion` entry point accepts both existing headerless LAFAN1
CSV files and BONES-SEED headered CSV files. `load_lafan1_csv` is unchanged.

BONES input contract:
- 29 joint columns are selected by `G1_JOINT_NAMES` plus the `_dof` suffix,
  independently of the CSV column order.
- `root_translateX/Y/Z`: centimeters -> meters.
- `root_rotateX/Y/Z`: degrees, lowercase SciPy `xyz` convention
  (extrinsic XYZ, Rz @ Ry @ Rx) -> normalized wxyz quaternion.
- Joint angles: degrees -> radians.
- Consecutive integer `Frame` samples, source rate explicitly 120 Hz.
- No clipping, rebasing or downsampling inside the reader.
- The existing MotionLib interpolates position/joints and SLERPs rotations
  directly from 120 Hz to its 50 Hz control rate.

The rotation convention follows the computation in
GR00T-WholeBodyControl's
`gear_sonic/data_process/convert_soma_csv_to_motion_lib.py::load_bones_csv`;
this adapter is an independent NumPy implementation, with no new runtime dependency.

Training/evaluation continue to use `--motion-dir` / `--motion_dir` as before.
MotionLib still selects only `*.csv` directly inside the chosen directory;
it does **not** recursively load all date directories. Select a date directory
or prepare a flat subset directory of CSV symlinks. The unrestricted nonempty
CSV count (including the existing 45-clip dataset) is preserved.

Tests on 229:
```bash
cd /home/cqu023/blinkmimic
PYTHONPATH=src OMP_NUM_THREADS=1 .venv/bin/python -m pytest tests/test_bones.py -q
```

This change validates reading and resampling only, not BONES tracking performance,
subset splitting, policy training, or real robot deployment.
