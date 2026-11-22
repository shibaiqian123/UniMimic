"""BONES-SEED G1 CSV adapter (120 Hz, centimeters, degrees).

Rotation convention matches GR00T-WholeBodyControl's
gear_sonic/data_process/convert_soma_csv_to_motion_lib.py::load_bones_csv:
SciPy Rotation.from_euler("xyz", ..., degrees=True), i.e. Rz @ Ry @ Rx.
Implemented independently with NumPy; no SciPy runtime dependency.
"""
import csv
from collections.abc import Sequence

import numpy as np

SOURCE_FPS = 120.0


def load_bones_csv(path: str, joint_names: Sequence[str]):
    """Return position [T,3] meters, quaternion [T,4] wxyz, joints [T,29] radians, Hz.

    Column order is irrelevant. The supplied joint names define model order.
    Frames must be contiguous: dropping rows would silently change motion timing.
    No resampling, coordinate rebasing, or joint clipping happens here.
    """
    if len(joint_names) != 29 or len(set(joint_names)) != 29:
        raise ValueError("BONES G1 requires 29 unique model joint names")
    with open(path, encoding="utf-8-sig", newline="") as stream:
        header = next(csv.reader(stream), [])
    if len(header) != len(set(header)):
        raise ValueError(f"{path}: duplicate CSV column names")
    required = (["Frame"] + [f"root_translate{axis}" for axis in "XYZ"]
                + [f"root_rotate{axis}" for axis in "XYZ"]
                + [f"{name}_dof" for name in joint_names])
    missing = [name for name in required if name not in header]
    if missing:
        raise ValueError(f"{path}: missing BONES columns: {missing}")
    raw = np.loadtxt(path, delimiter=",", skiprows=1,
                     usecols=[header.index(name) for name in required],
                     ndmin=2, encoding="utf-8-sig")
    if len(raw) < 2 or not np.isfinite(raw).all():
        raise ValueError(f"{path}: expected at least two finite motion frames")
    frames = raw[:, 0]
    if not np.all(frames == np.floor(frames)) or not np.all(np.diff(frames) == 1):
        raise ValueError(f"{path}: Frame must contain consecutive integer samples at 120 Hz")
    position = raw[:, 1:4] / 100.0
    half = np.deg2rad(raw[:, 4:7]) / 2.0
    cx, cy, cz = np.cos(half).T
    sx, sy, sz = np.sin(half).T
    quaternion = np.stack((cx*cy*cz + sx*sy*sz,
                           sx*cy*cz - cx*sy*sz,
                           cx*sy*cz + sx*cy*sz,
                           cx*cy*sz - sx*sy*cz), axis=-1)
    quaternion /= np.linalg.norm(quaternion, axis=-1, keepdims=True)
    joints = np.deg2rad(raw[:, 7:])
    return position, quaternion, joints, SOURCE_FPS
