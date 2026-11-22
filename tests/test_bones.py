import csv

import numpy as np
import pytest

from blinkmimic.g1.g1_constants import (
    G1_JOINT_NAMES, SPEC, load_bones_csv, load_g1_csv, load_lafan1_csv,
)


def fixture_csv(tmp_path, frames=121, reorder=False):
    header = (["Frame"] + [f"root_translate{x}" for x in "XYZ"]
              + [f"root_rotate{x}" for x in "XYZ"]
              + [n + "_dof" for n in G1_JOINT_NAMES])
    raw = np.zeros((frames, len(header)))
    raw[:, 0] = np.arange(frames)
    raw[:, 1] = np.arange(frames) * 100 / 120
    raw[:, 3] = 80
    raw[:, 4:7] = [20, -30, 70]
    raw[:, 7:] = np.arange(29) * 2
    order = np.arange(len(header))[::-1] if reorder else np.arange(len(header))
    path = tmp_path / ("shuffled.csv" if reorder else "motion.csv")
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([header[i] for i in order])
        writer.writerows(raw[:, order])
    return path, raw


def test_units_names_rotation(tmp_path):
    from scipy.spatial.transform import Rotation
    path, raw = fixture_csv(tmp_path, reorder=True)
    p, q, j, fps = load_bones_csv(path)
    np.testing.assert_allclose(p, raw[:, 1:4] / 100)
    expected = Rotation.from_euler("xyz", raw[:, 4:7], degrees=True).as_quat()[:, [3,0,1,2]]
    np.testing.assert_allclose(q, expected, atol=1e-12)
    np.testing.assert_allclose(j, np.deg2rad(raw[:, 7:]))
    assert fps == 120
    for actual, expected in zip(load_g1_csv(path), (p,q,j,fps)):
        np.testing.assert_array_equal(actual, expected)


def test_lafan_unchanged(tmp_path):
    raw = np.random.default_rng(7).normal(size=(15,36))
    path = tmp_path / "lafan.csv"
    np.savetxt(path, raw, delimiter=",")
    expected = (raw[:, :3], raw[:, [6,3,4,5]], raw[:, 7:36], 30.)
    for reader in (load_lafan1_csv, load_g1_csv, SPEC.load_motion):
        for actual, value in zip(reader(path), expected):
            np.testing.assert_array_equal(actual, value)


@pytest.mark.parametrize("problem", ["missing", "duplicate", "gap", "nan", "one_frame"])
def test_invalid_source(tmp_path, problem):
    path, _ = fixture_csv(tmp_path, frames=2 if problem != "one_frame" else 1)
    rows = list(csv.reader(path.open()))
    if problem == "missing":
        rows[0][-1] = "unknown_joint"
    elif problem == "duplicate":
        rows[0][-1] = rows[0][-2]
    elif problem == "gap":
        rows[2][0] = "2"
    elif problem == "nan":
        rows[1][1] = "nan"
    with path.open("w", newline="") as f:
        csv.writer(f).writerows(rows)
    with pytest.raises(ValueError):
        load_bones_csv(path)


def test_motionlib_resamples_120_to_50(tmp_path):
    from blinkmimic.kinematics import Kinematics
    from blinkmimic.motion import MotionLib
    path, _ = fixture_csv(tmp_path)
    lib = MotionLib.from_dir(str(tmp_path), SPEC, Kinematics(SPEC, "cpu"), "cpu")
    assert lib.num_frames == 51  # 121 source samples span exactly one second.
    np.testing.assert_allclose(lib.root_p[:, 0], np.arange(51)/50, atol=1e-6)
    np.testing.assert_allclose(lib.q[0], lib.q[-1])
    assert lib.names == [path.stem]
