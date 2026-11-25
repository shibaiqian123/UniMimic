import importlib.util
from pathlib import Path
import sys
import types
import numpy as np

spec=importlib.util.spec_from_file_location("pairs",Path(__file__).parents[1]/"tools/build_bones_smpl_pairs.py")
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

def test_exact_identity_and_mirror():
    r={"relative_path":"g1/csv/date/walk__A001_M.csv","metadata":{"filename":"walk__A001_M"}}
    assert m.match_key(r)=="walk__A001_M"
    r["metadata"]["filename"]="walk__A001"
    import pytest
    with pytest.raises(ValueError):m.match_key(r)

def test_timing_and_invalid_arrays(tmp_path,monkeypatch):
    f=tmp_path/"a.pkl";f.write_bytes(b"fixture")
    d={"fps":50.0,"pose_aa":np.zeros((50,72)),"smpl_joints":np.zeros((50,24,3)),"transl":np.zeros((50,3))}
    monkeypatch.setitem(sys.modules,"joblib",types.SimpleNamespace(load=lambda _:d))
    result=m.inspect_pair(("a",str(f),120,120))
    assert result["status"]=="matched"
    assert result["common_frames50"]==50
    assert m.inspect_pair(("a",str(f),240,120))["status"]=="duration_mismatch"
    d["pose_aa"][0,0]=np.nan
    assert m.inspect_pair(("a",str(f),120,120))["status"]=="invalid_smpl"

def test_two_stage_endpoint_truncation(tmp_path,monkeypatch):
    f=tmp_path/"a.pkl";f.write_bytes(b"fixture")
    d={"fps":50.0,"pose_aa":np.zeros((1150,72)),"smpl_joints":np.zeros((1150,24,3)),
       "transl":np.zeros((1150,3)),"original_fps":30.0,"original_pose_aa":np.zeros((691,72))}
    monkeypatch.setitem(sys.modules,"joblib",types.SimpleNamespace(load=lambda _:d))
    result=m.inspect_pair(("a",str(f),2764,120))
    assert result["status"]=="matched"
    assert abs(result["duration_delta_s"]+.045)<1e-6
    # A longer final sequence is not explained by truncation.
    d["original_pose_aa"]=np.zeros((680,72))
    assert m.inspect_pair(("a",str(f),2764,120))["status"]=="duration_mismatch"
