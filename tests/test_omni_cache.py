import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
import joblib
import numpy as np
import pytest
import torch


def test_prepare_resume_fingerprints_and_end(tmp_path, monkeypatch):
    spec=importlib.util.spec_from_file_location("prepare_omni",Path(__file__).parents[1]/"tools/prepare_omni.py")
    tool=importlib.util.module_from_spec(spec);spec.loader.exec_module(tool)
    source=tmp_path/"motion.csv";source.write_text("synthetic source")
    xml=tmp_path/"robot.xml";xml.write_text("synthetic robot fingerprint")
    smpl=tmp_path/"smpl.pkl"
    joblib.dump(dict(fps=50,pose_aa=np.zeros((12,72),np.float32),smpl_joints=np.zeros((12,24,3),np.float32)),smpl)
    record=dict(motion_id="synthetic",g1_path="motion.csv",smpl_path="smpl.pkl",split="train",mirror=False,
                source_group_id="synthetic",split_group_id="synthetic",
                pairing=dict(common_frames50=10,smpl_sha256=tool.sha(smpl)))
    key=hashlib.sha256(b"synthetic").hexdigest()
    tp=tmp_path/"tp"/"clips"/"train"/key[:2]/(key+".npz");tp.parent.mkdir(parents=True)
    metadata=dict(motion_id="synthetic",frames=10,source_sha256=tool.sha(source),smpl_sha256=tool.sha(smpl),robot_xml_sha256=tool.sha(xml))
    np.savez(tp,metadata=json.dumps(metadata),position_local=np.zeros((10,3,3),np.float32),
             quaternion_local=np.tile(np.array([1,0,0,0],np.float32),(10,3,1)))
    q=torch.arange(12.)[:,None].expand(12,29).clone();q[10:]=999
    robot=SimpleNamespace(mjcf_path=xml,load_motion=lambda path: ())
    monkeypatch.setattr(tool,"resample_motion",lambda: (torch.zeros(12,3),torch.tensor([1.,0,0,0]).expand(12,4),q))
    import blinkmimic.g1.g1_constants as constants
    monkeypatch.setattr(constants,"SPEC",robot)
    manifest=tmp_path/"train.jsonl";manifest.write_text(json.dumps(record)+"\n")
    out=tmp_path/"out"
    args=["prepare_omni.py","--manifest",str(manifest),"--data-root",str(tmp_path),"--threepoint-root",str(tmp_path/"tp"),"--out",str(out)]
    monkeypatch.setattr(sys,"argv",args);tool.main()
    values=np.load(out/"shard_000000.npy")
    assert values.shape==(10,184)
    np.testing.assert_allclose(values[:,29:58],50.,atol=1e-4) # derivative before shared endpoint
    before=(out/"shard_000000.npy").stat().st_mtime_ns
    monkeypatch.setattr(sys,"argv",args+["--resume"]);tool.main()
    assert (out/"shard_000000.npy").stat().st_mtime_ns==before
    assert json.loads((out/"index.json").read_text())["records"][0]["frames"]==10
    source.write_text("changed")
    with pytest.raises(ValueError,match="Source changed"):tool.main()
    assert json.loads((out/"failures.json").read_text())
