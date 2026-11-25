import numpy as np
import torch
import mujoco
from blinkmimic.g1.g1_constants import SPEC
from blinkmimic.kinematics import Kinematics
from blinkmimic.motion import resample_motion
from blinkmimic.rotations import quat_to_mat,quat_slerp
from blinkmimic.threepoint import poses,BODIES,OFFSETS

def test_mujoco_and_numpy_frames():
    kin=Kinematics(SPEC,"cpu")
    model=mujoco.MjModel.from_xml_path(SPEC.mjcf_path);data=mujoco.MjData(model)
    gen=torch.Generator().manual_seed(4)
    p=torch.randn(12,3,generator=gen)
    q=torch.randn(12,4,generator=gen);q=q/q.norm(dim=-1,keepdim=True)
    j=torch.tensor(SPEC.default_joint_pos_list())[None]+torch.randn(12,29,generator=gen)*.15
    out=poses(kin,p,q,j)
    opposite=poses(kin,p,-q,j)
    for k in out:np.testing.assert_allclose(out[k],opposite[k],atol=1e-6)
    Rw=quat_to_mat(torch.from_numpy(out["quaternion_world"])).numpy()
    Rl=quat_to_mat(torch.from_numpy(out["quaternion_local"])).numpy()
    adr=[model.jnt_qposadr[model.joint(n).id] for n in SPEC.joint_names]
    for t in range(12):
        data.qpos[:3]=p[t];data.qpos[3:7]=q[t];data.qpos[adr]=j[t]
        mujoco.mj_forward(model,data)
        a=model.body("pelvis").id
        ar=data.xmat[a].reshape(3,3);ap=data.xpos[a]
        for i,name in enumerate(BODIES):
            b=model.body(name).id;r=data.xmat[b].reshape(3,3)
            w=data.xpos[b]+r@np.array(OFFSETS[i])
            np.testing.assert_allclose(out["position_world"][t,i],w,atol=1e-5)
            np.testing.assert_allclose(out["position_local"][t,i],ar.T@(w-ap),atol=1e-5)
            np.testing.assert_allclose(Rw[t,i],r,atol=1e-5)
            np.testing.assert_allclose(Rl[t,i],ar.T@r,atol=1e-5)
    # Local features invariant under global translation and full global rotation.
    rot=quat_to_mat(torch.tensor([[.8,.2,-.3,.4]]))[0]
    from blinkmimic.rotations import mat_to_quat
    transformed=poses(kin,p@rot.T+torch.tensor([2.,3.,4.]),mat_to_quat(rot@quat_to_mat(q)),j)
    np.testing.assert_allclose(out["position_local"],transformed["position_local"],atol=2e-6)
    np.testing.assert_allclose(Rl,quat_to_mat(torch.from_numpy(transformed["quaternion_local"])),atol=2e-6)
    for name in ("quaternion_world","quaternion_local","pelvis_quaternion_world"):
        assert np.min((out[name][1:]*out[name][:-1]).sum(-1))>=-1e-6

def test_resampling_exact_legacy_formula():
    rng=np.random.default_rng(3)
    p=rng.normal(size=(124,3));q=rng.normal(size=(124,4));q/=np.linalg.norm(q,axis=-1,keepdims=True)
    j=rng.normal(size=(124,29))
    result=resample_motion(p,q,j,120)
    p,q,j=(torch.tensor(x,dtype=torch.float32) for x in (p,q,j))
    n=int(np.floor(((len(p)-1)/120)/.02))+1
    t=torch.arange(n,dtype=torch.float32)*.02*120
    i=t.floor().long().clamp(max=len(p)-2);a=(t-i.float()).clamp(0,1)
    expected=(p[i]*(1-a[:,None])+p[i+1]*a[:,None],
              quat_slerp(q[i],q[i+1],a),j[i]*(1-a[:,None])+j[i+1]*a[:,None])
    for x,y in zip(result,expected):assert torch.equal(x,y)

def test_cache_resume_invalid_and_changed_source(tmp_path,monkeypatch):
    import importlib.util,json
    from pathlib import Path
    import blinkmimic.threepoint as tp
    spec=importlib.util.spec_from_file_location("generator",Path(__file__).parents[1]/"tools/generate_threepoint.py")
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    mod.ROOT=tmp_path;mod.OUT=tmp_path/"out";mod.RESUME=True
    mod.CONFIG=tp.CONTRACT;mod.SPECIFICATION=None;mod.KIN=None
    source=tmp_path/"a.csv";source.write_text("first")
    calls=[]
    def fake(*args):
        calls.append(1)
        return dict(time_s=np.array([0,.02],dtype=np.float32),
                    position_world=np.zeros((2,3,3),np.float32),position_local=np.zeros((2,3,3),np.float32),
                    quaternion_world=np.tile(np.array([1,0,0,0],np.float32),(2,3,1)),
                    quaternion_local=np.tile(np.array([1,0,0,0],np.float32),(2,3,1)),
                    pelvis_position_world=np.zeros((2,3),np.float32),
                    pelvis_quaternion_world=np.tile(np.array([1,0,0,0],np.float32),(2,1)))
    monkeypatch.setattr(tp,"from_csv",fake)
    r=dict(motion_id="a",split="train",mirror=False,source_group_id="a",split_group_id="actor",
           g1_path="a.csv",smpl_path="s.pkl",pairing=dict(smpl_sha256="x",common_frames50=2,common_end_s=.02))
    x=mod.process(r);assert x["status"]=="ok" and not x["reused"]
    assert mod.process(r)["reused"]
    (mod.OUT/x["cache"]).write_bytes(b"broken")
    assert not mod.process(r)["reused"]
    source.write_text("changed")
    assert not mod.process(r)["reused"]
    assert len(calls)==3
