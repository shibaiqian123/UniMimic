import numpy as np
import pytest
import torch
from blinkmimic.g1.g1_constants import SPEC
from blinkmimic.kinematics import Kinematics
from blinkmimic.motion import MotionLib
from blinkmimic.mdp import MDP
from blinkmimic.probe_batches import probe_tiles,select_probe_batch


def motions(tmp_path,device):
    for i,n in enumerate((31,61)):
        x=np.zeros((n,36));x[:,2]=.78;x[:,6]=1
        x[:,0]=np.arange(n)*.01
        x[:,7:]=np.asarray(SPEC.default_joint_pos_list())+np.sin(np.arange(n)[:,None]*.04)*.02
        np.savetxt(tmp_path/f"clip{i}.csv",x,delimiter=",")
    kin=Kinematics(SPEC,device)
    dense=MotionLib.from_dir(str(tmp_path),SPEC,kin,device)
    compact=MotionLib.from_dir(str(tmp_path),SPEC,kin,device,compact=True)
    return kin,dense,compact


def test_compact_reference_parity_and_bins(tmp_path):
    kin,a,b=motions(tmp_path,"cpu")
    for name in ("clip_len","clip_start","clip_end","clip_id","bin_clip","bin_start","frame_bin","root_R","q","qd"):
        assert torch.equal(getattr(a,name),getattr(b,name)),name
    f=torch.arange(a.num_frames)
    for x,y in zip(a.ref_bodies(f),b.ref_bodies(f)):
        torch.testing.assert_close(x,y,rtol=0,atol=0)
    assert not hasattr(b,"body_R")
    feet=[3,6]
    a.set_ref_contact(feet);b.set_ref_contact(feet)
    assert torch.equal(a.ref_contact,b.ref_contact)
    # Independent legacy mask-based contact computation.
    z=a.body_p[:,feet,2];vh=a.body_v[:,feet,:2].norm(dim=-1)
    old=torch.empty_like(z)
    for c in range(len(a.names)):
        mask=a.clip_id==c
        old[mask]=((z[mask]<z[mask].quantile(.05,dim=0)+.015)&(vh[mask]<.5)).float()
    assert torch.equal(a.ref_contact,old)
    for c in range(len(a.names)):
        s,e=int(a.clip_start[c]),int(a.clip_end[c])
        base=int(a.frame_bin[s])
        assert torch.equal(a.frame_bin[s:e+1],base+torch.arange(e-s+1)//50)
    f=torch.tensor([0,49,50,51,150])
    qa=a.ref_state(f)[4].clone().requires_grad_()
    qb=qa.detach().clone().requires_grad_()
    results=[]
    for motion,q in ((a,qa),(b,qb)):
        mdp=MDP(kin,motion,torch.tensor(SPEC.default_joint_pos_list()),0)
        state=list(motion.ref_state(f));state[0]=state[0]+.03;state[4]=q+.05
        action=torch.zeros(len(f),29)
        r=mdp.reward(tuple(state),f,action,action)
        grad=torch.autograd.grad(r.sum(),q)[0]
        cf=mdp.critic_features(tuple(state),f,torch.zeros(len(f),0))
        results.append((r,grad,cf))
    for x,y in zip(*results):torch.testing.assert_close(x,y,atol=1e-6,rtol=1e-6)


def test_probe_rotation_complete_and_fixed_shape():
    tiles=probe_tiles(torch.tensor([0,501]),torch.tensor([500,1001]),300)
    assert tiles.tolist()==[0,300,501,801]
    seen=set()
    for sweep in range(4):
        b=select_probe_batch(tiles,3,sweep)
        assert len(b)==3
        seen.update(b.tolist())
    assert seen==set(tiles.tolist())
    assert torch.equal(select_probe_batch(tiles,0,12),tiles)


@pytest.mark.skipif(not torch.cuda.is_available(),reason="CUDA required")
def test_compact_compile_and_cuda_graph(tmp_path):
    _,a,b=motions(tmp_path,"cuda")
    f=torch.tensor([0,49,50,51,150],device="cuda")
    fn=torch.compile(b.ref_bodies,fullgraph=True)
    for _ in range(3):out=fn(f)
    expected=a.ref_bodies(f)
    for x,y in zip(out,expected):torch.testing.assert_close(x,y,atol=1e-5,rtol=1e-5)
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):fn(f)
    torch.cuda.current_stream().wait_stream(stream)
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):out=fn(f)
    for _ in range(5):graph.replay()
    torch.cuda.synchronize()
    for x,y in zip(out,expected):torch.testing.assert_close(x,y,atol=1e-5,rtol=1e-5)


@pytest.mark.parametrize("total", [1, 51, 342231, 40868168, 1000000001])
def test_large_diagnostic_indices(total):
    from blinkmimic.probe_batches import diagnostic_frames
    for device in (["cpu","cuda"] if torch.cuda.is_available() else ["cpu"]):
        f=diagnostic_frames(total,256,device)
        assert f.dtype==torch.int64 and len(f)==min(total,256)
        assert int(f[0])==0 and int(f[-1])==total-1
        assert bool(((f>=0)&(f<total)).all())
        assert bool((f[1:]>f[:-1]).all())
