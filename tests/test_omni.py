import torch
from blinkmimic.omni import OmniPolicy,DIMS,MODALITIES,assemble,future_indices,smpl_reference,auxiliary_total
from blinkmimic.rotations import so3_exp

def test_modalities_and_gradients():
    torch.manual_seed(17)
    pi=OmniPolicy((32,24),(32,24))
    inputs={m:torch.randn(6,d) for m,d in DIMS.items()}
    h=torch.randn(6,10,93)
    pi.update_norms(h,inputs)
    mode=torch.arange(6)%3
    out=pi(h,inputs,mode)
    for i,m in enumerate(MODALITIES):
        explicit=pi.decode_tokens(h,pi.encode_tokens(m,inputs[m]))
        torch.testing.assert_close(out[mode==i],explicit[mode==i])
    total,n=auxiliary_total(pi.auxiliary(inputs))
    (out.square().mean()+total/n).backward()
    for encoder in pi.encoders.values():
        grads=[p.grad for p in encoder.parameters()]
        assert all(g is not None and torch.isfinite(g).all() for g in grads)
        assert sum(g.square().sum() for g in grads)>0
    assert pi.decode_tokens(h,torch.zeros(6,2,32)).shape==(6,29)

def test_window_and_features():
    f=torch.tensor([2,98]);end=torch.tensor([99,99])
    assert future_indices(f,end,5)[0].tolist()==list(range(2,48,5))
    assert future_indices(f,end,1)[1].tolist()==[98]+[99]*9
    b=2
    q={"motion":torch.randn(b,10,29),"smpl":torch.randn(b,10,29)}
    R=so3_exp(torch.randn(b,10,3))
    robot=so3_exp(torch.randn(b,3)).requires_grad_()
    x=assemble(q,torch.randn(b,10,29),R,torch.randn(b,10,24,3),R,
               torch.randn(b,3,3),R[:,:3],robot,list(range(12)),list(range(23,29)))
    assert {k:v.shape[-1] for k,v in x.items()}==DIMS
    sum(v.sum() for v in x.values()).backward()
    assert robot.grad.abs().sum()>0

def test_smpl_official_formula():
    import numpy as np
    from scipy.spatial.transform import Rotation
    aa=torch.randn(8,72)*.3
    j=torch.randn(8,24,3)
    local,R=smpl_reference(aa,j)
    C=Rotation.from_rotvec([np.pi/2,0,0]).as_matrix()
    base=Rotation.from_quat([.5,.5,.5,.5]).as_matrix()
    expected=C@Rotation.from_rotvec(aa[:,:3].numpy()).as_matrix()@base.T
    np.testing.assert_allclose(R.numpy(),expected,atol=1e-6)
    np.testing.assert_allclose(local.numpy(),np.einsum("tji,tkj->tki",expected,j.numpy()),atol=1e-6)

def test_numpy_deploy_parity():
    import numpy as np
    from types import SimpleNamespace
    from blinkmimic.omni_data import pack,reference_inputs
    from blinkmimic.deploy.omni import features
    T=15
    torch.manual_seed(3)
    q=torch.randn(T,29);qd=torch.randn(T,29)
    R=so3_exp(torch.randn(T,3));sr=so3_exp(torch.randn(T,3))
    x=pack(q,qd,R,torch.randn(T,24,3),sr,torch.randn(T,3,3),so3_exp(torch.randn(T,3,3)))
    lower=list(range(12));wrists=[19,26,20,27,21,28]
    b=SimpleNamespace(motion={"omni_reference":x},cfg={"omni_lower":lower,"omni_wrists":wrists})
    for frame in (0,7,14):
        for phase in ("pre","motion","post"):
            fm=np.minimum(frame+np.arange(10)*5,T-1);fs=np.minimum(frame+np.arange(10),T-1)
            if phase!="motion":fm[:]=frame;fs[:]=frame
            refs={k:torch.from_numpy(x[ids].copy())[None] for k,ids in (("motion",fm),("smpl",fs))}
            if phase!="motion":refs["motion"][...,29:58]=0
            heading=so3_exp(torch.tensor([0.,0.,.8]))
            robot=so3_exp(torch.tensor([.2,-.1,.7]))
            # Equivalent change of world frame; local reference features unchanged.
            for k in refs:
                for sl in (slice(58,67),slice(139,148)):
                    refs[k][...,sl]=(heading@refs[k][...,sl].reshape(1,10,3,3)).reshape(1,10,9)
            expected=reference_inputs(refs,robot[None],lower,wrists)
            for m in MODALITIES:
                np.testing.assert_allclose(features(b,frame,robot.numpy(),heading.numpy(),phase,m),
                                           expected[m][0].numpy(),atol=1e-5)

def test_onnx_split(tmp_path):
    from blinkmimic.omni_export import export
    pi=OmniPolicy((32,24),(32,24)).eval()
    report=export(pi,tmp_path)
    assert report["passed"]
    assert max(report["errors"][m+"_action"] for m in MODALITIES)<=1e-3

def test_optional_replay_modality():
    from blinkmimic.buffer import ReplayBuffer
    b=ReplayBuffer(4,2,"cpu",29,1)
    b.fields["modality"]=((),torch.long)
    b.data["modality"]=torch.zeros(4,2,dtype=torch.long)
    tr={k:torch.zeros_like(v[0]) for k,v in b.data.items()}
    tr["modality"]=torch.tensor([1,2])
    b.add(tr)
    s,_=b.starts_at(torch.tensor([0,0]),torch.tensor([0,1]),1)
    assert s["modality"].tolist()==[1,2]

def test_imagination_rejects_short_clip_starts():
    from types import SimpleNamespace
    from blinkmimic.omni_runtime import OmniRuntime
    rt=OmniRuntime.__new__(OmniRuntime)
    # Legacy 2B selection would admit invalid frames if fewer than B are valid.
    f=torch.tensor([0,3,3,3])
    rt.g=dict(args=SimpleNamespace(pi_batch=8,horizon=2),dev='cpu',Hh=10,
              motion=SimpleNamespace(clip_end=torch.tensor([3]),clip_id=torch.zeros(4,dtype=torch.long)),
              buffer=SimpleNamespace(sample_starts=lambda *a: ({'f':f},{'f':f[:,None]})))
    rt.starts=rt.imag_ref=None
    rt.prefetch=lambda frames:{'frames':frames.clone()}
    rt.prepare_imagination()
    assert rt.starts[0]['f'].tolist()==[0]*8
    assert rt.imag_ref['frames'].max()==1
    rt.g['buffer'].sample_starts=lambda *a: ({'f':torch.full((4,),3)},{'f':torch.full((4,1),3)})
    import pytest
    with pytest.raises(RuntimeError,match='No replay start'):rt.prepare_imagination()
