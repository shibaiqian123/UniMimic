import copy
from pathlib import Path
from types import SimpleNamespace,MethodType
import subprocess
import numpy as np
import pytest
import torch
from blinkmimic import networks
from blinkmimic.networks import Policy,policy_kwargs
from blinkmimic.motion import MotionLib
from blinkmimic.policy_inputs import future_orientation,policy_action,diagnostic_orientation,matrix_from_6d
from blinkmimic.rotations import so3_exp,mat_to_6d,mat_to_quat
from blinkmimic.deploy.core import reference_orientation_window,rotation,DeployCore,yaw_rotation
from blinkmimic.latent import check_resume

def motion():
    angles=torch.tensor([[.03*i,-.02*i,.14*i] for i in range(9)])
    m=SimpleNamespace(root_R=so3_exp(angles),clip_end=torch.tensor([3,8]),
                      clip_id=torch.tensor([0]*4+[1]*5))
    m.future_frames=MethodType(MotionLib.future_frames,m)
    return m

def policy(mode="fsq64"):
    return Policy(99,730,29,enc_hidden=(32,16),dec_hidden=(32,16),
                  latent_mode=mode,observation_mode="proprio93")

def test_window_parity_and_clip_clamping():
    m=motion();frames=torch.tensor([0,2,3,4,7,8])
    R=so3_exp(torch.tensor([[.2,-.3,.8]]).expand(len(frames),3))
    H=torch.from_numpy(yaw_rotation(.7)).float()
    result=future_orientation(m,frames,R,heading=H)
    for i,f in enumerate(frames.tolist()):
        start,end=(0,3) if f<4 else (4,8)
        actual=reference_orientation_window(m.root_R[start:end+1].numpy(),f-start,R[i].numpy(),H.numpy(),"motion")
        np.testing.assert_allclose(result[i].numpy(),actual,atol=1e-6)
    assert m.future_frames(torch.tensor([2]),10).tolist()==[[3]*10]
    assert not torch.allclose(result[0,0],result[0,1])

def test_quaternion_and_static_reference():
    m=motion()
    q=mat_to_quat(so3_exp(torch.tensor([.2,-.3,.8]))).numpy()
    np.testing.assert_array_equal(rotation(q),rotation(-q))
    H=yaw_rotation(.7)
    for phase,frame in (("pre",0),("post",8)):
        w=reference_orientation_window(m.root_R.numpy(),frame,rotation(q),H,phase)
        np.testing.assert_allclose(w,np.repeat(w[:1],10,0),atol=1e-7)
        changed=reference_orientation_window(m.root_R.numpy(),frame,yaw_rotation(.2)@rotation(q),H,phase)
        assert not np.allclose(w,changed)

@pytest.mark.parametrize("mode",["continuous64","fsq64"])
def test_encoder_decoder_dependency(mode):
    networks.AMP_DTYPE=None
    p=policy(mode);h=torch.randn(2,10,99);g=torch.randn(2,730)
    m=motion();f=torch.tensor([0,4])
    ori=future_orientation(m,f,m.root_R[f])
    z=p.encode_tokens(g,ori)
    assert p.obs_norm.mean.numel()==93 and p.goal_norm.mean.numel()==790
    assert p.dec[0].in_features==994 and p.enc[0].in_features==790
    a=policy_action(p,h,g,ori)
    changed=h.clone();changed[...,93:]+=1000
    torch.testing.assert_close(a,policy_action(p,changed,g,ori),rtol=0,atol=0)
    torch.testing.assert_close(a,p.decode_tokens(h[...,:93],z),rtol=0,atol=0)
    assert not torch.allclose(p.enc_raw(g,ori),p.enc_raw(g,ori.flip(1)))
    with pytest.raises(ValueError,match="explicit"):p.encode_tokens(g)
    with pytest.raises(ValueError):p.encode_tokens(g,ori[:,0])
    with pytest.raises(RuntimeError):p.decode_tokens(h,z)

@pytest.mark.parametrize("mode",["continuous64","fsq64"])
def test_robot_rotation_gradient_and_compilation(mode):
    p=policy(mode);m=motion();f=torch.tensor([0,4])
    h=torch.randn(2,10,99);g=torch.randn(2,730)
    angle=torch.tensor([[.2,.3,.4],[.1,.2,.3]],requires_grad=True)
    ori=future_orientation(m,f,so3_exp(angle))
    policy_action(p,h,g,ori).square().mean().backward()
    assert angle.grad.isfinite().all() and angle.grad.abs().sum()>0
    assert p.enc[0].weight.grad.abs().sum()>0
    compiled=torch.compile(p,backend="aot_eager",fullgraph=True)
    p.zero_grad()
    networks.AMP_DTYPE=torch.bfloat16
    try:
        compiled(h[...,:93],g,orientation6d_future=ori.detach()).square().mean().backward()
        assert p.enc[0].weight.grad.isfinite().all() and p.enc[0].weight.grad.abs().sum()>0
    finally:networks.AMP_DTYPE=None

def test_diagnostics_are_not_ten_identities():
    p=policy();m=motion();f=torch.tensor([0,4])
    z=diagnostic_orientation(p,m,f)
    torch.testing.assert_close(z,future_orientation(m,f,m.root_R[f]))
    assert not torch.allclose(z,z[:,0:1].expand_as(z))

def test_checkpoint_contract():
    old=Policy(99,730,29,enc_hidden=(32,16),dec_hidden=(32,16),latent_mode="fsq64")
    new=policy()
    with pytest.raises(ValueError):new.load_state_dict(old.state_dict())
    with pytest.raises(ValueError):old.load_state_dict(new.state_dict())
    bad=copy.deepcopy(new.state_dict());bad["_observation_kind"].fill_(1)
    with pytest.raises(ValueError):new.load_state_dict(bad)
    with pytest.raises(ValueError):check_resume({"latent_mode":"fsq64"},{"latent_mode":"fsq64","observation_mode":"proprio93"})
    with pytest.raises(ValueError):Policy(99,730,29,observation_mode="proprio93")
    with pytest.raises(ValueError):Policy(99,730,29,hist=8,latent_mode="fsq64",observation_mode="proprio93")

def test_core_history_and_reference_window():
    m=motion();seen=[]
    b=SimpleNamespace(T=9,q0=np.zeros(29),scale=np.ones(29),cfg={"obs_dim":93},
         motion={"root_R":m.root_R.numpy(),"goal":np.zeros((9,730)),"static_goal":np.zeros((2,730))})
    def callback(h,g,ori):
        seen.append((h.copy(),ori.copy()))
        return np.arange(29)*.001
    c=DeployCore(b,callback);R=yaw_rotation(.6)
    c.align(R);c.step(np.zeros(29),np.zeros(29),np.zeros(3),R)
    assert seen[-1][0].shape==(10,93)
    np.testing.assert_allclose(seen[-1][0],np.repeat(seen[-1][0][:1],10,0))
    c.start_motion();c.step(np.ones(29)*.01,np.zeros(29),np.zeros(3),R)
    np.testing.assert_allclose(seen[-1][1],reference_orientation_window(b.motion["root_R"],0,R,c.heading,"motion"))
    np.testing.assert_allclose(seen[-1][0][-1,-29:],np.arange(29)*.001)
    c.phase="post";c.frame=8;c.step(np.zeros(29),np.zeros(29),np.zeros(3),R)
    np.testing.assert_allclose(seen[-1][1],np.repeat(seen[-1][1][:1],10,0))

def test_legacy_fsq_checkpoint_exact():
    path=Path("logs/lafan_fsq64_a100_s0/ckpt_003500.pt")
    if not path.exists():pytest.skip("local legacy99 FSQ integration checkpoint absent")
    from blinkmimic.robot import get_robot
    ck=torch.load(path,map_location="cpu",weights_only=False);cfg=ck["args"]
    widths=lambda k:tuple(map(int,cfg[k].split(",")))
    ns={"__name__":"blinkmimic._legacy_networks","__package__":"blinkmimic"}
    exec(compile(subprocess.check_output(["git","show","HEAD:src/blinkmimic/networks.py"],text=True),"legacy","exec"),ns)
    kw=dict(hist=10,enc_hidden=widths("enc_hidden"),dec_hidden=widths("dec_hidden"))
    old=ns["Policy"](99,730,29,**kw,**ns["policy_kwargs"](cfg,get_robot("g1")))
    new=Policy(99,730,29,**kw,**policy_kwargs(cfg,get_robot("g1")))
    old.load_state_dict(ck["pi"]);new.load_state_dict(ck["pi"])
    ns["AMP_DTYPE"]=None;networks.AMP_DTYPE=None
    h=torch.randn(2,10,99);g=torch.randn(2,730)
    with torch.no_grad():torch.testing.assert_close(old(h,g),new(h,g),atol=0,rtol=0)

@pytest.mark.parametrize("mode",["continuous64","fsq64"])
def test_three_onnx_dynamic_batch(tmp_path,mode):
    import sys
    sys.path.insert(0,str(Path("tools").resolve()))
    from export_deploy import ExportPolicy,ExportEncoder,ExportDecoder
    import onnxruntime as ort
    p=policy(mode).eval();networks.AMP_DTYPE=None
    h=torch.randn(3,10,93);g=torch.randn(3,730);ori=torch.randn(3,10,6)
    z=p.encode_tokens(g,ori).detach()
    sessions={}
    options=ort.SessionOptions();options.intra_op_num_threads=1
    for name,module,inputs,names,out in (
        ("policy",ExportPolicy(p),(h[:1],g[:1],ori[:1]),["obs_history","goal","orientation6d_future"],"actions"),
        ("encoder",ExportEncoder(p),(g[:1],ori[:1]),["goal","orientation6d_future"],"tokens"),
        ("decoder",ExportDecoder(p),(h[:1],z[:1]),["obs_history","tokens"],"actions")):
        path=tmp_path/(name+".onnx")
        torch.onnx.export(module,inputs,str(path),input_names=names,output_names=[out],
                         dynamic_axes={k:{0:"batch"} for k in names+[out]},opset_version=18,dynamo=False)
        sessions[name]=ort.InferenceSession(str(path),options,providers=["CPUExecutionProvider"])
    a=sessions["policy"].run(None,{"obs_history":h.numpy(),"goal":g.numpy(),"orientation6d_future":ori.numpy()})[0]
    z=sessions["encoder"].run(None,{"goal":g.numpy(),"orientation6d_future":ori.numpy()})[0]
    split=sessions["decoder"].run(None,{"obs_history":h.numpy(),"tokens":z})[0]
    ref=p(h,g,orientation6d_future=ori).detach().numpy()
    np.testing.assert_allclose(a,ref,atol=1e-3,rtol=0)
    np.testing.assert_allclose(split,ref,atol=1e-3,rtol=0)
    assert {x.name for x in sessions["decoder"].get_inputs()}=={"obs_history","tokens"}

def test_baked_interface_reconstructs_future():
    m=motion();f=torch.tensor([0,4])
    robot=so3_exp(torch.tensor([[.3,-.2,.7],[-.2,.1,.8]]))
    current=mat_to_6d(robot.transpose(-1,-2)@m.root_R[f])
    relative=m.root_R[f].transpose(-1,-2)[:,None]@m.root_R[m.future_frames(f,10)]
    reconstructed=mat_to_6d(matrix_from_6d(current)[:,None]@relative)
    torch.testing.assert_close(reconstructed,future_orientation(m,f,robot),atol=1e-6,rtol=0)
