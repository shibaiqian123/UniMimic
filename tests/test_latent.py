import copy
import io
import json
import subprocess
import numpy as np
import pytest
import torch
from blinkmimic import networks
from blinkmimic.networks import Policy
from blinkmimic.latent import bound, quantize_bounded, quantize_tokens, check_resume, validate_args
from blinkmimic.latent_contract import validate_v2

def policy(mode):
    return Policy(9,13,4,hist=2,enc_hidden=(16,),dec_hidden=(16,),latent_mode=mode)

def test_all_levels_and_ties():
    u=torch.arange(-16,16,dtype=torch.float32)
    assert torch.equal(quantize_bounded(u)*16,u)
    ties=torch.arange(-15.5,15,1)
    assert torch.equal(quantize_bounded(ties)*16,torch.round(ties))
    assert quantize_tokens(torch.tensor([-100.,100.])).tolist()==[-1.,.9375]
    z=quantize_bounded(bound(torch.linspace(-50,50,20000)))
    assert z.unique().numel()==32
    assert z.min()==-1 and z.max()==.9375

@pytest.mark.parametrize("mode",["continuous64","fsq64"])
def test_split_gradient_and_restore(mode):
    networks.AMP_DTYPE=None
    torch.manual_seed(2)
    p=policy(mode);h=torch.randn(8,2,9);g=torch.randn(8,13)
    z=p.encode_tokens(g)
    assert z.shape==(8,2,32)
    torch.testing.assert_close(p(h,g),p.decode_tokens(h,z),atol=0,rtol=0)
    optim=torch.optim.Adam(p.parameters(),lr=.001)
    target=torch.randn(8,4)
    for _ in range(4):
        optim.zero_grad();loss=(p(h,g)-target).square().mean();loss.backward()
        assert p.enc[0].weight.grad.isfinite().all()
        assert p.enc[0].weight.grad.abs().sum()>0
        optim.step()
    f=io.BytesIO();torch.save(dict(pi=p.state_dict(),opt=optim.state_dict()),f);f.seek(0)
    ck=torch.load(f,weights_only=False)
    new=policy(mode);new.load_state_dict(ck["pi"])
    restored=torch.optim.Adam(new.parameters(),lr=.001);restored.load_state_dict(ck["opt"])
    torch.testing.assert_close(new(h,g),p(h,g),atol=0,rtol=0)
    for model,opt in [(p,optim),(new,restored)]:
        opt.zero_grad();(model(h,g)-target).square().mean().backward();opt.step()
    torch.testing.assert_close(new(h,g),p(h,g),atol=0,rtol=0)

def test_cross_mode_and_skip_refused():
    with pytest.raises(ValueError,match="mode mismatch"):
        policy("fsq64").load_state_dict(policy("continuous64").state_dict())
    with pytest.raises(ValueError):
        check_resume({"latent_mode":"continuous64"},{"latent_mode":"fsq64"})
    with pytest.raises(ValueError):
        Policy(9,13,4,latent_mode="fsq64",ref_skip=(0,[0]*4,[1]*4))
    with pytest.raises(ValueError):
        validate_args({"latent_mode":"fsq64","latent_spec":{}})
    assert validate_args({})["mode"]=="continuous32"
    validate_v2({"latent_spec":validate_args({"latent_mode":"fsq64"})})

def test_continuous32_matches_prechange_source():
    # Repository HEAD is the baseline captured before this implementation.
    source=subprocess.check_output(["git","show","HEAD:src/blinkmimic/networks.py"],text=True)
    ns={"__name__":"blinkmimic._legacy_networks","__package__":"blinkmimic"}
    exec(compile(source,"legacy_networks.py","exec"),ns)
    ns["AMP_DTYPE"]=None;networks.AMP_DTYPE=None
    for skip in (None,(0,[0]*4,[.2]*4)):
        old=ns["Policy"](9,13,4,hist=2,enc_hidden=(16,),dec_hidden=(16,),ref_skip=skip)
        new=Policy(9,13,4,hist=2,enc_hidden=(16,),dec_hidden=(16,),ref_skip=skip)
        new.load_state_dict(old.state_dict(),strict=True)
        h=torch.randn(3,2,9);g=torch.randn(3,13)
        torch.testing.assert_close(old(h,g),new(h,g),atol=0,rtol=0)

@pytest.mark.parametrize("mode",["continuous64","fsq64"])
def test_amp_backward(mode):
    p=policy(mode);networks.AMP_DTYPE=torch.bfloat16
    try:
        a=p(torch.randn(4,2,9),torch.randn(4,13));a.square().sum().backward()
        assert p.enc[0].weight.grad.isfinite().all()
        assert p.enc[0].weight.grad.abs().sum()>0
    finally:networks.AMP_DTYPE=None

def test_onnx_quantizer_boundaries(tmp_path):
    ort=pytest.importorskip("onnxruntime")
    class Quant(torch.nn.Module):
        def forward(self,x):return quantize_bounded(x)
    # Test exact half-integers and immediate neighboring float32 values.
    ties=torch.arange(-15.5,15,1)
    x=torch.cat([ties,torch.nextafter(ties,torch.full_like(ties,float("-inf"))),
                 torch.nextafter(ties,torch.full_like(ties,float("inf")))])
    path=tmp_path/"quant.onnx"
    torch.onnx.export(Quant(),(x,),str(path),input_names=["x"],output_names=["q"],
                     opset_version=18,dynamo=False)
    opts=ort.SessionOptions();opts.intra_op_num_threads=1
    sess=ort.InferenceSession(str(path),opts,providers=["CPUExecutionProvider"])
    np.testing.assert_array_equal(sess.run(None,{"x":x.numpy()})[0],Quant()(x).numpy())

def test_fixed_reference_source_formula():
    import ast
    from pathlib import Path
    path=Path(".cache/fsq_reference/vector_quantize_pytorch/finite_scalar_quantization.py")
    if not path.exists():
        pytest.skip("install vector-quantize-pytorch==1.27.15 in .cache/fsq_reference for source comparison")
    tree=ast.parse(path.read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=="FSQ")
    fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=="bound")
    from functools import partial
    ns=dict(torch=torch,tanh=torch.tanh,atanh=torch.atanh,clamp=torch.clamp,
            partial=partial,identity=lambda x:x,round_ste=lambda x:x+(x.round()-x).detach())
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),str(path),"exec"),ns)
    from types import SimpleNamespace
    ref=SimpleNamespace(_levels=torch.full((32,),32,dtype=torch.int))
    x=torch.linspace(-20,20,64000).reshape(-1,32)
    np.testing.assert_array_equal(ns["bound"](ref,x).numpy(),quantize_bounded(bound(x)).numpy())

@pytest.mark.parametrize("mode",["continuous64","fsq64"])
def test_ema_buffers(mode):
    p=policy(mode)
    avg=torch.optim.swa_utils.AveragedModel(p,multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(.995),use_buffers=True)
    avg.update_parameters(p);avg.update_parameters(p)
    policy(mode).load_state_dict(avg.module.state_dict())

@pytest.mark.parametrize("mode",["continuous64","fsq64"])
def test_compiled_forward_backward(mode):
    p=policy(mode)
    compiled=torch.compile(p,backend="aot_eager",fullgraph=True)
    h=torch.randn(4,2,9);g=torch.randn(4,13)
    compiled(h,g).square().sum().backward()
    assert p.enc[0].weight.grad.abs().sum()>0

def test_existing_walk_checkpoint_and_bundle():
    from pathlib import Path
    from blinkmimic.robot import get_robot
    from blinkmimic.networks import policy_kwargs
    from blinkmimic.deploy.core import Bundle
    path=Path("logs/lafan1_csv_a100_s0/ckpt_003500.pt")
    if not path.exists():pytest.skip("existing checkpoint is local integration fixture")
    ck=torch.load(path,map_location="cpu",weights_only=False)
    args=ck["args"];robot=get_robot("g1")
    widths=lambda k:tuple(map(int,args.get(k,"2048,1024,512,256").split(",")))
    kwargs=dict(hist=10,enc_hidden=widths("enc_hidden"),dec_hidden=widths("dec_hidden"))
    ns={"__name__":"blinkmimic._legacy_networks","__package__":"blinkmimic"}
    source=subprocess.check_output(["git","show","HEAD:src/blinkmimic/networks.py"],text=True)
    exec(compile(source,"legacy_networks.py","exec"),ns)
    ns["AMP_DTYPE"]=None;networks.AMP_DTYPE=None
    old=ns["Policy"](99,730,29,**kwargs,**ns["policy_kwargs"](args,robot))
    new=Policy(99,730,29,**kwargs,**policy_kwargs(args,robot))
    old.load_state_dict(ck["pi"]);new.load_state_dict(ck["pi"])
    bundle=Bundle("deploy_artifacts/final_3500_walk1")
    h=torch.zeros(2,10,99)
    g=torch.from_numpy(bundle.motion["goal"][[0,30]])
    with torch.no_grad():
        torch.testing.assert_close(old(h,g),new(h,g),atol=0,rtol=0)

@pytest.mark.parametrize("mode",["continuous64","fsq64"])
@pytest.mark.skipif(not torch.cuda.is_available(),reason="CUDA graph integration test")
def test_cuda_graph_amp_ste(mode):
    p=policy(mode).cuda()
    h=torch.randn(4,2,9,device="cuda");g=torch.randn(4,13,device="cuda")
    networks.AMP_DTYPE=torch.bfloat16
    stream=torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    def step():
        p.zero_grad(set_to_none=True)
        loss=p(h,g).square().sum()
        loss.backward()
        return loss
    try:
        with torch.cuda.stream(stream):
            for _ in range(3):step()
        torch.cuda.current_stream().wait_stream(stream)
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):loss=step()
        graph.replay();torch.cuda.synchronize()
        assert torch.isfinite(loss)
        assert p.enc[0].weight.grad.isfinite().all()
        assert p.enc[0].weight.grad.abs().sum()>0
    finally:networks.AMP_DTYPE=None
