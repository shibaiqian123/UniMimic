import copy
from types import SimpleNamespace
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import pytest
from blinkmimic.distributed import Distributed,split_global,optimizer_step,update_bins
from blinkmimic.networks import RunningNorm

def worker(rank,path):
    dist.init_process_group("gloo",init_method="file://"+path,rank=rank,world_size=2)
    ctx=Distributed("cpu",test=True)
    try:
        torch.manual_seed(rank+3)
        model=torch.nn.Linear(3,2)
        ctx.broadcast_models(model)
        reference=copy.deepcopy(model)
        x=torch.arange(18,dtype=torch.float32).reshape(6,3)/18
        local=x[:2] if rank==0 else x[2:]
        loss=model(local).square().sum()*2/6
        loss.backward()
        opt=torch.optim.Adam(model.parameters(),lr=.01)
        refopt=torch.optim.Adam(reference.parameters(),lr=.01)
        reference(x).square().sum().div(6).backward()
        torch.nn.utils.clip_grad_norm_(reference.parameters(),1.)
        refopt.step()
        _,finite=optimizer_step(ctx,model,opt)
        assert finite
        for p,r in zip(model.parameters(),reference.parameters()):
            torch.testing.assert_close(p,r,atol=1e-6,rtol=1e-5)
        norm=RunningNorm(3);refnorm=RunningNorm(3)
        for scale in [1.,3.]:
            ctx.moments([norm],[local*scale]);refnorm.update(x*scale)
            for key in ("mean","var","count"):
                torch.testing.assert_close(getattr(norm,key),getattr(refnorm,key),atol=1e-6,rtol=1e-5)
        norms=[RunningNorm(93),RunningNorm(790)]
        refs=[RunningNorm(93),RunningNorm(790)]
        data=[torch.arange(6*d,dtype=torch.float32).reshape(6,d)/100 for d in (93,790)]
        slices=[x[:2] if rank==0 else x[2:] for x in data]
        ctx.moments(norms,slices)
        for norm,ref,values in zip(norms,refs,data):
            ref.update(values)
            for key in ("mean","var","count"):
                torch.testing.assert_close(getattr(norm,key),getattr(ref,key),atol=1e-4,rtol=1e-5)
        before=copy.deepcopy(model.state_dict());osd=copy.deepcopy(opt.state_dict())
        for p in model.parameters():p.grad=torch.ones_like(p)
        if rank==1:next(model.parameters()).grad.fill_(float("nan"))
        _,finite=optimizer_step(ctx,model,opt)
        assert not finite
        for k,v in model.state_dict().items():torch.testing.assert_close(v,before[k],atol=0,rtol=0)
        for k,v in opt.state_dict()["state"].items():
            for key,value in v.items():torch.testing.assert_close(value,osd["state"][k][key],atol=0,rtol=0)
        # Masked WM objective must retain global batch denominator, not valid-frame denominator.
        model.zero_grad()
        mask=torch.tensor([1.,0.,1.,1.,0.,0.])
        part=slice(rank*3,(rank+1)*3)
        (model(x[part]).square().mean(-1)*mask[part]).sum().div(3).backward()
        ref=copy.deepcopy(model);ref.zero_grad()
        (ref(x).square().mean(-1)*mask).sum().div(6).backward()
        ctx.gradients(model)
        for p,r in zip(model.parameters(),ref.parameters()):torch.testing.assert_close(p.grad,r.grad)
        states=ctx.gather_objects(dict(rank=rank,rng=torch.get_rng_state()))
        assert [s["rank"] for s in states]==[0,1]
        focus=torch.tensor([.2,.8]) if rank==0 else torch.zeros(2)
        ctx.broadcast(focus);torch.testing.assert_close(focus,torch.tensor([.2,.8]))
        avg=torch.optim.swa_utils.AveragedModel(model)
        avg.update_parameters(model)
        ctx.broadcast_models(avg)
        env=SimpleNamespace(bin_G=torch.zeros(2),bin_fail=torch.zeros(2),bin_w_avg=torch.ones(2)/2,G_max=1.,fail_reset=.3)
        increments=torch.tensor([[1.,0.],[2.,0.],[1.,0.],[1.,0.]]) if rank==0 else torch.tensor([[1.,1.],[4.,3.],[1.,1.],[0.,1.]])
        ctx.sum(increments)
        update_bins(env,(*increments,increments[0].sum()))
        torch.testing.assert_close(env.bin_G,torch.tensor([.6,.6]))
        # Next empty step must not add cumulative counts or decay again.
        keep=copy.deepcopy(env)
        update_bins(env,(*(torch.zeros(2) for _ in range(4)),torch.tensor(0.)))
        torch.testing.assert_close(env.bin_G,keep.bin_G)
        torch.testing.assert_close(env.bin_fail,keep.bin_fail)
        torch.testing.assert_close(env.bin_w_avg,keep.bin_w_avg)
    finally:dist.destroy_process_group()

def test_two_cpu_ranks(tmp_path):
    mp.spawn(worker,args=(str(tmp_path/"init"),),nprocs=2,join=True)

def test_global_sizes():
    a=SimpleNamespace(num_envs=4096,wm_batch=2048,pi_batch=2048,horizon=32,critic_minibatches=4)
    saved=split_global(a,2)
    assert a.num_envs==2048 and a.wm_batch==1024 and a.pi_batch==1024
    assert saved["num_envs"]==4096 and saved["distributed"]["lr_scale"]==1.
    with pytest.raises(ValueError):
        split_global(SimpleNamespace(num_envs=4095,wm_batch=2048,pi_batch=2048,horizon=32,critic_minibatches=4),2)

def test_pin(monkeypatch):
    from blinkmimic.distributed_bootstrap import pin_rank
    monkeypatch.setenv("WORLD_SIZE","2");monkeypatch.setenv("LOCAL_WORLD_SIZE","2")
    monkeypatch.setenv("LOCAL_RANK","1");monkeypatch.setenv("CUDA_VISIBLE_DEVICES","2,5")
    assert pin_rank()
    import os
    assert os.environ["CUDA_VISIBLE_DEVICES"]=="5"
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES","0,0")
    with pytest.raises(ValueError):pin_rank()

def test_world_size_resume_contract():
    from blinkmimic.distributed import check_resume_world
    check_resume_world({},1)
    with pytest.raises(ValueError,match="world_size"):
        check_resume_world({},2)
    with pytest.raises(ValueError,match="world_size"):
        check_resume_world({"distributed":{"world_size":2,"rank_states":[{},{}]}},1)
    with pytest.raises(ValueError,match="incomplete"):
        check_resume_world({"distributed":{"world_size":2,"rank_states":[{}]}},2)
    check_resume_world({"distributed":{"world_size":2,"rank_states":[{},{}]}},2)
