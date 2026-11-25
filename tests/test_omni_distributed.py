"""Global-batch equivalence for the new independent normalizers and auxiliary gradients."""
import copy
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from blinkmimic.omni import OmniPolicy,DIMS,MODALITIES,auxiliary_total
from blinkmimic.distributed import Distributed,optimizer_step

def worker(rank,path):
    torch.set_num_threads(1)
    dist.init_process_group("gloo",init_method="file://"+path,rank=rank,world_size=2)
    try:
        ctx=Distributed("cpu",test=True)
        torch.manual_seed(11)
        p=OmniPolicy((16,12),(16,12))
        reference=copy.deepcopy(p)
        inputs={m:torch.randn(6,d) for m,d in DIMS.items()}
        h=torch.randn(6,10,93)
        part=slice(rank*3,(rank+1)*3)
        local={m:x[part] for m,x in inputs.items()}
        ctx.moments([p.obs_norm]+list(p.input_norms.values()),[h[part]]+[local[m] for m in MODALITIES])
        reference.update_norms(h,inputs)
        for k,v in p.state_dict().items():
            torch.testing.assert_close(v,reference.state_dict()[k],atol=1e-5,rtol=1e-5)
        mask=torch.tensor([1.,0.,0.,1.,1.,1.])
        count=mask[part].sum();ctx.sum(count)
        total,_=auxiliary_total(p.auxiliary(local),mask[part])
        (total*ctx.world/count).backward()
        full,n=auxiliary_total(reference.auxiliary(inputs),mask)
        (full/n).backward()
        ctx.gradients(p)
        for x,y in zip(p.parameters(),reference.parameters()):
            if y.grad is None:
                assert x.grad.abs().sum()==0
            else:torch.testing.assert_close(x.grad,y.grad,atol=2e-5,rtol=1e-4)
        before=copy.deepcopy(p.state_dict())
        if rank==1:next(p.encoders["smpl"].parameters()).grad.fill_(float("nan"))
        opt=torch.optim.Adam(p.parameters(),lr=.001)
        _,ok=optimizer_step(ctx,p,opt)
        assert not ok
        for k,v in p.state_dict().items():torch.testing.assert_close(v,before[k],rtol=0,atol=0)
    finally:dist.destroy_process_group()

def test_omni_two_rank(tmp_path):
    mp.spawn(worker,args=(str(tmp_path/"rendezvous"),),nprocs=2,join=True)
