import copy
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from blinkmimic.omni import OmniPolicy, DIMS, auxiliary_total
from blinkmimic.distributed import Distributed
from blinkmimic.training_diagnostics import gradient_contributions
from blinkmimic.buffer import ReplayBuffer


def losses(p, h, x, modes):
    objective = p(h, x, modes).square().mean()
    total, n = auxiliary_total(p.auxiliary(x))
    return objective, total/n


def fixture():
    torch.manual_seed(81)
    p = OmniPolicy((16, 12), (16, 12))
    h = torch.randn(6, 10, 93)
    x = {m: torch.randn(6, d) for m, d in DIMS.items()}
    return p, h, x, torch.arange(6) % 3


def test_decomposition_preserves_training_gradient():
    p,h,x,m = fixture()
    reference = copy.deepcopy(p)
    policy,aux = losses(p,h,x,m)
    # Existing gradient buffers, including capture-owned ones, must not be replaced.
    for v in p.parameters(): v.grad = torch.full_like(v, 7.)
    pointers = [v.grad.data_ptr() for v in p.parameters()]
    result = gradient_contributions(p,policy,aux,Distributed("cpu"))
    assert all(torch.all(v.grad == 7) for v in p.parameters())
    assert pointers == [v.grad.data_ptr() for v in p.parameters()]
    assert result["gradient/action_decoder/auxiliary_norm"] == 0
    assert result["gradient/action_decoder/policy_norm"] > 0
    assert result["gradient/reconstruction/policy_norm"] == 0
    assert result["gradient/reconstruction/auxiliary_norm"] > 0
    for m in p.encoders:
        assert result[f"gradient/encoder_{m}/policy_norm"] > 0
        assert result[f"gradient/encoder_{m}/auxiliary_norm"] > 0
    p.zero_grad()
    (policy+aux).backward()
    a,b=losses(reference,h,x,torch.arange(6)%3)
    (a+b).backward()
    for v,w in zip(p.parameters(), reference.parameters()):
        torch.testing.assert_close(v.grad,w.grad,rtol=0,atol=0)


def worker(rank, path):
    torch.set_num_threads(1)
    dist.init_process_group("gloo",init_method="file://"+path,rank=rank,world_size=2)
    try:
        p,h,x,m=fixture()
        sl=slice(rank*3,(rank+1)*3)
        a,b=losses(p,h[sl],{k:v[sl] for k,v in x.items()},m[sl])
        actual=gradient_contributions(p,a,b,Distributed("cpu"))
        a,b=losses(p,h,x,m)
        local=Distributed("cpu");local.world=1
        expected=gradient_contributions(p,a,b,local)
        for k in actual:
            torch.testing.assert_close(torch.tensor(actual[k]),torch.tensor(expected[k]),atol=1e-5,rtol=2e-4)
    finally:
        dist.destroy_process_group()


def test_global_gradient_vectors(tmp_path):
    mp.spawn(worker,args=(str(tmp_path/"rdzv"),),nprocs=2,join=True)


def test_replay_optional_fields_preserve_default_sampling():
    b=ReplayBuffer(8,2,"cpu",29,1)
    for _ in range(8):
        b.add({k:torch.zeros_like(v[0]) for k,v in b.data.items()})
    torch.manual_seed(1);old=b.sample_windows(3,4)
    torch.manual_seed(1);new=b.sample_windows(3,4,extra_keys=("f","r","done"))
    assert set(new)-set(old)=={"f","r","done"}
    for k in old:torch.testing.assert_close(old[k],new[k])
