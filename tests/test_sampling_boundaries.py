from types import SimpleNamespace
import pytest
import torch
from blinkmimic.env import TrackingSim


def sampler(device, starts, ends):
    starts=torch.tensor(starts,device=device)
    ends=torch.tensor(ends,device=device)
    m=SimpleNamespace(num_bins=len(starts),bin_start=starts,
                      clip_end=ends,bin_clip=torch.arange(len(starts),device=device))
    return SimpleNamespace(motion=m,bin_G=torch.zeros(len(starts),device=device),
        G_max=1.,bin_w_avg=torch.ones(len(starts),device=device),fail_reset=0.,
        focus_static=False,focus_q=None,focus_share=0.,device=device)


def test_singleton_final_bin_never_samples_terminal():
    # Actual BONES1000 last-bin indices from 342231 total reference frames.
    e=sampler("cpu",[342230],[342230])
    frames,bins=TrackingSim.sample_frames(e,4096)
    assert torch.all(frames==342229)
    assert torch.all(frames+1<=e.motion.clip_end[bins])


def test_internal_and_final_clip_boundaries(monkeypatch):
    e=sampler("cpu",[0,50,100,101,151,201],[100,100,100,201,201,201])
    monkeypatch.setattr(torch,"multinomial",lambda probs,n,replacement:torch.arange(6))
    for u in (0.,.5,1.-2**-24):
        monkeypatch.setattr(torch,"rand",lambda n,device:torch.full((n,),u,device=device))
        f,b=TrackingSim.sample_frames(e,6)
        assert torch.all(f+1<=e.motion.clip_end[b])
        assert f[2]==99 and f[5]==200
        # Nonempty bins are numerically identical to the original formula.
        old_start=e.motion.bin_start[b]
        old_end=torch.minimum(old_start+50,e.motion.clip_end[b])
        valid=old_start<old_end
        expected=old_start+(torch.full((6,),u)*(old_end-old_start).float()).long()
        assert torch.equal(f[valid],expected[valid])


@pytest.mark.skipif(not torch.cuda.is_available(),reason="CUDA required")
def test_singleton_reset_cuda_graph():
    e=sampler("cuda",[342230],[342230])
    stream=torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            TrackingSim.sample_frames(e,4096,gate=torch.tensor(True,device="cuda"))
    torch.cuda.current_stream().wait_stream(stream)
    graph=torch.cuda.CUDAGraph()
    gate=torch.tensor(True,device="cuda")
    with torch.cuda.graph(graph):
        f,b=TrackingSim.sample_frames(e,4096,gate=gate)
    for _ in range(64):
        graph.replay()
    torch.cuda.synchronize()
    assert torch.all(f==342229)
    assert torch.all(f+1<=e.motion.clip_end[b])
