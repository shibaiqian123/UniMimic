"""CPU test of GPU rollout bookkeeping: completed replicas must not accumulate extra rewards."""
from types import SimpleNamespace
import numpy as np
import torch
from blinkmimic.deploy.warp_runner import WarpRunner

def test_finished_replica_is_frozen_and_not_double_counted(monkeypatch):
    monkeypatch.setattr(torch.cuda,"synchronize",lambda *_:None)
    class Env:
        def __init__(self):
            self.done_eval=torch.zeros(2,dtype=torch.bool)
            self.failed_eval=torch.zeros(2,dtype=torch.bool)
            self.f=torch.zeros(2,dtype=torch.long)
            self.sim=SimpleNamespace(data=SimpleNamespace(qpos=torch.zeros(2,36)))
            self.calls=0
        def observe(self):return torch.zeros(2,99)
        def step(self,a):
            self.calls+=1
            self.sim.data.qpos+=1
            self.f+=1
            self.done_eval[0]=True;self.failed_eval[0]=True
            if self.calls==2:self.done_eval[1]=True
            return {"r":torch.ones(2)}
        def state(self):return (torch.zeros(2,3),None,None,None,torch.zeros(2,29),None)
    r=WarpRunner.__new__(WarpRunner)
    r.env=Env();r.b=SimpleNamespace(T=3);r.device=torch.device("cpu")
    r.pi=lambda h,g:torch.zeros(2,29)
    r.motion=SimpleNamespace(goal=lambda f,k:torch.zeros(2,730))
    r.focus=0;r.record=True;r.records=[];r.history=None;r.returns=torch.zeros(2)
    r.steps=torch.zeros(2,dtype=torch.long);r.final_qpos=torch.zeros(2,36)
    r.final_frame=torch.zeros(2,dtype=torch.long);r.ticks=0;r.finished=False;r.tick_ms=[]
    assert r.tick()
    assert not r.tick()
    assert r.returns.tolist()==[1,2] and r.steps.tolist()==[1,2]
    assert len(r.records)==1 and r.records[0]["frame"]==0
    q,f,done,failed=r.snapshot()
    np.testing.assert_array_equal(q,np.ones(36))
    assert (f,done,failed)==(1,True,True)
