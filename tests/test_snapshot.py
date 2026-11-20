import numpy as np
from blinkmimic.deploy.snapshot import read_snapshot

def test_collision_retried_not_consumed():
    buf=np.zeros(1,dtype=[("n_rx","i8")])
    class Bridge:
        calls=0
        def read_snap(self,b):
            self.calls+=1
            b["n_rx"]=99  # failed copy contents must not be trusted
            return self.calls==3
    b=Bridge();clock=[0.]
    def sleep(dt):clock[0]+=dt
    assert read_snapshot(b,buf,clock=lambda:clock[0],sleep=sleep)
    assert b.calls==3

def test_failed_copy_bounded():
    buf=np.zeros(1,dtype=[("n_rx","i8")]);buf["n_rx"]=99
    class Bridge:
        def read_snap(self,b):return False
    clock=[0.]
    def sleep(dt):clock[0]+=dt
    assert not read_snapshot(Bridge(),buf,clock=lambda:clock[0],sleep=sleep)
    assert .002<=clock[0]<.003

def test_no_initial_sample():
    buf=np.zeros(1,dtype=[("n_rx","i8")])
    class Bridge:
        def read_snap(self,b):return True
    assert not read_snapshot(Bridge(),buf)
