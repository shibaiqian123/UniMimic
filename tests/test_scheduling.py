import os
from blinkmimic.deploy.scheduling import SleepMonitor,scheduler_snapshot

def test_actual_scheduler_is_reported():
    s=scheduler_snapshot()
    assert s["priority"]==os.sched_getparam(0).sched_priority
    assert s["cpus"]==sorted(os.sched_getaffinity(0))
    assert s["tid"] in [t["tid"] for t in s["threads"]]

def test_sleep_evidence_separates_requested_time(monkeypatch):
    from blinkmimic.deploy import scheduling as mod
    m=SleepMonitor()
    times=iter([10.,10.080])
    cpus=iter([1.,1.0005])
    monkeypatch.setattr(mod.time,"monotonic",lambda:next(times))
    monkeypatch.setattr(mod.time,"thread_time",lambda:next(cpus))
    asked=[]
    monkeypatch.setattr(mod.time,"sleep",asked.append)
    r=m.sleep_until(10.020)
    assert abs(asked[0]-.020)<1e-9
    assert abs(r["overshoot_ms"]-60)<1e-6
    assert abs(r["thread_cpu_ms"]-.5)<1e-6
    assert m.previous is r

def test_missing_procfs_is_explicit(monkeypatch):
    from blinkmimic.deploy import scheduling as mod
    monkeypatch.setattr(mod,"_text",lambda path:None)
    m=SleepMonitor()
    r=m.sleep_until(0)
    assert r["schedstat_delta"] is None
    assert r["requested_ms"]==0
