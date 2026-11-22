import json
import queue
from types import SimpleNamespace
import numpy as np
from blinkmimic.deploy.runtime_io import RuntimeLog,cpu_list,affinity_plan

def test_async_log_preserves_arrays_order_and_flushes(tmp_path):
    log=RuntimeLog(tmp_path/"runner.jsonl")
    assert log.write(dict(event="start"))
    assert log.write(dict(event="state",q=np.arange(29,dtype=float)))
    log.status(tmp_path/"status.json",dict(state="HOLD"))
    assert log.write(dict(event="end"))
    log.close()
    rows=[json.loads(line) for line in (tmp_path/"runner.jsonl").read_text().splitlines()]
    assert [r["event"] for r in rows]==["start","state","end"]
    assert rows[1]["q"]==list(range(29))
    assert json.loads((tmp_path/"status.json").read_text())["state"]=="HOLD"
    assert json.loads((tmp_path/"logging_report.json").read_text())["complete"]

def test_queue_full_is_nonblocking_and_counted():
    log=RuntimeLog.__new__(RuntimeLog)
    log.q=queue.Queue(1);log.dropped=0
    log.process=SimpleNamespace(is_alive=lambda:True)
    assert log.write({"n":1})
    assert not log.write({"n":2})
    assert log.dropped==1

def test_dead_writer_is_counted():
    log=RuntimeLog.__new__(RuntimeLog);log.dropped=0
    log.process=SimpleNamespace(is_alive=lambda:False)
    assert not log.write({"n":1})
    assert log.dropped==1

def test_cpu_list():
    assert cpu_list("2,4")=={2,4}
    assert cpu_list(None) is None

def test_policy_cpu_timing_does_not_change_actions():
    from test_deploy_runtime import bundle
    from blinkmimic.deploy.core import DeployCore
    c=DeployCore(bundle(),lambda h,g:np.ones(29)*.25)
    c.align(np.eye(3))
    r=c.step(np.zeros(29),np.zeros(29),np.zeros(3),np.eye(3))
    np.testing.assert_array_equal(r["action"],np.ones(29)*.25)
    assert r["inference_thread_cpu_ms"]>=0
    assert r["inference_ms"]>=0

def test_affinity_separates_dds_endpoints_when_cores_available():
    plan=affinity_plan()
    if "bridge" in plan:
        critical=[set(plan[n]) for n in ("physics","sim_bridge","runner","bridge")]
        for i,cpus in enumerate(critical):
            assert not cpus.intersection(plan["viewer"])
            for other in critical[i+1:]:assert not cpus.intersection(other)
        assert len(plan["sim_bridge"])>=2
        assert len(plan["bridge"])>=2
        assert len(plan["runner"])==2


def test_affinity_cpu_pools_and_smt_exclusion(monkeypatch):
    import re
    from pathlib import Path
    import blinkmimic.deploy.runtime_io as runtime
    original=Path.read_text
    for count in (4, 8, 12, 64):
        monkeypatch.setattr(runtime.os, "sched_getaffinity", lambda pid, n=count: set(range(n*2)))
        def topology(path, *args, n=count, **kwargs):
            match=re.search(r"/cpu(\d+)/topology/(physical_package_id|core_id)$", str(path))
            if not match:
                return original(path, *args, **kwargs)
            return "0" if match[2]=="physical_package_id" else str(int(match[1]) % n)
        monkeypatch.setattr(Path, "read_text", topology)
        plan=runtime.affinity_plan()
        roles=("physics","runner","sim_bridge")+(("bridge",) if "bridge" in plan else ())
        used=set()
        for role in roles:
            physical={c % count for c in plan[role]}
            assert not used & physical
            used |= physical
        assert not used & {c % count for c in plan["viewer"]}
        if count>=8:
            assert len(plan["sim_bridge"])==min(8,(count-4)//2)
            assert len(plan["bridge"])==len(plan["sim_bridge"])
            assert len(plan["runner"])==2
