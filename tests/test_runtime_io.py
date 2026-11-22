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
        assert len(plan["sim_bridge"])==2
