"""No DDS, no policy and no motors: measure Linux sleep/wakeup timing."""
import argparse
import gc
import json
import os
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"src"))
from blinkmimic.deploy.scheduling import SleepMonitor,scheduler_snapshot

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seconds",type=float,default=60)
    ap.add_argument("--cpu",type=int,required=True)
    ap.add_argument("--rt-priority",type=int,default=0)
    ap.add_argument("--out",type=Path,required=True)
    a=ap.parse_args()
    if a.seconds<=0 or not 0<=a.rt_priority<=10:ap.error("positive seconds; priority 0..10")
    os.sched_setaffinity(0,{a.cpu})
    if a.rt_priority:os.sched_setscheduler(0,os.SCHED_RR,os.sched_param(a.rt_priority))
    start_info=scheduler_snapshot()
    print(json.dumps(start_info),flush=True)
    gc.collect();was_enabled=gc.isenabled();gc.disable()
    m=SleepMonitor();rows=[];start=time.monotonic();deadline=start
    try:
        while time.monotonic()-start<a.seconds:
            deadline=max(deadline+.02,time.monotonic())
            rows.append(m.sleep_until(deadline))
    finally:
        if was_enabled:gc.enable()
    report=dict(scope="sleep only; no DDS or motor commands",scheduler=start_info,
                samples=len(rows),max_overshoot_ms=max(r["overshoot_ms"] for r in rows),
                over20ms=sum(r["overshoot_ms"]>20 for r in rows),records=rows)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!="records"}),flush=True)
if __name__=="__main__":main()
