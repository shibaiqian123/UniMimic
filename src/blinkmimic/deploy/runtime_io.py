"""Bounded out-of-process logging: no JSON encoding or filesystem IO in control."""
import json
import multiprocessing as mp
import os
import queue
import signal
from pathlib import Path
import numpy as np
from .session import atomic_json

def plain(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    raise TypeError(str(type(x)))

def _writer(path,q,cpus):
    signal.signal(signal.SIGINT,signal.SIG_IGN)
    # The writer must never inherit the optional real-time priority.
    os.sched_setscheduler(0,os.SCHED_OTHER,os.sched_param(0))
    if cpus:os.sched_setaffinity(0,cpus)
    with open(path,"w",buffering=1) as f:
        while True:
            item=q.get()
            if item is None:break
            kind,data=item
            if kind=="log":f.write(json.dumps(data,default=plain)+"\n")
            else:
                target,value=data
                atomic_json(target,value)

class RuntimeLog:
    def __init__(self,path,cpus=None,capacity=2048):
        self.path=Path(path);self.dropped=0
        ctx=mp.get_context("spawn")
        self.q=ctx.Queue(capacity)
        self.process=ctx.Process(target=_writer,args=(str(path),self.q,cpus),daemon=True)
        self.process.start()
    def _put(self,item):
        if not self.process.is_alive():
            self.dropped+=1;return False
        try:self.q.put_nowait(item);return True
        except queue.Full:self.dropped+=1;return False
    def write(self,data):return self._put(("log",data))
    def status(self,path,data):return self._put(("status",(str(path),data)))
    def close(self):
        clean=False
        try:
            self.q.put(None,timeout=2)
            self.process.join(10)
            clean=not self.process.is_alive() and self.process.exitcode==0
        except queue.Full:pass
        if self.process.is_alive():self.process.terminate();self.process.join(2)
        self.q.cancel_join_thread();self.q.close()
        atomic_json(self.path.parent/"logging_report.json",
                    dict(dropped_records=self.dropped,clean_shutdown=clean,
                         complete=clean and self.dropped==0))

def cpu_list(value):
    if not value:return None
    cpus={int(v) for v in value.split(",")}
    if not cpus or min(cpus)<0:raise ValueError("invalid CPU list")
    return cpus

def affinity_plan():
    """One logical CPU from each physical core; avoid SMT siblings across critical roles."""
    allowed=sorted(os.sched_getaffinity(0));groups={}
    for cpu in allowed:
        base=Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
        key=((base/"physical_package_id").read_text().strip(),(base/"core_id").read_text().strip())
        groups.setdefault(key,[]).append(cpu)
    cores=list(groups.values())
    if len(cores)<4:raise ValueError("auto affinity requires four physical cores; use --runtime-affinity none")
    selected=cores[-4:]
    critical={v for group in selected[:3] for v in group}
    return dict(physics=[selected[0][0]],sim_bridge=[selected[1][0]],
                runner=[selected[2][0]],logger=[selected[3][0]],
                viewer=[c for c in allowed if c not in critical])
