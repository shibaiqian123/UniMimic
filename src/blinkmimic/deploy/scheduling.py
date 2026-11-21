"""Read-only Linux scheduling evidence; never changes priorities or control deadlines."""
import os
from pathlib import Path
import resource
import threading
import time

def _text(path):
    try:return Path(path).read_text().strip()
    except OSError:return None

def scheduler_snapshot():
    tid=threading.get_native_id()
    policy=os.sched_getscheduler(0)
    names={getattr(os,n):n for n in ("SCHED_OTHER","SCHED_FIFO","SCHED_RR","SCHED_BATCH","SCHED_IDLE") if hasattr(os,n)}
    threads=[]
    for p in Path("/proc/self/task").iterdir():
        try:
            t=int(p.name)
            threads.append(dict(tid=t,policy=names.get(os.sched_getscheduler(t),"unknown"),
                                priority=os.sched_getparam(t).sched_priority,cpus=sorted(os.sched_getaffinity(t))))
        except (OSError,ValueError):pass
    return dict(pid=os.getpid(),tid=tid,policy=names.get(policy,str(policy)),
                priority=os.sched_getparam(0).sched_priority,cpus=sorted(os.sched_getaffinity(0)),
                rtprio_limit=list(resource.getrlimit(resource.RLIMIT_RTPRIO)),
                timerslack_ns=_text("/proc/self/timerslack_ns"),
                schedstats_enabled=_text("/proc/sys/kernel/sched_schedstats"),
                rt_runtime_us=_text("/proc/sys/kernel/sched_rt_runtime_us"),
                rt_period_us=_text("/proc/sys/kernel/sched_rt_period_us"),threads=threads)

class SleepMonitor:
    def __init__(self):
        self.path=f"/proc/self/task/{threading.get_native_id()}/schedstat"
        self.previous=None
        self.wait_stats_enabled=_text("/proc/sys/kernel/sched_schedstats")=="1"
    def _stat(self):
        raw=_text(self.path)
        try:return [int(x) for x in raw.split()] if raw else None
        except ValueError:return None
    def sleep_until(self,deadline):
        # Bracket only the sleep call; diagnostic reads are outside the measured interval.
        a=self._stat();u=resource.getrusage(resource.RUSAGE_THREAD)
        cpu=time.thread_time();start=time.monotonic()
        requested=max(0.,deadline-start)
        time.sleep(requested)
        end=time.monotonic();cpu_ms=(time.thread_time()-cpu)*1000
        v=resource.getrusage(resource.RUSAGE_THREAD);b=self._stat()
        self.previous=dict(sleep_start=start,sleep_return=end,deadline=deadline,
            requested_ms=requested*1000,elapsed_ms=(end-start)*1000,
            overshoot_ms=max(0.,end-max(start,deadline))*1000,thread_cpu_ms=cpu_ms,
            voluntary_switches=v.ru_nvcsw-u.ru_nvcsw,involuntary_switches=v.ru_nivcsw-u.ru_nivcsw,
            minor_faults=v.ru_minflt-u.ru_minflt,major_faults=v.ru_majflt-u.ru_majflt,
            schedstat_delta=None if a is None or b is None else dict(
                runtime_ns=b[0]-a[0],runqueue_wait_ns=(b[1]-a[1]) if self.wait_stats_enabled else None,timeslices=b[2]-a[2]),
            runqueue_wait_available=self.wait_stats_enabled)
        return self.previous
