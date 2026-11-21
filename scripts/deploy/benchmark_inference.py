"""Replay recorded policy inputs; CPU/CUDA parity and end-to-end inference timing. No DDS."""
import argparse,json,time
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"src"))
from blinkmimic.deploy.core import Bundle,OnnxPolicy

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bundle",type=Path,required=True)
    ap.add_argument("--runner-log",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--samples",type=int,default=128)
    ap.add_argument("--iterations",type=int,default=500)
    a=ap.parse_args()
    if a.samples<1 or a.iterations<1:ap.error("positive samples and iterations required")
    b=Bundle(a.bundle);frames=[];history=[];matched=False
    for line in a.runner_log.open():
        r=json.loads(line)
        if r.get("event")=="start":
            if r.get("bundle_fingerprint")!=b.fingerprint:raise ValueError("recording/bundle mismatch")
            matched=True
        o=r.get("result",{})
        if "observation" not in o:continue
        history.append(np.asarray(o["observation"],dtype=np.float32));history=history[-10:]
        h=np.stack([history[0]]*(10-len(history))+history)
        f=o["frame"];phase=o["phase"]
        g=b.motion["goal"][f] if phase=="motion" else b.motion["static_goal"][0 if phase=="pre" else 1]
        orient=np.asarray(o["orientation6d_future"],np.float32) if b.cfg["obs_dim"]==93 else None
        frames.append((h,g,orient))
    if not matched or not frames:raise ValueError("no matching policy records")
    frames=[frames[i] for i in np.linspace(0,len(frames)-1,min(a.samples,len(frames)),dtype=int)]
    cpu=OnnxPolicy(b,"cpu");gpu=OnnxPolicy(b,"cuda:0")
    errors=[float(np.max(np.abs(cpu(*x)-gpu(*x)))) for x in frames]
    report=dict(bundle=b.fingerprint,source=str(a.runner_log),samples=len(frames),max_action_error=max(errors),
                parity_passed=max(errors)<=1e-3,cpu=cpu.info,gpu=gpu.info)
    for name,policy in (("cpu",cpu),("gpu",gpu)):
        times=[];deadline=time.monotonic()
        for i in range(a.iterations):
            start=time.perf_counter();policy(*frames[i%len(frames)])
            times.append((time.perf_counter()-start)*1000)
            deadline=max(deadline+.02,time.monotonic())
            time.sleep(max(0,deadline-time.monotonic()))
        report[name+"_timing_ms"]=dict(zip(("p50","p99","max"),np.percentile(times,[50,99,100]).tolist()))
        report[name+"_over20ms"]=sum(t>20 for t in times)
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
    if not report["parity_passed"]:raise SystemExit(1)
if __name__=="__main__":main()
