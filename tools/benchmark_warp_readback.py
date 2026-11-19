"""Compare full vs compact readback on the SAME GPU states; does not contact hardware."""
import argparse,json,time
from pathlib import Path
import numpy as np
import mujoco
import mujoco_warp as mw
import warp as wp
from blinkmimic.deploy.core import Bundle
from blinkmimic.deploy.warp_official import WarpPlant

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--bundle",required=True,type=Path);ap.add_argument("--unitree-xml",required=True,type=Path)
    ap.add_argument("--out",required=True,type=Path);ap.add_argument("--steps",type=int,default=20)
    ap.add_argument("--device",default="cuda:0")
    a=ap.parse_args()
    if a.out.exists() or a.steps<5:ap.error("new output and at least five samples required")
    a.out.mkdir(parents=True)
    b=Bundle(a.bundle);p=WarpPlant(b,a.unitree_xml,device=a.device)
    old=mujoco.MjData(p.m);full=[];compact=[];step_times=[];max_error=0.;counter_error=0
    for i in range(a.steps+5):
        p.command(b.motion["q"][0],b.kp,b.kd)
        p.step()
        with wp.ScopedDevice(p.gpu_device):
            # Alternate measurement order to reduce fixed-order caching bias.
            for mode in (("full","compact") if i%2==0 else ("compact","full")):
                t=time.perf_counter()
                if mode=="full":
                    mw.get_data_into(old,p.m,p.gpu_data,world_id=0)
                else:p.pull_state()
                elapsed=(time.perf_counter()-t)*1000
                if i>=5:(full if mode=="full" else compact).append(elapsed)
            counts=p.counts.numpy();peaks=p.peaks.numpy()
        max_error=max(max_error,float(abs(old.qpos-p.d.qpos).max()),float(abs(old.qvel-p.d.qvel).max()),abs(old.time-p.d.time),
                      abs(float(peaks[0])-p.max_torque_ratio),abs(float(peaks[1])-p.max_velocity_ratio))
        counter_error=max(counter_error,abs(int(counts[0])-p.torque_saturated),abs(int(counts[1])-p.model_saturated))
        if i>=5:step_times.append(p.last_step_ms)
    report=dict(bundle_fingerprint=b.fingerprint,samples=a.steps,
                full_readback_median_ms=float(np.median(full)),compact_readback_median_ms=float(np.median(compact)),
                full_readback_p95_ms=float(np.percentile(full,95)),compact_readback_p95_ms=float(np.percentile(compact,95)),
                readback_speedup=float(np.median(full)/np.median(compact)),
                optimized_physics_step_median_ms=float(np.median(step_times)),
                max_state_difference=max_error,counter_difference=counter_error,passed=max_error==0 and counter_error==0,
                bytes_per_compact_snapshot=int(p.host_packed.size*8),
                scope="same-state readback correctness and timing only; not whole-viewer FPS or complete motion")
    (a.out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2),flush=True)
    if not report["passed"]:raise SystemExit(1)
if __name__=="__main__":main()
