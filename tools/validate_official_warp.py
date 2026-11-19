"""Compare official-model CPU/GPU PD and short trajectories; never communicates with a robot."""
import argparse,json
from pathlib import Path
import numpy as np
import warp as wp
from blinkmimic.deploy.core import Bundle,OnnxPolicy
from blinkmimic.deploy.sim2sim import Plant
from blinkmimic.deploy.warp_official import WarpPlant,TorchCudaPolicy,pd_step

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bundle",required=True,type=Path);ap.add_argument("--unitree-xml",required=True,type=Path)
    ap.add_argument("--ckpt",required=True,type=Path);ap.add_argument("--out",required=True,type=Path)
    ap.add_argument("--device",default="cuda:0");ap.add_argument("--steps",type=int,default=20)
    a=ap.parse_args()
    if a.out.exists() or a.steps<1:ap.error("new output directory and positive steps required")
    a.out.mkdir(parents=True);b=Bundle(a.bundle)
    report=dict(bundle_fingerprint=b.fingerprint,device=a.device,passed=False,status="RUNNING")
    (a.out/"report.json").write_text(json.dumps(report))
    results=[]
    for delay in (0.,10.):
        cpu=Plant(b,a.unitree_xml,"g2",latency_ms=delay)
        gpu=WarpPlant(b,a.unitree_xml,"g2",latency_ms=delay,device=a.device)
        if cpu.hash!=gpu.hash:raise RuntimeError("models differ before backend conversion")
        qerr=perr=0.;target_err=0.
        q0=cpu.state()[0]
        for i in range(a.steps):
            target=q0+.01*np.sin(np.arange(29)*.3+i*.02)
            cpu.command(target,b.kp,b.kd);gpu.command(target,b.kp,b.kd)
            cpu.step();gpu.step()
            qerr=max(qerr,float(abs(cpu.state()[0]-gpu.state()[0]).max()))
            perr=max(perr,float(abs(cpu.d.qpos[cpu.fq:cpu.fq+3]-gpu.d.qpos[gpu.fq:gpu.fq+3]).max()))
            target_err=max(target_err,float(abs(cpu.target-gpu.target).max()))
        results.append(dict(latency_ms=delay,seconds=a.steps*.02,joint_max_abs_error_rad=qerr,
                            root_max_abs_error_m=perr,scheduled_target_error=target_err,
                            passed=qerr<=.02 and perr<=.01 and target_err==0.,
                            graph=gpu.graph is not None,graph_fallback_reason=gpu.capture_error))
        print(json.dumps(results[-1]),flush=True)
    ort=OnnxPolicy(b);cuda=TorchCudaPolicy(b,a.ckpt,a.device)
    rng=np.random.default_rng(17);error=0.
    for i in np.linspace(0,b.T-1,10,dtype=int):
        h=rng.normal(0,.1,(10,99));g=b.motion["goal"][i]
        error=max(error,float(abs(ort(h,g)-cuda(h,g)).max()))
    # Pure PD kernel test: reversed actuator order, effort clip then model ctrl clip.
    with wp.ScopedDevice(a.device):
        arr=lambda x,dtype=float:wp.array(np.asarray(x),dtype=dtype)
        q=np.zeros((1,29),np.float32);dq=np.linspace(-2,2,29,dtype=np.float32)[None]
        target=np.linspace(-1,1,29,dtype=np.float32)[None];kp=np.full((1,29),20,np.float32);kd=np.full((1,29),2,np.float32)
        ix=np.arange(29,dtype=np.int32);act=ix[::-1].copy();ctrl=wp.zeros((1,29),dtype=float)
        counts=wp.zeros(2,dtype=int);peaks=wp.zeros(2,dtype=float)
        wp.launch(pd_step,dim=29,inputs=[arr(q),arr(dq),ctrl,arr(ix,int),arr(ix,int),arr(act,int),arr(target),arr(kp),arr(kd),
            arr(np.full(29,5)),arr(np.full(29,10)),arr(np.ones(29),int),arr(np.full(29,-3)),arr(np.full(29,3)),counts,peaks,0])
        actual=ctrl.numpy()[0];expected=np.zeros(29)
        expected[act]=np.clip(np.clip((kp*(target-q)-kd*dq)[0],-5,5),-3,3)
        pd_error=float(abs(actual-expected).max())
    report.update(status="COMPLETE",trajectories=results,policy_max_error=error,policy_tolerance=1e-3,
                  pd_kernel_error=pd_error,passed=all(r["passed"] for r in results) and error<=1e-3 and pd_error<=1e-6,
                  scope="short open-loop CPU/GPU physics comparison plus policy/PD checks; not full motion or hardware")
    (a.out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))
    if not report["passed"]:raise SystemExit(1)
if __name__=="__main__":main()
