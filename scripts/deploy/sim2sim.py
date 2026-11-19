"""BlinkMimic sim2sim: training-model baseline, official G1 PD, transitions, sweeps and viewer."""
import argparse
import itertools
import json
import os
from pathlib import Path
import queue
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"src"))
import mujoco
import numpy as np
from blinkmimic.deploy.core import Bundle,OnnxPolicy
from blinkmimic.deploy.sim2sim import Plant,Experiment

def floats(s):
    try:
        out=[float(v) for v in s.split(",")]
        if not out or not np.isfinite(out).all():raise ValueError()
        return out
    except ValueError:raise argparse.ArgumentTypeError("comma-separated finite numbers required")

def compare_models(b,train_xml,unitree_xml):
    a=Plant(b,train_xml,"g1"); c=Plant(b,unitree_xml,"g2")
    rows=[]
    for k,n in enumerate(b.cfg["joint_names"]):
        i,j=a.jids[k],c.jids[k]
        rows.append(dict(joint=n,training_range=a.m.jnt_range[i].tolist(),unitree_range=c.m.jnt_range[j].tolist(),
                         training_armature=float(a.m.dof_armature[a.vidx[k]]),unitree_armature=float(c.m.dof_armature[c.vidx[k]]),
                         training_damping=float(a.m.dof_damping[a.vidx[k]]),unitree_damping=float(c.m.dof_damping[c.vidx[k]]),
                         trained_effort=float(b.effort[k]),unitree_ctrlrange=c.m.actuator_ctrlrange[c.act[k]].tolist()))
    return dict(training_mass=float(a.m.body_mass.sum()),unitree_mass=float(c.m.body_mass.sum()),
                training_model_fingerprint=a.hash,unitree_model_fingerprint=c.hash,joints=rows)

def reference_error(exp,path):
    with np.load(path,allow_pickle=False) as z:
        if str(z["bundle_fingerprint"])!=exp.b.fingerprint:raise ValueError("reference bundle mismatch")
        if not exp.records: return dict(status="NO_SAMPLES")
        frames=np.array([r["frame"] for r in exp.records])
        n=min(len(frames),len(z["frame"]))
        if n==0 or not np.array_equal(frames[:n],z["frame"][:n]):raise ValueError("reference frame alignment mismatch")
        q=np.array([r["q"] for r in exp.records]); root=np.array([r["qpos"][exp.plant.fq:exp.plant.fq+3] for r in exp.records])
        n=min(len(q),len(z["q"]))
        qe=float(np.sqrt(np.mean((q[:n]-z["q"][:n])**2)))
        pe=float(np.sqrt(np.mean((root[:n,:2]-z["root_p"][:n,:2])**2)))
        return dict(status="COMPARED",samples=n,joint_rmse=qe,root_xy_rmse=pe,
                    passed=qe<=.05 and pe<=.10 and bool(z["complete"]) and n==exp.b.T-1,reference_complete=bool(z["complete"]))

def run_job(job):
    b=Bundle(job["bundle"])
    p=Plant(b,job["xml"],"g1" if job["cmd"]=="g1" else "g2",
            job["latency"],job["kp"],job["mass"],job["friction"])
    e=Experiment(b,p,job["cmd"],job["stage"],job["diagnostic"],job["standby_s"])
    limit=job["max_seconds"]
    while not e.finished and (limit is None or e.time<limit-1e-9):
        e.tick()
        if int(e.time*50)%500==0:print(f"{job['label']}: {e.state} frame={e.frame} sim={e.time:.1f}s",flush=True)
    extra=dict(settings=job)
    if job.get("reference"): extra["training_comparison"]=reference_error(e,job["reference"])
    elif job["cmd"]=="g1":extra["training_comparison"]={"status":"NOT_RUN","reason":"pass --reference from record_sim2sim_reference.py"}
    if job.get("reference") and not extra["training_comparison"].get("passed",False):
        extra["passed"]=False
    report=e.save(job["out"],extra)
    return report

def view(job,speed):
    from mujoco import viewer
    b=Bundle(job["bundle"]); events=queue.SimpleQueue()
    def build():
        p=Plant(b,job["xml"],"g1" if job["cmd"]=="g1" else "g2",job["latency"],job["kp"],job["mass"],job["friction"])
        return Experiment(b,p,job["cmd"],job["stage"],job["diagnostic"],job["standby_s"])
    e=build();p=e.plant;ref=mujoco.MjData(p.m);paused=False;ghost=True;follow=True; deadline=time.monotonic()
    def draw(scn):
        scn.ngeom=0
        if not ghost:return
        f=min(e.frame,b.T-1)
        ref.qpos[:]=p.d.qpos
        ref.qpos[p.fq:p.fq+3]=b.motion["root_p"][f]+[0,0,p.ground_shift]
        mujoco.mju_mat2Quat(ref.qpos[p.fq+3:p.fq+7],b.motion["root_R"][f].reshape(9))
        ref.qpos[p.qidx]=b.motion["q"][f]
        mujoco.mj_kinematics(p.m,ref)
        for i in range(1,p.m.nbody):
            if scn.ngeom+2>scn.maxgeom:break
            xyz=ref.xpos[i];parent=p.m.body_parentid[i]
            g=scn.geoms[scn.ngeom];mujoco.mjv_initGeom(g,mujoco.mjtGeom.mjGEOM_SPHERE,[.023,0,0],xyz,np.eye(3).reshape(9),[.1,.7,1,.8]);scn.ngeom+=1
            if parent>0 and np.linalg.norm(xyz-ref.xpos[parent])>1e-6:
                g=scn.geoms[scn.ngeom];mujoco.mjv_initGeom(g,mujoco.mjtGeom.mjGEOM_CAPSULE,[.01,0,0],[0,0,0],np.eye(3).reshape(9),[.1,.7,1,.6])
                mujoco.mjv_connector(g,mujoco.mjtGeom.mjGEOM_CAPSULE,.01,ref.xpos[parent],xyz);scn.ngeom+=1
    with viewer.launch_passive(p.m,p.d,key_callback=events.put,show_left_ui=False,show_right_ui=False) as v:
        v.cam.distance=3.;v.cam.azimuth=135.;v.cam.elevation=-15.
        while v.is_running():
            single=False
            while not events.empty():
                key=events.get()
                if key==32:paused=not paused
                elif key==78:paused=True;single=True
                elif key==71:ghost=not ghost
                elif key==70:follow=not follow
                elif key in (61,334):speed=min(2.,speed*2)
                elif key in (45,333):speed=max(.125,speed/2)
                elif key==82:
                    # Reset plant/controller in place so viewer remains attached to its original model/data.
                    fresh=build()
                    with v.lock():
                        p.d.qpos[:]=fresh.plant.d.qpos;p.d.qvel[:]=fresh.plant.d.qvel;p.d.time=0
                        p.d.ctrl[:]=fresh.plant.d.ctrl;mujoco.mj_forward(p.m,p.d)
                        p.pending.clear();p.steps=0;p.target=fresh.plant.target;p.kp=fresh.plant.kp;p.kd=fresh.plant.kd
                        p.max_torque_ratio=p.max_velocity_ratio=0;p.torque_saturated=p.model_saturated=0
                        e=Experiment(b,p,job["cmd"],job["stage"],job["diagnostic"],job["standby_s"])
                    paused=False
            if not e.finished and (single or (not paused and time.monotonic()>=deadline)):
                with v.lock():e.tick()
                if e.finished:paused=True;print(json.dumps(e.save(job["out"],dict(settings=job)),indent=2),flush=True)
                deadline=time.monotonic()+.02/speed
            with v.lock():
                draw(v.user_scn)
                if follow:v.cam.lookat[:]=p.d.qpos[p.fq:p.fq+3]
            text=(f"SIM2SIM {job['cmd']} / CPU MuJoCo / NO DDS\n"
                  f"{e.state} {'PAUSED' if paused else 'RUNNING'} | frame {e.frame}/{b.T-1} | sim {e.time:.2f}s\n"
                  f"PD {round(1/p.dt)}Hz | policy 50Hz | delay {job['latency']}ms | speed {speed:g}x\n"
                  f"kp scale {job['kp']} | mass +{job['mass']}kg | mu {job['friction']}\n"
                  f"Blue skeleton = reference | support {e.support}\n{e.stop_reason}\n"
                  "SPACE pause | N step | R reset | +/- speed | G reference | F camera")
            v.set_texts((None,mujoco.mjtGridPos.mjGRID_TOPLEFT,text,""));v.sync();time.sleep(.005)
    report=e.save(job["out"],dict(settings=job,viewer_closed=True))
    print(json.dumps(report,indent=2))

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bundle",type=Path,required=True)
    ap.add_argument("--cmd",choices=["g1","g2","standby","sweep","compare","timing"],default="g2")
    ap.add_argument("--train-xml",type=Path)
    ap.add_argument("--unitree-xml",type=Path,default=os.environ.get("G1_UNITREE_XML"))
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--viewer",action="store_true")
    ap.add_argument("--speed",type=float,default=1.)
    ap.add_argument("--max-seconds",type=float)
    ap.add_argument("--latency-ms",type=float,default=0.)
    ap.add_argument("--kp-scale",type=float,default=1.)
    ap.add_argument("--extra-mass",type=float,default=0.)
    ap.add_argument("--friction",type=float)
    ap.add_argument("--stage",choices=["air","feet","slack","ground"],default="ground")
    ap.add_argument("--standby-seconds",type=float,default=5.)
    ap.add_argument("--diagnostic-continue",action="store_true",help="g2 only: log deployment guards without stopping; never accepted as a pass")
    ap.add_argument("--reference",type=Path,help="g1 GPU-training trajectory")
    ap.add_argument("--latencies",type=floats,default=[0,5,10,20])
    ap.add_argument("--kp-scales",type=floats,default=[.85,1.,1.15])
    ap.add_argument("--masses",type=floats,default=[0,1.8])
    ap.add_argument("--frictions",type=floats,default=[.6,1.])
    a=ap.parse_args()
    if a.out.exists():ap.error("--out must be a new directory")
    numeric=[a.speed,a.latency_ms,a.kp_scale,a.extra_mass,a.standby_seconds]+([] if a.friction is None else [a.friction])+([] if a.max_seconds is None else [a.max_seconds])
    if not np.isfinite(numeric).all() or a.speed<=0 or a.latency_ms<0 or a.kp_scale<=0 or a.extra_mass<0 or a.standby_seconds<3 or (a.friction is not None and a.friction<=0) or (a.max_seconds is not None and a.max_seconds<=0):ap.error("invalid numeric settings")
    if any(x<0 for x in a.latencies+a.masses) or any(x<=0 for x in a.kp_scales+a.frictions):ap.error("invalid sweep grid")
    if a.viewer and a.cmd not in ("g1","g2","standby"):ap.error("viewer supports g1/g2/standby")
    if a.diagnostic_continue and a.cmd!="g2":ap.error("diagnostic continuation is g2 only")
    if a.reference and a.cmd!="g1":ap.error("--reference is g1 only")
    b=Bundle(a.bundle)
    a.out.mkdir(parents=True)
    if a.cmd=="timing":
        policy=OnnxPolicy(b);h=np.zeros((10,99));g=b.motion["goal"][0]
        for _ in range(20):policy(h,g)
        ts=[]
        for _ in range(1000):
            t=time.perf_counter();policy(h,g);ts.append((time.perf_counter()-t)*1000)
        result=dict(bundle_fingerprint=b.fingerprint,p50_ms=float(np.percentile(ts,50)),p99_ms=float(np.percentile(ts,99)),max_ms=max(ts),passed=max(ts)<20.,scope="ONNX inference only, not full control loop")
        (a.out/"report.json").write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2));return
    if a.cmd=="compare":
        if not a.train_xml or not a.unitree_xml:ap.error("compare requires both XML paths")
        result=compare_models(b,a.train_xml,a.unitree_xml);(a.out/"report.json").write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!="joints"},indent=2));return
    xml=a.train_xml if a.cmd=="g1" else a.unitree_xml
    if not xml:ap.error("provide --train-xml for g1, --unitree-xml or G1_UNITREE_XML otherwise")
    base=dict(bundle=str(a.bundle.resolve()),xml=str(Path(xml).resolve()),cmd=a.cmd,
              latency=a.latency_ms,kp=a.kp_scale,mass=a.extra_mass,friction=a.friction,
              out=str(a.out.resolve()),stage=a.stage,diagnostic=a.diagnostic_continue,
              standby_s=a.standby_seconds,max_seconds=a.max_seconds,reference=str(a.reference) if a.reference else None,label=a.cmd)
    if a.cmd=="g1" and (a.latency_ms!=0 or a.kp_scale!=1 or a.extra_mass!=0 or a.friction is not None):
        ap.error("g1 baseline does not accept plant perturbations")
    if a.cmd=="sweep":
        reports=[]
        for i,(lat,kp,mass,mu) in enumerate(itertools.product(a.latencies,a.kp_scales,a.masses,a.frictions)):
            job=dict(base,cmd="g2",latency=lat,kp=kp,mass=mass,friction=mu,label=f"case_{i:03d}",out=str(a.out/f"case_{i:03d}"))
            reports.append(run_job(job))
            (a.out/"report.json").write_text(json.dumps(dict(cases=reports,finished_cases=len(reports)),indent=2))
        print(f"{sum(r['passed'] for r in reports)}/{len(reports)} full scenarios passed");return
    if a.viewer:view(base,a.speed)
    else:
        r=run_job(base);print(json.dumps(r,indent=2))
        if not r["passed"]:raise SystemExit(1 if r["stop_reason"] else 2)
if __name__=="__main__":main()
