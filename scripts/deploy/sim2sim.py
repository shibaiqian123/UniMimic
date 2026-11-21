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

def build_experiment(b,job):
    cls=Plant;kw={};policy=None
    if job.get("backend")=="warp":
        from blinkmimic.deploy.warp_official import WarpPlant,TorchCudaPolicy
        cls=WarpPlant;kw["device"]=job["device"]
        if job.get("ckpt"):policy=TorchCudaPolicy(b,job["ckpt"],job["device"])
    p=cls(b,job["xml"],"g1" if job["cmd"]=="g1" else "g2",
          job["latency"],job["kp"],job["mass"],job["friction"],**kw)
    return Experiment(b,p,job["cmd"],job["stage"],job["diagnostic"],job["standby_s"],policy=policy)

def run_job(job):
    b=Bundle(job["bundle"])
    e=build_experiment(b,job)
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

def view(job,speed,render_hz=30.):
    from mujoco import viewer
    from blinkmimic.deploy.viewer_utils import ControlClock,RenderMirror
    b=Bundle(job["bundle"]);events=queue.SimpleQueue()
    e=build_experiment(b,job);p=e.plant;mirror=RenderMirror(p.m)
    mirror.update(p.d)
    ref=mujoco.MjData(mirror.m);paused=False;ghost=True;follow=True
    clock=ControlClock(.02/speed,time.monotonic())
    render_period=1./render_hz;next_render=0.;last_tick_ms=0.
    def draw(scn):
        scn.ngeom=0
        if not ghost:return
        f=min(e.frame,b.T-1)
        ref.qpos[:]=p.d.qpos
        ref.qpos[p.fq:p.fq+3]=b.motion["root_p"][f]+[0,0,p.ground_shift]
        mujoco.mju_mat2Quat(ref.qpos[p.fq+3:p.fq+7],b.motion["root_R"][f].reshape(9))
        ref.qpos[p.qidx]=b.motion["q"][f]
        mujoco.mj_kinematics(mirror.m,ref)
        for i in range(1,mirror.m.nbody):
            if scn.ngeom+2>scn.maxgeom:break
            xyz=ref.xpos[i];parent=mirror.m.body_parentid[i]
            g=scn.geoms[scn.ngeom];mujoco.mjv_initGeom(g,mujoco.mjtGeom.mjGEOM_SPHERE,[.023,0,0],xyz,np.eye(3).reshape(9),[.1,.7,1,.8]);scn.ngeom+=1
            if parent>0 and np.linalg.norm(xyz-ref.xpos[parent])>1e-6:
                g=scn.geoms[scn.ngeom];mujoco.mjv_initGeom(g,mujoco.mjtGeom.mjGEOM_CAPSULE,[.01,0,0],[0,0,0],np.eye(3).reshape(9),[.1,.7,1,.6])
                mujoco.mjv_connector(g,mujoco.mjtGeom.mjGEOM_CAPSULE,.01,ref.xpos[parent],xyz);scn.ngeom+=1
    # Separate MjData: the render thread cannot race with p.d during physics/readback.
    with viewer.launch_passive(mirror.m,mirror.d,key_callback=events.put,show_left_ui=False,show_right_ui=False) as v:
        v.cam.distance=3.;v.cam.azimuth=135.;v.cam.elevation=-15.
        while v.is_running():
            single=False;redraw=False
            while not events.empty():
                key=events.get();redraw=True
                if key==32:
                    paused=not paused;clock.set_period(.02/speed,time.monotonic())
                elif key==78:paused=True;single=True
                elif key==71:ghost=not ghost
                elif key==70:follow=not follow
                elif key in (61,334):
                    speed=min(2.,speed*2);clock.set_period(.02/speed,time.monotonic())
                elif key in (45,333):
                    speed=max(.125,speed/2);clock.set_period(.02/speed,time.monotonic())
                elif key==82:
                    # Reuse model and policy; GPU reset does not rebuild collision kernels.
                    policy=e.core.policy;p.restart()
                    e=Experiment(b,p,job["cmd"],job["stage"],job["diagnostic"],job["standby_s"],policy=policy)
                    paused=False;clock.set_period(.02/speed,time.monotonic())
            if not e.finished and (single or (not paused and clock.due(time.monotonic()))):
                started=time.monotonic()
                e.tick()  # no viewer lock while executing CPU/GPU physics
                finished=time.monotonic();last_tick_ms=(finished-started)*1000
                clock.advance(finished)
                if e.finished:
                    paused=True;redraw=True
                    print("Simulation stopped: "+(e.stop_reason or "COMPLETE"),flush=True)
            now=time.monotonic()
            if redraw or now>=next_render:
                with v.lock():
                    mirror.update(p.d);draw(v.user_scn)
                    if follow:v.cam.lookat[:]=mirror.d.qpos[p.fq:p.fq+3]
                text=(f"SIM2SIM {job['cmd']} / {job.get('backend','cpu')} physics / NO DDS\n"
                      f"{e.state} {'PAUSED' if paused else 'RUNNING'} | frame {e.frame}/{b.T-1} | sim {e.time:.2f}s\n"
                      f"PD {round(1/p.dt)}Hz | policy 50Hz | delay {job['latency']}ms | speed {speed:g}x\n"
                      f"Control tick {last_tick_ms:.1f}ms / budget {20/speed:.1f}ms | display {render_hz:g}Hz\n"
                      f"Blue skeleton = reference | support {e.support}\n{e.stop_reason}\n"
                      "SPACE pause | N step | R reset | +/- speed | G reference | F camera")
                v.set_texts((None,mujoco.mjtGridPos.mjGRID_TOPLEFT,text,""))
                v.sync()
                next_render=now+render_period
            due=next_render
            if not paused and not e.finished:due=min(due,clock.deadline)
            time.sleep(min(.005,max(0.,due-time.monotonic())))
    report=e.save(job["out"],dict(settings=job,viewer_closed=True,render_hz=render_hz))
    print(json.dumps(report,indent=2))

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bundle",type=Path,required=True)
    ap.add_argument("--cmd",choices=["g1","g2","standby","sweep","compare","timing","final"],default="g2")
    ap.add_argument("--train-xml",type=Path)
    ap.add_argument("--unitree-xml",type=Path,default=os.environ.get("G1_UNITREE_XML"))
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--viewer",action="store_true")
    ap.add_argument("--gl-renderer",choices=["auto","system","nvidia"],default="auto",help="interactive OpenGL selection; auto replaces llvmpipe when NVIDIA offload works")
    ap.add_argument("--speed",type=float,default=1.)
    ap.add_argument("--render-hz",type=float,default=30.,help="display refresh limit; independent of 50Hz control")
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
    ap.add_argument("--backend",choices=["cpu","warp"],default="cpu",help="warp: GPU physics; g1 training model, g2/standby official torque model")
    ap.add_argument("--ckpt",type=Path,help="matching .pt checkpoint, required for warp")
    ap.add_argument("--device",default="cuda:0")
    ap.add_argument("--num-envs",type=int,default=1)
    ap.add_argument("--view-env",type=int,default=0)
    ap.add_argument("--precision",choices=["checkpoint","none","bf16"],default="checkpoint")
    ap.add_argument("--record",action="store_true",help="warp: record selected replica trajectory")
    ap.add_argument("--screenshot",action="store_true",help="warp headless: render final state to out/viewer.png (MUJOCO_GL=egl)")
    ap.add_argument("--rt-priority",type=int,default=0,help="final only: optional SCHED_RR priority 1..10; requires OS permission, no silent fallback")
    ap.add_argument("--runtime-affinity",choices=["auto","none"],default="auto",help="final: separate physics, DDS, control, and logging CPU resources")
    ap.add_argument("--auto",action="store_true",help="final simulation only: send the same keyboard/button events automatically")
    ap.add_argument("--duration",type=float,help="final wall-clock duration, then request stop")
    ap.add_argument("--dds-domain",type=int,choices=[1,2],default=1)
    a=ap.parse_args()
    if not np.isfinite(a.render_hz) or not 1<=a.render_hz<=120:ap.error("--render-hz must be in [1,120]")
    if a.cmd=="final":
        if a.backend!="cpu":ap.error("final uses CPU official model; GPU modes remain diagnostic")
        if a.max_seconds is not None:ap.error("final uses --duration in wall-clock seconds")
        if a.speed!=1 or a.latency_ms!=0 or a.kp_scale!=1 or a.extra_mass!=0 or a.friction is not None or a.num_envs!=1 or a.screenshot or a.reference or a.diagnostic_continue:ap.error("final uses real-time shared configuration; diagnostic overrides are not supported")
        if a.duration is not None and (not np.isfinite(a.duration) or a.duration<=0):ap.error("positive duration required")
        if a.viewer:
            from blinkmimic.deploy.graphics import configure_opengl
            configure_opengl(a.gl_renderer)
        from blinkmimic.deploy.final_session import run
        run(a);return
    if a.out.exists():
        from blinkmimic.deploy.session import new_session
        a.out=new_session(a.out)
    else:a.out.mkdir(parents=True,exist_ok=True)

    numeric=[a.speed,a.latency_ms,a.kp_scale,a.extra_mass,a.standby_seconds]+([] if a.friction is None else [a.friction])+([] if a.max_seconds is None else [a.max_seconds])
    if not np.isfinite(numeric).all() or a.speed<=0 or a.latency_ms<0 or a.kp_scale<=0 or a.extra_mass<0 or a.standby_seconds<3 or (a.friction is not None and a.friction<=0) or (a.max_seconds is not None and a.max_seconds<=0):ap.error("invalid numeric settings")
    if any(x<0 for x in a.latencies+a.masses) or any(x<=0 for x in a.kp_scales+a.frictions):ap.error("invalid sweep grid")
    if a.viewer and a.cmd not in ("g1","g2","standby"):ap.error("viewer supports g1/g2/standby")
    if a.diagnostic_continue and a.cmd!="g2":ap.error("diagnostic continuation is g2 only")
    if a.reference and a.cmd!="g1":ap.error("--reference is g1 only")
    if a.viewer:
        from blinkmimic.deploy.graphics import configure_opengl
        a.graphics_info=configure_opengl(a.gl_renderer)
    else:a.graphics_info=None
    if a.backend=="warp" and a.cmd=="g1":
        if a.ckpt is None:ap.error("--ckpt is required for warp")
        if a.num_envs<1 or not 0<=a.view_env<a.num_envs:ap.error("invalid replica count/view index")
        if a.reference or a.diagnostic_continue or a.latency_ms!=0 or a.kp_scale!=1 or a.extra_mass!=0 or a.friction is not None:
            ap.error("warp uses original training dynamics; no CPU perturbation/reference flags")
        if a.screenshot and a.viewer:ap.error("--screenshot is headless only")
        b=Bundle(a.bundle);a.out.mkdir(parents=True,exist_ok=True)
        from blinkmimic.deploy.warp_runner import run
        run(a,b);return
    if a.backend=="warp":
        if a.cmd not in ("g2","standby","sweep"):ap.error("warp supports g1, g2, standby, sweep")
        if a.num_envs!=1 or a.view_env!=0 or a.screenshot or a.record or a.reference:ap.error("official GPU backend is single-world; trajectories are saved automatically")
        if a.precision not in ("checkpoint","none"):ap.error("official GPU policy uses fp32 to match ONNX")
    elif a.ckpt or a.num_envs!=1 or a.view_env!=0 or a.record or a.screenshot or a.precision!="checkpoint":
        ap.error("GPU-only options require --backend warp")
    b=Bundle(a.bundle)
    a.out.mkdir(parents=True,exist_ok=True)
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
              standby_s=a.standby_seconds,max_seconds=a.max_seconds,reference=str(a.reference) if a.reference else None,label=a.cmd,
              backend=a.backend,device=a.device,graphics=a.graphics_info,ckpt=str(a.ckpt.resolve()) if a.ckpt else None)
    if a.cmd=="g1" and (a.latency_ms!=0 or a.kp_scale!=1 or a.extra_mass!=0 or a.friction is not None):
        ap.error("g1 baseline does not accept plant perturbations")
    if a.cmd=="sweep":
        reports=[]
        for i,(lat,kp,mass,mu) in enumerate(itertools.product(a.latencies,a.kp_scales,a.masses,a.frictions)):
            job=dict(base,cmd="g2",latency=lat,kp=kp,mass=mass,friction=mu,label=f"case_{i:03d}",out=str(a.out/f"case_{i:03d}"))
            reports.append(run_job(job))
            (a.out/"report.json").write_text(json.dumps(dict(cases=reports,finished_cases=len(reports)),indent=2))
        print(f"{sum(r['passed'] for r in reports)}/{len(reports)} full scenarios passed");return
    if a.viewer:view(base,a.speed,a.render_hz)
    else:
        r=run_job(base);print(json.dumps(r,indent=2))
        if not r["passed"]:raise SystemExit(1 if r["stop_reason"] else 2)
if __name__=="__main__":main()
