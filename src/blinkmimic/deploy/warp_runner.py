"""GPU physics and GPU policy using the original BlinkMimic mjlab/MuJoCo Warp environment."""
import copy
import json
from pathlib import Path
import queue
import time
import numpy as np
import torch
from .core import Bundle,digest

class WarpRunner:
    def __init__(self,bundle,ckpt,device="cuda:0",num_envs=1,precision="checkpoint",focus=0,record=False):
        if not torch.cuda.is_available():raise RuntimeError("CUDA is required for --backend warp")
        self.device=torch.device(device)
        if self.device.type!="cuda":raise ValueError("Warp backend requires a CUDA device")
        torch.cuda.set_device(self.device)
        if num_envs<1 or not 0<=focus<num_envs:raise ValueError("invalid replica count / view index")
        self.b=bundle;self.N=num_envs;self.focus=focus;self.record=record
        if digest(ckpt)!=bundle.cfg["checkpoint_sha256"]:raise ValueError("checkpoint does not match bundle")
        csv=Path(bundle.cfg["source_motion"])
        if not csv.exists() or digest(csv)!=bundle.cfg["source_motion_sha256"]:
            raise ValueError("matching source CSV must exist at "+str(csv))
        from blinkmimic import networks
        from blinkmimic.networks import Policy,policy_kwargs
        from blinkmimic.robot import get_robot
        from blinkmimic.kinematics import Kinematics
        from blinkmimic.motion import MotionLib
        from blinkmimic.mdp import MDP
        from blinkmimic.dr import XiLayout
        from blinkmimic.env import TrackingSim,make_sim
        torch.set_num_threads(4);torch.manual_seed(0)
        ck=torch.load(ckpt,map_location="cpu",weights_only=False);cfg=ck["args"]
        if cfg["hist"]!=10 or cfg["goal_frames"]!=10:raise ValueError("10-frame history/goal required")
        self.precision=cfg.get("amp","none") if precision=="checkpoint" else precision
        networks.AMP_DTYPE=torch.bfloat16 if self.precision=="bf16" else None
        torch.backends.cuda.matmul.allow_tf32=self.precision=="bf16"
        robot=get_robot("g1");kin=Kinematics(robot,str(self.device))
        self.motion=MotionLib([str(csv)],kin,str(self.device),robot.load_motion)
        if self.motion.num_frames!=bundle.T:raise ValueError("motion frame-count mismatch")
        mdp=MDP(kin,self.motion,torch.tensor(bundle.q0,dtype=torch.float32,device=self.device),
                XiLayout(robot,str(self.device)).dim,robot.sensing)
        self.env=TrackingSim(make_sim(robot),num_envs,robot,self.motion,mdp,str(self.device),
                             eval_mode=True,randomize=False)
        widths=lambda k:tuple(map(int,cfg.get(k,"2048,1024,512,256").split(",")))
        self.pi=Policy(99,730,29,hist=10,z_dim=32,enc_hidden=widths("enc_hidden"),
                       dec_hidden=widths("dec_hidden"),**policy_kwargs(cfg,robot)).to(self.device).eval()
        self.pi.load_state_dict(ck["pi"],strict=True)
        self.ids=torch.arange(num_envs,device=self.device)
        self.reset()
    def reset(self):
        self.env.done_eval[:]=False;self.env.failed_eval[:]=False
        self.env.reset(self.ids,torch.zeros(self.N,dtype=torch.long,device=self.device))
        self.env.refresh()
        self.history=None;self.returns=torch.zeros(self.N,device=self.device)
        self.steps=torch.zeros(self.N,dtype=torch.long,device=self.device)
        self.final_qpos=self.env.sim.data.qpos.clone()
        self.final_frame=torch.zeros(self.N,dtype=torch.long,device=self.device)
        self.ticks=0;self.finished=False;self.records=[];self.tick_ms=[]
        torch.cuda.synchronize(self.device)
    @torch.no_grad()
    def tick(self):
        if self.finished:return False
        t=time.perf_counter()
        alive=~self.env.done_eval.clone()
        before=self.env.f.clone()
        o=self.env.observe()
        self.history=o[:,None].repeat(1,10,1) if self.history is None else torch.cat((self.history[:,1:],o[:,None]),1)
        a=self.pi(self.history,self.motion.goal(self.env.f,10))
        out=self.env.step(a)
        self.returns+=torch.where(alive,out["r"],0.)
        self.steps+=alive.long()
        newly_done=alive & self.env.done_eval
        self.final_qpos=torch.where(newly_done[:,None],self.env.sim.data.qpos,self.final_qpos)
        self.final_frame=torch.where(newly_done,self.env.f,self.final_frame)
        if self.record and bool(alive[self.focus]):
            S=self.env.state();k=self.focus
            self.records.append(dict(frame=int(before[k]),q=S[4][k].cpu().numpy().copy(),
                root_p=S[0][k].cpu().numpy().copy(),action=a[k].cpu().numpy().copy(),
                qpos=self.env.sim.data.qpos[k].cpu().numpy().copy()))
        self.env.f=torch.minimum(self.env.f,torch.full_like(self.env.f,self.b.T-2))
        self.ticks+=1
        self.finished=bool(self.env.done_eval.all())
        torch.cuda.synchronize(self.device)
        self.tick_ms.append((time.perf_counter()-t)*1000)
        return not self.finished
    def snapshot(self):
        k=self.focus;done=bool(self.env.done_eval[k])
        qpos=self.final_qpos[k] if done else self.env.sim.data.qpos[k]
        frame=self.final_frame[k] if done else self.env.f[k]
        return qpos.cpu().numpy().copy(),int(frame),done,bool(self.env.failed_eval[k])
    def report(self):
        from blinkmimic.mdp import MAX_STEP_REWARD
        norms=(self.returns/((self.b.T-1)*MAX_STEP_REWARD)).cpu().tolist()
        failed=self.env.failed_eval.cpu().tolist();done=self.env.done_eval.cpu().tolist()
        return dict(backend="mjlab/MuJoCo Warp",physics_device=str(self.env.sim.data.qpos.device),
                    policy_device=str(next(self.pi.parameters()).device),gpu_name=torch.cuda.get_device_name(self.device),
                    model="original BlinkMimic training robot and position actuators",
                    bundle_fingerprint=self.b.fingerprint,precision=self.precision,num_envs=self.N,
                    selected_env=self.focus,control_dt=.02,physics_dt=.005,steps=self.steps.cpu().tolist(),
                    simulated_seconds=self.ticks*.02,completed=all(done),n_failed=sum(failed),failed=failed,
                    passed=all(done) and not any(failed),normalized_returns=norms,norm_return=float(np.mean(norms)),
                    return_is_partial=not all(done),tick_p99_ms=float(np.percentile(self.tick_ms,99)) if self.tick_ms else None,
                    timing_scope="synchronized policy + GPU physics + bookkeeping; excludes initialization/rendering",
                    scope="training-environment GPU playback/evaluation; NOT official torque-motor G2, DDS, or hardware")
    def save(self,path):
        path=Path(path);path.mkdir(parents=True,exist_ok=True)
        r=self.report();(path/"report.json").write_text(json.dumps(r,indent=2)+"\n")
        if self.record and self.records:
            np.savez_compressed(path/"trajectory.npz",**{k:np.array([x[k] for x in self.records])
                               for k in ("frame","q","root_p","action","qpos")},
                               bundle_fingerprint=np.array(self.b.fingerprint),
                               complete=np.array(bool(self.env.done_eval[self.focus]) and not bool(self.env.failed_eval[self.focus])),
                               precision=np.array(self.precision))
        return r

class Display:
    """CPU rendering mirror only. No mj_step is called on this model."""
    def __init__(self,runner):
        import mujoco
        self.r=runner
        self.m=copy.copy(runner.env.sim.mj_model)
        self.d=mujoco.MjData(self.m);self.ref=mujoco.MjData(self.m)
        free=np.flatnonzero(self.m.jnt_type==mujoco.mjtJoint.mjJNT_FREE)
        if len(free)!=1:raise ValueError("one floating base expected in render model")
        self.fq=int(self.m.jnt_qposadr[free[0]])
        names=[self.m.joint(i).name.rsplit("/",1)[-1] for i in range(self.m.njnt)]
        self.qidx=[self.m.jnt_qposadr[names.index(n)] for n in runner.b.cfg["joint_names"]]
        self.sync()
    def sync(self):
        import mujoco
        self.qpos,self.frame,self.done,self.failed=self.r.snapshot()
        self.d.qpos[:]=self.qpos
        mujoco.mj_forward(self.m,self.d)  # rendering kinematics only, no physics integration
    def ghost(self,scene,visible=True):
        import mujoco
        scene.ngeom=0
        if not visible:return
        f=min(self.frame,self.r.b.T-1);b=self.r.b
        self.ref.qpos[:]=self.qpos
        self.ref.qpos[self.fq:self.fq+3]=b.motion["root_p"][f]
        mujoco.mju_mat2Quat(self.ref.qpos[self.fq+3:self.fq+7],b.motion["root_R"][f].reshape(9))
        self.ref.qpos[self.qidx]=b.motion["q"][f]
        mujoco.mj_kinematics(self.m,self.ref)
        for i in range(1,self.m.nbody):
            if scene.ngeom+2>scene.maxgeom:break
            pos=self.ref.xpos[i];parent=self.m.body_parentid[i]
            g=scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(g,mujoco.mjtGeom.mjGEOM_SPHERE,[.022,0,0],pos,np.eye(3).reshape(9),[.1,.7,1,.8]);scene.ngeom+=1
            if parent>0 and np.linalg.norm(pos-self.ref.xpos[parent])>1e-6:
                g=scene.geoms[scene.ngeom];mujoco.mjv_initGeom(g,mujoco.mjtGeom.mjGEOM_CAPSULE,[.01,0,0],[0,0,0],np.eye(3).reshape(9),[.1,.7,1,.6])
                mujoco.mjv_connector(g,mujoco.mjtGeom.mjGEOM_CAPSULE,.01,self.ref.xpos[parent],pos);scene.ngeom+=1
    def screenshot(self,path):
        import mujoco
        from PIL import Image
        self.sync();cam=mujoco.MjvCamera();mujoco.mjv_defaultCamera(cam)
        cam.lookat[:]=self.qpos[self.fq:self.fq+3];cam.distance=3.;cam.azimuth=135.;cam.elevation=-15.
        with mujoco.Renderer(self.m,height=480,width=640) as ren:
            ren.update_scene(self.d,camera=cam)
            ghost=mujoco.MjvScene(self.m,maxgeom=200);self.ghost(ghost)
            for i in range(ghost.ngeom):
                if ren.scene.ngeom>=ren.scene.maxgeom:break
                s=ghost.geoms[i];g=ren.scene.geoms[ren.scene.ngeom]
                mujoco.mjv_initGeom(g,s.type,s.size,s.pos,s.mat.reshape(9),s.rgba);ren.scene.ngeom+=1
            Image.fromarray(ren.render()).save(path)

def run(args,bundle):
    if args.max_seconds is not None and args.max_seconds<.02:raise ValueError("--max-seconds must be >= 0.02")
    r=WarpRunner(bundle,args.ckpt,args.device,args.num_envs,args.precision,args.view_env,args.record)
    limit=round(args.max_seconds/.02) if args.max_seconds is not None else bundle.T-1
    if not args.viewer:
        while not r.finished and r.ticks<limit:r.tick()
        if args.screenshot:Display(r).screenshot(args.out/"viewer.png")
    else:
        import mujoco
        from mujoco import viewer
        display=Display(r);keys=queue.SimpleQueue();paused=False;ghost=True;follow=True;speed=args.speed
        deadline=time.monotonic();failure_paused=False;dirty=True
        with viewer.launch_passive(display.m,display.d,key_callback=keys.put,show_left_ui=False,show_right_ui=False) as v:
            v.cam.distance=3.;v.cam.azimuth=135.;v.cam.elevation=-15.
            while v.is_running():
                single=False
                while not keys.empty():
                    k=keys.get()
                    if k==32:paused=not paused
                    elif k==78:single=True;paused=True
                    elif k==71:ghost=not ghost
                    elif k==70:follow=not follow
                    elif k in (61,334):speed=min(2.,speed*2)
                    elif k in (45,333):speed=max(.125,speed/2)
                    elif k==82:r.reset();paused=False;failure_paused=False;dirty=True
                if not r.finished and r.ticks<limit and (single or (not paused and time.monotonic()>=deadline)):
                    r.tick();dirty=True;deadline=time.monotonic()+.02/speed
                with v.lock():
                    if dirty:display.sync();dirty=False
                    display.ghost(v.user_scn,ghost)
                    if follow:v.cam.lookat[:]=display.qpos[display.fq:display.fq+3]
                if display.failed and not failure_paused:paused=True;failure_paused=True
                if r.finished or r.ticks>=limit:paused=True
                text=(f"GPU PHYSICS + GPU POLICY / {r.device} / {r.precision}\n"
                      f"Training model, 200Hz physics / 50Hz policy / NO DDS\n"
                      f"Replica {r.focus+1}/{r.N} | frame {display.frame}/{bundle.T-1} | {speed:g}x\n"
                      f"{'FAILED' if display.failed else 'DONE' if display.done else 'TRACKING'} / {'PAUSED' if paused else 'RUNNING'}\n"
                      "Blue skeleton: reference | SPACE pause | N step | R reset\n"
                      "+/- speed | G reference | F follow camera\n"
                      "Not the official G1 torque-motor sim2sim.")
                v.set_texts((None,mujoco.mjtGridPos.mjGRID_TOPLEFT,text,""));v.sync();time.sleep(.005)
    report=r.save(args.out);print(json.dumps(report,indent=2),flush=True)
    if not report["passed"]:raise SystemExit(1 if report["n_failed"] else 2)
