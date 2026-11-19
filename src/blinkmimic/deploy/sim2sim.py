"""Native MuJoCo sim2sim primitives. Runtime: numpy, mujoco, onnxruntime; no torch/DDS."""
from collections import deque
import hashlib
import json
from pathlib import Path
import time
import mujoco
import numpy as np
from .core import DeployCore, rotation, limit_target, state_guard
from .controller import Controller,PREP,ARM,PLAY

PROFILE=dict(imu_frame="pelvis_rigid",quaternion_order="wxyz",gyro_units="rad/s",R_pelvis_from_imu_wxyz=[1,0,0,0])

def model_fingerprint(m):
    h=hashlib.sha256()
    for key in ("body_mass","body_inertia","body_pos","body_quat","jnt_range","dof_armature",
                "dof_damping","dof_frictionloss","geom_pos","geom_quat","geom_size","geom_friction",
                "mesh_vert","mesh_face","actuator_trnid","actuator_gear","actuator_gainprm",
                "actuator_biasprm","actuator_ctrlrange","actuator_forcerange"):
        h.update(key.encode()); h.update(np.asarray(getattr(m,key)).tobytes())
    return h.hexdigest()

class Plant:
    def __init__(self,b,xml,kind,latency_ms=0.,kp_scale=1.,mass=0.,friction=None):
        self.b=b; self.kind=kind
        self.m=mujoco.MjModel.from_xml_path(str(xml)); self.d=mujoco.MjData(self.m)
        m=self.m
        self.xml=str(Path(xml).resolve())
        if kind not in ("g1","g2"):raise ValueError("unknown plant kind")
        self.dt=.005 if kind=="g1" else .001
        m.opt.timestep=self.dt
        self.substeps=round(.02/self.dt)
        self.delay_steps=round(latency_ms/(1000*self.dt))
        if abs(self.delay_steps*self.dt*1000-latency_ms)>1e-8:raise ValueError("latency must be an integer number of physics steps")
        self.jids=np.array([m.joint(n).id for n in b.cfg["joint_names"]])
        if len(set(self.jids))!=29:raise ValueError("invalid joint map")
        free=np.flatnonzero(m.jnt_type==mujoco.mjtJoint.mjJNT_FREE)
        if len(free)!=1:raise ValueError("one floating base required")
        self.fq=int(m.jnt_qposadr[free[0]]); self.fv=int(m.jnt_dofadr[free[0]])
        self.pelvis=int(m.jnt_bodyid[free[0]])
        if m.body(self.pelvis).name!="pelvis":raise ValueError("floating body must be pelvis")
        self.qidx=m.jnt_qposadr[self.jids];self.vidx=m.jnt_dofadr[self.jids]
        acts=[]
        for j in self.jids:
            ids=np.flatnonzero((m.actuator_trnid[:,0]==j)&(m.actuator_trntype==mujoco.mjtTrn.mjTRN_JOINT))
            if len(ids)!=1:raise ValueError("one actuator per joint required")
            acts.append(ids[0])
        self.act=np.array(acts)
        if not np.allclose(m.actuator_gear[self.act,0],1):raise ValueError("unit motor gear required")
        if kind!="g1":
            if not np.allclose(m.actuator_gainprm[self.act,0],1) or np.any(m.actuator_biastype[self.act]!=mujoco.mjtBias.mjBIAS_NONE):
                raise ValueError("official plant must use torque motors")
        else:
            if not np.allclose(m.actuator_gainprm[self.act,0],b.kp,rtol=1e-4):
                raise ValueError("training model PD mismatch")
        if mass:
            body=m.body("torso_link").id; old=m.body_mass[body]
            m.body_mass[body]+=mass;m.body_inertia[body]*=(old+mass)/old
        if friction is not None:m.geom_friction[:,0]=friction
        if mass:mujoco.mj_setConst(m,self.d)
        self.kp_scale=kp_scale
        self.hash=model_fingerprint(m)
        self.pending=deque();self.steps=0
        self.target=b.motion["q"][0].copy();self.kp=b.kp.copy();self.kd=b.kd.copy()
        self.torque_saturated=0;self.model_saturated=0;self.max_torque_ratio=0.
        self.max_velocity_ratio=0.;self.ground_shift=0.
        self.reset(reference_velocity=(kind=="g1"))
    def reset(self,reference_velocity=False):
        m,d=self.m,self.d;b=self.b
        mujoco.mj_resetData(m,d)
        d.qpos[self.fq:self.fq+3]=b.motion["root_p"][0]
        mujoco.mju_mat2Quat(d.qpos[self.fq+3:self.fq+7],b.motion["root_R"][0].reshape(9))
        d.qpos[self.qidx]=np.clip(b.motion["q"][0],b.lower,b.upper)
        if reference_velocity:
            required=("qd","root_v","root_w")
            if any(k not in b.motion for k in required):raise ValueError("re-export bundle to include reference velocities for g1")
            for k,width in (("qd",29),("root_v",3),("root_w",3)):
                if b.motion[k].shape!=(b.T,width) or not np.isfinite(b.motion[k]).all():
                    raise ValueError("invalid reference velocity "+k)
            d.qvel[self.vidx]=b.motion["qd"][0]
            d.qvel[self.fv:self.fv+3]=b.motion["root_v"][0]
            d.qvel[self.fv+3:self.fv+6]=b.motion["root_R"][0].T@b.motion["root_w"][0]
        mujoco.mj_forward(m,d)
        if self.kind!="g1":
            lows=[]
            for i in range(m.ngeom):
                if not (m.geom_contype[i] or m.geom_conaffinity[i]):continue
                if "ankle_roll" not in m.body(m.geom_bodyid[i]).name:continue
                if m.geom_type[i]==mujoco.mjtGeom.mjGEOM_SPHERE:
                    lows.append(d.geom_xpos[i,2]-m.geom_size[i,0])
                elif m.geom_type[i]==mujoco.mjtGeom.mjGEOM_CAPSULE:
                    axis=d.geom_xmat[i].reshape(3,3)[:,2]*m.geom_size[i,1]
                    lows.append(d.geom_xpos[i,2]-abs(axis[2])-m.geom_size[i,0])
            if not lows:raise ValueError("no supported foot collision geometry for ground placement")
            self.ground_shift=-float(min(lows))
            d.qpos[self.fq+2]+=self.ground_shift;mujoco.mj_forward(m,d)
        self.target=d.qpos[self.qidx].copy()
        if self.kind=="g1":d.ctrl[self.act]=self.target
    def state(self):
        d=self.d
        return (d.qpos[self.qidx].copy(),d.qvel[self.vidx].copy(),
                d.qvel[self.fv+3:self.fv+6].copy(),d.qpos[self.fq+3:self.fq+7].copy())
    def command(self,target,kp,kd):
        self.pending.append((self.steps+self.delay_steps,np.array(target).copy(),
                             np.array(kp)*self.kp_scale,np.array(kd).copy()))
    def step(self,support=False):
        for _ in range(self.substeps):
            while self.pending and self.pending[0][0]<=self.steps:
                _,self.target,self.kp,self.kd=self.pending.popleft()
            q,dq,g,quat=self.state();m,d,b=self.m,self.d,self.b
            d.xfrc_applied[:]=0
            if support:
                R=rotation(quat);E=b.motion["root_R"][0]@R.T
                err=np.array([E[2,1]-E[1,2],E[0,2]-E[2,0],E[1,0]-E[0,1]])*.5
                anchor=b.motion["root_p"][0].copy();anchor[2]+=self.ground_shift
                d.xfrc_applied[self.pelvis,:3]=2000*(anchor-d.qpos[self.fq:self.fq+3])-100*d.qvel[self.fv:self.fv+3]+[0,0,m.body_mass.sum()*9.81]
                d.xfrc_applied[self.pelvis,3:]=500*err-50*(R@g)
            desired=self.kp*(self.target-q)-self.kd*dq
            self.max_torque_ratio=max(self.max_torque_ratio,float(np.max(abs(desired)/b.effort)))
            self.max_velocity_ratio=max(self.max_velocity_ratio,float(np.max(abs(dq)/b.velocity)))
            self.torque_saturated+=int(np.count_nonzero(abs(desired)>b.effort))
            if self.kind=="g1":
                m.actuator_gainprm[self.act,0]=self.kp
                m.actuator_biasprm[self.act,1]=-self.kp;m.actuator_biasprm[self.act,2]=-self.kd
                d.ctrl[self.act]=self.target
            else:
                tau=np.clip(desired,-b.effort,b.effort)
                limited=m.actuator_ctrllimited[self.act].astype(bool)
                bounds=m.actuator_ctrlrange[self.act]
                self.model_saturated+=int(np.count_nonzero(limited&((tau<bounds[:,0])|(tau>bounds[:,1]))))
                tau[limited]=np.clip(tau[limited],bounds[limited,0],bounds[limited,1])
                d.ctrl[self.act]=tau
            mujoco.mj_step(m,d);self.steps+=1

class Experiment:
    def __init__(self,b,plant,cmd,stage="ground",diagnostic=False,standby_s=5.):
        self.b=b;self.plant=plant;self.cmd=cmd;self.stage=b.cfg["stages"][stage]
        self.diagnostic=diagnostic;self.standby_s=standby_s
        self.controller=Controller(b,PROFILE,stage) if cmd=="standby" else None
        self.core=self.controller.core if self.controller else DeployCore(b)
        if not self.controller:self.core.align(rotation(plant.state()[3]));self.core.start_motion()
        self.time=0.;self.frame=0;self.state="PASSIVE" if self.controller else "POLICY"
        self.events=[];self.records=[];self.finished=False;self.failed=False;self.support=False
        self.sent=set();self.keys=0;self.keys_until=0.;self.post_started=None;self.stop_reason=""
    def event(self,reason):
        if not any(e["reason"]==reason for e in self.events):
            self.events.append(dict(frame=self.frame,sim_time=self.time,reason=reason))
    def tick(self):
        if self.finished:return False
        p=self.plant;b=self.b;q,dq,g,quat=p.state();R=rotation(quat)
        s=dict(q=q,dq=dq,gyro=g,quat=quat,temp=np.full((29,2),30.),motorstate=np.zeros(29),
               mode_machine=5,mode_pr=0,t_rx=self.time,t_tick_change=self.time,keys=0)
        out=None
        try:
            if self.controller:
                c=self.controller;self.support=c.state in ("PASSIVE","INTERP","HOLD")
                if self.time>=self.keys_until:self.keys=0
                def press(tag,key):
                    if tag not in self.sent:self.keys=key;self.keys_until=self.time+.16;self.sent.add(tag)
                if c.state=="PASSIVE" and self.time>.1:press("prep",PREP)
                if c.state=="HOLD" and self.time-c.since>.5:press("arm",ARM)
                if c.state=="STANDBY_PRE":
                    elapsed=self.time-c.since;self.support=elapsed<1.
                    # A fresh PLAY request each second after minimum settling; actual stability gate remains in Controller.
                    if elapsed>self.standby_s and self.keys==0:press("play"+str(int(elapsed)),PLAY)
                    if elapsed>self.standby_s+15:raise ValueError("standby did not become stable")
                s["keys"]=self.keys;out=c.update(s,self.time);self.state=c.state
                if c.state=="DAMP":raise ValueError(c.reason)
            else:
                self.frame=self.core.frame
                # G1 is a numerical baseline: no deployment target clipping or deployment aborts.
                if self.cmd!="g1":
                    fault=state_guard(s,self.time,b,self.stage,b.cfg["limits"])
                    if fault:
                        self.event(fault);self.failed=True
                        if fault=="joint outside hard limits":
                            j=int(np.argmax(np.maximum(b.lower-q,q-b.upper)))
                            self.event(f"{b.cfg['joint_names'][j]}={q[j]:.6f} limits=[{b.lower[j]:.6f},{b.upper[j]:.6f}]")
                        if fault=="joint overspeed":
                            j=int(np.argmax(abs(dq)/b.velocity))
                            self.event(f"{b.cfg['joint_names'][j]} velocity={dq[j]:.6f}")
                        if not self.diagnostic:raise ValueError(fault)
                out=self.core.step(q,dq,g,R)
                target=out["target"]
                if self.cmd!="g1":
                    if abs(out["action"]).max()>self.stage["max_action"]:raise ValueError("action magnitude")
                    target,n=limit_target(target,q,dq,b,self.stage)
                out.update(target=target,kp=b.kp,kd=b.kd)
            if out is not None:
                self.frame=out["frame"]
                p.command(out["target"],out["kp"],out["kd"])
                p.step(self.support)
                q1,dq1,g1,quat1=p.state();f1=min(self.frame+1,b.T-1) if self.core.phase!="pre" else 0
                expected=b.motion["root_p"][f1].copy();expected[2]+=p.ground_shift
                height=abs(float(p.d.qpos[p.fq+2]-expected[2]))
                tilt=abs(float(rotation(quat1)[2,2]-b.motion["root_R"][f1,2,2]))
                # Exact BlinkMimic training termination constants are recorded separately from deployment aborts.
                if height>.25 or tilt>.8:raise ValueError("training height/tilt termination")
                self.records.append(dict(frame=self.frame,state=self.state,qpos=p.d.qpos.copy(),
                    q=q1,dq=dq1,action=out["action"].copy(),target=out["target"].copy(),
                    ref_q=b.motion["q"][f1].copy(),root_error=float(np.linalg.norm(p.d.qpos[p.fq:p.fq+3]-expected)),
                    inference_ms=out["inference_ms"]))
                if self.core.phase=="post":
                    if not self.controller:self.finished=True;self.state="COMPLETE"
                    else:
                        if self.post_started is None:self.post_started=self.time
                        if self.time-self.post_started>=self.standby_s:self.finished=True
            self.time+=.02
        except Exception as exc:
            self.failed=True;self.finished=True;self.stop_reason=str(exc);self.event(str(exc))
            if self.controller:self.controller.damp(str(exc));self.state="DAMP"
        return not self.finished
    def report(self):
        rec=self.records
        rmse=float(np.sqrt(np.mean([(r["q"]-r["ref_q"])**2 for r in rec]))) if rec else None
        return dict(bundle_fingerprint=self.b.fingerprint,command=self.cmd,plant_xml=self.plant.xml,
                    model_fingerprint=self.plant.hash,physics_dt=self.plant.dt,control_dt=.02,
                    ground_shift=self.plant.ground_shift,sim_seconds=self.time,control_steps=len(rec),
                    completed=self.finished and not self.failed,passed=self.finished and not self.failed,
                    diagnostic_continue=self.diagnostic,state=self.state,stop_reason=self.stop_reason,events=self.events,
                    joint_tracking_rmse_rad=rmse,root_position_error_max_m=max((r["root_error"] for r in rec),default=None),
                    inference_p99_ms=float(np.percentile([r["inference_ms"] for r in rec],99)) if rec else None,
                    torque_saturation_samples=self.plant.torque_saturated,
                    model_ctrl_saturation_samples=self.plant.model_saturated,
                    max_requested_torque_ratio=self.plant.max_torque_ratio,max_velocity_ratio=self.plant.max_velocity_ratio,
                    scope="CPU MuJoCo sim2sim, no DDS or hardware; not the 40-clip training return")
    def save(self,path,extra=None):
        path=Path(path);path.mkdir(parents=True,exist_ok=True)
        report=self.report();report.update(extra or {})
        (path/"report.json").write_text(json.dumps(report,indent=2)+"\n")
        if self.records:
            np.savez_compressed(path/"trajectory.npz",**{k:np.array([r[k] for r in self.records])
                                 for k in ("frame","state","qpos","q","dq","action","target","inference_ms")})
        return report
