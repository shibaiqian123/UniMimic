"""Live SONIC input adapter around BlinkMimic's shared DDS controller.
Only LowState supplies robot observations. No simulator-state dependency.
"""
import json,time
import numpy as np
from .core import JOINTS,OnnxPolicy,DeployCore,rotation,yaw,yaw_rotation,state_guard,transform_imu,limit_target
from .controller import Controller,STOP
from .sonic_wire import ISAAC_NAMES

def matrices(quats):
    return np.stack([rotation(q) for q in np.asarray(quats).reshape(-1,4)])

def inputs(message,bundle,R,heading,pose_heading):
    mode=message["mode"];m=message["planner"]
    qpos=np.asarray(message["qpos"],float);qd=np.asarray(message["qd"],float)
    if qpos.shape!=(46,36) or qd.shape!=(46,29):raise ValueError("invalid planner reference window")
    q=qpos[:,7:].copy();qd=qd.copy()
    if mode=="PLANNER_FROZEN_UPPER":
        upper=[JOINTS.index(n) for n in ISAAC_NAMES if n in JOINTS[12:]]
        q[:,upper]=np.asarray(m["upper_body_position"],float);qd[:,upper]=0.
    ids=np.arange(10)*5
    refR=matrices(qpos[ids,3:7])
    rel=(R.T@heading@refR)[...,:2].reshape(10,6)
    if mode=="POSE":
        p=message["pose"]
        if p is None:raise ValueError("waiting for 10 consecutive 50 Hz SMPL frames")
        local=np.asarray(p["smpl"],float).reshape(10,72)
        roots=matrices(p["quat"])
        ori=(R.T@pose_heading@roots)[...,:2].reshape(10,6)
        # SONIC wire joint_pos is IsaacLab; trained wrist feature order is explicit.
        index=[ISAAC_NAMES.index(JOINTS[i]) for i in bundle.cfg["omni_wrists"]]
        wrist=np.asarray(p["joints"],float)[:,index]
        x=np.concatenate((local,ori,wrist),axis=-1).reshape(-1)
        return "smpl",x,pose_heading@roots[0],dict(smpl_local=local.reshape(10,24,3)[0].tolist())
    if mode=="VR_3PT":
        points=np.asarray(m["vr_position"],float).reshape(3,3)
        rots=matrices(np.asarray(m["vr_orientation"],float).reshape(3,4))
        low=bundle.cfg["omni_lower"]
        x=np.concatenate((q[ids][:,low].reshape(-1),qd[ids][:,low].reshape(-1),
                          points.reshape(-1),rots[...,:2].reshape(-1),rel[0]))
        return "teleop",x,heading@refR[0],dict(vr_position=points.tolist(),vr_rotation=rots.tolist())
    qv=np.concatenate((q[ids].reshape(-1),qd[ids].reshape(-1))).reshape(10,58)
    x=np.concatenate((qv,rel),axis=-1).reshape(-1)
    return "motion",x,heading@refR[0],dict(reference_q=q[0].tolist())

class LiveCore(DeployCore):
    def __init__(self,bundle,policies):
        super().__init__(bundle,policies["motion"])
        self.policies=policies;self.message=None;self.pose_heading=None;self.robot_yaw=None
        self.live_mode="OFF";self.visual={};self.previous_modality=None;self.last_reference=None;self.pose_delta_base=0.
    def align(self,R):
        self.robot_yaw=yaw(R);self.heading=yaw_rotation(self.robot_yaw)
    def reference(self):return 0,self.b.motion["static_goal"][0]
    def step(self,q,dq,gyro,R):
        # Align source coordinate frames at mode edges, preserving the last reference
        # heading (including tracking error), never resetting to measured yaw each tick.
        mode=self.message["mode"];delta=float(self.message.get("heading_delta",0.))
        desired=self.last_reference if self.last_reference is not None else self.heading
        if mode=="POSE" and self.previous_modality!="smpl":
            root=rotation(self.message["pose"]["quat"][0])
            self.pose_heading=yaw_rotation(yaw(desired)-yaw(root))
            self.pose_delta_base=delta
        elif mode!="POSE" and self.previous_modality=="smpl":
            root=rotation(self.message["qpos"][0][3:7])
            self.heading=yaw_rotation(yaw(desired)-yaw(root))
        pose_heading=yaw_rotation(delta-self.pose_delta_base) @ (
            self.pose_heading if self.pose_heading is not None else self.heading)
        modality,x,ref,visual=inputs(self.message,self.b,R,self.heading,pose_heading)
        self.previous_modality=modality;self.last_reference=ref.copy()
        o=np.concatenate((q-self.b.q0,dq,gyro,-R[2],self.last_action))
        if o.shape!=(93,) or not np.isfinite(o).all() or not np.isfinite(x).all():raise ValueError("nonfinite live features")
        self.history=np.repeat(o[None],10,axis=0) if self.history is None else np.concatenate((self.history[1:],o[None]))
        self.orientation6d=(R.T@ref)[:,:2].reshape(6)
        t=time.perf_counter();cpu=time.thread_time()
        a=np.asarray(self.policies[modality](self.history,x),float)
        ms=(time.perf_counter()-t)*1000;cpu_ms=(time.thread_time()-cpu)*1000
        if a.shape!=(29,) or not np.isfinite(a).all():raise ValueError("invalid live action")
        self.last_action=a.copy();self.frame+=1;self.phase="motion";self.live_mode=self.message["mode"];self.visual=dict(visual,reference_R=ref.tolist())
        return dict(action=a,target=self.b.q0+self.b.scale*a,frame=self.frame,phase=self.phase,
                    inference_ms=ms,inference_thread_cpu_ms=cpu_ms,modality=modality,live_mode=self.live_mode,
                    reference_R=ref,features=x,pose_interval_s=self.message.get("pose_interval_s",0.),
                    pose_padded_frames=self.message.get("pose_padded_frames",0),
                    pose_delay_s=self.message.get("pose_delay_s",0.),
                    pose_received_time=self.message.get("pose_received_time",0.),
                    pose_sensor_time=self.message.get("pose_sensor_time"),
                    switch_wait_s=self.message.get("switch_wait_s",0.),requested_mode=self.message.get("requested_mode"),source_age_s=time.monotonic()-self.message["source_time"])

class ReferenceClient:
    def __init__(self,endpoint,feedback_bind):
        import zmq
        self.zmq=zmq;self.ctx=zmq.Context()
        self.sub=self.ctx.socket(zmq.SUB);self.sub.setsockopt(zmq.CONFLATE,1)
        self.sub.setsockopt_string(zmq.SUBSCRIBE,"reference");self.sub.connect(endpoint)
        self.pub=self.ctx.socket(zmq.PUB);self.pub.setsockopt(zmq.SNDHWM,2);self.pub.bind(feedback_bind)
        self.value=None;self.index=0
    def poll(self):
        try:self.value=json.loads(self.sub.recv(self.zmq.NOBLOCK)[len(b"reference"):])
        except self.zmq.Again:pass
        return self.value
    def feedback(self,s,out):
        import msgpack
        # Same topic and MuJoCo/motor order as SONIC; hand hardware not part of BlinkMimic 29DoF.
        x=dict(control_loop_type="blinkmimic",index=self.index,body_q_measured=s["q"].tolist(),
               body_q=s["q"].tolist(),body_dq=s["dq"].tolist(),base_quat=s["quat"].tolist(),
               base_quat_measured=s["quat"].tolist(),base_ang_vel=s["gyro"].tolist(),
               left_hand_q_measured=[0.]*7,right_hand_q_measured=[0.]*7,
               hands_available=False,body_q_target=np.asarray(out["target"]).tolist())
        self.index+=1
        try:self.pub.send(b"g1_debug"+msgpack.packb(x,use_bin_type=True),self.zmq.NOBLOCK)
        except self.zmq.Again:pass
    def close(self):
        self.sub.close(0);self.pub.close(0);self.ctx.term()

class ModeTransition:
    """Blend a mode's target from the last sent target; never alter raw action history."""
    def __init__(self,duration=.5):
        self.duration=duration;self.mode=None;self.last=None;self.seed=None;self.since=0.
    def apply(self,target,mode,now):
        target=np.asarray(target,float)
        if mode!=self.mode:
            self.seed=target.copy() if self.last is None else self.last.copy()
            self.since=now;self.mode=mode
        u=np.clip((now-self.since)/self.duration,0.,1.)
        weight=float(u*u*(3-2*u))
        out=self.seed*(1-weight)+target*weight
        self.last=out.copy()
        return out,weight

class LiveController(Controller):
    def __init__(self,bundle,profile,stage,client,device="cpu"):
        policies={m:OnnxPolicy(bundle,device,m) for m in ("motion","smpl","teleop")}
        super().__init__(bundle,profile,stage,policies["motion"])
        if bundle.cfg["schema"]!="blinkmimic_single_motion_v4":raise ValueError("VR requires trained omni v4 bundle")
        self.core=LiveCore(bundle,policies);self.client=client;self.starts=0;self.stops=0
        self.live_timeout=1.;self.live_mode="OFF";self.target_transition=ModeTransition()
    def update(self,s,now,bridge_fault=None):
        message=self.client.poll()
        if s["keys"]&STOP==STOP:self.damp("remote stop")
        if message is not None and message["stop"]>self.stops:
            self.stops=message["stop"];self.damp("SONIC remote stop")
        fault=bridge_fault or state_guard(s,now,self.b,self.stage,self.limits,
                                        check_velocity=self.state=="POLICY")
        if fault:self.damp(fault)
        if self.state=="DAMP":return self.damping(s)
        try:
            R,gyro=transform_imu(s["quat"],s["gyro"],self.mount)
            if self.state=="PASSIVE":
                if message is None or message["start"]<=self.starts:
                    out=self.damping(s);out.update(state="PASSIVE",reason="")
                    return out
                self.starts=message["start"];self.from_q=s["q"].copy();self.core.align(R)
                self.transition("INTERP",now)
            if message is None or now-message["t"]>self.live_timeout:raise ValueError("VR source process stale")
            if message["error"]:raise ValueError(message["error"])
            # Stream loss is latched through the same DAMP path as LowState failures.
            if now-message["source_time"]>self.live_timeout:raise ValueError("VR input stream stale")
            if self.state in ("INTERP","HOLD"):
                u=min(1.,(now-self.since)/self.limits["interpolation_s"]) if self.state=="INTERP" else 1.
                target=self.from_q+(self.b.q0-self.from_q)*(u*u*(3-2*u))
                if self.state=="INTERP" and u>=1:self.transition("HOLD",now)
                if self.state=="HOLD" and now-self.since>=1.:
                    self.transition("POLICY",now)
                gain=self.b.cfg.get("preparation_gain_scale",1.)
                kp=self.b.kp*gain;kd=self.b.kd*np.sqrt(gain)
                out=dict(action=np.zeros(29),frame=0,phase="pre",inference_ms=0.)
            else:
                if now>message["plan_end"]-.04 and message["mode"]!="POSE":raise ValueError("planner reference expired")
                if message["mode"]=="POSE" and message["pose"] is None:
                    raise ValueError("POSE has no complete 10-frame window")
                self.core.message=message
                out=self.core.step(s["q"],s["dq"],gyro,R)
                ref=out["reference_R"]
                tilt=float(np.arccos(np.clip(np.dot(R[2],ref[2]),-1,1)))
                if tilt>self.limits["policy_tilt"]:raise ValueError("reference tilt error")
                if abs(out["action"]).max()>self.stage["max_action"]:raise ValueError("action magnitude")
                raw_target=out["target"].copy()
                target,weight=self.target_transition.apply(raw_target,message["mode"],now)
                out.update(raw_policy_target=raw_target,transition_weight=weight,transition_duration_s=self.target_transition.duration)
                kp=self.b.kp;kd=self.b.kd
            requested=np.array(target,copy=True)
            target,n=limit_target(target,s["q"],s["dq"],self.b,self.stage,kp,kd)
            self.target_transition.last=np.array(target,copy=True)
            self.live_mode=message["mode"]
            out.update(target=target,kp=kp,kd=kd,n_clamped=n,state=self.state,live_mode=self.live_mode,
                       reject_reason=self.reject_reason,requested_target=requested,
                       pd_torque_estimate=kp*(target-s["q"])-kd*s["dq"])
            if self.core.history is not None:out["observation"]=self.core.history[-1].copy()
            return out
        except Exception as e:
            self.damp(str(e));return self.damping(s)
