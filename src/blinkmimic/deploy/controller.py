"""Single-motion state machine shared by the DDS runner and deterministic tests."""
import time
import numpy as np
from .core import JOINTS, DeployCore, state_guard, limit_target, transform_imu, sensor_transform

PREP=(1<<5)|(1<<12)
ARM=(1<<0)|(1<<8)
PLAY=(1<<0)|(1<<10)
STOP=(1<<5)|(1<<9)

class Controller:
    def __init__(self,bundle,profile,stage,policy=None):
        self.b=bundle
        self.core=DeployCore(bundle,policy)
        self.mount=sensor_transform(profile)
        self.stage=dict(bundle.cfg["stages"][stage])
        self.stage["clamp_positions"]=False
        if stage=="ground":self.stage["torque_fraction"]=1.5
        # Shared simulator/hardware runtime defaults, following SuperMimic.
        self.limits=dict(bundle.cfg["limits"],state_age_s=.02,max_temperature=80.,
                         hold_s=1.,hold_tol=.1,start_max_dq=.5,start_tilt=0.1745329252,warn_temperature=70.,
                         policy_tilt=.35)
        self.limits.update(bundle.cfg.get("runner_limits",{}))
        self.reject_reason=""
        self.state="PASSIVE"
        self.reason=""
        self.since=0.
        self.stable_since=None
        self.prev_keys=0
        self.key_since=0.
        self.accepted=0
        self.keys_released=False
        self.from_q=None

    def damp(self,reason):
        if self.state!="DAMP":
            self.state="DAMP"; self.reason=str(reason)

    def transition(self,state,t):
        self.state=state; self.since=t; self.stable_since=None; self.reject_reason=""

    def update(self,s,now,bridge_fault=None):
        # STOP is immediate; other combinations require 60 ms stable and release before reuse.
        keys=s["keys"]
        if keys&STOP==STOP:
            self.damp("remote stop")
        if keys != self.prev_keys:
            self.key_since=now; self.accepted=0; self.prev_keys=keys
        if keys==0:
            self.keys_released=True
        event=0
        if self.keys_released and now-self.key_since>=0.06 and not self.accepted:
            event=keys; self.accepted=1
        fault=bridge_fault or state_guard(s,now,self.b,self.stage,self.limits,check_velocity=self.state in ("STANDBY_PRE","POLICY","STANDBY_POST"))
        if fault:
            self.damp(fault)
        if self.state=="DAMP":
            return self.damping(s)
        try:
            R,gyro=transform_imu(s["quat"],s["gyro"],self.mount)
            if self.state=="PASSIVE":
                if event&PREP==PREP:
                    target=self.b.motion["q"][0]
                    self.from_q=s["q"].copy()
                    self.core.align(R)
                    self.transition("INTERP",now)
                else:
                    if event:self.reject_reason="press 1 / L2+up to prepare first"
                    out=self.damping(s);out["state"]="PASSIVE";out["reason"]=""
                    return out
            f,_=self.core.reference()
            ref=self.core.heading@self.b.motion["root_R"][f]
            tilt=float(np.arccos(np.clip(np.dot(-R[2],-ref[2]),-1,1)))
            tilt_limit=self.limits["policy_tilt"] if self.state in ("STANDBY_PRE","POLICY","STANDBY_POST") else self.stage["max_tilt"]
            if tilt>tilt_limit:
                raise ValueError("reference tilt error")
            preparing=self.state in ("INTERP","HOLD")
            gain=self.b.cfg.get("preparation_gain_scale",1.0) if preparing else 1.0
            kp,kd=self.b.kp*gain,self.b.kd*np.sqrt(gain)
            result=dict(frame=f,phase=self.core.phase,action=np.zeros(29),inference_ms=0.)
            if self.state=="INTERP":
                u=min(1.,(now-self.since)/self.limits["interpolation_s"])
                u=u*u*(3-2*u)
                target=self.from_q+(self.b.motion["q"][0]-self.from_q)*u
                if u>=1:
                    self.transition("HOLD",now)
            elif self.state=="HOLD":
                target=self.b.motion["q"][0]
                if event&ARM==ARM:
                    if now-self.since<self.limits["hold_s"]:self.reject_reason="HOLD needs at least 1 second"
                    elif abs(s["q"]-target).max()>=self.limits["hold_tol"]:
                        j=int(np.argmax(abs(s["q"]-target)))
                        self.reject_reason=(f"not at start pose: {JOINTS[j]} "
                            f"error={abs(s['q'][j]-target[j]):.3f}rad "
                            f"limit={self.limits['hold_tol']:.3f}rad")
                    elif abs(s["dq"]).max()>=self.limits["start_max_dq"]:self.reject_reason="joint velocity not settled"
                    elif tilt>=self.limits["start_tilt"]:self.reject_reason="start tilt not settled"
                    elif np.max(s["temp"])>self.limits["warn_temperature"]:self.reject_reason="motor temperature too high to start"
                    else:self.transition("STANDBY_PRE",now)
                elif event&PLAY==PLAY:self.reject_reason="enter policy standby with 2 / R1+A first"
            else:
                stable=tilt<self.limits["standby_tilt"] and abs(s["dq"]).max()<self.limits["standby_dq"]
                if self.state=="STANDBY_PRE":
                    self.stable_since=(now if self.stable_since is None else self.stable_since) if stable else None
                    if event&PLAY==PLAY:
                        if self.stable_since is not None and now-self.stable_since>=self.limits["standby_s"]:
                            self.core.start_motion(); self.transition("POLICY",now)
                        else:self.reject_reason="standby not yet continuously stable"
                result=self.core.step(s["q"],s["dq"],gyro,R)
                if abs(result["action"]).max()>self.stage["max_action"]:
                    raise ValueError("action magnitude")
                target=result["target"]
                if self.core.phase=="post" and self.state=="POLICY":
                    self.transition("STANDBY_POST",now)
            requested_target=np.array(target,copy=True)
            target,n=limit_target(target,s["q"],s["dq"],self.b,self.stage,kp,kd)
            # Diagnostic PD estimate at this sensor snapshot, not measured motor torque.
            result.update(requested_target=requested_target,
                          requested_target_error=requested_target-s["q"],
                          pd_torque_estimate=kp*(target-s["q"])-kd*s["dq"],
                          torque_limit=self.b.effort*self.stage["torque_fraction"])
            result.update(target=target,kp=kp,kd=kd,n_clamped=n,tilt=tilt,state=self.state,reject_reason=self.reject_reason)
            if self.core.history is not None:
                result["observation"]=self.core.history[-1].copy()
                result["orientation6d"]=self.core.orientation6d.copy()
                if self.core.orientation6d_future is not None:
                    result["orientation6d_future"]=self.core.orientation6d_future.copy()
            return result
        except Exception as exc:
            self.damp(exc)
            return self.damping(s)

    def damping(self,s):
        # q is ignored by bridge's measured-position flag; sanitize before serialization.
        q=np.nan_to_num(s["q"],nan=0.,posinf=0.,neginf=0.)
        return dict(target=q,kp=np.zeros(29),kd=np.full(29,3.),state="DAMP",reason=self.reason,
                    frame=self.core.frame,action=np.zeros(29),n_clamped=0,inference_ms=0.)
