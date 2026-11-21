"""BlinkMimic single-motion runner. No implicit robot interface or DDS domain."""
import argparse
import json
import signal
import sys
import time
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent/"bridge"))
import g1_bridge_io as gb
from blinkmimic.deploy.core import Bundle, STAGES
from blinkmimic.deploy.controller import Controller
from blinkmimic.deploy.lowstate import decode_snapshot
from blinkmimic.deploy.snapshot import read_snapshot
from blinkmimic.deploy.session import DEFAULT_PROFILE,new_session,atomic_json

def plain(x):
    if isinstance(x,np.ndarray): return x.tolist()
    if isinstance(x,np.generic): return x.item()
    raise TypeError(str(type(x)))

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bundle",type=Path,required=True)
    ap.add_argument("--hardware-profile",type=Path,help="optional IMU transform overrides")
    ap.add_argument("--iface",required=True)
    ap.add_argument("--domain",type=int,required=True)
    ap.add_argument("--stage",choices=list(STAGES),required=True)
    ap.add_argument("--log-dir",type=Path,default=Path("deploy_logs/real"))
    ap.add_argument("--status-file",type=Path,help="optional UI status output")
    ap.add_argument("--sim",action="store_true")
    ap.add_argument("--read-only",action="store_true")
    ap.add_argument("--seconds",type=float,default=10.,help="read-only duration")
    ap.add_argument("--inject-fault",choices=["nan","timeout"],help="SIM ONLY, inject at policy call 10")
    a=ap.parse_args()
    if a.inject_fault and not a.sim: ap.error("fault injection is simulation only")
    if a.sim and (a.domain not in (1,2) or a.iface!="lo"):
        ap.error("simulation requires domain 1 or 2 and lo")
    if not a.sim and (a.domain!=0 or a.iface=="lo"):
        ap.error("hardware requires explicit domain 0 and robot NIC")
    b=Bundle(a.bundle)
    profile=dict(DEFAULT_PROFILE)
    if a.hardware_profile:profile.update(json.loads(a.hardware_profile.read_text()))
    profile.pop("verified_on_this_robot",None)
    ctl=Controller(b,profile,a.stage)
    if a.inject_fault:
        original=ctl.core.policy
        calls=[0]
        def injected(h,g):
            calls[0]+=1
            if calls[0]>=10:
                if a.inject_fault=="nan": return np.full(29,np.nan)
                time.sleep(.04)
            return original(h,g)
        ctl.core.policy=injected
    if not a.sim and not a.read_only:
        if input("Robot supported / area clear / L2+B ready? Type yes: ").strip()!="yes":return
    a.log_dir=new_session(a.log_dir.resolve())
    print(f"Runner logs: {a.log_dir}",flush=True)
    log=open(a.log_dir/"runner.jsonl","w",buffering=1)
    log.write(json.dumps(dict(event="start",bundle_fingerprint=b.fingerprint,config=b.cfg,
                              temperature_channels=["casing","winding"],
                              temperature_channel_source="unitreerobotics/unitree_sdk2 include/unitree/robot/g1/common/terminations.hpp",
                              torque_log_semantics={"s.tau":"LowState tau_est, Nm",
                                  "result.pd_torque_estimate":"PD estimate at sampled q/dq; not measured torque"},
                              profile=profile,effective_limits=ctl.limits,effective_stage=ctl.stage,arguments={k:str(v) for k,v in vars(a).items()}))+"\n")
    stop=[False]
    for sig in (signal.SIGINT,signal.SIGTERM):
        signal.signal(sig,lambda *_: stop.__setitem__(0,True))
    bridge=gb.Bridge(a.domain,a.iface,True,500.,100.,3.,a.log_dir/"bridge.log",listen_only=a.read_only)
    snap=np.zeros(1,gb.SNAP_DTYPE); status=np.zeros(1,gb.STATUS_DTYPE)
    q=np.zeros(35); kp=q.copy(); kd=q.copy(); modes=bytes([1]*29+[0]*6)
    began=time.monotonic(); deadline=began; damp_since=None; last_pub=last_lc=0; excess=0
    observed=False
    last_rx=None
    command_started=False
    try:
        # Observe other publishers for 1 second before allowing any active command.
        while time.monotonic()-began<1.0 and not stop[0]:
            bridge.read_status(status)
            if status["lc_count"][0] and not a.read_only:
                raise RuntimeError("another lowcmd publisher present")
            time.sleep(.02)
        deadline=time.monotonic()
        while bridge.alive():
            now=time.monotonic()
            bridge.read_status(status); bs=status[0]
            got=read_snapshot(bridge,snap); r=snap[0]
            if not got:
                # Startup wait and a transient seqlock collision are different cases.
                elapsed=time.monotonic()
                if not observed and elapsed-began>3:
                    raise RuntimeError("no initial lowstate")
                if observed and elapsed-last_rx>ctl.limits["state_age_s"]:
                    raise RuntimeError("lowstate snapshot unavailable within freshness deadline")
                if stop[0]:raise RuntimeError("operator exit")
                time.sleep(.001); continue
            observed=True
            s=decode_snapshot(r)
            last_rx=s["t_rx"]
            if a.read_only:
                log.write(json.dumps(dict(event="state",t=now,s=s,other_lowcmd=int(bs["lc_count"])),default=plain)+"\n")
                if stop[0] or now-began>=a.seconds: break
                time.sleep(.02); continue
            # Bridge's own publications are counted too; compare accumulated counts with a burst allowance.
            last_pub=int(bs["n_pub"]); last_lc=int(bs["lc_count"])
            excess=max(excess,last_lc-last_pub)
            fault=None
            if excess>5 or (not bs["pub_started"] and last_lc):
                fault="foreign lowcmd publisher"
            if bs["wd_fired"] or bs["fail_consec"] or now-float(bs["t_hb"])>.05:
                fault="bridge watchdog / publication fault"
            if stop[0]: ctl.damp("operator exit")
            if ctl.state!="PASSIVE" and now-deadline>.04:
                fault="control loop overrun"
            controller_started=time.monotonic()
            out=ctl.update(s,now,fault)
            controller_ms=(time.monotonic()-controller_started)*1000
            policy_inference_ms=out.get("inference_ms") if out is not None else None
            if ctl.state=="DAMP" and not command_started:
                raise RuntimeError("refused before first command: "+ctl.reason)
            if out is not None:
                # Reject results whose sensor sample became stale during inference.
                age=time.monotonic()-s["t_rx"]
                if age>ctl.limits["state_age_s"]:
                    ctl.damp("state expired during policy"); out=ctl.damping(s)
                q[:29]=out["target"]; kp[:29]=out["kp"]; kd[:29]=out["kd"]
                flags=gb.CMD_NOW | (gb.CMD_Q_MEASURED if ctl.state=="DAMP" else 0)
                if not bridge.send(q,kp,kd,modes,5,flags):
                    ctl.damp("bridge send failed")
                else:
                    command_started=True
                log.write(json.dumps(dict(t=now,state=ctl.state,reason=ctl.reason,s=s,result=out,
                                          state_age_s=age,policy_inference_ms=policy_inference_ms,
                                          controller_ms=controller_ms,
                                          sample_age_before_control_ms=(controller_started-s["t_rx"])*1000),default=plain)+"\n")
            if a.status_file:
                atomic_json(a.status_file,dict(t=now,state=ctl.state,frame=ctl.core.frame,phase=ctl.core.phase,
                    reason=ctl.reason,reject_reason=ctl.reject_reason,keys=s["keys"],
                    heading=ctl.core.heading.tolist() if ctl.core.heading is not None else None,log_dir=str(a.log_dir)))
            if ctl.state=="DAMP":
                damp_since=now if damp_since is None else damp_since
                if now-damp_since>=10.: break
            deadline=max(deadline+.02,time.monotonic())
            time.sleep(max(0,deadline-time.monotonic()))
    except BaseException as exc:
        ctl.damp(str(exc))
        fault_time=time.monotonic()
        log.write(json.dumps(dict(event="exception",t=fault_time,state="DAMP",
                                 frame=ctl.core.frame,reason=str(exc)))+"\n")
        if a.status_file:
            atomic_json(a.status_file,dict(t=fault_time,state="DAMP",frame=ctl.core.frame,phase=ctl.core.phase,
                reason=ctl.reason,reject_reason="",keys=0,
                heading=ctl.core.heading.tolist() if ctl.core.heading is not None else None,log_dir=str(a.log_dir)))
        if not a.read_only and command_started and bridge.alive():
            q.fill(0); kp.fill(0); kd[:29]=3.
            until=time.monotonic()+10.
            while time.monotonic()<until and bridge.alive():
                bridge.send(q,kp,kd,modes,5,gb.CMD_NOW|gb.CMD_Q_MEASURED)
                time.sleep(.02)
        raise
    finally:
        bridge.quit()
        log.write(json.dumps(dict(event="end",t=time.monotonic(),state=ctl.state,frame=ctl.core.frame,reason=ctl.reason))+"\n")
        log.close()
if __name__=="__main__":
    main()
