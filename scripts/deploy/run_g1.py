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

def plain(x):
    if isinstance(x,np.ndarray): return x.tolist()
    if isinstance(x,np.generic): return x.item()
    raise TypeError(str(type(x)))

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bundle",type=Path,required=True)
    ap.add_argument("--hardware-profile",type=Path,required=True)
    ap.add_argument("--iface",required=True)
    ap.add_argument("--domain",type=int,required=True)
    ap.add_argument("--stage",choices=list(STAGES),required=True)
    ap.add_argument("--log-dir",type=Path,required=True)
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
    profile=json.loads(a.hardware_profile.read_text())
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
        if not profile.get("verified_on_this_robot",False):
            ap.error("hardware profile is not verified")
        for filename in ("export_validation.json","observation_validation.json","sim_validation.json","dds_validation.json","dds_physics_validation.json"):
            p=b.path/filename
            if not p.exists():
                ap.error("missing acceptance report: "+filename)
            report=json.loads(p.read_text())
            if report.get("bundle_fingerprint")!=b.fingerprint or not report.get("passed"):
                ap.error("failed or mismatched acceptance report: "+filename)
        if input("G1 supported, remote stop ready, correct robot/NIC/profile. Type ARM: ")!="ARM":
            return
    a.log_dir.mkdir(parents=True,exist_ok=False)
    log=open(a.log_dir/"runner.jsonl","w",buffering=1)
    log.write(json.dumps(dict(event="start",bundle_fingerprint=b.fingerprint,config=b.cfg,
                              profile=profile,arguments={k:str(v) for k,v in vars(a).items()}))+"\n")
    stop=[False]
    for sig in (signal.SIGINT,signal.SIGTERM):
        signal.signal(sig,lambda *_: stop.__setitem__(0,True))
    bridge=gb.Bridge(a.domain,a.iface,True,500.,100.,3.,a.log_dir/"bridge.log",listen_only=a.read_only)
    snap=np.zeros(1,gb.SNAP_DTYPE); status=np.zeros(1,gb.STATUS_DTYPE)
    q=np.zeros(35); kp=q.copy(); kd=q.copy(); modes=bytes([1]*29+[0]*6)
    began=time.monotonic(); deadline=began; damp_since=None; last_pub=last_lc=0; excess=0
    observed=False
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
            got=bridge.read_snap(snap); r=snap[0]
            if not got or not r["n_rx"]:
                if now-began>3:
                    raise RuntimeError("no consistent lowstate")
                time.sleep(.01); continue
            observed=True
            s={k:np.array(r[k][:29],copy=True) for k in ("q","dq","temp","motorstate")}
            s.update({k:np.array(r[k],copy=True) for k in ("gyro","quat")})
            s.update({k:float(r[k]) for k in ("t_rx","t_tick_change")})
            s.update(mode_machine=int(r["mode_machine"]),mode_pr=int(r["mode_pr"]),
                     keys=int(r["wireless_remote"][2])|int(r["wireless_remote"][3])<<8)
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
            if ctl.state!="PASSIVE" and now-deadline>.05:
                fault="control loop overrun"
            out=ctl.update(s,now,fault)
            if ctl.state=="DAMP" and not command_started:
                raise RuntimeError("refused before first command: "+ctl.reason)
            if out is not None:
                # Reject results whose sensor sample became stale during inference.
                age=time.monotonic()-s["t_rx"]
                if age>b.cfg["limits"]["state_age_s"]:
                    ctl.damp("state expired during policy"); out=ctl.damping(s)
                q[:29]=out["target"]; kp[:29]=out["kp"]; kd[:29]=out["kd"]
                flags=gb.CMD_NOW | (gb.CMD_Q_MEASURED if ctl.state=="DAMP" else 0)
                if not bridge.send(q,kp,kd,modes,5,flags):
                    ctl.damp("bridge send failed")
                else:
                    command_started=True
                log.write(json.dumps(dict(t=now,state=ctl.state,reason=ctl.reason,s=s,result=out,
                                          state_age_s=age),default=plain)+"\n")
            if ctl.state=="DAMP":
                damp_since=now if damp_since is None else damp_since
                if now-damp_since>=10.: break
            deadline+=.02
            time.sleep(max(0,deadline-time.monotonic()))
    except BaseException as exc:
        ctl.damp(str(exc))
        log.write(json.dumps(dict(event="exception",reason=str(exc)))+"\n")
        if not a.read_only and command_started and bridge.alive():
            q.fill(0); kp.fill(0); kd[:29]=3.
            until=time.monotonic()+10.
            while time.monotonic()<until and bridge.alive():
                bridge.send(q,kp,kd,modes,5,gb.CMD_NOW|gb.CMD_Q_MEASURED)
                time.sleep(.02)
        raise
    finally:
        bridge.quit()
        log.write(json.dumps(dict(event="end",state=ctl.state,reason=ctl.reason))+"\n")
        log.close()
if __name__=="__main__":
    main()
