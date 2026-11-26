"""BlinkMimic single-motion runner. No implicit robot interface or DDS domain."""
import argparse
import gc
import json
import os
import signal
import sys
import time
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent/"bridge"))
import g1_bridge_io as gb
from blinkmimic.deploy.core import Bundle, STAGES, OnnxPolicy, inference_device
from blinkmimic.deploy.controller import Controller
from blinkmimic.deploy.lowstate import decode_snapshot
from blinkmimic.deploy.snapshot import read_snapshot
from blinkmimic.deploy.runtime_io import RuntimeLog,cpu_list
from blinkmimic.deploy.scheduling import SleepMonitor,scheduler_snapshot
from blinkmimic.deploy.session import DEFAULT_PROFILE,new_session

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inference-device",type=inference_device,default="cpu",help="cpu or cuda:N; CUDA is explicit and never silently falls back")
    ap.add_argument("--cpu-affinity",help="comma-separated Linux CPUs for control, applied before ONNX initialization")
    ap.add_argument("--logger-cpus",default=os.environ.get("BLINK_LOG_CPUS"))
    ap.add_argument("--bridge-cpus",default=os.environ.get("BLINK_BRIDGE_CPUS"))
    ap.add_argument("--modality",choices=["motion","smpl","teleop"],default="motion")
    ap.add_argument("--vr-reference", help="live SONIC reference endpoint; enables VR controller")
    ap.add_argument("--vr-feedback-bind", default="tcp://*:5557")
    ap.add_argument("--bundle",type=Path,required=True)
    ap.add_argument("--motor-kp-scale",default="")
    ap.add_argument("--motor-kd-scale",default="")
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
    if a.cpu_affinity:os.sched_setaffinity(0,cpu_list(a.cpu_affinity))
    b=Bundle(a.bundle)
    from blinkmimic.deploy.hardware import apply_gains,release_motion
    effective_gains=apply_gains(b,a.motor_kp_scale,a.motor_kd_scale)
    profile=dict(DEFAULT_PROFILE)
    if a.hardware_profile:profile.update(json.loads(a.hardware_profile.read_text()))
    profile.pop("verified_on_this_robot",None)
    policy=OnnxPolicy(b,a.inference_device,a.modality)
    client=None
    if a.vr_reference:
        from blinkmimic.deploy.sonic_control import ReferenceClient,LiveController
        client=ReferenceClient(a.vr_reference,a.vr_feedback_bind)
        ctl=LiveController(b,profile,a.stage,client,a.inference_device)
        policy.info["input"]="SONIC live ZMQ"
    else:
        ctl=Controller(b,profile,a.stage,policy=policy)
    print("Inference: "+json.dumps(policy.info),flush=True)
    if a.inject_fault:
        original=ctl.core.policy
        calls=[0]
        def injected(*inputs):
            calls[0]+=1
            if calls[0]>=10:
                if a.inject_fault=="nan": return np.full(29,np.nan)
                time.sleep(.04)
            return original(*inputs)
        injected.modality=original.modality
        ctl.core.policy=injected
    if not a.sim and not a.read_only:
        if input("Robot supported / area clear / L2+B ready? Type yes: ").strip()!="yes":return
    takeover=release_motion(Path(__file__).resolve().parent/"bridge",a.iface,a.domain,a.sim,a.read_only)
    print("Motion takeover: "+json.dumps(takeover),flush=True)
    a.log_dir=new_session(a.log_dir.resolve())
    print(f"Runner logs: {a.log_dir}",flush=True)
    log=RuntimeLog(a.log_dir/"runner.jsonl",cpu_list(a.logger_cpus))
    log.write(dict(event="start",bundle_fingerprint=b.fingerprint,config=b.cfg,inference=policy.info,
                              temperature_channels=["casing","winding"],
                              temperature_channel_source="unitreerobotics/unitree_sdk2 include/unitree/robot/g1/common/terminations.hpp",
                              torque_log_semantics={"s.tau":"LowState tau_est, Nm",
                                  "result.pd_torque_estimate":"PD estimate at sampled q/dq; not measured torque"},
                              profile=profile,takeover=takeover,effective_gains=effective_gains,effective_limits=ctl.limits,effective_stage=ctl.stage,arguments={k:str(v) for k,v in vars(a).items()},control_cpus=sorted(os.sched_getaffinity(0))))
    stop=[False]
    for sig in (signal.SIGINT,signal.SIGTERM):
        signal.signal(sig,lambda *_: stop.__setitem__(0,True))
    control_cpus=os.sched_getaffinity(0)
    if a.bridge_cpus:os.sched_setaffinity(0,cpu_list(a.bridge_cpus))
    bridge=gb.Bridge(a.domain,a.iface,True,500.,100.,3.,a.log_dir/"bridge.log",listen_only=a.read_only)
    os.sched_setaffinity(0,control_cpus)
    scheduling=scheduler_snapshot()
    print(f"Actual controller scheduling: {scheduling['policy']} priority={scheduling['priority']} cpus={scheduling['cpus']}",flush=True)
    log.write(dict(event="scheduler",t=time.monotonic(),scheduling=scheduling))
    sleep_monitor=SleepMonitor()
    timing_fault_logged=False
    snap=np.zeros(1,gb.SNAP_DTYPE); status=np.zeros(1,gb.STATUS_DTYPE)
    q=np.zeros(35); kp=q.copy(); kd=q.copy(); modes=bytes([1]*29+[0]*6)
    began=time.monotonic(); deadline=began; damp_since=None; last_pub=last_lc=0; excess=0
    observed=False
    last_rx=None
    command_started=False
    # Collect setup objects once; control records are acyclic and released by refcounts.
    # Avoid process-wide cyclic-GC pauses in a deadline-bound loop.
    gc_was_enabled=gc.isenabled()
    gc.collect();gc.disable()
    last_cycle_t=None;last_cycle_cpu=None;last_process_cpu=None;previous_enqueue_ms=0.
    try:
        # Observe other publishers for 1 second before allowing any active command.
        while time.monotonic()-began<1.0 and not stop[0]:
            bridge.read_status(status)
            if status["lc_count"][0] and not a.read_only:
                raise RuntimeError("another lowcmd publisher present")
            time.sleep(.02)
        if a.sim and os.environ.get("BLINK_SIM_READY_FILE"):
            Path(os.environ["BLINK_SIM_READY_FILE"]).touch()
        deadline=time.monotonic()
        while bridge.alive():
            now=time.monotonic()
            cycle_cpu=time.thread_time()
            process_cpu=time.process_time()
            cycle_process_cpu_ms=None if last_process_cpu is None else (process_cpu-last_process_cpu)*1000
            last_process_cpu=process_cpu
            cycle_interval_ms=None if last_cycle_t is None else (now-last_cycle_t)*1000
            cycle_cpu_ms=None if last_cycle_cpu is None else (cycle_cpu-last_cycle_cpu)*1000
            last_cycle_t=now;last_cycle_cpu=cycle_cpu
            wake_lateness_ms=max(0.,(now-deadline)*1000)
            sleep_evidence=sleep_monitor.previous
            after_sleep_ms=None if sleep_evidence is None else (now-sleep_evidence["sleep_return"])*1000
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
                log.write(dict(event="state",t=now,s=s,other_lowcmd=int(bs["lc_count"])))
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
                fault=f"control loop overrun: wake_late_ms={wake_lateness_ms:.2f}, cycle_ms={cycle_interval_ms}"
            controller_cpu_started=time.thread_time()
            controller_started=time.monotonic()
            out=ctl.update(s,now,fault)
            controller_ms=(time.monotonic()-controller_started)*1000
            controller_cpu_ms=(time.thread_time()-controller_cpu_started)*1000
            inference_cpu_ms=out.get("inference_thread_cpu_ms") if out is not None else None
            policy_inference_ms=out.get("inference_ms") if out is not None else None
            if ctl.state=="DAMP" and not command_started:
                raise RuntimeError("refused before first command: "+ctl.reason)
            if out is not None:
                # Reject results whose sensor sample became stale during inference.
                age=time.monotonic()-s["t_rx"]
                if age>ctl.limits["state_age_s"]:
                    ctl.damp(f"state expired after control: phase={ctl.state}, age_ms={age*1000:.2f}, limit_ms={ctl.limits['state_age_s']*1000:.2f}"); out=ctl.damping(s)
                q[:29]=out["target"]; kp[:29]=out["kp"]; kd[:29]=out["kd"]
                flags=gb.CMD_NOW | (gb.CMD_Q_MEASURED if ctl.state=="DAMP" else 0)
                if not bridge.send(q,kp,kd,modes,s["mode_machine"],flags):
                    ctl.damp("bridge send failed")
                else:
                    command_started=True
                enqueue_started=time.monotonic()
                log.write(dict(t=now,state=ctl.state,reason=ctl.reason,s=s,result=out,
                                          state_age_s=age,policy_inference_ms=policy_inference_ms,
                                          controller_ms=controller_ms,controller_thread_cpu_ms=controller_cpu_ms,
                                          cycle_interval_ms=cycle_interval_ms,cycle_thread_cpu_ms=cycle_cpu_ms,
                                          cycle_process_cpu_ms=cycle_process_cpu_ms,
                                          previous_sleep=sleep_evidence,after_sleep_ms=after_sleep_ms,
                                          wake_lateness_ms=wake_lateness_ms,previous_enqueue_ms=previous_enqueue_ms,
                                          policy_inference_thread_cpu_ms=inference_cpu_ms,
                                          sample_age_before_control_ms=(controller_started-s["t_rx"])*1000))
            if client and out is not None:client.feedback(s,out)
            if a.status_file:
                log.status(a.status_file,dict(t=now,state=ctl.state,frame=ctl.core.frame,phase=ctl.core.phase,
                    reason=ctl.reason,reject_reason=ctl.reject_reason,keys=s["keys"],
                    live_mode=getattr(ctl,"live_mode",None),live_visual=getattr(ctl.core,"visual",{}),
                    heading=ctl.core.heading.tolist() if ctl.core.heading is not None else None,log_dir=str(a.log_dir)))
            if out is not None:previous_enqueue_ms=(time.monotonic()-enqueue_started)*1000
            if ctl.state=="DAMP":
                if not timing_fault_logged:
                    # After the damping command is sent: never put procfs diagnostics on the send path.
                    log.write(dict(event="fault_scheduling",t=time.monotonic(),reason=ctl.reason,
                                   scheduling=scheduler_snapshot(),previous_sleep=sleep_evidence,
                                   after_sleep_ms=after_sleep_ms))
                    timing_fault_logged=True
                damp_since=now if damp_since is None else damp_since
                if now-damp_since>=10.: break
            deadline=max(deadline+.02,time.monotonic())
            sleep_monitor.sleep_until(deadline)
    except BaseException as exc:
        ctl.damp(str(exc))
        fault_time=time.monotonic()
        log.write(dict(event="exception",t=fault_time,state="DAMP",
                                 frame=ctl.core.frame,reason=str(exc)))
        if a.status_file:
            log.status(a.status_file,dict(t=fault_time,state="DAMP",frame=ctl.core.frame,phase=ctl.core.phase,
                reason=ctl.reason,reject_reason="",keys=0,
                heading=ctl.core.heading.tolist() if ctl.core.heading is not None else None,log_dir=str(a.log_dir)))
        if not a.read_only and command_started and bridge.alive():
            q.fill(0); kp.fill(0); kd[:29]=3.
            until=time.monotonic()+10.
            while time.monotonic()<until and bridge.alive():
                bridge.send(q,kp,kd,modes,s["mode_machine"],gb.CMD_NOW|gb.CMD_Q_MEASURED)
                time.sleep(.02)
        raise
    finally:
        bridge.quit()
        if client:client.close()
        log.write(dict(event="end",t=time.monotonic(),state=ctl.state,frame=ctl.core.frame,reason=ctl.reason))
        log.close()
        if gc_was_enabled:gc.enable()
if __name__=="__main__":
    main()
