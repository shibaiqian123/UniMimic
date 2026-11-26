"""Final CPU sim2sim supervisor. Uses the SAME hardware runner over isolated DDS."""
import json,os,signal,socket,subprocess,sys,time
from pathlib import Path
from .runtime_io import affinity_plan
from .core import Bundle
from .session import new_session,atomic_json,DEFAULT_PROFILE,buttons_address
from .sim_ipc import SimIPC,BRIDGE

ROOT=Path(__file__).resolve().parents[3]
def send(path,event):
    with socket.socket(socket.AF_UNIX,socket.SOCK_DGRAM) as s:s.sendto(json.dumps(event).encode(),buttons_address(path))
def read(path):
    try:return json.loads(Path(path).read_text())
    except (OSError,ValueError):return {}
def stop_process(p,timeout=3):
    if p is None or p.poll() is not None:return
    p.terminate()
    try:p.wait(timeout)
    except subprocess.TimeoutExpired:p.kill();p.wait()

def run(a):
    if a.backend!="cpu":raise ValueError("--cmd final uses CPU official-model physics; GPU is diagnostic")
    if not a.unitree_xml:raise ValueError("--unitree-xml is required")
    priority=getattr(a,"rt_priority",0)
    if not 0<=priority<=10:raise ValueError("--rt-priority must be 0..10")
    if priority:
        check=subprocess.run(["chrt","-r",str(priority),"true"],capture_output=True,text=True)
        if check.returncode:raise RuntimeError("SCHED_RR permission unavailable; grant this shell RLIMIT_RTPRIO first: "+check.stderr.strip())
    plan=affinity_plan() if getattr(a,"runtime_affinity","auto")=="auto" else {}
    live=bool(getattr(a,"vr_host",None))
    b=Bundle(a.bundle)
    session=new_session(a.out.resolve())
    meta=dict(live_vr=live,modality=getattr(a,"modality","motion"),kind="blink_final_sim",coordinated_start=True,domain=a.dds_domain,iface="lo",bundle=str(a.bundle.resolve()),
              bundle_fingerprint=b.fingerprint,unitree_xml=str(a.unitree_xml.resolve()),
              inference_device=getattr(a,"inference_device","cpu"),rt_priority=priority,runtime_affinity=plan,stage=a.stage,sensor_profile=DEFAULT_PROFILE,physics_hz=1000,lowstate_hz=1000,
              lowcmd_hz=500,policy_hz=50,auto=a.auto)
    atomic_json(session/"session.json",meta)
    ipc=SimIPC(session/"sim.mmap",create=True)
    print(f"FINAL SIM SESSION: {session}",flush=True)
    print(f"Keyboard: {sys.executable} scripts/deploy/sim_keyboard.py --session {session}",flush=True)
    env=dict(os.environ,PYTHONPATH=str(ROOT/"src"),OMP_NUM_THREADS="1")
    env.update(OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1",BLINK_SIM_READY_FILE=str(session/"runner_ready"))
    if plan:env.update(BLINK_LOG_CPUS=",".join(map(str,plan["logger"])),BLINK_BRIDGE_CPUS=",".join(map(str,plan.get("bridge",plan["sim_bridge"]))))
    children=[];files=[]
    def spawn(name,cmd):
        f=open(session/(name+".log"),"w");files.append(f)
        if plan and name in plan:cmd=["taskset","-c",",".join(map(str,plan[name]))]+cmd
        if priority and name in ("physics","sim_bridge","runner"):cmd=["chrt","-r",str(priority)]+cmd
        p=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
        children.append(p);return p
    terminate=[False]
    old={}
    for sig in (signal.SIGINT,signal.SIGTERM):
        old[sig]=signal.signal(sig,lambda *_:terminate.__setitem__(0,True))
    physics=bridge=runner=window=source=None;seen=set();failure="";stop_sent=False;stop_at=None
    since=time.monotonic();state_since=since;previous=None;last_action=0.;policy_support=False
    auto_events=[];post_support_since=None
    try:
        physics=spawn("physics",[sys.executable,"-u","-m","blinkmimic.deploy.final_physics","--session",str(session)])
        deadline=time.monotonic()+15
        while not (session/"buttons_ready").exists():
            if physics.poll() is not None or time.monotonic()>deadline:raise RuntimeError("physics startup failed; see physics.log")
            time.sleep(.05)
        bridge=spawn("sim_bridge",[str(BRIDGE/"g1_sim_bridge"),str(session/"sim.mmap"),str(a.dds_domain)])
        vr_args=[]
        if live:
            source=spawn("vr_source",[sys.executable,"-u","-m","blinkmimic.deploy.sonic_source",
                "--bundle",str(a.bundle.resolve()),"--planner-model",str(a.planner_model.resolve()),
                "--vr-endpoint",f"tcp://{a.vr_host}:{a.vr_port}",
                "--feedback-endpoint",f"tcp://127.0.0.1:{a.vr_feedback_port}",
                "--reference-bind",f"tcp://127.0.0.1:{a.vr_reference_port}",
                "--record",str(session/"vr_packets.msgpack")]+(["--stationary-lower-body"] if a.stationary_lower_body else []))
            vr_args=["--vr-reference",f"tcp://127.0.0.1:{a.vr_reference_port}",
                     "--vr-feedback-bind",f"tcp://*:{a.vr_feedback_port}"]
        runner=spawn("runner",[sys.executable,"-u",str(ROOT/"scripts/deploy/run_g1.py"),
            "--bundle",str(a.bundle.resolve()),"--iface","lo","--domain",str(a.dds_domain),"--sim",
            "--stage",a.stage,"--modality",getattr(a,"modality","motion"),"--inference-device",getattr(a,"inference_device","cpu"),"--log-dir",str(session/"runner"),"--status-file",str(session/"runner_status.json")]+vr_args+["--motor-kp-scale",a.motor_kp_scale,"--motor-kd-scale",a.motor_kd_scale])
        if a.viewer:
            window=spawn("viewer",[sys.executable,"-u","-m","blinkmimic.deploy.final_view","--session",str(session),
                "--gl-renderer",a.gl_renderer,"--render-hz",str(a.render_hz)])
        # Initialize CUDA, DDS and OpenGL before starting the 1 kHz physical clock.
        # This is automatic setup, not a reset/pause of a running experiment.
        ready_deadline=time.monotonic()+60
        while not (session/"runner_ready").exists() or (a.viewer and not (session/"viewer_ready").exists()):
            if any(p.poll() is not None for p in (physics,bridge,runner)) or (window and window.poll() is not None):
                raise RuntimeError("simulation initialization failed; see child logs")
            if time.monotonic()>ready_deadline:raise RuntimeError("simulation initialization timed out")
            time.sleep(.02)
        (session/"physics_start").touch()
        atomic_json(session/"processes.json",dict(physics=physics.pid,sim_bridge=bridge.pid,runner=runner.pid,
                                                viewer=window.pid if window is not None else None))
        while True:
            now=time.monotonic();rs=read(session/"runner_status.json");ps=read(session/"physics_status.json")
            state=rs.get("state")
            if state and state!=previous:
                print(f"{state}: {rs.get('reason') or rs.get('reject_reason','')}",flush=True)
                previous=state;state_since=now;seen.add(state)
            if state=="POLICY" and ps.get("support"):policy_support=True
            if source is not None and source.poll() is not None:
                failure="VR source exited; see vr_source.log"
                send(session,dict(type="key",key="D"));stop_sent=True;stop_at=stop_at or now
            if a.auto and not stop_sent and now-last_action>.4:
                key=None
                if state=="PASSIVE":key="1"
                elif state=="HOLD" and now-state_since>1.2:key="2"
                elif state=="STANDBY_PRE":
                    if now-state_since>1 and ps.get("support"):key="B"
                    elif now-state_since>5:key="3"
                elif state=="STANDBY_POST" and now-state_since>5:
                    if not ps.get("support"):
                        key="B";post_support_since=None
                    else:
                        post_support_since=post_support_since or now
                        if now-post_support_since>1.:key="D";stop_sent=True;stop_at=now
                if key:
                    send(session,dict(type="key",key=key));last_action=now
                    auto_events.append(dict(t=now,key=key,state=state))
            if state=="DAMP":
                stop_sent=True;stop_at=stop_at or now
                reason=rs.get("reason","")
                if reason not in ("remote stop","operator exit","SONIC remote stop"):failure=reason
            request_stop=terminate[0] or (a.duration is not None and now-since>=a.duration)
            if window is not None and window.poll() is not None:
                request_stop=True
                if window.returncode:failure=f"viewer exited {window.returncode}; see viewer.log"
            if request_stop and not stop_sent:
                send(session,dict(type="key",key="D"));stop_sent=True;stop_at=now
            if stop_sent and state!="DAMP" and now-last_action>.4:
                # A stop pressed during the runner's startup observation must not be lost.
                send(session,dict(type="key",key="D"));last_action=now
            if physics.poll() is not None or bridge.poll() is not None:
                failure="simulation process exited unexpectedly";break
            if runner.poll() is not None:
                if runner.returncode:
                    failure=f"runner exited {runner.returncode}; see runner.log"
                    # Keep the simulated plant/receiver alive to observe the real bridge's orphan damping.
                    time.sleep(10.5)
                break
            if stop_at and now-stop_at>15:
                failure=failure or "runner did not finish after stop";break
            # Auto mode must not wait forever for a motion that cannot pass normal start conditions.
            if a.auto and state in ("HOLD","STANDBY_PRE") and now-state_since>30 and not stop_sent:
                failure=rs.get("reject_reason") or "start condition did not become ready"
                send(session,dict(type="key",key="D"));stop_sent=True;stop_at=now
            time.sleep(.02)
    except Exception as exc:
        failure=str(exc)
        try:send(session,dict(type="key",key="D"));time.sleep(.2)
        except OSError:pass
    finally:
        if runner is not None and runner.poll() is None:stop_process(runner,12)
        stop_process(source);stop_process(bridge);stop_process(window)
        try:send(session,dict(type="quit"))
        except OSError:pass
        if physics is not None:
            try:physics.wait(8)
            except subprocess.TimeoutExpired:stop_process(physics)
        counts=ipc.counts();ipc.close()
        for f in files:f.close()
        for sig,handler in old.items():signal.signal(sig,handler)
    rs=read(session/"runner_status.json");pr=read(session/"physics_report.json")
    if live:
        report=dict(session=str(session),scope="live SONIC input / BlinkMimic policy / simulated LowState",
                    states=sorted(seen),reason=failure or rs.get("reason",""),physics=pr,
                    hardware_verified=False,tracking_validation="not scored against the bundled recorded motion",
                    source_recording=str(session/"vr_packets.msgpack"),runner_log_dir=rs.get("log_dir"))
        atomic_json(session/"report.json",report)
        print(json.dumps(report,indent=2),flush=True)
        if failure:raise SystemExit(1)
        return
    completed="STANDBY_POST" in seen
    required={"PASSIVE","INTERP","HOLD","STANDBY_PRE","POLICY","STANDBY_POST","DAMP"}
    flow=completed and required.issubset(seen) and not failure and not policy_support
    report=dict(session=str(session),bundle_fingerprint=b.fingerprint,states=sorted(seen),
        motion_completed=completed,flow_valid=flow,policy_had_support=policy_support,
        timing_valid=bool(pr.get("timing_valid")),passed=flow and bool(pr.get("timing_valid")),
        reason=failure or ("" if completed else "operator stopped before full motion"),
        sensor_calibrated=False,hardware_verified=False,physics=pr,
        dds=dict(lowstate_published=counts[0],lowcmd_received=counts[1],crc_bad=counts[2],write_fail=counts[3]),
        runner_log_dir=rs.get("log_dir"),auto_events=auto_events,
        scope="shared real runner, modeled sensors, official CPU physics; no real hardware")
    from .final_evaluate import evaluate
    tracking=evaluate(session,rs["log_dir"],b) if rs.get("log_dir") else {"available":False,"passed":False}
    report["tracking"]=tracking
    runtime={}
    if rs.get("log_dir"):
        records=[]
        with open(Path(rs["log_dir"])/"runner.jsonl") as f:
            for line in f:
                try:records.append(json.loads(line))
                except ValueError:pass
        runtime["inference"]=next((r.get("inference") for r in records if r.get("event")=="start"),None)
        for field in ("controller_ms","controller_thread_cpu_ms","policy_inference_ms","policy_inference_thread_cpu_ms"):
            vals=[r[field] for r in records if isinstance(r.get(field),(int,float))]
            runtime["max_"+field]=max(vals) if vals else None
        limits=b.cfg.get("runner_limits",{})
        threshold=limits.get("state_age_s",.02)
        runtime["state_age_overruns"]=sum(r.get("state_age_s",0)>threshold for r in records if r.get("state")!="DAMP")
        runtime["control_deadline_exceeded"]=any(r.get("controller_ms",0)>20 for r in records)
        runtime["timing_fault"]=any(any(word in r.get("reason","") for word in
            ("state expired","stale state","control loop overrun","snapshot unavailable")) for r in records)
        runtime["logging"]=read(Path(rs["log_dir"])/"logging_report.json")
    report["runtime"]=runtime
    report["physics_timing_valid"]=bool(pr.get("timing_valid"))
    report["timing_valid"]=bool(report["physics_timing_valid"] and not runtime.get("timing_fault",True)
                                and not runtime.get("control_deadline_exceeded",True))
    report["passed"]=bool(report["passed"] and report["timing_valid"] and runtime.get("logging",{}).get("complete",False))
    report["passed"]=bool(report["passed"] and tracking.get("passed") and not tracking.get("policy_support_samples",0) and tracking.get("stop_stage",{}).get("passed",False))
    if counts[2] or counts[3]:report["passed"]=False
    atomic_json(session/"report.json",report)
    print(json.dumps(report,indent=2),flush=True)
    if not report["passed"]:raise SystemExit(1 if failure else 2)
