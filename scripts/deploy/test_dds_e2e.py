"""Actual runner + bridge + isolated DDS fixture/physics. NEVER uses domain 0."""
import argparse,json,os,signal,subprocess,sys,time
from pathlib import Path
from blinkmimic.deploy.core import Bundle
from blinkmimic.deploy.controller import PREP,ARM,PLAY,STOP

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bundle",required=True,type=Path)
    ap.add_argument("--out",required=True,type=Path)
    ap.add_argument("--physics",action="store_true",help="use physics instead of replay fixture; validation may fail")
    ap.add_argument("--domain",type=int,choices=[1,2],default=2)
    a=ap.parse_args(); a.bundle=a.bundle.resolve(); a.out=a.out.resolve()
    a.out.mkdir(parents=True,exist_ok=False); b=Bundle(a.bundle)
    root=Path(__file__).resolve().parents[2]
    results={}
    cases=["normal","stale","hot","nan","timeout","runner_exit","readonly"]
    if a.physics: cases=["normal"]
    for case in cases:
        d=a.out/case; d.mkdir()
        controls=d/"controls.json"
        cfg=dict(keys=0,support=True)
        def put(**kw):
            cfg.update(kw); tmp=controls.with_suffix(".tmp"); tmp.write_text(json.dumps(cfg)); tmp.replace(controls)
        put()
        simlog=open(d/"sim.log","w"); runlog=open(d/"process.log","w")
        simcmd=[sys.executable,str(root/"scripts/deploy/sim_dds.py"),"--bundle",str(a.bundle),
                "--scene",str(a.bundle/"sim_scene.xml"),"--buttons",str(controls),"--domain",str(a.domain),"--duration",str(max(90,b.T*.02+40))]
        if not a.physics: simcmd+=["--fixture"]
        sim=subprocess.Popen(simcmd,stdout=simlog,stderr=subprocess.STDOUT,cwd=root)
        runner=None
        def rows():
            try:
                return [json.loads(l) for l in (d/"run/runner.jsonl").read_text().splitlines() if l.strip()]
            except (OSError,ValueError): return []
        def wait_state(state,timeout=12):
            until=time.monotonic()+timeout
            while time.monotonic()<until:
                rs=rows()
                if any(r.get("state")==state for r in rs): return rs
                if any(r.get("state")=="DAMP" for r in rs) or (runner is not None and runner.poll() is not None):
                    raise RuntimeError("unexpected DAMP/exit: "+str(next((r.get("reason") for r in reversed(rs) if r.get("reason")),"no reason")))
                time.sleep(.1)
            raise RuntimeError("timeout waiting "+state)
        def press(key):
            put(keys=key); time.sleep(.18); put(keys=0); time.sleep(.12)
        try:
            time.sleep(2)
            cmd=[sys.executable,str(root/"scripts/deploy/run_g1.py"),"--bundle",str(a.bundle),
                 "--hardware-profile",str(root/"scripts/deploy/hardware_profile.example.json"),
                 "--iface","lo","--domain",str(a.domain),"--stage","ground","--sim","--log-dir",str(d/"run")]
            if case in ("nan","timeout"):cmd+=["--inject-fault",case]
            if case=="readonly":cmd+=["--read-only","--seconds","4"]
            runner=subprocess.Popen(cmd,stdout=runlog,stderr=subprocess.STDOUT,cwd=root)
            if case=="readonly":
                runner.wait(10)
                status=json.loads(controls.with_suffix(".status.json").read_text())
                if runner.returncode!=0 or status["n_commands"]!=0: raise RuntimeError("read-only published or failed")
                results[case]=dict(passed=True,commands_received=0)
                continue
            time.sleep(3); press(PREP); wait_state("HOLD")
            press(ARM); wait_state("STANDBY_PRE")
            if case=="normal":
                # Release the support for physical stability, never for the replay fixture.
                put(support=False); time.sleep(3.5); press(PLAY); wait_state("POLICY")
                if a.physics:
                    wait_state("STANDBY_POST",b.T*.02+20)
                    time.sleep(3.5)
                    if any(r.get("state")=="DAMP" for r in rows()): raise RuntimeError("post standby failed")
                else:
                    time.sleep(.3)
                press(STOP)
            elif case=="stale":put(drop_state=True)
            elif case=="hot":put(temperature=90)
            elif case=="runner_exit":
                runner.kill(); runner.wait()
                time.sleep(.3)
                log=(d/"run/bridge.log").read_text()
                if "runner gone" not in log and "WATCHDOG" not in log: raise RuntimeError("orphan not detected")
                status=json.loads(controls.with_suffix(".status.json").read_text())
                if status["kp"]!=[0.]*29 or status["kd"]!=[3.]*29: raise RuntimeError("no orphan damping on DDS")
                # Bridge has a bounded 10 s orphan damping interval.
                time.sleep(10.5)
                results[case]=dict(passed=True,evidence="bridge detected parent exit; see bridge.log")
                continue
            rs=wait_state("DAMP",5)
            time.sleep(.4)
            rs=rows()
            damp=[r for r in rs if r.get("state")=="DAMP" and "result" in r]
            if not damp or any(any(v!=0 for v in r["result"]["kp"]) for r in damp):
                raise RuntimeError("damping command invalid")
            first=next(i for i,r in enumerate(rs) if r.get("state")=="DAMP")
            if any(r.get("state") not in ("DAMP",None) for r in rs[first:]): raise RuntimeError("not latched")
            status=json.loads(controls.with_suffix(".status.json").read_text())
            if status["kp"]!=[0.]*29 or status["kd"]!=[3.]*29: raise RuntimeError("no damping on DDS")
            reason=damp[0].get("reason","")
            expected={"normal":"remote stop","stale":"stale state","hot":"overtemperature",
                      "nan":"policy action","timeout":"policy timeout"}[case]
            if expected not in reason: raise RuntimeError("wrong fault: "+reason)
            results[case]=dict(passed=True,reason=reason,states=sorted(set(r.get("state","") for r in rs)))
        except Exception as exc:
            results[case]=dict(passed=False,error=str(exc))
        finally:
            if runner is not None and runner.poll() is None:
                runner.send_signal(signal.SIGTERM)
                try:runner.wait(12)
                except subprocess.TimeoutExpired:runner.kill(); runner.wait()
            sim.terminate()
            try:sim.wait(3)
            except subprocess.TimeoutExpired:sim.kill();sim.wait()
            simlog.close();runlog.close()
            time.sleep(.3)
        print(case,json.dumps(results[case]),flush=True)
    report=dict(bundle_fingerprint=b.fingerprint,passed=all(x["passed"] for x in results.values()),
                mode="physics" if a.physics else "DDS sensor replay fixture, not physical motion acceptance",
                cases=results,logs=str(a.out))
    name="dds_physics_validation.json" if a.physics else "dds_validation.json"
    (a.bundle/name).write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))
    if not report["passed"]:raise SystemExit(1)
if __name__=="__main__":main()
