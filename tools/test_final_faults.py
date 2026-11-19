"""Test final-session stalls, state loss and parent exit on DDS domain 2 only."""
import argparse,json,os,signal,subprocess,sys,time
from pathlib import Path
import numpy as np
from blinkmimic.deploy.sim_ipc import SimIPC
from blinkmimic.deploy.final_session import read,send

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--bundle",type=Path,required=True)
    ap.add_argument("--unitree-xml",type=Path,required=True);ap.add_argument("--out",type=Path,required=True)
    a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=False);results={}
    for kind in ("delay","stale","runner_exit"):
        root=a.out/kind;log=open(a.out/(kind+".log"),"w")
        process=subprocess.Popen([sys.executable,"-u","scripts/deploy/sim2sim.py","--cmd","final",
            "--bundle",str(a.bundle),"--unitree-xml",str(a.unitree_xml),"--dds-domain","2","--auto","--duration","25",
            "--out",str(root)],stdout=log,stderr=subprocess.STDOUT)
        ipc=None;runner_pid=None;suspended=False
        try:
            deadline=time.monotonic()+25;session=None
            while time.monotonic()<deadline:
                dirs=sorted(root.glob("*")) if root.exists() else []
                if dirs:
                    session=dirs[-1];status=read(session/"runner_status.json")
                    if status.get("state")=="STANDBY_PRE":break
                    if status.get("state")=="DAMP":raise RuntimeError(status.get("reason"))
                if process.poll() is not None:raise RuntimeError("session exited before standby")
                time.sleep(.02)
            else:raise RuntimeError("standby timeout")
            ipc=SimIPC(session/"sim.mmap");runner_pid=read(session/"processes.json")["runner"]
            before=ipc.state();t=time.monotonic()
            if kind=="delay":
                os.kill(runner_pid,signal.SIGSTOP);suspended=True
                time.sleep(.07)
                after=ipc.state()
                os.kill(runner_pid,signal.SIGCONT);suspended=False
                advanced=float(after["sim_time"]-before["sim_time"])
                if advanced<.04:raise RuntimeError("physics paused with runner")
            elif kind=="stale":
                send(session,dict(type="fault",drop_state=True));time.sleep(.08)
                advanced=float(ipc.state()["sim_time"]-before["sim_time"])
                if advanced<.04:raise RuntimeError("physics paused on sensor loss")
            else:
                os.kill(runner_pid,signal.SIGKILL);time.sleep(.25)
                advanced=float(ipc.state()["sim_time"]-before["sim_time"])
            until=time.monotonic()+3
            while time.monotonic()<until:
                cmd=ipc.command()
                if cmd["received"]>t and np.allclose(cmd["kp"],0) and np.allclose(cmd["kd"],3):break
                time.sleep(.02)
            else:raise RuntimeError("no actual damping LowCmd received by simulator")
            status=read(session/"runner_status.json")
            results[kind]=dict(passed=True,physics_advanced_s=advanced,runner_state=status.get("state"),
                              reason=status.get("reason"),damping_received=True)
            process.wait(20)
        except Exception as exc:
            results[kind]=dict(passed=False,error=str(exc))
        finally:
            if suspended and runner_pid:
                try:os.kill(runner_pid,signal.SIGCONT)
                except ProcessLookupError:pass
            if ipc:ipc.close()
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
                try:process.wait(20)
                except subprocess.TimeoutExpired:process.kill();process.wait()
            log.close()
        print(kind,results[kind],flush=True)
    report=dict(passed=all(x["passed"] for x in results.values()),cases=results)
    (a.out/"report.json").write_text(json.dumps(report,indent=2))
    if not report["passed"]:raise SystemExit(1)
if __name__=="__main__":main()
