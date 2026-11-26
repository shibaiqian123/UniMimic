"""SONIC-style VR launch: identical source and controller in sim and real."""
import argparse,os,subprocess,sys,time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("target",choices=["sim","real"])
    p.add_argument("--bundle",type=Path,required=True)
    p.add_argument("--groot-root",type=Path,required=True,help="existing GR00T install; used only for planner asset")
    p.add_argument("--vr-host",default="127.0.0.1")
    p.add_argument("--vr-port",type=int,default=5556)
    p.add_argument("--feedback-port",type=int,default=5557)
    p.add_argument("--reference-port",type=int,default=5558)
    p.add_argument("--unitree-xml",type=Path)
    p.add_argument("--iface",help="real robot NIC; required only for real")
    p.add_argument("--hardware-profile",type=Path)
    p.add_argument("--viewer",action="store_true")
    p.add_argument("--gl-renderer",default="nvidia",choices=["auto","system","nvidia"])
    p.add_argument("--dds-domain",type=int,default=2,choices=[1,2],help="simulation domain")
    p.add_argument("--duration",type=float,help="simulation wall seconds")
    p.add_argument("--inference-device",default="cpu")
    p.add_argument("--out",type=Path,default=Path("deploy_logs/vr_live"))
    a=p.parse_args()
    planner=a.groot_root/"gear_sonic_deploy/planner/target_vel/V2/planner_sonic.onnx"
    if not planner.is_file():p.error("SONIC V2 planner ONNX not found: "+str(planner))
    a.bundle=a.bundle.resolve();a.out=a.out.resolve()
    env=dict(os.environ,PYTHONPATH=str(ROOT/"src"),OMP_NUM_THREADS="1")
    if a.target=="sim":
        if not a.unitree_xml:p.error("sim requires --unitree-xml")
        cmd=[sys.executable,"-u",str(ROOT/"scripts/deploy/sim2sim.py"),"--cmd","final","--bundle",str(a.bundle),
             "--unitree-xml",str(a.unitree_xml),"--vr-host",a.vr_host,"--vr-port",str(a.vr_port),
             "--vr-feedback-port",str(a.feedback_port),"--vr-reference-port",str(a.reference_port),
             "--planner-model",str(planner),"--dds-domain",str(a.dds_domain),"--out",str(a.out),
             "--inference-device",a.inference_device]
        if a.viewer:cmd+=["--viewer","--gl-renderer",a.gl_renderer]
        if a.duration is not None:cmd+=["--duration",str(a.duration)]
        raise SystemExit(subprocess.call(cmd,cwd=ROOT,env=env))
    if not a.iface or a.iface=="lo":p.error("real requires explicit --iface (robot NIC)")
    if a.viewer or a.duration is not None:p.error("--viewer/--duration are simulation only")
    from blinkmimic.deploy.session import new_session
    out=new_session(a.out)
    log=open(out/"vr_source.log","w")
    source=subprocess.Popen([sys.executable,"-u","-m","blinkmimic.deploy.sonic_source",
        "--bundle",str(a.bundle),"--planner-model",str(planner),"--vr-endpoint",f"tcp://{a.vr_host}:{a.vr_port}",
        "--feedback-endpoint",f"tcp://127.0.0.1:{a.feedback_port}",
        "--reference-bind",f"tcp://127.0.0.1:{a.reference_port}","--record",str(out/"vr_packets.msgpack")],
        cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
    cmd=[sys.executable,"-u",str(ROOT/"scripts/deploy/run_g1.py"),"--bundle",str(a.bundle),
         "--iface",a.iface,"--domain","0","--stage","ground",
         "--vr-reference",f"tcp://127.0.0.1:{a.reference_port}","--vr-feedback-bind",f"tcp://*:{a.feedback_port}",
         "--inference-device",a.inference_device,"--log-dir",str(out/"runner")]
    if a.hardware_profile:cmd+=["--hardware-profile",str(a.hardware_profile)]
    try:raise SystemExit(subprocess.call(cmd,cwd=ROOT,env=env))
    finally:
        source.terminate()
        try:source.wait(5)
        except subprocess.TimeoutExpired:source.kill();source.wait()
        log.close()
if __name__=="__main__":main()
