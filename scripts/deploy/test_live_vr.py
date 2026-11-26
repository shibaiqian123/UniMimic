"""Loopback-only live VR protocol smoke; synthetic inputs, NOT tracking acceptance."""
import json,os,subprocess,sys,time
from pathlib import Path
import numpy as np
import zmq
from blinkmimic.deploy.sonic_wire import pack,ISAAC_NAMES
from blinkmimic.deploy.core import JOINTS
ROOT=Path(__file__).resolve().parents[2]
def main():
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument("--bundle",type=Path,required=True)
    ap.add_argument("--groot-root",type=Path,required=True);ap.add_argument("--unitree-xml",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True);ap.add_argument("--disconnect",action="store_true")
    a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    z=np.load(a.bundle/"motion.npz")
    r=z["omni_reference"][0]
    from scipy.spatial.transform import Rotation
    quat=Rotation.from_matrix(r[139:148].reshape(3,3)).as_quat()[[3,0,1,2]]
    pointq=Rotation.from_matrix(r[157:184].reshape(3,3,3)).as_quat()[:,[3,0,1,2]]
    q=r[:29][[JOINTS.index(n) for n in ISAAC_NAMES]]
    ctx=zmq.Context();pub=ctx.socket(zmq.PUB);pub.bind("tcp://127.0.0.1:15556")
    cmd=[sys.executable,"-u",str(ROOT/"scripts/deploy/teleop.py"),"sim","--bundle",str(a.bundle.resolve()),
         "--groot-root",str(a.groot_root),"--unitree-xml",str(a.unitree_xml),"--vr-port","15556",
         "--feedback-port","15557","--reference-port","15558","--stationary-lower-body","--stage","slack","--duration","45","--out",str(a.out/"sessions")]
    logfile=open(a.out/"console.log","w")
    proc=subprocess.Popen(cmd,stdout=logfile,stderr=subprocess.STDOUT,cwd=ROOT)
    phase=-1;started=None;frame=0;events=[]
    try:
        deadline=time.monotonic()+90
        while proc.poll() is None and time.monotonic()<deadline:
            sessions=list((a.out/"sessions").glob("*")) if (a.out/"sessions").exists() else []
            if not sessions:time.sleep(.1);continue
            session=sessions[0]
            ready=session/"vr_source.log"
            if not ready.exists() or "source ready" not in ready.read_text():time.sleep(.1);continue
            if started is None:started=time.monotonic()
            elapsed=time.monotonic()-started
            new=0 if elapsed<7 else 1 if elapsed<13 else 2 if elapsed<16 else 3 if elapsed<22 else 4
            if new!=phase:
                phase=new;events.append(dict(t=elapsed,phase=phase))
                command=dict(start=np.array([phase==0],np.uint8),stop=np.array([phase==4 and not a.disconnect],np.uint8),
                             planner=np.array([phase!=1],np.uint8))
                if not (phase==4 and a.disconnect):pub.send(pack("command",command))
            if phase==4 and a.disconnect:time.sleep(.02);continue
            if phase==1:
                n=5;ids=np.arange(frame,frame+n,dtype=np.int64);frame+=n
                # Five 50Hz samples per packet, matching the restored PICO sender.
                pub.send(pack("pose",dict(frame_index=ids,smpl_joints=np.tile(r[67:139].reshape(24,3),(n,1,1)).astype(np.float32),
                     body_quat_w=np.tile(quat,(n,1)).astype(np.float32),joint_pos=np.tile(q,(n,1)).astype(np.float32))))
            else:
                fields=dict(mode=np.array([0],np.int32),movement=np.zeros(3,np.float32),facing=np.array([1,0,0],np.float32),
                            speed=np.array([-1],np.float32),height=np.array([-1],np.float32))
                if phase==3:
                    fields.update(vr_position=r[148:157].astype(np.float32),vr_orientation=pointq.astype(np.float32).reshape(-1))
                pub.send(pack("planner",fields))
            time.sleep(.1 if phase==1 else .02)
        if proc.poll() is None:proc.terminate()
        code=proc.wait(timeout=20)
        sessions=list((a.out/"sessions").glob("*"));report=json.loads((sessions[0]/"report.json").read_text()) if sessions else {}
        (a.out/"smoke.json").write_text(json.dumps(dict(exit_code=code,events=events,report=report,synthetic=True),indent=2))
        print(json.dumps(dict(exit_code=code,session=str(sessions[0]),report=report),indent=2))
    finally:
        if proc.poll() is None:proc.terminate();proc.wait(timeout=20)
        logfile.close();pub.close(0);ctx.term()
if __name__=="__main__":main()
