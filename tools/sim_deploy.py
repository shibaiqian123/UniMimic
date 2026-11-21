"""Independent CPU MuJoCo rollout; reports separate motion and endpoint evidence."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from blinkmimic.deploy.core import Bundle,DeployCore,limit_target
from blinkmimic.deploy.simulation import CpuSim
from blinkmimic.robot import get_robot
from blinkmimic.motion import MotionLib
from blinkmimic.kinematics import Kinematics
from blinkmimic.mdp import MDP,MAX_STEP_REWARD
from blinkmimic.dr import XiLayout
from blinkmimic.networks import Policy,policy_kwargs
from blinkmimic import networks

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bundle",required=True,type=Path)
    ap.add_argument("--ckpt",required=True,type=Path,help="same checkpoint, for torch control baseline")
    ap.add_argument("--standby-seconds",type=float,default=5.)
    a=ap.parse_args()
    if a.standby_seconds<3: ap.error("at least 3 seconds of endpoint verification")
    torch.set_num_threads(1)
    b=Bundle(a.bundle)
    from blinkmimic.deploy.core import digest
    if digest(a.ckpt)!=b.cfg["checkpoint_sha256"]: raise ValueError("checkpoint mismatch")
    (a.bundle/"sim_validation.json").write_text(json.dumps(dict(bundle_fingerprint=b.fingerprint,passed=False,status="RUNNING"))+"\n")
    robot=get_robot("g1"); kin=Kinematics(robot,"cpu")
    motion=MotionLib([b.cfg["source_motion"]],kin,"cpu",robot.load_motion)
    mdp=MDP(kin,motion,torch.tensor(b.q0,dtype=torch.float32),XiLayout(robot,"cpu").dim,robot.sensing)
    ck=torch.load(a.ckpt,map_location="cpu",weights_only=False); cfg=ck["args"]
    widths=lambda k: tuple(map(int,cfg.get(k,"2048,1024,512,256").split(",")))
    pi=Policy(99,730,29,enc_hidden=widths("enc_hidden"),dec_hidden=widths("dec_hidden"),**policy_kwargs(cfg,robot)).eval()
    pi.load_state_dict(ck["pi"]); networks.AMP_DTYPE=None
    def torch_policy(h,g,orientation6d_future=None):
        with torch.no_grad(): return pi(torch.tensor(h[None],dtype=torch.float32),torch.tensor(g[None],dtype=torch.float32),orientation6d_future=None if orientation6d_future is None else torch.tensor(orientation6d_future[None],dtype=torch.float32))[0].numpy()
    results={}
    for name,policy in (("onnx",None),("torch",torch_policy)):
        sim=CpuSim(b); core=DeployCore(b,policy); core.align(sim.state()[3])
        ret=0.; failed=False; clamp_count=0; inference=[]; endpoint={}; completed=0
        for phase,steps in (("pre",round(a.standby_seconds/.02)),("motion",b.T-1),("post",round(a.standby_seconds/.02))):
            if phase=="motion": core.start_motion()
            max_dq=max_tilt=0.; stable_window=[]
            for k in range(steps):
                q,dq,g,R=sim.state()
                f,_=core.reference(); ref=core.heading@b.motion["root_R"][f]
                tilt=float(np.arccos(np.clip(np.dot(-R[2],-ref[2]),-1,1)))
                stable_window.append(tilt<.14 and abs(dq).max()<1.)
                max_dq=max(max_dq,float(abs(dq).max())); max_tilt=max(max_tilt,tilt)
                prev=core.last_action.copy()
                try:
                    out=core.step(q,dq,g,R)
                    target,n=limit_target(out["target"],q,dq,b,b.cfg["stages"]["ground"])
                except ValueError as exc:
                    failed=True; endpoint["reason"]=str(exc); break
                inference.append(out["inference_ms"]); clamp_count+=n
                sim.step(target)
                q1,dq1,g1,R1=sim.state()
                cv=lambda x: torch.tensor(np.asarray(x)[None],dtype=torch.float32)
                S=tuple(map(cv,(sim.d.qpos[:3],R1,sim.d.qvel[:3],R1@g1,q1,dq1)))
                ff=torch.tensor([min(f+1,b.T-1) if phase=="motion" else f])
                with torch.no_grad():
                    term=bool(mdp.termination(S,ff)[0])
                    if phase=="motion":
                        ret+=float(mdp.reward(S,ff,cv(out["action"]),cv(prev))[0]); completed+=1
                if term: failed=True; endpoint["reason"]="training termination criterion"; break
            if phase!="motion":
                endpoint[phase]=dict(max_dq=max_dq,max_tilt=max_tilt,
                                     stable=bool(len(stable_window)>=150 and all(stable_window[-150:]) and not failed))
            if failed: break
        results[name]=dict(norm_return=ret/(max(b.T-1,1)*MAX_STEP_REWARD),
                           failed=failed,motion_steps=completed,endpoints=endpoint,
                           clamped_joints_total=clamp_count,
                           inference_p99_ms=float(np.percentile(inference,99)) if inference else None)
        print(name,json.dumps(results[name]),flush=True)
    passed=all(not r["failed"] and r["motion_steps"]==b.T-1 and
               all(r["endpoints"].get(p,{}).get("stable",False) for p in ("pre","post")) for r in results.values())
    report=dict(bundle_fingerprint=b.fingerprint,passed=passed,results=results,
                plant="independent CPU MuJoCo, implicit PD, dt=2ms; not the original GPU benchmark",
                return_difference=abs(results["onnx"]["norm_return"]-results["torch"]["norm_return"]))
    (a.bundle/"sim_validation.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))
    if not passed: raise SystemExit(1)
if __name__=="__main__": main()
