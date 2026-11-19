"""Record original mjlab/Warp policy rollout for G1 sim2sim comparison (training environment)."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from blinkmimic.deploy.core import Bundle,digest
from blinkmimic.robot import get_robot
from blinkmimic.kinematics import Kinematics
from blinkmimic.motion import MotionLib
from blinkmimic.mdp import MDP
from blinkmimic.dr import XiLayout
from blinkmimic.env import TrackingSim,make_sim
from blinkmimic.networks import Policy,policy_kwargs
from blinkmimic import networks

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bundle",required=True,type=Path);ap.add_argument("--ckpt",required=True,type=Path)
    ap.add_argument("--out",required=True,type=Path);ap.add_argument("--steps",type=int,default=0)
    a=ap.parse_args()
    if a.out.exists():ap.error("output must not exist")
    if a.steps<0:ap.error("--steps must be nonnegative")
    torch.set_num_threads(4);b=Bundle(a.bundle)
    if digest(a.ckpt)!=b.cfg["checkpoint_sha256"]:raise ValueError("checkpoint mismatch")
    robot=get_robot("g1");kin=Kinematics(robot,"cuda:0")
    if digest(b.cfg["source_motion"])!=b.cfg["source_motion_sha256"]:raise ValueError("source CSV changed")
    motion=MotionLib([b.cfg["source_motion"]],kin,"cuda:0",robot.load_motion)
    mdp=MDP(kin,motion,torch.tensor(b.q0,dtype=torch.float32,device="cuda:0"),XiLayout(robot,"cuda:0").dim,robot.sensing)
    env=TrackingSim(make_sim(robot),1,robot,motion,mdp,"cuda:0",eval_mode=True,randomize=False)
    env.reset(torch.tensor([0],device="cuda:0"),torch.tensor([0],device="cuda:0"));env.refresh()
    ck=torch.load(a.ckpt,map_location="cpu",weights_only=False);cfg=ck["args"]
    widths=lambda k:tuple(map(int,cfg.get(k,"2048,1024,512,256").split(",")))
    pi=Policy(99,730,29,enc_hidden=widths("enc_hidden"),dec_hidden=widths("dec_hidden"),**policy_kwargs(cfg,robot)).cuda().eval()
    pi.load_state_dict(ck["pi"]);networks.AMP_DTYPE=None
    torch.backends.cuda.matmul.allow_tf32=False
    history=None;records=[];failed=False
    with torch.no_grad():
        for i in range(min(a.steps or b.T-1,b.T-1)):
            o=env.observe();history=o[:,None].repeat(1,10,1) if history is None else torch.cat((history[:,1:],o[:,None]),1)
            action=pi(history,motion.goal(env.f,10))
            result=env.step(action);S=env.state()
            records.append(dict(frame=i,q=S[4][0].cpu().numpy().copy(),root_p=S[0][0].cpu().numpy().copy(),
                                action=action[0].cpu().numpy().copy()))
            if bool(result["term"][0]):failed=True;break
    a.out.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(a.out,**{k:np.array([r[k] for r in records]) for k in ("frame","q","root_p","action")},
                        bundle_fingerprint=np.array(b.fingerprint),complete=np.array(len(records)==b.T-1 and not failed),
                        precision=np.array("fp32 TF32 disabled"))
    print(json.dumps(dict(frames=len(records),failed=failed,complete=len(records)==b.T-1 and not failed,out=str(a.out))))
if __name__=="__main__":main()
