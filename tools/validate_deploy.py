"""CPU training/deployment observation, goal and history parity (no DDS)."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from blinkmimic.deploy.core import Bundle,DeployCore,yaw_rotation
from blinkmimic.robot import get_robot
from blinkmimic.kinematics import Kinematics
from blinkmimic.motion import MotionLib
from blinkmimic.mdp import MDP
from blinkmimic.dr import XiLayout
from blinkmimic.rotations import so3_exp,mat_to_6d,yaw_mat

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--bundle",required=True,type=Path); a=ap.parse_args()
    torch.set_num_threads(4); torch.manual_seed(10)
    b=Bundle(a.bundle); robot=get_robot("g1"); kin=Kinematics(robot,"cpu")
    motion=MotionLib([b.cfg["source_motion"]],kin,"cpu",robot.load_motion)
    layout=XiLayout(robot,"cpu")
    mdp=MDP(kin,motion,torch.tensor(b.q0,dtype=torch.float32),layout.dim,robot.sensing)
    max_obs=max_goal=max_hist=0.
    for angle in (0.,1.2,-2.5):
        core=DeployCore(b,lambda h,g:np.zeros(29)); core.align(yaw_rotation(angle)@motion.root_R[0].numpy())
        core.start_motion(); hist=None
        for f in list(range(15))+[motion.num_frames-2,motion.num_frames-1]:
            core.frame=f
            p,R,v,w,q,dq=[x[f:f+1].clone() for x in
                          (motion.root_p,motion.root_R,motion.root_v,motion.root_w,motion.q,motion.qd)]
            R=R@so3_exp(torch.randn(1,3)*0.05); q+=torch.randn_like(q)*0.02
            prev=torch.randn(1,29)*0.1; core.last_action=prev[0].numpy()
            original=mdp.obs((p,R,v,w,q,dq),prev,torch.zeros(1,layout.dim),torch.tensor([f]))[0].numpy()
            hist=np.repeat(original[None],10,axis=0) if hist is None else np.concatenate((hist[1:],original[None]))
            actual=core.observe(q[0].numpy(),dq[0].numpy(),(R[0].T@w[0]).numpy(),yaw_rotation(angle)@R[0].numpy())
            max_obs=max(max_obs,float(abs(actual[-1]-original).max()))
            max_hist=max(max_hist,float(abs(actual-hist).max()))
            max_goal=max(max_goal,float(abs(core.reference()[1]-motion.goal(torch.tensor([f]),10)[0].numpy()).max()))
    for idx,f in enumerate((0,motion.num_frames-1)):
        R=motion.root_R[f]
        one=torch.cat((torch.zeros(3),mat_to_6d(yaw_mat(R).T@R),torch.zeros(6),motion.q[f],torch.zeros(29)))
        max_goal=max(max_goal,float(abs(b.motion["static_goal"][idx]-one.repeat(10).numpy()).max()))
    report=dict(bundle_fingerprint=b.fingerprint,observation_error=max_obs,history_error=max_hist,
                goal_error=max_goal,tolerance=1e-4,passed=max(max_obs,max_hist,max_goal)<=1e-4)
    (a.bundle/"observation_validation.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))
    assert report["passed"]
if __name__=="__main__": main()
