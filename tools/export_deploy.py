"""Export a standalone single-motion bundle (does not arm or contact a robot)."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from blinkmimic import networks
from blinkmimic.networks import Policy, policy_kwargs
from blinkmimic.kinematics import Kinematics
from blinkmimic.motion import MotionLib, CONTROL_DT
from blinkmimic.robot import get_robot
from blinkmimic.rotations import mat_to_6d, yaw_mat
from blinkmimic.g1.g1_constants import motor_of
from blinkmimic.deploy.core import digest, OBS_LAYOUT, Bundle, OnnxPolicy, STAGES, LIMITS
from export_onnx import pd_gains

class ExportPolicy(nn.Module):
    def __init__(self, policy):
        super().__init__()
        self.policy = policy
    def forward(self, obs_history, goal):
        return self.policy(obs_history,goal)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt",required=True,type=Path)
    ap.add_argument("--motion",required=True,type=Path)
    ap.add_argument("--out",required=True,type=Path)
    a=ap.parse_args()
    if a.out.exists():
        raise ValueError("output must be a new directory")
    torch.set_num_threads(4)
    ck=torch.load(a.ckpt,map_location="cpu",weights_only=False)
    cfg=ck["args"]
    if cfg["hist"] != 10 or cfg["goal_frames"] != 10:
        raise ValueError("v1 requires 10 history and 10 goal frames")
    robot=get_robot("g1")
    if robot.sensing.stateful:
        raise ValueError("stateful sensing unsupported by v1")
    kin=Kinematics(robot,"cpu")
    motion=MotionLib([str(a.motion)],kin,"cpu",robot.load_motion)
    widths=lambda k: tuple(map(int,cfg.get(k,"2048,1024,512,256").split(",")))
    pi=Policy(99,730,29,hist=10,z_dim=32,enc_hidden=widths("enc_hidden"),dec_hidden=widths("dec_hidden"),
              **policy_kwargs(cfg,robot)).eval()
    pi.load_state_dict(ck["pi"],strict=True)
    networks.AMP_DTYPE=None
    wrapped=ExportPolicy(pi).eval()
    frames=torch.arange(motion.num_frames)
    goals=motion.goal(frames,10)
    static=[]
    for f in (0,motion.num_frames-1):
        R=motion.root_R[f]
        rot=mat_to_6d(yaw_mat(R).T@R)
        one=torch.cat((torch.zeros(3),rot,torch.zeros(6),motion.q[f],torch.zeros(29)))
        static.append(one.repeat(10))
    a.out.mkdir(parents=True)
    torch.onnx.export(wrapped,(torch.zeros(1,10,99),goals[:1]),str(a.out/"policy.onnx"),
                      input_names=["obs_history","goal"],output_names=["actions"],
                      dynamic_axes={"obs_history":{0:"batch"},"goal":{0:"batch"},"actions":{0:"batch"}},
                      opset_version=18,dynamo=False)
    np.savez_compressed(a.out/"motion.npz",q=motion.q.numpy(),root_R=motion.root_R.numpy(),
                        root_p=motion.root_p.numpy(),goal=goals.numpy(),static_goal=torch.stack(static).numpy())
    kp,kd=pd_gains(robot)
    motors=[motor_of(n) for n in robot.joint_names]
    meta=dict(schema="blinkmimic_single_motion_v1",joint_names=list(robot.joint_names),
              q0=robot.default_joint_pos_list(),scale=robot.action_scale_list(),kp=kp,kd=kd,
              effort=[x[0] for x in motors],velocity=[x[1] for x in motors],
              lower=kin.q_lo.tolist(),upper=kin.q_hi.tolist(),history=10,goal_frames=10,
              obs_dim=99,goal_dim=730,latent_dim=32,control_dt=CONTROL_DT,obs_layout=OBS_LAYOUT,
              checkpoint_iteration=int(ck["iteration"]),checkpoint_sha256=digest(a.ckpt),
              source_motion_sha256=digest(a.motion),source_motion=str(a.motion.resolve()),
              sha256={n:digest(a.out/n) for n in ("policy.onnx","motion.npz")},
              stages=STAGES,limits=LIMITS,preparation_gain_scale=4.0)
    (a.out/"deploy_config.json").write_text(json.dumps(meta,indent=2)+"\n")
    b=Bundle(a.out)
    ort=OnnxPolicy(b)
    rng=np.random.default_rng(0)
    err=0.0
    with torch.no_grad():
        for f in np.linspace(0,motion.num_frames-1,64).astype(int):
            h=pi.obs_norm.mean + pi.obs_norm.std()*torch.from_numpy(rng.standard_normal((1,10,99)).astype(np.float32))
            for g in (goals[f],torch.stack(static)[f%2]):
                ref=wrapped(h,g[None])[0].numpy()
                err=max(err,float(abs(ref-ort(h[0].numpy(),g.numpy())).max()))
    report=dict(max_action_error=err,tolerance=1e-3,passed=err<=1e-3,bundle_fingerprint=b.fingerprint)
    (a.out/"export_validation.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))
    if not report["passed"]:
        raise RuntimeError("ONNX parity failed; bundle not accepted")
if __name__=="__main__":
    main()
