"""Export one paired reference clip for the existing final DDS runner."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from blinkmimic.omni import CONTRACT,WRISTS
from blinkmimic.omni_export import load_policy,export
from blinkmimic.omni_data import ReferenceStore
from blinkmimic.motion import MotionLib
from blinkmimic.kinematics import Kinematics
from blinkmimic.g1.g1_constants import SPEC,motor_of
from blinkmimic.rotations import mat_to_6d,yaw_mat
from blinkmimic.deploy.core import digest,PROPRIO_LAYOUT,STAGES,LIMITS
from export_onnx import pd_gains

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt",type=Path,required=True)
    p.add_argument("--index",type=Path,required=True)
    p.add_argument("--motion-id",required=True)
    p.add_argument("--data-root",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--preparation-gain-scale",type=float,default=4.)
    a=p.parse_args()
    if a.out.exists():p.error("use new output directory")
    if not np.isfinite(a.preparation_gain_scale) or a.preparation_gain_scale<=0:p.error("positive gain required")
    ck=torch.load(a.ckpt,map_location="cpu",weights_only=False)
    policy=load_policy(ck)
    store=ReferenceStore(a.index,SPEC.joint_names)
    cid=store.by_id[a.motion_id];r=store.records[cid];n=r["frames"]
    kin=Kinematics(SPEC,"cpu")
    motion=MotionLib([str(a.data_root/r["g1_path"])],kin,"cpu",SPEC.load_motion,frame_limits=[n])
    reference=store.gather(np.full(n,cid),np.arange(n))
    export(policy,a.out)
    goals=motion.goal(torch.arange(n),10)
    static=[]
    for f in (0,n-1):
        rot=mat_to_6d(yaw_mat(motion.root_R[f]).T@motion.root_R[f])
        static.append(torch.cat((torch.zeros(3),rot,torch.zeros(6),motion.q[f],torch.zeros(29))).repeat(10))
    np.savez_compressed(a.out/"motion.npz",q=motion.q.numpy(),qd=motion.qd.numpy(),
        root_R=motion.root_R.numpy(),root_p=motion.root_p.numpy(),root_v=motion.root_v.numpy(),
        root_w=motion.root_w.numpy(),goal=goals.numpy(),static_goal=torch.stack(static).numpy(),
        omni_reference=reference)
    kp,kd=pd_gains(SPEC);motors=[motor_of(n) for n in SPEC.joint_names]
    files=["motion.npz","motion_encoder.onnx","smpl_encoder.onnx","teleop_encoder.onnx","decoder.onnx"]
    meta=dict(schema="blinkmimic_single_motion_v4",joint_names=list(SPEC.joint_names),
        q0=SPEC.default_joint_pos_list(),scale=SPEC.action_scale_list(),kp=kp,kd=kd,
        effort=[x[0] for x in motors],velocity=[x[1] for x in motors],
        lower=kin.q_lo.tolist(),upper=kin.q_hi.tolist(),history=10,goal_frames=10,
        obs_dim=93,goal_dim=730,latent_dim=64,control_dt=.02,obs_layout=PROPRIO_LAYOUT,
        checkpoint_iteration=int(ck["iteration"]),checkpoint_sha256=digest(a.ckpt),
        source_motion_sha256=digest(a.data_root/r["g1_path"]),source_motion=r["g1_path"],
        sha256={name:digest(a.out/name) for name in files},stages=STAGES,limits=LIMITS,
        preparation_gain_scale=a.preparation_gain_scale,latent_spec=policy.latent_spec,
        omni_contract=CONTRACT,omni_lower=store.lower,omni_wrists=store.wrists,
        motion_id=a.motion_id,paired_index_sha256=digest(a.index))
    (a.out/"deploy_config.json").write_text(json.dumps(meta,indent=2))
    from blinkmimic.deploy.core import Bundle
    Bundle(a.out)
    print(a.out)
if __name__=="__main__":main()
