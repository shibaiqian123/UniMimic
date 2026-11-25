"""SONIC reference three-point poses. No policy or physical robot input."""
import numpy as np
import torch
from .motion import resample_motion
from .rotations import quat_to_mat, mat_to_quat

SONIC_COMMIT = "1e1ec4f5f27e6c3e03048f693bd7c60cce04afda"
BODIES = ("left_wrist_yaw_link", "right_wrist_yaw_link", "torso_link")
OFFSETS = ((.18,-.025,0.),(.18,.025,0.),(0.,0.,.35))
CONTRACT = dict(schema="sonic_reference_threepoint_v1", sonic_commit=SONIC_COMMIT,
    point_order=["left_hand","right_hand","virtual_head"], bodies=BODIES,
    offsets_m=OFFSETS, anchor="pelvis", fps=50, quaternion="wxyz",
    local_frame="full reference pelvis rotation", virtual_head="torso orientation, torso + rotated [0,0,.35]",
    scope="reference geometry, not measured VR; no lower-body command or robot-relative heading")


def continuous_quaternion(R):
    q = mat_to_quat(R).detach().cpu().numpy().astype(np.float32)
    if len(q)>1:
        flips = np.where((q[1:]*q[:-1]).sum(-1)<0,-1.,1.)
        q[1:] *= np.cumprod(flips,axis=0)[...,None]
    return q


@torch.no_grad()
def poses(kin, p, quaternion, joints):
    p, quaternion, joints = (x.to(kin.device) for x in (p,quaternion,joints))
    R = quat_to_mat(quaternion)
    bp,bR,_,_ = kin.forward(p,R,torch.zeros_like(p),torch.zeros_like(p),
                           joints,torch.zeros_like(joints),all_bodies=True)
    ids = [kin.body_names.index(x) for x in BODIES]
    anchor = kin.body_names.index("pelvis")
    ap, aR = bp[:,anchor], bR[:,anchor]
    rot = bR[:,ids]
    off = torch.tensor(OFFSETS,device=p.device,dtype=p.dtype)
    world = bp[:,ids] + (rot@off[...,None]).squeeze(-1)
    inv = aR.transpose(-1,-2)
    local = (inv[:,None]@(world-ap[:,None])[...,None]).squeeze(-1)
    local_R = inv[:,None]@rot
    arr=lambda x:x.detach().cpu().numpy().astype(np.float32)
    return dict(time_s=np.arange(len(p),dtype=np.float32)*np.float32(.02),
        position_world=arr(world), quaternion_world=continuous_quaternion(rot),
        position_local=arr(local), quaternion_local=continuous_quaternion(local_R),
        pelvis_position_world=arr(ap), pelvis_quaternion_world=continuous_quaternion(aR))


def from_csv(path, record, spec, kin):
    p,q,j=resample_motion(*spec.load_motion(str(path)))
    n=int(record["pairing"]["common_frames50"])
    if n<2 or n>len(p) or (n-1)*.02>record["pairing"]["common_end_s"]+1e-6:
        raise ValueError("Invalid shared time range")
    return poses(kin,p[:n],q[:n],j[:n])
