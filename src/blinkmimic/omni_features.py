"""SONIC release feature formulas, adapted to RobotSpec joint order.

Reference commit 2f07b35b3f3b9053b1f4fd2adeaf56b6bce32366, Apache-2.0.
See THIRD_PARTY_NOTICES.md and docs/omni.md.
"""
import torch
from .rotations import mat_to_6d, quat_to_mat, so3_exp

def future_indices(frame, end, stride):
    return torch.minimum(frame[...,None] + torch.arange(10,device=frame.device)*stride, end[...,None])

def smpl_reference(pose_aa, joints):
    """Exact release root conversion and per-frame local joint feature.

    Official smpl_joints are consumed as stored (no transl subtraction).
    Root = Rx(pi/2) @ axis_angle_root @ inverse(SMPL base rotation).
    """
    root = so3_exp(pose_aa[...,:3])
    c = root.new_tensor([[1,0,0],[0,0,-1],[0,1,0]])
    base = quat_to_mat(root.new_tensor([.5,.5,.5,.5]))
    root = c @ root @ base.T
    local = (root.transpose(-1,-2)[...,None,:,:] @ joints[...,None]).squeeze(-1)
    return local, root

def assemble(q, qd, root_R, smpl_local, smpl_R, points, point_R, robot_R, lower, wrists):
    """All arguments are paired windows, except current three-point targets.

    q/qd/root_R use offsets 0,5,...45. SMPL and wrist_q use 0,...9.
    q is a dict with 'motion' and 'smpl' entries, in RobotSpec order.
    """
    relative = mat_to_6d(robot_R.transpose(-1,-2)[...,None,:,:] @ root_R)
    # Preserve release concat(q_flat, qd_flat).reshape(10,58) packing.
    qv = torch.cat((q["motion"].flatten(-2),qd.flatten(-2)),-1).unflatten(-1,(10,58))
    motion = torch.cat((qv,relative),-1).flatten(-2)
    smpl_ori = mat_to_6d(robot_R.transpose(-1,-2)[...,None,:,:] @ smpl_R)
    smpl = torch.cat((smpl_local.flatten(-2),smpl_ori,q["smpl"][...,wrists]),-1).flatten(-2)
    teleop = torch.cat((q["motion"][...,lower].flatten(-2),qd[...,lower].flatten(-2),
                        points.flatten(-2),mat_to_6d(point_R).flatten(-2),relative[...,0,:]),-1)
    return dict(motion=motion,smpl=smpl,teleop=teleop)

