"""Simulator packet adapters; reference features never enter the new decoder history."""
import torch
from .observation_contract import MODES,contract,validate_args
from .rotations import mat_to_6d

def future_orientation(motion,frames,robot_R,K=10,heading=None):
    """Differentiate through the current robot rotation; clamp within each source clip."""
    refs=motion.root_R[motion.future_frames(frames,K)]
    if heading is not None:refs=heading.unsqueeze(-3)@refs
    return mat_to_6d(robot_R.transpose(-1,-2).unsqueeze(-3)@refs)

def orientation_for(policy,motion,frames,robot_R,K=10):
    if getattr(policy,"observation_mode","legacy99")=="legacy99":return None
    return future_orientation(motion,frames,robot_R,K)

def environment_orientation(policy,env,motion,K=10):
    if getattr(policy,"observation_mode","legacy99")=="legacy99":return None
    return future_orientation(motion,env.f,env.state()[1],K)

def normalizer_inputs(policy,observation,goal,orientation6d_future=None):
    if getattr(policy,"observation_mode","legacy99")=="legacy99":return observation,goal
    if observation.shape[-1]!=99:raise ValueError("training packet must contain 93 proprio + 6 legacy command")
    if orientation6d_future is None:raise ValueError("explicit future orientation required")
    return observation[...,:93],torch.cat((goal,orientation6d_future.flatten(-2)),-1)

def policy_action(policy,history,goal,orientation6d_future=None):
    if getattr(policy,"observation_mode","legacy99")=="legacy99":return policy(history,goal)
    if history.shape[-1]!=99:raise ValueError("training history must be a 99D feature packet")
    if orientation6d_future is None:raise ValueError("explicit future orientation required")
    return policy(history[...,:93],goal,orientation6d_future=orientation6d_future)

def update_norms(policy,observation,goal,orientation6d_future=None):
    o,g=normalizer_inputs(policy,observation,goal,orientation6d_future)
    policy.obs_norm.update(o);policy.goal_norm.update(g)

def diagnostic_orientation(policy,motion,frames):
    if getattr(policy,"observation_mode","legacy99")=="legacy99":return None
    # Perfectly aligned NOW does not mean future reference orientations are identity.
    return future_orientation(motion,frames,motion.root_R[frames],10)

def matrix_from_6d(value):
    columns=value.reshape(*value.shape[:-1],3,2)
    b1=torch.nn.functional.normalize(columns[...,0],dim=-1)
    a2=columns[...,1]
    b2=torch.nn.functional.normalize(a2-(b1*a2).sum(-1,keepdim=True)*b1,dim=-1)
    return torch.stack((b1,b2,torch.linalg.cross(b1,b2,dim=-1)),-1)
