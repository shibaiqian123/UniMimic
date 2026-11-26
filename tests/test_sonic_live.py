import numpy as np
import pytest
from blinkmimic.deploy.sonic_wire import pack,unpack,ISAAC_NAMES
from blinkmimic.deploy.sonic_source import Source
from blinkmimic.deploy.sonic_control import inputs
from blinkmimic.deploy.core import JOINTS,rotation
from blinkmimic.deploy.sonic_planner import sample_qpos

@pytest.mark.parametrize("size",[1024,1280])
def test_wire(size):
    values={"start":np.array([1],np.uint8),"stop":np.array([0],np.uint8),"planner":np.array([1],np.uint8)}
    topic,out=unpack(pack("command",values,size))
    assert topic=="command"
    for k in values:np.testing.assert_array_equal(values[k],out[k])
    with pytest.raises(ValueError):unpack(pack("command",values,size)[:-1])
def test_stop_wins():
    s=Source();s.accept("command",{"start":[1],"stop":[1],"planner":[1]},1.)
    assert s.start==0 and s.stop==1

def pose(ids):
    n=len(ids)
    return dict(frame_index=np.array(ids),smpl_joints=np.zeros((n,24,3)),
                body_quat_w=np.tile([1.,0,0,0],(n,1)),joint_pos=np.zeros((n,29)))
def test_pose_window_and_freshness():
    s=Source();s.accept("pose",pose(range(3)),1.)
    assert s.pose_window()["frame"]==0
    assert s.pose_window()["smpl"].shape==(10,24,3)
    s.accept("pose",pose(range(3,10)),2.)
    assert s.pose_window()["frame"]==3
    s.accept("pose",pose(range(3,10)),3.)
    assert s.pose_time==2.
    s.accept("command",dict(planner=[0]),3.)
    assert s.pose_window() is None

def test_reference_rotation_and_joint_mapping():
    from types import SimpleNamespace
    wrists=[JOINTS.index(f"{s}_wrist_{a}_joint") for a in ("roll","pitch","yaw") for s in ("left","right")]
    b=SimpleNamespace(cfg=dict(omni_wrists=wrists,omni_lower=list(range(12))))
    qpos=np.zeros((46,36));qpos[:,3]=1
    p=pose(range(10));p["joint_pos"][:]=np.arange(29)
    msg=dict(mode="POSE",planner={},qpos=qpos,qd=np.zeros((46,29)),
             pose=dict(smpl=p["smpl_joints"],quat=p["body_quat_w"],joints=p["joint_pos"]))
    R=rotation([.9238795,0,0,.3826834])
    modality,x,ref,_=inputs(msg,b,R,np.eye(3),np.eye(3))
    assert modality=="smpl" and x.shape==(840,)
    np.testing.assert_allclose(x.reshape(10,84)[:,72:78],np.tile(R.T[:,:2].reshape(6),(10,1)))
    expected=[ISAAC_NAMES.index(JOINTS[i]) for i in wrists]
    np.testing.assert_array_equal(x.reshape(10,84)[0,-6:],expected)
    msg["mode"]="VR_3PT";msg["planner"]=dict(vr_position=np.arange(9),vr_orientation=np.tile([1,0,0,0],3))
    modality,x,_,_=inputs(msg,b,R,np.eye(3),np.eye(3))
    assert modality=="teleop" and x.shape==(273,)
    np.testing.assert_allclose(x[-6:],R.T[:,:2].reshape(6))
    msg["mode"]="PLANNER";assert inputs(msg,b,R,np.eye(3),np.eye(3))[1].shape==(640,)

def test_slerp_sign_and_clamp():
    x=np.zeros((4,36),np.float32);x[:,3]=[1,-1,1,-1];x[:,7]=np.arange(4)
    y=sample_qpos(x,[-1,.5,3,5])
    np.testing.assert_allclose(abs(y[:,3]),1)
    np.testing.assert_allclose(y[:,7],[0,.5,3,3])

def test_single_frame_padding_does_not_hide_dropout():
    s=Source()
    x=pose([42]);x["smpl_joints"][:]=7
    s.accept("pose",x,1.)
    np.testing.assert_array_equal(s.pose_window()["smpl"],np.full((10,24,3),7))
    s.accept("pose",x,9.)
    assert s.pose_time==1.
    assert s.pose_window()["frame"]==42
