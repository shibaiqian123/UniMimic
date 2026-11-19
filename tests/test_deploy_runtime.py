"""Hardware-free tests of the actual deployment state machine and sensor contract."""
from types import SimpleNamespace
import numpy as np
import pytest
from blinkmimic.deploy.core import DeployCore,rotation,yaw_rotation,sensor_transform,transform_imu,STAGES,LIMITS
from blinkmimic.deploy.controller import Controller,PREP,ARM,PLAY,STOP

PROFILE=dict(imu_frame="pelvis_rigid",quaternion_order="wxyz",gyro_units="rad/s",R_pelvis_from_imu_wxyz=[1,0,0,0])
def bundle():
    return SimpleNamespace(T=3,q0=np.zeros(29),scale=np.ones(29)*.1,kp=np.ones(29)*50,kd=np.ones(29),
        effort=np.ones(29)*100,velocity=np.ones(29)*20,lower=np.ones(29)*-2,upper=np.ones(29)*2,
        cfg=dict(stages=STAGES,limits=LIMITS),motion=dict(q=np.zeros((3,29)),root_R=np.repeat(np.eye(3)[None],3,axis=0),
        goal=np.ones((3,730)),static_goal=np.stack([np.zeros(730),np.full(730,2.)])))
def state(t):
    return dict(q=np.zeros(29),dq=np.zeros(29),gyro=np.zeros(3),quat=np.array([1.,0,0,0]),temp=np.full((29,2),30.),
                motorstate=np.zeros(29),mode_machine=5,mode_pr=0,t_rx=t,t_tick_change=t,keys=0)
def test_heading_history_and_raw_action():
    b=bundle(); c=DeployCore(b,lambda h,g:np.ones(29)*.2)
    R=yaw_rotation(1.2); c.align(R)
    first=c.step(np.zeros(29),np.zeros(29),np.zeros(3),R)
    np.testing.assert_allclose(c.history[0],c.history[-1])
    np.testing.assert_allclose(c.history[-1,-6:],[1,0,0,1,0,0],atol=1e-12)
    c.start_motion(); c.step(np.zeros(29),np.zeros(29),np.zeros(3),R)
    np.testing.assert_allclose(c.history[-1,64:93],.2)
    assert c.frame==1
    with pytest.raises(ValueError): c.align(R)
    c.step(np.zeros(29),np.zeros(29),np.zeros(3),R)
    c.step(np.zeros(29),np.zeros(29),np.zeros(3),R)
    assert c.phase=="post" and c.reference()[0]==2 and np.all(c.reference()[1]==2)
def test_imu_mount_and_sign():
    q=np.array([np.cos(.3),0,0,np.sin(.3)])
    np.testing.assert_allclose(rotation(q),rotation(-q))
    mount=rotation(q); R,g=transform_imu(q,[1,0,0],mount)
    np.testing.assert_allclose(R,np.eye(3),atol=1e-12)
    np.testing.assert_allclose(g,mount[:,0])
    with pytest.raises(ValueError): sensor_transform(dict(PROFILE,imu_frame="torso"))
@pytest.mark.parametrize("fault",["stale","nan","hot","motor","locked","quaternion","bridge"])
def test_fault_latches(fault):
    c=Controller(bundle(),PROFILE,"ground",lambda h,g:np.zeros(29)); s=state(10.)
    bridge=None
    if fault=="stale": s["t_rx"]=0.
    if fault=="nan": s["q"][0]=np.nan
    if fault=="hot": s["temp"][0,0]=90.
    if fault=="motor": s["motorstate"][0]=1
    if fault=="locked": s["mode_machine"]=6
    if fault=="quaternion": s["quat"]*=2
    if fault=="bridge": bridge="watchdog"
    out=c.update(s,10.,bridge)
    assert c.state=="DAMP" and np.all(out["kp"]==0)
    c.update(state(11.),11.)
    assert c.state=="DAMP"
def test_complete_state_sequence():
    c=Controller(bundle(),PROFILE,"ground",lambda h,g:np.zeros(29))
    states=set()
    for i in range(550):
        t=i*.02; s=state(t)
        if .1<t<.4:s["keys"]=PREP
        if 4.5<t<4.8:s["keys"]=ARM
        if 8.<t<8.3:s["keys"]=PLAY
        if t>10:s["keys"]=STOP
        c.update(s,t); states.add(c.state)
    assert states=={"PASSIVE","INTERP","HOLD","STANDBY_PRE","POLICY","STANDBY_POST","DAMP"}
def test_nonfinite_policy_damps():
    c=Controller(bundle(),PROFILE,"ground",lambda h,g:np.full(29,np.nan))
    c.core.align(np.eye(3)); c.state="STANDBY_PRE"
    c.update(state(1.),1.)
    assert c.state=="DAMP" and "policy action" in c.reason
def test_timeout_damps():
    import time
    def slow(h,g): time.sleep(.025); return np.zeros(29)
    c=Controller(bundle(),PROFILE,"ground",slow); c.core.align(np.eye(3)); c.state="STANDBY_PRE"
    c.update(state(1.),1.)
    assert c.state=="DAMP" and "timeout" in c.reason

def test_preparation_gains_and_policy_gains():
    b=bundle(); b.cfg["preparation_gain_scale"]=4.
    c=Controller(b,PROFILE,"ground",lambda h,g:np.zeros(29))
    c.update(state(0.),0.)
    s=state(.1); s["keys"]=PREP; c.update(s,.1)
    s=state(.2); s["keys"]=PREP; out=c.update(s,.2)
    np.testing.assert_allclose(out["kp"],b.kp*4)
    np.testing.assert_allclose(out["kd"],b.kd*2)
    c.state="STANDBY_PRE"; out=c.update(state(.2),.2)
    np.testing.assert_allclose(out["kp"],b.kp)
    np.testing.assert_allclose(out["kd"],b.kd)

def test_force_limit_uses_effective_gains():
    from blinkmimic.deploy.core import limit_target
    b=bundle(); stage=b.cfg["stages"]["air"]
    q=np.zeros(29); dq=np.full(29,.2)
    kp=b.kp*4; kd=b.kd*2
    target,n=limit_target(np.ones(29)*2,q,dq,b,stage,kp,kd)
    assert n==29
    assert np.max(np.abs(kp*(target-q)-kd*dq))<=50.+1e-9

def test_preheld_button_does_not_arm():
    c=Controller(bundle(),PROFILE,"ground",lambda h,g:np.zeros(29))
    for t in (0.,.1,.2):
        s=state(t); s["keys"]=PREP; assert c.update(s,t) is None
    assert c.state=="PASSIVE"
    c.update(state(.3),.3)
    for t in (.4,.5):
        s=state(t); s["keys"]=PREP; c.update(s,t)
    assert c.state=="INTERP"
