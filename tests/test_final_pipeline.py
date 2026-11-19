import json
from types import SimpleNamespace
import numpy as np
from blinkmimic.deploy.buttons import ButtonPulses
from blinkmimic.deploy.controller import PREP,ARM,PLAY,STOP,Controller
from blinkmimic.deploy.core import state_guard,limit_target,rotation
from blinkmimic.deploy.imu_model import ImuModel
from blinkmimic.deploy.lowstate import decode_snapshot
from blinkmimic.deploy.session import DEFAULT_PROFILE
from test_deploy_runtime import bundle,state

def test_pulse_duration_release_repeat_stop_priority():
    a=ButtonPulses()
    assert a.send("1",0.)
    assert a.sample(.149)==PREP
    assert not a.send("2",.05)
    assert a.sample(.151)==0
    assert not a.send("2",.20)
    assert a.send("2",.251)
    assert a.send("D",.26)
    a.release(.27)
    assert a.sample(.28)==STOP
    assert a.sample(.411)==0

def test_focus_loss_releases_normal_key_without_changing_controller():
    a=ButtonPulses();a.send("3",1.);a.release(1.02)
    assert a.sample(1.03)==0
    assert not a.send("1",1.05)

def test_keyboard_transport_paths_have_identical_bits():
    window=ButtonPulses();terminal=ButtonPulses()
    for k,t in [("1",0.),("2",.3),("3",.6),("D",.61)]:
        packet=json.loads(json.dumps({"type":"key","key":k}))
        window.send(k,t);terminal.send(packet["key"],t)
        assert window.sample(t+.01)==terminal.sample(t+.01)

def test_estimated_imu_not_reset_by_truth():
    imu=ImuModel(np.array([1.,0,0,0]))
    imu.integrate(np.array([0.,0.,12.]),.1)
    before=imu.q.copy()
    gyro,q=imu.observe([0.,0.,12.],np.array([1.,0,0,0]),0)
    np.testing.assert_array_equal(q,before)
    assert gyro[2]==imu.limit
    assert abs(q[3])>.1

def test_refused_start_is_reported_not_a_fault():
    c=Controller(bundle(),DEFAULT_PROFILE,"ground",lambda h,g:np.zeros(29))
    c.core.align(np.eye(3));c.state="HOLD";c.since=0
    for t in (2.,2.08):
        s=state(t);s["q"][0]=.2;s["keys"]=ARM
        if t==2.:c.keys_released=True
        out=c.update(s,t)
    assert c.state=="HOLD"
    assert out["reject_reason"]=="not at start pose"

def test_position_limit_is_not_an_added_software_stop_or_clip():
    b=bundle();s=state(1.);s["q"][0]=2.2
    assert state_guard(s,1.,b,b.cfg["stages"]["ground"]) is None
    b.effort[:]=10000
    target,n=limit_target(np.ones(29)*3,np.zeros(29),np.zeros(29),b,b.cfg["stages"]["ground"])
    np.testing.assert_allclose(target,3.)
    assert n==0

def test_lowstate_roundtrip_preserves_observation_6d_goal_and_actions():
    b=bundle()
    policy=lambda h,g:np.tanh(h[-1,:29])*.01
    direct=Controller(b,DEFAULT_PROFILE,"ground",policy)
    decoded=Controller(b,DEFAULT_PROFILE,"ground",policy)
    for i in range(650):
        t=i*.02;s=state(t)
        s["q"][:]=.001*np.sin(t)
        if .1<t<.4:s["keys"]=PREP
        if 5.5<t<5.8:s["keys"]=ARM
        if 9.<t<9.3:s["keys"]=PLAY
        wire={k:np.asarray(s[k],np.float32).astype(float) for k in ("q","dq","temp","motorstate","gyro","quat")}
        wire.update(acc=np.zeros(3),t_rx=t,t_tick_change=t,mode_machine=5,mode_pr=0,tick=i,
                    wireless_remote=np.array([0,0,s["keys"]&255,s["keys"]>>8]+[0]*36,np.uint8))
        result1=direct.update(s,t);result2=decoded.update(decode_snapshot(wire),t)
        assert direct.state==decoded.state
        if direct.core.history is not None:
            np.testing.assert_allclose(direct.core.history,decoded.core.history,atol=1e-7)
            np.testing.assert_array_equal(direct.core.reference()[1],decoded.core.reference()[1])
            np.testing.assert_allclose(result1["action"],result2["action"],atol=1e-7)

def test_actual_window_and_terminal_event_builders_match():
    from blinkmimic.deploy.buttons import WindowInput,key_event
    emitted=[];window=WindowInput(emitted.append)
    for k in ("1","2","3","D","B"):
        assert window.key(ord(k),1)
        assert emitted[-1]==key_event(k)
        n=len(emitted)
        assert not window.key(ord(k),2)
        assert len(emitted)==n
    window.focus(False)
    assert emitted[-1]=={"type":"release"}
