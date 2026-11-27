"""Live startup sequencing; no DDS, robot, or inference backend is launched."""
from types import SimpleNamespace
import numpy as np
import pytest
from blinkmimic.deploy import sonic_control as sc
from blinkmimic.deploy.controller import STOP


def setup_controller(monkeypatch):
    c=sc.LiveController.__new__(sc.LiveController)
    c.b=SimpleNamespace(q0=np.ones(29)*.1,kp=np.ones(29)*20,kd=np.ones(29),
                        cfg={"preparation_gain_scale":1.})
    c.stage={"max_action":20}
    c.limits={"interpolation_s":2.,"hold_s":1.,"policy_tilt":.35}
    c.client=SimpleNamespace(value=None)
    c.client.poll=lambda:c.client.value
    c.core=SimpleNamespace(frame=0,history=None,align=lambda R:None)
    def step(*args):
        c.core.frame+=1
        return dict(action=np.zeros(29),target=c.b.q0.copy(),frame=c.core.frame,
                    reference_R=np.eye(3),inference_ms=0.)
    c.core.step=step
    c.mount=None;c.state="PASSIVE";c.reason="";c.reject_reason=""
    c.since=0.;c.starts=0;c.stops=0;c.from_q=None
    c.live_timeout=1.;c.live_mode="OFF";c.prepare_requested=False
    c.target_transition=sc.ModeTransition()
    monkeypatch.setattr(sc,"state_guard",lambda *a,**k:None)
    monkeypatch.setattr(sc,"transform_imu",lambda *a:(np.eye(3),np.zeros(3)))
    monkeypatch.setattr(sc,"limit_target",lambda target,*a:(target,0))
    state=dict(q=np.zeros(29),dq=np.zeros(29),keys=0,quat=np.array([1.,0,0,0]),gyro=np.zeros(3))
    return c,state


def msg(now,start=0,stop=0,mode="PLANNER"):
    return dict(t=now,source_time=now,error="",start=start,stop=stop,mode=mode,plan_end=now+2)


def prepare(c,s):
    c.request_prepare()
    assert c.update(s,0.)["state"]=="INTERP"
    assert c.update(s,2.)["state"]=="HOLD"


def test_confirmation_then_hold_then_abxy(monkeypatch):
    c,s=setup_controller(monkeypatch)
    # A remote start cannot replace the runner's explicit preparation request.
    c.client.value=msg(0.,start=1)
    assert c.update(s,0.)["state"]=="PASSIVE"
    prepare(c,s)
    c.client.value=None
    assert c.update(s,10.)["state"]=="HOLD"
    assert c.core.frame==0
    np.testing.assert_allclose(c.update(s,11.)["target"],c.b.q0)
    c.client.value=msg(12.,start=2)
    assert c.update(s,12.)["state"]=="POLICY"
    assert c.update(s,12.02)["live_mode"]=="PLANNER"
    assert c.core.frame==1


def test_early_start_is_consumed_not_queued(monkeypatch):
    c,s=setup_controller(monkeypatch)
    c.request_prepare();c.update(s,0.)
    c.client.value=msg(1.,start=1)
    c.update(s,1.);c.update(s,2.)
    c.client.value=msg(2.5,start=2)
    assert c.update(s,2.5)["state"]=="HOLD"
    assert "settling" in c.reject_reason
    c.client.value=msg(4.,start=2)
    assert c.update(s,4.)["state"]=="HOLD"
    c.client.value=msg(4.1,start=3)
    assert c.update(s,4.1)["state"]=="POLICY"


@pytest.mark.parametrize("mode",["POSE","VR_3PT"])
def test_start_requires_planner(monkeypatch,mode):
    c,s=setup_controller(monkeypatch);prepare(c,s)
    c.client.value=msg(4.,start=1,mode=mode)
    assert c.update(s,4.)["state"]=="HOLD"
    assert "PLANNER" in c.reject_reason


def test_stale_entry_rejected_and_policy_disconnect_latched(monkeypatch):
    c,s=setup_controller(monkeypatch);prepare(c,s)
    c.client.value=msg(0.,start=1)
    assert c.update(s,4.)["state"]=="HOLD"
    assert "stale" in c.reject_reason
    c.client.value=msg(4.1,start=2)
    assert c.update(s,4.1)["state"]=="POLICY"
    c.client.value=msg(5.2,start=2);c.client.value["source_time"]=4.1
    assert c.update(s,5.2)["state"]=="DAMP"
    assert c.reason=="VR input stream stale"
    c.request_prepare();c.client.value=msg(6.,start=3)
    assert c.update(s,6.)["state"]=="DAMP"


@pytest.mark.parametrize("remote",["g1","pico"])
def test_stop_wins_during_preparation(monkeypatch,remote):
    c,s=setup_controller(monkeypatch);c.request_prepare()
    if remote=="g1":s["keys"]=STOP
    else:c.client.value=msg(0.,start=1,stop=1)
    assert c.update(s,0.)["state"]=="DAMP"
    assert c.core.frame==0


def test_bridge_fault_still_blocks_preparation(monkeypatch):
    c,s=setup_controller(monkeypatch);c.request_prepare()
    assert c.update(s,0.,"watchdog")["state"]=="DAMP"
