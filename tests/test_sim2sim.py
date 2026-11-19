"""Unit tests independent of checkpoints: real MuJoCo motor mapping, PD and delayed commands."""
from types import SimpleNamespace
import numpy as np
import pytest
from blinkmimic.deploy.core import JOINTS
from blinkmimic.deploy.sim2sim import Plant

def make(tmp_path,latency=0):
    # Reverse actuator order deliberately; named joint mapping must still work.
    bodies="".join(f'<body name="b{i}" pos="0 0 {0.01*i}"><joint name="{n}" axis="0 0 1" range="-2 2"/><geom type="sphere" size=".01" mass=".1"/></body>' for i,n in enumerate(JOINTS))
    acts="".join(f'<motor name="m{i}" joint="{n}" ctrlrange="-2 2"/>' for i,n in reversed(list(enumerate(JOINTS))))
    xml=tmp_path/"test.xml"
    xml.write_text(f'<mujoco><compiler angle="radian"/><option gravity="0 0 0"/><worldbody><body name="pelvis" pos="0 0 1"><freejoint/><geom type="sphere" size=".1" mass="5"/>{bodies}<body name="left_ankle_roll_link"><geom type="sphere" size=".01" mass=".1"/></body></body></worldbody><actuator>{acts}</actuator></mujoco>')
    b=SimpleNamespace(cfg=dict(joint_names=JOINTS),motion=dict(root_p=np.array([[0,0,1.],[0,0,1.]]),root_R=np.repeat(np.eye(3)[None],2,0),q=np.zeros((2,29))),
        kp=np.ones(29)*10,kd=np.ones(29),effort=np.ones(29)*5,velocity=np.ones(29)*10,lower=np.ones(29)*-2,upper=np.ones(29)*2)
    p=Plant(b,xml,"g2",latency_ms=latency);return p
def test_named_actuator_map_and_pd_saturation(tmp_path):
    p=make(tmp_path)
    assert p.act.tolist()==list(reversed(range(29)))
    p.command(np.ones(29),p.b.kp,p.b.kd);p.step()
    assert p.torque_saturated>0 and p.model_saturated>0
    assert np.all(abs(p.d.ctrl)<=2)
def test_delay_queue(tmp_path):
    p=make(tmp_path,20)
    p.command(np.ones(29)*.1,p.b.kp,p.b.kd)
    p.step();np.testing.assert_allclose(p.target,0.)
    p.step();np.testing.assert_allclose(p.target,.1)
def test_delay_resolution(tmp_path):
    with pytest.raises(ValueError,match="latency"):
        make(tmp_path,.5)
