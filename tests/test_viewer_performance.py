import mujoco
import numpy as np
import pytest
from blinkmimic.deploy.viewer_utils import ControlClock,RenderMirror

def test_compute_time_is_inside_period():
    c=ControlClock(.02,0.)
    c.advance(.007)
    assert c.deadline==pytest.approx(.02)
    assert not c.due(.019) and c.due(.02)

def test_overrun_does_not_add_extra_sleep_or_catchup_debt():
    c=ControlClock(.02,0.)
    c.advance(.075)
    assert c.deadline==pytest.approx(.075)
    c.advance(.08)
    assert c.deadline==pytest.approx(.095)
    c.set_period(.04,10.)
    assert c.due(10.)

def test_render_mirror_is_not_the_physics_data(monkeypatch):
    m=mujoco.MjModel.from_xml_string('<mujoco><worldbody><body><freejoint/><geom type="sphere" size=".1" mass="1"/></body></worldbody></mujoco>')
    d=mujoco.MjData(m);d.qpos[:3]=[1,2,3];d.time=.2
    def forbidden(*args):raise AssertionError("render must not run dynamics")
    monkeypatch.setattr(mujoco,"mj_step",forbidden)
    monkeypatch.setattr(mujoco,"mj_forward",forbidden)
    mirror=RenderMirror(m);mirror.update(d)
    np.testing.assert_allclose(mirror.d.geom_xpos[0],[1,2,3])
    mirror.d.qpos[0]=9
    assert d.qpos[0]==1 and mirror.d.time==.2

def test_compact_pack_preserves_large_integer_counts():
    import warp as wp
    from blinkmimic.deploy.warp_official import pack_state
    with wp.ScopedDevice("cpu"):
        arr=lambda x,dtype=float:wp.array(np.asarray(x),dtype=dtype,device="cpu")
        packed=wp.zeros(2+1+5,dtype=wp.float64,device="cpu")
        wp.launch(pack_state,dim=8,inputs=[arr([[.25,.5]]),arr([[.75]]),arr([.125]),
                  arr([16777217,20000001],int),arr([3.,4.]),packed,2,1],device="cpu")
        np.testing.assert_array_equal(packed.numpy(),[.25,.5,.75,.125,16777217,20000001,3.,4.])
