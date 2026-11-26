from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pytest
from blinkmimic.deploy.hardware import gain_scales,apply_gains,release_motion
from blinkmimic.deploy.core import transform_imu,rotation,JOINTS
from blinkmimic.deploy.sonic_source import Source
from test_sonic_live import pose

def test_gain_motor_order_and_override():
    assert JOINTS[4]=="left_ankle_pitch_joint" and JOINTS[10]=="right_ankle_pitch_joint"
    b=SimpleNamespace(kp=np.arange(1,30,dtype=float),kd=np.ones(29))
    log=apply_gains(b,"4,10=1.5;15=2","4,10=0.8")
    assert b.kp[4]==7.5 and b.kp[10]==16.5 and b.kp[15]==32
    assert b.kd[4]==.8 and b.kp[0]==1
    assert log["kp"]==b.kp.tolist()
    for x in ("29=1","-1=2","1=nan","1=0","1=2;1=3"):
        with pytest.raises(ValueError):gain_scales(x)

def test_release_never_runs_in_sim_or_readonly_and_has_timeout():
    with patch("blinkmimic.deploy.hardware.subprocess.run") as run:
        release_motion(Path("/bridge"),"eth0",0,read_only=True)
        release_motion(Path("/bridge"),"lo",2,sim=True)
        run.assert_not_called()
        run.return_value=SimpleNamespace(stdout="released")
        release_motion(Path("/bridge"),"eth0",0)
        assert run.call_args.kwargs["timeout"]==35
        assert run.call_args.args[0]==["/bridge/g1_release_mode","eth0"]

def test_imu_wxyz_mount_and_sign():
    mount=rotation([np.cos(.2),np.sin(.2),0,0])
    q=np.array([np.cos(.3),0,0,np.sin(.3)])
    R,g=transform_imu(q,[1,2,3],mount)
    np.testing.assert_allclose(R,rotation(q)@mount.T)
    np.testing.assert_allclose(g,mount@np.array([1,2,3]))
    np.testing.assert_allclose(transform_imu(-q,[1,2,3],mount)[0],R)

def test_heading_increment_applied_once_and_reset_explicit():
    s=Source();x=pose([0]);x["heading_increment"]=np.array([.2])
    s.accept("pose",x,1);s.accept("pose",x,2)
    assert s.heading_delta==.2
    s.accept("command",dict(planner=[0],delta_heading=[.5]),3)
    assert s.heading_delta==.5 and s.pose_window() is None
    x=pose([0]);x["heading_increment"]=np.array([.1])
    s.accept("pose",x,4)
    assert s.heading_delta==.6

def test_modes_keep_heading_error_and_history():
    from blinkmimic.deploy.sonic_control import LiveCore
    from blinkmimic.deploy.core import Bundle,yaw_rotation
    path=Path("deploy_artifacts/omni_15000_walk_A049")
    if not path.exists():pytest.skip("real omni bundle not available")
    b=Bundle(path);b.history_len=10
    c=LiveCore(b,{m:lambda h,x:np.zeros(29) for m in ("motion","smpl","teleop")})
    c.align(yaw_rotation(.4))
    qpos=np.zeros((46,36));qpos[:,3]=1
    message=dict(mode="PLANNER",qpos=qpos,qd=np.zeros((46,29)),planner={},
                 pose=None,source_time=0.,heading_delta=0.)
    c.message=message;c.step(b.q0,np.zeros(29),np.zeros(3),yaw_rotation(.2))
    ref=c.last_reference.copy()
    p=pose([0]);s=Source();s.accept("pose",p,1)
    message.update(mode="POSE",pose=s.pose_window())
    c.step(b.q0,np.zeros(29),np.zeros(3),yaw_rotation(.2))
    np.testing.assert_allclose(c.last_reference,ref)
    assert c.history.shape==(10,93)
    message["heading_delta"]=.1
    c.step(b.q0,np.zeros(29),np.zeros(3),yaw_rotation(.2))
    np.testing.assert_allclose(c.last_reference,yaw_rotation(.1)@ref,atol=1e-7)
    ref=c.last_reference.copy();message["mode"]="PLANNER"
    c.step(b.q0,np.zeros(29),np.zeros(3),yaw_rotation(.2))
    np.testing.assert_allclose(c.last_reference,ref,atol=1e-7)

def test_repackaged_old_sensor_frame_does_not_refresh():
    s=Source();x=pose([0]);x["timestamp_monotonic"]=np.array([10.])
    s.accept("pose",x,11.)
    x["frame_index"]=np.array([1])
    s.accept("pose",x,12.)
    assert s.pose_time==11. and s.last_frame==0
    x["timestamp_monotonic"]=np.array([10.02])
    s.accept("pose",x,12.02)
    assert s.pose_time==12.02 and s.last_frame==1

def test_all_runner_send_paths_use_received_machine_mode():
    # Includes exception damping: never reintroduce the simulator-only value 5.
    import ast
    tree=ast.parse(Path("scripts/deploy/run_g1.py").read_text())
    calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call)
           and isinstance(n.func,ast.Attribute) and n.func.attr=="send"
           and isinstance(n.func.value,ast.Name) and n.func.value.id=="bridge"]
    assert len(calls)==2
    for call in calls:
        assert ast.unparse(call.args[4])=="s['mode_machine']"
