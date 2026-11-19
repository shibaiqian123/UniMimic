"""Isolated CPU MuJoCo DDS robot. Domain 1/lo only; never connects to a real robot."""
import argparse,json,sys,time,threading
from pathlib import Path
import mujoco
import numpy as np
from unitree_sdk2py.core.channel import ChannelFactoryInitialize,ChannelPublisher,ChannelSubscriber
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_,LowState_
from unitree_sdk2py.utils.crc import CRC
from blinkmimic.deploy.core import Bundle,rotation

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bundle",type=Path,required=True)
    ap.add_argument("--scene",type=Path,required=True)
    ap.add_argument("--buttons",type=Path,required=True,help="JSON with keys and fault injection fields")
    ap.add_argument("--duration",type=float,default=120.)
    ap.add_argument("--fixture",action="store_true",help="sensor replay for transport tests, NOT physics acceptance")
    ap.add_argument("--domain",type=int,choices=[1,2],default=1)
    a=ap.parse_args()
    b=Bundle(a.bundle); m=mujoco.MjModel.from_xml_path(str(a.scene)); d=mujoco.MjData(m)
    qi=np.array([m.jnt_qposadr[m.joint(n).id] for n in b.cfg["joint_names"]])
    vi=np.array([m.jnt_dofadr[m.joint(n).id] for n in b.cfg["joint_names"]])
    d.qpos[:3]=b.motion["root_p"][0]
    mujoco.mju_mat2Quat(d.qpos[3:7],b.motion["root_R"][0].reshape(9))
    d.qpos[qi]=b.motion["q"][0]; mujoco.mj_forward(m,d)
    ChannelFactoryInitialize(a.domain,"lo"); crc=CRC(); lock=threading.Lock(); latest=[None]; ncmd=[0]
    def receive(msg):
        if crc.Crc(msg)!=msg.crc: return
        with lock: latest[0]=msg; ncmd[0]+=1
    sub=ChannelSubscriber("rt/lowcmd",LowCmd_); sub.Init(receive,10)
    pub=ChannelPublisher("rt/lowstate",LowState_); pub.Init()
    t0=time.monotonic(); tick=0; deadline=t0; last_status=0.
    while time.monotonic()-t0<a.duration:
        now=time.monotonic()
        try: cfg=json.loads(a.buttons.read_text())
        except (OSError,ValueError): cfg={}
        with lock: cmd=latest[0]
        if cmd is not None and not a.fixture:
            for i in range(29):
                c=cmd.motor_cmd[i]
                m.actuator_gainprm[i,0]=c.kp
                m.actuator_biasprm[i,1]=-c.kp
                m.actuator_biasprm[i,2]=-c.kd
                d.ctrl[i]=c.q
            d.xfrc_applied[:]=0.
            if cfg.get("support",False):
                bid=m.body("pelvis").id
                anchor=b.motion["root_p"][0]
                d.xfrc_applied[bid,:3]=2000*(anchor-d.qpos[:3])-100*d.qvel[:3]+np.array([0,0,m.body_mass.sum()*9.81])
                R=rotation(d.qpos[3:7]); E=b.motion["root_R"][0]@R.T
                err=np.array([E[2,1]-E[1,2],E[0,2]-E[2,0],E[1,0]-E[0,1]])*.5
                d.xfrc_applied[bid,3:]=500*err-50*(R@d.qvel[3:6])
            mujoco.mj_step(m,d)
        if not cfg.get("drop_state",False):
            msg=unitree_hg_msg_dds__LowState_()
            tick+=1; msg.tick=tick; msg.mode_machine=5; msg.mode_pr=0
            msg.imu_state.quaternion[:]=d.qpos[3:7].tolist()
            msg.imu_state.gyroscope[:]=d.qvel[3:6].tolist()
            for i in range(29):
                msg.motor_state[i].q=float(d.qpos[qi[i]])
                msg.motor_state[i].dq=float(d.qvel[vi[i]])
                msg.motor_state[i].temperature[:]=[int(cfg.get("temperature",30))]*2
            keys=int(cfg.get("keys",0)); msg.wireless_remote[2]=keys&255; msg.wireless_remote[3]=(keys>>8)&255
            if cfg.get("nan_state"): msg.motor_state[0].q=float("nan")
            msg.crc=crc.Crc(msg); pub.Write(msg)
        if now-last_status>.05:
            status=dict(sim_time=float(d.time),wall_time=now-t0,n_commands=ncmd[0],kp=[float(cmd.motor_cmd[i].kp) for i in range(29)] if cmd is not None else [],
                        kd=[float(cmd.motor_cmd[i].kd) for i in range(29)] if cmd is not None else [],t=now)
            tmp=a.buttons.with_suffix(".status.tmp"); tmp.write_text(json.dumps(status))
            tmp.replace(a.buttons.with_suffix(".status.json")); last_status=now
        deadline+=m.opt.timestep
        time.sleep(max(0,deadline-time.monotonic()))
if __name__=="__main__": main()
