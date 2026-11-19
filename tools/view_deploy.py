"""Interactive CPU MuJoCo deployment viewer. No DDS, no hardware communication."""
import argparse
import json
import queue
import time
from pathlib import Path
import mujoco
import numpy as np
from blinkmimic.deploy.core import Bundle
from blinkmimic.deploy.controller import Controller, PREP, ARM, PLAY, STOP
from blinkmimic.deploy.simulation import CpuSim

PROFILE = dict(imu_frame="pelvis_rigid", quaternion_order="wxyz", gyro_units="rad/s",
               R_pelvis_from_imu_wxyz=[1,0,0,0])

class Session:
    def __init__(self,bundle,stage):
        self.b=bundle; self.stage=stage
        self.sim=CpuSim(bundle)
        self.ref=mujoco.MjData(self.sim.m)
        self.reset()
    def reset(self):
        self.sim.reset()
        self.controller=Controller(self.b,PROFILE,self.stage)
        self.support=True; self.t=0.; self.key=0; self.key_until=0.
        self.out=None; self.fault=""; self.frame=0
    def press(self,key):
        self.key=key; self.key_until=self.t+.16
    def tick(self):
        sim=self.sim; ctl=self.controller
        if self.t>=self.key_until:self.key=0
        q,dq,gyro,R=sim.state()
        s=dict(q=q,dq=dq,gyro=gyro,quat=sim.d.qpos[3:7].copy(),temp=np.full((29,2),30.),
               motorstate=np.zeros(29),mode_machine=5,mode_pr=0,t_rx=self.t,t_tick_change=self.t,keys=self.key)
        self.out=ctl.update(s,self.t)
        self.frame=self.out["frame"] if self.out is not None else 0
        if ctl.state=="DAMP":
            below=self.b.lower-q; above=q-self.b.upper
            j=int(np.argmax(np.maximum(below,above)))
            self.fault=ctl.reason
            if max(below[j],above[j])>0:
                self.fault+=f" | {self.b.cfg['joint_names'][j]}={q[j]:.4f}, limits [{self.b.lower[j]:.4f},{self.b.upper[j]:.4f}]"
            return False
        if self.out is not None:
            sim.step(self.out["target"],self.out["kp"],self.out["kd"],before_substep=self.support_forces)
        self.t+=.02
        return True
    def support_forces(self):
        # The support is a physics force: recompute every 2 ms, as in sim_dds.
        sim=self.sim
        sim.d.xfrc_applied[:]=0
        if not self.support:return
        R=sim.state()[3]; bid=sim.m.body("pelvis").id
        E=self.b.motion["root_R"][0]@R.T
        err=np.array([E[2,1]-E[1,2],E[0,2]-E[2,0],E[1,0]-E[0,1]])*.5
        sim.d.xfrc_applied[bid,:3]=2000*(self.b.motion["root_p"][0]-sim.d.qpos[:3])-100*sim.d.qvel[:3]+[0,0,sim.m.body_mass.sum()*9.81]
        sim.d.xfrc_applied[bid,3:]=500*err-50*(R@sim.d.qvel[3:6])
    def reference_scene(self,scene,enabled=True):
        scene.ngeom=0
        if not enabled:return
        f=min(self.frame,self.b.T-1)
        d=self.ref; m=self.sim.m
        d.qpos[:3]=self.b.motion["root_p"][f]
        heading=self.controller.core.heading
        R=self.b.motion["root_R"][f]
        if heading is not None:
            origin=self.b.motion["root_p"][0]
            d.qpos[:3]=origin+heading@(d.qpos[:3]-origin)
            R=heading@R
        mujoco.mju_mat2Quat(d.qpos[3:7],R.reshape(9))
        d.qpos[self.sim.qidx]=self.b.motion["q"][f]
        mujoco.mj_kinematics(m,d)
        for i in range(1,m.nbody):
            if scene.ngeom+2>scene.maxgeom:break
            p=d.xpos[i]; parent=m.body_parentid[i]
            g=scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(g,mujoco.mjtGeom.mjGEOM_SPHERE,[.025,0,0],p,np.eye(3).reshape(9),[.1,.6,1.,.75])
            scene.ngeom+=1
            if parent>0 and np.linalg.norm(p-d.xpos[parent])>1e-5:
                g=scene.geoms[scene.ngeom]
                mujoco.mjv_initGeom(g,mujoco.mjtGeom.mjGEOM_CAPSULE,[.012,0,0],[0,0,0],np.eye(3).reshape(9),[.1,.6,1.,.6])
                mujoco.mjv_connector(g,mujoco.mjtGeom.mjGEOM_CAPSULE,.012,d.xpos[parent],p)
                scene.ngeom+=1
    def text(self,paused,speed):
        ctl=self.controller
        latency=0 if self.out is None else self.out["inference_ms"]
        clamps=0 if self.out is None else self.out["n_clamped"]
        return (f"CPU simulation ONLY / no DDS\nState: {ctl.state}   {'PAUSED' if paused else 'RUNNING'}\n"
                f"Frame: {self.frame}/{self.b.T-1}  motion: {self.frame*.02:.2f}s  sim: {self.t:.2f}s\n"
                f"Speed: {speed:g}x   Support: {'ON' if self.support else 'OFF'}\n"
                f"Inference: {latency:.2f}ms   Clamped joints: {clamps}\n"
                f"Blue skeleton: reference (world trajectory)\n"
                f"{self.fault}\n\n"
                "1 prepare | 2 policy standby | B toggle support | 3 play\n"
                "SPACE pause | N single control step | +/- speed | R reset\n"
                "G reference | F follow camera | D stop (latched)\n"
                "GUI timing is NOT real-time DDS validation.")

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bundle",required=True,type=Path)
    ap.add_argument("--stage",choices=["air","feet","slack","ground"],default="ground")
    ap.add_argument("--speed",type=float,choices=[.1,.25,.5,1.,2.],default=1.)
    ap.add_argument("--auto",action="store_true",help="SIM ONLY: prepare, standby, release support, then request playback")
    ap.add_argument("--headless",action="store_true")
    ap.add_argument("--steps",type=int,default=600,help="headless control ticks")
    ap.add_argument("--screenshot",type=Path,help="headless PNG; set MUJOCO_GL=egl")
    a=ap.parse_args()
    if a.steps<1:ap.error("--steps must be positive")
    b=Bundle(a.bundle); session=Session(b,a.stage); speed=a.speed
    speeds=[.1,.25,.5,1.,2.]; actions=queue.SimpleQueue()
    auto_sent=set()
    def automatic():
        if not a.auto:return
        st=session.controller.state
        if st=="PASSIVE" and session.t>.1 and "prep" not in auto_sent:
            session.press(PREP); auto_sent.add("prep")
        if st=="HOLD" and session.t-session.controller.since>.5 and "arm" not in auto_sent:
            session.press(ARM); auto_sent.add("arm")
        if st=="STANDBY_PRE":
            elapsed=session.t-session.controller.since
            if elapsed>1.:session.support=False
            if elapsed>5. and "play" not in auto_sent:
                session.press(PLAY); auto_sent.add("play")
    if a.headless:
        for _ in range(a.steps):
            automatic()
            if not session.tick():break
        if a.screenshot:
            from PIL import Image
            cam=mujoco.MjvCamera(); mujoco.mjv_defaultCamera(cam)
            cam.lookat[:]=session.sim.d.qpos[:3]; cam.distance=3.; cam.azimuth=135.; cam.elevation=-18.
            with mujoco.Renderer(session.sim.m,height=480,width=640) as render:
                render.update_scene(session.sim.d,camera=cam)
                # Append reference geoms without touching simulation physics.
                extra=mujoco.MjvScene(session.sim.m,maxgeom=200)
                session.reference_scene(extra)
                for i in range(extra.ngeom):
                    if render.scene.ngeom>=render.scene.maxgeom:break
                    src=extra.geoms[i]; dst=render.scene.geoms[render.scene.ngeom]
                    mujoco.mjv_initGeom(dst,src.type,src.size,src.pos,src.mat.reshape(9),src.rgba)
                    render.scene.ngeom+=1
                a.screenshot.parent.mkdir(parents=True,exist_ok=True)
                Image.fromarray(render.render()).save(a.screenshot)
        print(json.dumps(dict(state=session.controller.state,frame=session.frame,sim_seconds=session.t,
                              fault=session.fault,reference_bodies=session.sim.m.nbody-1)))
        return
    from mujoco import viewer as mjviewer
    paused=False; ghost=True; follow=True; deadline=time.monotonic()
    with mjviewer.launch_passive(session.sim.m,session.sim.d,key_callback=actions.put,
                                     show_left_ui=False,show_right_ui=False) as viewer:
        viewer.cam.distance=3.; viewer.cam.azimuth=135.; viewer.cam.elevation=-18.
        while viewer.is_running():
            single=False
            while not actions.empty():
                key=actions.get()
                if key==32:paused=not paused
                elif key in (78,110):single=True; paused=True
                elif key==49:session.press(PREP)
                elif key==50:session.press(ARM)
                elif key==51:session.press(PLAY)
                elif key in (66,98):session.support=not session.support
                elif key in (68,100):session.press(STOP);single=True
                elif key in (71,103):ghost=not ghost
                elif key in (70,102):follow=not follow
                elif key in (82,114):
                    with viewer.lock():session.reset()
                    auto_sent.clear(); paused=False
                elif key in (61,334):speed=speeds[min(speeds.index(speed)+1,len(speeds)-1)]
                elif key in (45,333):speed=speeds[max(speeds.index(speed)-1,0)]
            now=time.monotonic()
            if single or (not paused and now>=deadline):
                with viewer.lock():
                    automatic()
                    if not session.tick():
                        paused=True
                        print(f"PAUSED ON FAULT: {session.fault}",flush=True)
                deadline=time.monotonic()+.02/speed
            with viewer.lock():
                session.reference_scene(viewer.user_scn,ghost)
                if follow:viewer.cam.lookat[:]=session.sim.d.qpos[:3]
            viewer.set_texts((None,mujoco.mjtGridPos.mjGRID_TOPLEFT,session.text(paused,speed),""))
            viewer.sync()
            time.sleep(.005)
if __name__=="__main__":main()
