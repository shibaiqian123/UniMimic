"""CPU MuJoCo test plant; never opens DDS or hardware. Training dependencies allowed here."""
import mujoco
import numpy as np
from .core import rotation
class CpuSim:
    def __init__(self,bundle):
        from blinkmimic.g1.g1_constants import get_spec,motor_of
        self.b=bundle
        spec=get_spec()
        spec.option.timestep=.002
        spec.option.integrator=mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        spec.worldbody.add_geom(name="deploy_floor",type=mujoco.mjtGeom.mjGEOM_PLANE,size=[0,0,.1],
                                contype=1,conaffinity=1,friction=[1.,.005,.0001])
        for i,n in enumerate(bundle.cfg["joint_names"]):
            spec.joint(n).armature=motor_of(n)[2]
            act=spec.add_actuator(name=n+"_pd",target=n,trntype=mujoco.mjtTrn.mjTRN_JOINT,
                                 forcelimited=True,forcerange=[-bundle.effort[i],bundle.effort[i]])
            act.set_to_position(bundle.kp[i],bundle.kd[i])
        self.spec=spec; self.m=spec.compile(); self.d=mujoco.MjData(self.m)
        self.jids=np.array([self.m.joint(n).id for n in bundle.cfg["joint_names"]])
        self.qidx=self.m.jnt_qposadr[self.jids]; self.vidx=self.m.jnt_dofadr[self.jids]
        self.reset()
    def reset(self):
        mujoco.mj_resetData(self.m,self.d)
        self.d.qpos[:3]=self.b.motion["root_p"][0]
        mujoco.mju_mat2Quat(self.d.qpos[3:7],self.b.motion["root_R"][0].reshape(9))
        self.d.qpos[self.qidx]=self.b.motion["q"][0]
        self.d.ctrl[:]=self.d.qpos[self.qidx]
        mujoco.mj_forward(self.m,self.d)
    def state(self):
        return (self.d.qpos[self.qidx].copy(),self.d.qvel[self.vidx].copy(),
                self.d.qvel[3:6].copy(),rotation(self.d.qpos[3:7]))
    def step(self,target,kp=None,kd=None,before_substep=None):
        if kp is not None:
            self.m.actuator_gainprm[:,0]=kp
            self.m.actuator_biasprm[:,1]=-kp
            self.m.actuator_biasprm[:,2]=-kd
        self.d.ctrl[:]=target
        for _ in range(10):
            if before_substep is not None: before_substep()
            mujoco.mj_step(self.m,self.d)
