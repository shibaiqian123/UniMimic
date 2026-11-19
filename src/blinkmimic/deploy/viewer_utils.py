"""Small display-only helpers. No simulation integration or GPU synchronization."""
import copy
import math
import mujoco

class ControlClock:
    def __init__(self,period,now):
        self.set_period(period,now)
    def set_period(self,period,now):
        if not math.isfinite(period) or period<=0:raise ValueError("positive finite period required")
        self.period=period;self.deadline=now
    def due(self,now):
        return now>=self.deadline
    def advance(self,finished):
        # Compute time consumes the period; overruns do not accumulate catch-up debt.
        self.deadline=max(self.deadline+self.period,finished)

class RenderMirror:
    def __init__(self,model):
        self.m=copy.copy(model)
        self.d=mujoco.MjData(self.m)
    def update(self,source):
        self.d.qpos[:]=source.qpos
        self.d.qvel[:]=source.qvel
        self.d.time=source.time
        mujoco.mj_kinematics(self.m,self.d)
        mujoco.mj_comPos(self.m,self.d)
        mujoco.mj_camlight(self.m,self.d)
