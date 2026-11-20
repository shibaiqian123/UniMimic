"""Official G1 torque-motor physics and 1 kHz PD on MuJoCo Warp. No mjlab."""
from pathlib import Path
import numpy as np
import warnings
import time
import warp as wp
import mujoco_warp as mjw
from .sim2sim import Plant
from .core import digest

@wp.kernel
def pd_step(qpos:wp.array2d(dtype=float),qvel:wp.array2d(dtype=float),
            ctrl:wp.array2d(dtype=float),qidx:wp.array(dtype=int),vidx:wp.array(dtype=int),
            acts:wp.array(dtype=int),targets:wp.array2d(dtype=float),kp:wp.array2d(dtype=float),
            kd:wp.array2d(dtype=float),effort:wp.array(dtype=float),velocity:wp.array(dtype=float),
            limited:wp.array(dtype=int),lo:wp.array(dtype=float),hi:wp.array(dtype=float),
            counts:wp.array(dtype=int),peaks:wp.array(dtype=float),substep:int):
    j=wp.tid()
    dq=qvel[0,vidx[j]]
    desired=kp[substep,j]*(targets[substep,j]-qpos[0,qidx[j]])-kd[substep,j]*dq
    wp.atomic_max(peaks,0,wp.abs(desired)/effort[j])
    wp.atomic_max(peaks,1,wp.abs(dq)/velocity[j])
    if wp.abs(desired)>effort[j]:wp.atomic_add(counts,0,1)
    tau=wp.clamp(desired,-effort[j],effort[j])
    if limited[j]!=0:
        if tau<lo[j] or tau>hi[j]:wp.atomic_add(counts,1,1)
        tau=wp.clamp(tau,lo[j],hi[j])
    ctrl[0,acts[j]]=tau

@wp.kernel
def support_force(qpos:wp.array2d(dtype=float),qvel:wp.array2d(dtype=float),
                  force:wp.array2d(dtype=wp.spatial_vector),enabled:wp.array(dtype=int),
                  anchor:wp.array(dtype=wp.vec3),reference:wp.array(dtype=wp.quat),
                  fq:int,fv:int,pelvis:int,weight:float):
    value=wp.spatial_vector(0.,0.,0.,0.,0.,0.)
    if enabled[0]!=0:
        q=wp.normalize(wp.quat(qpos[0,fq+4],qpos[0,fq+5],qpos[0,fq+6],qpos[0,fq+3]))
        pos=wp.vec3(qpos[0,fq],qpos[0,fq+1],qpos[0,fq+2])
        v=wp.vec3(qvel[0,fv],qvel[0,fv+1],qvel[0,fv+2])
        gyro=wp.vec3(qvel[0,fv+3],qvel[0,fv+4],qvel[0,fv+5])
        errq=reference[0]*wp.quat_inverse(q)
        err=2.*errq[3]*wp.vec3(errq[0],errq[1],errq[2])
        f=2000.*(anchor[0]-pos)-100.*v+wp.vec3(0.,0.,weight)
        t=500.*err-50.*wp.quat_rotate(q,gyro)
        value=wp.spatial_vector(f[0],f[1],f[2],t[0],t[1],t[2])
    force[0,pelvis]=value

@wp.kernel
def pack_state(qpos:wp.array2d(dtype=float),qvel:wp.array2d(dtype=float),clock:wp.array(dtype=float),
               counts:wp.array(dtype=int),peaks:wp.array(dtype=float),packed:wp.array(dtype=wp.float64),
               nq:int,nv:int):
    i=wp.tid()
    if i<nq:packed[i]=wp.float64(qpos[0,i])
    elif i<nq+nv:packed[i]=wp.float64(qvel[0,i-nq])
    elif i==nq+nv:packed[i]=wp.float64(clock[0])
    elif i<nq+nv+3:packed[i]=wp.float64(counts[i-nq-nv-1])
    else:packed[i]=wp.float64(peaks[i-nq-nv-3])

class WarpPlant(Plant):
    def __init__(self,b,xml,kind="g2",latency_ms=0.,kp_scale=1.,mass=0.,friction=None,device="cuda:0"):
        if kind!="g2":raise ValueError("official GPU plant is g2 only")
        super().__init__(b,xml,kind,latency_ms,kp_scale,mass,friction)
        self.gpu_device=wp.get_device(device)
        if not self.gpu_device.is_cuda:raise ValueError("CUDA device required")
        with wp.ScopedDevice(self.gpu_device):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                self.gpu_model=mjw.put_model(self.m)
            self.backend_warnings=[str(w.message) for w in caught]
            for message in self.backend_warnings:warnings.warn(message)
            arr=lambda x,dtype=float:wp.array(np.asarray(x),dtype=dtype,device=self.gpu_device)
            self.gqidx=arr(self.qidx,wp.int32);self.gvidx=arr(self.vidx,wp.int32);self.gact=arr(self.act,wp.int32)
            self.geffort=arr(b.effort);self.gvelocity=arr(b.velocity)
            self.glimited=arr(self.m.actuator_ctrllimited[self.act],wp.int32)
            self.glo=arr(self.m.actuator_ctrlrange[self.act,0]);self.ghi=arr(self.m.actuator_ctrlrange[self.act,1])
            self.targets=wp.zeros((20,29),dtype=float);self.kps=wp.zeros((20,29),dtype=float);self.kds=wp.zeros((20,29),dtype=float)
            self.counts=wp.zeros(2,dtype=int);self.peaks=wp.zeros(2,dtype=float)
            self.enabled=wp.zeros(1,dtype=int)
            # Persistent pinned host buffers: no allocation/free per 50 Hz update.
            self.host_targets=wp.zeros((20,29),dtype=float,device="cpu",pinned=True)
            self.host_kps=wp.zeros((20,29),dtype=float,device="cpu",pinned=True)
            self.host_kds=wp.zeros((20,29),dtype=float,device="cpu",pinned=True)
            self.host_enabled=wp.zeros(1,dtype=int,device="cpu",pinned=True)
            self.state_size=self.m.nq+self.m.nv+5
            self.packed=wp.zeros(self.state_size,dtype=wp.float64)
            self.host_packed=wp.zeros(self.state_size,dtype=wp.float64,device="cpu",pinned=True)
            self.last_step_ms=0.
            self.target_schedule=np.empty((20,29),dtype=np.float32)
            self.kp_schedule=np.empty((20,29),dtype=np.float32)
            self.kd_schedule=np.empty((20,29),dtype=np.float32)

            self.anchor=arr([b.motion["root_p"][0]+[0,0,self.ground_shift]],wp.vec3)
            import mujoco
            quat=np.empty(4);mujoco.mju_mat2Quat(quat,b.motion["root_R"][0].reshape(9))
            self.reference=arr([[quat[1],quat[2],quat[3],quat[0]]],wp.quat)
            self.gpu_data=mjw.put_data(self.m,self.d,nworld=1,nconmax=256,njmax=1024)
            self._upload(np.repeat(self.target[None],20,0),np.repeat(b.kp[None],20,0),np.repeat(b.kd[None],20,0),False)
            # Compile all kernels before capture; warmup does not alter the CPU initial state.
            self._simulate();wp.synchronize_device(self.gpu_device)
            self.sync_reset()
    def _simulate(self):
        for i in range(20):
            wp.launch(support_force,dim=1,inputs=[self.gpu_data.qpos,self.gpu_data.qvel,self.gpu_data.xfrc_applied,
                       self.enabled,self.anchor,self.reference,self.fq,self.fv,self.pelvis,float(self.m.body_mass.sum()*9.81)])
            wp.launch(pd_step,dim=29,inputs=[self.gpu_data.qpos,self.gpu_data.qvel,self.gpu_data.ctrl,
                self.gqidx,self.gvidx,self.gact,self.targets,self.kps,self.kds,self.geffort,self.gvelocity,
                self.glimited,self.glo,self.ghi,self.counts,self.peaks,i])
            mjw.step(self.gpu_model,self.gpu_data)
    def _upload(self,targets,kp,kd,support):
        for dst,host,src in ((self.targets,self.host_targets,targets),
                             (self.kps,self.host_kps,kp),(self.kds,self.host_kds,kd)):
            np.copyto(host.numpy(),src,casting="unsafe")
            wp.copy(dst,host)
        self.host_enabled.numpy()[0]=int(support)
        wp.copy(self.enabled,self.host_enabled)

    def pull_state(self):
        """One compact D2H copy; contact/constraint arrays stay on GPU."""
        wp.launch(pack_state,dim=self.state_size,inputs=[self.gpu_data.qpos,self.gpu_data.qvel,
            self.gpu_data.time,self.counts,self.peaks,self.packed,self.m.nq,self.m.nv])
        wp.copy(self.host_packed,self.packed)
        wp.synchronize_stream(wp.get_stream(self.gpu_device))
        state=self.host_packed.numpy()
        nq,nv=self.m.nq,self.m.nv
        self.d.qpos[:]=state[:nq]
        self.d.qvel[:]=state[nq:nq+nv]
        self.d.time=float(state[nq+nv])
        self.torque_saturated=int(state[nq+nv+1]);self.model_saturated=int(state[nq+nv+2])
        self.max_torque_ratio=float(state[nq+nv+3]);self.max_velocity_ratio=float(state[nq+nv+4])
        # Geometry/contact fields in self.d are intentionally stale.
        # Viewers use a separate render MjData with CPU kinematics only.

    def sync_reset(self):
        with wp.ScopedDevice(self.gpu_device):
            self.gpu_data=mjw.put_data(self.m,self.d,nworld=1,nconmax=256,njmax=1024)
            self.counts.zero_();self.peaks.zero_()
            try:
                with wp.ScopedCapture(device=self.gpu_device) as cap:self._simulate()
                self.graph=cap.graph;self.capture_error=None
            except Exception as exc:
                self.graph=None;self.capture_error=str(exc)
                self.gpu_data=mjw.put_data(self.m,self.d,nworld=1,nconmax=256,njmax=1024)
    def step(self,support=False):
        started=time.perf_counter()
        for i in range(20):
            while self.pending and self.pending[0][0]<=self.steps+i:
                _,self.target,self.kp,self.kd=self.pending.popleft()
            self.target_schedule[i]=self.target;self.kp_schedule[i]=self.kp;self.kd_schedule[i]=self.kd
        with wp.ScopedDevice(self.gpu_device):
            self._upload(self.target_schedule,self.kp_schedule,self.kd_schedule,support)
            if self.graph is None:self._simulate()
            else:wp.capture_launch(self.graph)
            self.pull_state()
        self.steps+=20
        self.last_step_ms=(time.perf_counter()-started)*1000

class TorchCudaPolicy:
    """FP32 GPU inference equivalent to the existing exported ONNX contract."""
    def __init__(self,b,ckpt,device):
        import torch
        from blinkmimic.networks import Policy,policy_kwargs
        from blinkmimic.robot import get_robot
        from blinkmimic import networks
        if digest(ckpt)!=b.cfg["checkpoint_sha256"]:raise ValueError("checkpoint/bundle mismatch")
        self.torch=torch;self.device=torch.device(device)
        if self.device.type!="cuda" or not torch.cuda.is_available():raise ValueError("CUDA policy required")
        torch.cuda.set_device(self.device);torch.backends.cuda.matmul.allow_tf32=False
        networks.AMP_DTYPE=None
        ck=torch.load(ckpt,map_location="cpu",weights_only=False);c=ck["args"]
        if c["hist"]!=10 or c["goal_frames"]!=10:raise ValueError("unsupported checkpoint horizons")
        widths=lambda k:tuple(map(int,c.get(k,"2048,1024,512,256").split(",")))
        self.net=Policy(99,730,29,hist=10,enc_hidden=widths("enc_hidden"),dec_hidden=widths("dec_hidden"),
                        **policy_kwargs(c,get_robot("g1"))).to(self.device).eval()
        self.net.load_state_dict(ck["pi"],strict=True)
    def __call__(self,h,g):
        with self.torch.no_grad():
            return self.net(self.torch.as_tensor(h[None],dtype=self.torch.float32,device=self.device),
                            self.torch.as_tensor(g[None],dtype=self.torch.float32,device=self.device))[0].cpu().numpy()
