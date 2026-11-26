"""SONIC V2 planner ONNX adapter. Reference: localmotion_kplanner*.hpp.
Runs in the VR source process, never in the 50 Hz motor controller.
"""
import time
import numpy as np
from scipy.spatial.transform import Rotation,Slerp

def sample_qpos(seq,frames):
    frames=np.clip(np.asarray(frames),0,len(seq)-1)
    lo=frames.astype(int);hi=np.minimum(lo+1,len(seq)-1);w=(frames-lo)[:,None]
    out=seq[lo]*(1-w)+seq[hi]*w
    qs=seq[:,3:7].copy()
    qs/=np.linalg.norm(qs,axis=1,keepdims=True)
    out[:,3:7]=Slerp(np.arange(len(seq)),Rotation.from_quat(qs[:,[1,2,3,0]]))(frames).as_quat()[:,[3,0,1,2]]
    return out.astype(np.float32)

class Planner:
    def __init__(self,path):
        import onnxruntime as ort
        so=ort.SessionOptions();so.intra_op_num_threads=2;so.inter_op_num_threads=1
        so.add_session_config_entry("session.intra_op.allow_spinning","0")
        self.session=ort.InferenceSession(str(path),so,providers=["CPUExecutionProvider"])
        self.inputs={i.name:np.zeros(i.shape,np.int64 if i.type=="tensor(int64)" else np.float32) for i in self.session.get_inputs()}
        expected={"context_mujoco_qpos","target_vel","mode","movement_direction","facing_direction","random_seed",
                  "has_specific_target","specific_target_positions","specific_target_headings","allowed_pred_num_tokens","height"}
        if set(self.inputs)!=expected:raise ValueError("expected SONIC V2 11-input planner")
        self.inputs["allowed_pred_num_tokens"][:,:6]=1
        self.inputs["random_seed"][:]=1234

    def generate(self,context,command):
        x=self.inputs
        x["context_mujoco_qpos"][0]=context
        for field,default in (("mode",[0]),("target_vel",[-1.]),("height",[-1.]),
                              ("movement_direction",[0.,0.,0.]),("facing_direction",[1.,0.,0.])):
            source={"target_vel":"speed","movement_direction":"movement","facing_direction":"facing"}.get(field,field)
            x[field][...]=np.asarray(command.get(source,default)).reshape(x[field].shape)
        t=time.monotonic()
        q,n=self.session.run(None,x);n=int(n[0])
        if not 4<=n<=64 or not np.isfinite(q[:,:n]).all():raise ValueError("invalid SONIC planner output")
        raw=q[0,:n]
        seq=sample_qpos(raw,np.arange(int(np.floor(n/30*50)))*30/50)
        return seq,(time.monotonic()-t)*1000

def initial_context(q):
    x=np.zeros((4,36),np.float32);x[:,2]=.788740;x[:,3]=1;x[:,7:]=q
    return x

def planner_window(seq,start,now):
    frame=max(0.,(now-start)*50)
    x=sample_qpos(seq,frame+np.arange(46))
    nxt=sample_qpos(seq,frame+np.arange(46)+1)
    return x,(nxt[:,7:]-x[:,7:])*50

def blend_plan(old, old_start, new, new_start, frames=8):
    """Cross-fade overlapping references on the same 50Hz timeline.
    First sample remains the old trajectory; quaternion interpolation is shortest-path.
    """
    old_aligned=sample_qpos(old,(new_start-old_start)*50+np.arange(len(new)))
    out=np.asarray(new,dtype=np.float32).copy()
    n=min(frames+1,len(out))
    w=np.minimum(np.arange(n)/max(frames,1),1.)
    out[:n]=old_aligned[:n]*(1-w[:,None])+out[:n]*w[:,None]
    for i in range(n):
        quats=np.stack((old_aligned[i,3:7],new[i,3:7]))
        rot=Rotation.from_quat(quats[:,[1,2,3,0]])
        out[i,3:7]=Slerp([0,1],rot)([w[i]]).as_quat()[0,[3,0,1,2]]
    return out
