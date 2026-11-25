"""NumPy-only paired-reference feature adapter for the shared LowState runner."""
import numpy as np

def features(bundle,frame,robot_R,heading,phase,modality):
    a=bundle.motion["omni_reference"]
    end=len(a)-1
    fm=np.minimum(frame+np.arange(10)*5,end)
    fs=np.minimum(frame+np.arange(10),end)
    if phase!="motion":fm[:]=frame;fs[:]=frame
    m,s=a[fm],a[fs]
    q=m[:,:29];qd=m[:,29:58].copy()
    if phase!="motion":qd[:]=0
    root=m[:,58:67].reshape(10,3,3)
    ori=(robot_R.T@heading@root)[...,:2].reshape(10,6)
    if modality=="motion":
        qv=np.concatenate((q.reshape(-1),qd.reshape(-1))).reshape(10,58)
        return np.concatenate((qv,ori),-1).reshape(-1).astype(np.float32)
    if modality=="smpl":
        local=s[:,67:139].reshape(10,72)
        sr=s[:,139:148].reshape(10,3,3)
        rel=(robot_R.T@heading@sr)[...,:2].reshape(10,6)
        return np.concatenate((local,rel,s[:,:29][:,bundle.cfg["omni_wrists"]]),-1).reshape(-1).astype(np.float32)
    if modality=="teleop":
        points=m[0,148:157]
        point_R=m[0,157:184].reshape(3,3,3)
        lower=bundle.cfg["omni_lower"]
        return np.concatenate((q[:,lower].reshape(-1),qd[:,lower].reshape(-1),
                               points,point_R[...,:2].reshape(-1),ori[0])).astype(np.float32)
    raise ValueError("unknown omni modality")
