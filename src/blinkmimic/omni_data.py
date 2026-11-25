"""CPU memory-mapped reference shards; all disk I/O stays outside CUDA graphs."""
import json
from pathlib import Path
import numpy as np
import torch
from .omni_features import assemble
from .omni_contract import WRISTS

WIDTH=184
SLICES={"q":(0,29),"qd":(29,58),"root_R":(58,67),"smpl_local":(67,139),
        "smpl_R":(139,148),"points":(148,157),"point_R":(157,184)}

def pack(q,qd,root_R,smpl_local,smpl_R,points,point_R):
    values=[q,qd,root_R,smpl_local,smpl_R,points,point_R]
    return np.concatenate([np.asarray(v,dtype=np.float32).reshape(len(q),-1) for v in values],-1)

def unpack(x):
    shapes={"root_R":(3,3),"smpl_local":(24,3),"smpl_R":(3,3),"points":(3,3),"point_R":(3,3,3)}
    out={k:x[...,a:b] for k,(a,b) in SLICES.items()}
    for k,shape in shapes.items():out[k]=out[k].reshape(*x.shape[:-1],*shape)
    return out

class ReferenceStore:
    def __init__(self,index,joint_names):
        self.path=Path(index)
        payload=json.loads(self.path.read_text())
        from .omni_contract import CONTRACT
        if payload["contract"]!=CONTRACT:raise ValueError("Incompatible omni cache contract")
        self.records=payload["records"]
        self.by_id={r["motion_id"]:i for i,r in enumerate(self.records)}
        if len(self.by_id)!=len(self.records):raise ValueError("Duplicate motion IDs")
        from collections import OrderedDict
        self.shards=OrderedDict()
        self.shard_names=sorted({r["shard"] for r in self.records})
        names={n:i for i,n in enumerate(self.shard_names)}
        self.shard_ids=np.asarray([names[r["shard"]] for r in self.records])
        self.offsets=np.asarray([r["offset"] for r in self.records])
        self.lower=[i for i,n in enumerate(joint_names) if any(s in n for s in ("hip_","knee_","ankle_"))]
        self.wrists=[list(joint_names).index(n) for n in WRISTS]
        if len(self.lower)!=12:raise ValueError("Expected twelve leg joints")
        self.lengths=np.asarray([r["frames"] for r in self.records])

    def gather(self,clip,frame):
        """CPU gather, grouped per shard; frame may have arbitrary leading shape."""
        clip=np.asarray(clip,dtype=np.int64)
        frame=np.asarray(frame,dtype=np.int64)
        if clip.shape!=frame.shape:raise ValueError("clip/frame shape mismatch")
        if np.any(clip<0) or np.any(clip>=len(self.records)):raise IndexError("clip")
        if np.any(frame<0) or np.any(frame>=self.lengths[clip]):raise IndexError("frame")
        flatc,flatf=clip.ravel(),frame.ravel()
        out=np.empty((len(flatc),WIDTH),np.float32)
        ids=self.shard_ids[flatc]
        rows=self.offsets[flatc]+flatf
        # Sort once rather than scan the entire batch once per shard.
        order=np.argsort(ids,kind="stable")
        boundaries=np.r_[0,np.flatnonzero(np.diff(ids[order]))+1,len(order)]
        for lo,hi in zip(boundaries[:-1],boundaries[1:]):
            positions=order[lo:hi]
            name=self.shard_names[ids[positions[0]]]
            if name not in self.shards:
                self.shards[name]=np.load(self.path.parent/name,mmap_mode="r",allow_pickle=False)
            self.shards.move_to_end(name)
            out[positions]=self.shards[name][rows[positions]]
            while len(self.shards)>32:self.shards.popitem(last=False)
        return out.reshape(*clip.shape,WIDTH)

    def prefetch(self,clip,frame,device):
        """Fetch both temporal windows once outside captured local computation."""
        clip=np.asarray(clip,dtype=np.int64);frame=np.asarray(frame,dtype=np.int64)
        end=self.lengths[clip]-1
        fm=np.minimum(frame[...,None]+np.arange(10)*5,end[...,None])
        fs=np.minimum(frame[...,None]+np.arange(10),end[...,None])
        c=np.broadcast_to(clip[...,None],fm.shape)
        return {k:torch.from_numpy(v).to(device) for k,v in
                (("motion",self.gather(c,fm)),("smpl",self.gather(c,fs)))}

    def inputs(self,reference,robot_R):
        return reference_inputs(reference,robot_R,self.lower,self.wrists)

def reference_inputs(reference,robot_R,lower,wrists):
    m,s=unpack(reference["motion"]),unpack(reference["smpl"])
    return assemble({"motion":m["q"],"smpl":s["q"]},m["qd"],m["root_R"],
                    s["smpl_local"],s["smpl_R"],m["points"][...,0,:,:],
                    m["point_R"][...,0,:,:,:],robot_R,lower,wrists)
