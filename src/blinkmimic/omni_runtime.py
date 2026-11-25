"""Bridge paired reference prefetch into BlinkMimic's staged world-model trainer."""
import json
from pathlib import Path
import numpy as np
import torch
from .motion import MotionLib
from .omni import MODALITIES
from .mdp import unflatten_state

def load_motion(store,root,robot,kin,device):
    paths=[str(Path(root)/r["g1_path"]) for r in store.records]
    return MotionLib(paths,kin,device,robot.load_motion,compact=True,
                     frame_limits=[r["frames"] for r in store.records])

def copy_tree(old,new):
    if isinstance(new,dict):
        if old is None:return {k:copy_tree(None,v) for k,v in new.items()}
        for k,v in new.items():copy_tree(old[k],v)
    elif isinstance(new,(tuple,list)):
        if old is None:return tuple(copy_tree(None,v) for v in new)
        for a,b in zip(old,new):copy_tree(a,b)
    else:
        if old is None:return new.clone()
        old.copy_(new)
    return old

class OmniRuntime:
    def __init__(self,g,store):
        self.g,self.store=g,store
        store.lower=torch.tensor(store.lower,device=g["dev"])
        store.wrists=torch.tensor(store.wrists,device=g["dev"])
        self.collect_ref=None;self.imag_ref=None;self.starts=None
        self.modality=torch.randint(0,3,(g["args"].num_envs,),device=g["dev"])
        b=g["buffer"]
        b.fields["modality"]=((),torch.long)
        b.data["modality"]=torch.zeros((b.T,b.N),dtype=torch.long,device=g["dev"])
        self.aux=torch.zeros(5,device=g["dev"])
        self.stats=torch.zeros(3,4,device=g["dev"]) # transitions,episodes,failed,normalized return sum

    def prefetch(self,f):
        motion=self.g["motion"]
        clip=motion.clip_id[f]
        local=f-motion.clip_start[clip]
        return self.store.prefetch(clip.cpu().numpy(),local.cpu().numpy(),self.g["dev"])

    def prepare_collection(self):
        self.collect_ref=copy_tree(self.collect_ref,self.prefetch(self.g["env"].f))

    def collection_inputs(self):
        return self.store.inputs(self.collect_ref,self.g["env"].state()[1])

    def collect_action(self,history):
        return self.g["pi"](history[...,:93],self.collection_inputs(),self.modality)

    def record(self,out):
        from .mdp import MAX_STEP_REWARD
        mode=torch.nn.functional.one_hot(self.modality,3).float()
        done=out["done"].float()
        env=self.g["env"]
        ret=env.ep_ret/(env.ep_step.float().clamp_min(1)*MAX_STEP_REWARD)
        values=torch.stack((torch.ones_like(done),done,out["term"].float()*done,ret*done),-1)
        self.stats.add_(mode.T@values)

    def reset_modalities(self,done):
        self.modality.copy_(torch.where(done,torch.randint_like(self.modality,0,3),self.modality))

    def prepare_imagination(self):
        g=self.g
        # Host-side prefetch boundary: never rely on the legacy 2B approximation
        # for short paired clips. Select valid starts, with replacement if needed.
        B=g["args"].pi_batch
        batches=[]; histories=[]; count=0
        for _ in range(16):
            start,hist=g["buffer"].sample_starts(2*B,g["Hh"])
            f=start["f"]
            valid=(f+g["args"].horizon<=g["motion"].clip_end[g["motion"].clip_id[f]]).nonzero().flatten()
            if len(valid):
                batches.append({k:v[valid] for k,v in start.items()})
                histories.append({k:v[valid] for k,v in hist.items()})
                count+=len(valid)
            if count>=B:break
        if not count:raise RuntimeError("No replay start fits the imagination horizon within a paired clip")
        ids=torch.randperm(count,device=g["dev"])[:B] if count>=B else torch.randint(count,(B,),device=g["dev"])
        new=tuple({k:torch.cat([b[k] for b in group])[ids] for k in group[0]}
                  for group in (batches,histories))
        self.starts=copy_tree(self.starts,new)
        f=self.starts[0]["f"]
        frames=f[None]+torch.arange(g["args"].horizon,device=f.device)[:,None]
        self.imag_ref=copy_tree(self.imag_ref,self.prefetch(frames))

    def imagination_inputs(self,t,robot_R):
        return self.store.inputs({k:v[t] for k,v in self.imag_ref.items()},robot_R)

    def imag_step(self,S,history,previous,f,xi,xi_n,t,mode):
        g=self.g
        action=g["pi"](history[...,:93],self.imagination_inputs(t,S[1]),mode)
        next_state=g["wm"].step(S,action,previous,xi_n)
        f1=f+1
        obs=g["mdp"].add_noise(g["mdp"].obs(next_state,action,xi,f1))
        history=torch.cat((history[:,1:],obs[:,None]),1)
        return next_state,action,history,f1,g["motion"].goal(f1,g["K"])

    @torch.no_grad()
    def log(self,ctx,tb,root,it):
        from .latent import bound,quantize_bounded,HALF
        inputs=self.collection_inputs()
        row=dict(iteration=it,modalities={})
        stats=self.stats.clone();ctx.sum(stats);self.stats.zero_()
        aux=self.aux.clone();ctx.sum(aux);aux/=ctx.world
        row["auxiliary"]=dict(zip(("g1_recon","g1_smpl_latent","g1_teleop_latent",
                                  "teleop_smpl_latent","reencoded_smpl_g1_latent"),aux.cpu().tolist()))
        pi=self.g["pi"]
        for i,m in enumerate(MODALITIES):
            u=bound(pi.enc_raw(m,inputs[m][:256]))
            z=quantize_bounded(u)
            ids=(z*HALF+HALF).long().clamp(0,31)
            counts=torch.stack([(ids==j).sum(0) for j in range(32)],-1).float();ctx.sum(counts)
            err=torch.stack(((z-u/HALF).square().sum(),u.new_tensor(u.numel())));ctx.sum(err)
            grads=[p.grad.square().sum() for p in pi.encoders[m].parameters() if p.grad is not None]
            row["modalities"][m]=dict(transitions=float(stats[i,0]),episodes=float(stats[i,1]),
                failed=float(stats[i,2]),return_mean=float(stats[i,3]/stats[i,1].clamp_min(1)),
                quantization_mse=float(err[0]/err[1]),used_levels=counts.gt(0).sum(-1).cpu().tolist(),
                saturation=((counts[:,0]+counts[:,-1])/counts.sum(-1).clamp_min(1)).cpu().tolist(),
                encoder_grad_norm=float(torch.stack(grads).sum().sqrt()) if grads else 0.)
        if ctx.main:
            with (root/"omni_metrics.jsonl").open("a") as f:f.write(json.dumps(row)+"\n")
            if tb:
                for m,values in row["modalities"].items():
                    for k,v in values.items():
                        if isinstance(v,(int,float)):tb.add_scalar(f"omni/{m}/{k}",v,it)
                for k,v in row["auxiliary"].items():tb.add_scalar("omni/aux/"+k,v,it)
