"""Explicit data parallelism: fixed global batches, communication outside CUDA graphs.
Inspired by SuperMimic-Repro-Sim2Real's explicit gradient reduction. Implemented
independently for BlinkMimic's global-batch semantics; no LR scaling.
"""
import os
import time
from datetime import timedelta
import torch
import torch.distributed as dist

class Distributed:
    def __init__(self, device="cuda:0", test=False):
        self.device=torch.device(device)
        self.world=dist.get_world_size() if dist.is_initialized() else 1
        self.rank=dist.get_rank() if dist.is_initialized() else 0
        self.main=self.rank==0
        self.comm_events=[]
        self.comm_host_s=0.
        self.buffers={}
        self.test=test

    def _collective(self,fn,device):
        t=time.monotonic()
        gpu=torch.device(device).type=="cuda"
        if gpu:
            begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            begin.record()
        fn()
        if gpu:
            end.record();self.comm_events.append((begin,end))
        self.comm_host_s+=time.monotonic()-t

    def sum(self,x):
        if self.world>1:self._collective(lambda:dist.all_reduce(x,op=dist.ReduceOp.SUM),x.device)
        return x

    def broadcast(self,x):
        if self.world>1:self._collective(lambda:dist.broadcast(x,src=0),x.device)
        return x

    def barrier(self):
        if self.world>1:self._collective(dist.barrier,self.device)

    def broadcast_models(self,*models):
        for m in models:
            for x in list(m.parameters())+list(m.buffers()):
                self.broadcast(x.data)

    def gather_objects(self,value):
        if self.world==1:return [value]
        out=[None]*self.world
        dist.all_gather_object(out,value)
        return out

    def communication_seconds(self):
        if self.device.type=="cuda":
            torch.cuda.synchronize()
            seconds=sum(a.elapsed_time(b)/1000 for a,b in self.comm_events)
        else:seconds=self.comm_host_s
        self.comm_events.clear();self.comm_host_s=0.
        return seconds

    @torch.no_grad()
    def moments(self,norms,xs):
        # Center around the shared running mean to reduce sum-of-squares cancellation.
        parts=[]
        for norm,x in zip(norms,xs):
            x=x.reshape(-1,x.shape[-1]).float()
            d=x-norm.mean
            parts.extend((d.sum(0),d.square().sum(0),x.new_tensor([x.shape[0]])))
        key=("moments",tuple(n.mean.numel() for n in norms))
        local=torch.cat(parts)
        if key not in self.buffers:self.buffers[key]=torch.empty_like(local)
        packed=self.buffers[key];packed.copy_(local);self.sum(packed)
        offset=0
        for norm in norms:
            k=norm.mean.numel()
            ds,ss,n=packed[offset:offset+k],packed[offset+k:offset+2*k],packed[offset+2*k]
            offset+=2*k+1
            delta=ds/n.clamp_min(1)
            v=(ss/n.clamp_min(1)-delta.square()).clamp_min(0)
            total=norm.count+n
            mean=norm.mean+delta*n/total.clamp_min(1)
            var=(norm.var*norm.count+v*n+delta.square()*norm.count*n/total.clamp_min(1))/total.clamp_min(1)
            live=(norm.count<norm.max_count)&(n>0)
            norm.mean.copy_(torch.where(live,mean,norm.mean))
            norm.var.copy_(torch.where(live,var,norm.var))
            norm.count.copy_(torch.where(live,total,norm.count))

    @torch.no_grad()
    def gradients(self,model):
        # Stable parameter order / buffer; missing local gradients contribute zeros.
        params=list(model.parameters())
        key=("grad",id(model))
        if key not in self.buffers:
            self.buffers[key]=torch.empty(sum(p.numel() for p in params),device=params[0].device,dtype=torch.float32)
        flat=self.buffers[key];offset=0
        for p in params:
            slot=flat[offset:offset+p.numel()]
            if p.grad is None:slot.zero_()
            else:slot.copy_(p.grad.reshape(-1))
            offset+=p.numel()
        self.sum(flat);flat.div_(self.world)
        finite=flat.isfinite().all()
        flat.copy_(torch.where(finite,flat,0.))
        offset=0
        for p in params:
            if p.grad is None:p.grad=torch.zeros_like(p)
            p.grad.copy_(flat[offset:offset+p.numel()].view_as(p))
            offset+=p.numel()
        return finite

def init():
    world=int(os.environ.get("WORLD_SIZE","1"))
    if world!=2:raise ValueError("distributed production mode supports exactly two GPUs")
    if torch.cuda.device_count()!=1:
        raise ValueError("each rank must be pinned to exactly one visible GPU before torch import")
    torch.cuda.set_device(0)
    dist.init_process_group("nccl",timeout=timedelta(minutes=10))
    return Distributed()

def split_global(args,world):
    global_args=dict(vars(args))
    for key in ("num_envs","wm_batch","pi_batch"):
        n=getattr(args,key)
        if n<=0 or n%world:raise ValueError(f"--{key}={n} must be positive and divisible by world_size={world}")
        setattr(args,key,n//world)
    if args.pi_batch*args.horizon%args.critic_minibatches:
        raise ValueError("local imagined batch must divide evenly into critic minibatches")
    global_args["distributed"]=dict(world_size=world,batch_semantics="global",lr_scale=1.,
        per_rank={k:getattr(args,k) for k in ("num_envs","wm_batch","pi_batch")})
    return global_args

@torch.no_grad()
def update_bins(env,totals):
    """One global EMA update from newly completed episodes, never sum old EMA buffers."""
    count,g_sum,f_count,f_sum,ended=totals
    env.bin_G.copy_(torch.where(count>0,.8*env.bin_G+.2*g_sum/count.clamp_min(1),env.bin_G))
    if env.fail_reset>0:
        faded=torch.where(ended>0,env.bin_fail*.999,env.bin_fail)
        env.bin_fail.copy_(torch.where(f_count>0,.9*faded+.1*f_sum/f_count.clamp_min(1),faded))
    soft=torch.softmax(-25.*env.bin_G/env.G_max,0)
    env.bin_w_avg.copy_(torch.where(ended>0,.999*env.bin_w_avg+.001*soft,env.bin_w_avg))

def zero_grads(model):
    for p in model.parameters():
        if p.grad is None:p.grad=torch.zeros_like(p)
        else:p.grad.zero_()

def optimizer_step(ctx,model,opt,clip_value=0.):
    finite=ctx.gradients(model)
    gn=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
    if clip_value>0:torch.nn.utils.clip_grad_value_(model.parameters(),clip_value)
    if not isinstance(opt,torch.optim.Optimizer):
        opt.step(ok=finite)
    elif opt.defaults.get("fused"):
        opt.found_inf=(~finite).float()
        try:opt.step()
        finally:opt.found_inf=None
    elif bool(finite):
        opt.step()
    return gn,finite

class LocalGraph:
    """No-argument local phase. No collectives or optimizer updates inside captures.
    Inputs live in stable buffers owned by the caller. First calls warm the phase.
    """
    def __init__(self,fn,ctx,enabled=True,sim=None):
        self.fn,self.ctx,self.enabled,self.sim=fn,ctx,enabled,sim
        self.calls=0;self.graph=None;self.output=None;self.stream=None
    def __call__(self):
        if not self.enabled:return self.fn()
        self.calls+=1
        if self.calls<=3:return self.fn()
        if self.graph is None:
            self.ctx.barrier()
            torch.cuda.synchronize()
            self.graph=torch.cuda.CUDAGraph();self.stream=torch.cuda.Stream()
            if self.sim is not None:
                import warp as wp
                from mjlab.sim.sim_data import TorchArray
                old,enabled=TorchArray.__setitem__,self.sim.use_cuda_graph
                TorchArray.__setitem__=lambda me,idx,value:me._tensor.__setitem__(idx,value)
                self.sim.use_cuda_graph=False
            try:
                with torch.cuda.graph(self.graph,stream=self.stream,capture_error_mode="thread_local"):
                    if self.sim is None:self.output=self.fn()
                    else:
                        ws=wp.stream_from_torch(torch.cuda.current_stream())
                        with wp.ScopedStream(ws,sync_enter=False,sync_exit=False):
                            wp.capture_begin(stream=ws,external=True)
                            try:self.output=self.fn()
                            finally:self.wp_graph=wp.capture_end(stream=ws)
            finally:
                if self.sim is not None:
                    TorchArray.__setitem__,self.sim.use_cuda_graph=old,enabled
            self.ctx.barrier()
        self.graph.replay()
        return self.output

def state_digest(models,optimizers=()):
    """Optional validation only: exact parameter/buffer/optimizer state digest."""
    import hashlib
    h=hashlib.sha256()
    def visit(x):
        if torch.is_tensor(x):
            x=x.detach().cpu().contiguous()
            h.update(str((x.dtype,tuple(x.shape))).encode())
            h.update(x.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(x,dict):
            for k in sorted(x,key=str):h.update(str(k).encode());visit(x[k])
        elif isinstance(x,(tuple,list)):
            for value in x:visit(value)
        else:h.update(repr(x).encode())
    for model in models:visit(model.state_dict())
    for opt in optimizers:visit(opt.state_dict())
    return h.hexdigest()

def check_resume_world(checkpoint,world):
    saved=checkpoint.get("distributed",{}).get("world_size",1)
    if saved!=world:
        raise ValueError(f"resume world_size mismatch: saved {saved}, requested {world}")
    if world>1 and len(checkpoint["distributed"].get("rank_states",[]))!=world:
        raise ValueError("incomplete per-rank resume state")
