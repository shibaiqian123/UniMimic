"""Two-rank orchestration; same global work per iteration as the single-GPU recipe."""
import json
import os
import random
import time
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from .distributed_phases import Collector, Updates
from .latent import check_resume
from .distributed import check_resume_world

def rng_state():
    return dict(torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state(),
                numpy=np.random.get_state(),python=random.getstate())

def restore_rng(state):
    torch.set_rng_state(state["torch"].cpu());torch.cuda.set_rng_state(state["cuda"].cpu())
    np.random.set_state(state["numpy"]);random.setstate(state["python"])

def atomic_save(value,path):
    tmp=str(path)+".tmp";torch.save(value,tmp);os.replace(tmp,path)

def run(g):
    ctx,a=g["DIST_CTX"],g["args"]
    root=Path(a.log_dir);wm,pi,v=g["wm"],g["pi"],g["critic"]
    avg=g["pi_avg"];env=g["env"]
    # Keep the reference single-GPU runner independent of distributed control flow.
    if g["pipe"].stateful or env.push_enabled:
        raise ValueError("first distributed collector supports stateless G1 sensing without pushes")
    if not a.bin_fix:
        raise ValueError("distributed sampling requires --bin_fix for deterministic per-bin statistics")
    collector=Collector(g,ctx);updates=Updates(g,ctx)
    start=0;offset=0.;sim_steps=0;step_acc=0.;decay_at=None
    if a.resume:
        ck=torch.load(a.resume,map_location=g["dev"],weights_only=False)
        check_resume_world(ck,ctx.world)
        check_resume(ck["args"],g["GLOBAL_ARGS"])
        for key in ("num_envs","wm_batch","pi_batch","horizon","utd","buffer_steps","warmup_steps",
                    "critic_minibatches","critic_epochs","wm_batches","seed","wm_mirror"):
            if ck["args"].get(key)!=g["GLOBAL_ARGS"].get(key):
                raise ValueError("resume global training setting mismatch: "+key)
        wm.load_state_dict(ck["wm"]);pi.load_state_dict(ck.get("pi_live",ck["pi"]));v.load_state_dict(ck["critic"])
        g["load_opt_states"](ck)
        if avg is not None:
            if ck.get("pi_avg") is None:raise ValueError("EMA resume state missing")
            avg.load_state_dict(ck["pi_avg"])
        ctrl=ck["ctrl"]
        for name,val in ctrl["env_bins"].items():getattr(env,name).copy_(val)
        env.set_focus(ctrl["focus_q"],ctrl["focus_share"])
        if g["EPROBE"] and ctrl.get("ep_fail") is not None:g["EP_FAIL"].copy_(ctrl["ep_fail"])
        start,offset,sim_steps=ck["iteration"],ck["wall_time"],ck["sim_steps"]
        step_acc=ctrl["step_acc"];decay_at=ctrl["decay_at"]
        rank_states=ck["distributed"]["rank_states"]
        if len(rank_states)!=ctx.world:raise ValueError("incomplete per-rank resume state")
        restore_rng(rank_states[ctx.rank])
        if "nonfinite" in ctrl:updates.nonfinite.copy_(ctrl["nonfinite"])
        del ck
    ctx.broadcast_models(wm,pi,v)
    if avg is not None:ctx.broadcast_models(avg)
    if ctx.main:
        print(f"Distributed global batches: {g['GLOBAL_ARGS']['distributed']}; LR unchanged; "
              f"local update graphs={a.cuda_graph}, collection graphs={a.collect_graph}",flush=True)
    # Warmup refills local replay buffers. It is additional sampling, not optimizer updates.
    init_start=time.monotonic()
    for _ in range(a.warmup_steps):collector.step();sim_steps+=g["GLOBAL_ARGS"]["num_envs"]
    torch.cuda.synchronize()
    warm_s=time.monotonic()-init_start
    ctx.communication_seconds()  # exclude startup broadcast and buffer warmup from steady-state counters
    began=time.monotonic();log_start=began
    last_save_wall=offset
    logf=tb=None
    if ctx.main:
        path=root/"train_log.csv";new=not path.exists()
        logf=path.open("a",buffering=1)
        if new:logf.write("iteration,wall_h,sim_steps,wm_loss,wm_err8,pi_loss,v_loss,imag_rew,pi_gradnorm,imag_survival,sim_rew,ep_len,ep_ret_norm,term_frac,it_per_s,lr,pi_lr,pi_mult,nonfinite\n")
        if a.tensorboard:
            tb=g["SummaryWriter"](log_dir=str(root/"tensorboard"),purge_step=start+1 if a.resume else None,flush_secs=10)
        (root/"distributed_runtime.json").write_text(json.dumps(dict(world_size=ctx.world,
            global_args=g["GLOBAL_ARGS"],local_sizes={k:getattr(a,k) for k in ("num_envs","wm_batch","pi_batch")},
            warmup_wall_s=warm_s,local_graphs=bool(a.cuda_graph),collection_graphs=bool(a.collect_graph),
            hardware=[torch.cuda.get_device_name()],resume_is_warm_restart=True),indent=2))
    # Fixed-layout local accumulators: wm loss, error sum/count, policy losses and completed episode sums.
    agg=torch.zeros(14,device=g["dev"]);nlog=0
    phase_events=[]
    log_steps=sim_steps
    def event():
        e=torch.cuda.Event(enable_timing=True);e.record();return e
    try:
        for it in range(start,a.iterations):
            updates.diagnostics.begin(it+1)
            lr=g["lr_at"](it)
            mult=a.pi_lr_mult
            if decay_at is not None:mult=a.pi_lr_mult+(1-a.pi_lr_mult)*min(1.,(it-decay_at)/max(a.lr_decay_len,1))
            for opt in (g["opt_wm"],g["opt_pi"],g["opt_v"]):
                for group in opt.param_groups:
                    value=lr*(mult if opt is g["opt_pi"] else 1.)
                    if torch.is_tensor(group["lr"]):group["lr"].fill_(value)
                    else:group["lr"]=value
            if g["EPROBE"]:
                if a.eprobe_async:
                    g["ITER_NOW"][0]=it
                    g["eprobe_async_tick"](launch=False)
                if it>=a.eprobe_start and it%a.eprobe_every==0:
                    if a.eprobe_async and g["EP_ASYNC"]:raise RuntimeError("probe sweep overlaps its next scheduled launch")
                    g["eprobe_burst"](it)
                if a.eprobe_async:g["eprobe_async_tick"](launch=True)
            # Rank 0 is the only prober. Publish one consistent focus snapshot at iteration boundary.
            if a.eprobe_share>0:
                ctx.broadcast(env.focus_q);ctx.broadcast(env.focus_share_t)
            step_acc+=1./a.utd;n=int(step_acc);step_acc-=n
            e0=event()
            for _ in range(n):
                stats=collector.step()
                agg[9:14].add_(stats)
                sim_steps+=g["GLOBAL_ARGS"]["num_envs"]
            e1=event()
            for _ in range(a.wm_batches):wl,we,wc=updates.world()
            agg[:3].add_(torch.stack((wl,we,wc)))
            if it>=a.wm_only_iters:
                pl,vl,ir,gn,surv=updates.policy()
                agg[3:8].add_(torch.stack((pl,vl,ir,gn,surv)));agg[8]+=1
            updates.diagnostics.finish(tb)
            e2=event();phase_events.append((e0,e1,e2))
            nlog+=1
            if (it+1)%a.log_every==0 or it+1==a.iterations:
                total=agg.clone();ctx.sum(total)
                torch.cuda.synchronize()
                elapsed=time.monotonic()-log_start;log_start=time.monotonic()
                # All ranks use the same global completed-episode return for the LR trigger.
                vals=total.cpu().tolist();pn=max(vals[8],1);episodes=max(vals[10],1)
                row=[it+1,(time.monotonic()-began+offset)/3600,sim_steps,vals[0]/(nlog*ctx.world),
                     vals[1]/max(vals[2],1),vals[3]/pn,vals[4]/pn,vals[5]/pn,vals[6]/pn,vals[7]/pn,
                     vals[9]/max(sim_steps-log_steps,1),
                     vals[11]/episodes,vals[12]/episodes,vals[13]/episodes,nlog/elapsed,lr,
                     lr*mult,mult,int(updates.nonfinite[1])]
                # Reward denominator counts actual collected transitions, including alternating 2/3 steps.
                timing=torch.tensor([sum(x.elapsed_time(y) for x,y,_ in phase_events)/1000,
                                     sum(y.elapsed_time(z) for _,y,z in phase_events)/1000,
                                     ctx.communication_seconds(),elapsed,
                                     torch.cuda.max_memory_allocated()/2**30],device=g["dev"])
                gathered=ctx.gather_objects(timing.cpu().tolist())
                if a.lr_decay_ret>0 and decay_at is None and (row[12]>=a.lr_decay_ret or (a.lr_decay_by>0 and it+1>=a.lr_decay_by)):
                    decay_at=it+1
                if ctx.main:
                    logf.write(",".join(str(x) for x in row)+"\n")
                    print(f"it {it+1} | global_steps {sim_steps} | return {row[12]:.4f} "
                          f"| {row[14]:.2f} it/s | world={ctx.world}",flush=True)
                    perf=dict(iteration=it+1,world_size=ctx.world,iterations=nlog,wall_s=elapsed,
                              per_rank=gathered,columns=["collection_s","learning_s","communication_s","wall_s","peak_allocated_GB"],
                              global_envs=g["GLOBAL_ARGS"]["num_envs"],global_samples=sim_steps-log_steps)
                    with (root/"distributed_perf.jsonl").open("a") as f:f.write(json.dumps(perf)+"\n")
                    if tb:
                        names="wm_loss wm_err8 pi_loss v_loss imag_rew pi_gradnorm imag_survival sim_rew ep_len ep_ret_norm term_frac it_per_s lr pi_lr pi_mult nonfinite".split()
                        for name,value in zip(names,row[3:]):tb.add_scalar("train/"+name,value,it+1)
                        tb.add_scalar("distributed/communication_s_max",max(x[2] for x in gathered),it+1)
                if a.dist_audit:
                    from .distributed import state_digest
                    digest=state_digest([wm,pi,v]+([avg] if avg is not None else []),
                                        [g["opt_wm"],g["opt_pi"],g["opt_v"]])
                    hashes=ctx.gather_objects(digest)
                    if len(set(hashes))!=1:raise RuntimeError("distributed state audit failed")
                    if ctx.main:
                        with (root/"distributed_audit.jsonl").open("a") as f:
                            f.write(json.dumps(dict(iteration=it+1,world_size=ctx.world,hashes=hashes,passed=True))+"\n")
                latent_log(g,ctx,tb,root,it+1)
                log_steps=sim_steps
                agg.zero_();nlog=0;phase_events.clear()
            # A rank-0 clock chooses scheduled checkpoints and early stop for every rank.
            wall=time.monotonic()-began+offset
            flags=torch.tensor([int(a.max_hours is not None and wall>=a.max_hours*3600),
                                int(a.ckpt_minutes>0 and wall-last_save_wall>=a.ckpt_minutes*60)],
                               device=g["dev"])
            ctx.broadcast(flags)
            stop,timed=map(bool,flags.tolist())
            final=it+1==a.iterations or stop
            if timed:last_save_wall=wall
            full_due=bool(a.full_ckpt_every and (it+1)%a.full_ckpt_every==0)
            if final or timed or full_due or (it+1)%a.ckpt_every==0:
                light=a.ckpt_light and not (final or full_due)
                rank_states=None if light else ctx.gather_objects(rng_state())
                if ctx.main:
                    ck=dict(pi=avg.module.state_dict() if avg else pi.state_dict(),
                            iteration=it+1,wall_time=wall,sim_steps=sim_steps,args=g["GLOBAL_ARGS"],
                            saved_at=time.strftime("%Y-%m-%d %H:%M:%S"))
                    if not light:
                        ck.update(wm=wm.state_dict(),pi_live=pi.state_dict(),critic=v.state_dict(),
                                  opt_wm=g["opt_wm"].state_dict(),opt_pi=g["opt_pi"].state_dict(),
                                  opt_v=g["opt_v"].state_dict(),pi_avg=avg.state_dict() if avg else None,
                                  distributed=dict(world_size=ctx.world,rank_states=rank_states))
                        ck["ctrl"]=dict(step_acc=step_acc,decay_at=decay_at,nonfinite=updates.nonfinite.cpu(),
                            ep_fail=g["EP_FAIL"].cpu() if g["EPROBE"] else None,
                            focus_q=env.focus_q.cpu(),focus_share=float(env.focus_share_t),
                            env_bins={k:getattr(env,k).cpu() for k in ("bin_G","bin_fail","bin_w_avg")})
                        atomic_save(ck,root/"last.pt")
                        if full_due:atomic_save(ck,root/f"full_{it+1:06d}.pt")
                    compact={k:ck[k] for k in ("pi","iteration","wall_time","sim_steps","args","saved_at")}
                    atomic_save(compact,root/f"ckpt_{it+1:06d}.pt")
                ctx.barrier()
            if stop:break
        phase_names={"observe":collector.observe,"physics":collector.physics,"reset":collector.reset,
                     "world_backward":updates.wphase,"policy_backward":updates.pphase,"critic_backward":updates.cphase}
        captures=ctx.gather_objects({k:dict(calls=p.calls,captured=p.graph is not None) for k,p in phase_names.items()})
        if ctx.main:
            (root/"distributed_phases.json").write_text(json.dumps(captures,indent=2)+"\n")
    finally:
        if logf:logf.close()
        if tb:tb.close()
        if dist.is_initialized():dist.destroy_process_group()

@torch.no_grad()
def latent_log(g,ctx,tb,root,it):
    if g.get("omni_runtime") is not None:
        g["omni_runtime"].log(ctx,tb,root,it)
        return
    if g["args"].latent_mode=="continuous32":return
    from .latent import bound,quantize_bounded,HALF
    motion,pi=g["motion"],g["pi"]
    from blinkmimic.probe_batches import diagnostic_frames
    ids=diagnostic_frames(motion.num_frames,256,g["dev"])
    ids=ids[ctx.rank::ctx.world]
    from .policy_inputs import diagnostic_orientation
    goals=motion.goal(ids,g["K"])
    u=bound(pi.enc_raw(goals,diagnostic_orientation(pi,motion,ids)));z=quantize_bounded(u)
    indices=(z*HALF+HALF).long().clamp(0,31).reshape(-1,64)
    counts=torch.stack([(indices==j).sum(0) for j in range(32)],-1).float();ctx.sum(counts)
    err=torch.stack(((z-u/HALF).square().sum(),u.new_tensor(u.numel())));ctx.sum(err)
    if ctx.main:
        used=counts.gt(0).sum(-1).cpu().tolist()
        sat=((counts[:,0]+counts[:,-1])/counts.sum(-1).clamp_min(1)).cpu().tolist()
        gs=[p.grad.square().sum() for p in pi.enc.parameters() if p.grad is not None]
        grad=torch.stack(gs).sum().sqrt() if gs else torch.zeros((),device=g["dev"])
        result=dict(iteration=it,mode=g["args"].latent_mode,observation_mode=pi.observation_mode,
                    orientation_condition="current_reference_aligned" if pi.observation_mode=="proprio93" else "reference_only",quantization_mse=float(err[0]/err[1]),
                    used_levels=used,utilization=[x/32 for x in used],saturation=sat,encoder_grad_norm=float(grad))
        with (root/"latent_metrics.jsonl").open("a") as f:f.write(json.dumps(result)+"\n")
        if tb:
            for key in ("quantization_mse","encoder_grad_norm"):tb.add_scalar("latent/"+key,result[key],it)
            for i in range(64):
                for key in ("used_levels","utilization","saturation"):tb.add_scalar(f"latent/{key}/{i}",result[key][i],it)
