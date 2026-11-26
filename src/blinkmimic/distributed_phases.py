"""Local computation phases for global-batch distributed training."""
from blinkmimic.policy_inputs import policy_action, update_norms, normalizer_inputs, environment_orientation
import torch
from .distributed import LocalGraph, zero_grads, optimizer_step, update_bins
from .mdp import unflatten_state, MAX_STEP_REWARD
from .objective import actor_objective

class Collector:
    def __init__(self,g,ctx):
        self.g,self.ctx,self.a=g,ctx,g["args"]
        self.env,self.pi,self.wm,self.motion=g["env"],g["pi"],g["wm"],g["motion"]
        self.buffer=g["buffer"]
        self.omni=g.get("omni_runtime")
        self.env.use_static_focus()
        self.env.distributed_sampling=True
        self.f,self.ep,self.prev=self.env.f,self.env.ep_step,self.env.a_prev
        self.hist=g["obs_hist"]
        self.goal=torch.zeros(self.a.num_envs,g["GOAL_DIM"],device=g["dev"])
        self.orientation=torch.zeros(self.a.num_envs,10,6,device=g["dev"]) if self.pi.observation_mode=="proprio93" else None
        self.done=torch.zeros(self.a.num_envs,dtype=torch.bool,device=g["dev"])
        self.buffer.use_device_ptr(device_size=True)
        self.observe=LocalGraph(self._observe,ctx,self.a.collect_graph)
        self.physics=LocalGraph(self._physics,ctx,self.a.collect_graph,sim=self.env.sim)
        self.reset=LocalGraph(self._reset,ctx,self.a.collect_graph,sim=self.env.sim)

    @torch.no_grad()
    def _observe(self):
        o=self.env.observe()
        h=torch.where((self.env.ep_step==0)[:,None,None],o[:,None].expand_as(self.hist),
                      torch.cat((self.hist[:,1:],o[:,None]),1))
        self.hist.copy_(h)
        self.goal.copy_(self.motion.goal(self.env.f,self.a.goal_frames))
        if self.orientation is not None:self.orientation.copy_(environment_orientation(self.pi,self.env,self.motion,self.a.goal_frames))
        return o

    @torch.no_grad()
    def _physics(self):
        a=(self.omni.collect_action(self.hist) if self.omni else
           policy_action(self.pi,self.hist,self.goal,self.orientation))
        a=(a+self.a.action_noise*torch.randn_like(a)).clamp(-20,20)
        out=self.env.step_core(a)
        if self.omni:
            out["modality"]=self.omni.modality
            self.omni.record(out)
        self.buffer.add_device(out)
        wmdata=self.g["wm_norm_data_c"](unflatten_state(out["s"]),unflatten_state(out["s1"]),
                                      out["a"],out["a_prev"],out["xi"])
        self.f.copy_(self.env.f);self.ep.copy_(self.env.ep_step);self.prev.copy_(self.env.a_prev)
        self.env.f,self.env.ep_step,self.env.a_prev=self.f,self.ep,self.prev
        dw=out["done"].float();T=self.env.ep_step.float()
        boot=torch.where(out["term"],0.,(self.a.gamma**T)*(self.env.ep_ret/T.clamp_min(1))/(1-self.a.gamma))
        G=self.env.ep_disc_ret+boot
        bins=self.env.start_bin
        fb=self.motion.frame_bin[out["f"].clamp(max=self.motion.num_frames-1)]
        base=torch.zeros_like(self.env.bin_G)
        totals=torch.stack((base.index_add(0,bins,dw),base.index_add(0,bins,G*dw),
                            base.index_add(0,fb,dw),base.index_add(0,fb,out["term"].float()*dw)))
        stats=torch.stack((out["r"].sum()/MAX_STEP_REWARD,dw.sum(),(T*dw).sum(),
                           (self.env.ep_ret/(T.clamp_min(1)*MAX_STEP_REWARD)*dw).sum(),
                           (out["term"].float()*dw).sum()))
        return wmdata,totals,stats,out["done"]

    @torch.no_grad()
    def _reset(self):
        self.env.reset_masked(self.done)
        if self.omni:self.omni.reset_modalities(self.done)
        return self.done

    @torch.no_grad()
    def step(self):
        if self.omni:self.omni.prepare_collection()
        o=self.observe()
        if self.omni:
            from .omni import MODALITIES
            inputs=self.omni.collection_inputs()
            self.ctx.moments([self.pi.obs_norm]+[self.pi.input_norms[m] for m in MODALITIES],
                             [o[...,:93]]+[inputs[m] for m in MODALITIES])
        else:
            self.ctx.moments([self.pi.obs_norm,self.pi.goal_norm],normalizer_inputs(self.pi,o,self.goal,self.orientation))
        wmdata,totals,stats,done=self.physics()
        self.buffer.advance_host()
        self.ctx.moments([self.wm.in_norm,self.wm.out_norm,self.wm.inc_norm],wmdata)
        # Reduce NEW sufficient statistics, once per vectorized collection step.
        totals=totals.clone();self.ctx.sum(totals)
        ended=totals[0].sum()
        update_bins(self.env,(*totals,ended))
        self.done.copy_(done)
        self.reset()
        return stats.clone()

class Updates:
    def __init__(self,g,ctx):
        self.g,self.ctx,self.a=g,ctx,g["args"]
        self.wm,self.pi,self.v=g["wm"],g["pi"],g["critic"]
        self.cx=self.cy=None
        self.omni=g.get("omni_runtime")
        self.wphase=LocalGraph(self._world,ctx,self.a.cuda_graph)
        self.pphase=LocalGraph(self._policy,ctx,self.a.cuda_graph)
        self.cphase=LocalGraph(self._critic,ctx,self.a.cuda_graph)
        self.nonfinite=torch.zeros(3,device=g["dev"])
        from .training_diagnostics import TrainingDiagnostics
        self.diagnostics=TrainingDiagnostics(g,ctx)

    def _world(self):
        w=self.g["buffer"].sample_windows(self.a.wm_batch,self.a.wm_window)
        if self.g["mirror"] is not None:w=self.g["_mirror_windows"](w,int(self.a.wm_batch*self.a.wm_mirror))
        predicted=self.g["wm_rollout"](w["s"],w["a"],w["a_prev"],w["xi"])
        loss,e=self.g["wm_err"](predicted,w["s1"],w["mask"])
        zero_grads(self.wm);loss.backward()
        # Original objective is sum(masked errors)/B, NOT /mask.sum().
        mask=w["mask"][:,-1]
        return loss.detach()/self.a.wm_window,(e.detach()*mask).sum(),mask.sum()

    def world(self):
        result=self.wphase()
        _,ok=optimizer_step(self.ctx,self.wm,self.g["opt_wm"])
        self.nonfinite[0].add_((~ok).float())
        return tuple(x.detach().clone() for x in result)

    def _policy(self, diagnostic=False):
        g,a=self.g,self.a
        s,h=self.omni.starts if self.omni else g["sample_valid_starts"](a.pi_batch)
        S=unflatten_state(s["s"]);S0=S
        xi,f,previous=s["xi"],s["f"],s["a_prev"];xi_n=g["normalize_xi"](xi)
        with torch.no_grad():
            history=g["mdp"].obs(unflatten_state(h["s"]),h["a_prev"],h["xi"],h["f"])
            history=g["mdp"].add_noise(history)
        goal0=goal=g["motion"].goal(f,a.goal_frames)
        states,actions,prevs,frames,goals=[],[],[],[],[]
        for t in range(a.horizon):
            prevs.append(previous)
            if self.omni:
                S,previous,history,f,goal=self.omni.imag_step(S,history,previous,f,xi,xi_n,t,s["modality"])
            else:
                S,previous,history,f,goal=g["imag_step"](S,history,previous,f,xi,xi_n,goal)
            states.append(S);actions.append(previous);frames.append(f);goals.append(goal)
        flat=lambda xs:torch.stack(xs).flatten(0,1)
        all_states=tuple(flat([s0[i] for s0 in states]) for i in range(6))
        r,d=g["reward_term"](all_states,flat(frames),flat(actions),flat(prevs))
        r,d=r.view(a.horizon,a.pi_batch),d.view(a.horizon,a.pi_batch)
        boot=self.v(g["critic_input_c"](S,f,xi_n,goal))
        J,ct,c=actor_objective(r,d,boot,a.gamma,loop_cumprod=a.cuda_graph)
        loss=policy_loss=-J.mean()
        if self.omni:
            from .omni import auxiliary_total
            inputs=self.omni.imagination_inputs(0,S0[1])
            terms=self.pi.auxiliary(inputs)
            numerator,count=auxiliary_total(terms)
            auxiliary_loss=numerator/count.clamp_min(1)
            loss=loss+auxiliary_loss
            if diagnostic:
                self.diagnostics.gradients(policy_loss, auxiliary_loss)
                self.diagnostics.scalar("objective/bootstrap_contribution", (a.gamma**a.horizon)*c*boot)
                self.diagnostics.scalar("objective/total_return", J)
            self.omni.aux.copy_(torch.stack([v.mean().detach() for v in terms.values()]))
        zero_grads(self.pi);loss.backward()
        with torch.no_grad():
            sc=tuple(torch.cat((S0[i][None],torch.stack([s0[i].detach() for s0 in states])),0).flatten(0,1) for i in range(6))
            fc=torch.cat((s["f"][None],torch.stack(frames)),0).flatten()
            gc=torch.cat((goal0[None],torch.stack(goals)),0).flatten(0,1)
            xc=xi_n[None].expand(a.horizon+1,-1,-1).flatten(0,1)
            X=g["critic_input_c"](sc,fc,xc,gc).view(a.horizon+1,a.pi_batch,-1)
        return loss.detach(),((ct*r).detach().sum(0)/a.horizon/.1).mean(),c.mean().detach(),X,r.detach(),d.detach()

    def _critic(self):
        loss=self.g["critic_loss_c"](self.cx,self.cy)
        zero_grads(self.v);loss.backward()
        return loss.detach()

    def policy(self):
        g,a=self.g,self.a
        if self.omni:self.omni.prepare_imagination()
        g["set_requires_grad"](self.wm,False);g["set_requires_grad"](self.v,False)
        try:
            pl,ir,surv,X,r,d=self._policy(diagnostic=True) if self.diagnostics.active else self.pphase()
        finally:
            g["set_requires_grad"](self.wm,True);g["set_requires_grad"](self.v,True)
        gn,ok=optimizer_step(self.ctx,self.pi,g["opt_pi"],a.pi_clip_value)
        self.nonfinite[1].add_((~ok).float())
        avg=g["pi_avg"]
        if avg is not None:
            with torch.no_grad():
                first=avg.n_averaged==0
                for dst,src in zip(list(avg.module.parameters())+list(avg.module.buffers()),
                                   list(self.pi.parameters())+list(self.pi.buffers())):
                    mixed=torch.where(first,src.detach(),dst*a.pi_ema+src.detach()*(1-a.pi_ema))
                    dst.copy_(torch.where(ok,mixed,dst))
                avg.n_averaged.add_(ok.long())
        self.ctx.moments([self.v.norm],[X])
        with torch.no_grad():Y=g["gae_targets_c"](self.v(X),r,d)
        if self.diagnostics.active:self.diagnostics.imagined(X,r,d,Y)
        xf=X[:-1].reshape(-1,X.shape[-1]);yf=Y.reshape(-1)
        n=xf.shape[0];mb=n//a.critic_minibatches
        if self.cx is None:
            self.cx=torch.empty_like(xf[:mb]);self.cy=torch.empty_like(yf[:mb])
        losses=[]
        for _ in range(a.critic_epochs):
            perm=torch.randperm(n,device=xf.device)
            for j in range(a.critic_minibatches):
                ids=perm[j*mb:(j+1)*mb]
                self.cx.copy_(xf[ids]);self.cy.copy_(yf[ids])
                loss=self.cphase()
                _,valid=optimizer_step(self.ctx,self.v,g["opt_v"])
                self.nonfinite[2].add_((~valid).float())
                losses.append(loss.clone())
        return tuple(x.detach().clone() for x in (pl,torch.stack(losses).mean(),ir,gn,surv))
