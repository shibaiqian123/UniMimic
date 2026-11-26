"""Opt-in eager diagnostics; never part of graph capture or optimizer updates."""
import json
import time
from pathlib import Path
import torch
from .mdp import unflatten_state


def gradient_contributions(model, policy_loss, auxiliary_loss, ctx):
    """Global mean gradients BEFORE clipping. autograd.grad leaves .grad unchanged.

    Current omni losses have equal local sample counts. Reduce gradient vectors,
    not norms, so cancellation across ranks is represented correctly.
    """
    named = list(model.named_parameters())
    params = [p for _, p in named]
    gp = torch.autograd.grad(policy_loss, params, retain_graph=True, allow_unused=True)
    ga = torch.autograd.grad(auxiliary_loss, params, retain_graph=True, allow_unused=True)
    groups = {**{"encoder_" + m: "encoders." + m + "." for m in model.encoders},
              "action_decoder": "dec.", "reconstruction": "reconstruction."}
    result = {}
    for group, prefix in groups.items():
        ids = [i for i, (name, _) in enumerate(named) if name.startswith(prefix)]
        vectors = []
        for grads in (gp, ga):
            flat = torch.cat([(torch.zeros_like(params[i]) if grads[i] is None else grads[i])
                              .detach().float().reshape(-1) for i in ids])
            ctx.sum(flat); flat.div_(ctx.world)
            vectors.append(flat)
        p, a = vectors
        pn, an = p.norm(), a.norm()
        dot = (p * a).sum()
        result.update({f"gradient/{group}/policy_norm": float(pn),
                       f"gradient/{group}/auxiliary_norm": float(an),
                       f"gradient/{group}/combined_norm": float((p + a).norm()),
                       f"gradient/{group}/cosine": float(dot / (pn * an).clamp_min(1e-30)),
                       f"gradient/{group}/cosine_defined": float((pn > 0) & (an > 0)),
                       f"gradient/{group}/aux_over_policy": float(an / pn.clamp_min(1e-30))})
    return result


class TrainingDiagnostics:
    def __init__(self, g, ctx):
        self.g, self.ctx, self.a = g, ctx, g["args"]
        self.root = Path(self.a.log_dir)
        self.iteration = 0
        self.active = False
        self.metrics = {}
        self.panel = None
        self.started = 0.

    def begin(self, iteration):
        self.iteration = iteration
        every = self.a.diagnostic_every
        self.active = bool(every and iteration >= self.a.diagnostic_start
                           and (not self.a.diagnostic_end or iteration <= self.a.diagnostic_end)
                           and iteration % every == 0)
        self.metrics = {}
        if self.active:
            self.started = time.monotonic()

    def gradients(self, policy, auxiliary):
        self.metrics.update(gradient_contributions(self.g["pi"], policy, auxiliary, self.ctx))
        self.scalar("loss/policy", policy)
        self.scalar("loss/auxiliary_weighted", auxiliary)

    def scalar(self, name, value):
        value = value.detach().float().mean().clone()
        self.ctx.sum(value)
        self.metrics[name] = float(value / self.ctx.world)

    @torch.no_grad()
    def masked(self, name, values, mask):
        total = torch.stack(((values * mask).sum(), mask.sum())).float()
        self.ctx.sum(total)
        self.metrics[name] = float(total[0] / total[1].clamp_min(1))
        self.metrics[name + "_count"] = float(total[1])

    @torch.no_grad()
    def imagined(self, X, r, d, Y):
        v = self.g["critic"](X)
        self.scalar("critic/imagined_target_mse_before_update", (v[:-1] - Y).square())
        td = r + self.a.gamma * (1 - d) * v[1:] - v[:-1]
        self.scalar("critic/imagined_td_mse", td.square())
        self.scalar("critic/imagined_value_mean", v)
        self.scalar("critic/imagined_value_square_mean", v.square())
        self.scalar("critic/imagined_target_mean", Y)

    @torch.no_grad()
    def replay(self):
        g, a = self.g, self.a
        path = self.root / f"diagnostic_panel_rank{self.ctx.rank}.pt"
        contract = dict(world_size=self.ctx.world, rank=self.ctx.rank,
                        index=g["GLOBAL_ARGS"].get("omni_index_sha256"),
                        window=a.diagnostic_window, batch=a.diagnostic_batch, seed=a.seed)
        if self.panel is None:
            if path.exists():
                saved = torch.load(path, map_location=g["dev"], weights_only=False)
                if saved["contract"] != contract:
                    raise ValueError("diagnostic panel contract mismatch")
                self.panel = saved["samples"]
            else:
                # Fork restores CPU and CUDA RNG; diagnostic sampling must not perturb training.
                with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                    torch.manual_seed(a.seed + 91073 + self.ctx.rank)
                    samples = g["buffer"].sample_windows(a.diagnostic_batch, a.diagnostic_window,
                                                          extra_keys=("f", "r", "done"))
                self.panel = {k: v.detach().clone() for k, v in samples.items()}
                from .train_distributed import atomic_save
                atomic_save(dict(contract=contract, iteration=self.iteration,
                                 samples={k: v.cpu() for k, v in self.panel.items()}), path)
        w = self.panel
        wm = g["wm"]
        state = unflatten_state(w["s"][:, 0])
        std = torch.maximum(wm.inc_norm.std(), g["ERR_FLOOR"])
        for t in range(a.diagnostic_window):
            state = wm.step(state, w["a"][:, t], w["a_prev"][:, t],
                            g["normalize_xi"](w["xi"][:, t]))
            if t + 1 in {1, 4, 8, 16, 32, a.diagnostic_window}:
                error = wm.state_error(state, unflatten_state(w["s1"][:, t]))
                mask = w["mask"][:, t]
                self.masked(f"world/fixed_normalized_mse_h{t+1}",
                            (error / std).square().mean(-1), mask)
                for name, sl in (("position_m", slice(0, 3)), ("rotation_rad", slice(3, 6)),
                                 ("velocity_m_s", slice(6, 9)), ("omega_rad_s", slice(9, 12)),
                                 ("joint_rad", slice(12, 41)), ("joint_velocity_rad_s", slice(41, 70))):
                    key = f"world/fixed_{name}_rmse_h{t+1}"
                    self.masked(key, error[:, sl].square().mean(-1), mask)
                    self.metrics[key] = self.metrics[key] ** .5
        self.scalar("world/normalization_std_mean", std)
        # One-step Bellman residual on the same real simulator transitions.
        # This is NOT a ground-truth Monte Carlo value error.
        values = []
        for key, offset in (("s", 0), ("s1", 1)):
            f = (w["f"] + offset).flatten()
            S = unflatten_state(w[key].flatten(0, 1))
            x = g["critic_input_c"](S, f, g["normalize_xi"](w["xi"].flatten(0, 1)),
                                     g["motion"].goal(f, a.goal_frames))
            values.append(g["critic"](x).view_as(w["r"]))
        # Match the actor/critic soft continuation; episode timeouts are not terminal values.
        next_f = (w["f"] + 1).flatten()
        soft_done = g["mdp"].soft_termination(unflatten_state(w["s1"].flatten(0, 1)),
                                             next_f).view_as(w["r"])
        td = w["r"] + a.gamma * (1 - soft_done) * values[1] - values[0]
        self.masked("critic/fixed_real_td_mse", td.square(), w["mask"])
        self.masked("critic/fixed_real_value_mean", values[0], w["mask"])

    def finish(self, tb):
        if not self.active:
            return
        self.replay()
        elapsed = torch.tensor(time.monotonic() - self.started, device=self.g["dev"])
        self.scalar("diagnostic_wall_s_mean", elapsed)
        if self.ctx.main:
            row = dict(iteration=self.iteration, metrics=self.metrics,
                       scope="preclip global gradients; fixed training-replay panel, not held-out validation",
                       policy_gradient_point="before actor update",
                       replay_error_point="after world, actor and critic updates")
            with (self.root / "training_diagnostics.jsonl").open("a") as f:
                f.write(json.dumps(row) + "\n")
            if tb:
                for key, value in self.metrics.items():
                    tb.add_scalar("diagnostic/" + key, value, self.iteration)
