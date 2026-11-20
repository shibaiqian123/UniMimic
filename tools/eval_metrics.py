"""Tracking-accuracy and contact metrics for a set of checkpoints (same rollout protocol as the evaluator:
each clip once from its first frame, friction 1.0, no randomization or noise, deterministic policy).

Per checkpoint, averaged over all alive control steps of all 40 clips:
  body_pos_cm   RMS error of the 14 tracked bodies (reference re-anchored to the robot root, as in the reward)
  body_ori_deg  RMS orientation error of the tracked bodies
  root_pos_cm   global root position error (drift)
  joint_deg     mean absolute joint-angle error
  slip_cm_s     horizontal foot speed while the simulated foot is in contact (ankle link < CONTACT_Z)
  ref_slip_cm_s the same on the reference motion (retargeting artifacts set a floor)
  contact_acc   fraction of foot-frames whose contact state matches the reference
  pen_mm        mean ankle-height deficit below the flat-foot height (ground penetration proxy)
plus normalized return and failed clips.

    uv run python tools/eval_metrics.py logs/<run_a>/ckpt_003500.pt logs/<run_b>/ckpt_003500.pt --out results/metrics
"""

import argparse
import hashlib
import time
from pathlib import Path
import json
import os

parser = argparse.ArgumentParser()
parser.add_argument("ckpts", nargs="+")
parser.add_argument("--motion_dir", type=str, default=None)
parser.add_argument("--robot", type=str, default="g1")
parser.add_argument("--out", type=str, default="results/metrics")
args = parser.parse_args()

import torch

import blinkmimic.networks as networks
from blinkmimic.dr import XiLayout
from blinkmimic.env import TrackingSim, make_sim
from blinkmimic.kinematics import Kinematics
from blinkmimic.mdp import MAX_STEP_REWARD, MDP
from blinkmimic.motion import MotionLib
from blinkmimic.networks import Policy, policy_kwargs
from blinkmimic.robot import get_robot

CONTACT_Z = 0.045  # ankle-roll link origin height below which a foot counts as in contact (flat foot: ~0.035 m)
FLAT_Z = 0.033

torch.backends.cuda.matmul.allow_tf32 = True
dev = "cuda:0"
robot = get_robot(args.robot)
kin = Kinematics(robot, dev)
motion = MotionLib.from_dir(args.motion_dir or robot.motion_dir, robot, kin, dev)
mdp = MDP(kin, motion, torch.tensor(robot.default_joint_pos_list(), device=dev), XiLayout(robot, dev).dim, robot.sensing)
C = len(motion.names)
FEET = [robot.tracked_bodies.index(b) for b in ("left_ankle_roll_link", "right_ankle_roll_link")]


def load_policy(ck):
    a_ = ck["args"]
    widths = lambda key: tuple(int(x) for x in a_.get(key, "2048,1024,512,256").split(","))  # noqa: E731
    pi = Policy(mdp.obs_dim, motion.goal_dim(a_["goal_frames"]), robot.num_joints, hist=a_["hist"],
                enc_hidden=widths("enc_hidden"), dec_hidden=widths("dec_hidden"), **policy_kwargs(a_, robot)).to(dev)  # fmt: skip
    pi.load_state_dict(ck["pi"])
    return pi.eval()


@torch.no_grad()
def run(cks):
    M = len(cks)
    N = C * M
    env = TrackingSim(make_sim(robot), N, robot, motion, mdp, dev, eval_mode=True, randomize=False)
    pis = [load_policy(ck) for ck in cks]
    K, Hh = cks[0]["args"]["goal_frames"], cks[0]["args"]["hist"]
    networks.AMP_DTYPE = torch.bfloat16 if cks[0]["args"].get("amp", "none") == "bf16" else None
    clip_of = torch.arange(N, device=dev) % C
    env.reset(torch.arange(N, device=dev), motion.clip_start[clip_of].clone())
    clip_end = motion.clip_end[clip_of]
    keys = ("n", "ret", "pos", "ori", "root", "joint", "slip", "slip_n", "ref_slip", "ref_slip_n", "c_ok", "c_n", "pen", "action_delta", "delta_n")
    acc = {k: torch.zeros(N, device=dev) for k in keys}
    hist, a = None, torch.zeros(N, robot.num_joints, device=dev)
    previous=None
    started=last_progress=time.monotonic()
    steps=0
    print(f"Evaluation started: {M} checkpoints x {C} clips = {N} environments", flush=True)
    while not bool(env.done_eval.all()):
        o = env.observe()
        hist = o.unsqueeze(1).repeat(1, Hh, 1) if hist is None else torch.cat((hist[:, 1:], o.unsqueeze(1)), 1)
        g = motion.goal(env.f, K)
        for m, pi in enumerate(pis):
            sl = slice(m * C, (m + 1) * C)
            a[sl] = pi(hist[sl], g[sl])
        alive = (~env.done_eval).float()
        if previous is not None:
            acc["action_delta"] += alive * (a-previous).square().mean(-1)
            acc["delta_n"] += alive
        previous=a.clone()
        f1 = env.f + 1
        out = env.step(a)
        S1 = env.state()
        t = out["terms"]
        bp, _, bv, _ = kin.forward(*S1)
        fz, fv = bp[:, FEET, 2], bv[:, FEET, :2].norm(dim=-1)
        rz, rv = motion.body_p[f1][:, FEET, 2], motion.body_v[f1][:, FEET, :2].norm(dim=-1)
        c_sim, c_ref = fz < CONTACT_Z, rz < CONTACT_Z
        acc["n"] += alive
        acc["ret"] += alive * out["r"]
        acc["pos"] += alive * t["e_body_pos"]
        acc["ori"] += alive * t["e_body_ori"]
        acc["root"] += alive * t["e_root_pos"].sqrt()
        acc["joint"] += alive * (S1[4] - motion.q[f1]).abs().mean(-1)
        acc["slip"] += alive * (fv * c_sim).sum(-1)
        acc["slip_n"] += alive * c_sim.float().sum(-1)
        acc["ref_slip"] += alive * (rv * c_ref).sum(-1)
        acc["ref_slip_n"] += alive * c_ref.float().sum(-1)
        acc["c_ok"] += alive * (c_sim == c_ref).float().sum(-1)
        acc["c_n"] += alive * 2
        acc["pen"] += alive * (FLAT_Z - fz).clamp(min=0).sum(-1) / 2
        env.f = torch.minimum(env.f, clip_end - 1)
        steps+=1
        now=time.monotonic()
        if now-last_progress>=5:
            done=env.done_eval.reshape(M,C).sum(-1).tolist()
            failed=env.failed_eval.reshape(M,C).sum(-1).tolist()
            print(f"[eval] steps={steps} elapsed={now-started:.0f}s speed={steps/max(now-started,1e-6):.1f} steps/s "
                  f"completed={done}/{C} failed={failed} (checkpoint argument order)", flush=True)
            last_progress=now
    clip_len = motion.clip_len[clip_of]
    res = []
    for m in range(M):
        s = {k: v[m * C : (m + 1) * C].sum().item() for k, v in acc.items()}
        n = s["n"]
        res.append({
            "action_delta_rms": (s["action_delta"]/max(s["delta_n"],1))**.5,
            "norm_return": (acc["ret"][m * C : (m + 1) * C] / ((clip_len[m * C : (m + 1) * C] - 1).float() * MAX_STEP_REWARD)).mean().item(),
            "n_failed": int(env.failed_eval[m * C : (m + 1) * C].sum().item()),
            "body_pos_cm": 100 * (s["pos"] / n) ** 0.5,
            "body_ori_deg": 57.2958 * (s["ori"] / n) ** 0.5,
            "root_pos_cm": 100 * s["root"] / n,
            "joint_deg": 57.2958 * s["joint"] / n,
            "slip_cm_s": 100 * s["slip"] / max(s["slip_n"], 1),
            "ref_slip_cm_s": 100 * s["ref_slip"] / max(s["ref_slip_n"], 1),
            "contact_acc": s["c_ok"] / s["c_n"],
            "pen_mm": 1000 * s["pen"] / n,
        })  # fmt: skip
    return res


cks = [torch.load(p, map_location=dev, weights_only=False) for p in args.ckpts]
contracts={(ck["args"]["hist"],ck["args"]["goal_frames"],ck["args"].get("amp","none")) for ck in cks}
if len(contracts)!=1:raise ValueError("comparison requires equal history, goal window and precision")
res = run(cks)
cols = ["action_delta_rms", "norm_return", "n_failed", "body_pos_cm", "body_ori_deg", "root_pos_cm", "joint_deg", "slip_cm_s", "ref_slip_cm_s", "contact_acc", "pen_mm"]
lines = ["| checkpoint | train h | " + " | ".join(cols) + " |", "|---" * (len(cols) + 2) + "|"]
for p, ck, r in zip(args.ckpts, cks, res):
    name = f"{os.path.basename(os.path.dirname(p))}/{os.path.basename(p)}"
    lines.append(f"| {name} | {ck['wall_time'] / 3600:.2f} | " + " | ".join(f"{r[c]:.3f}" if isinstance(r[c], float) else str(r[c]) for c in cols) + " |")
table = "\n".join(lines)
print(table)
os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
open(args.out + ".md", "w").write(table + "\n")
json.dump({p: r for p, r in zip(args.ckpts, res)}, open(args.out + ".json", "w"), indent=1)

manifest = dict(protocol="clip-start-no-noise-no-DR", motion_dir=str(Path(args.motion_dir or robot.motion_dir).resolve()),
                files=[dict(name=p.name,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
                       for p in sorted(Path(args.motion_dir or robot.motion_dir).glob(robot.motion_glob))],
                checkpoints=[dict(path=str(Path(p).resolve()),sha256=hashlib.sha256(Path(p).read_bytes()).hexdigest(),
                                  latent_spec=ck["args"].get("latent_spec"),seed=ck["args"].get("seed"),
                                  iteration=ck["iteration"]) for p,ck in zip(args.ckpts,cks)])
json.dump(manifest,open(args.out+".manifest.json","w"),indent=2)
