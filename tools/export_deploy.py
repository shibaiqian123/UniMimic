"""Export a standalone single-motion bundle (does not arm or contact a robot)."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from blinkmimic import networks
from blinkmimic.networks import Policy, policy_kwargs
from blinkmimic.kinematics import Kinematics
from blinkmimic.motion import MotionLib, CONTROL_DT
from blinkmimic.robot import get_robot
from blinkmimic.rotations import mat_to_6d, yaw_mat
from blinkmimic.g1.g1_constants import motor_of
from blinkmimic.deploy.core import digest, OBS_LAYOUT, Bundle, OnnxPolicy, STAGES, LIMITS
from export_onnx import pd_gains

class ExportPolicy(nn.Module):
    def __init__(self, policy):
        super().__init__()
        self.policy = policy
    def forward(self, obs_history, goal):
        return self.policy(obs_history,goal)

class ExportEncoder(nn.Module):
    def __init__(self, policy):
        super().__init__(); self.policy=policy
    def forward(self, goal):
        return self.policy.encode_tokens(goal)

class ExportDecoder(nn.Module):
    def __init__(self, policy):
        super().__init__(); self.policy=policy
    def forward(self, obs_history, tokens):
        return self.policy.decode_tokens(obs_history,tokens)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt",required=True,type=Path)
    ap.add_argument("--motion",required=True,type=Path)
    ap.add_argument("--out",required=True,type=Path)
    ap.add_argument("--preparation-gain-scale",type=float,default=4.,help="shared simulation/hardware preparation kp multiplier; kd uses its square root")
    a=ap.parse_args()
    if not np.isfinite(a.preparation_gain_scale) or a.preparation_gain_scale<=0:ap.error("positive preparation gain required")
    if a.out.exists():
        raise ValueError("output must be a new directory")
    torch.set_num_threads(4)
    ck=torch.load(a.ckpt,map_location="cpu",weights_only=False)
    cfg=ck["args"]
    if cfg["hist"] != 10 or cfg["goal_frames"] != 10:
        raise ValueError("v1 requires 10 history and 10 goal frames")
    robot=get_robot("g1")
    if robot.sensing.stateful:
        raise ValueError("stateful sensing unsupported by v1")
    kin=Kinematics(robot,"cpu")
    motion=MotionLib([str(a.motion)],kin,"cpu",robot.load_motion)
    widths=lambda k: tuple(map(int,cfg.get(k,"2048,1024,512,256").split(",")))
    pi=Policy(99,730,29,hist=10,enc_hidden=widths("enc_hidden"),dec_hidden=widths("dec_hidden"),
              **policy_kwargs(cfg,robot)).eval()
    pi.load_state_dict(ck["pi"],strict=True)
    networks.AMP_DTYPE=None
    wrapped=ExportPolicy(pi).eval()
    frames=torch.arange(motion.num_frames)
    goals=motion.goal(frames,10)
    static=[]
    for f in (0,motion.num_frames-1):
        R=motion.root_R[f]
        rot=mat_to_6d(yaw_mat(R).T@R)
        one=torch.cat((torch.zeros(3),rot,torch.zeros(6),motion.q[f],torch.zeros(29)))
        static.append(one.repeat(10))
    a.out.mkdir(parents=True)
    torch.onnx.export(wrapped,(torch.zeros(1,10,99),goals[:1]),str(a.out/"policy.onnx"),
                      input_names=["obs_history","goal"],output_names=["actions"],
                      dynamic_axes={"obs_history":{0:"batch"},"goal":{0:"batch"},"actions":{0:"batch"}},
                      opset_version=18,dynamo=False)
    files=["policy.onnx","motion.npz"]
    if pi.latent_mode != "continuous32":
        for name, model, inputs, input_names, output_name in (
            ("motion_encoder.onnx", ExportEncoder(pi), (goals[:1],), ["goal"], "tokens"),
            ("decoder.onnx", ExportDecoder(pi), (torch.zeros(1,10,99), pi.encode_tokens(goals[:1]).detach()), ["obs_history","tokens"], "actions"),
        ):
            torch.onnx.export(model.eval(), inputs, str(a.out/name), input_names=input_names,
                output_names=[output_name], dynamic_axes={n:{0:"batch"} for n in input_names+[output_name]},
                opset_version=18, dynamo=False)
            files.append(name)
    np.savez_compressed(a.out/"motion.npz",q=motion.q.numpy(),root_R=motion.root_R.numpy(),
                        root_p=motion.root_p.numpy(),qd=motion.qd.numpy(),root_v=motion.root_v.numpy(),root_w=motion.root_w.numpy(),goal=goals.numpy(),static_goal=torch.stack(static).numpy())
    kp,kd=pd_gains(robot)
    motors=[motor_of(n) for n in robot.joint_names]
    meta=dict(schema="blinkmimic_single_motion_v1" if pi.latent_mode=="continuous32" else "blinkmimic_single_motion_v2",joint_names=list(robot.joint_names),
              q0=robot.default_joint_pos_list(),scale=robot.action_scale_list(),kp=kp,kd=kd,
              effort=[x[0] for x in motors],velocity=[x[1] for x in motors],
              lower=kin.q_lo.tolist(),upper=kin.q_hi.tolist(),history=10,goal_frames=10,
              obs_dim=99,goal_dim=730,latent_dim=pi.latent_spec["latent_dim"],control_dt=CONTROL_DT,obs_layout=OBS_LAYOUT,
              checkpoint_iteration=int(ck["iteration"]),checkpoint_sha256=digest(a.ckpt),
              source_motion_sha256=digest(a.motion),source_motion=str(a.motion.resolve()),
              sha256={n:digest(a.out/n) for n in files},
              stages=STAGES,limits=LIMITS,preparation_gain_scale=a.preparation_gain_scale)
    if pi.latent_mode != "continuous32":
        meta["latent_spec"]=pi.latent_spec
    (a.out/"deploy_config.json").write_text(json.dumps(meta,indent=2)+"\n")
    b=Bundle(a.out)
    ort=OnnxPolicy(b)
    enc_ort=dec_ort=None
    if pi.latent_mode != "continuous32":
        import onnxruntime as ort_runtime
        options=ort_runtime.SessionOptions();options.intra_op_num_threads=1
        enc_ort=ort_runtime.InferenceSession(str(a.out/"motion_encoder.onnx"),options,providers=["CPUExecutionProvider"])
        dec_ort=ort_runtime.InferenceSession(str(a.out/"decoder.onnx"),options,providers=["CPUExecutionProvider"])
    split_err=0.
    rng=np.random.default_rng(0)
    err=0.0
    with torch.no_grad():
        for f in np.linspace(0,motion.num_frames-1,64).astype(int):
            h=pi.obs_norm.mean + pi.obs_norm.std()*torch.from_numpy(rng.standard_normal((1,10,99)).astype(np.float32))
            for g in (goals[f],torch.stack(static)[f%2]):
                ref=wrapped(h,g[None])[0].numpy()
                err=max(err,float(abs(ref-ort(h[0].numpy(),g.numpy())).max()))
                if enc_ort is not None:
                    tokens=enc_ort.run(None,{"goal":g[None].numpy()})[0]
                    action=dec_ort.run(None,{"obs_history":h.numpy(),"tokens":tokens})[0][0]
                    split_err=max(split_err,float(abs(action-ref).max()))
    report=dict(max_action_error=err,tolerance=1e-3,max_split_action_error=split_err,passed=max(err,split_err)<=1e-3,bundle_fingerprint=b.fingerprint)
    (a.out/"export_validation.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))
    if not report["passed"]:
        raise RuntimeError("ONNX parity failed; bundle not accepted")
if __name__=="__main__":
    main()
