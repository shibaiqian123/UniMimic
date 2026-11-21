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
from blinkmimic.rotations import mat_to_6d, yaw_mat, so3_exp
from blinkmimic.g1.g1_constants import motor_of
from blinkmimic.deploy.core import digest, OBS_LAYOUT, PROPRIO_LAYOUT, Bundle, OnnxPolicy, STAGES, LIMITS
from export_onnx import pd_gains

class ExportPolicy(nn.Module):
    def __init__(self, policy):
        super().__init__()
        self.policy = policy
    def forward(self, obs_history, goal, orientation6d_future=None):
        return self.policy(obs_history,goal,orientation6d_future=orientation6d_future)

class ExportEncoder(nn.Module):
    def __init__(self, policy):
        super().__init__(); self.policy=policy
    def forward(self, goal, orientation6d_future=None):
        return self.policy.encode_tokens(goal,orientation6d_future)

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
    proprio93=pi.observation_mode=="proprio93"
    O=93 if proprio93 else 99
    frames=torch.arange(motion.num_frames)
    goals=motion.goal(frames,10)
    static=[]
    for f in (0,motion.num_frames-1):
        R=motion.root_R[f]
        rot=mat_to_6d(yaw_mat(R).T@R)
        one=torch.cat((torch.zeros(3),rot,torch.zeros(6),motion.q[f],torch.zeros(29)))
        static.append(one.repeat(10))
    a.out.mkdir(parents=True)
    orientation=mat_to_6d(motion.root_R[0].T @ motion.root_R[motion.future_frames(torch.zeros(1,dtype=torch.long),10)])
    h0=torch.zeros(1,10,O)
    pinputs=(h0,goals[:1],orientation) if proprio93 else (h0,goals[:1])
    pnames=["obs_history","goal"]+(["orientation6d_future"] if proprio93 else [])
    torch.onnx.export(wrapped,pinputs,str(a.out/"policy.onnx"),
                      input_names=pnames,output_names=["actions"],
                      dynamic_axes={n:{0:"batch"} for n in pnames+["actions"]},opset_version=18,dynamo=False)
    files=["policy.onnx","motion.npz"]
    if pi.latent_mode != "continuous32":
        einputs=(goals[:1],orientation) if proprio93 else (goals[:1],)
        enames=["goal"]+(["orientation6d_future"] if proprio93 else [])
        tokens=pi.encode_tokens(goals[:1],orientation if proprio93 else None).detach()
        for name,model,inputs,names,output in (
            ("motion_encoder.onnx",ExportEncoder(pi),einputs,enames,"tokens"),
            ("decoder.onnx",ExportDecoder(pi),(h0,tokens),["obs_history","tokens"],"actions"),
        ):
            torch.onnx.export(model.eval(),inputs,str(a.out/name),input_names=names,
                              output_names=[output],dynamic_axes={n:{0:"batch"} for n in names+[output]},
                              opset_version=18,dynamo=False)
            files.append(name)
    np.savez_compressed(a.out/"motion.npz",q=motion.q.numpy(),root_R=motion.root_R.numpy(),
                        root_p=motion.root_p.numpy(),qd=motion.qd.numpy(),root_v=motion.root_v.numpy(),root_w=motion.root_w.numpy(),goal=goals.numpy(),static_goal=torch.stack(static).numpy())
    kp,kd=pd_gains(robot)
    motors=[motor_of(n) for n in robot.joint_names]
    meta=dict(schema="blinkmimic_single_motion_v3" if proprio93 else ("blinkmimic_single_motion_v1" if pi.latent_mode=="continuous32" else "blinkmimic_single_motion_v2"),joint_names=list(robot.joint_names),
              q0=robot.default_joint_pos_list(),scale=robot.action_scale_list(),kp=kp,kd=kd,
              effort=[x[0] for x in motors],velocity=[x[1] for x in motors],
              lower=kin.q_lo.tolist(),upper=kin.q_hi.tolist(),history=10,goal_frames=10,
              obs_dim=O,goal_dim=730,latent_dim=pi.latent_spec["latent_dim"],control_dt=CONTROL_DT,obs_layout=PROPRIO_LAYOUT if proprio93 else OBS_LAYOUT,
              checkpoint_iteration=int(ck["iteration"]),checkpoint_sha256=digest(a.ckpt),
              source_motion_sha256=digest(a.motion),source_motion=str(a.motion.resolve()),
              sha256={n:digest(a.out/n) for n in files},
              stages=STAGES,limits=LIMITS,preparation_gain_scale=a.preparation_gain_scale)
    if pi.latent_mode != "continuous32":
        meta["latent_spec"]=pi.latent_spec
    if proprio93:meta["observation_spec"]=pi.observation_spec
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
            h=pi.obs_norm.mean+pi.obs_norm.std()*torch.from_numpy(rng.standard_normal((1,10,O)).astype(np.float32))
            robot_R=so3_exp(torch.tensor([.12,-.08,.41])) @ motion.root_R[f]
            for is_static,g in ((False,goals[f]),(True,torch.stack(static)[f%2])):
                endpoint=0 if f%2==0 else motion.num_frames-1
                refs=motion.root_R[endpoint].expand(10,3,3) if is_static else motion.root_R[motion.future_frames(torch.tensor([f]),10)[0]]
                ori=mat_to_6d(robot_R.T@refs)[None] if proprio93 else None
                ref=wrapped(h,g[None],ori)[0].numpy()
                actual=ort(h[0].numpy(),g.numpy(),None if ori is None else ori[0].numpy())
                err=max(err,float(abs(ref-actual).max()))
                if enc_ort is not None:
                    feed={"goal":g[None].numpy()}
                    if proprio93:feed["orientation6d_future"]=ori.numpy()
                    tokens=enc_ort.run(None,feed)[0]
                    action=dec_ort.run(None,{"obs_history":h.numpy(),"tokens":tokens})[0][0]
                    split_err=max(split_err,float(abs(action-ref).max()))
    report=dict(max_action_error=err,tolerance=1e-3,max_split_action_error=split_err,passed=max(err,split_err)<=1e-3,bundle_fingerprint=b.fingerprint)
    (a.out/"export_validation.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))
    if not report["passed"]:
        raise RuntimeError("ONNX parity failed; bundle not accepted")
if __name__=="__main__":
    main()
