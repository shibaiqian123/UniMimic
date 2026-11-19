"""Offline physical tracking evaluation. Never imported by the controller or runner."""
import json
from pathlib import Path
import numpy as np
import mujoco
from .core import Bundle

def evaluate(session,runner_dir,bundle):
    session=Path(session)
    meta=json.loads((session/"session.json").read_text())
    model=mujoco.MjModel.from_xml_path(meta["unitree_xml"])
    free=np.flatnonzero(model.jnt_type==mujoco.mjtJoint.mjJNT_FREE)
    if len(free)!=1:raise ValueError("one floating base required for evaluation")
    fq=int(model.jnt_qposadr[free[0]])
    qi=[model.jnt_qposadr[model.joint(n).id] for n in bundle.cfg["joint_names"]]
    records=[]
    for line in (Path(runner_dir)/"runner.jsonl").open():
        row=json.loads(line)
        if "result" in row and "t" in row:
            records.append((float(row["t"]),row.get("state",""),int(row["result"]["frame"])))
    if not records or not (session/"physics.npz").exists():return {"available":False,"passed":False}
    times=np.array([r[0] for r in records]);states=np.array([r[1] for r in records]);frames=np.array([r[2] for r in records])
    with np.load(session/"physics.npz",allow_pickle=False) as z:
        idx=np.searchsorted(times,z["t"],side="right")-1
        valid=idx>=0;idx=np.maximum(idx,0)
        mask=valid&(states[idx]=="POLICY")
        if not mask.any():return {"available":False,"passed":False}
        f=np.clip(frames[idx[mask]],0,bundle.T-1)
        qpos=z["qpos"][mask];pose0=json.loads((session/"plant.json").read_text())["initial_root"]
        ref_height=float(pose0[2])+bundle.motion["root_p"][f,2]-bundle.motion["root_p"][0,2]
        height=np.abs(qpos[:,fq+2]-ref_height)
        quat=qpos[:,fq+3:fq+7];quat=quat/np.linalg.norm(quat,axis=1,keepdims=True)
        actual_r22=1-2*(quat[:,1]**2+quat[:,2]**2)
        tilt=np.abs(actual_r22-bundle.motion["root_R"][f,2,2])
        violation=(height>.25)|(tilt>.8)
        rmse=float(np.sqrt(np.mean((qpos[:,qi]-bundle.motion["q"][f])**2)))
        damping=valid&(states[idx]=="DAMP")
        stop_info={"available":bool(damping.any()),"passed":False}
        if damping.any():
            stop_pose=z["qpos"][damping]
            dz=np.abs(stop_pose[:,fq+2]-stop_pose[0,fq+2])
            sq=stop_pose[:,fq+3:fq+7];sq=sq/np.linalg.norm(sq,axis=1,keepdims=True)
            tilt_change=np.abs((1-2*(sq[:,1]**2+sq[:,2]**2))-(1-2*(sq[0,1]**2+sq[0,2]**2)))
            supported=bool(z["support"][damping].all())
            stop_info=dict(available=True,supported=supported,max_height_change_m=float(dz.max()),
                           max_tilt_r22_change=float(tilt_change.max()),
                           passed=supported and bool(dz.max()<=.25 and tilt_change.max()<=.8))
        return dict(available=True,passed=not bool(violation.any()),samples=int(mask.sum()),
                    max_height_error_m=float(height.max()),max_tilt_r22_error=float(tilt.max()),
                    joint_tracking_rmse_rad=rmse,violation_samples=int(violation.sum()),
                    first_violation_frame=int(f[np.flatnonzero(violation)[0]]) if violation.any() else None,
                    policy_support_samples=int(z["support"][mask].sum()),
                    stop_stage=stop_info,scope="offline truth evaluator only; cannot stop or change the running controller")
