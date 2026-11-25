"""Build atomic, resumable CPU-mappable paired reference shards (no training)."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import torch
from blinkmimic.omni_contract import CONTRACT
from blinkmimic.omni_features import smpl_reference
from blinkmimic.omni_data import pack
from blinkmimic.motion import resample_motion,_gradient,CONTROL_DT
from blinkmimic.rotations import quat_to_mat

def sha(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""):h.update(chunk)
    return h.hexdigest()

def atomic_json(p,obj):
    t=p.with_suffix(p.suffix+".tmp");t.write_text(json.dumps(obj,indent=2));os.replace(t,p)

def convert(record,root,threepoint,spec):
    import joblib
    n=record["pairing"]["common_frames50"]
    p,quat,q=resample_motion(*spec.load_motion(str(root/record["g1_path"])))
    # Shared-end truncation precedes derivatives, matching paired MotionLib.
    q=q[:n];qd=_gradient(q,CONTROL_DT)
    if len(q)!=n:raise ValueError("Shared duration exceeds G1 source")
    path=root/record["smpl_path"]
    if sha(path)!=record["pairing"]["smpl_sha256"]:raise ValueError("SMPL fingerprint changed")
    smpl=joblib.load(path)
    if float(smpl["fps"])!=50:raise ValueError("Expected matched 50Hz SMPL")
    aa=torch.as_tensor(smpl["pose_aa"][:n],dtype=torch.float32)
    joints=torch.as_tensor(smpl["smpl_joints"][:n],dtype=torch.float32)
    if aa.shape!=(n,72) or joints.shape!=(n,24,3):raise ValueError("SMPL shape mismatch")
    local,R=smpl_reference(aa,joints)
    key=hashlib.sha256(record["motion_id"].encode()).hexdigest()
    tp=threepoint/"clips"/record["split"]/key[:2]/(key+".npz")
    with np.load(tp,allow_pickle=False) as z:
        meta=json.loads(str(z["metadata"]))
        if meta["motion_id"]!=record["motion_id"] or meta["frames"]!=n:raise ValueError("Threepoint identity mismatch")
        if meta["source_sha256"]!=sha(root/record["g1_path"]):raise ValueError("G1 fingerprint changed")
        if meta["smpl_sha256"]!=record["pairing"]["smpl_sha256"]:raise ValueError("Threepoint pairing changed")
        if meta["robot_xml_sha256"]!=sha(spec.mjcf_path):raise ValueError("Robot XML mismatch")
        values=pack(q,qd,quat_to_mat(quat[:n]),local,R,z["position_local"],
                    quat_to_mat(torch.from_numpy(z["quaternion_local"])))
    if values.shape!=(n,184) or not np.isfinite(values).all():raise ValueError("Invalid features")
    return values,dict(g1=meta["source_sha256"],smpl=meta["smpl_sha256"],threepoint=sha(tp))

def main():
    from blinkmimic.g1.g1_constants import SPEC
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest",type=Path,required=True)
    p.add_argument("--data-root",type=Path,required=True)
    p.add_argument("--threepoint-root",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--clips-per-shard",type=int,default=128)
    p.add_argument("--limit",type=int)
    p.add_argument("--resume",action="store_true")
    a=p.parse_args()
    if a.clips_per_shard<1:p.error("clips-per-shard must be positive")
    if a.out.exists() and not a.resume:p.error("Output exists; use --resume")
    torch.set_num_threads(1);a.out.mkdir(parents=True,exist_ok=True)
    records=[json.loads(x) for x in a.manifest.read_text().splitlines() if x.strip()]
    if a.limit is not None:records=records[:a.limit]
    signature=dict(manifest=sha(a.manifest),contract=CONTRACT,xml=sha(SPEC.mjcf_path),
                   clips_per_shard=a.clips_per_shard,limit=a.limit)
    config=a.out/"config.json"
    if config.exists() and json.loads(config.read_text())!=signature:raise ValueError("Resume configuration mismatch")
    atomic_json(config,signature)
    result=[];failures=[]
    for start in range(0,len(records),a.clips_per_shard):
        group=records[start:start+a.clips_per_shard]
        name=f"shard_{start//a.clips_per_shard:06d}.npy"
        dest=a.out/name;meta=dest.with_suffix(".json")
        try:
            if a.resume and dest.exists() and meta.exists():
                cached=json.loads(meta.read_text())
                if cached["sha256"]!=sha(dest):raise ValueError("Corrupt shard")
                if [r["motion_id"] for r in cached["records"]]!=[r["motion_id"] for r in group]:raise ValueError("Shard identities changed")
                for r,c in zip(group,cached["records"]):
                    if sha(a.data_root/r["g1_path"])!=c["fingerprints"]["g1"] or sha(a.data_root/r["smpl_path"])!=c["fingerprints"]["smpl"]:
                        raise ValueError("Source changed")
                    key=hashlib.sha256(r["motion_id"].encode()).hexdigest()
                    tp=a.threepoint_root/"clips"/r["split"]/key[:2]/(key+".npz")
                    if sha(tp)!=c["fingerprints"]["threepoint"]:raise ValueError("Threepoint cache changed")
                result.extend(cached["records"])
            else:
                arrays=[];index=[];offset=0
                for r in group:
                    x,fp=convert(r,a.data_root,a.threepoint_root,SPEC)
                    arrays.append(x)
                    index.append(dict(motion_id=r["motion_id"],split=r["split"],mirror=r["mirror"],
                        source_group_id=r["source_group_id"],split_group_id=r["split_group_id"],
                        g1_path=r["g1_path"],smpl_path=r["smpl_path"],frames=len(x),shard=name,offset=offset,fingerprints=fp))
                    offset+=len(x)
                tmp=dest.with_suffix(".tmp")
                with tmp.open("wb") as f:np.save(f,np.concatenate(arrays),allow_pickle=False)
                os.replace(tmp,dest)
                atomic_json(meta,dict(sha256=sha(dest),records=index))
                result.extend(index)
            print(f"omni cache: {min(start+len(group),len(records))}/{len(records)}",flush=True)
        except Exception as e:
            failures.append(dict(start=start,motion_ids=[r["motion_id"] for r in group],error=str(e)))
            atomic_json(a.out/"failures.json",failures)
            raise
    atomic_json(a.out/"index.json",dict(contract=CONTRACT,records=result,signature=signature))
    atomic_json(a.out/"failures.json",failures)
    print(f"Completed {len(result)} paired clips",flush=True)

if __name__=="__main__":main()
