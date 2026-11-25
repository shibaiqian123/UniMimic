"""Generate SONIC three-point reference caches from frozen paired JSONL manifests."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def atomic_json(path, value):
    tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(value,indent=2))
    os.replace(tmp,path)

def init_worker(root,device,out,resume):
    import torch
    from blinkmimic.g1.g1_constants import SPEC
    from blinkmimic.kinematics import Kinematics
    from blinkmimic.threepoint import CONTRACT
    torch.set_num_threads(1)
    global ROOT,OUT,RESUME,KIN,SPECIFICATION,CONFIG
    ROOT,OUT,RESUME=Path(root),Path(out),resume
    SPECIFICATION=SPEC
    KIN=Kinematics(SPEC,device)
    CONFIG=dict(CONTRACT,robot_xml_sha256=sha(SPEC.mjcf_path))
    
def process(record):
    import numpy as np
    from blinkmimic.threepoint import from_csv
    mid=record["motion_id"]
    key=hashlib.sha256(mid.encode()).hexdigest()
    rel=Path("clips")/record["split"]/key[:2]/(key+".npz")
    dest=OUT/rel
    try:
        source=ROOT/record["g1_path"]
        meta=dict(CONFIG,motion_id=mid,split=record["split"],mirror=record["mirror"],
            source_group_id=record["source_group_id"],split_group_id=record["split_group_id"],
            source_path=record["g1_path"],smpl_path=record["smpl_path"],
            source_sha256=sha(source),smpl_sha256=record["pairing"]["smpl_sha256"],
            frames=int(record["pairing"]["common_frames50"]),
            common_end_s=record["pairing"]["common_end_s"])
        # JSON-normalize tuples before comparing resumed metadata.
        meta=json.loads(json.dumps(meta))
        reused=False
        if RESUME and dest.exists():
            try:
                with np.load(dest,allow_pickle=False) as z:
                    n=meta["frames"]
                    shapes={"time_s":(n,),"position_world":(n,3,3),"position_local":(n,3,3),
                            "quaternion_world":(n,3,4),"quaternion_local":(n,3,4),
                            "pelvis_position_world":(n,3),"pelvis_quaternion_world":(n,4)}
                    reused=(json.loads(str(z["metadata"]))==meta and all(
                        z[k].shape==shape and z[k].dtype==np.float32 and np.isfinite(z[k]).all()
                        for k,shape in shapes.items()))
            except Exception: reused=False
        if not reused:
            arrays=from_csv(source,record,SPECIFICATION,KIN)
            if any(not np.isfinite(v).all() for v in arrays.values()):
                raise ValueError("Nonfinite output")
            dest.parent.mkdir(parents=True,exist_ok=True)
            fd,tmp=tempfile.mkstemp(dir=dest.parent,suffix=".tmp")
            try:
                with os.fdopen(fd,"wb") as f:
                    np.savez_compressed(f,**arrays,metadata=np.array(json.dumps(meta)))
                os.replace(tmp,dest)
            finally:
                if os.path.exists(tmp):os.unlink(tmp)
        return dict(motion_id=mid,split=record["split"],source_group_id=record["source_group_id"],
                    split_group_id=record["split_group_id"],mirror=record["mirror"],
                    cache=str(rel),frames=meta["frames"],bytes=dest.stat().st_size,
                    source_sha256=meta["source_sha256"],reused=reused,status="ok")
    except Exception as e:
        return dict(motion_id=mid,split=record["split"],status="failed",error=str(e))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest",type=Path,nargs="+",required=True)
    p.add_argument("--data-root",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--device",default="cpu")
    p.add_argument("--workers",type=int,default=1)
    p.add_argument("--limit",type=int)
    p.add_argument("--resume",action="store_true")
    a=p.parse_args()
    if a.device!="cpu" and a.workers!=1:p.error("CUDA uses one worker")
    if a.workers<1:p.error("workers must be positive")
    if a.out.exists() and not a.resume:p.error("Output exists; use --resume")
    a.out.mkdir(parents=True,exist_ok=True)
    records=[]
    for f in a.manifest:
        records.extend(json.loads(line) for line in f.open() if line.strip())
    if a.limit is not None: records=records[:a.limit]
    ids=[r["motion_id"] for r in records]
    if len(ids)!=len(set(ids)):raise ValueError("Duplicate motion_id in input")
    start=time.monotonic()
    init_args=(str(a.data_root),a.device,str(a.out),a.resume)
    pool=None
    if a.workers==1:
        init_worker(*init_args)
        results=map(process,records)
    else:
        pool=ProcessPoolExecutor(a.workers,initializer=init_worker,initargs=init_args)
        results=pool.map(process,records,chunksize=8)
    rows=[]
    with (a.out/"progress.jsonl").open("w") as progress:
        for i,r in enumerate(results,1):
            rows.append(r);progress.write(json.dumps(r)+"\n")
            if i%1000==0 or i==len(records):
                progress.flush()
                print(f"Three-point: {i}/{len(records)}, elapsed={time.monotonic()-start:.1f}s",flush=True)
    if pool:pool.shutdown()
    ok={r["motion_id"]:r for r in rows if r["status"]=="ok"}
    counts={}
    for split in ("train","val","test"):
        subset=[r for r in rows if r["split"]==split and r["status"]=="ok"]
        counts[split]=len(subset)
        tmp=a.out/(split+".jsonl.tmp")
        tmp.write_text("".join(json.dumps(r)+"\n" for r in subset))
        os.replace(tmp,a.out/(split+".jsonl"))
    fixed={}
    for split in ("val","test"):
        source=a.manifest[0].parent/(split+"_fixed100.jsonl")
        if source.exists():
            expected=[json.loads(s)["motion_id"] for s in source.read_text().splitlines()]
            chosen=[ok[x] for x in expected if x in ok]
            (a.out/(split+"_fixed100.jsonl")).write_text("".join(json.dumps(r)+"\n" for r in chosen))
            fixed[split]=dict(expected=len(expected),generated=len(chosen))
    failures=[r for r in rows if r["status"]!="ok"]
    (a.out/"failures.jsonl").write_text("".join(json.dumps(r)+"\n" for r in failures))
    report=dict(counts=counts,requested=len(records),completed=len(ok),failed=len(failures),
        elapsed_seconds=time.monotonic()-start,device=a.device,workers=a.workers,
        payload_bytes=sum(r["bytes"] for r in ok.values()),reused=sum(r["reused"] for r in ok.values()),
        fixed_evaluation=fixed,manifests={str(f):sha(f) for f in a.manifest},
        scope="Reference three-point geometry only; not VR calibration, policy training or dynamics")
    atomic_json(a.out/"report.json",report)
    print(json.dumps(report,indent=2),flush=True)
    if failures:raise SystemExit(1)

if __name__=="__main__":main()
