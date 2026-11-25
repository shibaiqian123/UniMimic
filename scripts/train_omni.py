"""Launch paired three-modality training, with global batch semantics."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--index",type=Path,required=True)
    p.add_argument("--data-root",type=Path,required=True)
    p.add_argument("--log-dir",type=Path,required=True)
    p.add_argument("--gpus",type=int,choices=(1,2),default=1)
    p.add_argument("--resume",type=Path)
    p.add_argument("--dry-run",action="store_true")
    a,extra=p.parse_known_args()
    from hashlib import sha256
    index=json.loads(a.index.read_text())
    if not index["records"]:p.error("empty paired cache")
    forbidden={"--omni-index","--omni-data-root","--latent-mode","--observation-mode","--pi_ref_skip","--motion_dir","--log_dir","--resume"}
    if any(x.split("=")[0] in forbidden for x in extra):p.error("use dedicated paired wrapper options")
    recipe=subprocess.check_output(["bash","-c",'source scripts/presets.sh; printf "%s" "${PRESET[main]}"'],cwd=ROOT,text=True)
    flags=[x for x in shlex.split(recipe) if x!="--pi_ref_skip"]
    prefix=[sys.executable,"-u"]
    if a.gpus==2:prefix+=["-m","torch.distributed.run","--standalone","--nnodes=1","--nproc-per-node=2"]
    cmd=prefix+["-m","blinkmimic.train"]+flags+[
        "--latent-mode","fsq64","--observation-mode","proprio93",
        "--omni-index",str(a.index.resolve()),"--omni-data-root",str(a.data_root.resolve()),
        "--log_dir",str(a.log_dir.resolve()),"--motion-compact",
        "--eprobe_share","0","--eprobe_async","0","--lr_decay_by","0"]+extra
    if a.resume:cmd+=["--resume",str(a.resume.resolve())]
    print(shlex.join(cmd),flush=True)
    if a.dry_run:return
    marker=dict(index_sha256=sha256(a.index.read_bytes()).hexdigest(),contract=index["contract"],gpus=a.gpus)
    a.log_dir.mkdir(parents=True,exist_ok=True)
    path=a.log_dir/"omni_experiment.json"
    if path.exists():
        if not a.resume or json.loads(path.read_text())!=marker:p.error("existing experiment requires compatible resume")
    elif any(a.log_dir.glob("*.pt")):p.error("output already contains checkpoints")
    path.write_text(json.dumps(marker,indent=2))
    env=dict(os.environ,PYTHONPATH=str(ROOT/"src")+os.pathsep+os.environ.get("PYTHONPATH",""))
    env.setdefault("OMP_NUM_THREADS","4")
    with (a.log_dir/"stdout.log").open("a") as log:
        proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        for line in proc.stdout:print(line,end="",flush=True);log.write(line);log.flush()
        raise SystemExit(proc.wait())
if __name__=="__main__":main()
