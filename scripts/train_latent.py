"""Matched continuous64 / FSQ64 recipes; no implicit training launch on import."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--latent-mode",required=True,choices=["continuous64","fsq64"])
    ap.add_argument("--log-dir",required=True,type=Path)
    ap.add_argument("--motion-dir",type=Path,default=ROOT/"data/lafan1_g1")
    ap.add_argument("--seed",type=int,default=0)
    ap.add_argument("--resume",type=Path)
    ap.add_argument("--dry-run",action="store_true")
    a,extra=ap.parse_known_args()
    # Preserve the released recipe except for the deliberately removed goal bypass.
    recipe=subprocess.check_output(["bash","-c",'source scripts/presets.sh; printf "%s" "${PRESET[main]}"'],cwd=ROOT,text=True)
    flags=[x for x in shlex.split(recipe) if x!="--pi_ref_skip"]
    forbidden=("--pi_ref_skip","--latent-mode","--latent_mode","--motion_dir","--log_dir","--seed","--resume")
    if any(x.split("=")[0] in forbidden for x in extra):
        ap.error("use wrapper options for mode/data/log/seed/resume; reference skip is forbidden")
    paths=sorted(a.motion_dir.resolve().glob("*.csv"))
    if len(paths)!=40:ap.error(f"matched LAFAN1 recipe requires 40 CSV clips, found {len(paths)}")
    files=[dict(name=p.name,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths]
    cmd=[sys.executable,"-u","-m","blinkmimic.train",*flags,
         "--latent-mode",a.latent_mode,"--motion_dir",str(a.motion_dir.resolve()),
         "--log_dir",str(a.log_dir.resolve()),"--seed",str(a.seed),*extra]
    if a.resume:cmd+=["--resume",str(a.resume.resolve())]
    manifest=dict(protocol="lafan40_matched_v1",mode=a.latent_mode,seed=a.seed,
                  split="all 40 training clips; evaluation on the same clips is in-distribution",
                  files=files,command=cmd)
    print(shlex.join(cmd),flush=True)
    if a.dry_run:return
    a.log_dir.mkdir(parents=True,exist_ok=True)
    dest=a.log_dir/"experiment.json"
    if dest.exists():
        old=json.loads(dest.read_text())
        if not a.resume or any(old[k]!=manifest[k] for k in ("protocol","mode","seed","files")):
            ap.error("existing experiment requires matching resume; use a new output directory")
    elif (a.log_dir/"last.pt").exists():
        ap.error("existing checkpoint without experiment manifest; use a new output directory")
    dest.write_text(json.dumps(manifest,indent=2)+"\n")
    env=dict(os.environ,PYTHONPATH=str(ROOT/"src")+os.pathsep+os.environ.get("PYTHONPATH",""))
    env.setdefault("OMP_NUM_THREADS","4")
    with (a.log_dir/"stdout.log").open("a") as log:
        proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        for line in proc.stdout:
            print(line,end="",flush=True);log.write(line);log.flush()
        raise SystemExit(proc.wait())

if __name__=="__main__":main()
