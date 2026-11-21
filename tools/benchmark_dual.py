"""Explicit single/dual fixed-work benchmark. Running this tool launches both short runs."""
import argparse,csv,json,os,subprocess,sys
from pathlib import Path

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--latent-mode",default="fsq64",choices=["continuous32","continuous64","fsq64"])
    ap.add_argument("--observation-mode",choices=["legacy99","proprio93"],default="legacy99")
    ap.add_argument("--motion-dir",type=Path,default=Path("data/lafan1_g1"))
    ap.add_argument("--warmup-iterations",type=int,default=50)
    ap.add_argument("--measure-iterations",type=int,default=200)
    ap.add_argument("--num-envs",type=int,default=4096)
    ap.add_argument("--dry-run",action="store_true")
    a=ap.parse_args()
    if a.warmup_iterations<=0 or a.warmup_iterations%50 or a.measure_iterations<=0 or a.measure_iterations%50:
        ap.error("warmup and measurement must be positive multiples of 50")
    devices=os.environ.get("CUDA_VISIBLE_DEVICES","0,1").split(",")
    if len(devices)!=2:ap.error("set CUDA_VISIBLE_DEVICES to exactly two GPUs")
    commands=[];total=a.warmup_iterations+a.measure_iterations
    for n in (1,2):
        out=a.out/f"gpu{n}"
        cmd=[sys.executable,"scripts/train_latent.py","--gpus",str(n),"--latent-mode",a.latent_mode,"--observation-mode",a.observation_mode,
             "--motion-dir",str(a.motion_dir),"--log-dir",str(out),"--seed","0",
             "--num_envs",str(a.num_envs),"--iterations",str(total),"--log_every","50",
             "--ckpt_every",str(total),"--ckpt_minutes","0"]
        commands.append((n,out,cmd))
        print(" ".join(cmd),flush=True)
    if a.dry_run:return
    summary={}
    for n,out,cmd in commands:
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=",".join(devices[:n]))
        subprocess.run(cmd,env=env,check=True)
        rows=list(csv.DictReader((out/"train_log.csv").open()))
        lo=next(r for r in rows if int(r["iteration"])==a.warmup_iterations)
        hi=next(r for r in rows if int(r["iteration"])==total)
        seconds=(float(hi["wall_h"])-float(lo["wall_h"]))*3600
        summary[str(n)]=dict(wall_s=seconds,iterations=a.measure_iterations,
            iterations_per_second=a.measure_iterations/seconds,
            global_samples=int(hi["sim_steps"])-int(lo["sim_steps"]),
            global_samples_per_second=(int(hi["sim_steps"])-int(lo["sim_steps"]))/seconds,
            excluded_warmup_wall_s=float(lo["wall_h"])*3600,
            log_dir=str(out),note="same global batches; first 50 iterations excluded; default probe start remains 1000")
        runtime=out/("distributed_runtime.json" if n==2 else "runtime_stats.json")
        if runtime.exists():summary[str(n)]["runtime"]=json.loads(runtime.read_text())
    summary["speedup"]=summary["1"]["wall_s"]/summary["2"]["wall_s"]
    summary["faster"]=summary["speedup"]>1.
    (a.out/"comparison.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary,indent=2))

if __name__=="__main__":main()
