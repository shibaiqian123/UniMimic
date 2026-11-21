"""Build a separate LAFAN1 + custom G1 CSV training set without editing the originals."""
import argparse,hashlib,json,shutil
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--lafan-dir",type=Path,default=ROOT/"data/lafan1_g1")
    ap.add_argument("--custom",type=Path,nargs="+",default=[Path("/home/cqu023/SuperMimic-Repro-Sim2Real/motions/custom_dance/shanshenbu.csv")])
    ap.add_argument("--out",type=Path,default=ROOT/"data/lafan1_plus_shanshenbu")
    a=ap.parse_args()
    if a.out.exists():ap.error("output already exists; original data will not be overwritten")
    originals=sorted(a.lafan_dir.glob("*.csv"))
    if len(originals)!=40:ap.error(f"expected original 40 LAFAN1 clips, found {len(originals)}")
    inputs=[(p,"lafan1") for p in originals]+[(p,"nju_custom_video_retarget") for p in a.custom]
    if len({p.name for p,_ in inputs})!=len(inputs):ap.error("duplicate CSV filenames")
    entries=[]
    for p,origin in inputs:
        x=np.loadtxt(p,delimiter=",")
        if x.ndim!=2 or x.shape[1]!=36 or len(x)<2 or not np.isfinite(x).all():
            raise ValueError(f"invalid 30Hz G1 CSV: {p}")
        if not np.allclose(np.linalg.norm(x[:,3:7],axis=1),1.,atol=1e-3):
            raise ValueError(f"invalid xyzw quaternion: {p}")
        entries.append(dict(name=p.name,source=str(p.resolve()),origin=origin,sha256=digest(p),
                            frames=len(x),source_fps=30,span_seconds=(len(x)-1)/30))
    a.out.mkdir(parents=True)
    for (p,_),entry in zip(inputs,entries):
        shutil.copy2(p,a.out/p.name)
        if digest(a.out/p.name)!=entry["sha256"]:raise RuntimeError("copy checksum mismatch")
    manifest=dict(schema="blinkmimic_mixed_csv_v1",name=a.out.name,
        clip_count=len(entries),lafan_count=40,custom_count=len(a.custom),
        format="30Hz, no header, root_xyz_m + root_quaternion_xyzw + 29 motor-order joint angles_rad",
        training_split=f"all {len(entries)} clips are training inputs; custom clips are NOT held out",
        sampling="existing BlinkMimic adaptive sampler; no duplication or custom oversampling",
        files=entries)
    (a.out/"dataset.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n")
    (a.out/"SHA256SUMS").write_text("".join(e["sha256"]+"  "+e["name"]+"\n" for e in entries))
    (a.out/"README.txt").write_text(
        f"LAFAN1 40 + {len(a.custom)} custom clips; {len(entries)} total.\n"
        "Original sources unchanged; byte-for-byte copies, no trimming or smoothing.\n"
        "Source CSV: 30Hz; BlinkMimic MotionLib resamples to 50Hz.\n"
        "Custom clips are video-retargeted robot trajectories, not raw mocap.\n"
        "All clips are training inputs; combo overlaps individual clips, so it is not an independent held-out test.\n"
        "Source docs report a left-knee velocity spike in the original gouxiong_duosuomao; this copy preserves it.\n"
        "Data preparation does not establish tracking or real-robot performance.\n"
        "Provenance and rights: SOURCE_CUSTOM_MOTIONS.md.\n")
    notice=a.custom[0].parents[2]/"docs/CUSTOM_MOTIONS.md"
    if notice.exists():shutil.copy2(notice,a.out/"SOURCE_CUSTOM_MOTIONS.md")
    sidecars=[]
    for source in a.custom:
        sidecar=source.with_suffix(".json")
        if sidecar.exists():
            shutil.copy2(sidecar,a.out/sidecar.name)
            sidecars.append(dict(name=sidecar.name,sha256=digest(sidecar)))
    manifest["sidecars"]=sidecars
    (a.out/"dataset.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n")
    with (a.out/"SHA256SUMS").open("a") as f:
        for item in sidecars:f.write(item["sha256"]+"  "+item["name"]+"\n")
    print(json.dumps(dict(path=str(a.out.resolve()),clips=len(entries),custom=entries[40:]),ensure_ascii=False,indent=2))
if __name__=="__main__":main()
