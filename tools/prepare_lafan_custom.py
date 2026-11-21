"""Build a separate LAFAN1 + custom G1 CSV training set without editing the originals."""
import argparse,hashlib,json,shutil
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--lafan-dir",type=Path,default=ROOT/"data/lafan1_g1")
    ap.add_argument("--custom",type=Path,default=Path("/home/cqu023/SuperMimic-Repro-Sim2Real/motions/custom_dance/shanshenbu.csv"))
    ap.add_argument("--out",type=Path,default=ROOT/"data/lafan1_plus_shanshenbu")
    a=ap.parse_args()
    if a.out.exists():ap.error("output already exists; original data will not be overwritten")
    originals=sorted(a.lafan_dir.glob("*.csv"))
    if len(originals)!=40:ap.error(f"expected original 40 LAFAN1 clips, found {len(originals)}")
    inputs=[(p,"lafan1") for p in originals]+[(a.custom,"nju_custom_video_retarget")]
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
    manifest=dict(schema="blinkmimic_mixed_csv_v1",name="lafan1_plus_shanshenbu",
        clip_count=len(entries),lafan_count=40,custom_count=1,
        format="30Hz, no header, root_xyz_m + root_quaternion_xyzw + 29 motor-order joint angles_rad",
        training_split="all 41 clips; shanshenbu is NOT held out",
        sampling="existing BlinkMimic adaptive sampler; no duplication or custom oversampling",
        files=entries)
    (a.out/"dataset.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n")
    (a.out/"SHA256SUMS").write_text("".join(e["sha256"]+"  "+e["name"]+"\n" for e in entries))
    (a.out/"README.txt").write_text(
        "LAFAN1 40 + shanshenbu.csv (216 source frames, about 7.2 seconds).\n"
        "Original source directories are unchanged. Files copied byte-for-byte; no concatenation, trimming or resampling here.\n"
        "BlinkMimic MotionLib resamples the 30Hz source to its 50Hz control reference.\n"
        "shanshenbu is a video-retargeted robot trajectory, not raw motion capture.\n"
        "Source: NJU-R-L-Group-Embodied-Lab/SuperMimic-Repro-Sim2Real, motions/custom_dance/shanshenbu.csv.\n"
        "All 41 clips are training inputs; testing shanshenbu after training measures learned-motion deployment, not unseen-motion generalization.\n"
        "No training or real-robot validation is implied by this dataset preparation.\n"
        "Source provenance and rights notice: SOURCE_CUSTOM_MOTIONS.md.\n")
    notice=a.custom.parents[2]/"docs/CUSTOM_MOTIONS.md"
    if notice.exists():shutil.copy2(notice,a.out/"SOURCE_CUSTOM_MOTIONS.md")
    print(json.dumps(dict(path=str(a.out.resolve()),clips=len(entries),custom=entries[-1]),ensure_ascii=False,indent=2))
if __name__=="__main__":main()
