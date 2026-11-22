"""Expand the frozen BONES split to all train-side originals and mirrors."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import shutil
import tempfile


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    a=ap.parse_args()
    base=a.root.resolve()
    frozen=base/"subsets/bones1000_v1_s0"
    config=json.loads((frozen/"config.json").read_text())
    assignment=json.loads((frozen/"group_assignments.json").read_text())
    data=(base/"raw/metadata/seed_metadata_v004.csv").read_bytes()
    if hashlib.sha256(data).hexdigest()!=config["metadata_sha256"]:
        raise ValueError("Metadata differs from the frozen split")
    if (base/"REVISION.txt").read_text().strip()!=config["revision"]:
        raise ValueError("Dataset revision differs from frozen split")
    if a.out.exists():raise FileExistsError(a.out)
    rows=list(csv.DictReader(io.StringIO(data.decode("utf-8-sig"))))
    names=set();source_splits=defaultdict(set);mirror_splits=defaultdict(set)
    stats={s:dict(clips=0,frames50=0,hours=0.,original=0,mirror=0) for s in ("train","val","test")}
    for r in rows:
        path=Path(r["move_g1_path"])
        if path.is_absolute() or ".." in path.parts or str(path) in names:
            raise ValueError("Unsafe or duplicate source path: "+str(path))
        names.add(str(path))
        split=assignment[r["actor_uid"]]
        source_splits[(r["take_date"],r["actor_uid"],r["take_org_name"])].add(split)
        canonical=str(path)[:-6]+".csv" if str(path).endswith("_M.csv") else str(path)
        mirror_splits[canonical].add(split)
        n=int(r["move_duration_frames"])
        frames=int(math.floor((n-1)/120/.02))+1
        if frames<2:raise ValueError("Too few motion frames: "+str(path))
        if not (base/"extracted"/path).is_file():raise FileNotFoundError(base/"extracted"/path)
        x=stats[split];x["clips"]+=1;x["frames50"]+=frames;x["hours"]+=(n-1)/120/3600
        x["mirror" if r["is_mirror"]=="True" else "original"]+=1
    if any(len(x)>1 for x in source_splits.values()) or any(len(x)>1 for x in mirror_splits.values()):
        raise ValueError("Source/mirror leakage")
    a.out.parent.mkdir(parents=True,exist_ok=True)
    stage=Path(tempfile.mkdtemp(prefix=a.out.name+".building-",dir=a.out.parent))
    try:
        (stage/"train").mkdir()
        files={s:(stage/(s+".catalog.jsonl")).open("w") for s in stats}
        link_names=set()
        try:
            for r in rows:
                rel=Path(r["move_g1_path"]);split=assignment[r["actor_uid"]]
                link=r["take_date"]+"__"+rel.name
                if len(link.encode())>240:link=r["take_date"]+"__"+hashlib.sha256(str(rel).encode()).hexdigest()[:24]+".csv"
                record=dict(motion_id=str(rel.with_suffix("")),relative_path=str(rel),split=split,
                    split_group_id=r["actor_uid"],source_group_id="|".join((r["take_date"],r["actor_uid"],r["take_org_name"])),
                    link_name=link,source_fps=120,metadata=r)
                files[split].write(json.dumps(record,ensure_ascii=False)+"\n")
                if split=="train":
                    if link in link_names:raise ValueError("link collision: "+link)
                    link_names.add(link)
                    (stage/"train"/link).symlink_to(base/"extracted"/rel)
        finally:
            for f in files.values():f.close()
        # Preserve the exact 100-clip evaluation sets, not newly sampled holdouts.
        for split in ("val","test"):
            (stage/split).symlink_to(frozen/split,target_is_directory=True)
            shutil.copyfile(frozen/(split+".jsonl"),stage/(split+".jsonl"))
        shutil.copyfile(frozen/"group_assignments.json",stage/"group_assignments.json")
        report=dict(schema="bones_full_train_v1",revision=config["revision"],metadata_sha256=config["metadata_sha256"],
                    frozen_split=str(frozen),split_groups="unchanged actor-disjoint assignments, including previous15 quarantine",
                    stats=stats,source_cross_split=0,mirror_cross_split=0,
                    evaluation="val/test retain the original fixed 100 clips each; full reserved catalogs are not used in training",
                    filters="No props/duration/category/mirror exclusions; requires existing files and >=2 resampled frames",
                    quality="Manifest eligibility only; no per-clip dynamics or physical feasibility certification")
        (stage/"config.json").write_text(json.dumps(report,indent=2))
        stage.rename(a.out)
    except BaseException:
        shutil.rmtree(stage);raise
    print(json.dumps(report,indent=2),flush=True)


if __name__=="__main__":main()
