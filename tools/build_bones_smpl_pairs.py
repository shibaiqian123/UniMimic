"""Pair official SONIC SMPL with BONES G1, retaining frozen source splits.

Identity rule follows NVlabs/GR00T-WholeBodyControl motion_lib_base.py:
exact motion basename + .pkl, including mirror suffix. No fuzzy matching.
Run only on trusted official PKLs (joblib deserialization).
"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import math


def match_key(record):
    key = Path(record["relative_path"]).stem
    if record["metadata"]["filename"] != key:
        raise ValueError("Metadata identity mismatch: " + key)
    return key


def inspect_pair(item):
    import joblib
    import numpy as np
    key, path, frames, source_fps = item
    result = {"key": key}
    try:
        blob = Path(path).read_bytes()
        result["smpl_sha256"] = hashlib.sha256(blob).hexdigest()
        import io
        data = joblib.load(io.BytesIO(blob))
        fps = float(data["fps"])
        pose = np.asarray(data["pose_aa"])
        joints = np.asarray(data["smpl_joints"])
        transl = np.asarray(data["transl"])
        n = len(pose)
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("Invalid fps")
        if n < 2 or pose.shape != (n, 72) or joints.shape != (n, 24, 3) or transl.shape != (n, 3):
            raise ValueError("Unexpected SMPL shapes")
        if not all(np.isfinite(x).all() for x in (pose, joints, transl)):
            raise ValueError("Nonfinite SMPL")
        g1_end = (frames - 1) / source_fps
        smpl_end = (n - 1) / fps
        delta = smpl_end - g1_end
        original_fps = float(data.get("original_fps", fps))
        original_pose = np.asarray(data.get("original_pose_aa", pose))
        if (not math.isfinite(original_fps) or original_fps <= 0
                or original_pose.ndim != 2 or original_pose.shape[1] != 72
                or len(original_pose) < 2 or not np.isfinite(original_pose).all()):
            raise ValueError("Invalid original SMPL sequence")
        original_end = (len(original_pose)-1) / original_fps
        # Each resampling can discard its last fractional frame, including a
        # floating-point endpoint. Check each stage, not a loose global cutoff.
        stage1 = original_end - g1_end
        stage2 = smpl_end - original_end
        timing_ok = (-1/original_fps-1e-6 <= stage1 <= 1/source_fps+1e-6
                     and -1/fps-1e-6 <= stage2 <= 1e-6)
        result.update(smpl_fps=fps, smpl_frames=n, g1_frames=frames,
                      g1_fps=source_fps, duration_delta_s=delta,
                      common_end_s=min(g1_end, smpl_end),
                      target_fps=50,
                      common_frames50=int(math.floor(min(g1_end, smpl_end)*50+1e-7))+1,
                      original_smpl_fps=original_fps, original_smpl_frames=len(original_pose),
                      original_minus_g1_end_s=stage1, final_minus_original_end_s=stage2,
                      status="matched" if timing_ok else "duration_mismatch")
    except Exception as exc:
        result.update(status="invalid_smpl", error=str(exc))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--workers", type=int, default=8)
    a = p.parse_args()
    root = a.root.resolve()
    catalog = root / "subsets/bones_full_v1_s0"
    records = []
    for split in ("train", "val", "test"):
        records.extend(json.loads(s) for s in (catalog / (split+".catalog.jsonl")).read_text().splitlines())
    keys = defaultdict(list)
    for r in records:
        keys[match_key(r)].append(r)
    smpl = defaultdict(list)
    for f in sorted((root/"smpl_extracted/smpl_filtered").rglob("*.pkl")):
        if f.stem != "metadata":
            smpl[f.stem].append(f)
    a.out.mkdir(parents=True, exist_ok=False)
    status = {}
    jobs = []
    for key, rows in keys.items():
        if len(rows) != 1 or len(smpl.get(key, [])) > 1:
            status[key] = dict(key=key, status="ambiguous_identity")
        elif key not in smpl:
            status[key] = dict(key=key, status="missing_smpl")
        else:
            r = rows[0]
            jobs.append((key, str(smpl[key][0]), int(r["metadata"]["move_duration_frames"]), r["source_fps"]))
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for i, value in enumerate(pool.map(inspect_pair, jobs, chunksize=64), 1):
            status[value["key"]] = value
            if i % 5000 == 0:
                print(f"Validated SMPL pairs: {i}/{len(jobs)}", flush=True)
    counts = {s: Counter() for s in ("train","val","test")}
    groups = defaultdict(set)
    mirrors = defaultdict(set)
    files = {s:(a.out/(s+".jsonl")).open("w") for s in counts}
    rejected = (a.out/"unmatched.jsonl").open("w")
    paired = {}
    try:
        for r in records:
            key = match_key(r)
            result = status[key]
            counts[r["split"]][result["status"]] += 1
            rec = dict(r, pairing=result,
                       smpl_path=str(smpl[key][0].relative_to(root)) if len(smpl.get(key,[]))==1 else None,
                       g1_path="extracted/"+r["relative_path"],
                       mirror=r["metadata"]["is_mirror"]=="True")
            if result["status"] == "matched":
                files[r["split"]].write(json.dumps(rec,ensure_ascii=False)+"\n")
                paired[r["motion_id"]] = rec
                groups[r["source_group_id"]].add(r["split"])
                groups[r["split_group_id"]].add(r["split"])
                mirrors[key.removesuffix("_M")].add(r["split"])
            else:
                rejected.write(json.dumps(rec,ensure_ascii=False)+"\n")
    finally:
        rejected.close()
        for f in files.values(): f.close()
    if any(len(s)>1 for s in [*groups.values(),*mirrors.values()]):
        raise ValueError("Split leakage")
    eval_counts = {}
    for split in ("val","test"):
        fixed = [json.loads(s) for s in (catalog/(split+".jsonl")).read_text().splitlines()]
        selected = [paired[r["motion_id"]] for r in fixed if r["motion_id"] in paired]
        (a.out/(split+"_fixed100.jsonl")).write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in selected))
        eval_counts[split] = {"paired":len(selected),"original":len(fixed),
                             "missing_ids":[r["motion_id"] for r in fixed if r["motion_id"] not in paired]}
    report = dict(schema="bones_g1_smpl_pairs_v2", counts=counts,
        smpl_files=sum(map(len,smpl.values())), g1_records=len(records),
        smpl_without_g1=sorted(set(smpl)-set(keys)),
        fixed_evaluation=eval_counts, source_cross_split=0, mirror_cross_split=0,
        revisions={name:(root/name).read_text().strip() for name in ("REVISION.txt","SMPL_REVISION.txt")},
        matching="Exact metadata filename / G1 stem / SMPL stem, mirror suffix retained; ambiguous names excluded",
        timing="t=0 shared origin; sample at t=k/50 through min(last G1 time,last SMPL time); no time stretching",
        timing_tolerance="Each 120-to-original and original-to-final resampling endpoint checked separately; common support only; trajectory alignment not yet certified",
        scope="Identity, schema, finite arrays and duration validation; NOT spatial/semantic/retargeting validation",
        coordinate_status="SMPL position units, axes, heading, joint order and three-point device mapping require explicit adapter validation",
        training="For joint G1/SMPL/teleop encoders with shared FSQ decoder; no precomputed teacher tokens required",
        sonic_reference="https://github.com/NVlabs/GR00T-WholeBodyControl/blob/main/gear_sonic/utils/motion_lib/motion_lib_base.py")
    (a.out/"report.json").write_text(json.dumps(report,indent=2))
    print(json.dumps({k:report[k] for k in ("counts","smpl_files","g1_records","fixed_evaluation")},indent=2),flush=True)


if __name__ == "__main__":
    main()
