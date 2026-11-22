"""Build deterministic BONES manifests and flat CSV symlink views; no training."""
import argparse
from collections import Counter, defaultdict
import csv
import io
import hashlib
import json
import math
from pathlib import Path
import shutil
import tempfile


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def actor_split(actor, seed, seen_actors):
    if actor in seen_actors:
        return "train"
    bucket = int(digest(f"bones-actor-v1:{seed}:{actor}")[:16], 16) % 10000
    return "train" if bucket < 8000 else ("val" if bucket < 9000 else "test")


def eligible(row):
    return (row["is_mirror"].lower() == "false"
            and row["content_props"].strip() == "0"
            and 2 <= (int(row["move_duration_frames"])-1)/120 <= 20)


def ranked_candidates(rows, seed):
    """Square-root category weighting: diversify without equally oversampling rare labels."""
    pools = defaultdict(list)
    for row in rows:
        pools[row["category"]].append(row)
    for group in pools.values():
        group.sort(key=lambda r: digest(f'{seed}:{r["move_g1_path"]}'))
    used = Counter()
    while pools:
        category = min(pools, key=lambda k: (used[k]/math.sqrt(len(pools[k])+used[k]), k))
        row = pools[category].pop(0)
        used[category] += 1
        if not pools[category]:
            del pools[category]
        yield row


def build(metadata, extracted, revision_file, seen_manifest, out, seed=0, counts=None):
    counts = counts or {"train": 1000, "val": 100, "test": 100}
    if out.exists():
        raise FileExistsError(f"Refusing to replace existing subset: {out}")
    metadata_bytes = metadata.read_bytes()
    rows = list(csv.DictReader(io.StringIO(metadata_bytes.decode("utf-8-sig"))))
    by_path = {}
    for row in rows:
        path = row["move_g1_path"]
        if path in by_path:
            raise ValueError(f"Duplicate metadata path: {path}")
        if not row["actor_uid"] or not row["take_date"] or not row["take_org_name"]:
            raise ValueError(f"Missing source identity: {path}")
        by_path[path] = row
    seen_actors = set()
    if seen_manifest:
        for clip in json.loads(seen_manifest.read_text())["clips"]:
            seen_actors.add(by_path[clip["source"]]["actor_uid"])
    assignments = {a: actor_split(a, seed, seen_actors) for a in sorted({r["actor_uid"] for r in rows})}
    # Audit the entire catalog, not just selected originals. Mirrors and any
    # slices retaining their actor ID cannot cross this actor-disjoint boundary.
    source_splits = defaultdict(set)
    mirror_splits = defaultdict(set)
    for row in rows:
        split = assignments[row["actor_uid"]]
        source = (row["take_date"], row["actor_uid"], row["take_org_name"])
        source_splits[source].add(split)
        path = row["move_g1_path"]
        canonical = path[:-6]+".csv" if path.endswith("_M.csv") else path
        mirror_splits[canonical].add(split)
    if any(len(v)!=1 for v in source_splits.values()) or any(len(v)!=1 for v in mirror_splits.values()):
        raise ValueError("Catalog source/mirror leakage")
    candidates = {s: [] for s in counts}
    for row in rows:
        if eligible(row):
            candidates[assignments[row["actor_uid"]]].append(row)
    selected, rejected, all_hashes, all_sources = {}, Counter(), {}, set()
    base = extracted.resolve()
    for split, count in counts.items():
        chosen = []
        for row in ranked_candidates(candidates[split], seed):
            rel = Path(row["move_g1_path"])
            if rel.is_absolute() or ".." in rel.parts:
                raise ValueError(f"Unsafe source path: {rel}")
            path = base / rel
            if not path.is_file():
                rejected["missing_file"] += 1
                continue
            source_key = (row["take_date"], row["actor_uid"], row["take_org_name"])
            # One representative per source take in this first release.
            if source_key in all_sources:
                rejected["same_source_take"] += 1
                continue
            data = path.read_bytes()
            sha = hashlib.sha256(data).hexdigest()
            if sha in all_hashes:
                rejected["duplicate_content"] += 1
                continue
            n = sum(bool(line.strip()) for line in data.splitlines())-1
            if n != int(row["move_duration_frames"]):
                rejected["metadata_frame_mismatch"] += 1
                continue
            link = row["take_date"]+"__"+rel.name
            if len(link.encode())>240:
                link = row["take_date"]+"__"+digest(str(rel))[:24]+".csv"
            chosen.append(dict(motion_id=str(rel.with_suffix("")), split=split,
                               source_group_id="|".join(source_key), split_group_id=row["actor_uid"],
                               relative_path=str(rel), link_name=link, sha256=sha, source_frames=n,
                               source_fps=120, duration_s=(n-1)/120, metadata=dict(row)))
            all_hashes[sha] = split
            all_sources.add(source_key)
            if len(chosen)==count:
                break
        if len(chosen)!=count:
            raise ValueError(f"{split}: only {len(chosen)} eligible sources, requested {count}")
        if len({r["link_name"] for r in chosen})!=len(chosen):
            raise ValueError("Symlink name collision")
        selected[split] = chosen
    revision = revision_file.read_text().strip()
    if not revision:
        raise ValueError("Empty dataset revision")
    config = dict(schema="bones_subset_v1", seed=seed, dataset="bones-studio/seed", revision=revision,
                  metadata_file=str(metadata.resolve()),metadata_sha256=hashlib.sha256(metadata_bytes).hexdigest(),
                  source_root=str(base), source_fps=120, target_fps=50, requested_counts=counts,
                  split_rule="actor-disjoint SHA256 80/10/10; previous 15-clip actors forced to train",
                  seen_manifest=str(seen_manifest.resolve()) if seen_manifest else None,
                  seen_manifest_sha256=hashlib.sha256(seen_manifest.read_bytes()).hexdigest() if seen_manifest else None,
                  seen_actors=sorted(seen_actors), filters=dict(props="0",mirror=False,min_seconds=2,max_seconds=20),
                  category_sampling="square-root category weighting; deterministic hashed order",
                  source_sampling="at most one selected clip per date/actor/original take",
                  pairing_status="SOMA paths preserved verbatim; SMPL correspondence not yet verified")
    audit = dict(catalog_rows=len(rows),catalog_source_groups=len(source_splits),
                 catalog_mirror_groups=len(mirror_splits),catalog_source_cross_split=0,
                 catalog_mirror_cross_split=0,selected_actor_cross_split=0,
                 selected_source_cross_split=0,selected_sha256_duplicates=0,
                 rejected_candidates=dict(rejected),split_summary={})
    for split, clips in selected.items():
        audit["split_summary"][split] = dict(clips=len(clips),actors=len({r["split_group_id"] for r in clips}),
            duration_minutes=sum(r["duration_s"] for r in clips)/60,
            categories=dict(sorted(Counter(r["metadata"]["category"] for r in clips).items())),
            packages=dict(sorted(Counter(r["metadata"]["package"] for r in clips).items())))
    # Assert actual selections as well as catalog-level assignments.
    for i,a in enumerate(counts):
        for b in list(counts)[i+1:]:
            assert not {r["split_group_id"] for r in selected[a]} & {r["split_group_id"] for r in selected[b]}
            assert not {r["source_group_id"] for r in selected[a]} & {r["source_group_id"] for r in selected[b]}
            assert not {r["sha256"] for r in selected[a]} & {r["sha256"] for r in selected[b]}
    out.parent.mkdir(parents=True,exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=out.name+".building-",dir=out.parent))
    try:
        for split, clips in selected.items():
            view = staging / split
            view.mkdir()
            for clip in clips:
                (view / clip["link_name"]).symlink_to(base / clip["relative_path"])
            (staging/(split+".jsonl")).write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in clips))
        (staging/"config.json").write_text(json.dumps(config,ensure_ascii=False,indent=2))
        (staging/"group_assignments.json").write_text(json.dumps(assignments,indent=2))
        audit["manifest_sha256"] = {s:hashlib.sha256((staging/(s+".jsonl")).read_bytes()).hexdigest() for s in counts}
        (staging/"audit.json").write_text(json.dumps(audit,ensure_ascii=False,indent=2))
        staging.rename(out)
    except BaseException:
        shutil.rmtree(staging)
        raise
    return audit


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--metadata",type=Path,required=True)
    ap.add_argument("--extracted",type=Path,required=True)
    ap.add_argument("--revision-file",type=Path,required=True)
    ap.add_argument("--seen-manifest",type=Path)
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--seed",type=int,default=0)
    a=ap.parse_args()
    result=build(a.metadata,a.extracted,a.revision_file,a.seen_manifest,a.out,a.seed)
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)


if __name__=="__main__":
    main()
