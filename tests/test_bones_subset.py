import csv
import importlib.util
import json
from pathlib import Path

import pytest

spec=importlib.util.spec_from_file_location("bones_subset",Path(__file__).resolve().parents[1]/"tools/build_bones_subset.py")
builder=importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


def fixture(tmp_path):
    fields=["actor_uid","take_date","take_org_name","take_name","move_g1_path","move_duration_frames",
            "is_mirror","content_props","category","package"]
    rows=[]
    source=tmp_path/"extracted"
    for i in range(100):
        actor=f"A{i:03}"
        for cut in (0,1):
            for mirror in (False,True):
                relative=f"g1/csv/240101/motion{i}_cut{cut}"+("_M" if mirror else "")+".csv"
                p=source/relative
                p.parent.mkdir(parents=True,exist_ok=True)
                p.write_text("Frame,value\n"+"".join(f"{f},{i*10+cut}\n" for f in range(241)))
                rows.append(dict(actor_uid=actor,take_date="240101",take_org_name=f"take{i}",
                    take_name=f"take{i}_cut{cut}",move_g1_path=relative,move_duration_frames="241",
                    is_mirror=str(mirror),content_props="0",category="walk" if i%2 else "gesture",package="test"))
    metadata=tmp_path/"metadata.csv"
    with metadata.open("w",newline="") as f:
        w=csv.DictWriter(f,fields);w.writeheader();w.writerows(rows)
    revision=tmp_path/"REVISION.txt";revision.write_text("testrevision\n")
    seen=tmp_path/"seen.json"
    seen.write_text(json.dumps({"clips":[{"source":rows[0]["move_g1_path"]}]}))
    return metadata,source,revision,seen


def test_deterministic_leakage_free_and_seen_quarantine(tmp_path):
    args=fixture(tmp_path)
    counts={"train":5,"val":3,"test":3}
    outputs=[tmp_path/"a",tmp_path/"b"]
    for out in outputs:
        audit=builder.build(*args,out,counts=counts)
        assert audit["catalog_source_cross_split"]==audit["catalog_mirror_cross_split"]==0
        assert json.loads((out/"group_assignments.json").read_text())["A000"]=="train"
    for split,count in counts.items():
        assert (outputs[0]/f"{split}.jsonl").read_bytes()==(outputs[1]/f"{split}.jsonl").read_bytes()
        rows=[json.loads(l) for l in (outputs[0]/f"{split}.jsonl").read_text().splitlines()]
        assert len(rows)==count
        assert len({r["source_group_id"] for r in rows})==count
        assert all(r["metadata"]["is_mirror"]=="False" for r in rows)
        assert all(p.is_symlink() and p.is_file() for p in (outputs[0]/split).glob("*.csv"))


def test_refuses_overwrite(tmp_path):
    args=fixture(tmp_path)
    out=tmp_path/"out";out.mkdir()
    with pytest.raises(FileExistsError):
        builder.build(*args,out)


def test_inadequate_candidates_leaves_no_partial_output(tmp_path):
    args=fixture(tmp_path)
    out=tmp_path/"out"
    with pytest.raises(ValueError,match="requested"):
        builder.build(*args,out,counts={"train":10000,"val":1,"test":1})
    assert not out.exists()
