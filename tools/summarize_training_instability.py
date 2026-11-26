"""Summarize recorded losses near a suspected instability; no causal attribution."""
import argparse
import csv
import json
from pathlib import Path
from statistics import median


def summarize(rows, intervals):
    names = ("wm_err8", "v_loss", "pi_gradnorm", "sim_rew", "imag_rew",
             "ep_ret_norm", "pi_lr", "nonfinite")
    result = {}
    for label, start, end in intervals:
        selected = [r for r in rows if start <= int(r["iteration"]) <= end]
        result[label] = dict(start=start, end=end, samples=len(selected), metrics={})
        for name in names:
            values = [float(r[name]) for r in selected]
            if values:
                result[label]["metrics"][name] = dict(median=median(values), min=min(values), max=max(values))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("csv", type=Path)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    with a.csv.open() as f:
        rows = list(csv.DictReader(f))
    report = dict(source=str(a.csv.resolve()),
                  windows=summarize(rows, [("baseline",16000,17000),("onset",17100,17750),
                                           ("collapse",17800,18100),("late",19000,19400)]),
                  selected=[r for r in rows if int(r["iteration"]) in (17000,17450,17800,17850,18000,19000,19400)],
                  limits=["Logged wm_err8 uses changing replay data and running normalization.",
                          "v_loss measures fitting imagined GAE targets, not true value accuracy.",
                          "pi_gradnorm is the combined preclip norm; it cannot separate auxiliary gradients.",
                          "Temporal coincidence does not prove a causal direction."])
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report["windows"],indent=2))


if __name__ == "__main__":
    main()
