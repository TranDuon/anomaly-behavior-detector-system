"""Compare vandalism confirmation rules offline on score_sequences.jsonl (event level).

    .venv\\Scripts\\python.exe -m extensions.vandalism.gate_benchmark

For each rule: share of vandalism clips that raise an alarm (Kinetics), ASHEN test
videos alarmed inside the annotated interval, false-alarm rate on negative clips
(Kinetics hard negatives + Charades) and on normal CCTV videos, and alarm delay.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from fight.runtime import EventGate

ROOT = Path(__file__).resolve().parents[2]
SEQUENCES = ROOT / "datasets" / "vandalism" / "raw" / "eval_v1" / "score_sequences.jsonl"
SPEC = ROOT / "weights" / "extensions" / "vandalism" / "vandalism.json"

RULES = {
    "current (cadence 2, raw, confirm 0.8)": dict(cadence=2, smooth="raw", confirm=0.8),
    "cadence 1": dict(cadence=1, smooth="raw", confirm=0.8),
    "cadence 1 + max of last 2": dict(cadence=1, smooth="max2", confirm=0.8),
    "cadence 1 + max of last 3": dict(cadence=1, smooth="max3", confirm=0.8),
    "cadence 1 + mean of last 3": dict(cadence=1, smooth="mean3", confirm=0.8),
    "cadence 1 + confirm 0.5": dict(cadence=1, smooth="raw", confirm=0.5),
    "cadence 1 + max2 + confirm 0.5": dict(cadence=1, smooth="max2", confirm=0.5),
    "cadence 1 + max3 + confirm 0.5": dict(cadence=1, smooth="max3", confirm=0.5),
    "cadence 1 + mean3 + confirm 0.5": dict(cadence=1, smooth="mean3", confirm=0.5),
    "cadence 2 + max of last 2": dict(cadence=2, smooth="max2", confirm=0.8),
}


def smooth(values, kind):
    if kind == "raw":
        return values
    size = int(kind[-1])
    reducer = np.max if kind.startswith("max") else np.mean
    return [float(reducer(values[max(0, i - size + 1):i + 1])) for i in range(len(values))]


def run_rule(times, scores, rule, threshold, release):
    picked = list(range(0, len(times), rule["cadence"]))
    t = [times[i] for i in picked]
    s = smooth([scores[i] for i in picked], rule["smooth"])
    gate = EventGate(threshold, threshold * 0.7, confirm_seconds=rule["confirm"], release_seconds=release)
    alarms = []
    for ti, si in zip(t, s):
        gate.update(min(1.0, max(0.0, si)), ti)
        alarms.append(gate.active)
    return t, alarms


def evaluate(rows, rule, threshold, release):
    out = {"kin_pos": [], "kin_delay": [], "neg": [], "ashen_pos": [], "ashen_neg": [], "ashen_fp_steps": 0,
           "ashen_neg_seconds": 0.0}
    for row in rows:
        t, alarms = run_rule(row["t"], row["score"], rule, threshold, release)
        fired = any(alarms)
        if row["source"] == "ashen":
            if row["binary"]:
                inside = any(a and lab for a, lab in zip(alarms, [row["window_label"][i] for i in range(0, len(row["t"]), rule["cadence"])]))
                out["ashen_pos"].append(inside)
            else:
                out["ashen_neg"].append(fired)
                out["ashen_neg_seconds"] += (t[-1] - t[0]) if len(t) > 1 else 0
                out["ashen_fp_steps"] += sum(alarms)
        elif row["binary"]:
            out["kin_pos"].append(fired)
            if fired:
                out["kin_delay"].append(t[alarms.index(True)])
        else:
            out["neg"].append(fired)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequences", type=Path, default=SEQUENCES)
    args = parser.parse_args()
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    threshold, release = float(spec["threshold"]), float(spec.get("release_seconds", 1.0))
    rows = [json.loads(l) for l in args.sequences.read_text(encoding="utf-8").splitlines() if l]
    print(f"{len(rows)} clips, threshold {threshold}\n")
    header = f"{'rule':38s} {'Kinetics recall':>16s} {'ASHEN recall':>13s} {'neg FP':>10s} {'CCTV normal FP':>15s} {'first alarm s':>14s}"
    print(header)
    print("-" * len(header))
    results = {}
    for name, rule in RULES.items():
        r = evaluate(rows, rule, threshold, release)
        results[name] = r
        def pct(values):
            return f"{sum(values)}/{len(values)} ({100 * np.mean(values):.0f}%)" if values else "-"
        delay = f"{np.median(r['kin_delay']):.1f}" if r["kin_delay"] else "-"
        print(f"{name:38s} {pct(r['kin_pos']):>16s} {pct(r['ashen_pos']):>13s} {pct(r['neg']):>10s} {pct(r['ashen_neg']):>15s} {delay:>14s}")
    print("\nfalse alarms by negative class:")
    shortlist = ["current (cadence 2, raw, confirm 0.8)", "cadence 1 + confirm 0.5", "cadence 1 + mean3 + confirm 0.5",
                 "cadence 1 + max2 + confirm 0.5"]
    labels = sorted({row["label"] for row in rows if not row["binary"] and row["source"] != "ashen"})
    print(f"{'class':40s}" + "".join(f"{name[:22]:>24s}" for name in shortlist))
    for label in labels:
        subset = [row for row in rows if row["label"] == label]
        cells = []
        for name in shortlist:
            fired = sum(any(run_rule(r["t"], r["score"], RULES[name], threshold, release)[1]) for r in subset)
            cells.append(f"{fired}/{len(subset)}")
        print(f"{label[:40]:40s}" + "".join(f"{c:>24s}" for c in cells))
    (args.sequences.parent / "gate_benchmark.json").write_text(json.dumps(
        {k: {kk: (vv if not isinstance(vv, list) else [bool(x) if isinstance(x, (bool, np.bool_)) else x for x in vv])
             for kk, vv in v.items()} for k, v in results.items()}, indent=1, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
