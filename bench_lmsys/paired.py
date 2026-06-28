"""Paired cross-seed delta between two policies (collapses cross-seed variance).

The seeds are paired: seed k feeds the SAME query stream to every policy, so the
right confidence statement on "policy A beats policy B" is the per-seed paired
delta A-B, not the difference of unpaired means. With few seeds we report the
paired mean +/- std and the sign-consistency count (positive in N/seeds).

Usage:
  python bench_lmsys/paired.py "bench_lmsys/win_z11_seed*.json" CA_W_TINYLFU LRU
"""

import glob
import json
import statistics
import sys
from collections import defaultdict

METRICS = [("cost_weighted_hit_rate", "cost_wt"), ("hit_rate", "hit"), ("token_saving_ratio", "tok")]


def main():
    pattern, a, b = sys.argv[1], sys.argv[2], sys.argv[3]
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(f"No files match {pattern!r}")
    # per[cs][metric] = list of per-seed (A-B) deltas, in pp
    per = defaultdict(lambda: defaultdict(list))
    for fp in files:
        d = json.load(open(fp))
        by = {(r["policy"], r["cache_size"]): r for r in d["results"]}
        css = {cs for (_, cs) in by}
        for cs in css:
            if (a, cs) not in by or (b, cs) not in by:
                continue
            for mkey, _ in METRICS:
                per[cs][mkey].append((by[(a, cs)][mkey] - by[(b, cs)][mkey]) * 100.0)

    n = len(files)
    print(f"Paired {a} - {b}  ({n} seeds)\n")
    print(f"  {'cache':<7}" + " ".join(f"{lbl+'(pp)':>22}" for _, lbl in METRICS))
    for cs in sorted(per):
        cells = []
        for mkey, _ in METRICS:
            v = per[cs][mkey]
            m, s = statistics.mean(v), (statistics.stdev(v) if len(v) > 1 else 0.0)
            pos = sum(x > 0 for x in v)
            cells.append(f"{m:+6.2f}+/-{s:4.2f} [{pos}/{n}]")
        print(f"  {cs:<7}" + " ".join(f"{c:>22}" for c in cells))


if __name__ == "__main__":
    main()
