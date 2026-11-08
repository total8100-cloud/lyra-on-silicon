"""Summarize lyra_redundancy_baseline.py over its runs (PRD Sec 9.31).
Usage: python pilot/redundancy_summary.py   (prints the tables; writes redundancy_summary.json)
"""
import glob
import json
import os
import re

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SHARE = {"sign": 1 / 64, "exponent": 11 / 64, "mantissa_high": 12 / 64, "mantissa_low": 40 / 64}
LOW = 20       # low-severity faults: at most this many unsafe passes per stream, as in the paper


def wmean(rows, key):
    w = np.array([r["w"] for r in rows], float)
    v = np.array([float(r[key]) for r in rows])
    return float((w * v).sum() / w.sum()) if rows else float("nan")


def ms(vals):
    return f"{np.mean(vals):8.2f} ±{np.std(vals, ddof=1) if len(vals) > 1 else 0:5.2f}"


files = sorted(f for f in glob.glob(os.path.join(HERE, "redundancy_results_s*.json")) if re.search(r"_s\d+\.json$", f))
if not files:
    raise SystemExit("no redundancy_results_s*.json next to this script (extract raw_results.zip first)")
runs = [json.load(open(f)) for f in files]
out = dict(runs=len(runs), replay=[r["replay_check"] for r in runs])
print(f"runs: {len(runs)}; replay mismatches: {[r['replay_check']['mismatches'] for r in runs]}")

print("\n== A. fault in the shield: monitor B4 vs. DMR B5, harmful faults, weighted by multiplicity")
print(f"{'class':<9}{'B4 recall':>17}{'B5 recall':>17}{'B4 exposure':>17}{'B5 exposure':>17}{'B5 latency':>17}")
out["A"] = {}
for cls in ("T1", "T3", "D", "I", "C", "ALL", "LOW"):
    rec4, rec5, exp4, exp5, lat5 = [], [], [], [], []
    for r in runs:
        rows = [f for f in r["faults"] if f["harmful"]]
        if cls == "LOW":
            rows = [f for f in rows if f["unsafe"] <= LOW]
        elif cls != "ALL":
            rows = [f for f in rows if f["cls"] == cls]
        rec4.append(wmean(rows, "B4c") * 100)
        rec5.append(wmean(rows, "B5") * 100)
        exp4.append(wmean(rows, "B4c_exposure"))
        exp5.append(wmean(rows, "B5_exposure"))
        lat5.append(wmean([f for f in rows if f["B5_lat"] is not None], "B5_lat"))
    print(f"{cls:<9}{ms(rec4):>17}{ms(rec5):>17}{ms(exp4):>17}{ms(exp5):>17}{ms(lat5):>17}")
    out["A"][cls] = dict(B4=float(np.mean(rec4)), B4_sd=float(np.std(rec4, ddof=1)), B5=float(np.mean(rec5)),
                         B5_min=float(np.min(rec5)), B4_exposure=float(np.mean(exp4)), B5_exposure=float(np.mean(exp5)),
                         B5_latency=float(np.mean(lat5)))

print("\n== B. fault in the reference, healthy shield, executed decision = shield OR reference")
tot = [len(r["faults"]) for r in runs]
vis = [sum(f["behav_visible"] for f in r["faults"]) for r in runs]
uns = [sum(f["ref_unsafe_exec"] for f in r["faults"]) for r in runs]
print(f"enumerated faults {tot[0]}; decision-changing in the test streams (alarm at the first change): "
      f"{ms(vis)}; silent {ms([t - v for t, v in zip(tot, vis)])}; unsafe executed decisions in all runs: {sum(uns)}")
out["B"] = dict(enumerated=tot[0], visible=float(np.mean(vis)), unsafe_executed=int(sum(uns)))

print("\n== C. floating-point reference vs. the Q4.12 shield, no fault anywhere")
out["C"] = {}
for setting in ("in_distribution", "low", "high"):
    share, per = [], []
    for r in runs:
        d = [v["float_reference"][setting] for v in r["domains"].values()]
        share.append([x["streams_with_mismatch"] / x["streams"] * 100 for x in d])
        per.append(np.mean([x["mismatches_per_stream"] for x in d]))
    share = np.array(share)
    print(f"{setting:<16} streams with a false alarm: mean {share.mean():5.1f}%  worst domain/run {share.max():5.1f}%"
          f"  best {share.min():5.1f}%; mismatches per stream {np.mean(per):.2f}")
    out["C"][setting] = dict(mean=float(share.mean()), worst=float(share.max()), best=float(share.min()))

print("\n== Host cost (Python on this machine; a rough ratio only)")
tm = [v["t_mlp_us"] for r in runs for v in r["domains"].values()]
tr = [v["t_reference_us"] for r in runs for v in r["domains"].values()]
np0 = {k: v["n_surrogate_params"] for k, v in runs[0]["domains"].items()}
print(f"per decision: MLP {np.mean(tm):.3f} us, fixed-point reference {np.mean(tr):.3f} us; ratio {np.mean(tm) / np.mean(tr):.1f}")
for k, v in runs[0]["domains"].items():
    print(f"  {k:<10} surrogate {np0[k]} params = {v['surrogate_kbytes_f64']:.1f} KiB (float64); "
          f"reference {v['n_thresholds']} thresholds = {v['threshold_bytes_q412']} bytes (plus the expressions)")
out["cost"] = dict(t_mlp_us=float(np.mean(tm)), t_reference_us=float(np.mean(tr)))

print("\n== D. monitor weight flips, healthy shield (population-weighted over a uniformly random bit)")
print(f"{'group':<15}{'share':>7}{'alarm %':>9}{'worst':>8}{'NaN %':>8}{'worst':>8}{'changed %':>11}{'extra blocks/2000':>19}")
rows_d = {}
for g, sh in SHARE.items():
    a = np.array([[v["weight_flips"][g]["alarm"] / v["weight_flips"][g]["n"] * 100 for v in r["domains"].values()] for r in runs])
    n = np.array([[v["weight_flips"][g]["nan"] / v["weight_flips"][g]["n"] * 100 for v in r["domains"].values()] for r in runs])
    c = np.array([[v["weight_flips"][g]["changed"] / v["weight_flips"][g]["n"] * 100 for v in r["domains"].values()] for r in runs])
    e = np.array([[v["weight_flips"][g]["extra_block_steps"] / v["weight_flips"][g]["n"] for v in r["domains"].values()] for r in runs])
    rows_d[g] = dict(alarm=a.mean(), nan=n.mean(), changed=c.mean(), extra=e.mean())
    print(f"{g:<15}{sh * 100:6.1f}%{a.mean():9.2f}{a.max():8.2f}{n.mean():8.3f}{n.max():8.3f}{c.mean():11.2f}{e.mean():19.2f}")
pop = {k: sum(SHARE[g] * rows_d[g][k] for g in SHARE) for k in ("alarm", "nan", "changed", "extra")}
print(f"{'any bit':<15}{100:6.1f}%{pop['alarm']:9.2f}{'':8}{pop['nan']:8.3f}{'':8}{pop['changed']:11.2f}{pop['extra']:19.2f}")
out["D"] = dict(by_group={g: {k: float(v) for k, v in d.items()} for g, d in rows_d.items()},
                population={k: float(v) for k, v in pop.items()})

print("\n== E. double fault: a flip that raised no alarm + a harmful shield fault")
print(f"{'group':<15}{'flips':>7}{'NaN flips':>10}{'recall retained %':>19}{'flips losing every detection':>30}")
ret = {}
quiet = {}
for g in SHARE:
    fl = nan = det = base = lost = lost_nan = 0
    q = []
    for r in runs:
        for v in r["domains"].values():
            d = v["double_fault"][g]
            fl += d["flips"]
            nan += d["nan_flips"]
            det += d["detected"]
            base += d["baseline_detected"]
            lost += d["flips_losing_every_detection"]
            lost_nan += d["losing_flips_with_nan"]
            q.append(d["quiet_share"])
    ret[g] = det / base * 100
    quiet[g] = float(np.mean(q))
    print(f"{g:<15}{fl:7d}{nan:10d}{ret[g]:19.1f}{lost:>24d}/{fl}   (with NaN: {lost_nan})")
wt = {g: SHARE[g] * quiet[g] for g in SHARE}
overall = sum(wt[g] * ret[g] for g in SHARE) / sum(wt.values())
print(f"{'any bit':<15}{'':7}{'':10}{overall:19.1f}")
panel = [v["panel_baseline_detected"] / v["panel_size"] * 100 for r in runs for v in r["domains"].values()]
print(f"panel: {runs[0]['domains']['Financial']['panel_size']} faults or so per domain; baseline detection on the panel {np.mean(panel):.1f}%"
      " (values above 100% retained mean a corrupted monitor raises more evidence, not that it is better)")
out["E"] = dict(retained=ret, overall=float(overall), panel_baseline=float(np.mean(panel)))

with open(os.path.join(HERE, "redundancy_summary.json"), "w") as fh:
    json.dump(out, fh, indent=1)
