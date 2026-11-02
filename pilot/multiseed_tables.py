"""Emit the LaTeX bodies of the paper's monitor tables from multiseed_summary.json, so that no
repeat-run number is typed by hand (PRD Sec 9.25). Prints four blocks: tab:monitor rows,
tab:trigger rows, tab:llm rows and the learner-ablation rows."""
import json

import numpy as np

from multiseed_summary import CLASSES, HERE, SEEDS, load, tag, wmean

S = json.load(open(HERE / "multiseed_summary.json"))
D, T, L, A = S["dense"], S["trigger"], S["llm"], S["ablation"]


def m(sec, k, f="{:.1f}"):
    return f.format(sec[k]["mean"])


def pm(sec, k, f="{:.1f}"):
    return f"{f.format(sec[k]['mean'])} $\\pm$ {f.format(sec[k]['std'])}"


# number of harmful faults per class, averaged over runs
harm = {c: [] for c in CLASSES + ["ALL"]}
for s in SEEDS:
    r = load(f"behavioral_monitor_results{tag(s)}.json")
    H = [f for f in r["faults"] if f["harmful"]]
    for c in CLASSES:
        harm[c].append(sum(f["cls"] == c for f in H))
    harm["ALL"].append(len(H))
names = {"T1": r"T1 $\theta$, no ECC  ", "T3": r"T3 $\theta$, SECDED  ", "D": r"D expr.\ register    ",
         "I": r"I input register     ", "C": r"C comparator         ", "ALL": "All                  "}
print("% ---- tab:monitor")
for c in CLASSES + ["ALL"]:
    if c == "ALL":
        print(r"\midrule")
    cells = [f"{np.mean(harm[c]):.0f}", m(D, f"{c}.B1"), m(D, f"{c}.B2"), m(D, f"{c}.B3"), m(D, f"{c}.B4"),
             r"\textbf{" + m(D, f"{c}.B4c") + "}", m(D, f"{c}.unsafe"), m(D, f"{c}.B3_exposure"),
             r"\textbf{" + m(D, f"{c}.B4c_exposure", "{:.2f}") + "}", m(D, f"{c}.veto_unsafe", "{:.2f}")]
    print(names[c] + " & " + " & ".join(cells) + r" \\")
print(r"\quad s.d.\ over runs & & " + " & ".join(
    f"{D[f'ALL.{k}']['std']:.1f}" for k in ("B1", "B2", "B3", "B4", "B4c")) + " & & " +
    f"{D['ALL.B3_exposure']['std']:.1f} & {D['ALL.B4c_exposure']['std']:.2f} & {D['ALL.veto_unsafe']['std']:.2f}" + r" \\")

print("% ---- tab:trigger (dense | trigger-gated), mean +- s.d. over runs")
rows = [("Recall on harmful faults, B4 (CUSUM)", pm(D, "ALL.B4c") + r"\%", pm(T, "B4c") + r"\%"),
        (r"Recall on harmful faults, B2 ($\theta$ IsolationForest)", pm(D, "ALL.B2") + r"\%", pm(T, "B2") + r"\%"),
        ("Recall on harmful faults, B3 (BLOCK-rate)", pm(D, "ALL.B3") + r"\%", pm(T, "B3") + r"\%"),
        ("B4 latency, median (steps)", pm(D, "lat_med"), pm(T, "lat_med")),
        ("B4 latency, 90th percentile (steps)", pm(D, "lat_p90", "{:.0f}"), pm(T, "lat_p90", "{:.0f}")),
        ("Unsafe passes before B4 alarm", pm(D, "ALL.B4c_exposure", "{:.2f}"), pm(T, "exposure", "{:.2f}")),
        ("Unsafe passes surviving the veto", pm(D, "ALL.veto_unsafe", "{:.2f}"), pm(T, "veto_unsafe", "{:.2f}")),
        ("Deepest unsafe pass surviving the veto (max)", f"{D['veto_depth_max']['max']:.3f}", f"{T['veto_depth_max']['max']:.3f}"),
        ("False alarms under input shift, B4 (worst run)", f"{D['shift_fa_cusum_max']['max']:.1f}" + r"\%",
         f"{T['shift_fa_cusum_max']['max']:.1f}" + r"\%")]
for a, b, c in rows:
    print(f"{a} & {b} & {c} \\\\")

print("% ---- tab:llm (commission regime), mean +- s.d. over runs")
ag = ["llama8b", "llama70b", "gptoss20b"]
k = lambda a, x: f"commission.{a}.{x}"
lines = [("Recall, B4 (CUSUM)", lambda a: pm(L, k(a, "B4c")) + r"\%"),
         (r"Recall, B2 ($\theta$ readout)", lambda a: m(L, k(a, "B2")) + r"\%"),
         ("False alarms, B4 / B3", lambda a: m(L, k(a, "fa")) + r"\% / " + m(L, k(a, "fa_rate"), "{:.0f}") + r"\%"),
         ("Median B4 latency (steps)", lambda a: pm(L, k(a, "lat_med"), "{:.0f}")),
         ("Unsafe passes before alarm", lambda a: pm(L, k(a, "exposure"), "{:.1f}")),
         (r"\quad of which not vetoed", lambda a: pm(L, k(a, "not_vetoed"), "{:.2f}")),
         ("Deepest unsafe pass, never-flagged fault (max)", lambda a: f"{L[k(a, 'missed_depth_max')]['max']:.2f}"),
         ("Veto over-blocking (fault-free steps)", lambda a: pm(L, k(a, "overblock"), "{:.2f}") + r"\%")]
for name, fn in lines:
    print(name + " & " + " & ".join(fn(a) for a in ag) + r" \\")

AR = S["ablation_runs"]
n_runs = min(AR[k]["ALL.B4c"]["n"] for k in AR)
print(f"% ---- ablation (mean over {n_runs} runs; worst run for shift FA)")
lab = {"mlp": r"MLP (64--64), used", "hgb": "Gradient-boosted trees", "knn": r"$k$-nearest neighbours ($k=15$)",
       "logreg": "Logistic regression"}
for kind in ("mlp", "hgb", "knn", "logreg"):
    a = AR[kind]
    print(f"{lab[kind]} & {a['surrogate_acc_min']['mean']:.2f}\\% & {pm(a, 'ALL.B4c')}\\% & {m(a, 'D.B4c')}\\% & "
          f"{m(a, 'I.B4c')}\\% & {m(a, 'ALL.B4c_exposure', '{:.2f}')} & {m(a, 'ALL.veto_unsafe', '{:.2f}')} & "
          f"{a['shift_fa_cusum_max']['max']:.1f}\\% \\\\")
