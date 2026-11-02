"""Compare monitor variants of the improvement study (PRD Sec 9.26) against the paper's
configuration, over the seeds available for each: dense synthetic setting, and (with
--llm) the commissioning-trained monitor on the real LLM-agent traffic.

Usage: python pilot/improve_compare.py [--llm]
"""
import sys

import numpy as np

from multiseed_summary import AGENTS, DOMAINS, load, wmean

SEEDS = [0, 1, 2, 3, 4]
# name -> tag suffix builder (seed -> file tag), matching lyra_behavioral_monitor.TAG
CONFIGS = {
    "paper (mlp, gate .3, p1 4)": lambda s: "" if s == 0 else f"_s{s}_mlp",
    "gate 0.2": lambda s: f"_s{s}_mlp_g0.2",
    "gate 0.4": lambda s: f"_s{s}_mlp_g0.4",
    "p1 2": lambda s: f"_s{s}_mlp_p2",
    "p1 8": lambda s: f"_s{s}_mlp_p8",
    "ens_avg": lambda s: f"_s{s}_ens_avg",
    "ens_min": lambda s: f"_s{s}_ens_min",
}


def metrics(F, shift=None):
    H = [f for f in F if f["harmful"]]
    low = [f for f in H if f["unsafe"] <= 20]
    silent = [f for f in F if not f["behav_visible"]]
    sev = sorted(f["B4c_sev"] for f in H if f["B4c_sev"] > 0)
    m = dict(recall=100 * wmean(H, "B4c"), recall_low=100 * wmean(low, "B4c"),
             D=100 * wmean([f for f in H if f["cls"] == "D"], "B4c") if any(f["cls"] == "D" for f in H) else np.nan,
             I=100 * wmean([f for f in H if f["cls"] == "I"], "B4c") if any(f["cls"] == "I" for f in H) else np.nan,
             exposure=wmean(H, "B4c_exposure"), veto=wmean(H, "veto_unsafe"),
             missed_max=max(sev) if sev else 0.0, missed_p90=float(np.percentile(sev, 90)) if sev else 0.0,
             silent_alarm=100 * wmean(silent, "B4c") if silent else 0.0)
    if shift is not None:
        m["shift_fa_max"] = shift
    return m


def show(name, runs):
    if not runs:
        print(f"{name:<28} (no runs yet)")
        return
    keys = runs[0].keys()
    cells = []
    for k in keys:
        v = np.array([r[k] for r in runs], float)
        cells.append(f"{k}={np.nanmean(v):.2f}" + (f"±{np.nanstd(v, ddof=1):.2f}" if len(v) > 1 else ""))
    print(f"{name:<28} n={len(runs)}  " + "  ".join(cells))


def dense():
    print("== dense synthetic setting")
    for name, tg in CONFIGS.items():
        runs = []
        for s in SEEDS:
            r = load(f"behavioral_monitor_results{tg(s)}.json")
            if r:
                shift = 100 * max(max(v["shift_false_alarm"][d]["cusum"] for d in ("low", "high"))
                                  for v in r["domains"].values())
                runs.append(metrics(r["faults"], shift))
        show(name, runs)


def llm(names):
    print("== real LLM-agent traffic, commissioning-trained monitor")
    for name in names:
        tg = CONFIGS[name]
        for ag in AGENTS:
            runs = []
            for s in SEEDS:
                per = {d: load(f"llm_monitor_results_{d}{tg(s)}.json") for d in DOMAINS}
                if all(per.values()):
                    F = [f for r in per.values() for f in r["regimes"]["commission"]["faults"][ag]]
                    fa = 100 * float(np.mean([r["regimes"]["commission"]["agents"][ag]["false_alarm"]["cusum"]
                                              for r in per.values()]))
                    m = metrics(F)
                    m["fa"] = fa
                    m["not_vetoed"] = wmean([f for f in F if f["harmful"]], "B4c_exposure_veto")
                    m["overblock"] = 100 * float(np.mean([r["regimes"]["commission"]["agents"][ag]["veto_overblock_rate"]
                                                          for r in per.values()]))
                    runs.append(m)
            show(f"{name} / {ag}", runs)


if __name__ == "__main__":
    dense()
    if "--llm" in sys.argv:
        llm([n for n in CONFIGS if n.startswith("paper")] + sys.argv[sys.argv.index("--llm") + 1:])
