"""Aggregate the repeat runs of PRD Sec 9.25: mean +- sample std over seeds 0-4 of every
number the paper's monitor tables report, plus the surrogate-learner ablation.
Reads the per-seed JSON files written by lyra_behavioral_monitor.py,
lyra_behavioral_monitor_trigger.py and lyra_llm_monitor_eval.py (LYRA_SEED / LYRA_SURROGATE).

Usage: python pilot/multiseed_summary.py      (writes pilot/multiseed_summary.json)
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SEEDS = [0, 1, 2, 3, 4]
CLASSES = ["T1", "T3", "D", "I", "C"]
AGENTS = ["llama8b", "llama70b", "gptoss20b"]
DOMAINS = ["financial", "medical", "cyber", "power"]


def tag(seed, kind="mlp"):
    return "" if (seed == 0 and kind == "mlp") else f"_s{seed}_{kind}"


def wmean(rows, k):
    if not rows:
        return float("nan")
    w = np.array([r["w"] for r in rows], float)
    return float((w * np.array([float(r[k]) for r in rows])).sum() / w.sum())


def lat(rows, key, q):
    xs = [x for r in rows for x in r[key]]
    return float(np.percentile(xs, q)) if xs else float("nan")


def dense_metrics(res):
    F = res["faults"]
    H = [r for r in F if r["harmful"]]
    m = {}
    for c in CLASSES + ["ALL"]:
        rows = [r for r in H if c == "ALL" or r["cls"] == c]
        for k in ("B1", "B2", "B3", "B4", "B4c"):
            m[f"{c}.{k}"] = 100 * wmean(rows, k)
        for k in ("unsafe", "B3_exposure", "B4c_exposure", "veto_unsafe"):
            m[f"{c}.{k}"] = wmean(rows, k)
    m["lat_med"], m["lat_p90"] = lat(H, "B4c_lat", 50), lat(H, "B4c_lat", 90)
    m["veto_depth_max"] = max(r.get("veto_sev", 0.0) for r in F)
    shift = [v["shift_false_alarm"] for v in res["domains"].values()]
    m["shift_fa_cusum_max"] = 100 * max(max(s[d]["cusum"] for d in s) for s in shift)
    m["shift_fa_rate_min"] = 100 * min(min(s[d]["rate"] for d in s) for s in shift)
    m["surrogate_acc_min"] = 100 * min(v["surrogate_acc"] for v in res["domains"].values())
    m["veto_overblock_max"] = 100 * max(v["veto_overblock_rate"] for v in res["domains"].values())
    return m


def trigger_metrics(res):
    F = res["faults"]
    H = [r for r in F if r["harmful"]]
    m = {k: 100 * wmean(H, k) for k in ("B2", "B3", "B4c")}
    m["lat_med"], m["lat_p90"] = lat(H, "B4c_lat", 50), lat(H, "B4c_lat", 90)
    m["exposure"] = wmean(H, "B4c_exposure")
    m["veto_unsafe"] = wmean(H, "veto_unsafe")
    m["veto_depth_max"] = max(r["veto_sev"] for r in F)
    shift = [v["shift_false_alarm"] for v in res["domains"].values()]
    m["shift_fa_cusum_max"] = 100 * max(max(s[d]["cusum"] for d in s) for s in shift)
    return m


def llm_metrics(per_domain):
    m = {}
    for reg in ("agent", "commission_small", "commission"):
        for ag in AGENTS:
            H = [f for r in per_domain.values() for f in r["regimes"][reg]["faults"][ag] if f["harmful"]]
            P = [r["regimes"][reg]["agents"][ag] for r in per_domain.values()]
            m[f"{reg}.{ag}.B4c"] = 100 * wmean(H, "B4c")
            m[f"{reg}.{ag}.fa"] = 100 * float(np.mean([p["false_alarm"]["cusum"] for p in P]))
            if reg != "commission":
                continue
            m[f"{reg}.{ag}.B2"] = 100 * wmean(H, "B2")
            m[f"{reg}.{ag}.fa_rate"] = 100 * float(np.mean([p["false_alarm"]["rate"] for p in P]))
            m[f"{reg}.{ag}.lat_med"] = lat(H, "B4c_lat", 50)
            m[f"{reg}.{ag}.exposure"] = wmean(H, "B4c_exposure")
            m[f"{reg}.{ag}.not_vetoed"] = wmean(H, "B4c_exposure_veto")
            m[f"{reg}.{ag}.missed_depth_max"] = max([f["B4c_sev"] for f in H] or [0.0])
            m[f"{reg}.{ag}.overblock"] = 100 * float(np.mean([p["veto_overblock_rate"] for p in P]))
    return m


def stats(runs):
    keys = runs[0].keys()
    out = {}
    for k in keys:
        v = np.array([r[k] for r in runs], float)
        out[k] = dict(mean=float(np.nanmean(v)), std=float(np.nanstd(v, ddof=1)) if len(v) > 1 else 0.0,
                      min=float(np.nanmin(v)), max=float(np.nanmax(v)), n=len(v))
    return out


def load(name):
    p = HERE / name
    return json.load(open(p)) if p.exists() else None


def main():
    out = {}
    runs = [dense_metrics(r) for s in SEEDS if (r := load(f"behavioral_monitor_results{tag(s)}.json"))]
    out["dense"] = stats(runs)
    runs = [trigger_metrics(r) for s in SEEDS if (r := load(f"behavioral_monitor_trigger_results{tag(s)}.json"))]
    out["trigger"] = stats(runs)
    runs = []
    for s in SEEDS:
        per = {d: load(f"llm_monitor_results_{d}{tag(s)}.json") for d in DOMAINS}
        if all(per.values()):
            runs.append(llm_metrics(per))
    out["llm"] = stats(runs) if runs else {}
    out["ablation"] = {}          # first run only, kept for the record
    out["ablation_runs"] = {}     # every available seed, per learner
    for kind in ("mlp", "hgb", "knn", "logreg"):
        r = load(f"behavioral_monitor_results{tag(0, kind)}.json")
        if r:
            out["ablation"][kind] = dense_metrics(r)
        runs = [dense_metrics(r) for s in SEEDS if (r := load(f"behavioral_monitor_results{tag(s, kind)}.json"))]
        if runs:
            out["ablation_runs"][kind] = stats(runs)
    json.dump(out, open(HERE / "multiseed_summary.json", "w"), indent=1)

    def show(sec, keys):
        for k in keys:
            if k in out[sec]:
                v = out[sec][k]
                print(f"  {k:<34} {v['mean']:9.2f} +- {v['std']:6.2f}   [{v['min']:.2f}, {v['max']:.2f}]  n={v['n']}")
    print("== dense (Table monitor)")
    show("dense", [f"{c}.B4c" for c in CLASSES + ["ALL"]] + ["ALL.B2", "ALL.B3", "ALL.B4c_exposure",
         "ALL.veto_unsafe", "lat_med", "lat_p90", "shift_fa_cusum_max", "veto_depth_max"])
    print("== trigger (Table trigger)")
    show("trigger", list(out["trigger"].keys()))
    if out["llm"]:
        print("== llm (Table llm)")
        show("llm", [k for k in out["llm"] if k.startswith("commission.")] +
             [k for k in out["llm"] if k.startswith(("agent.", "commission_small.")) and k.endswith(".B4c")])
    print("== ablation (seed 0)")
    for kind, m in out["ablation"].items():
        print(f"  {kind:<7} recall ALL={m['ALL.B4c']:.1f}%  T3={m['T3.B4c']:.1f}  D={m['D.B4c']:.1f}  I={m['I.B4c']:.1f}"
              f"  exposure={m['ALL.B4c_exposure']:.2f}  veto={m['ALL.veto_unsafe']:.2f}  shiftFA={m['shift_fa_cusum_max']:.1f}%"
              f"  acc_min={m['surrogate_acc_min']:.3f}%  overblock_max={m['veto_overblock_max']:.3f}%")


if __name__ == "__main__":
    main()
