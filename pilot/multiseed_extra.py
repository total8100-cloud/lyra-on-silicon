"""Secondary repeat-run statistics quoted in the paper text (PRD Sec 9.25), beyond the
table metrics of multiseed_summary.py."""
import json

import numpy as np

from multiseed_summary import AGENTS, DOMAINS, HERE, SEEDS, load, tag, wmean


def ms(v):
    v = np.array(v, float)
    return f"{v.mean():.2f} +- {v.std(ddof=1):.2f} [{v.min():.2f}, {v.max():.2f}]"


dense = [load(f"behavioral_monitor_results{tag(s)}.json") for s in SEEDS]
trig = [load(f"behavioral_monitor_trigger_results{tag(s)}.json") for s in SEEDS]
for name, runs in (("dense", dense), ("trigger", trig)):
    fa_rate = [100 * min(r["domains"][d]["shift_false_alarm"][x]["rate"]
                         for d in r["domains"] if d != "Power" for x in ("low", "high")) for r in runs]
    fa_rate_max = [100 * max(r["domains"][d]["shift_fa" if False else "shift_false_alarm"][x]["rate"]
                             for d in r["domains"] if d != "Power" for x in ("low", "high")) for r in runs]
    print(name, "B3 shift FA, non-Power domains: min", ms(fa_rate), " max", ms(fa_rate_max))
dense_B1 = [100 * wmean([f for f in r["faults"] if f["harmful"]], "B1") for r in dense]
dense_B4w = [100 * wmean([f for f in r["faults"] if f["harmful"]], "B4") for r in dense]
dense_ratio = [wmean([f for f in r["faults"] if f["harmful"]], "B3_exposure") /
               wmean([f for f in r["faults"] if f["harmful"]], "B4c_exposure") for r in dense]
print("dense B1", ms(dense_B1), " B4 window", ms(dense_B4w), " B3/B4 exposure ratio", ms(dense_ratio))
low = []
for r in trig:
    rows = [f for f in r["faults"] if f["harmful"] and f.get("target_active_fraction", 1) < 0.3]
    low.append(np.median([x for f in rows for x in f["B4c_lat"]]))
print("trigger latency median, constraints active <30% of steps", ms(low))
dense_ws = [wmean([f for f in r["faults"] if f["harmful"]], "B4c_exposure") for r in dense]
trig_ws = [wmean([f for f in r["faults"] if f["harmful"]], "B4c_exposure") for r in trig]
print("exposure dense", ms(dense_ws), " trigger", ms(trig_ws))
silent = [sum(not f["behav_visible"] for f in r["faults"]) for r in trig]
silent_alarm = [100 * wmean([f for f in r["faults"] if not f["behav_visible"]], "B4c") for r in trig]
print("trigger silent faults", silent, "alarm rate", silent_alarm)

for ag in AGENTS:
    cv, cob, ob, ex = [], [], [], []
    for s in SEEDS:
        per = {d: load(f"llm_monitor_results_{d}{tag(s)}.json") for d in DOMAINS}
        H = [f for r in per.values() for f in r["regimes"]["commission"]["faults"][ag] if f["harmful"]]
        cv.append(wmean(H, "B4c_exposure_cveto"))
        cob.append(100 * np.mean([r["regimes"]["commission"]["agents"][ag]["conf_veto_overblock_rate"] for r in per.values()]))
        ob.append(100 * np.mean([r["regimes"]["commission"]["agents"][ag]["veto_overblock_rate"] for r in per.values()]))
        depths = sorted(f["B4c_sev"] for f in H if f["B4c_sev"] > 0)
        ex.append(np.percentile(depths, 90) if depths else 0.0)
    print(f"{ag}: confident-veto unvetoed-before-alarm {ms(cv)}; conf-veto overblock {ms(cob)}; "
          f"full-veto overblock {ms(ob)}; never-flagged depth p90 {ms(ex)}")
