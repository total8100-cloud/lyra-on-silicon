"""The paper's monitor (uniform commissioning sweep) on real LLM-agent traffic, seeds 11-16, with every
metric of Table llm (PRD Sec 9.37). Replays the random streams of lyra_active_llm_eval.py, which saved only
part of the metrics, and checks each fault against that run. Then pools seeds 0-4 and 11-16.

Usage: LYRA_SEED=<11..16> python pilot/lyra_llm_extra_seeds.py --domain medical
       python pilot/lyra_llm_extra_seeds.py --summary
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import lyra_active_llm_eval as al  # noqa: E402
from lyra_active_llm_eval import bm, ev, tg, load_schema  # noqa: E402

PAPER = [0, 1, 2, 3, 4]
EXTRA = [11, 12, 13, 14, 15, 16]


def run_domain(dom):
    d, seed = ev.DOMAIN_KEY[dom], bm.SEED
    if seed not in EXTRA:
        raise SystemExit(f"seeds are {EXTRA}")
    rng = np.random.default_rng(80000 + seed)
    sh = tg.TriggeredShield(load_schema(bm.DOMAINS[dom]))
    theta_if = bm.train_theta_if(
        [t for p in bm.DOMAINS.values() for t in bm.Shield(load_schema(p)).theta.tolist()],
        np.random.default_rng(1 + 1000 * seed))
    logs = {ag: ev.load_episodes(ag, d) for ag in ev.AGENTS}
    ka = sorted(logs[ev.TRAIN_AGENT])
    held_out = {ev.TRAIN_AGENT: [k for k in ka if k % 4 == 3],
                **{ag: sorted(logs[ag]) for ag in ev.AGENTS if ag != ev.TRAIN_AGENT}}
    raw_fa = {ag: [ev.resampled(logs[ag], held_out[ag], rng) for _ in range(ev.N_CAL)] for ag in ev.AGENTS}
    raw_test = {ag: [ev.resampled(logs[ag], held_out[ag], rng) for _ in range(bm.R)] for ag in ev.AGENTS}
    mon, Xt, at = al.commission("uniform", sh, np.random.default_rng(81000 + 10 * seed))
    A_t = sh.active(at)
    rate = tg.ActiveRateMonitor().fit(sh.decide(Xt) & A_t, A_t)
    rc = np.random.default_rng(82000 + seed)
    cal = [ev.materialize(sh, mon, bm.sample_states(rc, sh.names, ev.T), sh.sample_actions(rc, ev.T))
           for _ in range(ev.N_CAL)]
    win = bm.SurrogateMonitor()
    win.calibrate([(P, D_) for _, _, _, P, D_ in cal])
    cus = bm.CusumMonitor().fit([(P, D_) for _, _, _, P, D_ in cal])
    rate.calibrate([(D_, A_) for _, A_, _, _, D_ in cal])
    out = dict(cusum_p0=cus.p0, cusum_h=cus.tau, agents={}, faults={})
    for ag in ev.AGENTS:
        X, a = ev.concat(logs[ag], held_out[ag])
        _, A, _, P, D = ev.materialize(sh, mon, X, a)
        conf = A & (np.abs(P - 0.5) > bm.CONF_GATE)
        fa = {"cusum": 0, "window": 0, "rate": 0}
        for Xr, ar in raw_fa[ag]:
            _, Ar, _, Pr, Dr = ev.materialize(sh, mon, Xr, ar)
            fa["cusum"] += cus.stat(Pr, Dr).max() >= cus.tau
            fa["window"] += win.stat(Pr, Dr).max() >= win.tau
            fa["rate"] += rate.stat(Dr, Ar).max() >= rate.tau
        tests = [ev.materialize(sh, mon, Xr, ar) for Xr, ar in raw_test[ag]]
        act_frac = np.mean([A_.mean(axis=0) for _, A_, _, _, _ in tests], axis=0)
        out["agents"][ag] = dict(
            false_alarm={k: float(v / ev.N_CAL) for k, v in fa.items()},
            veto_overblock_rate=float((bm.blocks(P) & ~D & A).any(axis=1).mean()),
            conf_veto_overblock_rate=float((bm.blocks(P) & ~D & conf).any(axis=1).mean()))
        out["faults"][ag] = tg.evaluate_faults(dom, sh, tests, theta_if, win, cus, rate, act_frac)
    # the replay must match the earlier run of the same seed, fault by fault
    old = json.load(open(HERE / f"active_llm_results_{d}_s{seed}.json"))["methods"]["uniform"]["agents"]
    bad = sum(n["B4c"] != o["det"] or n["B4c_exposure"] != o["exp"] or n["harmful"] != o["harmful"]
              for ag in ev.AGENTS for n, o in zip(out["faults"][ag], old[ag]["faults"]))
    bad += sum(out["agents"][ag]["false_alarm"]["cusum"] != old[ag]["fa"] for ag in ev.AGENTS)
    print(f"[{dom} s{seed}] replay mismatches: {bad}", flush=True)
    if bad:
        sys.exit(2)
    json.dump(dict(domain=dom, seed=seed, regimes={"commission": out}),
              open(HERE / f"llm_extra_results_{d}_s{seed}.json", "w"))


def wmean(rows, k):
    if not rows:
        return float("nan")
    w = np.array([r["w"] for r in rows], float)
    return float((w * np.array([float(r[k]) for r in rows])).sum() / w.sum())


def load(seed):
    per = {}
    for dom, d in ev.DOMAIN_KEY.items():
        name = (f"llm_monitor_results_{d}{'' if seed == 0 else f'_s{seed}_mlp'}.json" if seed in PAPER
                else f"llm_extra_results_{d}_s{seed}.json")
        per[dom] = json.load(open(HERE / name))["regimes"]["commission"]
    return per


def summarize():
    out = {}
    for label, seeds in (("paper_5", PAPER), ("extra_6", EXTRA), ("all_11", PAPER + EXTRA)):
        out[label] = {}
        print(f"\n===== {label}: seeds {seeds}")
        runs = {s: load(s) for s in seeds}
        for ag in ev.AGENTS:
            rows = {}
            for s, per in runs.items():
                F = [f for r in per.values() for f in r["faults"][ag]]
                H = [f for f in F if f["harmful"]]
                S = [f for f in F if not f["behav_visible"]]
                dep = sorted(f["B4c_sev"] for f in H if f["B4c_sev"] > 0)
                lat = [x for f in H for x in f["B4c_lat"]]
                P = [r["agents"][ag] for r in per.values()]
                dom_rec = {dom: 100 * wmean([f for f in r["faults"][ag] if f["harmful"]], "B4c") for dom, r in per.items()}
                v = dict(harmful=len(H), recall=100 * wmean(H, "B4c"), b2=100 * wmean(H, "B2"),
                         fa=100 * float(np.mean([p["false_alarm"]["cusum"] for p in P])),
                         fa_worst=100 * max(p["false_alarm"]["cusum"] for p in P),
                         fa_rate=100 * float(np.mean([p["false_alarm"]["rate"] for p in P])),
                         lat=float(np.median(lat)) if lat else float("nan"), exposure=wmean(H, "B4c_exposure"),
                         not_vetoed=wmean(H, "B4c_exposure_veto"), depth=max(dep or [0.0]),
                         depth_p90=float(np.percentile(dep, 90)) if dep else 0.0,
                         veto_depth=max(f["veto_sev"] for f in F),
                         overblock=100 * float(np.mean([p["veto_overblock_rate"] for p in P])),
                         cveto_over=100 * float(np.mean([p["conf_veto_overblock_rate"] for p in P])),
                         cveto_not=wmean(H, "B4c_exposure_cveto"), cveto_depth=max(f["vg_sev"] for f in F),
                         silent=100 * wmean(S, "B4c"))
                for k, x in v.items():
                    rows.setdefault(k, []).append(x)
                rows.setdefault("per_domain", []).append(dom_rec)
            print(f"-- {ag}")
            res = {}
            for k, x in rows.items():
                if k == "per_domain":
                    continue
                x = np.array(x, float)
                res[k] = dict(mean=float(np.nanmean(x)), sd=float(np.nanstd(x, ddof=1)), min=float(np.nanmin(x)), max=float(np.nanmax(x)))
                print(f"   {k:<12} mean {x.mean():8.3f} sd {x.std(ddof=1):7.3f} min {x.min():8.3f} max {x.max():8.3f}")
            low = int(np.argmin(rows["recall"]))
            print(f"   lowest run: seed {seeds[low]} {rows['recall'][low]:.1f}%  by domain {({k: round(v, 1) for k, v in rows['per_domain'][low].items()})}")
            res["lowest"] = dict(seed=seeds[low], recall=rows["recall"][low], by_domain=rows["per_domain"][low])
            out[label][ag] = res
        fa_cases = [(s, dom, ag, r["agents"][ag]["false_alarm"]["cusum"]) for s, per in runs.items()
                    for dom, r in per.items() for ag in ev.AGENTS if r["agents"][ag]["false_alarm"]["cusum"] > 0]
        print("false-alarm cases:", fa_cases, "of", len(seeds) * 4 * 3)
        out[label]["fa_cases"] = fa_cases
    json.dump(out, open(HERE / "llm_eleven_runs_summary.json", "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", choices=list(ev.DOMAIN_KEY.values()))
    ap.add_argument("--summary", action="store_true")
    args = ap.parse_args()
    if args.summary:
        summarize()
    else:
        run_domain({v: k for k, v in ev.DOMAIN_KEY.items()}[args.domain])
