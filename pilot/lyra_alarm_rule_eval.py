"""Mitigating the monitor's real-traffic false alarms through the alarm rule (PRD Sec 9.38).

The false alarms traced in PRD Sec 9.36 come from one confident surrogate error firing an alarm whose
calibrated threshold is close to zero. The surrogate, the calibration streams and the veto stay as in the
paper; only the rule that turns confident disagreements into an alarm changes.

Rules: base (the paper); k2, k3, k5 (at least k confident disagreements, a floor (k - 0.5) * up on the
CUSUM threshold); d2, d3 (evidence counts only on a state not yet flagged in this stream, threshold
recalibrated on that statistic, floor for d distinct states).

Fixed before running:
  Selection on seeds 11-16 (seen, they contain the false-alarm case): among the five rules with no
    real-traffic combination above 1% false alarms, the highest LLM recall pooled over agents.
  Evaluation on fresh seeds 20-25. The selected rule passes only if:
    1. no real-traffic combination of agent, domain and seed is above 1% false alarms;
    2. LLM recall loss against base is at most 2 points on each agent;
    3. dense recall loss is at most 1 point and worst shift false alarm is not above base.
  It is post hoc, so even if it passes it is reported as an operating point, not as the headline.

Usage: LYRA_SEED=<n> python pilot/lyra_alarm_rule_eval.py --domain medical
       LYRA_SEED=<n> python pilot/lyra_alarm_rule_eval.py --dense
       python pilot/lyra_alarm_rule_eval.py --summary
"""
import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
os.environ.setdefault("LYRA_SEED", "20")
import lyra_active_llm_eval as al  # noqa: E402
from lyra_active_llm_eval import bm, ev, tg, load_schema  # noqa: E402

RULES = ("base", "k2", "k3", "k5", "d2", "d3")
CANDIDATES = RULES[1:]
DEV = (11, 12, 13, 14, 15, 16)
EVAL = (20, 21, 22, 23, 24, 25)
LOW = 20


def hits_of(P, D):
    return np.flatnonzero(bm.disagree(P, D).any(axis=1))


def distinct(hits, X):
    seen, out = set(), []
    for t in hits:
        key = tuple(np.round(X[t] * bm.SCALE).astype(np.int64).tolist())
        if key not in seen:
            seen.add(key)
            out.append(t)
    return out


def peak(hits, up, down):
    s, prev, best = 0.0, -1, 0.0
    for t in hits:
        s = max(0.0, s + down * (t - prev - 1)) + up
        best = max(best, s)
        prev = t
    return best


def first(hits, up, down, h):
    s, prev = 0.0, -1
    for t in hits:
        s = max(0.0, s + down * (t - prev - 1)) + up
        if s >= h:
            return int(t)
        prev = t
    return None


class Rules:
    """Thresholds of every rule, from the fault-free calibration streams [(X, P, D), ...]."""

    def __init__(self, cal):
        self.cus = bm.CusumMonitor().fit([(P, D) for _, P, D in cal])
        self.up, self.down = self.cus.up, self.cus.down
        tau_d = float(np.quantile([peak(distinct(hits_of(P, D), X), self.up, self.down) for X, P, D in cal],
                                  1 - bm.ALPHA)) + 1e-9
        self.h = {"base": self.cus.tau}
        for k in (2, 3, 5):
            self.h[f"k{k}"] = max(self.cus.tau, (k - 0.5) * self.up)
        for d in (2, 3):
            self.h[f"d{d}"] = max(tau_d, (d - 0.5) * self.up)

    def alarms(self, X, P, D):
        H = hits_of(P, D)
        Hd = distinct(H, X) if len(H) else []
        return {r: first(Hd if r[0] == "d" else H, self.up, self.down, self.h[r]) for r in RULES}


def run_llm(dom):
    d, seed = ev.DOMAIN_KEY[dom], bm.SEED
    rng = np.random.default_rng(80000 + seed)
    sh = tg.TriggeredShield(load_schema(bm.DOMAINS[dom]))
    logs = {ag: ev.load_episodes(ag, d) for ag in ev.AGENTS}
    ka = sorted(logs[ev.TRAIN_AGENT])
    held_out = {ev.TRAIN_AGENT: [k for k in ka if k % 4 == 3],
                **{ag: sorted(logs[ag]) for ag in ev.AGENTS if ag != ev.TRAIN_AGENT}}
    raw_fa = {ag: [ev.resampled(logs[ag], held_out[ag], rng) for _ in range(ev.N_CAL)] for ag in ev.AGENTS}
    raw_test = {ag: [ev.resampled(logs[ag], held_out[ag], rng) for _ in range(bm.R)] for ag in ev.AGENTS}
    mon, _, _ = al.commission("uniform", sh, np.random.default_rng(81000 + 10 * seed))
    rc = np.random.default_rng(82000 + seed)
    cal = []
    for _ in range(ev.N_CAL):
        X_ = bm.sample_states(rc, sh.names, ev.T)
        a_ = sh.sample_actions(rc, ev.T)
        Xm, _, _, P, D = ev.materialize(sh, mon, X_, a_)
        cal.append((Xm, P, D))
    R = Rules(cal)
    out = dict(domain=dom, seed=seed, h={k: float(v) for k, v in R.h.items()}, agents={})
    faults = bm.enumerate_faults(sh)
    for ag in ev.AGENTS:
        fa = dict.fromkeys(RULES, 0)
        for Xr, ar in raw_fa[ag]:
            X_, _, _, P, D = ev.materialize(sh, mon, Xr, ar)
            for r, a in R.alarms(X_, P, D).items():
                fa[r] += a is not None
        tests = [ev.materialize(sh, mon, Xr, ar) for Xr, ar in raw_test[ag]]
        rows = []
        for f in faults:
            unsafe, over = [], []
            det = {r: [] for r in RULES}
            exp = {r: [] for r in RULES}
            for X_, A, E, P0, D0 in tests:
                Df = sh.decide(X_, f, E=E) & A
                P = np.where(A, P0, Df.astype(float))
                u = (D0 & ~Df).any(axis=1)
                unsafe.append(int(u.sum()))
                over.append(int((~D0 & Df).any(axis=1).sum()))
                for r, a in R.alarms(X_, P, Df).items():
                    det[r].append(a is not None)
                    exp[r].append(int(u[: (a + 1 if a is not None else len(X_))].sum()))
            rows.append(dict(cls=f["cls"], w=f["w"], harmful=bool(np.mean(unsafe) > 0), unsafe=float(np.mean(unsafe)),
                             behav_visible=bool(np.mean(unsafe) + np.mean(over) > 0),
                             det={r: float(np.mean(det[r])) for r in RULES},
                             exp={r: float(np.mean(exp[r])) for r in RULES}))
        out["agents"][ag] = dict(fa={r: fa[r] / ev.N_CAL for r in RULES}, faults=rows)
    if seed in DEV:     # the base rule must reproduce the earlier run of this seed
        old = json.load(open(HERE / f"llm_extra_results_{d}_s{seed}.json"))["regimes"]["commission"]
        bad = sum(n["det"]["base"] != o["B4c"] or n["exp"]["base"] != o["B4c_exposure"]
                  for ag in ev.AGENTS for n, o in zip(out["agents"][ag]["faults"], old["faults"][ag]))
        bad += sum(out["agents"][ag]["fa"]["base"] != old["agents"][ag]["false_alarm"]["cusum"] for ag in ev.AGENTS)
        print(f"[{dom} s{seed}] replay mismatches: {bad}", flush=True)
        if bad:
            sys.exit(2)
    json.dump(out, open(HERE / f"alarm_llm_results_{d}_s{seed}.json", "w"))
    print(f"[{dom} s{seed}] done", flush=True)


def run_dense():
    seed = bm.SEED
    rng = np.random.default_rng(50000 + seed)
    out = dict(seed=seed, domains={}, faults=[])
    for dom, path in bm.DOMAINS.items():
        sh = bm.Shield(load_schema(path))
        Xt = bm.sample_states(np.random.default_rng(60000 + seed * 100), sh.names, 100000)
        mon = bm.SurrogateMonitor(seed, "mlp").fit(Xt, sh.decide(Xt))
        cal = []
        for _ in range(bm.N_CAL_STREAMS):
            X_ = bm.sample_states(rng, sh.names, bm.T)
            cal.append((X_, mon.predict(X_), sh.decide(X_)))
        R = Rules(cal)
        shift = dict.fromkeys(RULES, 0.0)
        for dist in ("low", "high"):
            streams = []
            for _ in range(bm.N_CAL_STREAMS):
                X_ = bm.sample_states(rng, sh.names, bm.T, dist)
                streams.append((X_, sh.decide(X_)))
            cnt = dict.fromkeys(RULES, 0)
            for X_, D in streams:
                for r, a in R.alarms(X_, mon.predict(X_), D).items():
                    cnt[r] += a is not None
            for r in RULES:
                shift[r] = max(shift[r], cnt[r] / bm.N_CAL_STREAMS)
        tests = []
        for _ in range(bm.R):
            X_ = bm.sample_states(rng, sh.names, bm.T)
            E = sh.expr_q(X_)
            tests.append((X_, E, mon.predict(X_), sh.decide(X_, E=E)))
        out["domains"][dom] = dict(shift=shift)
        for f in bm.enumerate_faults(sh):
            unsafe, over = [], []
            det = {r: [] for r in RULES}
            for X_, E, P, D0 in tests:
                Df = sh.decide(X_, f, E=E)
                unsafe.append(int((D0 & ~Df).any(axis=1).sum()))
                over.append(int((~D0 & Df).any(axis=1).sum()))
                for r, a in R.alarms(X_, P, Df).items():
                    det[r].append(a is not None)
            out["faults"].append(dict(cls=f["cls"], w=f["w"], harmful=bool(np.mean(unsafe) > 0), unsafe=float(np.mean(unsafe)),
                                      behav_visible=bool(np.mean(unsafe) + np.mean(over) > 0),
                                      det={r: float(np.mean(det[r])) for r in RULES}))
        print(f"[{dom}] done", flush=True)
    json.dump(out, open(HERE / f"alarm_dense_results_s{seed}.json", "w"))


def wmean(rows, fn):
    w = np.array([x["w"] for x in rows], float)
    return float((w * np.array([fn(x) for x in rows], float)).sum() / w.sum()) if rows else float("nan")


def llm_table(seeds):
    res = {}
    fa_all = {r: [] for r in RULES}
    for ag in ev.AGENTS:
        v = {r: dict(recall=[], low=[], exposure=[], silent=[]) for r in RULES}
        for s in seeds:
            per = [json.load(open(f)) for f in glob.glob(str(HERE / f"alarm_llm_results_*_s{s}.json"))]
            assert len(per) == 4, (s, len(per))
            F = [x for p in per for x in p["agents"][ag]["faults"]]
            H = [x for x in F if x["harmful"]]
            L = [x for x in H if x["unsafe"] <= LOW]
            S = [x for x in F if not x["behav_visible"]]
            for r in RULES:
                v[r]["recall"].append(wmean(H, lambda x: x["det"][r]) * 100)
                v[r]["low"].append(wmean(L, lambda x: x["det"][r]) * 100)
                v[r]["exposure"].append(wmean(H, lambda x: x["exp"][r]))
                v[r]["silent"].append(wmean(S, lambda x: x["det"][r]) * 100)
                for p in per:
                    fa_all[r].append((s, p["domain"], ag, p["agents"][ag]["fa"][r]))
        res[ag] = v
    return res, fa_all


def summarize():
    out = {}
    for label, seeds in (("dev", DEV), ("eval", EVAL)):
        if not glob.glob(str(HERE / f"alarm_llm_results_*_s{seeds[0]}.json")):
            continue
        res, fa = llm_table(seeds)
        print(f"\n===== {label}: seeds {seeds} (LLM traffic)")
        print(f"{'rule':<6}{'8B':>8}{'70B':>8}{'gpt':>8}  {'low 8B':>8}{'70B':>8}{'gpt':>8}  {'exp 8B':>7}{'70B':>7}{'gpt':>7}  {'silent max':>10}  worst FA, combos above 1%, above 0")
        table = {}
        for r in RULES:
            rec = [float(np.mean(res[a][r]["recall"])) for a in ev.AGENTS]
            low = [float(np.mean(res[a][r]["low"])) for a in ev.AGENTS]
            exp = [float(np.mean(res[a][r]["exposure"])) for a in ev.AGENTS]
            sil = max(max(res[a][r]["silent"]) for a in ev.AGENTS)
            vals = [x[3] for x in fa[r]]
            table[r] = dict(recall=rec, low=low, exposure=exp, silent_max=sil, fa_worst=max(vals),
                            above1=sum(x > 0.01 for x in vals), above0=sum(x > 0 for x in vals), combos=len(vals),
                            cases=[x for x in fa[r] if x[3] > 0])
            print(f"{r:<6}" + "".join(f"{x:8.2f}" for x in rec) + "  " + "".join(f"{x:8.2f}" for x in low) + "  " +
                  "".join(f"{x:7.2f}" for x in exp) + f"  {sil:10.2f}  {max(vals) * 100:.1f}%  {table[r]['above1']}  {table[r]['above0']} / {len(vals)}")
        for r in RULES:
            if table[r]["cases"]:
                print("   ", r, table[r]["cases"])
        out[label] = table
        if label == "dev":
            ok = [r for r in CANDIDATES if table[r]["above1"] == 0]
            pick = max(ok, key=lambda r: np.mean(table[r]["recall"])) if ok else None
            print("qualifying", ok, "-> selected", pick)
            out["selected"] = pick
    files = sorted(glob.glob(str(HERE / "alarm_dense_results_s*.json")))
    if files:
        print("\n===== dense synthetic, seeds", [json.load(open(f))["seed"] for f in files])
        dense = {}
        for r in RULES:
            rec, low, shift = [], [], []
            for f in files:
                d = json.load(open(f))
                if d["seed"] not in EVAL:
                    continue
                H = [x for x in d["faults"] if x["harmful"]]
                rec.append(wmean(H, lambda x: x["det"][r]) * 100)
                low.append(wmean([x for x in H if x["unsafe"] <= LOW], lambda x: x["det"][r]) * 100)
                shift.append(max(v["shift"][r] for v in d["domains"].values()) * 100)
            if rec:
                dense[r] = dict(recall=float(np.mean(rec)), low=float(np.mean(low)), shift_worst=float(max(shift)))
                print(f"{r:<6} recall {np.mean(rec):7.2f}  low {np.mean(low):7.2f}  worst shift FA {max(shift):5.1f}%")
        out["dense"] = dense
        for f in files:
            d = json.load(open(f))
            if d["seed"] in DEV:
                H = [x for x in d["faults"] if x["harmful"]]
                print(f"replay check, dense seed {d['seed']}: base recall {wmean(H, lambda x: x['det']['base']) * 100:.4f}")
    pick = out.get("selected")
    if pick and "eval" in out and out.get("dense"):
        e, b = out["eval"], out["dense"]
        c1 = e[pick]["above1"] == 0
        loss = [x - y for x, y in zip(e["base"]["recall"], e[pick]["recall"])]
        c2 = all(x <= 2 for x in loss)
        c3a = b["base"]["recall"] - b[pick]["recall"] <= 1
        c3b = b[pick]["shift_worst"] <= b["base"]["shift_worst"]
        print(f"\n== criteria for {pick}\n  1 no combination above 1%: {c1}\n  2 LLM recall loss <= 2 on each agent: {c2} {np.round(loss, 2)}"
              f"\n  3a dense recall loss <= 1: {c3a} ({b['base']['recall'] - b[pick]['recall']:.2f})\n  3b worst shift FA <= base: {c3b}\n  PASS: {c1 and c2 and c3a and c3b}")
        out["criteria"] = dict(c1=bool(c1), c2=bool(c2), c3a=bool(c3a), c3b=bool(c3b), loss=loss)
    json.dump(out, open(HERE / "alarm_rule_summary.json", "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", choices=list(ev.DOMAIN_KEY.values()))
    ap.add_argument("--dense", action="store_true")
    ap.add_argument("--summary", action="store_true")
    args = ap.parse_args()
    if args.summary:
        summarize()
    elif args.dense:
        run_dense()
    else:
        run_llm({v: k for k, v in ev.DOMAIN_KEY.items()}[args.domain])
