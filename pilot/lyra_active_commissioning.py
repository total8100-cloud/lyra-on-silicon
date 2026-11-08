"""Small-scale test: does active learning during the commissioning sweep close the monitor's gap
to the redundancy ceiling? (PRD Sec 9.34)

Commissioning has free query access to the known-good shield but currently draws states uniformly.
Here the surrogate spends the same query budget near its own decision boundary: fit on an initial
uniform sample, then repeatedly score a large unlabelled pool with the surrogate (free), query the
shield on the states whose most uncertain output is closest to p = 0.5, and refit. Only shield
queries count toward the budget.

Fixed before any run:
  Seeds     0-4 were seen in the paper and are not used. Seed 10 is for choosing a configuration,
            seeds 11-13 are for evaluation, and the configuration is not changed after seed 10.
  Selection on seed 10, budget 100000, among A and B: the higher recall on low-severity faults,
            provided the worst false alarm under input shift is at most uniform's + 1 point and
            overall recall is not below uniform's. If neither qualifies, report no improvement and
            evaluate A anyway so that the negative result is on fresh seeds.
  Evaluation seeds 11-13, budgets 100000 and 25000, uniform vs. the selected configuration on the
            same calibration streams, shift streams, test streams and faults.

Usage: LYRA_SEED=<n> ACTIVE_BUDGET=<queries> ACTIVE_CONFIGS=uniform,A,B python pilot/lyra_active_commissioning.py
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lyra_behavioral_monitor as lbm  # noqa: E402

SEED = lbm.SEED
BUDGET = int(os.environ.get("ACTIVE_BUDGET", "100000"))
CONFIGS = os.environ.get("ACTIVE_CONFIGS", "uniform,A,B").split(",")
if SEED in range(5):
    raise SystemExit("seeds 0-4 were seen in the paper; use 10 for selection and 11-13 for evaluation")
if lbm.SURROGATE != "mlp" or lbm.GATE_ENV or lbm.P1_ENV:
    raise SystemExit("run with the paper's monitor configuration only")

# init: share of the budget drawn uniformly first; rounds: query rounds after that;
# mix: share of each round's queries chosen by uncertainty (the rest stay uniform)
SETTINGS = {"A": dict(init=0.4, rounds=3, mix=1.0), "B": dict(init=0.2, rounds=4, mix=0.5)}
POOL_FACTOR = 10
LOW = 20
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   f"active_results_s{SEED}_b{BUDGET}_{'-'.join(CONFIGS)}.json")


def commission(sh, cfg, rng):
    """Returns (fitted surrogate, number of shield queries)."""
    names = sh.names
    if cfg == "uniform":
        X = lbm.sample_states(rng, names, BUDGET)
        return lbm.SurrogateMonitor(SEED, "mlp").fit(X, sh.decide(X)), len(X)
    st = SETTINGS[cfg]
    n0 = int(BUDGET * st["init"])
    X = lbm.sample_states(rng, names, n0)
    Y = sh.decide(X)
    mon = lbm.SurrogateMonitor(SEED, "mlp").fit(X, Y)
    for r in range(st["rounds"]):
        batch = (BUDGET - len(X)) // (st["rounds"] - r)
        nu = int(batch * st["mix"])
        pool = lbm.sample_states(rng, names, POOL_FACTOR * max(nu, 1))
        score = np.abs(mon.predict(pool) - 0.5).min(axis=1)     # surrogate only: no shield query
        new = pool[np.argpartition(score, nu - 1)[:nu]]
        if batch > nu:
            new = np.concatenate([new, lbm.sample_states(rng, names, batch - nu)])
        X = np.concatenate([X, new])
        Y = np.concatenate([Y, sh.decide(new)])
        mon = lbm.SurrogateMonitor(SEED, "mlp").fit(X, Y)
    return mon, len(X)


def wmean(rows, fn):
    w = np.array([r["w"] for r in rows], float)
    return float((w * np.array([fn(r) for r in rows], float)).sum() / w.sum()) if rows else float("nan")


def summarize(res):
    out = {}
    F = res["faults"]
    H = [f for f in F if f["harmful"]]
    S = [f for f in F if not f["behav_visible"]]
    groups = {"ALL": H, "LOW": [f for f in H if f["unsafe"] <= LOW]}
    for c in ("T1", "T3", "D", "I", "C"):
        groups[c] = [f for f in H if f["cls"] == c]
    for cfg in CONFIGS:
        d = {g: wmean(rows, lambda r: r["det"][cfg]) * 100 for g, rows in groups.items()}
        d["exposure"] = wmean(H, lambda r: r["exp"][cfg])
        d["veto"] = wmean(H, lambda r: r["veto"][cfg])
        d["silent_alarm"] = wmean(S, lambda r: r["det"][cfg]) * 100
        dom = [v[cfg] for v in res["domains"].values()]
        d["shift_fa_max"] = max(x["shift_fa_max"] for x in dom) * 100
        d["acc_min"] = min(x["acc"] for x in dom) * 100
        d["overblock_max"] = max(x["overblock"] for x in dom) * 100
        d["queries"] = max(x["queries"] for x in dom)
        out[cfg] = d
    out["ceiling"] = {g: wmean(rows, lambda r: r["B5"]) * 100 for g, rows in groups.items()}
    return out


def run():
    t0 = time.time()
    rng = np.random.default_rng(50000 + SEED)      # shared streams
    res = dict(seed=SEED, budget=BUDGET, configs=CONFIGS, settings=SETTINGS, domains={}, faults=[])
    for dom, path in lbm.DOMAINS.items():
        sh = lbm.Shield(lbm.load_schema(path))
        names = sh.names
        mons, info = {}, {}
        for i, cfg in enumerate(CONFIGS):
            # the initial uniform draw differs per configuration; the streams below do not
            mons[cfg], q = commission(sh, cfg, np.random.default_rng(60000 + SEED * 100 + i))
            info[cfg] = dict(queries=q)
        cal = []
        for _ in range(lbm.N_CAL_STREAMS):
            Xs = lbm.sample_states(rng, names, lbm.T)
            cal.append((Xs, sh.decide(Xs)))
        cus = {}
        for cfg in CONFIGS:
            Ps = [(mons[cfg].predict(X), D) for X, D in cal]
            cus[cfg] = lbm.CusumMonitor().fit(Ps)
            info[cfg]["acc"] = float(np.mean(np.concatenate([(lbm.blocks(P) == D).ravel() for P, D in Ps])))
            info[cfg]["overblock"] = float(np.mean(np.concatenate([(lbm.blocks(P) & ~D).ravel() for P, D in Ps])))
            info[cfg]["p0"] = cus[cfg].p0
            info[cfg]["shift_fa_max"] = 0.0
        for dist in ("low", "high"):
            streams = []
            for _ in range(lbm.N_CAL_STREAMS):
                Xs = lbm.sample_states(rng, names, lbm.T, dist)
                streams.append((Xs, sh.decide(Xs)))
            for cfg in CONFIGS:
                fa = sum(cus[cfg].stat(mons[cfg].predict(X), D).max() >= cus[cfg].tau for X, D in streams)
                info[cfg]["shift_fa_max"] = max(info[cfg]["shift_fa_max"], fa / lbm.N_CAL_STREAMS)
        tests = []
        for _ in range(lbm.R):
            Xs = lbm.sample_states(rng, names, lbm.T)
            Es = sh.expr_q(Xs)
            tests.append((Xs, Es, {c: mons[c].predict(Xs) for c in CONFIGS}, sh.decide(Xs, E=Es)))
        res["domains"][dom] = info
        for f in lbm.enumerate_faults(sh):
            unsafe, over, b5 = [], [], []
            det = {c: [] for c in CONFIGS}
            exp = {c: [] for c in CONFIGS}
            veto = {c: [] for c in CONFIGS}
            for Xs, Es, Pm, D0 in tests:
                Df = sh.decide(Xs, f, E=Es)
                u = (D0 & ~Df).any(axis=1)
                unsafe.append(int(u.sum()))
                over.append(int((~D0 & Df).any(axis=1).sum()))
                b5.append(bool((Df != D0).any()))
                for c in CONFIGS:
                    a = lbm.first_alarm(cus[c].stat(Pm[c], Df), cus[c].tau)
                    det[c].append(a is not None)
                    exp[c].append(int(u[: (a + 1 if a is not None else lbm.T)].sum()))
                    veto[c].append(int((D0 & ~(Df | lbm.blocks(Pm[c]))).any(axis=1).sum()))
            res["faults"].append(dict(
                cls=f["cls"], w=f["w"], unsafe=float(np.mean(unsafe)), harmful=bool(np.mean(unsafe) > 0),
                behav_visible=bool(np.mean(unsafe) + np.mean(over) > 0), B5=float(np.mean(b5)),
                det={c: float(np.mean(det[c])) for c in CONFIGS},
                exp={c: float(np.mean(exp[c])) for c in CONFIGS},
                veto={c: float(np.mean(veto[c])) for c in CONFIGS}))
        print(f"[{dom}] {(time.time() - t0) / 60:.1f} min {info}", flush=True)
    s = summarize(res)
    res["summary"] = s
    with open(OUT, "w") as fh:
        json.dump(res, fh)
    print(json.dumps(s, indent=1), flush=True)


if __name__ == "__main__":
    run()
