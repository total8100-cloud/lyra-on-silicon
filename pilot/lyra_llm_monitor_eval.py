"""Evaluate the trigger-aware integrity monitor on real LLM-agent decision logs (PRD Sec 9.21).

Inputs: pilot/llm_logs/{agent}_{domain}.jsonl from lyra_llm_agent_logs.py -- the states an
LLM agent visited and the actions it proposed, in Paper 1's environments, under the Lyra
shield, with Paper 1's prompts.

Protocol (per domain):
  - two training regimes: "agent" (below) and "commission" (synthetic sweep of the
    known-good shield before any agent runs; see build_monitor).
  - training agent = Llama-3.1-8B. Its episodes are split by index into train (ep % 4 in
    {0, 1}), calibration (ep % 4 == 2) and test (ep % 4 == 3). The surrogate (one MLP per
    constraint, as in lyra_behavioral_monitor_trigger.py) is fitted on the train episodes
    only; every alarm threshold is calibrated on the calibration episodes only.
  - cross-agent test = Llama-3.3-70B and gpt-oss-20b: all of their episodes are test data;
    neither model contributes anything to training or calibration (no retraining).
  - every evaluation stream, for every agent, is 2,000 steps assembled by resampling that
    agent's held-out episodes with replacement (episodes kept intact), so agents are
    compared on equal-length streams.
  - the fault space, detectors, confidence gate and calibration rule are unchanged.

Usage: python pilot/lyra_llm_monitor_eval.py --domain medical      (one domain per process)
       python pilot/lyra_llm_monitor_eval.py --summary              (after all four)
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import lyra_behavioral_monitor as bm  # noqa: E402
import lyra_behavioral_monitor_trigger as tg  # noqa: E402

from experiment.lyra.constraint_lang import load_schema  # noqa: E402

LOG_DIR = HERE / "llm_logs"
TRAIN_AGENT = "llama8b"
AGENTS = ["llama8b", "llama70b", "gptoss20b"]
DOMAIN_KEY = {"Financial": "financial", "Medical": "medical", "Cyber": "cyber", "Power": "power"}
REGIMES = ("agent", "commission_small", "commission")
MIN_ACTIVE = 50        # fewer triggered training steps than this -> no surrogate for that constraint
T = 2000
N_CAL = 200


def load_episodes(agent, domain):
    eps = {}
    for line in open(LOG_DIR / f"{agent}_{domain}.jsonl"):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:           # a line still being written
            continue
        eps.setdefault(r["ep"], []).append(r)
    full = {k: sorted(v, key=lambda s: s["t"]) for k, v in eps.items() if len(v) == 8}
    return {k: (np.array([s["x"] for s in v]), np.array([s["a"] for s in v]),
                np.array([s["valid"] for s in v]))
            for k, v in sorted(full.items())}


class SparseSurrogate(tg.PerConstraintSurrogate):
    """Per-constraint surrogate that abstains (p = 0.49: never confident, never vetoes)
    for a constraint with too few triggered training steps to learn from."""

    def fit(self, X, D, A):
        self.n_active = A.sum(axis=0)
        keep = self.n_active >= MIN_ACTIVE
        # fit only the learnable constraints (the parent class cannot handle an empty one)
        super().fit(X, D[:, keep], A[:, keep])
        fitted = iter(self.models[:keep.sum()])
        self.models = [next(fitted) if k else 0.49 for k in keep]
        return self


def concat(eps, keys):
    X = np.concatenate([eps[k][0] for k in keys])
    a = np.concatenate([eps[k][1] for k in keys])
    return X, a


def resampled(eps, keys, rng, n=T):
    out, total = [], 0
    while total < n:
        k = keys[rng.integers(len(keys))]
        out.append(k)
        total += len(eps[k][0])
    X, a = concat(eps, out)
    return X[:n], a[:n]


def build_monitor(regime, sh, logs, split, rng):
    """'agent': fitted and calibrated on the training agent's own fault-free logs.
    'commission': fitted and calibrated on a fault-free commissioning sweep of the shield
    (synthetic states, every action), before any agent has run; no agent data at all.
    'commission_small': the same sweep cut to the agent regime's training size, to separate
    the effect of data volume from that of coverage."""
    n_agent = len(concat(logs[TRAIN_AGENT], split["train"])[0])
    if regime == "agent":
        X, a = concat(logs[TRAIN_AGENT], split["train"])
        mon_cls = SparseSurrogate
    else:
        n = bm.N_TRAIN if regime == "commission" else n_agent
        X = bm.sample_states(rng, sh.names, n)
        a = sh.sample_actions(rng, n)
        mon_cls = SparseSurrogate                     # every constraint has ~20k+ triggered steps here
    A = sh.active(a)
    D = sh.decide(X) & A
    mon = mon_cls(len(sh.specs)).fit(X, D, A)
    rate = tg.ActiveRateMonitor().fit(D, A)
    if regime == "agent":
        cal_raw = [resampled(logs[TRAIN_AGENT], split["cal"], rng) for _ in range(N_CAL)]
    else:
        cal_raw = [(bm.sample_states(rng, sh.names, T), sh.sample_actions(rng, T)) for _ in range(N_CAL)]
    cal = [materialize(sh, mon, Xc, ac) for Xc, ac in cal_raw]
    win = bm.SurrogateMonitor()
    win.calibrate([(P, D_) for _, _, _, P, D_ in cal])
    cus = bm.CusumMonitor().fit([(P, D_) for _, _, _, P, D_ in cal])
    rate.calibrate([(D_, A_) for _, A_, _, _, D_ in cal])
    info = dict(n_train_steps=int(len(X)), active_train=mon.n_active.tolist(),
                abstained=[s.name for s, m in zip(sh.specs, mon.models) if m == 0.49],
                tau_window=win.tau, cusum_p0=cus.p0, cusum_h=cus.tau)
    return mon, win, cus, rate, info


def materialize(sh, mon, X, a):
    A = sh.active(a)
    E = sh.expr_q(X)
    D = sh.decide(X, E=E) & A
    return X, A, E, tg.masked(mon.predict(X), D, A), D


def run_domain(dom):
    d = DOMAIN_KEY[dom]
    rng = np.random.default_rng(11 + 1000 * bm.SEED)
    sh = tg.TriggeredShield(load_schema(bm.DOMAINS[dom]))
    theta_if = bm.train_theta_if(
        [t for p in bm.DOMAINS.values() for t in bm.Shield(load_schema(p)).theta.tolist()],
        np.random.default_rng(1 + 1000 * bm.SEED))
    logs = {ag: load_episodes(ag, d) for ag in AGENTS}
    ka = sorted(logs[TRAIN_AGENT])
    split = {"train": [k for k in ka if k % 4 in (0, 1)], "cal": [k for k in ka if k % 4 == 2],
             "test": [k for k in ka if k % 4 == 3]}
    held_out = {TRAIN_AGENT: split["test"], **{ag: sorted(logs[ag]) for ag in AGENTS if ag != TRAIN_AGENT}}
    # the same raw evaluation streams are used for both regimes
    raw_fa = {ag: [resampled(logs[ag], held_out[ag], rng) for _ in range(N_CAL)] for ag in AGENTS}
    raw_test = {ag: [resampled(logs[ag], held_out[ag], rng) for _ in range(bm.R)] for ag in AGENTS}

    res = dict(domain=dom, regimes={})
    for regime in REGIMES:
        mon, win, cus, rate, info = build_monitor(regime, sh, logs, split, rng)
        out = dict(info, agents={}, faults={})
        for ag in AGENTS:
            X, a = concat(logs[ag], held_out[ag])
            _, A, _, P, D = materialize(sh, mon, X, a)
            valid = np.concatenate([logs[ag][k][2] for k in held_out[ag]])
            conf = A & (np.abs(P - 0.5) > bm.CONF_GATE)
            prof = dict(
                steps=int(len(X)), invalid_parse=float(1 - valid.mean()),
                action_dist=np.bincount(a, minlength=len(sh.actions)).tolist(),
                active_fraction=A.mean(axis=0).tolist(),
                risky_proposal_rate=float(D.any(axis=1).mean()),     # the fault-free shield blocks something
                surrogate_acc_active=float((bm.blocks(P)[A] == D[A]).mean()),
                confident_error_rate=float(((bm.blocks(P) != D) & conf).any(axis=1).mean()),
                veto_overblock_rate=float((bm.blocks(P) & ~D & A).any(axis=1).mean()),
                conf_veto_overblock_rate=float((bm.blocks(P) & ~D & conf).any(axis=1).mean()))
            fa = {"cusum": 0, "window": 0, "rate": 0}
            for Xr, ar in raw_fa[ag]:
                _, Ar, _, Pr, Dr = materialize(sh, mon, Xr, ar)
                fa["cusum"] += cus.stat(Pr, Dr).max() >= cus.tau
                fa["window"] += win.stat(Pr, Dr).max() >= win.tau
                fa["rate"] += rate.stat(Dr, Ar).max() >= rate.tau
            prof["false_alarm"] = {k: v / N_CAL for k, v in fa.items()}
            out["agents"][ag] = prof
            print(f"[{dom}/{regime}/{ag}] steps={prof['steps']} invalid={prof['invalid_parse']:.1%} "
                  f"risky={prof['risky_proposal_rate']:.1%} acc={prof['surrogate_acc_active']:.3%} "
                  f"conf_err={prof['confident_error_rate']:.3%} FA={prof['false_alarm']}", flush=True)
            tests = [materialize(sh, mon, Xr, ar) for Xr, ar in raw_test[ag]]
            act_frac = np.mean([A_.mean(axis=0) for _, A_, _, _, _ in tests], axis=0)
            out["faults"][ag] = tg.evaluate_faults(dom, sh, tests, theta_if, win, cus, rate, act_frac)
        res["regimes"][regime] = out
    json.dump(res, open(HERE / f"llm_monitor_results_{d}{bm.TAG}.json", "w"), indent=1)


def summarize():
    def wmean(rows, k):
        w = np.array([r["w"] for r in rows], float)
        return float((w * np.array([float(r[k]) for r in rows])).sum() / w.sum()) if rows else float("nan")

    R = {d: json.load(open(HERE / f"llm_monitor_results_{d}{bm.TAG}.json")) for d in DOMAIN_KEY.values()
         if (HERE / f"llm_monitor_results_{d}{bm.TAG}.json").exists()}
    for regime in REGIMES:
        if not all(regime in r["regimes"] for r in R.values()):
            continue
        print(f"\n===== regime: {regime} =====")
        for d, r in R.items():
            g = r["regimes"][regime]
            print(f"{d}: train steps={g['n_train_steps']} triggered per constraint={g['active_train']} "
                  f"abstained={g['abstained']} p0={g['cusum_p0']:.2e}")
        print(f"{'agent':<10}{'#harm':>6}{'B2':>7}{'B3':>7}{'B4cus':>7}{'unsafe/T':>9}{'B4 exp':>8}"
              f"{'veto':>7}{'lat med':>8}{'FA cus':>8}{'FA rate':>8}{'overblk':>8}")
        for ag in AGENTS:
            H = [f for r in R.values() for f in r["regimes"][regime]["faults"][ag] if f["harmful"]]
            lat = [x for f in H for x in f["B4c_lat"]]
            fa_c = np.mean([r["regimes"][regime]["agents"][ag]["false_alarm"]["cusum"] for r in R.values()])
            fa_r = np.mean([r["regimes"][regime]["agents"][ag]["false_alarm"]["rate"] for r in R.values()])
            ob = np.mean([r["regimes"][regime]["agents"][ag]["veto_overblock_rate"] for r in R.values()])
            print(f"{ag:<10}{len(H):6d}{wmean(H,'B2')*100:6.1f}%{wmean(H,'B3')*100:6.1f}%{wmean(H,'B4c')*100:6.1f}%"
                  f"{wmean(H,'unsafe'):9.1f}{wmean(H,'B4c_exposure'):8.2f}{wmean(H,'veto_unsafe'):7.2f}"
                  f"{np.median(lat) if lat else float('nan'):8.0f}{fa_c*100:7.1f}%{fa_r*100:7.1f}%{ob*100:7.2f}%")
            vs = [f["veto_sev"] for r in R.values() for f in r["regimes"][regime]["faults"][ag]]
            per_dom = {d: (r["regimes"][regime]["agents"][ag]["false_alarm"]["cusum"],
                           round(r["regimes"][regime]["agents"][ag]["risky_proposal_rate"], 3)) for d, r in R.items()}
            print(f"{'':<10}worst veto-surviving depth={max(vs) if vs else 0:.4f}  per-domain (FA cusum, risky rate)={per_dom}")
            obg = np.mean([r["regimes"][regime]["agents"][ag]["conf_veto_overblock_rate"] for r in R.values()])
            vgs = [f["vg_sev"] for r in R.values() for f in r["regimes"][regime]["faults"][ag]]
            print(f"{'':<10}unsafe & not vetoed before alarm: veto={wmean(H,'B4c_exposure_veto'):.2f}"
                  f"  confident veto={wmean(H,'B4c_exposure_cveto'):.2f}")
            print(f"{'':<10}confident veto: residual unsafe={wmean(H,'vg_unsafe'):.2f}  worst depth={max(vgs) if vgs else 0:.4f}"
                  f"  fault-free over-block={obg*100:.2f}%")



if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", choices=list(DOMAIN_KEY.values()))
    ap.add_argument("--summary", action="store_true")
    args = ap.parse_args()
    if args.summary:
        summarize()
    else:
        run_domain({v: k for k, v in DOMAIN_KEY.items()}[args.domain])
