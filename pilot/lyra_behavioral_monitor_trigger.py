"""Trigger-aware rerun of the behavioral integrity monitor (PRD Sec 9.20).

lyra_behavioral_monitor.py evaluates all 17 constraints on every state. Real Lyra
does not: each constraint carries a `trigger` in its YAML (e.g. iob_cap_large only
fires when the agent proposes large_bolus; `always` constraints fire every step),
and auto_compiler.py makes H_i = 0 whenever the proposed action is not the trigger.
So in deployment the monitor only observes a constraint's decision on the steps
where the agent proposed that constraint's trigger action -- far sparser evidence
-- while, equally, a faulted constraint can only cause an unsafe pass on exactly
those steps.

This script keeps the fault space, detectors, calibration and gating of
lyra_behavioral_monitor.py unchanged and changes only the decision stream:
  - each step is (x_t, a_t) with a_t the agent's proposed action; here a_t is drawn
    uniformly (the LLM-agent logs of the next step replace this);
  - constraint c is ACTIVE at step t iff trigger_c is None or a_t == trigger_c;
    inactive constraints produce no observation and cannot cause an unsafe pass;
  - the surrogate is one MLP per constraint (identical architecture and
    hyperparameters for all 17), trained only on that constraint's active steps
    of the same 100k-step fault-free log -- it never sees an inactive decision.

Usage: python pilot/lyra_behavioral_monitor_trigger.py   (writes behavioral_monitor_trigger_results.json)
"""
import json
import os
import sys

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lyra_behavioral_monitor as bm  # noqa: E402

from experiment.lyra.constraint_lang import load_schema  # noqa: E402  (path set by bm)

OUT_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"behavioral_monitor_trigger_results{bm.TAG}.json")


class TriggeredShield(bm.Shield):
    def __init__(self, schema):
        super().__init__(schema)
        self.actions = [a.name for a in schema.actions]
        self.trig = [None if s.trigger is None else self.actions.index(s.trigger) for s in self.specs]

    def sample_actions(self, rng, n):
        return rng.integers(0, len(self.actions), n)

    def active(self, a):
        """(n x C) mask: constraint c is evaluated at step t."""
        return np.stack([np.ones(len(a), bool) if t is None else (a == t) for t in self.trig], axis=1)


class PerConstraintSurrogate:
    """One MLP per constraint, each trained only on the steps where it was active."""

    def __init__(self, n_out):
        self.models = [None] * n_out

    def fit(self, X, D, A):
        for c in range(D.shape[1]):
            m = A[:, c]
            y = D[m, c].astype(int)
            if y.min() == y.max():                     # constant label: nothing to learn
                self.models[c] = float(y[0])
                continue
            # early stopping holds out a stratified 10% split, impossible when the minority
            # class has fewer than ~10 examples (seen only on short real-agent logs)
            es = np.bincount(y).min() >= 10
            mlp = MLPClassifier(hidden_layer_sizes=(64, 64), max_iter=400, early_stopping=es,
                                random_state=bm.SEED, tol=1e-5, n_iter_no_change=20).fit(X[m], y)
            if bm.SURROGATE.startswith("ens_"):     # improvement study: MLP + boosted-tree pair
                hgb = HistGradientBoostingClassifier(random_state=bm.SEED).fit(X[m], y)
                self.models[c] = (mlp, hgb)
            else:
                self.models[c] = mlp
        return self

    def predict(self, X):
        def col(mdl):
            if isinstance(mdl, float):
                return np.full(len(X), mdl)
            if isinstance(mdl, tuple):
                return bm.combine(bm.SURROGATE, *(mm.predict_proba(X)[:, 1] for mm in mdl))
            return mdl.predict_proba(X)[:, 1]
        return np.stack([col(mdl) for mdl in self.models], axis=1)


def masked(P, D, A):
    """Hide inactive entries from every detector: set the surrogate's prediction equal to
    the shield's own (inactive => PASS) there, so they can never count as disagreement
    or trigger a veto."""
    return np.where(A, P, D.astype(float))


class ActiveRateMonitor(bm.RateMonitor):
    """B3 restricted to observed decisions: windowed BLOCK rate among active steps."""

    def fit(self, D, A):
        self.p0 = np.array([D[A[:, c], c].mean() for c in range(D.shape[1])]).clip(1e-3, 1 - 1e-3)
        return self

    def stat(self, D, A):
        n = bm.window_sum(A.astype(float), bm.W)
        r = bm.window_sum((D & A).astype(float), bm.W) / np.maximum(n, 1)
        z = np.abs(r - self.p0) / np.sqrt(self.p0 * (1 - self.p0) / np.maximum(n, 1))
        z[n < bm.W // 4] = 0                    # require a minimum of observed decisions per output
        return z.max(axis=1)

    def calibrate(self, streams):
        maxes = np.array([self.stat(D, A).max() for D, A in streams])
        self.tau = float(np.quantile(maxes, 1 - bm.ALPHA)) + 1e-9
        return self.tau


def run():
    rng = np.random.default_rng(7 + 1000 * bm.SEED)
    shields = {dom: TriggeredShield(load_schema(p)) for dom, p in bm.DOMAINS.items()}
    theta_if = bm.train_theta_if([t for sh in shields.values() for t in sh.theta.tolist()],
                                 np.random.default_rng(1 + 1000 * bm.SEED))
    T = bm.T
    results = {"config": dict(N_TRAIN=bm.N_TRAIN, T=T, W=bm.W, R=bm.R, ALPHA=bm.ALPHA,
                              CONF_GATE=bm.CONF_GATE, actions="uniform"), "domains": {}, "faults": []}

    for dom, sh in shields.items():
        names = sh.names
        Xtr = bm.sample_states(rng, names, bm.N_TRAIN)
        Atr = sh.active(sh.sample_actions(rng, bm.N_TRAIN))
        Dtr = sh.decide(Xtr) & Atr
        mon = PerConstraintSurrogate(len(sh.specs)).fit(Xtr, Dtr, Atr)
        rate = ActiveRateMonitor().fit(Dtr, Atr)

        def stream(n, dist="uniform"):
            X = bm.sample_states(rng, names, n, dist)
            A = sh.active(sh.sample_actions(rng, n))
            E = sh.expr_q(X)
            D = sh.decide(X, E=E) & A
            return X, A, E, masked(mon.predict(X), D, A), D

        cal = [stream(T) for _ in range(bm.N_CAL_STREAMS)]
        win = bm.SurrogateMonitor()
        tau_s = win.calibrate([(P, D) for _, _, _, P, D in cal])
        cus = bm.CusumMonitor().fit([(P, D) for _, _, _, P, D in cal])
        tau_r = rate.calibrate([(D, A) for _, A, _, _, D in cal])
        act_frac = Atr.mean(axis=0)
        acc = [float((bm.blocks(P)[A[:, c], c] == D[A[:, c], c]).mean())
               for c in range(len(sh.specs)) for _, A, _, P, D in cal[:20]]

        shift = {}
        for dist in ("low", "high"):
            fa_s = fa_r = fa_c = 0
            for _ in range(bm.N_CAL_STREAMS):
                _, A, _, P, D = stream(T, dist)
                fa_s += win.stat(P, D).max() >= tau_s
                fa_c += cus.stat(P, D).max() >= cus.tau
                fa_r += rate.stat(D, A).max() >= tau_r
            shift[dist] = dict(surrogate=fa_s / bm.N_CAL_STREAMS, cusum=fa_c / bm.N_CAL_STREAMS,
                               rate=fa_r / bm.N_CAL_STREAMS)
        results["domains"][dom] = dict(
            constraints=[s.name for s in sh.specs], triggers=[s.trigger for s in sh.specs],
            active_fraction=act_frac.tolist(), surrogate_acc_active=float(np.mean(acc)),
            tau_surrogate=tau_s, cusum_p0=cus.p0, cusum_h=cus.tau, tau_rate=tau_r, shift_false_alarm=shift)
        print(f"[{dom}] active fraction per constraint={np.round(act_frac, 2).tolist()}  "
              f"surrogate acc(active)={np.mean(acc)*100:.3f}%  tau_s={tau_s} p0={cus.p0:.2e}  shift FA: {shift}",
              flush=True)

        tests = [stream(T) for _ in range(bm.R)]
        results["faults"].extend(evaluate_faults(dom, sh, tests, theta_if, win, cus, rate, act_frac))

    json.dump(results, open(OUT_JSON, "w"), indent=1)
    summarize(results)


def evaluate_faults(dom, sh, tests, theta_if, win, cus, rate, act_frac):
    """Replay every enumerated fault over the given fault-free test streams
    [(X, A, E, P, D), ...] and score every detector. Streams may differ in length."""
    out = []
    for f in bm.enumerate_faults(sh):
        rec = dict(domain=dom, cls=f["cls"], w=f["w"],
                   target=(sh.specs[f["c"]].name if "c" in f else f["var"]))
        if "c" in f:
            rec["target_active_fraction"] = float(act_frac[f["c"]])
        if f["kind"] == "theta":
            rel = (f["theta_q"] / bm.SCALE - sh.theta[f["c"]]) / (abs(sh.theta[f["c"]]) + 1e-6)
            rec["B1"] = bool(abs(rel) > 3 * bm.JITTER_SIGMA)
            rec["B2"] = bool(theta_if.predict([[rel]])[0] == -1)
        else:
            rec["B1"] = rec["B2"] = False
        per = {k: [] for k in ("unsafe", "overblock", "c_det", "c_lat", "c_exp", "s_det", "r_det",
                               "r_exp", "veto_unsafe", "veto_sev", "c_sev", "vg_unsafe", "vg_sev", "vg_over", "c_exp_v", "c_exp_vg")}
        for X, A, E, P0, D0 in tests:
            n = len(X)
            Df = sh.decide(X, f, E=E) & A
            P = np.where(A, P0, Df.astype(float))           # re-mask against the faulted decisions
            unsafe_t = (D0 & ~Df).any(axis=1)
            viol = np.where(sh.above, E - sh.theta_q, sh.theta_q - E) / bm.SCALE
            v = np.where(D0 & ~Df, viol, 0.0).max(axis=1)
            a_c = bm.first_alarm(cus.stat(P, Df), cus.tau)
            a_s = bm.first_alarm(win.stat(P, Df), win.tau)
            a_r = bm.first_alarm(rate.stat(Df, A), rate.tau)
            end_c = a_c + 1 if a_c is not None else n
            per["unsafe"].append(int(unsafe_t.sum()))
            per["overblock"].append(int((~D0 & Df).any(axis=1).sum()))
            per["c_det"].append(a_c is not None)
            per["c_lat"].append(a_c)
            per["c_exp"].append(int(unsafe_t[:end_c].sum()))
            per["c_sev"].append(float(v.max()) if (a_c is None and unsafe_t.any()) else 0.0)
            per["s_det"].append(a_s is not None)
            per["r_det"].append(a_r is not None)
            per["r_exp"].append(int(unsafe_t[: (a_r + 1 if a_r is not None else n)].sum()))
            surv = D0 & ~(Df | bm.blocks(P))
            per["veto_unsafe"].append(int(surv.any(axis=1).sum()))
            per["veto_sev"].append(float(np.where(surv, viol, 0.0).max()))
            # confident veto: override only when the surrogate is confident it should BLOCK
            vg = bm.blocks(P) & (np.abs(P - 0.5) > bm.CONF_GATE)
            surv_g = D0 & ~(Df | vg)
            per["vg_unsafe"].append(int(surv_g.any(axis=1).sum()))
            per["vg_sev"].append(float(np.where(surv_g, viol, 0.0).max()))
            per["vg_over"].append(int((vg & ~Df & ~D0 & A).any(axis=1).sum()))
            # what the deployed combination lets through: unsafe, not vetoed, before the alarm
            per["c_exp_v"].append(int(surv.any(axis=1)[:end_c].sum()))
            per["c_exp_vg"].append(int(surv_g.any(axis=1)[:end_c].sum()))
        rec.update(
            unsafe=float(np.mean(per["unsafe"])), overblock=float(np.mean(per["overblock"])),
            harmful=bool(np.mean(per["unsafe"]) > 0),
            behav_visible=bool(np.mean(per["unsafe"]) + np.mean(per["overblock"]) > 0),
            B3=float(np.mean(per["r_det"])), B3_exposure=float(np.mean(per["r_exp"])),
            B4=float(np.mean(per["s_det"])),
            B4c=float(np.mean(per["c_det"])), B4c_lat=[x for x in per["c_lat"] if x is not None],
            B4c_exposure=float(np.mean(per["c_exp"])), B4c_sev=float(np.max(per["c_sev"])),
            veto_unsafe=float(np.mean(per["veto_unsafe"])), veto_sev=float(np.max(per["veto_sev"])),
            vg_unsafe=float(np.mean(per["vg_unsafe"])), vg_sev=float(np.max(per["vg_sev"])),
            vg_over=float(np.mean(per["vg_over"])),
            B4c_exposure_veto=float(np.mean(per["c_exp_v"])), B4c_exposure_cveto=float(np.mean(per["c_exp_vg"])))
        out.append(rec)
    return out


def summarize(res):
    F = res["faults"]

    def wmean(rows, k):
        w = np.array([r["w"] for r in rows], float)
        return float((w * np.array([float(r[k]) for r in rows])).sum() / w.sum()) if rows else float("nan")

    H = [r for r in F if r["harmful"]]
    print(f"\n[trigger-aware] harmful faults: {len(H)}/{len(F)}")
    print(f"{'class':<6}{'#harm':>6}{'B1':>7}{'B2':>7}{'B3':>7}{'B4win':>7}{'B4cus':>7}"
          f"{'unsafe/T':>9}{'B3 exp':>8}{'B4c exp':>8}{'veto':>6}")
    for cls in ("T1", "T3", "D", "I", "C", "ALL"):
        rows = [r for r in H if cls == "ALL" or r["cls"] == cls]
        if rows:
            print(f"{cls:<6}{len(rows):6d}{wmean(rows,'B1')*100:6.1f}%{wmean(rows,'B2')*100:6.1f}%"
                  f"{wmean(rows,'B3')*100:6.1f}%{wmean(rows,'B4')*100:6.1f}%{wmean(rows,'B4c')*100:6.1f}%"
                  f"{wmean(rows,'unsafe'):9.1f}{wmean(rows,'B3_exposure'):8.2f}{wmean(rows,'B4c_exposure'):8.2f}"
                  f"{wmean(rows,'veto_unsafe'):6.2f}")
    print("Recall by severity (unsafe passes per T):")
    for lo, hi in [(0, 5), (5, 20), (20, 100), (100, np.inf)]:
        rows = [r for r in H if lo < r["unsafe"] <= hi]
        if rows:
            print(f"  ({lo},{hi}] n={len(rows):5d}  B2={wmean(rows,'B2')*100:5.1f}%  B3={wmean(rows,'B3')*100:5.1f}%"
                  f"  B4cus={wmean(rows,'B4c')*100:5.1f}%  exposure B4cus={wmean(rows,'B4c_exposure'):.2f}")
    print("B4cus recall vs. how often the faulted constraint is active (constraint-level faults):")
    for lo, hi in [(0, 0.3), (0.3, 1.01)]:
        rows = [r for r in H if "target_active_fraction" in r and lo <= r["target_active_fraction"] < hi]
        if rows:
            lat = [x for r in rows for x in r["B4c_lat"]]
            print(f"  active in [{lo},{hi}) of steps: n={len(rows)}  recall={wmean(rows,'B4c')*100:.1f}%  "
                  f"median latency={np.median(lat) if lat else float('nan'):.0f}  exposure={wmean(rows,'B4c_exposure'):.2f}")
    silent = [r for r in F if not r["behav_visible"]]
    print(f"Silent faults: {len(silent)}, B4cus alarm rate {wmean(silent,'B4c')*100 if silent else 0:.1f}%")
    print(f"Worst depth of an unsafe pass surviving the veto: {max(r['veto_sev'] for r in F):.4f}; "
          f"worst depth from a never-flagged fault: {max(r['B4c_sev'] for r in H):.4f}")
    lat = [x for r in H for x in r["B4c_lat"]]
    print(f"B4cus latency on harmful faults: median={np.median(lat):.0f}, p90={np.percentile(lat, 90):.0f}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--summary":
        summarize(json.load(open(OUT_JSON)))
    else:
        run()
