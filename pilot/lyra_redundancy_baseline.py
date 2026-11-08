"""Two follow-up questions a reviewer asks about the learned integrity monitor (PRD Sec 9.31).

Part A/B/C -- dual modular redundancy (DMR, baseline B5): why not re-evaluate the compiled
specification on the host and compare decisions? B5 alarms on the first decision that differs.
  A  fault in the shield, healthy reference: B5 alarms on every stream in which the fault changes
     a decision, so its recall is the CEILING for any detector that sees only decisions. It is
     below 100% because a rare fault changes no decision in some test streams. Measured here for
     that ceiling, latency and exposure, on the same streams as the monitor.
  B  fault in the reference, healthy shield: does a corrupted copy ever cause an unsafe pass when
     the executed decision is the OR of the two? Is it silent, or does it alarm?
  C  a floating-point reference instead of a bit-exact fixed-point one: false alarms on fault-free
     streams, in-distribution and under input shift.
  Plus host cost: wall-clock per decision and the amount of stored state.

Part D/E -- faults in the monitor itself: bit flips in the MLP's float64 weights and intercepts.
  D  healthy shield: false alarms, NaN outputs, extra veto blocks. Flips are sampled, stratified by
     bit group, and population-weighted by the summary script.
  E  double fault: flips that raised no alarm in D, combined with a panel of harmful shield faults:
     how much recall is lost?

The random stream of lyra_behavioral_monitor.run() is replayed exactly, so the monitor's numbers
here are checked fault by fault against that run's result file before anything is written.

Usage: LYRA_SEED=<n> python pilot/lyra_redundancy_baseline.py   (writes redundancy_results_s<n>.json)
       LYRA_RB_QUICK=1 reduces the flip counts for a smoke test (writes ..._quick.json).
"""
import json
import os
import struct
import sys
import time

import numpy as np
from scipy.special import expit

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lyra_behavioral_monitor as lbm  # noqa: E402

SEED = lbm.SEED
QUICK = bool(os.environ.get("LYRA_RB_QUICK"))
if lbm.SURROGATE != "mlp" or lbm.GATE_ENV or lbm.P1_ENV:
    raise SystemExit("run with the paper's monitor configuration only")

GROUPS = {"sign": [63], "exponent": list(range(52, 63)),
          "mantissa_high": list(range(40, 52)), "mantissa_low": list(range(0, 40))}
N_FLIPS = 100 if QUICK else 5000      # per bit group and domain (healthy shield)
N_DOUBLE = 5 if QUICK else 100        # flips per bit group in the double-fault test
N_PANEL = 10                          # harmful shield faults per fault class in the panel
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   f"redundancy_results_s{SEED}{'_quick' if QUICK else ''}.json")


# ---------------------------------------------------------------- MLP forward pass with one flipped bit
def flip_bit(v, bit):
    return struct.unpack("<d", struct.pack("<Q", struct.unpack("<Q", struct.pack("<d", float(v)))[0] ^ (1 << bit)))[0]


class Forward:
    """Same computation as MLPClassifier._forward_pass_fast, restarting from the first affected
    layer. Checked against predict_proba on the unmodified model and on real flips (check_faithful)."""

    def __init__(self, clf, X):
        self.W, self.b = clf.coefs_, clf.intercepts_
        self.n = len(self.W)
        if clf.activation != "relu" or clf.out_activation_ != "logistic":
            raise SystemExit("unexpected MLP activations")
        self.acts = [X]
        a = X
        for k in range(self.n - 1):
            a = np.maximum(a @ self.W[k] + self.b[k], 0)     # as sklearn: infinities stay infinite
            self.acts.append(a)
        self.P0 = expit(self.acts[-1] @ self.W[-1] + self.b[-1])
        self.params = [("W", k, i, j) for k in range(self.n)
                       for i in range(self.W[k].shape[0]) for j in range(self.W[k].shape[1])]
        self.params += [("b", k, j, None) for k in range(self.n) for j in range(self.b[k].shape[0])]

    def run(self, param, bit):
        kind, k, i, j = param
        arr, idx = (self.W[k], (i, j)) if kind == "W" else (self.b[k], (i,))
        old = arr[idx]
        arr[idx] = flip_bit(old, bit)
        try:
            a = self.acts[k]
            for m in range(k, self.n):
                a = a @ self.W[m] + self.b[m]
                if m < self.n - 1:
                    a = np.maximum(a, 0)
            return expit(a)
        finally:
            arr[idx] = old


def check_faithful(clf, fwd, X, rng):
    assert np.array_equal(clf.predict_proba(X), fwd.P0)
    # half of the checks on the high exponent bits, where infinities and NaNs appear
    bits = [int(b) for b in rng.integers(64, size=100)] + [int(b) for b in rng.integers(58, 63, size=100)]
    for bit in bits:
        param = fwd.params[rng.integers(len(fwd.params))]
        kind, k, i, j = param
        arr, idx = (clf.coefs_[k], (i, j)) if kind == "W" else (clf.intercepts_[k], (i,))
        mine = fwd.run(param, bit)
        old = arr[idx]
        arr[idx] = flip_bit(old, bit)
        real = clf.predict_proba(X)
        arr[idx] = old
        assert np.array_equal(real, mine, equal_nan=True)


def cusum_alarm(cus, P, D):
    if not lbm.disagree(P, D).any():      # no evidence at all: the statistic stays at zero
        return None
    return lbm.first_alarm(cus.stat(P, D), cus.tau)


# ---------------------------------------------------------------- main
def run():
    np.seterr(all="ignore")
    t0 = time.time()
    # identical random stream to lyra_behavioral_monitor.run()
    rng = np.random.default_rng(7 + 1000 * SEED)
    own = np.random.default_rng(90210 + SEED)      # everything new draws from here, never from rng
    reference = json.load(open(lbm.OUT_JSON))["faults"]
    shields = {dom: lbm.Shield(lbm.load_schema(path)) for dom, path in lbm.DOMAINS.items()}
    res = dict(seed=SEED, config=dict(N_FLIPS=N_FLIPS, N_DOUBLE=N_DOUBLE, N_PANEL=N_PANEL, quick=QUICK),
               domains={}, faults=[])
    mismatches = checked = 0

    for dom, sh in shields.items():
        names = sh.names
        Xtr = lbm.sample_states(rng, names, lbm.N_TRAIN)
        Ytr = sh.decide(Xtr)
        mon = lbm.SurrogateMonitor().fit(Xtr, Ytr)

        cal = []
        for _ in range(lbm.N_CAL_STREAMS):
            Xs = lbm.sample_states(rng, names, lbm.T)
            cal.append((Xs, mon.predict(Xs), sh.decide(Xs)))
        mon.calibrate([(P, D) for _, P, D in cal])
        cus = lbm.CusumMonitor().fit([(P, D) for _, P, D in cal])

        # ---- Part C: floating-point reference, no fault anywhere
        def float_decide(X):
            vals = [np.broadcast_to(np.asarray(fn(*X.T), dtype=float), (len(X),)) for fn in sh.fns]
            E = np.stack(vals, axis=1)
            return np.where(sh.above, E > sh.theta, E < sh.theta)

        def float_stats(streams):
            per = [int((float_decide(X) != D).any(axis=1).sum()) for X, D in streams]
            return dict(streams=len(per), streams_with_mismatch=int(np.sum(np.array(per) > 0)),
                        mismatches_per_stream=float(np.mean(per)))

        float_ref = dict(in_distribution=float_stats([(X, D) for X, _, D in cal]))
        for dist in ("low", "high"):
            shifted = []
            for _ in range(lbm.N_CAL_STREAMS):
                Xs = lbm.sample_states(rng, names, lbm.T, dist)
                shifted.append((Xs, sh.decide(Xs)))
            float_ref[dist] = float_stats(shifted)

        tests = []
        for _ in range(lbm.R):
            Xs = lbm.sample_states(rng, names, lbm.T)
            Es = sh.expr_q(Xs)
            tests.append((Xs, Es, mon.predict(Xs), sh.decide(Xs, E=Es)))

        # ---- host cost (one single-threaded process; a Python datapath, so only a rough ratio)
        Xc = tests[0][0]
        t = time.perf_counter()
        for _ in range(50):
            mon.predict(Xc)
        t_mlp = (time.perf_counter() - t) / (50 * len(Xc))
        t = time.perf_counter()
        for _ in range(50):
            sh.decide(Xc)
        t_ref = (time.perf_counter() - t) / (50 * len(Xc))
        n_params = sum(w.size for w in mon.clf.coefs_) + sum(b.size for b in mon.clf.intercepts_)

        # ---- Parts A and B: every enumerated fault, on the shield or on the reference
        faults = lbm.enumerate_faults(sh)
        harmful_by_class = {}
        for f in faults:
            per = dict(unsafe=[], over=[], c_det=[], c_exp=[], d_det=[], d_lat=[], d_exp=[])
            Dfs = []
            for Xs, Es, P, D0 in tests:
                Df = sh.decide(Xs, f, E=Es)
                Dfs.append(Df)
                unsafe_t = (D0 & ~Df).any(axis=1)
                per["unsafe"].append(int(unsafe_t.sum()))
                per["over"].append(int((~D0 & Df).any(axis=1).sum()))
                a_c = lbm.first_alarm(cus.stat(P, Df), cus.tau)
                per["c_det"].append(a_c is not None)
                per["c_exp"].append(int(unsafe_t[: (a_c + 1 if a_c is not None else lbm.T)].sum()))
                a_d = lbm.first_alarm((Df != D0).any(axis=1).astype(int), 1)
                per["d_det"].append(a_d is not None)
                if a_d is not None:
                    per["d_lat"].append(a_d)
                per["d_exp"].append(int(unsafe_t[: (a_d + 1 if a_d is not None else lbm.T)].sum()))
            rec = dict(domain=dom, cls=f["cls"], w=f["w"],
                       target=(sh.specs[f["c"]].name if "c" in f else f["var"]),
                       unsafe=float(np.mean(per["unsafe"])), overblock=float(np.mean(per["over"])),
                       harmful=bool(np.mean(per["unsafe"]) > 0),
                       behav_visible=bool(np.mean(per["unsafe"]) + np.mean(per["over"]) > 0),
                       B4c=float(np.mean(per["c_det"])), B4c_exposure=float(np.mean(per["c_exp"])),
                       B5=float(np.mean(per["d_det"])), B5_exposure=float(np.mean(per["d_exp"])),
                       B5_lat=float(np.mean(per["d_lat"])) if per["d_lat"] else None)
            # the fault on the REFERENCE, executed decision = shield OR reference (fail-closed):
            # the OR contains every fault-free BLOCK, so the count below must be zero
            rec["ref_unsafe_exec"] = int(sum(((D0 & ~(D0 | Df)).any(axis=1).sum())
                                             for (_, _, _, D0), Df in zip(tests, Dfs)))
            res["faults"].append(rec)

            old = reference[checked]
            if not (old["domain"] == dom and old["cls"] == f["cls"] and old["harmful"] == rec["harmful"]
                    and old["B4c"] == rec["B4c"] and old["B4c_exposure"] == rec["B4c_exposure"]):
                mismatches += 1
            checked += 1
            if rec["harmful"] and tests:
                # panel candidates must be harmful in the stream used by the double-fault test
                if (tests[0][3] & ~Dfs[0]).any():
                    harmful_by_class.setdefault(f["cls"], []).append(f)

        # ---- Part D: monitor weight flips, healthy shield, first test stream
        Xs, Es, _, D0 = tests[0]
        fwd = Forward(mon.clf, Xs)
        check_faithful(mon.clf, fwd, Xs, own)
        base_over = int((lbm.blocks(fwd.P0) & ~D0).any(axis=1).sum())
        groups = {}
        quiet = {}
        for gname, bits in GROUPS.items():
            g = dict(n=0, alarm=0, nan=0, changed=0, extra_block_steps=0.0, lost_block_steps=0.0)
            quiet[gname] = []
            for _ in range(N_FLIPS):
                param = fwd.params[own.integers(len(fwd.params))]
                bit = int(own.choice(bits))
                P = fwd.run(param, bit)
                has_nan = bool(np.isnan(P).any())
                over = int((lbm.blocks(P) & ~D0).any(axis=1).sum())
                alarm = cusum_alarm(cus, P, D0) is not None
                g["n"] += 1
                g["alarm"] += alarm
                g["nan"] += has_nan
                g["changed"] += bool((lbm.blocks(P) != lbm.blocks(fwd.P0)).any())
                g["extra_block_steps"] += max(over - base_over, 0)
                if not alarm:
                    quiet[gname].append((param, bit, has_nan))
            groups[gname] = g

        # ---- Part E: quiet flips (no alarm above) combined with harmful shield faults
        panel = []
        for cls, fs in sorted(harmful_by_class.items()):
            idx = own.permutation(len(fs))[:N_PANEL]
            panel.extend(fs[i] for i in idx)
        Dps = [sh.decide(Xs, f, E=Es) for f in panel]
        base_det = np.array([lbm.first_alarm(cus.stat(fwd.P0, Df), cus.tau) is not None for Df in Dps])
        double = {}
        for gname, qs in quiet.items():
            pick = own.permutation(len(qs))[:N_DOUBLE]
            det_sum = base_sum = 0
            n_flips = nan_flips = lost_all = lost_with_nan = 0
            for ix in pick:
                param, bit, has_nan = qs[ix]
                P = fwd.run(param, bit)
                det = np.array([cusum_alarm(cus, P, Df) is not None for Df in Dps])
                det_sum += int(det.sum())
                base_sum += int(base_det.sum())
                n_flips += 1
                nan_flips += has_nan
                lost = bool(base_det.any() and not (det & base_det).any())
                lost_all += lost
                lost_with_nan += lost and has_nan
            double[gname] = dict(flips=n_flips, nan_flips=nan_flips, detected=det_sum,
                                 baseline_detected=base_sum, flips_losing_every_detection=lost_all,
                                 losing_flips_with_nan=lost_with_nan,
                                 quiet_share=len(qs) / N_FLIPS)

        res["domains"][dom] = dict(
            float_reference=float_ref, t_mlp_us=t_mlp * 1e6, t_reference_us=t_ref * 1e6,
            n_surrogate_params=int(n_params), surrogate_kbytes_f64=n_params * 8 / 1024,
            n_thresholds=len(sh.specs), threshold_bytes_q412=len(sh.specs) * 2,
            base_overblock_steps=base_over, weight_flips=groups, double_fault=double,
            panel_size=len(panel), panel_baseline_detected=int(base_det.sum()))
        print(f"[{dom}] done at {(time.time() - t0) / 60:.1f} min; mismatches so far {mismatches}/{checked}", flush=True)

    res["replay_check"] = dict(checked=checked, mismatches=mismatches)
    if len(reference) != checked:
        raise SystemExit(f"fault count differs from the published run: {checked} vs {len(reference)}")
    with open(OUT, "w") as fh:
        json.dump(res, fh)
    print(f"replay check: {mismatches} mismatches of {checked}; wrote {OUT}", flush=True)
    if mismatches:
        sys.exit(2)


if __name__ == "__main__":
    run()
