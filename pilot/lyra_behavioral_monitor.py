"""Behavioral (black-box) integrity monitor for the compiled Lyra shield --
the redesigned AI layer (PRD Sec 9.19), replacing the theta-deviation
IsolationForest of lyra_ai_drift_detector.py as the paper's AI contribution.

Why a redesign: the old detector's only feature was
  (theta_observed - theta_nominal) / |theta_nominal|,
which (a) requires reading the theta register out of the datapath and
(b) requires a trusted stored copy of theta_nominal -- and, being one-dimensional,
is functionally equivalent to a single `|dev| > eps` comparator (this script
measures that directly as baseline B1). It is also structurally blind to every
fault that does not live in the theta register.

Monitor (ours): a surrogate of the shield's decision function learned ONLY from
fault-free decision logs (x_t, d_t), one multi-label MLP per domain predicting
the whole decision vector (one output per constraint), with identical
architecture/hyperparameters for all 4 domains. It needs neither the compiled
spec, nor theta, nor any register access -- only what any host already sees: the
state it sends to the shield and the BLOCK/PASS decisions that come back. At run
time it is used two ways:
  (1) integrity alarm: count surrogate/shield disagreements in a sliding window
      of W decisions; the alarm threshold is calibrated automatically on
      fault-free streams to a 1% stream-level false-alarm rate (no hand tuning);
  (2) fail-closed veto: if the surrogate predicts BLOCK while the shield says
      PASS, the decision is overridden to BLOCK (can only add blocks, so it never
      weakens the shield; its cost is measured as fault-free over-blocking).

Baselines:
  B1 theta-rule   |rel. theta dev.| > 3% (3 sigma of the jitter model)  [needs theta readout + theta_nominal]
  B2 theta-IF     the paper's previous IsolationForest                [needs theta readout + theta_nominal]
  B3 rate-monitor windowed BLOCK-rate z-test per output, calibrated  [black-box, no ML]
  B4 surrogate    this monitor                                      [black-box, learned]

Fault classes (16-bit Q4.12, every case enumerated, not sampled):
  T1  theta register, no ECC, single-bit flip                (17 x 16)
  T3  theta register, SECDED, triple-bit -> miscorrected      (17 x 1330 triples, miscorrected subset)
  D   expression-result register stuck-at-0/1               (17 x 16 x 2)
  I   input register stuck-at-0/1 (shared by constraints)    (18 vars x 16 x 2)
  C   comparator output stuck-at-PASS / stuck-at-BLOCK       (17 x 2)

Usage: python pilot/lyra_behavioral_monitor.py   (writes pilot/behavioral_monitor_results.json)
"""
import os as _os
_REPO = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
LYRA_BACKEND = _os.environ.get("LYRA_BACKEND", _os.path.join(_REPO, "lyra_backend"))
import sys
sys.path.insert(0, LYRA_BACKEND)

import itertools
import json
import os

import numpy as np
import sympy as sp
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

# Repeat-run controls (PRD Sec 9.25). Seed 0 with the MLP reproduces the published run;
# other seeds re-draw every random choice (data streams, surrogate initialisation).
SEED = int(os.environ.get("LYRA_SEED", "0"))
SURROGATE = os.environ.get("LYRA_SURROGATE", "mlp")      # mlp | hgb | knn | logreg | ens_avg | ens_min
# improvement study (PRD Sec 9.26): confidence gate and CUSUM design rate, defaults = paper
GATE_ENV = os.environ.get("LYRA_CONF_GATE")
P1_ENV = os.environ.get("LYRA_P1")
TAG = "" if (SEED == 0 and SURROGATE == "mlp") else f"_s{SEED}_{SURROGATE}"
if GATE_ENV:
    TAG += f"_g{GATE_ENV}" if TAG else f"_s0_mlp_g{GATE_ENV}"
if P1_ENV:
    TAG += f"_p{P1_ENV}" if TAG else f"_s0_mlp_p{P1_ENV}"

from experiment.lyra.constraint_lang import load_schema
from experiment.lyra.auto_compiler import _parse


def _load_secded_helpers():
    """Pull the SECDED codec out of lyra_triple_fault_sweep.py without running that
    script's sweep (it executes at module level): exec only its function defs."""
    import ast
    src_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lyra_triple_fault_sweep.py")
    tree = ast.parse(open(src_path, encoding="utf-8").read())
    tree.body = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    ns = {"np": np}
    exec(compile(tree, src_path, "exec"), ns)
    return ns


_sd = _load_secded_helpers()
hamming_params, encode_secded, decode_secded = _sd["hamming_params"], _sd["encode_secded"], _sd["decode_secded"]
extract_data_bits, int_to_bits, bits_to_int_signed = _sd["extract_data_bits"], _sd["int_to_bits"], _sd["bits_to_int_signed"]

DOMAINS = {
    "Financial": _os.path.join(LYRA_BACKEND, "experiment", "lyra", "domains", "yamls", "financial.yaml"),
    "Medical":   _os.path.join(LYRA_BACKEND, "experiment", "lyra", "domains", "yamls", "medical.yaml"),
    "Cyber":     _os.path.join(LYRA_BACKEND, "experiment", "lyra", "domains", "yamls", "cyber.yaml"),
    "Power":     _os.path.join(LYRA_BACKEND, "experiment", "lyra", "domains", "yamls", "power.yaml"),
}
SIGNED_VARS = {"bg_rate"}
BINARY_VARS = {"auth_state", "meal_flag"}   # documented 0/1 flags in the YAMLs
TOTAL_BITS, FRAC_BITS = 16, 12
SCALE = 1 << FRAC_BITS
QMIN, QMAX = -(1 << (TOTAL_BITS - 1)), (1 << (TOTAL_BITS - 1)) - 1

N_TRAIN = 100000         # fault-free decision log used to fit the surrogate (20k gave ~3x the boundary error)
N_CAL_STREAMS = 200      # fault-free streams used to calibrate every alarm threshold
T = 2000                 # monitoring horizon per stream (decisions)
W = 100                  # sliding window (decisions)
R = 5                    # independent test streams per fault
ALPHA = 0.01             # target stream-level false-alarm rate
JITTER_SIGMA = 0.01
CONF_GATE = float(GATE_ENV or 0.3)   # a disagreement counts as evidence only if |p(BLOCK) - 0.5| > CONF_GATE:
                         # the surrogate's own fault-free errors are all low-confidence boundary cases
OUT_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"behavioral_monitor_results{TAG}.json")


# ---------------------------------------------------------------- fixed-point helpers
def q(v):
    return np.clip(np.round(np.asarray(v, dtype=float) * SCALE), QMIN, QMAX).astype(np.int64)


def stuck(qv, bit, val):
    u = np.asarray(qv, dtype=np.int64) & 0xFFFF
    u = (u | (1 << bit)) if val else (u & ~(1 << bit))
    return np.where(u & 0x8000, u - 0x10000, u)


def flip(qv, bit):
    u = (int(qv) & 0xFFFF) ^ (1 << bit)
    return u - 0x10000 if u & 0x8000 else u


# ---------------------------------------------------------------- state distributions
def sample_states(rng, names, n, dist="uniform"):
    cols = []
    for nm in names:
        if dist == "uniform":
            u = rng.uniform(0, 1, n)
        elif dist == "low":            # mass shifted toward 0 (e.g. a calm operating regime)
            u = rng.beta(2, 5, n)
        elif dist == "high":           # mass shifted toward 1 (e.g. a stressed regime)
            u = rng.beta(5, 2, n)
        else:
            raise ValueError(dist)
        if nm in BINARY_VARS:
            u = (rng.uniform(0, 1, n) < 0.5).astype(float)
        cols.append(2 * u - 1 if nm in SIGNED_VARS else u)
    return np.stack(cols, axis=1)


# ---------------------------------------------------------------- the compiled shield (integer datapath)
class Shield:
    """Fixed-point realization of one domain's compiled hard constraints:
    input registers -> expression -> expression-result register -> comparator vs theta register."""

    def __init__(self, schema):
        self.names = schema.state_names()
        syms = [sp.Symbol(n) for n in self.names]
        sym_map = dict(zip(self.names, syms))
        self.specs = [s for s in schema.constraints
                      if s.violation.type in ("threshold_above", "threshold_below")]
        self.fns = [sp.lambdify(syms, _parse(s.violation.expr, sym_map), modules="numpy")
                    for s in self.specs]
        self.above = np.array([s.violation.type == "threshold_above" for s in self.specs])
        self.theta = np.array([s.violation.threshold for s in self.specs])
        self.theta_q = q(self.theta)
        self.used = [{str(v) for v in _parse(s.violation.expr, sym_map).free_symbols} for s in self.specs]

    def expr_q(self, X, fault=None):
        """Expression-result registers (n x C), with an optional input-register fault."""
        Xq = q(X)
        if fault and fault.get("kind") == "input":
            j = self.names.index(fault["var"])
            Xq[:, j] = stuck(Xq[:, j], fault["bit"], fault["val"])
        Xr = Xq / SCALE
        return np.stack([q(np.broadcast_to(np.asarray(fn(*Xr.T), dtype=float), (len(X),)))
                         for fn in self.fns], axis=1)

    def decide(self, X, fault=None, E=None):
        """Returns BLOCK matrix (n x C). `fault` = dict(kind, ...) or None.
        `E` = cached fault-free expr_q(X); reused unless the fault hits an input register."""
        fault = fault or {}
        if E is None or fault.get("kind") == "input":
            E = self.expr_q(X, fault)
        out = np.empty((len(X), len(self.specs)), dtype=bool)
        for c in range(len(self.specs)):
            eq = E[:, c]
            th = self.theta_q[c]
            if fault.get("c") == c:
                if fault["kind"] == "theta":
                    th = fault["theta_q"]
                elif fault["kind"] == "expr":
                    eq = stuck(eq, fault["bit"], fault["val"])
            blk = (eq > th) if self.above[c] else (eq < th)
            if fault.get("c") == c and fault["kind"] == "cmp":
                blk = np.full(len(X), fault["val"], dtype=bool)
            out[:, c] = blk
        return out


# ---------------------------------------------------------------- streaming detectors
def window_sum(x, w):
    """x: (T,) or (T,k). Sum over the trailing window of length w (partial at start)."""
    cs = np.cumsum(x, axis=0)
    shifted = np.zeros_like(cs)
    shifted[w:] = cs[:-w]
    return cs - shifted


def blocks(Pp):
    """Surrogate's BLOCK prediction from its per-output probabilities."""
    return Pp >= 0.5


def disagree(Pp, D):
    """Evidence of datapath corruption: the surrogate confidently disagrees with the shield."""
    return (blocks(Pp) != D) & (np.abs(Pp - 0.5) > CONF_GATE)


def first_alarm(stat, tau):
    idx = np.flatnonzero(stat >= tau)
    return int(idx[0]) if idx.size else None


def make_surrogate(kind, seed):
    """The surrogate learner; 'mlp' is the paper's, the others are the ablation."""
    if kind == "mlp":
        return MLPClassifier(hidden_layer_sizes=(64, 64), max_iter=400, early_stopping=True,
                             random_state=seed, tol=1e-5, n_iter_no_change=20)
    if kind == "hgb":
        return MultiOutputClassifier(HistGradientBoostingClassifier(random_state=seed))
    if kind == "knn":
        return KNeighborsClassifier(n_neighbors=15)
    if kind == "logreg":
        return MultiOutputClassifier(LogisticRegression(max_iter=2000))
    raise ValueError(kind)


def combine(kind, P1, P2):
    """Ensemble rules (improvement study, PRD Sec 9.26).
    ens_avg: average the two members' p(BLOCK).
    ens_min: evidence only where both members lean the same way; the combined
    probability is the less confident of the two, so one over-confident member
    cannot raise a disagreement on its own."""
    if kind == "ens_avg":
        return (P1 + P2) / 2
    same = np.sign(P1 - 0.5) == np.sign(P2 - 0.5)
    weaker = np.where(np.abs(P1 - 0.5) <= np.abs(P2 - 0.5), P1, P2)
    return np.where(same, weaker, 0.5)


class SurrogateMonitor:
    def __init__(self, seed=None, kind=None):
        self.kind = kind or SURROGATE
        seed = SEED if seed is None else seed
        if self.kind.startswith("ens_"):
            self.members = [SurrogateMonitor(seed, "mlp"), SurrogateMonitor(seed, "hgb")]
        else:
            self.clf = make_surrogate(self.kind, seed)
        self.tau = None

    def fit(self, X, Y):
        if self.kind.startswith("ens_"):
            for mem in self.members:
                mem.fit(X, Y)
            return self
        self.clf.fit(X, Y.astype(int))
        return self

    def predict(self, X):
        """(n, C) p(BLOCK) per output. The MLP returns that array directly; the
        per-output learners return one (n, n_classes) array per output."""
        if self.kind.startswith("ens_"):
            return combine(self.kind, *(mem.predict(X) for mem in self.members))
        p = self.clf.predict_proba(X)
        if not isinstance(p, list):
            return np.asarray(p)
        classes = (self.clf.classes_ if not hasattr(self.clf, "estimators_")
                   else [e.classes_ for e in self.clf.estimators_])
        return np.stack([pj[:, list(cj).index(1)] if 1 in cj else np.zeros(len(X))
                         for pj, cj in zip(p, classes)], axis=1)

    def stat(self, P, D):
        return window_sum(disagree(P, D).sum(axis=1), W)

    def calibrate(self, streams):
        """streams: list of (P, D) fault-free. Smallest integer tau with <= ALPHA streams ever alarming."""
        maxes = np.array([self.stat(P, D).max() for P, D in streams])
        self.tau = int(np.floor(np.quantile(maxes, 1 - ALPHA))) + 1
        return self.tau


class CusumMonitor:
    """Sequential variant of the surrogate monitor (ours, main): Bernoulli CUSUM on the
    per-decision indicator "surrogate disagrees with the shield on any output". A
    persistent hardware fault raises that rate for as long as it persists, so evidence
    is accumulated over the whole stream instead of a fixed window -- catching low-rate
    faults a window test cannot separate from the surrogate's own boundary error.
    p0 is estimated on fault-free calibration streams, the CUSUM is tuned to detect a
    rate of P1_FACTOR*p0, and the threshold h is calibrated like every other alarm."""

    P1_FACTOR = float(P1_ENV or 4.0)

    def fit(self, streams):
        z = np.concatenate([disagree(P, D).any(axis=1) for P, D in streams])
        self.p0 = max(float(z.mean()), 1.0 / (len(z) + 1))
        p1 = min(self.P1_FACTOR * self.p0, 0.5)
        self.up, self.down = np.log(p1 / self.p0), np.log((1 - p1) / (1 - self.p0))
        maxes = np.array([self.stat(P, D).max() for P, D in streams])
        self.tau = float(np.quantile(maxes, 1 - ALPHA)) + 1e-9
        return self

    def stat(self, P, D):
        inc = np.where(disagree(P, D).any(axis=1), self.up, self.down)
        s, out = 0.0, np.empty(len(inc))
        for t, v in enumerate(inc):
            s = max(0.0, s + v)
            out[t] = s
        return out


class RateMonitor:
    """Black-box non-learned baseline: per-output BLOCK-rate z-test vs fault-free rate."""

    def fit(self, Y):
        self.p0 = Y.mean(axis=0).clip(1e-3, 1 - 1e-3)
        return self

    def stat(self, D):
        n = np.minimum(np.arange(1, len(D) + 1), W)[:, None]
        r = window_sum(D.astype(float), W) / n
        z = np.abs(r - self.p0) / np.sqrt(self.p0 * (1 - self.p0) / n)
        z[: W // 2] = 0                       # require a half-full window before testing
        return z.max(axis=1)

    def calibrate(self, streams):
        maxes = np.array([self.stat(D).max() for D in streams])
        self.tau = float(np.quantile(maxes, 1 - ALPHA)) + 1e-9
        return self.tau


def train_theta_if(all_thetas, rng):
    rows = []
    for th in all_thetas:
        jit = rng.normal(0, JITTER_SIGMA, 600)
        rows.extend(((th * (1 + jit)) - th) / (abs(th) + 1e-6))
    clf = IsolationForest(n_estimators=200, contamination=0.01, random_state=SEED)
    clf.fit(np.array(rows)[:, None])
    return clf


# ---------------------------------------------------------------- fault enumeration
def secded_miscorrected_thetas(theta_q):
    k, m = TOTAL_BITS, hamming_params(TOTAL_BITS)
    n = k + m
    word0, p0, _ = encode_secded(int_to_bits(int(theta_q), k), k, m)
    out = {}
    for trip in itertools.combinations(range(1, n + 1), 3):
        w = dict(word0)
        for b in trip:
            w[b] ^= 1
        wf, status = decode_secded(w, p0, n)
        if status == "corrected":
            tq = bits_to_int_signed(extract_data_bits(wf, k, m), k)
            if tq != theta_q:
                out[tq] = out.get(tq, 0) + 1
    return out          # {decoded theta_q: multiplicity}


def enumerate_faults(sh):
    faults = []
    for c in range(len(sh.specs)):
        for b in range(TOTAL_BITS):
            faults.append(dict(cls="T1", kind="theta", c=c, theta_q=flip(sh.theta_q[c], b), w=1))
        for tq, mult in secded_miscorrected_thetas(sh.theta_q[c]).items():
            faults.append(dict(cls="T3", kind="theta", c=c, theta_q=tq, w=mult))
        for b in range(TOTAL_BITS):
            for v in (0, 1):
                faults.append(dict(cls="D", kind="expr", c=c, bit=b, val=v, w=1))
        for v in (False, True):
            faults.append(dict(cls="C", kind="cmp", c=c, val=v, w=1))
    for var in sh.names:
        for b in range(TOTAL_BITS):
            for v in (0, 1):
                faults.append(dict(cls="I", kind="input", var=var, bit=b, val=v, w=1))
    return faults


# ---------------------------------------------------------------- main experiment
def run():
    rng = np.random.default_rng(7 + 1000 * SEED)
    all_thetas = []
    shields = {}
    for dom, path in DOMAINS.items():
        shields[dom] = Shield(load_schema(path))
        all_thetas.extend(shields[dom].theta.tolist())
    theta_if = train_theta_if(all_thetas, np.random.default_rng(1 + 1000 * SEED))

    results = {"config": dict(N_TRAIN=N_TRAIN, T=T, W=W, R=R, ALPHA=ALPHA), "domains": {}, "faults": []}

    for dom, sh in shields.items():
        names = sh.names
        # ---- fault-free training log -> surrogate + rate baseline
        Xtr = sample_states(rng, names, N_TRAIN)
        Ytr = sh.decide(Xtr)
        mon = SurrogateMonitor().fit(Xtr, Ytr)
        rate = RateMonitor().fit(Ytr)

        # ---- calibration on fresh fault-free streams (in-distribution only)
        cal = []
        for _ in range(N_CAL_STREAMS):
            Xs = sample_states(rng, names, T)
            cal.append((Xs, mon.predict(Xs), sh.decide(Xs)))
        tau_s = mon.calibrate([(P, D) for _, P, D in cal])
        tau_r = rate.calibrate([D for _, _, D in cal])
        cus = CusumMonitor().fit([(P, D) for _, P, D in cal])
        acc = float(np.mean(np.concatenate([(blocks(P) == D).ravel() for _, P, D in cal])))
        overblock = float(np.mean(np.concatenate([(blocks(P) & ~D).ravel() for _, P, D in cal])))

        # ---- false alarms under input-distribution shift (no fault present)
        shift = {}
        for dist in ("low", "high"):
            fa_s = fa_r = fa_c = 0
            for _ in range(N_CAL_STREAMS):
                Xs = sample_states(rng, names, T, dist)
                P, D = mon.predict(Xs), sh.decide(Xs)
                fa_s += mon.stat(P, D).max() >= tau_s
                fa_r += rate.stat(D).max() >= tau_r
                fa_c += cus.stat(P, D).max() >= cus.tau
            shift[dist] = dict(surrogate=fa_s / N_CAL_STREAMS, rate=fa_r / N_CAL_STREAMS,
                               cusum=fa_c / N_CAL_STREAMS)

        results["domains"][dom] = dict(constraints=[s.name for s in sh.specs], surrogate_acc=acc,
                                       veto_overblock_rate=overblock, tau_surrogate=tau_s,
                                       tau_rate=tau_r, cusum_p0=cus.p0, cusum_h=cus.tau,
                                       shift_false_alarm=shift)
        print(f"[{dom}] surrogate per-output acc={acc*100:.3f}%  veto over-block={overblock*100:.3f}%  "
              f"tau_s={tau_s} tau_r={tau_r:.2f}  shift FA: {shift}")

        # ---- fixed test streams, shared by every fault in this domain
        tests = []
        for _ in range(R):
            Xs = sample_states(rng, names, T)
            Es = sh.expr_q(Xs)
            tests.append((Xs, Es, mon.predict(Xs), sh.decide(Xs, E=Es)))

        for f in enumerate_faults(sh):
            rec = dict(domain=dom, cls=f["cls"], w=f["w"],
                       target=(sh.specs[f["c"]].name if "c" in f else f["var"]))
            if f["kind"] == "theta":
                th_obs = f["theta_q"] / SCALE
                th0 = sh.theta[f["c"]]
                rel = (th_obs - th0) / (abs(th0) + 1e-6)
                rec["theta_rel_dev"] = float(rel)
                rec["B1"] = bool(abs(rel) > 3 * JITTER_SIGMA)
                rec["B2"] = bool(theta_if.predict([[rel]])[0] == -1)
            else:                               # theta register untouched -> invisible to B1/B2
                rec["B1"] = rec["B2"] = False

            per = dict(unsafe=[], overblock=[], s_det=[], s_lat=[], s_exp=[], r_det=[], r_exp=[],
                       veto_unsafe=[], c_det=[], c_lat=[], c_exp=[], c_sev=[], none_sev=[], veto_sev=[])
            for Xs, Es, P, D0 in tests:
                Df = sh.decide(Xs, f, E=Es)
                unsafe_t = (D0 & ~Df).any(axis=1)          # should BLOCK, faulted shield passes
                per["unsafe"].append(int(unsafe_t.sum()))
                per["overblock"].append(int((~D0 & Df).any(axis=1).sum()))
                a_s = first_alarm(mon.stat(P, Df), tau_s)
                a_r = first_alarm(rate.stat(Df), tau_r)
                per["s_det"].append(a_s is not None)
                per["s_lat"].append(a_s)
                per["s_exp"].append(int(unsafe_t[: (a_s + 1 if a_s is not None else T)].sum()))
                per["r_det"].append(a_r is not None)
                per["r_exp"].append(int(unsafe_t[: (a_r + 1 if a_r is not None else T)].sum()))
                per["veto_unsafe"].append(int((D0 & ~(Df | blocks(P))).any(axis=1).sum()))
                a_c = first_alarm(cus.stat(P, Df), cus.tau)
                end_c = a_c + 1 if a_c is not None else T
                per["c_det"].append(a_c is not None)
                per["c_lat"].append(a_c)
                per["c_exp"].append(int(unsafe_t[:end_c].sum()))
                # residual risk of faults the monitor NEVER flags: how far into the violation region
                # the TRUE state of their unsafe passes was (|expr - theta|, normalized units)
                viol = np.where(sh.above, Es - sh.theta_q, sh.theta_q - Es) / SCALE
                v = np.where(D0 & ~Df, viol, 0.0).max(axis=1)
                per["c_sev"].append(float(v.max()) if (a_c is None and unsafe_t.any()) else 0.0)
                per["none_sev"].append(float(v.max()) if unsafe_t.any() else 0.0)
                # unsafe passes that survive the fail-closed veto, and how deep they are
                vres = np.where(D0 & ~(Df | blocks(P)), viol, 0.0).max(axis=1)
                per["veto_sev"].append(float(vres.max()))
            rec.update(
                unsafe=float(np.mean(per["unsafe"])), overblock=float(np.mean(per["overblock"])),
                harmful=bool(np.mean(per["unsafe"]) > 0),
                behav_visible=bool(np.mean(per["unsafe"]) + np.mean(per["overblock"]) > 0),
                B4=float(np.mean(per["s_det"])), B3=float(np.mean(per["r_det"])),
                B4_lat=[x for x in per["s_lat"] if x is not None],
                B4_exposure=float(np.mean(per["s_exp"])), B3_exposure=float(np.mean(per["r_exp"])),
                veto_unsafe=float(np.mean(per["veto_unsafe"])),
                B4c=float(np.mean(per["c_det"])), B4c_lat=[x for x in per["c_lat"] if x is not None],
                B4c_exposure=float(np.mean(per["c_exp"])), B4c_sev=float(np.max(per["c_sev"])),
                undetected_sev=float(np.max(per["none_sev"])), veto_sev=float(np.max(per["veto_sev"])),
            )
            results["faults"].append(rec)

    with open(OUT_JSON, "w") as fh:
        json.dump(results, fh, indent=1)
    summarize(results)


# ---------------------------------------------------------------- reporting
def summarize(res):
    F = res["faults"]

    def wmean(rows, key):
        w = np.array([r["w"] for r in rows], float)
        v = np.array([float(r[key]) for r in rows])
        return float((w * v).sum() / w.sum()) if len(rows) else float("nan")

    print("\nRecall on HARMFUL faults (>=1 unsafe pass in the stream), weighted by multiplicity:")
    print(f"{'class':<6}{'#harmful':>9}{'B1 rule':>9}{'B2 IF':>8}{'B3 rate':>9}{'B4 win':>9}{'B4 cus':>9}"
          f"{'unsafe/T':>10}{'B3 exp':>8}{'B4w exp':>8}{'B4c exp':>8}{'veto res':>9}")
    for cls in ("T1", "T3", "D", "I", "C", "ALL"):
        rows = [r for r in F if r["harmful"] and (cls == "ALL" or r["cls"] == cls)]
        if not rows:
            continue
        print(f"{cls:<6}{len(rows):9d}{wmean(rows,'B1')*100:8.1f}%{wmean(rows,'B2')*100:7.1f}%"
              f"{wmean(rows,'B3')*100:8.1f}%{wmean(rows,'B4')*100:8.1f}%{wmean(rows,'B4c')*100:8.1f}%"
              f"{wmean(rows,'unsafe'):10.1f}{wmean(rows,'B3_exposure'):8.2f}{wmean(rows,'B4_exposure'):8.2f}"
              f"{wmean(rows,'B4c_exposure'):8.2f}"
              f"{wmean(rows,'veto_unsafe'):9.2f}")
    benign = [r for r in F if not r["behav_visible"]]
    print(f"\nBehaviorally silent faults (no decision changes in any test stream): {len(benign)} "
          f"-- B4 alarm rate on them {wmean(benign,'B4')*100 if benign else 0:.1f}% (should be ~0: "
          f"nothing observable changed, and nothing unsafe happened)")
    H = [r for r in F if r["harmful"]]
    print("\nRecall by fault severity (unsafe passes per T decisions without any monitor):")
    for lo, hi in [(0, 5), (5, 20), (20, 100), (100, np.inf)]:
        rows = [r for r in H if lo < r["unsafe"] <= hi]
        if rows:
            print(f"  ({lo},{hi}]  n={len(rows):5d}  B1={wmean(rows,'B1')*100:5.1f}%  B2={wmean(rows,'B2')*100:5.1f}%"
                  f"  B3={wmean(rows,'B3')*100:5.1f}%  B4win={wmean(rows,'B4')*100:5.1f}%  B4cus={wmean(rows,'B4c')*100:5.1f}%")
    print(f"Worst violation depth of any unsafe pass from a fault B4cus never flagged: "
          f"{max(r['B4c_sev'] for r in H):.4f} (vs. {max(r['undetected_sev'] for r in H):.4f} with no monitor)")
    vs = [r["veto_sev"] for r in F]
    print(f"Worst violation depth of any unsafe pass surviving the veto (any fault, any stream): "
          f"{max(vs):.4f}; faults with any surviving unsafe pass: {sum(v > 0 for v in vs)}/{len(F)}")
    lc = [x for r in H for x in r["B4c_lat"]]
    if lc:
        print(f"B4cus latency on harmful faults: median={np.median(lc):.0f}, p90={np.percentile(lc,90):.0f}")
    lats = [x for r in F if r["harmful"] for x in r["B4_lat"]]
    if lats:
        print(f"B4 detection latency on harmful faults (decisions): median={np.median(lats):.0f}, "
              f"p90={np.percentile(lats,90):.0f}, max={max(lats)}")
    for dom, d in res["domains"].items():
        print(f"  {dom:<10} shift false-alarm: low={d['shift_false_alarm']['low']}, "
              f"high={d['shift_false_alarm']['high']}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--summary":
        with open(OUT_JSON) as fh:
            summarize(json.load(fh))
    else:
        run()
