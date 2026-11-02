"""Phase 1 mitigation, third candidate (PRD Sec 4.4-3): an AI-based threshold-drift
anomaly detector -- the "novel AI aspect" meant to satisfy EAAI's desk-reject
filter (parity/SECDED are classical fault-tolerance, not AI).

Design: a SINGLE IsolationForest, trained once on pooled "nominal" relative-
deviation features from ALL 17 constraints across all 4 Paper-1 domains, that
flags abnormal theta_i drift WITHOUT any per-constraint hand-tuned threshold --
unlike SECDED/parity, which protect the register itself, this is a monitoring
layer that would catch drift from ANY source (bit-flip, voltage-induced analog
drift, a corrupted recompilation) using only the same single feature across
wildly different constraints (thresholds ranging from 0.0 to 1.0, positive and
negative).

Feature: relative deviation of the observed theta from its compiled nominal,
  delta = (theta_observed - theta_nominal) / (|theta_nominal| + eps)
"Nominal" (normal) samples simulate harmless floating-point jitter from
legitimate recompilation (Paper 1 Sec 5.5 hot-swap: the YAML is semantically
unchanged but SymPy/lambdify numerics are not bit-identical run to run) --
modelled here as small relative Gaussian noise. "Faulted" samples are the
actual single-bit-flip-induced theta values from the Phase 1 sweep
(lyra_bitflip_pilot.py), pooled across all bit positions/bitwidths/constraints.

Usage: python pilot/lyra_ai_drift_detector.py
"""
import os as _os
_REPO = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
LYRA_BACKEND = _os.environ.get("LYRA_BACKEND", _os.path.join(_REPO, "lyra_backend"))
import sys
sys.path.insert(0, LYRA_BACKEND)

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score

from experiment.lyra.constraint_lang import load_schema

DOMAINS = {
    "Financial": _os.path.join(LYRA_BACKEND, "experiment", "lyra", "domains", "yamls", "financial.yaml"),
    "Medical":   _os.path.join(LYRA_BACKEND, "experiment", "lyra", "domains", "yamls", "medical.yaml"),
    "Cyber":     _os.path.join(LYRA_BACKEND, "experiment", "lyra", "domains", "yamls", "cyber.yaml"),
    "Power":     _os.path.join(LYRA_BACKEND, "experiment", "lyra", "domains", "yamls", "power.yaml"),
}
BITWIDTHS = [8, 16, 32]
NOMINAL_JITTER_SIGMA = 0.01   # 1% relative jitter: legitimate recompilation noise
N_NOMINAL_PER_CONSTRAINT = 200
EPS = 1e-6
rng = np.random.default_rng(1)


def quantize(val, frac_bits, total_bits):
    scale = 2 ** frac_bits
    qmax = 2 ** (total_bits - 1) - 1
    q = int(np.clip(round(val * scale), -qmax - 1, qmax))
    return q, scale


def flip_bit(q, bit_pos, total_bits):
    mask = 1 << bit_pos
    u = q & ((1 << total_bits) - 1)
    u ^= mask
    if u & (1 << (total_bits - 1)):
        u -= (1 << total_bits)
    return u


nominal_rows = []   # (domain, constraint, bits, relative_deviation)
faulted_rows = []

for domain_name, path in DOMAINS.items():
    schema = load_schema(path)
    for spec in schema.constraints:
        theta_true = spec.violation.threshold
        denom = abs(theta_true) + EPS

        # ---- nominal class: harmless recompilation jitter ----
        for total_bits in BITWIDTHS:
            jitter = rng.normal(0, NOMINAL_JITTER_SIGMA, N_NOMINAL_PER_CONSTRAINT)
            theta_nom = theta_true * (1 + jitter)
            rel_dev = (theta_nom - theta_true) / denom
            for rd in rel_dev:
                nominal_rows.append((domain_name, spec.name, total_bits, rd))

        # ---- faulted class: every single-bit flip, every bitwidth ----
        for total_bits in BITWIDTHS:
            frac_bits = total_bits - 4
            q_theta, scale = quantize(theta_true, frac_bits, total_bits)
            for bit_pos in range(total_bits):
                q_f = flip_bit(q_theta, bit_pos, total_bits)
                theta_f = q_f / scale
                rel_dev = (theta_f - theta_true) / denom
                faulted_rows.append((domain_name, spec.name, total_bits, bit_pos, rel_dev))

X_nominal = np.array([[r[3]] for r in nominal_rows])
X_faulted = np.array([[r[4]] for r in faulted_rows])

print(f"Nominal (jitter) samples: {len(X_nominal)}   Faulted (bit-flip) samples: {len(X_faulted)}")
print(f"Nominal |rel_dev| range: [{np.abs(X_nominal).min():.4f}, {np.abs(X_nominal).max():.4f}]")
print(f"Faulted |rel_dev| range: [{np.abs(X_faulted).min():.4f}, {np.abs(X_faulted).max():.4f}]")

# ---- train/test split on nominal (to measure false-positive rate on HELD-OUT normal data) ----
idx = rng.permutation(len(X_nominal))
split = int(0.7 * len(idx))
train_idx, test_idx = idx[:split], idx[split:]
X_train = X_nominal[train_idx]
X_test_nominal = X_nominal[test_idx]

clf = IsolationForest(n_estimators=200, contamination=0.01, random_state=0)
clf.fit(X_train)

pred_nominal = clf.predict(X_test_nominal)       # +1 normal, -1 anomaly
pred_faulted = clf.predict(X_faulted)

fpr = np.mean(pred_nominal == -1)
recall = np.mean(pred_faulted == -1)

scores_nominal = -clf.score_samples(X_test_nominal)   # higher = more anomalous
scores_faulted = -clf.score_samples(X_faulted)
y_true = np.concatenate([np.zeros(len(scores_nominal)), np.ones(len(scores_faulted))])
y_score = np.concatenate([scores_nominal, scores_faulted])
auc = roc_auc_score(y_true, y_score)

print(f"\nSingle IsolationForest, trained ONCE on pooled nominal jitter across all "
      f"17 constraints x 4 domains x 3 bitwidths (no per-constraint hand-tuned threshold):")
print(f"  False-positive rate (held-out nominal flagged as anomaly): {fpr*100:.2f}%")
print(f"  Recall (bit-flip-faulted thetas flagged as anomaly):        {recall*100:.2f}%")
print(f"  ROC-AUC (nominal vs faulted, pooled across everything):     {auc:.4f}")

# ---- recall broken down by how SMALL the fault's relative deviation is ----
# (the hardest cases: faults that only mildly shift theta, e.g. low bit positions)
abs_dev = np.abs(X_faulted[:, 0])
for thresh_lo, thresh_hi, label in [(0, 0.05, "tiny (<5% rel. dev.)"),
                                      (0.05, 0.5, "moderate (5-50%)"),
                                      (0.5, np.inf, "large (>50%)")]:
    mask = (abs_dev >= thresh_lo) & (abs_dev < thresh_hi)
    if mask.sum() == 0:
        continue
    r = np.mean(clf.predict(X_faulted[mask]) == -1)
    print(f"  Recall on {label:<22} ({mask.sum():5d} faults): {r*100:6.2f}%")
