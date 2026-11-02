"""Hardware-in-the-loop validation of the AI drift detector (PRD follow-up,
Paper 3 AI-contribution strengthening pass): feeds REAL, silicon-decoded
theta_corrected values -- collected live over JTAG from zone_isolation on the
real Arty S7-50 under a sample of triple-bit fault injections
(vivado/program_and_test_aidbg.tcl) -- into the EXACT SAME IsolationForest
training procedure as pilot/lyra_ai_drift_detector.py (same pooled-jitter
training set, same random_state), and reports the detector's recall on this
real-hardware-measured sample instead of only the software-simulated one.

Usage: python pilot/lyra_ai_hw_validation.py <tcl_log_path>
"""
import os as _os
_REPO = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
LYRA_BACKEND = _os.environ.get("LYRA_BACKEND", _os.path.join(_REPO, "lyra_backend"))
import re
import sys

import numpy as np
from sklearn.ensemble import IsolationForest

sys.path.insert(0, LYRA_BACKEND)
from experiment.lyra.constraint_lang import load_schema

DOMAINS = {
    "Financial": _os.path.join(LYRA_BACKEND, "experiment", "lyra", "domains", "yamls", "financial.yaml"),
    "Medical":   _os.path.join(LYRA_BACKEND, "experiment", "lyra", "domains", "yamls", "medical.yaml"),
    "Cyber":     _os.path.join(LYRA_BACKEND, "experiment", "lyra", "domains", "yamls", "cyber.yaml"),
    "Power":     _os.path.join(LYRA_BACKEND, "experiment", "lyra", "domains", "yamls", "power.yaml"),
}
BITWIDTHS = [8, 16, 32]
NOMINAL_JITTER_SIGMA = 0.01
N_NOMINAL_PER_CONSTRAINT = 200
EPS = 1e-6
rng = np.random.default_rng(1)

THETA_TRUE = 0.30   # Cyber/zone_isolation
SCALE = 4096         # Q4.12


def hex_to_signed16(h):
    v = int(h, 16) & 0xFFFF
    if v & 0x8000:
        v -= 0x10000
    return v


# ---- re-train the identical detector (same training data/seed as lyra_ai_drift_detector.py) ----
nominal_rows = []
for domain_name, path in DOMAINS.items():
    schema = load_schema(path)
    for spec in schema.constraints:
        theta_true = spec.violation.threshold
        denom = abs(theta_true) + EPS
        for total_bits in BITWIDTHS:
            jitter = rng.normal(0, NOMINAL_JITTER_SIGMA, N_NOMINAL_PER_CONSTRAINT)
            theta_nom = theta_true * (1 + jitter)
            rel_dev = (theta_nom - theta_true) / denom
            for rd in rel_dev:
                nominal_rows.append(rd)

X_nominal = np.array([[r] for r in nominal_rows])
idx = rng.permutation(len(X_nominal))
split = int(0.7 * len(idx))
X_train = X_nominal[idx[:split]]

clf = IsolationForest(n_estimators=200, contamination=0.01, random_state=0)
clf.fit(X_train)

# ---- parse the real hardware samples ----
tcl_log_path = sys.argv[1] if len(sys.argv) > 1 else "tcl_test_log_aidbg.txt"
samples = []
with open(tcl_log_path, encoding="utf-8", errors="ignore") as f:
    for line in f:
        m = re.search(r"SAMPLE label=(\S+) fault=(\d+) theta_hex=([0-9a-fA-F]+) status=(\d) block=(\d)", line)
        if m:
            label, fault, theta_hex, status, block = m.groups()
            theta_q = hex_to_signed16(theta_hex)
            theta_real = theta_q / SCALE
            rel_dev = (theta_real - THETA_TRUE) / (abs(THETA_TRUE) + EPS)
            samples.append({
                "label": label, "fault": int(fault), "theta_q": theta_q,
                "theta_real": theta_real, "status": int(status), "block": int(block),
                "rel_dev": rel_dev,
            })

if not samples:
    print(f"No SAMPLE lines found in {tcl_log_path}")
    sys.exit(1)

baseline = [s for s in samples if s["label"] == "baseline"]
triples = [s for s in samples if s["label"] != "baseline"]

print(f"Parsed {len(samples)} real-hardware samples ({len(triples)} triple-bit fault cases, "
      f"{len(baseline)} baseline).")
if baseline:
    b = baseline[0]
    print(f"Baseline: theta_real={b['theta_real']:.4f} (expected {THETA_TRUE}), "
          f"status={b['status']}, block={b['block']}")

X_hw = np.array([[s["rel_dev"]] for s in triples])
preds = clf.predict(X_hw)  # +1 normal, -1 anomaly
flagged = preds == -1
recall_hw = flagged.mean()

print(f"\nHardware-in-the-loop AI detector validation (real silicon, triple-bit faults, "
      f"zone_isolation, N={len(triples)}):")
print(f"  Detector flagged as anomalous: {flagged.sum()}/{len(triples)} ({recall_hw*100:.1f}%)")

status_counts = {}
for s in triples:
    status_counts[s["status"]] = status_counts.get(s["status"], 0) + 1
print(f"  SECDED status distribution on these real samples: {status_counts} "
      f"(0=ok, 1=corrected/miscorrected, 2=uncorrectable)")

# cross-tab: flagged vs SECDED status
for st in sorted(status_counts):
    mask = np.array([s["status"] == st for s in triples])
    if mask.sum() == 0:
        continue
    r = flagged[mask].mean()
    print(f"    status={st}: {mask.sum()} samples, AI-flagged {r*100:.1f}%")

print("\nPer-sample detail:")
for s, f in zip(triples, flagged):
    print(f"  {s['label']:<16} theta={s['theta_real']:8.4f}  rel_dev={s['rel_dev']:8.3f}  "
          f"status={s['status']}  AI_flag={'ANOMALY' if f else 'normal'}")
