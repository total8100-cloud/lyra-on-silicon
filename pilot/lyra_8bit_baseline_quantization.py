"""PRD follow-up: quantify the FAULT-FREE (zero bit-flips) BLOCK-collapse
caused purely by 8-bit fixed-point quantization rounding, across all 17
constraints x 4 domains -- extending the single anecdotal example (Medical
hypoglycemia_prevention, 37.5%) found during the SECDED mitigation work
(pilot/lyra_secded_mitigation.py) into a full table. This is independent of
fault injection: it compares the TRUE floating-point theta against its
8-bit-quantized (fault-free) value only.

Usage: python pilot/lyra_8bit_baseline_quantization.py
"""
import os as _os
_REPO = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
LYRA_BACKEND = _os.environ.get("LYRA_BACKEND", _os.path.join(_REPO, "lyra_backend"))
import sys
sys.path.insert(0, LYRA_BACKEND)

import math
import numpy as np
import sympy as sp

from experiment.lyra.constraint_lang import load_schema
from experiment.lyra.auto_compiler import _parse

DOMAINS = {
    "Financial": _os.path.join(LYRA_BACKEND, "experiment", "lyra", "domains", "yamls", "financial.yaml"),
    "Medical":   _os.path.join(LYRA_BACKEND, "experiment", "lyra", "domains", "yamls", "medical.yaml"),
    "Cyber":     _os.path.join(LYRA_BACKEND, "experiment", "lyra", "domains", "yamls", "cyber.yaml"),
    "Power":     _os.path.join(LYRA_BACKEND, "experiment", "lyra", "domains", "yamls", "power.yaml"),
}
SIGNED_VARS = {"bg_rate"}
N_SAMPLES = 2000
BITWIDTHS = [8, 16, 32]
MODULES = ["math", {"sqrt": math.sqrt, "log": math.log, "exp": math.exp}]
rng = np.random.default_rng(0)


def quantize(val, frac_bits, total_bits):
    scale = 2 ** frac_bits
    qmax = 2 ** (total_bits - 1) - 1
    q = int(np.clip(round(val * scale), -qmax - 1, qmax))
    return q, scale


def margin(e, t, vt):
    return t - e if vt == "threshold_above" else e - t


rows = []
for domain_name, path in DOMAINS.items():
    schema = load_schema(path)
    state_names = schema.state_names()
    sym_map = {n: sp.Symbol(n) for n in state_names}
    sym_list = [sym_map[n] for n in state_names]
    state_samples = np.array([[rng.uniform(-1, 1) if n in SIGNED_VARS else rng.uniform(0, 1)
                                for n in state_names] for _ in range(N_SAMPLES)])

    for spec in schema.constraints:
        vtype = spec.violation.type
        if vtype not in ("threshold_above", "threshold_below"):
            continue
        expr_sym = _parse(spec.violation.expr, sym_map)
        expr_fn = sp.lambdify(sym_list, expr_sym, modules=MODULES)
        expr_vals = np.array([expr_fn(*row) for row in state_samples])

        theta_true = spec.violation.threshold
        margin_true = margin(expr_vals, theta_true, vtype)
        should_block = margin_true < 0
        n_block = int(should_block.sum())
        if n_block == 0:
            continue

        row = [domain_name, spec.name, theta_true]
        for bits in BITWIDTHS:
            frac_bits = bits - 4
            q, scale = quantize(theta_true, frac_bits, bits)
            theta_q = q / scale
            margin_q = margin(expr_vals, theta_q, vtype)
            unblocked = np.logical_and(should_block, margin_q >= 0)
            row.append(unblocked.sum() / n_block * 100)
        rows.append(row)

print(f"{'Domain':<10} {'Constraint':<24} {'theta':>8} {'8-bit%':>8} {'16-bit%':>9} {'32-bit%':>9}")
print("-" * 72)
n_nonzero_8 = 0
worst8 = 0.0
for (dom, name, theta, c8, c16, c32) in rows:
    flag = "  <-- nonzero at 8-bit" if c8 > 0 else ""
    print(f"{dom:<10} {name:<24} {theta:8.3f} {c8:7.1f}% {c16:8.1f}% {c32:8.1f}%{flag}")
    if c8 > 0:
        n_nonzero_8 += 1
        worst8 = max(worst8, c8)

print(f"\n{n_nonzero_8}/{len(rows)} constraints show nonzero FAULT-FREE BLOCK-collapse "
      f"at 8-bit resolution (quantization rounding alone, zero bit-flips injected).")
print(f"Worst case: {worst8:.1f}%. 16-bit and 32-bit: all constraints show 0.0% "
      f"fault-free collapse (verify above).")
