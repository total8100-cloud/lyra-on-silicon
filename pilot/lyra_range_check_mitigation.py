"""PRD follow-up: compare SECDED against a cheaper candidate mitigation
mentioned but never implemented in Sec 9.3 (parity) follow-up discussion --
a plain RANGE-CHECK GUARD. Idea: since every compiled theta_i is known at
compile time to lie in a narrow valid band (all 17 published thetas lie in
[-1, 1], most in [0, 1]), a single comparator that forces fallback whenever
the (possibly faulted) decoded theta falls outside [theta_min, theta_max]
costs far less hardware than a full Hamming SECDED codeword (no parity bits
stored/computed at all, just two signed comparators against the plain
stored theta). Question: does this cheap guard close the same single-bit
BLOCK-collapse hole that parity/SECDED close, or is it strictly weaker?

Usage: python pilot/lyra_range_check_mitigation.py
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
TOTAL_BITS, FRAC_BITS = 16, 12
MODULES = ["math", {"sqrt": math.sqrt, "log": math.log, "exp": math.exp}]
rng = np.random.default_rng(0)

# Compile-time-known valid band for a normalized threshold-comparison theta.
# Every one of Lyra's 17 published thetas lies in [-1, 1] (most in [0, 1]);
# a margin of +/-1 covers the signed-variable constraints too. This is the
# *best case* for the guard: the tightest band that still accepts every
# legitimate published theta.
RANGE_MIN, RANGE_MAX = -1.0, 1.0


def quantize(val, frac_bits, total_bits):
    scale = 2 ** frac_bits
    qmax = 2 ** (total_bits - 1) - 1
    q = int(np.clip(round(val * scale), -qmax - 1, qmax))
    return q, scale


def flip_bit(q, bit_pos, total_bits):
    mask = 1 << bit_pos
    u = (q & ((1 << total_bits) - 1)) ^ mask
    if u & (1 << (total_bits - 1)):
        u -= (1 << total_bits)
    return u


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

        scale = 2 ** FRAC_BITS
        q_theta, _ = quantize(theta_true, FRAC_BITS, TOTAL_BITS)

        # Single-bit: does the guard catch every fault that Phase 1 showed causes collapse?
        n_single_evading = 0  # in-range AND nonzero collapse -> guard misses it
        for b in range(TOTAL_BITS):
            q1 = flip_bit(q_theta, b, TOTAL_BITS)
            theta_f = q1 / scale
            in_range = (RANGE_MIN <= theta_f <= RANGE_MAX)
            if in_range:
                margin_f = margin(expr_vals, theta_f, vtype)
                unblocked = np.logical_and(should_block, margin_f >= 0)
                collapse = unblocked.sum() / n_block * 100
                if collapse > 0:
                    n_single_evading += 1

        n_evading_pairs = 0
        n_total_pairs = 0
        worst_collapse_evading = 0.0
        for b0 in range(TOTAL_BITS):
            for b1 in range(b0 + 1, TOTAL_BITS):
                n_total_pairs += 1
                q1 = flip_bit(q_theta, b0, TOTAL_BITS)
                q2 = flip_bit(q1, b1, TOTAL_BITS)
                theta_f = q2 / scale
                in_range = (RANGE_MIN <= theta_f <= RANGE_MAX)
                if in_range:
                    # guard does NOT trigger fallback -> faulted theta used as-is
                    margin_f = margin(expr_vals, theta_f, vtype)
                    unblocked = np.logical_and(should_block, margin_f >= 0)
                    collapse = unblocked.sum() / n_block * 100
                    if collapse > 0:
                        n_evading_pairs += 1
                        worst_collapse_evading = max(worst_collapse_evading, collapse)

        rows.append((domain_name, spec.name, theta_true, n_single_evading,
                     n_evading_pairs, n_total_pairs, worst_collapse_evading))

print(f"{'Domain':<10} {'Constraint':<24} {'theta':>8} {'single-bit evading/16':>22} "
      f"{'double-bit evading/total':>26} {'worst%':>8}")
print("-" * 100)
total_single_evading = 0
total_evading = 0
total_pairs = 0
worst_overall = 0.0
for dom, name, theta, nsingle, nev, ntot, worst in rows:
    print(f"{dom:<10} {name:<24} {theta:8.3f} {nsingle:19d}/16 {nev:21d}/{ntot:<8d} {worst:7.1f}%")
    total_single_evading += nsingle
    total_evading += nev
    total_pairs += ntot
    worst_overall = max(worst_overall, worst)

print(f"\nSingle-bit: {total_single_evading}/{16*len(rows)} total single-bit faults evade "
      f"the range-check guard AND still cause nonzero collapse, across {len(rows)} constraints.")
print(f"Double-bit: {total_evading}/{total_pairs} pairs evade the guard AND still cause "
      f"nonzero collapse.")
print(f"Worst-case collapse among double-bit evading pairs: {worst_overall:.1f}%.")
print("\nFor comparison (from Sec 7/SECDED, same 51-combo single/double-bit sweep "
      "at 16-bit): SECDED shows 0 miscorrections across all double-bit pairs for "
      "all 51 combinations (fully closed). The range-check guard is strictly "
      "weaker whenever a double fault lands the faulted theta back inside the "
      "compile-time-valid band but on the WRONG side of the true threshold --"
      " exactly what the numbers above quantify.")
