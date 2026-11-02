"""Phase 1 mitigation: parity-protected compiled thresholds + fail-safe fallback.

Design (PRD Sec 4.4, candidate 2/2bis): store one extra parity bit alongside each
compiled theta_i register. On every evaluation, recompute parity from the current
register value; a mismatch means the register has been corrupted since compilation
(the register is a compile-time constant and should never legitimately change
except through a full hot-swap/recompile event, which would also refresh the
parity bit -- see Paper 1 Sec 5.5's existing SHA-256 Q-table hash check for the
same idea applied to agent weights). On mismatch, the shield does NOT trust the
computed margin and instead forces Lyra's existing `fallback_i` safe action
(Paper 1 Sec 4.3) -- i.e. integrity failure is treated as a hard violation, in
keeping with Lyra's fail-first philosophy.

This re-runs the exact Phase 1 sweep (pilot/lyra_bitflip_pilot.py) and reports,
for every domain/constraint/bitwidth/bit position:
  - whether the single-bit fault is DETECTED by parity (it always should be --
    flipping exactly one bit always flips popcount parity; this is a closed-form
    property of parity, verified here empirically rather than assumed)
  - the resulting BLOCK-collapse rate AFTER mitigation (should drop to 0%,
    since every detected fault forces fallback instead of trusting a corrupted
    margin)
  - an explicit double-bit-flip stress test on the single worst bit-pair per
    constraint, to honestly characterise parity's known blind spot (even-weight
    errors are invisible to single-bit parity) -- this is a textbook property,
    not a new finding, but the paper needs it measured on these exact constraints
    rather than asserted.

Usage: python pilot/lyra_parity_mitigation.py
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


def flip_bit(q, bit_pos, total_bits):
    mask = 1 << bit_pos
    u = q & ((1 << total_bits) - 1)
    u ^= mask
    if u & (1 << (total_bits - 1)):
        u -= (1 << total_bits)
    return u


def popcount_parity(q, total_bits):
    u = q & ((1 << total_bits) - 1)
    return bin(u).count("1") % 2


def margin(expr_val, theta, vtype):
    if vtype == "threshold_above":
        return theta - expr_val
    if vtype == "threshold_below":
        return expr_val - theta
    raise ValueError(f"unsupported type: {vtype}")


rows = []            # per constraint/bitwidth: detection rate, post-mitigation collapse
double_fault_rows = []  # double-bit-flip blind-spot stress test

for domain_name, path in DOMAINS.items():
    schema = load_schema(path)
    state_names = schema.state_names()
    sym_map = {n: sp.Symbol(n) for n in state_names}
    sym_list = [sym_map[n] for n in state_names]

    state_samples = []
    for _ in range(N_SAMPLES):
        vec = [rng.uniform(-1, 1) if n in SIGNED_VARS else rng.uniform(0, 1) for n in state_names]
        state_samples.append(vec)
    state_samples = np.array(state_samples)

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

        for total_bits in BITWIDTHS:
            frac_bits = total_bits - 4
            q_theta, scale = quantize(theta_true, frac_bits, total_bits)
            parity_true = popcount_parity(q_theta, total_bits)

            n_detected, worst_post_collapse = 0, 0.0
            single_bit_collapse = []
            for bit_pos in range(total_bits):
                q_f = flip_bit(q_theta, bit_pos, total_bits)
                parity_f = popcount_parity(q_f, total_bits)
                detected = (parity_f != parity_true)
                n_detected += int(detected)

                if detected:
                    # integrity failure -> forced fallback -> treat as correctly
                    # enforced (0% collapse for this bit)
                    post_collapse = 0.0
                else:
                    theta_f = q_f / scale
                    margin_f = margin(expr_vals, theta_f, vtype)
                    unblocked = np.logical_and(should_block, margin_f >= 0)
                    post_collapse = unblocked.sum() / n_block
                single_bit_collapse.append(post_collapse)
                worst_post_collapse = max(worst_post_collapse, post_collapse)

            detection_rate = n_detected / total_bits
            rows.append((domain_name, spec.name, vtype, total_bits,
                         detection_rate, worst_post_collapse))

            # ---- double-bit-flip blind-spot stress test ----
            # flip the worst single bit (from Phase 1) together with every other bit;
            # parity is blind to any even-weight error, so some pair must slip through.
            worst_single_bit = int(np.argmax(single_bit_collapse))
            best_evasion = 0.0
            for other_bit in range(total_bits):
                if other_bit == worst_single_bit:
                    continue
                q_f = flip_bit(flip_bit(q_theta, worst_single_bit, total_bits), other_bit, total_bits)
                parity_f = popcount_parity(q_f, total_bits)
                if parity_f == parity_true:  # undetected (even-weight error)
                    theta_f = q_f / scale
                    margin_f = margin(expr_vals, theta_f, vtype)
                    unblocked = np.logical_and(should_block, margin_f >= 0)
                    collapse = unblocked.sum() / n_block
                    best_evasion = max(best_evasion, collapse)
            double_fault_rows.append((domain_name, spec.name, total_bits, worst_single_bit, best_evasion))

print(f"{'Domain':<10} {'Constraint':<24} {'bits':>5} {'detect%':>8} {'post-collapse%':>15}")
print("-" * 68)
for (dom, name, vtype, tb, det, coll) in rows:
    print(f"{dom:<10} {name:<24} {tb:5d} {det*100:7.1f}% {coll*100:14.1f}%")

all_detect = [r[4] for r in rows]
all_collapse = [r[5] for r in rows]
print(f"\nOverall: single-bit-flip detection rate = {np.mean(all_detect)*100:.1f}% "
      f"(min {np.min(all_detect)*100:.1f}%), "
      f"post-mitigation worst-case collapse = {np.max(all_collapse)*100:.1f}%")

print(f"\nDouble-bit-flip blind-spot stress test (worst single bit + every other bit):")
print(f"{'Domain':<10} {'Constraint':<24} {'bits':>5} {'evasion-collapse%':>18}")
print("-" * 62)
for (dom, name, tb, wb, ev) in double_fault_rows:
    flag = "  <-- UNDETECTED PAIR EXISTS" if ev > 0 else ""
    print(f"{dom:<10} {name:<24} {tb:5d} {ev*100:17.1f}%{flag}")
n_vulnerable = sum(1 for r in double_fault_rows if r[-1] > 0)
print(f"\n{n_vulnerable}/{len(double_fault_rows)} (constraint, bitwidth) combos have at least one "
      f"undetected double-bit-flip pair causing >0% collapse (expected -- single parity "
      f"cannot catch even-weight errors; this is the known, honestly-reported limitation).")
