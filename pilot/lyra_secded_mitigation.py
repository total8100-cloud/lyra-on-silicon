"""Phase 1 mitigation, extended: SECDED (Single Error Correct, Double Error Detect)
Hamming code protecting the compiled theta_i register, closing the double-bit
blind spot that plain single-parity (lyra_parity_mitigation.py) left open.

Generic Hamming-SECDED: for a k-bit data word, m parity bits are placed at
power-of-two positions (1,2,4,...) among a (k+m)-bit Hamming code, plus one
extra overall-parity bit p0 covering the whole (k+m)-bit word for double-error
detection. On decode:
  - syndrome==0, overall parity OK            -> no error
  - syndrome!=0, overall parity MISMATCH       -> single-bit error -> CORRECTED
  - syndrome!=0, overall parity OK             -> double-bit error -> UNCORRECTABLE,
                                                   flagged, forces fallback_i
  - syndrome==0, overall parity MISMATCH       -> error in p0 itself -> data intact

This directly re-tests: (a) every single-bit fault in the full codeword (should
now be CORRECTED, giving 0% collapse with no fallback needed -- strictly better
than plain parity, which could only fail safe, not recover); (b) the exact
worst-case bit pairs that evaded plain single-parity in lyra_parity_mitigation.py
(should now be DETECTED as uncorrectable -> fallback -> 0% collapse).

Usage: python pilot/lyra_secded_mitigation.py
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


# ---------------------------------------------------------------- SECDED ----
def hamming_params(k):
    m = 1
    while (2 ** m) < (k + m + 1):
        m += 1
    return m


def encode_secded(data_bits, k, m):
    """data_bits: list of k bits (LSB first, i.e. data_bits[0] is theta's bit 0).
    Returns full codeword as dict {position(1..k+m): bit}, plus overall parity p0.
    Positions 1..(k+m), power-of-two positions are Hamming parity, rest are data."""
    n = k + m
    word = {}
    data_iter = iter(data_bits)
    for pos in range(1, n + 1):
        if (pos & (pos - 1)) == 0:   # power of two -> parity placeholder
            word[pos] = 0
        else:
            word[pos] = next(data_iter)
    for p_pos in [1 << i for i in range(m)]:
        parity = 0
        for pos in range(1, n + 1):
            if pos == p_pos:
                continue
            if pos & p_pos:
                parity ^= word[pos]
        word[p_pos] = parity
    p0 = 0
    for pos in range(1, n + 1):
        p0 ^= word[pos]
    return word, p0, n


def decode_secded(word, p0, n):
    """Returns (corrected_word, status) where status in {'ok','corrected','uncorrectable'}."""
    syndrome = 0
    for pos in range(1, n + 1):
        if word[pos]:
            syndrome ^= pos
    overall = p0
    for pos in range(1, n + 1):
        overall ^= word[pos]
    word = dict(word)
    if syndrome == 0 and overall == 0:
        return word, p0, "ok"
    if syndrome != 0 and overall == 1:
        word[syndrome] ^= 1   # correct the single-bit error
        return word, p0, "corrected"
    if syndrome != 0 and overall == 0:
        return word, p0, "uncorrectable"
    # syndrome == 0 and overall == 1 -> error in p0 itself, data untouched
    return word, p0 ^ 1, "ok"


def extract_data_bits(word, k, m):
    n = k + m
    bits = []
    for pos in range(1, n + 1):
        if (pos & (pos - 1)) != 0:   # not power of two -> data bit
            bits.append(word[pos])
    return bits


def int_to_bits(q, total_bits):
    u = q & ((1 << total_bits) - 1)
    return [(u >> i) & 1 for i in range(total_bits)]


def bits_to_int_signed(bits, total_bits):
    u = 0
    for i, b in enumerate(bits):
        u |= (b << i)
    if u & (1 << (total_bits - 1)):
        u -= (1 << total_bits)
    return u


def quantize(val, frac_bits, total_bits):
    scale = 2 ** frac_bits
    qmax = 2 ** (total_bits - 1) - 1
    q = int(np.clip(round(val * scale), -qmax - 1, qmax))
    return q, scale


def margin(expr_val, theta, vtype):
    if vtype == "threshold_above":
        return theta - expr_val
    if vtype == "threshold_below":
        return expr_val - theta
    raise ValueError(vtype)


# -------------------------------------------------------------- experiment ----
single_bit_rows = []
double_bit_rows = []

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
            k, m = total_bits, hamming_params(total_bits)
            data_bits = int_to_bits(q_theta, k)
            word0, p0, n = encode_secded(data_bits, k, m)

            # ---- (a) every single-bit fault in the full (n+1)-bit codeword ----
            n_bits_total = n + 1   # + overall parity bit
            n_corrected_ok = 0
            worst_collapse_single = 0.0
            for flip_idx in range(n_bits_total):
                w = dict(word0)
                p0_f = p0
                if flip_idx == 0:
                    p0_f ^= 1
                else:
                    w[flip_idx] ^= 1
                w_fixed, p0_fixed, status = decode_secded(w, p0_f, n)
                if status in ("ok", "corrected"):
                    recovered_bits = extract_data_bits(w_fixed, k, m)
                    theta_rec = bits_to_int_signed(recovered_bits, k) / scale
                    margin_rec = margin(expr_vals, theta_rec, vtype)
                    unblocked = np.logical_and(should_block, margin_rec >= 0)
                    collapse = unblocked.sum() / n_block
                    n_corrected_ok += int(collapse == 0.0)
                else:
                    collapse = 0.0   # uncorrectable -> forced fallback -> safe
                    n_corrected_ok += 1
                worst_collapse_single = max(worst_collapse_single, collapse)
            single_bit_rows.append((domain_name, spec.name, total_bits,
                                     n_corrected_ok / n_bits_total, worst_collapse_single))

            # ---- (b) double-bit faults: exhaustive pairs over codeword data bits ----
            data_positions = [pos for pos in range(1, n + 1) if (pos & (pos - 1)) != 0]
            n_pairs, n_uncaught = 0, 0
            worst_collapse_double = 0.0
            for i in range(len(data_positions)):
                for j in range(i + 1, len(data_positions)):
                    pi, pj = data_positions[i], data_positions[j]
                    w = dict(word0)
                    w[pi] ^= 1
                    w[pj] ^= 1
                    w_fixed, p0_fixed, status = decode_secded(w, p0, n)
                    n_pairs += 1
                    if status == "uncorrectable":
                        collapse = 0.0   # detected -> fallback -> safe
                    elif status == "corrected":
                        # SECDED should never claim "corrected" for a true double error;
                        # if it does, that's a mis-correction (worse than detection failure)
                        recovered_bits = extract_data_bits(w_fixed, k, m)
                        theta_rec = bits_to_int_signed(recovered_bits, k) / scale
                        margin_rec = margin(expr_vals, theta_rec, vtype)
                        unblocked = np.logical_and(should_block, margin_rec >= 0)
                        collapse = unblocked.sum() / n_block
                        if collapse > 0:
                            n_uncaught += 1
                    else:  # "ok" -- should not happen for a genuine 2-bit error
                        collapse = 0.0
                    worst_collapse_double = max(worst_collapse_double, collapse)
            double_bit_rows.append((domain_name, spec.name, total_bits,
                                     n_pairs, n_uncaught, worst_collapse_double))

print(f"{'Domain':<10} {'Constraint':<24} {'bits':>5} {'all-single-safe':>16} {'worst-collapse%':>16}")
print("-" * 76)
for (dom, name, tb, frac_ok, worst) in single_bit_rows:
    print(f"{dom:<10} {name:<24} {tb:5d} {frac_ok*100:15.1f}% {worst*100:15.1f}%")

print(f"\nSingle-bit-fault summary: {sum(1 for r in single_bit_rows if r[3]==1.0)}/"
      f"{len(single_bit_rows)} combos have ALL codeword single-bit faults safely "
      f"corrected or fail-safe (0% collapse).")

print(f"\n{'Domain':<10} {'Constraint':<24} {'bits':>5} {'#pairs':>7} {'miscorrected':>13} {'worst-collapse%':>16}")
print("-" * 86)
for (dom, name, tb, npairs, nuncaught, worst) in double_bit_rows:
    flag = "  <-- MISCORRECTION" if nuncaught > 0 else ""
    print(f"{dom:<10} {name:<24} {tb:5d} {npairs:7d} {nuncaught:13d} {worst*100:15.1f}%{flag}")

n_clean = sum(1 for r in double_bit_rows if r[4] == 0)
print(f"\nDouble-bit-fault summary: {n_clean}/{len(double_bit_rows)} combos have ZERO "
      f"miscorrections across ALL exhaustive data-bit pairs -- every double-bit error is "
      f"either correctly flagged uncorrectable (-> fallback, safe) or (rarely, by SECDED's "
      f"design) never silently accepted as if it were single-bit.")
