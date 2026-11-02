"""PRD Sec 9.9 item 2: quantify SECDED behavior under TRIPLE-bit faults (beyond
its designed single-correct/double-detect guarantee), across all 17 hard
constraints x 4 domains at the adopted 16-bit (Q4.12) resolution.

SECDED's guarantee is formally scoped to <=2 simultaneous bit errors. A
3-bit error can:
  - be correctly flagged "uncorrectable" (status=2) -> safe fail-closed, OR
  - be MISCORRECTED: the decoder's syndrome happens to alias a different
    single-bit-error pattern, so it reports status=1 ("corrected") with
    HIGH CONFIDENCE while actually returning a wrong data word -- the
    dangerous case, since this looks identical to a genuine, safe correction
    to anything consuming secded_status downstream (no behavioral signal
    distinguishes it from a real single-bit recovery).

This script exhaustively tests ALL C(21,3)=1330 word-bit triples per
constraint (21 = Hamming data+parity bits, excluding the overall-parity bit
whose own triple-faults reduce to simpler cases already covered by the
double-bit sweep) and reports, per constraint: miscorrection rate and
resulting worst-case BLOCK-collapse among miscorrected triples.

Usage: python pilot/lyra_triple_fault_sweep.py
"""
import os as _os
_REPO = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
LYRA_BACKEND = _os.environ.get("LYRA_BACKEND", _os.path.join(_REPO, "lyra_backend"))
import sys
sys.path.insert(0, LYRA_BACKEND)

import itertools
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
N_SAMPLES = 1000           # smaller than Phase 1's 2000 -- this sweep is combinatorially heavy
TOTAL_BITS, FRAC_BITS = 16, 12   # the adopted resolution (Sec 9.4 finding: 8-bit is numerically unfit)
MODULES = ["math", {"sqrt": math.sqrt, "log": math.log, "exp": math.exp}]
rng = np.random.default_rng(2)


def hamming_params(k):
    m = 1
    while (2 ** m) < (k + m + 1):
        m += 1
    return m


def encode_secded(data_bits, k, m):
    n = k + m
    word = {}
    data_iter = iter(data_bits)
    for pos in range(1, n + 1):
        if (pos & (pos - 1)) == 0:
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
    syndrome = 0
    for pos in range(1, n + 1):
        if word[pos]:
            syndrome ^= pos
    overall = p0
    for pos in range(1, n + 1):
        overall ^= word[pos]
    word = dict(word)
    if syndrome == 0 and overall == 0:
        return word, "ok"
    if syndrome != 0 and overall == 1:
        # For a genuine single-bit error, syndrome always lands in 1..n. A
        # multi-bit error can produce a "phantom" syndrome outside that range
        # (no such bit exists in the codeword) -- real hardware has nothing
        # valid to flip there, so the decoder confidently reports "corrected"
        # while leaving the (still-corrupted) data untouched. This is itself
        # part of what this sweep measures: a phantom-syndrome "correction"
        # is tracked as a miscorrection by the caller (theta_rec will differ
        # from the true value since nothing was actually fixed).
        if 1 <= syndrome <= n:
            word[syndrome] ^= 1
        return word, "corrected"
    if syndrome != 0 and overall == 0:
        return word, "uncorrectable"
    return word, "ok"


def extract_data_bits(word, k, m):
    n = k + m
    return [word[pos] for pos in range(1, n + 1) if (pos & (pos - 1)) != 0]


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
    return theta - expr_val if vtype == "threshold_above" else expr_val - theta


results = []

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

        k, m = TOTAL_BITS, hamming_params(TOTAL_BITS)
        n = k + m
        q_theta, scale = quantize(theta_true, FRAC_BITS, TOTAL_BITS)
        data_bits = int_to_bits(q_theta, k)
        word0, p0, _ = encode_secded(data_bits, k, m)
        word_positions = list(range(1, n + 1))   # include Hamming parity bits too

        n_triples = 0
        n_uncorrectable = 0
        n_miscorrected = 0
        worst_miscorrect_collapse = 0.0

        for b0, b1, b2 in itertools.combinations(word_positions, 3):
            w = dict(word0)
            w[b0] ^= 1; w[b1] ^= 1; w[b2] ^= 1
            w_fixed, status = decode_secded(w, p0, n)
            n_triples += 1
            if status == "uncorrectable":
                n_uncorrectable += 1
            elif status == "corrected":
                recovered_bits = extract_data_bits(w_fixed, k, m)
                theta_rec = bits_to_int_signed(recovered_bits, k) / scale
                if theta_rec != (q_theta / scale):
                    n_miscorrected += 1
                    margin_rec = margin(expr_vals, theta_rec, vtype)
                    unblocked = np.logical_and(should_block, margin_rec >= 0)
                    collapse = unblocked.sum() / n_block
                    worst_miscorrect_collapse = max(worst_miscorrect_collapse, collapse)
            # status == "ok" for a genuine triple error should not occur; not separately tracked

        results.append((domain_name, spec.name, n_triples, n_uncorrectable,
                         n_miscorrected, worst_miscorrect_collapse))

print(f"{'Domain':<10} {'Constraint':<24} {'#triples':>9} {'uncorrect%':>11} "
      f"{'miscorrect%':>12} {'worst-collapse%':>16}")
print("-" * 86)
for (dom, name, nt, nu, nm, wc) in results:
    print(f"{dom:<10} {name:<24} {nt:9d} {nu/nt*100:10.2f}% {nm/nt*100:11.2f}% {wc*100:15.1f}%")

tot_triples = sum(r[2] for r in results)
tot_miscorrect = sum(r[4] for r in results)
tot_uncorrect = sum(r[3] for r in results)
print(f"\nOverall (pooled across all {len(results)} constraints, {TOTAL_BITS}-bit): "
      f"{tot_triples} triple-bit combinations tested.")
print(f"  Uncorrectable (safe fail-closed):   {tot_uncorrect/tot_triples*100:.2f}%")
print(f"  MISCORRECTED (confident but wrong): {tot_miscorrect/tot_triples*100:.2f}%")
worst_overall = max(r[5] for r in results)
print(f"  Worst-case BLOCK-collapse among miscorrected triples (any constraint): {worst_overall*100:.1f}%")
