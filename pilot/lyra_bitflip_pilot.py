"""Phase 1 pilot (expanded): hardware-style single-bit-flip fault injection against
every `hard`/mandatory-style threshold (theta_i) compiled by Lyra's ACTUAL compiler,
across ALL FOUR regulated domains from Paper 1 (Financial/Medical/Cyber/Power).

Reuses `experiment.lyra.constraint_lang.load_schema` and
`experiment.lyra.auto_compiler._parse` / `_build_margin_expr` unmodified -- this
script only adds the fixed-point quantization + bit-flip fault model on top of
theta_i, the real compiled constant. See Project-10/PRD.md Sec 3 (KPI 1) / Sec 9.1.

Usage: python pilot/lyra_bitflip_pilot.py
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

# bg_rate (Medical) is the only signed [-1,1] state variable in these four domains;
# everything else is documented in the YAML comments as normalised to [0,1] (or a
# binary flag, which [0,1] sampling still stress-tests at the continuous relaxation
# level the compiler itself uses).
SIGNED_VARS = {"bg_rate"}

N_SAMPLES = 2000
BITWIDTHS = [8, 16, 32]   # total bits; frac_bits = total_bits - 4 (4 integer bits)
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


def margin(expr_val, theta, vtype):
    if vtype == "threshold_above":
        return theta - expr_val          # safe (margin>=0) iff expr <= theta
    if vtype == "threshold_below":
        return expr_val - theta          # safe (margin>=0) iff expr >= theta
    raise ValueError(f"unsupported type for this pilot: {vtype}")


results = []  # rows for the summary table

for domain_name, path in DOMAINS.items():
    schema = load_schema(path)
    state_names = schema.state_names()
    sym_map = {n: sp.Symbol(n) for n in state_names}
    sym_list = [sym_map[n] for n in state_names]

    # sample N_SAMPLES full state vectors once per domain
    state_samples = []
    for _ in range(N_SAMPLES):
        vec = []
        for n in state_names:
            if n in SIGNED_VARS:
                vec.append(rng.uniform(-1, 1))
            else:
                vec.append(rng.uniform(0, 1))
        state_samples.append(vec)
    state_samples = np.array(state_samples)

    for spec in schema.constraints:
        vtype = spec.violation.type
        if vtype not in ("threshold_above", "threshold_below"):
            print(f"[skip] {domain_name}/{spec.name}: unsupported type {vtype}")
            continue

        expr_sym = _parse(spec.violation.expr, sym_map)
        expr_fn = sp.lambdify(sym_list, expr_sym, modules=MODULES)
        expr_vals = np.array([expr_fn(*row) for row in state_samples])

        theta_true = spec.violation.threshold
        margin_true = margin(expr_vals, theta_true, vtype)
        should_block = margin_true < 0
        n_block = int(should_block.sum())
        if n_block == 0:
            print(f"[note] {domain_name}/{spec.name}: 0/{N_SAMPLES} sampled states "
                  f"should BLOCK under true theta={theta_true} -- skipping (sampling range too narrow)")
            continue

        for total_bits in BITWIDTHS:
            frac_bits = total_bits - 4
            q_theta, scale = quantize(theta_true, frac_bits, total_bits)
            worst_frac, worst_bit, worst_theta = 0.0, None, None
            for bit_pos in range(total_bits):
                q_f = flip_bit(q_theta, bit_pos, total_bits)
                theta_f = q_f / scale
                margin_f = margin(expr_vals, theta_f, vtype)
                unblocked = np.logical_and(should_block, margin_f >= 0)
                frac = unblocked.sum() / n_block
                if frac > worst_frac:
                    worst_frac, worst_bit, worst_theta = frac, bit_pos, theta_f
            results.append((domain_name, spec.name, vtype, theta_true, total_bits,
                             n_block, worst_bit, worst_theta, worst_frac))

print(f"\n{'Domain':<10} {'Constraint':<24} {'theta':>7} {'bits':>5} "
      f"{'#BLOCK':>7} {'worst_bit':>9} {'theta_flt':>10} {'collapse%':>10}")
print("-" * 92)
for (dom, name, vtype, theta, tb, nblk, wb, wt, wf) in results:
    print(f"{dom:<10} {name:<24} {theta:7.3f} {tb:5d} {nblk:7d} {wb:9d} {wt:10.3f} {wf*100:9.1f}%")

# ---- domain-level summary: worst-case collapse fraction across all constraints/bitwidths ----
print("\nWorst-case BLOCK-collapse fraction per domain (any constraint, any bitwidth, any bit):")
for dom in DOMAINS:
    dom_rows = [r for r in results if r[0] == dom]
    if not dom_rows:
        continue
    worst = max(dom_rows, key=lambda r: r[-1])
    print(f"  {dom:<10}: {worst[-1]*100:5.1f}%  (constraint={worst[1]}, {worst[4]}-bit, bit {worst[6]})")
