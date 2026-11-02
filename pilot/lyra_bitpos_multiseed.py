"""Multi-seed statistical rigor pass (PRD follow-up, matching Papers 1/2's
seed>=5 convention) for the bit-position BLOCK-collapse curve
(Cyber/zone_isolation, 16-bit) shown in the paper's Figure 1. The
"worst-bit = 100%" headline finding is a mathematical certainty once theta
is pushed outside the expression's valid range (no sampling variance), but
the INTERMEDIATE partial-collapse bits (8, 9, 11 for this constraint) do
depend on which states are sampled -- this script reruns the sweep across
5 independent seeds and reports mean +/- 95% CI per bit.

Usage: python pilot/lyra_bitpos_multiseed.py
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

path = _os.path.join(LYRA_BACKEND, "experiment", "lyra", "domains", "yamls", "cyber.yaml")
schema = load_schema(path)
state_names = schema.state_names()
sym_map = {n: sp.Symbol(n) for n in state_names}
sym_list = [sym_map[n] for n in state_names]
spec = next(c for c in schema.constraints if c.name == "zone_isolation")
expr_sym = _parse(spec.violation.expr, sym_map)
modules = ["math", {"sqrt": math.sqrt, "log": math.log, "exp": math.exp}]
expr_fn = sp.lambdify(sym_list, expr_sym, modules=modules)
theta_true = spec.violation.threshold

TOTAL_BITS, FRAC_BITS = 16, 12
scale = 2 ** FRAC_BITS
qmax = 2 ** (TOTAL_BITS - 1) - 1
q_theta = int(np.clip(round(theta_true * scale), -qmax - 1, qmax))


def flip_bit(q, bit_pos):
    mask = 1 << bit_pos
    u = (q & ((1 << TOTAL_BITS) - 1)) ^ mask
    if u & (1 << (TOTAL_BITS - 1)):
        u -= (1 << TOTAL_BITS)
    return u


SEEDS = [0, 1, 2, 3, 4]
N_SAMPLES = 5000
all_rates = []  # [seed][bit]

for seed in SEEDS:
    rng = np.random.default_rng(seed)
    state_samples = np.array([[rng.uniform(-1, 1) if n == "bg_rate" else rng.uniform(0, 1)
                                for n in state_names] for _ in range(N_SAMPLES)])
    expr_vals = np.array([expr_fn(*row) for row in state_samples])
    margin_true = theta_true - expr_vals
    should_block = margin_true < 0
    n_block = should_block.sum()

    rates = []
    for bit in range(TOTAL_BITS):
        theta_f = flip_bit(q_theta, bit) / scale
        margin_f = theta_f - expr_vals
        unblocked = np.logical_and(should_block, margin_f >= 0)
        rates.append(unblocked.sum() / n_block * 100)
    all_rates.append(rates)

all_rates = np.array(all_rates)  # shape (5 seeds, 16 bits)
means = all_rates.mean(axis=0)
stds = all_rates.std(axis=0, ddof=1)
ci95 = 1.96 * stds / np.sqrt(len(SEEDS))

print(f"{'bit':>4} {'mean%':>8} {'std%':>7} {'95% CI':>14}")
for b in range(TOTAL_BITS):
    print(f"{b:4d} {means[b]:7.2f}% {stds[b]:6.2f}% [{means[b]-ci95[b]:6.2f}, {means[b]+ci95[b]:6.2f}]")

np.save(_os.path.join(_REPO, "pilot", "bitpos_multiseed_means.npy"), means)
np.save(_os.path.join(_REPO, "pilot", "bitpos_multiseed_ci95.npy"), ci95)
print("\nSaved means/CI arrays for figure regeneration.")
