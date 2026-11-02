"""Generate the paper's figures (PRD-backed, academic style, no decoration
beyond what the data needs). Writes directly into docs/overleaf_upload/fig/.

Fig 1: bit-position BLOCK-collapse curve for Cyber/zone_isolation (16-bit),
       recomputed from the real Lyra compiler -- the paper's "money plot"
       for Sec. 5 (Phase 1 sweep).
Fig 2: triple-bit SECDED outcome breakdown -- Sec. 8 (0.00% safe vs 95.94%
       confidently-wrong), using the exact pooled numbers already reported.
Fig 3: three-stage validation schematic (Python model -> RTL sim -> silicon)
       for Sec. 11, a simple box-and-arrow diagram, not data-driven.

Usage: python tools/make_paper_figs.py
"""
import os as _os
_REPO = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
LYRA_BACKEND = _os.environ.get("LYRA_BACKEND", _os.path.join(_REPO, "lyra_backend"))
import sys
sys.path.insert(0, LYRA_BACKEND)

import math
import numpy as np
import sympy as sp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from experiment.lyra.constraint_lang import load_schema
from experiment.lyra.auto_compiler import _parse

OUT = _os.path.join(_REPO, "figures")
_os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({
    "font.size": 11, "font.family": "DejaVu Sans",
    "axes.spines.top": False, "axes.spines.right": False,
})

# ---------------------------------------------------------------- Fig 1 ----
def fig1_bitpos_curve():
    path = _os.path.join(LYRA_BACKEND, "experiment", "lyra", "domains", "yamls", "cyber.yaml")
    schema = load_schema(path)
    state_names = schema.state_names()
    sym_map = {n: sp.Symbol(n) for n in state_names}
    sym_list = [sym_map[n] for n in state_names]
    spec = next(c for c in schema.constraints if c.name == "zone_isolation")
    expr_sym = _parse(spec.violation.expr, sym_map)
    modules = ["math", {"sqrt": math.sqrt, "log": math.log, "exp": math.exp}]
    expr_fn = sp.lambdify(sym_list, expr_sym, modules=modules)

    rng = np.random.default_rng(0)
    state_samples = np.array([[rng.uniform(-1, 1) if n == "bg_rate" else rng.uniform(0, 1)
                                for n in state_names] for _ in range(5000)])
    expr_vals = np.array([expr_fn(*row) for row in state_samples])
    theta_true = spec.violation.threshold  # 0.30
    margin_true = theta_true - expr_vals
    should_block = margin_true < 0
    n_block = should_block.sum()

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

    rates = []
    for bit in range(TOTAL_BITS):
        theta_f = flip_bit(q_theta, bit) / scale
        margin_f = theta_f - expr_vals
        unblocked = np.logical_and(should_block, margin_f >= 0)
        rates.append(unblocked.sum() / n_block * 100)

    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    colors = ["#4C72B0" if r < 50 else "#C44E52" for r in rates]
    ax.bar(range(TOTAL_BITS), rates, color=colors, width=0.7)
    ax.set_xlabel("flipped bit position (0 = LSB, 15 = sign bit)")
    ax.set_ylabel("BLOCK-collapse rate (%)")
    ax.set_title(r"Cyber/zone_isolation, $\theta$=0.30, Q4.12 (16-bit): single-bit-flip sweep", fontsize=10)
    ax.set_xticks(range(TOTAL_BITS))
    ax.set_ylim(0, 105)
    ax.axhline(100, color="#888888", linewidth=0.6, linestyle=":")
    fig.tight_layout()
    fig.savefig(f"{OUT}/bitpos_collapse.png", dpi=200)
    print(f"wrote bitpos_collapse.png  rates={[round(r,1) for r in rates]}")


# ---------------------------------------------------------------- Fig 2 ----
def fig2_triplebit_outcomes():
    labels = ["Uncorrectable\n(safe fail-closed)", "Miscorrected\n(confident, wrong)",
              "Coincidentally\ncorrect"]
    values = [0.00, 95.94, 4.06]
    colors = ["#55A868", "#C44E52", "#CCB974"]

    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    bars = ax.bar(labels, values, color=colors, width=0.55)
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.2f}%",
                 ha="center", fontsize=11, fontweight="bold")
    ax.set_ylabel("fraction of 22,610 triple-bit\nfault combinations")
    ax.set_ylim(0, 108)
    ax.set_title("SECDED decoder outcome under triple-bit faults\n(pooled, all 17 constraints, 16-bit)", fontsize=10)
    fig.tight_layout()
    fig.savefig(f"{OUT}/triplebit_outcomes.png", dpi=200)
    print("wrote triplebit_outcomes.png")


# ---------------------------------------------------------------- Fig 3 ----
def fig3_validation_chain():
    fig, ax = plt.subplots(figsize=(7.4, 2.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 2.6)
    ax.axis("off")

    stages = [
        ("Python model\n(pilot/*.py)", "#4C72B0"),
        ("RTL simulation\n(iverilog/vvp)\n232/232 PASS", "#55A868"),
        ("Real silicon\n(Arty S7-50, JTAG)\nPASS", "#C44E52"),
    ]
    box_w, box_h, gap = 2.6, 1.5, 0.9
    x = 0.4
    centers = []
    for label, color in stages:
        box = FancyBboxPatch((x, 0.5), box_w, box_h, boxstyle="round,pad=0.08,rounding_size=0.12",
                              facecolor=color, edgecolor="none", alpha=0.85)
        ax.add_patch(box)
        ax.text(x + box_w / 2, 0.5 + box_h / 2, label, ha="center", va="center",
                 color="white", fontsize=9.5, fontweight="bold")
        centers.append(x + box_w)
        x += box_w + gap

    for cx in centers[:-1]:
        ax.annotate("", xy=(cx + gap - 0.1, 0.5 + box_h / 2), xytext=(cx + 0.1, 0.5 + box_h / 2),
                     arrowprops=dict(arrowstyle="-|>", color="#333333", lw=2))
        ax.text(cx + gap / 2, 0.5 + box_h / 2 + 0.35, "bit-exact\nagreement",
                 ha="center", fontsize=8, color="#333333", style="italic")

    fig.tight_layout()
    fig.savefig(f"{OUT}/validation_chain.png", dpi=200, bbox_inches="tight")
    print("wrote validation_chain.png")


# ---------------------------------------------------------------- Fig 4 ----
def fig4_monitor_recall():
    """Recall on harmful faults by fault class, per detector (Sec. 10): the mean of the five
    repeat runs, read from pilot/multiseed_summary.json -- no numbers typed by hand."""
    import json
    S = json.load(open(_os.path.join(_REPO, "pilot", "multiseed_summary.json")))["dense"]
    classes = [("T1", "$\\theta$ reg.\n1-bit, no ECC"), ("T3", "$\\theta$ reg.\nSECDED 3-bit"),
               ("D", "expr.-result\nreg. stuck-at"), ("I", "input reg.\nstuck-at"),
               ("C", "comparator\nstuck-at")]
    dets = [("B1", "B1 $\\theta$ rule (needs readout)", "#2a78d6"),
            ("B2", "B2 $\\theta$ IsolationForest (needs readout)", "#eb6834"),
            ("B3", "B3 BLOCK-rate monitor (black-box)", "#1baf7a"),
            ("B4c", "B4 learned surrogate + CUSUM (ours, black-box)", "#eda100")]

    def wmean(rows, k):
        w = np.array([r["w"] for r in rows], float)
        return float((w * np.array([float(r[k]) for r in rows])).sum() / w.sum()) * 100

    fig, ax = plt.subplots(figsize=(8.4, 3.6))
    bw = 0.19
    x = np.arange(len(classes))
    for j, (k, lab, col) in enumerate(dets):
        vals = [round(S[f"{c}.{k}"]["mean"], 1) for c, _ in classes]
        ax.bar(x + (j - 1.5) * bw, vals, bw - 0.02, color=col, label=lab, zorder=2)
        for xi, v in zip(x, vals):
            if v == 0:      # a zero bar is invisible; say so instead of leaving a gap
                ax.text(xi + (j - 1.5) * bw, 1.5, "0", ha="center", va="bottom",
                        fontsize=8, color="#55544e")
    ax.set_xticks(x)
    ax.set_xticklabels([c[1] for c in classes], fontsize=9)
    ax.set_ylabel("Recall on harmful faults (%)")
    ax.set_ylim(0, 105)
    ax.yaxis.grid(True, color="#e4e3dd", linewidth=0.8, zorder=0)
    ax.legend(fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2)
    fig.tight_layout()
    fig.savefig(f"{OUT}/monitor_recall.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig1_bitpos_curve()
    fig2_triplebit_outcomes()
    fig3_validation_chain()
    fig4_monitor_recall()
