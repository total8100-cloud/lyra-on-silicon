"""
Statistical analysis for the conformity experiment.

Functions used by conformity_runner._aggregate() and
accessible directly for offline analysis.
"""
from __future__ import annotations

import math
from typing import Optional

try:
    from scipy import stats as _scipy_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = mean(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / (len(values) - 1))


def median(values: list[float]) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    n = len(s)
    return (s[n // 2 - 1] + s[n // 2]) / 2 if n % 2 == 0 else s[n // 2]


def confidence_interval_95(values: list[float]) -> tuple[float, float]:
    """95% CI using t-distribution approximation (z=1.96 for large n)."""
    if len(values) < 2:
        return (0.0, 0.0)
    m  = mean(values)
    se = std(values) / math.sqrt(len(values))
    z  = 1.96
    return (round(m - z * se, 4), round(m + z * se, 4))


def mann_whitney_u(x: list[float], y: list[float]) -> dict:
    """
    Two-sided Mann-Whitney U test.
    Returns U statistic and approximate p-value using normal approximation.
    p_value is numerical; use p_str for publication-ready formatting.
    """
    n1, n2 = len(x), len(y)
    if n1 == 0 or n2 == 0:
        return {"U": None, "z": None, "p_value": None, "p_str": "N/A",
                "significant": False, "cles": None}

    # Compute U
    u1 = sum(1 if xi > yj else 0.5 if xi == yj else 0
             for xi in x for yj in y)
    u2 = n1 * n2 - u1
    U  = min(u1, u2)

    # Common Language Effect Size: P(X > Y)
    cles = round(max(u1, u2) / (n1 * n2), 4)

    # Prefer scipy (exact / better approximation), fall back to manual normal approx
    if HAS_SCIPY:
        res   = _scipy_stats.mannwhitneyu(x, y, alternative="two-sided")
        z_val = 0.0  # scipy doesn't return z directly
        p_raw = float(res.pvalue)
        U_use = float(res.statistic)
    else:
        U_use = float(U)
        mu_u  = n1 * n2 / 2
        sig_u = math.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
        z_val = (U_use - mu_u) / sig_u if sig_u > 0 else 0.0
        p_raw = 2 * _norm_sf(abs(z_val))

    # Normal approximation z (always computed for reporting)
    mu_u  = n1 * n2 / 2
    sig_u = math.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
    z     = (min(u1, u2) - mu_u) / sig_u if sig_u > 0 else 0.0

    # Clamp numerical p to float precision floor; format for publication
    p_num    = max(p_raw, 1e-16)
    if p_raw < 0.001:
        p_str = "< 0.001"
    elif p_raw < 0.01:
        p_str = f"{p_raw:.4f}"
    else:
        p_str = f"{p_raw:.4f}"

    return {
        "U":           round(U, 2),
        "z":           round(z, 4),
        "p_value":     round(p_num, 6),
        "p_str":       p_str,
        "significant": p_raw < 0.05,
        "cles":        cles,
        "scipy":       HAS_SCIPY,
    }


def cohen_d(x: list[float], y: list[float]) -> Optional[float]:
    """Pooled Cohen's d effect size."""
    if len(x) < 2 or len(y) < 2:
        return None
    mean_diff = mean(x) - mean(y)
    pooled_s  = math.sqrt(
        ((len(x) - 1) * std(x) ** 2 + (len(y) - 1) * std(y) ** 2)
        / (len(x) + len(y) - 2)
    )
    return round(mean_diff / pooled_s, 4) if pooled_s > 0 else 0.0


def rank_biserial_r(x: list[float], y: list[float]) -> Optional[float]:
    """
    Rank-biserial correlation r from Mann-Whitney U.

    r = (U_yx - U_xy) / (n1 * n2)  ∈ [-1, 1]

    Interpretation:
      r = +1.0  →  every x < every y  (x is always smaller / better)
      r = -1.0  →  every x > every y
      r =  0.0  →  no systematic difference

    Cohen conventions for |r|: 0.1=small, 0.3=medium, 0.5=large.
    Appropriate for non-normal data and naturally paired with Mann-Whitney U.
    Unlike Cohen's d, bounded and not inflated by near-zero within-group SD.
    """
    n1, n2 = len(x), len(y)
    if n1 == 0 or n2 == 0:
        return None
    # u_xy = # pairs where x > y
    u_xy = sum(1 if xi > yj else 0.5 if xi == yj else 0
               for xi in x for yj in y)
    u_yx = n1 * n2 - u_xy
    return round((u_yx - u_xy) / (n1 * n2), 4)


def effect_size_label(d: Optional[float]) -> str:
    if d is None:
        return "N/A"
    d = abs(d)
    if d < 0.2:   return "negligible"
    if d < 0.5:   return "small"
    if d < 0.8:   return "medium"
    return "large"


def summarise(values: list[float]) -> dict:
    ci = confidence_interval_95(values)
    return {
        "mean":   round(mean(values), 4),
        "std":    round(std(values), 4),
        "median": round(median(values), 4),
        "ci_95":  ci,
        "n":      len(values),
    }


def wilcoxon_signed_rank(
    x: list[float], y: list[float]
) -> dict:
    """
    Paired Wilcoxon signed-rank test (two-sided).
    Use when x[i] and y[i] are matched runs (same environment seed).
    Returns W statistic and approximate p-value (normal approximation).
    """
    if len(x) != len(y):
        return {"W": None, "z": None, "p_value": None,
                "p_str": "N/A", "significant": False}
    diffs = [xi - yi for xi, yi in zip(x, y)]
    nonzero = [(abs(d), d) for d in diffs if d != 0.0]
    n = len(nonzero)
    if n == 0:
        return {"W": 0, "z": 0.0, "p_value": 1.0,
                "p_str": "1.0000", "significant": False}

    # Rank absolute differences (handle ties with average ranks)
    sorted_nd = sorted(enumerate(nonzero), key=lambda t: t[1][0])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j < n - 1 and sorted_nd[j + 1][1][0] == sorted_nd[i][1][0]:
            j += 1
        avg_rank = (i + 1 + j + 1) / 2.0
        for k in range(i, j + 1):
            ranks[sorted_nd[k][0]] = avg_rank
        i = j + 1

    W_plus  = sum(r for r, (_, d) in zip(ranks, nonzero) if d > 0)
    W_minus = sum(r for r, (_, d) in zip(ranks, nonzero) if d < 0)
    W       = min(W_plus, W_minus)

    # Normal approximation
    mu_w  = n * (n + 1) / 4.0
    sig_w = math.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
    z     = (W - mu_w) / sig_w if sig_w > 0 else 0.0
    p_raw = 2 * _norm_sf(abs(z))

    p_str = "< 0.001" if p_raw < 0.001 else f"{p_raw:.4f}"
    return {
        "W":          round(W, 2),
        "z":          round(z, 4),
        "p_value":    round(max(p_raw, 1e-16), 6),
        "p_str":      p_str,
        "significant": p_raw < 0.05,
    }


def compare_conditions(
    x: list[float],
    y: list[float],
    label: str = "",
    paired: bool = False,
) -> dict:
    """
    Compare two run-level metric vectors.

    paired=True: x[i] and y[i] are matched (same environment seed).
      Uses paired Wilcoxon signed-rank as primary test.
      Also reports Mann-Whitney for reference.
    paired=False: independent runs.
      Uses Mann-Whitney U as primary test.
    """
    mw    = mann_whitney_u(x, y)
    d     = cohen_d(x, y)
    r     = rank_biserial_r(x, y)
    delta = round(mean(y) - mean(x), 4)

    result = {
        "label":           label,
        "paired":          paired,
        "mean_x":          round(mean(x), 4),
        "mean_y":          round(mean(y), 4),
        "delta":           delta,
        "improved":        delta > 0,
        "mann_whitney":    mw,
        "cohen_d":         d,
        "rank_biserial_r": r,
        "effect_size":     effect_size_label(d),
    }
    if paired and len(x) == len(y):
        result["wilcoxon"] = wilcoxon_signed_rank(x, y)
        # Primary p-value is Wilcoxon when paired
        result["primary_p"]    = result["wilcoxon"]["p_value"]
        result["primary_test"] = "wilcoxon"
    else:
        result["primary_p"]    = mw.get("p_value")
        result["primary_test"] = "mann_whitney"
    return result


# ── Normal distribution survival function (standard normal) ───────────────────

def _norm_sf(z: float) -> float:
    """P(Z > z) for standard normal — Abramowitz & Stegun approximation."""
    if z < 0:
        return 1.0 - _norm_sf(-z)
    t = 1.0 / (1.0 + 0.2316419 * z)
    poly = t * (0.319381530
                + t * (-0.356563782
                       + t * (1.781477937
                              + t * (-1.821255978
                                     + t * 1.330274429))))
    return poly * math.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)
