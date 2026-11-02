"""Correlate the Tcl test-step log (with wall-clock timestamps) against the
independently-captured UART log, and check each step's UART-derived
block/status against its expected value.

Usage: python correlate_uart.py <tcl_log_path> <uart_log_path>
"""
import re
import sys
from collections import Counter

tcl_log_path, uart_log_path = sys.argv[1], sys.argv[2]

steps = []
with open(tcl_log_path, encoding="utf-8", errors="ignore") as f:
    for line in f:
        m = re.search(
            r"STEP t=(\d+) label=(\S+) fault=(\d+) bg=(-?\d+) vio_block=(\d) vio_status=(\d)",
            line,
        )
        if m:
            t, label, fault, bg, vio_block, vio_status = m.groups()
            steps.append({
                "t": int(t), "label": label, "fault": int(fault), "bg": int(bg),
                "vio_block": int(vio_block), "vio_status": int(vio_status),
            })

uart = []
with open(uart_log_path, encoding="utf-8", errors="ignore") as f:
    for line in f:
        if line.startswith("#"):
            continue
        parts = line.strip().split(",")
        if len(parts) != 4:
            continue
        t_ms, byte_hex, block, status = parts
        uart.append((int(t_ms), int(block), int(status)))

WINDOW_MS = 400  # each step's settle window before its own timestamp

print(f"{len(steps)} Tcl steps, {len(uart)} UART samples")
print(f"{'label':<16} {'exp_block':>9} {'vio_block':>9} {'vio_status':>10} "
      f"{'uart_block':>10} {'uart_status':>11} {'uart_n':>7} {'match':>6}")
print("-" * 90)

n_mismatch_vio = 0
n_mismatch_uart = 0

for i, s in enumerate(steps):
    label = s["label"]
    # expected: baseline-BLOCK/bitN-BLOCK -> block=1, else block=0,
    # except pairX_Y-SAFE (double-bit, uncorrectable) -> forced block=1
    if "pair" in label:
        expected_block = 1
    elif "BLOCK" in label:
        expected_block = 1
    else:
        expected_block = 0

    t = s["t"]
    window = [(b, st) for (tm, b, st) in uart if t - WINDOW_MS <= tm <= t]
    if window:
        block_counts = Counter(b for b, st in window)
        status_counts = Counter(st for b, st in window)
        uart_block = block_counts.most_common(1)[0][0]
        uart_status = status_counts.most_common(1)[0][0]
        n = len(window)
    else:
        uart_block, uart_status, n = None, None, 0

    vio_ok = (s["vio_block"] == expected_block)
    uart_ok = (uart_block == expected_block)
    if not vio_ok:
        n_mismatch_vio += 1
    if not uart_ok:
        n_mismatch_uart += 1

    match_str = "OK" if (vio_ok and uart_ok) else ("VIO-BAD" if not vio_ok and uart_ok
                 else ("UART-BAD" if vio_ok and not uart_ok else "BOTH-BAD"))
    print(f"{label:<16} {expected_block:>9} {s['vio_block']:>9} {s['vio_status']:>10} "
          f"{str(uart_block):>10} {str(uart_status):>11} {n:>7} {match_str:>8}")

print("-" * 90)
print(f"VIO mismatches: {n_mismatch_vio}/{len(steps)}")
print(f"UART mismatches: {n_mismatch_uart}/{len(steps)}")
if n_mismatch_uart == 0:
    print("UART_VALIDATION PASS")
else:
    print("UART_VALIDATION FAIL")
