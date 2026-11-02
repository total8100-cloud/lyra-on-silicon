"""Hardware-in-the-loop validation of the behavioral (black-box) integrity monitor
(PRD Sec 9.19) on the real Arty S7-50 running the SECDED-protected zone_isolation
shield (vivado/program_and_test_behavioral.tcl, aidbg VIO harness).

The monitor sees exactly what a deployed host sees and nothing more: the state it
sends to the shield and the BLOCK/PASS bit the silicon returns. It is the Cyber
surrogate from lyra_behavioral_monitor.py, trained only on SOFTWARE fault-free
decision logs, never on hardware data and never on faults.

  python pilot/lyra_behavioral_hil.py gen      -> vivado/behavioral_stimulus.txt (+ .json)
  (run program_and_test_behavioral.tcl on the board -> vivado/tcl_test_log_behavioral.txt)
  python pilot/lyra_behavioral_hil.py analyze  -> pilot/behavioral_hil_results.json
"""
import itertools
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import lyra_behavioral_monitor as bm  # noqa: E402

from experiment.lyra.constraint_lang import load_schema  # noqa: E402  (path set by bm)

ROOT = os.path.dirname(HERE)
STIM_TXT = os.path.join(ROOT, "vivado", "behavioral_stimulus.txt")
STIM_JSON = os.path.join(ROOT, "vivado", "behavioral_stimulus.json")
HW_LOG = os.path.join(ROOT, "vivado", "tcl_test_log_behavioral.txt")
OUT_JSON = os.path.join(HERE, "behavioral_hil_results.json")

S_FAULT = 200        # decisions per fault segment
S_CLEAN = 1000       # fault-free silicon segment
N_UNSAFE, N_OVERBLOCK, N_COINCIDENT = 12, 6, 2
W = bm.W
CONSTRAINT = "zone_isolation"


def build_monitor():
    rng = np.random.default_rng(11)
    sh = bm.Shield(load_schema(bm.DOMAINS["Cyber"]))
    c = [s.name for s in sh.specs].index(CONSTRAINT)
    Xtr = bm.sample_states(rng, sh.names, bm.N_TRAIN)
    mon = bm.SurrogateMonitor().fit(Xtr, sh.decide(Xtr))
    # calibrate tau on the single output that exists on silicon, at the fault-segment length
    cal = []
    for _ in range(1000):
        Xs = bm.sample_states(rng, sh.names, S_FAULT)
        cal.append((mon.predict(Xs)[:, c:c + 1], sh.decide(Xs)[:, c:c + 1]))
    maxes = [bm.window_sum(bm.disagree(P, D)[:, 0].astype(int), W).max() for P, D in cal]
    tau = int(np.floor(np.quantile(maxes, 1 - bm.ALPHA))) + 1
    mon.cusum = bm.CusumMonitor().fit(cal)
    return sh, c, mon, tau, rng


def secded_triples(theta_q):
    k, m = bm.TOTAL_BITS, bm.hamming_params(bm.TOTAL_BITS)
    n = k + m
    word0, p0, _ = bm.encode_secded(bm.int_to_bits(int(theta_q), k), k, m)
    out = []
    for trip in itertools.combinations(range(1, n + 1), 3):
        w = dict(word0)
        for b in trip:
            w[b] ^= 1
        wf, status = bm.decode_secded(w, p0, n)
        tq = bm.bits_to_int_signed(bm.extract_data_bits(wf, k, m), k)
        mask = sum(1 << (b - 1) for b in trip)        # codeword[20:0] = positions 1..21
        out.append(dict(mask=mask, status=status, theta_q=tq))
    return out


def gen():
    sh, c, mon, tau, rng = build_monitor()
    th0 = int(sh.theta_q[c])
    trip = secded_triples(th0)
    unsafe = [t for t in trip if t["status"] == "corrected" and t["theta_q"] > th0]
    overb = [t for t in trip if t["status"] == "corrected" and t["theta_q"] < th0]
    coinc = [t for t in trip if t["theta_q"] == th0]
    pick = (list(rng.choice(unsafe, N_UNSAFE, replace=False)) +
            list(rng.choice(overb, N_OVERBLOCK, replace=False)) +
            list(rng.choice(coinc, N_COINCIDENT, replace=False)))
    segs = [dict(label="clean", mask=0, n=S_CLEAN, sw_theta_q=th0)]
    segs += [dict(label=f"f{i:02d}", mask=int(t["mask"]), n=S_FAULT, sw_theta_q=int(t["theta_q"]))
             for i, t in enumerate(pick)]
    lines, meta = [], []
    for sg in segs:
        X = bm.sample_states(rng, sh.names, sg["n"])
        zd_q = bm.q(X[:, sh.names.index("zone_distance")])
        auth = X[:, sh.names.index("auth_state")].astype(int)
        for i in range(sg["n"]):
            lines.append(f"{sg['label']} {sg['mask']} {int(zd_q[i])} {auth[i]}")
        meta.append(dict(sg, X=X.tolist()))
    with open(STIM_TXT, "w", newline="\n") as fh:
        fh.write("\n".join(lines) + "\n")
    with open(STIM_JSON, "w") as fh:
        json.dump(dict(tau=tau, segments=meta), fh)
    print(f"tau={tau}  segments={len(segs)}  decisions={len(lines)}  -> {STIM_TXT}")


def analyze():
    sh, c, mon, tau, _ = build_monitor()     # thresholds re-derived from software calibration only
    stim = json.load(open(STIM_JSON))
    hw = {}
    for line in open(HW_LOG, encoding="utf-8", errors="ignore"):
        m = re.match(r"D (\S+) (\d+) (\d) (\d) ([0-9a-fA-F]+)", line.strip())
        if m:
            lab, i, blk, st, th = m.groups()
            hw.setdefault(lab, []).append((int(i), int(blk), int(st), int(th, 16)))
    theta_if = bm.train_theta_if(
        [t for d in bm.DOMAINS.values() for t in bm.Shield(load_schema(d)).theta.tolist()],
        np.random.default_rng(1))

    rows, mismatch_total = [], 0
    for sg in stim["segments"]:
        X = np.array(sg["X"])
        recs = sorted(hw[sg["label"]])
        assert len(recs) == sg["n"], (sg["label"], len(recs))
        D_hw = np.array([r[1] for r in recs], bool)
        status = np.array([r[2] for r in recs])
        th_hw = np.array([r[3] - 0x10000 if r[3] & 0x8000 else r[3] for r in recs])
        D_ref = sh.decide(X)[:, c]                               # what a fault-free shield decides
        D_sw = sh.decide(X, dict(kind="theta", c=c, theta_q=sg["sw_theta_q"]))[:, c]
        mismatch = int((D_sw != D_hw).sum())                     # software model vs silicon
        mismatch_total += mismatch
        P = mon.predict(X)[:, c]
        dis = bm.disagree(P, D_hw).astype(int)
        stat = bm.window_sum(dis, W)
        a = bm.first_alarm(stat, tau)
        ac = bm.first_alarm(mon.cusum.stat(P[:, None], D_hw[:, None]), mon.cusum.tau)
        unsafe_t = D_ref & ~D_hw
        rel = (th_hw[0] / bm.SCALE - sh.theta[c]) / abs(sh.theta[c])
        rows.append(dict(
            label=sg["label"], mask=sg["mask"], n=sg["n"], hw_theta=float(th_hw[0] / bm.SCALE),
            hw_status=int(status[0]), sw_hw_mismatch=mismatch,
            unsafe=int(unsafe_t.sum()), overblock=int((~D_ref & D_hw).sum()),
            disagreements=int(dis.sum()), alarm_at=a,
            exposure=int(unsafe_t[: (a + 1 if a is not None else sg["n"])].sum()),
            veto_unsafe=int((D_ref & ~(D_hw | bm.blocks(P))).sum()),
            cusum_alarm_at=ac,
            cusum_exposure=int(unsafe_t[: (ac + 1 if ac is not None else sg["n"])].sum()),
            B2_theta_if=bool(theta_if.predict([[rel]])[0] == -1)))
    json.dump(dict(tau=tau, rows=rows), open(OUT_JSON, "w"), indent=1)

    print(f"tau={tau} cusum_h={mon.cusum.tau:.2f} p0={mon.cusum.p0:.2e}  software-model vs silicon decision mismatches: {mismatch_total}")
    print(f"{'seg':<6}{'theta_hw':>9}{'st':>3}{'unsafe':>7}{'overblk':>8}{'disagr':>7}"
          f"{'alarm@':>7}{'expos':>6}{'cus@':>6}{'cexp':>5}{'veto_res':>9}{'B2':>4}")
    for r in rows:
        print(f"{r['label']:<6}{r['hw_theta']:9.4f}{r['hw_status']:3d}{r['unsafe']:7d}{r['overblock']:8d}"
              f"{r['disagreements']:7d}{str(r['alarm_at']):>7}{r['exposure']:6d}"
              f"{str(r['cusum_alarm_at']):>6}{r['cusum_exposure']:5d}{r['veto_unsafe']:9d}"
              f"{'Y' if r['B2_theta_if'] else '-':>4}")
    flt = [r for r in rows if r["label"] != "clean"]
    vis = [r for r in flt if r["unsafe"] + r["overblock"] > 0]
    harm = [r for r in flt if r["unsafe"] > 0]
    print(f"\nbehaviorally visible faults detected: {sum(r['alarm_at'] is not None for r in vis)}/{len(vis)}"
          f"   harmful detected: {sum(r['alarm_at'] is not None for r in harm)}/{len(harm)}"
          f"   silent faults alarmed: {sum(r['alarm_at'] is not None for r in flt if r not in vis)}"
          f"/{len(flt) - len(vis)}")
    print(f"CUSUM: visible detected {sum(r['cusum_alarm_at'] is not None for r in vis)}/{len(vis)}, "
          f"silent alarmed {sum(r['cusum_alarm_at'] is not None for r in flt if r not in vis)}"
          f"/{len(flt) - len(vis)}, exposure {sum(r['cusum_exposure'] for r in harm)}")
    print(f"unsafe passes: total {sum(r['unsafe'] for r in harm)}, before alarm "
          f"{sum(r['exposure'] for r in harm)}, with veto {sum(r['veto_unsafe'] for r in harm)}")


if __name__ == "__main__":
    {"gen": gen, "analyze": analyze}[sys.argv[1]]()
