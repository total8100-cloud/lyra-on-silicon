"""Run the missing repeat runs of the improvement study (PRD Sec 9.26): every variant on the
given seeds, in the dense setting and (ensembles only) on the LLM-agent logs. A run whose
result file already exists is skipped, so nothing is overwritten and the script can resume.

Usage: python pilot/improve_run.py [seed ...]        (default: 3 4)
"""
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
LOGS = HERE / "improve_logs"
WORKERS = 14
DOMAINS = ["financial", "medical", "cyber", "power"]
DENSE = {
    "gate0.2": dict(LYRA_CONF_GATE="0.2"),
    "gate0.4": dict(LYRA_CONF_GATE="0.4"),
    "p1_2": dict(LYRA_P1="2"),
    "p1_8": dict(LYRA_P1="8"),
    "ens_avg": dict(LYRA_SURROGATE="ens_avg"),
    "ens_min": dict(LYRA_SURROGATE="ens_min"),
}
ENSEMBLES = ["ens_avg", "ens_min"]


def tag(seed, env):
    t = f"_s{seed}_{env.get('LYRA_SURROGATE', 'mlp')}"
    if "LYRA_CONF_GATE" in env:
        t += f"_g{env['LYRA_CONF_GATE']}"
    if "LYRA_P1" in env:
        t += f"_p{env['LYRA_P1']}"
    return t


def jobs(seeds):
    for s in seeds:
        for name, env in DENSE.items():
            yield (f"{name}_s{s}", ["lyra_behavioral_monitor.py"], dict(env, LYRA_SEED=str(s)),
                   f"behavioral_monitor_results{tag(s, env)}.json")
        for name in ENSEMBLES:
            env = dict(LYRA_SURROGATE=name)
            for d in DOMAINS:
                yield (f"llm_{name}_{d}_s{s}", ["lyra_llm_monitor_eval.py", "--domain", d],
                       dict(env, LYRA_SEED=str(s)), f"llm_monitor_results_{d}{tag(s, env)}.json")


def main():
    seeds = [int(a) for a in sys.argv[1:]] or [3, 4]
    todo = [j for j in jobs(seeds) if not (HERE / j[3]).exists()]
    print(f"{len(todo)} runs to do", flush=True)
    running, failed, done, total, t0 = [], [], 0, len(todo), time.time()
    while todo or running:
        while todo and len(running) < WORKERS:
            name, args, env, out = todo.pop(0)
            log = open(LOGS / f"{name}.log", "w", encoding="utf-8")
            e = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", PYTHONIOENCODING="utf-8", **env)
            p = subprocess.Popen([sys.executable, *args], cwd=HERE, env=e, stdout=log, stderr=subprocess.STDOUT)
            running.append((name, p, log, out))
        time.sleep(5)
        for item in list(running):
            name, p, log, out = item
            if p.poll() is not None:
                log.close()
                running.remove(item)
                ok = p.returncode == 0 and (HERE / out).exists()
                done += 1
                if not ok:
                    failed.append(name)
                print(f"[{(time.time() - t0) / 60:6.1f} min] {done}/{total} {name} {'ok' if ok else 'FAILED'}", flush=True)
    print("failed:", failed or "none", flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
