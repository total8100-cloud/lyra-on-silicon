"""Collect real LLM-agent decision logs for the integrity-monitor experiment (PRD Sec 9.21).

The trigger-aware monitor experiment (lyra_behavioral_monitor_trigger.py) draws the
agent's proposed actions uniformly and its states from synthetic distributions. This
script replaces both with what a deployed Lyra shield actually sees: the state x_t and
the action a_t that an LLM agent proposes in Paper 1's own environments.

Everything agent- and environment-side is Paper 1's code, imported unmodified from
Project-04 (experiment/lyra/llm_pilot_domain.py): the same system/user prompts, the same
DomainLLMAgent (temperature 0.3, same parsing, hold-action fallback on an unparseable
reply), the same environments, and the same run_domain_episode loop under the "lyra"
condition, so the trajectory is the one a shielded agent would follow. The only addition
is a wrapper around the agent that records (x_t, a_t) at the moment the shield receives
it. Each step is appended to a JSONL file as it happens, so an interrupted run resumes
where it stopped.

Usage:
  python pilot/lyra_llm_agent_logs.py --model llama8b  --domain medical --episodes 313
  python pilot/lyra_llm_agent_logs.py --model all --domain all          (sequential)
"""
import os as _os
_REPO = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
LYRA_BACKEND = _os.environ.get("LYRA_BACKEND", _os.path.join(_REPO, "lyra_backend"))
import argparse
import json
import logging
import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent      # resolve before the chdir below
BACKEND = LYRA_BACKEND
sys.path.insert(0, BACKEND)
os.chdir(BACKEND)   # Paper 1 resolves its YAML paths relative to backend/

from experiment.lyra import llm_pilot_domain as p1  # noqa: E402
from experiment.lyra.auto_compiler import compile_yaml  # noqa: E402

LOG_DIR = ROOT / "pilot" / "llm_logs"
MODELS = {
    "llama8b":  ("openrouter", "meta-llama/llama-3.1-8b-instruct"),
    "llama70b": ("openrouter", "meta-llama/llama-3.3-70b-instruct"),
    "gptoss20b": ("groq", "openai/gpt-oss-20b"),
}
N_STEPS = 8          # Paper 1's episode length


def load_keys():
    for name, var in (("Groq_API_key.env", "GROQ_API_KEY"), ("Openrouter_API_key.env", "OPENROUTER_API_KEY")):
        if os.environ.get(var) or not (ROOT / name).exists():
            continue                     # already set in the environment, or no key file
        raw = (ROOT / name).read_text().strip()
        for line in raw.splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                os.environ[k.strip()] = v.strip()
                break
        else:
            os.environ[var] = raw


class LoggingAgent(p1.DomainLLMAgent):
    """Paper 1's agent, plus a record of every (state, proposed action) the shield receives."""

    def __init__(self, *a, sink, episode, **kw):
        super().__init__(*a, **kw)
        self.sink, self.episode, self.step = sink, episode, 0

    def choose(self, x):
        a, lat, valid = super().choose(x)
        self.sink.write(json.dumps(dict(ep=self.episode, t=self.step, x=[float(v) for v in x],
                                        a=int(a), valid=bool(valid), ms=round(lat, 1))) + "\n")
        self.sink.flush()
        self.step += 1
        return a, lat, valid


def collect(model_key, domain, n_episodes):
    provider, model = MODELS[model_key]
    cfg = p1.DOMAIN_CFG[domain]
    client = p1._make_client(provider)
    compiled = compile_yaml(p1._yaml_path(cfg["yaml"]))
    path = LOG_DIR / f"{model_key}_{domain}.jsonl"
    done_eps = set()
    if path.exists():
        for line in path.read_text().splitlines():
            done_eps.add(json.loads(line)["ep"])
    # drop a partially written last episode so every episode in the file is complete
    if done_eps:
        lines = path.read_text().splitlines()
        last = max(done_eps)
        if sum(json.loads(l)["ep"] == last for l in lines) < N_STEPS:
            path.write_text("".join(l + "\n" for l in lines if json.loads(l)["ep"] != last))
            done_eps.discard(last)
    rng = random.Random(f"{domain}-env")       # same environment seeds for every model
    seeds = [rng.randint(0, 10**9) for _ in range(n_episodes)]
    t0 = time.time()
    with open(path, "a") as sink:
        for ep in range(n_episodes):
            if ep in done_eps:
                continue
            env = p1._load_env(cfg["env_class"], seed=seeds[ep])
            agent = LoggingAgent(client, model=model, domain=domain, sink=sink, episode=ep)
            p1.run_domain_episode(env, agent, compiled, "lyra", cfg["tau_low"], cfg["tau_high"],
                                  evaluator=None, n_steps=N_STEPS)
            if ep % 25 == 0:
                print(f"[{model_key}/{domain}] episode {ep + 1}/{n_episodes}  "
                      f"{time.time() - t0:.0f}s elapsed", flush=True)
    print(f"[{model_key}/{domain}] DONE -> {path}", flush=True)


def main():
    logging.basicConfig(level=logging.WARNING)
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="all", choices=list(MODELS) + ["all"])
    ap.add_argument("--domain", default="all", choices=p1.ALL_DOMAINS + ["all"])
    ap.add_argument("--episodes", type=int, default=313)      # 313 x 8 = 2,504 steps
    args = ap.parse_args()
    load_keys()
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    for m in (MODELS if args.model == "all" else [args.model]):
        for d in (p1.ALL_DOMAINS if args.domain == "all" else [args.domain]):
            collect(m, d, args.episodes)


if __name__ == "__main__":
    main()
