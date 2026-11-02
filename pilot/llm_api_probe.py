"""Probe the three agent models for the LLM-agent decision-log experiment (PRD Sec 9.21):
are they served, how long does one Paper-1-style call take, and what rate limits apply.
Keys are read from Project-10's *_API_key.env files (same parsing as Paper 1's
llm_pilot_domain.py) and are never printed.

Usage: python pilot/llm_api_probe.py
"""
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS = [("groq", "llama-3.1-8b-instant"), ("groq", "llama-3.3-70b-versatile"),
          ("openrouter", "openai/gpt-oss-20b")]


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


def client(provider):
    if provider == "groq":
        from groq import Groq
        return Groq(api_key=os.environ["GROQ_API_KEY"])
    from openai import OpenAI
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])


def main():
    load_keys()
    print("keys loaded:", {k: bool(os.environ.get(k)) for k in ("GROQ_API_KEY", "OPENROUTER_API_KEY")})
    for prov in ("groq", "openrouter"):
        try:
            ids = {m.id for m in client(prov).models.list().data}
            for p, m in MODELS:
                if p == prov:
                    print(f"[{prov}] {m}: {'served' if m in ids else 'NOT LISTED'}")
        except Exception as e:
            print(f"[{prov}] model list failed: {type(e).__name__}: {e}")
    for prov, model in MODELS:
        c = client(prov)
        lat = []
        for _ in range(3):
            t0 = time.perf_counter()
            try:
                raw = c.chat.completions.with_raw_response.create(
                    model=model, temperature=0.3,
                    max_tokens=512 if "gpt-oss" in model else 8,
                    messages=[{"role": "system", "content": "You are a grid operator."},
                              {"role": "user", "content": "Reply with exactly ONE word: no_action cap_on cap_off"}])
                resp = raw.parse()
                lat.append((time.perf_counter() - t0) * 1000)
                text = (resp.choices[0].message.content or "").strip()[:40]
                hdr = {k: v for k, v in raw.headers.items() if "ratelimit" in k.lower()}
            except Exception as e:
                print(f"[{prov}] {model}: call failed: {type(e).__name__}: {str(e)[:200]}")
                break
        else:
            print(f"[{prov}] {model}: latency ms {[round(x) for x in lat]}  reply={text!r}")
            for k, v in sorted(hdr.items()):
                print(f"      {k}: {v}")


if __name__ == "__main__":
    main()
