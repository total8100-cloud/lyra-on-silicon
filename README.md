# Lyra on Silicon

Code, data and hardware sources for the paper

> **Diagnosing Random Hardware Faults in a Formally Verified Compliance Shield
> with a Learned Black-Box Integrity Monitor**
> Gyu-Hyun Lim, Yuchul Jung (Kumoh National Institute of Technology)

The paper studies shields that compile regulatory constraints into threshold comparisons. It takes the
Lyra compliance shield ([total8100-cloud/lyra](https://github.com/total8100-cloud/lyra)),
unmodified, and asks what a random hardware fault does to it, what detects the fault, and what
remains undetected. This repository contains the fault-injection experiments, the SECDED
mitigation, the learned black-box integrity monitor, the real LLM-agent decision logs, the
Verilog RTL and testbenches, the Vivado scripts for the Arty S7-50 board, and the raw results
behind every table.

## Setup

```bash
pip install -r requirements.txt        # Python 3.11
python setup_lyra.py                   # fetches the pinned Lyra commit into lyra_backend/
python -c "import zipfile; zipfile.ZipFile('raw_results.zip').extractall('.')"   # raw results into pilot/
```

`setup_lyra.py` places Lyra's `lyra/` package at `lyra_backend/experiment/lyra/`, the import path
the scripts use. To use a Lyra checkout elsewhere, set `LYRA_BACKEND` to a directory that
contains `experiment/lyra/`. `lyra_backend/experiment/stats.py` is a small helper from the
Lyra code base that its LLM harness imports.

If `setup_lyra.py` stops with `CERTIFICATE_VERIFY_FAILED`, point Python at certifi's CA bundle
and run it again: `pip install certifi`, then `SSL_CERT_FILE=$(python -m certifi) python setup_lyra.py`.

All experiment scripts are run from the repository root as `python pilot/<script>.py` and write
their results next to themselves in `pilot/`.

## Where each result comes from

| Paper | Result | Script |
|---|---|---|
| Sec. 5, Fig. 1 | Single-bit fault sweep over all 17 constraints (8/16/32-bit) | `pilot/lyra_bitflip_pilot.py`, `pilot/lyra_bitpos_multiseed.py` |
| Sec. 6 | Parity mitigation | `pilot/lyra_parity_mitigation.py` |
| Sec. 7 | Range-check guard (rejected alternative) | `pilot/lyra_range_check_mitigation.py` |
| Sec. 8 | SECDED mitigation | `pilot/lyra_secded_mitigation.py` |
| Sec. 9, Fig. 2 | Triple-bit faults (SECDED miscorrection) | `pilot/lyra_triple_fault_sweep.py` |
| Sec. 10, Results 1-3, Tables `monitor`, `learners` | Learned integrity monitor, dense setting; baselines B1-B3 | `pilot/lyra_behavioral_monitor.py` (B2 uses `pilot/lyra_ai_drift_detector.py`) |
| Sec. 10, Result 4, Table `trigger` | Trigger-gated evidence | `pilot/lyra_behavioral_monitor_trigger.py` |
| Sec. 10, Result 5, Table `llm` | Real LLM-agent traffic | `pilot/lyra_llm_agent_logs.py` (collect), `pilot/lyra_llm_monitor_eval.py` (evaluate) |
| Sec. 10, Table `improve` | Improvement study (gate, CUSUM, ensembles) | `pilot/improve_run.py` (runs), `pilot/improve_compare.py` (compares) |
| Sec. 10, Results 6-7, Table `dmr` | Bit-exact redundancy baseline B5; bit flips in the monitor's own weights | `pilot/lyra_redundancy_baseline.py` (`LYRA_SEED=0..4`), `pilot/redundancy_summary.py` |
| Sec. 10, Result 5, Table `llm` (runs added after the first five) | The monitor on LLM traffic, 24 added seeds, and the pooled 29-run figures | `pilot/lyra_llm_extra_seeds.py` |
| Sec. 10, active commissioning (negative result) | Uncertainty-sampled commissioning, two attempts | `pilot/lyra_active_commissioning.py`, `pilot/lyra_active_llm_eval.py`, `pilot/lyra_active_boundary_eval.py` |
| Sec. 10, mitigating the false alarms (negative results) | Alarm rules; pre-deployment acceptance test | `pilot/lyra_alarm_rule_eval.py`, `pilot/lyra_acceptance_eval.py` |
| Sec. 10, all monitor tables, Fig. 3 | Mean +- s.d. over repeat runs | `pilot/multiseed_summary.py`, `pilot/multiseed_extra.py`, `pilot/multiseed_tables.py` |
| Sec. 11 | RTL simulation | `rtl/`, `tb/`, `tools/gen_secded_params*.py` |
| Sec. 12, App. B | FPGA synthesis and live JTAG fault injection | `vivado/`, `scripts/`, `reports/vivado/` |
| App. C | Fault-free quantization error at 8 bits | `pilot/lyra_8bit_baseline_quantization.py` |
| App. D | Hardware-in-the-loop replay of the monitor | `pilot/lyra_behavioral_hil.py`, `vivado/program_and_test_behavioral.tcl` |
| Figures | All figures | `tools/make_paper_figs.py` (writes `figures/`) |

### Repeat runs and variants of the monitor

The monitor scripts read four environment variables:

| Variable | Meaning | Paper's value |
|---|---|---|
| `LYRA_SEED` | run index; re-draws the surrogate initialization and the test streams | `0`-`4` |
| `LYRA_SURROGATE` | `mlp`, `hgb`, `knn`, `logreg`, `ens_avg` (mean ensemble), `ens_min` (agreement ensemble) | `mlp` |
| `LYRA_CONF_GATE` | confidence gate on surrogate disagreement | `0.3` |
| `LYRA_P1` | CUSUM tuned for `P1 * p0` | `4` |

Seed 0 with the defaults writes untagged files (e.g. `behavioral_monitor_results.json`); every
other run writes `..._s{seed}_{surrogate}[_g{gate}|_p{p1}].json`. Example:

```bash
LYRA_SEED=3 python pilot/lyra_behavioral_monitor.py
LYRA_SEED=1 LYRA_SURROGATE=ens_avg python pilot/lyra_llm_monitor_eval.py
python pilot/multiseed_summary.py && python pilot/multiseed_tables.py
python pilot/improve_compare.py --llm ens_min ens_avg
```

`raw_results.zip` holds the result files of every run reported in the paper;
`pilot/multiseed_summary.json` regenerates from them exactly.

### LLM-agent decision logs

`pilot/llm_logs/{agent}_{domain}.jsonl` are the decision logs used in Result 5: 313 episodes of
8 steps per agent and domain, collected with Lyra's LLM harness (`llm_pilot_domain.py`)
unmodified. Each line is one step: episode, step, state `x`, proposed action `a`, whether the
reply parsed, and latency.

| Agent | Model identifier | Route |
|---|---|---|
| `llama8b` | `meta-llama/llama-3.1-8b-instruct` | OpenRouter |
| `llama70b` | `meta-llama/llama-3.3-70b-instruct` | OpenRouter |
| `gptoss20b` | `openai/gpt-oss-20b` | Groq |

The logs were collected on 18 October 2026. Re-collecting them needs `OPENROUTER_API_KEY` and
`GROQ_API_KEY` in the environment (or in `Openrouter_API_key.env` / `Groq_API_key.env` at the
repository root, which `.gitignore` excludes). Provider-side model updates mean re-collected
logs will not be identical; evaluating the monitor needs only the logs in this repository.

## Hardware

Target: Digilent Arty S7-50 (`xc7s50csga324-1`), Vivado 2026.1.

- **RTL simulation** (Icarus Verilog):
  ```bash
  iverilog -g2012 -I rtl -o zone.vvp tb/tb_zone_isolation_secded.v rtl/lyra_zone_isolation_core.v rtl/secded_decoder.v && vvp zone.vvp
  iverilog -g2012 -I rtl -o fall.vvp tb/tb_falling_glucose_secded.v rtl/lyra_falling_glucose_core.v rtl/secded_decoder.v && vvp fall.vvp
  ```
  Both print `PASS`. `tools/gen_secded_params*.py` regenerate the `rtl/*_params.vh` codewords from
  Lyra's compiled thresholds.
- **Synthesis and programming:** `scripts/vivado_flow.ps1` and `scripts/vio_flow.ps1` build the
  bitstreams from `vivado/build_*.tcl`; `vivado/program_and_test_*.tcl` program the board and run
  the scripted fault injection over JTAG (VIO). Logs of the runs reported in the paper are in
  `vivado/tcl_test_log*.txt`; utilization and timing reports in `reports/vivado/`; the
  bitstreams used are in `build/vivado/`.
- The Vivado scripts were written for a Windows host and contain absolute build paths
  (`D:/lyra_vio_build`, a `D:\lyra_fpga` junction, the Vivado install directory). They work
  around Vivado failing on project paths that contain parentheses. Edit `ROOT` and the paths at
  the top of each script for your machine.

## License

MIT, see `LICENSE`. Lyra itself is distributed under its own license in its repository.
