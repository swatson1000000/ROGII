# CLAUDE.md — Execution Guidelines

> Canonical execution policy is in `.github/copilot-instructions.md`. This file retains the
> environment setup, the `plan.md` rule, and the 12-rule template.

## Writing Conventions

**Always use US English spellings; never British.** This applies everywhere — code, comments,
identifiers, docs, commit messages, and chat responses. E.g. `color` not `colour`, `normalize`
not `normalise`, `behavior` not `behaviour`, `center` not `centre`, `modeling` not `modelling`.

## ⚠️ ALWAYS Consult plan.md Before Any Action

**Before suggesting or starting any experiment or code change, read `plan.md` first.**

`plan.md` is the single source of truth for:
- What has already been tried (LB Submission History, Experiment Log)
- What is currently running (Current State table)
- What to do next (Prioritized Action Plan, next `⬜` item)
- Active gates and decision criteria

Never propose the next step from memory or inference alone — always verify against `plan.md`.
`SUMMARY.md` holds the verified competition facts (task, schema, metric, units).

## Environment Setup

**The conda env depends on the machine** (see Compute Environment below):
- **skynet (local, aarch64)** → `kaggle-arch` — verified: Python 3.11 + numpy/pandas/
  scikit-learn/lightgbm/scipy/pyarrow + a working `kaggle` CLI (1.7.4.5, authed). Note:
  **xgboost is not installed** here (`pip install xgboost` if needed). The **base-env** `kaggle`
  CLI (`~/miniconda3/bin/kaggle`) is broken — use the one inside `kaggle-arch`.
- **deepthought (remote, x86_64)** → `kaggle` — has **detectron2 0.6** (GPU, built from source
  against torch 2.7.1+cu126; verified on the RTX 4080). Not available on skynet/hal9000.

Always activate before running ANY command — Python scripts, the `kaggle` CLI, etc.:

```bash
# local (skynet / aarch64)
source ~/miniconda3/etc/profile.d/conda.sh && conda activate kaggle-arch
```

⚠️ Plain `conda activate <env>` fails in non-interactive shells (Bash tool). Use the `source`
prefix above.

## Compute Environment — three machines

Three GPU-capable machines are available via passwordless SSH. **Each host has exactly one
GPU** (no PCIe scale-up; only LAN between them).

| Host | Where | GPU | Arch | Conda env | Dispatch |
|------|-------|-----|------|-----------|----------|
| **skynet** | local (`spark-4685`) | GB10 (DGX Spark), ~119 GB unified LPDDR5X, sm_121 | aarch64 | `kaggle-arch` | run commands directly |
| **deepthought** | remote | RTX 4080, 16 GB GDDR6X, sm_89 | x86_64 | `kaggle` | `runon deepthought <cmd>`; pull results with `syncback deepthought` |
| **hal9000** | remote (192.168.1.150) | GTX 1650, ~4 GB, sm_75 | x86_64 (RHEL 9.7) | — | `ssh hal9000` |

**Routing rules:**
- **deepthought** is the default for GPU-heavy training (~4–5× faster than skynet; the GB10 is
  memory-bandwidth-bound and lacks sm_121 kernel parity). It is **multi-tenant** — check
  `ssh deepthought nvidia-smi` before dispatching long jobs.
- **skynet** (local) for CPU-bound / I/O-heavy work, or when deepthought is busy.
- **hal9000** is a light third lane (~10× slower than the 4080) — small models, inference
  probes, dev only.

**Multi-GPU:** don't bother with cross-machine DDP. With one GPU per host + only LAN, the
slowest-GPU bottleneck plus Ethernet AllReduce makes it net-negative vs. single-GPU on
deepthought. For parallelism, run **independent jobs** per machine (different folds/seeds), not
a split single job. (Full analysis lived in BirdCLEF's `multigpu.md` — not copied here because
it's specific to that audio pipeline.)

## Python Script Execution Policy

All Python scripts **MUST** run in the background with `nohup` and a timestamped log in `log/`.

### ⚠️ NEVER use `conda run` for scripts that write log files
`conda run` buffers stdout/stderr — the log file stays empty while the process runs. Activate
the env directly, then `nohup`.

```bash
# ✅ CORRECT  (local = skynet; use `kaggle` on deepthought)
conda activate kaggle-arch
cd /home/swatson/work/kaggle/ROGII
rm -f log/train_*.log                     # clean before each new training run
nohup python -u src/train.py > log/train_$(date +%Y%m%d_%H%M%S).log 2>&1 &
tail -f log/train_*.log
```

- Core implementation lives in `src/*.py` — **not** notebooks. Notebooks (`jupyter/`) are for
  Kaggle submission only.
- Shell scripts in `scripts/` must use **absolute paths**.
- Clean `log/train_*.log` before each new training run.

## Monitoring

```bash
tail -f log/<name>_*.log          # live log
jobs -l ; ps aux | grep python    # running processes
kill <PID>                        # stop
```

---

# CLAUDE.md — 12-rule template

These rules apply to every task in this project unless explicitly overridden.
Bias: caution over speed on non-trivial work. Use judgment on trivial tasks.

## Rule 1 — Think Before Coding
State assumptions explicitly. If uncertain, ask rather than guess. Present multiple
interpretations when ambiguity exists. Push back when a simpler approach exists. Stop when
confused — name what's unclear.

## Rule 2 — Simplicity First
Minimum code that solves the problem. Nothing speculative. No abstractions for single-use code.

## Rule 3 — Surgical Changes
Touch only what you must. Don't "improve" adjacent code. Match existing style.

## Rule 4 — Goal-Driven Execution
Define success criteria. Loop until verified.

## Rule 5 — Use the model only for judgment calls
Code answers what code can answer. Don't use the model for deterministic transforms.

## Rule 6 — Token budgets are not advisory
If approaching budget, summarize and start fresh. Surface the breach.

## Rule 7 — Surface conflicts, don't average them
If two patterns contradict, pick one (more recent / more tested), explain why, flag the other.

## Rule 8 — Read before you write
Before adding code, read exports, immediate callers, shared utilities.

## Rule 9 — Tests verify intent, not just behavior
Tests must encode WHY behavior matters, not just WHAT it does.

## Rule 10 — Checkpoint after every significant step
Summarize what was done, what's verified, what's left.

## Rule 11 — Match the codebase's conventions, even if you disagree
Conformance > taste inside the codebase. Surface harmful conventions; don't fork silently.

## Rule 12 — Fail loud
"Completed" is wrong if anything was skipped silently. Default to surfacing uncertainty.
