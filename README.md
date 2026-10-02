# deepseek-self-improver

A **local, autonomous learning loop** for `deepseek-r1:1.5b` running under [Ollama](https://ollama.com). One button
(GUI) or one command (CLI) runs a timed session: discover internet-sourced tasks -> ask the model -> **independently
verify** -> retry with verifier feedback (max 3) -> keep only verified lessons -> measure progress with a fixed
objective benchmark -> write a report. No cloud LLM is used; CPU-first; PyTorch is optional.

> **Status of this build.** Built and unit/integration-tested in a cloud sandbox that has **no Ollama and no
> internet**. All Ollama and web access is mocked in the test-suite (114 tests). **No real DeepSeek inference and no
> live public-source retrieval was performed during the build.** See "Final local validation" below.

## Quick start (Windows)

1. Install Python 3.10+ (with tcl/tk) and [Ollama](https://ollama.com/download).
2. Double-click **`SETUP.bat`** (creates `.venv`, installs deps, pulls `deepseek-r1:1.5b`).
3. Double-click **`START_SELF_IMPROVER.bat`** (starts `ollama serve` if needed, then opens the GUI). No PowerShell
   execution-policy change is needed - the scripts are plain batch files.

Optional: set `SELF_IMPROVER_CONTACT=you@example.com` so the User-Agent sent to SEC EDGAR contains a contact (SEC
requires one and may return 403 otherwise; the adapter then just falls back to other sources).

## Commands

```
python -m app.main doctor            # checks python/sympy/tkinter/db/Ollama/model/hardware
python -m app.main run --minutes 30  # headless session (GUI offers 20/30/45)
python -m app.main benchmark         # fixed 35-task benchmark (add --quick for the 12-task subset)
python -m app.main stats
python -m app.main lessons [--domain statistics]
python -m app.main prepare-training  # verified-only JSONL dataset in data/training/
python -m app.main train-adapter     # optional LoRA; refuses on CPU-only machines
python -m app.main gui
```
`run` flags: `--skip-benchmark`, `--quick-benchmark`. `--model TAG` overrides the model.

## How a session works

| step | where |
|---|---|
| 1-2 check Ollama + model | `app/ollama/client.py`, `Controller.preflight` |
| before-benchmark (outside the learning deadline; reused if nothing changed) | `app/benchmark/runner.py` |
| 3 hard deadline starts | `Controller._learning_loop` (monotonic clock; model calls are cancelled at the deadline/STOP) |
| 4 discover task (fallback chain, de-dup, resume pending) | `app/discovery/` |
| 5-6 solve via `/api/chat`, `stream=false`, `think=true` for difficulty >= 3, robust parsing | `app/solver/`, `app/ollama/parsing.py` |
| 7 independent verification | `app/verifier/` |
| 8 retry <= 3 with the verifier failure fed back (expected answer is never shown) | `app/solver/solver.py` |
| 9-10 save + lessons **only from verified** results | `app/memory/lessons.py` (DB-level guard) |
| 12 after-benchmark, 13 JSON + Markdown report | `app/reporting/report.py` -> `reports/` |

Total wall time = before-benchmark + chosen minutes + after-benchmark (the minutes budget covers the learning loop).
On slow CPUs use `--quick-benchmark` (12 fixed tasks) to keep the benchmarks short. `STOP` ends the loop, skips the
after-benchmark and still writes a report (no improvement is claimed).

### Honest improvement measurement
"Improved" is reported **only** when the objective benchmark score (deterministic verifier, fixed tasks, temperature 0,
fixed seed, tasks never used for lessons/training) is higher after than before. The model never grades itself; its
`self_check` and confidence are stored but ignored. The benchmark runs *with* the same lesson-retrieval procedure
before and after, so a gain reflects verified lessons accumulated in memory. Small models are noisy: treat a +1 task
change on a 12-35 task benchmark with caution.

### Verification (never self-verification)
`verify(task, model_answer) -> VerificationResult(status in verified|incorrect|uncertain, failure_category, ...)`.
Only `verified` counts; `uncertain` (ambiguous answer, verifier error, source unavailable...) is **never** verified.

* math/statistics/probability: exact `Fraction`/`Decimal` recomputation, SymPy for symbolic answers (derivatives,
  integrals, roots); Project Euler via independent reference solvers.
* finance: independent formula recomputation (FV, PV, NPV, bond price, loan payment, CAGR, margin, growth, EAR) with
  unit/date/range validation; inputs are re-fetched from the public source when possible (mismatch => task invalid;
  source unreachable => recorded as `unavailable` in the verification details with URL + timestamp).
* logic: truth-table enumeration with a restricted AST evaluator (no `eval`).
* coding: model code runs in a fresh temp dir, `python -I`, scrubbed env, wall-clock timeout, POSIX rlimits, against
  hidden tests. *This contains accidents; it is not a hardened sandbox.*
* failure categories (`arithmetic, formula, interpretation, unit_conversion, sign_error, unsupported_assumption,
  missing_data, hallucination, date_mismatch, source_mismatch, coding_error, other`) are heuristic labels assigned by
  the verifier (e.g. exact negation => `sign_error`, x100 => `unit_conversion`); the pass/fail decision never depends on them.

### Internet discovery (all real public endpoints; nothing invented)
| adapter | endpoint | tasks |
|---|---|---|
| `wikipedia_concepts` | `en.wikipedia.org/api/rest_v1/page/summary/<title>` | probability, logic, numerical, math, coding. The page is a **concept reference** (`source_role: concept_reference`); numbers are generated locally and answers verified locally. |
| `project_euler` | `projecteuler.net/minimal=<n>` | statement fetched live, checked for expected wording, verified with local reference solvers (15 solvable problems) |
| `nist_strd` | `itl.nist.gov/div898/strd/univ/data/<name>.dat` | statistics on a window of the retrieved dataset |
| `us_treasury` | `api.fiscaldata.treasury.gov/.../avg_interest_rates` | finance formulas using the retrieved average interest rate |
| `sec_edgar` | `data.sec.gov/api/xbrl/companyfacts/CIK<cik>.json` | net margin / revenue growth from retrieved 10-K facts |

Adapters are tried round-robin with fallback; failures are logged in the `sources` table; if every source fails the
session reports the errors instead of fabricating tasks. **These parsers were written from the public documentation
and could not be exercised against the live sites from the build sandbox** - the first real run on your PC is the
validation (a failing adapter simply falls back and shows up in the report's `sources`/`errors`).

### Storage
`data/self_improver.db` (SQLite): `sessions, tasks, attempts, verifications, lessons, benchmarks, sources`. Everything
persists across restarts; duplicate tasks are rejected by content hash; sessions killed mid-run are marked
`interrupted` on next start (report written, unfinished tasks re-queued and served first).

### Training (optional)
Normal operation never imports PyTorch/Transformers/PEFT. On CPU-only machines nothing is trained automatically;
`prepare-training` keeps a clean dataset (`data/training/train.jsonl`, verified attempts only, benchmark excluded, the
model's own `self_check` replaced by a neutral note). `train-adapter` (needs `pip install -r requirements-training.txt`
and a CUDA/XPU GPU) trains a LoRA adapter into `data/adapters/adapter_<timestamp>/`; the base model is loaded read-only
and never saved or overwritten, and Ollama's model store is never touched.

## Tests
```
pip install -r requirements-dev.txt
python -m pytest                     # 114 tests, fully mocked Ollama + internet
```
Covers Ollama unavailable/model missing/malformed/timeouts, source failure + fallback, duplicates, SQLite persistence,
math/finance/logic/code verification, wrong answers, retries and the 3-retry cap, verified-only lessons, benchmark
isolation + scoring, time limit, STOP, recovery/resume, reports, GUI presenter + real Tk window (skipped without a
display) driving the shared controller, and an end-to-end session against a fake Ollama backend.

## Final local validation (must be run on the PC with Ollama)
```
ollama pull deepseek-r1:1.5b
python -m app.main doctor
set RUN_LIVE_OLLAMA=1 && python -m pytest tests/live -s          (PowerShell: $env:RUN_LIVE_OLLAMA=1; python -m pytest tests/live -s)
python -m app.main run --minutes 2 --skip-benchmark              # 2-minute real smoke test
python -m app.main run --minutes 20 --quick-benchmark            # first real before/after session
```
With a 1.5B model on CPU a single "thinking" answer can take minutes; in a 2-minute smoke test a task may be cut off
by the deadline (reported as aborted/re-queued, not as a failure). That still proves the pipeline end-to-end; check
`reports/` and `logs/self_improver.log`.

## Known limitations
* Not validated against a real Ollama/DeepSeek or the live public sources (see above).
* Failure-category labels are heuristics. Lessons' `correct_method` text is the model's explanation of an answer that
  was verified correct; the answer is verified, the explanation is not.
* Code isolation is process-level, not a security sandbox. Don't point this at untrusted prompts.
* Some adapter tasks may be too hard for a 1.5B model; that is expected and simply yields `failed` results.
