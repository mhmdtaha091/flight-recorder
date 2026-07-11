# CLAUDE.md — Flight Recorder

Local-first, framework-agnostic tracing/replay/evals harness for AI agents.
Records every LLM + tool call (tokens, cost, latency) to SQLite; deterministic
replay; golden-task scoring; regression diffs; FastAPI dashboard.
Plan: `_docs/plans/AGENT_FLIGHT_RECORDER_PLAN.md`. This project exists to back
the "evals & observability" resume claim. **Not yet pushed — no commits, no
remote; treat as fragile.**

## Commands

```bash
pip install -e ".[dev]"        # Python >= 3.11
pytest                          # 40 tests (tests/) — all green
python examples/demo.py         # offline end-to-end demo (no API key)
flightrec --db traces.db list   # recent runs
flightrec --db traces.db show <run-id>
flightrec --db traces.db stats
flightrec --db traces.db serve  # dashboard → http://localhost:8421
```

## Architecture (flightrec/)

- `__init__.py` — public SDK: `TraceContext` context manager + `@trace`
  decorator; `TraceStore` (SQLite: runs → llm_calls | tool_calls).
- `replay.py` — deterministic replay from recorded responses; A/B mode (swap
  model/prompt, diff outcomes).
- `scoring.py` — golden-task suites (YAML): rule checks + LLM-judge.
- `regression.py` — run-set comparison: pass-rate delta, per-task diffs,
  cost/latency movement, flake detection (re-run failures 3×).
- `cli.py` — click CLI (`flightrec` entry point).
- `dashboard/` — single-page UI served by FastAPI.

## Gotchas (learned the hard way — verified 2026-07-06)

- `TraceStore.conn` must set `row_factory = sqlite3.Row` or every `dict(row)`
  query crashes — the whole read path (CLI/dashboard/scoring/replay/regression)
  depends on it. Covered by tests now; don't remove it.
- `pyproject.toml` build-backend must be `setuptools.build_meta` (a bogus value
  makes `pip install` fail entirely).
- Windows: `format_report()` and CLI output contain emoji/box-drawing chars;
  plain `print()` crashes on a cp1252 console — reconfigure stdout to UTF-8
  (see `examples/demo.py`). The Rich-based CLI handles this itself.
- Regression cost/latency must be attributed per version by diffing the run set
  before/after each suite — not by sampling "recent runs" (that mixes B into A).

## Design rules

- **Local-first, no telemetry, no SaaS** — this is the product positioning;
  never add network calls that ship trace data anywhere.
- Framework-agnostic: must keep working on hand-rolled agent loops (Jarvis,
  PentestAI's custom ReAct), not just frameworks.
- Emit OpenTelemetry GenAI semantic-convention spans where practical.
- Positioning honesty in README: LangSmith/Langfuse/Braintrust exist; the
  differentiators are local-first + replay-centric + framework-agnostic. Never
  claim to beat them.
- Dogfood targets (in order): Jarvis 5 core jobs → BugScout → PentestAI
  benchmark playbook (`docs/comparison-playbook.md` empty cells).
