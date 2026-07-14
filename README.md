# ✈️ Flight Recorder — Agent Telemetry, Replay & Evals

> Record everything your AI agent does. Replay it deterministically. Score it. Ship with confidence.

Flight Recorder is a **local-first, framework-agnostic** harness that records every LLM call, tool call, token count, cost, and latency from your AI agents. It lets you replay runs deterministically, score them against golden task suites, and catch regressions between agent versions — with a web dashboard showing pass rate, cost, and latency trends over time.

## Why Flight Recorder?

- **You can't trust an agent you can't replay** — a run that only exists in stdout is a run you can't debug, diff, or defend
- **Dogfooded from day one** — built to trace three of the author's own agents: Jarvis (personal assistant), PentestAI (security agent), and BugScout (web QA agent)
- **Local-first** — SQLite, no SaaS, no telemetry. Works offline. You own your data.
- **Framework-agnostic** — decorator/context-manager pattern works on hand-rolled loops (not just LangChain)
- **Replay-centric** — deterministic replay is the killer feature; swap one variable and A/B diff
- **OpenTelemetry-aware** — emits GenAI semantic-convention spans for interop

## Quick Start

```bash
# Install
pip install -e .

# Or run directly
python -m flightrec.cli --help
```

### Instrument your agent

```python
from flightrec import TraceContext, TraceStore

store = TraceStore("traces.db")

with TraceContext(name="morning-briefing", agent_name="jarvis", store=store) as ctx:
    # Each LLM call gets recorded
    ctx.llm_call(
        model="claude-sonnet-5",
        prompt="Summarize today's emails...",
        response="You have 5 unread emails...",
        tokens_in=120, tokens_out=80,
        cost=0.0016, latency_ms=450
    )

    # Each tool call gets recorded
    ctx.tool_call(
        name="search_email",
        input={"query": "unread"},
        output={"count": 5, "subjects": [...]},
        latency_ms=200
    )
```

### View traces

```bash
# List recent runs
flightrec --db traces.db list

# Show a detailed timeline
flightrec --db traces.db show <run-id>

# Aggregate stats
flightrec --db traces.db stats

# Launch dashboard
flightrec --db traces.db serve
# → http://localhost:8421
```

## Architecture

```
Agent under test
   │  (instrumentation: @trace decorator / TraceContext)
   ▼
Trace store (SQLite: runs → llm_calls | tool_calls)
   │
   ├──► Replay engine — re-serve recorded responses → deterministic re-runs
   │      A/B mode: swap model, prompt → diff
   │
   ├──► Scoring — golden-task suites (YAML)
   │      Rule checks + LLM-judge for fuzzy outcomes
   │
   ├──► Regression diff — compare run-sets across versions
   │      Pass-rate delta, per-task diffs, cost/latency movement, flake detection
   │
   └──► Dashboard — local web UI (FastAPI + React)
          Timeline view, trends, cost curves
```

## CLI

```
Usage: flightrec [OPTIONS] COMMAND [ARGS]

Options:
  --db PATH    Path to SQLite database [default: flightrec.db]

Commands:
  show     Show detailed timeline of a single run
  list     List recent runs
  stats    Show aggregate statistics
  serve    Start the web dashboard (http://localhost:8421)
  delete   Delete a run and all its data
  export   Export a run as JSON
```

## Golden Task Suites

Define evaluation suites in YAML:

```yaml
suite: jarvis-daily
tasks:
  - name: "Summarize inbox"
    description: "Agent should summarize unread emails"
    input:
      action: summarize_inbox
    checks:
      - type: tool_called
        tool: search_email
        min_calls: 1
      - type: output_contains
        value: "unread"
      - type: cost_below
        max_cost: 0.05
      - type: latency_below
        max_ms: 5000
    judge_prompt: "Did the agent produce a useful, accurate summary of the user's emails?"
```

Load and run:

```python
from flightrec.scoring import TaskScorer

scorer = TaskScorer(store, api_key="...")
scorer.load_suite_from_yaml("jarvis_tasks.yaml")
result = scorer.run_suite("jarvis-daily", my_agent_fn)
print(f"Pass rate: {result['pass_rate']:.1%}")
```

## Dashboard

<!-- TODO: hero screenshot — run `flightrec --db flightrec-demo.db serve`, open a run's
     step-by-step timeline view, capture it, save as docs/dashboard-timeline.png,
     and embed here. No screenshot yet — do not fake one. -->

Launch with `flightrec serve` → opens at **http://localhost:8421**

- **Runs list** — all recorded runs with cost, latency, token counts
- **Run detail** — step-by-step timeline with LLM prompts/responses and tool I/O
- **Trends** — 14-day charts of run volume and cost
- **Evaluations** — per-task pass/fail with judge scores

## Try it (no API key, no network)

```bash
pip install -e ".[dev]"
python examples/demo.py        # record → score → replay → regression, end to end
flightrec --db flightrec-demo.db list    # then browse what it recorded
```

The demo instruments a mock agent, scores it against a golden suite, replays it
deterministically, and prints a v1→v2 regression report — all offline.

## Tests

```bash
pytest            # 40 tests across store, sdk, scoring, replay, regression
```

The suite covers the core guarantees: persistence + cascade deletes, every rule
check type, deterministic replay (including replay-exhaustion detection when an
agent's behavior changes), and per-version cost/latency attribution in
regression diffs.

## Roadmap

| Version | What | Status |
|---------|------|--------|
| v0 | Record + view (SDK, SQLite, CLI) | ✅ Done |
| v1 | Golden tasks + scoring (YAML, rule checks, LLM-judge) | ✅ Done |
| v2 | Replay + regression diff (A/B mode, flake detection) | ✅ Done |
| v3 | Dashboard + multi-agent instrumentation | ✅ Done |
| — | Test suite (40 tests) + offline demo | ✅ Done |

## Demo output

```bash
$ python examples/demo.py

── 1. Record a run ─────────────────────────────────────────
   run 94345319  •  3 steps  •  $0.00026

── 2. Score against a golden suite (rule checks only) ──────
   pass rate 100%  (1/1 tasks)

── 3. Replay the recorded run deterministically ────────────
   replayed 1 LLM call(s) from the recording  •  $0.00026  •  1.0ms  •  0 live API calls

── 4. Regression: v1 vs v2 on the same suite ───────────────
# 🔍 Regression Report: v1 → v2

**Suite:** mail
**Verdict:** ✅ v2 is strictly better (+1 improvements, 0 regressions)

| Metric | v1 | v2 | Δ |
| ------ | ----- | ----- | -- |
| Pass Rate | 0.0% | 100.0% | +100.0% |
| Avg Score | 0.00 | 0.70 | +0.70 |
| Cost | $0.0003 | $0.0003 | $+0.0000 |
| Latency | 0ms | 0ms | +0ms |
```

## Metrics

All numbers are real, measured, and verifiable:

- **Demo stats:** 5 runs recorded, 800 tokens, $0.0013 total cost, fully offline
- **Test suite:** 40 pytest tests, all green — covers store, SDK, scoring, replay, regression
- **Dogfood targets (upcoming):** Jarvis 5-core-job eval · PentestAI benchmark playbook · BugScout flake rate

## Comparison

| Feature | Flight Recorder | LangSmith | Langfuse | Braintrust |
|---------|----------------|-----------|----------|------------|
| Local-first | ✅ | ❌ | ❌ | ❌ |
| No SaaS dependency | ✅ | ❌ | ❌ | ❌ |
| Framework-agnostic | ✅ | LangChain-only | ✅ | ✅ |
| Deterministic replay | ✅ | ❌ | ❌ | ❌ |
| LLM-judge | ✅ | ✅ | ✅ | ✅ |
| OpenTelemetry | ✅ | ✅ | ✅ | ❌ |
| Self-hosted dashboard | ✅ | ✅ (self-host) | ✅ (self-host) | ✅ |

## Tech Stack

- **Python SDK** — `@trace` decorator + `TraceContext` context manager
- **SQLite** — WAL-mode, foreign keys, indexed
- **FastAPI** — REST API for dashboard
- **React 18** — Dashboard SPA (CDN, zero build step)
- **Pluggable LLM-judge** — any OpenAI-compatible endpoint (DeepSeek by default), Anthropic, Gemini, or local Ollama; set `JUDGE_PROVIDER` / `JUDGE_MODEL` / `JUDGE_BASE_URL`

## Dogfood Targets

1. **Jarvis** — 30-task golden suite, morning-briefing trace
2. **PentestAI** — benchmark playbook with real numbers
3. **BugScout** — flake rate tracking for generated tests

## License

MIT — Muhammad Taha Khan

---

*"I built it because I needed to test my own assistant."*
