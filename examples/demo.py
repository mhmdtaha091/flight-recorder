"""
End-to-end Flight Recorder demo — no API key, no network, fully deterministic.

Exercises the whole pipeline against a mock email-triage agent:
  1. Record a run (trace LLM + tool calls to SQLite)
  2. Score it against a golden task suite (rule checks, no LLM judge)
  3. Replay it deterministically (recorded responses, zero API calls)
  4. Regression-diff two agent versions (v1 vs v2)

Run:   python examples/demo.py
Then:  flightrec --db flightrec-demo.db list
       flightrec --db flightrec-demo.db serve   # dashboard at :8421
"""

from __future__ import annotations

import os
import sys

# Reports contain box-drawing/emoji chars; force UTF-8 so plain print() doesn't
# crash on a legacy Windows (cp1252) console.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from flightrec import TraceContext, TraceStore, estimate_cost
from flightrec.scoring import TaskScorer
from flightrec.replay import ReplayEngine, ReplayConfig
from flightrec.regression import RegressionRunner

DB = "flightrec-demo.db"


def make_agent(store: TraceStore, version: str):
    """A mock email-triage agent. v2 also categorizes email (an extra tool call)."""

    def agent_fn(task_input: dict) -> str:
        with TraceContext(
            name=f"triage-{version}", agent_name="mailbot", tags=[version], store=store
        ) as ctx:
            ctx.tool_call(name="search_email", input=task_input, output={"unread": 3})
            if version == "v2":
                ctx.tool_call(name="categorize", input={"n": 3}, output={"urgent": 1, "fyi": 2})
            tin, tout = 120, 40
            response = (
                "You have 3 unread emails: 1 urgent, 2 FYI."
                if version == "v2"
                else "You have 3 unread emails."
            )
            ctx.llm_call(
                model="claude-haiku-4-5",
                prompt="Summarize my unread email",
                response=response,
                tokens_in=tin,
                tokens_out=tout,
                cost=estimate_cost("claude-haiku-4-5", tin, tout),
                latency_ms=140,
            )
        return ctx.run.id

    return agent_fn


def main() -> None:
    if os.path.exists(DB):
        os.remove(DB)
    store = TraceStore(DB)

    print("── 1. Record a run ─────────────────────────────────────────")
    run_id = make_agent(store, "v2")({"folder": "inbox"})
    run = store.get_run(run_id)
    print(f"   run {run_id[:8]}  •  {run['step_count']} steps  •  ${run['total_cost']:.5f}")

    print("\n── 2. Score against a golden suite (rule checks only) ──────")
    store.save_golden_task({
        "id": "t-summary",
        "suite_name": "mail",
        "task_name": "summarize inbox",
        "input": {"folder": "inbox"},
        "checks": [
            {"type": "tool_called", "tool": "search_email"},
            {"type": "tool_called", "tool": "categorize"},
            {"type": "contains", "value": "unread"},
            {"type": "cost_below", "max_cost": 0.01},
            {"type": "no_errors"},
        ],
    })
    scorer = TaskScorer(store)
    res = scorer.run_suite("mail", make_agent(store, "v2"), use_judge=False)
    print(f"   pass rate {res['pass_rate']:.0%}  ({res['passed']}/{res['total']} tasks)")

    print("\n── 3. Replay the recorded run deterministically ────────────")
    engine = ReplayEngine(store)

    def replay_agent(ctx: TraceContext) -> None:
        # Same shape as the recorded agent; the LLM response comes from the recording.
        ctx.tool_call(name="search_email", input={}, output={})
        ctx.tool_call(name="categorize", input={}, output={})
        ctx.llm_call(model="ignored", prompt="Summarize my unread email")

    rr = engine.replay(ReplayConfig(run_id=run_id), replay_agent)
    print(
        f"   replayed {rr.llm_calls_replayed} LLM call(s) from the recording  •  "
        f"${rr.total_cost:.5f}  •  {rr.duration_ms:.1f}ms  •  0 live API calls"
    )

    print("\n── 4. Regression: v1 vs v2 on the same suite ───────────────")
    runner = RegressionRunner(store)
    report = runner.compare(
        "mail",
        make_agent(store, "v1"),  # baseline: no categorize → fails a check
        make_agent(store, "v2"),  # candidate: categorizes → passes
        version_a="v1",
        version_b="v2",
        use_judge=False,
    )
    print(runner.format_report(report))

    store.close()
    print(f"\nDone. Explore it:  flightrec --db {DB} list")


if __name__ == "__main__":
    main()
