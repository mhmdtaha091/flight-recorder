"""Tests for regression comparison between two agent versions."""

import pytest

from flightrec.sdk import TraceContext
from flightrec.regression import RegressionRunner


def _suite(store):
    store.save_golden_task({
        "id": "t1", "suite_name": "s", "task_name": "inbox", "input": {},
        "checks": [{"type": "tool_called", "tool": "search_email"}],
    })


def _agent(store, *, call_tool: bool, cost: float = 0.0):
    def agent_fn(task_input):
        with TraceContext(name="run", agent_name="a", store=store) as ctx:
            if call_tool:
                ctx.tool_call(name="search_email", input=task_input, output={"n": 1})
            ctx.llm_call(model="m", prompt="p", response="ok", cost=cost)
        return ctx.run.id
    return agent_fn


def test_detects_improvement(store):
    _suite(store)
    runner = RegressionRunner(store)
    report = runner.compare(
        "s",
        _agent(store, call_tool=False),  # v1 fails
        _agent(store, call_tool=True),   # v2 passes
        use_judge=False,
    )
    assert len(report.improvements) == 1
    assert len(report.regressions) == 0
    assert "strictly better" in report.verdict
    assert report.pass_rate_delta == pytest.approx(1.0)


def test_detects_regression(store):
    _suite(store)
    runner = RegressionRunner(store)
    report = runner.compare(
        "s",
        _agent(store, call_tool=True),   # v1 passes
        _agent(store, call_tool=False),  # v2 fails
        use_judge=False,
    )
    assert len(report.regressions) == 1
    assert len(report.improvements) == 0
    assert "worse" in report.verdict


def test_cost_attribution_is_per_version(store):
    """v1 is free, v2 costs 0.10/run — the report must attribute cost correctly.

    Regression: both were sampled from the most-recent runs *after both* suites
    ran, so cost_a == cost_b. This asserts they are separated per version.
    """
    _suite(store)
    runner = RegressionRunner(store)
    report = runner.compare(
        "s",
        _agent(store, call_tool=True, cost=0.0),   # v1 cheap
        _agent(store, call_tool=True, cost=0.10),  # v2 expensive
        use_judge=False,
    )
    assert report.cost_a == pytest.approx(0.0)
    assert report.cost_b == pytest.approx(0.10)
    assert report.cost_delta == pytest.approx(0.10)


def test_format_report_renders_markdown(store):
    _suite(store)
    runner = RegressionRunner(store)
    report = runner.compare(
        "s", _agent(store, call_tool=True), _agent(store, call_tool=True),
        version_a="baseline", version_b="candidate", use_judge=False,
    )
    md = runner.format_report(report)
    assert "Pass Rate" in md
    assert "baseline" in md and "candidate" in md
