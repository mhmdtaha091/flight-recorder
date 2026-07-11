"""Tests for the tracing SDK: TraceContext, decorator, cost estimation."""

import pytest

from flightrec.sdk import (
    TraceContext,
    trace,
    get_active_run,
    estimate_cost,
    LLMCallRecord,
    ToolCallRecord,
)


def test_context_records_calls_and_totals():
    with TraceContext(name="r", agent_name="a") as ctx:
        ctx.llm_call(model="m", prompt="p", response="x", tokens_in=10, tokens_out=5, cost=0.02, latency_ms=100)
        ctx.llm_call(model="m", prompt="p2", response="y", tokens_in=20, tokens_out=8, cost=0.03, latency_ms=50)
        ctx.tool_call(name="t", input={"a": 1}, output={"b": 2}, latency_ms=10)

    assert ctx.run.total_tokens_in == 30
    assert ctx.run.total_tokens_out == 13
    assert ctx.run.total_cost == pytest.approx(0.05)
    assert ctx.run.step_count == 3
    assert ctx.step_count == 3


def test_status_completed_on_clean_exit():
    with TraceContext(name="ok") as ctx:
        ctx.llm_call(model="m", prompt="p", response="r")
    assert ctx.run.status == "completed"
    assert ctx.run.ended_at is not None


def test_status_failed_on_exception():
    with pytest.raises(ValueError):
        with TraceContext(name="boom") as ctx:
            ctx.llm_call(model="m", prompt="p", response="r")
            raise ValueError("kaboom")
    assert ctx.run.status == "failed"


def test_active_run_contextvar():
    assert get_active_run() is None
    with TraceContext(name="active") as ctx:
        assert get_active_run() is ctx
    assert get_active_run() is None


def test_decorator_creates_run_and_persists(store):
    @trace(name="briefing", agent_name="jarvis", store=store)
    def do_work():
        get_active_run().tool_call(name="noop", input={}, output={})
        return 42

    assert do_work() == 42
    runs = store.list_runs()
    assert len(runs) == 1
    assert runs[0]["name"] == "briefing"
    assert runs[0]["agent_name"] == "jarvis"


def test_tool_call_records_failure():
    with TraceContext(name="r") as ctx:
        rec = ctx.tool_call(name="t", input={}, output=None, success=False, error="timeout")
    assert isinstance(rec, ToolCallRecord)
    assert rec.success is False
    assert rec.error == "timeout"


def test_estimate_cost_known_and_unknown_model():
    # haiku: 0.80 in / 4.0 out per Mtok
    cost = estimate_cost("claude-haiku-4-5", 1_000_000, 1_000_000)
    assert cost == pytest.approx(0.80 + 4.0)
    # unknown model falls back to sonnet-class rates (3.0 / 15.0)
    fallback = estimate_cost("some-unknown-model", 1_000_000, 0)
    assert fallback == pytest.approx(3.0)


def test_records_have_unique_ids():
    a = LLMCallRecord()
    b = LLMCallRecord()
    assert a.id != b.id
    assert len(a.id) == 12
