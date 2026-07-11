"""Tests for the deterministic replay engine."""

import pytest

from flightrec.sdk import TraceContext
from flightrec.replay import ReplayEngine, ReplayConfig


def _record_two_call_run(store):
    with TraceContext(name="orig", agent_name="a", store=store) as ctx:
        ctx.llm_call(model="claude-sonnet-5", prompt="p1", response="alpha",
                     tokens_in=10, tokens_out=5, cost=0.02)
        ctx.llm_call(model="claude-sonnet-5", prompt="p2", response="beta",
                     tokens_in=20, tokens_out=8, cost=0.03)
    return ctx.run.id


def test_replay_is_deterministic(store):
    run_id = _record_two_call_run(store)
    engine = ReplayEngine(store)

    served = []

    def agent_fn(ctx):
        # agent "asks" but the recorded response is what comes back
        served.append(ctx.llm_call(model="x", prompt="whatever"))
        served.append(ctx.llm_call(model="x", prompt="whatever2"))

    result = engine.replay(ReplayConfig(run_id=run_id), agent_fn)

    assert result.llm_calls_replayed == 2
    assert result.steps_replayed == 2
    # served responses match the recording, regardless of the prompts sent
    replayed = store.get_run(result.replay_run_id)
    assert [c["response"] for c in replayed["llm_calls"]] == ["alpha", "beta"]
    # cost is reproduced from the recording
    assert result.total_cost == pytest.approx(0.05)


def test_replay_exhaustion_raises_when_agent_changes(store):
    run_id = _record_two_call_run(store)
    engine = ReplayEngine(store)

    def greedy_agent(ctx):
        ctx.llm_call(model="x", prompt="1")
        ctx.llm_call(model="x", prompt="2")
        ctx.llm_call(model="x", prompt="3")  # one more than recorded

    with pytest.raises(RuntimeError, match="Replay exhausted"):
        engine.replay(ReplayConfig(run_id=run_id), greedy_agent)


def test_replay_missing_run_raises(store):
    engine = ReplayEngine(store)
    with pytest.raises(ValueError, match="Run not found"):
        engine.replay(ReplayConfig(run_id="ghost"), lambda ctx: None)


def test_model_override_applied(store):
    run_id = _record_two_call_run(store)
    engine = ReplayEngine(store)

    def agent_fn(ctx):
        ctx.llm_call(model="ignored", prompt="p")
        ctx.llm_call(model="ignored", prompt="p")

    result = engine.replay(
        ReplayConfig(run_id=run_id, model_override="claude-haiku-4-5"), agent_fn
    )
    replayed = store.get_run(result.replay_run_id)
    assert all(c["model"] == "claude-haiku-4-5" for c in replayed["llm_calls"])


def test_diff_replays_identical_vs_different(store):
    run_id = _record_two_call_run(store)
    engine = ReplayEngine(store)

    def two_calls(ctx):
        ctx.llm_call(model="x", prompt="p")
        ctx.llm_call(model="x", prompt="p")

    a = engine.replay(ReplayConfig(run_id=run_id), two_calls)
    b = engine.replay(ReplayConfig(run_id=run_id), two_calls)
    diff = engine.diff_replays(a, b)
    assert diff["verdict"] == "identical"
    assert diff["delta"]["steps"] == 0
