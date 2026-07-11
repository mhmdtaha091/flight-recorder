"""Tests for the SQLite trace store: persistence, queries, cascades, golden tasks."""

import pytest

from flightrec.sdk import TraceContext


def test_save_and_get_run(store, sample_run):
    run = store.get_run(sample_run)
    assert run is not None
    assert run["name"] == "sample"
    assert run["status"] == "completed"
    assert len(run["llm_calls"]) == 1
    assert len(run["tool_calls"]) == 1
    assert run["llm_calls"][0]["response"] == "You have 3 unread emails."
    assert run["tool_calls"][0]["name"] == "search_email"


def test_get_missing_run_returns_none(store):
    assert store.get_run("does-not-exist") is None


def test_calls_ordered_by_step_index(store):
    with TraceContext(name="ordered", store=store) as ctx:
        for i in range(5):
            ctx.llm_call(model="m", prompt=f"p{i}", response=f"r{i}")
    run = store.get_run(ctx.run.id)
    steps = [c["step_index"] for c in run["llm_calls"]]
    assert steps == sorted(steps)
    assert [c["response"] for c in run["llm_calls"]] == [f"r{i}" for i in range(5)]


def test_list_runs_filters_and_orders(store):
    with TraceContext(name="a1", agent_name="alpha", store=store):
        pass
    with TraceContext(name="b1", agent_name="beta", store=store):
        pass
    assert len(store.list_runs()) == 2
    alpha = store.list_runs(agent_name="alpha")
    assert len(alpha) == 1 and alpha[0]["agent_name"] == "alpha"
    completed = store.list_runs(status="completed")
    assert len(completed) == 2


def test_search_runs(store):
    with TraceContext(name="morning-briefing", store=store):
        pass
    with TraceContext(name="evening-recap", store=store):
        pass
    hits = store.search_runs("briefing")
    assert len(hits) == 1
    assert hits[0]["name"] == "morning-briefing"


def test_stats_aggregate(store, sample_run):
    with TraceContext(name="failed-run", store=store) as ctx:
        try:
            with TraceContext(name="inner", store=store):
                raise RuntimeError("x")
        except RuntimeError:
            pass
    stats = store.get_stats()
    assert stats["total_runs"] >= 1
    assert stats["completed_runs"] >= 1
    assert stats["total_tokens"] >= 70  # 50 in + 20 out from sample_run


def test_delete_cascades_to_calls(store, sample_run):
    assert store.get_run(sample_run) is not None
    assert store.delete_run(sample_run) is True
    assert store.get_run(sample_run) is None
    # calls are gone too (FK ON DELETE CASCADE)
    rows = store.conn.execute(
        "SELECT COUNT(*) FROM llm_calls WHERE run_id = ?", (sample_run,)
    ).fetchone()[0]
    assert rows == 0


def test_golden_tasks_roundtrip(store):
    store.save_golden_task({
        "id": "t1",
        "suite_name": "daily",
        "task_name": "Summarize inbox",
        "checks": [{"type": "tool_called", "tool": "search_email"}],
        "judge_prompt": "Useful?",
    })
    store.save_golden_task({
        "id": "t2", "suite_name": "daily", "task_name": "Other", "enabled": False,
    })
    tasks = store.get_golden_tasks("daily")
    # only enabled task returned
    assert len(tasks) == 1
    assert tasks[0]["task_name"] == "Summarize inbox"


def test_eval_results_roundtrip(store, sample_run):
    store.save_eval_result({
        "id": "e1", "run_id": sample_run, "task_id": "t1",
        "passed": True, "score": 0.9, "details": {"why": "good"},
    })
    results = store.get_eval_results(sample_run)
    assert len(results) == 1
    assert results[0]["passed"] == 1
    assert results[0]["score"] == pytest.approx(0.9)
