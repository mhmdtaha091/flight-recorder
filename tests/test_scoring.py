"""Tests for golden-task scoring: rule checks, suite runs, YAML loading, flakes."""

import pytest

from flightrec.sdk import TraceContext
from flightrec.scoring import RuleCheck, TaskScorer, detect_flakes


# ── RuleCheck unit tests (one per check type) ──

def _run_data(**over):
    base = {
        "llm_calls": [{"response": "You have 3 unread emails"}],
        "tool_calls": [{"name": "search_email", "success": True, "output_json": '{"count": 3}'}],
        "total_cost": 0.01,
        "total_latency_ms": 500,
        "status": "completed",
    }
    base.update(over)
    return base


def test_check_contains():
    assert RuleCheck({"type": "contains", "value": "unread"}).evaluate(_run_data())[0] is True
    assert RuleCheck({"type": "contains", "value": "spreadsheet"}).evaluate(_run_data())[0] is False


def test_check_matches_regex():
    assert RuleCheck({"type": "matches", "pattern": r"\d+ unread"}).evaluate(_run_data())[0] is True
    assert RuleCheck({"type": "matches", "pattern": r"\d+ deleted"}).evaluate(_run_data())[0] is False


def test_check_matches_invalid_regex_is_false():
    ok, reason = RuleCheck({"type": "matches", "pattern": "("}).evaluate(_run_data())
    assert ok is False
    assert "Invalid regex" in reason


def test_check_tool_called():
    assert RuleCheck({"type": "tool_called", "tool": "search_email"}).evaluate(_run_data())[0] is True
    assert RuleCheck({"type": "tool_called", "tool": "send_email"}).evaluate(_run_data())[0] is False
    assert RuleCheck({"type": "tool_called", "tool": "search_email", "min_calls": 2}).evaluate(_run_data())[0] is False


def test_check_cost_below():
    assert RuleCheck({"type": "cost_below", "max_cost": 0.05}).evaluate(_run_data())[0] is True
    assert RuleCheck({"type": "cost_below", "max_cost": 0.001}).evaluate(_run_data())[0] is False


def test_check_latency_below():
    assert RuleCheck({"type": "latency_below", "max_ms": 1000}).evaluate(_run_data())[0] is True
    assert RuleCheck({"type": "latency_below", "max_ms": 100}).evaluate(_run_data())[0] is False


def test_check_no_errors():
    assert RuleCheck({"type": "no_errors"}).evaluate(_run_data())[0] is True
    bad = _run_data(tool_calls=[{"name": "x", "success": False, "error": "boom"}])
    assert RuleCheck({"type": "no_errors"}).evaluate(bad)[0] is False
    assert RuleCheck({"type": "no_errors"}).evaluate(_run_data(status="failed"))[0] is False


def test_check_output_contains():
    assert RuleCheck({"type": "output_contains", "value": "count"}).evaluate(_run_data())[0] is True
    assert RuleCheck({"type": "output_contains", "value": "missing"}).evaluate(_run_data())[0] is False


def test_unknown_check_type():
    ok, reason = RuleCheck({"type": "nonsense"}).evaluate(_run_data())
    assert ok is False
    assert "Unknown check type" in reason


# ── Suite-level scoring against a mock agent (no LLM judge) ──

def _mock_agent(store):
    """Return an agent_fn that records a run and returns its run_id."""
    def agent_fn(task_input):
        with TraceContext(name="mock", agent_name="mock", store=store) as ctx:
            ctx.tool_call(name="search_email", input=task_input, output={"count": 3})
            ctx.llm_call(model="m", prompt="p", response="You have 3 unread emails")
        return ctx.run.id
    return agent_fn


def test_run_suite_all_pass(store):
    store.save_golden_task({
        "id": "t1", "suite_name": "daily", "task_name": "inbox",
        "input": {"q": "unread"},
        "checks": [
            {"type": "tool_called", "tool": "search_email"},
            {"type": "contains", "value": "unread"},
        ],
    })
    scorer = TaskScorer(store)
    result = scorer.run_suite("daily", _mock_agent(store), use_judge=False)
    assert result["total"] == 1
    assert result["passed"] == 1
    assert result["pass_rate"] == pytest.approx(1.0)
    # rule-only score is 0.7 weight → 0.7 when judge disabled
    assert result["results"][0]["score"] == pytest.approx(0.7)


def test_run_suite_failing_check(store):
    store.save_golden_task({
        "id": "t1", "suite_name": "daily", "task_name": "inbox",
        "input": {},
        "checks": [{"type": "contains", "value": "NEVER-PRESENT"}],
    })
    scorer = TaskScorer(store)
    result = scorer.run_suite("daily", _mock_agent(store), use_judge=False)
    assert result["passed"] == 0
    assert result["pass_rate"] == pytest.approx(0.0)


def test_run_suite_missing_suite(store):
    scorer = TaskScorer(store)
    result = scorer.run_suite("nope", _mock_agent(store))
    assert "error" in result
    assert result["pass_rate"] == 0.0


def test_load_suite_from_yaml(store, tmp_path):
    yaml_file = tmp_path / "suite.yaml"
    yaml_file.write_text(
        "suite: daily\n"
        "tasks:\n"
        "  - name: inbox\n"
        "    input:\n"
        "      q: unread\n"
        "    checks:\n"
        "      - type: tool_called\n"
        "        tool: search_email\n",
        encoding="utf-8",
    )
    scorer = TaskScorer(store)
    n = scorer.load_suite_from_yaml(str(yaml_file))
    assert n == 1
    assert len(store.get_golden_tasks("daily")) == 1


def test_detect_flakes_stable_agent_has_no_flakes(store):
    store.save_golden_task({
        "id": "t1", "suite_name": "daily", "task_name": "inbox", "input": {},
        "checks": [{"type": "tool_called", "tool": "search_email"}],
    })
    scorer = TaskScorer(store)
    report = detect_flakes(scorer, "daily", _mock_agent(store), runs=3)
    assert report["runs"] == 3
    assert report["flake_count"] == 0
