"""Shared pytest fixtures for Flight Recorder tests."""

import pytest

from flightrec.store import TraceStore
from flightrec.sdk import TraceContext


@pytest.fixture
def store(tmp_path):
    """A fresh SQLite store on a temp path, closed after the test."""
    s = TraceStore(str(tmp_path / "test.db"))
    # touch the connection so the schema is created
    _ = s.conn
    yield s
    s.close()


@pytest.fixture
def sample_run(store):
    """Persist one completed run with 1 LLM call + 1 tool call; return its id."""
    with TraceContext(name="sample", agent_name="tester", store=store) as ctx:
        ctx.llm_call(
            model="claude-haiku-4-5",
            prompt="Summarize the inbox",
            response="You have 3 unread emails.",
            tokens_in=50,
            tokens_out=20,
            cost=0.0001,
            latency_ms=120,
        )
        ctx.tool_call(
            name="search_email",
            input={"query": "unread"},
            output={"count": 3},
            latency_ms=30,
        )
    return ctx.run.id
