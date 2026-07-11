"""
Flight Recorder SDK — decorator and context-manager for instrumenting AI agents.

Usage:
    # Context manager (recommended for agent loops)
    with TraceContext(name="jarvis-morning-briefing") as ctx:
        ctx.llm_call(model="claude-sonnet-5", prompt="...", response="...", ...)
        ctx.tool_call(name="search_email", input={...}, output={...}, ...)

    # Decorator (for simple functions)
    @trace(name="summarize-inbox")
    def summarize_inbox(messages): ...

    # Manual tracing
    trace_llm(run_id, model="...", prompt="...", response="...")
    trace_tool(run_id, name="...", input={...}, output={...})
"""

from __future__ import annotations

import time
import uuid
import json
import functools
from contextvars import ContextVar
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Optional, Dict

# --- Thread-safe context for active trace runs ---
_active_run: ContextVar[Optional["TraceContext"]] = ContextVar(
    "flightrec_active_run", default=None
)


@dataclass
class LLMCallRecord:
    """A single LLM API call within a run."""

    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    run_id: str = ""
    step_index: int = 0
    model: str = ""
    provider: str = "anthropic"
    prompt: str = ""
    response: str = ""
    tokens_in: int = 0
    tokens_out: int = 0
    cost: float = 0.0
    latency_ms: float = 0.0
    timestamp: float = field(default_factory=time.time)


@dataclass
class ToolCallRecord:
    """A tool invocation within a run."""

    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    run_id: str = ""
    step_index: int = 0
    name: str = ""
    input_json: str = "{}"
    output_json: str = "{}"
    success: bool = True
    error: str = ""
    latency_ms: float = 0.0
    timestamp: float = field(default_factory=time.time)


@dataclass
class RunRecord:
    """Top-level record for a single agent run."""

    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    name: str = ""
    agent_name: str = ""
    status: str = "running"  # running | completed | failed
    tags: str = ""  # comma-separated
    metadata_json: str = "{}"
    total_tokens_in: int = 0
    total_tokens_out: int = 0
    total_cost: float = 0.0
    total_latency_ms: float = 0.0
    step_count: int = 0
    started_at: float = field(default_factory=time.time)
    ended_at: Optional[float] = None


class TraceContext:
    """
    Context manager for tracing an agent run.

    Usage:
        with TraceContext(name="my-run", agent="jarvis") as ctx:
            ctx.llm_call(model="claude-sonnet-5", prompt="...", response="...",
                         tokens_in=100, tokens_out=50)
            ctx.tool_call(name="send_email", input={...}, output={...})
    """

    def __init__(
        self,
        name: str = "",
        agent_name: str = "",
        tags: Optional[list[str]] = None,
        metadata: Optional[dict] = None,
        store=None,  # Optional[TraceStore] — lazy import to avoid circular
    ):
        self.run = RunRecord(
            name=name or f"run-{uuid.uuid4().hex[:8]}",
            agent_name=agent_name,
            tags=",".join(tags) if tags else "",
            metadata_json=json.dumps(metadata or {}),
        )
        self._step_index = 0
        self._llm_calls: list[LLMCallRecord] = []
        self._tool_calls: list[ToolCallRecord] = []
        self._store = store

    @property
    def run_id(self) -> str:
        return self.run.id

    @property
    def step_count(self) -> int:
        return self._step_index

    def llm_call(
        self,
        model: str,
        prompt: str,
        response: str,
        *,
        tokens_in: int = 0,
        tokens_out: int = 0,
        cost: float = 0.0,
        latency_ms: float = 0.0,
        provider: str = "anthropic",
    ) -> LLMCallRecord:
        """Record an LLM API call within this run."""
        record = LLMCallRecord(
            run_id=self.run.id,
            step_index=self._step_index,
            model=model,
            provider=provider,
            prompt=prompt,
            response=response,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost=cost,
            latency_ms=latency_ms,
        )
        self._llm_calls.append(record)
        self.run.total_tokens_in += tokens_in
        self.run.total_tokens_out += tokens_out
        self.run.total_cost += cost
        self.run.total_latency_ms += latency_ms
        self._step_index += 1
        self.run.step_count = self._step_index
        return record

    def tool_call(
        self,
        name: str,
        input: Any = None,
        output: Any = None,
        *,
        success: bool = True,
        error: str = "",
        latency_ms: float = 0.0,
    ) -> ToolCallRecord:
        """Record a tool invocation within this run."""
        record = ToolCallRecord(
            run_id=self.run.id,
            step_index=self._step_index,
            name=name,
            input_json=json.dumps(input, default=str),
            output_json=json.dumps(output, default=str),
            success=success,
            error=error,
            latency_ms=latency_ms,
        )
        self._tool_calls.append(record)
        self.run.total_latency_ms += latency_ms
        self._step_index += 1
        self.run.step_count = self._step_index
        return record

    def __enter__(self) -> "TraceContext":
        self._token = _active_run.set(self)
        self.run.started_at = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.run.ended_at = time.time()
        self.run.total_latency_ms = (self.run.ended_at - self.run.started_at) * 1000

        if exc_type is not None:
            self.run.status = "failed"
        else:
            self.run.status = "completed"

        # Persist to store if provided
        if self._store is not None:
            try:
                self._store.save_run(self.run, self._llm_calls, self._tool_calls)
            except Exception as e:
                import sys

                print(f"[flightrec] Failed to persist run {self.run.id}: {e}", file=sys.stderr)

        _active_run.reset(self._token)


def trace(
    name: str = "",
    agent_name: str = "",
    tags: Optional[list[str]] = None,
    store=None,
):
    """
    Decorator: trace a function as a Flight Recorder run.

    Usage:
        @trace(name="morning-briefing", agent_name="jarvis")
        def run_briefing():
            ctx = get_active_run()
            ctx.llm_call(...)
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            run_name = name or func.__name__
            with TraceContext(
                name=run_name,
                agent_name=agent_name,
                tags=tags,
                store=store,
            ) as ctx:
                try:
                    result = func(*args, **kwargs)
                    return result
                except Exception:
                    ctx.run.status = "failed"
                    raise

        return wrapper

    return decorator


def get_active_run() -> Optional[TraceContext]:
    """Get the currently active TraceContext from anywhere in the call stack."""
    return _active_run.get()


# --- Standalone trace functions for manual use ---

def trace_llm(
    run_id: str,
    model: str,
    prompt: str,
    response: str,
    *,
    tokens_in: int = 0,
    tokens_out: int = 0,
    cost: float = 0.0,
    latency_ms: float = 0.0,
    provider: str = "anthropic",
    store=None,
) -> LLMCallRecord:
    """Record an LLM call outside of a TraceContext (manual mode)."""
    record = LLMCallRecord(
        run_id=run_id,
        model=model,
        provider=provider,
        prompt=prompt[:2000],  # truncate for storage
        response=response[:2000],
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        cost=cost,
        latency_ms=latency_ms,
    )
    if store:
        store.save_llm_call(record)
    return record


def trace_tool(
    run_id: str,
    name: str,
    input: Any = None,
    output: Any = None,
    *,
    success: bool = True,
    error: str = "",
    latency_ms: float = 0.0,
    store=None,
) -> ToolCallRecord:
    """Record a tool call outside of a TraceContext (manual mode)."""
    record = ToolCallRecord(
        run_id=run_id,
        name=name,
        input_json=json.dumps(input, default=str),
        output_json=json.dumps(output, default=str),
        success=success,
        error=error,
        latency_ms=latency_ms,
    )
    if store:
        store.save_tool_call(record)
    return record


def estimate_cost(model: str, tokens_in: int, tokens_out: int) -> float:
    """Estimate Claude API cost for a call."""
    rates: dict[str, tuple[float, float]] = {
        "claude-haiku-4-5-20251001": (0.80, 4.0),
        "claude-haiku-4-5": (0.80, 4.0),
        "claude-sonnet-5": (3.0, 15.0),
        "claude-opus-4-8": (15.0, 75.0),
        "claude-fable-5": (25.0, 100.0),
    }
    rate_in, rate_out = rates.get(model, (3.0, 15.0))
    return (tokens_in * rate_in + tokens_out * rate_out) / 1_000_000
