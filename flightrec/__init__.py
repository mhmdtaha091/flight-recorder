"""
Flight Recorder — Local-first evaluation, tracing & replay harness for AI agents.

Record every LLM call, tool call, token count, cost, and latency.
Replay deterministically. Score against golden tasks. Diff regressions.
"""

from flightrec.sdk import (
    trace,
    TraceContext,
    trace_llm,
    trace_tool,
    get_active_run,
    estimate_cost,
)
from flightrec.store import TraceStore

__all__ = [
    "trace",
    "TraceContext",
    "trace_llm",
    "trace_tool",
    "get_active_run",
    "estimate_cost",
    "TraceStore",
]
__version__ = "0.1.0"
