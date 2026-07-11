"""
Deterministic replay engine — re-serve recorded LLM responses to re-run agent code
without live API calls. Supports A/B mode: swap model/prompt variable and diff.

v2 per AGENT_FLIGHT_RECORDER_PLAN.md.
"""

from __future__ import annotations

import copy
import json
import time
from typing import Any, Callable, Optional
from dataclasses import dataclass, field

from flightrec.store import TraceStore
from flightrec.sdk import TraceContext, LLMCallRecord, ToolCallRecord


@dataclass
class ReplayConfig:
    """Configuration for a replay run."""

    run_id: str
    """Whether to actually execute tool calls or just replay their outputs"""
    replay_tools: bool = True
    """Override model name for all LLM calls (A/B testing)"""
    model_override: Optional[str] = None
    """Override prompt template — {prompt} is replaced with original"""
    prompt_template: Optional[str] = None
    """Intercept LLM calls with this function before replaying"""
    llm_interceptor: Optional[Callable[[LLMCallRecord], LLMCallRecord]] = None


@dataclass
class ReplayResult:
    """Result of a replay run."""

    original_run_id: str
    replay_run_id: str
    steps_replayed: int
    llm_calls_replayed: int
    tool_calls_replayed: int
    total_cost: float
    duration_ms: float
    diffs: list[dict] = field(default_factory=list)


class ReplayEngine:
    """
    Deterministic replay: re-serves recorded LLM responses.
    The agent code runs again, but LLM calls return recorded responses
    instead of making live API calls.
    """

    def __init__(self, store: TraceStore):
        self.store = store

    def load_run(self, run_id: str) -> Optional[dict]:
        """Load a recorded run for replay."""
        return self.store.get_run(run_id)

    def replay(
        self,
        config: ReplayConfig,
        agent_fn: Callable[[TraceContext], Any],
    ) -> ReplayResult:
        """
        Replay a recorded run. The agent_fn is called with a TraceContext
        that replays recorded LLM responses.

        Args:
            config: Replay configuration
            agent_fn: The agent function to run. Receives a TraceContext and
                      should call ctx.llm_call() / ctx.tool_call() as normal.
                      LLM calls will return recorded responses.
        """
        recorded = self.load_run(config.run_id)
        if not recorded:
            raise ValueError(f"Run not found: {config.run_id}")

        recorded_llm_calls = recorded.get("llm_calls", [])
        recorded_tool_calls = recorded.get("tool_calls", [])

        # Create a replay run
        replay_name = f"replay-of-{config.run_id[:8]}"
        if config.model_override:
            replay_name += f"-model-{config.model_override}"

        start_time = time.time()

        with TraceContext(
            name=replay_name,
            agent_name=recorded.get("agent_name", ""),
            store=self.store,
        ) as ctx:
            # Monkey-patch ctx.llm_call to replay recorded responses
            llm_index = [0]  # mutable counter

            original_llm_call = ctx.llm_call

            def replay_llm_call(
                model: str,
                prompt: str,
                response: str = "",
                **kwargs,
            ) -> LLMCallRecord:
                idx = llm_index[0]
                if idx < len(recorded_llm_calls):
                    rec = recorded_llm_calls[idx]
                    # Apply overrides
                    effective_model = config.model_override or rec["model"]
                    effective_prompt = prompt
                    if config.prompt_template:
                        effective_prompt = config.prompt_template.replace(
                            "{prompt}", prompt
                        )

                    result = original_llm_call(
                        model=effective_model,
                        prompt=effective_prompt,
                        response=rec["response"],
                        tokens_in=rec["tokens_in"],
                        tokens_out=rec["tokens_out"],
                        cost=rec["cost"],
                        latency_ms=0,  # replay is instant
                        provider=rec.get("provider", "anthropic"),
                    )
                    llm_index[0] += 1
                    return result
                else:
                    # More LLM calls than recorded — make a live call?
                    raise RuntimeError(
                        f"Replay exhausted: {idx} calls recorded but agent made more. "
                        "The agent behavior has changed."
                    )

            ctx.llm_call = replay_llm_call  # type: ignore[method-assign]

            # Run the agent
            agent_fn(ctx)

        duration_ms = (time.time() - start_time) * 1000

        return ReplayResult(
            original_run_id=config.run_id,
            replay_run_id=ctx.run.id,
            steps_replayed=llm_index[0],
            llm_calls_replayed=llm_index[0],
            tool_calls_replayed=len(ctx._tool_calls),
            total_cost=ctx.run.total_cost,
            duration_ms=duration_ms,
        )

    def diff_replays(
        self,
        result_a: ReplayResult,
        result_b: ReplayResult,
    ) -> dict:
        """Compare two replay results and produce a diff."""
        run_a = self.store.get_run(result_a.replay_run_id)
        run_b = self.store.get_run(result_b.replay_run_id)

        if not run_a or not run_b:
            return {"error": "One or both replay runs not found"}

        return {
            "run_a": {
                "id": result_a.replay_run_id,
                "steps": result_a.steps_replayed,
                "cost": result_a.total_cost,
                "duration_ms": result_a.duration_ms,
            },
            "run_b": {
                "id": result_b.replay_run_id,
                "steps": result_b.steps_replayed,
                "cost": result_b.total_cost,
                "duration_ms": result_b.duration_ms,
            },
            "delta": {
                "steps": result_b.steps_replayed - result_a.steps_replayed,
                "cost": result_b.total_cost - result_a.total_cost,
                "duration_ms": result_b.duration_ms - result_a.duration_ms,
            },
            "verdict": (
                "identical"
                if result_a.steps_replayed == result_b.steps_replayed
                and abs(result_a.total_cost - result_b.total_cost) < 0.0001
                else "different"
            ),
        }
