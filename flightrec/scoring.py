"""
Golden task scoring — YAML task suites, rule checks, LLM-judge.
v1 per AGENT_FLIGHT_RECORDER_PLAN.md.

Usage:
    scorer = TaskScorer(store)
    results = scorer.run_suite("jarvis-daily", agent_fn)
    # results: {task_name: {passed: bool, score: float, details: {...}}}
"""

from __future__ import annotations

import copy
import json
import os
import re
import time
import uuid
from typing import Any, Callable, Optional

import yaml

from flightrec.store import TraceStore
from flightrec.sdk import TraceContext


class RuleCheck:
    """A deterministic check that evaluates a run's output."""

    def __init__(self, check_def: dict):
        self.check_def = check_def
        self.check_type = check_def.get("type", "contains")

    def evaluate(self, run_data: dict) -> tuple[bool, str]:
        """
        Evaluate this check against a run's data.
        Returns (passed, reason).
        """
        check_type = self.check_type

        if check_type == "contains":
            return self._check_contains(run_data)
        elif check_type == "matches":
            return self._check_matches(run_data)
        elif check_type == "tool_called":
            return self._check_tool_called(run_data)
        elif check_type == "cost_below":
            return self._check_cost_below(run_data)
        elif check_type == "latency_below":
            return self._check_latency_below(run_data)
        elif check_type == "no_errors":
            return self._check_no_errors(run_data)
        elif check_type == "output_contains":
            return self._check_output_contains(run_data)
        else:
            return False, f"Unknown check type: {check_type}"

    def _check_contains(self, run_data: dict) -> tuple[bool, str]:
        target = self.check_def.get("value", "")
        # Search all LLM responses
        for call in run_data.get("llm_calls", []):
            if target.lower() in call.get("response", "").lower():
                return True, f"Found '{target}' in LLM response"
        return False, f"'{target}' not found in any LLM response"

    def _check_matches(self, run_data: dict) -> tuple[bool, str]:
        pattern = self.check_def.get("pattern", "")
        try:
            regex = re.compile(pattern, re.IGNORECASE | re.DOTALL)
        except re.error as e:
            return False, f"Invalid regex: {e}"
        for call in run_data.get("llm_calls", []):
            if regex.search(call.get("response", "")):
                return True, f"Pattern '{pattern}' matched"
        return False, f"Pattern '{pattern}' not matched"

    def _check_tool_called(self, run_data: dict) -> tuple[bool, str]:
        tool_name = self.check_def.get("tool", "").lower()
        min_calls = self.check_def.get("min_calls", 1)
        count = sum(
            1
            for c in run_data.get("tool_calls", [])
            if c.get("name", "").lower() == tool_name
        )
        if count >= min_calls:
            return True, f"Tool '{tool_name}' called {count} time(s)"
        return False, f"Tool '{tool_name}' called {count}/{min_calls} time(s)"

    def _check_cost_below(self, run_data: dict) -> tuple[bool, str]:
        max_cost = self.check_def.get("max_cost", 1.0)
        actual = run_data.get("total_cost", 0)
        if actual <= max_cost:
            return True, f"Cost ${actual:.4f} ≤ ${max_cost:.4f}"
        return False, f"Cost ${actual:.4f} > ${max_cost:.4f}"

    def _check_latency_below(self, run_data: dict) -> tuple[bool, str]:
        max_ms = self.check_def.get("max_ms", 10_000)
        actual = run_data.get("total_latency_ms", 0)
        if actual <= max_ms:
            return True, f"Latency {actual:.0f}ms ≤ {max_ms}ms"
        return False, f"Latency {actual:.0f}ms > {max_ms}ms"

    def _check_no_errors(self, run_data: dict) -> tuple[bool, str]:
        for call in run_data.get("tool_calls", []):
            if not call.get("success", True):
                return False, f"Tool '{call.get('name')}' failed: {call.get('error', 'unknown')}"
        if run_data.get("status") == "failed":
            return False, "Run status is 'failed'"
        return True, "No errors detected"

    def _check_output_contains(self, run_data: dict) -> tuple[bool, str]:
        target = self.check_def.get("value", "")
        # Search tool call outputs
        for call in run_data.get("tool_calls", []):
            output = call.get("output_json", "{}")
            if target.lower() in output.lower():
                return True, f"Found '{target}' in tool output"
        return False, f"'{target}' not found in any tool output"


class LLMJudge:
    """LLM-as-judge for fuzzy outcome scoring.

    Provider-agnostic: defaults to openai-compatible (DeepSeek) for cost.
    Set JUDGE_PROVIDER=anthropic to use Claude, or JUDGE_MODEL / JUDGE_BASE_URL
    for any OpenAI-compatible endpoint (Ollama, Groq, Gemini, etc.).
    """

    def __init__(
        self,
        api_key: str = "",
        model: str = "",
        base_url: str = "",
        provider: str = "",
    ):
        self.provider = provider or os.environ.get("JUDGE_PROVIDER", "openai-compatible")
        self.api_key = api_key or os.environ.get(
            "ANTHROPIC_API_KEY" if self.provider == "anthropic" else "OPENAI_API_KEY",
            "",
        )
        # Default model per provider
        default_models = {
            "anthropic": "claude-haiku-4-5-20251001",
            "openai": "gpt-4o-mini",
            "openai-compatible": "deepseek-chat",
            "gemini": "gemini-2.0-flash",
        }
        self.model = model or os.environ.get("JUDGE_MODEL") or default_models.get(self.provider, "deepseek-chat")
        # Default base URL per provider
        default_urls = {
            "anthropic": "https://api.anthropic.com/v1",
            "openai": "https://api.openai.com/v1",
            "openai-compatible": "https://api.deepseek.com/v1",
            "gemini": "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
        }
        self.base_url = base_url or os.environ.get("JUDGE_BASE_URL") or default_urls.get(
            self.provider, "https://api.deepseek.com/v1"
        )

    def evaluate(
        self, judge_prompt: str, run_data: dict, expected: str
    ) -> tuple[float, str]:
        """
        Ask the LLM to judge the run output against expectations.
        Returns (score 0-1, reasoning).
        """
        # Build context from run data
        responses = "\n".join(
            c.get("response", "")[:1000]
            for c in run_data.get("llm_calls", [])
        )
        tool_outputs = "\n".join(
            f"{c.get('name')}: {c.get('output_json', '')[:500]}"
            for c in run_data.get("tool_calls", [])
        )

        prompt = f"""{judge_prompt}

Expected outcome: {expected}

Agent's LLM responses:
{responses[:3000]}

Agent's tool outputs:
{tool_outputs[:2000]}

Score the agent's performance on a scale of 0.0 to 1.0, where 1.0 is perfect.
Output a JSON object with:
- "score": float (0.0-1.0)
- "reasoning": string (one sentence explaining the score)
- "passed": boolean (score >= 0.7)

Respond with ONLY the JSON object."""

        try:
            import httpx

            if self.provider == "anthropic":
                text = self._call_anthropic(prompt)
            else:
                text = self._call_openai_compatible(prompt)

            # Extract JSON
            match = re.search(r"\{[\s\S]*\}", text)
            if match:
                result = json.loads(match.group(0))
                return (result.get("score", 0.0), result.get("reasoning", "No reasoning"))

            return (0.5, f"Could not parse judge response: {text[:100]}")

        except Exception as e:
            return (0.0, f"Judge evaluation failed: {e}")

    def _call_anthropic(self, prompt: str) -> str:
        import httpx

        resp = httpx.post(
            f"{self.base_url}/messages",
            headers={
                "Content-Type": "application/json",
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
            },
            json={
                "model": self.model,
                "max_tokens": 512,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=30,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"Judge API error ({self.provider}): {resp.status_code}")
        data = resp.json()
        return data["content"][0]["text"]

    def _call_openai_compatible(self, prompt: str) -> str:
        import httpx

        resp = httpx.post(
            f"{self.base_url}/chat/completions",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            json={
                "model": self.model,
                "max_tokens": 512,
                "temperature": 0.1,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=30,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"Judge API error ({self.provider}): {resp.status_code} - {resp.text[:200]}")
        data = resp.json()
        return data["choices"][0]["message"]["content"]


class TaskScorer:
    """Score agent runs against golden task suites."""

    def __init__(
        self,
        store: TraceStore,
        api_key: str = "",
        judge_model: str = "claude-sonnet-5",
    ):
        self.store = store
        self.judge = LLMJudge(api_key=api_key, model=judge_model)

    def run_suite(
        self,
        suite_name: str,
        agent_fn: Callable[[dict], Any],
        use_judge: bool = True,
    ) -> dict:
        """
        Run all golden tasks in a suite against the agent.

        Args:
            suite_name: Name of the golden task suite
            agent_fn: Function that takes task input and runs the agent.
                      Should return a run_id (str) or run data dict.
            use_judge: Whether to use LLM judge for fuzzy checks

        Returns:
            {suite_name, results: [{task_id, task_name, passed, score, details}],
             pass_rate, total_score}
        """
        tasks = self.store.get_golden_tasks(suite_name)
        if not tasks:
            return {
                "suite_name": suite_name,
                "results": [],
                "pass_rate": 0.0,
                "total_score": 0.0,
                "error": f"No tasks found for suite '{suite_name}'",
            }

        results = []
        for task in tasks:
            checks = json.loads(
                task["checks_json"]
                if isinstance(task["checks_json"], str)
                else task["checks_json"]
            )
            judge_prompt = task.get("judge_prompt", "")

            # Run the agent for this task
            try:
                run_result = agent_fn(json.loads(task["input_json"]))
            except Exception as e:
                results.append({
                    "task_id": task["id"],
                    "task_name": task["task_name"],
                    "passed": False,
                    "score": 0.0,
                    "details": {"error": str(e)},
                })
                continue

            # Get the run data
            if isinstance(run_result, str):
                run_data = self.store.get_run(run_result)
            else:
                run_data = run_result

            if not run_data:
                results.append({
                    "task_id": task["id"],
                    "task_name": task["task_name"],
                    "passed": False,
                    "score": 0.0,
                    "details": {"error": "Run data not found"},
                })
                continue

            # Evaluate rule checks
            all_checks_passed = True
            check_details = []
            for check_def in checks:
                rule = RuleCheck(check_def)
                passed, reason = rule.evaluate(run_data)
                check_details.append({
                    "type": check_def.get("type", ""),
                    "passed": passed,
                    "reason": reason,
                })
                if not passed:
                    all_checks_passed = False

            # LLM judge for fuzzy checks
            judge_score = 0.0
            judge_reasoning = ""
            if use_judge and judge_prompt:
                judge_score, judge_reasoning = self.judge.evaluate(
                    judge_prompt,
                    run_data,
                    task.get("expected_output", ""),
                )

            # Final score: rule checks (weight 0.7) + judge (weight 0.3)
            rule_score = 1.0 if all_checks_passed else 0.0
            final_score = rule_score * 0.7 + judge_score * 0.3
            passed = final_score >= 0.7

            result_entry = {
                "task_id": task["id"],
                "task_name": task["task_name"],
                "passed": passed,
                "score": final_score,
                "details": {
                    "checks": check_details,
                    "judge_score": judge_score,
                    "judge_reasoning": judge_reasoning,
                },
            }
            results.append(result_entry)

            # Persist eval result
            self.store.save_eval_result({
                "id": f"eval-{uuid.uuid4().hex[:12]}",
                "run_id": run_data.get("id", ""),
                "task_id": task["id"],
                "passed": passed,
                "score": final_score,
                "details": result_entry["details"],
            })

        passed_count = sum(1 for r in results if r["passed"])
        total_score = sum(r["score"] for r in results)

        return {
            "suite_name": suite_name,
            "results": results,
            "pass_rate": passed_count / len(results) if results else 0.0,
            "total_score": total_score / len(results) if results else 0.0,
            "passed": passed_count,
            "total": len(results),
        }

    def load_suite_from_yaml(self, yaml_path: str) -> int:
        """
        Load a golden task suite from a YAML file into the store.
        Returns the number of tasks loaded.

        YAML format:
            suite: jarvis-daily
            tasks:
              - name: "Summarize inbox"
                description: "Agent should summarize unread emails"
                input:
                  action: summarize_inbox
                checks:
                  - type: tool_called
                    tool: search_email
                    min_calls: 1
                  - type: output_contains
                    value: "unread"
                judge_prompt: "Did the agent produce a useful summary?"
        """
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)

        suite_name = data.get("suite", "default")
        count = 0
        for task in data.get("tasks", []):
            task_def = {
                "suite_name": suite_name,
                "task_name": task.get("name", ""),
                "description": task.get("description", ""),
                "input": task.get("input", {}),
                "expected_output": task.get("expected_output", ""),
                "checks": task.get("checks", []),
                "judge_prompt": task.get("judge_prompt", ""),
                "enabled": task.get("enabled", True),
            }
            self.store.save_golden_task(task_def)
            count += 1

        return count


# ── Flake detection ──

def detect_flakes(
    scorer: TaskScorer,
    suite_name: str,
    agent_fn: Callable[[dict], Any],
    runs: int = 3,
) -> dict:
    """
    Run a suite multiple times to detect flaky tests.
    A task is flaky if it passes in some runs and fails in others.
    """
    all_results = []
    for i in range(runs):
        result = scorer.run_suite(suite_name, agent_fn, use_judge=False)
        all_results.append(result)

    task_names = [r["task_name"] for r in all_results[0]["results"]] if all_results else []

    flakes = []
    for task_name in task_names:
        outcomes = []
        for run_result in all_results:
            for tr in run_result["results"]:
                if tr["task_name"] == task_name:
                    outcomes.append(tr["passed"])
        if not all(o == outcomes[0] for o in outcomes):
            flakes.append({
                "task_name": task_name,
                "outcomes": outcomes,
                "flake_rate": sum(1 for o in outcomes if o != outcomes[0]) / len(outcomes),
            })

    return {
        "runs": runs,
        "flaky_tasks": flakes,
        "flake_count": len(flakes),
        "suite_results": all_results,
    }
