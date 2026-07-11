"""
Regression diffing — compare run-sets across agent versions.
v2 per AGENT_FLIGHT_RECORDER_PLAN.md.

Answers: "Did upgrading the model help?" with a table, not a feeling.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from flightrec.store import TraceStore
from flightrec.scoring import TaskScorer


@dataclass
class RegressionReport:
    """A full regression comparison between two agent versions."""

    version_a: str
    version_b: str
    suite_name: str
    pass_rate_a: float
    pass_rate_b: float
    pass_rate_delta: float
    avg_score_a: float
    avg_score_b: float
    avg_score_delta: float
    cost_a: float
    cost_b: float
    cost_delta: float
    latency_a: float
    latency_b: float
    latency_delta: float
    regressions: list[dict] = field(default_factory=list)
    improvements: list[dict] = field(default_factory=list)
    verdict: str = ""


class RegressionRunner:
    """Compare two agent versions across a golden task suite."""

    def __init__(self, store: TraceStore, api_key: str = ""):
        self.store = store
        self.scorer = TaskScorer(store, api_key=api_key)

    def compare(
        self,
        suite_name: str,
        agent_a: Callable[[dict], Any],
        agent_b: Callable[[dict], Any],
        version_a: str = "v1",
        version_b: str = "v2",
        use_judge: bool = False,
    ) -> RegressionReport:
        """
        Run the same suite against two agent versions and diff the results.

        Args:
            suite_name: Golden task suite name
            agent_a: Baseline agent function
            agent_b: New agent function
            version_a: Label for baseline
            version_b: Label for new version
            use_judge: Whether to use LLM judge

        Returns:
            RegressionReport with full comparison
        """
        # Attribute cost/latency to each version by diffing the run set before and
        # after each suite runs — otherwise sampling "recent runs" after both suites
        # have run mixes B's runs into A's totals.
        def _run_ids(rows):
            return {r["id"] for r in rows}

        before = _run_ids(self.store.list_runs(limit=100_000))

        result_a = self.scorer.run_suite(suite_name, agent_a, use_judge=use_judge)
        after_a = self.store.list_runs(limit=100_000)
        a_ids = _run_ids(after_a) - before
        cost_a = sum(r["total_cost"] for r in after_a if r["id"] in a_ids)
        latency_a = sum(r["total_latency_ms"] for r in after_a if r["id"] in a_ids)

        result_b = self.scorer.run_suite(suite_name, agent_b, use_judge=use_judge)
        after_b = self.store.list_runs(limit=100_000)
        b_ids = _run_ids(after_b) - before - a_ids
        cost_b = sum(r["total_cost"] for r in after_b if r["id"] in b_ids)
        latency_b = sum(r["total_latency_ms"] for r in after_b if r["id"] in b_ids)

        # Identify regressions and improvements
        regressions: list[dict] = []
        improvements: list[dict] = []

        results_a_map = {
            r["task_name"]: r for r in result_a.get("results", [])
        }
        results_b_map = {
            r["task_name"]: r for r in result_b.get("results", [])
        }

        for task_name, ra in results_a_map.items():
            rb = results_b_map.get(task_name)
            if not rb:
                continue

            if ra["passed"] and not rb["passed"]:
                regressions.append({
                    "task_name": task_name,
                    "score_before": ra["score"],
                    "score_after": rb["score"],
                    "delta": rb["score"] - ra["score"],
                })
            elif not ra["passed"] and rb["passed"]:
                improvements.append({
                    "task_name": task_name,
                    "score_before": ra["score"],
                    "score_after": rb["score"],
                    "delta": rb["score"] - ra["score"],
                })

        pass_rate_delta = result_b["pass_rate"] - result_a["pass_rate"]
        avg_score_delta = result_b["total_score"] - result_a["total_score"]

        # Verdict
        if len(regressions) == 0 and len(improvements) > 0:
            verdict = f"✅ {version_b} is strictly better (+{len(improvements)} improvements, 0 regressions)"
        elif len(regressions) > 0 and len(improvements) > len(regressions):
            verdict = f"⚠️ {version_b} improves {len(improvements)} tasks but regresses on {len(regressions)} — review regressions"
        elif len(regressions) > 0 and len(improvements) <= len(regressions):
            verdict = f"❌ {version_b} is worse ({len(regressions)} regressions)"
        elif pass_rate_delta > 0:
            verdict = f"📈 {version_b} improves pass rate by {pass_rate_delta:.1%}"
        elif pass_rate_delta < 0:
            verdict = f"📉 {version_b} decreases pass rate by {abs(pass_rate_delta):.1%}"
        else:
            verdict = "➡️ No significant change"

        return RegressionReport(
            version_a=version_a,
            version_b=version_b,
            suite_name=suite_name,
            pass_rate_a=result_a["pass_rate"],
            pass_rate_b=result_b["pass_rate"],
            pass_rate_delta=pass_rate_delta,
            avg_score_a=result_a["total_score"],
            avg_score_b=result_b["total_score"],
            avg_score_delta=avg_score_delta,
            cost_a=cost_a,
            cost_b=cost_b,
            cost_delta=cost_b - cost_a,
            latency_a=latency_a,
            latency_b=latency_b,
            latency_delta=latency_b - latency_a,
            regressions=regressions,
            improvements=improvements,
            verdict=verdict,
        )

    def format_report(self, report: RegressionReport) -> str:
        """Format a regression report as a markdown table."""
        lines = [
            f"# 🔍 Regression Report: {report.version_a} → {report.version_b}",
            "",
            f"**Suite:** {report.suite_name}  •  **Verdict:** {report.verdict}",
            "",
            "## 📊 Summary",
            "",
            "| Metric | {report.version_a} | {report.version_b} | Δ |".replace(
                "{report.version_a}", report.version_a
            ).replace("{report.version_b}", report.version_b),
            "| ------ | ----- | ----- | -- |",
            f"| Pass Rate | {report.pass_rate_a:.1%} | {report.pass_rate_b:.1%} | {report.pass_rate_delta:+.1%} |",
            f"| Avg Score | {report.avg_score_a:.2f} | {report.avg_score_b:.2f} | {report.avg_score_delta:+.2f} |",
            f"| Cost | ${report.cost_a:.4f} | ${report.cost_b:.4f} | ${report.cost_delta:+.4f} |",
            f"| Latency | {report.latency_a:.0f}ms | {report.latency_b:.0f}ms | {report.latency_delta:+.0f}ms |",
            "",
        ]

        if report.regressions:
            lines.append("## ❌ Regressions")
            lines.append("")
            lines.append("| Task | Score Before | Score After | Δ |")
            lines.append("| ---- | ------------ | ----------- | - |")
            for r in report.regressions:
                lines.append(
                    f"| {r['task_name']} | {r['score_before']:.2f} | {r['score_after']:.2f} | {r['delta']:.2f} |"
                )
            lines.append("")

        if report.improvements:
            lines.append("## ✅ Improvements")
            lines.append("")
            lines.append("| Task | Score Before | Score After | Δ |")
            lines.append("| ---- | ------------ | ----------- | - |")
            for imp in report.improvements:
                lines.append(
                    f"| {imp['task_name']} | {imp['score_before']:.2f} | {imp['score_after']:.2f} | +{imp['delta']:.2f} |"
                )
            lines.append("")

        return "\n".join(lines)
