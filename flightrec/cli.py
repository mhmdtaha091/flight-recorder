"""
Flight Recorder CLI — `flightrec` command.
"""

from __future__ import annotations

import json
import sys
import time
from typing import Optional

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.tree import Tree
from rich import box

from flightrec.store import TraceStore

console = Console()


@click.group()
@click.option("--db", default="flightrec.db", help="Path to SQLite database", envvar="FLIGHTREC_DB")
@click.pass_context
def main(ctx, db):
    """Flight Recorder — trace, replay, and evaluate AI agents."""
    ctx.ensure_object(dict)
    ctx.obj["store"] = TraceStore(db)


@main.command()
@click.argument("run_id")
@click.pass_context
def show(ctx, run_id):
    """Show a detailed timeline of a single run."""
    store: TraceStore = ctx.obj["store"]
    run = store.get_run(run_id)

    if not run:
        console.print(f"[red]Run not found: {run_id}[/red]")
        sys.exit(1)

    # Header
    status_color = "green" if run["status"] == "completed" else "red" if run["status"] == "failed" else "yellow"
    console.print(Panel(
        f"[bold]{run['name']}[/bold]\n"
        f"Agent: {run['agent_name'] or 'unknown'}  •  "
        f"Status: [{status_color}]{run['status']}[/{status_color}]  •  "
        f"Steps: {run['step_count']}",
        title=f"Run {run['id']}",
        border_style="cyan",
    ))

    # Stats row
    stats = Table(show_header=False, box=box.SIMPLE)
    stats.add_column()
    stats.add_column()
    stats.add_row("Tokens", f"{run['total_tokens_in']:,} in / {run['total_tokens_out']:,} out")
    stats.add_row("Cost", f"${run['total_cost']:.4f}")
    stats.add_row("Latency", f"{run['total_latency_ms']:.0f}ms")
    started = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(run["started_at"]))
    stats.add_row("Started", started)
    if run["ended_at"]:
        ended = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(run["ended_at"]))
        stats.add_row("Ended", ended)
    console.print(stats)
    console.print()

    # Timeline
    tree = Tree(f"[bold]Timeline[/bold] ({run['step_count']} steps)")
    all_steps: list[dict] = []
    for call in run.get("llm_calls", []):
        all_steps.append({"type": "llm", "step": call["step_index"], "data": call})
    for call in run.get("tool_calls", []):
        all_steps.append({"type": "tool", "step": call["step_index"], "data": call})
    all_steps.sort(key=lambda s: s["step"])

    for step in all_steps:
        if step["type"] == "llm":
            d = step["data"]
            node = tree.add(
                f"[bold cyan]Step {d['step_index']}[/bold cyan] "
                f"[yellow]🤖 LLM[/yellow] → {d['model']}  "
                f"({d['tokens_in']:,}↑ {d['tokens_out']:,}↓)  "
                f"[green]${d['cost']:.4f}[/green]  "
                f"{d['latency_ms']:.0f}ms"
            )
            if d["prompt"]:
                prompt_preview = d["prompt"][:200].replace("\n", " ")
                node.add(f"Prompt: {prompt_preview}...")
            if d["response"]:
                resp_preview = d["response"][:200].replace("\n", " ")
                node.add(f"Response: {resp_preview}...")
        else:
            d = step["data"]
            icon = "✅" if d["success"] else "❌"
            node = tree.add(
                f"[bold cyan]Step {d['step_index']}[/bold cyan] "
                f"[blue]{icon} 🔧 {d['name']}[/blue]  "
                f"{d['latency_ms']:.0f}ms"
            )
            if d["error"]:
                node.add(f"[red]Error: {d['error']}[/red]")

    console.print(tree)

    # Eval results
    evals = store.get_eval_results(run_id)
    if evals:
        console.print()
        console.print("[bold]📊 Evaluations[/bold]")
        eval_table = Table(box=box.SIMPLE)
        eval_table.add_column("Task")
        eval_table.add_column("Passed")
        eval_table.add_column("Score")
        for ev in evals:
            status = "[green]✓[/green]" if ev["passed"] else "[red]✗[/red]"
            eval_table.add_row(ev["task_id"], status, f"{ev['score']:.2f}")
        console.print(eval_table)


@main.command()
@click.option("--limit", default=20, help="Number of runs to show")
@click.option("--agent", default="", help="Filter by agent name")
@click.option("--status", default="", help="Filter by status (completed/failed/running)")
@click.pass_context
def list(ctx, limit, agent, status):
    """List recent runs."""
    store: TraceStore = ctx.obj["store"]
    runs = store.list_runs(limit=limit, agent_name=agent, status=status)

    if not runs:
        console.print("[dim]No runs found.[/dim]")
        return

    table = Table(box=box.SIMPLE)
    table.add_column("Run ID", style="dim")
    table.add_column("Name")
    table.add_column("Agent")
    table.add_column("Status")
    table.add_column("Steps")
    table.add_column("Cost")
    table.add_column("When")

    for r in runs:
        status_color = "green" if r["status"] == "completed" else "red" if r["status"] == "failed" else "yellow"
        when = time.strftime("%m/%d %H:%M", time.localtime(r["started_at"]))
        table.add_row(
            r["id"][:8],
            r["name"][:40],
            r["agent_name"] or "-",
            f"[{status_color}]{r['status']}[/{status_color}]",
            str(r["step_count"]),
            f"${r['total_cost']:.4f}",
            when,
        )

    console.print(table)


@main.command()
@click.pass_context
def stats(ctx):
    """Show aggregate statistics."""
    store: TraceStore = ctx.obj["store"]
    s = store.get_stats()

    console.print(Panel("[bold]Flight Recorder Stats[/bold]", border_style="cyan"))
    table = Table(box=box.SIMPLE)
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("Total runs", str(s["total_runs"]))
    table.add_row("Completed", f"[green]{s['completed_runs']}[/green]")
    table.add_row("Failed", f"[red]{s['failed_runs']}[/red]" if s["failed_runs"] > 0 else f"[dim]{s['failed_runs']}[/dim]")
    table.add_row("Total tokens", f"{s['total_tokens']:,}")
    table.add_row("Total cost", f"${s['total_cost']:.4f}")
    table.add_row("Avg latency", f"{s['avg_latency_ms']:.0f}ms")
    console.print(table)


@main.command()
@click.pass_context
def serve(ctx):
    """Start the web dashboard (FastAPI + React)."""
    import uvicorn
    from flightrec.server import create_app

    store: TraceStore = ctx.obj["store"]
    app = create_app(store)
    console.print("[cyan]🚀 Flight Recorder dashboard → http://localhost:8421[/cyan]")
    uvicorn.run(app, host="127.0.0.1", port=8421, log_level="info")


@main.command()
@click.argument("run_id")
@click.pass_context
def delete(ctx, run_id):
    """Delete a run and all its data."""
    store: TraceStore = ctx.obj["store"]
    if store.delete_run(run_id):
        console.print(f"[green]Deleted run {run_id}[/green]")
    else:
        console.print(f"[red]Run not found: {run_id}[/red]")


@main.command()
@click.argument("run_id")
@click.option("--output", "-o", default="-", help="Output file (default: stdout)")
@click.pass_context
def export(ctx, run_id, output):
    """Export a run as JSON."""
    store: TraceStore = ctx.obj["store"]
    run = store.get_run(run_id)
    if not run:
        console.print(f"[red]Run not found: {run_id}[/red]")
        sys.exit(1)
    data = json.dumps(run, indent=2, default=str)
    if output == "-":
        console.print(data)
    else:
        with open(output, "w") as f:
            f.write(data)
        console.print(f"[green]Exported to {output}[/green]")


if __name__ == "__main__":
    main()
