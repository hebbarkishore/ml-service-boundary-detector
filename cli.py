#!/usr/bin/env python3
import json
import logging
import os
import sys
from pathlib import Path

import click

sys.path.insert(0, os.path.dirname(__file__))
from config import OUTPUT_DIR, API_HOST, API_PORT
from core.pipeline import PipelineConfig, ServiceBoundaryPipeline
from feedback.feedback_store import FeedbackStore

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(str(OUTPUT_DIR / "sbd.log"), mode="a"),
    ],
)
log = logging.getLogger("sbd.cli")

@click.group()
def cli():
    """Service Boundary Detector - ML-assisted legacy system modularization."""


@cli.command()
@click.option("--code",    "-c", multiple=True, required=True,
              help="Source root(s) - directories or individual files.")
@click.option("--traces",  "-t", multiple=True, default=[],
              help="Runtime trace / log files.")
@click.option("--docs",    "-d", multiple=True, default=[],
              help="Document files (PDF/DOCX/MD) to ingest.")
@click.option("--repo",    "-r", default=None,
              help="Git repo root for evolutionary signals.")
@click.option("--labels",  "-l", default=None,
              help="JSON labels file (overrides/supplements feedback store).")
@click.option("--top-k",   default=30, show_default=True)
@click.option("--no-behavioral",   is_flag=True, default=False)
@click.option("--no-evolutionary", is_flag=True, default=False)
@click.option("--no-documents",    is_flag=True, default=False)
@click.option("--no-feedback",     is_flag=True, default=False,
              help="Skip loading stored architect feedback (not recommended).")
def analyze(code, traces, docs, repo, labels, top_k,
            no_behavioral, no_evolutionary, no_documents, no_feedback):
    """Scan a codebase and identify service boundary candidates."""

    click.echo(click.style("\n═══ Service Boundary Detector ═══\n", fg="cyan", bold=True))

    store = FeedbackStore()
    stats = store.stats()
    if stats["total"] > 0:
        click.echo(click.style(
            f"  ↺ Loading {stats['total']} stored architect decisions "
            f"(accepted={stats['accepted']}, rejected={stats['rejected']})\n",
            fg="yellow"
        ))

    cfg = PipelineConfig(
        code_roots        = list(code),
        trace_files       = list(traces),
        doc_files         = list(docs),
        git_repo_path     = repo,
        labels_file       = labels,
        top_k             = top_k,
        skip_behavioral   = no_behavioral or not traces,
        skip_evolutionary = no_evolutionary,
        skip_documents    = no_documents,
        skip_feedback     = no_feedback,
    )

    result = ServiceBoundaryPipeline(cfg).run()

    mode = result.meta.get("stages", {}).get("ranking", {}).get("mode", "?")
    click.echo(click.style("── Summary ──────────────────────────────────────", fg="green"))
    click.echo(f"  Units scanned  : {len(result.units)}")
    click.echo(f"  Pipeline time  : {result.meta.get('total_elapsed_s', '?')}s")
    click.echo(f"  Model mode     : {mode}")
    click.echo(f"  Top boundaries : {len(result.top_candidates)}")

    if result.cluster_map:
        real = {k: v for k, v in result.cluster_map.items() if k != -1}
        click.echo(f"  Service clusters: {len(real)} suggested")

    click.echo(click.style("\n── Top Boundary Candidates ──────────────────────", fg="yellow"))
    for i, c in enumerate(result.top_candidates[:15], 1):
        bar   = "x" * int(c.boundary_score * 20)
        color = "red" if c.boundary_score >= 0.65 else "yellow" if c.boundary_score >= 0.35 else "green"
        click.echo(
            f"  {i:>2}. "
            + click.style(f"[{bar:<20}]", fg=color)
            + f" {c.boundary_score:.3f}  {_short(c.comp_a)} ↔ {_short(c.comp_b)}"
        )
        if c.suggested_service:
            click.echo(f"       → {c.suggested_service}")

    if result.feature_importances:
        click.echo(click.style("\n── Feature Importances (top 8) ──────────────────", fg="cyan"))
        for feat, imp in list(result.feature_importances.items())[:8]:
            bar = "*" * int(imp * 50)
            click.echo(f"  {feat:<38} {imp:.4f}  {bar}")

    if result.cluster_map:
        real = {k: v for k, v in result.cluster_map.items() if k != -1}
        if real:
            click.echo(click.style("\n── Suggested Service Groupings ──────────────────", fg="blue"))
            for cid, members in sorted(real.items()):
                short_members = ", ".join(_short(m) for m in members[:5])
                extra = f" … (+{len(members)-5} more)" if len(members) > 5 else ""
                click.echo(f"  Cluster {cid}: {short_members}{extra}")

    report_path = OUTPUT_DIR / "boundary_report.json"
    click.echo(click.style(f"\n Full report → {report_path}", fg="green", bold=True))
    click.echo(click.style(
        f" Review UI  → python cli.py serve  then open http://localhost:{API_PORT}/\n",
        fg="green"
    ))


@cli.command()
@click.option("--host",  default=API_HOST, show_default=True)
@click.option("--port",  default=API_PORT, show_default=True, type=int)
@click.option("--debug", is_flag=True, default=False)
def serve(host, port, debug):
    """Start the Flask API server + Architectural Review UI."""
    click.echo(click.style(f"\nStarting Service Boundary Detector", fg="cyan", bold=True))
    click.echo(f"  Review UI → http://{host}:{port}/")
    click.echo(f"  Health    → http://{host}:{port}/api/v1/health")
    click.echo(f"  Report    → http://{host}:{port}/api/v1/report\n")
    from api.app import app
    app.run(host=host, port=port, debug=debug)


@cli.command("gen-labels")
@click.option("--top-k", default=50, show_default=True)
def gen_labels(top_k):
    """
    Generate a label template JSON from the last report.
    Edit 'label' fields (1=boundary, 0=keep together) then use:
        python cli.py feedback import --file output/labels_template.json
    """
    report_path = OUTPUT_DIR / "boundary_report.json"
    if not report_path.exists():
        click.echo("No report found. Run 'analyze' first.", err=True)
        sys.exit(1)

    with open(report_path) as f:
        report = json.load(f)

    store = FeedbackStore()
    fb    = store.get_all_labels()

    boundaries = report.get("top_boundaries", [])[:top_k]
    template   = []
    for b in boundaries:
        a, c   = b["component_a"], b["component_b"]
        key    = (min(a, c), max(a, c))
        existing_label = fb.get(key)
        template.append({
            "comp_a":          a,
            "comp_b":          c,
            "boundary_score":  b["boundary_score"],
            "label":           existing_label,   
            "_instruction":    "Set label to 1 (valid boundary) or 0 (keep together). Remove _instruction before import.",
        })

    out = OUTPUT_DIR / "labels_template.json"
    with open(out, "w") as f:
        json.dump(template, f, indent=2)

    already_labelled = sum(1 for t in template if t["label"] is not None)
    click.echo(click.style(f"✓ Template saved → {out}", fg="green", bold=True))
    click.echo(f"  {len(template)} pairs total  |  {already_labelled} already labelled  |  {len(template)-already_labelled} need review")
    click.echo(f"\n  After editing, import with:")
    click.echo(f"    python cli.py feedback import --file {out}")


@cli.group()
def feedback():
    """Manage the persistent architect feedback store."""


@feedback.command("stats")
def feedback_stats():
    """Show feedback store statistics."""
    store = FeedbackStore()
    s     = store.stats()
    click.echo(click.style("\n── Feedback Store Stats ──────────────────────────", fg="cyan"))
    click.echo(f"  Total decisions : {s['total']}")
    click.echo(f"  Accepted        : {s['accepted']} (valid boundaries)")
    click.echo(f"  Rejected        : {s['rejected']} (keep together)")
    click.echo(f"  Last updated    : {s['last_updated']}")
    click.echo(f"  Training-ready  : {'YES ✓' if store.is_sufficient_for_training() else 'NO (need ≥10 with both classes)'}")
    click.echo()


@feedback.command("list")
@click.option("--limit", default=20, show_default=True)
def feedback_list(limit):
    """List stored architect decisions."""
    store   = FeedbackStore()
    entries = store.get_all_entries()
    if not entries:
        click.echo("No feedback stored yet. Review candidates in the UI or use gen-labels.")
        return

    click.echo(click.style(f"\n-- {len(entries)} Stored Decisions ──────────────────────────", fg="cyan"))
    for e in entries[:limit]:
        label_str = click.style(" ACCEPT", fg="green") if e.label == 1 else click.style("✗ REJECT", fg="red")
        click.echo(f"  {label_str}  {_short(e.comp_a)} ↔ {_short(e.comp_b)}")
        if e.rationale_text:
            click.echo(f"          Note: {e.rationale_text[:80]}")
    if len(entries) > limit:
        click.echo(f"  … and {len(entries)-limit} more")
    click.echo()


@feedback.command("import")
@click.option("--file", "-f", required=True, help="JSON labels file to import.")
@click.option("--overwrite/--no-overwrite", default=True,
              help="Overwrite existing decisions for the same pair.")
def feedback_import(file, overwrite):
    """Import labels from a JSON file into the feedback store."""
    if not Path(file).exists():
        click.echo(f"File not found: {file}", err=True)
        sys.exit(1)

    with open(file) as f:
        items = json.load(f)


    valid = [
        i for i in items
        if i.get("label") in (0, 1, True, False)
        and i.get("comp_a") and i.get("comp_b")
    ]
    skipped = len(items) - len(valid)

    store = FeedbackStore()
    count = store.bulk_record(valid)

    click.echo(click.style(f"✓ Imported {count} decisions", fg="green", bold=True))
    if skipped:
        click.echo(f"  Skipped {skipped} entries with null/missing labels")
    click.echo(f"  Store now has {store.stats()['total']} total decisions")
    click.echo(f"\n  Re-run analyze to use these labels:")
    click.echo(f"    python cli.py analyze --code <your-src>")


@feedback.command("clear")
@click.confirmation_option(prompt="This will delete ALL stored feedback. Continue?")
def feedback_clear():
    """Clear all stored architect decisions."""
    store = FeedbackStore()
    count = store.clear_all()
    click.echo(click.style(f"✓ Deleted {count} feedback entries.", fg="yellow"))


def _short(uid: str, max_len: int = 32) -> str:
    if len(uid) <= max_len:
        return uid
    parts = uid.split(".")
    return "…" + ".".join(parts[-2:]) if len(parts) >= 2 else uid[-max_len:]


if __name__ == "__main__":
    cli()
