"""Command-line interface: `patsearch build | search | similar | claim | show | bench | serve`.

Thin layer over SearchEngine: parse options (click), call the engine, print the response (rich).
Registered as the `patsearch` command in pyproject.toml ([project.scripts]).
"""

from __future__ import annotations

import functools
import json
import re
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.text import Text

from .config import DEFAULT_ARTIFACTS_DIR, DEFAULT_DATA_DIR, MODEL_ID
from .filters import FilterSpec
from .search import STRATEGIES, SearchEngine, SearchResponse

console = Console()


# ----------------------------------------------------------------------------------------------
# shared options
# ----------------------------------------------------------------------------------------------
def filter_options(f):
    """Decorator adding the shared hybrid-filter options to a command."""
    opts = [
        click.option("--cpc", multiple=True, help="CPC constraint, repeatable (OR): B60, B60B, 'B60C 11', 'B60C 11/03'."),
        click.option("--kw", "keywords", multiple=True, help="Keyword/phrase that must appear in title/abstract (repeatable). Trailing * = prefix."),
        click.option("--kw-any", is_flag=True, help="Require ANY keyword instead of ALL."),
        click.option("--kw-field", type=click.Choice(["both", "title", "abstract"]), default="both", show_default=True),
        click.option("--title", help="Exact title (case/punctuation-insensitive)."),
        click.option("--title-contains", help="Phrase that must appear in the title."),
        click.option("--date-from", help="Publication date >= (YYYY-MM-DD)."),
        click.option("--date-to", help="Publication date <= (YYYY-MM-DD)."),
    ]
    for o in reversed(opts):  # applied bottom-up, so reverse to keep this order in --help
        f = o(f)
    return f


def output_options(f):
    """Decorator adding the shared output options (-k, --json, --full, --index)."""
    opts = [
        click.option("-k", "--top-k", "k", default=10, show_default=True, help="Number of results."),
        click.option("--json", "as_json", is_flag=True, help="Print the raw JSON response."),
        click.option("--full", is_flag=True, help="Print full evidence passages instead of snippets."),
        click.option("--index", "index_dir", type=click.Path(path_type=Path), default=None,
                    help="Build directory (default: artifacts/LATEST)."),
    ]
    for o in reversed(opts):
        f = o(f)
    return f


def _spec(cpc, keywords, kw_any, kw_field, title, title_contains, date_from, date_to) -> FilterSpec:
    """Filter options from the command line -> FilterSpec."""
    return FilterSpec(
        cpc=list(cpc), keywords=list(keywords), kw_mode="any" if kw_any else "all", kw_field=kw_field,
        title=title, title_contains=title_contains, date_from=date_from, date_to=date_to,
    )


def handle_errors(f):
    """Show expected errors (unknown patent, bad filter, missing index) as one red line, not a traceback."""
    @functools.wraps(f)
    def wrapper(*a, **kw):
        try:
            return f(*a, **kw)
        except (KeyError, ValueError, FileNotFoundError) as exc:
            msg = exc.args[0] if exc.args else str(exc)
            console.print(f"[red]error:[/red] {msg}")
            sys.exit(2)
    return wrapper


# ----------------------------------------------------------------------------------------------
# rendering
# ----------------------------------------------------------------------------------------------
def _snippet(text: str, terms: list[str], width: int = 300) -> str:
    """Shorten a passage for the terminal, centred on the first keyword hit if there is one."""
    if len(text) <= width:
        return text
    for t in terms:
        m = re.search(re.escape(t), text, re.I)
        if m:
            start = max(0, m.start() - width // 3)
            s = text[start : start + width]
            return ("…" if start else "") + s + "…"
    return text[:width] + "…"


def render(resp: SearchResponse, full: bool = False) -> None:
    """Pretty-print a response: one line per patent, its evidence passages below, timings last."""
    spec = resp.filters
    # words to highlight: the keyword filters (the semantic query itself isn't highlighted,
    # because matches are by meaning, not by shared words)
    terms = [t.rstrip("*") for t in spec.get("keywords", [])]
    if spec.get("title_contains"):
        terms.append(spec["title_contains"].rstrip("*"))
    q = resp.query
    if resp.mode == "similar":
        console.print(f"[bold]Patents similar to {q['doc_number']}[/bold] ({q['source_vectors']} source vectors)")
    elif resp.mode == "claim":
        head = f"{q.get('doc_number', 'pasted text')} {q.get('label', '')}".strip()
        console.print(f"[bold]Passages closest to {head}[/bold]: {_snippet(q.get('text', ''), [], 160)}")
    for w in resp.warnings:
        console.print(f"[yellow]warning:[/yellow] {w}")
    if not resp.hits:
        console.print("[dim]no results[/dim]")
    for h in resp.hits:
        line = Text()
        line.append(f"{h.rank:>2}. ", style="bold")
        if h.score is not None:
            line.append(f"{h.score:.3f}  ", style="cyan")
        line.append(f"{h.doc_number}  ", style="bold")
        line.append(f"{h.pub_date or '?'}  {h.cpc or '-':<12} ", style="dim")
        title = Text(h.title)
        title.highlight_words(terms, style="bold yellow", case_sensitive=False)
        line.append_text(title)
        if h.also_published_as:
            line.append(f"  [also {', '.join(h.also_published_as)}]", style="magenta")
        if h.same_family:
            line.append("  [same family]", style="magenta")
        if not h.has_description:
            line.append("  [no description]", style="dim")
        console.print(line)
        for p in h.evidence:
            ev = Text("      ")
            src = f"{p.matched_from} → " if p.matched_from else ""
            ev.append(f"{src}{p.label}", style="green")
            ev.append(f" ({p.score:.3f}) ", style="dim")
            body = Text(p.text if full else _snippet(p.text, terms))
            body.highlight_words(terms, style="bold yellow", case_sensitive=False)
            ev.append_text(body)
            console.print(ev)
    t = resp.timings_ms
    parts = [f"{k} {v:.2f}" for k, v in t.items() if k != "total"]
    sel = ""
    if resp.n_candidates is not None and resp.filters:
        sel = f" · filter kept {resp.n_candidates}/{resp.n_docs} patents ({resp.selectivity_docs:.1%}, rows {resp.selectivity_rows:.1%})"
    console.print(
        f"[dim]{len(resp.hits)} results{sel} · strategy={resp.strategy} path={resp.path} · "
        f"ms: {', '.join(parts)} · total {t['total']:.2f}[/dim]"
    )


def emit(resp: SearchResponse, as_json: bool, full: bool) -> None:
    """Print the response as JSON (--json) or as formatted text."""
    if as_json:
        click.echo(json.dumps(resp.to_dict(), indent=2, ensure_ascii=False))
    else:
        render(resp, full)


# ----------------------------------------------------------------------------------------------
# commands
# ----------------------------------------------------------------------------------------------
@click.group()
def main() -> None:
    """Hybrid semantic search over patent applications."""


@main.command()
@click.option("--data-dir", type=click.Path(path_type=Path, exists=True), default=DEFAULT_DATA_DIR, show_default=True)
@click.option("--out", "artifacts_root", type=click.Path(path_type=Path), default=DEFAULT_ARTIFACTS_DIR, show_default=True)
@click.option("--model", "model_id", default=MODEL_ID, show_default=True, help="sentence-transformers model id (or hash-256 for a model-free smoke test).")
@click.option("--device", default=None, help="mps / cuda / cpu (default: auto).")
@click.option("--batch-size", default=64, show_default=True)
@click.option("--limit-files", type=int, default=None, help="Only ingest the first N JSON files.")
@handle_errors
def build(data_dir, artifacts_root, model_id, device, batch_size, limit_files):
    """Ingest the JSON files and build the index (SQLite + embeddings)."""
    from .build import build_index  # imported here so search commands don't load the build code

    out_dir, m = build_index(
        data_dir, artifacts_root, model_id=model_id, device=device, batch_size=batch_size,
        limit_files=limit_files, log=lambda s: console.print(f"[dim]{s}[/dim]"),
    )
    c = m["counts"]
    console.print(f"[bold green]built[/bold green] {out_dir}")
    console.print(
        f"docs {c['docs']} (aliases {c['aliases']}, without description {c['docs_without_description']}), "
        f"chunks {c['chunks']} {c['chunks_by_kind']}, truncated {c['chunks_truncated']}"
    )
    console.print(f"claim units {c['claim_units']}, canceled claims {c['claims_canceled']}")
    console.print("stage timings (ms): " + ", ".join(f"{k} {v:.0f}" for k, v in m["timings_ms"].items()))
    console.print(f"throughput: {m['throughput']}")


@main.command()
@click.argument("query", required=False)
@filter_options
@output_options
@click.option("--scope", default=None, help="Restrict matching to chunk kinds: abstract,claims,description.")
@click.option("--granularity", type=click.Choice(["patent", "passage"]), default="patent", show_default=True)
@click.option("--strategy", type=click.Choice(STRATEGIES), default="pre", show_default=True, help="How hybrid filters are applied.")
@handle_errors
def search(query, cpc, keywords, kw_any, kw_field, title, title_contains, date_from, date_to, k, as_json, full,
        index_dir, scope, granularity, strategy):
    """Natural-language search. Without QUERY, lists patents matching the filters (newest first)."""
    eng = SearchEngine(index_dir)
    spec = _spec(cpc, keywords, kw_any, kw_field, title, title_contains, date_from, date_to)
    emit(eng.search(query, spec, k=k, scope=scope, granularity=granularity, strategy=strategy), as_json, full)


@main.command()
@click.argument("doc_number")
@filter_options
@output_options
@click.option("--scope", default=None, help="Candidate chunk kinds to match against.")
@click.option("--candidates", default=50, show_default=True, help="Stage-1 candidates re-scored by max-sim.")
@handle_errors
def similar(doc_number, cpc, keywords, kw_any, kw_field, title, title_contains, date_from, date_to, k, as_json, full,
            index_dir, scope, candidates):
    """Patents most similar to DOC_NUMBER (uses its abstract + claims; no model call)."""
    eng = SearchEngine(index_dir)
    spec = _spec(cpc, keywords, kw_any, kw_field, title, title_contains, date_from, date_to)
    emit(eng.similar(doc_number, spec, k=k, scope=scope, n_candidates=candidates), as_json, full)


@main.command()
@click.argument("ref", required=False)
@click.option("--text", help="Paste claim text instead of referencing DOC:CLAIM.")
@filter_options
@output_options
@click.option("--scope", default="claims,description", show_default=True)
@click.option("--granularity", type=click.Choice(["patent", "passage"]), default="patent", show_default=True)
@click.option("--include-source", is_flag=True, help="Do not exclude the claim's own patent.")
@handle_errors
def claim(ref, text, cpc, keywords, kw_any, kw_field, title, title_contains, date_from, date_to, k, as_json, full,
        index_dir, scope, granularity, include_source):
    """Claim mapping: REF like 20240051333:1 -> closest claims/paragraphs in other patents."""
    eng = SearchEngine(index_dir)
    spec = _spec(cpc, keywords, kw_any, kw_field, title, title_contains, date_from, date_to)
    resp = eng.claim(ref, text, spec, k=k, scope=scope, granularity=granularity, exclude_source=not include_source)
    emit(resp, as_json, full)


@main.command()
@click.argument("doc_number")
@click.option("--index", "index_dir", type=click.Path(path_type=Path), default=None)
@click.option("--chunks", "show_chunks", is_flag=True, help="Also list the indexed chunks.")
@handle_errors
def show(doc_number, index_dir, show_chunks):
    """Show a patent's metadata, abstract and re-assembled claims."""
    eng = SearchEngine(index_dir)
    s = eng.store
    d = s.resolve(doc_number)
    if d is None:
        raise KeyError(f"unknown patent {doc_number!r}")
    m = s.docs([d])[d]
    aliases = s.aliases_of([d]).get(d, [])
    console.print(f"[bold]{m['doc_number']}[/bold] {m['kind'] or ''}  {m['title']}")
    console.print(
        f"published {m['pub_date']} · CPC {m['cpc_display']} (raw {m['cpc_raw']}) · "
        f"description: {'yes' if m['has_description'] else 'no'} · claim units {m['n_claim_units']} · "
        f"canceled {m['n_canceled']}" + (f" · also published as {', '.join(aliases)}" if aliases else "")
    )
    flags = json.loads(m["flags"])
    if flags:
        console.print(f"[yellow]data flags:[/yellow] {', '.join(flags)}")
    console.print(f"\n[bold]Abstract[/bold]\n{m['abstract']}\n\n[bold]Claims[/bold]")
    for c in s.claims_of(d):
        dep = f" (depends on {c['depends_on']})" if c["depends_on"] else ""
        console.print(f"[green]{c['label']}[/green]{dep}: {c['text']}")
    if show_chunks:
        rows = s.con.execute("SELECT chunk_id, kind, label, length(text) AS n FROM chunks WHERE doc_id=? ORDER BY chunk_id", (d,))
        console.print("\n[bold]Chunks[/bold]")
        for r in rows:
            console.print(f"  #{r['chunk_id']} {r['label']} ({r['n']} chars)")


@main.command()
@click.option("--index", "index_dir", type=click.Path(path_type=Path), default=None)
@click.option("--reps", default=200, show_default=True)
@click.option("--warmup", default=5, show_default=True)
@click.option("--synthetic", default="", help="Comma-separated synthetic corpus sizes in chunk rows, e.g. 100000,1000000.")
@click.option("--out", "out_md", type=click.Path(path_type=Path), default=Path("results/bench_part1.md"), show_default=True)
@handle_errors
def bench(index_dir, reps, warmup, synthetic, out_md):
    """Time search with vs without hybrid filters (writes a markdown report)."""
    from .bench import run_benchmark

    sizes = [int(x) for x in synthetic.split(",") if x.strip()]
    run_benchmark(index_dir, reps=reps, warmup=warmup, synthetic_sizes=sizes, out_md=out_md, log=console.print)


@main.command()
@click.option("--index", "index_dir", type=click.Path(path_type=Path), default=None)
@click.option("--host", default="127.0.0.1", show_default=True)
@click.option("--port", default=8000, show_default=True)
def serve(index_dir, host, port):
    """Serve the test web UI at / and the JSON API (docs at /docs)."""
    import uvicorn

    from .api import create_app

    console.print("[dim]loading index and warming up the model…[/dim]")
    app = create_app(index_dir, warm=True, mmap=False)  # 37 MB matrix: keep it in RAM
    console.print(f"[bold green]open http://{host}:{port}[/bold green]  (API docs: http://{host}:{port}/docs)")
    uvicorn.run(app, host=host, port=port, log_level="warning")


if __name__ == "__main__":
    main()
