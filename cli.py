# rdfknife/cli.py
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.table import Table

from rdflib import Graph

from . import __version__
from .core import (
    RDFToolError,
    load_graph,
    graph_info,
    search_literals,
    search_uris,
    run_sparql,
    graph_to_json_triples,
    export_json_ld,
    add_triple,
    remove_triple,
    wordnet_synsets_for_lemma,
    wordnet_synonyms_for_lemma,
    owl_entities,
)
from .shell import start_shell

console = Console()


class AppState:
    """Shared application state for the CLI."""

    def __init__(self) -> None:
        self.graph: Optional[Graph] = None
        self.input_paths: list[Path] = []
        self.rdf_format: Optional[str] = None


pass_state = click.make_pass_decorator(AppState, ensure=True)


def _configure_logging(verbosity: int) -> None:
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")


@click.group()
@click.option(
    "-i",
    "--input",
    "input_paths",
    type=click.Path(exists=True, dir_okay=False, readable=True, path_type=Path),
    multiple=True,
    help="RDF input file(s). Can be turtle, RDF/XML, JSON-LD, etc.",
)
@click.option(
    "-f",
    "--format",
    "rdf_format",
    type=str,
    default=None,
    help="Force RDF parser format (e.g. 'turtle', 'xml', 'json-ld').",
)
@click.option(
    "-v",
    "--verbose",
    count=True,
    help="Increase verbosity (-v, -vv).",
)
@pass_state
def cli(
    state: AppState,
    input_paths: tuple[Path, ...],
    rdf_format: Optional[str],
    verbose: int,
) -> None:
    """RDF Swiss-knife for WordNet RDF and OWL 2 graphs."""
    _configure_logging(verbose)
    state.rdf_format = rdf_format
    state.input_paths = list(input_paths)
    if input_paths:
        state.graph = load_graph(input_paths, rdf_format=rdf_format)


@cli.command("version")
def version_cmd() -> None:
    """Show version."""
    console.print(f"rdfknife {__version__}")


def _require_graph(state: AppState) -> Graph:
    if state.graph is None:
        raise click.UsageError("No graph loaded. Use -i/--input to provide RDF files.")
    return state.graph


@cli.command("info")
@pass_state
def info_cmd(state: AppState) -> None:
    """Show graph statistics."""
    graph = _require_graph(state)
    info = graph_info(graph)

    table = Table(title="Graph info")
    table.add_column("Metric")
    table.add_column("Value")

    for key in (
        "num_triples",
        "num_subjects",
        "num_predicates",
        "num_objects",
        "num_classes",
        "num_instances",
    ):
        table.add_row(key, str(info[key]))
    console.print(table)

    ns_table = Table(title="Namespaces")
    ns_table.add_column("Prefix")
    ns_table.add_column("Namespace")
    for prefix, uri in info["namespaces"]:
        ns_table.add_row(prefix or "(none)", uri)
    console.print(ns_table)


@cli.command("search-literal")
@click.argument("pattern", type=str)
@click.option(
    "--case-sensitive/--ignore-case",
    default=False,
    help="Use case-sensitive matching.",
)
@click.option("--limit", type=int, default=50, show_default=True)
@pass_state
def search_literal_cmd(
    state: AppState, pattern: str, case_sensitive: bool, limit: int
) -> None:
    """Search for literals containing PATTERN."""
    graph = _require_graph(state)
    results = search_literals(
        graph, pattern, case_insensitive=not case_sensitive, limit=limit
    )
    _print_triples(results)


@cli.command("search-uri")
@click.argument("pattern", type=str)
@click.option(
    "--case-sensitive/--ignore-case",
    default=False,
    help="Use case-sensitive matching.",
)
@click.option("--limit", type=int, default=50, show_default=True)
@pass_state
def search_uri_cmd(
    state: AppState, pattern: str, case_sensitive: bool, limit: int
) -> None:
    """Search URIRefs containing PATTERN."""
    graph = _require_graph(state)
    results = search_uris(
        graph, pattern, case_insensitive=not case_sensitive, limit=limit
    )
    _print_triples(results)


@cli.command("sparql")
@click.option(
    "-q",
    "--query",
    "query_str",
    type=str,
    help="SPARQL query string. If omitted, read from stdin.",
)
@click.option(
    "-f",
    "--file",
    "query_file",
    type=click.Path(exists=True, dir_okay=False, readable=True, path_type=Path),
    help="File containing a SPARQL query.",
)
@pass_state
def sparql_cmd(
    state: AppState,
    query_str: Optional[str],
    query_file: Optional[Path],
) -> None:
    """Run a SPARQL query."""
    graph = _require_graph(state)

    if query_file is not None:
        query_str = query_file.read_text(encoding="utf8")
    if not query_str:
        query_str = click.get_text_stream("stdin").read()

    if not query_str.strip():
        raise click.UsageError("Empty SPARQL query.")

    try:
        result = run_sparql(graph, query_str)
    except RDFToolError as exc:
        raise click.ClickException(str(exc)) from exc

    if isinstance(result, bool):
        console.print(f"[bold]ASK result:[/bold] {result}")
    elif isinstance(result, list):
        if not result:
            console.print("[yellow]No rows.[/yellow]")
            return
        vars_ = sorted(result[0].keys())
        table = Table(title="SPARQL SELECT results")
        for v in vars_:
            table.add_column(v)
        for row in result:
            table.add_row(*(str(row.get(v, "")) for v in vars_))
        console.print(table)
    elif isinstance(result, Graph):
        console.print(
            f"[green]Constructed graph with {len(result)} triples.[/green]"
        )
    else:
        console.print(repr(result))


@cli.command("export-json")
@click.option(
    "--mode",
    type=click.Choice(["triples", "json-ld"]),
    default="triples",
    show_default=True,
    help="Export format.",
)
@click.option(
    "-o",
    "--output",
    "output_path",
    type=click.Path(writable=True, dir_okay=False, path_type=Path),
    default=None,
    help="Output file. If omitted, print to stdout.",
)
@pass_state
def export_json_cmd(
    state: AppState, mode: str, output_path: Optional[Path]
) -> None:
    """Export the graph to JSON (triples or JSON-LD)."""
    graph = _require_graph(state)

    if mode == "triples":
        data = graph_to_json_triples(graph)
        text = json.dumps(data, indent=2, ensure_ascii=False)
    else:
        try:
            text = export_json_ld(graph)
        except RDFToolError as exc:
            raise click.ClickException(str(exc)) from exc

    if output_path is None:
        click.echo(text)
    else:
        output_path.write_text(text, encoding="utf8")
        console.print(f"[green]Wrote {output_path}[/green]")


@cli.command("add-triple")
@click.option("--subject", "-s", required=True, type=str)
@click.option("--predicate", "-p", required=True, type=str)
@click.option("--object", "-o", "object_value", required=True, type=str)
@click.option(
    "--object-type",
    type=click.Choice(["literal", "uri"]),
    default="literal",
    show_default=True,
)
@click.option("--datatype", type=str, default=None)
@click.option("--lang", type=str, default=None)
@pass_state
def add_triple_cmd(
    state: AppState,
    subject: str,
    predicate: str,
    object_value: str,
    object_type: str,
    datatype: Optional[str],
    lang: Optional[str],
) -> None:
    """Add a triple to the graph."""
    graph = _require_graph(state)
    try:
        s, p, o = add_triple(
            graph,
            subject=subject,
            predicate=predicate,
            object_value=object_value,
            object_type=object_type,
            datatype=datatype,
            lang=lang,
        )
    except RDFToolError as exc:
        raise click.ClickException(str(exc)) from exc
    console.print(f"[green]Added:[/green] {s} {p} {o}")


@cli.command("remove-triple")
@click.option("--subject", "-s", type=str, default=None)
@click.option("--predicate", "-p", type=str, default=None)
@click.option("--object", "-o", "object_value", type=str, default=None)
@pass_state
def remove_triple_cmd(
    state: AppState,
    subject: Optional[str],
    predicate: Optional[str],
    object_value: Optional[str],
) -> None:
    """Remove triples matching the given pattern."""
    graph = _require_graph(state)
    try:
        count = remove_triple(graph, subject, predicate, object_value)
    except RDFToolError as exc:
        raise click.ClickException(str(exc)) from exc
    console.print(f"[green]Removed {count} triples.[/green]")


@cli.command("save")
@click.option(
    "-o",
    "--output",
    "output_path",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    default=None,
    help="Output file. Defaults to first input file.",
)
@click.option(
    "-f",
    "--format",
    "rdf_format",
    type=str,
    default=None,
    help="RDF serialization format (turtle, xml, json-ld, ...).",
)
@pass_state
def save_cmd(
    state: AppState,
    output_path: Optional[Path],
    rdf_format: Optional[str],
) -> None:
    """Serialize the current graph back to disk."""
    graph = _require_graph(state)

    if output_path is None:
        if not state.input_paths:
            raise click.UsageError(
                "No output path specified and no input file to overwrite."
            )
        output_path = state.input_paths[0]

    fmt = rdf_format or state.rdf_format or "turtle"
    text = graph.serialize(destination=None, format=fmt)
    output_path.write_text(text, encoding="utf8")
    console.print(f"[green]Saved graph to {output_path} (format={fmt}).[/green]")


@cli.command("wn-synsets")
@click.argument("lemma", type=str)
@click.option("--lang", type=str, default=None, help="Lexical form language.")
@click.option(
    "--ignore-case/--case-sensitive",
    default=True,
    help="Case-insensitive lemma match.",
)
@pass_state
def wn_synsets_cmd(
    state: AppState,
    lemma: str,
    lang: Optional[str],
    ignore_case: bool,
) -> None:
    """List WordNet synsets for a lemma."""
    graph = _require_graph(state)
    try:
        synsets = wordnet_synsets_for_lemma(
            graph,
            lemma,
            lang=lang,
            case_insensitive=ignore_case,
        )
    except RDFToolError as exc:
        raise click.ClickException(str(exc)) from exc

    if not synsets:
        console.print("[yellow]No synsets found.[/yellow]")
        return

    table = Table(title=f"WordNet synsets for {lemma!r}")
    table.add_column("Synset URI")
    table.add_column("Synset ID")
    table.add_column("Label")
    table.add_column("Gloss")
    table.add_column("Words")

    for s in synsets:
        table.add_row(
            str(s.uri),
            s.synset_id or "",
            s.label or "",
            (s.gloss or "")[:80],
            ", ".join(s.words),
        )
    console.print(table)


@cli.command("wn-synonyms")
@click.argument("lemma", type=str)
@click.option("--lang", type=str, default=None, help="Lexical form language.")
@click.option(
    "--ignore-case/--case-sensitive",
    default=True,
    help="Case-insensitive lemma match.",
)
@pass_state
def wn_synonyms_cmd(
    state: AppState,
    lemma: str,
    lang: Optional[str],
    ignore_case: bool,
) -> None:
    """List WordNet synonyms for a lemma."""
    graph = _require_graph(state)
    try:
        mapping = wordnet_synonyms_for_lemma(
            graph,
            lemma,
            lang=lang,
            case_insensitive=ignore_case,
        )
    except RDFToolError as exc:
        raise click.ClickException(str(exc)) from exc

    if not mapping:
        console.print("[yellow]No synonyms found.[/yellow]")
        return

    table = Table(title=f"WordNet synonyms for {lemma!r}")
    table.add_column("Synonym")
    table.add_column("Synsets (IDs or URIs)")

    for syn, synsets in sorted(mapping.items(), key=lambda x: x[0]):
        ids = [
            s.synset_id if s.synset_id is not None else str(s.uri)
            for s in synsets
        ]
        table.add_row(syn, ", ".join(ids))
    console.print(table)


@cli.command("owl-entities")
@pass_state
def owl_entities_cmd(state: AppState) -> None:
    """Summarize OWL entities (classes, properties, individuals)."""
    graph = _require_graph(state)
    ents = owl_entities(graph)
    for key, values in ents.items():
        table = Table(title=key)
        table.add_column("URI")
        for uri in values:
            table.add_row(str(uri))
        console.print(table)


@cli.command("repl")
@pass_state
def repl_cmd(state: AppState) -> None:
    """Start interactive RDF shell."""
    graph = _require_graph(state)
    start_shell(graph)


def _print_triples(results: list[dict]) -> None:
    """Pretty-print triples returned by search commands."""
    if not results:
        console.print("[yellow]No matches.[/yellow]")
        return
    table = Table(title=f"{len(results)} matches")
    table.add_column("Subject")
    table.add_column("Predicate")
    table.add_column("Object")
    for r in results:
        table.add_row(str(r["subject"]), str(r["predicate"]), str(r["object"]))
    console.print(table)


def main() -> None:
    """Entry point for `python -m rdfknife.cli`."""
    cli(standalone_mode=True)


if __name__ == "__main__":
    main()
