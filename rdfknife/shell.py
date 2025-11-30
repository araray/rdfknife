# rdfknife/shell.py
from __future__ import annotations

import shlex
from typing import Optional

from rdflib import Graph
from rich.console import Console
from rich.table import Table

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter

from .core import (
    RDFToolError,
    graph_info,
    search_literals,
    search_uris,
    run_sparql,
    graph_to_json_triples,
    export_json_ld,
    wordnet_synsets_for_lemma,
    wordnet_synonyms_for_lemma,
    owl_entities,
)

console = Console()


def start_shell(graph: Graph) -> None:
    """Start an interactive RDF shell.

    Args:
        graph: RDF graph to operate on.

    Commands:
        help
        info
        search-literal <pattern>
        search-uri <pattern>
        sparql <query>      # query must be a single token or quoted
        export-json [triples|json-ld]
        wn-synsets <lemma>
        wn-synonyms <lemma>
        owl-entities
        quit / exit
    """
    commands = [
        "help",
        "info",
        "search-literal",
        "search-uri",
        "sparql",
        "export-json",
        "wn-synsets",
        "wn-synonyms",
        "owl-entities",
        "quit",
        "exit",
    ]
    completer = WordCompleter(commands, ignore_case=True, sentence=True)
    session = PromptSession(completer=completer)

    console.print("[bold green]rdfknife REPL[/bold green] – type 'help' for commands.")
    while True:
        try:
            line = session.prompt("rdfknife> ")
        except (KeyboardInterrupt, EOFError):
            console.print()
            break

        line = line.strip()
        if not line:
            continue

        try:
            tokens = shlex.split(line)
        except ValueError as exc:  # malformed quoting
            console.print(f"[red]Parse error:[/red] {exc}")
            continue

        cmd = tokens[0].lower()
        args = tokens[1:]

        if cmd in {"quit", "exit"}:
            break
        if cmd == "help":
            _shell_help()
        elif cmd == "info":
            _shell_info(graph)
        elif cmd == "search-literal":
            _shell_search_literal(graph, args)
        elif cmd == "search-uri":
            _shell_search_uri(graph, args)
        elif cmd == "sparql":
            _shell_sparql(graph, args)
        elif cmd == "export-json":
            _shell_export_json(graph, args)
        elif cmd == "wn-synsets":
            _shell_wn_synsets(graph, args)
        elif cmd == "wn-synonyms":
            _shell_wn_synonyms(graph, args)
        elif cmd == "owl-entities":
            _shell_owl_entities(graph)
        else:
            console.print(f"[red]Unknown command:[/red] {cmd!r}")


def _shell_help() -> None:
    """Print shell help."""
    table = Table(title="rdfknife commands", show_lines=True)
    table.add_column("Command")
    table.add_column("Description")
    table.add_row("help", "Show this help.")
    table.add_row("info", "Show graph statistics.")
    table.add_row("search-literal <pattern>", "Search for literals containing pattern.")
    table.add_row("search-uri <pattern>", "Search in URIRefs containing pattern.")
    table.add_row("sparql <query>", "Run a SPARQL query (quote multi-word).")
    table.add_row("export-json [triples|json-ld]", "Export graph as JSON.")
    table.add_row("wn-synsets <lemma>", "List WordNet synsets for lemma.")
    table.add_row("wn-synonyms <lemma>", "List WordNet synonyms for lemma.")
    table.add_row("owl-entities", "Summarize OWL entities in graph.")
    table.add_row("quit / exit", "Leave the shell.")
    console.print(table)


def _shell_info(graph: Graph) -> None:
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


def _shell_search_literal(graph: Graph, args: list[str]) -> None:
    if not args:
        console.print("[red]Usage:[/red] search-literal <pattern>")
        return
    pattern = args[0]
    results = search_literals(graph, pattern)
    _print_triples(results)


def _shell_search_uri(graph: Graph, args: list[str]) -> None:
    if not args:
        console.print("[red]Usage:[/red] search-uri <pattern>")
        return
    pattern = args[0]
    results = search_uris(graph, pattern)
    _print_triples(results)


def _shell_sparql(graph: Graph, args: list[str]) -> None:
    if not args:
        console.print(
            "[red]Usage:[/red] sparql \"SELECT ...\"   "
            "(wrap the query in quotes)"
        )
        return
    query = " ".join(args)
    try:
        result = run_sparql(graph, query)
    except RDFToolError as exc:
        console.print(f"[red]SPARQL error:[/red] {exc}")
        return

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


def _shell_export_json(graph: Graph, args: list[str]) -> None:
    mode = "triples"
    if args and args[0] in {"triples", "json-ld"}:
        mode = args[0]
    if mode == "triples":
        data = graph_to_json_triples(graph)
        console.print(data)
    else:
        try:
            text = export_json_ld(graph)
        except RDFToolError as exc:
            console.print(f"[red]JSON-LD export failed:[/red] {exc}")
            return
        console.print(text)


def _shell_wn_synsets(graph: Graph, args: list[str]) -> None:
    if not args:
        console.print("[red]Usage:[/red] wn-synsets <lemma>")
        return
    lemma = args[0]
    try:
        synsets = wordnet_synsets_for_lemma(graph, lemma)
    except RDFToolError as exc:
        console.print(f"[red]WordNet error:[/red] {exc}")
        return

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


def _shell_wn_synonyms(graph: Graph, args: list[str]) -> None:
    if not args:
        console.print("[red]Usage:[/red] wn-synonyms <lemma>")
        return
    lemma = args[0]
    try:
        mapping = wordnet_synonyms_for_lemma(graph, lemma)
    except RDFToolError as exc:
        console.print(f"[red]WordNet error:[/red] {exc}")
        return

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


def _shell_owl_entities(graph: Graph) -> None:
    ents = owl_entities(graph)
    for key, values in ents.items():
        table = Table(title=key)
        table.add_column("URI")
        for uri in values:
            table.add_row(str(uri))
        console.print(table)


def _print_triples(results: list[dict]) -> None:
    """Pretty-print triples from search commands."""
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
