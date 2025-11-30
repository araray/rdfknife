# rdfknife/__init__.py
"""
Top-level package for rdfknife.

This package provides a small Swiss-army-knife for working with RDF graphs,
with a focus on W3C WordNet RDF and OWL 2 ontologies mapped to RDF graphs.
"""

from __future__ import annotations

from .core import (
    RDFToolError,
    load_graph,
    graph_info,
    search_literals,
    search_uris,
    run_sparql,
    graph_to_json_triples,
    add_triple,
    remove_triple,
    detect_wordnet_schema,
    wordnet_synsets_for_lemma,
    wordnet_synonyms_for_lemma,
    owl_entities,
)

__all__ = [
    "RDFToolError",
    "load_graph",
    "graph_info",
    "search_literals",
    "search_uris",
    "run_sparql",
    "graph_to_json_triples",
    "add_triple",
    "remove_triple",
    "detect_wordnet_schema",
    "wordnet_synsets_for_lemma",
    "wordnet_synonyms_for_lemma",
    "owl_entities",
    "__version__",
]

__version__ = "0.1.0"
