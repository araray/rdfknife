# rdfknife/core.py
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Callable, Set

import logging

from rdflib import BNode, Graph, Literal, URIRef
from rdflib.namespace import RDF, RDFS, OWL, XSD, split_uri

logger = logging.getLogger(__name__)


class RDFToolError(Exception):
    """Base exception for rdfknife errors."""


@dataclass
class WordNetSchema:
    """Container for detected WordNet RDF schema predicates.

    Attributes:
        lexical_form: Predicate whose local name is 'lexicalForm'.
        contains_word_sense: Predicate whose local name is 'containsWordSense'.
        word: Predicate whose local name is 'word'.
        gloss: Predicate whose local name is 'gloss'.
        synset_id: Predicate whose local name is 'synsetId'.
        hyponym_of: Predicate whose local name is 'hyponymOf'.
        hypernym_of: Predicate whose local name is 'hypernymOf'.
        in_synset: Predicate whose local name is 'inSynset'.
    """

    lexical_form: Optional[URIRef] = None
    contains_word_sense: Optional[URIRef] = None
    word: Optional[URIRef] = None
    gloss: Optional[URIRef] = None
    synset_id: Optional[URIRef] = None
    hyponym_of: Optional[URIRef] = None
    hypernym_of: Optional[URIRef] = None
    in_synset: Optional[URIRef] = None

    @property
    def has_synset_link(self) -> bool:
        """True if we have *some* Synset–WordSense link predicate."""
        return self.contains_word_sense is not None or self.in_synset is not None

    @property
    def is_detected(self) -> bool:
        """Heuristic: graph is 'WordNet-like enough' for our algorithms.

        For OpenWordnet‑PT and W3C WordNet RDF, we expect at least:

        - lexicalForm (Word → lexical form literal)
        - word (WordSense → Word)
        - some synset/sense link: containsWordSense or inSynset :contentReference[oaicite:1]{index=1}
        """
        return (
            self.lexical_form is not None
            and self.word is not None
            and (self.contains_word_sense is not None or self.in_synset is not None)
        )



@dataclass
class WordNetSynset:
    """Lightweight representation of a WordNet synset within an RDF graph.

    Attributes:
        uri: Synset URI.
        label: rdfs:label if present.
        synset_id: String synset identifier if present (e.g. '00001740-n').
        gloss: Gloss text if present.
        words: List of lexical forms (strings) belonging to this synset.
    """

    uri: URIRef
    label: Optional[str]
    synset_id: Optional[str]
    gloss: Optional[str]
    words: List[str]


def _as_path(path: str | Path) -> Path:
    """Normalize a path-like input into a Path."""
    return path if isinstance(path, Path) else Path(path)


def guess_format(path: str | Path) -> Optional[str]:
    """Guess an rdflib parser format from a file extension.

    Args:
        path: File path.

    Returns:
        The guessed rdflib format string (e.g. 'turtle', 'xml') or None.
    """
    suffix = _as_path(path).suffix.lower()
    mapping = {
        ".ttl": "turtle",
        ".n3": "n3",
        ".nt": "nt",
        ".nq": "nquads",
        ".trig": "trig",
        ".rdf": "xml",
        ".owl": "xml",
        ".xml": "xml",
        ".jsonld": "json-ld",
        ".json": "json-ld",
    }
    fmt = mapping.get(suffix)
    logger.debug("Guessed RDF format %r for %s", fmt, path)
    return fmt


def load_graph(
    paths: Iterable[str | Path],
    rdf_format: Optional[str] = None,
) -> Graph:
    """Load one or more RDF files into a single in-memory graph.

    Args:
        paths: Iterable of file paths or URLs.
        rdf_format: Optional rdflib format to force parsing. If None, the
            format is guessed from each file's extension.

    Returns:
        An `rdflib.Graph` instance containing all triples.

    Raises:
        RDFToolError: If any file cannot be parsed.
    """
    graph = Graph()
    for p in paths:
        path = str(p)
        fmt = rdf_format or guess_format(path)
        try:
            logger.info("Parsing %s (format=%r)", path, fmt)
            graph.parse(path, format=fmt)
        except Exception as exc:  # noqa: BLE001
            msg = f"Failed to parse {path!r}: {exc}"
            logger.exception(msg)
            raise RDFToolError(msg) from exc
    return graph


def graph_info(graph: Graph) -> Dict[str, Any]:
    """Compute some quick statistics about a graph.

    Args:
        graph: RDF graph.

    Returns:
        Dictionary with high-level stats (triples, subjects, predicates, etc.).
    """
    num_triples = len(graph)
    subjects: set = set()
    predicates: set = set()
    objects: set = set()
    for s, p, o in graph:
        subjects.add(s)
        predicates.add(p)
        objects.add(o)

    # Classes and individuals
    classes: set = set()
    instances: set = set()
    for subj, _, obj in graph.triples((None, RDF.type, None)):
        if obj in (RDFS.Class, OWL.Class):
            classes.add(subj)
        else:
            instances.add(subj)

    namespaces = [(prefix, str(uri)) for prefix, uri in graph.namespaces()]
    info = {
        "num_triples": num_triples,
        "num_subjects": len(subjects),
        "num_predicates": len(predicates),
        "num_objects": len(objects),
        "num_classes": len(classes),
        "num_instances": len(instances),
        "namespaces": namespaces,
    }
    logger.debug("Graph info: %r", info)
    return info


def search_literals(
    graph: Graph,
    pattern: str,
    case_insensitive: bool = True,
    limit: int = 50,
) -> List[Dict[str, Any]]:
    """Search for a string pattern within literal values.

    Args:
        graph: RDF graph.
        pattern: Substring to look for.
        case_insensitive: Whether to use case-insensitive matching.
        limit: Maximum number of matches to return. Use a non-positive value
            to disable limiting.

    Returns:
        List of dictionaries, each describing one matching triple.
    """
    if case_insensitive:
        pattern_cmp = pattern.lower()
    results: List[Dict[str, Any]] = []

    for s, p, o in graph:
        if not isinstance(o, Literal):
            continue
        value = str(o)
        if case_insensitive:
            if pattern_cmp not in value.lower():
                continue
        else:
            if pattern not in value:
                continue
        results.append({"subject": s, "predicate": p, "object": o})
        if limit > 0 and len(results) >= limit:
            break

    logger.debug(
        "search_literals(%r) -> %d results", pattern, len(results)
    )
    return results


def search_uris(
    graph: Graph,
    pattern: str,
    case_insensitive: bool = True,
    limit: int = 50,
) -> List[Dict[str, Any]]:
    """Search for a string pattern in URIRefs appearing in triples.

    This checks subjects, predicates and objects that are URIRef instances.

    Args:
        graph: RDF graph.
        pattern: Substring to look for.
        case_insensitive: Whether to use case-insensitive matching.
        limit: Maximum number of matches.

    Returns:
        List of dictionaries for triples in which at least one URIRef
        component matches the pattern.
    """
    if case_insensitive:
        pattern_cmp = pattern.lower()
    results: List[Dict[str, Any]] = []

    for s, p, o in graph:
        for term in (s, p, o):
            if not isinstance(term, URIRef):
                continue
            text = str(term)
            if case_insensitive:
                if pattern_cmp not in text.lower():
                    continue
            else:
                if pattern not in text:
                    continue
            results.append({"subject": s, "predicate": p, "object": o})
            break  # Avoid adding the same triple multiple times
        if limit > 0 and len(results) >= limit:
            break

    logger.debug("search_uris(%r) -> %d results", pattern, len(results))
    return results


def run_sparql(graph: Graph, query_str: str) -> Any:
    """Execute a SPARQL query on the graph.

    Args:
        graph: RDF graph.
        query_str: SPARQL query string.

    Returns:
        - SELECT: list of dicts mapping variable names to RDF terms.
        - ASK: bool.
        - CONSTRUCT / DESCRIBE: a new Graph instance.
        - Otherwise: the raw rdflib query result.

    Raises:
        RDFToolError: If SPARQL parsing or execution fails.
    """
    from rdflib.query import Result  # Lazy import to keep import surface small

    try:
        result: Result = graph.query(query_str)
    except Exception as exc:  # noqa: BLE001
        msg = f"SPARQL query failed: {exc}"
        logger.exception(msg)
        raise RDFToolError(msg) from exc

    if result.type == "SELECT":
        rows: List[Dict[str, Any]] = []
        vars_ = list(result.vars)
        for row in result:
            row_map = {str(var): row[var] for var in vars_}
            rows.append(row_map)
        return rows

    if result.type == "ASK":
        # Result is an iterable with a single boolean row
        for row in result:
            return bool(row[0])
        return False

    if result.type in {"CONSTRUCT", "DESCRIBE"}:
        g = Graph()
        for triple in result:
            g.add(triple)
        return g

    return result


def _term_to_json(term: Any) -> Dict[str, Any]:
    """Convert an RDF term to a JSON-friendly structure.

    Args:
        term: Any RDF term (URIRef, BNode, Literal).

    Returns:
        JSON-serializable dict describing the term.
    """
    if isinstance(term, URIRef):
        return {"type": "uri", "value": str(term)}
    if isinstance(term, BNode):
        return {"type": "bnode", "value": str(term)}
    if isinstance(term, Literal):
        data: Dict[str, Any] = {
            "type": "literal",
            "value": str(term),
        }
        if term.datatype:
            data["datatype"] = str(term.datatype)
        if term.language:
            data["lang"] = term.language
        return data
    return {"type": "unknown", "value": repr(term)}


def graph_to_json_triples(graph: Graph) -> List[Dict[str, Any]]:
    """Convert a graph to a list-of-triples JSON representation.

    This is intentionally simple and lossless for RDF terms.

    Args:
        graph: RDF graph.

    Returns:
        List of dicts with 'subject', 'predicate', 'object' keys, each holding
        a JSON-serializable term representation.
    """
    triples: List[Dict[str, Any]] = []
    for s, p, o in graph:
        triples.append(
            {
                "subject": _term_to_json(s),
                "predicate": _term_to_json(p),
                "object": _term_to_json(o),
            }
        )
    return triples


def export_json_ld(graph: Graph, indent: int = 2) -> str:
    """Serialize the graph as JSON-LD.

    Args:
        graph: RDF graph.
        indent: Indentation level for pretty-printing.

    Returns:
        JSON-LD string.

    Raises:
        RDFToolError: If serialization fails.
    """
    try:
        # JSON-LD serialization is supported in rdflib 6+.
        # JSON-LD is a W3C recommendation for RDF serialization. :contentReference[oaicite:0]{index=0}
        return graph.serialize(format="json-ld", indent=indent)  # type: ignore[arg-type]
    except Exception as exc:  # noqa: BLE001
        msg = f"JSON-LD export failed: {exc}"
        logger.exception(msg)
        raise RDFToolError(msg) from exc


def _parse_uriref(graph: Graph, value: str) -> URIRef:
    """Parse a string into a URIRef, supporting CURIEs.

    Args:
        graph: RDF graph, used for namespace expansion.
        value: String like 'http://example.org/x', 'wn20:synset-…' or '<http://…>'.

    Returns:
        URIRef instance.

    Raises:
        RDFToolError: If the value cannot be interpreted as a URI.
    """
    text = value.strip()
    if text.startswith("<") and text.endswith(">"):
        return URIRef(text[1:-1])
    if "://" in text:
        return URIRef(text)
    # Attempt CURIE expansion (e.g. wn20:something)
    if ":" in text:
        try:
            uri = graph.namespace_manager.expand_curie(text)
            return URIRef(str(uri))
        except Exception:  # noqa: BLE001
            # Fall through to raw URIRef
            pass
    if not text:
        raise RDFToolError("Empty URI value is not allowed.")
    return URIRef(text)


def add_triple(
    graph: Graph,
    subject: str,
    predicate: str,
    object_value: str,
    object_type: str = "literal",
    datatype: Optional[str] = None,
    lang: Optional[str] = None,
) -> Tuple[Any, Any, Any]:
    """Add a triple to the graph.

    Args:
        graph: RDF graph.
        subject: Subject URI/CURIE.
        predicate: Predicate URI/CURIE.
        object_value: Object URI or literal string.
        object_type: Either 'literal' or 'uri'.
        datatype: Optional datatype URI/CURIE for literals.
        lang: Optional language tag for literals.

    Returns:
        The (subject, predicate, object) terms added to the graph.

    Raises:
        RDFToolError: For invalid URIs or object types.
    """
    subj = _parse_uriref(graph, subject)
    pred = _parse_uriref(graph, predicate)

    if object_type == "uri":
        obj = _parse_uriref(graph, object_value)
    elif object_type == "literal":
        dt_uri = _parse_uriref(graph, datatype) if datatype else None
        obj = Literal(object_value, datatype=dt_uri, lang=lang)
    else:
        raise RDFToolError(
            f"Unsupported object_type {object_type!r}. Use 'literal' or 'uri'."
        )

    graph.add((subj, pred, obj))
    logger.info("Added triple (%s, %s, %s)", subj, pred, obj)
    return subj, pred, obj


def remove_triple(
    graph: Graph,
    subject: Optional[str] = None,
    predicate: Optional[str] = None,
    object_value: Optional[str] = None,
) -> int:
    """Remove triples from the graph.

    Args:
        graph: RDF graph.
        subject: Optional subject URI/CURIE or blank node id.
        predicate: Optional predicate URI/CURIE.
        object_value: Optional object URI/CURIE or literal string. If provided,
            rdfknife will first try it as a URIRef, then as a Literal.

    Returns:
        Number of triples removed.

    Raises:
        RDFToolError: If all of subject, predicate and object_value are None.
    """
    if subject is None and predicate is None and object_value is None:
        raise RDFToolError(
            "Refusing to remove everything. Provide at least one of "
            "subject, predicate, or object."
        )

    subj_term: Any = None
    pred_term: Any = None
    obj_term: Any = None

    if subject is not None:
        if subject.startswith("_:"):
            subj_term = BNode(subject[2:])
        else:
            subj_term = _parse_uriref(graph, subject)
    if predicate is not None:
        pred_term = _parse_uriref(graph, predicate)
    if object_value is not None:
        # Try URIRef first, then treat as literal.
        try:
            obj_term = _parse_uriref(graph, object_value)
        except RDFToolError:
            obj_term = Literal(object_value)

    triples_to_remove = list(graph.triples((subj_term, pred_term, obj_term)))
    for triple in triples_to_remove:
        graph.remove(triple)

    logger.info("Removed %d triples", len(triples_to_remove))
    return len(triples_to_remove)


def _predicates_by_local_name(graph: Graph, local_name: str) -> List[URIRef]:
    """Find all predicates with the given local name (case-insensitive)."""
    target = local_name.lower()
    preds: set[URIRef] = set()
    for _, p, _ in graph:
        if not isinstance(p, URIRef):
            continue
        try:
            _, ln = split_uri(p)
        except ValueError:
            txt = str(p)
            if "#" in txt:
                ln = txt.rsplit("#", 1)[1]
            else:
                ln = txt.rsplit("/", 1)[-1]
        if ln.lower() == target:
            preds.add(p)
    return sorted(preds, key=str)


def _infer_lexical_form_predicate(
    graph: Graph,
    word_predicate: URIRef,
) -> Optional[URIRef]:
    """Heuristically infer the lemma/lexical-form predicate for Word nodes.

    We assume `word_predicate` is the WordSense → Word property
    (e.g. wn:word, own:word).

    Algorithm:

    1. Collect all Word nodes as objects of (?ws word_predicate ?w).
    2. For those Word nodes, inspect triples (w, p, lit) where:
        * lit is a Literal
        * p is in the same namespace as word_predicate
    3. Return the predicate p that appears most often.

    This works for variants like OWN where the lexical form property may
    not literally be named 'lexicalForm', but still lives on Word nodes
    in the same namespace as `word`. :contentReference[oaicite:1]{index=1}
    """
    # Derive the namespace of the `word_predicate`
    base = str(word_predicate)
    if "#" in base:
        base = base.rsplit("#", 1)[0] + "#"
    else:
        base = base.rsplit("/", 1)[0] + "/"

    # 1. Collect Word nodes (objects of WordSense → Word predicate)
    words: Set[URIRef] = set()
    for ws, _, w in graph.triples((None, word_predicate, None)):
        if isinstance(w, URIRef):
            words.add(w)

    if not words:
        logger.info(
            "Cannot infer lexical form predicate: no Word nodes found via %s",
            word_predicate,
        )
        return None

    # 2. Count literal-valued predicates on those Word nodes
    counts: Dict[URIRef, int] = defaultdict(int)
    for w in words:
        for _, p, o in graph.triples((w, None, None)):
            if not isinstance(p, URIRef) or not isinstance(o, Literal):
                continue
            if not str(p).startswith(base):
                # ignore properties outside the OWN schema namespace
                continue
            counts[p] += 1

    if not counts:
        logger.info(
            "Cannot infer lexical form predicate: no literal-valued "
            "predicates on Word nodes in namespace %s",
            base,
        )
        return None

    best_pred, best_n = max(counts.items(), key=lambda kv: kv[1])
    logger.info(
        "Heuristically inferred lexical_form predicate as %s "
        "from %d occurrences on Word nodes",
        best_pred,
        best_n,
    )
    return best_pred


def _predicates_by_local_name(graph: Graph, local_name: str) -> list[URIRef]:
    """Find all predicates whose local name (after '/' or '#') matches."""
    target = local_name.lower()
    preds: set[URIRef] = set()
    for _, p, _ in graph.triples((None, None, None)):
        if not isinstance(p, URIRef):
            continue
        txt = str(p)
        if "#" in txt:
            ln = txt.rsplit("#", 1)[1]
        else:
            ln = txt.rstrip("/").rsplit("/", 1)[-1]
        if ln.lower() == target:
            preds.add(p)
    return sorted(preds, key=str)


def detect_wordnet_schema(graph: Graph) -> "WordNetSchema":
    """Detect WordNet/OWN-style schema predicates in a graph.

    First, we try exact local-name matching (`lexicalForm`, `word`,
    `containsWordSense`, `inSynset`) as in the W3C WordNet RDF spec. :contentReference[oaicite:2]{index=2}

    If `lexicalForm` is not found but `word` is, we heuristically infer the
    lemma predicate by looking for the most frequent literal-valued property
    attached to Word nodes (objects of the `word` predicate) in the same
    namespace as `word`. This covers newer OWN distributions where the
    lexical form property is renamed but structurally identical. :contentReference[oaicite:3]{index=3}
    """

    def first_or_none(preds: list[URIRef]) -> Optional[URIRef]:
        return preds[0] if preds else None

    lexical_form = first_or_none(_predicates_by_local_name(graph, "lexicalForm"))
    word = first_or_none(_predicates_by_local_name(graph, "word"))
    contains_word_sense = first_or_none(
        _predicates_by_local_name(graph, "containsWordSense")
    )
    in_synset = first_or_none(_predicates_by_local_name(graph, "inSynset"))
    gloss = first_or_none(_predicates_by_local_name(graph, "gloss"))
    synset_id = first_or_none(_predicates_by_local_name(graph, "synsetId"))

    schema = WordNetSchema(
        lexical_form=lexical_form,
        word=word,
        contains_word_sense=contains_word_sense,
        in_synset=in_synset,
        gloss=gloss,
        synset_id=synset_id,
    )

    # 🔧 NEW: heuristic fallback for lexical_form
    if schema.lexical_form is None and schema.word is not None:
        inferred = _infer_lexical_form_predicate(graph, schema.word)
        if inferred is not None:
            schema.lexical_form = inferred

    logger.debug("Detected WordNet schema: %r", schema)
    return schema


def _wordnet_words_for_synset(
    graph: Graph,
    synset: URIRef,
    schema: WordNetSchema,
) -> List[str]:
    """Return lexical forms belonging to a synset."""
    words: List[str] = []
    seen: set[str] = set()

    # Walk Synset -> WordSense -> Word -> lexicalForm
    if schema.contains_word_sense is not None and schema.word is not None and schema.lexical_form is not None:
        for ws in graph.objects(synset, schema.contains_word_sense):
            for w in graph.objects(ws, schema.word):
                for lf in graph.objects(w, schema.lexical_form):
                    if isinstance(lf, Literal):
                        text = str(lf)
                        if text not in seen:
                            seen.add(text)
                            words.append(text)

    # Fallback: use rdfs:label if no lexicalForm present
    if not words:
        for label in graph.objects(synset, RDFS.label):
            if isinstance(label, Literal):
                text = str(label)
                if text not in seen:
                    seen.add(text)
                    words.append(text)

    return words


def wordnet_synsets_for_lemma(
    graph: Graph,
    lemma: str,
    lang: Optional[str] = None,
    case_insensitive: bool = False,
    limit: int = 50,
) -> List[WordNetSynset]:
    """Resolve a lemma to WordNet synsets in a WordNet RDF graph.

    This uses the W3C WordNet RDF schema, following lexicalForm → Word →
    WordSense → Synset links. :contentReference[oaicite:2]{index=2}

    Args:
        graph: RDF graph containing WordNet data.
        lemma: Lexical form to search for.
        lang: Optional language tag to filter lexicalForm literals.
        case_insensitive: Whether to match lemma case-insensitively.
        limit: Maximum number of synsets to return.

    Returns:
        List of `WordNetSynset` instances.

    Raises:
        RDFToolError: If the graph does not look like WordNet RDF.
    """
    schema = detect_wordnet_schema(graph)
    if not schema.is_detected:
        raise RDFToolError(
            "Graph does not look like W3C WordNet RDF (lexicalForm/word/"
            "containsWordSense predicates not detected)."
        )

    if case_insensitive:
        lemma_cmp = lemma.lower()

    # 1. Find Word resources with matching lexicalForm
    word_candidates: set[URIRef] = set()
    if schema.lexical_form is None:
        raise RDFToolError("WordNet schema lacks lexicalForm predicate.")
    for w, _, lit in graph.triples((None, schema.lexical_form, None)):
        if not isinstance(lit, Literal):
            continue
        if lang is not None and lit.language != lang:
            continue
        val = str(lit)
        if case_insensitive:
            if lemma_cmp != val.lower():
                continue
        else:
            if lemma != val:
                continue
        if isinstance(w, URIRef):
            word_candidates.add(w)

    if not word_candidates:
        return []

    # 2. Word -> WordSense
    if schema.word is None:
        raise RDFToolError("WordNet schema lacks 'word' predicate.")

    word_senses: set[URIRef] = set()
    for ws, _, w in graph.triples((None, schema.word, None)):
        if w in word_candidates and isinstance(ws, URIRef):
            word_senses.add(ws)

    if not word_senses:
        return []

    # 3. WordSense -> Synset
    synsets: set[URIRef] = set()
    if schema.contains_word_sense is not None:
        for syn, _, ws in graph.triples((None, schema.contains_word_sense, None)):
            if ws in word_senses and isinstance(syn, URIRef):
                synsets.add(syn)
    if schema.in_synset is not None:
        for ws, _, syn in graph.triples((None, schema.in_synset, None)):
            if ws in word_senses and isinstance(syn, URIRef):
                synsets.add(syn)

    synset_list: List[WordNetSynset] = []
    for syn in synsets:
        label: Optional[str] = None
        synset_id: Optional[str] = None
        gloss: Optional[str] = None

        # rdfs:label
        for lab in graph.objects(syn, RDFS.label):
            if isinstance(lab, Literal):
                label = str(lab)
                break

        # synsetId
        if schema.synset_id is not None:
            for sid in graph.objects(syn, schema.synset_id):
                if isinstance(sid, Literal):
                    synset_id = str(sid)
                    break

        # gloss
        if schema.gloss is not None:
            for gl in graph.objects(syn, schema.gloss):
                if isinstance(gl, Literal):
                    gloss = str(gl)
                    break

        words = _wordnet_words_for_synset(graph, syn, schema)
        synset_list.append(
            WordNetSynset(
                uri=syn,
                label=label,
                synset_id=synset_id,
                gloss=gloss,
                words=words,
            )
        )

        if limit > 0 and len(synset_list) >= limit:
            break

    return sorted(synset_list, key=lambda s: (s.synset_id or str(s.uri)))


def wordnet_synonyms_for_lemma(
    graph: Graph,
    lemma: str,
    lang: Optional[str] = None,
    case_insensitive: bool = False,
    limit: int = 100,
) -> Dict[str, List[WordNetSynset]]:
    """Return synonyms for a lemma, grouped by synset.

    Args:
        graph: RDF graph containing WordNet data.
        lemma: Lemma string to look up.
        lang: Optional lexicalForm language tag.
        case_insensitive: Whether lemma matching is case-insensitive.
        limit: Maximum total synonym count (approximate).

    Returns:
        Mapping of synonym string to the synsets it appears in.
    """
    synsets = wordnet_synsets_for_lemma(
        graph, lemma, lang=lang, case_insensitive=case_insensitive
    )
    synonyms: Dict[str, List[WordNetSynset]] = {}
    lemma_cmp = lemma.lower() if case_insensitive else lemma

    count = 0
    for syn in synsets:
        for word in syn.words:
            word_cmp = word.lower() if case_insensitive else word
            if word_cmp == lemma_cmp:
                continue
            lst = synonyms.setdefault(word, [])
            lst.append(syn)
            count += 1
            if limit > 0 and count >= limit:
                return synonyms
    return synonyms


def owl_entities(graph: Graph) -> Dict[str, List[URIRef]]:
    """Inspect OWL 2 mapped-to-RDF entities in the graph.

    OWL 2 ontologies can be serialized as RDF graphs according to the
    OWL 2 Mapping to RDF Graphs specification. :contentReference[oaicite:3]{index=3}

    This helper inspects common OWL entity types: classes, object properties,
    datatype properties, annotation properties and individuals.

    Args:
        graph: RDF graph.

    Returns:
        Dictionary with keys: 'classes', 'object_properties',
        'datatype_properties', 'annotation_properties', 'individuals'.
    """
    classes: set[URIRef] = set()
    obj_props: set[URIRef] = set()
    dt_props: set[URIRef] = set()
    ann_props: set[URIRef] = set()
    individuals: set[URIRef] = set()

    for s, _, o in graph.triples((None, RDF.type, None)):
        if not isinstance(s, URIRef):
            continue
        if o in (OWL.Class, RDFS.Class):
            classes.add(s)
        elif o == OWL.ObjectProperty:
            obj_props.add(s)
        elif o == OWL.DatatypeProperty:
            dt_props.add(s)
        elif o == OWL.AnnotationProperty:
            ann_props.add(s)
        else:
            individuals.add(s)

    return {
        "classes": sorted(classes, key=str),
        "object_properties": sorted(obj_props, key=str),
        "datatype_properties": sorted(dt_props, key=str),
        "annotation_properties": sorted(ann_props, key=str),
        "individuals": sorted(individuals, key=str),
    }

def wordnet_all_lemma_synonyms(
    graph: Graph,
    lang: Optional[str] = None,
    progress: Optional[Callable[[str], None]] = None,
) -> Dict[str, List[str]]:
    """Compute synonyms for all lemmas in a WordNet RDF graph.

    This follows the Synset–WordSense–Word structure used by the W3C
    WordNet RDF recommendation and OpenWordnet‑PT: Synset is linked
    to WordSense via containsWordSense or inSynset, WordSense to Word
    via word, and Word to its lexical form via lexicalForm. :contentReference[oaicite:3]{index=3}

    For each synset, all lemmas in that synset are considered mutual
    synonyms. The result merges across all senses: a polysemous lemma
    will have the union of synonyms from all its synsets.

    Args:
        graph: RDF graph containing WordNet/OWN-PT data.
        lang: Optional language code (e.g. "pt", "en"). If provided,
            only lexicalForm literals with that language tag are used.
        progress: Optional callback taking a human-readable progress
            message; used by the CLI to report progress.

    Returns:
        Mapping lemma -> sorted list of unique synonyms (may be empty).

    Raises:
        RDFToolError: If the graph does not look like a WordNet RDF
            graph, or if no lexicalForm literals are found for the
            requested language.
    """
    def log(msg: str) -> None:
        if progress is not None:
            progress(msg)

    schema = detect_wordnet_schema(graph)
    if not schema.is_detected:
        raise RDFToolError(
            "Graph does not look like W3C/OWN-PT WordNet RDF "
            "(lexicalForm/word plus a synset–sense link predicate "
            "not detected)."
        )

    if schema.lexical_form is None or schema.word is None:
        raise RDFToolError(
            "WordNet schema is missing lexicalForm or word predicate."
        )

    log("Step 1/3: indexing Words via lexicalForm ...")

    # 1. Word -> lemma (language-filtered)
    word_to_lemma: Dict[URIRef, str] = {}
    for word_uri, _, lit in graph.triples((None, schema.lexical_form, None)):
        if not isinstance(lit, Literal):
            continue
        if not isinstance(word_uri, URIRef):
            continue
        if lang is not None and lit.language != lang:
            continue
        word_to_lemma[word_uri] = str(lit)

    if not word_to_lemma:
        msg = "No Word lexicalForm literals found"
        if lang is not None:
            msg += f" for language {lang!r}"
        raise RDFToolError(msg + ".")

    log(f"  - collected {len(word_to_lemma)} Word nodes with lexicalForm.")

    # 2. WordSense -> Word
    log("Step 2/3: indexing WordSenses via word ...")

    ws_to_word: Dict[URIRef, URIRef] = {}
    for ws_uri, _, word_uri in graph.triples((None, schema.word, None)):
        if not isinstance(ws_uri, URIRef) or not isinstance(word_uri, URIRef):
            continue
        if word_uri not in word_to_lemma:
            continue
        ws_to_word[ws_uri] = word_uri

    log(f"  - mapped {len(ws_to_word)} WordSense nodes to Words.")

    # 3. Synset -> WordSense (containsWordSense / inSynset)
    log("Step 3/3: grouping WordSenses by Synset ...")

    synset_to_senses: Dict[URIRef, set[URIRef]] = defaultdict(set)

    if schema.contains_word_sense is not None:
        for synset_uri, _, ws_uri in graph.triples(
            (None, schema.contains_word_sense, None)
        ):
            if not isinstance(synset_uri, URIRef) or not isinstance(ws_uri, URIRef):
                continue
            synset_to_senses[synset_uri].add(ws_uri)

    if schema.in_synset is not None:
        for ws_uri, _, synset_uri in graph.triples(
            (None, schema.in_synset, None)
        ):
            if not isinstance(synset_uri, URIRef) or not isinstance(ws_uri, URIRef):
                continue
            synset_to_senses[synset_uri].add(ws_uri)

    # Synset -> lemmas
    synset_to_lemmas: Dict[URIRef, List[str]] = defaultdict(list)
    for synset_uri, senses in synset_to_senses.items():
        for ws_uri in senses:
            word_uri = ws_to_word.get(ws_uri)
            if word_uri is None:
                continue
            lemma = word_to_lemma.get(word_uri)
            if lemma is None:
                continue
            synset_to_lemmas[synset_uri].append(lemma)

    # 4. Lemma -> set of synonyms
    lemma_synonyms: Dict[str, set[str]] = defaultdict(set)
    all_lemmas = set(word_to_lemma.values())

    for lemmas in synset_to_lemmas.values():
        # Deduplicate per-synset while preserving order
        seen_local: set[str] = set()
        unique_lemmas: List[str] = []
        for lemma in lemmas:
            if lemma in seen_local:
                continue
            seen_local.add(lemma)
            unique_lemmas.append(lemma)

        for i, lemma in enumerate(unique_lemmas):
            for j, other in enumerate(unique_lemmas):
                if i == j:
                    continue
                lemma_synonyms[lemma].add(other)

    # 5. Normalize: lemma -> sorted list, include lemmas without synonyms
    result: Dict[str, List[str]] = {}
    for lemma in sorted(all_lemmas):
        syns = sorted(lemma_synonyms.get(lemma, set()))
        result[lemma] = syns

    log(f"Done: computed synonyms for {len(result)} lemmas.")
    return result
