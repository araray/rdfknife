# tests/test_core.py
from __future__ import annotations

from rdflib import Graph, Namespace, Literal, URIRef
from rdflib.namespace import RDF, RDFS

from rdfknife.core import (
    graph_info,
    search_literals,
    search_uris,
    graph_to_json_triples,
    detect_wordnet_schema,
    wordnet_synsets_for_lemma,
    wordnet_synonyms_for_lemma,
)


WN = Namespace("http://www.w3.org/2006/03/wn/wn20/schema/")
EX = Namespace("http://example.org/wn/")


def make_test_graph() -> Graph:
    """Create a tiny WordNet-like RDF graph for testing."""
    g = Graph()
    g.bind("wn20", WN)
    g.bind("ex", EX)

    # Word, WordSense, Synset with one lemma 'car'
    word_car = EX["word-car"]
    ws_car = EX["wordsense-car-1"]
    syn_car = EX["synset-car-noun-1"]

    g.add((word_car, RDF.type, WN.Word))
    g.add((word_car, WN.lexicalForm, Literal("car", lang="en")))

    g.add((ws_car, RDF.type, WN.WordSense))
    g.add((ws_car, WN.word, word_car))

    g.add((syn_car, RDF.type, WN.Synset))
    g.add((syn_car, WN.containsWordSense, ws_car))
    g.add((syn_car, WN.synsetId, Literal("01234567-n")))
    g.add((syn_car, WN.gloss, Literal("a road vehicle")))
    g.add((syn_car, RDFS.label, Literal("car", lang="en")))

    # Another word in same synset: 'automobile'
    word_auto = EX["word-automobile"]
    ws_auto = EX["wordsense-automobile-1"]
    g.add((word_auto, RDF.type, WN.Word))
    g.add((word_auto, WN.lexicalForm, Literal("automobile", lang="en")))
    g.add((ws_auto, RDF.type, WN.WordSense))
    g.add((ws_auto, WN.word, word_auto))
    g.add((syn_car, WN.containsWordSense, ws_auto))

    # Non-WordNet triple for generic tests
    foo = URIRef("http://example.org/foo")
    bar = URIRef("http://example.org/bar")
    g.add((foo, bar, Literal("Some literal value", lang="en")))

    return g


def test_graph_info_basic() -> None:
    g = make_test_graph()
    info = graph_info(g)
    assert info["num_triples"] > 0
    assert info["num_subjects"] > 0
    assert any(prefix == "wn20" for prefix, _ in info["namespaces"])


def test_search_literals_finds_value() -> None:
    g = make_test_graph()
    results = search_literals(g, "road vehicle")
    assert len(results) == 1
    triple = results[0]
    assert str(triple["object"]) == "a road vehicle"


def test_search_uris_finds_subject() -> None:
    g = make_test_graph()
    results = search_uris(g, "synset-car-noun-1")
    assert any("synset-car-noun-1" in str(r["subject"]) for r in results)


def test_graph_to_json_triples_roundtrip_shape() -> None:
    g = make_test_graph()
    triples = graph_to_json_triples(g)
    assert triples
    sample = triples[0]
    assert {"subject", "predicate", "object"} <= set(sample.keys())
    assert sample["subject"]["type"] in {"uri", "bnode"}


def test_detect_wordnet_schema() -> None:
    g = make_test_graph()
    schema = detect_wordnet_schema(g)
    assert schema.is_detected
    assert schema.lexical_form is not None
    assert schema.contains_word_sense is not None
    assert schema.word is not None


def test_wordnet_synsets_for_lemma() -> None:
    g = make_test_graph()
    synsets = wordnet_synsets_for_lemma(g, "car", lang="en")
    assert len(synsets) == 1
    syn = synsets[0]
    assert syn.synset_id == "01234567-n"
    assert "car" in syn.words
    assert "automobile" in syn.words


def test_wordnet_synonyms_for_lemma() -> None:
    g = make_test_graph()
    synonyms = wordnet_synonyms_for_lemma(g, "car", lang="en")
    # 'automobile' should be a synonym of 'car'
    assert "automobile" in synonyms
    synsets = synonyms["automobile"]
    assert synsets
    assert synsets[0].synset_id == "01234567-n"
