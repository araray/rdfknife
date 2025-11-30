# rdfknife

A small but sharp Swiss‑army knife for RDF graphs, with first‑class support
for:

- W3C RDF/OWL representation of WordNet :contentReference[oaicite:5]{index=5}
- OWL 2 ontologies mapped to RDF graphs :contentReference[oaicite:6]{index=6}

Built on top of [`rdflib`](https://rdflib.readthedocs.io/), `click`, `rich`,
and `prompt_toolkit`.

## Features

- Load one or more RDF files (Turtle, RDF/XML, JSON‑LD, N‑Triples, …)
- Graph inspection: triple counts, namespaces, classes, instances
- Search:
  - Literal substring search
  - URI substring search
- SPARQL querying (SELECT/ASK/CONSTRUCT/DESCRIBE)
- Editing:
  - Add triples (URI or literal objects)
  - Remove triples by pattern
- Export:
  - JSON triple list (lossless)
  - JSON‑LD (via rdflib’s JSON‑LD serializer) :contentReference[oaicite:7]{index=7}
- WordNet helpers (W3C WordNet RDF):
  - Detect WordNet schema predicates
  - Lemma → synsets (with glosses and IDs)
  - Lemma → synonyms, grouped by synset
- OWL helpers (OWL 2 Mapping to RDF):
  - Inspect OWL classes, properties, individuals

## Installation

```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

pip install -r requirements.txt
````

You can install it as a package (editable mode) from the project root:

```bash
pip install -e .
```

*(If you package it with `pyproject.toml`, add an appropriate console script
entry pointing at `rdfknife.cli:main`.)*

## Quick start

### Load a graph and inspect it

```bash
rdfknife info -i data/wn-sample.ttl
```

### Search literals and URIs

```bash
rdfknife search-literal "vehicle" -i data/wn-sample.ttl
rdfknife search-uri "synset-car-noun-1" -i data/wn-sample.ttl
```

### Run SPARQL

```bash
rdfknife sparql -i data/wn-sample.ttl -q \
"PREFIX wn20: <http://www.w3.org/2006/03/wn/wn20/schema/>
 SELECT ?syn WHERE {
   ?syn wn20:containsWordSense ?ws .
   ?ws wn20:word ?w .
   ?w wn20:lexicalForm \"car\"@en .
 }"
```

### Export as JSON

```bash
# Triples-as-JSON
rdfknife export-json -i data/wn-sample.ttl --mode triples -o wn.json

# JSON-LD (requires rdflib >= 6.0 with JSON-LD plugin built-in)
rdfknife export-json -i data/wn-sample.ttl --mode json-ld -o wn.jsonld
```

### Edit triples

```bash
# Add a label
rdfknife add-triple -i data/ontology.ttl \
    --subject ex:Car \
    --predicate rdfs:label \
    --object "Car" \
    --object-type literal \
    --lang en

# Remove all triples with that label literal
rdfknife remove-triple -i data/ontology.ttl \
    --object "Car"
```

### WordNet helpers

The tool is designed around the W3C WordNet RDF/OWL schema, which models
`Synset`, `Word`, and `WordSense` with properties such as `lexicalForm`,
`containsWordSense`, `gloss`, `synsetId`, and semantic relations like
`hyponymOf` and `hypernymOf`. ([W3C][2])

```bash
# Lemma -> synsets
rdfknife wn-synsets car -i data/wn-full.ttl --lang en

# Lemma -> synonyms
rdfknife wn-synonyms car -i data/wn-full.ttl --lang en
```

### OWL helpers

OWL 2 ontologies can be mapped to RDF graphs using the OWL 2 Mapping to RDF
Graphs recommendation. `rdfknife` inspects typical OWL entity types in such
graphs. ([Wikipedia][3])

```bash
rdfknife owl-entities -i data/ontology.owl
```

### REPL

```bash
rdfknife repl -i data/wn-sample.ttl
```

Inside:

```text
rdfknife> info
rdfknife> search-literal "vehicle"
rdfknife> wn-synsets car
rdfknife> sparql "SELECT ?s WHERE { ?s ?p ?o } LIMIT 10"
rdfknife> exit
```

## Programmatic usage

```python
from rdfknife import load_graph, search_literals, wordnet_synsets_for_lemma

g = load_graph(["data/wn-sample.ttl"])
matches = search_literals(g, "vehicle")
synsets = wordnet_synsets_for_lemma(g, "car", lang="en")
for s in synsets:
    print(s.synset_id, s.words, s.gloss)
```

## Testing

```bash
pytest -q
```

## References

1. W3C RDF/OWL representation of WordNet. ([W3C][2])
2. Van Assem et al., “Conversion of WordNet to a standard RDF/OWL representation.” ([Vrije Universiteit Amsterdam][4])
3. Rademaker et al., OpenWordNet‑PT project and its RDF modeling. ([ACL Anthology][5])
4. W3C OWL 2 Web Ontology Language: Mapping to RDF Graphs. ([Wikipedia][3])
5. RDFLib documentation and project site. ([RDFLib][1])
6. W3C Turtle syntax and RDF serialization overview. ([Wikipedia][6])
7. JSON‑LD specification and background. ([Wikipedia][7])
