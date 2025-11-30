# rdfknife usage guide

This document expands on the CLI usage and shows concrete workflows.

## 1. Basic commands

### 1.1 `info`

Show high‑level stats and namespaces:

```bash
rdfknife info -i data/wn-full.ttl
````

Output includes:

* Number of triples, subjects, predicates, objects
* Rough counts of classes and instances
* Namespace bindings (prefix → URI)

### 1.2 `search-literal`

Find triples whose object literal contains a substring:

```bash
rdfknife search-literal "road vehicle" -i data/wn-sample.ttl
rdfknife search-literal "vehicle" -i data/wn-sample.ttl --limit 10
rdfknife search-literal "Vehicle" -i data/wn-sample.ttl --case-sensitive
```

### 1.3 `search-uri`

Search for URIs containing a substring:

```bash
rdfknife search-uri "synset-car" -i data/wn-sample.ttl
rdfknife search-uri "wn20/schema" -i data/wn-sample.ttl
```

## 2. SPARQL

### 2.1 Short query inline

```bash
rdfknife sparql -i data/wn-sample.ttl -q \
"SELECT (COUNT(*) AS ?n) WHERE { ?s ?p ?o }"
```

### 2.2 Query from file

```bash
rdfknife sparql -i data/wn-sample.ttl -f queries/synsets_for_car.rq
```

### 2.3 Query from stdin

```bash
cat queries/synsets_for_car.rq | rdfknife sparql -i data/wn-sample.ttl
```

## 3. Editing

### 3.1 Add a triple

```bash
rdfknife add-triple -i data/ontology.ttl \
  --subject "http://example.org/Car" \
  --predicate "http://www.w3.org/2000/01/rdf-schema#label" \
  --object "Car" \
  --object-type literal \
  --lang en
```

### 3.2 Remove triples

Remove all triples whose object literal is `"Car"`:

```bash
rdfknife remove-triple -i data/ontology.ttl --object "Car"
```

Remove triples matching a specific subject and predicate:

```bash
rdfknife remove-triple -i data/ontology.ttl \
  --subject "http://example.org/Car" \
  --predicate "http://www.w3.org/2000/01/rdf-schema#label"
```

## 4. Saving data

### 4.1 Overwrite first input file

```bash
rdfknife add-triple -i data/ontology.ttl ...  # modify graph in-memory
rdfknife save -i data/ontology.ttl           # overwrites ontology.ttl in turtle
```

### 4.2 Save to new file and format

```bash
rdfknife save -i data/ontology.ttl \
  --output data/ontology.rdf \
  --format xml
```

Formats are rdflib serializers: `turtle`, `xml`, `nt`, `trig`, `json-ld`, etc. ([RDFLib][1])

## 5. WordNet workflows

Assumes a W3C WordNet RDF distribution (or OpenWN‑PT etc.) with schema matching
the WordNet RDF/OWL recommendation. ([W3C][2])

### 5.1 Lemma → synsets

```bash
rdfknife wn-synsets bank -i data/wn-full.ttl --lang en
```

Output per synset:

* Synset URI
* `synsetId`
* `rdfs:label`
* Gloss
* All words in the synset

### 5.2 Lemma → synonyms

```bash
rdfknife wn-synonyms car -i data/wn-full.ttl --lang en
```

This:

* Detects the WordNet schema (lexicalForm, word, containsWordSense)
* Resolves synsets containing the lemma
* Aggregates other words in those synsets as synonyms

Result is grouped by synonym; for each synonym you get the synset IDs in which
it co‑occurs with the original lemma.

## 6. OWL workflows

If you have an OWL 2 ontology serialized as RDF/XML or Turtle (following the
OWL 2 Mapping to RDF Graphs), you can quickly inspect its entities. ([Wikipedia][3])

```bash
rdfknife owl-entities -i data/ontology.owl
```

This prints tables for:

* `classes` (OWL/RDFS classes)
* `object_properties` (owl:ObjectProperty)
* `datatype_properties` (owl:DatatypeProperty)
* `annotation_properties` (owl:AnnotationProperty)
* `individuals` (any remaining typed resources)

## 7. REPL patterns

Start:

```bash
rdfknife repl -i data/wn-sample.ttl
```

Examples:

```text
rdfknife> info
rdfknife> search-literal vehicle
rdfknife> wn-synsets car
rdfknife> wn-synonyms car
rdfknife> sparql "SELECT ?w WHERE { ?w ?p ?o } LIMIT 5"
rdfknife> exit
```
