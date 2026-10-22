"""Common retrieval types.

Every retriever returns a ranked list of *facts* (instance triples with provenance), so all
four strategies are compared on the same unit and the same context budget ``k``.
Passage retrievers rank abstracts and then expand each abstract into the facts it states,
using the sentence-level provenance stored in the graph (reified kg:asserts statements).
That models a perfect information extractor on top of passage retrieval, which isolates the
quality of the retrieval step itself.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from functools import cached_property
from typing import Protocol

from rdflib import URIRef

from kgrag.kg import KnowledgeGraph, Triple
from kgrag.namespaces import KG, RDF


@dataclass(frozen=True)
class Fact:
    triple: Triple
    score: float
    text: str
    sources: tuple[URIRef, ...]  # publications that state this triple


@dataclass
class RetrievalResult:
    retriever: str
    facts: list[Fact]
    passages: list[tuple[URIRef, float]] = field(default_factory=list)
    seeds: list[URIRef] = field(default_factory=list)
    notes: dict[str, str] = field(default_factory=dict)
    # Ontology facts used for reasoning (e.g. subclass links); shown, but outside the budget k.
    schema_facts: list[Fact] = field(default_factory=list)

    @property
    def triples(self) -> list[Triple]:
        return [f.triple for f in self.facts]


class Retriever(Protocol):
    name: str

    def retrieve(self, question: str, k: int = 20) -> RetrievalResult: ...


class Provenance:
    """Index of which publication asserts which triple (built from rdf:Statement nodes)."""

    def __init__(self, kg: KnowledgeGraph) -> None:
        self.kg = kg

    @cached_property
    def by_publication(self) -> dict[URIRef, list[Triple]]:
        out: dict[URIRef, list[Triple]] = defaultdict(list)
        g = self.kg.graph
        for pub, st in sorted(g.subject_objects(KG.asserts)):
            s, p, o = g.value(st, RDF.subject), g.value(st, RDF.predicate), g.value(st, RDF.object)
            out[pub].append((s, p, o))
        return out

    @cached_property
    def by_triple(self) -> dict[Triple, tuple[URIRef, ...]]:
        out: dict[Triple, list[URIRef]] = defaultdict(list)
        for pub, triples in self.by_publication.items():
            for t in triples:
                out[t].append(pub)
        return {t: tuple(sorted(p)) for t, p in out.items()}

    def sources(self, triple: Triple) -> tuple[URIRef, ...]:
        return self.by_triple.get(triple, ())

    def make_fact(self, triple: Triple, score: float) -> Fact:
        return Fact(triple, score, self.kg.verbalise(triple), self.sources(triple))


def passages_to_facts(
    prov: Provenance, ranked: list[tuple[URIRef, float]], k: int, retriever: str
) -> RetrievalResult:
    facts: list[Fact] = []
    seen: set[Triple] = set()
    used: list[tuple[URIRef, float]] = []
    for pub, score in ranked:
        if len(facts) >= k:
            break
        used.append((pub, score))
        for t in prov.by_publication.get(pub, []):
            if t not in seen and len(facts) < k:
                seen.add(t)
                facts.append(prov.make_fact(t, score))
    return RetrievalResult(retriever, facts, passages=used)
