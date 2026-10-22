"""Knowledge graph access layer: labels, types, reasoning-lite and SPARQL.

Reasoning is deliberately limited to what retrieval needs:
  * rdfs:subClassOf transitive closure (computed once, cached as Python dicts);
  * type resolution through the closure (a disease subclass is a kg:Disease);
  * synonym expansion via skos:altLabel.
A full OWL reasoner would add little here and would make index builds slower.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

from rdflib import Graph, Literal, URIRef

from kgrag.namespaces import ENTITY_TYPES, FACT_PREDICATES, KG, OWL, RDF, RDFS, SKOS

Triple = tuple[URIRef, URIRef, URIRef]


@dataclass(frozen=True)
class Passage:
    iri: URIRef
    label: str
    text: str
    mentions: frozenset[URIRef]


class KnowledgeGraph:
    def __init__(self, graph: Graph) -> None:
        self.graph = graph

    # ------------------------------------------------------------------ io

    @classmethod
    def load(cls, path: str | Path) -> KnowledgeGraph:
        g = Graph()
        g.parse(str(path), format="turtle")
        return cls(g)

    @classmethod
    def synthetic(cls, seed: int = 7) -> KnowledgeGraph:
        from kgrag.generator import GeneratorConfig, generate_graph

        return cls(generate_graph(GeneratorConfig(seed=seed)))

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.graph.serialize(destination=str(path), format="turtle")

    def _query(self, query: str, bindings: dict):
        init_ns = {"kg": KG, "rdfs": RDFS, "skos": SKOS, "rdf": RDF, "owl": OWL}
        return self.graph.query(query, initNs=init_ns, initBindings=bindings or None)

    def sparql(self, query: str, **bindings: URIRef | Literal) -> list[tuple]:
        """Run a SELECT query; ``bindings`` pre-bind variables (e.g. ``x=iri``)."""
        return [tuple(r) for r in self._query(query, bindings)]

    def ask(self, query: str, **bindings: URIRef | Literal) -> bool:
        return bool(self._query(query, bindings).askAnswer)

    # ------------------------------------------------------------------ ontology closure

    @cached_property
    def _super_direct(self) -> dict[URIRef, set[URIRef]]:
        out: dict[URIRef, set[URIRef]] = defaultdict(set)
        for s, o in self.graph.subject_objects(RDFS.subClassOf):
            if isinstance(s, URIRef) and isinstance(o, URIRef):
                out[s].add(o)
        return out

    @cached_property
    def _sub_direct(self) -> dict[URIRef, set[URIRef]]:
        out: dict[URIRef, set[URIRef]] = defaultdict(set)
        for s, supers in self._super_direct.items():
            for o in supers:
                out[o].add(s)
        return out

    @staticmethod
    def _closure(start: URIRef, edges: dict[URIRef, set[URIRef]]) -> set[URIRef]:
        seen: set[URIRef] = set()
        stack = list(edges.get(start, ()))
        while stack:
            n = stack.pop()
            if n not in seen:
                seen.add(n)
                stack.extend(edges.get(n, ()))
        return seen

    def superclasses(self, cls: URIRef, reflexive: bool = False) -> set[URIRef]:
        out = self._closure(cls, self._super_direct)
        return out | {cls} if reflexive else out

    def subclasses(self, cls: URIRef, reflexive: bool = False) -> set[URIRef]:
        out = self._closure(cls, self._sub_direct)
        return out | {cls} if reflexive else out

    def is_subclass_of(self, sub: URIRef, sup: URIRef) -> bool:
        return sub == sup or sup in self.superclasses(sub)

    def materialise_subclass_closure(self) -> int:
        """Add inferred rdfs:subClassOf triples; returns the number of new triples."""
        added = 0
        for cls in list(self._super_direct):
            for sup in self.superclasses(cls):
                if (cls, RDFS.subClassOf, sup) not in self.graph:
                    self.graph.add((cls, RDFS.subClassOf, sup))
                    added += 1
        self.__dict__.pop("_super_direct", None)
        self.__dict__.pop("_sub_direct", None)
        return added

    # ------------------------------------------------------------------ entities

    @cached_property
    def _types(self) -> dict[URIRef, URIRef]:
        types: dict[URIRef, URIRef] = {}
        for etype in ENTITY_TYPES:
            for s in self.graph.subjects(RDF.type, etype):
                types.setdefault(s, etype)
        # Disease subclasses are classes, typed through the subclass closure.
        for d in self.subclasses(KG.Disease):
            types[d] = KG.Disease
        return types

    def entity_type(self, iri: URIRef) -> URIRef | None:
        return self._types.get(iri)

    def entities(self, etype: URIRef | None = None) -> list[URIRef]:
        items = [e for e, t in self._types.items() if etype is None or t == etype]
        return sorted(items)

    @cached_property
    def _labels(self) -> dict[URIRef, str]:
        return {s: str(o) for s, o in self.graph.subject_objects(RDFS.label) if isinstance(s, URIRef)}

    @cached_property
    def _alt(self) -> dict[URIRef, list[str]]:
        out: dict[URIRef, list[str]] = defaultdict(list)
        for s, o in self.graph.subject_objects(SKOS.altLabel):
            out[s].append(str(o))
        return {k: sorted(v) for k, v in out.items()}

    def label(self, iri: URIRef) -> str:
        return self._labels.get(iri) or str(iri).rsplit("/", 1)[-1]

    def alt_labels(self, iri: URIRef) -> list[str]:
        return self._alt.get(iri, [])

    def surface_forms(self, iri: URIRef) -> list[str]:
        """Synonym expansion: preferred label first, then skos:altLabel values."""
        return [self.label(iri), *self.alt_labels(iri)]

    def is_disease_group(self, iri: URIRef) -> bool:
        return self.entity_type(iri) == KG.Disease and bool(self._sub_direct.get(iri))

    # ------------------------------------------------------------------ facts

    @cached_property
    def facts(self) -> list[Triple]:
        out = []
        for p in FACT_PREDICATES:
            if p == KG.mentions:
                continue
            out.extend((s, p, o) for s, o in self.graph.subject_objects(p))
        return sorted(out)

    def neighbours(self, iri: URIRef) -> Iterator[tuple[URIRef, URIRef, bool]]:
        """Yield (predicate, other node, forward?) over domain facts (no mentions)."""
        for p, o in self.graph.predicate_objects(iri):
            if p in FACT_PREDICATES and p != KG.mentions and isinstance(o, URIRef):
                yield p, o, True
        for s, p in self.graph.subject_predicates(iri):
            if p in FACT_PREDICATES and p != KG.mentions:
                yield p, s, False

    @cached_property
    def passages(self) -> list[Passage]:
        out = []
        for pub in sorted(self.graph.subjects(RDF.type, KG.Publication)):
            text = self.graph.value(pub, KG.abstractText)
            mentions = frozenset(self.graph.objects(pub, KG.mentions))
            out.append(Passage(pub, self.label(pub), str(text), mentions))
        return out

    def predicate_label(self, p: URIRef) -> str:
        return self.label(p)

    def verbalise(self, triple: Triple) -> str:
        s, p, o = triple
        return f"{self.label(s)} {self.predicate_label(p)} {self.label(o)}."

    def has_fact(self, triple: Triple) -> bool:
        return triple in self.graph

    def domain_range(self, p: URIRef) -> tuple[URIRef | None, URIRef | None]:
        return self.graph.value(p, RDFS.domain), self.graph.value(p, RDFS.range)

    def type_matches(self, iri: URIRef, cls: URIRef | None) -> bool:
        """Instance-of check through the class hierarchy (Target is a Protein, etc.)."""
        if cls is None:
            return True
        t = self.entity_type(iri)
        return t is not None and (t == cls or self.is_subclass_of(t, cls))

    def stats(self) -> dict[str, int]:
        counts = {self.label(t): len(self.entities(t)) for t in ENTITY_TYPES}
        counts["facts"] = len(self.facts)
        counts["triples"] = len(self.graph)
        return counts


def unique(items: Iterable[URIRef]) -> list[URIRef]:
    return sorted(set(items))
