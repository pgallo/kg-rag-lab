"""Question analysis shared by the graph retriever and the stub reader.

Two symbolic signals are extracted from a question:
  * the expected answer type, from the wh-phrase ("Which pathways ..." -> kg:Pathway);
  * a ranked list of *schema paths*: sequences of ontology relations that lead from the
    type of a linked question entity to the answer type, found by search over the
    rdfs:domain / rdfs:range declarations, and scored by how many of their relations are
    cued by words in the question.

For "Which pathway is targeted by drugs studied in trials for X?" the schema offers
Disease <-associatedWith- Target -participatesIn-> Pathway (2 hops) and
Disease <-investigates- Trial <-studiedIn- Drug -inhibits-> Target -participatesIn-> Pathway
(4 hops). The cue words "targeted", "studied", "trials" select the second, which is the
reading the question actually asks for.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from rdflib import URIRef

from kgrag.kg import KnowledgeGraph
from kgrag.linking import TOKEN_RE, Link
from kgrag.namespaces import KG

ANSWER_NOUNS: dict[str, URIRef] = {
    "drug": KG.Drug,
    "drugs": KG.Drug,
    "compound": KG.Drug,
    "compounds": KG.Drug,
    "target": KG.Target,
    "targets": KG.Target,
    "protein": KG.Target,
    "proteins": KG.Target,
    "disease": KG.Disease,
    "diseases": KG.Disease,
    "condition": KG.Disease,
    "conditions": KG.Disease,
    "indication": KG.Disease,
    "indications": KG.Disease,
    "pathway": KG.Pathway,
    "pathways": KG.Pathway,
    "trial": KG.ClinicalTrial,
    "trials": KG.ClinicalTrial,
    "study": KG.ClinicalTrial,
    "studies": KG.ClinicalTrial,
}

PREDICATE_CUES: dict[URIRef, frozenset[str]] = {
    KG.inhibits: frozenset(
        "inhibit inhibits inhibited inhibitor inhibitors target targets targeted targeting act acts "
        "block blocks blocked".split()
    ),
    KG.associatedWith: frozenset("associated association linked implicated".split()),
    KG.participatesIn: frozenset("pathway pathways participate participates component cascade".split()),
    KG.studiedIn: frozenset("studied tested evaluated trial trials".split()),
    KG.investigates: frozenset("trial trials investigate investigates investigated".split()),
}
UNCUED_PENALTY = 0.5


@dataclass(frozen=True)
class Step:
    predicate: URIRef
    forward: bool  # True: subject -> object along the predicate

    def __str__(self) -> str:
        name = str(self.predicate).rsplit("#", 1)[-1]
        return name if self.forward else f"^{name}"


@dataclass(frozen=True)
class SchemaPath:
    start: URIRef
    end: URIRef
    steps: tuple[Step, ...]
    score: float

    def __str__(self) -> str:
        return " / ".join(str(s) for s in self.steps)


def answer_type(question: str) -> URIRef | None:
    q = question.lower()
    m = re.search(r"\b(?:which|what)\s+(?:\w+\s+)?(\w+)", q)
    if m:
        for word in m.group(0).split()[1:]:
            if word in ANSWER_NOUNS:
                return ANSWER_NOUNS[word]
    return None


def cue_words(question: str, links: list[Link]) -> set[str]:
    """Lower-cased question tokens outside entity mentions (labels like 'X pathway' are not cues)."""
    tokens = TOKEN_RE.findall(question)
    inside = {i for link in links for i in range(link.mention.start, link.mention.end)}
    return {t.lower() for i, t in enumerate(tokens) if i not in inside}


def schema_paths(kg: KnowledgeGraph, start: URIRef, end: URIRef, max_len: int = 4) -> list[tuple[Step, ...]]:
    """All simple relation paths between two classes, from rdfs:domain / rdfs:range."""
    edges: list[tuple[URIRef, Step, URIRef]] = []
    for p in PREDICATE_CUES:
        dom, rng = kg.domain_range(p)
        if dom is None or rng is None:
            continue
        edges.append((dom, Step(p, True), rng))
        edges.append((rng, Step(p, False), dom))

    def matches(node_cls: URIRef, decl: URIRef) -> bool:
        return kg.is_subclass_of(node_cls, decl)

    out: list[tuple[Step, ...]] = []

    def walk(cls: URIRef, path: list[Step], visited: list[URIRef]) -> None:
        if path and cls == end:
            out.append(tuple(path))
            return
        if len(path) == max_len:
            return
        for dom, step, rng in edges:
            if matches(cls, dom) and rng not in visited:
                # Ranges are declared on the most specific class used for instances.
                nxt = KG.Target if rng == KG.Protein else rng
                walk(nxt, [*path, step], [*visited, nxt])

    walk(start, [], [start])
    return out


def rank_schema_paths(
    kg: KnowledgeGraph, start: URIRef, end: URIRef, cues: set[str], max_len: int = 4
) -> list[SchemaPath]:
    ranked = []
    for steps in schema_paths(kg, start, end, max_len):
        cued = sum(1 for s in steps if PREDICATE_CUES[s.predicate] & cues)
        score = cued - UNCUED_PENALTY * (len(steps) - cued)
        ranked.append(SchemaPath(start, end, steps, score))
    ranked.sort(key=lambda sp: (-sp.score, len(sp.steps), str(sp)))
    return ranked


@dataclass
class QuestionAnalysis:
    links: list[Link]
    seeds: list[URIRef]
    answer_type: URIRef | None
    paths: list[tuple[URIRef, SchemaPath]]  # (seed, best schema path from that seed)

    @property
    def best_path(self) -> SchemaPath | None:
        return self.paths[0][1] if self.paths else None


def analyse(kg: KnowledgeGraph, linker, question: str, links: list[Link] | None = None) -> QuestionAnalysis:
    links = linker.link(question) if links is None else links
    seeds = list(dict.fromkeys(link.entity for link in links))
    atype = answer_type(question)
    cues = cue_words(question, links)
    paths = []
    if atype is not None:
        for seed in seeds:
            stype = kg.entity_type(seed)
            if stype is None or stype == atype:
                continue
            ranked = rank_schema_paths(kg, stype, atype, cues)
            if ranked:
                paths.append((seed, ranked[0]))
    return QuestionAnalysis(links, seeds, atype, paths)
