"""Graph retrieval: link entities, expand a k-hop neighbourhood, rank facts with personalised PageRank.

Steps for a question:
  1. entity linking gives seed nodes; a seed that is a disease *group* is expanded through the
     subclass closure, so "trials for solid tumours" reaches trials of each tumour subtype;
  2. breadth-first expansion up to ``hops`` over domain facts, skipping hub-like detours;
  3. relation-type weights: each predicate has a base weight, multiplied when the predicate lies
     on the best schema path for the question (see ``analysis``);
  4. personalised PageRank over the weighted subgraph, restarting at the seeds;
  5. fact score = relation weight x (PPR(s) + PPR(o)), with a bonus for facts that lie on an
     instance path that follows the selected schema path from a seed. Those facts form the
     reasoning chain the reader needs, so they are ranked first.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import networkx as nx
from rdflib import URIRef

from kgrag.analysis import QuestionAnalysis, SchemaPath, analyse
from kgrag.kg import KnowledgeGraph, Triple
from kgrag.linking import EntityLinker
from kgrag.namespaces import KG, RDFS
from kgrag.retrieval.base import Fact, Provenance, RetrievalResult

DEFAULT_RELATION_WEIGHTS: dict[URIRef, float] = {
    KG.inhibits: 1.0,
    KG.associatedWith: 0.8,
    KG.participatesIn: 0.8,
    KG.studiedIn: 0.9,
    KG.investigates: 0.9,
}


@dataclass
class GraphRetrieverConfig:
    hops: int = 4
    alpha: float = 0.85
    cued_boost: float = 3.0
    path_bonus: float = 1.0
    max_frontier: int = 400
    relation_weights: dict[URIRef, float] = field(default_factory=lambda: dict(DEFAULT_RELATION_WEIGHTS))


def follow_schema_path(
    kg: KnowledgeGraph, starts: list[URIRef], path: SchemaPath, allowed: set[Triple] | None = None
) -> tuple[set[URIRef], list[Triple]]:
    """Walk instance edges along a schema path. Returns (end nodes, triples used).

    Only triples that lie on a complete start-to-end chain are returned, so dead-end branches
    do not pollute the reasoning context. ``allowed`` restricts the walk to a set of triples
    (the reader uses it to stay inside the retrieved context).
    """
    layers: list[dict[URIRef, list[tuple[URIRef, Triple]]]] = []
    frontier = set(starts)
    for step in path.steps:
        nxt: dict[URIRef, list[tuple[URIRef, Triple]]] = {}
        for node in sorted(frontier):
            if step.forward:
                pairs = [(o, (node, step.predicate, o)) for o in kg.graph.objects(node, step.predicate)]
            else:
                pairs = [(s, (s, step.predicate, node)) for s in kg.graph.subjects(step.predicate, node)]
            for other, triple in pairs:
                if allowed is not None and triple not in allowed:
                    continue
                nxt.setdefault(other, []).append((node, triple))
        layers.append(nxt)
        frontier = set(nxt)
    ends = set(frontier)
    # Backward pass keeps only edges that reach an end node.
    keep: list[Triple] = []
    alive = set(ends)
    for layer in reversed(layers):
        prev_alive = set()
        for node in alive:
            for parent, triple in layer.get(node, []):
                keep.append(triple)
                prev_alive.add(parent)
        alive = prev_alive
    return ends, sorted(set(keep))


def chain_order(
    kg: KnowledgeGraph, starts: list[URIRef], path: SchemaPath, chain: set[Triple], max_paths: int = 500
) -> list[Triple]:
    """Order chain triples path by path (depth-first), so any prefix holds complete chains.

    PageRank alone would rank all first-hop facts (closest to the seed) above all last-hop
    facts; with a small budget k the reader would then see many half chains and no answer.
    """
    ordered: dict[Triple, None] = {}
    n_paths = 0

    def step_edges(node: URIRef, depth: int) -> list[tuple[URIRef, Triple]]:
        st = path.steps[depth]
        if st.forward:
            cand = [(o, (node, st.predicate, o)) for o in kg.graph.objects(node, st.predicate)]
        else:
            cand = [(x, (x, st.predicate, node)) for x in kg.graph.subjects(st.predicate, node)]
        return sorted(c for c in cand if c[1] in chain)

    def dfs(node: URIRef, depth: int, acc: list[Triple]) -> None:
        nonlocal n_paths
        if n_paths >= max_paths:
            return
        if depth == len(path.steps):
            n_paths += 1
            ordered.update(dict.fromkeys(acc))
            return
        for nxt, triple in step_edges(node, depth):
            dfs(nxt, depth + 1, [*acc, triple])

    for start in starts:
        dfs(start, 0, [])
    ordered.update(dict.fromkeys(sorted(chain)))  # anything left over (path cap reached)
    return list(ordered)


def expand_seeds(kg: KnowledgeGraph, seeds: list[URIRef]) -> tuple[list[URIRef], list[Triple]]:
    """Ontology reasoning-lite: disease groups expand to all their subclasses."""
    out, sub_triples = list(seeds), []
    for s in seeds:
        if kg.is_disease_group(s):
            for sub in sorted(kg.subclasses(s)):
                out.append(sub)
                sub_triples.append((sub, RDFS.subClassOf, s))
    return list(dict.fromkeys(out)), sub_triples


class GraphRetriever:
    name = "graph"

    def __init__(
        self,
        kg: KnowledgeGraph,
        linker: EntityLinker | None = None,
        provenance: Provenance | None = None,
        config: GraphRetrieverConfig | None = None,
    ) -> None:
        self.kg = kg
        self.linker = linker or EntityLinker(kg)
        self.prov = provenance or Provenance(kg)
        self.cfg = config or GraphRetrieverConfig()

    def _weights(self, analysis: QuestionAnalysis) -> dict[URIRef, float]:
        w = dict(self.cfg.relation_weights)
        if analysis.best_path:
            for step in analysis.best_path.steps:
                w[step.predicate] = w.get(step.predicate, 1.0) * self.cfg.cued_boost
        return w

    def _expand(self, seeds: list[URIRef]) -> set[Triple]:
        facts: set[Triple] = set()
        depth = dict.fromkeys(seeds, 0)
        queue = deque(seeds)
        while queue:
            node = queue.popleft()
            if depth[node] >= self.cfg.hops:
                continue
            for p, other, forward in self.kg.neighbours(node):
                facts.add((node, p, other) if forward else (other, p, node))
                if other not in depth and len(depth) < self.cfg.max_frontier:
                    depth[other] = depth[node] + 1
                    queue.append(other)
        return facts

    def retrieve(self, question: str, k: int = 20) -> RetrievalResult:
        analysis = analyse(self.kg, self.linker, question)
        seeds, sub_triples = expand_seeds(self.kg, analysis.seeds)
        if not seeds:
            return RetrievalResult(self.name, [], notes={"reason": "no entities linked"})
        weights = self._weights(analysis)
        sub = self._expand(seeds)

        g = nx.Graph()
        g.add_nodes_from(seeds)
        for s, p, o in sub:
            w = weights.get(p, 0.5)
            if g.has_edge(s, o):
                g[s][o]["weight"] += w
            else:
                g.add_edge(s, o, weight=w)
        personalization = {n: (1.0 if n in seeds else 0.0) for n in g.nodes}
        ppr = nx.pagerank(g, alpha=self.cfg.alpha, personalization=personalization, weight="weight")

        chain: list[Triple] = []
        for seed, path in analysis.paths:
            start_nodes, _ = expand_seeds(self.kg, [seed])
            _, used = follow_schema_path(self.kg, start_nodes, path)
            chain += chain_order(self.kg, start_nodes, path, set(used))
        chain_rank = {t: i for i, t in enumerate(dict.fromkeys(chain))}

        scored: dict[Triple, float] = {}
        for t in sub | set(chain_rank):
            s, p, o = t
            base = weights.get(p, 0.5) * (ppr.get(s, 0.0) + ppr.get(o, 0.0))
            if t in chain_rank:  # chain facts first, in path order
                base = self.cfg.path_bonus + 1.0 / (1 + chain_rank[t])
            scored[t] = base
        ranked = sorted(scored.items(), key=lambda kv: (-kv[1], kv[0]))
        facts = [self.prov.make_fact(t, sc) for t, sc in ranked]
        schema_facts = [Fact(t, 1.0, self.kg.verbalise(t), ()) for t in sub_triples]
        notes = {"seeds": ", ".join(self.kg.label(s) for s in analysis.seeds)}
        if analysis.best_path:
            notes["schema_path"] = str(analysis.best_path)
        return RetrievalResult(self.name, facts[:k], seeds=seeds, notes=notes, schema_facts=schema_facts)
