"""Entity linking: token trie over labels and altLabels, fuzzy fallback, context disambiguation.

Pipeline for a piece of text:
  1. tokenise (keeping original case for each token);
  2. greedy longest-match scan against a token trie built from every surface form;
     short all-caps abbreviations only match case-sensitively, so "ZS" links but "is" does not;
  3. fuzzy fallback (edit-distance ratio) on capitalised spans that the trie did not cover, which
     catches typos such as "Mirrokinbi";
  4. when a span maps to several entities (colliding abbreviations), pick the candidate with the
     best combination of type cue from the question, graph coherence with the other linked
     entities, and a degree prior.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field

from rapidfuzz import fuzz, process
from rdflib import URIRef

from kgrag.kg import KnowledgeGraph
from kgrag.namespaces import KG

TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*")

# Words that signal the expected type of a nearby mention ("trials for X" -> X is a disease).
TYPE_CUES: dict[str, URIRef] = {
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
    "patients": KG.Disease,
    "for": KG.Disease,
    "with": KG.Disease,
    "pathway": KG.Pathway,
    "pathways": KG.Pathway,
    "trial": KG.ClinicalTrial,
    "trials": KG.ClinicalTrial,
    "study": KG.ClinicalTrial,
}


def normalise(token: str) -> str:
    t = token.lower()
    return t[:-2] if t.endswith("'s") else t


def is_abbreviation(form: str) -> bool:
    return form.isupper() and form.isalpha() and len(form) <= 4


@dataclass(frozen=True)
class Mention:
    start: int  # token offsets, end exclusive
    end: int
    text: str


@dataclass
class Link:
    mention: Mention
    entity: URIRef
    score: float  # 1.0 exact, fuzzy ratio / 100 otherwise
    method: str  # "exact" | "fuzzy"
    candidates: list[URIRef] = field(default_factory=list)


class _TrieNode:
    __slots__ = ("children", "entities")

    def __init__(self) -> None:
        self.children: dict[str, _TrieNode] = {}
        self.entities: set[URIRef] = set()


class EntityLinker:
    def __init__(self, kg: KnowledgeGraph, fuzzy_cutoff: float = 86.0) -> None:
        self.kg = kg
        self.fuzzy_cutoff = fuzzy_cutoff
        self._root = _TrieNode()
        self._abbrev: dict[str, set[URIRef]] = defaultdict(set)
        self._forms: dict[str, set[URIRef]] = defaultdict(set)
        for e in kg.entities():
            if kg.entity_type(e) == KG.Publication:
                continue
            for form in kg.surface_forms(e):
                if is_abbreviation(form):
                    self._abbrev[form].add(e)
                    continue
                self._insert(form, e)
                self._forms[form.lower()].add(e)
        self._form_keys = sorted(self._forms)
        self._degree = {e: sum(1 for _ in kg.neighbours(e)) for e in kg.entities()}

    def _insert(self, form: str, entity: URIRef) -> None:
        node = self._root
        for tok in TOKEN_RE.findall(form):
            node = node.children.setdefault(normalise(tok), _TrieNode())
        node.entities.add(entity)

    # ------------------------------------------------------------------ public

    def link(self, text: str) -> list[Link]:
        tokens = TOKEN_RE.findall(text)
        raw = self._exact(tokens)
        covered = {i for link in raw for i in range(link.mention.start, link.mention.end)}
        raw += self._fuzzy(tokens, covered)
        raw.sort(key=lambda link: link.mention.start)
        return self._disambiguate(raw, tokens)

    def link_restricted(self, text: str, candidates: set[URIRef], cutoff: float = 70.0) -> list[Link]:
        """Fuzzy-link capitalised tokens against a small candidate set (e.g. retrieved context).

        A lower cutoff is safe here because the candidate set is tiny: the retriever has
        already narrowed the search space, so a badly misspelled name can still be resolved.
        """
        forms = {
            f.lower(): e
            for e in sorted(candidates)
            for f in self.kg.surface_forms(e)
            if not is_abbreviation(f)
        }
        keys = sorted(forms)
        links = []
        for i, tok in enumerate(TOKEN_RE.findall(text)):
            if i == 0 or not tok[:1].isupper() or len(tok) < 5:
                continue
            hit = process.extractOne(tok.lower(), keys, scorer=fuzz.ratio, score_cutoff=cutoff)
            if hit:
                form, score, _ = hit
                links.append(
                    Link(Mention(i, i + 1, tok), forms[form], score / 100, "fuzzy-context", [forms[form]])
                )
        return links

    def entities(self, text: str) -> list[URIRef]:
        return list(dict.fromkeys(link.entity for link in self.link(text)))

    # ------------------------------------------------------------------ stages

    def _exact(self, tokens: list[str]) -> list[Link]:
        links, i = [], 0
        while i < len(tokens):
            node, best = self._root, None
            for j in range(i, len(tokens)):
                node = node.children.get(normalise(tokens[j]))
                if node is None:
                    break
                if node.entities:
                    best = (j + 1, node.entities)
            if best is None and tokens[i] in self._abbrev:
                best = (i + 1, self._abbrev[tokens[i]])
            if best:
                end, ents = best
                m = Mention(i, end, " ".join(tokens[i:end]))
                links.append(Link(m, URIRef(""), 1.0, "exact", sorted(ents)))
                i = end
            else:
                i += 1
        return links

    def _fuzzy(self, tokens: list[str], covered: set[int]) -> list[Link]:
        links: list[Link] = []
        taken = set(covered)
        # Longer spans first so "Zonaedne sarcoma" beats "sarcoma" alone.
        for n in (3, 2, 1):
            for i in range(len(tokens) - n + 1):
                span = range(i, i + n)
                if any(k in taken for k in span):
                    continue
                if not (tokens[i][:1].isupper() or any(c.isdigit() for c in tokens[i])):
                    continue
                text = " ".join(tokens[i : i + n])
                if len(text) < 5:
                    continue
                hit = process.extractOne(
                    text.lower(), self._form_keys, scorer=fuzz.ratio, score_cutoff=self.fuzzy_cutoff
                )
                if hit:
                    form, score, _ = hit
                    links.append(
                        Link(
                            Mention(i, i + n, text),
                            URIRef(""),
                            score / 100,
                            "fuzzy",
                            sorted(self._forms[form]),
                        )
                    )
                    taken.update(span)
        return links

    def _disambiguate(self, links: list[Link], tokens: list[str]) -> list[Link]:
        settled = {link.candidates[0] for link in links if len(link.candidates) == 1}
        for link in links:
            if len(link.candidates) == 1:
                link.entity = link.candidates[0]
                continue
            cue = self._type_cue(tokens, link.mention.start)
            link.entity = max(link.candidates, key=lambda c: self._context_score(c, cue, settled))
            settled.add(link.entity)
        return links

    def _type_cue(self, tokens: list[str], start: int) -> URIRef | None:
        for k in range(start - 1, max(-1, start - 4), -1):
            cue = TYPE_CUES.get(tokens[k].lower())
            if cue is not None:
                return cue
        return None

    def _context_score(self, cand: URIRef, cue: URIRef | None, context: set[URIRef]) -> tuple:
        type_ok = cue is not None and self.kg.entity_type(cand) == cue
        one_hop = {o for _, o, _ in self.kg.neighbours(cand)}
        two_hop = {o2 for o in one_hop for _, o2, _ in self.kg.neighbours(o)}
        coherence = 2 * len(one_hop & context) + len(two_hop & context)
        return (type_ok, coherence, self._degree.get(cand, 0), str(cand))
