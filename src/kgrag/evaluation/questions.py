"""Deterministic question set with gold answers computed by SPARQL.

Gold answers are produced by SPARQL property-path queries (including ``rdfs:subClassOf*``),
an implementation independent of the retrievers and of the stub reader. Each question also
records its gold schema path, used to extract the gold evidence chain for recall@k.

Categories:
  single-hop  one relation, answerable from a single abstract
  two-hop     two relations, usually spread over two abstracts
  multi-hop   three or four relations (drug, trial, disease, target, pathway chains)
  hierarchy   needs the disease subclass closure (a question about a disease group)
  misspelled  single-hop question whose entity name is corrupted beyond the linker's
              global fuzzy threshold (tests what happens when symbolic linking fails)
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from importlib import resources
from pathlib import Path

from rdflib import URIRef

from kgrag.kg import KnowledgeGraph
from kgrag.namespaces import KG

PREFIXES = """PREFIX kg: <https://example.org/kgrag/ontology#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
"""

# name -> (category, templates, path, SPARQL body with ?x as the question entity and ?a as answer)
TEMPLATES: dict[str, tuple[str, URIRef, tuple[str, ...], tuple[str, ...], str]] = {
    "drug_targets": (
        "single-hop",
        KG.Drug,
        ("Which targets does {x} inhibit?", "Which proteins are inhibited by {x}?"),
        ("inhibits",),
        "?x kg:inhibits ?a .",
    ),
    "trial_drugs": (
        "single-hop",
        KG.ClinicalTrial,
        ("Which drugs were studied in the {x}?", "Which compounds were tested in {x}?"),
        ("^studiedIn",),
        "?a kg:studiedIn ?x .",
    ),
    "target_pathways": (
        "single-hop",
        KG.Target,
        ("Which pathways does {x} participate in?",),
        ("participatesIn",),
        "?x kg:participatesIn ?a .",
    ),
    "drug_pathways": (
        "two-hop",
        KG.Drug,
        ("Which pathways are targeted by {x}?", "Which pathway does {x} act on through its targets?"),
        ("inhibits", "participatesIn"),
        "?x kg:inhibits ?t . ?t kg:participatesIn ?a .",
    ),
    "drug_trial_diseases": (
        "two-hop",
        KG.Drug,
        ("Which diseases were investigated in trials of {x}?",),
        ("studiedIn", "investigates"),
        "?x kg:studiedIn ?tr . ?tr kg:investigates ?a .",
    ),
    "trial_targets": (
        "two-hop",
        KG.ClinicalTrial,
        ("Which targets are inhibited by drugs tested in the {x}?",),
        ("^studiedIn", "inhibits"),
        "?d kg:studiedIn ?x . ?d kg:inhibits ?a .",
    ),
    "disease_trial_targets": (
        "multi-hop",
        KG.Disease,
        ("Which targets are inhibited by drugs studied in trials for {x}?",),
        ("^investigates", "^studiedIn", "inhibits"),
        "?tr kg:investigates ?x . ?d kg:studiedIn ?tr . ?d kg:inhibits ?a .",
    ),
    "disease_trial_pathways": (
        "multi-hop",
        KG.Disease,
        (
            "Which pathway is targeted by drugs studied in trials for {x}?",
            "Which pathways are targeted by drugs tested in trials for patients with {x}?",
        ),
        ("^investigates", "^studiedIn", "inhibits", "participatesIn"),
        "?tr kg:investigates ?x . ?d kg:studiedIn ?tr . ?d kg:inhibits ?t . ?t kg:participatesIn ?a .",
    ),
    "pathway_drug_diseases": (
        "multi-hop",
        KG.Pathway,
        ("Which diseases were investigated in trials of drugs that inhibit targets in the {x}?",),
        ("^participatesIn", "^inhibits", "studiedIn", "investigates"),
        "?t kg:participatesIn ?x . ?d kg:inhibits ?t . ?d kg:studiedIn ?tr . ?tr kg:investigates ?a .",
    ),
    "group_trial_drugs": (
        "hierarchy",
        KG.Disease,
        ("Which drugs were studied in trials for {x}?",),
        ("^investigates", "^studiedIn"),
        "?d0 rdfs:subClassOf* ?x . ?tr kg:investigates ?d0 . ?a kg:studiedIn ?tr .",
    ),
    "group_trials": (
        "hierarchy",
        KG.Disease,
        ("Which trials investigate a form of {x}?",),
        ("^investigates",),
        "?d0 rdfs:subClassOf* ?x . ?a kg:investigates ?d0 .",
    ),
}

NOISY_QUOTA = 5

QUOTA = {
    "drug_targets": 6,
    "trial_drugs": 5,
    "target_pathways": 5,
    "drug_pathways": 6,
    "drug_trial_diseases": 5,
    "trial_targets": 4,
    "disease_trial_targets": 7,
    "disease_trial_pathways": 8,
    "pathway_drug_diseases": 4,
    "group_trial_drugs": 6,
    "group_trials": 5,
}
MAX_ANSWERS = 8
MAX_EVIDENCE = 20  # equal to the default budget k, so every question is answerable in budget


@dataclass
class Question:
    id: str
    category: str
    template: str
    question: str
    entity: str  # IRI of the question entity
    path: list[str]
    gold: list[str]  # gold answer labels
    gold_iris: list[str]


def gold_answers(kg: KnowledgeGraph, body: str, x: URIRef) -> list[URIRef]:
    q = PREFIXES + f"SELECT DISTINCT ?a WHERE {{ {body} }}"
    return sorted(r[0] for r in kg.sparql(q, x=x))


def _surface(kg: KnowledgeGraph, rng: random.Random, x: URIRef) -> str:
    """Mostly the preferred label, sometimes a synonym (code, symbol, eponym form)."""
    forms = kg.surface_forms(x)
    return forms[0] if rng.random() < 0.7 or len(forms) == 1 else rng.choice(forms[1:])


def corrupt(rng: random.Random, word: str) -> str:
    """Swap two adjacent characters and drop one: typical of hurried typing or OCR."""
    c = list(word)
    i = rng.randrange(1, len(c) - 2)
    c[i], c[i + 1] = c[i + 1], c[i]
    del c[rng.randrange(1, len(c) - 1)]
    return "".join(c)


def generate_questions(kg: KnowledgeGraph, seed: int = 11) -> list[Question]:
    from kgrag.analysis import Step
    from kgrag.evaluation.metrics import evidence_chain
    from kgrag.linking import EntityLinker

    rng = random.Random(seed)
    out: list[Question] = []
    for name, (category, etype, templates, path, body) in TEMPLATES.items():
        pool = kg.entities(etype)
        if name.startswith("group_"):
            pool = [d for d in pool if kg.is_disease_group(d)]
        elif etype == KG.Disease:
            pool = [d for d in pool if not kg.is_disease_group(d)]
        rng.shuffle(pool)
        made = 0
        for x in pool:
            if made >= QUOTA[name]:
                break
            gold = gold_answers(kg, body, x)
            if not gold or len(gold) > MAX_ANSWERS:
                continue
            steps = tuple(Step(KG[p.lstrip("^")], not p.startswith("^")) for p in path)
            if len(evidence_chain(kg, x, steps)) > MAX_EVIDENCE:
                continue
            text = rng.choice(templates).format(x=_surface(kg, rng, x))
            out.append(
                Question(
                    id=f"q{len(out) + 1:02d}",
                    category=category,
                    template=name,
                    question=text,
                    entity=str(x),
                    path=list(path),
                    gold=sorted(kg.label(g) for g in gold),
                    gold_iris=[str(g) for g in gold],
                )
            )
            made += 1
    linker = EntityLinker(kg)
    drugs = kg.entities(KG.Drug)
    rng.shuffle(drugs)
    made = 0
    for x in drugs:
        if made >= NOISY_QUOTA:
            break
        text = f"Which targets does {corrupt(rng, kg.label(x))} inhibit?"
        if linker.link(text):
            continue  # still linkable: not a hard case
        gold = gold_answers(kg, TEMPLATES["drug_targets"][4], x)
        out.append(
            Question(
                id=f"q{len(out) + 1:02d}",
                category="misspelled",
                template="noisy_drug_targets",
                question=text,
                entity=str(x),
                path=["inhibits"],
                gold=sorted(kg.label(g) for g in gold),
                gold_iris=[str(g) for g in gold],
            )
        )
        made += 1
    return out


def save_questions(questions: list[Question], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps([asdict(q) for q in questions], indent=2) + "\n")


def load_questions(path: str | Path | None = None) -> list[Question]:
    if path is None:
        text = resources.files("kgrag.data").joinpath("questions.json").read_text()
    else:
        text = Path(path).read_text()
    return [Question(**d) for d in json.loads(text)]
