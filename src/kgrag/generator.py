"""Deterministic generator for the synthetic knowledge graph and abstract corpus.

Everything is driven by one seeded ``random.Random``; no set iteration or hash order
leaks into the output, so the same seed always yields byte-identical Turtle.

Biological consistency rules enforced here:
  * every drug inhibits 1-2 targets, every target is inhibited by at least one drug;
  * every target participates in 1-2 pathways and is associated with 1-3 leaf diseases;
  * a trial of a drug investigates a disease associated with one of that drug's targets
    (or the family-level parent of such a disease, i.e. a "basket" trial).

The abstracts are deliberately split by topic: drug-centric abstracts state mechanism
and trial facts, target-centric abstracts state pathway and disease facts. A multi-hop
question therefore needs evidence from several abstracts, which is the property that
separates graph retrieval from passage retrieval in the ablation.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from importlib import resources

from rdflib import Graph, Literal, URIRef

from kgrag.namespaces import ID, KG, OWL, RDF, RDFS, SKOS, XSD

ONSETS = (
    "Vel",
    "Qar",
    "Zon",
    "Thy",
    "Brev",
    "Ost",
    "Kael",
    "Mirr",
    "Dras",
    "Yul",
    "Pell",
    "Sarv",
    "Quen",
    "Xal",
    "Ivr",
    "Corv",
    "Halv",
    "Rull",
    "Fenn",
    "Jor",
    "Wex",
    "Ombr",
    "Tesk",
    "Vorn",
    "Zir",
    "Plo",
    "Hesk",
    "Nuv",
)
NUCLEI = ("o", "a", "e", "i", "u", "y", "ae")
CODAS = ("r", "l", "n", "x", "v", "th", "m", "s", "d", "k")
DRUG_STEMS = ("inib", "umab", "astat", "idone", "orant")
TARGET_FAMILIES = ("kinase", "receptor", "protease", "phosphatase", "transporter")

# category -> [(family, [leaf nouns])]; leaf diseases get invented eponyms.
DISEASE_TREE: dict[str, list[tuple[str, list[str]]]] = {
    "Oncological disorder": [
        ("Solid tumour", ["carcinoma", "sarcoma"]),
        ("Haematological malignancy", ["leukaemia", "lymphoma"]),
    ],
    "Inflammatory disorder": [
        ("Autoimmune arthritis", ["arthritis"]),
        ("Inflammatory bowel disorder", ["colitis", "enteritis"]),
    ],
    "Neurodegenerative disorder": [
        ("Movement disorder", ["palsy", "ataxia"]),
        ("Cognitive decline disorder", ["dementia"]),
    ],
    "Metabolic disorder": [
        ("Lipid storage disorder", ["lipidosis"]),
        ("Glucose regulation disorder", ["glycaemia"]),
    ],
    "Cardiovascular disorder": [
        ("Cardiomyopathy", ["cardiomyopathy"]),
        ("Vascular inflammation", ["vasculitis", "arteritis"]),
    ],
    "Fibrotic disorder": [
        ("Pulmonary fibrosis", ["lung fibrosis"]),
        ("Hepatic fibrosis", ["liver fibrosis"]),
    ],
}
GROUP_SYNONYMS = {"disorder": "disease", "tumour": "tumor", "haematological": "hematological"}
LEAVES_PER_FAMILY = 3


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


@dataclass
class GeneratorConfig:
    seed: int = 7
    n_drugs: int = 90
    n_targets: int = 60
    n_pathways: int = 24
    typo_rate: float = 0.04


@dataclass
class _Entity:
    iri: URIRef
    label: str
    alt: list[str] = field(default_factory=list)


class SyntheticKGGenerator:
    def __init__(self, config: GeneratorConfig | None = None) -> None:
        self.cfg = config or GeneratorConfig()
        self.rng = random.Random(self.cfg.seed)
        self._used_words: set[str] = set()
        self.g = Graph()
        self.g.bind("kg", KG)
        self.g.bind("id", ID)
        self.g.bind("skos", SKOS)
        self.g.bind("owl", OWL)

    # ------------------------------------------------------------------ names

    def _word(self, min_len: int = 5) -> str:
        for _ in range(10_000):
            w = self.rng.choice(ONSETS) + self.rng.choice(NUCLEI) + self.rng.choice(CODAS)
            if self.rng.random() < 0.5:
                w += self.rng.choice(("ane", "el", "ix", "or", "en", "is"))
            if len(w) >= min_len and w.lower() not in self._used_words:
                self._used_words.add(w.lower())
                return w
        raise RuntimeError("name space exhausted")

    def _add(self, ent: _Entity, rdf_type: URIRef) -> _Entity:
        self.g.add((ent.iri, RDF.type, rdf_type))
        self.g.add((ent.iri, RDFS.label, Literal(ent.label, lang="en")))
        for a in ent.alt:
            self.g.add((ent.iri, SKOS.altLabel, Literal(a, lang="en")))
        return ent

    # ------------------------------------------------------------------ entities

    def _diseases(self) -> tuple[list[_Entity], dict[URIRef, URIRef]]:
        leaves: list[_Entity] = []
        parent: dict[URIRef, URIRef] = {}
        for cat_label, families in DISEASE_TREE.items():
            cat = self._disease_class(cat_label, KG.Disease)
            for fam_label, nouns in families:
                fam = self._disease_class(fam_label, cat.iri)
                parent[fam.iri] = cat.iri
                for i in range(LEAVES_PER_FAMILY):
                    noun = nouns[i % len(nouns)]
                    epo = self._word()
                    label = f"{epo} {noun}"
                    abbrev = (epo[0] + "".join(p[0] for p in noun.split())).upper()
                    ent = self._disease_class(label, fam.iri, [f"{epo}'s {noun}", abbrev])
                    parent[ent.iri] = fam.iri
                    leaves.append(ent)
        return leaves, parent

    def _disease_class(self, label: str, sup: URIRef, alt: list[str] | None = None) -> _Entity:
        if alt is None:
            words = label.split()
            alt_words = [GROUP_SYNONYMS.get(w.lower(), w.lower()) for w in words]
            alt = [" ".join(alt_words).capitalize()] if alt_words != [w.lower() for w in words] else []
        ent = _Entity(ID[f"disease/{slug(label)}"], label, alt)
        self._add(ent, OWL.Class)
        self.g.add((ent.iri, RDFS.subClassOf, sup))
        return ent

    def _targets(self) -> list[_Entity]:
        out, symbols = [], set()
        for i in range(self.cfg.n_targets):
            epo = self._word()
            family = TARGET_FAMILIES[i % len(TARGET_FAMILIES)]
            num = 1 + self.rng.randrange(9)
            consonants = [c for c in epo.upper() if c not in "AEIOUY"]
            sym = f"{consonants[0]}{consonants[-1]}{family[0].upper()}{num}"
            while sym in symbols:
                num += 1
                sym = f"{consonants[0]}{consonants[-1]}{family[0].upper()}{num}"
            symbols.add(sym)
            label = f"{epo} {family} {num}"
            ent = _Entity(ID[f"target/{slug(sym)}"], label, [sym, f"{sym} protein"])
            out.append(self._add(ent, KG.Target))
        return out

    def _pathways(self) -> list[_Entity]:
        out = []
        for _ in range(self.cfg.n_pathways):
            epo = self._word()
            label = f"{epo} signalling pathway"
            ent = _Entity(ID[f"pathway/{slug(epo)}"], label, [f"{epo} cascade", f"{epo} axis"])
            out.append(self._add(ent, KG.Pathway))
        return out

    def _drugs(self) -> list[_Entity]:
        out = []
        for i in range(self.cfg.n_drugs):
            stem = DRUG_STEMS[i % len(DRUG_STEMS)]
            name = self._word(min_len=4) + stem
            code = f"LAB-{1000 + 17 * i}"
            ent = _Entity(ID[f"drug/{slug(name)}"], name, [code])
            out.append(self._add(ent, KG.Drug))
        return out

    def _trial(self, n: int, phase: int) -> _Entity:
        acr = self._word().upper()
        tid = f"SYN-{2000 + n}"
        ent = _Entity(ID[f"trial/{slug(tid)}"], acr, [f"{acr} trial", f"{acr} study", tid])
        self._add(ent, KG.ClinicalTrial)
        self.g.add((ent.iri, KG.phase, Literal(phase, datatype=XSD.integer)))
        return ent

    # ------------------------------------------------------------------ build

    def generate(self) -> Graph:
        rng = self.rng
        leaves, parent = self._diseases()
        targets, pathways, drugs = self._targets(), self._pathways(), self._drugs()

        # target -> pathways / diseases (every pathway and leaf covered at least once)
        t_paths: dict[URIRef, list[_Entity]] = {}
        t_dis: dict[URIRef, list[_Entity]] = {}
        for i, t in enumerate(targets):
            ps = [pathways[i % len(pathways)]]
            if rng.random() < 0.45:
                ps.append(rng.choice(pathways))
            ds = [leaves[i % len(leaves)], *rng.sample(leaves, rng.choice((0, 1, 2)))]
            t_paths[t.iri] = _dedup(ps)
            t_dis[t.iri] = _dedup(ds)
            for p in t_paths[t.iri]:
                self.g.add((t.iri, KG.participatesIn, p.iri))
            for d in t_dis[t.iri]:
                self.g.add((t.iri, KG.associatedWith, d.iri))

        # drug -> targets (round robin guarantees coverage)
        d_tgts: dict[URIRef, list[_Entity]] = {}
        for i, d in enumerate(drugs):
            ts = [targets[i % len(targets)]]
            if rng.random() < 0.3:
                ts.append(rng.choice(targets))
            d_tgts[d.iri] = _dedup(ts)
            for t in d_tgts[d.iri]:
                self.g.add((d.iri, KG.inhibits, t.iri))

        # trials: consistent with mechanism; ~15% are two-drug combination trials
        d_trials: dict[URIRef, list[tuple[_Entity, _Entity]]] = {d.iri: [] for d in drugs}
        n_trials = 0
        trial_drugs: list[tuple[_Entity, list[_Entity]]] = []
        for d in drugs:
            for _ in range(1 if rng.random() < 0.65 else 2):
                members = [d]
                if rng.random() < 0.15:
                    members.append(rng.choice(drugs))
                members = _dedup(members)
                trial = self._trial(n_trials, rng.choice((1, 2, 2, 3, 3)))
                n_trials += 1
                trial_drugs.append((trial, members))
        disease_by_iri = {e.iri: e for e in leaves}
        group_labels = {
            s: str(o) for s, o in self.g.subject_objects(RDFS.label) if str(s).startswith(str(ID))
        }
        for trial, members in trial_drugs:
            candidates = _dedup([x for t in d_tgts[members[0].iri] for x in t_dis[t.iri]])
            disease = rng.choice(candidates)
            if rng.random() < 0.2:  # basket trial at family level
                fam = parent[disease.iri]
                disease = _Entity(fam, group_labels[fam])
            disease_by_iri.setdefault(disease.iri, disease)
            self.g.add((trial.iri, KG.investigates, disease.iri))
            for m in members:
                self.g.add((m.iri, KG.studiedIn, trial.iri))
                d_trials[m.iri].append((trial, disease))

        self._abstracts(drugs, targets, d_tgts, d_trials, t_paths, t_dis, leaves)
        return self.g

    # ------------------------------------------------------------------ abstracts

    def _surface(self, ent: _Entity, allow_typo: bool = False) -> str:
        rng = self.rng
        forms = [ent.label] + [a for a in ent.alt if not _is_abbrev(a)]
        if any(_is_abbrev(a) for a in ent.alt) and rng.random() < 0.15:
            forms = [a for a in ent.alt if _is_abbrev(a)]
        s = forms[0] if rng.random() < 0.4 else rng.choice(forms)
        if allow_typo and rng.random() < self.cfg.typo_rate and len(s) > 6:
            j = rng.randrange(2, len(s) - 2)
            s = s[:j] + s[j + 1] + s[j] + s[j + 2 :]
        return s

    def _abstracts(self, drugs, targets, d_tgts, d_trials, t_paths, t_dis, leaves) -> None:
        rng, n = self.rng, 0
        for d in drugs:
            mentioned = [d.iri]
            ts = d_tgts[d.iri]
            stated = [(d.iri, KG.inhibits, t.iri) for t in ts]
            mentioned += [t.iri for t in ts]
            tlist = _join([self._surface(t) for t in ts])
            dname = self._surface(d, allow_typo=True)
            sents = [
                rng.choice(
                    (
                        f"{dname} is an investigational inhibitor of {tlist}.",
                        f"{dname} selectively inhibits {tlist} in cellular assays.",
                        f"{dname} blocks {tlist} activity with nanomolar potency.",
                    )
                )
            ]
            for trial, disease in d_trials[d.iri]:
                mentioned += [trial.iri, disease.iri]
                stated += [(d.iri, KG.studiedIn, trial.iri), (trial.iri, KG.investigates, disease.iri)]
                ph = self.g.value(trial.iri, KG.phase)
                tr, ds, dn = self._surface(trial), self._surface(disease), self._surface(d)
                sents.append(
                    rng.choice(
                        (
                            f"In the {tr}, a phase {ph} study in patients with {ds}, "
                            f"{dn} was evaluated against placebo.",
                            f"{dn} was tested in {tr}, which enrolled participants with {ds}.",
                        )
                    )
                )
            if rng.random() < 0.3:
                other = rng.choice([t for t in targets if t not in ts])
                mentioned.append(other.iri)
                sents.append(f"No meaningful activity against {other.label} was observed.")
            sents.append(rng.choice(_FILLERS))
            self._publication(n, " ".join(sents), mentioned, stated)
            n += 1
        for t in targets:
            mentioned = [t.iri]
            ps, ds = t_paths[t.iri], t_dis[t.iri]
            stated = [(t.iri, KG.participatesIn, p.iri) for p in ps]
            stated += [(t.iri, KG.associatedWith, x.iri) for x in ds]
            mentioned += [p.iri for p in ps] + [x.iri for x in ds]
            tn = self._surface(t)
            sents = [  # surface forms are drawn in sentence order to keep the rng sequence stable
                rng.choice(
                    (
                        f"{tn} participates in the {_join([self._surface(p) for p in ps])}.",
                        f"{tn} is a core component of the {_join([self._surface(p) for p in ps])}.",
                    )
                ),
                rng.choice(
                    (
                        "Elevated {t} expression was associated with {d} in patient-derived samples.",
                        "Genetic variants in {t} are linked to {d}.",
                    )
                ),
            ]
            sents[1] = sents[1].format(t=self._surface(t), d=_join([self._surface(x) for x in ds]))
            if rng.random() < 0.3:
                other = rng.choice([x for x in leaves if x not in ds])
                mentioned.append(other.iri)
                sents.append(f"{t.label} levels were unchanged in {other.label} controls.")
            sents.append(rng.choice(_FILLERS))
            self._publication(n, " ".join(sents), mentioned, stated)
            n += 1

    def _publication(self, n: int, text: str, mentioned: list[URIRef], stated: list[tuple]) -> None:
        iri = ID[f"publication/syn-p{n:04d}"]
        self.g.add((iri, RDF.type, KG.Publication))
        self.g.add((iri, RDFS.label, Literal(f"SYN-P{n:04d}", lang="en")))
        self.g.add((iri, KG.abstractText, Literal(text, lang="en")))
        for m in _dedup(mentioned):
            self.g.add((iri, KG.mentions, m))
        # Statement-level provenance with standard RDF reification.
        for j, (subj, pred, obj) in enumerate(_dedup(stated)):
            st = ID[f"statement/syn-p{n:04d}-{j}"]
            self.g.add((iri, KG.asserts, st))
            self.g.add((st, RDF.type, RDF.Statement))
            self.g.add((st, RDF.subject, subj))
            self.g.add((st, RDF.predicate, pred))
            self.g.add((st, RDF.object, obj))


_FILLERS = (
    "Pharmacokinetic data supported once-daily dosing.",
    "Further mechanistic work is ongoing.",
    "The safety profile was considered acceptable.",
    "Biomarker analyses are planned for the next cohort.",
)


def _is_abbrev(s: str) -> bool:
    return s.isupper() and s.isalpha() and len(s) <= 4


def _dedup(items: list) -> list:
    seen, out = set(), []
    for x in items:
        key = x.iri if hasattr(x, "iri") else x
        if key not in seen:
            seen.add(key)
            out.append(x)
    return out


def _join(parts: list[str]) -> str:
    return parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]


def load_ontology() -> Graph:
    g = Graph()
    g.parse(data=resources.files("kgrag.data").joinpath("ontology.ttl").read_text(), format="turtle")
    return g


def generate_graph(config: GeneratorConfig | None = None) -> Graph:
    """Ontology (TBox) plus generated instance data (ABox) in one graph."""
    g = load_ontology()
    data = SyntheticKGGenerator(config).generate()
    for prefix, ns in data.namespaces():
        g.bind(prefix, ns, override=False)
    for t in data:
        g.add(t)
    return g
