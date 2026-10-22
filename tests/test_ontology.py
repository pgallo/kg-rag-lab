from rdflib import URIRef

from kgrag.generator import GeneratorConfig, SyntheticKGGenerator, generate_graph
from kgrag.kg import KnowledgeGraph
from kgrag.namespaces import ID, KG, RDFS


def test_generator_is_deterministic():
    a = generate_graph(GeneratorConfig(seed=7)).serialize(format="nt")
    b = generate_graph(GeneratorConfig(seed=7)).serialize(format="nt")
    assert sorted(a.splitlines()) == sorted(b.splitlines())
    c = SyntheticKGGenerator(GeneratorConfig(seed=8)).generate().serialize(format="nt")
    assert sorted(a.splitlines()) != sorted(c.splitlines())


def test_entity_counts_in_target_range(kg):
    stats = kg.stats()
    entities = sum(v for key, v in stats.items() if key not in {"facts", "triples"})
    assert 300 <= entities <= 800
    assert stats["Publication"] == 150


def test_subclass_closure_is_transitive(kg):
    leaf = next(d for d in kg.entities(KG.Disease) if not kg.is_disease_group(d))
    supers = kg.superclasses(leaf)
    assert KG.Disease in supers and KG.Entity in supers
    for s in list(supers):
        assert kg.superclasses(s) <= supers  # closure is closed
    assert kg.is_subclass_of(KG.Target, KG.Protein)
    assert kg.is_subclass_of(KG.Target, KG.Entity)
    assert not kg.is_subclass_of(KG.Drug, KG.Disease)


def test_subclasses_inverse_of_superclasses(kg):
    group = ID["disease/solid-tumour"]
    for sub in kg.subclasses(group):
        assert group in kg.superclasses(sub)
    assert len(kg.subclasses(ID["disease/oncological-disorder"])) == 2 + 2 * 3


def test_materialised_closure_matches_sparql_property_path():
    kg = KnowledgeGraph.synthetic()
    query = "SELECT ?c WHERE { ?c rdfs:subClassOf+ kg:Disease }"
    before = {r[0] for r in kg.sparql(query)}
    added = kg.materialise_subclass_closure()
    assert added > 0
    direct = {r[0] for r in kg.sparql("SELECT ?c WHERE { ?c rdfs:subClassOf kg:Disease }")}
    assert direct == before


def test_synonym_expansion(kg):
    drug = kg.entities(KG.Drug)[0]
    forms = kg.surface_forms(drug)
    assert forms[0] == kg.label(drug)
    assert any(f.startswith("LAB-") for f in forms[1:])
    target = kg.entities(KG.Target)[0]
    assert len(kg.surface_forms(target)) >= 3


def test_kg_is_mechanistically_consistent(kg):
    """Every trial disease is reachable from one of the trial drug's targets (or its parent)."""
    rows = kg.sparql("""
        SELECT ?trial ?disease WHERE { ?trial kg:investigates ?disease }""")
    for trial, disease in rows:
        ok = kg.ask(
            """
            ASK { ?drug kg:studiedIn ?trial ; kg:inhibits ?t . ?t kg:associatedWith ?d .
                  ?d rdfs:subClassOf* ?disease }""",
            trial=trial,
            disease=disease,
        )
        assert ok, (trial, disease)


def test_provenance_statements_are_true_facts(kg):
    from kgrag.retrieval import Provenance

    prov = Provenance(kg)
    assert len(prov.by_publication) == 150
    for triples in prov.by_publication.values():
        for t in triples:
            assert kg.has_fact(t)
    # Every domain fact is stated by at least one abstract.
    assert set(prov.by_triple) == set(kg.facts)


def test_ontology_declares_domains(kg):
    dom, rng = kg.domain_range(KG.inhibits)
    assert (dom, rng) == (KG.Drug, KG.Target)
    assert kg.label(RDFS.subClassOf) == "is a subtype of"
    assert isinstance(dom, URIRef)
