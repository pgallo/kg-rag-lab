from kgrag.linking import EntityLinker
from kgrag.namespaces import KG


def test_exact_label_and_multiword(kg):
    linker = EntityLinker(kg)
    disease = next(d for d in kg.entities(KG.Disease) if not kg.is_disease_group(d))
    links = linker.link(f"Which trials studied patients with {kg.label(disease)}?")
    assert [lk.entity for lk in links] == [disease]
    assert links[0].method == "exact"


def test_alt_label_synonyms(kg):
    linker = EntityLinker(kg)
    drug = kg.entities(KG.Drug)[3]
    code = next(a for a in kg.alt_labels(drug) if a.startswith("LAB-"))
    assert linker.entities(f"What does {code} inhibit?") == [drug]
    target = kg.entities(KG.Target)[5]
    symbol = kg.alt_labels(target)[0]
    assert linker.entities(f"Is {symbol} protein a kinase?") == [target]


def test_possessive_eponym_form(kg):
    linker = EntityLinker(kg)
    disease = next(d for d in kg.entities(KG.Disease) if any("'s " in a for a in kg.alt_labels(d)))
    form = next(a for a in kg.alt_labels(disease) if "'s " in a)
    assert linker.entities(f"Drugs for {form}?") == [disease]


def test_fuzzy_typo(kg):
    linker = EntityLinker(kg)
    drug = kg.entities(KG.Drug)[10]
    name = kg.label(drug)
    typo = name[:3] + name[4] + name[3] + name[5:]
    links = linker.link(f"Which targets does {typo} inhibit?")
    assert links and links[0].entity == drug and links[0].method == "fuzzy"
    assert 0.86 <= links[0].score < 1.0


def test_no_spurious_links_on_plain_words(kg):
    linker = EntityLinker(kg)
    assert linker.link("What is the best way to do this in general?") == []


def test_ambiguous_abbreviation_resolved_by_context(kg):
    linker = EntityLinker(kg)
    abbrev, cands = next((a, sorted(c)) for a, c in linker._abbrev.items() if len(c) > 1)
    a, b = cands
    # Put a target associated with candidate b in the question: coherence should pick b.
    target = next(iter(kg.graph.subjects(KG.associatedWith, b)), None)
    if target is None:
        a, b = b, a
        target = next(iter(kg.graph.subjects(KG.associatedWith, b)))
    links = linker.link(f"Is {kg.label(target)} relevant in {abbrev}?")
    resolved = {lk.mention.text: lk.entity for lk in links}
    assert resolved[abbrev] == b
    assert set(next(lk for lk in links if lk.mention.text == abbrev).candidates) == {a, b}


def test_restricted_linking_recovers_heavy_typo(kg):
    linker = EntityLinker(kg)
    drug = kg.entities(KG.Drug)[20]
    name = kg.label(drug)
    broken = name[0] + name[2] + name[1] + name[3:6] + name[7:]
    question = f"Which targets does {broken} inhibit?"
    restricted = linker.link_restricted(question, {drug, kg.entities(KG.Drug)[21]})
    assert [lk.entity for lk in restricted] == [drug]
