from kgrag.evaluation.harness import gold_evidence
from kgrag.retrieval import HashingSVDEmbedder, RetrievalResult
from kgrag.retrieval.bm25 import BM25Index, tokenize
from kgrag.retrieval.hybrid import adaptive_weights


def test_bm25_index_scores_and_tokenizer():
    assert tokenize("Which drugs inhibit LAB-1017?") == ["drugs", "inhibit", "lab-1017"]
    idx = BM25Index([["alpha", "beta"], ["beta", "gamma", "gamma"], ["delta"]])
    s = idx.scores(["gamma"])
    assert s[1] > 0 and s[0] == 0 and s[2] == 0
    rare, common = idx.scores(["delta"])[2], idx.scores(["beta"])[0]
    assert rare > common  # idf


def test_bm25_finds_abstract_by_drug_name(pipe, kg):
    bm25 = pipe.retrievers["bm25"]
    passage = kg.passages[0]
    drug = next(m for m in sorted(passage.mentions) if "/drug/" in str(m))
    ranked = bm25.rank(f"What does {kg.label(drug)} do?")
    assert passage.iri in [p for p, _ in ranked[:3]]
    res = bm25.retrieve(f"What does {kg.label(drug)} do?", k=5)
    assert 0 < len(res.facts) <= 5 and all(f.sources for f in res.facts)
    assert res.passages[0][0] == passage.iri


def test_hashing_embedder_is_deterministic_and_normalised():
    texts = ["Velorinib inhibits a kinase", "a kinase pathway", "unrelated trial text", "another text"]
    a = HashingSVDEmbedder(dim=3).fit(texts).encode(texts)
    b = HashingSVDEmbedder(dim=3).fit(texts).encode(texts)
    assert (abs(a - b) < 1e-9).all()
    assert abs((a**2).sum(axis=1) - 1).max() < 1e-6


def test_dense_tolerates_misspelling(pipe, kg):
    dense = pipe.retrievers["dense"]
    passage = kg.passages[2]
    drug = next(m for m in sorted(passage.mentions) if "/drug/" in str(m))
    name = kg.label(drug)
    typo = name[:2] + name[3] + name[2] + name[4:-2] + name[-1]
    top = [p for p, _ in dense.rank(f"Which targets does {typo} inhibit?")[:5]]
    assert passage.iri in top


def test_graph_retriever_follows_multi_hop_chain(pipe, multi_hop_question):
    q = multi_hop_question
    res = pipe.retrievers["graph"].retrieve(q.question, k=20)
    assert res.notes["schema_path"] == "^investigates / ^studiedIn / inhibits / participatesIn"
    assert gold_evidence(pipe, q) <= set(res.triples)
    assert all(a.facts == sorted(a.facts, key=lambda f: -f.score) for a in [res])


def test_graph_retriever_expands_disease_groups(pipe, questions):
    q = next(q for q in questions if q.category == "hierarchy")
    res = pipe.retrievers["graph"].retrieve(q.question, k=20)
    assert res.schema_facts, "subclass links should be reported"
    assert gold_evidence(pipe, q) <= set(res.triples)


def test_graph_retriever_without_entities(pipe):
    res = pipe.retrievers["graph"].retrieve("What is the weather like?", k=10)
    assert res.facts == [] and res.notes["reason"] == "no entities linked"


def test_graph_beats_text_on_multi_hop_evidence(pipe, questions):
    multi = [q for q in questions if q.category == "multi-hop"]

    def mean_recall(name):
        total = 0.0
        for q in multi:
            gold = gold_evidence(pipe, q)
            total += len(gold & set(pipe.retrievers[name].retrieve(q.question, 20).triples)) / len(gold)
        return total / len(multi)

    graph = mean_recall("graph")
    assert graph > mean_recall("bm25") + 0.2
    assert graph > mean_recall("dense") + 0.2


def test_adaptive_weights_rules():
    def res(seeds, path="", schema=()):
        return RetrievalResult(
            "graph", [], seeds=seeds, notes={"schema_path": path} if path else {}, schema_facts=list(schema)
        )

    text = RetrievalResult("bm25", [])
    assert adaptive_weights([res([]), text, text]) == [0.0, 1.0, 1.0]
    assert adaptive_weights([res(["x"], "inhibits"), text, text]) == [1.0, 0.5, 0.5]
    assert adaptive_weights([res(["x"], "a / b / c"), text, text]) == [1.0, 0.2, 0.2]
    assert adaptive_weights([res(["x"], "a / b", schema=["s"]), text, text]) == [1.0, 0.2, 0.2]
