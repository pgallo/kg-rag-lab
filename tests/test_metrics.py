import pytest

from kgrag.evaluation.judge import StubJudge, alias_table, calibrate
from kgrag.evaluation.metrics import (
    answer_entity_recall,
    exact_match,
    faithfulness,
    fuzzy_f1,
    recall_at_k,
    supported_answers,
)
from kgrag.namespaces import KG


def test_recall_at_k():
    assert recall_at_k({"a", "b"}, ["a", "x", "b"], k=2) == 0.5
    assert recall_at_k({"a", "b"}, ["a", "x", "b"], k=3) == 1.0
    assert recall_at_k(set(), [], k=5) == 1.0


def test_answer_entity_recall():
    triples = [("s1", "p", "o1"), ("s2", "p", "o2")]
    assert answer_entity_recall(["o1", "o2"], triples, k=1) == 0.5
    assert answer_entity_recall(["s2"], triples, k=2) == 1.0


def test_exact_match_normalises_case_and_space():
    assert exact_match(["Foo  bar", "baz."], ["baz", "foo bar"])
    assert not exact_match(["foo"], ["foo", "bar"])


def test_fuzzy_f1():
    assert fuzzy_f1(["Haematological malignancy"], ["Haematological  malignancy"]) == 1.0
    assert fuzzy_f1(["a thing", "other"], ["a thing"]) == pytest.approx(2 / 3)
    assert fuzzy_f1([], ["x"]) == 0.0 and fuzzy_f1([], []) == 1.0


def test_faithfulness_and_support(kg):
    drug = kg.entities(KG.Drug)[0]
    target = next(iter(kg.graph.objects(drug, KG.inhibits)))
    fact = (drug, KG.inhibits, target)
    other = (drug, KG.studiedIn, next(iter(kg.graph.objects(drug, KG.studiedIn))))
    assert faithfulness([fact], [fact, other]) == 1.0
    assert faithfulness([fact, other], [fact]) == 0.5
    assert faithfulness([], []) == 1.0
    assert supported_answers(kg, [kg.label(target)], [fact], [fact]) == 1.0
    assert supported_answers(kg, [kg.label(target)], [fact], []) == 0.0


def test_stub_judge_rubric(kg):
    judge = StubJudge(alias_table(kg))
    facts = ["Velorinib inhibits Alpha kinase 1.", "Velorinib inhibits Beta receptor 2."]
    assert judge.judge("q", ["Alpha kinase 1"], ["Alpha kinase 1"], facts).correctness == 2
    v = judge.judge("q", ["Alpha kinase 1", "Beta receptor 2"], ["Alpha kinase 1"], facts)
    assert (v.correctness, v.faithfulness) == (1, 2)
    v = judge.judge("q", ["Alpha kinase 1"], ["Gamma"], facts)
    assert (v.correctness, v.faithfulness) == (0, 0)
    assert judge.judge("q", ["x"], [], facts).correctness == 0


def test_stub_judge_accepts_synonyms_that_exact_match_rejects(kg):
    judge = StubJudge(alias_table(kg))
    target = kg.entities(KG.Target)[0]
    symbol = kg.alt_labels(target)[0]
    gold = [kg.label(target)]
    assert not exact_match([symbol], gold)
    assert judge.judge("q", gold, [symbol], [f"X inhibits {gold[0]}."]).correct


def test_calibration_kappa():
    c = calibrate([True, True, False, False], [True, True, False, False])
    assert (c.agreement, c.kappa) == (1.0, 1.0)
    c = calibrate([True, True, True, False], [True, False, True, False])
    assert c.agreement == 0.75 and c.judge_only == 1 and c.em_only == 0
    assert 0 < c.kappa < 1
