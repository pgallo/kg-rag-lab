import pytest

from kgrag.retrieval.hybrid import reciprocal_rank_fusion


def test_rrf_scores_match_formula():
    fused = dict(reciprocal_rank_fusion([["a", "b", "c"], ["b", "a"]], k=60))
    assert fused["a"] == pytest.approx(1 / 61 + 1 / 62)
    assert fused["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert fused["c"] == pytest.approx(1 / 63)


def test_rrf_rewards_agreement_over_single_top_rank():
    fused = [x for x, _ in reciprocal_rank_fusion([["solo", "both"], ["x", "both"], ["y", "both"]], k=1)]
    assert fused[0] == "both"


def test_rrf_weights_and_empty_lists():
    fused = reciprocal_rank_fusion([["a"], ["b"], []], weights=[1.0, 2.0, 5.0])
    assert [x for x, _ in fused] == ["b", "a"]
    assert reciprocal_rank_fusion([[], []]) == []


def test_rrf_is_deterministic_on_ties():
    a = reciprocal_rank_fusion([["p", "q"], ["q", "p"]])
    b = reciprocal_rank_fusion([["p", "q"], ["q", "p"]])
    assert a == b and [x for x, _ in a] == ["p", "q"]


def test_hybrid_retriever_merges_sources(pipe, multi_hop_question):
    res = pipe.retrievers["hybrid"].retrieve(multi_hop_question.question, k=20)
    assert len(res.facts) == 20
    assert len({f.triple for f in res.facts}) == 20
    assert "fusion_weights" in res.notes
