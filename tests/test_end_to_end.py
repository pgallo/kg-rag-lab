import json

import pytest

from kgrag.cli import main
from kgrag.evaluation.harness import ablation_markdown, evaluate
from kgrag.evaluation.metrics import exact_match
from kgrag.evaluation.questions import generate_questions, load_questions
from kgrag.llm import NO_ANSWER, StubLLM, hosted_available, make_llm


def test_ask_multi_hop_with_provenance(pipe, multi_hop_question):
    q = multi_hop_question
    context, answer = pipe.ask(q.question, retriever="hybrid")
    assert exact_match(answer.items, q.gold)
    assert answer.provenance and set(answer.provenance) <= set(context.triples)
    # Every cited triple is a real fact and is backed by at least one abstract.
    by_triple = {f.triple: f for f in context.facts}
    for t in answer.provenance:
        assert pipe.kg.has_fact(t)
        assert by_triple[t].sources


def test_stub_reader_declines_when_chain_is_missing(pipe, multi_hop_question):
    context, _ = pipe.ask(multi_hop_question.question, retriever="graph")
    context.facts = [f for f in context.facts if "participates in" not in f.text]
    answer = pipe.llm.answer(multi_hop_question.question, context)
    assert answer.text == NO_ANSWER and answer.items == []


def test_stub_reader_recovers_misspelled_entity_from_context(pipe, questions):
    noisy = [q for q in questions if q.category == "misspelled"]
    assert all(pipe.linker.link(q.question) == [] for q in noisy)
    recovered = 0
    for q in noisy:
        _, answer = pipe.ask(q.question, retriever="dense")
        recovered += exact_match(answer.items, q.gold)
        _, graph_answer = pipe.ask(q.question, retriever="graph")
        assert graph_answer.items == []  # nothing linked, nothing to expand
    assert recovered >= 1


def test_make_llm_defaults_offline(kg, monkeypatch):
    for name in ("LLM_API_URL", "LLM_API_KEY", "LLM_MODEL"):
        monkeypatch.delenv(name, raising=False)
    assert not hosted_available()
    assert isinstance(make_llm("auto", kg), StubLLM)
    with pytest.raises(RuntimeError):
        make_llm("hosted", kg)
    # A partial configuration is still unavailable: there are no defaults for URL or model.
    monkeypatch.setenv("LLM_API_KEY", "test")
    monkeypatch.setenv("LLM_API_URL", "http://localhost.invalid/v1/chat/completions")
    assert not hosted_available()
    assert isinstance(make_llm("auto", kg), StubLLM)


def test_bundled_questions_are_reproducible(kg, questions):
    assert 55 <= len(questions) <= 70
    regenerated = generate_questions(kg)
    assert [q.question for q in regenerated] == [q.question for q in questions]
    assert [q.gold for q in regenerated] == [q.gold for q in questions]
    cats = {q.category for q in questions}
    assert cats == {"single-hop", "two-hop", "multi-hop", "hierarchy", "misspelled"}


def test_ablation_graph_and_hybrid_win_on_multi_hop(pipe, questions):
    report = evaluate(pipe, questions, k=20)
    s = {name: report.summary(name, "multi-hop") for name in ("bm25", "dense", "graph", "hybrid")}
    for structured in ("graph", "hybrid"):
        for text in ("bm25", "dense"):
            assert s[structured]["exact"] >= s[text]["exact"] + 0.3
            assert s[structured]["evidence_recall"] > s[text]["evidence_recall"]
    # Text retrieval is competitive on single-hop questions.
    assert report.summary("bm25", "single-hop")["exact"] >= 0.75
    # Dense retrieval covers questions where symbolic linking fails; graph alone cannot.
    assert report.summary("dense", "misspelled")["exact"] > report.summary("graph", "misspelled")["exact"]
    # The stub reader is faithful by construction.
    assert all(r.faithful == 1.0 for r in report.results)
    md = ablation_markdown(report)
    assert md.count("| hybrid |") == 3 and "Cohen's kappa" in md
    cal = report.calibration
    assert cal["system answers"].agreement >= 0.95
    assert cal["synonym-rewritten answers"].judge_only > 0


def test_cli_build_ask_eval(tmp_path, capsys):
    graph = tmp_path / "kg.ttl"
    qfile = tmp_path / "questions.json"
    assert main(["build", "--out", str(graph), "--questions", str(qfile)]) == 0
    assert graph.exists() and len(json.loads(qfile.read_text())) == len(load_questions())
    out = capsys.readouterr().out
    assert "Drug" in out and "Publication" in out

    q = load_questions()[0]
    assert main(["ask", q.question, "--graph", str(graph), "--k", "8"]) == 0
    out = capsys.readouterr().out
    assert "Retrieved facts (hybrid" in out and "Provenance triples:" in out
    for g in q.gold:
        assert g in out

    md_path = tmp_path / "eval.md"
    assert main(["eval", "--graph", str(graph), "--retrievers", "bm25,graph", "--out", str(md_path)]) == 0
    out = capsys.readouterr().out
    assert out.startswith("### Overall") and "| graph |" in out
    assert md_path.read_text().strip() == out.strip()


def test_cli_sparql(capsys):
    assert main(["sparql", "SELECT (COUNT(?d) AS ?n) WHERE { ?d a kg:Drug }", "--graph", "missing.ttl"]) == 0
    assert capsys.readouterr().out.strip() == "90"
