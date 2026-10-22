"""Ablation harness: every retriever, every question, the same reader and the same budget k."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from statistics import mean

from rdflib import URIRef

from kgrag.analysis import Step
from kgrag.evaluation.judge import Calibration, Judge, StubJudge, alias_table, calibrate
from kgrag.evaluation.metrics import (
    answer_entity_recall,
    evidence_chain,
    exact_match,
    faithfulness,
    fuzzy_f1,
    recall_at_k,
    supported_answers,
)
from kgrag.evaluation.questions import Question
from kgrag.namespaces import KG
from kgrag.pipeline import RETRIEVERS, Pipeline

CATEGORIES = ("single-hop", "two-hop", "multi-hop", "hierarchy", "misspelled")
CALIBRATION_MAX_GOLD = 3  # calibrate on questions with short gold lists, where exact match is reliable


@dataclass
class QuestionResult:
    qid: str
    category: str
    retriever: str
    evidence_recall: float
    answer_recall: float
    exact: bool
    fuzzy: float
    faithful: float
    supported: float
    judge_correct: bool
    judge_score: int
    n_gold: int = 0
    # Synonym probe: the same answer rewritten with skos:altLabel forms.
    probe_exact: bool = False
    probe_judge_correct: bool = False


@dataclass
class EvalReport:
    k: int
    results: list[QuestionResult]
    calibration: dict[str, Calibration] = field(default_factory=dict)

    def rows(self, retriever: str, category: str | None = None) -> list[QuestionResult]:
        return [
            r
            for r in self.results
            if r.retriever == retriever and (category is None or r.category == category)
        ]

    def summary(self, retriever: str, category: str | None = None) -> dict[str, float]:
        rows = self.rows(retriever, category)
        if not rows:
            return {}
        return {
            "n": len(rows),
            "evidence_recall": mean(r.evidence_recall for r in rows),
            "answer_recall": mean(r.answer_recall for r in rows),
            "exact": mean(r.exact for r in rows),
            "fuzzy": mean(r.fuzzy for r in rows),
            "faithful": mean(r.faithful for r in rows),
            "supported": mean(r.supported for r in rows),
            "judge": mean(r.judge_score for r in rows) / 2,
        }


def gold_evidence(pipe: Pipeline, q: Question) -> set:
    steps = tuple(Step(KG[p.lstrip("^")], not p.startswith("^")) for p in q.path)
    return evidence_chain(pipe.kg, URIRef(q.entity), steps)


def evaluate(
    pipe: Pipeline,
    questions: list[Question],
    k: int = 20,
    retrievers: tuple[str, ...] = RETRIEVERS,
    judge: Judge | None = None,
) -> EvalReport:
    judge = judge or StubJudge(alias_table(pipe.kg))
    synonym = synonym_rewriter(pipe)
    results: list[QuestionResult] = []
    gold_ev = {q.id: gold_evidence(pipe, q) for q in questions}
    for name in retrievers:
        for q in questions:
            context, answer = pipe.ask(q.question, retriever=name, k=k)
            triples = context.triples
            fact_texts = [f.text for f in context.facts]
            verdict = judge.judge(q.question, q.gold, answer.items, fact_texts)
            probe = [synonym(i) for i in answer.items]
            probe_verdict = judge.judge(q.question, q.gold, probe, fact_texts)
            results.append(
                QuestionResult(
                    qid=q.id,
                    category=q.category,
                    retriever=name,
                    evidence_recall=recall_at_k(gold_ev[q.id], triples, k),
                    answer_recall=answer_entity_recall([URIRef(g) for g in q.gold_iris], triples, k),
                    exact=exact_match(answer.items, q.gold),
                    fuzzy=fuzzy_f1(answer.items, q.gold),
                    faithful=faithfulness(answer.provenance, triples),
                    supported=supported_answers(pipe.kg, answer.items, answer.provenance, triples),
                    judge_correct=verdict.correct,
                    judge_score=verdict.correctness,
                    n_gold=len(q.gold),
                    probe_exact=exact_match(probe, q.gold),
                    probe_judge_correct=probe_verdict.correct,
                )
            )
    report = EvalReport(k, results)
    subset = [r for r in results if r.n_gold <= CALIBRATION_MAX_GOLD]
    report.calibration["system answers"] = calibrate(
        [r.judge_correct for r in subset], [r.exact for r in subset]
    )
    report.calibration["synonym-rewritten answers"] = calibrate(
        [r.probe_judge_correct for r in subset], [r.probe_exact for r in subset]
    )
    return report


def synonym_rewriter(pipe: Pipeline):
    """Map a preferred label to a deterministic synonym (first altLabel), if the entity has one."""
    by_label = {pipe.kg.label(e): pipe.kg.alt_labels(e) for e in pipe.kg.entities()}

    def rewrite(item: str) -> str:
        alts = by_label.get(item) or []
        return alts[0] if alts else item

    return rewrite


def _pct(x: float) -> str:
    return f"{100 * x:.1f}"


def ablation_markdown(report: EvalReport) -> str:
    names = sorted({r.retriever for r in report.results}, key=RETRIEVERS.index)
    k = report.k
    lines = [
        f"### Overall (n={len(report.rows(names[0]))} questions, context budget k={k} facts)",
        "",
        f"| retriever | evidence recall@{k} | answer recall@{k} | exact match | fuzzy F1 "
        "| faithfulness | judge |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for n in names:
        s = report.summary(n)
        lines.append(
            f"| {n} | {_pct(s['evidence_recall'])} | {_pct(s['answer_recall'])} | {_pct(s['exact'])} "
            f"| {_pct(s['fuzzy'])} | {_pct(s['faithful'])} | {_pct(s['judge'])} |"
        )
    lines += [
        "",
        "### Exact match by question category",
        "",
        "| retriever | " + " | ".join(f"{c} (n={len(report.rows(names[0], c))})" for c in CATEGORIES) + " |",
        "|---|" + "---:|" * len(CATEGORIES),
    ]
    for n in names:
        cells = [_pct(report.summary(n, c).get("exact", 0.0)) for c in CATEGORIES]
        lines.append(f"| {n} | " + " | ".join(cells) + " |")
    lines += [
        "",
        f"### Evidence recall@{k} by question category",
        "",
        "| retriever | " + " | ".join(CATEGORIES) + " |",
        "|---|" + "---:|" * len(CATEGORIES),
    ]
    for n in names:
        cells = [_pct(report.summary(n, c).get("evidence_recall", 0.0)) for c in CATEGORIES]
        lines.append(f"| {n} | " + " | ".join(cells) + " |")
    lines += [
        "",
        f"### Judge calibration (all retrievers, questions with <= {CALIBRATION_MAX_GOLD} gold answers)",
        "",
        "Agreement between the judge verdict 'fully correct' and exact match. The probe rewrites "
        "each answer with an altLabel synonym: exact match rejects it, a usable judge should accept it.",
        "",
        "| answers judged | n | agreement | Cohen's kappa | judge-only correct | EM-only correct |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for key, c in report.calibration.items():
        lines.append(
            f"| {key} | {c.n} | {_pct(c.agreement)} | {c.kappa:.2f} | {c.judge_only} | {c.em_only} |"
        )
    return "\n".join(lines)


def per_category_wins(report: EvalReport, metric: str = "exact") -> dict[str, str]:
    out = {}
    names = sorted({r.retriever for r in report.results})
    for c in CATEGORIES:
        scores = defaultdict(float, {n: report.summary(n, c).get(metric, 0.0) for n in names})
        out[c] = max(names, key=lambda n: (scores[n], n == "hybrid"))
    return out
