"""LLM-as-judge harness.

The judge grades an answer against the gold answer and the retrieved context with a rubric
(correctness 0-2, faithfulness 0-2). ``StubJudge`` applies the same rubric deterministically
so the harness runs offline; ``HostedJudge`` sends ``RUBRIC_PROMPT`` to a hosted model.

Calibration: before trusting any judge, compare its verdicts with a mechanical reference on
a subset where the reference is reliable. ``calibrate`` reports agreement and Cohen's kappa
between "judge says fully correct" and exact match. A judge that disagrees with exact match
on unambiguous questions should not be used to score the ambiguous ones.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Protocol

from kgrag.evaluation.metrics import fuzzy_f1

RUBRIC_PROMPT = """You are grading an answer produced by a retrieval-augmented system.

Question: {question}
Gold answer(s): {gold}
System answer: {answer}
Retrieved facts:
{facts}

Score two criteria.
correctness:
  2 = the answer names all gold entities and no wrong entity
  1 = partially correct: at least one gold entity, or all gold entities plus extra wrong ones
  0 = no gold entity, or the system declined to answer
faithfulness:
  2 = every entity in the answer is supported by the retrieved facts
  1 = some entities are supported
  0 = the answer is not supported by the retrieved facts
Return JSON only: {{"correctness": <0-2>, "faithfulness": <0-2>, "rationale": "<one sentence>"}}"""


@dataclass
class Verdict:
    correctness: int
    faithfulness: int
    rationale: str

    @property
    def correct(self) -> bool:
        return self.correctness == 2


class Judge(Protocol):
    name: str

    def judge(self, question: str, gold: list[str], answer_items: list[str], facts: list[str]) -> Verdict: ...


class StubJudge:
    """Deterministic rubric implementation.

    Correctness uses fuzzy item matching after mapping synonyms to preferred labels (when an
    alias table is given), so it is a genuinely different signal from exact match: it forgives
    surface variation such as a drug code instead of its name, as a human or model judge would.
    Faithfulness checks that each answer string occurs in a retrieved fact.
    """

    name = "stub-judge"

    def __init__(self, aliases: dict[str, str] | None = None, threshold: float = 90.0) -> None:
        self.aliases = {k.lower(): v for k, v in (aliases or {}).items()}
        self.threshold = threshold

    def canonical(self, item: str) -> str:
        return self.aliases.get(item.lower(), item)

    def judge(self, question: str, gold: list[str], answer_items: list[str], facts: list[str]) -> Verdict:
        if not answer_items:
            return Verdict(0, 0, "No answer given.")
        f1 = fuzzy_f1([self.canonical(a) for a in answer_items], gold, self.threshold)
        correctness = 2 if f1 >= 0.999 else 1 if f1 > 0 else 0
        blob = " ".join(facts).lower()
        supported = sum(1 for a in answer_items if self.canonical(a).lower() in blob or a.lower() in blob)
        faithful = 2 if supported == len(answer_items) else 1 if supported else 0
        return Verdict(
            correctness, faithful, f"fuzzy F1={f1:.2f}; {supported}/{len(answer_items)} supported."
        )


def alias_table(kg) -> dict[str, str]:
    """skos:altLabel -> preferred label, for every entity (used by the stub judge)."""
    return {alt: kg.label(e) for e in kg.entities() for alt in kg.alt_labels(e)}


class HostedJudge:  # pragma: no cover - requires network and an API key
    name = "hosted-judge"

    def __init__(self) -> None:
        from kgrag.llm import AnthropicLLM

        self.llm = AnthropicLLM(model=os.environ.get("KGRAG_JUDGE_MODEL"))

    def judge(self, question: str, gold: list[str], answer_items: list[str], facts: list[str]) -> Verdict:
        from kgrag.llm import parse_json_object

        prompt = RUBRIC_PROMPT.format(
            question=question,
            gold="; ".join(gold),
            answer="; ".join(answer_items) or "(none)",
            facts="\n".join(f"- {f}" for f in facts),
        )
        data = parse_json_object(self.llm.complete(prompt))
        return Verdict(
            int(data.get("correctness", 0)), int(data.get("faithfulness", 0)), str(data.get("rationale", ""))
        )


@dataclass
class Calibration:
    n: int
    agreement: float
    kappa: float
    judge_only: int  # judge says correct, exact match says wrong
    em_only: int  # exact match says correct, judge says wrong


def calibrate(judge_correct: list[bool], em_correct: list[bool]) -> Calibration:
    n = len(judge_correct)
    if n == 0:
        return Calibration(0, 1.0, 1.0, 0, 0)
    agree = sum(a == b for a, b in zip(judge_correct, em_correct, strict=True)) / n
    pj, pe = sum(judge_correct) / n, sum(em_correct) / n
    chance = pj * pe + (1 - pj) * (1 - pe)
    kappa = 1.0 if chance == 1 else (agree - chance) / (1 - chance)
    judge_only = sum(a and not b for a, b in zip(judge_correct, em_correct, strict=True))
    em_only = sum(b and not a for a, b in zip(judge_correct, em_correct, strict=True))
    return Calibration(n, agree, kappa, judge_only, em_only)
