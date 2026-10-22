import pytest

from kgrag.evaluation.questions import load_questions
from kgrag.kg import KnowledgeGraph
from kgrag.pipeline import Pipeline


@pytest.fixture(scope="session")
def kg() -> KnowledgeGraph:
    return KnowledgeGraph.synthetic()


@pytest.fixture(scope="session")
def pipe(kg) -> Pipeline:
    return Pipeline.build(kg)


@pytest.fixture(scope="session")
def questions():
    return load_questions()


@pytest.fixture(scope="session")
def multi_hop_question(questions):
    return next(q for q in questions if q.template == "disease_trial_pathways" and len(q.gold) >= 2)
