"""Command line interface: ``kgrag build | ask | eval | sparql``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from kgrag.kg import KnowledgeGraph
from kgrag.pipeline import RETRIEVERS, Pipeline

DEFAULT_GRAPH = Path("build/kg.ttl")


def cmd_build(args: argparse.Namespace) -> int:
    from kgrag.evaluation.questions import generate_questions, save_questions

    kg = KnowledgeGraph.synthetic(seed=args.seed)
    inferred = kg.materialise_subclass_closure() if args.materialise else 0
    kg.save(args.out)
    print(f"Wrote {args.out}")
    for key, value in kg.stats().items():
        print(f"  {key:<15} {value}")
    if args.materialise:
        print(f"  {'inferred':<15} {inferred} rdfs:subClassOf triples")
    if args.questions:
        qs = generate_questions(kg)
        save_questions(qs, args.questions)
        print(f"Wrote {len(qs)} questions to {args.questions}")
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    pipe = Pipeline.from_path(args.graph, llm=args.llm)
    context, answer = pipe.ask(args.question, retriever=args.retriever, k=args.k)
    kg = pipe.kg
    links = pipe.linker.link(args.question)
    print(f"Question : {args.question}")
    linked = [
        f"'{lk.mention.text}' -> {kg.label(lk.entity)} [{kg.label(kg.entity_type(lk.entity))}, {lk.method}]"
        for lk in links
    ]
    print("Linked   : " + (", ".join(linked) or "(none)"))
    for key, value in context.notes.items():
        print(f"{key:<9}: {value}")
    print(f"\nRetrieved facts ({context.retriever}, top {len(context.facts)}):")
    for i, f in enumerate(context.facts, start=1):
        src = f"  <- {', '.join(kg.label(s) for s in f.sources)}" if f.sources else ""
        print(f"  [{i:>2}] {f.text}{src}")
    if context.schema_facts:
        print("\nOntology facts used:")
        for f in context.schema_facts:
            print(f"   -   {f.text}")
    print(f"\nAnswer ({answer.model}): {answer.text}")
    if answer.provenance:
        print("\nProvenance triples:")
        for s, p, o in answer.provenance:
            print(f"  <{s}> <{p}> <{o}> .")
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    from kgrag.evaluation.harness import ablation_markdown, evaluate
    from kgrag.evaluation.questions import load_questions

    pipe = Pipeline.from_path(args.graph)
    questions = load_questions(args.questions)
    retrievers = tuple(args.retrievers.split(",")) if args.retrievers else RETRIEVERS
    report = evaluate(pipe, questions, k=args.k, retrievers=retrievers)
    md = ablation_markdown(report)
    print(md)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(md + "\n")
    return 0


def cmd_sparql(args: argparse.Namespace) -> int:
    kg = KnowledgeGraph.load(args.graph) if Path(args.graph).exists() else KnowledgeGraph.synthetic()
    for row in kg.sparql(args.query):
        print(
            "\t".join(kg.label(v) if hasattr(v, "n3") and str(v).startswith("http") else str(v) for v in row)
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="kgrag", description="Neuro-symbolic retrieval lab")
    sub = p.add_subparsers(dest="command", required=True)

    b = sub.add_parser("build", help="generate the synthetic knowledge graph as Turtle")
    b.add_argument("--out", type=Path, default=DEFAULT_GRAPH)
    b.add_argument("--seed", type=int, default=7)
    b.add_argument("--materialise", action="store_true", help="add inferred subClassOf triples")
    b.add_argument("--questions", type=Path, help="also regenerate the question set to this path")
    b.set_defaults(func=cmd_build)

    a = sub.add_parser("ask", help="answer a question with retrieved facts and provenance")
    a.add_argument("question")
    a.add_argument("--retriever", choices=RETRIEVERS, default="hybrid")
    a.add_argument("--k", type=int, default=20)
    a.add_argument("--llm", choices=("stub", "hosted", "auto"), default="stub")
    a.add_argument("--graph", type=Path, default=DEFAULT_GRAPH)
    a.set_defaults(func=cmd_ask)

    e = sub.add_parser("eval", help="run the ablation and print a markdown table")
    e.add_argument("--k", type=int, default=20)
    e.add_argument("--retrievers", help="comma separated subset of " + ",".join(RETRIEVERS))
    e.add_argument("--questions", type=Path, help="question set JSON (default: bundled set)")
    e.add_argument("--graph", type=Path, default=DEFAULT_GRAPH)
    e.add_argument("--out", type=Path, help="also write the markdown to this file")
    e.set_defaults(func=cmd_eval)

    s = sub.add_parser("sparql", help="run a SPARQL query (prefixes kg, rdfs, skos, rdf, owl predefined)")
    s.add_argument("query")
    s.add_argument("--graph", type=Path, default=DEFAULT_GRAPH)
    s.set_defaults(func=cmd_sparql)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
