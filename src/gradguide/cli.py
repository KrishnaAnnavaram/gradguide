"""``gradguide`` command line."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from gradguide.config import Settings


def _settings(args: argparse.Namespace) -> Settings:
    s = Settings.from_env()
    if args.docs:
        s.docs_dir = Path(args.docs).resolve()
    if args.index_dir:
        s.index_dir = Path(args.index_dir).resolve()
    return s


def _service(args: argparse.Namespace, force: bool = False):
    from gradguide.service import GradGuide

    return GradGuide(_settings(args), force_reindex=force)


def cmd_index(args: argparse.Namespace) -> int:
    svc = _service(args, force=args.force)
    info = svc.describe()
    print(f"{svc.report}. {info['chunks']} chunks from {info['sources']} sources -> {svc.settings.index_dir}")
    print(f"embedder={info['embedder']}  tokenizer={svc.tokenizer.name}  chunk_tokens={svc.settings.chunk_tokens}")
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    from gradguide.render import plain

    answer = _service(args).ask(args.question, mode=args.mode)
    print(json.dumps(answer.to_dict(), indent=2) if args.json else plain(answer))
    return 0


def cmd_chat(args: argparse.Namespace) -> int:
    from gradguide.render import plain

    svc = _service(args)
    history: list[tuple[str, str]] = []
    print("gradguide - ask about your program's policies. Empty line to quit.")
    while True:
        try:
            question = input("\n> ").strip()
        except EOFError:
            break
        if not question:
            break
        answer = svc.ask(question, history)
        history.append((question, answer.text))
        print("\n" + plain(answer))
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    from gradguide.eval.harness import (answer_report, chunk_size_ablation, load_gold, markdown_table,
                                        retrieval_report)

    svc = _service(args)
    gold = load_gold(args.gold)
    n_ans = sum(g.answerable for g in gold)
    results: dict = {"setup": svc.describe(), "questions": len(gold), "answerable": n_ans}

    retriever = svc.retriever
    if retriever.reranker is None:  # still ablate re-ranking, with the dependency-free lexical reranker
        from gradguide.retrieve.hybrid import Retriever
        from gradguide.retrieve.rerank import LexicalReranker

        retriever = Retriever(svc.index, svc.embedder, LexicalReranker(), candidate_k=svc.settings.candidate_k)
    retrieval = retrieval_report(retriever, gold)
    results["retrieval"] = retrieval
    print(f"## Retrieval ({n_ans} answerable questions)\n\n{markdown_table(retrieval, 'mode')}\n")

    answers = answer_report(svc, gold)
    results["answers"] = answers
    print(f"## Answers ({len(gold)} questions, {len(gold) - n_ans} unanswerable)\n")
    print("\n".join(f"- {k}: {v:.3f}" for k, v in answers.items()) + "\n")

    if args.chunk_sizes:
        sizes = [int(x) for x in args.chunk_sizes.split(",") if x.strip()]
        ablation = chunk_size_ablation(svc.settings, gold, sizes, embedder=svc.embedder, model=svc.model)
        results["chunk_size_ablation"] = {str(k): v for k, v in ablation.items()}
        print(f"## Chunk size ablation (hybrid)\n\n{markdown_table(ablation, 'chunk tokens')}\n")

    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"results written to {args.out}")
    return 0


def cmd_sources(args: argparse.Namespace) -> int:
    from gradguide.ingest.sources import load_manifest, stale_sources

    manifest = load_manifest(_settings(args).docs_dir)
    if not manifest:
        print("no sources.toml manifest in the documents folder")
        return 0
    stale = {s.path for s in stale_sources(manifest)}
    for info in manifest.values():
        flag = "STALE" if info.path in stale else "ok"
        print(f"{flag:5}  {info.path}  owner={info.owner or '-'}  updated={info.updated or '-'}  "
              f"refresh_days={info.refresh_days}")
    return 1 if (stale and args.strict) else 0


def cmd_purge_logs(args: argparse.Namespace) -> int:
    from gradguide.privacy.querylog import QueryLog

    s = _settings(args)
    removed = QueryLog(s.query_log_path, True, s.log_retention_days).purge()
    print(f"removed {removed} log records older than {s.log_retention_days} days")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from gradguide.api import create_app

    uvicorn.run(create_app(settings=_settings(args)), host=args.host, port=args.port)
    return 0


def cmd_ui(args: argparse.Namespace) -> int:
    return subprocess.call([sys.executable, "-m", "streamlit", "run", str(Path(__file__).with_name("app.py"))])


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="gradguide", description="Document-grounded graduate advising assistant")
    p.add_argument("--docs", help="documents folder (overrides GRADGUIDE_DOCS_DIR)")
    p.add_argument("--index-dir", help="index folder (overrides GRADGUIDE_INDEX_DIR)")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("index", help="build or incrementally update the index")
    s.add_argument("--force", action="store_true", help="re-chunk and re-embed everything")
    s.set_defaults(func=cmd_index)

    s = sub.add_parser("ask", help="answer one question")
    s.add_argument("question")
    s.add_argument("--mode", choices=("bm25", "vector", "hybrid", "hybrid+rerank"))
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_ask)

    sub.add_parser("chat", help="interactive session").set_defaults(func=cmd_chat)

    s = sub.add_parser("eval", help="retrieval metrics, abstention checks and ablations")
    s.add_argument("--gold", default="eval/gold_set.jsonl")
    s.add_argument("--chunk-sizes", default="", help="comma-separated token sizes to ablate, e.g. 120,220,400")
    s.add_argument("--out", help="write results as JSON")
    s.set_defaults(func=cmd_eval)

    s = sub.add_parser("sources", help="list sources from sources.toml and flag stale ones")
    s.add_argument("--strict", action="store_true", help="exit 1 if any source is stale")
    s.set_defaults(func=cmd_sources)

    sub.add_parser("purge-logs", help="apply the query-log retention period now").set_defaults(func=cmd_purge_logs)

    s = sub.add_parser("serve", help="run the HTTP API (needs the 'api' extra)")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.set_defaults(func=cmd_serve)

    sub.add_parser("ui", help="run the Streamlit UI (needs the 'ui' extra)").set_defaults(func=cmd_ui)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
