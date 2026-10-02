"""Command-line interface: ``spanner train | ask | eval``."""

from __future__ import annotations

import argparse

from .train import DEFAULT_BASE_MODEL

DEFAULT_MODEL_DIR = "models/spanner-kangaroo"


def cmd_train(args: argparse.Namespace) -> None:
    from .train import train

    train(
        data_path=args.data,
        output_dir=args.output,
        base_model=args.base_model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_length=args.max_length,
        stride=args.stride,
        seed=args.seed,
        device=args.device,
    )


def cmd_ask(args: argparse.Namespace) -> None:
    from .data import read_text
    from .predict import QuestionAnswerer

    qa = QuestionAnswerer(args.model, device=args.device)
    context = read_text(args.context)

    def respond(question: str) -> None:
        answer = qa.answer(question, context)
        print(f"Answer: {answer.text}  (confidence {answer.score:.2f})")

    if args.question:
        respond(" ".join(args.question))
        return

    print(f"Ask questions about {args.context}. Press Ctrl+D or type 'exit' to quit.")
    while True:
        try:
            question = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if question.lower() in {"exit", "quit"}:
            break
        if question:
            respond(question)


def cmd_eval(args: argparse.Namespace) -> None:
    from .data import load_examples
    from .evaluate import evaluate
    from .predict import QuestionAnswerer

    qa = QuestionAnswerer(args.model, device=args.device)
    results = evaluate(qa, load_examples(args.data), verbose=args.verbose)
    print(f"Exact match: {results['exact_match']:.1f}  F1: {results['f1']:.1f}  ({results['count']} questions)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="spanner", description="Extractive question answering with BERT.")
    parser.add_argument("--device", default=None, help="cpu, cuda or mps (auto-detected by default)")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("train", help="fine-tune a QA model on a JSON dataset")
    p.add_argument("--data", default="data/kangaroo_qa.json")
    p.add_argument("--output", default=DEFAULT_MODEL_DIR)
    p.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--max-length", type=int, default=384)
    p.add_argument("--stride", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("ask", help="answer questions about a text file")
    p.add_argument("question", nargs="*", help="question to ask; omit for interactive mode")
    p.add_argument("--context", default="data/kangaroo.txt", help="text file to answer from")
    p.add_argument("--model", default=DEFAULT_MODEL_DIR, help="model directory or Hugging Face model id")
    p.set_defaults(func=cmd_ask)

    p = sub.add_parser("eval", help="report Exact Match and F1 on a JSON dataset")
    p.add_argument("--data", default="data/kangaroo_qa.json")
    p.add_argument("--model", default=DEFAULT_MODEL_DIR)
    p.add_argument("-v", "--verbose", action="store_true", help="print every prediction")
    p.set_defaults(func=cmd_eval)

    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
