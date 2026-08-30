"""
CLI chat entry point.

Usage:
    python -m vinrei.chat "explain this bug"
    python -m vinrei.chat --model deepseek-coder:1.3b "write a sorting function"
    python -m vinrei.chat  # interactive mode (no prompt argument)
"""

import argparse
import sys

from vinrei.ollama_client import DEFAULT_MODEL, stream
from vinrei.prompts import build as build_prompt
from vinrei import repo as repo_mod


def _stream_reply(prompt: str, model: str, task: str | None = None) -> None:
    """Stream the model reply to stdout."""
    system = build_prompt(task)
    try:
        for chunk in stream(prompt, model=model, system=system):
            print(chunk, end="", flush=True)
        print()  # newline after response
    except OSError as e:
        print(f"\n[error] Could not reach Ollama: {e}", file=sys.stderr)
        print("[error] Is 'ollama serve' running?", file=sys.stderr)
        sys.exit(1)


def interactive(model: str, task: str | None = None) -> None:
    """Start an interactive REPL session."""
    print(f"vinrei — {model}  (type 'exit' or Ctrl-C to quit)")
    print("─" * 50)
    while True:
        try:
            prompt = input("\n> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye.")
            break
        if not prompt:
            continue
        if prompt.lower() in {"exit", "quit"}:
            print("Bye.")
            break
        _stream_reply(prompt, model, task)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="vinrei",
        description="Local AI coding assistant powered by Ollama.",
    )
    parser.add_argument(
        "prompt",
        nargs="?",
        help="Prompt to send. Omit for interactive mode.",
    )
    parser.add_argument(
        "--model",
        "-m",
        default=DEFAULT_MODEL,
        help=f"Ollama model tag (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--task",
        "-t",
        choices=["explain", "edit", "debug", "search", "diff"],
        default=None,
        help="Task mode — tunes the system prompt (default: general)",
    )
    parser.add_argument(
        "--repo",
        '-r',
        default=None,
        help="Path to repo root. Injects tree + file context into the prompt"
    )
    parser.add_argument(
        "--file",
        "-f",
        action="append",
        dest="files",
        help="File(s) to include in context. Can be used multiple times"
    )
    args = parser.parse_args()

    if args.repo:
        ctx = repo_mod.context(args.repo, args.files)
        prompt = f"{ctx}\n\n{args.prompt or ''}"
    else:
        prompt = args.prompt or ""

    if prompt.strip():
        _stream_reply(prompt, args.model, args.task)
    else:
        interactive(args.model, args.task)


if __name__ == "__main__":
    main()
