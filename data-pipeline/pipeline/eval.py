"""
eval.py — Smoke eval harness for vinrei.

Runs a fixed set of 20 coding tasks against the model and scores
each response automatically using simple heuristics:

    - contains_code   : response has a fenced code block
    - compiles        : the extracted code runs without SyntaxError
    - matches_keyword : response contains an expected keyword

This is not a rigorous benchmark — it's a quick sanity check to
confirm the model is producing useful output and to track whether
fine-tuning makes things measurably better.

Results are saved to datasets/eval_results/<timestamp>.json

Usage:
    python -m pipeline.eval
    python -m pipeline.eval --model qwen2.5-coder:1.5b
    python -m pipeline.eval --model qwen2.5-coder:1.5b --save
"""

import ast
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# reach into the ai package
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "ai"))
from vinrei.ollama_client import DEFAULT_MODEL, complete

RESULTS_DIR = Path(__file__).parent.parent / "datasets" / "eval_results"

# ---------------------------------------------------------------------------
# 20-task eval suite
# ---------------------------------------------------------------------------

TASKS = [
    # --- codegen ---
    {
        "id": "cg_01",
        "category": "codegen",
        "prompt": "Write a Python function that returns the nth Fibonacci number using recursion.",
        "keywords": ["def", "fibonacci", "return"],
    },
    {
        "id": "cg_02",
        "category": "codegen",
        "prompt": "Write a Python function that checks if a string is a palindrome.",
        "keywords": ["def", "return"],
    },
    {
        "id": "cg_03",
        "category": "codegen",
        "prompt": "Write a Python function that flattens a nested list.",
        "keywords": ["def", "return"],
    },
    {
        "id": "cg_04",
        "category": "codegen",
        "prompt": "Write a Python class for a stack with push, pop, and peek methods.",
        "keywords": ["class", "def", "push", "pop"],
    },
    {
        "id": "cg_05",
        "category": "codegen",
        "prompt": "Write a Python function that performs binary search on a sorted list.",
        "keywords": ["def", "return", "mid"],
    },
    # --- debugging ---
    {
        "id": "db_01",
        "category": "debug",
        "prompt": (
            "This Python code has a bug. Find and fix it:\n"
            "```python\n"
            "def divide(a, b):\n"
            "    return a / b\n"
            "\n"
            "print(divide(10, 0))\n"
            "```"
        ),
        "keywords": ["ZeroDivisionError", "zero", "try", "except", "if"],
    },
    {
        "id": "db_02",
        "category": "debug",
        "prompt": (
            "This Python code has a bug. Find and fix it:\n"
            "```python\n"
            "def get_first(lst):\n"
            "    return lst[0]\n"
            "\n"
            "print(get_first([]))\n"
            "```"
        ),
        "keywords": ["IndexError", "empty", "if", "len"],
    },
    {
        "id": "db_03",
        "category": "debug",
        "prompt": (
            "Why does this code print unexpected results?\n"
            "```python\n"
            "def add_to_list(item, lst=[]):\n"
            "    lst.append(item)\n"
            "    return lst\n"
            "\n"
            "print(add_to_list(1))\n"
            "print(add_to_list(2))\n"
            "```"
        ),
        "keywords": ["mutable", "default", "argument", "None"],
    },
    # --- explanation ---
    {
        "id": "ex_01",
        "category": "explain",
        "prompt": "Explain what a Python decorator is and show a simple example.",
        "keywords": ["decorator", "function", "wrapper"],
    },
    {
        "id": "ex_02",
        "category": "explain",
        "prompt": "Explain the difference between a list and a tuple in Python.",
        "keywords": ["mutable", "immutable"],
    },
    {
        "id": "ex_03",
        "category": "explain",
        "prompt": "Explain how Python's GIL affects multithreading.",
        "keywords": ["GIL", "thread", "lock"],
    },
    # --- refactoring ---
    {
        "id": "rf_01",
        "category": "refactor",
        "prompt": (
            "Refactor this code to use a list comprehension:\n"
            "```python\n"
            "result = []\n"
            "for i in range(10):\n"
            "    if i % 2 == 0:\n"
            "        result.append(i * i)\n"
            "```"
        ),
        "keywords": ["for", "if", "in"],
    },
    {
        "id": "rf_02",
        "category": "refactor",
        "prompt": (
            "Add type hints to this function:\n"
            "```python\n"
            "def greet(name, times):\n"
            "    return name * times\n"
            "```"
        ),
        "keywords": ["str", "int", "->"],
    },
    {
        "id": "rf_03",
        "category": "refactor",
        "prompt": (
            "Rewrite this using a context manager:\n"
            "```python\n"
            "f = open('data.txt')\n"
            "data = f.read()\n"
            "f.close()\n"
            "```"
        ),
        "keywords": ["with", "open", "as"],
    },
    # --- completion ---
    {
        "id": "cp_01",
        "category": "completion",
        "prompt": (
            "Complete this Python function:\n"
            "```python\n"
            "def count_vowels(s: str) -> int:\n"
            "    # count the number of vowels in s\n"
            "```"
        ),
        "keywords": ["def", "return", "vowel"],
    },
    {
        "id": "cp_02",
        "category": "completion",
        "prompt": (
            "Complete this Python function:\n"
            "```python\n"
            "def is_prime(n: int) -> bool:\n"
            "    # return True if n is prime\n"
            "```"
        ),
        "keywords": ["def", "return", "True", "False"],
    },
    # --- agentic ---
    {
        "id": "ag_01",
        "category": "agentic",
        "prompt": (
            "You are a coding assistant. The user wants to add logging to this function.\n"
            "Show the updated code:\n"
            "```python\n"
            "def process(data):\n"
            "    result = data * 2\n"
            "    return result\n"
            "```"
        ),
        "keywords": ["import logging", "logging.", "logger"],
    },
    {
        "id": "ag_02",
        "category": "agentic",
        "prompt": (
            "The user asked you to write tests for this function:\n"
            "```python\n"
            "def add(a: int, b: int) -> int:\n"
            "    return a + b\n"
            "```\n"
            "Write pytest tests."
        ),
        "keywords": ["def test_", "assert", "pytest"],
    },
    {
        "id": "ag_03",
        "category": "agentic",
        "prompt": "List the steps you would take to debug a Python script that raises a KeyError.",
        "keywords": ["KeyError", "key", "dict"],
    },
    {
        "id": "ag_04",
        "category": "agentic",
        "prompt": (
            "Review this code and suggest improvements:\n"
            "```python\n"
            "def get_user(id):\n"
            "    db = connect_db()\n"
            "    user = db.query('SELECT * FROM users WHERE id=' + str(id))\n"
            "    return user\n"
            "```"
        ),
        "keywords": ["injection", "parameterized", "SQL", "f-string"],
    },
]


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _extract_code(response: str) -> str:
    """Extract the first fenced code block from the response."""
    match = re.search(r"```(?:\w+)?\n(.*?)```", response, re.DOTALL)
    return match.group(1).strip() if match else ""


def score(task: dict, response: str) -> dict:
    """
    Score a single response.

    Checks:
    - contains_code : response has a fenced code block
    - compiles      : extracted code parses without SyntaxError
    - has_keywords  : response contains all expected keywords (case-insensitive)

    Returns a dict with individual check results and a total 0–3 score.
    """
    code = _extract_code(response)
    response_lower = response.lower()

    contains_code = bool(code)

    compiles = False
    if code:
        try:
            ast.parse(code)
            compiles = True
        except SyntaxError:
            pass

    keywords = task.get("keywords", [])
    has_keywords = all(kw.lower() in response_lower for kw in keywords)

    total = sum([contains_code, compiles, has_keywords])

    return {
        "contains_code": contains_code,
        "compiles": compiles,
        "has_keywords": has_keywords,
        "score": total,
        "max_score": 3,
    }


# ---------------------------------------------------------------------------
# Running the eval
# ---------------------------------------------------------------------------

def run_eval(model: str = DEFAULT_MODEL) -> dict:
    """
    Run all 20 tasks against the model and return results.

    Args:
        model: Ollama model tag.

    Returns:
        Dict with per-task results and aggregate stats.
    """
    results = []
    total_score = 0
    total_max = 0

    print(f"[eval] Running {len(TASKS)} tasks on {model}")
    print("─" * 50)

    for task in TASKS:
        print(f"  [{task['id']}] {task['category']}: {task['prompt'][:60]}...")
        start = time.perf_counter()

        try:
            response = complete(task["prompt"], model=model, temperature=0.1, max_tokens=300)
        except Exception as e:
            response = ""
            print(f"    [error] {e}")

        elapsed = time.perf_counter() - start
        result = score(task, response)

        print(
            f"    score: {result['score']}/3 "
            f"(code={result['contains_code']} "
            f"compiles={result['compiles']} "
            f"keywords={result['has_keywords']}) "
            f"[{elapsed:.1f}s]"
        )

        results.append({
            "id": task["id"],
            "category": task["category"],
            "prompt": task["prompt"],
            "response": response,
            **result,
            "elapsed_sec": round(elapsed, 2),
        })

        total_score += result["score"]
        total_max += result["max_score"]

    pct = (total_score / total_max * 100) if total_max else 0
    print("─" * 50)
    print(f"[eval] Total: {total_score}/{total_max}  ({pct:.1f}%)")

    # breakdown by category
    categories: dict[str, list] = {}
    for r in results:
        categories.setdefault(r["category"], []).append(r["score"])
    for cat, scores in sorted(categories.items()):
        cat_pct = sum(scores) / (len(scores) * 3) * 100
        print(f"  {cat}: {sum(scores)}/{len(scores)*3}  ({cat_pct:.1f}%)")

    return {
        "model": model,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_score": total_score,
        "total_max": total_max,
        "percent": round(pct, 1),
        "tasks": results,
    }


def save_results(results: dict) -> Path:
    """Save eval results to datasets/eval_results/<timestamp>.json"""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"{ts}_{results['model'].replace(':', '_')}.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"[eval] Results saved to {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run smoke eval on a model")
    parser.add_argument("--model", "-m", default=DEFAULT_MODEL)
    parser.add_argument("--save", action="store_true", help="Save results to disk")
    args = parser.parse_args()

    results = run_eval(model=args.model)

    if args.save:
        save_results(results)
