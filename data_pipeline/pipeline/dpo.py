"""
dpo.py — Generate candidate responses and build DPO preference pairs.

DPO (Direct Preference Optimization) needs pairs of:
    { prompt, chosen (good response), rejected (bad response) }

Pipeline:
  1. generate() : run each prompt N times to get candidate responses
  2. rank()     : score each candidate (reuse eval.py scoring)
  3. build_pairs(): pick best as chosen, worst as rejected
  4. save()     : write preference pairs to jsonl

Usage:
    python -m pipeline.dpo generate --model vinrei:v1
    python -m pipeline.dpo pairs
"""

import json 
import random 
import sys
from pathlib import Path 

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "ai"))
from vinrei.ollama_client import DEFAULT_MODEL, complete 
from pipeline.eval import score as eval_score

DATASETS_DIR = Path(__file__).parent.parent / "datasets"
DPO_DIR = DATASETS_DIR / "dpo"
CANDIDATES_FILE = DPO_DIR / "candidates.jsonl"
PAIRS_FILE = DPO_DIR / "preference_pairs.jsonl"

N_CANDIDATES = 3 

SEED_PROMPTS = [
    "Write a Python function that returns the nth Fibonacci number using recursion.",
    "Write a Python function that checks if a string is a palindrome.",
    "Write a Python function that flattens a nested list.",
    "Write a Python class for a stack with push, pop, and peek methods.",
    "Write a Python function that performs binary search on a sorted list.",
    "This Python code has a bug. Find and fix it:\n```python\ndef divide(a, b):\n    return a / b\nprint(divide(10, 0))\n```",
    "Explain what a Python decorator is and show a simple example.",
    "Refactor this code to use a list comprehension:\n```python\nresult = []\nfor i in range(10):\n    if i % 2 == 0:\n        result.append(i * i)\n```",
    "Add type hints to this function:\n```python\ndef greet(name, times):\n    return name * times\n```",
    "Complete this Python function:\n```python\ndef is_prime(n: int) -> bool:\n    # return True if n is prime\n```",
]

#Generate candidates 

def generate(
  model: str = DEFAULT_MODEL,
  prompts: list[str] = SEED_PROMPTS,
  n: int = N_CANDIDATES,
  temperature: float = 0.7,
) -> list[dict]:
  DPO_DIR.mkdir(parents=True, exist_ok=True)
  results = []

  for i, prompt in enumerate(prompts):
    print(f"[{i+1}/{len(prompts)}] Generating {n} candidates...")
    candidates = []
    for j in range(n):
      print(f"[{i+1}/{len(prompts)}] Generating {n} candidates...")
      try:
        res = complete(prompt, model=model, temperature=temperature, max_tokens=300)
        candidates.append(res)
        print("done")
      except Exception as e: 
        print(f"error: {e}")
        candidates.append("")
    
    results.append({
      "prompt": prompt, 
      "candidates": candidates,
      "model": model,
    })
  
  with CANDIDATES_FILE.open("w") as f: 
    for r in results:
      f.write(json.dumps(r) + "\n")
  print(f"\n[dpo] Saved {len(results)} prompts × {n} candidates → {CANDIDATES_FILE}")

  return results 

#Score and rank candidates

def _score_response(prompt: str, response: str) -> int:
    dummy_task = {"keywords": []}
    result = eval_score(dummy_task, response)
    base = int(result["contains_code"]) + int(result["compiles"])
    # penalize very short responses — likely incomplete
    if len(response.strip()) < 100:
        base = max(0, base - 1)
    return base

def rank(candidates_file: Path = CANDIDATES_FILE) -> list[dict]:
  ranked = []
  with candidates_file.open() as f:
    for line in f:
      entry = json.loads(line.strip())
      prompt = entry["prompt"]
      scored = []
      for response in entry["candidates"]:
        s = _score_response(prompt, response)
        scored.append({"response": response, "score": s})
      scored.sort(key=lambda x: x['score'], reverse=True)
      ranked.append({
        "prompt": prompt, 
        "model": entry["model"],
        "ranked": scored,
      })
  return ranked 

def build_pairs(ranked: list[dict]) -> list[dict]:
  pairs = []
  skipped = 0

  for entry in ranked: 
    candidates = entry["ranked"]
    if len(candidates) < 2:
      skipped += 1 
      continue 
  
    best = candidates[0]
    worst = candidates[-1]

    if best["score"] == worst["score"]:
      skipped += 1
      continue

    pairs.append({
      "prompt": entry["prompt"],
      "chosen": best["response"],
      "rejected": worst["response"],
      "chosen_score": best["score"],
      "rejected_score": worst["score"],
      "model": entry["model"],
    })
  
  print(f"[dpo] Built {len(pairs)} pairs ({skipped} skipped - no quality difference)")
  return pairs 

def save_pairs(pairs: list[dict], out: Path = PAIRS_FILE) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")
    print(f"[dpo] Saved {len(pairs)} preference pairs → {out}")

#CLI
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="DPO candidate generation and pairing")
    parser.add_argument(
        "command",
        choices=["generate", "pairs", "all"],
        help="generate: create candidates | pairs: build preference pairs | all: both",
    )
    parser.add_argument("--model", "-m", default=DEFAULT_MODEL)
    parser.add_argument("--n", type=int, default=N_CANDIDATES, help="Candidates per prompt")
    args = parser.parse_args()

    if args.command in ("generate", "all"):
        generate(model=args.model, n=args.n)

    if args.command in ("pairs", "all"):
        if not CANDIDATES_FILE.exists():
            print("[dpo] No candidates file found. Run 'generate' first.")
            sys.exit(1)
        ranked = rank()
        pairs = build_pairs(ranked)
        save_pairs(pairs)
        print(f"[dpo] Done — {len(pairs)} preference pairs ready for DPO training")