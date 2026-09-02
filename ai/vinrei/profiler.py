"""
profiler.py — Hardware profiling for vinrei.

Measures:
  - Time to first token (prompt prefill speed)
  - Sustained decode speed (tokens/sec)
  - Memory usage during inference
  - Context window performance at different lengths

Usage:
    python -m vinrei.profiler
    python -m vinrei.profiler --model vinrei:v1
    python -m vinrei.profiler --full
"""

import json
import time
import urllib.request
import psutil
import os
from pathlib import Path

BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "vinrei:v1"

RESULTS_DIR = Path(__file__).parent.parent / "profiles"


# ---------------------------------------------------------------------------
# Low-level streaming with timing
# ---------------------------------------------------------------------------

def _stream_timed(prompt: str, model: str, system: str = "") -> dict:
    """
    Stream a completion and record detailed timing metrics.

    Returns:
        {
            time_to_first_token: float,  # seconds
            total_time: float,           # seconds
            token_count: int,            # approximate
            tokens_per_sec: float,
            response: str,
        }
    """
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "temperature": 0.1,
    }

    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{BASE_URL}/v1/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    chunks = []
    first_token_time = None
    start = time.perf_counter()

    with urllib.request.urlopen(req) as resp:
        for raw_line in resp:
            line = raw_line.decode().strip()
            if not line or not line.startswith("data:"):
                continue
            data_str = line[len("data:"):].strip()
            if data_str == "[DONE]":
                break
            chunk = json.loads(data_str)
            token = chunk["choices"][0]["delta"].get("content", "")
            if token:
                if first_token_time is None:
                    first_token_time = time.perf_counter() - start
                chunks.append(token)

    total_time = time.perf_counter() - start
    response = "".join(chunks)
    # rough token count
    token_count = len(response.split())

    return {
        "time_to_first_token": round(first_token_time or 0, 3),
        "total_time": round(total_time, 3),
        "token_count": token_count,
        "tokens_per_sec": round(token_count / total_time, 2) if total_time > 0 else 0,
        "response": response,
    }


# ---------------------------------------------------------------------------
# Memory measurement
# ---------------------------------------------------------------------------

def _memory_mb() -> float:
    """Return current process RSS memory in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


# ---------------------------------------------------------------------------
# Profile runs
# ---------------------------------------------------------------------------

def profile_ttft(model: str) -> dict:
    """
    Measure time to first token with short vs long prompts.

    Short prompt = fast prefill
    Long prompt = slow prefill (shows how context length affects prefill)
    """
    print("\n[profiler] Time to first token...")

    short_prompt = "Write a Python one-liner to reverse a string."
    long_prompt = ("Review this code and suggest improvements:\n" +
                   "```python\n" + "x = []\n" * 50 + "```")

    results = {}
    for label, prompt in [("short_prompt", short_prompt), ("long_prompt", long_prompt)]:
        print(f"  {label} ({len(prompt)} chars)...", end=" ", flush=True)
        m = _stream_timed(prompt, model)
        results[label] = {
            "prompt_chars": len(prompt),
            "time_to_first_token": m["time_to_first_token"],
            "total_time": m["total_time"],
            "tokens_per_sec": m["tokens_per_sec"],
        }
        print(f"ttft={m['time_to_first_token']}s  tps={m['tokens_per_sec']}")

    return results


def profile_decode_speed(model: str) -> dict:
    """
    Measure sustained decode speed across different response lengths.
    """
    print("\n[profiler] Decode speed...")

    prompts = [
        ("short_response", "What is 2 + 2? Answer in one word."),
        ("medium_response", "Write a Python function to check if a number is prime."),
        ("long_response", "Write a Python class implementing a binary search tree with insert, search, and delete methods."),
    ]

    results = {}
    for label, prompt in prompts:
        print(f"  {label}...", end=" ", flush=True)
        m = _stream_timed(prompt, model)
        results[label] = {
            "tokens": m["token_count"],
            "total_time": m["total_time"],
            "tokens_per_sec": m["tokens_per_sec"],
        }
        print(f"~{m['token_count']} tokens  {m['tokens_per_sec']} t/s  [{m['total_time']}s]")

    return results


def profile_memory(model: str) -> dict:
    """
    Measure Python process memory before and during inference.
    Note: this measures the client process, not Ollama's model memory.
    """
    print("\n[profiler] Memory usage...")

    before = _memory_mb()
    print(f"  Before inference: {before:.1f} MB")

    _stream_timed("Write a sorting algorithm in Python.", model)
    during = _memory_mb()
    print(f"  During inference: {during:.1f} MB")

    return {
        "before_mb": round(before, 1),
        "during_mb": round(during, 1),
        "delta_mb": round(during - before, 1),
    }


def profile_context_window(model: str) -> dict:
    """
    Test performance at different context lengths.
    Sends increasingly long prompts to see where speed degrades.
    """
    print("\n[profiler] Context window scaling...")

    base_code = "def foo(x):\n    return x * 2\n\n"
    results = {}

    for n_lines in [10, 50, 100, 200]:
        context = base_code * (n_lines // 4)
        prompt = f"Explain this code briefly:\n```python\n{context}```"
        chars = len(prompt)
        print(f"  {n_lines} code lines ({chars} chars)...", end=" ", flush=True)
        m = _stream_timed(prompt, model)
        results[f"{n_lines}_lines"] = {
            "prompt_chars": chars,
            "time_to_first_token": m["time_to_first_token"],
            "tokens_per_sec": m["tokens_per_sec"],
        }
        print(f"ttft={m['time_to_first_token']}s  tps={m['tokens_per_sec']}")

    return results


# ---------------------------------------------------------------------------
# Full profile run
# ---------------------------------------------------------------------------

def run_profile(model: str = DEFAULT_MODEL, full: bool = False) -> dict:
    """
    Run the hardware profile suite.

    Args:
        model: Ollama model tag to profile.
        full: If True, run all tests including slow context window test.

    Returns:
        Dict with all profile results.
    """
    print(f"[profiler] Profiling {model} on {_get_cpu_info()}")
    print("─" * 50)

    results = {
        "model": model,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "hardware": _get_hw_info(),
    }

    results["ttft"] = profile_ttft(model)
    results["decode_speed"] = profile_decode_speed(model)
    results["memory"] = profile_memory(model)

    if full:
        results["context_window"] = profile_context_window(model)

    # summary
    tps_values = [v["tokens_per_sec"] for v in results["decode_speed"].values()]
    avg_tps = sum(tps_values) / len(tps_values)
    ttft_short = results["ttft"]["short_prompt"]["time_to_first_token"]

    print("\n" + "─" * 50)
    print(f"[profiler] Summary for {model}:")
    print(f"  Avg tokens/sec : {avg_tps:.1f}")
    print(f"  TTFT (short)   : {ttft_short}s")
    print(f"  Memory delta   : {results['memory']['delta_mb']} MB")

    return results


def save_profile(results: dict) -> Path:
    """Save profile results to profiles/<timestamp>_<model>.json"""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    model_slug = results["model"].replace(":", "_").replace("/", "_")
    out = RESULTS_DIR / f"{ts}_{model_slug}.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"[profiler] Results saved to {out}")
    return out


def _get_cpu_info() -> str:
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if "model name" in line:
                    return line.split(":")[1].strip()
    except OSError:
        pass
    return "unknown CPU"


def _get_hw_info() -> dict:
    vm = psutil.virtual_memory()
    return {
        "cpu": _get_cpu_info(),
        "cpu_cores": psutil.cpu_count(logical=False),
        "cpu_threads": psutil.cpu_count(logical=True),
        "ram_total_gb": round(vm.total / 1024**3, 1),
        "ram_available_gb": round(vm.available / 1024**3, 1),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Hardware profiler for vinrei")
    parser.add_argument("--model", "-m", default=DEFAULT_MODEL)
    parser.add_argument("--full", action="store_true", help="Run full suite including context window test")
    parser.add_argument("--save", action="store_true", help="Save results to disk")
    args = parser.parse_args()

    results = run_profile(model=args.model, full=args.full)

    if args.save:
        save_profile(results)
