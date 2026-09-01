"""
FIM — Fill-in-the-Middle completion.

Uses Ollama's /api/generate endpoint with the FIM prompt format.
This powers inline code suggestions: give it code before and after
the cursor, it fills in the gap.

FIM prompt format (qwen2.5-coder / deepseek-coder / starcoder2):
    <|fim_prefix|>{prefix}<|fim_suffix|>{suffix}<|fim_middle|>

Usage:
    python -m vinrei.fim --prefix "def add(a, b):" --suffix "    return result"
"""

import json
import urllib.request

BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "qwen2.5-coder:1.5b"

# FIM special tokens — these are correct for qwen2.5-coder and deepseek-coder
FIM_PREFIX = "<|fim_prefix|>"
FIM_SUFFIX = "<|fim_suffix|>"
FIM_MIDDLE = "<|fim_middle|>"


def _fim_prompt(prefix: str, suffix: str) -> str:
    """Build the FIM prompt string from prefix and suffix."""
    return f"{FIM_PREFIX}{prefix}{FIM_SUFFIX}{suffix}{FIM_MIDDLE}"


def complete(
    prefix: str,
    suffix: str = "",
    model: str = DEFAULT_MODEL,
    max_tokens: int = 128,
    temperature: float = 0.1,
) -> str:
    """
    Request a fill-in-the-middle completion from Ollama.

    Args:
        prefix: Code before the cursor.
        suffix: Code after the cursor (can be empty).
        model: Ollama model tag.
        max_tokens: Max tokens to generate.
        temperature: Low temperature = more deterministic completions.

    Returns:
        The generated completion as a string.
    """
    prompt = _fim_prompt(prefix, suffix)

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
            "stop": ["<|fim_pad|>", "<|endoftext|>", "</s>"],
        },
    }

    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{BASE_URL}/api/generate",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(req) as resp:  # noqa: S310 — localhost only
        body = json.loads(resp.read())

    return body.get("response", "")


def stream(
    prefix: str,
    suffix: str = "",
    model: str = DEFAULT_MODEL,
    max_tokens: int = 128,
    temperature: float = 0.1,
):
    """
    Stream a FIM completion token by token.

    Yields each text chunk as it arrives.

    Args:
        prefix: Code before the cursor.
        suffix: Code after the cursor.
        model: Ollama model tag.
        max_tokens: Max tokens to generate.
        temperature: Sampling temperature.

    Yields:
        Text chunks.
    """
    prompt = _fim_prompt(prefix, suffix)

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": True,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
            "stop": ["<|fim_pad|>", "<|endoftext|>", "</s>"],
        },
    }

    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{BASE_URL}/api/generate",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(req) as resp:  # noqa: S310
        for raw_line in resp:
            line = raw_line.decode().strip()
            if not line:
                continue
            chunk = json.loads(line)
            token = chunk.get("response", "")
            if token:
                yield token
            if chunk.get("done"):
                break


# ---------------------------------------------------------------------------
# CLI — python -m vinrei.fim
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="FIM inline completion")
    parser.add_argument("--prefix", "-p", required=True, help="Code before the cursor")
    parser.add_argument("--suffix", "-s", default="", help="Code after the cursor")
    parser.add_argument("--model", "-m", default=DEFAULT_MODEL)
    parser.add_argument("--max-tokens", type=int, default=128)
    args = parser.parse_args()

    print("Completion:")
    print("─" * 40)
    for chunk in stream(args.prefix, args.suffix, model=args.model, max_tokens=args.max_tokens):
        print(chunk, end="", flush=True)
    print()
