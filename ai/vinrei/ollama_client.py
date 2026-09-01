"""
Thin client for the Ollama OpenAI-compatible API.

Talks directly to http://localhost:11434/v1 — no external SDKs, no token counting.
"""

import json
import time
import urllib.request
from typing import Iterator

BASE_URL = "http://localhost:11434/v1"
DEFAULT_MODEL = "qwen2.5-coder:1.5b"


def _post(endpoint: str, payload: dict) -> urllib.request.http.client.HTTPResponse:
    """Send a POST request and return the raw response."""
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{BASE_URL}{endpoint}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    return urllib.request.urlopen(req)  # noqa: S310 — localhost only


def complete(
    prompt: str,
    model: str = DEFAULT_MODEL,
    system: str | None = None,
    temperature: float = 0.2,
    max_tokens: int = 512
) -> str:
    """
    Send a single prompt and return the full response as a string.

    Args:
        prompt: The user message.
        model: Ollama model tag (e.g. 'qwen2.5-coder:1.5b').
        system: Optional system prompt.
        temperature: Sampling temperature (lower = more deterministic).

    Returns:
        The assistant's reply as plain text.
    """
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "stream": False,
        "max_tokens": max_tokens,
    }

    with _post("/chat/completions", payload) as resp:
        body = json.loads(resp.read())

    return body["choices"][0]["message"]["content"]


def stream(
    prompt: str,
    model: str = DEFAULT_MODEL,
    system: str | None = None,
    temperature: float = 0.2,
) -> Iterator[str]:
    """
    Stream a completion token by token.

    Yields each text chunk as it arrives so the caller can print
    or process incrementally.

    Args:
        prompt: The user message.
        model: Ollama model tag.
        system: Optional system prompt.
        temperature: Sampling temperature.

    Yields:
        Text chunks (may be partial words).
    """
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "stream": True,
    }

    with _post("/chat/completions", payload) as resp:
        for raw_line in resp:
            line = raw_line.decode().strip()
            if not line or not line.startswith("data:"):
                continue
            data = line[len("data:"):].strip()
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            delta = chunk["choices"][0]["delta"].get("content", "")
            if delta:
                yield delta


def smoke_test(model: str = DEFAULT_MODEL) -> None:
    """
    Stream a short completion and report tokens/sec.

    This is the Phase 0 smoke test — run it to verify the server is up
    and to get a baseline speed measurement for your hardware.
    """
    prompt = "Write a Python one-liner that reverses a string."
    print(f"Model : {model}")
    print(f"Prompt: {prompt}")
    print("─" * 50)

    tokens = 0
    start = time.perf_counter()

    for chunk in stream(prompt, model=model):
        print(chunk, end="", flush=True)
        # rough token count: split on whitespace as a proxy
        tokens += len(chunk.split())

    elapsed = time.perf_counter() - start
    print(f"\n{'─' * 50}")
    print(f"~{tokens} tokens in {elapsed:.1f}s  →  ~{tokens / elapsed:.1f} tokens/sec")


if __name__ == "__main__":
    import sys

    model = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL
    smoke_test(model)
