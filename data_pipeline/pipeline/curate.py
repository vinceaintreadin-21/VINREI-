"""
curate.py — Download and filter open coding datasets.

Pulls from HuggingFace Hub and converts each dataset to the same
normalized format as normalize.py:

    {
        "prompt": str,
        "completion": str,
        "source": str,       # dataset name
        "task": str,
        "model": str,        # empty — human/unknown source
        "timestamp": str,    # empty
        "diff": str,         # empty
    }

Supported datasets:
    - code_alpaca     : 20k coding instruction samples (permissive license)
    - magicoder_evol  : high-quality evolved coding instructions

Filters applied:
    - Drop samples shorter than MIN_CHARS
    - Drop samples with no code (no backtick fences)
    - Deduplicate by prompt hash

Usage:
    python -m pipeline.curate --dataset code_alpaca
    python -m pipeline.curate --dataset magicoder_evol
    python -m pipeline.curate --dataset all
"""

import hashlib
import json
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

DATASETS_DIR = Path(__file__).parent.parent / "datasets"
MIN_CHARS = 50
MAX_SAMPLES = 5_000   # cap per dataset — enough for fine-tuning on small hardware


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------

def _has_code(text: str) -> bool:
    """Return True if the text contains a fenced code block."""
    return "```" in text


def _hash(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()


def _filter(samples: list[dict]) -> list[dict]:
    """
    Apply quality filters and deduplicate.

    Keeps samples that:
    - Have prompt and completion both >= MIN_CHARS
    - Contain at least one fenced code block in prompt or completion
    - Are not duplicates (by prompt hash)
    """
    seen = set()
    out = []
    for s in samples:
        prompt = s.get("prompt", "").strip()
        completion = s.get("completion", "").strip()

        if len(prompt) < MIN_CHARS or len(completion) < MIN_CHARS:
            continue
        if not _has_code(prompt) and not _has_code(completion):
            continue

        h = _hash(prompt)
        if h in seen:
            continue
        seen.add(h)
        out.append(s)
    return out


# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------

def _load_code_alpaca(max_samples: int = MAX_SAMPLES) -> list[dict]:
    """
    Load sahil2801/CodeAlpaca-20k from HuggingFace.

    Schema: instruction, input, output
    """
    print("[curate] Downloading code_alpaca ...")
    ds = load_dataset("sahil2801/CodeAlpaca-20k", split="train")
    samples = []

    for row in tqdm(ds, total=min(len(ds), max_samples), desc="code_alpaca"):
        if len(samples) >= max_samples:
            break

        instruction = row.get("instruction", "").strip()
        inp = row.get("input", "").strip()
        output = row.get("output", "").strip()

        prompt = f"{instruction}\n{inp}".strip() if inp else instruction

        samples.append({
            "prompt": prompt,
            "completion": output,
            "source": "code_alpaca",
            "task": "codegen",
            "model": "",
            "timestamp": "",
            "diff": "",
        })

    return samples


def _load_magicoder_evol(max_samples: int = MAX_SAMPLES) -> list[dict]:
    """
    Load ise-uiuc/Magicoder-Evol-Instruct-110K from HuggingFace.

    Schema: instruction, response
    """
    print("[curate] Downloading magicoder_evol ...")
    ds = load_dataset("ise-uiuc/Magicoder-Evol-Instruct-110K", split="train")
    samples = []

    for row in tqdm(ds, total=min(len(ds), max_samples), desc="magicoder_evol"):
        if len(samples) >= max_samples:
            break

        prompt = row.get("instruction", "").strip()
        completion = row.get("response", "").strip()

        samples.append({
            "prompt": prompt,
            "completion": completion,
            "source": "magicoder_evol",
            "task": "codegen",
            "model": "",
            "timestamp": "",
            "diff": "",
        })

    return samples


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save(samples: list[dict], name: str) -> Path:
    """Save curated samples to datasets/<name>.jsonl"""
    out_path = DATASETS_DIR / f"{name}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")
    print(f"[curate] Saved {len(samples)} samples → {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

LOADERS = {
    "code_alpaca": _load_code_alpaca,
    "magicoder_evol": _load_magicoder_evol,
}

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download and curate open coding datasets")
    parser.add_argument(
        "--dataset",
        choices=[*LOADERS.keys(), "all"],
        default="code_alpaca",
        help="Which dataset to pull (default: code_alpaca)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=MAX_SAMPLES,
        help=f"Max samples per dataset (default: {MAX_SAMPLES})",
    )
    args = parser.parse_args()

    targets = list(LOADERS.keys()) if args.dataset == "all" else [args.dataset]

    for name in targets:
        raw = LOADERS[name](max_samples=args.max_samples)
        filtered = _filter(raw)
        print(f"[curate] {name}: {len(raw)} raw → {len(filtered)} after filtering")
        save(filtered, name)
