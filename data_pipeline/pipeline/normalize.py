"""
normalize.py — Convert raw session logs to fine-tune format.

Takes .jsonl session files from sessions/ and converts each accepted
turn into a single training sample:

    {
        "prompt": str,       # the user's input
        "completion": str,   # the model's response
        "source": str,       # where it came from (session id)
        "task": str,         # explain / edit / debug / etc.
        "model": str,        # which model produced this
        "timestamp": str,    # when it was recorded
    }

Only turns where accepted=True are kept.
Samples below MIN_CHARS are dropped as too short to be useful.

Usage:
    python -m pipeline.normalize
    python -m pipeline.normalize --sessions ../sessions --out ../datasets/normalized.jsonl
"""

import json
from pathlib import Path

SESSIONS_DIR = Path(__file__).parent.parent / "sessions"
DATASETS_DIR = Path(__file__).parent.parent / "datasets"
MIN_CHARS = 30  # drop samples shorter than this


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------

def load_session(path: Path) -> list[dict]:
    """
    Load all turns from a single .jsonl session file.

    Returns a list of turn dicts (type == 'turn' only).
    """
    entries = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entries.append(json.loads(line))
    return entries


def _get_session_meta(entries: list[dict]) -> dict:
    """Extract model and task from the session_start entry."""
    for entry in entries:
        if entry.get("type") == "session_start":
            return {
                "model": entry.get("model", ""),
                "task": entry.get("task", ""),
            }
    return {"model": "", "task": ""}


# ---------------------------------------------------------------------------
# Normalizing
# ---------------------------------------------------------------------------

def normalize_session(entries: list[dict]) -> list[dict]:
    """
    Convert a list of session entries into normalized training samples.

    Filters:
    - Only type == 'turn'
    - Only accepted == True
    - Drops samples where prompt or completion is too short

    Args:
        entries: Raw entries from load_session().

    Returns:
        List of normalized sample dicts.
    """
    meta = _get_session_meta(entries)
    samples = []

    for entry in entries:
        if entry.get("type") != "turn":
            continue
        if not entry.get("accepted", True):
            continue

        prompt = entry.get("prompt", "").strip()
        completion = entry.get("response", "").strip()

        # drop too-short samples
        if len(prompt) < MIN_CHARS or len(completion) < MIN_CHARS:
            continue

        samples.append({
            "prompt": prompt,
            "completion": completion,
            "source": entry.get("session_id", ""),
            "task": meta["task"],
            "model": meta["model"],
            "timestamp": entry.get("timestamp", ""),
            "diff": entry.get("diff", ""),
        })

    return samples


def normalize_all(
    sessions_dir: str | Path = SESSIONS_DIR,
) -> list[dict]:
    """
    Load and normalize all .jsonl files in sessions_dir.

    Returns:
        Combined list of all normalized samples.
    """
    sessions_dir = Path(sessions_dir)
    all_samples = []

    session_files = sorted(sessions_dir.glob("*.jsonl"))
    if not session_files:
        print(f"[normalize] No session files found in {sessions_dir}")
        return []

    for path in session_files:
        entries = load_session(path)
        samples = normalize_session(entries)
        print(f"  {path.name} → {len(samples)} samples")
        all_samples.extend(samples)

    return all_samples


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------

def save(samples: list[dict], out_path: str | Path) -> None:
    """Save normalized samples to a .jsonl file."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")
    print(f"[normalize] Saved {len(samples)} samples to {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Normalize session logs to fine-tune format")
    parser.add_argument("--sessions", default=str(SESSIONS_DIR), help="Path to sessions dir")
    parser.add_argument("--out", default=str(DATASETS_DIR / "normalized.jsonl"), help="Output file")
    args = parser.parse_args()

    print(f"[normalize] Reading sessions from {args.sessions}")
    samples = normalize_all(args.sessions)
    print(f"[normalize] Total samples: {len(samples)}")

    if samples:
        save(samples, args.out)
