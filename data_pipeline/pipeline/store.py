"""
store.py — Versioned Parquet storage with train/val/test splits.

Takes all .jsonl files from datasets/ (normalized sessions + curated
open datasets), merges them, deduplicates, splits into train/val/test,
and saves as versioned Parquet files.

Output structure:
    datasets/
        v1/
            train.parquet
            val.parquet
            test.parquet
            manifest.json    ← hash, counts, sources, timestamp

Usage:
    python -m pipeline.store
    python -m pipeline.store --version v2 --train 0.85 --val 0.1 --test 0.05
"""

import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

DATASETS_DIR = Path(__file__).parent.parent / "datasets"
DEFAULT_VERSION = "v1"
DEFAULT_SPLITS = {"train": 0.85, "val": 0.10, "test": 0.05}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> list[dict]:
    """Load all records from a .jsonl file."""
    records = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_all(datasets_dir: Path = DATASETS_DIR) -> list[dict]:
    """
    Load every .jsonl file in datasets_dir (non-recursive, top level only).

    Skips files inside version subdirectories.
    """
    all_records = []
    for path in sorted(datasets_dir.glob("*.jsonl")):
        records = load_jsonl(path)
        print(f"  {path.name}: {len(records)} records")
        all_records.extend(records)
    return all_records


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def deduplicate(records: list[dict]) -> list[dict]:
    """Remove duplicate records by prompt hash."""
    seen = set()
    out = []
    for r in records:
        h = hashlib.md5(r.get("prompt", "").encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            out.append(r)
    dupes = len(records) - len(out)
    if dupes:
        print(f"  Removed {dupes} duplicates")
    return out


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------

def split(
    records: list[dict],
    ratios: dict = DEFAULT_SPLITS,
    seed: int = 42,
) -> dict[str, list[dict]]:
    """
    Shuffle and split records into train/val/test.

    Args:
        records: Full list of samples.
        ratios: Dict with keys train/val/test summing to 1.0.
        seed: Random seed for reproducibility.

    Returns:
        Dict with keys 'train', 'val', 'test'.
    """
    assert abs(sum(ratios.values()) - 1.0) < 1e-6, "Ratios must sum to 1.0"

    random.seed(seed)
    shuffled = records.copy()
    random.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * ratios["train"])
    n_val = int(n * ratios["val"])

    return {
        "train": shuffled[:n_train],
        "val": shuffled[n_train:n_train + n_val],
        "test": shuffled[n_train + n_val:],
    }


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------

def _hash_records(records: list[dict]) -> str:
    """Compute a stable hash of the full dataset for versioning."""
    content = json.dumps(records, sort_keys=True).encode()
    return hashlib.sha256(content).hexdigest()[:16]


def save(
    splits: dict[str, list[dict]],
    datasets_dir: Path = DATASETS_DIR,
    version: str = DEFAULT_VERSION,
) -> Path:
    """
    Save splits as Parquet files under datasets/<version>/.

    Also writes a manifest.json with counts, hash, and source breakdown.

    Args:
        splits: Dict with keys train/val/test.
        datasets_dir: Root datasets directory.
        version: Version string (e.g. 'v1', 'v2').

    Returns:
        Path to the version directory.
    """
    version_dir = datasets_dir / version
    version_dir.mkdir(parents=True, exist_ok=True)

    all_records = splits["train"] + splits["val"] + splits["test"]

    for split_name, records in splits.items():
        if not records:
            continue
        df = pd.DataFrame(records)
        out_path = version_dir / f"{split_name}.parquet"
        df.to_parquet(out_path, index=False)
        print(f"  {split_name}: {len(records)} samples → {out_path.name}")

    # source breakdown
    sources: dict[str, int] = {}
    for r in all_records:
        src = r.get("source", "unknown")
        sources[src] = sources.get(src, 0) + 1

    manifest = {
        "version": version,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "hash": _hash_records(all_records),
        "total": len(all_records),
        "splits": {k: len(v) for k, v in splits.items()},
        "sources": sources,
    }

    manifest_path = version_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"  manifest → {manifest_path}")

    return version_dir


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build versioned Parquet dataset splits")
    parser.add_argument("--version", default=DEFAULT_VERSION, help="Version tag (default: v1)")
    parser.add_argument("--train", type=float, default=0.85)
    parser.add_argument("--val", type=float, default=0.10)
    parser.add_argument("--test", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ratios = {"train": args.train, "val": args.val, "test": args.test}

    print(f"[store] Loading all datasets from {DATASETS_DIR}")
    records = load_all()
    print(f"[store] Total before dedup: {len(records)}")

    records = deduplicate(records)
    print(f"[store] Total after dedup: {len(records)}")

    splits = split(records, ratios=ratios, seed=args.seed)
    print(f"[store] Saving version {args.version} ...")
    version_dir = save(splits, version=args.version)

    print(f"[store] Done — {version_dir}")
