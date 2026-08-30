"""
REPO GROUNDING: give the model context based on user's codebase

Provides:
- tree(): directory tree(respects .gitignore)
- read(): file contents with size cap
- grep(): regex search across ifles
- context(): bundle tree + relevant file snippets into a single string
"""



import os
import re 
import subprocess 
from pathlib import Path

#Max bytes to read a single file before truncating
FILE_SIZE_CAP = 8_000

#Max total chars for full context bundle
CONTEXT_CAP = 12_000


def _gitignore_patterns(root: Path) -> list[str]:
    gitignore = root / ".gitignore"
    if not gitignore.exists():
        return[]
    patterns = []
    for line in gitignore.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            patterns.append(line)
    return patterns 

def _is_ignored(path: Path, root: Path, patterns: list[str]) -> bool:
    rel = str(path.relative_to(root))
    for pattern in patterns: 
        if pattern.endswith("/"):
            if rel.startswith(pattern.rstrip("/")):
                return True 
        elif pattern.startswith("*"):
            if rel.endswith(pattern.lstrip("*")):
                return True 
        else: 
            if pattern in rel:
                return True 
    return False 

def tree(root: str | Path = '.', max_depth: int = 4) -> str: 
    root = Path(root).resolve()
    patterns = _gitignore_patterns(root)

    always_skip = {".git", "__pycache__", ".venv", "node_modules", ".mypy_cache"}

    lines: list[str] = []

    def _walk(current: Path, depth: int) -> None:
        if depth > max_depth:
            return 
        try: 
            entries = sorted(current.iterdir())
        except PermissionError:
            return 

        for entry in entries:
            if entry.name in always_skip:
                continue 
            if _is_ignored(entry, root, patterns): 
                continue

            rel = entry.relative_to(root)
            indent = "  " * (depth - 1)
            if entry.is_dir():
                lines.append(f"{indent}{rel}/")
                _walk(entry, depth + 1)
            else:
                lines.append(f"{indent}{rel}")
    _walk(root, 1)
    return "\n".join(lines)

def read(path: str | Path, cap: int = FILE_SIZE_CAP) -> str:
    path = Path(path)
    if not path.exists():
        return f"[error] File not found: {path}"
    if not path.is_file():
        return f"[error] Not a file: {path}"

    raw = path.read_bytes()
    if len(raw) > cap:
        truncated = raw[:cap].decode(errors="replace")
        return truncated + f"\n\n[truncated — file exceeds {cap} bytes]"
    return raw.decode(errors="replace")

def grep(pattern: str, root: str | Path = ".", include: str = "*.py") -> list[dict]:
    root = Path(root).resolve()
    results: list[dict] = []

    try: 
        proc = subprocess.run(
            ["rg", "--line-number", "--glob", include, pattern, str(root)],
            capture_output=True,
            text=True,
            timeout=10,
        )
        for line in proc.stdout.splitlines():
            parts = line.split(":", 2)
            if len(parts) == 3:
                results.append({
                    "file": parts[0],
                    "line": int(parts[1]),
                    "match": parts[2],           
                })
            return results 
    except FileNotFoundError:
        pass

    compiled = re.compile(pattern)
    for filepath in Path(root).rglob(include):
        try:
            for i, text in enumerate(filepath.read_text(errors="replace").splitlines(), 1):
                if compiled.search(text):
                    results.append({
                        "file": str(filepath),
                        "line": i,
                        "text": text,
                    })
        except (OSError, PermissionError):
            continue 
    return results

def context(root: str | Path = ".", files: list[str] | None = None) -> str:
    root = Path(root).resolve()
    parts: list[str] = []

    parts.append(f"## Repo tree ({root.name}/)\n```\n{tree(root)}\n```")

    if files: 
        for f in files: 
            path = Path(f) if Path(f).is_absolute() else root / f
            contents = read(path)
            parts.append(f"## {f}\n```\n{contents}\n```")
    
    result = "\n\n".join(parts)

    if len(result) > CONTEXT_CAP:
       result = result[:CONTEXT_CAP] + "\n\n[context truncated]"

    return result 