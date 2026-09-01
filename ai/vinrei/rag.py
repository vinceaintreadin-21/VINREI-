"""
RAG — Retrieval-Augmented Generation for your codebase.

Pipeline:
  1. chunk()   : split source files into overlapping text chunks
  2. index()   : embed all chunks and save to disk (vector store)
  3. retrieve() : given a query, return the top-k most relevant chunks
  4. context() : format retrieved chunks into a string ready for the prompt
"""

import json 
import os 
from pathlib import Path 

import numpy as np 
from sentence_transformers import SentenceTransformer

#CONFIGS 
EMBED_MODEL = "all-MiniLM-L6-v2"
CHUNK_LINES = 50 
CHUNK_OVERLAP = 10 
TOP_K = 5
INDEX_DIR = ".vinrei_index"

#CHUNKING 
def chunk_file(path: str | Path, chunk_lines: int = CHUNK_LINES, overlap: int = CHUNK_OVERLAP):
    path = Path(path)

    try:
        lines = path.read_text(errors="replace").splitlines()
    except OSError:
        return []
    
    chunks = []
    step = chunk_lines - overlap 
    for start in range(0, max(len(lines), 1), step):
        end = min(start + chunk_lines, len(lines))
        text = "\n".join(lines[start:end]).strip()
        if text: 
            chunks.append({
                "file": str(path),
                "start": start + 1,
                "end": end,
                "text": text 
            })
        if end == len(lines):
            break 
    
    return chunks

def chunk_repo(root: str | Path = ".", include_exts: tuple = (".py", ".md", ".toml", ".text")) -> list[dict]:
    root = Path(root).resolve() 
    always_skip = {".git", "__pycache__", ".venv", "node_modules", INDEX_DIR}
    all_chunks = []

    for filepath in root.rglob("*"):
        if any(part in always_skip for part in filepath.parts):
            continue 
        if filepath.is_file() and filepath.suffix in include_exts:
            all_chunks.extend(chunk_file(filepath))
    
    return all_chunks 

#Indexing 

def index(root: str | Path = ".", index_dir: str | Path | None = None) -> Path:
    root = Path(root).resolve()
    index_dir = Path(index_dir) if index_dir else root / INDEX_DIR 
    index_dir.mkdir(exist_ok=True)

    print(f"Chunking {root} ...")
    chunks = chunk_repo(root)
    print(f"{len(chunks)} chunks found")

    print(f"Loading embedding model({EMBED_MODEL}) ...")
    model = SentenceTransformer(EMBED_MODEL)

    texts = [c["text"] for c in chunks]
    print("Embedding chunks...")
    embedding = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

    chunks_path = index_dir / "chunks.json"
    embeddings_path = index_dir / "embeddings.npy"

    chunks_path.write_text(json.dumps(chunks, indent=2))
    np.save(str(embeddings_path), embedding.astype(np.float32))

    print(f"Index saved to {index_dir}")
    return index_dir


def _cosine_similarity(query_vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-10)
    matrix_norm = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-10)
    return matrix_norm @ query_norm 

def retrieve(
    query: str,
    root: str | Path = ".",
    index_dir: str | Path | None = None,
    top_k: int = TOP_K,
) -> list[dict]:
    root = Path(root).resolve()
    index_dir = Path(index_dir) if index_dir else root / INDEX_DIR 

    chunks_path = index_dir / "chunks.json"
    embeddings_path = index_dir / "embeddings.npy"

    if not chunks_path.exists() or not embeddings_path.exists():
        raise FileNotFoundError(
            f"No index found at {index_dir}. Run `python -m vinrei.rag index <repo>` first."
        )
    
    chunks = json.loads(chunks_path.read_text())
    embeddings = np.load(str(embeddings_path))

    model = SentenceTransformer(EMBED_MODEL)
    query_vec = model.encode([query], convert_to_numpy=True)[0]

    scores = _cosine_similarity(query_vec, embeddings)
    top_indices = np.argsort(scores)[::-1][:top_k]

    return [chunks[i] for i in top_indices]

#Format for prompt injection

def context(query: str, root: str | Path = ".", top_k: int = TOP_K) -> str:
    """
    Retrieve relevant chunks and format them as a prompt context string.

    Args:
        query: The user's question.
        root: Repo root.
        top_k: Number of chunks to include.

    Returns:
        Formatted string ready to prepend to the user prompt.
    """
    try:
        chunks = retrieve(query, root=root, top_k=top_k)
    except FileNotFoundError as e:
        return f"[RAG] {e}"

    parts = []
    for chunk in chunks:
        header = f"## {chunk['file']} (lines {chunk['start']}–{chunk['end']})"
        parts.append(f"{header}\n```\n{chunk['text']}\n```")

    return "### Relevant code from your repo\n\n" + "\n\n".join(parts)

#CLI

if __name__ == "__main__":
    import sys 

    if len(sys.argv) < 2 or sys.argv[1] != "index":
        print("Usage: python -m vinrei.rag index <repo_root>")
        sys.exit(1)

    repo_root = sys.argv[2] if len(sys.argv) > 2 else "."
    index(repo_root)


