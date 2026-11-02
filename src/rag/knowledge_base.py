import threading
from pathlib import Path

import chromadb

# Look for policies.md: new knowledge/ location first, then legacy fallbacks
_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parents[1]
_CANDIDATES = [
    _PROJECT_ROOT / "knowledge" / "policies.md",
    _PROJECT_ROOT / "policies.md",
    _HERE.parent / "policies.md",
    _HERE.parents[1] / "policies.md",
]

_collection = None
_lock = threading.Lock()


def _find_policies_file() -> Path:
    for p in _CANDIDATES:
        if p.exists():
            return p
    raise FileNotFoundError("policies.md not found. Put it in the knowledge/ folder.")


def _split_into_chunks(text: str) -> list[str]:
    """One chunk per '## ' section."""
    chunks = []
    for part in text.split("\n## ")[1:]:
        clean = part.strip()
        if clean:
            chunks.append(clean)
    return chunks


def build_index() -> None:
    """Build the vector store once. Call it at agent start."""
    global _collection
    with _lock:
        if _collection is not None:
            return

        text = _find_policies_file().read_text(encoding="utf-8")
        chunks = _split_into_chunks(text)

        client = chromadb.Client()  # in memory
        _collection = client.create_collection("policies")
        _collection.add(
            documents=chunks,
            ids=[f"p{i}" for i in range(len(chunks))],
        )
        print(f"RAG index ready: {len(chunks)} chunks")


def search(query: str, k: int = 2) -> list[str]:
    """Search policies with chunk deduplication and bounded text size."""
    build_index()
    if _collection is None:
        return []

    count = _collection.count()
    if count == 0:
        return []

    fetch_k = min(max(1, k), count)
    result = _collection.query(query_texts=[query], n_results=fetch_k)
    docs = result.get("documents", [[]])[0] if result else []

    deduped: list[str] = []
    seen: set[str] = set()
    for d in docs:
        trimmed = d.strip()
        if trimmed and trimmed not in seen:
            seen.add(trimmed)
            # Cap individual chunk at 400 chars to avoid bloating context
            deduped.append(trimmed[:400])

    return deduped[:k]
