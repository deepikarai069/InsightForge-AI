"""RAG: business glossary + schema documentation in a ChromaDB vector index with semantic search.

Embeddings: `hashing` (default, offline, deterministic char/word n-gram vectors) or `openai`.
If ChromaDB cannot start, a NumPy in-memory index with the same interface is used.
"""
from __future__ import annotations

import hashlib
import re
from typing import Any, Dict, List, Optional

import numpy as np

from app.config import Settings, get_settings


class HashingEmbedder:
    """Deterministic lexical embedder: hashed char 3-5grams + word uni/bigrams, L2-normalised."""

    def __init__(self, dim: int = 768):
        from sklearn.feature_extraction.text import HashingVectorizer
        half = dim // 2
        self.char = HashingVectorizer(n_features=half, analyzer="char_wb", ngram_range=(3, 5), alternate_sign=False, norm=None)
        self.word = HashingVectorizer(n_features=dim - half, analyzer="word", ngram_range=(1, 2), alternate_sign=False, norm=None,
                                      token_pattern=r"(?u)\b\w+\b")

    def embed(self, texts: List[str]) -> List[List[float]]:
        from scipy.sparse import hstack
        texts = [t.lower().replace("_", " ") for t in texts]
        m = hstack([self.char.transform(texts), self.word.transform(texts) * 2.0]).toarray()
        m = np.log1p(m)
        n = np.linalg.norm(m, axis=1, keepdims=True)
        return (m / np.where(n == 0, 1, n)).tolist()


class OpenAIEmbedder:
    def __init__(self):
        from langchain_openai import OpenAIEmbeddings
        self.e = OpenAIEmbeddings(model="text-embedding-3-small")

    def embed(self, texts: List[str]) -> List[List[float]]:
        return self.e.embed_documents(texts)


def chunk_markdown(text: str, max_chars: int = 900) -> List[Dict[str, str]]:
    """Split on markdown headings; oversized sections are split on paragraphs."""
    parts = re.split(r"(?m)^#{1,3}\s+", text)
    heads = re.findall(r"(?m)^#{1,3}\s+(.*)$", text)
    chunks: List[Dict[str, str]] = []
    if not heads:
        parts, heads = [""] + [text], ["Note"]
    for title, body in zip(heads, parts[1:]):
        body = body.split("\n", 1)[1] if "\n" in body and body.split("\n", 1)[0].strip() == title.strip() else body
        body = body.strip()
        if not body:
            continue
        acc = ""
        for para in re.split(r"\n\s*\n", body):
            if acc and len(acc) + len(para) > max_chars:
                chunks.append({"title": title.strip(), "text": acc.strip()})
                acc = ""
            acc += para + "\n\n"
        if acc.strip():
            chunks.append({"title": title.strip(), "text": acc.strip()})
    return chunks


class _NumpyIndex:
    def __init__(self):
        self.items: Dict[str, Dict[str, Any]] = {}

    def upsert(self, ids, embeddings, documents, metadatas):
        for i, e, d, m in zip(ids, embeddings, documents, metadatas):
            self.items[i] = {"e": np.asarray(e), "d": d, "m": m}

    def query(self, emb, k, kind=None):
        cands = [(i, v) for i, v in self.items.items() if kind is None or v["m"].get("kind") == kind]
        q = np.asarray(emb)
        scored = sorted(((float(v["e"] @ q), v) for _, v in cands), key=lambda x: -x[0])[:k]
        return [{"text": v["d"], **v["m"], "score": round(s, 4)} for s, v in scored]

    def delete_kind(self, kind):
        self.items = {i: v for i, v in self.items.items() if v["m"].get("kind") != kind}

    def count(self):
        return len(self.items)


class KnowledgeBase:
    def __init__(self, settings: Optional[Settings] = None, collection: str = "insightforge_kb"):
        self.settings = settings or get_settings()
        try:
            self.embedder = OpenAIEmbedder() if self.settings.embedding_provider == "openai" else HashingEmbedder()
        except Exception:
            self.embedder = HashingEmbedder()
        self.backend = "numpy"
        self._col = None
        self._np = _NumpyIndex()
        try:
            import chromadb
            from chromadb.config import Settings as CS
            path = self.settings.chroma_path
            client = chromadb.EphemeralClient(settings=CS(anonymized_telemetry=False, allow_reset=True)) \
                if path == ":memory:" else chromadb.PersistentClient(path=path, settings=CS(anonymized_telemetry=False))
            try:
                self._col = client.get_or_create_collection(collection, metadata={"hnsw:space": "cosine"}, embedding_function=None)
            except TypeError:
                self._col = client.get_or_create_collection(collection, metadata={"hnsw:space": "cosine"})
            self.backend = "chromadb"
        except Exception:
            self._col = None

    # -- writes ---------------------------------------------------------
    def add_document(self, title: str, text: str, kind: str = "glossary") -> int:
        chunks = chunk_markdown(text) if kind == "glossary" and "#" in text else [{"title": title, "text": text}]
        if not chunks:
            return 0
        self._upsert(kind, [{"title": c["title"] if c["title"] != "Note" else title, "text": c["text"]} for c in chunks])
        return len(chunks)

    def _upsert(self, kind: str, chunks: List[Dict[str, str]]) -> None:
        docs = [f"{c['title']}: {c['text']}" for c in chunks]
        ids = [hashlib.sha1(f"{kind}|{d}".encode()).hexdigest() for d in docs]
        metas = [{"kind": kind, "title": c["title"]} for c in chunks]
        embs = self.embedder.embed([f"{c['title']}. {c['title']}. {c['text']}" for c in chunks])  # title-weighted
        if self._col is not None:
            self._col.upsert(ids=ids, embeddings=embs, documents=docs, metadatas=metas)
        else:
            self._np.upsert(ids, embs, docs, metas)

    def replace_schema_docs(self, schema: Dict[str, List[Dict[str, Any]]]) -> None:
        self.delete_kind("schema")
        chunks = []
        for t, cols in schema.items():
            chunks.append({"title": f"table {t}", "text": f"Table {t} has columns: " + ", ".join(f"{c['name']} ({c['type']})" for c in cols)})
            for c in cols:
                ex = ", ".join(c["sample"][:3])
                chunks.append({"title": f"{t}.{c['name']}", "text": f"Column {c['name']} in table {t}, type {c['type']}, e.g. {ex}"})
        if chunks:
            self._upsert("schema", chunks)

    def delete_kind(self, kind: str) -> None:
        if self._col is not None:
            self._col.delete(where={"kind": kind})
        else:
            self._np.delete_kind(kind)

    # -- reads ----------------------------------------------------------
    def search(self, query: str, k: int = 4, kind: Optional[str] = None, min_score: float = 0.0) -> List[Dict[str, Any]]:
        q = self.embedder.embed([query])[0]
        if self._col is not None:
            if self._col.count() == 0:
                return []
            res = self._col.query(query_embeddings=[q], n_results=min(k, self._col.count()),
                                  where={"kind": kind} if kind else None)
            out = [{"text": d, **m, "score": round(1 - dist, 4)}
                   for d, m, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0])]
        else:
            out = self._np.query(q, k, kind)
        return [o for o in out if o["score"] >= min_score]

    def count(self) -> int:
        return self._col.count() if self._col is not None else self._np.count()
