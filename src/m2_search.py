from __future__ import annotations

"""Module 2: Hybrid Search — BM25 (Vietnamese) + Dense + RRF."""

import hashlib
import math
import os
import re
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (QDRANT_HOST, QDRANT_PORT, COLLECTION_NAME, EMBEDDING_MODEL,
                    EMBEDDING_DIM, BM25_TOP_K, DENSE_TOP_K, HYBRID_TOP_K)


@dataclass
class SearchResult:
    text: str
    score: float
    metadata: dict
    method: str  # "bm25", "dense", "hybrid"


def segment_vietnamese(text: str) -> str:
    """Segment Vietnamese text into words."""
    try:
        from underthesea import word_tokenize

        segmented = word_tokenize(text, format="text")
    except (ImportError, OSError):
        segmented = text
    # Keep compound words compatible with ordinary user queries.
    return " ".join(segmented.replace("_", " ").casefold().split())


class _SimpleBM25:
    """Small BM25 implementation used only when rank-bm25 is unavailable."""

    def __init__(self, corpus: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.corpus = corpus
        self.k1 = k1
        self.b = b
        self.avgdl = sum(map(len, corpus)) / max(len(corpus), 1)
        self.doc_freqs: dict[str, int] = {}
        for document in corpus:
            for token in set(document):
                self.doc_freqs[token] = self.doc_freqs.get(token, 0) + 1

    def get_scores(self, query: list[str]) -> list[float]:
        scores = []
        document_count = len(self.corpus)
        for document in self.corpus:
            frequencies: dict[str, int] = {}
            for token in document:
                frequencies[token] = frequencies.get(token, 0) + 1
            score = 0.0
            for token in query:
                frequency = frequencies.get(token, 0)
                if not frequency:
                    continue
                document_frequency = self.doc_freqs.get(token, 0)
                idf = math.log(1 + (document_count - document_frequency + 0.5) /
                               (document_frequency + 0.5))
                normalization = frequency + self.k1 * (
                    1 - self.b + self.b * len(document) / max(self.avgdl, 1e-9)
                )
                score += idf * frequency * (self.k1 + 1) / normalization
            scores.append(score)
        return scores


class BM25Search:
    def __init__(self):
        self.corpus_tokens = []
        self.documents = []
        self.bm25 = None

    def index(self, chunks: list[dict]) -> None:
        """Build BM25 index from chunks."""
        self.documents = list(chunks)
        self.corpus_tokens = [segment_vietnamese(chunk["text"]).split()
                              for chunk in self.documents]
        if not self.documents:
            self.bm25 = None
            return
        try:
            from rank_bm25 import BM25Okapi

            self.bm25 = BM25Okapi(self.corpus_tokens)
        except ImportError:
            self.bm25 = _SimpleBM25(self.corpus_tokens)

    def search(self, query: str, top_k: int = BM25_TOP_K) -> list[SearchResult]:
        """Search using BM25."""
        if self.bm25 is None or top_k <= 0:
            return []
        scores = self.bm25.get_scores(segment_vietnamese(query).split())
        top_indices = sorted(range(len(scores)), key=lambda index: float(scores[index]), reverse=True)
        return [
            SearchResult(
                text=self.documents[index]["text"],
                score=float(scores[index]),
                metadata=dict(self.documents[index].get("metadata", {})),
                method="bm25",
            )
            for index in top_indices[:top_k]
            if float(scores[index]) > 0
        ]


class DenseSearch:
    def __init__(self):
        self.client = None
        self._encoder = None
        self._local_collections: dict[str, tuple[list[dict], list[list[float]]]] = {}

    def _get_encoder(self):
        if self._encoder is None:
            try:
                from sentence_transformers import SentenceTransformer

                self._encoder = SentenceTransformer(EMBEDDING_MODEL)
            except Exception as exc:
                print(f"  Dense embedding using lexical fallback: {exc}")
                self._encoder = False
        return self._encoder

    def _encode(self, texts: str | list[str]) -> list[float] | list[list[float]]:
        encoder = self._get_encoder()
        if encoder:
            encoded = (
                encoder.encode(texts, show_progress_bar=True)
                if isinstance(texts, list) else encoder.encode(texts)
            )
            return encoded.tolist() if hasattr(encoded, "tolist") else encoded
        if isinstance(texts, str):
            return _hash_embedding(texts)
        return [_hash_embedding(text) for text in texts]

    def index(self, chunks: list[dict], collection: str = COLLECTION_NAME) -> None:
        """Index chunks into Qdrant."""
        documents = list(chunks)
        texts = [chunk["text"] for chunk in documents]
        vectors = self._encode(texts) if texts else []
        self._local_collections[collection] = (documents, vectors)
        if not documents:
            return

        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, PointStruct, VectorParams

            if self.client is None:
                self.client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=5)
            vector_size = len(vectors[0])
            self.client.recreate_collection(
                collection_name=collection,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )
            points = [
                PointStruct(
                    id=index,
                    vector=vector,
                    payload={**document.get("metadata", {}), "text": document["text"]},
                )
                for index, (document, vector) in enumerate(zip(documents, vectors))
            ]
            self.client.upsert(collection_name=collection, points=points)
        except Exception as exc:
            self.client = None
            print(f"  Dense search using in-memory fallback: {exc}")

    def search(self, query: str, top_k: int = DENSE_TOP_K, collection: str = COLLECTION_NAME) -> list[SearchResult]:
        """Search using dense vectors."""
        if top_k <= 0:
            return []
        query_vector = self._encode(query)
        if self.client is not None:
            try:
                response = self.client.query_points(
                    collection_name=collection,
                    query=query_vector,
                    limit=top_k,
                )
                return [
                    SearchResult(
                        text=point.payload.get("text", ""),
                        score=float(point.score),
                        metadata={key: value for key, value in point.payload.items() if key != "text"},
                        method="dense",
                    )
                    for point in response.points
                ]
            except Exception as exc:
                print(f"  Qdrant query failed, using in-memory fallback: {exc}")

        documents, vectors = self._local_collections.get(collection, ([], []))
        scored = sorted(
            ((_cosine(query_vector, vector), document)
             for document, vector in zip(documents, vectors)),
            key=lambda item: item[0],
            reverse=True,
        )
        return [
            SearchResult(
                text=document["text"],
                score=float(score),
                metadata=dict(document.get("metadata", {})),
                method="dense",
            )
            for score, document in scored[:top_k]
        ]


def reciprocal_rank_fusion(results_list: list[list[SearchResult]], k: int = 60,
                           top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
    """Merge ranked lists using RRF: score(d) = Σ 1/(k + rank)."""
    if k < 0:
        raise ValueError("k must be non-negative")
    if top_k <= 0:
        return []
    fused: dict[str, dict] = {}
    for result_list in results_list:
        seen: set[str] = set()
        for rank, result in enumerate(result_list):
            if result.text in seen:
                continue
            seen.add(result.text)
            entry = fused.setdefault(result.text, {"score": 0.0, "result": result})
            entry["score"] += 1.0 / (k + rank + 1)

    ranked = sorted(fused.values(), key=lambda entry: entry["score"], reverse=True)
    return [
        SearchResult(
            text=entry["result"].text,
            score=float(entry["score"]),
            metadata=dict(entry["result"].metadata),
            method="hybrid",
        )
        for entry in ranked[:top_k]
    ]


def _hash_embedding(text: str, dimensions: int = EMBEDDING_DIM) -> list[float]:
    vector = [0.0] * dimensions
    tokens = re.findall(r"[^\W_]+", segment_vietnamese(text), re.UNICODE)
    features = tokens + [f"{left}_{right}" for left, right in zip(tokens, tokens[1:])]
    for feature in features:
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        vector[int.from_bytes(digest, "big") % dimensions] += 1.0
    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [value / norm for value in vector]


def _cosine(left, right) -> float:
    return sum(float(a) * float(b) for a, b in zip(left, right))


class HybridSearch:
    """Combines BM25 + Dense + RRF. (Đã implement sẵn — dùng classes ở trên)"""
    def __init__(self):
        self.bm25 = BM25Search()
        self.dense = DenseSearch()

    def index(self, chunks: list[dict]) -> None:
        self.bm25.index(chunks)
        self.dense.index(chunks)

    def search(self, query: str, top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
        bm25_results = self.bm25.search(query, top_k=BM25_TOP_K)
        dense_results = self.dense.search(query, top_k=DENSE_TOP_K)
        return reciprocal_rank_fusion([bm25_results, dense_results], top_k=top_k)


if __name__ == "__main__":
    print(f"Original:  Nhân viên được nghỉ phép năm")
    print(f"Segmented: {segment_vietnamese('Nhân viên được nghỉ phép năm')}")
