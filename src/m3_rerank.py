from __future__ import annotations

"""Module 3: Reranking — Cross-encoder top-20 → top-3 + latency benchmark."""

import math
import os
import re
import sys
import time
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RERANK_TOP_K


@dataclass
class RerankResult:
    text: str
    original_score: float
    rerank_score: float
    metadata: dict
    rank: int


class CrossEncoderReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        self.model_name = model_name
        self._model = None

    def _load_model(self):
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder

                self._model = CrossEncoder(self.model_name)
            except Exception as exc:
                print(f"  Reranker using lexical fallback: {exc}")
                self._model = _LexicalCrossEncoder()
        return self._model

    def rerank(self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K) -> list[RerankResult]:
        """Rerank documents: top-20 → top-k."""
        if not documents or top_k <= 0:
            return []
        pairs = [(query, document["text"]) for document in documents]
        scores = self._load_model().predict(pairs)
        if isinstance(scores, (int, float)):
            scores = [scores]
        elif hasattr(scores, "tolist"):
            scores = scores.tolist()
        scored = sorted(
            zip(scores, documents),
            key=lambda item: float(item[0]),
            reverse=True,
        )
        return [
            RerankResult(
                text=document["text"],
                original_score=float(document.get("score", 0.0)),
                rerank_score=float(score),
                metadata=dict(document.get("metadata", {})),
                rank=rank,
            )
            for rank, (score, document) in enumerate(scored[:top_k], start=1)
        ]


class _LexicalCrossEncoder:
    """Deterministic fallback that approximates query-document relevance."""

    @staticmethod
    def predict(pairs: list[tuple[str, str]]) -> list[float]:
        return [_lexical_score(query, document) for query, document in pairs]


def _tokens(text: str) -> list[str]:
    return re.findall(r"[^\W_]+", text.casefold(), re.UNICODE)


def _lexical_score(query: str, document: str) -> float:
    query_tokens = _tokens(query)
    document_tokens = _tokens(document)
    if not query_tokens or not document_tokens:
        return 0.0
    query_counts = {token: query_tokens.count(token) for token in set(query_tokens)}
    document_counts = {token: document_tokens.count(token) for token in set(document_tokens)}
    numerator = sum(query_counts[token] * document_counts.get(token, 0)
                    for token in query_counts)
    query_norm = math.sqrt(sum(value * value for value in query_counts.values()))
    document_norm = math.sqrt(sum(value * value for value in document_counts.values()))
    return numerator / (query_norm * document_norm + 1e-9)


class FlashrankReranker:
    """Lightweight alternative (<5ms). Optional."""
    def __init__(self):
        self._model = None

    def rerank(self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K) -> list[RerankResult]:
        if not documents or top_k <= 0:
            return []
        try:
            from flashrank import Ranker, RerankRequest

            if self._model is None:
                self._model = Ranker()
            passages = [
                {"id": index, "text": document["text"], "meta": document.get("metadata", {})}
                for index, document in enumerate(documents)
            ]
            ranked = self._model.rerank(RerankRequest(query=query, passages=passages))[:top_k]
            return [
                RerankResult(
                    text=item["text"],
                    original_score=float(documents[int(item["id"])].get("score", 0.0)),
                    rerank_score=float(item["score"]),
                    metadata=dict(item.get("meta", {})),
                    rank=rank,
                )
                for rank, item in enumerate(ranked, start=1)
            ]
        except (ImportError, OSError, RuntimeError, ValueError, KeyError):
            return CrossEncoderReranker().rerank(query, documents, top_k)


def benchmark_reranker(reranker, query: str, documents: list[dict], n_runs: int = 5) -> dict:
    """Benchmark latency over n_runs. (Đã implement sẵn)"""
    if n_runs <= 0:
        raise ValueError("n_runs must be positive")
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        reranker.rerank(query, documents)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
    return {"avg_ms": sum(times) / len(times), "min_ms": min(times), "max_ms": max(times)}


if __name__ == "__main__":
    query = "Nhân viên được nghỉ phép bao nhiêu ngày?"
    docs = [
        {"text": "Nhân viên được nghỉ 12 ngày/năm.", "score": 0.8, "metadata": {}},
        {"text": "Mật khẩu thay đổi mỗi 90 ngày.", "score": 0.7, "metadata": {}},
        {"text": "Thời gian thử việc là 60 ngày.", "score": 0.75, "metadata": {}},
    ]
    reranker = CrossEncoderReranker()
    for r in reranker.rerank(query, docs):
        print(f"[{r.rank}] {r.rerank_score:.4f} | {r.text}")
