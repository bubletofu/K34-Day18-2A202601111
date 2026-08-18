from __future__ import annotations

"""
Module 1: Advanced Chunking Strategies
=======================================
Implement semantic, hierarchical, và structure-aware chunking.
So sánh với basic chunking (baseline) để thấy improvement.

Test: pytest tests/test_m1.py
"""

import glob
import hashlib
import math
import os
import re
import sys
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (DATA_DIR, HIERARCHICAL_PARENT_SIZE, HIERARCHICAL_CHILD_SIZE,
                    SEMANTIC_THRESHOLD)


@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)
    parent_id: str | None = None


def _extract_pdf_text(path: str) -> str:
    """Extract text layer từ PDF. Trả về "" nếu PDF là scan ảnh (không có text)."""
    from pypdf import PdfReader

    reader = PdfReader(path)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages).strip()


def load_documents(data_dir: str = DATA_DIR) -> list[dict]:
    """Load tất cả markdown và PDF (có text layer) từ data/. (Đã implement sẵn)

    - .md: đọc trực tiếp.
    - .pdf: trích text layer bằng pypdf. PDF scan ảnh (không có text) bị bỏ qua
      kèm cảnh báo — RAG text-based không xử lý được scan nếu chưa OCR.
    """
    docs = []
    for fp in sorted(glob.glob(os.path.join(data_dir, "*.md"))):
        with open(fp, encoding="utf-8") as f:
            docs.append({"text": f.read(), "metadata": {"source": os.path.basename(fp)}})

    for fp in sorted(glob.glob(os.path.join(data_dir, "*.pdf"))):
        try:
            text = _extract_pdf_text(fp)
        except ModuleNotFoundError as exc:
            if exc.name != "pypdf":
                raise
            print("  Bỏ qua các file PDF: chưa cài pypdf. "
                  "Chạy: python -m pip install pypdf")
            break
        if text:
            docs.append({"text": text, "metadata": {"source": os.path.basename(fp)}})
        else:
            print(f"  ⚠️  Bỏ qua {os.path.basename(fp)}: PDF scan ảnh, không có text layer (cần OCR).")

    return docs


# ─── Baseline: Basic Chunking (để so sánh) ──────────────


def chunk_basic(text: str, chunk_size: int = 500, metadata: dict | None = None) -> list[Chunk]:
    """
    Basic chunking: split theo paragraph (\\n\\n).
    Đây là baseline — KHÔNG phải mục tiêu của module này.
    (Đã implement sẵn)
    """
    metadata = metadata or {}
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current = ""
    for i, para in enumerate(paragraphs):
        if len(current) + len(para) > chunk_size and current:
            chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
            current = ""
        current += para + "\n\n"
    if current.strip():
        chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
    return chunks


# ─── Strategy 1: Semantic Chunking ───────────────────────


def chunk_semantic(text: str, threshold: float = SEMANTIC_THRESHOLD,
                   metadata: dict | None = None) -> list[Chunk]:
    """
    Split text by sentence similarity — nhóm câu cùng chủ đề.
    Tốt hơn basic vì không cắt giữa ý.
    """
    metadata = metadata or {}
    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+|\n{2,}", text)
        if sentence.strip()
    ]
    if not sentences:
        return []

    effective_threshold = max(0.0, min(float(threshold), 1.0))
    try:
        from sentence_transformers import SentenceTransformer

        embeddings = SentenceTransformer("all-MiniLM-L6-v2").encode(sentences)
    except Exception as exc:
        # Character n-grams work reasonably well for Vietnamese and keep the
        # chunker deterministic when the embedding model is not installed.
        print(f"  Semantic chunking using lexical fallback: {exc}")
        embeddings = [_hashed_text_vector(sentence) for sentence in sentences]
        effective_threshold = min(effective_threshold, 0.18)

    groups: list[list[str]] = [[sentences[0]]]
    for index in range(1, len(sentences)):
        if _cosine_similarity(embeddings[index - 1], embeddings[index]) < effective_threshold:
            groups.append([])
        groups[-1].append(sentences[index])

    return [
        Chunk(
            text="\n\n".join(group),
            metadata={**metadata, "strategy": "semantic", "chunk_index": index},
        )
        for index, group in enumerate(groups)
    ]


def _hashed_text_vector(text: str, dimensions: int = 256) -> list[float]:
    normalized = " ".join(re.findall(r"[^\W_]+", text.casefold(), re.UNICODE))
    features = re.findall(r"[^\W_]+", normalized, re.UNICODE)
    compact = normalized.replace(" ", "_")
    features.extend(compact[i:i + 3] for i in range(max(0, len(compact) - 2)))
    vector = [0.0] * dimensions
    for feature in features:
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        vector[int.from_bytes(digest, "big") % dimensions] += 1.0
    return vector


def _cosine_similarity(left, right) -> float:
    numerator = sum(float(a) * float(b) for a, b in zip(left, right))
    left_norm = math.sqrt(sum(float(value) ** 2 for value in left))
    right_norm = math.sqrt(sum(float(value) ** 2 for value in right))
    return numerator / (left_norm * right_norm + 1e-9)


# ─── Strategy 2: Hierarchical Chunking ──────────────────


def chunk_hierarchical(text: str, parent_size: int = HIERARCHICAL_PARENT_SIZE,
                       child_size: int = HIERARCHICAL_CHILD_SIZE,
                       metadata: dict | None = None) -> tuple[list[Chunk], list[Chunk]]:
    """
    Parent-child hierarchy: retrieve child (precision) → return parent (context).
    Đây là default recommendation cho production RAG.

    Returns:
        (parents, children) — mỗi child có parent_id link đến parent.
    """
    if parent_size <= 0 or child_size <= 0:
        raise ValueError("parent_size and child_size must be positive")

    metadata = metadata or {}
    parent_texts = _split_to_size(text, parent_size)
    parents: list[Chunk] = []
    children: list[Chunk] = []
    source = str(metadata.get("source", "document"))
    source_key = re.sub(r"[^a-zA-Z0-9_.-]+", "_", source)

    for parent_index, parent_text in enumerate(parent_texts):
        parent_id = f"{source_key}:parent_{parent_index}"
        parent = Chunk(
            text=parent_text,
            metadata={
                **metadata,
                "chunk_type": "parent",
                "parent_id": parent_id,
                "chunk_index": parent_index,
            },
        )
        parents.append(parent)
        for child_index, child_text in enumerate(_split_to_size(parent_text, child_size)):
            children.append(Chunk(
                text=child_text,
                metadata={
                    **metadata,
                    "chunk_type": "child",
                    "chunk_index": child_index,
                },
                parent_id=parent_id,
            ))

    return parents, children


def _split_to_size(text: str, max_size: int) -> list[str]:
    """Pack paragraphs without dropping text, splitting oversized units by words."""
    paragraphs = [part.strip() for part in re.split(r"\n{2,}", text) if part.strip()]
    units: list[str] = []
    for paragraph in paragraphs:
        if len(paragraph) <= max_size:
            units.append(paragraph)
            continue
        current = ""
        for word in paragraph.split():
            if len(word) > max_size:
                if current:
                    units.append(current)
                    current = ""
                units.extend(word[i:i + max_size] for i in range(0, len(word), max_size))
            elif current and len(current) + len(word) + 1 > max_size:
                units.append(current)
                current = word
            else:
                current = f"{current} {word}".strip()
        if current:
            units.append(current)

    chunks: list[str] = []
    current = ""
    for unit in units:
        separator = "\n\n" if current else ""
        if current and len(current) + len(separator) + len(unit) > max_size:
            chunks.append(current)
            current = unit
        else:
            current = f"{current}{separator}{unit}"
    if current:
        chunks.append(current)
    return chunks


# ─── Strategy 3: Structure-Aware Chunking ────────────────


def chunk_structure_aware(text: str, metadata: dict | None = None) -> list[Chunk]:
    """
    Parse markdown headers → chunk theo logical structure.
    Giữ nguyên tables, code blocks, lists — không cắt giữa chừng.
    """
    metadata = metadata or {}
    header_pattern = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
    chunks: list[Chunk] = []
    current_header = ""
    current_lines: list[str] = []

    def flush() -> None:
        body = "\n".join(current_lines).strip()
        combined = "\n\n".join(part for part in (current_header, body) if part).strip()
        if not combined:
            return
        section = current_header.lstrip("# ").strip() or "Preamble"
        chunks.append(Chunk(
            text=combined,
            metadata={
                **metadata,
                "section": section,
                "strategy": "structure",
                "chunk_index": len(chunks),
            },
        ))

    for line in text.splitlines():
        if header_pattern.match(line):
            flush()
            current_header = line.strip()
            current_lines = []
        else:
            current_lines.append(line)
    flush()
    return chunks


# ─── A/B Test: Compare All Strategies ────────────────────


def compare_strategies(documents: list[dict]) -> dict:
    """
    Run all strategies on documents and compare.
    (Đã implement sẵn — sẽ hoạt động khi bạn implement 3 strategies ở trên)
    """
    def _stats(chunk_list):
        lengths = [len(c.text) for c in chunk_list]
        if not lengths:
            return {"count": 0, "avg_len": 0, "min_len": 0, "max_len": 0}
        return {
            "count": len(lengths),
            "avg_len": round(sum(lengths) / len(lengths)),
            "min_len": min(lengths),
            "max_len": max(lengths),
        }

    all_text = "\n\n".join(d["text"] for d in documents)
    meta = {"source": "all"}

    basic = chunk_basic(all_text, metadata=meta)
    semantic = chunk_semantic(all_text, metadata=meta)
    parents, children = chunk_hierarchical(all_text, metadata=meta)
    structure = chunk_structure_aware(all_text, metadata=meta)

    results = {
        "basic": _stats(basic),
        "semantic": _stats(semantic),
        "hierarchical": {**_stats(children), "parents": len(parents)},
        "structure": _stats(structure),
    }

    print(f"{'Strategy':<15} {'Chunks':>7} {'Avg':>5} {'Min':>5} {'Max':>5}")
    for name, s in results.items():
        print(f"{name:<15} {s['count']:>7} {s['avg_len']:>5} {s['min_len']:>5} {s['max_len']:>5}")

    return results


if __name__ == "__main__":
    docs = load_documents()
    print(f"Loaded {len(docs)} documents")
    results = compare_strategies(docs)
    for name, stats in results.items():
        print(f"  {name}: {stats}")
