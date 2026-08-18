from __future__ import annotations

"""
Module 5: Enrichment Pipeline
==============================
Làm giàu chunks TRƯỚC khi embed: Summarize, HyQA, Contextual Prepend, Auto Metadata.

Test: pytest tests/test_m5.py
"""

import json
import os
import re
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OPENAI_API_KEY


@dataclass
class EnrichedChunk:
    """Chunk đã được làm giàu."""
    original_text: str
    enriched_text: str
    summary: str
    hypothesis_questions: list[str]
    auto_metadata: dict
    method: str  # "contextual", "summary", "hyqa", "full"


# ─── Technique 1: Chunk Summarization ────────────────────


def summarize_chunk(text: str) -> str:
    """
    Tạo summary ngắn cho chunk.
    Embed summary thay vì (hoặc cùng với) raw chunk → giảm noise.
    """
    if not text.strip():
        return ""
    if OPENAI_API_KEY:
        try:
            return _chat(
                "Tóm tắt đoạn văn sau trong 2-3 câu ngắn gọn bằng tiếng Việt.",
                text,
                max_tokens=150,
            )
        except Exception as exc:
            print(f"  OpenAI summarize failed: {exc}")

    return _summarize_fallback(text)


# ─── Technique 2: Hypothesis Question-Answer (HyQA) ─────


def generate_hypothesis_questions(text: str, n_questions: int = 3) -> list[str]:
    """
    Generate câu hỏi mà chunk có thể trả lời.
    Index cả questions lẫn chunk → query match tốt hơn (bridge vocabulary gap).
    """
    if n_questions <= 0 or not text.strip():
        return []
    if OPENAI_API_KEY:
        try:
            content = _chat(
                f"Dựa trên đoạn văn, tạo {n_questions} câu hỏi mà đoạn văn có thể trả lời. "
                "Trả về mỗi câu hỏi trên một dòng.",
                text,
                max_tokens=200,
            )
            questions = [_clean_list_item(line) for line in content.splitlines()]
            return [question for question in questions if question][:n_questions]
        except Exception as exc:
            print(f"  OpenAI HyQA failed: {exc}")

    return _questions_fallback(text, n_questions)


# ─── Technique 3: Contextual Prepend (Anthropic style) ──


def contextual_prepend(text: str, document_title: str = "") -> str:
    """
    Prepend context giải thích chunk nằm ở đâu trong document.
    Anthropic benchmark: giảm 49% retrieval failure (alone).
    """
    if not text:
        return ""
    if OPENAI_API_KEY:
        try:
            context = _chat(
                "Viết một câu ngắn mô tả đoạn văn nằm ở đâu trong tài liệu và nói về chủ đề gì.",
                f"Tài liệu: {document_title}\n\nĐoạn văn:\n{text}",
                max_tokens=80,
            )
            return f"{context}\n\n{text}" if context else text
        except Exception as exc:
            print(f"  OpenAI contextual failed: {exc}")

    source = document_title or "tài liệu nguồn"
    return f"Trích từ tài liệu {source}.\n\n{text}"


# ─── Technique 4: Auto Metadata Extraction ──────────────


def extract_metadata(text: str) -> dict:
    """
    LLM extract metadata tự động: topic, entities, date_range, category.
    """
    if OPENAI_API_KEY:
        try:
            content = _chat(
                'Trích xuất metadata và chỉ trả JSON: {"topic":"...","entities":[],"category":"policy|hr|it|finance","language":"vi|en"}.',
                text,
                max_tokens=150,
                json_mode=True,
            )
            parsed = _parse_json(content)
            if isinstance(parsed, dict):
                return parsed
        except Exception as exc:
            print(f"  OpenAI metadata failed: {exc}")
    return _fallback_metadata(text)


# ─── Combined Single-Call Mode ───────────────────────────


def _enrich_single_call(text: str, source: str) -> dict:
    """Single LLM call to get summary + questions + context + metadata.

    ⚠️ Cost optimization: 1 API call thay vì 4 calls riêng lẻ.
    """
    if OPENAI_API_KEY:
        try:
            prompt = (
                "Phân tích đoạn văn và chỉ trả JSON gồm summary, questions, "
                "context và metadata. Metadata phải có topic, entities, category "
                "(policy|hr|it|finance) và language (vi|en)."
            )
            content = _chat(
                prompt,
                f"Tài liệu: {source}\n\nĐoạn văn:\n{text}",
                max_tokens=400,
                json_mode=True,
            )
            result = _parse_json(content)
            if isinstance(result, dict):
                return result
        except Exception as exc:
            print(f"  Enrichment API failed: {exc}")

    title = source or "tài liệu nguồn"
    return {
        "summary": _summarize_fallback(text),
        "questions": _questions_fallback(text, 3),
        "context": f"Đoạn trích từ {title} cung cấp thông tin chính sách liên quan.",
        "metadata": _fallback_metadata(text),
    }


def _chat(system_prompt: str, user_prompt: str, max_tokens: int,
          json_mode: bool = False) -> str:
    from openai import OpenAI
    from config import OPENAI_BASE_URL, LLM_MODEL

    extra = {"response_format": {"type": "json_object"}} if json_mode else {}
    kwargs = {"api_key": OPENAI_API_KEY}
    if OPENAI_BASE_URL:
        kwargs["base_url"] = OPENAI_BASE_URL
    response = OpenAI(**kwargs).chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=max_tokens,
        temperature=0,
        **extra,
    )
    return (response.choices[0].message.content or "").strip()


def _sentences(text: str) -> list[str]:
    flattened = re.sub(r"\s+", " ", text).strip()
    return [
        part.strip()
        for part in re.split(r"(?<=[.!?])\s+", flattened)
        if part.strip()
    ]


def _summarize_fallback(text: str) -> str:
    sentences = _sentences(text)
    return " ".join(sentences[:2]) if sentences else text.strip()


def _questions_fallback(text: str, n_questions: int) -> list[str]:
    questions = []
    for sentence in _sentences(text):
        statement = sentence.rstrip(".!? ")
        if len(statement) > 10:
            questions.append(f"Thông tin nào được quy định về {statement}?")
    return questions[:max(0, n_questions)]


def _clean_list_item(line: str) -> str:
    return re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", line).strip()


def _parse_json(content: str):
    cleaned = content.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(
            r"^```(?:json)?\s*|\s*```$",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )
    return json.loads(cleaned)


def _fallback_metadata(text: str) -> dict:
    normalized = text.casefold()
    category_terms = {
        "it": ("mật khẩu", "vpn", "dữ liệu", "bảo mật", "hệ thống"),
        "finance": ("lương", "chi phí", "tạm ứng", "tài chính", "thưởng"),
        "hr": ("nhân viên", "nghỉ", "thử việc", "đào tạo", "hiệu suất"),
    }
    category = next(
        (name for name, terms in category_terms.items()
         if any(term in normalized for term in terms)),
        "policy",
    )
    sentences = _sentences(text)
    topic = sentences[0][:120].rstrip(".!? ") if sentences else "general"
    entities = sorted(set(re.findall(r"\b[A-ZĐ][\wÀ-ỹ-]{2,}\b", text)))[:10]
    return {"topic": topic, "entities": entities, "category": category, "language": "vi"}


# ─── Full Enrichment Pipeline ────────────────────────────


def enrich_chunks(
    chunks: list[dict],
    methods: list[str] | None = None,
) -> list[EnrichedChunk]:
    """
    Chạy enrichment pipeline trên danh sách chunks. (Đã implement sẵn — dùng functions ở trên)

    Có 2 chế độ:
    - methods cụ thể (["summary"], ["contextual"]...): gọi từng function riêng (tốt cho học/debug)
    - methods=["combined"] hoặc None: 1 API call duy nhất cho tất cả (tốt cho production)

    Args:
        chunks: List of {"text": str, "metadata": dict}
        methods: Default None → combined mode (1 call/chunk).
                 Options: "summary", "hyqa", "contextual", "metadata", "combined"
    """
    if methods is None:
        methods = ["combined"]

    allowed_methods = {"summary", "hyqa", "contextual", "metadata", "combined"}
    unknown_methods = set(methods) - allowed_methods
    if unknown_methods:
        raise ValueError(f"Unknown enrichment methods: {sorted(unknown_methods)}")

    use_combined = "combined" in methods

    enriched = []
    for i, chunk in enumerate(chunks):
        text = chunk["text"]
        source = chunk.get("metadata", {}).get("source", "")

        if use_combined:
            result = _enrich_single_call(text, source)
            summary = str(result.get("summary", "")).strip()
            raw_questions = result.get("questions", [])
            questions = (
                [str(question).strip() for question in raw_questions if str(question).strip()]
                if isinstance(raw_questions, list) else []
            )
            context_line = str(result.get("context", "")).strip()
            enriched_parts = []
            if context_line:
                enriched_parts.append(context_line)
            if summary:
                enriched_parts.append(f"Tóm tắt: {summary}")
            if questions:
                enriched_parts.append("Câu hỏi liên quan: " + " | ".join(questions))
            enriched_parts.append(text)
            enriched_text = "\n\n".join(enriched_parts)
            raw_metadata = result.get("metadata", {})
            auto_meta = raw_metadata if isinstance(raw_metadata, dict) else {}
        else:
            summary = summarize_chunk(text) if "summary" in methods else ""
            questions = generate_hypothesis_questions(text) if "hyqa" in methods else []
            enriched_text = contextual_prepend(text, source) if "contextual" in methods else text
            auto_meta = extract_metadata(text) if "metadata" in methods else {}

        enriched.append(EnrichedChunk(
            original_text=text,
            enriched_text=enriched_text,
            summary=summary,
            hypothesis_questions=questions,
            auto_metadata={**chunk.get("metadata", {}), **auto_meta},
            method="+".join(methods),
        ))

        if (i + 1) % 10 == 0 or (i + 1) == len(chunks):
            print(f"  Enriched {i + 1}/{len(chunks)} chunks...", flush=True)

    return enriched


# ─── Main ────────────────────────────────────────────────

if __name__ == "__main__":
    sample = "Nhân viên chính thức được nghỉ phép năm 12 ngày làm việc mỗi năm. Số ngày nghỉ phép tăng thêm 1 ngày cho mỗi 5 năm thâm niên công tác."

    print("=== Enrichment Pipeline Demo ===\n")
    print(f"Original: {sample}\n")

    s = summarize_chunk(sample)
    print(f"Summary: {s}\n")

    qs = generate_hypothesis_questions(sample)
    print(f"HyQA questions: {qs}\n")

    ctx = contextual_prepend(sample, "Sổ tay nhân viên VinUni 2024")
    print(f"Contextual: {ctx}\n")

    meta = extract_metadata(sample)
    print(f"Auto metadata: {meta}")
