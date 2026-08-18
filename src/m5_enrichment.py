from __future__ import annotations

"""
Module 5: Enrichment Pipeline
==============================
Làm giàu chunks TRƯỚC khi embed: Summarize, HyQA, Contextual Prepend, Auto Metadata.

Test: pytest tests/test_m5.py
"""

import os, sys
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OPENAI_API_KEY

ENRICH_MODEL = "gpt-4o-mini"


@dataclass
class EnrichedChunk:
    """Chunk đã được làm giàu."""
    original_text: str
    enriched_text: str
    summary: str
    hypothesis_questions: list[str]
    auto_metadata: dict
    method: str  # "contextual", "summary", "hyqa", "full"


_CLIENT = None


def _get_client():
    """Lazy-init + tái sử dụng OpenAI client (tránh mở connection pool mới mỗi call)."""
    global _CLIENT
    if _CLIENT is None:
        from openai import OpenAI

        _CLIENT = OpenAI()
    return _CLIENT


# ─── Technique 1: Chunk Summarization ────────────────────


def summarize_chunk(text: str) -> str:
    """
    Tạo summary ngắn cho chunk.
    Embed summary thay vì (hoặc cùng với) raw chunk → giảm noise.
    """
    if OPENAI_API_KEY:
        try:
            resp = _get_client().chat.completions.create(
                model=ENRICH_MODEL,
                messages=[
                    {"role": "system", "content": "Tóm tắt đoạn văn sau trong 2-3 câu ngắn gọn bằng tiếng Việt."},
                    {"role": "user", "content": text},
                ],
                max_tokens=150,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            print(f"  ⚠️  OpenAI summarize failed: {e}")

    # Extractive fallback (không cần API): lấy 2 câu đầu
    sentences = [s.strip() for s in text.replace("\n", " ").split(". ") if s.strip()]
    return ". ".join(sentences[:2]) + "." if sentences else text


# ─── Technique 2: Hypothesis Question-Answer (HyQA) ─────


def generate_hypothesis_questions(text: str, n_questions: int = 3) -> list[str]:
    """
    Generate câu hỏi mà chunk có thể trả lời.
    Index cả questions lẫn chunk → query match tốt hơn (bridge vocabulary gap).
    """
    if OPENAI_API_KEY:
        try:
            resp = _get_client().chat.completions.create(
                model=ENRICH_MODEL,
                messages=[
                    {"role": "system", "content": f"Dựa trên đoạn văn, tạo {n_questions} câu hỏi mà đoạn văn có thể trả lời. Trả về mỗi câu hỏi trên 1 dòng."},
                    {"role": "user", "content": text},
                ],
                max_tokens=200,
            )
            questions = resp.choices[0].message.content.strip().split("\n")
            return [q.strip().lstrip("0123456789.-) ") for q in questions if q.strip()][:n_questions]
        except Exception as e:
            print(f"  ⚠️  OpenAI HyQA failed: {e}")

    # Extractive fallback: biến câu khẳng định thành câu hỏi thô
    import re

    sentences = [s.strip() for s in re.split(r"[.!?\n]", text) if len(s.strip()) > 10]
    return [f"{s.rstrip('.')}?" for s in sentences[:n_questions]]


# ─── Technique 3: Contextual Prepend (Anthropic style) ──


def contextual_prepend(text: str, document_title: str = "") -> str:
    """
    Prepend context giải thích chunk nằm ở đâu trong document.
    Anthropic benchmark: giảm 49% retrieval failure (alone).
    """
    if OPENAI_API_KEY:
        try:
            resp = _get_client().chat.completions.create(
                model=ENRICH_MODEL,
                messages=[
                    {"role": "system", "content": "Viết 1 câu ngắn mô tả đoạn văn này nằm ở đâu trong tài liệu và nói về chủ đề gì. Chỉ trả về 1 câu."},
                    {"role": "user", "content": f"Tài liệu: {document_title}\n\nĐoạn văn:\n{text}"},
                ],
                max_tokens=80,
            )
            context = resp.choices[0].message.content.strip()
            return f"{context}\n\n{text}"
        except Exception as e:
            print(f"  ⚠️  OpenAI contextual failed: {e}")

    # Simple fallback: ít nhất gắn tên tài liệu nguồn vào chunk
    prefix = f"Trích từ {document_title}. " if document_title else ""
    return f"{prefix}{text}"


# ─── Technique 4: Auto Metadata Extraction ──────────────


def extract_metadata(text: str) -> dict:
    """
    LLM extract metadata tự động: topic, entities, date_range, category.
    """
    if OPENAI_API_KEY:
        try:
            import json as _json

            resp = _get_client().chat.completions.create(
                model=ENRICH_MODEL,
                messages=[
                    {"role": "system", "content": 'Trích xuất metadata từ đoạn văn. Trả về JSON: {"topic": "...", "entities": ["..."], "category": "policy|hr|it|finance", "language": "vi|en"}'},
                    {"role": "user", "content": text},
                ],
                max_tokens=150,
                response_format={"type": "json_object"},
            )
            return _json.loads(resp.choices[0].message.content)
        except Exception as e:
            print(f"  ⚠️  OpenAI metadata failed: {e}")

    return {"topic": "general", "entities": [], "category": "policy", "language": "vi"}


# ─── Combined Single-Call Mode ───────────────────────────

_COMBINED_SYSTEM_PROMPT = """Phân tích đoạn văn và trả về JSON:
{
  "summary": "tóm tắt 2-3 câu",
  "questions": ["câu hỏi 1", "câu hỏi 2", "câu hỏi 3"],
  "context": "1 câu mô tả đoạn văn nằm ở đâu trong tài liệu và nói về chủ đề gì",
  "metadata": {"topic": "...", "entities": ["..."], "category": "policy|hr|it|finance", "language": "vi|en"}
}
Nếu tên tài liệu chứa version (v1/v2/v2023/v2024), nêu rõ version đó trong "context".
Chỉ trả về JSON, không thêm giải thích."""


def _enrich_single_call(text: str, source: str) -> dict:
    """Single LLM call to get summary + questions + context + metadata.

    ⚠️ Cost optimization: 1 API call thay vì 4 calls riêng lẻ (giảm 75% cost + latency).
    """
    if not OPENAI_API_KEY:
        return {}
    try:
        import json as _json

        resp = _get_client().chat.completions.create(
            model=ENRICH_MODEL,
            messages=[
                {"role": "system", "content": _COMBINED_SYSTEM_PROMPT},
                {"role": "user", "content": f"Tài liệu: {source}\n\nĐoạn văn:\n{text}"},
            ],
            max_tokens=400,
            response_format={"type": "json_object"},  # ép JSON hợp lệ, tránh parse lỗi
        )
        return _json.loads(resp.choices[0].message.content)
    except Exception as e:
        print(f"  ⚠️  Enrichment API failed: {e}")
        return {}


# ─── Full Enrichment Pipeline ────────────────────────────


def _build_enriched(chunk: dict, methods: list[str], use_combined: bool) -> EnrichedChunk:
    """Enrich 1 chunk (dùng chung cho cả chạy tuần tự lẫn song song)."""
    text = chunk["text"]
    source = chunk.get("metadata", {}).get("source", "")

    if use_combined:
        result = _enrich_single_call(text, source)
        summary = result.get("summary", "")
        questions = result.get("questions", []) or []
        context_line = result.get("context", "")
        auto_meta = result.get("metadata", {}) or {}
        # Contextual prepend + HyQA cùng nằm trong text được embed:
        # câu context giúp chunk tự mô tả, câu hỏi giả định bắc cầu từ vựng query↔doc.
        parts = [p for p in [context_line, text] if p]
        if questions:
            parts.append("Câu hỏi liên quan: " + " ".join(str(q) for q in questions))
        enriched_text = "\n\n".join(parts) if parts else text
    else:
        summary = summarize_chunk(text) if "summary" in methods else ""
        questions = generate_hypothesis_questions(text) if "hyqa" in methods else []
        enriched_text = contextual_prepend(text, source) if "contextual" in methods else text
        auto_meta = extract_metadata(text) if "metadata" in methods else {}

    # Metadata LLM sinh ra có thể chứa list/dict → ép về string cho Qdrant payload gọn
    flat_meta = {k: (", ".join(map(str, v)) if isinstance(v, (list, tuple)) else v)
                 for k, v in auto_meta.items() if not isinstance(v, dict)}

    return EnrichedChunk(
        original_text=text,
        enriched_text=enriched_text,
        summary=summary,
        hypothesis_questions=[str(q) for q in questions],
        auto_metadata={**chunk.get("metadata", {}), **flat_meta},
        method="+".join(methods),
    )


def enrich_chunks(
    chunks: list[dict],
    methods: list[str] | None = None,
    max_workers: int = 8,
) -> list[EnrichedChunk]:
    """
    Chạy enrichment pipeline trên danh sách chunks.

    Có 2 chế độ:
    - methods cụ thể (["summary"], ["contextual"]...): gọi từng function riêng (tốt cho học/debug)
    - methods=["combined"] hoặc None: 1 API call duy nhất cho tất cả (tốt cho production)

    Args:
        chunks: List of {"text": str, "metadata": dict}
        methods: Default None → combined mode (1 call/chunk).
                 Options: "summary", "hyqa", "contextual", "metadata", "combined"
        max_workers: Số chunk enrich song song (API-bound → thread pool là đủ).
    """
    if methods is None:
        methods = ["combined"]

    use_combined = "combined" in methods

    if not chunks:
        return []

    # Không có API key → chạy tuần tự bằng fallback, khỏi tốn thread
    workers = max_workers if OPENAI_API_KEY else 1

    if workers <= 1:
        enriched = []
        for i, chunk in enumerate(chunks):
            enriched.append(_build_enriched(chunk, methods, use_combined))
            if (i + 1) % 10 == 0 or (i + 1) == len(chunks):
                print(f"  Enriched {i + 1}/{len(chunks)} chunks...", flush=True)
        return enriched

    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_build_enriched, c, methods, use_combined) for c in chunks]
        enriched = []
        for i, future in enumerate(futures):  # giữ nguyên thứ tự chunk gốc
            enriched.append(future.result())
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
    print(f"Auto metadata: {meta}\n")

    combined = _enrich_single_call(sample, "nghi_phep_nam_v2024.md")
    print(f"Combined single call: {combined}")
